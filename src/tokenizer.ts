/**
 * BERT WordPiece tokenizer matching `transformers.BertTokenizer` as configured by the gate's
 * `tokenizer_config.json` (`do_lower_case: true`, `strip_accents: null` -> stripped,
 * `tokenize_chinese_chars: true`). Verified token-for-token against the Python tokenizer in
 * `reference.fixture.json`.
 */
export const SPECIAL_TOKENS = {
  pad: "[PAD]",
  unk: "[UNK]",
  cls: "[CLS]",
  sep: "[SEP]",
  mask: "[MASK]",
} as const;

const MAX_INPUT_CHARS_PER_WORD = 100;

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class WordPieceTokenizer {
  readonly #ids = new Map<string, number>();
  readonly unkId: number;
  readonly clsId: number;
  readonly sepId: number;
  readonly #specialSplit: RegExp;

  constructor(vocab: string) {
    const lines = vocab.split("\n");
    if (lines.at(-1) === "") lines.pop();
    lines.forEach((token, id) => this.#ids.set(token, id));
    this.unkId = this.#requireSpecial(SPECIAL_TOKENS.unk);
    this.clsId = this.#requireSpecial(SPECIAL_TOKENS.cls);
    this.sepId = this.#requireSpecial(SPECIAL_TOKENS.sep);
    this.#requireSpecial(SPECIAL_TOKENS.pad);
    this.#requireSpecial(SPECIAL_TOKENS.mask);
    const escaped = Object.values(SPECIAL_TOKENS).map(escapeRegex);
    this.#specialSplit = new RegExp(`(${escaped.join("|")})`);
  }

  get vocabSize(): number {
    return this.#ids.size;
  }

  #requireSpecial(token: string): number {
    const id = this.#ids.get(token);
    if (id === undefined) throw new Error(`vocab.txt has no ${token} entry; not a BERT vocabulary`);
    return id;
  }

  /** Content token ids for `text`, without [CLS]/[SEP]. */
  encode(text: string): number[] {
    const out: number[] = [];
    for (const segment of text.split(this.#specialSplit)) {
      const special = this.#ids.get(segment);
      if (segment in SPECIAL_SET && special !== undefined) {
        out.push(special);
        continue;
      }
      for (const word of basicTokenize(segment)) out.push(...this.#wordPiece(word));
    }
    return out;
  }

  /**
   * Content token ids plus each token's `[start, end)` UTF-16 offsets into `text` (`text.slice(start, end)`),
   * matching `tokenizers` BertNormalizer + BertPreTokenizer + WordPiece with `return_offsets_mapping`
   * (which reports code points; astral characters are the only difference).
   */
  encodeWithOffsets(text: string): { ids: number[]; offsets: Array<[number, number]> } {
    const chars = Array.from(text);
    const ids: number[] = [];
    const offsets: Array<[number, number]> = [];
    let segment = 0;
    let i = 0;
    while (i < chars.length) {
      const special = chars[i] === "[" ? SPECIAL_LIST.find((t) => matchesAt(chars, i, t)) : undefined;
      if (special === undefined) {
        i += 1;
        continue;
      }
      this.#encodeSegment(chars, segment, i, ids, offsets);
      ids.push(this.#ids.get(special) as number);
      offsets.push([i, i + special.length]);
      i += special.length;
      segment = i;
    }
    this.#encodeSegment(chars, segment, chars.length, ids, offsets);
    const utf16 = [0];
    for (const ch of chars) utf16.push(utf16[utf16.length - 1] + ch.length);
    return { ids, offsets: offsets.map(([a, b]) => [utf16[a], utf16[b]]) };
  }

  #encodeSegment(chars: readonly string[], from: number, to: number, ids: number[], offsets: Array<[number, number]>): void {
    const norm: string[] = [];
    const origin: number[] = [];
    for (let k = from; k < to; k++) {
      const ch = chars[k];
      const code = ch.codePointAt(0) ?? 0;
      if (code === 0 || code === 0xfffd || (ch !== "\t" && ch !== "\n" && ch !== "\r" && /\p{C}/u.test(ch))) continue;
      if (ch === "\t" || ch === "\n" || ch === "\r" || /\p{White_Space}/u.test(ch)) {
        norm.push(" ");
        origin.push(k);
        continue;
      }
      if (isChineseChar(code)) {
        norm.push(" ", ch, " ");
        origin.push(k, k, k);
        continue;
      }
      for (const d of ch.normalize("NFD")) {
        if (/\p{Mn}/u.test(d)) continue;
        for (const l of d.toLowerCase()) {
          norm.push(l);
          origin.push(k);
        }
      }
    }
    const emit = (a: number, b: number) => {
      if (b <= a) return;
      const pieces = this.#wordPieceSpans(norm.slice(a, b));
      if (pieces === null) {
        ids.push(this.unkId);
        offsets.push([origin[a], origin[b - 1] + 1]);
        return;
      }
      for (const [id, pa, pb] of pieces) {
        ids.push(id);
        offsets.push([origin[a + pa], origin[a + pb - 1] + 1]);
      }
    };
    let start = 0;
    for (let k = 0; k < norm.length; k++) {
      if (norm[k] === " ") {
        emit(start, k);
        start = k + 1;
      } else if (isPunctuation(norm[k])) {
        emit(start, k);
        emit(k, k + 1);
        start = k + 1;
      }
    }
    emit(start, norm.length);
  }

  /** WordPiece pieces as `[id, start, end)` over `chars`, or null when the word is `[UNK]`. */
  #wordPieceSpans(chars: readonly string[]): Array<[number, number, number]> | null {
    if (chars.length > MAX_INPUT_CHARS_PER_WORD) return null;
    const pieces: Array<[number, number, number]> = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found: number | undefined;
      while (start < end) {
        const id = this.#ids.get((start > 0 ? "##" : "") + chars.slice(start, end).join(""));
        if (id !== undefined) {
          found = id;
          break;
        }
        end -= 1;
      }
      if (found === undefined) return null;
      pieces.push([found, start, end]);
      start = end;
    }
    return pieces;
  }

  #wordPiece(word: string): number[] {
    const chars = Array.from(word);
    if (chars.length > MAX_INPUT_CHARS_PER_WORD) return [this.unkId];
    const pieces: number[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let found: number | undefined;
      while (start < end) {
        const candidate = (start > 0 ? "##" : "") + chars.slice(start, end).join("");
        const id = this.#ids.get(candidate);
        if (id !== undefined) {
          found = id;
          break;
        }
        end -= 1;
      }
      if (found === undefined) return [this.unkId];
      pieces.push(found);
      start = end;
    }
    return pieces;
  }
}

const SPECIAL_LIST: readonly string[] = Object.values(SPECIAL_TOKENS);

function matchesAt(chars: readonly string[], at: number, token: string): boolean {
  for (let j = 0; j < token.length; j++) if (chars[at + j] !== token[j]) return false;
  return true;
}

const SPECIAL_SET: Record<string, true> = Object.fromEntries(
  Object.values(SPECIAL_TOKENS).map((t) => [t, true]),
);

function basicTokenize(text: string): string[] {
  const cleaned = spaceChineseChars(cleanText(text));
  const words: string[] = [];
  for (const token of cleaned.split(" ")) {
    if (token === "") continue;
    const lowered = stripAccents(token.toLowerCase());
    words.push(...splitOnPunctuation(lowered));
  }
  return words;
}

/** Drops NUL, U+FFFD and control characters (Cc/Cf other than tab/newline) and maps whitespace to " ". */
function cleanText(text: string): string {
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (code === 0 || code === 0xfffd) continue;
    if (/\s/u.test(ch) || ch === "\t" || ch === "\n" || ch === "\r") {
      out += " ";
      continue;
    }
    if (/\p{Cc}|\p{Cf}/u.test(ch)) continue;
    out += ch;
  }
  return out;
}

function isChineseChar(code: number): boolean {
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x20000 && code <= 0x2a6df) ||
    (code >= 0x2a700 && code <= 0x2b73f) ||
    (code >= 0x2b740 && code <= 0x2b81f) ||
    (code >= 0x2b820 && code <= 0x2ceaf) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0x2f800 && code <= 0x2fa1f)
  );
}

function spaceChineseChars(text: string): string {
  let out = "";
  for (const ch of text) {
    out += isChineseChar(ch.codePointAt(0) ?? 0) ? ` ${ch} ` : ch;
  }
  return out;
}

function stripAccents(text: string): string {
  return text.normalize("NFD").replace(/\p{Mn}/gu, "");
}

function isPunctuation(ch: string): boolean {
  const code = ch.codePointAt(0) ?? 0;
  if (
    (code >= 33 && code <= 47) ||
    (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) ||
    (code >= 123 && code <= 126)
  ) {
    return true;
  }
  return /\p{P}/u.test(ch);
}

function splitOnPunctuation(word: string): string[] {
  const out: string[] = [];
  let current = "";
  for (const ch of word) {
    if (isPunctuation(ch)) {
      if (current !== "") out.push(current);
      out.push(ch);
      current = "";
    } else {
      current += ch;
    }
  }
  if (current !== "") out.push(current);
  return out;
}
