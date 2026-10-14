import BUILT_IN from "./allowlist-data.json";
import type { RedactedSpan } from "./spans.js";
import { canonPhone } from "./validators/allowlist.js";
import type { Detection, GateResult } from "./validators/index.js";
import { isGovEmail } from "./validators/rules.js";

/**
 * Public names, phone numbers and addresses that may pass the gate. A message the model blocks is
 * allowed when every span it returned is covered by an allowlist entry or a .gov/.mil email and no validator hard-blocks;
 * a listed phone number is demoted from hard to soft in the validators.
 */
export type Allowlist = {
  /** Full names of public people, e.g. "Alexander Hamilton". Case and accents are ignored. */
  readonly names?: readonly string[];
  /** Complete phone numbers, any format, e.g. "1-800-772-1213". */
  readonly phones?: readonly string[];
  /** Complete addresses, e.g. "1600 Pennsylvania Ave NW, Washington, DC 20500". Matches with or without the city, state and ZIP. */
  readonly addresses?: readonly string[];
};

export type AllowlistOptions = Allowlist & {
  /** Include the built-in public-figure, government phone and government address lists (default true). */
  readonly builtIn?: boolean;
};

/** The lists shipped with the package. */
export const BUILT_IN_ALLOWLIST: Required<Allowlist> = BUILT_IN;

const ABBREVIATIONS: Record<string, string> = {
  avenue: "ave", av: "ave", street: "st", boulevard: "blvd", road: "rd", drive: "dr", place: "pl", court: "ct",
  lane: "ln", parkway: "pkwy", highway: "hwy", square: "sq", terrace: "ter", circle: "cir", plaza: "plz",
  north: "n", south: "s", east: "e", west: "w", northwest: "nw", northeast: "ne", southwest: "sw", southeast: "se",
  suite: "ste", floor: "fl", room: "rm", building: "bldg", center: "ctr", centre: "ctr", mount: "mt", fort: "ft",
};
const DIRECTIONS = new Set(["n", "s", "e", "w", "nw", "ne", "sw", "se"]);
const SUFFIXES = new Set(["jr", "sr", "ii", "iii", "iv"]);
/** Capitalized words that may sit inside a name span without making it a different person. */
const TITLES = new Set([
  "dr", "mr", "mrs", "ms", "mx", "miss", "sir", "dame", "lord", "lady", "prof", "professor", "rev", "reverend",
  "saint", "st", "pope", "king", "queen", "prince", "princess", "emperor", "president", "vice", "senator", "sen",
  "representative", "rep", "congressman", "congresswoman", "governor", "gov", "mayor", "secretary", "justice",
  "judge", "chief", "general", "gen", "commissioner", "director", "the", "jr", "sr", "ii", "iii", "iv",
]);

/** First-person possessives in the seven languages: "my landlord John Smith" is a private person even when a public figure shares the name. */
const PERSONAL = new Set([
  "my", "our", "mi", "mis", "nuestro", "nuestra", "nuestros", "nuestras", "mon", "ma", "mes", "notre", "nos",
  "mein", "meine", "meinen", "meinem", "meiner", "unser", "unsere", "mio", "mia", "miei", "mie", "nostro", "nostra",
  "meu", "minha", "meus", "minhas", "nosso", "nossa", "mijn", "onze", "ons",
]);
const PERSONAL_WINDOW = 3;

type Word = { readonly text: string; readonly start: number; readonly end: number };
const WORD = /[\p{L}\p{N}]+/gu;

function words(text: string): Word[] {
  return [...text.matchAll(WORD)].map((m) => ({ text: m[0], start: m.index, end: m.index + m[0].length }));
}
function fold(word: string): string {
  return word.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}
function addressWord(word: string): string {
  const w = fold(word);
  return ABBREVIATIONS[w] ?? w;
}

type Occurrence = { readonly start: number; readonly end: number };

function entries(list: unknown, kind: string): readonly string[] {
  if (list === undefined) return [];
  if (!Array.isArray(list) || list.some((v) => typeof v !== "string")) throw new TypeError(`allowlist ${kind} must be an array of strings`);
  return list;
}

export class CompiledAllowlist {
  readonly #names = new Set<string>();
  readonly #addresses = new Set<string>();
  readonly #addressPrefixes = new Set<string>();
  readonly #phones = new Set<string>();
  #maxName = 0;
  #maxAddress = 0;

  constructor(lists: readonly Allowlist[]) {
    for (const list of lists) {
      for (const name of entries(list.names, "names")) this.#addName(name);
      for (const phone of entries(list.phones, "phones")) this.#addPhone(phone);
      for (const address of entries(list.addresses, "addresses")) this.#addAddress(address);
    }
  }

  #addName(name: string): void {
    const ws = words(name).map((w) => fold(w.text));
    if (ws.length === 0) throw new RangeError(`allowlist name has no letters: ${JSON.stringify(name)}`);
    const variants = [ws, ws.filter((w) => w.length > 1), ws.filter((w) => !SUFFIXES.has(w))];
    for (const v of variants) {
      if (v.length === 0) continue;
      this.#names.add(v.join(" "));
      this.#maxName = Math.max(this.#maxName, v.length);
    }
  }

  #addPhone(phone: string): void {
    const digits = canonPhone(phone);
    if (digits.length < 10) throw new RangeError(`allowlist phone must be a complete number: ${JSON.stringify(phone)}`);
    this.#phones.add(digits);
  }

  #addAddress(address: string): void {
    const full = words(address).map((w) => addressWord(w.text));
    const street = words(address.split(",")[0]).map((w) => addressWord(w.text));
    if (street.length < 2 || !street.some((w) => /\d/.test(w))) {
      throw new RangeError(`allowlist address needs a number and a street: ${JSON.stringify(address)}`);
    }
    const complete = [full, street];
    if (DIRECTIONS.has(street[street.length - 1])) complete.push(street.slice(0, -1));
    for (const key of complete) this.#addresses.add(key.join(" "));
    for (let i = 1; i <= full.length; i++) this.#addressPrefixes.add(full.slice(0, i).join(" "));
    this.#maxAddress = Math.max(this.#maxAddress, full.length);
  }

  isPhone(value: string): boolean {
    return this.#phones.has(canonPhone(value));
  }

  /** Demotes hard phone hits on a listed number to soft and recomputes `block`. */
  applyToValidators(result: GateResult): GateResult {
    if (!result.hard.some((d) => this.#allowedPhone(d))) return result;
    const demoted = result.hard.filter((d) => this.#allowedPhone(d)).map((d) => ({ ...d, hard: false, evidence: `${d.evidence};allowlist=host` }));
    const hard = result.hard.filter((d) => !this.#allowedPhone(d));
    return { block: hard.length > 0, hard, soft: [...result.soft, ...demoted].sort((a, b) => a.start - b.start) };
  }

  #allowedPhone(d: Detection): boolean {
    return (d.label === "PHONE" || d.label === "PHONE_E164") && this.isPhone(d.value);
  }

  /** Text ranges of listed names, addresses and phone numbers, and of .gov/.mil emails. */
  occurrences(text: string, validators: GateResult): Occurrence[] {
    const ws = words(text);
    const out: Occurrence[] = [];
    for (let i = 0; i < ws.length; i++) {
      const personal = ws.slice(Math.max(0, i - PERSONAL_WINDOW), i).some((w) => PERSONAL.has(fold(w.text)));
      const name = personal ? 0 : this.#longest(ws, i, this.#maxName, fold, (k) => this.#names.has(k));
      const address = this.#addressAt(ws, i);
      const n = Math.max(name, address);
      if (n > 0) {
        out.push({ start: ws[i].start, end: ws[i + n - 1].end });
        i += n - 1;
      }
    }
    for (const d of [...validators.hard, ...validators.soft]) {
      if (this.#allowedPhone(d) || isGovEmail(d)) out.push({ start: d.start, end: d.end });
    }
    return out;
  }

  #longest(ws: readonly Word[], i: number, max: number, key: (w: string) => string, has: (k: string) => boolean): number {
    let best = 0;
    let k = "";
    for (let n = 1; n <= max && i + n <= ws.length; n++) {
      k = n === 1 ? key(ws[i].text) : `${k} ${key(ws[i + n - 1].text)}`;
      if (has(k)) best = n;
    }
    return best;
  }

  /** Longest listed street line or full address at word `i`, extended over the entry's own city, state and ZIP. */
  #addressAt(ws: readonly Word[], i: number): number {
    let matched = 0;
    let k = "";
    for (let n = 1; n <= this.#maxAddress && i + n <= ws.length; n++) {
      k = n === 1 ? addressWord(ws[i].text) : `${k} ${addressWord(ws[i + n - 1].text)}`;
      if (this.#addresses.has(k)) matched = n;
      else if (matched > 0 && this.#addressPrefixes.has(k)) matched = n;
      else if (!this.#addressPrefixes.has(k) && !this.#addresses.has(k)) break;
    }
    return matched;
  }

  /** Splits spans into those covered by an allowlist entry and the rest. */
  filterSpans(text: string, spans: readonly RedactedSpan[], validators: GateResult): { kept: RedactedSpan[]; allowed: RedactedSpan[] } {
    const occ = this.occurrences(text, validators);
    const kept: RedactedSpan[] = [];
    const allowed: RedactedSpan[] = [];
    for (const s of spans) {
      const hard = validators.hard.some((d) => d.start < s.end && s.start < d.end);
      (!hard && this.#covers(text, s, occ) ? allowed : kept).push(s);
    }
    return { kept, allowed };
  }

  #covers(text: string, span: RedactedSpan, occ: readonly Occurrence[]): boolean {
    const surface = text.slice(span.start, span.end);
    if (/\d/.test(surface) && canonPhone(surface).length >= 10 && this.isPhone(surface)) return true;
    const overlapping = occ.filter((o) => o.start < span.end && span.start < o.end);
    if (overlapping.length === 0) return false;
    return words(surface).every((w) => {
      const start = span.start + w.start;
      if (overlapping.some((o) => o.start <= start && start + w.text.length <= o.end)) return true;
      const significant = /\d/.test(w.text) || /^\p{Lu}/u.test(w.text);
      return !significant || TITLES.has(fold(w.text));
    });
  }
}

let builtIn: CompiledAllowlist | null = null;
const empty = new CompiledAllowlist([]);

/** Compiles the host lists, plus the built-in lists unless `builtIn: false`. */
export function compileAllowlist(options: AllowlistOptions = {}): CompiledAllowlist {
  const host = [options].filter((o) => o.names || o.phones || o.addresses);
  const useBuiltIn = options.builtIn ?? true;
  if (host.length === 0) return useBuiltIn ? (builtIn ??= new CompiledAllowlist([BUILT_IN_ALLOWLIST])) : empty;
  return new CompiledAllowlist(useBuiltIn ? [BUILT_IN_ALLOWLIST, ...host] : host);
}
