/** Token-head decoding shared by the gate's span head and the span redactor: windows, BIO decode, validator union. */
import { assertThreshold, MAX_TOKENS, STRIDE } from "./scoring.js";
import { gate as validatorGate, type GateResult, type Label } from "./validators/index.js";

/** The nine coarse PII span types. */
export const SPAN_TYPES = [
  "NAME",
  "ADDRESS",
  "EMAIL",
  "PHONE",
  "GOV_ID",
  "FINANCIAL",
  "DOB_KEYED",
  "URL_HANDLE",
  "OTHER_ID",
] as const;
export type SpanType = (typeof SPAN_TYPES)[number];
/** Class-agnostic model spans are `SPAN`; typed model spans and validator spans carry their coarse type. */
export type RedactedSpanType = SpanType | "SPAN";

const VALIDATOR_TYPES: Partial<Record<Label, SpanType>> = {
  SSN: "GOV_ID",
  ITIN: "GOV_ID",
  EIN: "GOV_ID",
  MEDICARE_MBI: "GOV_ID",
  USCIS_A_NUMBER: "GOV_ID",
  USCIS_RECEIPT: "GOV_ID",
  PASSPORT: "GOV_ID",
  MRZ: "GOV_ID",
  DRIVERS_LICENSE: "GOV_ID",
  ROUTING_NUMBER: "FINANCIAL",
  BANK_ACCOUNT: "FINANCIAL",
  CREDIT_CARD: "FINANCIAL",
  IBAN: "FINANCIAL",
  PHONE: "PHONE",
  PHONE_E164: "PHONE",
  EMAIL: "EMAIL",
};

export type RedactedSpan = {
  /** UTF-16 offsets into the input (`text.slice(start, end)`), end exclusive. */
  readonly start: number;
  readonly end: number;
  readonly type: RedactedSpanType;
  /** Model spans: mean over tokens of the decoded label's probability, the higher score when spans merge; a fragment left after a validator hit is cut out keeps its parent's score. 1 for validator spans. */
  readonly probability: number;
  readonly source: "model" | "validator";
  /** Validator rule that produced the span. */
  readonly rule?: string;
};

export type SpanWindow = { start: number; end: number; coreStart: number; coreEnd: number };

export type TokenProbabilities = {
  readonly ids: readonly number[];
  readonly offsets: readonly (readonly [number, number])[];
  readonly windows: readonly SpanWindow[];
  /** Per window, per content token: softmax over the head's labels. */
  readonly probabilities: readonly (readonly (readonly number[])[])[];
};

/** Labels a token head may declare: `O` first, then `B`/`I` (agnostic) or `B-<type>`/`I-<type>` pairs (typed). */
export function isSpanLabelSet(labels: readonly string[]): boolean {
  if (labels.length < 3 || labels[0] !== "O") return false;
  return labels.slice(1).every((l) => l === "B" || l === "I" || (/^[BI]-/.test(l) && (SPAN_TYPES as readonly string[]).includes(l.slice(2))));
}

/**
 * Windows of `maxTokens - 2` content tokens sharing `stride` tokens, the last one flush with the end; each overlap
 * is owned half by either side until `snapOwnership` moves the boundary.
 */
export function spanWindows(n: number, maxTokens = MAX_TOKENS, stride = STRIDE): SpanWindow[] {
  const body = maxTokens - 2;
  if (body <= stride) throw new Error("stride must be smaller than maxTokens - 2");
  if (n <= body) return [{ start: 0, end: n, coreStart: 0, coreEnd: n }];
  const step = body - stride;
  const starts: number[] = [];
  for (let s = 0; s < n - body; s += step) starts.push(s);
  starts.push(n - body);
  const wins = starts.map((s) => ({ start: s, end: s + body, coreStart: s, coreEnd: s + body }));
  for (let i = 1; i < wins.length; i++) {
    const mid = Math.floor((wins[i].start + wins[i - 1].end) / 2);
    wins[i - 1].coreEnd = mid;
    wins[i].coreStart = mid;
  }
  return wins;
}

/** Moves each ownership boundary to the overlap token nearest the middle that both windows label O. */
export function snapOwnership(wins: readonly SpanWindow[], perWindow: readonly (readonly number[])[], oId = 0): SpanWindow[] {
  const out = wins.map((w) => ({ ...w }));
  for (let i = 1; i < out.length; i++) {
    const prev = out[i - 1];
    const cur = out[i];
    const mid = cur.coreStart;
    const candidates: number[] = [];
    for (let t = cur.start; t < prev.end; t++) candidates.push(t);
    candidates.sort((a, b) => Math.abs(a - mid) - Math.abs(b - mid) || a - b);
    for (const t of candidates) {
      if (perWindow[i - 1][t - prev.start] === oId && perWindow[i][t - cur.start] === oId) {
        prev.coreEnd = t;
        cur.coreStart = t;
        break;
      }
    }
  }
  return out;
}

/** Per token softmax row -> label id and its probability: argmax when `threshold` is null, else PII when `1 - P(O) >= threshold` (most likely non-O label). */
export function labelTokens(probabilities: readonly (readonly number[])[], threshold: number | null, oId = 0): { labels: number[]; scores: number[] } {
  const t = threshold === null ? null : Math.fround(assertThreshold(threshold, "span threshold"));
  const labels: number[] = [];
  const scores: number[] = [];
  for (const row of probabilities) {
    let best = -1;
    for (let k = 0; k < row.length; k++) {
      if (t !== null && k === oId) continue;
      if (best < 0 || row[k] > row[best]) best = k;
    }
    if (t === null || Math.fround(1 - Math.fround(row[oId])) >= t) {
      labels.push(best);
      scores.push(row[best]);
    } else {
      labels.push(oId);
      scores.push(row[oId]);
    }
  }
  return { labels, scores };
}

/** Softmax over `labels` for each content token of one window's `[1, tokens, labels]` logits (drops [CLS] and [SEP]). */
export function softmaxRows(logits: Float32Array, tokens: number, labels: number): number[][] {
  const rows: number[][] = [];
  for (let t = 1; t < tokens - 1; t++) {
    const row = Array.from(logits.subarray(t * labels, (t + 1) * labels));
    const max = Math.max(...row);
    const exp = row.map((x) => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    rows.push(exp.map((x) => x / sum));
  }
  return rows;
}

type Span = { start: number; end: number; type: RedactedSpanType; score: number; source: "model" | "validator"; rule?: string };

function labelType(label: string): RedactedSpanType | null {
  if (label === "O") return null;
  return label.includes("-") ? (label.slice(2) as SpanType) : "SPAN";
}

/** Lenient BIO: an I- without a preceding span of the same type opens one; a span ends at its last token's end. */
function decode(offsets: readonly (readonly [number, number])[], labelIds: readonly number[], scores: readonly number[], labels: readonly string[]): Span[] {
  const out: Span[] = [];
  let cur: Span | null = null;
  let n = 0;
  offsets.forEach(([a, b], ti) => {
    if (b <= a) return;
    const label = labels[labelIds[ti]];
    const type = labelType(label);
    if (type === null) {
      if (cur) out.push(cur);
      cur = null;
      return;
    }
    if (label.startsWith("B") || cur === null || cur.type !== type) {
      if (cur) out.push(cur);
      cur = { start: a, end: b, type, score: scores[ti], source: "model" };
      n = 1;
    } else {
      cur.end = b;
      cur.score = (cur.score * n + scores[ti]) / (n + 1);
      n += 1;
    }
  });
  if (cur) out.push(cur);
  return out;
}

const LEADING = new Set([..." \t\n\r(\"'«“‘"]);
const TRAILING = new Set([..." \t\n\r.,;:!?)\"'»”’"]);
const CONNECTORS = new Set([..." \t,.-'’"]);

function trimSpans(chars: readonly string[], spans: readonly Span[]): Span[] {
  const res: Span[] = [];
  for (const s of spans) {
    let a = s.start;
    let b = s.end;
    while (a < b && LEADING.has(chars[a])) a += 1;
    while (b > a && TRAILING.has(chars[b - 1])) b -= 1;
    if (b > a) res.push({ ...s, start: a, end: b });
  }
  return res;
}

function sortSpans(spans: readonly Span[]): Span[] {
  return [...spans].sort((x, y) => x.start - y.start || y.end - x.end);
}

function joins(chars: readonly string[], p: Span, s: Span): boolean {
  const gap = s.start >= p.end ? chars.slice(p.end, s.start) : [];
  return p.type === s.type && (s.start <= p.end || (gap.length <= 1 && gap.every((c) => CONNECTORS.has(c))));
}

/** Model spans: same-type spans that overlap or sit one connector apart merge; a different-type overlap keeps the higher-scored span. */
function mergePredicted(chars: readonly string[], spans: readonly Span[]): Span[] {
  const out: Span[] = [];
  for (const s of sortSpans(spans)) {
    const p = out.at(-1);
    if (p && joins(chars, p, s)) {
      p.end = Math.max(p.end, s.end);
      p.score = Math.max(p.score, s.score);
      continue;
    }
    if (p && s.start < p.end) {
      if (s.score > p.score) out[out.length - 1] = { ...s };
      continue;
    }
    out.push({ ...s });
  }
  return out;
}

/** Union pass: like `mergePredicted`, but different-type overlaps cover their union and a validator's type and rule win. */
function mergeSpans(chars: readonly string[], spans: readonly Span[]): Span[] {
  const out: Span[] = [];
  for (const s of sortSpans(spans)) {
    const p = out.at(-1);
    if (p && (joins(chars, p, s) || s.start < p.end)) {
      p.end = Math.max(p.end, s.end);
      p.score = Math.max(p.score, s.score);
      if (p.type !== s.type && s.source === "validator" && p.source !== "validator") {
        p.type = s.type;
        p.source = s.source;
        p.rule = s.rule;
      }
      continue;
    }
    out.push({ ...s });
  }
  return out;
}

function toSpan(s: Span): RedactedSpan {
  return { start: s.start, end: s.end, type: s.type, probability: s.score, source: s.source, ...(s.rule ? { rule: s.rule } : {}) };
}

function fromSpan(s: RedactedSpan): Span {
  return { start: s.start, end: s.end, type: s.type, score: s.probability, source: s.source, rule: s.rule };
}

/** Model spans for `text` from per-window token probabilities: label, stitch across windows, BIO decode, trim, merge. */
export function decodeSpans(
  text: string,
  tokens: TokenProbabilities,
  labels: readonly string[],
  threshold: number | null,
): { spans: RedactedSpan[]; windows: SpanWindow[] } {
  if (tokens.ids.length === 0) return { spans: [], windows: [] };
  const perWindow = tokens.probabilities.map((p) => labelTokens(p, threshold));
  const snapped = snapOwnership(tokens.windows, perWindow.map((p) => p.labels));
  const ids = new Array<number>(tokens.ids.length).fill(0);
  const scores = new Array<number>(tokens.ids.length).fill(0);
  snapped.forEach((w, i) => {
    for (let t = w.coreStart; t < w.coreEnd; t++) {
      ids[t] = perWindow[i].labels[t - w.start];
      scores[t] = perWindow[i].scores[t - w.start];
    }
  });
  const chars = text.split("");
  return { spans: mergePredicted(chars, trimSpans(chars, decode(tokens.offsets, ids, scores, labels))).map(toSpan), windows: snapped };
}

/** Validator hard hits of the structured types. */
function validatorSpans(dets: GateResult): Span[] {
  const out: Span[] = [];
  for (const d of dets.hard) {
    const type = VALIDATOR_TYPES[d.label];
    if (type) out.push({ start: d.start, end: d.end, type, score: 1, source: "validator", rule: d.rule });
  }
  return out;
}

/** Date shapes the validators see: a keyed date of birth is `DOB_KEYED`, a bare date stays `SPAN` (shape only). The gate treats both as soft. */
function dateShapes(dets: GateResult): Span[] {
  const out: Span[] = [];
  for (const d of [...dets.hard, ...dets.soft]) {
    if (d.rule !== "date") continue;
    out.push({ start: d.start, end: d.end, type: d.label === "DOB" ? "DOB_KEYED" : "SPAN", score: 1, source: "validator", rule: d.rule });
  }
  return out;
}

/** A model span that covers part of a date shape is widened to the whole shape, so "07/22/1983" never comes out as "07[REDACTED]22/1983". */
function snapToDates(model: readonly Span[], dates: readonly Span[]): Span[] {
  return model.map((s) => {
    let { start, end } = s;
    for (const d of dates) {
      if (d.start < end && start < d.end) {
        start = Math.min(start, d.start);
        end = Math.max(end, d.end);
      }
    }
    return { ...s, start, end };
  });
}

/** Model spans with the validator intervals cut out (whitespace at the cut trimmed), so every validator hit is reported with its own type. */
function subtract(chars: readonly string[], model: readonly Span[], validators: readonly Span[]): Span[] {
  const out: Span[] = [];
  for (const m of model) {
    let pieces: Span[] = [m];
    for (const v of validators) {
      pieces = pieces.flatMap((p) => {
        if (v.end <= p.start || p.end <= v.start) return [p];
        const rest: Span[] = [];
        if (p.start < v.start) rest.push({ ...p, end: v.start });
        if (v.end < p.end) rest.push({ ...p, start: v.end });
        return rest;
      });
    }
    for (const p of pieces) {
      let { start, end } = p;
      while (start < end && /\s/.test(chars[start])) start += 1;
      while (end > start && /\s/.test(chars[end - 1])) end -= 1;
      if (start < end) out.push({ ...p, start, end });
    }
  }
  return out;
}

/**
 * Model spans unioned with the deterministic pass: structured validator hits (which own the text they cover),
 * keyed dates of birth, and model spans widened to whole dates. A model span overlapping a validator hit keeps the part outside it.
 */
export function deterministicUnion(text: string, modelSpans: readonly RedactedSpan[], dets: GateResult = validatorGate(text)): RedactedSpan[] {
  const dates = dateShapes(dets);
  const model = snapToDates(modelSpans.map(fromSpan), dates);
  const validators = [...validatorSpans(dets), ...dates.filter((d) => d.type === "DOB_KEYED")];
  const chars = text.split("");
  return mergeSpans(chars, [...subtract(chars, model, validators), ...validators]).map(toSpan);
}
