/**
 * Rampart Edge span redactor: marks likely PII in text (for example text extracted from a PDF, or a chat message the gate
 * just blocked) with the gate model's span head and replaces it with `[REDACTED]`. Model spans are unioned with the
 * deterministic validators' hard hits, widened to whole dates, and keyed dates of birth are redacted even when the model misses them.
 *
 * Only reachable through the `./redactor` package export. `createRedactor({ gate })` reuses a loaded gate and downloads nothing.
 */
import { assertThreshold } from "./scoring.js";
import { DEFAULT_MODEL_BASE_URL, fetchGateModel, type Gate, type GateOptions } from "./gate.js";
import { gate as validatorGate } from "./validators/index.js";
import { isGovEmail } from "./validators/rules.js";
import { GateModelError, type GateModel } from "./model.js";
import {
  decodeSpans,
  deterministicUnion,
  labelTokens,
  snapOwnership,
  SPAN_TYPES,
  spanWindows,
  type RedactedSpan,
  type RedactedSpanType,
  type SpanType,
  type SpanWindow,
  type TokenProbabilities,
} from "./spans.js";
import { WordPieceTokenizer } from "./tokenizer.js";

export { GateModelError, DEFAULT_MODEL_BASE_URL };

export const REDACTED_PLACEHOLDER = "[REDACTED]";

export type RedactorWindow = SpanWindow;
export { SPAN_TYPES, labelTokens, snapOwnership, spanWindows, type RedactedSpan, type RedactedSpanType, type SpanType, type TokenProbabilities };

export type RedactResult = {
  /** The input with every span replaced by the placeholder. */
  readonly text: string;
  readonly spans: readonly RedactedSpan[];
  /** Number of 384-token windows the model read. */
  readonly windows: number;
  /** The decode threshold; null when tokens take the most likely label (argmax). */
  readonly threshold: number | null;
};

/** Replaces each span (UTF-16 offsets, non-overlapping) with `placeholder`. */
export function applyRedactions(text: string, spans: readonly Pick<RedactedSpan, "start" | "end">[], placeholder = REDACTED_PLACEHOLDER): string {
  let out = "";
  let at = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start < at) continue;
    out += text.slice(at, s.start) + placeholder;
    at = s.end;
  }
  return out + text.slice(at);
}

export type RedactorModel = {
  readonly tokenizer: WordPieceTokenizer;
  /** Labels of the token head (`O` first). */
  readonly labels: readonly string[];
  /** Default decode threshold; null = argmax. */
  readonly threshold: number | null;
  /** Token ids, UTF-16 offsets and softmax probabilities per window. The text never leaves the process. */
  tokenProbabilities(text: string): Promise<TokenProbabilities>;
  /** Model spans only (no validators), decoded at `threshold`. */
  spans(text: string, threshold?: number | null): Promise<{ spans: RedactedSpan[]; windows: RedactorWindow[] }>;
};

function asRedactorModel(
  tokenizer: WordPieceTokenizer,
  labels: readonly string[],
  threshold: number | null,
  tokenProbabilities: (text: string) => Promise<TokenProbabilities>,
): RedactorModel {
  return {
    tokenizer,
    labels,
    threshold,
    tokenProbabilities,
    async spans(text, t = threshold) {
      return decodeSpans(text, await tokenProbabilities(text), labels, t);
    },
  };
}

/** The span head of a loaded gate model, so redaction reuses the gate's single model. */
export function redactorModelFromGate(model: GateModel): RedactorModel {
  const head = model.spanHead;
  if (head === null) throw new GateModelError("graph", "the gate model has no span head");
  return asRedactorModel(model.tokenizer, head.labels, head.threshold, async (text) => {
    const { tokens } = await model.analyze(text);
    if (tokens === null) throw new GateModelError("inference", "the gate model returned no token probabilities");
    return tokens;
  });
}

export type RedactOptions = {
  /** Redact when `1 - P(O)` is at least this; null takes each token's most likely label. Defaults to the model's operating point. */
  readonly threshold?: number | null;
  /** Union model spans with the validators' hard hits (default true). */
  readonly validators?: boolean;
  readonly placeholder?: string;
};

export class Redactor {
  readonly model: RedactorModel;
  readonly threshold: number | null;
  readonly validators: boolean;
  readonly placeholder: string;

  constructor(model: RedactorModel, options: RedactOptions = {}) {
    this.model = model;
    const threshold = options.threshold === undefined ? model.threshold : options.threshold;
    this.threshold = threshold === null ? null : assertThreshold(threshold, "span threshold");
    this.validators = options.validators ?? true;
    this.placeholder = options.placeholder ?? REDACTED_PLACEHOLDER;
  }

  async redact(text: string): Promise<RedactResult> {
    const { spans: modelSpans, windows } = await this.model.spans(text, this.threshold);
    const dets = validatorGate(text);
    const govEmails = [...dets.hard, ...dets.soft].filter(isGovEmail);
    const kept = modelSpans.filter(
      (s) => !govEmails.some((e) => s.start < e.end && e.start < s.end && (s.type === "EMAIL" || (s.start >= e.start && s.end <= e.end))),
    );
    const spans = this.validators ? deterministicUnion(text, kept, dets) : kept;
    return { text: applyRedactions(text, spans, this.placeholder), spans, windows: windows.length, threshold: this.threshold };
  }
}

export interface RedactorOptions extends RedactOptions, Pick<GateOptions, "modelBaseUrl" | "engineWasm" | "signal"> {
  /** A gate already loaded on this thread; redaction reuses its model and downloads nothing. */
  readonly gate?: Gate;
}

/** Builds the redactor from the gate model's span head, reusing `options.gate` or downloading the gate's pinned files. Throws `GateModelError` on any failure. */
export async function createRedactor(options: RedactorOptions = {}): Promise<Redactor> {
  if (options.gate?.model) return new Redactor(redactorModelFromGate(options.gate.model), options);
  const model = await fetchGateModel(options);
  return new Redactor(redactorModelFromGate(model), options);
}
