import config from "../model/config.json";
import type { RedactedSpan } from "./spans.js";
import type { GateResult } from "./validators/index.js";

/** Operating point of the shipped model: the lowest dev threshold with at most 1% false blocks. */
export const BLOCK_THRESHOLD: number = config.block_threshold;
/** Window geometry the gate was trained and evaluated with. */
export const MAX_TOKENS: number = config.max_tokens;
export const STRIDE: number = config.stride;
/** Content tokens per window once [CLS] and [SEP] take their two positions. */
export const CONTENT_TOKENS = MAX_TOKENS - 2;

/**
 * Splits content ids into 382-token windows that overlap by `STRIDE` tokens, so every token is
 * scored at least once. The first window is exactly what the Python evaluation scored; longer
 * inputs add windows starting every `CONTENT_TOKENS - STRIDE` tokens, the Hugging Face stride
 * convention. A text that fits yields one window; empty text yields one empty window.
 */
export function contentWindows(ids: readonly number[]): number[][] {
  if (ids.length <= CONTENT_TOKENS) return [ids.slice()];
  const windows: number[][] = [];
  const step = CONTENT_TOKENS - STRIDE;
  for (let start = 0; ; start += step) {
    windows.push(ids.slice(start, start + CONTENT_TOKENS));
    if (start + CONTENT_TOKENS >= ids.length) return windows;
  }
}

export function withSpecialTokens(
  window: readonly number[],
  clsId: number,
  sepId: number,
): number[] {
  return [clsId, ...window, sepId];
}

export function sigmoid(logit: number): number {
  return 1 / (1 + Math.exp(-logit));
}

export type ModelScore = {
  /** BLOCK probability per window, in input order. */
  readonly windows: readonly number[];
  /** Conservative combination: the highest window probability. */
  readonly probability: number;
};

export function combineWindows(probabilities: readonly number[]): ModelScore {
  if (probabilities.length === 0) throw new Error("a model score needs at least one window");
  for (const p of probabilities) {
    if (!Number.isFinite(p) || p < 0 || p > 1)
      throw new Error(`window probability out of range: ${p}`);
  }
  return { windows: probabilities, probability: Math.max(...probabilities) };
}

/** Throws unless `threshold` is a finite number in (0, 1]: NaN would make every comparison false and fail open. */
export function assertThreshold(threshold: unknown, name = "threshold"): number {
  if (typeof threshold !== "number" || !Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError(`${name} must be a finite number in (0, 1], got ${String(threshold)}`);
  }
  return threshold;
}

export function modelBlocks(score: ModelScore, threshold = BLOCK_THRESHOLD): boolean {
  return score.probability >= assertThreshold(threshold);
}

export type Verdict = "BLOCK" | "ALLOW";
export type DecisionBasis =
  | "validator-and-model"
  | "validator"
  | "model"
  | "clear"
  /** The model blocked, but every span it found is on the allowlist. */
  | "allowlist"
  | "validator-only-model-unavailable"
  | "clear-model-unavailable";

export type GateDecision = {
  readonly verdict: Verdict;
  readonly basis: DecisionBasis;
  readonly validators: GateResult;
  readonly model: ModelScore | null;
  readonly threshold: number;
  /** Model spans unioned with validator spans, minus allowlisted ones; present only when the model has a span head. */
  readonly spans?: readonly RedactedSpan[];
};

/** Production semantics: deterministic validators OR model. The model is advisory when it failed to load. */
export function decide(
  validators: GateResult,
  model: ModelScore | null,
  threshold = BLOCK_THRESHOLD,
): GateDecision {
  assertThreshold(threshold);
  const validatorHit = validators.block;
  if (model === null) {
    return {
      verdict: validatorHit ? "BLOCK" : "ALLOW",
      basis: validatorHit ? "validator-only-model-unavailable" : "clear-model-unavailable",
      validators,
      model,
      threshold,
    };
  }
  const modelHit = modelBlocks(model, threshold);
  const basis: DecisionBasis =
    validatorHit && modelHit
      ? "validator-and-model"
      : validatorHit
        ? "validator"
        : modelHit
          ? "model"
          : "clear";
  return {
    verdict: validatorHit || modelHit ? "BLOCK" : "ALLOW",
    basis,
    validators,
    model,
    threshold,
  };
}
