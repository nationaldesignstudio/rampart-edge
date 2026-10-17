import config from "../model/config.json";
import { createEngine, EngineError, engineWasmBytes, type GateEngine } from "./engine.js";
import { combineWindows, contentWindows, sigmoid, withSpecialTokens, type ModelScore } from "./scoring.js";
import { isSpanLabelSet, softmaxRows, spanWindows, type TokenProbabilities } from "./spans.js";
import { WordPieceTokenizer } from "./tokenizer.js";

/** The graph and decode fields of `model/config.json`. `output` is one name (gate only) or `[gate, tokens]` (gate + span head). */
export type ModelConfig = {
  readonly inputs: readonly string[];
  readonly output: string | readonly string[];
  readonly max_tokens: number;
  readonly stride: number;
  readonly span_head?: "typed" | "agnostic";
  readonly span_labels?: readonly string[];
  /** A token is a span when `1 - P(O)` is at least this; null takes the most likely label. */
  readonly span_threshold?: number | null;
};

export type SpanHead = {
  readonly output: string;
  readonly type: "typed" | "agnostic";
  readonly labels: readonly string[];
  readonly threshold: number | null;
};

/** The span head a config declares, or null for a gate-only model. Throws on an inconsistent config. */
export function readSpanHead(c: ModelConfig): SpanHead | null {
  if (typeof c.output === "string") {
    if (c.span_head === undefined && c.span_labels === undefined && c.span_threshold === undefined) return null;
    throw new GateModelError("graph", "model config declares a span head but output is a single name; use [gate, tokens]");
  }
  const [, output, ...rest] = c.output;
  const labels = c.span_labels ?? [];
  const type = c.span_head;
  const typed = labels.some((l) => l.includes("-"));
  const threshold = c.span_threshold;
  if (output === undefined || rest.length > 0 || !isSpanLabelSet(labels) || type !== (typed ? "typed" : "agnostic") ||
    threshold === undefined || (threshold !== null && !(threshold > 0 && threshold <= 1))) {
    throw new GateModelError("graph", "model config: output [gate, tokens] needs span_labels (O first), a matching span_head and span_threshold");
  }
  return { output, type, labels, threshold };
}

const MODEL_CONFIG = config as ModelConfig;
/** Input names of the source graph `model/onnx/model_q4.onnx`, kept for provenance; the engine takes the same two inputs. */
export const MODEL_INPUTS = ["input_ids", "attention_mask"] as const;
/** Name of the gate's logit output in the source graph. */
export const MODEL_OUTPUT: string = typeof MODEL_CONFIG.output === "string" ? MODEL_CONFIG.output : MODEL_CONFIG.output[0];
/** The shipped model's span head, or null when it is gate only. */
export const MODEL_SPAN_HEAD: SpanHead | null = readSpanHead(MODEL_CONFIG);

/**
 * One error class for every way the model half can fail, so hosts can fall back to `createValidatorGate()` and log
 * the `stage`. Engine failures keep their `EngineError` as `cause` and map onto these stages:
 *
 * | `EngineError.stage` | `GateModelError.stage` | meaning |
 * |---|---|---|
 * | `wasm` | `runtime` | the engine wasm could not be compiled or instantiated (no WebAssembly SIMD128) |
 * | `format` | `graph` | the weights are not the RMPW v1 model this package was built for |
 * | `integrity` | `integrity` | the weight payload does not match its embedded SHA-256 (like a failed file pin) |
 * | `init` | `session` | the engine trapped or refused while expanding the weights |
 * | `range`, `numeric` | `inference` | a window was rejected or produced a non-finite logit |
 */
export class GateModelError extends Error {
  readonly stage: "runtime" | "download" | "integrity" | "session" | "graph" | "inference";
  constructor(stage: GateModelError["stage"], message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "GateModelError";
    this.stage = stage;
  }
}

const ENGINE_STAGES: Record<EngineError["stage"], GateModelError["stage"]> = {
  wasm: "runtime",
  format: "graph",
  integrity: "integrity",
  init: "session",
  range: "inference",
  numeric: "inference",
};

/** Wraps an engine failure as the `GateModelError` stage hosts already handle; other errors become `inference`. */
export function fromEngineError(error: unknown, fallback: GateModelError["stage"] = "inference"): GateModelError {
  if (error instanceof GateModelError) return error;
  if (error instanceof EngineError) return new GateModelError(ENGINE_STAGES[error.stage], error.message, { cause: error });
  return new GateModelError(fallback, error instanceof Error ? error.message : String(error), { cause: error });
}

export type PinnedFile = { readonly bytes: number; readonly sha256: string };

export async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", view as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Throws a `GateModelError("integrity")` unless `bytes` has the pinned length and SHA-256. */
export async function verifyPinned(name: string, bytes: Uint8Array, pin: PinnedFile): Promise<void> {
  if (bytes.byteLength !== pin.bytes) {
    throw new GateModelError("integrity", `${name} is ${bytes.byteLength} bytes, expected ${pin.bytes}`);
  }
  if ((await sha256Hex(bytes)) !== pin.sha256) {
    throw new GateModelError("integrity", `${name} does not match its pinned SHA-256`);
  }
}

/** Fetches `url` and verifies it against `pin` before returning the bytes. */
export async function fetchPinned(url: string | URL, pin: PinnedFile, signal?: AbortSignal): Promise<Uint8Array> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new GateModelError("download", `${url} answered ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await verifyPinned(String(url), bytes, pin);
  return bytes;
}

export type ModelAnalysis = {
  readonly score: ModelScore;
  /** Per-window token probabilities from the span head of the same run; null for a gate-only model. */
  readonly tokens: TokenProbabilities | null;
};

export type GateModel = {
  readonly tokenizer: WordPieceTokenizer;
  readonly spanHead: SpanHead | null;
  /** The engine running the forward pass (`loadMs`, `memoryPages`). */
  readonly engine: GateEngine;
  /** BLOCK probability for `text`: max over 384-token windows. The text never leaves the process. */
  score(text: string): Promise<ModelScore>;
  /** One inference per window: the gate score and, with a span head, token probabilities. */
  analyze(text: string): Promise<ModelAnalysis>;
};

export type GateModelOptions = {
  /** Graph and decode fields; defaults to the shipped `model/config.json`. */
  readonly config?: ModelConfig;
};

export type GateModelSource = {
  /** Verified `model.rmpw` bytes. */
  readonly weights: Uint8Array;
  /** Contents of the matching `vocab.txt`. */
  readonly vocab: string;
  /** The engine wasm; defaults to the copy inlined in this package (`engineWasmBytes()`). */
  readonly wasm?: BufferSource | WebAssembly.Module;
} & GateModelOptions;

/** Builds the engine from `weights` (and the inlined wasm unless `wasm` is given), then the classifier half of the gate. */
export async function createGateModelFromWeights(source: GateModelSource): Promise<GateModel> {
  let engine: GateEngine;
  try {
    engine = await createEngine(source.wasm ?? engineWasmBytes(), source.weights);
  } catch (error) {
    throw fromEngineError(error, "session");
  }
  return createGateModel(engine, source.vocab, { config: source.config });
}

/**
 * Builds the classifier half of the gate from a loaded engine and the matching `vocab.txt`. Runs one probe window so
 * an engine whose span head does not match `config` fails here with `GateModelError("graph")`, not on the first message.
 */
export function createGateModel(engine: GateEngine, vocab: string, options: GateModelOptions = {}): GateModel {
  const tokenizer = new WordPieceTokenizer(vocab);
  const modelConfig = options.config ?? MODEL_CONFIG;
  const spanHead = readSpanHead(modelConfig);
  const nLabels = spanHead?.labels.length ?? 0;

  function forward(ids: readonly number[]): { gateLogit: number; tokenLogits: Float32Array } {
    try {
      return engine.run(ids);
    } catch (error) {
      throw fromEngineError(error);
    }
  }

  const probe = forward(withSpecialTokens([], tokenizer.clsId, tokenizer.sepId));
  if (probe.tokenLogits.length % 2 !== 0 || (spanHead !== null && probe.tokenLogits.length !== 2 * nLabels)) {
    throw new GateModelError("graph", `engine span head has ${probe.tokenLogits.length / 2} labels, config declares ${nLabels}`);
  }

  function runWindow(window: readonly number[]): { probability: number; rows: number[][] | null } {
    const ids = withSpecialTokens(window, tokenizer.clsId, tokenizer.sepId);
    const { gateLogit, tokenLogits } = forward(ids);
    if (spanHead === null) return { probability: sigmoid(gateLogit), rows: null };
    if (tokenLogits.length !== ids.length * nLabels) {
      throw new GateModelError("inference", `token logits are not [${ids.length}, ${nLabels}]`);
    }
    return { probability: sigmoid(gateLogit), rows: softmaxRows(tokenLogits, ids.length, nLabels) };
  }

  async function analyze(text: string): Promise<ModelAnalysis> {
    if (spanHead === null) {
      const probabilities: number[] = [];
      for (const window of contentWindows(tokenizer.encode(text))) probabilities.push(runWindow(window).probability);
      return { score: combineWindows(probabilities), tokens: null };
    }
    const { ids, offsets } = tokenizer.encodeWithOffsets(text);
    if (ids.length === 0) {
      return { score: combineWindows([runWindow([]).probability]), tokens: { ids, offsets, windows: [], probabilities: [] } };
    }
    const windows = spanWindows(ids.length, modelConfig.max_tokens, modelConfig.stride);
    const gate: number[] = [];
    const probabilities: number[][][] = [];
    for (const w of windows) {
      const { probability, rows } = runWindow(ids.slice(w.start, w.end));
      gate.push(probability);
      probabilities.push(rows ?? []);
    }
    return { score: combineWindows(gate), tokens: { ids, offsets, windows, probabilities } };
  }

  return {
    tokenizer,
    spanHead,
    engine,
    analyze,
    async score(text) {
      return (await analyze(text)).score;
    },
  };
}
