import config from "../model/config.json";
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
/** Exact graph names of `model/onnx/model_q4.onnx`; a different export fails loudly at load. */
export const MODEL_INPUTS = ["input_ids", "attention_mask"] as const;
/** Name of the gate's logit output. */
export const MODEL_OUTPUT: string = typeof MODEL_CONFIG.output === "string" ? MODEL_CONFIG.output : MODEL_CONFIG.output[0];
/** The shipped model's span head, or null when it is gate only. */
export const MODEL_SPAN_HEAD: SpanHead | null = readSpanHead(MODEL_CONFIG);

export type OrtTensor = { readonly data: unknown };
export type OrtSession = {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
};
/** The slice of `onnxruntime-web` (or `onnxruntime-node`) the gate needs. */
export type OrtLike = {
  InferenceSession: {
    create(model: Uint8Array, options: { executionProviders: string[] }): Promise<OrtSession>;
  };
  Tensor: new (type: "int64", data: BigInt64Array, dims: number[]) => OrtTensor;
};

export class GateModelError extends Error {
  readonly stage: "runtime" | "download" | "integrity" | "session" | "graph" | "inference";
  constructor(stage: GateModelError["stage"], message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "GateModelError";
    this.stage = stage;
  }
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
  /** BLOCK probability for `text`: max over 384-token windows. The text never leaves the process. */
  score(text: string): Promise<ModelScore>;
  /** One inference per window: the gate score and, with a span head, token probabilities. */
  analyze(text: string): Promise<ModelAnalysis>;
};

export type GateModelOptions = {
  readonly executionProviders?: readonly string[];
  /** Graph and decode fields; defaults to the shipped `model/config.json`. */
  readonly config?: ModelConfig;
};

/** Builds the classifier half of the gate from verified model bytes and the matching `vocab.txt`. */
export async function createGateModel(
  ort: OrtLike,
  modelBytes: Uint8Array,
  vocab: string,
  options: GateModelOptions = {},
): Promise<GateModel> {
  const tokenizer = new WordPieceTokenizer(vocab);
  let session: OrtSession;
  try {
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: [...(options.executionProviders ?? ["wasm"])],
    });
  } catch (error) {
    throw new GateModelError("session", "the runtime could not create a session for the gate model", {
      cause: error,
    });
  }
  const modelConfig = options.config ?? MODEL_CONFIG;
  const spanHead = readSpanHead(modelConfig);
  const gateOutput = typeof modelConfig.output === "string" ? modelConfig.output : modelConfig.output[0];
  const outputs = spanHead === null ? [gateOutput] : [gateOutput, spanHead.output];
  const inputs = [...session.inputNames].sort().join(",");
  if (inputs !== [...MODEL_INPUTS].sort().join(",") || !outputs.every((o) => session.outputNames.includes(o))) {
    throw new GateModelError(
      "graph",
      `gate graph exposes ${session.inputNames.join(",")} -> ${session.outputNames.join(",")}`,
    );
  }
  const nLabels = spanHead?.labels.length ?? 0;

  async function runWindow(window: readonly number[]): Promise<{ probability: number; rows: number[][] | null }> {
    const ids = withSpecialTokens(window, tokenizer.clsId, tokenizer.sepId);
    const dims = [1, ids.length];
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), dims),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), dims),
    };
    let result: Record<string, OrtTensor>;
    try {
      result = await session.run(feeds);
    } catch (error) {
      throw new GateModelError("inference", "gate inference failed", { cause: error });
    }
    const gate = result[gateOutput]?.data;
    if (!(gate instanceof Float32Array) || gate.length !== 1) {
      throw new GateModelError("inference", `${gateOutput} is not a single float32`);
    }
    if (spanHead === null) return { probability: sigmoid(gate[0]), rows: null };
    const logits = result[spanHead.output]?.data;
    if (!(logits instanceof Float32Array) || logits.length !== ids.length * nLabels) {
      throw new GateModelError("inference", `${spanHead.output} is not float32 [1, ${ids.length}, ${nLabels}]`);
    }
    return { probability: sigmoid(gate[0]), rows: softmaxRows(logits, ids.length, nLabels) };
  }

  async function analyze(text: string): Promise<ModelAnalysis> {
    if (spanHead === null) {
      const probabilities: number[] = [];
      for (const window of contentWindows(tokenizer.encode(text))) probabilities.push((await runWindow(window)).probability);
      return { score: combineWindows(probabilities), tokens: null };
    }
    const { ids, offsets } = tokenizer.encodeWithOffsets(text);
    if (ids.length === 0) {
      return { score: combineWindows([(await runWindow([])).probability]), tokens: { ids, offsets, windows: [], probabilities: [] } };
    }
    const windows = spanWindows(ids.length, modelConfig.max_tokens, modelConfig.stride);
    const gate: number[] = [];
    const probabilities: number[][][] = [];
    for (const w of windows) {
      const { probability, rows } = await runWindow(ids.slice(w.start, w.end));
      gate.push(probability);
      probabilities.push(rows ?? []);
    }
    return { score: combineWindows(gate), tokens: { ids, offsets, windows, probabilities } };
  }

  return {
    tokenizer,
    spanHead,
    analyze,
    async score(text) {
      return (await analyze(text)).score;
    },
  };
}
