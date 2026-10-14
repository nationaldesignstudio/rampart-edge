import config from "../model/config.json";
import { compileAllowlist, type AllowlistOptions, type CompiledAllowlist } from "./allowlist.js";
import { createGateModel, fetchPinned, GateModelError, type GateModel, type OrtLike } from "./model.js";
import { assertThreshold, decide, type GateDecision } from "./scoring.js";
import { decodeSpans, deterministicUnion } from "./spans.js";
import { gate as validatorGate, type GateResult } from "./validators/index.js";
import { applyBundledRuntime, type OrtWasmEnv } from "./runtime.js";

export const MODEL_ID = config.model_id;
export const MODEL_FILES = config.files;
/** Where `createGate` fetches the model from unless `modelBaseUrl` points at your own origin. */
export const DEFAULT_MODEL_BASE_URL = `https://huggingface.co/${config.model_id}/resolve/main/`;

export interface GateOptions {
  /** An `onnxruntime-web` (or `onnxruntime-node`) module. Defaults to `import("onnxruntime-web")`. */
  readonly ort?: OrtLike;
  /** Directory that serves `onnx/model_q4.onnx` and `vocab.txt`. Self-host it for production. */
  readonly modelBaseUrl?: string | URL;
  /** BLOCK when the model probability is at least this. Defaults to the published operating point. */
  readonly threshold?: number;
  /** Names, phone numbers and addresses that may pass. Adds to the built-in lists unless `builtIn: false`. */
  readonly allowlist?: AllowlistOptions;
  readonly executionProviders?: readonly string[];
  readonly signal?: AbortSignal;
}

/**
 * The Rampart Edge gate: deterministic validators OR the classifier. `check` answers BLOCK or ALLOW
 * for one message; nothing about the message is sent anywhere.
 */
export class Gate {
  readonly threshold: number;
  readonly model: GateModel | null;
  readonly allowlist: CompiledAllowlist;

  constructor(model: GateModel | null, threshold: number = config.block_threshold, allowlist: AllowlistOptions = {}) {
    this.model = model;
    this.threshold = assertThreshold(threshold);
    this.allowlist = compileAllowlist(allowlist);
  }

  /** Deterministic validators only (SSNs, cards, phones, emails, ...), with allowlisted phone numbers demoted to soft. Synchronous. */
  validators(text: string): GateResult {
    return this.allowlist.applyToValidators(validatorGate(text));
  }

  async check(text: string): Promise<GateDecision> {
    const validators = this.validators(text);
    if (this.model === null) return decide(validators, null, this.threshold);
    const { score, tokens } = await this.model.analyze(text);
    const decision = decide(validators, score, this.threshold);
    const head = this.model.spanHead;
    if (head === null || tokens === null) return decision;
    const { spans } = decodeSpans(text, tokens, head.labels, head.threshold);
    const { kept, allowed } = this.allowlist.filterSpans(text, deterministicUnion(text, spans, validators), validators);
    if (decision.verdict === "BLOCK" && !validators.block && kept.length === 0 && allowed.length > 0) {
      return { ...decision, verdict: "ALLOW", basis: "allowlist", spans: kept };
    }
    return { ...decision, spans: kept };
  }
}

/** A gate without the classifier: the fail-open fallback when the model cannot be loaded. */
export function createValidatorGate(threshold?: number, allowlist?: AllowlistOptions): Gate {
  return new Gate(null, threshold, allowlist);
}

export async function defaultRuntime(): Promise<OrtLike> {
  let loaded: unknown;
  try {
    loaded = await import("onnxruntime-web");
  } catch (error) {
    throw new GateModelError("runtime", "onnxruntime-web is not installed; pass `ort` or add the peer dependency", {
      cause: error,
    });
  }
  const ort = loaded as Partial<OrtLike> & {
    env?: OrtWasmEnv & { wasm?: { numThreads?: number; proxy?: boolean } };
  };
  if (!ort.InferenceSession || !ort.Tensor) {
    throw new GateModelError("runtime", "onnxruntime-web did not export InferenceSession and Tensor");
  }
  applyBundledRuntime(ort.env);
  if (ort.env?.wasm) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
  }
  return ort as OrtLike;
}

/** Downloads and verifies the pinned model files, then builds the gate. Throws `GateModelError` on any failure. */
export async function createGate(options: GateOptions = {}): Promise<Gate> {
  const base = new URL(options.modelBaseUrl ?? DEFAULT_MODEL_BASE_URL, globalThis.location?.href);
  const [ort, modelBytes, vocabBytes] = await Promise.all([
    options.ort ?? defaultRuntime(),
    fetchPinned(new URL("onnx/model_q4.onnx", base), MODEL_FILES["onnx/model_q4.onnx"], options.signal),
    fetchPinned(new URL("vocab.txt", base), MODEL_FILES["vocab.txt"], options.signal),
  ]);
  const model = await createGateModel(ort, modelBytes, new TextDecoder().decode(vocabBytes), {
    executionProviders: options.executionProviders,
  });
  return new Gate(model, options.threshold, options.allowlist);
}
