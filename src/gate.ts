import config from "../model/config.json";
import { compileAllowlist, type AllowlistOptions, type CompiledAllowlist } from "./allowlist.js";
import { createGateModelFromWeights, fetchPinned, type GateModel, type PinnedFile } from "./model.js";
import { assertThreshold, decide, type GateDecision } from "./scoring.js";
import { decodeSpans, deterministicUnion } from "./spans.js";
import { gate as validatorGate, type GateResult } from "./validators/index.js";

export const MODEL_ID = config.model_id;
/** Byte count and SHA-256 of the two files `createGate` downloads. */
export const MODEL_FILES: Readonly<Record<"model.rmpw" | "vocab.txt", PinnedFile>> = config.files;
/** The inlined engine: weight format, and the pin of the wasm `engineWasmBytes()` decodes to. */
export const MODEL_ENGINE = config.engine;
/** Where `createGate` fetches the model from unless `modelBaseUrl` points at your own origin. */
export const DEFAULT_MODEL_BASE_URL = `https://huggingface.co/${config.model_id}/resolve/main/`;

export interface GateOptions {
  /** Directory that serves `model.rmpw` and `vocab.txt`. Self-host it for production. */
  readonly modelBaseUrl?: string | URL;
  /** The engine wasm, when you serve it yourself instead of using the copy inlined in this package. */
  readonly engineWasm?: BufferSource | WebAssembly.Module;
  /** BLOCK when the model probability is at least this. Defaults to the published operating point. */
  readonly threshold?: number;
  /** Names, phone numbers and addresses that may pass. Adds to the built-in lists unless `builtIn: false`. */
  readonly allowlist?: AllowlistOptions;
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

/** Downloads and verifies `model.rmpw` and `vocab.txt` under `modelBaseUrl`, then builds the engine and model. */
export async function fetchGateModel(options: Pick<GateOptions, "modelBaseUrl" | "engineWasm" | "signal"> = {}): Promise<GateModel> {
  const base = new URL(options.modelBaseUrl ?? DEFAULT_MODEL_BASE_URL, globalThis.location?.href);
  const [weights, vocabBytes] = await Promise.all([
    fetchPinned(new URL("model.rmpw", base), MODEL_FILES["model.rmpw"], options.signal),
    fetchPinned(new URL("vocab.txt", base), MODEL_FILES["vocab.txt"], options.signal),
  ]);
  return createGateModelFromWeights({ weights, vocab: new TextDecoder().decode(vocabBytes), wasm: options.engineWasm });
}

/** Downloads and verifies the pinned model files, then builds the gate. Throws `GateModelError` on any failure. */
export async function createGate(options: GateOptions = {}): Promise<Gate> {
  const model = await fetchGateModel(options);
  return new Gate(model, options.threshold, options.allowlist);
}
