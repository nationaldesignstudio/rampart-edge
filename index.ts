/**
 * @nationaldesignstudio/rampart-edge
 *
 * Client-side PII gate for AI chat. Before a message leaves the browser it is checked twice: by
 * deterministic validators (SSNs, cards, routing numbers, phones, emails, ...) and by a 1.3 MB 4-bit
 * model run by a 27 KB wasm engine inlined in this package. Either one saying BLOCK blocks the message.
 * The same model call marks the PII spans (`decision.spans`); `/redactor` replaces them with [REDACTED].
 */

export { BUILT_IN_ALLOWLIST, compileAllowlist, CompiledAllowlist, type Allowlist, type AllowlistOptions } from "./src/allowlist.js";
export {
  createEngine,
  EngineError,
  ENGINE_WASM_BASE64,
  engineWasmBytes,
  parseWeightHeader,
  type EngineOutput,
  type GateEngine,
  type WeightHeader,
} from "./src/engine.js";
export {
  createGate,
  createValidatorGate,
  fetchGateModel,
  Gate,
  DEFAULT_MODEL_BASE_URL,
  MODEL_ENGINE,
  MODEL_FILES,
  MODEL_ID,
  type GateOptions,
} from "./src/gate.js";
export {
  createGateModel,
  createGateModelFromWeights,
  fetchPinned,
  fromEngineError,
  verifyPinned,
  sha256Hex,
  GateModelError,
  MODEL_INPUTS,
  MODEL_OUTPUT,
  MODEL_SPAN_HEAD,
  readSpanHead,
  type GateModel,
  type GateModelOptions,
  type GateModelSource,
  type ModelAnalysis,
  type ModelConfig,
  type SpanHead,
  type PinnedFile,
} from "./src/model.js";
export {
  assertThreshold,
  BLOCK_THRESHOLD,
  CONTENT_TOKENS,
  MAX_TOKENS,
  STRIDE,
  combineWindows,
  contentWindows,
  decide,
  modelBlocks,
  sigmoid,
  withSpecialTokens,
  type DecisionBasis,
  type GateDecision,
  type ModelScore,
  type Verdict,
} from "./src/scoring.js";
export { createWorkerGate, registerGateWorker, WorkerGate, type WorkerGateOptions, type WorkerPort } from "./src/worker.js";
export type { RedactedSpan, RedactedSpanType, SpanType } from "./src/spans.js";
export { WordPieceTokenizer, SPECIAL_TOKENS } from "./src/tokenizer.js";
export { detect, detectAll, gate as validatorGate, type Detection, type GateResult, type Label } from "./src/validators/index.js";
