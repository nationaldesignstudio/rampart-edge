/**
 * @nationaldesignstudio/rampart-edge
 *
 * Client-side PII gate for AI chat. Before a message leaves the browser it is checked twice: by
 * deterministic validators (SSNs, cards, routing numbers, phones, emails, ...) and by a 1.45 MB
 * ONNX model. Either one saying BLOCK blocks the message. The same model call marks the PII spans
 * (`decision.spans`); `/redactor` replaces them with [REDACTED].
 */

export { BUILT_IN_ALLOWLIST, compileAllowlist, CompiledAllowlist, type Allowlist, type AllowlistOptions } from "./src/allowlist.js";
export { createGate, createValidatorGate, Gate, DEFAULT_MODEL_BASE_URL, MODEL_FILES, MODEL_ID, type GateOptions } from "./src/gate.js";
export {
  createGateModel,
  fetchPinned,
  verifyPinned,
  sha256Hex,
  GateModelError,
  MODEL_INPUTS,
  MODEL_OUTPUT,
  MODEL_SPAN_HEAD,
  readSpanHead,
  type GateModel,
  type GateModelOptions,
  type ModelAnalysis,
  type ModelConfig,
  type SpanHead,
  type OrtLike,
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
export {
  applyBundledRuntime,
  bundledRuntimePaths,
  BUNDLED_RUNTIME_ORT_VERSION,
  type BundledRuntimePaths,
  type OrtWasmEnv,
} from "./src/runtime.js";
export { createWorkerGate, registerGateWorker, WorkerGate, type WorkerGateOptions, type WorkerPort } from "./src/worker.js";
export type { RedactedSpan, RedactedSpanType, SpanType } from "./src/spans.js";
export { WordPieceTokenizer, SPECIAL_TOKENS } from "./src/tokenizer.js";
export { detect, detectAll, gate as validatorGate, type Detection, type GateResult, type Label } from "./src/validators/index.js";
