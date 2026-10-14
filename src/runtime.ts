/** onnxruntime-web release the bundled engine was built from. Its JS glue and the wasm must match exactly. */
export const BUNDLED_RUNTIME_ORT_VERSION = "1.30.0";

export interface BundledRuntimePaths {
  readonly wasm: string;
  readonly mjs: string;
}

/** The trimmed ORT Web engine shipped in `runtime/`: only the kernels Rampart and Rampart Edge use. */
export function bundledRuntimePaths(): BundledRuntimePaths {
  return {
    wasm: new URL("../runtime/ort-wasm-simd-threaded.wasm", import.meta.url).href,
    mjs: new URL("../runtime/ort-wasm-simd-threaded.mjs", import.meta.url).href,
  };
}

export interface OrtWasmEnv {
  readonly versions?: { readonly web?: string };
  readonly wasm?: { wasmPaths?: unknown; wasmBinary?: unknown };
}

/**
 * Points onnxruntime-web at the bundled engine. Leaves `env` alone, and returns false, when the host already set
 * `wasmPaths` or `wasmBinary`, or when its onnxruntime-web is not the release the engine was built from.
 */
export function applyBundledRuntime(env: OrtWasmEnv | undefined): boolean {
  const wasm = env?.wasm;
  if (!wasm || wasm.wasmPaths !== undefined || wasm.wasmBinary !== undefined) return false;
  if (env?.versions?.web !== BUNDLED_RUNTIME_ORT_VERSION) return false;
  wasm.wasmPaths = bundledRuntimePaths();
  return true;
}
