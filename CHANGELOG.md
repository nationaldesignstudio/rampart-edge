# Changelog

## 3.0.0

Rampart Edge no longer depends on onnxruntime-web. The 4-bit model ships as `model.rmpw` and runs on a hand-written
Rust→WebAssembly engine (fp32 math on dequantized weights, SIMD128, no threads, no SharedArrayBuffer, no cross-origin
isolation) that is inlined in the package, so a host downloads two files instead of four and about half as many bytes.

| | 2.0.5 | 3.0.0 |
|---|---:|---:|
| Package bundle `dist/index.js` (Brotli) | 57.5 KB | 73.1 KB (includes the 27,260-byte engine wasm, inlined) [MEASURED] |
| Engine (Brotli) | onnxruntime-web 1.30, trimmed: 1,445 KB | 8.6 KB, inside the bundle [MEASURED] |
| Weights (Brotli) | `onnx/model_q4.onnx` 1,172 KB | `model.rmpw` 1,165 KB [MEASURED] |
| Vocabulary (Brotli) | 70.4 KB | 70.4 KB [MEASURED] |
| Complete download (Brotli) | **2,746 KB** | **1,309 KB** [MEASURED] |

Sizes are `zlib.brotliCompressSync` at quality 11 of the built files on this branch and of the 2.0.5 build.

Parity with the previous runtime: on the archived output of onnxruntime-web 1.30 (WASM, one thread, Chromium 153) for
`model_q4.onnx`, the engine matches to max |Δ gate logit| 1.9e-6 (fixture, 14 windows), 2.1e-5 (1,000 dev rows, 1,025
windows), 3.3e-5 (354 near-threshold dev rows, 379 windows); identical token argmax on all 105,854 tokens; 0 decision
flips at the shipped threshold [MEASURED, `bun run test` → `__tests__/parity.test.ts`]. The two runtimes compute the
same fp32 math on the same dequantized weights; nothing was retrained, requantized or re-thresholded.

Browser smoke of this build (`bun run smoke:browser`: `dist/index.js` in a plain page and in a Worker over HTTP, no
COOP/COEP, `model.rmpw` fetched over HTTP, p50 of 20 full `gate.check` calls) [MEASURED]:

| | Chromium 153 | WebKit 26.6 |
|---|---:|---:|
| `createGate` (fetch 1.5 MB from localhost + verify pins + engine load) | 68 ms | 74 ms |
| engine load (`GateModel.engine.loadMs`) | 23 ms | 14 ms |
| first `check` | 10 ms | 13 ms |
| p50 `check`, 32 / 128 tokens, main thread | 6.3 / 25.9 ms | 10 / 41 ms |
| p50 `check`, 32 / 128 tokens, through the Worker | 8.1 / 31.3 ms | 10 / 43 ms |
| wasm memory after load | 194 pages (12.1 MB) | 194 pages |
| `crossOriginIsolated` / `SharedArrayBuffer` | false / absent | false / absent |

### Migration from 2.x

- **Install**: `bun add @nationaldesignstudio/rampart-edge`. Remove `onnxruntime-web`; it is no longer a peer dependency
  and `createGate` never imports it.
- **Serve two files** under `modelBaseUrl`: `model.rmpw` and `vocab.txt` (both in the package's `model/`). Delete
  `onnx/model_q4.onnx` and the `onnx/` folder from your asset directory, and delete the self-hosted `runtime/` /
  `ort-wasm-simd-threaded.*` files. The pins in `model/config.json#files` are now keyed `model.rmpw` and `vocab.txt`
  (`MODEL_FILES`). The Hugging Face repository behind `DEFAULT_MODEL_BASE_URL` must carry `model.rmpw` and the new
  `config.json` before 3.0.0 is published.
- **`GateOptions`**: `ort` and `executionProviders` are gone. New optional `engineWasm?: BufferSource | WebAssembly.Module`
  when you want to serve the engine yourself instead of using the inlined copy (`runtime/rampart-engine.wasm` in the
  package is the same bytes). `RedactorOptions` changed the same way. `WorkerGateOptions` no longer has `ort` to omit.
- **Removed**: `defaultRuntime`, `applyBundledRuntime`, `bundledRuntimePaths`, `BUNDLED_RUNTIME_ORT_VERSION`,
  `BundledRuntimePaths`, `OrtWasmEnv`, `OrtLike`, `OrtSession`, `OrtTensor`, `GateModelOptions.executionProviders`,
  `src/runtime.ts`, `runtime/ort-wasm-simd-threaded.{mjs,wasm}`, `tools/ort-engine/`. `model/onnx/model_q4.onnx`
  stays in the repository as the parity reference (`model/onnx/README.md`) but is not in the npm package; neither are
  `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` (still in the repository and on Hugging Face).
- **Added**: `createEngine`, `EngineError`, `parseWeightHeader`, `engineWasmBytes`, `ENGINE_WASM_BASE64`, types
  `GateEngine`, `EngineOutput`, `WeightHeader`; `fetchGateModel(options)`; `createGateModelFromWeights({ weights, vocab,
  wasm?, config? })`; `fromEngineError`; `MODEL_ENGINE` (`config.engine`: `{ format: "rmpw", version: 1, wasm_sha256,
  wasm_bytes }`); `GateModel.engine` (`loadMs`, `memoryPages`).
- **`createGateModel(engine, vocab, options?)`** takes a `GateEngine` instead of `(ort, modelBytes, vocab, options)`
  and is now synchronous (it still works under `await`). `GateModel.analyze` / `score` are unchanged.
- **`GateModelError.stage`** keeps the same six values. Engine failures arrive with the `EngineError` as `cause`:
  `wasm → runtime`, `format → graph`, `integrity → integrity`, `init → session`, `range`/`numeric → inference`. The
  fail-open pattern (`catch GateModelError → createValidatorGate()`) is unchanged.
- **Vite**: `optimizeDeps.exclude` for this package is no longer needed (nothing is served from `runtime/`);
  `worker: { format: "es" }` still applies to the worker gate.
- **Browser support** is WebAssembly SIMD128: Chrome 91+, Firefox 89+, Safari 16.4+ [DOC]. 2.x needed the same for
  onnxruntime-web's SIMD build.

### Also in this release

- `engine/`: the Rust crate (`no_std`, SIMD128), `scripts/build-engine.ts` (reproducible build, byte-identical check,
  48 KB budget), `.github/workflows/engine.yml`.
- `__tests__/parity.test.ts` with `__tests__/fixtures/parity/{fixture,dev-1000,near-threshold}.json`: token ids and
  logits archived from browser onnxruntime-web (no text), gated at max |Δ gate| ≤ 1e-4, 100% argmax, 0 flips.
- `__tests__/package.test.ts`: inlined wasm = `runtime/rampart-engine.wasm` = `config.engine` pin; file pins = files.
- `.github/workflows/ci.yml`: `bun install`, `bun run test`, `type-check`, `build`.
- `tools/browser-smoke/`: Playwright page + Worker smoke of `dist/` over plain HTTP (no COOP/COEP).

## 2.0.5 and earlier

See the Git history.
