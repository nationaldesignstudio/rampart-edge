# Reduced-ops ONNX Runtime Web engine for Rampart

Rampart's browser-side PII filter runs on the stock
`onnxruntime-web` CPU engine (`ort-wasm-simd-threaded.{wasm,mjs}`), which
carries every CPU kernel ORT has. The Rampart model uses 24 ONNX operators plus
two `com.microsoft` contrib ops. This directory builds a drop-in replacement
engine from the exact upstream commit the vendored npm package was published
from, with only the kernels `model_q4.onnx` needs, and measures/verifies it.

## Shipped engine: Rampart + Rampart Edge (ORT 1.30.0 @ f2c39fe2, emsdk 4.0.23)

`runtime/ort-wasm-simd-threaded.{wasm,mjs}` in the package is
`build.sh reduced <dir> config/rampart-edge.ops.config`. The config is the union of
both models' ops, including the ops the optimizer creates at levels basic, extended and
all.

| build | raw | gzip -9 | brotli -11 |
|---|---:|---:|---:|
| stock `onnxruntime-web` 1.30.0 | 14,239,897 | 3,659,955 | 2,355,011 |
| reduced-ops (`rampart-edge.ops.config`) | 7,638,929 | 2,209,585 | 1,437,148 |

Parity against stock 1.30.0 (Bun, `threads=1`, optimization `all`):
- Rampart Edge, 411 synthetic texts: gate max |ΔP| = 0, no verdict flips at 0.417801, and span-head argmax agrees on 24,395/24,395 tokens.
- Rampart, 200 synthetic inputs: 12,352/12,352 tokens agree, max |Δlogit| = 0.

Latency is unchanged.

Adding a `!globally_allowed_types` row drops the int4 `GatherBlockQuantized` kernel,
so Edge fails to load with it. Ship the ops-only config.

## Result (ORT 1.26.0 @ b7804b05, emsdk 4.0.23)

### ort-wasm-simd-threaded.wasm

| build | raw | gzip -9 | brotli -11 | Δ raw vs stock | Δ gzip | Δ brotli |
|---|---:|---:|---:|---:|---:|---:|
| stock (npm `1.26.0-dev.20260416-b7804b056c`) | 12,942,611 | 3,321,755 | 2,118,516 | — | — | — |
| baseline (same command, rebuilt here) | 12,946,956 | 3,322,394 | 2,118,912 | +4,345 (0.0%) | +639 | +396 |
| reduced-ops | 6,775,943 | 1,946,084 | 1,260,850 | −6,166,668 (−47.6%) | −1,375,671 (−41.4%) | −857,666 (−40.5%) |
| reduced-ops-types (stretch) | 6,521,753 | 1,908,802 | 1,234,830 | −6,420,858 (−49.6%) | −1,412,953 (−42.5%) | −883,686 (−41.7%) |

### ort-wasm-simd-threaded.mjs

| build | raw | gzip -9 | brotli -11 |
|---|---:|---:|---:|
| stock | 24,180 | 9,090 | 7,998 |
| baseline | 24,180 | 9,090 | 8,001 |
| reduced-ops | 22,961 | 8,654 | 7,639 |
| reduced-ops-types | 22,961 | 8,654 | 7,639 |

Parity (Node, `compare.mjs`, 200 synthetic texts / 4,173 tokens, `threads=1`
and `threads=4`, graph optimization `all`): max |Δlogit| = 0, argmax agreement
4173/4173 for reduced-ops, reduced-ops-types and baseline vs stock. Latency on
an idle 8-core VM, p50 per text: `threads=1` stock 31.6–34.2 ms vs reduced-ops
34.8 ms / reduced-ops-types 32.1 ms; `threads=4` stock 9.9–12.8 ms vs
reduced-ops 10.0–12.6 ms — i.e. within run-to-run noise; no fusion is lost. The baseline `.mjs` differs from the npm one only in two embedded
`EM_ASM` address constants, i.e. the toolchain reproduces the published
artifact.

`reduced-ops-types` adds `!globally_allowed_types;bool,uint8_t,int32_t,int64_t,float,MLFloat16`
(the model's tensor types) and saves a further 37 KB gzip / 26 KB brotli on top
of the ops-only build. It is numerically identical on this model but is the
more fragile option: a future model that introduces e.g. an `int8` or `double`
tensor would fail to load instead of just needing a config regen for a new op.
`reduced-ops` is the recommended shippable artifact.

Browser (Chromium, a production web-app build with the reduced-ops engine in place
of the stock one): engine request 6,775,943 B, the model loads, synthetic SSN and street address are blocked
with zero `/api/` requests, a plain question passes the gate, and a PDF with a
synthetic 555 number is scrubbed to `[PRIVATE_PHONE_1]`. `crossOriginIsolated`
is false on that server, so the run exercised the single-thread fallback path
(`numThreads` capped to 1), same as Safari/iOS without COOP/COEP.

## Layout

| file | purpose |
|---|---|
| `fetch-ort.sh` | shallow-clone `microsoft/onnxruntime` at the pinned commit, init `emsdk` + `onnx` submodules, install emsdk 4.0.23, cmake ≥ 3.28, ninja, flatbuffers |
| `gen-ops-config.sh` | regenerate an ops config from a model: ops in the `.onnx` **plus** ops ORT's optimizer creates at levels basic, extended and all (for Edge: `Gelu`, `BiasGelu`, `FusedMatMul`) |
| `config/rampart-edge.ops.config` | Rampart + Rampart Edge union; builds the engine shipped in `runtime/` |
| `config/rampart.ops.config` | operator allow-list used for the shipped reduced build |
| `config/rampart.ops-types.config` | same + `!globally_allowed_types` row (stretch experiment, see below) |
| `build.sh` | `build.sh baseline <dir>` reproduces upstream's "Build (simd + threads)" step; `build.sh reduced <dir> <config>` adds the reduction flags |
| `measure.mjs` | raw / gzip -9 / brotli -11 table with Δ vs stock for any number of build dirs |
| `compare.mjs`, `texts.mjs` | A/B harness: tokenizes synthetic texts with transformers.js, runs them through two engines via `onnxruntime-web`, reports max \|Δlogit\|, argmax agreement, p50/p95 latency |

## Build

```bash
cd tools/ort-engine
export ORT_SRC=$HOME/ort/onnxruntime          # default
./fetch-ort.sh                                 # ~1 min + emsdk download
./build.sh baseline $HOME/ort/build/baseline   # control, ~25 min on 8 cores
./build.sh reduced  $HOME/ort/build/reduced-ops config/rampart.ops.config
```

Exact `build.py` invocation (both builds):

```
python3 tools/ci_build/build.py --parallel --config Release --skip_submodule_sync \
  --build_wasm --enable_wasm_simd --enable_wasm_threads --target onnxruntime_webassembly \
  --skip_tests --disable_rtti --enable_wasm_api_exception_catching --build_dir <dir>
```

`reduced` appends:

```
--include_ops_by_config config/rampart.ops.config --enable_reduced_operator_type_support \
--disable_ml_ops --disable_generation_ops --disable_types string float4 float8 optional sparsetensor
```

Toolchain used for the numbers above: Python 3.12.13, cmake 4.4.4 (pip),
ninja, emsdk/emscripten 4.0.23 (installed by `fetch-ort.sh`; `build.py`
re-activates it), Node 24, Ubuntu 22.04 x86_64. SIMD and threads stay on so the
output is a drop-in for the stock `ort-wasm-simd-threaded.{wasm,mjs}` filenames
(`wasm.proxy = true` path included). `--disable_exceptions` is deliberately *not* used: ORT
surfaces model/kernel errors as JS exceptions through
`--enable_wasm_api_exception_catching`, and a missing kernel in a reduced build
fails closed at session creation with
`Could not find an implementation for <Op>(<opset>) node` (verified with a
probe model using `ReduceMean`).

## Measure and verify

```bash
bun install            # or npm i — pins onnxruntime-web 1.26.0-dev.20260416-b7804b056c
STOCK=$(dirname $(node -p "require.resolve('onnxruntime-web/package.json')"))/dist
node measure.mjs --stock $STOCK --baseline $HOME/ort/build/baseline/Release \
  --reduced-ops $HOME/ort/build/reduced-ops/Release
node compare.mjs --model model/onnx/model_q4.onnx --tokenizer <dir with tokenizer.json> \
  --a $STOCK --b $HOME/ort/build/reduced-ops/Release --threads 1 --count 200
```

`compare.mjs` runs each engine in its own Node child process (one
`onnxruntime-web` backend per process), feeds identical `input_ids` /
`attention_mask` / `token_type_ids`, and diffs `logits`. Texts in `texts.mjs`
are synthetic only (555 phone numbers, example.com emails, made-up SSNs, names
and addresses, plain service questions in EN/ES/FR) — never real user data.

## Operator set

`gen-ops-config.sh` writes the union of the ops in the model file and the ops
present after ORT's CPU graph optimizer runs at level `all` (what
transformers.js uses by default). For `model_q4.onnx` the optimizer only adds
`BiasGelu`; `Attention`/`SkipLayerNormalization`/`FusedMatMul` fusions do not
fire on this graph (the LayerNorm is already an `ai.onnx.LayerNormalization` op
and the attention pattern is broken up by `MatMulNBits`), so nothing is lost by
omitting them. If a kernel the optimizer wants is missing, the fusion simply
does not apply; if a kernel the *model* needs is missing, session creation
fails with the error above — there is no silent fallback.

The older Rampart model (14.7 MB `model_q4.onnx`) uses a strict subset of this operator set (no `Floor`), so the reduced engine
loads it as well.

## Maintenance

What has to happen when the host app bumps `@huggingface/transformers` /
`onnxruntime-web`:

1. Read the new `onnxruntime-web` version string; the suffix after `-dev.<date>-`
   is the upstream commit. Set `ORT_COMMIT` and rerun `fetch-ort.sh`
   (`EMSDK_VERSION` must match `tools/ci_build/build_args.py` at that commit).
2. Rebuild `reduced` (~15–20 min on 8 cores; the baseline control is optional).
3. Run `compare.mjs` against the stock engine from the new package and expect
   max |Δlogit| = 0 (or ≤1e-6 if MLAS kernels changed).
4. Re-vendor the engine into the host app and update any pinned hashes.

When the *model* changes (new Rampart release): rerun `gen-ops-config.sh` on the
new `.onnx`; if the op set grew, rebuild. A model that needs an op the engine
lacks fails loudly at load time, so a stale engine cannot silently misclassify.

A vendoring check that requires files byte-identical to the npm package's `dist/`
fails against this build by design; pin the engine's sha256 instead.
