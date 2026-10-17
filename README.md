# Rampart Edge

Rampart Edge is a client-side PII gate for AI chat. Before a message leaves the browser it is read twice: by deterministic validators (SSNs, cards, routing numbers, phones, emails, national ids) and by a 1.3 MB 4-bit model running on a 30 KB WebAssembly engine inlined in the package. If either says BLOCK, the message is not sent. The gate answers **BLOCK** or **ALLOW** plus the model's probability, and the same model call marks the PII spans, so the redactor needs no second model.

Built by National Design Studio as a new, smaller version of [Rampart](https://github.com/nationaldesignstudio/rampart), tuned for chat. Everything runs on the user's device; no server is in the loop.

| | Rampart Edge | Rampart |
|---|---:|---:|
| Complete download, Brotli (engine + weights + vocab) | **1.31 MB** [MEASURED] | 14.7 MB + onnxruntime-web |
| PII caught, frozen test (30,863 rows) | **98.19%** | 87.48% |
| Ordinary messages blocked, frozen test | **1.13%** | 11.77% |
| p50 latency in the browser at 128 tokens, WASM, one thread | 39 ms [MEASURED, Chromium, see Runtime] | — |

Frozen-test rows are the shipped model at its shipped threshold, scored once after the model and threshold were fixed. On the dev set it was selected on: 98.24% catch at 0.99% false blocks. See [MODEL_CARD.md](MODEL_CARD.md) for every number and where it comes from.

## Install

```bash
bun add @nationaldesignstudio/rampart-edge
```

One package, no runtime dependency. The model runs on a hand-written Rust→WebAssembly engine (fp32 math on 4-bit weights expanded at load, SIMD128, one thread) that is inlined in `dist/index.js`, so the browser fetches the two model files and nothing else. Upgrading from 2.x: see [`CHANGELOG.md`](CHANGELOG.md).

```ts
import { createGate } from "@nationaldesignstudio/rampart-edge";

const gate = await createGate({ modelBaseUrl: "/models/rampart-edge/" });

const landlord = await gate.check("My landlord Marcus Delgado won't return my security deposit.");
landlord.verdict; // "BLOCK"
landlord.basis;   // "model"
landlord.spans;   // [{ start: 12, end: 26, type: "NAME", probability: 0.99995, source: "model" }]

const park = await gate.check("Is Shenandoah National Park open this weekend?");
park.verdict; // "ALLOW"
park.basis;   // "clear"

const ssn = await gate.check("My SSN is 452-78-1934.");
ssn.basis;    // "validator-and-model"
ssn.spans;    // [{ start: 10, end: 21, type: "GOV_ID", probability: 1, source: "validator", rule: "ssn" }]
```

These are the outputs of the shipped model.

Self-host `model/model.rmpw` and `model/vocab.txt` under `modelBaseUrl`. Both files are checked against the byte count and SHA-256 in [`model/config.json`](model/config.json) before they are used; a mismatch throws `GateModelError("integrity")`. Without `modelBaseUrl` the files come from Hugging Face. If the model cannot load, fall back to `createValidatorGate()` so the deterministic half keeps working (see [`examples/basic-chat.ts`](https://github.com/nationaldesignstudio/rampart-edge/blob/main/examples/basic-chat.ts)).

Inference runs on the calling thread by default. To keep the page responsive, run the gate in a Web Worker: bundle [`examples/gate-worker.ts`](https://github.com/nationaldesignstudio/rampart-edge/blob/main/examples/gate-worker.ts) as the worker entry and create the client with `createWorkerGate(new Worker(url, { type: "module" }), { modelBaseUrl })`. It returns the same `check(text)` contract as `createGate`.

The worker builds its own engine from the inlined wasm; no SharedArrayBuffer, COOP/COEP headers or extra files are involved. With Vite, set `worker: { format: "es" }` for the worker gate.

## Runtime

| | 3.0 (this engine) | 2.x (onnxruntime-web 1.30, trimmed) |
|---|---:|---:|
| Engine, Brotli | 9.2 KB, inlined in `dist/index.js` [MEASURED] | 1,445 KB (`runtime/`) [MEASURED] |
| `model.rmpw` / `model_q4.onnx`, Brotli | 1,165 KB [MEASURED] | 1,172 KB [MEASURED] |
| `vocab.txt`, Brotli | 70 KB [MEASURED] | 70 KB [MEASURED] |
| Complete download, Brotli (`dist/index.js` + weights + vocab) | **1,310 KB** [MEASURED] | **2,746 KB** [MEASURED] |
| Engine load (compile + expand weights) | 10 ms [MEASURED] | 325 ms [MEASURED] |
| p50 `check` at 32 / 128 tokens, one thread | 8.6 / 39 ms [MEASURED] | 18 / 50 ms [MEASURED] |
| Wasm memory after load | 194 pages (12.1 MB) [MEASURED] | — |

Sizes are Brotli quality 11 of the built files. Load and p50 are the bake-off that approved the engine: Playwright Chromium on one Linux x86 VM, the fp32 engine against onnxruntime-web 1.30 on `model_q4.onnx`, forward pass only. The shipped build's own smoke (`bun run smoke:browser`: `dist/index.js` in a plain page and in a Worker, no COOP/COEP, `model.rmpw` over HTTP, p50 of 20 full `gate.check` calls including tokenizer and validators): Chromium 153 — engine load 38 ms (incl. kernel calibration), first `check` 11 ms, p50 7.1 / 28.5 ms at 32 / 128 tokens on the page and 6.3 / 25.6 ms through the Worker; WebKit 26.6 — load 38 ms, first `check` 9 ms, p50 7.0 / 29.0 ms on the page and 7.0 / 29.0 ms through the Worker; 194 wasm pages in both, `crossOriginIsolated` false, identical decisions on both threads [MEASURED].

Parity: the engine is checked in CI against the archived output of onnxruntime-web 1.30 (WASM, one thread, Chromium) on `model_q4.onnx`, the runtime 2.x shipped: max |Δ gate logit| 1.9e-6 on the 12 reference texts, 2.1e-5 on 1,000 dev rows, 3.3e-5 on 354 near-threshold dev rows; identical token argmax on 105,854 tokens; 0 decision flips at the shipped threshold [MEASURED, `__tests__/parity.test.ts`]. Both compute fp32 on the same dequantized weights; nothing was retrained.

The engine has two GEMM microkernels with bit-identical output (`"unrolled"`, fastest on V8; `"compact"`, fastest on JavaScriptCore); `createEngine(wasm, weights, { kernel })` selects one, and the default `"auto"`, which the gate uses, times both at load and keeps the faster (`GateModel.engine.kernel`). Browser support is WebAssembly SIMD128: Chrome 91+, Firefox 89+, Safari 16.4+ [DOC]. If `createGate` throws `GateModelError` (`stage: "runtime"` when WebAssembly SIMD is unavailable, `"download"`, `"integrity"`, `"session"`, `"graph"` or `"inference"`), fall back to `createValidatorGate()`.

## Allowlist

Public names, phone numbers and addresses can pass the gate. The model may block "Write a story about Alexander Hamilton"; when every span it returns is on the allowlist and no validator hard-blocks, the verdict is `ALLOW` with `basis: "allowlist"`. A listed phone number is demoted from a hard validator hit to soft.

```ts
const gate = await createGate({
  allowlist: {
    names: ["Priya Ramanathan"],
    phones: ["1-800-555-0100"],
    addresses: ["300 Spring Garden St, Philadelphia, PA 19123"],
  },
});
```

Host entries add to the built-in lists (`BUILT_IN_ALLOWLIST`: 308 names of US government and historical figures, plus placeholders like John Doe; 1,087 government phone numbers; 2,588 government addresses); pass `builtIn: false` to use only your own. Names match case- and accent-insensitively, and a name introduced as someone's own contact ("my landlord John Smith") is never allowlisted. Addresses match with or without the city, state and ZIP. Entries that can never match (an incomplete phone number, an address without a house number) throw.

## How it works

1. **Validators** (`src/validators/`): checksummed, keyword-cued rules for structured identifiers. Luhn for cards, SSN area/group rules, ABA routing checksum, IBAN mod-97, E.164 phones, Medicare MBI, USCIS numbers, seven non-US national ids. Format examples, role mailboxes and every .gov/.mil email are demoted to soft. Any remaining hard hit blocks.
2. **Model** (`model/`): a 4-layer, 256-wide encoder derived from `microsoft/xtremedistil-l6-h256-uncased`, with one trained layer shared across the four slots, a rank-48 factorized embedding table pruned to 23,910 word pieces, and 4-bit weights. It reads 384-token windows (stride 96) and has two outputs: `gate_logit` (the message's score is the max over windows; BLOCK at probability ≥ 0.417801) and `token_logits`, a typed BIO head over nine span types.
3. **Decision** (`src/scoring.ts`): `validators OR model`. The model is advisory when it is unavailable. Model spans are unioned with validator hits into `decision.spans`, minus allowlisted ones.

What blocks: every complete street or PO address and every complete phone number unless it is on the allowlist, and full names unless the person is on the allowlist. Bare first names and surnames, .gov/.mil emails, role mailboxes and public URLs allow, and so do format examples and bare digits unless a validator confirms them.

## Span redactor

The gate says whether text may go; `decision.spans` says what to remove. `@nationaldesignstudio/rampart-edge/redactor` replaces those spans with `[REDACTED]`, for text extracted from PDFs and for highlighting PII after a BLOCK. It reuses the gate model: `createRedactor({ gate })` downloads nothing, and without `gate` it loads the same two files.

```ts
import { createRedactor } from "@nationaldesignstudio/rampart-edge/redactor";

const redactor = await createRedactor({ gate });
const result = await redactor.redact("Name: Tomas Elling\nPhone: 937-555-0148");
result.text;  // "Name: [REDACTED]\nPhone: [REDACTED]"
result.spans; // [{ start, end, type, probability, source }, ...] with UTF-16 offsets: text.slice(start, end)
```

- Decode: each token takes its most likely label (`span_threshold: null`). Windows of 384 tokens, stride 96; overlapping windows hand over at a token both label O, so no span is cut in two.
- Model spans and validator hits cover their union: validator hits keep their type (`GOV_ID`, `FINANCIAL`, `PHONE`, `EMAIL`), and a model span that overlaps a validator hit keeps only the part outside it. A date next to a birth keyword (`DOB`, `date of birth`, `born`) is a deterministic `DOB_KEYED` span, and a model span that covers part of a date is widened to the whole date.
- Span types:

| type | what it covers |
|---|---|
| `NAME` | a person's full name (bare first names, bare surnames and listed public figures are not spans) |
| `ADDRESS` | a complete residential or mailing address (a bare street, city, state or ZIP is not) |
| `EMAIL` | a personal email address (.gov/.mil and role mailboxes are not) |
| `PHONE` | a complete phone number |
| `GOV_ID` | US government identifiers: SSN, ITIN, EIN, Medicare MBI, A-number, USCIS receipt, passport, driver's license |
| `FINANCIAL` | card, IBAN, bank account and routing numbers |
| `DOB_KEYED` | a date of birth together with the person it belongs to (a bare date is not) |
| `URL_HANDLE` | a personal profile URL or handle |
| `OTHER_ID` | case, claim, customer and medical record numbers tied to a person |

| dataset | leak | over-redaction | relaxed P / R / F1 |
|---|---:|---:|---:|
| Public redacted-source benchmark, held-out partition: 118 PDF documents, 1,857 gold spans | 10.2% | 8.0% | 0.755 / 0.815 / **0.784** |
| Synthetic PDF dev set (`span.pdf-dev.v2`): 268 rendered PDF documents, 1,302 gold spans | 2.8% | 45.6% | 0.536 / 0.963 / **0.689** |

Scored with the shipped runtime (4-bit model, validators, date rules). Leak is the share of gold PII values still visible after redaction; on the benchmark most leaks are `OTHER_ID` (58% leaked), `URL_HANDLE` (62%) and `NAME` (12%). Over-redaction is the share of public terms (agencies, places, hotlines, listed officials) that were redacted; on the PDF set it is agency hotlines (188 of 189) and named public officials (511 of 677). Relaxed F1 counts a predicted span as correct when it overlaps a gold span.

See [`examples/redact-pdf-text.ts`](https://github.com/nationaldesignstudio/rampart-edge/blob/main/examples/redact-pdf-text.ts) for the gate-then-redact flow.

## Evaluate

```bash
bun install
bun run test                  # tokenizer parity, windowing, validators, model vs Python reference, engine vs archived browser ORT output
bun run smoke:browser         # Playwright Chromium/WebKit smoke of dist/ (see tools/browser-smoke/README.md)
```

The false-positive regression set is 137 ordinary messages Rampart blocked plus 29 BLOCK twins. The shipped model blocks 4 of 137 and catches 29 of 29.

## Training data

The model was trained on a corpus built for this project: public government text, public questions, synthetic PII in seven languages, noise, hard negatives and redacted government releases. No chat logs or private conversations were used; every identifier in the corpus is synthetic or already public. This repository holds the runtime and the weights.

## Layout

```
index.ts            public API
src/                gate, model, engine (src/engine.ts + generated src/engine-wasm.ts), tokenizer, scoring, validators
engine/             the Rust→wasm engine crate; `bun scripts/build-engine.ts` rebuilds runtime/rampart-engine.wasm
model/              model.rmpw (RMPW v1 weights), vocab, config; onnx/ holds the ONNX reference export (Git LFS, not shipped)
src/redactor.ts     span redactor on the gate model (`@nationaldesignstudio/rampart-edge/redactor`)
src/worker.ts       Web Worker host + client (`createWorkerGate`)
__tests__/          vitest suites, the Python reference fixture, archived browser-ORT parity references (fixtures/parity/)
tools/browser-smoke Playwright page + Worker smoke of dist/ over plain HTTP
examples/           browser usage
```

## Limitations

- Tuned for English and Spanish government-service chat; French, German, Italian, Portuguese and Dutch are covered by synthetic data only.
- Frozen-test false blocks (1.13%) are higher than dev (0.99%).
- Spans on scanned and released government documents leak more than on chat text: case and record numbers and personal handles are the most missed types.
- The redactor does not apply the gate's allowlist: it marks named public officials and agency hotlines. It does leave .gov/.mil emails.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — files, architecture, training data and evaluation numbers.

## License

Code, weights and documentation: [CC0 1.0](LICENSE). Third-party parts keep their own licenses: `model/vocab.txt` is Google's BERT vocabulary (Apache-2.0).
