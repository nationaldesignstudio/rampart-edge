---
library_name: onnx
license: cc0-1.0
language:
  - en
  - es
  - fr
  - de
  - it
  - pt
  - nl
tags:
  - pii
  - privacy
  - onnx
  - onnxruntime-web
  - browser
  - client-side
  - token-classification
base_model: microsoft/xtremedistil-l6-h256-uncased
metrics:
  - recall
  - false-positive-rate
---

# Rampart Edge

`rampart-edge` is a 1.45 MB ONNX model that decides whether a chat message contains personally identifiable information before the message leaves the user's device, and marks the PII spans in the same call. It is the model half of **Rampart Edge**, a client-side PII gate released by National Design Studio; the other half is a set of deterministic validators, and the gate blocks when either half says BLOCK.

## Usage

Install `@nationaldesignstudio/rampart-edge` and `onnxruntime-web`, and serve `onnx/model_q4.onnx` and `vocab.txt` from this repo under one URL (here `/models/rampart-edge/`, keeping the `onnx/` subfolder).

```ts
import { createGate } from "@nationaldesignstudio/rampart-edge";

const gate = await createGate({ modelBaseUrl: "/models/rampart-edge/" });

const landlord = await gate.check("My landlord Marcus Delgado won't return my security deposit.");
landlord.verdict; // "BLOCK"
landlord.spans; // [{ start: 12, end: 26, type: "NAME", probability: 0.99995, source: "model" }]

const park = await gate.check("Is Shenandoah National Park open this weekend?");
park.verdict; // "ALLOW"

const ssn = await gate.check("My SSN is 452-78-1934.");
ssn.basis; // "validator-and-model"
```

`createGate` checks both files against the byte counts and SHA-256s below before it uses them. `modelBaseUrl` is resolved against the page URL, so a path like `/models/rampart-edge/` works in a browser only; in Bun or Node pass an absolute `http://` URL to a directory that serves the two files. Without `modelBaseUrl` the files come from this Hugging Face repository.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `onnx/model_q4.onnx` | 1,454,373 | `73fce00c229d8318d7e1f8decccc5ad2dbf7aa45a872364b54d632c38a9eb16c` |
| `vocab.txt` | 173,327 | `ee0684683c7fb15744d0136e671cd7def76ae4d4ddafb72ced864e94c8d32c9a` |

`tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json` describe the same tokenizer for other runtimes. Model + vocab: 1,289,572 bytes gzipped (`gzip -9`, each file separately).

Graph inputs `input_ids`, `attention_mask` (int64, `[1, n]`); outputs `gate_logit` (float32, `[1]`; BLOCK probability = sigmoid) and `token_logits` (float32, `[1, n, 19]`) over the BIO labels in `config.json`.

## Architecture

- Base: `microsoft/xtremedistil-l6-h256-uncased`, truncated from 6 layers to 4, with the first layer's weights shared by all four slots.
- Word embeddings factorized to rank 48 (`vocab × 48` table plus `48 → 256` projection, initialized from the SVD of the pretrained table) and pruned to 23,910 word pieces.
- Two heads on one encoder: pooled `[CLS]` → `gate_logit`; every token → `token_logits` (`O` plus `B-`/`I-` for nine span types).
- Exported to ONNX and quantized with MatMulNBits (4-bit, block size 32).

## Inference

- BERT WordPiece tokenizer, lower-cased, accents stripped (`tokenizer_config.json`).
- Windows of 384 positions (382 content tokens + `[CLS]`/`[SEP]`), stride 96; a message's probability is the max over its windows.
- Operating point `t = 0.417801`: the lowest dev threshold with at most 1% false blocks. Not tuned on any benchmark.
- Spans: each token takes its most likely label (`span_threshold: null`); overlapping windows hand over at a token both label O.
- 32.8 ms p50 / 37.5 ms p95 at 128 tokens, single-thread WASM with SIMD, `onnxruntime-web` 1.30.0, Chromium.
- 26.1 ms p50 / 69.3 ms p95 over 799 OpenPII validation messages (one `check` per message, single-thread WASM, Chromium 156, Linux x86 CPU).

## Span types

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

The runtime unions model spans with validator hits (validators keep their type), adds a `DOB_KEYED` span for a date next to a birth keyword and widens a model span that covers part of a date to the whole date. The probability on a model span is a ranking score, not a calibrated chance.

## Training

- Objective: the smallest browser download subject to ≥ 95% catch at ≤ 1% false blocks on dev; latency under 100 ms is a constraint, not a tie-breaker.
- Data: a from-scratch corpus in seven Latin-script languages: public US government text, public questions, synthetic PII rows, noise, hard negatives, rows aimed at Rampart 1's historical false positives, the training partition of seven public redacted-source sets and synthetic rendered PDF documents. Every identifier is synthetic or already public. Labels are message-level for the gate and span-level for the token head.
- Both heads are trained together, 384/96 windows; the epoch and the threshold were chosen on dev.

## Evaluation

Dev set (32,202 rows), gate = validators OR model, threshold 0.417801:

| metric | value |
|---|---:|
| catch at ≤ 1% false blocks | 98.24% |
| false blocks | 0.99% |
| AUROC | 0.99865 |
| hard ALLOW set (144 rows: company names near amounts, OCR noise, redaction placeholders), false blocks | 19 (13.2%) |

The 19 hard-ALLOW false blocks: 11 company names near contract amounts, 5 redaction placeholders, 3 OCR noise.

Frozen test `test.v1` (30,863 rows, sha256 `5671887f…`), scored once for the shipped model after the threshold was fixed on dev. Nothing was chosen on test.

| system | catch | false blocks |
|---|---:|---:|
| **Rampart Edge (t = 0.417801)** | **98.19%** | **1.13%** |
| Rampart 1 default (all labels) | 87.48% | 11.77% |
| validators alone | 51.70% | 0.05% |

AUROC on `test.v1`: 0.99694.

Historical false-positive regression set (137 ALLOW / 29 BLOCK): 4 ALLOW rows blocked (two tracking numbers next to a government-id word, one public SSA login URL, one format-example passport number); 29 of 29 BLOCK twins caught.

Public redacted-source benchmark (seven FOIA / court release sets, held-out partition): 98.44% catch (441 of 448), 21 of 470 benign pages blocked (4.5%), shipped runtime. 

Public PII systems on OpenPII (`ai4privacy/pii-masking-openpii-1.5m` validation @ `a785eb5`, 30,000-row held-out slice, 28,428 PII / 1,572 ALLOW; message-level, a row is PII if any gold span maps to a type Rampart Edge blocks; baselines at default settings, mapping fixed before scoring):

| system | catch | false blocks | size |
|---|---:|---:|---:|
| **Rampart Edge (validators OR model)** | 89.8% | **9.2%** | **1.45 MB** |
| validators alone | 59.6% | 0.0% | — |
| Rampart 1, all labels on | 99.4% | 48.7% | 14.7 MB |
| Microsoft Presidio + spaCy `en_core_web_lg` | 96.4% | 67.7% | 445 MB |
| GLiNER small v2.1 | 91.1% | 62.0% | 613 MB |
| BERT-small PII (gravitee-io) | 96.6% | 65.5% | 115 MB |
| OpenAI Privacy Filter | 98.0% | 46.1% | 2.83 GB |

Rampart Edge misses full names (49.6% caught when the name is the only PII), street addresses (48.2%) and non-US ids (36–65%); emails 100%, phones 94.3%. OpenPII is not a fully fair test: it labels by different rules than Rampart Edge, tagging non-US national ids and ID-shaped strings Rampart Edge allows and tagging some full names as first names only. OpenPII is synthetic, multilingual and mostly non-US; Rampart 1 was fine-tuned on about 1.4 million OpenPII rows; Rampart Edge was not trained on it. Without the 1,564 id-only rows (non-US national ids are out of scope) catch is 90.2%.

Span redaction, shipped runtime, class-agnostic matching:

| dataset | leak | over-redaction | relaxed P / R / F1 |
|---|---:|---:|---:|
| Public redacted-source benchmark, held-out partition: 118 PDF documents, 1,857 gold spans | 10.2% | 8.0% | 0.755 / 0.815 / **0.784** |
| Synthetic PDF dev set (`span.pdf-dev.v2`): 268 documents, 1,302 gold spans | 2.8% | 45.6% | 0.536 / 0.963 / **0.689** |

Leak is the share of gold PII values still visible after redaction; on the benchmark the leaks are mostly `OTHER_ID` (58%), `URL_HANDLE` (62%) and `NAME` (12%). Over-redaction is the share of public terms (agencies, places, hotlines, listed officials) redacted; on the PDF set it is agency hotlines (188 of 189) and named public officials (511 of 677). The synthetic PDF set was used for model selection; the benchmark was not.

## Intended use and limitations

- Use as a pre-send gate in chat UIs. Pair it with the validators in `@nationaldesignstudio/rampart-edge`; alone, the model catches less structured PII.
- Spans on scanned and released government documents leak more than on chat text; case and record numbers and handles are missed most.
- Strong on English and Spanish government-service chat; other languages are covered by synthetic data only.
- Blocks every complete phone number and street address that is not on the allowlist. The built-in allowlist has 308 names, 1,087 government phone numbers and 2,588 government addresses, and hosts can add their own. Allows bare first names, .gov/.mil emails, role mailboxes and public URLs.
- Not a guarantee. Treat it as a first line of defense.

## Calibration

The gate thresholds the maximum window probability at t = 0.417801, chosen on the dev set as the lowest threshold that keeps false blocks at or under 1%. The probability is a ranking score, not a calibrated chance that a message contains PII: a message at p = 0.4 is not "40% likely" to carry PII. Raise the threshold to trade catch for fewer false blocks; `GateOptions.threshold` overrides the default. The validators are not thresholded; a hard validator hit blocks at any threshold.

## Fairness and limitations

- Language: English and Spanish are covered by public text and public questions; French, German, Italian, Portuguese and Dutch mostly by synthetic rows. Expect more false blocks and misses outside English and Spanish.
- Names: full names block unless they are on the built-in list of US government officials (308 names) or your own allowlist.
- Scope: US identifiers only. Non-US national id numbers are out of scope and are not blocked unless the model generalizes.
- Phones and addresses: a complete phone number or street address blocks unless it is on the allowlist. An agency number or office address missing from the built-in list blocks until the host adds it.
- The historical false-positive set (137 ALLOW / 29 BLOCK) is the only measured check against Rampart 1's historical false positives; other false-positive families may exist.

## Reproducibility

The steps below run from a clone of the [source repository](https://github.com/nationaldesignstudio/rampart-edge) (`bun install`); the Hugging Face files alone hold the weights, tokenizers and this card.

- Runtime: `bun run test` checks the pinned SHA-256s and reruns the reference texts in `__tests__/fixtures/joint-reference.json` against the Python probabilities and spans.
- The frozen test split was scored once for the shipped model and is not rescored for this card.

## Citation

```bibtex
@software{rampartedge_2026,
  author = {{National Design Studio}},
  title = {Rampart Edge: a browser-side PII gate},
  year = {2026},
  url = {https://huggingface.co/nationaldesignstudio/rampart-edge},
  license = {CC0-1.0}
}
```

## License

CC0 1.0. No attribution required.
