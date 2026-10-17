# `model_q4.onnx` — reference artifact (not shipped)

`model_q4.onnx` (1,454,373 bytes, sha256 `73fce00c…`, Git LFS) is the ONNX export of the shipped model: xtremedistil
h256, one shared layer applied four times, rank-48 int4 embedding table, MatMulNBits 4-bit block-32 encoder, gate head
and 19-label BIO span head. Since 3.0.0 the package does not load it; `model/model.rmpw` holds the same weights in the
`RMPW` v1 format the Rust→wasm engine reads, and `config.json` records the ONNX pin as `architecture.source_onnx_sha256`.

It stays in the repository, at this path, because it is the canonical reference:

- **Parity.** The engine's acceptance criterion is browser onnxruntime-web 1.30 (WASM, one thread) output on this file
  (`__tests__/fixtures/parity/*.json`, `__tests__/fixtures/engine-reference.json`). Regenerating those references
  (rampart-2 `tools/model/edge-parity/`) needs the file at `model/onnx/model_q4.onnx`.
- **Export.** `engine/export/export_weights.py` reads this file and must reproduce `model/model.rmpw` byte for byte.

It is excluded from the npm package by `package.json#files`.
