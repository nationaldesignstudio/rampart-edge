"""Exporter: model_q4.onnx -> packed little-endian weight file (rampart-edge hand-wasm engine, format v1).
Bytes are copied exactly as stored in the ONNX file (4-bit nibbles + fp32 scales, int4 embedding table, uint8 position/type
tables, fp32 biases/LayerNorm); the engine expands them at load. Layout:
  magic "RMPW" | u32 version=1 | u32 hidden, ffn, heads, layers, vocab, rank, max_pos, labels, n_tensors | 32 B sha256(payload)
  | table n_tensors x (u32 id, u32 dtype{0=u8,1=f32}, u32 nelems, u32 offset, u32 nbytes) | pad to 16 | payload (each tensor 16-B aligned)
"""
import hashlib, json, struct, sys, numpy as np, onnx
from onnx import numpy_helper as nh
src = sys.argv[1] if len(sys.argv) > 1 else "model/onnx/model_q4.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "model/model.rmpw"
m = onnx.load(src); g = m.graph
init = {t.name: t for t in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value": consts[n.output[0]] = nh.to_array(a.t)
def raw(name): return bytes(init[name].raw_data)
def f32(name): return nh.to_array(init[name]).astype("<f4").tobytes()
# sanity on the graph assumptions the engine hard-codes
assert init["m.encoder.embeddings.word_embeddings.table.weight_Q4"].data_type == 22  # INT4 signed
assert list(init["m.encoder.embeddings.word_embeddings.table.weight_Q4"].dims) == [23910, 48]
qc = float(consts["/encoder/encoder/layer.0/attention/self/Constant_6_output_0"].ravel()[0]); kc = float(consts["/encoder/encoder/layer.0/attention/self/Constant_7_output_0"].ravel()[0])
assert abs(consts["/encoder/encoder/layer.0/intermediate/intermediate_act_fn/Constant_output_0"] - 2 ** 0.5) < 1e-6
assert consts["/encoder/encoder/layer.0/attention/self/Constant_9_output_0"].ravel()[0] == -np.inf
for nd in g.node:
    if nd.op_type == "MatMulNBits":
        at = {a.name: onnx.helper.get_attribute_value(a) for a in nd.attribute}; assert at["bits"] == 4 and at["block_size"] == 32, at
    if nd.op_type == "LayerNormalization":
        assert abs(onnx.helper.get_attribute_value(nd.attribute[1]) - 1e-12) < 1e-15 or any(abs(onnx.helper.get_attribute_value(a) - 1e-12) < 1e-15 for a in nd.attribute if a.name == "epsilon")
def scalar(name): return float(nh.to_array(init[name]).ravel()[0])
U8, F32 = 0, 1
T = []  # (id, dtype, bytes)
def add(i, dt, b): T.append((i, dt, b))
add(0, U8, raw("m.encoder.embeddings.word_embeddings.table.weight_Q4")); add(1, F32, f32("m.encoder.embeddings.word_embeddings.table.weight_scales"))
mats = [("onnx::MatMul_541", 2), ("onnx::MatMul_544", 4), ("onnx::MatMul_551", 6), ("onnx::MatMul_552", 8), ("onnx::MatMul_558", 10), ("onnx::MatMul_559", 12), ("onnx::MatMul_560", 14),
        ("m.encoder.pooler.dense.weight_T", 16), ("m.gate_head.weight_T", 18), ("onnx::MatMul_612", 20)]
for base, i in mats: add(i, U8, raw(base + "_Q4")); add(i + 1, F32, f32(base + "_scales"))
add(22, U8, raw("m.encoder.embeddings.position_embeddings.weight_quantized"))
add(23, F32, np.array([scalar("m.encoder.embeddings.position_embeddings.weight_scale"), scalar("m.encoder.embeddings.position_embeddings.weight_zero_point")], "<f4").tobytes())
add(24, U8, raw("m.encoder.embeddings.token_type_embeddings.weight_quantized"))
add(25, F32, np.array([scalar("m.encoder.embeddings.token_type_embeddings.weight_scale"), scalar("m.encoder.embeddings.token_type_embeddings.weight_zero_point")], "<f4").tobytes())
P = "m.encoder.encoder.layer.0."
for i, name in enumerate([P + "attention.self.query.bias", P + "attention.self.key.bias", P + "attention.self.value.bias", P + "attention.output.dense.bias", P + "intermediate.dense.bias", P + "output.dense.bias",
                          "m.encoder.pooler.dense.bias", "m.gate_head.bias", "m.span_head.bias", "m.encoder.embeddings.LayerNorm.weight", "m.encoder.embeddings.LayerNorm.bias",
                          P + "attention.output.LayerNorm.weight", P + "attention.output.LayerNorm.bias", P + "output.LayerNorm.weight", P + "output.LayerNorm.bias"]):
    add(26 + i, F32, f32(name))
add(41, F32, np.array([qc, kc], "<f4").tobytes())
def align(n, a=16): return (n + a - 1) // a * a
payload = bytearray(); table = []
for i, dt, b in T:
    off = align(len(payload)); payload += b"\0" * (off - len(payload)); payload += b
    table.append((i, dt, len(b) // (4 if dt == F32 else 1), off, len(b)))
sha = hashlib.sha256(payload).digest()
head = b"RMPW" + struct.pack("<10I", 1, 256, 1024, 8, 4, 23910, 48, 384, 19, len(T)) + sha
head += b"".join(struct.pack("<5I", *row) for row in table)
head += b"\0" * (align(len(head)) - len(head))
out = head + bytes(payload)
open(dst, "wb").write(out)
json.dump({"bytes": len(out), "header_bytes": len(head), "payload_bytes": len(payload), "sha256_file": hashlib.sha256(out).hexdigest(), "qk_scale": [qc, kc],
           "tensors": [{"id": i, "dtype": dt, "nelems": n, "offset": o, "bytes": b} for i, dt, n, o, b in table]}, open(dst + ".json", "w"), indent=1)
print(f"wrote {dst}: {len(out)} bytes (header {len(head)}, payload {len(payload)}), q/k scale {qc} {kc}")
