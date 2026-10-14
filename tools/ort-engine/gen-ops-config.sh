#!/usr/bin/env bash
# Regenerate an ops config from a model: the operators present in the .onnx on disk
# PLUS every op ORT's graph optimizer (CPU EP) creates at levels basic, extended and
# all. Intermediate levels matter: basic GeluFusion emits an ai.onnx Gelu (opset 20+)
# that level "all" later folds into BiasGelu. Without its kernel ORT inlines Gelu's
# function body and session creation fails on "Sqrt(13)".
# usage: gen-ops-config.sh <model.onnx> [out.config]
set -euo pipefail
ORT_SRC="${ORT_SRC:-$HOME/ort/onnxruntime}"
MODEL="$(realpath "$1")"
OUT="$(realpath -m "${2:-$(dirname "$0")/config/rampart.ops.config}")"
WORK="$(mktemp -d)"
python3 -m pip install -q onnxruntime onnx
python3 - "$MODEL" "$WORK" <<'PY'
import sys, onnxruntime as ort
levels = {
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
for name, level in levels.items():
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.optimized_model_filepath = f"{sys.argv[2]}/optimized_{name}.onnx"
    ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
PY
cp "$MODEL" "$WORK/original.onnx"
python3 "$ORT_SRC/tools/python/create_reduced_build_config.py" "$WORK" "$OUT"
echo "wrote $OUT"; cat "$OUT"
rm -rf "$WORK"
