#!/usr/bin/env bash
# Shallow-fetch microsoft/onnxruntime at the exact commit the vendored
# onnxruntime-web was published from, then install the matching emsdk.
#   ORT_COMMIT: from onnxruntime-web's version string, e.g.
#   1.26.0-dev.20260416-b7804b056c -> b7804b056c (prefix). Releases use the tag's
#   commit: v1.30.0 -> f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7 (the default below).
set -euo pipefail
ORT_SRC="${ORT_SRC:-$HOME/ort/onnxruntime}"
ORT_COMMIT="${ORT_COMMIT:-f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7}"
EMSDK_VERSION="${EMSDK_VERSION:-4.0.23}"   # must match tools/ci_build/build_args.py --emsdk_version default

if [ ! -d "$ORT_SRC/.git" ]; then
  mkdir -p "$ORT_SRC" && cd "$ORT_SRC"
  git init -q
  git remote add origin https://github.com/microsoft/onnxruntime.git
fi
cd "$ORT_SRC"
git fetch --depth 1 origin "$ORT_COMMIT"
git checkout -q FETCH_HEAD
git submodule update --init --depth 1 cmake/external/emsdk cmake/external/onnx
echo "onnxruntime VERSION_NUMBER: $(cat VERSION_NUMBER) @ $(git rev-parse HEAD)"

python3 -m pip install -q "cmake>=3.28" ninja flatbuffers onnx
(cd cmake/external/emsdk && ./emsdk install "$EMSDK_VERSION" && ./emsdk activate "$EMSDK_VERSION")
