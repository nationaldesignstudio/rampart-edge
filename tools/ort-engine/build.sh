#!/usr/bin/env bash
# Build onnxruntime-web's plain CPU engine, ort-wasm-simd-threaded.{wasm,mjs},
# from the checkout fetch-ort.sh prepared.
#
#   build.sh baseline <build_dir>              # upstream "Build (simd + threads)" step, unchanged
#   build.sh reduced  <build_dir> <ops.config>  # + operator/type reduction for the Rampart model
#
# Outputs land in <build_dir>/Release/ort-wasm-simd-threaded.{wasm,mjs}.
# Base flags mirror .github/workflows/linux-wasm-ci-build-and-test-workflow.yml
# ("Build (simd + threads)", Release) minus --use_cache/--use_vcpkg: deps come
# from cmake/deps.txt via FetchContent instead of vcpkg.
set -euo pipefail
ORT_SRC="${ORT_SRC:-$HOME/ort/onnxruntime}"
MODE="$1"; BUILD_DIR="$(realpath -m "$2")"
case "$MODE" in
  baseline) EXTRA=() ;;
  reduced)
    CONFIG="$(realpath "$3")"
    EXTRA=(
      --include_ops_by_config "$CONFIG"
      --enable_reduced_operator_type_support
      --disable_ml_ops
      --disable_generation_ops
      --disable_types string float4 float8 optional sparsetensor
    ) ;;
  *) echo "usage: build.sh baseline|reduced <build_dir> [ops.config]" >&2; exit 2 ;;
esac
cd "$ORT_SRC"
python3 tools/ci_build/build.py \
  --parallel \
  --config Release \
  --skip_submodule_sync \
  --build_wasm \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --target onnxruntime_webassembly \
  --skip_tests \
  --disable_rtti \
  --enable_wasm_api_exception_catching \
  --build_dir "$BUILD_DIR" \
  "${EXTRA[@]}"
ls -l "$BUILD_DIR/Release/ort-wasm-simd-threaded."{wasm,mjs}
sha256sum "$BUILD_DIR/Release/ort-wasm-simd-threaded."{wasm,mjs}
