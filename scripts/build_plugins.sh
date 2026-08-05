#!/usr/bin/env bash
# Builds tensorrt_wan/plugins/csrc/build/libtensorrt_wan_plugins.so.
#
# Requires a CUDA toolkit and a TensorRT SDK install (headers + libs). Not run automatically by
# `pip install` or by this repository's CI (if any) — see PLAN.md's development rule and
# docs/plugins.md. Run this yourself on a machine with the CUDA/TensorRT SDK installed.
#
# Usage:
#   TENSORRT_ROOT=/path/to/TensorRT ./scripts/build_plugins.sh [build_type]
#
# build_type: Release (default) | Debug | RelWithDebInfo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC_DIR="${SCRIPT_DIR}/../tensorrt_wan/plugins/csrc"
BUILD_DIR="${CSRC_DIR}/build"
BUILD_TYPE="${1:-Release}"

if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  echo "error: TENSORRT_ROOT is not set (path to a TensorRT SDK install with include/ and lib/)" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found on PATH — install the CUDA toolkit first" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cmake -S "${CSRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DTENSORRT_ROOT="${TENSORRT_ROOT}"
cmake --build "${BUILD_DIR}" --parallel

echo "Built: ${BUILD_DIR}/libtensorrt_wan_plugins.so"
