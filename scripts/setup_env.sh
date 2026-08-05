#!/usr/bin/env bash
# Creates a venv at .venv and installs TensorRT-Wan into it. Not run automatically — see
# docs/installation.md. TensorRT itself (the `tensorrt` package) requires a matching CUDA install
# and is only pulled in if --tensorrt is passed, since this script must also work on a machine
# with no GPU (e.g. for `pip install -e .` + `trtwan gpu-report` only).
#
# Usage:
#   ./scripts/setup_env.sh [--tensorrt] [--dev]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VENV_DIR="${REPO_ROOT}/.venv"

extras=()
for arg in "$@"; do
  case "${arg}" in
    --tensorrt) extras+=("tensorrt") ;;
    --dev) extras+=("dev") ;;
    --comfyui) extras+=("comfyui") ;;
    *) echo "unknown flag: ${arg}" >&2; exit 1 ;;
  esac
done

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

if [[ ${#extras[@]} -eq 0 ]]; then
  pip install -e "${REPO_ROOT}"
else
  extras_joined=$(IFS=,; echo "${extras[*]}")
  pip install -e "${REPO_ROOT}[${extras_joined}]"
fi

echo "Environment ready at ${VENV_DIR}. Activate with: source ${VENV_DIR}/bin/activate"
