#!/usr/bin/env bash
# Wire this repo's comfyui/ custom-node package + example workflow + known-good DiT engines into
# a running ComfyUI install on the RunPod pod. Run this on the pod, from the repo root, any time
# after rsyncing fresh code up (mirrors this repo's comfyui/ package into ComfyUI's custom_nodes/,
# it does not read/write anything back into this repo).
#
# By default this points the example workflow's DiT engines at the pinned known-working copies in
# trtwan_known_working_engines/ (see docs/known_working/), NOT whatever's newest in the engine
# cache -- so loading the example workflow always reproduces the confirmed-coherent result, even
# if you've since rebuilt/experimented with other engines. Pass `--latest` to link the newest
# cache-built dit engines instead (e.g. after running build_engines.sh dit for a new resolution).
#
# Usage: ./scripts/deploy_comfyui_integration.sh [--latest]
set -euo pipefail

COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/runpod-slim/ComfyUI}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/workspace/runpod-slim/trtwan_model}"
KNOWN_WORKING_DIR="${KNOWN_WORKING_DIR:-/workspace/runpod-slim/trtwan_known_working_engines}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-/workspace/runpod-slim/trtwan_model_inputs}"
USE_LATEST="${1:-}"

echo "== 1/4: syncing comfyui/ custom node package =="
mkdir -p "$COMFYUI_ROOT/custom_nodes/tensorrt_wan_comfyui"
rsync -a --delete --exclude="__pycache__" \
  "$REPO_ROOT/comfyui/" "$COMFYUI_ROOT/custom_nodes/tensorrt_wan_comfyui/"

echo "== 2/4: installing example workflow =="
mkdir -p "$COMFYUI_ROOT/user/default/workflows"
cp "$REPO_ROOT/comfyui/examples/tensorrt_wan_i2v_example.json" \
   "$COMFYUI_ROOT/user/default/workflows/tensorrt_wan_i2v_example.json"

echo "== 3/4: copying input images =="
mkdir -p "$COMFYUI_ROOT/input"
for f in "$INPUT_IMAGE_DIR"/close_green_chair_start.png "$INPUT_IMAGE_DIR"/close_green_chair_end.png; do
  [ -f "$f" ] && cp -n "$f" "$COMFYUI_ROOT/input/" || true
done

echo "== 4/4: pointing dit_{high,low}_noise.engine =="
mkdir -p "$MODEL_DIR"
if [ "$USE_LATEST" = "--latest" ]; then
  echo "Using newest cache-built DiT engines (skip -- run build_engines.sh dit's own symlink step, or re-run without --latest to pin to known-working)."
else
  high="$(ls "$KNOWN_WORKING_DIR"/dit_high_noise_*.engine 2>/dev/null | head -1)"
  low="$(ls "$KNOWN_WORKING_DIR"/dit_low_noise_*.engine 2>/dev/null | head -1)"
  if [ -z "$high" ] || [ -z "$low" ]; then
    echo "No pinned known-working engines found in $KNOWN_WORKING_DIR -- run build_engines.sh dit and re-run with --latest, or restore the pinned copies." >&2
    exit 1
  fi
  ln -sf "$high" "$MODEL_DIR/dit_high_noise.engine"
  ln -sf "$low" "$MODEL_DIR/dit_low_noise.engine"
  echo "dit_high_noise.engine -> $high"
  echo "dit_low_noise.engine -> $low"
fi

cat <<EOF

Done. In the ComfyUI UI:
  1. Restart the ComfyUI server (custom node code only loads at startup).
  2. Open the "Workflows" sidebar -> "tensorrt_wan_i2v_example" (or File > Open, browse to
     user/default/workflows/tensorrt_wan_i2v_example.json).
  3. Each TensorRTDiTLoader node's engine_path widget should already point at
     $MODEL_DIR/dit_{high,low}_noise.engine -- if you changed MODEL_DIR, update those widgets.
EOF
