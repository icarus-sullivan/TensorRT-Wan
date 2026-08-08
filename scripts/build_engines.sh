#!/usr/bin/env bash
# Build all four TensorRT-Wan engines (text_encoder, vae_encoder, vae_decoder, dit high+low
# noise) and assemble them into a model_dir usable by WanEngine.from_pretrained(). Runs on the
# RunPod pod, from the repo root -- this is the exact recipe documented in
# docs/runpod_setup.md's "The actual build commands" section, just made runnable.
#
# Usage: ./scripts/build_engines.sh [dit|vae|text|all]
#   all (default) -- build everything
#   dit  -- rebuild only dit_high_noise + dit_low_noise (bf16, the usual thing you need to redo)
#   vae  -- rebuild only vae_encoder + vae_decoder (static, T=81 encoder / T=21-latent decoder)
#   text -- rebuild only text_encoder
#
# Override paths via env vars if your pod layout differs:
#   CACHE_DIR, CKPT_DIR, MODEL_DIR, HEIGHT, WIDTH
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/workspace/runpod-slim/trtwan_engines}"
CKPT_DIR="${CKPT_DIR:-/workspace/runpod-slim/ComfyUI/models}"
MODEL_DIR="${MODEL_DIR:-/workspace/runpod-slim/trtwan_model}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
RES="${WIDTH}x${HEIGHT}"
TARGET="${1:-all}"

cd "$(dirname "$0")/.."
mkdir -p "$CACHE_DIR" "$MODEL_DIR"
TRT="python3 -m tensorrt_wan.cli.main --cache-dir $CACHE_DIR"

build_text_encoder() {
  $TRT export onnx --component text_encoder \
    --loader examples.loaders.wan_comfyui_loader:load_text_encoder \
    --checkpoint "$CKPT_DIR/text_encoders/umt5_xxl_fp16.safetensors" \
    --output "$CACHE_DIR/text_encoder.onnx" \
    --exporter-kwargs '{"hidden_dim": 4096}'
  $TRT build engine --component text_encoder --onnx "$CACHE_DIR/text_encoder.onnx" \
    --loader examples.loaders.wan_comfyui_loader:load_text_encoder \
    --checkpoint "$CKPT_DIR/text_encoders/umt5_xxl_fp16.safetensors" \
    --exporter-kwargs '{"hidden_dim": 4096}' --resolutions "$RES" --precision fp16
}

build_vae() {
  # frames=81 (matches _build_image_to_video_conditioning's whole-video encode_video() call --
  # NOT frames=1, that only works with the older per-frame encode_image() path).
  $TRT export onnx --component vae_encoder \
    --loader examples.loaders.wan_comfyui_loader:load_vae_encoder \
    --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
    --output "$CACHE_DIR/vae_encoder.onnx" \
    --exporter-kwargs "{\"latent_channels\": 16, \"frames\": 81, \"height\": $HEIGHT, \"width\": $WIDTH, \"static\": true}"
  $TRT build engine --component vae_encoder --onnx "$CACHE_DIR/vae_encoder.onnx" \
    --loader examples.loaders.wan_comfyui_loader:load_vae_encoder \
    --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
    --exporter-kwargs "{\"latent_channels\": 16, \"frames\": 81, \"height\": $HEIGHT, \"width\": $WIDTH, \"static\": true}" \
    --resolutions "$RES" --precision fp16

  LATENT_H=$((HEIGHT / 8)); LATENT_W=$((WIDTH / 8))
  $TRT export onnx --component vae_decoder \
    --loader examples.loaders.wan_comfyui_loader:load_vae_decoder \
    --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
    --output "$CACHE_DIR/vae_decoder.onnx" \
    --exporter-kwargs "{\"latent_channels\": 16, \"latent_frames\": 21, \"latent_height\": $LATENT_H, \"latent_width\": $LATENT_W, \"static\": true}"
  $TRT build engine --component vae_decoder --onnx "$CACHE_DIR/vae_decoder.onnx" \
    --loader examples.loaders.wan_comfyui_loader:load_vae_decoder \
    --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
    --exporter-kwargs "{\"latent_channels\": 16, \"latent_frames\": 21, \"latent_height\": $LATENT_H, \"latent_width\": $LATENT_W, \"static\": true}" \
    --resolutions "$RES" --precision fp16
}

build_dit() {
  # bf16 is load-bearing (not a perf tuning knob) -- fp16 DiT is 100% NaN at real attention
  # scale. See docs/runpod_setup.md.
  $TRT export onnx --component dit --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    --output "$CACHE_DIR/dit_high_noise.onnx" \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'
  $TRT build engine --component dit --onnx "$CACHE_DIR/dit_high_noise.onnx" \
    --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}' --resolutions "$RES" --precision bf16

  $TRT export onnx --component dit --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    --output "$CACHE_DIR/dit_low_noise.onnx" \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'
  $TRT build engine --component dit --onnx "$CACHE_DIR/dit_low_noise.onnx" \
    --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}' --resolutions "$RES" --precision bf16
}

case "$TARGET" in
  all)  build_text_encoder; build_vae; build_dit ;;
  dit)  build_dit ;;
  vae)  build_vae ;;
  text) build_text_encoder ;;
  *) echo "Unknown target: $TARGET (want: all|dit|vae|text)" >&2; exit 1 ;;
esac

# Assemble/refresh model_dir symlinks from whatever's newest in the cache for each component.
# dit builds twice (high/low) with the same "component": "dit" tag in the sidecar json, so we
# can't just glob by component name -- match by onnx source file instead via mtime of the two
# most recent dit engines.
python3 - "$CACHE_DIR" "$MODEL_DIR" "$TARGET" <<'PYEOF'
import json, sys
from pathlib import Path

cache_dir, model_dir, target = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
model_dir.mkdir(exist_ok=True)

metas = []
for meta_path in cache_dir.glob("*.json"):
    meta = json.loads(meta_path.read_text())
    metas.append((meta_path.stat().st_mtime, meta_path, meta))

if target in ("all", "text"):
    latest = max((m for m in metas if m[2]["component"] == "text_encoder"), default=None)
    if latest:
        link = model_dir / "text_encoder.engine"
        link.unlink(missing_ok=True)
        link.symlink_to(latest[1].with_suffix(".engine"))
        print(f"text_encoder.engine -> {latest[1].with_suffix('.engine').name}")

if target in ("all", "vae"):
    for comp in ("vae_encoder", "vae_decoder"):
        latest = max((m for m in metas if m[2]["component"] == comp), default=None)
        if latest:
            link = model_dir / f"{comp}.engine"
            link.unlink(missing_ok=True)
            link.symlink_to(latest[1].with_suffix(".engine"))
            print(f"{comp}.engine -> {latest[1].with_suffix('.engine').name}")

if target in ("all", "dit"):
    dit_metas = sorted((m for m in metas if m[2]["component"] == "dit"), key=lambda m: -m[0])[:2]
    # Most-recently-built dit engine is the one from the *last* build_dit() call above
    # (low_noise); second-most-recent is high_noise. Order in the shell function is fixed.
    if len(dit_metas) >= 2:
        high_meta, low_meta = dit_metas[1], dit_metas[0]
        for name, meta in (("dit_high_noise", high_meta), ("dit_low_noise", low_meta)):
            link = model_dir / f"{name}.engine"
            link.unlink(missing_ok=True)
            link.symlink_to(meta[1].with_suffix(".engine"))
            print(f"{name}.engine -> {meta[1].with_suffix('.engine').name}")
PYEOF

WAN_MODEL_JSON="$MODEL_DIR/wan_model.json"
if [ ! -f "$WAN_MODEL_JSON" ]; then
  cat > "$WAN_MODEL_JSON" <<JSONEOF
{
  "latent_channels": 16, "vae_temporal_scale": 4, "vae_spatial_scale": 8,
  "text_embed_dim": 4096, "tokenizer_name": "google/umt5-xxl",
  "default_num_frames": 81, "default_resolution": [$HEIGHT, $WIDTH], "fps": 16, "max_text_tokens": 512
}
JSONEOF
  echo "wrote $WAN_MODEL_JSON"
fi

echo "Done. model_dir: $MODEL_DIR"
ls -la "$MODEL_DIR"
