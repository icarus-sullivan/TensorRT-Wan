# Pickup notes — TensorRT-Wan

Last session ended: 2026-08-08, ~3:36am. Local repo clean, all committed (`git log -1`:
`9075dc4 Document Refit-API LoRA groundwork and DiT-only strategic direction`).

Strategic direction (see `docs/roadmap.md`'s top note and `docs/known_working/README.md`):
DiT-only TensorRT + real ComfyUI CLIP/VAE/scheduler is the recommended path. Standalone
`WanEngine.generate()` + this project's own VAE/text-encoder engines are deprioritized (still
build fine, orchestration produces noise, root cause never found — not worth more time right now).

Currently mid-investigation: TensorRT Refit API for LoRA support (see
`docs/wan2.2_i2v_14b_notes.md`'s 2026-08-08 "Refit-API LoRA support" entry for full findings —
LoRA key formats/ranks/dimensions, REFIT flag availability, root cause of LoRA currently doing
nothing). Next step when resuming: check whether the DiT ONNX's initializer names match the LoRA
checkpoint's `diffusion_model.blocks.{i}.{submodule}` naming — determines whether refit is a plain
name lookup or needs a mapping table. This didn't finish last session (export got killed
unfinished) — command is in "Resume: same pod" below.

## Resume: same pod (state likely intact)

1. Confirm pod alive, GPU clear:
   ```bash
   ssh -p 44619 -i ~/.ssh/id_ed25519 root@91.199.227.82 "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
   ```
2. Confirm pinned known-working engines survived:
   ```bash
   ssh -p 44619 -i ~/.ssh/id_ed25519 root@91.199.227.82 "ls -la /workspace/runpod-slim/trtwan_known_working_engines/"
   ```
   Should show both `.engine` files (~28.6GB each, read-only) + `.json` sidecars. If empty/gone,
   the persistent volume didn't survive — go to "Cold start: net-new pod" below instead.
3. Re-run the deploy script (idempotent, safe to always run):
   ```bash
   ssh -p 44619 -i ~/.ssh/id_ed25519 root@91.199.227.82 "cd /workspace/runpod-slim/TensorRT-Wan && ./scripts/deploy_comfyui_integration.sh"
   ```
4. Restart ComfyUI server (custom node code only loads at startup — your own pod-restart flow).
5. Pick the Refit investigation back up:
   ```bash
   ssh -p 44619 -i ~/.ssh/id_ed25519 root@91.199.227.82 "cd /workspace/runpod-slim/TensorRT-Wan && \
   CACHE_DIR=/workspace/runpod-slim/trtwan_engines && CKPT_DIR=/workspace/runpod-slim/ComfyUI/models && \
   python3 -m tensorrt_wan.cli.main --cache-dir \$CACHE_DIR export onnx --component dit --loader examples.loaders.wan_comfyui_loader:load_dit \
     --checkpoint \"\$CKPT_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors\" \
     --output \"\$CACHE_DIR/dit_high_noise.onnx\" \
     --exporter-kwargs '{\"in_channels\": 36, \"text_dim\": 4096}'"
   ```
   Then inspect `dit_high_noise.onnx`'s initializer names against the LoRA key convention in
   `docs/wan2.2_i2v_14b_notes.md`'s 2026-08-08 entry.

## Cold start: net-new pod, nothing carried over

0. Get connection details from the new pod's RunPod dashboard (host/port change every time) —
   replace `<PORT>`/`<HOST>` below.

1. Check what actually survived (ComfyUI + checkpoints are usually baked into the `runpod-slim`
   persistent-volume template even on a fresh container — don't assume they're gone too):
   ```bash
   ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> "ls /workspace/runpod-slim/TensorRT-Wan 2>/dev/null; ls /workspace/runpod-slim/trtwan_engines/*.engine 2>/dev/null; ls /workspace/runpod-slim/ComfyUI 2>/dev/null; python3 -c 'import tensorrt' 2>&1"
   ```

2. Sync the repo up fresh:
   ```bash
   rsync -rlptDz --exclude='.git' --exclude='runpod_session_*' --exclude='__pycache__' \
     --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.venv' \
     -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
     /Users/csullivan/Desktop/TensorRT-Wan/ root@<HOST>:/workspace/runpod-slim/TensorRT-Wan/
   ```

3. Install deps (fresh container needs this even if `/workspace` itself persisted):
   ```bash
   ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> "apt-get update && apt-get install -y rsync; cd /workspace/runpod-slim/TensorRT-Wan && pip install -e '.[tensorrt]' transformers pytest"
   ```

4. Build the two DiT engines from scratch (only thing that matters now — VAE/text-encoder are
   skipped, real ComfyUI handles those). ~15-45 min per expert at max optimization level:
   ```bash
   ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> "cd /workspace/runpod-slim/TensorRT-Wan && ./scripts/build_engines.sh dit"
   ```

5. Re-pin the new build as known-working (once verified — don't skip verification just because
   it's a rebuild):
   ```bash
   ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> "mkdir -p /workspace/runpod-slim/trtwan_known_working_engines && \
     DIGEST_HIGH=\$(readlink -f /workspace/runpod-slim/trtwan_model/dit_high_noise.engine | xargs basename -s .engine) && \
     DIGEST_LOW=\$(readlink -f /workspace/runpod-slim/trtwan_model/dit_low_noise.engine | xargs basename -s .engine) && \
     cp /workspace/runpod-slim/trtwan_engines/\$DIGEST_HIGH.{engine,json} /workspace/runpod-slim/trtwan_known_working_engines/ && \
     cp /workspace/runpod-slim/trtwan_engines/\$DIGEST_LOW.{engine,json} /workspace/runpod-slim/trtwan_known_working_engines/ && \
     chmod a-w /workspace/runpod-slim/trtwan_known_working_engines/*.engine"
   ```

6. Deploy to ComfyUI + verify:
   ```bash
   ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> "cd /workspace/runpod-slim/TensorRT-Wan && ./scripts/deploy_comfyui_integration.sh --latest"
   ```
   (`--latest` here since there's no prior pinned copy to fall back to yet — step 5 makes this
   run's build the new pin for *next* time.)

7. Restart ComfyUI, load the `tensorrt_wan_i2v_example` workflow, confirm it looks like
   `docs/known_working/`'s reference (`mean=92.15`, recognizable green chair) before trusting the
   new build.

Steps 2/3/6 are one-time-per-pod, step 4 is the expensive one and the only unavoidable step if the
engine cache truly didn't survive.
