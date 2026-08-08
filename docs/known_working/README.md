# Known-working reference: TensorRT DiT inside the real ComfyUI pipeline

**Date:** 2026-08-08. **Status:** confirmed coherent, fully working.

This is the first (and so far only) fully coherent I2V generation using this project's TensorRT
DiT engine. It resolved a full session's investigation into "generate() produces pure noise" —
see `docs/wan2.2_i2v_14b_notes.md`'s "BREAKTHROUGH" entry for the complete story.

## What this proves

The TensorRT DiT engine (both `high_noise`/`low_noise` experts, `bf16`, the NaN fix from earlier
this session) is **numerically correct**. Every other bug found this session (mask polarity,
missing latent normalization, all-zero CFG embedding, per-frame VAE encoding) was real but was
never the actual cause of the noise — the actual cause was this project's own scheduler/CFG-loop
reimplementation, not yet identified precisely. Swapping in ComfyUI's *real* CLIP, VAE,
conditioning construction (`WanFMLFPluggable`), and sampler (`two_phase_sampler`,
`sampler=euler`, `scheduler=sgm_uniform`) — keeping *only* the DiT as our TensorRT engine —
produced this coherent result immediately.

## Exact reproduction recipe

- Script: `real_pipeline_trt_dit_test_480x832_WORKING.py` (copy of
  `scripts/real_pipeline_trt_dit_test.py` at the moment this ran successfully — the live script
  may have since changed for further experiments, e.g. adding width/height args).
- Resolution: **832x480 (landscape)** — this is the DiT's tuned opt point. Portrait (480x832) was
  tried immediately after and hit a real, different, unresolved bug (internal TensorRT shape
  mismatch, "30 != 52" — see notes doc) — not yet fixed, do not assume portrait works.
- Checkpoints: `wan2.2_i2v_high_noise_14B_fp16.safetensors` / `..._low_noise_...` (shells, for
  `model_sampling`/`latent_format`/`concat_keys` config only — real weights discarded, swapped for
  our TensorRT engine), `wan2.2_i2v_high_noise` TensorRT engines at
  `/workspace/runpod-slim/trtwan_model/dit_{high,low}_noise.engine` (the dynamic-H/W-range build
  from this session, `bf16`).
- Text encoder: real ComfyUI CLIP, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `CLIPType.WAN`.
- VAE: real ComfyUI VAE, `wan_2.1_vae.safetensors`.
- Conditioning: real `custom_nodes/spnxx/nodes/wan_fmlf_pluggable.py`'s `WanFMLFPluggable`,
  **no LoRA, no CLIP vision** (deliberately dropped to isolate the DiT-correctness question — our
  exported DiT graph has no `clip_fea` input at all, and no LoRA was ever applied to our engine).
- Sampler: real `custom_nodes/spnxx/sampler/two_phase_sampler.py`, `steps=12`,
  `high_cfg=1.8, low_cfg=1.1`, 50/50 step split, `sampler_name="euler"`, `scheduler="sgm_uniform"`.
- Prompt: `"a green chair"`. Images: `close_green_chair_start.png`/`_end.png` (`comfyui/examples/`).

## Real integration details required to make this work

See the script itself for full comments, summarized:
1. `custom_nodes/spnxx/__init__.py` needs a minimal `server.PromptServer.instance` mock (it isn't
   actually running a server) to import without crashing.
2. `WAN22.concat_cond` introspects `diffusion_model.patch_embedding.weight.shape[1]` — the
   TensorRT wrapper needs *some* module there with the right weight shape (36 channels), unused
   in `forward()`.
3. **ComfyUI batches cond+uncond into one `batch=2` call by default; our exported DiT graph has
   `batch` specialized to 1** (`torch.export` traced it that way). The wrapper splits any
   `batch>1` call into `batch=1` calls and re-concatenates.
4. Our `context` input has no dynamic axis (fixed `max_text_tokens=512`) — pad/truncate
   defensively since ComfyUI's tokenizer doesn't necessarily match that length.
5. `vae.decode()`'s internal in-place ops fail on our TensorRT-produced tensors ("inference
   tensor" outside inference mode) — monkeypatch `vae.process_output` to clone first.

## Result

`frames: shape=(81, 480, 832, 3) mean=92.15, per-frame std=~58-65` — real image statistics.
Visually confirmed coherent (a recognizable green chair matching the source image) across first,
middle, and last frames.

## Pinned engine copies (do not rely on the mutable cache/symlinks for this)

`trtwan_engines/`'s content-addressed cache and `trtwan_model/*.engine`'s symlinks are both
mutable — confirmed the hard way same day (2026-08-08): a `vae_encoder.engine` symlink silently
pointed at a stale, incompatible build after an unrelated exporter change, with no error until
runtime OOM. To make sure this exact known-good result can always be reproduced regardless of
what else gets built later, the two DiT engines that produced it are copied (not symlinked),
read-only, to a name that nothing else will ever write to:

```
/workspace/runpod-slim/trtwan_known_working_engines/dit_high_noise_c56f29a277b8a35a_480x832_bf16.engine
/workspace/runpod-slim/trtwan_known_working_engines/dit_low_noise_7d16ae577fe5bc92_480x832_bf16.engine
```

Sidecar `.json` (component/model_hash/tensorrt_version/cuda_version/gpu_architecture/
optimization_profile/precision/input_shape_digest) copied alongside each, unchanged from the
original cache entry:

- high_noise: `model_hash=c21c21efa368d529`, `input_shape_digest=b8f95a1c8b28`, `precision=bf16`, `optimization_profile=480x832`
- low_noise: `model_hash=edb89340c8a6fbf1`, `input_shape_digest=b8f95a1c8b28`, `precision=bf16`, `optimization_profile=480x832`

`scripts/deploy_comfyui_integration.sh` (no `--latest` flag) always relinks
`trtwan_model/dit_{high,low}_noise.engine` to these pinned files before pointing the example
workflow at them — this is the safe default. Only pass `--latest` if you deliberately want to
test a freshly-built DiT engine instead.

Note: `vae_encoder.engine`/`vae_decoder.engine`/`text_encoder.engine` are **not** part of this
known-good reference — the real ComfyUI pipeline uses its own CLIP/VAE, not this project's
engines, for everything except the DiT. Those three engines only matter for the standalone
`WanEngine.generate()` path, which is still producing noise (see wan2.2_i2v_14b_notes.md) and is
not currently the recommended path.

## Next step (in progress at time of writing)

Formalize this as a real ComfyUI custom node (`comfyui/nodes/`) outputting a standard `MODEL`
socket, so it drops into a real ComfyUI workflow graph (like `wan-slim-example.json`) instead of
needing a standalone script. This project's existing `comfyui/` custom node package
(`sampler.py`, `scheduler.py`, `vae_encoder.py`, `vae_decoder.py`, `text_encoder.py`) is a fully
separate, custom-typed (`TRTWAN_*` sockets) reimplementation of the whole pipeline — exactly the
kind of code that caused every bug this session. The new direction: only the DiT needs to be
TensorRT-accelerated (it dominates total generation cost; VAE/text-encoder are comparatively
cheap), so the new node wraps *only* the DiT and lets real ComfyUI handle everything else.
