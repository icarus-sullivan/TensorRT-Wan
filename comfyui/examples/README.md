# `tensorrt_wan_i2v_example.json`

A real ComfyUI workflow graph reproducing this project's `docs/known_working/` result: real
`CLIPLoader`/`VAELoader`/`CLIPTextEncode`/`LoadImage`/`VAEDecode` (stock ComfyUI), real
`WanFMLFPluggable`/`SpnxxMultiKSampler` (the `custom_nodes/spnxx` package), with **only** the two
DiT experts loaded through this project's own `TensorRTDiTLoader` node
(`comfyui/nodes/dit_loader.py`) instead of a stock `UNETLoader`.

**Not yet verified by loading in the ComfyUI UI directly** — built programmatically from each real
node's `INPUT_TYPES`/schema (confirmed by reading source: `nodes.py`'s `CLIPLoader`/`VAELoader`/
`CLIPTextEncode`/`LoadImage`/`VAEDecode`/`SaveAnimatedWEBP`, `custom_nodes/spnxx`'s
`WanFMLFPluggable`/`SpnxxMultiKSampler`, and this package's own `TensorRTDiTLoader`), matching the
exact parameters that produced the coherent result documented in `docs/known_working/`. If
anything doesn't load cleanly, check node ID/link consistency first — this JSON was assembled by
a small script (not hand-edited), so a mismatch would be systematic, not a typo.

## Before running

1. **Requires `custom_nodes/spnxx`** (`WanFMLFPluggable`, `SpnxxMultiKSampler`) already installed
   in this ComfyUI instance.
2. **Requires this project's `comfyui/` package installed as a custom node** (e.g.
   `custom_nodes/tensorrt_wan_comfyui/`, pointing at this repo) for `TensorRTDiTLoader` to be
   registered.
3. **Put the input images in ComfyUI's own `input/` directory** — `LoadImage` nodes only see
   files there, not arbitrary paths. Default widget values reference
   `close_green_chair_start.png`/`close_green_chair_end.png` (`runpod_session_2026-08-06/`
   locally) — copy them in, or repoint the `LoadImage` nodes at your own images.
4. **Update the two `TensorRTDiTLoader` nodes' `engine_path` widgets** if your built engines live
   somewhere other than `/workspace/runpod-slim/trtwan_model/dit_{high,low}_noise.engine`.
5. **`unet_name`** on each `TensorRTDiTLoader` must be a real checkpoint filename ComfyUI's
   `folder_paths` can find under `diffusion_models/` — it's only used to derive the correct
   `model_sampling`/`latent_format` shell config, the real weights are discarded and replaced by
   the TensorRT engine.

## What it does

`CLIPLoader` (real UMT5) + `VAELoader` (real Wan VAE) + two `TensorRTDiTLoader`s (our TensorRT
DiT, high/low noise experts) feed `CLIPTextEncode` (positive/negative prompts) and
`WanFMLFPluggable` (real first/last-frame conditioning construction, gray-fill + single whole-video
VAE encode) into `SpnxxMultiKSampler` (real two-phase MoE sampler: `steps=12`, `high_cfg=1.8`,
`low_cfg=1.1`, 50/50 step split, `sampler=euler`, `scheduler=sgm_uniform`), then `VAEDecode` +
`SaveAnimatedWEBP` for output. Resolution is 832x480 (landscape) — the DiT's tuned/confirmed-working
shape; portrait (480x832) hits a real, separate, unresolved TensorRT shape bug (see
`docs/wan2.2_i2v_14b_notes.md`).
