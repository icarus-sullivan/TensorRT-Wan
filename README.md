# TensorRT-RT

Two self-contained, drag-and-droppable ComfyUI custom nodes:

- **`comfyui-wanrt/nodes/vae_rt.py`** — TensorRT-accelerated Wan VAE encode/decode.
- **`comfyui-wanrt/nodes/rife_rt.py`** — TensorRT-accelerated RIFE frame interpolation, modeled on
  [ComfyUI-Rife-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt).

Each file has **no dependency on anything else in this repo** — only `torch`, `tensorrt`,
`requests`, and ComfyUI's own `comfy`/`folder_paths` modules, all already present in a normal
ComfyUI install. Copy either file on its own into any `custom_nodes/*/` package's node list, or
copy this whole `comfyui-wanrt/` directory into `ComfyUI/custom_nodes/` to get both.

## What "just works" without setup

Both nodes auto-download their base model and auto-build/cache their TensorRT engine the first
time they're used — no separate export/build step, no CLI:

- **VAE**: checks `ComfyUI/models/vae/<file>` for the checkpoint; downloads it from HuggingFace
  if missing. Default is `wan_2.1_vae.safetensors` (correct for Wan 2.1 *and* Wan 2.2's 14B I2V
  models); `wan2.2_vae.safetensors` is offered as an explicit second option for Wan 2.2's separate
  5B TI2V model — these are different VAE architectures, never auto-selected by guessing from a
  DiT checkpoint name.
- **RIFE**: checks `ComfyUI/models/onnx/rife/` for the pretrained ONNX model; downloads it from
  `yuvraj108c/rife-onnx` on HuggingFace if missing.
- Both then build a TensorRT engine from that checkpoint/ONNX the first time it's requested,
  caching it under `ComfyUI/models/tensorrt/{vae,rife}/` and reusing it on every later run.

## Arbitrary size, with one real caveat

Each engine covers a *wide range of resolutions* via a TensorRT dynamic-shape profile (256–1088px
for the VAE, 256–3840px for RIFE) — arbitrary width/height within that range needs no rebuild.

The VAE's **frame-count** axis is the exception: Wan's VAE runs a data-dependent chunked
causal-conv loop internally, so `torch.export` bakes the loop's trip count in as a constant at
trace time — an engine built for one frame count only ever accepts that frame count. Each distinct
frame count you actually request gets its own engine, built once and cached from then on, same
"build on first use" idea as the resolution envelope, just keyed on a dimension that can't be
folded into one profile. RIFE has no such axis (it always operates on exactly two frames plus one
interpolation timestep), so this only applies to `vae_rt.py`.

## A known tradeoff, inherited from this project's own prior work

TensorRT-accelerating the VAE is comparatively low-value (cheap next to a full DiT denoise loop)
and comparatively risky (a from-scratch reimplementation is more likely to have a subtle
correctness bug than to need the speedup) — a conclusion this project reached before pivoting to
these two nodes. Every TensorRT call in `vae_rt.py` therefore falls back to eager PyTorch
(`comfy.sd.VAE`) on failure instead of crashing generation.

## Nodes

| File | Loader | Encode/Decode/Interpolate |
|---|---|---|
| `vae_rt.py` | `TensorRTWanVAELoader` (vae filename, precision) | `TensorRTWanVAEEncode` (IMAGE → LATENT), `TensorRTWanVAEDecode` (LATENT → IMAGE) |
| `rife_rt.py` | `TensorRTRifeLoader` (model, precision) | `TensorRTRifeInterpolate` (IMAGE batch + multiplier → IMAGE batch) |

## Tests

`tests/test_vae_rt.py` / `tests/test_rife_rt.py` exercise only the pure, non-GPU logic (cache
filenames, envelope bounds, the interpolation frame-pairing loop, checkpoint-exists-vs-download
branching) via `unittest`, with `tensorrt`/`folder_paths` injected as fakes — no TensorRT/CUDA/
ComfyUI install required to run them:

```bash
python -m unittest tests.test_vae_rt tests.test_rife_rt -v
```

Actual engine build + inference correctness needs a GPU and a real ComfyUI install to verify.

## License

Apache 2.0 — see [LICENSE](LICENSE).
