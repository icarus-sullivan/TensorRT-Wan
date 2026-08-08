# RunPod session setup — read this first, before rebuilding anything

**Golden rule: never edit files live on the pod.** Always edit the local Mac checkout
(`/Users/csullivan/Desktop/TensorRT-Wan`) and `rsync` up to `/workspace/runpod-slim/TensorRT-Wan`,
never the reverse and never a direct `ssh ... 'sed -i ...'`/inline edit on the remote copy. A pod
is ephemeral outside `/workspace`, and even `/workspace` itself isn't guaranteed to be the *same*
persistent volume next session — a fix that only ever existed as a live edit on a pod's checkout
dies with that pod, silently, with no trace in git history or local files. This is the leading
suspect for why last night's (2026-08-06) session's DiT build apparently didn't hit the RoPE
float64-einsum NaN bug this session found (2026-08-07) even though the bug is structural, not
shape-dependent: if last night's working state included a live-only fix on that pod that was never
synced back to the local repo, a fresh pod today would have no way to inherit it. Not confirmed
(that pod's gone), but plausible enough to make this a hard rule going forward, not just a
convenience.

**Reality check:** a new RunPod container is ephemeral outside `/workspace` (the persistent
network volume). Expect to redo environment setup on most fresh pods. Before doing *anything*
below, check whether you're actually on a fresh pod or reconnecting to one that already has state:

```bash
ls /workspace/runpod-slim/TensorRT-Wan 2>/dev/null          # repo already synced?
ls /workspace/runpod-slim/trtwan_engines/*.engine 2>/dev/null  # engines already built?
python3 -c "import tensorrt" 2>&1                             # deps already installed?
```

If all three are present, skip straight to "Assembling a model_dir and running generate()" below —
do not rebuild anything without a specific reason (see "Why an engine might need rebuilding").

## One-shot environment setup (fresh pod only)

```bash
rsync -rlptDz --exclude='.git' --exclude='runpod_session_*' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.venv' \
  -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
  /Users/csullivan/Desktop/TensorRT-Wan/ root@<HOST>:/workspace/runpod-slim/TensorRT-Wan/

ssh ... 'cd /workspace/runpod-slim/TensorRT-Wan && pip install -e ".[tensorrt]" transformers pytest'
```

This installs TensorRT (pip resolves the right `tensorrt_cu*` wheel automatically), onnx/onnxscript,
polygraphy, onnxruntime-gpu, pytest. Confirmed working: TensorRT 11.2.1.2 on CUDA 12.8/Blackwell,
2026-08-06/07.

`COMFYUI_ROOT` defaults to `/workspace/runpod-slim/ComfyUI` in
`examples/loaders/wan_comfyui_loader.py` — no env var needed if that's where ComfyUI actually is on
this pod (it has been, both sessions so far, since it's baked into the persistent volume's
`runpod-slim` template).

## Speeding up iterative debug builds

TensorRT's build time is almost entirely CPU-orchestrated GPU kernel-tactic search — there's no
way to make it GPU-only. But `export/trt_build.py` sets `config.builder_optimization_level`
explicitly — defaults to `5` (max quality, deliberately not relying on TensorRT's own implicit
default of `3`) — and reads `TRTWAN_BUILDER_OPT_LEVEL` to override it. A lower level (try `1`)
searches far fewer tactics, trading final inference speed for much faster builds. Set this when
bisecting a bug and rebuilding repeatedly; leave it unset for a real deployment build. `export
TRTWAN_BUILDER_OPT_LEVEL=1` before the `build engine` call.

## Building engines: the exact commands and every gotcha, from real failures

**Always pass `--precision` explicitly — `fp16` for text_encoder/vae_encoder/vae_decoder,
`bf16` for dit specifically.** Never rely on `--precision auto` — on Blackwell it silently resolves
to `"fp8"` with no real per-op quality gate behind it (`runtime/precision.py`'s docstring promises
one; the code doesn't have one). See wan2.2_i2v_14b_notes.md's 2026-08-06 session.

**`dit` needs `bf16`, not `fp16` — this is load-bearing, not a performance tuning choice.** A
`fp16` DiT engine returns 100% NaN on every input at this project's real target scale
(~32,760-token self-attention) — confirmed via a full bisection session (attention decomposition,
TensorRT's `STRICT_NANS` flag, and query-chunking were all tried and ruled out; the actual cause is
`fp16`'s dynamic range inside TensorRT's self-attention kernel at that scale). `load_dit()`
(`examples/loaders/wan_comfyui_loader.py`) and `DiTExporter.dtype`
(`tensorrt_wan/export/exporters/dit.py`) both hardcode `bf16` now and ignore
`TRTWAN_LOADER_DTYPE` (logging a warning if it's set to anything else) specifically so this can't
be silently reintroduced by an env var — but the `build engine` CLI's `--precision` flag is
separate from that and still must be passed as `bf16` explicitly for `dit`, or
`_validate_precision` will reject the mismatch outright. text_encoder/vae_encoder/vae_decoder are
confirmed fine at `fp16` and still default to it via `TRTWAN_LOADER_DTYPE`; don't change those
without a reason. See wan2.2_i2v_14b_notes.md's 2026-08-07 session for the full investigation.

**Always pass `"static": true` in `--exporter-kwargs` for `vae_encoder`/`vae_decoder`.** A wide
dynamic H/W profile (the exporters' default) caused a real ~94GiB execution-context allocation
failure at actual inference time on this hardware — TensorRT appears to size scratch memory for the
profile's worst-case bound, not the shape actually used. `static=true` makes `dynamic_axes()`
return `{}`, producing a fully static shape end-to-end (same mechanism that already makes the DiT's
`context` input, which has no dynamic axis at all, work). This project only ever runs one resolution
per generation call anyway, so there's no real reason not to build static. (`dit`/`text_encoder`
haven't needed this — their dynamic ranges haven't triggered the same OOM — but the `static` param
exists on all four exporters if it's ever needed.)

**`vae_encoder` must be built at `frames=1`, not a larger T.** `_build_image_to_video_conditioning`
(`api/wan_engine.py`) does one `encode_image` (T=1) call per *distinct* pixel content (gray,
first, last), not a single whole-video `encode_video()` call — deliberately, because a `T>1`
`vae_encoder` build fails at the TensorRT step regardless of resolution (bisected: 9 frames @
256x256 builds, but 9 @ 480x832 and 21 @ 256x256 both fail identically — a real joint
frame-count/resolution complexity ceiling in this TensorRT version, not yet root-caused). Don't
try to rebuild `vae_encoder` at `frames=81` "to match the real algorithm" — it won't build.

**`dit`'s `context` input needs the tokenizer padded to exactly `max_text_tokens` (512 by
default).** It has no dynamic axis at all — a shorter/padding-mismatched `context` is a hard
static-dimension-mismatch build-time-baked failure, not a runtime nicety. `WanModelConfig` has a
`max_text_tokens` field for this; `_HFTokenizerAdapter`/`load_default_tokenizer` (and the ComfyUI
`TensorRTWanLoader` node) already use it — if you write a new tokenizer path, pad to this exactly.

**VAE checkpoint must be `models/vae/wan_2.1_vae.safetensors`, not `wan2.2_vae.safetensors`.**
The `2.2` file is for the separate 5B TI2V model (z_dim=48); these 14B checkpoints need z_dim=16
(`wan_2.1_vae.safetensors`). Confirmed the hard way, twice (once last night, once when a stale
docstring in `wan_comfyui_loader.py` still said the wrong file — now fixed).

**The RoPE `rope()` monkeypatch is required, not optional, for a non-NaN DiT engine.**
`comfy/ldm/flux/math.py`'s `rope()` computes its frequency table (`scale`/`omega`) in `float64`,
then combines it with a `float32` tensor inside a single `torch.einsum` call — fine under eager
execution's implicit type promotion, but `torch.export`/ONNX freezes that as a genuine
mixed-dtype Einsum node. `onnxruntime` correctly refuses to even load the resulting graph
(`Type parameter (T) of Optype (Einsum) bound to different types (tensor(float) and
tensor(double))`); TensorRT's parser accepts it, but the built engine then returns **100% NaN on
every single call, including its own trivial all-zero example_inputs()** — completely
input-independent. `load_dit()` in `wan_comfyui_loader.py` now monkeypatches
`comfy.ldm.flux.layers.rope` (not `comfy.ldm.wan.model.rope` — `EmbedND.forward()`, which calls
`rope()` unconditionally on every DiT forward pass, resolves the name against `flux/layers.py`'s
own module namespace) to `_rope_fp32`, a float32-only clone. **Not yet confirmed this actually
fixes it end-to-end** (a fresh DiT rebuild + real generate() run was in progress when these notes
were written) — check wan2.2_i2v_14b_notes.md's latest session section for the outcome before
trusting this line.

### The actual build commands (text_encoder → vae_encoder → vae_decoder → dit)

See `build_all.sh` pattern from the 2026-08-06/07 sessions (not committed — recreate from this):

```bash
CACHE_DIR=/workspace/runpod-slim/trtwan_engines
CKPT_DIR=/workspace/runpod-slim/ComfyUI/models
TRT="python3 -m tensorrt_wan.cli.main --cache-dir $CACHE_DIR"

# text_encoder
$TRT export onnx --component text_encoder --loader examples.loaders.wan_comfyui_loader:load_text_encoder \
  --checkpoint "$CKPT_DIR/text_encoders/umt5_xxl_fp16.safetensors" --output "$CACHE_DIR/text_encoder.onnx" \
  --exporter-kwargs '{"hidden_dim": 4096}'
$TRT build engine --component text_encoder --onnx "$CACHE_DIR/text_encoder.onnx" \
  --loader examples.loaders.wan_comfyui_loader:load_text_encoder \
  --checkpoint "$CKPT_DIR/text_encoders/umt5_xxl_fp16.safetensors" --exporter-kwargs '{"hidden_dim": 4096}' \
  --resolutions 480x832 --precision fp16

# vae_encoder (frames=1, static)
$TRT export onnx --component vae_encoder --loader examples.loaders.wan_comfyui_loader:load_vae_encoder \
  --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" --output "$CACHE_DIR/vae_encoder.onnx" \
  --exporter-kwargs '{"latent_channels": 16, "frames": 1, "height": 480, "width": 832, "static": true}'
$TRT build engine --component vae_encoder --onnx "$CACHE_DIR/vae_encoder.onnx" \
  --loader examples.loaders.wan_comfyui_loader:load_vae_encoder \
  --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
  --exporter-kwargs '{"latent_channels": 16, "frames": 1, "height": 480, "width": 832, "static": true}' \
  --resolutions 480x832 --precision fp16

# vae_decoder (static)
$TRT export onnx --component vae_decoder --loader examples.loaders.wan_comfyui_loader:load_vae_decoder \
  --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" --output "$CACHE_DIR/vae_decoder.onnx" \
  --exporter-kwargs '{"latent_channels": 16, "latent_frames": 21, "latent_height": 60, "latent_width": 104, "static": true}'
$TRT build engine --component vae_decoder --onnx "$CACHE_DIR/vae_decoder.onnx" \
  --loader examples.loaders.wan_comfyui_loader:load_vae_decoder \
  --checkpoint "$CKPT_DIR/vae/wan_2.1_vae.safetensors" \
  --exporter-kwargs '{"latent_channels": 16, "latent_frames": 21, "latent_height": 60, "latent_width": 104, "static": true}' \
  --resolutions 480x832 --precision fp16

# dit (high_noise expert -- Wan2.2 MoE, no expert-switching implemented, see roadmap.md)
$TRT export onnx --component dit --loader examples.loaders.wan_comfyui_loader:load_dit \
  --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --output "$CACHE_DIR/dit_high_noise.onnx" --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'
$TRT build engine --component dit --onnx "$CACHE_DIR/dit_high_noise.onnx" \
  --loader examples.loaders.wan_comfyui_loader:load_dit \
  --checkpoint "$CKPT_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}' --resolutions 480x832 --precision bf16
```

Expected sizes (fp16, this exact checkpoint set): text_encoder ~20.6GiB, vae_encoder ~40MB,
vae_decoder ~0.2GiB (static, was ~0.2GiB dynamic too — size didn't change much, the OOM was a
runtime execution-context issue not a weight-size issue), dit ~26.6GiB.

**Cache gotcha:** `EngineCache` keys on `(component, model_hash, tensorrt_version, cuda_version,
gpu_architecture, optimization_profile, precision, input_shape_digest)` — the last field
(`ModelExporter.shape_digest()`) was added this session specifically because
`optimization_profile` is just a *name* string ("480x832"), unrelated to the exporter's actual
traced shape/static-vs-dynamic-ness. If you ever see `build engine` print "Using cached engine"
when you expected a rebuild, suspect this — check the `.json` sidecar's `precision`/
`input_shape_digest` fields match what you actually meant to build, not just the file's presence.

## Assembling a model_dir and running generate()

`WanEngine.from_pretrained(model_dir)` expects fixed filenames
(`{text_encoder,dit,vae_encoder,vae_decoder}.engine` + `wan_model.json`), but `EngineCache` writes
content-addressed `{digest}.engine`/`.json` — you have to assemble a `model_dir` yourself:

```python
import json, shutil
from pathlib import Path
CACHE_DIR, MODEL_DIR = Path("/workspace/runpod-slim/trtwan_engines"), Path("/workspace/runpod-slim/trtwan_model")
MODEL_DIR.mkdir(exist_ok=True)
for meta_path in CACHE_DIR.glob("*.json"):
    meta = json.loads(meta_path.read_text())
    shutil.copyfile(meta_path.with_suffix(".engine"), MODEL_DIR / f"{meta['component']}.engine")
```

`wan_model.json` needs (confirmed against these checkpoints):
```json
{
  "latent_channels": 16, "vae_temporal_scale": 4, "vae_spatial_scale": 8,
  "text_embed_dim": 4096, "tokenizer_name": "google/umt5-xxl",
  "default_num_frames": 81, "default_resolution": [480, 832], "fps": 16, "max_text_tokens": 512
}
```
(`tokenizer_name`: ComfyUI's own T5/UMT5 tokenizer is a raw SentencePiece file, not a HF
`AutoTokenizer`-loadable repo — `google/umt5-xxl` is a reasonable real HF equivalent for the
standalone API's default tokenizer path, unconfirmed to produce byte-identical tokenization to
ComfyUI's own, but not a known problem so far.)

Test images (`close_green_chair_start.png`/`_end.png`, in `runpod_session_2026-08-06/` locally)
are **RGBA**, not RGB — strip the alpha channel (`image[:3]`) before feeding them to the VAE
encoder, or you'll get a channel-count shape mismatch.

## Where things actually are

- Repo: `/workspace/runpod-slim/TensorRT-Wan` (synced via rsync from the Mac, not git-cloned —
  this session's fixes aren't pushed to `origin` yet)
- ComfyUI + model checkpoints: `/workspace/runpod-slim/ComfyUI` (pre-existing on the persistent
  volume, not something this repo's tooling set up)
- Engine cache: `/workspace/runpod-slim/trtwan_engines` (persistent — survives pod restarts, unlike
  the default `~/.cache/tensorrt_wan/engines`, which is on the ephemeral root disk; always pass
  `--cache-dir /workspace/runpod-slim/trtwan_engines` explicitly)
- Assembled model_dir for `WanEngine.from_pretrained()`: `/workspace/runpod-slim/trtwan_model`
