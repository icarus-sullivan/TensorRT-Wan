# TensorRT-RT

Three self-contained, drag-and-droppable ComfyUI custom nodes:

- **`comfyui-wanrt/nodes/vae_rt.py`** — TensorRT-accelerated Wan VAE encode/decode.
- **`comfyui-wanrt/nodes/rife_rt.py`** — TensorRT-accelerated RIFE frame interpolation, modeled on
  [ComfyUI-Rife-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt).
- **`comfyui-wanrt/nodes/tensorrt_perf.py`** — Wan 2.2 model loaders (`TensorRTDiffusionLoader`,
  `TensorRTCheckpointLoader`) with optional
  [SageAttention3](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell)
  (Blackwell FP4) and [MagCache](https://github.com/Zehong-Ma/ComfyUI-MagCache) applied to the
  returned `MODEL`.

Each file has **no dependency on anything else in this repo** — only `torch`, `numpy`, and
ComfyUI's own `comfy`/`folder_paths` modules, all already present in a normal ComfyUI install
(`vae_rt.py` additionally needs `tensorrt` + `requests`; `tensorrt_perf.py` additionally
pip-installs `sageattn3` on first use of SageAttention — see below). Copy any file on its own into
any `custom_nodes/*/` package's node list, or copy this whole `comfyui-wanrt/` directory into
`ComfyUI/custom_nodes/` to get all three.

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

Each engine covers a *wide range of resolutions* via a TensorRT dynamic-shape profile (256–1536px
for the VAE, 256–3840px for RIFE) — arbitrary width/height within that range needs no rebuild. The
VAE's bound is deliberately narrower than RIFE's: this specific VAE architecture (not RIFE's) has a
confirmed real OOM at wide profile bounds — see the comment above `ENCODER_HEIGHT` in `vae_rt.py`
before widening it further.

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
| `rife_rt.py` | `TensorRTRifeLoader` (model, precision) | `TensorRTRifeInterpolate` (IMAGE batch + multiplier → IMAGE batch), `TensorRTRifeResampleFPS` (IMAGE batch + source/target fps → IMAGE batch) |
| `tensorrt_perf.py` | `TensorRTDiffusionLoader` (diffusion_model → MODEL), `TensorRTCheckpointLoader` (checkpoint → MODEL, CLIP, VAE) | — both loaders take the same `SageAttention`/`MagCache` dropdowns |

## SageAttention3 + MagCache loaders (`tensorrt_perf.py`)

`TensorRTDiffusionLoader` and `TensorRTCheckpointLoader` load a model exactly like ComfyUI's own
`UNETLoader`/`CheckpointLoaderSimple` (`TensorRTDiffusionLoader` includes the same `weight_dtype`
fp8 option `UNETLoader` has, threaded into `comfy.sd.load_diffusion_model`'s `model_options` the
same way — checked against this project's actual running `nodes.py`, not assumed; `CheckpointLoaderSimple`
has no such option, so neither does `TensorRTCheckpointLoader`), then optionally patch the returned `MODEL` — never the
weights on disk, never a global/process-wide monkeypatch:

- **`SageAttention: Disabled / Enabled`** — Blackwell-native FP4 attention
  ([thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)'s `sageattention3_blackwell`
  subproject, package `sageattn3`). **This node only supports Blackwell — there is no fallback to
  SpargeAttention or SageAttention2 for other GPUs, on purpose**: checked directly against both
  projects' current `setup.py` (not assumed from a README), `thu-ml/SpargeAttn` has no Blackwell
  kernel path at all (`SUPPORTED_ARCHS = {8.0, 8.6, 8.7, 8.9, 9.0}`, Ampere–Hopper only — confirmed
  by a real failed build attempt on an RTX PRO 6000 Blackwell, 2026-08-15), while
  `sageattention3_blackwell` is the mirror image: a dedicated Blackwell FP4 CUTLASS kernel
  (`-gencode arch=compute_120a,code=sm_120a` etc.) that hard-errors on any other compute
  capability. So `TensorRTDiffusionLoader`/`TensorRTCheckpointLoader` require exactly
  `sm_100`/`sm_120`/`sm_121` (B200 / RTX PRO 6000 Blackwell / RTX 50-series) and raise a clear
  `[TensorRT-RT Perf]` error naming the detected GPU otherwise, rather than silently no-op'ing.
  Enabling it sets `model_options["transformer_options"]["optimized_attention_override"]` on a
  `model.clone()`, which is ComfyUI's own supported per-model attention hook
  (`comfy.ldm.modules.attention.wrap_attn` checks for this key on every attention call) — not a
  patch of `comfy.ldm.wan.model.optimized_attention` or any other module-level global, so it can
  never affect an unrelated node's model. The override reshapes q/k/v into the `HND` layout
  `sageattn3_blackwell` expects and falls back to ComfyUI's normal attention (the `func` argument
  the hook is called with) whenever the FP4 kernel's real constraints aren't met — an explicit
  attention mask, `head_dim` not in `{64, 128}`, sequence < 128 tokens, or any runtime error — so a
  call it can't handle degrades gracefully instead of crashing generation. `sageattn3` isn't a
  normal ComfyUI dependency; the first time `SageAttention` is enabled, `tensorrt_perf.py`
  pip-installs `ninja` then
  `git+https://github.com/thu-ml/SageAttention.git#subdirectory=sageattention3_blackwell` into the
  same Python environment ComfyUI is running in (`sys.executable`, never a bare `pip`), compiling
  its CUTLASS-based CUDA extension once and reusing it on every later ComfyUI start (that build
  step clones NVIDIA/cutlass and requires CUDA ≥ 12.8, matching this repo's target). Note
  SageAttention3's own README says it hasn't been validated as lossless on every model family (Wan
  isn't in its explicitly-tested list — CogVideoX-2B/HunyuanVideo/Mochi/Flux/SD3.5 are) — it's
  FP4-quantized, more aggressive than int8 sparse kernels, so watch output quality and prefer
  `MagCache`-only if it visibly degrades a generation.
- **`MagCache: Disabled / Fast / Balanced / Quality / Custom`** — magnitude-aware step caching
  ([Zehong-Ma/ComfyUI-MagCache](https://github.com/Zehong-Ma/ComfyUI-MagCache)), ported (not
  reinvented) from that project's real `magcache_wanmodel_forward` and its calibrated Wan 2.2
  magnitude-ratio tables for `t2v_14B`/`i2v_14B`/`ti2v_5B` (picked by matching those substrings in
  the loaded filename). It patches `diffusion_model.forward_orig` only for the duration of one
  forward call via `unittest.mock.patch.multiple` inside a
  `set_model_unet_function_wrapper`, restored immediately after — never a standing monkeypatch.
  MagCache hard-errors instead of silently no-op'ing if the loaded model isn't Wan
  (`comfy.ldm.wan.*`) or the filename doesn't match a known Wan 2.2 variant. Its running
  error/skip/residual state lives in `model_options["transformer_options"]["magcache_state"]` — a
  plain dict, not an attribute on the shared `diffusion_model` module — specifically so that
  `model.clone()` (which deep-copies `model_options`) gives two differently-configured `MODEL`s
  genuinely independent cache state, and it's reset to its initial no-skip values whenever the
  sampler wrapper detects step index 0 (a new generation), so no run ever reuses a previous
  generation's cached residual. **If MagCache output looks blurry/degraded**, first test with
  `MagCache = Disabled` to isolate whether MagCache is actually the cause (vs. e.g. a mismatched
  LoRA) before tuning; `Quality` (thresh 0.03, K=1, retention 0.25) skips the least of the presets.
  **Don't use MagCache with a distilled few-step schedule** (e.g. LightX2V-style LoRAs running
  ~4–12 steps per phase) — confirmed via a real report of output turning washed-out/brown starting
  a couple steps in. The `mag_ratios` calibration was captured on a ~38–76 step standard Wan2.2
  trajectory; at 4–12 real steps each one carries far more denoising work than that curve assumes,
  so skip decisions end up systematically wrong. `tensorrt_perf.py` prints one loud
  `[TensorRT-RT Perf] WARNING` the first time a MagCache-enabled run has under ~20 total steps
  rather than silently producing bad output — there's also little to gain from MagCache on an
  already-few-step schedule even when it behaves.
  The `magcache_thresh`/`magcache_K`/`magcache_retention_ratio` optional widgets only matter for
  `Custom` — they're ignored for the named presets. If this whole `comfyui-wanrt/` package is
  installed (not just `tensorrt_perf.py` dragged in alone), `web/tensorrt_perf.js` auto-fills those
  three widgets with the selected preset's real values whenever `MagCache` changes, so Fast/
  Balanced/Quality never need touching them by hand; that file is purely cosmetic, so a missing/
  outdated frontend never affects what actually gets applied.
- **LoRA compatibility**: both patches live in `model_options`/the unet function wrapper, not in
  the weights, so `Load LoRA` → `Load LoRA` → `KSampler` after either loader works exactly as it
  would after ComfyUI's own loaders — LoRA patching is a separate, orthogonal system. SageAttention
  and MagCache also compose with each other: SageAttention only intercepts the innermost attention
  call, MagCache only wraps the outer unet forward, neither touches the other's hook.

Example workflows:

```text
TensorRTDiffusionLoader (SageAttention=Enabled, MagCache=Disabled)
        ↓
    Load LoRA
        ↓
    KSampler
```

```text
TensorRTDiffusionLoader (SageAttention=Enabled, MagCache=Balanced)
        ↓
    Load LoRA
        ↓
    KSampler
```

## Pre-building the VAE engine

Engines build lazily on first real use inside a workflow, which means the first build happens
mid-generation and a build failure is silently masked by the eager-PyTorch fallback. To force-build
and cache the VAE engines ahead of time (and get a loud failure instead of a silent fallback), run
on the pod:

```bash
python comfyui-wanrt/build_vae_engine.py --encoder-frames 1 --latent-frames 21
```

No resolution flags — the built engine already covers `vae_rt.py`'s full dynamic-shape range
(currently 256–1536px), not a single target size; only frame count is a real per-engine choice
(the one axis that genuinely can't be made dynamic, see the caveat above). See `--help` for all
options. RIFE has no equivalent script yet — its loader only builds on first real
`TensorRTRifeInterpolate`/`ResampleFPS` call.

## Tests

`tests/test_vae_rt.py` / `tests/test_rife_rt.py` / `tests/test_tensorrt_perf.py` exercise only the
pure, non-GPU logic (cache filenames, envelope bounds, the interpolation frame-pairing loop,
checkpoint-exists-vs-download branching, MagCache preset resolution, the attention-override
reshape math on CPU with a fake `sageattn3`, MagCache's step-0 state reset) via `unittest`, with
`tensorrt`/`folder_paths`/`sageattn3` injected as fakes or mocked — no TensorRT/CUDA/ComfyUI
install required to run them:

```bash
python -m unittest tests.test_vae_rt tests.test_rife_rt tests.test_tensorrt_perf -v
```

Actual engine build + inference correctness needs a GPU and a real ComfyUI install to verify.

## License

Apache 2.0 — see [LICENSE](LICENSE).
