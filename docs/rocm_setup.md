# AMD ROCm setup (GMKtec K8 Plus / gfx1103)

Status: **unverified against real hardware** — everything below was built and reasoned from the
codebase and fetched official docs (ONNX Runtime's MIGraphX EP page, AMD's ROCm/AMDMIGraphX
supported-ops page, ComfyUI's own `comfy/ldm/wan/model.py`/`comfy/ops.py` source), not run on a
real gfx1103 device (no GPU execution happens locally in this project, see `PLAN.md`'s dev rule).
Expect to iterate once actually running this on the K8 Plus — matches this repo's existing
"Supported, untested" convention for every non-Blackwell row in
[supported_gpus.md](supported_gpus.md).

## Why this isn't just "point TensorRT at AMD"

TensorRT has no ROCm build at all — not a config flag, a hard vendor wall. AMD's accelerated ONNX
runtime is **MIGraphX** (via ONNX Runtime's `MIGraphXExecutionProvider`), which is what the DiT
build path below actually targets. Two real constraints shaped this, not assumptions:

- **MIGraphX caps out at ONNX opset 19** and has no native `RMSNormalization` or `Attention` op
  (both added at opset 23). This project's TensorRT export already decomposes attention for
  unrelated reasons (`_decomposed_sdpa` in `examples/loaders/wan_comfyui_loader.py`), so that part
  is free. RMSNorm needed a new decomposition (`_decomposed_rms_norm`,
  `_decompose_rms_norm_for_export`, same file) — Wan's `WanSelfAttention.norm_q`/`norm_k` bottom
  out in `torch.nn.functional.rms_norm` (traced via `comfy.ops.disable_weight_init.RMSNorm`,
  itself a `torch.nn.RMSNorm` subclass), and that's the function this decomposition monkeypatches.
- **MIGraphX's dynamic-shape support is inconsistent across ops.** This project already learned
  that lesson the hard way for the VAE decoder (a wide dynamic profile caused a real ~94GiB
  context-allocation failure on TensorRT — see `docs/wan2.2_i2v_14b_notes.md`). The DiT MIGraphX
  build reuses the same fix: `static=True`, one build per resolution profile, not one dynamic
  build covering a range.

The 8 custom TensorRT plugins (`plugins/csrc/*`) are **not** involved here — confirmed by grep,
they're not wired into the actual export pipeline at all (only a docstring comment references
one), so there's nothing to port to HIP/MIGraphX for the DiT to work.

## Text encoder / VAE

Stay on stock, eager ComfyUI nodes (`CLIPLoader`, `VAELoader`, `CLIPTextEncode`, `VAEDecode`) —
not MIGraphX-accelerated. The DiT is the expensive part (50-100 forward passes per generation);
text encoding and VAE encode/decode run once or twice each and aren't worth this build path's
complexity.

## Install

```bash
pip install --force-reinstall \
    "torch==2.14.0+rocm10.1.0a20260822" \
    "amd-torch-device-gfx1103==2.14.0+rocm10.1.0a20260822" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/
pip install --force-reinstall \
    "torchvision==0.29.0a0+rocm10.1.0a20260822" \
    "torchaudio==2.11.0+rocm10.1.0a20260822" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/
```

`onnxruntime` with the MIGraphX EP compiled in is a separate install. Prebuilt wheels exist via
AMD's `repo.radeon.com`, but they're typically pinned to specific ROCm point releases — they may
not match this nightly build exactly. If `providers=[("MIGraphXExecutionProvider", ...)]` fails to
load or `InferenceSession.get_providers()` doesn't include it, building `onnxruntime` from source
against this exact installed ROCm toolchain is the fallback (see ONNX Runtime's own BUILD docs).

**Troubleshooting**: if ROCm doesn't recognize gfx1103 out of the box on this build, try setting
`HSA_OVERRIDE_GFX_VERSION` before launching — a thing to try, not asserted as required here.

## The bf16-not-fp16 requirement (carries over unchanged)

This is model math, not a TensorRT artifact: this project's own investigation
(`docs/wan2.2_i2v_14b_notes.md`, 2026-08-07 session) found the DiT's self-attention returns 100%
NaN in **fp16** at real target scale (~32,760 tokens), and is clean in **bf16**. That's independent
of which runtime executes the graph — always export/build the DiT with `bf16`, never `fp16`, on
this path too.

## Build recipe

One static ONNX export + MIGraphX validation per resolution profile you actually need (see
`config/schema.py`'s `DEFAULT_RESOLUTION_PROFILES` — includes `720x1088`/`1088x720` alongside the
existing set):

```bash
# 1. Export: --target migraphx caps the opset at 19 and decomposes RMSNorm (see above).
#    static=True (via --exporter-kwargs) is required -- MIGraphX's dynamic-shape support is
#    inconsistent across ops (see above).
TRTWAN_EXPORT_TARGET=migraphx python3 -m tensorrt_wan.cli.main export onnx \
    --component dit --target migraphx \
    --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint /path/to/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --output dit_high_noise_720x1088.onnx \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096, "static": true, "latent_height": 90, "latent_width": 136}'

# 2. Build: validates the ONNX compiles under MIGraphXExecutionProvider and caches the ONNX file
#    itself (see export/migraphx_build.py's module docstring for why there's no separate
#    "compiled engine" blob the way TensorRT produces). Exactly one --resolutions profile.
python3 -m tensorrt_wan.cli.main build engine \
    --component dit --backend migraphx --precision bf16 \
    --onnx dit_high_noise_720x1088.onnx \
    --loader examples.loaders.wan_comfyui_loader:load_dit \
    --checkpoint /path/to/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --exporter-kwargs '{"in_channels": 36, "text_dim": 4096, "static": true, "latent_height": 90, "latent_width": 136}' \
    --resolutions 720x1088
```

Repeat per resolution and per MoE expert (`high_noise`/`low_noise`). Latent height/width = pixel
height/width / 8 (VAE spatial scale) — `720x1088` pixels -> `latent_height=90, latent_width=136`.

## ComfyUI graph

`MIGraphXDiTLoader` (`comfyui/nodes/dit_loader_migraphx.py`) — same shell-loading contract as
`TensorRTDiTLoader`, points `engine_name` at the cached `.onnx` instead of a `.engine` file.
Pair with stock `CLIPLoader` → `CLIPTextEncode` → `KSamplerAdvanced` → `VAEDecode`, and optionally
`WanResolutionProfile` (`comfyui/nodes/resolution_profile.py`) feeding
`EmptyHunyuanLatentVideo`'s width/height — same node either backend, it's just a lookup table.

## Standalone API

`WanEngine.from_pretrained()` auto-detects the backend via `RuntimeManager.backend` (`"migraphx"`
when ROCm is detected and TensorRT isn't — see `runtime/manager.py`'s `_resolve_backend`) and
looks for `dit_high_noise.onnx`/`dit_low_noise.onnx` instead of `.engine` files in `model_dir`.
Text encoder/VAE still expect `.engine` files regardless of backend (see "Text encoder / VAE"
above) — a ROCm `model_dir` needs those built via a real NVIDIA TensorRT environment, or the
standalone API isn't usable end-to-end yet on this hardware; the ComfyUI graph path (stock
CLIP/VAE nodes) doesn't have this limitation.

## Diagnostics

`trtwan gpu-report` now recognizes AMD/ROCm devices (`GPUArchitecture.AMD_RDNA3` for gfx11xx,
classified from ROCm's `gcnArchName` rather than CUDA SM major/minor — see `runtime/gpu.py`) and
reports `Backend: migraphx` when TensorRT isn't installed on ROCm hardware.
