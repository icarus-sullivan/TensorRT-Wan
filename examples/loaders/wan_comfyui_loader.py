"""--loader adapter for `trtwan export onnx` / `trtwan build engine` that reuses ComfyUI's own
Wan model-loading code instead of reimplementing Wan's architecture in this repo.

See docs/wan2.2_i2v_14b_notes.md for how this was derived (traced from `UNETLoader.load_unet` in
ComfyUI's nodes.py) and verified end to end on RunPod hardware: `load_dit()` here, plus a
standalone `torch.export.export()` call against the real 14.29B-param checkpoint, both succeed
(see the notes doc's "torch.export succeeded" section).

Usage:
    COMFYUI_ROOT=/workspace/runpod-slim/ComfyUI trtwan export onnx \
        --component dit \
        --loader examples.loaders.wan_comfyui_loader:load_dit \
        --checkpoint /workspace/runpod-slim/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
        --output dit_high_noise.onnx \
        --exporter-kwargs '{"in_channels": 36, "text_dim": 4096}'

`in_channels=36`, not 16 — see docs/wan2.2_i2v_14b_notes.md's conditioning-mismatch section.
`DiTExporter`/`DiTEngine` now use the right input names (`x`/`timestep`/`context`, confirmed
against `WanModel.forward`) and the right shape, but still don't *construct* a real 36-channel
`x` for I2V (noise + image-latent + mask, channel-concatenated) — `example_inputs()` traces
against zeros. Fine for a T2V-only export attempt; a real I2V export still needs that built
first. `DiTEngine._build_inputs` also raises `NotImplementedError` if any non-text conditioning
is present, rather than silently mis-routing it, for the same reason.
"""

from __future__ import annotations

import os
import sys

import torch

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch RoPE application. Cloned (not imported) from comfy/ldm/flux/math.py's own
    `_apply_rope1` fallback, monkeypatched into comfy.ldm.wan.model in place of the default
    `comfy_kitchen` custom op — see the comment at the call site in `load_dit` for why.

    Cloned rather than depended on: (1) this loader should keep working even outside a ComfyUI
    environment eventually — a from-scratch/non-ComfyUI loader needs this exact math anyway, so
    it belongs owned in this project, not borrowed at runtime from comfy internals that could
    change; (2) it decouples us from `comfy.model_management.in_training`'s exact semantics and
    blast radius (that flag may gate other comfy_kitchen ops we haven't exercised/verified) —
    patching this one function is scoped to exactly what's been confirmed necessary.

    Correctness note: this is Wan's actual convention — interleaved-pair 2D rotation via a
    per-position 2x2 matrix from `rope()` — NOT the rotate-half convention our own
    `RotaryEmbedding` TensorRT plugin (`plugins/csrc/rotary_embedding/`) currently implements.
    See docs/wan2.2_i2v_14b_notes.md's RoPE-convention finding; the plugin needs fixing to match
    this before it can be trusted. This clone is the reference to fix it against.
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def _add_comfyui_to_path() -> None:
    """ComfyUI isn't a pip-installed package — its `comfy` module only imports if ComfyUI's own
    repo root is on sys.path. Set COMFYUI_ROOT, or this defaults to the path this loader was
    originally written against.
    """
    comfyui_root = os.environ.get("COMFYUI_ROOT", "/workspace/runpod-slim/ComfyUI")
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)


def load_dit(checkpoint_path: str) -> torch.nn.Module:
    """Load a Wan DiT checkpoint (.safetensors under ComfyUI's `diffusion_models/`) via
    `comfy.sd.load_diffusion_model` and return the underlying transformer module, moved to GPU
    and cast to the target export dtype.

    `comfy.sd.load_diffusion_model` alone is not enough: ComfyUI keeps a full-precision (fp32)
    master copy on the CPU "offload device" and only moves/casts a model when its own execution
    engine runs a forward pass, which we never trigger here — confirmed empirically on the
    RunPod instance, see docs/wan2.2_i2v_14b_notes.md's "actually ran" section. Without the
    explicit `.to()` below, `torch.export` would trace against fp32-on-CPU tensors instead of
    what the built engine is actually meant to run.

    This is the single high-noise or low-noise expert for a Wan 2.2 MoE checkpoint — there is no
    engine-level support in this repo yet for switching between two experts mid-schedule (see
    docs/wan2.2_i2v_14b_notes.md's MoE section); call this once per expert and export/build each
    as its own engine.

    Override device/dtype via `TRTWAN_LOADER_DEVICE` (default `cuda`) / `TRTWAN_LOADER_DTYPE`
    (default `fp16`; one of fp32/fp16/bf16) if you need to debug against a CPU or fp32 trace.
    """
    _add_comfyui_to_path()

    import comfy.ldm.wan.model as wan_model  # noqa: E402
    import comfy.sd  # noqa: E402

    # comfy/ldm/wan/model.py imports apply_rope1 by name from comfy/ldm/flux/math.py, which
    # dispatches to an opaque `comfy_kitchen` custom op with no ONNX translation by default —
    # confirmed to break torch.onnx.export with "No ONNX function found for
    # comfy_kitchen.apply_rope1" on real hardware (see docs/wan2.2_i2v_14b_notes.md). Monkeypatch
    # just this one call site to our own cloned pure-PyTorch implementation (`_apply_rope1`
    # above) rather than flipping comfy.model_management.in_training, a broad global flag whose
    # full effect on comfy's other custom-kernel dispatch we haven't audited. Only affects this
    # process, not a running ComfyUI server (separate process/memory space).
    wan_model.apply_rope1 = _apply_rope1

    device = os.environ.get("TRTWAN_LOADER_DEVICE", "cuda")
    dtype = _DTYPES[os.environ.get("TRTWAN_LOADER_DTYPE", "fp16")]

    model_patcher = comfy.sd.load_diffusion_model(checkpoint_path, model_options={})
    diffusion_model = model_patcher.model.diffusion_model
    diffusion_model = diffusion_model.to(device=device, dtype=dtype)

    # forward_orig (comfy/ldm/wan/model.py) does `self.patch_embedding(x.float()).to(x.dtype)` -
    # it deliberately runs this one conv in fp32 regardless of the rest of the model's dtype, then
    # casts the result straight back down. comfy.ops's conv here does no dtype reconciliation of
    # its own (unlike some of its other op variants), so patch_embedding's weight must actually be
    # fp32 to match, or eager/export both fail on a dtype mismatch. Cheap to do - this submodule is
    # ~1.3M params - versus casting the whole 14.29B-param model to fp32 (which OOM'd a 95GB GPU
    # already holding another ComfyUI-loaded copy; see docs/wan2.2_i2v_14b_notes.md).
    if dtype != torch.float32 and hasattr(diffusion_model, "patch_embedding"):
        diffusion_model.patch_embedding = diffusion_model.patch_embedding.to(dtype=torch.float32)

    diffusion_model.eval()
    return diffusion_model
