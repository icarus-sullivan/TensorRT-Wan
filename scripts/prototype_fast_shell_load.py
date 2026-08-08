"""Prototype: can we get a correctly-configured Wan model shell (model_sampling/latent_format/
concat_keys) without reading 28GB of real weight data we're about to discard anyway?

comfy.sd.load_diffusion_model() = comfy.utils.load_torch_file() (calls safetensors'
f.get_tensor(k) for every key -- materializes real data via mmap+copy, CPU-bound, ~28GB) +
load_diffusion_model_state_dict() (config detection, purely shape/key-name based per
comfy.model_detection). This builds a "shape-only" state dict (same tensor names/shapes/dtypes,
but torch.empty() meta placeholders -- zero real data read) and feeds that into the same real
detection path, comparing timing and confirming the resulting model config matches.

Usage: python3 prototype_fast_shell_load.py <checkpoint_path>
"""
import sys
import time

import torch
from safetensors import safe_open

COMFYUI_ROOT = "/workspace/runpod-slim/ComfyUI"
sys.path.insert(0, COMFYUI_ROOT)

import comfy.model_base  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

ckpt_path = sys.argv[1]

# --- Real (slow) path, for comparison ---
t0 = time.monotonic()
real_model = comfy.sd.load_diffusion_model(ckpt_path, model_options={})
t_real = time.monotonic() - t0
print(f"REAL load_diffusion_model: {t_real:.1f}s")
print(f"  model_sampling.shift={real_model.model.model_sampling.shift}")
print(f"  latent_format={type(real_model.model.latent_format).__name__}")
print(f"  concat_keys={getattr(real_model.model, 'concat_keys', None)}")
del real_model
torch.cuda.empty_cache()

# --- Fast (shape-only) path ---
t0 = time.monotonic()
with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    metadata = f.metadata()
    fake_sd = {}
    for k in f.keys():
        slice_ = f.get_slice(k)
        shape = slice_.get_shape()
        dtype_str = slice_.get_dtype()  # safetensors dtype string, e.g. "F16"
        torch_dtype = {
            "F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16,
            "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
            "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        }.get(dtype_str, torch.float32)
        fake_sd[k] = torch.empty(shape, dtype=torch_dtype, device="meta")
t_header = time.monotonic() - t0
print(f"\nHeader-only read ({len(fake_sd)} tensors): {t_header:.1f}s")

# load_diffusion_model_state_dict -> BaseModel.load_model_weights -> diffusion_model.load_state_dict(assign=...)
# `assign` comes from model_patcher.is_dynamic(), which is hardcoded False on this ComfyUI
# version's CoreModelPatcher (an alias to plain ModelPatcher, not the dynamic subclass) -- so this
# always tries a real .copy_() into our meta placeholders and fails ("Cannot copy out of meta
# tensor; no data!"). We don't need the copy to succeed at all: the caller only wants
# model_sampling/latent_format/concat_keys off the resulting shell, then discards
# `diffusion_model` entirely in favor of our own TensorRT wrapper. No-op the weight-copy step for
# the duration of this one call instead of fighting the meta-tensor copy.
_orig_load_model_weights = comfy.model_base.BaseModel.load_model_weights
comfy.model_base.BaseModel.load_model_weights = lambda self, sd, unet_prefix="", assign=False: self
t0 = time.monotonic()
try:
    fast_model = comfy.sd.load_diffusion_model_state_dict(fake_sd, model_options={}, metadata=metadata)
finally:
    comfy.model_base.BaseModel.load_model_weights = _orig_load_model_weights
t_fast = time.monotonic() - t0
print(f"FAST load_diffusion_model_state_dict (meta tensors): {t_fast:.1f}s")
if fast_model is None:
    print("FAILED: detection returned None")
else:
    print(f"  model_sampling.shift={fast_model.model.model_sampling.shift}")
    print(f"  latent_format={type(fast_model.model.latent_format).__name__}")
    print(f"  concat_keys={getattr(fast_model.model, 'concat_keys', None)}")
    print(f"  diffusion_model.patch_embedding weight shape="
          f"{fast_model.model.diffusion_model.patch_embedding.weight.shape if hasattr(fast_model.model.diffusion_model, 'patch_embedding') else 'N/A'}")

print(f"\nSpeedup: {t_real / (t_header + t_fast):.1f}x ({t_real:.1f}s -> {t_header + t_fast:.1f}s)")
