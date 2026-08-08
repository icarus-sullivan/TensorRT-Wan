"""Run the real eager PyTorch DiT model (no export, no TensorRT) against the exact same trivial
all-zero example_inputs() used throughout the NaN investigation, at real target scale
(in_channels=36, latent 21x60x104, text_dim=4096, max_text_tokens=512 -- ~32760 image tokens).

Every eager-vs-TensorRT comparison so far in this investigation used non-trivial "byte-identical
real" inputs, and only at a smaller pre-real-scale test. Nobody has checked whether eager itself
already produces NaN on trivial all-zero inputs at the real ~32760-token scale -- if it does, this
was never a TensorRT/export bug at all, it's the model math itself hitting a degenerate case
(e.g. timestep=0, or an all-zero activation triggering a div-by-zero/log-of-zero somewhere).

Usage: python3 eager_trivial_check.py <checkpoint_path> <loader_module:loader_fn>
"""
import sys

import torch

sys.path.insert(0, "/workspace/runpod-slim/TensorRT-Wan")

checkpoint_path, loader_ref = sys.argv[1], sys.argv[2]
module_name, fn_name = loader_ref.split(":")
import importlib

_original_sdpa = torch.nn.functional.scaled_dot_product_attention

loader = getattr(importlib.import_module(module_name), fn_name)

model = loader(checkpoint_path)
model.eval()

# load_dit() unconditionally monkeypatches scaled_dot_product_attention to the ONNX-export
# reference decomposition, which materializes the full (heads, 32760, 32760) attention matrix --
# ~80GB per block, OOMs immediately in eager. Restore the real native SDPA (flash/memory-efficient
# attention, no full-matrix materialization) for this eager check -- we want to know whether the
# actual fast kernel PyTorch runs at real scale produces NaN, not the export-only reference math.
torch.nn.functional.scaled_dot_product_attention = _original_sdpa

device = next(model.parameters()).device
# Deliberately fixed fp16, not inferred from next(model.parameters()).dtype -- this model
# intentionally mixes precision (patch_embedding stays fp32), matching ModelExporter.dtype
# (tensorrt_wan/export/base.py). See that docstring for the real failure this avoids.
dtype = torch.float16
print(f"model loaded: device={device} dtype={dtype}")

x = torch.zeros(1, 36, 21, 60, 104, device=device, dtype=dtype)
timestep = torch.zeros(1, device=device, dtype=dtype)
context = torch.zeros(1, 512, 4096, device=device, dtype=dtype)

with torch.no_grad():
    out = model(x=x, timestep=timestep, context=context)

if isinstance(out, (tuple, list)):
    out = out[0]

nan_frac = torch.isnan(out.float()).float().mean().item()
inf_frac = torch.isinf(out.float()).float().mean().item()
print(f"noise_pred: shape={tuple(out.shape)} nan_frac={nan_frac:.4f} inf_frac={inf_frac:.4f}")
print(f"min={out.float().min().item()} max={out.float().max().item()}")
