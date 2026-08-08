"""Run trivial all-zero DiT inputs through a debug-tap-augmented TensorRT engine and report which
taps come back NaN. Companion to add_debug_outputs.py -- pass its names_out.txt for idx labels.

Reuses TensorRTEngineWrapper (tensorrt_wan/engine/base.py) rather than touching the TensorRT API
directly -- it already handles every named output generically (engine.num_io_tensors), so a
debug-tap engine with extra outputs works unmodified.

Usage: python3 bisect_debug_taps.py <engine_path> <names_out.txt>
"""
import sys

import torch

from tensorrt_wan.engine.base import TensorRTEngineWrapper

engine_path, names_path = sys.argv[1], sys.argv[2]

idx_by_name = {}
with open(names_path) as f:
    for line in f:
        idx, name = line.strip().split("\t")
        idx_by_name[name] = int(idx)

wrapper = TensorRTEngineWrapper(engine_path)
wrapper.load()

# Matches DiTExporter.example_inputs() / the build_all.sh dit build command's shapes exactly
# (in_channels=36, text_dim=4096, latent_frames/height/width defaults 21/60/104, max_text_tokens=512).
inputs = {
    "x": torch.zeros(1, 36, 21, 60, 104, dtype=torch.float16, device="cuda"),
    "timestep": torch.zeros(1, dtype=torch.float16, device="cuda"),
    "context": torch.zeros(1, 512, 4096, dtype=torch.float16, device="cuda"),
}

outputs = wrapper.infer(inputs)

results = []
for name, tensor in outputs.items():
    t = tensor.float()
    nan_frac = torch.isnan(t).float().mean().item()
    results.append((idx_by_name.get(name, -1), name, nan_frac, tuple(tensor.shape)))

results.sort()
for idx, name, nan_frac, shape in results:
    print(f"idx={idx:5d}  name={name:20s}  nan_frac={nan_frac:.4f}  shape={shape}")
