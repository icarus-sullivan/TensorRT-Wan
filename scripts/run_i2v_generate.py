"""Real end-to-end I2V generate() test against a built model_dir. CPU-side image prep + real
TensorRT inference through WanEngine.

Usage: python3 run_i2v_generate.py <model_dir> <start_png> <end_png> <out_mp4> [height] [width]
Requires the DiT/VAE engines to have been built with a dynamic H/W profile covering
[height, width] -- see DiTExporter/VAEEncoderExporter/VAEDecoderExporter's min/max_latent_*
kwargs -- if height/width aren't given, falls back to wan_model.json's default_resolution.
"""
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/runpod-slim/TensorRT-Wan")
from tensorrt_wan.api.wan_engine import WanEngine

model_dir, start_png, end_png, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

engine = WanEngine.from_pretrained(model_dir, precision="bf16")
if len(sys.argv) > 6:
    height, width = int(sys.argv[5]), int(sys.argv[6])
else:
    height, width = engine.model_config.default_resolution


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    # Center-crop to the target aspect ratio before resizing, instead of a naive stretch-resize --
    # the real source images here are 720x1088 (portrait), the target is 832x480 (landscape); a
    # plain .resize() to (width, height) squishes the whole scene into the wrong aspect ratio.
    src_w, src_h = img.size
    target_aspect = width / height
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        new_w = round(src_h * target_aspect)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = round(src_w / target_aspect)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    img = img.resize((width, height), Image.LANCZOS)
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(height, width, 3)
    pixels = arr.permute(2, 0, 1).float() / 127.5 - 1.0  # (C, H, W) in [-1, 1]
    return pixels.unsqueeze(0).to(device=engine.device, dtype=torch.float16)


start = load_image(start_png)
end = load_image(end_png)

print(f"Running generate(): {engine.model_config.default_num_frames} frames @ {width}x{height}")
output = engine.generate(
    prompt="a green chair",
    image=start,
    last_image=end,
    resolution=(height, width),
    num_inference_steps=50,
    guidance_scale=3.0,
    guidance_scale_low_noise=1.0,
    seed=0,
)

frames = output.as_numpy()
import numpy as np

print(f"frames shape={frames.shape} dtype={frames.dtype}")
print(f"nan-safe (uint8, can't be NaN): min={frames.min()} max={frames.max()} mean={frames.mean():.2f}")
print(f"per-frame std (0 = a solid/degenerate frame): {[round(float(frames[t].std()), 2) for t in range(0, frames.shape[0], 10)]}")

output.save(out_path)
print(f"saved: {out_path}")
