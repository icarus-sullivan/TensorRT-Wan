"""Bypass the DiT/scheduler entirely: encode a real image, immediately decode it back, and save
the result next to the original. If this alone isn't recognizable, the bug is in the VAE
(encoder, decoder, or the channel/scale convention between them), not the DiT or conditioning --
every generate() output passes through the decoder regardless of how correct the DiT's latent
prediction is, so a broken round-trip here would explain persistent noise even with a fully
correct DiT.

Usage: python3 vae_roundtrip_check.py <model_dir> <input_png> <output_png>
"""
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/runpod-slim/TensorRT-Wan")
from tensorrt_wan.api.model_config import WanModelConfig
from tensorrt_wan.engine.vae_engine import VAEDecoderEngine, VAEEncoderEngine

model_dir, in_png, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

config = WanModelConfig.load(f"{model_dir}/wan_model.json")
height, width = config.default_resolution
device = torch.device("cuda")

img = Image.open(in_png).convert("RGB").resize((width, height), Image.LANCZOS)
arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(height, width, 3)
pixels = arr.permute(2, 0, 1).float() / 127.5 - 1.0
pixels = pixels.unsqueeze(0).to(device=device, dtype=torch.float16)  # (1, 3, H, W) in [-1, 1]

vae_encoder = VAEEncoderEngine(f"{model_dir}/vae_encoder.engine", device=device)
vae_encoder.load()
latent = vae_encoder.encode_image(pixels)  # (1, C, 1, h, w)
vae_encoder.unload()
print(f"latent: shape={tuple(latent.shape)} min={latent.float().min().item():.3f} "
      f"max={latent.float().max().item():.3f} mean={latent.float().mean().item():.3f} "
      f"std={latent.float().std().item():.3f}")

# vae_decoder was built static for latent_frames=21 -- pad the single encoded frame out to that
# length (repeat) so the shape matches what the engine expects, then take just the first decoded
# frame back out. Not the real generation algorithm, just enough to isolate the VAE round-trip.
latent_frames = 21
latent_padded = latent.repeat(1, 1, latent_frames, 1, 1)

vae_decoder = VAEDecoderEngine(f"{model_dir}/vae_decoder.engine", device=device)
vae_decoder.load()
decoded = vae_decoder.decode(latent_padded)  # (1, 3, T, H, W) in [-1, 1]
vae_decoder.unload()

frame = decoded[0, :, 0]  # (3, H, W)
print(f"decoded frame: min={frame.float().min().item():.3f} max={frame.float().max().item():.3f} "
      f"mean={frame.float().mean().item():.3f} std={frame.float().std().item():.3f}")

out = ((frame.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
Image.fromarray(out).save(out_png)
print(f"saved: {out_png}")
