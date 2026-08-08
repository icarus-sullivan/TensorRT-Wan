"""Validates the real TensorRTDiTLoader custom node class directly (no live ComfyUI server
needed) -- instantiates it the same way ComfyUI itself would call a node's FUNCTION method, then
runs the exact same real-pipeline flow as scripts/real_pipeline_trt_dit_test.py (real CLIP, real
VAE, real WanFMLFPluggable conditioning, real two_phase_sampler) to confirm the formalized node
code still produces a coherent result, not just the ad-hoc TRTDiTWrapper prototype it replaces.

Usage: python3 test_dit_loader_node.py <start_png> <end_png> <out_mp4>
"""
import sys
import time

import torch
from PIL import Image

_t_start = time.monotonic()


def _elapsed() -> str:
    return f"{time.monotonic() - _t_start:6.1f}s"


COMFYUI_ROOT = "/workspace/runpod-slim/ComfyUI"
sys.path.insert(0, COMFYUI_ROOT)
sys.path.insert(0, "/workspace/runpod-slim/TensorRT-Wan")

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import server  # noqa: E402


class _FakeRoutes:
    def __getattr__(self, _name):
        def _decorator(*_args, **_kwargs):
            return lambda fn: fn

        return _decorator


class _FakePromptServer:
    prompt_queue = None
    routes = _FakeRoutes()


server.PromptServer.instance = _FakePromptServer()

from custom_nodes.spnxx.nodes.wan_fmlf_pluggable import WanFMLFPluggable  # noqa: E402
from custom_nodes.spnxx.sampler.two_phase_sampler import two_phase_sampler  # noqa: E402
from custom_nodes.tensorrt_wan_comfyui.nodes.dit_loader import TensorRTDiTLoader  # noqa: E402

start_png, end_png, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
device = torch.device("cuda")
width, height, num_frames = 832, 480, 81
TRT_MODEL_DIR = "/workspace/runpod-slim/trtwan_model"


def load_image_comfy(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(img.height, img.width, 3)
    return (arr.float() / 255.0).unsqueeze(0).to(device)


print(f"[{_elapsed()}] Loading DiT models via the real TensorRTDiTLoader node class...")
loader_node = TensorRTDiTLoader()
(high_model,) = loader_node.load(
    unet_name="wan2.2_i2v_high_noise_14B_fp16.safetensors",
    engine_path=f"{TRT_MODEL_DIR}/dit_high_noise.engine",
    in_channels=36, max_text_tokens=512,
)
(low_model,) = loader_node.load(
    unet_name="wan2.2_i2v_low_noise_14B_fp16.safetensors",
    engine_path=f"{TRT_MODEL_DIR}/dit_low_noise.engine",
    in_channels=36, max_text_tokens=512,
)

print(f"[{_elapsed()}] Loading real CLIP (umt5_xxl)...")
clip_path = folder_paths.get_full_path_or_raise("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
clip = comfy.sd.load_clip(
    ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.WAN, model_options={},
)

print(f"[{_elapsed()}] Loading real VAE (wan_2.1_vae)...")
vae_path = folder_paths.get_full_path_or_raise("vae", "wan_2.1_vae.safetensors")
vae_sd, vae_metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
_t_setup_done = time.monotonic()

print(f"[{_elapsed()}] Encoding text (real CLIP)...")
positive = clip.encode_from_tokens_scheduled(clip.tokenize("a green chair"))
negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

print(f"[{_elapsed()}] Building conditioning (real WanFMLFPluggable)...")
start_image = load_image_comfy(start_png)
end_image = load_image_comfy(end_png)
positive, negative, latent = WanFMLFPluggable.execute(
    positive=positive, negative=negative, vae=vae,
    width=width, height=height, length=num_frames,
    start_image=start_image, end_image=end_image,
)
_t_cond_done = time.monotonic()

print(f"[{_elapsed()}] Running real two_phase_sampler...")
result = two_phase_sampler(
    high_model=high_model, low_model=low_model,
    high_cfg=1.8, low_cfg=1.1,
    high_start_step=0, high_end_step=6,
    low_start_step=6, low_end_step=10000,
    seed=0, steps=12,
    sampler_name="euler", scheduler="sgm_uniform",
    positive=positive, negative=negative,
    start_latent=latent, noise_amount=1.0,
)
_t_sample_done = time.monotonic()

high_model.model.diffusion_model.unload = lambda: high_model.model.diffusion_model._wrapper.unload()
high_model.model.diffusion_model.unload()
low_model.model.diffusion_model._wrapper.unload()
torch.cuda.empty_cache()

print(f"[{_elapsed()}] Decoding (real VAE)...")
_original_process_output = vae.process_output
vae.process_output = lambda image: _original_process_output(image.clone())
samples = result[0]["samples"].clone()
pixels = vae.decode(samples)

frames_np = (pixels.clamp(0, 1) * 255).to(torch.uint8).numpy(force=True)
if frames_np.ndim == 5:
    frames_np = frames_np[0]
print(f"frames: shape={frames_np.shape} mean={frames_np.mean():.2f} "
      f"per-frame std: {[round(float(frames_np[t].std()), 2) for t in range(0, frames_np.shape[0], 10)]}")

import imageio.v3 as iio

iio.imwrite(out_path, frames_np, fps=16, codec="libx264")
_t_end = time.monotonic()
print(f"saved: {out_path}")
print(f"TOTAL: {_t_end - _t_start:.1f}s")
