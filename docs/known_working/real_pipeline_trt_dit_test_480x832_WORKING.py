"""Decisive test: real ComfyUI CLIP + real ComfyUI VAE + real ComfyUI conditioning/sampling
machinery (the exact code path that produced a coherent reference video this session), with ONLY
the diffusion model swapped for our TensorRT DiT engine. If this converges, the bug was entirely in
this project's own reimplementation (conditioning construction, scheduler, CFG) and never the DiT
itself. If it's still noise, the DiT/TensorRT engine itself is the remaining suspect.

Deliberately narrowed vs. the full reference workflow to isolate one variable: no lightx2v LoRA (our
engine has no LoRA applied), no CLIP vision (our exported DiT graph has no clip_fea input), 480x832
landscape (our DiT's tuned opt point, not the portrait native resolution -- one variable at a time).
Everything else (CLIP, VAE, WanFMLFPluggable conditioning, two_phase_sampler) is 100% real ComfyUI/
custom-node code, imported and called directly, not reimplemented.

Usage: python3 real_pipeline_trt_dit_test.py <start_png> <end_png> <out_mp4> [width] [height]
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

# custom_nodes/spnxx/__init__.py pulls in nodes/video.py at package-import time, which assumes a
# running ComfyUI server (server.PromptServer.instance) -- we're calling two specific node
# functions directly, not running the server, so mock just enough to satisfy that module-load-time
# attribute access.
import server  # noqa: E402


class _FakeRoutes:
    """aiohttp-style route decorator (@routes.get("/path")) -- returns the function unchanged,
    since we're never actually starting a web server to register routes on."""

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

from tensorrt_wan.engine.base import TensorRTEngineWrapper  # noqa: E402

start_png, end_png, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
width = int(sys.argv[4]) if len(sys.argv) > 4 else 832
height = int(sys.argv[5]) if len(sys.argv) > 5 else 480
device = torch.device("cuda")
num_frames = 81


class TRTDiTWrapper(torch.nn.Module):
    """Drop-in replacement for a real WanModel as `BaseModel.diffusion_model` -- matches the call
    signature `apply_model()` (comfy/model_base.py) actually uses:
    `self.diffusion_model(xc, t, context=context, control=control,
    transformer_options=transformer_options, **extra_conds)`. `xc` arrives already
    channel-concatenated (noise+mask+image_latent) and already cast to `self.dtype` by
    `apply_model()` itself (via `get_dtype_inference()` reading this module's own `.dtype`).
    """

    def __init__(self, engine_path: str) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        # WAN22.concat_cond (comfy/model_base.py) introspects
        # `diffusion_model.patch_embedding.weight.shape[1]` to compute how many extra
        # conditioning channels it needs (36 - 16 = 20 = mask(4) + image_latent(16) for this I2V
        # checkpoint) -- only `.weight.shape[1]` is ever read, no real conv needed.
        self.patch_embedding = torch.nn.Conv3d(36, 1, kernel_size=1)
        self._wrapper = TensorRTEngineWrapper(engine_path, device=device)
        self._wrapper.load()

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **extra_conds):
        # Our exported DiT's `context` input has no dynamic axis (max_text_tokens=512, baked in
        # at export time) -- ComfyUI's own tokenizer may not pad to exactly that length by
        # default, so pad/truncate defensively rather than let TensorRT reject a mismatched shape.
        if context is not None and context.shape[1] != 512:
            if context.shape[1] < 512:
                pad = torch.zeros(
                    context.shape[0], 512 - context.shape[1], context.shape[2],
                    device=context.device, dtype=context.dtype,
                )
                context = torch.cat([context, pad], dim=1)
            else:
                context = context[:, :512]

        batch = x.shape[0]
        if batch == 1:
            out = self._wrapper.infer({"x": x, "timestep": timestep, "context": context})
            return out["noise_pred"]

        # ComfyUI's real CFG batches cond+uncond into one batch=N call by default; our engine's
        # batch dim was specialized to 1 at export time (torch.export traced it that way
        # regardless of Dim.AUTO -- see DiTExporter.dynamic_axes()'s docstring). Split into N
        # batch=1 calls and re-concatenate, matching exactly what this project's own
        # DiTEngine.denoise_step() already does for its own (separately-invoked) CFG passes.
        outputs = []
        for i in range(batch):
            t_i = timestep[i : i + 1] if timestep.shape[0] == batch else timestep
            c_i = context[i : i + 1] if context is not None else None
            out = self._wrapper.infer({"x": x[i : i + 1], "timestep": t_i, "context": c_i})
            outputs.append(out["noise_pred"])
        return torch.cat(outputs, dim=0)

    def unload(self) -> None:
        self._wrapper.unload()


def load_image_comfy(path: str) -> torch.Tensor:
    """ComfyUI IMAGE convention: (B, H, W, C) float in [0, 1], native resolution -- WanFMLFPluggable
    itself resizes via comfy.utils.common_upscale, so no manual resize/crop needed here.
    """
    img = Image.open(path).convert("RGB")
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(img.height, img.width, 3)
    return (arr.float() / 255.0).unsqueeze(0).to(device)


print(f"[{_elapsed()}] Resolution: {width}x{height} (WxH), {num_frames} frames")

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

print(f"[{_elapsed()}] Loading real model shells (for correct model_sampling/latent_format/concat_keys config)...")
high_path = folder_paths.get_full_path_or_raise("diffusion_models", "wan2.2_i2v_high_noise_14B_fp16.safetensors")
low_path = folder_paths.get_full_path_or_raise("diffusion_models", "wan2.2_i2v_low_noise_14B_fp16.safetensors")
high_model = comfy.sd.load_diffusion_model(high_path, model_options={})
low_model = comfy.sd.load_diffusion_model(low_path, model_options={})

print(f"[{_elapsed()}] Swapping in our TensorRT DiT engines...")
TRT_MODEL_DIR = "/workspace/runpod-slim/trtwan_model"
high_trt = TRTDiTWrapper(f"{TRT_MODEL_DIR}/dit_high_noise.engine")
low_trt = TRTDiTWrapper(f"{TRT_MODEL_DIR}/dit_low_noise.engine")
high_model.model.diffusion_model = high_trt
low_model.model.diffusion_model = low_trt
_t_setup_done = time.monotonic()

print(f"[{_elapsed()}] Encoding text (real CLIP)...")
positive_tokens = clip.tokenize("a green chair")
positive = clip.encode_from_tokens_scheduled(positive_tokens)
negative_tokens = clip.tokenize("")
negative = clip.encode_from_tokens_scheduled(negative_tokens)

print(f"[{_elapsed()}] Building conditioning (real WanFMLFPluggable, no LoRA/clip_vision)...")
start_image = load_image_comfy(start_png)
end_image = load_image_comfy(end_png)
positive, negative, latent = WanFMLFPluggable.execute(
    positive=positive, negative=negative, vae=vae,
    width=width, height=height, length=num_frames,
    start_image=start_image, end_image=end_image,
)
_t_cond_done = time.monotonic()

print(f"[{_elapsed()}] Running real two_phase_sampler (steps=12, high_cfg=1.8, low_cfg=1.1, euler/sgm_uniform)...")
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

high_trt.unload()
low_trt.unload()
torch.cuda.empty_cache()

print(f"[{_elapsed()}] Decoding (real VAE)...")
# vae.decode()'s internal tiled-decode path computes a fresh output tensor, then applies
# comfy.sd.VAE's own process_output in-place (.add_/.div_/.clamp_) -- that fresh tensor is itself
# an inference tensor (this whole process appears to run under a persistent/global inference-mode
# state, not just a scoped context -- cloning the *input* samples didn't help, since the error is
# on a tensor computed fresh internally, not on our input). Monkeypatch process_output to clone
# right before ComfyUI's own in-place chain runs, sidestepping this regardless of where the
# inference-tensor taint actually originates.
_original_process_output = vae.process_output
vae.process_output = lambda image: _original_process_output(image.clone())

samples = result[0]["samples"].clone()
pixels = vae.decode(samples)  # (B, T, H, W, C) in [0, 1], ComfyUI convention

frames_np = (pixels.clamp(0, 1) * 255).to(torch.uint8).numpy(force=True)
if frames_np.ndim == 5:
    frames_np = frames_np[0]
print(f"frames: shape={frames_np.shape} mean={frames_np.mean():.2f} "
      f"per-frame std: {[round(float(frames_np[t].std()), 2) for t in range(0, frames_np.shape[0], 10)]}")

import imageio.v3 as iio

iio.imwrite(out_path, frames_np, fps=16, codec="libx264")
_t_end = time.monotonic()
print(f"saved: {out_path}")
print(
    f"\nTiming ({width}x{height}, {num_frames} frames, 12 steps):\n"
    f"  setup (load CLIP/VAE/model shells/TRT engines): {_t_setup_done - _t_start:6.1f}s\n"
    f"  text encode + conditioning build:               {_t_cond_done - _t_setup_done:6.1f}s\n"
    f"  sampling (12 steps, both experts):               {_t_sample_done - _t_cond_done:6.1f}s\n"
    f"  decode + save:                                   {_t_end - _t_sample_done:6.1f}s\n"
    f"  TOTAL:                                            {_t_end - _t_start:6.1f}s"
)
