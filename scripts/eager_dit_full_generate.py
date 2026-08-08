"""Real I2V generate(), but the DiT denoising loop runs eager PyTorch (load_dit(), no export, no
TensorRT) instead of the built TensorRT engines -- text_encoder and vae_encoder/vae_decoder stay
on TensorRT (already independently confirmed correct: the text embedding path hasn't been
questioned, and the VAE round-trip was directly verified this session).

Purpose: every attempt to fix "generate() produces pure noise" by tuning steps/CFG/expert-switching
has failed. Before hunting further in TensorRT-specific territory, this checks whether the *model
itself*, run natively at real (non-zero) timesteps with real conditioning, converges to anything
coherent at all -- if eager ALSO produces noise, the bug is upstream of TensorRT entirely (scheduler
math, conditioning construction, or the model call itself), not a TensorRT execution gap. If eager
IS coherent, that pins the bug specifically to TensorRT's DiT execution at real (non-trivial-input)
generation conditions -- surprising, given the trivial-input eager-vs-TensorRT comparison matched
closely, but every one of those comparisons used timestep=0, never the real high-noise regime.

Usage: python3 eager_dit_full_generate.py <model_dir> <high_noise_ckpt> <low_noise_ckpt> <start_png> <end_png> <out_mp4> [vae_encoder_t81_engine]
"""
import sys

import torch
from PIL import Image

sys.path.insert(0, "/workspace/runpod-slim/TensorRT-Wan")
_original_sdpa = torch.nn.functional.scaled_dot_product_attention

from examples.loaders.wan_comfyui_loader import load_dit  # noqa: E402
from tensorrt_wan.api.model_config import WanModelConfig
from tensorrt_wan.engine.text_encoder_engine import TextEncoderEngine
from tensorrt_wan.engine.vae_engine import VAEDecoderEngine, VAEEncoderEngine
from tensorrt_wan.scheduler.flow_match import FlowMatchEulerScheduler

(model_dir, high_ckpt, low_ckpt, start_png, end_png, out_path) = sys.argv[1:7]
vae_encoder_t81_path = sys.argv[7] if len(sys.argv) > 7 else None
device = torch.device("cuda")

config = WanModelConfig.load(f"{model_dir}/wan_model.json")
height, width = config.default_resolution
num_frames = config.default_num_frames
num_inference_steps = 20  # keep short -- this is a yes/no diagnostic, not a quality run
guidance_scale = 3.0


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    target_aspect = width / height
    if src_w / src_h > target_aspect:
        new_w = round(src_h * target_aspect)
        img = img.crop(((src_w - new_w) // 2, 0, (src_w - new_w) // 2 + new_w, src_h))
    else:
        new_h = round(src_w / target_aspect)
        img = img.crop((0, (src_h - new_h) // 2, src_w, (src_h - new_h) // 2 + new_h))
    img = img.resize((width, height), Image.LANCZOS)
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(height, width, 3)
    pixels = arr.permute(2, 0, 1).float() / 127.5 - 1.0
    return pixels.unsqueeze(0).to(device=device, dtype=torch.float16)


from tensorrt_wan.api.wan_engine import load_default_tokenizer  # noqa: E402

tokenizer = load_default_tokenizer(config.tokenizer_name, config.max_text_tokens)
text_encoder = TextEncoderEngine(f"{model_dir}/text_encoder.engine", tokenizer, device=device)
text_encoder.load()
text_embeds = text_encoder.encode_text("a green chair")
# Real CFG convention (SD3/Flux/Wan-style T5 text encoders): the unconditional pass uses a real
# *empty-string* encoding through the same tokenizer/embedding path (padded like any other
# prompt), not an all-zero tensor. An all-zero embedding is out-of-distribution input to
# cross-attention -- the model never saw one during training -- and CFG's
# `uncond + scale*(cond-uncond)` formula would then amplify whatever garbage that produces. This
# project's `_null_conditioning` (dit_engine.py) zeros the embedding instead; testing the real
# convention here before deciding whether to change it project-wide.
null_text_embeds = text_encoder.encode_text("")
text_encoder.unload()
print(f"text_embeds: shape={tuple(text_embeds.shape)} dtype={text_embeds.dtype}")

start = load_image(start_png)
end = load_image(end_png)
scale = config.vae_temporal_scale
latent_t = (num_frames - 1) // scale + 1

if vae_encoder_t81_path:
    # Real algorithm (comfy custom_nodes/spnxx/nodes/wan_fmlf_pluggable.py, WanFMLFPluggable):
    # gray-fill the *entire* pixel-space video, place start/end frames at their real positions,
    # and VAE-encode the whole padded video in one call -- not one encode_image() call per
    # distinct content (gray/start/end) like this project's own _build_image_to_video_conditioning
    # does. Wan's VAE is a causal 3D-conv VAE, so each output latent frame has a receptive field
    # over neighboring pixel frames; encoding gray filler in total isolation (this project's
    # existing approach) discards that shared context entirely, not just "a faint trace" as
    # previously assumed. Only tractable now because a T=81 vae_encoder engine build, previously
    # documented as failing at this resolution even for much smaller T, succeeded this session
    # (see docs/wan2.2_i2v_14b_notes.md) -- unclear why the earlier attempt failed and this one
    # didn't; not investigated, just confirmed working via a fresh build+test this session.
    print("Using real whole-video single VAE encode (T=81 engine)")
    vae_encoder_full = VAEEncoderEngine(vae_encoder_t81_path, device=device)
    vae_encoder_full.load()
    video = torch.zeros(1, 3, num_frames, height, width, device=device, dtype=torch.float16)
    video[:, :, 0] = start  # start/end are (1, 3, H, W); this slice is also (1, 3, H, W)
    video[:, :, -1] = end
    image_latent_full = vae_encoder_full.encode_video(video)  # (1, C, latent_t, h, w)
    vae_encoder_full.unload()
    image_latent = image_latent_full
else:
    vae_encoder = VAEEncoderEngine(f"{model_dir}/vae_encoder.engine", device=device)
    vae_encoder.load()
    gray = torch.zeros(1, 3, height, width, device=device, dtype=torch.float16)
    gray_latent = vae_encoder.encode_image(gray)[:, :, 0]
    start_latent = vae_encoder.encode_image(start)[:, :, 0]
    end_latent = vae_encoder.encode_image(end)[:, :, 0]
    vae_encoder.unload()
    latent_frames_list = [gray_latent] * latent_t
    latent_frames_list[0] = start_latent
    latent_frames_list[-1] = end_latent
    image_latent = torch.stack(latent_frames_list, dim=2)  # (1, C, T, h, w)

# Confirmed against real ComfyUI source (comfy/latent_formats.py's Wan21.latents_mean/std, used
# via process_latent_in/process_latent_out) -- WAN21.concat_cond applies this to the
# image-conditioning latent before concatenation, and samplers.py applies the inverse to the final
# denoised latent before decode. This project never did either. See docs/wan2.2_i2v_14b_notes.md.
_LATENTS_MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]).view(1, 16, 1, 1, 1).to(device)
_LATENTS_STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]).view(1, 16, 1, 1, 1).to(device)

image_latent = (image_latent - _LATENTS_MEAN.to(image_latent.dtype)) / _LATENTS_STD.to(image_latent.dtype)
h, w = image_latent.shape[-2], image_latent.shape[-1]
# Flipped polarity (was 1=known/0=to-generate): the real custom node this session traced
# (wan_fmlf_pluggable.py) builds its mask starting all-ones, then sets *known* positions to 0.0 --
# "not masking data there" (0 = nothing masked/hidden = known content is visible; 1 = masked/
# hidden = needs generation). The prior 1=known assumption came from a previous session's docstring
# claim (dit_engine.py) about stock WanImageToVideo/concat_cond's *net* effect after an internal
# inversion, never independently re-verified this session. Testing the directly-observed polarity.
mask = torch.ones(1, scale, latent_t, h, w, device=device, dtype=image_latent.dtype)
mask[:, :, 0] = 0.0
mask[:, :, -1] = 0.0
print(f"image_latent: {tuple(image_latent.shape)}, mask: {tuple(mask.shape)}")

generator = torch.Generator(device=device).manual_seed(0)
latents = torch.randn(
    (1, config.latent_channels, latent_t, h, w), generator=generator, device=device, dtype=torch.bfloat16
)

# load_dit() hardcodes bf16 for the model itself (see its docstring); text_encoder/vae_encoder
# are still fp16 engines (only the DiT needed bf16 to fix the NaN bug), so their outputs need an
# explicit cast here before feeding the eager bf16 model, or every Linear call mismatches dtypes.
text_embeds = text_embeds.to(torch.bfloat16)
null_text_embeds = null_text_embeds.to(torch.bfloat16)
image_latent = image_latent.to(torch.bfloat16)
mask = mask.to(torch.bfloat16)

scheduler = FlowMatchEulerScheduler()
state = scheduler.prepare(num_inference_steps, device)
switch_step = num_inference_steps // 2

model = None
current_ckpt = None


def ensure_model(ckpt: str) -> torch.nn.Module:
    global model, current_ckpt
    if current_ckpt == ckpt:
        return model
    if model is not None:
        del model
        torch.cuda.empty_cache()
    model = load_dit(ckpt)
    # load_dit() unconditionally monkeypatches scaled_dot_product_attention to the export-only
    # decomposed reference form, which materializes the full (heads, seq, seq) attention matrix --
    # ~80GiB at this project's real self-attention scale, instant OOM in eager. Restore the real
    # native SDPA (flash/memory-efficient, no full-matrix materialization) after each load() call.
    # See scripts/eager_trivial_check.py, which hit the same thing earlier this session.
    torch.nn.functional.scaled_dot_product_attention = _original_sdpa
    current_ckpt = ckpt
    return model


while not state.done:
    ckpt = high_ckpt if state.step_index < switch_step else low_ckpt
    m = ensure_model(ckpt)
    timestep = state.current_timestep.reshape(1).to(torch.bfloat16)
    x = torch.cat([latents, mask, image_latent], dim=1)
    with torch.no_grad():
        cond_out = m(x=x, timestep=timestep, context=text_embeds)
        uncond_out = m(x=x, timestep=timestep, context=null_text_embeds)
    noise_pred = uncond_out + guidance_scale * (cond_out - uncond_out)
    nan_frac = torch.isnan(noise_pred.float()).float().mean().item()
    print(f"step={state.step_index} ckpt={'high' if ckpt == high_ckpt else 'low'} "
          f"t={float(timestep.item()):.1f} nan_frac={nan_frac:.3f} "
          f"pred_std={noise_pred.float().std().item():.4f}")
    latents = scheduler.step(state, noise_pred, latents)

del model
torch.cuda.empty_cache()

latents = latents * _LATENTS_STD.to(latents.dtype) + _LATENTS_MEAN.to(latents.dtype)

vae_decoder = VAEDecoderEngine(f"{model_dir}/vae_decoder.engine", device=device)
vae_decoder.load()
pixels = vae_decoder.decode(latents)
vae_decoder.unload()
frames = ((pixels.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)[0].permute(1, 2, 3, 0).contiguous()
frames_np = frames.numpy(force=True)
print(f"frames: shape={frames_np.shape} mean={frames_np.mean():.2f} "
      f"per-frame std: {[round(float(frames_np[t].std()), 2) for t in range(0, frames_np.shape[0], 10)]}")

import imageio.v3 as iio

iio.imwrite(out_path, frames_np, fps=config.fps, codec="libx264")
print(f"saved: {out_path}")
