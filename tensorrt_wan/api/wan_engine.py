"""WanEngine: the standalone Python API's single entry point.

Composes the same pieces the ComfyUI nodes use (`RuntimeManager`, `ConditioningManager`,
`TextEncoderEngine`, `DiTEngine`, `VAEEncoderEngine`/`VAEDecoderEngine`) — the ComfyUI package
is a thin node wrapper around this class, not a parallel implementation. See
docs/architecture.md for how these pieces fit together.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tensorrt_wan.api.model_config import WanModelConfig
from tensorrt_wan.api.video_output import VideoOutput
from tensorrt_wan.config.schema import PrecisionMode, TensorRTWanConfig
from tensorrt_wan.conditioning.manager import ConditioningManager
from tensorrt_wan.conditioning.sources import ImageConditioningSource, TextConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, UnifiedConditioning
from tensorrt_wan.engine.dit_engine import DiTEngine
from tensorrt_wan.engine.text_encoder_engine import TextEncoderEngine
from tensorrt_wan.engine.vae_engine import VAEDecoderEngine, VAEEncoderEngine
from tensorrt_wan.runtime.manager import RuntimeManager
from tensorrt_wan.scheduler.flow_match import FlowMatchEulerScheduler
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

# Wan 2.1/2.2's (16-channel VAE, z_dim=16) per-channel latent normalization -- confirmed against
# real ComfyUI source (`comfy/latent_formats.py`'s `Wan21.latents_mean`/`latents_std`, used via
# `process_latent_in`/`process_latent_out`), not derived or guessed. Real bug, found and fixed
# 2026-08-07 (see docs/wan2.2_i2v_14b_notes.md): this project never applied this normalization
# anywhere -- every `generate()` call fed the DiT raw, unnormalized VAE-encoder output as image
# conditioning, and fed the raw VAE decoder unnormalized DiT output latents, despite ComfyUI's own
# pipeline applying it in two specific places: `WAN21.concat_cond` (`comfy/model_base.py`) applies
# `process_latent_in` to the image-conditioning latent before channel-concatenating it into `x`,
# and `samplers.py`'s `inner_sample` applies `process_latent_out` to the *final* denoised latent
# before it's ever handed to the VAE decoder. The initial noise latent does NOT need this (flow
# matching draws it directly as unit normal, already in the same normalized space `process_in`
# maps real latents into -- confirmed via `inner_sample`'s own "don't shift the empty latent image"
# skip for an all-zero starting point, which is the T2V/I2V-from-scratch case this project uses).
_WAN21_LATENTS_MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]).view(1, 16, 1, 1, 1)
_WAN21_LATENTS_STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]).view(1, 16, 1, 1, 1)


def _wan_latent_process_in(latent: torch.Tensor) -> torch.Tensor:
    """VAE-native latent -> the normalized space the DiT was trained on. Apply to any real
    (VAE-encoded) latent used as DiT conditioning -- never to synthetic noise."""
    mean = _WAN21_LATENTS_MEAN.to(latent.device, latent.dtype)
    std = _WAN21_LATENTS_STD.to(latent.device, latent.dtype)
    return (latent - mean) / std


def _wan_latent_process_out(latent: torch.Tensor) -> torch.Tensor:
    """The DiT's normalized output space -> VAE-native latent. Apply to the final denoised latent
    before decoding, never mid-schedule."""
    mean = _WAN21_LATENTS_MEAN.to(latent.device, latent.dtype)
    std = _WAN21_LATENTS_STD.to(latent.device, latent.dtype)
    return latent * std + mean


class WanEngine:
    def __init__(
        self,
        runtime: RuntimeManager,
        model_config: WanModelConfig,
        text_encoder: TextEncoderEngine,
        dit_high_noise: DiTEngine,
        vae_encoder: VAEEncoderEngine,
        vae_decoder: VAEDecoderEngine,
        device: torch.device,
        *,
        dit_low_noise: DiTEngine | None = None,
    ) -> None:
        """`dit_low_noise=None` means single-expert mode: `dit_high_noise` runs the entire
        schedule, matching this project's original (pre-MoE) behavior and any `model_dir` that
        only has one `dit_high_noise.engine`/`dit.engine`. Wan 2.2 is a two-expert MoE (a
        `high_noise` expert for the early/high-sigma steps, a separate `low_noise` expert for the
        rest) -- see `generate()`'s docstring for the real switch rule, confirmed against
        ComfyUI's own "Image to Video (Wan 2.2)"/"Text to Video (Wan 2.2)" blueprints
        (docs/wan2.2_i2v_14b_notes.md, 2026-08-07 session) rather than assumed.
        """
        self.runtime = runtime
        self.model_config = model_config
        self.text_encoder = text_encoder
        self.dit_high_noise = dit_high_noise
        self.dit_low_noise = dit_low_noise
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.device = device
        self.scheduler = FlowMatchEulerScheduler()

        self.conditioning_manager = ConditioningManager()
        self.conditioning_manager.register(TextConditioningSource(text_encoder))
        self.conditioning_manager.register(ImageConditioningSource(vae_encoder, ConditioningKind.IMAGE))
        # First/last-frame I2V conditioning is built directly in generate() via
        # _build_image_to_video_conditioning, not through a registered ConditioningSource — it
        # needs the full target video length/resolution up front to match ComfyUI's real
        # gray-fill + single-vae-encode algorithm, which a per-kind ConditioningSource.encode()
        # call (one frame, no length) can't do. See that function's docstring.

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        precision: PrecisionMode = "auto",
        device: torch.device | None = None,
        runtime_config: TensorRTWanConfig | None = None,
        tokenizer: object | None = None,
    ) -> "WanEngine":
        """Load a Wan model whose engines were already built via `trtwan build engine` (or the
        ComfyUI "TensorRT Engine Builder" node) into `model_dir`.

        Expects `model_dir` to contain `wan_model.json` (see `WanModelConfig`) plus
        `text_encoder.engine`, `dit_high_noise.engine`, `vae_encoder.engine`,
        `vae_decoder.engine`, and optionally `dit_low_noise.engine` (Wan 2.2's second MoE expert
        — see `generate()`'s docstring; single-expert mode is used if it's absent, with a logged
        warning, since Wan 2.2 needs both experts for a coherent result). `dit.engine` is also
        accepted as a single-expert fallback name for a `model_dir` assembled before MoE support
        existed. This raises `FileNotFoundError` until the required files exist — this
        repository's development phase produces the exporters and this loader, not built engines
        (see docs/engine_generation.md).

        Does **not** load any engine's weights onto the GPU yet — `generate()` loads/unloads each
        component around its own stage of use (text_encoder, then vae_encoder, then dit for the
        whole denoising loop, then vae_decoder), since holding all four resident at once wastes
        tens of GiB of GPU memory nothing is using simultaneously. Call `.load()` on a specific
        component directly first if you need to use it outside `generate()`.
        """
        model_dir = Path(model_dir)
        config = WanModelConfig.load(model_dir / "wan_model.json")

        runtime_config = runtime_config or TensorRTWanConfig()
        runtime_config.precision.mode = precision
        runtime = RuntimeManager(runtime_config)
        device = device or torch.device(f"cuda:{runtime.primary_gpu.index}" if runtime.primary_gpu else "cuda")

        tokenizer = tokenizer or load_default_tokenizer(config.tokenizer_name, config.max_text_tokens)

        text_encoder = TextEncoderEngine(model_dir / "text_encoder.engine", tokenizer, device=device)
        # "dit_high_noise.engine" is the real MoE filename; "dit.engine" is accepted as a
        # single-expert fallback for any `model_dir` assembled before MoE support existed. Prefers
        # the former if both happen to be present.
        high_noise_path = model_dir / "dit_high_noise.engine"
        if not high_noise_path.exists():
            high_noise_path = model_dir / "dit.engine"
        dit_high_noise = DiTEngine(high_noise_path, device=device)
        low_noise_path = model_dir / "dit_low_noise.engine"
        dit_low_noise = DiTEngine(low_noise_path, device=device) if low_noise_path.exists() else None
        if dit_low_noise is None:
            logger.warning(
                "No dit_low_noise.engine found in %s -- running the entire denoising schedule on "
                "the high_noise expert. Wan 2.2 is a two-expert MoE; single-expert mode is a "
                "known-degraded fallback (see docs/wan2.2_i2v_14b_notes.md's 2026-08-07 session), "
                "not the intended configuration.",
                model_dir,
            )
        vae_encoder = VAEEncoderEngine(model_dir / "vae_encoder.engine", device=device)
        vae_decoder = VAEDecoderEngine(model_dir / "vae_decoder.engine", device=device)

        # Deliberately not loaded here -- each is ~tens of GiB (each dit expert ~28.6GiB,
        # text_encoder ~20.6GiB); holding every component resident at once when `generate()` only
        # ever uses one at a time (text_encoder, then vae_encoder, then one dit expert at a time,
        # then vae_decoder) wastes GPU memory for no benefit. `generate()` loads/unloads each
        # around its own stage of use instead. See docs/wan2.2_i2v_14b_notes.md's 2026-08-07 session.
        return cls(
            runtime, config, text_encoder, dit_high_noise, vae_encoder, vae_decoder, device,
            dit_low_noise=dit_low_noise,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        image: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        num_frames: int | None = None,
        resolution: tuple[int, int] | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        guidance_scale_low_noise: float | None = None,
        seed: int | None = None,
    ) -> VideoOutput:
        """Run text-to-video generation, or image-to-video if `image`/`last_image` are given.

        `image`/`last_image` condition the first/last frame (I2V) — either, both, or neither may
        be passed, matching ComfyUI's real `WanImageToVideo`/`WanFirstLastFrameToVideo` nodes (see
        `_build_image_to_video_conditioning`). Video-to-video and future editing workflows follow
        the text-conditioning shape — encode the extra conditioning via `self.conditioning_manager`
        and pass it through `inputs` below — and are added by extending `inputs`, not by adding a
        parallel code path.

        **Wan 2.2 MoE expert switching:** if `self.dit_low_noise` is set, the denoising loop runs
        the first half of `num_inference_steps` on `dit_high_noise` and the second half on
        `dit_low_noise` — confirmed against ComfyUI's own "Image to Video (Wan 2.2)"/"Text to Video
        (Wan 2.2)" blueprints (`blueprints/*.json`'s embedded subgraphs both chain two
        `KSamplerAdvanced` nodes split at exactly step `total/2`, e.g. `[0,2]`/`[2,4]` for a
        4-step schedule), not assumed. This is a step-count split, not a separately-computed sigma
        threshold, but it's step-count-*independent* in effect: `FlowMatchEulerScheduler.prepare`'s
        sigma schedule is `shift * linspace(1,0,N+1) / (...)`, and the raw `linspace` value at
        step-fraction 0.5 is always exactly 0.5 regardless of `N` — so splitting by step-fraction
        lands on the same sigma boundary (~0.889 at this project's default `shift=8.0`) no matter
        how many steps are requested. Each expert is loaded only while it's actually stepping and
        unloaded before the other loads, matching the load-what's-needed pattern used for
        text_encoder/vae_encoder/vae_decoder elsewhere in this method.

        `guidance_scale_low_noise`, if given, overrides `guidance_scale` for the `dit_low_noise`
        half only. By the time the low-noise expert takes over, the coarse structure/composition
        is already set (that's what the high-noise steps did) and remaining steps are mostly
        adding fine detail — over-guiding that phase with the same CFG strength used early can
        oversharpen/distort detail rather than help. Not sourced from a specific Wan 2.2 reference
        value (unlike the 50%-step switch point); a reasonable default hasn't been confirmed yet,
        so `None` keeps `guidance_scale` uniform across both experts rather than guessing one.
        """
        num_frames = num_frames or self.model_config.default_num_frames
        height, width = resolution or self.model_config.default_resolution

        self.text_encoder.load()
        conditioning = self.conditioning_manager.combine({ConditioningKind.TEXT: prompt})
        # Real empty-string encoding for the CFG unconditional pass, not an all-zero tensor --
        # see DiTEngine._null_conditioning's docstring for why. Built here (not left to
        # DiTEngine's zeroing fallback) because this is the one call site with a loaded
        # text_encoder in scope.
        null_text_embeds = self.text_encoder.encode_text("")
        self.text_encoder.unload()

        if image is not None or last_image is not None:
            self.vae_encoder.load()
            image_latent, mask = _build_image_to_video_conditioning(
                self.vae_encoder,
                first_frame=image.to(self.device) if image is not None else None,
                last_frame=last_image.to(self.device) if last_image is not None else None,
                num_frames=num_frames,
                height=height,
                width=width,
                model_config=self.model_config,
                device=self.device,
            )
            self.vae_encoder.unload()
            conditioning.embeddings[ConditioningKind.IMAGE_VIDEO.value] = image_latent
            conditioning.masks[ConditioningKind.IMAGE_VIDEO.value] = mask

        # Built after image conditioning is finalized above, sharing conditioning.masks by
        # reference (image/mask conditioning must be identical for both cond and uncond passes)
        # and copying embeddings only to swap in the null text embedding.
        uncond_conditioning = UnifiedConditioning(
            embeddings={**conditioning.embeddings, ConditioningKind.TEXT.value: null_text_embeds},
            masks=conditioning.masks,
            lora_weights=conditioning.lora_weights,
            metadata=conditioning.metadata,
        )

        latents = self._initial_latents(num_frames, height, width, seed)
        latents = self._denoise(
            latents, conditioning, uncond_conditioning, num_inference_steps, guidance_scale,
            guidance_scale_low_noise if guidance_scale_low_noise is not None else guidance_scale,
        )

        # Real, previously-missing normalization -- see _wan_latent_process_out's docstring.
        # `latents` here is the DiT's raw output, in the normalized space it was trained on;
        # `samplers.py`'s real sampling loop always applies this before decoding, we never did.
        latents = _wan_latent_process_out(latents)

        self.vae_decoder.load()
        pixels = self.vae_decoder.decode(latents)  # (B, C, T, H, W) in [-1, 1]
        self.vae_decoder.unload()
        frames = ((pixels.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        frames = frames[0].permute(1, 2, 3, 0).contiguous()  # (T, H, W, C)
        return VideoOutput(frames=frames, fps=self.model_config.fps)

    def _denoise(
        self,
        latents: torch.Tensor,
        conditioning: UnifiedConditioning,
        uncond_conditioning: UnifiedConditioning,
        num_inference_steps: int,
        guidance_scale_high_noise: float,
        guidance_scale_low_noise: float,
    ) -> torch.Tensor:
        """Run the full denoising loop, switching from `dit_high_noise` to `dit_low_noise` (and
        from `guidance_scale_high_noise` to `guidance_scale_low_noise`) at the halfway step if
        `dit_low_noise` is loaded (see `generate()`'s docstring for the switch rule). Single-expert
        mode (`dit_low_noise is None`) just runs `dit_high_noise`/`guidance_scale_high_noise` for
        every step.
        """
        switch_step = num_inference_steps // 2
        state = self.scheduler.prepare(num_inference_steps, self.device)

        self.dit_high_noise.load()
        active, guidance_scale = self.dit_high_noise, guidance_scale_high_noise
        while not state.done:
            if self.dit_low_noise is not None and state.step_index == switch_step:
                self.dit_high_noise.unload()
                self.dit_low_noise.load()
                active, guidance_scale = self.dit_low_noise, guidance_scale_low_noise
            timestep = state.current_timestep
            noise_pred = active.denoise_step(
                latents, timestep, conditioning, guidance_scale, uncond_conditioning
            )
            latents = self.scheduler.step(state, noise_pred, latents)
        active.unload()

        return latents

    def _initial_latents(self, num_frames: int, height: int, width: int, seed: int | None) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        latent_t = (num_frames - 1) // self.model_config.vae_temporal_scale + 1
        latent_h = height // self.model_config.vae_spatial_scale
        latent_w = width // self.model_config.vae_spatial_scale
        shape = (1, self.model_config.latent_channels, latent_t, latent_h, latent_w)
        # Real bug, confirmed via a real generate() attempt (2026-08-07): torch.randn() with no
        # dtype defaults to float32, silently diverging from every other conditioning tensor in
        # this pipeline (image_latent/mask/text_embeds are all built fp16, matching
        # export.base.ModelExporter.dtype's project-wide fp16 convention). TensorRTEngineWrapper's
        # per-input dtype cast (engine/base.py) masks this for the TensorRT path (a real numeric
        # downcast, not the byte-reinterpretation bug fixed the night before), but it broke an
        # eager-model comparison outright (torch.cat silently promotes the concatenated `x` to
        # float32 -- self.time_embedding's fp16 weights then reject it: "mat1 and mat2 must have
        # the same dtype, but got Float and Half"). See docs/wan2.2_i2v_14b_notes.md.
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)


def _build_image_to_video_conditioning(
    vae_encoder: VAEEncoderEngine,
    *,
    first_frame: torch.Tensor | None,
    last_frame: torch.Tensor | None,
    num_frames: int,
    height: int,
    width: int,
    model_config: WanModelConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full-length image_latent/mask pair for I2V conditioning.

    Real algorithm, matching ComfyUI's own `WanFirstLastFrameToVideo.execute()`
    (`comfy_extras/nodes_wan.py`) and the equivalent custom-node path this project traced directly
    (`custom_nodes/spnxx/nodes/wan_fmlf_pluggable.py`'s `WanFMLFPluggable`, 2026-08-08 session):
    gray-fill a whole `num_frames`-long pixel video, place the real first/last frames at their
    positions, and VAE-encode the *entire padded video in one `encode_video()` call* -- not one
    `encode_image()` call per distinct content (gray/first/last) like this function used to. Wan's
    VAE is a causal 3D-conv VAE, so each output latent frame has a receptive field over neighboring
    pixel frames; encoding gray filler in total isolation discarded that shared context entirely.

    Previously blocked by a real TensorRT build limitation (a `vae_encoder` at `T=num_frames=81`
    was documented as failing to build at this resolution) -- retried 2026-08-08 and it built
    successfully (unclear why the earlier attempt failed; not investigated, just confirmed working
    via a fresh build+test). `vae_encoder` must now be built with a fixed `frames=num_frames`
    (the causal-conv trace specializes the frame count, it can't be made dynamic -- see
    `VAEEncoderExporter.dynamic_axes()`), so this function requires the caller's `vae_encoder`
    engine to have been built at exactly this `num_frames`; `encode_image()` (T=1) is no longer
    used here at all.

    Mask polarity: `1=needs generation, 0=already known` -- confirmed directly against
    `wan_fmlf_pluggable.py`'s real mask construction (`mask_base` starts all-ones, known positions
    set to `0.0`, "not masking data there"). This is the *opposite* of what this function used
    before (`1=known/0=to-generate`), which came from a previous session's docstring claim about
    stock `WanImageToVideo`/`concat_cond`'s *net* effect after an internal inversion -- never
    independently re-verified against that stock code path in this session, only against this
    custom node's directly-observed convention. See docs/wan2.2_i2v_14b_notes.md.
    """
    scale = model_config.vae_temporal_scale
    latent_t = (num_frames - 1) // scale + 1
    reference = first_frame if first_frame is not None else last_frame
    assert reference is not None

    video = torch.zeros(1, 3, num_frames, height, width, device=device, dtype=reference.dtype)
    known_first = first_frame is not None
    known_last = last_frame is not None
    if known_first:
        video[:, :, 0] = first_frame
    if known_last:
        video[:, :, -1] = last_frame

    image_latent = vae_encoder.encode_video(video)  # (B, C, latent_t, h, w)
    h, w = image_latent.shape[-2], image_latent.shape[-1]

    mask = torch.ones(1, scale, latent_t, h, w, device=device, dtype=image_latent.dtype)
    if known_first:
        mask[:, :, 0] = 0.0
    if known_last:
        mask[:, :, -1] = 0.0

    # Real, previously-missing normalization -- see _wan_latent_process_in's docstring. This is
    # raw VAE-encoder output at this point; the DiT was trained on ComfyUI's normalized latent
    # space (`WAN21.concat_cond` applies exactly this before channel-concatenating), not the VAE's
    # native scale.
    image_latent = _wan_latent_process_in(image_latent)

    return image_latent, mask


class _HFTokenizerAdapter:
    """Adapts a HF tokenizer's `__call__` to `TokenizerLike`: `prompt -> dict[str, Tensor]`.

    Pads to a fixed `max_tokens` (`padding="max_length"`), not just the batch's longest sequence
    (`padding=True`) — the built DiT engine's `context` input has no dynamic axis at all
    (`DiTExporter.dynamic_axes()` only covers `x`), so it's a hard fixed-length requirement, not a
    convenience default. See `WanModelConfig.max_text_tokens`'s docstring.
    """

    def __init__(self, tokenizer: object, max_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def __call__(self, prompt: str) -> dict[str, torch.Tensor]:
        return dict(
            self._tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self._max_tokens,
                truncation=True,
            )
        )


def load_default_tokenizer(tokenizer_name: str, max_tokens: int) -> _HFTokenizerAdapter:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            f"No tokenizer passed to WanEngine.from_pretrained() and 'transformers' is not "
            f"installed to auto-load {tokenizer_name!r}. Install transformers or pass tokenizer=..."
        ) from exc
    return _HFTokenizerAdapter(AutoTokenizer.from_pretrained(tokenizer_name), max_tokens)
