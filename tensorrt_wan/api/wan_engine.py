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
from tensorrt_wan.conditioning.types import ConditioningKind
from tensorrt_wan.engine.dit_engine import DiTEngine
from tensorrt_wan.engine.text_encoder_engine import TextEncoderEngine
from tensorrt_wan.engine.vae_engine import VAEDecoderEngine, VAEEncoderEngine
from tensorrt_wan.runtime.manager import RuntimeManager
from tensorrt_wan.scheduler.flow_match import FlowMatchEulerScheduler
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


class WanEngine:
    def __init__(
        self,
        runtime: RuntimeManager,
        model_config: WanModelConfig,
        text_encoder: TextEncoderEngine,
        dit: DiTEngine,
        vae_encoder: VAEEncoderEngine,
        vae_decoder: VAEDecoderEngine,
        device: torch.device,
    ) -> None:
        self.runtime = runtime
        self.model_config = model_config
        self.text_encoder = text_encoder
        self.dit = dit
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.device = device
        self.scheduler = FlowMatchEulerScheduler()

        self.conditioning_manager = ConditioningManager()
        self.conditioning_manager.register(TextConditioningSource(text_encoder))
        self.conditioning_manager.register(ImageConditioningSource(vae_encoder, ConditioningKind.IMAGE))
        self.conditioning_manager.register(ImageConditioningSource(vae_encoder, ConditioningKind.FIRST_FRAME))

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
        `text_encoder.engine`, `dit.engine`, `vae_encoder.engine`, `vae_decoder.engine`. This
        raises `FileNotFoundError` until those exist — this repository's development phase
        produces the exporters and this loader, not built engines (see docs/engine_generation.md).
        """
        model_dir = Path(model_dir)
        config = WanModelConfig.load(model_dir / "wan_model.json")

        runtime_config = runtime_config or TensorRTWanConfig()
        runtime_config.precision.mode = precision
        runtime = RuntimeManager(runtime_config)
        device = device or torch.device(f"cuda:{runtime.primary_gpu.index}" if runtime.primary_gpu else "cuda")

        tokenizer = tokenizer or load_default_tokenizer(config.tokenizer_name)

        text_encoder = TextEncoderEngine(model_dir / "text_encoder.engine", tokenizer, device=device)
        dit = DiTEngine(model_dir / "dit.engine", device=device)
        vae_encoder = VAEEncoderEngine(model_dir / "vae_encoder.engine", device=device)
        vae_decoder = VAEDecoderEngine(model_dir / "vae_decoder.engine", device=device)

        for component in (text_encoder, dit, vae_encoder, vae_decoder):
            component.load()

        return cls(runtime, config, text_encoder, dit, vae_encoder, vae_decoder, device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        image: torch.Tensor | None = None,
        num_frames: int | None = None,
        resolution: tuple[int, int] | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int | None = None,
    ) -> VideoOutput:
        """Run text-to-video (or image-to-video, if `image` is given) generation.

        `image` conditions the first frame (I2V) when provided; T2V otherwise. Video-to-video
        and future editing workflows follow the same shape — encode the extra conditioning via
        `self.conditioning_manager` and pass it through `inputs` below — and are added by
        extending `inputs`, not by adding a parallel code path.
        """
        num_frames = num_frames or self.model_config.default_num_frames
        height, width = resolution or self.model_config.default_resolution

        inputs: dict[ConditioningKind, object] = {ConditioningKind.TEXT: prompt}
        if image is not None:
            inputs[ConditioningKind.FIRST_FRAME] = image.to(self.device)
        conditioning = self.conditioning_manager.combine(inputs)

        latents = self._initial_latents(num_frames, height, width, seed)
        latents = self.dit.generate(
            latents, conditioning, self.scheduler, num_inference_steps, guidance_scale
        )

        pixels = self.vae_decoder.decode(latents)  # (B, C, T, H, W) in [-1, 1]
        frames = ((pixels.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        frames = frames[0].permute(1, 2, 3, 0).contiguous()  # (T, H, W, C)
        return VideoOutput(frames=frames, fps=self.model_config.fps)

    def _initial_latents(self, num_frames: int, height: int, width: int, seed: int | None) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        latent_t = (num_frames - 1) // self.model_config.vae_temporal_scale + 1
        latent_h = height // self.model_config.vae_spatial_scale
        latent_w = width // self.model_config.vae_spatial_scale
        shape = (1, self.model_config.latent_channels, latent_t, latent_h, latent_w)
        return torch.randn(shape, generator=generator, device=self.device)


class _HFTokenizerAdapter:
    """Adapts a HF tokenizer's `__call__` to `TokenizerLike`: `prompt -> dict[str, Tensor]`."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    def __call__(self, prompt: str) -> dict[str, torch.Tensor]:
        return dict(self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True))


def load_default_tokenizer(tokenizer_name: str) -> _HFTokenizerAdapter:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            f"No tokenizer passed to WanEngine.from_pretrained() and 'transformers' is not "
            f"installed to auto-load {tokenizer_name!r}. Install transformers or pass tokenizer=..."
        ) from exc
    return _HFTokenizerAdapter(AutoTokenizer.from_pretrained(tokenizer_name))
