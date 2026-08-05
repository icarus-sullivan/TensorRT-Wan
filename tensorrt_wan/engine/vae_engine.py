from __future__ import annotations

from pathlib import Path

import torch

from tensorrt_wan.engine.base import TensorRTEngineWrapper


class VAEEncoderEngine:
    """Encodes pixel-space frames to Wan's latent space. Used by I2V, V2V, and future editing workflows."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
        torch_fallback: torch.nn.Module | None = None,
    ) -> None:
        self._wrapper = TensorRTEngineWrapper(engine_path, device=device, torch_fallback=torch_fallback)

    def load(self) -> None:
        self._wrapper.load()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """`image`: (B, C, H, W) in [-1, 1]. Returns a single-frame latent."""
        return self._wrapper.infer({"pixels": image})["latent"]

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """`frames`: (B, C, T, H, W) in [-1, 1]. Returns the full video latent."""
        return self._wrapper.infer({"pixels": frames})["latent"]


class VAEDecoderEngine:
    """Decodes denoised latents back to pixel-space video frames — the final pipeline stage."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
        torch_fallback: torch.nn.Module | None = None,
    ) -> None:
        self._wrapper = TensorRTEngineWrapper(engine_path, device=device, torch_fallback=torch_fallback)

    def load(self) -> None:
        self._wrapper.load()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """`latents`: (B, C, T, H, W). Returns pixel-space frames in [-1, 1]."""
        return self._wrapper.infer({"latent": latents})["pixels"]
