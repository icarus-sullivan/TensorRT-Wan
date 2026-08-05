from __future__ import annotations

import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class VAEEncoderExporter(ModelExporter):
    name = "vae_encoder"

    def __init__(self, model: torch.nn.Module, latent_channels: int, height: int = 480, width: int = 832) -> None:
        super().__init__(model)
        self.latent_channels = latent_channels
        self.height = height
        self.width = width

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {"pixels": torch.zeros(1, 3, self.height, self.width, device=self.device, dtype=self.dtype)}

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        return {
            "pixels": [
                DynamicAxis(name="dim0", min=1, opt=1, max=4),
                DynamicAxis(name="dim2", min=64, opt=self.height, max=1920),
                DynamicAxis(name="dim3", min=64, opt=self.width, max=1920),
            ]
        }

    @property
    def input_names(self) -> list[str]:
        return ["pixels"]

    @property
    def output_names(self) -> list[str]:
        return ["latent"]


class VAEDecoderExporter(ModelExporter):
    name = "vae_decoder"

    def __init__(
        self,
        model: torch.nn.Module,
        latent_channels: int,
        latent_frames: int = 21,
        latent_height: int = 60,
        latent_width: int = 104,
    ) -> None:
        super().__init__(model)
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "latent": torch.zeros(
                1, self.latent_channels, self.latent_frames, self.latent_height, self.latent_width,
                device=self.device, dtype=self.dtype,
            )
        }

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        return {
            "latent": [
                DynamicAxis(name="dim0", min=1, opt=1, max=4),
                DynamicAxis(name="dim2", min=1, opt=self.latent_frames, max=self.latent_frames * 2),
                DynamicAxis(name="dim3", min=32, opt=self.latent_height, max=self.latent_height * 2),
                DynamicAxis(name="dim4", min=32, opt=self.latent_width, max=self.latent_width * 2),
            ]
        }

    @property
    def input_names(self) -> list[str]:
        return ["latent"]

    @property
    def output_names(self) -> list[str]:
        return ["pixels"]
