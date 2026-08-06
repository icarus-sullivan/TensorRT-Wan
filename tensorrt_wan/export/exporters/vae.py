from __future__ import annotations

import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class VAEEncoderExporter(ModelExporter):
    """Exports the Wan VAE encoder as a single unified engine covering both single-image (I2V
    reference frame) and full-video (V2V) encoding, via a dynamic frame-count axis — same
    "one engine, every workflow" pattern PLAN.md requires for the DiT.

    `pixels` is always 5D `(B, 3, T, H, W)`, T dynamic starting at 1: matches the fact that
    Wan's VAE is a causal 3D-conv VAE operating on video tensors, where a single image is just
    T=1 (not a structurally different 4D case) — unconfirmed against ComfyUI's actual VAE module
    source in this environment (see docs/wan2.2_i2v_14b_notes.md), but this is what keeps
    `VAEEncoderEngine.encode_image`/`.encode_video` (engine/vae_engine.py) calling into the same
    engine instance instead of one built for a rank-4 input and one caller passing rank-5.
    """

    name = "vae_encoder"

    def __init__(
        self,
        model: torch.nn.Module,
        latent_channels: int,
        frames: int = 1,
        height: int = 480,
        width: int = 832,
    ) -> None:
        super().__init__(model)
        self.latent_channels = latent_channels
        self.frames = frames
        self.height = height
        self.width = width

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "pixels": torch.zeros(1, 3, self.frames, self.height, self.width, device=self.device, dtype=self.dtype)
        }

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        # Batch (dim0) *and* frame-count (dim2) deliberately omitted — confirmed via a real
        # torch.export run that both specialize to a fixed value regardless of Dim.AUTO, same
        # "0/1 specialized" warning DiTExporter's batch dim hit (see docs/wan2.2_i2v_14b_notes.md
        # and DiTExporter.dynamic_axes()'s docstring). Root cause here specifically:
        # `WanVAE.encode`'s chunked causal-conv loop (`comfy/ldm/wan/vae2_2.py`) iterates
        # `range(1 + (t - 1) // 4)`, a data-dependent trip count `torch.export` bakes in as a
        # constant rather than keeping symbolic — this export is only valid for the traced
        # `frames` value (T=1 by default, i.e. single-image/I2V-reference-frame encoding), not a
        # genuinely T-dynamic engine. A different `frames` value needs its own export/engine,
        # same "static per profile" strategy PLAN.md already treats as first-class. Only
        # height/width remain dynamic, matching what's actually true of the exported graph.
        return {
            "pixels": [
                DynamicAxis(name="dim3", min=64, opt=self.height, max=1920),
                DynamicAxis(name="dim4", min=64, opt=self.width, max=1920),
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
        # Batch (dim0) *and* frame-count (dim2) omitted, by analogy with VAEEncoderExporter's
        # confirmed finding above: `WanVAE.decode` (comfy/ldm/wan/vae2_2.py) has the same
        # data-dependent chunked-loop shape (`for i in range(z.shape[2])`) as `.encode` does, so
        # the same specialization is expected here too. Not independently re-confirmed via a
        # decoder-specific export run (encoder's was) — if this turns out wrong, the real
        # symptom would be TensorRT's "Dimension mismatch ... profile has min=...,max=... but
        # tensor has N" error, same as every other instance of this finding in this file. See
        # docs/wan2.2_i2v_14b_notes.md.
        return {
            "latent": [
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
