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
        min_height: int = 256,
        max_height: int = 1088,
        min_width: int = 256,
        max_width: int = 1088,
        static: bool = False,
    ) -> None:
        super().__init__(model, static=static)
        self.latent_channels = latent_channels
        self.frames = frames
        self.height = height
        self.width = width
        # Narrower than DiT's [256, 1280] -- confirmed via a real OOM (2026-08-08) that this
        # engine's execution context sizes scratch memory for the profile's *worst-case* bound at
        # context-creation time, not the actual runtime shape requested (same failure class
        # runpod_setup.md already documented for a wide vae_decoder profile). [256, 1280] requested
        # 103GiB and failed at *every* shape, including the previously-working 480x832 opt point --
        # this project's DiT does not have this problem at the same [32,160]-latent range (directly
        # confirmed working), so it's specific to this VAE architecture/TensorRT's scratch-sizing
        # for it, not a general dynamic-profile issue. 1088 covers both real target shapes
        # (480x832, 720x1088); narrow further if this still OOMs. See
        # docs/wan2.2_i2v_14b_notes.md, 2026-08-08.
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

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
        if self.static:
            return {}
        return {
            "pixels": [
                DynamicAxis(name="dim3", min=self.min_height, opt=self.height, max=self.max_height),
                DynamicAxis(name="dim4", min=self.min_width, opt=self.width, max=self.max_width),
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
        min_latent_height: int = 32,
        max_latent_height: int = 136,
        min_latent_width: int = 32,
        max_latent_width: int = 136,
        static: bool = False,
    ) -> None:
        super().__init__(model, static=static)
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        # Narrower than DiT's [32, 160] latent range -- confirmed via a real OOM (2026-08-08) that
        # this VAE's execution context sizes scratch for the profile's worst-case bound at
        # context-creation time regardless of actual runtime shape (same class of failure
        # runpod_setup.md already documented; DiT does not have this problem at the wider range,
        # confirmed working directly, so it's VAE-specific). [32,160] (1280px) requested 103GiB and
        # OOM'd at *every* shape including the previously-working 480x832 opt point. 136 (1088px)
        # covers both real target shapes (480x832, 720x1088); narrow further if this still OOMs.
        # See docs/wan2.2_i2v_14b_notes.md, 2026-08-08.
        self.min_latent_height = min_latent_height
        self.max_latent_height = max_latent_height
        self.min_latent_width = min_latent_width
        self.max_latent_width = max_latent_width

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
        if self.static:
            return {}
        return {
            "latent": [
                DynamicAxis(
                    name="dim3", min=self.min_latent_height, opt=self.latent_height,
                    max=self.max_latent_height,
                ),
                DynamicAxis(
                    name="dim4", min=self.min_latent_width, opt=self.latent_width,
                    max=self.max_latent_width,
                ),
            ]
        }

    @property
    def input_names(self) -> list[str]:
        return ["latent"]

    @property
    def output_names(self) -> list[str]:
        return ["pixels"]
