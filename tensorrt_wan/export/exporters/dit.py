from __future__ import annotations

import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class DiTExporter(ModelExporter):
    """Exports the unified Wan DiT backbone.

    Argument names/shapes match `WanModel.forward(x, timestep, context, ...)` exactly (`x`,
    `timestep`, `context` — not `latents`/`text`) — confirmed against a real Wan 2.2 14B I2V
    checkpoint on RunPod hardware, see docs/wan2.2_i2v_14b_notes.md.

    `in_channels` is the DiT's actual input channel count, not the VAE's latent channel count:
    Wan I2V-capable checkpoints channel-concatenate noise-latent + image-latent + mask before
    patch embedding (36 = 16 + 16 + 4 for the checkpoint this was verified against), so
    `in_channels` != the VAE's own latent channel count (16) for those checkpoints. A pure
    T2V-only architecture would use 16 here instead. `text_dim`/`patch_size` are likewise read
    from the loaded checkpoint's own config (not hardcoded) so this exporter works unmodified
    across Wan releases with different dimensions — only the numbers passed in change.
    """

    name = "dit"

    def __init__(
        self,
        model: torch.nn.Module,
        in_channels: int,
        text_dim: int,
        max_text_tokens: int = 512,
        latent_frames: int = 21,
        latent_height: int = 60,
        latent_width: int = 104,
    ) -> None:
        super().__init__(model)
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.max_text_tokens = max_text_tokens
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "x": torch.zeros(
                1, self.in_channels, self.latent_frames, self.latent_height, self.latent_width,
                device=self.device, dtype=self.dtype,
            ),
            "timestep": torch.zeros(1, device=self.device, dtype=self.dtype),
            "context": torch.zeros(
                1, self.max_text_tokens, self.text_dim, device=self.device, dtype=self.dtype
            ),
        }

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        # Batch (dim0 on x/timestep/context) is deliberately absent: confirmed via a real
        # torch.export + TensorRT build attempt that this model's traced code specializes batch
        # to a fixed value regardless of Dim.AUTO (see docs/wan2.2_i2v_14b_notes.md) — declaring
        # a profile range for a dimension the ONNX graph doesn't actually mark dynamic makes
        # TensorRT's builder reject the profile outright ("Dimension mismatch ... profile has
        # min=1,opt=1,max=4 but tensor has 1"). timestep/context end up fully static once their
        # only dynamic candidate (batch) specializes, so they're omitted entirely here rather
        # than given an empty axis list.
        return {
            "x": [
                DynamicAxis(name="dim2", min=1, opt=self.latent_frames, max=self.latent_frames * 2),
                DynamicAxis(name="dim3", min=32, opt=self.latent_height, max=self.latent_height * 2),
                DynamicAxis(name="dim4", min=32, opt=self.latent_width, max=self.latent_width * 2),
            ],
        }

    @property
    def input_names(self) -> list[str]:
        return ["x", "timestep", "context"]

    @property
    def output_names(self) -> list[str]:
        return ["noise_pred"]
