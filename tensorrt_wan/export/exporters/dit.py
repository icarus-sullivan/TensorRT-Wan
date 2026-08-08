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
        min_latent_height: int = 32,
        max_latent_height: int = 160,
        min_latent_width: int = 32,
        max_latent_width: int = 160,
        static: bool = False,
    ) -> None:
        super().__init__(model, static=static)
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.max_text_tokens = max_text_tokens
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        # Pixel-space [256, 1280] / vae_spatial_scale(8) = latent [32, 160] -- confirmed range
        # 2026-08-08: covers every resolution named so far (480x832 today's built shape, 720x1088
        # the real I2V source images, 960x1248 a stated future target), all divisible by 16 per
        # Wan's hard requirement. `opt` stays at today's known-good 480x832
        # (latent_height=60/latent_width=104) so that shape keeps its best-tuned tactics; other
        # shapes within [min, max] still work, just not as optimally tuned as the profile's own
        # opt point. See docs/wan2.2_i2v_14b_notes.md.
        self.min_latent_height = min_latent_height
        self.max_latent_height = max_latent_height
        self.min_latent_width = min_latent_width
        self.max_latent_width = max_latent_width

    @property
    def dtype(self) -> torch.dtype:
        """Always `bf16`, ignoring `TRTWAN_LOADER_DTYPE` -- overrides `ModelExporter.dtype`'s
        env-var-driven default. Must match `examples/loaders/wan_comfyui_loader.py`'s `load_dit()`,
        which hardcodes the same thing for the same reason: a `fp16` DiT at this project's real
        target scale (~32,760-token self-attention) returns 100% NaN on every input, confirmed via
        a full bisection (docs/wan2.2_i2v_14b_notes.md, 2026-08-07 session) that ruled out every
        other candidate (attention decomposition, TensorRT's NaN strictness, the modulation/gate
        path). If this diverges from `load_dit`'s own dtype, `example_inputs()` builds tensors in
        one dtype against a model loaded in another -- exactly the failure `ModelExporter.dtype`'s
        base docstring already describes for the untargeted case.
        """
        return torch.bfloat16

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
        if self.static:
            return {}
        return {
            "x": [
                DynamicAxis(name="dim2", min=1, opt=self.latent_frames, max=self.latent_frames * 2),
                DynamicAxis(
                    name="dim3", min=self.min_latent_height, opt=self.latent_height,
                    max=self.max_latent_height,
                ),
                DynamicAxis(
                    name="dim4", min=self.min_latent_width, opt=self.latent_width,
                    max=self.max_latent_width,
                ),
            ],
        }

    @property
    def input_names(self) -> list[str]:
        return ["x", "timestep", "context"]

    @property
    def output_names(self) -> list[str]:
        return ["noise_pred"]
