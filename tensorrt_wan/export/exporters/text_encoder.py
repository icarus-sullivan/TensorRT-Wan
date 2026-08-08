from __future__ import annotations

import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class TextEncoderExporter(ModelExporter):
    """Exports a Wan text encoder (T5/UMT5-family). `hidden_dim`/`max_tokens` come from the
    loaded model's own config rather than being hardcoded here, since they vary across Wan
    releases and encoder choices.
    """

    name = "text_encoder"

    def __init__(
        self, model: torch.nn.Module, hidden_dim: int, max_tokens: int = 512, static: bool = False
    ) -> None:
        super().__init__(model, static=static)
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros(1, self.max_tokens, dtype=torch.int64, device=self.device),
            "attention_mask": torch.ones(1, self.max_tokens, dtype=torch.int64, device=self.device),
        }

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        # Batch (dim0) deliberately omitted, same finding as DiTExporter.dynamic_axes(): a real
        # TensorRT build failed with "Dimension mismatch ... profile has min=1,opt=1,max=8 but
        # tensor has 1" — torch.export specializes this model's batch dim to a fixed value
        # regardless of Dim.AUTO (confirmed by the exporter's own "0/1 specialized" warning at
        # export time), so declaring a profile range for a dimension the ONNX graph doesn't
        # actually mark dynamic makes the builder reject the profile outright. See
        # docs/wan2.2_i2v_14b_notes.md.
        if self.static:
            return {}
        seq_axis = [DynamicAxis(name="dim1", min=1, opt=self.max_tokens, max=self.max_tokens)]
        return {"input_ids": seq_axis, "attention_mask": seq_axis}

    @property
    def input_names(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    @property
    def output_names(self) -> list[str]:
        return ["text_embeds"]
