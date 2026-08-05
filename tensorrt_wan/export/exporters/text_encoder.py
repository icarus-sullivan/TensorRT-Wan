from __future__ import annotations

import torch

from tensorrt_wan.export.base import DynamicAxis, ModelExporter


class TextEncoderExporter(ModelExporter):
    """Exports a Wan text encoder (T5/UMT5-family). `hidden_dim`/`max_tokens` come from the
    loaded model's own config rather than being hardcoded here, since they vary across Wan
    releases and encoder choices.
    """

    name = "text_encoder"

    def __init__(self, model: torch.nn.Module, hidden_dim: int, max_tokens: int = 512) -> None:
        super().__init__(model)
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

    def example_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros(1, self.max_tokens, dtype=torch.int64, device=self.device),
            "attention_mask": torch.ones(1, self.max_tokens, dtype=torch.int64, device=self.device),
        }

    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        seq_axis = [DynamicAxis(name="dim0", min=1, opt=1, max=8), DynamicAxis(name="dim1", min=1, opt=self.max_tokens, max=self.max_tokens)]
        return {"input_ids": seq_axis, "attention_mask": seq_axis}

    @property
    def input_names(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    @property
    def output_names(self) -> list[str]:
        return ["text_embeds"]
