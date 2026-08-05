from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor

if TYPE_CHECKING:
    import torch


class TextEncoderEngineLike(Protocol):
    def encode_text(self, prompt: str) -> torch.Tensor: ...


class TextConditioningSource(ConditioningSource):
    """Wraps `engine.text_encoder_engine.TextEncoderEngine` for prompt conditioning."""

    kind = ConditioningKind.TEXT

    def __init__(self, text_encoder: TextEncoderEngineLike) -> None:
        self.text_encoder = text_encoder

    def encode(self, inputs: str) -> ConditioningTensor:
        embedding = self.text_encoder.encode_text(inputs)
        return ConditioningTensor(kind=self.kind, embedding=embedding, metadata={"prompt": inputs})
