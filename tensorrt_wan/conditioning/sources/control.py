from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor

if TYPE_CHECKING:
    import torch


class ControlEncoderLike(Protocol):
    def encode_control(self, control_map: torch.Tensor) -> torch.Tensor: ...


class ControlNetConditioningSource(ConditioningSource):
    """ControlNet-style dense control map (pose, depth, edge maps, ...) conditioning."""

    kind = ConditioningKind.CONTROL

    def __init__(self, control_encoder: ControlEncoderLike) -> None:
        self.control_encoder = control_encoder

    def encode(self, inputs: torch.Tensor) -> ConditioningTensor:
        embedding = self.control_encoder.encode_control(inputs)
        return ConditioningTensor(kind=self.kind, embedding=embedding)
