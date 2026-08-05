from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor

if TYPE_CHECKING:
    import torch


class IPAdapterProjectorLike(Protocol):
    def project(self, image_embedding: torch.Tensor) -> torch.Tensor: ...


class IPAdapterConditioningSource(ConditioningSource):
    """IP-Adapter reference-image conditioning: projects an image embedding into DiT cross-attention space."""

    kind = ConditioningKind.IP_ADAPTER

    def __init__(self, projector: IPAdapterProjectorLike) -> None:
        self.projector = projector

    def encode(self, inputs: torch.Tensor) -> ConditioningTensor:
        embedding = self.projector.project(inputs)
        return ConditioningTensor(kind=self.kind, embedding=embedding)
