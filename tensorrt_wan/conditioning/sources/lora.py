from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor

if TYPE_CHECKING:
    import torch


@dataclass
class LoRASpec:
    """A LoRA to apply: which weights file, and how strongly to blend it in."""

    path: str
    scale: float = 1.0


class LoRALoaderLike(Protocol):
    def load_state_dict(self, path: str) -> dict[str, torch.Tensor]: ...


class LoRAConditioningSource(ConditioningSource):
    """LoRA is not a tensor concatenated into the DiT's input — it's a set of weight deltas
    merged into (or applied alongside) the engine's own weights. It's still routed through
    `ConditioningManager` so callers configure it the same way as every other conditioning
    source; `ConditioningManager.combine()` files its output into `UnifiedConditioning.lora_weights`
    instead of `.embeddings`.
    """

    kind = ConditioningKind.LORA

    def __init__(self, loader: LoRALoaderLike) -> None:
        self.loader = loader

    def encode(self, inputs: LoRASpec) -> ConditioningTensor:
        state_dict = self.loader.load_state_dict(inputs.path)
        return ConditioningTensor(
            kind=self.kind,
            metadata={"state_dict": state_dict, "scale": inputs.scale, "path": inputs.path},
        )
