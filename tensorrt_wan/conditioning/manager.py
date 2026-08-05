"""ConditioningManager: combines every registered conditioning source into one payload.

This is the fan-in point the architecture diagram in PLAN.md calls the "Unified Conditioning
Manager" — the DiT engine calls `combine()` once per generation and never needs to know how many
or which conditioning sources were involved.
"""

from __future__ import annotations

from typing import Any, Iterable

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor, UnifiedConditioning
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


class ConditioningManager:
    def __init__(self) -> None:
        self._sources: dict[ConditioningKind, ConditioningSource] = {}

    def register(self, source: ConditioningSource) -> None:
        self._sources[source.kind] = source
        logger.debug("Registered conditioning source for kind=%s", source.kind.value)

    def combine(self, inputs: dict[ConditioningKind, Any]) -> UnifiedConditioning:
        """Encode every provided raw input with its registered source and merge the results.

        Raises `KeyError` if an input is given for a kind with no registered source — a workflow
        asking for conditioning the manager can't produce is a configuration bug, not something
        to silently skip.
        """
        unified = UnifiedConditioning()
        for kind, raw_input in inputs.items():
            if kind not in self._sources:
                raise KeyError(f"No ConditioningSource registered for kind={kind.value}")
            self._merge(unified, self._sources[kind].encode(raw_input))
        return unified

    def combine_encoded(self, tensors: Iterable[ConditioningTensor]) -> UnifiedConditioning:
        """Merge already-encoded `ConditioningTensor`s, skipping the `.encode()` step.

        For callers (notably the ComfyUI node graph, see comfyui/nodes/conditioning_manager.py)
        where each conditioning source is its own upstream node and encoding already happened —
        this only does the fan-in `combine()` also does, without needing registered sources.
        """
        unified = UnifiedConditioning()
        for tensor in tensors:
            self._merge(unified, tensor)
        return unified

    @staticmethod
    def _merge(unified: UnifiedConditioning, result: ConditioningTensor) -> None:
        kind = result.kind
        if kind is ConditioningKind.LORA:
            path = result.metadata.get("path", kind.value)
            unified.lora_weights[path] = result.metadata["state_dict"]
            return

        if result.embedding is not None:
            unified.embeddings[kind.value] = result.embedding
        if result.mask is not None:
            unified.masks[kind.value] = result.mask
        if result.metadata:
            unified.metadata[kind.value] = result.metadata
