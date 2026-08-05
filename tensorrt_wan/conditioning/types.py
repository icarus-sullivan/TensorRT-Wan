"""Shared types for the conditioning system.

`UnifiedConditioning` is the one payload shape the DiT engine ever consumes — every
`ConditioningSource`, regardless of what kind of input it wraps, ultimately contributes to one
of these instead of the DiT engine special-casing per-workflow conditioning shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class ConditioningKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    CONTROL = "control"
    IP_ADAPTER = "ip_adapter"
    LORA = "lora"


@dataclass
class ConditioningTensor:
    """Output of a single `ConditioningSource.encode()` call."""

    kind: ConditioningKind
    embedding: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedConditioning:
    """Merged output of `ConditioningManager.combine()` — the DiT engine's sole conditioning input.

    `embeddings`/`masks` are keyed by `ConditioningKind.value` rather than being fixed fields so
    adding a future conditioning kind never requires changing this dataclass or the DiT engine's
    call signature — only registering a new `ConditioningSource`.
    """

    embeddings: dict[str, torch.Tensor] = field(default_factory=dict)
    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    lora_weights: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
