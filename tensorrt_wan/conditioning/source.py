"""ConditioningSource: the one interface every conditioning input implements.

Adding support for a new conditioning method (a new adapter type, a future last-frame variant)
means writing one `ConditioningSource` subclass and registering it — never touching
`ConditioningManager` or the DiT engine. This is what lets the unified engine stay unified as
conditioning methods grow (see docs/architecture.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor


class ConditioningSource(ABC):
    """Encodes one kind of conditioning input into a `ConditioningTensor`.

    Implementations should be stateless with respect to a single `encode()` call — any model
    weights they need (a text encoder, an IP-Adapter projection) are owned by the engine wrapper
    passed in at construction, not reloaded per call.
    """

    kind: ConditioningKind

    @abstractmethod
    def encode(self, inputs: Any) -> ConditioningTensor:
        """Encode a raw input (prompt string, PIL image, control map, ...) into a ConditioningTensor."""
        raise NotImplementedError
