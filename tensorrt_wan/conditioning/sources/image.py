from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor

if TYPE_CHECKING:
    import torch


class ImageEncoderEngineLike(Protocol):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...


class ImageConditioningSource(ConditioningSource):
    """Wraps `engine.vae_engine.VAEEncoderEngine` for any single-frame image conditioning.

    Image, first-frame, and future last-frame conditioning are all "encode one frame to a
    latent" — they differ only in which `ConditioningKind` the result is filed under, so this
    one class is parameterized by `kind` instead of being duplicated three times.
    """

    def __init__(self, image_encoder: ImageEncoderEngineLike, kind: ConditioningKind = ConditioningKind.IMAGE) -> None:
        if kind not in (ConditioningKind.IMAGE, ConditioningKind.FIRST_FRAME, ConditioningKind.LAST_FRAME):
            raise ValueError(f"ImageConditioningSource does not support kind={kind}")
        self.kind = kind
        self.image_encoder = image_encoder

    def encode(self, inputs: torch.Tensor) -> ConditioningTensor:
        embedding = self.image_encoder.encode_image(inputs)
        return ConditioningTensor(kind=self.kind, embedding=embedding)
