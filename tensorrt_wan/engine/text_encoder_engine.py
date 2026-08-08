from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from tensorrt_wan.engine.base import TensorRTEngineWrapper


class TokenizerLike(Protocol):
    def __call__(self, prompt: str) -> dict[str, torch.Tensor]: ...


class TextEncoderEngine:
    """TensorRT-accelerated Wan text encoder (T5/UMT5-family, depending on Wan release).

    Tokenization stays on CPU/PyTorch (it's not a meaningful cost center and TensorRT has no
    tokenizer op) — only the transformer forward pass runs through TensorRT.
    """

    def __init__(
        self,
        engine_path: str | Path,
        tokenizer: TokenizerLike,
        *,
        device: torch.device | None = None,
        torch_fallback: torch.nn.Module | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._wrapper = TensorRTEngineWrapper(engine_path, device=device, torch_fallback=torch_fallback)

    def load(self) -> None:
        self._wrapper.load()

    def unload(self) -> None:
        self._wrapper.unload()

    def encode_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt)
        outputs = self._wrapper.infer(tokens)
        return outputs["text_embeds"]
