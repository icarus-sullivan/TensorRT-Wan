"""Per-model metadata that varies across Wan releases.

Keeping this in a small JSON file next to a model's built engines (rather than hardcoding
dimensions in `WanEngine`) is what "future Wan model support straightforward" (PLAN.md) means in
practice: supporting a new Wan release is adding a `wan_model.json`, not editing engine code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WanModelConfig:
    latent_channels: int
    vae_temporal_scale: int
    vae_spatial_scale: int
    text_embed_dim: int
    tokenizer_name: str
    default_num_frames: int = 81
    default_resolution: tuple[int, int] = (480, 832)
    fps: int = 16
    # Must match TextEncoderExporter/DiTExporter's max_tokens/max_text_tokens the engines were
    # built with — the DiT's `context` input has no dynamic axis at all (DiTExporter.dynamic_axes()
    # only covers `x`), so it's baked in as this exact fixed length, not just a padding default.
    # Confirmed via a real generate() attempt: the default tokenizer's `padding=True` (pads only to
    # the batch's longest sequence) produced an 11-token `context` against an engine expecting
    # exactly 512, raising `IExecutionContext::setInputShape`'s static-dimension-mismatch error.
    # See docs/wan2.2_i2v_14b_notes.md.
    max_text_tokens: int = 512

    @classmethod
    def load(cls, path: str | Path) -> "WanModelConfig":
        data = json.loads(Path(path).read_text())
        if "default_resolution" in data:
            data["default_resolution"] = tuple(data["default_resolution"])
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
