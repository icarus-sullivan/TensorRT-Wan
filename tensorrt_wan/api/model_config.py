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

    @classmethod
    def load(cls, path: str | Path) -> "WanModelConfig":
        data = json.loads(Path(path).read_text())
        if "default_resolution" in data:
            data["default_resolution"] = tuple(data["default_resolution"])
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
