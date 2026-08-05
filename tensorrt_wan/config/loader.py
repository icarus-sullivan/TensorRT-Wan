"""Load TensorRTWanConfig from JSON or YAML, or produce defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from tensorrt_wan.config.schema import (
    AttentionConfig,
    CacheConfig,
    EnginePathsConfig,
    MemoryConfig,
    PluginConfig,
    PrecisionConfig,
    ResolutionProfile,
    TensorRTWanConfig,
)

T = TypeVar("T")


def load_config(path: str | Path) -> TensorRTWanConfig:
    """Load a config file, applying schema defaults for any field the file omits."""
    path = Path(path)
    raw = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(raw) or {}
    elif path.suffix == ".json":
        data = json.loads(raw) or {}
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix} (expected .json/.yaml/.yml)")
    return _from_dict(TensorRTWanConfig, data)


def save_config(config: TensorRTWanConfig, path: str | Path) -> None:
    """Write a config to disk. Format is chosen from the file extension."""
    path = Path(path)
    data = to_dict(config)
    if path.suffix in (".yaml", ".yml"):
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    elif path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2, default=str))
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix} (expected .json/.yaml/.yml)")


def to_dict(config: TensorRTWanConfig) -> dict[str, Any]:
    return asdict(config)


def default_config() -> TensorRTWanConfig:
    return TensorRTWanConfig()


_SECTION_TYPES: dict[str, type] = {
    "precision": PrecisionConfig,
    "cache": CacheConfig,
    "engine_paths": EnginePathsConfig,
    "plugins": PluginConfig,
    "attention": AttentionConfig,
    "memory": MemoryConfig,
}


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from a dict, recursing into nested dataclass fields.

    Unknown keys in `data` are ignored so configs stay forward-compatible with older files
    written against earlier schema versions.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "resolution_profiles" and isinstance(value, list):
            kwargs[f.name] = [
                v if isinstance(v, ResolutionProfile) else ResolutionProfile(**v) for v in value
            ]
        elif f.name in _SECTION_TYPES and isinstance(value, dict):
            kwargs[f.name] = _from_dict(_SECTION_TYPES[f.name], value)
        elif f.name in ("directory", "engine_dir", "onnx_dir") and isinstance(value, str):
            kwargs[f.name] = Path(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[return-value]
