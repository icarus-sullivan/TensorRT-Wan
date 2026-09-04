"""Configuration schema for TensorRT-Wan.

All sections are plain dataclasses so they serialize losslessly to/from JSON and YAML
(see :mod:`tensorrt_wan.config.loader`) and can be constructed directly in Python without
going through a file at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PrecisionMode = Literal["auto", "fp8", "fp16", "bf16", "fp32"]
AttentionImpl = Literal["auto", "flash", "flash2", "flash3", "sage", "torch_sdpa"]


@dataclass
class PrecisionConfig:
    """Precision selection. `mode="auto"` defers to runtime GPU-architecture detection.

    Quality is never traded for memory: `allow_fp8` only permits FP8 where the runtime's
    calibration marks it quality-neutral for a given op, per the project's precision strategy.
    """

    mode: PrecisionMode = "auto"
    allow_fp8: bool = True
    allow_bf16: bool = True


@dataclass
class ResolutionProfile:
    """One (height, width[, num_frames]) shape TensorRT-Wan should build/optimize for."""

    name: str
    height: int
    width: int
    num_frames: int | None = None
    dynamic: bool = False


DEFAULT_RESOLUTION_PROFILES: tuple[ResolutionProfile, ...] = (
    ResolutionProfile("480x832", 480, 832),
    ResolutionProfile("512x512", 512, 512),
    ResolutionProfile("720x1280", 720, 1280),
    ResolutionProfile("768x768", 768, 768),
    ResolutionProfile("1024x1024", 1024, 1024),
    ResolutionProfile("1080x1920", 1080, 1920),
    ResolutionProfile("720x1088", 720, 1088),
    ResolutionProfile("1088x720", 1088, 720),
)


@dataclass
class CacheConfig:
    """Engine cache location and invalidation behavior.

    Invalidation keys (model hash, TensorRT version, CUDA version, GPU SM architecture,
    optimization profile, precision) are computed by `runtime.cache.EngineCache`, not here —
    this only configures where the cache lives and whether it's used.
    """

    enabled: bool = True
    directory: Path = field(default_factory=lambda: Path.home() / ".cache" / "tensorrt_wan" / "engines")
    max_size_gb: float | None = None


@dataclass
class PluginConfig:
    """Enable/disable individual TensorRT plugins by registry name.

    Unlisted plugins default to enabled; entries here only override that default.
    See `tensorrt_wan.plugins.registry` for the set of known plugin names.
    """

    enabled: dict[str, bool] = field(default_factory=dict)


@dataclass
class AttentionConfig:
    implementation: AttentionImpl = "auto"


@dataclass
class MemoryConfig:
    workspace_limit_mb: int | None = None
    memory_pool_limit_mb: int | None = None


@dataclass
class EnginePathsConfig:
    engine_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "tensorrt_wan" / "engines")
    onnx_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "tensorrt_wan" / "onnx")


@dataclass
class TensorRTWanConfig:
    """Root configuration object. See docs/architecture.md for how each section is consumed."""

    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    resolution_profiles: list[ResolutionProfile] = field(
        default_factory=lambda: list(DEFAULT_RESOLUTION_PROFILES)
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    engine_paths: EnginePathsConfig = field(default_factory=EnginePathsConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    log_level: str = "INFO"
