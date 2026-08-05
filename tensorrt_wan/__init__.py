"""TensorRT-Wan: a TensorRT acceleration framework for Wan video generation models.

See PLAN.md for the project spec and README.md for usage. Public surface is intentionally small:
`WanEngine` for the standalone API, plus the config/runtime types most callers building a custom
pipeline (rather than using `WanEngine` directly) will need.
"""

from __future__ import annotations

from tensorrt_wan.api import VideoOutput, WanEngine, WanModelConfig
from tensorrt_wan.config import TensorRTWanConfig
from tensorrt_wan.runtime import RuntimeManager

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "WanEngine",
    "WanModelConfig",
    "VideoOutput",
    "TensorRTWanConfig",
    "RuntimeManager",
]
