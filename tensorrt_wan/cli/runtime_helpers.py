"""Shared helper for CLI commands that need a `RuntimeManager` respecting the global
`--cache-dir` flag (see `cli/main.py`) — one place instead of duplicating the same override
logic in every command that touches the engine cache (`build`, `cache`, `optimization-report`,
`gpu-report`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_wan.config.schema import CacheConfig, TensorRTWanConfig
from tensorrt_wan.runtime.manager import RuntimeManager


def build_runtime(args: argparse.Namespace) -> RuntimeManager:
    cache_dir = getattr(args, "cache_dir", None)
    if not cache_dir:
        return RuntimeManager()
    return RuntimeManager(TensorRTWanConfig(cache=CacheConfig(directory=Path(cache_dir))))
