"""Resolves a user-supplied `module:function` loader string to a callable that returns a
loaded PyTorch `nn.Module`.

TensorRT-Wan deliberately does not vendor Wan's own model-definition code — Wan releases new
architectures independently of this framework, and hardcoding a model loader here would need
updating on every release. Instead, `export`/`build` CLI commands take `--loader module:function`;
that function is the caller's (or a thin adapter package's) responsibility and only needs to
return an `nn.Module` given a checkpoint path.
"""

from __future__ import annotations

import importlib
from typing import Callable

import torch


def resolve_loader(loader_spec: str) -> Callable[[str], torch.nn.Module]:
    """Parse `"package.module:function_name"` and return the resolved function."""
    if ":" not in loader_spec:
        raise ValueError(f"--loader must be of the form 'module.path:function_name', got {loader_spec!r}")
    module_name, func_name = loader_spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {func_name!r}") from exc
