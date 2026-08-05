"""End-to-end PyTorch -> torch.export -> ONNX -> TensorRT pipeline, cache-aware.

This is the function the `trtwan build engine` CLI command and the ComfyUI "TensorRT Engine
Builder" node both call — neither reimplements the export sequence itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tensorrt_wan.config.schema import ResolutionProfile
from tensorrt_wan.export.base import ModelExporter
from tensorrt_wan.export.onnx_export import export_to_onnx
from tensorrt_wan.export.torch_export import export_to_torch_export
from tensorrt_wan.export.trt_build import build_tensorrt_engine
from tensorrt_wan.runtime.cache import CacheKey
from tensorrt_wan.runtime.manager import RuntimeManager
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


def run_export_pipeline(
    exporter: ModelExporter,
    runtime: RuntimeManager,
    resolution_profiles: list[ResolutionProfile],
    *,
    model_hash: str,
    force: bool = False,
) -> Path:
    """Produce (or fetch from cache) a TensorRT engine for `exporter.model`.

    `model_hash` identifies the checkpoint (e.g. a hash of its state_dict or file digest) and is
    supplied by the caller rather than computed here, since hashing weights is expensive enough
    that callers building several engines from one already-loaded checkpoint shouldn't pay for it
    per engine.
    """
    gpu = runtime.primary_gpu
    if gpu is None:
        raise RuntimeError("run_export_pipeline requires a detected GPU (see runtime.gpu.require_gpu)")
    precision = runtime.select_precision(gpu.index).precision

    profile_key = _profiles_digest(resolution_profiles)
    cache_key = CacheKey(
        model_hash=model_hash,
        tensorrt_version=runtime.tensorrt.version or "unknown",
        cuda_version=gpu.cuda_version or "unknown",
        gpu_architecture=gpu.architecture.value,
        optimization_profile=profile_key,
        precision=precision,
    )

    if not force:
        cached = runtime.cache.get(cache_key)
        if cached is not None:
            logger.info("Using cached engine for %s: %s", exporter.name, cached)
            return cached

    onnx_path = runtime.config.engine_paths.onnx_dir / f"{exporter.name}.onnx"
    exported_program = export_to_torch_export(exporter)
    export_to_onnx(exported_program, exporter, onnx_path)

    engine_bytes = build_tensorrt_engine(
        onnx_path,
        exporter,
        resolution_profiles,
        precision,
        workspace_limit_mb=runtime.config.memory.workspace_limit_mb,
    )
    return runtime.cache.put(cache_key, engine_bytes)


def _profiles_digest(resolution_profiles: list[ResolutionProfile]) -> str:
    names = sorted(p.name for p in resolution_profiles)
    return hashlib.sha256(",".join(names).encode()).hexdigest()[:12]
