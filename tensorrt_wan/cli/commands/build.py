from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tensorrt_wan.cli.loader import resolve_loader
from tensorrt_wan.cli.runtime_helpers import build_runtime
from tensorrt_wan.config.schema import DEFAULT_RESOLUTION_PROFILES, ResolutionProfile
from tensorrt_wan.export.exporters import DiTExporter, TextEncoderExporter, VAEDecoderExporter, VAEEncoderExporter
from tensorrt_wan.export.trt_build import build_tensorrt_engine
from tensorrt_wan.lora import onnx_weight_name_map, save_weight_name_map, weight_map_path_for_engine
from tensorrt_wan.runtime.cache import CacheKey
from tensorrt_wan.runtime.manager import RuntimeManager

_EXPORTERS = {
    "text_encoder": TextEncoderExporter,
    "dit": DiTExporter,
    "vae_encoder": VAEEncoderExporter,
    "vae_decoder": VAEDecoderExporter,
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("build", help="Build TensorRT engines")
    build_sub = parser.add_subparsers(dest="build_command", required=True)

    engine_parser = build_sub.add_parser("engine", help="ONNX -> TensorRT engine")
    engine_parser.add_argument("--component", choices=sorted(_EXPORTERS), required=True)
    engine_parser.add_argument("--onnx", required=True, help="Path to the ONNX file from 'trtwan export onnx'")
    engine_parser.add_argument(
        "--loader", required=True, help="Same loader used for export, to reconstruct shape metadata"
    )
    engine_parser.add_argument("--checkpoint", required=True)
    engine_parser.add_argument("--exporter-kwargs", default="{}")
    engine_parser.add_argument(
        "--resolutions", default=None, help="Comma-separated profile names from config; default: all configured"
    )
    engine_parser.add_argument("--precision", choices=["auto", "fp8", "fp16", "bf16", "fp32"], default="auto")
    engine_parser.add_argument("--force", action="store_true", help="Rebuild even if a cached engine matches")
    engine_parser.set_defaults(func=run_engine)


def run_engine(args: argparse.Namespace) -> int:
    loader = resolve_loader(args.loader)
    model = loader(args.checkpoint)
    exporter = _EXPORTERS[args.component](model, **json.loads(args.exporter_kwargs))

    # build_tensorrt_engine only reads exporter.dynamic_axes()/example_inputs() for their shapes
    # (not weight values), and only parses --onnx from disk — it never needs the model's actual
    # weights resident on GPU. Move it to CPU (not del: example_inputs() still calls
    # self.device/self.dtype, which read next(self.model.parameters()) and would break on a
    # None model) before the build's own workspace/kernel-autotuning allocation, which is
    # comparable in size to the model itself. Confirmed necessary on real hardware: TensorRT's
    # build OOM'd ("Requested amount of GPU memory (28579323904 bytes) could not be allocated")
    # while the ~28GB model was still resident — see docs/wan2.2_i2v_14b_notes.md.
    import torch

    model.to("cpu")
    torch.cuda.empty_cache()

    runtime = build_runtime(args)
    runtime.config.precision.mode = args.precision
    gpu = runtime.primary_gpu
    if gpu is None:
        raise SystemExit("No GPU detected; engine builds require a CUDA-capable device.")

    profiles = _resolve_profiles(runtime, args.resolutions)
    precision = runtime.select_precision(gpu.index).precision
    model_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()[:16] if Path(
        args.checkpoint
    ).is_file() else hashlib.sha256(args.checkpoint.encode()).hexdigest()[:16]

    cache_key = CacheKey(
        component=exporter.name,
        model_hash=model_hash,
        tensorrt_version=runtime.tensorrt.version or "unknown",
        cuda_version=gpu.cuda_version or "unknown",
        gpu_architecture=gpu.architecture.value,
        optimization_profile=",".join(p.name for p in profiles),
        precision=precision,
        input_shape_digest=exporter.shape_digest(),
    )

    if not args.force:
        cached = runtime.cache.get(cache_key)
        if cached is not None:
            print(f"Using cached engine: {cached}")
            return 0

    engine_bytes = build_tensorrt_engine(
        args.onnx,
        exporter,
        profiles,
        precision,
        workspace_limit_mb=runtime.config.memory.workspace_limit_mb,
        timing_cache_path=runtime.cache.directory / "trt_timing_cache.bin",
    )
    engine_path = runtime.cache.put(cache_key, engine_bytes)
    print(f"Built {args.component} engine -> {engine_path}")

    # Sidecar survives the onnx file's routine post-build deletion (see
    # docs/wan2.2_i2v_14b_notes.md) -- comfyui/nodes/lora_loader.py needs this mapping at inference
    # time and must not depend on the onnx file still existing.
    if os.environ.get("TRTWAN_ENABLE_REFIT", "0") == "1":
        weight_map = onnx_weight_name_map(args.onnx)
        map_path = weight_map_path_for_engine(engine_path)
        save_weight_name_map(weight_map, map_path)
        print(f"Wrote LoRA weight-name map -> {map_path}")

    return 0


def _resolve_profiles(runtime: RuntimeManager, resolutions_arg: str | None) -> list[ResolutionProfile]:
    available = {p.name: p for p in runtime.config.resolution_profiles or DEFAULT_RESOLUTION_PROFILES}
    if not resolutions_arg:
        return list(available.values())
    names = [n.strip() for n in resolutions_arg.split(",")]
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"Unknown resolution profile(s): {missing}. Known: {sorted(available)}")
    return [available[n] for n in names]
