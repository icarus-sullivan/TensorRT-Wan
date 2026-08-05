from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_wan.cli.runtime_helpers import build_runtime


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inspect", help="Inspect a built TensorRT engine")
    parser.add_argument("engine_path", help="Path to a .engine file, or a cache digest prefix")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    path = Path(args.engine_path)
    if not path.exists():
        path = _resolve_from_cache(args, args.engine_path)

    print(f"Engine file: {path}")
    print(f"Size: {path.stat().st_size / (1 << 20):.1f} MiB")

    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        print(f"Cache metadata: {meta_path.read_text()}")

    try:
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        print(f"I/O tensors: {engine.num_io_tensors}")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(f"  {name}: dtype={engine.get_tensor_dtype(name)} shape={engine.get_tensor_shape(name)}")
    except ImportError:
        print("(install the 'tensorrt' package for layer-level inspection)")
    return 0


def _resolve_from_cache(args: argparse.Namespace, digest_prefix: str) -> Path:
    runtime = build_runtime(args)
    matches = list(runtime.cache.directory.glob(f"{digest_prefix}*.engine"))
    if not matches:
        raise SystemExit(f"No engine found at path or cache digest {digest_prefix!r}")
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous digest prefix {digest_prefix!r}: {matches}")
    return matches[0]
