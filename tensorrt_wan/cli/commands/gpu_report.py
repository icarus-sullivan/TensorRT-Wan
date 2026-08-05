from __future__ import annotations

import argparse

from tensorrt_wan.cli.runtime_helpers import build_runtime


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("gpu-report", help="Detect GPUs and TensorRT capability")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    runtime = build_runtime(args)
    print(runtime.diagnostics().as_text())
    return 0
