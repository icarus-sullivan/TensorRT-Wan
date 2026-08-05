from __future__ import annotations

import argparse

from tensorrt_wan.cli.runtime_helpers import build_runtime


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cache", help="Inspect or clear the engine cache")
    cache_sub = parser.add_subparsers(dest="cache_command", required=True)

    list_parser = cache_sub.add_parser("list", help="List cached engines")
    list_parser.set_defaults(func=run_list)

    clear_parser = cache_sub.add_parser("clear", help="Delete every cached engine")
    clear_parser.set_defaults(func=run_clear)


def run_list(args: argparse.Namespace) -> int:
    runtime = build_runtime(args)
    entries = runtime.cache.list()
    if not entries:
        print("Engine cache is empty.")
        return 0
    for entry in entries:
        print(
            f"model={entry['model_hash'][:12]} precision={entry['precision']} "
            f"gpu={entry['gpu_architecture']} trt={entry['tensorrt_version']} "
            f"profile={entry['optimization_profile']}"
        )
    return 0


def run_clear(args: argparse.Namespace) -> int:
    runtime = build_runtime(args)
    removed = runtime.cache.clear()
    print(f"Removed {removed} cached engine(s) from {runtime.cache.directory}")
    return 0
