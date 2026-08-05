from __future__ import annotations

import argparse

from tensorrt_wan.cli.runtime_helpers import build_runtime
from tensorrt_wan.config.schema import DEFAULT_RESOLUTION_PROFILES


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "optimization-report", help="Show precision/profile decisions and cache coverage"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    runtime = build_runtime(args)
    print(runtime.diagnostics().as_text())

    print("\nConfigured resolution profiles:")
    profiles = runtime.config.resolution_profiles or list(DEFAULT_RESOLUTION_PROFILES)
    cached_profile_keys = {entry["optimization_profile"] for entry in runtime.cache.list()}
    for profile in profiles:
        covered = any(profile.name in key for key in cached_profile_keys)
        status = "cached" if covered else "not built"
        print(f"  {profile.name} ({profile.height}x{profile.width}): {status}")
    return 0
