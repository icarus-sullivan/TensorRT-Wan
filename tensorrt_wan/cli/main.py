"""`trtwan` CLI entry point. Each subcommand implements its own logic (see cli/commands/) and is
only ever run when the user invokes it — nothing here runs automatically or at import time.
"""

from __future__ import annotations

import argparse
import sys

from tensorrt_wan.cli.commands import ALL_COMMANDS
from tensorrt_wan.utils.logging import configure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trtwan", description="TensorRT-Wan command-line tools")
    parser.add_argument(
        "--log-level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Override the engine cache directory (default: ~/.cache/tensorrt_wan/engines). "
            "Matters on environments where $HOME isn't persistent storage — e.g. running as "
            "root in a container where only a specific mounted volume survives a restart; "
            "confirmed the hard way on a RunPod instance where the default silently wrote a "
            "26GB engine to /root/.cache, off the persistent volume. See "
            "docs/wan2.2_i2v_14b_notes.md."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_module in ALL_COMMANDS:
        command_module.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
