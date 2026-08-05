from __future__ import annotations

import argparse

from tensorrt_wan.cli.commands.cache import run_list


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("list", help="List cached engines")
    engines_sub = parser.add_subparsers(dest="list_command", required=True)
    engines_parser = engines_sub.add_parser("engines", help="List cached engines (alias for 'cache list')")
    engines_parser.set_defaults(func=run_list)
