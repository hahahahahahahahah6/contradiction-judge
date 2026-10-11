"""Command-line interface: ``contradiction-judge check memory.json``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, check_memories
from .hooks import load_hook, set_judge


def _load_entries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if not isinstance(data, list):
        raise ValueError("memory file must be a JSON list of entries")
    return data


def cmd_check(args: argparse.Namespace) -> int:
    if args.hook:
        set_judge(load_hook(args.hook))

    try:
        entries = _load_entries(args.memory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"contradiction-judge: error: {exc}", file=sys.stderr)
        return 2

    report = check_memories(entries)
    n = len(report["contradictions"])
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print(
        f"checked {report['entries']} entries "
        f"({report['pairs_checked']} pairs): {n} contradiction(s) found",
        file=sys.stderr,
    )
    if args.fail and n:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contradiction-judge",
        description="Detect contradictions between AI memory entries.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="Check a JSON memory file for contradictions.",
    )
    check.add_argument(
        "memory",
        help="Path to a JSON file: a list of {id, text, created_at?} entries.",
    )
    check.add_argument(
        "--hook",
        metavar="MODULE:FUNC",
        help="Pluggable LLM judge for semantic pairs rules miss.",
    )
    check.add_argument(
        "--fail",
        action="store_true",
        help="Exit 1 when any contradiction is found (for CI).",
    )
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
