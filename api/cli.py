"""
CLI entry for isolated compilation.

Usage:
    python -m api.cli compile --backend native --stdin < source.rs
    python -m api.cli compile --backend python --file input.rs

Exit codes:
    0 = success, stdout has one name per line
    1 = compile error, stderr has the error
    2 = trust violation / bad input
"""

from __future__ import annotations

import argparse
import sys

from ast_nodes import Program
from backends import UntrustedCompileError, get_backend
from lexer import lex
from parser import parse


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    compile_p = sub.add_parser("compile")
    compile_p.add_argument("--backend", choices=["python", "native"], required=True)
    src = compile_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--file", type=str)
    compile_p.add_argument("--trusted", action="store_true")
    args = parser.parse_args()

    if args.command != "compile":
        return 2

    if args.stdin:
        source = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8") as f:
            source = f.read()

    try:
        program: Program = parse(lex(source))
        backend = get_backend(args.backend, trusted=args.trusted)
        ns = backend.compile(program)
    except UntrustedCompileError as e:
        print(f"trust: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1

    for name in sorted(ns):
        if not name.startswith("_"):
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
