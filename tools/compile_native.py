"""
Pre-compile all native test sources into one cached build.

Usage:
    python tools/compile_native.py
    python tools/compile_native.py --tests-dir path/to/tests
    python tools/compile_native.py --check   # just collect, don't compile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _native_sources import (
    CARGO,
    collect_sources,
    compile_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compile native test sources.",
    )
    parser.add_argument(
        "--tests-dir", default=ROOT / "tests", type=Path,
        help="Directory containing test_native_*.py (default: tests/)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only collect and report; do not compile.",
    )
    args = parser.parse_args()

    if CARGO is None:
        print("cargo not found on PATH", file=sys.stderr)
        return 1

    if not args.tests_dir.is_dir():
        print(f"Not a directory: {args.tests_dir}", file=sys.stderr)
        return 1

    print(f"Collecting SOURCES from {args.tests_dir}...")
    sources = collect_sources(args.tests_dir)
    if not sources:
        print("No SOURCES found.", file=sys.stderr)
        return 1
    print(f"Found {len(sources)} native functions.")

    if args.check:
        for name in sources:
            print(f"  {name}")
        return 0

    print("Compiling...")
    ns = compile_sources(sources)

    missing = [name for name in sources if name not in ns]
    if missing:
        print(f"Missing from namespace: {missing}", file=sys.stderr)
        return 1

    print(f"OK: {len(ns)} names in namespace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
