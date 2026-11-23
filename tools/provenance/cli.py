"""Command-line entry point for deterministic build provenance verification.

Usage:

    python tools/provenance/cli.py [--backend python|zig] [--runs 2] [--json]

Exits non-zero when builds are non-deterministic or when cache scopes leak
state across runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Allow ``python tools/provenance/cli.py ...`` as well as package imports.
if __package__ in (None, ""):
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.provenance.recorder import default_scope_dimensions
from tools.provenance.verifier import (
    verify_cache_isolation,
    verify_deterministic_build,
)

# A no-op build function used for self-checks: it hashes each input and writes
# one output file per input into ``output_dir``.
_NOOP_OUTPUT_NAME = "provenance.out"


def _noop_build(inputs: dict[str, str], env: dict[str, str], output_dir: Path):
    import hashlib

    outputs: list[Path] = []
    for index, (name, text) in enumerate(inputs.items()):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out_path = output_dir / f"{index}_{name.replace('/', '_')}.out"
        out_path.write_text(f"{digest}\n", encoding="utf-8")
        outputs.append(out_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify deterministic build provenance and cache integrity.",
    )
    parser.add_argument(
        "--backend",
        default="python",
        choices=["python", "zig"],
        help="Backend being verified (default: python).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of repeated runs for the determinism check (default: 2).",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Cache/provenance root (default: a fresh temp directory).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cache_root = (
        Path(args.cache_root)
        if args.cache_root
        else Path(__import__("tempfile").mkdtemp(prefix="polyglot-provenance-root-"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    inputs = {
        "add.poly": "fn add(a: i64, b: i64) -> i64 { a + b }\n",
        "main.poly": "fn main() -> i64 { add(2, 3) }\n",
    }

    dimension = default_scope_dimensions(backend=args.backend)

    determinism = verify_deterministic_build(
        cache_root=cache_root,
        build_fn=_noop_build,
        inputs=inputs,
        backend=args.backend,
        runs=args.runs,
    )
    isolation = verify_cache_isolation(
        cache_root=cache_root,
        build_fn=_noop_build,
        inputs=inputs,
        backend=args.backend,
    )

    clean = determinism.clean and isolation.clean

    if args.json:
        payload: dict[str, Any] = {
            "clean": clean,
            "backend": args.backend,
            "cache_root": str(cache_root),
            "scope_dimensions": dimension,
            "determinism": determinism.to_dict(),
            "isolation": isolation.to_dict(),
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        print(f"backend: {args.backend}")
        print(f"cache_root: {cache_root}")
        print(f"scope_dimensions: {dimension}")
        print(f"determinism: {'clean' if determinism.clean else 'VIOLATION'}")
        print(f"  runs: {determinism.runs}")
        print(f"  fingerprints: {determinism.fingerprints}")
        print(f"  message: {determinism.message}")
        print(f"isolation: {'clean' if isolation.clean else 'VIOLATION'}")
        print(f"  scope_keys: {isolation.scope_keys}")
        print(f"  message: {isolation.message}")

    if not args.json:
        if clean:
            print("RESULT: CLEAN - builds are deterministic and caches are isolated")
        else:
            print(
                "RESULT: VIOLATIONS DETECTED - non-deterministic output or cache leakage"
            )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
