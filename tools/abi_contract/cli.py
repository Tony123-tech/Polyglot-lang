"""Command-line entry point for the cross-backend ABI/FFI contract verifier.

Usage:

    python tools/abi_contract/cli.py [--word 8] [--json]

Exits non-zero when any backend drifts from the contract or when the
``E001``-``E008`` error surface is inconsistent across backends.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

# Allow ``python tools/abi_contract/cli.py ...`` as well as package imports.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path

    # Add the project root so ``tools`` and ``zig`` packages are importable.
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.abi_contract.verifier import verify_backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify cross-backend ABI/FFI contract consistency.",
    )
    parser.add_argument(
        "--word",
        type=int,
        default=8,
        choices=(4, 8),
        help="Target word size in bytes (default: 8).",
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

    result = verify_backends(word=args.word)

    if args.json:
        payload: dict[str, Any] = {
            "clean": result["clean"],
            "contract_fingerprint": result["contract_fingerprint"],
            "word": args.word,
            "backends": {
                name: [violation.to_dict() for violation in violations]
                for name, violations in result["backends"].items()
            },
            "error_code_parity": [
                violation.to_dict() for violation in result["error_code_parity"]
            ],
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        print(result["contract_fingerprint"])
        for backend_name in sorted(result["backends"]):
            violations = result["backends"][backend_name]
            print(f"backend: {backend_name}")
            if not violations:
                print("  (clean)")
                continue
            for violation in violations:
                print(f"  - [{violation.check}] {violation.message}")
        if result["error_code_parity"]:
            print("error_code_parity:")
            for violation in result["error_code_parity"]:
                print(f"  - [{violation.check}] {violation.message}")
        elif not result["clean"]:
            print("error_code_parity: (clean)")

    if result["clean"]:
        print("RESULT: CLEAN - all backends match the contract")
        return 0
    print("RESULT: VIOLATIONS DETECTED - a backend drifted from the ABI/FFI contract")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
