"""Cross-backend parity: the same .poly file must produce the same main() result in the Python and native backends. Validates SPEC 1.7."""

from __future__ import annotations

from pathlib import Path

import pytest

from backends import get_backend
from lexer import lex
from parser import parse

# Anchor to the conftest.py directory, not the current working directory
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.poly"))

# Mark this test as slow (needs cargo) so it is deselected by default
pytestmark = pytest.mark.slow


def _compile(backend_name: str, source: str) -> dict:
    # Assumes get_backend accepts trusted=True
    backend = get_backend(backend_name, trusted=True)
    return backend.compile(parse(lex(source)))


def _call_main(ns: dict, label: str):
    if "main" not in ns:
        pytest.skip(f"{label} has no main(); parity N/A")
    return ns["main"]()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_cross_backend_parity(example):
    source = example.read_text(encoding="utf-8")
    py_ns = _compile("python", source)
    native_ns = _compile("native", source)

    py_result = _call_main(py_ns, f"{example.name} (python)")
    native_result = _call_main(native_ns, f"{example.name} (native)")

    assert py_result == native_result, (
        f"{example.name}: Python returned {py_result!r}, "
        f"native returned {native_result!r}"
    )
