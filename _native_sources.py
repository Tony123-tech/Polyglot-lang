"""
Shared helper: collect native test sources, merge, compile once.

Used by:
- conftest.py              (pytest_collection_modifyitems hook)
- tools/compile_native.py  (CI pre-build)

Test modules declare:

    PREAMBLE = "..."          # optional: shared struct/enum definitions
    SOURCES = {"fn_name": "..."}

Keys are Rust function names, must be unique across all modules.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

# Walk up until we find the project root (contains ast_nodes.py).
ROOT = Path(__file__).resolve().parent
if not (ROOT / "ast_nodes.py").exists():
    for parent in ROOT.parents:
        if (parent / "ast_nodes.py").exists():
            ROOT = parent
            break

CARGO = shutil.which("cargo")

_PREAMBLE_KEY = "__preamble__"


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect_sources(tests_dir: Path) -> dict[str, str]:
    """Load every test_native_*.py and merge their SOURCES dicts."""
    sources: dict[str, str] = {}
    preambles: list[str] = []
    for path in sorted(tests_dir.glob("test_native_*.py")):
        mod = _load_module(path)
        preamble = getattr(mod, "PREAMBLE", None)
        if preamble is not None:
            preambles.append(preamble)
        module_sources = getattr(mod, "SOURCES", None)
        if module_sources is None:
            continue
        for fn_name, src in module_sources.items():
            if fn_name in sources:
                raise RuntimeError(
                    f"Duplicate native function {fn_name!r} in {path.name}"
                )
            sources[fn_name] = src
    return _with_preamble(sources, preambles)


def collect_from_modules(modules: list[Any]) -> dict[str, str]:
    """Merge SOURCES from already-imported test modules (used by conftest)."""
    sources: dict[str, str] = {}
    preambles: list[str] = []
    for mod in modules:
        preamble = getattr(mod, "PREAMBLE", None)
        if preamble is not None:
            preambles.append(preamble)
        module_sources = getattr(mod, "SOURCES", None)
        if module_sources is None:
            continue
        for fn_name, src in module_sources.items():
            if fn_name in sources:
                raise RuntimeError(
                    f"Duplicate native function {fn_name!r} in "
                    f"{mod.__name__}"
                )
            sources[fn_name] = src
    return _with_preamble(sources, preambles)


def _with_preamble(
    sources: dict[str, str], preambles: list[str]
) -> dict[str, str]:
    if not preambles:
        return sources
    return {_PREAMBLE_KEY: "\n".join(preambles), **sources}


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_sources(sources: dict[str, str]) -> dict[str, Any]:
    """Merge all sources, parse, compile via native backend, return namespace."""
    if not sources:
        return {}

    from ast_nodes import Program
    from backends import get_backend
    from lexer import lex
    from parser import parse

    # Sort by name for deterministic cache keys across callers
    # (compile_native.py vs conftest.py may see sources in different order).
    ordered = sorted(sources.items())
    combined = "\n".join(src for _, src in ordered)
    program: Program = parse(lex(combined))
    backend = get_backend("native", trusted=True)
    return backend.compile(program)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_root_on_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
