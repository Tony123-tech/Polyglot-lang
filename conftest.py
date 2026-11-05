"""
Pytest plugin: merge all native SOURCES and compile once per session.

Test modules declare `SOURCES = {fn_name: source}` at module level.
`pytest_collection_modifyitems` merges every module's SOURCES, compiles
once, and exposes the resulting namespace via the `native_ns` fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from _native_sources import (
    CARGO,
    collect_from_modules,
    compile_sources,
    ensure_root_on_path,
)

ensure_root_on_path()


_NATIVE_NS: dict[str, Any] | None = None


def pytest_configure(config):
    config.addinivalue_line("markers", "native: requires cargo on PATH")


def pytest_collection_modifyitems(config, items):
    """Merge all SOURCES from test modules, compile once, store namespace."""
    global _NATIVE_NS

    if CARGO is None:
        for item in items:
            if "native" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="cargo not found"))
        return

    # Collect unique modules (items may share a module).
    seen_modules: dict[str, Any] = {}
    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        seen_modules[module.__name__] = module

    sources = collect_from_modules(list(seen_modules.values()))
    if not sources:
        return

    _NATIVE_NS = compile_sources(sources)


@pytest.fixture(scope="session")
def native_ns() -> dict[str, Any]:
    """Return the compiled namespace containing all native test functions."""
    if _NATIVE_NS is None:
        pytest.skip("native namespace not built (no SOURCES found)")
    return _NATIVE_NS


@pytest.fixture(scope="session")
def native_backend():
    """Raw NativeBackend instance for tests that need full control."""
    if CARGO is None:
        pytest.skip("cargo not found on PATH")
    from backends import get_backend
    return get_backend("native", trusted=True)
