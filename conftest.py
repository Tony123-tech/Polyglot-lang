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

# Modules a new contributor can run on a machine with no Rust and no Zig
# install: the front end (lexing, parsing, AST, spans), type checking,
# ownership and borrow analysis, the diagnostics code table, the CLI floor,
# the include expander and the packaging contract. `pytest -m core_only` is
# documented in CONTRIBUTING.md as the pre-toolchain gate; the marker is
# applied here rather than in twenty-two files so the list stays in one place.
CORE_ONLY_MODULES = frozenset(
    {
        "test_frontend_lexing",
        "test_frontend_parsing",
        "test_frontend_ast_utils",
        "test_frontend_spans",
        "test_parser_recovery",
        "test_edge_cases",
        "test_host_interpreter",
        "test_packaging_config",
        "test_std_include",
        "test_inline_lang",
        "test_termination_frame",
        "test_borrow_ergo",
        "test_generics",
        "test_m7_diagnostics",
        "test_tier1_option_result_map",
        "test_types_rollout",
        "test_ownership_drop",
        "test_move_parity",
        "test_known_limitations",
        "test_pillar_feature_gates",
        "test_diagnostics_rollout",
        "test_f1_cli",
    }
)


def pytest_configure(config):
    config.addinivalue_line("markers", "native: requires cargo on PATH")
    config.addinivalue_line(
        "markers",
        "core_only: front-end, analysis and diagnostics tests that need no "
        "Rust or Zig toolchain",
    )


def pytest_collection_modifyitems(config, items):
    """Merge all SOURCES from test modules, compile once, store namespace."""
    global _NATIVE_NS

    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        if module.__name__.rsplit(".", 1)[-1] in CORE_ONLY_MODULES:
            item.add_marker(pytest.mark.core_only)

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
