"""N0: native backend can compile and load an empty Program."""

from __future__ import annotations

import uuid

import pytest

from compiler.core.ast_nodes import Program

SOURCE = """
fn add(a: i64, b: i64) -> i64 { a + b }
fn scale(a: i64, b: i64) -> i64 { a * b }
fn main() -> i64 { add(2, 3) + scale(4, 5) }
"""


def _child_cpu_seconds() -> float:
    """CPU time spent in cargo/rustc children of this process.

    Returns 0.0 where ``resource`` is unavailable (Windows), which disables
    the timing assertion below rather than failing on an unsupported host.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


@pytest.mark.native
def test_native_empty_module(native_backend):
    ns = native_backend.compile(Program(1, 1, []))
    assert isinstance(ns, dict)


def test_build_parser_accepts_incremental_flag():
    from api.cli import build_parser

    args = build_parser().parse_args(["build", "hello.poly", "--incremental"])
    assert args.incremental is True


@pytest.mark.native
def test_incremental_rebuild_only_recompiles_changed_function(native_backend):
    from compiler.core.lexer import lex
    from compiler.core.parser import parse

    # A per-run module name keeps this test from hitting an artifact built by
    # an earlier run, so the second build really exercises incremental Cargo.
    # The cargo scope has to be per-run as well: a shared scope keeps a warm
    # ``target`` directory, which makes the *first* build as cheap as the
    # incremental one and turns the ratio below into a coin flip.
    module_name = f"polyglot_incr_{uuid.uuid4().hex[:8]}"
    cache_scope = f"incremental-{module_name}"
    backend = native_backend
    first = parse(lex(SOURCE))
    started_cpu = _child_cpu_seconds()
    ns = backend.compile(
        first,
        module_name=module_name,
        incremental=True,
        cache_scope=cache_scope,
    )
    first_build_cpu = _child_cpu_seconds() - started_cpu
    assert ns["main"]() == 25

    changed_source = SOURCE.replace("a + b", "a + b + 10")
    changed = parse(lex(changed_source))
    started_cpu = _child_cpu_seconds()
    ns = backend.compile(
        changed,
        module_name=module_name,
        incremental=True,
        cache_scope=cache_scope,
    )
    rebuild_cpu = _child_cpu_seconds() - started_cpu

    assert ns["main"]() == 35
    assert backend.last_build["changed"] == ["fn:add"]
    # Wall-clock bounds are unreliable while xdist workers share the machine,
    # so measure the CPU time cargo/rustc actually consumed. Recompiling only
    # the changed fragment must cost clearly less than the initial build.
    budget = max(0.2, first_build_cpu * 0.75)
    assert rebuild_cpu < budget, (
        f"incremental rebuild burned {rebuild_cpu:.2f}s of CPU "
        f"vs {first_build_cpu:.2f}s for the initial build"
    )
