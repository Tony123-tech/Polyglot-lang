"""N0: native backend can compile and load an empty Program."""

from __future__ import annotations

import time
import uuid

import pytest

from ast_nodes import Program

SOURCE = """
fn add(a: i64, b: i64) -> i64 { a + b }
fn scale(a: i64, b: i64) -> i64 { a * b }
fn main() -> i64 { add(2, 3) + scale(4, 5) }
"""


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
    from lexer import lex
    from parser import parse

    # A per-run module name keeps this test from hitting an artifact built by
    # an earlier run, so the second build really exercises incremental Cargo.
    module_name = f"polyglot_incr_{uuid.uuid4().hex[:8]}"
    backend = native_backend
    first = parse(lex(SOURCE))
    ns = backend.compile(
        first,
        module_name=module_name,
        incremental=True,
        cache_scope="incremental-test",
    )
    assert ns["main"]() == 25

    changed_source = SOURCE.replace("a + b", "a + b + 10")
    changed = parse(lex(changed_source))
    started = time.perf_counter()
    ns = backend.compile(
        changed,
        module_name=module_name,
        incremental=True,
        cache_scope="incremental-test",
    )
    elapsed = time.perf_counter() - started

    assert ns["main"]() == 35
    assert backend.last_build["changed"] == ["fn:add"]
    assert elapsed < 1.0, f"incremental rebuild took {elapsed:.2f}s"
