"""Known-limitations audit.

Each test pins one entry from the README's "Known limitations" section so the
documented behavior cannot regress silently.
"""

from __future__ import annotations

import asyncio

import pytest

from backends.python.translator import TranslateError, compile_rust
from backends.rust.codegen import RustCodegen
from lexer import lex
from parser import ParseError, parse


def _run_python(src: str) -> dict:
    return compile_rust(parse(lex(src)))


def _rust_source(src: str) -> str:
    return RustCodegen().generate(parse(lex(src)))


# ---------------------------------------------------------------------------
# Option.map collision with Vec.map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "src",
    [
        "fn f() -> i64 { Some(1).map(|x| x + 1).unwrap_or(0) }",
        "fn f() -> i64 { let x: Option<i64> = Some(1); x.map(|v| v + 1).unwrap_or(0) }",
        "fn f(mut xs: Vec<i64>) -> i64 { xs.pop().map(|v| v + 1).unwrap_or(0) }",
    ],
)
def test_option_map_is_rejected_in_python(src):
    with pytest.raises(TranslateError, match="Option.map is unsupported"):
        _run_python(src)


@pytest.mark.parametrize(
    "src",
    [
        "fn f() -> i64 { Some(1).map(|x| x + 1).unwrap_or(0) }",
        "fn f() -> i64 { let x: Option<i64> = Some(1); x.map(|v| v + 1).unwrap_or(0) }",
        "fn f(mut xs: Vec<i64>) -> i64 { xs.pop().map(|v| v + 1).unwrap_or(0) }",
    ],
)
def test_option_map_is_rejected_in_rust_codegen(src):
    with pytest.raises(NotImplementedError, match="Option.map is unsupported"):
        _rust_source(src)


def test_vec_map_still_lowers():
    ns = _run_python("fn f(xs: Vec<i64>) -> Vec<i64> { xs.map(|v| v + 1) }")
    assert ns["f"]([1, 2, 3]) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Result combinators are Python-only inside let-bindings
# ---------------------------------------------------------------------------

def test_result_combinators_and_conversions_are_python_only():
    ns = _run_python(
        """
fn is_ok() -> bool { Ok(1).is_ok() }
fn is_err() -> bool { Err("x").is_err() }
fn map_err() -> Result<i64, String> {
    Err("bad").map_err(|s| format!("wrapped: {}", s))
}
fn ok_or() -> Result<i64, String> {
    let value: Option<i64> = None;
    value.ok_or("missing".to_string())
}
"""
    )
    assert ns["is_ok"]() is True
    assert ns["is_err"]() is True
    assert ns["map_err"]() == ns["_Result"].from_err("wrapped: bad")
    assert ns["ok_or"]() == ns["_Result"].from_err("missing")


# ---------------------------------------------------------------------------
# Async impl methods
# ---------------------------------------------------------------------------

def test_async_impl_method_runs_in_python():
    ns = _run_python(
        """
struct Counter { value: i64 }

impl Counter {
    fn new(value: i64) -> Counter { Counter { value } }
    async fn get(&self) -> i64 { self.value }
}
"""
    )
    counter = ns["Counter"].new(7)
    assert asyncio.run(counter.get()) == 7


def test_async_impl_method_native_codegen_is_sync_only():
    """Native codegen has no async-method lowering; it emits a plain method.

    This pins the documented limitation rather than the ideal behavior, so a
    future native async implementation must update both this test and the
    README.
    """
    generated = _rust_source(
        "struct Counter { value: i64 }\n"
        "impl Counter { async fn get(&self) -> i64 { self.value } }\n"
    )
    assert "fn get(&self) -> i64" in generated
    assert "PyFuture" not in generated.split("impl Counter")[-1]


# ---------------------------------------------------------------------------
# Box patterns / enum payload slices / nested Vec slices are Python-only
# ---------------------------------------------------------------------------

def test_box_pattern_works_in_python():
    ns = _run_python(
        "fn unbox(x: Box<i64>) -> i64 { match x { box v => *v } }"
    )
    assert ns["unbox"](5) == 5


def test_enum_payload_slice_pattern_works_in_python():
    ns = _run_python(
        """
enum Holder { Values(Vec<i64>) }
fn first_value(holder: Holder) -> i64 {
    match holder {
        Holder::Values([first, ..]) => first,
        _ => 0,
    }
}
"""
    )
    assert ns["first_value"](ns["Holder"].Values([7, 8])) == 7


def test_nested_vec_slice_pattern_works_in_python():
    ns = _run_python(
        """
fn first_inner(xs: Vec<Vec<i64>>) -> i64 {
    match xs {
        [[first, ..], ..] => first,
        _ => 0,
    }
}
"""
    )
    assert ns["first_inner"]([[4, 5], [6]]) == 4


# ---------------------------------------------------------------------------
# while let with iter.next()
# ---------------------------------------------------------------------------

def test_while_let_next_is_a_parse_error():
    with pytest.raises(ParseError, match="expected expression, got LET"):
        parse(
            lex(
                "fn f(mut xs: Vec<i64>) -> i64 {\n"
                "    let mut total = 0;\n"
                "    while let Some(x) = xs.next() { total += x; }\n"
                "    total\n"
                "}"
            )
        )


# ---------------------------------------------------------------------------
# PyFuture thread affinity
# ---------------------------------------------------------------------------

def test_pyfuture_runtime_is_marked_unsendable():
    generated = _rust_source("fn noop() {}")
    assert "#[pyclass(unsendable)]\npub struct PyFuture" in generated
    assert "#[pyclass(unsendable)]\npub struct PyFutureIterator" in generated
