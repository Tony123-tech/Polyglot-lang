"""Known-limitations audit.

Each test pins one entry from the README's "Known limitations" section so the
documented behavior cannot regress silently.
"""

from __future__ import annotations

import asyncio

import pytest

from backends.python.translator import compile_rust
from backends.rust.codegen import RustCodegen
from lexer import lex
from parser import ParseError, parse


def _run_python(src: str) -> dict:
    return compile_rust(parse(lex(src)))


def _rust_source(src: str) -> str:
    return RustCodegen().generate(parse(lex(src)))


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


def test_async_impl_method_native_codegen_returns_pyfuture():
    """Native async impl methods return a PyFuture instead of a plain method."""
    generated = _rust_source(
        "struct Counter { value: i64 }\n"
        "impl Counter { async fn get(&self) -> i64 { self.value } }\n"
    )
    method = generated.split("impl Counter")[-1]
    assert "fn get(&self) -> PyResult<PyFuture>" in method
    assert "fn get(&self) -> i64" not in method
    assert "PyFuture" in method
    assert "py_future(async move" in method
    assert "let __self_value = self.value.clone();" in method


def test_async_mut_self_method_native_codegen_uses_borrow_guard():
    generated = _rust_source(
        "struct Counter { value: i64 }\n"
        "impl Counter { async fn bump(&mut self) { self.value += 1; } }\n"
    )
    method = generated.split("impl Counter")[-1]
    assert (
        "fn bump(slf: pyo3::Py<Self>, py: pyo3::Python<'_>) -> PyResult<PyFuture>"
        in method
    )
    assert 'PolyglotBorrowGuard::acquire_mut(&__borrow_state, "Counter")' in method
    assert "let mut __self_mut_value = __this.value.clone();" in method
    assert "__this.value = __self_mut_value;" in method
    assert "not supported" not in method


# ---------------------------------------------------------------------------
# Box patterns / enum payload slices / nested Vec slices are Python-only
# ---------------------------------------------------------------------------


def test_box_pattern_works_in_python():
    ns = _run_python("fn unbox(x: Box<i64>) -> i64 { match x { box v => *v } }")
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
# PyFuture Send awareness
# ---------------------------------------------------------------------------


def test_pyfuture_runtime_is_send_aware():
    generated = _rust_source("fn noop() {}")
    assert "#[pyclass(unsendable)]" not in generated
    assert "pub struct PyFuture {" in generated
    assert "pub struct PyFutureIterator {" in generated
    assert "dyn Future<Output = PyResult<PyObject>> + Send" in generated
    assert "F: Future<Output = PyResult<PyObject>> + Send" in generated
