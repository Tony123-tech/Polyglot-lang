"""Tier 1: type-aware ``Option::map`` and ``Result::map``.

Both backends lower ``Option::map``/``Result::map`` to the matching standard
combinator while ``Vec::map`` keeps its collection semantics. The native
functions in ``SOURCES`` are compiled once by the session ``native_ns``
fixture; the Python cases compile a fresh backend per assertion.
"""

from __future__ import annotations

import pytest

from backends.python.translator import compile_rust
from compiler.core.lexer import lex
from compiler.core.parser import parse

SOURCES = {
    "tier1_map_option_some": """
fn tier1_map_option_some() -> i64 {
    Some(2).map(|x| x + 1).unwrap_or(0)
}
""",
    "tier1_map_option_none": """
fn tier1_map_option_none() -> i64 {
    let o: Option<i64> = None;
    o.map(|x| x + 1).unwrap_or(99)
}
""",
    "tier1_map_result_ok": """
fn tier1_map_result_ok_values() -> Result<i64, String> {
    Ok(2)
}

fn tier1_map_result_ok() -> i64 {
    let r = tier1_map_result_ok_values();
    r.map(|x| x + 1).unwrap_or(0)
}
""",
    "tier1_map_result_err": """
fn tier1_map_result_err_values() -> Result<i64, String> {
    Err("bad")
}

fn tier1_map_result_err() -> i64 {
    let r = tier1_map_result_err_values();
    match r.map(|x| x + 1) {
        Ok(v) => v,
        Err(_) => -1,
    }
}
""",
    "tier1_map_annotated": """
fn tier1_map_annotated() -> i64 {
    let o: Option<i64> = Some(5);
    o.map(|x| x * 2).unwrap_or(0)
}
""",
    "tier1_map_pop": """
fn tier1_map_pop(mut xs: Vec<i64>) -> i64 {
    xs.pop().map(|x| x + 1).unwrap_or(-1)
}
""",
    "tier1_map_capture": """
fn tier1_map_capture() -> i64 {
    let k = 10;
    Some(2).map(|x| x * k).unwrap_or(0)
}
""",
    "tier1_map_chain": """
fn tier1_map_chain() -> i64 {
    Some(1).map(|x| x + 1).map(|x| x * 10).unwrap_or(0)
}
""",
    "tier1_map_free_fn": """
fn tier1_map_helper() -> Option<i64> {
    Some(3)
}

fn tier1_map_free_fn() -> i64 {
    tier1_map_helper().map(|x| x * 2).unwrap_or(0)
}
""",
    "tier1_map_vec_regression": """
fn tier1_map_vec_regression(xs: Vec<i64>) -> Vec<i64> {
    xs.map(|x| x + 1)
}
""",
}


def _python_fn(source: str, name: str):
    return compile_rust(parse(lex(source)))[name]


def test_option_map_some_python():
    fn = _python_fn(SOURCES["tier1_map_option_some"], "tier1_map_option_some")
    assert fn() == 3


def test_option_map_none_python():
    fn = _python_fn(SOURCES["tier1_map_option_none"], "tier1_map_option_none")
    assert fn() == 99


def test_result_map_ok_python():
    fn = _python_fn(SOURCES["tier1_map_result_ok"], "tier1_map_result_ok")
    assert fn() == 3


def test_result_map_err_python():
    fn = _python_fn(SOURCES["tier1_map_result_err"], "tier1_map_result_err")
    assert fn() == -1


def test_annotated_binding_map_python():
    fn = _python_fn(SOURCES["tier1_map_annotated"], "tier1_map_annotated")
    assert fn() == 10


def test_pop_map_python():
    fn = _python_fn(SOURCES["tier1_map_pop"], "tier1_map_pop")
    assert fn([10, 20, 30]) == 31
    assert fn([]) == -1


def test_map_captures_local_python():
    fn = _python_fn(SOURCES["tier1_map_capture"], "tier1_map_capture")
    assert fn() == 20


def test_chained_map_python():
    fn = _python_fn(SOURCES["tier1_map_chain"], "tier1_map_chain")
    assert fn() == 20


def test_free_function_result_dispatches_python():
    fn = _python_fn(SOURCES["tier1_map_free_fn"], "tier1_map_free_fn")
    assert fn() == 6


def test_vec_map_regression_python():
    fn = _python_fn(SOURCES["tier1_map_vec_regression"], "tier1_map_vec_regression")
    assert fn([1, 2, 3]) == [2, 3, 4]


@pytest.mark.native
def test_option_result_map_native(native_ns):
    assert native_ns["tier1_map_option_some"]() == 3
    assert native_ns["tier1_map_option_none"]() == 99
    assert native_ns["tier1_map_result_ok"]() == 3
    assert native_ns["tier1_map_result_err"]() == -1
    assert native_ns["tier1_map_annotated"]() == 10
    assert native_ns["tier1_map_capture"]() == 20
    assert native_ns["tier1_map_chain"]() == 20
    assert native_ns["tier1_map_free_fn"]() == 6


@pytest.mark.native
def test_pop_map_native(native_ns):
    assert native_ns["tier1_map_pop"]([10, 20, 30]) == 31
    assert native_ns["tier1_map_pop"]([]) == -1


@pytest.mark.native
def test_vec_map_regression_native(native_ns):
    assert native_ns["tier1_map_vec_regression"]([1, 2, 3]) == [2, 3, 4]
