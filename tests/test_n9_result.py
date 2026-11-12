# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N9: Platform-defined Result and Option combinators."""

from __future__ import annotations

import pytest

from lexer import lex
from parser import parse

SOURCES = {
    "n9_result_predicates": """
fn n9_is_ok_true() -> bool { Ok(7).is_ok() }
fn n9_make_error() -> Result<i64, String> { Err("bad") }
fn n9_is_ok_false() -> bool { n9_make_error().is_ok() }
fn n9_is_err_true() -> bool { n9_make_error().is_err() }
fn n9_is_err_false() -> bool { Ok(7).is_err() }
fn n9_map_err_ok() -> Result<i64, String> {
    Ok(5).map_err(|error| error)
}
fn n9_map_err_failure() -> Result<i64, String> {
    Err("wrapped").map_err(|error| error)
}
fn n9_map_err_change() -> Result<i64, String> {
    Err("original").map_err(|_error| {
        pyo3::exceptions::PyRuntimeError::new_err("changed")
    })
}
""",
    "n9_option_conversions": """
fn n9_ok_or_some() -> Result<i64, String> {
    let value: Option<i64> = Some(5);
    value.ok_or("missing".to_string())
}
fn n9_ok_or_none() -> Result<i64, String> {
    let value: Option<i64> = None;
    value.ok_or("missing".to_string())
}
fn n9_ok_or_else_some() -> Result<i64, String> {
    let value: Option<i64> = Some(6);
    value.ok_or_else(|| "missing".to_string())
}
fn n9_ok_or_else_none() -> Result<i64, String> {
    let value: Option<i64> = None;
    value.ok_or_else(|| "missing".to_string())
}
""",
    "n9_result_continuations": """
fn n9_and_then_success() -> Result<i64, String> {
    Ok(5).and_then(|value| Ok(value + 1))
}
fn n9_and_then_failure() -> Result<i64, String> {
    Err("and_then error").and_then(|value| Ok(value + 1))
}
fn n9_or_else_success() -> Result<i64, String> {
    Ok(4).or_else(|_error| Ok(8))
}
fn n9_or_else_recovery() -> Result<i64, String> {
    Err("or_else error").or_else(|_error| Ok(8))
}
""",
}

PYTHON_MAP_ERR_SOURCE = """
fn n9_python_map_err_changed() -> Result<i64, String> {
    Err("original").map_err(|error| format!("mapped: {}", error))
}
"""


def _run_python(extra_source: str = ""):
    from backends.python.translator import compile_rust

    source = "\n".join((*SOURCES.values(), extra_source))
    return compile_rust(parse(lex(source)))


def test_python_result_is_ok():
    ns = _run_python()
    assert ns["n9_is_ok_true"]() is True
    assert ns["n9_is_ok_false"]() is False


@pytest.mark.native
def test_native_result_is_ok(native_ns):
    assert native_ns["n9_is_ok_true"]() is True
    assert native_ns["n9_is_ok_false"]() is False


def test_python_result_is_err():
    ns = _run_python()
    assert ns["n9_is_err_true"]() is True
    assert ns["n9_is_err_false"]() is False


@pytest.mark.native
def test_native_result_is_err(native_ns):
    assert native_ns["n9_is_err_true"]() is True
    assert native_ns["n9_is_err_false"]() is False


def test_python_result_map_err():
    ns = _run_python(PYTHON_MAP_ERR_SOURCE)
    assert ns["n9_map_err_ok"]() == ns["_Result"].from_ok(5)
    assert ns["n9_map_err_failure"]() == ns["_Result"].from_err("wrapped")
    assert ns["n9_python_map_err_changed"]() == ns["_Result"].from_err(
        "mapped: original"
    )


@pytest.mark.native
def test_native_result_map_err(native_ns):
    assert native_ns["n9_map_err_ok"]() == 5
    with pytest.raises(RuntimeError, match="changed"):
        native_ns["n9_map_err_change"]()


def test_python_option_ok_or():
    ns = _run_python()
    assert ns["n9_ok_or_some"]() == ns["_Result"].from_ok(5)
    assert ns["n9_ok_or_none"]() == ns["_Result"].from_err("missing")


@pytest.mark.native
def test_native_option_ok_or(native_ns):
    assert native_ns["n9_ok_or_some"]() == 5
    with pytest.raises(RuntimeError, match="missing"):
        native_ns["n9_ok_or_none"]()


def test_python_option_ok_or_else():
    ns = _run_python()
    assert ns["n9_ok_or_else_some"]() == ns["_Result"].from_ok(6)
    assert ns["n9_ok_or_else_none"]() == ns["_Result"].from_err("missing")


@pytest.mark.native
def test_native_option_ok_or_else(native_ns):
    assert native_ns["n9_ok_or_else_some"]() == 6
    with pytest.raises(RuntimeError, match="missing"):
        native_ns["n9_ok_or_else_none"]()


def test_python_result_and_then_and_or_else():
    ns = _run_python()
    assert ns["n9_and_then_success"]() == ns["_Result"].from_ok(6)
    assert ns["n9_and_then_failure"]() == ns["_Result"].from_err(
        "and_then error"
    )
    assert ns["n9_or_else_success"]() == ns["_Result"].from_ok(4)
    assert ns["n9_or_else_recovery"]() == ns["_Result"].from_ok(8)


@pytest.mark.native
def test_native_result_and_then_and_or_else(native_ns):
    assert native_ns["n9_and_then_success"]() == 6
    assert native_ns["n9_or_else_success"]() == 4
    assert native_ns["n9_or_else_recovery"]() == 8
    with pytest.raises(RuntimeError, match="and_then error"):
        native_ns["n9_and_then_failure"]()
