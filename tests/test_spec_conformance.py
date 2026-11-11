# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""Regression tests for specification gaps that require both backends."""

from __future__ import annotations

import pytest

from backends.python.translator import compile_rust
from backends.rust.compiler import RustCompileError
from lexer import lex
from parser import parse

SOURCES = {
    "spec_gap_parse_dynamic": """
fn spec_gap_parse_dynamic(value: String) -> i64 {
    value.parse::<i64>()
}
""",
    "spec_gap_repeat": """
fn spec_gap_repeat(value: String, count: i64) -> String {
    value.repeat(count)
}
""",
    "spec_gap_rotate_left": """
fn spec_gap_rotate_left(mut values: Vec<i64>, count: i64) -> Vec<i64> {
    values.rotate_left(count);
    values
}
""",
    "spec_gap_rotate_right": """
fn spec_gap_rotate_right(mut values: Vec<i64>, count: i64) -> Vec<i64> {
    values.rotate_right(count);
    values
}
""",
    "spec_gap_swap": """
fn spec_gap_swap(mut values: Vec<i64>, left: i64, right: i64) -> Vec<i64> {
    values.swap(left, right);
    values
}
""",
    "spec_gap_string_slice": """
fn spec_gap_string_slice(value: String) -> String {
    &value[1..2]
}
""",
}


def _run_python(source: str) -> dict:
    return compile_rust(parse(lex(source)))


def _assert_native_panic(function, expected_message: str, *args) -> None:
    with pytest.raises(BaseException, match=expected_message):
        function(*args)


def test_native_invalid_literal_parse_fails_during_compile(native_backend):
    source = 'fn spec_invalid_literal() -> i64 { "not an integer".parse::<i64>() }'
    with pytest.raises(RustCompileError, match="invalid literal for i64 parse"):
        native_backend.compile(parse(lex(source)))


@pytest.mark.native
def test_native_dynamic_parse_failure_remains_a_runtime_error(native_ns):
    _assert_native_panic(
        native_ns["spec_gap_parse_dynamic"],
        "ParseIntError",
        "not an integer",
    )


def test_python_invalid_literal_parse_raises_value_error():
    namespace = _run_python(
        'fn spec_invalid_literal() -> i64 { "not an integer".parse::<i64>() }'
    )
    with pytest.raises(ValueError):
        namespace["spec_invalid_literal"]()


def test_python_dynamic_parse_failure_raises_value_error():
    namespace = _run_python(
        """
fn spec_dynamic_parse(value: String) -> i64 {
    value.parse::<i64>()
}
"""
    )
    with pytest.raises(ValueError):
        namespace["spec_dynamic_parse"]("not an integer")


def test_parse_success_matches_both_backends(native_ns):
    python_ns = _run_python(
        """
fn spec_parse_success(value: String) -> i64 {
    value.parse::<i64>()
}
"""
    )
    assert python_ns["spec_parse_success"]("-42") == -42
    assert native_ns["spec_gap_parse_dynamic"]("42") == 42


@pytest.mark.native
@pytest.mark.parametrize(
    ("function_name", "expected_message", "args"),
    [
        ("spec_gap_repeat", "capacity overflow", ("x", -1)),
        ("spec_gap_rotate_left", "assertion failed", ([1, 2], 3)),
        ("spec_gap_rotate_right", "assertion failed", ([1, 2], 3)),
        ("spec_gap_swap", "index out of bounds", ([1, 2], 0, 2)),
        ("spec_gap_string_slice", "char boundary", ("éx",)),
    ],
)
def test_native_boundary_errors_are_not_silent(
    native_ns, function_name, expected_message, args,
):
    _assert_native_panic(
        native_ns[function_name], expected_message, *args,
    )


@pytest.mark.parametrize(
    ("source", "function_name", "args"),
    [
        (
            SOURCES["spec_gap_repeat"],
            "spec_gap_repeat",
            ("x", -1),
        ),
        (
            SOURCES["spec_gap_rotate_left"],
            "spec_gap_rotate_left",
            ([1, 2], 3),
        ),
        (
            SOURCES["spec_gap_rotate_right"],
            "spec_gap_rotate_right",
            ([1, 2], 3),
        ),
        (
            SOURCES["spec_gap_swap"],
            "spec_gap_swap",
            ([1, 2], 0, 2),
        ),
        (
            SOURCES["spec_gap_string_slice"],
            "spec_gap_string_slice",
            ("éx",),
        ),
    ],
)
def test_python_boundary_errors_are_not_silent(source, function_name, args):
    namespace = _run_python(source)
    if function_name in {
        "spec_gap_repeat",
        "spec_gap_rotate_left",
        "spec_gap_rotate_right",
    }:
        error = ValueError
    elif function_name == "spec_gap_swap":
        error = IndexError
    else:
        error = UnicodeDecodeError
    with pytest.raises(error):
        namespace[function_name](*args)


_NON_EXHAUSTIVE_MATCHES = [
    (
        "Option",
        """
fn spec_non_exhaustive_option(value: Option<i64>) -> i64 {
    match value { Some(item) => item }
}
""",
        lambda namespace: namespace["_Option"].none(),
    ),
    (
        "Result",
        """
fn spec_non_exhaustive_result(value: Result<i64, String>) -> i64 {
    match value { Ok(item) => item }
}
""",
        lambda namespace: namespace["_Result"].from_err("failure"),
    ),
]


@pytest.mark.parametrize(
    ("scrutinee_type", "source"),
    [(kind, source) for kind, source, _factory in _NON_EXHAUSTIVE_MATCHES],
)
def test_native_non_exhaustive_known_match_is_compile_error(
    native_backend, scrutinee_type, source,
):
    with pytest.warns(
        UserWarning,
        match=f"non-exhaustive {scrutinee_type} match",
    ), pytest.raises(RustCompileError):
        native_backend.compile(parse(lex(source)))


@pytest.mark.parametrize(
    ("scrutinee_type", "source", "make_missing_value"),
    _NON_EXHAUSTIVE_MATCHES,
)
def test_python_non_exhaustive_known_match_warns_and_raises(
    scrutinee_type, source, make_missing_value,
):
    with pytest.warns(
        UserWarning,
        match=f"non-exhaustive {scrutinee_type} match",
    ) as warnings:
        namespace = _run_python(source)
    assert len(warnings) == 1
    function_name = (
        "spec_non_exhaustive_option"
        if scrutinee_type == "Option"
        else "spec_non_exhaustive_result"
    )
    with pytest.raises(RuntimeError, match=f"non-exhaustive {scrutinee_type}"):
        namespace[function_name](make_missing_value(namespace))
