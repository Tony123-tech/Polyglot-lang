# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N7: usize and isize are distinct from i64."""

from __future__ import annotations

import pytest

from backends.base import get_backend
from backends.python.runtime import (
    HOST_TARGET_WIDTH,
    get_target_width,
    set_target_width,
)
from lexer import lex
from parser import parse

SOURCES = {
    "n7_explicit_cast": """
fn n7_len_as_i64(values: Vec<i64>) -> i64 {
    values.len() as i64
}
fn n7_len_as_usize(values: Vec<i64>) -> usize {
    values.len()
}
fn n7_cast_param_to_usize(value: i64) -> usize {
    value as usize
}
fn n7_index(values: Vec<i64>, index: i64) -> i64 {
    values[index as usize]
}
fn n7_take_i64(value: i64) -> i64 { value }
fn n7_call_explicit(values: Vec<i64>) -> i64 {
    n7_take_i64(values.len() as i64)
}
fn n7_range_sum(values: Vec<i64>) -> i64 {
    let mut total: i64 = 0;
    for index in 0..values.len() {
        total = total + values[index];
    }
    total
}
fn n7_range_i64_sum(length: i64) -> i64 {
    let mut total: i64 = 0;
    for index in 0..length {
        total = total + index;
    }
    total
}
fn n7_vec_length(values: Vec<i64>) -> usize { values.len() }
fn n7_string_length(value: String) -> usize { value.len() }
struct N7Length { value: i64 }
impl N7Length {
    fn new(value: i64) -> N7Length { N7Length { value: value } }
    fn len(&self) -> i64 { self.value }
}
fn n7_user_length(value: N7Length) -> i64 { value.len() }
""",
}


def _run_python(source: str) -> dict:
    from backends.python.translator import compile_rust

    return compile_rust(parse(lex(source)))


def _implicit_i64_source() -> str:
    return "fn n7_implicit_i64(values: Vec<i64>) -> i64 { values.len() }"


def _assert_boundary_error(
    error: Exception,
    source_type: str = "usize",
    expected_type: str = "i64",
) -> None:
    expected = (
        f"error: type mismatch at line {error.line}, column {error.col}\n"
        f"  found:    {source_type}\n"
        f"  expected: {expected_type}\n"
        f"  help: write `as {expected_type}` here\n"
        "  note: `as` may wrap or truncate the value; check the range first"
    )
    assert str(error) == expected


def test_python_explicit_len_cast_and_usize_return():
    source = next(iter(SOURCES.values()))
    ns = _run_python(source)
    assert ns["n7_len_as_i64"]([1, 2, 3]) == 3
    assert ns["n7_len_as_usize"]([1, 2, 3]) == 3
    assert ns["n7_index"]([10, 20], 1) == 20
    assert ns["n7_call_explicit"]([1, 2, 3]) == 3
    assert ns["n7_range_sum"]([1, 2, 3]) == 6
    assert ns["n7_range_i64_sum"](4) == 6
    assert ns["n7_vec_length"]([1, 2]) == 2
    assert ns["n7_string_length"]("café") == 5
    assert ns["n7_user_length"](ns["N7Length"](8)) == 8


@pytest.mark.native
def test_native_explicit_len_cast_and_usize_return(native_ns):
    assert native_ns["n7_len_as_i64"]([1, 2, 3]) == 3
    assert native_ns["n7_len_as_usize"]([1, 2, 3]) == 3
    assert native_ns["n7_index"]([10, 20], 1) == 20
    assert native_ns["n7_call_explicit"]([1, 2, 3]) == 3
    assert native_ns["n7_range_sum"]([1, 2, 3]) == 6
    assert native_ns["n7_range_i64_sum"](4) == 6
    assert native_ns["n7_vec_length"]([1, 2]) == 2
    assert native_ns["n7_string_length"]("café") == 5
    assert native_ns["n7_user_length"](native_ns["N7Length"](8)) == 8


def test_python_implicit_len_return_reports_boundary_error():
    from backends.python.translator import TypeErrorForBoundary

    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(_implicit_i64_source())
    _assert_boundary_error(exc_info.value)


def test_native_implicit_len_return_reports_boundary_error(tmp_path):
    from backends.rust.compiler import (
        NativeBackend,
    )
    from backends.rust.compiler import (
        TypeErrorForBoundary as NativeTypeErrorForBoundary,
    )

    with pytest.raises(NativeTypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(
            parse(lex(_implicit_i64_source())),
        )
    _assert_boundary_error(exc_info.value)


def test_python_call_argument_requires_explicit_cast():
    from backends.python.translator import TypeErrorForBoundary

    source = """
fn n7_take_i64(value: i64) -> i64 { value }
fn n7_bad_call(values: Vec<i64>) -> i64 {
    n7_take_i64(values.len())
}
"""
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(source)
    _assert_boundary_error(exc_info.value)


def test_native_call_argument_requires_explicit_cast(tmp_path):
    from backends.rust.compiler import (
        NativeBackend,
    )
    from backends.rust.compiler import (
        TypeErrorForBoundary as NativeTypeErrorForBoundary,
    )

    source = """
fn n7_take_i64(value: i64) -> i64 { value }
fn n7_bad_call(values: Vec<i64>) -> i64 {
    n7_take_i64(values.len())
}
"""
    with pytest.raises(NativeTypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))
    _assert_boundary_error(exc_info.value)


def test_python_implicit_len_let_reports_boundary_error():
    from backends.python.translator import TypeErrorForBoundary

    source = (
        "fn n7_implicit_let(values: Vec<i64>) -> i64 { "
        "let length: i64 = values.len(); length }"
    )
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(source)
    _assert_boundary_error(exc_info.value)


def test_python_i64_index_requires_explicit_cast():
    from backends.python.translator import TypeErrorForBoundary

    source = (
        "fn n7_implicit_index(values: Vec<i64>, index: i64) -> i64 { "
        "values[index] }"
    )
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(source)
    _assert_boundary_error(exc_info.value, "i64", "usize")


def test_native_i64_index_requires_explicit_cast(tmp_path):
    from backends.rust.compiler import (
        NativeBackend,
    )
    from backends.rust.compiler import (
        TypeErrorForBoundary as NativeTypeErrorForBoundary,
    )

    source = (
        "fn n7_implicit_index(values: Vec<i64>, index: i64) -> i64 { "
        "values[index] }"
    )
    with pytest.raises(NativeTypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))
    _assert_boundary_error(exc_info.value, "i64", "usize")


def test_python_range_variable_inherits_usize_type():
    from backends.python.translator import TypeErrorForBoundary

    source = """
fn n7_bad_range_sum(values: Vec<i64>) -> i64 {
    let mut total: i64 = 0;
    for index in 0..values.len() {
        total = total + index;
    }
    total
}
"""
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(source)
    _assert_boundary_error(exc_info.value, "usize", "i64")


def test_native_range_variable_inherits_usize_type(tmp_path):
    from backends.rust.compiler import (
        NativeBackend,
    )
    from backends.rust.compiler import (
        TypeErrorForBoundary as NativeTypeErrorForBoundary,
    )

    source = """
fn n7_bad_range_sum(values: Vec<i64>) -> i64 {
    let mut total: i64 = 0;
    for index in 0..values.len() {
        total = total + index;
    }
    total
}
"""
    with pytest.raises(NativeTypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))
    _assert_boundary_error(exc_info.value, "usize", "i64")


def test_python_user_len_method_uses_declared_return_type():
    from backends.python.translator import TypeErrorForBoundary

    source = """
struct N7CustomLength { value: i64 }
impl N7CustomLength {
    fn len(&self) -> i64 { self.value }
}
fn n7_bad_user_len(value: N7CustomLength) -> i64 {
    let size: usize = value.len();
    size as i64
}
"""
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        _run_python(source)
    _assert_boundary_error(exc_info.value, "i64", "usize")


def test_native_user_len_method_uses_declared_return_type(tmp_path):
    from backends.rust.compiler import (
        NativeBackend,
    )
    from backends.rust.compiler import (
        TypeErrorForBoundary as NativeTypeErrorForBoundary,
    )

    source = """
struct N7CustomLength { value: i64 }
impl N7CustomLength {
    fn len(&self) -> i64 { self.value }
}
fn n7_bad_user_len(value: N7CustomLength) -> i64 {
    let size: usize = value.len();
    size as i64
}
"""
    with pytest.raises(NativeTypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))
    _assert_boundary_error(exc_info.value, "i64", "usize")


@pytest.mark.parametrize(
    "source",
    [
        "fn n7_negative() -> usize { -1 as usize }",
        "fn n7_too_large() -> usize { 18446744073709551616 as usize }",
    ],
)
def test_python_usize_cast_overflow_raises(source: str):
    namespace = _run_python(source)
    function_name = source.split("fn ", 1)[1].split("(", 1)[0]
    with pytest.raises(OverflowError):
        namespace[function_name]()


def test_python_usize_cast_uses_selectable_target_width():
    source = (
        "fn n7_target_width(value: i64) -> usize { value as usize }"
    )
    namespace = _run_python(source)
    original_width = get_target_width()
    try:
        set_target_width(32)
        with pytest.raises(OverflowError):
            namespace["n7_target_width"](1 << 32)
        assert namespace["n7_target_width"]((1 << 32) - 1) == (1 << 32) - 1
    finally:
        set_target_width(original_width)


def test_python_backend_target_width_is_instance_scoped():
    source = "fn n7_target_width(value: i64) -> usize { value as usize }"
    initial_width = get_target_width()
    narrow = get_backend("python", trusted=True, target_width=32).compile(
        parse(lex(source)),
    )
    with pytest.raises(OverflowError):
        narrow["n7_target_width"](1 << 32)
    assert narrow["n7_target_width"]((1 << 32) - 1) == (1 << 32) - 1
    assert get_target_width() == initial_width

    default = get_backend("python", trusted=True).compile(parse(lex(source)))
    assert default["n7_target_width"]((1 << HOST_TARGET_WIDTH) - 1) == (
        (1 << HOST_TARGET_WIDTH) - 1
    )
    with pytest.raises(OverflowError):
        default["n7_target_width"](1 << HOST_TARGET_WIDTH)
