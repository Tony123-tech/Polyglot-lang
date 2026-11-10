# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N8: primitive type semantics (casts, integer division, overflow)."""

from __future__ import annotations

import pytest

from lexer import lex
from parser import parse

SOURCES = {
    "n8_casts": """
fn n8_cast_u8(value: i64) -> i64 { (value as u8) as i64 }
fn n8_cast_i8(value: i64) -> i64 { (value as i8) as i64 }
fn n8_cast_float(value: f64) -> i64 { value as i64 }
fn n8_cast_negative_float(value: f64) -> i64 { value as i64 }
fn n8_cast_to_float(value: i64) -> f64 { value as f64 }
fn n8_cast_bool(value: bool) -> i64 { value as i64 }
fn n8_char_code(value: char) -> i64 { value as i64 }
fn n8_char_from_code(value: u8) -> char { value as char }
fn n8_cast_f32(value: f64) -> f32 { value as f32 }
""",
    "n8_integer_math": """
fn n8_div(left: i64, right: i64) -> i64 { left / right }
fn n8_rem(left: i64, right: i64) -> i64 { left % right }
fn n8_float_div(left: f64, right: f64) -> f64 { left / right }
fn n8_float_rem(left: f64, right: f64) -> f64 { left % right }
""",
    "n8_overflow": """
fn n8_wrap_i8_add(value: i8) -> i8 {
    let mut result: i8 = value;
    result += 1;
    result
}
fn n8_wrap_i8_sub(value: i8) -> i8 {
    let mut result: i8 = value;
    result -= 1;
    result
}
fn n8_wrap_u8_add(value: u8) -> u8 {
    let mut result: u8 = value;
    result += 1;
    result
}
fn n8_wrap_i8_return(left: i8, right: i8) -> i8 { left + right }
fn n8_wrap_i16_mul(value: i16) -> i16 {
    let mut result: i16 = value;
    result *= 2;
    result
}
""",
    "n8_sized": """
fn n8_as_usize(value: i64) -> usize { value as usize }
fn n8_as_isize(value: i64) -> isize { value as isize }
fn n8_usize_add(value: usize) -> usize {
    let mut result: usize = value;
    result += 1;
    result
}
""",
}


def _run_python(src: str) -> dict:
    from backends.python.translator import compile_rust

    return compile_rust(parse(lex(src)))


_USIZE_MAX = (1 << 64) - 1
_F32_ONE_TENTH = 0.10000000149011612


@pytest.mark.native
def test_native_casts(native_ns):
    ns = native_ns
    assert ns["n8_cast_u8"](300) == 44
    assert ns["n8_cast_u8"](-1) == 255
    assert ns["n8_cast_i8"](200) == -56
    assert ns["n8_cast_float"](3.9) == 3
    assert ns["n8_cast_negative_float"](-3.9) == -3
    assert ns["n8_cast_to_float"](7) == 7.0
    assert ns["n8_cast_bool"](True) == 1
    assert ns["n8_cast_bool"](False) == 0
    assert ns["n8_char_code"]("a") == 97
    assert ns["n8_char_from_code"](97) == "a"
    assert ns["n8_cast_f32"](0.1) == _F32_ONE_TENTH


@pytest.mark.native
def test_native_integer_math(native_ns):
    ns = native_ns
    assert ns["n8_div"](7, 2) == 3
    assert ns["n8_div"](-7, 2) == -3
    assert ns["n8_div"](7, -2) == -3
    assert ns["n8_rem"](-7, 3) == -1
    assert ns["n8_rem"](7, -3) == 1
    assert ns["n8_float_div"](7.0, 2.0) == 3.5
    assert ns["n8_float_rem"](-7.5, 2.0) == -1.5


@pytest.mark.native
def test_native_overflow_wraps(native_ns):
    ns = native_ns
    assert ns["n8_wrap_i8_add"](127) == -128
    assert ns["n8_wrap_i8_sub"](-128) == 127
    assert ns["n8_wrap_u8_add"](255) == 0
    assert ns["n8_wrap_i8_return"](100, 100) == -56
    assert ns["n8_wrap_i16_mul"](20_000) == -25_536


@pytest.mark.native
def test_native_sized_casts(native_ns):
    ns = native_ns
    assert ns["n8_as_usize"](-1) == _USIZE_MAX
    assert ns["n8_as_isize"](-1) == -1
    assert ns["n8_usize_add"](_USIZE_MAX) == 0


def test_python_casts():
    ns = _run_python(SOURCES["n8_casts"])
    assert ns["n8_cast_u8"](300) == 44
    assert ns["n8_cast_u8"](-1) == 255
    assert ns["n8_cast_i8"](200) == -56
    assert ns["n8_cast_float"](3.9) == 3
    assert ns["n8_cast_negative_float"](-3.9) == -3
    assert ns["n8_cast_to_float"](7) == 7.0
    assert ns["n8_cast_bool"](True) == 1
    assert ns["n8_cast_bool"](False) == 0
    assert ns["n8_char_code"]("a") == 97
    assert ns["n8_char_from_code"](97) == "a"
    assert ns["n8_cast_f32"](0.1) == _F32_ONE_TENTH


def test_python_integer_math():
    ns = _run_python(SOURCES["n8_integer_math"])
    assert ns["n8_div"](7, 2) == 3
    assert ns["n8_div"](-7, 2) == -3
    assert ns["n8_div"](7, -2) == -3
    assert ns["n8_rem"](-7, 3) == -1
    assert ns["n8_rem"](7, -3) == 1
    assert ns["n8_float_div"](7.0, 2.0) == 3.5
    assert ns["n8_float_rem"](-7.5, 2.0) == -1.5


def test_python_overflow_wraps():
    ns = _run_python(SOURCES["n8_overflow"])
    assert ns["n8_wrap_i8_add"](127) == -128
    assert ns["n8_wrap_i8_sub"](-128) == 127
    assert ns["n8_wrap_u8_add"](255) == 0
    assert ns["n8_wrap_i8_return"](100, 100) == -56
    assert ns["n8_wrap_i16_mul"](20_000) == -25_536


def test_python_sized_casts():
    ns = _run_python(SOURCES["n8_sized"])
    assert ns["n8_as_usize"](-1) == _USIZE_MAX
    assert ns["n8_as_isize"](-1) == -1
    assert ns["n8_usize_add"](_USIZE_MAX) == 0
