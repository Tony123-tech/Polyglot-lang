"""M4-remaining: additional integer types."""

from __future__ import annotations

import pytest

from backends import get_backend
from lexer import lex
from parser import parse


def _run_py(src: str):
    ns = get_backend("python", trusted=True).compile(parse(lex(src)))
    return ns["main"]()


def _run_native(src: str, tmp_path):
    from backends.rust.compiler import NativeBackend

    ns = NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    return ns["main"]()


def test_casts_i8_i64_succeed_python():
    assert _run_py("fn main() -> i64 { let y = 1i8 as i64; y }\n") == 1
    assert _run_py("fn main() -> i64 { let y = 1i64 as i8; y as i64 }\n") == 1


@pytest.mark.native
def test_casts_i8_i64_succeed_native(tmp_path):
    assert _run_native("fn main() -> i64 { let y = 1i8 as i64; y }\n", tmp_path) == 1
    assert (
        _run_native("fn main() -> i64 { let y = 1i64 as i8; y as i64 }\n", tmp_path)
        == 1
    )


def test_literal_out_of_range_raises_overflow_python():
    with pytest.raises(OverflowError):
        _run_py("fn main() -> i64 { let y = 200i64 as i8; y as i64 }\n")


@pytest.mark.native
def test_literal_wraps_native(tmp_path):
    # Native wraps (Rust as-cast semantics)
    assert (
        _run_native("fn main() -> i64 { let y = 200i64 as i8; y as i64 }\n", tmp_path)
        == -56
    )


def test_implicit_conversion_boundary_error_e001():
    src = "fn main(a: i64) -> i8 { let x: i8 = a; x }\n"
    with pytest.raises(Exception) as ei:
        get_backend("python", trusted=True).compile(parse(lex(src)))
    msg = str(ei.value)
    assert "error[E001]:" in msg
    assert "found:" in msg and "expected:" in msg


@pytest.mark.parametrize(
    "ty,val,cast_back",
    [
        ("i8", "127i8", "i64"),
        ("i16", "32767i16", "i64"),
        ("i32", "2147483647i32", "i64"),
        ("u8", "255u8", "i64"),
        ("u16", "65535u16", "i64"),
        ("u32", "4294967295u32", "i64"),
        ("u64", "1u64", "i64"),
        ("i128", "1i128", "i64"),
        ("u128", "1u128", "i64"),
    ],
)
def test_cross_backend_matrix_python(ty, val, cast_back):
    src = f"fn main() -> i64 {{ let x: {ty} = {val}; x as {cast_back} }}\n"
    # just ensure it compiles and runs in Python
    out = _run_py(src)
    assert isinstance(out, int)


@pytest.mark.native
@pytest.mark.parametrize(
    "src,expected",
    [
        ("fn main() -> i64 { let x = 1i8 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1i16 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1i32 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1i128 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1u8 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1u16 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1u32 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1u64 as i64; x }", 1),
        ("fn main() -> i64 { let x = 1u128 as i64; x }", 1),
        ("fn main() -> i64 { let x = 200i64 as i8; x as i64 }", -56),
        ("fn main() -> i64 { let x = 300i64 as u8; x as i64 }", 44),
        ("fn main() -> i64 { let x = 255u8 as i8; x as i64 }", -1),
        ("fn main() -> i64 { let x = 1i128 as u128; x as i64 }", 1),
    ],
)
def test_native_integer_matrix(src, expected, tmp_path):
    from backends.rust.compiler import NativeBackend

    ns = NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    assert ns["main"]() == expected


def test_native_boundary_error_for_implicit_conversion():
    from backends.rust.compiler import NativeBackend, TypeErrorForBoundary

    src = "fn main(a: i64) -> i8 { let x: i8 = a; x }"
    with pytest.raises(TypeErrorForBoundary):
        NativeBackend(cache_dir="/tmp/m4_test").compile(parse(lex(src)))


# ---------------------------------------------------------------------------
# C17: full cross-backend integer matrix
# ---------------------------------------------------------------------------

INT_SPECS = {
    "i8": (8, True),
    "i16": (16, True),
    "i32": (32, True),
    "i64": (64, True),
    "i128": (128, True),
    "u8": (8, False),
    "u16": (16, False),
    "u32": (32, False),
    "u64": (64, False),
    "u128": (128, False),
}


def _wrap(value: int, bits: int, signed: bool) -> int:
    v = value & ((1 << bits) - 1)
    if signed and v >= (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _wrap_i64(value: int) -> int:
    return _wrap(value, 64, True)


def _bounds(ty: str) -> tuple[int, int]:
    bits, signed = INT_SPECS[ty]
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_integer_matrix_min_max_zero_python(ty):
    lo, hi = _bounds(ty)
    src = f"""
fn main() -> i64 {{
    let a: {ty} = {lo};
    let b: {ty} = {hi};
    let c: {ty} = 0;
    if (a as i64) == {_wrap_i64(lo)}
        && (b as i64) == {_wrap_i64(hi)}
        && (c as i64) == 0 {{
        1
    }} else {{
        0
    }}
}}
"""
    assert _run_py(src) == 1


@pytest.mark.native
@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_integer_matrix_min_max_zero_native(ty, tmp_path):
    lo, hi = _bounds(ty)
    src = f"""
fn main() -> i64 {{
    let a: {ty} = {lo};
    let b: {ty} = {hi};
    let c: {ty} = 0;
    if (a as i64) == {_wrap_i64(lo)}
        && (b as i64) == {_wrap_i64(hi)}
        && (c as i64) == 0 {{
        1
    }} else {{
        0
    }}
}}
"""
    assert _run_native(src, tmp_path) == 1


@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_python_literal_out_of_range_raises(ty):
    lo, hi = _bounds(ty)
    for literal in (hi + 1, lo - 1):
        # Parenthesise so `-N as ty` is parsed as `(-N) as ty`, not `-(N as ty)`.
        src = f"fn main() -> i64 {{ let x = ({literal}) as {ty}; x as i64 }}\n"
        with pytest.raises(OverflowError):
            _run_py(src)


@pytest.mark.native
@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_native_literal_out_of_range_rejected(ty, tmp_path):
    # Rust infers the literal type from the cast target, so an out-of-range
    # literal is a compile error rather than a run-time wrap.
    from backends.rust.compiler import NativeBackend, RustCompileError

    lo, hi = _bounds(ty)
    for literal in (hi + 1, lo - 1):
        src = f"fn main() -> i64 {{ let x = ({literal}) as {ty}; x as i64 }}\n"
        with pytest.raises(RustCompileError):
            NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))


@pytest.mark.native
@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_native_variable_cast_wraps(ty, tmp_path):
    # A cast from a variable performs Rust's run-time `as` conversion, which
    # wraps for integers.
    bits, signed = INT_SPECS[ty]
    _lo, hi = _bounds(ty)
    value = hi + 1 if hi + 1 <= (1 << 63) - 1 else -1
    src = f"fn main() -> i64 {{ let x: i64 = {value}; let y = x as {ty}; y as i64 }}\n"
    assert _run_native(src, tmp_path) == _wrap_i64(_wrap(value, bits, signed))


@pytest.mark.parametrize("ty", list(INT_SPECS))
def test_python_variable_cast_wraps(ty):
    bits, signed = INT_SPECS[ty]
    _lo, hi = _bounds(ty)
    # Choose an in-i64 value outside the target range.
    value = hi + 1 if hi + 1 <= (1 << 63) - 1 else -1
    src = f"fn main() -> i64 {{ let x: i64 = {value}; let y = x as {ty}; y as i64 }}\n"
    assert _run_py(src) == _wrap_i64(_wrap(value, bits, signed))


# ---------------------------------------------------------------------------
# C18: native boundary matrix, E001 six-line format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", list(INT_SPECS))
@pytest.mark.parametrize("target_type", list(INT_SPECS))
def test_native_implicit_boundary_matrix(source_type, target_type):
    if source_type == target_type:
        return
    from backends.rust.compiler import NativeBackend, TypeErrorForBoundary

    src = (
        f"fn main(a: {source_type}) -> {target_type} {{ let x: {target_type} = a; x }}"
    )
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        NativeBackend(cache_dir="/tmp/m4_boundary_matrix").compile(parse(lex(src)))
    message = str(exc_info.value)
    assert "error[E001]:" in message
    assert "found:" in message and "expected:" in message
    assert "(span " in message


# ---------------------------------------------------------------------------
# C19: i128/u128 above 2^64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            (
                "fn main() -> i64 { let x: i128 = 18446744073709551616i128; "
                "(x / 18446744073709551616i128) as i64 }"
            ),
            1,
        ),
        (
            (
                "fn main() -> i64 { let x: u128 = 18446744073709551616u128; "
                "(x / 18446744073709551616u128) as i64 }"
            ),
            1,
        ),
    ],
)
def test_i128_u128_above_2_64_python(src, expected):
    assert _run_py(src) == expected


@pytest.mark.native
@pytest.mark.parametrize(
    "src,expected",
    [
        (
            (
                "fn main() -> i64 { let x: i128 = 18446744073709551616i128; "
                "(x / 18446744073709551616i128) as i64 }"
            ),
            1,
        ),
        (
            (
                "fn main() -> i64 { let x: u128 = 18446744073709551616u128; "
                "(x / 18446744073709551616u128) as i64 }"
            ),
            1,
        ),
    ],
)
def test_i128_u128_above_2_64_native(src, expected, tmp_path):
    assert _run_native(src, tmp_path) == expected


# ---------------------------------------------------------------------------
# C20: float-to-int casts
# ---------------------------------------------------------------------------


def test_float_to_int_truncates_toward_zero_both_backends(tmp_path):
    src = "fn main() -> i64 { let f: f64 = -3.9; f as i64 }"
    assert _run_py(src) == -3
    assert _run_native(src, tmp_path) == -3


@pytest.mark.native
def test_float_to_int_range_difference_is_documented(tmp_path):
    # Python wraps after truncation; native saturates like Rust.
    src = "fn main() -> i64 { let f: f64 = 300.0; let x = f as u8; x as i64 }"
    assert _run_py(src) == 44
    assert _run_native(src, tmp_path) == 255


# ---------------------------------------------------------------------------
# C21: char casts
# ---------------------------------------------------------------------------


def test_char_as_i64_both_backends(tmp_path):
    src = "fn main() -> i64 { let c = 'A'; c as i64 }"
    assert _run_py(src) == 65
    assert _run_native(src, tmp_path) == 65


def test_python_integer_as_char_invalid_code_points():
    for code in (0xD800, 0xDFFF, 0x110000):
        src = f"fn main() -> i64 {{ let x: i64 = {code}; let c = x as char; c as i64 }}"
        with pytest.raises(ValueError):
            _run_py(src)


def test_python_integer_as_char_valid_code_point():
    src = "fn main() -> i64 { let x: i64 = 65; let c = x as char; c as i64 }"
    assert _run_py(src) == 65


@pytest.mark.native
def test_native_u8_as_char_is_supported(tmp_path):
    src = "fn main() -> i64 { let x: u8 = 65; let c = x as char; c as i64 }"
    assert _run_native(src, tmp_path) == 65


@pytest.mark.native
def test_native_integer_as_char_other_widths_rejected(tmp_path):
    from backends.rust.compiler import NativeBackend, RustCompileError

    src = "fn main() -> i64 { let x: i64 = 65; let c = x as char; c as i64 }"
    with pytest.raises(RustCompileError) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    assert "only `u8` can be cast as `char`" in str(exc_info.value)


# ---------------------------------------------------------------------------
# C22: bool casts
# ---------------------------------------------------------------------------


def test_bool_as_i64_both_backends(tmp_path):
    src = "fn main() -> i64 { let b = true; b as i64 }"
    assert _run_py(src) == 1
    assert _run_native(src, tmp_path) == 1


def test_python_i64_as_bool():
    src = (
        "fn main() -> i64 { let x: i64 = 0; let y: i64 = 2; "
        "let a = x as bool; let b = y as bool; "
        "if a { 1 } else if b { 2 } else { 0 } }"
    )
    assert _run_py(src) == 2


@pytest.mark.native
def test_native_i64_as_bool_rejected(tmp_path):
    from backends.rust.compiler import NativeBackend, RustCompileError

    src = (
        "fn main() -> i64 { let x: i64 = 1; let b = x as bool; if b { 1 } else { 0 } }"
    )
    with pytest.raises(RustCompileError) as exc_info:
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    assert "cannot cast `i64` as `bool`" in str(exc_info.value)


def test_runtime_cast_negative_usize_raises_overflow():
    from backends.python.translator import _runtime_cast

    with pytest.raises(OverflowError):
        _runtime_cast(-1, "usize")
    assert _runtime_cast(7, "usize") == 7
