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
    assert _run_native("fn main() -> i64 { let y = 1i64 as i8; y as i64 }\n", tmp_path) == 1


def test_literal_out_of_range_raises_overflow_python():
    with pytest.raises(OverflowError):
        _run_py("fn main() -> i64 { let y = 200i64 as i8; y as i64 }\n")


@pytest.mark.native
def test_literal_wraps_native(tmp_path):
    # Native wraps (Rust as-cast semantics)
    assert _run_native("fn main() -> i64 { let y = 200i64 as i8; y as i64 }\n", tmp_path) == -56


def test_implicit_conversion_boundary_error_e001():
    src = "fn main(a: i64) -> i8 { let x: i8 = a; x }\n"
    with pytest.raises(Exception) as ei:
        get_backend("python", trusted=True).compile(parse(lex(src)))
    msg = str(ei.value)
    assert "error[E001]:" in msg
    assert "found:" in msg and "expected:" in msg


@pytest.mark.parametrize("ty,val,cast_back", [
    ("i8", "127i8", "i64"),
    ("i16", "32767i16", "i64"),
    ("i32", "2147483647i32", "i64"),
    ("u8", "255u8", "i64"),
    ("u16", "65535u16", "i64"),
    ("u32", "4294967295u32", "i64"),
    ("u64", "1u64", "i64"),
    ("i128", "1i128", "i64"),
    ("u128", "1u128", "i64"),
])
def test_cross_backend_matrix_python(ty, val, cast_back):
    src = f"fn main() -> i64 {{ let x: {ty} = {val}; x as {cast_back} }}\n"
    # just ensure it compiles and runs in Python
    out = _run_py(src)
    assert isinstance(out, int)
