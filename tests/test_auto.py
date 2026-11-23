from __future__ import annotations

from typing import cast

from backends import get_backend
from backends.auto import AutoBackend
from compiler.core.lexer import lex
from compiler.core.parser import parse


def test_auto_selects_zig_for_simd() -> None:
    source = """
    fn vector_add(a: Simd4<f64>, b: Simd4<f64>) -> Simd4<f64> {
        a + b
    }
    """
    program = parse(lex(source))
    backend = cast(AutoBackend, get_backend("auto", trusted=True))
    target = backend.analyze_ast(program)
    assert target == "zig"


def test_auto_selects_zig_for_saturating() -> None:
    source = """
    fn sat_add(a: i64, b: i64) -> i64 {
        a +| b
    }
    """
    program = parse(lex(source))
    backend = cast(AutoBackend, get_backend("auto", trusted=True))
    target = backend.analyze_ast(program)
    assert target == "zig"


def test_auto_selects_native_for_parallel() -> None:
    source = """
    #[parallel]
    fn par_sum(values: Vec<i64>) -> i64 {
        let mut total: i64 = 0;
        for value in values {
            total = total + value;
        }
        total
    }
    """
    program = parse(lex(source))
    backend = cast(AutoBackend, get_backend("auto", trusted=True))
    target = backend.analyze_ast(program)
    assert target == "native"


def test_auto_selects_python_for_simple() -> None:
    source = """
    fn add(a: i64, b: i64) -> i64 {
        a + b
    }
    """
    program = parse(lex(source))
    backend = cast(AutoBackend, get_backend("auto", trusted=True))
    target = backend.analyze_ast(program)
    assert target == "python"


def test_auto_compiles_simd_program() -> None:
    source = """
    fn vector_add(a: Simd4<f64>, b: Simd4<f64>) -> Simd4<f64> {
        a + b
    }
    """
    program = parse(lex(source))
    ns = get_backend("auto", trusted=True).compile(program)
    assert "vector_add" in ns


def test_auto_compiles_saturating_program() -> None:
    source = """
    fn sat_add(a: i64, b: i64) -> i64 {
        a +| b
    }
    """
    program = parse(lex(source))
    ns = get_backend("auto", trusted=True).compile(program)
    assert "sat_add" in ns


def test_auto_compiles_parallel_program() -> None:
    source = """
    #[parallel]
    fn par_sum(values: Vec<i64>) -> i64 {
        let mut total: i64 = 0;
        for value in values {
            total = total + value;
        }
        total
    }
    """
    program = parse(lex(source))
    ns = get_backend("auto", trusted=True).compile(program)
    assert "par_sum" in ns
