"""Paired Python/native tests for bounded syntax additions."""

import pytest

from backends.python.translator import compile_rust
from compiler.core.lexer import lex
from compiler.core.parser import parse

SOURCES = {
    "let_else": """
fn syntax_let_else(value: Option<i64>) -> i64 {
    let Some(inner) = value else { return -1; };
    inner + 1
}
""",
    "struct_update": """
struct SyntaxPoint { x: i64, y: i64 }

fn syntax_point_new(x: i64, y: i64) -> SyntaxPoint {
    SyntaxPoint { x: x, y: y }
}

fn syntax_struct_update(point: SyntaxPoint) -> SyntaxPoint {
    SyntaxPoint { x: 9, ..point }
}

fn syntax_struct_update_value(point: SyntaxPoint) -> i64 {
    let updated = syntax_struct_update(point);
    updated.x + updated.y
}
""",
    "type_alias": """
type Score = i64;

fn syntax_type_alias(value: Score) -> Score {
    let boosted: Score = value + 1;
    boosted
}
""",
    "use_mod": """
use std::collections::HashMap;

mod syntax_inner {
    pub fn helper() -> i64 {
        7
    }
}

fn syntax_use_mod() -> i64 {
    syntax_inner::helper()
}
""",
    "labelled_loops": """
fn syntax_labelled_break() -> i64 {
    let mut total = 0;
    'stop: loop {
        total += 1;
        if total == 3 {
            break 'stop;
        }
    }
    total
}

fn syntax_labelled_continue() -> i64 {
    let mut total = 0;
    'outer: for i in 0..3 {
        for j in 0..3 {
            if j == 1 {
                continue 'outer;
            }
            if i == 2 {
                break 'outer;
            }
            total += i * 10 + j;
        }
    }
    total
}
""",
}


@pytest.mark.native
def test_native_let_else(native_ns):
    assert native_ns["syntax_let_else"](4) == 5
    assert native_ns["syntax_let_else"](None) == -1


def test_python_let_else():
    namespace = compile_rust(parse(lex(SOURCES["let_else"])))

    assert namespace["syntax_let_else"](namespace["_Option"].some(4)) == 5
    assert namespace["syntax_let_else"](namespace["_Option"].none()) == -1


@pytest.mark.native
def test_native_struct_update(native_ns):
    point = native_ns["syntax_point_new"](1, 2)

    assert native_ns["syntax_struct_update_value"](point) == 11


def test_python_struct_update():
    namespace = compile_rust(parse(lex(SOURCES["struct_update"])))
    point = namespace["syntax_point_new"](1, 2)

    assert namespace["syntax_struct_update_value"](point) == 11


@pytest.mark.native
def test_native_type_alias(native_ns):
    assert native_ns["syntax_type_alias"](4) == 5


def test_python_type_alias():
    namespace = compile_rust(parse(lex(SOURCES["type_alias"])))

    assert namespace["syntax_type_alias"](4) == 5


@pytest.mark.native
def test_native_use_and_inline_mod(native_ns):
    assert native_ns["syntax_use_mod"]() == 7


def test_python_use_and_inline_mod():
    namespace = compile_rust(parse(lex(SOURCES["use_mod"])))

    assert namespace["syntax_use_mod"]() == 7


@pytest.mark.native
def test_native_labelled_loop_control(native_ns):
    assert native_ns["syntax_labelled_break"]() == 3
    assert native_ns["syntax_labelled_continue"]() == 10


def test_python_labelled_loop_control():
    namespace = compile_rust(parse(lex(SOURCES["labelled_loops"])))

    assert namespace["syntax_labelled_break"]() == 3
    assert namespace["syntax_labelled_continue"]() == 10
