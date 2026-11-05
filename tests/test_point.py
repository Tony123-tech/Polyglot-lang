"""
Golden test for v21.2 OOP: struct + impl + &self.

Covers:
- struct definition -> @dataclass
- impl block -> methods merged into the class
- &self parameter -> `self` prepended automatically
- struct literal -> dataclass constructor
- method call -> Python attribute call
- return type annotation -> int
"""

import sys
from pathlib import Path

# Allow running from project root or tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.python.translator import compile_rust
from lexer import Lexer
from parser import Parser

SRC = """
struct Point { x: i64, y: i64 }

impl Point {
    fn new(x: i64, y: i64) -> Point { Point { x: x, y: y } }
    fn sum(&self) -> i64 { self.x + self.y }
}

fn main() -> i64 {
    let p = Point::new(1, 2);
    p.sum()
}
"""


def _run(source: str) -> dict:
    tokens = Lexer(source).lex()
    program = Parser(tokens).parse_program()
    return compile_rust(program)


def test_point_sum_returns_three():
    ns = _run(SRC)
    assert ns["main"]() == 3


def test_point_is_dataclass_with_fields():
    ns = _run(SRC)
    p = ns["Point"](3, 4)
    assert p.x == 3
    assert p.y == 4


def test_point_sum_method():
    ns = _run(SRC)
    p = ns["Point"](10, 20)
    assert p.sum() == 30


def test_point_new_constructor():
    ns = _run(SRC)
    p = ns["Point"].new(7, 8)
    assert isinstance(p, ns["Point"])
    assert p.x == 7
    assert p.y == 8


def test_point_new_accepts_positional():
    ns = _run(SRC)
    p = ns["Point"].new(1, 2)
    assert p.sum() == 3


def test_point_has_repr_from_dataclass():
    ns = _run(SRC)
    p = ns["Point"](1, 2)
    assert "x=1" in repr(p)
    assert "y=2" in repr(p)
