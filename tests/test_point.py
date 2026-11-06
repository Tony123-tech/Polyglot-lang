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
    fn duplicate(&self) -> Point { self.clone() }
}
impl Point {
    fn product(&self) -> i64 { self.x * self.y }
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


def test_point_method_returns_self():
    ns = _run(SRC)
    point = ns["Point"](5, 8)
    duplicate = point.duplicate()

    assert duplicate is not point
    assert (duplicate.x, duplicate.y) == (5, 8)


def test_point_methods_across_impl_blocks():
    ns = _run(SRC)
    point = ns["Point"](5, 8)

    assert point.sum() == 13
    assert point.product() == 40


def test_python_struct_enum_impl_program_regression():
    ns = _run("""
struct BatchTenPoint { x: i64 }
enum BatchTenChoice { A, B }
impl BatchTenPoint {
    fn get_x(&self) -> i64 { self.x }
}
impl BatchTenChoice {
    fn value(&self) -> i64 {
        match self {
            BatchTenChoice::A => 1,
            BatchTenChoice::B => 2,
        }
    }
}
""")
    point = ns["BatchTenPoint"](19)

    assert point.get_x() == 19
    assert ns["BatchTenChoice"].A.value() == 1
    assert ns["BatchTenChoice"].B.value() == 2


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


def test_python_struct_field_of_later_struct():
    source = """
struct Outer { inner: Inner }
struct Inner { value: i64 }

fn create() -> Outer {
    Outer { inner: Inner { value: 21 } }
}
"""
    ns = _run(source)
    outer = ns["create"]()

    assert isinstance(outer.inner, ns["Inner"])
    assert outer.inner.value == 21


def test_python_unit_enum_variant_representation():
    ns = _run("""
enum Choice { A, B }
impl Choice {
    fn code(&self) -> i64 { 7 }
}
""")

    assert ns["Choice"].A != ns["Choice"].B
    assert ns["Choice"].A.name == "A"
    assert ns["Choice"].A.code() == 7


def test_python_payload_enum_representation():
    source = """
enum Maybe { Some(i64), None }

fn unwrap_or_zero(value: Maybe) -> i64 {
    match value {
        Maybe::Some(number) => number,
        Maybe::None => 0,
    }
}
"""
    ns = _run(source)

    assert ns["unwrap_or_zero"](ns["Maybe"].Some(42)) == 42
    assert ns["unwrap_or_zero"](getattr(ns["Maybe"], "None_")) == 0


def test_python_struct_with_vec_field():
    ns = _run("""
struct VecHolder { values: Vec<i64> }
""")
    holder = ns["VecHolder"]([2, 4, 6])

    assert holder.values == [2, 4, 6]


def test_python_generic_struct_defaults_type_parameter_to_int():
    ns = _run("""
struct GenericHolder<T> { value: T }
impl<T> GenericHolder<T> {
    fn new(value: T) -> GenericHolder<T> {
        GenericHolder { value: value }
    }
}
""")
    holder = ns["GenericHolder"].new(34)

    assert holder.value == 34
