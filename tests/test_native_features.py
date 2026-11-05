"""N1 native backend tests."""

import pytest

SOURCES = {
    "test_native_add": """
fn test_native_add() -> i64 {
    let a: i64 = 2;
    let b: i64 = 3;
    a + b
}
""",
    "test_native_bool": """
fn test_native_bool(x: bool) -> bool { !x }
""",
    "test_native_float": """
fn test_native_float(x: f64) -> f64 { x * 2.0 }
""",
    "test_native_string": """
fn test_native_string(name: String) -> String {
    format!("hello {}", name)
}
""",
    "test_native_vec_sum": """
fn test_native_vec_sum(xs: Vec<i64>) -> i64 {
    let mut total: i64 = 0;
    for x in xs { total = total + x; }
    total
}
""",
    "test_native_point_make": """
struct Point { x: i64, y: i64 }

impl Point {
    fn new(x: i64, y: i64) -> Point { Point { x: x, y: y } }
    fn sum(&self) -> i64 { self.x + self.y }
}

fn test_native_point_make(x: i64, y: i64) -> i64 {
    let p = Point::new(x, y);
    p.sum()
}
""",
    "test_native_abs": """
fn test_native_abs(x: i64) -> i64 {
    if x < 0 { 0 - x } else { x }
}
""",
}


@pytest.mark.native
def test_native_add(native_ns):
    assert native_ns["test_native_add"]() == 5


@pytest.mark.native
def test_native_bool_and_float(native_ns):
    assert native_ns["test_native_bool"](True) is False
    assert native_ns["test_native_bool"](False) is True
    assert native_ns["test_native_float"](1.5) == 3.0


@pytest.mark.native
def test_native_string(native_ns):
    assert native_ns["test_native_string"]("world") == "hello world"


@pytest.mark.native
def test_native_vec_sum(native_ns):
    assert native_ns["test_native_vec_sum"]([1, 2, 3, 4]) == 10


@pytest.mark.native
def test_native_point_struct(native_ns):
    assert native_ns["test_native_point_make"](3, 4) == 7


@pytest.mark.native
def test_native_if_else(native_ns):
    assert native_ns["test_native_abs"](-5) == 5
    assert native_ns["test_native_abs"](5) == 5
    assert native_ns["test_native_abs"](0) == 0
