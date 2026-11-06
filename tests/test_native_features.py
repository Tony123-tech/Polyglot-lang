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
    fn duplicate(&self) -> Point { self.clone() }
}
impl Point {
    fn product(&self) -> i64 { self.x * self.y }
}

fn test_native_point_make(x: i64, y: i64) -> i64 {
    let p = Point::new(x, y);
    p.sum()
}
""",
    "test_native_multiple_pyclasses": """
struct BatchOne { value: i64 }
impl BatchOne {
    fn new(value: i64) -> BatchOne { BatchOne { value: value } }
}

struct BatchTwo { label: String }
impl BatchTwo {
    fn new(label: String) -> BatchTwo { BatchTwo { label: label } }
}

fn test_native_multiple_pyclasses() -> i64 { 1 }
""",
    "test_native_nested_struct_field": """
struct Outer { inner: Inner }
struct Inner { value: i64 }

impl Inner {
    fn new(value: i64) -> Inner { Inner { value: value } }
}
impl Outer {
    fn new(inner: Inner) -> Outer { Outer { inner: inner } }
}

fn test_native_nested_struct_field() -> i64 { 1 }
""",
    "test_native_vec_struct_field": """
struct VecHolder { values: Vec<i64> }
impl VecHolder {
    fn new(values: Vec<i64>) -> VecHolder {
        VecHolder { values: values }
    }
}
""",
    "test_native_generic_struct": """
struct GenericHolder<T> { value: T }
impl<T> GenericHolder<T> {
    fn new(value: T) -> GenericHolder<T> {
        GenericHolder { value: value }
    }
}
""",
    "test_native_choice_enum": """
enum NativeChoiceForBatch3 { A, B }
impl NativeChoiceForBatch3 {
    fn code(&self) -> i64 { 7 }
}
""",
    "test_native_payload_enum": """
enum NativeMaybeForBatch4 { Some(i64), None }
""",
    "test_native_struct_enum_impl_regression": """
struct BatchTenPoint { x: i64 }
enum BatchTenChoice { A, B }
impl BatchTenPoint {
    fn new(x: i64) -> BatchTenPoint { BatchTenPoint { x: x } }
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
def test_native_struct_method_returns_self(native_ns):
    point = native_ns["Point"](5, 8)
    duplicate = point.duplicate()

    assert duplicate is not point
    assert (duplicate.x, duplicate.y) == (5, 8)


@pytest.mark.native
def test_native_multiple_impl_blocks(native_ns):
    point = native_ns["Point"](5, 8)

    assert point.sum() == 13
    assert point.product() == 40


@pytest.mark.native
def test_native_register_multiple_pyclasses(native_ns):
    first = native_ns["BatchOne"](12)
    second = native_ns["BatchTwo"]("polyglot")

    assert first.value == 12
    assert second.label == "polyglot"


@pytest.mark.native
def test_native_struct_field_of_struct(native_ns):
    inner = native_ns["Inner"](21)
    outer = native_ns["Outer"](inner)

    assert outer.inner.value == 21


@pytest.mark.native
def test_native_struct_with_vec_field(native_ns):
    holder = native_ns["VecHolder"]([2, 4, 6])

    assert holder.values == [2, 4, 6]


@pytest.mark.native
def test_native_generic_struct_defaults_to_i64(native_ns):
    holder = native_ns["GenericHolder"](34)

    assert holder.value == 34


@pytest.mark.native
def test_native_unit_enum_variants(native_ns):
    choice = native_ns["NativeChoiceForBatch3"]

    assert choice.A != choice.B
    assert choice.A == choice.A
    assert choice.A.code() == 7


@pytest.mark.skip(
    reason="Payload enums are not exposed as PyO3 classes in the native backend"
)
@pytest.mark.native
def test_native_payload_enum_is_exposed(native_ns):
    assert "NativeMaybeForBatch4" in native_ns


@pytest.mark.native
def test_native_struct_enum_impl_regression(native_ns):
    point = native_ns["BatchTenPoint"](19)
    choice = native_ns["BatchTenChoice"]

    assert point.get_x() == 19
    assert choice.A.value() == 1
    assert choice.B.value() == 2


@pytest.mark.native
def test_native_if_else(native_ns):
    assert native_ns["test_native_abs"](-5) == 5
    assert native_ns["test_native_abs"](5) == 5
    assert native_ns["test_native_abs"](0) == 0
