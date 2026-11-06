"""N2b native backend tests: Vec methods."""

import pytest

SOURCES = {
    "test_vec_len": """
fn test_vec_len() -> i64 {
    let v: Vec<i64> = vec![10, 20, 30];
    v.len()
}
""",
    "test_vec_push": """
fn test_vec_push() -> i64 {
    let mut v: Vec<i64> = vec![];
    v.push(1);
    v.push(2);
    v.push(3);
    v.len()
}
""",
    "test_vec_push_and_sum": """
fn test_vec_push_and_sum() -> i64 {
    let mut v: Vec<i64> = vec![];
    v.push(10);
    v.push(20);
    let mut total: i64 = 0;
    for x in v { total = total + x; }
    total
}
""",
    "test_vec_pop_returns_value": """
fn test_vec_pop_returns_value() -> i64 {
    let mut v: Vec<i64> = vec![10, 20, 30];
    match v.pop() {
        Some(x) => x,
        None => 0,
    }
}
""",
    "test_vec_pop_empty_returns_none": """
fn test_vec_pop_empty_returns_none() -> i64 {
    let mut v: Vec<i64> = vec![];
    match v.pop() {
        Some(x) => x,
        None => -1,
    }
}
""",
    "test_vec_index": """
fn test_vec_index(v: Vec<i64>, i: i64) -> i64 { v[i] }
""",
    "test_vec_index_zero": """
fn test_vec_index_zero(v: Vec<i64>) -> i64 { v[0] }
""",
    "test_vec_mixed": """
fn test_vec_mixed() -> i64 {
    let mut v: Vec<i64> = vec![1, 2, 3];
    v.push(4);
    v.push(5);
    let x = v.pop();
    let first = v[0];
    let length = v.len();
    first + length
}
""",
}


@pytest.mark.native
def test_vec_len(native_ns):
    assert native_ns["test_vec_len"]() == 3


@pytest.mark.native
def test_vec_push(native_ns):
    assert native_ns["test_vec_push"]() == 3


@pytest.mark.native
def test_vec_push_and_sum(native_ns):
    assert native_ns["test_vec_push_and_sum"]() == 30


@pytest.mark.native
def test_vec_pop_returns_value(native_ns):
    assert native_ns["test_vec_pop_returns_value"]() == 30


@pytest.mark.native
def test_vec_pop_empty_returns_none(native_ns):
    assert native_ns["test_vec_pop_empty_returns_none"]() == -1


@pytest.mark.native
def test_vec_index(native_ns):
    assert native_ns["test_vec_index"]([10, 20, 30], 1) == 20


@pytest.mark.native
def test_vec_index_zero(native_ns):
    assert native_ns["test_vec_index_zero"]([42, 99]) == 42


@pytest.mark.native
def test_vec_mixed(native_ns):
    assert native_ns["test_vec_mixed"]() == 5
