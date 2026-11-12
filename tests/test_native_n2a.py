"""N2a native backend tests: while / loop / match."""

import pytest

SOURCES = {
    "test_while_count_down": """
fn test_while_count_down(n: i64) -> i64 {
    let mut total: i64 = 0;
    while n > 0 {
        total = total + n;
        n = n - 1;
    }
    total
}
""",
    "test_while_never_enters": """
fn test_while_never_enters(n: i64) -> i64 {
    let mut x: i64 = 0;
    while n < 0 { x = x + 1; }
    x
}
""",
    "test_loop_with_break": """
fn test_loop_with_break(xs: Vec<i64>) -> i64 {
    let mut i: i64 = 0;
    loop {
        if i >= xs.len() as i64 { break; }
        if xs[i as usize] < 0 { return i; }
        i = i + 1;
    }
    0 - 1
}
""",
    "test_loop_sum": """
fn test_loop_sum(xs: Vec<i64>) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    loop {
        if i >= xs.len() as i64 { break; }
        total = total + xs[i as usize];
        i = i + 1;
    }
    total
}
""",
    "test_match_classify": """
fn test_match_classify(x: i64) -> i64 {
    match x {
        0 => 0,
        1 => 10,
        2 => 20,
        _ => 100,
    }
}
""",
    "test_match_sign": """
fn test_match_sign(x: i64) -> i64 {
    match x {
        0 => 0,
        n if n > 0 => 1,
        _ => 0 - 1,
    }
}
""",
    "test_count_evens": """
fn test_count_evens(xs: Vec<i64>) -> i64 {
    let mut i: i64 = 0;
    let mut count: i64 = 0;
    while i < xs.len() as i64 {
        if xs[i as usize] % 2 == 0 { count = count + 1; }
        i = i + 1;
    }
    count
}
""",
    "test_find": """
fn test_find(xs: Vec<i64>, target: i64) -> i64 {
    let mut i: i64 = 0;
    while i < xs.len() as i64 {
        if xs[i as usize] == target { return i; }
        i = i + 1;
    }
    0 - 1
}
""",
    "test_describe": """
fn test_describe(x: i64) -> i64 {
    match x {
        0 => { let y: i64 = 100; y }
        _ => { let y: i64 = 200; y }
    }
}
""",
}


@pytest.mark.native
def test_while_count_down(native_ns):
    assert native_ns["test_while_count_down"](5) == 15


@pytest.mark.native
def test_while_never_enters(native_ns):
    assert native_ns["test_while_never_enters"](5) == 0


@pytest.mark.native
def test_loop_with_break(native_ns):
    assert native_ns["test_loop_with_break"]([1, 2, -3, 4]) == 2
    assert native_ns["test_loop_with_break"]([1, 2, 3]) == -1


@pytest.mark.native
def test_loop_sum(native_ns):
    assert native_ns["test_loop_sum"]([1, 2, 3, 4]) == 10


@pytest.mark.native
def test_match_literal_arms(native_ns):
    assert native_ns["test_match_classify"](0) == 0
    assert native_ns["test_match_classify"](1) == 10
    assert native_ns["test_match_classify"](2) == 20
    assert native_ns["test_match_classify"](99) == 100


@pytest.mark.native
def test_match_with_guard(native_ns):
    assert native_ns["test_match_sign"](5) == 1
    assert native_ns["test_match_sign"](0) == 0
    assert native_ns["test_match_sign"](-5) == -1


@pytest.mark.native
def test_while_nested_if(native_ns):
    assert native_ns["test_count_evens"]([1, 2, 3, 4, 5, 6]) == 3


@pytest.mark.native
def test_loop_with_return(native_ns):
    assert native_ns["test_find"]([10, 20, 30], 20) == 1
    assert native_ns["test_find"]([10, 20, 30], 99) == -1


@pytest.mark.native
def test_match_nested_block_body(native_ns):
    assert native_ns["test_describe"](0) == 100
    assert native_ns["test_describe"](1) == 200
