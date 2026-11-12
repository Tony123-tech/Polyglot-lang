"""N3 native and Python vector iteration tests."""

import pytest

from backends.python.translator import compile_rust
from lexer import lex
from parser import parse

SOURCES = {
    "iter_string_lengths": """
fn iter_string_lengths(values: Vec<String>) -> i64 {
    let mut total = 0;
    for value in values {
        total = total + value.len() as i64;
    }
    total
}
""",
    "iter_tuple_sums": """
fn iter_tuple_sums(pairs: Vec<(i64, i64)>) -> i64 {
    let mut total = 0;
    for (left, right) in pairs {
        total = total + left + right;
    }
    total
}
""",
    "vec_contains": """
fn vec_contains(values: Vec<i64>, target: i64) -> bool {
    values.contains(target)
}
""",
    "vec_sort": """
fn vec_sort(mut values: Vec<i64>) -> i64 {
    values.sort();
    values[0]
}
""",
    "vec_sort_by": """
fn vec_sort_by(mut values: Vec<i64>) -> i64 {
    values.sort_by(|left, right| left - right);
    values[0]
}
""",
    "vec_map": """
fn vec_map(values: Vec<i64>) -> Vec<i64> {
    values.map(|value| value + 1)
}
""",
    "vec_filter": """
fn vec_filter(values: Vec<i64>) -> Vec<i64> {
    values.filter(|value| *value > 2)
}
""",
    "vec_fold": """
fn vec_fold(values: Vec<i64>) -> i64 {
    values.fold(10, |total, value| total + value)
}
""",
    "vec_slice": """
fn vec_slice(values: Vec<i64>) -> Vec<i64> {
    &values[1..3]
}
""",
    "vec_extend": """
fn vec_extend(mut left: Vec<i64>, right: Vec<i64>) -> Vec<i64> {
    left.extend(right);
    left
}
""",
    "vec_insert": """
fn vec_insert(mut values: Vec<i64>, index: i64, value: i64) -> i64 {
    values.insert(index as usize, value);
    values[index as usize]
}
""",
    "vec_remove": """
fn vec_remove(mut values: Vec<i64>, index: i64) -> i64 {
    values.remove(index as usize)
}
""",
    "vec_retain": """
fn vec_retain(mut values: Vec<i64>) -> Vec<i64> {
    values.retain(|value| *value % 2 == 0);
    values
}
""",
    "vec_dedup": """
fn vec_dedup(mut values: Vec<i64>) -> Vec<i64> {
    values.dedup();
    values
}
""",
    "vec_windows": """
fn vec_windows(values: Vec<i64>) -> Vec<Vec<i64>> {
    values.windows(2)
}
""",
    "vec_chunks": """
fn vec_chunks(values: Vec<i64>) -> Vec<Vec<i64>> {
    values.chunks(2)
}
""",
}


@pytest.mark.native
def test_native_iter_vec_string(native_ns):
    assert native_ns["iter_string_lengths"](["oak", "pine"]) == 7


@pytest.mark.native
def test_native_iter_vec_tuple(native_ns):
    assert native_ns["iter_tuple_sums"]([(1, 2), (3, 4)]) == 10


@pytest.mark.native
def test_native_vec_contains(native_ns):
    assert native_ns["vec_contains"]([2, 4, 6], 4) is True
    assert native_ns["vec_contains"]([2, 4, 6], 5) is False


@pytest.mark.native
def test_native_vec_sort_and_sort_by(native_ns):
    assert native_ns["vec_sort"]([4, 1, 3]) == 1
    assert native_ns["vec_sort_by"]([4, 1, 3]) == 1


@pytest.mark.native
def test_native_vec_map(native_ns):
    assert native_ns["vec_map"]([1, 2, 3]) == [2, 3, 4]


@pytest.mark.native
def test_native_vec_filter(native_ns):
    assert native_ns["vec_filter"]([1, 2, 3, 4]) == [3, 4]


@pytest.mark.native
def test_native_vec_fold(native_ns):
    assert native_ns["vec_fold"]([1, 2, 3]) == 16


@pytest.mark.native
def test_native_vec_slice(native_ns):
    assert native_ns["vec_slice"]([10, 20, 30, 40]) == [20, 30]


@pytest.mark.native
def test_native_vec_extend(native_ns):
    assert native_ns["vec_extend"]([1, 2], [3, 4]) == [1, 2, 3, 4]


@pytest.mark.native
def test_native_vec_insert_remove_retain_dedup(native_ns):
    assert native_ns["vec_insert"]([1, 3], 1, 2) == 2
    assert native_ns["vec_remove"]([10, 20, 30], 1) == 20
    assert native_ns["vec_retain"]([1, 2, 3, 4]) == [2, 4]
    assert native_ns["vec_dedup"]([1, 1, 2, 2, 1]) == [1, 2, 1]


@pytest.mark.native
def test_native_vec_windows_and_chunks(native_ns):
    assert native_ns["vec_windows"]([1, 2, 3]) == [[1, 2], [2, 3]]
    assert native_ns["vec_chunks"]([1, 2, 3]) == [[1, 2], [3]]


@pytest.mark.parametrize(
    ("name", "argument", "expected"),
    [
        ("iter_string_lengths", ["oak", "pine"], 7),
        ("iter_tuple_sums", [(1, 2), (3, 4)], 10),
    ],
)
def test_python_iter_vec_parity(name, argument, expected):
    program = parse(lex(SOURCES[name]))
    namespace = compile_rust(program)

    assert namespace[name](argument) == expected


def test_python_vec_contains_parity():
    program = parse(lex(SOURCES["vec_contains"]))
    namespace = compile_rust(program)

    assert namespace["vec_contains"]([2, 4, 6], 4) is True
    assert namespace["vec_contains"]([2, 4, 6], 5) is False


@pytest.mark.parametrize("name", ["vec_sort", "vec_sort_by"])
def test_python_vec_sort_parity(name):
    program = parse(lex(SOURCES[name]))
    namespace = compile_rust(program)

    assert namespace[name]([4, 1, 3]) == 1


def test_python_vec_map_parity():
    program = parse(lex(SOURCES["vec_map"]))
    namespace = compile_rust(program)

    assert namespace["vec_map"]([1, 2, 3]) == [2, 3, 4]


def test_python_vec_filter_parity():
    program = parse(lex(SOURCES["vec_filter"]))
    namespace = compile_rust(program)

    assert namespace["vec_filter"]([1, 2, 3, 4]) == [3, 4]


def test_python_vec_fold_parity():
    program = parse(lex(SOURCES["vec_fold"]))
    namespace = compile_rust(program)

    assert namespace["vec_fold"]([1, 2, 3]) == 16


def test_python_vec_slice_parity():
    program = parse(lex(SOURCES["vec_slice"]))
    namespace = compile_rust(program)

    assert namespace["vec_slice"]([10, 20, 30, 40]) == [20, 30]


def test_python_vec_extend_parity():
    program = parse(lex(SOURCES["vec_extend"]))
    namespace = compile_rust(program)

    assert namespace["vec_extend"]([1, 2], [3, 4]) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("vec_insert", ([1, 3], 1, 2), 2),
        ("vec_remove", ([10, 20, 30], 1), 20),
        ("vec_retain", ([1, 2, 3, 4],), [2, 4]),
        ("vec_dedup", ([1, 1, 2, 2, 1],), [1, 2, 1]),
    ],
)
def test_python_vec_mutation_parity(name, args, expected):
    program = parse(lex(SOURCES[name]))
    namespace = compile_rust(program)

    assert namespace[name](*args) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("vec_windows", [[1, 2], [2, 3]]),
        ("vec_chunks", [[1, 2], [3]]),
    ],
)
def test_python_vec_windows_and_chunks_parity(name, expected):
    program = parse(lex(SOURCES[name]))
    namespace = compile_rust(program)

    assert namespace[name]([1, 2, 3]) == expected

def test_python_vec_slice_start_greater_than_end():
    program = parse(lex("""
fn vec_slice(values: Vec<i64>) -> Vec<i64> {
    &values[3..1]
}
"""))
    namespace = compile_rust(program)
    with pytest.raises(ValueError):
        namespace["vec_slice"]([10, 20, 30, 40])


def test_python_vec_sort_empty_panics():
    program = parse(lex(SOURCES["vec_sort"]))
    namespace = compile_rust(program)
    with pytest.raises(IndexError):
        namespace["vec_sort"]([])
