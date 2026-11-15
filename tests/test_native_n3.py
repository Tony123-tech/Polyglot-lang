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
    "vec_binary_search": """
fn vec_binary_search(values: Vec<i64>, target: i64) -> i64 {
    values.binary_search(&target).unwrap_or(999) as i64
}
""",
    "vec_split_at": """
fn vec_split_at(values: Vec<i64>, index: i64) -> (Vec<i64>, Vec<i64>) {
    values.split_at(index as usize)
}
""",
    "vec_is_sorted_by": """
fn vec_is_sorted_by(values: Vec<i64>) -> bool {
    values.is_sorted_by(|left, right| left <= right)
}
""",
    "vec_starts_ends_with": """
fn vec_starts_with(values: Vec<i64>, prefix: Vec<i64>) -> bool {
    values.starts_with(prefix)
}
fn vec_ends_with(values: Vec<i64>, suffix: Vec<i64>) -> bool {
    values.ends_with(suffix)
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
    "vec_chunks_exact": """
fn vec_chunks_exact(values: Vec<i64>, size: i64) -> Vec<Vec<i64>> {
    values.chunks_exact(size as usize)
}
""",
    "vec_rchunks": """
fn vec_rchunks(values: Vec<i64>, size: i64) -> Vec<Vec<i64>> {
    values.rchunks(size as usize)
}
""",
    "vec_rchunks_exact": """
fn vec_rchunks_exact(values: Vec<i64>, size: i64) -> Vec<Vec<i64>> {
    values.rchunks_exact(size as usize)
}
""",
    "vec_concat": """
fn vec_concat_strings(values: Vec<String>) -> String {
    values.concat()
}
fn vec_concat_ints(values: Vec<Vec<i64>>) -> Vec<i64> {
    values.concat()
}
""",
    "vec_swap_remove": """
fn vec_swap_remove(mut values: Vec<i64>, index: i64) -> Vec<i64> {
    values.swap_remove(index as usize);
    values
}
""",
    "vec_dedup_by_key": """
fn vec_dedup_by_key(mut values: Vec<i64>) -> Vec<i64> {
    values.dedup_by_key(|value| *value % 10);
    values
}
""",
    "vec_sort_unstable_by_key": """
fn vec_sort_unstable_by_key(mut values: Vec<i64>) -> Vec<i64> {
    values.sort_unstable_by_key(|value| -*value);
    values
}
""",
    "vec_is_sorted_by_key": """
fn vec_is_sorted_by_key(values: Vec<i64>) -> bool {
    values.is_sorted_by_key(|value| *value)
}
""",
    "vec_position": """
fn vec_position(values: Vec<i64>, target: i64) -> Option<i64> {
    values.position(|value| value == target)
}
""",
    "vec_any": """
fn vec_any(values: Vec<i64>, target: i64) -> bool {
    values.any(|value| value == target)
}
""",
    "vec_all": """
fn vec_all(values: Vec<i64>) -> bool {
    values.all(|value| value > 0)
}
""",
    "vec_min": """
fn vec_min(values: Vec<i64>) -> Option<i64> {
    values.min()
}
""",
    "vec_max": """
fn vec_max(values: Vec<i64>) -> Option<i64> {
    values.max()
}
""",
    "vec_sum_product": """
fn vec_sum(values: Vec<i64>) -> i64 {
    values.sum()
}
fn vec_product(values: Vec<i64>) -> i64 {
    values.product()
}
""",
    "vec_rev": """
fn vec_rev(values: Vec<i64>) -> Vec<i64> {
    values.rev()
}
""",
    "vec_skip": """
fn vec_skip(values: Vec<i64>) -> Vec<i64> {
    values.skip(2)
}
""",
    "vec_enumerate": """
fn vec_enumerate(values: Vec<i64>) -> Vec<(i64, i64)> {
    values.enumerate()
}
""",
    "vec_zip": """
fn vec_zip(left: Vec<i64>, right: Vec<i64>) -> Vec<(i64, i64)> {
    left.zip(right)
}
""",
    "vec_flatten": """
fn vec_flatten(values: Vec<Vec<i64>>) -> Vec<i64> {
    values.flatten()
}
""",
    "vec_chain": """
fn vec_chain(left: Vec<i64>, right: Vec<i64>) -> Vec<i64> {
    left.chain(right)
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


@pytest.mark.native
def test_native_vec_chunks_exact(native_ns):
    assert native_ns["vec_chunks_exact"]([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4]]
    assert native_ns["vec_chunks_exact"]([1, 2, 3], 2) == [[1, 2]]
    assert native_ns["vec_chunks_exact"]([1], 2) == []


def test_python_vec_chunks_exact_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_chunks_exact"])))

    assert namespace["vec_chunks_exact"]([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4]]
    assert namespace["vec_chunks_exact"]([1, 2, 3], 2) == [[1, 2]]
    assert namespace["vec_chunks_exact"]([1], 2) == []


@pytest.mark.native
def test_native_vec_rchunks(native_ns):
    assert native_ns["vec_rchunks"]([1, 2, 3, 4, 5], 2) == [[4, 5], [2, 3], [1]]
    assert native_ns["vec_rchunks"]([1, 2, 3], 2) == [[2, 3], [1]]
    assert native_ns["vec_rchunks"]([], 2) == []


def test_python_vec_rchunks_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_rchunks"])))

    assert namespace["vec_rchunks"]([1, 2, 3, 4, 5], 2) == [[4, 5], [2, 3], [1]]
    assert namespace["vec_rchunks"]([1, 2, 3], 2) == [[2, 3], [1]]
    assert namespace["vec_rchunks"]([], 2) == []


@pytest.mark.native
def test_native_vec_rchunks_exact(native_ns):
    assert native_ns["vec_rchunks_exact"]([1, 2, 3, 4, 5], 2) == [[4, 5], [2, 3]]
    assert native_ns["vec_rchunks_exact"]([1, 2, 3], 2) == [[2, 3]]
    assert native_ns["vec_rchunks_exact"]([1], 2) == []


def test_python_vec_rchunks_exact_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_rchunks_exact"])))

    assert namespace["vec_rchunks_exact"]([1, 2, 3, 4, 5], 2) == [[4, 5], [2, 3]]
    assert namespace["vec_rchunks_exact"]([1, 2, 3], 2) == [[2, 3]]
    assert namespace["vec_rchunks_exact"]([1], 2) == []


@pytest.mark.native
def test_native_vec_concat(native_ns):
    assert native_ns["vec_concat_strings"](["ab", "cd", "e"]) == "abcde"
    assert native_ns["vec_concat_ints"]([[1, 2], [3], []]) == [1, 2, 3]


def test_python_vec_concat_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_concat"])))

    assert namespace["vec_concat_strings"](["ab", "cd", "e"]) == "abcde"
    assert namespace["vec_concat_ints"]([[1, 2], [3], []]) == [1, 2, 3]


@pytest.mark.native
def test_native_vec_swap_remove(native_ns):
    assert native_ns["vec_swap_remove"]([1, 2, 3, 4], 1) == [1, 4, 3]
    assert native_ns["vec_swap_remove"]([1, 2, 3], 2) == [1, 2]
    assert native_ns["vec_swap_remove"]([1], 0) == []


def test_python_vec_swap_remove_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_swap_remove"])))

    assert namespace["vec_swap_remove"]([1, 2, 3, 4], 1) == [1, 4, 3]
    assert namespace["vec_swap_remove"]([1, 2, 3], 2) == [1, 2]
    assert namespace["vec_swap_remove"]([1], 0) == []


@pytest.mark.native
def test_native_vec_dedup_by_key(native_ns):
    assert native_ns["vec_dedup_by_key"]([1, 11, 2, 22, 3]) == [1, 2, 3]
    assert native_ns["vec_dedup_by_key"]([]) == []


def test_python_vec_dedup_by_key_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_dedup_by_key"])))

    assert namespace["vec_dedup_by_key"]([1, 11, 2, 22, 3]) == [1, 2, 3]
    assert namespace["vec_dedup_by_key"]([]) == []


@pytest.mark.native
def test_native_vec_sort_unstable_by_key(native_ns):
    assert native_ns["vec_sort_unstable_by_key"]([1, 3, 2, 4]) == [4, 3, 2, 1]
    assert native_ns["vec_sort_unstable_by_key"]([]) == []


def test_python_vec_sort_unstable_by_key_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_sort_unstable_by_key"])))

    assert namespace["vec_sort_unstable_by_key"]([1, 3, 2, 4]) == [4, 3, 2, 1]
    assert namespace["vec_sort_unstable_by_key"]([]) == []


@pytest.mark.native
def test_native_vec_is_sorted_by_key(native_ns):
    assert native_ns["vec_is_sorted_by_key"]([1, 2, 2, 3]) is True
    assert native_ns["vec_is_sorted_by_key"]([1, 3, 2]) is False
    assert native_ns["vec_is_sorted_by_key"]([]) is True


def test_python_vec_is_sorted_by_key_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_is_sorted_by_key"])))

    assert namespace["vec_is_sorted_by_key"]([1, 2, 2, 3]) is True
    assert namespace["vec_is_sorted_by_key"]([1, 3, 2]) is False
    assert namespace["vec_is_sorted_by_key"]([]) is True


@pytest.mark.native
def test_native_vec_position(native_ns):
    assert native_ns["vec_position"]([4, 2, 7], 7) == 2
    assert native_ns["vec_position"]([4, 2, 7], 9) is None


def test_python_vec_position_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_position"])))

    assert namespace["vec_position"]([4, 2, 7], 7) == namespace["_Option"].some(2)
    assert namespace["vec_position"]([4, 2, 7], 9) == namespace["_Option"].none()


@pytest.mark.native
def test_native_vec_any(native_ns):
    assert native_ns["vec_any"]([1, 2, 3], 2) is True
    assert native_ns["vec_any"]([1, 2, 3], 9) is False
    assert native_ns["vec_any"]([], 1) is False


def test_python_vec_any_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_any"])))

    assert namespace["vec_any"]([1, 2, 3], 2) is True
    assert namespace["vec_any"]([1, 2, 3], 9) is False
    assert namespace["vec_any"]([], 1) is False


@pytest.mark.native
def test_native_vec_all(native_ns):
    assert native_ns["vec_all"]([1, 2, 3]) is True
    assert native_ns["vec_all"]([1, 0, 3]) is False
    assert native_ns["vec_all"]([]) is True


def test_python_vec_all_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_all"])))

    assert namespace["vec_all"]([1, 2, 3]) is True
    assert namespace["vec_all"]([1, 0, 3]) is False
    assert namespace["vec_all"]([]) is True


@pytest.mark.native
def test_native_vec_min(native_ns):
    assert native_ns["vec_min"]([4, 2, 7]) == 2
    assert native_ns["vec_min"]([]) is None


def test_python_vec_min_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_min"])))

    assert namespace["vec_min"]([4, 2, 7]) == namespace["_Option"].some(2)
    assert namespace["vec_min"]([]) == namespace["_Option"].none()


@pytest.mark.native
def test_native_vec_max(native_ns):
    assert native_ns["vec_max"]([4, 2, 7]) == 7
    assert native_ns["vec_max"]([]) is None


def test_python_vec_max_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_max"])))

    assert namespace["vec_max"]([4, 2, 7]) == namespace["_Option"].some(7)
    assert namespace["vec_max"]([]) == namespace["_Option"].none()


@pytest.mark.native
def test_native_vec_sum(native_ns):
    assert native_ns["vec_sum"]([1, 2, 3, 4]) == 10
    assert native_ns["vec_sum"]([]) == 0


def test_python_vec_sum_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_sum_product"])))

    assert namespace["vec_sum"]([1, 2, 3, 4]) == 10
    assert namespace["vec_sum"]([]) == 0


@pytest.mark.native
def test_native_vec_product(native_ns):
    assert native_ns["vec_product"]([1, 2, 3, 4]) == 24
    assert native_ns["vec_product"]([]) == 1


def test_python_vec_product_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_sum_product"])))

    assert namespace["vec_product"]([1, 2, 3, 4]) == 24
    assert namespace["vec_product"]([]) == 1


@pytest.mark.native
def test_native_vec_rev(native_ns):
    assert native_ns["vec_rev"]([1, 2, 3]) == [3, 2, 1]
    assert native_ns["vec_rev"]([]) == []


def test_python_vec_rev_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_rev"])))

    assert namespace["vec_rev"]([1, 2, 3]) == [3, 2, 1]
    assert namespace["vec_rev"]([]) == []


@pytest.mark.native
def test_native_vec_skip(native_ns):
    assert native_ns["vec_skip"]([1, 2, 3, 4]) == [3, 4]
    assert native_ns["vec_skip"]([1, 2]) == []


def test_python_vec_skip_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_skip"])))

    assert namespace["vec_skip"]([1, 2, 3, 4]) == [3, 4]
    assert namespace["vec_skip"]([1, 2]) == []


@pytest.mark.native
def test_native_vec_enumerate(native_ns):
    assert native_ns["vec_enumerate"]([10, 20]) == [(0, 10), (1, 20)]
    assert native_ns["vec_enumerate"]([]) == []


def test_python_vec_enumerate_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_enumerate"])))

    assert namespace["vec_enumerate"]([10, 20]) == [(0, 10), (1, 20)]
    assert namespace["vec_enumerate"]([]) == []


@pytest.mark.native
def test_native_vec_zip(native_ns):
    assert native_ns["vec_zip"]([1, 2], [3, 4, 5]) == [(1, 3), (2, 4)]
    assert native_ns["vec_zip"]([], [1]) == []


def test_python_vec_zip_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_zip"])))

    assert namespace["vec_zip"]([1, 2], [3, 4, 5]) == [(1, 3), (2, 4)]
    assert namespace["vec_zip"]([], [1]) == []


@pytest.mark.native
def test_native_vec_flatten(native_ns):
    assert native_ns["vec_flatten"]([[1, 2], [], [3]]) == [1, 2, 3]
    assert native_ns["vec_flatten"]([]) == []


def test_python_vec_flatten_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_flatten"])))

    assert namespace["vec_flatten"]([[1, 2], [], [3]]) == [1, 2, 3]
    assert namespace["vec_flatten"]([]) == []


@pytest.mark.native
def test_native_vec_chain(native_ns):
    assert native_ns["vec_chain"]([1, 2], [3]) == [1, 2, 3]
    assert native_ns["vec_chain"]([], []) == []


def test_python_vec_chain_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_chain"])))

    assert namespace["vec_chain"]([1, 2], [3]) == [1, 2, 3]
    assert namespace["vec_chain"]([], []) == []


@pytest.mark.native
def test_native_vec_binary_search(native_ns):
    assert native_ns["vec_binary_search"]([1, 3, 5, 7], 5) == 2
    assert native_ns["vec_binary_search"]([1, 3, 5, 7], 4) == 999
    assert native_ns["vec_binary_search"]([], 4) == 999


@pytest.mark.native
def test_native_vec_split_at(native_ns):
    assert native_ns["vec_split_at"]([1, 2, 3, 4], 2) == ([1, 2], [3, 4])
    assert native_ns["vec_split_at"]([1, 2], 0) == ([], [1, 2])
    assert native_ns["vec_split_at"]([1, 2], 2) == ([1, 2], [])


@pytest.mark.native
def test_native_vec_is_sorted_by(native_ns):
    assert native_ns["vec_is_sorted_by"]([1, 2, 2, 3]) is True
    assert native_ns["vec_is_sorted_by"]([1, 3, 2]) is False
    assert native_ns["vec_is_sorted_by"]([]) is True


@pytest.mark.native
def test_native_vec_starts_ends_with(native_ns):
    assert native_ns["vec_starts_with"]([1, 2, 3], [1, 2]) is True
    assert native_ns["vec_starts_with"]([1, 2, 3], [2, 3]) is False
    assert native_ns["vec_ends_with"]([1, 2, 3], [2, 3]) is True
    assert native_ns["vec_ends_with"]([1, 2, 3], [1, 2]) is False
    assert native_ns["vec_ends_with"]([1], [2, 3]) is False


def test_python_vec_starts_ends_with_parity():
    namespace = compile_rust(parse(lex(SOURCES["vec_starts_ends_with"])))

    assert namespace["vec_starts_with"]([1, 2, 3], [1, 2]) is True
    assert namespace["vec_starts_with"]([1, 2, 3], [2, 3]) is False
    assert namespace["vec_ends_with"]([1, 2, 3], [2, 3]) is True
    assert namespace["vec_ends_with"]([1, 2, 3], [1, 2]) is False
    assert namespace["vec_ends_with"]([1], [2, 3]) is False


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


def test_python_vec_binary_search_parity():
    program = parse(lex(SOURCES["vec_binary_search"]))
    namespace = compile_rust(program)

    assert namespace["vec_binary_search"]([1, 3, 5, 7], 5) == 2
    assert namespace["vec_binary_search"]([1, 3, 5, 7], 4) == 999
    assert namespace["vec_binary_search"]([], 4) == 999


def test_python_vec_split_at_parity():
    program = parse(lex(SOURCES["vec_split_at"]))
    namespace = compile_rust(program)

    assert namespace["vec_split_at"]([1, 2, 3, 4], 2) == ([1, 2], [3, 4])
    assert namespace["vec_split_at"]([1, 2], 0) == ([], [1, 2])
    assert namespace["vec_split_at"]([1, 2], 2) == ([1, 2], [])


def test_python_vec_is_sorted_by_parity():
    program = parse(lex(SOURCES["vec_is_sorted_by"]))
    namespace = compile_rust(program)

    assert namespace["vec_is_sorted_by"]([1, 2, 2, 3]) is True
    assert namespace["vec_is_sorted_by"]([1, 3, 2]) is False
    assert namespace["vec_is_sorted_by"]([]) is True


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
    program = parse(
        lex("""
fn vec_slice(values: Vec<i64>) -> Vec<i64> {
    &values[3..1]
}
""")
    )
    namespace = compile_rust(program)
    with pytest.raises(ValueError):
        namespace["vec_slice"]([10, 20, 30, 40])


def test_python_vec_sort_empty_panics():
    program = parse(lex(SOURCES["vec_sort"]))
    namespace = compile_rust(program)
    with pytest.raises(IndexError):
        namespace["vec_sort"]([])
