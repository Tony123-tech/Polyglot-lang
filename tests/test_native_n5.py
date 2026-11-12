# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N5: Option and Result."""

from __future__ import annotations

import pytest

from ast_nodes import OptionNone, OptionSome, ResultErr, ResultOk
from lexer import lex
from parser import parse

SOURCES = {
    "option_some_construction": """
fn n5_option_some_construction() -> Option<i64> {
    Some(42)
}
""",
    "option_none_construction": """
fn n5_option_none_construction() -> Option<i64> {
    None
}
""",
    "result_ok_construction": """
fn n5_result_ok_construction() -> Result<i64, String> {
    Ok(7)
}
""",
    "result_err_construction": """
fn n5_result_err_construction() -> Result<i64, String> {
    Err("bad")
}
""",
    "option_unwrap": """
fn n5_option_unwrap_get() -> i64 {
    Some(5).unwrap()
}
fn n5_option_unwrap_fail() -> i64 {
    None.unwrap()
}
""",
    "option_unwrap_or": """
fn n5_option_unwrap_or_some() -> i64 { Some(5).unwrap_or(0) }
fn n5_option_unwrap_or_none() -> i64 { None.unwrap_or(0) }
""",
    "option_unwrap_or_else": """
fn n5_option_unwrap_or_else_some() -> i64 { Some(5).unwrap_or_else(|| 0) }
fn n5_option_unwrap_or_else_none() -> i64 { None.unwrap_or_else(|| 99) }
""",
    "option_is_some": """
fn n5_option_is_some_true() -> bool { Some(1).is_some() }
fn n5_option_is_none_false() -> bool { Some(1).is_none() }
""",
    "option_is_none_with_annotation": """
fn n5_option_is_none_true() -> bool {
    let x: Option<i64> = None;
    x.is_none()
}
fn n5_option_is_some_false() -> bool {
    let x: Option<i64> = None;
    x.is_some()
}
""",
    "option_unwrap_or_else_called_on_none": """
fn n5_option_unwrap_or_else_on_none() -> i64 { None.unwrap_or_else(|| 99) }
""",
    "nested_option_construction": """
fn n5_nested_option() -> Option<Option<i64>> {
    Some(Some(1))
}
""",
    "result_string_ok": """
fn n5_result_string_ok() -> Result<String, String> {
    Ok("hello".to_string())
}
""",
    "result_string_err": """
fn n5_result_string_err() -> Result<String, String> {
    Err("bad".to_string())
}
""",
    "async_add": """
async fn n5_async_add(a: i64, b: i64) -> i64 { a + b }
""",
    "async_identity": """
async fn n5_async_identity(x: i64) -> i64 { x }
""",
    "vec_is_empty": """
fn n5_vec_is_empty(values: Vec<i64>) -> bool {
    values.is_empty()
}
""",
    "vec_clear": """
fn n5_vec_clear(mut values: Vec<i64>) -> Vec<i64> {
    values.clear();
    values
}
""",
    "vec_reverse": """
fn n5_vec_reverse(mut values: Vec<i64>) -> Vec<i64> {
    values.reverse();
    values
}
""",
    "string_repeat": """
fn n5_string_repeat(value: String, count: i64) -> String {
    value.repeat(count)
}
""",
    "vec_first": """
fn n5_vec_first(values: Vec<i64>) -> i64 {
    values.first().unwrap_or(-1)
}
""",
    "vec_last": """
fn n5_vec_last(values: Vec<i64>) -> i64 {
    values.last().unwrap_or(-1)
}
""",
    "vec_get": """
fn n5_vec_get(values: Vec<i64>, index: i64) -> i64 {
    values.get(index).unwrap_or(-1)
}
""",
    "string_contains_find": """
fn n5_string_contains(value: String, needle: String) -> bool {
    value.contains(needle)
}
fn n5_string_find(value: String, needle: String) -> i64 {
    value.find(needle).unwrap_or(0) as i64
}
""",
    "vec_truncate_string_predicates": """
fn n5_vec_truncate(values: Vec<i64>, length: i64) -> i64 {
    values.truncate(length);
    values.len()
}
fn n5_string_is_empty(value: String) -> bool {
    value.is_empty()
}
fn n5_string_starts_with(value: String, prefix: String) -> bool {
    value.starts_with(prefix)
}
fn n5_string_ends_with(value: String, suffix: String) -> bool {
    value.ends_with(suffix)
}
""",
    "string_trim_ascii": """
fn n5_string_trim_start(value: String) -> String {
    value.trim_start()
}
fn n5_string_trim_end(value: String) -> String {
    value.trim_end()
}
fn n5_string_is_ascii(value: String) -> bool {
    value.is_ascii()
}
""",
    "string_lines_whitespace_and_vec_sort_unstable": """
fn n5_string_lines(value: String) -> Vec<String> {
    value.lines()
}
fn n5_string_split_whitespace(value: String) -> Vec<String> {
    value.split_whitespace()
}
fn n5_vec_sort_unstable(values: Vec<i64>) -> Vec<i64> {
    values.sort_unstable();
    values
}
""",
    "string_bytes_boundaries_ascii_case": """
fn n5_string_as_bytes(value: String) -> Vec<u8> {
    value.as_bytes()
}
fn n5_string_is_char_boundary(value: String, index: i64) -> bool {
    value.is_char_boundary(index)
}
fn n5_string_to_ascii_lowercase(value: String) -> String {
    value.to_ascii_lowercase()
}
fn n5_string_to_ascii_uppercase(value: String) -> String {
    value.to_ascii_uppercase()
}
""",
    "vec_rotate_fill_swap": """
fn n5_vec_rotate_left(mut values: Vec<i64>, amount: i64) -> Vec<i64> {
    values.rotate_left(amount);
    values
}
fn n5_vec_rotate_right(mut values: Vec<i64>, amount: i64) -> Vec<i64> {
    values.rotate_right(amount);
    values
}
fn n5_vec_fill(mut values: Vec<i64>, value: i64) -> Vec<i64> {
    values.fill(value);
    values
}
fn n5_vec_swap(mut values: Vec<i64>, left: i64, right: i64) -> Vec<i64> {
    values.swap(left, right);
    values
}
""",
    "vec_extend_from_slice_resize_split_off": """
fn n5_vec_extend_from_slice(mut values: Vec<i64>, other: Vec<i64>) -> Vec<i64> {
    values.extend_from_slice(other);
    values
}
fn n5_vec_resize(mut values: Vec<i64>, length: i64, value: i64) -> Vec<i64> {
    values.resize(length, value);
    values
}
fn n5_vec_split_off(mut values: Vec<i64>, index: i64) -> i64 {
    let tail = values.split_off(index);
    values.len() * 10 + tail.len()
}
""",
    "string_rfind_strip_split_once": """
fn n5_string_rfind(value: String, needle: String) -> i64 {
    value.rfind(needle).unwrap_or(0) as i64
}
fn n5_string_strip_prefix(value: String, prefix: String) -> String {
    value.strip_prefix(prefix).unwrap_or("missing".to_string())
}
fn n5_string_strip_suffix(value: String, suffix: String) -> String {
    value.strip_suffix(suffix).unwrap_or("missing".to_string())
}
fn n5_string_split_once(value: String, separator: String) -> Option<(String, String)> {
    value.split_once(separator)
}
fn n5_string_rsplit_once(value: String, separator: String) -> Option<(String, String)> {
    value.rsplit_once(separator)
}
""",
    "vec_partition_sort_by_key_is_sorted": """
fn n5_vec_partition(values: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
    values.into_iter().partition(|value| value % 2 == 0)
}
fn n5_vec_sort_by_key(mut values: Vec<i64>) -> Vec<i64> {
    values.sort_by_key(|value| -value);
    values
}
fn n5_vec_is_sorted(values: Vec<i64>) -> bool {
    values.is_sorted()
}
""",
}


def _parse_expr(src: str):
    program = parse(lex(f"fn _f() -> i64 {{ {src} }}"))
    return program.items[0].body.tail


def _run_python(src: str) -> dict:
    from backends.python.translator import compile_rust

    return compile_rust(parse(lex(src)))


def _run_python_sandboxed(src: str) -> dict:
    from backends.python.translator import PythonBackend

    return PythonBackend(sandbox=True).compile(parse(lex(src)))


# ---------------------------------------------------------------------------
# AST parsing
# ---------------------------------------------------------------------------

def test_parse_option_some():
    expr = _parse_expr("Some(1)")
    assert isinstance(expr, OptionSome)
    assert expr.value.value == 1


def test_parse_option_none():
    expr = _parse_expr("None")
    assert isinstance(expr, OptionNone)


def test_parse_result_ok():
    expr = _parse_expr("Ok(1)")
    assert isinstance(expr, ResultOk)


def test_parse_result_err():
    expr = _parse_expr('Err("bad")')
    assert isinstance(expr, ResultErr)


# ---------------------------------------------------------------------------
# Construction (native)
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_option_some_construction(native_ns):
    result = native_ns["n5_option_some_construction"]()
    assert result is not None
    assert result == 42


@pytest.mark.native
def test_native_option_none_construction(native_ns):
    assert native_ns["n5_option_none_construction"]() is None


@pytest.mark.native
def test_native_vec_clear_empty_reverse_and_string_repeat(native_ns):
    assert native_ns["n5_vec_is_empty"]([]) is True
    assert native_ns["n5_vec_is_empty"]([1]) is False
    assert native_ns["n5_vec_clear"]([1, 2, 3]) == []
    assert native_ns["n5_vec_reverse"]([1, 2, 3]) == [3, 2, 1]
    assert native_ns["n5_string_repeat"]("ab", 3) == "ababab"


@pytest.mark.native
def test_native_vec_first_last_get_and_string_search(native_ns):
    assert native_ns["n5_vec_first"]([8, 9]) == 8
    assert native_ns["n5_vec_first"]([]) == -1
    assert native_ns["n5_vec_last"]([8, 9]) == 9
    assert native_ns["n5_vec_last"]([]) == -1
    assert native_ns["n5_vec_get"]([8, 9], 1) == 9
    assert native_ns["n5_vec_get"]([8, 9], -1) == -1
    assert native_ns["n5_string_contains"]("café", "fé") is True
    assert native_ns["n5_string_contains"]("café", "tea") is False
    assert native_ns["n5_string_find"]("café", "é") == 3
    assert native_ns["n5_string_find"]("café", "tea") == 0
    assert native_ns["n5_string_find"]("café x", "x") == 6
    assert native_ns["n5_string_find"]("café", "") == 0


@pytest.mark.native
def test_native_vec_truncate_and_string_predicates(native_ns):
    assert native_ns["n5_vec_truncate"]([1, 2, 3], 2) == 2
    assert native_ns["n5_vec_truncate"]([1, 2], -1) == 2
    assert native_ns["n5_string_is_empty"]("") is True
    assert native_ns["n5_string_is_empty"]("polyglot") is False
    assert native_ns["n5_string_starts_with"]("polyglot", "poly") is True
    assert native_ns["n5_string_starts_with"]("polyglot", "glot") is False
    assert native_ns["n5_string_ends_with"]("polyglot", "glot") is True
    assert native_ns["n5_string_ends_with"]("polyglot", "poly") is False


@pytest.mark.native
def test_native_string_trim_start_end_and_is_ascii(native_ns):
    assert native_ns["n5_string_trim_start"]("  polyglot \t") == "polyglot \t"
    assert native_ns["n5_string_trim_end"]("  polyglot \t") == "  polyglot"
    assert native_ns["n5_string_is_ascii"]("polyglot") is True
    assert native_ns["n5_string_is_ascii"]("café") is False


@pytest.mark.native
def test_native_string_lines_whitespace_and_vec_sort_unstable(native_ns):
    assert native_ns["n5_string_lines"]("one\r\ntwo\n") == ["one", "two"]
    assert native_ns["n5_string_lines"]("") == []
    assert native_ns["n5_string_split_whitespace"](" one\t two\nthree ") == [
        "one", "two", "three",
    ]
    assert native_ns["n5_vec_sort_unstable"]([3, 1, 2, 1]) == [1, 1, 2, 3]


@pytest.mark.native
def test_native_string_bytes_boundaries_and_ascii_case(native_ns):
    assert native_ns["n5_string_as_bytes"]("café") == b"caf\xc3\xa9"
    assert native_ns["n5_string_is_char_boundary"]("café", 3) is True
    assert native_ns["n5_string_is_char_boundary"]("café", 4) is False
    assert native_ns["n5_string_is_char_boundary"]("café", -1) is False
    assert native_ns["n5_string_to_ascii_lowercase"]("CAFÉ") == "cafÉ"
    assert native_ns["n5_string_to_ascii_uppercase"]("café") == "CAFé"


@pytest.mark.native
def test_native_vec_rotate_fill_and_swap(native_ns):
    assert native_ns["n5_vec_rotate_left"]([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert native_ns["n5_vec_rotate_left"]([1, 2, 3], 3) == [1, 2, 3]
    assert native_ns["n5_vec_rotate_right"]([1, 2, 3, 4], 2) == [3, 4, 1, 2]
    assert native_ns["n5_vec_fill"]([1, 2, 3], 7) == [7, 7, 7]
    assert native_ns["n5_vec_swap"]([1, 2, 3], 0, 2) == [3, 2, 1]


@pytest.mark.native
def test_native_vec_extend_from_slice_resize_and_split_off(native_ns):
    assert native_ns["n5_vec_extend_from_slice"]([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert native_ns["n5_vec_resize"]([1, 2, 3], 2, 9) == [1, 2]
    assert native_ns["n5_vec_resize"]([1], 3, 9) == [1, 9, 9]
    assert native_ns["n5_vec_split_off"]([1, 2, 3, 4], 2) == 22
    assert native_ns["n5_vec_split_off"]([1, 2], 0) == 2
    assert native_ns["n5_vec_split_off"]([1, 2], 2) == 20


@pytest.mark.native
def test_native_string_rfind_strip_and_split_once(native_ns):
    assert native_ns["n5_string_rfind"]("caféé", "é") == 5
    assert native_ns["n5_string_rfind"]("café", "tea") == 0
    assert native_ns["n5_string_rfind"]("café x", "x") == 6
    assert native_ns["n5_string_strip_prefix"]("café", "ca") == "fé"
    assert native_ns["n5_string_strip_prefix"]("café", "tea") == "missing"
    assert native_ns["n5_string_strip_prefix"]("abc", "") == "abc"
    assert native_ns["n5_string_strip_suffix"]("café", "fé") == "ca"
    assert native_ns["n5_string_strip_suffix"]("café", "tea") == "missing"
    assert native_ns["n5_string_strip_suffix"]("abc", "") == "abc"
    assert native_ns["n5_string_split_once"]("a:b:c", ":") == ("a", "b:c")
    assert native_ns["n5_string_rsplit_once"]("a:b:c", ":") == ("a:b", "c")
    assert native_ns["n5_string_split_once"]("abc", ":") is None
    assert native_ns["n5_string_split_once"]("abc", "") == ("", "abc")
    assert native_ns["n5_string_rsplit_once"]("abc", "") == ("abc", "")


@pytest.mark.native
def test_native_vec_partition_sort_by_key_and_is_sorted(native_ns):
    assert native_ns["n5_vec_partition"]([1, 2, 3, 4]) == ([2, 4], [1, 3])
    assert native_ns["n5_vec_sort_by_key"]([1, 3, 2, 4]) == [4, 3, 2, 1]
    assert native_ns["n5_vec_is_sorted"]([1, 1, 2, 3]) is True
    assert native_ns["n5_vec_is_sorted"]([1, 3, 2]) is False


@pytest.mark.native
def test_native_result_ok_construction(native_ns):
    assert native_ns["n5_result_ok_construction"]() == 7


@pytest.mark.native
def test_native_result_err_construction(native_ns):
    with pytest.raises(Exception, match="bad"):
        native_ns["n5_result_err_construction"]()


# ---------------------------------------------------------------------------
# Construction (Python)
# ---------------------------------------------------------------------------

def test_python_option_result_construction():
    ns = _run_python("""
    fn some() -> Option<i64> { Some(1) }
    fn none() -> Option<i64> { None }
    fn ok() -> Result<i64, String> { Ok(2) }
    fn err() -> Result<i64, String> { Err("x") }
    """)
    assert ns["some"]() == ns["_Option"].some(1)
    assert ns["none"]() == ns["_Option"].none()
    assert ns["ok"]() == ns["_Result"].from_ok(2)
    assert ns["err"]() == ns["_Result"].from_err("x")


def test_python_vec_clear_empty_reverse_and_string_repeat():
    ns = _run_python("""
    fn n5_vec_is_empty(values: Vec<i64>) -> bool {
        values.is_empty()
    }
    fn n5_vec_clear(mut values: Vec<i64>) -> Vec<i64> {
        values.clear();
        values
    }
    fn n5_vec_reverse(mut values: Vec<i64>) -> Vec<i64> {
        values.reverse();
        values
    }
    fn n5_string_repeat(value: String, count: i64) -> String {
        value.repeat(count)
    }
    """)
    assert ns["n5_vec_is_empty"]([]) is True
    assert ns["n5_vec_is_empty"]([1]) is False
    assert ns["n5_vec_clear"]([1, 2, 3]) == []
    assert ns["n5_vec_reverse"]([1, 2, 3]) == [3, 2, 1]
    assert ns["n5_string_repeat"]("ab", 3) == "ababab"


def test_python_vec_first_last_get_and_string_search():
    ns = _run_python("""
    fn n5_vec_first(values: Vec<i64>) -> i64 {
        values.first().unwrap_or(-1)
    }
    fn n5_vec_last(values: Vec<i64>) -> i64 {
        values.last().unwrap_or(-1)
    }
    fn n5_vec_get(values: Vec<i64>, index: i64) -> i64 {
        values.get(index).unwrap_or(-1)
    }
    fn n5_string_contains(value: String, needle: String) -> bool {
        value.contains(needle)
    }
    fn n5_string_find(value: String, needle: String) -> i64 {
        value.find(needle).unwrap_or(0) as i64
    }
    """)
    assert ns["n5_vec_first"]([8, 9]) == 8
    assert ns["n5_vec_first"]([]) == -1
    assert ns["n5_vec_last"]([8, 9]) == 9
    assert ns["n5_vec_last"]([]) == -1
    assert ns["n5_vec_get"]([8, 9], 1) == 9
    assert ns["n5_vec_get"]([8, 9], -1) == -1
    assert ns["n5_string_contains"]("café", "fé") is True
    assert ns["n5_string_contains"]("café", "tea") is False
    assert ns["n5_string_find"]("café", "é") == 3
    assert ns["n5_string_find"]("café", "tea") == 0
    assert ns["n5_string_find"]("café x", "x") == 6
    assert ns["n5_string_find"]("café", "") == 0


def test_python_vec_truncate_and_string_predicates():
    ns = _run_python("""
    fn n5_vec_truncate(values: Vec<i64>, length: i64) -> i64 {
        values.truncate(length);
        values.len()
    }
    fn n5_string_is_empty(value: String) -> bool {
        value.is_empty()
    }
    fn n5_string_starts_with(value: String, prefix: String) -> bool {
        value.starts_with(prefix)
    }
    fn n5_string_ends_with(value: String, suffix: String) -> bool {
        value.ends_with(suffix)
    }
    """)
    assert ns["n5_vec_truncate"]([1, 2, 3], 2) == 2
    assert ns["n5_vec_truncate"]([1, 2], -1) == 2
    assert ns["n5_string_is_empty"]("") is True
    assert ns["n5_string_is_empty"]("polyglot") is False
    assert ns["n5_string_starts_with"]("polyglot", "poly") is True
    assert ns["n5_string_starts_with"]("polyglot", "glot") is False
    assert ns["n5_string_ends_with"]("polyglot", "glot") is True
    assert ns["n5_string_ends_with"]("polyglot", "poly") is False


def test_python_string_trim_start_end_and_is_ascii():
    ns = _run_python("""
    fn n5_string_trim_start(value: String) -> String {
        value.trim_start()
    }
    fn n5_string_trim_end(value: String) -> String {
        value.trim_end()
    }
    fn n5_string_is_ascii(value: String) -> bool {
        value.is_ascii()
    }
    """)
    assert ns["n5_string_trim_start"]("  polyglot \t") == "polyglot \t"
    assert ns["n5_string_trim_end"]("  polyglot \t") == "  polyglot"
    assert ns["n5_string_is_ascii"]("polyglot") is True
    assert ns["n5_string_is_ascii"]("café") is False


def test_python_string_lines_whitespace_and_vec_sort_unstable():
    ns = _run_python("""
    fn n5_string_lines(value: String) -> Vec<String> {
        value.lines()
    }
    fn n5_string_split_whitespace(value: String) -> Vec<String> {
        value.split_whitespace()
    }
    fn n5_vec_sort_unstable(values: Vec<i64>) -> Vec<i64> {
        values.sort_unstable();
        values
    }
    """)
    assert ns["n5_string_lines"]("one\r\ntwo\n") == ["one", "two"]
    assert ns["n5_string_lines"]("") == []
    assert ns["n5_string_split_whitespace"](" one\t two\nthree ") == [
        "one", "two", "three",
    ]
    assert ns["n5_vec_sort_unstable"]([3, 1, 2, 1]) == [1, 1, 2, 3]


def test_python_string_bytes_boundaries_and_ascii_case():
    ns = _run_python("""
    fn n5_string_as_bytes(value: String) -> Vec<u8> {
        value.as_bytes()
    }
    fn n5_string_is_char_boundary(value: String, index: i64) -> bool {
        value.is_char_boundary(index)
    }
    fn n5_string_to_ascii_lowercase(value: String) -> String {
        value.to_ascii_lowercase()
    }
    fn n5_string_to_ascii_uppercase(value: String) -> String {
        value.to_ascii_uppercase()
    }
    """)
    assert ns["n5_string_as_bytes"]("café") == b"caf\xc3\xa9"
    assert ns["n5_string_is_char_boundary"]("café", 3) is True
    assert ns["n5_string_is_char_boundary"]("café", 4) is False
    assert ns["n5_string_is_char_boundary"]("café", -1) is False
    assert ns["n5_string_to_ascii_lowercase"]("CAFÉ") == "cafÉ"
    assert ns["n5_string_to_ascii_uppercase"]("café") == "CAFé"


def test_python_vec_rotate_fill_and_swap():
    ns = _run_python("""
    fn n5_vec_rotate_left(mut values: Vec<i64>, amount: i64) -> Vec<i64> {
        values.rotate_left(amount);
        values
    }
    fn n5_vec_rotate_right(mut values: Vec<i64>, amount: i64) -> Vec<i64> {
        values.rotate_right(amount);
        values
    }
    fn n5_vec_fill(mut values: Vec<i64>, value: i64) -> Vec<i64> {
        values.fill(value);
        values
    }
    fn n5_vec_swap(mut values: Vec<i64>, left: i64, right: i64) -> Vec<i64> {
        values.swap(left, right);
        values
    }
    """)
    assert ns["n5_vec_rotate_left"]([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert ns["n5_vec_rotate_left"]([1, 2, 3], 3) == [1, 2, 3]
    assert ns["n5_vec_rotate_right"]([1, 2, 3, 4], 2) == [3, 4, 1, 2]
    assert ns["n5_vec_fill"]([1, 2, 3], 7) == [7, 7, 7]
    assert ns["n5_vec_swap"]([1, 2, 3], 0, 2) == [3, 2, 1]


def test_python_vec_extend_from_slice_resize_and_split_off():
    ns = _run_python("""
    fn n5_vec_extend_from_slice(mut values: Vec<i64>, other: Vec<i64>) -> Vec<i64> {
        values.extend_from_slice(other);
        values
    }
    fn n5_vec_resize(mut values: Vec<i64>, length: i64, value: i64) -> Vec<i64> {
        values.resize(length, value);
        values
    }
    fn n5_vec_split_off(mut values: Vec<i64>, index: i64) -> i64 {
        let tail = values.split_off(index);
        values.len() * 10 + tail.len()
    }
    """)
    assert ns["n5_vec_extend_from_slice"]([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert ns["n5_vec_resize"]([1, 2, 3], 2, 9) == [1, 2]
    assert ns["n5_vec_resize"]([1], 3, 9) == [1, 9, 9]
    assert ns["n5_vec_split_off"]([1, 2, 3, 4], 2) == 22
    assert ns["n5_vec_split_off"]([1, 2], 0) == 2
    assert ns["n5_vec_split_off"]([1, 2], 2) == 20


def test_python_string_rfind_strip_and_split_once():
    ns = _run_python("""
    fn n5_string_rfind(value: String, needle: String) -> i64 {
        value.rfind(needle).unwrap_or(0) as i64
    }
    fn n5_string_strip_prefix(value: String, prefix: String) -> String {
        value.strip_prefix(prefix).unwrap_or("missing".to_string())
    }
    fn n5_string_strip_suffix(value: String, suffix: String) -> String {
        value.strip_suffix(suffix).unwrap_or("missing".to_string())
    }
    fn n5_string_split_once(value: String, separator: String) -> Option<(String, String)> {
        value.split_once(separator)
    }
    fn n5_string_rsplit_once(value: String, separator: String) -> Option<(String, String)> {
        value.rsplit_once(separator)
    }
    """)
    assert ns["n5_string_rfind"]("caféé", "é") == 5
    assert ns["n5_string_rfind"]("café", "tea") == 0
    assert ns["n5_string_rfind"]("café x", "x") == 6
    assert ns["n5_string_strip_prefix"]("café", "ca") == "fé"
    assert ns["n5_string_strip_prefix"]("café", "tea") == "missing"
    assert ns["n5_string_strip_prefix"]("abc", "") == "abc"
    assert ns["n5_string_strip_suffix"]("café", "fé") == "ca"
    assert ns["n5_string_strip_suffix"]("café", "tea") == "missing"
    assert ns["n5_string_strip_suffix"]("abc", "") == "abc"
    assert ns["n5_string_split_once"]("a:b:c", ":").unwrap() == ("a", "b:c")
    assert ns["n5_string_rsplit_once"]("a:b:c", ":").unwrap() == ("a:b", "c")
    assert ns["n5_string_split_once"]("abc", ":").is_none()
    assert ns["n5_string_split_once"]("abc", "").unwrap() == ("", "abc")
    assert ns["n5_string_rsplit_once"]("abc", "").unwrap() == ("abc", "")


def test_python_vec_partition_sort_by_key_and_is_sorted():
    ns = _run_python("""
    fn n5_vec_partition(values: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
        values.into_iter().partition(|value| value % 2 == 0)
    }
    fn n5_vec_sort_by_key(mut values: Vec<i64>) -> Vec<i64> {
        values.sort_by_key(|value| -value);
        values
    }
    fn n5_vec_is_sorted(values: Vec<i64>) -> bool {
        values.is_sorted()
    }
    """)
    assert ns["n5_vec_partition"]([1, 2, 3, 4]) == ([2, 4], [1, 3])
    assert ns["n5_vec_sort_by_key"]([1, 3, 2, 4]) == [4, 3, 2, 1]
    assert ns["n5_vec_is_sorted"]([1, 1, 2, 3]) is True
    assert ns["n5_vec_is_sorted"]([1, 3, 2]) is False

# ---------------------------------------------------------------------------
# Option unwrap (native)
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_option_unwrap(native_ns):
    assert native_ns["n5_option_unwrap_get"]() == 5
    with pytest.raises(BaseException):  # noqa: B017
        native_ns["n5_option_unwrap_fail"]()


@pytest.mark.native
def test_native_option_unwrap_or(native_ns):
    assert native_ns["n5_option_unwrap_or_some"]() == 5
    assert native_ns["n5_option_unwrap_or_none"]() == 0


@pytest.mark.native
def test_native_option_unwrap_or_else(native_ns):
    assert native_ns["n5_option_unwrap_or_else_some"]() == 5
    assert native_ns["n5_option_unwrap_or_else_none"]() == 99


@pytest.mark.native
def test_native_option_unwrap_or_else_called_on_none(native_ns):
    assert native_ns["n5_option_unwrap_or_else_on_none"]() == 99


def test_python_unwrap_parity():
    ns = _run_python("""
    fn some() -> i64 { Some(5).unwrap() }
    fn none_or() -> i64 { None.unwrap_or(0) }
    fn none_or_else() -> i64 { None.unwrap_or_else(|| 99) }
    """)
    assert ns["some"]() == 5
    assert ns["none_or"]() == 0
    assert ns["none_or_else"]() == 99


# ---------------------------------------------------------------------------
# Option predicates
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_option_is_some(native_ns):
    assert native_ns["n5_option_is_some_true"]() is True
    assert native_ns["n5_option_is_none_false"]() is False


@pytest.mark.native
def test_native_option_is_none_with_annotation(native_ns):
    assert native_ns["n5_option_is_none_true"]() is True
    assert native_ns["n5_option_is_some_false"]() is False


def test_python_option_predicates_parity():
    ns = _run_python("""
    fn a() -> bool { Some(1).is_some() }
    fn b() -> bool { Some(1).is_none() }
    fn c() -> bool {
        let x: Option<i64> = None;
        x.is_none()
    }
    fn d() -> bool {
        let x: Option<i64> = None;
        x.is_some()
    }
    """)
    assert ns["a"]() is True
    assert ns["b"]() is False
    assert ns["c"]() is True
    assert ns["d"]() is False


# ---------------------------------------------------------------------------
# Result predicates (Python only)
# ---------------------------------------------------------------------------

def test_python_result_predicates_parity():
    ns = _run_python("""
    fn a() -> bool { Ok(1).is_ok() }
    fn b() -> bool { Ok(1).is_err() }
    fn c() -> bool { Err("x").is_err() }
    fn d() -> bool { Err("x").is_ok() }
    """)
    assert ns["a"]() is True
    assert ns["b"]() is False
    assert ns["c"]() is True
    assert ns["d"]() is False


# ---------------------------------------------------------------------------
# ok_or (Python only)
# ---------------------------------------------------------------------------

def test_python_option_ok_or_parity():
    ns = _run_python("""
    fn some() -> Result<i64, String> {
        Some(5).ok_or("missing".to_string())
    }
    fn none() -> Result<i64, String> {
        let x: Option<i64> = None;
        x.ok_or("missing".to_string())
    }
    """)
    assert ns["some"]() == ns["_Result"].from_ok(5)
    assert ns["none"]() == ns["_Result"].from_err("missing")


# ---------------------------------------------------------------------------
# map_err (Python only)
# ---------------------------------------------------------------------------

def test_python_result_map_err_parity():
    ns = _run_python("""
    fn ok() -> Result<i64, String> {
        Ok(5).map_err(|s| format!("wrapped: {}", s))
    }
    fn err() -> Result<i64, String> {
        Err("bad").map_err(|s| format!("wrapped: {}", s))
    }
    """)
    assert ns["ok"]() == ns["_Result"].from_ok(5)
    assert ns["err"]() == ns["_Result"].from_err("wrapped: bad")


# ---------------------------------------------------------------------------
# unwrap_or_else error branch (Python)
# ---------------------------------------------------------------------------

def test_python_option_unwrap_or_else_called_on_none():
    ns = _run_python("""
    fn a() -> i64 { None.unwrap_or_else(|| 99) }
    """)
    assert ns["a"]() == 99


# ---------------------------------------------------------------------------
# Nested Option
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_nested_option_construction(native_ns):
    assert native_ns["n5_nested_option"]() == 1


def test_python_nested_option_construction():
    ns = _run_python("""
    fn a() -> Option<Option<i64>> {
        Some(Some(1))
    }
    """)
    assert ns["a"]() == ns["_Option"].some(ns["_Option"].some(1))


# ---------------------------------------------------------------------------
# Result with String payload
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_result_string_ok(native_ns):
    assert native_ns["n5_result_string_ok"]() == "hello"


@pytest.mark.native
def test_native_result_string_err(native_ns):
    with pytest.raises(Exception, match="bad"):
        native_ns["n5_result_string_err"]()


def test_python_result_string_parity():
    ns = _run_python("""
    fn ok() -> Result<String, String> {
        Ok("hello".to_string())
    }
    fn err() -> Result<String, String> {
        Err("bad".to_string())
    }
    """)
    assert ns["ok"]() == ns["_Result"].from_ok("hello")
    assert ns["err"]() == ns["_Result"].from_err("bad")


# ---------------------------------------------------------------------------
# Async (native)
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_async_add(native_ns):
    import asyncio

    async def _await():
        return await native_ns["n5_async_add"](3, 4)

    assert asyncio.run(_await()) == 7


@pytest.mark.native
def test_native_async_identity(native_ns):
    import asyncio

    async def _await():
        return await native_ns["n5_async_identity"](42)

    assert asyncio.run(_await()) == 42


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def test_sandbox_blocks_import():
    with pytest.raises(NameError):
        ns = _run_python_sandboxed("""
        fn evil() -> i64 {
            let x = __import__("os");
            0
        }
        """)
        ns["evil"]()


def test_sandbox_blocks_eval():
    with pytest.raises(NameError):
        ns = _run_python_sandboxed("""
        fn evil() -> i64 {
            let x = eval("1 + 1");
            0
        }
        """)
        ns["evil"]()


def test_sandbox_blocks_open():
    with pytest.raises(NameError):
        ns = _run_python_sandboxed("""
        fn evil() -> i64 {
            let x = open("/etc/passwd");
            0
        }
        """)
        ns["evil"]()


def test_sandbox_blocks_compile():
    with pytest.raises(NameError):
        ns = _run_python_sandboxed("""
        fn evil() -> i64 {
            let x = compile("1 + 1", "<s>", "eval");
            0
        }
        """)
        ns["evil"]()


def test_sandbox_blocks_globals():
    with pytest.raises(NameError):
        ns = _run_python_sandboxed("""
        fn evil() -> i64 {
            let x = globals();
            0
        }
        """)
        ns["evil"]()


def test_sandbox_allows_safe_builtins():
    ns = _run_python_sandboxed("""
    fn safe() -> i64 {
        let xs = [1, 2, 3];
        xs.len()
    }
    """)
    assert ns["safe"]() == 3


def test_sandbox_allows_isinstance():
    ns = _run_python_sandboxed("""
    fn safe(x: i64) -> bool {
        true
    }
    """)
    assert ns["safe"](1) is True


def test_sandbox_allows_len():
    ns = _run_python_sandboxed("""
    fn safe(s: String) -> i64 {
        s.len()
    }
    """)
    assert ns["safe"]("abc") == 3


def test_sandbox_allows_exceptions():
    ns = _run_python_sandboxed("""
    fn safe(x: i64) -> i64 {
        if x < 0 {
            panic!("negative")
        }
        x
    }
    """)
    assert ns["safe"](5) == 5
    with pytest.raises(RuntimeError, match="negative"):
        ns["safe"](-1)
