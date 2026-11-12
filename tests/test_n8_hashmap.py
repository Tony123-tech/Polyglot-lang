# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N8: HashMap support in the collections layer."""

from __future__ import annotations

import pytest

from lexer import lex
from parser import parse

SOURCES = {
    "n8_map_core": """
fn n8_insert(mut values: HashMap<String, i64>, key: String, value: i64) -> bool {
    values.insert(key.clone(), value);
    values.contains_key(&key)
}
fn n8_insert_previous(
    mut values: HashMap<String, i64>, key: String, value: i64,
) -> Option<i64> {
    values.insert(key, value)
}
fn n8_get(values: HashMap<String, i64>, key: String) -> Option<i64> {
    values.get(&key)
}
fn n8_remove(mut values: HashMap<String, i64>, key: String) -> Option<i64> {
    values.remove(&key)
}
fn n8_contains(values: HashMap<String, i64>, key: String) -> bool {
    values.contains_key(&key)
}
fn n8_len(values: HashMap<String, i64>) -> usize {
    values.len()
}
fn n8_is_empty(values: HashMap<String, i64>) -> bool {
    values.is_empty()
}
fn n8_new_map() -> HashMap<String, i64> {
    let mut values = HashMap::new();
    values.insert("created".to_string(), 6);
    values
}
fn n8_keys(values: HashMap<String, i64>) -> Vec<String> {
    values.keys()
}
fn n8_values(values: HashMap<String, i64>) -> Vec<i64> {
    values.values()
}
fn n8_iter_sum(values: HashMap<String, i64>) -> i64 {
    let mut total: i64 = 0;
    for (_key, value) in values.iter() {
        total = total + value;
    }
    total
}
fn n8_entry(mut values: HashMap<String, i64>, key: String, value: i64) -> i64 {
    *values.entry(key).or_insert(value)
}
fn n8_get_mut(mut values: HashMap<String, i64>, key: String) -> i64 {
    let slot = values.get_mut(&key).unwrap();
    *slot += 1;
    *slot
}
fn n8_get_mut_return(mut values: HashMap<String, i64>, key: String) -> Option<i64> {
    values.get_mut(&key)
}
""",
    "n8_map_supported_values": """
fn n8_i64_string_roundtrip(values: HashMap<i64, String>) -> HashMap<i64, String> {
    values
}
fn n8_string_bool_roundtrip(values: HashMap<String, bool>) -> HashMap<String, bool> {
    values
}
fn n8_string_f64_roundtrip(values: HashMap<String, f64>) -> HashMap<String, f64> {
    values
}
fn n8_string_vec_roundtrip(
    values: HashMap<String, Vec<i64>>,
) -> HashMap<String, Vec<i64>> {
    values
}
""",
}


def _run_python():
    from backends.python.translator import compile_rust

    source = "\n".join(SOURCES.values())
    return compile_rust(parse(lex(source)))


def test_python_hashmap_operations_and_option_results():
    ns = _run_python()
    values = {"existing": 4}
    assert ns["n8_insert"](values, "new", 7) is True
    assert ns["n8_insert_previous"](
        {"existing": 4}, "existing", 7
    ).unwrap() == 4
    assert ns["n8_insert_previous"]({}, "new", 7).is_none()
    assert ns["n8_get"](values, "new").unwrap() == 7
    assert ns["n8_get"](values, "missing").is_none()
    assert ns["n8_contains"](values, "existing") is True
    assert ns["n8_contains"](values, "missing") is False
    assert ns["n8_remove"](values, "new").unwrap() == 7
    assert ns["n8_remove"](values, "missing").is_none()
    assert ns["n8_is_empty"]({}) is True
    assert ns["n8_is_empty"]({"a": 1}) is False
    assert ns["n8_new_map"]() == {"created": 6}


@pytest.mark.native
def test_native_hashmap_operations_and_option_results(native_ns):
    assert native_ns["n8_insert"]({"existing": 4}, "new", 7) is True
    assert native_ns["n8_insert_previous"](
        {"existing": 4}, "existing", 7
    ) == 4
    assert native_ns["n8_insert_previous"]({}, "new", 7) is None
    assert native_ns["n8_get"]({"new": 7}, "new") == 7
    assert native_ns["n8_get"]({}, "missing") is None
    assert native_ns["n8_contains"]({"existing": 4}, "existing") is True
    assert native_ns["n8_contains"]({}, "missing") is False
    assert native_ns["n8_remove"]({"new": 7}, "new") == 7
    assert native_ns["n8_remove"]({}, "missing") is None
    assert native_ns["n8_is_empty"]({}) is True
    assert native_ns["n8_is_empty"]({"a": 1}) is False
    assert native_ns["n8_new_map"]() == {"created": 6}


def test_python_hashmap_len_and_boundary_error():
    from backends.python.translator import TypeErrorForBoundary

    ns = _run_python()
    assert ns["n8_len"]({"a": 1, "b": 2}) == 2
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        from backends.python.translator import compile_rust

        compile_rust(parse(lex(
            "fn n8_bad_len(values: HashMap<String, i64>) -> i64 "
            "{ values.len() }"
        )))
    assert "found:    usize" in str(exc_info.value)
    assert "expected: i64" in str(exc_info.value)


def test_native_hashmap_len_and_boundary_error(native_backend):
    from backends.rust.compiler import TypeErrorForBoundary

    assert native_backend.compile(parse(lex(
        "fn n8_valid_len(values: HashMap<String, i64>) -> usize "
        "{ values.len() }"
    )))["n8_valid_len"]({"a": 1, "b": 2}) == 2
    with pytest.raises(TypeErrorForBoundary) as exc_info:
        native_backend.compile(parse(lex(
            "fn n8_bad_len(values: HashMap<String, i64>) -> i64 "
            "{ values.len() }"
        )))
    assert "found:    usize" in str(exc_info.value)
    assert "expected: i64" in str(exc_info.value)


def test_python_hashmap_views_and_iteration():
    ns = _run_python()
    values = {"a": 2, "b": 3}
    assert sorted(ns["n8_keys"](values)) == ["a", "b"]
    assert sorted(ns["n8_values"](values)) == [2, 3]
    assert ns["n8_iter_sum"](values) == 5


@pytest.mark.native
def test_native_hashmap_views_and_iteration(native_ns):
    values = {"a": 2, "b": 3}
    assert sorted(native_ns["n8_keys"](values)) == ["a", "b"]
    assert sorted(native_ns["n8_values"](values)) == [2, 3]
    assert native_ns["n8_iter_sum"](values) == 5


def test_python_hashmap_mutable_access_and_entry():
    ns = _run_python()
    assert ns["n8_get_mut"]({"a": 4}, "a") == 5
    assert ns["n8_entry"]({}, "a", 9) == 9
    assert ns["n8_entry"]({"a": 3}, "a", 9) == 3

    values = {"a": 4}
    slot = ns["n8_get_mut_return"](values, "a").unwrap()
    slot.set(12)
    assert values["a"] == 12


@pytest.mark.native
def test_native_hashmap_mutable_access_and_entry(native_ns):
    assert native_ns["n8_get_mut"]({"a": 4}, "a") == 5
    assert native_ns["n8_entry"]({}, "a", 9) == 9
    assert native_ns["n8_entry"]({"a": 3}, "a", 9) == 3

    original = {"a": 4}
    copied_value = native_ns["n8_get_mut_return"](original, "a")
    assert copied_value == 4
    assert original == {"a": 4}


@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("n8_i64_string_roundtrip", {1: "one"}),
        ("n8_string_bool_roundtrip", {"enabled": True}),
        ("n8_string_f64_roundtrip", {"ratio": 1.25}),
        ("n8_string_vec_roundtrip", {"numbers": [1, 2, 3]}),
    ],
)
def test_python_hashmap_supported_key_value_types(function_name, values):
    assert _run_python()[function_name](values) == values


@pytest.mark.native
@pytest.mark.parametrize(
    ("function_name", "values"),
    [
        ("n8_i64_string_roundtrip", {1: "one"}),
        ("n8_string_bool_roundtrip", {"enabled": True}),
        ("n8_string_f64_roundtrip", {"ratio": 1.25}),
        ("n8_string_vec_roundtrip", {"numbers": [1, 2, 3]}),
    ],
)
def test_native_hashmap_supported_key_value_types(
    native_ns, function_name, values,
):
    assert native_ns[function_name](values) == values


def test_hashmap_operations_match_across_backends(native_ns):
    python_ns = _run_python()
    values = {"a": 2, "b": 3}
    assert python_ns["n8_len"](values) == native_ns["n8_len"](values)
    assert sorted(python_ns["n8_keys"](values)) == sorted(
        native_ns["n8_keys"](values)
    )
    assert sorted(python_ns["n8_values"](values)) == sorted(
        native_ns["n8_values"](values)
    )
    assert python_ns["n8_iter_sum"](values) == native_ns["n8_iter_sum"](
        values
    )
