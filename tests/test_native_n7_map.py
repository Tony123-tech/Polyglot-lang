# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N6: HashMap."""

from __future__ import annotations

import pytest

from lexer import lex
from parser import parse

SOURCES = {
    "map_lookup": """
fn n6_map_get(values: HashMap<String, i64>, key: String) -> i64 {
    values.get(&key).unwrap_or(-1)
}
fn n6_map_contains(values: HashMap<String, i64>, key: String) -> bool {
    values.contains_key(&key)
}
""",
    "map_insert": """
fn n6_map_insert(mut values: HashMap<String, i64>, key: String, value: i64) -> i64 {
    values.insert(key, value);
    values.len() as i64
}
fn n6_map_insert_previous(
    mut values: HashMap<String, i64>, key: String, value: i64,
) -> Option<i64> {
    values.insert(key, value)
}
""",
    "map_remove": """
fn n6_map_remove(mut values: HashMap<String, i64>, key: String) -> Option<i64> {
    values.remove(&key)
}
fn n6_map_is_empty(values: HashMap<String, i64>) -> bool {
    values.is_empty()
}
""",
    "map_local": """
fn n6_map_local() -> i64 {
    let mut values = HashMap::new();
    values.insert("a".to_string(), 5);
    values.get(&"a".to_string()).unwrap_or(-1)
}
""",
    "map_views": """
fn n6_map_keys(values: HashMap<String, i64>) -> Vec<String> {
    values.keys()
}
fn n6_map_values(values: HashMap<String, i64>) -> Vec<i64> {
    values.values()
}
fn n6_map_iter(values: HashMap<String, i64>) -> Vec<(String, i64)> {
    values.iter()
}
""",
    "map_get_mut": """
fn n6_map_bump(mut values: HashMap<String, i64>, key: String) -> i64 {
    let slot = values.get_mut(&key).unwrap();
    *slot += 1;
    *slot
}
""",
    "map_struct": """
struct N6Counter { values: HashMap<String, i64> }

impl N6Counter {
    fn new() -> N6Counter {
        N6Counter { values: HashMap::new() }
    }
    fn get(&self, key: String) -> i64 {
        self.values.get(&key).unwrap_or(-1)
    }
    fn entry_bump(&mut self, key: String) -> i64 {
        *self.values.entry(key).or_insert(0)
    }
    fn bump(&mut self, key: String) -> i64 {
        let slot = self.values.get_mut(&key).unwrap();
        *slot += 1;
        *slot
    }
}
""",
    "map_receivers": """
struct N7Inner { values: HashMap<String, i64> }
struct N7Outer { inner: N7Inner, direct: HashMap<String, i64> }

impl N7Inner {
    fn new() -> N7Inner {
        N7Inner { values: HashMap::new() }
    }
    fn seed(&mut self) {
        self.values.insert("a".to_string(), 1);
    }
}

impl N7Outer {
    fn new() -> N7Outer {
        N7Outer { inner: N7Inner::new(), direct: HashMap::new() }
    }
    fn seed(&mut self) {
        self.inner.seed();
        self.direct.insert("b".to_string(), 2);
    }
    fn nested_get(&self, key: String) -> i64 {
        self.inner.values.get(&key).unwrap_or(-1)
    }
    fn direct_get(&self, key: String) -> i64 {
        self.direct.get(&key).unwrap_or(-1)
    }
}

fn n7_param_map_get(target: &N7Outer, key: String) -> i64 {
    target.direct.get(&key).unwrap_or(-1)
}
fn n7_param_nested_get(target: &N7Outer, key: String) -> i64 {
    target.inner.values.get(&key).unwrap_or(-1)
}
""",
    "map_entry_key": """
fn n7_entry_key(mut values: HashMap<String, i64>, key: String) -> String {
    values.entry(key).key().clone()
}
fn n7_entry_or_insert_with_key(
    mut values: HashMap<String, i64>, key: String,
) -> i64 {
    *values.entry(key.clone()).or_insert_with_key(|k| k.len() as i64)
}
""",
    "map_entry": """
fn n6_map_entry_or_insert(
    mut values: HashMap<String, i64>, key: String, value: i64,
) -> i64 {
    *values.entry(key).or_insert(value)
}
fn n6_map_entry_or_insert_with(
    mut values: HashMap<String, i64>, key: String,
) -> i64 {
    *values.entry(key).or_insert_with(|| 7)
}
fn n6_map_and_modify(
    mut values: HashMap<String, i64>, key: String,
) -> i64 {
    values.entry(key.clone())
        .and_modify(|value| *value += 1)
        .or_insert(0);
    values.get(&key).unwrap_or(-1)
}
""",
}


def _run_python(src: str) -> dict:
    from backends.python.translator import compile_rust

    return compile_rust(parse(lex(src)))


@pytest.mark.native
def test_native_map_lookup(native_ns):
    ns = native_ns
    values = {"a": 1, "b": 2}
    assert ns["n6_map_get"](values, "a") == 1
    assert ns["n6_map_get"](values, "missing") == -1
    assert ns["n6_map_contains"](values, "b") is True
    assert ns["n6_map_contains"](values, "missing") is False


@pytest.mark.native
def test_native_map_insert_and_remove(native_ns):
    ns = native_ns
    assert ns["n6_map_insert"]({"a": 1}, "b", 2) == 2
    previous = ns["n6_map_insert_previous"]({"a": 1}, "a", 9)
    assert previous == 1
    assert ns["n6_map_insert_previous"]({}, "a", 9) is None
    assert ns["n6_map_remove"]({"a": 1, "b": 2}, "a") == 1
    assert ns["n6_map_remove"]({"a": 1}, "missing") is None
    assert ns["n6_map_is_empty"]({}) is True
    assert ns["n6_map_is_empty"]({"a": 1}) is False


@pytest.mark.native
def test_native_map_views(native_ns):
    ns = native_ns
    values = {"b": 2, "a": 1}
    assert sorted(ns["n6_map_keys"](values)) == ["a", "b"]
    assert sorted(ns["n6_map_values"](values)) == [1, 2]
    assert sorted(ns["n6_map_iter"](values)) == [("a", 1), ("b", 2)]


@pytest.mark.native
def test_native_map_get_mut(native_ns):
    ns = native_ns
    assert ns["n6_map_bump"]({"a": 1}, "a") == 2


@pytest.mark.native
def test_native_map_entry(native_ns):
    ns = native_ns
    assert ns["n6_map_entry_or_insert"]({}, "a", 3) == 3
    assert ns["n6_map_entry_or_insert"]({"a": 9}, "a", 3) == 9
    assert ns["n6_map_entry_or_insert_with"]({}, "a") == 7
    assert ns["n6_map_entry_or_insert_with"]({"a": 2}, "a") == 2
    assert ns["n6_map_and_modify"]({"a": 1}, "a") == 2
    assert ns["n6_map_and_modify"]({}, "a") == 0


@pytest.mark.native
def test_native_map_struct_field(native_ns):
    counter = native_ns["N6Counter"]()
    assert counter.get("a") == -1
    assert counter.entry_bump("a") == 0
    assert counter.bump("a") == 1
    assert counter.get("a") == 1


def test_python_map_struct_field():
    ns = _run_python(SOURCES["map_struct"])
    counter = ns["N6Counter"].new()
    assert counter.get("a") == -1
    assert counter.entry_bump("a") == 0
    assert counter.bump("a") == 1
    assert counter.get("a") == 1


def test_python_map_lookup():
    ns = _run_python(SOURCES["map_lookup"])
    values = {"a": 1, "b": 2}
    assert ns["n6_map_get"](values, "a") == 1
    assert ns["n6_map_get"](values, "missing") == -1
    assert ns["n6_map_contains"](values, "b") is True
    assert ns["n6_map_contains"](values, "missing") is False


def test_python_map_insert_and_remove():
    ns = _run_python(SOURCES["map_insert"] + SOURCES["map_remove"])
    assert ns["n6_map_insert"]({"a": 1}, "b", 2) == 2
    previous = ns["n6_map_insert_previous"]({"a": 1}, "a", 9)
    assert previous.unwrap() == 1
    assert ns["n6_map_insert_previous"]({}, "a", 9).is_none()
    assert ns["n6_map_remove"]({"a": 1, "b": 2}, "a").unwrap() == 1
    assert ns["n6_map_remove"]({"a": 1}, "missing").is_none()
    assert ns["n6_map_is_empty"]({}) is True
    assert ns["n6_map_is_empty"]({"a": 1}) is False


@pytest.mark.native
def test_native_map_local_binding(native_ns):
    assert native_ns["n6_map_local"]() == 5


def test_python_map_local_binding():
    ns = _run_python(SOURCES["map_local"])
    assert ns["n6_map_local"]() == 5


def test_python_map_views():
    ns = _run_python(SOURCES["map_views"])
    values = {"b": 2, "a": 1}
    assert sorted(ns["n6_map_keys"](values)) == ["a", "b"]
    assert sorted(ns["n6_map_values"](values)) == [1, 2]
    assert sorted(ns["n6_map_iter"](values)) == [("a", 1), ("b", 2)]


def test_python_map_get_mut():
    ns = _run_python(SOURCES["map_get_mut"])
    assert ns["n6_map_bump"]({"a": 1}, "a") == 2


def test_python_map_entry():
    ns = _run_python(SOURCES["map_entry"])
    assert ns["n6_map_entry_or_insert"]({}, "a", 3) == 3
    assert ns["n6_map_entry_or_insert"]({"a": 9}, "a", 3) == 9
    assert ns["n6_map_entry_or_insert_with"]({}, "a") == 7
    assert ns["n6_map_entry_or_insert_with"]({"a": 2}, "a") == 2
    assert ns["n6_map_and_modify"]({"a": 1}, "a") == 2
    assert ns["n6_map_and_modify"]({}, "a") == 0


@pytest.mark.native
def test_native_map_nested_and_param_receivers(native_ns):
    ns = native_ns
    outer = ns["N7Outer"]()
    outer.seed()
    assert outer.nested_get("a") == 1
    assert outer.direct_get("b") == 2
    assert outer.direct_get("missing") == -1
    assert ns["n7_param_map_get"](outer, "b") == 2
    assert ns["n7_param_nested_get"](outer, "a") == 1


def test_python_map_nested_and_param_receivers():
    ns = _run_python(SOURCES["map_receivers"])
    outer = ns["N7Outer"].new()
    outer.seed()
    assert outer.nested_get("a") == 1
    assert outer.direct_get("b") == 2
    assert outer.direct_get("missing") == -1
    assert ns["n7_param_map_get"](outer, "b") == 2
    assert ns["n7_param_nested_get"](outer, "a") == 1


@pytest.mark.native
def test_native_map_entry_key(native_ns):
    ns = native_ns
    assert ns["n7_entry_key"]({}, "missing") == "missing"
    assert ns["n7_entry_or_insert_with_key"]({}, "abcd") == 4
    assert ns["n7_entry_or_insert_with_key"]({"abcd": 9}, "abcd") == 9


def test_python_map_entry_key():
    ns = _run_python(SOURCES["map_entry_key"])
    assert ns["n7_entry_key"]({}, "missing") == "missing"
    assert ns["n7_entry_or_insert_with_key"]({}, "abcd") == 4
    assert ns["n7_entry_or_insert_with_key"]({"abcd": 9}, "abcd") == 9


def test_python_invalid_dereference_assignment_rejected():
    from backends.python.translator import TranslateError

    src = """
fn n6_bad_deref() -> i64 {
    let mut x = 1;
    *x += 1;
    x
}
"""
    with pytest.raises(TranslateError, match="mutable map slot"):
        _run_python(src)


def test_python_closure_slot_mutation_rejected_for_unknown_target():
    from backends.python.translator import TranslateError

    src = """
fn n6_bad_closure_slot(mut values: HashMap<String, i64>) {
    values.entry("a".to_string()).and_modify(|_value| *missing += 1);
}
"""
    with pytest.raises(TranslateError, match="mutable map slot"):
        _run_python(src)
