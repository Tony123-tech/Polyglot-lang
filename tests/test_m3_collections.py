"""M3-remaining: HashMap API completion, BTreeMap, HashSet."""

from __future__ import annotations

import pytest

from backends import get_backend
from compiler.core.lexer import lex
from compiler.core.parser import parse


def _run(src: str, backend: str, tmp_path=None):
    program = parse(lex(src))
    if backend == "python":
        ns = get_backend("python", trusted=True).compile(program)
    else:
        from backends.rust.compiler import NativeBackend

        ns = NativeBackend(cache_dir=tmp_path).compile(program)
    return ns["main"]()


# ---------------------------------------------------------------------------
# HashSet
# ---------------------------------------------------------------------------


HASHSET_SRC = r"""
fn main() -> i64 {
    let mut s: HashSet<i64> = HashSet::new();
    let a = s.insert(1);
    let b = s.insert(1);
    s.insert(2);
    let c = s.contains(&2);
    let r = s.remove(&2);
    let mut total: i64 = 0;
    for v in s.iter() {
        total = total + v;
    }
    if a && !b && c && r { (s.len() as i64) + total } else { -1 }
}
"""


def test_hashset_insert_contains_remove_len_iter_python():
    assert _run(HASHSET_SRC, "python") == 2


@pytest.mark.native
def test_hashset_insert_contains_remove_len_iter_native(tmp_path):
    assert _run(HASHSET_SRC, "native", tmp_path) == 2


def test_hashset_string_elements_both_backends(tmp_path):
    src = r"""
fn main() -> i64 {
    let mut s: HashSet<String> = HashSet::new();
    s.insert("alpha".to_string());
    s.insert("beta".to_string());
    let has = s.contains("alpha".to_string());
    let removed = s.remove("beta".to_string());
    if has && removed { s.len() as i64 } else { -1 }
}
"""
    assert _run(src, "python") == 1
    assert _run(src, "native", tmp_path) == 1


def test_hashset_iteration_order_is_unspecified_both_backends(tmp_path):
    # Iteration order is not guaranteed; sort before comparing.
    src = r"""
fn main() -> i64 {
    let mut s: HashSet<i64> = HashSet::new();
    s.insert(3);
    s.insert(1);
    s.insert(2);
    let mut xs = s.iter();
    xs.sort();
    let mut out: i64 = 0;
    for x in xs {
        out = out * 10 + x;
    }
    out
}
"""
    assert _run(src, "python") == 123
    assert _run(src, "native", tmp_path) == 123


# ---------------------------------------------------------------------------
# BTreeMap
# ---------------------------------------------------------------------------


BTREEMAP_SRC = r"""
fn main() -> i64 {
    let mut m: BTreeMap<i64, i64> = BTreeMap::new();
    m.insert(3, 30);
    m.insert(1, 10);
    m.insert(2, 20);
    let mut out: i64 = 0;
    for (k, v) in m.iter() {
        out = out * 10 + k;
        out = out * 10 + (v / 10);
    }
    out
}
"""


def test_btreemap_sorted_iteration_python():
    assert _run(BTREEMAP_SRC, "python") == 112233


@pytest.mark.native
def test_btreemap_sorted_iteration_native(tmp_path):
    assert _run(BTREEMAP_SRC, "native", tmp_path) == 112233


def test_btreemap_core_operations_both_backends(tmp_path):
    src = r"""
fn main() -> i64 {
    let mut m: BTreeMap<String, i64> = BTreeMap::new();
    m.insert("b".to_string(), 2);
    m.insert("a".to_string(), 1);
    let a = m.get("a".to_string()).unwrap();
    let c = m.contains_key("b".to_string());
    let keys = m.keys();
    let values = m.values();
    let r = m.remove("b".to_string()).unwrap();
    let empty = m.is_empty();
    if c && !empty && (keys.len() as i64) == 2 && (values.len() as i64) == 2 {
        a * 10 + r
    } else {
        -1
    }
}
"""
    assert _run(src, "python") == 12
    assert _run(src, "native", tmp_path) == 12


def test_btreemap_supported_value_types_both_backends(tmp_path):
    src = r"""
fn main() -> i64 {
    let mut b: BTreeMap<i64, bool> = BTreeMap::new();
    b.insert(1, true);
    let mut f: BTreeMap<i64, f64> = BTreeMap::new();
    f.insert(1, 1.5);
    let mut v: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    v.insert(1, [1, 2].to_vec());
    let mut s: BTreeMap<i64, String> = BTreeMap::new();
    s.insert(1, "x".to_string());
    if b.get(1).unwrap() {
        (s.get(1).unwrap().len() as i64)
            + (v.get(1).unwrap().len() as i64)
            + (f.get(1).unwrap() as i64)
    } else {
        -1
    }
}
"""
    assert _run(src, "python") == 4
    assert _run(src, "native", tmp_path) == 4


def test_btreemap_shuffled_insert_order_is_sorted_both_backends(tmp_path):
    src = r"""
fn main() -> i64 {
    let mut m: BTreeMap<i64, i64> = BTreeMap::new();
    m.insert(4, 0);
    m.insert(1, 0);
    m.insert(5, 0);
    m.insert(2, 0);
    m.insert(3, 0);
    let mut out: i64 = 0;
    for (k, _v) in m.iter() {
        out = out * 10 + k;
    }
    out
}
"""
    assert _run(src, "python") == 12345
    assert _run(src, "native", tmp_path) == 12345


# ---------------------------------------------------------------------------
# HashMap entry extensions
# ---------------------------------------------------------------------------


ENTRY_SRC = r"""
fn main() -> i64 {
    let mut m: HashMap<String, i64> = HashMap::new();
    m.entry("a".to_string()).or_insert_with(|| 5);
    m.entry("a".to_string()).and_modify(|v| { *v = *v + 1; });
    m.entry("b".to_string()).or_insert_with_key(|k| k.len() as i64);
    m.get("a".to_string()).unwrap() + m.get("b".to_string()).unwrap()
}
"""


def test_entry_or_insert_with_and_modify_python():
    assert _run(ENTRY_SRC, "python") == 7


@pytest.mark.native
def test_entry_or_insert_with_and_modify_native(tmp_path):
    assert _run(ENTRY_SRC, "native", tmp_path) == 7


# ---------------------------------------------------------------------------
# Nested HashMap (depth 1)
# ---------------------------------------------------------------------------


NESTED_SRC = r"""
fn main() -> i64 {
    let mut inner: HashMap<i64, i64> = HashMap::new();
    inner.insert(1, 10);
    let mut outer: HashMap<String, HashMap<i64, i64>> = HashMap::new();
    outer.insert("x".to_string(), inner);
    let v = outer.get("x".to_string()).unwrap();
    v.get(1).unwrap()
}
"""


def test_nested_hashmap_depth1_python():
    assert _run(NESTED_SRC, "python") == 10


@pytest.mark.native
def test_nested_hashmap_depth1_native(tmp_path):
    assert _run(NESTED_SRC, "native", tmp_path) == 10


# ---------------------------------------------------------------------------
# get_mut copy semantics
# ---------------------------------------------------------------------------


def test_get_mut_copy_semantics_btreemap_python_boundary():
    src = r"""
fn main() -> i64 {
    let mut m: BTreeMap<i64, i64> = BTreeMap::new();
    m.insert(1, 10);
    let slot = m.get_mut(1).unwrap();
    slot.set(20);
    m.get(1).unwrap()
}
"""
    assert _run(src, "python") == 20


@pytest.mark.native
def test_get_mut_copy_semantics_btreemap_native_boundary(tmp_path):
    # At the native Python boundary the returned Option<V> is an owned copy:
    # mutating it must not mutate the input dictionary.
    src = r"""
fn peek(m: BTreeMap<i64, i64>, k: i64) -> Option<i64> {
    match m.get_mut(k) { Some(v) => Some(*v), None => None }
}
"""
    from backends.rust.compiler import NativeBackend

    ns = NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    mapping = {1: 10}
    assert ns["peek"](mapping, 1) == 10
    assert mapping == {1: 10}


@pytest.mark.native
def test_get_mut_copy_semantics_hashset_native_boundary(tmp_path):
    src = r"""
fn peek(s: HashSet<i64>, v: i64) -> Option<i64> {
    s.get_mut(v)
}
"""
    from backends.rust.compiler import NativeBackend

    ns = NativeBackend(cache_dir=tmp_path).compile(parse(lex(src)))
    values = {1, 2}
    assert ns["peek"](values, 1) == 1
    assert ns["peek"](values, 3) is None
    assert values == {1, 2}
