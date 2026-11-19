"""Cross-backend parity and edge cases for the priority feature batch.

Covers vec! lowering, String::from, lazy unwrap_or_else, std:fs, if-let in
expression position and the extended HashMap entry API. Native cases use the
shared session namespace where possible; std:fs compiles its own module so
include expansion is exercised end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backends.errors import TranslateError
from backends.python.translator import PythonBackend, _Option
from backends.rust.codegen import RustCodegen
from include_expander import load_program
from lexer import lex
from parser import parse

SOURCES = {
    "pf_vec_list": """
fn pf_vec_list() -> usize { vec![1, 2, 3].len() }
""",
    "pf_vec_repeat": """
fn pf_vec_repeat(n: usize) -> usize { vec![7; n].len() }
""",
    "pf_vec_nested": """
fn pf_vec_nested() -> usize { vec![vec![1, 2], vec![3, 4, 5]].len() }
""",
    "pf_string_from": """
fn pf_string_from(name: String) -> String { String::from("hello") + name }
""",
    "pf_option_lazy": """
fn pf_explode() -> i64 { let x: Option<i64> = None; x.unwrap() }
fn pf_option_lazy(x: Option<i64>) -> i64 { x.unwrap_or_else(|| pf_explode()) }
""",
    "pf_result_lazy": """
fn pf_result_lazy(ok: bool) -> i64 {
    let x: Result<i64, String> = if ok { Ok(1) } else { Err("bad".to_string()) };
    x.unwrap_or_else(|_| pf_explode())
}
""",
    "pf_iflet_expr_some": """
fn pf_iflet_expr_some(x: Option<i64>) -> i64 {
    let y = if let Some(v) = x { v } else { 0 };
    y + 1
}
""",
    "pf_iflet_expr_none": """
fn pf_iflet_expr_none(x: Option<i64>) -> i64 {
    return if let None = x { 7 } else { 1 };
}
""",
    "pf_iflet_expr_result": """
fn pf_iflet_expr_result(is_err: bool) -> i64 {
    let x: Result<i64, String> = if is_err { Err("bad".to_string()) } else { Ok(1) };
    return if let Err(_) = x { -1 } else { 1 };
}
""",
    "pf_entry_default": """
fn pf_entry_default(values: HashMap<String, i64>, key: String) -> i64 {
    *values.entry(key).or_default()
}
""",
    "pf_entry_modify_default": """
fn pf_entry_modify_default(values: HashMap<String, i64>, key: String) -> i64 {
    *values.entry(key).and_modify(|v| *v += 1).or_default()
}
""",
    "pf_entry_with_key": """
fn pf_entry_with_key(values: HashMap<String, i64>, key: String) -> i64 {
    *values.entry(key).or_insert_with_key(|k| k.len() as i64)
}
""",
}


def _python_ns(source: str) -> dict:
    return PythonBackend().compile(parse(lex(source)))


# ---------------------------------------------------------------------------
# vec! macro lowering (feature 1)
# ---------------------------------------------------------------------------


def test_python_vec_lowering_all_forms():
    ns = _python_ns(
        SOURCES["pf_vec_list"] + SOURCES["pf_vec_repeat"] + SOURCES["pf_vec_nested"]
    )
    assert ns["pf_vec_list"]() == 3
    assert ns["pf_vec_repeat"](4) == 4
    assert ns["pf_vec_repeat"](0) == 0
    assert ns["pf_vec_nested"]() == 2


@pytest.mark.native
def test_native_vec_lowering_all_forms(native_ns):
    assert native_ns["pf_vec_list"]() == 3
    assert native_ns["pf_vec_repeat"](4) == 4
    assert native_ns["pf_vec_repeat"](0) == 0
    assert native_ns["pf_vec_nested"]() == 2


def test_vec_lowering_reuses_array_ast():
    program = parse(lex(SOURCES["pf_vec_list"] + SOURCES["pf_vec_repeat"]))
    kinds = {fn.name: fn.body.tail.receiver.__class__.__name__ for fn in program.items}
    assert kinds == {"pf_vec_list": "ArrayExpr", "pf_vec_repeat": "RepeatArray"}


# ---------------------------------------------------------------------------
# String::from lowering (feature 2)
# ---------------------------------------------------------------------------


def test_python_string_from():
    ns = _python_ns(SOURCES["pf_string_from"])
    assert ns["pf_string_from"](" world") == "hello world"


@pytest.mark.native
def test_native_string_from(native_ns):
    assert native_ns["pf_string_from"](" world") == "hello world"


def test_rust_string_literal_is_not_double_wrapped():
    code = RustCodegen(module_name="pf").generate(parse(lex(SOURCES["pf_string_from"])))
    assert "String::from(String::from(" not in code
    assert 'String::from("hello")' in code


# ---------------------------------------------------------------------------
# Lazy unwrap_or_else (feature 3)
# ---------------------------------------------------------------------------


def test_python_unwrap_or_else_is_lazy():
    ns = _python_ns(SOURCES["pf_option_lazy"] + SOURCES["pf_result_lazy"])
    assert ns["pf_option_lazy"](_Option.some(5)) == 5
    with pytest.raises(ValueError, match="unwrap"):
        ns["pf_option_lazy"](_Option.none())
    assert ns["pf_result_lazy"](True) == 1
    with pytest.raises(ValueError, match="unwrap"):
        ns["pf_result_lazy"](False)


@pytest.mark.native
def test_native_unwrap_or_else_is_lazy(native_ns):
    assert native_ns["pf_option_lazy"](5) == 5
    assert native_ns["pf_result_lazy"](True) == 1
    with pytest.raises(BaseException, match="unwrap"):
        native_ns["pf_option_lazy"](None)
    with pytest.raises(BaseException, match="unwrap"):
        native_ns["pf_result_lazy"](False)


# ---------------------------------------------------------------------------
# std:fs host seam parity (feature 4)
# ---------------------------------------------------------------------------

FS_SOURCE = """
include!("std:fs");

fn pf_fs_roundtrip(path: String, contents: String) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    match write_text(path.clone(), contents) {
        Ok(_) => { out.push("write".to_string()); }
        Err(_) => { out.push("error".to_string()); }
    }
    match read_text(path.clone()) {
        Ok(text) => { out.push(text); }
        Err(_) => { out.push("error".to_string()); }
    }
    match append_text(path.clone(), "!".to_string()) {
        Ok(_) => { out.push("append".to_string()); }
        Err(_) => { out.push("error".to_string()); }
    }
    match read_text(path.clone()) {
        Ok(text) => { out.push(text); }
        Err(_) => { out.push("error".to_string()); }
    }
    out
}

fn pf_fs_exists(path: String) -> bool { exists(path) }

fn pf_fs_remove(path: String) -> bool {
    match remove(path) { Ok(_) => true, Err(_) => false }
}

fn pf_fs_read_or(path: String) -> String { read_text_or(path, "fallback".to_string()) }

fn pf_fs_nonempty(path: String) -> String {
    match read_text_nonempty(path) {
        Ok(text) => text,
        Err(_) => "empty".to_string(),
    }
}
"""


def _fs_program(tmp_path: Path):
    main = tmp_path / "pf_fs_main.poly"
    main.write_text(FS_SOURCE, encoding="utf-8")
    return load_program(main)


def test_python_fs_roundtrip(tmp_path: Path):
    from backends import get_backend

    ns = get_backend("python", trusted=True).compile(_fs_program(tmp_path))
    path = str(tmp_path / "hello.txt")
    assert ns["pf_fs_roundtrip"](path, "hello") == [
        "write",
        "hello",
        "append",
        "hello!",
    ]
    assert ns["pf_fs_exists"](path) is True
    assert ns["pf_fs_read_or"](path) == "hello!"
    assert ns["pf_fs_nonempty"](path) == "hello!"
    assert ns["pf_fs_remove"](path) is True
    assert ns["pf_fs_exists"](path) is False
    assert ns["pf_fs_read_or"](path) == "fallback"
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert "empty" in ns["pf_fs_nonempty"](str(empty))


@pytest.mark.native
def test_native_fs_roundtrip(tmp_path: Path):
    from backends import get_backend

    ns = get_backend("native", trusted=True).compile(
        _fs_program(tmp_path),
        incremental=True,
        cache_scope="priority-fs-parity",
    )
    path = str(tmp_path / "hello.txt")
    assert ns["pf_fs_roundtrip"](path, "hello") == [
        "write",
        "hello",
        "append",
        "hello!",
    ]
    assert ns["pf_fs_exists"](path) is True
    assert ns["pf_fs_read_or"](path) == "hello!"
    assert ns["pf_fs_nonempty"](path) == "hello!"
    assert ns["pf_fs_remove"](path) is True
    assert ns["pf_fs_exists"](path) is False
    assert ns["pf_fs_read_or"](path) == "fallback"
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert "empty" in ns["pf_fs_nonempty"](str(empty))


def test_fs_host_arity_error_is_span_aware():
    source = "fn bad() -> bool { __fs_exists() }"
    for generate in (
        lambda p: PythonBackend().compile(p),
        lambda p: RustCodegen(module_name="pf").generate(p),
    ):
        with pytest.raises(TranslateError) as exc:
            generate(parse(lex(source)))
        assert exc.value.code == "E003"
        assert "expects 1 argument, got 0" in exc.value.msg
        assert exc.value.line == 1
        assert exc.value.span is not None


# ---------------------------------------------------------------------------
# if let expression position (feature 5)
# ---------------------------------------------------------------------------


def test_python_iflet_expression_position():
    ns = _python_ns(
        SOURCES["pf_iflet_expr_some"]
        + SOURCES["pf_iflet_expr_none"]
        + SOURCES["pf_iflet_expr_result"]
    )
    assert ns["pf_iflet_expr_some"](_Option.some(4)) == 5
    assert ns["pf_iflet_expr_some"](_Option.none()) == 1
    assert ns["pf_iflet_expr_none"](_Option.none()) == 7
    assert ns["pf_iflet_expr_none"](_Option.some(1)) == 1
    assert ns["pf_iflet_expr_result"](True) == -1
    assert ns["pf_iflet_expr_result"](False) == 1


@pytest.mark.native
def test_native_iflet_expression_position(native_ns):
    assert native_ns["pf_iflet_expr_some"](4) == 5
    assert native_ns["pf_iflet_expr_some"](None) == 1
    assert native_ns["pf_iflet_expr_none"](None) == 7
    assert native_ns["pf_iflet_expr_none"](1) == 1
    assert native_ns["pf_iflet_expr_result"](True) == -1
    assert native_ns["pf_iflet_expr_result"](False) == 1


def test_iflet_expression_emitted_as_rust_expression():
    code = RustCodegen(module_name="pf").generate(
        parse(lex(SOURCES["pf_iflet_expr_some"]))
    )
    assert "(if let Some(v) = x { v } else { 0 })" in code


# ---------------------------------------------------------------------------
# HashMap entry combinators (feature 6)
# ---------------------------------------------------------------------------


def test_python_entry_combinators():
    ns = _python_ns(
        SOURCES["pf_entry_default"]
        + SOURCES["pf_entry_modify_default"]
        + SOURCES["pf_entry_with_key"]
    )
    assert ns["pf_entry_default"]({}, "a") == 0
    assert ns["pf_entry_default"]({"a": 5}, "a") == 5
    assert ns["pf_entry_modify_default"]({}, "a") == 0
    assert ns["pf_entry_modify_default"]({"a": 5}, "a") == 6
    assert ns["pf_entry_with_key"]({}, "abc") == 3


@pytest.mark.native
def test_native_entry_combinators(native_ns):
    assert native_ns["pf_entry_default"]({}, "a") == 0
    assert native_ns["pf_entry_default"]({"a": 5}, "a") == 5
    assert native_ns["pf_entry_modify_default"]({}, "a") == 0
    assert native_ns["pf_entry_modify_default"]({"a": 5}, "a") == 6
    assert native_ns["pf_entry_with_key"]({}, "abc") == 3
