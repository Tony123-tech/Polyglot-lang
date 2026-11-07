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


# ---------------------------------------------------------------------------
# Option unwrap (native)
# ---------------------------------------------------------------------------

@pytest.mark.native
def test_native_option_unwrap(native_ns):
    assert native_ns["n5_option_unwrap_get"]() == 5
    with pytest.raises(BaseException):
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
