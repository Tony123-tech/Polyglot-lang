# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""N5 native and Python Option/Result tests."""

import pytest

from ast_nodes import MatchExpr
from backends.python.translator import compile_rust
from lexer import lex
from parser import parse

SOURCES = {
    "n5_option_is_some": """
fn n5_option_is_some() -> bool {
    Some(7).is_some()
}
""",
    "n5_option_is_none": """
fn n5_option_is_none() -> bool {
    let value: Option<i64> = None;
    value.is_none()
}
""",
    "n5_option_unwrap": """
fn n5_option_unwrap() -> i64 {
    Some(7).unwrap()
}
""",
    "n5_option_unwrap_or": """
fn n5_option_unwrap_or() -> i64 {
    let value: Option<i64> = None;
    value.unwrap_or(9)
}
""",
    "n5_option_unwrap_or_else": """
fn n5_option_unwrap_or_else() -> i64 {
    let value: Option<i64> = None;
    value.unwrap_or_else(|| 11)
}
""",
    "n5_option_unwrap_or_else_some": """
fn n5_option_unwrap_or_else_some() -> i64 {
    Some(13).unwrap_or_else(|| 17)
}
""",
    "n5_option_and_then_some": """
fn n5_option_and_then_some() -> i64 {
    Some(3).and_then(|value| Some(value + 2)).unwrap_or(0)
}
""",
    "n5_option_and_then_none": """
fn n5_option_and_then_none() -> i64 {
    let value: Option<i64> = None;
    value.and_then(|number| Some(number + 2)).unwrap_or(4)
}
""",
    "n5_option_or_else_some": """
fn n5_option_or_else_some() -> i64 {
    Some(8).or_else(|| Some(10)).unwrap_or(0)
}
""",
    "n5_option_or_else_none": """
fn n5_option_or_else_none() -> i64 {
    let value: Option<i64> = None;
    value.or_else(|| Some(10)).unwrap_or(0)
}
""",
    "n5_option_filter_some": """
fn n5_option_filter_some() -> i64 {
    Some(6).filter(|value| *value > 3).unwrap_or(0)
}
""",
    "n5_option_filter_none": """
fn n5_option_filter_none() -> i64 {
    Some(2).filter(|value| *value > 3).unwrap_or(0)
}
""",
    "n5_option_ok_or_some": """
fn n5_option_ok_or_some() -> bool {
    Some(7).ok_or("failure").is_ok()
}
""",
    "n5_option_ok_or_none": """
fn n5_option_ok_or_none() -> bool {
    let value: Option<i64> = None;
    value.ok_or("failure").is_err()
}
""",
    "n5_result_ok_to_option": """
fn n5_result_ok_to_option() -> i64 {
    Ok(7).ok().unwrap_or(0)
}
""",
    "n5_result_err_to_option": """
fn n5_result_err_to_option() -> bool {
    let result: Result<i64, PyErr> = Err("failure");
    result.err().is_some()
}
""",
    "n5_parse_pair": """
fn n5_parse_pair(s: String) -> Result<i64, String> {
    let n = s.parse::<i64>()?;
    Ok(n * 2)
}
""",
    "n5_if_let_some": """
fn n5_if_let_some() -> i64 {
    let value: Option<i64> = Some(7);
    if let Some(number) = value {
        return number + 1;
    } else {
        return 0;
    }
}
""",
    "n5_if_let_none": """
fn n5_if_let_none() -> i64 {
    let value: Option<i64> = None;
    if let Some(number) = value {
        return number + 1;
    } else {
        return 0;
    }
}
""",
}


def _python_ns(name):
    return compile_rust(parse(lex(SOURCES[name])))


@pytest.mark.native
def test_native_option_methods(native_ns):
    assert native_ns["n5_option_is_some"]() is True
    assert native_ns["n5_option_is_none"]() is True
    assert native_ns["n5_option_unwrap"]() == 7
    assert native_ns["n5_option_unwrap_or"]() == 9
    assert native_ns["n5_option_unwrap_or_else"]() == 11
    assert native_ns["n5_option_unwrap_or_else_some"]() == 13
    assert native_ns["n5_option_and_then_some"]() == 5
    assert native_ns["n5_option_and_then_none"]() == 4
    assert native_ns["n5_option_or_else_some"]() == 8
    assert native_ns["n5_option_or_else_none"]() == 10
    assert native_ns["n5_option_filter_some"]() == 6
    assert native_ns["n5_option_filter_none"]() == 0
    assert native_ns["n5_option_ok_or_some"]() is True
    assert native_ns["n5_option_ok_or_none"]() is True
    assert native_ns["n5_result_ok_to_option"]() == 7
    assert native_ns["n5_result_err_to_option"]() is True


@pytest.mark.native
def test_native_result_try_operator(native_ns):
    assert native_ns["n5_parse_pair"]("21") == 42


@pytest.mark.native
def test_native_if_let_some_and_none(native_ns):
    assert native_ns["n5_if_let_some"]() == 8
    assert native_ns["n5_if_let_none"]() == 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("n5_option_is_some", True),
        ("n5_option_is_none", True),
        ("n5_option_unwrap", 7),
        ("n5_option_unwrap_or", 9),
        ("n5_option_unwrap_or_else", 11),
        ("n5_option_unwrap_or_else_some", 13),
        ("n5_option_and_then_some", 5),
        ("n5_option_and_then_none", 4),
        ("n5_option_or_else_some", 8),
        ("n5_option_or_else_none", 10),
        ("n5_option_filter_some", 6),
        ("n5_option_filter_none", 0),
        ("n5_option_ok_or_some", True),
        ("n5_option_ok_or_none", True),
        ("n5_result_ok_to_option", 7),
        ("n5_result_err_to_option", True),
    ],
)
def test_python_option_methods_parity(name, expected):
    assert _python_ns(name)[name]() == expected


def test_python_result_predicates():
    source = """
fn n5_result_is_ok() -> bool { Ok(7).is_ok() }
fn n5_result_is_err() -> bool { Err("failure").is_err() }
"""
    namespace = compile_rust(parse(lex(source)))
    assert namespace["n5_result_is_ok"]() is True
    assert namespace["n5_result_is_err"]() is True


def test_python_result_conversions_and_combinators():
    source = """
fn n5_result_ok_to_option() -> i64 { Ok(7).ok().unwrap_or(0) }
fn n5_result_err_to_option() -> bool { Err("failure").err().is_some() }
fn n5_result_and_then() -> i64 {
    Ok(3).and_then(|value| Ok(value + 2)).unwrap_or(0)
}
fn n5_result_or_else() -> i64 {
    Err("failure").or_else(|error| Ok(11)).unwrap_or(0)
}
"""
    namespace = compile_rust(parse(lex(source)))
    assert namespace["n5_result_ok_to_option"]() == 7
    assert namespace["n5_result_err_to_option"]() is True
    assert namespace["n5_result_and_then"]() == 5
    assert namespace["n5_result_or_else"]() == 11


def test_python_result_try_operator():
    namespace = _python_ns("n5_parse_pair")
    assert namespace["n5_parse_pair"]("21").unwrap() == 42
    result = namespace["n5_parse_pair"]("not a number")
    assert result.is_err()


def test_python_if_let_some_and_none_parity():
    namespace = _python_ns("n5_if_let_some")
    assert namespace["n5_if_let_some"]() == 8
    namespace = _python_ns("n5_if_let_none")
    assert namespace["n5_if_let_none"]() == 0


def test_option_result_match_warn_once_per_match():
    source = """
fn n5_match_option(value: Option<i64>) -> i64 {
    match value { Some(number) => number }
}
fn n5_match_result(value: Result<i64, String>) -> i64 {
    match value { Ok(number) => number }
}
fn n5_match_option_complete(value: Option<i64>) -> i64 {
    match value { Some(number) => number, None => 0 }
}
fn n5_match_result_complete(value: Result<i64, String>) -> i64 {
    match value { Ok(number) => number, Err(_) => 0 }
}
enum N5CustomOption { Some(i64), None }
fn n5_match_custom_enum(value: N5CustomOption) -> i64 {
    match value { N5CustomOption::Some(number) => number }
}
"""
    program = parse(lex(source))
    matches = [
        item.body.tail
        for item in program.items
        if (
            getattr(item, "body", None) is not None
            and isinstance(item.body.tail, MatchExpr)
        )
    ]
    assert len(matches) == 5
    with pytest.warns(UserWarning) as recorded:
        from backends.python.translator import Translator
        Translator().translate_program(program)
    assert len(recorded) == 2
    assert "missing None arm" in str(recorded[0].message)
    assert "missing Err arm" in str(recorded[1].message)
    with pytest.warns(UserWarning) as recorded:
        from backends.rust.codegen import RustCodegen
        RustCodegen().generate(program)
    assert len(recorded) == 2


@pytest.mark.skip(
    reason="Native Result and_then/or_else need typed Err values in let bindings",
)
def test_native_result_and_then_or_else_requires_typed_err():
    pass


@pytest.mark.skip(
    reason="Python has no Option-returning Rust-style iterator next() adapter",
)
def test_n5_while_let_option_iterator_not_supported():
    pass
