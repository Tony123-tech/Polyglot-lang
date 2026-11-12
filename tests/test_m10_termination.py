# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""M10: conservative checking for #[terminates] functions."""

from __future__ import annotations

import pytest

from lexer import lex
from parser import parse
from termination import TerminationError, check_function

SOURCES = {
    "m10_terminating_functions": """
#[terminates]
fn m10_gcd(a: i64, b: i64) -> i64 {
    if b > 0 {
        m10_gcd(a, b - 1)
    } else {
        a
    }
}

#[terminates]
fn m10_sum(xs: Vec<i64>) -> i64 {
    if xs.is_empty() {
        0
    } else {
        xs[0] + m10_sum(xs[1..].to_vec())
    }
}

#[terminates]
fn m10_countdown(mut n: i64) -> i64 {
    while n > 0 {
        n = n - 1;
    }
    n
}

#[terminates]
fn m10_bounded_for(n: i64) {
    for i in 0..n {};
}

fn m10_unannotated_recursion(n: i64) -> i64 {
    m10_unannotated_recursion(n)
}
""",
}


def _run_python(source: str) -> dict:
    from backends.python.translator import compile_rust

    return compile_rust(parse(lex(source)))


def test_terminates_attribute_is_parsed_on_and_off():
    annotated = parse(lex("#[terminates] fn m10_annotated() {}"))
    unannotated = parse(lex("fn m10_plain() {}"))
    assert annotated.items[0].terminates is True
    assert unannotated.items[0].terminates is False


def test_fn_decl_terminates_flag_is_preserved_in_ast():
    program = parse(lex("#[terminates] fn m10_ast_flag() {}"))
    assert program.items[0].terminates


def test_checker_ignores_unannotated_functions():
    program = parse(lex("fn m10_plain(n: i64) -> i64 { m10_plain(n) }"))
    check_function(program.items[0])


def test_python_terminating_patterns_execute():
    namespace = _run_python(SOURCES["m10_terminating_functions"])
    assert namespace["m10_gcd"](5, 3) == 5
    assert namespace["m10_sum"]([1, 2, 3, 4]) == 10
    assert namespace["m10_countdown"](4) == 0
    assert namespace["m10_bounded_for"](5) is None


@pytest.mark.native
def test_native_terminating_patterns_execute(native_ns):
    assert native_ns["m10_gcd"](5, 3) == 5
    assert native_ns["m10_sum"]([1, 2, 3, 4]) == 10
    assert native_ns["m10_countdown"](4) == 0
    assert native_ns["m10_bounded_for"](5) is None


def test_python_unannotated_recursion_compiles_without_running():
    namespace = _run_python(SOURCES["m10_terminating_functions"])
    assert callable(namespace["m10_unannotated_recursion"])


@pytest.mark.native
def test_native_unannotated_recursion_compiles_without_running(native_ns):
    assert callable(native_ns["m10_unannotated_recursion"])


NON_TERMINATING_SOURCES = [
    "fn m10_bad(n: i64) -> i64 { m10_bad(n) }",
    "fn m10_bad(n: i64) -> i64 { m10_bad(n + 1) }",
    """
#[terminates]
fn m10_bad(mut n: i64) {
    while n > 0 {
        n = n + 1;
    }
}
""",
    """
#[terminates]
fn m10_bad(mut n: i64) {
    while n > 0 {
        let mut n = 1;
        n = n - 1;
    }
}
""",
    """
#[terminates]
fn m10_bad() {
    loop {}
}
""",
    """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    if n > 0 {
        let n = n + 1;
        m10_bad(n - 1)
    } else {
        0
    }
}
""",
]


@pytest.mark.parametrize("source", NON_TERMINATING_SOURCES)
def test_python_non_terminating_patterns_rejected(source: str):
    if source.startswith("fn m10_bad"):
        source = "#[terminates] " + source
    with pytest.raises(TerminationError):
        _run_python(source)


@pytest.mark.parametrize("source", NON_TERMINATING_SOURCES)
def test_native_non_terminating_patterns_rejected(source: str, tmp_path):
    from backends.rust.compiler import NativeBackend

    if source.startswith("fn m10_bad"):
        source = "#[terminates] " + source
    with pytest.raises(TerminationError):
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))


def test_termination_error_message_is_backend_identical():
    from backends.rust.compiler import NativeBackend

    source = "#[terminates] fn m10_bad(n: i64) -> i64 { m10_bad(n) }"
    with pytest.raises(TerminationError) as python_error:
        _run_python(source)
    with pytest.raises(TerminationError) as native_error:
        NativeBackend().compile(parse(lex(source)))
    assert native_error.value.format() == python_error.value.format()
    assert str(native_error.value) == str(python_error.value)
    assert len(str(native_error.value).splitlines()) == 5
    assert "  reason: " in str(native_error.value)
    assert "  help: " in str(native_error.value)
    assert "  note: this function is annotated #[terminates]" in str(
        native_error.value
    )
    assert "  at: line " in str(native_error.value)
