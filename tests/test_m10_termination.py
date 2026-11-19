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
    assert "  note: this function is annotated #[terminates]" in str(native_error.value)
    assert "  at: line " in str(native_error.value)


# ---------------------------------------------------------------------------
# Multiple self-recursive calls (SPEC 2.8.3 / 10.22)
# ---------------------------------------------------------------------------

FIB_SOURCE = """
#[terminates]
fn m10_fib(n: i64) -> i64 {
    if n > 1 {
        m10_fib(n - 1) + m10_fib(n - 2)
    } else {
        n
    }
}
"""

DIFFERENT_PARAMETERS_SOURCE = """
#[terminates]
fn m10_pair_step(n: i64, m: i64) -> i64 {
    if n > 0 && m > 0 {
        m10_pair_step(n - 1, m) + m10_pair_step(n, m - 1)
    } else {
        n + m
    }
}
"""

ONE_CONSTANT_SOURCE = """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    if n > 0 {
        m10_bad(n - 1) + m10_bad(n)
    } else {
        0
    }
}
"""

MULTI_RECURSION_REJECT_SOURCES = {
    "call_outside_guard": """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    m10_bad(n - 1)
}
""",
    "guard_only_one_branch": """
#[terminates]
fn m10_bad(n: i64, m: i64) -> i64 {
    if n > 0 {
        m10_bad(n - 1, m)
    } else {
        m10_bad(n, m - 1)
    }
}
""",
    "one_decreasing_one_constant": ONE_CONSTANT_SOURCE,
    "mutual_recursion": """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    if n > 0 { m10_other(n - 1) } else { 0 }
}

#[terminates]
fn m10_other(n: i64) -> i64 {
    if n > 0 { m10_bad(n - 1) } else { 0 }
}
""",
    "mutated_parameter_with_multiple_calls": """
#[terminates]
fn m10_bad(mut n: i64, m: i64) -> i64 {
    if n > 0 && m > 0 {
        n = n - 1;
        m10_bad(n, m) + m10_bad(n - 1, m)
    } else {
        0
    }
}
""",
    "shadowed_parameter_with_multiple_calls": """
#[terminates]
fn m10_bad(n: i64, m: i64) -> i64 {
    if n > 0 && m > 0 {
        let n = n - 1;
        m10_bad(n, m) + m10_bad(n - 1, m)
    } else {
        0
    }
}
""",
    "shadowed_function_name_with_multiple_calls": """
#[terminates]
fn m10_bad(n: i64, m: i64) -> i64 {
    if n > 0 && m > 0 {
        let m10_bad = n - 1;
        m10_bad(n, m) + m10_bad(n - 1, m)
    } else {
        0
    }
}
""",
    "non_recursive_call_with_multiple_calls": """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    if n > 0 { m10_helper(n - 1) + m10_bad(n - 1) } else { 0 }
}

fn m10_helper(n: i64) -> i64 { n }
""",
    "guard_encloses_only_nested_function": """
#[terminates]
fn m10_bad(n: i64) -> i64 {
    if n > 0 {
        fn m10_inner() -> i64 { m10_bad(n - 1) }
        0
    } else {
        0
    }
}
""",
}


def test_python_multiple_self_calls_under_lower_bound_are_accepted():
    check_function(parse(lex(FIB_SOURCE)).items[0])
    assert _run_python(FIB_SOURCE)["m10_fib"](10) == 55


def test_python_multiple_calls_on_different_decreasing_parameters_are_accepted():
    check_function(parse(lex(DIFFERENT_PARAMETERS_SOURCE)).items[0])
    assert _run_python(DIFFERENT_PARAMETERS_SOURCE)["m10_pair_step"](1, 1) == 2


@pytest.mark.parametrize("name", sorted(MULTI_RECURSION_REJECT_SOURCES))
def test_python_multi_call_rejections(name: str):
    with pytest.raises(TerminationError):
        _run_python(MULTI_RECURSION_REJECT_SOURCES[name])


@pytest.mark.parametrize("name", sorted(MULTI_RECURSION_REJECT_SOURCES))
def test_native_multi_call_rejections(name: str, tmp_path):
    from backends.rust.compiler import NativeBackend

    source = MULTI_RECURSION_REJECT_SOURCES[name]
    with pytest.raises(TerminationError):
        NativeBackend(cache_dir=tmp_path).compile(parse(lex(source)))


@pytest.mark.parametrize(
    "name",
    [
        "call_outside_guard",
        "guard_only_one_branch",
        "one_decreasing_one_constant",
        "mutual_recursion",
        "mutated_parameter_with_multiple_calls",
    ],
)
def test_multi_call_error_message_is_backend_identical(name: str):
    from backends.rust.compiler import NativeBackend

    source = MULTI_RECURSION_REJECT_SOURCES[name]
    with pytest.raises(TerminationError) as python_error:
        _run_python(source)
    with pytest.raises(TerminationError) as native_error:
        NativeBackend().compile(parse(lex(source)))
    assert native_error.value.format() == python_error.value.format()
    assert str(native_error.value) == str(python_error.value)


def test_mutated_parameter_with_multiple_calls_reports_shadowing_reason():
    with pytest.raises(TerminationError) as error:
        check_function(
            parse(
                lex(
                    MULTI_RECURSION_REJECT_SOURCES[
                        "mutated_parameter_with_multiple_calls"
                    ]
                )
            ).items[0]
        )
    assert error.value.reason == "a recursive parameter is reassigned or shadowed"


def test_mutual_recursion_reports_the_non_recursive_call():
    with pytest.raises(TerminationError) as error:
        check_function(
            parse(lex(MULTI_RECURSION_REJECT_SOURCES["mutual_recursion"])).items[0]
        )
    assert error.value.reason == (
        "a non-recursive function call is outside the supported proof subset"
    )


def test_e002_decrease_failure_points_at_the_offending_call_site():
    function = parse(lex(ONE_CONSTANT_SOURCE)).items[0]
    with pytest.raises(TerminationError) as error:
        check_function(function)

    termination_error = error.value
    assert termination_error.code == "E002"
    assert (
        termination_error.reason
        == "a recursive call has no provably decreasing argument"
    )
    # The second call, `m10_bad(n)`, is the one that does not decrease.
    assert termination_error.span == (83, 92)
    assert termination_error.line == 5
    assert termination_error.col == 33
    assert (
        ONE_CONSTANT_SOURCE[termination_error.span[0] : termination_error.span[1]]
        == "m10_bad(n"
    )
    assert termination_error.span != function.span
    message = str(termination_error)
    assert "at: line 5, column 33 (span 83..92)" in message
