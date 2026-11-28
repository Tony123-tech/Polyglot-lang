"""Warnings for control flow that can never execute."""

import warnings

from backends.python.translator import compile_rust
from compiler.core.lexer import lex
from compiler.core.parser import parse

UNREACHABLE = """\
fn dead_after_return(value: i64) -> i64 {
    return value;
    let later = 1;
    later
}
"""

REACHABLE = """\
fn no_dead_code(value: i64) -> i64 {
    let doubled = value * 2;
    doubled
}
"""


def _dead_warnings(source: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_rust(parse(lex(source)))
    return [
        item for item in caught if "unreachable code after return" in str(item.message)
    ]


def test_unreachable_statement_after_return_warns_once():
    found = _dead_warnings(UNREACHABLE)

    assert len(found) == 1
    assert "unreachable code after return at 3:5" in str(found[0].message)


def test_plain_function_does_not_warn():
    assert _dead_warnings(REACHABLE) == []
