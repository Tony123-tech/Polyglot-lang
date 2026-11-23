"""Python lowering of Rust's ``assert!`` macro.

The Python backend used to emit a bare ``assert()`` call with no arguments for
``assert!(condition)``, which is not valid Python and silently disabled the
check. These tests pin the current lowering to the ``_assert`` runtime helper
with the condition and optional format arguments.
"""

from __future__ import annotations

import pytest

from backends import get_backend
from backends.python.translator import RuntimeFailure
from compiler.core.lexer import lex
from compiler.core.parser import parse


def _run(source: str):
    namespace = get_backend("python", trusted=True).compile(parse(lex(source)))
    return namespace["main"]()


def test_assert_macro_passes_when_the_condition_holds():
    assert (
        _run(
            """
fn main() -> i64 {
    let x = 3;
    assert!(x == 3);
    x
}
"""
        )
        == 3
    )


def test_assert_macro_raises_when_the_condition_fails():
    with pytest.raises(RuntimeFailure):
        _run(
            """
fn main() -> i64 {
    let x = 3;
    assert!(x == 4);
    x
}
"""
        )


def test_assert_macro_formats_the_message():
    with pytest.raises(RuntimeFailure, match="x was 3"):
        _run(
            """
fn main() -> i64 {
    let x = 3;
    assert!(x > 10, "x was {}", x);
    x
}
"""
        )


def test_assert_macro_accepts_a_literal_message():
    with pytest.raises(RuntimeFailure, match="unreachable"):
        _run(
            """
fn main() -> i64 {
    assert!(1 == 2, "unreachable");
    0
}
"""
        )


@pytest.mark.slow
def test_assert_macro_native_emits_rust_assert(native_backend):
    namespace = native_backend.compile(
        parse(
            lex(
                """
fn main() -> i64 {
    let x = 3;
    assert!(x == 3);
    x
}
"""
            )
        )
    )
    assert namespace["main"]() == 3
