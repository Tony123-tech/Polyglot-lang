"""M2: compiled Python programs must not clobber each other in sys.modules."""

from __future__ import annotations

import sys

from backends.python.translator import compile_rust
from lexer import lex
from parser import parse


def _compile(src: str) -> dict:
    return compile_rust(parse(lex(src)))


def test_consecutive_compilations_use_distinct_modules():
    first = _compile("struct First { value: i64 }\nfn first() -> i64 { 1 }")
    second = _compile(
        "struct Second { value: i64 }\nfn second() -> i64 { 2 }"
    )

    assert first["__name__"] != second["__name__"]
    assert first["First"].__module__ == first["__name__"]
    assert second["Second"].__module__ == second["__name__"]
    assert first["__name__"] in sys.modules
    assert second["__name__"] in sys.modules


def test_first_compilation_stays_usable_after_second():
    first = _compile("struct Keep { value: i64 }\nfn keep() -> i64 { 7 }")
    _compile("struct Other { value: i64 }\nfn other() -> i64 { 8 }")

    instance = first["Keep"](5)
    assert instance.value == 5
    assert first["keep"]() == 7
    # The dataclass was resolved against its own module, not the newer one.
    assert sys.modules[first["__name__"]].Keep is first["Keep"]
