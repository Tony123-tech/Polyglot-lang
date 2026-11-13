"""
Backend registry for Polyglot.

Public API:
    get_backend(name="python" | "native", *, trusted=None, target_width=None)
        -> Backend
        Return a backend. Call backend.compile(program) to get a
        namespace dict.

    Backend
        Protocol with .name, .trusted, and .compile(program).

    UntrustedCompileError
        Raised when a backend refuses untrusted input.

Example:
    from backends import get_backend
    from lexer import lex
    from parser import parse

    program = parse(lex("fn add(a: i64, b: i64) -> i64 { a + b }"))
    ns = get_backend("python", trusted=True).compile(program)
    assert ns["add"](2, 3) == 5
"""

from backends.base import (
    Backend,
    UntrustedCompileError,
    get_backend,
)

__all__ = ["Backend", "UntrustedCompileError", "get_backend"]
