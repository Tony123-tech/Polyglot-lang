"""Recoverable lexing, shared by :mod:`compiler.frontend.lexing` and ``framework``.

Both frontend modules drive the real :class:`compiler.core.lexer.Lexer` one token
at a time so they can turn a lexical failure into a value instead of an exception.
Each kept its own copy of the dispatch step, the recovery boundary and the
catchable error shape; the copies were identical apart from the exception class
they raised. They live here now, so there is one dispatch table and one failure
shape to keep in step with ``Lexer.lex``.
"""

from __future__ import annotations

from typing import NoReturn

from compiler.core.lexer import Lexer
from compiler.frontend.lexeme import raw_string_prefix


class RecoveryLexError(Exception):
    """A located lexer failure that callers can catch and keep lexing after."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


class RecoveringLexer(Lexer):
    """Real lexer whose ``error`` raises a catchable, located failure."""

    def error(  # type: ignore[override]
        self, msg: str, line: int | None = None, col: int | None = None
    ) -> NoReturn:
        raise RecoveryLexError(
            msg, self.line if line is None else line, self.col if col is None else col
        )


def dispatch_one(lexer: Lexer) -> None:
    """Run exactly one token dispatch step of ``Lexer.lex``."""
    line, col, pos = lexer.line, lexer.col, lexer.i
    ch = lexer.peek()
    if ch == "r" and raw_string_prefix(lexer.src, pos) is not None:
        lexer.lex_raw_string(line, col, pos)
    elif lexer.src.startswith('"""', pos):
        lexer.lex_multiline_string(line, col, pos)
    elif ch == '"':
        lexer.lex_string(line, col, pos)
    elif ch == "'":
        lexer.lex_quote(line, col, pos)
    elif ch.isalpha() or ch == "_":
        lexer.lex_ident(line, col, pos)
    elif ch.isdigit():
        lexer.lex_number(line, col, pos)
    elif lexer.lex_operator(line, col, pos):
        return
    else:
        raise RecoveryLexError(f"unexpected character {ch!r}", line, col)


def recovery_stop(source: str, start: int) -> int:
    """First newline, semicolon or brace at or after *start*."""
    candidates = [
        found
        for found in (source.find(mark, start + 1) for mark in ("\n", ";", "}"))
        if found >= 0
    ]
    return min(candidates) if candidates else len(source)


__all__ = [
    "RecoveringLexer",
    "RecoveryLexError",
    "dispatch_one",
    "recovery_stop",
]
