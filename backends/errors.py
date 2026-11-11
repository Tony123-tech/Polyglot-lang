"""Shared backend errors."""

from __future__ import annotations


class TranslateError(Exception):
    """Raised when an AST node cannot be translated to the target backend."""

    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(f"{msg} at {line}:{col}")
        self.msg = msg
        self.line = line
        self.col = col


class UnsupportedNativePatternError(TranslateError):
    """Raised when a source pattern cannot be represented by stable Rust."""


class InvalidIntegerParseLiteralError(Exception):
    """Raised when a literal receiver cannot parse as a native integer."""

    def __init__(self, value: str, line: int, col: int) -> None:
        super().__init__(value)
        self.value = value
        self.line = line
        self.col = col
