"""Shared backend errors."""

from __future__ import annotations


class TranslateError(Exception):
    """Raised when an AST node cannot be translated to the target backend."""

    code = "E003"

    def __init__(
        self,
        msg: str,
        line: int = 0,
        col: int = 0,
        span: tuple[int, int] | None = None,
    ):
        self.msg = msg
        self.line = line
        self.col = col
        self.span = span
        super().__init__(self.format())

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{self.code}]: {self.msg}\n"
            f"  reason: {self.msg}\n"
            f"  help: fix the translation error and try again\n"
            f"  note: {self.__class__.__name__}\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()


# Arity table for the compiler-reserved std:fs host seam. Both backends use
# this table so malformed calls produce the same E003 diagnostic and span.
HOST_CALL_ARITIES = {
    "__fs_read_text": 1,
    "__fs_write_text": 2,
    "__fs_append_text": 2,
    "__fs_exists": 1,
    "__fs_remove": 1,
}


def check_host_call_arity(name: str, args, *, line: int, col: int, span=None) -> None:
    """Raise a span-aware E003 error for a malformed host-seam call."""
    expected = HOST_CALL_ARITIES.get(name)
    if expected is None or len(args) == expected:
        return
    plural = "" if expected == 1 else "s"
    raise TranslateError(
        f"{name} expects {expected} argument{plural}, got {len(args)}",
        line,
        col,
        span,
    )


class UnsupportedNativePatternError(TranslateError):
    """Raised when a source pattern cannot be represented by stable Rust."""

    code = "E007"


class InvalidIntegerParseLiteralError(Exception):
    """Raised when a literal receiver cannot parse as a native integer."""

    code = "E008"

    def __init__(
        self,
        value: str,
        line: int,
        col: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.value = value
        self.line = line
        self.col = col
        self.span = span
        super().__init__(self.format())

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        msg = f"invalid integer parse literal: {self.value}"
        return (
            f"error[{self.code}]: {msg}\n"
            f"  reason: {msg}\n"
            f"  help: provide a valid integer literal\n"
            f"  note: {self.__class__.__name__}\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()
