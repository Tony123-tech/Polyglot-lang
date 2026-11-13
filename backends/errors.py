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
