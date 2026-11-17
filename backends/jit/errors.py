"""Exceptions raised by the Apple Silicon JIT backend."""

from __future__ import annotations


class JITError(Exception):
    """Base class for all JIT backend errors."""


class JITUnsupportedError(JITError):
    """The source uses a construct the JIT backend does not lower yet."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        col: int | None = None,
    ) -> None:
        self.line = line
        self.col = col
        if line is not None:
            message = f"{message} (line {line}, column {col})"
        super().__init__(message)


class JITCompileError(JITError):
    """The source is valid Polyglot but cannot be encoded by the JIT."""


class JITRuntimeError(JITError):
    """A machine-code trap raised at the Python/JIT boundary."""

    code: int = 0

    def __init__(self, message: str, code: int = 0) -> None:
        self.code = code
        super().__init__(message)
