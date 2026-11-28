"""Structured diagnostics shared by the experimental pillar modules.

The project's public error surface is E001–E009.  The experimental modules
reuse those codes rather than inventing a parallel namespace, so a caller can
keep one diagnostic pipeline.  The mapping is deliberate:

E001  type / layout / ABI boundary
E002  semantic proof or termination obligation
E003  translation, AST, or API misuse
E004  compiler / native toolchain failure
E005  dependency cycle
E006  missing module, snapshot, or artifact
E007  unsupported native pattern or unavailable platform feature
E008  malformed serialized state or literal encoding
E009  ownership, move, drop, or borrow violation
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# A read-only mapping: extending it at runtime would silently widen the public
# diagnostic contract, so the exact E001-E009 set is enforced here.
ERROR_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "E001": "type, layout, or ABI boundary",
        "E002": "semantic proof or termination obligation",
        "E003": "translation, AST, or API misuse",
        "E004": "compiler or native toolchain failure",
        "E005": "dependency cycle",
        "E006": "missing module, snapshot, or artifact",
        "E007": "unsupported native pattern or platform feature",
        "E008": "malformed serialized state or literal encoding",
        "E009": "ownership, move, drop, or borrow violation",
    }
)


@dataclass(frozen=True)
class Diagnostic:
    """A location-aware diagnostic payload."""

    code: str
    message: str
    reason: str
    help_text: str
    note: str
    line: int = 0
    col: int = 0
    span: tuple[int, int] | None = None

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{self.code}]: {self.message}\n"
            f"  reason: {self.reason}\n"
            f"  help: {self.help_text}\n"
            f"  note: {self.note}\n"
            f"  {at}"
        )


def _normalize_span(span: tuple[int, int] | None) -> tuple[int, int] | None:
    """Coerce a span to ints, or drop a malformed one.

    Formatting a diagnostic must never raise because a caller supplied a bad
    span; a location-less error is strictly better than no error at all.
    """
    if span is None:
        return None
    if not isinstance(span, (tuple, list)) or len(span) != 2:
        return None
    try:
        return (int(span[0]), int(span[1]))
    except TypeError, ValueError:
        return None


class PillarError(Exception):
    """An experimental-pillar failure with the toolchain's error format."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        reason: str | None = None,
        help_text: str | None = None,
        note: str | None = None,
        line: int = 0,
        col: int = 0,
        span: tuple[int, int] | None = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown diagnostic code {code!r}")
        self.diagnostic = Diagnostic(
            code=code,
            message=message,
            reason=reason or message,
            help_text=help_text or _DEFAULT_HELP.get(code, "fix the reported error"),
            note=note or self.__class__.__name__,
            line=line,
            col=col,
            span=_normalize_span(span),
        )
        super().__init__(self.diagnostic.format())

    @property
    def code(self) -> str:
        return self.diagnostic.code

    @property
    def line(self) -> int:
        return self.diagnostic.line

    @property
    def col(self) -> int:
        return self.diagnostic.col

    @property
    def span(self) -> tuple[int, int] | None:
        return self.diagnostic.span

    @property
    def reason(self) -> str:
        return self.diagnostic.reason

    @property
    def help_text(self) -> str:
        return self.diagnostic.help_text

    def pretty(self) -> str:
        return self.diagnostic.format()

    def format(self) -> str:
        return self.diagnostic.format()

    def __str__(self) -> str:
        return self.diagnostic.format()


_DEFAULT_HELP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "E001": "make the old and new layouts ABI-compatible, or supply a migration default",
        "E002": "strengthen the specification or the synthesized candidate",
        "E003": "check the AST shape, version, and call sequence",
        "E004": "inspect the compiler/toolchain output and retry in a sandbox",
        "E005": "break the dependency cycle before patching or expanding",
        "E006": "check that the referenced module, snapshot, or artifact exists",
        "E007": "use the Python fallback on this platform",
        "E008": "rebuild the snapshot/certificate and verify its digest",
        "E009": "clone the value or end the borrow before the move",
    }
)


__all__ = ["ERROR_CODES", "Diagnostic", "PillarError"]
