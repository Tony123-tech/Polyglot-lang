"""Structured frontend diagnostics for the 1,000-function framework rollout.

The frontend frame raises :class:`PolyglotError`, a thin frontend-facing
specialization of :class:`runtime.diagnostics.PillarError`.  It therefore uses
the exact same ``E001``–``E009`` surface, formatting, and location payloads as
the rest of the toolchain while giving frontend callers a single, stable
exception type to catch.

Diagnostic mapping used by this package:

* ``E001`` – invalid lexical/layout input (bad identifier, bad escape, bad BOM);
* ``E002`` – semantic source obligation that cannot be satisfied;
* ``E003`` – framework API misuse or parse failure (wrong type, bad span, bad
  syntax);
* ``E004`` – frontend/toolchain failure;
* ``E005`` – dependency/cycle in source structure;
* ``E006`` – missing source artifact or token;
* ``E007`` – unsupported platform/model-only frontend operation;
* ``E008`` – malformed encoded literal (numeric or escape encoding).
"""

from __future__ import annotations

from runtime.diagnostics import ERROR_CODES, PillarError

__all__ = ["ERROR_CODES", "PolyglotError", "frontend_fail"]


class PolyglotError(PillarError):
    """A frontend failure carrying a structured ``E001``–``E009`` diagnostic.

    The constructor mirrors :class:`PillarError` and adds a ``phase`` label so
    diagnostics can be attributed to the frontend without changing their
    machine-readable payload.
    """

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
        phase: str = "frontend",
    ) -> None:
        super().__init__(
            code,
            message,
            reason=reason,
            help_text=help_text,
            note=note or f"PolyglotError/{phase}",
            line=line,
            col=col,
            span=span,
        )
        self.phase = phase


def frontend_fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    note: str | None = None,
    line: int = 0,
    col: int = 0,
    span: tuple[int, int] | None = None,
    phase: str = "frontend",
) -> None:
    """Raise :class:`PolyglotError` with frontend defaults.

    Kept as a function (not a ``NoReturn`` alias) so call sites read clearly and
    type checkers continue to understand the control flow at each raise point.
    """
    raise PolyglotError(
        code,
        message,
        reason=reason,
        help_text=help_text,
        note=note,
        line=line,
        col=col,
        span=span,
        phase=phase,
    )
