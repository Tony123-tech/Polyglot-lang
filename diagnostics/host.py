"""Host interpreter requirement for the Python backend.

The language declares a single host interpreter floor (SPEC 9.9). This module
holds that floor in one place and turns an unsupported interpreter into a
readable refusal instead of an internal traceback:

* :data:`MIN_VERSION` is the floor the packaging metadata and the spec both
  name, ``3.14.8``;
* :func:`require_supported_python` raises
  :class:`UnsupportedInterpreterError` on anything older, carrying the two
  versions and the reason;
* the CLI calls it before it parses arguments, so an unsupported interpreter
  never reaches the lexer, the backends, or a decorator that would fail with
  a confusing ``NameError``.

Why 3.14: the front end relies on PEP 649 deferred annotation evaluation, which
is the default only from Python 3.14. On 3.12 the same test module failed with
``NameError: name 'i64' is not defined`` because annotations were evaluated
eagerly and the framework decorators received strings where they expected
types. That is a host-behaviour difference the language does not paper over.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

__all__ = [
    "MIN_LABEL",
    "MIN_VERSION",
    "REASON",
    "UnsupportedInterpreterError",
    "is_supported",
    "require_supported_python",
]

#: Interpreter floor, kept in sync with ``pyproject.toml``'s
#: ``requires-python = ">=3.14.8"``. A test asserts the two agree.
MIN_VERSION = (3, 14, 8)

MIN_LABEL = "Python " + ".".join(str(part) for part in MIN_VERSION)

REASON = (
    "the front end relies on PEP 649 deferred annotation evaluation, which is "
    "the default only from Python 3.14; on older interpreters annotations are "
    "evaluated eagerly and decorator-based tooling sees strings instead of "
    "types"
)


def _normalise(version_info: Sequence[int]) -> tuple[int, int, int]:
    """Take a ``sys.version_info``-shaped value and return a 3-tuple."""
    if len(version_info) < 2:
        raise ValueError("version_info needs at least major and minor parts")
    parts = tuple(int(part) for part in version_info[:3])
    while len(parts) < 3:
        parts += (0,)
    return parts


class UnsupportedInterpreterError(Exception):
    """The host interpreter is older than :data:`MIN_VERSION`."""

    def __init__(self, actual: tuple[int, int, int]) -> None:
        self.actual = actual
        self.actual_label = "Python " + ".".join(str(part) for part in actual)
        self.message = (
            f"polyglot-lang requires {MIN_LABEL} or later; "
            f"this interpreter is {self.actual_label}"
        )
        self.help_text = (
            f"{REASON}. Run it on {MIN_LABEL} or later, or use a different "
            "interpreter (see README, 'Host interpreter')"
        )
        super().__init__(self.message)


def is_supported(version_info: Sequence[int] | None = None) -> bool:
    """Is this interpreter at or above the declared floor?"""
    actual = _normalise(sys.version_info if version_info is None else version_info)
    return actual >= MIN_VERSION


def require_supported_python(version_info: Sequence[int] | None = None) -> None:
    """Raise :class:`UnsupportedInterpreterError` on an older interpreter.

    Pass ``version_info`` to check a candidate interpreter rather than the
    running one; that is what the tests do.
    """
    actual = _normalise(sys.version_info if version_info is None else version_info)
    if actual < MIN_VERSION:
        raise UnsupportedInterpreterError(actual)
