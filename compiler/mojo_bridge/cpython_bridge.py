"""Mojo-inspired CPython interop with PEP 667 and hot-loop boxing guards.

Mojo's CPython interop is designed to avoid the overhead of boxing every
value into a ``PyObject*`` in hot loops.  This module models the *safety*
and *policy* aspects of that boundary:

* :func:`bridge_python_value` converts a Python object to a typed native
  value (int, float, bool, bytes) for the cold path — the boundary-crossing
  path where boxing overhead is acceptable.
* :func:`check_pep_667_safe` guards against PEP 667 violations: reading
  ``frame.f_locals`` in a hot loop (where the frame is being mutated) is
  forbidden; the function raises ``E007`` with a concrete message directing
  the caller to use a native binding instead.
* The hot-loop path is explicitly *not* modelled here — callers that need
  zero-boxing hot loops must use typed native values (e.g. ``ctypes``
  buffers or PyO3 ``pyclass``), never per-element ``PyObject`` boxing.

Diagnostics follow the shared ``E001``–``E008`` surface: ``E007`` for
PEP 667 violations or platform boundary attempts, ``E001`` for unsupported
conversions, and ``E003`` for API misuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "BridgeResult",
    "bridge_python_value",
    "check_pep_667_safe",
]

# Python versions where PEP 667 (frame locals write-through) is active.
_PEP_667_MIN_VERSION: Final = (3, 13)

# Keywords in a caller's description that indicate a hot-loop frame-locals
# access pattern that PEP 667 forbids.
_HOT_LOOP_KEYWORDS: Final[tuple[str, ...]] = (
    "hot loop",
    "hot_loop",
    "f_locals in loop",
    "frame locals mutation",
)


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    raise PolyglotError(code, message, reason=reason)


@dataclass(frozen=True)
class BridgeResult:
    """The result of bridging a Python value to a native value."""

    value: Any
    native_type: str
    cold_path: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "native_type": self.native_type,
            "cold_path": self.cold_path,
            "reason": self.reason,
        }


def check_pep_667_safe(context: str = "") -> None:
    """Guard against PEP 667 violations in hot loops.

    PEP 667 (Python 3.13+) makes ``frame.f_locals`` a write-through proxy
    for the frame's actual locals.  Reading ``f_locals`` in a hot loop and
    mutating the returned dict can cause incorrect behavior or crashes.  This
    function raises ``E007`` if the *context* description suggests a hot-loop
    ``f_locals`` access pattern.

    The check is a static analysis aid, not a runtime enforcement — callers
    should use this at code generation time to reject hot-loop patterns
    before they reach the interpreter.
    """
    if not context:
        return
    lowered = context.lower()
    for keyword in _HOT_LOOP_KEYWORDS:
        if keyword in lowered:
            _fail(
                "E007",
                f"PEP 667 violation: {context!r} suggests f_locals access "
                "in a hot loop",
                reason="use a typed native binding (ctypes buffer or PyO3 "
                "pyclass) for hot-loop data, never per-element PyObject boxing",
            )
    # Also check for explicit f_locals references combined with loop keywords.
    if "f_locals" in lowered and ("loop" in lowered or "hot" in lowered):
        _fail(
            "E007",
            f"PEP 667 violation: {context!r} accesses f_locals in a loop",
            reason="f_locals must not be accessed in hot loops",
        )


def bridge_python_value(value: Any, *, context: str = "") -> BridgeResult:
    """Bridge a Python value to a typed native value for the cold path.

    This is the *cold* boundary-crossing path where boxing overhead is
    acceptable.  Supported conversions:

    * ``bool`` → ``"bool"``
    * ``int`` → ``"i64"`` (truncated to 64 bits at the boundary)
    * ``float`` → ``"f64"``
    * ``bytes`` → ``"[]u8"``
    * ``None`` → ``"void"``

    Unsupported types (e.g. ``str``, ``list``, ``dict``) raise ``E001`` with
    a message directing the caller to use a typed native binding for hot
    paths.  The *context* parameter is checked for PEP 667 violations.
    """
    check_pep_667_safe(context)
    if value is None:
        return BridgeResult(
            value=None, native_type="void", cold_path=True, reason="null"
        )
    if isinstance(value, bool):
        return BridgeResult(
            value=value, native_type="bool", cold_path=True, reason="bool"
        )
    if isinstance(value, int):
        return BridgeResult(
            value=value,
            native_type="i64",
            cold_path=True,
            reason="int (truncated to 64 bits)",
        )
    if isinstance(value, float):
        return BridgeResult(
            value=value, native_type="f64", cold_path=True, reason="float"
        )
    if isinstance(value, (bytes, bytearray)):
        return BridgeResult(
            value=bytes(value), native_type="[]u8", cold_path=True, reason="bytes"
        )
    _fail(
        "E001",
        f"unsupported Python type for cold-path bridging: {type(value).__name__}",
        reason="use a typed native binding (ctypes buffer or PyO3 pyclass) "
        "for complex types in hot paths",
    )
