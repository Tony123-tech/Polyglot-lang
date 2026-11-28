"""Deterministic runtime helpers for the Python backend framework rollout.

The frame helpers model PEP 667 frame-local semantics with plain dictionaries
and :class:`types.MappingProxyType` views.  A real :class:`types.FrameType` may
be *read* to produce a snapshot, but helpers that write state reject real
frames outright; mutating ``f_locals`` or debugger trace hooks is never part of
this module.  The remaining helpers implement Option/Result tagging, checked
integer arithmetic, assertions, panic dispatch, and tagged-enum encoding with
structured ``E001``-``E008`` diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping, Sequence
from types import FrameType, MappingProxyType
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "pyruntime_assert_eq",
    "pyruntime_assert_ne",
    "pyruntime_assert_true",
    "pyruntime_checked_add",
    "pyruntime_checked_mul",
    "pyruntime_enum_decode",
    "pyruntime_enum_encode",
    "pyruntime_frame_apply_proxied",
    "pyruntime_frame_identity",
    "pyruntime_frame_model_check",
    "pyruntime_frame_proxy_view",
    "pyruntime_frame_restore_plan",
    "pyruntime_frame_snapshot",
    "pyruntime_frame_trace_flags",
    "pyruntime_int_cast",
    "pyruntime_int_div",
    "pyruntime_int_rem",
    "pyruntime_match_exhaustiveness",
    "pyruntime_option_none",
    "pyruntime_option_some",
    "pyruntime_option_unwrap",
    "pyruntime_panic",
    "pyruntime_result_err",
    "pyruntime_result_ok",
    "pyruntime_result_unwrap",
]

_CODES: Final[frozenset[str]] = frozenset({f"E{index:03d}" for index in range(1, 10)})
_TYPE_PARTS: Final[dict[str, tuple[int, bool]]] = {
    "i8": (8, True),
    "i16": (16, True),
    "i32": (32, True),
    "i64": (64, True),
    "i128": (128, True),
    "isize": (64, True),
    "u8": (8, False),
    "u16": (16, False),
    "u32": (32, False),
    "u64": (64, False),
    "u128": (128, False),
    "usize": (64, False),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured runtime diagnostic."""
    raise PolyglotError(code, message)


def _text(value: object, label: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty str")
    return value


def _int(value: object, label: str) -> int:
    """Require a plain integer (``bool`` is not accepted)."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E001", f"{label} must be an int")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a string-keyed mapping and copy it into a plain dict."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E003", f"{label} keys must be str")
        result[key] = item
    return result


def _snapshot(snapshot: object) -> dict[str, Any]:
    """Validate and copy a modelled frame snapshot."""
    record = _mapping(snapshot, "snapshot")
    required = ("locals", "globals", "builtins", "lineno", "lasti", "trace", "code")
    if any(field not in record for field in required):
        _fail("E003", "snapshot is missing required PEP 667 fields")
    for field in ("locals", "globals", "builtins"):
        _mapping(record[field], field)
    if not isinstance(record["trace"], Mapping):
        _fail("E003", "snapshot trace must be a mapping")
    if not isinstance(record["code"], Mapping):
        _fail("E003", "snapshot code must be a mapping")
    return record


def _bounds(width: int, signed: bool) -> tuple[int, int]:
    """Return inclusive integer bounds for an ABI width."""
    if width not in (8, 16, 32, 64, 128):
        _fail("E001", f"unsupported integer width: {width}")
    if signed:
        return -(2 ** (width - 1)), 2 ** (width - 1) - 1
    return 0, 2**width - 1


def _tagged(value: object, expected: str) -> tuple[Any, str]:
    """Validate a tagged Option/Result pair and return payload and tag."""
    if isinstance(value, Mapping):
        tag = value.get("tag")
        payload = value.get("value")
        if tag not in ("Some", "None", "Ok", "Err"):
            _fail("E008", f"invalid runtime tag: {tag!r}")
    elif isinstance(value, tuple) and value:
        tag, *rest = value
        payload = rest[0] if len(rest) == 1 else tuple(rest)
        if tag not in ("Some", "None", "Ok", "Err"):
            _fail("E008", f"invalid runtime tag: {tag!r}")
    else:
        _fail("E008", "tagged runtime value must be a mapping or tuple")
    if tag != expected:
        _fail("E002", f"expected {expected}, got {tag}")
    return payload, tag


# ---------------------------------------------------------------------------
# Public runtime helpers
# ---------------------------------------------------------------------------


def pyruntime_frame_snapshot(
    locals_view: Mapping[str, Any] | FrameType | None = None,
    *,
    globals_view: Mapping[str, Any] | None = None,
    builtins_view: Mapping[str, Any] | None = None,
    lineno: int = 0,
    lasti: int = 0,
    trace_lines: bool = False,
    trace_opcodes: bool = False,
    code_name: str = "<module>",
    filename: str = "<string>",
) -> dict[str, Any]:
    """Snapshot frame state without writing to a real frame.

    A real :class:`~types.FrameType` is only read.  PEP 667 makes ``f_locals``
    a write-through proxy, so the copied ``locals`` dict is an isolated model:
    later changes to it never reach the frame.
    """
    if isinstance(locals_view, FrameType):
        frame = locals_view
        copied_locals = _mapping(frame.f_locals, "frame.f_locals")
        copied_globals = _mapping(frame.f_globals, "frame.f_globals")
        copied_builtins = _mapping(frame.f_builtins, "frame.f_builtins")
        line = frame.f_lineno
        offset = frame.f_lasti
        lines = bool(frame.f_trace_lines)
        opcodes = bool(frame.f_trace_opcodes)
        name = frame.f_code.co_name
        path = frame.f_code.co_filename
    else:
        copied_locals = _mapping(locals_view or {}, "locals")
        copied_globals = _mapping(globals_view or {}, "globals")
        copied_builtins = _mapping(builtins_view or {}, "builtins")
        line = _int(lineno, "lineno")
        offset = _int(lasti, "lasti")
        lines = bool(trace_lines)
        opcodes = bool(trace_opcodes)
        name = _text(code_name, "code_name")
        path = _text(filename, "filename")
    return {
        "locals": copied_locals,
        "globals": copied_globals,
        "builtins": copied_builtins,
        "lineno": line,
        "lasti": offset,
        "trace": {"lines": lines, "opcodes": opcodes},
        "code": {"name": name, "filename": path},
        "version": 1,
        "writethrough": True,
    }


def pyruntime_frame_proxy_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Expose a read-only mapping-proxy view of a modelled locals dict."""
    record = _snapshot(snapshot)
    locals_copy = dict(record["locals"])
    return {
        "view": MappingProxyType(locals_copy),
        "keys": sorted(locals_copy),
        "count": len(locals_copy),
        "writethrough": False,
    }


def pyruntime_frame_restore_plan(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Plan PEP 667 proxy writes and deletes between two frame snapshots."""
    before = _snapshot(source)
    after = _snapshot(target)
    old, new = before["locals"], after["locals"]
    writes = {
        key: value for key, value in new.items() if old.get(key, object()) != value
    }
    deletes = [key for key in old if key not in new]
    return {
        "writes": writes,
        "deletes": sorted(deletes),
        "unchanged": sorted(set(new) - set(writes)),
        "semantics": "PEP 667 proxy write-through (modelled)",
        "count": len(writes) + len(deletes),
        "trace_fields_changed": before["trace"] != after["trace"],
    }


def pyruntime_frame_apply_proxied(
    mapping: MutableMapping[str, Any],
    writes: Mapping[str, Any],
    *,
    deletes: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply a write/delete plan to a plain mapping that models frame locals.

    Real frames are rejected with ``E007``: this helper never mutates
    ``frame.f_locals``, even though PEP 667 proxies would forward writes.
    """
    if isinstance(mapping, FrameType):
        _fail("E007", "real frame mutation is unsupported; pass a plain mapping")
    if not isinstance(mapping, MutableMapping):
        _fail("E003", "mapping must be a mutable mapping")
    if isinstance(deletes, (str, bytes)) or not isinstance(deletes, Sequence):
        _fail("E003", "deletes must be a sequence of names")
    assignments = _mapping(writes, "writes")
    before = dict(mapping)
    missing = [name for name in deletes if name not in mapping]
    if missing:
        _fail("E006", f"cannot delete absent locals: {', '.join(sorted(missing))}")
    for name, value in assignments.items():
        mapping[name] = value
    for name in deletes:
        del mapping[name]
    return {
        "before": before,
        "after": dict(mapping),
        "written": sorted(assignments),
        "deleted": sorted(deletes),
        "proxy": True,
    }


def pyruntime_frame_trace_flags(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise trace flags from a modelled frame snapshot."""
    record = _snapshot(snapshot)
    flags = record["trace"]
    lines = bool(flags.get("lines", False))
    opcodes = bool(flags.get("opcodes", False))
    return {
        "lines": lines,
        "opcodes": opcodes,
        "trace_enabled": bool(snapshot.get("trace_enabled", False)),
        "mask": (1 if lines else 0) | (2 if opcodes else 0),
    }


def pyruntime_frame_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a stable identity for a modelled frame snapshot."""
    record = _snapshot(snapshot)
    code = record["code"]
    components = {
        "name": _text(code.get("name", "<module>"), "code name"),
        "filename": _text(code.get("filename", "<string>"), "filename"),
        "lineno": _int(record["lineno"], "lineno"),
        "lasti": _int(record["lasti"], "lasti"),
    }
    payload = json.dumps(components, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "id": hashlib.sha256(payload).hexdigest()[:16],
        "components": components,
        "digest_length": 16,
    }


def pyruntime_frame_model_check(value: object) -> dict[str, Any]:
    """Check that *value* is a safe modelled frame state, not a real frame."""
    if isinstance(value, FrameType):
        _fail("E007", "real frames cannot be used as writable runtime state")
    record = _snapshot(value)
    warnings: list[str] = []
    if "writethrough" not in record:
        warnings.append("missing writethrough marker")
    if "version" not in record:
        warnings.append("missing model version")
    return {
        "valid": True,
        "keys": sorted(record),
        "frame_type": False,
        "warnings": warnings,
    }


def pyruntime_option_some(value: Any) -> dict[str, Any]:
    """Build a tagged ``Some`` option record."""
    return {
        "tag": "Some",
        "value": value,
        "encoded": ("Some", value),
        "is_some": True,
    }


def pyruntime_option_none() -> dict[str, Any]:
    """Build the tagged ``None`` option record."""
    return {"tag": "None", "value": None, "encoded": ("None",), "is_some": False}


def pyruntime_option_unwrap(option: Any) -> dict[str, Any]:
    """Unwrap a ``Some`` option, raising ``E002`` for ``None`` options."""
    payload, tag = _tagged(option, "Some")
    return {"value": payload, "tag": tag, "is_some": True}


def pyruntime_result_ok(value: Any) -> dict[str, Any]:
    """Build a tagged ``Ok`` result record."""
    return {"tag": "Ok", "value": value, "ok": True, "encoded": ("Ok", value)}


def pyruntime_result_err(error: Any) -> dict[str, Any]:
    """Build a tagged ``Err`` result record with a string payload."""
    message = str(error)
    return {"tag": "Err", "value": message, "ok": False, "encoded": ("Err", message)}


def pyruntime_result_unwrap(result: Any) -> dict[str, Any]:
    """Unwrap an ``Ok`` result, exposing ``Err`` payloads as ``E002`` failures."""
    payload, tag = _tagged(result, "Ok")
    return {"value": payload, "tag": tag, "ok": True}


def pyruntime_int_cast(
    value: Any,
    target: str,
    *,
    wrap: bool = False,
) -> dict[str, Any]:
    """Convert an integer to a checked ABI width such as ``i32`` or ``u8``."""
    number = _int(value, "value")
    type_name = _text(target, "target")
    if type_name not in _TYPE_PARTS:
        _fail("E001", f"unsupported integer type: {type_name}")
    width, signed = _TYPE_PARTS[type_name]
    low, high = _bounds(width, signed)
    overflow = number < low or number > high
    if overflow and not wrap:
        _fail("E001", f"{number} does not fit in {type_name}")
    converted = number
    if overflow:
        modulus = 2**width
        converted = (number - low) % modulus + low if signed else number % modulus
    return {
        "value": converted,
        "type": type_name,
        "width": width,
        "signed": signed,
        "wrapped": overflow,
    }


def pyruntime_int_div(left: Any, right: Any, *, signed: bool = True) -> dict[str, Any]:
    """Compute truncating integer division with a zero-divisor check."""
    dividend, divisor = _int(left, "left"), _int(right, "right")
    if divisor == 0:
        _fail("E002", "integer division by zero")
    quotient = dividend // divisor
    if (dividend < 0) != (divisor < 0) and dividend % divisor:
        quotient += 1
    return {
        "value": quotient,
        "operator": "/",
        "dividend": dividend,
        "divisor": divisor,
        "signed": signed,
    }


def pyruntime_int_rem(left: Any, right: Any) -> dict[str, Any]:
    """Compute a signed remainder using truncating-division semantics."""
    dividend, divisor = _int(left, "left"), _int(right, "right")
    if divisor == 0:
        _fail("E002", "integer remainder by zero")
    quotient = pyruntime_int_div(dividend, divisor)["value"]
    return {
        "value": dividend - quotient * divisor,
        "operator": "%",
        "dividend": dividend,
        "divisor": divisor,
    }


def pyruntime_checked_add(
    left: Any,
    right: Any,
    *,
    width: int = 64,
    signed: bool = True,
    wrap: bool = False,
) -> dict[str, Any]:
    """Add two integers with an ABI-width overflow check."""
    first, second = _int(left, "left"), _int(right, "right")
    low, high = _bounds(width, signed)
    total = first + second
    overflow = total < low or total > high
    if overflow and not wrap:
        _fail("E001", f"addition overflow for width {width}")
    modulus = 2**width
    value = (
        (total - low) % modulus + low
        if overflow and signed
        else (total % modulus if overflow else total)
    )
    return {"value": value, "overflow": overflow, "width": width, "signed": signed}


def pyruntime_checked_mul(
    left: Any,
    right: Any,
    *,
    width: int = 64,
    signed: bool = True,
    wrap: bool = False,
) -> dict[str, Any]:
    """Multiply two integers with an ABI-width overflow check."""
    first, second = _int(left, "left"), _int(right, "right")
    low, high = _bounds(width, signed)
    product = first * second
    overflow = product < low or product > high
    if overflow and not wrap:
        _fail("E001", f"multiplication overflow for width {width}")
    modulus = 2**width
    value = (
        (product - low) % modulus + low
        if overflow and signed
        else (product % modulus if overflow else product)
    )
    return {"value": value, "overflow": overflow, "width": width, "signed": signed}


def pyruntime_assert_true(
    condition: Any, *, message: str | None = None
) -> dict[str, Any]:
    """Assert a truthy condition, raising ``E002`` when it fails."""
    if not condition:
        _fail("E002", message or "assertion failed")
    return {"ok": True, "message": message or "assertion succeeded", "condition": True}


def pyruntime_assert_eq(
    left: Any, right: Any, *, message: str | None = None
) -> dict[str, Any]:
    """Assert equality of two runtime values."""
    if left != right:
        _fail("E002", message or f"assertion failed: {left!r} != {right!r}")
    return {"ok": True, "left": left, "right": right, "comparison": "eq"}


def pyruntime_assert_ne(
    left: Any, right: Any, *, message: str | None = None
) -> dict[str, Any]:
    """Assert inequality of two runtime values."""
    if left == right:
        _fail("E002", message or f"assertion failed: {left!r} == {right!r}")
    return {"ok": True, "left": left, "right": right, "comparison": "ne"}


def pyruntime_panic(message: str, *, code: str = "E002") -> NoReturn:
    """Raise a structured panic diagnostic with an E001-E008 code."""
    description = _text(message, "message")
    selected = _text(code, "code")
    if selected not in _CODES:
        _fail("E003", f"unknown panic code: {selected}")
    _fail(selected, f"panic: {description}")


def pyruntime_match_exhaustiveness(
    tags: Sequence[str],
    *,
    wildcard: bool = False,
    variants: Sequence[str] = (),
) -> dict[str, Any]:
    """Check match-arm coverage against a declared set of enum variants."""
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        _fail("E003", "tags must be a sequence of strings")
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        _fail("E003", "variants must be a sequence of strings")
    covered = [_text(tag, "tag") for tag in tags]
    declared = [_text(variant, "variant") for variant in variants]
    if len(set(covered)) != len(covered):
        _fail("E001", "duplicate match tags are unreachable")
    missing = [variant for variant in declared if variant not in covered]
    unreachable = [tag for tag in covered if declared and tag not in declared]
    exhaustive = wildcard or (bool(declared) and not missing)
    return {
        "exhaustive": exhaustive,
        "tags": sorted(set(covered)),
        "missing": missing,
        "unreachable": unreachable,
        "wildcard": wildcard,
    }


def pyruntime_enum_decode(
    encoded: Any, *, variants: Sequence[str] = ()
) -> dict[str, Any]:
    """Decode a tagged ``(tag, *payload)`` tuple into a structured record."""
    if not isinstance(encoded, (tuple, list)) or not encoded:
        _fail("E008", "encoded enum must be a non-empty tagged tuple")
    tag = _text(encoded[0], "tag")
    payload = tuple(encoded[1:])
    names: list[str] = []
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        _fail("E003", "variants must be a sequence of strings")
    for variant in variants:
        names.append(_text(variant, "variant"))
    if names and tag not in names:
        _fail("E002", f"unknown enum variant: {tag}")
    return {
        "tag": tag,
        "payload": payload,
        "arity": len(payload),
        "variant": tag,
        "valid": True,
    }


def pyruntime_enum_encode(tag: str, payload: Sequence[Any] = ()) -> dict[str, Any]:
    """Encode a variant tag and payload as a tagged tuple."""
    name = _text(tag, "tag")
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        _fail("E003", "payload must be a sequence")
    values = tuple(payload)
    encoded = (name, *values)
    return {
        "encoded": encoded,
        "tag": name,
        "payload": values,
        "arity": len(values),
    }
