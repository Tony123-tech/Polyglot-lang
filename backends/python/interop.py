"""Safe CPython interop modelling helpers for the Python backend rollout.

These helpers describe native bindings, ABI layouts, and marshalled values
without importing extension modules or executing untrusted code.  A caller
supplies an explicit registry of already-trusted callables and a list of
allowlisted symbols; every public function returns a structured record that can
be inspected by generated Python or by the parity harness.

The ``E001``-``E008`` diagnostics mirror the rest of the toolchain: layout and
arity errors use ``E001``, unmet semantic obligations use ``E002``, API misuse
uses ``E003``, toolchain failures use ``E004``, alias cycles use ``E005``,
missing modules or symbols use ``E006``, unsupported platforms use ``E007``,
and malformed wire state uses ``E008``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "pyinterop_abi_layout",
    "pyinterop_abi_pointer_width",
    "pyinterop_bind_callable",
    "pyinterop_bind_registry",
    "pyinterop_call_plan",
    "pyinterop_check_abi_compatibility",
    "pyinterop_describe_abi",
    "pyinterop_error_translation",
    "pyinterop_extension_manifest",
    "pyinterop_marshal_result",
    "pyinterop_marshal_scalar",
    "pyinterop_marshal_sequence",
    "pyinterop_native_module_probe",
    "pyinterop_plan_binding",
    "pyinterop_resolve_fallback",
    "pyinterop_symbol_digest",
    "pyinterop_unmarshal_result",
    "pyinterop_unmarshal_scalar",
    "pyinterop_unmarshal_sequence",
    "pyinterop_validate_symbol_name",
]

_PLATFORMS: Final[frozenset[str]] = frozenset(
    {"darwin", "linux", "win32", "freebsd", "cygwin"}
)
_FEATURES: Final[frozenset[str]] = frozenset(
    {"cffi", "cpython_api", "limited_api", "native_extension", "pyo3"}
)
_SCALARS: Final[dict[str, str]] = {
    "bool": "bool",
    "bytes": "bytes",
    "float": "float",
    "int": "int",
    "none": "none",
    "str": "str",
}
_CODES: Final[frozenset[str]] = frozenset({f"E{index:03d}" for index in range(1, 10)})
_FALLBACK_POLICIES: Final[frozenset[str]] = frozenset(
    {"fallback", "native_only", "strict"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured interop diagnostic."""
    raise PolyglotError(code, message)


def _text(value: object, label: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty str")
    return value


def _identifier(value: object, label: str, *, dotted: bool = False) -> str:
    """Validate a Python identifier or dotted native symbol path."""
    text = _text(value, label)
    parts = text.split(".") if dotted else [text]
    if not all(part.isidentifier() for part in parts):
        _fail("E001", f"{label} is not a valid identifier: {text!r}")
    return text


def _mapping(value: object, label: str) -> dict[str, Any]:
    """Require a string-keyed mapping without copying its values."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("E003", f"{label} keys must be str")
        result[key] = item
    return result


def _sequence(value: object, label: str) -> list[Any]:
    """Require a sequence that is not a string or bytes object."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("E003", f"{label} must be a sequence")
    return list(value)


def _encode_scalar(value: object) -> dict[str, Any]:
    """Encode one supported scalar as a JSON-compatible wire record."""
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            _fail("E001", "native int must fit in a signed 64-bit field")
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("E008", "native float must be finite")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "value": bytes(value).hex()}
    _fail("E001", f"unsupported native scalar: {type(value).__name__}")


def _decode_scalar(record: object) -> tuple[Any, str]:
    """Decode one wire record, rejecting malformed state with ``E008``."""
    if not isinstance(record, Mapping) or "type" not in record or "value" not in record:
        _fail("E008", "malformed scalar wire record")
    kind = record["type"]
    value = record["value"]
    if kind not in _SCALARS:
        _fail("E008", f"unknown scalar wire type: {kind!r}")
    if kind == "none" and value is not None:
        _fail("E008", "none wire record must carry a null value")
    if kind == "bool" and not isinstance(value, bool):
        _fail("E008", "bool wire record must carry a bool")
    if kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
        _fail("E008", "int wire record must carry an int")
    if kind == "float" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        _fail("E008", "float wire record must carry a number")
    if kind == "str" and not isinstance(value, str):
        _fail("E008", "str wire record must carry a str")
    if kind == "bytes":
        if not isinstance(value, str):
            _fail("E008", "bytes wire record must carry a hex string")
        try:
            value = bytes.fromhex(value)
        except ValueError:
            _fail("E008", "bytes wire record has invalid hex")
    return value, kind


def _alias_target(name: str, aliases: Mapping[str, str]) -> str:
    """Follow a chain of aliases, raising ``E005`` on a cycle."""
    seen: list[str] = [name]
    current = name
    while current in aliases:
        current = aliases[current]
        if current in seen:
            _fail("E005", f"native alias cycle: {' -> '.join([*seen, current])}")
        seen.append(current)
    return current


# ---------------------------------------------------------------------------
# Public interop helpers
# ---------------------------------------------------------------------------


def pyinterop_describe_abi(
    *,
    platform_name: str | None = None,
    pointer_width: int = 64,
    features: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe a supported native ABI without probing the host toolchain."""
    platform_text = (
        sys.platform if platform_name is None else _text(platform_name, "platform_name")
    )
    if platform_text not in _PLATFORMS:
        _fail("E007", f"unsupported native platform: {platform_text!r}")
    if pointer_width not in (32, 64):
        _fail("E001", f"pointer_width must be 32 or 64, got {pointer_width}")
    selected = list(dict.fromkeys(_text(feature, "feature") for feature in features))
    unknown = sorted(set(selected) - _FEATURES)
    if unknown:
        _fail("E007", f"unsupported native features: {', '.join(unknown)}")
    flags = ["pep667-frame-locals", "structured-errors", *selected]
    return {
        "abi": f"{platform_text}-{pointer_width}",
        "platform": platform_text,
        "pointer_width": pointer_width,
        "endianness": "little",
        "features": sorted(selected),
        "flags": sorted(set(flags)),
    }


def pyinterop_validate_symbol_name(name: str) -> dict[str, Any]:
    """Validate a dotted native symbol name and split it into its parts."""
    text = _identifier(name, "symbol name", dotted=True)
    parts = text.split(".")
    return {
        "name": text,
        "module": ".".join(parts[:-1]),
        "attribute": parts[-1],
        "parts": parts,
        "valid": True,
    }


def pyinterop_plan_binding(
    module_name: str,
    *,
    requested: Sequence[str] = (),
    registry: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Plan symbol resolution from trusted registries without importing code."""
    module = _identifier(module_name, "module_name", dotted=True)
    names = list(dict.fromkeys(_identifier(name, "symbol") for name in requested))
    native = _mapping(registry or {}, "registry")
    fallbacks = _mapping(fallback or {}, "fallback")
    alias_map = _mapping(aliases or {}, "aliases")
    for source, target in alias_map.items():
        _identifier(source, "alias source")
        _identifier(target, "alias target")
    for name in names:
        _alias_target(name, alias_map)
    plan: list[dict[str, str]] = []
    resolved_native: list[str] = []
    resolved_fallback: list[str] = []
    missing: list[str] = []
    for name in names:
        target = _alias_target(name, alias_map)
        if target in native:
            source = "native"
            resolved_native.append(name)
        elif target in fallbacks:
            source = "fallback"
            resolved_fallback.append(name)
        else:
            source = "missing"
            missing.append(name)
        plan.append({"name": name, "alias": target, "source": source})
    return {
        "module": module,
        "requested": names,
        "native": resolved_native,
        "fallback": resolved_fallback,
        "missing": missing,
        "aliases": dict(sorted(alias_map.items())),
        "plan": plan,
    }


def pyinterop_bind_callable(
    name: str,
    callable_value: Any,
    *,
    source: str = "native",
    arity: int | None = None,
) -> dict[str, Any]:
    """Bind one trusted callable with an optional required positional arity."""
    symbol = _identifier(name, "symbol")
    if not callable(callable_value):
        _fail("E003", f"{symbol} is not callable")
    origin = _text(source, "source")
    if origin not in ("native", "fallback"):
        _fail("E001", f"unsupported binding source: {origin!r}")
    try:
        signature = inspect.signature(callable_value)
    except TypeError, ValueError:
        _fail("E004", f"cannot inspect callable {symbol}")
    required = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if arity is not None:
        if isinstance(arity, bool) or not isinstance(arity, int) or arity < 0:
            _fail("E003", "arity must be a non-negative int")
        if required != arity:
            _fail("E001", f"{symbol} requires {required} arguments, expected {arity}")
    return {
        "name": symbol,
        "callable": callable_value,
        "source": origin,
        "arity": required,
        "bound": True,
    }


def pyinterop_bind_registry(
    registry: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bind an explicit allowlist of trusted native and fallback callables."""
    native = _mapping(registry, "registry")
    fallbacks = _mapping(fallback or {}, "fallback")
    selected = (
        sorted(native)
        if names is None
        else list(dict.fromkeys(_identifier(name, "symbol") for name in names))
    )
    bindings: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in selected:
        if name in native:
            bindings[name] = pyinterop_bind_callable(
                name, native[name], source="native"
            )
        elif name in fallbacks:
            bindings[name] = pyinterop_bind_callable(
                name, fallbacks[name], source="fallback"
            )
        else:
            missing.append(name)
    return {
        "bindings": bindings,
        "order": selected,
        "missing": missing,
        "count": len(bindings),
    }


def pyinterop_check_abi_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two ABI descriptors and explain every incompatibility."""
    first, second = _mapping(left, "left"), _mapping(right, "right")
    required = ("platform", "pointer_width", "endianness")
    for label, descriptor in (("left", first), ("right", second)):
        if any(field not in descriptor for field in required):
            _fail("E001", f"{label} ABI descriptor is missing required fields")
    reasons: list[str] = []
    for field in required:
        if first[field] != second[field]:
            reasons.append(f"{field}: {first[field]!r} != {second[field]!r}")
    first_features = set(first.get("features", ()))
    second_features = set(second.get("features", ()))
    if first_features - second_features:
        reasons.append(
            f"missing features: {', '.join(sorted(first_features - second_features))}"
        )
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "left": dict(sorted(first.items(), key=lambda item: item[0])),
        "right": dict(sorted(second.items(), key=lambda item: item[0])),
    }


def pyinterop_marshal_scalar(value: object) -> dict[str, Any]:
    """Encode one supported scalar into a stable native wire record."""
    encoded = _encode_scalar(value)
    return {
        "kind": encoded["type"],
        "value": value,
        "encoded": encoded,
        "size": len(json.dumps(encoded, sort_keys=True).encode("utf-8")),
    }


def pyinterop_unmarshal_scalar(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a native scalar wire record into a Python value."""
    value, kind = _decode_scalar(encoded)
    return {"value": value, "kind": kind, "valid": True}


def pyinterop_marshal_sequence(values: Sequence[object]) -> dict[str, Any]:
    """Encode an ordered sequence of scalars into native wire records."""
    items = _sequence(values, "values")
    encoded = [_encode_scalar(value) for value in items]
    return {"encoded": encoded, "length": len(encoded), "kind": "sequence"}


def pyinterop_unmarshal_sequence(
    encoded: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decode an ordered list of native scalar wire records."""
    records = _sequence(encoded, "encoded")
    decoded = [_decode_scalar(record) for record in records]
    return {
        "values": [value for value, _ in decoded],
        "kinds": [kind for _, kind in decoded],
        "length": len(decoded),
    }


def pyinterop_marshal_result(
    value: object,
    *,
    error_type: type[Exception] = RuntimeError,
    variant: str | None = None,
) -> dict[str, Any]:
    """Marshal a value, exception, or tagged tuple into a Result wire record."""
    if not isinstance(error_type, type) or not issubclass(error_type, Exception):
        _fail("E003", "error_type must be an Exception subclass")
    tag = "ok"
    payload: Any = value
    label = variant
    if isinstance(value, BaseException):
        tag = "err"
        payload = value
        label = label or type(value).__name__
    elif isinstance(value, tuple) and len(value) == 2 and value[0] in ("Ok", "Err"):
        tag = "ok" if value[0] == "Ok" else "err"
        payload = value[1]
        label = label or value[0]
    encoded_payload = (
        {"type": "error", "value": str(payload)}
        if tag == "err"
        else _encode_scalar(payload)
    )
    return {
        "tag": tag,
        "variant": label or ("Ok" if tag == "ok" else "Err"),
        "payload": payload,
        "encoded": {"tag": tag, "payload": encoded_payload},
    }


def pyinterop_unmarshal_result(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a Result wire record into a structured success/error outcome."""
    if not isinstance(encoded, Mapping):
        _fail("E008", "result wire record must be a mapping")
    tag = encoded.get("tag")
    if tag not in ("ok", "err"):
        _fail("E008", f"invalid result tag: {tag!r}")
    payload = encoded.get("payload")
    if not isinstance(payload, Mapping):
        _fail("E008", "result payload must be a wire record")
    if payload.get("type") == "error":
        if not isinstance(payload.get("value"), str):
            _fail("E008", "error payload must carry a message string")
        value, kind = payload["value"], "error"
    else:
        value, kind = _decode_scalar(payload)
    ok = tag == "ok"
    return {
        "ok": ok,
        "value": value if ok else None,
        "error": None if ok else value,
        "kind": kind,
    }


def pyinterop_resolve_fallback(
    name: str,
    *,
    native: Mapping[str, Any] | None = None,
    fallback: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Resolve a symbol from trusted native/fallback mappings without imports."""
    symbol = _identifier(name, "symbol")
    native_map = _mapping(native or {}, "native")
    fallback_map = _mapping(fallback or {}, "fallback")
    if symbol in native_map:
        return {
            "name": symbol,
            "callable": native_map[symbol],
            "source": "native",
            "used_fallback": False,
            "reason": "native symbol present in trusted registry",
        }
    if symbol in fallback_map:
        return {
            "name": symbol,
            "callable": fallback_map[symbol],
            "source": "fallback",
            "used_fallback": True,
            "reason": "native symbol absent; pure-Python fallback selected",
        }
    if strict:
        _fail("E006", f"missing native symbol and fallback: {symbol}")
    return {
        "name": symbol,
        "callable": None,
        "source": "missing",
        "used_fallback": False,
        "reason": "symbol is optional and no fallback was supplied",
    }


def pyinterop_native_module_probe(
    module_name: str,
    *,
    available: Sequence[str] = (),
    strict: bool = False,
) -> dict[str, Any]:
    """Probe a declared module inventory without importing any extension."""
    name = _identifier(module_name, "module_name", dotted=True)
    inventory = sorted(
        {
            _identifier(candidate, "available module", dotted=True)
            for candidate in available
        }
    )
    present = name in inventory
    if not present and strict:
        _fail("E006", f"native module is not available: {name}")
    return {
        "module": name,
        "present": present,
        "origin": "inventory" if present else "missing",
        "available": inventory,
        "reason": "declared in inventory" if present else "not declared in inventory",
    }


def pyinterop_abi_pointer_width(width: int) -> dict[str, Any]:
    """Report integer and alignment limits for a supported pointer width."""
    if width not in (32, 64):
        _fail("E001", f"pointer width must be 32 or 64, got {width}")
    bits = 32 if width == 32 else 64
    return {
        "width": width,
        "bytes": width // 8,
        "bits": bits,
        "min_signed": -(2 ** (bits - 1)),
        "max_signed": 2 ** (bits - 1) - 1,
        "max_unsigned": 2**bits - 1,
        "alignment": width // 8,
    }


def pyinterop_abi_layout(
    fields: Sequence[tuple[str, int, int]], *, pointer_width: int = 64
) -> dict[str, Any]:
    """Compute a deterministic native struct layout from field specifications."""
    if pointer_width not in (32, 64):
        _fail("E001", f"pointer_width must be 32 or 64, got {pointer_width}")
    layout: list[dict[str, Any]] = []
    offset = 0
    overall_alignment = 1
    for field in _sequence(fields, "fields"):
        if not isinstance(field, tuple) or len(field) != 3:
            _fail("E001", "each field must be a (name, size, alignment) triple")
        name = _identifier(field[0], "field name")
        size, alignment = field[1], field[2]
        if any(
            isinstance(part, bool) or not isinstance(part, int)
            for part in (size, alignment)
        ):
            _fail("E001", "field size and alignment must be ints")
        if size < 0 or alignment < 1 or alignment & (alignment - 1):
            _fail("E001", "field alignment must be a positive power of two")
        offset = (offset + alignment - 1) // alignment * alignment
        layout.append(
            {"name": name, "offset": offset, "size": size, "alignment": alignment}
        )
        offset += size
        overall_alignment = max(overall_alignment, alignment)
    total = (offset + overall_alignment - 1) // overall_alignment * overall_alignment
    return {
        "size": total,
        "alignment": overall_alignment,
        "pointer_width": pointer_width,
        "fields": layout,
    }


def pyinterop_symbol_digest(
    module_name: str,
    symbols: Sequence[str],
    *,
    abi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a stable digest for a module, symbol allowlist, and ABI."""
    name = _identifier(module_name, "module_name", dotted=True)
    names = sorted({_identifier(symbol, "symbol", dotted=True) for symbol in symbols})
    descriptor = dict(sorted((abi or {}).items(), key=lambda item: item[0]))
    payload = json.dumps(
        {"module": name, "symbols": names, "abi": descriptor},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "digest": hashlib.sha256(payload).hexdigest(),
        "algorithm": "sha256",
        "inputs": {"module": name, "symbols": names, "abi": descriptor},
        "length": 64,
    }


def pyinterop_call_plan(
    name: str,
    *,
    arguments: Sequence[Mapping[str, Any]] = (),
    keywords: Mapping[str, Mapping[str, Any]] | None = None,
    budget: int = 8,
) -> dict[str, Any]:
    """Build a bounded native call plan from already-encoded wire records."""
    symbol = _identifier(name, "symbol", dotted=True)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        _fail("E001", "budget must be a non-negative int")
    args = [_encode_scalar(argument) for argument in _sequence(arguments, "arguments")]
    pairs = list(_mapping(keywords or {}, "keywords").items())
    rendered_keywords: list[dict[str, Any]] = []
    for key, value in pairs:
        rendered_keywords.append(
            {"name": _identifier(key, "keyword"), "encoded": _encode_scalar(value)}
        )
    total = len(args) + len(rendered_keywords)
    return {
        "name": symbol,
        "arguments": args,
        "keywords": rendered_keywords,
        "budget": budget,
        "allowed": total <= budget,
        "call": f"{symbol}(<{len(args)} positional>, <{len(rendered_keywords)} keyword>)",
    }


def pyinterop_error_translation(
    code: str, message: str, *, payload: object = None
) -> dict[str, Any]:
    """Translate an interop failure into a structured E-code diagnostic."""
    diagnostic_code = _text(code, "code")
    if diagnostic_code not in _CODES:
        _fail("E003", f"unknown diagnostic code: {diagnostic_code}")
    description = _text(message, "message")
    return {
        "code": diagnostic_code,
        "message": description,
        "diagnostic": {
            "code": diagnostic_code,
            "message": description,
            "payload": payload,
        },
        "attribution": "cpython-interop",
    }


def pyinterop_extension_manifest(
    module_name: str,
    *,
    symbols: Sequence[str] = (),
    abi: Mapping[str, Any] | None = None,
    fallback_policy: str = "fallback",
    registry: Mapping[str, Any] | None = None,
    toolchain_error: str | None = None,
) -> dict[str, Any]:
    """Assemble a complete interop manifest for one native extension module."""
    if toolchain_error is not None:
        _fail(
            "E004",
            f"native toolchain unavailable: {_text(toolchain_error, 'toolchain_error')}",
        )
    policy = _text(fallback_policy, "fallback_policy")
    if policy not in _FALLBACK_POLICIES:
        _fail("E001", f"unsupported fallback policy: {policy!r}")
    descriptor = dict(abi or {})
    plan = pyinterop_plan_binding(module_name, requested=symbols, registry=registry)
    digest = pyinterop_symbol_digest(module_name, symbols, abi=descriptor)
    warnings: list[str] = []
    if plan["missing"]:
        warnings.append(f"unresolved symbols: {', '.join(plan['missing'])}")
    if policy == "strict" and plan["missing"]:
        _fail(
            "E006", f"strict manifest cannot omit symbols: {', '.join(plan['missing'])}"
        )
    return {
        "module": plan["module"],
        "manifest": {
            "module": plan["module"],
            "symbols": plan["requested"],
            "abi": descriptor,
            "fallback_policy": policy,
            "resolved": plan["plan"],
        },
        "digest": digest["digest"],
        "warnings": warnings,
    }
