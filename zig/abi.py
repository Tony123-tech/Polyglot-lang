"""Model-only ABI and layout decisions for the Zig backend rollout.

The functions here describe calling conventions, struct layout, C-ABI shims and
opaque address *keys* without invoking a toolchain, allocating executable
memory, or dereferencing any address.  A raw address is only ever used as a
key into a caller-supplied Python mapping.  All invalid inputs raise
:class:`PolyglotError` with the E001-E009 diagnostic surface.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "zabi_abi_signature",
    "zabi_alignment",
    "zabi_arg_slots",
    "zabi_bitcast_plan",
    "zabi_buffer_layout",
    "zabi_cabi_type",
    "zabi_call_frame",
    "zabi_calling_convention",
    "zabi_canonical_signature",
    "zabi_data_layout",
    "zabi_enum_repr",
    "zabi_error_map",
    "zabi_export_registry",
    "zabi_field_offsets",
    "zabi_float_policy",
    "zabi_int_conversion",
    "zabi_layout_compare",
    "zabi_layout_fingerprint",
    "zabi_opaque_address",
    "zabi_param_class",
    "zabi_pointer_width",
    "zabi_return_class",
    "zabi_serialization_format",
    "zabi_shim_body",
    "zabi_sizeof",
    "zabi_slice_repr",
    "zabi_struct_repr",
    "zabi_symbol_name",
    "zabi_target_spec",
    "zabi_type_class",
]

_TARGETS: Final[frozenset[str]] = frozenset(
    {"aarch64", "native", "wasm32", "wasm64", "x86_64"}
)
_ABIS: Final[frozenset[str]] = frozenset({"c", "fast", "zig"})
_LANES: Final[frozenset[int]] = frozenset({2, 4, 8, 16, 32, 64})
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER: Final[re.Pattern[str]] = re.compile(r"^([iu])([1-9][0-9]*)$")
_MAX_INTEGER_BITS: Final[int] = 65535
_VECTOR: Final[re.Pattern[str]] = re.compile(r"^@Vector\((\d+),\s*([A-Za-z0-9_]+)\)$")
_SIGNATURE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)(?:->([A-Za-z_][A-Za-z0-9_]*))?$"
)
_FORMATS: Final[frozenset[str]] = frozenset({"bincode", "json", "model_v1"})
_ERRORS: Final[dict[str, tuple[int, str, bool]]] = {
    "E001": (22, "invalid_argument", True),
    "E002": (5, "unmet_obligation", False),
    "E003": (22, "api_misuse", True),
    "E004": (74, "toolchain_failure", True),
    "E005": (40, "dependency_cycle", False),
    "E006": (2, "missing_artifact", True),
    "E007": (95, "unsupported_feature", False),
    "E008": (84, "malformed_state", False),
    "E009": (22, "ownership_violation", False),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured Zig-ABI diagnostic."""
    raise PolyglotError(code, message, reason=reason, phase="zig-abi")


def _text(value: object, label: str) -> str:
    """Return *value* as text or raise E003 for the wrong API type."""
    if not isinstance(value, str):
        _fail("E003", f"{label} must be a str")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    """Return an int, rejecting booleans and values below *minimum*."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("E003", f"{label} must be an int")
    if minimum is not None and value < minimum:
        _fail("E003", f"{label} must be >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    """Return a bool or raise E003 when another type was supplied."""
    if not isinstance(value, bool):
        _fail("E003", f"{label} must be a bool")
    return value


def _items(value: object, label: str) -> tuple[Any, ...]:
    """Freeze a non-string sequence into a tuple, rejecting mappings."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        _fail("E003", f"{label} must be a sequence")
    return tuple(value)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    """Return a string-keyed mapping or raise E003."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{label} must be a mapping")
    return value


def _identifier(value: object, label: str) -> str:
    """Validate a Zig symbol name, raising E001 for illegal names."""
    name = _text(value, label)
    if not _IDENTIFIER.fullmatch(name):
        _fail("E001", f"{label} is not a legal Zig symbol: {name!r}")
    return name


def _target(value: object) -> str:
    """Validate a model target, raising E007 for unknown platforms."""
    name = _text(value, "target")
    if name not in _TARGETS:
        _fail("E007", f"unsupported target: {name!r}")
    return name


def _word(value: object) -> int:
    """Validate a modeled pointer word size (4 or 8 bytes)."""
    width = _integer(value, "word", minimum=1)
    if width not in (4, 8):
        _fail("E001", "word size must be 4 or 8 bytes")
    return width


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return ((value + alignment - 1) // alignment) * alignment


def _pow2_ceil(value: int) -> int:
    """Smallest power of two greater than or equal to *value*."""
    return 1 << max(0, value - 1).bit_length()


def _digest(*parts: object) -> str:
    """Stable 16-hex-digit digest of the string form of *parts*."""
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _type_info(value: object, word: int) -> dict[str, Any]:
    """Resolve a type name into a modeled class, size and alignment."""
    name = _text(value, "type name")
    if name in ("ptr", "anyopaque"):
        return {
            "type": name,
            "kind": "pointer",
            "size": word,
            "align": word,
            "bits": word * 8,
            "signed": None,
        }
    if name == "void":
        return {
            "type": "void",
            "kind": "void",
            "size": 0,
            "align": 1,
            "bits": 0,
            "signed": None,
        }
    if name in ("isize", "usize"):
        return {
            "type": name,
            "kind": "int",
            "size": word,
            "align": word,
            "bits": word * 8,
            "signed": name == "isize",
        }
    integer = _INTEGER.fullmatch(name)
    if integer is not None:
        bits = int(integer.group(2))
        if not 1 <= bits <= _MAX_INTEGER_BITS:
            _fail("E001", f"integer width out of range: {bits}")
        size = max(1, (bits + 7) // 8)
        return {
            "type": name,
            "kind": "int",
            "size": size,
            "align": _pow2_ceil(size),
            "bits": bits,
            "signed": integer.group(1) == "i",
        }
    if name == "bool":
        return {
            "type": "bool",
            "kind": "bool",
            "size": 1,
            "align": 1,
            "bits": 1,
            "signed": None,
        }
    if name in ("f32", "f64"):
        size = 4 if name == "f32" else 8
        return {
            "type": name,
            "kind": "float",
            "size": size,
            "align": size,
            "bits": size * 8,
            "signed": None,
        }
    if name.startswith("[]"):
        return {
            "type": name,
            "kind": "slice",
            "size": 2 * word,
            "align": word,
            "bits": 2 * word * 8,
            "signed": None,
        }
    if name.startswith(("[*]", "*")):
        return {
            "type": name,
            "kind": "pointer",
            "size": word,
            "align": word,
            "bits": word * 8,
            "signed": None,
        }
    vector = _VECTOR.fullmatch(name)
    if vector is not None:
        lanes = int(vector.group(1))
        if lanes not in _LANES:
            _fail("E001", f"unsupported vector lane count: {lanes}")
        element = _type_info(vector.group(2), word)
        if element["kind"] == "float":
            _fail("E007", "float vector ABI is not modeled")
        if element["kind"] not in ("bool", "int"):
            _fail("E001", "vector elements must be integers")
        return {
            "type": f"@Vector({lanes}, {element['type']})",
            "kind": "vector",
            "size": element["size"] * lanes,
            "align": element["align"],
            "bits": element["size"] * lanes * 8,
            "signed": element["signed"],
        }
    _fail("E001", f"unsupported ABI type: {name!r}")


def _abi_type(value: object, word: int) -> str:
    """Canonicalize one ABI type token, allowing the opaque ``ptr`` token."""
    name = _text(value, "ABI type")
    if name in ("ptr", "anyopaque", "void"):
        return name
    return _type_info(name, word)["type"]


def _field_offsets(fields: object, word: int) -> dict[str, Any]:
    """Compute field offsets, total size, alignment and padding."""
    rows: list[dict[str, Any]] = []
    offset = 0
    maximum = 1
    padding = 0
    for item in _items(fields, "fields"):
        pair = _items(item, "field")
        if len(pair) != 2:
            _fail("E003", "each field must be a (name, type) pair")
        name = _identifier(pair[0], "field name")
        info = _type_info(pair[1], word)
        gap = (-offset) % info["align"]
        offset += gap
        padding += gap
        rows.append(
            {
                "name": name,
                "type": info["type"],
                "offset": offset,
                "size": info["size"],
                "align": info["align"],
            }
        )
        offset += info["size"]
        maximum = max(maximum, info["align"])
    total = _align_up(offset, maximum)
    padding += total - offset
    return {"fields": tuple(rows), "size": total, "align": maximum, "padding": padding}


def _signature_params(signature: object, word: int) -> tuple[list[str], str]:
    """Extract canonical parameter types and result type from a signature."""
    if isinstance(signature, Mapping):
        data = _mapping(signature, "signature")
        if "params" not in data or "result" not in data:
            _fail("E008", "malformed serialized signature mapping")
        params = [
            _abi_type(item, word) for item in _items(data["params"], "signature params")
        ]
        result = _abi_type(data["result"], word)
        return params, result
    canonical = zabi_canonical_signature(signature)
    params = [_abi_type(item, word) for item in canonical["params"]]
    return params, _abi_type(canonical["result"], word)


# ---------------------------------------------------------------------------
# Public ABI functions
# ---------------------------------------------------------------------------


def zabi_abi_signature(name: str, params: Sequence[str], result: str) -> dict[str, Any]:
    """Create a canonical symbolic signature for one exported function."""
    symbol = _identifier(name, "function name")
    types = tuple(_abi_type(item, 8) for item in _items(params, "params"))
    output = _abi_type(result, 8)
    canonical = f"{symbol}({','.join(types)})->{output}"
    return {
        "name": symbol,
        "params": types,
        "result": output,
        "canonical": canonical,
        "fingerprint": _digest("signature", canonical),
    }


def zabi_alignment(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Report the modeled alignment of a type for a target word size."""
    info = _type_info(type_name, _word(word))
    return {
        "type": info["type"],
        "align": info["align"],
        "size": info["size"],
        "kind": info["kind"],
    }


def zabi_arg_slots(
    signature: Mapping[str, Any] | str, *, word: int = 8
) -> dict[str, Any]:
    """Assign aligned stack slots to each signature parameter."""
    params, _result = _signature_params(signature, _word(word))
    rows: list[dict[str, Any]] = []
    offset = 0
    maximum = 1
    for index, type_name in enumerate(params):
        info = _type_info(type_name, word)
        gap = (-offset) % info["align"]
        offset += gap
        rows.append(
            {
                "index": index,
                "type": info["type"],
                "offset": offset,
                "size": info["size"],
                "align": info["align"],
            }
        )
        offset += info["size"]
        maximum = max(maximum, info["align"])
    return {
        "slots": tuple(rows),
        "bytes": _align_up(offset, maximum),
        "align": maximum,
        "count": len(rows),
    }


def zabi_bitcast_plan(
    source_type: str, target_type: str, *, word: int = 8
) -> dict[str, Any]:
    """Plan a bit-preserving reinterpretation between equal-sized types."""
    width = _word(word)
    source = _type_info(source_type, width)
    target = _type_info(target_type, width)
    if source["size"] != target["size"]:
        _fail(
            "E001",
            f"bitcast size mismatch: {source['type']} ({source['size']}) "
            f"vs {target['type']} ({target['size']})",
        )
    return {
        "source": source["type"],
        "target": target["type"],
        "size": source["size"],
        "allowed": True,
        "expr": f"@as({target['type']}, @bitCast(value))",
    }


def zabi_buffer_layout(
    capacity: int, *, elem_type: str = "u8", word: int = 8
) -> dict[str, Any]:
    """Describe a fixed-capacity modeled buffer layout."""
    count = _integer(capacity, "capacity", minimum=0)
    element = _type_info(elem_type, _word(word))
    return {
        "capacity": count,
        "elem_type": element["type"],
        "elem_size": element["size"],
        "size": count * element["size"],
        "align": element["align"],
        "expr": f"[{count}]{element['type']}",
        "model_only": True,
    }


def zabi_cabi_type(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Map a modeled Zig type to its C-ABI representation."""
    width = _word(word)
    info = _type_info(type_name, width)
    kind = info["kind"]
    if kind == "void":
        mapped = "void"
    elif kind in ("bool", "float"):
        mapped = info["type"]
    elif kind == "int" and info["size"] <= 8:
        mapped = (
            ("i64" if info["signed"] else "u64")
            if info["type"]
            not in (
                "isize",
                "usize",
            )
            else info["type"]
        )
    elif kind == "int":
        _fail("E007", "wide integers use an aggregate C-ABI model")
    elif kind in ("pointer", "slice"):
        mapped = "[*]const u8"
    else:
        _fail("E007", f"vector types have no modeled C ABI: {info['type']}")
    return {
        "type": info["type"],
        "cabi_type": mapped,
        "size": info["size"],
        "align": info["align"],
        "model_only": True,
    }


def zabi_call_frame(
    signature: Mapping[str, Any] | str, *, word: int = 8
) -> dict[str, Any]:
    """Model the argument area and return slot of one call frame."""
    width = _word(word)
    slots = zabi_arg_slots(signature, word=width)
    _params, result = _signature_params(signature, width)
    output = _type_info(result, width)
    frame_align = max(width, slots["align"], output["align"])
    args_size = _align_up(slots["bytes"], frame_align)
    return_slot = args_size
    frame_size = _align_up(return_slot + max(1, output["size"]), frame_align)
    return {
        "args_size": args_size,
        "result_size": output["size"],
        "return_slot": return_slot,
        "frame_size": frame_size,
        "align": frame_align,
        "model_only": True,
    }


def zabi_calling_convention(abi: str, *, target: str = "native") -> dict[str, Any]:
    """Select a modeled calling convention and its guarantees."""
    kind = _text(abi, "abi")
    if kind not in _ABIS:
        _fail("E003", f"unknown calling convention: {kind!r}")
    platform = _target(target)
    if platform in ("wasm32", "wasm64") and kind != "c":
        _fail("E007", f"calling convention {kind!r} is unavailable on {platform}")
    name = ".c" if kind == "c" else ".zig" if kind == "zig" else ".fast"
    preserves = ("callee_saved_registers",) if kind == "c" else ()
    return {
        "abi": kind,
        "target": platform,
        "name": name,
        "expr": f"callconv({name})",
        "preserves": preserves,
        "model_only": True,
    }


def zabi_canonical_signature(signature: str) -> dict[str, Any]:
    """Parse and canonicalize a serialized symbolic function signature."""
    raw = _text(signature, "signature")
    normalized = re.sub(r"\s+", "", raw)
    match = _SIGNATURE.fullmatch(normalized)
    if match is None:
        _fail("E008", f"malformed serialized signature: {raw!r}")
    head, body, tail = match.groups()
    params = tuple(part for part in body.split(",") if part)
    if any(not _IDENTIFIER.fullmatch(part) for part in params):
        _fail("E008", f"malformed signature parameters: {raw!r}")
    result = tail or head
    canonical = f"{result}({','.join(params)})"
    return {
        "input": raw,
        "canonical": canonical,
        "result": result,
        "params": params,
    }


def zabi_data_layout(target: str = "native") -> dict[str, Any]:
    """Return the modeled data layout for a target platform."""
    name = _target(target)
    word = 4 if name == "wasm32" else 8
    return {
        "target": name,
        "word": word,
        "pointer_size": word,
        "max_align": 8 if word == 4 else 16,
        "endian": "little",
        "stack_direction": "down",
        "model_only": True,
    }


def zabi_enum_repr(
    variants: Sequence[str | tuple[str, int]], *, tag: str = "u8"
) -> dict[str, Any]:
    """Describe the storage representation of a tagged enum."""
    tag_info = _type_info(tag, 8)
    if tag_info["kind"] != "int":
        _fail("E001", "enum tag must be an integer type")
    names: list[str] = []
    maximum = 0
    for index, item in enumerate(_items(variants, "variants")):
        if isinstance(item, str):
            label = _identifier(item, "variant name")
            value = index
        else:
            pair = _items(item, "variant")
            if len(pair) != 2:
                _fail("E003", "variant must be a name or (name, value) pair")
            label = _identifier(pair[0], "variant name")
            value = _integer(pair[1], "variant value", minimum=0)
        names.append(label)
        maximum = max(maximum, value)
    if maximum > tag_info["bits"] and tag_info["bits"] < 64:
        _fail("E001", "variant value exceeds enum tag range")
    return {
        "tag": tag_info["type"],
        "variants": tuple(names),
        "max_value": maximum,
        "size": tag_info["size"],
        "align": tag_info["align"],
        "expr": f"enum({tag_info['type']})",
    }


def zabi_error_map(
    code: str, *, target: str = "native", operation: str = "map"
) -> dict[str, Any]:
    """Map one structured E001-E009 code to a modeled platform error."""
    symbol = _text(code, "code")
    if symbol not in _ERRORS:
        _fail("E003", f"invalid diagnostic code: {symbol!r}")
    mode = _text(operation, "operation")
    if mode not in ("map", "classify"):
        _fail("E003", f"unsupported mapping operation: {mode!r}")
    platform = _target(target)
    errno, name, recoverable = _ERRORS[symbol]
    return {
        "code": symbol,
        "errno": errno,
        "name": name,
        "recoverable": recoverable,
        "target": platform,
        "operation": mode,
        "phase": "zig-abi",
    }


def zabi_export_registry(
    exports: Sequence[tuple[str, str]], *, prefix: str = "__cabi_"
) -> dict[str, Any]:
    """Register exported symbols with canonical signatures."""
    marker = _text(prefix, "prefix")
    by_symbol: dict[str, dict[str, Any]] = {}
    for item in _items(exports, "exports"):
        pair = _items(item, "export")
        if len(pair) != 2:
            _fail("E003", "each export must be a (name, signature) pair")
        name = _identifier(pair[0], "export name")
        canonical = zabi_canonical_signature(pair[1])["canonical"]
        symbol = f"{marker}{name}"
        if symbol in by_symbol and by_symbol[symbol]["signature"] != canonical:
            _fail("E001", f"conflicting export signature for {symbol!r}")
        by_symbol[symbol] = {"name": name, "signature": canonical}
    symbols = tuple(sorted(by_symbol))
    return {
        "prefix": marker,
        "symbols": symbols,
        "by_symbol": {key: dict(by_symbol[key]) for key in symbols},
        "count": len(symbols),
    }


def zabi_field_offsets(
    fields: Sequence[tuple[str, str]], *, word: int = 8
) -> dict[str, Any]:
    """Compute padded field offsets with a validated target word size."""
    return _field_offsets(fields, _word(word))


def zabi_float_policy(type_name: str, *, target: str = "native") -> dict[str, Any]:
    """Report the modeled float ABI policy for a scalar type."""
    platform = _target(target)
    info = _type_info(type_name, 8)
    if info["kind"] != "float":
        _fail("E001", f"not a float type: {type_name!r}")
    if info["type"].startswith("@Vector"):
        _fail("E007", "float vector ABI is not modeled")
    return {
        "type": info["type"],
        "target": platform,
        "class": "float",
        "precision": "single" if info["bits"] == 32 else "double",
        "vector_ok": False,
        "note": "scalar floats only; integer-safe vectors are modeled",
        "model_only": True,
    }


def zabi_int_conversion(
    value: int, *, source: str = "i64", target: str = "u8"
) -> dict[str, Any]:
    """Wrap an integer into a destination integer type using two's complement."""
    number = _integer(value, "value")
    origin = _type_info(source, 8)
    destination = _type_info(target, 8)
    if origin["kind"] != "int" or destination["kind"] != "int":
        _fail("E001", "integer conversion requires integer type names")
    bits = destination["bits"]
    mask = (1 << bits) - 1
    wrapped = number & mask
    if destination["signed"] and wrapped >= (1 << (bits - 1)):
        wrapped -= 1 << bits
    return {
        "value": number,
        "source": origin["type"],
        "target": destination["type"],
        "wrapped": wrapped,
        "bits": bits,
        "overflow": wrapped != number,
    }


def zabi_layout_compare(
    left: Mapping[str, Any], right: Mapping[str, Any], *, word: int = 8
) -> dict[str, Any]:
    """Compare two layout mappings and enumerate their concrete differences."""
    width = _word(word)
    first = _mapping(left, "left layout")
    second = _mapping(right, "right layout")
    for record in (first, second):
        if "size" not in record or "align" not in record or "fields" not in record:
            _fail("E001", "layout must declare size, align and fields")
    a = _field_offsets(first["fields"], width)
    b = _field_offsets(second["fields"], width)
    size_a = _integer(first["size"], "left size", minimum=0)
    size_b = _integer(second["size"], "right size", minimum=0)
    align_a = _integer(first["align"], "left align", minimum=1)
    align_b = _integer(second["align"], "right align", minimum=1)
    differences: list[str] = []
    if size_a != size_b:
        differences.append("size")
    if align_a != align_b:
        differences.append("align")
    left_fields = {row["name"]: row for row in a["fields"]}
    right_fields = {row["name"]: row for row in b["fields"]}
    for name in sorted(set(left_fields) | set(right_fields)):
        if left_fields.get(name) != right_fields.get(name):
            differences.append(f"field:{name}")
    return {
        "equal": not differences,
        "size_delta": size_b - size_a,
        "align_delta": align_b - align_a,
        "differences": tuple(differences),
    }


def zabi_layout_fingerprint(layout: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint a layout mapping from its declared shape."""
    data = _mapping(layout, "layout")
    size = _integer(data.get("size"), "layout size", minimum=0)
    align = _integer(data.get("align"), "layout align", minimum=1)
    if "fields" not in data:
        _fail("E008", "malformed serialized layout: fields missing")
    names = tuple(
        _identifier(_items(item, "field")[0], "field name")
        for item in _items(data["fields"], "fields")
    )
    return {
        "fingerprint": _digest("layout", size, align, names),
        "size": size,
        "align": align,
        "fields": names,
        "digest_kind": "sha256-16",
    }


def zabi_opaque_address(
    address: int,
    *,
    state: Mapping[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Look up a raw address as an opaque key in caller-owned model state.

    The integer is never treated as a pointer: no load, store or protection
    change is performed, and a missing cell is reported as E006.
    """
    key = _integer(address, "address", minimum=0)
    marker = f"0x{key:x}"
    table = None if state is None else _mapping(state, "state")
    name = None if label is None else _text(label, "label")
    if table is None:
        return {
            "address": key,
            "key": marker,
            "present": False,
            "label": name,
            "state_size": 0,
            "model_only": True,
            "dereferenced": False,
        }
    if marker not in table:
        _fail("E006", f"no model state cell for address {marker}")
    cell = table[marker]
    return {
        "address": key,
        "key": marker,
        "present": True,
        "label": name,
        "state_size": len(table),
        "cell_type": type(cell).__name__,
        "model_only": True,
        "dereferenced": False,
    }


def zabi_param_class(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Classify a parameter as integer, float, pointer, slice or vector."""
    info = _type_info(type_name, _word(word))
    kind = info["kind"]
    if kind in ("int", "bool"):
        classification = "int"
    elif kind == "float":
        classification = "float"
    elif kind in ("pointer", "slice"):
        classification = "pointer"
    elif kind == "vector":
        classification = "vector"
    else:
        _fail("E001", "void cannot be used as a parameter type")
    return {
        "type": info["type"],
        "class": classification,
        "by_value": classification in ("int", "float"),
        "size": info["size"],
        "align": info["align"],
    }


def zabi_pointer_width(target: str = "native") -> dict[str, Any]:
    """Return the modeled pointer width of a target platform."""
    name = _target(target)
    bits = 32 if name == "wasm32" else 64
    return {
        "target": name,
        "bits": bits,
        "bytes": bits // 8,
        "model_only": True,
    }


def zabi_return_class(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Classify a result as void, register-returned or struct-returned."""
    width = _word(word)
    info = _type_info(type_name, width)
    if info["kind"] == "void":
        classification, registers = "void", 0
    elif info["kind"] == "slice":
        classification, registers = "reg", 2
    elif info["size"] <= 2 * width:
        classification, registers = "reg", 1
    else:
        classification, registers = "sret", 0
    return {
        "type": info["type"],
        "class": classification,
        "registers": registers,
        "size": info["size"],
        "align": info["align"],
        "model_only": True,
    }


def zabi_serialization_format(name: str, *, version: int = 1) -> dict[str, Any]:
    """Describe a supported model serialization format and version header."""
    format_name = _text(name, "format name")
    if format_name not in _FORMATS:
        _fail("E007", f"unsupported serialization format: {format_name!r}")
    number = _integer(version, "version", minimum=1)
    if number > 2:
        _fail("E008", f"unknown serialized format version: {number}")
    return {
        "name": format_name,
        "version": number,
        "endian": "little",
        "header": f"PGLY-{format_name}-{number}",
        "checksum": "crc32-model",
        "model_only": True,
    }


def zabi_shim_body(
    func_name: str,
    params: Sequence[tuple[str, str]],
    result: str,
    *,
    word: int = 8,
) -> dict[str, Any]:
    """Emit a C-ABI shim body that casts arguments to modeled Zig types."""
    width = _word(word)
    symbol = _identifier(func_name, "function name")
    output = _type_info(result, width)
    calls: list[str] = []
    names: list[str] = []
    for item in _items(params, "params"):
        pair = _items(item, "param")
        if len(pair) != 2:
            _fail("E003", "each param must be a (name, type) pair")
        arg = _identifier(pair[0], "parameter name")
        target = _type_info(pair[1], width)
        names.append(arg)
        if target["kind"] == "int":
            calls.append(f"@as({target['type']}, @intCast({arg}))")
        else:
            calls.append(arg)
    invocation = f"{symbol}({', '.join(calls)})"
    body = (
        f"    return {invocation};"
        if output["kind"] != "void"
        else f"    {invocation};"
    )
    lines = (
        f"export fn __cabi_{symbol}({', '.join(f'{name}: i64' for name in names)}) "
        f"callconv(.c) {output['type']} {{",
        body,
        "}",
    )
    return {
        "source": "\n".join(lines),
        "lines": lines,
        "symbol": f"__cabi_{symbol}",
        "arg_count": len(names),
        "result_type": output["type"],
    }


def zabi_sizeof(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Report the modeled byte size, bit width and class of a type."""
    info = _type_info(type_name, _word(word))
    return {
        "type": info["type"],
        "size": info["size"],
        "align": info["align"],
        "bits": info["bits"],
        "kind": info["kind"],
    }


def zabi_slice_repr(
    type_name: str, *, const: bool = True, word: int = 8
) -> dict[str, Any]:
    """Describe the pointer/length pair used to model a Zig slice."""
    width = _word(word)
    element = _type_info(type_name, width)
    constant = _boolean(const, "const")
    qualifier = "const " if constant else ""
    return {
        "expr": f"[]{qualifier}{element['type']}",
        "ptr_type": f"[*]{qualifier}{element['type']}",
        "len_type": "usize",
        "size": 2 * width,
        "align": width,
        "model_only": True,
    }


def zabi_struct_repr(
    name: str,
    fields: Sequence[tuple[str, str]],
    *,
    packed: bool = False,
    word: int = 8,
) -> dict[str, Any]:
    """Describe a packed or aligned external struct representation."""
    struct_name = _identifier(name, "struct name")
    is_packed = _boolean(packed, "packed")
    computed = _field_offsets(fields, _word(word))
    size = (
        sum(row["size"] for row in computed["fields"])
        if is_packed
        else computed["size"]
    )
    align = 1 if is_packed else computed["align"]
    return {
        "name": struct_name,
        "expr": "packed struct" if is_packed else "extern struct",
        "size": size,
        "align": align,
        "fields": computed["fields"],
        "padding": computed["padding"] if not is_packed else 0,
        "model_only": True,
    }


def zabi_symbol_name(
    name: str, *, namespace: str = "polyglot", abi: str = "c", version: int = 1
) -> dict[str, Any]:
    """Build a stable exported symbol name for a function and ABI."""
    symbol = _identifier(name, "function name")
    space = _identifier(namespace, "namespace")
    kind = _text(abi, "abi")
    if kind not in ("c", "zig"):
        _fail("E003", f"unknown ABI: {kind!r}")
    number = _integer(version, "version", minimum=1)
    mangled = (
        f"__cabi_{space}_{symbol}_v{number}" if kind == "c" else f"{space}.{symbol}"
    )
    return {
        "symbol": symbol,
        "namespace": space,
        "abi": kind,
        "version": number,
        "mangled": mangled,
    }


def zabi_target_spec(target: str = "native") -> dict[str, Any]:
    """Return the complete ABI model specification for one target."""
    name = _target(target)
    word = 4 if name == "wasm32" else 8
    abis = ("c",) if name in ("wasm32", "wasm64") else ("c", "fast", "zig")
    features = ("cabi",) if name in ("wasm32", "wasm64") else ("simd", "wide_int")
    return {
        "target": name,
        "word": word,
        "pointer_bits": word * 8,
        "max_align": 8 if word == 4 else 16,
        "endian": "little",
        "abis": abis,
        "features": features,
        "model_only": True,
    }


def zabi_type_class(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Classify a type for ABI lowering decisions."""
    info = _type_info(type_name, _word(word))
    return {
        "type": info["type"],
        "class": info["kind"],
        "size": info["size"],
        "align": info["align"],
        "signed": info["signed"],
        "model_only": True,
    }
