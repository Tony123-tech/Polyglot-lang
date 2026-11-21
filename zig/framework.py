"""Deterministic, model-only framework helpers for the Zig backend rollout.

This module supplies the planning, layout, caching, diagnostic and validation
surface used by the batch-6 Zig framework functions.  Every value is computed
from its arguments with the standard library only: no Zig compiler is invoked,
no executable memory is prepared and raw addresses are never dereferenced.
All failures are raised as :class:`PolyglotError` with the E001-E008 surface.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "zf_build_profile",
    "zf_cache_entry",
    "zf_cache_evict",
    "zf_cache_key",
    "zf_cache_lookup",
    "zf_dependency_graph",
    "zf_dependency_order",
    "zf_diagnostic",
    "zf_diagnostic_format",
    "zf_dispatch_table",
    "zf_emit_context",
    "zf_error_surface",
    "zf_escape_identifier",
    "zf_escape_string",
    "zf_feature_gate",
    "zf_int_bounds",
    "zf_integer_widths",
    "zf_layout_align",
    "zf_layout_offsets",
    "zf_layout_size",
    "zf_layout_validate",
    "zf_module_fingerprint",
    "zf_module_io",
    "zf_module_manifest",
    "zf_module_plan",
    "zf_module_stats",
    "zf_name_mangle",
    "zf_op_cost_model",
    "zf_optimize_plan",
    "zf_parallel_plan",
    "zf_pass_pipeline",
    "zf_pass_trace",
    "zf_resource_budget",
    "zf_saturating_model",
    "zf_source_map",
    "zf_symbol_registry",
    "zf_target_model",
    "zf_type_canonical",
    "zf_type_layout",
    "zf_validation_report",
]

_TARGETS: Final[frozenset[str]] = frozenset(
    {"aarch64", "native", "wasm32", "wasm64", "x86_64"}
)
_PROFILES: Final[frozenset[str]] = frozenset(
    {"debug", "release_fast", "release_safe", "release_small"}
)
_FEATURES: Final[frozenset[str]] = frozenset(
    {"cabi", "parallel", "simd", "saturating", "wide_int"}
)
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VECTOR: Final[re.Pattern[str]] = re.compile(r"^@Vector\((\d+),\s*([A-Za-z0-9_]+)\)$")
_CODE: Final[re.Pattern[str]] = re.compile(r"^E00[1-8]$")
_LANES: Final[frozenset[int]] = frozenset({2, 4, 8, 16, 32, 64})
_MAX_INTEGER_BITS: Final[int] = 65535
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "addrspace",
        "align",
        "and",
        "break",
        "callconv",
        "catch",
        "comptime",
        "const",
        "continue",
        "defer",
        "else",
        "enum",
        "errdefer",
        "export",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "inline",
        "null",
        "or",
        "orelse",
        "packed",
        "pub",
        "return",
        "struct",
        "switch",
        "test",
        "this",
        "threadlocal",
        "true",
        "try",
        "type",
        "undef",
        "undefined",
        "union",
        "usingnamespace",
        "var",
        "volatile",
        "while",
    }
)
_ERROR_MEANINGS: Final[dict[str, tuple[str, bool, bool]]] = {
    "E001": ("invalid layout or literal input", True, False),
    "E002": ("unmet source obligation", False, False),
    "E003": ("framework API misuse", True, False),
    "E004": ("model toolchain failure", True, True),
    "E005": ("dependency cycle", False, True),
    "E006": ("missing model artifact", True, True),
    "E007": ("unsupported platform feature", False, False),
    "E008": ("malformed serialized state", False, False),
}
_OPTIMIZATIONS: Final[dict[str, int]] = {
    "bounds_elim": 3,
    "constant_fold": 0,
    "dead_code": 1,
    "inline": 2,
}
_PASSES: Final[dict[str, int]] = {"emit": 3, "lower": 1, "opt": 2, "parse": 0}
_PROFILE_FLAGS: Final[dict[str, tuple[str, bool, bool]]] = {
    "debug": ("none", False, True),
    "release_fast": ("speed", True, False),
    "release_safe": ("safety", False, False),
    "release_small": ("size", True, False),
}
_OPTIMIZE_FLAGS: Final[dict[str, str]] = {
    "none": "-ODebug",
    "safety": "-OReleaseSafe",
    "size": "-OReleaseSmall",
    "speed": "-OReleaseFast",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured Zig-framework diagnostic."""
    raise PolyglotError(code, message, reason=reason, phase="zig-framework")


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
    """Return a bool or raise E003 when the caller supplied another type."""
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
    """Validate a Zig identifier, rejecting reserved words with E001."""
    name = _text(value, label)
    if not _IDENTIFIER.fullmatch(name) or name in _RESERVED:
        _fail("E001", f"{label} is not a legal Zig identifier: {name!r}")
    return name


def _word(value: object) -> int:
    """Validate a modeled target word size (4 or 8 bytes)."""
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


def _profile(value: object) -> str:
    """Validate a build profile name."""
    name = _text(value, "profile")
    if name not in _PROFILES:
        _fail("E003", f"unknown build profile: {name!r}")
    return name


def _target(value: object) -> str:
    """Validate a model target name, raising E007 for unknown platforms."""
    name = _text(value, "target")
    if name not in _TARGETS:
        _fail("E007", f"unsupported target: {name!r}")
    return name


def _signature(value: object, label: str) -> str:
    """Normalize whitespace in a symbolic signature string."""
    raw = _text(value, label)
    if not raw.strip():
        _fail("E003", f"{label} must not be empty")
    return re.sub(r"\s+", "", raw)


def _scalar(name: str, word: int) -> dict[str, Any] | None:
    """Return the modeled layout of a primitive, or ``None`` if unknown."""
    if name in ("isize", "usize"):
        return {
            "type": name,
            "kind": "int",
            "size": word,
            "align": word,
            "bits": word * 8,
            "signed": name == "isize",
            "element": None,
            "width": 1,
        }
    integer = re.fullmatch(r"([iu])(\d+)", name)
    if integer is not None:
        bits = int(integer.group(2))
        if not 1 <= bits <= _MAX_INTEGER_BITS:
            return None
        size = max(1, (bits + 7) // 8)
        return {
            "type": name,
            "kind": "int",
            "size": size,
            "align": _pow2_ceil(size),
            "bits": bits,
            "signed": integer.group(1) == "i",
            "element": None,
            "width": 1,
        }
    if name == "bool":
        return {
            "type": "bool",
            "kind": "bool",
            "size": 1,
            "align": 1,
            "bits": 1,
            "signed": None,
            "element": None,
            "width": 1,
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
            "element": None,
            "width": 1,
        }
    return None


def _layout(name: str, word: int) -> dict[str, Any]:
    """Resolve a modeled Zig type into a fresh layout record."""
    scalar = _scalar(name, word)
    if scalar is not None:
        return scalar
    if name.startswith("[*]"):
        tail = name[3:].strip()
        constant = tail.startswith("const ")
        inner = tail[6:].strip() if constant else tail
        element = _layout(inner, word)
        if element["kind"] == "vector":
            _fail("E001", "pointers to vectors are not modeled")
        qualifier = "const " if constant else ""
        return {
            "type": f"[*]{qualifier}{element['type']}",
            "kind": "pointer",
            "size": word,
            "align": word,
            "bits": word * 8,
            "signed": None,
            "element": element,
            "width": 1,
        }
    vector = _VECTOR.fullmatch(name)
    if vector is not None:
        lanes = int(vector.group(1))
        if lanes not in _LANES:
            _fail("E001", f"unsupported vector lane count: {lanes}")
        element = _layout(vector.group(2), word)
        if element["kind"] == "float":
            _fail("E007", "float vector layouts are not modeled")
        if element["kind"] not in ("bool", "int"):
            _fail("E001", "vector elements must be integer types")
        return {
            "type": f"@Vector({lanes}, {element['type']})",
            "kind": "vector",
            "size": element["size"] * lanes,
            "align": element["align"],
            "bits": element["size"] * lanes * 8,
            "signed": element["signed"],
            "element": element,
            "width": lanes,
        }
    _fail("E001", f"unsupported Zig type: {name!r}")


def _fields(fields: object, word: int) -> dict[str, Any]:
    """Compute deterministic field offsets, size, alignment and padding."""
    rows: list[dict[str, Any]] = []
    offset = 0
    maximum = 1
    padding = 0
    for field in _items(fields, "fields"):
        pair = _items(field, "field")
        if len(pair) != 2:
            _fail("E003", "each field must be a (name, type) pair")
        name = _identifier(pair[0], "field name")
        layout = _layout(_text(pair[1], "field type"), word)
        gap = (-offset) % layout["align"]
        offset += gap
        padding += gap
        rows.append(
            {
                "name": name,
                "type": layout["type"],
                "offset": offset,
                "size": layout["size"],
                "align": layout["align"],
            }
        )
        offset += layout["size"]
        maximum = max(maximum, layout["align"])
    total = _align_up(offset, maximum)
    padding += total - offset
    return {"fields": tuple(rows), "size": total, "align": maximum, "padding": padding}


# ---------------------------------------------------------------------------
# Public framework functions
# ---------------------------------------------------------------------------


def zf_build_profile(
    profile: str,
    *,
    optimize: str | None = None,
    strip: bool | None = None,
    debug_symbols: bool | None = None,
) -> dict[str, Any]:
    """Model Zig build flags for a named profile with optional overrides."""
    name = _profile(profile)
    default_optimize, default_strip, default_debug = _PROFILE_FLAGS[name]
    mode = default_optimize if optimize is None else _text(optimize, "optimize")
    if mode not in _OPTIMIZE_FLAGS:
        _fail("E003", f"unknown optimization mode: {mode!r}")
    stripped = default_strip if strip is None else _boolean(strip, "strip")
    debug = (
        default_debug
        if debug_symbols is None
        else _boolean(debug_symbols, "debug_symbols")
    )
    flags = [_OPTIMIZE_FLAGS[mode]]
    if stripped:
        flags.append("-fstrip")
    if debug:
        flags.append("-g")
    return {
        "profile": name,
        "optimize": mode,
        "strip": stripped,
        "debug_symbols": debug,
        "flags": tuple(flags),
        "model_only": True,
    }


def zf_cache_entry(
    key: str, *, source: str, artifacts: Sequence[str] = ()
) -> dict[str, Any]:
    """Build a deterministic cache entry record for one source snapshot."""
    name = _text(key, "key")
    if not name:
        _fail("E003", "cache key must not be empty")
    text = _text(source, "source")
    names = tuple(_text(item, "artifact") for item in _items(artifacts, "artifacts"))
    return {
        "key": name,
        "source_hash": _digest("source", text),
        "artifacts": names,
        "byte_length": len(text.encode("utf-8")),
        "fingerprint": _digest("entry", name, text, names),
    }


def zf_cache_evict(
    cache: dict[str, Any], key: str, *, max_entries: int = 8
) -> dict[str, Any]:
    """Remove one key and trim insertion-order overflow from a model cache."""
    if not isinstance(cache, dict):
        _fail("E003", "cache must be a mutable dict")
    name = _text(key, "key")
    limit = _integer(max_entries, "max_entries", minimum=1)
    removed = cache.pop(name, None) is not None
    evicted: list[str] = []
    while len(cache) > limit:
        oldest = next(iter(cache))
        del cache[oldest]
        evicted.append(oldest)
    return {
        "removed": removed,
        "evicted": tuple(evicted),
        "remaining": tuple(cache),
        "size": len(cache),
    }


def zf_cache_key(
    module: str, *, source: str, profile: str = "release_fast", target: str = "native"
) -> dict[str, Any]:
    """Derive a stable cache key from module, source, profile and target."""
    name = _identifier(module, "module")
    text = _text(source, "source")
    mode = _profile(profile)
    platform = _target(target)
    key = _digest("cache", name, text, mode, platform)
    return {
        "key": key,
        "module": name,
        "profile": mode,
        "target": platform,
        "length": len(key),
    }


def zf_cache_lookup(cache: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Look up a cache key without mutating the supplied model mapping."""
    table = _mapping(cache, "cache")
    name = _text(key, "key")
    if name not in table:
        return {"key": name, "hit": False, "entry": None, "checked": len(table)}
    return {"key": name, "hit": True, "entry": table[name], "checked": len(table)}


def zf_dependency_graph(edges: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Build a normalized provider-to-consumer dependency graph."""
    adjacency: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    normalized: list[tuple[str, str]] = []
    for edge in _items(edges, "edges"):
        pair = _items(edge, "edge")
        if len(pair) != 2:
            _fail("E003", "each edge must be a (provider, consumer) pair")
        provider = _identifier(pair[0], "provider")
        consumer = _identifier(pair[1], "consumer")
        adjacency.setdefault(provider, [])
        adjacency.setdefault(consumer, [])
        indegree.setdefault(provider, 0)
        indegree.setdefault(consumer, 0)
        adjacency[provider].append(consumer)
        indegree[consumer] += 1
        normalized.append((provider, consumer))
    nodes = tuple(sorted(indegree))
    return {
        "nodes": nodes,
        "edges": tuple(sorted(normalized)),
        "indegree": {node: indegree[node] for node in nodes},
        "adjacency": {node: tuple(sorted(adjacency.get(node, ()))) for node in nodes},
    }


def zf_dependency_order(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Topologically order a graph, raising E005 when a cycle exists."""
    data = _mapping(graph, "graph")
    if "edges" not in data:
        _fail("E003", "graph must contain an 'edges' key")
    indegree: dict[str, int] = {}
    successors: dict[str, list[str]] = {}
    for edge in _items(data["edges"], "graph edges"):
        pair = _items(edge, "edge")
        if len(pair) != 2:
            _fail("E008", "malformed serialized graph edge")
        provider = _identifier(pair[0], "provider")
        consumer = _identifier(pair[1], "consumer")
        indegree.setdefault(provider, 0)
        indegree.setdefault(consumer, 0)
        successors.setdefault(provider, []).append(consumer)
        indegree[consumer] += 1
    ready = sorted(node for node in indegree if indegree[node] == 0)
    order: list[str] = []
    levels: list[tuple[str, ...]] = []
    while ready:
        levels.append(tuple(ready))
        following: list[str] = []
        for node in ready:
            order.append(node)
            for consumer in successors.get(node, ()):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    following.append(consumer)
        ready = sorted(following)
    if len(order) != len(indegree):
        _fail("E005", "dependency cycle detected")
    return {
        "order": tuple(order),
        "levels": tuple(levels),
        "acyclic": True,
        "count": len(order),
    }


def zf_diagnostic(
    code: str, message: str, *, line: int = 1, col: int = 1, hint: str | None = None
) -> dict[str, Any]:
    """Create one structured E001-E008 diagnostic with a rendered form."""
    symbol = _text(code, "code")
    if not _CODE.fullmatch(symbol):
        _fail("E003", f"invalid diagnostic code: {symbol!r}")
    text = _text(message, "message")
    row = _integer(line, "line", minimum=1)
    column = _integer(col, "col", minimum=1)
    note = None if hint is None else _text(hint, "hint")
    return {
        "code": symbol,
        "message": text,
        "line": row,
        "col": column,
        "hint": note,
        "rendered": f"error[{symbol}]: {text}",
        "location": f"{row}:{column}",
    }


def zf_diagnostic_format(
    diagnostics: Sequence[Mapping[str, Any]], *, color: bool = False
) -> dict[str, Any]:
    """Render diagnostic records into stable text, optionally with ANSI red."""
    use_color = _boolean(color, "color")
    lines: list[str] = []
    codes: list[str] = []
    for item in _items(diagnostics, "diagnostics"):
        record = _mapping(item, "diagnostic")
        code = _text(record.get("code"), "code")
        message = _text(record.get("message"), "message")
        codes.append(code)
        head = f"\x1b[31merror[{code}]\x1b[0m" if use_color else f"error[{code}]"
        lines.append(f"{head}: {message}")
    return {
        "lines": tuple(lines),
        "text": "\n".join(lines),
        "count": len(lines),
        "codes": tuple(codes),
        "colored": use_color,
    }


def zf_dispatch_table(
    entries: Sequence[tuple[str, str]], *, deduplicate: bool = True
) -> dict[str, Any]:
    """Assign stable slots to exported symbols with signature collision checks."""
    collapse = _boolean(deduplicate, "deduplicate")
    slots: list[dict[str, Any]] = []
    by_symbol: dict[str, dict[str, Any]] = {}
    collisions: list[str] = []
    for entry in _items(entries, "entries"):
        pair = _items(entry, "entry")
        if len(pair) != 2:
            _fail("E003", "each dispatch entry must be a (symbol, signature) pair")
        symbol = _identifier(pair[0], "symbol")
        signature = _signature(pair[1], "signature")
        if symbol in by_symbol:
            if by_symbol[symbol]["signature"] != signature:
                _fail("E001", f"conflicting signatures for symbol {symbol!r}")
            if collapse:
                continue
            collisions.append(symbol)
        slot = {
            "index": len(slots),
            "symbol": symbol,
            "signature": signature,
        }
        slots.append(slot)
        by_symbol[symbol] = slot
    return {
        "slots": slots,
        "count": len(slots),
        "by_symbol": by_symbol,
        "collisions": tuple(collisions),
    }


def zf_emit_context(
    name: str, *, indent: int = 0, target: str = "native"
) -> dict[str, Any]:
    """Create a fresh source-emission context for one model module."""
    module = _identifier(name, "module")
    width = _integer(indent, "indent", minimum=0)
    platform = _target(target)
    return {
        "name": module,
        "indent": width,
        "pad": " " * width,
        "target": platform,
        "lines": [],
        "helpers": [],
        "model_only": True,
    }


def zf_error_surface(code: str, *, operation: str) -> dict[str, Any]:
    """Map a diagnostic code to recoverability and retry metadata."""
    symbol = _text(code, "code")
    if symbol not in _ERROR_MEANINGS:
        _fail("E003", f"invalid diagnostic code: {symbol!r}")
    action = _text(operation, "operation")
    meaning, recoverable, retryable = _ERROR_MEANINGS[symbol]
    return {
        "code": symbol,
        "operation": action,
        "meaning": meaning,
        "recoverable": recoverable,
        "retryable": retryable,
        "phase": "zig",
    }


def zf_escape_identifier(name: str) -> dict[str, Any]:
    """Map arbitrary text to a legal Zig identifier without collisions."""
    value = _text(name, "name")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    reserved = cleaned in _RESERVED
    escaped = f"_{cleaned}" if reserved else cleaned
    return {
        "name": value,
        "escaped": escaped,
        "changed": escaped != value,
        "reserved": reserved,
    }


def zf_escape_string(value: str) -> dict[str, Any]:
    """Render text as a deterministic Zig string literal with escapes."""
    text = _text(value, "value")
    pieces: list[str] = []
    escapes: list[tuple[str, str]] = []
    for ch in text:
        point = ord(ch)
        if ch == "\\":
            rendered = "\\\\"
        elif ch == '"':
            rendered = '\\"'
        elif ch == "\n":
            rendered = "\\n"
        elif ch == "\r":
            rendered = "\\r"
        elif ch == "\t":
            rendered = "\\t"
        elif point < 32 or point == 127:
            rendered = f"\\x{point:02x}"
        elif point > 126:
            rendered = f"\\u{{{point:x}}}"
        else:
            rendered = ch
        pieces.append(rendered)
        if len(rendered) > 1:
            escapes.append((ch, rendered))
    literal = '"' + "".join(pieces) + '"'
    return {
        "value": text,
        "literal": literal,
        "escapes": tuple(escapes),
        "byte_length": len(text.encode("utf-8")),
    }


def zf_feature_gate(
    feature: str, *, target: str = "native", profile: str = "debug"
) -> dict[str, Any]:
    """Report whether a Zig feature is enabled for a target and profile."""
    name = _text(feature, "feature")
    if name not in _FEATURES:
        _fail("E003", f"unknown Zig feature: {name!r}")
    platform = _target(target)
    mode = _profile(profile)
    if name in ("parallel", "simd") and platform in ("wasm32", "wasm64"):
        _fail("E007", f"feature {name!r} is unavailable on {platform}")
    if name == "parallel" and mode == "debug":
        return {
            "feature": name,
            "target": platform,
            "profile": mode,
            "enabled": False,
            "reason": "debug builds omit the parallel helper library",
            "code": None,
        }
    if name == "wide_int" and mode == "release_small":
        return {
            "feature": name,
            "target": platform,
            "profile": mode,
            "enabled": False,
            "reason": "size-optimized builds omit wide integer helpers",
            "code": None,
        }
    return {
        "feature": name,
        "target": platform,
        "profile": mode,
        "enabled": True,
        "reason": None,
        "code": None,
    }


def zf_int_bounds(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Return inclusive modeled bounds of a Zig integer type."""
    layout = _layout(_text(type_name, "type name"), _word(word))
    if layout["kind"] != "int":
        _fail("E001", f"not an integer type: {type_name!r}")
    bits = layout["bits"]
    if layout["signed"]:
        low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        low, high = 0, (1 << bits) - 1
    return {
        "type": layout["type"],
        "min": low,
        "max": high,
        "bits": bits,
        "signed": layout["signed"],
    }


def zf_integer_widths(*, word: int = 8) -> dict[str, Any]:
    """List the modeled integer widths and their byte sizes."""
    width = _word(word)
    bits = (1, 2, 4, 8, 16, 32, 64, 128)
    return {
        "word": width,
        "widths": bits,
        "byte_sizes": tuple(max(1, (value + 7) // 8) for value in bits),
        "max_lanes": max(_LANES),
        "model_only": True,
    }


def zf_layout_align(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Report the modeled alignment of a Zig type."""
    layout = _layout(_text(type_name, "type name"), _word(word))
    return {
        "type": layout["type"],
        "align": layout["align"],
        "size": layout["size"],
        "kind": layout["kind"],
    }


def zf_layout_offsets(
    fields: Sequence[tuple[str, str]], *, word: int = 8
) -> dict[str, Any]:
    """Compute padded field offsets for a modeled Zig struct."""
    return _fields(fields, _word(word))


def zf_layout_size(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Report the modeled byte size and bit width of a Zig type."""
    layout = _layout(_text(type_name, "type name"), _word(word))
    return {
        "type": layout["type"],
        "size": layout["size"],
        "align": layout["align"],
        "bits": layout["bits"],
    }


def zf_layout_validate(layout: Mapping[str, Any], *, word: int = 8) -> dict[str, Any]:
    """Check declared struct size/alignment against the computed model."""
    data = _mapping(layout, "layout")
    if "fields" not in data:
        _fail("E002", "layout must declare its fields")
    name = _identifier(data.get("name"), "layout name")
    computed = _fields(data["fields"], _word(word))
    declared_size = data.get("size", computed["size"])
    declared_align = data.get("align", computed["align"])
    if isinstance(declared_size, bool) or not isinstance(declared_size, int):
        _fail("E008", "malformed layout size")
    if isinstance(declared_align, bool) or not isinstance(declared_align, int):
        _fail("E008", "malformed layout alignment")
    if declared_size != computed["size"] or declared_align != computed["align"]:
        _fail("E001", f"layout mismatch for {name!r}")
    return {
        "name": name,
        "size": computed["size"],
        "align": computed["align"],
        "offsets": computed["fields"],
        "padding": computed["padding"],
        "valid": True,
    }


def zf_module_fingerprint(module: str, *, sources: Mapping[str, str]) -> dict[str, Any]:
    """Fingerprint a module from its named sources in canonical file order."""
    name = _identifier(module, "module")
    data = _mapping(sources, "sources")
    pairs: list[tuple[str, str]] = []
    for filename in sorted(data):
        label = _text(filename, "filename")
        content = _text(data[filename], "source content")
        pairs.append((label, content))
    canonical = "\n".join(
        f"{label}:{len(content)}:{_digest(content)}" for label, content in pairs
    )
    return {
        "module": name,
        "fingerprint": _digest("module", name, canonical),
        "files": tuple(label for label, _ in pairs),
        "byte_total": sum(len(content.encode("utf-8")) for _, content in pairs),
        "digest_kind": "sha256-16",
    }


def zf_module_io(
    module: str, *, inputs: Sequence[str] = (), outputs: Sequence[str] = ()
) -> dict[str, Any]:
    """Describe the modeled input/output interface of a Zig module."""
    name = _identifier(module, "module")
    incoming = tuple(_text(item, "input") for item in _items(inputs, "inputs"))
    outgoing = tuple(_text(item, "output") for item in _items(outputs, "outputs"))
    return {
        "module": name,
        "inputs": incoming,
        "outputs": outgoing,
        "interface": tuple(sorted(set(incoming) | set(outgoing))),
        "arity": (len(incoming), len(outgoing)),
        "fingerprint": _digest("io", name, incoming, outgoing),
    }


def zf_module_manifest(
    plan: Mapping[str, Any], *, layout: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize a module plan into stable manifest sections."""
    data = _mapping(plan, "plan")
    required = ("module", "target", "profile", "features")
    missing = [key for key in required if key not in data]
    if missing:
        _fail("E002", f"module plan is missing: {', '.join(missing)}")
    name = _identifier(data["module"], "module")
    platform = _text(data["target"], "target")
    mode = _text(data["profile"], "profile")
    features = tuple(
        _text(item, "feature") for item in _items(data["features"], "features")
    )
    size = None
    if layout is not None:
        size = _integer(
            _mapping(layout, "layout").get("size"), "layout size", minimum=0
        )
    sections = ("header", "types", "functions", "exports")
    if size is not None:
        sections += ("layout",)
    return {
        "module": name,
        "target": platform,
        "profile": mode,
        "features": features,
        "layout_size": size,
        "sections": sections,
        "fingerprint": _digest("manifest", name, platform, mode, features, size),
    }


def zf_module_plan(
    name: str,
    *,
    target: str = "native",
    profile: str = "release_fast",
    features: Sequence[str] = (),
) -> dict[str, Any]:
    """Plan a model Zig module with validated target, profile and features."""
    module = _identifier(name, "module")
    platform = _target(target)
    mode = _profile(profile)
    selected: list[str] = []
    for item in _items(features, "features"):
        feature = _text(item, "feature")
        if feature not in _FEATURES:
            _fail("E007", f"unknown Zig feature: {feature!r}")
        if feature not in selected:
            selected.append(feature)
    chosen = tuple(sorted(selected))
    flags = ("-femit-bin",) if "cabi" in chosen else ()
    imports = ("std",) if chosen else ()
    return {
        "module": module,
        "target": platform,
        "profile": mode,
        "features": chosen,
        "imports": imports,
        "flags": flags,
        "output_name": f"{module}.zig",
        "fingerprint": _digest("plan", module, platform, mode, chosen),
        "model_only": True,
    }


def zf_module_stats(source: str) -> dict[str, Any]:
    """Measure lines, bytes and widths of a Zig source string."""
    text = _text(source, "source")
    lines = text.split("\n")
    return {
        "lines": len(lines),
        "blank_lines": sum(not line.strip() for line in lines),
        "characters": len(text),
        "bytes": len(text.encode("utf-8")),
        "max_line_length": max((len(line) for line in lines), default=0),
        "trailing_newline": text.endswith("\n"),
    }


def zf_name_mangle(
    symbol: str, *, namespace: str = "polyglot", abi: str = "c"
) -> dict[str, Any]:
    """Mangle a symbol for the modeled C or Zig ABI namespace."""
    name = _identifier(symbol, "symbol")
    space = _identifier(namespace, "namespace")
    kind = _text(abi, "abi")
    if kind not in ("c", "zig"):
        _fail("E003", f"unknown ABI: {kind!r}")
    mangled = f"{space}__{name}" if kind == "c" else f"{space}.{name}"
    return {
        "symbol": name,
        "namespace": space,
        "abi": kind,
        "mangled": mangled,
        "length": len(mangled),
    }


def zf_op_cost_model(
    op: str, *, type_name: str = "i64", lanes: int = 1
) -> dict[str, Any]:
    """Estimate the deterministic model cost of an integer operation."""
    name = _text(op, "op")
    costs = {
        "add": 1,
        "div": 8,
        "load": 2,
        "mod": 10,
        "mul": 3,
        "reduce": 4,
        "shuffle": 3,
        "store": 2,
        "sub": 1,
    }
    if name not in costs:
        _fail("E003", f"unknown operation: {name!r}")
    count = _integer(lanes, "lanes", minimum=1)
    layout = _layout(_text(type_name, "type name"), 8)
    vectorized = name in ("add", "div", "mod", "mul", "reduce", "sub")
    return {
        "op": name,
        "type": layout["type"],
        "lanes": count,
        "base_cost": costs[name],
        "cost": costs[name] * (count if vectorized else 1),
        "model_only": True,
    }


def zf_optimize_plan(
    source_size: int, *, profile: str = "release_fast", passes: Sequence[str] = ()
) -> dict[str, Any]:
    """Order optimization passes and estimate their model output size."""
    size = _integer(source_size, "source_size", minimum=0)
    if size > 10_000_000:
        _fail("E004", "model optimizer source size limit exceeded")
    mode = _profile(profile)
    selected: list[str] = []
    for item in _items(passes, "passes"):
        name = _text(item, "pass")
        if name not in _OPTIMIZATIONS:
            _fail("E007", f"unknown optimization pass: {name!r}")
        if name not in selected:
            selected.append(name)
    selected.sort(key=lambda name: _OPTIMIZATIONS[name])
    estimate = size // 2 if selected else size
    return {
        "profile": mode,
        "source_size": size,
        "passes": tuple(selected),
        "estimated_size": estimate,
        "fingerprint": _digest("opt", size, mode, tuple(selected)),
    }


def zf_parallel_plan(
    workers: int, *, items: int, chunk_size: int = 4
) -> dict[str, Any]:
    """Partition model work items into deterministic contiguous chunks."""
    count = _integer(workers, "workers", minimum=1)
    size = _integer(items, "items", minimum=0)
    step = _integer(chunk_size, "chunk_size", minimum=1)
    chunks = tuple((start, min(start + step, size)) for start in range(0, size, step))
    used = min(count, len(chunks))
    return {
        "workers": count,
        "items": size,
        "chunk_size": step,
        "chunks": chunks,
        "used_workers": used,
        "balanced": size % (count * step) == 0,
    }


def zf_pass_pipeline(passes: Sequence[str] = ()) -> dict[str, Any]:
    """Order compiler passes into the canonical parse/lower/opt/emit pipeline."""
    selected: list[str] = []
    for item in _items(passes, "passes"):
        name = _text(item, "pass")
        if name not in _PASSES:
            _fail("E007", f"unknown compiler pass: {name!r}")
        if name in selected:
            _fail("E003", f"duplicate compiler pass: {name!r}")
        selected.append(name)
    order = tuple(sorted(selected, key=lambda name: _PASSES[name]))
    return {
        "passes": tuple(selected),
        "order": order,
        "count": len(order),
        "fingerprint": _digest("pipeline", order),
    }


def zf_pass_trace(
    pipeline: Mapping[str, Any], *, source_size: int = 0
) -> dict[str, Any]:
    """Trace deterministic input/output sizes through a pass pipeline."""
    data = _mapping(pipeline, "pipeline")
    order = data.get("order", data.get("passes"))
    if order is None:
        _fail("E003", "pipeline must contain 'order' or 'passes'")
    current = _integer(source_size, "source_size", minimum=0)
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(_items(order, "pass order")):
        name = _text(item, "pass")
        if name == "opt":
            produced = current - current // 10
        else:
            produced = current
        steps.append(
            {
                "pass": name,
                "input_size": current,
                "output_size": produced,
                "note": f"step {index + 1}",
            }
        )
        current = produced
    return {
        "steps": tuple(steps),
        "total_steps": len(steps),
        "final_size": current,
    }


def zf_resource_budget(
    *, max_memory: int = 64, max_symbols: int = 128, max_lines: int = 100_000
) -> dict[str, Any]:
    """Describe the model-only resource limits for one Zig backend run."""
    memory = _integer(max_memory, "max_memory", minimum=1)
    symbols = _integer(max_symbols, "max_symbols", minimum=1)
    lines = _integer(max_lines, "max_lines", minimum=1)
    return {
        "max_memory_mb": memory,
        "max_symbols": symbols,
        "max_lines": lines,
        "limits": ("lines", "memory", "symbols"),
        "model_only": True,
    }


def zf_saturating_model(
    op: str, left: int, right: int, *, type_name: str = "i8"
) -> dict[str, Any]:
    """Evaluate a saturating integer operation against modeled type bounds."""
    symbol = _text(op, "op")
    if symbol not in ("+|", "-|", "*|"):
        _fail("E003", f"unsupported saturating operator: {symbol!r}")
    first = _integer(left, "left")
    second = _integer(right, "right")
    bounds = zf_int_bounds(type_name)
    raw = {
        "+|": first + second,
        "-|": first - second,
        "*|": first * second,
    }[symbol]
    value = max(bounds["min"], min(bounds["max"], raw))
    return {
        "op": symbol,
        "left": first,
        "right": second,
        "raw": raw,
        "value": value,
        "type": bounds["type"],
        "saturated": value != raw,
    }


def zf_source_map(source: str, *, origin: str = "input.pg") -> dict[str, Any]:
    """Map each source line to character and UTF-8 byte spans."""
    text = _text(source, "source")
    name = _text(origin, "origin")
    rows: list[dict[str, Any]] = []
    char_offset = 0
    byte_offset = 0
    for number, line in enumerate(text.split("\n"), start=1):
        char_end = char_offset + len(line)
        byte_end = byte_offset + len(line.encode("utf-8"))
        rows.append(
            {
                "index": number,
                "text": line,
                "start": char_offset,
                "end": char_end,
                "byte_start": byte_offset,
                "byte_end": byte_end,
            }
        )
        char_offset = char_end + 1
        byte_offset = byte_end + 1
    return {
        "origin": name,
        "lines": tuple(rows),
        "total_lines": len(rows),
        "total_bytes": len(text.encode("utf-8")),
    }


def zf_symbol_registry(
    specs: Sequence[tuple[str, int, str]] = (),
) -> dict[str, Any]:
    """Register exported symbols with arity and result type metadata."""
    by_name: dict[str, dict[str, Any]] = {}
    for item in _items(specs, "specs"):
        triple = _items(item, "spec")
        if len(triple) != 3:
            _fail("E003", "each spec must be a (name, arity, result) triple")
        name = _identifier(triple[0], "symbol")
        arity = _integer(triple[1], "arity", minimum=0)
        result = _text(triple[2], "result type")
        record = {"arity": arity, "result": result}
        if name in by_name and by_name[name] != record:
            _fail("E003", f"conflicting registry entry for {name!r}")
        by_name[name] = record
    names = tuple(sorted(by_name))
    return {
        "names": names,
        "by_name": {name: dict(by_name[name]) for name in names},
        "count": len(names),
        "fingerprint": _digest(
            "registry", names, tuple(by_name[name]["arity"] for name in names)
        ),
    }


def zf_target_model(target: str = "native") -> dict[str, Any]:
    """Describe the modeled word size, endianness and feature set of a target."""
    name = _target(target)
    word = 4 if name == "wasm32" else 8
    if name in ("wasm32", "wasm64"):
        features = ("cabi",)
    else:
        features = ("cabi", "parallel", "saturating", "simd", "wide_int")
    return {
        "target": name,
        "word": word,
        "endian": "little",
        "pointer_align": word,
        "max_align": 8 if word == 4 else 16,
        "features": features,
        "model_only": True,
    }


def zf_type_canonical(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Canonicalize a modeled Zig type string and report its layout class."""
    input_name = _text(type_name, "type name")
    layout = _layout(input_name, _word(word))
    return {
        "input": input_name,
        "canonical": layout["type"],
        "kind": layout["kind"],
        "size": layout["size"],
        "align": layout["align"],
    }


def zf_type_layout(type_name: str, *, word: int = 8) -> dict[str, Any]:
    """Return a fresh, structured layout record for a modeled Zig type."""
    name = _text(type_name, "type name")
    return dict(_layout(name, _word(word)))


def zf_validation_report(
    checks: Sequence[tuple[str, bool]],
) -> dict[str, Any]:
    """Summarize named boolean checks into a deterministic validation report."""
    passed: list[str] = []
    failed: list[str] = []
    for item in _items(checks, "checks"):
        pair = _items(item, "check")
        if len(pair) != 2:
            _fail("E003", "each check must be a (name, result) pair")
        label = _text(pair[0], "check name")
        outcome = _boolean(pair[1], "check result")
        (passed if outcome else failed).append(label)
    total = len(passed) + len(failed)
    return {
        "passed": tuple(passed),
        "failed": tuple(failed),
        "ok": not failed,
        "count": total,
        "summary": f"{len(passed)}/{total} checks passed",
    }
