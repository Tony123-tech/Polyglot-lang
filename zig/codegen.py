"""Deterministic Zig source emitters for the model backend rollout.

Each function returns a structured record containing the emitted source and
the metadata needed to reason about it.  Emission is pure string construction:
the real Zig compiler is never invoked and no generated artifact is executed.
Vector examples are restricted to integer element types, avoiding the known
float-vector defect in the legacy ``zig.backend`` emitter.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Final, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "zgc_alignof_expr",
    "zgc_array_literal",
    "zgc_binary_expr",
    "zgc_block",
    "zgc_cast_expr",
    "zgc_comment",
    "zgc_const_decl",
    "zgc_enum_decl",
    "zgc_export_shim",
    "zgc_expr_stmt",
    "zgc_fn_decl",
    "zgc_for_loop",
    "zgc_helper_lib",
    "zgc_if_expr",
    "zgc_import_decl",
    "zgc_index_expr",
    "zgc_integer_literal",
    "zgc_match_expr",
    "zgc_module_header",
    "zgc_module_source",
    "zgc_ptr_type",
    "zgc_return_stmt",
    "zgc_simd_expr",
    "zgc_slice_type",
    "zgc_struct_decl",
    "zgc_switch_expr",
    "zgc_type_expr",
    "zgc_unary_expr",
    "zgc_var_decl",
    "zgc_while_loop",
]

_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER: Final[re.Pattern[str]] = re.compile(r"^([iu])([1-9][0-9]*)$")
_MAX_INTEGER_BITS: Final[int] = 65535
_VECTOR: Final[re.Pattern[str]] = re.compile(r"^@Vector\((\d+),\s*([A-Za-z0-9_]+)\)$")
_PRIMITIVES: Final[frozenset[str]] = frozenset({"bool", "f32", "f64", "isize", "usize"})
_LANES: Final[frozenset[int]] = frozenset({2, 4, 8, 16, 32, 64})
_HELPERS: Final[frozenset[str]] = frozenset({"saturating", "simd", "wide_int"})
_BINARY_OPS: Final[frozenset[str]] = frozenset(
    {
        "%",
        "&",
        "*",
        "+",
        "+|",
        "-",
        "-|",
        "/",
        "<<",
        "<",
        "<=",
        "==",
        ">",
        ">=",
        ">>",
        "^",
        "|",
        "*|",
        "and",
        "!=",
        "or",
    }
)
_UNARY_OPS: Final[frozenset[str]] = frozenset({"-", "!", "~", "&", "*"})
_CASTS: Final[dict[str, str]] = {
    "as": "@as",
    "float": "@floatFromInt",
    "int": "@intCast",
    "truncate": "@truncate",
}
_RADIX_PREFIX: Final[dict[int, str]] = {2: "0b", 8: "0o", 10: "", 16: "0x"}
_RADIX_FORMAT: Final[dict[int, str]] = {2: "b", 8: "o", 10: "d", 16: "x"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, *, reason: str | None = None) -> NoReturn:
    """Raise a structured Zig-codegen diagnostic."""
    raise PolyglotError(code, message, reason=reason, phase="zig-codegen")


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
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
        _fail("E003", f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, label: str) -> str:
    """Validate an emitted Zig identifier, raising E001 for illegal names."""
    name = _text(value, label)
    if not _IDENTIFIER.fullmatch(name):
        _fail("E001", f"{label} is not a legal Zig identifier: {name!r}")
    return name


def _digest(*parts: object) -> str:
    """Stable 16-hex-digit digest of the string form of *parts*."""
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _pow2_ceil(value: int) -> int:
    """Smallest power of two greater than or equal to *value*."""
    return 1 << max(0, value - 1).bit_length()


def _type_expr(value: object) -> str:
    """Validate and canonicalize a modeled Zig type expression."""
    name = _text(value, "type name")
    if name == "void":
        return name
    integer = _INTEGER.fullmatch(name)
    if name in _PRIMITIVES or integer is not None:
        if integer is not None and int(integer.group(2)) > _MAX_INTEGER_BITS:
            _fail("E001", f"integer width out of range: {name!r}")
        return name
    if name.startswith("[*]"):
        tail = name[3:].strip()
        constant = tail.startswith("const ")
        inner = tail[6:].strip() if constant else tail
        return f"[*]{'const ' if constant else ''}{_type_expr(inner)}"
    if name.startswith("[]"):
        tail = name[2:].strip()
        constant = tail.startswith("const ")
        inner = tail[6:].strip() if constant else tail
        return f"[]{'const ' if constant else ''}{_type_expr(inner)}"
    if name.startswith("*"):
        tail = name[1:].strip()
        constant = tail.startswith("const ")
        inner = tail[6:].strip() if constant else tail
        return f"*{'const ' if constant else ''}{_type_expr(inner)}"
    vector = _VECTOR.fullmatch(name)
    if vector is not None:
        lanes = int(vector.group(1))
        if lanes not in _LANES:
            _fail("E001", f"unsupported vector lane count: {lanes}")
        element = vector.group(2)
        if element in ("f32", "f64"):
            _fail("E007", "float vector codegen is not modeled")
        return f"@Vector({lanes}, {_type_expr(element)})"
    _fail("E001", f"unsupported Zig type: {name!r}")


def _type_info(value: object) -> dict[str, Any]:
    """Return canonical text, kind, size and alignment for a modeled type."""
    expr = _type_expr(value)
    if expr == "void":
        kind, size, align = "void", 0, 1
    elif expr == "bool":
        kind, size, align = "bool", 1, 1
    elif expr in ("f32", "f64"):
        kind = "float"
        size = align = 4 if expr == "f32" else 8
    elif expr in ("isize", "usize"):
        kind = "int"
        size = align = 8
    elif _INTEGER.fullmatch(expr):
        kind = "int"
        bits = int(_INTEGER.fullmatch(expr).group(2))
        size = max(1, (bits + 7) // 8)
        align = _pow2_ceil(size)
    elif expr.startswith("[]"):
        kind, size, align = "slice", 16, 8
    elif expr.startswith(("[*]", "*")):
        kind, size, align = "pointer", 8, 8
    else:
        lanes = int(_VECTOR.fullmatch(expr).group(1))
        element = _type_info(_VECTOR.fullmatch(expr).group(2))
        kind, size, align = "vector", element["size"] * lanes, element["align"]
    return {"expr": expr, "kind": kind, "size": size, "align": align}


def _statements(value: object, label: str) -> tuple[str, ...]:
    """Validate a sequence of already-emitted statement strings."""
    return tuple(_text(item, label) for item in _items(value, label))


def _pairs(value: object, label: str) -> tuple[tuple[str, str], ...]:
    """Validate a sequence of (name, type) style string pairs."""
    rows: list[tuple[str, str]] = []
    for item in _items(value, label):
        pair = _items(item, label)
        if len(pair) != 2:
            _fail("E003", f"each {label} entry must be a pair")
        rows.append((_text(pair[0], f"{label} name"), _text(pair[1], f"{label} value")))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Public codegen functions
# ---------------------------------------------------------------------------


def zgc_alignof_expr(type_name: str) -> dict[str, Any]:
    """Emit a deterministic ``@alignOf`` expression for a modeled type."""
    info = _type_info(type_name)
    return {
        "expr": f"@alignOf({info['expr']})",
        "type": info["expr"],
        "kind": info["kind"],
        "align": info["align"],
        "builtin": True,
    }


def zgc_array_literal(
    items: Sequence[int], *, type_name: str | None = None
) -> dict[str, Any]:
    """Emit an integer array literal with an optional explicit element type."""
    values = tuple(_integer(item, "array item") for item in _items(items, "items"))
    element = None if type_name is None else _type_expr(type_name)
    inner = ", ".join(str(value) for value in values)
    expr = (
        f".{{ {inner} }}"
        if element is None
        else f"[{len(values)}]{element}{{ {inner} }}"
    )
    return {
        "expr": expr,
        "items": values,
        "length": len(values),
        "element_type": element,
    }


def zgc_binary_expr(op: str, left: str, right: str) -> dict[str, Any]:
    """Emit a parenthesized binary expression from a whitelisted operator."""
    symbol = _text(op, "operator")
    if symbol not in _BINARY_OPS:
        _fail("E003", f"unsupported binary operator: {symbol!r}")
    first = _text(left, "left operand")
    second = _text(right, "right operand")
    return {
        "expr": f"({first} {symbol} {second})",
        "op": symbol,
        "left": first,
        "right": second,
        "associative": symbol in ("+", "*", "&", "|", "^", "and", "or", "+|", "*|"),
    }


def zgc_block(
    statements: Sequence[str], *, label: str | None = None, indent: int = 0
) -> dict[str, Any]:
    """Emit a braced statement block with optional label and base indentation."""
    rows = _statements(statements, "statements")
    width = _integer(indent, "indent", minimum=0)
    pad = " " * width
    tag = None if label is None else _identifier(label, "block label")
    head = f"{tag}: {{" if tag else "{"
    lines = (head, *(f"{pad}    {item}" for item in rows), f"{pad}}}")
    return {
        "source": "\n".join(lines),
        "lines": lines,
        "label": tag,
        "count": len(rows),
    }


def zgc_cast_expr(type_name: str, expr: str, *, kind: str = "as") -> dict[str, Any]:
    """Emit a Zig cast builtin for an operand and target type."""
    target = _type_expr(type_name)
    operand = _text(expr, "operand")
    name = _text(kind, "cast kind")
    if name not in _CASTS:
        _fail("E003", f"unknown cast kind: {name!r}")
    builtin = _CASTS[name]
    rendered = f"@as({target}, {operand})" if name == "as" else f"{builtin}({operand})"
    return {
        "expr": rendered,
        "type": target,
        "kind": name,
        "builtin": builtin,
        "operand": operand,
    }


def zgc_comment(text: str, *, style: str = "line", indent: int = 0) -> dict[str, Any]:
    """Emit line or block comments with stable indentation."""
    content = _text(text, "comment text")
    kind = _text(style, "style")
    if kind not in ("block", "line"):
        _fail("E003", f"unknown comment style: {kind!r}")
    width = _integer(indent, "indent", minimum=0)
    pad = " " * width
    lines = tuple(
        f"{pad}// {part}" if kind == "line" else f"{pad}/* {part} */"
        for part in content.split("\n")
    )
    return {
        "text": content,
        "style": kind,
        "source": "\n".join(lines),
        "lines": lines,
        "indent": width,
    }


def zgc_const_decl(
    name: str, value: str, *, type_name: str | None = None
) -> dict[str, Any]:
    """Emit a ``const`` declaration with an optional type annotation."""
    symbol = _identifier(name, "constant name")
    expression = _text(value, "value")
    annotation = None if type_name is None else _type_expr(type_name)
    suffix = f": {annotation}" if annotation else ""
    source = f"const {symbol}{suffix} = {expression};"
    return {
        "source": source,
        "name": symbol,
        "value": expression,
        "type": annotation,
        "mutable": False,
    }


def zgc_enum_decl(
    name: str, variants: Sequence[str | tuple[str, int]], *, tag: str = "u8"
) -> dict[str, Any]:
    """Emit a tagged enum declaration with optional explicit discriminants."""
    enum_name = _identifier(name, "enum name")
    tag_type = _type_expr(tag)
    if not tag_type.startswith(("i", "u")):
        _fail("E001", "enum tag must be an integer type")
    rows: list[str] = []
    labels: list[str] = []
    for item in _items(variants, "variants"):
        if isinstance(item, str):
            label = _identifier(item, "variant name")
            rows.append(f"{label},")
        else:
            pair = _items(item, "variant")
            if len(pair) != 2:
                _fail("E003", "each variant must be a name or (name, value) pair")
            label = _identifier(pair[0], "variant name")
            number = _integer(pair[1], "variant value", minimum=0)
            rows.append(f"{label} = {number},")
        labels.append(label)
    lines = (
        f"const {enum_name} = enum({tag_type}) {{",
        *tuple(f"    {row}" for row in rows),
        "};",
    )
    return {
        "source": "\n".join(lines),
        "name": enum_name,
        "tag": tag_type,
        "variants": tuple(labels),
        "count": len(labels),
    }


def zgc_export_shim(
    name: str,
    *,
    params: Sequence[tuple[str, str]] = (),
    result: str = "void",
    abi: str = "c",
) -> dict[str, Any]:
    """Emit an ``export fn`` shim that forwards to a model function."""
    symbol = _identifier(name, "function name")
    pairs = _pairs(params, "params")
    args = tuple(
        f"{_identifier(key, 'parameter')}: {_type_expr(value)}" for key, value in pairs
    )
    result_type = _type_expr(result)
    kind = _text(abi, "abi")
    if kind not in ("c", "zig"):
        _fail("E003", f"unknown ABI: {kind!r}")
    convention = "callconv(.c) " if kind == "c" else ""
    call = f"{symbol}({', '.join(_identifier(key, 'parameter') for key, _ in pairs)})"
    body = f"    return {call};" if result_type != "void" else f"    {call};"
    lines = (
        f"export fn __cabi_{symbol}({', '.join(args)}) {convention}{result_type} {{",
        body,
        "}",
    )
    return {
        "source": "\n".join(lines),
        "lines": lines,
        "symbol": f"__cabi_{symbol}",
        "params": tuple(key for key, _ in pairs),
        "result": result_type,
        "callconv": kind,
    }


def zgc_expr_stmt(expr: str) -> dict[str, Any]:
    """Emit one expression statement with its terminating semicolon."""
    expression = _text(expr, "expression")
    if not expression:
        _fail("E003", "expression must not be empty")
    return {
        "source": f"{expression};",
        "expr": expression,
        "terminated": True,
    }


def zgc_fn_decl(
    name: str,
    *,
    params: Sequence[tuple[str, str]] = (),
    result: str = "void",
    body: Sequence[str] = (),
    is_pub: bool = False,
) -> dict[str, Any]:
    """Emit a complete Zig function declaration from model metadata."""
    symbol = _identifier(name, "function name")
    pairs = _pairs(params, "params")
    args = tuple(
        f"{_identifier(key, 'parameter')}: {_type_expr(value)}" for key, value in pairs
    )
    result_type = _type_expr(result)
    rows = _statements(body, "body")
    public = _boolean(is_pub, "is_pub")
    head = f"{'pub ' if public else ''}fn {symbol}({', '.join(args)}) {result_type} {{"
    lines = (head, *tuple(f"    {row}" for row in rows), "}")
    return {
        "source": "\n".join(lines),
        "name": symbol,
        "params": args,
        "result": result_type,
        "public": public,
        "line_count": len(lines),
    }


def zgc_for_loop(
    var_name: str,
    start: str,
    end: str,
    *,
    inclusive: bool = False,
    body: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit an explicit integer-range while loop with a continue expression."""
    variable = _identifier(var_name, "loop variable")
    first = _text(start, "start")
    last = _text(end, "end")
    closed = _boolean(inclusive, "inclusive")
    rows = _statements(body, "body")
    lines = (
        f"var {variable}: usize = {first};",
        f"while ({variable} {'<=' if closed else '<'} {last}) : ({variable} += 1) {{",
        *tuple(f"    {row}" for row in rows),
        "}",
    )
    return {
        "source": "\n".join(lines),
        "var": variable,
        "start": first,
        "end": last,
        "inclusive": closed,
        "body_count": len(rows),
    }


def zgc_helper_lib(*, features: Sequence[str] = ()) -> dict[str, Any]:
    """Emit integer-safe helper functions for requested Zig model features."""
    chosen: list[str] = []
    for item in _items(features, "features"):
        name = _text(item, "feature")
        if name not in _HELPERS:
            _fail("E007", f"unknown helper feature: {name!r}")
        if name not in chosen:
            chosen.append(name)
    selected = tuple(sorted(chosen))
    blocks: list[str] = []
    helpers: list[str] = []
    if "saturating" in selected:
        helpers.append("_zig_sat_add")
        blocks.append(
            "fn _zig_sat_add(a: i32, b: i32) i32 {\n"
            "    const pair = @addWithOverflow(a, b);\n"
            "    return if (pair[1] != 0) @as(i32, 2147483647) else pair[0];\n"
            "}"
        )
    if "simd" in selected:
        helpers.append("_zig_simd_sum4")
        blocks.append(
            "fn _zig_simd_sum4(values: @Vector(4, i32)) i32 {\n"
            "    return @reduce(.Add, values);\n"
            "}"
        )
    if "wide_int" in selected:
        helpers.append("_zig_wide_add")
        blocks.append(
            "fn _zig_wide_add(a: u128, b: u128) u128 {\n    return a +% b;\n}"
        )
    return {
        "source": "\n\n".join(blocks),
        "features": selected,
        "helpers": tuple(helpers),
        "fingerprint": _digest("helpers", selected, blocks),
    }


def zgc_if_expr(condition: str, then_expr: str, else_expr: str) -> dict[str, Any]:
    """Emit a Zig if-expression with explicit then and else branches."""
    cond = _text(condition, "condition")
    affirmative = _text(then_expr, "then expression")
    negative = _text(else_expr, "else expression")
    return {
        "expr": f"if ({cond}) {affirmative} else {negative}",
        "condition": cond,
        "then": affirmative,
        "else": negative,
        "exhaustive": True,
    }


def zgc_import_decl(module: str) -> dict[str, Any]:
    """Emit a Zig ``@import`` declaration with a derived local alias."""
    path = _text(module, "module path")
    if not path or any(char.isspace() for char in path):
        _fail("E003", f"invalid module path: {path!r}")
    segment = re.sub(r"[^A-Za-z0-9_]", "_", path.replace(".", "_").split("_")[-1])
    if not segment or segment[0].isdigit():
        segment = f"_{segment}"
    alias = _identifier(segment, "import alias")
    source = f'const {alias} = @import("{path}");'
    return {
        "source": source,
        "module": path,
        "alias": alias,
        "parts": tuple(path.split(".")),
    }


def zgc_index_expr(base: str, index: str) -> dict[str, Any]:
    """Emit a bracketed indexing expression from opaque text operands."""
    receiver = _text(base, "base expression")
    position = _text(index, "index expression")
    return {
        "expr": f"{receiver}[{position}]",
        "base": receiver,
        "index": position,
        "bounds_checked_model": True,
    }


def zgc_integer_literal(
    value: int, *, radix: int = 10, type_name: str | None = None
) -> dict[str, Any]:
    """Emit an integer literal in radix 2, 8, 10 or 16 with optional cast."""
    number = _integer(value, "value")
    base = _integer(radix, "radix", minimum=2)
    if base not in _RADIX_PREFIX:
        _fail("E003", f"unsupported radix: {base}")
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    digits = format(magnitude, _RADIX_FORMAT[base])
    prefix = _RADIX_PREFIX.get(base, "")
    literal = f"{sign}{prefix}{digits}"
    target = None if type_name is None else _type_expr(type_name)
    expr = f"@as({target}, {literal})" if target else literal
    return {
        "expr": expr,
        "value": number,
        "radix": base,
        "digit_count": len(digits),
        "type": target,
    }


def zgc_match_expr(
    subject: str,
    arms: Sequence[tuple[str, str]],
    *,
    otherwise: str | None = None,
) -> dict[str, Any]:
    """Emit a multi-line Zig switch expression modeling a match expression."""
    value = _text(subject, "subject")
    patterns = _pairs(arms, "arms")
    if not patterns:
        _fail("E003", "match expression needs at least one arm")
    fallback = None if otherwise is None else _text(otherwise, "otherwise expression")
    body = [f"    {pattern} => {result}," for pattern, result in patterns]
    if fallback is not None:
        body.append(f"    else => {fallback},")
    lines = (f"switch ({value}) {{", *body, "}")
    return {
        "expr": "\n".join(lines),
        "subject": value,
        "arms": patterns,
        "otherwise": fallback,
        "exhaustive": fallback is not None,
    }


def zgc_module_header(
    name: str, *, std_import: bool = False, features: Sequence[str] = ()
) -> dict[str, Any]:
    """Emit the deterministic header and optional std import of a Zig module."""
    module = _identifier(name, "module")
    use_std = _boolean(std_import, "std_import")
    selected = tuple(
        sorted({_text(item, "feature") for item in _items(features, "features")})
    )
    lines = [
        f"// Generated by polyglot-lang for {module}",
        f"// Features: {', '.join(selected) if selected else 'none'}",
    ]
    imports: list[str] = []
    if use_std:
        imports.append('const std = @import("std");')
    return {
        "source": "\n".join(lines + imports),
        "module": module,
        "features": selected,
        "imports": tuple(imports),
        "lines": tuple(lines + imports),
    }


def zgc_module_source(
    name: str,
    *,
    functions: Sequence[str] = (),
    features: Sequence[str] = (),
    imports: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble a complete model Zig module from emitted declarations."""
    module = _identifier(name, "module")
    selected = tuple(
        sorted({_text(item, "feature") for item in _items(features, "features")})
    )
    paths = tuple(_text(item, "import path") for item in _items(imports, "imports"))
    declarations = _statements(functions, "functions")
    lines = [f"// Generated by polyglot-lang: {module}"]
    if selected:
        lines.append(f"// Features: {', '.join(selected)}")
    for path in paths:
        lines.append(
            f'const {"_".join(re.findall(r"[A-Za-z0-9]+", path)) or "mod"} = @import("{path}");'
        )
    lines.append("")
    lines.extend(declarations)
    source = "\n".join(lines)
    if len(source) > 1_000_000:
        _fail("E004", "model emission size limit exceeded")
    return {
        "source": source,
        "module": module,
        "features": selected,
        "imports": paths,
        "function_count": len(declarations),
        "line_count": len(lines),
        "fingerprint": _digest("module", module, source),
    }


def zgc_ptr_type(
    type_name: str, *, const: bool = True, many: bool = True
) -> dict[str, Any]:
    """Emit a single-item or many-item Zig pointer type."""
    inner = _type_expr(type_name)
    constant = _boolean(const, "const")
    multiple = _boolean(many, "many")
    qualifier = "const " if constant else ""
    expr = f"{'[*]' if multiple else '*'}{qualifier}{inner}"
    return {
        "expr": expr,
        "type": inner,
        "qualifier": qualifier.strip() or "mutable",
        "many": multiple,
        "size": 8,
        "model_only": True,
    }


def zgc_return_stmt(expr: str | None = None) -> dict[str, Any]:
    """Emit a value-returning or bare Zig return statement."""
    value = None if expr is None else _text(expr, "return expression")
    source = "return;" if value is None else f"return {value};"
    return {"source": source, "expr": value, "void": value is None}


def zgc_simd_expr(
    op: str,
    lanes: int,
    *,
    type_name: str = "i32",
    values: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Emit an integer-only SIMD expression using Zig vector builtins."""
    name = _text(op, "op")
    if name not in ("add", "reduce_add", "reduce_max", "splat"):
        _fail("E003", f"unknown SIMD operation: {name!r}")
    count = _integer(lanes, "lanes", minimum=1)
    if count not in _LANES:
        _fail("E001", f"unsupported vector lane count: {count}")
    element = _type_expr(type_name)
    if element in ("f32", "f64"):
        _fail("E007", "float vector codegen is not modeled")
    if not _INTEGER.fullmatch(element) and element not in ("isize", "usize"):
        _fail("E001", "SIMD element type must be an integer")
    supplied = (
        ()
        if values is None
        else tuple(_integer(item, "vector value") for item in _items(values, "values"))
    )
    if supplied:
        expected = 1 if name == "splat" else count
        if len(supplied) != expected:
            _fail("E001", "vector value count must match the operation arity")
    if supplied:
        actual = supplied
    elif name == "splat":
        actual = (0,)
    else:
        actual = (0,) * count
    vector_type = f"@Vector({count}, {element})"
    if name == "splat":
        expr = f"@splat({actual[0]})"
    elif name == "add":
        left, right = actual[0], actual[1] if count > 1 else actual[0]
        expr = (
            f"(@as({vector_type}, @splat({left})) + "
            f"@as({vector_type}, @splat({right})))"
        )
    else:
        reduction = "Add" if name == "reduce_add" else "Max"
        literal = ", ".join(str(item) for item in actual)
        expr = f"@reduce(.{reduction}, @as({vector_type}, .{{ {literal} }}))"
    return {
        "expr": expr,
        "op": name,
        "lanes": count,
        "type": vector_type,
        "values": actual,
        "float_safe": True,
        "note": "integer-only model",
    }


def zgc_slice_type(type_name: str, *, const: bool = True) -> dict[str, Any]:
    """Emit a Zig slice type with the requested const qualification."""
    inner = _type_expr(type_name)
    constant = _boolean(const, "const")
    expr = f"[]{'const ' if constant else ''}{inner}"
    return {
        "expr": expr,
        "type": inner,
        "qualifier": "const" if constant else "mutable",
        "size": 16,
        "align": 8,
        "model_only": True,
    }


def zgc_struct_decl(
    name: str, fields: Sequence[tuple[str, str]], *, packed: bool = False
) -> dict[str, Any]:
    """Emit a regular or packed Zig struct declaration."""
    struct_name = _identifier(name, "struct name")
    pairs = _pairs(fields, "fields")
    rows = tuple(
        f"{_identifier(key, 'field name')}: {_type_expr(value)},"
        for key, value in pairs
    )
    is_packed = _boolean(packed, "packed")
    kind = "packed struct" if is_packed else "struct"
    lines = (
        f"const {struct_name} = {kind} {{",
        *tuple(f"    {row}" for row in rows),
        "};",
    )
    return {
        "source": "\n".join(lines),
        "name": struct_name,
        "kind": kind,
        "fields": tuple(key for key, _ in pairs),
        "count": len(rows),
    }


def zgc_switch_expr(
    subject: str,
    arms: Sequence[tuple[str, str]],
    *,
    otherwise: str = "unreachable",
) -> dict[str, Any]:
    """Emit a single-line Zig switch expression with an explicit default arm."""
    value = _text(subject, "subject")
    patterns = _pairs(arms, "arms")
    if not patterns:
        _fail("E003", "switch expression needs at least one arm")
    fallback = _text(otherwise, "otherwise expression")
    body = ", ".join(f"{pattern} => {result}" for pattern, result in patterns)
    expr = f"switch ({value}) {{ {body}, else => {fallback} }}"
    return {
        "expr": expr,
        "subject": value,
        "arms": patterns,
        "otherwise": fallback,
        "exhaustive": True,
    }


def zgc_type_expr(type_name: str) -> dict[str, Any]:
    """Canonicalize a Zig type expression and report its modeled layout."""
    info = _type_info(type_name)
    return {
        "expr": info["expr"],
        "type": info["expr"],
        "kind": info["kind"],
        "size": info["size"],
        "align": info["align"],
        "model_only": True,
    }


def zgc_unary_expr(op: str, operand: str) -> dict[str, Any]:
    """Emit a prefix unary expression from a whitelisted operator."""
    symbol = _text(op, "operator")
    if symbol not in _UNARY_OPS:
        _fail("E003", f"unsupported unary operator: {symbol!r}")
    value = _text(operand, "operand")
    return {
        "expr": f"{symbol}{value}",
        "op": symbol,
        "operand": value,
        "prefix": True,
    }


def zgc_var_decl(
    name: str, value: str, *, type_name: str | None = None, constant: bool = False
) -> dict[str, Any]:
    """Emit a mutable or constant Zig variable declaration."""
    symbol = _identifier(name, "variable name")
    expression = _text(value, "value")
    annotation = None if type_name is None else _type_expr(type_name)
    is_constant = _boolean(constant, "constant")
    keyword = "const" if is_constant else "var"
    suffix = f": {annotation}" if annotation else ""
    source = f"{keyword} {symbol}{suffix} = {expression};"
    return {
        "source": source,
        "name": symbol,
        "value": expression,
        "type": annotation,
        "mutable": not is_constant,
    }


def zgc_while_loop(
    condition: str, *, body: Sequence[str] = (), defer_expr: str | None = None
) -> dict[str, Any]:
    """Emit a Zig while loop with optional defer and stable indentation."""
    cond = _text(condition, "condition")
    rows = _statements(body, "body")
    cleanup = None if defer_expr is None else _text(defer_expr, "defer expression")
    lines = [f"while ({cond}) {{"]
    if cleanup is not None:
        lines.append(f"    defer {cleanup};")
    lines.extend(f"    {row}" for row in rows)
    lines.append("}")
    return {
        "source": "\n".join(lines),
        "condition": cond,
        "body_count": len(rows),
        "defer": cleanup,
        "lines": tuple(lines),
    }
