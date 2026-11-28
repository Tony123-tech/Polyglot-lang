"""Read-only analysis helpers for the interactive Polyglot visualizer.

This module is deliberately dependency-free: it turns the parser's AST into a
JSON tree, runs the monomorphization pass and renders a concrete source view,
and extracts structured E001-E009 diagnostics. Execution lives in
:mod:`api.visualizer_server`, which shells out to ``api.cli exec`` so that user
programs never run inside the HTTP server process.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from typing import Any

import compiler.core.ast_nodes as ast
from compiler.analysis.monomorph import MonomorphizationError, monomorphize
from compiler.core.lexer import LexError, lex
from compiler.core.parser import ParseError, parse

MAX_TREE_NODES = 20_000
MAX_TREE_DEPTH = 96

_SKIP_FIELDS = {"line", "col", "span", "source_lines"}
_ITEM_LABELS = {
    "FnDecl": "function",
    "StructDecl": "struct",
    "EnumDecl": "enum",
    "ImplBlock": "impl",
    "TraitDecl": "trait",
    "GenericParam": "generic",
    "ConstDecl": "constant",
    "TypeAlias": "type alias",
    "UseDecl": "use",
    "ModDecl": "module",
    "IncludeDecl": "include",
}


@dataclasses.dataclass
class _Synthetic:
    """A small synthetic tree node for tuple-shaped AST fields."""

    type_name: str
    label: str
    category: str
    children: list[tuple[str, Any]]
    line: int = 0
    col: int = 0
    span: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def line_col_to_offset(source: str, line: int | None, col: int | None) -> int | None:
    """Character offset of a 1-based (line, col) pair, or ``None``."""
    if line is None or col is None or line < 1:
        return None
    lines = source.splitlines(keepends=True)
    if line > len(lines):
        return None
    return sum(len(text) for text in lines[: line - 1]) + max(0, col - 1)


def _offset_to_line_col(source: str, offset: int) -> tuple[int, int]:
    offset = max(0, min(offset, len(source)))
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    col = offset - last_newline
    return line, col


def _token_span_at(source: str, offset: int) -> tuple[int, int]:
    if not source:
        return (0, 0)
    offset = max(0, min(offset, len(source) - 1))
    match = re.match(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S",
        source[offset:],
    )
    if match is None:
        return (offset, min(offset + 1, len(source)))
    return (offset, offset + len(match.group(0)))


def _extract_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    text = str(error)
    match = re.search(r"error\[(E\d+)\]", text)
    if match is not None:
        return match.group(1)
    name = type(error).__name__
    if name in {"LexError", "ParseError"}:
        return "SYNTAX"
    return None


def _clean_message(error: BaseException) -> str:
    text = str(error).strip()
    if not text:
        return type(error).__name__
    first = text.splitlines()[0]
    first = re.sub(r"^error(?:\[E\d+\])?:\s*", "", first).strip()
    reason = getattr(error, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return first or type(error).__name__


def _error_span(
    error: BaseException, source: str
) -> tuple[int, int] | tuple[None, None]:
    span = getattr(error, "span", None)
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except TypeError, ValueError:
            pass
        else:
            if 0 <= start <= end <= len(source):
                return start, end
    line = getattr(error, "line", None)
    if line is None:
        line = getattr(error, "source_line", None)
    col = getattr(error, "col", None)
    if col is None:
        col = getattr(error, "source_col", None)
    offset = line_col_to_offset(source, line, col)
    if offset is None:
        return None, None
    return _token_span_at(source, offset)


@dataclasses.dataclass
class FixHint:
    """A deterministic, dependency-free fix suggestion for a diagnostic.

    ``edits`` uses the LSP ``TextEdit`` shape so the CLI JSON output and the
    language server can share exactly the same payload.  No edit is produced
    when a correct replacement cannot be determined locally; a bare message is
    still actionable and never claims more than the compiler actually knows.
    """

    message: str
    edits: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "edits": [dict(edit) for edit in self.edits],
        }


# Public codes are E001-E009 (``SYNTAX`` covers lexer/parser failures).
_FIXABLE_CODES = frozenset({"SYNTAX"} | {f"E{index:03d}" for index in range(1, 10)})

_SUPPORTED_MACROS = (
    "println!",
    "print!",
    "eprintln!",
    "eprint!",
    "format!",
    "vec!",
    "panic!",
    "assert!",
    "assert_eq!",
    "assert_ne!",
)


def _utf16_length(text: str) -> int:
    """UTF-16 code units for ``text`` (non-BMP code points use a pair)."""
    return len(text) + sum(1 for character in text if ord(character) > 0xFFFF)


def offset_to_lsp_position(source: str, offset: int) -> dict[str, int]:
    """Convert a 0-based Python character offset to a 0-based LSP ``Position``.

    LSP 3.17 defaults to UTF-16 positions. Internal spans stay in Python
    character offsets; only this boundary translates them.
    """
    offset = max(0, min(int(offset), len(source)))
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    return {
        "line": line - 1,
        "character": _utf16_length(source[last_newline + 1 : offset]),
    }


def lsp_position_to_offset(source: str, position: dict[str, int]) -> int:
    """Convert a UTF-16 LSP ``Position`` to a 0-based Python character offset."""
    line = max(0, int(position.get("line", 0)))
    character = max(0, int(position.get("character", 0)))
    lines = source.split("\n")
    if line >= len(lines):
        return len(source)
    line_start = sum(len(part) + 1 for part in lines[:line])
    line_text = lines[line]
    units = 0
    for index, value in enumerate(line_text):
        if units >= character:
            return line_start + index
        units += 2 if ord(value) > 0xFFFF else 1
    return line_start + len(line_text)


def span_to_lsp_range(
    source: str,
    span: tuple[int, int] | list[int] | None,
    *,
    line: int | None = None,
    col: int | None = None,
) -> dict[str, Any]:
    """Build a 0-based LSP ``Range`` from an exact character span.

    When the span is missing (or does not fit the supplied source) the 1-based
    ``line``/``col`` pair is used as a conservative single-token fallback.
    """
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except TypeError, ValueError:
            start, end = -1, -1
        if 0 <= start <= end <= len(source):
            return {
                "start": offset_to_lsp_position(source, start),
                "end": offset_to_lsp_position(source, end),
            }
    start_line = max(0, int(line or 1) - 1)
    start_col = max(0, int(col or 1) - 1)
    lines = source.splitlines(keepends=True)
    if start_line < len(lines):
        line_text = lines[start_line]
        start_offset = sum(len(part) for part in lines[:start_line]) + min(
            start_col, len(line_text)
        )
        start = offset_to_lsp_position(source, start_offset)
    else:
        # Preserve the reported location even when no source text is available.
        line_text = ""
        start = {"line": start_line, "character": start_col}
    width = 1
    if isinstance(span, (tuple, list)) and len(span) == 2:
        try:
            width = max(1, int(span[1]) - int(span[0]))
        except TypeError, ValueError:
            width = 1
    if start_line < len(lines):
        end_offset = start["character"] + _utf16_length(
            line_text[start_col : start_col + width]
        )
    else:
        end_offset = start["character"] + width
    return {
        "start": start,
        "end": {"line": start["line"], "character": end_offset},
    }


def _detail_field(detail: str, name: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*(\S+)", detail)
    return match.group(1) if match else None


def fix_hint(error: BaseException, source: str) -> FixHint | None:
    """Return an actionable hint for a known compiler diagnostic, else None."""
    code = _extract_code(error)
    if code not in _FIXABLE_CODES:
        return None

    start, end = _error_span(error, source)
    line = getattr(error, "line", None)
    col = getattr(error, "col", None)
    if line is None:
        line = getattr(error, "source_line", None)
    if col is None:
        col = getattr(error, "source_col", None)
    detail = str(error)
    span = (start, end) if start is not None else None

    if code == "E001":
        expected = getattr(error, "expected_type", None) or _detail_field(
            detail, "expected"
        )
        if expected:
            insert = dict(span_to_lsp_range(source, span, line=line, col=col)["end"])
            return FixHint(
                message=f"Write `as {expected}` to make the integer cast explicit.",
                edits=[
                    {
                        "range": {"start": dict(insert), "end": dict(insert)},
                        "newText": f" as {expected}",
                    }
                ],
            )
    if code == "E002":
        suggestion = getattr(error, "suggestion", None)
        if not suggestion:
            match = re.search(r"(?m)^\s*help:\s*(.+)$", detail)
            suggestion = match.group(1).strip() if match else None
        if suggestion:
            return FixHint(message=suggestion)
    if code == "E003":
        lowered = detail.lower()
        if "unknown macro" in lowered:
            return FixHint(
                message=(
                    "Use a supported macro instead: "
                    + ", ".join(_SUPPORTED_MACROS)
                    + "."
                )
            )
        if "unsupported" in lowered:
            return FixHint(
                message="Replace the unsupported syntax with a supported Polyglot construct."
            )
    if code == "E006":
        help_text = getattr(error, "help_text", None)
        if isinstance(help_text, str) and help_text.strip():
            return FixHint(message=help_text.strip())

    help_text = getattr(error, "help_text", None)
    if isinstance(help_text, str) and help_text.strip():
        return FixHint(message=help_text.strip())
    return None


def diagnostic(
    error: BaseException,
    source: str,
    *,
    phase: str,
) -> dict[str, Any]:
    """Return a JSON-safe diagnostic with a character span when possible."""
    code = _extract_code(error)
    start, end = _error_span(error, source)
    line = getattr(error, "line", None)
    col = getattr(error, "col", None)
    if line is None:
        line = getattr(error, "source_line", None)
    if col is None:
        col = getattr(error, "source_col", None)
    if start is not None and (line is None or col is None):
        line, col = _offset_to_line_col(source, start)
    excerpt = ""
    if line is not None:
        lines = source.splitlines()
        if 1 <= line <= len(lines):
            excerpt = lines[line - 1]
    diagnostic: dict[str, Any] = {
        "code": code,
        "phase": phase,
        "message": _clean_message(error),
        "note": type(error).__name__,
        "line": line,
        "col": col,
        "span": [start, end] if start is not None else None,
        "excerpt": excerpt,
        "detail": str(error),
    }
    help_text = getattr(error, "help_text", None)
    if isinstance(help_text, str) and help_text:
        diagnostic["help"] = help_text
    hint = fix_hint(error, source)
    diagnostic["fix"] = hint.to_dict() if hint is not None else None
    return diagnostic


# ---------------------------------------------------------------------------
# AST -> JSON tree
# ---------------------------------------------------------------------------


def _category(node: Any) -> str:
    name = type(node).__name__
    if isinstance(node, _Synthetic):
        return node.category
    if name == "Program":
        return "program"
    if name == "FnDecl":
        return "function"
    if name in {"StructDecl", "StructField"}:
        return "struct"
    if name in {"EnumDecl", "EnumVariant"}:
        return "enum"
    if name in {"ImplBlock", "TraitDecl", "TraitItem"}:
        return "impl"
    if name == "GenericParam":
        return "generic"
    if name in _ITEM_LABELS:
        return "item"
    if name.startswith("Pat"):
        return "pattern"
    if name.startswith("Type"):
        return "type"
    if name.startswith("Lit"):
        return "literal"
    if name in {"LetStmt", "ExprStmt", "ItemStmt"}:
        return "statement"
    if name == "Attr":
        return "attribute"
    return "expression"


def _generic_text(node: ast.FnDecl | ast.StructDecl | ast.EnumDecl) -> str:
    generics = getattr(node, "generics", None) or []
    if not generics:
        return ""
    parts = []
    for param in generics:
        bounds = getattr(param, "bounds", None) or []
        suffix = "".join(f" + {bound.name}" for bound in bounds)
        parts.append(f"{param.name}{suffix}")
    return f"<{', '.join(parts)}>"


def _pattern_text(node: Any) -> str:
    if isinstance(node, ast.PatWildcard):
        return "_"
    if isinstance(node, ast.PatIdent):
        return f"mut {node.name}" if node.mutable else node.name
    if isinstance(node, ast.PatLiteral):
        value = node.value
        if isinstance(value, str):
            return json.dumps(value)
        return repr(value)
    if isinstance(node, ast.PatTuple):
        return "(" + ", ".join(_pattern_text(item) for item in node.elems) + ")"
    if isinstance(node, ast.PatStruct):
        fields = ", ".join(f"{name}: {_pattern_text(pat)}" for name, pat in node.fields)
        rest = ", .." if node.rest else ""
        return f"{node.path} {{ {fields}{rest} }}"
    if isinstance(node, ast.PatEnum):
        args = ", ".join(_pattern_text(item) for item in node.args)
        return f"{node.path}({args})" if args else node.path
    if isinstance(node, ast.PatOr):
        return " | ".join(_pattern_text(item) for item in node.alts)
    if isinstance(node, ast.PatRange):
        start = _expr_text(node.start) if node.start is not None else ""
        end = _expr_text(node.end) if node.end is not None else ""
        return f"{start}{'..=' if node.inclusive else '..'}{end}"
    if isinstance(node, ast.PatBinding):
        return f"{node.name} @ {_pattern_text(node.inner)}"
    if isinstance(node, ast.PatRef):
        return f"&{'mut ' if node.mutable else ''}{_pattern_text(node.inner)}"
    if isinstance(node, ast.PatRest):
        return ".."
    if isinstance(node, ast.PatBox):
        return f"box {_pattern_text(node.inner)}"
    if isinstance(node, ast.PatConst):
        return node.path
    if isinstance(node, ast.PatMacro):
        return f"{node.path}!(...)"
    if isinstance(node, ast.PatPath):
        return node.path
    if isinstance(node, ast.PatSlice):
        return "[" + ", ".join(_pattern_text(item) for item in node.elems) + "]"
    return type(node).__name__


def _expr_text(node: Any) -> str:
    """Short expression text used by labels; full printing lives below."""
    if isinstance(node, ast.LitInt):
        return node.raw
    if isinstance(node, ast.LitFloat):
        return node.raw
    if isinstance(node, ast.LitStr):
        return json.dumps(node.value)
    if isinstance(node, ast.LitChar):
        return repr(node.value)
    if isinstance(node, ast.LitBool):
        return "true" if node.value else "false"
    if isinstance(node, ast.Ident):
        return node.name
    if isinstance(node, ast.Path):
        return "::".join(node.segments)
    if isinstance(node, ast.Binary):
        return f"{_expr_text(node.left)} {node.op} {_expr_text(node.right)}"
    if isinstance(node, ast.Call):
        return f"{_expr_text(node.func)}(...)"
    if isinstance(node, ast.MethodCall):
        return f"{_expr_text(node.receiver)}.{node.method}(...)"
    if isinstance(node, ast.Field):
        return f"{_expr_text(node.receiver)}.{node.name}"
    if isinstance(node, ast.Index):
        return f"{_expr_text(node.receiver)}[...]"
    if isinstance(node, ast.StructLiteral):
        return f"{node.path} {{ ... }}"
    if isinstance(node, (ast.OptionSome, ast.ResultOk, ast.ResultErr)):
        return f"{type(node).__name__}(...)"
    if isinstance(node, ast.OptionNone):
        return "None"
    if isinstance(node, ast.Paren):
        return f"({_expr_text(node.inner)})"
    if isinstance(node, ast.Cast):
        return f"{_expr_text(node.expr)} as {ast.type_display_name(node.ty)}"
    return type(node).__name__


def _label(node: Any) -> str:
    if isinstance(node, _Synthetic):
        return node.label
    if isinstance(node, ast.Program):
        return f"Program · {len(node.items)} items"
    if isinstance(node, ast.FnDecl):
        prefix = ""
        if node.is_pub:
            prefix += "pub "
        if node.is_async:
            prefix += "async "
        if node.is_unsafe:
            prefix += "unsafe "
        params = ", ".join(
            f"{_pattern_text(param.pattern)}"
            + (f": {ast.type_display_name(param.ty)}" if param.ty else "")
            for param in node.params
        )
        ret = f" -> {ast.type_display_name(node.ret)}" if node.ret else ""
        attribute = "#[terminates] " if node.terminates else ""
        return f"{attribute}{prefix}fn {node.name}{_generic_text(node)}({params}){ret}"
    if isinstance(node, ast.StructDecl):
        return f"{'pub ' if node.is_pub else ''}struct {node.name}{_generic_text(node)}"
    if isinstance(node, ast.EnumDecl):
        return f"{'pub ' if node.is_pub else ''}enum {node.name}{_generic_text(node)}"
    if isinstance(node, ast.ImplBlock):
        trait = f"{ast.type_display_name(node.trait)} for " if node.trait else ""
        return f"impl{_generic_text(node)} {trait}{ast.type_display_name(node.ty)}"
    if isinstance(node, ast.TraitDecl):
        return f"{'pub ' if node.is_pub else ''}trait {node.name}{_generic_text(node)}"
    if isinstance(node, ast.TraitItem):
        return "trait item"
    if isinstance(node, ast.GenericParam):
        bounds = "".join(f" + {bound.name}" for bound in node.bounds)
        return f"{node.name}{bounds}"
    if isinstance(node, ast.FnParam):
        suffix = f": {ast.type_display_name(node.ty)}" if node.ty else ""
        return f"{_pattern_text(node.pattern)}{suffix}"
    if isinstance(node, ast.EnumVariant):
        if node.fields is None:
            return node.name
        fields = ", ".join(ast.type_display_name(item) for item in node.fields)
        return f"{node.name}({fields})"
    if isinstance(node, ast.LetStmt):
        mutable = "mut " if node.mutable else ""
        ty = f": {ast.type_display_name(node.ty)}" if node.ty else ""
        return f"let {mutable}{_pattern_text(node.pattern)}{ty}"
    if isinstance(node, ast.ExprStmt):
        return "expression statement"
    if isinstance(node, ast.ItemStmt):
        return "item statement"
    if isinstance(node, ast.ConstDecl):
        kind = "static" if node.is_static else "const"
        return f"{'pub ' if node.is_pub else ''}{kind} {node.name}"
    if isinstance(node, ast.TypeAlias):
        return f"{'pub ' if node.is_pub else ''}type {node.name}"
    if isinstance(node, ast.UseDecl):
        return f"use {node.path}"
    if isinstance(node, ast.ModDecl):
        return f"mod {node.name}"
    if isinstance(node, ast.IncludeDecl):
        return f'include!("{node.path}")'
    if isinstance(node, ast.Attr):
        return f"#[{node.name}]"
    if isinstance(node, ast.MatchArm):
        guard = f" if {_expr_text(node.guard)}" if node.guard else ""
        return f"{_pattern_text(node.pattern)}{guard} => ..."
    if isinstance(node, ast.Block):
        return "block"
    if isinstance(node, ast.LitInt):
        return node.raw
    if isinstance(node, ast.LitFloat):
        return node.raw
    if isinstance(node, ast.LitStr):
        text = json.dumps(node.value)
        return text if len(text) <= 80 else text[:77] + "..."
    if isinstance(node, ast.LitChar):
        return repr(node.value)
    if isinstance(node, ast.LitBool):
        return "true" if node.value else "false"
    if isinstance(node, ast.Ident):
        return node.name
    if isinstance(node, ast.Path):
        base = "::".join(node.segments)
        if node.generic_args:
            args = ", ".join(ast.type_display_name(arg) for arg in node.generic_args)
            return f"{base}::<{args}>"
        return base
    if isinstance(node, ast.Binary):
        return f"Binary {node.op}"
    if isinstance(node, ast.Unary):
        return f"Unary {node.op}"
    if isinstance(node, ast.Assign):
        return "Assign"
    if isinstance(node, ast.CompoundAssign):
        return f"CompoundAssign {node.op}"
    if isinstance(node, ast.Call):
        return "Call"
    if isinstance(node, ast.MethodCall):
        return f"MethodCall .{node.method}()"
    if isinstance(node, ast.Field):
        return f"Field .{node.name}"
    if isinstance(node, ast.Index):
        return "Index"
    if isinstance(node, ast.StructLiteral):
        return f"StructLiteral {node.path}"
    if isinstance(node, ast.IfExpr):
        return "if"
    if isinstance(node, ast.IfLet):
        return f"if let {_pattern_text(node.pattern)}"
    if isinstance(node, ast.MatchExpr):
        return "match"
    if isinstance(node, ast.Loop):
        return "loop"
    if isinstance(node, ast.While):
        return "while"
    if isinstance(node, ast.For):
        return f"for {_pattern_text(node.pattern)} in ..."
    if isinstance(node, ast.Closure):
        return "closure"
    if isinstance(node, ast.Range):
        return f"Range {'..=' if node.inclusive else '..'}"
    if isinstance(node, ast.Cast):
        return f"Cast as {ast.type_display_name(node.ty)}"
    if isinstance(node, ast.Return):
        return "return"
    if isinstance(node, ast.Break):
        return "break"
    if isinstance(node, ast.Continue):
        return "continue"
    if isinstance(node, ast.MacroCall):
        return f"{node.path}!"
    if isinstance(node, ast.Await):
        return "await"
    if isinstance(node, ast.Try):
        return "? operator"
    if isinstance(node, ast.Paren):
        return "(...)"
    if isinstance(node, ast.OptionSome):
        return "Some(...)"
    if isinstance(node, ast.OptionNone):
        return "None"
    if isinstance(node, ast.ResultOk):
        return "Ok(...)"
    if isinstance(node, ast.ResultErr):
        return "Err(...)"
    if isinstance(node, ast.TypeName):
        return ast.type_display_name(node)
    if isinstance(node, (ast.TypeRef, ast.TypeSlice, ast.TypeArray, ast.TypeTuple)):
        return ast.type_display_name(node)
    if isinstance(node, ast.TypeFn):
        return ast.type_display_name(node)
    if isinstance(node, ast.TypeImplTrait):
        return "impl " + " + ".join(bound.name for bound in node.bounds)
    if isinstance(node, ast.TypeDynTrait):
        return "dyn " + " + ".join(bound.name for bound in node.bounds)
    if isinstance(node, ast.TypeUnit):
        return "()"
    if isinstance(node, ast.PatIdent):
        return _pattern_text(node)
    if isinstance(node, ast.PatWildcard):
        return "_"
    if isinstance(node, ast.PatTuple):
        return _pattern_text(node)
    if isinstance(node, ast.PatStruct):
        return _pattern_text(node)
    if isinstance(node, ast.PatEnum):
        return _pattern_text(node)
    if isinstance(node, ast.PatOr):
        return _pattern_text(node)
    if isinstance(node, ast.PatRange):
        return _pattern_text(node)
    if isinstance(node, ast.PatBinding):
        return _pattern_text(node)
    if isinstance(node, ast.PatRef):
        return _pattern_text(node)
    if isinstance(node, ast.PatRest):
        return ".."
    if isinstance(node, ast.PatBox):
        return _pattern_text(node)
    if isinstance(node, ast.PatConst):
        return node.path
    if isinstance(node, ast.PatMacro):
        return f"{node.path}!"
    if isinstance(node, ast.PatPath):
        return node.path
    if isinstance(node, ast.PatSlice):
        return _pattern_text(node)
    if isinstance(node, ast.PatLiteral):
        return _pattern_text(node)
    return type(node).__name__


def _synthetic_children(node: Any) -> tuple[list[tuple[str, Any]], set[str]]:
    if isinstance(node, ast.StructDecl):
        children = []
        for name, ty, is_pub in node.fields:
            label = f"{'pub ' if is_pub else ''}{name}: {ast.type_display_name(ty)}"
            children.append(
                (
                    name,
                    _Synthetic(
                        "StructField",
                        label,
                        "struct",
                        [("type", ty)],
                        node.line,
                        node.col,
                    ),
                )
            )
        return children, {"fields"}
    if isinstance(node, ast.StructLiteral):
        children = []
        for name, value in node.fields:
            children.append(
                (
                    name,
                    _Synthetic(
                        "FieldInit",
                        name,
                        "expression",
                        [("value", value)],
                        node.line,
                        node.col,
                    ),
                )
            )
        return children, {"fields"}
    if isinstance(node, ast.Closure):
        children = []
        for index, (pattern, ty) in enumerate(node.params):
            label = _pattern_text(pattern)
            nested: list[tuple[str, Any]] = []
            if ty is not None:
                label += f": {ast.type_display_name(ty)}"
                nested.append(("type", ty))
            children.append(
                (
                    f"param[{index}]",
                    _Synthetic(
                        "ClosureParam",
                        label,
                        "pattern",
                        nested,
                        node.line,
                        node.col,
                    ),
                )
            )
        return children, {"params"}
    return [], set()


def _child_nodes(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, _Synthetic):
        return list(node.children)
    synthetic, handled = _synthetic_children(node)
    children: list[tuple[str, Any]] = list(synthetic)
    for field in dataclasses.fields(node):
        if field.name in _SKIP_FIELDS or field.name in handled:
            continue
        value = getattr(node, field.name)
        if isinstance(value, ast.Node):
            children.append((field.name, value))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.Node):
                    children.append((f"{field.name}[{index}]", item))
                elif isinstance(item, tuple):
                    children.extend(_tuple_children(field.name, index, item))
        elif isinstance(value, tuple):
            children.extend(_tuple_children(field.name, None, value))
    return children


def _tuple_children(
    field_name: str, index: int | None, value: tuple[Any, ...]
) -> list[tuple[str, Any]]:
    prefix = field_name if index is None else f"{field_name}[{index}]"
    if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], ast.Node):
        return [
            (
                f"{prefix}.{value[0]}",
                _Synthetic(
                    "FieldInit",
                    value[0],
                    "expression",
                    [("value", value[1])],
                ),
            )
        ]
    if value and isinstance(value[0], ast.Node):
        return [
            (f"{prefix}[{pos}]", item)
            for pos, item in enumerate(value)
            if isinstance(item, ast.Node)
        ]
    return []


def _primitive_props(node: Any) -> dict[str, Any]:
    if isinstance(node, _Synthetic) or not dataclasses.is_dataclass(node):
        return {}
    props: dict[str, Any] = {}
    for field in dataclasses.fields(node):
        if field.name in _SKIP_FIELDS:
            continue
        value = getattr(node, field.name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if field.name in {"generics", "params", "items", "fields", "variants"}:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                props[field.name] = getattr(node, "raw", repr(value))
            else:
                props[field.name] = value
    return props


def _node_to_dict(
    node: Any,
    *,
    counter: list[int],
    depth: int = 0,
) -> dict[str, Any]:
    counter[0] += 1
    node_id = f"n{counter[0]}"
    if counter[0] > MAX_TREE_NODES:
        return {
            "id": node_id,
            "type": "Truncated",
            "category": "truncated",
            "label": "tree truncated",
            "line": None,
            "col": None,
            "span": None,
            "props": {},
            "children": [],
        }
    truncated = depth >= MAX_TREE_DEPTH
    children: list[dict[str, Any]] = []
    if not truncated:
        for role, child in _child_nodes(node):
            serialized = _node_to_dict(child, counter=counter, depth=depth + 1)
            serialized["role"] = role
            children.append(serialized)
    span = getattr(node, "span", None)
    return {
        "id": node_id,
        "type": getattr(node, "type_name", type(node).__name__),
        "category": _category(node),
        "label": _label(node),
        "line": getattr(node, "line", None),
        "col": getattr(node, "col", None),
        "span": list(span) if isinstance(span, (tuple, list)) else None,
        "props": _primitive_props(node),
        "children": children,
    }


def program_to_tree(program: ast.Program) -> dict[str, Any]:
    """Serialize an AST to a JSON-safe, color-categorised tree."""
    return _node_to_dict(program, counter=[0])


# ---------------------------------------------------------------------------
# Concrete source rendering
# ---------------------------------------------------------------------------

_BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    ">": 4,
    "<=": 4,
    ">=": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}
_UNARY_PRECEDENCE = 11
_POSTFIX_PRECEDENCE = 12
_ATOM_PRECEDENCE = 13


class SourcePrinter:
    """Render a concrete AST back to readable Polyglot source."""

    def __init__(self, *, indent: str = "    ") -> None:
        self.indent = indent

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def generic_params(node: Any) -> str:
        generics = getattr(node, "generics", None) or []
        if not generics:
            return ""
        parts = []
        for param in generics:
            bounds = getattr(param, "bounds", None) or []
            suffix = "".join(f" + {bound.name}" for bound in bounds)
            parts.append(f"{param.name}{suffix}")
        return f"<{', '.join(parts)}>"

    def type(self, ty: ast.Type | None) -> str:
        return ast.type_display_name(ty)

    def pattern(self, pattern: Any) -> str:
        return _pattern_text(pattern)

    def block(self, block: ast.Block, level: int) -> str:
        lines = ["{"]
        for stmt in block.stmts:
            lines.append(self.stmt(stmt, level + 1))
        if block.tail is not None:
            lines.append(self.pad(level + 1) + self.expr(block.tail))
        lines.append(self.pad(level) + "}")
        return "\n".join(lines)

    def pad(self, level: int) -> str:
        return self.indent * level

    # -- program / items -------------------------------------------------
    def program(self, program: ast.Program) -> str:
        rendered = [self.item(item, 0) for item in program.items]
        return "\n\n".join(part for part in rendered if part).rstrip() + "\n"

    def item(self, node: Any, level: int) -> str:
        prefix = self.pad(level)
        if isinstance(node, ast.FnDecl):
            return prefix + self.fn(node, level)
        if isinstance(node, ast.StructDecl):
            return prefix + self.struct(node, level)
        if isinstance(node, ast.EnumDecl):
            return prefix + self.enum(node, level)
        if isinstance(node, ast.ImplBlock):
            return prefix + self.impl(node, level)
        if isinstance(node, ast.TraitDecl):
            return prefix + self.trait(node, level)
        if isinstance(node, ast.ConstDecl):
            visibility = "pub " if node.is_pub else ""
            keyword = "static" if node.is_static else "const"
            ty = f": {self.type(node.ty)}" if node.ty else ""
            return f"{prefix}{visibility}{keyword} {node.name}{ty} = {self.expr(node.value)};"
        if isinstance(node, ast.TypeAlias):
            visibility = "pub " if node.is_pub else ""
            generics = self.generic_params(node)
            return f"{prefix}{visibility}type {node.name}{generics} = {self.type(node.ty)};"
        if isinstance(node, ast.UseDecl):
            return f"{prefix}use {node.path};"
        if isinstance(node, ast.ModDecl):
            if node.items is None:
                return f"{prefix}mod {node.name};"
            body = "\n".join(self.item(item, level + 1) for item in node.items)
            return f"{prefix}mod {node.name} {{\n{body}\n{prefix}}}"
        if isinstance(node, ast.IncludeDecl):
            return f'{prefix}include!("{node.path}");'
        return f"{prefix}/* {type(node).__name__} */"

    def fn(self, node: ast.FnDecl, level: int) -> str:
        prefix = ""
        if node.is_pub:
            prefix += "pub "
        if node.is_async:
            prefix += "async "
        if node.is_unsafe:
            prefix += "unsafe "
        attribute = "#[terminates]\n" + self.pad(level) if node.terminates else ""
        generics = self.generic_params(node)
        params = ", ".join(self.param(param) for param in node.params)
        ret = f" -> {self.type(node.ret)}" if node.ret else ""
        signature = f"{attribute}{prefix}fn {node.name}{generics}({params}){ret}"
        if node.body is None:
            return signature + ";"
        return signature + " " + self.block(node.body, level)

    def param(self, param: ast.FnParam) -> str:
        ty = f": {self.type(param.ty)}" if param.ty else ""
        return f"{self.pattern(param.pattern)}{ty}"

    def struct(self, node: ast.StructDecl, level: int) -> str:
        visibility = "pub " if node.is_pub else ""
        generics = self.generic_params(node)
        if not node.fields:
            return f"{visibility}struct {node.name}{generics} {{}}"
        lines = [f"{visibility}struct {node.name}{generics} {{"]
        for name, ty, is_pub in node.fields:
            field_visibility = "pub " if is_pub else ""
            lines.append(
                f"{self.pad(level + 1)}{field_visibility}{name}: {self.type(ty)},"
            )
        lines.append(self.pad(level) + "}")
        return "\n".join(lines)

    def enum(self, node: ast.EnumDecl, level: int) -> str:
        visibility = "pub " if node.is_pub else ""
        generics = self.generic_params(node)
        lines = [f"{visibility}enum {node.name}{generics} {{"]
        for variant in node.variants:
            if variant.fields is None:
                lines.append(f"{self.pad(level + 1)}{variant.name},")
            else:
                fields = ", ".join(self.type(ty) for ty in variant.fields)
                lines.append(f"{self.pad(level + 1)}{variant.name}({fields}),")
        lines.append(self.pad(level) + "}")
        return "\n".join(lines)

    def impl(self, node: ast.ImplBlock, level: int) -> str:
        generics = self.generic_params(node)
        trait = f"{self.type(node.trait)} for " if node.trait else ""
        header = f"impl{generics} {trait}{self.type(node.ty)} {{"
        body = "\n\n".join(self.item(item, level + 1) for item in node.items)
        return f"{header}\n{body}\n{self.pad(level)}}}"

    def trait(self, node: ast.TraitDecl, level: int) -> str:
        visibility = "pub " if node.is_pub else ""
        generics = self.generic_params(node)
        lines = [f"{visibility}trait {node.name}{generics} {{"]
        for item in node.items:
            if isinstance(item, ast.TraitItem) and item.fn is not None:
                lines.append(self.pad(level + 1) + self.fn(item.fn, level + 1))
            elif isinstance(item, ast.TraitItem) and item.ty is not None:
                lines.append(f"{self.pad(level + 1)}type Alias = {self.type(item.ty)};")
        lines.append(self.pad(level) + "}")
        return "\n".join(lines)

    # -- statements ------------------------------------------------------
    def stmt(self, stmt: Any, level: int) -> str:
        prefix = self.pad(level)
        if isinstance(stmt, ast.LetStmt):
            mutable = "mut " if stmt.mutable else ""
            ty = f": {self.type(stmt.ty)}" if stmt.ty else ""
            value = f" = {self.expr(stmt.value)}" if stmt.value is not None else ""
            return f"{prefix}let {mutable}{self.pattern(stmt.pattern)}{ty}{value};"
        if isinstance(stmt, ast.ExprStmt):
            expr = self.expr(stmt.expr)
            return f"{prefix}{expr};"
        if isinstance(stmt, ast.ItemStmt):
            return self.item(stmt.item, level)
        return f"{prefix}/* {type(stmt).__name__} */"

    # -- expressions -----------------------------------------------------
    def expr(self, node: Any, parent_precedence: int = 0) -> str:
        if node is None:
            return ""
        text, precedence = self._expr(node)
        if precedence < parent_precedence:
            return f"({text})"
        return text

    def _expr(self, node: Any) -> tuple[str, int]:
        if isinstance(node, ast.LitInt):
            return node.raw, _ATOM_PRECEDENCE
        if isinstance(node, ast.LitFloat):
            return node.raw, _ATOM_PRECEDENCE
        if isinstance(node, ast.LitStr):
            return json.dumps(node.value), _ATOM_PRECEDENCE
        if isinstance(node, ast.LitChar):
            return repr(node.value), _ATOM_PRECEDENCE
        if isinstance(node, ast.LitBool):
            return ("true" if node.value else "false"), _ATOM_PRECEDENCE
        if isinstance(node, ast.Ident):
            return node.name, _ATOM_PRECEDENCE
        if isinstance(node, ast.Path):
            base = "::".join(node.segments)
            if node.generic_args:
                args = ", ".join(self.type(arg) for arg in node.generic_args)
                return f"{base}::<{args}>", _ATOM_PRECEDENCE
            return base, _ATOM_PRECEDENCE
        if isinstance(node, ast.Unary):
            operand = self.expr(node.operand, _UNARY_PRECEDENCE)
            if node.op in {"&", "&mut"}:
                return f"&{'mut ' if node.mutable else ''}{operand}", _UNARY_PRECEDENCE
            return f"{node.op}{operand}", _UNARY_PRECEDENCE
        if isinstance(node, ast.Binary):
            precedence = _BINARY_PRECEDENCE.get(node.op, 9)
            left = self.expr(node.left, precedence)
            right = self.expr(node.right, precedence + 1)
            return f"{left} {node.op} {right}", precedence
        if isinstance(node, ast.Assign):
            return f"{self.expr(node.target)} = {self.expr(node.value)}", 0
        if isinstance(node, ast.CompoundAssign):
            return f"{self.expr(node.target)} {node.op} {self.expr(node.value)}", 0
        if isinstance(node, ast.Call):
            args = ", ".join(self.expr(arg) for arg in node.args)
            return (
                f"{self.expr(node.func, _POSTFIX_PRECEDENCE)}({args})",
                _POSTFIX_PRECEDENCE,
            )
        if isinstance(node, ast.MethodCall):
            receiver = self.expr(node.receiver, _POSTFIX_PRECEDENCE)
            generics = ""
            if node.generic_args:
                args = ", ".join(self.type(arg) for arg in node.generic_args)
                generics = f"::<{args}>"
            args = ", ".join(self.expr(arg) for arg in node.args)
            return (
                f"{receiver}.{node.method}{generics}({args})",
                _POSTFIX_PRECEDENCE,
            )
        if isinstance(node, ast.Field):
            return (
                f"{self.expr(node.receiver, _POSTFIX_PRECEDENCE)}.{node.name}",
                _POSTFIX_PRECEDENCE,
            )
        if isinstance(node, ast.Index):
            receiver = self.expr(node.receiver, _POSTFIX_PRECEDENCE)
            return f"{receiver}[{self.expr(node.index)}]", _POSTFIX_PRECEDENCE
        if isinstance(node, ast.TupleExpr):
            inner = ", ".join(self.expr(item) for item in node.elems)
            if len(node.elems) == 1:
                inner += ","
            return f"({inner})", _ATOM_PRECEDENCE
        if isinstance(node, ast.ArrayExpr):
            inner = ", ".join(self.expr(item) for item in node.elems)
            return f"[{inner}]", _ATOM_PRECEDENCE
        if isinstance(node, ast.RepeatArray):
            return (
                f"[{self.expr(node.elem)}; {self.expr(node.count)}]",
                _ATOM_PRECEDENCE,
            )
        if isinstance(node, ast.StructLiteral):
            fields = ", ".join(
                f"{name}: {self.expr(value)}" for name, value in node.fields
            )
            if node.rest is not None:
                fields += f", ..{self.expr(node.rest)}"
            return f"{node.path} {{ {fields} }}", _ATOM_PRECEDENCE
        if isinstance(node, ast.IfExpr):
            text = f"if {self.expr(node.cond)} {self.block(node.then_block, 0)}"
            text += self._else_suffix(node.else_branch)
            return text, 0
        if isinstance(node, ast.IfLet):
            text = (
                f"if let {self.pattern(node.pattern)} = {self.expr(node.value)} "
                f"{self.block(node.then_block, 0)}"
            )
            text += self._else_suffix(node.else_branch)
            return text, 0
        if isinstance(node, ast.MatchExpr):
            arms = []
            for arm in node.arms:
                guard = f" if {self.expr(arm.guard)}" if arm.guard else ""
                arms.append(
                    f"{self.pattern(arm.pattern)}{guard} => {self.expr(arm.body)},"
                )
            return (
                f"match {self.expr(node.scrutinee)} {{ " + " ".join(arms) + " }",
                0,
            )
        if isinstance(node, ast.Block):
            return self.block(node, 0), 0
        if isinstance(node, ast.Loop):
            return f"loop {self.block(node.body, 0)}", 0
        if isinstance(node, ast.While):
            return f"while {self.expr(node.cond)} {self.block(node.body, 0)}", 0
        if isinstance(node, ast.For):
            return (
                f"for {self.pattern(node.pattern)} in {self.expr(node.iter)} "
                f"{self.block(node.body, 0)}",
                0,
            )
        if isinstance(node, ast.Closure):
            params = []
            for pattern, ty in node.params:
                text = self.pattern(pattern)
                if ty is not None:
                    text += f": {self.type(ty)}"
                params.append(text)
            ret = f" -> {self.type(node.ret)}" if node.ret else ""
            return f"|{', '.join(params)}|{ret} {self.expr(node.body)}", 0
        if isinstance(node, ast.Range):
            start = self.expr(node.start) if node.start is not None else ""
            end = self.expr(node.end) if node.end is not None else ""
            return f"{start}{'..=' if node.inclusive else '..'}{end}", 0
        if isinstance(node, ast.Cast):
            return (
                f"{self.expr(node.expr, _POSTFIX_PRECEDENCE)} as {self.type(node.ty)}",
                _POSTFIX_PRECEDENCE,
            )
        if isinstance(node, ast.Return):
            if node.value is None:
                return "return", 0
            return f"return {self.expr(node.value)}", 0
        if isinstance(node, ast.Break):
            if node.value is None:
                return "break", 0
            return f"break {self.expr(node.value)}", 0
        if isinstance(node, ast.Continue):
            return "continue", 0
        if isinstance(node, ast.MacroCall):
            if isinstance(node.args, str):
                return f"{node.path}!({node.args})", _POSTFIX_PRECEDENCE
            args = ", ".join(self.expr(arg) for arg in node.args)
            return f"{node.path}!({args})", _POSTFIX_PRECEDENCE
        if isinstance(node, ast.Await):
            return (
                f"{self.expr(node.expr, _POSTFIX_PRECEDENCE)}.await",
                _POSTFIX_PRECEDENCE,
            )
        if isinstance(node, ast.Try):
            return f"{self.expr(node.expr, _POSTFIX_PRECEDENCE)}?", _POSTFIX_PRECEDENCE
        if isinstance(node, ast.Paren):
            return f"({self.expr(node.inner)})", _ATOM_PRECEDENCE
        if isinstance(node, ast.OptionSome):
            return f"Some({self.expr(node.value)})", _POSTFIX_PRECEDENCE
        if isinstance(node, ast.OptionNone):
            return "None", _ATOM_PRECEDENCE
        if isinstance(node, ast.ResultOk):
            return f"Ok({self.expr(node.value)})", _POSTFIX_PRECEDENCE
        if isinstance(node, ast.ResultErr):
            return f"Err({self.expr(node.value)})", _POSTFIX_PRECEDENCE
        return f"/* {type(node).__name__} */", _ATOM_PRECEDENCE

    def _else_suffix(self, branch: Any) -> str:
        if branch is None:
            return ""
        if isinstance(branch, ast.Block):
            return " else " + self.block(branch, 0)
        if isinstance(branch, ast.IfExpr):
            return " else " + self.expr(branch)
        if isinstance(branch, ast.IfLet):
            return " else " + self.expr(branch)
        return " else " + self.expr(branch)


def program_to_source(program: ast.Program) -> str:
    """Render a program back to concrete Polyglot source."""
    return SourcePrinter().program(program)


# ---------------------------------------------------------------------------
# Monomorphization summary
# ---------------------------------------------------------------------------


def _item_kind(item: Any) -> str:
    return _ITEM_LABELS.get(type(item).__name__, type(item).__name__)


def _generic_items(program: ast.Program) -> list[Any]:
    return [
        item
        for item in program.items
        if getattr(item, "generics", None)
        and isinstance(item, (ast.FnDecl, ast.StructDecl, ast.EnumDecl, ast.ImplBlock))
    ]


def _instantiation_summary(
    original: ast.Program, concrete: ast.Program
) -> list[dict[str, Any]]:
    generic_bases = {
        item.name: _item_kind(item)
        for item in _generic_items(original)
        if hasattr(item, "name")
    }
    before = {item.name for item in original.items if hasattr(item, "name")}
    summary: list[dict[str, Any]] = []
    for item in concrete.items:
        name = getattr(item, "name", None)
        if not isinstance(name, str) or name in before:
            continue
        for base, kind in sorted(generic_bases.items(), key=lambda pair: -len(pair[0])):
            if name == base or name.startswith(base + "_"):
                summary.append(
                    {
                        "base": base,
                        "concrete": name,
                        "kind": kind,
                        "signature": _label(item),
                    }
                )
                break
    summary.sort(key=lambda entry: (entry["base"], entry["concrete"]))
    return summary


def analyze(source: str) -> dict[str, Any]:
    """Parse, serialize, and monomorphize ``source`` for the visualizer."""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "ok": False,
        "source": source,
        "diagnostics": [],
        "original": None,
        "monomorphized": None,
        "timing": {},
    }
    parse_start = time.perf_counter()
    try:
        program = parse(lex(source))
    except (LexError, ParseError) as error:
        result["diagnostics"] = [diagnostic(error, source, phase="parse")]
        result["timing"] = {
            "parse_ms": round((time.perf_counter() - parse_start) * 1000, 3)
        }
        return result
    parse_ms = (time.perf_counter() - parse_start) * 1000

    original_tree = program_to_tree(program)
    result["original"] = {
        "tree": original_tree,
        "item_count": len(program.items),
        "generic_count": len(_generic_items(program)),
        "source": source,
    }

    mono_start = time.perf_counter()
    try:
        concrete = monomorphize(program)
    except MonomorphizationError as error:
        result["diagnostics"] = [diagnostic(error, source, phase="monomorphize")]
        result["timing"] = {
            "parse_ms": round(parse_ms, 3),
            "monomorphize_ms": round((time.perf_counter() - mono_start) * 1000, 3),
        }
        return result

    mono_ms = (time.perf_counter() - mono_start) * 1000
    result["monomorphized"] = {
        "tree": program_to_tree(concrete),
        "source": program_to_source(concrete),
        "item_count": len(concrete.items),
        "instantiations": _instantiation_summary(program, concrete),
    }
    result["ok"] = True
    result["timing"] = {
        "parse_ms": round(parse_ms, 3),
        "monomorphize_ms": round(mono_ms, 3),
        "total_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    return result


__all__ = [
    "MAX_TREE_DEPTH",
    "MAX_TREE_NODES",
    "FixHint",
    "SourcePrinter",
    "analyze",
    "diagnostic",
    "fix_hint",
    "lsp_position_to_offset",
    "offset_to_lsp_position",
    "program_to_source",
    "program_to_tree",
    "span_to_lsp_range",
]
