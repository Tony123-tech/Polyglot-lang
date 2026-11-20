"""Frame 8: diagnostics, error reporting, and telemetry helpers.

The functions in this module form a small, deterministic pipeline:

``context -> error -> map/format/aggregate -> redact -> measure/export``.

Everything is pure Python with no third-party dependencies.  Telemetry inputs
are explicit samples (or an injected wrapper/clock) so reports are reproducible
in tests and CI; no background thread, environment probe, or wall-clock read is
performed unless a caller supplies a clock/wrapper explicitly.  ``PillarError``
and the E001-E008 namespace come from :mod:`runtime.diagnostics`, span helpers
come from :mod:`api.visualizer`, and backend-specific error classes come from
:mod:`backends.errors`.

Platform boundaries: path redaction is textual and portable (POSIX and Windows
spellings are both recognised); the optional default home/workspace prefixes
are derived from the current platform but can always be overridden by callers.
Benchmark timings are *models* of the reported measurements, not executions of
the Python/Rust/Zig backends.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from api.visualizer import (
    diagnostic as visualizer_diagnostic,
)
from api.visualizer import (
    offset_to_lsp_position,
    span_to_lsp_range,
)
from backends.errors import (
    InvalidIntegerParseLiteralError,
    TranslateError,
    UnsupportedNativePatternError,
)
from runtime.diagnostics import ERROR_CODES, PillarError

SCHEMA_DIAGNOSTICS: Final = "polyglot-diagnostics/v1"
SCHEMA_AGGREGATE: Final = "polyglot-diagnostic-aggregate/v1"
SCHEMA_TIMING: Final = "polyglot-compile-timing/v1"
SCHEMA_CACHE: Final = "polyglot-cache-metrics/v1"
SCHEMA_BENCHMARK: Final = "polyglot-benchmark-summary/v1"

CANONICAL_PHASES: Final[tuple[str, ...]] = (
    "lexing",
    "parsing",
    "typecheck",
    "monomorphization",
    "codegen",
)
SEVERITIES: Final[frozenset[str]] = frozenset({"error", "warning", "info"})

_PHASE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lex": "lexing",
        "lexer": "lexing",
        "tokenize": "lexing",
        "tokenizing": "lexing",
        "tokens": "lexing",
        "parse": "parsing",
        "parser": "parsing",
        "syntax": "parsing",
        "type": "typecheck",
        "type_check": "typecheck",
        "typechecking": "typecheck",
        "infer": "typecheck",
        "mono": "monomorphization",
        "monomorph": "monomorphization",
        "monomorphize": "monomorphization",
        "emit": "codegen",
        "emit_python": "codegen",
        "emit_rust": "codegen",
        "lower": "codegen",
    }
)
_BACKEND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "py": "python",
        "cpython": "python",
        "python3": "python",
        "rs": "rust",
        "pyo3": "rust",
        "native": "rust",
        "rustc": "rust",
        "zig": "zig",
        "wasm": "wasm",
    }
)
_UNIT_TO_SECONDS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 0.001,
        "msec": 0.001,
        "millisecond": 0.001,
        "milliseconds": 0.001,
        "ns": 1e-9,
        "nsec": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
)
_UNIT_TO_MS: Final[Mapping[str, float]] = MappingProxyType(
    {name: scale * 1000.0 for name, scale in _UNIT_TO_SECONDS.items()}
)
_EXCEPTION_CODE_HINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "UnsupportedNativePatternError": "E007",
        "InvalidIntegerParseLiteralError": "E008",
        "TranslateError": "E003",
        "LexError": "E003",
        "ParseError": "E003",
        "TypeCheckError": "E001",
        "LayoutError": "E001",
        "AbiError": "E001",
        "MonomorphizationError": "E001",
        "TerminationError": "E002",
        "ProofError": "E002",
        "BorrowError": "E002",
        "MoveError": "E002",
        "LifetimeError": "E002",
        "DependencyCycleError": "E005",
        "CycleError": "E005",
        "ModuleNotFoundError": "E006",
        "MissingSnapshotError": "E006",
        "IncludeError": "E006",
        "ToolchainError": "E004",
        "NativeCompileError": "E004",
        "CodegenError": "E004",
    }
)
# Longest keyword wins; the tuple order breaks ties deterministically.
_FAILURE_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "E007",
        (
            "unsupported native pattern",
            "platform feature",
            "feature gate",
            "not supported",
            "unavailable",
            "unsupported",
            "platform",
            "wasm",
        ),
    ),
    (
        "E008",
        (
            "invalid integer parse",
            "malformed serialized",
            "literal encoding",
            "snapshot digest",
            "certificate",
            "checksum",
            "deserialize",
            "serialization",
            "malformed",
            "decode",
            "digest",
            "literal",
            "snapshot",
        ),
    ),
    (
        "E005",
        ("circular dependency", "dependency cycle", "import cycle", "cycle"),
    ),
    (
        "E006",
        (
            "module not found",
            "missing module",
            "missing snapshot",
            "missing artifact",
            "file not found",
            "include not found",
            "no such file",
            "artifact missing",
        ),
    ),
    (
        "E004",
        (
            "compilation failed",
            "compiler failure",
            "native build",
            "toolchain",
            "codegen",
            "compiler",
            "linker",
            "cargo",
            "rustc",
            "pyo3",
            "zig",
        ),
    ),
    (
        "E002",
        (
            "proof obligation",
            "semantic proof",
            "runtime failure",
            "runtime panic",
            "exhaustiveness",
            "non-exhaustive",
            "termination",
            "ownership",
            "lifetime",
            "borrow",
            "safety",
            "runtime",
            "move",
            "panic",
            "proof",
            "trap",
        ),
    ),
    (
        "E001",
        (
            "signature mismatch",
            "type mismatch",
            "incompatible type",
            "trait bound",
            "typecheck",
            "type check",
            "layout",
            "abi",
            "type",
        ),
    ),
    (
        "E003",
        (
            "unexpected token",
            "api misuse",
            "lexer",
            "parser",
            "syntax",
            "token",
            "translate",
            "translation",
            "macro",
            "arity",
            "ast",
            "lex",
            "parse",
        ),
    ),
)
_PHASE_CODE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lexing": "E003",
        "parsing": "E003",
        "typecheck": "E001",
        "monomorphization": "E001",
        "codegen": "E004",
    }
)
_ANSI: Final[Mapping[str, str]] = MappingProxyType(
    {"error": "\x1b[31m", "warning": "\x1b[33m", "info": "\x1b[36m"}
)
_ANSI_RESET: Final = "\x1b[0m"
_TAB_WIDTH: Final = 4


@dataclass(frozen=True)
class DiagnosticContext:
    """Reusable source/phase/backend context shared by Frame 8 helpers."""

    source: str
    phase: str
    backend: str = "python"
    code_map: Mapping[str, str] = field(default_factory=dict)
    path: str | None = None

    def code_for(self, key: str, default: str | None = None) -> str | None:
        """Resolve a diagnostic alias through the context's code map."""
        text = str(key).strip()
        for candidate in (text, text.lower(), text.upper()):
            mapped = self.code_map.get(candidate)
            if mapped is not None:
                return mapped
        return default

    def span_range(
        self,
        span: tuple[int, int] | list[int] | None,
        *,
        line: int | None = None,
        col: int | None = None,
    ) -> dict[str, Any]:
        """Convert a character span to the shared 0-based LSP range shape."""
        return span_to_lsp_range(self.source, span, line=line, col=col)

    def offset_position(self, offset: int) -> dict[str, int]:
        """Convert a character offset to a 0-based LSP position."""
        return offset_to_lsp_position(self.source, offset)

    def line_text(self, line: int) -> str:
        """Return a 1-based source line, or an empty string when out of range."""
        lines = self.source.splitlines()
        if 1 <= line <= len(lines):
            return lines[line - 1]
        return ""

    def with_phase(self, phase: str) -> DiagnosticContext:
        """Return a copy of this context bound to another compiler phase."""
        return replace(self, phase=phase)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the context."""
        return {
            "source": self.source,
            "phase": self.phase,
            "backend": self.backend,
            "code_map": dict(self.code_map),
            "path": self.path,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow mapping-style access to the context fields."""
        return self.to_dict()[key]


@dataclass(frozen=True)
class DiagnosticRecord:
    """Normalized, JSON-safe description of one compiler diagnostic."""

    code: str
    message: str
    severity: str = "error"
    span: tuple[int, int] | None = None
    line: int = 1
    col: int = 1
    phase: str = "compile"
    backend: str = "python"
    help_text: str | None = None
    note: str | None = None
    path: str | None = None
    cause: int | str | None = None
    root_cause: bool = True
    cascading: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this record."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "span": list(self.span) if self.span is not None else None,
            "line": self.line,
            "col": self.col,
            "phase": self.phase,
            "backend": self.backend,
            "help": self.help_text,
            "note": self.note,
            "path": self.path,
            "cause": self.cause,
            "root_cause": self.root_cause,
            "cascading": self.cascading,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow mapping-style access to the record fields."""
        return self.to_dict()[key]


@dataclass(frozen=True)
class AggregatedDiagnostics:
    """A cascade-aware summary that never drops the first root cause."""

    root_cause: DiagnosticRecord | None
    diagnostics: tuple[DiagnosticRecord, ...]
    suppressed: int
    cascade_count: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the aggregate."""
        return {
            "schema": SCHEMA_AGGREGATE,
            "total": self.total,
            "cascade_count": self.cascade_count,
            "suppressed": self.suppressed,
            "truncated": self.suppressed > 0,
            "root_cause_preserved": self.root_cause is not None,
            "root_cause": self.root_cause.to_dict() if self.root_cause else None,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def __iter__(self):
        """Iterate over the retained diagnostics."""
        return iter(self.diagnostics)

    def __len__(self) -> int:
        """Return the number of retained diagnostics."""
        return len(self.diagnostics)

    def __getitem__(self, index: int) -> DiagnosticRecord:
        """Return a retained diagnostic by position."""
        return self.diagnostics[index]


def _canonical_backend(value: str | None) -> str:
    text = (value or "python").strip().lower()
    return _BACKEND_ALIASES.get(text, text or "python")


def _canonical_phase(value: str | None, *, default: str = "compile") -> str:
    text = (value or default).strip().lower()
    return _PHASE_ALIASES.get(text, text or default)


def _canonical_opt_level(value: str | None) -> str:
    text = (value or "O0").strip().upper()
    if re.fullmatch(r"[0-4]", text):
        return "O" + text
    return text or "O0"


def _unit_scale(unit: str, table: Mapping[str, float]) -> float:
    key = unit.strip().lower()
    try:
        return table[key]
    except KeyError as error:
        raise ValueError(f"unknown time unit {unit!r}") from error


def _as_nonnegative_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected a non-negative integer, got {value!r}") from error
    if result < 0:
        raise ValueError(f"expected a non-negative integer, got {value!r}")
    return result


def _as_finite_float(value: Any, *, label: str = "duration") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return result


def _normalize_severity(value: Any) -> str:
    text = str(value or "error").strip().lower()
    aliases = {"warn": "warning", "fatal": "error", "critical": "error"}
    text = aliases.get(text, text)
    if text not in SEVERITIES:
        raise ValueError(f"unknown severity {value!r}; expected error, warning or info")
    return text


def _normalized_code_text(value: str) -> str:
    text = value.strip().upper()
    return text if re.fullmatch(r"E00[1-8]", text) else ""


def _code_from_exception(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str):
        normalized = _normalized_code_text(code)
        if normalized:
            return normalized
    if isinstance(error, UnsupportedNativePatternError):
        return "E007"
    if isinstance(error, InvalidIntegerParseLiteralError):
        return "E008"
    if isinstance(error, TranslateError):
        return "E003"
    name = type(error).__name__
    hinted = _EXCEPTION_CODE_HINTS.get(name)
    if hinted:
        return hinted
    match = re.search(r"\bE00[1-8]\b", str(error).upper())
    return match.group(0) if match else None


def _normalize_code(
    value: str | BaseException | None,
    *,
    code_map: Mapping[str, str] | None = None,
    default: str | None = None,
) -> str:
    if isinstance(value, BaseException):
        candidate = _code_from_exception(value)
    elif isinstance(value, str):
        text = value.strip()
        mapped = None
        if code_map:
            for key in (text, text.lower(), text.upper()):
                if key in code_map:
                    mapped = code_map[key]
                    break
        candidate = _normalized_code_text(str(mapped or text))
        if not candidate:
            candidate = _EXCEPTION_CODE_HINTS.get(text)
    else:
        candidate = None
    if candidate and candidate in ERROR_CODES:
        return candidate
    if default is not None:
        fallback = _normalized_code_text(default)
        if fallback in ERROR_CODES:
            return fallback
    raise ValueError(f"unknown or unmappable diagnostic code {value!r}")


def _normalize_span(
    span: tuple[int, int] | list[int] | Sequence[Any] | None,
    source_length: int | None = None,
) -> tuple[int, int] | None:
    if not isinstance(span, (tuple, list)) or len(span) != 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except TypeError, ValueError, OverflowError:
        return None
    start, end = max(0, start), max(0, end)
    if end < start:
        start, end = end, start
    if source_length is not None:
        start, end = min(start, source_length), min(end, source_length)
    return start, end


def _line_col_to_offset(source: str, line: int, col: int) -> int:
    lines = source.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        return 0
    return sum(len(text) for text in lines[: line - 1]) + max(0, col - 1)


def _span_from_line_col(source: str, line: int, col: int) -> tuple[int, int] | None:
    if line < 1 or col < 1:
        return None
    offset = _line_col_to_offset(source, line, col)
    if offset > len(source):
        return None
    end = offset
    while end < len(source) and (source[end].isalnum() or source[end] == "_"):
        end += 1
    if end == offset and offset < len(source):
        end = offset + 1
    return offset, min(end, len(source))


def _clean_message(value: BaseException) -> str:
    reason = getattr(value, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    text = str(value).strip()
    first = text.splitlines()[0] if text else ""
    first = re.sub(r"^error(?:\[E\d+\])?:\s*", "", first).strip()
    return first or type(value).__name__


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _coerce_record(
    value: BaseException | Mapping[str, Any] | DiagnosticRecord,
    *,
    context: DiagnosticContext | None = None,
    phase: str | None = None,
    backend: str | None = None,
) -> DiagnosticRecord:
    source = context.source if context else ""
    base_phase = phase or (context.phase if context else "compile")
    base_backend = backend or (context.backend if context else "python")
    code_map = context.code_map if context else None
    if isinstance(value, DiagnosticRecord):
        return value
    if isinstance(value, BaseException):
        code = _normalize_code(value, code_map=code_map, default="E003")
        message = _clean_message(value)
        line = _as_nonnegative_int(getattr(value, "line", 0))
        col = _as_nonnegative_int(getattr(value, "col", 0))
        span = _normalize_span(getattr(value, "span", None), len(source) or None)
        if span is None and (line and col):
            span = _span_from_line_col(source, line, col)
        cause: int | str | None = None
        for key in ("caused_by", "cause", "parent"):
            raw_cause = getattr(value, key, None)
            if raw_cause is not None:
                cause = raw_cause if isinstance(raw_cause, int) else str(raw_cause)
                break
        return DiagnosticRecord(
            code=code,
            message=message,
            severity=_normalize_severity(getattr(value, "severity", "error")),
            span=span,
            line=max(1, line),
            col=max(1, col),
            phase=str(getattr(value, "phase", None) or base_phase),
            backend=_canonical_backend(getattr(value, "backend", None) or base_backend),
            help_text=getattr(value, "help_text", None),
            note=getattr(value, "note", None),
            path=getattr(value, "path", None) or (context.path if context else None),
            cause=cause,
            root_cause=bool(getattr(value, "root_cause", cause is None)),
            cascading=bool(getattr(value, "cascading", cause is not None)),
        )
    if isinstance(value, Mapping):
        code = _normalize_code(value.get("code"), code_map=code_map, default="E003")
        message = str(
            value.get("message")
            or value.get("reason")
            or value.get("detail")
            or "diagnostic"
        )
        line = _as_nonnegative_int(value.get("line", 0))
        col = _as_nonnegative_int(value.get("col", 0))
        span = _normalize_span(value.get("span"), len(source) or None)
        if span is None and (line and col):
            span = _span_from_line_col(source, line, col)
        if span is not None and not (line and col):
            position = offset_to_lsp_position(source, span[0])
            line, col = position["line"] + 1, position["character"] + 1
        cause = None
        for key in ("caused_by", "cause", "parent"):
            raw_cause = value.get(key)
            if raw_cause is not None:
                cause = raw_cause if isinstance(raw_cause, int) else str(raw_cause)
                break
        return DiagnosticRecord(
            code=code,
            message=message,
            severity=_normalize_severity(value.get("severity", "error")),
            span=span,
            line=max(1, line),
            col=max(1, col),
            phase=str(value.get("phase") or base_phase),
            backend=_canonical_backend(value.get("backend") or base_backend),
            help_text=value.get("help") or value.get("help_text"),
            note=value.get("note"),
            path=value.get("path") or (context.path if context else None),
            cause=cause,
            root_cause=bool(value.get("root_cause", cause is None)),
            cascading=bool(value.get("cascading", cause is not None)),
        )
    raise TypeError(
        f"expected an exception, mapping, or DiagnosticRecord; got {type(value).__name__}"
    )


def diag_create_diagnostic_context(
    source: str,
    phase: str = "compile",
    backend: str = "python",
    code_map: Mapping[str, str] | None = None,
    *,
    path: str | None = None,
) -> DiagnosticContext:
    """Construct a reusable source/phase/backend context for diagnostics."""
    if not isinstance(source, str):
        raise TypeError("diagnostic context source must be a string")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("diagnostic context phase must be a non-empty string")
    normalized_map: dict[str, str] = {}
    for key, value in (code_map or {}).items():
        normalized_map[str(key).strip().lower()] = _normalize_code(value)
    return DiagnosticContext(
        source=source,
        phase=phase.strip(),
        backend=_canonical_backend(backend),
        code_map=MappingProxyType(normalized_map),
        path=path,
    )


def diag_create_pillar_error(
    code: str | BaseException,
    message: str | None = None,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    note: str | None = None,
    severity: str = "error",
    line: int | None = None,
    col: int | None = None,
    span: tuple[int, int] | list[int] | None = None,
    phase: str | None = None,
    context: DiagnosticContext | None = None,
    code_map: Mapping[str, str] | None = None,
) -> PillarError:
    """Create a structured :class:`PillarError` with severity and span metadata."""
    merged_map: dict[str, str] = dict(context.code_map if context else {})
    merged_map.update(
        {str(key).lower(): value for key, value in (code_map or {}).items()}
    )
    source = context.source if context else ""
    from_exception = code if isinstance(code, BaseException) else None
    resolved = _normalize_code(code, code_map=merged_map or None)
    if from_exception is not None:
        text = message if message is not None else _clean_message(from_exception)
        resolved = _normalize_code(from_exception, code_map=merged_map or None)
        line = line if line is not None else getattr(from_exception, "line", 0)
        col = col if col is not None else getattr(from_exception, "col", 0)
        span = span if span is not None else getattr(from_exception, "span", None)
        reason = reason or getattr(from_exception, "reason", None)
        help_text = help_text or getattr(from_exception, "help_text", None)
    else:
        text = message if message is not None else f"diagnostic {resolved}"
    error = PillarError(
        resolved,
        text,
        reason=reason,
        help_text=help_text,
        note=note,
        line=_as_nonnegative_int(line),
        col=_as_nonnegative_int(col),
        span=_normalize_span(span, len(source) or None),
    )
    error.severity = _normalize_severity(severity)
    error.phase = phase or (context.phase if context else "compile")
    error.backend = context.backend if context else "python"
    if context and context.path:
        error.path = context.path
    return error


def diag_map_error_code_e001_to_e008(
    failure: str | BaseException,
    *,
    phase: str | None = None,
    default: str = "E003",
) -> str:
    """Map a lexer/parser/type/borrow/codegen/runtime failure to E001-E008."""
    fallback = _normalize_code(default)
    if isinstance(failure, str):
        text = failure.strip()
        exact = _normalized_code_text(text)
        if exact:
            return exact
        hint = _EXCEPTION_CODE_HINTS.get(text)
        if hint:
            return hint
        search_text = text.lower()
    else:
        direct = _code_from_exception(failure)
        if direct:
            return direct
        search_text = f"{type(failure).__name__} {failure}".lower()
    if phase:
        search_text += " " + _canonical_phase(phase, default="").lower()
    best: tuple[int, int, str] | None = None
    for code, keywords in _FAILURE_KEYWORDS:
        for keyword in keywords:
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])", search_text
            ):
                score = (len(keyword), -_FAILURE_KEYWORDS.index((code, keywords)), code)
                if best is None or score > best:
                    best = score
    if best is not None:
        return best[2]
    if phase:
        canonical = _canonical_phase(phase, default="")
        mapped = _PHASE_CODE.get(canonical)
        if mapped:
            return mapped
    return fallback


def diag_format_terminal_snippet(
    error: BaseException | Mapping[str, Any] | DiagnosticRecord,
    source: str | None = None,
    *,
    context: DiagnosticContext | None = None,
    span: tuple[int, int] | list[int] | None = None,
    line: int | None = None,
    col: int | None = None,
    severity: str | None = None,
    color: bool = False,
    max_lines: int = 3,
) -> str:
    """Render one diagnostic as source lines with a caret-annotated span."""
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")
    text = source if source is not None else (context.source if context else "")
    record = _coerce_record(
        error,
        context=context,
        phase=context.phase if context else None,
        backend=context.backend if context else None,
    )
    if span is not None:
        normalized = _normalize_span(span, len(text) or None)
        if normalized is None:
            raise ValueError("span must contain exactly two integer offsets")
        record = replace(record, span=normalized)
    if line is not None or col is not None:
        record = replace(
            record,
            line=max(1, _as_nonnegative_int(line, default=record.line)),
            col=max(1, _as_nonnegative_int(col, default=record.col)),
        )
    selected_severity = _normalize_severity(severity or record.severity)
    label = (
        "warning"
        if selected_severity == "warning"
        else ("info" if selected_severity == "info" else "error")
    )
    header = f"{label}[{record.code}]: {record.message}"
    if color:
        header = f"{_ANSI[selected_severity]}{header}{_ANSI_RESET}"
    output = [header]
    lsp_range = span_to_lsp_range(text, record.span, line=record.line, col=record.col)
    start, end = lsp_range["start"], lsp_range["end"]
    start_line = max(1, int(start["line"]) + 1)
    start_col = max(1, int(start["character"]) + 1)
    end_line = max(start_line, int(end["line"]) + 1)
    end_col = max(1, int(end["character"]) + 1)
    display_path = record.path or (context.path if context else None) or "<source>"
    output.append(f"  --> {display_path}:{start_line}:{start_col}")
    source_lines = text.splitlines()
    if not source_lines:
        output.append("   |  <no source available>")
    else:
        first_index = start_line - 1
        last_index = min(end_line - 1, len(source_lines) - 1)
        shown_indices = list(
            range(first_index, min(last_index, first_index + max_lines - 1) + 1)
        )
        gutter = len(str(shown_indices[-1] + 1)) if shown_indices else 1
        for index in shown_indices:
            number = index + 1
            raw = source_lines[index]
            low = start_col - 1 if number == start_line else 0
            high = end_col - 1 if number == end_line else len(raw)
            if end_line > start_line and number == end_line and high == 0:
                continue
            low = max(0, min(low, len(raw)))
            high = max(0, min(high, len(raw)))
            if high <= low:
                high = min(len(raw), low + 1)
            prefix = raw[:low]
            highlight = raw[low:high]
            prefix_width = len(prefix.expandtabs(_TAB_WIDTH))
            caret_width = max(
                1,
                len((prefix + highlight).expandtabs(_TAB_WIDTH)) - prefix_width,
            )
            caret = "^" * caret_width
            if color:
                caret = f"{_ANSI[selected_severity]}{caret}{_ANSI_RESET}"
            output.append(f"{number:>{gutter}} | {raw.expandtabs(_TAB_WIDTH)}")
            output.append(f"{' ' * gutter} | {' ' * prefix_width}{caret}")
        hidden = last_index - shown_indices[-1]
        if hidden > 0:
            suffix = "line" if hidden == 1 else "lines"
            output.append(f"{' ' * gutter} |  ... {hidden} more {suffix} suppressed")
        output.append(f"{' ' * gutter} |")
    if record.help_text:
        output.append(f"  help: {record.help_text}")
    if record.note:
        output.append(f"  note: {record.note}")
    output.append(f"  context: phase={record.phase} backend={record.backend}")
    return "\n".join(output)


def _mapping_diagnostic(
    value: Mapping[str, Any],
    source: str,
    phase: str,
    backend: str,
    code_map: Mapping[str, str] | None,
) -> dict[str, Any]:
    message = str(
        value.get("message")
        or value.get("reason")
        or value.get("detail")
        or "diagnostic"
    )
    line = _as_nonnegative_int(value.get("line", 0))
    col = _as_nonnegative_int(value.get("col", 0))
    span = _normalize_span(value.get("span"), len(source) or None)
    if span is None and (line and col):
        span = _span_from_line_col(source, line, col)
    if span is not None and not (line and col):
        position = offset_to_lsp_position(source, span[0])
        line, col = position["line"] + 1, position["character"] + 1
    excerpt = value.get("excerpt")
    if excerpt is None:
        lines = source.splitlines()
        excerpt = lines[line - 1] if 1 <= line <= len(lines) else ""
    return {
        "code": _normalize_code(value.get("code"), code_map=code_map, default="E003"),
        "phase": str(value.get("phase") or phase),
        "backend": _canonical_backend(value.get("backend") or backend),
        "severity": _normalize_severity(value.get("severity", "error")),
        "message": message,
        "note": str(value.get("note") or type(value).__name__),
        "line": max(1, line),
        "col": max(1, col),
        "span": list(span) if span is not None else None,
        "excerpt": str(excerpt),
        "detail": str(value.get("detail") or message),
        "help": value.get("help") or value.get("help_text"),
        "fix": _json_safe(value.get("fix")) if value.get("fix") is not None else None,
    }


def diag_format_json_payload(
    errors: (
        BaseException
        | Mapping[str, Any]
        | DiagnosticRecord
        | Iterable[BaseException | Mapping[str, Any] | DiagnosticRecord]
    ),
    source: str | None = None,
    *,
    phase: str | None = None,
    context: DiagnosticContext | None = None,
    backend: str | None = None,
    as_object: bool = False,
    pretty: bool = False,
) -> str | dict[str, Any]:
    """Serialize one or many diagnostics to a JSON-safe editor/tooling payload."""
    text = source if source is not None else (context.source if context else "")
    selected_phase = phase or (context.phase if context else "compile")
    selected_backend = _canonical_backend(
        backend or (context.backend if context else "python")
    )
    code_map = context.code_map if context else None
    if isinstance(errors, (BaseException, Mapping, DiagnosticRecord, str)):
        items: list[BaseException | Mapping[str, Any] | DiagnosticRecord] = [errors]  # type: ignore[list-item]
    else:
        items = list(errors)
    rendered: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, BaseException):
            payload = dict(visualizer_diagnostic(item, text, phase=selected_phase))
            payload["code"] = _normalize_code(
                item, code_map=code_map, default=payload.get("code")
            )
            payload["severity"] = _normalize_severity(
                getattr(item, "severity", "error")
            )
            payload["backend"] = selected_backend
            span = _normalize_span(payload.get("span"), len(text) or None)
            payload["span"] = list(span) if span is not None else None
            if span is not None and not (payload.get("line") and payload.get("col")):
                position = offset_to_lsp_position(text, span[0])
                payload["line"] = position["line"] + 1
                payload["col"] = position["character"] + 1
                lines = text.splitlines()
                payload["excerpt"] = (
                    lines[payload["line"] - 1]
                    if 1 <= payload["line"] <= len(lines)
                    else ""
                )
            payload["fix"] = _json_safe(payload.get("fix"))
        elif isinstance(item, DiagnosticRecord):
            rendered.append(
                _mapping_diagnostic(
                    item.to_dict(), text, selected_phase, selected_backend, code_map
                )
            )
            continue
        elif isinstance(item, Mapping):
            rendered.append(
                _mapping_diagnostic(
                    item, text, selected_phase, selected_backend, code_map
                )
            )
            continue
        else:
            raise TypeError(f"unsupported diagnostic input {type(item).__name__}")
        rendered.append(payload)
    payload_root: dict[str, Any] = {
        "schema": SCHEMA_DIAGNOSTICS,
        "count": len(rendered),
        "phase": selected_phase,
        "backend": selected_backend,
        "diagnostics": rendered,
    }
    if len(rendered) == 1:
        for key, value in rendered[0].items():
            if key not in {"schema", "count", "diagnostics"}:
                payload_root[key] = value
    if as_object:
        return payload_root
    return json.dumps(
        _json_safe(payload_root),
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def diag_aggregate_without_truncating_root_cause(
    diagnostics: Iterable[BaseException | Mapping[str, Any] | DiagnosticRecord],
    *,
    max_reported: int = 8,
    context: DiagnosticContext | None = None,
) -> AggregatedDiagnostics:
    """Aggregate cascading errors while always retaining the first root cause."""
    records = tuple(
        _coerce_record(
            item,
            context=context,
            phase=context.phase if context else None,
            backend=context.backend if context else None,
        )
        for item in diagnostics
    )
    if not records:
        return AggregatedDiagnostics(None, (), 0, 0, 0)
    root_index = next(
        (
            index
            for index, record in enumerate(records)
            if record.root_cause and not record.cascading
        ),
        0,
    )
    root = records[root_index]
    cascade_count = sum(
        1 for record in records if record.cascading or record.cause is not None
    )
    limit = max(1, _as_nonnegative_int(max_reported, default=8))
    retained = [root]
    for index, record in enumerate(records):
        if index == root_index or len(retained) >= limit:
            continue
        retained.append(record)
    return AggregatedDiagnostics(
        root_cause=root,
        diagnostics=tuple(retained),
        suppressed=len(records) - len(retained),
        cascade_count=cascade_count,
        total=len(records),
    )


def _path_prefixes(
    home: str | os.PathLike[str] | None,
    workspace: str | os.PathLike[str] | None,
) -> tuple[str, str]:
    home_text = "" if home is None else os.fspath(home).rstrip("/\\")
    workspace_text = "" if workspace is None else os.fspath(workspace).rstrip("/\\")
    return home_text, workspace_text


def diag_redact_sensitive_paths(
    value: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    home: str | os.PathLike[str] | None = None,
    workspace: str | os.PathLike[str] | None = None,
    placeholders: Mapping[str, str] | None = None,
) -> str | Mapping[str, Any] | Sequence[Any]:
    """Normalize and redact local user/workspace/tmp paths in reports or logs."""
    labels = {"home": "~", "workspace": "<workspace>", "tmp": "<tmp>"}
    labels.update({str(key): str(item) for key, item in (placeholders or {}).items()})
    home_text, workspace_text = _path_prefixes(
        Path.home() if home is None else home,
        Path.cwd() if workspace is None else workspace,
    )
    patterns: list[tuple[re.Pattern[str], str]] = []

    def add_prefix(prefix: str, replacement: str) -> None:
        if not prefix:
            return
        escaped = re.escape(prefix)
        patterns.append(
            (
                re.compile(
                    rf"(?<![A-Za-z0-9_])(?:{escaped})(?=[/\\]|$|[\s\"'])",
                    re.IGNORECASE,
                ),
                replacement,
            )
        )

    add_prefix(workspace_text, labels["workspace"])
    add_prefix(home_text, labels["home"])
    patterns.append(
        (
            re.compile(
                r"(?<![A-Za-z0-9_])(?:/Users/|/home/|[A-Za-z]:[/\\]?Users[/\\])"
                r"[^/\\\s\"']+",
                re.IGNORECASE,
            ),
            labels["home"],
        )
    )
    patterns.append(
        (
            re.compile(
                r"(?<![A-Za-z0-9_])(?:/private/var/folders|/var/folders|/tmp|"
                r"[A-Za-z]:\\?Temp)(?=[/\\]|$)",
                re.IGNORECASE,
            ),
            labels["tmp"],
        )
    )

    def redact_text(text: str) -> str:
        for pattern, replacement in patterns:
            text = pattern.sub(lambda _match, value=replacement: value, text)
        return text

    if isinstance(value, os.PathLike):
        return redact_text(os.fspath(value))
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            key: diag_redact_sensitive_paths(
                item,
                home=home_text or None,
                workspace=workspace_text or None,
                placeholders=labels,
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            diag_redact_sensitive_paths(
                item,
                home=home_text or None,
                workspace=workspace_text or None,
                placeholders=labels,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            diag_redact_sensitive_paths(
                item,
                home=home_text or None,
                workspace=workspace_text or None,
                placeholders=labels,
            )
            for item in value
        ]
    return value


def _elapsed_seconds(value: Any, *, unit: str, label: str) -> float:
    if isinstance(value, Mapping):
        if "seconds" in value:
            elapsed = _as_finite_float(value["seconds"], label=label)
        elif "ms" in value:
            elapsed = _as_finite_float(value["ms"], label=label) * 0.001
        elif "duration" in value:
            elapsed = _as_finite_float(value["duration"], label=label) * _unit_scale(
                unit, _UNIT_TO_SECONDS
            )
        else:
            raise ValueError(f"timing sample {value!r} has no duration field")
    elif callable(value):
        raise TypeError("a callable sample must be measured with clock or wrapper")
    else:
        elapsed = _as_finite_float(value, label=label) * _unit_scale(
            unit, _UNIT_TO_SECONDS
        )
    if elapsed < 0:
        raise ValueError(f"{label} must not be negative")
    return elapsed


def diag_record_compile_phase_timing[T](
    samples: (
        Mapping[str, float | Mapping[str, Any] | Callable[[], T]]
        | Sequence[tuple[str, float | Mapping[str, Any] | Callable[[], T]]]
        | None
    ) = None,
    *,
    clock: Callable[[], float] | None = None,
    wrapper: Callable[[str, Callable[[], T]], Any] | None = None,
    phases: Sequence[str] | None = None,
    unit: str = "seconds",
) -> dict[str, Any]:
    """Record per-pass compile durations from explicit samples or a wrapper."""
    _unit_scale(unit, _UNIT_TO_SECONDS)
    normalized: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    if isinstance(samples, Mapping):
        pairs: Iterable[tuple[str, Any]] = samples.items()
    elif samples is None:
        pairs = ()
    else:
        pairs = samples
    for raw_phase, sample in pairs:
        canonical = _canonical_phase(str(raw_phase), default=str(raw_phase))
        normalized[canonical] = sample
        aliases[canonical] = str(raw_phase)
    if phases:
        ordered = list(
            dict.fromkeys(
                _canonical_phase(str(item), default=str(item)) for item in phases
            )
        )
    else:
        ordered = list(CANONICAL_PHASES)
    for phase in sorted(normalized):
        if phase not in ordered:
            ordered.append(phase)
    phase_report: dict[str, Any] = {}
    measured: list[str] = []
    total_seconds = 0.0
    for phase in ordered:
        if phase not in normalized:
            phase_report[phase] = {
                "seconds": None,
                "ms": None,
                "observed": False,
                "alias": aliases.get(phase, phase),
            }
            continue
        sample = normalized[phase]
        if callable(sample):
            if wrapper is not None:
                outcome = wrapper(phase, sample)
                if isinstance(outcome, tuple) and len(outcome) == 2:
                    outcome = outcome[1]
                elapsed = _elapsed_seconds(outcome, unit=unit, label=f"{phase} timing")
            else:
                timer = clock or time.perf_counter
                started = timer()
                sample()
                elapsed = _as_finite_float(timer() - started, label=f"{phase} timing")
                if elapsed < 0:
                    raise ValueError(f"{phase} clock moved backwards")
            observed = True
        else:
            elapsed = _elapsed_seconds(sample, unit=unit, label=f"{phase} timing")
            observed = False
        total_seconds += elapsed
        measured.append(phase)
        phase_report[phase] = {
            "seconds": round(elapsed, 9),
            "ms": round(elapsed * 1000.0, 3),
            "observed": observed,
            "alias": aliases.get(phase, phase),
        }
    critical = max(
        measured, key=lambda name: phase_report[name]["seconds"], default=None
    )
    return {
        "schema": SCHEMA_TIMING,
        "unit": "seconds",
        "phases": phase_report,
        "phase_ms": {name: entry["ms"] for name, entry in phase_report.items()},
        "total_seconds": round(total_seconds, 9),
        "total_ms": round(total_seconds * 1000.0, 3),
        "measured_phases": measured,
        "critical_phase": critical,
    }


def _count(value: Any, *, label: str) -> int:
    return (
        _as_nonnegative_int(value, default=0) if label else _as_nonnegative_int(value)
    )


def diag_record_cache_hit_metrics(
    samples: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    backend: str | Sequence[str] = "python",
    hits: int | None = None,
    misses: int | None = None,
    rebuild_reasons: Mapping[str, int] | Sequence[str] | None = None,
    wrapper: Callable[[str], Any] | None = None,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Track Python/Rust cache hits, misses, and rebuild reasons deterministically."""
    if isinstance(backend, str):
        backend_names = [_canonical_backend(backend)]
    else:
        backend_names = list(
            dict.fromkeys(_canonical_backend(item) for item in backend)
        )
    default_backend = backend_names[0]
    counters: dict[str, dict[str, Any]] = {
        name: {"hits": 0, "misses": 0, "reasons": {}} for name in backend_names
    }
    observations = 0

    def ensure(name: str) -> dict[str, Any]:
        canonical = _canonical_backend(name)
        return counters.setdefault(canonical, {"hits": 0, "misses": 0, "reasons": {}})

    def note(backend_name: str, is_hit: bool, reason: str | None = None) -> None:
        nonlocal observations
        bucket = ensure(backend_name)
        observations += 1
        if is_hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1
            if reason:
                bucket["reasons"][str(reason)] = (
                    bucket["reasons"].get(str(reason), 0) + 1
                )

    def add_reasons(source: Mapping[str, int] | Sequence[str] | None) -> None:
        if source is None:
            return
        if isinstance(source, Mapping):
            pairs: Iterable[tuple[str, Any]] = source.items()
        else:
            pairs = ((str(item), 1) for item in source)
        for reason, count in pairs:
            value = _count(count, label=reason)
            bucket = ensure(default_backend)
            bucket["reasons"][str(reason)] = (
                bucket["reasons"].get(str(reason), 0) + value
            )

    def observe_value(value: Any) -> None:
        if isinstance(value, bool):
            note(default_backend, value)
        elif isinstance(value, Mapping):
            name = str(value.get("backend") or default_backend)
            raw_hit = value.get("hit", value.get("result", value.get("cache_hit")))
            is_hit = (
                bool(raw_hit)
                if isinstance(raw_hit, bool)
                else str(raw_hit).lower() in {"hit", "true", "1", "cached"}
            )
            note(name, is_hit, value.get("reason") or value.get("rebuild_reason"))
        elif isinstance(value, str):
            lowered = value.lower()
            if lowered in {"hit", "cached"}:
                note(default_backend, True)
            elif lowered in {"miss", "cold"}:
                note(default_backend, False)
            else:
                note(default_backend, False, value)
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            first, second = value[0], value[1]
            if isinstance(first, str) and first.lower() in {
                "hit",
                "miss",
                "cached",
                "cold",
            }:
                note(default_backend, first.lower() in {"hit", "cached"}, str(second))
            else:
                name = str(first)
                is_hit = (
                    bool(second)
                    if isinstance(second, bool)
                    else str(second).lower() in {"hit", "true", "1", "cached"}
                )
                reason = str(value[2]) if len(value) > 2 else None
                note(name, is_hit, reason)
        else:
            raise TypeError(f"unsupported cache observation {value!r}")

    if hits is not None:
        ensure(default_backend)["hits"] += _count(hits, label="hits")
    if misses is not None:
        ensure(default_backend)["misses"] += _count(misses, label="misses")
    add_reasons(rebuild_reasons)
    if isinstance(samples, Mapping):
        reserved = {"hits", "misses", "rebuild_reasons", "backend", "backends"}
        for key, value in samples.items():
            if key == "hits":
                ensure(default_backend)["hits"] += _count(value, label="hits")
            elif key == "misses":
                ensure(default_backend)["misses"] += _count(value, label="misses")
            elif key == "rebuild_reasons":
                add_reasons(value)
            elif key in {"backend", "backends"}:
                for name in [value] if isinstance(value, str) else value:
                    ensure(str(name))
            elif isinstance(value, Mapping):
                bucket = ensure(str(key))
                if "hits" in value:
                    bucket["hits"] += _count(value["hits"], label="hits")
                if "misses" in value:
                    bucket["misses"] += _count(value["misses"], label="misses")
                add_reasons(value.get("rebuild_reasons"))
                if "hits" not in value and "misses" not in value:
                    observe_value(value)
            elif str(key) in reserved:
                continue
            else:
                observe_value(value)
    elif samples is not None:
        for item in samples:
            observe_value(item)
    if wrapper is not None:
        for key in keys or ():
            observe_value(wrapper(key))
    summaries: dict[str, Any] = {}
    total_hits = total_misses = 0
    merged_reasons: dict[str, int] = {}
    for name in sorted(counters):
        bucket = counters[name]
        count = bucket["hits"] + bucket["misses"]
        total_hits += bucket["hits"]
        total_misses += bucket["misses"]
        reasons = dict(sorted(bucket["reasons"].items()))
        for reason, value in reasons.items():
            merged_reasons[reason] = merged_reasons.get(reason, 0) + value
        summaries[name] = {
            "hits": bucket["hits"],
            "misses": bucket["misses"],
            "total": count,
            "hit_rate": round(bucket["hits"] / count, 4) if count else 0.0,
            "rebuild_reasons": reasons,
            "rebuild_total": sum(reasons.values()),
        }
    overall = total_hits + total_misses
    return {
        "schema": SCHEMA_CACHE,
        "backends": summaries,
        "hits": total_hits,
        "misses": total_misses,
        "total": overall,
        "hit_rate": round(total_hits / overall, 4) if overall else 0.0,
        "rebuild_reasons": dict(sorted(merged_reasons.items())),
        "rebuild_total": sum(merged_reasons.values()),
        "observations": observations,
    }


def _duration_to_ms(value: Any, unit: str) -> float:
    if isinstance(value, Mapping):
        for key in ("duration_ms", "ms", "milliseconds"):
            if key in value:
                milliseconds = _as_finite_float(value[key], label=key)
                break
        else:
            for key in ("duration_seconds", "seconds", "s"):
                if key in value:
                    milliseconds = _as_finite_float(value[key], label=key) * 1000.0
                    break
            else:
                raise ValueError(f"benchmark record {value!r} has no duration field")
    else:
        milliseconds = _as_finite_float(value, label="duration") * _unit_scale(
            unit, _UNIT_TO_MS
        )
    if milliseconds < 0:
        raise ValueError("benchmark duration must not be negative")
    return milliseconds


def _measurements_from_mapping(
    results: Mapping[str, Any], unit: str, workload: str
) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    nested = results.get("results")
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
        for item in nested:
            measurements.append(_measurement_from_record(item, unit, workload))
        return measurements
    for backend_name, levels in results.items():
        if backend_name in {"schema", "baseline", "workload", "results"}:
            continue
        canonical = _canonical_backend(str(backend_name))
        if isinstance(levels, Mapping):
            for level, value in levels.items():
                duration = _duration_to_ms(value, unit)
                measurements.append(
                    {
                        "backend": canonical,
                        "opt_level": _canonical_opt_level(str(level)),
                        "duration_ms": round(duration, 6),
                        "workload": str(value.get("workload", workload))
                        if isinstance(value, Mapping)
                        else workload,
                        "source": "explicit",
                    }
                )
        else:
            measurements.append(
                {
                    "backend": canonical,
                    "opt_level": "O0",
                    "duration_ms": round(_duration_to_ms(levels, unit), 6),
                    "workload": workload,
                    "source": "explicit",
                }
            )
    return measurements


def _measurement_from_record(record: Any, unit: str, workload: str) -> dict[str, Any]:
    if isinstance(record, Mapping):
        duration_value = record.get("duration_ms", record.get("duration", record))
        return {
            "backend": _canonical_backend(str(record.get("backend") or "python")),
            "opt_level": _canonical_opt_level(
                str(record.get("opt_level") or record.get("opt") or record.get("level"))
            ),
            "duration_ms": round(_duration_to_ms(duration_value, unit), 6),
            "workload": str(record.get("workload") or workload),
            "source": str(record.get("source") or "explicit"),
        }
    if isinstance(record, (tuple, list)) and len(record) >= 3:
        backend, level, duration = record[0], record[1], record[2]
    elif isinstance(record, (tuple, list)) and len(record) == 2:
        backend, level, duration = record[0], "O0", record[1]
    else:
        raise TypeError(f"unsupported benchmark record {record!r}")
    return {
        "backend": _canonical_backend(str(backend)),
        "opt_level": _canonical_opt_level(str(level)),
        "duration_ms": round(_duration_to_ms(duration, unit), 6),
        "workload": workload,
        "source": "explicit",
    }


def _parse_baseline(baseline: str) -> tuple[str, str | None]:
    text = baseline.strip()
    match = re.split(r"[:@/]", text, maxsplit=1)
    backend = _canonical_backend(match[0])
    level = _canonical_opt_level(match[1]) if len(match) == 2 else None
    return backend, level


def diag_export_benchmark_summary(
    results: Mapping[str, Any] | Sequence[Any],
    *,
    baseline: str = "python",
    unit: str = "ms",
    opt_levels: Sequence[str] | None = None,
    wrapper: Callable[[str, str], Any] | None = None,
    workload: str = "default",
    precision: int = 3,
) -> dict[str, Any]:
    """Export deterministic backend/optimization-level benchmark comparisons."""
    _unit_scale(unit, _UNIT_TO_MS)
    if precision < 0:
        raise ValueError("precision must be non-negative")
    if isinstance(results, Mapping):
        measurements = _measurements_from_mapping(results, unit, workload)
        combo_keys = {
            (name, _canonical_opt_level(str(level)))
            for name, levels in results.items()
            if name not in {"schema", "baseline", "workload", "results"}
            and isinstance(levels, Mapping)
            for level in levels
        }
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        measurements = [
            _measurement_from_record(item, unit, workload) for item in results
        ]
        combo_keys = {(item["backend"], item["opt_level"]) for item in measurements}
    else:
        raise TypeError("benchmark results must be a mapping or a sequence of records")
    if wrapper is not None:
        if combo_keys:
            combos = sorted(combo_keys)
        else:
            backends = sorted({item["backend"] for item in measurements}) or [
                "python",
                "rust",
            ]
            levels = [
                _canonical_opt_level(str(item)) for item in (opt_levels or ("O0", "O2"))
            ]
            combos = [(name, level) for name in backends for level in levels]
        for name, level in combos:
            measurements.append(
                {
                    "backend": name,
                    "opt_level": level,
                    "duration_ms": round(
                        _duration_to_ms(wrapper(name, level), unit), 6
                    ),
                    "workload": workload,
                    "source": "wrapper",
                }
            )
    if not measurements:
        raise ValueError("benchmark summary requires at least one measurement")
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in measurements:
        groups.setdefault(item["workload"], []).append(item)
    rank = {"O0": 0, "O1": 1, "O2": 2, "O3": 3, "O4": 4}
    baseline_backend, baseline_level = _parse_baseline(baseline)
    summaries: dict[str, Any] = {}
    fallback_baselines: list[str] = []
    for name in sorted(groups):
        records = sorted(
            groups[name],
            key=lambda item: (
                item["backend"],
                rank.get(item["opt_level"], 99),
                item["opt_level"],
            ),
        )
        if baseline_level is not None:
            candidates = [
                item
                for item in records
                if item["backend"] == baseline_backend
                and item["opt_level"] == baseline_level
            ]
        else:
            candidates = [
                item for item in records if item["backend"] == baseline_backend
            ]
        if candidates:
            reference = (
                candidates[0]
                if baseline_level
                else min(
                    candidates,
                    key=lambda item: (
                        rank.get(item["opt_level"], 99),
                        item["opt_level"],
                    ),
                )
            )
        else:
            reference = min(records, key=lambda item: item["duration_ms"])
            fallback_baselines.append(name)
        baseline_ms = reference["duration_ms"]
        comparisons = []
        for item in records:
            if item is reference:
                continue
            duration = item["duration_ms"]
            speedup = round(baseline_ms / duration, precision) if duration > 0 else None
            delta = round(duration - baseline_ms, precision)
            percent = (
                round((duration - baseline_ms) / baseline_ms * 100.0, precision)
                if baseline_ms > 0
                else None
            )
            comparisons.append(
                {
                    "backend": item["backend"],
                    "opt_level": item["opt_level"],
                    "duration_ms": duration,
                    "speedup": speedup,
                    "delta_ms": delta,
                    "percent_change": percent,
                }
            )
        fastest = min(records, key=lambda item: item["duration_ms"])
        slowest = max(records, key=lambda item: item["duration_ms"])
        summaries[name] = {
            "measurements": records,
            "baseline": dict(reference),
            "baseline_ms": baseline_ms,
            "fastest": dict(fastest),
            "slowest": dict(slowest),
            "comparisons": comparisons,
        }
    all_records = [item for group in groups.values() for item in group]
    return {
        "schema": SCHEMA_BENCHMARK,
        "baseline": {"backend": baseline_backend, "opt_level": baseline_level},
        "baseline_fallback_workloads": sorted(fallback_baselines),
        "workloads": summaries,
        "backends": sorted({item["backend"] for item in all_records}),
        "opt_levels": sorted(
            {item["opt_level"] for item in all_records},
            key=lambda item: (rank.get(item, 99), item),
        ),
        "wrapper_used": wrapper is not None,
    }


__all__ = [
    "CANONICAL_PHASES",
    "SEVERITIES",
    "AggregatedDiagnostics",
    "DiagnosticContext",
    "DiagnosticRecord",
    "diag_aggregate_without_truncating_root_cause",
    "diag_create_diagnostic_context",
    "diag_create_pillar_error",
    "diag_export_benchmark_summary",
    "diag_format_json_payload",
    "diag_format_terminal_snippet",
    "diag_map_error_code_e001_to_e008",
    "diag_record_cache_hit_metrics",
    "diag_record_compile_phase_timing",
    "diag_redact_sensitive_paths",
]
