"""FRAME 5 — Rust / PyO3 native backend framework functions.

This module turns compiler metadata into deterministic PyO3 scaffolding, maps
Python/Rust boundary values and errors, emits RAII/GIL-aware Rust, and runs
isolated Cargo checks.  It deliberately uses only the standard library plus
the project's existing Rust backend helpers; every generated string is stable
for a given set of inputs so checked-in scaffolds can be diffed for drift.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NoReturn

from compiler.core.ast_nodes import Program
from backends.rust.codegen import RustCodegen
from runtime.diagnostics import ERROR_CODES, PillarError

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RsFieldSpec:
    """A struct field in source order."""

    name: str
    type: str
    visibility: str = "pub"
    python_access: bool = True


@dataclass(frozen=True, slots=True)
class RsStructSpec:
    """A Rust struct (optionally generic) to emit."""

    name: str
    fields: tuple[RsFieldSpec, ...] = ()
    derives: tuple[str, ...] = ("Clone", "Debug")
    generic_params: tuple[str, ...] = ()
    visibility: str = "pub"
    pyclass: bool = True
    substitutions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RsVariantSpec:
    """A tuple-style enum variant."""

    name: str
    field_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RsEnumSpec:
    """A Rust enum to emit."""

    name: str
    variants: tuple[RsVariantSpec, ...] = ()
    derives: tuple[str, ...] = ("Clone", "Debug", "PartialEq", "Eq")
    generic_params: tuple[str, ...] = ()
    visibility: str = "pub"
    pyclass: bool = False
    substitutions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RsParamSpec:
    """A function parameter with its Rust type."""

    name: str
    type: str
    required: bool = True
    default: str | None = None


@dataclass(frozen=True, slots=True)
class RsFunctionSpec:
    """Metadata for one exported PyO3 function."""

    name: str
    params: tuple[RsParamSpec, ...] = ()
    result: str | None = None
    body: str = ""
    doc: str = ""
    generic_params: tuple[str, ...] = ()
    where_bounds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RsTraitImplSpec:
    """An inherent or trait impl block with optional where bounds."""

    target: str
    trait: str | None = None
    methods: tuple[RsFunctionSpec, ...] = ()
    bounds: tuple[str, ...] = ()
    generic_params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RsArgMapping:
    """Result of mapping Python call arguments at the Rust boundary."""

    code: str
    values: dict[str, Any]
    rust_types: dict[str, str]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RsResultMapping:
    """Result of mapping a Rust Result<T, E> into a Python return value."""

    code: str
    ok_type: str
    error_type: str | None
    python_exception: str
    rust_exception: str


@dataclass(frozen=True, slots=True)
class RsGilPlan:
    """GIL-minimizing wrapper code plus the rules it obeys."""

    code: str
    rules: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RsGuardSpec:
    """One RAII guard to instantiate from an acquire/release expression."""

    name: str
    kind: str
    acquire: str
    release: str | None = None


@dataclass(frozen=True, slots=True)
class ScaffoldVerification:
    """Structured drift report for checked-in scaffold files."""

    ok: bool
    matches: tuple[str, ...]
    mismatched: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    details: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CargoStepResult:
    """Structured output of one Cargo subcommand."""

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    diagnostics: tuple[str, ...]
    target_dir: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class CargoCheckResult:
    """Structured outcome of ``cargo check`` and ``cargo clippy``."""

    ok: bool
    check: CargoStepResult
    clippy: CargoStepResult | None
    target_dir: str


CargoRunner = Callable[
    [tuple[str, ...], Path, dict[str, str]], subprocess.CompletedProcess[str]
]


class RsPillarError(PillarError):
    """Typed Python exception for Rust/PyO3 pillar failures (E001-E008)."""

    rust_exception_name: str = "PyPillarError"

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
    ) -> None:
        super().__init__(
            code,
            message,
            reason=reason,
            help_text=help_text,
            note=note or "RsPillarError (Rust/PyO3 backend)",
            line=line,
            col=col,
            span=span,
        )

    @property
    def message(self) -> str:
        """The diagnostic's short message (mirrors Rust ``PillarError``)."""
        return self.diagnostic.message


# ---------------------------------------------------------------------------
# Validation and small deterministic helpers
# ---------------------------------------------------------------------------

_RUST_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "unsafe",
        "use",
        "where",
        "while",
    }
)
_VISIBILITIES: Final[frozenset[str]] = frozenset(
    {"", "pub", "pub(crate)", "pub(super)", "pub(self)"}
)
_INT_RANGES: Final[Mapping[str, tuple[int, int]]] = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "isize": (-(2**63), 2**63 - 1),
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "usize": (0, 2**64 - 1),
}
_TYPE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _fail(message: str, *, code: str = "E003", **kwargs: Any) -> NoReturn:
    raise RsPillarError(code, message, **kwargs)


def _identifier(value: str, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        _fail(f"{what} must be a valid Rust identifier, got {value!r}")
    if value in _RUST_KEYWORDS and value not in {"self", "Self"}:
        _fail(f"{what} uses reserved Rust keyword {value!r}")
    return value


def _type_path(value: str, what: str) -> str:
    if not isinstance(value, str) or _TYPE_PATH_RE.fullmatch(value) is None:
        _fail(f"{what} must be a Rust type path, got {value!r}")
    return value


def _visibility(value: str, what: str) -> str:
    if value not in _VISIBILITIES:
        _fail(f"{what} must be one of {sorted(_VISIBILITIES)}, got {value!r}")
    return value


def _type_text(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{what} must be a non-empty Rust type")
    text = " ".join(value.split())
    if ";" in text or "\n" in text:
        _fail(f"{what} must be a type expression, got {value!r}")
    if text.count("<") != text.count(">"):
        _fail(f"{what} has unbalanced generic brackets: {value!r}")
    return text


def _string_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _balanced(text: str, *, what: str) -> str:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            _fail(f"{what} has unbalanced delimiters")
    if stack:
        _fail(f"{what} has unbalanced delimiters")
    return text


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    pieces: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">])" and depth:
            depth -= 1
        elif char == separator and depth == 0:
            piece = text[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _named_type_parts(text: str) -> tuple[str, str]:
    """Split ``Name<A, B>`` into ``(Name, "A, B")`` without touching tuples/refs."""
    for index, char in enumerate(text):
        if char != "<":
            continue
        name = text[:index].strip()
        if _TYPE_PATH_RE.fullmatch(name) is None:
            continue
        depth = 1
        for end in range(index + 1, len(text)):
            if text[end] == "<":
                depth += 1
            elif text[end] == ">":
                depth -= 1
                if depth == 0:
                    return name, text[index + 1 : end].strip()
        _fail(f"unbalanced generics in {text!r}")
    return text.strip(), ""


def _canonical_type(text: str) -> str:
    """Canonicalize a type string using the real Rust code generator."""
    text = _type_text(text, "Rust type")
    name, args_text = _named_type_parts(text)
    if name == "Result" and args_text:
        args = _split_top_level(args_text)
        if len(args) == 2:
            return f"Result<{_canonical_type(args[0])}, {_canonical_type(args[1])}>"
    args = [_canonical_type(arg) for arg in _split_top_level(args_text)]
    if args:
        return f"{name}<{', '.join(args)}>"
    codegen = RustCodegen(module_name="polyglot_f5_types")
    from compiler.core.ast_nodes import TypeName

    return codegen.type_to_rust(TypeName(1, 1, name))


def _substitute_type(text: str, substitutions: Mapping[str, str]) -> str:
    """Replace generic type parameters inside a (possibly nested) type."""
    result = _canonical_type(text)
    for _ in range(4):
        changed = False
        tokens: list[str] = []
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^A-Za-z0-9_]+", result):
            replacement = substitutions.get(token)
            if replacement is not None and token not in substitutions.values():
                tokens.append(f" {replacement} ")
                changed = True
            else:
                tokens.append(token)
        result = " ".join("".join(tokens).split())
        if not changed:
            break
    return _canonical_type(result)


def _type_suffix(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    suffix = "".join(token[:1].upper() + token[1:] for token in tokens)
    if not suffix:
        suffix = hashlib.sha256(text.encode()).hexdigest()[:8]
    return suffix or "T"


def _dedupe_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:6]
    candidate = f"{name}_{digest}"
    while candidate in used:
        candidate += "_"
    used.add(candidate)
    return candidate


def _bound_text(bound: str, what: str) -> str:
    text = " ".join(str(bound).split())
    if not text or any(char in text for char in "{};"):
        _fail(f"{what} must be a where-bound such as 'T: Clone + Send', got {bound!r}")
    if ":" not in text:
        _fail(f"{what} must contain ':' between type and trait bounds: {bound!r}")
    head, tail = text.split(":", 1)
    _type_path(head.strip(), f"{what} type")
    if not tail.strip():
        _fail(f"{what} has no trait bounds after ':'")
    return f"{head.strip()}: {tail.strip()}"


def _param_decl(param: RsParamSpec) -> str:
    name = _identifier(param.name, "parameter name")
    if name == "self":
        return param.type if param.type in {"self", "&self", "&mut self"} else "self"
    ty = _canonical_type(param.type)
    if ty in {"self", "&self", "&mut self"}:
        return ty
    if not param.required and param.default is not None:
        return f"{name}: {ty}"
    return f"mut {name}: {ty}"


def _method_signature(method: RsFunctionSpec) -> str:
    name = _identifier(method.name, "method name")
    params = ", ".join(_param_decl(param) for param in method.params)
    result = f" -> {_canonical_type(method.result)}" if method.result else ""
    return f"fn {name}({params}){result}"


def _render_method(method: RsFunctionSpec, indent: str = "    ") -> list[str]:
    _balanced(method.body, what=f"body of {method.name!r}")
    if not method.body.strip():
        _fail(f"method {method.name!r} needs a Rust body")
    lines = [
        *(
            f"{indent}/// {line.strip()}"
            for line in method.doc.splitlines()
            if line.strip()
        ),
        f"{indent}{_method_signature(method)} {{",
    ]
    lines.extend(f"{indent}    {line}" for line in method.body.splitlines())
    lines.append(f"{indent}}}")
    return lines


def _coerce_spec(value: Any, kind: str) -> Any:
    """Coerce a mapping/dataclass into one of the typed spec dataclasses."""
    factories: Final[Mapping[str, Any]] = {
        "field": (RsFieldSpec, {"type": "type", "rust_type": "type"}),
        "struct": (RsStructSpec, {}),
        "variant": (RsVariantSpec, {"fields": "field_types"}),
        "enum": (RsEnumSpec, {}),
        "param": (RsParamSpec, {"rust_type": "type"}),
        "function": (RsFunctionSpec, {"rust_type": "result", "returns": "result"}),
        "impl": (RsTraitImplSpec, {"where": "bounds"}),
        "guard": (RsGuardSpec, {"expression": "acquire"}),
    }
    try:
        factory, aliases = factories[kind]
    except KeyError as error:  # pragma: no cover - internal guard
        raise AssertionError(f"unknown spec kind {kind!r}") from error
    if isinstance(value, factory):
        return value
    if not isinstance(value, Mapping):
        _fail(
            f"{kind} spec must be a {factory.__name__} or mapping, got {type(value).__name__}"
        )
    payload = {aliases.get(str(key), str(key)): item for key, item in value.items()}
    try:
        return factory(**payload)
    except TypeError as error:
        _fail(f"invalid {kind} spec {dict(value)!r}: {error}")


def _spec_list(values: Sequence[Any], kind: str) -> tuple[Any, ...]:
    return tuple(_coerce_spec(value, kind) for value in values)


def _merge_by_name(*groups: Sequence[Any]) -> tuple[Any, ...]:
    merged: dict[str, Any] = {}
    for group in groups:
        for spec in group:
            merged[spec.name] = spec
    return tuple(merged[name] for name in sorted(merged))


def _dedent_rust(inner: str) -> str:
    lines = inner.strip("\n").splitlines()
    padding = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    return "\n".join(
        line[padding:] if len(line) >= padding else line for line in lines
    ).strip()


def _extract_fn_body(fragment: str, name: str) -> str:
    start = fragment.find("{")
    end = fragment.rfind("}")
    if start < 0 or end <= start:
        _fail(f"code generator produced no body for function {name!r}")
    return _dedent_rust(fragment[start + 1 : end])


def _method_body_from_impl(fragments: Sequence[str], name: str) -> str | None:
    """Find a method body inside the impl-level fragments from RustCodegen."""
    pattern = re.compile(rf"\bfn\s+{re.escape(name)}\s*\(")
    for fragment in fragments:
        match = pattern.search(fragment)
        if match is None:
            continue
        start = fragment.find("{", match.end())
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(fragment)):
            if fragment[index] == "{":
                depth += 1
            elif fragment[index] == "}":
                depth -= 1
                if depth == 0:
                    return _dedent_rust(fragment[start + 1 : index])
    return None


def _ast_param_spec(param: Any, index: int, codegen: RustCodegen) -> RsParamSpec:
    """Convert an AST parameter into a scaffold parameter, preserving receivers."""
    from compiler.core.ast_nodes import PatIdent, PatRef

    pattern = getattr(param, "pattern", None)
    name = f"arg{index}"
    if isinstance(pattern, PatIdent):
        name = pattern.name
    elif isinstance(pattern, PatRef) and isinstance(pattern.inner, PatIdent):
        name = pattern.inner.name
    if name == "self":
        return RsParamSpec(
            name="self", type=codegen.param_to_rust(param, as_method=True)
        )
    ty = codegen.type_to_rust(param.ty) if getattr(param, "ty", None) else "()"
    return RsParamSpec(name=name, type=_canonical_type(ty))


def _specs_from_program(
    program: Program, *, module_name: str
) -> tuple[
    tuple[RsFunctionSpec, ...],
    tuple[RsStructSpec, ...],
    tuple[RsEnumSpec, ...],
    tuple[RsTraitImplSpec, ...],
]:
    """Derive scaffold metadata from an AST using the real Rust codegen."""
    from compiler.core.ast_nodes import EnumDecl, FnDecl, ImplBlock, StructDecl
    from compiler.analysis.monomorph import monomorphize

    codegen = RustCodegen(module_name=module_name)
    codegen.generate(program)
    functions: list[RsFunctionSpec] = []
    structs: list[RsStructSpec] = []
    enums: list[RsEnumSpec] = []
    impls: list[RsTraitImplSpec] = []

    for item in monomorphize(program).items:
        if isinstance(item, FnDecl):
            params = tuple(
                _ast_param_spec(param, index, codegen)
                for index, param in enumerate(item.params)
            )
            result = None
            if item.ret is not None:
                result = _canonical_type(codegen.type_to_rust(item.ret))
            fragment = codegen.fragments.get(f"fn:{item.name}", "")
            functions.append(
                RsFunctionSpec(
                    name=item.name,
                    params=params,
                    result=result,
                    body=_extract_fn_body(fragment, item.name),
                    generic_params=tuple(
                        str(getattr(generic, "name", generic))
                        for generic in item.generics
                    ),
                )
            )
        elif isinstance(item, StructDecl):
            structs.append(
                RsStructSpec(
                    name=item.name,
                    fields=tuple(
                        RsFieldSpec(
                            name=name,
                            type=_canonical_type(codegen.type_to_rust(ty)),
                            visibility="pub" if is_pub else "",
                        )
                        for name, ty, is_pub in item.fields
                    ),
                    generic_params=tuple(
                        str(getattr(generic, "name", generic))
                        for generic in item.generics
                    ),
                    visibility="pub" if item.is_pub else "",
                )
            )
        elif isinstance(item, EnumDecl):
            enums.append(
                RsEnumSpec(
                    name=item.name,
                    variants=tuple(
                        RsVariantSpec(
                            name=variant.name,
                            field_types=tuple(
                                _canonical_type(codegen.type_to_rust(ty))
                                for ty in (variant.fields or ())
                            ),
                        )
                        for variant in item.variants
                    ),
                    generic_params=tuple(
                        str(getattr(generic, "name", generic))
                        for generic in item.generics
                    ),
                    visibility="pub" if item.is_pub else "",
                    pyclass=all(variant.fields is None for variant in item.variants),
                )
            )
        elif isinstance(item, ImplBlock):
            target = _canonical_type(codegen.type_to_rust(item.ty))
            impl_fragments = [
                text
                for key, text in codegen.fragments.items()
                if key == f"impl:{target}" or key.startswith(f"impl:{target}#")
            ]
            methods: list[RsFunctionSpec] = []
            for method in item.items:
                if not isinstance(method, FnDecl):
                    continue
                own_fragment = codegen.fragments.get(f"fn:{method.name}", "")
                body = (
                    _extract_fn_body(own_fragment, method.name)
                    if own_fragment
                    else (_method_body_from_impl(impl_fragments, method.name) or "")
                )
                if not body:
                    continue
                methods.append(
                    RsFunctionSpec(
                        name=method.name,
                        params=tuple(
                            _ast_param_spec(param, index, codegen)
                            for index, param in enumerate(method.params)
                        ),
                        result=(
                            _canonical_type(codegen.type_to_rust(method.ret))
                            if method.ret
                            else None
                        ),
                        body=body,
                    )
                )
            if methods:
                impls.append(
                    RsTraitImplSpec(
                        target=target,
                        trait=(
                            _canonical_type(codegen.type_to_rust(item.trait))
                            if item.trait
                            else None
                        ),
                        methods=tuple(methods),
                    )
                )
    return tuple(functions), tuple(structs), tuple(enums), tuple(impls)


def _specs_from_native_tests(
    tests_dir: str | Path, *, module_name: str
) -> tuple[
    tuple[RsFunctionSpec, ...],
    tuple[RsStructSpec, ...],
    tuple[RsEnumSpec, ...],
    tuple[RsTraitImplSpec, ...],
]:
    """Collect native SOURCES via the shared helper, without invoking Cargo."""
    from _native_sources import collect_sources
    from compiler.core.lexer import lex
    from compiler.core.parser import parse
    from tools.compile_native import ROOT as COMPILE_NATIVE_ROOT

    tests_path = Path(tests_dir)
    if not tests_path.is_absolute():
        tests_path = COMPILE_NATIVE_ROOT / tests_path
    sources = collect_sources(tests_path)
    functions: list[RsFunctionSpec] = []
    structs: list[RsStructSpec] = []
    enums: list[RsEnumSpec] = []
    impls: list[RsTraitImplSpec] = []
    for name, source in sorted(sources.items()):
        program = parse(lex(source))
        found = _specs_from_program(program, module_name=module_name)
        if name != "__preamble__":
            functions.extend(found[0])
        structs.extend(found[1])
        enums.extend(found[2])
        impls.extend(found[3])
    return tuple(functions), tuple(structs), tuple(enums), tuple(impls)


def _default_pyo3_version() -> str:
    """Read the PyO3 version from the checked-in native_pillars manifest."""
    manifest = Path(__file__).resolve().parents[2] / "native_pillars" / "Cargo.toml"
    try:
        dependencies = tomllib.loads(manifest.read_text(encoding="utf-8")).get(
            "dependencies", {}
        )
    except OSError, tomllib.TOMLDecodeError:
        return "0.26"
    dependency = dependencies.get("pyo3", "0.26")
    if isinstance(dependency, Mapping):
        return str(dependency.get("version", "0.26"))
    return str(dependency)


def _emit_typed_error_bridge(module_name: str) -> str:
    """Emit Rust code for the typed PillarError -> Python exception bridge."""
    _identifier(module_name, "module name")
    return "\n".join(
        [
            "// Typed Pillar diagnostics: never leak a bare PyRuntimeError.",
            "#[derive(Debug, Clone)]",
            "pub struct PillarError {",
            "    pub code: String,",
            "    pub message: String,",
            "    pub reason: String,",
            "    pub help_text: String,",
            "    pub note: String,",
            "    pub line: i64,",
            "    pub col: i64,",
            "}",
            "",
            "impl PillarError {",
            "    pub fn new(code: &str, message: impl Into<String>) -> Self {",
            "        let message = message.into();",
            "        Self {",
            "            code: code.to_string(),",
            "            reason: message.clone(),",
            '            help_text: "fix the reported error".to_string(),',
            '            note: "PillarError".to_string(),',
            "            line: 0,",
            "            col: 0,",
            "            message,",
            "        }",
            "    }",
            "",
            "    pub fn format_pillar(&self) -> String {",
            "        format!(",
            '            "error[{}]: {}\\n  reason: {}\\n  help: {}\\n  note: {}\\n  at: line {}, column {}",',
            "            self.code,",
            "            self.message,",
            "            self.reason,",
            "            self.help_text,",
            "            self.note,",
            "            self.line,",
            "            self.col,",
            "        )",
            "    }",
            "}",
            "",
            "impl std::fmt::Display for PillarError {",
            "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {",
            "        f.write_str(&self.format_pillar())",
            "    }",
            "}",
            "",
            "impl std::error::Error for PillarError {}",
            "",
            "pyo3::create_exception!(",
            f"    {module_name},",
            "    PyPillarError,",
            "    pyo3::exceptions::PyException,",
            '    "Structured Pillar diagnostic (E001-E008)."',
            ");",
            "",
            "impl From<PillarError> for pyo3::PyErr {",
            "    fn from(error: PillarError) -> Self {",
            "        PyPillarError::new_err(error.format_pillar())",
            "    }",
            "}",
        ]
    )


# ---------------------------------------------------------------------------
# 41 / 42 — scaffold generation and drift detection
# ---------------------------------------------------------------------------


def rs_generate_pyo3_scaffold(
    *,
    module_name: str = "polyglot_native",
    functions: Sequence[RsFunctionSpec | Mapping[str, Any]] = (),
    structs: Sequence[RsStructSpec | Mapping[str, Any]] = (),
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
    impls: Sequence[RsTraitImplSpec | Mapping[str, Any]] = (),
    substitutions: Mapping[str, str] | None = None,
    program: Program | None = None,
    native_tests_dir: str | Path | None = None,
    pyo3_version: str | None = None,
    edition: str = "2021",
) -> dict[str, str]:
    """Generate deterministic PyO3 scaffold files from compiler metadata.

    Metadata may be supplied directly, derived from a parsed ``Program`` via
    ``RustCodegen``, or collected from native test SOURCES (the same helper
    used by ``tools/compile_native.py``).  The returned mapping is keyed by
    relative POSIX path so it can be written to disk or diffed as-is.
    """
    _identifier(module_name, "module name")
    if edition not in {"2018", "2021", "2024"}:
        _fail(f"unsupported Rust edition {edition!r}")

    functions_out = _spec_list(functions, "function")
    structs_out = _spec_list(structs, "struct")
    enums_out = _spec_list(enums, "enum")
    impls_out = _spec_list(impls, "impl")

    if program is not None:
        derived = _specs_from_program(program, module_name=module_name)
        functions_out = _merge_by_name(derived[0], functions_out)  # type: ignore[arg-type]
        structs_out = _merge_by_name(derived[1], structs_out)  # type: ignore[arg-type]
        enums_out = _merge_by_name(derived[2], enums_out)  # type: ignore[arg-type]
        impls_out = tuple(derived[3]) + impls_out
    if native_tests_dir is not None:
        derived = _specs_from_native_tests(native_tests_dir, module_name=module_name)
        functions_out = _merge_by_name(derived[0], functions_out)  # type: ignore[arg-type]
        structs_out = _merge_by_name(derived[1], structs_out)  # type: ignore[arg-type]
        enums_out = _merge_by_name(derived[2], enums_out)  # type: ignore[arg-type]
        impls_out = tuple(derived[3]) + impls_out

    version = pyo3_version or _default_pyo3_version()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version):
        _fail(f"invalid PyO3 version {version!r}")

    lines: list[str] = [
        "// Generated by rs_generate_pyo3_scaffold; regenerate instead of editing.",
        "#![allow(unused_mut, unused_variables, dead_code, unused_imports)]",
        "",
        "use pyo3::prelude::*;",
        "use pyo3::types::PyTuple;",
        "use pyo3::IntoPyObjectExt;",
        "",
        _emit_typed_error_bridge(module_name),
        "",
        rs_emit_raii_guards(()),
        "",
        rs_emit_monomorphized_structs(
            structs_out, enums_out, substitutions=substitutions
        ),
        rs_emit_trait_impls_and_bounds(impls_out),
    ]

    for function in functions_out:
        _balanced(function.body, what=f"body of {function.name!r}")
        if not function.body.strip():
            _fail(f"function {function.name!r} needs a Rust body")
        arg_mapping = rs_map_python_args_to_rust(function.params)
        result_text = _canonical_type(function.result) if function.result else "()"
        result_mapping = rs_map_rust_results_to_py(result_text)
        return_type = (
            f"PyResult<{result_mapping.ok_type}>"
            if result_mapping.ok_type != "()"
            else "PyResult<()>"
        )
        lines.extend(
            [
                *(
                    f"/// {line.strip()}"
                    for line in function.doc.splitlines()
                    if line.strip()
                ),
                "#[pyfunction]",
                (
                    f"fn {_identifier(function.name, 'function name')}("
                    "py: Python<'_>, args: &Bound<'_, PyTuple>"
                    f") -> {return_type} {{"
                ),
            ]
        )
        for line in arg_mapping.code.splitlines():
            lines.append(f"    {line}")
        lines.append("    let __result = {")
        for line in function.body.splitlines():
            lines.append(f"        {line}")
        lines.append("    };")
        for line in result_mapping.code.splitlines():
            lines.append(f"    {line}")
        lines.extend(["}", ""])

    lines.extend(
        [
            "#[pymodule]",
            f"fn {module_name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{",
            '    m.add("PillarError", m.py().get_type::<PyPillarError>())?;',
        ]
    )
    for function in functions_out:
        lines.append(f"    m.add_function(wrap_pyfunction!({function.name}, m)?)?;")
    lines.extend(["    Ok(())", "}", ""])

    cargo_toml = "\n".join(
        [
            "[package]",
            f'name = "{module_name}"',
            'version = "0.1.0"',
            f'edition = "{edition}"',
            "",
            "[lib]",
            f'name = "{module_name}"',
            'crate-type = ["cdylib"]',
            "",
            "[dependencies]",
            f'pyo3 = {{ version = "{version}", features = ["extension-module"] }}',
            "",
        ]
    )
    pyproject = "\n".join(
        [
            "[build-system]",
            'requires = ["maturin>=1.0,<2.0"]',
            'build-backend = "maturin"',
            "",
            "[project]",
            f'name = "{module_name}"',
            'version = "0.1.0"',
            'requires-python = ">=3.14"',
            "",
            "[tool.maturin]",
            f'module-name = "{module_name}"',
            "",
        ]
    )
    return {
        "Cargo.toml": cargo_toml,
        "pyproject.toml": pyproject,
        "src/lib.rs": "\n".join(lines) + "\n",
    }


def rs_verify_scaffold_matches_generator(
    root: str | Path,
    generated: Mapping[str, str] | None = None,
    *,
    module_name: str = "polyglot_native",
    functions: Sequence[RsFunctionSpec | Mapping[str, Any]] = (),
    structs: Sequence[RsStructSpec | Mapping[str, Any]] = (),
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
    impls: Sequence[RsTraitImplSpec | Mapping[str, Any]] = (),
    substitutions: Mapping[str, str] | None = None,
    program: Program | None = None,
    native_tests_dir: str | Path | None = None,
    pyo3_version: str | None = None,
    ignore_extra: bool = True,
) -> ScaffoldVerification:
    """Compare checked-in scaffold files with generator output to block drift."""
    expected = dict(
        generated
        if generated is not None
        else rs_generate_pyo3_scaffold(
            module_name=module_name,
            functions=functions,
            structs=structs,
            enums=enums,
            impls=impls,
            substitutions=substitutions,
            program=program,
            native_tests_dir=native_tests_dir,
            pyo3_version=pyo3_version,
        )
    )
    base = Path(root)
    matches: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    details: list[str] = []
    for relative in sorted(expected):
        path = base / relative
        if not path.is_file():
            missing.append(relative)
            details.append(f"{relative}: file is missing")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual == expected[relative]:
            matches.append(relative)
            continue
        mismatched.append(relative)
        expected_lines = expected[relative].splitlines()
        actual_lines = actual.splitlines()
        line_number = next(
            (
                index + 1
                for index, (left, right) in enumerate(zip(expected_lines, actual_lines))
                if left != right
            ),
            min(len(expected_lines), len(actual_lines)) + 1,
        )
        details.append(
            f"{relative}: first difference at line {line_number} "
            f"(generated {len(expected_lines)} lines, found {len(actual_lines)})"
        )

    extra: list[str] = []
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(base).as_posix()
            parts = set(relative.split("/"))
            if parts & {"target", ".git", "__pycache__", ".pytest_cache"}:
                continue
            if relative not in expected:
                extra.append(relative)
    ok = not missing and not mismatched and (ignore_extra or not extra)
    return ScaffoldVerification(
        ok=ok,
        matches=tuple(matches),
        mismatched=tuple(mismatched),
        missing=tuple(missing),
        extra=tuple(extra),
        details=tuple(details),
    )


# ---------------------------------------------------------------------------
# 43 / 44 — type and trait emission
# ---------------------------------------------------------------------------


def rs_emit_monomorphized_structs(
    structs: Sequence[RsStructSpec | Mapping[str, Any]],
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
    *,
    substitutions: Mapping[str, str] | None = None,
    name_overrides: Mapping[str, str] | None = None,
) -> str:
    """Emit Rust structs/enums with field order and visibility preserved.

    Generic parameters are substituted with concrete types; emitted names are
    mangled deterministically (``Box<T>`` with ``T = i64`` becomes ``BoxI64``).
    """
    global_subs = dict(substitutions or {})
    for parameter, replacement in global_subs.items():
        _identifier(parameter, "type parameter")
        _type_text(replacement, "type substitution")
    overrides = dict(name_overrides or {})
    used_names: set[str] = set()
    chunks: list[str] = []

    for spec in _spec_list(structs, "struct"):
        local = {**global_subs, **dict(spec.substitutions)}
        missing = [param for param in spec.generic_params if param not in local]
        if missing:
            _fail(
                f"struct {spec.name!r} has no monomorphization for "
                f"type parameter(s) {', '.join(missing)}",
                code="E001",
            )
        emitted_name = overrides.get(spec.name) or spec.name
        if spec.generic_params:
            emitted_name += "".join(
                _type_suffix(local[param]) for param in spec.generic_params
            )
        emitted_name = _dedupe_name(
            _identifier(emitted_name, "struct name"), used_names
        )
        lines: list[str] = []
        if spec.pyclass:
            lines.append("#[pyclass]")
        if spec.derives:
            lines.append(f"#[derive({', '.join(spec.derives)})]")
        visibility = _visibility(spec.visibility, "struct visibility")
        lines.append(
            f"{visibility + ' ' if visibility else ''}struct {emitted_name} {{"
        )
        for field_spec in spec.fields:
            field_name = _identifier(field_spec.name, "field name")
            field_vis = _visibility(field_spec.visibility, "field visibility")
            field_type = _substitute_type(field_spec.type, local)
            leftovers = [
                p for p in spec.generic_params if re.search(rf"\b{p}\b", field_type)
            ]
            if leftovers:
                _fail(
                    f"field {spec.name}.{field_spec.name} still mentions generic "
                    f"parameter(s) {', '.join(leftovers)}",
                    code="E001",
                )
            if field_spec.python_access and spec.pyclass:
                lines.append("    #[pyo3(get, set)]")
            prefix = f"{field_vis} " if field_vis else ""
            lines.append(f"    {prefix}{field_name}: {field_type},")
        lines.extend(["}", ""])
        chunks.append("\n".join(lines))

    for spec in _spec_list(enums, "enum"):
        local = {**global_subs, **dict(spec.substitutions)}
        missing = [param for param in spec.generic_params if param not in local]
        if missing:
            _fail(
                f"enum {spec.name!r} has no monomorphization for "
                f"type parameter(s) {', '.join(missing)}",
                code="E001",
            )
        emitted_name = overrides.get(spec.name) or spec.name
        if spec.generic_params:
            emitted_name += "".join(
                _type_suffix(local[param]) for param in spec.generic_params
            )
        emitted_name = _dedupe_name(_identifier(emitted_name, "enum name"), used_names)
        lines = []
        if spec.pyclass:
            lines.append("#[pyclass(eq, eq_int)]")
        if spec.derives:
            lines.append(f"#[derive({', '.join(spec.derives)})]")
        visibility = _visibility(spec.visibility, "enum visibility")
        lines.append(f"{visibility + ' ' if visibility else ''}enum {emitted_name} {{")
        for variant in spec.variants:
            variant_name = _identifier(variant.name, "variant name")
            types = tuple(_substitute_type(item, local) for item in variant.field_types)
            if types:
                lines.append(f"    {variant_name}({', '.join(types)}),")
            else:
                lines.append(f"    {variant_name},")
        lines.extend(["}", ""])
        chunks.append("\n".join(lines))

    return "\n".join(chunks)


def rs_emit_trait_impls_and_bounds(
    impls: Sequence[RsTraitImplSpec | Mapping[str, Any]],
) -> str:
    """Emit impl blocks with syntactically valid, rustc-clean where bounds."""
    chunks: list[str] = []
    for spec in _spec_list(impls, "impl"):
        target = _canonical_type(spec.target)
        _type_path(target, "impl target")
        trait = _canonical_type(spec.trait) if spec.trait else None
        if trait is not None:
            _type_path(trait, "trait name")
        generics = tuple(
            _identifier(param, "impl type parameter") for param in spec.generic_params
        )
        bounds = tuple(_bound_text(bound, "where bound") for bound in spec.bounds)
        header = "impl"
        if generics:
            header += f"<{', '.join(generics)}>"
        header += f" {trait} for {target}" if trait else f" {target}"
        lines = [header if bounds else f"{header} {{"]
        if bounds:
            lines.append("where")
            lines.extend(f"    {bound}," for bound in bounds)
            lines.append("{")
        for method in spec.methods:
            lines.extend(_render_method(method))
        lines.extend(["}", ""])
        chunks.append("\n".join(lines))
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# 45 / 46 — boundary value mapping
# ---------------------------------------------------------------------------


def _coerce_value(value: Any, rust_type: str, label: str) -> Any:
    text = _canonical_type(rust_type)
    name, args_text = _named_type_parts(text)
    args = _split_top_level(args_text) if args_text else []
    if name == "Option" and len(args) == 1:
        return None if value is None else _coerce_value(value, args[0], label)
    if name in {"Vec", "Array", "ArrayVec"} and len(args) == 1:
        if not isinstance(value, (list, tuple)):
            _fail(f"{label} must be a list for {text}", code="E001")
        return [
            _coerce_value(item, args[0], f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if name in {"HashMap", "BTreeMap"} and len(args) == 2:
        if not isinstance(value, Mapping):
            _fail(f"{label} must be a mapping for {text}", code="E001")
        return {
            _coerce_value(key, args[0], f"{label} key"): _coerce_value(
                item, args[1], f"{label}[{key!r}]"
            )
            for key, item in value.items()
        }
    if name == "bool":
        if not isinstance(value, bool):
            _fail(f"{label} must be a bool, got {type(value).__name__}", code="E001")
        return value
    if name in _INT_RANGES:
        if not isinstance(value, int) or isinstance(value, bool):
            _fail(f"{label} must be an int for {text}", code="E001")
        low, high = _INT_RANGES[name]
        if not low <= value <= high:
            _fail(
                f"{label} value {value} is outside the range of Rust {text} "
                f"[{low}, {high}]",
                code="E001",
            )
        return value
    if name in {"f32", "f64"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _fail(f"{label} must be a number for {text}", code="E001")
        return float(value)
    if name in {"String", "str", "char"}:
        if not isinstance(value, str):
            _fail(f"{label} must be a str for {text}", code="E001")
        if name == "char" and len(value) != 1:
            _fail(f"{label} must contain exactly one character for char", code="E001")
        return value
    if text == "()":
        if value is not None:
            _fail(f"{label} must be None for Rust unit type", code="E001")
        return None
    _fail(f"unsupported Rust type {text!r} for Python argument {label!r}", code="E001")


def rs_map_python_args_to_rust(
    params: Sequence[RsParamSpec | Mapping[str, Any] | tuple[str, str]],
    *,
    values: Mapping[str, Any] | None = None,
    args_expr: str = "args",
) -> RsArgMapping:
    """Map Python arguments to Rust extractions and validate supplied values.

    The generated code extracts from a ``&Bound<PyTuple>`` argument tuple with
    explicit ``PyTypeError`` messages; when ``values`` is given the same types
    are validated and returned in Python form with deterministic diagnostics.
    """
    _identifier(args_expr, "argument expression")
    specs: list[RsParamSpec] = []
    for param in params:
        if isinstance(param, tuple) and len(param) == 2:
            specs.append(RsParamSpec(name=param[0], type=param[1]))
        else:
            specs.append(_coerce_spec(param, "param"))

    code_lines = [f"let __py_args = {args_expr};"]
    checked: dict[str, Any] = {}
    rust_types: dict[str, str] = {}
    diagnostics: list[str] = []
    for index, param in enumerate(specs):
        name = _identifier(param.name, "argument name")
        ty = _canonical_type(param.type)
        rust_types[name] = ty
        missing_message = _string_literal(
            f"missing required argument '{name}' at position {index}"
        )
        type_message = _string_literal(
            f"argument '{name}' at position {index} must be {ty}"
        )
        if param.required:
            code_lines.extend(
                [
                    (
                        f"let __arg_{index} = __py_args.get_item({index}).map_err(|_| "
                        f"pyo3::exceptions::PyTypeError::new_err({missing_message}))?;"
                    ),
                    (
                        f"let {name}: {ty} = __arg_{index}.extract::<{ty}>().map_err(|_| "
                        f"pyo3::exceptions::PyTypeError::new_err({type_message}))?;"
                    ),
                ]
            )
        else:
            if param.default is None or not param.default.strip():
                _fail(f"optional argument {name!r} needs a Rust default expression")
            _balanced(param.default, what=f"default of {name!r}")
            code_lines.append(
                f"let {name}: {ty} = match __py_args.get_item({index}) {{"
            )
            code_lines.append(
                "    Ok(item) => item.extract::<"
                f"{ty}>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("
                f"{type_message}))?,"
            )
            code_lines.append(f"    Err(_) => {param.default},")
            code_lines.append("};")
        if values is not None:
            if name not in values:
                if param.required:
                    _fail(
                        f"missing required argument '{name}' at position {index}",
                        code="E001",
                    )
                continue
            original = values[name]
            checked[name] = _coerce_value(original, ty, f"argument '{name}'")
            if (
                isinstance(original, int)
                and not isinstance(original, bool)
                and ty in {"f32", "f64"}
            ):
                diagnostics.append(
                    f"argument '{name}' was widened from int to Rust {ty}"
                )

    return RsArgMapping(
        code="\n".join(code_lines),
        values=checked,
        rust_types=rust_types,
        diagnostics=tuple(diagnostics),
    )


def rs_map_rust_results_to_py(
    rust_type: str | Mapping[str, Any],
    *,
    error_source: str = "__result",
    error_class: str = "PyPillarError",
) -> RsResultMapping:
    """Map ``Result<T, E>`` (or a plain ``T``) into a typed PyO3 return."""
    if isinstance(rust_type, Mapping):
        payload = dict(rust_type)
        error_source = str(payload.pop("error_source", error_source))
        error_class = str(payload.pop("error_class", error_class))
        if len(payload) != 1:
            _fail("result mapping needs exactly one Rust type entry")
        text = str(next(iter(payload.values())))
    else:
        text = str(rust_type)
    _identifier(error_source, "result variable name")
    _identifier(error_class, "Python exception name")

    name, args_text = _named_type_parts(_canonical_type(text))
    args = _split_top_level(args_text) if args_text else []
    if name == "Result" and len(args) == 2:
        ok_type, error_type = args[0], args[1]
    elif name == "Result":
        _fail("Result requires both an Ok and an Err type", code="E001")
    else:
        ok_type, error_type = text, None

    if error_type is None:
        code = f"Ok({error_source})"
        python_exception = "None"
    elif error_type in {"PillarError", "RsPillarError"}:
        python_exception = "PillarError"
        code = "\n".join(
            [
                f"match {error_source} {{",
                "    Ok(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),",
                f"    Err(error) => Err({error_class}::new_err(error.format_pillar())),",
                "}",
            ]
        )
    elif error_type in {"pyo3::PyErr", "PyErr"}:
        python_exception = "PyErr"
        code = "\n".join(
            [
                f"match {error_source} {{",
                "    Ok(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),",
                "    Err(error) => Err(error),",
                "}",
            ]
        )
    elif error_type == "String":
        python_exception = "ValueError"
        code = "\n".join(
            [
                f"match {error_source} {{",
                "    Ok(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),",
                (
                    "    Err(message) => Err(pyo3::exceptions::PyValueError::new_err("
                    "message)),"
                ),
                "}",
            ]
        )
    else:
        _fail(
            f"unsupported Rust error type {error_type!r}; use PillarError, "
            "pyo3::PyErr, or String",
            code="E001",
        )
    return RsResultMapping(
        code=code,
        ok_type=_canonical_type(ok_type),
        error_type=error_type,
        python_exception=python_exception,
        rust_exception=error_class,
    )


# ---------------------------------------------------------------------------
# 47 — typed Pillar error
# ---------------------------------------------------------------------------


def rs_raise_typed_pillar_error(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
    note: str | None = None,
    line: int = 0,
    col: int = 0,
    span: tuple[int, int] | None = None,
) -> NoReturn:
    """Raise :class:`RsPillarError` instead of a raw ``PyRuntimeError``.

    The exception mirrors the Rust ``PyPillarError`` bridge emitted by
    :func:`rs_generate_pyo3_scaffold` and keeps the E001-E008 diagnostic codes.
    """
    if code not in ERROR_CODES:
        raise RsPillarError(
            "E003",
            f"unknown Pillar diagnostic code {code!r}; expected one of "
            f"{', '.join(ERROR_CODES)}",
        )
    raise RsPillarError(
        code,
        message,
        reason=reason,
        help_text=help_text,
        note=note,
        line=line,
        col=col,
        span=span,
    )


# ---------------------------------------------------------------------------
# 48 — GIL and Python reference handling
# ---------------------------------------------------------------------------


def rs_manage_gil_and_py_refs(
    work: str,
    *,
    borrowed: Sequence[str] = (),
    owned: Sequence[str] = (),
    reentrant: bool = True,
    result_type: str = "i64",
) -> RsGilPlan:
    """Plan GIL-minimizing Rust code that never holds borrowed refs across calls.

    Borrowed ``Bound`` references are cloned into owned ``Py<T>`` handles before
    ``Python::allow_threads``; when ``reentrant`` is true, directly using a
    borrowed name inside the released-GIL closure is rejected with E007.
    """
    _balanced(work, what="GIL work expression")
    if not work.strip():
        _fail("GIL work expression must not be empty")
    ty = _canonical_type(result_type)
    borrowed_names = tuple(_identifier(name, "borrowed reference") for name in borrowed)
    owned_names = tuple(_identifier(name, "owned reference") for name in owned)
    overlap = set(borrowed_names) & set(owned_names)
    if overlap:
        _fail(f"names cannot be both borrowed and owned: {sorted(overlap)}")

    warnings: list[str] = []
    for name in borrowed_names:
        if re.search(rf"\b{re.escape(name)}\b", work):
            if reentrant:
                _fail(
                    f"borrowed PyO3 reference '{name}' would cross a re-entrant "
                    "GIL-release boundary; clone it into an owned Py<T> first",
                    code="E007",
                )
            warnings.append(
                f"borrowed reference '{name}' is only safe while the GIL is held"
            )
    if re.search(r"\bpy\.|Python::", work):
        warnings.append(
            "work calls Python APIs inside allow_threads; move them outside"
        )
    if re.search(r"\bself\b", work):
        warnings.append("work captures 'self'; prefer owned field snapshots")

    lines = [
        "// GIL plan: release only for pure Rust work; keep owned Py<T> handles.",
    ]
    for name in borrowed_names:
        lines.append(f"let __gil_owned_{name} = {name}.clone().unbind();")
    for name in owned_names:
        lines.append(f"let __gil_move_{name} = {name};")
    lines.append(f"let __gil_result: {ty} = py.allow_threads(|| {{")
    for line in work.splitlines():
        lines.append(f"    {line}")
    lines.append("});")
    rules = (
        "extract/clone all borrowed Bound<T> values before allow_threads",
        "never call Python APIs or re-enter Python from inside the closure",
        "re-acquire the GIL only after the pure Rust work has finished",
    )
    return RsGilPlan(code="\n".join(lines), rules=rules, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# 49 — RAII guards
# ---------------------------------------------------------------------------


def rs_emit_raii_guards(
    guards: Sequence[RsGuardSpec | Mapping[str, Any]],
    *,
    include_runtime: bool = True,
) -> str:
    """Emit RAII guard templates whose Drop impl releases on every return path."""
    specs = _spec_list(guards, "guard")
    if not specs and not include_runtime:
        return ""
    lines: list[str] = []
    if include_runtime:
        lines.extend(
            [
                "// RAII guards release locks/handles/buffers on every return path.",
                '#[must_use = "the guard releases its resource when dropped"]',
                "struct PolyglotRaiiGuard<T> {",
                "    resource: Option<T>,",
                "    on_drop: Option<std::boxed::Box<dyn FnOnce(T)>>,",
                "}",
                "",
                "impl<T> PolyglotRaiiGuard<T> {",
                "    fn new(resource: T, on_drop: std::boxed::Box<dyn FnOnce(T)>) -> Self {",
                "        Self { resource: Some(resource), on_drop: Some(on_drop) }",
                "    }",
                "",
                "    fn get(&self) -> Option<&T> {",
                "        self.resource.as_ref()",
                "    }",
                "",
                "    fn get_mut(&mut self) -> Option<&mut T> {",
                "        self.resource.as_mut()",
                "    }",
                "",
                "    fn into_inner(mut self) -> Option<T> {",
                "        self.resource.take()",
                "    }",
                "}",
                "",
                "impl<T> Drop for PolyglotRaiiGuard<T> {",
                "    fn drop(&mut self) {",
                "        let on_drop = self.on_drop.take();",
                "        if let (Some(resource), Some(on_drop)) = (self.resource.take(), on_drop) {",
                "            on_drop(resource);",
                "        }",
                "    }",
                "}",
                "",
            ]
        )
    default_release = {
        "lock": "drop(__resource);",
        "handle": "drop(__resource);",
        "buffer": "drop(__resource);",
    }
    for spec in sorted(specs, key=lambda item: (item.kind, item.name)):
        name = _identifier(spec.name, "guard name")
        if spec.kind not in default_release:
            _fail(
                f"guard {name!r} has unknown kind {spec.kind!r}; "
                f"expected one of {sorted(default_release)}"
            )
        acquire = _balanced(spec.acquire, what=f"acquire expression of guard {name!r}")
        release = spec.release or default_release[spec.kind]
        _balanced(release, what=f"release expression of guard {name!r}")
        if not acquire.strip() or not release.strip():
            _fail(f"guard {name!r} needs non-empty acquire/release expressions")
        lines.extend(
            [
                f"// {spec.kind} guard '{name}': released by Drop on all paths.",
                f"let __raii_{name} = PolyglotRaiiGuard::new(",
                f"    {acquire},",
                "    std::boxed::Box::new(|__resource| {",
                f"        {release}",
                "    }),",
                ");",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# 50 — isolated Cargo checks
# ---------------------------------------------------------------------------


def _cargo_diagnostics(*streams: str) -> tuple[str, ...]:
    found: list[str] = []
    for stream in streams:
        for line in stream.splitlines():
            stripped = line.strip()
            if stripped.startswith(("error", "warning")) and stripped not in found:
                found.append(stripped)
    return tuple(found)


def _run_cargo_step(
    name: str,
    command: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    target_dir: Path,
    runner: CargoRunner | None,
) -> CargoStepResult:
    if runner is not None:
        completed = runner(command, cwd, env)
    elif shutil.which(command[0]) is None:
        completed = subprocess.CompletedProcess(
            command,
            127,
            "",
            f"{command[0]} not found on PATH",
        )
    else:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    return CargoStepResult(
        name=name,
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        diagnostics=_cargo_diagnostics(completed.stdout or "", completed.stderr or ""),
        target_dir=str(target_dir),
    )


def rs_run_cargo_check_and_clippy(
    crate_root: str | Path,
    *,
    target_dir: str | Path | None = None,
    clippy: bool = True,
    deny_warnings: bool = False,
    extra_args: Sequence[str] = (),
    runner: CargoRunner | None = None,
) -> CargoCheckResult:
    """Run cargo check and clippy in separate isolated target directories.

    No source files are modified.  A custom ``runner`` receives the exact argv,
    crate root, and environment so tests can avoid a real Cargo build while
    still exercising the command construction and structured result parsing.
    """
    root = Path(crate_root)
    if not (root / "Cargo.toml").is_file():
        _fail(
            f"no Cargo.toml found in crate root {root}",
            code="E006",
        )
    extras = tuple(extra_args)
    if any(not isinstance(arg, str) for arg in extras):
        _fail("extra_args must contain only strings")
    if "--target-dir" in extras:
        _fail("extra_args may not override --target-dir (checks must stay isolated)")

    base = Path(target_dir) if target_dir is not None else root / "target" / "f5-check"
    check_dir = base / "check"
    clippy_dir = base / "clippy"
    for directory in (check_dir, clippy_dir if clippy else check_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def environment(directory: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(directory)
        env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
        return env

    check_command = (
        "cargo",
        "check",
        "--all-targets",
        "--target-dir",
        str(check_dir),
        *extras,
    )
    check_result = _run_cargo_step(
        "check", check_command, root, environment(check_dir), check_dir, runner
    )
    clippy_result: CargoStepResult | None = None
    if clippy:
        clippy_command = (
            "cargo",
            "clippy",
            "--all-targets",
            "--target-dir",
            str(clippy_dir),
            *extras,
            *(("--", "-D", "warnings") if deny_warnings else ()),
        )
        clippy_result = _run_cargo_step(
            "clippy",
            clippy_command,
            root,
            environment(clippy_dir),
            clippy_dir,
            runner,
        )
    ok = check_result.ok and (clippy_result is None or clippy_result.ok)
    return CargoCheckResult(
        ok=ok,
        check=check_result,
        clippy=clippy_result,
        target_dir=str(base),
    )


__all__ = [
    "CargoCheckResult",
    "CargoRunner",
    "CargoStepResult",
    "RsArgMapping",
    "RsEnumSpec",
    "RsFieldSpec",
    "RsFunctionSpec",
    "RsGilPlan",
    "RsGuardSpec",
    "RsParamSpec",
    "RsPillarError",
    "RsResultMapping",
    "RsStructSpec",
    "RsTraitImplSpec",
    "RsVariantSpec",
    "ScaffoldVerification",
    "rs_emit_monomorphized_structs",
    "rs_emit_raii_guards",
    "rs_emit_trait_impls_and_bounds",
    "rs_generate_pyo3_scaffold",
    "rs_manage_gil_and_py_refs",
    "rs_map_python_args_to_rust",
    "rs_map_rust_results_to_py",
    "rs_raise_typed_pillar_error",
    "rs_run_cargo_check_and_clippy",
    "rs_verify_scaffold_matches_generator",
]
