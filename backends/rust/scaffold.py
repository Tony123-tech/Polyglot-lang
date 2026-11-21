"""Deterministic PyO3 crate scaffolding for the Rust native backend.

Every function in this module turns compiler-facing metadata (structs, enums,
functions and impl blocks) into stable Rust or TOML text.  No subprocesses or
filesystem writes are performed: callers receive structured records that can be
diffed, hashed, or written by the rollout driver.  Invalid layouts and ABI
combinations raise :class:`PolyglotError` with the framework E001-E008 codes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, NoReturn

from backends.rust.framework import (
    RsEnumSpec,
    RsFieldSpec,
    RsFunctionSpec,
    RsParamSpec,
    RsStructSpec,
    RsTraitImplSpec,
    RsVariantSpec,
)
from compiler.frontend.errors import PolyglotError

__all__ = [
    "rsc_abi_settings",
    "rsc_crate_layout",
    "rsc_crate_manifest",
    "rsc_derive_line",
    "rsc_doc_comment",
    "rsc_enum_def",
    "rsc_feature_flags",
    "rsc_field_line",
    "rsc_function_def",
    "rsc_generic_suffix",
    "rsc_layout_plan",
    "rsc_method_def",
    "rsc_module_header",
    "rsc_module_init",
    "rsc_param_decl",
    "rsc_pyproject_manifest",
    "rsc_return_type",
    "rsc_scaffold_diff",
    "rsc_scaffold_fingerprint",
    "rsc_scaffold_manifest",
    "rsc_scaffold_plan",
    "rsc_scaffold_render",
    "rsc_struct_def",
    "rsc_trait_impl",
    "rsc_type_substitute",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RUST_KEYWORDS = frozenset(
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
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TYPE_PATH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*\Z")
_IMPORT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*(?:::\*)?\Z"
)
_FEATURE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}\Z")
_EDITIONS = ("2018", "2021", "2024")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^A-Za-z0-9_]+")
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured native-scaffold diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-scaffold")


def _text(value: object, what: str) -> str:
    """Validate and normalize one caller-supplied string."""
    if not isinstance(value, str):
        _fail("E003", f"{what} must be a str, got {type(value).__name__}")
    return value


def _identifier(value: object, what: str) -> str:
    """Validate a Rust identifier without silently rewriting reserved words."""
    text = _text(value, what)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a valid Rust identifier, got {text!r}")
    if text in _RUST_KEYWORDS and text not in {"self", "Self"}:
        _fail("E003", f"{what} uses the reserved Rust keyword {text!r}")
    return text


def _type_path(value: object, what: str) -> str:
    """Validate a dotted Rust type path such as ``std::string::String``."""
    text = _text(value, what)
    if _TYPE_PATH_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a Rust type path, got {text!r}")
    return text


def _import_path(value: object, what: str) -> str:
    """Validate a Rust ``use`` path, including a trailing ``::*`` glob."""
    text = _text(value, what)
    if _IMPORT_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a Rust use path, got {text!r}")
    return text


def _feature_name(value: object, what: str) -> str:
    """Validate a Cargo feature name, which may contain hyphens."""
    text = _text(value, what)
    if _FEATURE_RE.fullmatch(text) is None:
        _fail("E003", f"{what} must be a valid Cargo feature name, got {text!r}")
    return text


def _type_text(value: object, what: str) -> str:
    """Validate a Rust type expression, including nested generic brackets."""
    text = " ".join(_text(value, what).split())
    if not text:
        _fail("E003", f"{what} must not be empty")
    if ";" in text or "\n" in text:
        _fail("E003", f"{what} must be a type expression, got {text!r}")
    if text.count("<") != text.count(">"):
        _fail("E001", f"{what} has unbalanced generic brackets: {text!r}")
    return text


def _balanced(value: object, what: str) -> str:
    """Validate balanced Rust delimiters in a code fragment."""
    text = _text(value, what)
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in _PAIRS and (not stack or stack.pop() != _PAIRS[char]):
            _fail("E003", f"{what} has unbalanced delimiters")
    if stack:
        _fail("E003", f"{what} has unbalanced delimiters")
    return text


def _split_top_level(text: str, separator: str = ",") -> tuple[str, ...]:
    """Split generic arguments while respecting nested brackets."""
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
    return tuple(pieces)


def _named_type(text: str) -> tuple[str, tuple[str, ...]]:
    """Split ``Name<A, B>`` into its base name and top-level arguments."""
    normalized = _type_text(text, "Rust type")
    if "<" not in normalized:
        return normalized, ()
    base, rest = normalized.split("<", 1)
    base = base.strip()
    if _TYPE_PATH_RE.fullmatch(base) is None or not rest.endswith(">"):
        _fail("E001", f"malformed Rust type {text!r}")
    return base, _split_top_level(rest[:-1])


def _string_literal(value: str) -> str:
    """Render a stable escaped Rust string literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    """Coerce a mapping-like input or raise E003."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _coerce_spec(value: Any, kind: str) -> Any:
    """Coerce a dataclass or mapping into the framework spec dataclasses."""
    factories: dict[str, Any] = {
        "field": RsFieldSpec,
        "param": RsParamSpec,
        "variant": RsVariantSpec,
        "struct": RsStructSpec,
        "enum": RsEnumSpec,
        "function": RsFunctionSpec,
        "impl": RsTraitImplSpec,
    }
    aliases: dict[str, dict[str, str]] = {
        "field": {"rust_type": "type"},
        "param": {"rust_type": "type"},
        "function": {"rust_type": "result", "returns": "result"},
        "impl": {"where": "bounds"},
    }
    nested: dict[str, dict[str, str]] = {
        "struct": {"fields": "field"},
        "enum": {"variants": "variant"},
        "function": {"params": "param"},
        "impl": {"methods": "function"},
    }
    factory = factories[kind]
    if isinstance(value, factory):
        if kind not in nested:
            return value
        payload = asdict(value)
    else:
        payload = _as_mapping(value, f"{kind} spec")
    renamed: dict[str, Any] = {
        aliases.get(kind, {}).get(str(key), str(key)): item
        for key, item in payload.items()
    }
    for field_name, child_kind in nested.get(kind, {}).items():
        children = renamed.get(field_name, ())
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            _fail("E003", f"{field_name} of {kind} spec must be a sequence")
        renamed[field_name] = tuple(
            _coerce_spec(child, child_kind) for child in children
        )
    try:
        return factory(**renamed)
    except TypeError as error:
        _fail("E003", f"invalid {kind} spec: {error}")


def _spec_list(values: Sequence[Any], kind: str) -> tuple[Any, ...]:
    """Coerce a sequence of specs, rejecting a bare non-sequence."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        _fail("E003", f"{kind} specs must be a sequence")
    return tuple(_coerce_spec(value, kind) for value in values)


def _dedupe(name: str, used: set[str]) -> str:
    """Return a deterministic unique name for duplicate emitted definitions."""
    if name not in used:
        used.add(name)
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
    candidate = f"{name}_{digest}"
    while candidate in used:
        candidate += "_"
    used.add(candidate)
    return candidate


def _substitute(
    text: str, substitutions: Mapping[str, str]
) -> tuple[str, tuple[str, ...]]:
    """Replace whole-word generic parameters and report unresolved names."""
    result = _type_text(text, "Rust type")
    known = {
        _identifier(key, "type parameter"): _type_text(item, "type substitution")
        for key, item in _as_mapping(substitutions, "substitutions").items()
    }
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(result):
        replacement = known.get(token)
        tokens.append(replacement if replacement is not None else token)
    result = " ".join("".join(tokens).split())
    unresolved = tuple(
        token
        for token in _TOKEN_RE.findall(result)
        if token in known and known[token] != token
    )
    return result, unresolved


def _render_doc(doc: object, indent: str) -> tuple[str, tuple[str, ...]]:
    """Render documentation text as Rust doc comments."""
    lines = tuple(
        f"{indent}/// {line.strip()}"
        for line in _text(doc, "doc text").splitlines()
        if line.strip()
    )
    return ("\n".join(lines), lines)


def _result_conversion_lines(error_type: str | None) -> tuple[str, ...]:
    """Emit the typed ``Ok``/``Err`` conversion for a Rust result type."""
    if error_type is None:
        return ("Ok(__result)",)
    if error_type in {"PillarError", "RsPillarError"}:
        return (
            "match __result {",
            "    Ok(value) => Ok(value),",
            "    Err(error) => Err(PyPillarError::new_err(error.format_pillar())),",
            "}",
        )
    if error_type in {"String", "str"}:
        return (
            "match __result {",
            "    Ok(value) => Ok(value),",
            (
                "    Err(message) => Err(pyo3::exceptions::PyValueError::new_err("
                "message)),"
            ),
            "}",
        )
    _fail("E007", f"unsupported Rust error type {error_type!r} in scaffold wrapper")


def _base_type_name(text: str) -> str:
    """Extract the leading Rust type name without validating the whole type."""
    match = re.match(
        r"\s*&?\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?([A-Za-z_][A-Za-z0-9_]*)", text
    )
    if match is None:
        _fail("E003", f"cannot determine base type of {text!r}")
    return match.group(1)


# ---------------------------------------------------------------------------
# 1. rsc_crate_manifest
# ---------------------------------------------------------------------------


def rsc_crate_manifest(
    *,
    module_name: str = "polyglot_native",
    pyo3_version: str = "0.26",
    edition: str = "2021",
    features: Sequence[str] = ("extension-module",),
    crate_type: str = "cdylib",
) -> dict[str, Any]:
    """Render a deterministic Cargo manifest for a PyO3 extension crate."""
    name = _identifier(module_name, "module name")
    version = _text(pyo3_version, "PyO3 version")
    if _VERSION_RE.fullmatch(version) is None:
        _fail("E003", f"invalid PyO3 version {version!r}")
    if edition not in _EDITIONS:
        _fail("E001", f"unsupported Rust edition {edition!r}; expected {_EDITIONS}")
    if crate_type not in {"cdylib", "rlib", "staticlib"}:
        _fail("E001", f"unsupported crate type {crate_type!r}")
    selected = tuple(_feature_name(feature, "Cargo feature") for feature in features)
    if not selected:
        _fail("E001", "a PyO3 extension crate needs at least one Cargo feature")
    feature_text = ", ".join(_string_literal(feature) for feature in selected)
    text = "\n".join(
        [
            "[package]",
            f"name = {_string_literal(name)}",
            'version = "0.1.0"',
            f"edition = {_string_literal(edition)}",
            "",
            "[lib]",
            f"name = {_string_literal(name)}",
            f"crate-type = {_string_literal(crate_type)}",
            "",
            "[dependencies]",
            f"pyo3 = {{ version = {_string_literal(version)}, features = [{feature_text}] }}",
            "",
        ]
    )
    return {
        "path": "Cargo.toml",
        "text": text,
        "module_name": name,
        "pyo3_version": version,
        "edition": edition,
        "features": selected,
        "crate_type": crate_type,
    }


# ---------------------------------------------------------------------------
# 2. rsc_pyproject_manifest
# ---------------------------------------------------------------------------


def rsc_pyproject_manifest(
    *,
    module_name: str = "polyglot_native",
    requires_python: str = ">=3.14",
    build_backend: str = "maturin",
    build_requires: Sequence[str] = ("maturin>=1.0,<2.0",),
) -> dict[str, Any]:
    """Render the maturin pyproject metadata that accompanies the crate."""
    name = _identifier(module_name, "module name")
    requirement = _text(requires_python, "requires-python")
    if re.fullmatch(r"(>=|==|~=)[0-9]+\.[0-9]+(?:\.[0-9]+)?", requirement) is None:
        _fail("E001", f"unsupported Python requirement {requirement!r}")
    if build_backend != "maturin":
        _fail("E001", f"unsupported build backend {build_backend!r}")
    backend_requires = tuple(
        _text(item, "build requirement") for item in build_requires
    )
    if not backend_requires:
        _fail("E003", "at least one build-system requirement is needed")
    requires = ", ".join(_string_literal(item) for item in backend_requires)
    text = "\n".join(
        [
            "[build-system]",
            f"requires = [{requires}]",
            f"build-backend = {_string_literal(build_backend)}",
            "",
            "[project]",
            f"name = {_string_literal(name)}",
            'version = "0.1.0"',
            f"requires-python = {_string_literal(requirement)}",
            "",
            "[tool.maturin]",
            f"module-name = {_string_literal(name)}",
            "",
        ]
    )
    return {
        "path": "pyproject.toml",
        "text": text,
        "module_name": name,
        "requires_python": requirement,
        "build_requires": backend_requires,
    }


# ---------------------------------------------------------------------------
# 3. rsc_module_header
# ---------------------------------------------------------------------------


def rsc_module_header(
    *,
    module_name: str = "polyglot_native",
    imports: Sequence[str] = ("pyo3::prelude::*", "pyo3::types::PyTuple"),
    allow_unused: bool = True,
) -> dict[str, Any]:
    """Render the stable ``src/lib.rs`` prelude for generated crates."""
    name = _identifier(module_name, "module name")
    selected = tuple(_import_path(item, "Rust import") for item in imports)
    attributes: tuple[str, ...] = ()
    if allow_unused:
        attributes = (
            "#![allow(unused_mut, unused_variables, dead_code, unused_imports)]",
        )
    lines = [
        "// Generated by backends.rust.scaffold; regenerate instead of editing.",
        *attributes,
        "",
        *(f"use {item};" for item in selected),
        "",
    ]
    return {
        "text": "\n".join(lines),
        "module_name": name,
        "imports": selected,
        "attributes": attributes,
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 4. rsc_module_init
# ---------------------------------------------------------------------------


def rsc_module_init(
    module_name: str,
    *,
    functions: Sequence[str] = (),
    classes: Sequence[str] = (),
    error_class: str = "PyPillarError",
) -> dict[str, Any]:
    """Render the ``#[pymodule]`` initializer and typed-error registration."""
    name = _identifier(module_name, "module name")
    function_names = tuple(_identifier(item, "function name") for item in functions)
    class_names = tuple(_identifier(item, "class name") for item in classes)
    error = _identifier(error_class, "error class")
    if len(set(function_names)) != len(function_names):
        _fail("E003", "duplicate function names in module initializer")
    if len(set(class_names)) != len(class_names):
        _fail("E003", "duplicate class names in module initializer")
    lines = [
        "#[pymodule]",
        f"fn {name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{",
        f'    m.add("PillarError", m.py().get_type::<{error}>())?;',
        *(
            f"    m.add_function(wrap_pyfunction!({item}, m)?)?;"
            for item in function_names
        ),
        *(f"    m.add_class::<{item}>()?;" for item in class_names),
        "    Ok(())",
        "}",
        "",
    ]
    return {
        "text": "\n".join(lines),
        "module_name": name,
        "functions": function_names,
        "classes": class_names,
        "error_class": error,
        "registration_count": len(function_names) + len(class_names),
    }


# ---------------------------------------------------------------------------
# 5. rsc_struct_def
# ---------------------------------------------------------------------------


def rsc_struct_def(
    spec: RsStructSpec | Mapping[str, Any],
    *,
    substitutions: Mapping[str, str] | None = None,
    name_override: str | None = None,
) -> dict[str, Any]:
    """Emit one monomorphized ``#[pyclass]`` struct with ordered fields."""
    struct = _coerce_spec(spec, "struct")
    subs = dict(substitutions or {})
    for parameter in struct.generic_params:
        if parameter not in subs:
            _fail(
                "E001",
                f"struct {struct.name!r} has no substitution for {parameter!r}",
            )
    emitted = _identifier(name_override or struct.name, "struct name")
    if struct.generic_params:
        emitted += "".join(
            rsc_generic_suffix(subs[parameter])["suffix"]
            for parameter in struct.generic_params
        )
    emitted = _identifier(emitted, "emitted struct name")
    visibility = _text(struct.visibility, "visibility")
    if visibility not in {"", "pub", "pub(crate)", "pub(super)"}:
        _fail("E001", f"unsupported struct visibility {visibility!r}")
    lines: list[str] = []
    if struct.pyclass:
        lines.append("#[pyclass]")
    if struct.derives:
        lines.append(rsc_derive_line(struct.derives)["text"])
    prefix = f"{visibility} " if visibility else ""
    lines.append(f"{prefix}struct {emitted} {{")
    field_lines: list[dict[str, Any]] = []
    for field in struct.fields:
        local = {**subs, **dict(struct.substitutions)}
        rendered = rsc_field_line(field, pyclass=struct.pyclass, substitutions=local)
        field_lines.append(rendered)
        lines.append(rendered["text"])
    lines.extend(["}", ""])
    return {
        "name": emitted,
        "source_name": struct.name,
        "text": "\n".join(lines),
        "fields": tuple(item["name"] for item in field_lines),
        "rust_types": tuple(item["rust_type"] for item in field_lines),
        "pyclass": struct.pyclass,
        "derives": tuple(struct.derives),
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 6. rsc_enum_def
# ---------------------------------------------------------------------------


def rsc_enum_def(
    spec: RsEnumSpec | Mapping[str, Any],
    *,
    substitutions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Emit one enum definition with deterministic variant order."""
    enum = _coerce_spec(spec, "enum")
    subs = {**dict(substitutions or {}), **dict(enum.substitutions)}
    for parameter in enum.generic_params:
        if parameter not in subs:
            _fail("E001", f"enum {enum.name!r} has no substitution for {parameter!r}")
    emitted = _identifier(enum.name, "enum name")
    if enum.generic_params:
        emitted += "".join(
            rsc_generic_suffix(subs[parameter])["suffix"]
            for parameter in enum.generic_params
        )
    emitted = _identifier(emitted, "emitted enum name")
    lines: list[str] = []
    if enum.pyclass:
        lines.append("#[pyclass(eq, eq_int)]")
    if enum.derives:
        lines.append(rsc_derive_line(enum.derives)["text"])
    visibility = _text(enum.visibility, "visibility")
    if visibility not in {"", "pub", "pub(crate)", "pub(super)"}:
        _fail("E001", f"unsupported enum visibility {visibility!r}")
    prefix = f"{visibility} " if visibility else ""
    lines.append(f"{prefix}enum {emitted} {{")
    variants: list[dict[str, Any]] = []
    for variant in enum.variants:
        name = _identifier(variant.name, "variant name")
        types = tuple(
            rsc_type_substitute(item, subs)["text"] for item in variant.field_types
        )
        declaration = f"    {name}({', '.join(types)})," if types else f"    {name},"
        lines.append(declaration)
        variants.append({"name": name, "field_types": types})
    lines.extend(["}", ""])
    return {
        "name": emitted,
        "source_name": enum.name,
        "text": "\n".join(lines),
        "variants": tuple(variants),
        "pyclass": enum.pyclass,
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# 7. rsc_function_def
# ---------------------------------------------------------------------------


def rsc_function_def(
    spec: RsFunctionSpec | Mapping[str, Any],
    *,
    include_module_registration: bool = False,
) -> dict[str, Any]:
    """Emit one ``#[pyfunction]`` wrapper with typed result conversion."""
    function = _coerce_spec(spec, "function")
    name = _identifier(function.name, "function name")
    for param in function.params:
        rsc_param_decl(param.name, param.type, mutable=param.required)
    result = rsc_return_type(function.result or "()", py_result=True)
    doc = rsc_doc_comment(function.doc)
    attributes = ["#[pyfunction]"]
    lines = [
        *(doc["lines"]),
        *attributes,
        (
            f"fn {name}(py: Python<'_>, args: &Bound<'_, PyTuple>)"
            f" -> {result['text']} {{"
        ),
        "    let __result = {",
        *(
            f"        {line}"
            for line in _balanced(function.body, "function body").splitlines()
        ),
        "    };",
        *_result_conversion_lines(result["error_type"]),
        "}",
        "",
    ]
    registration = ""
    if include_module_registration:
        registration = f"m.add_function(wrap_pyfunction!({name}, m)?)?;"
    return {
        "name": name,
        "text": "\n".join(lines),
        "signature": lines[len(doc["lines"]) + len(attributes)],
        "attributes": tuple(attributes),
        "result_type": result["text"],
        "registration": registration,
    }


# ---------------------------------------------------------------------------
# 8. rsc_method_def
# ---------------------------------------------------------------------------


def rsc_method_def(
    spec: RsFunctionSpec | Mapping[str, Any],
    *,
    receiver: str = "&self",
    indent: str = "    ",
) -> dict[str, Any]:
    """Emit one ``#[pymethods]`` entry for an impl block."""
    method = _coerce_spec(spec, "function")
    name = _identifier(method.name, "method name")
    if receiver not in {"&self", "&mut self", "self", "none"}:
        _fail("E001", f"unsupported method receiver {receiver!r}")
    params = [receiver] if receiver != "none" else []
    params.extend(
        rsc_param_decl(param.name, param.type, mutable=False)["text"]
        for param in method.params
    )
    result = ""
    if method.result:
        result = f" -> {rsc_return_type(method.result, py_result=True)['text']}"
    body = _balanced(method.body, f"body of {name!r}")
    if not body.strip():
        _fail("E003", f"method {name!r} needs a Rust body")
    doc = rsc_doc_comment(method.doc, indent=indent)
    lines = [
        *doc["lines"],
        f"{indent}fn {name}({', '.join(params)}){result} {{",
        *(f"{indent}    {line}" for line in body.splitlines()),
        f"{indent}}}",
    ]
    return {
        "name": name,
        "text": "\n".join(lines),
        "receiver": receiver,
        "signature": f"fn {name}({', '.join(params)}){result}",
        "method_attrs": ("#[pymethods]",),
    }


# ---------------------------------------------------------------------------
# 9. rsc_trait_impl
# ---------------------------------------------------------------------------


def rsc_trait_impl(
    target: str,
    *,
    methods: Sequence[RsFunctionSpec | Mapping[str, Any]] = (),
    trait: str | None = None,
    bounds: Sequence[str] = (),
    generics: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit an inherent or trait impl block with validated where bounds."""
    target_name = _type_path(_type_text(target, "impl target"), "impl target")
    trait_name = (
        _type_path(_type_text(trait, "trait name"), "trait name") if trait else None
    )
    generic_names = tuple(_identifier(item, "generic parameter") for item in generics)
    checked_bounds: list[str] = []
    for bound in bounds:
        text = " ".join(_text(bound, "where bound").split())
        if ":" not in text or any(char in text for char in "{};"):
            _fail("E003", f"malformed where bound {bound!r}")
        head, tail = text.split(":", 1)
        checked_bounds.append(
            f"{_type_path(head.strip(), 'bound type')}: {tail.strip()}"
        )
    header = "impl"
    if generic_names:
        header += f"<{', '.join(generic_names)}>"
    header += f" {trait_name} for {target_name}" if trait_name else f" {target_name}"
    lines = [header if checked_bounds else f"{header} {{"]
    if checked_bounds:
        lines.append("where")
        lines.extend(f"    {bound}," for bound in checked_bounds)
        lines.append("{")
    emitted_methods = tuple(rsc_method_def(method) for method in methods)
    for method in emitted_methods:
        lines.extend(method["text"].splitlines())
    lines.extend(["}", ""])
    return {
        "text": "\n".join(lines),
        "target": target_name,
        "trait": trait_name,
        "bounds": tuple(checked_bounds),
        "generics": generic_names,
        "methods": tuple(item["name"] for item in emitted_methods),
    }


# ---------------------------------------------------------------------------
# 10. rsc_derive_line
# ---------------------------------------------------------------------------


def rsc_derive_line(derives: Sequence[str] = ()) -> dict[str, Any]:
    """Render a ``#[derive(...)]`` attribute with deterministic ordering."""
    items = tuple(_identifier(item, "derive name") for item in derives)
    if len(set(items)) != len(items):
        _fail("E003", "duplicate derive names")
    return {
        "text": f"#[derive({', '.join(items)})]" if items else "",
        "derives": items,
        "count": len(items),
    }


# ---------------------------------------------------------------------------
# 11. rsc_field_line
# ---------------------------------------------------------------------------


def rsc_field_line(
    field: RsFieldSpec | Mapping[str, Any],
    *,
    pyclass: bool = True,
    substitutions: Mapping[str, str] | None = None,
    indent: str = "    ",
) -> dict[str, Any]:
    """Render one struct field, including PyO3 accessors when requested."""
    spec = _coerce_spec(field, "field")
    name = _identifier(spec.name, "field name")
    rust_type = rsc_type_substitute(spec.type, substitutions or {})["text"]
    visibility = _text(spec.visibility, "field visibility")
    if visibility not in {"", "pub", "pub(crate)", "pub(super)"}:
        _fail("E001", f"unsupported field visibility {visibility!r}")
    python_access = bool(spec.python_access and pyclass)
    lines: list[str] = []
    if python_access:
        lines.append(f"{indent}#[pyo3(get, set)]")
    prefix = f"{visibility} " if visibility else ""
    lines.append(f"{indent}{prefix}{name}: {rust_type},")
    return {
        "text": "\n".join(lines),
        "name": name,
        "rust_type": rust_type,
        "visibility": visibility,
        "python_access": python_access,
    }


# ---------------------------------------------------------------------------
# 12. rsc_param_decl
# ---------------------------------------------------------------------------


def rsc_param_decl(
    name: str,
    rust_type: str,
    *,
    receiver: bool = False,
    mutable: bool = False,
) -> dict[str, Any]:
    """Render one Rust parameter declaration for a generated function."""
    if receiver:
        text = _text(rust_type, "receiver type")
        if text not in {"&self", "&mut self", "self", "Py<Self>"}:
            _fail("E001", f"unsupported receiver type {text!r}")
        return {"text": text, "name": "self", "rust_type": text, "receiver": True}
    identifier = _identifier(name, "parameter name")
    canonical = rsc_type_substitute(rust_type, {})["text"]
    prefix = "mut " if mutable else ""
    return {
        "text": f"{prefix}{identifier}: {canonical}",
        "name": identifier,
        "rust_type": canonical,
        "receiver": False,
        "mutable": mutable,
    }


# ---------------------------------------------------------------------------
# 13. rsc_return_type
# ---------------------------------------------------------------------------


def rsc_return_type(rust_type: str, *, py_result: bool = True) -> dict[str, Any]:
    """Compute the PyO3 return type for a Rust result specification."""
    base, args = _named_type(rust_type)
    if base == "Result":
        if len(args) != 2:
            _fail("E001", "Result requires exactly two type arguments")
        inner, error = args
    else:
        inner, error = _type_text(rust_type, "Rust return type"), None
    text = f"PyResult<{inner}>" if py_result else inner
    return {
        "text": text,
        "inner": inner,
        "error_type": error,
        "py_result": py_result,
    }


# ---------------------------------------------------------------------------
# 14. rsc_doc_comment
# ---------------------------------------------------------------------------


def rsc_doc_comment(doc: str, *, indent: str = "") -> dict[str, Any]:
    """Render documentation into stable Rust doc-comment lines."""
    prefix = _text(indent, "indent")
    if any(char not in " \t" for char in prefix):
        _fail("E003", "indent may contain only spaces and tabs")
    text, lines = _render_doc(doc, prefix)
    return {"text": text, "lines": lines, "line_count": len(lines)}


# ---------------------------------------------------------------------------
# 15. rsc_generic_suffix
# ---------------------------------------------------------------------------


def rsc_generic_suffix(rust_type: str) -> dict[str, Any]:
    """Mangle a concrete type into a deterministic monomorphization suffix."""
    canonical = _type_text(rust_type, "Rust type")
    suffix_parts: list[str] = []
    for token in _TOKEN_RE.findall(canonical):
        if not token[0].isalnum():
            continue
        suffix_parts.append(token[0].upper() + token[1:])
    suffix = "".join(suffix_parts)
    if not suffix:
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return {"suffix": suffix, "rust_type": canonical, "length": len(suffix)}


# ---------------------------------------------------------------------------
# 16. rsc_type_substitute
# ---------------------------------------------------------------------------


def rsc_type_substitute(
    rust_type: str,
    substitutions: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Substitute generic parameters inside a Rust type expression."""
    text, unresolved = _substitute(rust_type, substitutions or {})
    if strict and unresolved:
        _fail(
            "E001",
            f"unsubstituted type parameter(s) in {rust_type!r}: {', '.join(unresolved)}",
        )
    return {
        "text": text,
        "source": _type_text(rust_type, "Rust type"),
        "substitutions": dict(substitutions or {}),
        "unresolved": unresolved,
    }


# ---------------------------------------------------------------------------
# 17. rsc_layout_plan
# ---------------------------------------------------------------------------


def rsc_layout_plan(
    structs: Sequence[RsStructSpec | Mapping[str, Any]] = (),
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Order type definitions so dependencies are emitted before their users."""
    struct_specs = _spec_list(structs, "struct")
    enum_specs = _spec_list(enums, "enum")
    nodes: dict[str, str] = {}
    edges: dict[str, tuple[str, ...]] = {}
    for spec in struct_specs:
        name = _identifier(spec.name, "struct name")
        nodes[name] = "struct"
        dependencies = set()
        known_types = {item.name for item in struct_specs} | {
            item.name for item in enum_specs
        }
        for field in spec.fields:
            base = _base_type_name(field.type)
            if base in known_types:
                dependencies.add(base)
        edges[name] = tuple(sorted(dependencies))
    for spec in enum_specs:
        name = _identifier(spec.name, "enum name")
        nodes[name] = "enum"
        dependencies = set()
        for variant in spec.variants:
            for field_type in variant.field_types:
                base = _base_type_name(field_type)
                if base in nodes:
                    dependencies.add(base)
        edges[name] = tuple(sorted(dependencies))
    if len(nodes) != len(struct_specs) + len(enum_specs):
        _fail("E003", "duplicate type names in layout plan")
    ordered: list[str] = []
    remaining = set(nodes)
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(dep not in remaining for dep in edges[name])
        )
        if not ready:
            _fail("E001", f"cyclic type layout among {sorted(remaining)}")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return {
        "order": tuple((name, nodes[name]) for name in ordered),
        "names": tuple(ordered),
        "edges": edges,
        "types": dict(nodes),
        "levels": len(
            tuple(
                1
                for index, name in enumerate(ordered)
                if index == 0 or edges[name] != edges[ordered[index - 1]]
            )
        ),
    }


# ---------------------------------------------------------------------------
# 18. rsc_scaffold_plan
# ---------------------------------------------------------------------------


def rsc_scaffold_plan(
    *,
    module_name: str = "polyglot_native",
    functions: Sequence[RsFunctionSpec | Mapping[str, Any]] = (),
    structs: Sequence[RsStructSpec | Mapping[str, Any]] = (),
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
    impls: Sequence[RsTraitImplSpec | Mapping[str, Any]] = (),
    substitutions: Mapping[str, str] | None = None,
    edition: str = "2021",
    pyo3_version: str = "0.26",
) -> dict[str, Any]:
    """Normalize metadata into a deterministic scaffold emission plan."""
    name = _identifier(module_name, "module name")
    if edition not in _EDITIONS:
        _fail("E001", f"unsupported Rust edition {edition!r}")
    function_specs = _spec_list(functions, "function")
    struct_specs = _spec_list(structs, "struct")
    enum_specs = _spec_list(enums, "enum")
    impl_specs = _spec_list(impls, "impl")
    layout = rsc_layout_plan(struct_specs, enum_specs)
    rendered_structs = tuple(
        rsc_struct_def(spec, substitutions=substitutions) for spec in struct_specs
    )
    rendered_enums = tuple(
        rsc_enum_def(spec, substitutions=substitutions) for spec in enum_specs
    )
    rendered_functions = tuple(rsc_function_def(spec) for spec in function_specs)
    rendered_impls = tuple(
        rsc_trait_impl(
            spec.target,
            methods=spec.methods,
            trait=spec.trait,
            bounds=spec.bounds,
            generics=spec.generic_params,
        )
        for spec in impl_specs
    )
    return {
        "module_name": name,
        "edition": edition,
        "pyo3_version": pyo3_version,
        "substitutions": dict(substitutions or {}),
        "layout": layout,
        "structs": rendered_structs,
        "enums": rendered_enums,
        "functions": rendered_functions,
        "impls": rendered_impls,
        "function_names": tuple(item["name"] for item in rendered_functions),
        "class_names": tuple(item["name"] for item in rendered_structs),
    }


# ---------------------------------------------------------------------------
# 19. rsc_scaffold_render
# ---------------------------------------------------------------------------


def rsc_scaffold_render(
    plan: Mapping[str, Any],
    *,
    include_guards: bool = False,
) -> dict[str, Any]:
    """Render the ``src/lib.rs`` text of a previously built scaffold plan."""
    payload = _as_mapping(plan, "scaffold plan")
    for key in ("module_name", "structs", "enums", "functions", "impls"):
        if key not in payload:
            _fail("E008", f"scaffold plan is missing {key!r}")
    header = rsc_module_header(module_name=str(payload["module_name"]))
    sections: dict[str, str] = {"header": header["text"]}
    if include_guards:
        sections["guards"] = (
            "// RAII guards own Py<T> handles and release on every return path.\n"
            "struct PyO3OwnedGuard { handle: Option<Py<PyAny>> }\n"
            "impl Drop for PyO3OwnedGuard {\n"
            "    fn drop(&mut self) {\n"
            "        self.handle.take();\n"
            "    }\n"
            "}\n"
        )
    sections["structs"] = "\n".join(str(item["text"]) for item in payload["structs"])
    sections["enums"] = "\n".join(str(item["text"]) for item in payload["enums"])
    sections["impls"] = "\n".join(str(item["text"]) for item in payload["impls"])
    sections["functions"] = "\n".join(
        str(item["text"]) for item in payload["functions"]
    )
    init = rsc_module_init(
        str(payload["module_name"]),
        functions=tuple(str(item["name"]) for item in payload["functions"]),
        classes=tuple(str(item["name"]) for item in payload["structs"]),
    )
    sections["module_init"] = init["text"]
    ordered_sections = (
        "header",
        "guards",
        "structs",
        "enums",
        "impls",
        "functions",
        "module_init",
    )
    text = "\n".join(sections[key] for key in ordered_sections if key in sections)
    return {
        "path": "src/lib.rs",
        "text": text,
        "sections": sections,
        "section_order": tuple(key for key in ordered_sections if key in sections),
        "line_count": text.count("\n") + 1,
    }


# ---------------------------------------------------------------------------
# 20. rsc_scaffold_fingerprint
# ---------------------------------------------------------------------------


def rsc_scaffold_fingerprint(files: Mapping[str, str]) -> dict[str, Any]:
    """Hash a scaffold file mapping in a path-sorted canonical form."""
    payload = _as_mapping(files, "scaffold files")
    paths = tuple(sorted(payload))
    canonical = "\n".join(
        f"{path}\0{_text(payload[path], f'contents of {path}')}" for path in paths
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "algorithm": "sha256",
        "digest": digest,
        "paths": paths,
        "byte_length": len(canonical.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# 21. rsc_scaffold_diff
# ---------------------------------------------------------------------------


def rsc_scaffold_diff(
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> dict[str, Any]:
    """Compare two scaffold mappings without touching the filesystem."""
    old = _as_mapping(left, "left scaffold")
    new = _as_mapping(right, "right scaffold")
    for label, mapping in (("left", old), ("right", new)):
        for path, text in mapping.items():
            _text(path, f"{label} path")
            _text(text, f"{label} contents of {path}")
    old_paths, new_paths = set(old), set(new)
    added = tuple(sorted(new_paths - old_paths))
    removed = tuple(sorted(old_paths - new_paths))
    changed: list[dict[str, Any]] = []
    for path in sorted(old_paths & new_paths):
        if old[path] == new[path]:
            continue
        old_lines = old[path].splitlines()
        new_lines = new[path].splitlines()
        first = next(
            (
                index + 1
                for index, (before, after) in enumerate(zip(old_lines, new_lines))
                if before != after
            ),
            min(len(old_lines), len(new_lines)) + 1,
        )
        changed.append(
            {
                "path": path,
                "first_difference": first,
                "left_lines": len(old_lines),
                "right_lines": len(new_lines),
            }
        )
    return {
        "added": added,
        "removed": removed,
        "changed": tuple(changed),
        "identical": not (added or removed or changed),
        "comparisons": len(old_paths & new_paths),
    }


# ---------------------------------------------------------------------------
# 22. rsc_feature_flags
# ---------------------------------------------------------------------------


def rsc_feature_flags(
    features: Mapping[str, Sequence[str]] | None = None,
    *,
    default: Sequence[str] = (),
) -> dict[str, Any]:
    """Render a Cargo ``[features]`` table from feature-to-dependency maps."""
    table = _as_mapping(features or {}, "features")
    defaults = tuple(_feature_name(item, "default feature") for item in default)
    rendered: list[str] = []
    if defaults:
        rendered.append(
            f"default = [{', '.join(_string_literal(item) for item in defaults)}]"
        )
    names: list[str] = []
    for key in sorted(table):
        name = _feature_name(key, "feature name")
        names.append(name)
        values = tuple(_text(item, f"member of feature {name}") for item in table[key])
        rendered.append(
            f"{name} = [{', '.join(_string_literal(item) for item in values)}]"
        )
    text = "\n".join(["[features]", *rendered, ""]) if rendered else "[features]\n"
    return {
        "text": text,
        "features": tuple(names),
        "default": defaults,
        "entry_count": len(rendered),
    }


# ---------------------------------------------------------------------------
# 23. rsc_abi_settings
# ---------------------------------------------------------------------------


def rsc_abi_settings(
    *,
    pyo3_version: str = "0.26",
    abi3: bool = False,
    python_version: str = "3.14",
    platform: str | None = None,
) -> dict[str, Any]:
    """Validate a PyO3 ABI configuration and report the required features."""
    version = _text(pyo3_version, "PyO3 version")
    if _VERSION_RE.fullmatch(version) is None:
        _fail("E008", f"malformed PyO3 version {version!r}")
    if re.fullmatch(r"[0-9]+\.[0-9]+", python_version) is None:
        _fail("E001", f"unsupported Python ABI version {python_version!r}")
    major, minor = (int(part) for part in python_version.split("."))
    if (major, minor) < (3, 8):
        _fail("E001", f"PyO3 abi3 requires Python >= 3.8, got {python_version!r}")
    features = ["extension-module"]
    if abi3:
        features.append(f"abi3-py{major}{minor}")
    if platform is not None:
        name = _identifier(platform, "platform")
        if name not in {"linux", "macos", "windows"}:
            _fail("E001", f"unsupported native platform {platform!r}")
    else:
        name = None
    return {
        "pyo3_version": version,
        "python_version": python_version,
        "abi3": abi3,
        "platform": name,
        "features": tuple(sorted(features)),
        "manifest_line": (
            "pyo3 = { version = "
            + _string_literal(version)
            + ", features = ["
            + ", ".join(_string_literal(item) for item in sorted(features))
            + "] }"
        ),
    }


# ---------------------------------------------------------------------------
# 24. rsc_crate_layout
# ---------------------------------------------------------------------------


def rsc_crate_layout(
    files: Mapping[str, str],
    *,
    required: Sequence[str] = ("Cargo.toml", "pyproject.toml", "src/lib.rs"),
) -> dict[str, Any]:
    """Validate that a scaffold mapping contains the expected crate layout."""
    payload = _as_mapping(files, "crate files")
    required_paths = tuple(_text(item, "required path") for item in required)
    present = tuple(sorted(payload))
    missing = tuple(path for path in required_paths if path not in payload)
    if missing:
        _fail("E001", f"crate layout is missing {', '.join(missing)}")
    extra = tuple(
        path
        for path in present
        if path not in required_paths and not path.startswith("src/")
    )
    if extra:
        _fail("E001", f"crate layout has unexpected files: {', '.join(extra)}")
    for path in present:
        _text(payload[path], f"contents of {path}")
    return {
        "ok": True,
        "present": present,
        "required": required_paths,
        "missing": missing,
        "extra": extra,
        "file_count": len(present),
    }


# ---------------------------------------------------------------------------
# 25. rsc_scaffold_manifest
# ---------------------------------------------------------------------------


def rsc_scaffold_manifest(
    *,
    module_name: str = "polyglot_native",
    pyo3_version: str = "0.26",
    edition: str = "2021",
    functions: Sequence[RsFunctionSpec | Mapping[str, Any]] = (),
    structs: Sequence[RsStructSpec | Mapping[str, Any]] = (),
    enums: Sequence[RsEnumSpec | Mapping[str, Any]] = (),
    impls: Sequence[RsTraitImplSpec | Mapping[str, Any]] = (),
    substitutions: Mapping[str, str] | None = None,
    include_guards: bool = False,
) -> dict[str, Any]:
    """Assemble the three generated crate files and verify their layout."""
    plan = rsc_scaffold_plan(
        module_name=module_name,
        functions=functions,
        structs=structs,
        enums=enums,
        impls=impls,
        substitutions=substitutions,
        edition=edition,
        pyo3_version=pyo3_version,
    )
    manifest = rsc_crate_manifest(
        module_name=module_name, pyo3_version=pyo3_version, edition=edition
    )
    pyproject = rsc_pyproject_manifest(module_name=module_name)
    lib = rsc_scaffold_render(plan, include_guards=include_guards)
    files = {
        str(manifest["path"]): str(manifest["text"]),
        str(pyproject["path"]): str(pyproject["text"]),
        str(lib["path"]): str(lib["text"]),
    }
    layout = rsc_crate_layout(files)
    fingerprint = rsc_scaffold_fingerprint(files)
    return {
        "files": files,
        "layout": layout,
        "fingerprint": fingerprint["digest"],
        "module_name": plan["module_name"],
        "function_names": plan["function_names"],
        "class_names": plan["class_names"],
        "sections": lib["sections"],
    }
