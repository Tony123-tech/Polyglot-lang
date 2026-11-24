"""FRAME 4: Python backend emission and CPython interop helpers.

The functions in this module are the canonical implementations for framework
functions 31-40. They are deliberately small, deterministic building blocks:

* emitters produce Python *source* that composes with
  :func:`py_emit_module_header`;
* interop helpers bind optional native extensions and normalise their results;
* the parity harness compiles trusted Polyglot source through the regular
  Python backend and compares it with reference or native callables.

No new third-party dependencies are introduced. Runtime helpers are reused from
``backends.python.translator`` so emitted code has the same semantics as the
main backend.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import FrameType, MappingProxyType, ModuleType, TracebackType
from typing import Any

from api.visualizer import program_to_source
from backends import get_backend
from backends.python.translator import (
    Translator,
    _Option,
    _Result,
    _ResultErrSentinel,
)
from compiler.core.ast_nodes import (
    EnumDecl,
    FnDecl,
    ImplBlock,
    MacroCall,
    MatchExpr,
    PatIdent,
    PatRef,
    Program,
    StructDecl,
    TraitDecl,
    Type,
    TypeDynTrait,
    TypeImplTrait,
    TypeName,
    TypeRef,
)
from compiler.core.lexer import lex
from compiler.core.parser import parse

__all__ = [
    "FrameTraceGuard",
    "FrameTraceSnapshot",
    "MarshalledResult",
    "NativeBinding",
    "ParityCase",
    "ParityReport",
    "py_bind_native_extension_module",
    "py_emit_macro_expansions",
    "py_emit_match_lowering",
    "py_emit_module_header",
    "py_emit_struct_and_enum_classes",
    "py_emit_trait_dispatch_adapters",
    "py_format_with_ruff_compatible_style",
    "py_guard_frame_and_trace_state",
    "py_marshal_result_and_errors",
    "py_run_parity_harness",
]

# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------

_CORE_HELPERS: tuple[str, ...] = (
    "_Option",
    "_PythonEnumVariant",
    "_Result",
    "_ResultErrSentinel",
    "_assert",
    "_assert_eq",
    "_assert_ne",
    "_panic",
    "_runtime_non_exhaustive_match",
    "_div",
    "_mod",
    "_cast_i8",
    "_cast_i16",
    "_cast_i32",
    "_cast_i64",
    "_cast_i128",
    "_cast_u8",
    "_cast_u16",
    "_cast_u32",
    "_cast_u64",
    "_cast_u128",
)
_WIDTH_HELPERS: tuple[str, ...] = ("_cast", "_cast_isize", "_cast_usize")
_RUNTIME_HELPERS: tuple[str, ...] = _CORE_HELPERS + _WIDTH_HELPERS
_DEFAULT_HELPERS: tuple[str, ...] = _RUNTIME_HELPERS

# The translator's namespace aliases (``_assert`` etc.) are installed in
# compiled modules; generated source imports the underlying functions directly.
_RUNTIME_HELPER_SOURCES: dict[str, str] = {name: name for name in _CORE_HELPERS} | {
    "_assert": "_runtime_assert",
    "_assert_eq": "_runtime_assert_eq",
    "_assert_ne": "_runtime_assert_ne",
    "_panic": "_runtime_panic",
    "_div": "_runtime_int_div",
    "_mod": "_runtime_int_rem",
    "_cast_i8": "cast_i8",
    "_cast_i16": "cast_i16",
    "_cast_i32": "cast_i32",
    "_cast_i64": "cast_i64",
    "_cast_i128": "cast_i128",
    "_cast_u8": "cast_u8",
    "_cast_u16": "cast_u16",
    "_cast_u32": "cast_u32",
    "_cast_u64": "cast_u64",
    "_cast_u128": "cast_u128",
}

_SIMPLE_SINGLE_QUOTED = re.compile(
    r"(?<![A-Za-z0-9_])((?:r|b|br|rb)?)'((?:[^'\\\n]|\\.)*)'(?![A-Za-z0-9_])"
)
_FROM_IMPORT = re.compile(r"from\s+([\w.]+)\s+import\s+(.+)")
_TOP_LEVEL_DEF = ("class ", "def ", "async def ")


def _identifier(value: str, label: str) -> str:
    """Validate a Python identifier and return it unchanged."""
    if not value.isidentifier():
        raise ValueError(f"{label} must be a valid identifier, got {value!r}")
    return value


def _safe_name(name: str) -> str:
    """Map a source-level name to a legal Python attribute name."""
    candidate = name if name.isidentifier() else re.sub(r"\W", "_", name)
    if not candidate or not candidate.isidentifier() or candidate[0].isdigit():
        candidate = f"_{candidate}"
    from keyword import iskeyword

    return f"{candidate}_" if iskeyword(candidate) else candidate


def _string_literal(value: str) -> str:
    """Render a string literal with ruff's preferred double quotes."""
    return json.dumps(value)


def _string_tuple(values: Sequence[str]) -> str:
    """Render a tuple of strings with deterministic double-quoted literals."""
    items = ", ".join(_string_literal(value) for value in values)
    comma = "," if len(values) == 1 else ""
    return f"({items}{comma})"


def _import_bindings(statement: str) -> tuple[str, ...]:
    """Return the local names bound by a single import statement."""
    try:
        node = ast.parse(statement).body[0]
    except SyntaxError as error:
        raise ValueError(f"invalid import statement: {statement!r}") from error
    if isinstance(node, ast.Import):
        return tuple(
            alias.asname or alias.name.split(".", 1)[0] for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        return tuple(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )
    return ()


def _indent(source: str, spaces: int) -> str:
    """Indent every non-empty line of ``source`` by ``spaces`` characters."""
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in source.splitlines())


def _type_name(ty: Type | None) -> str:
    """Return the bare name of a named/reference type."""
    if isinstance(ty, TypeName):
        return ty.name
    if isinstance(ty, TypeRef):
        return _type_name(ty.inner)
    if isinstance(ty, TypeDynTrait | TypeImplTrait):
        return ty.bounds[0].name if ty.bounds else ""
    return ""


def _has_self_param(fn: FnDecl) -> bool:
    """Return True when the first parameter is ``self`` (optionally borrowed)."""
    if not fn.params:
        return False
    pattern = fn.params[0].pattern
    if isinstance(pattern, PatRef):
        pattern = pattern.inner
    return isinstance(pattern, PatIdent) and pattern.name == "self"


def _prepared_translator(
    declarations: Sequence[StructDecl | EnumDecl] = (),
    impls: Sequence[ImplBlock] = (),
) -> Translator:
    """Build a translator primed with the declarations we are emitting."""
    translator = Translator()
    for declaration in declarations:
        translator.generic_type_names.update(
            param.name for param in getattr(declaration, "generics", ())
        )
    for declaration in declarations:
        if isinstance(declaration, StructDecl):
            translator.struct_names.add(declaration.name)
            translator.struct_fields[declaration.name] = {
                name for name, _ty, _pub in declaration.fields
            }
            translator.struct_field_types[declaration.name] = {
                name: _ty.name
                for name, _ty, _pub in declaration.fields
                if isinstance(_ty, TypeName)
            }
        elif isinstance(declaration, EnumDecl):
            translator.enum_names.add(declaration.name)
            translator.enum_variants[declaration.name] = {
                variant.name for variant in declaration.variants
            }
    for impl in impls:
        translator.impl_blocks.setdefault(_type_name(impl.ty), []).append(impl)
    return translator


def _annotation(translator: Translator, ty: Type | None) -> str:
    """Render a Polyglot type as a Python annotation string."""
    rendered = translator.translate_type(ty)
    if rendered is None:
        return "object"
    return ast.unparse(ast.fix_missing_locations(rendered))


def _method_sources(
    translator: Translator,
    impl: ImplBlock,
    *,
    indent: int = 4,
) -> list[str]:
    """Translate the function items of an impl block into indented source."""
    sources: list[str] = []
    for item in impl.items:
        if not isinstance(item, FnDecl):
            continue
        node = translator.translate_fn(item, is_method=_has_self_param(item))
        sources.append(_indent(ast.unparse(ast.fix_missing_locations(node)), indent))
    return sources


def _package_version() -> str:
    """Return the installed package version, with a deterministic fallback."""
    try:
        return importlib.metadata.version("polyglot-lang")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


# ---------------------------------------------------------------------------
# 31. Module header
# ---------------------------------------------------------------------------


def py_emit_module_header(
    *,
    module_name: str = "_polyglot_generated",
    version: str | None = None,
    target_width: int = 64,
    imports: Sequence[str] = (),
    helpers: Sequence[str] = (),
) -> str:
    """Emit a deterministic Python preamble for generated Polyglot modules.

    The preamble imports ``dataclass``/``Enum`` support, selects the runtime
    helpers from :mod:`backends.python.translator`, and records version
    metadata in ``__polyglot__``. Extra imports are limited to plain
    single-line ``import``/``from`` statements to keep generated code safe.

    Args:
        module_name: Logical name stored in the metadata dictionary.
        version: Package version to report; defaults to the installed
            ``polyglot-lang`` distribution or ``"0.0.0"``.
        target_width: Simulated pointer width; must be 32 or 64.
        imports: Additional single-line import statements.
        helpers: Runtime helper names to import from the translator.

    Returns:
        Valid, deterministic Python source ending in a newline.

    Raises:
        ValueError: If a name, width, import, or helper is invalid.
    """
    _identifier(module_name, "module_name")
    if target_width not in (32, 64):
        raise ValueError(f"target_width must be 32 or 64, got {target_width}")
    selected = tuple(helpers) if helpers else _DEFAULT_HELPERS
    unknown = sorted(set(selected) - set(_RUNTIME_HELPERS))
    if unknown:
        raise ValueError(f"unknown runtime helpers: {', '.join(unknown)}")
    extra_imports: list[str] = []
    extra_bound_names: list[str] = []
    for statement in imports:
        stripped = statement.strip()
        if (
            "\n" in stripped
            or ";" in stripped
            or not stripped.startswith(("import ", "from "))
        ):
            raise ValueError(f"unsupported import statement: {statement!r}")
        extra_imports.append(stripped)
        extra_bound_names.extend(_import_bindings(stripped))
    selected_names = sorted(set(selected))
    direct_imports = [
        f"    {_RUNTIME_HELPER_SOURCES[name]} as {name}"
        for name in selected_names
        if name not in _WIDTH_HELPERS
    ]
    width_imports = [
        "    _runtime_cast as _polyglot_runtime_cast",
        "    cast_isize as _polyglot_cast_isize",
        "    cast_usize as _polyglot_cast_usize",
    ]
    import_entries = direct_imports + width_imports
    resolved_version = _package_version() if version is None else version
    all_names = [
        *selected_names,
        "_cast",
        "_cast_isize",
        "_cast_usize",
        "_polyglot_runtime_cast",
        "_polyglot_cast_isize",
        "_polyglot_cast_usize",
        "dataclass",
        "Enum",
        "auto",
        "Any",
        "ClassVar",
        "Protocol",
        "runtime_checkable",
        *extra_bound_names,
    ]
    lines = [
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from enum import Enum, auto",
        "from typing import Any, ClassVar, Protocol, runtime_checkable",
        "from backends.python.translator import (",
        *[f"{entry}," for entry in import_entries[:-1]],
        f"{import_entries[-1]},",
        ")",
    ]
    lines.extend(extra_imports)
    lines.extend(
        [
            "",
            "",
            "def _cast(value: object, ty: str) -> object:",
            '    """Cast with the target width selected for this module."""',
            f"    return _polyglot_runtime_cast(value, ty, target_width={target_width})",
            "",
            "",
            "def _cast_isize(value: object) -> object:",
            '    """Cast a pointer-sized signed integer."""',
            f"    return _polyglot_cast_isize(value, target_width={target_width})",
            "",
            "",
            "def _cast_usize(value: object) -> object:",
            '    """Cast a pointer-sized unsigned integer."""',
            f"    return _polyglot_cast_usize(value, target_width={target_width})",
            "",
            "",
            "__polyglot__: dict[str, Any] = {",
            '    "backend": "python",',
            f'    "module": {json.dumps(module_name)},',
            f'    "target_width": {target_width},',
            f'    "version": {json.dumps(resolved_version)},',
            "}",
            "",
            "__all__ = [",
            *[f"    {_string_literal(name)}," for name in sorted(set(all_names))],
            "]",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 32. Structs and enums
# ---------------------------------------------------------------------------


def _emit_struct(
    translator: Translator,
    declaration: StructDecl,
    impls: Sequence[ImplBlock],
    *,
    include_impls: bool,
) -> str:
    """Render one struct as a slotted dataclass with optional impl methods."""
    lines = [
        "@dataclass(slots=True)",
        f"class {_safe_name(declaration.name)}:",
        f'    """Slotted dataclass generated for {declaration.name}."""',
    ]
    for name, ty, _pub in declaration.fields:
        lines.append(f"    {_safe_name(name)}: {_annotation(translator, ty)}")
    if include_impls:
        for impl in impls:
            if _type_name(impl.ty) == declaration.name:
                lines.extend(_method_sources(translator, impl))
    return "\n".join(lines)


def _emit_enum(
    translator: Translator,
    declaration: EnumDecl,
    impls: Sequence[ImplBlock],
    *,
    include_impls: bool,
) -> str:
    """Render one enum as ``Enum`` or as a tuple-tagged union."""
    name = _safe_name(declaration.name)
    has_payloads = any(variant.fields for variant in declaration.variants)
    if not has_payloads:
        lines = [
            f"class {name}(Enum):",
            f'    """Enum generated for {declaration.name}."""',
        ]
        for variant in declaration.variants:
            lines.append(f"    {_safe_name(variant.name)} = auto()")
    else:
        lines = [
            "@dataclass(frozen=True, slots=True)",
            f"class {name}:",
            '    """Tagged union; payload variants use ``(tag, *payload)``."""',
            "    tag: str",
            "    payload: tuple[object, ...] = ()",
            "",
        ]
        for variant in declaration.variants:
            attr = _safe_name(variant.name)
            lines.append(
                f"    {attr} = _PythonEnumVariant({_string_literal(variant.name)})"
            )
        lines.extend(
            [
                "",
                "    @classmethod",
                f"    def decode(cls, value: object) -> {name}:",
                "        if isinstance(value, tuple) and value:",
                '            tag = getattr(value[0], "name", str(value[0]))',
                "            return cls(str(tag), tuple(value[1:]))",
                '        tag = getattr(value, "name", str(value))',
                "        return cls(str(tag))",
                "",
            ]
        )
        for variant in declaration.variants:
            attr = _safe_name(variant.name)
            payload_types = variant.fields or ()
            params = ", ".join(
                f"value{index}: {_annotation(translator, ty)}"
                for index, ty in enumerate(payload_types)
            )
            values = ", ".join(f"value{index}" for index in range(len(payload_types)))
            if payload_types:
                lines.extend(
                    [
                        "    @staticmethod",
                        f"    def make_{attr}({params}) -> tuple[object, ...]:",
                        f"        return ({name}.{attr}, {values})",
                        "",
                        "    @staticmethod",
                        f"    def is_{attr}(value: object) -> bool:",
                        "        return (",
                        "            isinstance(value, tuple)",
                        f"            and len(value) == {len(payload_types) + 1}",
                        f"            and value[0] is {name}.{attr}",
                        "        )",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "    @staticmethod",
                        f"    def make_{attr}() -> _PythonEnumVariant:",
                        f"        return {name}.{attr}",
                        "",
                        "    @staticmethod",
                        f"    def is_{attr}(value: object) -> bool:",
                        f"        return value is {name}.{attr}",
                        "",
                    ]
                )
    if include_impls:
        for impl in impls:
            if _type_name(impl.ty) == declaration.name:
                lines.extend(_method_sources(translator, impl))
        if not lines[-1]:
            lines.pop()
    return "\n".join(lines)


def py_emit_struct_and_enum_classes(
    declarations: Sequence[StructDecl | EnumDecl],
    *,
    impls: Sequence[ImplBlock] = (),
    include_impls: bool = True,
) -> str:
    """Generate slotted structs and runtime-compatible enum representations.

    Structs become ``@dataclass(slots=True)`` classes. Enums containing only
    unit variants become ``Enum`` subclasses; enums with payloads become frozen
    dataclasses whose variant constructors return the same ``(tag, *payload)``
    tuples used by the main Python translator.

    Args:
        declarations: Struct and enum declarations from the Polyglot AST.
        impls: Impl blocks whose methods should be attached to those classes.
        include_impls: Set False to omit impl methods.

    Returns:
        Python class definitions separated by blank lines.

    Raises:
        TypeError: If a declaration is neither a struct nor an enum.
    """
    translator = _prepared_translator(declarations, impls)
    chunks: list[str] = []
    for declaration in declarations:
        if isinstance(declaration, StructDecl):
            chunks.append(
                _emit_struct(
                    translator, declaration, impls, include_impls=include_impls
                )
            )
        elif isinstance(declaration, EnumDecl):
            chunks.append(
                _emit_enum(translator, declaration, impls, include_impls=include_impls)
            )
        else:
            raise TypeError(
                f"unsupported declaration type: {type(declaration).__name__}"
            )
    return "\n\n\n".join(chunks) + ("\n" if chunks else "")


# ---------------------------------------------------------------------------
# 33. Trait dispatch
# ---------------------------------------------------------------------------


def _trait_method_names(trait: TraitDecl) -> tuple[str, ...]:
    """List the method names declared by a trait, in source order."""
    return tuple(
        item.fn.name
        for item in trait.items
        if item.fn is not None and item.fn.name and item.fn.name not in ("",)
    )


def py_emit_trait_dispatch_adapters(
    traits: Sequence[TraitDecl],
    impls: Sequence[ImplBlock] = (),
) -> str:
    """Generate protocols and dispatch tables aligned with trait bounds.

    Each trait becomes a ``runtime_checkable`` ``Protocol`` carrying its method
    signatures. Three module-level tables record methods, generic bounds and
    impl bindings; ``supports_trait`` and ``dispatch_trait`` use them without
    importing native code.

    Args:
        traits: Trait declarations from the Polyglot AST.
        impls: Impl blocks, including ``impl Trait for Type`` blocks.

    Returns:
        Python source defining protocols, tables, and dispatch helpers.
    """
    translator = _prepared_translator((), impls)
    trait_names = [trait.name for trait in traits]
    for name in trait_names:
        _identifier(name, "trait name")
    chunks: list[str] = []
    method_rows: list[str] = []
    bound_rows: list[str] = []
    impl_rows: list[str] = []
    protocol_rows: list[str] = []
    for trait in traits:
        trait_name = _safe_name(trait.name)
        methods = _trait_method_names(trait)
        protocol_lines = [
            "@runtime_checkable",
            f"class {trait_name}(Protocol):",
            f'    """Protocol generated for trait {trait.name}."""',
        ]
        for item in trait.items:
            if item.fn is None:
                continue
            node = translator.translate_fn(item.fn, is_method=True)
            node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
            protocol_lines.append(
                _indent(ast.unparse(ast.fix_missing_locations(node)), 4)
            )
        chunks.append("\n".join(protocol_lines))
        trait_key = _string_literal(trait.name)
        method_rows.append(f"    {trait_key}: {_string_tuple(methods)},")
        bounds = tuple(
            sorted({bound.name for param in trait.generics for bound in param.bounds})
        )
        bound_rows.append(f"    {trait_key}: {_string_tuple(bounds)},")
        implementations: dict[str, tuple[str, ...]] = {}
        for impl in impls:
            if _type_name(impl.trait) != trait.name:
                continue
            target = _type_name(impl.ty)
            if not target:
                continue
            names = tuple(
                item.name
                for item in impl.items
                if isinstance(item, FnDecl) and item.name in methods
            )
            implementations[target] = names
        impl_entries = ", ".join(
            f"{_string_literal(name)}: {_string_tuple(values)}"
            for name, values in sorted(implementations.items())
        )
        impl_rows.append(f"    {trait_key}: {{{impl_entries}}},")
        protocol_rows.append(f"    {trait_key}: {trait_name},")
    chunks.append(
        "\n".join(
            [
                "_TRAIT_METHODS: dict[str, tuple[str, ...]] = {",
                *method_rows,
                "}",
                "",
                "_TRAIT_BOUNDS: dict[str, tuple[str, ...]] = {",
                *bound_rows,
                "}",
                "",
                "_TRAIT_IMPLS: dict[str, dict[str, tuple[str, ...]]] = {",
                *impl_rows,
                "}",
                "",
                "_TRAIT_PROTOCOLS: dict[str, type] = {",
                *protocol_rows,
                "}",
                "",
                "",
                "def supports_trait(trait: str, receiver: object) -> bool:",
                '    """Return True when ``receiver`` satisfies the named protocol."""',
                "    protocol = _TRAIT_PROTOCOLS.get(trait)",
                "    if protocol is None:",
                "        return False",
                "    try:",
                "        return isinstance(receiver, protocol)",
                "    except TypeError:",
                "        return False",
                "",
                "",
                "def dispatch_trait(",
                "    trait: str,",
                "    receiver: object,",
                "    method: str,",
                "    *args: object,",
                "    **kwargs: object,",
                ") -> object:",
                '    """Call a trait method through the generated dispatch table."""',
                "    if trait not in _TRAIT_IMPLS:",
                '        raise KeyError("unknown trait " + repr(trait))',
                "    if method not in _TRAIT_METHODS.get(trait, ()):",
                "        raise TypeError(",
                '            "not a method of trait " + repr(trait) + ": " + repr(method)',
                "        )",
                "    implementations = _TRAIT_IMPLS[trait]",
                "    type_name = type(receiver).__name__",
                "    if type_name not in implementations:",
                "        raise TypeError(",
                '            type_name + " does not implement trait " + repr(trait)',
                "        )",
                "    if method not in implementations[type_name]:",
                "        raise TypeError(",
                '            type_name + " has no implementation for " + repr(method)',
                "        )",
                "    target = getattr(receiver, method, None)",
                "    if not callable(target):",
                "        raise TypeError(",
                '            "bound method is missing: " + repr(method)',
                "        )",
                "    return target(*args, **kwargs)",
                "",
            ]
        )
    )
    return "\n\n".join(chunk for chunk in chunks if chunk) + "\n"


# ---------------------------------------------------------------------------
# 34. Match lowering
# ---------------------------------------------------------------------------


def _fallback_statement(translator: Translator, match: MatchExpr) -> ast.Expr:
    """Build the non-exhaustive-match fallback for a lowered match."""
    scrutinee_type = translator._match_scrutinee_type(match) or "<match>"
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_runtime_non_exhaustive_match", ctx=ast.Load()),
            args=[ast.Constant(value=scrutinee_type)],
            keywords=[],
        )
    )


def py_emit_match_lowering(
    match: MatchExpr,
    *,
    subject: str = "_match_subject",
    result: str = "_match_result",
) -> str:
    """Lower a match expression into a Python if/elif statement chain.

    The returned source is intended to be used as a function body: it assigns
    the scrutinee to ``subject``, binds pattern variables inside each arm,
    assigns the selected arm value to ``result``, and returns it. Guards run
    *after* bindings (so they may use pattern variables), and non-exhaustive
    matches call the same runtime helper as the main translator.

    Args:
        match: A parsed ``MatchExpr`` node.
        subject: Local name used to hold the scrutinee.
        result: Local name used to hold the selected arm value.

    Returns:
        Indentation-ready Python source (no leading indentation).
    """
    _identifier(subject, "subject")
    _identifier(result, "result")
    translator = _prepared_translator()
    scrutinee_type = translator._match_scrutinee_type(match)
    subject_expr = ast.Name(id=subject, ctx=ast.Load())
    chain: ast.stmt | None = None
    for arm in reversed(match.arms):
        test, bindings = translator._pattern_test_bindings(
            arm.pattern, subject_expr, scrutinee_type
        )
        body: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=value,
            )
            for name, value in reversed(list(bindings.items()))
        ]
        result_assignment = ast.Assign(
            targets=[ast.Name(id=result, ctx=ast.Store())],
            value=translator.translate_expr(arm.body),
        )
        is_catch_all = isinstance(test, ast.Constant) and test.value is True
        if arm.guard is not None:
            guard_orelse = (
                [chain]
                if chain is not None
                else [_fallback_statement(translator, match)]
            )
            body.append(
                ast.If(
                    test=translator.translate_expr(arm.guard),
                    body=[result_assignment],
                    orelse=guard_orelse,
                )
            )
            is_catch_all = False
        else:
            body.append(result_assignment)
        if chain is not None:
            orelse = [chain]
        elif is_catch_all:
            orelse = []
        else:
            orelse = [_fallback_statement(translator, match)]
        chain = ast.If(
            test=ast.Constant(value=True) if is_catch_all else test,
            body=body,
            orelse=orelse,
        )
    if chain is None:
        chain = _fallback_statement(translator, match)
    statements: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=subject, ctx=ast.Store())],
            value=translator.translate_expr(match.scrutinee),
        ),
        chain,
        ast.Return(value=ast.Name(id=result, ctx=ast.Load())),
    ]
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    return ast.unparse(module)


# ---------------------------------------------------------------------------
# 35. Macro expansion
# ---------------------------------------------------------------------------


def py_emit_macro_expansions(
    calls: Sequence[MacroCall],
    *,
    prefix: str = "_macro_result",
) -> str:
    """Expand Rust-style macros into deterministic Python expressions.

    ``vec!``, ``assert!``, ``assert_eq!``, ``assert_ne!``, ``format!`` and the
    print family are expanded through the translator's macro table. Each
    expansion is stored in a numbered local and ``<prefix>_all`` collects them
    in invocation order.

    Args:
        calls: Macro-call nodes from the Polyglot AST.
        prefix: Prefix for generated local variable names.

    Returns:
        Python source statements that can be executed after the module header.
    """
    _identifier(prefix, "prefix")
    translator = _prepared_translator()
    lines: list[str] = []
    names: list[str] = []
    for index, call in enumerate(calls):
        expression = translator._translate_macro(call)
        name = f"{prefix}_{index}"
        names.append(name)
        lines.append(f"{name} = {ast.unparse(ast.fix_missing_locations(expression))}")
    collected = ", ".join(names)
    lines.append(
        f"{prefix}_all: tuple[object, ...] = ({collected}{',' if len(names) == 1 else ''})"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 36. Native extension binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeBinding:
    """An immutable view of native callables plus their fallback functions."""

    module: str
    loaded: bool
    functions: Mapping[str, Callable[..., Any]]
    fallbacks: tuple[str, ...]
    reason: str | None = None

    def __getitem__(self, name: str) -> Callable[..., Any]:
        """Return the bound callable named ``name``."""
        try:
            return self.functions[name]
        except KeyError as error:
            raise KeyError(f"no native or fallback function named {name!r}") from error

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a bound callable by name."""
        return self[name](*args, **kwargs)


@contextmanager
def _temporary_sys_path(paths: Sequence[str | Path]) -> Iterator[None]:
    """Prepend ``paths`` to ``sys.path`` for the duration of the context."""
    additions = [str(path) for path in paths if str(path) not in sys.path]
    if not additions:
        yield
        return
    original = list(sys.path)
    sys.path[:0] = additions
    try:
        yield
    finally:
        sys.path[:] = original


def py_bind_native_extension_module(
    module_name: str,
    *,
    fallback: Mapping[str, Callable[..., Any]] | None = None,
    required: Sequence[str] = (),
    search_paths: Sequence[str | Path] = (),
) -> NativeBinding:
    """Import a compiled PyO3 module and bind an explicit callable allowlist.

    Import failures are degraded into fallback callables rather than propagated,
    which keeps generated Python usable on platforms without a native build.
    A required name with neither a native symbol nor a fallback raises
    ``RuntimeError`` so silent misconfiguration cannot occur.

    Args:
        module_name: Dotted import path of the native extension.
        fallback: Pure-Python callables used when native symbols are absent.
        required: Symbol names that must resolve natively or via fallback.
        search_paths: Temporary directories prepended to ``sys.path``.

    Returns:
        A :class:`NativeBinding` describing what was loaded.

    Raises:
        ValueError: If ``module_name`` is not a dotted identifier.
        RuntimeError: If a required symbol cannot be resolved.
    """
    parts = module_name.split(".")
    if not all(part.isidentifier() for part in parts):
        raise ValueError(f"invalid native module name: {module_name!r}")
    fallbacks = dict(fallback or {})
    names = tuple(dict.fromkeys((*required, *fallbacks)))
    for name in names:
        _identifier(name, "function name")
    module: ModuleType | None = None
    reason: str | None = None
    with _temporary_sys_path(search_paths):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError, AttributeError) as error:
            spec = None
            reason = f"cannot locate {module_name!r}: {error}"
        if spec is not None:
            try:
                module = importlib.import_module(module_name)
            except Exception as error:  # native load failures are expected
                module = None
                reason = f"cannot import {module_name!r}: {error}"
    if module is None and reason is None:
        reason = f"native module {module_name!r} is unavailable"
    functions: dict[str, Callable[..., Any]] = {}
    used_fallbacks: list[str] = []
    missing: list[str] = []
    for name in names:
        symbol = getattr(module, name, None) if module is not None else None
        if callable(symbol):
            functions[name] = symbol
        elif name in fallbacks and callable(fallbacks[name]):
            functions[name] = fallbacks[name]
            used_fallbacks.append(name)
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"native binding {module_name!r} is missing required symbols: "
            + ", ".join(sorted(missing))
        )
    return NativeBinding(
        module=module_name,
        loaded=module is not None,
        functions=MappingProxyType(functions),
        fallbacks=tuple(used_fallbacks),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 37. Result and error marshalling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarshalledResult:
    """A structured view of a native Result, error, or plain value."""

    ok: bool
    value: Any
    error: Exception | None
    variant: str

    def unwrap(self) -> Any:
        """Return the success value or raise the marshalled exception."""
        if not self.ok:
            if self.error is not None:
                raise self.error
            raise RuntimeError(f"marshalled error variant {self.variant!r}")
        return self.value


def _as_error(
    payload: Any,
    error_type: type[Exception],
    variant: str,
) -> Exception:
    """Convert an error payload into an exception carrying structured data."""
    if isinstance(payload, error_type):
        error = payload
    elif isinstance(payload, BaseException):
        error = error_type(str(payload))
    elif isinstance(payload, str):
        error = error_type(payload)
    else:
        error = error_type(f"{variant}: {payload!r}")
    error.polyglot_payload = payload  # type: ignore[attr-defined]
    error.polyglot_variant = variant  # type: ignore[attr-defined]
    return error


def _variant_tag(value: Any) -> str:
    """Return the tag name of a marker or string variant."""
    if isinstance(value, str):
        return value.split("::")[-1]
    name = getattr(value, "name", None)
    return str(name) if name is not None else type(value).__name__


def py_marshal_result_and_errors(
    value: Any,
    *,
    error_type: type[Exception] = RuntimeError,
    variant: str | None = None,
) -> MarshalledResult:
    """Convert native Result/Err variants into Python values or exceptions.

    Accepts translator ``_Result``/``_Option`` values, ``_ResultErrSentinel``
    exceptions, one- or two-element tagged tuples such as ``("Ok", 7)``,
    arbitrary exceptions, and plain values. Error payloads are wrapped in
    ``error_type`` and annotated with ``polyglot_payload``/``polyglot_variant``.

    Args:
        value: The native or generated value to marshal.
        error_type: Exception class used for non-exception error payloads.
        variant: Explicit variant label overriding the detected one.

    Returns:
        A :class:`MarshalledResult` with ``ok``, ``value``, ``error`` and
        ``variant`` fields.

    Raises:
        TypeError: If ``error_type`` is not an Exception subclass.
    """
    if not isinstance(error_type, type) or not issubclass(error_type, Exception):
        raise TypeError("error_type must be an Exception subclass")
    detected = variant
    if isinstance(value, _ResultErrSentinel):
        value = value.result
    if isinstance(value, _Result):
        tag = "Ok" if value.is_ok() else "Err"
        payload = value._value
        detected = detected or tag
        if tag == "Ok":
            return MarshalledResult(True, payload, None, detected)
        error = _as_error(payload, error_type, detected)
        return MarshalledResult(False, None, error, detected)
    if isinstance(value, _Option):
        tag = "Some" if value.is_some() else "None"
        detected = detected or tag
        payload = value._value
        return MarshalledResult(True, payload, None, detected)
    if isinstance(value, BaseException):
        detected = detected or type(value).__name__
        return MarshalledResult(
            False, None, _as_error(value, error_type, detected), detected
        )
    if isinstance(value, tuple | list) and 1 <= len(value) <= 2:
        tag = _variant_tag(value[0])
        payload = value[1] if len(value) == 2 else None
        detected = detected or tag
        if tag in {"Err", "Error", "Failure"}:
            return MarshalledResult(
                False, None, _as_error(payload, error_type, detected), detected
            )
        if tag in {"None", "Null"}:
            return MarshalledResult(True, None, None, detected)
        return MarshalledResult(True, payload, None, detected)
    return MarshalledResult(True, value, None, detected or "Value")


# ---------------------------------------------------------------------------
# 38. Frame and trace guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTraceSnapshot:
    """State captured before generated code temporarily suspends tracing."""

    trace: object | None
    frame_trace: object | None
    line_tracing: bool
    opcode_tracing: bool
    implementation: str

    def summary(self, frame: FrameType | None) -> dict[str, Any]:
        """Return a value-free description of ``frame`` for safe diagnostics."""
        if frame is None:
            return {"available": False}
        return {
            "available": True,
            "function": frame.f_code.co_name,
            "filename": frame.f_code.co_filename,
            "lineno": frame.f_lineno,
            "locals": tuple(sorted(frame.f_locals)),
            "globals": tuple(
                sorted(name for name in frame.f_globals if not name.startswith("__"))
            ),
        }


class FrameTraceGuard:
    """Context manager that suspends and restores CPython trace state."""

    def __init__(
        self,
        *,
        frame: FrameType | None = None,
        suspend_trace: bool = True,
    ) -> None:
        self._frame = frame
        self._suspend_trace = suspend_trace
        self._target: FrameType | None = None
        self._snapshot: FrameTraceSnapshot | None = None

    def _caller_frame(self) -> FrameType | None:
        """Return the frame that called ``__enter__`` when CPython allows it."""
        getter = getattr(sys, "_getframe", None)
        if getter is None:
            return None
        try:
            return getter(2)  # _caller_frame -> __enter__ -> caller
        except ValueError:
            return None

    def __enter__(self) -> FrameTraceSnapshot:
        """Suspend tracing on the caller frame and return its snapshot."""
        target = self._frame if self._frame is not None else self._caller_frame()
        self._target = target
        snapshot = FrameTraceSnapshot(
            trace=sys.gettrace(),
            frame_trace=getattr(target, "f_trace", None),
            line_tracing=bool(getattr(target, "f_trace_lines", False)),
            opcode_tracing=bool(getattr(target, "f_trace_opcodes", False)),
            implementation=sys.implementation.name,
        )
        self._snapshot = snapshot
        if self._suspend_trace and snapshot.trace is not None:
            sys.settrace(None)
        if target is not None:
            target.f_trace = None
            target.f_trace_lines = False
            target.f_trace_opcodes = False
        return snapshot

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restore trace state even when generated code raises."""
        snapshot = self._snapshot
        if snapshot is not None and self._suspend_trace and snapshot.trace is not None:
            sys.settrace(snapshot.trace)  # type: ignore[arg-type]
        target = self._target
        if target is not None and snapshot is not None:
            target.f_trace = snapshot.frame_trace  # type: ignore[assignment]
            target.f_trace_lines = snapshot.line_tracing
            target.f_trace_opcodes = snapshot.opcode_tracing
        return False


def py_guard_frame_and_trace_state(
    *,
    frame: FrameType | None = None,
    suspend_trace: bool = True,
) -> FrameTraceGuard:
    """Create a guard that keeps frame inspection hooks safe.

    On CPython, generated code can otherwise observe or mutate an active
    debugger's ``f_trace`` hooks. The guard snapshots ``sys.gettrace()`` and the
    frame flags, clears them for the generated block, and restores them in
    ``__exit__``. ``FrameTraceSnapshot.summary`` exposes names only, never
    local values.

    Args:
        frame: Frame to guard; defaults to the caller's frame.
        suspend_trace: Set False to snapshot without suspending tracing.

    Returns:
        A reusable context manager object.
    """
    return FrameTraceGuard(frame=frame, suspend_trace=suspend_trace)


# ---------------------------------------------------------------------------
# 39. Ruff-compatible formatting
# ---------------------------------------------------------------------------


def _normalize_quotes(source: str) -> str:
    """Normalise simple single-quoted strings to double quotes."""

    def replace(match: re.Match[str]) -> str:
        prefix, body = match.group(1), match.group(2)
        if "\\'" in body or '@"' in body:
            return match.group(0)
        return f'{prefix}"{body}"'

    return _SIMPLE_SINGLE_QUOTED.sub(replace, source)


def _add_blank_lines(source: str) -> str:
    """Ensure two blank lines before top-level definitions."""
    lines = source.splitlines()
    starts: set[int] = set()
    for index, line in enumerate(lines):
        if line.startswith(_TOP_LEVEL_DEF):
            starts.add(index)
        elif line.startswith("@") and (
            index == 0 or not lines[index - 1].startswith("@")
        ):
            cursor = index
            while cursor < len(lines) and lines[cursor].startswith("@"):
                cursor += 1
            if cursor < len(lines) and lines[cursor].startswith(_TOP_LEVEL_DEF):
                starts.add(index)
    result: list[str] = []
    for index, line in enumerate(lines):
        if index in starts and result:
            while result and not result[-1]:
                result.pop()
            result.extend(["", ""])
        result.append(line)
    return "\n".join(result)


def _matching_paren(text: str, opening: int) -> int | None:
    """Return the index of the bracket matching ``text[opening]``."""
    depth = 0
    quote: str | None = None
    index = opening
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _split_arguments(text: str) -> list[str]:
    """Split comma-separated call arguments at bracket depth zero."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _wrap_long_from_import(line: str, line_length: int) -> list[str] | None:
    """Return a parenthesised form of a long ``from ... import ...`` line."""
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    match = _FROM_IMPORT.fullmatch(stripped)
    if match is None:
        return None
    names = _split_arguments(match.group(2))
    if len(names) < 2:
        return None
    candidate = [
        f"{indent}from {match.group(1)} import (",
        *(f"{indent}    {name}," for name in names),
        f"{indent})",
    ]
    if any(len(part) > line_length for part in candidate):
        return None
    return candidate


def _wrap_bracketed_line(line: str, line_length: int) -> list[str] | None:
    """Return a vertically wrapped form of a long bracketed line."""
    indent = line[: len(line) - len(line.lstrip())]
    for opening, char in enumerate(line):
        if char not in "([{":
            continue
        closing = _matching_paren(line, opening)
        if closing is None or line[closing + 1 :].strip() not in {"", ":"}:
            continue
        arguments = _split_arguments(line[opening + 1 : closing])
        if len(arguments) < 2:
            continue
        candidate = [line[: opening + 1].rstrip()]
        candidate.extend(f"{indent}    {argument}," for argument in arguments)
        candidate.append(f"{indent}{line[closing:].rstrip()}")
        if any(len(part) > line_length for part in candidate):
            continue
        return candidate
    return None


def _wrap_long_lines(source: str, line_length: int) -> str:
    """Conservatively wrap long imports, calls, literals and collections."""
    result: list[str] = []
    for line in source.splitlines():
        if len(line) <= line_length:
            result.append(line)
            continue
        wrapped = _wrap_long_from_import(line, line_length)
        if wrapped is None:
            wrapped = _wrap_bracketed_line(line, line_length)
        if wrapped is None:
            result.append(line)
        else:
            result.extend(wrapped)
    return "\n".join(result)


def py_format_with_ruff_compatible_style(
    source: str,
    *,
    line_length: int = 88,
) -> str:
    """Normalise Python source into deterministic, ruff-compatible style.

    The formatter parses the input (rejecting invalid syntax), re-emits it via
    ``ast.unparse``, normalises simple string quotes, enforces two blank lines
    before top-level definitions, strips trailing whitespace and wraps long
    imports, calls and collections conservatively. Every rewrite is
    syntax-checked before it is kept.

    Args:
        source: Python source to format.
        line_length: Maximum preferred line length.

    Returns:
        A canonical source string ending with exactly one newline.

    Raises:
        SyntaxError: If ``source`` is not valid Python.
        ValueError: If ``line_length`` is smaller than 40.
    """
    if line_length < 40:
        raise ValueError("line_length must be at least 40")
    ast.parse(source)
    rendered = ast.unparse(ast.parse(source))
    formatted = _add_blank_lines(_normalize_quotes(rendered))
    formatted = _wrap_long_lines(formatted, line_length)
    ast.parse(formatted)
    cleaned = "\n".join(line.rstrip() for line in formatted.splitlines())
    return cleaned.rstrip() + "\n"


# ---------------------------------------------------------------------------
# 40. Parity harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityCase:
    """The outcome of one function/input pair across available backends."""

    function: str
    case: int
    python: Any
    reference: Any
    native: Any
    matches_reference: bool
    matches_native: bool


@dataclass(frozen=True)
class ParityReport:
    """Aggregated results of a Python-vs-reference/native parity run."""

    canonical_source: str
    cases: tuple[ParityCase, ...]
    passed: bool
    mismatches: tuple[str, ...]

    @property
    def case_count(self) -> int:
        """Return the number of executed comparison cases."""
        return len(self.cases)


def _canonical_outcome(value: Any) -> Any:
    """Map a backend result to a stable, comparable value."""
    if isinstance(value, _Result):
        marshalled = py_marshal_result_and_errors(value)
        if marshalled.ok:
            return _canonical_outcome(marshalled.value)
        return _canonical_outcome(marshalled.error)
    if isinstance(value, _Option):
        return _canonical_outcome(value._value)
    if isinstance(value, BaseException):
        return ("raises", type(value).__name__, str(value))
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _canonical_outcome(item)) for key, item in value.items())
        )
    if isinstance(value, list | tuple):
        return tuple(_canonical_outcome(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            _canonical_outcome(getattr(value, field.name)) for field in fields(value)
        )
    return value


def _call_outcome(
    function: Callable[..., Any] | None,
    arguments: Mapping[str, Any],
) -> Any:
    """Call ``function`` and return a canonical value-or-exception outcome."""
    if function is None:
        return None
    try:
        return _canonical_outcome(function(**arguments))
    except BaseException as error:  # parity treats exceptions as values
        return _canonical_outcome(error)


def py_run_parity_harness(
    source: str,
    *,
    inputs: Sequence[Mapping[str, Any]] = (),
    functions: Sequence[str] = (),
    reference: Mapping[str, Callable[..., Any]] | None = None,
    native: Mapping[str, Callable[..., Any]] | None = None,
) -> ParityReport:
    """Execute Python-generated functions and compare them with oracles.

    The source is parsed and compiled through the trusted Python backend.
    Reference and native callables are optional; Results are marshalled so a
    Python ``Err("x")`` compares equal to a native ``RuntimeError("x")``.

    Args:
        source: Polyglot source text to compile.
        inputs: Keyword-argument mappings, one per case.
        functions: Functions to compare; defaults to public namespace callables.
        reference: Pure-Python oracle callables keyed by function name.
        native: Native backend callables keyed by function name.

    Returns:
        A :class:`ParityReport` with one :class:`ParityCase` per pairing.
    """
    program: Program = parse(lex(source))
    namespace = get_backend("python", trusted=True).compile(program)
    reference_functions = dict(reference or {})
    native_functions = dict(native or {})
    if functions:
        names = tuple(dict.fromkeys(functions))
    else:
        names = tuple(
            sorted(
                name
                for name, value in namespace.items()
                if callable(value) and not name.startswith("_")
            )
        )
    cases: list[ParityCase] = []
    mismatches: list[str] = []
    for name in names:
        python_function = namespace.get(name)
        if not callable(python_function):
            raise KeyError(
                f"function {name!r} is not present in the compiled namespace"
            )
        reference_function = reference_functions.get(name)
        native_function = native_functions.get(name)
        for case_index, arguments in enumerate(inputs):
            python_outcome = _call_outcome(python_function, arguments)
            reference_outcome = _call_outcome(reference_function, arguments)
            native_outcome = _call_outcome(native_function, arguments)
            matches_reference = (
                reference_function is None or python_outcome == reference_outcome
            )
            matches_native = native_function is None or python_outcome == native_outcome
            if not matches_reference:
                mismatches.append(
                    f"{name} case {case_index}: python={python_outcome!r} "
                    f"reference={reference_outcome!r}"
                )
            if not matches_native:
                mismatches.append(
                    f"{name} case {case_index}: python={python_outcome!r} "
                    f"native={native_outcome!r}"
                )
            cases.append(
                ParityCase(
                    function=name,
                    case=case_index,
                    python=python_outcome,
                    reference=reference_outcome,
                    native=native_outcome,
                    matches_reference=matches_reference,
                    matches_native=matches_native,
                )
            )
    return ParityReport(
        canonical_source=program_to_source(program),
        cases=tuple(cases),
        passed=not mismatches,
        mismatches=tuple(mismatches),
    )
