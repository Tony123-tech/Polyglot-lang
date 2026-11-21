"""Deterministic Python source emitters for the backend framework rollout.

Every public ``pygen_*`` function returns a structured record whose ``source``
entry is a stable Python source string.  Inputs are either genuine
:mod:`ast_nodes` declarations or pre-validated one-line source snippets; the
latter are checked by parsing them into a small closed expression/statement
subset so emitted source cannot acquire arbitrary executable text.  The module
never imports or executes generated code.

Diagnostics follow the project's ``E001``-``E008`` surface: ``E001`` for
invalid names and emitted layouts, ``E002`` for AST shapes outside the supported
emission subset, ``E003`` for API misuse, and ``E008`` for malformed literals.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn

import ast_nodes as astn
from compiler.frontend.errors import PolyglotError

__all__ = [
    "pygen_assignment",
    "pygen_call_expression",
    "pygen_conditional",
    "pygen_dataclass_field",
    "pygen_decorator",
    "pygen_docstring_block",
    "pygen_enum_union",
    "pygen_enum_variant_constructor",
    "pygen_exception_guard",
    "pygen_function_body",
    "pygen_function_signature",
    "pygen_impl_methods",
    "pygen_import_block",
    "pygen_literal_source",
    "pygen_loop_block",
    "pygen_macro_call_source",
    "pygen_match_dispatch",
    "pygen_module_exports",
    "pygen_module_header",
    "pygen_native_binding_stub",
    "pygen_pattern_guard",
    "pygen_return_statement",
    "pygen_struct_dataclass",
    "pygen_trait_protocol",
    "pygen_type_annotation",
]

_PRIMITIVES: Final[dict[str, str]] = {
    "bool": "bool",
    "char": "str",
    "f32": "float",
    "f64": "float",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "i128": "int",
    "isize": "int",
    "str": "str",
    "String": "str",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "u128": "int",
    "unit": "None",
    "usize": "int",
}
_CONTAINERS: Final[dict[str, str]] = {
    "ArrayVec": "list",
    "HashMap": "dict",
    "HashSet": "set",
    "Option": "object",
    "Result": "object",
    "Vec": "list",
}
_HELPER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "_Option",
        "_Result",
        "_assert",
        "_assert_eq",
        "_panic",
        "_runtime_non_exhaustive_match",
        "_runtime_int_div",
        "_runtime_int_rem",
    }
)
_EXPR_NODES: Final[tuple[type[ast.AST], ...]] = (
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Dict,
    ast.IfExp,
    ast.List,
    ast.Name,
    ast.Tuple,
    ast.UnaryOp,
)
_EXPR_INTERNALS: Final[frozenset[type[ast.AST]]] = frozenset(
    {
        ast.Add,
        ast.And,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.Div,
        ast.Eq,
        ast.FloorDiv,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.Is,
        ast.IsNot,
        ast.Lt,
        ast.LtE,
        ast.Mod,
        ast.Mult,
        ast.Not,
        ast.NotEq,
        ast.NotIn,
        ast.Or,
        ast.Sub,
        ast.UAdd,
        ast.USub,
        ast.Expression,
        ast.Load,
        ast.keyword,
        *(_EXPR_NODES),
    }
)
_STMT_NODES: Final[tuple[type[ast.stmt], ...]] = (
    ast.AnnAssign,
    ast.Assign,
    ast.AugAssign,
    ast.Delete,
    ast.Expr,
    ast.Pass,
    ast.Raise,
    ast.Return,
)
_BINARY_OPS: Final[dict[str, str]] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "==": "==",
    "!=": "!=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "&&": "and",
    "||": "or",
    "&": "&",
    "|": "|",
    "^": "^",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured emitter diagnostic."""
    raise PolyglotError(code, message)


def _text(value: object, label: str) -> str:
    """Require *value* to be a non-empty string."""
    if not isinstance(value, str) or not value:
        _fail("E003", f"{label} must be a non-empty str")
    return value


def _identifier(value: object, label: str, *, dotted: bool = False) -> str:
    """Validate a Python identifier (or a dotted chain of identifiers)."""
    text = _text(value, label)
    parts = text.split(".") if dotted else [text]
    if not all(part.isidentifier() for part in parts):
        _fail("E001", f"{label} is not a valid identifier: {text!r}")
    return text


def _expression(value: object, label: str) -> str:
    """Accept a one-line expression inside the closed emitter subset."""
    text = _text(value, label)
    if "\n" in text or ";" in text:
        _fail("E003", f"{label} must be a single expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        _fail("E003", f"{label} is not valid Python: {text!r}")
    if any(type(node) not in _EXPR_INTERNALS for node in ast.walk(tree)):
        _fail("E003", f"{label} uses an unsupported expression form")
    if any(
        isinstance(node, ast.Call) and not isinstance(node.func, ast.Name)
        for node in ast.walk(tree)
    ):
        _fail("E003", f"{label} may only call plain names")
    return text


def _decorator_argument(value: object) -> str:
    """Accept a plain positional or keyword decorator argument."""
    text = _text(value, "decorator argument")
    if "\n" in text or ";" in text:
        _fail("E003", "decorator argument must be a single expression")
    try:
        tree = ast.parse(f"_decorator({text})", mode="eval")
    except SyntaxError:
        _fail("E003", f"invalid decorator argument: {text!r}")
    if not isinstance(tree.body, ast.Call) or any(
        isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda))
        for node in ast.walk(tree)
    ):
        _fail("E003", f"unsupported decorator argument: {text!r}")
    return text


def _statement(value: object, label: str) -> str:
    """Accept a single simple statement inside the closed emitter subset."""
    text = _text(value, label)
    if "\n" in text or ";" in text:
        _fail("E003", f"{label} must be a single line")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        _fail("E003", f"{label} is not valid Python: {text!r}")
    if len(tree.body) != 1 or not isinstance(tree.body[0], _STMT_NODES):
        _fail("E003", f"{label} is not a supported statement")
    node = tree.body[0]
    if isinstance(node, ast.Expr) and not isinstance(node.value, _EXPR_NODES):
        _fail("E003", f"{label} uses an unsupported expression")
    return text


def _indent(lines: Sequence[str], level: int) -> list[str]:
    """Indent non-empty source lines by *level* four-space steps."""
    return [("    " * level + line) if line.strip() else "" for line in lines]


def _import_statement(value: object) -> tuple[str, list[str]]:
    """Validate one plain import statement and return its bound names."""
    text = _text(value, "import")
    if "\n" in text or ";" in text:
        _fail("E001", f"unsupported import statement: {text!r}")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        _fail("E001", f"invalid import statement: {text!r}")
    if len(tree.body) != 1 or not isinstance(
        tree.body[0], (ast.Import, ast.ImportFrom)
    ):
        _fail("E001", f"unsupported import statement: {text!r}")
    node = tree.body[0]
    bindings: list[str] = []
    for alias in node.names:
        if alias.name == "*":
            _fail("E001", "star imports are not allowed in generated modules")
        bindings.append(alias.asname or alias.name.split(".")[0])
    return text, bindings


def _annotation(ty: object) -> tuple[str, str, list[str]]:
    """Map a project type node to ``(source, kind, generic_args)``."""
    if ty is None:
        return "None", "unit", []
    if not isinstance(ty, astn.Node):
        _fail("E003", "type must be a project AST type node or None")
    if isinstance(ty, astn.TypeName):
        name = _text(ty.name, "type name")
        if name in _PRIMITIVES:
            return _PRIMITIVES[name], "primitive", []
        if name in _CONTAINERS:
            args = [_annotation(arg)[0] for arg in ty.args]
            suffix = f"[{', '.join(args)}]" if args else ""
            return f"{_CONTAINERS[name]}{suffix}", "container", args
        return name, "named", [name]
    if isinstance(ty, astn.TypeRef):
        return _annotation(ty.inner)[0], "reference", []
    if isinstance(ty, astn.TypeSlice):
        return f"list[{_annotation(ty.inner)[0]}]", "slice", []
    if isinstance(ty, astn.TypeArray):
        return f"list[{_annotation(ty.inner)[0]}]", "array", []
    if isinstance(ty, astn.TypeTuple):
        args = [_annotation(elem)[0] for elem in ty.elems]
        return f"tuple[{', '.join(args)}]" if args else "tuple[()]", "tuple", args
    if isinstance(ty, astn.TypeFn):
        params = [_annotation(param)[0] for param in ty.params]
        ret = _annotation(ty.ret)[0]
        return f"Callable[[{', '.join(params)}], {ret}]", "function", params
    if isinstance(ty, (astn.TypeImplTrait, astn.TypeDynTrait)):
        bounds = [_text(bound.name, "trait bound") for bound in ty.bounds]
        return "object", "trait", bounds
    if isinstance(ty, astn.TypeUnit):
        return "None", "unit", []
    _fail("E002", f"unsupported type node: {type(ty).__name__}")


def _literal(node: object) -> tuple[str, str]:
    """Render a literal AST node as a Python expression."""
    if isinstance(node, astn.LitInt):
        return str(int(node.value)), "int"
    if isinstance(node, astn.LitFloat):
        value = float(node.value)
        if not math.isfinite(value):
            _fail("E008", "float literal must be finite")
        return repr(value), "float"
    if isinstance(node, (astn.LitStr, astn.LitChar)):
        return json.dumps(str(node.value)), "string"
    if isinstance(node, astn.LitBool):
        return ("True" if node.value else "False"), "bool"
    if isinstance(node, astn.OptionSome):
        return f'("Some", {_render_expr(node.value)})', "option_some"
    if isinstance(node, astn.OptionNone):
        return '("None",)', "option_none"
    if isinstance(node, astn.ResultOk):
        return f'("Ok", {_render_expr(node.value)})', "result_ok"
    if isinstance(node, astn.ResultErr):
        return f'("Err", {_render_expr(node.value)})', "result_err"
    _fail("E003", f"unsupported literal node: {type(node).__name__}")


def _render_expr(node: object) -> str:
    """Render a small AST expression without evaluating it."""
    if isinstance(
        node, (astn.LitInt, astn.LitFloat, astn.LitStr, astn.LitChar, astn.LitBool)
    ):
        return _literal(node)[0]
    if isinstance(node, astn.Ident):
        return _identifier(node.name, "identifier")
    if isinstance(node, astn.Paren):
        return f"({_render_expr(node.inner)})"
    if isinstance(node, astn.Unary):
        if node.op not in ("-", "+"):
            _fail("E002", f"unsupported unary operator: {node.op}")
        return f"{node.op}{_render_expr(node.operand)}"
    if isinstance(node, astn.Binary):
        operator = _BINARY_OPS.get(node.op)
        if operator is None:
            _fail("E002", f"unsupported binary operator: {node.op}")
        return f"({_render_expr(node.left)} {operator} {_render_expr(node.right)})"
    if isinstance(node, astn.TupleExpr):
        values = ", ".join(_render_expr(elem) for elem in node.elems)
        return f"({values},)" if len(node.elems) == 1 else f"({values})"
    if isinstance(node, astn.ArrayExpr):
        return "[" + ", ".join(_render_expr(elem) for elem in node.elems) + "]"
    if isinstance(
        node, (astn.OptionSome, astn.OptionNone, astn.ResultOk, astn.ResultErr)
    ):
        return _literal(node)[0]
    _fail("E002", f"unsupported expression node: {type(node).__name__}")


def _pattern(pattern: object, subject: str) -> tuple[str, list[str]]:
    """Build a boolean match test and the names bound by a pattern."""
    if not isinstance(pattern, astn.Node):
        _fail("E003", "pattern must be a project AST pattern node")
    if isinstance(pattern, astn.PatWildcard):
        return "True", []
    if isinstance(pattern, astn.PatIdent):
        name = _identifier(pattern.name, "binding name")
        return "True", [name]
    if isinstance(pattern, astn.PatLiteral):
        value = pattern.value
        source = (
            _literal(value)[0] if isinstance(value, astn.Node) else json.dumps(value)
        )
        return f"{subject} == {source}", []
    if isinstance(pattern, (astn.PatPath, astn.PatConst)):
        return f"{subject}[0] == {json.dumps(_text(pattern.path, 'path'))}", []
    if isinstance(pattern, astn.PatEnum):
        tag = _text(pattern.path, "variant path").split("::")[-1]
        checks = [f"{subject}[0] == {json.dumps(tag)}"]
        bindings: list[str] = []
        for index, argument in enumerate(pattern.args):
            clause, names = _pattern(argument, f"{subject}[{index + 1}]")
            checks.append(clause)
            bindings.extend(names)
        return " and ".join(checks), bindings
    if isinstance(pattern, astn.PatTuple):
        bindings = []
        checks = [f"len({subject}) == {len(pattern.elems)}"]
        for index, element in enumerate(pattern.elems):
            clause, names = _pattern(element, f"{subject}[{index}]")
            checks.append(clause)
            bindings.extend(names)
        return " and ".join(checks), bindings
    if isinstance(pattern, (astn.PatRef, astn.PatBox)):
        return _pattern(pattern.inner, subject)
    if isinstance(pattern, astn.PatBinding):
        clause, inner = _pattern(pattern.inner, subject)
        return clause, [_identifier(pattern.name, "binding name"), *inner]
    if isinstance(pattern, astn.PatOr):
        clauses: list[str] = []
        names: list[str] = []
        for alternative in pattern.alts:
            clause, bound = _pattern(alternative, subject)
            clauses.append(clause)
            names.extend(bound)
        return "(" + " or ".join(clauses) + ")", list(dict.fromkeys(names))
    if isinstance(pattern, astn.PatRange):
        comparisons: list[str] = []
        if pattern.start is not None:
            comparisons.append(f"{subject} >= {_render_expr(pattern.start)}")
        if pattern.end is not None:
            operator = "<=" if pattern.inclusive else "<"
            comparisons.append(f"{subject} {operator} {_render_expr(pattern.end)}")
        return " and ".join(comparisons) if comparisons else "True", []
    if isinstance(pattern, astn.PatSlice):
        checks = [f"len({subject}) == {len(pattern.elems)}"]
        names = []
        for index, element in enumerate(pattern.elems):
            clause, bound = _pattern(element, f"{subject}[{index}]")
            checks.append(clause)
            names.extend(bound)
        return " and ".join(checks), names
    _fail("E002", f"unsupported pattern node: {type(pattern).__name__}")


def _check_source(source: str) -> str:
    """Parse emitted source so a malformed emitter is caught immediately."""
    try:
        ast.parse(source)
    except SyntaxError as error:
        _fail("E001", f"generated invalid Python source: {error.msg}")
    return source


# ---------------------------------------------------------------------------
# Public emitters
# ---------------------------------------------------------------------------


def pygen_module_header(
    module_name: str = "_polyglot_generated",
    *,
    imports: Sequence[str] = (),
    helpers: Sequence[str] = (),
    target_width: int = 64,
    version: str = "0.0.0",
) -> dict[str, Any]:
    """Emit a deterministic module preamble with runtime alias and metadata.

    Only plain single-line imports are accepted.  The emitted module imports
    the translator under one alias (``_polyglot_runtime``) so every other
    emitter can refer to the same trusted helper namespace.
    """
    name = _identifier(module_name, "module_name")
    if target_width not in (32, 64):
        _fail("E001", f"target_width must be 32 or 64, got {target_width}")
    version_text = _text(version, "version")
    normalized_imports: list[str] = []
    bindings: list[str] = ["_polyglot_runtime"]
    for statement in imports:
        source, names = _import_statement(statement)
        normalized_imports.append(source)
        bindings.extend(names)
    selected = list(dict.fromkeys(_text(helper, "helper") for helper in helpers))
    unknown = [helper for helper in selected if helper not in _HELPER_NAMES]
    if unknown:
        _fail("E001", f"unknown runtime helpers: {', '.join(sorted(unknown))}")
    lines = [
        f'"""Generated Polyglot module {name}."""',
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from enum import Enum, auto",
        "import backends.python.translator as _polyglot_runtime",
    ]
    lines.extend(sorted(normalized_imports))
    lines.extend(
        [
            "",
            "__polyglot__: dict[str, object] = {",
            f'    "module": {json.dumps(name)},',
            f'    "version": {json.dumps(version_text)},',
            f'    "target_width": {target_width},',
            "}",
            "",
        ]
    )
    source = _check_source("\n".join(lines))
    return {
        "source": source,
        "module": name,
        "version": version_text,
        "target_width": target_width,
        "imports": sorted(normalized_imports),
        "helpers": selected,
        "bindings": sorted(set(bindings)),
    }


def pygen_import_block(
    imports: Sequence[str], *, sorted_imports: bool = True
) -> dict[str, Any]:
    """Emit a sorted block of plain imports and list the names they bind."""
    if isinstance(imports, (str, bytes)) or not isinstance(imports, Sequence):
        _fail("E003", "imports must be a sequence of import statements")
    records = [_import_statement(statement) for statement in imports]
    statements = [record[0] for record in records]
    if sorted_imports:
        statements.sort()
    bindings = sorted({name for _, names in records for name in names})
    source = _check_source("\n".join(statements) + ("\n" if statements else ""))
    return {
        "source": source,
        "statements": statements,
        "bindings": bindings,
        "count": len(statements),
    }


def pygen_struct_dataclass(
    declaration: astn.StructDecl, *, slots: bool = True, docstring: str | None = None
) -> dict[str, Any]:
    """Emit a slotted dataclass for a Polyglot ``struct`` declaration."""
    if not isinstance(declaration, astn.StructDecl):
        _fail("E003", "declaration must be a StructDecl node")
    name = _identifier(declaration.name, "struct name")
    fields: list[dict[str, Any]] = []
    lines: list[str] = [
        f"@dataclass(slots={slots!r}, frozen=False)",
        f"class {name}:",
    ]
    description = (
        docstring if docstring is not None else f"Generated dataclass for {name}."
    )
    lines.extend(_indent([f'"""{_text(description, "docstring")}"""'], 1))
    for field in declaration.fields:
        if not isinstance(field, tuple) or len(field) != 3:
            _fail("E003", "struct fields must be (name, type, is_pub) triples")
        field_name = _identifier(field[0], "field name")
        annotation, kind, _ = _annotation(field[1])
        lines.extend(_indent([f"{field_name}: {annotation}"], 1))
        fields.append({"name": field_name, "annotation": annotation, "kind": kind})
    if not declaration.fields:
        lines.extend(_indent(["pass"], 1))
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "class_name": name,
        "fields": fields,
        "slots": slots,
    }


def pygen_enum_union(
    declaration: astn.EnumDecl, *, representation: str | None = None
) -> dict[str, Any]:
    """Emit an ``Enum`` or tagged-tuple class for a Polyglot ``enum``."""
    if not isinstance(declaration, astn.EnumDecl):
        _fail("E003", "declaration must be an EnumDecl node")
    name = _identifier(declaration.name, "enum name")
    payloads = [variant.fields for variant in declaration.variants]
    is_unit = all(fields is None for fields in payloads)
    kind = representation or ("enum" if is_unit else "tagged")
    if kind not in ("enum", "tagged"):
        _fail("E001", f"unsupported enum representation: {kind}")
    if kind == "enum" and not is_unit:
        _fail("E001", "payload variants require the tagged representation")
    variants = [
        _identifier(variant.name, "variant name") for variant in declaration.variants
    ]
    if len(set(variants)) != len(variants):
        _fail("E001", f"duplicate variants in enum {name}")
    if kind == "enum":
        lines = [f"class {name}(Enum):", f'    """Generated enum for {name}."""']
        for variant in variants:
            lines.append(f"    {variant} = auto()")
    else:
        lines = [
            "@dataclass(frozen=True)",
            f"class {name}:",
            f'    """Tagged union generated for {name}."""',
            "    tag: str",
            "    payload: tuple[object, ...]",
        ]
        for variant in declaration.variants:
            tag = _identifier(variant.name, "variant name")
            arity = 0 if variant.fields is None else len(variant.fields)
            arguments = ", ".join(f"value{index}" for index in range(arity))
            tuple_value = f"({arguments},)" if arity == 1 else f"({arguments})"
            lines.extend(
                [
                    "",
                    "    @staticmethod",
                    f'    def make_{tag}({arguments}) -> "{name}":',
                    f"        return {name}({json.dumps(tag)}, {tuple_value})",
                    "",
                    f"    def is_{tag}(self) -> bool:",
                    f"        return self.tag == {json.dumps(tag)}",
                ]
            )
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "class_name": name,
        "variants": variants,
        "representation": kind,
    }


def pygen_trait_protocol(declaration: astn.TraitDecl) -> dict[str, Any]:
    """Emit a ``runtime_checkable`` protocol for a Polyglot trait."""
    if not isinstance(declaration, astn.TraitDecl):
        _fail("E003", "declaration must be a TraitDecl node")
    name = _identifier(declaration.name, "trait name")
    methods: list[dict[str, Any]] = []
    lines = [
        "@runtime_checkable",
        f"class {name}(Protocol):",
        f'    """Runtime-checkable protocol for trait {name}."""',
    ]
    for item in declaration.items:
        if not isinstance(item, astn.TraitItem) or item.fn is None:
            continue
        signature = pygen_function_signature(item.fn, class_method=True)
        methods.append(
            {
                "name": signature["name"],
                "params": signature["params"],
                "returns": signature["returns"],
            }
        )
        lines.extend(_indent([signature["source"]], 1))
        lines.extend(_indent(["raise NotImplementedError"], 2))
    if not methods:
        lines.extend(_indent(["pass"], 1))
    source = _check_source("\n".join(lines) + "\n")
    return {"source": source, "protocol": name, "methods": methods}


def pygen_impl_methods(
    block: astn.ImplBlock,
    *,
    bodies: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Emit impl-block methods as a standalone class mixin."""
    if not isinstance(block, astn.ImplBlock):
        _fail("E003", "block must be an ImplBlock node")
    target = _annotation(block.ty)[0]
    trait = None if block.trait is None else _annotation(block.trait)[0]
    mixin = f"_{target.replace('[', '_').replace(']', '')}Impl"
    lines = [f"class {mixin}:", f'    """Generated method mixin for {target}."""']
    emitted: list[str] = []
    provided = dict(bodies or {})
    for item in block.items:
        if not isinstance(item, astn.FnDecl):
            _fail("E002", "impl blocks support function items only")
        signature = pygen_function_signature(item, class_method=True)
        lines.extend(_indent([signature["source"]], 1))
        body = [f'raise NotImplementedError("{signature["name"]}")']
        if signature["name"] in provided:
            body = list(provided[signature["name"]])
        lines.extend(_indent(body, 2))
        emitted.append(signature["name"])
    if not emitted:
        lines.extend(_indent(["pass"], 1))
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "mixin": mixin,
        "target": target,
        "trait": trait,
        "methods": emitted,
    }


def pygen_function_signature(
    declaration: astn.FnDecl, *, class_method: bool = False
) -> dict[str, Any]:
    """Emit a Python ``def`` signature for a Polyglot function declaration."""
    if not isinstance(declaration, astn.FnDecl):
        _fail("E003", "declaration must be an FnDecl node")
    name = _identifier(declaration.name, "function name")
    params: list[dict[str, str]] = []
    rendered: list[str] = []
    if class_method:
        rendered.append("self")
    for parameter in declaration.params:
        pattern = parameter.pattern
        if (
            class_method
            and isinstance(pattern, astn.PatRef)
            and isinstance(pattern.inner, astn.PatIdent)
            and pattern.inner.name == "self"
        ):
            continue
        if not isinstance(pattern, astn.PatIdent):
            _fail("E002", "only identifier parameters can be emitted")
        param_name = _identifier(parameter.pattern.name, "parameter name")
        annotation = _annotation(parameter.ty)[0]
        rendered.append(f"{param_name}: {annotation}")
        params.append({"name": param_name, "annotation": annotation})
    returns = _annotation(declaration.ret)[0]
    prefix = "async def" if declaration.is_async else "def"
    suffix = f" -> {returns}" if returns != "None" else ""
    source = f"{prefix} {name}({', '.join(rendered)}){suffix}:"
    _check_source(source + "\n    pass\n")
    return {
        "source": source,
        "name": name,
        "params": params,
        "returns": returns,
        "is_async": declaration.is_async,
    }


def pygen_function_body(
    lines: Sequence[str], *, indent: int = 1, tail: str | None = None
) -> dict[str, Any]:
    """Indent validated statements into a function body and append a return."""
    if isinstance(lines, (str, bytes)) or not isinstance(lines, Sequence):
        _fail("E003", "lines must be a sequence of source statements")
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        _fail("E003", "indent must be a non-negative int")
    statements = [_statement(line, "body line") for line in lines]
    result = _indent(statements, indent)
    if tail is not None:
        result.append(f"return {_expression(tail, 'tail')}")
    elif not statements:
        result.append("pass")
    source = "\n".join(result)
    return {
        "source": source,
        "lines": statements,
        "indent": indent,
        "has_tail": tail is not None,
    }


def pygen_match_dispatch(
    match: astn.MatchExpr,
    *,
    subject: str = "_match_subject",
    result: str = "_match_result",
    fallback: str = "_polyglot_runtime._runtime_non_exhaustive_match",
) -> dict[str, Any]:
    """Emit an if/elif dispatch chain for a Polyglot ``match`` expression."""
    if not isinstance(match, astn.MatchExpr):
        _fail("E003", "match must be a MatchExpr node")
    subject_name = _identifier(subject, "subject")
    result_name = _identifier(result, "result")
    fallback_name = _identifier(fallback, "fallback", dotted=True)
    lines = [f"{subject_name} = {_render_expr(match.scrutinee)}"]
    arm_info: list[dict[str, Any]] = []
    for index, arm in enumerate(match.arms):
        if not isinstance(arm, astn.MatchArm):
            _fail("E003", "match arms must be MatchArm nodes")
        clause, bindings = _pattern(arm.pattern, subject_name)
        if arm.guard is not None:
            clause = f"({clause}) and ({_render_expr(arm.guard)})"
        prefix = "if" if index == 0 else "elif"
        lines.append(f"{prefix} {clause}:")
        for binding in bindings:
            lines.extend(_indent([f"{binding} = {subject_name}"], 1))
        guard_name = f"{result_name}_{index}"
        lines.extend(_indent([f"{guard_name} = {_render_expr(arm.body)}"], 1))
        lines.extend(_indent([f"{result_name} = {guard_name}"], 1))
        arm_info.append(
            {"index": index, "bindings": bindings, "guarded": arm.guard is not None}
        )
    lines.append("else:")
    lines.extend(_indent([f"raise {fallback_name}({subject_name})"], 1))
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "arms": arm_info,
        "subject": subject_name,
        "result": result_name,
        "fallback": fallback_name,
    }


def pygen_pattern_guard(
    pattern: astn.Node, *, subject: str = "_subject"
) -> dict[str, Any]:
    """Emit the boolean guard and binding plan for one match pattern."""
    subject_name = _identifier(subject, "subject")
    clause, bindings = _pattern(pattern, subject_name)
    try:
        ast.parse(clause, mode="eval")
    except SyntaxError:
        _fail("E001", f"generated malformed pattern guard: {clause!r}")
    return {
        "source": clause,
        "kind": type(pattern).__name__,
        "bindings": bindings,
        "subject": subject_name,
    }


def pygen_literal_source(node: astn.Node) -> dict[str, Any]:
    """Render a project literal node as a stable Python expression."""
    source, kind = _literal(node)
    _expression(source, "literal")
    return {"source": source, "kind": kind, "node": type(node).__name__}


def pygen_type_annotation(ty: astn.Node | None) -> dict[str, Any]:
    """Map a project type node to its Python annotation source."""
    source, kind, arguments = _annotation(ty)
    return {
        "source": source,
        "kind": kind,
        "generic_args": arguments,
        "node": "None" if ty is None else type(ty).__name__,
    }


def pygen_call_expression(
    callee: str,
    *,
    arguments: Sequence[str] = (),
    keywords: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Emit a call expression from validated callee and argument snippets."""
    callee_source = _expression(callee, "callee")
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        _fail("E003", "arguments must be a sequence of expressions")
    args = [_expression(argument, "argument") for argument in arguments]
    pairs = list((keywords or {}).items())
    rendered_keywords = []
    for key, value in pairs:
        rendered_keywords.append(
            f"{_identifier(key, 'keyword')}={_expression(value, 'keyword value')}"
        )
    source = f"{callee_source}({', '.join([*args, *rendered_keywords])})"
    _expression(source, "call")
    return {
        "source": source,
        "callee": callee_source,
        "arity": len(args),
        "keywords": [key for key, _ in pairs],
    }


def pygen_assignment(
    target: str, expression: str, *, annotation: str | None = None
) -> dict[str, Any]:
    """Emit a simple, annotated, or augmented Python assignment."""
    target_source = _expression(target, "target")
    value_source = _expression(expression, "expression")
    annotated = annotation is not None
    if annotated:
        source = f"{target_source}: {_text(annotation, 'annotation')} = {value_source}"
    else:
        source = f"{target_source} = {value_source}"
    _statement(source, "assignment")
    return {
        "source": source,
        "target": target_source,
        "annotated": annotated,
        "operator": "=",
    }


def pygen_return_statement(expression: str | None = None) -> dict[str, Any]:
    """Emit a ``return`` statement with or without a value expression."""
    if expression is None:
        source = "return"
        has_value = False
    else:
        source = f"return {_expression(expression, 'expression')}"
        has_value = True
    _statement(source, "return")
    return {"source": source, "has_value": has_value}


def pygen_conditional(
    branches: Sequence[tuple[str, Sequence[str]]], *, else_lines: Sequence[str] = ()
) -> dict[str, Any]:
    """Emit an if/elif/else chain from condition and body source pairs."""
    if isinstance(branches, (str, bytes)) or not isinstance(branches, Sequence):
        _fail("E003", "branches must be a sequence of (condition, body) pairs")
    lines: list[str] = []
    conditions: list[str] = []
    for index, branch in enumerate(branches):
        if not isinstance(branch, tuple) or len(branch) != 2:
            _fail("E003", "each branch must be a (condition, body) pair")
        condition = _expression(branch[0], "condition")
        conditions.append(condition)
        body = branch[1]
        if isinstance(body, (str, bytes)) or not isinstance(body, Sequence):
            _fail("E003", "branch body must be a sequence of statements")
        lines.append(f"{'if' if index == 0 else 'elif'} {condition}:")
        lines.extend(_indent([_statement(line, "branch line") for line in body], 1))
    has_else = bool(tuple(else_lines))
    if has_else:
        lines.append("else:")
        lines.extend(_indent([_statement(line, "else line") for line in else_lines], 1))
    source = _check_source("\n".join(lines) + ("\n" if lines else ""))
    return {
        "source": source,
        "conditions": conditions,
        "branches": len(branches),
        "has_else": has_else,
    }


def pygen_loop_block(
    kind: str,
    *,
    body: Sequence[str],
    condition: str | None = None,
    target: str | None = None,
    iterator: str | None = None,
) -> dict[str, Any]:
    """Emit an infinite, ``while``, or ``for`` loop from safe snippets."""
    kind_text = _text(kind, "kind")
    statements = [_statement(line, "loop body line") for line in body]
    if kind_text == "loop":
        header = "while True:"
    elif kind_text == "while":
        header = f"while {_expression(condition, 'condition')}:"
    elif kind_text == "for":
        header = f"for {_identifier(target, 'target')} in {_expression(iterator, 'iterator')}:"
    else:
        _fail("E001", f"unsupported loop kind: {kind_text}")
    source = _check_source("\n".join([header, *_indent(statements, 1)]) + "\n")
    return {
        "source": source,
        "kind": kind_text,
        "body_lines": statements,
        "iterator": iterator,
    }


def pygen_exception_guard(
    body: Sequence[str],
    *,
    handlers: Sequence[tuple[str, str | None, Sequence[str]]] = (),
    finally_lines: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit a ``try`` / ``except`` / ``finally`` statement."""
    statements = [_statement(line, "body line") for line in body]
    lines = ["try:", *_indent(statements, 1)]
    handler_info: list[dict[str, Any]] = []
    for exception, alias, handler_body in handlers:
        exception_source = _identifier(exception, "exception", dotted=True)
        bound = _identifier(alias, "alias") if alias else None
        header = f"except {exception_source}"
        if bound:
            header += f" as {bound}"
        lines.append(f"{header}:")
        lines.extend(
            _indent([_statement(line, "handler line") for line in handler_body], 1)
        )
        handler_info.append({"exception": exception_source, "alias": bound})
    has_finally = bool(tuple(finally_lines))
    if has_finally:
        lines.append("finally:")
        lines.extend(
            _indent([_statement(line, "finally line") for line in finally_lines], 1)
        )
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "handlers": handler_info,
        "has_finally": has_finally,
    }


def pygen_dataclass_field(
    name: str, annotation: str, *, default: str | None = None
) -> dict[str, Any]:
    """Emit one annotated dataclass field line."""
    field_name = _identifier(name, "field name")
    annotation_source = _text(annotation, "annotation")
    has_default = default is not None
    if has_default:
        source = (
            f"{field_name}: {annotation_source} = {_expression(default, 'default')}"
        )
    else:
        source = f"{field_name}: {annotation_source}"
    _statement(source, "dataclass field")
    return {
        "source": source,
        "name": field_name,
        "annotation": annotation_source,
        "has_default": has_default,
    }


def pygen_decorator(name: str, *, arguments: Sequence[str] = ()) -> dict[str, Any]:
    """Emit a dotted decorator with positional or keyword arguments."""
    decorator_name = _identifier(name, "decorator", dotted=True)
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        _fail("E003", "arguments must be a sequence of expressions")
    values = [_decorator_argument(argument) for argument in arguments]
    source = (
        f"@{decorator_name}({', '.join(values)})" if values else f"@{decorator_name}"
    )
    _check_source(source + "\ndef _decorated() -> None:\n    pass\n")
    return {"source": source, "name": decorator_name, "arguments": values}


def pygen_docstring_block(text: str, *, indent: int = 0) -> dict[str, Any]:
    """Emit a safely escaped triple-quoted docstring block."""
    value = _text(text, "text")
    escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        _fail("E003", "indent must be a non-negative int")
    source = "\n".join(_indent([f'"""{escaped}"""'], indent))
    return {"source": source, "text": value, "escaped": escaped != value, "lines": 1}


def pygen_enum_variant_constructor(
    enum_name: str, variant: astn.EnumVariant
) -> dict[str, Any]:
    """Emit a static constructor that builds a tagged enum tuple."""
    _identifier(enum_name, "enum name")
    if not isinstance(variant, astn.EnumVariant):
        _fail("E003", "variant must be an EnumVariant node")
    tag = _identifier(variant.name, "variant name")
    arity = 0 if variant.fields is None else len(variant.fields)
    parameters = ", ".join(f"value{index}" for index in range(arity))
    if arity == 0:
        encoded = f"({json.dumps(tag)},)"
    elif arity == 1:
        encoded = f"({json.dumps(tag)}, value0)"
    else:
        values = ", ".join(f"value{index}" for index in range(arity))
        encoded = f"({json.dumps(tag)}, {values})"
    source = "\n".join(
        [
            "@staticmethod",
            f"def make_{tag}({parameters}) -> tuple[object, ...]:",
            f"    return {encoded}",
        ]
    )
    _check_source(source + "\n")
    return {"source": source, "name": f"make_{tag}", "variant": tag, "arity": arity}


def pygen_macro_call_source(call: astn.MacroCall) -> dict[str, Any]:
    """Emit a deterministic expression for a supported Polyglot macro."""
    if not isinstance(call, astn.MacroCall):
        _fail("E003", "call must be a MacroCall node")
    name = _text(call.path, "macro path").split("::")[-1].removesuffix("!")
    if isinstance(call.args, str):
        _fail("E003", "opaque macro token streams cannot be emitted safely")
    arguments = [_render_expr(argument) for argument in call.args]
    if name == "vec":
        source = "[" + ", ".join(arguments) + "]"
    elif name == "assert":
        if len(arguments) != 1:
            _fail("E001", "assert! expects exactly one argument")
        source = f"_assert({arguments[0]})"
    elif name in ("assert_eq", "assert_ne"):
        if len(arguments) != 2:
            _fail("E001", f"{name}! expects exactly two arguments")
        helper = "_assert_eq" if name == "assert_eq" else "_assert_ne"
        source = f"{helper}({arguments[0]}, {arguments[1]})"
    elif name == "format":
        if not arguments:
            _fail("E001", "format! expects a format string")
        source = f"{arguments[0]}.format({', '.join(arguments[1:])})"
    elif name in ("print", "println"):
        source = f"print({', '.join(arguments)})"
    else:
        _fail("E002", f"unsupported macro: {name}")
    _expression(source, "macro expansion")
    return {"source": source, "macro": name, "argument_count": len(arguments)}


def pygen_module_exports(names: Sequence[str]) -> dict[str, Any]:
    """Emit a deterministic ``__all__`` declaration for exported names."""
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        _fail("E003", "names must be a sequence of identifiers")
    exports = list(dict.fromkeys(_identifier(name, "export name") for name in names))
    lines = ["__all__: tuple[str, ...] = ("]
    lines.extend(_indent([f"{json.dumps(name)}," for name in exports], 1))
    lines.append(")")
    source = _check_source("\n".join(lines) + "\n")
    return {"source": source, "names": exports, "count": len(exports)}


def pygen_native_binding_stub(
    module_name: str,
    *,
    symbols: Sequence[str] = (),
    fallback_module: str | None = None,
) -> dict[str, Any]:
    """Emit a stub that binds native symbols from an explicit registry.

    The generated function never imports a native extension itself: callers
    pass a trusted registry mapping, which keeps untrusted modules out of the
    binding path.
    """
    name = _identifier(module_name, "module_name", dotted=True)
    symbol_names = list(
        dict.fromkeys(_identifier(symbol, "symbol") for symbol in symbols)
    )
    fallback = (
        None
        if fallback_module is None
        else _identifier(fallback_module, "fallback_module", dotted=True)
    )
    lines = [
        f"NATIVE_MODULE = {json.dumps(name)}",
        "NATIVE_SYMBOLS: tuple[str, ...] = (",
        *(_indent([f"{json.dumps(symbol)}," for symbol in symbol_names], 1)),
        ")",
        "",
        "def bind_native(registry: dict[str, object]) -> dict[str, object]:",
        '    """Bind the allowlisted native symbols from a trusted registry."""',
        "    missing = [name for name in NATIVE_SYMBOLS if name not in registry]",
        "    if missing:",
        '        raise KeyError(", ".join(missing))',
        "    return {name: registry[name] for name in NATIVE_SYMBOLS}",
    ]
    if fallback is not None:
        lines.extend(
            [
                "",
                f"FALLBACK_MODULE = {json.dumps(fallback)}",
                "",
                "def bind_fallback(fallbacks: dict[str, object]) -> dict[str, object]:",
                '    """Bind pure-Python fallbacks for the same symbol list."""',
                "    return {name: fallbacks[name] for name in NATIVE_SYMBOLS if name in fallbacks}",
            ]
        )
    source = _check_source("\n".join(lines) + "\n")
    return {
        "source": source,
        "module": name,
        "symbols": symbol_names,
        "fallback_module": fallback,
    }
