"""Deterministic high-level IR helpers for the FRAME 3 rollout.

This module operates on the typed HIR dataclasses already defined by
:mod:`compiler.ir.framework`.  It never emits code, allocates native memory, or
evaluates user expressions: every result is a plain dictionary (or a fresh HIR
dataclass nested inside one) derived from the structure of the input program.
Diagnostics use the structured ``E001``–``E008`` surface via
:class:`compiler.frontend.errors.PolyglotError`.

Diagnostic conventions used here:

* ``E001`` – a resolved HIR type is ill-formed or cannot be unified;
* ``E003`` – API misuse (wrong node type, malformed replacement table);
* ``E004`` – source text could not be parsed into a block;
* ``E006`` – a required function or symbol is absent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any

import compiler.core.ast_nodes as ast
from compiler.frontend.errors import PolyglotError
from compiler.ir.framework import (
    BOOL_TYPE,
    I64_TYPE,
    UNKNOWN_TYPE,
    HirBranch,
    HirExpr,
    HirFunction,
    HirModule,
    HirStmt,
    HirType,
    ir_lower_ast_to_hir,
)
from compiler.core.lexer import LexError, lex
from compiler.core.parser import ParseError, parse

__all__ = [
    "hir_alpha_rename",
    "hir_bind_locals",
    "hir_call_graph",
    "hir_collect_symbols",
    "hir_const_env",
    "hir_desugar_match",
    "hir_drop_scopes",
    "hir_expr_depth",
    "hir_function_summary",
    "hir_literal_table",
    "hir_lower_block",
    "hir_lower_loops",
    "hir_lower_short_circuit",
    "hir_module_fingerprint",
    "hir_monomorph_plan",
    "hir_normalize_spans",
    "hir_pattern_bindings",
    "hir_ref_types",
    "hir_resolve_calls",
    "hir_rewrite_ops",
    "hir_stmt_kinds",
    "hir_substitute",
    "hir_type_expr",
    "hir_unify_types",
    "hir_validate_module",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMERIC = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "usize",
        "f32",
        "f64",
    }
)
_NON_BINDING = frozenset({"_", "self", "Self", "true", "false", "mut", "ref", "box"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, **context: Any) -> None:
    """Raise a structured IR diagnostic with optional context keys."""
    details = ", ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
    raise PolyglotError(
        code,
        f"{message} ({details})" if details else message,
        phase="ir",
    )


def _require_module(module: object) -> HirModule:
    """Return *module* or raise E003 when it is not a HIR module."""
    if not isinstance(module, HirModule):
        _fail("E003", "expected a HirModule", value=type(module).__name__)
    return module


def _require_expr(expr: object) -> HirExpr:
    """Return *expr* or raise E003 when it is not a HIR expression."""
    if not isinstance(expr, HirExpr):
        _fail("E003", "expected a HirExpr", value=type(expr).__name__)
    return expr


def _require_type(ty: object) -> HirType:
    """Return *ty* or raise E001 when it is not a HirType."""
    if not isinstance(ty, HirType):
        _fail("E001", "expected a HirType", value=type(ty).__name__)
    return ty


def _walk(expr: HirExpr) -> tuple[HirExpr, ...]:
    """Yield an expression and every nested HIR expression in source order."""
    result = [expr]
    for arg in expr.args:
        result.extend(_walk(arg))
    for branch in expr.branches:
        if branch.condition is not None:
            result.extend(_walk(branch.condition))
        result.extend(_walk(branch.body))
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt) and stmt.value is not None:
            result.extend(_walk(stmt.value))
    return tuple(result)


def _map_expr(expr: HirExpr, transform: Any) -> HirExpr:
    """Rebuild an expression bottom-up, applying *transform* to each node."""
    args = tuple(_map_expr(arg, transform) for arg in expr.args)
    branches = tuple(
        HirBranch(
            None
            if branch.condition is None
            else _map_expr(branch.condition, transform),
            _map_expr(branch.body, transform),
            dict(branch.meta),
        )
        for branch in expr.branches
    )
    stmts: list[HirStmt] = []
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt):
            value = None if stmt.value is None else _map_expr(stmt.value, transform)
            stmts.append(replace(stmt, value=value))
    meta = dict(expr.meta)
    if stmts or "stmts" in meta:
        meta["stmts"] = tuple(stmts)
    return transform(replace(expr, args=args, branches=branches, meta=meta))


def _statements(expr: HirExpr) -> tuple[HirStmt, ...]:
    """Collect HIR statements from every nested block in deterministic order."""
    result: list[HirStmt] = []
    for node in _walk(expr):
        for stmt in node.meta.get("stmts", ()):
            if isinstance(stmt, HirStmt):
                result.append(stmt)
    return tuple(result)


def _calls(expr: HirExpr) -> tuple[HirExpr, ...]:
    """Collect call-like HIR nodes in deterministic order."""
    return tuple(node for node in _walk(expr) if node.kind in {"call", "await", "try"})


def _binding_names(stmt: HirStmt) -> tuple[str, ...]:
    """Names introduced by a let statement, including destructured bindings."""
    if stmt.kind != "let":
        return ()
    names = stmt.meta.get("bindings", ())
    if isinstance(names, (tuple, list)) and all(
        isinstance(name, str) for name in names
    ):
        result = tuple(names)
    elif stmt.name is not None:
        result = (stmt.name,)
    else:
        result = ()
    return result or (() if stmt.name is None else (stmt.name,))


def _text(ty: HirType) -> str:
    """Stable printable type spelling."""
    return ty.text


def _literal_kind(value: Any) -> str:
    """Classify a compile-time literal value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "unknown"


def _unify(left: HirType, right: HirType) -> tuple[HirType, bool, str]:
    """Internal type unification returning the type, compatibility, and rule."""
    if left == right:
        return left, True, "equal"
    if left == UNKNOWN_TYPE:
        return right, True, "left_unknown"
    if right == UNKNOWN_TYPE:
        return left, True, "right_unknown"
    if left.name in {"f32", "f64"} or right.name in {"f32", "f64"}:
        return HirType("f64"), True, "float_promotion"
    if left.name in _NUMERIC and right.name in _NUMERIC:
        return left, True, "numeric_width"
    return left, False, "mismatch"


def _infer_type(expr: HirExpr, scope: dict[str, HirType]) -> HirType:
    """Compute one HIR expression type bottom-up."""
    if expr.kind == "literal":
        value = expr.value
        if isinstance(value, bool):
            return BOOL_TYPE
        if isinstance(value, int):
            return I64_TYPE
        if isinstance(value, float):
            return HirType("f64")
        if isinstance(value, str):
            return HirType("str")
        return expr.ty
    if expr.kind == "var":
        return scope.get(str(expr.value), expr.ty)
    if expr.kind in {"bin", "logical_and", "logical_or"}:
        op = str(expr.value)
        if op in {"==", "!=", "<", "<=", ">", ">="}:
            return BOOL_TYPE
        left = _infer_type(expr.args[0], scope) if expr.args else UNKNOWN_TYPE
        right = _infer_type(expr.args[1], scope) if len(expr.args) > 1 else UNKNOWN_TYPE
        return _unify(left, right)[0]
    if expr.kind == "unop" and str(expr.value) == "!":
        return BOOL_TYPE
    if expr.kind in {"if", "match"}:
        types = [_infer_type(branch.body, scope) for branch in expr.branches]
        result = types[0] if types else UNKNOWN_TYPE
        for ty in types[1:]:
            result = _unify(result, ty)[0]
        return result
    if expr.kind == "block":
        return _infer_type(expr.args[0], scope) if expr.args else expr.ty
    if expr.kind in {"assign", "guard", "break", "continue", "return"}:
        return expr.ty
    return expr.ty


def _binding_names_from_expr(expr: HirExpr) -> tuple[str, ...]:
    """Yield every let-bound name in one HIR tree."""
    return tuple(name for stmt in _statements(expr) for name in _binding_names(stmt))


def _cyclic_components(
    graph: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic sorted components containing a directed cycle."""
    color: dict[str, int] = {}
    active: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(name: str) -> None:
        color[name] = 1
        active.append(name)
        for target in graph.get(name, ()):
            if color.get(target, 0) == 0:
                visit(target)
            elif color.get(target) == 1 and target in active:
                cycles.add(tuple(sorted(active[active.index(target) :])))
        active.pop()
        color[name] = 2

    for name in sorted(graph):
        if color.get(name, 0) == 0:
            visit(name)
    return tuple(sorted(cycles))


# ---------------------------------------------------------------------------
# Public HIR helpers
# ---------------------------------------------------------------------------


def hir_lower_block(source: str | ast.Block) -> dict[str, Any]:
    """Lower a block of Polyglot source (or a parsed ``ast.Block``) into HIR.

    A bare block is wrapped in a synthetic function before using the real
    frontend lowering pipeline; this guarantees identical expression, type, and
    span handling without synthesising any AST node by hand.  Parse and lex
    failures are reported as E004.
    """
    if isinstance(source, ast.Block):
        program = ast.Program(
            line=1,
            col=1,
            items=[
                ast.FnDecl(
                    line=source.line,
                    col=source.col,
                    name="__block__",
                    params=[],
                    ret=None,
                    body=source,
                    span=source.span,
                )
            ],
        )
    elif isinstance(source, str):
        try:
            program = parse(lex(f"fn __block__() {{ {source} }}"))
        except (LexError, ParseError) as error:
            _fail("E004", "could not parse block source", parse_error=str(error))
    else:
        _fail("E003", "source must be str or ast.Block", value=type(source).__name__)

    module = ir_lower_ast_to_hir(program)
    function = module.functions.get("__block__")
    if function is None:
        _fail("E004", "lowering produced no synthetic block function")
    expr = function.body
    names = [name for stmt in _statements(expr) for name in _binding_names(stmt)]
    tail = expr.args[0] if expr.args else None
    return {
        "kind": expr.kind,
        "expr": expr,
        "bindings": tuple(names),
        "tail_type": _text(tail.ty if tail is not None else expr.ty),
        "statement_count": len(_statements(expr)),
    }


def hir_bind_locals(expr: HirExpr) -> dict[str, Any]:
    """Collect local bindings and their types from an HIR expression tree."""
    root = _require_expr(expr)
    bindings: list[dict[str, Any]] = []
    for stmt in _statements(root):
        for name in _binding_names(stmt):
            ty = _require_type(stmt.ty)
            bindings.append(
                {
                    "name": name,
                    "type": _text(ty),
                    "mutable": bool(stmt.mutable),
                    "scope": stmt.meta.get("scope"),
                    "span": stmt.span,
                }
            )
    names = tuple(binding["name"] for binding in bindings)
    seen: set[str] = set()
    shadowed: list[str] = []
    for name in names:
        if name in seen:
            shadowed.append(name)
        seen.add(name)
    return {
        "bindings": tuple(bindings),
        "names": names,
        "shadowed": tuple(dict.fromkeys(shadowed)),
        "count": len(bindings),
    }


def hir_desugar_match(expr: HirExpr) -> dict[str, Any]:
    """Rewrite an HIR ``match`` into a right-nested chain of ``if`` expressions."""
    root = _require_expr(expr)
    if root.kind != "match":
        _fail("E003", "expected a match expression", kind=root.kind)
    if not root.args:
        _fail("E001", "match expression has no scrutinee")
    scrutinee = root.args[0]
    current = root.branches[-1].body if root.branches else HirExpr("block", root.ty)
    for branch in reversed(root.branches[:-1]):
        condition = branch.condition
        if condition is None:
            current = branch.body
            continue
        then_branch = HirBranch(None, branch.body, {"kind": "then"})
        else_branch = HirBranch(None, current, {"kind": "else"})
        current = HirExpr(
            "if",
            root.ty,
            args=(condition,),
            branches=(then_branch, else_branch),
            span=branch.body.span,
            meta={"desugared_match": True, "pattern": branch.meta.get("pattern")},
        )
    return {
        "expr": current,
        "scrutinee": scrutinee,
        "arms": len(root.branches),
        "wildcard": any(branch.condition is None for branch in root.branches),
    }


def hir_type_expr(
    expr: HirExpr, env: dict[str, HirType] | None = None
) -> dict[str, Any]:
    """Infer the type of an HIR expression from its shape and local environment."""
    root = _require_expr(expr)
    scope: dict[str, HirType] = {}
    if env is not None:
        if not isinstance(env, dict):
            _fail("E003", "env must be a dict of HirType values")
        for name, ty in env.items():
            if not isinstance(name, str):
                _fail("E003", "env keys must be strings", key=repr(name))
            scope[name] = _require_type(ty)
    inferred = _infer_type(root, scope)
    return {
        "type": inferred,
        "text": _text(inferred),
        "confident": inferred != UNKNOWN_TYPE,
        "env": dict(scope),
    }


def hir_unify_types(left: HirType, right: HirType) -> dict[str, Any]:
    """Unify two HIR types using deterministic scalar promotion rules."""
    a = _require_type(left)
    b = _require_type(right)
    unified, compatible, rule = _unify(a, b)
    return {
        "type": unified,
        "text": _text(unified),
        "compatible": compatible,
        "rule": rule,
    }


def hir_resolve_calls(
    module: HirModule, *, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Resolve every HIR call against local functions and modeled builtins."""
    root = _require_module(module)
    if not isinstance(required, (tuple, list)) or not all(
        isinstance(name, str) for name in required
    ):
        _fail("E003", "required must be a tuple of function names")
    names = set(root.functions)
    for name in required:
        if name not in names:
            _fail("E006", "required function is missing", name=name)
    builtins = frozenset({"abs", "len", "max", "min", "sqrt", "print", "println"})
    resolved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    edges: dict[str, tuple[str, ...]] = {}
    for name, function in root.functions.items():
        callees = sorted({str(node.value) for node in _calls(function.body)})
        edges[name] = tuple(callees)
        for callee in callees:
            if callee in names:
                resolved.append({"function": name, "callee": callee, "kind": "local"})
            elif callee in builtins:
                resolved.append({"function": name, "callee": callee, "kind": "builtin"})
            else:
                unresolved.append(
                    {"function": name, "callee": callee, "kind": "unknown"}
                )
    return {
        "resolved": tuple(resolved),
        "unresolved": tuple(unresolved),
        "edges": edges,
        "count": len(resolved) + len(unresolved),
    }


def hir_rewrite_ops(module: HirModule, replacements: dict[str, str]) -> dict[str, Any]:
    """Replace operator spellings throughout a HIR module using an explicit table."""
    root = _require_module(module)
    if not isinstance(replacements, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in replacements.items()
    ):
        _fail("E003", "replacements must map str to str")
    changed: list[str] = []

    def transform(node: HirExpr) -> HirExpr:
        if node.kind not in {"bin", "unop", "logical_and", "logical_or"}:
            return node
        op = str(node.value)
        if op not in replacements or replacements[op] == op:
            return node
        return replace(node, value=replacements[op])

    functions: dict[str, HirFunction] = {}
    for name, function in root.functions.items():
        body = _map_expr(function.body, transform)
        for node in _walk(function.body):
            if node.kind in {"bin", "unop"} and str(node.value) in replacements:
                changed.append(f"{name}:{node.value}")
        functions[name] = replace(function, body=body)
    return {
        "module": HirModule(functions, root.source),
        "rewritten": tuple(sorted(set(changed))),
        "count": len(changed),
    }


def hir_collect_symbols(module: HirModule) -> dict[str, Any]:
    """Collect defined and referenced identifiers across a HIR module."""
    root = _require_module(module)
    defined: set[str] = set(root.functions)
    referenced: set[str] = set()
    counts: dict[str, int] = {}
    for function in root.functions.values():
        for param in function.params:
            defined.add(param.name)
        for name in _binding_names_from_expr(function.body):
            defined.add(name)
        for node in _walk(function.body):
            if node.kind == "var":
                name = str(node.value)
                referenced.add(name)
                counts[name] = counts.get(name, 0) + 1
    return {
        "defined": tuple(sorted(defined)),
        "referenced": tuple(sorted(referenced)),
        "undefined": tuple(sorted(referenced - defined)),
        "use_counts": {name: counts[name] for name in sorted(counts)},
        "count": len(counts),
    }


def hir_stmt_kinds(module: HirModule) -> dict[str, Any]:
    """Count HIR statement kinds in deterministic, alphabetically sorted order."""
    root = _require_module(module)
    counts: dict[str, int] = {}
    for function in root.functions.values():
        for stmt in _statements(function.body):
            counts[stmt.kind] = counts.get(stmt.kind, 0) + 1
    return {
        "counts": {kind: counts[kind] for kind in sorted(counts)},
        "total": sum(counts.values()),
        "functions": tuple(sorted(root.functions)),
    }


def hir_const_env(expr: HirExpr) -> dict[str, Any]:
    """Extract literal-valued let bindings as a deterministic constant environment."""
    root = _require_expr(expr)
    constants: dict[str, Any] = {}
    types: dict[str, str] = {}
    for stmt in _statements(root):
        if stmt.kind != "let" or stmt.value is None or stmt.value.kind != "literal":
            continue
        for name in _binding_names(stmt):
            constants[name] = stmt.value.value
            types[name] = _text(stmt.ty)
    return {
        "constants": {name: constants[name] for name in sorted(constants)},
        "types": {name: types[name] for name in sorted(types)},
        "count": len(constants),
    }


def hir_substitute(expr: HirExpr, replacements: dict[str, HirExpr]) -> dict[str, Any]:
    """Substitute HIR variable references with supplied HIR expressions."""
    root = _require_expr(expr)
    if not isinstance(replacements, dict):
        _fail("E003", "replacements must be a dict")
    mapping: dict[str, HirExpr] = {}
    for name, value in replacements.items():
        if not isinstance(name, str):
            _fail("E003", "replacement keys must be strings")
        mapping[name] = _require_expr(value)
    count = sum(
        1 for node in _walk(root) if node.kind == "var" and str(node.value) in mapping
    )

    def transform(node: HirExpr) -> HirExpr:
        if node.kind != "var":
            return node
        return mapping.get(str(node.value), node)

    return {
        "expr": _map_expr(root, transform),
        "names": tuple(sorted(mapping)),
        "count": count,
    }


def hir_alpha_rename(module: HirModule, prefix: str = "r") -> dict[str, Any]:
    """Rename every local binding and use with a deterministic generated prefix."""
    root = _require_module(module)
    if not isinstance(prefix, str) or not _IDENTIFIER.fullmatch(prefix):
        _fail("E003", "prefix must be an identifier", prefix=prefix)
    functions: dict[str, HirFunction] = {}
    mappings: dict[str, dict[str, str]] = {}
    for name, function in root.functions.items():
        names = sorted(
            {param.name for param in function.params}
            | set(_binding_names_from_expr(function.body))
        )
        mapping = {old: f"{prefix}{index}_{old}" for index, old in enumerate(names, 1)}
        params = tuple(
            replace(param, name=mapping[param.name]) for param in function.params
        )

        def transform(node: HirExpr, mapping: dict[str, str] = mapping) -> HirExpr:
            if node.kind != "var":
                return node
            old = str(node.value)
            return (
                replace(node, value=mapping.get(old, old)) if old in mapping else node
            )

        body = _map_expr(function.body, transform)
        functions[name] = replace(function, params=params, body=body)
        mappings[name] = dict(mapping)
    return {
        "module": HirModule(functions, root.source),
        "renames": mappings,
        "count": sum(len(mapping) for mapping in mappings.values()),
    }


def hir_normalize_spans(expr: HirExpr) -> dict[str, Any]:
    """Collect and canonically order all source spans in a HIR expression."""
    root = _require_expr(expr)
    spans: list[tuple[int, int]] = []
    missing = 0
    kinds: dict[str, int] = {}
    for node in _walk(root):
        kinds[node.kind] = kinds.get(node.kind, 0) + 1
        if node.span is None:
            missing += 1
        else:
            start, end = node.span
            spans.append((int(start), int(end)))
    canonical = tuple(sorted(set(spans)))
    return {
        "spans": canonical,
        "missing": missing,
        "distinct": len(canonical),
        "kinds": {kind: kinds[kind] for kind in sorted(kinds)},
    }


def hir_expr_depth(expr: HirExpr) -> dict[str, Any]:
    """Measure maximum expression nesting depth, node count, and branch width."""
    root = _require_expr(expr)

    def depth(node: HirExpr) -> int:
        children = [depth(arg) for arg in node.args]
        for branch in node.branches:
            if branch.condition is not None:
                children.append(depth(branch.condition))
            children.append(depth(branch.body))
        for stmt in node.meta.get("stmts", ()):
            if isinstance(stmt, HirStmt) and stmt.value is not None:
                children.append(depth(stmt.value))
        return 1 + max(children, default=0)

    nodes = _walk(root)
    return {
        "depth": depth(root),
        "nodes": len(nodes),
        "max_branch_width": max((len(node.branches) for node in nodes), default=0),
        "kinds": tuple(sorted({node.kind for node in nodes})),
    }


def hir_pattern_bindings(expr: HirExpr) -> dict[str, Any]:
    """List pattern strings and candidate binding names used by HIR match arms."""
    root = _require_expr(expr)
    patterns: list[str] = []
    by_arm: dict[int, tuple[str, ...]] = {}
    for node in _walk(root):
        if node.kind != "match":
            continue
        for branch in node.branches:
            arm = int(branch.meta.get("arm", len(patterns)))
            pattern = str(branch.meta.get("pattern", "_"))
            patterns.append(pattern)
            candidates = tuple(
                word
                for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pattern)
                if word not in _NON_BINDING and not word[0].isupper()
            )
            by_arm[arm] = tuple(dict.fromkeys(candidates))
    names = tuple(sorted({name for values in by_arm.values() for name in values}))
    return {
        "patterns": tuple(patterns),
        "bindings": names,
        "by_arm": {arm: by_arm[arm] for arm in sorted(by_arm)},
        "count": len(patterns),
    }


def hir_lower_short_circuit(expr: HirExpr) -> dict[str, Any]:
    """Desugar HIR ``&&`` / ``||`` nodes into explicit branch expressions."""
    root = _require_expr(expr)
    count = 0
    operators: list[str] = []

    def transform(node: HirExpr) -> HirExpr:
        nonlocal count
        if node.kind not in {"logical_and", "logical_or"} or len(node.args) != 2:
            return node
        left, right = node.args
        shortcut = node.kind == "logical_or"
        const = HirExpr("literal", BOOL_TYPE, value=shortcut)
        then_branch = HirBranch(None, right, {"kind": "then"})
        else_branch = HirBranch(None, const, {"kind": "else"})
        count += 1
        operators.append(str(node.value))
        return HirExpr(
            "if",
            BOOL_TYPE,
            args=(left,),
            branches=(then_branch, else_branch),
            span=node.span,
            meta={"short_circuit": str(node.value), "lowered": True},
        )

    lowered = _map_expr(root, transform)
    return {
        "expr": lowered,
        "rewritten": count,
        "operators": tuple(operators),
        "remaining": sum(
            1 for node in _walk(lowered) if node.kind in {"logical_and", "logical_or"}
        ),
    }


def hir_lower_loops(expr: HirExpr) -> dict[str, Any]:
    """Normalize HIR loop variants into a canonical annotated ``loop`` node."""
    root = _require_expr(expr)
    count = 0
    variants: list[str] = []

    def transform(node: HirExpr) -> HirExpr:
        nonlocal count
        if node.kind != "loop":
            return node
        variant = str(node.meta.get("loop", "loop"))
        meta = dict(node.meta)
        meta["loop"] = variant
        meta["has_condition"] = variant == "while"
        meta["bindings"] = tuple(meta.get("bindings", ()))
        meta["normalized"] = True
        count += 1
        variants.append(variant)
        return replace(node, meta=meta)

    return {
        "expr": _map_expr(root, transform),
        "rewritten": count,
        "variants": tuple(sorted(variants)),
    }


def hir_drop_scopes(module: HirModule) -> dict[str, Any]:
    """Annotate each HIR statement with its lexical block nesting depth."""
    root = _require_module(module)
    depths: dict[str, tuple[int, ...]] = {}
    maximum = 0
    functions: dict[str, HirFunction] = {}

    def transform(node: HirExpr, assigned: list[int], counter: list[int]) -> HirExpr:
        nonlocal maximum
        stmts: list[HirStmt] = []
        for stmt in node.meta.get("stmts", ()):
            if not isinstance(stmt, HirStmt):
                continue
            depth = counter[0]
            assigned.append(depth)
            maximum = max(maximum, depth)
            meta = dict(stmt.meta)
            meta["scope_depth"] = depth
            stmts.append(replace(stmt, meta=meta))
        if stmts or "stmts" in node.meta:
            meta = dict(node.meta)
            meta["stmts"] = tuple(stmts)
            return replace(node, meta=meta)
        return node

    def descend(node: HirExpr, assigned: list[int], counter: list[int]) -> HirExpr:
        result = transform(node, assigned, counter)
        args = tuple(descend(arg, assigned, counter) for arg in result.args)
        branches = tuple(
            HirBranch(
                None
                if branch.condition is None
                else descend(branch.condition, assigned, counter),
                descend(branch.body, assigned, counter),
                dict(branch.meta),
            )
            for branch in result.branches
        )
        return replace(result, args=args, branches=branches)

    for name, function in root.functions.items():
        assigned: list[int] = []
        counter = [0]
        body = descend(function.body, assigned, counter)
        functions[name] = replace(function, body=body)
        depths[name] = tuple(assigned)
    return {
        "module": HirModule(functions, root.source),
        "depths": depths,
        "max_depth": maximum,
        "annotated": sum(len(values) for values in depths.values()),
    }


def hir_call_graph(module: HirModule) -> dict[str, Any]:
    """Build a deterministic call graph and report its cyclic strongly connected sets."""
    root = _require_module(module)
    edges = hir_resolve_calls(root)["edges"]
    graph: dict[str, tuple[str, ...]] = {
        name: tuple(callee for callee in callees if callee in root.functions)
        for name, callees in edges.items()
    }
    reverse: dict[str, tuple[str, ...]] = {
        name: tuple(
            sorted(caller for caller, callees in graph.items() if name in callees)
        )
        for name in graph
    }
    cycles = _cyclic_components(graph)
    return {
        "edges": graph,
        "reverse": reverse,
        "cycles": cycles,
        "has_cycle": bool(cycles),
    }


def hir_function_summary(function: HirFunction) -> dict[str, Any]:
    """Summarize one lowered function with size, type, and call metadata."""
    if not isinstance(function, HirFunction):
        _fail("E003", "expected a HirFunction", value=type(function).__name__)
    nodes = _walk(function.body)
    callees = sorted({str(node.value) for node in _calls(function.body)})
    return {
        "name": function.name,
        "params": len(function.params),
        "return_type": _text(function.ret),
        "nodes": len(nodes),
        "depth": hir_expr_depth(function.body)["depth"],
        "pure": bool(function.pure),
        "recursive": bool(function.recursive),
        "callees": tuple(callees),
    }


def hir_literal_table(module: HirModule) -> dict[str, Any]:
    """List every literal in a HIR module with its kind, value, type, and span."""
    root = _require_module(module)
    literals: list[dict[str, Any]] = []
    for name, function in root.functions.items():
        for node in _walk(function.body):
            if node.kind != "literal":
                continue
            literals.append(
                {
                    "function": name,
                    "kind": _literal_kind(node.value),
                    "value": node.value,
                    "type": _text(node.ty),
                    "span": node.span,
                }
            )
    by_type: dict[str, int] = {}
    for literal in literals:
        by_type[literal["type"]] = by_type.get(literal["type"], 0) + 1
    return {
        "literals": tuple(literals),
        "by_type": {key: by_type[key] for key in sorted(by_type)},
        "count": len(literals),
    }


def hir_validate_module(module: HirModule) -> dict[str, Any]:
    """Check a HIR module for structural issues without raising for source problems."""
    root = _require_module(module)
    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    for name, function in root.functions.items():
        if not _IDENTIFIER.fullmatch(name):
            issues.append(
                {
                    "code": "E001",
                    "function": name,
                    "message": "invalid function identifier",
                }
            )
        if name in seen:
            issues.append(
                {"code": "E001", "function": name, "message": "duplicate function name"}
            )
        seen.add(name)
        if function.body.kind != "block":
            issues.append(
                {"code": "E003", "function": name, "message": "body is not a block"}
            )
        for param in function.params:
            if not _IDENTIFIER.fullmatch(param.name):
                issues.append(
                    {
                        "code": "E001",
                        "function": name,
                        "message": f"invalid parameter {param.name!r}",
                    }
                )
            _require_type(param.ty)
        for callee in sorted({str(node.value) for node in _calls(function.body)}):
            if callee not in root.functions and callee not in {
                "abs",
                "len",
                "max",
                "min",
                "sqrt",
            }:
                issues.append(
                    {
                        "code": "E006",
                        "function": name,
                        "message": f"unresolved call {callee!r}",
                    }
                )
    return {
        "ok": not issues,
        "issues": tuple(issues),
        "checked": ("identifiers", "unique_names", "block_bodies", "call_resolution"),
        "functions": len(root.functions),
    }


def hir_monomorph_plan(module: HirModule) -> dict[str, Any]:
    """Plan deterministic monomorphizations from explicit call-site type arguments."""
    root = _require_module(module)
    requests: list[dict[str, Any]] = []
    for name, function in root.functions.items():
        for node in _calls(function.body):
            args = node.meta.get("type_args", ())
            if not isinstance(args, (tuple, list)) or not args:
                continue
            types = tuple(str(getattr(arg, "text", arg)) for arg in args)
            requests.append(
                {
                    "caller": name,
                    "callee": str(node.value),
                    "args": types,
                    "key": f"{node.value}<{', '.join(types)}>",
                }
            )
    unique = tuple(sorted({request["key"] for request in requests}))
    return {
        "requests": tuple(requests),
        "unique": unique,
        "count": len(requests),
    }


def hir_ref_types(module: HirModule) -> dict[str, Any]:
    """Collect reference-type annotations and their mutability across a module."""
    root = _require_module(module)
    references: list[dict[str, Any]] = []
    for name, function in root.functions.items():
        candidates: list[tuple[str, HirType]] = [
            (f"{name}::{param.name}", param.ty) for param in function.params
        ]
        for stmt in _statements(function.body):
            candidates.append((f"{name}::{stmt.name}", stmt.ty))
        for node in _walk(function.body):
            candidates.append((f"{name}::{node.kind}", node.ty))
        for location, ty in candidates:
            if not ty.referenced:
                continue
            references.append(
                {
                    "location": location,
                    "type": _text(ty),
                    "mutable": bool(ty.mutable),
                    "inner": _text(HirType(ty.name, ty.args)),
                }
            )
    return {
        "references": tuple(references),
        "mutable_count": sum(1 for ref in references if ref["mutable"]),
        "shared_count": sum(1 for ref in references if not ref["mutable"]),
        "types": tuple(sorted({ref["type"] for ref in references})),
    }


def hir_module_fingerprint(module: HirModule) -> dict[str, Any]:
    """Compute a SHA-256 digest of the canonical structural form of a HIR module."""
    root = _require_module(module)

    def encode(expr: HirExpr) -> tuple[Any, ...]:
        stmts = tuple(
            (stmt.kind, stmt.name, _text(stmt.ty), stmt.mutable)
            for stmt in expr.meta.get("stmts", ())
            if isinstance(stmt, HirStmt)
        )
        return (
            expr.kind,
            repr(expr.value),
            _text(expr.ty),
            stmts,
            tuple(encode(arg) for arg in expr.args),
            tuple(
                (
                    None if branch.condition is None else encode(branch.condition),
                    encode(branch.body),
                    tuple(
                        sorted((key, repr(value)) for key, value in branch.meta.items())
                    ),
                )
                for branch in expr.branches
            ),
        )

    canonical = tuple(
        (
            name,
            tuple((param.name, _text(param.ty)) for param in function.params),
            _text(function.ret),
            function.pure,
            function.recursive,
            encode(function.body),
        )
        for name, function in sorted(root.functions.items())
    )
    digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()
    return {
        "digest": digest,
        "functions": len(root.functions),
        "nodes": sum(len(_walk(fn.body)) for fn in root.functions.values()),
    }
