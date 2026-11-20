"""FRAME 2 of the 100-function framework: types, traits, monomorphization.

Everything in this module works on the project's existing :mod:`ast_nodes`
tree and on already parsed Polyglot source.  The type checker is deliberately
small and deterministic: it propagates annotated types into expressions,
unifies call arguments with signatures, joins control-flow branches, and
records the result per expression so a caller can query it with
:meth:`InferenceResult.type_of`.

The monomorphization helpers wrap the existing :mod:`monomorph` pass.  They
make its placeholder-before-expansion behavior explicit, resolve recursive and
mutually recursive generics, canonicalize alias cache keys, prune unreachable
specializations, and emit a deterministic template-bloat report.

Diagnostic codes follow the repository's E001-E008 surface:

* ``E001`` - type/layout mismatch or unsatisfied trait bound;
* ``E003`` - invalid framework API input (unknown item, wrong type-arg count);
* ``E005-OC1`` / ``E005-OC2`` - direct / mutual occurs-check cycle;
* ``E002-INF1`` / ``E002-INF2`` - template inflation thresholds exceeded.

No platform-specific or native-backend code is used here; the helpers are
pure Python and run identically on every host.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import ast_nodes as ast
import monomorph
from runtime.diagnostics import PillarError

_DEFAULT_TYPE = "i64"
_VAR_PREFIX = "$T"
_INFERENCE_MISMATCH = "E001"
_API_ERROR = "E003"
_INFLATION_LIMIT = "E002"


# ---------------------------------------------------------------------------
# Small shared AST helpers
# ---------------------------------------------------------------------------


def _display(ty: ast.Type | None) -> str:
    """Return the canonical source-like spelling of ``ty``."""
    return ast.type_display_name(ty)


def _atomic(name: str, line: int = 0, col: int = 0) -> ast.TypeName:
    return ast.TypeName(line, col, name)


def _default_args(params: Sequence[ast.GenericParam]) -> list[ast.Type]:
    return [_atomic(_DEFAULT_TYPE) for _ in params]


def _node_children(node: object) -> Iterator[object]:
    """Yield direct dataclass children of an AST node (including nested lists)."""
    if isinstance(node, Mapping):
        yield from node.values()
        return
    if isinstance(node, (list, tuple, set, frozenset)):
        yield from node
        return
    if not is_dataclass(node) or isinstance(node, type):
        return
    for field_info in fields(node):
        if field_info.name in {"line", "col", "span", "source_lines"}:
            continue
        yield getattr(node, field_info.name)


def _walk(value: object) -> Iterator[ast.Node]:
    """Depth-first walk over every :class:`ast.Node` reachable from ``value``."""
    if isinstance(value, ast.Node):
        yield value
    for child in _node_children(value):
        yield from _walk(child)


def _iter_items(items: Iterable[ast.Item]) -> Iterator[ast.Item]:
    """Yield items, descending into inline modules."""
    for item in items:
        yield item
        if isinstance(item, ast.ModDecl) and item.items is not None:
            yield from _iter_items(item.items)


def _substitute_type(
    ty: ast.Type | None, mapping: Mapping[str, ast.Type]
) -> ast.Type | None:
    """Substitute bare type-parameter names without instantiating generics."""
    if ty is None:
        return None
    if isinstance(ty, ast.TypeName):
        if not ty.args and ty.name in mapping:
            return _substitute_type(mapping[ty.name], mapping)
        return ast.TypeName(
            ty.line,
            ty.col,
            ty.name,
            [
                arg
                for arg in (_substitute_type(item, mapping) for item in ty.args)
                if arg
            ],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeRef):
        return ast.TypeRef(
            ty.line,
            ty.col,
            _substitute_type(ty.inner, mapping),
            ty.mutable,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeSlice):
        return ast.TypeSlice(
            ty.line, ty.col, _substitute_type(ty.inner, mapping), span=ty.span
        )
    if isinstance(ty, ast.TypeArray):
        return ast.TypeArray(
            ty.line,
            ty.col,
            _substitute_type(ty.inner, mapping),
            ty.size,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeTuple):
        return ast.TypeTuple(
            ty.line,
            ty.col,
            [item for item in (_substitute_type(e, mapping) for e in ty.elems) if item],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeFn):
        return ast.TypeFn(
            ty.line,
            ty.col,
            [
                item
                for item in (_substitute_type(p, mapping) for p in ty.params)
                if item
            ],
            _substitute_type(ty.ret, mapping),
            span=ty.span,
        )
    return ty


def _type_vars(ty: ast.Type | None) -> set[str]:
    """Collect free type-variable names from a type expression."""
    if ty is None:
        return set()
    if isinstance(ty, ast.TypeName):
        if not ty.args:
            return {ty.name}
        found: set[str] = set()
        for arg in ty.args:
            found.update(_type_vars(arg))
        return found
    if isinstance(ty, (ast.TypeRef, ast.TypeSlice, ast.TypeArray)):
        return _type_vars(ty.inner)
    if isinstance(ty, ast.TypeTuple):
        found = set()
        for item in ty.elems:
            found.update(_type_vars(item))
        return found
    if isinstance(ty, ast.TypeFn):
        found = set()
        for param in ty.params:
            found.update(_type_vars(param))
        found.update(_type_vars(ty.ret))
        return found
    return set()


def _is_var(ty: ast.Type | None) -> bool:
    return (
        isinstance(ty, ast.TypeName) and not ty.args and ty.name.startswith(_VAR_PREFIX)
    )


def _resolve_type(
    ty: ast.Type | None,
    substitution: Mapping[str, ast.Type],
    seen: frozenset[str] = frozenset(),
) -> ast.Type | None:
    """Apply ``substitution`` repeatedly, stopping at a cyclic binding."""
    if ty is None:
        return None
    if isinstance(ty, ast.TypeName) and not ty.args and ty.name in substitution:
        if ty.name in seen:
            return ty
        return _resolve_type(substitution[ty.name], substitution, seen | {ty.name})
    if isinstance(ty, ast.TypeName):
        return ast.TypeName(
            ty.line,
            ty.col,
            ty.name,
            [
                resolved
                for arg in ty.args
                if (resolved := _resolve_type(arg, substitution, seen)) is not None
            ],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeRef):
        return ast.TypeRef(
            ty.line,
            ty.col,
            _resolve_type(ty.inner, substitution, seen),
            ty.mutable,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeSlice):
        return ast.TypeSlice(
            ty.line, ty.col, _resolve_type(ty.inner, substitution, seen), span=ty.span
        )
    if isinstance(ty, ast.TypeArray):
        return ast.TypeArray(
            ty.line,
            ty.col,
            _resolve_type(ty.inner, substitution, seen),
            ty.size,
            span=ty.span,
        )
    if isinstance(ty, ast.TypeTuple):
        return ast.TypeTuple(
            ty.line,
            ty.col,
            [
                resolved
                for item in ty.elems
                if (resolved := _resolve_type(item, substitution, seen)) is not None
            ],
            span=ty.span,
        )
    if isinstance(ty, ast.TypeFn):
        return ast.TypeFn(
            ty.line,
            ty.col,
            [
                resolved
                for param in ty.params
                if (resolved := _resolve_type(param, substitution, seen)) is not None
            ],
            _resolve_type(ty.ret, substitution, seen),
            span=ty.span,
        )
    return ty


def _match_type(
    pattern: ast.Type | None,
    concrete: ast.Type | None,
    mapping: dict[str, ast.Type],
) -> bool:
    """Match a generic impl pattern such as ``Box<T>`` against ``Box<i64>``."""
    if pattern is None or concrete is None:
        return pattern is concrete
    concrete = _resolve_type(concrete, mapping)
    if isinstance(pattern, ast.TypeName) and not pattern.args:
        if pattern.name in mapping:
            return _display(mapping[pattern.name]) == _display(concrete)
        if pattern.name.startswith(_VAR_PREFIX) or pattern.name == "Self":
            mapping[pattern.name] = concrete
            return True
        return isinstance(concrete, ast.TypeName) and pattern.name == concrete.name
    if isinstance(pattern, ast.TypeName) and isinstance(concrete, ast.TypeName):
        if pattern.name != concrete.name or len(pattern.args) != len(concrete.args):
            return False
        return all(
            _match_type(left, right, mapping)
            for left, right in zip(pattern.args, concrete.args)
        )
    if isinstance(pattern, ast.TypeRef) and isinstance(concrete, ast.TypeRef):
        return pattern.mutable == concrete.mutable and _match_type(
            pattern.inner, concrete.inner, mapping
        )
    if isinstance(pattern, ast.TypeSlice) and isinstance(concrete, ast.TypeSlice):
        return _match_type(pattern.inner, concrete.inner, mapping)
    if isinstance(pattern, ast.TypeTuple) and isinstance(concrete, ast.TypeTuple):
        return len(pattern.elems) == len(concrete.elems) and all(
            _match_type(left, right, mapping)
            for left, right in zip(pattern.elems, concrete.elems)
        )
    return _display(pattern) == _display(concrete)


def _count_nodes(value: object) -> int:
    """Count AST nodes in a tree (used for specialization size growth)."""
    if isinstance(value, ast.Node):
        return 1 + sum(_count_nodes(child) for child in _node_children(value))
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return sum(_count_nodes(child) for child in _node_children(value))
    return 0


# ---------------------------------------------------------------------------
# 11 - expression type inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeDiagnostic:
    """A deterministic type-checking diagnostic bound to an AST location."""

    code: str
    message: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class InferenceResult:
    """Types inferred for bindings and expressions in a program."""

    bindings: Mapping[str, ast.Type]
    expression_types: Mapping[int, ast.Type]
    function_bindings: Mapping[str, Mapping[str, ast.Type]]
    diagnostics: tuple[TypeDiagnostic, ...] = ()

    def type_of(self, node: ast.Node | None) -> ast.Type | None:
        """Return the inferred type of ``node``, or ``None`` when unknown."""
        if node is None:
            return None
        return self.expression_types.get(id(node))

    def binding(self, name: str, function: str | None = None) -> ast.Type | None:
        """Return a binding's type, optionally scoped to one function."""
        if function is not None:
            scoped = self.function_bindings.get(function, {})
            if name in scoped:
                return scoped[name]
        return self.bindings.get(name)


class _ExpressionInferrer:
    """Unification-based inference over one parsed program."""

    def __init__(self, program: ast.Program) -> None:
        self.program = program
        self.substitution: dict[str, ast.Type] = {}
        self.counter = 0
        self.current_fn = "<module>"
        self.bindings: dict[str, ast.Type] = {}
        self.function_bindings: dict[str, dict[str, ast.Type]] = defaultdict(dict)
        self.expression_types: dict[int, ast.Type] = {}
        self.diagnostics: list[TypeDiagnostic] = []
        self.signatures: dict[
            str, tuple[list[ast.Type | None], ast.Type | None, list[ast.GenericParam]]
        ] = {}
        self.struct_fields: dict[str, dict[str, ast.Type]] = {}
        self.struct_generics: dict[str, list[ast.GenericParam]] = {}
        self.enum_variants: dict[str, dict[str, list[ast.Type] | None]] = {}
        self.enum_generics: dict[str, list[ast.GenericParam]] = {}
        self.method_signatures: dict[
            tuple[str, str], tuple[list[ast.Type | None], ast.Type | None]
        ] = {}
        self._collect()

    def _collect(self) -> None:
        for item in _iter_items(self.program.items):
            if isinstance(item, ast.FnDecl):
                self.signatures[item.name] = (
                    [param.ty for param in item.params],
                    item.ret,
                    item.generics,
                )
                for stmt in _iter_stmts(item.body):
                    if isinstance(stmt, ast.ItemStmt) and isinstance(
                        stmt.item, ast.FnDecl
                    ):
                        self.signatures[stmt.item.name] = (
                            [param.ty for param in stmt.item.params],
                            stmt.item.ret,
                            stmt.item.generics,
                        )
            elif isinstance(item, ast.StructDecl):
                self.struct_fields[item.name] = {
                    name: ty for name, ty, _pub in item.fields
                }
                self.struct_generics[item.name] = item.generics
            elif isinstance(item, ast.EnumDecl):
                self.enum_variants[item.name] = {
                    variant.name: variant.fields for variant in item.variants
                }
                self.enum_generics[item.name] = item.generics
            elif isinstance(item, ast.ImplBlock) and isinstance(item.ty, ast.TypeName):
                for method in item.items:
                    if isinstance(method, ast.FnDecl):
                        self.method_signatures[(item.ty.name, method.name)] = (
                            [param.ty for param in method.params],
                            method.ret,
                        )

    def _fresh(self) -> ast.TypeName:
        self.counter += 1
        return _atomic(f"{_VAR_PREFIX}{self.counter}")

    def _report(self, code: str, message: str, node: ast.Node | None) -> None:
        self.diagnostics.append(
            TypeDiagnostic(
                code, message, getattr(node, "line", 0), getattr(node, "col", 0)
            )
        )

    def _bind(self, name: str, ty: ast.Type, node: ast.Node | None) -> None:
        resolved = _resolve_type(ty, self.substitution)
        if _is_var(resolved) and resolved.name == name:
            return  # T = T is a tautology, not an occurs-check failure.
        if name in _type_vars(resolved):
            self._report(
                "E005-OC1",
                f"type variable {name} occurs inside its own binding {_display(resolved)}",
                node,
            )
            return
        self.substitution[name] = resolved

    def _unify(
        self,
        left: ast.Type | None,
        right: ast.Type | None,
        node: ast.Node | None = None,
    ) -> None:
        lhs = _resolve_type(left, self.substitution)
        rhs = _resolve_type(right, self.substitution)
        if lhs is None or rhs is None:
            return
        if _is_var(lhs):
            self._bind(lhs.name, rhs, node)
            return
        if _is_var(rhs):
            self._bind(rhs.name, lhs, node)
            return
        if isinstance(lhs, ast.TypeName) and isinstance(rhs, ast.TypeName):
            if lhs.name == rhs.name and len(lhs.args) == len(rhs.args):
                for l_arg, r_arg in zip(lhs.args, rhs.args):
                    self._unify(l_arg, r_arg, node)
            else:
                self._report(
                    _INFERENCE_MISMATCH,
                    f"type mismatch: expected {_display(lhs)}, found {_display(rhs)}",
                    node,
                )
            return
        if type(lhs) is type(rhs):
            if isinstance(
                lhs, (ast.TypeRef, ast.TypeSlice, ast.TypeArray)
            ) and isinstance(rhs, (ast.TypeRef, ast.TypeSlice, ast.TypeArray)):
                self._unify(lhs.inner, rhs.inner, node)
                return
            if isinstance(lhs, ast.TypeTuple) and isinstance(rhs, ast.TypeTuple):
                if len(lhs.elems) == len(rhs.elems):
                    for l_item, r_item in zip(lhs.elems, rhs.elems):
                        self._unify(l_item, r_item, node)
                else:
                    self._report(
                        _INFERENCE_MISMATCH,
                        f"tuple arity mismatch: {_display(lhs)} vs {_display(rhs)}",
                        node,
                    )
                return
            if isinstance(lhs, ast.TypeFn) and isinstance(rhs, ast.TypeFn):
                if len(lhs.params) == len(rhs.params):
                    for l_param, r_param in zip(lhs.params, rhs.params):
                        self._unify(l_param, r_param, node)
                    self._unify(lhs.ret, rhs.ret, node)
                else:
                    self._report(
                        _INFERENCE_MISMATCH,
                        f"function arity mismatch: {_display(lhs)} vs {_display(rhs)}",
                        node,
                    )
                return
            if _display(lhs) == _display(rhs):
                return
        self._report(
            _INFERENCE_MISMATCH,
            f"type mismatch: expected {_display(lhs)}, found {_display(rhs)}",
            node,
        )

    def run(self) -> InferenceResult:
        for item in _iter_items(self.program.items):
            if isinstance(item, ast.FnDecl) and item.body is not None:
                self.current_fn = item.name
                self._walk_fn(item)
        self.current_fn = "<module>"
        return InferenceResult(
            bindings=dict(self.bindings),
            expression_types=dict(self.expression_types),
            function_bindings={
                name: dict(b) for name, b in self.function_bindings.items()
            },
            diagnostics=tuple(self.diagnostics),
        )

    def _walk_fn(self, fn: ast.FnDecl) -> None:
        env: dict[str, ast.Type] = {}
        for index, param in enumerate(fn.params):
            param_ty = param.ty or self._fresh()
            if param.ty is None:
                self._report(
                    _API_ERROR,
                    f"parameter {index} of {fn.name} has no type annotation",
                    param,
                )
            if isinstance(param.pattern, ast.PatIdent):
                env[param.pattern.name] = param_ty
                self._record_binding(param.pattern.name, param_ty)
        if fn.ret is not None and fn.body is not None:
            self._unify(self._walk_block(fn.body, env), fn.ret, fn.body)
        elif fn.body is not None:
            self._walk_block(fn.body, env)

    def _record_binding(self, name: str, ty: ast.Type) -> None:
        resolved = _resolve_type(ty, self.substitution)
        self.bindings[name] = resolved
        self.function_bindings[self.current_fn][name] = resolved

    def _walk_block(
        self, block: ast.Block, env: Mapping[str, ast.Type]
    ) -> ast.Type | None:
        inner = dict(env)
        for stmt in block.stmts:
            if isinstance(stmt, ast.LetStmt):
                binding_ty = stmt.ty or self._fresh()
                if stmt.value is not None:
                    self._infer(stmt.value, inner, expected=binding_ty)
                resolved = _resolve_type(binding_ty, self.substitution)
                self._bind_pattern(stmt.pattern, resolved, inner)
            elif isinstance(stmt, ast.ExprStmt):
                self._infer(stmt.expr, inner)
            elif isinstance(stmt, ast.ItemStmt) and isinstance(stmt.item, ast.FnDecl):
                self.signatures[stmt.item.name] = (
                    [param.ty for param in stmt.item.params],
                    stmt.item.ret,
                    stmt.item.generics,
                )
        if block.tail is not None:
            return self._infer(block.tail, inner)
        return None

    def _bind_pattern(
        self,
        pattern: ast.Pattern,
        ty: ast.Type,
        env: MutableMapping[str, ast.Type] | None = None,
    ) -> None:
        if isinstance(pattern, ast.PatIdent):
            self._record_binding(pattern.name, ty)
            if env is not None:
                env[pattern.name] = ty
        elif isinstance(pattern, ast.PatBinding):
            self._record_binding(pattern.name, ty)
            if env is not None:
                env[pattern.name] = ty
            self._bind_pattern(pattern.inner, ty, env)
        elif isinstance(pattern, (ast.PatRef, ast.PatBox)):
            inner = ty.inner if isinstance(ty, ast.TypeRef) else ty
            self._bind_pattern(pattern.inner, inner, env)
        elif isinstance(pattern, ast.PatTuple):
            if isinstance(ty, ast.TypeTuple) and len(ty.elems) == len(pattern.elems):
                for sub, elem in zip(pattern.elems, ty.elems):
                    self._bind_pattern(sub, elem, env)
        elif isinstance(pattern, ast.PatOr):
            for sub in pattern.alts:
                self._bind_pattern(sub, ty, env)

    def _infer(
        self,
        expr: ast.Expr | None,
        env: Mapping[str, ast.Type],
        expected: ast.Type | None = None,
    ) -> ast.Type | None:
        if expr is None:
            return None
        inferred = self._infer_expr(expr, env, expected)
        if inferred is not None:
            inferred = _resolve_type(inferred, self.substitution)
            self.expression_types[id(expr)] = inferred
            if expected is not None:
                self._unify(inferred, expected, expr)
                inferred = _resolve_type(inferred, self.substitution)
                self.expression_types[id(expr)] = inferred
        return inferred

    def _infer_expr(
        self,
        expr: ast.Expr,
        env: Mapping[str, ast.Type],
        expected: ast.Type | None,
    ) -> ast.Type | None:
        if isinstance(expr, ast.Paren):
            return self._infer(expr.inner, env, expected)
        if isinstance(expr, ast.LitInt):
            return _int_literal_type(expr)
        if isinstance(expr, ast.LitFloat):
            return _atomic("f64")
        if isinstance(expr, ast.LitStr):
            return _atomic("String")
        if isinstance(expr, ast.LitChar):
            return _atomic("char")
        if isinstance(expr, ast.LitBool):
            return _atomic("bool")
        if isinstance(expr, ast.Ident):
            return env.get(expr.name)
        if isinstance(expr, ast.Path):
            if len(expr.segments) == 1:
                return env.get(expr.segments[0])
            base = expr.segments[0]
            if base in self.enum_variants or base in self.struct_generics:
                args = list(expr.generic_args)
                if not args and isinstance(expected, ast.TypeName) and expected.args:
                    args = list(expected.args)
                return ast.TypeName(expr.line, expr.col, base, args)
            return None
        if isinstance(expr, ast.Unary):
            if expr.op == "!":
                return _atomic("bool")
            inner = self._infer(expr.operand, env)
            if expr.op in {"&", "&mut"} and inner is not None:
                return ast.TypeRef(expr.line, expr.col, inner, expr.mutable)
            return inner
        if isinstance(expr, ast.Binary):
            if expr.op in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"}:
                self._infer(expr.left, env)
                self._infer(expr.right, env)
                return _atomic("bool")
            left = self._infer(expr.left, env)
            right = self._infer(expr.right, env)
            if left is not None and right is not None:
                self._unify(left, right, expr)
                return _resolve_type(left, self.substitution)
            return left or right
        if isinstance(expr, ast.Cast):
            self._infer(expr.expr, env)
            return expr.ty
        if isinstance(expr, ast.Call):
            return self._infer_call(expr, env, expected)
        if isinstance(expr, ast.MethodCall):
            return self._infer_method(expr, env)
        if isinstance(expr, ast.Field):
            receiver = self._infer(expr.receiver, env)
            return self._infer_field(receiver, expr)
        if isinstance(expr, ast.Index):
            receiver = self._infer(expr.receiver, env)
            self._infer(expr.index, env)
            if isinstance(receiver, ast.TypeName) and receiver.args:
                return receiver.args[0]
            if isinstance(receiver, (ast.TypeSlice, ast.TypeArray)):
                return receiver.inner
            return None
        if isinstance(expr, ast.StructLiteral):
            return self._infer_struct_literal(expr, env, expected)
        if isinstance(expr, (ast.ArrayExpr, ast.RepeatArray)):
            return monomorph_propagate_vec_elem_type(expr, expected, env=env).vec_type
        if isinstance(expr, ast.TupleExpr):
            elems = [self._infer(elem, env) for elem in expr.elems]
            return ast.TypeTuple(
                expr.line, expr.col, [elem for elem in elems if elem is not None]
            )
        if isinstance(expr, ast.IfExpr):
            self._infer(expr.cond, env, _atomic("bool"))
            then_ty = self._walk_block(expr.then_block, env)
            else_ty = self._infer_branch(expr.else_branch, env, expected)
            if then_ty is not None and else_ty is not None:
                self._unify(then_ty, else_ty, expr)
            return then_ty or else_ty
        if isinstance(expr, ast.IfLet):
            value_ty = self._infer(expr.value, env)
            let_env = dict(env)
            self._bind_pattern(expr.pattern, value_ty or self._fresh(), let_env)
            then_ty = self._walk_block(expr.then_block, let_env)
            else_ty = self._infer_branch(expr.else_branch, env, expected)
            if then_ty is not None and else_ty is not None:
                self._unify(then_ty, else_ty, expr)
            return then_ty or else_ty
        if isinstance(expr, ast.MatchExpr):
            scrutinee = self._infer(expr.scrutinee, env)
            branch_types: list[ast.Type | None] = []
            for arm in expr.arms:
                arm_env = dict(env)
                if (
                    isinstance(scrutinee, ast.TypeName)
                    and scrutinee.name in self.enum_variants
                ):
                    self._bind_arm_pattern(arm, scrutinee, arm_env)
                branch_types.append(
                    self._infer(arm.body, arm_env, expected)
                    if arm.body is not None
                    else None
                )
            joined: ast.Type | None = None
            for branch in branch_types:
                if branch is None:
                    continue
                if joined is None:
                    joined = branch
                else:
                    self._unify(joined, branch, expr)
            return joined
        if isinstance(expr, ast.Block):
            return self._walk_block(expr, env)
        if isinstance(expr, ast.Range):
            return self._infer(expr.start, env) or self._infer(expr.end, env)
        if isinstance(expr, ast.OptionSome):
            inner = self._infer(expr.value, env)
            return ast.TypeName(expr.line, expr.col, "Option", [inner] if inner else [])
        if isinstance(expr, ast.OptionNone):
            return _atomic("Option")
        if isinstance(expr, ast.ResultOk):
            inner = self._infer(expr.value, env)
            return ast.TypeName(expr.line, expr.col, "Result", [inner] if inner else [])
        if isinstance(expr, ast.ResultErr):
            inner = self._infer(expr.value, env)
            return ast.TypeName(expr.line, expr.col, "Result", [inner] if inner else [])
        if isinstance(expr, (ast.Try, ast.Await)):
            return self._infer(expr.expr, env)
        if isinstance(expr, ast.MacroCall):
            if expr.path in {"println", "print", "eprintln", "eprint", "assert"}:
                return _atomic("()")
            return None
        if isinstance(expr, (ast.Loop, ast.While, ast.For)):
            if isinstance(expr, ast.For):
                self._infer(expr.iter, env)
            return _atomic("()")
        return None

    def _infer_branch(
        self,
        branch: ast.Expr | ast.Block | None,
        env: Mapping[str, ast.Type],
        expected: ast.Type | None,
    ) -> ast.Type | None:
        if branch is None:
            return None
        if isinstance(branch, ast.Block):
            return self._walk_block(branch, env)
        return self._infer(branch, env, expected)

    def _bind_arm_pattern(
        self,
        arm: ast.MatchArm,
        scrutinee: ast.TypeName,
        env: MutableMapping[str, ast.Type] | None = None,
    ) -> None:
        pattern = arm.pattern
        variant_name: str | None = None
        if (
            isinstance(pattern, ast.PatEnum)
            or isinstance(pattern, ast.PatPath)
            and "::" in pattern.path
        ):
            variant_name = pattern.path.split("::")[-1]
        if variant_name is None:
            return
        fields = self.enum_variants[scrutinee.name].get(variant_name)
        if not fields:
            return
        args: list[ast.Type] = list(scrutinee.args)
        if not args:
            return
        mapping = {
            param.name: arg
            for param, arg in zip(self.enum_generics.get(scrutinee.name, []), args)
        }
        if isinstance(pattern, ast.PatEnum):
            for sub, field_ty in zip(pattern.args, fields):
                self._bind_pattern(
                    sub, _substitute_type(field_ty, mapping) or self._fresh(), env
                )

    def _infer_field(
        self, receiver: ast.Type | None, expr: ast.Field
    ) -> ast.Type | None:
        if not isinstance(receiver, ast.TypeName):
            return None
        field_ty = self.struct_fields.get(receiver.name, {}).get(expr.name)
        if field_ty is None:
            return None
        mapping = {
            param.name: arg
            for param, arg in zip(
                self.struct_generics.get(receiver.name, []), receiver.args
            )
        }
        return _substitute_type(field_ty, mapping)

    def _infer_call(
        self, call: ast.Call, env: Mapping[str, ast.Type], expected: ast.Type | None
    ) -> ast.Type | None:
        func = call.func
        explicit_args: list[ast.Type] = []
        name: str | None = None
        if isinstance(func, ast.Ident):
            name = func.name
        elif isinstance(func, ast.Path):
            name = func.segments[-1] if len(func.segments) == 1 else func.segments[0]
            explicit_args = list(func.generic_args)
        if name is not None and name in self.signatures:
            params, ret, generics = self.signatures[name]
            mapping = {
                param.name: arg if index < len(explicit_args) else self._fresh()
                for index, (param, arg) in enumerate(zip(generics, explicit_args))
            }
            for param in generics[len(explicit_args) :]:
                mapping[param.name] = self._fresh()
            for index, (param_ty, arg) in enumerate(zip(params, call.args)):
                wanted = _substitute_type(param_ty, mapping) if param_ty else None
                actual = self._infer(arg, env, wanted)
                if wanted is not None and actual is not None:
                    self._unify(wanted, actual, arg)
                if actual is None and index < len(explicit_args):
                    continue
            result = _substitute_type(ret, mapping) if ret is not None else None
            return _resolve_type(result, self.substitution)
        if isinstance(func, ast.Path) and len(func.segments) >= 2:
            base = func.segments[0]
            variant = func.segments[-1]
            fields = self.enum_variants.get(base, {}).get(variant)
            args = list(func.generic_args)
            if not args and isinstance(expected, ast.TypeName) and expected.args:
                args = list(expected.args)
            if fields is not None:
                for field_ty, arg in zip(fields, call.args):
                    self._infer(arg, env, field_ty)
            return ast.TypeName(call.line, call.col, base, args)
        return None

    def _infer_method(
        self, call: ast.MethodCall, env: Mapping[str, ast.Type]
    ) -> ast.Type | None:
        receiver = self._infer(call.receiver, env)
        for arg in call.args:
            self._infer(arg, env)
        if isinstance(receiver, ast.TypeName):
            signature = self.method_signatures.get((receiver.name, call.method))
            if signature is not None:
                params, ret = signature
                mapping = {
                    param.name: arg
                    for param, arg in zip(
                        self.struct_generics.get(receiver.name, []), receiver.args
                    )
                }
                for param_ty, arg in zip(params, call.args):
                    wanted = _substitute_type(param_ty, mapping) if param_ty else None
                    actual = self._infer(arg, env, wanted)
                    if wanted is not None and actual is not None:
                        self._unify(wanted, actual, arg)
                return _substitute_type(ret, mapping)
        if call.method in {"len", "chars_count", "byte_len"}:
            return _atomic("usize")
        if call.method in {"is_empty", "contains", "starts_with", "ends_with"}:
            return _atomic("bool")
        if call.method in {"to_string", "trim", "to_uppercase", "to_lowercase"}:
            return _atomic("String")
        if call.method == "clone":
            return receiver
        if (
            call.method == "unwrap"
            and isinstance(receiver, ast.TypeName)
            and receiver.args
        ):
            return receiver.args[0]
        return None

    def _infer_struct_literal(
        self,
        literal: ast.StructLiteral,
        env: Mapping[str, ast.Type],
        expected: ast.Type | None,
    ) -> ast.Type | None:
        base = literal.path.split("::")[-1]
        args = list(literal.generic_args)
        if not args and isinstance(expected, ast.TypeName) and expected.args:
            args = list(expected.args)
        mapping = {
            param.name: arg
            for param, arg in zip(self.struct_generics.get(base, []), args)
        }
        for name, value in literal.fields:
            field_ty = self.struct_fields.get(base, {}).get(name)
            wanted = _substitute_type(field_ty, mapping) if field_ty else None
            self._infer(value, env, wanted)
        return ast.TypeName(literal.line, literal.col, base, args)


def _iter_stmts(block: ast.Block | None) -> Iterator[ast.Stmt]:
    if block is None:
        return
    for stmt in block.stmts:
        yield stmt
        nested = getattr(stmt, "body", None)
        if isinstance(nested, ast.Block):
            yield from _iter_stmts(nested)


def _int_literal_type(expr: ast.LitInt) -> ast.TypeName:
    for suffix in (
        "i128",
        "u128",
        "isize",
        "usize",
        "i64",
        "u64",
        "i32",
        "u32",
        "i16",
        "u16",
        "i8",
        "u8",
    ):
        if expr.raw.lower().endswith(suffix):
            return _atomic(suffix)
    return _atomic(_DEFAULT_TYPE)


def typecheck_infer_expressions(
    program: ast.Program,
) -> InferenceResult:
    """Infer and propagate expression types across bindings, calls and control flow.

    Annotated ``let`` bindings and parameter types flow into initializers and
    call arguments.  Generic function calls instantiate their type parameters
    from the argument types, ``if``/``match`` arms are joined by unification,
    and every inferred node is addressable through :meth:`InferenceResult.type_of`.
    The input program is never mutated.
    """
    return _ExpressionInferrer(program).run()


# ---------------------------------------------------------------------------
# 12 - trait bounds
# ---------------------------------------------------------------------------

_BUILTIN_IMPLS: dict[str, frozenset[str]] = {
    "Clone": frozenset(
        {
            "bool",
            "char",
            "String",
            "usize",
            "isize",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "f64",
        }
    ),
    "Copy": frozenset({"bool", "char", "usize", "isize", "i8", "i16", "i32", "i64"}),
    "Debug": frozenset({"bool", "char", "String", "usize", "isize", "i64", "f64"}),
    "Display": frozenset({"bool", "char", "String", "usize", "isize", "i64", "f64"}),
    "PartialEq": frozenset({"bool", "char", "String", "usize", "isize", "i64", "f64"}),
    "Eq": frozenset({"bool", "char", "String", "usize", "isize", "i64"}),
    "Hash": frozenset({"bool", "char", "String", "usize", "isize", "i64"}),
    "Default": frozenset({"bool", "char", "String", "usize", "isize", "i64", "f64"}),
}
_CONTAINER_TRAITS = frozenset(
    {"Clone", "Copy", "Debug", "PartialEq", "Eq", "Hash", "Send", "Sync"}
)
_CONTAINERS = frozenset({"Vec", "Option", "Result", "Box"})


@dataclass(frozen=True)
class BoundCheck:
    """One checked ``argument: Trait`` obligation of a generic item."""

    item: str
    parameter: str
    trait_name: str
    argument: ast.Type
    satisfied: bool
    node: ast.Node | None = None
    code: str = ""

    @property
    def argument_text(self) -> str:
        return _display(self.argument)


@dataclass(frozen=True)
class TraitBoundReport:
    """All trait obligations found in a program, split into passes and failures."""

    checks: tuple[BoundCheck, ...]

    @property
    def satisfied(self) -> tuple[BoundCheck, ...]:
        return tuple(check for check in self.checks if check.satisfied)

    @property
    def violations(self) -> tuple[BoundCheck, ...]:
        return tuple(check for check in self.checks if not check.satisfied)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for diagnostics and telemetry."""
        return {
            "checked": len(self.checks),
            "satisfied": len(self.satisfied),
            "violations": [
                {
                    "item": check.item,
                    "parameter": check.parameter,
                    "trait": check.trait_name,
                    "argument": check.argument_text,
                    "code": check.code,
                }
                for check in self.violations
            ],
        }


def _trait_satisfied(
    ty: ast.Type,
    trait_name: str,
    impls: Mapping[str, list[tuple[ast.Type, list[ast.GenericParam]]]],
    assume_builtins: bool,
    visiting: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    key = (trait_name, _display(ty))
    if key in visiting:
        return False
    visiting = visiting | {key}
    if not isinstance(ty, ast.TypeName):
        return trait_name in {"Clone", "Copy"} and isinstance(ty, ast.TypeRef)
    name = ty.name
    if assume_builtins and name in _BUILTIN_IMPLS.get(trait_name, frozenset()):
        return True
    if name in _CONTAINERS and trait_name in _CONTAINER_TRAITS and ty.args:
        return all(
            _trait_satisfied(arg, trait_name, impls, assume_builtins, visiting)
            for arg in ty.args
        )
    if name in _BUILTIN_IMPLS.get(trait_name, frozenset()) and ty.args:
        return True
    for target, impl_params in impls.get(trait_name, []):
        mapping: dict[str, ast.Type] = {}
        if not _match_type(target, ty, mapping):
            continue
        if all(
            _trait_satisfied(
                mapping[param.name], bound.name, impls, assume_builtins, visiting
            )
            for param in impl_params
            for bound in param.bounds
            if param.name in mapping
        ):
            return True
    return False


def typecheck_enforce_trait_bounds(
    program: ast.Program, *, assume_builtins: bool = True
) -> TraitBoundReport:
    """Verify that concrete generic arguments satisfy their declared trait bounds.

    Bounds are read from ``GenericParam.bounds`` on functions, structs, enums,
    and generic impl blocks.  Concrete arguments are collected from turbofish
    calls, struct literals, and type annotations.  User traits need a matching
    ``impl Trait for Type``; a small builtin table covers primitives when
    ``assume_builtins`` is true.  Failures carry code ``E001``.
    """
    generic_items: dict[str, list[ast.GenericParam]] = {}
    impls: dict[str, list[tuple[ast.Type, list[ast.GenericParam]]]] = defaultdict(list)
    impl_blocks: list[ast.ImplBlock] = []
    for item in _iter_items(program.items):
        if (
            isinstance(item, (ast.FnDecl, ast.StructDecl, ast.EnumDecl))
            and item.generics
        ):
            generic_items[item.name] = item.generics
        elif isinstance(item, ast.ImplBlock):
            impl_blocks.append(item)
            if item.trait is not None and isinstance(item.trait, ast.TypeName):
                impls[item.trait.name].append((item.ty, item.generics))

    substitutions: dict[str, list[tuple[list[ast.Type], ast.Node | None]]] = (
        defaultdict(list)
    )
    for node in _walk(program):
        if isinstance(node, ast.TypeName) and node.args and node.name in generic_items:
            substitutions[node.name].append((list(node.args), node))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Path):
            if node.func.generic_args and node.func.segments[0] in generic_items:
                substitutions[node.func.segments[0]].append(
                    (list(node.func.generic_args), node)
                )
        elif isinstance(node, ast.StructLiteral) and node.generic_args:
            base = node.path.split("::")[-1]
            if base in generic_items:
                substitutions[base].append((list(node.generic_args), node))

    checks: list[BoundCheck] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item_name, params in generic_items.items():
        for args, node in substitutions.get(item_name, []):
            if len(args) != len(params):
                continue
            for param, arg in zip(params, args):
                for bound in param.bounds:
                    key = (item_name, param.name, bound.name, _display(arg))
                    if key in seen:
                        continue
                    seen.add(key)
                    ok = _trait_satisfied(arg, bound.name, impls, assume_builtins)
                    checks.append(
                        BoundCheck(
                            item=item_name,
                            parameter=param.name,
                            trait_name=bound.name,
                            argument=arg,
                            satisfied=ok,
                            node=node,
                            code="" if ok else _INFERENCE_MISMATCH,
                        )
                    )
    for impl in impl_blocks:
        if not impl.generics or not isinstance(impl.ty, ast.TypeName):
            continue
        base = impl.ty.name
        for args, node in substitutions.get(base, []):
            mapping: dict[str, ast.Type] = {}
            if not _match_type(
                impl.ty, ast.TypeName(impl.ty.line, impl.ty.col, base, args), mapping
            ):
                continue
            for param in impl.generics:
                if param.name not in mapping:
                    continue
                for bound in param.bounds:
                    key = (
                        f"impl {base}",
                        param.name,
                        bound.name,
                        _display(mapping[param.name]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    ok = _trait_satisfied(
                        mapping[param.name], bound.name, impls, assume_builtins
                    )
                    checks.append(
                        BoundCheck(
                            item=f"impl {base}",
                            parameter=param.name,
                            trait_name=bound.name,
                            argument=mapping[param.name],
                            satisfied=ok,
                            node=node,
                            code="" if ok else _INFERENCE_MISMATCH,
                        )
                    )
    return TraitBoundReport(checks=tuple(checks))


# ---------------------------------------------------------------------------
# 13 - occurs check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeConstraint:
    """A unification obligation ``variable = type`` for the occurs check."""

    variable: str
    type: ast.Type
    origin: str = ""


@dataclass(frozen=True)
class OccursCheckDiagnostic:
    """A cyclic type constraint with a stable diagnostic code and cycle path."""

    code: str
    variable: str
    cycle: tuple[str, ...]
    message: str
    origin: str = ""


def _occurs_below_root(name: str, ty: ast.Type, *, root: bool = True) -> bool:
    """True when ``name`` occurs somewhere other than as the whole RHS."""
    if isinstance(ty, ast.TypeName):
        if not ty.args:
            return ty.name == name and not root
        return any(_occurs_below_root(name, arg, root=False) for arg in ty.args)
    if isinstance(ty, (ast.TypeRef, ast.TypeSlice, ast.TypeArray)):
        return _occurs_below_root(name, ty.inner, root=False)
    if isinstance(ty, ast.TypeTuple):
        return any(_occurs_below_root(name, item, root=False) for item in ty.elems)
    if isinstance(ty, ast.TypeFn):
        return any(
            _occurs_below_root(name, item, root=False) for item in ty.params
        ) or (ty.ret is not None and _occurs_below_root(name, ty.ret, root=False))
    return False


def _cycle_key(cycle: Sequence[str]) -> tuple[str, ...]:
    """Normalize a cycle so the same loop has one diagnostic regardless of entry."""
    items = list(cycle)
    if not items:
        return ()
    smallest = min(range(len(items)), key=lambda index: (items[index], index))
    return tuple(items[smallest:] + items[:smallest])


def typecheck_detect_occurs_check_cycle(
    constraints: Iterable[TypeConstraint],
) -> tuple[OccursCheckDiagnostic, ...]:
    """Detect direct and indirect cyclic type constraints before expansion.

    ``E005-OC1`` marks a variable that occurs inside its own right-hand side
    (for example ``T = Vec<T>``); ``E005-OC2`` marks a longer loop such as
    ``T = U`` together with ``U = Vec<T>``.  The trivial equation ``T = T`` is
    treated as satisfied.  Results are ordered by first appearance and cycles
    are de-duplicated by their normalized variable path.
    """
    pending = list(constraints)
    direct: list[OccursCheckDiagnostic] = []
    graph: dict[str, list[str]] = defaultdict(list)
    origins = {constraint.variable: constraint.origin for constraint in pending}
    for constraint in pending:
        if _occurs_below_root(constraint.variable, constraint.type):
            direct.append(
                OccursCheckDiagnostic(
                    code="E005-OC1",
                    variable=constraint.variable,
                    cycle=(constraint.variable,),
                    message=(
                        f"type variable {constraint.variable} occurs inside "
                        f"{_display(constraint.type)}"
                    ),
                    origin=constraint.origin,
                )
            )
        for var in sorted(_type_vars(constraint.type)):
            if var != constraint.variable:
                graph[constraint.variable].append(var)

    indirect: list[OccursCheckDiagnostic] = []
    state: dict[str, int] = {}
    stack: list[str] = []
    emitted: set[tuple[str, ...]] = {_cycle_key(item.cycle) for item in direct}

    def visit(variable: str) -> None:
        state[variable] = 1
        stack.append(variable)
        for neighbour in graph.get(variable, []):
            if state.get(neighbour, 0) == 0:
                visit(neighbour)
            elif state.get(neighbour) == 1:
                cycle = _cycle_key(stack[stack.index(neighbour) :])
                if cycle not in emitted:
                    emitted.add(cycle)
                    indirect.append(
                        OccursCheckDiagnostic(
                            code="E005-OC2",
                            variable=neighbour,
                            cycle=cycle,
                            message="cyclic type constraint: " + " -> ".join(cycle),
                            origin=origins.get(neighbour, ""),
                        )
                    )
        stack.pop()
        state[variable] = 2

    for constraint in pending:
        if state.get(constraint.variable, 0) == 0:
            visit(constraint.variable)
    return tuple(direct + indirect)


# ---------------------------------------------------------------------------
# 14 - placeholder registration
# ---------------------------------------------------------------------------


def monomorph_cache_placeholder_register[ItemT: ast.Item](
    cache: MutableMapping[Any, ItemT], key: Any, placeholder: ItemT
) -> ItemT:
    """Register an in-progress instantiation in ``cache`` before expanding it.

    Recursive generics need their own cache entry to exist while fields or
    variants are being transformed.  The first placeholder for ``key`` wins;
    later registrations return that same object so self-references resolve to
    the canonical instance instead of creating a second expansion.
    """
    existing = cache.get(key)
    if existing is not None:
        return existing
    cache[key] = placeholder
    return placeholder


# ---------------------------------------------------------------------------
# 15 / 16 - recursive generic resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonomorphResolution:
    """The concrete result of resolving one recursive generic instantiation."""

    name: str
    concrete_name: str
    arguments: tuple[ast.Type, ...]
    declarations: tuple[ast.Item, ...]
    referenced_names: frozenset[str]
    cached: bool

    def has_self_reference(self) -> bool:
        """Whether the resolved declaration mentions its own concrete name."""
        return self.concrete_name in self.referenced_names


def _resolve_recursive_generic(
    program: ast.Program,
    name: str,
    args: Sequence[ast.Type] | None,
    max_depth: int,
    *,
    enum: bool,
) -> MonomorphResolution:
    source = copy.deepcopy(program)
    worker = monomorph.Monomorphizer(source, max_depth=max_depth)
    worker.run()
    decl = (worker.generic_enums if enum else worker.generic_structs).get(name)
    if decl is None:
        kind = "enum" if enum else "struct"
        raise PillarError(_API_ERROR, f"no generic {kind} named {name!r} is declared")
    supplied = list(args) if args is not None else _default_args(decl.generics)
    if len(supplied) != len(decl.generics):
        raise PillarError(
            _API_ERROR,
            f"generic {name!r} expects {len(decl.generics)} type argument(s), "
            f"got {len(supplied)}",
        )
    key = ("enum" if enum else "struct", name, tuple(_display(arg) for arg in supplied))
    cache = worker.enum_cache if enum else worker.struct_instances
    cached = key in cache
    concrete = (
        worker.request_enum(name, supplied, 0)
        if enum
        else worker.request_struct(name, supplied, 0)
    )
    resolved = cache[key]
    references: set[str] = set()
    for item in _walk(resolved):
        if isinstance(item, (ast.TypeName, ast.Ident)):
            references.add(item.name)
    relevant = tuple(
        item
        for item in worker.generated_structs
        if getattr(item, "name", None) in references | {concrete}
    )
    return MonomorphResolution(
        name=name,
        concrete_name=concrete,
        arguments=tuple(supplied),
        declarations=relevant,
        referenced_names=frozenset(references),
        cached=cached,
    )


def monomorph_resolve_recursive_enum(
    program: ast.Program,
    name: str,
    args: Sequence[ast.Type] | None = None,
    *,
    max_depth: int = monomorph.MAX_MONOMORPH_DEPTH,
) -> MonomorphResolution:
    """Resolve a self-referential generic enum such as ``Tree<T>``.

    The enum cache entry is installed before variant fields are transformed,
    so ``Node(Vec<Tree<T>>)`` points at the in-progress specialization.  When
    ``args`` is omitted the default ``i64`` instantiation is used.  Mutually
    reachable declarations (for example a sibling enum) are returned as well.
    """
    return _resolve_recursive_generic(program, name, args, max_depth, enum=True)


def monomorph_resolve_recursive_struct(
    program: ast.Program,
    name: str,
    args: Sequence[ast.Type] | None = None,
    *,
    max_depth: int = monomorph.MAX_MONOMORPH_DEPTH,
) -> MonomorphResolution:
    """Resolve self-recursive and mutually recursive generic structs.

    Struct placeholders are cached before their fields are expanded, which
    keeps ``struct Node<T> { children: Vec<Node<T>> }`` and pairs such as
    ``A<T> <-> B<T>`` finite.  Unknown names or wrong argument counts raise an
    ``E003`` :class:`~runtime.diagnostics.PillarError`.
    """
    return _resolve_recursive_generic(program, name, args, max_depth, enum=False)


# ---------------------------------------------------------------------------
# 17 - cache key unification
# ---------------------------------------------------------------------------


def monomorph_unify_cache_keys[KeyT, ValueT](
    cache: MutableMapping[KeyT, ValueT],
    aliases: Iterable[tuple[KeyT, KeyT]],
) -> int:
    """Map alias cache keys onto their canonical specialization instance.

    Each ``(canonical, alias)`` pair declares that both keys denote the same
    monomorphized item.  Alias chains are merged transitively; the canonical
    key's cached value wins, but an existing alias value is adopted when the
    canonical key is absent.  Returns the number of alias entries created or
    repointed.
    """
    pairs = list(aliases)
    parent: dict[KeyT, KeyT] = {}
    ordered: list[KeyT] = []

    def find(key: KeyT) -> KeyT:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    for canonical, alias in pairs:
        for key in (canonical, alias):
            if key not in parent:
                parent[key] = key
                ordered.append(key)
        canonical_root, alias_root = find(canonical), find(alias)
        if canonical_root != alias_root:
            parent[alias_root] = canonical_root

    groups: dict[KeyT, list[KeyT]] = defaultdict(list)
    for key in ordered:
        groups[find(key)].append(key)

    changed = 0
    for root, keys in groups.items():
        value: ValueT | None = cache.get(root)
        if value is None:
            for key in keys:
                if key in cache:
                    value = cache[key]
                    break
        if value is None:
            continue
        for key in keys:
            if key not in cache or cache[key] is not value:
                cache[key] = value
                if key != root:
                    changed += 1
    return changed


# ---------------------------------------------------------------------------
# 18 - vec element propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VecPropagation:
    """Element type resolved for a ``vec![...]`` literal or nested collection."""

    element_type: ast.Type | None
    vec_type: ast.Type
    conflicts: tuple[str, ...] = ()


def _basic_expr_type(
    expr: ast.Expr,
    env: Mapping[str, ast.Type],
    expected: ast.Type | None,
) -> ast.Type | None:
    """Infer a single element expression without a full function environment."""
    if isinstance(expr, ast.Paren):
        return _basic_expr_type(expr.inner, env, expected)
    if isinstance(expr, ast.LitInt):
        return _int_literal_type(expr)
    if isinstance(expr, ast.LitFloat):
        return _atomic("f64")
    if isinstance(expr, ast.LitStr):
        return _atomic("String")
    if isinstance(expr, ast.LitChar):
        return _atomic("char")
    if isinstance(expr, ast.LitBool):
        return _atomic("bool")
    if isinstance(expr, ast.Ident):
        return env.get(expr.name)
    if isinstance(expr, ast.Cast):
        return expr.ty
    if isinstance(expr, ast.Unary):
        if expr.op == "!":
            return _atomic("bool")
        inner = _basic_expr_type(expr.operand, env, expected)
        if expr.op in {"&", "&mut"} and inner is not None:
            return ast.TypeRef(expr.line, expr.col, inner, expr.mutable)
        return inner
    if isinstance(expr, (ast.ArrayExpr, ast.RepeatArray)):
        return monomorph_propagate_vec_elem_type(expr, expected, env=env).vec_type
    if isinstance(expr, ast.MethodCall):
        receiver = _basic_expr_type(expr.receiver, env, expected)
        if expr.method in {"len", "chars_count", "byte_len"}:
            return _atomic("usize")
        if expr.method in {"is_empty", "contains", "starts_with", "ends_with"}:
            return _atomic("bool")
        if expr.method in {"to_string", "trim", "to_uppercase", "to_lowercase"}:
            return _atomic("String")
        if expr.method == "clone":
            return receiver
        if (
            expr.method == "unwrap"
            and isinstance(receiver, ast.TypeName)
            and receiver.args
        ):
            return receiver.args[0]
        return None
    if isinstance(expr, ast.OptionSome):
        inner = _basic_expr_type(expr.value, env, expected)
        return ast.TypeName(expr.line, expr.col, "Option", [inner] if inner else [])
    if isinstance(expr, ast.ResultOk):
        inner = _basic_expr_type(expr.value, env, expected)
        return ast.TypeName(expr.line, expr.col, "Result", [inner] if inner else [])
    if isinstance(expr, ast.StructLiteral):
        args = list(expr.generic_args)
        if not args and isinstance(expected, ast.TypeName) and expected.args:
            args = list(expected.args)
        return ast.TypeName(expr.line, expr.col, expr.path.split("::")[-1], args)
    if isinstance(expr, ast.Path) and len(expr.segments) >= 2:
        args = list(expr.generic_args)
        if not args and isinstance(expected, ast.TypeName) and expected.args:
            args = list(expected.args)
        return ast.TypeName(expr.line, expr.col, expr.segments[0], args)
    return None


def monomorph_propagate_vec_elem_type(
    expression: ast.Expr,
    expected: ast.Type | None = None,
    *,
    env: Mapping[str, ast.Type] | None = None,
) -> VecPropagation:
    """Propagate an element type through ``vec![...]`` and nested collections.

    The parser lowers ``vec![a, b]`` and ``vec![a; n]`` to array AST nodes.
    Element types come from the elements themselves, from identifiers in
    ``env``, or from the argument of an expected ``Vec<T>`` annotation (which
    also types an empty literal).  Nested vectors are resolved recursively,
    yielding ``Vec<Vec<i64>>`` for ``vec![vec![1], vec![2]]``.  Conflicting
    element types are reported as ``E001`` diagnostics rather than guessed.
    """
    bindings = dict(env or {})
    if isinstance(expression, ast.ArrayExpr):
        elements = list(expression.elems)
    elif isinstance(expression, ast.RepeatArray):
        elements = [expression.elem]
    else:
        direct = _basic_expr_type(expression, bindings, expected)
        return VecPropagation(direct, direct or _atomic("()"))

    expected_elem: ast.Type | None = None
    conflicts: list[str] = []
    if expected is not None:
        if (
            isinstance(expected, ast.TypeName)
            and expected.name == "Vec"
            and expected.args
        ):
            expected_elem = expected.args[0]
        else:
            conflicts.append(
                f"E001: expected {_display(expected)}, found a Vec literal"
            )

    element: ast.Type | None = None
    for index, element_expr in enumerate(elements):
        found = _basic_expr_type(element_expr, bindings, expected_elem)
        if found is None:
            found = expected_elem
        if found is None:
            continue
        if element is None:
            element = found
        elif _display(element) != _display(found):
            conflicts.append(
                f"E001: element {index} has type {_display(found)}, "
                f"expected {_display(element)}"
            )
    element = element or expected_elem
    vec_type = ast.TypeName(
        expression.line,
        expression.col,
        "Vec",
        [element] if element is not None else [],
    )
    return VecPropagation(element, vec_type, tuple(conflicts))


# ---------------------------------------------------------------------------
# 19 - dead specialization pruning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PruneResult:
    """A monomorphized program with unreachable specializations removed."""

    program: ast.Program
    removed: tuple[str, ...]
    kept: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a compact summary of the pruning pass."""
        return {"removed": list(self.removed), "kept": list(self.kept)}


def _referenced_names(value: object) -> set[str]:
    """Collect every identifier/type/path name mentioned by an AST subtree."""
    names: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, ast.Ident):
            names.add(item.name)
        elif isinstance(item, ast.Path):
            names.update(item.segments)
        elif isinstance(item, ast.TypeName):
            names.add(item.name)
        elif isinstance(item, ast.StructLiteral):
            names.add(item.path.split("::")[-1])
        elif isinstance(item, ast.MacroCall):
            names.add(item.path)
        elif isinstance(
            item, (ast.PatEnum, ast.PatPath, ast.PatStruct, ast.PatConst, ast.PatMacro)
        ):
            names.add(item.path.split("::")[0])
        for child in _node_children(item):
            visit(child)

    visit(value)
    return names


_PRUNEABLE = (ast.FnDecl, ast.StructDecl, ast.EnumDecl, ast.ConstDecl, ast.TypeAlias)


def monomorph_prune_dead_specializations(
    program: ast.Program, *, roots: Iterable[str] = ("main",)
) -> PruneResult:
    """Remove monomorphized declarations that no root can reach.

    Reachability follows function calls, type annotations, struct literals,
    macro names, and pattern paths through the already-concrete AST.  Traits,
    modules, and use declarations are always retained; impl blocks are kept
    when their target or trait is reachable.  The returned program preserves
    the original item order and source-line map.
    """
    declarations: dict[str, ast.Item] = {}
    for item in program.items:
        if isinstance(item, _PRUNEABLE):
            declarations[item.name] = item

    reachable: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop()
        if name in reachable:
            continue
        declaration = declarations.get(name)
        if declaration is None:
            continue
        reachable.add(name)
        for reference in sorted(_referenced_names(declaration)):
            if reference in declarations and reference not in reachable:
                queue.append(reference)
        for impl in program.items:
            if (
                isinstance(impl, ast.ImplBlock)
                and isinstance(impl.ty, ast.TypeName)
                and impl.ty.name == name
            ):
                for reference in _referenced_names(impl):
                    if reference in declarations and reference not in reachable:
                        queue.append(reference)

    kept_items: list[ast.Item] = []
    removed: list[str] = []
    kept: list[str] = []
    for item in program.items:
        if isinstance(item, _PRUNEABLE):
            name = item.name
            if name in reachable:
                kept.append(name)
                kept_items.append(item)
            else:
                removed.append(name)
        elif isinstance(item, ast.ImplBlock):
            target = item.ty.name if isinstance(item.ty, ast.TypeName) else None
            if target in reachable or _referenced_names(item) & reachable:
                kept_items.append(item)
        else:
            kept_items.append(item)
    return PruneResult(
        program=ast.Program(
            program.line,
            program.col,
            kept_items,
            dict(program.source_lines),
            span=program.span,
        ),
        removed=tuple(removed),
        kept=tuple(kept),
    )


# ---------------------------------------------------------------------------
# 20 - inflation report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationEntry:
    """Specialization counts and AST-growth estimate for one generic family."""

    family: str
    instantiations: int
    generic_nodes: int
    specialized_nodes: int

    @property
    def growth_nodes(self) -> int:
        return max(0, self.specialized_nodes - self.generic_nodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "instantiations": self.instantiations,
            "generic_nodes": self.generic_nodes,
            "specialized_nodes": self.specialized_nodes,
            "growth_nodes": self.growth_nodes,
        }


@dataclass(frozen=True)
class InflationReport:
    """Deterministic template-bloat report for a monomorphized program."""

    entries: tuple[InflationEntry, ...]
    warnings: tuple[str, ...]

    @property
    def total_instantiations(self) -> int:
        return sum(entry.instantiations for entry in self.entries)

    @property
    def total_specialized_nodes(self) -> int:
        return sum(entry.specialized_nodes for entry in self.entries)

    @property
    def total_growth_nodes(self) -> int:
        return sum(entry.growth_nodes for entry in self.entries)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly summary for build/CI telemetry."""
        return {
            "families": [entry.as_dict() for entry in self.entries],
            "total_instantiations": self.total_instantiations,
            "total_specialized_nodes": self.total_specialized_nodes,
            "total_growth_nodes": self.total_growth_nodes,
            "warnings": list(self.warnings),
        }


def monomorph_emit_inflation_report(
    program: ast.Program,
    *,
    max_instantiations: int = 8,
    max_growth_nodes: int = 512,
) -> InflationReport:
    """Report per-family specialization counts and estimated AST-size growth.

    The source program supplies the generic baseline; the existing
    :class:`~monomorph.Monomorphizer` supplies concrete declarations.  Structs
    and enums are mapped back through ``instance_source``, while functions are
    mapped by their longest matching generic-family prefix.  Growth is the
    difference between emitted node counts and the original generic node
    count.  Crossing ``max_instantiations`` or ``max_growth_nodes`` adds a
    stable ``E002-INF1`` / ``E002-INF2`` warning.
    """
    worker = monomorph.Monomorphizer(program)
    worker.run()
    baseline: dict[str, int] = {}
    for item in _iter_items(program.items):
        if getattr(item, "generics", None) and hasattr(item, "name"):
            baseline[item.name] = _count_nodes(item)

    emitted: dict[str, ast.Item] = {}
    for item in worker.generated_structs + worker.generated_fns:
        emitted[item.name] = item
    source_of: dict[str, str] = {
        concrete: source.name for concrete, source in worker.instance_source.items()
    }
    families = sorted(
        set(worker.generic_fns)
        | set(worker.generic_structs)
        | set(worker.generic_enums)
    )
    for concrete in emitted:
        if concrete in source_of:
            continue
        matches = [
            family
            for family in families
            if concrete == family or concrete.startswith(family + "_")
        ]
        source_of[concrete] = (
            max(matches, key=len) if matches else concrete.split("_")[0]
        )

    counts: dict[str, int] = defaultdict(int)
    sizes: dict[str, int] = defaultdict(int)
    for concrete, family in source_of.items():
        counts[family] += 1
        declaration = emitted.get(concrete)
        if declaration is not None:
            sizes[family] += _count_nodes(declaration)

    entries = tuple(
        InflationEntry(
            family=family,
            instantiations=counts[family],
            generic_nodes=baseline.get(family, 0),
            specialized_nodes=sizes[family],
        )
        for family in sorted(counts)
    )
    warnings: list[str] = []
    for entry in entries:
        if entry.instantiations > max_instantiations:
            warnings.append(
                f"{_INFLATION_LIMIT}-INF1: {entry.family} produced "
                f"{entry.instantiations} specializations (limit {max_instantiations})"
            )
        if entry.growth_nodes > max_growth_nodes:
            warnings.append(
                f"{_INFLATION_LIMIT}-INF2: {entry.family} grew by "
                f"{entry.growth_nodes} AST nodes (limit {max_growth_nodes})"
            )
    return InflationReport(entries=entries, warnings=tuple(warnings))


__all__ = [
    "BoundCheck",
    "InferenceResult",
    "InflationEntry",
    "InflationReport",
    "MonomorphResolution",
    "OccursCheckDiagnostic",
    "PruneResult",
    "TraitBoundReport",
    "TypeConstraint",
    "TypeDiagnostic",
    "VecPropagation",
    "monomorph_cache_placeholder_register",
    "monomorph_emit_inflation_report",
    "monomorph_propagate_vec_elem_type",
    "monomorph_prune_dead_specializations",
    "monomorph_resolve_recursive_enum",
    "monomorph_resolve_recursive_struct",
    "monomorph_unify_cache_keys",
    "typecheck_detect_occurs_check_cycle",
    "typecheck_enforce_trait_bounds",
    "typecheck_infer_expressions",
]
