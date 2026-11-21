"""Compile-time aggregate specialization for the WASM backend.

The WAT emitter in :mod:`backends.wasm.codegen` has a scalar stack machine:
``i64`` and ``bool`` only.  Rather than duplicating a second linear-memory
compiler in that emitter, this pass lowers the *source-level* aggregate shapes
used by deterministic parity programs into the scalar expressions the emitter
already understands.  Struct literals, vectors, enum variants, methods and
pattern matches are resolved before WAT generation; the arithmetic that consumes
them is still executed natively by wasmtime.

Anything that requires a genuinely dynamic aggregate (an aggregate parameter,
an aggregate return value, a computed match guard, and so on) is rejected with
an explicit :class:`WasmUnsupportedError`.  The three-way harness turns that
boundary into a structured skip instead of a false mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ast_nodes import (
    ArrayExpr,
    Assign,
    Binary,
    Block,
    Call,
    Cast,
    CompoundAssign,
    EnumDecl,
    ExprStmt,
    Field,
    FnDecl,
    Ident,
    IfExpr,
    ImplBlock,
    Index,
    LetStmt,
    LitBool,
    LitInt,
    LitStr,
    MatchExpr,
    MethodCall,
    Node,
    OptionNone,
    OptionSome,
    Paren,
    PatEnum,
    Path,
    PatIdent,
    PatRef,
    PatWildcard,
    Program,
    ResultErr,
    ResultOk,
    Return,
    StructDecl,
    StructLiteral,
    TypeName,
    Unary,
    While,
)

Value = Any  # scalar AST node or one of the compile-time aggregate constants

_MAX_INLINE_DEPTH = 48


class _Lowerer:
    def __init__(self, program: Program) -> None:
        self.structs: dict[str, StructDecl] = {}
        self.enums: dict[str, EnumDecl] = {}
        self.methods: dict[tuple[str, str], FnDecl] = {}
        self.functions: dict[str, FnDecl] = {}
        self._depth = 0
        for item in program.items:
            if isinstance(item, StructDecl):
                self.structs[item.name] = item
            elif isinstance(item, EnumDecl):
                self.enums[item.name] = item
            elif isinstance(item, FnDecl):
                self.functions[item.name] = item
            elif isinstance(item, ImplBlock) and isinstance(item.ty, TypeName):
                for nested in item.items:
                    if isinstance(nested, FnDecl):
                        self.methods[(item.ty.name, nested.name)] = nested

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _unsupported(message: str, node: Any = None) -> Exception:
        from backends.wasm.codegen import WasmUnsupportedError

        return WasmUnsupportedError(message, node)

    @staticmethod
    def _copy(node_type: type[Any], source: Any, **kwargs: Any) -> Any:
        return node_type(
            line=source.line,
            col=source.col,
            span=getattr(source, "span", None),
            **kwargs,
        )

    @staticmethod
    def _scalar(value: Value) -> bool:
        return isinstance(value, Node)

    def _require_scalar(self, value: Value, node: Any, what: str) -> Any:
        if not self._scalar(value):
            raise self._unsupported(f"{what} must be a scalar i64/bool value", node)
        return value

    @staticmethod
    def _binding_name(pattern: Any, node: Any) -> str | None:
        if isinstance(pattern, PatIdent):
            return pattern.name
        if isinstance(pattern, PatRef):
            return _Lowerer._binding_name(pattern.inner, node)
        if isinstance(pattern, PatWildcard):
            return None
        return None

    # ---- entry point -------------------------------------------------------

    def lower(self, program: Program) -> Program:
        lowered: list[Any] = []
        for item in program.items:
            if not isinstance(item, FnDecl):
                # Structs, enums, impls and traits are consumed by the pass;
                # the WAT emitter only receives scalar functions.
                continue
            lowered.append(self._function(item))
        return self._copy(
            Program, program, items=lowered, source_lines=dict(program.source_lines)
        )

    def _function(self, fn: FnDecl) -> FnDecl:
        for param in fn.params:
            name = (
                param.ty.name
                if isinstance(param.ty, TypeName)
                else type(param.ty).__name__
            )
            if name not in {"i64", "bool"}:
                raise self._unsupported(
                    f"function {fn.name!r} has an unsupported parameter type {name!r}",
                    fn,
                )
        if fn.ret is not None:
            name = (
                fn.ret.name if isinstance(fn.ret, TypeName) else type(fn.ret).__name__
            )
            if name not in {"i64", "bool"}:
                raise self._unsupported(
                    f"function {fn.name!r} has an unsupported return type {name!r}",
                    fn,
                )
        env: dict[str, Value] = {}
        for param in fn.params:
            name = self._binding_name(param.pattern, param)
            if name:
                env[name] = self._copy(Ident, param, name=name)
        body = self._block(fn.body, env)
        result = (
            self._require_scalar(body.tail, fn, f"function {fn.name!r} result")
            if fn.ret is not None
            else None
        )
        return self._copy(
            FnDecl,
            fn,
            name=fn.name,
            params=list(fn.params),
            ret=fn.ret,
            body=self._copy(Block, body, stmts=list(body.stmts), tail=result),
            generics=[],
            is_pub=fn.is_pub,
            is_async=fn.is_async,
            is_unsafe=fn.is_unsafe,
            terminates=False,
            attributes=list(fn.attributes),
        )

    # ---- blocks and statements --------------------------------------------

    def _block(self, block: Block, env: dict[str, Value]) -> Block:
        statements: list[Any] = []
        for stmt in block.stmts:
            lowered = self._statement(stmt, env)
            if lowered is not None:
                statements.append(lowered)
        tail = self._expr(block.tail, env) if block.tail is not None else None
        return self._copy(Block, block, stmts=statements, tail=tail)

    def _statement(self, stmt: Any, env: dict[str, Value]) -> Any | None:
        if isinstance(stmt, LetStmt):
            value = self._expr(stmt.value, env) if stmt.value is not None else None
            name = self._binding_name(stmt.pattern, stmt)
            if value is not None and name:
                # Scalar bindings stay as real WASM locals so mutations and
                # loop-carried values are preserved.  Only aggregate values
                # are specialized as compile-time constants in the env.
                env[name] = (
                    self._copy(Ident, stmt, name=name) if self._scalar(value) else value
                )
            if value is None or not self._scalar(value):
                return None
            declared = (
                stmt.ty
                if isinstance(stmt.ty, TypeName) and stmt.ty.name in {"i64", "bool"}
                else None
            )
            return self._copy(
                LetStmt,
                stmt,
                pattern=stmt.pattern,
                ty=declared,
                value=value,
                mutable=stmt.mutable,
            )
        if isinstance(stmt, ExprStmt):
            value = self._expr(stmt.expr, env)
            if not self._scalar(value):
                return None
            return self._copy(ExprStmt, stmt, expr=value)
        if isinstance(stmt, Return):
            value = self._expr(stmt.value, env) if stmt.value is not None else None
            if value is not None:
                value = self._require_scalar(value, stmt, "return value")
            return self._copy(Return, stmt, value=value)
        raise self._unsupported(
            f"statement {type(stmt).__name__} is not supported by WASM lowering", stmt
        )

    # ---- expressions -------------------------------------------------------

    def _expr(self, expr: Any, env: dict[str, Value]) -> Value:
        if expr is None:
            raise self._unsupported("missing expression")
        if isinstance(expr, (LitInt, LitBool, LitStr)):
            return expr
        if isinstance(expr, Ident):
            if expr.name not in env:
                raise self._unsupported(f"unknown local {expr.name!r}", expr)
            return env[expr.name]
        if isinstance(expr, Paren):
            inner = self._expr(expr.inner, env)
            if not self._scalar(inner):
                return inner
            return self._copy(Paren, expr, inner=inner)
        if isinstance(expr, Unary):
            operand = self._require_scalar(
                self._expr(expr.operand, env), expr, "unary operand"
            )
            return self._copy(
                Unary, expr, op=expr.op, operand=operand, mutable=expr.mutable
            )
        if isinstance(expr, Binary):
            left = self._require_scalar(
                self._expr(expr.left, env), expr, "binary operand"
            )
            right = self._require_scalar(
                self._expr(expr.right, env), expr, "binary operand"
            )
            return self._copy(Binary, expr, op=expr.op, left=left, right=right)
        if isinstance(expr, Cast):
            inner = self._require_scalar(
                self._expr(expr.expr, env), expr, "cast operand"
            )
            name = expr.ty.name if isinstance(expr.ty, TypeName) else None
            if name not in {"i64", "bool", "usize", "isize"}:
                raise self._unsupported(f"cast to {name!r} is not supported", expr)
            return inner
        if isinstance(expr, Assign):
            if not isinstance(expr.target, Ident):
                raise self._unsupported("only local assignments are supported", expr)
            value = self._require_scalar(
                self._expr(expr.value, env), expr, "assignment value"
            )
            return self._copy(Assign, expr, target=expr.target, value=value)
        if isinstance(expr, CompoundAssign):
            if not isinstance(expr.target, Ident):
                raise self._unsupported("only local assignments are supported", expr)
            value = self._require_scalar(
                self._expr(expr.value, env), expr, "assignment value"
            )
            return self._copy(
                CompoundAssign, expr, op=expr.op, target=expr.target, value=value
            )
        if isinstance(expr, StructLiteral):
            return _StructConst(
                name=expr.path,
                fields={name: self._expr(value, env) for name, value in expr.fields},
            )
        if isinstance(expr, ArrayExpr):
            return _VecConst(tuple(self._expr(item, env) for item in expr.elems))
        if isinstance(expr, Field):
            receiver = self._expr(expr.receiver, env)
            if not isinstance(receiver, _StructConst):
                raise self._unsupported("field access needs a struct value", expr)
            if expr.name not in receiver.fields:
                raise self._unsupported(
                    f"struct {receiver.name!r} has no field {expr.name!r}", expr
                )
            return receiver.fields[expr.name]
        if isinstance(expr, Index):
            receiver = self._expr(expr.receiver, env)
            index = self._expr(expr.index, env)
            if not isinstance(receiver, _VecConst) or not isinstance(index, LitInt):
                raise self._unsupported(
                    "dynamic vector indexing is not supported", expr
                )
            if index.value < 0 or index.value >= len(receiver.elements):
                raise self._unsupported("vector index is out of bounds", expr)
            return receiver.elements[index.value]
        if isinstance(expr, MethodCall):
            return self._method_call(expr, env)
        if isinstance(expr, Call):
            return self._call(expr, env)
        if isinstance(expr, MatchExpr):
            return self._match(expr, env)
        if isinstance(expr, IfExpr):
            return self._if(expr, env)
        if isinstance(expr, Block):
            return self._block(expr, env)
        if isinstance(expr, While):
            cond = self._require_scalar(
                self._expr(expr.cond, env), expr, "while condition"
            )
            body = self._block(expr.body, env)
            return self._copy(While, expr, cond=cond, body=body)
        if isinstance(expr, Return):
            value = self._expr(expr.value, env) if expr.value is not None else None
            if value is not None:
                value = self._require_scalar(value, expr, "return value")
            return self._copy(Return, expr, value=value)
        if isinstance(expr, OptionSome):
            return _EnumConst("Option", "Some", (self._expr(expr.value, env),))
        if isinstance(expr, OptionNone):
            return _EnumConst("Option", "None", ())
        if isinstance(expr, ResultOk):
            return _EnumConst("Result", "Ok", (self._expr(expr.value, env),))
        if isinstance(expr, ResultErr):
            return _EnumConst("Result", "Err", (self._expr(expr.value, env),))
        raise self._unsupported(
            f"expression {type(expr).__name__} is not supported by WASM lowering", expr
        )

    def _method_call(self, expr: MethodCall, env: dict[str, Value]) -> Value:
        receiver = self._expr(expr.receiver, env)
        if expr.method in {"len", "is_empty"} and isinstance(receiver, _VecConst):
            count = 0 if expr.method == "is_empty" else len(receiver.elements)
            return self._copy(
                LitBool if expr.method == "is_empty" else LitInt,
                expr,
                value=(count == 0) if expr.method == "is_empty" else count,
                **({} if expr.method == "is_empty" else {"raw": str(count)}),
            )
        if expr.method in {"clone", "to_owned"} and self._scalar(receiver):
            return receiver
        if isinstance(receiver, _StructConst):
            method = self.methods.get((receiver.name, expr.method))
            if method is not None:
                args = [self._expr(arg, env) for arg in expr.args]
                return self._inline(method, args, receiver=receiver, env=env)
        raise self._unsupported(
            f"method {expr.method!r} is not supported by WASM lowering", expr
        )

    def _call(self, expr: Call, env: dict[str, Value]) -> Value:
        if isinstance(expr.func, Path):
            segments = expr.func.segments
            if len(segments) == 2 and segments[0] in self.enums:
                return _EnumConst(
                    segments[0],
                    segments[1],
                    tuple(self._expr(arg, env) for arg in expr.args),
                )
            if len(segments) == 2 and segments[0] in self.structs:
                method = self.methods.get((segments[0], segments[1]))
                if method is not None:
                    return self._inline(
                        method,
                        [self._expr(arg, env) for arg in expr.args],
                        env=env,
                    )
        if isinstance(expr.func, Ident):
            fn = self.functions.get(expr.func.name)
            if fn is not None:
                args = [self._expr(arg, env) for arg in expr.args]
                ret = fn.ret.name if isinstance(fn.ret, TypeName) else None
                if ret in {"i64", "bool", None} and all(
                    self._scalar(arg) for arg in args
                ):
                    return self._copy(
                        Call,
                        expr,
                        func=expr.func,
                        args=[
                            self._require_scalar(arg, expr, "call argument")
                            for arg in args
                        ],
                    )
                if fn.body is not None:
                    return self._inline(fn, args, env=env)
        raise self._unsupported(
            f"call {self._describe(expr.func)!r} is not supported", expr
        )

    @staticmethod
    def _describe(func: Any) -> str:
        if isinstance(func, Ident):
            return func.name
        if isinstance(func, Path):
            return "::".join(func.segments)
        return type(func).__name__

    def _inline(
        self,
        fn: FnDecl,
        args: list[Value],
        *,
        env: dict[str, Value],
        receiver: Value | None = None,
    ) -> Value:
        if self._depth >= _MAX_INLINE_DEPTH or fn.body is None:
            raise self._unsupported(f"cannot inline function {fn.name!r}", fn)
        child = dict(env)
        params = list(fn.params)
        if receiver is not None and params and isinstance(params[0].pattern, PatRef):
            name = self._binding_name(params[0].pattern, params[0])
            if name:
                child[name] = receiver
            params = params[1:]
        for param, value in zip(params, args):
            name = self._binding_name(param.pattern, param)
            if name:
                child[name] = value
        self._depth += 1
        try:
            body = self._block(fn.body, child)
        finally:
            self._depth -= 1
        if body.tail is None:
            raise self._unsupported(f"function {fn.name!r} has no result", fn)
        return body.tail

    def _match(self, expr: MatchExpr, env: dict[str, Value]) -> Value:
        scrutinee = self._expr(expr.scrutinee, env)
        if not isinstance(scrutinee, _EnumConst):
            raise self._unsupported("match needs an enum value", expr)
        for arm in expr.arms:
            pattern = arm.pattern
            if isinstance(pattern, PatEnum):
                if not pattern.path.endswith(scrutinee.variant):
                    continue
                if len(pattern.args) != len(scrutinee.payload):
                    raise self._unsupported("match pattern arity mismatch", arm)
                child = dict(env)
                for subpattern, value in zip(pattern.args, scrutinee.payload):
                    self._bind(subpattern, value, child)
            elif isinstance(pattern, PatIdent):
                child = dict(env)
                child[pattern.name] = scrutinee
            elif isinstance(pattern, PatWildcard):
                child = dict(env)
            else:
                raise self._unsupported(
                    f"match pattern {type(pattern).__name__} is not supported", arm
                )
            if arm.guard is not None:
                raise self._unsupported("match guards are not supported", arm)
            return self._block_value(self._expr(arm.body, child))
        raise self._unsupported(f"no match arm for variant {scrutinee.variant!r}", expr)

    def _bind(self, pattern: Any, value: Value, env: dict[str, Value]) -> None:
        if isinstance(pattern, PatWildcard):
            return
        if isinstance(pattern, PatIdent):
            env[pattern.name] = value
            return
        if isinstance(pattern, PatRef):
            self._bind(pattern.inner, value, env)
            return
        raise self._unsupported(
            f"binding pattern {type(pattern).__name__} is not supported", pattern
        )

    @staticmethod
    def _block_value(value: Value) -> Value:
        return value.tail if isinstance(value, Block) else value

    def _if(self, expr: IfExpr, env: dict[str, Value]) -> Value:
        cond = self._require_scalar(self._expr(expr.cond, env), expr, "if condition")
        lowered_then = self._expr(expr.then_block, env)
        then_value = self._block_value(lowered_then)
        lowered_else = (
            self._expr(expr.else_branch, env) if expr.else_branch is not None else None
        )
        else_value = (
            self._block_value(lowered_else) if lowered_else is not None else None
        )
        if isinstance(cond, LitBool) and (
            not self._scalar(then_value)
            or (else_value is not None and not self._scalar(else_value))
        ):
            selected = then_value if cond.value else else_value
            if selected is None:
                raise self._unsupported("if expression has no selected branch", expr)
            return selected

        def as_branch(lowered: Value | None, value: Value | None, original: Any) -> Any:
            if value is None:
                # A statement-only or early-return branch stays a Block.
                return (
                    lowered
                    if isinstance(lowered, Block)
                    else self._copy(Block, original, stmts=[], tail=None)
                )
            if isinstance(value, Block):
                return value
            return self._copy(Block, original, stmts=[], tail=value)

        then_block = as_branch(lowered_then, then_value, expr.then_block)
        else_branch = (
            as_branch(lowered_else, else_value, expr.else_branch)
            if expr.else_branch is not None
            else None
        )
        return self._copy(
            IfExpr, expr, cond=cond, then_block=then_block, else_branch=else_branch
        )


@dataclass(frozen=True)
class _StructConst:
    name: str
    fields: dict[str, Value]


@dataclass(frozen=True)
class _VecConst:
    elements: tuple[Value, ...]


@dataclass(frozen=True)
class _EnumConst:
    name: str
    variant: str
    payload: tuple[Value, ...]


def lower_wasm_aggregates(program: Program) -> Program:
    """Return a scalar-only program suitable for the WAT emitter."""
    return _Lowerer(program).lower(program)


__all__ = ["lower_wasm_aggregates"]
