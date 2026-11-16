"""WASM text-format code generator for a compute-only Polyglot subset.

The generator lowers top-level functions to WebAssembly text format (``.wat``)
and the backend instantiates the module through ``wasmtime``. Values are
restricted to ``i64`` and ``bool``; there are no strings, aggregates, or host
imports yet. Unsupported nodes raise :class:`WasmUnsupportedError` with the
source position instead of silently producing an invalid module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ast_nodes import (
    Assign,
    Binary,
    Block,
    Call,
    CompoundAssign,
    Expr,
    ExprStmt,
    FnDecl,
    Ident,
    IfExpr,
    ItemStmt,
    LetStmt,
    LitBool,
    LitInt,
    Paren,
    PatIdent,
    PatWildcard,
    Program,
    Return,
    Type,
    TypeName,
    Unary,
    While,
)
from backends.base import BackendBase

I64 = "i64"
BOOL = "i32"

_ARITHMETIC = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div_s",
    "%": "rem_s",
}
_COMPARISON = {
    "==": "eq",
    "!=": "ne",
    "<": "lt_s",
    "<=": "le_s",
    ">": "gt_s",
    ">=": "ge_s",
}


class WasmUnsupportedError(RuntimeError):
    """Raised when a node cannot be lowered by the WASM subset backend."""

    def __init__(self, message: str, node: Any | None = None) -> None:
        self.node = node
        self.line = getattr(node, "line", None)
        self.col = getattr(node, "col", None)
        location = (
            f" at line {self.line}, column {self.col}"
            if self.line is not None and self.col is not None
            else ""
        )
        super().__init__(f"wasm backend: {message}{location}")


class WasmRuntimeUnavailableError(RuntimeError):
    """Raised when wasmtime is not installed."""


class WasmCompileError(RuntimeError):
    """Raised when the generated WAT cannot be compiled."""


@dataclass(frozen=True)
class _Signature:
    params: tuple[str, ...]
    ret: str | None
    function_id: int
    decl: FnDecl


def _wasm_type(ty: Type | None, node: Any | None = None) -> str | None:
    if ty is None:
        return None
    if isinstance(ty, TypeName):
        if ty.name == "i64":
            return I64
        if ty.name == "bool":
            return BOOL
    raise WasmUnsupportedError(
        f"type {ty!r} is not supported (only i64 and bool)", node
    )


def _is_value(ty: str | None) -> bool:
    return ty is not None


class WasmCodegen:
    """Lower a Polyglot program to a WAT module string."""

    def __init__(self) -> None:
        self._signatures: dict[str, _Signature] = {}
        self.exported_fns: list[str] = []
        self._locals: dict[str, tuple[int, str]] = {}
        self._local_decls: list[tuple[int, str]] = []
        self._next_local_index = 0
        self._label = 0

    # ---- public ------------------------------------------------------------

    def generate(self, program: Program) -> str:
        self._signatures = {}
        self.exported_fns = []
        self._collect_signatures(program)
        self.exported_fns = list(self._signatures)

        lines = ["(module"]
        for name, signature in self._signatures.items():
            lines.extend(self._function(name, signature))
        lines.append(")")
        return "\n".join(lines) + "\n"

    # ---- signatures --------------------------------------------------------

    def _collect_signatures(self, program: Program) -> None:
        for item in program.items:
            if isinstance(item, FnDecl):
                if item.generics:
                    raise WasmUnsupportedError(
                        "generic functions are not supported", item
                    )
                if item.is_async:
                    raise WasmUnsupportedError(
                        "async functions are not supported", item
                    )
                if item.body is None:
                    raise WasmUnsupportedError(
                        "functions without a body are not supported", item
                    )
                if item.name in self._signatures:
                    raise WasmUnsupportedError(
                        f"duplicate function {item.name!r}", item
                    )
                params = tuple(self._param_type(param, item) for param in item.params)
                ret = _wasm_type(item.ret, item)
                self._signatures[item.name] = _Signature(
                    params=params,
                    ret=ret,
                    function_id=len(self._signatures),
                    decl=item,
                )
            else:
                raise WasmUnsupportedError(
                    f"top-level {type(item).__name__} is not supported", item
                )

    def _param_type(self, param: Any, owner: FnDecl) -> str:
        name = self._binding_name(param.pattern, owner)
        ty = _wasm_type(param.ty, param)
        if ty is None:
            raise WasmUnsupportedError(
                f"parameter {name!r} needs an explicit type", param
            )
        return ty

    # ---- locals ------------------------------------------------------------

    def _binding_name(self, pattern: Any, owner: Any) -> str:
        if isinstance(pattern, PatIdent):
            return pattern.name
        if isinstance(pattern, PatWildcard):
            return "_"
        raise WasmUnsupportedError(
            f"pattern {type(pattern).__name__} is not supported", owner
        )

    def _scan_locals(self, block: Block) -> None:
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                name = self._binding_name(stmt.pattern, stmt)
                if name == "_":
                    continue
                if name in self._locals:
                    raise WasmUnsupportedError(
                        f"local {name!r} is declared more than once", stmt
                    )
                declared = _wasm_type(stmt.ty, stmt)
                inferred = (
                    self._infer(stmt.value, stmt) if stmt.value is not None else None
                )
                ty = declared or inferred
                if ty is None:
                    raise WasmUnsupportedError(
                        f"cannot infer the type of local {name!r}", stmt
                    )
                if declared is not None and inferred is not None and declared != inferred:
                    raise WasmUnsupportedError(
                        f"local {name!r} is {declared} but its value is {inferred}",
                        stmt,
                    )
                index = self._next_local_index
                self._next_local_index += 1
                self._locals[name] = (index, ty)
                self._local_decls.append((index, ty))
            elif isinstance(stmt, ExprStmt):
                self._scan_nested(stmt.expr)
            elif isinstance(stmt, ItemStmt):
                raise WasmUnsupportedError(
                    "items inside function bodies are not supported", stmt
                )
        if block.tail is not None:
            self._scan_nested(block.tail)

    def _scan_nested(self, expr: Expr) -> None:
        if isinstance(expr, IfExpr):
            self._scan_locals(expr.then_block)
            self._scan_branch(expr.else_branch)
        elif isinstance(expr, While):
            self._scan_locals(expr.body)
        elif isinstance(expr, Block):
            self._scan_locals(expr)
        elif isinstance(expr, Paren):
            self._scan_nested(expr.inner)
        elif isinstance(expr, Return) and isinstance(expr.value, Block):
            self._scan_locals(expr.value)


    def _scan_branch(self, branch: Any) -> None:
        if isinstance(branch, Block):
            self._scan_locals(branch)
        elif isinstance(branch, (IfExpr, Paren)):
            self._scan_nested(branch)

    # ---- type inference ----------------------------------------------------

    def _infer(self, expr: Expr | None, node: Any | None = None) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, LitInt):
            return I64
        if isinstance(expr, LitBool):
            return BOOL
        if isinstance(expr, Ident):
            if expr.name not in self._locals:
                raise WasmUnsupportedError(
                    f"unknown local {expr.name!r}", expr
                )
            return self._locals[expr.name][1]
        if isinstance(expr, Paren):
            return self._infer(expr.inner, expr)
        if isinstance(expr, Unary):
            if expr.op == "!":
                return BOOL
            if expr.op == "-":
                return self._infer(expr.operand, expr)
            raise WasmUnsupportedError(
                f"unary operator {expr.op!r} is not supported", expr
            )
        if isinstance(expr, Binary):
            if expr.op in _COMPARISON or expr.op in {"&&", "||"}:
                return BOOL
            if expr.op in _ARITHMETIC:
                left = self._infer(expr.left, expr)
                right = self._infer(expr.right, expr)
                if left != right:
                    raise WasmUnsupportedError(
                        f"mismatched operand types {left} and {right} for {expr.op!r}",
                        expr,
                    )
                return left
            raise WasmUnsupportedError(
                f"binary operator {expr.op!r} is not supported", expr
            )
        if isinstance(expr, Call):
            if not isinstance(expr.func, Ident):
                raise WasmUnsupportedError("indirect calls are not supported", expr)
            signature = self._signatures.get(expr.func.name)
            if signature is None:
                raise WasmUnsupportedError(
                    f"unknown function {expr.func.name!r}", expr
                )
            if len(expr.args) != len(signature.params):
                raise WasmUnsupportedError(
                    f"{expr.func.name!r} expects {len(signature.params)} arguments, "
                    f"got {len(expr.args)}",
                    expr,
                )
            return signature.ret
        if isinstance(expr, IfExpr):
            then_ty = self._infer_block(expr.then_block, expr)
            else_ty = self._infer_branch(expr.else_branch)
            if expr.else_branch is None:
                return None
            if then_ty != else_ty:
                raise WasmUnsupportedError(
                    f"if branches have different types {then_ty} and {else_ty}",
                    expr,
                )
            return then_ty
        if isinstance(expr, Block):
            return self._infer_block(expr, expr)
        if isinstance(expr, (Assign, CompoundAssign)):
            if not isinstance(expr.target, Ident):
                raise WasmUnsupportedError(
                    "only local assignments are supported", expr
                )
            if expr.target.name not in self._locals:
                raise WasmUnsupportedError(
                    f"unknown local {expr.target.name!r}", expr
                )
            return self._locals[expr.target.name][1]
        if isinstance(expr, Return):
            return None
        if isinstance(expr, While):
            return None
        raise WasmUnsupportedError(
            f"expression {type(expr).__name__} is not supported", expr
        )

    def _infer_block(self, block: Block, node: Any | None = None) -> str | None:
        return self._infer(block.tail, node or block)

    def _infer_branch(self, branch: Any) -> str | None:
        if branch is None:
            return None
        if isinstance(branch, Block):
            return self._infer_block(branch, branch)
        return self._infer(branch, branch)

    # ---- functions ---------------------------------------------------------

    def _function(self, name: str, signature: _Signature) -> list[str]:
        self._locals = {}
        self._local_decls = []
        self._next_local_index = len(signature.decl.params)
        for index, param in enumerate(signature.decl.params):
            param_name = self._binding_name(param.pattern, param)
            if param_name == "_":
                param_name = f"__ignored_{index}"
            if param_name in self._locals:
                raise WasmUnsupportedError(
                    f"duplicate parameter {param_name!r}", param
                )
            ty = signature.params[index]
            self._locals[param_name] = (index, ty)

        self._scan_locals(signature.decl.body)
        header = [f'  (func $fn{signature.function_id} (export "{name}")']
        for index, param in enumerate(signature.decl.params):
            param_name = self._binding_name(param.pattern, param)
            wat_name = f"$v{index}"
            header.append(f"    (param {wat_name} {signature.params[index]})")
            if param_name != "_":
                self._locals[param_name] = (index, signature.params[index])
        if signature.ret is not None:
            header.append(f"    (result {signature.ret})")
        for index, ty in self._local_decls:
            header.append(f"    (local $v{index} {ty})")

        want_value = signature.ret is not None
        if want_value and signature.decl.body.tail is None:
            body, _body_ty = self._block(signature.decl.body, want_value=False)
            body = f"{body} (unreachable)"
            body_ty = signature.ret
        else:
            body, body_ty = self._block(signature.decl.body, want_value=want_value)
        if want_value:
            if body_ty != signature.ret:
                raise WasmUnsupportedError(
                    f"function {name!r} returns {body_ty or '()'} "
                    f"but declares {signature.ret}",
                    signature.decl,
                )
        elif _is_value(body_ty):
            body = f"(drop {body})"

        lines = header + [f"    {body}", "  )"]
        return lines

    # ---- blocks and statements --------------------------------------------

    def _block(self, block: Block, *, want_value: bool) -> tuple[str, str | None]:
        parts: list[str] = []
        for stmt in block.stmts:
            parts.append(self._statement(stmt))
        tail_ty: str | None = None
        if block.tail is not None:
            tail_code, tail_ty = self._expr(block.tail, want_value=want_value)
            if want_value:
                parts.append(tail_code)
            elif _is_value(tail_ty):
                parts.append(f"(drop {tail_code})")
                tail_ty = None
        if want_value:
            if tail_ty is None:
                raise WasmUnsupportedError(
                    "block is used as a value but has no tail expression", block
                )
            return (
                f"(block (result {tail_ty}) " + " ".join(parts) + ")",
                tail_ty,
            )
        return "(block " + " ".join(parts) + ")", None

    def _statement(self, stmt: Any) -> str:
        if isinstance(stmt, LetStmt):
            name = self._binding_name(stmt.pattern, stmt)
            if name == "_":
                value, ty = self._expr(stmt.value, want_value=False)
                return f"(drop {value})" if _is_value(ty) else value
            if stmt.value is None:
                raise WasmUnsupportedError(
                    f"local {name!r} has no initializer", stmt
                )
            index, ty = self._locals[name]
            value, value_ty = self._expr(stmt.value, want_value=True)
            if value_ty != ty:
                raise WasmUnsupportedError(
                    f"local {name!r} is {ty} but its value is {value_ty}", stmt
                )
            return f"(local.set $v{index} {value})"
        if isinstance(stmt, ExprStmt):
            value, ty = self._expr(stmt.expr, want_value=False)
            return f"(drop {value})" if _is_value(ty) else value
        if isinstance(stmt, ItemStmt):
            raise WasmUnsupportedError(
                "items inside function bodies are not supported", stmt
            )
        raise WasmUnsupportedError(
            f"statement {type(stmt).__name__} is not supported", stmt
        )

    # ---- expressions -------------------------------------------------------

    def _expr(self, expr: Expr | None, *, want_value: bool) -> tuple[str, str | None]:
        if expr is None:
            raise WasmUnsupportedError("missing expression")
        if isinstance(expr, LitInt):
            return f"(i64.const {expr.value})", I64
        if isinstance(expr, LitBool):
            return f"(i32.const {1 if expr.value else 0})", BOOL
        if isinstance(expr, Ident):
            if expr.name not in self._locals:
                raise WasmUnsupportedError(f"unknown local {expr.name!r}", expr)
            index, ty = self._locals[expr.name]
            return f"(local.get $v{index})", ty
        if isinstance(expr, Paren):
            return self._expr(expr.inner, want_value=want_value)
        if isinstance(expr, Unary):
            return self._unary(expr)
        if isinstance(expr, Binary):
            return self._binary(expr)
        if isinstance(expr, Call):
            return self._call(expr)
        if isinstance(expr, IfExpr):
            return self._if(expr, want_value=want_value)
        if isinstance(expr, Block):
            return self._block(expr, want_value=want_value)
        if isinstance(expr, Assign):
            return self._assign(expr)
        if isinstance(expr, CompoundAssign):
            return self._compound_assign(expr)
        if isinstance(expr, Return):
            if expr.value is None:
                return "(return)", None
            value, _ty = self._expr(expr.value, want_value=True)
            return f"(return {value})", None
        if isinstance(expr, While):
            return self._while(expr), None
        raise WasmUnsupportedError(
            f"expression {type(expr).__name__} is not supported", expr
        )

    def _unary(self, expr: Unary) -> tuple[str, str | None]:
        operand, ty = self._expr(expr.operand, want_value=True)
        if expr.op == "!":
            if ty != BOOL:
                raise WasmUnsupportedError("'!' expects a bool operand", expr)
            return f"(i32.eqz {operand})", BOOL
        if expr.op == "-":
            if ty != I64:
                raise WasmUnsupportedError("unary '-' expects an i64 operand", expr)
            return f"(i64.sub (i64.const 0) {operand})", I64
        raise WasmUnsupportedError(
            f"unary operator {expr.op!r} is not supported", expr
        )

    def _binary(self, expr: Binary) -> tuple[str, str | None]:
        left, left_ty = self._expr(expr.left, want_value=True)
        right, right_ty = self._expr(expr.right, want_value=True)

        if expr.op in {"&&", "||"}:
            if left_ty != BOOL or right_ty != BOOL:
                raise WasmUnsupportedError(
                    f"{expr.op!r} expects bool operands", expr
                )
            if expr.op == "&&":
                code = f"(if (result i32) {left} (then {right}) (else (i32.const 0)))"
            else:
                code = f"(if (result i32) {left} (then (i32.const 1)) (else {right}))"
            return code, BOOL

        if expr.op in _COMPARISON:
            if left_ty != right_ty:
                raise WasmUnsupportedError(
                    f"mismatched operand types {left_ty} and {right_ty}", expr
                )
            prefix = "i64" if left_ty == I64 else "i32"
            return f"({prefix}.{_COMPARISON[expr.op]} {left} {right})", BOOL

        if expr.op in _ARITHMETIC:
            if left_ty != right_ty or left_ty not in {I64, BOOL}:
                raise WasmUnsupportedError(
                    f"mismatched operand types {left_ty} and {right_ty}", expr
                )
            prefix = "i64" if left_ty == I64 else "i32"
            return f"({prefix}.{_ARITHMETIC[expr.op]} {left} {right})", left_ty

        raise WasmUnsupportedError(
            f"binary operator {expr.op!r} is not supported", expr
        )

    def _call(self, expr: Call) -> tuple[str, str | None]:
        if not isinstance(expr.func, Ident):
            raise WasmUnsupportedError("indirect calls are not supported", expr)
        signature = self._signatures.get(expr.func.name)
        if signature is None:
            raise WasmUnsupportedError(
                f"unknown function {expr.func.name!r}", expr
            )
        if len(expr.args) != len(signature.params):
            raise WasmUnsupportedError(
                f"{expr.func.name!r} expects {len(signature.params)} arguments, "
                f"got {len(expr.args)}",
                expr,
            )
        args = []
        for arg, expected in zip(expr.args, signature.params):
            code, ty = self._expr(arg, want_value=True)
            if ty != expected:
                raise WasmUnsupportedError(
                    f"argument type {ty} does not match parameter type {expected}",
                    arg,
                )
            args.append(code)
        rendered = " ".join(args)
        call = f"(call $fn{signature.function_id}" + (f" {rendered}" if rendered else "")
        call += ")"
        return call, signature.ret

    def _if(self, expr: IfExpr, *, want_value: bool) -> tuple[str, str | None]:
        cond, cond_ty = self._expr(expr.cond, want_value=True)
        if cond_ty != BOOL:
            raise WasmUnsupportedError("if condition must be a bool", expr)
        then_code, then_ty = self._block(expr.then_block, want_value=want_value)
        if not want_value:
            if expr.else_branch is None:
                return f"(if {cond} (then {then_code}))", None
            else_code = self._branch_statement(expr.else_branch)
            return f"(if {cond} (then {then_code}) (else {else_code}))", None
        if expr.else_branch is None:
            raise WasmUnsupportedError(
                "an if expression needs an else branch", expr
            )
        else_code, else_ty = self._branch_value(expr.else_branch)
        if then_ty != else_ty:
            raise WasmUnsupportedError(
                f"if branches have different types {then_ty} and {else_ty}", expr
            )
        return (
            (
                f"(if (result {then_ty}) {cond} (then {then_code}) "
                f"(else {else_code}))"
            ),
            then_ty,
        )

    def _branch_value(self, branch: Any) -> tuple[str, str]:
        if isinstance(branch, Block):
            code, ty = self._block(branch, want_value=True)
        else:
            code, ty = self._expr(branch, want_value=True)
        if ty is None:
            raise WasmUnsupportedError(
                "if branch does not produce a value", branch
            )
        return code, ty

    def _branch_statement(self, branch: Any) -> str:
        if isinstance(branch, Block):
            code, ty = self._block(branch, want_value=False)
            return code
        code, ty = self._expr(branch, want_value=False)
        return f"(drop {code})" if _is_value(ty) else code

    def _assign(self, expr: Assign) -> tuple[str, str | None]:
        if not isinstance(expr.target, Ident):
            raise WasmUnsupportedError(
                "only local assignments are supported", expr
            )
        if expr.target.name not in self._locals:
            raise WasmUnsupportedError(
                f"unknown local {expr.target.name!r}", expr
            )
        index, ty = self._locals[expr.target.name]
        value, value_ty = self._expr(expr.value, want_value=True)
        if value_ty != ty:
            raise WasmUnsupportedError(
                f"local {expr.target.name!r} is {ty} but its value is {value_ty}",
                expr,
            )
        return f"(local.set $v{index} {value})", None

    def _compound_assign(self, expr: CompoundAssign) -> tuple[str, str | None]:
        if not isinstance(expr.target, Ident):
            raise WasmUnsupportedError(
                "only local assignments are supported", expr
            )
        if expr.target.name not in self._locals:
            raise WasmUnsupportedError(
                f"unknown local {expr.target.name!r}", expr
            )
        index, ty = self._locals[expr.target.name]
        if ty != I64:
            raise WasmUnsupportedError(
                "compound assignment is supported only for i64", expr
            )
        op = expr.op[:-1]
        if op not in _ARITHMETIC:
            raise WasmUnsupportedError(
                f"compound operator {expr.op!r} is not supported", expr
            )
        value, value_ty = self._expr(expr.value, want_value=True)
        if value_ty != I64:
            raise WasmUnsupportedError(
                f"compound assignment expects i64, got {value_ty}", expr
            )
        code = (
            f"(local.set $v{index} "
            f"(i64.{_ARITHMETIC[op]} (local.get $v{index}) {value}))"
        )
        return code, None

    def _while(self, expr: While) -> str:
        cycle = self._label
        self._label += 1
        cond, cond_ty = self._expr(expr.cond, want_value=True)
        if cond_ty != BOOL:
            raise WasmUnsupportedError("while condition must be a bool", expr)
        body, _ty = self._block(expr.body, want_value=False)
        return (
            f"(block $break{cycle} "
            f"(loop $continue{cycle} "
            f"(br_if $break{cycle} (i32.eqz {cond})) "
            f"{body} "
            f"(br $continue{cycle})))"
        )


class WasmBackend(BackendBase):
    """Compile a program to WAT and instantiate it with wasmtime."""

    name = "wasm"

    def compile(
        self, program: Program, module_name: str = "polyglot_wasm"
    ) -> dict[str, Any]:
        self._check_trust()
        codegen = WasmCodegen()
        wat = codegen.generate(program)
        try:
            import wasmtime
        except ImportError as error:  # pragma: no cover - depends on env
            raise WasmRuntimeUnavailableError(
                "the wasm backend needs wasmtime; install polyglot-lang[wasm]"
            ) from error

        try:
            engine = wasmtime.Engine()
            module = wasmtime.Module(engine, wat)
            store = wasmtime.Store(engine)
            instance = wasmtime.Instance(store, module, [])
        except Exception as error:
            raise WasmCompileError(
                f"failed to instantiate the generated WASM module: {error}"
            ) from error

        exports = instance.exports(store)
        namespace: dict[str, Any] = {}
        for name in codegen.exported_fns:
            signature = codegen._signatures[name]
            namespace[name] = _wrap_export(
                exports[name], store, signature.params, signature.ret
            )
        return namespace


def _wrap_export(
    function: Any,
    store: Any,
    params: tuple[str, ...],
    ret: str | None,
) -> Any:
    def wrapper(*args: Any) -> Any:
        if len(args) != len(params):
            raise TypeError(
                f"expected {len(params)} arguments, got {len(args)}"
            )
        converted = []
        for value, param_type in zip(args, params):
            if param_type == BOOL:
                if not isinstance(value, (bool, int)):
                    raise TypeError("bool parameters expect bool or int")
                converted.append(int(bool(value)))
            else:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError("i64 parameters expect int")
                converted.append(value)
        result = function(store, *converted)
        if ret == BOOL:
            return bool(result)
        return result

    return wrapper


__all__ = [
    "WasmBackend",
    "WasmCodegen",
    "WasmCompileError",
    "WasmRuntimeUnavailableError",
    "WasmUnsupportedError",
]
