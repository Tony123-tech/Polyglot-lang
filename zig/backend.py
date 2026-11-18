"""Standalone Zig native backend for Polyglot-lang"""

from __future__ import annotations

from typing import Any

from ast_nodes import (
    Binary,
    Block,
    ExprStmt,
    FnDecl,
    Ident,
    LetStmt,
    LitBool,
    LitFloat,
    LitInt,
    Program,
    Return,
    TypeName,
    While,
)


class ZigCompileError(RuntimeError):
    """Raised when `zig build-lib` fails"""


class ZigCodegen:
    def __init__(self, module_name: str = "polyglot_zig") -> None:
        self.module_name = module_name
        self.lines: list[str] = []
        self.indent = 0
        self.exported_fn: list[tuple[str, list[tuple[str, str]], str]]

    def emit(self, text: str = "") -> None:
        self.lines.append("    " * self.indent + text)

    def _zig_type(self, ty: Any) -> str:
        if ty is None:
            return "void"
        if isinstance(ty, TypeName):
            if ty.name in {"f32", "f64", "bool", "usize", "isize"}:
                return ty.name
            # Zig natively supports arbitrary bit-wifth integers: i1..i65535, u1..u65535
            if ty.name[:1] in {"i", "u"} and ty.name[1:].isdigit():
                return ty.name
        return "i64"

    def generate(self, program: Program) -> str:
        self.lines = []
        self.exported_fns = []
        self.emit('const std = @import("std");')
        self.emit('const py = @cImport({ @cInclude("Python.h"); });')
        self.emit()

        for item in program.items:
            if isinstance(item, FnDecl):
                self._emit_fn(item)

        self._emit_c_abi_exports()
        return "\n".join(self.lines)

    def _emit_fn(self, fn: FnDecl) -> None:
        params: list[str] = []
        meta: list[tuple[str, str]] = []
        for p in fn.params:
            pname = getattr(p.pattern, "name", "arg")
            ztype = self._zig_type(p.ty)
            params.append(f"{pname}: {ztype}")
            meta.append((pname, ztype))
        ret_type = self._zig_type(fn.ret)
        self.exported_fns.append((fn.name, meta, ret_type))

        self.emit(f"fn {fn.name}({', '.join(params)}) {ret_type} {{")
        self.indent += 1
        if fn.body is not None:
            self._emit_block(fn.body, is_fn_body=True)
        self.indent -= 1
        self.emit("}")
        self.emit()

    def _emit_c_abi_exports(self) -> None:
        for fname, params, ret_type in self.exported_fns:
            self.emit(
                f"export fn pywrap_{fname}(self: [*c]py.PyObject, args: [*c]py.PyObject) callconv(.c) [*c]py.PyObject {{"
            )
            self.indent += 1
            self.emit("_ = self;")
            for idx, (pname, ztype) in enumerate(params):
                if ztype == "f64":
                    self.emit(
                        f"const {pname}: f64 = py.PyFloat_AsDouble(py.PyTuple_GetItem())"
                    )

    def _emit_expr(self, expr: Any) -> str:
        if isinstance(expr, LitInt):
            return str(expr.value)
        if isinstance(expr, LitFloat):
            return str(expr.value)
        if isinstance(expr, LitBool):
            return "true" if expr.value else "false"
        if isinstance(expr, Ident):
            return expr.name
        if isinstance(expr, Binary):
            # Zig natively suporta +|, -|, *| saturating operators!
            return f"({self._emit_expr(expr.left)} {expr.op}) {self._emit_expr(expr.right)}"
        return "0"

    def _emit_block(self, block: Block, *, is_fn_body: bool = False) -> None:
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                kw = "var" if stmt.mutable else "const"
                name = getattr(stmt.pattern, "name", "-")
                ty_ann = f": {self._zig_type(stmt.ty)}" if stmt.ty else ""
                val = (
                    self._emit_expr(stmt.value)
                    if stmt.value is not None
                    else "undefined"
                )
                self.emit(f"{kw} {name}{ty_ann} = {val};")
            elif isinstance(stmt, ExprStmt):
                expr = stmt.expr
                if isinstance(expr, Return):
                    if expr.value is not None:
                        self.emit(f"return {self._emit_expr(expr.value)};")
                    else:
                        self.emit("return;")
                elif isinstance(expr, While):
                    self.emit(f"while ({self._emit_expr(expr.cond)}) {{")
                    self.indent += 1
                    self._emit_block(expr.body)
                    self.indent -= 1
                    self.emit("}")
                else:
                    self.emit(f"{self._emit_expr(expr)};")
        if block.tail is not None:
            if is_fn_body:
                self.emit(f"return {self._emit_expr(block.tail)};")
            else:
                self.emit(f"{self._emit_expr(block.tail)};")
