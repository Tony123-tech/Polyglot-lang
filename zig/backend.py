from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from ast_nodes import (
    Assign,
    Binary,
    Block,
    Call,
    Cast,
    CompoundAssign,
    ExprStmt,
    FnDecl,
    For,
    Ident,
    Index,
    LetStmt,
    LitBool,
    LitFloat,
    LitInt,
    Paren,
    Program,
    Range,
    Return,
    TypeName,
    TypeRef,
    TypeSlice,
    While,
)
from ast_nodes import (
    Path as AstPath,
)
from backends.base import BackendBase


class ZigCompileError(RuntimeError):
    pass


def _ctypes_for(ztype: str) -> Any:
    if ztype.startswith("[*]"):
        return ctypes.c_void_p
    if ztype == "f64":
        return ctypes.c_double
    if ztype == "f32":
        return ctypes.c_float
    if ztype == "bool":
        return ctypes.c_bool
    if ztype == "void":
        return None
    return (
        ctypes.c_uint64
        if ztype in {"usize", "isize"} or ztype.startswith("u")
        else ctypes.c_int64
    )


def _cabi_type(ztype: str) -> str:
    if ztype.startswith("[*]") or ztype in {"f32", "f64", "bool", "void"}:
        return ztype
    return "u64" if ztype.startswith("u") else "i64"


class ZigCodegen:
    def __init__(self, module_name: str = "polyglot_zig") -> None:
        self.module_name = module_name
        self.lines: list[str] = []
        self.indent = 0
        self.exported_fns: list[tuple[str, list[tuple[str, str]], str]] = []
        self._helpers: list[str] = []
        self._locals: list[tuple[str, str]] = []

    def emit(self, text: str = "") -> None:
        self.lines.append("    " * self.indent + text)

    def _zig_type(self, ty: Any) -> str:
        if ty is None:
            return "void"
        if isinstance(ty, TypeRef):
            if isinstance(ty.inner, TypeSlice):
                return f"[*]{'' if ty.mutable else 'const '}{self._zig_type(ty.inner.inner)}"
            return self._zig_type(ty.inner)
        if isinstance(ty, TypeName):
            if ty.name.startswith("Simd") and ty.name[4:].isdigit() and len(ty.args) == 1:
                return f"@Vector({ty.name[4:]}, {self._zig_type(ty.args[0])})"
            if ty.name in {"f32", "f64", "bool", "usize", "isize"} or (
                ty.name[:1] in {"i", "u"} and ty.name[1:].isdigit()
            ):
                return ty.name
        return "i64"

    def generate(self, program: Program) -> str:
        self.lines, self.exported_fns, self._helpers = [], [], []
        if any(
            isinstance(i, FnDecl) and any(a.name == "parallel" for a in i.attributes)
            for i in program.items
        ):
            self.emit('const std = @import("std");\n')
        for item in program.items:
            if isinstance(item, FnDecl):
                self._emit_fn(item)
        self.lines.extend(self._helpers)
        for fname, params, ret_type in self.exported_fns:
            c_params = ", ".join(f"{p}: {_cabi_type(t)}" for p, t in params)
            c_ret = _cabi_type(ret_type)
            self.emit(f"export fn __cabi_{fname}({c_params}) callconv(.c) {c_ret} {{")
            self.indent += 1
            args = [
                p
                if (t.startswith("[*]") or t in {"f32", "f64", "bool", "u64", "i64"})
                else (
                    f"@as({t}, @intCast({p}))"
                    if t in {"usize", "isize"}
                    else f"@as({t}, @truncate({p}))"
                )
                for p, t in params
            ]
            if ret_type == "void":
                self.emit(f"{fname}({', '.join(args)});")
            else:
                self.emit(f"const res = {fname}({', '.join(args)});")
                self.emit(
                    "return res;"
                    if ret_type in {"f32", "f64", "bool", "u64", "i64"}
                    else f"return @as({c_ret}, @intCast(res));"
                )
            self.indent -= 1
            self.emit("}\n")
        return "\n".join(self.lines)

    def _emit_fn(self, fn: FnDecl) -> None:
        self._locals = []
        is_par = any(a.name == "parallel" for a in fn.attributes)
        meta = [
            (getattr(p.pattern, "name", "arg"), self._zig_type(p.ty)) for p in fn.params
        ]
        ret = self._zig_type(fn.ret)
        self.exported_fns.append((fn.name, meta, ret))
        self.emit(f"fn {fn.name}({', '.join(f'{n}: {t}' for n, t in meta)}) {ret} {{")
        self.indent += 1
        if fn.body:
            self._emit_block(fn.body, True, is_par, meta)
        self.indent -= 1
        self.emit("}\n")

    def _emit_block(
        self,
        block: Block,
        is_fn: bool = False,
        is_par: bool = False,
        params: list[tuple[str, str]] | None = None,
    ) -> None:
        for s in block.stmts:
            if isinstance(s, LetStmt):
                z_ty = self._zig_type(s.ty) if s.ty else ""
                if is_fn and z_ty:
                    self._locals.append((getattr(s.pattern, "name", "_"), z_ty))
                self.emit(
                    f"{'var' if s.mutable else 'const'} {getattr(s.pattern, 'name', '_')}{f': {z_ty}' if z_ty else ''} = {self._emit_expr(s.value) if s.value else 'undefined'};"
                )
            elif isinstance(s, ExprStmt):
                e = s.expr
                if isinstance(e, For) and isinstance(e.iter, Range):
                    v = getattr(e.pattern, "name", "i")
                    st = self._emit_expr(e.iter.start) if e.iter.start else "0"
                    en = self._emit_expr(e.iter.end) if e.iter.end else "0"
                    if is_par:
                        cid = len(self._helpers)
                        fields = "\n".join(
                            f"    {n}: {t}," for n, t in ((params or []) + self._locals)
                        )
                        unpack = "\n".join(
                            f"    const {n} = ctx.{n}; _ = &{n};"
                            for n, _ in ((params or []) + self._locals)
                        )
                        saved, self.lines, self.indent = self.lines, [], 2
                        self._emit_block(e.body)
                        body_str = "\n".join(self.lines)
                        self.lines, self.indent = saved, self.indent
                        self._helpers.append(
                            f"const Ctx_{cid} = struct {{\n{fields}\n}};\nfn worker_{cid}(ctx: Ctx_{cid}, s: usize, e: usize) void {{\n{unpack}\n    var {v}: usize = s;\n    while ({v} < e) : ({v} += 1) {{\n{body_str}\n    }}\n}}\n"
                        )
                        inits = ", ".join(
                            f".{n} = {n}" for n, _ in ((params or []) + self._locals)
                        )
                        self.emit(
                            f"const __end: usize = {en}{' + 1' if e.iter.inclusive else ''};\n    if (__end > {st}) {{\n        const __tot = __end - {st};\n        const __w = @min(__tot, @min(16, std.Thread.getCpuCount() catch 4));\n        var __threads: [16]std.Thread = undefined;\n        var __spawned: usize = 0;\n        const __ctx = Ctx_{cid}{{ {inits} }};\n        const __chk = (__tot + __w - 1) / __w;\n        var __i: usize = 0;\n        while (__i < __w) : (__i += 1) {{\n            const __cs = {st} + __i * __chk;\n            if (__cs >= __end) break;\n            __threads[__spawned] = std.Thread.spawn(.{{}}, worker_{cid}, .{{ __ctx, __cs, @min(__cs + __chk, __end) }}) catch unreachable;\n            __spawned += 1;\n        }}\n        var __j: usize = 0;\n        while (__j < __spawned) : (__j += 1) {{ __threads[__j].join(); }}\n    }}"
                        )
                    else:
                        self.emit(
                            f"var {v}: usize = {st}; while ({v} {'<=' if e.iter.inclusive else '<'} {en}) : ({v} += 1) {{"
                        )
                        self.indent += 1
                        self._emit_block(e.body)
                        self.indent -= 1
                        self.emit("}")
                elif isinstance(e, Return):
                    self.emit(
                        f"return {self._emit_expr(e.value)};" if e.value else "return;"
                    )
                elif isinstance(e, While):
                    self.emit(f"while ({self._emit_expr(e.cond)}) {{")
                    self.indent += 1
                    self._emit_block(e.body)
                    self.indent -= 1
                    self.emit("}")
                else:
                    self.emit(f"{self._emit_expr(e)};")
        if block.tail is not None:
            self.emit(f"{'return ' if is_fn else ''}{self._emit_expr(block.tail)};")

    def _emit_expr(self, e: Any) -> str:
        if isinstance(e, Paren):
            return f"({self._emit_expr(e.inner)})"
        if isinstance(e, Cast):
            return f"@as({self._zig_type(e.ty)}, @intCast({self._emit_expr(e.expr)}))"
        if isinstance(e, (LitInt, LitFloat)):
            return str(e.value)
        if isinstance(e, LitBool):
            return "true" if e.value else "false"
        if isinstance(e, Ident):
            return e.name
        if isinstance(e, Index):
            return f"{self._emit_expr(e.receiver)}[{self._emit_expr(e.index)}]"
        if isinstance(e, Assign):
            return f"{self._emit_expr(e.target)} = {self._emit_expr(e.value)}"
        if isinstance(e, CompoundAssign):
            return f"{self._emit_expr(e.target)} {e.op} {self._emit_expr(e.value)}"
        if isinstance(e, Binary):
            return f"({self._emit_expr(e.left)} {e.op} {self._emit_expr(e.right)})"
        if (
            isinstance(e, Call)
            and isinstance(e.func, AstPath)
            and e.func.segments[:1] == ["Simd"]
        ):
            m, a = e.func.segments[-1], e.args
            if m == "splat" and len(a) == 1:
                return f"@splat({self._emit_expr(a[0])})"
            if m == "load" and len(a) == 3:
                return f"{self._emit_expr(a[0])}[{self._emit_expr(a[1])}..][0..{self._emit_expr(a[2])}].*"
            if m == "store" and len(a) == 4:
                return f"{self._emit_expr(a[0])}[{self._emit_expr(a[1])}..][0..{self._emit_expr(a[2])}].* = {self._emit_expr(a[3])}"
        return "0"


class ZigBackend(BackendBase):
    name = "zig"

    def __init__(
        self, cache_dir: str | Path | None = None, *, trusted: bool = True
    ) -> None:
        super().__init__(trusted=trusted)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "polyglot-zig")
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def build(
        self, program: Program, module_name: str = "polyglot_zig", **_: Any
    ) -> Path:
        self._check_trust()
        cg = ZigCodegen(module_name=module_name)
        zig_src = cg.generate(program)
        build_dir = self.cache_dir / hashlib.sha256(zig_src.encode()).hexdigest()[:16]
        build_dir.mkdir(parents=True, exist_ok=True)
        dylib = build_dir / f"lib{module_name}.dylib"
        if not dylib.exists():
            src = build_dir / f"{module_name}.zig"
            src.write_text(zig_src, encoding="utf-8")
            res = subprocess.run(
                [
                    "zig",
                    "build-lib",
                    str(src),
                    "-dynamic",
                    "-O",
                    "ReleaseFast",
                    f"-femit-bin={dylib}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                raise ZigCompileError(f"Zig compilation failed:\n{res.stderr}")
        return dylib

    def compile(
        self, program: Program, module_name: str = "polyglot_zig"
    ) -> dict[str, Any]:
        self._check_trust()
        cg = ZigCodegen(module_name=module_name)
        zig_src = cg.generate(program)
        build_dir = self.cache_dir / hashlib.sha256(zig_src.encode()).hexdigest()[:16]
        build_dir.mkdir(parents=True, exist_ok=True)
        dylib = build_dir / f"lib{module_name}.dylib"
        if not dylib.exists():
            src = build_dir / f"{module_name}.zig"
            src.write_text(zig_src, encoding="utf-8")
            res = subprocess.run(
                [
                    "zig",
                    "build-lib",
                    str(src),
                    "-dynamic",
                    "-O",
                    "ReleaseFast",
                    f"-femit-bin={dylib}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                raise ZigCompileError(f"Zig compilation failed:\n{res.stderr}")
        lib = ctypes.CDLL(str(dylib))
        ns: dict[str, Any] = {"_zig_lib": lib}
        for fname, params, ret in cg.exported_fns:
            fn = getattr(lib, f"__cabi_{fname}")
            fn.argtypes = [_ctypes_for(t) for _, t in params]
            fn.restype = _ctypes_for(ret)
            ns[fname] = fn
        return ns
