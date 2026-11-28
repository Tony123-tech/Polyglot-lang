"""Zig backend for polyglot-lang, supporting Zig-exclusive semantics.

The emitter targets idiomatic Zig: ``comptime`` generic declarations, tagged
unions, optionals, slices, error unions, explicit allocators, ``defer`` /
``errdefer`` cleanup and C interop through ``@cImport`` / ``extern fn``.

A compact C ABI shim is generated for functions whose parameters and result can
cross ``ctypes`` safely.  Functions using slices, strings, optionals or error
unions remain callable from Zig but are intentionally not exposed through the
C ABI instead of being silently truncated.
"""

from __future__ import annotations

import ctypes
import json
import re
from pathlib import Path
from typing import Any, NoReturn

from backends.base import BackendBase
from compiler.core.ast_nodes import (
    ArrayExpr,
    Assign,
    Await,
    Binary,
    Block,
    Break,
    Call,
    Cast,
    Closure,
    CompoundAssign,
    ConstDecl,
    Continue,
    DeferStmt,
    EnumDecl,
    ErrDeferStmt,
    ExprStmt,
    Field,
    FnDecl,
    For,
    Ident,
    IfExpr,
    IfLet,
    ImplBlock,
    Index,
    ItemStmt,
    LetStmt,
    LitBool,
    LitChar,
    LitFloat,
    LitInt,
    LitStr,
    Loop,
    MacroCall,
    MatchExpr,
    MethodCall,
    OptionNone,
    OptionSome,
    Paren,
    PatBinding,
    PatConst,
    PatEnum,
    PatIdent,
    PatLiteral,
    PatOr,
    PatPath,
    PatRange,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    PatTuple,
    PatWildcard,
    Program,
    Range,
    RepeatArray,
    ResultErr,
    ResultOk,
    Return,
    StructDecl,
    StructLiteral,
    Try,
    TupleExpr,
    TypeAlias,
    TypeArray,
    TypeDynTrait,
    TypeFn,
    TypeImplTrait,
    TypeName,
    TypeRef,
    TypeSlice,
    TypeTuple,
    TypeUnit,
    Unary,
    UseDecl,
    While,
    WhileLet,
)
from compiler.core.ast_nodes import (
    Path as AstPath,
)
from compiler.frontend.errors import PolyglotError
from zig.build import (
    ZigBuildError,
    build_dynamic_library,
    build_dynamic_library_direct,
    normalize_optimize,
    zig_available,
)
from zig.intrinsics import (
    CONVERSION_INTRINSICS,
    MEMORY_INTRINSICS,
    OVERFLOW_INTRINSICS,
    ROTATE_INTRINSICS,
    TYPE_INTRINSICS,
    WRAPPING_INTRINSICS,
    IntrinsicError,
    bit_width,
    is_float,
    require_arithmetic_operand,
    require_float_operand,
    require_matching_operands,
    require_type_operand,
    require_value_operand,
    shape_for,
)


class ZigCompileError(RuntimeError):
    pass


def _fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    help_text: str | None = None,
) -> NoReturn:
    """Raise a structured Zig-backend diagnostic (E001–E008)."""
    raise PolyglotError(
        code, message, reason=reason, help_text=help_text, phase="zig-backend"
    )


_VECTOR_RE = re.compile(r"^@Vector\((\d+), (f32|f64|[iu]\d+)\)$")
_VECTOR_ABI_CACHE: dict[str, type[ctypes.Structure]] = {}
_PILLAR_ERRORS = tuple(f"E{index:03d}" for index in range(1, 10))
_INT_RE = re.compile(r"^[iu]\d+$")


def _vector_shape(ztype: str) -> tuple[int, str] | None:
    match = _VECTOR_RE.fullmatch(ztype)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _vector_struct_name(ztype: str) -> str:
    count, scalar = _vector_shape(ztype) or (0, "")
    return f"__cabi_vector_{count}_{scalar}".replace(".", "_")


def _ctypes_for(ztype: str) -> Any:
    shape = _vector_shape(ztype)
    if shape is not None:
        count, scalar = shape
        cached = _VECTOR_ABI_CACHE.get(ztype)
        if cached is None:
            element = _ctypes_for(scalar)
            cached = type(
                _vector_struct_name(ztype),
                (ctypes.Structure,),
                {"_fields_": [("v", element * count)]},
            )
            _VECTOR_ABI_CACHE[ztype] = cached
        return cached
    if ztype.startswith("[*]") or ztype in {"*anyopaque", "*const anyopaque"}:
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


def _cabi_type(ztype: str) -> str | None:
    """Return the C-ABI spelling of *ztype*, or ``None`` when unsupported."""
    if _vector_shape(ztype) is not None:
        return _vector_struct_name(ztype)
    if ztype.startswith("[*]") and "anyopaque" not in ztype:
        return ztype
    if ztype in {"f32", "f64", "bool", "void"}:
        return ztype
    if ztype in {"usize", "isize"} or _INT_RE.fullmatch(ztype):
        return "u64" if ztype.startswith("u") or ztype == "usize" else "i64"
    return None


def _cabi_arg(name: str, ztype: str) -> str:
    if _vector_shape(ztype) is not None:
        return f"@bitCast({name}.v)"
    if ztype.startswith("[*]") or ztype in {"f32", "f64", "bool", "u64", "i64"}:
        return name
    if ztype in {"usize", "isize"}:
        return f"@as({ztype}, @intCast({name}))"
    return f"@as({ztype}, @truncate({name}))"


def _cabi_return(ztype: str, cabi: str) -> str:
    if _vector_shape(ztype) is not None:
        return "return .{ .v = @bitCast(res) };"
    if ztype in {"f32", "f64", "bool", "u64", "i64"}:
        return "return res;"
    if _INT_RE.fullmatch(ztype) or ztype in {"usize", "isize"}:
        return f"return @as({cabi}, @intCast(res));"
    return "return res;"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _attr_arg(fn: FnDecl, *names: str) -> str | None:
    for attr in fn.attributes:
        if attr.name in names and attr.args:
            return attr.args.strip().strip('"')
    return None


def _has_attr(fn: FnDecl, *names: str) -> bool:
    return any(attr.name in names for attr in fn.attributes)


def _error_expr(value: Any, emit: Any) -> str:
    if isinstance(value, AstPath) and value.segments[-1] in _PILLAR_ERRORS:
        return f"error.{value.segments[-1]}"
    if isinstance(value, Ident) and value.name in _PILLAR_ERRORS:
        return f"error.{value.name}"
    """Map an ``Err(n)`` payload to a structured ``PillarError`` member."""
    if isinstance(value, LitInt) and 1 <= value.value <= 8:
        return f"error.E{value.value:03d}"
    return f"error.{_PILLAR_ERRORS[3]}"


class ZigCodegen:
    """Emit one deterministic Zig module from a Polyglot program."""

    def __init__(self, module_name: str = "polyglot_zig") -> None:
        self.module_name = module_name
        self.lines: list[str] = []
        self.indent = 0
        self.exported_fns: list[tuple[str, list[tuple[str, str]], str, bool]] = []
        self._helpers: list[str] = []
        self._locals: dict[str, str] = {}
        self._vector_abi_types: dict[str, str] = {}
        self._generic_fns: set[str] = set()
        self._struct_names: set[str] = set()
        self._enum_names: set[str] = set()
        self._method_names: dict[tuple[str, str], str] = {}
        self._method_self_pointer: dict[tuple[str, str], bool] = {}
        self._fn_returns: dict[str, str] = {}
        self._fn_params: dict[str, list[str]] = {}
        self._struct_fields: dict[str, dict[str, str]] = {}
        self._self_type: str | None = None
        self._current_ret = "void"
        self._label = 0
        self._cimports: list[str] = []
        self._extern_fns: list[FnDecl] = []
        self._impl_fns: list[tuple[str, FnDecl]] = []
        self._link_libs: list[str] = []
        self._uses_defer = False
        self._parallel_fallback = False
        self._allocator_name: str | None = None
        self._block_labels: list[str] = []

    # ------------------------------------------------------------------
    # Low-level emission
    # ------------------------------------------------------------------

    def emit(self, text: str = "") -> None:
        self.lines.append("    " * self.indent + text)

    def _next_label(self, prefix: str = "blk") -> str:
        self._label += 1
        return f"{prefix}_{self._label}"

    def _capture(self) -> tuple[list[str], int]:
        saved = self.lines
        width = self.indent
        self.lines = []
        self.indent = 0
        return saved, width

    def _restore(self, saved: tuple[list[str], int]) -> list[str]:
        body = self.lines
        self.lines, self.indent = saved
        return body

    # ------------------------------------------------------------------
    # Types
    # ------------------------------------------------------------------

    def _zig_type(self, ty: Any) -> str:
        if ty is None:
            return "void"
        if isinstance(ty, TypeUnit):
            return "void"
        if isinstance(ty, TypeRef):
            if isinstance(ty.inner, TypeSlice):
                qualifier = "" if ty.mutable else "const "
                return f"[*]{qualifier}{self._zig_type(ty.inner.inner)}"
            if isinstance(ty.inner, TypeArray):
                return f"*{self._zig_type(ty.inner)}"
            return f"*{'const ' if not ty.mutable else ''}{self._zig_type(ty.inner)}"
        if isinstance(ty, TypeSlice):
            return f"[]{self._zig_type(ty.inner)}"
        if isinstance(ty, TypeArray):
            size = self._emit_expr(ty.size)
            return f"[{size}]{self._zig_type(ty.inner)}"
        if isinstance(ty, TypeTuple):
            if not ty.elems:
                return "void"
            return "struct { " + ", ".join(self._zig_type(t) for t in ty.elems) + " }"
        if isinstance(ty, TypeFn):
            params = ", ".join(self._zig_type(t) for t in ty.params)
            ret = self._zig_type(ty.ret)
            return f"*const fn({params}) callconv(.c) {ret}"
        if isinstance(ty, (TypeImplTrait, TypeDynTrait)):
            return "*anyopaque"
        if not isinstance(ty, TypeName):
            return "i64"

        name = ty.name
        if name in {"f32", "f64", "bool", "usize", "isize", "void"}:
            return name
        if _INT_RE.fullmatch(name):
            return name
        if name == "String":
            return "[]const u8"
        if name == "Allocator":
            return "std.mem.Allocator"
        if name == "PillarError":
            return "PillarError"
        if name == "Self":
            return self._self_type or "Self"
        if name == "Simd" and ty.args:
            return self._zig_type(ty.args[0])
        if name.startswith("Simd") and name[4:].isdigit() and ty.args:
            return f"@Vector({name[4:]}, {self._zig_type(ty.args[0])})"
        if name == "Option" and ty.args:
            return f"?{self._zig_type(ty.args[0])}"
        if name == "Result" and ty.args:
            return f"PillarError!{self._zig_type(ty.args[0])}"
        if name == "Vec" and ty.args:
            return f"[]{self._zig_type(ty.args[0])}"
        if name in {"Box", "Rc", "Arc"} and ty.args:
            return f"*{self._zig_type(ty.args[0])}"
        if name == "HashMap" and len(ty.args) == 2:
            return f"std.AutoHashMap({self._zig_type(ty.args[0])}, {self._zig_type(ty.args[1])})"
        if ty.args:
            args = ", ".join(self._zig_type(arg) for arg in ty.args)
            return f"{name}({args})"
        return name

    def _params(self, fn: FnDecl) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for index, param in enumerate(fn.params):
            pattern = param.pattern
            if isinstance(pattern, PatRef) and isinstance(pattern.inner, PatIdent):
                name = pattern.inner.name
            elif isinstance(pattern, (PatIdent, PatBinding)):
                name = pattern.name
            else:
                name = f"arg{index}"
            if name == "self":
                owner = self._self_type or "Self"
                ztype = f"*{owner}" if isinstance(pattern, PatRef) else owner
            else:
                ztype = self._zig_type(param.ty) if param.ty else "anytype"
            rows.append((name, ztype))
        return rows

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _infer_type(self, expr: Any) -> str:
        if isinstance(expr, Ident):
            return self._locals.get(expr.name, "i64")
        if isinstance(expr, Paren):
            return self._infer_type(expr.inner)
        if isinstance(expr, OptionSome):
            return f"?{self._infer_type(expr.value)}"
        if isinstance(expr, ResultOk):
            return f"anyerror!{self._infer_type(expr.value)}"
        if isinstance(expr, OptionNone):
            return "?i64"
        if isinstance(expr, LitInt):
            return "i64"
        if isinstance(expr, LitFloat):
            return "f64"
        if isinstance(expr, Binary) and (
            self._infer_type(expr.left) in {"f32", "f64"}
            or self._infer_type(expr.right) in {"f32", "f64"}
        ):
            return "f64"
        if isinstance(expr, LitBool):
            return "bool"
        if isinstance(expr, LitStr):
            return "[]const u8"
        if isinstance(expr, StructLiteral):
            return self._emit_path_text(expr.path)
        if isinstance(expr, Field):
            if expr.name == "len":
                return "usize"
            receiver_type = self._infer_type(expr.receiver)
            clean_owner = receiver_type.lstrip("*").removeprefix("const ").strip()
            fields = self._struct_fields.get(clean_owner)
            if fields and expr.name in fields:
                return fields[expr.name]
            return "i64"
        if isinstance(expr, MethodCall) and expr.method == "len":
            return "usize"
        if isinstance(expr, (Call, AstPath)):
            if isinstance(expr, AstPath):
                name = expr.segments[-1]
            elif isinstance(expr.func, (Ident, AstPath)):
                name = (
                    expr.func.name
                    if isinstance(expr.func, Ident)
                    else expr.func.segments[-1]
                )
            else:
                return "i64"
            return self._fn_returns.get(name, "i64")
        if isinstance(expr, IfExpr) and expr.then_block.tail is not None:
            return self._infer_type(expr.then_block.tail)
        return "i64"

    # ------------------------------------------------------------------
    # Module assembly
    # ------------------------------------------------------------------

    def generate(self, program: Program) -> str:
        self.lines = []
        self.exported_fns = []
        self._helpers = []
        self._locals = {}
        self._vector_abi_types = {}
        self._generic_fns = set()
        self._struct_names = set()
        self._enum_names = set()
        self._method_names = {}
        self._method_self_pointer = {}
        self._fn_returns = {}
        self._fn_params = {}
        self._struct_fields = {}
        self._self_type = None
        self._current_ret = "void"
        self._label = 0
        self._cimports = []
        self._extern_fns = []
        self._impl_fns = []
        self._link_libs = []
        self._uses_defer = False
        self._parallel_fallback = False
        self._allocator_name = None
        self._block_labels = []

        self._scan(program)
        self.emit("// Generated by polyglot-lang (Zig backend).")
        self.emit('const std = @import("std");')
        self.emit()
        if self._cimports:
            self.emit("const c = @cImport({")
            self.indent += 1
            for header in self._cimports:
                self.emit(f"@cInclude({_quote(header)});")
            self.indent -= 1
            self.emit("});")
            self.emit()
        self.emit("pub const PillarError = error{ " + ", ".join(_PILLAR_ERRORS) + " };")
        self.emit()
        self._emit_zig_helpers()

        for item in program.items:
            if isinstance(item, StructDecl):
                self._emit_struct(item)
            elif isinstance(item, EnumDecl):
                self._emit_enum(item)
            elif isinstance(item, TypeAlias):
                self._emit_type_alias(item)
            elif isinstance(item, ConstDecl):
                self._emit_const(item)
        for owner, method in self._impl_fns:
            self._emit_fn(method, owner=owner)
        for fn in self._extern_fns:
            self._emit_extern_fn(fn)
        for item in program.items:
            if isinstance(item, FnDecl):
                self._emit_fn(item)

        for ztype, name in sorted(self._vector_abi_types.items()):
            count, scalar = _vector_shape(ztype) or (0, "f64")
            self.emit(f"const {name} = extern struct {{ v: [{count}]{scalar} }};\n")

        self._emit_cabi_shims()
        return "\n".join(self.lines)

    def _scan(self, program: Program) -> None:
        for item in program.items:
            if isinstance(item, FnDecl):
                if item.generics:
                    self._generic_fns.add(item.name)
                self._fn_returns[item.name] = self._zig_type(item.ret)
                self._fn_params[item.name] = [
                    self._zig_type(p.ty) if p.ty else "anytype" for p in item.params
                ]
                for param in item.params:
                    ztype = self._zig_type(param.ty) if param.ty else ""
                    if _vector_shape(ztype) is not None:
                        self._vector_abi_types.setdefault(
                            ztype, _vector_struct_name(ztype)
                        )
                ret_type = self._zig_type(item.ret)
                if _vector_shape(ret_type) is not None:
                    self._vector_abi_types.setdefault(
                        ret_type, _vector_struct_name(ret_type)
                    )
                if _has_attr(item, "extern", "c_extern", "external"):
                    self._extern_fns.append(item)
                header = _attr_arg(item, "cinclude", "cImport", "c_import")
                if header and header not in self._cimports:
                    self._cimports.append(header)
                for attr in item.attributes:
                    if attr.name in {"link", "link_lib"} and attr.args:
                        self._link_libs.append(attr.args.strip().strip('"'))
            elif isinstance(item, StructDecl):
                self._struct_names.add(item.name)
                self._struct_fields[item.name] = {
                    fname: self._zig_type(ftype) for fname, ftype, _ in item.fields
                }
            elif isinstance(item, EnumDecl):
                self._enum_names.add(item.name)
            elif isinstance(item, ImplBlock):
                owner = (
                    item.ty.name
                    if isinstance(item.ty, TypeName)
                    else self._zig_type(item.ty)
                )
                for member in item.items:
                    if isinstance(member, FnDecl):
                        name = f"{owner}_{member.name}"
                        self._method_names[(owner, member.name)] = name
                        self._method_self_pointer[(owner, member.name)] = bool(
                            member.params
                        ) and isinstance(member.params[0].pattern, PatRef)
                        self._fn_returns[member.name] = self._zig_type(member.ret)
                        self._impl_fns.append((owner, member))
            elif isinstance(item, UseDecl):
                # `use std::mem;` is already covered by the std import.
                continue
            elif isinstance(item, (TypeAlias, ConstDecl)):
                continue
            else:
                path = getattr(item, "path", None)
                if (
                    isinstance(path, str)
                    and path.endswith((".h", ".hpp"))
                    and path not in self._cimports
                ):
                    self._cimports.append(path)

    def _emit_zig_helpers(self) -> None:
        helper = """fn _zig_sat_add(a: anytype, b: anytype) @TypeOf(a) {
    const res = @addWithOverflow(a, b);
    if (res[1] != 0) {
        return if (a > 0) std.math.maxInt(@TypeOf(a)) else std.math.minInt(@TypeOf(a));
    }
    return res[0];
}

fn _zig_sat_sub(a: anytype, b: anytype) @TypeOf(a) {
    const res = @subWithOverflow(a, b);
    if (res[1] != 0) {
        return if (a > 0) std.math.minInt(@TypeOf(a)) else std.math.maxInt(@TypeOf(a));
    }
    return res[0];
}

fn _zig_sat_mul(a: anytype, b: anytype) @TypeOf(a) {
    const res = @mulWithOverflow(a, b);
    if (res[1] != 0) {
        return if ((a > 0) == (b > 0)) std.math.maxInt(@TypeOf(a)) else std.math.minInt(@TypeOf(a));
    }
    return res[0];
}

fn _zig_optional(value: anytype) ?@TypeOf(value) {
    return value;
}

fn _zig_bitcast(comptime T: type, value: anytype) T {
    return @bitCast(value);
}

fn _zig_fail(code: []const u8, message: []const u8, line: u32, col: u32) noreturn {
    std.debug.print("error[{s}]: {s}\\n  at: line {d}, column {d}\\n", .{ code, message, line, col });
    std.process.exit(1);
}

fn _zig_div(a: anytype, b: anytype, line: u32, col: u32) @TypeOf(a) {
    if (b == 0) _zig_fail("E002", "division by zero", line, col);
    return @divTrunc(a, b);
}

fn _zig_rem(a: anytype, b: anytype, line: u32, col: u32) @TypeOf(a) {
    if (b == 0) _zig_fail("E002", "remainder by zero", line, col);
    return @rem(a, b);
}

fn _zig_assert(condition: bool, message: []const u8, line: u32, col: u32) void {
    if (!condition) _zig_fail("E002", message, line, col);
}
"""
        self.lines.append(helper)

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _emit_generic_head(self, name: str, generics: list[Any]) -> None:
        params = ", ".join(f"comptime {generic.name}: type" for generic in generics)
        self.emit(f"fn {name}({params}) type {{")
        self.indent += 1
        self.emit("return struct {")
        self.indent += 1

    def _emit_struct(self, decl: StructDecl) -> None:
        name = decl.name
        if _has_attr(decl, "packed"):
            self._emit_packed_struct(decl)
            return
        if decl.generics:
            params = ", ".join(
                f"comptime {generic.name}: type" for generic in decl.generics
            )
            self.emit(f"fn {name}({params}) type {{")
            self.indent += 1
            self.emit("return struct {")
            self.indent += 1
            for field_name, field_type, _ in decl.fields:
                self.emit(f"{field_name}: {self._zig_type(field_type)},")
            self.indent -= 1
            self.emit("};")
            self.indent -= 1
            self.emit("}\n")
            return
        public = "pub " if decl.is_pub else ""
        self.emit(f"{public}const {name} = struct {{")
        self.indent += 1
        for field_name, field_type, _ in decl.fields:
            self.emit(f"{field_name}: {self._zig_type(field_type)},")
        self.indent -= 1
        self.emit("};\n")

    def _emit_packed_struct(self, decl: StructDecl) -> None:
        """Emit a Zig ``packed struct`` with a compile-time bit-size guarantee.

        Every field must have a known bit width and none may be a float: a
        packed struct's size is the sum of its fields, and a float would leave
        the layout to the backend rather than to the declaration.
        """
        name = decl.name
        if decl.generics:
            _fail(
                "E003",
                f"packed struct {name} cannot be generic",
                help_text="declare a concrete packed struct, or drop #[packed]",
            )
        if not decl.fields:
            _fail("E003", f"packed struct {name} has no fields")
        total = 0
        rows: list[str] = []
        for field_name, field_type, _is_pub in decl.fields:
            ztype = self._zig_type(field_type)
            if is_float(ztype):
                _fail(
                    "E001",
                    f"packed struct field {name}.{field_name} is {ztype}; "
                    "floats have no fixed bit width in this backend",
                    help_text="use an integer type, or drop #[packed] for a normal struct",
                )
            width = bit_width(ztype)
            if width is None:
                _fail(
                    "E001",
                    f"packed struct field {name}.{field_name} has type {ztype}, "
                    "which has no known bit width",
                    help_text="packed fields must be integers or bool",
                )
            total += width
            rows.append(f"{field_name}: {ztype},")
        public = "pub " if decl.is_pub else ""
        self.emit(f"{public}const {name} = packed struct {{")
        self.indent += 1
        for row in rows:
            self.emit(row)
        self.indent -= 1
        self.emit("};")
        self.emit(f"// {name}: {total} bits, packed\n")

    def _emit_enum(self, decl: EnumDecl) -> None:
        name = decl.name
        payload = any(variant.fields for variant in decl.variants)
        rows: list[str] = []
        if payload:
            for variant in decl.variants:
                if not variant.fields:
                    rows.append(f"{variant.name},")
                elif len(variant.fields) == 1:
                    rows.append(f"{variant.name}: {self._zig_type(variant.fields[0])},")
                else:
                    fields = ", ".join(
                        self._zig_type(field) for field in variant.fields
                    )
                    rows.append(f"{variant.name}: struct {{ {fields} }},")
            kind = "union(enum)"
        else:
            rows = [f"{variant.name}," for variant in decl.variants]
            kind = "enum"
        if decl.generics:
            params = ", ".join(
                f"comptime {generic.name}: type" for generic in decl.generics
            )
            self.emit(f"fn {name}({params}) type {{")
            self.indent += 1
            self.emit(f"return {kind} {{")
            self.indent += 1
            for row in rows:
                self.emit(row)
            self.indent -= 1
            self.emit("};")
            self.indent -= 1
            self.emit("}\n")
            return
        public = "pub " if decl.is_pub else ""
        self.emit(f"{public}const {name} = {kind} {{")
        self.indent += 1
        for row in rows:
            self.emit(row)
        self.indent -= 1
        self.emit("};\n")

    def _emit_type_alias(self, decl: TypeAlias) -> None:
        if decl.generics:
            params = ", ".join(
                f"comptime {generic.name}: type" for generic in decl.generics
            )
            self.emit(f"fn {decl.name}({params}) type {{")
            self.indent += 1
            self.emit(f"return {self._zig_type(decl.ty)};")
            self.indent -= 1
            self.emit("}\n")
            return
        public = "pub " if decl.is_pub else ""
        self.emit(f"{public}const {decl.name} = {self._zig_type(decl.ty)};\n")

    def _emit_const(self, decl: ConstDecl) -> None:
        public = "pub " if decl.is_pub else ""
        self.emit(f"{public}const {decl.name} = {self._emit_expr(decl.value)};\n")

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _emit_extern_fn(self, fn: FnDecl) -> None:
        params = ", ".join(f"{name}: {ztype}" for name, ztype in self._params(fn))
        ret = self._zig_type(fn.ret)
        self.emit(f"extern fn {fn.name}({params}) callconv(.c) {ret};")

    def _emit_fn(self, fn: FnDecl, *, owner: str | None = None) -> None:
        if _has_attr(fn, "extern", "c_extern", "external"):
            return
        self._self_type = owner
        params = self._params(fn)
        ret = self._zig_type(fn.ret)
        self._current_ret = ret
        self._locals = {
            name: ztype for name, ztype in params if ztype not in {"anytype"}
        }
        self._allocator_name = next(
            (name for name, ztype in params if ztype == "std.mem.Allocator"),
            None,
        )
        generic_prefix = ""
        if fn.generics:
            generic_prefix = (
                ", ".join(f"comptime {generic.name}: type" for generic in fn.generics)
                + ", "
            )
        is_par = _has_attr(fn, "parallel") and self._parallel_loop(fn) is not None
        oparams = [(f"__arg_{n}", z) for n, z in params] if is_par else params
        param_text = generic_prefix + ", ".join(f"{n}: {z}" for n, z in oparams)
        public = "pub " if fn.is_pub else ""
        self.emit(f"{public}fn {fn.name}({param_text}) {ret} {{")
        self.indent += 1
        if _has_attr(fn, "unchecked"):
            # Zig's safety checks (overflow, integer bounds) are disabled for
            # this function body only; the caller opts in per function.
            self.emit("@setRuntimeSafety(false);")
        parallel = _has_attr(fn, "parallel") and self._emit_parallel_body(fn, params)
        if parallel:
            pass
        if parallel:
            pass
        if not parallel:
            if _has_attr(fn, "parallel"):
                self._parallel_fallback = True
            if fn.body:
                self._emit_block(fn.body, is_fn=True)
        self.indent -= 1
        self.emit("}\n")
        self.exported_fns.append((fn.name, params, ret, owner is not None))

    def _emit_block(
        self,
        block: Block,
        *,
        is_fn: bool = False,
        as_expr: bool = False,
        label: str | None = None,
    ) -> None:
        saved_locals = dict(self._locals)
        for statement in block.stmts:
            self._emit_stmt(statement)
        if block.tail is not None:
            value = self._emit_expr(block.tail)
            if as_expr and label is not None:
                self.emit(f"break :{label} {value};")
            elif is_fn:
                self.emit(f"return {value};")
            else:
                self.emit(f"_ = {value};")
        elif (
            as_expr
            and label is not None
            and not (
                block.stmts
                and isinstance(getattr(block.stmts[-1], "expr", None), Break)
            )
        ):
            self.emit(f"break :{label} {{}};")
        self._locals = saved_locals

    def _emit_stmt(self, statement: Any) -> None:
        if isinstance(statement, LetStmt):
            self._emit_let(statement)
        elif isinstance(statement, ExprStmt):
            self._emit_expr_stmt(statement.expr)
        elif isinstance(statement, (DeferStmt, ErrDeferStmt)):
            keyword = "defer" if isinstance(statement, DeferStmt) else "errdefer"
            self._uses_defer = True
            if isinstance(statement.body, Block):
                self.emit(f"{keyword} {{")
                self.indent += 1
                self._emit_block(statement.body)
                self.indent -= 1
                self.emit("}")
            else:
                self.emit(f"{keyword} {self._emit_expr(statement.body)};")
        elif isinstance(statement, ItemStmt):
            item = statement.item
            if isinstance(item, ConstDecl):
                self._emit_const(item)
            elif isinstance(item, (FnDecl, StructDecl, EnumDecl)):
                self.emit(f"// nested item `{getattr(item, 'name', '_')}` omitted")
            else:
                self.emit("// unsupported nested item")
        else:
            self.emit("// unsupported statement")

    def _emit_let(self, statement: LetStmt) -> None:
        pattern = statement.pattern
        name = (
            pattern.name
            if isinstance(pattern, (PatIdent, PatBinding))
            else self._next_label("tmp")
        )
        keyword = "var" if statement.mutable else "const"
        annotation = (
            f": {self._zig_type(statement.ty)}" if statement.ty is not None else ""
        )
        value = self._emit_expr(statement.value) if statement.value else "undefined"
        if isinstance(pattern, (PatTuple, PatStruct)):
            tuple_pattern = self._emit_pattern(pattern, binding=True)
            self.emit(f"{keyword} {tuple_pattern}{annotation} = {value};")
        else:
            self.emit(f"{keyword} {name}{annotation} = {value};")
        inferred = (
            self._zig_type(statement.ty)
            if statement.ty is not None
            else self._infer_type(statement.value)
        )
        self._locals[name] = inferred

    def _emit_expr_stmt(self, expr: Any) -> None:
        if isinstance(expr, For):
            self._emit_for(expr)
            return
        if isinstance(expr, While):
            self.emit(f"while ({self._emit_expr(expr.cond)}) {{")
            self.indent += 1
            self._emit_block(expr.body)
            self.indent -= 1
            self.emit("}")
            return
        if isinstance(expr, Loop):
            self.emit("while (true) {")
            self.indent += 1
            self._emit_block(expr.body)
            self.indent -= 1
            self.emit("}")
            return
        if isinstance(expr, Return):
            if expr.value is None:
                self.emit("return;")
            else:
                self.emit(f"return {self._emit_expr(expr.value)};")
            return
        if isinstance(expr, Break):
            if expr.label is None and expr.value is not None and self._block_labels:
                self.emit(
                    f"break :{self._block_labels[-1]} {self._emit_expr(expr.value)};"
                )
                return
            label = f" :{expr.label}" if expr.label else ""
            value = f" {self._emit_expr(expr.value)}" if expr.value else ""
            self.emit(f"break{label}{value};")
            return
        if isinstance(expr, Continue):
            label = f" :{expr.label}" if expr.label else ""
            self.emit(f"continue{label};")
            return
        if isinstance(expr, (IfExpr, IfLet, MatchExpr)):
            rendered = self._emit_expr(expr)
            if isinstance(expr, IfExpr) and expr.else_branch is None:
                self.emit(rendered)
            else:
                self.emit(f"_ = {rendered};")
            return
        self.emit(f"{self._emit_expr(expr)};")

    def _emit_for(self, node: For) -> None:
        pattern = node.pattern
        name = pattern.name if isinstance(pattern, PatIdent) else "item"
        if isinstance(node.iter, Range):
            start = self._emit_expr(node.iter.start) if node.iter.start else "0"
            end = self._emit_expr(node.iter.end) if node.iter.end else "0"
            bound = "<=" if node.iter.inclusive else "<"
            self._locals[name] = "usize"
            self.emit(f"var {name}: usize = {start};")
            self.emit(f"while ({name} {bound} {end}) : ({name} += 1) {{")
            self.indent += 1
            self._emit_block(node.body)
            self.indent -= 1
            self.emit("}")
            return
        iterable = self._emit_expr(node.iter)
        capture = name if isinstance(pattern, PatIdent) else self._emit_pattern(pattern)
        self.emit(f"for ({iterable}) |{capture}| {{")
        self.indent += 1
        self._emit_block(node.body)
        self.indent -= 1
        self.emit("}")

    # ------------------------------------------------------------------
    # Parallel range lowering
    # ------------------------------------------------------------------

    def _parallel_loop(self, fn: FnDecl) -> tuple[For, Range] | None:
        if fn.body is None or fn.body.tail is not None:
            return None
        loops = [
            statement.expr
            for statement in fn.body.stmts
            if isinstance(statement, ExprStmt)
            and isinstance(statement.expr, For)
            and isinstance(statement.expr.iter, Range)
        ]
        if len(loops) != 1 or len(fn.body.stmts) != 1:
            return None
        loop = loops[0]
        if isinstance(loop.pattern, (PatTuple, PatStruct, PatSlice)):
            return None
        if self._contains_flow(loop.body):
            return None
        return loop, loop.iter

    def _contains_flow(self, block: Block) -> bool:
        for statement in block.stmts:
            if isinstance(statement, ExprStmt) and isinstance(
                statement.expr, (Return, Break, Continue)
            ):
                return True
            if isinstance(statement, ExprStmt) and isinstance(
                statement.expr, (For, While, WhileLet, Loop)
            ):
                return True
        return False

    def _emit_parallel_body(self, fn: FnDecl, params: list[tuple[str, str]]) -> bool:
        found = self._parallel_loop(fn)
        if found is None:
            return False
        loop, range_expr = found
        name = loop.pattern.name if isinstance(loop.pattern, PatIdent) else "index"
        start = self._emit_expr(range_expr.start) if range_expr.start else "0"
        end = self._emit_expr(range_expr.end) if range_expr.end else "0"
        oparams = [(f"__arg_{n}", z) for n, z in params]
        for (pn, _), (on, _) in zip(params, oparams):
            start = start.replace(pn, on)
            end = end.replace(pn, on)
        if range_expr.inclusive:
            end = f"({end}) + 1"
        oparams = [(f"__arg_{n}", z) for n, z in params]
        initializers = ", ".join(
            f".{p} = {o}" for (p, _), (o, _) in zip(params, oparams)
        )
        self.emit("const Ctx = struct {")
        self.indent += 1
        for param, ztype in params:
            self.emit(f"{param}: {ztype},")
        self.emit("start: usize,")
        self.emit("end: usize,")
        self.emit("next: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),")
        self.indent -= 1
        self.emit("};")
        self.emit(
            f"var ctx = Ctx{{ {initializers}, .start = {start}, .end = {end}, "
            f".next = std.atomic.Value(usize).init({start}) }};"
        )
        self.emit("const Worker = struct {")
        self.indent += 1
        self.emit("fn work(c: *Ctx) void {")
        self.indent += 1
        for param, _ in params:
            self.emit(f"const {param} = c.{param}; _ = &{param};")
        self.emit("while (true) {")
        self.indent += 1
        self.emit("const index = c.next.fetchAdd(1, .monotonic);")
        self.emit("if (index >= c.end) break;")
        self.emit(f"const {name} = index;")
        self._emit_block(loop.body)
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("}")
        self.indent -= 1
        self.emit("};")
        self.emit("const count = @min(std.Thread.getCpuCount() catch 1, 64);")
        self.emit("var threads: [64]std.Thread = undefined;")
        self.emit("var spawned: usize = 0;")
        self.emit("while (spawned < count) : (spawned += 1) {")
        self.indent += 1
        self.emit(
            "threads[spawned] = std.Thread.spawn(.{}, Worker.work, .{&ctx}) "
            "catch break;"
        )
        self.indent -= 1
        self.emit("}")
        self.emit("for (threads[0..spawned]) |thread| thread.join();")
        self.emit("Worker.work(&ctx);")
        return True

    # ------------------------------------------------------------------
    # Patterns and expressions
    # ------------------------------------------------------------------

    def _emit_pattern(self, pattern: Any, *, binding: bool = False) -> str:
        if isinstance(pattern, PatWildcard):
            return "_"
        if isinstance(pattern, PatIdent):
            return pattern.name if binding else "else"
        if isinstance(pattern, PatLiteral):
            return self._emit_expr(pattern.value)
        if isinstance(pattern, PatPath):
            return self._emit_path_text(pattern.path, switch_pattern=True)
        if isinstance(pattern, PatEnum):
            return self._emit_path_text(pattern.path, switch_pattern=True)
        if isinstance(pattern, PatTuple):
            items = ", ".join(self._emit_pattern(item) for item in pattern.elems)
            return f".{{ {items} }}"
        if isinstance(pattern, PatStruct):
            fields = ", ".join(
                f".{name} = {self._emit_pattern(value)}"
                for name, value in pattern.fields
            )
            if pattern.rest:
                fields = f"{fields}, .." if fields else ".."
            return f".{{ {fields} }}"
        if isinstance(pattern, PatOr):
            return ", ".join(self._emit_pattern(alt) for alt in pattern.alts)
        if isinstance(pattern, PatRange):
            start = self._emit_expr(pattern.start) if pattern.start else ""
            end = self._emit_expr(pattern.end) if pattern.end else ""
            operator = "..." if pattern.inclusive else ".."
            return f"{start}{operator}{end}"
        if isinstance(pattern, PatBinding):
            return f"{pattern.name} @ {self._emit_pattern(pattern.inner)}"
        if isinstance(pattern, PatRef):
            return self._emit_pattern(pattern.inner)
        if isinstance(pattern, PatRest):
            return ".."
        if isinstance(pattern, PatSlice):
            return (
                ".{ "
                + ", ".join(self._emit_pattern(item) for item in pattern.elems)
                + " }"
            )
        if isinstance(pattern, PatConst):
            return self._emit_path_text(pattern.path)
        return "_"

    def _emit_path_text(self, path: str, *, switch_pattern: bool = False) -> str:
        parts = path.split("::")
        if parts[-1] == "None":
            return "null"
        if (
            switch_pattern
            and len(parts) >= 2
            and parts[0] in (self._struct_names | self._enum_names)
        ):
            return "." + ".".join(parts[1:])
        return ".".join(parts)

    def _emit_capture(self, pattern: Any) -> str:
        if isinstance(pattern, PatEnum) and pattern.args:
            inner = ", ".join(
                self._emit_pattern(arg, binding=True) for arg in pattern.args
            )
            return f" |{inner}|"
        if isinstance(pattern, PatIdent):
            return f" |{pattern.name}|"
        return ""

    def _emit_expr(self, expr: Any) -> str:
        if isinstance(expr, Paren):
            return f"({self._emit_expr(expr.inner)})"
        if isinstance(expr, Cast):
            return self._emit_cast(expr)
        if isinstance(expr, LitInt):
            return str(expr.value)
        if isinstance(expr, LitFloat):
            return str(expr.value)
        if isinstance(expr, LitStr):
            return _quote(expr.value)
        if isinstance(expr, LitChar):
            return f"'{expr.value}'"
        if isinstance(expr, LitBool):
            return "true" if expr.value else "false"
        if isinstance(expr, Ident):
            return "null" if expr.name == "None" else expr.name
        if isinstance(expr, AstPath):
            return self._emit_path_text("::".join(expr.segments))
        if isinstance(expr, Index):
            return f"{self._emit_expr(expr.receiver)}[{self._emit_expr(expr.index)}]"
        if isinstance(expr, Field):
            return f"{self._emit_expr(expr.receiver)}.{expr.name}"
        if isinstance(expr, Assign):
            return f"{self._emit_expr(expr.target)} = {self._emit_expr(expr.value)}"
        if isinstance(expr, CompoundAssign):
            target = self._emit_expr(expr.target)
            value = self._emit_expr(expr.value)
            if expr.op in {"+|", "-|", "*|"}:
                helper = {
                    "+|": "_zig_sat_add",
                    "-|": "_zig_sat_sub",
                    "*|": "_zig_sat_mul",
                }[expr.op]
                return f"{target} = {helper}({target}, {value})"
            return f"{target} {expr.op} {value}"
        if isinstance(expr, Unary):
            operator = "&" if expr.op in {"&", "&mut"} else expr.op
            return f"{operator}{self._emit_expr(expr.operand)}"
        if isinstance(expr, Binary):
            return self._emit_binary(expr)
        if isinstance(expr, Call):
            return self._emit_call(expr)
        if isinstance(expr, MethodCall):
            return self._emit_method(expr)
        if isinstance(expr, TupleExpr):
            items = ", ".join(self._emit_expr(item) for item in expr.elems)
            return f".{{ {items} }}"
        if isinstance(expr, ArrayExpr):
            items = ", ".join(self._emit_expr(item) for item in expr.elems)
            return f".{{ {items} }}"
        if isinstance(expr, RepeatArray):
            return f"[_]{self._emit_expr(expr.elem)} ** {self._emit_expr(expr.count)}"
        if isinstance(expr, StructLiteral):
            name = self._emit_path_text(expr.path)
            if expr.generic_args:
                args = ", ".join(self._zig_type(arg) for arg in expr.generic_args)
                name = f"{name}({args})"
            fields = ", ".join(
                f".{key} = {self._emit_expr(value)}" for key, value in expr.fields
            )
            if expr.rest is not None:
                fields = f"{fields}, ..{self._emit_expr(expr.rest)}"
            return f"{name}{{ {fields} }}"
        if isinstance(expr, IfExpr):
            return self._emit_if(expr)
        if isinstance(expr, IfLet):
            return self._emit_if_let(expr)
        if isinstance(expr, MatchExpr):
            return self._emit_match(expr)
        if isinstance(expr, Block):
            return self._render_block_expr(expr)
        if isinstance(expr, Range):
            start = self._emit_expr(expr.start) if expr.start else ""
            end = self._emit_expr(expr.end) if expr.end else ""
            return f"{start}{'..=' if expr.inclusive else '..'}{end}"
        if isinstance(expr, Return):
            return (
                "return;"
                if expr.value is None
                else f"return {self._emit_expr(expr.value)}"
            )
        if isinstance(expr, Break):
            if expr.label is None and expr.value is not None and self._block_labels:
                statement = (
                    f"break :{self._block_labels[-1]} {self._emit_expr(expr.value)};"
                )
                self.emit(statement)
                # Return the text as well as emitting it: this method is typed
                # to return the rendered expression, and a bare `return` would
                # hand callers `None` to interpolate into a string.
                return statement
            return "break"
        if isinstance(expr, Continue):
            return "continue"
        if isinstance(expr, Try):
            return f"try {self._emit_expr(expr.expr)}"
        if isinstance(expr, Await):
            return self._emit_expr(expr.expr)
        if isinstance(expr, OptionSome):
            return self._emit_expr(expr.value)
        if isinstance(expr, OptionNone):
            return "null"
        if isinstance(expr, ResultOk):
            return self._emit_expr(expr.value)
        if isinstance(expr, ResultErr):
            return _error_expr(expr.value, None)
        if isinstance(expr, MacroCall):
            return self._emit_macro(expr)
        if isinstance(expr, Closure):
            return '@compileError("closures are not supported by the Zig backend")'
        return "0"

    def _emit_cast(self, expr: Cast) -> str:
        target = self._zig_type(expr.ty)
        inner = self._emit_expr(expr.expr)
        if target in {"f32", "f64"}:
            if isinstance(expr.expr, LitFloat):
                return f"@as({target}, {inner})"
            return f"@as({target}, @floatFromInt({inner}))"
        if (
            target in {"i64", "i32", "u64", "u32", "u16", "u8"}
            and isinstance(expr.expr, LitFloat)
            or self._infer_type(expr.expr) in {"f32", "f64"}
        ):
            return f"@as({target}, @intFromFloat({inner}))"
        return f"@as({target}, @intCast({inner}))"

    def _emit_binary(self, expr: Binary) -> str:
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        if expr.op in {"+|", "-|", "*|"}:
            helper = {
                "+|": "_zig_sat_add",
                "-|": "_zig_sat_sub",
                "*|": "_zig_sat_mul",
            }[expr.op]
            return f"{helper}({left}, {right})"
        integer = self._infer_type(expr.left)
        is_integer = integer in {"isize", "usize"} or (
            integer[:1] in {"i", "u"} and integer[1:].isdigit()
        )
        if expr.op in {"/", "%"} and is_integer:
            helper = "_zig_div" if expr.op == "/" else "_zig_rem"
            return (
                f"{helper}({left}, {right}, "
                f"{getattr(expr, 'line', 0) or 0}, {getattr(expr, 'col', 0) or 0})"
            )
        operator = {"&&": "and", "||": "or"}.get(expr.op, expr.op)
        return f"({left} {operator} {right})"

    def _coerce_call_arg(self, arg: Any, expected_ztype: str | None) -> str:
        rendered = self._emit_expr(arg)
        if not expected_ztype or expected_ztype == "anytype":
            return rendered
        inferred = self._infer_type(arg)
        if inferred == expected_ztype:
            return rendered
        if (
            (_INT_RE.fullmatch(expected_ztype) or expected_ztype in {"usize", "isize"})
            and (_INT_RE.fullmatch(inferred) or inferred in {"usize", "isize"})
            and not isinstance(arg, LitInt)
        ):
            return f"@as({expected_ztype}, @intCast({rendered}))"
        return rendered

    def _emit_call(self, expr: Call) -> str:
        func = expr.func
        fn_name = ""
        if isinstance(func, Ident):
            fn_name = func.name
        elif isinstance(func, AstPath):
            fn_name = func.segments[-1]
        expected_params = self._fn_params.get(fn_name, [])
        args = ", ".join(
            self._coerce_call_arg(
                arg, expected_params[idx] if idx < len(expected_params) else None
            )
            for idx, arg in enumerate(expr.args)
        )
        if isinstance(func, AstPath):
            segments = func.segments
            name = segments[-1]
            if name in {"Some", "Ok"} and len(expr.args) == 1:
                return self._emit_expr(expr.args[0])
            if name in {"None"}:
                return "null"
            if name == "Err" and len(expr.args) == 1:
                return _error_expr(expr.args[0], None)
            if segments[:1] == ["Simd"]:
                return self._emit_simd(name, expr.args)
            if segments[:1] == ["Vec"] and name in {"new", "with_capacity"}:
                return "&.{}"
            if segments[:1] == ["String"] and name == "from" and expr.args:
                return self._emit_expr(expr.args[0])
            if name in {"print", "println", "eprintln"}:
                return f"std.debug.print({args})"
            if name == "panic":
                return self._emit_fail(expr, args)
            base = ".".join(segments)
            if func.generic_args:
                type_args = ", ".join(self._zig_type(arg) for arg in func.generic_args)
                if segments[-1] in self._generic_fns:
                    return f"{base}({type_args}{', ' if args else ''}{args})"
            return f"{base}({args})"
        if isinstance(func, Ident):
            name = func.name
            if name in {"Some", "Ok"} and len(expr.args) == 1:
                return self._emit_expr(expr.args[0])
            if name == "None":
                return "null"
            if name == "Err" and len(expr.args) == 1:
                return _error_expr(expr.args[0], None)
            if name in {"print", "println", "eprintln"}:
                return f"std.debug.print({args})"
            if name == "panic":
                return self._emit_fail(expr, args)
            if name in {"assert"}:
                return self._emit_assert(expr)
            return f"{name}({args})"
        if isinstance(func, Field):
            receiver = self._emit_expr(func.receiver)
            return f"{receiver}.{func.name}({args})"
        return f"{self._emit_expr(func)}({args})"

    def _emit_simd(self, method: str, args: list[Any]) -> str:
        if method == "splat" and len(args) == 1:
            return f"@splat({self._emit_expr(args[0])})"
        if method == "load" and len(args) == 3:
            return (
                f"{self._emit_expr(args[0])}[{self._emit_expr(args[1])}..]"
                f"[0..{self._emit_expr(args[2])}].*"
            )
        if method == "store" and len(args) == 4:
            return (
                f"{self._emit_expr(args[0])}[{self._emit_expr(args[1])}..]"
                f"[0..{self._emit_expr(args[2])}].* = {self._emit_expr(args[3])}"
            )
        if method in {
            "reduce_add",
            "reduce_mul",
            "reduce_min",
            "reduce_max",
            "reduce_and",
            "reduce_or",
            "reduce_xor",
        }:
            op = {
                "reduce_add": ".Add",
                "reduce_mul": ".Mul",
                "reduce_min": ".Min",
                "reduce_max": ".Max",
                "reduce_and": ".And",
                "reduce_or": ".Or",
                "reduce_xor": ".Xor",
            }[method]
            return f"@reduce({op}, {self._emit_expr(args[0])})"
        return f"Simd.{method}()"

    def _emit_method(self, expr: MethodCall) -> str:
        receiver = self._emit_expr(expr.receiver)
        args = ", ".join(self._emit_expr(arg) for arg in expr.args)
        receiver_type = self._infer_type(expr.receiver)
        method = expr.method
        owner = receiver_type.lstrip("*").strip()
        method_fn = self._method_names.get((owner, method))
        if method_fn is not None:
            receiver_arg = (
                f"&{receiver}"
                if self._method_self_pointer.get((owner, method))
                else receiver
            )
            return f"{method_fn}({receiver_arg}{', ' if args else ''}{args})"
        if method == "len":
            return f"({receiver}).len"
        if method in {"to_string", "clone", "as_str", "as_bytes"}:
            return receiver
        if method == "unwrap":
            if "!" in receiver_type and not receiver_type.startswith("?"):
                return f"({receiver}) catch unreachable"
            return f"({receiver}).?"
        if method == "unwrap_or":
            if receiver_type.startswith("?"):
                return f"(({receiver}) orelse ({args}))"
            return f"(({receiver}) catch ({args}))"
        if method == "expect":
            if receiver_type.startswith("?"):
                return f"(({receiver}) orelse unreachable)"
            return f"(({receiver}) catch unreachable)"
        if method == "is_some":
            return f"(({receiver}) != null)"
        if method == "is_none":
            return f"(({receiver}) == null)"
        if method == "is_ok":
            return f"if ({receiver}) |_| true else |_| false"
        if method == "is_err":
            return f"!(if ({receiver}) |_| true else |_| false)"
        if method == "split":
            sep = expr.args[0] if expr.args else None
            helper = (
                "splitScalar"
                if isinstance(sep, LitStr) and len(sep.value) == 1
                else "splitSequence"
            )
            return f"std.mem.{helper}(u8, {receiver}, {args})"
        if method == "join" and self._allocator_name:
            return f"std.mem.join({self._allocator_name}, {args}, {receiver})"
        if method == "replace" and self._allocator_name:
            return (
                f"std.mem.replaceOwned(u8, {self._allocator_name}, {receiver}, {args})"
            )
        if method == "alloc" and expr.generic_args:
            types = ", ".join(self._zig_type(arg) for arg in expr.generic_args)
            return f"{receiver}.alloc({types}{', ' if args else ''}{args})"
        if method == "free":
            return f"{receiver}.free({args})"
        if method in {"push", "append"}:
            allocator = self._allocator_name or "allocator"
            return f"{receiver}.append({allocator}, {args})"
        if method == "pop":
            return f"{receiver}.pop()"
        if expr.generic_args:
            types = ", ".join(self._zig_type(arg) for arg in expr.generic_args)
            return f"{receiver}.{method}({types}{', ' if args else ''}{args})"
        return f"{receiver}.{method}({args})"

    def _emit_macro(self, expr: MacroCall) -> str:
        if isinstance(expr.args, str):
            return f"@{expr.path}({expr.args})"
        if shape_for(expr.path) is not None:
            return self._emit_value_intrinsic(expr)
        if expr.path == "bitcast":
            return self._emit_bitcast(expr)
        if expr.path in TYPE_INTRINSICS:
            return self._emit_type_intrinsic(expr)
        if expr.path in OVERFLOW_INTRINSICS or expr.path in WRAPPING_INTRINSICS:
            return self._emit_overflow_intrinsic(expr)
        if expr.path in ROTATE_INTRINSICS:
            return self._emit_rotate_intrinsic(expr)
        if expr.path in CONVERSION_INTRINSICS:
            return self._emit_conversion_intrinsic(expr)
        if expr.path in MEMORY_INTRINSICS:
            return self._emit_memory_intrinsic(expr)
        args = ", ".join(self._emit_expr(arg) for arg in expr.args)
        if expr.path == "panic":
            return self._emit_fail(expr, args)
        if expr.path == "assert":
            return self._emit_assert(expr)
        if expr.path in {"println", "print"}:
            return f"std.debug.print({args})"
        if expr.path == "vec":
            return f".{{ {args} }}"
        return f"@{expr.path}({args})"

    def _emit_fail(self, expr: MacroCall, args: str) -> str:
        """Lower ``panic!`` onto the guarded failure helper."""
        message = args or '"panic called"'
        return self._fail_call(message, expr)

    def _fail_call(self, message: str, expr: Any) -> str:
        line = getattr(expr, "line", 0) or 0
        col = getattr(expr, "col", 0) or 0
        return f'_zig_fail("E002", {message}, {line}, {col})'

    def _emit_assert(self, expr: MacroCall) -> str:
        """Lower ``assert!`` onto the guarded helper so it reports a position."""
        if not expr.args or isinstance(expr.args, str):
            return '_zig_assert(false, "assertion failed", 0, 0)'
        condition = self._emit_expr(expr.args[0])
        message = '"assertion failed"'
        if len(expr.args) >= 2:
            message = self._emit_expr(expr.args[1])
        line = getattr(expr, "line", 0) or 0
        col = getattr(expr, "col", 0) or 0
        return f"_zig_assert({condition}, {message}, {line}, {col})"

    def _emit_value_intrinsic(self, expr: MacroCall) -> str:
        """Lower popcount!/clz!/ctz!/bswap!/bitreverse! onto Zig builtins."""
        if len(expr.args) != 1:
            _fail(
                "E003",
                f"{expr.path}! takes exactly one operand, got {len(expr.args)}",
            )
        operand = expr.args[0]
        operand_type = self._infer_type(operand)
        try:
            shape = require_value_operand(expr.path, operand_type)
        except IntrinsicError as error:
            _fail("E001", str(error))
        pinned = f"@as({operand_type}, {self._emit_expr(operand)})"
        builtin = f"{shape.builtin}({pinned})"
        if expr.path in {"bswap", "bitreverse"}:
            # These preserve the operand width, so the caller's return type
            # decides the widening; an explicit i64 cast could trap on u64.
            return builtin
        # The counting builtins return a width-dependent integer (u7 for
        # @popCount of a u64); widen to the documented i64 result.
        return f"@as(i64, @intCast({builtin}))"

    def _type_argument(self, arg: Any, name: str) -> str:
        if isinstance(arg, Ident):
            return arg.name
        if isinstance(arg, AstPath):
            return arg.segments[-1]
        if isinstance(arg, TypeName):
            return arg.name
        _fail(
            "E003",
            f"{name}! second argument must be a type name, not an expression",
            help_text="pass a declared type, for example bitsizeof!(Flags)",
        )

    def _emit_type_intrinsic(self, expr: MacroCall) -> str:
        if len(expr.args) != 1:
            _fail("E003", f"{expr.path}! takes exactly one type, got {len(expr.args)}")
        type_name = self._type_argument(expr.args[0], expr.path)
        if type_name in self._struct_names or type_name in self._enum_names:
            builtin = TYPE_INTRINSICS[expr.path]
        else:
            try:
                builtin = require_type_operand(expr.path, type_name)
            except IntrinsicError as error:
                _fail("E001", str(error))
        return f"@as(i64, {builtin}({type_name}))"

    def _emit_bitcast(self, expr: MacroCall) -> str:
        """Lower bitcast!(value, T) to the generic @bitCast helper."""
        if len(expr.args) != 2:
            _fail(
                "E003",
                f"bitcast! takes a value and a type, got {len(expr.args)} argument(s)",
            )
        value_type = self._infer_type(expr.args[0])
        target = self._type_argument(expr.args[1], "bitcast")
        source_width = bit_width(value_type)
        target_width = bit_width(target)
        if source_width is None or target_width is None:
            _fail(
                "E001",
                f"bitcast! needs types with known bit widths, got {value_type} -> {target}",
                help_text="use fixed-width integers or floats",
            )
        if source_width != target_width:
            _fail(
                "E001",
                f"bitcast! cannot reinterpret {value_type} ({source_width} bits) "
                f"as {target} ({target_width} bits)",
                help_text="both sides of a bit cast must have the same bit count",
            )
        return f"_zig_bitcast({target}, {self._emit_expr(expr.args[0])})"

    def _emit_overflow_intrinsic(self, expr: MacroCall) -> str:
        """Lower add_with_overflow!/mul_with_overflow!/wrapping_add!/wrapping_mul!.

        `@addWithOverflow` returns a tuple `.{ value, flag }`. Struct
        construction is not part of this backend's subset, so the two entry
        points index the tuple directly instead of returning a struct.
        """
        name = expr.path
        if len(expr.args) != 2:
            _fail("E003", f"{name}! takes two operands, got {len(expr.args)}")
        left_type = self._infer_type(expr.args[0])
        right_type = self._infer_type(expr.args[1])
        try:
            require_matching_operands(name, left_type, right_type)
            require_arithmetic_operand(name, left_type)
        except IntrinsicError as error:
            _fail("E001", str(error))
        builtin = OVERFLOW_INTRINSICS.get(name) or WRAPPING_INTRINSICS[name]
        operands = (
            f"@as({left_type}, {self._emit_expr(expr.args[0])}), "
            f"@as({left_type}, {self._emit_expr(expr.args[1])})"
        )
        if name in WRAPPING_INTRINSICS:
            # The wrapped value keeps the operand width.
            return f"{builtin}({operands})[0]"
        # The overflow flag is a u1; widen it to the documented i64 result.
        return f"@as(i64, @intCast({builtin}({operands})[1]))"

    def _emit_rotate_intrinsic(self, expr: MacroCall) -> str:
        """Lower rotl!/rotr! onto `std.math.rotl`/`rotr`, keeping the width."""
        name = expr.path
        if len(expr.args) != 2:
            _fail("E003", f"{name}! takes a value and a count, got {len(expr.args)}")
        value_type = self._infer_type(expr.args[0])
        count_type = self._infer_type(expr.args[1])
        try:
            require_arithmetic_operand(name, value_type)
            require_arithmetic_operand(f"{name} count", count_type)
        except IntrinsicError as error:
            _fail("E001", str(error))
        function = f"std.math.{ROTATE_INTRINSICS[name]}"
        value = f"@as({value_type}, {self._emit_expr(expr.args[0])})"
        # The count must be the operand's own integer type; a negative or
        # over-wide count traps on @intCast rather than rotating silently.
        count = f"@as({value_type}, @intCast({self._emit_expr(expr.args[1])}))"
        return f"{function}({value_type}, {value}, {count})"

    def _emit_conversion_intrinsic(self, expr: MacroCall) -> str:
        """Lower int_from_float! (-> i64) and float_from_int! (-> f64)."""
        name = expr.path
        if len(expr.args) != 1:
            _fail("E003", f"{name}! takes exactly one operand, got {len(expr.args)}")
        operand_type = self._infer_type(expr.args[0])
        try:
            if name == "int_from_float":
                require_float_operand(name, operand_type)
            else:
                require_arithmetic_operand(name, operand_type)
        except IntrinsicError as error:
            _fail("E001", str(error))
        builtin = CONVERSION_INTRINSICS[name]
        result = "i64" if name == "int_from_float" else "f64"
        return f"@as({result}, {builtin}({self._emit_expr(expr.args[0])}))"

    def _emit_memory_intrinsic(self, expr: MacroCall) -> str:
        """Lower memcpy_bytes!/memset_bytes! onto `@memcpy`/`@memset`."""
        name = expr.path
        if len(expr.args) != 3:
            _fail(
                "E003",
                f"{name}! takes a destination, a "
                f"{'source' if name == 'memcpy_bytes' else 'byte value'} "
                f"and a count, got {len(expr.args)}",
            )
        destination = self._emit_expr(expr.args[0])
        destination_type = self._infer_type(expr.args[0])
        if "*" not in destination_type and "[" not in destination_type:
            _fail(
                "E001",
                f"{name}! needs a pointer or slice destination, "
                f"got {destination_type!r}",
                help_text="pass a &mut [u8], a *u8, or another byte buffer",
            )
        count = self._emit_expr(expr.args[2])
        destination_view = f"({destination})[0..{count}]"
        if name == "memcpy_bytes":
            source = self._emit_expr(expr.args[1])
            source_type = self._infer_type(expr.args[1])
            if "*" not in source_type and "[" not in source_type:
                _fail(
                    "E001",
                    f"{name}! needs a pointer or slice source, got {source_type!r}",
                )
            return f"@memcpy({destination_view}, ({source})[0..{count}])"
        value = self._emit_expr(expr.args[1])
        return f"@memset({destination_view}, @as(u8, @intCast({value})))"

    # ------------------------------------------------------------------
    # Control-flow expressions
    # ------------------------------------------------------------------

    def _render_block_expr(self, block: Block) -> str:
        if block.tail is not None and not block.stmts:
            return self._emit_expr(block.tail)
        label = self._next_label()
        saved = self._capture()
        self._block_labels.append(label)
        try:
            self._emit_block(block, as_expr=True, label=label)
        finally:
            self._block_labels.pop()
        body = self._restore(saved)
        text = "\n".join(f"    {line}" for line in body)
        return f"{label}: {{\n{text}\n}}"

    def _emit_if(self, expr: IfExpr) -> str:
        condition = self._emit_expr(expr.cond)
        if expr.else_branch is None:
            saved = self._capture()
            self._emit_block(expr.then_block, is_fn=False, as_expr=False)
            body = self._restore(saved)
            text = chr(10).join(f"    {line}" for line in body)
            return f"if ({condition}) {{{chr(10)}{text}{chr(10)}}}"
        then_value = self._render_block_expr(expr.then_block)
        if isinstance(expr.else_branch, Block):
            else_value = self._render_block_expr(expr.else_branch)
        elif isinstance(expr.else_branch, (IfExpr, IfLet)):
            else_value = self._emit_expr(expr.else_branch)
        else:
            else_value = self._emit_expr(expr.else_branch)
        return f"if ({condition}) {then_value} else {else_value}"

    def _emit_if_let(self, expr: IfLet) -> str:
        value = self._emit_expr(expr.value)
        pattern = expr.pattern
        then_value = self._render_block_expr(expr.then_block)
        else_value = (
            self._render_block_expr(expr.else_branch)
            if isinstance(expr.else_branch, Block)
            else self._emit_expr(expr.else_branch)
            if expr.else_branch is not None
            else "{}"
        )
        if isinstance(pattern, PatEnum) and pattern.path.endswith(("Ok", "Some")):
            capture = (
                self._emit_pattern(pattern.args[0], binding=True)
                if pattern.args
                else "_"
            )
            if pattern.path.endswith("Some"):
                return f"if ({value}) |{capture}| {then_value} else {else_value}"
            return f"if ({value}) |{capture}| {then_value} else |_| {else_value}"
        if isinstance(pattern, PatIdent):
            return f"if ({value}) |{pattern.name}| {then_value} else {else_value}"
        if isinstance(pattern, PatPath) and pattern.path.endswith("None"):
            return f"if ({value} == null) {then_value} else {else_value}"
        return f"if ({value}) {then_value} else {else_value}"

    def _emit_match(self, expr: MatchExpr) -> str:
        subject = self._emit_expr(expr.scrutinee)
        subject_type = self._infer_type(expr.scrutinee)
        if subject_type.startswith("?") or "!" in subject_type:
            return self._emit_option_match(subject, subject_type, expr)
        lines = [f"switch ({subject}) {{"]
        has_else = False
        for arm in expr.arms:
            pattern = self._emit_pattern(arm.pattern)
            capture = self._emit_capture(arm.pattern)
            if pattern in {"_", "else"}:
                pattern = "else"
                has_else = True
            guard = f" if {self._emit_expr(arm.guard)}" if arm.guard else ""
            body = (
                self._render_block_expr(arm.body)
                if isinstance(arm.body, Block)
                else self._emit_expr(arm.body)
            )
            lines.append(f"    {pattern}{capture}{guard} => {body},")
        if not has_else:
            lines.append("    else => unreachable,")
        lines.append("}")
        return "\n".join(lines)

    def _emit_option_match(
        self, subject: str, subject_type: str, expr: MatchExpr
    ) -> str:
        some_arm = None
        none_arm = None
        for arm in expr.arms:
            pattern = arm.pattern
            text = pattern.path if isinstance(pattern, (PatEnum, PatPath)) else ""
            if text.endswith("None"):
                none_arm = arm
            elif text.endswith(("Some", "Ok")):
                some_arm = arm
            elif isinstance(pattern, PatWildcard):
                none_arm = none_arm or arm
        if some_arm is None:
            return "unreachable"
        capture = "_"
        pattern = some_arm.pattern
        if isinstance(pattern, PatEnum) and pattern.args:
            capture = self._emit_pattern(pattern.args[0], binding=True)
        then_value = (
            self._render_block_expr(some_arm.body)
            if isinstance(some_arm.body, Block)
            else self._emit_expr(some_arm.body)
        )
        else_value = (
            self._render_block_expr(none_arm.body)
            if none_arm is not None and isinstance(none_arm.body, Block)
            else self._emit_expr(none_arm.body)
            if none_arm is not None
            else "unreachable"
        )
        if subject_type.startswith("?"):
            return f"if ({subject}) |{capture}| {then_value} else {else_value}"
        return f"if ({subject}) |{capture}| {then_value} else |_| {else_value}"

    # ------------------------------------------------------------------
    # C ABI shims
    # ------------------------------------------------------------------

    def _emit_cabi_shims(self) -> None:
        for fname, params, ret, skip in self.exported_fns:
            if skip:
                continue
            abi_params: list[tuple[str, str, str]] = []
            supported = True
            for name, ztype in params:
                cabi = _cabi_type(ztype)
                if cabi is None:
                    supported = False
                    break
                abi_params.append((name, ztype, cabi))
            abi_ret = _cabi_type(ret)
            if not supported or abi_ret is None:
                continue
            for ztype in [ztype for _, ztype, _ in abi_params] + [ret]:
                if _vector_shape(ztype) is not None:
                    self._vector_abi_types.setdefault(ztype, _vector_struct_name(ztype))
            c_params = ", ".join(f"{name}: {cabi}" for name, _, cabi in abi_params)
            self.emit(f"export fn __cabi_{fname}({c_params}) callconv(.c) {abi_ret} {{")
            self.indent += 1
            args = [_cabi_arg(name, ztype) for name, ztype, _ in abi_params]
            if ret == "void":
                self.emit(f"{fname}({', '.join(args)});")
            else:
                self.emit(f"const res = {fname}({', '.join(args)});")
                self.emit(_cabi_return(ret, abi_ret))
            self.indent -= 1
            self.emit("}\n")


class ZigBackend(BackendBase):
    name = "zig"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        trusted: bool = True,
        scope: str | None = None,
        optimize: str = "ReleaseFast",
        incremental: bool = False,
        use_build_graph: bool = False,
    ) -> None:
        super().__init__(trusted=trusted)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "polyglot-zig")
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.scope = scope or "default"
        self.optimize = normalize_optimize(optimize)
        self.incremental = incremental
        self.use_build_graph = use_build_graph
        self.last_build: Any = None
        self.last_codegen: ZigCodegen | None = None

    def _ensure_library(
        self,
        zig_src: str,
        module_name: str,
        *,
        optimize: str | None = None,
        incremental: bool | None = None,
        scope: str | None = None,
        use_build_graph: bool | None = None,
        link_libs: list[str] | None = None,
    ) -> Path:
        """Build ``zig_src`` once and return the dynamic library path."""
        mode = normalize_optimize(optimize or self.optimize)
        want_incremental = self.incremental if incremental is None else incremental
        graph = self.use_build_graph if use_build_graph is None else use_build_graph
        build_scope = scope or self.scope
        try:
            if graph:
                result = build_dynamic_library(
                    zig_src,
                    module_name,
                    cache_root=self.cache_dir,
                    scope=build_scope,
                    optimize=mode,
                    incremental=want_incremental,
                    link_libs=tuple(link_libs or ()),
                )
            else:
                result = build_dynamic_library_direct(
                    zig_src,
                    module_name,
                    cache_root=self.cache_dir,
                    scope=build_scope,
                    optimize=mode,
                )
        except ZigBuildError as error:
            raise ZigCompileError(str(error)) from error
        self.last_build = result
        return result.artifact

    def build(
        self,
        program: Program,
        module_name: str = "polyglot_zig",
        *,
        incremental: bool = False,
        cache_scope: str | None = None,
        optimize: str | None = None,
        use_build_graph: bool | None = None,
        **_: Any,
    ) -> Path:
        self._check_trust()
        codegen = ZigCodegen(module_name=module_name)
        zig_src = codegen.generate(program)
        self.last_codegen = codegen
        return self._ensure_library(
            zig_src,
            module_name,
            optimize=optimize,
            incremental=incremental,
            scope=cache_scope,
            use_build_graph=use_build_graph,
            link_libs=codegen._link_libs,
        )

    def compile(
        self, program: Program, module_name: str = "polyglot_zig"
    ) -> dict[str, Any]:
        self._check_trust()
        codegen = ZigCodegen(module_name=module_name)
        zig_src = codegen.generate(program)
        self.last_codegen = codegen
        dylib = self._ensure_library(
            zig_src,
            module_name,
            link_libs=codegen._link_libs,
        )
        lib = ctypes.CDLL(str(dylib))
        namespace: dict[str, Any] = {"_zig_lib": lib}
        for name, params, ret, skip in codegen.exported_fns:
            if skip:
                continue
            if any(_cabi_type(ztype) is None for _, ztype in params):
                continue
            if _cabi_type(ret) is None:
                continue
            fn = getattr(lib, f"__cabi_{name}")
            fn.argtypes = [_ctypes_for(ztype) for _, ztype in params]
            fn.restype = _ctypes_for(ret)
            namespace[name] = fn
        if self.last_build is not None:
            namespace["__zig_build__"] = self.last_build.as_dict()
        return namespace


__all__ = [
    "ZigBackend",
    "ZigCodegen",
    "ZigCompileError",
    "zig_available",
]
