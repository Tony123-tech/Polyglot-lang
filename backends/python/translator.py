"""
Polyglot translator: Rust AST -> Python AST -> callable.

Pipeline:
    source -> Lexer -> Parser -> Program
    Program -> Translator -> Python ast.Module
    ast.Module -> compile() -> exec() -> namespace

Covers roadmap layer 3:

- literals, idents, paths
- unary, binary, assignment, compound assignment
- call, method call, field, index
- tuple, array, repeat array
- struct literal -> dataclass constructor call
- if expr -> IfExp / If statement
- match -> if/elif chain
- closure -> lambda
- range -> range() call
- cast -> runtime no-op
- macro: println!/print!/eprintln!/format!/vec! -> print/format/list
- let / return / break / continue
- fn -> FunctionDef / AsyncFunctionDef
- struct -> @dataclass ClassDef
- enum -> Enum ClassDef
- impl -> methods injected into the target ClassDef
- trait -> class with NotImplementedError stubs
- use / type alias -> skipped
- const / static -> module-level assignment

Assign / CompoundAssign are translated in statement position only.
In expression position they raise TranslateError (Rust semantics).
"""

from __future__ import annotations

import ast
import sys
from copy import copy
from functools import cmp_to_key, reduce

from ast_nodes import (
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
    EnumDecl,
    Expr,
    ExprStmt,
    Field,
    FnDecl,
    FnParam,
    For,
    Ident,
    IfExpr,
    ImplBlock,
    Index,
    Item,
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
    ModDecl,
    Paren,
    PatBinding,
    PatBox,
    PatConst,
    PatEnum,
    Path,
    PatIdent,
    PatLiteral,
    PatMacro,
    PatOr,
    PatPath,
    PatRange,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    Pattern,
    PatTuple,
    PatWildcard,
    Program,
    Range,
    RepeatArray,
    Return,
    Stmt,
    StructDecl,
    StructLiteral,
    TraitDecl,
    Try,
    TupleExpr,
    Type,
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
)
from backends.base import Backend
from match_diagnostics import warn_match_issues
from struct_order import order_struct_declarations

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TranslateError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(f"{msg} at {line}:{col}")
        self.msg = msg
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "i8": "int", "i16": "int", "i32": "int", "i64": "int",
    "i128": "int", "isize": "int",
    "u8": "int", "u16": "int", "u32": "int", "u64": "int",
    "u128": "int", "usize": "int",
    "f32": "float", "f64": "float",
    "bool": "bool",
    "char": "str",
    "str": "str",
    "String": "str",
    "Vec": "list",
    "HashMap": "dict",
    "HashSet": "set",
    "Option": "object",
    "Result": "object",
}


# ---------------------------------------------------------------------------
# Macro mapping
# ---------------------------------------------------------------------------

MACRO_MAP: dict[str, str] = {
    "println": "print",
    "print": "print",
    "eprintln": "print",
    "eprint": "print",
    "format": "format",
    "vec": "list",
    "panic": "_panic",
    "assert": "assert",
    "assert_eq": "_assert_eq",
    "assert_ne": "_assert_ne",
}


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _runtime_panic(msg: str = "") -> None:
    raise RuntimeError(msg or "panic")


def _runtime_assert_eq(a, b) -> None:
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")


def _runtime_assert_ne(a, b) -> None:
    if a == b:
        raise AssertionError(f"{a!r} == {b!r}")


def _runtime_rust_len(value) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(value)


def _runtime_rust_slice(value, start, end):
    if isinstance(value, str):
        data = value.encode("utf-8")
        if start > end:
            raise ValueError("slice index starts at %d but ends at %d" % (start, end))
        return data[start:end].decode("utf-8")
    if start > end:
        raise ValueError("slice index starts at %d but ends at %d" % (start, end))
    return value[start:end]


def _runtime_vec_retain(values: list, predicate) -> None:
    values[:] = [value for value in values if predicate(value)]


def _runtime_vec_dedup(values: list) -> None:
    if values:
        values[:] = [
            values[0],
            *(value for previous, value in zip(values, values[1:])
              if value != previous),
        ]


def _runtime_vec_windows(values: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("window size must be positive")
    return [
        values[start:start + size]
        for start in range(max(0, len(values) - size + 1))
    ]


def _runtime_vec_chunks(values: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [
        values[start:start + size]
        for start in range(0, len(values), size)
    ]


class _PythonEnumVariant:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *values):
        return (self, *values)


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_dataclass():
    from dataclasses import dataclass
    return dataclass


def _import_enum():
    from enum import Enum
    return Enum


def _import_auto():
    from enum import auto
    return auto


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class Translator:
    def __init__(self) -> None:
        self.namespace: dict = {}
        self.struct_names: set[str] = set()
        self.struct_fields: dict[str, set[str]] = {}
        self.enum_names: set[str] = set()
        self.enum_variants: dict[str, set[str]] = {}
        self.constant_names: set[str] = set()
        self.impl_blocks: dict[str, list[ImplBlock]] = {}
        self.generic_struct_names: set[str] = set()
        self.generic_type_names: set[str] = set()

    # ---- entry -------------------------------------------------------------

    def translate_program(self, program: Program) -> ast.Module:
        self.struct_names.clear()
        self.struct_fields.clear()
        self.enum_names.clear()
        self.enum_variants.clear()
        self.constant_names.clear()
        self.generic_struct_names.clear()
        self.generic_type_names.clear()
        # Pass 1: collect struct / enum / impl names.
        for item in program.items:
            if isinstance(item, StructDecl):
                self.struct_names.add(item.name)
                self.struct_fields[item.name] = {
                    name for name, _ty, _is_pub in item.fields
                }
                if item.generics:
                    self.generic_struct_names.add(item.name)
                    self.generic_type_names.update(
                        param.name for param in item.generics
                    )
            elif isinstance(item, EnumDecl):
                self.enum_names.add(item.name)
                self.enum_variants[item.name] = {
                    variant.name for variant in item.variants
                }
            elif isinstance(item, ConstDecl):
                self.constant_names.add(item.name)
            elif isinstance(item, ImplBlock):
                target = self._type_name_of(item.ty)
                self.impl_blocks.setdefault(target, []).append(item)

        # Pass 2: emit statements. impl blocks are consumed here.
        body: list[ast.stmt] = []
        emitted_classes: set[str] = set()
        struct_items = [
            item for item in program.items if isinstance(item, StructDecl)
        ]
        for item in order_struct_declarations(struct_items):
            body.append(self.translate_struct(item))
            emitted_classes.add(item.name)

        for item in program.items:
            if isinstance(item, ImplBlock):
                continue
            if isinstance(item, StructDecl):
                continue
            elif isinstance(item, EnumDecl):
                body.append(self.translate_enum(item))
                emitted_classes.add(item.name)
            else:
                body.extend(self.translate_item(item))

        # Orphan impl blocks.
        for target, impls in self.impl_blocks.items():
            if target in emitted_classes:
                continue
            methods: list[ast.stmt] = []
            for impl in impls:
                methods.extend(self._impl_methods(impl))
            if not methods:
                methods = [ast.Pass()]
            body.append(ast.ClassDef(
                name=target, bases=[], keywords=[], body=methods,
                decorator_list=[],
            ))

        return ast.fix_missing_locations(
            ast.Module(body=body, type_ignores=[])
        )

    def compile_rust(self, program: Program) -> dict:
        import types

        module = self.translate_program(program)
        code = compile(module, "<polyglot>", "exec")
        ns: dict = {
            "__name__": "polyglot",
            "dataclass": _import_dataclass(),
            "Enum": _import_enum(),
            "auto": _import_auto(),
        }
        ns["_panic"] = _runtime_panic
        ns["_assert_eq"] = _runtime_assert_eq
        ns["_assert_ne"] = _runtime_assert_ne
        ns["_rust_len"] = _runtime_rust_len
        ns["_rust_slice"] = _runtime_rust_slice
        ns["_PythonEnumVariant"] = _PythonEnumVariant
        ns["_clone"] = copy
        ns["_cmp_to_key"] = cmp_to_key
        ns["_reduce"] = reduce
        ns["_vec_retain"] = _runtime_vec_retain
        ns["_vec_dedup"] = _runtime_vec_dedup
        ns["_vec_windows"] = _runtime_vec_windows
        ns["_vec_chunks"] = _runtime_vec_chunks

        # Register the namespace as a module so @dataclass can look it up
        # via sys.modules[cls.__module__].
        mod = types.ModuleType("polyglot")
        mod.__dict__.update(ns)
        sys.modules["polyglot"] = mod

        exec(code, mod.__dict__)  # noqa: S102
        self.namespace = mod.__dict__
        return mod.__dict__

    # ---- items -------------------------------------------------------------

    def translate_item(self, item: Item) -> list[ast.stmt]:
        if isinstance(item, FnDecl):
            return [self.translate_fn(item)]
        if isinstance(item, StructDecl):
            return [self.translate_struct(item)]
        if isinstance(item, EnumDecl):
            return [self.translate_enum(item)]
        if isinstance(item, ImplBlock):
            target = self._type_name_of(item.ty)
            methods: list[ast.stmt] = self._impl_methods(item)
            if not methods:
                methods = [ast.Pass()]
            cls = ast.ClassDef(
                name=target, bases=[], keywords=[], body=methods,
                decorator_list=[],
            )
            out: list[ast.stmt] = [cls]
            return out
        if isinstance(item, ConstDecl):
            return [self.translate_const(item)]
        if isinstance(item, UseDecl):
            return []
        if isinstance(item, ModDecl):
            out2: list[ast.stmt] = []
            for sub in item.items or []:
                out2.extend(self.translate_item(sub))
            return out2
        if isinstance(item, TypeAlias):
            return []
        if isinstance(item, TraitDecl):
            return [self.translate_trait(item)]
        raise TranslateError(
            f"unsupported item {type(item).__name__}",
            getattr(item, "line", 0), getattr(item, "col", 0),
        )

    # ---- fn ----------------------------------------------------------------

    def translate_fn(
        self, fn: FnDecl, *, is_method: bool = False,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        params: list[ast.arg] = [self.translate_fn_param(p) for p in fn.params]
        has_self = bool(params) and params[0].arg == "self"
        if is_method and not has_self:
            params.insert(0, ast.arg(arg="self", annotation=None))
        args = ast.arguments(
            posonlyargs=[], args=params, vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
        )
        body: list[ast.stmt] = []
        if fn.body is not None:
            body = self.translate_block_as_fn_body(fn.body)
        if not body:
            body = [ast.Pass()]
        ret: ast.expr | None = (
            self.translate_type(fn.ret) if fn.ret is not None else None
        )
        decorators: list[ast.expr] = []
        # impl method without self -> associated function, expose as
        # @staticmethod so `Point.new(x, y)` binds correctly.
        if is_method and not has_self:
            decorators.append(
                ast.Name(id="staticmethod", ctx=ast.Load())
            )
        if fn.is_async:
            return ast.AsyncFunctionDef(
                name=fn.name, args=args, body=body,
                decorator_list=decorators, returns=ret,
                type_comment=None,
            )
        return ast.FunctionDef(
            name=fn.name, args=args, body=body,
            decorator_list=decorators, returns=ret,
            type_comment=None,
        )

    def translate_fn_param(self, p: FnParam) -> ast.arg:
        pat = p.pattern
        if isinstance(pat, PatRef):
            pat = pat.inner
        if isinstance(pat, PatIdent):
            name = pat.name
        elif isinstance(pat, PatWildcard):
            name = "_"
        else:
            raise TranslateError(
                f"unsupported fn parameter pattern {type(pat).__name__}",
                p.line, p.col,
            )
        if name == "self":
            return ast.arg(arg="self", annotation=None)
        return ast.arg(arg=name, annotation=self._type_or_object(p.ty))

    def translate_block_as_fn_body(self, block: Block) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        for stmt in block.stmts:
            body.extend(self.translate_stmt(stmt))
        if block.tail is not None:
            body.append(ast.Return(value=self.translate_expr(block.tail)))
        return body

    # ---- struct ------------------------------------------------------------

    def translate_struct(self, s: StructDecl) -> ast.ClassDef:
        decorators: list[ast.expr] = [ast.Name(id="dataclass", ctx=ast.Load())]
        body: list[ast.stmt] = []
        for fname, fty, _pub in s.fields:
            body.append(ast.AnnAssign(
                target=ast.Name(id=fname, ctx=ast.Store()),
                annotation=self._type_or_object(fty),
                value=None,
                simple=1,
            ))
        for impl in self.impl_blocks.get(s.name, []):
            body.extend(self._impl_methods(impl))
        if not body:
            body = [ast.Pass()]
        return ast.ClassDef(
            name=s.name, bases=[], keywords=[], body=body,
            decorator_list=decorators,
        )

    def _impl_methods(self, impl: ImplBlock) -> list[ast.stmt]:
        methods: list[ast.stmt] = []
        for sub in impl.items:
            if isinstance(sub, FnDecl):
                methods.append(
                    self.translate_fn(sub, is_method=self._fn_is_method(sub))
                )
        return methods

    def _fn_is_method(self, fn: FnDecl) -> bool:
        if not fn.params:
            return False
        pat = fn.params[0].pattern
        if isinstance(pat, PatRef):
            pat = pat.inner
        return isinstance(pat, PatIdent) and pat.name == "self"

    # ---- enum --------------------------------------------------------------

    def translate_enum(self, e: EnumDecl) -> ast.ClassDef:
        body: list[ast.stmt] = []
        has_payloads = any(v.fields is not None for v in e.variants)
        for v in e.variants:
            body.append(ast.Assign(
                targets=[
                    ast.Name(
                        id=self._python_identifier(v.name),
                        ctx=ast.Store(),
                    ),
                ],
                value=(
                    ast.Call(
                        func=ast.Name(
                            id="_PythonEnumVariant", ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=v.name)],
                        keywords=[],
                    )
                    if has_payloads
                    else ast.Call(
                        func=ast.Name(id="auto", ctx=ast.Load()),
                        args=[], keywords=[],
                    )
                ),
            ))
        for impl in self.impl_blocks.get(e.name, []):
            body.extend(self._impl_methods(impl))
        if not body:
            body = [ast.Pass()]
        bases: list[ast.expr] = (
            [] if has_payloads else [ast.Name(id="Enum", ctx=ast.Load())]
        )
        return ast.ClassDef(
            name=e.name, bases=bases, keywords=[], body=body,
            decorator_list=[],
        )

    # ---- impl (standalone) -------------------------------------------------

    def translate_impl(self, impl: ImplBlock) -> list[ast.stmt]:
        target = self._type_name_of(impl.ty)
        methods: list[ast.stmt] = self._impl_methods(impl)
        if not methods:
            methods = [ast.Pass()]
        cls = ast.ClassDef(
            name=target, bases=[], keywords=[], body=methods,
            decorator_list=[],
        )
        out: list[ast.stmt] = [cls]
        return out

    def _type_name_of(self, ty: Type) -> str:
        if isinstance(ty, TypeName):
            return ty.name
        if isinstance(ty, TypeRef):
            return self._type_name_of(ty.inner)
        raise TranslateError(
            f"cannot get type name from {type(ty).__name__}",
            getattr(ty, "line", 0), getattr(ty, "col", 0),
        )

    # ---- trait -------------------------------------------------------------

    def translate_trait(self, t: TraitDecl) -> ast.ClassDef:
        body: list[ast.stmt] = []
        for item in t.items:
            if item.fn is not None:
                fn = self.translate_fn(item.fn, is_method=True)
                fn.body = [ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                        args=[], keywords=[],
                    ),
                    cause=None,
                )]
                body.append(fn)
        if not body:
            body = [ast.Pass()]
        return ast.ClassDef(
            name=t.name, bases=[], keywords=[], body=body,
            decorator_list=[],
        )

    # ---- const / static ----------------------------------------------------

    def translate_const(self, c: ConstDecl) -> ast.Assign:
        value = self.translate_expr(c.value)
        return ast.Assign(
            targets=[ast.Name(id=c.name, ctx=ast.Store())],
            value=value,
        )

    # ---- statements --------------------------------------------------------

    def translate_stmt(self, s: Stmt) -> list[ast.stmt]:
        if isinstance(s, LetStmt):
            return [self.translate_let(s)]
        if isinstance(s, ExprStmt):
            expr = s.expr
            if isinstance(expr, Assign):
                return [self._translate_assign_stmt(expr)]
            if isinstance(expr, CompoundAssign):
                return [self._translate_compound_assign_stmt(expr)]
            if isinstance(expr, MacroCall) and expr.path == "write":
                return [self._translate_write_macro(expr)]
            if isinstance(expr, IfExpr):
                return [self.translate_if_stmt(expr)]
            if isinstance(expr, While):
                return [self.translate_while(expr)]
            if isinstance(expr, For):
                return [self.translate_for(expr)]
            if isinstance(expr, Loop):
                return [self.translate_loop(expr)]
            if isinstance(expr, MatchExpr):
                return [self.translate_match_stmt(expr)]
            if isinstance(expr, Return):
                return [ast.Return(value=self.translate_opt_expr(expr.value))]
            if isinstance(expr, Break):
                return [ast.Break()]
            if isinstance(expr, Continue):
                return [ast.Continue()]
            if isinstance(expr, Block):
                return self.translate_block_as_body(expr)
            return [ast.Expr(value=self.translate_expr(expr))]
        if isinstance(s, ItemStmt):
            return self.translate_item(s.item)
        raise TranslateError(
            f"unsupported statement {type(s).__name__}",
            getattr(s, "line", 0), getattr(s, "col", 0),
        )

    def _translate_write_macro(self, macro: MacroCall) -> ast.AugAssign:
        if not isinstance(macro.args, list) or len(macro.args) < 2:
            raise TranslateError(
                "write! expects a mutable string and format string",
                macro.line, macro.col,
            )
        target_expr = macro.args[0]
        if isinstance(target_expr, Unary) and target_expr.op in ("&", "&mut"):
            target_expr = target_expr.operand
        if not isinstance(target_expr, Ident):
            raise TranslateError(
                "write! target must be an identifier",
                macro.line, macro.col,
            )
        format_value = self.translate_expr(macro.args[1])
        values = [self.translate_expr(arg) for arg in macro.args[2:]]
        if values:
            format_value = ast.Call(
                func=ast.Attribute(
                    value=format_value, attr="format", ctx=ast.Load(),
                ),
                args=values,
                keywords=[],
            )
        return ast.AugAssign(
            target=ast.Name(id=target_expr.name, ctx=ast.Store()),
            op=ast.Add(),
            value=format_value,
        )

    def _translate_assign_stmt(self, e: Assign) -> ast.Assign:
        target = self._translate_lvalue(e.target)
        value = self.translate_expr(e.value)
        return ast.Assign(targets=[target], value=value)

    def _translate_compound_assign_stmt(self, e: CompoundAssign) -> ast.AugAssign:
        target = self._translate_lvalue(e.target)
        value = self.translate_expr(e.value)
        op_map: dict[str, type[ast.operator]] = {
            "+=": ast.Add, "-=": ast.Sub,
            "*=": ast.Mult, "/=": ast.Div,
        }
        if e.op not in op_map:
            raise TranslateError(
                f"unsupported compound assign op {e.op!r}",
                e.line, e.col,
            )
        return ast.AugAssign(target=target, op=op_map[e.op](), value=value)

    def translate_block_as_body(self, block: Block) -> list[ast.stmt]:
        body: list[ast.stmt] = []
        for stmt in block.stmts:
            body.extend(self.translate_stmt(stmt))
        if block.tail is not None:
            body.append(ast.Expr(value=self.translate_expr(block.tail)))
        if not body:
            body = [ast.Pass()]
        return body

    def translate_let(self, s: LetStmt) -> ast.stmt:
        if not isinstance(s.pattern, PatIdent):
            raise TranslateError(
                f"unsupported let pattern {type(s.pattern).__name__}",
                s.line, s.col,
            )
        target = ast.Name(id=s.pattern.name, ctx=ast.Store())
        if s.value is None:
            return ast.AnnAssign(
                target=target,
                annotation=self._type_or_object(s.ty),
                value=None, simple=1,
            )
        value = self.translate_expr(s.value)
        return ast.Assign(targets=[target], value=value)

    def translate_while(self, w: While) -> ast.While:
        test = self.translate_expr(w.cond)
        body = self.translate_block_as_body(w.body)
        return ast.While(test=test, body=body, orelse=[])

    def translate_for(self, f: For) -> ast.For:
        if isinstance(f.pattern, PatIdent):
            target: ast.expr = ast.Name(
                id=f.pattern.name, ctx=ast.Store(),
            )
        elif isinstance(f.pattern, PatTuple):
            target = ast.Tuple(
                elts=[self._for_pattern_target(p) for p in f.pattern.elems],
                ctx=ast.Store(),
            )
        else:
            raise TranslateError(
                f"unsupported for pattern {type(f.pattern).__name__}",
                f.line, f.col,
            )
        iter_expr = self.translate_expr(f.iter)
        body = self.translate_block_as_body(f.body)
        return ast.For(target=target, iter=iter_expr, body=body, orelse=[])

    def _for_pattern_target(self, pattern: Pattern) -> ast.expr:
        if isinstance(pattern, PatIdent):
            return ast.Name(id=pattern.name, ctx=ast.Store())
        if isinstance(pattern, PatTuple):
            return ast.Tuple(
                elts=[self._for_pattern_target(p) for p in pattern.elems],
                ctx=ast.Store(),
            )
        raise TranslateError(
            f"unsupported for pattern {type(pattern).__name__}",
            pattern.line, pattern.col,
        )

    def translate_loop(self, lp: Loop) -> ast.While:
        body = self.translate_block_as_body(lp.body)
        return ast.While(test=ast.Constant(value=True), body=body, orelse=[])

    def translate_if_stmt(self, ifx: IfExpr) -> ast.If:
        test = self.translate_expr(ifx.cond)
        body = self.translate_block_as_body(ifx.then_block)
        orelse: list[ast.stmt] = []
        if isinstance(ifx.else_branch, Block):
            orelse = self.translate_block_as_body(ifx.else_branch)
        elif isinstance(ifx.else_branch, IfExpr):
            orelse = [self.translate_if_stmt(ifx.else_branch)]
        return ast.If(test=test, body=body, orelse=orelse)

    def translate_match_stmt(self, m: MatchExpr) -> ast.If:
        warn_match_issues(m)
        subject = self.translate_expr(m.scrutinee)
        node: ast.stmt | None = None
        for arm in reversed(m.arms):
            self._validate_pattern_bindings(arm.pattern)
            test, bindings = self._pattern_test_bindings(arm.pattern, subject)
            body: list[ast.stmt] = self._translate_match_body(arm.body)
            for name, value in reversed(list(bindings.items())):
                body.insert(0, ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=value,
                ))
            if arm.guard is not None:
                guarded_body: ast.stmt = ast.If(
                    test=self.translate_expr(arm.guard),
                    body=body,
                    orelse=[node] if node is not None else [],
                )
                body = [guarded_body]
            if test is None:
                node = ast.If(
                    test=ast.Constant(value=True), body=body,
                    orelse=[node] if node is not None else [],
                )
            else:
                node = ast.If(
                    test=test, body=body,
                    orelse=[node] if node is not None else [],
                )
        if node is None:
            return ast.If(
                test=ast.Constant(value=False),
                body=[ast.Pass()], orelse=[],
            )
        return node

    def _pattern_test(
        self, pat: Pattern, subject: ast.expr,
    ) -> ast.expr | None:
        return self._pattern_test_bindings(pat, subject)[0]

    def _pattern_test_bindings(
        self, pat: Pattern, subject: ast.expr,
    ) -> tuple[ast.expr, dict[str, ast.expr]]:
        if isinstance(pat, PatWildcard):
            return ast.Constant(value=True), {}
        if isinstance(pat, PatIdent):
            return ast.Constant(value=True), {pat.name: subject}
        if isinstance(pat, PatLiteral):
            return ast.Compare(
                left=subject, ops=[ast.Eq()],
                comparators=[self.translate_expr(pat.value)],
            ), {}
        if isinstance(pat, PatRange):
            tests: list[ast.expr] = []
            if pat.start is not None:
                tests.append(ast.Compare(
                    left=subject, ops=[ast.GtE()],
                    comparators=[self.translate_expr(pat.start)],
                ))
            if pat.end is not None:
                tests.append(ast.Compare(
                    left=subject,
                    ops=[ast.LtE() if pat.inclusive else ast.Lt()],
                    comparators=[self.translate_expr(pat.end)],
                ))
            return self._and_pattern_tests(tests), {}
        if isinstance(pat, PatTuple):
            tests: list[ast.expr] = [
                ast.Call(
                    func=ast.Name(id="isinstance", ctx=ast.Load()),
                    args=[subject, ast.Name(id="tuple", ctx=ast.Load())],
                    keywords=[],
                ),
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[subject], keywords=[],
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=len(pat.elems))],
                ),
            ]
            bindings: dict[str, ast.expr] = {}
            for index, child in enumerate(pat.elems):
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Subscript(
                        value=subject, slice=ast.Constant(value=index),
                        ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatSlice):
            rest_index = next(
                (
                    index for index, elem in enumerate(pat.elems)
                    if isinstance(elem, PatRest)
                    or (
                        isinstance(elem, PatBinding)
                        and isinstance(elem.inner, PatRest)
                    )
                ),
                None,
            )
            has_rest = rest_index is not None
            fixed_count = len(pat.elems) - int(has_rest)
            tests: list[ast.expr] = [
                ast.Call(
                    func=ast.Name(id="isinstance", ctx=ast.Load()),
                    args=[
                        subject,
                        ast.Tuple(
                            elts=[
                                ast.Name(id="list", ctx=ast.Load()),
                                ast.Name(id="tuple", ctx=ast.Load()),
                            ],
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                ),
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[subject], keywords=[],
                    ),
                    ops=[ast.GtE() if has_rest else ast.Eq()],
                    comparators=[ast.Constant(value=fixed_count)],
                ),
            ]
            bindings: dict[str, ast.expr] = {}
            trailing_count = (
                len(pat.elems) - rest_index - 1
                if rest_index is not None else 0
            )
            for index, child in enumerate(pat.elems):
                is_rest = isinstance(child, PatRest) or (
                    isinstance(child, PatBinding)
                    and isinstance(child.inner, PatRest)
                )
                if is_rest:
                    upper = (
                        ast.UnaryOp(
                            op=ast.USub(),
                            operand=ast.Constant(value=trailing_count),
                        )
                        if trailing_count else None
                    )
                    rest_value = ast.Subscript(
                        value=subject,
                        slice=ast.Slice(
                            lower=ast.Constant(value=rest_index),
                            upper=upper,
                            step=None,
                        ),
                        ctx=ast.Load(),
                    )
                    if isinstance(child, PatBinding):
                        bindings[child.name] = rest_value
                    continue
                sequence_index = index
                if rest_index is not None and index > rest_index:
                    sequence_index = index - len(pat.elems)
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Subscript(
                        value=subject,
                        slice=ast.Constant(value=sequence_index),
                        ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatStruct):
            class_name = pat.path.split("::")[-1]
            declared_fields = self.struct_fields.get(class_name)
            if declared_fields is not None:
                used_fields = {name for name, _child in pat.fields}
                extra_fields = used_fields - declared_fields
                if extra_fields:
                    raise TranslateError(
                        f"struct pattern {pat.path} has unknown fields: "
                        f"{', '.join(sorted(extra_fields))}",
                        pat.line, pat.col,
                    )
                missing_fields = declared_fields - used_fields
                if missing_fields and not pat.rest:
                    raise TranslateError(
                        f"struct pattern {pat.path} is missing fields; "
                        "include '..'",
                        pat.line, pat.col,
                    )
            tests = [ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[subject, ast.Name(id=class_name, ctx=ast.Load())],
                keywords=[],
            )]
            bindings = {}
            for name, child in pat.fields:
                child_test, child_bindings = self._pattern_test_bindings(
                    child,
                    ast.Attribute(
                        value=subject, attr=name, ctx=ast.Load(),
                    ),
                )
                tests.append(child_test)
                bindings.update(child_bindings)
            return self._and_pattern_tests(tests), bindings
        if isinstance(pat, PatBinding):
            if isinstance(pat.inner, PatRest):
                return ast.Constant(value=True), {pat.name: subject}
            if not pat.name.isidentifier():
                raise TranslateError(
                    f"invalid binding name {pat.name!r}",
                    pat.line, pat.col,
                )
            test, bindings = self._pattern_test_bindings(pat.inner, subject)
            bindings[pat.name] = subject
            return test, bindings
        if isinstance(pat, PatBox):
            return self._pattern_test_bindings(pat.inner, subject)
        if isinstance(pat, PatConst):
            parts = pat.path.split("::")
            resolved = (
                parts[0] in self.constant_names
                if len(parts) == 1
                else (
                    parts[0] in self.enum_variants
                    and parts[-1] in self.enum_variants[parts[0]]
                )
            )
            if not resolved:
                raise TranslateError(
                    f"unresolved constant pattern {pat.path!r}",
                    pat.line, pat.col,
                )
            return ast.Compare(
                left=subject,
                ops=[ast.Eq()],
                comparators=[self._path_to_expr(pat.path)],
            ), {}
        if isinstance(pat, PatPath):
            return ast.Compare(
                left=subject,
                ops=[ast.Eq()],
                comparators=[self._path_to_expr(pat.path)],
            ), {}
        if isinstance(pat, PatMacro):
            raise TranslateError(
                f"pattern macro {pat.path}! is not supported",
                pat.line, pat.col,
            )
        if isinstance(pat, PatRef):
            return self._pattern_test_bindings(pat.inner, subject)
        if isinstance(pat, PatOr):
            alternatives = [
                self._pattern_test_bindings(alt, subject)
                for alt in pat.alts
            ]
            binding_sets = [
                frozenset(bindings) for _test, bindings in alternatives
            ]
            if binding_sets and any(
                names != binding_sets[0] for names in binding_sets[1:]
            ):
                raise TranslateError(
                    "or-pattern alternatives must bind the same names",
                    pat.line, pat.col,
                )
            tests = [test for test, _bindings in alternatives]
            names = set().union(*(bindings for _test, bindings in alternatives))
            merged: dict[str, ast.expr] = {}
            for name in names:
                value = alternatives[-1][1][name]
                for test, bindings in reversed(alternatives[:-1]):
                    value = ast.IfExp(
                        test=test, body=bindings[name], orelse=value,
                    )
                merged[name] = value
            return ast.BoolOp(op=ast.Or(), values=tests), merged
        if isinstance(pat, PatEnum):
            enum_name, variant = self._split_path(pat.path)
            comparator: ast.expr
            if enum_name is not None:
                comparator = ast.Attribute(
                    value=ast.Name(id=enum_name, ctx=ast.Load()),
                    attr=self._python_identifier(variant),
                    ctx=ast.Load(),
                )
            else:
                comparator = ast.Name(
                    id=self._python_identifier(variant),
                    ctx=ast.Load(),
                )
            if pat.args:
                tests: list[ast.expr] = [
                    ast.Call(
                        func=ast.Name(id="isinstance", ctx=ast.Load()),
                        args=[subject, ast.Name(id="tuple", ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Compare(
                        left=ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[subject], keywords=[],
                        ),
                        ops=[ast.Eq()],
                        comparators=[
                            ast.Constant(value=len(pat.args) + 1),
                        ],
                    ),
                    ast.Compare(
                        left=ast.Subscript(
                            value=subject, slice=ast.Constant(value=0),
                            ctx=ast.Load(),
                        ),
                        ops=[ast.Eq()],
                        comparators=[comparator],
                    ),
                ]
                bindings: dict[str, ast.expr] = {}
                for index, child in enumerate(pat.args, start=1):
                    child_test, child_bindings = self._pattern_test_bindings(
                        child,
                        ast.Subscript(
                            value=subject, slice=ast.Constant(value=index),
                            ctx=ast.Load(),
                        ),
                    )
                    tests.append(child_test)
                    bindings.update(child_bindings)
                return self._and_pattern_tests(tests), bindings
            return ast.Compare(
                left=subject, ops=[ast.Eq()], comparators=[comparator],
            ), {}
        raise TranslateError(
            f"unsupported match pattern {type(pat).__name__}",
            getattr(pat, "line", 0), getattr(pat, "col", 0),
        )

    def _validate_pattern_bindings(self, pattern: Pattern) -> set[str]:
        if isinstance(pattern, PatIdent):
            return {pattern.name}
        if isinstance(pattern, PatBinding):
            names = self._validate_pattern_bindings(pattern.inner)
            if pattern.name in names:
                self._duplicate_binding_error(pattern, pattern.name)
            return names | {pattern.name}
        if isinstance(pattern, PatOr):
            alternatives = [
                self._validate_pattern_bindings(alt)
                for alt in pattern.alts
            ]
            if alternatives and any(
                names != alternatives[0] for names in alternatives[1:]
            ):
                raise TranslateError(
                    "or-pattern alternatives must bind the same names in "
                    f"{self._pattern_text(pattern)}",
                    pattern.line, pattern.col,
                )
            return alternatives[0] if alternatives else set()
        if isinstance(pattern, PatRef | PatBox):
            return self._validate_pattern_bindings(pattern.inner)
        if isinstance(pattern, (PatTuple, PatSlice)):
            children = pattern.elems
        elif isinstance(pattern, PatEnum):
            children = pattern.args
        elif isinstance(pattern, PatStruct):
            children = [child for _name, child in pattern.fields]
        else:
            return set()
        names: set[str] = set()
        for child in children:
            child_names = self._validate_pattern_bindings(child)
            duplicates = names & child_names
            if duplicates:
                self._duplicate_binding_error(
                    pattern, min(duplicates),
                )
            names.update(child_names)
        return names

    def _duplicate_binding_error(self, pattern: Pattern, name: str) -> None:
        raise TranslateError(
            f"duplicate binding {name!r} in pattern "
            f"{self._pattern_text(pattern)}",
            pattern.line, pattern.col,
        )

    def _pattern_text(self, pattern: Pattern) -> str:
        if isinstance(pattern, PatIdent):
            return pattern.name
        if isinstance(pattern, PatWildcard):
            return "_"
        if isinstance(pattern, PatLiteral):
            value = pattern.value
            if hasattr(value, "raw"):
                return value.raw
            return str(getattr(value, "value", value))
        if isinstance(pattern, PatTuple):
            inner = ", ".join(self._pattern_text(item) for item in pattern.elems)
            if len(pattern.elems) == 1:
                inner += ","
            return f"({inner})"
        if isinstance(pattern, PatSlice):
            return f"[{', '.join(self._pattern_text(item) for item in pattern.elems)}]"
        if isinstance(pattern, PatOr):
            return " | ".join(self._pattern_text(item) for item in pattern.alts)
        if isinstance(pattern, PatBinding):
            return f"{pattern.name} @ {self._pattern_text(pattern.inner)}"
        if isinstance(pattern, PatRef):
            return f"&{self._pattern_text(pattern.inner)}"
        if isinstance(pattern, PatRest):
            return ".."
        if isinstance(pattern, PatEnum):
            args = ", ".join(self._pattern_text(item) for item in pattern.args)
            return f"{pattern.path}({args})"
        if isinstance(pattern, PatStruct):
            fields = ", ".join(
                f"{name}: {self._pattern_text(child)}"
                for name, child in pattern.fields
            )
            return f"{pattern.path} {{ {fields} }}"
        return type(pattern).__name__

    def _and_pattern_tests(self, tests: list[ast.expr]) -> ast.expr:
        if not tests:
            return ast.Constant(value=True)
        if len(tests) == 1:
            return tests[0]
        return ast.BoolOp(op=ast.And(), values=tests)

    def _split_path(self, path: str) -> tuple[str | None, str]:
        parts = path.split("::")
        if len(parts) == 1:
            return None, parts[0]
        return parts[-2], parts[-1]

    def _path_to_expr(self, path: str) -> ast.expr:
        parts = path.split("::")
        expr: ast.expr = ast.Name(
            id=self._python_identifier(parts[0]), ctx=ast.Load(),
        )
        for part in parts[1:]:
            expr = ast.Attribute(
                value=expr,
                attr=self._python_identifier(part),
                ctx=ast.Load(),
            )
        return expr

    def _python_identifier(self, name: str) -> str:
        if name in {"None", "True", "False"}:
            return f"{name}_"
        return name

    def _translate_match_body(self, body: Expr) -> list[ast.stmt]:
        if isinstance(body, Block):
            return self.translate_block_as_fn_body(body)
        return [ast.Return(value=self.translate_expr(body))]

    # ---- expressions -------------------------------------------------------

    def translate_opt_expr(self, e: Expr | None) -> ast.expr | None:
        if e is None:
            return None
        return self.translate_expr(e)

    def translate_expr(self, e: Expr) -> ast.expr:
        if isinstance(e, LitInt):
            return ast.Constant(value=e.value)
        if isinstance(e, LitFloat):
            return ast.Constant(value=e.value)
        if isinstance(e, LitStr):
            return ast.Constant(value=e.value)
        if isinstance(e, LitChar):
            return ast.Constant(value=e.value)
        if isinstance(e, LitBool):
            return ast.Constant(value=e.value)
        if isinstance(e, Ident):
            return ast.Name(id=e.name, ctx=ast.Load())
        if isinstance(e, Path):
            # A::B::C -> if A is a known struct/enum, emit A.B (or A.B.C)
            if len(e.segments) >= 2:
                first = e.segments[0]
                if first in self.struct_names or first in self.enum_names:
                    node: ast.expr = ast.Name(id=first, ctx=ast.Load())
                    for seg in e.segments[1:]:
                        node = ast.Attribute(
                            value=node,
                            attr=self._python_identifier(seg),
                            ctx=ast.Load(),
                        )
                    return node
            return ast.Name(
                id=self._python_identifier(e.segments[-1]),
                ctx=ast.Load(),
            )
        if isinstance(e, Paren):
            return self.translate_expr(e.inner)
        if isinstance(e, Unary):
            return self._translate_unary(e)
        if isinstance(e, Binary):
            return self._translate_binary(e)
        if isinstance(e, (Assign, CompoundAssign)):
            raise TranslateError(
                "Assign cannot appear in expression position",
                e.line, e.col,
            )
        if isinstance(e, Call):
            func = self.translate_expr(e.func)
            args = [self.translate_expr(a) for a in e.args]
            return ast.Call(func=func, args=args, keywords=[])
        if isinstance(e, MethodCall):
            recv = self.translate_expr(e.receiver)
            if e.method in ("windows", "chunks") and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(
                        id=f"_vec_{e.method}", ctx=ast.Load(),
                    ),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "retain" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Name(id="_vec_retain", ctx=ast.Load()),
                    args=[recv, self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "dedup" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_vec_dedup", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "remove" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="pop", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "fold" and len(e.args) == 2:
                return ast.Call(
                    func=ast.Name(id="_reduce", ctx=ast.Load()),
                    args=[
                        self.translate_expr(e.args[1]),
                        recv,
                        self.translate_expr(e.args[0]),
                    ],
                    keywords=[],
                )
            if e.method == "map" and len(e.args) == 1:
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "Vec.map expects a single identifier parameter",
                        e.line, e.col,
                    )
                target = ast.Name(
                    id=closure.params[0][0].name, ctx=ast.Store(),
                )
                body = (
                    self.translate_expr(closure.body.tail)
                    if isinstance(closure.body, Block)
                    and closure.body.tail is not None
                    else self.translate_expr(closure.body)
                )
                return ast.ListComp(
                    elt=body,
                    generators=[
                        ast.comprehension(
                            target=target,
                            iter=recv,
                            ifs=[],
                            is_async=0,
                        ),
                    ],
                )
            if e.method == "filter" and len(e.args) == 1:
                closure = e.args[0]
                if (
                    not isinstance(closure, Closure)
                    or len(closure.params) != 1
                    or not isinstance(closure.params[0][0], PatIdent)
                ):
                    raise TranslateError(
                        "Vec.filter expects a single identifier parameter",
                        e.line, e.col,
                    )
                target = ast.Name(
                    id=closure.params[0][0].name, ctx=ast.Store(),
                )
                body = (
                    self.translate_expr(closure.body.tail)
                    if isinstance(closure.body, Block)
                    and closure.body.tail is not None
                    else self.translate_expr(closure.body)
                )
                return ast.ListComp(
                    elt=ast.Name(
                        id=closure.params[0][0].name,
                        ctx=ast.Load(),
                    ),
                    generators=[
                        ast.comprehension(
                            target=target,
                            iter=recv,
                            ifs=[body],
                            is_async=0,
                        ),
                    ],
                )
            if e.method == "sort_by" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="sort", ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="key",
                            value=ast.Call(
                                func=ast.Name(
                                    id="_cmp_to_key", ctx=ast.Load(),
                                ),
                                args=[self.translate_expr(e.args[0])],
                                keywords=[],
                            ),
                        ),
                    ],
                )
            if e.method == "contains" and len(e.args) == 1:
                return ast.Compare(
                    left=self.translate_expr(e.args[0]),
                    ops=[ast.In()],
                    comparators=[recv],
                )
            if e.method == "clone" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_clone", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "as_slice" and not e.args:
                return recv
            if e.method == "len" and not e.args:
                return ast.Call(
                    func=ast.Name(id="_rust_len", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "chars" and not e.args:
                return ast.Call(
                    func=ast.Name(id="list", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "bytes" and not e.args:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="encode", ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value="utf-8")],
                    keywords=[],
                )
            if e.method == "parse" and not e.args:
                return ast.Call(
                    func=ast.Name(id="int", ctx=ast.Load()),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "split" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr="split", ctx=ast.Load(),
                    ),
                    args=[self.translate_expr(e.args[0])],
                    keywords=[],
                )
            if e.method == "join" and len(e.args) == 1:
                return ast.Call(
                    func=ast.Attribute(
                        value=self.translate_expr(e.args[0]),
                        attr="join",
                        ctx=ast.Load(),
                    ),
                    args=[recv],
                    keywords=[],
                )
            if e.method == "trim" and not e.args:
                method = "strip"
                args: list[ast.expr] = []
            elif e.method in ("to_uppercase", "to_lowercase"):
                method = {
                    "to_uppercase": "upper",
                    "to_lowercase": "lower",
                }[e.method]
                args = []
            elif e.method in ("replace", "starts_with", "ends_with"):
                method = {
                    "starts_with": "startswith",
                    "ends_with": "endswith",
                }.get(e.method, e.method)
                args = [self.translate_expr(arg) for arg in e.args]
            else:
                method = ""
                args = []
            if method:
                return ast.Call(
                    func=ast.Attribute(
                        value=recv, attr=method, ctx=ast.Load(),
                    ),
                    args=args,
                    keywords=[],
                )
            attr = ast.Attribute(value=recv, attr=e.method, ctx=ast.Load())
            args = [self.translate_expr(a) for a in e.args]
            return ast.Call(func=attr, args=args, keywords=[])
        if isinstance(e, Field):
            recv = self.translate_expr(e.receiver)
            return ast.Attribute(value=recv, attr=e.name, ctx=ast.Load())
        if isinstance(e, Index):
            recv = self.translate_expr(e.receiver)
            if isinstance(e.index, Range):
                return ast.Call(
                    func=ast.Name(id="_rust_slice", ctx=ast.Load()),
                    args=[
                        recv,
                        self.translate_opt_expr(e.index.start)
                        or ast.Constant(value=None),
                        self.translate_opt_expr(e.index.end)
                        or ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            idx = self.translate_expr(e.index)
            return ast.Subscript(value=recv, slice=idx, ctx=ast.Load())
        if isinstance(e, TupleExpr):
            elts: list[ast.expr] = [self.translate_expr(x) for x in e.elems]
            return ast.Tuple(elts=elts, ctx=ast.Load())
        if isinstance(e, ArrayExpr):
            elts: list[ast.expr] = [self.translate_expr(x) for x in e.elems]
            return ast.List(elts=elts, ctx=ast.Load())
        if isinstance(e, RepeatArray):
            elem = self.translate_expr(e.elem)
            count = self.translate_expr(e.count)
            return ast.BinOp(
                left=ast.List(elts=[elem], ctx=ast.Load()),
                op=ast.Mult(), right=count,
            )
        if isinstance(e, StructLiteral):
            return self._translate_struct_literal(e)
        if isinstance(e, IfExpr):
            return ast.IfExp(
                test=self.translate_expr(e.cond),
                body=self.translate_expr(e.then_block.tail)
                if e.then_block.tail is not None
                else ast.Constant(value=None),
                orelse=self._translate_else_expr(e.else_branch),
            )
        if isinstance(e, MatchExpr):
            return self._translate_match_expr(e)
        if isinstance(e, Closure):
            return self._translate_closure(e)
        if isinstance(e, Range):
            args: list[ast.expr] = []
            if e.start is not None:
                args.append(self.translate_expr(e.start))
            else:
                args.append(ast.Constant(value=0))
            if e.end is not None:
                end = self.translate_expr(e.end)
                if e.inclusive:
                    end = ast.BinOp(
                        left=end, op=ast.Add(),
                        right=ast.Constant(value=1),
                    )
                args.append(end)
            return ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=args, keywords=[],
            )
        if isinstance(e, Cast):
            return self.translate_expr(e.expr)
        if isinstance(e, Return):
            raise TranslateError("Return is a statement here", e.line, e.col)
        if isinstance(e, MacroCall):
            return self._translate_macro(e)
        if isinstance(e, Await):
            raise TranslateError(
                "Await requires async context", e.line, e.col,
            )
        if isinstance(e, Try):
            raise TranslateError(
                "Try (?) not implemented", e.line, e.col,
            )
        if isinstance(e, Block):
            if e.tail is not None:
                return self.translate_expr(e.tail)
            return ast.Constant(value=None)
        raise TranslateError(
            f"unsupported expression {type(e).__name__}",
            getattr(e, "line", 0), getattr(e, "col", 0),
        )

    def _translate_unary(self, e: Unary) -> ast.expr:
        operand = self.translate_expr(e.operand)
        if e.op == "-":
            return ast.UnaryOp(op=ast.USub(), operand=operand)
        if e.op == "!":
            return ast.UnaryOp(op=ast.Not(), operand=operand)
        if e.op in ("&", "&mut", "*"):
            return operand
        raise TranslateError(f"unknown unary op {e.op}", e.line, e.col)

    def _translate_binary(self, e: Binary) -> ast.expr:
        left = self.translate_expr(e.left)
        right = self.translate_expr(e.right)
        if e.op == "+":
            return ast.BinOp(left=left, op=ast.Add(), right=right)
        if e.op == "-":
            return ast.BinOp(left=left, op=ast.Sub(), right=right)
        if e.op == "*":
            return ast.BinOp(left=left, op=ast.Mult(), right=right)
        if e.op == "/":
            return ast.BinOp(left=left, op=ast.Div(), right=right)
        if e.op == "%":
            return ast.BinOp(left=left, op=ast.Mod(), right=right)
        if e.op == "==":
            return ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
        if e.op == "!=":
            return ast.Compare(left=left, ops=[ast.NotEq()], comparators=[right])
        if e.op == "<":
            return ast.Compare(left=left, ops=[ast.Lt()], comparators=[right])
        if e.op == ">":
            return ast.Compare(left=left, ops=[ast.Gt()], comparators=[right])
        if e.op == "<=":
            return ast.Compare(left=left, ops=[ast.LtE()], comparators=[right])
        if e.op == ">=":
            return ast.Compare(left=left, ops=[ast.GtE()], comparators=[right])
        if e.op == "&&":
            return ast.BoolOp(op=ast.And(), values=[left, right])
        if e.op == "||":
            return ast.BoolOp(op=ast.Or(), values=[left, right])
        if e.op == "&":
            return ast.BinOp(left=left, op=ast.BitAnd(), right=right)
        if e.op == "|":
            return ast.BinOp(left=left, op=ast.BitOr(), right=right)
        raise TranslateError(f"unknown binary op {e.op}", e.line, e.col)

    def _translate_struct_literal(self, e: StructLiteral) -> ast.expr:
        name = e.path.split("::")[-1]
        if name in self.struct_names:
            keywords: list[ast.keyword] = [
                ast.keyword(arg=k, value=self.translate_expr(v))
                for k, v in e.fields
            ]
            return ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[], keywords=keywords,
            )
        keys: list[ast.expr | None] = [
            ast.Constant(value=k) for k, _ in e.fields
        ]
        values: list[ast.expr] = [
            self.translate_expr(v) for _, v in e.fields
        ]
        return ast.Dict(keys=keys, values=values)

    def _translate_lvalue(
        self, e: Expr,
    ) -> ast.Name | ast.Attribute | ast.Subscript:
        if isinstance(e, Ident):
            return ast.Name(id=e.name, ctx=ast.Store())
        if isinstance(e, Field):
            recv = self.translate_expr(e.receiver)
            return ast.Attribute(value=recv, attr=e.name, ctx=ast.Store())
        if isinstance(e, Index):
            recv = self.translate_expr(e.receiver)
            idx = self.translate_expr(e.index)
            return ast.Subscript(value=recv, slice=idx, ctx=ast.Store())
        raise TranslateError(
            f"unsupported lvalue {type(e).__name__}",
            getattr(e, "line", 0), getattr(e, "col", 0),
        )

    def _translate_else_expr(self, branch) -> ast.expr:
        if branch is None:
            return ast.Constant(value=None)
        if isinstance(branch, Block):
            if branch.tail is not None:
                return self.translate_expr(branch.tail)
            return ast.Constant(value=None)
        if isinstance(branch, IfExpr):
            return self.translate_expr(branch)
        return self.translate_expr(branch)

    def _translate_match_expr(self, m: MatchExpr) -> ast.expr:
        warn_match_issues(m)
        subject = self.translate_expr(m.scrutinee)
        node: ast.expr = ast.Constant(value=None)
        for arm in reversed(m.arms):
            self._validate_pattern_bindings(arm.pattern)
            test, bindings = self._pattern_test_bindings(arm.pattern, subject)
            if isinstance(arm.body, Block):
                body: ast.expr = (
                    self.translate_expr(arm.body.tail)
                    if arm.body.tail is not None
                    else ast.Constant(value=None)
                )
            else:
                body = self.translate_expr(arm.body)
            if bindings:
                guarded = body
                if arm.guard is not None:
                    guarded = ast.IfExp(
                        test=self.translate_expr(arm.guard),
                        body=body, orelse=node,
                    )
                body = ast.Call(
                    func=ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg=name) for name in bindings],
                            kwonlyargs=[], kw_defaults=[], defaults=[],
                        ),
                        body=guarded,
                    ),
                    args=list(bindings.values()),
                    keywords=[],
                )
                node = ast.IfExp(test=test, body=body, orelse=node)
            else:
                if arm.guard is not None:
                    test = self._and_pattern_tests([
                        test, self.translate_expr(arm.guard),
                    ])
                node = ast.IfExp(test=test, body=body, orelse=node)
        return node

    def _translate_closure(self, c: Closure) -> ast.Lambda:
        args: list[ast.arg] = []
        for pat, _ty in c.params:
            if not isinstance(pat, PatIdent):
                raise TranslateError(
                    f"unsupported closure param {type(pat).__name__}",
                    c.line, c.col,
                )
            args.append(ast.arg(arg=pat.name, annotation=None))
        body_expr: ast.expr
        if isinstance(c.body, Block):
            if c.body.tail is not None:
                body_expr = self.translate_expr(c.body.tail)
            else:
                body_expr = ast.Constant(value=None)
        else:
            body_expr = self.translate_expr(c.body)
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=args, vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=body_expr,
        )

    # ---- macros ------------------------------------------------------------

    def _translate_macro(self, m: MacroCall) -> ast.expr:
        target = MACRO_MAP.get(m.path)
        if target is None:
            raise TranslateError(
                f"unknown macro {m.path}!", m.line, m.col,
            )
        if isinstance(m.args, str):
            raise TranslateError(
                f"macro {m.path}! with token-stream args not supported",
                m.line, m.col,
            )
        if m.path == "assert":
            return ast.Call(
                func=ast.Name(id="assert", ctx=ast.Load()),
                args=[], keywords=[],
            )
        args: list[ast.expr] = [self.translate_expr(a) for a in m.args]
        if m.path in ("println", "print", "eprintln", "eprint", "format"):
            if not args:
                return ast.Call(
                    func=ast.Name(id=target, ctx=ast.Load()),
                    args=[ast.Constant(value="")], keywords=[],
                )
            fmt = args[0]
            rest = args[1:]
            if rest:
                fmt = ast.Call(
                    func=ast.Attribute(
                        value=fmt, attr="format", ctx=ast.Load(),
                    ),
                    args=rest, keywords=[],
                )
            return ast.Call(
                func=ast.Name(id=target, ctx=ast.Load()),
                args=[fmt], keywords=[],
            )
        return ast.Call(
            func=ast.Name(id=target, ctx=ast.Load()),
            args=args, keywords=[],
        )

    # ---- types -------------------------------------------------------------

    def _type_or_object(self, ty: Type | None) -> ast.expr:
        """Return a non-None type expression, falling back to `object`."""
        if ty is None:
            return ast.Name(id="object", ctx=ast.Load())
        result = self.translate_type(ty)
        if result is None:
            return ast.Name(id="object", ctx=ast.Load())
        return result

    def translate_type(self, ty: Type | None) -> ast.expr | None:
        if ty is None:
            return None
        if isinstance(ty, TypeName):
            if ty.name in self.generic_type_names:
                return ast.Name(id="int", ctx=ast.Load())
            base = TYPE_MAP.get(ty.name, ty.name)
            if ty.name in self.generic_struct_names:
                return ast.Name(id=base, ctx=ast.Load())
            if ty.args:
                if len(ty.args) > 1:
                    slice_multi: ast.expr = ast.Tuple(
                        elts=[self._type_or_object(a) for a in ty.args],
                        ctx=ast.Load(),
                    )
                    slice_ = slice_multi
                else:
                    slice_ = self._type_or_object(ty.args[0])
                return ast.Subscript(
                    value=ast.Name(id=base, ctx=ast.Load()),
                    slice=slice_,
                    ctx=ast.Load(),
                )
            return ast.Name(id=base, ctx=ast.Load())
        if isinstance(ty, TypeRef):
            return self.translate_type(ty.inner)
        if isinstance(ty, TypeSlice):
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=self._type_or_object(ty.inner),
                ctx=ast.Load(),
            )
        if isinstance(ty, TypeArray):
            return ast.Subscript(
                value=ast.Name(id="list", ctx=ast.Load()),
                slice=self._type_or_object(ty.inner),
                ctx=ast.Load(),
            )
        if isinstance(ty, TypeTuple):
            elts: list[ast.expr] = [
                self._type_or_object(x) for x in ty.elems
            ]
            return ast.Tuple(elts=elts, ctx=ast.Load())
        if isinstance(ty, TypeUnit):
            return ast.Constant(value=None)
        if isinstance(ty, (TypeFn, TypeImplTrait, TypeDynTrait)):
            return ast.Name(id="object", ctx=ast.Load())
        raise TranslateError(
            f"unsupported type {type(ty).__name__}",
            getattr(ty, "line", 0), getattr(ty, "col", 0),
        )


# ---------------------------------------------------------------------------
# Backend wrapper
# ---------------------------------------------------------------------------
class PythonBackend(Backend):
    """Backend that translates Rust AST to Python ast and execs it."""

    name = "python"

    def __init__(self, *, trusted: bool = True) -> None:
        super().__init__(trusted=trusted)
        self.translator = Translator()

    def compile(self, program: Program) -> dict:
        self._check_trust()
        return self.translator.compile_rust(program)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_rust(program: Program) -> dict:
    """Translate a parsed Rust Program to Python and exec it."""
    return Translator().compile_rust(program)


def translate_rust(program: Program) -> ast.Module:
    """Translate a parsed Rust Program to a Python ast.Module (no exec)."""
    return Translator().translate_program(program)
