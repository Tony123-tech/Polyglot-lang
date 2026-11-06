"""
AST -> Rust source code generator for the native backend.

Supports (v2 native, N2b):
- free functions with #[pyfunction]
- basic types: integers, floats, bool, String, &str
- Vec<T> with push / pop / len / index
- struct + #[pyclass]
- impl methods + #[pymethods] / #[new]
- expressions & control flow: if / while / for / loop / match
"""

from __future__ import annotations

from ast_nodes import (
    ArrayExpr,
    Assign,
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
    IfLet,
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
    MatchArm,
    MatchExpr,
    MethodCall,
    ModDecl,
    Node,
    OptionNone,
    OptionSome,
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
    ResultErr,
    ResultOk,
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
    TypeName,
    TypeRef,
    TypeSlice,
    TypeTuple,
    TypeUnit,
    Unary,
    UseDecl,
    While,
)
from match_diagnostics import warn_match_issues
from struct_order import order_struct_declarations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unit_type(ty: Type) -> bool:
    return isinstance(ty, TypeUnit) or (
        isinstance(ty, TypeTuple) and len(ty.elems) == 0
    )

_RUNTIME_PREFIX = r"""
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

fn _noop_waker() -> Waker {
    unsafe {
        let raw = RawWaker::new(std::ptr::null(), &VTABLE);
        Waker::from_raw(raw)
    }
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |_| RawWaker::new(std::ptr::null(), &VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

#[pyclass(unsendable)]
pub struct PyFuture {
    future: Option<Pin<Box<dyn Future<Output = PyResult<PyObject>> + 'static>>>,
}

#[pymethods]
impl PyFuture {
    fn __await__(mut slf: PyRefMut<'_, Self>) -> PyResult<Py<PyFutureIterator>> {
        let py = slf.py();
        let iter = PyFutureIterator {
            future: slf.future.take(),
        };
        Py::new(py, iter)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
}

#[pyclass(unsendable)]
pub struct PyFutureIterator {
    future: Option<Pin<Box<dyn Future<Output = PyResult<PyObject>> + 'static>>>,
}

#[pymethods]
impl PyFutureIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyObject> {
        let Some(mut fut) = slf.future.take() else {
            return Err(pyo3::exceptions::PyStopIteration::new_err(()));
        };
        let waker = _noop_waker();
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(value)) => {
                Err(pyo3::exceptions::PyStopIteration::new_err(value))
            }
            Poll::Ready(Err(e)) => Err(e),
            Poll::Pending => {
                slf.future = Some(fut);
                Ok(pyo3::Python::with_gil(|py| py.None()))
            }
        }
    }
}

fn py_future<F>(future: F) -> PyFuture
where
    F: Future<Output = PyResult<PyObject>> + 'static,
{
    PyFuture {
        future: Some(Box::pin(future)),
    }
}
"""

class RustCodegen:
    """Generate Rust source (with PyO3) from a Polyglot Program."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent_level = 0
        self.exported_fns: list[str] = []
        self.exported_classes: list[str] = []
        self.struct_names: set[str] = set()
        self.pyclass_names: set[str] = set()
        self.generic_struct_names: set[str] = set()
        self.generic_type_names: set[str] = set()
        self.string_bindings: set[str] = set()
        self.vec_bindings: set[str] = set()
        self.binding_types: dict[str, str] = {}
        self.line_map: dict[int, tuple[int, int]] = {}
        self.source_line_text_map: dict[int, str] = {}
        self.source_lines: dict[int, str] = {}
        self._origin: tuple[int, int] | None = None
        self._in_pymethods = False
        self._result_types: tuple[str, str] | None = None

    # ---- helpers ----------------------------------------------------------

    def emit(self, line: str = "") -> None:
        self.lines.append("    " * self.indent_level + line)
        if self._origin is not None:
            self.line_map[len(self.lines)] = self._origin
            source_text = self.source_lines.get(self._origin[0])
            if source_text is not None:
                self.source_line_text_map[len(self.lines)] = source_text

    def set_origin(self, node: Node) -> None:
        self._origin = (node.line, node.col)

    def indent(self) -> None:
        self.indent_level += 1

    def dedent(self) -> None:
        self.indent_level = max(0, self.indent_level - 1)

    def generate(self, program: Program) -> str:
        self.lines = []
        self.exported_fns = []
        self.exported_classes = []
        self.struct_names = set()
        self.pyclass_names = set()
        self.generic_struct_names = set()
        self.generic_type_names = set()
        self.string_bindings = set()
        self.vec_bindings = set()
        self.binding_types = {}
        self.line_map = {}
        self.source_line_text_map = {}
        self.source_lines = program.source_lines
        self._origin = None
        self.indent_level = 0
        self._in_pymethods = False
        self._result_types = None

        for item in program.items:
            if isinstance(item, StructDecl):
                self.struct_names.add(item.name)
                self.pyclass_names.add(item.name)
                if item.generics:
                    self.generic_struct_names.add(item.name)
                    self.generic_type_names.update(
                        param.name for param in item.generics
                    )
            elif isinstance(item, EnumDecl) and all(
                variant.fields is None for variant in item.variants
            ):
                self.pyclass_names.add(item.name)

                # Suppress warnings from aggressive `mut` / stub bodies.
        self.emit(
            "#![allow(unused_mut, unused_variables, dead_code, unused_imports)]"
        )
        self.emit()
        self.emit("use pyo3::prelude::*;")
        self.emit("use std::fmt::Write as FmtWrite;")
        self.emit()
        for line in _RUNTIME_PREFIX.strip().splitlines():
            self.emit(line)
        self.emit()

        structs = [
            item for item in program.items if isinstance(item, StructDecl)
        ]
        for struct in order_struct_declarations(structs):
            self.generate_struct(struct)
        emitted_impls: set[str] = set()
        for item in program.items:
            if not isinstance(item, StructDecl):
                if isinstance(item, ImplBlock):
                    target = self.type_to_rust(item.ty)
                    if target in emitted_impls:
                        continue
                    impls = [
                        candidate for candidate in program.items
                        if isinstance(candidate, ImplBlock)
                        and self.type_to_rust(candidate.ty) == target
                    ]
                    self.generate_impls(impls)
                    emitted_impls.add(target)
                else:
                    self.generate_item(item)

        self.emit()
        self.emit("#[pymodule]")
        self.emit(
            "fn polyglot_native(m: &Bound<'_, PyModule>) -> PyResult<()> {"
        )
        self.indent()
        for name in self.exported_fns:
            self.emit(f"m.add_function(wrap_pyfunction!({name}, m)?)?;")
        for name in self.exported_classes:
            self.emit(f"m.add_class::<{name}>()?;")
        self.emit("Ok(())")
        self.dedent()
        self.emit("}")

        return "\n".join(self.lines)

    # ---- items ------------------------------------------------------------

    def generate_item(self, item: Item) -> None:
        self.set_origin(item)
        if isinstance(item, FnDecl):
            self.generate_fn(item, as_method=False)
        elif isinstance(item, StructDecl):
            self.generate_struct(item)
        elif isinstance(item, EnumDecl):
            self.generate_enum(item)
        elif isinstance(item, ImplBlock):
            self.generate_impl(item)
        elif isinstance(item, ConstDecl):
            self.generate_const(item)
        elif isinstance(item, UseDecl):
            self.emit(f"use {item.path};")
        elif isinstance(item, ModDecl):
            if item.items is not None:
                self.emit(f"mod {item.name} {{")
                self.indent()
                for sub in item.items:
                    self.generate_item(sub)
                self.dedent()
                self.emit("}")
        elif isinstance(item, TypeAlias):
            self.emit(f"type {item.name} = {self.type_to_rust(item.ty)};")
        elif isinstance(item, TraitDecl):
            pass

    def generate_fn(self, fn: FnDecl, *, as_method: bool) -> None:
        self.set_origin(fn)
        previous_string_bindings = self.string_bindings
        self.string_bindings = set(previous_string_bindings)
        previous_binding_types = self.binding_types
        self.binding_types = {
            param.pattern.name: param.ty.name
            for param in fn.params
            if (
                isinstance(param.pattern, PatIdent)
                and isinstance(param.ty, TypeName)
            )
        }
        previous_vec_bindings = self.vec_bindings
        self.vec_bindings = set(previous_vec_bindings)
        self.string_bindings.update(
            param.pattern.name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            and isinstance(param.ty, TypeName)
            and param.ty.name == "String"
        )
        self.vec_bindings.update(
            param.pattern.name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            and isinstance(param.ty, TypeName)
            and param.ty.name == "Vec"
        )
        previous_result_types = self._result_types
        if (
            fn.ret is not None
            and isinstance(fn.ret, TypeName)
            and fn.ret.name == "Result"
            and len(fn.ret.args) == 2
        ):
            self._result_types = (
                self.type_to_rust(fn.ret.args[0]),
                self.type_to_rust(fn.ret.args[1]),
            )
        else:
            self._result_types = None

        is_ctor = as_method and fn.name == "new"

        if not as_method:
            self.emit("#[pyfunction]")
            self.exported_fns.append(fn.name)
        elif is_ctor:
            self.emit("#[new]")

        params = [self.param_to_rust(p, as_method=as_method) for p in fn.params]
        param_str = ", ".join(params)

        ret = ""
        if fn.ret is not None:
            if (
                not as_method
                and isinstance(fn.ret, TypeName)
                and fn.ret.name == "Result"
                and len(fn.ret.args) == 2
            ):
                ok_ty = self.type_to_rust(fn.ret.args[0])
                ret = f" -> PyResult<{ok_ty}>"
            else:
                ret = f" -> {self.type_to_rust(fn.ret)}"
        elif is_ctor:
            ret = " -> Self"

        async_prefix = "async " if fn.is_async else ""
        unsafe_prefix = "unsafe " if fn.is_unsafe else ""

        self.emit(f"{unsafe_prefix}{async_prefix}fn {fn.name}({param_str}){ret} {{")
        self.indent()
        if fn.body is not None:
            self.generate_block_body(fn.body, as_fn_body=True)
            if (
                fn.ret is not None
                and not _is_unit_type(fn.ret)
                and not fn.body.stmts
                and fn.body.tail is None
            ):
                self.emit("unreachable!()")
        else:
            self.emit("unreachable!()")
        self.dedent()
        self.emit("}")
        self.emit()
        self.string_bindings = previous_string_bindings
        self.vec_bindings = previous_vec_bindings
        self.binding_types = previous_binding_types
        self._result_types = previous_result_types

    def generate_struct(self, s: StructDecl) -> None:
        self.set_origin(s)
        self.exported_classes.append(s.name)
        self.emit("#[pyclass]")
        self.emit("#[derive(Clone, Debug)]")
        self.emit(f"struct {s.name} {{")
        self.indent()
        for name, ty, _is_pub in s.fields:
            self.emit("#[pyo3(get, set)]")
            self.emit(f"{name}: {self.type_to_rust(ty)},")
        self.dedent()
        self.emit("}")
        self.emit()

    def generate_enum(self, e: EnumDecl) -> None:
        self.set_origin(e)
        is_unit_enum = all(variant.fields is None for variant in e.variants)
        if is_unit_enum:
            self.exported_classes.append(e.name)
            self.emit("#[pyclass(eq, eq_int)]")
            self.emit("#[derive(Debug, PartialEq, Eq, Clone)]")
        self.emit(f"enum {e.name} {{")
        self.indent()
        for v in e.variants:
            if v.fields is None:
                self.emit(f"{v.name},")
            else:
                fields = ", ".join(self.type_to_rust(t) for t in v.fields)
                self.emit(f"{v.name}({fields}),")
        self.dedent()
        self.emit("}")
        self.emit()

    def generate_impl(self, impl: ImplBlock) -> None:
        self.generate_impls([impl])

    def generate_impls(self, impls: list[ImplBlock]) -> None:
        if not impls:
            return
        self.set_origin(impls[0])
        target = self.type_to_rust(impls[0].ty)
        use_pymethods = target in self.pyclass_names

        if use_pymethods:
            self.emit("#[pymethods]")
            self._in_pymethods = True

        self.emit(f"impl {target} {{")
        self.indent()
        for impl in impls:
            for item in impl.items:
                if isinstance(item, FnDecl):
                    self.generate_fn(item, as_method=True)
        self.dedent()
        self.emit("}")
        self.emit()
        self._in_pymethods = False

    def generate_const(self, c: ConstDecl) -> None:
        ty = self.type_to_rust(c.ty) if c.ty else "_"
        val = self.expr_to_rust(c.value)
        keyword = "static" if c.is_static else "const"
        self.emit(f"{keyword} {c.name}: {ty} = {val};")

    # ---- statements / blocks ----------------------------------------------

    def generate_block_body(self, block: Block, *, as_fn_body: bool = False) -> None:
        for stmt in block.stmts:
            self.generate_stmt(stmt)
        if block.tail is not None:
            self.set_origin(block.tail)
            self.emit(self.expr_to_rust(block.tail))
        elif as_fn_body and not block.stmts:
            self.emit("()")

    def generate_stmt(self, stmt: Stmt) -> None:
        self.set_origin(stmt)
        if isinstance(stmt, LetStmt):
            mut = "mut " if stmt.mutable else ""
            pat = self.pattern_to_rust(stmt.pattern)
            ty = f": {self.type_to_rust(stmt.ty)}" if stmt.ty else ""
            previous_result_types = self._result_types
            if (
                stmt.ty is not None
                and isinstance(stmt.ty, TypeName)
                and stmt.ty.name == "Result"
                and len(stmt.ty.args) == 2
            ):
                self._result_types = (
                    self.type_to_rust(stmt.ty.args[0]),
                    self.type_to_rust(stmt.ty.args[1]),
                )
            if stmt.value is not None:
                self.emit(f"let {mut}{pat}{ty} = {self.expr_to_rust(stmt.value)};")
                if isinstance(stmt.pattern, PatIdent):
                    binding_type = (
                        stmt.ty.name
                        if isinstance(stmt.ty, TypeName)
                        else self._expr_variant_type(stmt.value)
                    )
                    if binding_type in {"Option", "Result"}:
                        self.binding_types[stmt.pattern.name] = binding_type
                if (
                    isinstance(stmt.pattern, PatIdent)
                    and self._is_string_expr(stmt.value)
                ):
                    self.string_bindings.add(stmt.pattern.name)
                if (
                    isinstance(stmt.pattern, PatIdent)
                    and (
                        isinstance(stmt.ty, TypeName)
                        and stmt.ty.name == "Vec"
                        or self._is_vec_expr(stmt.value)
                    )
                ):
                    self.vec_bindings.add(stmt.pattern.name)
            else:
                self.emit(f"let {mut}{pat}{ty};")
            self._result_types = previous_result_types
        elif isinstance(stmt, ExprStmt):
            expr = stmt.expr
            if isinstance(expr, IfExpr):
                self._emit_if(expr, as_stmt=True)
            elif isinstance(expr, IfLet):
                self._emit_if_let(expr)
            elif isinstance(expr, While):
                self._emit_while(expr)
            elif isinstance(expr, For):
                self._emit_for(expr)
            elif isinstance(expr, Loop):
                self._emit_loop(expr)
            elif isinstance(expr, MatchExpr):
                self._emit_match(expr, as_stmt=True)
            elif isinstance(expr, Assign):
                self.emit(
                    f"{self.expr_to_rust(expr.target)} = "
                    f"{self.expr_to_rust(expr.value)};"
                )
            elif isinstance(expr, CompoundAssign):
                self.emit(
                    f"{self.expr_to_rust(expr.target)} {expr.op} "
                    f"{self.expr_to_rust(expr.value)};"
                )
            elif isinstance(expr, Return):
                if expr.value is None:
                    self.emit("return;")
                else:
                    self.emit(f"return {self.expr_to_rust(expr.value)};")
            elif isinstance(expr, Break):
                self.emit("break;")
            elif isinstance(expr, Continue):
                self.emit("continue;")
            elif isinstance(expr, Block):
                self.emit("{")
                self.indent()
                self.generate_block_body(expr)
                self.dedent()
                self.emit("}")
            else:
                self.emit(f"{self.expr_to_rust(expr)};")
        elif isinstance(stmt, ItemStmt):
            self.generate_item(stmt.item)
        else:
            self.emit(f"/* unsupported stmt {type(stmt).__name__} */")

    def _emit_if(self, ifx: IfExpr, *, as_stmt: bool) -> None:
        self.emit(f"if {self.expr_to_rust(ifx.cond)} {{")
        self.indent()
        self.generate_block_body(ifx.then_block)
        self.dedent()
        branch = ifx.else_branch
        while isinstance(branch, IfExpr):
            self.emit(f"}} else if {self.expr_to_rust(branch.cond)} {{")
            self.indent()
            self.generate_block_body(branch.then_block)
            self.dedent()
            branch = branch.else_branch
        if isinstance(branch, Block):
            self.emit("} else {")
            self.indent()
            self.generate_block_body(branch)
            self.dedent()
            self.emit("}")
        else:
            self.emit("}")

    def _emit_if_let(self, ifx: IfLet) -> None:
        self.emit(
            f"if let {self.pattern_to_rust(ifx.pattern)} = "
            f"{self.expr_to_rust(ifx.value)} {{"
        )
        self.indent()
        self.generate_block_body(ifx.then_block)
        self.dedent()
        if isinstance(ifx.else_branch, Block):
            self.emit("} else {")
            self.indent()
            self.generate_block_body(ifx.else_branch)
            self.dedent()
        elif isinstance(ifx.else_branch, IfLet):
            self.emit("} else {")
            self.indent()
            self._emit_if_let(ifx.else_branch)
            self.dedent()
        elif isinstance(ifx.else_branch, IfExpr):
            self.emit("} else {")
            self.indent()
            self._emit_if(ifx.else_branch, as_stmt=True)
            self.dedent()
        self.emit("}")

    def _emit_while(self, w: While) -> None:
        self.emit(f"while {self.expr_to_rust(w.cond)} {{")
        self.indent()
        self.generate_block_body(w.body)
        self.dedent()
        self.emit("}")

    def _emit_for(self, f: For) -> None:
        pat = self.pattern_to_rust(f.pattern)
        self.emit(f"for {pat} in {self.expr_to_rust(f.iter)} {{")
        self.indent()
        self.generate_block_body(f.body)
        self.dedent()
        self.emit("}")

    def _emit_loop(self, lp: Loop) -> None:
        self.emit("loop {")
        self.indent()
        self.generate_block_body(lp.body)
        self.dedent()
        self.emit("}")

    def _emit_match(self, m: MatchExpr, *, as_stmt: bool) -> None:
        warn_match_issues(
            m, scrutinee_type=self._match_scrutinee_type(m),
        )
        scrutinee = self.expr_to_rust(m.scrutinee)
        if any(isinstance(arm.pattern, PatSlice) for arm in m.arms):
            scrutinee = f"({scrutinee}).as_slice()"
        self.emit(f"match {scrutinee} {{")
        self.indent()
        for arm in m.arms:
            self._emit_match_arm(arm)
        self.dedent()
        self.emit("}")

    def _emit_match_arm(self, arm: MatchArm) -> None:
        pat = self.pattern_to_rust(arm.pattern)
        guard = ""
        if arm.guard is not None:
            guard = f" if {self.expr_to_rust(arm.guard)}"
        body = arm.body
        if isinstance(body, Block):
            self.emit(f"{pat}{guard} => {{")
            self.indent()
            self.generate_block_body(body)
            self.dedent()
            self.emit("}")
        else:
            self.emit(f"{pat}{guard} => {self.expr_to_rust(body)},")

    def _format_str(self, e: Expr) -> str:
        """Emit a bare string literal for the first arg of format!/println!."""
        if isinstance(e, LitStr):
            escaped = e.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return self.expr_to_rust(e)

    # ---- expressions ------------------------------------------------------

    def expr_to_rust(self, e: Expr) -> str:
        if isinstance(e, LitInt):
            return e.raw
        if isinstance(e, LitFloat):
            return e.raw
        if isinstance(e, LitStr):
            escaped = e.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'String::from("{escaped}")'
        if isinstance(e, LitChar):
            return f"'{e.value}'"
        if isinstance(e, LitBool):
            return "true" if e.value else "false"
        if isinstance(e, Ident):
            return e.name
        if isinstance(e, Path):
            return "::".join(e.segments)
        if isinstance(e, Paren):
            return f"({self.expr_to_rust(e.inner)})"
        if isinstance(e, OptionSome):
            return f"Some({self.expr_to_rust(e.value)})"
        if isinstance(e, OptionNone):
            return "None"
        if isinstance(e, ResultOk):
            if self._result_types is not None:
                ok_ty, _err_ty = self._result_types
                return (
                    f"Ok::<{ok_ty}, PyErr>("
                    f"{self.expr_to_rust(e.value)})"
                )
            return f"Ok::<_, PyErr>({self.expr_to_rust(e.value)})"
        if isinstance(e, ResultErr):
            if self._result_types is not None:
                ok_ty, _err_ty = self._result_types
                return (
                    f"Err::<{ok_ty}, PyErr>("
                    f"pyo3::exceptions::PyRuntimeError::new_err("
                    f"{self.expr_to_rust(e.value)}))"
                )
            return (
                f"Err::<_, PyErr>(pyo3::exceptions::PyRuntimeError::new_err("
                f"{self.expr_to_rust(e.value)}))"
            )
        if isinstance(e, Unary):
            if (
                e.op == "&"
                and isinstance(e.operand, Index)
                and isinstance(e.operand.index, Range)
            ):
                return self.expr_to_rust(e.operand)
            op = "&mut " if e.op == "&mut" else e.op
            return f"({op}{self.expr_to_rust(e.operand)})"
        if isinstance(e, Binary):
            if e.op == "+" and (
                self._is_string_expr(e.left)
                or self._is_string_expr(e.right)
            ):
                return (
                    f'format!("{{}}{{}}", {self.expr_to_rust(e.left)}, '
                    f"{self.expr_to_rust(e.right)})"
                )
            return (
                f"({self.expr_to_rust(e.left)} {e.op} "
                f"{self.expr_to_rust(e.right)})"
            )
        if isinstance(e, Assign):
            return (
                f"({self.expr_to_rust(e.target)} = "
                f"{self.expr_to_rust(e.value)})"
            )
        if isinstance(e, CompoundAssign):
            return (
                f"({self.expr_to_rust(e.target)} {e.op} "
                f"{self.expr_to_rust(e.value)})"
            )
        if isinstance(e, Call):
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            return f"{self.expr_to_rust(e.func)}({args})"
        if isinstance(e, MethodCall):
            recv = self.expr_to_rust(e.receiver)
            if e.method == "map_err" and len(e.args) == 1:
                return f"{recv}.map_err({self.expr_to_rust(e.args[0])})"
            if e.method == "unwrap" and not e.args:
                return f"{recv}.unwrap()"
            if e.method == "unwrap_or" and len(e.args) == 1:
                return f"{recv}.unwrap_or({self.expr_to_rust(e.args[0])})"
            if e.method == "unwrap_or_else" and len(e.args) == 1:
                return f"{recv}.unwrap_or_else({self.expr_to_rust(e.args[0])})"
            if e.method == "chars" and not e.args:
                return f"{recv}.chars().collect::<Vec<char>>()"
            if e.method == "bytes" and not e.args:
                return f"{recv}.bytes().collect::<Vec<u8>>()"
            if e.method == "parse" and not e.args:
                target = (
                    self.type_to_rust(e.generic_args[0])
                    if e.generic_args
                    else "i64"
                )
                return f"{recv}.parse::<{target}>().unwrap()"
            if e.method == "split" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.split(&{separator})"
                    ".map(String::from).collect::<Vec<_>>()"
                )
            if e.method == "join" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return f"{recv}.join(&{separator})"
            if e.method == "trim" and not e.args:
                return f"{recv}.trim().to_string()"
            if e.method == "replace" and len(e.args) == 2:
                old = self.expr_to_rust(e.args[0])
                new = self.expr_to_rust(e.args[1])
                return f"{recv}.replace(&{old}, &{new})"
            if e.method in ("starts_with", "ends_with") and len(e.args) == 1:
                arg = self.expr_to_rust(e.args[0])
                return f"{recv}.{e.method}(&{arg})"
            if (
                e.method == "map"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
            ):
                return (
                    f"{recv}.into_iter().map("
                    f"{self.expr_to_rust(e.args[0])})"
                    ".collect::<Vec<_>>()"
                )
            if (
                e.method == "filter"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
                and self._is_vec_expr(e.receiver)
            ):
                return (
                    f"{recv}.into_iter().filter("
                    f"{self.expr_to_rust(e.args[0])})"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "fold" and len(e.args) == 2:
                return (
                    f"{recv}.into_iter().fold("
                    f"{self.expr_to_rust(e.args[0])}, "
                    f"{self.expr_to_rust(e.args[1])})"
                )
            if e.method == "len" and not e.args:
                return f"({recv}.len() as i64)"
            if e.method == "contains" and len(e.args) == 1:
                arg = self.expr_to_rust(e.args[0])
                return f"{recv}.contains(&{arg})"
            if e.method == "insert" and len(e.args) == 2:
                index = self.expr_to_rust(e.args[0])
                value = self.expr_to_rust(e.args[1])
                return f"{recv}.insert(({index}) as usize, {value})"
            if e.method == "remove" and len(e.args) == 1:
                index = self.expr_to_rust(e.args[0])
                return f"{recv}.remove(({index}) as usize)"
            if e.method in ("windows", "chunks") and len(e.args) == 1:
                size = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.{e.method}(({size}) as usize)"
                    ".map(|window| window.to_vec())"
                    ".collect::<Vec<_>>()"
                )
            if (
                e.method == "sort_by"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
                and len(e.args[0].params) == 2
                and isinstance(e.args[0].body, Binary)
                and e.args[0].body.op == "-"
            ):
                left_pat = e.args[0].params[0][0]
                right_pat = e.args[0].params[1][0]
                body = e.args[0].body
                if (
                    isinstance(left_pat, PatIdent)
                    and isinstance(right_pat, PatIdent)
                    and isinstance(body.left, Ident)
                    and isinstance(body.right, Ident)
                    and body.left.name == left_pat.name
                    and body.right.name == right_pat.name
                ):
                    return (
                        f"{recv}.sort_by(|{left_pat.name}, "
                        f"{right_pat.name}| {left_pat.name}.cmp("
                        f"{right_pat.name}))"
                    )
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            return f"{self.expr_to_rust(e.receiver)}.{e.method}({args})"
        if isinstance(e, Field):
            return f"{self.expr_to_rust(e.receiver)}.{e.name}"
        if isinstance(e, Index):
            recv = self.expr_to_rust(e.receiver)
            if isinstance(e.index, Range):
                start = (
                    self.expr_to_rust(e.index.start)
                    if e.index.start is not None
                    else ""
                )
                end = (
                    self.expr_to_rust(e.index.end)
                    if e.index.end is not None
                    else ""
                )
                op = "..=" if e.index.inclusive else ".."
                if self._is_string_expr(e.receiver):
                    return f"{recv}[{start}{op}{end}].to_string()"
                return f"{recv}[{start}{op}{end}].to_vec()"
            idx = self.expr_to_rust(e.index)
            return f"{recv}[({idx}) as usize]"
        if isinstance(e, TupleExpr):
            elems = ", ".join(self.expr_to_rust(x) for x in e.elems)
            if len(e.elems) == 1:
                return f"({elems},)"
            return f"({elems})"
        if isinstance(e, ArrayExpr):
            elems = ", ".join(self.expr_to_rust(x) for x in e.elems)
            return f"vec![{elems}]"
        if isinstance(e, RepeatArray):
            return (
                f"vec![{self.expr_to_rust(e.elem)}; "
                f"{self.expr_to_rust(e.count)}]"
            )
        if isinstance(e, StructLiteral):
            name = e.path.split("::")[-1]
            fields = ", ".join(
                f"{k}: {self.expr_to_rust(v)}" for k, v in e.fields
            )
            rest = ""
            if e.rest is not None:
                rest = f", ..{self.expr_to_rust(e.rest)}"
            return f"{name} {{ {fields}{rest} }}"
        if isinstance(e, IfExpr):
            then = self._block_as_expr(e.then_block)
            else_part = "()"
            if isinstance(e.else_branch, Block):
                else_part = self._block_as_expr(e.else_branch)
            elif e.else_branch is not None:
                else_part = self.expr_to_rust(e.else_branch)
            return (
                f"(if {self.expr_to_rust(e.cond)} "
                f"{{ {then} }} else {{ {else_part} }})"
            )
        if isinstance(e, MatchExpr):
            warn_match_issues(
                e, scrutinee_type=self._match_scrutinee_type(e),
            )
            arms = []
            for arm in e.arms:
                pat = self.pattern_to_rust(arm.pattern)
                guard = ""
                if arm.guard is not None:
                    guard = f" if {self.expr_to_rust(arm.guard)}"
                if isinstance(arm.body, Block):
                    body = f"{{ {self._block_as_expr(arm.body)} }}"
                else:
                    body = self.expr_to_rust(arm.body)
                arms.append(f"{pat}{guard} => {body}")
            scrutinee = self.expr_to_rust(e.scrutinee)
            if any(isinstance(arm.pattern, PatSlice) for arm in e.arms):
                scrutinee = f"({scrutinee}).as_slice()"
            return (
                f"(match {scrutinee} {{ {', '.join(arms)} }})"
            )
        if isinstance(e, Loop):
            return "/* loop-expr not supported as value */"
        if isinstance(e, While):
            return "/* while-expr not supported as value */"
        if isinstance(e, For):
            return "/* for-expr not supported as value */"
        if isinstance(e, Closure):
            params = []
            for pat, ty in e.params:
                p = self.pattern_to_rust(pat)
                if ty is not None:
                    p = f"{p}: {self.type_to_rust(ty)}"
                params.append(p)
            body = (
                self._block_as_expr(e.body)
                if isinstance(e.body, Block)
                else self.expr_to_rust(e.body)
            )
            ret = ""
            if e.ret is not None:
                ret = f" -> {self.type_to_rust(e.ret)}"
            return f"|{', '.join(params)}|{ret} {{ {body} }}"
        if isinstance(e, Range):
            start = self.expr_to_rust(e.start) if e.start is not None else ""
            end = self.expr_to_rust(e.end) if e.end is not None else ""
            op = "..=" if e.inclusive else ".."
            return f"({start}{op}{end})"
        if isinstance(e, Cast):
            return (
                f"({self.expr_to_rust(e.expr)} as "
                f"{self.type_to_rust(e.ty)})"
            )
        if isinstance(e, Try):
            if isinstance(e.expr, MethodCall) and e.expr.method == "parse":
                recv = self.expr_to_rust(e.expr.receiver)
                target = (
                    self.type_to_rust(e.expr.generic_args[0])
                    if e.expr.generic_args
                    else "i64"
                )
                return f"{recv}.parse::<{target}>()?"
            return f"{self.expr_to_rust(e.expr)}?"
        if isinstance(e, Return):
            if e.value is None:
                return "return"
            return f"return {self.expr_to_rust(e.value)}"
        if isinstance(e, Break):
            if e.value is not None:
                return f"break {self.expr_to_rust(e.value)}"
            return "break"
        if isinstance(e, Continue):
            return "continue"
        if isinstance(e, MacroCall):
            if isinstance(e.args, str):
                return f"{e.path}!({e.args})"
            if e.path in ("println", "print", "eprintln", "eprint", "format"):
                if not e.args:
                    return f"{e.path}!()"
                fmt_str = self._format_str(e.args[0])
                rest = ", ".join(self.expr_to_rust(a) for a in e.args[1:])
                if rest:
                    return f"{e.path}!({fmt_str}, {rest})"
                return f"{e.path}!({fmt_str})"
            if e.path == "write":
                if len(e.args) < 2:
                    return "write!().unwrap()"
                target = self.expr_to_rust(e.args[0])
                format_string = self._format_str(e.args[1])
                rest = ", ".join(
                    self.expr_to_rust(arg) for arg in e.args[2:]
                )
                args = f"{target}, {format_string}"
                if rest:
                    args += f", {rest}"
                return f"write!({args}).unwrap()"
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            if e.path == "vec":
                return f"vec![{args}]"
            return f"{e.path}!({args})"
        if isinstance(e, Block):
            return f"{{ {self._block_as_expr(e)} }}"
        return f"/* expr {type(e).__name__} */"

    def _is_string_expr(self, expr: Expr) -> bool:
        if isinstance(expr, LitStr):
            return True
        if isinstance(expr, Ident):
            return expr.name in self.string_bindings
        if isinstance(expr, MethodCall):
            return expr.method in {
                "replace",
                "trim",
                "to_uppercase",
                "to_lowercase",
                "parse",
            }
        if isinstance(expr, MacroCall):
            return expr.path == "format"
        return False

    def _is_vec_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            return expr.name in self.vec_bindings
        if isinstance(expr, (ArrayExpr, RepeatArray)):
            return True
        if isinstance(expr, MacroCall):
            return expr.path == "vec"
        if isinstance(expr, Paren):
            return self._is_vec_expr(expr.inner)
        return False

    def _match_scrutinee_type(self, match: MatchExpr) -> str | None:
        if isinstance(match.scrutinee, Ident):
            return self.binding_types.get(match.scrutinee.name)
        return self._expr_variant_type(match.scrutinee)

    def _expr_variant_type(self, expr: Expr) -> str | None:
        if isinstance(expr, (OptionSome, OptionNone)):
            return "Option"
        if isinstance(expr, (ResultOk, ResultErr)):
            return "Result"
        if isinstance(expr, MethodCall) and expr.method in {"pop", "next"}:
            return "Option"
        return None

    def _block_as_expr(self, block: Block) -> str:
        parts: list[str] = []
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                mut = "mut "
                pat = self.pattern_to_rust(stmt.pattern)
                ty = f": {self.type_to_rust(stmt.ty)}" if stmt.ty else ""
                if stmt.value is not None:
                    parts.append(
                        f"let {mut}{pat}{ty} = {self.expr_to_rust(stmt.value)};"
                    )
                else:
                    parts.append(f"let {mut}{pat}{ty};")
            elif isinstance(stmt, ExprStmt):
                parts.append(f"{self.expr_to_rust(stmt.expr)};")
        if block.tail is not None:
            parts.append(self.expr_to_rust(block.tail))
        elif not parts:
            parts.append("()")
        return " ".join(parts)

    # ---- types & patterns -------------------------------------------------

    def type_to_rust(self, ty: Type | None) -> str:
        if ty is None:
            return "()"
        if isinstance(ty, TypeName):
            name = ty.name
            if name in self.generic_type_names:
                return "i64"
            if name == "str":
                name = "String"
            if name in self.generic_struct_names:
                return name
            if ty.args:
                args = ", ".join(self.type_to_rust(a) for a in ty.args)
                if name == "Vec":
                    return f"Vec<{args}>"
                return f"{name}<{args}>"
            return name
        if isinstance(ty, TypeRef):
            inner = ty.inner
            if isinstance(inner, TypeName) and inner.name == "str":
                mut = "mut " if ty.mutable else ""
                return f"&{mut}str"
            mut = "mut " if ty.mutable else ""
            return f"&{mut}{self.type_to_rust(inner)}"
        if isinstance(ty, TypeSlice):
            return f"[{self.type_to_rust(ty.inner)}]"
        if isinstance(ty, TypeArray):
            return (
                f"[{self.type_to_rust(ty.inner)}; "
                f"{self.expr_to_rust(ty.size)}]"
            )
        if isinstance(ty, TypeTuple):
            elems = ", ".join(self.type_to_rust(t) for t in ty.elems)
            if len(ty.elems) == 1:
                elems += ","
            return f"({elems})"
        if isinstance(ty, TypeUnit):
            return "()"
        return "()"

    def param_to_rust(self, p: FnParam, *, as_method: bool = False) -> str:
        pat = self.pattern_to_rust(p.pattern)

        if isinstance(p.pattern, PatIdent) and p.pattern.name == "self":
            return "self"
        if isinstance(p.pattern, PatRef):
            inner = p.pattern.inner
            if isinstance(inner, PatIdent) and inner.name == "self":
                return "&mut self" if p.pattern.mutable else "&self"

        if p.ty is None:
            return pat

        if not (isinstance(p.pattern, PatIdent) and p.pattern.mutable):
            pat = f"mut {pat}"
        return f"{pat}: {self.type_to_rust(p.ty)}"

    def pattern_to_rust(self, pat: Pattern) -> str:
        if isinstance(pat, PatIdent):
            mut = "mut " if pat.mutable else ""
            return f"{mut}{pat.name}"
        if isinstance(pat, PatWildcard):
            return "_"
        if isinstance(pat, PatRef):
            mut = "mut " if pat.mutable else ""
            return f"&{mut}{self.pattern_to_rust(pat.inner)}"
        if isinstance(pat, PatLiteral):
            val = pat.value
            if isinstance(val, LitBool):
                return "true" if val.value else "false"
            if isinstance(val, Unary) and val.op == "-":
                return f"-{self._pattern_bound_to_rust(val.operand)}"
            raw = getattr(val, "raw", None)
            if raw is not None:
                return raw
            value = getattr(val, "value", None)
            if value is not None:
                return repr(value)
            return str(val)
        if isinstance(pat, PatTuple):
            elems = ", ".join(self.pattern_to_rust(p) for p in pat.elems)
            if len(pat.elems) == 1:
                elems += ","
            return f"({elems})"
        if isinstance(pat, PatStruct):
            fields = ", ".join(
                f"{name}: {self.pattern_to_rust(field_pattern)}"
                for name, field_pattern in pat.fields
            )
            if pat.rest:
                fields = f"{fields}, .." if fields else ".."
            return f"{pat.path} {{ {fields} }}"
        if isinstance(pat, PatEnum):
            args = ", ".join(self.pattern_to_rust(a) for a in pat.args)
            if args:
                return f"{pat.path}({args})"
            return pat.path
        if isinstance(pat, PatOr):
            return " | ".join(self.pattern_to_rust(alt) for alt in pat.alts)
        if isinstance(pat, PatRange):
            start = (
                self._pattern_bound_to_rust(pat.start)
                if pat.start is not None else ""
            )
            end = (
                self._pattern_bound_to_rust(pat.end)
                if pat.end is not None else ""
            )
            op = "..=" if pat.inclusive else ".."
            return f"{start}{op}{end}"
        if isinstance(pat, PatBinding):
            inner = self.pattern_to_rust(pat.inner)
            if isinstance(pat.inner, PatOr):
                inner = f"({inner})"
            return f"{pat.name} @ {inner}"
        if isinstance(pat, PatBox):
            return f"box {self.pattern_to_rust(pat.inner)}"
        if isinstance(pat, PatConst):
            return pat.path if "::" in pat.path else f"self::{pat.path}"
        if isinstance(pat, PatPath):
            return pat.path
        if isinstance(pat, PatMacro):
            return f"{pat.path}!{pat.delimiter[0]}{pat.tokens}{pat.delimiter[1]}"
        if isinstance(pat, PatSlice):
            elems = ", ".join(
                self.pattern_to_rust(
                    elem.inner if isinstance(elem, PatRef) else elem,
                )
                for elem in pat.elems
            )
            return f"[{elems}]"
        if isinstance(pat, PatRest):
            return ".."
        return "_"

    def _pattern_bound_to_rust(self, bound: Expr) -> str:
        if isinstance(bound, Unary) and bound.op == "-":
            return f"-{self._pattern_bound_to_rust(bound.operand)}"
        raw = getattr(bound, "raw", None)
        if raw is not None:
            return raw
        if isinstance(bound, LitChar):
            return f"'{bound.value}'"
        if isinstance(bound, LitBool):
            return "true" if bound.value else "false"
        return self.expr_to_rust(bound)
