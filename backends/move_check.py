"""Shared, bounded move checking (E009).

Polyglot follows Rust ownership for owned values: `String`, `Vec<T>`,
`HashMap`, user structs and enums move when used by value, while integers,
floats, `bool`, `char`, references and tuples of those are copied. The Python
backend represents those values as ordinary objects, so without a shared
check the same program can run there and fail on the native backend with
rustc's "use of moved value" — which breaks the parity contract the project
promises.

This pass runs on the shared front end, before either backend generates code,
and reports one deterministic `E009` with the binding name and both lines.

Scope is deliberately bounded (SPEC 8.16): the checker models direct local
bindings whose type it can name, follows straight-line blocks, `if`/`else`,
`if let`, `match` arms and loop bodies, and reports only what it can prove.
Unknown types, method semantics it does not model, and moves through
references are left to the backend instead of being guessed at, so the pass
never invents an error that rustc would accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

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
    Continue,
    DeferStmt,
    EnumDecl,
    ErrDeferStmt,
    Expr,
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
    ModDecl,
    OptionNone,
    OptionSome,
    Paren,
    PatBinding,
    PatEnum,
    Path,
    PatIdent,
    PatOr,
    PatRef,
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
    StructDecl,
    StructLiteral,
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
    While,
)

CODE = "E009"

# Named types whose values are copied rather than moved.
COPY_SCALARS = {
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "isize",
    "f32",
    "f64",
    "bool",
    "char",
}

# Generic containers that are Copy exactly when their arguments are.
COPY_WHEN_ARGS_COPY = {"Option", "Result"}

# Methods whose argument at the given position is taken by value.
CONSUMING_METHOD_ARGS = {
    "push": (0,),
    "push_back": (0,),
    "push_front": (0,),
    "insert": (1,),
    "or_insert": (0,),
}

# Methods that consume the receiver itself.
CONSUMING_RECEIVER_METHODS = {"into_iter", "into_bytes"}

# Macros whose arguments are borrowed rather than moved.
BORROWING_MACROS = {
    "println",
    "print",
    "eprintln",
    "eprint",
    "format",
    "dbg",
    "panic",
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
}


class MoveError(Exception):
    """A use of a value that was moved (or partially moved) earlier."""

    code = CODE

    def __init__(
        self,
        name: str,
        moved_at: int,
        line: int,
        col: int = 0,
        span: tuple[int, int] | None = None,
        kind: str = "moved",
    ):
        self.name = name
        self.moved_at = moved_at
        self.line = line
        self.col = col
        self.span = span
        self.kind = kind
        super().__init__(self.format())

    def format(self) -> str:
        if self.kind == "index":
            headline = f"cannot move out of index of `{self.name}`"
            reason = (
                "indexing returns a place, not a value; the element is owned "
                "(String, Vec or a struct), so reading it by value would take "
                "it out of the container"
            )
            help_text = (
                f"call `.clone()` on the element, or take a reference "
                f"(`&{self.name}[i]`)"
            )
        elif self.kind == "field":
            headline = f"use of partially moved value: `{self.name}`"
            reason = (
                f"a field of `{self.name}` was moved at line {self.moved_at}, "
                "so the binding is no longer a complete value"
            )
            help_text = (
                "move the whole struct once, or clone the field you need "
                "instead of moving it out"
            )
        else:
            headline = f"use of moved value: `{self.name}`"
            reason = (
                f"`{self.name}` was moved at line {self.moved_at} and is used "
                f"again at line {self.line}"
            )
            help_text = (
                "call `.clone()` before one of the two uses, pass a reference "
                f"(`&{self.name}`), or restructure so the value is moved once"
            )
        at = f"at: line {self.line}, column {self.col}"
        if self.span is not None:
            at += f" (span {self.span[0]}..{self.span[1]})"
        return (
            f"error[{CODE}]: {headline}\n"
            f"  reason: {reason}\n"
            f"  help: {help_text}\n"
            f"  note: MoveError\n"
            f"  {at}"
        )


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _split_generic(text: str) -> tuple[str, list[str]]:
    open_at = text.find("<")
    if open_at < 0 or not text.endswith(">"):
        return text, []
    return text[:open_at], _split_top_level(text[open_at + 1 : -1])


def render_type(ty: Type | None, aliases: dict[str, str] | None = None) -> str | None:
    """Render an AST type as a stable string, or None when it is not modelled."""
    if ty is None:
        return None
    if isinstance(ty, TypeName):
        if aliases and ty.name in aliases:
            return aliases[ty.name]
        if not ty.args:
            return ty.name
        args = [render_type(arg, aliases) or "?" for arg in ty.args]
        return f"{ty.name}<{', '.join(args)}>"
    if isinstance(ty, TypeRef):
        inner = render_type(ty.inner, aliases) or "?"
        return f"&mut {inner}" if ty.mutable else f"&{inner}"
    if isinstance(ty, TypeTuple):
        elems = [render_type(elem, aliases) or "?" for elem in ty.elems]
        return f"({', '.join(elems)})" if elems else "()"
    if isinstance(ty, TypeUnit):
        return "()"
    if isinstance(ty, TypeSlice):
        return f"[{render_type(ty.inner, aliases) or '?'}]"
    if isinstance(ty, TypeArray):
        return f"[{render_type(ty.elem, aliases) or '?'}]"
    if isinstance(ty, (TypeFn, TypeImplTrait, TypeDynTrait)):
        return None
    return None


def is_copy(text: str | None) -> bool:
    """True when a value of this type is copied on use.

    Unknown types count as copy: the checker stays quiet rather than inventing
    an error the backend would not produce.
    """
    if text is None:
        return True
    text = text.strip()
    if not text:
        return True
    if text in {"()", "!"}:
        return True
    if text.startswith("&"):
        return True
    if text.startswith("(") and text.endswith(")"):
        return all(is_copy(part) for part in _split_top_level(text[1:-1]))
    base, args = _split_generic(text)
    if base in COPY_SCALARS:
        return True
    if base in COPY_WHEN_ARGS_COPY:
        return all(is_copy(arg) for arg in args)
    return False


def _strip_ref(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if text.startswith("&mut "):
        return text[len("&mut ") :]
    if text.startswith("&"):
        return text[1:]
    return text


def _type_base(text: str | None) -> str | None:
    stripped = _strip_ref(text)
    if stripped is None:
        return None
    return _split_generic(stripped)[0]


def _type_args(text: str | None) -> list[str]:
    stripped = _strip_ref(text)
    if stripped is None:
        return []
    return _split_generic(stripped)[1]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _Local:
    ty: str | None
    moved_at: int | None = None
    moved_fields: set[str] = field(default_factory=set)
    partial_at: int | None = None

    def clone(self) -> _Local:
        return _Local(self.ty, self.moved_at, set(self.moved_fields), self.partial_at)


@dataclass
class _Sig:
    params: list[str | None]
    ret: str | None


class MoveChecker:
    """Walk a program and raise `MoveError` on the first provable violation."""

    def __init__(self, program: Program):
        self.locals: dict[str, _Local] = {}
        self.functions: dict[str, _Sig] = {}
        self.methods: dict[tuple[str, str], _Sig] = {}
        self.structs: dict[str, dict[str, str | None]] = {}
        self.enums: dict[str, list[list[str | None]]] = {}
        self.aliases: dict[str, str] = {}
        self._collect_items(program.items)

    # -- signature collection ------------------------------------------------

    def _collect_items(self, items) -> None:
        for item in items:
            if isinstance(item, FnDecl):
                self.functions[item.name] = _Sig(
                    params=[
                        render_type(param.ty, self.aliases) for param in item.params
                    ],
                    ret=render_type(item.ret, self.aliases),
                )
            elif isinstance(item, ImplBlock):
                self._collect_impl(item)
            elif isinstance(item, StructDecl):
                self.structs[item.name] = {
                    name: render_type(ty, self.aliases) for name, ty, _ in item.fields
                }
            elif isinstance(item, TypeAlias):
                rendered = render_type(item.ty, self.aliases)
                if rendered is not None:
                    self.aliases[item.name] = rendered
            elif isinstance(item, EnumDecl):
                self.enums[item.name] = [
                    [render_type(ty, self.aliases) for ty in (variant.fields or [])]
                    for variant in item.variants
                ]
            elif isinstance(item, ModDecl) and item.items:
                self._collect_items(item.items)

    def _collect_impl(self, block: ImplBlock) -> None:
        owner = _type_base(render_type(block.ty, self.aliases))
        if owner is None:
            return
        for item in block.items:
            if not isinstance(item, FnDecl):
                continue
            params = list(item.params)
            if (
                params
                and isinstance(params[0].pattern, PatIdent)
                and params[0].pattern.name == "self"
            ):
                params = params[1:]
            self.methods[(owner, item.name)] = _Sig(
                params=[render_type(param.ty, self.aliases) for param in params],
                ret=render_type(item.ret, self.aliases),
            )

    # -- entry --------------------------------------------------------------

    def check(self, program: Program) -> None:
        for item in program.items:
            self._check_item(item)

    def _check_item(self, item) -> None:
        if isinstance(item, FnDecl):
            if item.body is None:
                return
            saved = self.locals
            self.locals = {}
            for param in item.params:
                self._bind_pattern(
                    param.pattern, render_type(param.ty, self.aliases), param.line
                )
            self._block(item.body)
            self.locals = saved
        elif isinstance(item, ImplBlock) or isinstance(item, ModDecl) and item.items:
            for inner in item.items:
                self._check_item(inner)

    # -- bindings -----------------------------------------------------------

    def _bind_pattern(self, pattern: Pattern, ty: str | None, line: int) -> None:
        if isinstance(pattern, (PatIdent, PatBinding)):
            if isinstance(pattern, PatBinding):
                self._bind_pattern(pattern.inner, ty, line)
            self.locals[pattern.name] = _Local(None if is_copy(ty) else ty)
        elif isinstance(pattern, PatTuple):
            parts = _split_top_level(ty[1:-1]) if ty and ty.startswith("(") else []
            for position, elem in enumerate(pattern.elems):
                part = parts[position] if position < len(parts) else None
                self._bind_pattern(elem, part, line)
        elif isinstance(pattern, PatStruct):
            fields = self.structs.get(_type_base(ty) or "") or {}
            for name, inner in pattern.fields:
                self._bind_pattern(inner, fields.get(name), line)
        elif isinstance(pattern, PatEnum):
            for variant in self.enums.get(_type_base(ty) or "") or []:
                if len(variant) != len(pattern.args):
                    continue
                for position, inner in enumerate(pattern.args):
                    self._bind_pattern(inner, variant[position], line)
                break
        elif isinstance(pattern, PatOr):
            for alt in pattern.alts:
                self._bind_pattern(alt, ty, line)
        elif isinstance(pattern, PatRef):
            self._bind_pattern(pattern.inner, ty, line)
        elif isinstance(pattern, PatWildcard):
            return

    # -- reads and moves ----------------------------------------------------

    def _read(self, node: Ident, whole: bool = True) -> str | None:
        info = self.locals.get(node.name)
        if info is None:
            return None
        if info.moved_at is not None:
            raise MoveError(
                node.name, info.moved_at, node.line, node.col, span=node.span
            )
        if whole and info.moved_fields:
            raise MoveError(
                node.name,
                info.partial_at or node.line,
                node.line,
                node.col,
                span=node.span,
                kind="field",
            )
        return info.ty

    def _move(self, node: Ident) -> None:
        info = self.locals.get(node.name)
        if info is None or is_copy(info.ty):
            return
        if info.moved_at is not None or info.moved_fields:
            raise MoveError(
                node.name,
                info.moved_at or info.partial_at or node.line,
                node.line,
                node.col,
                span=node.span,
            )
        info.moved_at = node.line

    # -- statements --------------------------------------------------------

    def _block(self, block: Block) -> str | None:
        outer = set(self.locals)
        for stmt in block.stmts:
            self._stmt(stmt)
            if _stmt_terminates(stmt):
                # Statements after `return`/`break`/`continue` are unreachable.
                for name in list(self.locals):
                    if name not in outer:
                        del self.locals[name]
                return None
        tail_ty = self._eval(block.tail) if block.tail is not None else None
        for name in list(self.locals):
            if name not in outer:
                del self.locals[name]
        return tail_ty

    def _stmt(self, stmt) -> None:
        if isinstance(stmt, LetStmt):
            value_ty = self._eval(stmt.value, consume=True) if stmt.value else None
            annotated = render_type(stmt.ty, self.aliases)
            self._bind_pattern(stmt.pattern, annotated or value_ty, stmt.line)
        elif isinstance(stmt, ExprStmt):
            self._eval(stmt.expr)
        elif isinstance(stmt, (DeferStmt, ErrDeferStmt)):
            if isinstance(stmt.body, Block):
                self._block(stmt.body)
            else:
                self._eval(stmt.body)
        elif isinstance(stmt, ItemStmt):
            return

    # -- expressions -------------------------------------------------------

    def _eval(self, expr: Expr | None, consume: bool = False) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, LitStr):
            return "String"
        if isinstance(expr, LitInt):
            return "i64"
        if isinstance(expr, LitFloat):
            return "f64"
        if isinstance(expr, LitBool):
            return "bool"
        if isinstance(expr, LitChar):
            return "char"
        if isinstance(expr, Ident):
            ty = self._read(expr)
            if consume:
                self._move(expr)
            return ty
        if isinstance(expr, Paren):
            return self._eval(expr.inner, consume)
        if isinstance(expr, Path):
            return None
        if isinstance(expr, Unary):
            if expr.op in {"&", "&mut"}:
                inner = self._eval(expr.operand)
                return f"&{inner or '?'}"
            return self._eval(expr.operand)
        if isinstance(expr, Binary):
            left = self._eval(expr.left)
            right = self._eval(expr.right)
            if expr.op == "+" and _type_base(left) == "String":
                return "String"
            if expr.op in {"==", "!=", "<", ">", ">=", "&&", "||"}:
                return "bool"
            return right
        if isinstance(expr, Assign):
            value_ty = self._eval(expr.value, consume=True)
            self._assign_target(expr.target, value_ty)
            return None
        if isinstance(expr, CompoundAssign):
            self._eval(expr.target)
            self._eval(expr.value)
            return None
        if isinstance(expr, Call):
            return self._call(expr)
        if isinstance(expr, MethodCall):
            return self._method_call(expr)
        if isinstance(expr, Field):
            receiver_ty = self._eval(expr.receiver)
            return (self.structs.get(_type_base(receiver_ty) or "") or {}).get(
                expr.name
            )
        if isinstance(expr, Index):
            receiver_ty = self._eval(expr.receiver)
            self._eval(expr.index)
            element = self._element_type(receiver_ty)
            if consume and not is_copy(element):
                name = (
                    expr.receiver.name
                    if isinstance(expr.receiver, Ident)
                    else "this place"
                )
                raise MoveError(
                    name, expr.line, expr.line, expr.col, span=expr.span, kind="index"
                )
            return element
        if isinstance(expr, TupleExpr):
            types = [self._eval(elem, consume=True) for elem in expr.elems]
            return "(" + ", ".join(ty or "?" for ty in types) + ")" if types else "()"
        if isinstance(expr, ArrayExpr):
            types = [self._eval(elem, consume=True) for elem in expr.elems]
            return f"[{types[0] if types else '?'}]"
        if isinstance(expr, RepeatArray):
            elem_ty = self._eval(expr.elem)
            self._eval(expr.count)
            return f"[{elem_ty or '?'}]"
        if isinstance(expr, StructLiteral):
            for name, value in expr.fields:
                self._eval(value, consume=True)
            if expr.rest is not None:
                self._eval(expr.rest, consume=True)
            return expr.path
        if isinstance(expr, IfExpr):
            return self._branch(expr.cond, expr.then_block, expr.else_branch)
        if isinstance(expr, IfLet):
            value_ty = self._eval(expr.value)
            return self._if_let(expr, value_ty)
        if isinstance(expr, MatchExpr):
            return self._match(expr)
        if isinstance(expr, Block):
            return self._block(expr)
        if isinstance(expr, Loop):
            self._loop_body(expr.body)
            return None
        if isinstance(expr, While):
            self._eval(expr.cond)
            self._loop_body(expr.body)
            return None
        if isinstance(expr, For):
            self._for(expr)
            return None
        if isinstance(expr, Closure):
            saved = self.locals
            self.locals = dict(saved)
            for pattern, ty in expr.params:
                self._bind_pattern(pattern, render_type(ty, self.aliases), expr.line)
            self._eval(expr.body)
            self.locals = saved
            return None
        if isinstance(expr, Range):
            self._eval(expr.start)
            self._eval(expr.end)
            return None
        if isinstance(expr, Cast):
            self._eval(expr.expr)
            return render_type(expr.ty, self.aliases)
        if isinstance(expr, Return):
            self._eval(expr.value, consume=True)
            return None
        if isinstance(expr, Break):
            self._eval(expr.value, consume=True)
            return None
        if isinstance(expr, Continue):
            return None
        if isinstance(expr, MacroCall):
            return self._macro(expr)
        if isinstance(expr, Await):
            return self._eval(expr.expr)
        if isinstance(expr, Try):
            inner = self._eval(expr.expr, consume=True)
            args = _type_args(inner)
            return args[0] if args else None
        if isinstance(expr, OptionSome):
            value_ty = self._eval(expr.value, consume=True)
            return f"Option<{value_ty or '?'}>"
        if isinstance(expr, OptionNone):
            return "Option<?>"
        if isinstance(expr, ResultOk):
            value_ty = self._eval(expr.value, consume=True)
            return f"Result<{value_ty or '?'}, ?>"
        if isinstance(expr, ResultErr):
            value_ty = self._eval(expr.value, consume=True)
            return f"Result<?, {value_ty or '?'}>"
        return None

    def _assign_target(self, target: Expr, value_ty: str | None) -> None:
        if isinstance(target, Paren):
            self._assign_target(target.inner, value_ty)
            return
        if isinstance(target, Ident):
            info = self.locals.get(target.name)
            if info is None:
                return
            info.moved_at = None
            info.moved_fields.clear()
            info.partial_at = None
            if value_ty is not None:
                info.ty = None if is_copy(value_ty) else value_ty
            return
        if isinstance(target, Index):
            self._eval(target.receiver)
            self._eval(target.index)
            return
        if isinstance(target, Field):
            self._eval(target.receiver)
            return

    def _call(self, expr: Call) -> str | None:
        callee = expr.func
        while isinstance(callee, Paren):
            callee = callee.inner
        if isinstance(callee, Ident):
            sig = self.functions.get(callee.name)
            return self._apply(sig, expr.args)
        if isinstance(callee, Path):
            segments = callee.segments
            if (
                len(segments) > 1
                and segments[-2] == "String"
                and segments[-1]
                in {
                    "from",
                    "new",
                }
            ):
                for arg in expr.args:
                    self._eval(arg, consume=True)
                return "String"
            sig = self.functions.get(segments[-1])
            if sig is None and len(segments) > 1:
                sig = self.methods.get((segments[-2], segments[-1]))
            if sig is None:
                for arg in expr.args:
                    self._eval(arg)
                return None
            return self._apply(sig, expr.args)
        self._eval(callee)
        for arg in expr.args:
            self._eval(arg)
        return None

    def _apply(self, sig: _Sig | None, args: list[Expr]) -> str | None:
        for position, arg in enumerate(args):
            param_ty = (
                sig.params[position] if sig and position < len(sig.params) else None
            )
            if sig is None or is_copy(param_ty):
                self._eval(arg)
            else:
                self._eval(arg, consume=True)
        return sig.ret if sig else None

    def _method_call(self, expr: MethodCall) -> str | None:
        receiver_ty = self._eval(expr.receiver)
        if expr.method in CONSUMING_RECEIVER_METHODS and isinstance(
            expr.receiver, Ident
        ):
            self._move(expr.receiver)
        sig = self.methods.get((_type_base(receiver_ty) or "", expr.method))
        if sig is not None:
            self._apply(sig, expr.args)
            return sig.ret
        positions = CONSUMING_METHOD_ARGS.get(expr.method, ())
        for position, arg in enumerate(expr.args):
            self._eval(arg, consume=position in positions)
        if expr.method in {"clone", "to_string", "to_owned"}:
            return receiver_ty
        if expr.method in {
            "to_uppercase",
            "to_lowercase",
            "trim",
            "trim_start",
            "trim_end",
            "replace",
            "join",
            "to_owned_string",
        }:
            return "String"
        if expr.method in {"len", "count"}:
            return "usize"
        if expr.method in {"pop", "get", "top_key", "first", "last"}:
            return f"Option<{self._element_type(receiver_ty) or '?'}>"
        return None

    def _macro(self, expr: MacroCall) -> str | None:
        name = expr.path.split("::")[-1]
        if isinstance(expr.args, str):
            return None
        if name == "vec":
            elem_ty = None
            for arg in expr.args:
                elem_ty = elem_ty or self._eval(arg)
            if len(expr.args) > 1:
                for arg in expr.args:
                    self._eval(arg, consume=True)
            return f"Vec<{elem_ty or '?'}>"
        borrow = name in BORROWING_MACROS
        for arg in expr.args:
            self._eval(arg, consume=not borrow)
        return "String" if name == "format" else None

    @staticmethod
    def _element_type(container: str | None) -> str | None:
        stripped = _strip_ref(container)
        if stripped is None:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped[1:-1]
        args = _type_args(container)
        return args[0] if args else None

    # -- control flow ------------------------------------------------------

    def _branch(self, cond, then_block: Block, else_branch) -> str | None:
        self._eval(cond)
        entry = {name: local.clone() for name, local in self.locals.items()}
        then_ty = self._block(then_block)
        if terminates_block(then_block):
            then_state = entry
        else:
            then_state = {name: local.clone() for name, local in self.locals.items()}
        self.locals = {name: local.clone() for name, local in entry.items()}
        else_ty = None
        else_diverges = False
        if else_branch is not None:
            if isinstance(else_branch, Block):
                else_ty = self._block(else_branch)
                else_diverges = terminates_block(else_branch)
            else:
                else_ty = self._eval(else_branch, consume=False)
                else_diverges = terminates_expr(else_branch)
        if else_diverges:
            else_state = entry
        else:
            else_state = {name: local.clone() for name, local in self.locals.items()}
        self.locals = _merge(entry, then_state, else_state)
        return then_ty if then_ty == else_ty else None

    def _if_let(self, expr: IfLet, value_ty: str | None) -> str | None:
        entry = {name: local.clone() for name, local in self.locals.items()}
        self._bind_pattern(expr.pattern, value_ty, expr.line)
        then_ty = self._block(expr.then_block)
        if terminates_block(expr.then_block):
            then_state = entry
        else:
            then_state = {name: local.clone() for name, local in self.locals.items()}
        self.locals = {name: local.clone() for name, local in entry.items()}
        else_ty = None
        else_diverges = False
        if expr.else_branch is not None:
            if isinstance(expr.else_branch, Block):
                else_ty = self._block(expr.else_branch)
                else_diverges = terminates_block(expr.else_branch)
            else:
                else_ty = self._eval(expr.else_branch)
                else_diverges = terminates_expr(expr.else_branch)
        if else_diverges:
            else_state = entry
        else:
            else_state = {name: local.clone() for name, local in self.locals.items()}
        self.locals = _merge(entry, then_state, else_state)
        return then_ty if then_ty == else_ty else None

    def _match(self, expr: MatchExpr) -> str | None:
        self._eval(expr.scrutinee)
        entry = {name: local.clone() for name, local in self.locals.items()}
        states = []
        types = []
        for arm in expr.arms:
            self.locals = {name: local.clone() for name, local in entry.items()}
            self._bind_pattern(arm.pattern, None, arm.line)
            if arm.guard is not None:
                self._eval(arm.guard)
            types.append(self._eval(arm.body))
            if terminates_expr(arm.body):
                states.append(entry)
            else:
                states.append(
                    {name: local.clone() for name, local in self.locals.items()}
                )
        if not states:
            return None
        merged = entry
        for state in states:
            merged = _merge(entry, merged, state)
        self.locals = merged
        return types[0] if types and all(ty == types[0] for ty in types) else None

    def _loop_body(self, body: Block) -> None:
        entry = {name: local.clone() for name, local in self.locals.items()}
        self._block(body)
        if terminates_block(body):
            self.locals = entry
            return
        after = {name: local.clone() for name, local in self.locals.items()}
        self.locals = _merge(entry, entry, after)

    def _for(self, expr: For) -> None:
        iter_ty = self._eval(expr.iter, consume=True)
        element = self._element_type(iter_ty)
        entry = {name: local.clone() for name, local in self.locals.items()}
        self._bind_pattern(expr.pattern, element, expr.line)
        self._block(expr.body)
        after = {name: local.clone() for name, local in self.locals.items()}
        for name in list(after):
            if name not in entry:
                del after[name]
        self.locals = _merge(entry, entry, after)


def _stmt_terminates(stmt) -> bool:
    if isinstance(stmt, ExprStmt):
        return terminates_expr(stmt.expr)
    return False


def terminates_block(block: Block) -> bool:
    """True when control never falls out of the block."""
    if not block.stmts:
        return False
    return _stmt_terminates(block.stmts[-1])


def terminates_expr(expr: Expr) -> bool:
    if isinstance(expr, (Return, Break, Continue)):
        return True
    if isinstance(expr, Block):
        return terminates_block(expr)
    if isinstance(expr, IfExpr):
        if expr.else_branch is None:
            return False
        if not terminates_block(expr.then_block):
            return False
        if isinstance(expr.else_branch, Block):
            return terminates_block(expr.else_branch)
        return terminates_expr(expr.else_branch)
    if isinstance(expr, IfLet):
        if expr.else_branch is None:
            return False
        if not terminates_block(expr.then_block):
            return False
        if isinstance(expr.else_branch, Block):
            return terminates_block(expr.else_branch)
        return terminates_expr(expr.else_branch)
    if isinstance(expr, MatchExpr):
        return bool(expr.arms) and all(terminates_expr(arm.body) for arm in expr.arms)
    if isinstance(expr, MacroCall):
        return expr.path.split("::")[-1] in {"panic", "unreachable", "todo"}
    return False


def _merge(
    base: dict[str, _Local],
    first: dict[str, _Local],
    second: dict[str, _Local],
) -> dict[str, _Local]:
    merged: dict[str, _Local] = {}
    for name, local in base.items():
        left = first.get(name, local)
        right = second.get(name, local)
        moved_at = left.moved_at
        if right.moved_at is not None:
            moved_at = (
                right.moved_at if moved_at is None else min(moved_at, right.moved_at)
            )
        partial_at = left.partial_at
        if right.partial_at is not None:
            partial_at = (
                right.partial_at
                if partial_at is None
                else min(partial_at, right.partial_at)
            )
        merged[name] = _Local(
            left.ty if left.ty == right.ty else None,
            moved_at,
            left.moved_fields | right.moved_fields,
            partial_at,
        )
    return merged


def _check_host_arity(node: Any) -> None:
    """Report a malformed host-seam call before ownership is considered.

    Arity is a signature error, so it has to win over a move diagnostic: the
    fuzz template `__fs_read_text(p, p)` passes `p` twice, which the borrow
    pass would happily call a use-after-move. `E003` is the real problem there,
    and the native backend has to agree with Python on the code, so every
    caller of `check_moves` runs this scan first.
    """
    from backends.errors import HOST_CALL_ARITIES, check_host_call_arity

    if isinstance(node, (list, tuple)):
        for item in node:
            _check_host_arity(item)
        return
    if not is_dataclass(node) or isinstance(node, type):
        return
    if isinstance(node, Call) and isinstance(node.func, Ident):
        name = node.func.name
        if name in HOST_CALL_ARITIES:
            check_host_call_arity(
                name,
                node.args,
                line=getattr(node, "line", 0),
                col=getattr(node, "col", 0),
                span=getattr(node, "span", None),
            )
    for f in fields(node):
        if f.name in {"line", "col", "span"}:
            continue
        _check_host_arity(getattr(node, f.name))


def check_moves(program: Program) -> None:
    """Raise `MoveError` on the first provable ownership violation.

    Two passes share this entry point so every backend reports the same `E009`.
    The borrow/drop pass (`backends/borrow_checker.py`) runs first because it
    knows whether the value is still borrowed: moving or dropping a borrowed
    value is a borrow conflict (rustc's E0505), not a plain use-after-move.
    The move checker then adds its partial-move and field diagnostics.
    """
    from backends.borrow_checker import check_program_borrows

    _check_host_arity(program)
    check_program_borrows(program)
    MoveChecker(program).check(program)
