from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from compiler.core.ast_nodes import (
    ArrayExpr,
    Assign,
    Block,
    Call,
    ExprStmt,
    FnDecl,
    For,
    Ident,
    IfExpr,
    ImplBlock,
    Index,
    LetStmt,
    LitStr,
    Loop,
    MethodCall,
    Node,
    OptionSome,
    Paren,
    PatBinding,
    PatIdent,
    PatRef,
    Program,
    RepeatArray,
    ResultErr,
    ResultOk,
    StructLiteral,
    TupleExpr,
    TypeName,
    TypeRef,
    TypeTuple,
    TypeUnit,
    Unary,
    While,
)
from backends.move_check import MoveError

_COPY_TYPES = frozenset(
    {
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
        "isize",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
        "()",
    }
)


class BorrowError(MoveError):
    """A borrow/drop violation, reported through the shared ownership surface.

    Ownership is one language rule, so this reuses ``E009`` (SPEC 8.16) and
    carries the same fields as ``backends.move_check.MoveError``: ``name``,
    ``moved_at`` (the line that moved or dropped the value), ``line``/``col``
    of the use, and ``kind`` in {"drop", "double-drop", "moved", "borrow"}.
    """

    code = "E009"

    def __init__(
        self,
        fn_name: str,
        reason: str,
        help_text: str,
        line: int,
        col: int,
        span: tuple[int, int] | None = None,
        *,
        kind: str = "borrow",
        name: str = "",
        moved_at: int = 0,
    ) -> None:
        self.fn_name, self.reason, self.help_text, self.line, self.col, self.span = (
            fn_name,
            reason,
            help_text,
            line,
            col,
            span,
        )
        self.kind = kind
        self.name = name
        self.moved_at = moved_at
        Exception.__init__(self, self.format())

    def format(self) -> str:
        at = f"at: line {self.line}, column {self.col}" + (
            f" (span {self.span[0]}..{self.span[1]})" if self.span else ""
        )
        note = (
            f"ownership and borrow rules enforced in `{self.fn_name}` before lowering"
        )
        return (
            f"error[{self.code}]: {self.reason}\n"
            f"  reason: {self.reason}\n"
            f"  help: {self.help_text}\n"
            f"  note: {note}\n"
            f"  {at}"
        )

    def __str__(self) -> str:
        return self.format()


def _is_copy(ty: Any) -> bool:
    if ty is None or isinstance(ty, (TypeUnit, TypeRef)):
        return True
    if isinstance(ty, TypeTuple):
        return all(_is_copy(e) for e in ty.elems)
    return isinstance(ty, TypeName) and ty.name in _COPY_TYPES


def _is_owned(expr: Any, owned: set[str], ret_types: dict[str, Any]) -> bool:
    """Does this expression definitely produce a value that moves?

    `Copy` values (a `u64` from `_wadd`) never move, so calling them owned would
    report a move on every arithmetic helper -- the first full-suite run caught
    exactly that in `stdlib/random.poly`. A call is owned only when its callee's
    declared return type is known and non-`Copy`; an unknown callee is treated
    as non-owning, which loses a diagnostic rather than inventing one.
    """
    if isinstance(expr, Paren):
        return _is_owned(expr.inner, owned, ret_types)
    if isinstance(expr, Ident):
        return expr.name in owned
    if isinstance(
        expr,
        (
            LitStr,
            ArrayExpr,
            RepeatArray,
            StructLiteral,
            OptionSome,
            ResultOk,
            ResultErr,
        ),
    ):
        return True
    if isinstance(expr, TupleExpr):
        return any(_is_owned(e, owned, ret_types) for e in expr.elems)
    if isinstance(expr, Call):
        if isinstance(expr.func, Ident):
            if expr.func.name == "drop":
                return False
            if expr.func.name in ret_types:
                return not _is_copy(ret_types[expr.func.name])
        return False
    return isinstance(expr, MethodCall) and expr.method in {
        "to_string",
        "to_vec",
        "clone",
        "new",
    }


def _last_use_lines(node: Any, out: dict[str, int]) -> None:
    """Record the last line each name is read on, for bounded NLL.

    A borrow is dead once its holder is never read again, so a move or drop
    after that point is legal (rustc accepts it). Only the last use inside this
    function body is modelled; borrows that escape into a returned closure or a
    field are outside this pass (SPEC 8.16).
    """
    if isinstance(node, (list, tuple)):
        for item in node:
            _last_use_lines(item, out)
        return
    if not is_dataclass(node) or isinstance(node, type):
        return
    if isinstance(node, Ident):
        line = getattr(node, "line", 0) or 0
        out[node.name] = max(out.get(node.name, 0), line)
    for f in fields(node):
        if f.name in {"line", "col", "span"}:
            continue
        _last_use_lines(getattr(node, f.name), out)


def check_function_borrows(fn: FnDecl, ret_types: dict[str, Any] | None = None) -> None:
    if fn.body is None:
        return
    ret_types = ret_types or {}
    states: dict[str, str] = {}
    state_at: dict[str, int] = {}
    owned: set[str] = set()
    borrows: dict[str, list[str]] = {}
    holder_of: dict[str, str] = {}
    last_use: dict[str, int] = {}
    _last_use_lines(fn.body, last_use)
    for p in fn.params:
        if isinstance(p.pattern, PatRef):
            continue
        n = p.pattern.name if isinstance(p.pattern, (PatIdent, PatBinding)) else None
        if n and n != "self":
            states[n] = "alive"
            if p.ty is not None and not _is_copy(p.ty):
                owned.add(n)

    def fail(
        r: str,
        h: str,
        node: Node,
        *,
        kind: str = "borrow",
        name: str = "",
        moved_at: int = 0,
    ) -> None:
        raise BorrowError(
            fn.name,
            r,
            h,
            node.line,
            node.col,
            getattr(node, "span", None),
            kind=kind,
            name=name,
            moved_at=moved_at,
        )

    def live_borrows(n: str, at: int) -> list[str]:
        return [h for h in borrows.get(n, ()) if last_use.get(h, 0) > at]

    def release(h: str) -> None:
        t = holder_of.pop(h, None)
        if t and t in borrows:
            if h in borrows[t]:
                borrows[t].remove(h)
            if not borrows[t]:
                del borrows[t]

    def readable(n: str, node: Node) -> None:
        state = states.get(n)
        if state == "dropped":
            fail(
                f"use of dropped value: `{n}`",
                f"do not access `{n}` after `drop({n})`",
                node,
                kind="drop",
                name=n,
                moved_at=state_at.get(n, 0),
            )
        if state == "moved":
            fail(
                f"use of moved value: `{n}`",
                f"clone `{n}` or borrow `&{n}`",
                node,
                kind="moved",
                name=n,
                moved_at=state_at.get(n, 0),
            )

    def consume(n: str, act: str, node: Node) -> None:
        if act == "drop" and states.get(n) == "dropped":
            fail(
                f"cannot drop `{n}` twice",
                f"`{n}` was already released by `drop({n})` at line "
                f"{state_at.get(n, 0)}",
                node,
                kind="double-drop",
                name=n,
                moved_at=state_at.get(n, 0),
            )
        readable(n, node)
        live = live_borrows(n, node.line)
        if live:
            fail(
                f"cannot {act} `{n}` because it is borrowed by `{live[0]}`",
                "stop borrowing it, or end the borrow before moving",
                node,
                kind="borrow",
                name=n,
                moved_at=node.line,
            )
        if n in owned:
            states[n] = "dropped" if act == "drop" else "moved"
            state_at[n] = node.line

    def v_expr(e: Any) -> None:
        if not isinstance(e, Node):
            return
        if isinstance(e, Paren):
            return v_expr(e.inner)
        if isinstance(e, Ident):
            return readable(e.name, e)
        if isinstance(e, Unary):
            if e.op in {"&", "&mut"} and isinstance(e.operand, Ident):
                return readable(e.operand.name, e)
            return v_expr(e.operand)
        if isinstance(e, Call):
            if isinstance(e.func, Ident) and e.func.name == "drop":
                if len(e.args) != 1:
                    fail("`drop` expects 1 argument", "call `drop(x)`", e)
                a = e.args[0]
                if isinstance(a, Unary) and a.op in {"&", "&mut"}:
                    return v_expr(a)
                if isinstance(a, Ident):
                    return consume(a.name, "drop", e)
                return v_expr(a)
            v_expr(e.func)
            for a in e.args:
                if isinstance(a, Unary) and a.op in {"&", "&mut"}:
                    v_expr(a)
                elif isinstance(a, Ident) and a.name in owned:
                    consume(a.name, "move", a)
                else:
                    v_expr(a)
            return
        if isinstance(e, MethodCall):
            readable(e.receiver.name, e.receiver) if isinstance(
                e.receiver, Ident
            ) else v_expr(e.receiver)
            for a in e.args:
                if isinstance(a, Unary) and a.op in {"&", "&mut"}:
                    v_expr(a)
                elif isinstance(a, Ident) and a.name in owned:
                    consume(a.name, "move", a)
                else:
                    v_expr(a)
            return
        if isinstance(e, Index):
            readable(e.receiver.name, e.receiver) if isinstance(
                e.receiver, Ident
            ) else v_expr(e.receiver)
            return v_expr(e.index)
        if isinstance(e, Assign):
            v_expr(e.value)
            if isinstance(e.target, Ident):
                n = e.target.name
                if live_borrows(n, e.line):
                    fail(
                        f"cannot assign to `{n}` while borrowed by "
                        f"`{live_borrows(n, e.line)[0]}`",
                        "release borrow first",
                        e,
                    )
                release(n)
                if n in states:
                    states[n] = "alive"
                    if _is_owned(e.value, owned, ret_types):
                        owned.add(n)
                    else:
                        owned.discard(n)
            else:
                v_expr(e.target)
            return
        if isinstance(e, IfExpr):
            v_expr(e.cond)
            base = dict(states)
            v_blk(e.then_block)
            ts = dict(states)
            states.clear()
            states.update(base)
            if isinstance(e.else_branch, Block):
                v_blk(e.else_branch)
            elif e.else_branch is not None:
                v_expr(e.else_branch)
            es = dict(states)
            for k in base:
                states[k] = (
                    "dropped"
                    if "dropped" in (ts.get(k), es.get(k))
                    else ("moved" if "moved" in (ts.get(k), es.get(k)) else "alive")
                )
            return
        if isinstance(e, (While, For, Loop)):
            if isinstance(e, While):
                v_expr(e.cond)
            elif isinstance(e, For):
                v_expr(e.iter)
            # A loop needs a header fixpoint, not a single pass: a body that
            # reassigns a value before moving it is fine, and only the second
            # iteration can tell. Run the body once from the entry state, join
            # the two states at the top of the loop, then run it again -- any
            # violation the second pass reports is real on every later
            # iteration.
            pre = dict(states)
            v_blk(e.body)
            tail = dict(states)
            header: dict[str, str] = {}
            for k, value in pre.items():
                a, b = value, tail.get(k, value)
                header[k] = (
                    "dropped"
                    if "dropped" in (a, b)
                    else ("moved" if "moved" in (a, b) else "alive")
                )
            states.clear()
            states.update(header)
            v_blk(e.body)
            return
        if isinstance(e, Block):
            return v_blk(e)
        if is_dataclass(e):
            for f in fields(e):
                if f.name not in {"line", "col", "span"}:
                    val = getattr(e, f.name)
                    for item in val if isinstance(val, list) else [val]:
                        v_expr(item)

    def v_blk(blk: Block) -> None:
        before = set(states)
        for s in blk.stmts:
            if isinstance(s, LetStmt):
                n = (
                    s.pattern.name
                    if isinstance(s.pattern, (PatIdent, PatBinding))
                    else None
                )
                v = s.value
                if (
                    n
                    and isinstance(v, Unary)
                    and v.op in {"&", "&mut"}
                    and isinstance(v.operand, Ident)
                ):
                    readable(v.operand.name, v)
                    release(n)
                    holder_of[n] = v.operand.name
                    borrows.setdefault(v.operand.name, []).append(n)
                    states[n] = "alive"
                    continue
                if isinstance(v, Ident) and v.name in owned:
                    consume(v.name, "move", v)
                    if n:
                        states[n] = "alive"
                        owned.add(n)
                    continue
                v_expr(v)
                if n:
                    release(n)
                    states[n] = "alive"
                    if not _is_copy(s.ty) and (
                        s.ty is not None or _is_owned(v, owned, ret_types)
                    ):
                        owned.add(n)
                    else:
                        owned.discard(n)
            elif isinstance(s, ExprStmt):
                v_expr(s.expr)
        if blk.tail is not None:
            v_expr(blk.tail)
        for k in list(states):
            if k not in before:
                release(k)

    v_blk(fn.body)


def check_program_borrows(program: Program) -> None:
    ret_types: dict[str, Any] = {}
    for it in program.items:
        if isinstance(it, FnDecl):
            ret_types[it.name] = it.ret
        elif isinstance(it, ImplBlock):
            for m in it.items:
                if isinstance(m, FnDecl):
                    ret_types[m.name] = m.ret
    for it in program.items:
        if isinstance(it, FnDecl):
            check_function_borrows(it, ret_types)
        elif isinstance(it, ImplBlock):
            for m in it.items:
                if isinstance(m, FnDecl):
                    check_function_borrows(m, ret_types)
