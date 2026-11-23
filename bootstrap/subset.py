"""Lexer, parser, type checker, and emitters for a clean subset of the language.

The subset is intentionally tiny: integer functions with ``let``, ``if``/``else``,
``return``, calls, and five operators. It exists to be *self-hostable*: the IR it
produces is canonical enough that printing and re-reading it must reproduce the
same digest, which is what the bootstrap fixed-point check verifies.

Bounded model (SPEC 8.21, 10.51): stage 2 is the same Python implementation
driven from the printed IR, not a separate binary compiled by stage 1. What it
proves is IR canonicality and hash stability, not machine-code self-hosting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from runtime.diagnostics import PillarError

TOKEN = re.compile(
    r"\s*(->|==|=|<=|>=|\d+|[A-Za-z_][A-Za-z0-9_]*|\+|-|\*|/|<|>|\(|\)|\{|\}|,|;|:)"
)
KEYWORDS = {"fn", "let", "if", "else", "return", "i64", "bool"}

# -- AST -----------------------------------------------------------------


@dataclass(frozen=True)
class IntLit:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Applied:
    callee: str
    args: tuple[Expr, ...]


Expr = IntLit | Var | BinOp | Applied


@dataclass(frozen=True)
class Let:
    name: str
    value: Expr


@dataclass(frozen=True)
class Return:
    value: Expr


@dataclass(frozen=True)
class If:
    cond: Expr
    then: tuple[Stmt, ...]
    otherwise: tuple[Stmt, ...]


Stmt = Let | Return | If


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[str, ...]
    ret: str
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Program:
    functions: tuple[Function, ...]


# -- lexer ---------------------------------------------------------------
def tokenize(source: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(source):
        if source[position:].strip() == "":
            break
        match = TOKEN.match(source, position)
        if not match:
            raise PillarError(
                "E003",
                f"unexpected character at offset {position}: {source[position]!r}",
                help_text="the subset allows identifiers, integers, and + - * / < > == -> { } ( ) ; : ,",
            )
        tokens.append(match.group(1))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise PillarError("E003", "unexpected end of source", help_text="close the block or expression")
        if expected is not None and token != expected:
            raise PillarError(
                "E003",
                f"expected {expected!r}, found {token!r}",
                help_text="check the grammar: fn name(params) -> type { ... }",
            )
        self.position += 1
        return token

    def program(self) -> Program:
        functions: list[Function] = []
        while self.peek() is not None:
            functions.append(self.function())
        if not functions:
            raise PillarError("E003", "source defines no function")
        return Program(tuple(functions))

    def function(self) -> Function:
        self.take("fn")
        name = self.take()
        if name in KEYWORDS:
            raise PillarError("E003", f"cannot name a function {name!r}: it is a keyword")
        self.take("(")
        params: list[str] = []
        if self.peek() != ")":
            while True:
                param = self.take()
                if param in params:
                    raise PillarError("E003", f"duplicate parameter {param!r} in {name}")
                params.append(param)
                if self.peek() == ",":
                    self.take(",")
                    continue
                break
        self.take(")")
        self.take("->")
        ret = self.take()
        if ret not in ("i64", "bool"):
            raise PillarError("E001", f"unsupported return type {ret!r}", help_text="the subset has i64 and bool")
        self.take("{")
        body = self.block()
        return Function(name, tuple(params), ret, body)

    def block(self) -> tuple[Stmt, ...]:
        statements: list[Stmt] = []
        while self.peek() not in ("}", None):
            statements.append(self.statement())
        self.take("}")
        if not statements:
            raise PillarError("E003", "empty block")
        return tuple(statements)

    def statement(self) -> Stmt:
        token = self.peek()
        if token == "let":
            self.take("let")
            name = self.take()
            self.take("=")
            value = self.expr()
            self.take(";")
            return Let(name, value)
        if token == "return":
            self.take("return")
            value = self.expr()
            if self.peek() == ";":
                self.take(";")
            return Return(value)
        if token == "if":
            self.take("if")
            cond = self.expr()
            self.take("{")
            then = self.block()
            otherwise: tuple[Stmt, ...] = ()
            if self.peek() == "else":
                self.take("else")
                self.take("{")
                otherwise = self.block()
            return If(cond, then, otherwise)
        raise PillarError("E003", f"unknown statement starting with {token!r}")

    def expr(self) -> Expr:
        return self.comparison()

    def comparison(self) -> Expr:
        left = self.term()
        while self.peek() in ("==", "<", ">"):
            op = self.take()
            left = BinOp(op, left, self.term())
        return left

    def term(self) -> Expr:
        left = self.factor()
        while self.peek() in ("+", "-"):
            op = self.take()
            left = BinOp(op, left, self.factor())
        return left

    def factor(self) -> Expr:
        left = self.atom()
        while self.peek() in ("*", "/"):
            op = self.take()
            left = BinOp(op, left, self.atom())
        return left

    def atom(self) -> Expr:
        token = self.take()
        if token.isdigit():
            return IntLit(int(token))
        if token in ("i64", "bool") or token in ("(", ")", "{", "}", ";", ":"):
            raise PillarError("E003", f"expected an expression, found {token!r}")
        if self.peek() == "(":
            self.take("(")
            args: list[Expr] = []
            if self.peek() != ")":
                while True:
                    args.append(self.expr())
                    if self.peek() == ",":
                        self.take(",")
                        continue
                    break
            self.take(")")
            return Applied(token, tuple(args))
        return Var(token)


def parse_subset(source: str) -> Program:
    """Parse the subset, raising ``E003`` on any syntax error."""
    return _Parser(tokenize(source)).program()


# -- type checker -------------------------------------------------------
def _infer(expr: Expr, env: dict[str, str], signatures: dict[str, str]) -> str:
    if isinstance(expr, IntLit):
        return "i64"
    if isinstance(expr, Var):
        if expr.name not in env:
            raise PillarError(
                "E001",
                f"unknown variable {expr.name!r}",
                help_text="declare it with let before use",
            )
        return env[expr.name]
    if isinstance(expr, BinOp):
        left = _infer(expr.left, env, signatures)
        right = _infer(expr.right, env, signatures)
        if expr.op in ("<", ">", "=="):
            if left != right:
                raise PillarError("E001", f"cannot compare {left} with {right}")
            return "bool"
        if left != "i64" or right != "i64":
            raise PillarError("E001", f"operator {expr.op!r} needs i64 operands, got {left} and {right}")
        return "i64"
    if isinstance(expr, Applied):
        if expr.callee not in signatures:
            raise PillarError(
                "E006",
                f"call to undefined function {expr.callee!r}",
                help_text="define it in the same source file",
            )
        return signatures[expr.callee]
    raise PillarError("E001", f"cannot type expression {expr!r}")


def _check_block(statements: tuple[Stmt, ...], env: dict[str, str], ret: str, signatures: dict[str, str]) -> None:
    scope = dict(env)
    for statement in statements:
        if isinstance(statement, Let):
            scope[statement.name] = _infer(statement.value, scope, signatures)
        elif isinstance(statement, Return):
            actual = _infer(statement.value, scope, signatures)
            if actual != ret and not (ret == "bool" and isinstance(statement.value, BinOp) and statement.value.op in ("<", ">", "==")):
                raise PillarError(
                    "E001",
                    f"return type {actual} does not match declared {ret}",
                    help_text="change the return expression or the signature",
                )
        elif isinstance(statement, If):
            if _infer(statement.cond, scope, signatures) != "bool":
                raise PillarError("E001", "if condition must be bool")
            _check_block(statement.then, scope, ret, signatures)
            _check_block(statement.otherwise, scope, ret, signatures)
        else:  # pragma: no cover - exhaustive
            raise PillarError("E001", f"unknown statement {statement!r}")


def check_types(program: Program) -> None:
    """Reject undefined names, duplicate functions, and return mismatches."""
    signatures: dict[str, str] = {}
    for function in program.functions:
        if function.name in signatures:
            raise PillarError("E003", f"duplicate function {function.name!r}")
        signatures[function.name] = function.ret
    for function in program.functions:
        env = {name: "i64" for name in function.params}
        _check_block(function.body, env, function.ret, signatures)


# -- emitters ------------------------------------------------------------
def _emit_expr(expr: Expr) -> str:
    if isinstance(expr, IntLit):
        return f"(lit {expr.value})"
    if isinstance(expr, Var):
        return f"(var {expr.name})"
    if isinstance(expr, BinOp):
        op = {"+": "add", "-": "sub", "*": "mul", "/": "div", "<": "lt", ">": "gt", "==": "eq"}[expr.op]
        return f"({op} {_emit_expr(expr.left)} {_emit_expr(expr.right)})"
    if isinstance(expr, Applied):
        args = " ".join(_emit_expr(arg) for arg in expr.args)
        return f"(call {expr.callee}{' ' + args if args else ''})"
    raise PillarError("E004", f"cannot emit expression {expr!r}")


def _emit_stmt(statement: Stmt) -> str:
    if isinstance(statement, Let):
        return f"(let {statement.name} {_emit_expr(statement.value)})"
    if isinstance(statement, Return):
        return f"(return {_emit_expr(statement.value)})"
    if isinstance(statement, If):
        then = " ".join(_emit_stmt(item) for item in statement.then)
        otherwise = " ".join(_emit_stmt(item) for item in statement.otherwise)
        return f"(if {_emit_expr(statement.cond)} (block {then}) (block {otherwise}))"
    raise PillarError("E004", f"cannot emit statement {statement!r}")


def emit_ir(program: Program) -> str:
    """Canonical S-expression IR, one function per line."""
    lines: list[str] = []
    for function in program.functions:
        params = " ".join(function.params)
        body = " ".join(_emit_stmt(statement) for statement in function.body)
        lines.append(f"(fn {function.name} (params {params}) (ret {function.ret}) (block {body}))")
    return "\n".join(lines)


def emit_zig(program: Program) -> str:
    """A minimal Zig rendering, used to prove the stage-1 emitter ran."""
    lines: list[str] = []
    for function in program.functions:
        params = ", ".join(f"{name}: i64" for name in function.params)
        ret = "bool" if function.ret == "bool" else "i64"
        if function.ret == "bool":
            params = ", ".join(f"{name}: i64" for name in function.params)
        lines.append(f"fn {function.name}({params}) {ret} {{")
        for statement in function.body:
            if isinstance(statement, Return):
                lines.append(f"    return {_zig_expr(statement.value)};")
            else:
                raise PillarError(
                    "E007",
                    "the Zig emitter only handles a single return statement",
                    help_text="this is a demonstration emitter, not the native backend",
                )
        lines.append("}")
    return "\n".join(lines)


def _zig_expr(expr: Expr) -> str:
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, BinOp):
        symbol = {"<": "<", ">": ">", "==": "=="}.get(expr.op, expr.op)
        return f"({_zig_expr(expr.left)} {symbol} {_zig_expr(expr.right)})"
    if isinstance(expr, Applied):
        args = ", ".join(_zig_expr(arg) for arg in expr.args)
        return f"{expr.callee}({args})"
    raise PillarError("E004", f"cannot emit Zig expression {expr!r}")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
