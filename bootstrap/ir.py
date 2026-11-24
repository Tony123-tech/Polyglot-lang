"""Reader for the canonical S-expression IR.

Only the shapes that :func:`bootstrap.subset.emit_ir` can produce are accepted.
Anything else is malformed serialized state and fails with ``E008``, which keeps
the fixed-point check honest: it cannot pass by re-printing garbage.
"""

from __future__ import annotations

from runtime.diagnostics import PillarError

from .subset import Applied, BinOp, Function, If, IntLit, Let, Program, Return, Var

OPEN, CLOSE = "(", ")"


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    for char in text:
        if char in "()":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(char)
        elif char.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
        else:
            buffer += char
    if buffer:
        tokens.append(buffer)
    return tokens


def _parse(text: str):
    tokens = _tokens(text)
    position = 0

    def read():
        nonlocal position
        if position >= len(tokens):
            raise PillarError("E008", "IR ended in the middle of an expression")
        token = tokens[position]
        position += 1
        if token != OPEN:
            if token == CLOSE:
                raise PillarError("E008", "IR has an unbalanced ')'")
            return token
        items = []
        while True:
            if position >= len(tokens):
                raise PillarError("E008", "IR has an unclosed '('")
            if tokens[position] == CLOSE:
                position += 1
                return tuple(items)
            items.append(read())

    roots = []
    while position < len(tokens):
        roots.append(read())
    return roots


def _expr(node) -> object:
    if not isinstance(node, tuple) or not node:
        raise PillarError("E008", f"IR expression is not a list: {node!r}")
    head, tail = node[0], node[1:]
    if head == "lit" and len(tail) == 1 and tail[0].lstrip("-").isdigit():
        return IntLit(int(tail[0]))
    if head == "var" and len(tail) == 1:
        return Var(str(tail[0]))
    if head in {"add", "sub", "mul", "div", "lt", "gt", "eq"} and len(tail) == 2:
        symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "lt": "<",
            "gt": ">",
            "eq": "==",
        }[head]
        return BinOp(symbol, _expr(tail[0]), _expr(tail[1]))
    if head == "call" and tail:
        return Applied(str(tail[0]), tuple(_expr(arg) for arg in tail[1:]))
    raise PillarError("E008", f"IR node {head!r} is not part of the canonical subset")


def _block(node) -> tuple:
    if not isinstance(node, tuple) or node[:1] != ("block",):
        raise PillarError("E008", "expected a (block ...) node")
    statements = []
    for item in node[1:]:
        if not isinstance(item, tuple) or not item:
            raise PillarError("E008", f"IR statement is not a list: {item!r}")
        head = item[0]
        if head == "let" and len(item) == 3:
            statements.append(Let(str(item[1]), _expr(item[2])))
        elif head == "return" and len(item) == 2:
            statements.append(Return(_expr(item[1])))
        elif head == "if" and len(item) == 4:
            statements.append(If(_expr(item[1]), _block(item[2]), _block(item[3])))
        else:
            raise PillarError("E008", f"IR statement {head!r} is not supported")
    return tuple(statements)


def parse_ir(text: str) -> Program:
    """Read canonical IR back into a :class:`~bootstrap.subset.Program`."""
    roots = _parse(text)
    if not roots:
        raise PillarError("E008", "IR is empty")
    functions = []
    for entry in roots:
        if not isinstance(entry, tuple) or entry[:1] != ("fn",) or len(entry) != 5:
            raise PillarError(
                "E008", "expected (fn name (params ...) (ret T) (block ...))"
            )
        _tag, name, params, ret, body = entry
        if not isinstance(params, tuple) or params[:1] != ("params",):
            raise PillarError("E008", f"function {name!r} has no (params ...) node")
        if not isinstance(ret, tuple) or ret[:1] != ("ret",) or len(ret) != 2:
            raise PillarError("E008", f"function {name!r} has no (ret T) node")
        if ret[1] not in ("i64", "bool"):
            raise PillarError(
                "E008", f"function {name!r} has unsupported return type {ret[1]!r}"
            )
        functions.append(
            Function(
                str(name), tuple(str(p) for p in params[1:]), str(ret[1]), _block(body)
            )
        )
    return Program(tuple(functions))
