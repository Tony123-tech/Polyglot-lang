"""A miniature typed AST shared by the macro, parity, and bootstrap layers.

The nodes are deliberately small: enough structure to prove hygiene, typed
splicing, and cross-backend parity without claiming to model the full
polyglot-lang grammar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LitInt:
    value: int

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class LitStr:
    value: str

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class LitBool:
    value: bool

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class Var:
    name: str

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class Hole:
    """A macro parameter position, filled by a typed splice."""

    index: int

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class Add:
    left: Node
    right: Node

    def children(self) -> tuple[Node, ...]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Call:
    callee: str
    args: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.args


@dataclass(frozen=True)
class Let:
    name: str
    value: Node
    body: Node

    def children(self) -> tuple[Node, ...]:
        return (self.value, self.body)


@dataclass(frozen=True)
class If:
    cond: Node
    then: Node
    otherwise: Node

    def children(self) -> tuple[Node, ...]:
        return (self.cond, self.then, self.otherwise)


@dataclass(frozen=True)
class Return:
    value: Node

    def children(self) -> tuple[Node, ...]:
        return (self.value,)


@dataclass(frozen=True)
class Block:
    statements: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.statements


Node = (
    LitInt
    | LitStr
    | LitBool
    | Var
    | Hole
    | Add
    | Call
    | Let
    | If
    | Return
    | Block
)
