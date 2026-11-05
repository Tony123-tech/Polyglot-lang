"""
AST node definitions for Polyglot.

All nodes carry `line` and `col` so later layers (translator, runtime,
error handling) can map back to source location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ast_nodes.py — at the bottom

__all__ = [
    "ArrayExpr",
    "Assign",
    "Attr",
    "Await",
    "Binary",
    "Block",
    "Break",
    "Call",
    "Cast",
    "Closure",
    "CompoundAssign",
    "ConstDecl",
    "Continue",
    "EnumDecl",
    "EnumVariant",
    "Expr",
    "ExprStmt",
    "Field",
    "FnDecl",
    # Items
    "FnParam",
    "For",
    "GenericParam",
    "Ident",
    "IfExpr",
    "ImplBlock",
    "Index",
    "Item",
    "ItemStmt",
    # Statements
    "LetStmt",
    "LitBool",
    "LitChar",
    "LitFloat",
    # Expressions
    "LitInt",
    "LitStr",
    "Loop",
    "MacroCall",
    "MatchArm",
    "MatchExpr",
    "MethodCall",
    "ModDecl",
    # Base
    "Node",
    "Paren",
    "PatBinding",
    "PatBox",
    "PatConst",
    "PatEnum",
    "PatIdent",
    "PatLiteral",
    "PatMacro",
    "PatOr",
    "PatPath",
    "PatRange",
    "PatRef",
    "PatRest",
    "PatSlice",
    "PatStruct",
    "PatTuple",
    # Patterns
    "PatWildcard",
    "Path",
    "Pattern",
    "Program",
    "Range",
    "RepeatArray",
    "Return",
    "Stmt",
    "StructDecl",
    "StructLiteral",
    "TraitDecl",
    "TraitItem",
    "Try",
    "TupleExpr",
    "Type",
    "TypeAlias",
    "TypeArray",
    "TypeDynTrait",
    "TypeFn",
    "TypeImplTrait",
    # Types
    "TypeName",
    "TypeRef",
    "TypeSlice",
    "TypeTuple",
    "TypeUnit",
    "Unary",
    "UseDecl",
    "While",
]

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class Node:
    line: int
    col: int


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

@dataclass
class Program(Node):
    items: list[Item]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class TypeName(Node):
    """A named type, possibly with generic arguments: i64, Vec<T>, HashMap<K, V>."""
    name: str
    args: list[Type] = field(default_factory=list)


@dataclass
class TypeRef(Node):
    """Reference type: &T or &mut T."""
    inner: Type
    mutable: bool = False


@dataclass
class TypeSlice(Node):
    """Slice type: [T]."""
    inner: Type


@dataclass
class TypeArray(Node):
    """Array type: [T; N]."""
    inner: Type
    size: Expr


@dataclass
class TypeTuple(Node):
    """Tuple type: (A, B)."""
    elems: list[Type]


@dataclass
class TypeFn(Node):
    """Function pointer type: fn(A, B) -> C."""
    params: list[Type]
    ret: Type | None


@dataclass
class TypeImplTrait(Node):
    """impl Trait."""
    bounds: list[TypeName]


@dataclass
class TypeDynTrait(Node):
    """dyn Trait."""
    bounds: list[TypeName]


@dataclass
class TypeUnit(Node):
    """The unit type ()."""


Type = (
    TypeName | TypeRef | TypeSlice | TypeArray
    | TypeTuple | TypeFn | TypeImplTrait | TypeDynTrait | TypeUnit
)


# ---------------------------------------------------------------------------
# Patterns (for match, let, fn params)
# ---------------------------------------------------------------------------

@dataclass
class PatWildcard(Node):
    """_"""


@dataclass
class PatIdent(Node):
    """x or mut x"""
    name: str
    mutable: bool = False


@dataclass
class PatLiteral(Node):
    """Numeric / string / char / bool literal pattern."""
    value: Any


@dataclass
class PatTuple(Node):
    elems: list[Pattern]


@dataclass
class PatStruct(Node):
    path: str
    fields: list[tuple[str, Pattern]]
    rest: bool = False


@dataclass
class PatEnum(Node):
    path: str
    args: list[Pattern] = field(default_factory=list)


@dataclass
class PatOr(Node):
    alts: list[Pattern]


@dataclass
class PatRange(Node):
    start: Expr | None
    end: Expr | None
    inclusive: bool


@dataclass
class PatBinding(Node):
    """x @ pat"""
    name: str
    inner: Pattern


@dataclass
class PatRef(Node):
    """&pat or &mut pat"""
    inner: Pattern
    mutable: bool = False


@dataclass
class PatRest(Node):
    """A `..` rest pattern, optionally bound by an enclosing PatBinding."""


@dataclass
class PatBox(Node):
    inner: Pattern


@dataclass
class PatConst(Node):
    path: str


@dataclass
class PatMacro(Node):
    path: str
    tokens: str
    delimiter: str = "()"


@dataclass
class PatPath(Node):
    path: str


@dataclass
class PatSlice(Node):
    elems: list[Pattern]


Pattern = (
    PatWildcard | PatIdent | PatLiteral | PatTuple | PatStruct
    | PatEnum | PatOr | PatRange | PatBinding | PatRef | PatSlice | PatRest
    | PatBox | PatConst | PatMacro | PatPath
)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class LitInt(Node):
    value: int
    raw: str


@dataclass
class LitFloat(Node):
    value: float
    raw: str


@dataclass
class LitStr(Node):
    value: str


@dataclass
class LitChar(Node):
    value: str


@dataclass
class LitBool(Node):
    value: bool


@dataclass
class Ident(Node):
    name: str


@dataclass
class Path(Node):
    """A::B::C"""
    segments: list[str]


@dataclass
class Unary(Node):
    op: str  # "-", "!", "*", "&", "&mut"
    operand: Expr
    mutable: bool = False


@dataclass
class Binary(Node):
    op: str
    left: Expr
    right: Expr


@dataclass
class Assign(Node):
    target: Expr
    value: Expr


@dataclass
class CompoundAssign(Node):
    op: str  # "+=", "-=", "*=", "/="
    target: Expr
    value: Expr


@dataclass
class Call(Node):
    func: Expr
    args: list[Expr]


@dataclass
class MethodCall(Node):
    receiver: Expr
    method: str
    args: list[Expr]


@dataclass
class Field(Node):
    receiver: Expr
    name: str


@dataclass
class Index(Node):
    receiver: Expr
    index: Expr


@dataclass
class TupleExpr(Node):
    elems: list[Expr]


@dataclass
class ArrayExpr(Node):
    elems: list[Expr]


@dataclass
class RepeatArray(Node):
    """[expr; n]"""
    elem: Expr
    count: Expr


@dataclass
class StructLiteral(Node):
    path: str
    fields: list[tuple[str, Expr]]
    rest: Expr | None = None


@dataclass
class IfExpr(Node):
    cond: Expr
    then_block: Block
    else_branch: Expr | Block | IfExpr | None


@dataclass
class MatchExpr(Node):
    scrutinee: Expr
    arms: list[MatchArm]


@dataclass
class MatchArm(Node):
    pattern: Pattern
    guard: Expr | None
    body: Expr


@dataclass
class Block(Node):
    stmts: list[Stmt]
    tail: Expr | None = None


@dataclass
class Loop(Node):
    body: Block


@dataclass
class While(Node):
    cond: Expr
    body: Block


@dataclass
class For(Node):
    pattern: Pattern
    iter: Expr
    body: Block


@dataclass
class Closure(Node):
    params: list[tuple[Pattern, Type | None]]
    body: Expr
    ret: Type | None = None


@dataclass
class Range(Node):
    start: Expr | None
    end: Expr | None
    inclusive: bool


@dataclass
class Cast(Node):
    expr: Expr
    ty: Type


@dataclass
class Return(Node):
    value: Expr | None


@dataclass
class Break(Node):
    value: Expr | None = None
    label: str | None = None


@dataclass
class Continue(Node):
    label: str | None = None


@dataclass
class MacroCall(Node):
    path: str
    args: list[Expr] | str  # str for opaque token stream


@dataclass
class Await(Node):
    expr: Expr


@dataclass
class Try(Node):
    """The ? operator."""
    expr: Expr


@dataclass
class Paren(Node):
    inner: Expr


Expr = (
    LitInt | LitFloat | LitStr | LitChar | LitBool
    | Ident | Path | Unary | Binary | Assign | CompoundAssign
    | Call | MethodCall | Field | Index | TupleExpr | ArrayExpr
    | RepeatArray | StructLiteral | IfExpr | MatchExpr | Block
    | Loop | While | For | Closure | Range | Cast
    | Return | Break | Continue | MacroCall | Await | Try | Paren
)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class LetStmt(Node):
    pattern: Pattern
    ty: Type | None
    value: Expr | None
    mutable: bool = False


@dataclass
class ExprStmt(Node):
    expr: Expr


@dataclass
class ItemStmt(Node):
    item: Item


Stmt = LetStmt | ExprStmt | ItemStmt


# ---------------------------------------------------------------------------
# Items (top-level declarations)
# ---------------------------------------------------------------------------

@dataclass
class FnParam(Node):
    pattern: Pattern
    ty: Type | None


@dataclass
class FnDecl(Node):
    name: str
    params: list[FnParam]
    ret: Type | None
    body: Block | None
    generics: list[GenericParam] = field(default_factory=list)
    is_pub: bool = False
    is_async: bool = False
    is_unsafe: bool = False


@dataclass
class StructDecl(Node):
    name: str
    fields: list[tuple[str, Type, bool]]  # (name, type, is_pub)
    generics: list[GenericParam] = field(default_factory=list)
    is_pub: bool = False


@dataclass
class EnumVariant(Node):
    name: str
    fields: list[Type] | None  # None = unit, list = tuple-like, or named fields


@dataclass
class EnumDecl(Node):
    name: str
    variants: list[EnumVariant]
    generics: list[GenericParam] = field(default_factory=list)
    is_pub: bool = False


@dataclass
class ImplBlock(Node):
    ty: Type
    trait: Type | None
    items: list[Item]
    generics: list[GenericParam] = field(default_factory=list)


@dataclass
class TraitItem(Node):
    """A signature or default method inside a trait."""
    fn: FnDecl | None
    ty: Type | None


@dataclass
class TraitDecl(Node):
    name: str
    items: list[TraitItem]
    generics: list[GenericParam] = field(default_factory=list)
    is_pub: bool = False


@dataclass
class UseDecl(Node):
    path: str


@dataclass
class ModDecl(Node):
    name: str
    items: list[Item] | None


@dataclass
class ConstDecl(Node):
    name: str
    ty: Type | None
    value: Expr
    is_static: bool = False
    is_pub: bool = False


@dataclass
class TypeAlias(Node):
    name: str
    ty: Type
    generics: list[GenericParam] = field(default_factory=list)
    is_pub: bool = False


@dataclass
class GenericParam(Node):
    name: str
    bounds: list[TypeName] = field(default_factory=list)


@dataclass
class Attr(Node):
    """#[derive(...)] or #[cfg(...)]"""
    name: str
    args: str | None


Item = (
    FnDecl | StructDecl | EnumDecl | ImplBlock | TraitDecl
    | UseDecl | ModDecl | ConstDecl | TypeAlias
)
