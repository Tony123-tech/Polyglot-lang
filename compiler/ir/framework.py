"""FRAME 3: lowering, IR construction, and deterministic optimization passes.

This module is a **model compiler frontend**.  It lowers the existing Polyglot
surface AST into a small typed HIR, lowers that HIR into an explicit MIR control
flow graph, and runs inspection-friendly analysis/optimization passes over those
representations.  It deliberately does not emit Python or Rust source, invoke a
backend, allocate memory, or execute user code: every constant fold is evaluated
with a closed set of pure operators, and every pass returns new dataclasses.

The IR is intentionally compact.  HIR keeps source-level constructs (blocks,
loops, matches, short-circuit operators) with inferred types.  MIR exposes
basic blocks, explicit terminators, and instructions so that all control flow
transformations are deterministic and easy to verify in tests.
"""

from __future__ import annotations

import itertools
import operator
import re
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, get_args

import compiler.core.ast_nodes as ast
from compiler.core.lexer import lex
from compiler.core.parser import parse

__all__ = [
    "BasicBlock",
    "BorrowIssue",
    "BorrowReport",
    "Cfg",
    "CfgIssue",
    "DropPoint",
    "DropResult",
    "EliminationResult",
    "FoldResult",
    "HirBranch",
    "HirExpr",
    "HirFunction",
    "HirModule",
    "HirParam",
    "HirStmt",
    "HirType",
    "HoistResult",
    "InlineResult",
    "InvariantIssue",
    "InvariantReport",
    "MirFunction",
    "MirInstr",
    "MirModule",
    "MirOperand",
    "MirTerminator",
    "ValidationResult",
    "ir_check_borrow_and_move_state",
    "ir_eliminate_dead_blocks",
    "ir_fold_constant_expressions",
    "ir_hoist_loop_invariant_checks",
    "ir_inline_small_pure_functions",
    "ir_insert_raii_drop_points",
    "ir_lower_ast_to_hir",
    "ir_lower_hir_to_mir",
    "ir_validate_control_flow_graph",
    "ir_verify_dual_backend_invariants",
]

type Span = tuple[int, int] | None

UNIT = "()"
UNKNOWN = "?"
BOOL = "bool"
I64 = "i64"

OWNED_TYPES = frozenset({"String", "Vec", "Box", "HashMap", "HashSet", "Deque"})
BACKEND_SCALARS = frozenset(
    {
        UNIT,
        UNKNOWN,
        "bool",
        "char",
        "str",
        "i8",
        "i16",
        "i32",
        "i64",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "usize",
        "f32",
        "f64",
    }
)
BACKEND_CONTAINERS = frozenset({"Vec", "Box", "Option", "Result", "HashMap", "HashSet"})
PURE_BUILTINS = frozenset({"abs", "len", "max", "min", "sqrt"})
IMPURE_NAMES = frozenset(
    {"print", "println", "eprint", "input", "read", "write", "panic", "rand", "now"}
)
PYTHON_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_EXPR_TYPES = get_args(ast.Expr)


# ---------------------------------------------------------------------------
# Types and HIR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HirType:
    """Resolved type used by both HIR and MIR."""

    name: str
    args: tuple[HirType, ...] = ()
    mutable: bool = False
    referenced: bool = False

    @property
    def text(self) -> str:
        """Return a stable, printable spelling of this type."""
        inner = ", ".join(arg.text for arg in self.args)
        base = f"{self.name}<{inner}>" if inner else self.name
        if self.referenced:
            return f"&{'mut ' if self.mutable else ''}{base}"
        return base

    @property
    def owned(self) -> bool:
        """Whether a value of this type needs deterministic cleanup."""
        return self.name in OWNED_TYPES and not self.referenced

    @property
    def backend_supported(self) -> bool:
        """Whether both target backends have a direct spelling for this type."""
        if self.name in BACKEND_SCALARS:
            return True
        if self.name in BACKEND_CONTAINERS:
            return bool(self.args) and all(arg.backend_supported for arg in self.args)
        if self.name in {"slice", "array", "tuple"}:
            return all(arg.backend_supported for arg in self.args)
        return False


UNKNOWN_TYPE = HirType(UNKNOWN)
UNIT_TYPE = HirType(UNIT)
BOOL_TYPE = HirType(BOOL)
I64_TYPE = HirType(I64)


def _span(node: object | None) -> Span:
    """Extract a source span from an AST or IR node when present."""
    value = getattr(node, "span", None)
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def _type_from_ast(node: ast.Type | None) -> HirType:
    """Translate a surface type annotation into a :class:`HirType`."""
    if node is None:
        return UNIT_TYPE
    if isinstance(node, ast.TypeName):
        return HirType(node.name, tuple(_type_from_ast(arg) for arg in node.args))
    if isinstance(node, ast.TypeRef):
        inner = _type_from_ast(node.inner)
        return HirType(inner.name, inner.args, mutable=node.mutable, referenced=True)
    if isinstance(node, ast.TypeSlice):
        return HirType("slice", (_type_from_ast(node.inner),))
    if isinstance(node, ast.TypeArray):
        return HirType("array", (_type_from_ast(node.inner), _type_from_ast(node.size)))
    if isinstance(node, ast.TypeTuple):
        return HirType("tuple", tuple(_type_from_ast(elem) for elem in node.elems))
    if isinstance(node, ast.TypeFn):
        parts = tuple(_type_from_ast(item) for item in node.params)
        return HirType("fn", parts + (_type_from_ast(node.ret),))
    if isinstance(node, ast.TypeUnit):
        return UNIT_TYPE
    if isinstance(node, ast.TypeImplTrait | ast.TypeDynTrait):
        return HirType("dyn", tuple(HirType(bound.name) for bound in node.bounds))
    return UNKNOWN_TYPE


def _literal_value(node: ast.Expr | None) -> Any | None:
    """Return the compile-time value of a literal expression, if any."""
    if isinstance(node, ast.Paren):
        return _literal_value(node.inner)
    if isinstance(
        node, ast.LitInt | ast.LitFloat | ast.LitStr | ast.LitChar | ast.LitBool
    ):
        return node.value
    if isinstance(node, ast.Unary) and node.op in {"-", "!"}:
        inner = _literal_value(node.operand)
        if isinstance(inner, bool) and node.op == "!":
            return not inner
        if isinstance(inner, int | float) and node.op == "-":
            return -inner
    return None


def _literal_type(value: Any) -> HirType:
    """Infer the primitive type of a literal value."""
    if isinstance(value, bool):
        return BOOL_TYPE
    if isinstance(value, int):
        return I64_TYPE
    if isinstance(value, float):
        return HirType("f64")
    if isinstance(value, str):
        return HirType("str")
    return UNKNOWN_TYPE


def _pattern_names(pattern: ast.Pattern | None) -> tuple[str, ...]:
    """Collect binding names introduced by a pattern, in source order."""
    if pattern is None or isinstance(pattern, ast.PatWildcard | ast.PatRest):
        return ()
    if isinstance(pattern, ast.PatIdent):
        return (pattern.name,)
    if isinstance(pattern, ast.PatBinding):
        return (pattern.name, *_pattern_names(pattern.inner))
    if isinstance(pattern, ast.PatRef | ast.PatBox):
        return _pattern_names(pattern.inner)
    if isinstance(pattern, ast.PatTuple | ast.PatSlice):
        result: list[str] = []
        for elem in pattern.elems:
            result.extend(_pattern_names(elem))
        return tuple(result)
    if isinstance(pattern, ast.PatStruct):
        result = []
        for _, sub in pattern.fields:
            result.extend(_pattern_names(sub))
        return tuple(result)
    if isinstance(pattern, ast.PatEnum):
        result = []
        for sub in pattern.args:
            result.extend(_pattern_names(sub))
        return tuple(result)
    if isinstance(pattern, ast.PatOr):
        result = []
        for alt in pattern.alts:
            result.extend(_pattern_names(alt))
        return tuple(result)
    return ()


def _pattern_repr(pattern: ast.Pattern | None) -> str:
    """Render a pattern as a stable string used by MIR tests and guards."""
    if pattern is None or isinstance(pattern, ast.PatWildcard):
        return "_"
    if isinstance(pattern, ast.PatIdent):
        return pattern.name
    if isinstance(pattern, ast.PatLiteral):
        value = _literal_value(pattern.value)
        return repr(value) if value is not None else type(pattern.value).__name__
    if isinstance(pattern, ast.PatBinding):
        return f"{pattern.name} @ {_pattern_repr(pattern.inner)}"
    if isinstance(pattern, ast.PatRef):
        return f"&{'mut ' if pattern.mutable else ''}{_pattern_repr(pattern.inner)}"
    if isinstance(pattern, ast.PatBox):
        return f"box {_pattern_repr(pattern.inner)}"
    if isinstance(pattern, ast.PatEnum):
        args = ", ".join(_pattern_repr(arg) for arg in pattern.args)
        return f"{pattern.path}({args})"
    if isinstance(pattern, ast.PatStruct):
        fields = ", ".join(
            f"{name}: {_pattern_repr(sub)}" for name, sub in pattern.fields
        )
        suffix = ", .." if pattern.rest else ""
        return f"{pattern.path} {{ {fields}{suffix} }}"
    if isinstance(pattern, ast.PatTuple | ast.PatSlice):
        elems = ", ".join(_pattern_repr(elem) for elem in pattern.elems)
        return f"({elems})"
    if isinstance(pattern, ast.PatOr):
        return " | ".join(_pattern_repr(alt) for alt in pattern.alts)
    if isinstance(pattern, ast.PatRange):
        start = _pattern_repr(pattern.start) if pattern.start is not None else ""
        end = _pattern_repr(pattern.end) if pattern.end is not None else ""
        return f"{start}..{'=' if pattern.inclusive else ''}{end}"
    if isinstance(pattern, ast.PatConst | ast.PatPath | ast.PatMacro):
        return getattr(pattern, "path", UNKNOWN)
    return type(pattern).__name__


@dataclass
class HirExpr:
    """A typed high-level IR expression with a stable kind tag."""

    kind: str
    ty: HirType = UNKNOWN_TYPE
    value: Any = None
    args: tuple[HirExpr, ...] = ()
    branches: tuple[HirBranch, ...] = ()
    span: Span = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HirBranch:
    """One guarded arm of an ``if`` or ``match`` expression."""

    condition: HirExpr | None
    body: HirExpr
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HirStmt:
    """A high-level statement; values are stored as HIR expressions."""

    kind: str
    name: str | None = None
    value: HirExpr | None = None
    ty: HirType = UNIT_TYPE
    mutable: bool = False
    span: Span = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HirParam:
    """A named function parameter with a resolved type."""

    name: str
    ty: HirType
    span: Span = None


@dataclass
class HirFunction:
    """A lowered function with source-derived purity metadata."""

    name: str
    params: tuple[HirParam, ...]
    ret: HirType
    body: HirExpr
    pure: bool = False
    recursive: bool = False
    span: Span = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HirModule:
    """A collection of lowered functions in deterministic insertion order."""

    functions: dict[str, HirFunction]
    source: str = ""


# ---------------------------------------------------------------------------
# MIR / CFG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirOperand:
    """A variable reference or a compile-time constant in MIR."""

    kind: str
    value: Any = None
    ty: HirType = UNKNOWN_TYPE

    @property
    def text(self) -> str:
        """Stable textual form used by reports."""
        if self.kind == "var":
            return str(self.value)
        if self.kind == "const":
            return repr(self.value)
        return "()"


def _var(name: str, ty: HirType = UNKNOWN_TYPE) -> MirOperand:
    """Create a variable operand."""
    return MirOperand("var", name, ty)


def _const(value: Any, ty: HirType | None = None) -> MirOperand:
    """Create a constant operand, inferring its type when necessary."""
    return MirOperand("const", value, ty or _literal_type(value))


def _unit() -> MirOperand:
    """Create the unit operand."""
    return MirOperand("unit", None, UNIT_TYPE)


def _operand_from_literal(value: Any) -> MirOperand:
    """Create a constant operand with a normalized literal value."""
    if isinstance(value, ast.LitInt):
        return _const(value.value, I64_TYPE)
    if isinstance(value, ast.LitFloat):
        return _const(value.value, HirType("f64"))
    if isinstance(value, ast.LitStr):
        return _const(value.value, HirType("str"))
    if isinstance(value, ast.LitChar):
        return _const(value.value, HirType("char"))
    if isinstance(value, ast.LitBool):
        return _const(value.value, BOOL_TYPE)
    return _const(value)


@dataclass
class MirInstr:
    """A side-effecting or value-producing MIR instruction."""

    op: str
    dest: str | None = None
    args: tuple[MirOperand, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    span: Span = None


@dataclass
class MirTerminator:
    """The explicit final transfer of control in a basic block."""

    kind: str
    targets: tuple[str, ...] = ()
    cond: MirOperand | None = None
    value: MirOperand | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BasicBlock:
    """A basic block: straight-line instructions plus one terminator."""

    label: str
    instructions: list[MirInstr] = field(default_factory=list)
    terminator: MirTerminator = field(default_factory=lambda: MirTerminator(""))
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cfg:
    """A control flow graph with deterministic block labels."""

    entry: str
    blocks: dict[str, BasicBlock]

    def successors(self, label: str) -> tuple[str, ...]:
        """Return successor labels of ``label`` in terminator order."""
        block = self.blocks.get(label)
        return () if block is None else block.terminator.targets

    def predecessors(self) -> dict[str, tuple[str, ...]]:
        """Return predecessor labels for every block, sorted by label."""
        result: dict[str, list[str]] = {label: [] for label in self.blocks}
        for label, block in self.blocks.items():
            for target in block.terminator.targets:
                if target in result:
                    result[target].append(label)
        return {label: tuple(sorted(names)) for label, names in result.items()}

    def reachable(self) -> set[str]:
        """Return labels reachable from the entry block."""
        if self.entry not in self.blocks:
            return set()
        seen: set[str] = set()
        queue: deque[str] = deque([self.entry])
        while queue:
            label = queue.popleft()
            if label in seen:
                continue
            seen.add(label)
            queue.extend(
                target
                for target in self.successors(label)
                if target in self.blocks and target not in seen
            )
        return seen


@dataclass
class MirFunction:
    """A MIR function: signature plus its control flow graph."""

    name: str
    cfg: Cfg
    params: tuple[HirParam, ...] = ()
    ret: HirType = UNIT_TYPE
    pure: bool = False
    recursive: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MirModule:
    """A collection of lowered functions ready for backend-independent passes."""

    functions: dict[str, MirFunction]
    source: str = ""


# ---------------------------------------------------------------------------
# Pass result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Result of literal folding with a log of each replacement."""

    module: HirModule | MirModule
    replacements: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        """Number of folded expressions."""
        return len(self.replacements)


@dataclass(frozen=True)
class EliminationResult:
    """Result of unreachable-block and dead-assignment elimination."""

    module: MirModule
    removed_blocks: tuple[str, ...] = ()
    removed_assignments: tuple[str, ...] = ()


@dataclass(frozen=True)
class CfgIssue:
    """A validation finding with a stable machine-readable code."""

    code: str
    block: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class ValidationResult:
    """CFG validation outcome; ``valid`` ignores warning-severity issues."""

    valid: bool
    issues: tuple[CfgIssue, ...] = ()

    def __bool__(self) -> bool:
        """Allow ``if validate(...)`` style checks."""
        return self.valid


@dataclass(frozen=True)
class BorrowIssue:
    """A move, borrow, or use-after-move finding."""

    code: str
    variable: str
    block: str
    message: str


@dataclass(frozen=True)
class BorrowReport:
    """Borrow/move analysis outcome with per-block final state snapshots."""

    ok: bool
    issues: tuple[BorrowIssue, ...] = ()
    states: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DropPoint:
    """A deterministic cleanup insertion site."""

    function: str
    block: str
    variable: str
    reason: str


@dataclass(frozen=True)
class DropResult:
    """Result of RAII drop insertion."""

    module: MirModule
    insertions: tuple[DropPoint, ...] = ()


@dataclass(frozen=True)
class InlineResult:
    """Result of pure-function inlining."""

    module: HirModule | MirModule
    inlined_sites: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class HoistResult:
    """Result of loop-invariant guard hoisting."""

    module: MirModule
    hoisted: tuple[str, ...] = ()
    preheaders: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvariantIssue:
    """A pre-codegen invariant violation shared by both backends."""

    code: str
    function: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class InvariantReport:
    """Outcome of the dual-backend pre-codegen check."""

    ok: bool
    issues: tuple[InvariantIssue, ...] = ()
    checked: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# AST -> HIR
# ---------------------------------------------------------------------------


def _program_from_input(source: str | ast.Program) -> ast.Program:
    """Accept source text or a parsed program and return a program."""
    if isinstance(source, ast.Program):
        return source
    if isinstance(source, str):
        return parse(lex(source))
    raise TypeError("expected source text or an ast_nodes.Program")


def _common_type(left: HirType, right: HirType) -> HirType:
    """Choose a deterministic result type for a binary operation."""
    if left == right:
        return left
    if left == UNKNOWN_TYPE:
        return right
    if right == UNKNOWN_TYPE:
        return left
    if left.name in {"f32", "f64"} or right.name in {"f32", "f64"}:
        return HirType("f64")
    return left


class _Lowerer:
    """Stateful lowering of one surface program into typed HIR."""

    def __init__(self, program: ast.Program) -> None:
        self.program = program
        self.source = "\n".join(
            program.source_lines[line] for line in sorted(program.source_lines)
        )
        self._scopes = itertools.count(1)
        self.signatures: dict[
            str, tuple[tuple[tuple[str, HirType, bool], ...], HirType]
        ] = {}
        for item in program.items:
            if isinstance(item, ast.FnDecl):
                self.signatures[item.name] = self._signature(item)

    @staticmethod
    def _signature(
        fn: ast.FnDecl,
    ) -> tuple[tuple[tuple[str, HirType, bool], ...], HirType]:
        """Compute a function signature from its declaration."""
        params: list[tuple[str, HirType, bool]] = []
        for param in fn.params:
            names = _pattern_names(param.pattern)
            ty = _type_from_ast(param.ty)
            mutable = any(
                isinstance(name_node, ast.PatIdent) and name_node.mutable
                for name_node in _walk_patterns(param.pattern)
            )
            for name in names or ("_arg",):
                params.append((name, ty, mutable))
        return tuple(params), _type_from_ast(fn.ret)

    def lower(self) -> HirModule:
        """Lower every function in the program and annotate call metadata."""
        functions: dict[str, HirFunction] = {}
        for item in self.program.items:
            if isinstance(item, ast.FnDecl):
                functions[item.name] = self._lower_function(item)
        self._analyze(functions)
        return HirModule(functions, self.source)

    def _lower_function(self, fn: ast.FnDecl) -> HirFunction:
        """Lower one function declaration."""
        signature = self.signatures[fn.name]
        params = tuple(HirParam(name, ty, _span(fn)) for name, ty, _ in signature[0])
        env = {name: ty for name, ty, _ in signature[0]}
        body = self._lower_expr(fn.body, env)
        attributes = tuple(getattr(attr, "name", "") for attr in fn.attributes)
        return HirFunction(
            name=fn.name,
            params=params,
            ret=signature[1],
            body=body,
            span=_span(fn),
            meta={
                "attributes": attributes,
                "async": fn.is_async,
                "unsafe": fn.is_unsafe,
                "terminates": fn.terminates,
            },
        )

    def _lower_stmt(self, stmt: ast.Stmt, env: dict[str, HirType]) -> HirStmt:
        """Lower one surface statement."""
        if isinstance(stmt, ast.LetStmt):
            declared = _type_from_ast(stmt.ty)
            value = None if stmt.value is None else self._lower_expr(stmt.value, env)
            ty = (
                declared
                if declared != UNIT_TYPE
                else (value.ty if value else UNKNOWN_TYPE)
            )
            names = _pattern_names(stmt.pattern)
            name = names[0] if names else None
            for binding in names:
                env[binding] = ty
            return HirStmt(
                kind="let",
                name=name,
                value=value,
                ty=ty,
                mutable=stmt.mutable,
                span=_span(stmt),
                meta={"bindings": names, "scope": next(self._scopes)},
            )
        if isinstance(stmt, ast.ExprStmt):
            return HirStmt(
                kind="expr",
                value=self._lower_expr(stmt.expr, env),
                span=_span(stmt),
            )
        if isinstance(stmt, ast.ItemStmt):
            return HirStmt(
                kind="item",
                span=_span(stmt),
                meta={"item": type(stmt.item).__name__},
            )
        if isinstance(stmt, ast.ErrDeferStmt):
            body = self._lower_expr(stmt.body, env)
            return HirStmt(
                kind="errdefer",
                value=body,
                span=_span(stmt),
                meta={"scope": next(self._scopes)},
            )
        return HirStmt(kind="op", span=_span(stmt), meta={"node": type(stmt).__name__})

    def _lower_block(self, block: ast.Block, env: dict[str, HirType]) -> HirExpr:
        """Lower a block, keeping a copy of the surrounding environment."""
        local = dict(env)
        stmts = tuple(self._lower_stmt(stmt, local) for stmt in block.stmts)
        tail = None if block.tail is None else self._lower_expr(block.tail, local)
        args = (tail,) if tail is not None else ()
        return HirExpr(
            kind="block",
            ty=tail.ty if tail is not None else UNIT_TYPE,
            args=args,
            span=_span(block),
            meta={"stmts": stmts, "scope": next(self._scopes)},
        )

    def _lower_else(
        self,
        branch: ast.Expr | ast.Block | ast.IfExpr | ast.IfLet | None,
        env: dict[str, HirType],
    ) -> HirExpr:
        """Lower an else branch, synthesizing unit when absent."""
        if branch is None:
            return HirExpr(
                "block", UNIT_TYPE, meta={"stmts": (), "scope": next(self._scopes)}
            )
        if isinstance(branch, ast.Block):
            return self._lower_block(branch, env)
        return self._lower_expr(branch, env)

    def _lower_match_arm(
        self, arm: ast.MatchArm, env: dict[str, HirType], index: int
    ) -> HirBranch:
        """Lower one match arm into a guarded HIR branch."""
        local = dict(env)
        for name in _pattern_names(arm.pattern):
            local[name] = local.get(name, UNKNOWN_TYPE)
        body = self._lower_expr(arm.body, local)
        pattern = _pattern_repr(arm.pattern)
        condition = None
        if pattern != "_":
            condition = HirExpr(
                kind="pattern_test",
                ty=BOOL_TYPE,
                value=pattern,
                span=_span(arm.pattern),
            )
        if arm.guard is not None:
            guard = self._lower_expr(arm.guard, local)
            condition = (
                guard
                if condition is None
                else HirExpr(
                    "logical_and", BOOL_TYPE, value="&&", args=(condition, guard)
                )
            )
        return HirBranch(
            condition=condition,
            body=body,
            meta={"pattern": pattern, "arm": index, "guard": arm.guard is not None},
        )

    def _lower_expr(self, node: ast.Expr, env: dict[str, HirType]) -> HirExpr:
        """Recursively lower one expression with local type inference."""
        if isinstance(node, ast.Paren):
            return self._lower_expr(node.inner, env)
        if isinstance(
            node, ast.LitInt | ast.LitFloat | ast.LitStr | ast.LitChar | ast.LitBool
        ):
            value = node.value
            return HirExpr(
                "literal", _literal_type(value), value=value, span=_span(node)
            )
        if isinstance(node, ast.Ident):
            return HirExpr(
                "var",
                env.get(node.name, UNKNOWN_TYPE),
                value=node.name,
                span=_span(node),
            )
        if isinstance(node, ast.Path):
            name = "::".join(node.segments)
            return HirExpr("path", UNKNOWN_TYPE, value=name, span=_span(node))
        if isinstance(node, ast.Unary):
            operand = self._lower_expr(node.operand, env)
            if node.op in {"&", "&mut"}:
                ty = HirType(
                    operand.ty.name,
                    operand.ty.args,
                    mutable=node.op == "&mut",
                    referenced=True,
                )
            elif node.op == "!":
                ty = BOOL_TYPE
            else:
                ty = operand.ty
            return HirExpr("unop", ty, value=node.op, args=(operand,), span=_span(node))
        if isinstance(node, ast.Binary):
            left = self._lower_expr(node.left, env)
            right = self._lower_expr(node.right, env)
            if node.op in {"&&", "||"}:
                kind = "logical_and" if node.op == "&&" else "logical_or"
                return HirExpr(
                    kind, BOOL_TYPE, value=node.op, args=(left, right), span=_span(node)
                )
            ty = (
                BOOL_TYPE
                if node.op in {"==", "!=", "<", "<=", ">", ">="}
                else _common_type(left.ty, right.ty)
            )
            return HirExpr(
                "bin", ty, value=node.op, args=(left, right), span=_span(node)
            )
        if isinstance(node, ast.Assign):
            value = self._lower_expr(node.value, env)
            target = self._lower_expr(node.target, env)
            return HirExpr(
                "assign",
                UNIT_TYPE,
                args=(value,),
                span=_span(node),
                meta={"target": target.value},
            )
        if isinstance(node, ast.CompoundAssign):
            value = self._lower_expr(node.value, env)
            target = self._lower_expr(node.target, env)
            return HirExpr(
                "assign",
                UNIT_TYPE,
                value=node.op,
                args=(value,),
                span=_span(node),
                meta={"target": target.value, "compound": True},
            )
        if isinstance(node, ast.Call):
            callee = self._callee_name(node.func)
            args = tuple(self._lower_expr(arg, env) for arg in node.args)
            signature = self.signatures.get(callee)
            ty = signature[1] if signature else UNKNOWN_TYPE
            return HirExpr(
                "call",
                ty,
                value=callee,
                args=args,
                span=_span(node),
                meta={"callee": callee},
            )
        if isinstance(node, ast.MethodCall):
            receiver = self._lower_expr(node.receiver, env)
            args = (receiver, *(self._lower_expr(arg, env) for arg in node.args))
            return HirExpr(
                "call",
                UNKNOWN_TYPE,
                value=node.method,
                args=args,
                span=_span(node),
                meta={"method": True},
            )
        if isinstance(node, ast.IfExpr):
            cond = self._lower_expr(node.cond, env)
            then_body = self._lower_block(node.then_block, env)
            else_body = self._lower_else(node.else_branch, env)
            ty = (
                then_body.ty
                if then_body.ty == else_body.ty
                else _common_type(then_body.ty, else_body.ty)
            )
            return HirExpr(
                "if",
                ty,
                args=(cond,),
                branches=(
                    HirBranch(None, then_body, {"kind": "then"}),
                    HirBranch(None, else_body, {"kind": "else"}),
                ),
                span=_span(node),
            )
        if isinstance(node, ast.IfLet):
            value = self._lower_expr(node.value, env)
            local = dict(env)
            for name in _pattern_names(node.pattern):
                local[name] = value.ty
            test = HirExpr(
                "pattern_test",
                BOOL_TYPE,
                value=_pattern_repr(node.pattern),
                args=(value,),
                span=_span(node),
            )
            then_body = self._lower_block(node.then_block, local)
            else_body = self._lower_else(node.else_branch, env)
            return HirExpr(
                "if",
                then_body.ty,
                args=(test,),
                branches=(
                    HirBranch(
                        None,
                        then_body,
                        {"kind": "then", "pattern": _pattern_repr(node.pattern)},
                    ),
                    HirBranch(None, else_body, {"kind": "else"}),
                ),
                span=_span(node),
            )
        if isinstance(node, ast.MatchExpr):
            scrutinee = self._lower_expr(node.scrutinee, env)
            branches = tuple(
                self._lower_match_arm(arm, env, index)
                for index, arm in enumerate(node.arms)
            )
            ty = branches[0].body.ty if branches else UNIT_TYPE
            return HirExpr(
                "match",
                ty,
                args=(scrutinee,),
                branches=branches,
                span=_span(node),
                meta={
                    "exhaustive": any(
                        branch.meta.get("pattern") == "_" for branch in branches
                    )
                },
            )
        if isinstance(node, ast.Block):
            return self._lower_block(node, env)
        if isinstance(node, ast.Loop):
            body = self._lower_block(node.body, env)
            return HirExpr(
                "loop", body.ty, args=(body,), span=_span(node), meta={"loop": "loop"}
            )
        if isinstance(node, ast.While):
            cond = self._lower_expr(node.cond, env)
            body = self._lower_block(node.body, env)
            return HirExpr(
                "loop",
                UNIT_TYPE,
                args=(cond, body),
                span=_span(node),
                meta={"loop": "while"},
            )
        if isinstance(node, ast.For):
            iterable = self._lower_expr(node.iter, env)
            local = dict(env)
            for name in _pattern_names(node.pattern):
                local[name] = UNKNOWN_TYPE
            body = self._lower_block(node.body, local)
            return HirExpr(
                "loop",
                UNIT_TYPE,
                args=(iterable, body),
                span=_span(node),
                meta={
                    "loop": "for",
                    "pattern": _pattern_repr(node.pattern),
                    "bindings": _pattern_names(node.pattern),
                },
            )
        if isinstance(node, ast.Return):
            value = None if node.value is None else self._lower_expr(node.value, env)
            return HirExpr(
                "return",
                UNIT_TYPE,
                args=() if value is None else (value,),
                span=_span(node),
            )
        if isinstance(node, ast.Break):
            value = None if node.value is None else self._lower_expr(node.value, env)
            return HirExpr(
                "break",
                UNIT_TYPE,
                args=() if value is None else (value,),
                span=_span(node),
                meta={"label": node.label},
            )
        if isinstance(node, ast.Continue):
            return HirExpr(
                "continue", UNIT_TYPE, span=_span(node), meta={"label": node.label}
            )
        if isinstance(node, ast.Field):
            return HirExpr(
                "field",
                UNKNOWN_TYPE,
                value=node.name,
                args=(self._lower_expr(node.receiver, env),),
                span=_span(node),
            )
        if isinstance(node, ast.Index):
            return HirExpr(
                "index",
                UNKNOWN_TYPE,
                args=(
                    self._lower_expr(node.receiver, env),
                    self._lower_expr(node.index, env),
                ),
                span=_span(node),
            )
        if isinstance(node, ast.Cast):
            return HirExpr(
                "cast",
                _type_from_ast(node.ty),
                args=(self._lower_expr(node.expr, env),),
                span=_span(node),
                meta={"target": _type_from_ast(node.ty).text},
            )
        if isinstance(node, ast.TupleExpr | ast.ArrayExpr):
            return HirExpr(
                "aggregate",
                UNKNOWN_TYPE,
                args=tuple(self._lower_expr(elem, env) for elem in node.elems),
                span=_span(node),
            )
        if isinstance(node, ast.RepeatArray):
            return HirExpr(
                "aggregate",
                UNKNOWN_TYPE,
                args=(
                    self._lower_expr(node.elem, env),
                    self._lower_expr(node.count, env),
                ),
                span=_span(node),
            )
        if isinstance(node, ast.StructLiteral):
            return HirExpr(
                "struct",
                HirType(node.path),
                args=tuple(self._lower_expr(value, env) for _, value in node.fields),
                span=_span(node),
                meta={"fields": tuple(name for name, _ in node.fields)},
            )
        if isinstance(node, ast.Closure):
            local = dict(env)
            for pattern, annotation in node.params:
                ty = _type_from_ast(annotation)
                for name in _pattern_names(pattern):
                    local[name] = ty
            return HirExpr(
                "closure",
                UNKNOWN_TYPE,
                args=(self._lower_expr(node.body, local),),
                span=_span(node),
            )
        if isinstance(node, ast.Range):
            edges = tuple(
                self._lower_expr(edge, env)
                for edge in (node.start, node.end)
                if edge is not None
            )
            return HirExpr(
                "range",
                UNKNOWN_TYPE,
                args=edges,
                span=_span(node),
                meta={"inclusive": node.inclusive},
            )
        if isinstance(node, ast.MacroCall):
            raw = node.args if isinstance(node.args, str) else ""
            args = (
                ()
                if isinstance(node.args, str)
                else tuple(self._lower_expr(arg, env) for arg in node.args)
            )
            return HirExpr(
                "call",
                UNKNOWN_TYPE,
                value=node.path,
                args=args,
                span=_span(node),
                meta={"macro": True, "tokens": raw},
            )
        if isinstance(node, ast.Await):
            return HirExpr(
                "await",
                UNKNOWN_TYPE,
                args=(self._lower_expr(node.expr, env),),
                span=_span(node),
            )
        if isinstance(node, ast.Try):
            return HirExpr(
                "try",
                UNKNOWN_TYPE,
                args=(self._lower_expr(node.expr, env),),
                span=_span(node),
            )
        children = tuple(
            self._lower_expr(child, env)
            for child in vars(node).values()
            if isinstance(child, _EXPR_TYPES)
        )
        return HirExpr(
            "op",
            UNKNOWN_TYPE,
            value=type(node).__name__,
            args=children,
            span=_span(node),
        )

    @staticmethod
    def _callee_name(node: ast.Expr) -> str:
        """Extract a stable callee name from an identifier or path."""
        if isinstance(node, ast.Ident):
            return node.name
        if isinstance(node, ast.Path):
            return "::".join(node.segments)
        return type(node).__name__

    @staticmethod
    def _analyze(functions: dict[str, HirFunction]) -> None:
        """Compute purity and recursion metadata to a fixed point."""
        for fn in functions.values():
            fn.pure = _is_pure_body(fn)

        changed = True
        while changed:
            changed = False
            for fn in functions.values():
                if not fn.pure:
                    continue
                for callee in _called_names(fn.body):
                    target = functions.get(callee)
                    if target is None or not target.pure:
                        fn.pure = False
                        changed = True
                        break

        graph = {
            name: sorted(_called_names(fn.body) & functions.keys())
            for name, fn in functions.items()
        }
        for name in _recursive_functions(graph):
            functions[name].recursive = True


def _walk_patterns(pattern: ast.Pattern | None) -> tuple[ast.Pattern, ...]:
    """Yield a pattern and all nested subpatterns."""
    if pattern is None:
        return ()
    children: tuple[ast.Pattern, ...] = ()
    if isinstance(pattern, ast.PatBinding | ast.PatRef | ast.PatBox):
        children = (pattern.inner,)
    elif isinstance(pattern, ast.PatTuple | ast.PatSlice):
        children = tuple(
            child for child in pattern.elems if isinstance(child, ast.Pattern)
        )
    elif isinstance(pattern, ast.PatOr):
        children = tuple(pattern.alts)
    elif isinstance(pattern, ast.PatEnum):
        children = tuple(pattern.args)
    elif isinstance(pattern, ast.PatStruct):
        children = tuple(sub for _, sub in pattern.fields)
    return (pattern, *children)


def _walk_hir(expr: HirExpr) -> tuple[HirExpr, ...]:
    """Yield an HIR expression and all nested expression nodes."""
    result = [expr]
    for arg in expr.args:
        result.extend(_walk_hir(arg))
    for branch in expr.branches:
        if branch.condition is not None:
            result.extend(_walk_hir(branch.condition))
        result.extend(_walk_hir(branch.body))
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt) and stmt.value is not None:
            result.extend(_walk_hir(stmt.value))
    return tuple(result)


def _called_names(expr: HirExpr) -> set[str]:
    """Collect callee names referenced by an HIR expression."""
    names: set[str] = set()
    for node in _walk_hir(expr):
        if node.kind in {"call", "await", "try"} and isinstance(node.value, str):
            names.add(node.value)
    return names


def _is_pure_body(fn: HirFunction) -> bool:
    """Return whether a function body is free of modeled side effects."""
    if fn.meta.get("unsafe") or "no_inline" in fn.meta.get("attributes", ()):
        return False
    for node in _walk_hir(fn.body):
        if node.kind in {"assign", "store"}:
            return False
        if node.kind in {"call", "await", "try"} and isinstance(node.value, str):
            if node.value in IMPURE_NAMES:
                return False
            if node.value not in PURE_BUILTINS and node.value not in _KNOWN_CALLEES:
                return False
    return True


_KNOWN_CALLEES: set[str] = set()


def _recursive_functions(graph: dict[str, list[str]]) -> set[str]:
    """Return every function that participates in a call cycle."""
    color: dict[str, int] = {}
    active: list[str] = []
    recursive: set[str] = set()

    def visit(name: str) -> None:
        color[name] = 1
        active.append(name)
        for callee in graph.get(name, ()):
            if color.get(callee, 0) == 0:
                visit(callee)
            elif color.get(callee) == 1 and callee in active:
                recursive.update(active[active.index(callee) :])
        active.pop()
        color[name] = 2

    for name in sorted(graph):
        if color.get(name, 0) == 0:
            visit(name)
    return recursive


def ir_lower_ast_to_hir(source: str | ast.Program) -> HirModule:
    """Lower surface AST constructs into a typed high-level IR.

    ``source`` may be Polyglot source text or an already parsed
    :class:`ast_nodes.Program`.  Short-circuit operators, loops, and matches are
    preserved in HIR; :func:`ir_lower_hir_to_mir` desugars them into CFG blocks.
    """
    program = _program_from_input(source)
    _KNOWN_CALLEES.clear()
    _KNOWN_CALLEES.update(
        item.name for item in program.items if isinstance(item, ast.FnDecl)
    )
    return _Lowerer(program).lower()


# ---------------------------------------------------------------------------
# HIR -> MIR
# ---------------------------------------------------------------------------


class _MirBuilder:
    """Build one function CFG from lowered HIR."""

    def __init__(self, function: HirFunction) -> None:
        self.function = function
        self._labels = itertools.count(1)
        self._temps = itertools.count(1)
        self.blocks: dict[str, BasicBlock] = {}
        self.entry = self._new_block("entry")
        self._break_stack: list[tuple[str, str]] = []
        self._continue_stack: list[str] = []

    @property
    def cfg(self) -> Cfg:
        """Return the CFG built so far."""
        return Cfg(self.entry, self.blocks)

    @property
    def current(self) -> BasicBlock:
        """Return the block receiving new instructions."""
        return self.blocks[self._active]

    def _new_block(self, hint: str, meta: dict[str, Any] | None = None) -> str:
        """Allocate a deterministic block label."""
        label = f"{hint}_{next(self._labels)}"
        self.blocks[label] = BasicBlock(label, meta=dict(meta or {}))
        return label

    def _switch(self, label: str) -> None:
        """Make ``label`` the active insertion point."""
        if label not in self.blocks:
            raise ValueError(f"unknown basic block {label!r}")
        self._active = label

    def _temp(self, ty: HirType) -> str:
        """Allocate a deterministic temporary name."""
        return f"t{next(self._temps)}"

    def _emit(self, instr: MirInstr) -> None:
        """Append an instruction to the active block."""
        self.current.instructions.append(instr)

    def _terminate(
        self,
        kind: str,
        targets: tuple[str, ...] = (),
        cond: MirOperand | None = None,
        value: MirOperand | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Terminate the active block once; later requests are ignored."""
        if self.current.terminator.kind:
            return
        self.current.terminator = MirTerminator(
            kind, targets, cond, value, dict(meta or {})
        )

    def build(self) -> Cfg:
        """Lower the function body and close any unterminated tail block."""
        self._switch(self.entry)
        result = self._lower_expr(self.function.body)
        if not self.current.terminator.kind:
            value = result if result.kind != "unit" else _unit()
            self._terminate("return", value=value)
        return self.cfg

    def _lower_stmt(self, stmt: HirStmt) -> None:
        """Emit MIR instructions for one HIR statement."""
        if stmt.kind == "let":
            value = self._lower_expr(stmt.value) if stmt.value is not None else _unit()
            name = stmt.name or "_"
            op = "let"
            if value.kind == "var" and value.ty.owned:
                op = "move"
            if value.kind == "unit":
                op = "let"
            self._emit(
                MirInstr(
                    op,
                    dest=name,
                    args=(value,),
                    meta={
                        "scope": stmt.meta.get("scope"),
                        "mutable": stmt.mutable,
                        "bindings": tuple(stmt.meta.get("bindings", ())),
                        "ty": stmt.ty,
                    },
                    span=stmt.span,
                )
            )
            return
        if stmt.kind == "expr":
            if stmt.value is not None:
                self._lower_expr(stmt.value)
            return
        if stmt.kind == "errdefer":
            if stmt.value is not None:
                self._lower_expr(stmt.value)
            self._emit(
                MirInstr(
                    "errdefer",
                    meta={"scope": stmt.meta.get("scope")},
                    span=stmt.span,
                )
            )
            return
        self._emit(MirInstr("op", meta={"node": stmt.kind}, span=stmt.span))

    def _lower_block(self, expr: HirExpr) -> MirOperand:
        """Lower a block expression, emitting its statements and tail."""
        for stmt in expr.meta.get("stmts", ()):
            if isinstance(stmt, HirStmt):
                self._lower_stmt(stmt)
        if expr.args:
            return self._lower_expr(expr.args[0])
        return _unit()

    def _lower_logical(self, expr: HirExpr) -> MirOperand:
        """Desugar ``&&`` / ``||`` into branch, rhs, and join blocks."""
        is_and = expr.kind == "logical_and"
        left = self._lower_expr(expr.args[0])
        lhs_end = self.current.label
        rhs_label = self._new_block("and_rhs" if is_and else "or_rhs")
        join_label = self._new_block("and_join" if is_and else "or_join")
        if is_and:
            self._terminate("branch", (rhs_label, join_label), cond=left)
        else:
            self._terminate("branch", (join_label, rhs_label), cond=left)

        self._switch(rhs_label)
        right = self._lower_expr(expr.args[1])
        rhs_end = self.current.label
        self._terminate("jump", (join_label,), value=right)

        self._switch(join_label)
        dest = self._temp("guard_result")
        shortcut = _const(False, BOOL_TYPE) if is_and else _const(True, BOOL_TYPE)
        self._emit(
            MirInstr(
                "phi",
                dest=dest,
                args=(shortcut, right),
                meta={"blocks": (lhs_end, rhs_end), "short_circuit": expr.value},
            )
        )
        return _var(dest, BOOL_TYPE)

    def _lower_if(self, expr: HirExpr) -> MirOperand:
        """Lower an if expression into a diamond CFG with a phi result."""
        cond = self._lower_expr(expr.args[0])
        then_label = self._new_block("if_then")
        else_label = self._new_block("if_else")
        join_label = self._new_block("if_join")
        self._terminate("branch", (then_label, else_label), cond=cond)
        incoming: list[MirOperand] = []
        incoming_blocks: list[str] = []

        self._switch(then_label)
        then_value = self._lower_expr(expr.branches[0].body)
        if not self.current.terminator.kind:
            end = self.current.label
            self._terminate("jump", (join_label,), value=then_value)
            incoming.append(then_value)
            incoming_blocks.append(end)

        self._switch(else_label)
        else_value = (
            self._lower_expr(expr.branches[1].body)
            if len(expr.branches) > 1
            else _unit()
        )
        if not self.current.terminator.kind:
            end = self.current.label
            self._terminate("jump", (join_label,), value=else_value)
            incoming.append(else_value)
            incoming_blocks.append(end)

        self._switch(join_label)
        if not incoming:
            return _unit()
        dest = self._temp("if_result")
        self._emit(
            MirInstr(
                "phi",
                dest=dest,
                args=tuple(incoming),
                meta={"blocks": tuple(incoming_blocks)},
            )
        )
        return _var(dest, expr.ty)

    def _lower_match(self, expr: HirExpr) -> MirOperand:
        """Lower a match into a chain of pattern-test guard blocks."""
        scrutinee = self._lower_expr(expr.args[0])
        scrutinee_slot = self._temp("scrutinee")
        self._emit(MirInstr("copy", dest=scrutinee_slot, args=(scrutinee,)))
        join_label = self._new_block("match_join")
        test_label = self._new_block("match_test")
        self._terminate("jump", (test_label,))
        incoming: list[MirOperand] = []
        incoming_blocks: list[str] = []

        for branch in expr.branches:
            self._switch(test_label)
            arm_label = self._new_block(
                "match_arm",
                {
                    "match_arm": branch.meta.get("arm"),
                    "pattern": branch.meta.get("pattern", "_"),
                },
            )
            if branch.condition is None:
                self._terminate("jump", (arm_label,))
            else:
                condition = self._lower_expr(branch.condition)
                head = self.current.label
                next_label = self._new_block("match_test", {"match": True})
                self._terminate(
                    "branch",
                    (arm_label, next_label),
                    cond=condition,
                    meta={"match": True, "pattern": branch.meta.get("pattern")},
                )
                self.blocks[head].meta["match_test"] = branch.meta.get("arm")
                test_label = next_label

            self._switch(arm_label)
            value = self._lower_expr(branch.body)
            if not self.current.terminator.kind:
                end = self.current.label
                self._terminate("jump", (join_label,), value=value)
                incoming.append(value)
                incoming_blocks.append(end)

        self._switch(test_label)
        self._terminate(
            "unreachable",
            meta={"no_match": True, "exhaustive": expr.meta.get("exhaustive", False)},
        )
        self._switch(join_label)
        if not incoming:
            return _unit()
        dest = self._temp("match_result")
        self._emit(
            MirInstr(
                "phi",
                dest=dest,
                args=tuple(incoming),
                meta={"blocks": tuple(incoming_blocks)},
            )
        )
        return _var(dest, expr.ty)

    def _lower_loop(self, expr: HirExpr) -> MirOperand:
        """Lower loop/while/for into header, body, and exit blocks."""
        kind = expr.meta.get("loop", "loop")
        if kind == "loop":
            head = self._new_block("loop_head")
            exit_label = self._new_block("loop_exit")
            self._terminate("jump", (head,))
            self._switch(head)
            slot = self._temp("loop_value")
            self._break_stack.append((exit_label, slot))
            self._continue_stack.append(head)
            self._lower_expr(expr.args[0])
            if not self.current.terminator.kind:
                self._terminate("jump", (head,))
            self._break_stack.pop()
            self._continue_stack.pop()
            self._switch(exit_label)
            result = self._temp("loop_result")
            self._emit(MirInstr("copy", dest=result, args=(_var(slot, expr.ty),)))
            return _var(result, expr.ty)

        if kind == "while":
            head = self._new_block("while_head")
            body = self._new_block("while_body")
            exit_label = self._new_block("while_exit")
            self._terminate("jump", (head,))
            self._switch(head)
            cond = self._lower_expr(expr.args[0])
            self._terminate("branch", (body, exit_label), cond=cond)
            self._switch(body)
            slot = self._temp("loop_value")
            self._break_stack.append((exit_label, slot))
            self._continue_stack.append(head)
            self._lower_expr(expr.args[1])
            if not self.current.terminator.kind:
                self._terminate("jump", (head,))
            self._break_stack.pop()
            self._continue_stack.pop()
            self._switch(exit_label)
            return _unit()

        iterable = self._lower_expr(expr.args[0])
        head = self._new_block("for_head")
        body = self._new_block("for_body")
        exit_label = self._new_block("for_exit")
        self._terminate("jump", (head,))
        self._switch(head)
        item = self._temp("item")
        has_next = self._temp("has_next")
        self._emit(
            MirInstr(
                "iter_next",
                dest=item,
                args=(iterable,),
                meta={"has_next": has_next},
            )
        )
        self._terminate("branch", (body, exit_label), cond=_var(has_next, BOOL_TYPE))
        self._switch(body)
        pattern = str(expr.meta.get("pattern", "_"))
        if pattern != "_":
            self._emit(MirInstr("bind", dest=pattern, args=(_var(item),)))
        slot = self._temp("loop_value")
        self._break_stack.append((exit_label, slot))
        self._continue_stack.append(head)
        self._lower_expr(expr.args[1])
        if not self.current.terminator.kind:
            self._terminate("jump", (head,))
        self._break_stack.pop()
        self._continue_stack.pop()
        self._switch(exit_label)
        return _unit()

    def _lower_expr(self, expr: HirExpr, dest: str | None = None) -> MirOperand:
        """Lower one HIR expression, returning the operand holding its value."""
        if expr.kind == "literal":
            value = _const(expr.value, expr.ty)
            if dest:
                self._emit(MirInstr("const", dest=dest, args=(value,), span=expr.span))
                return _var(dest, expr.ty)
            return value
        if expr.kind == "var":
            operand = _var(str(expr.value), expr.ty)
            if dest:
                self._emit(MirInstr("copy", dest=dest, args=(operand,), span=expr.span))
                return _var(dest, expr.ty)
            return operand
        if expr.kind == "path":
            return _const(str(expr.value), expr.ty)
        if expr.kind == "block":
            value = self._lower_block(expr)
            if dest and value.kind != "unit":
                self._emit(MirInstr("copy", dest=dest, args=(value,)))
                return _var(dest, expr.ty)
            return value
        if expr.kind in {"logical_and", "logical_or"}:
            value = self._lower_logical(expr)
        elif expr.kind == "if":
            value = self._lower_if(expr)
        elif expr.kind == "match":
            value = self._lower_match(expr)
        elif expr.kind == "loop":
            value = self._lower_loop(expr)
        elif expr.kind == "return":
            value = self._lower_expr(expr.args[0]) if expr.args else _unit()
            self._terminate("return", value=value)
            return value
        elif expr.kind == "break":
            value = self._lower_expr(expr.args[0]) if expr.args else _unit()
            if self._break_stack:
                exit_label, slot = self._break_stack[-1]
                self._emit(MirInstr("copy", dest=slot, args=(value,), span=expr.span))
                self._terminate("jump", (exit_label,), value=value)
            else:
                self._terminate("unreachable", meta={"break_outside_loop": True})
            return _unit()
        elif expr.kind == "continue":
            if self._continue_stack:
                self._terminate("jump", (self._continue_stack[-1],))
            else:
                self._terminate("unreachable", meta={"continue_outside_loop": True})
            return _unit()
        elif expr.kind == "assign":
            value = self._lower_expr(expr.args[0]) if expr.args else _unit()
            target = str(expr.meta.get("target", ""))
            self._emit(
                MirInstr(
                    "store",
                    args=(value,),
                    meta={
                        "target": target,
                        "compound": expr.meta.get("compound", False),
                        "op": expr.value,
                    },
                    span=expr.span,
                )
            )
            return _unit()
        elif expr.kind == "guard":
            condition = (
                self._lower_expr(expr.args[0]) if expr.args else _const(True, BOOL_TYPE)
            )
            self._emit(
                MirInstr(
                    "guard",
                    args=(condition,),
                    meta={
                        "name": expr.value or "guard",
                        "message": expr.meta.get("message", ""),
                    },
                    span=expr.span,
                )
            )
            return _unit()
        elif expr.kind == "pattern_test":
            scrutinee = self._lower_expr(expr.args[0]) if expr.args else _unit()
            temp = self._temp("pattern")
            self._emit(
                MirInstr(
                    "test",
                    dest=temp,
                    args=(scrutinee,),
                    meta={"pattern": str(expr.value), "invariant": True},
                    span=expr.span,
                )
            )
            return _var(temp, BOOL_TYPE)
        else:
            children = tuple(self._lower_expr(arg) for arg in expr.args)
            op = {
                "bin": "binop",
                "unop": "unop",
                "call": "call",
                "field": "field",
                "index": "index",
                "cast": "cast",
                "aggregate": "aggregate",
                "struct": "aggregate",
                "range": "range",
                "closure": "closure",
                "await": "await",
                "try": "try",
                "op": "op",
            }.get(expr.kind, expr.kind)
            meta = dict(expr.meta)
            if expr.kind in {"bin", "unop"}:
                meta["operator"] = expr.value
            if expr.kind == "call":
                meta["callee"] = expr.value
            temp = dest or self._temp(expr.kind)
            self._emit(
                MirInstr(
                    op,
                    dest=temp,
                    args=children,
                    meta=meta,
                    span=expr.span,
                )
            )
            return _var(temp, expr.ty)

        if dest and value.kind != "unit" and not self.current.terminator.kind:
            self._emit(MirInstr("copy", dest=dest, args=(value,)))
            return _var(dest, expr.ty)
        return value


def ir_lower_hir_to_mir(module: HirModule) -> MirModule:
    """Desugar HIR loops, matches, and short-circuiting into explicit CFG blocks.

    Each function is lowered independently.  Block and temporary names are
    allocated from deterministic per-function counters, so repeated lowering of
    the same module produces structurally identical MIR.
    """
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        cfg = _MirBuilder(function).build()
        functions[name] = MirFunction(
            name=name,
            cfg=cfg,
            params=function.params,
            ret=function.ret,
            pure=function.pure,
            recursive=function.recursive,
            meta=dict(function.meta),
        )
    return MirModule(functions, module.source)


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

_BINARY_OPERATORS: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}


def _eval_binary(op: str, left: Any, right: Any) -> tuple[bool, Any]:
    """Evaluate a pure binary operator on two literal values."""
    func = _BINARY_OPERATORS.get(op)
    if func is None:
        return False, None
    if isinstance(left, bool) != isinstance(right, bool) and op in {"&", "|", "^"}:
        return False, None
    try:
        return True, func(left, right)
    except ArithmeticError, TypeError, ValueError, OverflowError:
        return False, None


def _eval_unary(op: str, value: Any) -> tuple[bool, Any]:
    """Evaluate a pure unary operator on a literal value."""
    try:
        if op == "-":
            return True, -value
        if op == "!":
            return True, not value
        if op == "+":
            return True, +value
    except ArithmeticError, TypeError, ValueError:
        return False, None
    return False, None


def _fold_hir_expr(expr: HirExpr, log: list[str]) -> HirExpr:
    """Rewrite one HIR expression, folding pure literal-only nodes."""
    if expr.kind == "literal":
        return expr
    args = tuple(_fold_hir_expr(arg, log) for arg in expr.args)
    branches = tuple(
        HirBranch(
            None if branch.condition is None else _fold_hir_expr(branch.condition, log),
            _fold_hir_expr(branch.body, log),
            dict(branch.meta),
        )
        for branch in expr.branches
    )
    stmts: list[HirStmt] = []
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt) and stmt.value is not None:
            stmts.append(replace(stmt, value=_fold_hir_expr(stmt.value, log)))
        elif isinstance(stmt, HirStmt):
            stmts.append(stmt)
    meta = dict(expr.meta)
    if stmts or "stmts" in expr.meta:
        meta["stmts"] = tuple(stmts)
    current = replace(expr, args=args, branches=branches, meta=meta)

    if current.kind in {"bin", "logical_and", "logical_or", "unop", "cast"}:
        if current.kind == "unop":
            ok, value = _eval_unary(str(current.value), current.args[0].value)
        elif current.kind == "cast":
            ok, value = True, current.args[0].value
        else:
            ok, value = _eval_binary(
                str(current.value), current.args[0].value, current.args[1].value
            )
        if ok and all(arg.kind == "literal" for arg in current.args):
            log.append(f"{current.kind}:{current.value} -> {value!r}")
            return HirExpr("literal", current.ty, value=value, span=current.span)

    if current.kind == "if" and current.args[0].kind == "literal":
        index = 0 if current.args[0].value else 1
        if index < len(current.branches):
            chosen = current.branches[index].body
            log.append(f"if:constant -> block {index}")
            return chosen

    if current.kind == "match" and current.args[0].kind == "literal":
        wanted = repr(current.args[0].value)
        for branch in current.branches:
            if branch.meta.get("pattern") in {wanted, "_"} and branch.condition is None:
                log.append(f"match:constant -> arm {branch.meta.get('arm')}")
                return branch.body
    return current


def _fold_mir_module(module: MirModule, log: list[str]) -> MirModule:
    """Fold constant operand instructions in each MIR function."""
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        blocks: dict[str, BasicBlock] = {}
        for label, block in function.cfg.blocks.items():
            instructions: list[MirInstr] = []
            propagated: dict[str, MirOperand] = {}
            for instr in block.instructions:
                args = tuple(
                    propagated.get(str(arg.value), arg) if arg.kind == "var" else arg
                    for arg in instr.args
                )
                instr = replace(instr, args=args)
                if (
                    instr.op in {"binop", "unop"}
                    and instr.args
                    and all(arg.kind == "const" for arg in instr.args)
                ):
                    if instr.op == "unop":
                        ok, value = _eval_unary(
                            str(instr.meta.get("operator")), instr.args[0].value
                        )
                    else:
                        ok, value = _eval_binary(
                            str(instr.meta.get("operator")),
                            instr.args[0].value,
                            instr.args[1].value,
                        )
                    if ok:
                        log.append(f"{name}:{label}:{instr.op} -> {value!r}")
                        constant = _const(value, instr.args[0].ty)
                        instructions.append(
                            MirInstr(
                                "const",
                                dest=instr.dest,
                                args=(constant,),
                                meta=dict(instr.meta),
                                span=instr.span,
                            )
                        )
                        if instr.dest:
                            propagated[instr.dest] = constant
                        continue
                instructions.append(instr)
            blocks[label] = BasicBlock(
                label, instructions, block.terminator, dict(block.meta)
            )
        cfg = Cfg(function.cfg.entry, blocks)
        functions[name] = replace(function, cfg=cfg)
    return MirModule(functions, module.source)


def ir_fold_constant_expressions(module: HirModule | MirModule) -> FoldResult:
    """Evaluate pure compile-time constant arithmetic and boolean expressions.

    No user code is executed: only a closed table of arithmetic, comparison,
    bitwise, and unary operators is consulted.  Division by zero and mixed-type
    operators are left untouched so the backend can diagnose them.
    """
    log: list[str] = []
    if isinstance(module, HirModule):
        functions = {
            name: replace(function, body=_fold_hir_expr(function.body, log))
            for name, function in module.functions.items()
        }
        result: HirModule | MirModule = HirModule(functions, module.source)
    elif isinstance(module, MirModule):
        result = _fold_mir_module(module, log)
    else:
        raise TypeError("expected a HirModule or MirModule")
    return FoldResult(result, tuple(log))


# ---------------------------------------------------------------------------
# Dead block / dead assignment elimination
# ---------------------------------------------------------------------------

_ASSIGN_OPS = frozenset(
    {
        "const",
        "copy",
        "binop",
        "unop",
        "call",
        "field",
        "index",
        "cast",
        "aggregate",
        "test",
        "phi",
        "let",
        "move",
        "iter_next",
    }
)


def _read_vars(cfg: Cfg) -> set[str]:
    """Collect every variable read by instructions and terminators."""
    reads: set[str] = set()
    for block in cfg.blocks.values():
        for instr in block.instructions:
            for operand in instr.args:
                if operand.kind == "var":
                    reads.add(str(operand.value))
            if instr.op == "store" and isinstance(instr.meta.get("target"), str):
                reads.add(instr.meta["target"])
            if instr.op == "iter_next" and isinstance(instr.meta.get("has_next"), str):
                reads.add(instr.meta["has_next"])
        for terminator in (block.terminator,):
            for operand in (terminator.cond, terminator.value):
                if operand is not None and operand.kind == "var":
                    reads.add(str(operand.value))
    return reads


def ir_eliminate_dead_blocks(module: MirModule) -> EliminationResult:
    """Prune unreachable basic blocks and dead local assignments.

    Reachability starts at each function's entry.  Assignment removal iterates
    to a fixed point, so chains such as ``t1 = 1; t2 = t1`` disappear when
    neither value is observed.
    """
    removed_blocks: list[str] = []
    removed_assignments: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        reachable = cfg.reachable()
        dropped = sorted(set(cfg.blocks) - reachable)
        removed_blocks.extend(f"{name}:{label}" for label in dropped)
        blocks = {
            label: BasicBlock(
                label,
                list(cfg.blocks[label].instructions),
                cfg.blocks[label].terminator,
                dict(cfg.blocks[label].meta),
            )
            for label in cfg.blocks
            if label in reachable
        }
        new_cfg = Cfg(cfg.entry, blocks)
        if cfg.entry not in blocks:
            functions[name] = function
            continue

        while True:
            reads = _read_vars(new_cfg)
            removed_here: list[str] = []
            for label in sorted(new_cfg.blocks):
                block = new_cfg.blocks[label]
                kept: list[MirInstr] = []
                for instr in block.instructions:
                    if (
                        instr.dest
                        and instr.dest not in reads
                        and instr.op in _ASSIGN_OPS
                        and not instr.meta.get("side_effects")
                    ):
                        removed_here.append(f"{name}:{label}:{instr.dest}")
                    else:
                        kept.append(instr)
                block.instructions = kept
            if not removed_here:
                break
            removed_assignments.extend(removed_here)
        functions[name] = replace(function, cfg=new_cfg)
    return EliminationResult(
        MirModule(functions, module.source),
        tuple(removed_blocks),
        tuple(removed_assignments),
    )


# ---------------------------------------------------------------------------
# CFG validation
# ---------------------------------------------------------------------------

_TERMINATOR_ARITY = {
    "jump": 1,
    "branch": 2,
    "switch": (1, None),
    "return": 0,
    "unreachable": 0,
}


def _dominators(cfg: Cfg) -> dict[str, set[str]]:
    """Compute dominator sets over blocks reachable from the entry."""
    reachable = cfg.reachable()
    if not reachable:
        return {}
    predecessors = cfg.predecessors()
    dominators: dict[str, set[str]] = {label: set(reachable) for label in reachable}
    dominators[cfg.entry] = {cfg.entry}
    changed = True
    while changed:
        changed = False
        for label in sorted(reachable):
            if label == cfg.entry:
                continue
            preds = [pred for pred in predecessors[label] if pred in reachable]
            common = set(reachable)
            for pred in preds:
                common &= dominators.get(pred, set())
            candidate = {label, *common}
            if candidate != dominators[label]:
                dominators[label] = candidate
                changed = True
    return dominators


def _strongly_connected(cfg: Cfg) -> list[set[str]]:
    """Return strongly connected components using iterative Tarjan."""
    reachable = cfg.reachable()
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    for root in sorted(reachable):
        if root in indices:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, position = work[-1]
            if position == 0:
                indices[node] = low[node] = index
                index += 1
                stack.append(node)
                on_stack.add(node)
            successors = [
                target for target in cfg.successors(node) if target in reachable
            ]
            if position < len(successors):
                work[-1] = (node, position + 1)
                target = successors[position]
                if target not in indices:
                    work.append((target, 0))
                elif target in on_stack:
                    low[node] = min(low[node], indices[target])
                continue
            work.pop()
            if low[node] == indices[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                components.append(component)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return components


def _validate_cfg(
    cfg: Cfg, prefix: str
) -> tuple[ValidationResult, dict[str, set[str]]]:
    """Validate one CFG and return its dominator map for shared checks."""
    issues: list[CfgIssue] = []
    if cfg.entry not in cfg.blocks:
        issues.append(CfgIssue("MissingEntry", cfg.entry, "entry block is missing"))
        return ValidationResult(False, tuple(issues)), {}

    reachable = cfg.reachable()
    for label in sorted(cfg.blocks):
        block = cfg.blocks[label]
        if label not in reachable:
            issues.append(
                CfgIssue("UnreachableBlock", label, "block is unreachable", "warning")
            )
            continue
        terminator = block.terminator
        if not terminator.kind:
            issues.append(
                CfgIssue("UnterminatedBlock", label, "block has no terminator")
            )
            continue
        expected = _TERMINATOR_ARITY.get(terminator.kind)
        if expected is None:
            issues.append(
                CfgIssue(
                    "UnknownTerminator",
                    label,
                    f"unknown terminator {terminator.kind!r}",
                )
            )
            continue
        low, high = (expected, expected) if isinstance(expected, int) else expected
        if not (low <= len(terminator.targets) <= high):
            issues.append(
                CfgIssue(
                    "BadTerminatorArity",
                    label,
                    f"{terminator.kind} expects {low}..{high} targets, got {len(terminator.targets)}",
                )
            )
        for target in terminator.targets:
            if target not in cfg.blocks:
                issues.append(
                    CfgIssue(
                        "DanglingTarget", label, f"target {target!r} does not exist"
                    )
                )
        if terminator.kind == "branch" and terminator.cond is None:
            issues.append(
                CfgIssue("MissingBranchCondition", label, "branch has no condition")
            )

    dominators = _dominators(cfg)
    for label in sorted(reachable):
        if cfg.entry not in dominators.get(label, set()):
            issues.append(
                CfgIssue("UndominatedBlock", label, "entry does not dominate block")
            )

    for component in _strongly_connected(cfg):
        if len(component) < 2:
            continue
        if not any(
            all(header in dominators.get(member, set()) for member in component)
            for header in component
        ):
            issues.append(
                CfgIssue(
                    "MissingLoopHeader",
                    ",".join(sorted(component)),
                    "cycle has no block dominating every member",
                )
            )
    valid = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(valid, tuple(issues)), dominators


def _as_mir_module(target: MirModule | MirFunction | Cfg) -> MirModule:
    """Normalize a MIR target into a module for shared pass logic."""
    if isinstance(target, MirModule):
        return target
    if isinstance(target, MirFunction):
        return MirModule({target.name: target})
    if isinstance(target, Cfg):
        function = MirFunction("<cfg>", target)
        return MirModule({function.name: function})
    raise TypeError("expected a MirModule, MirFunction, or Cfg")


def ir_validate_control_flow_graph(
    target: MirModule | MirFunction | Cfg,
) -> ValidationResult:
    """Verify block termination, dominance, and loop headers in a MIR CFG.

    Structural errors (missing targets, unterminated blocks, irreducible cycles)
    make the result invalid; unreachable blocks are reported as warnings so
    :func:`ir_eliminate_dead_blocks` can be run separately.
    """
    module = _as_mir_module(target)
    issues: list[CfgIssue] = []
    for name, function in module.functions.items():
        result, _ = _validate_cfg(function.cfg, name)
        issues.extend(
            replace(issue, block=f"{name}:{issue.block}") for issue in result.issues
        )
    valid = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(valid, tuple(issues))


# ---------------------------------------------------------------------------
# Borrow / move analysis
# ---------------------------------------------------------------------------

_MOVE_OPS = frozenset({"move", "take"})
_BORROW_OPS = frozenset({"borrow", "borrow_mut"})


def _variable_types(cfg: Cfg) -> dict[str, HirType]:
    """Infer variable types from operand uses in one CFG."""
    types: dict[str, HirType] = {}
    for block in cfg.blocks.values():
        for instr in block.instructions:
            declared = instr.meta.get("ty")
            if instr.dest and isinstance(declared, HirType):
                types.setdefault(instr.dest, declared)
            for operand in instr.args:
                if operand.kind == "var" and operand.ty != UNKNOWN_TYPE:
                    types.setdefault(str(operand.value), operand.ty)
    return types


def _check_operand(
    operand: MirOperand,
    states: dict[str, str],
    prefix: str,
    block: str,
    issues: list[BorrowIssue],
) -> None:
    """Report invalid uses of a variable in the current state."""
    if operand.kind != "var":
        return
    name = str(operand.value)
    state = states.get(name, "live")
    if state in {"moved", "maybe_moved", "dropped"}:
        issues.append(
            BorrowIssue(
                "UseAfterMove",
                name,
                block,
                f"{prefix}: use of {name} after move ({state})",
            )
        )
    elif state in {"mut_borrowed", "maybe_mut_borrowed", "conflict"}:
        issues.append(
            BorrowIssue(
                "BorrowConflict", name, block, f"{prefix}: use while mutably borrowed"
            )
        )


def _merge_states(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Join two branch states into a conservative deterministic state."""
    result: dict[str, str] = {}
    for name in sorted(set(left) | set(right)):
        a = left.get(name, "live")
        b = right.get(name, "live")
        if a == b:
            result[name] = a
        elif "conflict" in {a, b}:
            result[name] = "conflict"
        elif {"moved", "dropped"} & {a, b}:
            result[name] = "maybe_moved"
        elif "mut_borrowed" in {a, b} and "borrowed" in {a, b}:
            result[name] = "conflict"
        elif "mut_borrowed" in {a, b}:
            result[name] = "maybe_mut_borrowed"
        elif "borrowed" in {a, b}:
            result[name] = "maybe_borrowed"
        else:
            result[name] = "unknown"
    return result


def _borrow_module(module: MirModule) -> BorrowReport:
    """Run borrow/move dataflow over every function in a MIR module."""
    issues: list[BorrowIssue] = []
    states_out: dict[str, dict[str, str]] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        reachable = cfg.reachable()
        if cfg.entry not in reachable:
            issues.append(
                BorrowIssue(
                    "MissingEntry", "<entry>", cfg.entry, "entry is unreachable"
                )
            )
            continue
        types = _variable_types(cfg)
        incoming: dict[str, dict[str, str] | None] = {
            label: None for label in sorted(reachable)
        }
        visited: set[str] = set()
        work: deque[str] = deque(sorted(reachable))
        while work:
            label = work.popleft()
            states = dict(incoming[label] or {})
            for instr in cfg.blocks[label].instructions:
                if instr.op in _BORROW_OPS:
                    for operand in instr.args:
                        _check_operand(
                            operand, states, instr.op, f"{name}:{label}", issues
                        )
                        if operand.kind == "var":
                            source = str(operand.value)
                            if states.get(source, "live") in {
                                "moved",
                                "maybe_moved",
                                "mut_borrowed",
                                "conflict",
                            }:
                                issues.append(
                                    BorrowIssue(
                                        "BorrowConflict",
                                        source,
                                        f"{name}:{label}",
                                        f"cannot borrow {source} in its current state",
                                    )
                                )
                            else:
                                states[source] = (
                                    "mut_borrowed"
                                    if instr.op == "borrow_mut"
                                    else "borrowed"
                                )
                    continue
                for operand in instr.args:
                    _check_operand(operand, states, instr.op, f"{name}:{label}", issues)
                if instr.op == "drop":
                    for operand in instr.args:
                        if operand.kind == "var":
                            source = str(operand.value)
                            if states.get(source, "live") in {
                                "moved",
                                "maybe_moved",
                                "dropped",
                            }:
                                issues.append(
                                    BorrowIssue(
                                        "DoubleDrop",
                                        source,
                                        f"{name}:{label}",
                                        f"drop of consumed {source}",
                                    )
                                )
                            states[source] = "dropped"
                    continue
                if instr.dest:
                    if instr.op in _MOVE_OPS:
                        for operand in instr.args:
                            if operand.kind == "var":
                                states[str(operand.value)] = "moved"
                    elif instr.op in {"let", "copy"} and instr.args:
                        source = instr.args[0]
                        if (
                            source.kind == "var"
                            and source.ty.owned
                            and instr.op == "copy"
                            and types.get(str(source.value), source.ty).owned
                        ):
                            states[str(source.value)] = "moved"
                    if instr.op == "call":
                        for moved in instr.meta.get("moves", ()):
                            states[str(moved)] = "moved"
                    states[instr.dest] = "live"

            terminator = function.cfg.blocks[label].terminator
            if terminator.cond is not None:
                _check_operand(
                    terminator.cond, states, "branch", f"{name}:{label}", issues
                )
            if terminator.value is not None:
                _check_operand(
                    terminator.value, states, "return", f"{name}:{label}", issues
                )

            old = states_out.get(f"{name}:{label}", {})
            merged = _merge_states(old, states) if old else states
            if merged != old or label not in visited:
                visited.add(label)
                states_out[f"{name}:{label}"] = merged
                for target in cfg.successors(label):
                    if target not in reachable:
                        continue
                    before = incoming[target]
                    joined = (
                        dict(merged)
                        if before is None
                        else _merge_states(before, merged)
                    )
                    if joined != before:
                        incoming[target] = joined
                        work.append(target)
            elif f"{name}:{label}" not in states_out:
                states_out[f"{name}:{label}"] = states
    return BorrowReport(not issues, tuple(issues), states_out)


def ir_check_borrow_and_move_state(target: MirModule | MirFunction) -> BorrowReport:
    """Track moves, borrows, and use-after-move across all CFG branches.

    Branch joins merge states conservatively (``live`` + ``moved`` becomes
    ``maybe_moved``).  A variable of an owned type copied by value is treated as
    moved; ordinary scalar copies are not.
    """
    return _borrow_module(_as_mir_module(target))


# ---------------------------------------------------------------------------
# RAII drop insertion
# ---------------------------------------------------------------------------

_DEF_OPS = frozenset(
    {
        "let",
        "const",
        "copy",
        "binop",
        "unop",
        "call",
        "field",
        "index",
        "cast",
        "aggregate",
        "test",
        "phi",
        "iter_next",
        "move",
    }
)


def _liveness(cfg: Cfg) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Compute live-in and live-out variable sets for each block."""
    reachable = cfg.reachable()
    live_in = {label: set[str]() for label in reachable}
    live_out = {label: set[str]() for label in reachable}
    changed = True
    while changed:
        changed = False
        for label in sorted(reachable, reverse=True):
            block = cfg.blocks[label]
            new_out: set[str] = set()
            for target in block.terminator.targets:
                new_out |= live_in.get(target, set())
            defs: set[str] = set()
            uses: set[str] = set()
            for instr in block.instructions:
                for operand in instr.args:
                    if operand.kind == "var":
                        uses.add(str(operand.value))
                if instr.op == "store" and isinstance(instr.meta.get("target"), str):
                    uses.add(instr.meta["target"])
                if instr.dest and instr.op in _DEF_OPS:
                    defs.add(instr.dest)
            if (
                block.terminator.cond is not None
                and block.terminator.cond.kind == "var"
            ):
                uses.add(str(block.terminator.cond.value))
            if (
                block.terminator.value is not None
                and block.terminator.value.kind == "var"
            ):
                uses.add(str(block.terminator.value.value))
            new_in = uses | (new_out - defs)
            if new_in != live_in[label] or new_out != live_out[label]:
                live_in[label] = new_in
                live_out[label] = new_out
                changed = True
    return live_in, live_out


def ir_insert_raii_drop_points(module: MirModule) -> DropResult:
    """Place deterministic cleanup at scope exits and early returns.

    Drop points are derived from live ranges: an owned local is dropped in the
    block where it dies.  A ``return`` terminator has no successors, so every
    owned value live at that point dies there, which covers early returns.
    """
    insertions: list[DropPoint] = []
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        reachable = cfg.reachable()
        live_in, live_out = _liveness(cfg)
        types = _variable_types(cfg)
        blocks: dict[str, BasicBlock] = {}
        for label, block in cfg.blocks.items():
            instructions = list(block.instructions)
            if label in reachable:
                defs = {
                    instr.dest
                    for instr in instructions
                    if instr.dest and instr.op in _DEF_OPS
                }
                dying = (live_in[label] | defs) - live_out[label]
                already_dropped = {
                    str(operand.value)
                    for instr in instructions
                    if instr.op == "drop"
                    for operand in instr.args
                    if operand.kind == "var"
                }
                owned = sorted(
                    variable
                    for variable in dying
                    if variable not in already_dropped
                    and types.get(variable, UNKNOWN_TYPE).owned
                )
                for variable in owned:
                    reason = (
                        "return" if block.terminator.kind == "return" else "scope_exit"
                    )
                    instructions.append(
                        MirInstr(
                            "drop",
                            args=(_var(variable, types[variable]),),
                            meta={"reason": reason},
                        )
                    )
                    insertions.append(DropPoint(name, label, variable, reason))
            blocks[label] = BasicBlock(
                label, instructions, block.terminator, dict(block.meta)
            )
        functions[name] = replace(function, cfg=Cfg(cfg.entry, blocks))
    return DropResult(MirModule(functions, module.source), tuple(insertions))


# ---------------------------------------------------------------------------
# Pure function inlining
# ---------------------------------------------------------------------------


def _hir_cost(expr: HirExpr) -> int:
    """Count HIR nodes as a simple inlining cost model."""
    return len(_walk_hir(expr))


def _substitute_hir(expr: HirExpr, mapping: dict[str, HirExpr], prefix: str) -> HirExpr:
    """Clone an HIR expression with parameters replaced and locals renamed."""
    if expr.kind == "var" and str(expr.value) in mapping:
        return mapping[str(expr.value)]
    if expr.kind == "var" and str(expr.value).startswith("t"):
        expr = replace(expr, value=f"{prefix}{expr.value}")
    args = tuple(_substitute_hir(arg, mapping, prefix) for arg in expr.args)
    branches = tuple(
        HirBranch(
            None
            if branch.condition is None
            else _substitute_hir(branch.condition, mapping, prefix),
            _substitute_hir(branch.body, mapping, prefix),
            dict(branch.meta),
        )
        for branch in expr.branches
    )
    stmts: list[HirStmt] = []
    for stmt in expr.meta.get("stmts", ()):
        if not isinstance(stmt, HirStmt):
            continue
        value = (
            None if stmt.value is None else _substitute_hir(stmt.value, mapping, prefix)
        )
        name = stmt.name
        if name and name not in mapping:
            name = f"{prefix}{name}"
        stmts.append(replace(stmt, name=name, value=value))
    meta = dict(expr.meta)
    if stmts or "stmts" in expr.meta:
        meta["stmts"] = tuple(stmts)
    return replace(expr, args=args, branches=branches, meta=meta)


def _inline_body(function: HirFunction) -> HirExpr | None:
    """Extract a return-safe expression body suitable for inlining."""
    statements = function.body.meta.get("stmts", ())
    if len(statements) == 1:
        statement = statements[0]
        if (
            isinstance(statement, HirStmt)
            and statement.kind == "expr"
            and statement.value is not None
            and statement.value.kind == "return"
        ):
            return statement.value.args[0] if statement.value.args else None
    if not statements and function.body.args:
        return function.body.args[0]
    return None


def _inline_hir(module: HirModule, max_cost: int) -> InlineResult:
    """Inline small pure functions into HIR call sites."""
    candidates: dict[str, tuple[HirFunction, HirExpr]] = {}
    for name, function in module.functions.items():
        body = _inline_body(function)
        if (
            body is not None
            and function.pure
            and not function.recursive
            and _hir_cost(body) <= max_cost
            and "no_inline" not in function.meta.get("attributes", ())
        ):
            candidates[name] = (function, body)
    sites: list[str] = []
    skipped: list[str] = []
    counter = itertools.count(1)

    def visit(expr: HirExpr, caller: str) -> HirExpr:
        args = tuple(visit(arg, caller) for arg in expr.args)
        branches = tuple(
            HirBranch(
                None if branch.condition is None else visit(branch.condition, caller),
                visit(branch.body, caller),
                dict(branch.meta),
            )
            for branch in expr.branches
        )
        stmts: list[HirStmt] = []
        for stmt in expr.meta.get("stmts", ()):
            if isinstance(stmt, HirStmt) and stmt.value is not None:
                stmts.append(replace(stmt, value=visit(stmt.value, caller)))
            elif isinstance(stmt, HirStmt):
                stmts.append(stmt)
        meta = dict(expr.meta)
        if stmts or "stmts" in expr.meta:
            meta["stmts"] = tuple(stmts)
        current = replace(expr, args=args, branches=branches, meta=meta)
        if current.kind != "call":
            return current
        callee_name = str(current.value)
        target = candidates.get(callee_name)
        if target is None:
            if callee_name in module.functions:
                skipped.append(f"{caller}:{callee_name}")
            return current
        callee, body = target
        mapping = {
            param.name: argument
            for param, argument in zip(callee.params, current.args, strict=False)
        }
        prefix = f"{callee_name}${next(counter)}_"
        sites.append(f"{caller}:{callee_name}")
        return _substitute_hir(body, mapping, prefix)

    functions = {
        name: replace(fn, body=visit(fn.body, name))
        for name, fn in module.functions.items()
    }
    return InlineResult(
        HirModule(functions, module.source), tuple(sites), tuple(skipped)
    )


def _rename_mir_operand(
    operand: MirOperand,
    mapping: dict[str, MirOperand],
    prefix: str,
) -> MirOperand:
    """Rename one operand when splicing an inlined MIR callee."""
    if operand.kind == "var":
        base = str(operand.value)
        if base in mapping:
            return mapping[base]
        return _var(f"{prefix}{base}", operand.ty)
    return operand


def _inline_mir(module: MirModule, max_cost: int) -> InlineResult:
    """Inline single-block pure MIR functions at call sites."""
    candidates = {
        name: fn
        for name, fn in module.functions.items()
        if fn.pure
        and not fn.recursive
        and len(fn.cfg.blocks) == 1
        and sum(len(block.instructions) for block in fn.cfg.blocks.values()) <= max_cost
        and "no_inline" not in fn.meta.get("attributes", ())
    }
    sites: list[str] = []
    skipped: list[str] = []
    counter = itertools.count(1)
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        blocks: dict[str, BasicBlock] = {}
        for label, block in function.cfg.blocks.items():
            instructions: list[MirInstr] = []
            for instr in block.instructions:
                callee_name = (
                    str(instr.meta.get("callee", "")) if instr.op == "call" else ""
                )
                target = candidates.get(callee_name)
                if target is None:
                    if instr.op == "call" and callee_name in module.functions:
                        skipped.append(f"{name}:{callee_name}")
                    instructions.append(instr)
                    continue
                callee_block = next(iter(target.cfg.blocks.values()))
                prefix = f"{callee_name}${next(counter)}_"
                param_names = [param.name for param in target.params]
                mapping = dict(zip(param_names, instr.args, strict=False))

                for cloned in callee_block.instructions:
                    instructions.append(
                        MirInstr(
                            cloned.op,
                            dest=f"{prefix}{cloned.dest}" if cloned.dest else None,
                            args=tuple(
                                _rename_mir_operand(arg, mapping, prefix)
                                for arg in cloned.args
                            ),
                            meta=dict(cloned.meta),
                            span=cloned.span,
                        )
                    )
                returned = callee_block.terminator.value
                if instr.dest and returned is not None:
                    instructions.append(
                        MirInstr(
                            "copy",
                            dest=instr.dest,
                            args=(_rename_mir_operand(returned, mapping, prefix),),
                        )
                    )
                sites.append(f"{name}:{callee_name}")
            blocks[label] = BasicBlock(
                label, instructions, block.terminator, dict(block.meta)
            )
        functions[name] = replace(function, cfg=Cfg(function.cfg.entry, blocks))
    return InlineResult(
        MirModule(functions, module.source), tuple(sites), tuple(skipped)
    )


def ir_inline_small_pure_functions(
    module: HirModule | MirModule, *, max_cost: int = 12
) -> InlineResult:
    """Inline small, non-recursive, pure functions under a cost threshold.

    The threshold counts HIR nodes (or MIR instructions for an MIR module).
    Recursive, impure, and explicitly annotated ``no_inline`` functions are
    skipped, and renamed temporaries prevent capture of caller locals.
    """
    if max_cost < 0:
        raise ValueError("max_cost must be non-negative")
    if isinstance(module, HirModule):
        return _inline_hir(module, max_cost)
    if isinstance(module, MirModule):
        return _inline_mir(module, max_cost)
    raise TypeError("expected a HirModule or MirModule")


# ---------------------------------------------------------------------------
# Loop-invariant guard hoisting
# ---------------------------------------------------------------------------


def _loop_headers(cfg: Cfg) -> list[tuple[str, set[str]]]:
    """Find loop headers and their natural loop bodies via dominator back edges."""
    reachable = cfg.reachable()
    dominators = _dominators(cfg)
    loops: list[tuple[str, set[str]]] = []
    for source in sorted(reachable):
        for target in cfg.successors(source):
            if target not in reachable or target not in dominators.get(source, set()):
                continue
            body = {target, source}
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for pred in cfg.predecessors().get(node, ()):
                    if pred in reachable and pred not in body:
                        body.add(pred)
                        queue.append(pred)
            loops.append((target, body))
    unique: dict[str, set[str]] = {}
    for header, body in loops:
        unique[header] = unique.get(header, set()) | body
    return sorted(unique.items())


def ir_hoist_loop_invariant_checks(module: MirModule) -> HoistResult:
    """Move loop-invariant type/bound guards into new preheader blocks.

    A guard is hoisted only when none of its operand variables is defined inside
    the natural loop.  All predecessors of the header are redirected through
    the deterministic ``preheader_<header>`` block, preserving branch order.
    """
    hoisted: list[str] = []
    preheaders: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in module.functions.items():
        original = function.cfg
        copied_blocks = {
            label: BasicBlock(
                label,
                list(block.instructions),
                MirTerminator(
                    block.terminator.kind,
                    block.terminator.targets,
                    block.terminator.cond,
                    block.terminator.value,
                    dict(block.terminator.meta),
                ),
                dict(block.meta),
            )
            for label, block in original.blocks.items()
        }
        cfg = Cfg(original.entry, copied_blocks)
        reachable = cfg.reachable()
        if cfg.entry not in reachable:
            functions[name] = function
            continue
        loops = _loop_headers(cfg)
        for header, body in loops:
            loop_defs: set[str] = set()
            for label in sorted(body):
                for instr in cfg.blocks[label].instructions:
                    if instr.dest:
                        loop_defs.add(instr.dest)
            movable: list[tuple[str, int, MirInstr]] = []
            for label in sorted(body):
                for index, instr in enumerate(cfg.blocks[label].instructions):
                    if (
                        instr.op not in {"guard", "test"}
                        or instr.meta.get("invariant") is False
                    ):
                        continue
                    variables = {
                        str(operand.value)
                        for operand in instr.args
                        if operand.kind == "var"
                    }
                    if variables & loop_defs:
                        continue
                    movable.append((label, index, instr))
            if not movable:
                continue

            preheader_label = f"preheader_{header}"
            if preheader_label in cfg.blocks:
                continue
            preheader = BasicBlock(preheader_label)
            for label in sorted(cfg.blocks):
                if label in body or label == preheader_label:
                    continue
                terminator = cfg.blocks[label].terminator
                cfg.blocks[label].terminator = MirTerminator(
                    terminator.kind,
                    tuple(
                        preheader_label if target == header else target
                        for target in terminator.targets
                    ),
                    terminator.cond,
                    terminator.value,
                    dict(terminator.meta),
                )
            for _, _, instr in movable:
                preheader.instructions.append(instr)
                hoisted.append(f"{name}:{instr.meta.get('name', instr.op)}")
            preheader.terminator = MirTerminator("jump", (header,))
            cfg.blocks[preheader_label] = preheader
            if cfg.entry == header or cfg.entry not in cfg.blocks:
                cfg.entry = preheader_label
            preheaders.append(f"{name}:{preheader_label}")
            for label, index, _ in sorted(
                movable, key=lambda item: (item[0], -item[1])
            ):
                cfg.blocks[label].instructions.pop(index)
        functions[name] = replace(function, cfg=cfg)
    return HoistResult(
        MirModule(functions, module.source), tuple(hoisted), tuple(preheaders)
    )


# ---------------------------------------------------------------------------
# Dual-backend invariant verification
# ---------------------------------------------------------------------------


def _invariant_operand(
    operand: MirOperand, function: str, issues: list[InvariantIssue]
) -> None:
    """Check one operand for identifier and type portability."""
    if not operand.ty.backend_supported:
        issues.append(
            InvariantIssue(
                "UnsupportedType",
                function,
                f"type {operand.ty.text!r} has no shared backend spelling",
            )
        )
    if operand.kind == "var":
        name = str(operand.value)
        if not IDENTIFIER_RE.fullmatch(name) or name in PYTHON_KEYWORDS:
            issues.append(
                InvariantIssue(
                    "UnsafeIdentifier", function, f"identifier {name!r} is not portable"
                )
            )


def _invariant_hir(module: HirModule) -> InvariantReport:
    """Check invariants that can be verified before MIR lowering."""
    issues: list[InvariantIssue] = []
    checked = [
        "unique_function_names",
        "supported_types",
        "portable_identifiers",
        "call_resolution",
    ]
    for name, function in module.functions.items():
        if not IDENTIFIER_RE.fullmatch(name) or name in PYTHON_KEYWORDS:
            issues.append(
                InvariantIssue(
                    "UnsafeIdentifier", name, f"function {name!r} is not portable"
                )
            )
        if not function.ret.backend_supported:
            issues.append(
                InvariantIssue(
                    "UnsupportedType", name, f"return type {function.ret.text!r}"
                )
            )
        for param in function.params:
            if not param.ty.backend_supported:
                issues.append(
                    InvariantIssue("UnsupportedType", name, f"parameter {param.name!r}")
                )
            if not IDENTIFIER_RE.fullmatch(param.name) or param.name in PYTHON_KEYWORDS:
                issues.append(
                    InvariantIssue(
                        "UnsafeIdentifier", name, f"parameter {param.name!r}"
                    )
                )
        for callee in sorted(_called_names(function.body)):
            if callee not in module.functions and callee not in PURE_BUILTINS:
                issues.append(
                    InvariantIssue(
                        "UnresolvedCall", name, f"call to unknown {callee!r}"
                    )
                )
    ok = not any(issue.severity == "error" for issue in issues)
    return InvariantReport(ok, tuple(issues), tuple(checked))


def _invariant_mir(module: MirModule) -> InvariantReport:
    """Check invariants required by both Python and Rust code generators."""
    issues: list[InvariantIssue] = []
    checked = [
        "unique_function_names",
        "supported_types",
        "portable_identifiers",
        "call_resolution",
        "terminated_blocks",
        "lowered_short_circuit",
        "exhaustive_matches",
        "raii_drops",
    ]
    validation = ir_validate_control_flow_graph(module)
    for issue in validation.issues:
        if issue.severity == "error" and issue.code != "UndominatedBlock":
            issues.append(InvariantIssue(issue.code, issue.block, issue.message))

    for name, function in module.functions.items():
        if not IDENTIFIER_RE.fullmatch(name) or name in PYTHON_KEYWORDS:
            issues.append(
                InvariantIssue(
                    "UnsafeIdentifier", name, f"function {name!r} is not portable"
                )
            )
        if not function.ret.backend_supported:
            issues.append(
                InvariantIssue(
                    "UnsupportedType", name, f"return type {function.ret.text!r}"
                )
            )
        for param in function.params:
            if not param.ty.backend_supported:
                issues.append(
                    InvariantIssue("UnsupportedType", name, f"parameter {param.name!r}")
                )
        types = _variable_types(function.cfg)
        drops: set[str] = set()
        arm_patterns: set[str] = set()
        for label, block in function.cfg.blocks.items():
            for instr in block.instructions:
                if instr.op in {"logical_and", "logical_or"}:
                    issues.append(
                        InvariantIssue(
                            "UnloweredShortCircuit",
                            name,
                            f"{label}: {instr.op} remains",
                        )
                    )
                if instr.dest and (
                    not IDENTIFIER_RE.fullmatch(instr.dest)
                    or instr.dest in PYTHON_KEYWORDS
                ):
                    issues.append(
                        InvariantIssue(
                            "UnsafeIdentifier",
                            name,
                            f"{label}: destination {instr.dest!r}",
                        )
                    )
                for operand in instr.args:
                    _invariant_operand(operand, f"{name}:{label}", issues)
                if instr.op == "drop":
                    drops.update(
                        str(operand.value)
                        for operand in instr.args
                        if operand.kind == "var"
                    )
                if instr.op == "call":
                    callee = str(instr.meta.get("callee", ""))
                    if (
                        callee
                        and callee not in module.functions
                        and callee not in PURE_BUILTINS
                    ):
                        issues.append(
                            InvariantIssue(
                                "UnresolvedCall", name, f"call to unknown {callee!r}"
                            )
                        )
            if "pattern" in block.meta:
                arm_patterns.add(str(block.meta["pattern"]))
            terminator = block.terminator
            for operand in (terminator.cond, terminator.value):
                if operand is not None:
                    _invariant_operand(operand, f"{name}:{label}", issues)
        if arm_patterns and "_" not in arm_patterns:
            issues.append(
                InvariantIssue(
                    "NonExhaustiveMatch", name, "match lowering has no wildcard arm"
                )
            )
        owned = {variable for variable, ty in types.items() if ty.owned}
        missing_drops = sorted(owned - drops)
        for variable in missing_drops:
            issues.append(
                InvariantIssue(
                    "MissingDrop",
                    name,
                    f"owned variable {variable!r} has no drop",
                    "warning",
                )
            )
    ok = not any(issue.severity == "error" for issue in issues)
    return InvariantReport(ok, tuple(issues), tuple(checked))


def ir_verify_dual_backend_invariants(
    target: HirModule | MirModule | str | ast.Program,
) -> InvariantReport:
    """Check pre-codegen invariants shared by the Python and Rust backends.

    The check is purely structural: identifiers must be legal in both
    languages, types must have a shared spelling, calls must resolve, MIR must
    be fully lowered, and owned values must have cleanup points.  It never
    invokes either backend or inspects the host platform.
    """
    if isinstance(target, HirModule):
        return _invariant_hir(target)
    if isinstance(target, MirModule):
        return _invariant_mir(target)
    if isinstance(target, str | ast.Program):
        return _invariant_mir(ir_lower_hir_to_mir(ir_lower_ast_to_hir(target)))
    raise TypeError("expected source text, a Program, HirModule, or MirModule")
