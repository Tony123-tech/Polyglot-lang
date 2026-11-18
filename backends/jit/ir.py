"""Typed SSA intermediate representation and AST lowering for the JIT.

The JIT uses a deliberately small scalar subset of Polyglot.  Source *source
variables* are promoted to typed stack slots (``LocalGet``/``LocalSet``);
expression temporaries are SSA values.  ``Phi`` nodes are used for values that
flow out of control-flow expressions (``if``/``match`` and loop ``break``
values).  This keeps the CFG construction simple while still exercising the
SSA machinery in :mod:`backends.jit.regalloc`.

Only macOS/arm64 is supported by the backend, but this module has no platform
dependency and is importable everywhere for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ast_nodes import (
    ArrayExpr,
    Assign,
    Await,
    Binary,
    Block,
    Break,
    Closure,
    CompoundAssign,
    ConstDecl,
    Continue,
    Expr,
    ExprStmt,
    Field,
    FnDecl,
    For,
    Ident,
    IfExpr,
    IfLet,
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
    Path,
    PatIdent,
    PatLiteral,
    PatOr,
    PatRange,
    PatRef,
    PatWildcard,
    Program,
    RepeatArray,
    ResultErr,
    ResultOk,
    Return,
    StructLiteral,
    Try,
    TupleExpr,
    TypeName,
    TypeRef,
    TypeUnit,
    Unary,
    While,
)
from ast_nodes import (
    Call as AstCall,
)
from ast_nodes import (
    Cast as AstCast,
)
from ast_nodes import (
    Range as RangeExpr,
)
from backends.jit.errors import JITCompileError, JITUnsupportedError

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IRType:
    name: str
    bits: int
    signed: bool
    is_float: bool = False

    @property
    def is_int(self) -> bool:
        return not self.is_float

    @property
    def is_bool(self) -> bool:
        return self is BOOL

    @property
    def is_unit(self) -> bool:
        return self is UNIT

    @property
    def is_unsigned(self) -> bool:
        return self.is_int and not self.signed

    @property
    def size_bytes(self) -> int:
        return 8 if self.is_float or self.bits == 64 else max(1, self.bits // 8)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


I8 = IRType("i8", 8, True)
I16 = IRType("i16", 16, True)
I32 = IRType("i32", 32, True)
I64 = IRType("i64", 64, True)
U8 = IRType("u8", 8, False)
U16 = IRType("u16", 16, False)
U32 = IRType("u32", 32, False)
U64 = IRType("u64", 64, False)
BOOL = IRType("bool", 64, False)
F64 = IRType("f64", 64, True, is_float=True)
UNIT = IRType("()", 0, False)

_INT_TYPES: dict[str, IRType] = {
    "i8": I8,
    "i16": I16,
    "i32": I32,
    "i64": I64,
    "u8": U8,
    "u16": U16,
    "u32": U32,
    "u64": U64,
    "usize": U64,
    "isize": I64,
    "bool": BOOL,
    "f64": F64,
}

INT64_MASK = (1 << 64) - 1
INT64_SIGN = 1 << 63


def normalize_int(value: int, ty: IRType) -> int:
    """Truncate ``value`` to ``ty`` and sign/zero-extend to 64 bits."""
    if ty.is_float or ty.is_unit:
        return value
    bits = ty.bits if ty is not BOOL else 64
    mask = (1 << bits) - 1
    value &= mask
    if ty.signed and ty is not BOOL and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def type_from_ast(ty) -> IRType:
    if ty is None:
        return UNIT
    if isinstance(ty, TypeRef):
        return type_from_ast(ty.inner)
    if isinstance(ty, TypeUnit):
        return UNIT
    if isinstance(ty, TypeName):
        mapped = _INT_TYPES.get(ty.name)
        if mapped is not None:
            return mapped
        raise JITUnsupportedError(f"JIT does not support type `{ty.name}`")
    raise JITUnsupportedError(f"JIT does not support type {type(ty).__name__}")


def type_suffix(raw: str) -> str | None:
    """Return the integer suffix of a literal token, if any."""
    lowered = raw.lower().replace("_", "")
    for suffix in (
        "i128",
        "u128",
        "i64",
        "u64",
        "i32",
        "u32",
        "i16",
        "u16",
        "i8",
        "u8",
        "isize",
        "usize",
    ):
        if lowered.endswith(suffix):
            return suffix
    return None


# ---------------------------------------------------------------------------
# Values and slots
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Value:
    id: int
    ty: IRType
    name: str = ""
    fixed_reg: str | None = None

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"v{self.id}:{self.ty.name}"


@dataclass(eq=False)
class Slot:
    name: str
    ty: IRType
    index: int = -1
    offset: int = 0
    mutable: bool = False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"slot[{self.index}]:{self.name}:{self.ty.name}"


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


class Inst:
    """Base class for IR instructions.

    ``defs`` and ``uses`` are the only interface the allocator needs.
    """

    def defs(self) -> tuple[Value, ...]:
        return ()

    def uses(self) -> tuple[Value, ...]:
        return ()


@dataclass(eq=False)
class Const(Inst):
    value: Value
    const: int | float

    def __post_init__(self) -> None:
        self.value.name = self.value.name or "const"

    def defs(self) -> tuple[Value, ...]:
        return (self.value,)


@dataclass(eq=False)
class BinOp(Inst):
    op: str
    left: Value
    right: Value
    result: Value

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)

    def uses(self) -> tuple[Value, ...]:
        return (self.left, self.right)


@dataclass(eq=False)
class UnOp(Inst):
    op: str
    operand: Value
    result: Value

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)

    def uses(self) -> tuple[Value, ...]:
        return (self.operand,)


@dataclass(eq=False)
class Cmp(Inst):
    op: str
    left: Value
    right: Value
    result: Value
    operand_ty: IRType

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)

    def uses(self) -> tuple[Value, ...]:
        return (self.left, self.right)


@dataclass(eq=False)
class Cast(Inst):
    value: Value
    result: Value
    target: IRType
    source: IRType

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)

    def uses(self) -> tuple[Value, ...]:
        return (self.value,)


@dataclass(eq=False)
class LocalGet(Inst):
    slot: Slot
    result: Value

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)


@dataclass(eq=False)
class LocalSet(Inst):
    slot: Slot
    value: Value

    def uses(self) -> tuple[Value, ...]:
        return (self.value,)


@dataclass(eq=False)
class Phi(Inst):
    result: Value
    incoming: list[tuple[BasicBlock, Value]] = field(default_factory=list)

    def defs(self) -> tuple[Value, ...]:
        return (self.result,)

    def uses(self) -> tuple[Value, ...]:
        return tuple(value for _, value in self.incoming)

    def add_incoming(self, block: BasicBlock, value: Value) -> None:
        for index, (existing, _) in enumerate(self.incoming):
            if existing is block:
                self.incoming[index] = (block, value)
                return
        self.incoming.append((block, value))


@dataclass(eq=False)
class Call(Inst):
    name: str
    args: list[Value]
    result: Value | None
    arg_types: list[IRType] = field(default_factory=list)

    def defs(self) -> tuple[Value, ...]:
        return (self.result,) if self.result is not None else ()

    def uses(self) -> tuple[Value, ...]:
        return tuple(self.args)


@dataclass(eq=False)
class Nop(Inst):
    pass


# Terminators -----------------------------------------------------------------


class Terminator:
    pass


@dataclass(eq=False)
class Jmp(Terminator):
    target: BasicBlock


@dataclass(eq=False)
class BrCond(Terminator):
    cond: Value
    true_block: BasicBlock
    false_block: BasicBlock


@dataclass(eq=False)
class Ret(Terminator):
    value: Value | None = None


@dataclass(eq=False)
class Unreachable(Terminator):
    pass


# ---------------------------------------------------------------------------
# Blocks and functions
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class BasicBlock:
    label: str
    insts: list[Inst] = field(default_factory=list)
    terminator: Terminator | None = None

    def add(self, inst: Inst) -> Inst:
        self.insts.append(inst)
        return inst

    def is_terminated(self) -> bool:
        return self.terminator is not None

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


@dataclass(eq=False)
class IRFunction:
    name: str
    params: list[Value]
    ret_ty: IRType
    entry: BasicBlock
    blocks: list[BasicBlock] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    constants: dict[str, int | float] = field(default_factory=dict)
    _next_value: int = 0
    _next_block: int = 0
    _next_slot: int = 0

    def __post_init__(self) -> None:
        if self.entry not in self.blocks:
            self.blocks.append(self.entry)

    def new_value(self, ty: IRType, name: str = "") -> Value:
        value = Value(self._next_value, ty, name)
        self._next_value += 1
        return value

    def new_slot(self, name: str, ty: IRType, *, mutable: bool = False) -> Slot:
        slot = Slot(name, ty, self._next_slot, mutable=mutable)
        self._next_slot += 1
        self.slots.append(slot)
        return slot

    def new_block(self, label: str | None = None) -> BasicBlock:
        label = label or f"bb{self._next_block}"
        self._next_block += 1
        block = BasicBlock(label)
        self.blocks.append(block)
        return block

    # CFG helpers ---------------------------------------------------------

    def predecessors(self) -> dict[BasicBlock, set[BasicBlock]]:
        preds: dict[BasicBlock, set[BasicBlock]] = {b: set() for b in self.blocks}
        for block in self.blocks:
            for succ in self.successors(block):
                preds[succ].add(block)
        return preds

    @staticmethod
    def successors(block: BasicBlock) -> list[BasicBlock]:
        term = block.terminator
        if isinstance(term, Jmp):
            return [term.target]
        if isinstance(term, BrCond):
            return [term.true_block, term.false_block]
        return []

    def reachable_blocks(self) -> list[BasicBlock]:
        seen: set[BasicBlock] = set()
        order: list[BasicBlock] = []

        def visit(block: BasicBlock) -> None:
            if block in seen:
                return
            seen.add(block)
            order.append(block)
            for succ in self.successors(block):
                visit(succ)

        visit(self.entry)
        return order


# ---------------------------------------------------------------------------
# Dominance frontiers
# ---------------------------------------------------------------------------


def compute_dominators(function: IRFunction) -> dict[BasicBlock, set[BasicBlock]]:
    """Classic iterative dominator computation over the reachable CFG."""
    blocks = function.reachable_blocks()
    block_set = set(blocks)
    preds = function.predecessors()
    all_blocks = set(blocks)
    dom: dict[BasicBlock, set[BasicBlock]] = {
        b: ({b} if b is function.entry else set(all_blocks)) for b in blocks
    }
    changed = True
    while changed:
        changed = False
        for block in blocks:
            if block is function.entry:
                continue
            incoming = [p for p in preds[block] if p in block_set]
            if not incoming:
                new = {block}
            else:
                new = {block} | set.intersection(*(dom[p] for p in incoming))
            if new != dom[block]:
                dom[block] = new
                changed = True
    return dom


def compute_immediate_dominators(
    function: IRFunction,
) -> dict[BasicBlock, BasicBlock | None]:
    dom = compute_dominators(function)
    idom: dict[BasicBlock, BasicBlock | None] = {function.entry: None}
    for block in function.reachable_blocks():
        if block is function.entry:
            continue
        strict = dom[block] - {block}
        candidate: BasicBlock | None = None
        for other in strict:
            if all(other not in dom[c] or other is c for c in strict):
                candidate = other
                break
        idom[block] = candidate
    return idom


def compute_dominance_frontiers(
    function: IRFunction,
) -> dict[BasicBlock, set[BasicBlock]]:
    """Return the dominance frontier for every reachable block."""
    blocks = function.reachable_blocks()
    preds = function.predecessors()
    dom = compute_dominators(function)
    frontiers: dict[BasicBlock, set[BasicBlock]] = {b: set() for b in blocks}
    for block in blocks:
        incoming = [p for p in preds[block] if p in dom]
        if len(incoming) < 2:
            continue
        for pred in incoming:
            runner = pred
            while runner is not None and runner is not dom[block]:
                frontiers[runner].add(block)
                strict = dom[runner] - {runner}
                runner = next(
                    (c for c in strict if all(c in dom[o] or o is c for o in strict)),
                    None,
                )
    return frontiers


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


@dataclass
class FunctionSignature:
    name: str
    param_types: list[IRType]
    ret_ty: IRType


@dataclass
class LoopContext:
    header: BasicBlock
    continue_block: BasicBlock
    exit_block: BasicBlock
    break_values: list[tuple[BasicBlock, Value | None]]
    break_phi: Phi | None = None


class FunctionLowering:
    def __init__(
        self,
        fn: FnDecl,
        signatures: dict[str, FunctionSignature],
        constants: dict[str, int | float],
    ) -> None:
        self.fn = fn
        self.signatures = signatures
        self.constants = constants
        entry = BasicBlock("entry")
        self.func = IRFunction(fn.name, [], type_from_ast(fn.ret), entry)
        self.current = entry
        self.scopes: list[dict[str, Slot]] = [{}]
        self.loops: list[LoopContext] = []
        self._param_values: list[Value] = []
        self._build_parameters()

    # -- setup -------------------------------------------------------------

    def _build_parameters(self) -> None:
        int_index = 0
        float_index = 0
        for param in self.fn.params:
            if not isinstance(param.pattern, PatIdent):
                raise JITUnsupportedError(
                    "JIT only supports identifier parameters",
                    getattr(param, "line", None),
                    getattr(param, "col", None),
                )
            ty = type_from_ast(param.ty)
            if ty.is_unit:
                ty = I64
            if ty.is_float:
                if float_index >= 8:
                    raise JITUnsupportedError("more than 8 floating-point parameters")
                reg = f"D{float_index}"
                float_index += 1
            else:
                if int_index >= 8:
                    raise JITUnsupportedError("more than 8 integer parameters")
                reg = f"X{int_index}"
                int_index += 1
            value = self.func.new_value(ty, param.pattern.name)
            value.fixed_reg = reg
            slot = self.func.new_slot(
                param.pattern.name,
                ty,
                mutable=param.pattern.mutable,
            )
            self.scopes[0][param.pattern.name] = slot
            self.func.params.append(value)
            self._param_values.append(value)
            self.emit(LocalSet(slot, value))

    # -- helpers -----------------------------------------------------------

    def emit(self, inst: Inst) -> Inst:
        if not self.current.is_terminated():
            self.current.add(inst)
        return inst

    def terminate(self, term: Terminator) -> None:
        if not self.current.is_terminated():
            self.current.terminator = term

    def switch(self, block: BasicBlock) -> None:
        self.current = block

    def new_const(self, value: float, ty: IRType, name: str = "const") -> Value:
        result = self.func.new_value(ty, name)
        if ty.is_int and not ty.is_bool:
            value = normalize_int(int(value), ty)
        elif ty.is_bool:
            value = 1 if value else 0
        self.emit(Const(result, value))
        return result

    def new_temp(self, ty: IRType, name: str = "tmp") -> Value:
        return self.func.new_value(ty, name)

    def lookup_slot(self, name: str) -> Slot:
        for scope in reversed(self.scopes):
            slot = scope.get(name)
            if slot is not None:
                return slot
        raise JITUnsupportedError(f"unknown variable `{name}` used in JIT function")

    def find_slot(self, name: str) -> Slot | None:
        for scope in reversed(self.scopes):
            slot = scope.get(name)
            if slot is not None:
                return slot
        return None

    def bind_slot(self, name: str, ty: IRType, *, mutable: bool = False) -> Slot:
        slot = self.func.new_slot(name, ty, mutable=mutable)
        self.scopes[-1][name] = slot
        return slot

    def push_scope(self) -> None:
        self.scopes.append({})

    def pop_scope(self) -> None:
        self.scopes.pop()

    def coerce(self, value: Value, ty: IRType) -> Value:
        if value.ty is ty:
            return value
        result = self.new_temp(ty, "cast")
        self.emit(Cast(value, result, ty, value.ty))
        return result

    # -- function body -----------------------------------------------------

    def lower(self) -> IRFunction:
        tail = self.lower_block(self.fn.body, self.func.ret_ty, create_scope=False)
        if not self.current.is_terminated():
            if self.func.ret_ty.is_unit:
                self.terminate(Ret(None))
            elif tail is not None:
                self.terminate(Ret(self.coerce(tail, self.func.ret_ty)))
            else:
                self.terminate(Ret(self.new_const(0, self.func.ret_ty)))
        return self.func

    def lower_block(
        self,
        block: Block | None,
        expected: IRType | None = None,
        *,
        create_scope: bool = True,
    ) -> Value | None:
        if block is None:
            return None
        if create_scope:
            self.push_scope()
        try:
            for stmt in block.stmts:
                if self.current.is_terminated():
                    break
                self.lower_stmt(stmt)
            if self.current.is_terminated():
                return None
            if block.tail is not None:
                return self.lower_expr(block.tail, expected)
            return None
        finally:
            if create_scope:
                self.pop_scope()

    # -- statements --------------------------------------------------------

    def lower_stmt(self, stmt) -> None:
        if isinstance(stmt, LetStmt):
            self.lower_let(stmt)
            return
        if isinstance(stmt, ExprStmt):
            expr = stmt.expr
            if isinstance(expr, Assign):
                self.lower_assign(expr)
                return
            if isinstance(expr, CompoundAssign):
                self.lower_compound_assign(expr)
                return
            if isinstance(expr, (Return, Break, Continue)):
                self.lower_expr(expr)
                return
            self.lower_expr(expr)
            return
        if isinstance(stmt, ItemStmt):
            # Nested items are not part of the scalar JIT subset.
            item = stmt.item
            if isinstance(item, FnDecl):
                raise JITUnsupportedError(
                    "nested functions are not supported by the JIT"
                )
            return
        raise JITUnsupportedError(
            f"unsupported statement {type(stmt).__name__} in JIT function",
            getattr(stmt, "line", None),
            getattr(stmt, "col", None),
        )

    def lower_let(self, stmt: LetStmt) -> None:
        if not isinstance(stmt.pattern, PatIdent):
            raise JITUnsupportedError(
                "JIT only supports identifier let patterns",
                stmt.line,
                stmt.col,
            )
        declared = type_from_ast(stmt.ty) if stmt.ty is not None else None
        if stmt.value is None:
            value = self.new_const(0, declared or I64)
        else:
            value = self.lower_expr(stmt.value, declared)
        ty = declared or value.ty
        value = self.coerce(value, ty)
        slot = self.bind_slot(
            stmt.pattern.name,
            ty,
            mutable=stmt.mutable or stmt.pattern.mutable,
        )
        self.emit(LocalSet(slot, value))

    def lower_assign(self, stmt: Assign) -> Value:
        if not isinstance(stmt.target, Ident):
            raise JITUnsupportedError(
                "JIT only supports assignments to scalar variables",
                stmt.line,
                stmt.col,
            )
        slot = self.lookup_slot(stmt.target.name)
        value = self.lower_expr(stmt.value, slot.ty)
        value = self.coerce(value, slot.ty)
        self.emit(LocalSet(slot, value))
        return value

    def lower_compound_assign(self, stmt: CompoundAssign) -> Value:
        if not isinstance(stmt.target, Ident):
            raise JITUnsupportedError(
                "JIT only supports compound assignments to scalar variables",
                stmt.line,
                stmt.col,
            )
        slot = self.lookup_slot(stmt.target.name)
        left = self.new_temp(slot.ty, "load")
        self.emit(LocalGet(slot, left))
        right = self.lower_expr(stmt.value, slot.ty)
        op = stmt.op[:-1]
        result = self.lower_binop_values(op, left, right, slot.ty, stmt)
        self.emit(LocalSet(slot, result))
        return result

    # -- expressions -------------------------------------------------------

    def _literal_type(self, expr: LitInt, expected: IRType | None) -> IRType:
        suffix = type_suffix(expr.raw)
        if suffix is not None:
            mapped = _INT_TYPES.get(suffix)
            if mapped is None:
                raise JITUnsupportedError(f"JIT does not support `{suffix}` literals")
            return mapped
        if expected is not None and expected.is_int:
            return expected
        return I64

    def lower_expr(self, expr: Expr, expected: IRType | None = None) -> Value:
        if isinstance(expr, Paren):
            return self.lower_expr(expr.inner, expected)
        if isinstance(expr, LitInt):
            ty = self._literal_type(expr, expected)
            return self.new_const(expr.value, ty)
        if isinstance(expr, LitFloat):
            return self.new_const(float(expr.value), F64)
        if isinstance(expr, LitBool):
            return self.new_const(1 if expr.value else 0, BOOL)
        if isinstance(expr, LitChar):
            return self.new_const(ord(expr.value), U32)
        if isinstance(expr, LitStr):
            raise JITUnsupportedError("string literals are not supported by the JIT")
        if isinstance(expr, Ident):
            slot = self.find_slot(expr.name)
            if slot is not None:
                result = self.new_temp(slot.ty, expr.name)
                self.emit(LocalGet(slot, result))
                return result
            if expr.name in self.constants:
                value = self.constants[expr.name]
                ty = expected or (F64 if isinstance(value, float) else I64)
                return self.new_const(value, ty)
            raise JITUnsupportedError(
                f"unknown identifier `{expr.name}` in JIT function",
                getattr(expr, "line", None),
                getattr(expr, "col", None),
            )
        if isinstance(expr, Path):
            if len(expr.segments) == 1:
                return self.lower_expr(
                    Ident(expr.line, expr.col, expr.segments[0]), expected
                )
            raise JITUnsupportedError(
                f"JIT does not support path expression `{'::'.join(expr.segments)}`",
                expr.line,
                expr.col,
            )
        if isinstance(expr, Unary):
            return self.lower_unary(expr, expected)
        if isinstance(expr, Binary):
            return self.lower_binary(expr, expected)
        if isinstance(expr, AstCast):
            value = self.lower_expr(expr.expr)
            target = type_from_ast(expr.ty)
            result = self.new_temp(target, "cast")
            self.emit(Cast(value, result, target, value.ty))
            return result
        if isinstance(expr, AstCall):
            return self.lower_call(expr, expected)
        if isinstance(expr, IfExpr):
            return self.lower_if(expr, expected)
        if isinstance(expr, MatchExpr):
            return self.lower_match(expr, expected)
        if isinstance(expr, Block):
            return self.lower_block(expr, expected)
        if isinstance(expr, While):
            self.lower_while(expr)
            return self.new_const(0, UNIT)
        if isinstance(expr, Loop):
            return self.lower_loop(expr)
        if isinstance(expr, For):
            self.lower_for(expr)
            return self.new_const(0, UNIT)
        if isinstance(expr, Return):
            value = (
                self.lower_expr(expr.value, self.func.ret_ty)
                if expr.value is not None
                else None
            )
            if value is not None and not self.func.ret_ty.is_unit:
                value = self.coerce(value, self.func.ret_ty)
            self.terminate(Ret(value))
            return self.new_const(0, self.func.ret_ty)
        if isinstance(expr, Break):
            return self.lower_break(expr)
        if isinstance(expr, Continue):
            self.lower_continue(expr)
            return self.new_const(0, UNIT)
        if isinstance(expr, Assign):
            return self.lower_assign(expr)
        if isinstance(expr, CompoundAssign):
            return self.lower_compound_assign(expr)
        if isinstance(expr, OptionSome):
            raise JITUnsupportedError("Option values are not supported by the JIT")
        if isinstance(expr, OptionNone):
            raise JITUnsupportedError("Option values are not supported by the JIT")
        if isinstance(expr, (ResultOk, ResultErr)):
            raise JITUnsupportedError("Result values are not supported by the JIT")
        if isinstance(expr, IfLet):
            raise JITUnsupportedError("if let is not supported by the JIT")
        if isinstance(
            expr,
            (
                MethodCall,
                Field,
                Index,
                StructLiteral,
                ArrayExpr,
                RepeatArray,
                TupleExpr,
                Closure,
                RangeExpr,
                Try,
                Await,
                MacroCall,
            ),
        ):
            raise JITUnsupportedError(
                f"JIT does not support expression {type(expr).__name__}",
                getattr(expr, "line", None),
                getattr(expr, "col", None),
            )
        raise JITUnsupportedError(
            f"JIT does not support expression {type(expr).__name__}",
            getattr(expr, "line", None),
            getattr(expr, "col", None),
        )

    def lower_unary(self, expr: Unary, expected: IRType | None) -> Value:
        if expr.op == "-":
            operand = self.lower_expr(expr.operand, expected)
            if operand.ty.is_float:
                result = self.new_temp(F64, "neg")
                self.emit(UnOp("neg", operand, result))
                return result
            zero = self.new_const(0, operand.ty)
            return self.lower_binop_values("-", zero, operand, operand.ty, expr)
        if expr.op == "!":
            operand = self.lower_expr(expr.operand)
            if operand.ty.is_bool:
                zero = self.new_const(0, BOOL)
                return self.lower_cmp("==", operand, zero, BOOL)
            zero = self.new_const(0, operand.ty)
            return self.lower_cmp("==", operand, zero, operand.ty)
        if expr.op in {"&", "&mut", "*"}:
            return self.lower_expr(expr.operand, expected)
        raise JITUnsupportedError(f"unsupported unary operator `{expr.op}`")

    def _infer(self, expr: Expr) -> IRType | None:
        if isinstance(expr, Paren):
            return self._infer(expr.inner)
        if isinstance(expr, LitInt):
            suffix = type_suffix(expr.raw)
            return _INT_TYPES.get(suffix) if suffix else I64
        if isinstance(expr, LitFloat):
            return F64
        if isinstance(expr, LitBool):
            return BOOL
        if isinstance(expr, LitChar):
            return U32
        if isinstance(expr, Ident):
            slot = self.find_slot(expr.name)
            if slot is not None:
                return slot.ty
            value = self.constants.get(expr.name)
            if value is not None:
                return F64 if isinstance(value, float) else I64
            return None
        if isinstance(expr, AstCast):
            try:
                return type_from_ast(expr.ty)
            except JITUnsupportedError:
                return None
        if isinstance(expr, Unary):
            if expr.op == "!":
                return BOOL
            return self._infer(expr.operand)
        if isinstance(expr, Binary):
            if expr.op in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"}:
                return BOOL
            return self._infer(expr.left) or self._infer(expr.right)
        if isinstance(expr, AstCall):
            if isinstance(expr.func, Ident):
                sig = self.signatures.get(expr.func.name)
                if sig is not None:
                    return sig.ret_ty
            return None
        return None

    def lower_binary(self, expr: Binary, expected: IRType | None) -> Value:
        op = expr.op
        if op == "&&":
            return self.lower_short_circuit(expr, is_and=True)
        if op == "||":
            return self.lower_short_circuit(expr, is_and=False)

        comparisons = {"==", "!=", "<", ">", "<=", ">="}
        if op in comparisons:
            # The result of a comparison is `bool`, but its operands are
            # whatever the two sides evaluate to.
            operand_ty = self._infer(expr.left) or self._infer(expr.right)
        else:
            inferred = self._infer(expr)
            operand_ty = inferred if inferred is not None and inferred.is_int else None
        if operand_ty is None:
            operand_ty = expected if expected is not None and expected.is_int else None
        left = self.lower_expr(expr.left, operand_ty)
        if operand_ty is None:
            operand_ty = left.ty
        right = self.lower_expr(expr.right, operand_ty)
        if operand_ty is None:
            operand_ty = right.ty

        if op in comparisons:
            return self.lower_cmp(op, left, right, operand_ty)

        left = self.coerce(left, operand_ty)
        right = self.coerce(right, operand_ty)
        return self.lower_binop_values(op, left, right, operand_ty, expr)

    def lower_short_circuit(self, expr: Binary, *, is_and: bool) -> Value:
        left = self.lower_expr(expr.left, BOOL)
        left = self.coerce(left, BOOL)
        result_slot = self.func.new_slot("bool_tmp", BOOL)
        rhs_block = self.func.new_block("bool_rhs")
        short_block = self.func.new_block("bool_short")
        merge = self.func.new_block("bool_merge")
        if is_and:
            self.terminate(BrCond(left, rhs_block, short_block))
        else:
            self.terminate(BrCond(left, short_block, rhs_block))
        self.switch(short_block)
        short_value = self.new_const(1 if not is_and else 0, BOOL)
        self.emit(LocalSet(result_slot, short_value))
        self.terminate(Jmp(merge))
        self.switch(rhs_block)
        right = self.coerce(self.lower_expr(expr.right, BOOL), BOOL)
        self.emit(LocalSet(result_slot, right))
        self.terminate(Jmp(merge))
        self.switch(merge)
        result = self.new_temp(BOOL, "bool")
        self.emit(LocalGet(result_slot, result))
        return result

    def lower_cmp(
        self, op: str, left: Value, right: Value, operand_ty: IRType
    ) -> Value:
        left = self.coerce(left, operand_ty)
        right = self.coerce(right, operand_ty)
        result = self.new_temp(BOOL, "cmp")
        self.emit(Cmp(op, left, right, result, operand_ty))
        return result

    def lower_binop_values(
        self,
        op: str,
        left: Value,
        right: Value,
        ty: IRType,
        node,
    ) -> Value:
        if op not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
        }:
            raise JITUnsupportedError(
                f"unsupported binary operator `{op}` in JIT function",
                getattr(node, "line", None),
                getattr(node, "col", None),
            )
        result = self.new_temp(ty, "binop")
        self.emit(BinOp(op, left, right, result))
        if ty.is_int and ty.bits < 64:
            narrowed = self.new_temp(ty, "wrap")
            self.emit(Cast(result, narrowed, ty, ty))
            return narrowed
        return result

    def lower_call(self, expr: AstCall, expected: IRType | None) -> Value:
        if not isinstance(expr.func, Ident):
            raise JITUnsupportedError("JIT only supports direct function calls")
        name = expr.func.name
        sig = self.signatures.get(name)
        if sig is None:
            raise JITUnsupportedError(
                f"call to unknown or unsupported function `{name}` in JIT",
                expr.line,
                expr.col,
            )
        if len(expr.args) != len(sig.param_types):
            raise JITCompileError(
                f"`{name}` expects {len(sig.param_types)} arguments, "
                f"got {len(expr.args)}"
            )
        args = [
            self.coerce(self.lower_expr(arg, param_ty), param_ty)
            for arg, param_ty in zip(expr.args, sig.param_types)
        ]
        result = None
        if not sig.ret_ty.is_unit:
            result = self.new_temp(sig.ret_ty, "call")
        self.emit(Call(name, args, result, list(sig.param_types)))
        return result if result is not None else self.new_const(0, UNIT)

    # -- control flow ------------------------------------------------------

    def lower_if(self, expr: IfExpr, expected: IRType | None) -> Value:
        cond = self.coerce(self.lower_expr(expr.cond, BOOL), BOOL)
        then_block = self.func.new_block("if_then")
        else_block = self.func.new_block("if_else")
        merge = self.func.new_block("if_merge")
        self.terminate(BrCond(cond, then_block, else_block))

        self.switch(then_block)
        then_value = self.lower_block(expr.then_block, expected)
        if then_value is None and expected is not None and not expected.is_unit:
            then_value = self.new_const(0, expected)
        then_end = self.current
        if not self.current.is_terminated():
            self.terminate(Jmp(merge))

        self.switch(else_block)
        if expr.else_branch is None:
            else_value = self.new_const(0, expected or UNIT)
        elif isinstance(expr.else_branch, Block):
            else_value = self.lower_block(expr.else_branch, expected)
        elif isinstance(expr.else_branch, IfExpr):
            else_value = self.lower_if(expr.else_branch, expected)
        else:
            else_value = self.lower_expr(expr.else_branch, expected)
        if else_value is None and expected is not None and not expected.is_unit:
            else_value = self.new_const(0, expected)
        else_end = self.current
        if not self.current.is_terminated():
            self.terminate(Jmp(merge))

        self.switch(merge)
        if then_value is None and else_value is None:
            return self.new_const(0, UNIT)
        result_ty = (
            expected
            or (then_value.ty if then_value is not None else None)
            or (else_value.ty if else_value is not None else None)
            or UNIT
        )
        if result_ty.is_unit:
            return self.new_const(0, UNIT)
        phi = Phi(self.new_temp(result_ty, "if_phi"))
        for end, value in ((then_end, then_value), (else_end, else_value)):
            if end.is_terminated() and not isinstance(
                end.terminator, (Ret, Unreachable)
            ):
                phi.add_incoming(end, self.coerce(value, result_ty))
        if phi.incoming:
            self.emit(phi)
            return phi.result
        return self.new_const(0, result_ty)

    def lower_while(self, expr: While) -> None:
        header = self.func.new_block("while_cond")
        body = self.func.new_block("while_body")
        exit_block = self.func.new_block("while_exit")
        self.terminate(Jmp(header))
        self.switch(header)
        cond = self.coerce(self.lower_expr(expr.cond, BOOL), BOOL)
        self.terminate(BrCond(cond, body, exit_block))
        self.switch(body)
        ctx = LoopContext(header, header, exit_block, [])
        self.loops.append(ctx)
        try:
            self.lower_block(expr.body)
        finally:
            self.loops.pop()
        if not self.current.is_terminated():
            self.terminate(Jmp(header))
        self.switch(exit_block)

    def lower_loop(self, expr: Loop) -> Value:
        body = self.func.new_block("loop_body")
        exit_block = self.func.new_block("loop_exit")
        self.terminate(Jmp(body))
        self.switch(body)
        ctx = LoopContext(body, body, exit_block, [])
        self.loops.append(ctx)
        try:
            self.lower_block(expr.body)
        finally:
            self.loops.pop()
        if not self.current.is_terminated():
            self.terminate(Jmp(body))
        self.switch(exit_block)
        if not ctx.break_values:
            return self.new_const(0, UNIT)
        values = [value for _, value in ctx.break_values if value is not None]
        result_ty = values[0].ty if values else UNIT
        if result_ty.is_unit:
            return self.new_const(0, UNIT)
        phi = Phi(self.new_temp(result_ty, "loop_phi"))
        for block, value in ctx.break_values:
            phi.add_incoming(block, value or self.new_const(0, result_ty))
        self.emit(phi)
        return phi.result

    def lower_break(self, expr: Break) -> Value:
        if not self.loops:
            raise JITCompileError("break outside of a loop")
        ctx = self.loops[-1]
        value = self.lower_expr(expr.value) if expr.value is not None else None
        source = self.current
        ctx.break_values.append((source, value))
        self.terminate(Jmp(ctx.exit_block))
        return self.new_const(0, UNIT)

    def lower_continue(self, expr: Continue) -> None:
        if not self.loops:
            raise JITCompileError("continue outside of a loop")
        ctx = self.loops[-1]
        self.terminate(Jmp(ctx.continue_block))

    def lower_for(self, expr: For) -> None:
        if not isinstance(expr.iter, RangeExpr):
            raise JITUnsupportedError(
                "JIT only supports `for` over integer ranges",
                expr.line,
                expr.col,
            )
        range_expr = expr.iter
        if range_expr.start is None or range_expr.end is None:
            raise JITUnsupportedError("JIT requires explicit range bounds")
        element_ty = self._infer(range_expr.start) or I64
        start = self.coerce(self.lower_expr(range_expr.start, element_ty), element_ty)
        end = self.coerce(self.lower_expr(range_expr.end, element_ty), element_ty)
        var_slot = self.func.new_slot("for_var", element_ty)
        self.emit(LocalSet(var_slot, start))
        header = self.func.new_block("for_cond")
        body = self.func.new_block("for_body")
        step = self.func.new_block("for_step")
        exit_block = self.func.new_block("for_exit")
        self.terminate(Jmp(header))
        self.switch(header)
        current = self.new_temp(element_ty, "for_var")
        self.emit(LocalGet(var_slot, current))
        cmp_op = "<=" if range_expr.inclusive else "<"
        cond = self.lower_cmp(cmp_op, current, end, element_ty)
        self.terminate(BrCond(cond, body, exit_block))
        self.switch(body)
        ctx = LoopContext(header, step, exit_block, [])
        self.loops.append(ctx)
        try:
            self.push_scope()
            if isinstance(expr.pattern, PatIdent):
                self.scopes[-1][expr.pattern.name] = var_slot
            elif not isinstance(expr.pattern, PatWildcard):
                raise JITUnsupportedError(
                    "JIT only supports identifier for-loop patterns",
                    expr.line,
                    expr.col,
                )
            self.lower_block(expr.body, create_scope=False)
            self.pop_scope()
        finally:
            self.loops.pop()
        if not self.current.is_terminated():
            self.terminate(Jmp(step))
        self.switch(step)
        current = self.new_temp(element_ty, "for_var")
        self.emit(LocalGet(var_slot, current))
        one = self.new_const(1, element_ty)
        next_value = self.lower_binop_values("+", current, one, element_ty, expr)
        self.emit(LocalSet(var_slot, next_value))
        self.terminate(Jmp(header))
        self.switch(exit_block)

    # -- match -------------------------------------------------------------

    def lower_match(self, expr: MatchExpr, expected: IRType | None) -> Value:
        scrutinee = self.lower_expr(expr.scrutinee)
        merge = self.func.new_block("match_merge")
        incoming: list[tuple[BasicBlock, Value]] = []
        fail_block = self.func.new_block("match_fail")

        def lower_arm(index: int, entry_block: BasicBlock) -> None:
            if index >= len(expr.arms):
                self.switch(entry_block)
                self.terminate(Jmp(fail_block))
                return
            arm = expr.arms[index]
            body_block = self.func.new_block("match_arm")
            next_block = self.func.new_block("match_next")
            self.switch(entry_block)
            bindings: dict[str, Slot] = {}
            self._lower_pattern(
                arm.pattern,
                scrutinee,
                body_block,
                next_block,
                bindings,
            )
            self.switch(body_block)
            self.push_scope()
            self.scopes[-1].update(bindings)
            if arm.guard is not None:
                guard = self.coerce(self.lower_expr(arm.guard, BOOL), BOOL)
                guard_body = self.func.new_block("match_guard")
                self.terminate(BrCond(guard, guard_body, next_block))
                self.switch(guard_body)
            value = self.lower_expr(arm.body, expected)
            self.pop_scope()
            end = self.current
            if not self.current.is_terminated():
                self.terminate(Jmp(merge))
            incoming.append((end, value))
            lower_arm(index + 1, next_block)

        lower_arm(0, self.current)
        self.switch(merge)
        if not incoming:
            return self.new_const(0, expected or UNIT)
        result_ty = expected or incoming[0][1].ty
        if result_ty.is_unit:
            return self.new_const(0, UNIT)
        phi = Phi(self.new_temp(result_ty, "match_phi"))
        for block, value in incoming:
            phi.add_incoming(block, self.coerce(value, result_ty))
        self.emit(phi)
        # The fail block is unreachable for exhaustive matches; give it a
        # terminator so the code generator can emit it safely.
        self.switch(fail_block)
        self.terminate(Ret(self.new_const(0, self.func.ret_ty)))
        self.switch(merge)
        return phi.result

    def _pattern_literal(self, value, ty: IRType) -> Value:
        """Materialize a pattern literal (the parser stores an Expr)."""
        if isinstance(value, PatLiteral):
            value = value.value
        if isinstance(value, Paren):
            value = value.inner
        if isinstance(value, LitBool):
            return self.new_const(1 if value.value else 0, BOOL)
        if isinstance(value, LitChar):
            return self.new_const(ord(value.value), ty)
        if isinstance(value, LitInt):
            return self.new_const(value.value, ty)
        if isinstance(value, LitFloat):
            return self.new_const(float(value.value), ty)
        if isinstance(value, Unary) and value.op == "-":
            inner = self._pattern_literal(value.operand, ty)
            zero = self.new_const(0, ty)
            return self.lower_binop_values("-", zero, inner, ty, value)
        if isinstance(value, Ident) and value.name in self.constants:
            return self.new_const(self.constants[value.name], ty)
        if isinstance(value, (int, float, bool)):
            if isinstance(value, bool):
                return self.new_const(1 if value else 0, BOOL)
            return self.new_const(value, ty)
        raise JITUnsupportedError("JIT does not support this pattern literal")

    def _lower_pattern(
        self,
        pattern,
        value: Value,
        success: BasicBlock,
        failure: BasicBlock,
        bindings: dict[str, Slot],
    ) -> None:
        """Emit tests that jump to ``success``/``failure``."""
        if isinstance(pattern, PatWildcard):
            self.terminate(Jmp(success))
            return
        if isinstance(pattern, PatIdent):
            slot = self.func.new_slot(
                pattern.name,
                value.ty,
                mutable=pattern.mutable,
            )
            self.emit(LocalSet(slot, value))
            bindings[pattern.name] = slot
            self.terminate(Jmp(success))
            return
        if isinstance(pattern, PatLiteral):
            literal = self._pattern_literal(pattern, value.ty)
            cond = self.lower_cmp("==", value, literal, value.ty)
            self.terminate(BrCond(cond, success, failure))
            return
        if isinstance(pattern, PatRange):
            if pattern.start is None or pattern.end is None:
                raise JITUnsupportedError("JIT requires bounded range patterns")
            low = self._pattern_literal(pattern.start, value.ty)
            high = self._pattern_literal(pattern.end, value.ty)
            low_cond = self.lower_cmp(">=", value, low, value.ty)
            high_cond = self.lower_cmp(
                "<=" if pattern.inclusive else "<",
                value,
                high,
                value.ty,
            )
            and_block = self.func.new_block("pat_range")
            self.terminate(BrCond(low_cond, and_block, failure))
            self.switch(and_block)
            self.terminate(BrCond(high_cond, success, failure))
            return
        if isinstance(pattern, PatOr):
            joint = self.func.new_block("pat_or")
            for alt in pattern.alts:
                if not isinstance(alt, (PatWildcard, PatLiteral, PatRange, PatIdent)):
                    raise JITUnsupportedError(
                        "JIT or-patterns support literals, ranges, and identifiers"
                    )
                alt_block = self.func.new_block("pat_or_alt")
                next_block = self.func.new_block("pat_or_next")
                self.terminate(Jmp(alt_block))
                self.switch(alt_block)
                self._lower_pattern(alt, value, joint, next_block, bindings)
                self.switch(next_block)
            self.terminate(Jmp(failure))
            self.switch(joint)
            self.terminate(Jmp(success))
            return
        if isinstance(pattern, PatRef):
            self._lower_pattern(pattern.inner, value, success, failure, bindings)
            return
        if isinstance(pattern, PatBinding):
            slot = self.func.new_slot(pattern.name, value.ty)
            bindings[pattern.name] = slot
            self.emit(LocalSet(slot, value))
            self._lower_pattern(pattern.inner, value, success, failure, bindings)
            return
        raise JITUnsupportedError(
            f"JIT does not support pattern {type(pattern).__name__}"
        )


# ---------------------------------------------------------------------------
# Program lowering
# ---------------------------------------------------------------------------


def _evaluate_const(expr: Expr, constants: dict[str, int | float]) -> int | float:
    if isinstance(expr, Paren):
        return _evaluate_const(expr.inner, constants)
    if isinstance(expr, LitInt):
        return expr.value
    if isinstance(expr, LitFloat):
        return float(expr.value)
    if isinstance(expr, LitBool):
        return 1 if expr.value else 0
    if isinstance(expr, Unary) and expr.op == "-":
        return -_evaluate_const(expr.operand, constants)
    if isinstance(expr, Unary) and expr.op == "!":
        return 0 if _evaluate_const(expr.operand, constants) else 1
    if isinstance(expr, Ident) and expr.name in constants:
        return constants[expr.name]
    if isinstance(expr, Binary):
        left = _evaluate_const(expr.left, constants)
        right = _evaluate_const(expr.right, constants)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op in {"/", "%"}:
            left_int = int(left)
            right_int = int(right)
            if right_int == 0:
                raise ZeroDivisionError("division by zero")
            quotient = abs(left_int) // abs(right_int)
            if (left_int < 0) != (right_int < 0):
                quotient = -quotient
            if expr.op == "/":
                return quotient
            return left_int - quotient * right_int
        if expr.op == "&":
            return int(left) & int(right)
        if expr.op == "|":
            return int(left) | int(right)
        if expr.op == "^":
            return int(left) ^ int(right)
        if expr.op == "<<":
            return int(left) << int(right)
        if expr.op == ">>":
            return int(left) >> int(right)
    raise JITUnsupportedError("JIT cannot evaluate constant expression")


@dataclass
class LoweredProgram:
    functions: dict[str, IRFunction]
    signatures: dict[str, FunctionSignature]
    constants: dict[str, int | float]
    source_order: list[str]


def lower_program(program: Program) -> LoweredProgram:
    """Lower all scalar functions in ``program`` to SSA IR."""
    constants: dict[str, int | float] = {}
    for item in program.items:
        if isinstance(item, ConstDecl):
            try:
                value = _evaluate_const(item.value, constants)
            except JITUnsupportedError:
                raise JITUnsupportedError(
                    f"JIT cannot lower constant `{item.name}`",
                    item.line,
                    item.col,
                )
            if item.ty is not None:
                declared = type_from_ast(item.ty)
                if declared.is_float:
                    value = float(value)
                elif declared is BOOL:
                    value = 1 if value else 0
                elif declared.is_int:
                    value = normalize_int(int(value), declared)
            constants[item.name] = value
        elif not isinstance(item, FnDecl):
            # Struct/enum/impl/trait/type aliases are outside the scalar JIT
            # subset but do not block lowering of scalar functions.
            continue

    signatures: dict[str, FunctionSignature] = {}
    for item in program.items:
        if not isinstance(item, FnDecl):
            continue
        if item.is_async or item.is_unsafe:
            raise JITUnsupportedError(
                f"JIT does not support async/unsafe function `{item.name}`",
                item.line,
                item.col,
            )
        param_types = []
        for param in item.params:
            if not isinstance(param.pattern, PatIdent):
                raise JITUnsupportedError(
                    "JIT only supports identifier parameters",
                    item.line,
                    item.col,
                )
            param_types.append(type_from_ast(param.ty) if param.ty is not None else I64)
        signatures[item.name] = FunctionSignature(
            item.name,
            param_types,
            type_from_ast(item.ret),
        )

    functions: dict[str, IRFunction] = {}
    order: list[str] = []
    for item in program.items:
        if not isinstance(item, FnDecl) or item.body is None:
            continue
        try:
            functions[item.name] = FunctionLowering(item, signatures, constants).lower()
        except JITUnsupportedError:
            raise
        except Exception as error:  # pragma: no cover - defensive
            raise JITCompileError(
                f"failed to lower `{item.name}` to IR: {error}"
            ) from error
        order.append(item.name)
    return LoweredProgram(functions, signatures, constants, order)


__all__ = [
    "BOOL",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
    "UNIT",
    "BasicBlock",
    "BinOp",
    "BrCond",
    "Call",
    "Cast",
    "Cmp",
    "Const",
    "FunctionLowering",
    "FunctionSignature",
    "IRFunction",
    "IRType",
    "Inst",
    "Jmp",
    "LocalGet",
    "LocalSet",
    "LoopContext",
    "LoweredProgram",
    "Nop",
    "Phi",
    "Ret",
    "Slot",
    "UnOp",
    "Unreachable",
    "Value",
    "compute_dominance_frontiers",
    "compute_dominators",
    "compute_immediate_dominators",
    "lower_program",
    "normalize_int",
    "type_from_ast",
]
