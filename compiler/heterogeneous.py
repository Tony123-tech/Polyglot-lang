"""Reference foundations for heterogeneous (NPU/GPU) JIT offloading.

This module is deliberately a **model**, not a device compiler.  It recognizes
affine loop nests in the Polyglot AST, emits textual MLIR-like IR, builds
inspection-friendly descriptor objects, and evaluates small deterministic cost
models.  It never:

* invokes a CUDA, Metal, Vulkan, or ANE compiler or runtime;
* allocates, pins, or maps device/unified memory;
* launches a kernel or creates real streams/events; or
* promises a nanosecond-scale latency.

Every request that would require a real device (dispatching a kernel, mapping
unified memory, or selecting an unavailable target) fails uniformly with the
``E007`` diagnostic from :mod:`runtime.diagnostics`.  The existing source-level
gate that rejects ``#[gpu]`` / ``#[npu]`` is unchanged; this module only adds
offline modelling foundations.
"""

from __future__ import annotations

import itertools
import math
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from compiler.core.ast_nodes import (
    Assign,
    Binary,
    CompoundAssign,
    For,
    Ident,
    Index,
    LitInt,
    Node,
    Paren,
    PatIdent,
    Range,
    Unary,
)
from runtime.diagnostics import PillarError

_META_FIELDS = frozenset({"line", "col", "span", "source_lines"})
_DTYPES = frozenset({"f32", "f16", "bf16", "i8", "i32", "u8"})
_QUANT_DTYPES = frozenset({"fp16", "bf16", "int8"})
_TARGETS = ("metal", "cuda", "spirv")
_MODEL_HEADER = (
    "// MODEL-ONLY source text: no compiler, driver, or device API is invoked.\n"
)
_SPIRV_HEADER = "; MODEL-ONLY SPIR-V textual skeleton: not a valid SPIR-V module.\n"

# Notes are stable strings that tooling can key on without parsing messages.
_NOTE_AST = "NonAffineAst"
_NOTE_TYPE = "HeterogeneousTypeError"
_NOTE_PROOF = "QuantizationBoundViolation"
_NOTE_CYCLE = "FusionDependencyCycle"
_NOTE_DEVICE = "DeviceDispatchUnavailable"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _loc(
    node: object | None,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> tuple[tuple[int, int] | None, int, int]:
    """Resolve a diagnostic location: explicit arguments win over AST fields."""
    node_span = getattr(node, "span", None) if node is not None else None
    node_line = getattr(node, "line", 0) if node is not None else 0
    node_col = getattr(node, "col", 0) if node is not None else 0
    return (
        span if span is not None else node_span,
        line or node_line,
        col or node_col,
    )


def _fail(
    code: str,
    message: str,
    *,
    node: object | None = None,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    reason: str | None = None,
    help_text: str | None = None,
    note: str = _NOTE_AST,
) -> NoReturn:
    """Raise the one public error type used by this module."""
    use_span, use_line, use_col = _loc(node, span, line, col)
    raise PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text,
        note=note,
        line=use_line,
        col=use_col,
        span=use_span,
    )


def _device_failure(
    message: str,
    *,
    node: object | None = None,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    reason: str | None = None,
) -> NoReturn:
    """The single uniform E007 path for real-device requests."""
    _fail(
        "E007",
        message,
        node=node,
        span=span,
        line=line,
        col=col,
        reason=reason or "GPU/NPU kernel offloading is not implemented",
        help_text=(
            "use the emitted textual IR and descriptor models, or the supported "
            "Python/native CPU backend"
        ),
        note=_NOTE_DEVICE,
    )


def _require_descriptor(
    value: object,
    *,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> PolyhedralDescriptor:
    if not isinstance(value, PolyhedralDescriptor):
        _fail(
            "E001",
            "expected a PolyhedralDescriptor from accel_detect_tensor_polyhedral",
            node=value,
            span=span,
            line=line,
            col=col,
            note=_NOTE_TYPE,
        )
    if not value.loops or not value.accesses:
        _fail(
            "E003",
            "the descriptor has no loop nest or no tensor accesses",
            span=span,
            line=line,
            col=col,
        )
    return value


def _walk(value: object) -> Iterator[object]:
    """Yield AST nodes in deterministic field order."""
    if isinstance(value, Node):
        yield value
        for name in value.__dataclass_fields__:
            if name in _META_FIELDS:
                continue
            yield from _walk(getattr(value, name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


# ---------------------------------------------------------------------------
# Descriptors (81)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineExpr:
    """A canonical integer affine form: ``sum(coef * symbol) + constant``."""

    terms: tuple[tuple[str, int], ...] = ()
    constant: int = 0

    @property
    def is_constant(self) -> bool:
        return not self.terms

    @property
    def text(self) -> str:
        chunks: list[str] = []
        for name, coef in self.terms:
            if coef == 1:
                chunks.append(name)
            elif coef == -1:
                chunks.append(f"-{name}")
            else:
                chunks.append(f"{coef}*{name}")
        if self.constant or not chunks:
            chunks.append(str(self.constant))
        result = chunks[0]
        for chunk in chunks[1:]:
            if chunk.startswith("-"):
                result += f" - {chunk[1:]}"
            else:
                result += f" + {chunk}"
        return result

    def __add__(self, other: AffineExpr) -> AffineExpr:
        return AffineExpr._combine(self, other, 1)

    def __sub__(self, other: AffineExpr) -> AffineExpr:
        return AffineExpr._combine(self, other, -1)

    def __mul__(self, scalar: int) -> AffineExpr:
        return AffineExpr(
            tuple((name, coef * scalar) for name, coef in self.terms if coef * scalar),
            self.constant * scalar,
        )

    @staticmethod
    def _combine(left: AffineExpr, right: AffineExpr, sign: int) -> AffineExpr:
        values: dict[str, int] = {}
        for name, coef in left.terms:
            values[name] = values.get(name, 0) + coef
        for name, coef in right.terms:
            values[name] = values.get(name, 0) + sign * coef
        return AffineExpr(
            tuple(sorted((name, coef) for name, coef in values.items() if coef)),
            left.constant + sign * right.constant,
        )


@dataclass(frozen=True)
class LoopBound:
    """One affine loop level with an implicit unit step."""

    name: str
    lower: AffineExpr
    upper: AffineExpr
    inclusive: bool = False
    step: int = 1

    @property
    def trip_count(self) -> AffineExpr:
        count = self.upper - self.lower
        if self.inclusive:
            count = count + AffineExpr((), 1)
        return count


@dataclass(frozen=True)
class TensorAccess:
    """One tensor access inside the analyzed loop nest."""

    tensor: str
    indices: tuple[AffineExpr, ...]
    mode: str  # "read", "write", or "read_write"

    @property
    def rank(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class PolyhedralDescriptor:
    """A pure description of one perfect affine loop nest and its accesses."""

    loops: tuple[LoopBound, ...]
    accesses: tuple[TensorAccess, ...]
    op: str
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    @property
    def depth(self) -> int:
        return len(self.loops)

    @property
    def loop_names(self) -> tuple[str, ...]:
        return tuple(loop.name for loop in self.loops)

    @property
    def tensors(self) -> tuple[str, ...]:
        return tuple(sorted({access.tensor for access in self.accesses}))

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    access.tensor
                    for access in self.accesses
                    if access.mode in {"read", "read_write"}
                }
            )
        )

    @property
    def writes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    access.tensor
                    for access in self.accesses
                    if access.mode in {"write", "read_write"}
                }
            )
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        bound = set(self.loop_names)
        used: set[str] = set()
        for expr in self._all_exprs():
            used.update(name for name, _ in expr.terms if name not in bound)
        return tuple(sorted(used))

    @property
    def rank(self) -> int:
        return max((access.rank for access in self.accesses), default=1)

    def shape(self, default: int = 256) -> tuple[int, ...]:
        """Static trip counts, using ``default`` for symbolic bounds."""
        return tuple(
            loop.trip_count.constant if loop.trip_count.is_constant else default
            for loop in self.loops
        )

    def _all_exprs(self) -> Iterator[AffineExpr]:
        for loop in self.loops:
            yield loop.lower
            yield loop.upper
        for access in self.accesses:
            yield from access.indices

    def to_dict(self) -> dict[str, Any]:
        """A deterministic, JSON-friendly view of the descriptor."""
        return {
            "op": self.op,
            "loops": [
                {
                    "name": loop.name,
                    "lower": loop.lower.text,
                    "upper": loop.upper.text,
                    "step": loop.step,
                    "inclusive": loop.inclusive,
                    "trip_count": loop.trip_count.text,
                }
                for loop in self.loops
            ],
            "accesses": [
                {
                    "tensor": access.tensor,
                    "indices": [expr.text for expr in access.indices],
                    "mode": access.mode,
                }
                for access in self.accesses
            ],
            "tensors": list(self.tensors),
            "symbols": list(self.symbols),
        }


def _affine_from_ast(
    expr: object | None,
    *,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> AffineExpr:
    """Translate a restricted arithmetic expression into an affine form."""
    if expr is None:
        return AffineExpr()
    if isinstance(expr, LitInt):
        return AffineExpr((), expr.value)
    if isinstance(expr, Ident):
        return AffineExpr(((expr.name, 1),), 0)
    if isinstance(expr, Paren):
        return _affine_from_ast(expr.inner, span=span, line=line, col=col)
    if isinstance(expr, Unary):
        if expr.op == "-":
            return _affine_from_ast(expr.operand, span=span, line=line, col=col) * -1
        if expr.op == "+":
            return _affine_from_ast(expr.operand, span=span, line=line, col=col)
    if isinstance(expr, Binary):
        left = _affine_from_ast(expr.left, span=span, line=line, col=col)
        right = _affine_from_ast(expr.right, span=span, line=line, col=col)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            if left.is_constant:
                return right * left.constant
            if right.is_constant:
                return left * right.constant
        _fail(
            "E003",
            f"non-affine operator {expr.op!r} in loop bound or tensor index",
            node=expr,
            span=span,
            line=line,
            col=col,
            help_text="use only +, -, unary -, and multiplication by an integer constant",
        )
    _fail(
        "E003",
        f"non-affine expression {type(expr).__name__} in loop bound or tensor index",
        node=expr,
        span=span,
        line=line,
        col=col,
        help_text="loop bounds and tensor indices must be integer affine expressions",
    )


def _flatten_index(
    node: object,
    *,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> tuple[str, tuple[AffineExpr, ...]] | None:
    """Flatten ``a[i][j]`` into ``("a", (i, j))``."""
    if not isinstance(node, Index):
        return None
    reversed_indices: list[AffineExpr] = []
    current: object = node
    while isinstance(current, Index):
        reversed_indices.append(
            _affine_from_ast(current.index, span=span, line=line, col=col)
        )
        current = current.receiver
    if not isinstance(current, Ident):
        _fail(
            "E003",
            "tensor access base must be a plain identifier (e.g. a[i][j])",
            node=current,
            span=span,
            line=line,
            col=col,
        )
    reversed_indices.reverse()
    return current.name, tuple(reversed_indices)


def _collect_accesses(
    node: object,
    mode: str,
    out: list[TensorAccess],
    *,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> None:
    if isinstance(node, Index):
        flat = _flatten_index(node, span=span, line=line, col=col)
        assert flat is not None
        name, indices = flat
        out.append(TensorAccess(name, indices, mode))
        return
    if isinstance(node, Assign):
        _collect_accesses(node.target, "write", out, span=span, line=line, col=col)
        _collect_accesses(node.value, "read", out, span=span, line=line, col=col)
        return
    if isinstance(node, CompoundAssign):
        _collect_accesses(node.target, "read_write", out, span=span, line=line, col=col)
        _collect_accesses(node.value, "read", out, span=span, line=line, col=col)
        return
    if isinstance(node, Node):
        for name in node.__dataclass_fields__:
            if name in _META_FIELDS:
                continue
            _collect_accesses(
                getattr(node, name), mode, out, span=span, line=line, col=col
            )
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_accesses(item, mode, out, span=span, line=line, col=col)


def _first_loop(value: object) -> For | None:
    if isinstance(value, For):
        return value
    if isinstance(value, Node):
        for name in value.__dataclass_fields__:
            if name in _META_FIELDS:
                continue
            found = _first_loop(getattr(value, name))
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _first_loop(item)
            if found is not None:
                return found
    return None


def _detect_op(
    body: object,
    *,
    span: tuple[int, int] | None,
    line: int,
    col: int,
) -> str:
    """Classify the dominant body operation (a documented heuristic)."""
    for node in _walk(body):
        if not isinstance(node, Assign):
            continue
        written = _flatten_index(node.target, span=span, line=line, col=col)
        read = _flatten_index(node.value, span=span, line=line, col=col)
        if (
            written is not None
            and read is not None
            and written[0] != read[0]
            and len(written[1]) == 2
            and len(read[1]) == 2
            and written[1] == tuple(reversed(read[1]))
        ):
            return "transpose"

    ops = [node.op for node in _walk(body) if isinstance(node, Binary)]
    if any(
        isinstance(node, CompoundAssign) and node.op == "+=" for node in _walk(body)
    ):
        return "matmul" if "*" in ops else "reduce_sum"
    for symbol, name in (("*", "mul"), ("+", "add"), ("-", "sub"), ("/", "div")):
        if symbol in ops:
            return name
    return "copy"


def accel_detect_tensor_polyhedral(
    node: object,
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> PolyhedralDescriptor:
    """81. Detect one perfect affine loop nest and return its descriptor.

    Non-affine bounds or indices, missing loops, and accesses through anything
    other than a plain identifier are rejected with ``E003`` at the exact AST
    location.  Symbolic (non-induction) identifiers are allowed as parameters.
    """
    if not isinstance(node, Node):
        _fail(
            "E003",
            "accel_detect_tensor_polyhedral expects an AST Node",
            span=span,
            line=line,
            col=col,
        )
    outer = node if isinstance(node, For) else _first_loop(node)
    if outer is None:
        _fail(
            "E003",
            "no affine loop nest found in the AST",
            node=node,
            span=span,
            line=line,
            col=col,
        )

    loops: list[LoopBound] = []
    current: For | None = outer
    while current is not None:
        if not isinstance(current.pattern, PatIdent):
            _fail(
                "E003",
                "loop induction variable must be a plain identifier",
                node=current.pattern,
                span=span,
                line=line,
                col=col,
            )
        iterable = current.iter
        if not isinstance(iterable, Range) or iterable.end is None:
            _fail(
                "E003",
                "affine loops must iterate over a bounded Range",
                node=iterable,
                span=span,
                line=line,
                col=col,
            )
        loops.append(
            LoopBound(
                name=current.pattern.name,
                lower=_affine_from_ast(iterable.start, span=span, line=line, col=col),
                upper=_affine_from_ast(iterable.end, span=span, line=line, col=col),
                inclusive=iterable.inclusive,
            )
        )
        current = _first_loop(current.body)

    accesses: list[TensorAccess] = []
    _collect_accesses(outer.body, "read", accesses, span=span, line=line, col=col)
    if not accesses:
        _fail(
            "E003",
            "the affine loop nest contains no tensor accesses",
            node=outer,
            span=span,
            line=line,
            col=col,
        )

    descriptor_span, descriptor_line, descriptor_col = _loc(outer, span, line, col)
    return PolyhedralDescriptor(
        loops=tuple(loops),
        accesses=tuple(accesses),
        op=_detect_op(outer.body, span=span, line=line, col=col),
        span=descriptor_span,
        line=descriptor_line,
        col=descriptor_col,
    )


# ---------------------------------------------------------------------------
# Textual MLIR-like IR (82)
# ---------------------------------------------------------------------------


def _ssa(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return cleaned or "val"


def _tensor_type(rank: int) -> str:
    return "tensor<" + "x".join("?" for _ in range(max(1, rank))) + "xf32>"


def _indexing_map(indices: tuple[AffineExpr, ...], names: tuple[str, ...]) -> str:
    dims = ", ".join(f"d{i}" for i in range(len(names)))
    mapped: list[str] = []
    for expr in indices:
        if expr.is_constant:
            mapped.append(str(expr.constant))
        elif len(expr.terms) == 1 and expr.terms[0][1] == 1:
            symbol = expr.terms[0][0]
            mapped.append(f"d{names.index(symbol)}" if symbol in names else "0")
        else:
            mapped.append("0")  # simplified model map
    return f"affine_map<({dims}) -> ({', '.join(mapped)})>"


def _emit_affine(
    expr: AffineExpr,
    *,
    indent: int,
    env: dict[str, str],
    counter: Iterator[int],
    out: list[str],
) -> str:
    pad = "  " * indent
    current = f"%const{next(counter)}"
    out.append(f"{pad}{current} = arith.constant {expr.constant} : index")
    for name, coef in expr.terms:
        value = env.get(name, f"%sym_{_ssa(name)}")
        if coef != 1:
            const = f"%const{next(counter)}"
            out.append(f"{pad}{const} = arith.constant {coef} : index")
            scaled = f"%val{next(counter)}"
            out.append(f"{pad}{scaled} = arith.muli {value}, {const} : index")
            value = scaled
        total = f"%val{next(counter)}"
        out.append(f"{pad}{total} = arith.addi {current}, {value} : index")
        current = total
    return current


def accel_lower_ast_to_mlir(
    descriptor: PolyhedralDescriptor,
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> str:
    """82. Emit textual ``scf`` / ``linalg`` / ``tensor`` IR from a descriptor.

    The result is an inspectable string.  It is intentionally MLIR-*like*: no
    MLIR parser, dialect registration, or code generation is involved.
    """
    desc = _require_descriptor(descriptor, span=span, line=line, col=col)
    ty = _tensor_type(desc.rank)
    tensors = list(desc.tensors)
    tensor_args = ", ".join(f"%t_{_ssa(name)}: {ty}" for name in tensors)
    symbol_args = ", ".join(f"%sym_{_ssa(name)}: index" for name in desc.symbols)
    signature = ", ".join(part for part in (tensor_args, symbol_args) if part)

    lines = [
        "// Polyglot heterogeneous model IR: textual only, no device code is compiled.",
        f"// op = {desc.op}; depth = {desc.depth}; tensors = {', '.join(tensors)}",
        "module {",
    ]

    outs = desc.writes or (tensors[0],)
    arith_op = "arith.mulf" if desc.op in {"mul", "matmul"} else "arith.addf"
    counter = itertools.count()
    env: dict[str, str] = {name: f"%sym_{_ssa(name)}" for name in desc.symbols}
    body: list[str] = []
    lower_names: dict[str, str] = {}
    upper_names: dict[str, str] = {}
    for loop in desc.loops:
        lower_names[loop.name] = _emit_affine(
            loop.lower, indent=2, env=env, counter=counter, out=body
        )
        upper_names[loop.name] = _emit_affine(
            loop.upper, indent=2, env=env, counter=counter, out=body
        )
        env[loop.name] = f"%{_ssa(loop.name)}"

    indent = 2
    for loop in desc.loops:
        body.append(
            f"{'  ' * indent}scf.for %{_ssa(loop.name)} = "
            f"{lower_names[loop.name]} to {upper_names[loop.name]} "
            f"step {loop.step} {{"
        )
        indent += 1

    pad = "  " * indent
    zero = f"%zero{next(counter)}"
    body.append(f"{pad}{zero} = arith.constant 0.0 : f32")
    read_values: list[str] = []
    write_indices: list[tuple[str, str]] = []
    for access in desc.accesses:
        index_ssa = ", ".join(
            _emit_affine(expr, indent=indent, env=env, counter=counter, out=body)
            for expr in access.indices
        )
        arg = f"%t_{_ssa(access.tensor)}"
        if access.mode in {"read", "read_write"}:
            value = f"%read{next(counter)}"
            body.append(f"{pad}{value} = tensor.extract {arg}[{index_ssa}] : {ty}")
            read_values.append(value)
        if access.mode in {"write", "read_write"}:
            write_indices.append((access.tensor, index_ssa))

    computed = read_values[0] if read_values else zero
    for value in read_values[1:]:
        combined = f"%computed{next(counter)}"
        body.append(f"{pad}{combined} = {arith_op} {computed}, {value} : f32")
        computed = combined
    for name, index_ssa in write_indices:
        result = f"%insert{next(counter)}"
        body.append(
            f"{pad}{result} = tensor.insert {computed} into "
            f"%t_{_ssa(name)}[{index_ssa}] : {ty}"
        )

    for loop in reversed(desc.loops):
        indent -= 1
        body.append(f"{'  ' * indent}}}")

    lines.append(f"  func.func @accel_scf({signature}) -> {ty} {{")
    lines.extend(body)
    lines.append(f"  return %t_{_ssa(outs[0])} : {ty}")
    lines.append("  }")

    maps = ", ".join(
        _indexing_map(access.indices, desc.loop_names) for access in desc.accesses
    )
    iterators = ", ".join('"parallel"' for _ in desc.loops)
    ins = [access.tensor for access in desc.accesses if access.mode == "read"]
    ins_text = ", ".join(f"%t_{_ssa(name)}" for name in ins)
    ins_types = ", ".join(ty for _ in ins)
    out_text = ", ".join(f"%t_{_ssa(name)}" for name in outs)
    out_types = ", ".join(ty for _ in outs)
    lines.extend(
        [
            f"  func.func @accel_linalg({signature}) -> {ty} {{",
            "    %linalg = linalg.generic {",
            f"      indexing_maps = [{maps}],",
            f"      iterator_types = [{iterators}]",
            f"    }} ins({ins_text} : {ins_types}) outs({out_text} : {out_types})",
            "    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):",
            f"      %computed = {arith_op} %lhs, %rhs : f32",
            "      linalg.yield %computed : f32",
            f"    }} -> {ty}",
            f"    return %t_{_ssa(outs[0])} : {ty}",
            "  }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Kernel source text and uniform E007 rejection (83--85)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSource:
    """Textual kernel source for one model target; never compiled or dispatched."""

    target: str
    language: str
    entry_point: str
    source: str
    model_only: bool = True
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


def _kernel_source_text(desc: PolyhedralDescriptor, target: str) -> str:
    names = list(desc.tensors) + ["out", "out", "out"]
    a, b, out = (_ssa(names[index]) for index in range(3))
    detail = (
        f"// descriptor: op={desc.op}, depth={desc.depth}, "
        f"tensors={', '.join(desc.tensors)}"
    )
    if target == "metal":
        return (
            _MODEL_HEADER
            + "#include <metal_stdlib>\nusing namespace metal;\n\n"
            + f"{detail}\n"
            + "kernel void accel_kernel(\n"
            + f"    device const float* {a} [[buffer(0)]],\n"
            + f"    device const float* {b} [[buffer(1)]],\n"
            + f"    device float* {out} [[buffer(2)]],\n"
            + "    uint gid [[thread_position_in_grid]])\n{\n"
            + "    float acc = 0.0f;\n"
            + "    for (uint k = 0u; k < 16u; ++k) {\n"
            + f"        acc += {a}[gid * 16u + k] * {b}[k * 16u + gid];\n"
            + "    }\n"
            + f"    {out}[gid] = acc;\n}}\n"
        )
    if target == "cuda":
        return (
            _MODEL_HEADER
            + f"{detail}\n"
            + 'extern "C" __global__ void accel_kernel(\n'
            + f"    const float* {a}, const float* {b}, float* {out}) {{\n"
            + "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            + "  float acc = 0.0f;\n"
            + "  for (int k = 0; k < 16; ++k) {\n"
            + f"    acc += {a}[idx * 16 + k] * {b}[k * 16 + idx];\n"
            + "  }\n"
            + f"  {out}[idx] = acc;\n}}\n"
        )
    return (
        _SPIRV_HEADER
        + f"; {detail}\n"
        + "OpCapability Shader\n"
        + "OpMemoryModel Logical GLSL450\n"
        + 'OpEntryPoint GLCompute %accel_kernel "accel_kernel"\n'
        + "OpExecutionMode %accel_kernel LocalSize 16 16 1\n"
        + "; body intentionally omitted: this skeleton is not a SPIR-V binary\n"
    )


def accel_emit_metal_cuda_spirv(
    descriptor: PolyhedralDescriptor,
    *,
    target: str = "metal",
    compile_kernel: bool = False,
    dispatch: bool = False,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> KernelSource:
    """83. Emit illustrative Metal/CUDA/SPIR-V *source text*.

    Passing ``compile_kernel=True`` or ``dispatch=True`` is uniformly rejected
    with ``E007``: this module has no compiler, driver, or device runtime.
    """
    desc = _require_descriptor(descriptor, span=span, line=line, col=col)
    if compile_kernel or dispatch:
        _device_failure(
            "real kernel compilation and device dispatch are not implemented",
            node=desc,
            span=span,
            line=line,
            col=col,
        )
    if not isinstance(target, str):
        _fail(
            "E003",
            "target must be a string",
            node=desc,
            span=span,
            line=line,
            col=col,
        )
    normalized = target.strip().lower()
    if normalized not in _TARGETS:
        _device_failure(
            f"accelerator target {target!r} is not available in this model",
            node=desc,
            span=span,
            line=line,
            col=col,
            reason="no CUDA/Metal/Vulkan toolchain is integrated",
        )
    source_span, source_line, source_col = _loc(desc, span, line, col)
    return KernelSource(
        target=normalized,
        language={"metal": "metal", "cuda": "cuda", "spirv": "spirv-text"}[normalized],
        entry_point="accel_kernel",
        source=_kernel_source_text(desc, normalized),
        span=source_span,
        line=source_line,
        col=source_col,
    )


@dataclass(frozen=True)
class BufferRequest:
    """A declarative request for one host-visible buffer view."""

    name: str
    nbytes: int
    dtype: str = "f32"
    alignment: int = 64
    read_only: bool = False
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True)
class BufferView:
    name: str
    offset: int
    length: int
    padded_length: int
    dtype: str
    alignment: int
    read_only: bool

    @property
    def text(self) -> str:
        return (
            f"{self.name} offset={self.offset} length={self.length} "
            f"padded={self.padded_length} dtype={self.dtype} "
            f"alignment={self.alignment} read_only={str(self.read_only).lower()}"
        )


@dataclass(frozen=True)
class BufferViewPlan:
    """Host-side offsets only; no pointer is allocated or device-mapped."""

    views: tuple[BufferView, ...]
    total_nbytes: int
    device_mapped: bool = False
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    @property
    def text(self) -> str:
        body = "\n".join(
            f"  {index}: {view.text}" for index, view in enumerate(self.views)
        )
        return (
            "zero_copy_view @plan {\n"
            + body
            + "\n}\n// host-side offsets only; no device pointer is created\n"
        )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def accel_zero_copy_unified_mem(
    requests: BufferRequest | Sequence[BufferRequest],
    *,
    map_device: bool = False,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> BufferViewPlan:
    """84. Model aligned, zero-copy buffer views as pure offset arithmetic.

    ``map_device=True`` is rejected with ``E007``; this function never creates
    a pointer, pins host memory, or asks a driver for a unified-memory mapping.
    """
    if isinstance(requests, BufferRequest):
        items = (requests,)
    else:
        try:
            items = tuple(requests)
        except TypeError as error:
            _fail(
                "E001",
                "expected a BufferRequest or a sequence of BufferRequests",
                span=span,
                line=line,
                col=col,
                reason=str(error),
                note=_NOTE_TYPE,
            )
    if not items:
        _fail(
            "E003",
            "at least one BufferRequest is required",
            span=span,
            line=line,
            col=col,
        )
    if map_device:
        _device_failure(
            "real unified-memory device mapping is not implemented",
            node=items[0],
            span=span,
            line=line,
            col=col,
        )

    seen: set[str] = set()
    views: list[BufferView] = []
    offset = 0
    for request in items:
        if not isinstance(request, BufferRequest):
            _fail(
                "E001",
                "expected a BufferRequest",
                node=request,
                span=span,
                line=line,
                col=col,
                note=_NOTE_TYPE,
            )
        if not isinstance(request.name, str) or not request.name:
            _fail(
                "E003",
                "buffer name must be a non-empty string",
                node=request,
                span=span,
                line=line,
                col=col,
            )
        if request.name in seen:
            _fail(
                "E003",
                f"duplicate buffer name {request.name!r}",
                node=request,
                span=span,
                line=line,
                col=col,
            )
        seen.add(request.name)
        if not isinstance(request.nbytes, int) or isinstance(request.nbytes, bool):
            _fail(
                "E003",
                "nbytes must be an integer",
                node=request,
                span=span,
                line=line,
                col=col,
            )
        if request.nbytes <= 0:
            _fail(
                "E003",
                "nbytes must be positive",
                node=request,
                span=span,
                line=line,
                col=col,
            )
        if (
            not isinstance(request.alignment, int)
            or request.alignment < 1
            or request.alignment & (request.alignment - 1)
        ):
            _fail(
                "E003",
                "alignment must be a positive power of two",
                node=request,
                span=span,
                line=line,
                col=col,
            )
        if request.dtype not in _DTYPES:
            _fail(
                "E001",
                f"unsupported buffer dtype {request.dtype!r}",
                node=request,
                span=span,
                line=line,
                col=col,
                note=_NOTE_TYPE,
            )
        start = _align_up(offset, request.alignment)
        padded = _align_up(request.nbytes, request.alignment)
        views.append(
            BufferView(
                name=request.name,
                offset=start,
                length=request.nbytes,
                padded_length=padded,
                dtype=request.dtype,
                alignment=request.alignment,
                read_only=bool(request.read_only),
            )
        )
        offset = start + padded

    plan_span, plan_line, plan_col = _loc(items[0], span, line, col)
    return BufferViewPlan(
        views=tuple(views),
        total_nbytes=offset,
        span=plan_span,
        line=plan_line,
        col=plan_col,
    )


@dataclass(frozen=True)
class ANENode:
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def text(self) -> str:
        return f"{self.op}({', '.join(self.inputs)}) -> {', '.join(self.outputs)}"


@dataclass(frozen=True)
class ANEPartition:
    name: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    tile: tuple[int, int]
    nodes: tuple[ANENode, ...]

    @property
    def text(self) -> str:
        body = "\n".join(f"    {node.text}" for node in self.nodes)
        return (
            f"  partition {self.name} ({self.kind}, tile={self.tile[0]}x{self.tile[1]}) {{\n"
            + body
            + "\n  }"
        )


@dataclass(frozen=True)
class ANEGraphPlan:
    kind: str
    partitions: tuple[ANEPartition, ...]
    graph_text: str
    dispatchable: bool = False
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    @property
    def nodes(self) -> tuple[ANENode, ...]:
        return tuple(node for part in self.partitions for node in part.nodes)


def _validate_tile(tile: tuple[int, int], *, span, line, col) -> None:
    if len(tile) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in tile
    ):
        _fail(
            "E003",
            "tile must be a pair of positive integers",
            span=span,
            line=line,
            col=col,
        )


def accel_npu_ane_graph_compile(
    descriptor: PolyhedralDescriptor,
    *,
    tile: tuple[int, int] = (16, 16),
    dispatch: bool = False,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> ANEGraphPlan:
    """85. Partition a matrix-like descriptor into an ANE *graph description*.

    No ANE program is compiled and no network is started.  ``dispatch=True``
    is rejected with ``E007`` at the supplied location.
    """
    desc = _require_descriptor(descriptor, span=span, line=line, col=col)
    _validate_tile(tile, span=span, line=line, col=col)
    if dispatch:
        _device_failure(
            "real ANE graph compilation and dispatch are not implemented",
            node=desc,
            span=span,
            line=line,
            col=col,
        )
    matrix_like = desc.op in {"mul", "matmul"}
    if matrix_like and desc.depth < 2:
        _fail(
            "E003",
            "a matrix subgraph needs at least two affine loop levels",
            node=desc,
            span=span,
            line=line,
            col=col,
        )
    kind = (
        "matmul"
        if matrix_like
        else ("reduction" if desc.op == "reduce_sum" else "elementwise")
    )

    outputs = desc.writes or ("acc",)
    pure_reads = tuple(
        sorted({access.tensor for access in desc.accesses if access.mode == "read"})
    )
    inputs = (
        pure_reads
        or tuple(name for name in desc.reads if name not in outputs)
        or tuple(desc.tensors[:1])
    )
    partitions: list[ANEPartition] = []
    for index, output in enumerate(outputs):
        nodes = [ANENode("load", (name,), (f"v_{_ssa(name)}",)) for name in inputs]
        values = tuple(node.outputs[0] for node in nodes)
        nodes.append(ANENode(kind, values, (f"r_{index}",)))
        nodes.append(ANENode("store", (f"r_{index}",), (output,)))
        partitions.append(
            ANEPartition(
                name=f"part{index}",
                kind=kind,
                inputs=inputs,
                output=output,
                tile=tile,
                nodes=tuple(nodes),
            )
        )

    header = (
        f"// ANE graph model ({kind}): no ANE program is compiled or dispatched.\n"
        + f"// operation={desc.op}, depth={desc.depth}, tile={tile[0]}x{tile[1]}\n"
        + "ane.graph @model {\n"
    )
    if not desc.writes:
        header += "  // output 'acc' models the scalar accumulator in the AST\n"
    graph_text = header + "\n".join(part.text for part in partitions) + "\n}\n"
    graph_span, graph_line, graph_col = _loc(desc, span, line, col)
    return ANEGraphPlan(
        kind=kind,
        partitions=tuple(partitions),
        graph_text=graph_text,
        span=graph_span,
        line=graph_line,
        col=graph_col,
    )


# ---------------------------------------------------------------------------
# Fusion and stream modelling (86--87)
# ---------------------------------------------------------------------------


_FUSION_KINDS = {
    "add": "elementwise",
    "sub": "elementwise",
    "mul": "elementwise",
    "div": "elementwise",
    "matmul": "matmul",
    "reduce_sum": "reduction",
    "transpose": "transpose",
    "copy": "copy",
}


@dataclass(frozen=True)
class FusionStage:
    index: int
    kind: str
    op: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    shape: tuple[int, ...]

    @property
    def text(self) -> str:
        return f"  %{self.index} = {self.kind}.{self.op}({', '.join(self.reads)})"


@dataclass(frozen=True)
class FusionPlan:
    kernel_name: str
    stages: tuple[FusionStage, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    shape: tuple[int, ...]
    text: str
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


def accel_kernel_fusion_epilogue(
    descriptors: Sequence[PolyhedralDescriptor],
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> FusionPlan:
    """86. Fuse compatible descriptors into one ordered kernel *plan*.

    Data dependencies come from tensor names written by one stage and read by
    another.  A dependency cycle is reported with ``E005``; conflicting writes
    and unsupported operations are reported with ``E003``.
    """
    try:
        items = tuple(descriptors)
    except TypeError as error:
        _fail(
            "E001",
            "fusion expects a sequence of PolyhedralDescriptors",
            span=span,
            line=line,
            col=col,
            reason=str(error),
            note=_NOTE_TYPE,
        )
    if not items:
        _fail(
            "E003",
            "at least one descriptor is required for fusion",
            span=span,
            line=line,
            col=col,
        )
    for item in items:
        if not isinstance(item, PolyhedralDescriptor):
            _fail(
                "E001",
                "fusion expects PolyhedralDescriptor instances",
                node=item,
                span=span,
                line=line,
                col=col,
                note=_NOTE_TYPE,
            )

    stages = [
        FusionStage(
            index=index,
            kind=_FUSION_KINDS.get(item.op, ""),
            op=item.op,
            reads=item.reads,
            writes=item.writes,
            shape=item.shape(),
        )
        for index, item in enumerate(items)
    ]
    if len({len(stage.shape) for stage in stages}) != 1:
        _fail(
            "E003",
            "fusion stages must have the same tensor rank",
            span=span,
            line=line,
            col=col,
        )
    for stage in stages:
        if not stage.kind:
            _fail(
                "E003",
                f"operation {stage.op!r} cannot be fused",
                span=span,
                line=line,
                col=col,
            )

    producers: dict[str, int] = {}
    for stage in stages:
        for name in stage.writes:
            if name in producers:
                _fail(
                    "E003",
                    f"two fusion stages write tensor {name!r}",
                    span=span,
                    line=line,
                    col=col,
                )
            producers[name] = stage.index

    edges: dict[int, set[int]] = {stage.index: set() for stage in stages}
    for stage in stages:
        for name in stage.reads:
            producer = producers.get(name)
            if producer is not None and producer != stage.index:
                edges[producer].add(stage.index)

    indegree = {index: 0 for index in edges}
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        index = ready.pop(0)
        order.append(index)
        for target in sorted(edges[index]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(stages):
        _fail(
            "E005",
            "fusion dependency graph contains a cycle",
            span=span,
            line=line,
            col=col,
            reason="the descriptors form a circular tensor producer/consumer graph",
            note=_NOTE_CYCLE,
        )

    ordered = [
        next(stage for stage in stages if stage.index == index) for index in order
    ]
    reads = tuple(sorted({name for stage in ordered for name in stage.reads}))
    writes = tuple(sorted({name for stage in ordered for name in stage.writes}))
    shape = tuple(
        max(stage.shape[axis] for stage in ordered)
        for axis in range(max((len(stage.shape) for stage in ordered), default=0))
    )
    text = (
        "// FUSED EPILOGUE PLAN (model only: no kernel is generated or launched).\n"
        + "fused_kernel @fused_epilogue {\n"
        + "\n".join(stage.text for stage in ordered)
        + "\n  store %"
        + str(ordered[-1].index)
        + f" -> {', '.join(writes) or 'acc'}\n}}\n"
    )
    plan_span, plan_line, plan_col = _loc(items[0], span, line, col)
    return FusionPlan(
        kernel_name="fused_epilogue",
        stages=tuple(ordered),
        reads=reads,
        writes=writes,
        shape=shape,
        text=text,
        span=plan_span,
        line=plan_line,
        col=plan_col,
    )


@dataclass(frozen=True)
class PipelineEvent:
    name: str
    stream: str
    kind: str  # "dma" or "kernel"
    depends_on: tuple[str, ...]
    tick: int


@dataclass(frozen=True)
class PipelineSchedule:
    events: tuple[PipelineEvent, ...]
    depth: int
    streams: int
    serial_ticks: int
    pipelined_ticks: int
    text: str
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    @property
    def overlaps(self) -> bool:
        return self.pipelined_ticks < self.serial_ticks


def accel_async_stream_pipelining(
    source: PolyhedralDescriptor | FusionPlan,
    *,
    depth: int = 2,
    streams: int = 3,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> PipelineSchedule:
    """87. Model multi-stream DMA/kernel events and their dependencies.

    The result is a static event graph.  No CUDA stream, Metal command buffer,
    or event object is created.
    """
    if not isinstance(source, (PolyhedralDescriptor, FusionPlan)):
        _fail(
            "E001",
            "pipeline source must be a PolyhedralDescriptor or FusionPlan",
            node=source,
            span=span,
            line=line,
            col=col,
            note=_NOTE_TYPE,
        )
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        _fail(
            "E003",
            "pipeline depth must be a positive integer",
            span=span,
            line=line,
            col=col,
        )
    if not isinstance(streams, int) or isinstance(streams, bool) or streams < 2:
        _fail(
            "E003",
            "pipeline modelling requires at least two streams",
            span=span,
            line=line,
            col=col,
        )
    work = (
        [stage.kind for stage in source.stages]
        if isinstance(source, FusionPlan)
        else [source.op]
    )

    events: list[PipelineEvent] = []
    ticks: dict[str, int] = {}
    for index in range(depth):
        previous_in = f"dma_in_{index - 1}" if index else None
        reused = f"dma_out_{index - 2}" if index >= 2 else None
        in_deps = tuple(name for name in (previous_in, reused) if name is not None)
        events.append(
            PipelineEvent(
                name=f"dma_in_{index}",
                stream="copy_in",
                kind="dma",
                depends_on=in_deps,
                tick=0,
            )
        )
        ticks[f"dma_in_{index}"] = max((ticks[name] + 1 for name in in_deps), default=0)
        kernel_deps = (f"dma_in_{index}",)
        events.append(
            PipelineEvent(
                name=f"kernel_{index}",
                stream="compute",
                kind="kernel",
                depends_on=kernel_deps,
                tick=0,
            )
        )
        ticks[f"kernel_{index}"] = ticks[f"dma_in_{index}"] + 1
        out_deps = (f"kernel_{index}",)
        events.append(
            PipelineEvent(
                name=f"dma_out_{index}",
                stream="copy_out",
                kind="dma",
                depends_on=out_deps,
                tick=0,
            )
        )
        ticks[f"dma_out_{index}"] = ticks[f"kernel_{index}"] + 1

    resolved = tuple(
        PipelineEvent(
            name=event.name,
            stream=event.stream,
            kind=event.kind,
            depends_on=event.depends_on,
            tick=ticks[event.name],
        )
        for event in events
    )
    rows = "\n".join(
        f"  {event.stream:<9} {event.kind:<6} {event.name} "
        f"tick={event.tick} after=[{', '.join(event.depends_on)}]"
        for event in resolved
    )
    text = (
        f"async.pipeline @model(depth={depth}, streams={streams}, work={work}) {{\n"
        + rows
        + "\n}\n// model only; no device streams or events are created\n"
    )
    schedule_span, schedule_line, schedule_col = _loc(source, span, line, col)
    return PipelineSchedule(
        events=resolved,
        depth=depth,
        streams=streams,
        serial_ticks=3 * depth,
        pipelined_ticks=max(ticks.values(), default=-1) + 1,
        text=text,
        span=schedule_span,
        line=schedule_line,
        col=schedule_col,
    )


# ---------------------------------------------------------------------------
# Deterministic cost models (88--90)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileChoice:
    tile: tuple[int, ...]
    cost_cycles: int
    estimated_us: float
    details: str


@dataclass(frozen=True)
class TileSearchResult:
    best: TileChoice
    trials: tuple[TileChoice, ...]
    shape: tuple[int, ...]
    model: str
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _tile_cost(
    shape: tuple[int, ...], tile: tuple[int, ...], accesses: int, op: str
) -> int:
    """A deterministic integer cycle model; no wall-clock timing is used."""
    tile_count = _product(
        [(dim + size - 1) // size for dim, size in zip(shape, tile, strict=True)]
    )
    elements = _product(tile)
    working_set_bytes = elements * 4 * max(1, accesses)
    cache_penalty = max(0, working_set_bytes - 32768) // 1024
    compute = elements * (2 if op in {"mul", "matmul"} else 1)
    transfer = elements * 2
    return 64 + tile_count * (compute + transfer + cache_penalty)


def _default_tiles(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    base = (8, 16, 32, 64, 128)
    tiles = [
        tuple(min(size, dim) for dim, size in zip(shape, [b] * len(shape), strict=True))
        for b in base
    ]
    if len(shape) >= 2:
        for left, right in ((16, 32), (32, 16), (32, 64), (64, 32)):
            sizes = [left, right] + [16] * (len(shape) - 2)
            tiles.append(
                tuple(min(size, dim) for dim, size in zip(shape, sizes, strict=True))
            )
    return sorted(set(tiles))


def accel_auto_tune_tile_sizes(
    descriptor: PolyhedralDescriptor,
    *,
    candidates: Sequence[tuple[int, ...]] | None = None,
    default_dim: int = 256,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> TileSearchResult:
    """88. Sweep a fixed candidate set through a pure-Python cost model.

    The "micro-benchmark" is a deterministic arithmetic model evaluated for
    each tile; it never measures hardware or calls a timing API.
    """
    desc = _require_descriptor(descriptor, span=span, line=line, col=col)
    if (
        not isinstance(default_dim, int)
        or isinstance(default_dim, bool)
        or default_dim < 1
    ):
        _fail(
            "E003",
            "default_dim must be a positive integer",
            span=span,
            line=line,
            col=col,
        )
    shape = desc.shape(default_dim)
    if candidates is None:
        tiles = _default_tiles(shape)
    else:
        try:
            supplied = tuple(candidates)
        except TypeError as error:
            _fail(
                "E001",
                "candidates must be a sequence of integer tiles",
                span=span,
                line=line,
                col=col,
                reason=str(error),
                note=_NOTE_TYPE,
            )
        tiles = []
        for tile in supplied:
            if not isinstance(tile, (tuple, list)) or len(tile) != len(shape):
                _fail(
                    "E001",
                    "candidate tile rank does not match the loop nest rank",
                    span=span,
                    line=line,
                    col=col,
                    note=_NOTE_TYPE,
                )
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in tile
            ):
                _fail(
                    "E003",
                    "tile sizes must be positive integers",
                    span=span,
                    line=line,
                    col=col,
                )
            tiles.append(tuple(tile))
        tiles = sorted(set(tiles))
    if not tiles:
        _fail("E003", "the tile candidate set is empty", span=span, line=line, col=col)

    trials: list[TileChoice] = []
    for tile in tiles:
        cost = _tile_cost(shape, tile, len(desc.accesses), desc.op)
        trials.append(
            TileChoice(
                tile=tile,
                cost_cycles=cost,
                estimated_us=round(cost / 2400.0, 6),
                details=f"shape={shape} accesses={len(desc.accesses)} op={desc.op}",
            )
        )
    trials.sort(key=lambda choice: (choice.cost_cycles, choice.tile))
    result_span, result_line, result_col = _loc(desc, span, line, col)
    return TileSearchResult(
        best=trials[0],
        trials=tuple(trials),
        shape=shape,
        model="deterministic integer cycle model (2.4 cycles/ns, no measurement)",
        span=result_span,
        line=result_line,
        col=result_col,
    )


@dataclass(frozen=True)
class QuantizationResult:
    dtype: str
    quantized: tuple[float | int, ...]
    dequantized: tuple[float, ...]
    errors: tuple[float, ...]
    error_bounds: tuple[float, ...]
    max_error: float
    error_bound: float
    scale: float
    bits: tuple[str, ...]
    verified: bool
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


def _to_fp16(value: float) -> tuple[float, str]:
    packed = struct.pack(">e", value)
    result = struct.unpack(">e", packed)[0]
    if not math.isfinite(result):
        raise OverflowError("fp16 overflow")
    return result, f"{int.from_bytes(packed, 'big'):04x}"


def _to_bf16(value: float) -> tuple[float, str]:
    bits = int.from_bytes(struct.pack(">f", value), "big")
    low = bits & 0xFFFF
    rounded = (
        bits + 0x10000 if low > 0x8000 or (low == 0x8000 and bits & 0x10000) else bits
    )
    rounded &= 0xFFFF0000
    if rounded & 0x7F800000 == 0x7F800000:
        raise OverflowError("bf16 overflow")
    return struct.unpack(">f", rounded.to_bytes(4, "big"))[0], f"{rounded >> 16:04x}"


def _quantize_scalar(
    value: float, dtype: str, scale: float
) -> tuple[float | int, float, str, float]:
    """Return (quantized, dequantized, bit pattern, error bound)."""
    if dtype == "int8":
        quantized = max(-127, min(127, round(value / scale)))
        return (
            quantized,
            float(quantized * scale),
            f"{quantized & 0xFF:02x}",
            (scale / 2.0 + 1e-9 * max(1.0, abs(value))),
        )
    if dtype == "fp16":
        quantized, bits = _to_fp16(value)
        bound = abs(value) * 2.0**-11 + 2.0**-24
    else:
        quantized, bits = _to_bf16(value)
        bound = abs(value) * 2.0**-8 + 2.0**-133
    return quantized, float(quantized), bits, bound


def accel_mixed_precision_quantize(
    values: Sequence[int | float],
    *,
    dtype: str = "fp16",
    strict: bool = True,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> QuantizationResult:
    """89. Quantize to FP16/BF16/INT8 and verify explicit error bounds.

    ``strict=True`` (the default) raises ``E002`` if any dequantization error
    exceeds its analytic bound.  No NumPy, device kernel, or hardware instruction
    is used: the rounding is expressed with the standard library only.
    """
    if dtype not in _QUANT_DTYPES:
        _fail(
            "E003",
            f"unsupported quantization dtype {dtype!r}",
            span=span,
            line=line,
            col=col,
        )
    try:
        items = tuple(values)
    except TypeError as error:
        _fail(
            "E003",
            "quantization expects a sequence of finite numbers",
            span=span,
            line=line,
            col=col,
            reason=str(error),
        )
    if not items:
        _fail("E003", "at least one value is required", span=span, line=line, col=col)
    finite: list[float] = []
    for value in items:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _fail(
                "E003",
                "quantization inputs must be finite integers or floats",
                span=span,
                line=line,
                col=col,
            )
        cast = float(value)
        if not math.isfinite(cast):
            _fail(
                "E003",
                "quantization inputs must be finite (NaN and infinity are rejected)",
                span=span,
                line=line,
                col=col,
            )
        finite.append(cast)

    amax = max(abs(value) for value in finite)
    scale = amax / 127.0 if dtype == "int8" and amax else 1.0
    quantized: list[float | int] = []
    dequantized: list[float] = []
    errors: list[float] = []
    bounds: list[float] = []
    bits: list[str] = []
    for value in finite:
        try:
            q, deq, pattern, bound = _quantize_scalar(value, dtype, scale)
        except OverflowError as error:
            _fail(
                "E001",
                f"value {value!r} overflows {dtype}",
                span=span,
                line=line,
                col=col,
                reason=str(error),
                note=_NOTE_TYPE,
            )
        quantized.append(q)
        dequantized.append(deq)
        errors.append(abs(value - deq))
        bounds.append(bound)
        bits.append(pattern)

    verified = all(error <= bound for error, bound in zip(errors, bounds, strict=True))
    if strict and not verified:
        _fail(
            "E002",
            "quantization exceeded its declared error bound",
            span=span,
            line=line,
            col=col,
            reason="the dequantized values do not satisfy the analytic bound",
            note=_NOTE_PROOF,
        )
    return QuantizationResult(
        dtype=dtype,
        quantized=tuple(quantized),
        dequantized=tuple(dequantized),
        errors=tuple(errors),
        error_bounds=tuple(bounds),
        max_error=max(errors),
        error_bound=max(bounds),
        scale=scale,
        bits=tuple(bits),
        verified=verified,
        span=span,
        line=line,
        col=col,
    )


@dataclass(frozen=True)
class CPUFallbackPlan:
    iterations: int
    work_per_iteration: int
    lanes: int
    scalar_cycles: int
    estimated_cycles: int
    estimated_ns: float
    speedup_estimate: float
    vectorized: bool
    guaranteed_ns: None
    text: str
    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0


def accel_nanosecond_cpu_fallback(
    source: PolyhedralDescriptor | FusionPlan,
    *,
    lanes: int = 8,
    cycles_per_ns: float = 3.0,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> CPUFallbackPlan:
    """90. Build a SIMD-like CPU fallback plan with an honest timing *estimate*.

    ``guaranteed_ns`` is always ``None``.  The estimate is a deterministic
    cycle count divided by an illustrative clock rate; nothing is compiled or
    executed, and no latency is promised.
    """
    if not isinstance(source, (PolyhedralDescriptor, FusionPlan)):
        _fail(
            "E001",
            "fallback source must be a PolyhedralDescriptor or FusionPlan",
            node=source,
            span=span,
            line=line,
            col=col,
            note=_NOTE_TYPE,
        )
    if not isinstance(lanes, int) or isinstance(lanes, bool) or not 1 <= lanes <= 64:
        _fail(
            "E003",
            "lanes must be an integer between 1 and 64",
            span=span,
            line=line,
            col=col,
        )
    if (
        not isinstance(cycles_per_ns, (int, float))
        or isinstance(cycles_per_ns, bool)
        or not math.isfinite(float(cycles_per_ns))
        or cycles_per_ns <= 0
    ):
        _fail(
            "E003",
            "cycles_per_ns must be a positive finite number",
            span=span,
            line=line,
            col=col,
        )

    if isinstance(source, PolyhedralDescriptor):
        shape = source.shape()
        work = 1 + len(source.accesses)
        label = source.op
    else:
        shape = source.shape
        work = 1 + len(source.stages)
        label = source.kernel_name
    iterations = _product(shape)
    setup = 32 + 4 * lanes
    scalar_cycles = setup + iterations * work
    if lanes == 1:
        estimated_cycles = scalar_cycles
    else:
        vector_iters = iterations // lanes
        tail = iterations % lanes
        estimated_cycles = setup + vector_iters * work * 2 + tail * work
    estimated_ns = round(estimated_cycles / float(cycles_per_ns), 6)
    text = (
        "// CPU FALLBACK PLAN (model only; no code generated, no timing guarantee).\n"
        f"simd_loop @{label} {{\n"
        f"  lanes = {lanes}\n"
        f"  iterations = {iterations} (shape={shape}, work={work})\n"
        f"  vector_body = f32x{lanes} elementwise stages\n"
        "  tail = scalar remainder\n"
        f"  scalar_cycles = {scalar_cycles}; estimated_cycles = {estimated_cycles}\n"
        f"  estimated_ns = {estimated_ns:g} (illustrative, not measured)\n"
        "  guaranteed_ns = none\n"
        "}\n"
    )
    plan_span, plan_line, plan_col = _loc(source, span, line, col)
    return CPUFallbackPlan(
        iterations=iterations,
        work_per_iteration=work,
        lanes=lanes,
        scalar_cycles=scalar_cycles,
        estimated_cycles=estimated_cycles,
        estimated_ns=estimated_ns,
        speedup_estimate=round(scalar_cycles / estimated_cycles, 6),
        vectorized=lanes > 1,
        guaranteed_ns=None,
        text=text,
        span=plan_span,
        line=plan_line,
        col=plan_col,
    )


__all__ = [
    "ANEGraphPlan",
    "ANENode",
    "ANEPartition",
    "AffineExpr",
    "BufferRequest",
    "BufferView",
    "BufferViewPlan",
    "CPUFallbackPlan",
    "FusionPlan",
    "FusionStage",
    "KernelSource",
    "LoopBound",
    "PipelineEvent",
    "PipelineSchedule",
    "PolyhedralDescriptor",
    "QuantizationResult",
    "TensorAccess",
    "TileChoice",
    "TileSearchResult",
    "accel_async_stream_pipelining",
    "accel_auto_tune_tile_sizes",
    "accel_detect_tensor_polyhedral",
    "accel_emit_metal_cuda_spirv",
    "accel_kernel_fusion_epilogue",
    "accel_lower_ast_to_mlir",
    "accel_mixed_precision_quantize",
    "accel_nanosecond_cpu_fallback",
    "accel_npu_ane_graph_compile",
    "accel_zero_copy_unified_mem",
]
