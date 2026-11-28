"""Frame I — ranking variants, loop certificates, and resource fuel contracts.

The frame discharges only *explicit* termination obligations:

* recursive calls must supply a syntactic structural or lexicographic variant
  in a caller-provided :class:`VariantSpec`;
* loop bounds are inferred from counted ranges or monotone integer counters,
  and anything else is reported as ``UnboundedOrUnknown``; and
* resource contracts are charged through the deterministic sandbox fuel meter,
  so exhaustion is a structured diagnostic rather than a stack overflow.

General program termination and worst-case bounds are undecidable.  This frame
never claims them: unsupported forms are either ``UnboundedOrUnknown`` or
produce ``E005``/``E008`` diagnostics with the exact obligation that failed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

from compiler.core.ast_nodes import (
    Assign,
    Binary,
    Call,
    CompoundAssign,
    Continue,
    Expr,
    Field,
    FnDecl,
    For,
    Ident,
    LitInt,
    Loop,
    MethodCall,
    Paren,
    Program,
    Range,
    TupleExpr,
    Unary,
    While,
    WhileLet,
)
from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import PillarError
from security.sandbox import sandbox_deterministic_fuel_meter

Span = tuple[int, int] | None
MeasureKind = Literal["structural", "lexicographic", "numeric"]
LoopBoundKind = Literal["counted", "counter", "UnboundedOrUnknown"]
UNBOUNDED: LoopBoundKind = "UnboundedOrUnknown"

DEFAULT_SUBTERM_FIELDS: tuple[str, ...] = (
    "left",
    "right",
    "head",
    "tail",
    "next",
    "child",
    "children",
    "rest",
)
ALLOCATION_CALLEES: frozenset[str] = frozenset(
    {"alloc", "allocate", "new", "make", "box", "Vec", "String"}
)


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text
        or "supply an explicit variant or bound, or split the proof obligation",
        note="compiler.termination/frame-i",
        span=span,
    )


def _span(node: Any, fallback: Span = None) -> Span:
    value = getattr(node, "span", None)
    return value if value is not None else fallback


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminationDiagnostic:
    code: str
    message: str
    kind: str
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "kind": self.kind,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class VariantSpec:
    """A declared well-founded measure for one function.

    ``parameters`` are ordered from most to least significant for a
    lexicographic measure.  For a structural measure the first parameter is the
    subterm root.  ``callees`` defaults to the function itself but may include
    mutual-recursion targets.
    """

    function: str
    kind: MeasureKind
    parameters: tuple[str, ...]
    callees: tuple[str, ...] = ()
    subterm_fields: tuple[str, ...] = DEFAULT_SUBTERM_FIELDS
    span: Span = None

    def __post_init__(self) -> None:
        if self.kind not in {"structural", "lexicographic", "numeric"}:
            raise _fail("E003", f"unknown variant kind {self.kind!r}", span=self.span)
        if not self.function.strip() or not self.parameters:
            raise _fail(
                "E003",
                "variant specs require a function name and at least one parameter",
                span=self.span,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "kind": self.kind,
            "parameters": list(self.parameters),
            "callees": list(self.callees or (self.function,)),
            "subterm_fields": list(self.subterm_fields),
        }


@dataclass(frozen=True)
class VariantCall:
    function: str
    callee: str
    arguments: tuple[str, ...]
    decreases: tuple[str, ...]
    reason: str
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "callee": self.callee,
            "arguments": list(self.arguments),
            "decreases": list(self.decreases),
            "reason": self.reason,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class VariantReport:
    function: str
    proved: bool
    calls: tuple[VariantCall, ...]
    diagnostics: tuple[TerminationDiagnostic, ...]
    specs: tuple[VariantSpec, ...]

    def digest(self) -> str:
        return self.as_dict()["digest"]

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "function": self.function,
            "proved": self.proved,
            "calls": [item.as_dict() for item in self.calls],
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "specs": [item.as_dict() for item in self.specs],
            "caveats": (
                "structural subterm fields are assumed acyclic by the type checker",
                "lexicographic non-decreasing components must be syntactically unchanged",
                "mutual recursion is proved only for the declared callee set",
            ),
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return payload


@dataclass(frozen=True)
class LoopBoundCertificate:
    loop_id: str
    kind: LoopBoundKind
    iterations: str
    allocations: str
    proof: str
    span: Span = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "kind": self.kind,
            "iterations": self.iterations,
            "allocations": self.allocations,
            "proof": self.proof,
            "span": list(self.span) if self.span else None,
        }


@dataclass(frozen=True)
class LoopBoundReport:
    function: str
    proved: bool
    certificates: tuple[LoopBoundCertificate, ...]
    diagnostics: tuple[TerminationDiagnostic, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "proved": self.proved,
            "certificates": [item.as_dict() for item in self.certificates],
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "caveats": (
                "symbolic bounds assume non-negative counters and no unsigned wrap",
                "allocation bounds count syntactic allocation sites per iteration",
                "UnboundedOrUnknown is an explicit refusal to guess",
            ),
        }


@dataclass(frozen=True)
class FuelContract:
    name: str
    steps: int
    memory_bytes: int
    stack_depth: int
    span: Span = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise _fail("E003", "fuel contract requires a name", span=self.span)
        for field_name in ("steps", "memory_bytes", "stack_depth"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _fail(
                    "E003",
                    f"fuel contract {field_name} must be a positive integer",
                    span=self.span,
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "memory_bytes": self.memory_bytes,
            "stack_depth": self.stack_depth,
        }


@dataclass(frozen=True)
class FuelDecision:
    contract: str
    allowed: bool
    consumed: dict[str, int]
    remaining: dict[str, int]
    exhausted: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "allowed": self.allowed,
            "consumed": dict(self.consumed),
            "remaining": dict(self.remaining),
            "exhausted": list(self.exhausted),
        }


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


@dataclass
class _Functions:
    by_name: dict[str, FnDecl] = field(default_factory=dict)

    def select(self, function: str, span: Span) -> FnDecl:
        try:
            return self.by_name[function]
        except KeyError:
            names = ", ".join(sorted(self.by_name))
            raise _fail(
                "E006",
                f"function {function!r} not found (available: {names})",
                span=span,
            ) from None


def _collect_functions(source: str | Program | Sequence[FnDecl]) -> _Functions:
    if isinstance(source, str):
        program = parse(lex(source))
        items: Sequence[Any] = program.items
    elif isinstance(source, Program):
        items = source.items
    elif isinstance(source, (list, tuple)):
        items = source
    else:
        items = (source,)
    result = _Functions()
    for item in items:
        if isinstance(item, FnDecl):
            result.by_name[item.name] = item
    return result


def _walk(value: Any) -> Iterator[Any]:
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)
        return
    if not is_dataclass(value) or isinstance(value, type):
        return
    yield value
    for item in fields(value):
        if item.name in {"line", "col", "span"}:
            continue
        yield from _walk(getattr(value, item.name, None))


def _walk_local(value: Any) -> Iterator[Any]:
    """Walk a body without descending into nested function activations."""
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_local(child)
        return
    if not is_dataclass(value) or isinstance(value, type):
        return
    yield value
    if isinstance(value, FnDecl):
        return
    for item in fields(value):
        if item.name in {"line", "col", "span"}:
            continue
        yield from _walk_local(getattr(value, item.name, None))


def _unwrap(expr: Expr) -> Expr:
    while isinstance(expr, Paren):
        expr = expr.inner
    return expr


def _render(expr: Any) -> str:
    expr = _unwrap(expr)
    if isinstance(expr, Ident):
        return expr.name
    if isinstance(expr, LitInt):
        return str(expr.value)
    if isinstance(expr, Field):
        return f"{_render(expr.receiver)}.{expr.name}"
    if isinstance(expr, MethodCall):
        args = ", ".join(_render(arg) for arg in expr.args)
        return f"{_render(expr.receiver)}.{expr.method}({args})"
    if isinstance(expr, Call):
        args = ", ".join(_render(arg) for arg in expr.args)
        return f"{_render(expr.func)}({args})"
    if isinstance(expr, Binary):
        return f"({_render(expr.left)} {expr.op} {_render(expr.right)})"
    if isinstance(expr, Unary):
        return f"({expr.op}{_render(expr.operand)})"
    if isinstance(expr, TupleExpr):
        return "(" + ", ".join(_render(item) for item in expr.elems) + ")"
    if isinstance(expr, Range):
        start = _render(expr.start) if expr.start is not None else ""
        end = _render(expr.end) if expr.end is not None else ""
        suffix = "..=" if expr.inclusive else ".."
        return f"{start}{suffix}{end}"
    return type(expr).__name__


def _direct_calls(body: Any, callees: frozenset[str]) -> list[Call]:
    calls: list[Call] = []
    for node in _walk_local(body):
        if (
            isinstance(node, Call)
            and isinstance(node.func, Ident)
            and node.func.name in callees
        ):
            calls.append(node)
    return calls


# ---------------------------------------------------------------------------
# Variant proof
# ---------------------------------------------------------------------------


def _coerce_specs(
    specs: Mapping[str, VariantSpec | Mapping[str, Any]],
) -> tuple[VariantSpec, ...]:
    result: list[VariantSpec] = []
    for name, value in specs.items():
        if isinstance(value, VariantSpec):
            spec = value
        elif isinstance(value, Mapping):
            payload = dict(value)
            payload.setdefault("function", name)
            parameters = payload.get("parameters") or payload.get("variants") or ()
            if isinstance(parameters, str):
                parameters = (parameters,)
            payload["parameters"] = tuple(parameters)
            spec = VariantSpec(**payload)
        else:
            raise _fail(
                "E003",
                f"variant spec for {name!r} must be a VariantSpec or mapping",
            )
        result.append(spec)
    return tuple(result)


def _structural_step(expr: Expr, parameter: str, fields: frozenset[str]) -> str | None:
    expr = _unwrap(expr)
    if isinstance(expr, Field):
        receiver = _unwrap(expr.receiver)
        if isinstance(receiver, Ident) and receiver.name == parameter:
            return expr.name if expr.name in fields else None
    if isinstance(expr, MethodCall) and not expr.args:
        receiver = _unwrap(expr.receiver)
        if isinstance(receiver, Ident) and receiver.name == parameter:
            return expr.method if expr.method in fields else None
    return None


def _numeric_step(expr: Expr, parameter: str) -> int | None:
    expr = _unwrap(expr)
    if isinstance(expr, Binary) and expr.op == "-":
        left, right = _unwrap(expr.left), _unwrap(expr.right)
        if (
            isinstance(left, Ident)
            and left.name == parameter
            and isinstance(right, LitInt)
        ):
            return int(right.value) if int(right.value) > 0 else None
    return None


def _component_step(
    expr: Expr, parameter: str, kind: MeasureKind, fields: frozenset[str]
) -> tuple[bool, str]:
    if kind in {"numeric", "lexicographic"}:
        step = _numeric_step(expr, parameter)
        if step is not None:
            return True, f"{parameter} decreases by {step}"
        return False, f"{_render(expr)} is not a proved numeric decrease of {parameter}"
    structural = _structural_step(expr, parameter, fields)
    if structural is not None:
        return True, f"{parameter}.{structural} is a strict subterm"
    if kind == "structural":
        return False, f"{_render(expr)} is not a declared subterm of {parameter}"
    return False, f"{_render(expr)} is not a proved decrease of {parameter}"


def _check_call(
    *,
    spec: VariantSpec,
    call: Call,
    fallback: Span,
    max_components: int,
) -> VariantCall:
    span = _span(call, fallback)
    arguments = tuple(_render(arg) for arg in call.args)
    if len(arguments) < len(spec.parameters):
        return VariantCall(
            spec.function,
            str(call.func.name) if isinstance(call.func, Ident) else "<dynamic>",
            arguments,
            (),
            "fewer arguments than declared variant components",
            span,
        )
    if len(spec.parameters) > max_components:
        return VariantCall(
            spec.function,
            str(call.func.name) if isinstance(call.func, Ident) else "<dynamic>",
            arguments,
            (),
            f"variant arity {len(spec.parameters)} exceeds max_components {max_components}",
            span,
        )

    fields = frozenset(spec.subterm_fields)
    decreases: list[str] = []
    reasons: list[str] = []
    if spec.kind == "structural":
        parameter = spec.parameters[0]
        ok, reason = _component_step(call.args[0], parameter, spec.kind, fields)
        if ok:
            decreases.append(parameter)
        reasons.append(reason)
    else:
        first_decrease: int | None = None
        for index, parameter in enumerate(spec.parameters):
            ok, reason = _component_step(call.args[index], parameter, spec.kind, fields)
            if ok:
                if first_decrease is None:
                    first_decrease = index
                decreases.append(parameter)
            elif first_decrease is None:
                if arguments[index] != parameter:
                    return VariantCall(
                        spec.function,
                        str(call.func.name)
                        if isinstance(call.func, Ident)
                        else "<dynamic>",
                        arguments,
                        (),
                        f"component {index} ({parameter}) may increase: {reason}",
                        span,
                    )
                reasons.append(f"component {index} unchanged")
            else:
                reasons.append(f"component {index} unconstrained after decrease")
        if first_decrease is None:
            return VariantCall(
                spec.function,
                str(call.func.name) if isinstance(call.func, Ident) else "<dynamic>",
                arguments,
                (),
                "no lexicographic component is proved decreasing",
                span,
            )
    return VariantCall(
        spec.function,
        str(call.func.name) if isinstance(call.func, Ident) else "<dynamic>",
        arguments,
        tuple(decreases),
        "; ".join(reasons),
        span,
    )


def term_check_structural_and_lexicographic_variants(
    source: str | Program | Sequence[FnDecl],
    *,
    specs: Mapping[str, VariantSpec | Mapping[str, Any]],
    max_call_sites: int = 256,
    span: Span = None,
) -> VariantReport:
    """Prove every recursive call in ``specs`` has a decreasing variant.

    A call is accepted only when the supplied spec declares the measure and the
    call-site argument is a syntactic subterm/numeric decrease.  Anything else
    is reported with ``E008``; a missing variant on a recursive cycle is
    reported with ``E005``.
    """
    if max_call_sites <= 0:
        raise _fail("E003", "max_call_sites must be positive", span=span)
    functions = _collect_functions(source)
    resolved = _coerce_specs(specs)
    calls: list[VariantCall] = []
    diagnostics: list[TerminationDiagnostic] = []

    for spec in resolved:
        fn = functions.select(spec.function, span or spec.span)
        declared = set(spec.parameters)
        actual = {
            param.pattern.name for param in fn.params if hasattr(param.pattern, "name")
        }
        for parameter in spec.parameters:
            if parameter not in actual:
                diagnostics.append(
                    TerminationDiagnostic(
                        "E003",
                        f"variant parameter {parameter!r} is not a parameter of {spec.function}",
                        "spec",
                        span or spec.span,
                    )
                )
        del declared  # membership is checked with the rendered argument list

        callees = frozenset(spec.callees or (spec.function,))
        sites = _direct_calls(fn.body, callees)
        if len(sites) > max_call_sites:
            diagnostics.append(
                TerminationDiagnostic(
                    "E008",
                    f"call-site bound {max_call_sites} exhausted for {spec.function}",
                    "bound",
                    span or _span(fn, spec.span),
                )
            )
            sites = sites[:max_call_sites]
        for call in sites:
            checked = _check_call(
                spec=spec,
                call=call,
                fallback=span or _span(fn, spec.span),
                max_components=32,
            )
            calls.append(checked)
            if not checked.decreases:
                code = (
                    "E008"
                    if "variant" in checked.reason or "arity" in checked.reason
                    else "E005"
                )
                diagnostics.append(
                    TerminationDiagnostic(
                        code,
                        f"recursive call {checked.callee}({', '.join(checked.arguments)}) "
                        f"has no proved variant: {checked.reason}",
                        "proof",
                        checked.span,
                    )
                )

    proved = (
        bool(resolved) and all(call.decreases for call in calls) and not diagnostics
    )
    return VariantReport(
        function=", ".join(spec.function for spec in resolved),
        proved=proved,
        calls=tuple(calls),
        diagnostics=tuple(diagnostics),
        specs=resolved,
    )


# ---------------------------------------------------------------------------
# Loop bound certificates
# ---------------------------------------------------------------------------


def _allocation_multiplier(body: Any) -> int:
    sites = 0
    for node in _walk_local(body):
        if (
            isinstance(node, Call)
            and isinstance(node.func, Ident)
            and node.func.name in ALLOCATION_CALLEES
        ):
            sites += 1
    return sites


def _allocation_expression(iterations: str, body: Any) -> str:
    sites = _allocation_multiplier(body)
    if sites == 0:
        return "0"
    if iterations == "0":
        return "0"
    if sites == 1:
        return iterations
    return f"{sites} * {iterations}"


def _const_int(expr: Expr | None) -> int | None:
    if isinstance(expr, LitInt):
        return int(expr.value)
    return None


def _counter_update(body: Any, name: str) -> tuple[int | None, bool]:
    """Return (step, safe) where step is positive for a terminating direction."""
    step: int | None = None
    unsafe = False
    for node in _walk_local(body):
        if isinstance(node, Continue):
            unsafe = True
        if isinstance(node, CompoundAssign):
            target = _unwrap(node.target)
            if not isinstance(target, Ident) or target.name != name:
                continue
            value = _const_int(node.value)
            if node.op == "-=" and value is not None and value > 0:
                step = value if step is None else min(step, value)
            elif node.op == "+=" and value is not None and value > 0:
                step = -value if step is None else min(step, -value)
            else:
                unsafe = True
        if isinstance(node, Assign):
            target = _unwrap(node.target)
            if not isinstance(target, Ident) or target.name != name:
                continue
            value = _unwrap(node.value)
            if isinstance(value, Binary) and isinstance(value.left, Ident):
                if value.left.name != name:
                    unsafe = True
                    continue
                amount = _const_int(value.right)
                if amount is None or amount <= 0:
                    unsafe = True
                elif value.op == "-":
                    step = amount if step is None else min(step, amount)
                elif value.op == "+":
                    step = -amount if step is None else min(step, -amount)
                else:
                    unsafe = True
            else:
                unsafe = True
    return step, unsafe


def _while_bound(loop: While) -> tuple[LoopBoundKind, str, str]:
    cond = _unwrap(loop.cond)
    if not isinstance(cond, Binary) or not isinstance(cond.left, Ident):
        return UNBOUNDED, "UnboundedOrUnknown", "guard is not a variable comparison"
    name = cond.left.name
    bound = _const_int(cond.right)
    if bound is None:
        return (
            UNBOUNDED,
            "UnboundedOrUnknown",
            f"guard bound for {name} is input-dependent",
        )
    step, unsafe = _counter_update(loop.body, name)
    if step is None or unsafe:
        return (
            UNBOUNDED,
            "UnboundedOrUnknown",
            f"no safe monotone update proved for {name}",
        )
    if cond.op in {">", ">="} and step > 0:
        base = bound if cond.op == ">" else bound + 1
        return (
            "counter",
            f"max(0, {name} - {base}) / {step}",
            f"{name} decreases by {step}",
        )
    if cond.op in {"<", "<="} and step < 0:
        base = bound if cond.op == "<" else bound - 1
        return (
            "counter",
            f"max(0, {base} - {name}) / {-step}",
            f"{name} increases by {-step}",
        )
    return (
        UNBOUNDED,
        "UnboundedOrUnknown",
        f"guard {cond.op} disagrees with the update direction for {name}",
    )


def term_infer_loop_bound_certificates(
    source: str | Program | Sequence[FnDecl],
    *,
    function: str = "main",
    span: Span = None,
) -> LoopBoundReport:
    """Derive conservative iteration/allocation bounds for loops in one function.

    Only counted ``for`` ranges and monotone integer ``while`` counters produce
    certificates.  Every other loop is explicitly ``UnboundedOrUnknown``.
    """
    functions = _collect_functions(source)
    fn = functions.select(function, span)
    certificates: list[LoopBoundCertificate] = []
    diagnostics: list[TerminationDiagnostic] = []

    loops = [
        node
        for node in _walk_local(fn.body)
        if isinstance(node, (For, While, WhileLet, Loop))
    ]
    for index, loop in enumerate(loops):
        loop_span = _span(loop, span)
        loop_id = f"{type(loop).__name__}@{loop_span or index}"
        if isinstance(loop, For):
            iterator = _unwrap(loop.iter)
            if (
                not isinstance(iterator, Range)
                or iterator.start is None
                or iterator.end is None
            ):
                kind, iterations, proof = (
                    UNBOUNDED,
                    "UnboundedOrUnknown",
                    "iterator is not a statically bounded range",
                )
            else:
                start, end = _const_int(iterator.start), _const_int(iterator.end)
                if start is not None and end is not None:
                    count = max(0, end - start + (1 if iterator.inclusive else 0))
                    iterations = str(count)
                else:
                    extension = " + 1" if iterator.inclusive else ""
                    iterations = f"max(0, {_render(iterator.end)} - {_render(iterator.start)}{extension})"
                kind, proof = "counted", "range endpoints bound the iteration count"
        elif isinstance(loop, While):
            kind, iterations, proof = _while_bound(loop)
        else:
            kind, iterations, proof = (
                UNBOUNDED,
                "UnboundedOrUnknown",
                "unconditional loop has no finite bound",
            )
        if kind == UNBOUNDED:
            diagnostics.append(
                TerminationDiagnostic(
                    "E008",
                    f"{loop_id}: {proof}",
                    "bound",
                    loop_span,
                )
            )
        certificates.append(
            LoopBoundCertificate(
                loop_id=loop_id,
                kind=kind,
                iterations=iterations,
                allocations=_allocation_expression(iterations, loop.body),
                proof=proof,
                span=loop_span,
            )
        )

    proved = bool(certificates) and all(cert.kind != UNBOUNDED for cert in certificates)
    return LoopBoundReport(
        function=fn.name,
        proved=proved,
        certificates=tuple(certificates),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Resource fuel contracts
# ---------------------------------------------------------------------------


def _coerce_contract(value: FuelContract | Mapping[str, Any]) -> FuelContract:
    if isinstance(value, FuelContract):
        return value
    if isinstance(value, Mapping):
        return FuelContract(**dict(value))
    raise _fail("E003", "fuel contract must be a FuelContract or mapping")


def term_enforce_resource_fuel_contracts(
    contract: FuelContract | Mapping[str, Any],
    *,
    steps: int = 0,
    memory_bytes: int = 0,
    stack_depth: int = 0,
    strict: bool = True,
    span: Span = None,
) -> FuelDecision:
    """Charge a deterministic fuel contract and report exhaustion.

    ``strict=True`` raises ``E008`` with the exhausted resources; otherwise a
    ``FuelDecision`` with ``allowed=False`` is returned.  The step meter is the
    same replayable meter used by the security sandbox, so a failed charge
    never mutates the meter.
    """
    resolved = _coerce_contract(contract)
    usage = {
        "steps": steps,
        "memory_bytes": memory_bytes,
        "stack_depth": stack_depth,
    }
    for name, value in usage.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _fail(
                "E003", f"{name} usage must be a non-negative integer", span=span
            )
    limits = {
        "steps": resolved.steps,
        "memory_bytes": resolved.memory_bytes,
        "stack_depth": resolved.stack_depth,
    }
    remaining = {name: limits[name] - usage[name] for name in limits}
    exhausted = tuple(sorted(name for name, left in remaining.items() if left < 0))

    if usage["steps"]:
        meter = sandbox_deterministic_fuel_meter(
            limits["steps"], span=span or resolved.span
        )
        try:
            meter.consume(usage["steps"], span=span or resolved.span)
        except PillarError as exc:
            if "steps" not in exhausted:
                exhausted = tuple(sorted((*exhausted, "steps")))
            del exc  # the structured decision replaces the sandbox diagnostic

    decision = FuelDecision(
        contract=resolved.name,
        allowed=not exhausted,
        consumed=usage,
        remaining={name: max(0, value) for name, value in remaining.items()},
        exhausted=exhausted,
    )
    if strict and exhausted:
        raise _fail(
            "E008",
            f"resource fuel exhausted for {resolved.name}: {', '.join(exhausted)}",
            span=span or resolved.span,
            reason=f"usage={usage}, limits={limits}",
            help_text="increase the declared budget or reduce the input/workload",
        )
    return decision


__all__ = [
    "FuelContract",
    "FuelDecision",
    "LoopBoundCertificate",
    "LoopBoundReport",
    "TerminationDiagnostic",
    "VariantCall",
    "VariantReport",
    "VariantSpec",
    "term_check_structural_and_lexicographic_variants",
    "term_enforce_resource_fuel_contracts",
    "term_infer_loop_bound_certificates",
]
