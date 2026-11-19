"""Honest, dependency-free reference foundations for sandbox pillars 71-80.

This module deliberately separates two kinds of behaviour:

* Pure-Python semantics that really work here: policy validation, deterministic
  fuel accounting, bit-level taint propagation, fat-pointer checks,
  best-effort byte zeroization, and a violation journal.
* Hardware/OS primitives that pure Python cannot provide.  Those APIs emit
  declarative plans and reject every execution boundary with ``E007``.  A plan
  is documentation for a future native implementation, never enforcement.

No function in this module installs a signal handler, writes a protection key
register, generates eBPF bytecode, mprotects memory, or injects a CPU barrier.
The shared ``PillarError`` codes (E001-E008) are the only failure surface.
"""

from __future__ import annotations

import array
import builtins
import hashlib
import operator
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, Self

from runtime.diagnostics import PillarError

REFERENCE_CAVEAT: Final[str] = (
    "Reference model only: this Python module provides no hardware-enforced "
    "isolation and no zero-cost guarantee."
)
ZEROIZATION_CAVEAT: Final[str] = (
    "Best-effort Python mutation only: CPython copies, interpreter buffering, "
    "and dead-storage elimination are not controlled."
)
DYNAMIC_CODE_SINKS: Final[frozenset[str]] = frozenset({"exec", "eval", "compile"})

_VALID_PERMISSIONS: Final[frozenset[str]] = frozenset({"rw", "ro", "none"})
_RECOVERY_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"fallback", "skip", "retry", "abort"}
)
_SPECTRE_PLACEMENTS: Final[dict[str, str]] = {
    "bounds-check": "after-bounds-check",
    "indirect-branch": "before-indirect-call",
    "store-to-load": "between-store-and-load",
    "load-value": "after-load",
}
_CONSTRAINT_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=|<=|>=|<|>)\s*(?:-?\d+|[A-Za-z_][A-Za-z0-9_]*)$"
)


# ---------------------------------------------------------------------------
# Location and error plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Exact AST character span plus line/column, carried into diagnostics."""

    span: tuple[int, int] | None = None
    line: int = 0
    col: int = 0

    def resolve(
        self,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> SourceLocation:
        """Overlay explicit arguments on the stored location."""
        if span is None and line is None and col is None:
            return self
        return _location(
            self.span if span is None else span,
            self.line if line is None else line,
            self.col if col is None else col,
        )


def _fail(
    code: str,
    message: str,
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    reason: str | None = None,
    help_text: str | None = None,
    note: str | None = None,
) -> NoReturn:
    raise PillarError(
        code,
        message,
        reason=reason,
        help_text=help_text,
        note=note,
        line=line,
        col=col,
        span=span,
    )


def _location(
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> SourceLocation:
    if type(line) is not int or line < 0:
        _fail("E003", "line must be a non-negative integer", line=0, col=0)
    if type(col) is not int or col < 0:
        _fail("E003", "col must be a non-negative integer", line=line, col=0)
    if span is not None:
        if type(span) is not tuple or len(span) != 2:
            _fail("E003", "span must be a (start, end) tuple", line=line, col=col)
        start, end = span
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            _fail(
                "E003",
                "span must contain non-negative integers with start <= end",
                line=line,
                col=col,
            )
    return SourceLocation(span=span, line=line, col=col)


def _require_bool(value: Any, name: str, loc: SourceLocation) -> bool:
    if type(value) is not bool:
        _fail(
            "E003", f"{name} must be a bool", span=loc.span, line=loc.line, col=loc.col
        )
    return value


def _require_int(
    value: Any,
    name: str,
    loc: SourceLocation,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    code: str = "E003",
) -> int:
    if type(value) is not int:
        _fail(code, f"{name} must be an int", span=loc.span, line=loc.line, col=loc.col)
    if minimum is not None and value < minimum:
        _fail(
            code,
            f"{name} must be >= {minimum}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if maximum is not None and value > maximum:
        _fail(
            code,
            f"{name} must be <= {maximum}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return value


def _require_str(
    value: Any,
    name: str,
    loc: SourceLocation,
    *,
    nonempty: bool = True,
    identifier: bool = False,
    code: str = "E003",
) -> str:
    if type(value) is not str:
        _fail(code, f"{name} must be a str", span=loc.span, line=loc.line, col=loc.col)
    if nonempty and not value:
        _fail(
            code, f"{name} must not be empty", span=loc.span, line=loc.line, col=loc.col
        )
    if identifier and not value.isidentifier():
        _fail(
            code,
            f"{name} must be an identifier",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return value


def _require_iterable(value: Any, name: str, loc: SourceLocation) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        _fail(
            "E003",
            f"{name} must be a non-string iterable",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    try:
        return tuple(value)
    except Exception as error:
        _fail(
            "E003",
            f"{name} must be iterable",
            reason=f"{type(error).__name__}: {error}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )


def _require_mapping(value: Any, name: str, loc: SourceLocation) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        _fail(
            "E003",
            f"{name} must be a mapping",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return value


def _reject_hardware(
    primitive: str,
    loc: SourceLocation,
    *,
    detail: str | None = None,
) -> NoReturn:
    _fail(
        "E007",
        f"{primitive} is not enforceable in this Python reference model",
        reason=detail
        or f"{primitive} requires platform primitives unavailable to Python",
        help_text=(
            "consume the declarative plan outside this module; do not treat the "
            "plan as enforcement"
        ),
        note="hardware execution boundary",
        span=loc.span,
        line=loc.line,
        col=loc.col,
    )


def _digest(*parts: str) -> str:
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# 71. mprotect heap isolation plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeapRegionSpec:
    name: str
    size: int
    permissions: str = "rw"


@dataclass(frozen=True, slots=True)
class MprotectRegion:
    name: str
    base: int
    size: int
    permissions: str
    guard_before: tuple[int, int] | None
    guard_after: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class MprotectIsolationPlan:
    regions: tuple[MprotectRegion, ...]
    page_size: int
    guard_pages: int
    total_layout: int
    location: SourceLocation
    caveat: str = REFERENCE_CAVEAT

    def execute(self) -> NoReturn:
        _reject_hardware("mprotect guard pages", self.location)

    def apply(self) -> NoReturn:
        self.execute()


def _coerce_heap_region(item: Any, loc: SourceLocation, index: int) -> HeapRegionSpec:
    label = f"regions[{index}]"
    if isinstance(item, HeapRegionSpec):
        item = {
            "name": item.name,
            "size": item.size,
            "permissions": item.permissions,
        }
    if isinstance(item, Mapping):
        allowed = {"name", "size", "permissions"}
        unknown = set(item) - allowed
        if unknown:
            _fail(
                "E003",
                f"{label} has unknown keys: {sorted(unknown)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item.get("name"), f"{label}.name", loc, identifier=True)
        size = _require_int(
            item.get("size"), f"{label}.size", loc, minimum=1, code="E001"
        )
        permissions = item.get("permissions", "rw")
    elif isinstance(item, (tuple, list)):
        if not 2 <= len(item) <= 3:
            _fail(
                "E003",
                f"{label} must be (name, size) or (name, size, permissions)",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item[0], f"{label}.name", loc, identifier=True)
        size = _require_int(item[1], f"{label}.size", loc, minimum=1, code="E001")
        permissions = item[2] if len(item) == 3 else "rw"
    else:
        _fail(
            "E003",
            f"{label} must be a HeapRegionSpec, mapping, or tuple",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    permissions = _require_str(permissions, f"{label}.permissions", loc)
    if permissions not in _VALID_PERMISSIONS:
        _fail(
            "E003",
            f"{label}.permissions must be one of {sorted(_VALID_PERMISSIONS)}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return HeapRegionSpec(name=name, size=size, permissions=permissions)


def sandbox_mprotect_heap_isolate(
    regions: Iterable[Any],
    *,
    page_size: int = 4096,
    guard_pages: int = 1,
    max_layout: int = 1 << 48,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> MprotectIsolationPlan:
    """Validate a heap-isolation policy and return a declarative plan.

    Setting ``execute=True`` asks for real ``mprotect`` guard pages and is
    uniformly rejected with ``E007``.  The returned layout is arithmetic only;
    it does not reserve or protect any memory.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    page_size = _require_int(page_size, "page_size", loc, minimum=1, code="E001")
    if page_size & (page_size - 1):
        _fail(
            "E001",
            "page_size must be a power of two",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    guard_pages = _require_int(guard_pages, "guard_pages", loc, minimum=1)
    max_layout = _require_int(
        max_layout, "max_layout", loc, minimum=page_size, code="E001"
    )
    specs = tuple(
        _coerce_heap_region(item, loc, index)
        for index, item in enumerate(_require_iterable(regions, "regions", loc))
    )
    if not specs:
        _fail(
            "E003",
            "regions must contain at least one region",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        _fail(
            "E003",
            "region names must be unique",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )

    guard_size = guard_pages * page_size
    cursor = 0
    measured: list[tuple[HeapRegionSpec, int, int, tuple[int, int]]] = []
    for spec in specs:
        guard_start = cursor
        cursor += guard_size
        if cursor > max_layout:
            _fail(
                "E001",
                "layout exceeds max_layout",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        base = cursor
        size = ((spec.size + page_size - 1) // page_size) * page_size
        end = base + size
        if end > max_layout:
            _fail(
                "E001",
                "layout exceeds max_layout",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        measured.append((spec, base, size, (guard_start, base)))
        cursor = end
    trailing_start = cursor
    cursor += guard_size
    if cursor > max_layout:
        _fail(
            "E001",
            "layout exceeds max_layout",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )

    layout: list[MprotectRegion] = []
    for index, (spec, base, size, guard_before) in enumerate(measured):
        guard_after = (
            measured[index + 1][3]
            if index + 1 < len(measured)
            else (trailing_start, cursor)
        )
        layout.append(
            MprotectRegion(
                name=spec.name,
                base=base,
                size=size,
                permissions=spec.permissions,
                guard_before=guard_before,
                guard_after=guard_after,
            )
        )
    plan = MprotectIsolationPlan(
        regions=tuple(layout),
        page_size=page_size,
        guard_pages=guard_pages,
        total_layout=cursor,
        location=loc,
    )
    if execute:
        plan.execute()
    return plan


# ---------------------------------------------------------------------------
# 72. Declarative syscall allowlist (eBPF / Seatbelt plans)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyscallRule:
    name: str
    action: str = "allow"
    number: int | None = None
    constraints: tuple[str, ...] = ()

    def render(self) -> str:
        number = "symbolic" if self.number is None else str(self.number)
        return f"{self.action}:{self.name}:{number}:{','.join(self.constraints)}"


@dataclass(frozen=True, slots=True)
class SyscallAllowlistPlan:
    rules: tuple[SyscallRule, ...]
    default_action: str
    digest: str
    location: SourceLocation
    caveat: str = REFERENCE_CAVEAT

    def emit(self) -> tuple[SyscallRule, ...]:
        """Return the portable declarative allowlist; no bytecode is produced."""
        return self.rules

    def to_bpf(self) -> NoReturn:
        _reject_hardware(
            "eBPF bytecode generation",
            self.location,
            detail="no eBPF program is generated or loaded by this Python module",
        )

    def to_seatbelt(self) -> NoReturn:
        _reject_hardware(
            "Seatbelt profile generation",
            self.location,
            detail="no macOS sandbox profile is generated or applied",
        )

    def execute(self) -> NoReturn:
        self.to_bpf()


def _coerce_constraints(value: Any, loc: SourceLocation) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_require_iterable(value, "constraints", loc)):
        text = _require_str(item, f"constraints[{index}]", loc)
        if not _CONSTRAINT_RE.fullmatch(text):
            _fail(
                "E003",
                f"constraints[{index}] is not a portable predicate",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        result.append(text)
    return tuple(result)


def _coerce_syscall_rule(
    item: Any,
    loc: SourceLocation,
    index: int,
    *,
    action_default: str = "allow",
) -> SyscallRule:
    label = f"rules[{index}]"
    if isinstance(item, SyscallRule):
        item = {
            "name": item.name,
            "action": item.action,
            "number": item.number,
            "constraints": item.constraints,
        }
    if isinstance(item, str):
        name = _require_str(item, label, loc, identifier=True)
        return SyscallRule(name=name, action=action_default)
    if isinstance(item, Mapping):
        allowed = {"name", "action", "number", "constraints"}
        unknown = set(item) - allowed
        if unknown:
            _fail(
                "E003",
                f"{label} has unknown keys: {sorted(unknown)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item.get("name"), f"{label}.name", loc, identifier=True)
        action = item.get("action", action_default)
        number = item.get("number")
        constraints = _coerce_constraints(item.get("constraints", ()), loc)
    elif isinstance(item, (tuple, list)):
        if not 1 <= len(item) <= 4:
            _fail(
                "E003",
                f"{label} must have between 1 and 4 fields",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item[0], f"{label}.name", loc, identifier=True)
        action: Any = action_default
        number: Any = None
        constraints: tuple[str, ...] = ()
        if len(item) >= 2:
            second = item[1]
            if type(second) is str:
                action = second
            elif type(second) is int:
                number = second
            else:
                _fail(
                    "E003",
                    f"{label} second field must be an action or syscall number",
                    span=loc.span,
                    line=loc.line,
                    col=loc.col,
                )
        if len(item) >= 3:
            number = item[2]
        if len(item) == 4:
            constraints = _coerce_constraints(item[3], loc)
    else:
        _fail(
            "E003",
            f"{label} must be a SyscallRule, str, mapping, or tuple",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )

    action = _require_str(action, f"{label}.action", loc)
    if action not in {"allow", "deny"}:
        _fail(
            "E003",
            f"{label}.action must be 'allow' or 'deny'",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if number is not None:
        number = _require_int(number, f"{label}.number", loc, minimum=0, code="E001")
    return SyscallRule(
        name=name,
        action=action,
        number=number,
        constraints=constraints,
    )


def sandbox_ebpf_syscall_synthesize(
    allow: Iterable[Any],
    *,
    deny: Iterable[Any] = (),
    default_action: str = "deny",
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> SyscallAllowlistPlan:
    """Synthesize a deterministic, architecture-independent allowlist plan.

    Items in ``deny`` default to deny rules, and explicit deny rules override
    allow rules.  Real eBPF or Seatbelt emission is rejected with ``E007``;
    only the declarative rules are produced here.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    default_action = _require_str(default_action, "default_action", loc)
    if default_action not in {"allow", "deny"}:
        _fail(
            "E003",
            "default_action must be 'allow' or 'deny'",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    allow_rules = tuple(
        _coerce_syscall_rule(item, loc, index)
        for index, item in enumerate(_require_iterable(allow, "allow", loc))
    )
    deny_rules = tuple(
        _coerce_syscall_rule(item, loc, index, action_default="deny")
        for index, item in enumerate(_require_iterable(deny, "deny", loc))
    )

    by_name: dict[str, SyscallRule] = {}
    for rule in allow_rules + deny_rules:
        prior = by_name.get(rule.name)
        if prior is None:
            by_name[rule.name] = rule
            continue
        if (
            prior.number != rule.number
            or prior.constraints != rule.constraints
            or (prior.action != rule.action and rule.action == "allow")
        ):
            _fail(
                "E002",
                f"conflicting syscall rules for {rule.name!r}",
                reason=f"{prior.render()} versus {rule.render()}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        if rule.action == "deny":
            by_name[rule.name] = rule

    rules = tuple(by_name[name] for name in sorted(by_name))
    plan = SyscallAllowlistPlan(
        rules=rules,
        default_action=default_action,
        digest=_digest(default_action, *(rule.render() for rule in rules)),
        location=loc,
    )
    if execute:
        plan.execute()
    return plan


# ---------------------------------------------------------------------------
# 73. PKU / MPK domain tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PkuDomainSpec:
    name: str
    pkey: int
    permissions: str = "rw"


@dataclass(frozen=True, slots=True)
class PkuDomainToken:
    name: str
    pkey: int
    permissions: str

    def render(self) -> str:
        return f"{self.name}:pkey={self.pkey}:{self.permissions}"


@dataclass(frozen=True, slots=True)
class PkuTransition:
    index: int
    source: str
    target: str
    token: PkuDomainToken


@dataclass(frozen=True, slots=True)
class PkuDomainPlan:
    tokens: tuple[PkuDomainToken, ...]
    current: str
    transitions: tuple[PkuTransition, ...]
    location: SourceLocation
    caveat: str = REFERENCE_CAVEAT

    def activate(self) -> NoReturn:
        _reject_hardware(
            "MPK/POE domain switching",
            self.location,
            detail="no protection-key register or POE state is written by Python",
        )

    def execute(self) -> NoReturn:
        self.activate()


def _coerce_pku_domain(item: Any, loc: SourceLocation, index: int) -> PkuDomainSpec:
    label = f"domains[{index}]"
    if isinstance(item, PkuDomainSpec):
        item = {
            "name": item.name,
            "pkey": item.pkey,
            "permissions": item.permissions,
        }
    if isinstance(item, Mapping):
        allowed = {"name", "pkey", "permissions"}
        unknown = set(item) - allowed
        if unknown:
            _fail(
                "E003",
                f"{label} has unknown keys: {sorted(unknown)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item.get("name"), f"{label}.name", loc, identifier=True)
        pkey = _require_int(item.get("pkey"), f"{label}.pkey", loc, code="E001")
        permissions = item.get("permissions", "rw")
    elif isinstance(item, (tuple, list)):
        if not 2 <= len(item) <= 3:
            _fail(
                "E003",
                f"{label} must be (name, pkey) or (name, pkey, permissions)",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item[0], f"{label}.name", loc, identifier=True)
        pkey = _require_int(item[1], f"{label}.pkey", loc, code="E001")
        permissions = item[2] if len(item) == 3 else "rw"
    else:
        _fail(
            "E003",
            f"{label} must be a PkuDomainSpec, mapping, or tuple",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    _require_int(pkey, f"{label}.pkey", loc, minimum=0, maximum=15, code="E001")
    permissions = _require_str(permissions, f"{label}.permissions", loc)
    if permissions not in _VALID_PERMISSIONS:
        _fail(
            "E003",
            f"{label}.permissions must be one of {sorted(_VALID_PERMISSIONS)}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return PkuDomainSpec(name=name, pkey=pkey, permissions=permissions)


def _coerce_edge(item: Any, loc: SourceLocation, label: str) -> tuple[str, str]:
    if isinstance(item, Mapping):
        source = item.get("source")
        target = item.get("target")
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        source, target = item
    else:
        _fail(
            "E003",
            f"{label} must be a (source, target) pair",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return (
        _require_str(source, f"{label}.source", loc, identifier=True),
        _require_str(target, f"{label}.target", loc, identifier=True),
    )


def sandbox_pku_mpk_domain_switch(
    domains: Iterable[Any],
    *,
    switches: Iterable[Any] = (),
    allowed_edges: Iterable[Any] = (),
    current: str | None = None,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> PkuDomainPlan:
    """Simulate domain-token transitions and validate an explicit edge policy.

    The simulation is real bookkeeping; actual MPK/POE register switching is
    rejected with ``E007`` at ``execute=True`` or ``plan.activate()``.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    specs = tuple(
        _coerce_pku_domain(item, loc, index)
        for index, item in enumerate(_require_iterable(domains, "domains", loc))
    )
    if not specs:
        _fail(
            "E003",
            "domains must contain at least one domain",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    names = [spec.name for spec in specs]
    keys = [spec.pkey for spec in specs]
    if len(set(names)) != len(names) or len(set(keys)) != len(keys):
        _fail(
            "E001",
            "domain names and protection keys must be unique",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    tokens = tuple(
        PkuDomainToken(name=spec.name, pkey=spec.pkey, permissions=spec.permissions)
        for spec in specs
    )
    token_by_name = {token.name: token for token in tokens}

    if current is None:
        active = names[0]
    else:
        active = _require_str(current, "current", loc, identifier=True)
    if active not in token_by_name:
        _fail(
            "E003",
            f"current domain {active!r} is not declared",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )

    allowed = tuple(
        _coerce_edge(item, loc, f"allowed_edges[{index}]")
        for index, item in enumerate(
            _require_iterable(allowed_edges, "allowed_edges", loc)
        )
    )
    allowed_set = set(allowed)
    for source, target in allowed:
        if source not in token_by_name or target not in token_by_name:
            _fail(
                "E003",
                f"allowed edge {source!r} -> {target!r} names an unknown domain",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )

    transitions: list[PkuTransition] = []
    for index, item in enumerate(_require_iterable(switches, "switches", loc)):
        source, target = _coerce_edge(item, loc, f"switches[{index}]")
        if source not in token_by_name or target not in token_by_name:
            _fail(
                "E003",
                f"switches[{index}] names an unknown domain",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        if source != active:
            _fail(
                "E002",
                f"switches[{index}] starts in {source!r} but simulation is in {active!r}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        if allowed_set and (source, target) not in allowed_set:
            _fail(
                "E002",
                f"domain edge {source!r} -> {target!r} is not allowed",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        transitions.append(
            PkuTransition(
                index=index,
                source=source,
                target=target,
                token=token_by_name[target],
            )
        )
        active = target

    plan = PkuDomainPlan(
        tokens=tokens,
        current=active,
        transitions=tuple(transitions),
        location=loc,
    )
    if execute:
        plan.execute()
    return plan


# ---------------------------------------------------------------------------
# 74. Fat-pointer capability checks (real implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaggedCapability:
    object_id: str
    base: int
    size: int
    offset: int = 0
    generation: int = 1
    alignment: int = 1

    def address(self) -> int:
        return self.base + self.offset

    def render(self) -> str:
        return (
            f"{self.object_id}@{self.address()}+{self.size}"
            f"#gen={self.generation}#align={self.alignment}"
        )

    def at(
        self,
        offset: int,
        *,
        span: tuple[int, int] | None = None,
        line: int = 0,
        col: int = 0,
    ) -> TaggedCapability:
        """Return a capability with a new offset; never widen the allocation."""
        loc = _location(span, line, col)
        value = _require_int(offset, "offset", loc, minimum=0, code="E001")
        if value > self.size:
            _fail(
                "E002",
                "capability offset exceeds the tagged allocation",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        address = self.base + value
        if self.alignment > 1 and address % self.alignment:
            _fail(
                "E001",
                "capability address violates its alignment tag",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return TaggedCapability(
            object_id=self.object_id,
            base=self.base,
            size=self.size,
            offset=value,
            generation=self.generation,
            alignment=self.alignment,
        )


@dataclass(frozen=True, slots=True)
class CapabilityEpoch:
    object_id: str
    generation: int
    alive: bool = True


@dataclass(frozen=True, slots=True)
class BoundsVerification:
    capability: TaggedCapability
    epoch: CapabilityEpoch
    address: int
    access_size: int
    result: str = "proved"


def _validate_capability(
    cap: TaggedCapability, loc: SourceLocation, max_address: int
) -> None:
    _require_str(
        cap.object_id, "capability.object_id", loc, identifier=True, code="E001"
    )
    _require_int(cap.base, "capability.base", loc, minimum=0, code="E001")
    _require_int(cap.size, "capability.size", loc, minimum=1, code="E001")
    _require_int(cap.offset, "capability.offset", loc, minimum=0, code="E001")
    _require_int(cap.generation, "capability.generation", loc, minimum=1, code="E001")
    alignment = _require_int(
        cap.alignment, "capability.alignment", loc, minimum=1, code="E001"
    )
    if alignment & (alignment - 1):
        _fail(
            "E001",
            "capability.alignment must be a power of two",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if cap.base + cap.size > max_address:
        _fail(
            "E001",
            "capability allocation exceeds max_address",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )


def _coerce_capability(item: Any, loc: SourceLocation) -> TaggedCapability:
    if isinstance(item, TaggedCapability):
        return item
    if isinstance(item, Mapping):
        allowed = {"object_id", "base", "size", "offset", "generation", "alignment"}
        unknown = set(item) - allowed
        if unknown:
            _fail(
                "E003",
                f"capability has unknown keys: {sorted(unknown)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return TaggedCapability(
            object_id=item.get("object_id"),
            base=item.get("base"),
            size=item.get("size"),
            offset=item.get("offset", 0),
            generation=item.get("generation", 1),
            alignment=item.get("alignment", 1),
        )
    if isinstance(item, (tuple, list)) and 3 <= len(item) <= 6:
        fields = dict(
            zip(
                ("object_id", "base", "size", "offset", "generation", "alignment"),
                item,
            )
        )
        return TaggedCapability(**fields)
    _fail(
        "E003",
        "capability must be a TaggedCapability, mapping, or 3-6 field tuple",
        span=loc.span,
        line=loc.line,
        col=loc.col,
    )


def _coerce_epoch(item: Any, loc: SourceLocation) -> CapabilityEpoch:
    if isinstance(item, CapabilityEpoch):
        return item
    if isinstance(item, Mapping):
        epoch = CapabilityEpoch(
            object_id=item.get("object_id"),
            generation=item.get("generation"),
            alive=item.get("alive", True),
        )
    elif isinstance(item, (tuple, list)) and 2 <= len(item) <= 3:
        epoch = CapabilityEpoch(
            object_id=item[0],
            generation=item[1],
            alive=item[2] if len(item) == 3 else True,
        )
    else:
        _fail(
            "E003",
            "epoch must be a CapabilityEpoch, mapping, or tuple",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    _require_str(epoch.object_id, "epoch.object_id", loc, identifier=True, code="E008")
    _require_int(epoch.generation, "epoch.generation", loc, minimum=1, code="E008")
    _require_bool(epoch.alive, "epoch.alive", loc)
    return epoch


def sandbox_fat_pointer_bounds_verify(
    capability: Any,
    epoch: Any,
    *,
    access_size: int = 1,
    max_address: int = 1 << 64,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> BoundsVerification:
    """Verify spatial and temporal safety of a tagged capability.

    Spatial failures (out-of-bounds, alignment, overflow) and temporal failures
    (stale generation, retired object, wrong object) are reported with
    ``PillarError``.  Unlike the hardware pillars, these checks are real.
    """
    loc = _location(span, line, col)
    cap = _coerce_capability(capability, loc)
    current_epoch = _coerce_epoch(epoch, loc)
    access_size = _require_int(access_size, "access_size", loc, minimum=1)
    max_address = _require_int(max_address, "max_address", loc, minimum=1, code="E001")
    _validate_capability(cap, loc, max_address)

    address = cap.address()
    if address + access_size > max_address:
        _fail(
            "E001",
            "access exceeds max_address",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if current_epoch.object_id != cap.object_id:
        _fail(
            "E002",
            "capability refers to a different object than its epoch",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if not current_epoch.alive:
        _fail(
            "E002",
            "capability refers to a retired object",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if current_epoch.generation != cap.generation:
        _fail(
            "E002",
            "capability generation is stale",
            reason=f"capability {cap.generation} versus epoch {current_epoch.generation}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if cap.offset + access_size > cap.size:
        _fail(
            "E002",
            "capability access is outside the tagged allocation",
            reason=f"offset {cap.offset} + size {access_size} exceeds {cap.size}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if cap.alignment > 1 and address % cap.alignment:
        _fail(
            "E001",
            "capability address violates its alignment tag",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return BoundsVerification(
        capability=cap,
        epoch=current_epoch,
        address=address,
        access_size=access_size,
    )


# ---------------------------------------------------------------------------
# 75. CFI shadow-stack plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CfiTargetSpec:
    name: str
    return_sites: int
    signature: str = "()"

    def render(self) -> str:
        return f"{self.name}{self.signature}@{self.return_sites}"


@dataclass(frozen=True, slots=True)
class ShadowStackSlot:
    index: int
    target: str
    depth: int


@dataclass(frozen=True, slots=True)
class CfiShadowStackPlan:
    mode: str
    targets: tuple[CfiTargetSpec, ...]
    slots: tuple[ShadowStackSlot, ...]
    digest: str
    location: SourceLocation
    caveat: str = REFERENCE_CAVEAT

    def install(self) -> NoReturn:
        _reject_hardware(
            "CFI shadow stack installation",
            self.location,
            detail="no CET/shadow-stack state can be configured from Python",
        )

    def execute(self) -> NoReturn:
        self.install()


def _coerce_cfi_target(item: Any, loc: SourceLocation, index: int) -> CfiTargetSpec:
    label = f"targets[{index}]"
    if isinstance(item, CfiTargetSpec):
        item = {
            "name": item.name,
            "return_sites": item.return_sites,
            "signature": item.signature,
        }
    if isinstance(item, Mapping):
        allowed = {"name", "return_sites", "signature"}
        unknown = set(item) - allowed
        if unknown:
            _fail(
                "E003",
                f"{label} has unknown keys: {sorted(unknown)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        name = _require_str(item.get("name"), f"{label}.name", loc, identifier=True)
        return_sites = _require_int(
            item.get("return_sites"), f"{label}.return_sites", loc, minimum=0
        )
        signature = _require_str(item.get("signature", "()"), f"{label}.signature", loc)
    elif isinstance(item, (tuple, list)) and 2 <= len(item) <= 3:
        name = _require_str(item[0], f"{label}.name", loc, identifier=True)
        return_sites = _require_int(item[1], f"{label}.return_sites", loc, minimum=0)
        signature = (
            _require_str(item[2], f"{label}.signature", loc) if len(item) == 3 else "()"
        )
    else:
        _fail(
            "E003",
            f"{label} must be a CfiTargetSpec, mapping, or tuple",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return CfiTargetSpec(name=name, return_sites=return_sites, signature=signature)


def sandbox_cfi_shadow_stack_emit(
    targets: Iterable[Any],
    *,
    mode: str = "both",
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> CfiShadowStackPlan:
    """Emit a deterministic shadow-stack slot plan for protected targets.

    No control-flow integrity state is installed.  ``plan.install()`` or
    ``execute=True`` reports ``E007`` rather than pretending to arm a shadow
    stack.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    mode = _require_str(mode, "mode", loc)
    if mode not in {"forward", "return", "both"}:
        _fail(
            "E003",
            "mode must be 'forward', 'return', or 'both'",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    specs = tuple(
        _coerce_cfi_target(item, loc, index)
        for index, item in enumerate(_require_iterable(targets, "targets", loc))
    )
    if not specs:
        _fail(
            "E003",
            "targets must contain at least one function",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        _fail(
            "E003",
            "target names must be unique",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    slots: list[ShadowStackSlot] = []
    for spec in specs:
        for depth in range(1, spec.return_sites + 1):
            slots.append(
                ShadowStackSlot(index=len(slots), target=spec.name, depth=depth)
            )
    plan = CfiShadowStackPlan(
        mode=mode,
        targets=specs,
        slots=tuple(slots),
        digest=_digest(mode, *(spec.render() for spec in specs)),
        location=loc,
    )
    if execute:
        plan.execute()
    return plan


# ---------------------------------------------------------------------------
# 76. Spectre barrier annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpectreSite:
    name: str
    pattern: str


@dataclass(frozen=True, slots=True)
class BarrierAnnotation:
    index: int
    site: str
    pattern: str
    barrier: str
    placement: str


@dataclass(frozen=True, slots=True)
class SpectreBarrierPlan:
    annotations: tuple[BarrierAnnotation, ...]
    location: SourceLocation
    caveat: str = REFERENCE_CAVEAT

    def inject(self) -> NoReturn:
        _reject_hardware(
            "speculation barrier injection",
            self.location,
            detail="no CPU barrier instruction is emitted or executed by Python",
        )

    def execute(self) -> NoReturn:
        self.inject()


def _coerce_spectre_site(item: Any, loc: SourceLocation, index: int) -> SpectreSite:
    label = f"sites[{index}]"
    if isinstance(item, SpectreSite):
        site = item
    elif isinstance(item, Mapping):
        site = SpectreSite(
            name=item.get("name"),
            pattern=item.get("pattern"),
        )
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        site = SpectreSite(name=item[0], pattern=item[1])
    else:
        _fail(
            "E003",
            f"{label} must be a SpectreSite, mapping, or (name, pattern) pair",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    _require_str(site.name, f"{label}.name", loc, identifier=True)
    _require_str(site.pattern, f"{label}.pattern", loc)
    if site.pattern not in _SPECTRE_PLACEMENTS:
        _fail(
            "E003",
            f"{label}.pattern must be one of {sorted(_SPECTRE_PLACEMENTS)}",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    return site


def sandbox_spectre_lfence_inject(
    sites: Iterable[Any],
    *,
    barrier: str = "lfence",
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> SpectreBarrierPlan:
    """Emit barrier annotations for Spectre-sensitive sites.

    The annotations are inert.  Injecting real barriers is rejected with
    ``E007``; no speculation state is changed.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    barrier = _require_str(barrier, "barrier", loc, identifier=True)
    specs = tuple(
        _coerce_spectre_site(item, loc, index)
        for index, item in enumerate(_require_iterable(sites, "sites", loc))
    )
    if not specs:
        _fail(
            "E003",
            "sites must contain at least one site",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        _fail(
            "E003",
            "site names must be unique",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    annotations = tuple(
        BarrierAnnotation(
            index=index,
            site=spec.name,
            pattern=spec.pattern,
            barrier=barrier,
            placement=_SPECTRE_PLACEMENTS[spec.pattern],
        )
        for index, spec in enumerate(specs)
    )
    plan = SpectreBarrierPlan(annotations=annotations, location=loc)
    if execute:
        plan.execute()
    return plan


# ---------------------------------------------------------------------------
# 77. Deterministic fuel meter (real implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FuelSnapshot:
    limit: int
    used: int
    remaining: int
    exhausted: bool


class FuelMeter:
    """A deterministic, replayable loop-fuel counter."""

    def __init__(self, limit: int, used: int, location: SourceLocation) -> None:
        self._limit = limit
        self._used = used
        self.location = location

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._limit - self._used

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def snapshot(self) -> FuelSnapshot:
        return FuelSnapshot(
            limit=self._limit,
            used=self._used,
            remaining=self.remaining,
            exhausted=self.exhausted,
        )

    def consume(
        self,
        cost: int = 1,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> FuelSnapshot:
        """Atomically charge ``cost`` units; failed charges leave state intact."""
        loc = self.location.resolve(span=span, line=line, col=col)
        value = _require_int(cost, "cost", loc, minimum=0)
        if value > self.remaining:
            _fail(
                "E002",
                "fuel exhausted",
                reason=f"cost {value} exceeds remaining fuel {self.remaining}",
                help_text="reduce the loop bound or supply a larger fuel limit",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        self._used += value
        return self.snapshot()

    def reset(
        self,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> FuelSnapshot:
        """Rewind the meter for a deterministic replay."""
        self.location.resolve(span=span, line=line, col=col)
        self._used = 0
        return self.snapshot()


def sandbox_deterministic_fuel_meter(
    limit: int,
    *,
    initial_used: int = 0,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> FuelMeter:
    """Create a deterministic meter with an optional replay checkpoint."""
    loc = _location(span, line, col)
    value = _require_int(limit, "limit", loc, minimum=0)
    used = _require_int(initial_used, "initial_used", loc, minimum=0, maximum=value)
    return FuelMeter(limit=value, used=used, location=loc)


# ---------------------------------------------------------------------------
# 78. Bit-level taint tracking (real implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaintedValue:
    value: Any
    taint: int = 0

    @property
    def clean(self) -> bool:
        return self.taint == 0


@dataclass(frozen=True, slots=True)
class TaintSinkCheck:
    sink: str
    labels: tuple[str, ...]
    result: str = "allowed"


def _add_all(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def _mul_all(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = result * value
    return result


_OPERATORS: Final[dict[str, tuple[Callable[..., Any], int]]] = {
    "add": (_add_all, 0),
    "mul": (_mul_all, 0),
    "sub": (operator.sub, 2),
    "truediv": (operator.truediv, 2),
    "floordiv": (operator.floordiv, 2),
    "mod": (operator.mod, 2),
    "pow": (operator.pow, 2),
    "lshift": (operator.lshift, 2),
    "rshift": (operator.rshift, 2),
    "bitand": (operator.and_, 2),
    "bitor": (operator.or_, 2),
    "bitxor": (operator.xor, 2),
    "eq": (operator.eq, 2),
    "ne": (operator.ne, 2),
    "lt": (operator.lt, 2),
    "le": (operator.le, 2),
    "gt": (operator.gt, 2),
    "ge": (operator.ge, 2),
    "neg": (operator.neg, 1),
    "pos": (operator.pos, 1),
    "abs": (operator.abs, 1),
    "invert": (operator.invert, 1),
    "not_": (operator.not_, 1),
}


def _taint_bits(value: Any, loc: SourceLocation, seen: set[int] | None = None) -> int:
    if isinstance(value, TaintedValue):
        bits = _require_int(value.taint, "taint", loc, minimum=0, code="E008")
        return bits | _taint_bits(value.value, loc, seen)
    if value is None or isinstance(
        value, (str, bytes, bytearray, int, float, complex, bool)
    ):
        return 0
    active = set() if seen is None else seen
    identity = id(value)
    if identity in active:
        _fail(
            "E008",
            "cyclic container in taint state",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )
    if isinstance(value, Mapping):
        active.add(identity)
        bits = 0
        for key, item in value.items():
            bits |= _taint_bits(key, loc, active) | _taint_bits(item, loc, active)
        active.discard(identity)
        return bits
    if isinstance(value, (tuple, list, set, frozenset)):
        active.add(identity)
        bits = 0
        for item in value:
            bits |= _taint_bits(item, loc, active)
        active.discard(identity)
        return bits
    return 0


class TaintEngine:
    """Deterministic bit-level taint propagation with dynamic-sink checks."""

    def __init__(
        self,
        labels: Mapping[str, int] | None = None,
        location: SourceLocation | None = None,
    ) -> None:
        self.location = location or SourceLocation()
        self._labels: dict[str, int] = {}
        if labels is not None:
            mapping = _require_mapping(labels, "sources", self.location)
            for label, bit in mapping.items():
                name = _require_str(
                    label, "sources label", self.location, identifier=True, code="E008"
                )
                value = _require_int(
                    bit, f"sources[{name!r}]", self.location, minimum=1, code="E008"
                )
                if value & (value - 1):
                    _fail(
                        "E008",
                        f"taint bit for {name!r} is not a single bit",
                        span=self.location.span,
                        line=self.location.line,
                        col=self.location.col,
                    )
                if name in self._labels or value in self._labels.values():
                    _fail(
                        "E008",
                        "taint source labels and bits must be unique",
                        span=self.location.span,
                        line=self.location.line,
                        col=self.location.col,
                    )
                self._labels[name] = value
        self._next_bit = max(self._labels.values(), default=0) + 1

    @property
    def sources(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._labels.items(), key=lambda item: item[1]))

    def taint_of(
        self,
        value: Any,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> int:
        loc = self.location.resolve(span=span, line=line, col=col)
        return _taint_bits(value, loc)

    def labels_for(
        self,
        taint: int,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> tuple[str, ...]:
        loc = self.location.resolve(span=span, line=line, col=col)
        bits = _require_int(taint, "taint", loc, minimum=0, code="E008")
        labels: list[str] = []
        remaining = bits
        for name, bit in self.sources:
            if bits & bit:
                labels.append(name)
            remaining &= ~bit
        bit = 1
        while remaining:
            if remaining & bit:
                labels.append(f"bit:{bit.bit_length() - 1}")
            remaining &= ~bit
            bit <<= 1
        return tuple(labels)

    def source(
        self,
        label: str,
        value: Any,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> TaintedValue:
        loc = self.location.resolve(span=span, line=line, col=col)
        name = _require_str(label, "label", loc, identifier=True)
        if name not in self._labels:
            bit = self._next_bit
            used = set(self._labels.values())
            while bit in used:
                bit <<= 1
            self._labels[name] = bit
            self._next_bit = bit << 1
        bits = self.taint_of(value, span=loc.span, line=loc.line, col=loc.col)
        return TaintedValue(value=value, taint=bits | self._labels[name])

    def combine(
        self,
        op: str,
        *values: Any,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> TaintedValue:
        """Apply an operator and OR every operand's taint into the result."""
        loc = self.location.resolve(span=span, line=line, col=col)
        name = _require_str(op, "op", loc)
        entry = _OPERATORS.get(name)
        if entry is None:
            _fail(
                "E003",
                f"unknown taint operator {name!r}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        function, arity = entry
        if arity and len(values) != arity:
            _fail(
                "E003",
                f"operator {name!r} needs {arity} operands",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        if not arity and not values:
            _fail(
                "E003",
                f"operator {name!r} needs at least one operand",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        bits = 0
        raw: list[Any] = []
        for value in values:
            bits |= self.taint_of(value, span=loc.span, line=loc.line, col=loc.col)
            raw.append(value.value if isinstance(value, TaintedValue) else value)
        try:
            result = function(*raw)
        except Exception as error:
            _fail(
                "E001",
                f"taint operator {name!r} failed",
                reason=f"{type(error).__name__}: {error}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return TaintedValue(value=result, taint=bits)

    def project(
        self,
        container: Any,
        key: Any,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> TaintedValue:
        """Project an index/key while preserving container and index taint."""
        loc = self.location.resolve(span=span, line=line, col=col)
        bits = self.taint_of(container, span=loc.span, line=loc.line, col=loc.col)
        bits |= self.taint_of(key, span=loc.span, line=loc.line, col=loc.col)
        base = container.value if isinstance(container, TaintedValue) else container
        index = key.value if isinstance(key, TaintedValue) else key
        try:
            result = base[index]
        except Exception as error:
            _fail(
                "E001",
                "taint projection failed",
                reason=f"{type(error).__name__}: {error}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return TaintedValue(value=result, taint=bits)

    def _sink_name(self, sink: Any, loc: SourceLocation) -> str:
        if isinstance(sink, str):
            name = _require_str(sink, "sink", loc).lower()
        elif callable(sink):
            name = getattr(sink, "__name__", "")
            if (
                sink not in (builtins.exec, builtins.eval, builtins.compile)
                and name not in DYNAMIC_CODE_SINKS
            ):
                _fail(
                    "E003",
                    "callable sinks must be a dynamic-code builtin",
                    span=loc.span,
                    line=loc.line,
                    col=loc.col,
                )
            name = name.lower()
        else:
            _fail(
                "E003",
                "sink must be a name or dynamic-code builtin",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        if name not in DYNAMIC_CODE_SINKS:
            _fail(
                "E003",
                f"sink {name!r} is not a dynamic-code sink",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return name

    def enforce_sink(
        self,
        sink: Any,
        value: Any,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> TaintSinkCheck:
        """Block tainted input to exec/eval/compile; never execute the sink."""
        loc = self.location.resolve(span=span, line=line, col=col)
        name = self._sink_name(sink, loc)
        bits = self.taint_of(value, span=loc.span, line=loc.line, col=loc.col)
        if bits:
            labels = self.labels_for(bits, span=loc.span, line=loc.line, col=loc.col)
            _fail(
                "E008",
                f"{name} sink received tainted input",
                reason=f"taint bits {bits:#x} originate from {list(labels)}",
                help_text="do not pass tainted data to a dynamic-code sink",
                note="dynamic code sink boundary",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        return TaintSinkCheck(sink=name, labels=())


def sandbox_taint_flow_enforce(
    sources: Mapping[str, int] | None = None,
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> TaintEngine:
    """Create a taint engine with optional preassigned single-bit sources."""
    loc = _location(span, line, col)
    return TaintEngine(labels=sources, location=loc)


# ---------------------------------------------------------------------------
# 79. Best-effort secret zeroization (real wrapper, explicit caveat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZeroizationReceipt:
    secret_type: str
    length: int
    zeroized: bool
    zeroized_bytes: int
    caveat: str
    reason: str


class SecretDrop:
    """Wrapper that zeroes supported mutable buffers on ``drop()``."""

    def __init__(
        self,
        secret: Any,
        location: SourceLocation,
        *,
        require_mutable: bool = False,
    ) -> None:
        self.secret = secret
        self.location = location
        self._receipt: ZeroizationReceipt | None = None
        self._kind, self.zeroizable, self._length, self._reason = _classify_secret(
            secret, location
        )
        if require_mutable and not self.zeroizable:
            _fail(
                "E003",
                "secret cannot be zeroized by this Python wrapper",
                reason=self._reason,
                help_text="pass a mutable bytearray, memoryview, array, or list of ints",
                span=location.span,
                line=location.line,
                col=location.col,
            )

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def receipt(self) -> ZeroizationReceipt | None:
        return self._receipt

    def drop(
        self,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> ZeroizationReceipt:
        if self._receipt is not None:
            return self._receipt
        loc = self.location.resolve(span=span, line=line, col=col)
        zeroized_bytes = 0
        if self.zeroizable:
            try:
                zeroized_bytes = self._zeroize(loc)
            except PillarError:
                raise
            except Exception as error:
                _fail(
                    "E007",
                    "secret zeroization failed",
                    reason=f"{type(error).__name__}: {error}",
                    help_text="use a mutable contiguous buffer supported by this wrapper",
                    span=loc.span,
                    line=loc.line,
                    col=loc.col,
                )
        self._receipt = ZeroizationReceipt(
            secret_type=self._kind,
            length=self._length,
            zeroized=self.zeroizable,
            zeroized_bytes=zeroized_bytes,
            caveat=ZEROIZATION_CAVEAT,
            reason=self._reason,
        )
        return self._receipt

    def _zeroize(self, loc: SourceLocation) -> int:
        secret = self.secret
        if self._kind == "bytearray":
            secret[:] = bytes(self._length)
            return self._length
        if self._kind == "memoryview":
            target = secret if secret.format == "B" else secret.cast("B")
            target[:] = bytes(target.nbytes)
            return target.nbytes
        if self._kind in {"list", "array"}:
            for index in range(self._length):
                secret[index] = 0
            return self._length
        _fail(
            "E007",
            "secret kind is not zeroizable",
            span=loc.span,
            line=loc.line,
            col=loc.col,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.drop()
        return False


def _classify_secret(secret: Any, loc: SourceLocation) -> tuple[str, bool, int, str]:
    try:
        length = len(secret)
    except Exception:
        length = 0
    length = _require_int(length, "secret length", loc, minimum=0, code="E008")
    if isinstance(secret, bytearray):
        return (
            "bytearray",
            True,
            length,
            "mutable bytearray can be overwritten in place",
        )
    if isinstance(secret, array.array):
        if secret.typecode in "bBhHiIlLqQ":
            return "array", True, length, "integer array can be overwritten in place"
        return (
            "array",
            False,
            length,
            "non-integer array is not a byte secret",
        )
    if isinstance(secret, memoryview):
        if not secret.readonly:
            return "memoryview", True, length, "writable memoryview can be overwritten"
        return "memoryview", False, length, "read-only memoryview cannot be overwritten"
    if isinstance(secret, list):
        return "list", True, length, "mutable list slots can be overwritten in place"
    kind = type(secret).__name__
    return (
        kind,
        False,
        length,
        f"{kind} is immutable to this wrapper and cannot be wiped in place",
    )


def sandbox_zeroize_secret_drop(
    secret: Any,
    *,
    require_mutable: bool = False,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
) -> SecretDrop:
    """Wrap ``secret`` for best-effort in-place zeroization.

    The wrapper never claims cryptographic erasure: ``ZEROIZATION_CAVEAT`` is
    attached to every receipt.  Immutable Python values are reported, not
    silently declared wiped.
    """
    loc = _location(span, line, col)
    _require_bool(require_mutable, "require_mutable", loc)
    return SecretDrop(secret, loc, require_mutable=require_mutable)


# ---------------------------------------------------------------------------
# 80. Violation journal and recovery simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    index: int
    kind: str
    message: str
    location: SourceLocation
    strategy: str
    recovered: bool = False
    recovery_note: str = ""


class ViolationJournal:
    """Deterministic journal of sandbox violations and simulated recoveries."""

    def __init__(self, location: SourceLocation | None = None) -> None:
        self.location = location or SourceLocation()
        self._records: list[ViolationRecord] = []

    def record(
        self,
        kind: str,
        message: str,
        *,
        strategy: str = "fallback",
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> ViolationRecord:
        loc = self.location.resolve(span=span, line=line, col=col)
        name = _require_str(kind, "kind", loc, identifier=True)
        text = _require_str(message, "message", loc)
        choice = _require_str(strategy, "strategy", loc)
        if choice not in _RECOVERY_STRATEGIES:
            _fail(
                "E003",
                f"strategy must be one of {sorted(_RECOVERY_STRATEGIES)}",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        entry = ViolationRecord(
            index=len(self._records),
            kind=name,
            message=text,
            location=loc,
            strategy=choice,
        )
        self._records.append(entry)
        return entry

    def recover(
        self,
        index: int,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> ViolationRecord:
        loc = self.location.resolve(span=span, line=line, col=col)
        position = _require_int(index, "index", loc, minimum=0)
        if position >= len(self._records):
            _fail(
                "E006",
                f"violation record {position} does not exist",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        entry = self._records[position]
        if entry.recovered:
            _fail(
                "E003",
                f"violation record {position} was already recovered",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        updated = ViolationRecord(
            index=entry.index,
            kind=entry.kind,
            message=entry.message,
            location=entry.location,
            strategy=entry.strategy,
            recovered=True,
            recovery_note=f"simulated {entry.strategy} recovery; no signal handler installed",
        )
        self._records[position] = updated
        return updated

    def snapshot(self) -> tuple[ViolationRecord, ...]:
        return tuple(self._records)

    def pending(self) -> tuple[ViolationRecord, ...]:
        return tuple(entry for entry in self._records if not entry.recovered)

    def install_signal_handlers(
        self,
        *,
        span: tuple[int, int] | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> NoReturn:
        loc = self.location.resolve(span=span, line=line, col=col)
        _reject_hardware(
            "signal trapping",
            loc,
            detail="no SIGSEGV/SIGILL handler is installed by this Python module",
        )

    def execute(self) -> NoReturn:
        self.install_signal_handlers()


def sandbox_violation_trap_recover(
    violations: Iterable[Any] = (),
    *,
    span: tuple[int, int] | None = None,
    line: int = 0,
    col: int = 0,
    execute: bool = False,
) -> ViolationJournal:
    """Create a violation journal and optionally seed it with records.

    Signal trapping is not available here.  ``execute=True`` or
    ``journal.install_signal_handlers()`` reports ``E007``; recording and
    recovery simulation remain real, deterministic operations.
    """
    loc = _location(span, line, col)
    _require_bool(execute, "execute", loc)
    journal = ViolationJournal(loc)
    for index, item in enumerate(_require_iterable(violations, "violations", loc)):
        label = f"violations[{index}]"
        if isinstance(item, Mapping):
            kind = item.get("kind")
            message = item.get("message")
            strategy = item.get("strategy", "fallback")
        elif isinstance(item, (tuple, list)) and 2 <= len(item) <= 3:
            kind, message = item[0], item[1]
            strategy = item[2] if len(item) == 3 else "fallback"
        else:
            _fail(
                "E003",
                f"{label} must be a mapping or (kind, message, strategy) tuple",
                span=loc.span,
                line=loc.line,
                col=loc.col,
            )
        journal.record(kind, message, strategy=strategy)
    if execute:
        journal.install_signal_handlers()
    return journal


__all__ = [
    "DYNAMIC_CODE_SINKS",
    "REFERENCE_CAVEAT",
    "ZEROIZATION_CAVEAT",
    "BarrierAnnotation",
    "BoundsVerification",
    "CapabilityEpoch",
    "CfiShadowStackPlan",
    "CfiTargetSpec",
    "FuelMeter",
    "FuelSnapshot",
    "HeapRegionSpec",
    "MprotectIsolationPlan",
    "MprotectRegion",
    "PkuDomainPlan",
    "PkuDomainSpec",
    "PkuDomainToken",
    "PkuTransition",
    "SecretDrop",
    "ShadowStackSlot",
    "SourceLocation",
    "SpectreBarrierPlan",
    "SpectreSite",
    "SyscallAllowlistPlan",
    "SyscallRule",
    "TaggedCapability",
    "TaintEngine",
    "TaintSinkCheck",
    "TaintedValue",
    "ViolationJournal",
    "ViolationRecord",
    "ZeroizationReceipt",
    "sandbox_cfi_shadow_stack_emit",
    "sandbox_deterministic_fuel_meter",
    "sandbox_ebpf_syscall_synthesize",
    "sandbox_fat_pointer_bounds_verify",
    "sandbox_mprotect_heap_isolate",
    "sandbox_pku_mpk_domain_switch",
    "sandbox_spectre_lfence_inject",
    "sandbox_taint_flow_enforce",
    "sandbox_violation_trap_recover",
    "sandbox_zeroize_secret_drop",
]
