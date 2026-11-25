"""Backend-independent verification helpers for the FRAME 3 rollout.

Each checker inspects the model HIR/MIR dataclasses or an explicit verification
trace and returns a deterministic dictionary.  Checkers never emit code, invoke
a backend, allocate native memory, or execute user expressions.  Structural API
misuse raises :class:`compiler.frontend.errors.PolyglotError` with the shared
``E001``–``E008`` diagnostic codes; source-level findings are reported as plain
issue dictionaries instead of being raised.

Diagnostic conventions used here:

* ``E001`` – ill-formed HIR type or incompatible type unification;
* ``E003`` – API misuse or malformed IR/metadata/trace structure;
* ``E005`` – dependency-order or reachability violation;
* ``E006`` – dangling CFG target;
* ``E007`` – unsupported operand kind or type spelling;
* ``E008`` – serialized MIR does not round-trip.
"""

from __future__ import annotations

import re
from dataclasses import is_dataclass
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError
from compiler.ir import passes
from compiler.ir.framework import (
    HirExpr,
    HirModule,
    HirStmt,
    HirType,
    MirModule,
    ir_check_borrow_and_move_state,
    ir_validate_control_flow_graph,
    ir_verify_dual_backend_invariants,
)
from compiler.ir.hir import hir_module_fingerprint, hir_unify_types, hir_validate_module
from compiler.ir.mir import (
    mir_cfg_fingerprint,
    mir_deserialize,
    mir_predecessors,
    mir_reachable,
    mir_serialize,
    mir_terminator_arity,
)

__all__ = [
    "verify_backend_type_support",
    "verify_borrow_safety",
    "verify_cfg_structure",
    "verify_constant_folds",
    "verify_desugared_control_flow",
    "verify_dual_backend_invariants",
    "verify_function_signatures",
    "verify_hir_module",
    "verify_metadata_values",
    "verify_mir_roundtrip",
    "verify_no_unreachable_blocks",
    "verify_operand_kinds",
    "verify_pass_trace",
    "verify_phi_placement",
    "verify_type_unification",
]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_OPERANDS = frozenset({"var", "const", "unit"})
_FOLD_OPS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "&",
        "|",
        "^",
        "&&",
        "||",
    }
)
_FORBIDDEN_HIR_KINDS = frozenset({"logical_and", "logical_or", "match", "loop"})
_FORBIDDEN_MIR_OPS = frozenset({"logical_and", "logical_or", "match", "loop"})
_ORDER_RULES: tuple[tuple[str, str], ...] = (
    ("pass_propagate_copies", "pass_fold_mir_constants"),
    ("pass_fold_mir_constants", "pass_dedupe_constants"),
    ("pass_dead_branch_elimination", "pass_remove_unreachable"),
    ("pass_remove_unreachable", "pass_validate_cfg"),
    ("pass_simplify_phi", "pass_validate_cfg"),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, **context: Any) -> NoReturn:
    """Raise a structured IR diagnostic with optional context keys.

    Annotated `NoReturn` rather than `None` because it always raises: callers
    end with a bare `_fail(...)` line and rely on it to terminate the branch.
    With `None` a type checker reads every such function as falling off the end
    without a return.
    """
    details = ", ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
    raise PolyglotError(
        code,
        f"{message} ({details})" if details else message,
        phase="ir",
    )


def _issue(
    code: str,
    location: str,
    message: str,
    *,
    severity: str = "error",
) -> dict[str, Any]:
    """Build one sorted-order issue dictionary."""
    return {
        "code": code,
        "location": location,
        "message": message,
        "severity": severity,
    }


def _sorted_issues(issues: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Sort findings by code, location, and message for stable reports."""
    return tuple(
        sorted(
            issues, key=lambda item: (item["code"], item["location"], item["message"])
        )
    )


def _require_hir(module: object) -> HirModule:
    """Return *module* or raise E003 for a non-HIR module."""
    if not isinstance(module, HirModule):
        _fail("E003", "expected a HirModule", value=type(module).__name__)
    return module


def _require_mir(module: object) -> MirModule:
    """Return *module* or raise E003 for a non-MIR module."""
    if not isinstance(module, MirModule):
        _fail("E003", "expected a MirModule", value=type(module).__name__)
    return module


def _require_ir(target: object) -> tuple[str, HirModule | MirModule]:
    """Classify a supported IR module or raise E003."""
    if isinstance(target, HirModule):
        return "hir", target
    if isinstance(target, MirModule):
        return "mir", target
    _fail("E003", "expected a HirModule or MirModule", value=type(target).__name__)


def _require_type(ty: object) -> HirType:
    """Return *ty* or raise E001 for a non-HIR type."""
    if not isinstance(ty, HirType):
        _fail("E001", "expected a HirType", value=type(ty).__name__)
    return ty


def _walk_hir(expr: HirExpr) -> tuple[HirExpr, ...]:
    """Yield a HIR expression and every nested expression in source order."""
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


def _check_type(ty: object, location: str, issues: list[dict[str, Any]]) -> None:
    """Record ill-formed or unsupported types found in one type tree."""
    if not isinstance(ty, HirType):
        issues.append(_issue("E001", location, "expected a HirType"))
        return
    if not ty.backend_supported:
        issues.append(_issue("E007", location, f"unsupported backend type {ty.text!r}"))
    for index, arg in enumerate(ty.args):
        _check_type(arg, f"{location}.args[{index}]", issues)


def _check_span(span: object, location: str, issues: list[dict[str, Any]]) -> None:
    """Record malformed source spans without rejecting absent locations."""
    if span is None:
        return
    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in span
        )
        or span[0] < 0
        or span[1] < span[0]
    ):
        issues.append(_issue("E003", location, "malformed source span"))


def _check_metadata_value(
    value: object,
    location: str,
    issues: list[dict[str, Any]],
    seen: set[int],
) -> int:
    """Validate one metadata payload and return the number of dictionaries seen."""
    if is_dataclass(value):
        return 0
    if value is None or isinstance(value, (str, int, float, bool, HirType)):
        return 0
    if isinstance(value, (tuple, list)):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        return sum(
            _check_metadata_value(item, f"{location}[{index}]", issues, seen)
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        checked = 1
        for key, item in value.items():
            if not isinstance(key, str):
                issues.append(
                    _issue("E003", location, f"metadata key {key!r} is not a string")
                )
            checked += _check_metadata_value(item, f"{location}.{key}", issues, seen)
        return checked
    issues.append(
        _issue("E003", location, f"unsupported metadata value {type(value).__name__}")
    )
    return 0


def _normalize_issues(records: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """Normalize legacy issue dataclasses into plain dictionaries."""
    return _sorted_issues(
        tuple(
            _issue(
                getattr(record, "code", ""),
                getattr(
                    record,
                    "block",
                    getattr(record, "function", ""),
                ),
                getattr(record, "message", ""),
                severity=getattr(record, "severity", "error"),
            )
            for record in records
        )
    )


# ---------------------------------------------------------------------------
# Public verification helpers
# ---------------------------------------------------------------------------


def verify_backend_type_support(target: HirModule | MirModule) -> dict[str, Any]:
    """Check every declared type against the shared backend spelling table."""
    kind, module = _require_ir(target)
    issues: list[dict[str, Any]] = []
    checked = 0
    for name, function in module.functions.items():
        checked += 1
        _check_type(function.ret, f"{name}:return", issues)
        for index, param in enumerate(function.params):
            _check_type(param.ty, f"{name}:params[{index}]", issues)
        if kind == "hir":
            assert isinstance(module, HirModule)
            for node in _walk_hir(function.body):
                _check_type(node.ty, f"{name}:{node.kind}", issues)
                for stmt in node.meta.get("stmts", ()):
                    if isinstance(stmt, HirStmt):
                        _check_type(stmt.ty, f"{name}:{stmt.name or 'stmt'}", issues)
        else:
            assert isinstance(module, MirModule)
            for label, block in function.cfg.blocks.items():
                for index, instr in enumerate(block.instructions):
                    for operand_index, operand in enumerate(instr.args):
                        _check_type(
                            operand.ty,
                            f"{name}:{label}:instr[{index}].args[{operand_index}]",
                            issues,
                        )
                for role in ("cond", "value"):
                    operand = getattr(block.terminator, role)
                    if operand is not None:
                        _check_type(operand.ty, f"{name}:{label}:{role}", issues)
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "checked": checked,
        "unsupported": tuple(
            sorted({issue["message"] for issue in findings if issue["code"] == "E007"})
        ),
    }


def verify_borrow_safety(module: MirModule) -> dict[str, Any]:
    """Report borrow and move findings from the legacy dataflow analysis."""
    root = _require_mir(module)
    report = ir_check_borrow_and_move_state(root)
    issues = _sorted_issues(
        tuple(_issue(issue.code, issue.block, issue.message) for issue in report.issues)
    )
    return {
        "ok": report.ok,
        "issues": issues,
        "states": {key: dict(value) for key, value in sorted(report.states.items())},
        "count": len(issues),
    }


def verify_cfg_structure(module: MirModule) -> dict[str, Any]:
    """Combine structural CFG validation with terminator and target checks."""
    root = _require_mir(module)
    legacy = ir_validate_control_flow_graph(root)
    issues = list(_normalize_issues(legacy.issues))
    arity = mir_terminator_arity(root)
    issues.extend(
        _issue(
            str(item["code"]),
            str(item["block"]),
            str(item["message"]),
            severity=str(item.get("severity", "error")),
        )
        for item in arity["issues"]
    )
    checked = 0
    for name, function in root.functions.items():
        cfg = function.cfg
        for label, block in cfg.blocks.items():
            checked += 1
            for target in block.terminator.targets:
                if target not in cfg.blocks:
                    issues.append(
                        _issue(
                            "E006",
                            f"{name}:{label}",
                            f"dangling target {target!r}",
                        )
                    )
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": legacy.valid and not any(item["code"] == "E006" for item in findings),
        "issues": findings,
        "checked": checked,
        "count": len(findings),
    }


def verify_constant_folds(target: HirModule | MirModule) -> dict[str, Any]:
    """List foldable literal-only operations that a later pass should evaluate."""
    kind, module = _require_ir(target)
    sites: list[dict[str, Any]] = []
    for name, function in module.functions.items():
        if kind == "hir":
            assert isinstance(module, HirModule)
            for node in _walk_hir(function.body):
                if node.kind not in {"bin", "unop"} or not node.args:
                    continue
                op = str(node.value)
                if op not in _FOLD_OPS:
                    continue
                if (node.kind == "bin" and len(node.args) != 2) or (
                    node.kind == "unop" and len(node.args) != 1
                ):
                    continue
                if not all(arg.kind == "literal" for arg in node.args):
                    continue
                sites.append({"function": name, "kind": node.kind, "operator": op})
        else:
            assert isinstance(module, MirModule)
            for label, block in function.cfg.blocks.items():
                for instr in block.instructions:
                    if instr.op not in {"binop", "unop"} or not instr.args:
                        continue
                    op = str(instr.meta.get("operator", ""))
                    if op not in _FOLD_OPS:
                        continue
                    if (instr.op == "binop" and len(instr.args) != 2) or (
                        instr.op == "unop" and len(instr.args) != 1
                    ):
                        continue
                    if not all(arg.kind == "const" for arg in instr.args):
                        continue
                    sites.append({"function": name, "block": label, "operator": op})
    return {
        "ok": not sites,
        "sites": tuple(sites),
        "count": len(sites),
    }


def verify_desugared_control_flow(target: HirModule | MirModule) -> dict[str, Any]:
    """Check that source-level control constructs have been lowered away."""
    kind, module = _require_ir(target)
    issues: list[dict[str, Any]] = []
    checked = 0
    for name, function in module.functions.items():
        if kind == "hir":
            assert isinstance(module, HirModule)
            for node in _walk_hir(function.body):
                checked += 1
                if node.kind in _FORBIDDEN_HIR_KINDS:
                    issues.append(
                        _issue(
                            "E003",
                            f"{name}:{node.kind}",
                            "HIR control construct remains",
                        )
                    )
        else:
            assert isinstance(module, MirModule)
            for label, block in function.cfg.blocks.items():
                checked += 1
                for instr in block.instructions:
                    if instr.op in _FORBIDDEN_MIR_OPS:
                        issues.append(
                            _issue(
                                "E003",
                                f"{name}:{label}",
                                f"MIR opcode {instr.op!r} remains",
                            )
                        )
                if block.terminator.kind in _FORBIDDEN_MIR_OPS:
                    issues.append(
                        _issue(
                            "E003",
                            f"{name}:{label}",
                            f"MIR terminator {block.terminator.kind!r} remains",
                        )
                    )
    findings = _sorted_issues(tuple(issues))
    return {"ok": not findings, "issues": findings, "checked": checked}


def verify_dual_backend_invariants(target: object) -> dict[str, Any]:
    """Run the shared pre-codegen invariant check through a structured report."""
    if not isinstance(target, (str, HirModule, MirModule)):
        _fail(
            "E003",
            "expected source text, a HirModule, or a MirModule",
            value=type(target).__name__,
        )
    try:
        report = ir_verify_dual_backend_invariants(target)
    except (TypeError, ValueError) as error:
        _fail(
            "E003",
            "dual-backend verification rejected its input",
            error=str(error),
        )
    issues = _normalize_issues(report.issues)
    return {
        "ok": report.ok,
        "issues": issues,
        "checked": tuple(sorted(report.checked)),
        "count": len(issues),
    }


def verify_function_signatures(target: HirModule | MirModule) -> dict[str, Any]:
    """Check function identifiers, uniqueness, and declared type spellings."""
    _, module = _require_ir(target)
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, function in module.functions.items():
        if name in seen:
            issues.append(_issue("E003", name, "duplicate function name"))
        seen.add(name)
        if not _IDENTIFIER.fullmatch(name):
            issues.append(_issue("E003", name, "invalid function identifier"))
        _check_type(function.ret, f"{name}:return", issues)
        for index, param in enumerate(function.params):
            if not _IDENTIFIER.fullmatch(param.name):
                issues.append(_issue("E003", name, f"invalid parameter {param.name!r}"))
            _check_type(param.ty, f"{name}:params[{index}]", issues)
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "functions": tuple(sorted(module.functions)),
        "count": len(findings),
    }


def verify_hir_module(module: HirModule) -> dict[str, Any]:
    """Combine HIR structural checks with span validation and a stable digest."""
    root = _require_hir(module)
    result = hir_validate_module(root)
    issues = [
        _issue(str(item["code"]), str(item["function"]), str(item["message"]))
        for item in result["issues"]
    ]
    checked = int(result["functions"])
    for name, function in root.functions.items():
        _check_span(function.span, f"{name}:function", issues)
        for node in _walk_hir(function.body):
            _check_span(node.span, f"{name}:{node.kind}", issues)
            for branch in node.branches:
                _check_span(branch.body.span, f"{name}:branch", issues)
            for stmt in node.meta.get("stmts", ()):
                if isinstance(stmt, HirStmt):
                    _check_span(stmt.span, f"{name}:{stmt.name or 'stmt'}", issues)
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "checked": checked,
        "functions": int(result["functions"]),
        "digest": hir_module_fingerprint(root)["digest"],
    }


def verify_metadata_values(target: HirModule | MirModule) -> dict[str, Any]:
    """Ensure metadata payloads use only deterministic, inspectable value types."""
    kind, module = _require_ir(target)
    issues: list[dict[str, Any]] = []
    checked = 0
    for name, function in module.functions.items():
        checked += _check_metadata_value(
            function.meta, f"{name}:function.meta", issues, set()
        )
        if kind == "hir":
            assert isinstance(module, HirModule)
            for node in _walk_hir(function.body):
                checked += _check_metadata_value(
                    node.meta, f"{name}:{node.kind}.meta", issues, set()
                )
                for index, branch in enumerate(node.branches):
                    checked += _check_metadata_value(
                        branch.meta,
                        f"{name}:{node.kind}.branches[{index}]",
                        issues,
                        set(),
                    )
                for stmt in node.meta.get("stmts", ()):
                    if isinstance(stmt, HirStmt):
                        checked += _check_metadata_value(
                            stmt.meta,
                            f"{name}:{stmt.name or 'stmt'}.meta",
                            issues,
                            set(),
                        )
        else:
            assert isinstance(module, MirModule)
            for label, block in function.cfg.blocks.items():
                checked += _check_metadata_value(
                    block.meta, f"{name}:{label}.meta", issues, set()
                )
                checked += _check_metadata_value(
                    block.terminator.meta,
                    f"{name}:{label}.terminator.meta",
                    issues,
                    set(),
                )
                for index, instr in enumerate(block.instructions):
                    checked += _check_metadata_value(
                        instr.meta, f"{name}:{label}:instr[{index}].meta", issues, set()
                    )
    findings = _sorted_issues(tuple(issues))
    return {"ok": not findings, "issues": findings, "checked": checked}


def verify_mir_roundtrip(module: MirModule) -> dict[str, Any]:
    """Serialize and rebuild a MIR module, then compare canonical payloads."""
    root = _require_mir(module)
    first = mir_serialize(root)
    restored = mir_deserialize(first)["module"]
    second = mir_serialize(restored)
    issues: tuple[dict[str, Any], ...] = ()
    if first != second:
        issues = (_issue("E008", "<module>", "serialized MIR did not round-trip"),)
    return {
        "ok": not issues,
        "issues": issues,
        "digest": mir_cfg_fingerprint(root)["digest"],
        "functions": tuple(sorted(root.functions)),
        "blocks": sum(len(fn.cfg.blocks) for fn in root.functions.values()),
    }


def verify_no_unreachable_blocks(module: MirModule) -> dict[str, Any]:
    """Report CFG blocks that cannot be reached from the configured entry."""
    root = _require_mir(module)
    issues: list[dict[str, Any]] = []
    reachable: list[str] = []
    unreachable: list[str] = []
    for name, function in root.functions.items():
        result = mir_reachable(function.cfg)
        reachable.extend(f"{name}:{label}" for label in result["reachable"])
        unreachable.extend(f"{name}:{label}" for label in result["unreachable"])
        for label in result["unreachable"]:
            issues.append(_issue("E005", f"{name}:{label}", "block is unreachable"))
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "reachable": tuple(reachable),
        "unreachable": tuple(unreachable),
        "count": len(findings),
    }


def verify_operand_kinds(module: MirModule) -> dict[str, Any]:
    """Reject operand kinds that the model MIR format cannot represent."""
    root = _require_mir(module)
    issues: list[dict[str, Any]] = []
    checked = 0
    for name, function in root.functions.items():
        for label, block in function.cfg.blocks.items():
            for instr in block.instructions:
                for index, operand in enumerate(instr.args):
                    checked += 1
                    location = f"{name}:{label}:instr[{index}].args"
                    if getattr(operand, "kind", None) not in _ALLOWED_OPERANDS:
                        issues.append(
                            _issue(
                                "E007",
                                location,
                                f"unsupported operand kind {getattr(operand, 'kind', type(operand).__name__)!r}",
                            )
                        )
            for role in ("cond", "value"):
                operand = getattr(block.terminator, role)
                if operand is None:
                    continue
                checked += 1
                if getattr(operand, "kind", None) not in _ALLOWED_OPERANDS:
                    issues.append(
                        _issue(
                            "E007",
                            f"{name}:{label}:{role}",
                            f"unsupported operand kind {getattr(operand, 'kind', type(operand).__name__)!r}",
                        )
                    )
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "checked": checked,
        "unsupported": tuple(sorted({item["message"] for item in findings})),
    }


def verify_pass_trace(trace: tuple[str, ...]) -> dict[str, Any]:
    """Check pass names and deterministic dependency order in a pass trace."""
    if not isinstance(trace, tuple) or not all(isinstance(name, str) for name in trace):
        _fail("E003", "trace must be a tuple of pass names")
    issues: list[dict[str, Any]] = []
    known = frozenset(passes.__all__)
    positions: dict[str, int] = {}
    for index, name in enumerate(trace):
        positions.setdefault(name, index)
        if name not in known:
            issues.append(_issue("E003", name, "unknown pass name"))
    for before, after in _ORDER_RULES:
        if (
            before in positions
            and after in positions
            and positions[before] > positions[after]
        ):
            issues.append(_issue("E005", after, f"{before} must run before {after}"))
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "trace": trace,
        "count": len(findings),
    }


def verify_phi_placement(module: MirModule) -> dict[str, Any]:
    """Check that phi nodes are leading and match their predecessor metadata."""
    root = _require_mir(module)
    issues: list[dict[str, Any]] = []
    phis = 0
    checked = 0
    for name, function in root.functions.items():
        predecessors = mir_predecessors(function.cfg)["predecessors"]
        for label, block in function.cfg.blocks.items():
            checked += 1
            seen_non_phi = False
            for instr in block.instructions:
                if instr.op != "phi":
                    seen_non_phi = True
                    continue
                phis += 1
                location = f"{name}:{label}"
                if seen_non_phi:
                    issues.append(
                        _issue("E003", location, "phi follows a non-phi instruction")
                    )
                incoming = instr.meta.get("blocks", ())
                if not isinstance(incoming, (tuple, list)):
                    issues.append(
                        _issue("E003", location, "phi blocks metadata is malformed")
                    )
                elif len(incoming) != len(predecessors.get(label, ())):
                    issues.append(
                        _issue(
                            "E003",
                            location,
                            "phi incoming count does not match predecessors",
                        )
                    )
    findings = _sorted_issues(tuple(issues))
    return {
        "ok": not findings,
        "issues": findings,
        "checked": checked,
        "phis": phis,
    }


def verify_type_unification(left: HirType, right: HirType) -> dict[str, Any]:
    """Check whether two HIR types unify under the shared scalar rules."""
    a = _require_type(left)
    b = _require_type(right)
    result = hir_unify_types(a, b)
    issues: tuple[dict[str, Any], ...] = ()
    if not result["compatible"]:
        issues = (
            _issue(
                "E001",
                a.text,
                f"cannot unify {a.text!r} with {b.text!r}",
            ),
        )
    return {
        "ok": result["compatible"],
        "issues": issues,
        "left": a.text,
        "right": b.text,
        "type": result["text"],
        "rule": result["rule"],
    }
