"""Hard CFG/borrow/RAII edge cases (Frame C, IR half).

The helpers compose with :mod:`compiler.ir.framework` passes and add explicit
dominance/loop-header reporting, branch-aware move summaries, and deterministic
cleanup scheduling (including early returns and error unwinds).

Diagnostic map: ``E002`` for move/borrow or resource-lifetime violations,
``E005`` for malformed control flow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from compiler.ir.framework import (
    BorrowReport,
    MirModule,
    ir_check_borrow_and_move_state,
    ir_insert_raii_drop_points,
    ir_validate_control_flow_graph,
)
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None


def _fail(
    code: str, message: str, *, span: Span = None, reason: str | None = None
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text="simplify the CFG or ensure every path releases its resources",
        note="compiler.ir/hard",
        span=span,
    )


@dataclass
class DominanceReport:
    """Dominance and natural-loop validation for a MIR module."""

    valid: bool
    functions: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "functions": self.functions,
            "issues": list(self.issues),
        }


def ir_verify_dominance_and_loop_headers(
    module: MirModule, *, span: Span = None
) -> DominanceReport:
    """Validate dominance, identify loop headers, and reject malformed CFGs."""
    base = ir_validate_control_flow_graph(module)
    issues = tuple(str(issue.message) for issue in base.issues)
    functions: dict[str, dict[str, Any]] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        reachable = sorted(cfg.reachable())
        if cfg.entry not in reachable:
            issues += (f"{name}: entry block is unreachable",)
        predecessors = cfg.predecessors()
        successors = {label: sorted(cfg.successors(label)) for label in reachable}
        reachable_set = set(reachable)
        # Iterative dominator sets (intersection over predecessors).
        dominators: dict[str, set[str]] = {label: set(reachable) for label in reachable}
        if reachable:
            dominators[cfg.entry] = {cfg.entry}
        changed = True
        while changed:
            changed = False
            for label in reachable:
                if label == cfg.entry:
                    continue
                preds = [
                    pred
                    for pred in predecessors.get(label, ())
                    if pred in reachable_set
                ]
                common = set(reachable)
                for pred in preds:
                    common &= dominators.get(pred, set())
                candidate = common | {label}
                if candidate != dominators[label]:
                    dominators[label] = candidate
                    changed = True
        back_edges = [
            (source, target)
            for source in reachable
            for target in successors.get(source, ())
            if target in dominators.get(source, set())
        ]
        loop_headers = sorted({target for _source, target in back_edges})
        for source, target in back_edges:
            if target not in dominators.get(source, set()):
                issues += (
                    f"{name}: back edge {source}->{target} has a non-dominating header",
                )
        functions[name] = {
            "reachable": reachable,
            "dominators": {
                label: sorted(dominators.get(label, set())) for label in reachable
            },
            "back_edges": [list(edge) for edge in back_edges],
            "loop_headers": loop_headers,
        }
    return DominanceReport(
        valid=base.valid and not issues, functions=functions, issues=issues
    )


@dataclass
class MoveSummary:
    """Branch-aware move/borrow report."""

    report: BorrowReport
    merge_states: dict[str, dict[str, str]]
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.report.ok and not self.violations,
            "issues": [issue.message for issue in self.report.issues],
            "merge_states": self.merge_states,
            "violations": list(self.violations),
        }


def ir_track_branch_aware_moves(
    module: MirModule,
    *,
    initial_state: Mapping[str, str] | None = None,
    span: Span = None,
) -> MoveSummary:
    """Check moves on every branch and summarize join-point states."""
    report: BorrowReport = ir_check_borrow_and_move_state(module)
    violations = [issue.message for issue in report.issues]
    merge_states: dict[str, dict[str, str]] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        predecessors = cfg.predecessors()
        for label in sorted(cfg.reachable()):
            preds = [pred for pred in predecessors.get(label, ()) if pred != cfg.entry]
            if len(preds) < 2:
                continue
            states = [
                {
                    key: value
                    for key, value in report.states.get(pred, {}).items()
                    if key in (initial_state or {}) or initial_state is None
                }
                for pred in preds
            ]
            merged: dict[str, str] = {}
            for state in states:
                for variable, state_name in state.items():
                    if variable in merged and merged[variable] != state_name:
                        violations.append(
                            f"{name}:{label} joins divergent states for {variable}: "
                            f"{merged[variable]} vs {state_name}"
                        )
                    merged[variable] = state_name
            merge_states[f"{name}:{label}"] = merged
    if violations:
        raise _fail(
            "E002",
            f"branch-aware move analysis found {len(violations)} violation(s)",
            span=span,
            reason=violations[0],
        )
    return MoveSummary(
        report=report, merge_states=merge_states, violations=tuple(violations)
    )


@dataclass
class CleanupReport:
    """Deterministic RAII defer/drop scheduling result."""

    module: MirModule
    drops: tuple[tuple[str, str, str, str], ...]
    leaks: tuple[str, ...]
    zero_leak: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "drops": [list(item) for item in self.drops],
            "leaks": list(self.leaks),
            "zero_leak": self.zero_leak,
        }


def ir_schedule_raii_cleanup(
    module: MirModule,
    *,
    early_returns: Sequence[str] = (),
    unwinds: Sequence[str] = (),
    span: Span = None,
) -> CleanupReport:
    """Insert drops and verify LIFO cleanup on early returns and unwinds."""
    result = ir_insert_raii_drop_points(module)
    drops = tuple(
        (point.function, point.block, point.variable, point.reason)
        for point in result.insertions
    )
    leaks = tuple(
        sorted(
            (set(early_returns) | set(unwinds)) - {block for _, block, _, _ in drops}
        )
    )
    if leaks:
        raise _fail(
            "E002",
            f"resource leak on cleanup path(s): {', '.join(sorted(leaks))}",
            span=span,
            reason="a return or unwind path has no matching drop",
        )
    return CleanupReport(module=result.module, drops=drops, leaks=(), zero_leak=True)


__all__ = [
    "CleanupReport",
    "DominanceReport",
    "MoveSummary",
    "ir_schedule_raii_cleanup",
    "ir_track_branch_aware_moves",
    "ir_verify_dominance_and_loop_headers",
]
