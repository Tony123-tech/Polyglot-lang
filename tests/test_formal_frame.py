"""Focused tests for Frame H bounded formal methods."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from compiler.formal import (
    formal_check_relational_refinement,
    formal_require_bounded_proof,
    formal_sym_exec_bounded_paths,
    formal_synthesize_infeasible_path_witness,
)
from compiler.formal import framework as formal_frame
from runtime.diagnostics import PillarError

SPAN = (3, 9)

BRANCH_SOURCE = """
fn main(x: i64) -> i64 {
    if x > 0 { return 1; } else { return 0; }
}
"""

LOOP_SOURCE = """
fn main(x: i64) -> i64 {
    while x > 0 { x -= 1; }
    return x;
}
"""

DEAD_SOURCE = """
fn main(x: i64) -> i64 {
    if x > 0 {
        if x <= 0 { return 99; }
    }
    return x;
}
"""


def test_bounded_symbolic_execution_explores_both_branches() -> None:
    report = formal_sym_exec_bounded_paths(BRANCH_SOURCE, max_steps=32, span=SPAN)

    assert report.function == "main"
    assert report.complete and report.proved
    assert [path.result for path in report.paths] == ["1", "0"]
    assert all(path.feasibility == "feasible" for path in report.paths)
    assert {item["branch"] for item in report.branch_decisions} == {"then", "else"}
    assert report.explored == 2
    payload = report.as_dict()
    assert payload["digest"] == report.digest()
    assert json.loads(json.dumps(payload))["bounds"]["max_steps"] == 32


def test_proof_gate_rejects_unknown_feasibility_with_e002() -> None:
    # Multiplication is outside the interval solver; feasibility is Unknown
    # even though no bound was exhausted.
    source = "fn main(x: i64, y: i64) -> i64 { if x * y > 0 { return 1; } else { return 0; } }"
    report = formal_sym_exec_bounded_paths(source, span=SPAN)

    assert report.complete and not report.proved
    assert any(path.feasibility == "unknown" for path in report.paths)

    with pytest.raises(PillarError) as raised:
        formal_require_bounded_proof(report, span=SPAN)
    assert raised.value.code == "E002"
    assert raised.value.span == SPAN


def test_proof_gate_reports_exhausted_bound_diagnostic() -> None:
    report = formal_sym_exec_bounded_paths(
        LOOP_SOURCE, max_unroll=2, strict=False, span=SPAN
    )
    assert not report.complete and not report.proved

    with pytest.raises(PillarError) as raised:
        formal_require_bounded_proof(report, span=SPAN)
    assert raised.value.code == "E008"


def test_unroll_bound_is_structured_e008_and_strict_can_raise() -> None:
    partial = formal_sym_exec_bounded_paths(
        LOOP_SOURCE, max_unroll=2, strict=False, span=SPAN
    )
    assert any(diag.code == "E008" for diag in partial.diagnostics)
    assert any(path.truncated for path in partial.paths)

    with pytest.raises(PillarError) as raised:
        formal_sym_exec_bounded_paths(LOOP_SOURCE, max_unroll=2, span=SPAN)
    assert raised.value.code == "E008"
    assert "unroll bound" in raised.value.diagnostic.message


def test_solver_timeout_is_structured_e004(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0.0, 10.0))
    monkeypatch.setattr(
        formal_frame, "time", SimpleNamespace(monotonic=lambda: next(clock, 10.0))
    )
    report = formal_sym_exec_bounded_paths(BRANCH_SOURCE, strict=False, span=SPAN)

    assert any(diag.code == "E004" for diag in report.diagnostics)
    assert not report.complete
    with pytest.raises(PillarError) as raised:
        formal_require_bounded_proof(report, span=SPAN)
    assert raised.value.code == "E004"


def test_unsupported_construct_is_e003() -> None:
    with pytest.raises(PillarError) as raised:
        formal_sym_exec_bounded_paths(
            "fn main(x: i64) -> i64 { return external_call(x); }", span=SPAN
        )
    assert raised.value.code == "E003"
    assert raised.value.span is not None


def test_missing_function_is_e006() -> None:
    with pytest.raises(PillarError) as raised:
        formal_sym_exec_bounded_paths(BRANCH_SOURCE, function="missing", span=SPAN)
    assert raised.value.code == "E006"
    assert raised.value.span == SPAN


def test_relational_refinement_proves_bounded_equivalence() -> None:
    report = formal_check_relational_refinement(
        BRANCH_SOURCE, BRANCH_SOURCE, inputs={"x": (-4, 4)}, samples=32, span=SPAN
    )

    assert report.proved_bounded
    assert report.checked == 32
    assert report.divergences == ()
    assert "bounded equivalence" in report.as_dict()["caveats"][0]


def test_relational_refinement_emits_concrete_divergence() -> None:
    changed = BRANCH_SOURCE.replace("x > 0", "x >= 0")
    report = formal_check_relational_refinement(
        BRANCH_SOURCE, changed, inputs={"x": (-2, 2)}, samples=20, span=SPAN
    )

    assert not report.proved_bounded
    assert len(report.divergences) == 1
    witness = report.divergences[0]
    assert witness.inputs == {"x": 0}
    assert witness.left != witness.right

    with pytest.raises(PillarError) as raised:
        formal_check_relational_refinement(
            BRANCH_SOURCE,
            changed,
            inputs={"x": (-2, 2)},
            samples=20,
            strict=True,
            span=SPAN,
        )
    assert raised.value.code == "E002"
    assert "inputs=" in raised.value.diagnostic.reason


def test_relational_refinement_uses_injected_backend_runners() -> None:
    runners = {
        "python": lambda x: x + 1,
        "zig": lambda x: x - 1 if x == 0 else x + 1,
    }
    report = formal_check_relational_refinement(
        BRANCH_SOURCE,
        BRANCH_SOURCE,
        runners=runners,
        inputs={"x": (0, 3)},
        samples=8,
        span=SPAN,
    )

    assert report.checked == 8
    assert len(report.divergences) == 1
    assert report.divergences[0].inputs == {"x": 0}
    assert report.divergences[0].left_backend == "python"
    assert report.divergences[0].right_backend == "zig"


def test_infeasible_path_witness_is_minimized_and_explained() -> None:
    report = formal_synthesize_infeasible_path_witness(DEAD_SOURCE, span=SPAN)

    assert len(report.dead_branches) == 1
    assert len(report.witnesses) == 1
    witness = report.witnesses[0]
    assert witness.constraints == ("(x > 0)", "(x <= 0)")
    assert "x" in witness.intervals
    assert "empty after" in witness.intervals["x"]
    assert report.execution.pruned >= 1
