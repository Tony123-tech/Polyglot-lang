"""Focused tests for Frame K adversarial verification."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from runtime.diagnostics import PillarError
from tools.adversarial import (
    MutantSpec,
    adv_mutate_ast_and_type_constraints,
    adv_run_mutation_score_gate,
    adv_verify_spec_oracle_and_report,
)

SPAN = (3, 9)


class RecordingRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], Path, dict[str, str] | None]] = []

    def __call__(
        self,
        args: tuple[str, ...],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, Path(cwd), env))
        return subprocess.CompletedProcess(
            args, self.returncode, "ok" if self.returncode == 0 else "killed", ""
        )


def test_generation_is_deterministic_and_grammar_bounded() -> None:
    first = adv_mutate_ast_and_type_constraints(seed=11, count=15, max_depth=6)
    second = adv_mutate_ast_and_type_constraints(seed=11, count=15, max_depth=6)

    assert first.digest() == second.digest()
    assert len(first.cases) == 15
    assert first.diagnostics == ()
    assert len({case.case_id for case in first.cases}) == 15
    assert {case.category for case in first.cases} == {
        "boundary-int",
        "deep-nesting",
        "trait-bound",
        "match-guard",
        "error-path",
    }
    assert any("9223372036854775807" in case.source for case in first.cases)
    assert any(
        case.category == "match-guard" and "match" in case.source
        for case in first.cases
    )
    assert "grammar-bounded" in first.as_dict()["caveats"][0]


def test_generation_validates_arguments() -> None:
    with pytest.raises(PillarError) as raised:
        adv_mutate_ast_and_type_constraints(count=-1, span=SPAN)
    assert raised.value.code == "E003"
    with pytest.raises(PillarError) as raised:
        adv_mutate_ast_and_type_constraints(max_depth=0, span=SPAN)
    assert raised.value.code == "E003"


def test_mutation_gate_kills_mutants_in_isolated_workspaces(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner(returncode=1)
    cache = tmp_path / "cache"
    report = adv_run_mutation_score_gate(
        root=Path.cwd(),
        workspace_root=tmp_path / "workspaces",
        cache_root=cache,
        runner=runner,
        threshold=0.66,
    )

    assert report.passed
    assert report.score == 1.0
    assert report.killed == report.total == 3
    assert report.diagnostics == ()
    assert len(runner.calls) == 3
    assert all(call[1].is_relative_to(tmp_path) for call in runner.calls)
    assert all(
        call[2] is not None and "CARGO_TARGET_DIR" in call[2] for call in runner.calls
    )

    mutated = (
        tmp_path
        / "workspaces"
        / "formal-proof-gate"
        / "compiler"
        / "formal"
        / "framework.py"
    )
    assert "if True:" in mutated.read_text(encoding="utf-8")
    original = Path("compiler/formal/framework.py").read_text(encoding="utf-8")
    assert "if report.proved:" in original


def test_surviving_mutants_fail_the_gate(tmp_path: Path) -> None:
    runner = RecordingRunner(returncode=0)
    report = adv_run_mutation_score_gate(
        root=Path.cwd(),
        workspace_root=tmp_path / "workspaces",
        runner=runner,
        threshold=0.8,
    )

    assert not report.passed
    assert report.score == 0.0
    assert all(not result.killed for result in report.results)
    assert "test sensitivity" in report.as_dict()["caveats"][0]

    with pytest.raises(PillarError) as raised:
        adv_run_mutation_score_gate(
            root=Path.cwd(),
            workspace_root=tmp_path / "workspaces",
            runner=RecordingRunner(0),
            threshold=0.8,
            strict=True,
            span=SPAN,
        )
    assert raised.value.code == "E002"


def test_missing_mutant_target_is_diagnostic_not_crash(tmp_path: Path) -> None:
    report = adv_run_mutation_score_gate(
        mutants=[
            MutantSpec(
                mutant_id="missing",
                path="does/not/exist.py",
                original="x",
                replacement="y",
            )
        ],
        root=Path.cwd(),
        workspace_root=tmp_path / "workspaces",
        runner=RecordingRunner(1),
    )

    assert report.total == 0
    assert not report.passed
    assert report.diagnostics[0]["code"] == "E006"


def test_mutation_gate_requires_isolated_root() -> None:
    with pytest.raises(PillarError) as raised:
        adv_run_mutation_score_gate()
    assert raised.value.code == "E003"


def test_spec_oracle_compares_observations_and_reports_steps() -> None:
    report = adv_verify_spec_oracle_and_report(
        {
            "literal": (1, 1),
            "callable": lambda: (
                {"code": "E008", "message": "x"},
                {"code": "E008", "message": "x"},
            ),
            "mapping": {"expected": (1, 2), "observed": (1, 3)},
        },
        run_verification=False,
    )

    assert not report.passed
    assert [case.passed for case in report.cases] == [True, True, False]
    assert report.checks == ()
    assert report.diagnostics == ()
    assert report.next_steps == (
        "inspect divergent oracle cases and update the reference spec",
    )


def test_full_verification_records_exact_commands_and_outputs(tmp_path: Path) -> None:
    runner = RecordingRunner(returncode=0)
    cache = tmp_path / "verification-cache"
    report = adv_verify_spec_oracle_and_report(
        {"ok": (0, 0)},
        root=Path.cwd(),
        runner=runner,
        python="python3",
        ruff="ruff",
        cache_root=cache,
    )

    assert report.passed
    assert [check.name for check in report.checks] == [
        "ruff-format",
        "ruff-lint",
        "pytest",
        "docs-sync",
        "docs-lint",
        "cargo",
    ]
    assert report.checks[0].command == ("ruff", "format", "--check", ".")
    assert report.checks[2].command == ("python3", "-m", "pytest", "-q")
    assert report.checks[5].command[0] == "cargo"
    assert all(
        call[2] is not None and call[2]["CARGO_TARGET_DIR"].startswith(str(cache))
        for call in runner.calls
    )
    assert all(check.passed for check in report.checks)
    next_steps = report.next_steps
    assert next_steps[-1] == "extend adversarial cases and re-measure mutation score"


def test_failed_cargo_check_produces_e004_next_step(tmp_path: Path) -> None:
    runner = RecordingRunner(returncode=1)
    report = adv_verify_spec_oracle_and_report(
        {"ok": (True, True)},
        root=Path.cwd(),
        runner=runner,
        cache_root=tmp_path / "cache",
    )

    assert not report.passed
    assert any(item["code"] == "E004" for item in report.diagnostics)
    assert any("cargo" in step for step in report.next_steps)
    assert any(not check.passed for check in report.checks)


def test_invalid_runner_result_is_e004(tmp_path: Path) -> None:
    def bad_runner(args: tuple[str, ...], cwd: Path, env: Any = None) -> object:
        del args, cwd, env
        return "not a process"

    with pytest.raises(PillarError) as raised:
        adv_verify_spec_oracle_and_report(
            {"ok": (1, 1)},
            root=Path.cwd(),
            runner=bad_runner,
            cache_root=tmp_path / "cache",
        )
    assert raised.value.code == "E004"
