"""Unit tests for the ordered release gate runner.

Only fake runners are used: no real pytest, cargo, ruff, git, or differential
fuzz build is ever spawned.  The scaffold verifier is replaced in-process and
the fuzz runner is injected, so these tests stay fast and deterministic.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from backends.rust.framework import ScaffoldVerification
from tools import release_gate
from tools.release_gate import (
    FUZZ_CONFIG,
    GateOutcome,
    _run_gate_detailed,
    _scaffold_check,
    main,
    run_gate,
)

Command = tuple[str, ...]

CORE_SUITES = (
    "tests/test_point.py",
    "tests/test_generics.py",
    "tests/test_visualizer.py",
    "tests/test_cross_backend_parity.py",
)


class EventRunner:
    """Record command order and replay optional canned results."""

    def __init__(
        self,
        events: list[object],
        responses: dict[Command, subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.events = events
        self.responses = responses or {}
        self.env_values: list[tuple[Command, str | None]] = []

    def __call__(self, args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        self.events.append(args)
        self.env_values.append((args, os.environ.get("CARGO_TARGET_DIR")))
        return self.responses.get(args, subprocess.CompletedProcess(args, 0, "", ""))


class EnvCaptureRunner(EventRunner):
    """EventRunner that also records cargo target dirs seen during a call."""

    def __init__(self, events: list[object]) -> None:
        super().__init__(events)
        self.cargo_env: list[str | None] = []

    def __call__(self, args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        if args and args[0] == "cargo":
            self.cargo_env.append(os.environ.get("CARGO_TARGET_DIR"))
        return super().__call__(args, cwd)


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    cache = tmp_path / "cache"
    return root, cache


def _clean_scaffold(*args: Any, **kwargs: Any) -> ScaffoldVerification:
    del args, kwargs
    return ScaffoldVerification(True, (), (), (), (), ())


def _fuzz_ok(seed: int, cases: int) -> dict[str, Any]:
    return {
        "passed": True,
        "cases": cases,
        "mismatches": [],
        "skipped": 0,
        "seed": seed,
    }


@pytest.fixture(autouse=True)
def _stub_scaffold_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_gate, "rs_verify_scaffold_matches_generator", _clean_scaffold
    )


def _gate_kwargs(root: Path, cache: Path) -> dict[str, Any]:
    return {
        "root": root,
        "runner": EventRunner([]),
        "fuzz_runner": _fuzz_ok,
        "python": "python",
        "ruff": "ruff",
        "cargo": "cargo",
        "cache_root": cache,
        "scope_key": "test-scope",
    }


def _commands(events: Sequence[object]) -> list[Command]:
    return [event for event in events if isinstance(event, tuple)]


def test_gate_stage_order_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = _workspace(tmp_path)
    events: list[object] = []

    def scaffold_stub(*args: Any, **kwargs: Any) -> ScaffoldVerification:
        del args, kwargs
        events.append("scaffold")
        return _clean_scaffold()

    def fuzz_stub(seed: int, cases: int) -> dict[str, Any]:
        events.append(("fuzz", seed, cases))
        return _fuzz_ok(seed, cases)

    monkeypatch.setattr(
        release_gate, "rs_verify_scaffold_matches_generator", scaffold_stub
    )
    report = run_gate(
        **{
            **_gate_kwargs(root, cache),
            "runner": EventRunner(events),
            "fuzz_runner": fuzz_stub,
        }
    )
    assert report.passed

    commands = _commands(events)
    assert commands[:2] == [
        ("ruff", "format", "--check", "."),
        ("ruff", "check", "."),
    ]
    assert events[2] == "scaffold"

    pytest_commands = [
        command for command in commands if command[:3] == ("python", "-m", "pytest")
    ]
    assert pytest_commands == [
        (
            ("python", "-m", "pytest", "-q", suite, "-m", "slow")
            if suite == "tests/test_cross_backend_parity.py"
            else ("python", "-m", "pytest", "-q", suite)
        )
        for suite in CORE_SUITES
    ]

    fuzz_positions = [
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "fuzz"
    ]
    cargo_commands = [
        command for command in commands if command and command[0] == "cargo"
    ]
    docs_commands = [
        command for command in commands if len(command) == 2 and command[0] == "python"
    ]
    git_command = ("git", "status", "--short")

    assert cargo_commands == [
        ("cargo", "check", "--workspace", "--all-targets"),
        (
            "cargo",
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ),
    ]
    assert docs_commands == [
        ("python", "tools/check_docs_sync.py"),
        ("python", "tools/lint_docs.py"),
    ]
    assert git_command in commands

    def first(predicate: Any) -> int:
        return next(index for index, event in enumerate(events) if predicate(event))

    assert first(lambda event: event in pytest_commands) < min(fuzz_positions)
    assert max(
        index for index, event in enumerate(events) if event in pytest_commands
    ) < min(fuzz_positions)
    assert max(fuzz_positions) < first(lambda event: event in cargo_commands)
    assert first(lambda event: event in cargo_commands) < first(
        lambda event: event in docs_commands
    )
    assert max(
        index for index, event in enumerate(events) if event in docs_commands
    ) < first(lambda event: event == git_command)


def test_quick_full_and_default_fuzz_budgets(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    calls: dict[str, list[tuple[int, int]]] = {"quick": [], "full": [], "default": []}

    for mode, mode_calls in calls.items():
        events: list[object] = []

        def fuzz_stub(
            seed: int, cases: int, mode_calls: list[tuple[int, int]] = mode_calls
        ) -> dict[str, Any]:
            mode_calls.append((seed, cases))
            return _fuzz_ok(seed, cases)

        runner = EventRunner(events)
        report = run_gate(
            **{
                **_gate_kwargs(root, cache),
                "mode": mode,
                "runner": runner,
                "fuzz_runner": fuzz_stub,
            }
        )
        assert report.passed
        count, cases = FUZZ_CONFIG[mode]
        assert mode_calls == [
            (release_gate.FUZZ_BASE_SEED + index * release_gate.FUZZ_SEED_STRIDE, cases)
            for index in range(count)
        ]
        assert len({seed for seed, _ in mode_calls}) == count
        full_pytest = ("python", "-m", "pytest", "-q")
        assert (full_pytest in events) is (mode == "full")

    assert [len(calls[mode]) for mode in ("default", "quick", "full")] == [1, 3, 12]
    assert [calls[mode][0][1] for mode in ("default", "quick", "full")] == [8, 6, 24]


def test_all_pass_report_json_and_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, cache = _workspace(tmp_path)
    report = run_gate(**_gate_kwargs(root, cache))
    assert report.passed
    assert report.failed_checks == 0

    monkeypatch.setattr(
        release_gate,
        "_run_gate_detailed",
        lambda **kwargs: (
            report,
            (GateOutcome("ruff-format", "PASS", ("ruff", "format", "--check", ".")),),
        ),
    )
    assert main(["--json", "--root", str(root), "--quick"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert set(payload) == {
        "passed",
        "passed_checks",
        "failed_checks",
        "blockers",
        "checks",
    }
    assert payload == report.as_dict()
    assert all(set(check) == {"name", "ok", "detail"} for check in payload["checks"])
    assert out.lstrip().startswith("{")


def test_failing_fake_check_blocks_gate_and_json_mode_has_no_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, cache = _workspace(tmp_path)
    events: list[object] = []
    failing = subprocess.CompletedProcess(
        ("ruff", "check", "."), 1, "", "lint error on line 7"
    )
    runner = EventRunner(events, {("ruff", "check", "."): failing})
    report = run_gate(**{**_gate_kwargs(root, cache), "runner": runner})
    assert not report.passed
    assert any("ruff:ruff-lint: exit 1" in blocker for blocker in report.blockers)
    assert any("lint error on line 7" in check.detail for check in report.checks)

    monkeypatch.setattr(
        release_gate,
        "_run_gate_detailed",
        lambda **kwargs: (
            report,
            (GateOutcome("ruff-lint", "FAIL", ("ruff", "check", "."), "exit 1"),),
        ),
    )
    assert main(["--root", str(root), "--json"]) == 1
    out = capsys.readouterr().out
    assert json.loads(out)["passed"] is False
    assert "RELEASE GATE" not in out

    assert main(["--root", str(root)]) == 1
    text = capsys.readouterr().out
    assert "RELEASE GATE: FAIL" in text
    assert "ruff-lint" in text
    assert "ruff check ." in text
    assert "FAIL" in text


def test_scaffold_drift_is_translated_into_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = ScaffoldVerification(
        ok=False,
        matches=("Cargo.toml",),
        mismatched=("src/lib.rs",),
        missing=("pyproject.toml",),
        extra=(),
        details=(
            "src/lib.rs: first difference at line 3",
            "pyproject.toml: file is missing",
        ),
    )
    check = _scaffold_check(drifted)
    assert check.name == "scaffold-drift"
    assert check.returncode == 1
    assert "first difference at line 3" in check.stderr
    assert "missing=pyproject.toml" in check.stdout

    monkeypatch.setattr(
        release_gate, "rs_verify_scaffold_matches_generator", lambda *a, **k: drifted
    )
    root, cache = _workspace(tmp_path)
    report = run_gate(**_gate_kwargs(root, cache))
    status = next(
        item for item in report.checks if item.name == "scaffold:scaffold-drift"
    )
    assert status.ok is False
    assert "first difference at line 3" in status.detail
    assert any("scaffold" in blocker for blocker in report.blockers)


def test_cargo_commands_use_isolated_target_dir_and_restore_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    root, cache = _workspace(tmp_path)
    events: list[object] = []
    runner = EnvCaptureRunner(events)
    report = run_gate(**{**_gate_kwargs(root, cache), "runner": runner})
    assert report.passed

    expected_check = cache / "cargo" / "test-scope" / "check"
    expected_clippy = cache / "cargo" / "test-scope" / "clippy"
    assert runner.cargo_env == [str(expected_check), str(expected_clippy)]
    assert all(
        env is None
        for command, env in runner.env_values
        if not command or command[0] != "cargo"
    )
    assert "CARGO_TARGET_DIR" not in os.environ
    assert expected_check.is_dir() and expected_clippy.is_dir()


def test_allow_dirty_skips_worktree_with_explicit_status(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    events: list[object] = []
    report, outcomes = _run_gate_detailed(
        **{
            **_gate_kwargs(root, cache),
            "runner": EventRunner(events),
            "allow_dirty": True,
        }
    )
    assert report.passed
    assert ("git", "status", "--short") not in events
    skipped = [outcome for outcome in outcomes if outcome.name == "git-worktree-clean"]
    assert len(skipped) == 1
    assert skipped[0].status == "SKIP"
    assert "allow-dirty" in skipped[0].detail
    assert any("allow-dirty" in check.name and check.ok for check in report.checks)


def test_missing_fuzzer_script_is_a_clear_blocker(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    assert not (root / release_gate.FUZZ_SCRIPT).exists()
    report = run_gate(**{**_gate_kwargs(root, cache), "fuzz_runner": None})
    assert not report.passed
    assert any("fuzzer script not found" in blocker for blocker in report.blockers)
    status = next(
        item
        for item in report.checks
        if item.name == "diff-fuzz:diff-fuzz:missing-script"
    )
    assert status.ok is False
