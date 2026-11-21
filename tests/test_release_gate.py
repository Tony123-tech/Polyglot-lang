"""Unit tests for the ordered release gate runner.

Only fake runners are used: no real pytest, cargo, ruff, git, or differential
fuzz build is ever spawned.  The scaffold verifier and the in-process audit
verifiers are replaced in-process, and the fuzz runner is injected, so these
tests stay fast and deterministic.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from backends.rust.framework import ScaffoldVerification
from framework_functions import FUNCTIONS
from tools import release_gate
from tools.build_ci import framework as ci_framework
from tools.build_ci.framework import CheckResult
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
HARD_FRAME_SUITES = (
    "tests/test_jit_framework.py",
    "tests/test_wasm_framework.py",
    "tests/test_hard_frames.py",
    "tests/test_three_way_parity.py",
)
RESEARCH_FRAME_SUITES = (
    "tests/test_formal_frame.py",
    "tests/test_termination_frame.py",
    "tests/test_hard_state_frame.py",
    "tests/test_adversarial_frame.py",
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


class ConcurrencyProbe:
    """Fuzz runner that records pair coverage and maximum overlap."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, seed: int, case_count: int) -> dict[str, Any]:
        with self._lock:
            self.calls.append((seed, case_count))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return _fuzz_ok(seed, case_count)
        finally:
            with self._lock:
                self.active -= 1


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


def _parity_ok(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "cases": 12,
        "passed": True,
        "mismatches": [],
        "skipped": {},
        "artifacts": [],
    }


def _research_ok() -> dict[str, Any]:
    return {"passed": True, "cases": 4, "failed_cases": []}


def _pass_check(name: str) -> CheckResult:
    return CheckResult(
        name=name,
        command=(f"in-process:{name}",),
        returncode=0,
        stdout="stubbed",
    )


@pytest.fixture(autouse=True)
def _stub_scaffold_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_gate, "rs_verify_scaffold_matches_generator", _clean_scaffold
    )


@pytest.fixture(autouse=True)
def _stub_in_process_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep gate tests independent of real metadata, docs, and benchmarks."""
    monkeypatch.setattr(
        release_gate,
        "ci_verify_dependency_audit",
        lambda **kwargs: (_pass_check("pip-check"), _pass_check("metadata-audit")),
    )
    monkeypatch.setattr(
        release_gate,
        "ci_verify_doc_links",
        lambda **kwargs: _pass_check("doc-links"),
    )
    monkeypatch.setattr(
        release_gate,
        "ci_verify_benchmark_regression",
        lambda **kwargs: _pass_check("benchmark-regression"),
    )
    monkeypatch.setattr(
        release_gate,
        "ci_verify_release_metadata",
        lambda **kwargs: _pass_check("release-metadata"),
    )


def _gate_kwargs(root: Path, cache: Path) -> dict[str, Any]:
    return {
        "root": root,
        "runner": EventRunner([]),
        "fuzz_runner": _fuzz_ok,
        "parity_runner": _parity_ok,
        "research_runner": _research_ok,
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

    def parity_stub(seed: int) -> dict[str, Any]:
        events.append(("parity", seed))
        return _parity_ok(seed)

    def research_stub() -> dict[str, Any]:
        events.append(("research",))
        return _research_ok()

    monkeypatch.setattr(
        release_gate, "rs_verify_scaffold_matches_generator", scaffold_stub
    )
    report = run_gate(
        **{
            **_gate_kwargs(root, cache),
            "runner": EventRunner(events),
            "fuzz_runner": fuzz_stub,
            "parity_runner": parity_stub,
            "research_runner": research_stub,
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
        for suite in (*CORE_SUITES, *HARD_FRAME_SUITES, *RESEARCH_FRAME_SUITES)
    ]

    parity_positions = [
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "parity"
    ]
    assert parity_positions == [
        next(index for index, other in enumerate(events) if other == ("parity", seed))
        for seed in release_gate.HARD_FRAME_PARITY_SEEDS
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
        ("cargo", "fmt", "--all", "--", "--check"),
        ("cargo", "doc", "--no-deps", "--workspace"),
    ]
    assert docs_commands == [
        ("python", "tools/check_docs_sync.py"),
        ("python", "tools/lint_docs.py"),
    ]
    assert git_command in commands

    def first(predicate: Any) -> int:
        return next(index for index, event in enumerate(events) if predicate(event))

    core_pytest_commands = pytest_commands[: len(CORE_SUITES)]
    hard_pytest_commands = pytest_commands[
        len(CORE_SUITES) : len(CORE_SUITES) + len(HARD_FRAME_SUITES)
    ]
    research_pytest_commands = pytest_commands[
        len(CORE_SUITES) + len(HARD_FRAME_SUITES) :
    ]

    def command_positions(commands: Sequence[Command]) -> list[int]:
        return [index for index, event in enumerate(events) if event in commands]

    core_positions = command_positions(core_pytest_commands)
    hard_positions = command_positions(hard_pytest_commands)
    research_pytest_positions = command_positions(research_pytest_commands)
    research_positions = [
        index for index, event in enumerate(events) if event == ("research",)
    ]
    assert research_positions == [
        next(index for index, other in enumerate(events) if other == ("research",))
    ]
    assert max(core_positions) < min(hard_positions)
    assert max(hard_positions) < min(parity_positions)
    assert max(parity_positions) < min(research_pytest_positions)
    assert max(research_pytest_positions) < research_positions[0]
    assert research_positions[0] < min(fuzz_positions)
    assert max(fuzz_positions) < first(lambda event: event in cargo_commands)
    assert first(lambda event: event in cargo_commands) < first(
        lambda event: event in docs_commands
    )
    assert max(
        index for index, event in enumerate(events) if event in docs_commands
    ) < first(lambda event: event == git_command)
    assert first(lambda event: event == git_command) < first(
        lambda event: event == cargo_commands[2]
    )

    for gate_name in release_gate.GATE_ORDER[7:]:
        assert any(check.name.startswith(f"{gate_name}:") for check in report.checks), (
            gate_name
        )
    assert any(
        check.name == "hard-frames-parity:pytest:tests/test_jit_framework.py"
        for check in report.checks
    )
    assert any(
        check.name == "research-frames:pytest:tests/test_formal_frame.py"
        for check in report.checks
    )
    assert any(
        check.name == "research-frames:adversarial-oracle" for check in report.checks
    )
    assert [
        check.name for check in report.checks if ":three-way-parity:seed-" in check.name
    ] == [
        f"hard-frames-parity:three-way-parity:seed-{seed}"
        for seed in release_gate.HARD_FRAME_PARITY_SEEDS
    ]


def test_hard_frame_parity_mismatch_blocks_gate(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)

    def bad_parity(seed: int) -> dict[str, Any]:
        return {
            "seed": seed,
            "cases": 12,
            "passed": False,
            "mismatches": [{"case": "scalar-add", "backends": ["python", "wasm"]}],
            "skipped": {},
            "artifacts": [],
        }

    report = run_gate(**{**_gate_kwargs(root, cache), "parity_runner": bad_parity})
    assert not report.passed
    status = next(
        item
        for item in report.checks
        if item.name == "hard-frames-parity:three-way-parity:seed-2026"
    )
    assert status.ok is False
    assert "mismatches=1" in status.detail


def test_failed_research_oracle_blocks_gate(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)

    def bad_research() -> dict[str, Any]:
        return {
            "passed": False,
            "cases": 4,
            "failed_cases": ["hard-state-migration"],
        }

    report = run_gate(**{**_gate_kwargs(root, cache), "research_runner": bad_research})
    assert not report.passed
    status = next(
        item
        for item in report.checks
        if item.name == "research-frames:adversarial-oracle"
    )
    assert status.ok is False
    assert "hard-state-migration" in status.detail
    assert any("adversarial-oracle" in blocker for blocker in report.blockers)


def test_malformed_research_report_is_a_harness_error(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    report = run_gate(
        **{**_gate_kwargs(root, cache), "research_runner": lambda: "not-a-dict"}
    )
    assert not report.passed
    status = next(
        item
        for item in report.checks
        if item.name == "research-frames:adversarial-oracle"
    )
    assert status.ok is False
    assert "expected dict" in status.detail


def test_default_research_runner_evaluates_real_frames() -> None:
    report = release_gate._default_research_runner()

    assert report["passed"]
    assert report["cases"] == 4
    assert report["failed_cases"] == []
    assert report["diagnostics"] == []
    assert any("bounded oracle set" in item for item in report["caveats"])


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

    base = cache / "cargo" / "test-scope"
    expected = [str(base / profile) for profile in ("check", "clippy", "fmt", "doc")]
    assert runner.cargo_env == expected
    assert all(
        env is None
        for command, env in runner.env_values
        if not command or command[0] != "cargo"
    )
    assert "CARGO_TARGET_DIR" not in os.environ
    for profile in ("check", "clippy", "fmt", "doc"):
        assert (base / profile).is_dir()


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


def test_parallel_jobs_preserve_order_and_fuzz_pairs(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    results: dict[int, tuple[Any, Any]] = {}
    calls_by_jobs: dict[int, list[tuple[int, int]]] = {}
    for jobs in (1, 3):
        calls: list[tuple[int, int]] = []

        def fuzz_stub(
            seed: int, cases: int, calls: list[tuple[int, int]] = calls
        ) -> dict[str, Any]:
            calls.append((seed, cases))
            return _fuzz_ok(seed, cases)

        results[jobs] = _run_gate_detailed(
            **{
                **_gate_kwargs(root, cache),
                "mode": "quick",
                "jobs": jobs,
                "fuzz_runner": fuzz_stub,
            }
        )
        calls_by_jobs[jobs] = calls

    serial_report, serial_outcomes = results[1]
    parallel_report, parallel_outcomes = results[3]
    assert parallel_report.as_dict() == serial_report.as_dict()
    assert parallel_outcomes == serial_outcomes

    cases = FUZZ_CONFIG["quick"][1]
    expected = [
        (release_gate.FUZZ_BASE_SEED + index * release_gate.FUZZ_SEED_STRIDE, cases)
        for index in range(FUZZ_CONFIG["quick"][0])
    ]
    assert calls_by_jobs[1] == expected
    assert sorted(calls_by_jobs[3]) == expected
    assert [
        outcome.name
        for outcome in parallel_outcomes
        if outcome.name.startswith("diff-fuzz:")
    ] == [f"diff-fuzz:seed-{seed}" for seed, _ in expected]


def test_no_parallel_overrides_jobs(tmp_path: Path) -> None:
    root, cache = _workspace(tmp_path)
    parallel = ConcurrencyProbe()
    serial = ConcurrencyProbe()
    assert run_gate(
        **{
            **_gate_kwargs(root, cache),
            "mode": "quick",
            "jobs": 3,
            "fuzz_runner": parallel,
        }
    ).passed
    assert run_gate(
        **{
            **_gate_kwargs(root, cache),
            "mode": "quick",
            "jobs": 3,
            "no_parallel": True,
            "fuzz_runner": serial,
        }
    ).passed
    assert parallel.max_active > 1
    assert serial.max_active == 1
    assert sorted(parallel.calls) == sorted(serial.calls)


def test_cli_jobs_and_no_parallel_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    def fake_detailed(**kwargs: Any) -> tuple[Any, tuple[GateOutcome, ...]]:
        captured.update(kwargs)
        return ci_framework.ci_generate_release_gate_report(checks={}), ()

    monkeypatch.setattr(release_gate, "_run_gate_detailed", fake_detailed)
    assert main(["--jobs", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert captured["jobs"] == 3
    assert captured["no_parallel"] is False

    assert main(["--jobs", "3", "--no-parallel"]) == 0
    capsys.readouterr()
    assert captured["jobs"] == 3
    assert captured["no_parallel"] is True


def test_ci_verify_rust_format_and_doc_pass_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    crate = root / "native_pillars"
    crate.mkdir()
    (crate / "Cargo.toml").write_text("[package]\nname='probe'\n", encoding="utf-8")
    seen: list[tuple[Command, Path]] = []

    def fake(args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        seen.append((args, cwd))
        return subprocess.CompletedProcess(args, 0, "", "")

    checks = ci_framework.ci_verify_rust_format_and_doc(root=root, runner=fake)
    assert [check.name for check in checks] == ["cargo-fmt", "cargo-doc"]
    assert [check.command for check in checks] == [
        ("cargo", "fmt", "--all", "--", "--check"),
        ("cargo", "doc", "--no-deps", "--workspace"),
    ]
    assert all(cwd == crate for _, cwd in seen)

    (crate / "Cargo.toml").unlink()
    seen.clear()
    ci_framework.ci_verify_rust_format_and_doc(root=root, runner=fake)
    assert all(cwd == root for _, cwd in seen)

    def failing(args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        return subprocess.CompletedProcess(args, 1, "", "rust check failed")

    failed = ci_framework.ci_verify_rust_format_and_doc(root=root, runner=failing)
    assert all(not check.ok for check in failed)


def test_ci_verify_dependency_audit_pass_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="probe"\nlicense="MIT"\nrequires-python=">=3.14"\n'
        '[project.scripts]\nprobe="framework_functions:get_function"\n',
        encoding="utf-8",
    )

    def fake(args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        return subprocess.CompletedProcess(args, 0, "", "")

    checks = ci_framework.ci_verify_dependency_audit(
        root=root, runner=fake, python_executable="python"
    )
    assert [check.name for check in checks] == ["pip-check", "metadata-audit"]
    assert all(check.ok for check in checks)
    assert checks[0].command == ("python", "-m", "pip", "check")

    (root / "pyproject.toml").write_text('[project]\nname="probe"\n', encoding="utf-8")
    failed = ci_framework.ci_verify_dependency_audit(
        root=root, runner=fake, python_executable="python"
    )
    assert failed[1].returncode == 1
    assert "license" in failed[1].stderr
    assert "requires-python" in failed[1].stderr


def test_ci_verify_doc_links_pass_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    docs = root / "docs"
    docs.mkdir()
    (root / "GUIDE.md").write_text("# Guide\n", encoding="utf-8")
    (docs / "overview.md").write_text("# Overview\n", encoding="utf-8")
    (root / "README.md").write_text(
        "[guide](GUIDE.md)\n"
        "[overview](docs/overview.md#intro)\n"
        "[external](https://example.com)\n"
        "[placeholder](<path/to/doc>)\n"
        "```\n[fenced](missing-fenced.md)\n```\n",
        encoding="utf-8",
    )
    passed = ci_framework.ci_verify_doc_links(root=root)
    assert passed.name == "doc-links"
    assert passed.ok

    (root / "NOTES.md").write_text(
        "[bad](missing.md)\n[bad2](docs/also-missing.md)\n", encoding="utf-8"
    )
    failed = ci_framework.ci_verify_doc_links(root=root)
    assert not failed.ok
    assert "2 broken relative link(s)" in failed.stderr
    assert "missing.md" in failed.stderr
    assert "also-missing.md" in failed.stderr
    assert "missing-fenced.md" not in failed.stderr

    (root / "NOTES.md").unlink()
    (root / "CHANGELOG.md").write_text(
        "\n".join(f"[x](missing-{index}.md)" for index in range(12)), encoding="utf-8"
    )
    limited = ci_framework.ci_verify_doc_links(root=root)
    assert "12 broken relative link(s)" in limited.stderr
    assert limited.stderr.count("missing-") == 10


def test_ci_verify_benchmark_regression_pass_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    cache = tmp_path / "cache"
    first = ci_framework.ci_verify_benchmark_regression(root=root, cache_root=cache)
    assert first.name == "benchmark-regression"
    assert first.ok
    assert "baseline recorded" in first.stdout
    baseline = json.loads(
        (cache / ci_framework.BENCHMARK_BASELINE_FILENAME).read_text(encoding="utf-8")
    )
    assert baseline["operations"] > 0

    second = ci_framework.ci_verify_benchmark_regression(root=root, cache_root=cache)
    assert second.ok
    assert "ratio=1.000" in second.stdout

    failed = ci_framework.ci_verify_benchmark_regression(
        root=root, cache_root=cache, threshold=0.5
    )
    assert not failed.ok
    assert "benchmark regression" in failed.stderr


def test_ci_verify_release_metadata_pass_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    modules = sorted({item.module for item in FUNCTIONS})
    includes = sorted(
        "tools.build_ci*" if module.startswith("tools.") else f"{module}*"
        for module in modules
    )

    def write_pyproject(patterns: Sequence[str]) -> None:
        rendered = ", ".join(json.dumps(pattern) for pattern in patterns)
        (root / "pyproject.toml").write_text(
            "[tool.setuptools]\npy-modules=[]\n"
            f"[tool.setuptools.packages.find]\ninclude=[{rendered}]\n",
            encoding="utf-8",
        )

    write_pyproject(includes)
    passed = ci_framework.ci_verify_release_metadata(root=root)
    assert passed.name == "release-metadata"
    assert passed.ok

    includes.remove("compiler.frontend.framework*")
    write_pyproject(includes)
    failed = ci_framework.ci_verify_release_metadata(root=root)
    assert not failed.ok
    assert "compiler" in failed.stderr
