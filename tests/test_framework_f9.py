"""Behavioral tests for FRAME 9 (build cache, isolation, CI gates).

All command-running functions are exercised with an injectable fake runner; no
real pytest, cargo, ruff, or git process is spawned by these unit tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from tools.build_ci.framework import (
    CheckResult,
    ci_compute_cache_scope_key,
    ci_generate_release_gate_report,
    ci_isolate_cargo_target_dir,
    ci_isolate_python_venv_paths,
    ci_prune_stale_cache_entries,
    ci_verify_clean_git_worktree,
    ci_verify_docs_sync_and_lint,
    ci_verify_pytest_parity_suites,
    ci_verify_ruff_format_and_lint,
    ci_verify_rust_checks,
)

Command = tuple[str, ...]


class FakeRunner:
    """Record command invocations and replay configured results."""

    def __init__(
        self,
        responses: Mapping[Command, subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[Command, Path]] = []

    def __call__(self, args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, cwd))
        return self.responses.get(args, subprocess.CompletedProcess(args, 0, "", ""))


def _result(
    args: Command,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_cache_scope_key_is_stable_and_scoped(tmp_path: Path) -> None:
    kwargs = {
        "os_name": "Linux",
        "arch": "X86_64",
        "compiler_name": "RustC",
        "compiler_version": "1.90.0",
        "feature_flags": {"WASM": True, "ffi": False},
    }
    first = ci_compute_cache_scope_key(**kwargs)
    second = ci_compute_cache_scope_key(
        os_name="linux",
        arch="x86_64",
        compiler_name="rustc",
        compiler_version="1.90.0",
        feature_flags={"ffi": False, "wasm": "1"},
    )
    assert first == second
    assert first.startswith("linux-x86_64-rustc-")
    assert len(first.rsplit("-", 1)[1]) == 64

    assert (
        ci_compute_cache_scope_key(**{**kwargs, "compiler_version": "1.91.0"}) != first
    )
    assert ci_compute_cache_scope_key(**{**kwargs, "os_name": "Darwin"}) != first
    assert (
        ci_compute_cache_scope_key(**{**kwargs, "feature_flags": {"wasm": True}})
        != first
    )
    assert tmp_path.exists()  # key computation must not touch the filesystem


def test_cache_scope_key_accepts_flag_sequences_and_extra_dimensions() -> None:
    from_list = ci_compute_cache_scope_key(
        os_name="darwin",
        arch="arm64",
        compiler_version="1.90",
        feature_flags=["simd", "wasm"],
    )
    from_map = ci_compute_cache_scope_key(
        os_name="darwin",
        arch="arm64",
        compiler_version="1.90",
        feature_flags={"simd": True, "wasm": True},
    )
    assert from_list == from_map
    assert (
        ci_compute_cache_scope_key(
            os_name="darwin",
            arch="arm64",
            compiler_version="1.90",
            feature_flags=["simd", "wasm"],
            extra={"profile": "release"},
        )
        != from_list
    )


def test_cargo_target_dir_is_isolated_per_profile(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    env: dict[str, str] = {}
    debug = ci_isolate_cargo_target_dir(
        "test-debug", cache_root=cache, scope_key="macos-arm64-rustc-abc123", env=env
    )
    bench = ci_isolate_cargo_target_dir(
        "bench", cache_root=cache, scope_key="macos-arm64-rustc-abc123", env=env
    )
    assert debug != bench
    assert debug.is_dir() and bench.is_dir()
    assert debug.parent == bench.parent
    assert (
        ci_isolate_cargo_target_dir(
            "test-debug", cache_root=cache, scope_key="macos-arm64-rustc-abc123"
        )
        == debug
    )
    assert env == {"CARGO_TARGET_DIR": str(bench)}


def test_cargo_target_dir_rejects_unsafe_profile(tmp_path: Path) -> None:
    for profile in ("", "../escape", "a/b", "has space"):
        with pytest.raises(ValueError):
            ci_isolate_cargo_target_dir(profile, cache_root=tmp_path, scope_key="scope")


def test_python_venv_paths_stay_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    workspace.mkdir()
    env: dict[str, str] = {}
    paths = ci_isolate_python_venv_paths(
        workspace=workspace,
        cache_root=cache,
        scope_key="scope1",
        env=env,
    )
    assert list(workspace.iterdir()) == []
    for path in paths.values():
        assert not path.is_relative_to(workspace)
        assert path.is_relative_to(cache)
    assert paths["python"].parent.name in {"bin", "Scripts"}
    assert paths["pip_cache"].is_dir()
    assert paths["build_cache"].is_dir()
    assert env["VIRTUAL_ENV"] == str(paths["venv"])
    assert env["PIP_CACHE_DIR"] == str(paths["pip_cache"])
    assert env["PYTHONPYCACHEPREFIX"] == str(paths["pycache"])


def test_python_venv_paths_reject_overlapping_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError):
        ci_isolate_python_venv_paths(
            workspace=workspace,
            cache_root=workspace / "cache",
            scope_key="scope1",
        )
    with pytest.raises(ValueError):
        ci_isolate_python_venv_paths(
            workspace=workspace / "nested",
            cache_root=workspace,
            scope_key="scope1",
        )


def test_prune_removes_expired_entries_and_frees_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    old = cache / "old"
    old.mkdir(parents=True)
    payload = old / "blob.bin"
    payload.write_bytes(b"x" * 256)
    fresh = cache / "fresh.txt"
    fresh.write_bytes(b"keep")
    now = 1_000_000.0
    os.utime(payload, (now - 10_000, now - 10_000))
    os.utime(old, (now - 10_000, now - 10_000))
    os.utime(fresh, (now - 1, now - 1))

    report = ci_prune_stale_cache_entries(cache, max_age_seconds=60, now=now)

    assert not old.exists()
    assert fresh.is_file()
    assert report["removed"] == ["old"]
    assert report["expired_count"] == 1
    assert report["oversized_count"] == 0
    assert report["freed_bytes"] == 256
    assert report["retained_bytes"] == 4
    assert not (cache / ".prune.lock").exists()


def test_prune_evicts_oldest_entries_until_budget_is_met(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    now = 2_000_000.0
    for index in range(3):
        path = cache / f"entry-{index}.bin"
        path.write_bytes(b"y" * 100)
        os.utime(path, (now - 100 + index, now - 100 + index))

    report = ci_prune_stale_cache_entries(
        cache, max_age_seconds=10_000, max_total_bytes=150, now=now
    )

    assert report["removed"] == ["entry-0.bin", "entry-1.bin"]
    assert report["oversized_count"] == 2
    assert report["expired_count"] == 0
    assert report["retained_bytes"] == 100
    assert (cache / "entry-2.bin").is_file()


def test_prune_respects_concurrent_lock(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    entry = cache / "entry.txt"
    entry.write_text("safe", encoding="utf-8")
    lock = cache / ".prune.lock"
    lock.write_text("other-process", encoding="utf-8")

    with pytest.raises(TimeoutError):
        ci_prune_stale_cache_entries(cache, lock_timeout_seconds=0)
    assert entry.is_file()
    assert lock.is_file()


def test_ruff_gate_runs_format_and_lint_commands(tmp_path: Path) -> None:
    lint_args = ("ruff", "check", ".")
    runner = FakeRunner({lint_args: _result(lint_args, 1, "", "F401 violation")})
    checks = ci_verify_ruff_format_and_lint(root=tmp_path, runner=runner)

    assert [args for args, _ in runner.calls] == [
        ("ruff", "format", "--check", "."),
        lint_args,
    ]
    assert all(cwd == tmp_path for _, cwd in runner.calls)
    assert checks[0].ok and not checks[1].ok
    assert "F401" in checks[1].stderr


def test_pytest_parity_gate_covers_all_default_suites(tmp_path: Path) -> None:
    runner = FakeRunner()
    checks = ci_verify_pytest_parity_suites(root=tmp_path, runner=runner)

    assert len(checks) == 4
    assert [check.name for check in checks] == [
        "pytest:tests/test_point.py",
        "pytest:tests/test_generics.py",
        "pytest:tests/test_visualizer.py",
        "pytest:tests/test_cross_backend_parity.py",
    ]
    assert all(
        args == (sys.executable, "-m", "pytest", "-q", check.name.split(":", 1)[1])
        for (args, _), check in zip(runner.calls, checks, strict=True)
    )
    assert all(check.ok for check in checks)


def test_pytest_parity_gate_accepts_custom_suites_and_reports_failures(
    tmp_path: Path,
) -> None:
    suite = "tests/test_native_features.py"
    args = (sys.executable, "-m", "pytest", "-q", suite)
    runner = FakeRunner({args: _result(args, 2, "", "1 failed")})
    checks = ci_verify_pytest_parity_suites(
        root=tmp_path,
        runner=runner,
        python_executable=sys.executable,
        suites=[suite],
    )
    assert checks == (
        CheckResult(
            name=f"pytest:{suite}",
            command=args,
            returncode=2,
            stderr="1 failed",
        ),
    )
    with pytest.raises(ValueError):
        ci_verify_pytest_parity_suites(root=tmp_path, runner=runner, suites=[])


def test_rust_checks_gate_runs_workspace_check_and_clippy(tmp_path: Path) -> None:
    runner = FakeRunner()
    checks = ci_verify_rust_checks(
        root=tmp_path, runner=runner, cargo_executable="cargo"
    )
    assert [args for args, _ in runner.calls] == [
        ("cargo", "check", "--workspace", "--all-targets"),
        ("cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"),
    ]
    assert [check.name for check in checks] == [
        "cargo-check:workspace",
        "cargo-clippy:workspace",
    ]


def test_rust_checks_gate_supports_per_crate_targets(tmp_path: Path) -> None:
    runner = FakeRunner()
    checks = ci_verify_rust_checks(
        root=tmp_path, runner=runner, crates=("polyglot-lang", "native-pillars")
    )
    assert len(checks) == 4
    assert runner.calls[0][0] == (
        "cargo",
        "check",
        "-p",
        "polyglot-lang",
        "--all-targets",
    )
    assert runner.calls[3][0] == (
        "cargo",
        "clippy",
        "-p",
        "native-pillars",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    )
    assert all(check.ok for check in checks)


def test_docs_gate_runs_sync_script_before_lint_script(tmp_path: Path) -> None:
    lint_args = (sys.executable, "tools/lint_docs.py")
    runner = FakeRunner({lint_args: _result(lint_args, 1, "", "doc drift")})
    checks = ci_verify_docs_sync_and_lint(root=tmp_path, runner=runner)

    assert [args for args, _ in runner.calls] == [
        (sys.executable, "tools/check_docs_sync.py"),
        lint_args,
    ]
    assert [check.name for check in checks] == ["docs-sync", "docs-lint"]
    assert checks[0].ok and not checks[1].ok


@pytest.mark.parametrize(
    ("status", "expected_ok", "expected_text"),
    [
        ("", True, ""),
        ("?? tests/tmp_scratch.txt\n", False, "temporary file: tests/tmp_scratch.txt"),
        (" M main.py\n", False, "unexpected change: main.py"),
        ("?? .ruff_cache/x\n", False, "temporary file: .ruff_cache/x"),
    ],
)
def test_git_worktree_gate_flags_temp_files_and_unintended_edits(
    tmp_path: Path, status: str, expected_ok: bool, expected_text: str
) -> None:
    args = ("git", "status", "--short")
    runner = FakeRunner({args: _result(args, 0, status)})
    check = ci_verify_clean_git_worktree(root=tmp_path, runner=runner)
    assert runner.calls == [(args, tmp_path)]
    assert check.ok is expected_ok
    if expected_text:
        assert expected_text in check.stdout
    else:
        assert check.stdout == ""


def test_git_worktree_gate_honors_allowlist_and_patterns(tmp_path: Path) -> None:
    args = ("git", "status", "--short")
    status = " M main.py\n?? docs/generated.md\n"
    runner = FakeRunner({args: _result(args, 0, status)})
    check = ci_verify_clean_git_worktree(
        root=tmp_path,
        runner=runner,
        allowed_paths=["main.py"],
        ignore_patterns=["docs/generated.*"],
    )
    assert check.ok
    assert check.stdout == ""


def test_release_gate_report_counts_failures_and_collects_blockers() -> None:
    ok = CheckResult("fmt", ("ruff", "format"), 0, "ok")
    bad = CheckResult("lint", ("ruff", "check"), 1, "", "F401 unused import")
    report = ci_generate_release_gate_report(
        checks={
            "ruff": (ok, bad),
            "docs": True,
            "git": False,
        },
        blockers=["manual sign-off missing"],
        required_checks=["rust", "ruff"],
    )
    assert report.passed is False
    assert report.passed_checks == 2
    assert report.failed_checks == 3
    assert "ruff:lint: exit 1" in report.blockers
    assert "git: check failed" in report.blockers
    assert "rust: missing required check" in report.blockers
    assert "manual sign-off missing" in report.blockers
    assert len(report.blockers) == len(set(report.blockers))
    payload = report.as_dict()
    assert payload["passed"] is False
    assert payload["passed_checks"] == 2
    assert "RELEASE GATE: FAIL" in report.render()


def test_release_gate_report_passes_when_all_checks_succeed() -> None:
    report = ci_generate_release_gate_report(
        checks={
            "unit": (CheckResult("suite", ("pytest",), 0),),
            "docs": True,
        }
    )
    assert report.passed
    assert report.passed_checks == 2
    assert report.failed_checks == 0
    assert report.blockers == ()
    assert "checks: 2 passed, 0 failed" in report.render()


def test_release_gate_report_flags_empty_check_sequences() -> None:
    report = ci_generate_release_gate_report(checks={"empty": ()})
    assert not report.passed
    assert report.failed_checks == 1
    assert "empty: no results" in report.blockers
