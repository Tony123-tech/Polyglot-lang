"""Behavioral tests for Batch 10 (ids 0901-1000) of the framework rollout.

Command-running helpers are exercised only through injected fake runners; no
real pytest, cargo, ruff, or git process is spawned by this module.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from compiler.frontend.errors import PolyglotError
from tools.build_ci import cache_keys, parity, reports, verifiers
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
from tools.build_ci.rollout import FRAMEWORK_ROWS

Command = tuple[str, ...]
DIGEST = "a" * 64


class FakeRunner:
    """Record injected command calls and replay deterministic responses."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        raises: Exception | None = None,
        by_command: dict[Command, tuple[int, str, str]] | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raises = raises
        self.by_command = by_command or {}
        self.calls: list[tuple[Command, Path, dict[str, str]]] = []

    def __call__(
        self, args: Command, cwd: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, cwd, dict(env)))
        if self.raises is not None:
            raise self.raises
        if args in self.by_command:
            code, out, err = self.by_command[args]
            return subprocess.CompletedProcess(args, code, out, err)
        return subprocess.CompletedProcess(
            args, self.returncode, self.stdout, self.stderr
        )


class FakeLegacyRunner:
    """Two-argument runner matching the legacy FRAME 9 contract."""

    def __init__(self, *, stdout: str = "") -> None:
        self.stdout = stdout
        self.calls: list[tuple[Command, Path]] = []

    def __call__(self, args: Command, cwd: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, cwd))
        return subprocess.CompletedProcess(args, 0, self.stdout, "")


def test_rollout_has_one_hundred_resolvable_documented_rows() -> None:
    assert len(FRAMEWORK_ROWS) == 100
    names = [row[0] for row in FRAMEWORK_ROWS]
    assert len(set(names)) == 100
    assert names[:10] == [
        "ci_compute_cache_scope_key",
        "ci_isolate_cargo_target_dir",
        "ci_isolate_python_venv_paths",
        "ci_prune_stale_cache_entries",
        "ci_verify_ruff_format_and_lint",
        "ci_verify_pytest_parity_suites",
        "ci_verify_rust_checks",
        "ci_verify_docs_sync_and_lint",
        "ci_verify_clean_git_worktree",
        "ci_generate_release_gate_report",
    ]
    expected_modules = {
        "tools.build_ci.cache_keys": "cache",
        "tools.build_ci.verifiers": "verification",
        "tools.build_ci.reports": "reporting",
        "tools.build_ci.parity": "parity",
    }
    for name, module, category, test_module in FRAMEWORK_ROWS:
        assert test_module == "tests.test_build_ci_rollout"
        function = getattr(importlib.import_module(module), name)
        assert callable(function)
        assert function.__doc__
        if module == "tools.build_ci.framework":
            assert category in {"cache", "verification", "reporting"}
        else:
            assert expected_modules[module] == category
    assert len(FRAMEWORK_ROWS[10:]) == 90


def test_each_new_module_exposes_exactly_25_sorted_functions() -> None:
    for module, prefix in (
        (cache_keys, "bk_"),
        (verifiers, "bv_"),
        (reports, "br_"),
        (parity, "bp_"),
    ):
        names = module.__all__
        assert len(names) == 25
        assert names == sorted(names)
        assert all(name.startswith(prefix) for name in names)
        assert all(getattr(module, name).__doc__ for name in names)


def test_legacy_ci_compute_cache_scope_key_smoke() -> None:
    assert ci_compute_cache_scope_key(os_name="Linux", arch="x86_64").startswith(
        "linux-x86_64-rustc-"
    )


def test_legacy_ci_isolate_cargo_target_dir_smoke(tmp_path: Path) -> None:
    result = ci_isolate_cargo_target_dir(
        "debug", cache_root=tmp_path / "cache", scope_key="scope1"
    )
    assert result.is_dir()


def test_legacy_ci_isolate_python_venv_paths_smoke(tmp_path: Path) -> None:
    paths = ci_isolate_python_venv_paths(
        workspace=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
        scope_key="scope1",
    )
    assert set(paths) == {
        "venv",
        "python",
        "pip_cache",
        "build_cache",
        "pytest_cache",
        "pycache",
    }


def test_legacy_ci_prune_stale_cache_entries_smoke(tmp_path: Path) -> None:
    result = ci_prune_stale_cache_entries(tmp_path / "cache", now=1_000.0)
    assert result["removed_count"] == 0


def test_legacy_ci_verify_ruff_format_and_lint_smoke(tmp_path: Path) -> None:
    checks = ci_verify_ruff_format_and_lint(root=tmp_path, runner=FakeLegacyRunner())
    assert [check.name for check in checks] == ["ruff-format", "ruff-lint"]


def test_legacy_ci_verify_pytest_parity_suites_smoke(tmp_path: Path) -> None:
    checks = ci_verify_pytest_parity_suites(
        root=tmp_path,
        runner=FakeLegacyRunner(),
        suites=["tests/test_point.py"],
    )
    assert len(checks) == 1 and checks[0].ok


def test_legacy_ci_verify_rust_checks_smoke(tmp_path: Path) -> None:
    checks = ci_verify_rust_checks(
        root=tmp_path, runner=FakeLegacyRunner(), crates=["core"]
    )
    assert [check.name for check in checks] == [
        "cargo-check:core",
        "cargo-clippy:core",
    ]


def test_legacy_ci_verify_docs_sync_and_lint_smoke(tmp_path: Path) -> None:
    checks = ci_verify_docs_sync_and_lint(root=tmp_path, runner=FakeLegacyRunner())
    assert [check.name for check in checks] == ["docs-sync", "docs-lint"]


def test_legacy_ci_verify_clean_git_worktree_smoke(tmp_path: Path) -> None:
    result = ci_verify_clean_git_worktree(root=tmp_path, runner=FakeLegacyRunner())
    assert isinstance(result, CheckResult) and result.ok


def test_legacy_ci_generate_release_gate_report_smoke() -> None:
    report = ci_generate_release_gate_report(checks={"gate": True})
    assert report.passed and report.passed_checks == 1


def test_bk_assert_no_hash_usage_detects_only_builtin_calls() -> None:
    source = "hash(value)\nhashlib.sha256(b'x')\nobj.hash(value)\n"
    result = cache_keys.bk_assert_no_hash_usage(source)
    assert result["ok"] is False
    assert result["findings"] == [{"line": 1, "text": "hash(value)"}]
    assert result["scanned_lines"] == 3
    assert cache_keys.bk_assert_no_hash_usage("value = 1\n")["ok"] is True
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_assert_no_hash_usage(None)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_bk_canonical_json_is_order_independent() -> None:
    first = cache_keys.bk_canonical_json({"b": 1, "a": [2, 3]})
    second = cache_keys.bk_canonical_json({"a": [2, 3], "b": 1})
    assert first["text"] == '{"a":[2,3],"b":1}'
    assert first == second
    assert len(first["digest"]) == 64
    assert first["bytes"] == len(first["text"].encode("utf-8"))


def test_bk_canonicalize_env_sorts_and_filters() -> None:
    result = cache_keys.bk_canonicalize_env(
        {"B": "2", "SKIP": "x", "A": "1"}, allowed_prefixes=("A", "B")
    )
    assert result["entries"] == [["A", "1"], ["B", "2"]]
    assert result["count"] == 2
    assert len(result["fingerprint"]) == 64
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_canonicalize_env({"A": 1})  # type: ignore[dict-item]
    assert error.value.code == "E003"


def test_bk_canonicalize_flags_accepts_mapping_and_sequence() -> None:
    result = cache_keys.bk_canonicalize_flags({"WASM": True, "ffi": False})
    assert result["entries"] == [["ffi", "0"], ["wasm", "1"]]
    assert result["count"] == 2
    assert cache_keys.bk_canonicalize_flags(["simd", "wasm"])["entries"] == [
        ["simd", "1"],
        ["wasm", "1"],
    ]
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_canonicalize_flags(object())  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_bk_classify_cache_entry_distinguishes_temporary_names() -> None:
    names = (".prune.lock",)
    suffixes = (".log",)
    assert cache_keys.bk_classify_cache_entry(".PRUNE.LOCK", temp_names=names)[
        "temporary"
    ]
    assert (
        cache_keys.bk_classify_cache_entry("run.log", temp_suffixes=suffixes)["reason"]
        == "temporary suffix"
    )
    retained = cache_keys.bk_classify_cache_entry("blob.bin")
    assert retained["temporary"] is False and retained["reason"] == "retained"


def test_bk_combine_keys_builds_prefixed_composite_key() -> None:
    parts = [f"sources-{DIGEST}", f"env-{DIGEST}"]
    result = cache_keys.bk_combine_keys(parts, label="build")
    assert result["key"].startswith("build-")
    assert result["parts"] == parts and result["count"] == 2
    reversed_result = cache_keys.bk_combine_keys(list(reversed(parts)), label="build")
    assert reversed_result["key"] == result["key"]
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_combine_keys(["bad key!"])
    assert error.value.code == "E003"


def test_bk_describe_cache_entry_reports_size_and_age(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"abc")
    result = cache_keys.bk_describe_cache_entry(target, now=1_000.0)
    assert result["name"] == "blob.bin"
    assert result["size_bytes"] == 3
    assert result["digest"] == hashlib.sha256(b"abc").hexdigest()
    assert result["age_seconds"] >= 0
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_describe_cache_entry(tmp_path / "absent")
    assert error.value.code == "E006"


def test_bk_hash_payload_uses_sha256_over_canonical_json() -> None:
    first = cache_keys.bk_hash_payload({"x": 1, "y": [True, None]})
    second = cache_keys.bk_hash_payload({"y": [True, None], "x": 1})
    assert first == second
    assert first["algorithm"] == "sha256"
    assert len(first["digest"]) == 64


def test_bk_key_for_command_includes_argv_cwd_and_env(tmp_path: Path) -> None:
    first = cache_keys.bk_key_for_command(
        ("python", "-m", "pytest"), root=tmp_path, env={"A": "1"}
    )
    second = cache_keys.bk_key_for_command(
        ["python", "-m", "pytest"], root=str(tmp_path), env={"A": "2"}
    )
    assert first["key"].startswith("cmd-")
    assert first["argv"] == ["python", "-m", "pytest"]
    assert first["cwd"] == str(tmp_path)
    assert first["key"] != second["key"]
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_key_for_command("python", root=tmp_path)
    assert error.value.code == "E003"


def test_bk_key_for_env_filters_to_selected_prefixes() -> None:
    result = cache_keys.bk_key_for_env(
        {"CARGO_TARGET_DIR": "/tmp/target", "PATH": "/bin"},
        allowed_prefixes=("CARGO_",),
    )
    assert result["key"].startswith("env-")
    assert result["entries"] == [["CARGO_TARGET_DIR", "/tmp/target"]]
    assert result["count"] == 1


def test_bk_key_for_manifest_is_insertion_order_independent() -> None:
    first = cache_keys.bk_key_for_manifest({"b": 2, "a": 1})
    second = cache_keys.bk_key_for_manifest({"a": 1, "b": 2})
    assert first == second
    assert first["key"] == f"manifest-{first['digest']}"
    assert first["entry_count"] == 2
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_key_for_manifest(["a"])
    assert error.value.code == "E003"


def test_bk_key_for_platform_normalizes_components() -> None:
    result = cache_keys.bk_key_for_platform("Linux", "X86_64", abi="gnu")
    assert result["key"].startswith("platform-")
    assert (result["os"], result["arch"], result["abi"]) == (
        "linux",
        "x86_64",
        "gnu",
    )
    assert cache_keys.bk_key_for_platform("linux", "x86_64")["key"] != result["key"]


def test_bk_key_for_profile_rejects_unsafe_names() -> None:
    result = cache_keys.bk_key_for_profile("test-debug", scope_key="scope1")
    assert result["profile"] == "test-debug"
    assert result["scope_key"] == "scope1"
    assert result["key"].startswith("profile-")
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_key_for_profile("../escape", scope_key="scope1")
    assert error.value.code == "E003"


def test_bk_key_for_sources_hashes_file_contents_in_sorted_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")
    result = cache_keys.bk_key_for_sources(tmp_path)
    assert result["files"] == [
        ["a.py", hashlib.sha256(b"a").hexdigest()],
        ["b.py", hashlib.sha256(b"b").hexdigest()],
    ]
    assert result["count"] == 2
    filtered = cache_keys.bk_key_for_sources(tmp_path, exclude=("b.py",))
    assert filtered["count"] == 1 and filtered["key"] != result["key"]
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_key_for_sources(tmp_path, paths=["missing.py"])
    assert error.value.code == "E006"


def test_bk_key_for_toolchain_normalizes_versions() -> None:
    result = cache_keys.bk_key_for_toolchain({"RustC": " 1.90.0 "})
    assert result["components"] == {"rustc": "1.90.0"}
    assert result["key"] == f"toolchain-{result['digest']}"
    assert cache_keys.bk_key_for_toolchain({"rustc": "1.90.0"}) == result


def test_bk_measure_cache_usage_counts_recursive_bytes(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "blob.bin").write_bytes(b"12345")
    (tmp_path / "text.txt").write_bytes(b"ab")
    result = cache_keys.bk_measure_cache_usage(tmp_path)
    assert result["total_bytes"] == 7
    assert [item["name"] for item in result["entries"]] == ["nested", "text.txt"]
    assert result["count"] == 2
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_measure_cache_usage(tmp_path / "absent")
    assert error.value.code == "E006"


def test_bk_plan_prune_expires_then_evicts_oldest() -> None:
    entries = [
        {"name": "old", "age_seconds": 100, "size_bytes": 10, "modified": 1.0},
        {"name": "mid", "age_seconds": 10, "size_bytes": 20, "modified": 2.0},
        {"name": "new", "age_seconds": 1, "size_bytes": 30, "modified": 3.0},
    ]
    result = cache_keys.bk_plan_prune(
        entries, max_age_seconds=50, max_total_bytes=35, now=100.0
    )
    assert result["expired"] == ["old"]
    assert result["oversized"] == ["mid"]
    assert result["retained"] == ["new"]
    assert result["freed_bytes"] == 30
    assert result["retained_bytes"] == 30


def test_bk_prune_cache_entries_uses_exclusive_lock(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    old = cache / "old.txt"
    old.parent.mkdir()
    old.write_bytes(b"x" * 100)
    fresh = cache / "fresh.txt"
    fresh.write_bytes(b"keep")
    now = 10_000.0
    os.utime(old, (now - 500, now - 500))
    os.utime(fresh, (now - 1, now - 1))
    result = cache_keys.bk_prune_cache_entries(cache, max_age_seconds=60, now=now)
    assert result["removed"] == ["old.txt"]
    assert result["expired_count"] == 1
    assert result["freed_bytes"] == 100
    assert fresh.is_file()
    assert not (cache / cache_keys.PRUNE_LOCK_NAME).exists()
    lock = cache / cache_keys.PRUNE_LOCK_NAME
    lock.write_text("held")
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_prune_cache_entries(cache, now=now)
    assert error.value.code == "E004"
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_prune_cache_entries(cache, max_age_seconds=-1)
    assert error.value.code == "E003"


def test_bk_scoped_cache_dir_creates_component_once(tmp_path: Path) -> None:
    first = cache_keys.bk_scoped_cache_dir(
        tmp_path / "cache", scope_key="scope1", component="wheels"
    )
    second = cache_keys.bk_scoped_cache_dir(
        tmp_path / "cache", scope_key="scope1", component="wheels"
    )
    assert first["created"] is True and second["created"] is False
    assert Path(first["path"]).is_dir() and first["path"] == second["path"]
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_scoped_cache_dir(tmp_path, scope_key="../x", component="wheels")
    assert error.value.code == "E003"


def test_bk_scoped_cargo_target_dir_sets_env_snapshot(tmp_path: Path) -> None:
    result = cache_keys.bk_scoped_cargo_target_dir(
        tmp_path / "cache", scope_key="scope1", profile="debug", env={"PATH": "/bin"}
    )
    assert Path(result["path"]).is_dir()
    assert result["env"]["CARGO_TARGET_DIR"] == result["path"]
    assert result["env"]["PATH"] == "/bin"
    other = cache_keys.bk_scoped_cargo_target_dir(
        tmp_path / "cache", scope_key="scope1", profile="bench"
    )
    assert other["path"] != result["path"]


def test_bk_scoped_pip_cache_returns_scoped_env(tmp_path: Path) -> None:
    result = cache_keys.bk_scoped_pip_cache(tmp_path / "cache", scope_key="scope1")
    assert Path(result["path"]).is_dir()
    assert result["env"]["PIP_CACHE_DIR"] == result["path"]


def test_bk_scoped_pytest_cache_sets_cache_dir_option(tmp_path: Path) -> None:
    result = cache_keys.bk_scoped_pytest_cache(tmp_path / "cache", scope_key="scope1")
    assert Path(result["path"]).is_dir()
    assert result["env"]["PYTEST_ADDOPTS"] == f"-o cache_dir={result['path']}"


def test_bk_scoped_python_paths_stay_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = cache_keys.bk_scoped_python_paths(
        tmp_path / "cache", scope_key="scope1", workspace=workspace
    )
    for value in result["paths"].values():
        assert not Path(value).is_relative_to(workspace)
        assert Path(value).is_relative_to(tmp_path / "cache")
    assert result["env"]["VIRTUAL_ENV"] == result["paths"]["venv"]
    assert Path(result["paths"]["pycache"]).is_dir()
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_scoped_python_paths(
            tmp_path / "cache", scope_key="scope1", workspace=tmp_path / "cache" / "ws"
        )
    assert error.value.code == "E003"


def test_bk_scoped_ruff_cache_returns_scoped_env(tmp_path: Path) -> None:
    result = cache_keys.bk_scoped_ruff_cache(tmp_path / "cache", scope_key="scope1")
    assert Path(result["path"]).is_dir()
    assert result["env"]["RUFF_CACHE_DIR"] == result["path"]


def test_bk_validate_cache_key_checks_prefix_and_digest() -> None:
    result = cache_keys.bk_validate_cache_key(f"sources-{DIGEST}", prefix="sources")
    assert result["valid"] is True
    assert result["prefix"] == "sources" and result["digest"] == DIGEST
    assert result["length"] == len(f"sources-{DIGEST}")
    with pytest.raises(PolyglotError) as error:
        cache_keys.bk_validate_cache_key(f"env-{DIGEST}", prefix="sources")
    assert error.value.code == "E003"


def test_bv_aggregate_checks_tracks_failures_and_missing_requirements() -> None:
    result = verifiers.bv_aggregate_checks(
        [{"name": "a", "ok": True}, {"name": "b", "ok": False}], required=["c"]
    )
    assert result["passed"] is False
    assert result["total"] == 2 and result["passed_count"] == 1
    assert result["failed_names"] == ["b"]
    assert result["missing_required"] == ["c"]
    assert len(result["digest"]) == 64


def test_bv_check_env_allowlist_reports_disallowed_and_forbidden_keys() -> None:
    result = verifiers.bv_check_env_allowlist(
        {"GOOD_A": "1", "BAD_B": "2"},
        allowed_prefixes=("GOOD_",),
        forbidden_keys=("BAD_B",),
    )
    assert result["passed"] is False
    assert result["violations"] == ["forbidden: BAD_B", "not allowed: BAD_B"]
    assert result["checked"] == ["BAD_B", "GOOD_A"]


def test_bv_classify_check_result_normalizes_failure_detail() -> None:
    result = verifiers.bv_classify_check_result(
        {"name": "ruff", "returncode": 1, "stderr": "boom\n"}, name="lint"
    )
    assert result["name"] == "lint"
    assert result["ok"] is False and result["detail"] == "boom"
    assert len(result["fingerprint"]) == 64
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_classify_check_result(["bad"])  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_bv_classify_git_line_distinguishes_change_kinds() -> None:
    temporary = verifiers.bv_classify_git_line("?? tmp_run.log")
    unexpected = verifiers.bv_classify_git_line(" M src/main.py")
    allowed = verifiers.bv_classify_git_line(
        "?? allowed/file.py", allowed_paths=("allowed/file.py",)
    )
    ignored = verifiers.bv_classify_git_line("?? notes.md", ignore_patterns=("*.md",))
    assert temporary["category"] == "temporary"
    assert unexpected["category"] == "unexpected"
    assert allowed["category"] == "allowed"
    assert ignored["category"] == "ignored"


def test_bv_parse_command_output_handles_bytes_and_errors() -> None:
    result = verifiers.bv_parse_command_output(
        "unit", stdout=b"all good\n", stderr=b"warn\n", returncode=1
    )
    assert result["ok"] is False
    assert result["stdout_lines"] == ["all good"]
    assert result["stderr_lines"] == ["warn"]
    assert result["first_error"] == "warn"
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_parse_command_output("x", stdout="", stderr="", returncode="1")
    assert error.value.code == "E003"


def test_bv_plan_cargo_checks_covers_workspace_and_crates(tmp_path: Path) -> None:
    workspace = verifiers.bv_plan_cargo_checks(tmp_path)
    assert workspace["count"] == 2
    assert workspace["checks"][0]["args"] == [
        "cargo",
        "check",
        "--workspace",
        "--all-targets",
    ]
    crates = verifiers.bv_plan_cargo_checks(tmp_path, crates=["core"])
    assert crates["count"] == 2
    assert crates["checks"][1]["args"] == [
        "cargo",
        "clippy",
        "-p",
        "core",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_plan_cargo_checks(tmp_path, crates=[])
    assert error.value.code == "E003"


def test_bv_plan_doc_checks_uses_explicit_python(tmp_path: Path) -> None:
    plan = verifiers.bv_plan_doc_checks(tmp_path, python_executable="py")
    assert [item["name"] for item in plan["checks"]] == ["docs-sync", "docs-lint"]
    assert plan["checks"][0]["args"] == ["py", "tools/check_docs_sync.py"]
    assert plan["root"] == str(tmp_path)


def test_bv_plan_git_status_plans_one_command(tmp_path: Path) -> None:
    plan = verifiers.bv_plan_git_status(tmp_path)
    assert plan["count"] == 1
    assert plan["checks"][0]["args"] == ["git", "status", "--short"]


def test_bv_plan_pytest_suites_rejects_absolute_paths(tmp_path: Path) -> None:
    plan = verifiers.bv_plan_pytest_suites(
        tmp_path, suites=["tests/test_a.py", "tests/test_b.py"], python_executable="py"
    )
    assert plan["count"] == 2
    assert plan["checks"][0]["args"] == [
        "py",
        "-m",
        "pytest",
        "-q",
        "tests/test_a.py",
    ]
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_plan_pytest_suites(tmp_path, suites=["/tmp/test_a.py"])
    assert error.value.code == "E003"


def test_bv_plan_ruff_checks_orders_format_then_lint(tmp_path: Path) -> None:
    plan = verifiers.bv_plan_ruff_checks(tmp_path, ruff_executable="ruff-bin")
    assert [item["args"] for item in plan["checks"]] == [
        ["ruff-bin", "format", "--check", "."],
        ["ruff-bin", "check", "."],
    ]


def test_bv_render_check_summary_is_sorted_and_deterministic() -> None:
    results = [{"name": "b", "ok": False}, {"name": "a", "ok": True}]
    first = verifiers.bv_render_check_summary(results)
    second = verifiers.bv_render_check_summary(results)
    assert first == second
    assert first["lines"][0] == "CHECKS: 1 passed, 1 failed"
    assert first["lines"][1] == "  PASS a"
    assert first["lines"][2] == "  FAIL b"


def test_bv_require_runner_requires_an_explicit_callable() -> None:
    assert verifiers.bv_require_runner(FakeRunner())["accepted"] is True
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_require_runner(None)
    assert error.value.code == "E003"


def test_bv_verify_artifact_exists_reports_missing_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"data")
    result = verifiers.bv_verify_artifact_exists("artifact.bin", root=tmp_path)
    assert result["exists"] is True and result["size_bytes"] == 4
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_artifact_exists("missing.bin", root=tmp_path)
    assert error.value.code == "E006"


def test_bv_verify_cargo_injects_scoped_target_dir(tmp_path: Path) -> None:
    runner = FakeRunner()
    cache = tmp_path / "cache"
    result = verifiers.bv_verify_cargo(
        tmp_path,
        runner=runner,
        cache_root=cache,
        scope_key="scope1",
        profile="debug",
        crates=["core"],
    )
    expected = str(cache / "cargo" / "scope1" / "debug")
    assert result["passed"] is True
    assert result["cargo_target_dir"] == expected
    assert len(runner.calls) == 2
    assert all(call[2]["CARGO_TARGET_DIR"] == expected for call in runner.calls)
    assert all(call[1] == tmp_path for call in runner.calls)


def test_bv_verify_cargo_wraps_runner_failures_as_e004(tmp_path: Path) -> None:
    runner = FakeRunner(raises=RuntimeError("cargo missing"))
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_cargo(
            tmp_path,
            runner=runner,
            cache_root=tmp_path / "cache",
            scope_key="scope1",
            profile="debug",
        )
    assert error.value.code == "E004"


def test_bv_verify_cargo_env_detects_foreign_target_dir(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    correct = cache / "cargo" / "scope1" / "debug"
    good = verifiers.bv_verify_cargo_env(
        cache_root=cache,
        scope_key="scope1",
        profile="debug",
        env={"CARGO_TARGET_DIR": str(correct)},
    )
    assert good["isolated"] is True and good["expected"] == str(correct)
    bad = verifiers.bv_verify_cargo_env(
        cache_root=cache,
        scope_key="scope1",
        profile="debug",
        env={"CARGO_TARGET_DIR": "/outside/target"},
    )
    assert bad["isolated"] is False and bad["violations"]


def test_bv_verify_command_plan_rejects_wrong_cwd(tmp_path: Path) -> None:
    plan = verifiers.bv_plan_ruff_checks(tmp_path)
    assert verifiers.bv_verify_command_plan(plan, root=tmp_path)["valid"] is True
    mutated = dict(plan)
    mutated["checks"] = [{**item, "cwd": "/elsewhere"} for item in plan["checks"]]
    result = verifiers.bv_verify_command_plan(mutated, root=tmp_path)
    assert result["valid"] is False and len(result["violations"]) == 2


def test_bv_verify_docs_reports_failing_script(tmp_path: Path) -> None:
    runner = FakeRunner(
        by_command={("py", "tools/lint_docs.py"): (1, "", "lint failure\n")}
    )
    result = verifiers.bv_verify_docs(tmp_path, runner=runner, python_executable="py")
    assert result["passed"] is False
    assert result["failed_names"] == ["docs-lint"]
    assert result["checks"][1]["first_error"] == "lint failure"


def test_bv_verify_env_isolation_flags_paths_outside_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    inside = verifiers.bv_verify_env_isolation(
        {"CARGO_TARGET_DIR": str(cache / "cargo")}, cache_root=cache
    )
    assert inside["isolated"] is True and inside["checked"] == ["CARGO_TARGET_DIR"]
    outside = verifiers.bv_verify_env_isolation(
        {"CARGO_TARGET_DIR": "/outside/cargo"}, cache_root=cache
    )
    assert outside["isolated"] is False and outside["violations"]
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_env_isolation(["bad"], cache_root=cache)
    assert error.value.code == "E003"


def test_bv_verify_gate_matrix_collects_blockers_in_stable_order() -> None:
    result = verifiers.bv_verify_gate_matrix(
        {
            "gate-a": [{"name": "ruff", "ok": True}],
            "gate-b": [{"name": "pytest", "ok": False}],
        },
        required=["gate-c"],
    )
    assert result["passed"] is False
    assert result["blockers"] == ["gate-b: checks failed", "gate-c: missing gate"]
    assert result["gates"][0]["name"] == "gate-a"


def test_bv_verify_git_worktree_classifies_status_lines(tmp_path: Path) -> None:
    runner = FakeRunner(stdout="?? tmp_run.log\n M src/main.py\n?? allowed/file.py\n")
    result = verifiers.bv_verify_git_worktree(
        tmp_path, runner=runner, allowed_paths=("allowed/file.py",)
    )
    assert result["passed"] is False
    assert [item["category"] for item in result["findings"]] == [
        "temporary",
        "unexpected",
    ]
    assert result["allowed"] == ["allowed/file.py"]


def test_bv_verify_platform_support_rejects_unsupported_targets() -> None:
    result = verifiers.bv_verify_platform_support(
        "Linux",
        "X86_64",
        supported_platforms=("linux",),
        supported_architectures=("x86_64",),
    )
    assert result["supported"] is True
    assert result["platform"] == "linux" and result["arch"] == "x86_64"
    relaxed = verifiers.bv_verify_platform_support(
        "windows",
        "arm64",
        supported_platforms=("linux",),
        supported_architectures=("x86_64",),
        strict=False,
    )
    assert relaxed["supported"] is False and len(relaxed["reasons"]) == 2
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_platform_support(
            "windows",
            "x86_64",
            supported_platforms=("linux",),
            supported_architectures=("x86_64",),
        )
    assert error.value.code == "E007"


def test_bv_verify_pytest_uses_injected_runner_for_every_suite(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    result = verifiers.bv_verify_pytest(
        tmp_path,
        runner=runner,
        suites=["tests/test_a.py", "tests/test_b.py"],
        python_executable="py",
        env={"PYTEST_ADDOPTS": "-q"},
    )
    assert result["passed"] is True and result["suite_count"] == 2
    assert len(runner.calls) == 2
    assert all(call[2]["PYTEST_ADDOPTS"] == "-q" for call in runner.calls)
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_pytest(tmp_path, runner=runner, suites=["/abs/test.py"])
    assert error.value.code == "E003"


def test_bv_verify_ruff_reports_failed_lint_and_runner_errors(tmp_path: Path) -> None:
    runner = FakeRunner(by_command={("ruff", "check", "."): (1, "", "issue found\n")})
    result = verifiers.bv_verify_ruff(tmp_path, runner=runner)
    assert result["failed_names"] == ["ruff-lint"]
    assert result["checks"][1]["first_error"] == "issue found"
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_ruff(tmp_path, runner=FakeRunner(raises=ValueError("boom")))
    assert error.value.code == "E004"


def test_bv_verify_snapshot_checks_required_keys() -> None:
    snapshot = {"case": "example", "value": 1}
    assert verifiers.bv_verify_snapshot(snapshot, required_keys=["case", "value"])[
        "valid"
    ]
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_snapshot(snapshot, required_keys=["missing"])
    assert error.value.code == "E006"
    with pytest.raises(PolyglotError) as error:
        verifiers.bv_verify_snapshot(["bad"])  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_bv_verify_tool_availability_marks_missing_tools() -> None:
    result = verifiers.bv_verify_tool_availability(
        {"ruff": "/usr/bin/ruff", "cargo": None}, required=["ruff", "cargo"]
    )
    assert result["passed"] is False
    assert result["available"] == ["ruff"]
    assert result["missing"] == ["cargo"]


def test_br_attach_metadata_preserves_original_mapping() -> None:
    report = {"gates": {"a": True}}
    result = reports.br_attach_metadata(report, metadata={"run": "1"})
    assert result["report"]["metadata"] == {"run": "1"}
    assert "metadata" not in report
    assert len(result["digest"]) == 64


def test_br_build_cache_manifest_sorts_entries() -> None:
    result = reports.br_build_cache_manifest(
        [
            {"name": "b", "digest": DIGEST, "size_bytes": 2},
            {"name": "a", "digest": DIGEST, "size_bytes": 1},
        ],
        root=Path("/repo"),
    )
    assert [item["name"] for item in result["manifest"]["entries"]] == ["a", "b"]
    assert result["count"] == 2
    assert result["manifest"]["schema"] == 1


def test_br_build_check_record_creates_stable_fingerprint() -> None:
    result = reports.br_build_check_record(
        "ruff", ok=False, detail="lint failed", command=("ruff", "check")
    )
    assert result["ok"] is False and result["detail"] == "lint failed"
    assert result["command"] == ["ruff", "check"]
    assert len(result["fingerprint"]) == 64
    with pytest.raises(PolyglotError) as error:
        reports.br_build_check_record("ruff", ok=1)  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_br_build_gate_record_combines_failures_and_blockers() -> None:
    result = reports.br_build_gate_record(
        "lint",
        checks=[
            {"name": "format", "ok": True},
            {"name": "lint", "ok": False},
        ],
        blockers=["manual"],
    )
    assert result["passed"] is False
    assert result["check_count"] == 2
    assert result["blockers"] == ["lint", "manual"]


def test_br_build_parity_report_finds_mismatches_and_missing_backends() -> None:
    result = reports.br_build_parity_report(
        [
            {
                "case": "case-a",
                "backends": ["python", "rust"],
                "agreement": False,
            },
            {"case": "case-b", "backends": ["python", "rust"], "agreement": True},
        ],
        required_backends=["zig"],
    )
    assert result["passed"] is False
    assert result["mismatches"] == ["case-a"]
    assert result["missing"] == ["case-a:zig", "case-b:zig"]


def test_br_build_release_report_aggregates_gate_blockers() -> None:
    result = reports.br_build_release_report(
        {"gate-a": [{"name": "check", "ok": True}]}, blockers=["manual"]
    )
    assert result["passed"] is False
    assert result["gate_count"] == 1
    assert result["report"]["blockers"] == ["manual"]
    assert result["report"]["gates"]["gate-a"]["passed"] is True


def test_br_canonicalize_report_uses_sorted_compact_json() -> None:
    result = reports.br_canonicalize_report({"b": 2, "a": 1})
    assert result["text"] == '{"a":1,"b":2}'
    assert len(result["digest"]) == 64
    assert result["bytes"] == len(result["text"].encode("utf-8"))


def test_br_check_report_schema_reports_missing_keys_and_type_errors() -> None:
    result = reports.br_check_report_schema(
        {"checks": {"bad": "shape"}}, required_keys=["schema"]
    )
    assert result["valid"] is False
    assert result["missing"] == ["schema"]
    assert result["type_errors"] == ["checks must be a sequence"]


def test_br_collect_blockers_scans_gates_and_checks() -> None:
    result = reports.br_collect_blockers(
        {
            "blockers": ["explicit"],
            "gates": {"gate-a": {"passed": False}},
            "checks": [{"name": "lint", "ok": False}],
        }
    )
    assert result["blockers"] == [
        "explicit",
        "gate-a: gate failed",
        "lint: check failed",
    ]
    assert result["count"] == 3


def test_br_diff_reports_classifies_added_removed_and_changed_leaves() -> None:
    result = reports.br_diff_reports(
        {"a": 1, "b": 2, "old": 9}, {"a": 1, "b": 3, "new": 4}
    )
    assert result["added"] == ["new"]
    assert result["removed"] == ["old"]
    assert result["changed"] == [{"path": "b", "left": "2", "right": "3"}]
    assert result["equal"] is False


def test_br_flatten_report_uses_dotted_paths() -> None:
    result = reports.br_flatten_report({"gates": {"a": {"passed": True}}})
    assert result["entries"] == [
        {"path": "gates.a.passed", "value": "true", "digest": reports._digest("true")}
    ]
    assert result["count"] == 1


def test_br_merge_reports_records_conflicting_paths() -> None:
    result = reports.br_merge_reports([{"a": 1}, {"a": 2, "b": 3}])
    assert result["report"] == {"a": 2, "b": 3}
    assert result["conflicts"] == ["a"]
    assert result["sources"] == [0, 1]
    with pytest.raises(PolyglotError) as error:
        reports.br_merge_reports([])
    assert error.value.code == "E003"


def test_br_normalize_check_names_sorts_and_deduplicates() -> None:
    result = reports.br_normalize_check_names(["b", "a", "b"])
    assert result["names"] == ["a", "b"]
    assert result["duplicates"] == ["b"]
    assert result["count"] == 2


def test_br_parse_cache_manifest_validates_schema(tmp_path: Path) -> None:
    manifest = reports.br_build_cache_manifest(
        [{"name": "a", "digest": DIGEST, "size_bytes": 1}], root=tmp_path
    )["manifest"]
    parsed = reports.br_parse_cache_manifest(json.dumps(manifest))
    assert parsed["entry_count"] == 1 and parsed["digest"] == reports._digest(manifest)
    with pytest.raises(PolyglotError) as error:
        reports.br_parse_cache_manifest("{")
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        reports.br_parse_cache_manifest('{"schema": 2, "entries": []}')
    assert error.value.code == "E008"


def test_br_parse_report_json_rejects_non_objects() -> None:
    parsed = reports.br_parse_report_json('{"b": 1, "a": 2}')
    assert parsed["report"] == {"a": 2, "b": 1}
    assert len(parsed["digest"]) == 64
    with pytest.raises(PolyglotError) as error:
        reports.br_parse_report_json("[1, 2]")
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        reports.br_parse_report_json("not json")
    assert error.value.code == "E008"


def test_br_read_report_round_trips_a_written_report(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "gate.json"
    written = reports.br_write_report(
        path, report={"checks": [{"name": "a", "ok": True}]}
    )
    read = reports.br_read_report(path)
    assert read["report"] == {"checks": [{"name": "a", "ok": True}]}
    assert read["digest"] == written["digest"]
    assert (
        path.read_text(encoding="utf-8")
        == reports.br_canonicalize_report(read["report"])["text"]
    )
    with pytest.raises(PolyglotError) as error:
        reports.br_read_report(tmp_path / "absent.json")
    assert error.value.code == "E006"


def test_br_render_report_markdown_escapes_table_cells() -> None:
    result = reports.br_render_report_markdown(
        {"checks": [{"name": "a|b", "ok": True, "detail": "one|two"}]}
    )
    assert result["status"] == "PASS"
    assert "| a\\|b | PASS | one\\|two |" in result["lines"]


def test_br_render_report_text_lists_failing_checks() -> None:
    result = reports.br_render_report_text(
        {"checks": [{"name": "a", "ok": False, "detail": "boom"}]}
    )
    assert result["lines"][0] == "REPORT: FAIL"
    assert "  FAIL a - boom" in result["lines"]
    assert result["status"] == "FAIL"


def test_br_report_exit_status_respects_strict_mode() -> None:
    report = {"blockers": ["manual"], "checks": []}
    strict = reports.br_report_exit_status(report)
    relaxed = reports.br_report_exit_status(report, strict=False)
    assert strict["exit_code"] == 1 and strict["blocker_count"] == 1
    assert relaxed["exit_code"] == 0 and relaxed["passed"] is True
    with pytest.raises(PolyglotError) as error:
        reports.br_report_exit_status({"checks": {"bad": "shape"}})
    assert error.value.code == "E008"


def test_br_report_fingerprint_matches_canonical_digest() -> None:
    report = {"b": 2, "a": 1}
    result = reports.br_report_fingerprint(report)
    assert result["algorithm"] == "sha256"
    assert result["digest"] == reports.br_canonicalize_report(report)["digest"]


def test_br_select_report_fields_reports_missing_names() -> None:
    result = reports.br_select_report_fields({"a": 1, "b": 2}, fields=["a", "missing"])
    assert result["selected"] == {"a": 1}
    assert result["missing"] == ["missing"]
    assert result["count"] == 1


def test_br_stable_sort_records_uses_canonical_tie_break() -> None:
    result = reports.br_stable_sort_records(
        [{"key": "b", "id": 1}, {"key": "a", "id": 2}, {"key": "b", "id": 0}],
        key="key",
    )
    assert [item["id"] for item in result["records"]] == [2, 0, 1]
    with pytest.raises(PolyglotError) as error:
        reports.br_stable_sort_records([{"other": 1}], key="key")
    assert error.value.code == "E003"


def test_br_summarize_results_computes_pass_rate() -> None:
    result = reports.br_summarize_results(
        [{"name": "b", "ok": False}, {"name": "a", "ok": True}]
    )
    assert result["passed"] == 1 and result["failed"] == 1
    assert result["total"] == 2 and result["rate"] == 0.5
    assert result["names"] == ["a", "b"]


def test_br_validate_gate_report_rejects_malformed_serialized_data() -> None:
    valid = reports.br_validate_gate_report(
        {"checks": [{"name": "a", "ok": True}], "blockers": []}
    )
    assert valid["valid"] is True and valid["check_count"] == 1
    with pytest.raises(PolyglotError) as error:
        reports.br_validate_gate_report({"checks": [{"name": "a", "ok": "yes"}]})
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        reports.br_validate_gate_report({"blockers": [7]})
    assert error.value.code == "E008"


def test_br_write_report_replaces_existing_file_deterministically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate.json"
    first = reports.br_write_report(path, report={"a": 1})
    second = reports.br_write_report(path, report={"a": 1})
    assert first["replaced"] is False and second["replaced"] is True
    assert first["digest"] == second["digest"]
    assert path.read_bytes() == b'{"a":1}'


def _observations() -> list[dict[str, Any]]:
    return [
        {"backend": "python", "case": "add", "value": 2},
        {"backend": "rust", "case": "add", "value": 2},
        {"backend": "python", "case": "mul", "value": 6},
        {"backend": "rust", "case": "mul", "value": 7},
    ]


def test_bp_build_parity_matrix_marks_disagreeing_cases() -> None:
    matrix = parity.bp_build_parity_matrix(_observations())
    assert matrix["backends"] == ["python", "rust"]
    assert matrix["cases"] == ["add", "mul"]
    rows = {row["case"]: row for row in matrix["rows"]}
    assert rows["add"]["agreement"] is True
    assert rows["mul"]["agreement"] is False
    assert [item["backend"] for item in rows["add"]["backends"]] == ["python", "rust"]


def test_bp_build_snapshot_is_versioned_and_sorted() -> None:
    result = parity.bp_build_snapshot(
        [
            {"backend": "rust", "case": "b", "value": 2},
            {"backend": "python", "case": "a", "value": 1},
        ],
        platform_name="Linux",
        toolchain={"rustc": "1.90", "python": "3.12"},
    )
    snapshot = result["snapshot"]
    assert snapshot["schema"] == 1 and snapshot["platform"] == "linux"
    assert list(snapshot["toolchain"]) == ["python", "rustc"]
    assert [item["case"] for item in snapshot["observations"]] == ["a", "b"]
    assert result["observation_count"] == 2 and len(result["digest"]) == 64


def test_bp_canonicalize_observation_normalizes_optional_fields() -> None:
    result = parity.bp_canonicalize_observation(
        {
            "backend": "python",
            "case": "add",
            "value": {"b": 2, "a": 1},
            "platform": "Linux",
        }
    )
    record = result["record"]
    assert record["status"] == "ok" and record["platform"] == "Linux"
    assert record["output"] == ""
    assert record["value"] == {"a": 1, "b": 2}
    assert len(result["digest"]) == 64
    with pytest.raises(PolyglotError) as error:
        parity.bp_canonicalize_observation({"case": "add", "value": 1})
    assert error.value.code == "E003"


def test_bp_compare_diagnostics_ignores_list_order() -> None:
    left = [
        {"code": "E001", "line": 1, "col": 2, "message": "bad"},
        {"code": "E002", "line": 2, "col": 1, "message": "missing"},
    ]
    right = list(reversed(left))
    assert parity.bp_compare_diagnostics(left, right)["equal"] is True
    changed = [{"code": "E003", "line": 1, "col": 2, "message": "bad"}]
    result = parity.bp_compare_diagnostics(left, changed)
    assert result["equal"] is False and len(result["differences"]) == 3


def test_bp_compare_numeric_outputs_uses_explicit_tolerances() -> None:
    result = parity.bp_compare_numeric_outputs(1.0, "1.0000000001")
    assert result["equal"] is True and result["delta"] > 0
    assert parity.bp_compare_numeric_outputs(1.0, 2.0)["equal"] is False
    with pytest.raises(PolyglotError) as error:
        parity.bp_compare_numeric_outputs("not a number", 1.0)
    assert error.value.code == "E003"


def test_bp_compare_observations_lists_field_differences() -> None:
    result = parity.bp_compare_observations(
        {"backend": "python", "case": "add", "value": 1},
        {"backend": "python", "case": "add", "value": 2},
    )
    assert result["equal"] is False
    assert [item["field"] for item in result["differences"]] == ["value"]
    assert result["left_digest"] != result["right_digest"]


def test_bp_compare_text_outputs_ignores_ansi_space_and_case() -> None:
    result = parity.bp_compare_text_outputs(
        "\x1b[31mHello \r\nWorld\r\n", "hello\nWORLD\n", case_sensitive=False
    )
    assert result["equal"] is True
    assert result["left_lines"] == ["hello", "world"]


def test_bp_compare_values_is_type_strict_and_order_independent() -> None:
    assert parity.bp_compare_values(1, True)["equal"] is False
    assert parity.bp_compare_values({"b": 2, "a": 1}, {"a": 1, "b": 2})["equal"] is True
    assert parity.bp_compare_values("1", 1)["equal"] is False


def test_bp_extract_diagnostics_parses_codes_and_locations() -> None:
    result = parity.bp_extract_diagnostics(
        "E001: bad token at 3:7\nE002: missing node\n"
    )
    assert result["count"] == 2
    assert result["diagnostics"][0] == {
        "code": "E001",
        "message": "bad token",
        "line": 3,
        "col": 7,
    }
    assert result["diagnostics"][1]["line"] == 0
    with pytest.raises(PolyglotError) as error:
        parity.bp_extract_diagnostics("x", pattern="[")
    assert error.value.code == "E003"


def test_bp_find_mismatches_extracts_disagreeing_cases() -> None:
    matrix = parity.bp_build_parity_matrix(_observations())
    result = parity.bp_find_mismatches(matrix)
    assert result["count"] == 1
    assert result["mismatches"][0]["case"] == "mul"
    assert result["mismatches"][0]["backends"] == ["python", "rust"]
    assert len(result["mismatches"][0]["digests"]) == 2


def test_bp_group_by_case_sorts_cases_and_backends() -> None:
    result = parity.bp_group_by_case(_observations())
    assert list(result["groups"]) == ["add", "mul"]
    assert [item["backend"] for item in result["groups"]["add"]] == ["python", "rust"]
    assert result["case_count"] == 2 and result["observation_count"] == 4


def test_bp_index_by_backend_rejects_duplicate_observations() -> None:
    result = parity.bp_index_by_backend(_observations())
    assert result["backends"] == ["python", "rust"]
    assert result["index"]["python"]["add"]["value"] == 2
    with pytest.raises(PolyglotError) as error:
        parity.bp_index_by_backend(
            _observations() + [{"backend": "python", "case": "add", "value": 3}]
        )
    assert error.value.code == "E003"


def test_bp_load_snapshot_rejects_malformed_payloads() -> None:
    snapshot = parity.bp_build_snapshot(
        _observations(), platform_name="linux", toolchain={"rustc": "1.90"}
    )["snapshot"]
    loaded = parity.bp_load_snapshot(json.dumps(snapshot))
    assert loaded["observation_count"] == 4
    assert loaded["digest"] == parity.bp_validate_snapshot(snapshot)["digest"]
    with pytest.raises(PolyglotError) as error:
        parity.bp_load_snapshot("{")
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        parity.bp_load_snapshot("[1, 2]")
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        parity.bp_load_snapshot('{"schema": 99}')
    assert error.value.code == "E008"


def test_bp_measure_agreement_counts_cases_and_backend_coverage() -> None:
    matrix = parity.bp_build_parity_matrix(_observations())
    result = parity.bp_measure_agreement(matrix)
    assert result["total_cases"] == 2
    assert result["matching_cases"] == 1 and result["mismatching_cases"] == 1
    assert result["rate"] == 0.5
    assert result["by_backend"] == {"python": 2, "rust": 2}


def test_bp_merge_snapshots_rejects_conflicting_observations() -> None:
    first = parity.bp_build_snapshot(
        [{"backend": "python", "case": "add", "value": 1}],
        platform_name="linux",
        toolchain={},
    )["snapshot"]
    second = parity.bp_build_snapshot(
        [{"backend": "rust", "case": "mul", "value": 2}],
        platform_name="linux",
        toolchain={},
    )["snapshot"]
    merged = parity.bp_merge_snapshots([first, second])
    assert merged["observation_count"] == 2 and merged["sources"] == [0, 1]
    conflicting = parity.bp_build_snapshot(
        [{"backend": "python", "case": "add", "value": 99}],
        platform_name="linux",
        toolchain={},
    )["snapshot"]
    with pytest.raises(PolyglotError) as error:
        parity.bp_merge_snapshots([first, conflicting])
    assert error.value.code == "E008"


def test_bp_normalize_output_strips_ansi_and_newlines() -> None:
    result = parity.bp_normalize_output("\x1b[31mred\r\nx\ry")
    assert result["lines"] == ["red", "x", "y"]
    assert result["line_count"] == 3 and len(result["digest"]) == 64
    with pytest.raises(PolyglotError) as error:
        parity.bp_normalize_output("x", strip_ansi="yes")  # type: ignore[arg-type]
    assert error.value.code == "E003"


def test_bp_normalize_value_returns_stable_kinds() -> None:
    values: list[Any] = [None, True, 2, 1.5, "x", [1], {"a": 1}]
    assert [parity.bp_normalize_value(item)["kind"] for item in values] == [
        "null",
        "boolean",
        "number",
        "number",
        "string",
        "array",
        "object",
    ]
    assert parity.bp_normalize_value({"b": 2, "a": 1})["text"] == '{"a":1,"b":2}'
    with pytest.raises(PolyglotError) as error:
        parity.bp_normalize_value(object())
    assert error.value.code == "E003"


def test_bp_parity_fingerprint_is_canonical() -> None:
    matrix = parity.bp_build_parity_matrix(_observations())
    result = parity.bp_parity_fingerprint(matrix)
    assert result["algorithm"] == "sha256" and len(result["digest"]) == 64
    assert result["bytes"] > 0


def test_bp_plan_parity_cases_builds_a_backend_grid() -> None:
    result = parity.bp_plan_parity_cases(["mul", "add"], backends=["python", "rust"])
    assert [row["case"] for row in result["cases"]] == ["mul", "add"]
    assert result["cases"][0]["backends"] == ["python", "rust"]
    assert result["count"] == 2 and result["backend_count"] == 2
    with pytest.raises(PolyglotError) as error:
        parity.bp_plan_parity_cases(["../bad"], backends=["python"])
    assert error.value.code == "E003"


def test_bp_report_parity_blocks_on_disagreement_and_missing_backends() -> None:
    matrix = parity.bp_build_parity_matrix(_observations())
    result = parity.bp_report_parity(matrix, required_backends=["zig"])
    assert result["passed"] is False
    assert result["blockers"] == [
        "add: missing zig",
        "mul: backend disagreement",
        "mul: missing zig",
    ]
    assert result["report"]["coverage"]["total_cases"] == 2


def test_bp_require_backend_enforces_supported_list() -> None:
    result = parity.bp_require_backend("Python", supported=["python", "rust"])
    assert result["supported"] is True and result["backend"] == "python"
    relaxed = parity.bp_require_backend("zig", supported=["python"], strict=False)
    assert relaxed["supported"] is False and relaxed["strict"] is False
    with pytest.raises(PolyglotError) as error:
        parity.bp_require_backend("zig", supported=["python"])
    assert error.value.code == "E007"


def test_bp_require_platform_enforces_supported_list() -> None:
    result = parity.bp_require_platform("Linux", supported=["linux"])
    assert result["supported"] is True and result["platform"] == "linux"
    with pytest.raises(PolyglotError) as error:
        parity.bp_require_platform("windows", supported=["linux"])
    assert error.value.code == "E007"


def test_bp_select_backends_filters_without_mutating_inputs() -> None:
    observations = _observations()
    result = parity.bp_select_backends(observations, backends=["python", "zig"])
    assert result["count"] == 2
    assert result["selected"] == ["python"]
    assert result["missing"] == ["zig"]
    assert len(observations) == 4


def test_bp_snapshot_diff_finds_changed_observations() -> None:
    left = parity.bp_build_snapshot(
        [{"backend": "python", "case": "add", "value": 1}],
        platform_name="linux",
        toolchain={},
    )["snapshot"]
    right = parity.bp_build_snapshot(
        [{"backend": "python", "case": "add", "value": 2}],
        platform_name="linux",
        toolchain={},
    )["snapshot"]
    result = parity.bp_snapshot_diff(left, right)
    assert result["changed"] == ["python::add"]
    assert result["equal"] is False
    assert parity.bp_snapshot_diff(left, left)["equal"] is True


def test_bp_validate_snapshot_rejects_malformed_structure() -> None:
    snapshot = parity.bp_build_snapshot(
        _observations(), platform_name="linux", toolchain={}
    )["snapshot"]
    assert parity.bp_validate_snapshot(snapshot)["observation_count"] == 4
    with pytest.raises(PolyglotError) as error:
        parity.bp_validate_snapshot(
            {
                "schema": 1,
                "platform": "linux",
                "toolchain": {},
                "observations": [{"backend": "python"}],
            }
        )
    assert error.value.code == "E008"
    with pytest.raises(PolyglotError) as error:
        parity.bp_validate_snapshot([1, 2])  # type: ignore[arg-type]
    assert error.value.code == "E008"
