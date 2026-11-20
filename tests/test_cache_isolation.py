"""Cache-scope isolation invariants for builds and verification runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tools.build_ci.framework import (
    ci_compute_cache_scope_key,
    ci_isolate_cargo_target_dir,
    ci_isolate_python_venv_paths,
)


def test_cache_scope_key_is_stable_and_dimensioned() -> None:
    kwargs = {
        "os_name": "Darwin",
        "arch": "ARM64",
        "compiler_name": "RustC",
        "compiler_version": "1.97.1",
        "feature_flags": {"wasm": True, "ffi": False},
    }
    first = ci_compute_cache_scope_key(**kwargs)
    assert first == ci_compute_cache_scope_key(**kwargs)
    assert first == ci_compute_cache_scope_key(
        os_name="darwin",
        arch="arm64",
        compiler_name="rustc",
        compiler_version="1.97.1",
        feature_flags={"ffi": False, "wasm": "1"},
    )
    assert (
        ci_compute_cache_scope_key(**{**kwargs, "compiler_version": "1.96.0"}) != first
    )
    assert ci_compute_cache_scope_key(**{**kwargs, "extra": {"seed": "7"}}) != first
    assert "/" not in first and "\\" not in first


def test_cargo_target_dirs_are_distinct_and_outside_workspace(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scope_a = ci_compute_cache_scope_key(extra={"seed": "1"})
    scope_b = ci_compute_cache_scope_key(extra={"seed": "2"})
    env: dict[str, str] = {}
    debug_a = ci_isolate_cargo_target_dir(
        "debug", cache_root=cache_root, scope_key=scope_a, env=env
    )
    assert env["CARGO_TARGET_DIR"] == str(debug_a)
    bench_a = ci_isolate_cargo_target_dir(
        "bench", cache_root=cache_root, scope_key=scope_a
    )
    debug_b = ci_isolate_cargo_target_dir(
        "debug", cache_root=cache_root, scope_key=scope_b
    )
    assert debug_a != bench_a != debug_b
    assert len({debug_a, bench_a, debug_b}) == 3
    for path in (debug_a, bench_a, debug_b):
        assert path.is_dir()
        assert not path.is_relative_to(workspace)


def test_python_paths_are_isolated_and_env_is_scoped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_root = tmp_path / "cache"
    scope = ci_compute_cache_scope_key(extra={"seed": "py"})
    env: dict[str, str] = {}
    paths = ci_isolate_python_venv_paths(
        workspace=workspace, cache_root=cache_root, scope_key=scope, env=env
    )
    for path in paths.values():
        assert not path.is_relative_to(workspace)
    assert paths["pip_cache"].is_dir()
    assert paths["pytest_cache"].is_dir()
    assert env["PIP_CACHE_DIR"] == str(paths["pip_cache"])
    assert env["PYTHONPYCACHEPREFIX"] == str(paths["pycache"])
    assert env["VIRTUAL_ENV"] == str(paths["venv"])

    other = ci_isolate_python_venv_paths(
        workspace=workspace,
        cache_root=cache_root,
        scope_key=ci_compute_cache_scope_key(extra={"seed": "other"}),
    )
    assert other["venv"] != paths["venv"]

    try:
        ci_isolate_python_venv_paths(
            workspace=workspace, cache_root=workspace / "nested", scope_key=scope
        )
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping cache root must be rejected")


def test_verification_does_not_change_git_status(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    before = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    env = dict(os.environ)
    ci_isolate_cargo_target_dir(
        "debug",
        cache_root=tmp_path,
        scope_key=ci_compute_cache_scope_key(extra={"probe": "worktree"}),
        env=env,
    )
    after = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert after == before
