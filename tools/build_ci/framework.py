"""FRAME 9: build cache, isolation, and CI release gates.

The functions in this module implement framework functions 81-90.  They are
deterministic and side-effect-aware: cache keys never depend on Python's
per-process ``hash()``, cache pruning uses an exclusive lock file, and every
command-running verifier accepts an injectable :data:`CommandRunner` plus an
explicit repository root so unit tests can replace real pytest/cargo/ruff/git
invocations with fakes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

CommandRunner = Callable[[tuple[str, ...], Path], subprocess.CompletedProcess[str]]
"""Callable invoked as ``runner(args, cwd)``; defaults to :func:`subprocess.run`."""

FeatureValue = bool | str | int | None

DEFAULT_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
DEFAULT_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024

DEFAULT_PYTEST_SUITES: tuple[str, ...] = (
    "tests/test_point.py",
    "tests/test_generics.py",
    "tests/test_visualizer.py",
    "tests/test_cross_backend_parity.py",
)
"""Core unit, generic, visualizer, and native-parity suite files."""

DOCS_SYNC_SCRIPT = "tools/check_docs_sync.py"
DOCS_LINT_SCRIPT = "tools/lint_docs.py"

TRACKED_DOC_NAMES: tuple[str, ...] = (
    "README.md",
    "GUIDE.md",
    "SPECIFICATION.md",
    "NOTES.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "COMMIT_GUIDE.md",
)
"""Top-level Markdown files considered by :func:`ci_verify_doc_links`."""

DEFAULT_BENCHMARK_THRESHOLD = 2.0
BENCHMARK_SOURCE = "1 + 2 * 3 - 4 / 5"
BENCHMARK_REPETITIONS = 32
BENCHMARK_RUNS = 5
BENCHMARK_BASELINE_FILENAME = "benchmark-baseline.json"

_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^()\n]+)\)")
_MARKDOWN_FENCE_RE = re.compile(r"^\s*(```|~~~)")

_PRUNE_LOCK_NAME = ".prune.lock"
_STALE_LOCK_SECONDS = 300.0
_LOCK_POLL_SECONDS = 0.01
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_PREFIX_RE = re.compile(r"[^a-z0-9._-]+")
_TEMP_DIR_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "target", "build"}
)
_TEMP_SUFFIXES = (".tmp", ".log", ".bak", ".orig", ".rej", ".swp")


class CachePruneResult(TypedDict):
    """Summary returned by :func:`ci_prune_stale_cache_entries`."""

    removed: list[str]
    removed_count: int
    expired_count: int
    oversized_count: int
    freed_bytes: int
    retained_bytes: int


@dataclass(frozen=True)
class CheckResult:
    """Result of one CI command; ``ok`` is true only for a zero exit code."""

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """Whether the command exited successfully."""
        return self.returncode == 0


@dataclass(frozen=True)
class GateCheckStatus:
    """One line item in a release-gate report."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ReleaseGateReport:
    """Aggregate status of all release gates."""

    checks: tuple[GateCheckStatus, ...]
    blockers: tuple[str, ...]

    @property
    def passed_checks(self) -> int:
        """Number of successful gate line items."""
        return sum(1 for check in self.checks if check.ok)

    @property
    def failed_checks(self) -> int:
        """Number of failed gate line items."""
        return sum(1 for check in self.checks if not check.ok)

    @property
    def passed(self) -> bool:
        """True only when every check passed and no blockers were recorded."""
        return self.failed_checks == 0 and not self.blockers

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the report."""
        return {
            "passed": self.passed,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "blockers": list(self.blockers),
            "checks": [
                {"name": check.name, "ok": check.ok, "detail": check.detail}
                for check in self.checks
            ],
        }

    def render(self) -> str:
        """Return a deterministic human-readable summary."""
        lines = [
            f"RELEASE GATE: {'PASS' if self.passed else 'FAIL'}",
            f"checks: {self.passed_checks} passed, {self.failed_checks} failed",
        ]
        if self.blockers:
            lines.append("blockers:")
            lines.extend(f"  - {blocker}" for blocker in self.blockers)
        return "\n".join(lines)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _default_runner(
    args: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_check(
    name: str,
    args: tuple[str, ...],
    root: Path,
    runner: CommandRunner | None,
) -> CheckResult:
    process = (runner if runner is not None else _default_runner)(args, root)
    return CheckResult(
        name=name,
        command=args,
        returncode=int(process.returncode),
        stdout=_as_text(process.stdout),
        stderr=_as_text(process.stderr),
    )


def _in_process_check(
    name: str, *, ok: bool, detail: str, error: str = ""
) -> CheckResult:
    """Synthesize a gate result for a verifier that never spawns a command."""
    return CheckResult(
        name=name,
        command=(f"in-process:{name}",),
        returncode=0 if ok else 1,
        stdout=detail if ok else "",
        stderr="" if ok else (error or detail),
    )


def _resolve_root(root: Path | str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"root is not a directory: {path}")
    return path


def _safe_prefix_part(value: str) -> str:
    cleaned = _SAFE_PREFIX_RE.sub("-", value.strip().lower()).strip("-")
    return cleaned or "unknown"


def _validated_component(value: str, label: str) -> str:
    if not value or _COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe path component: {value!r}")
    return value


def _relative_path(value: str, label: str) -> str:
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if (
        not cleaned
        or cleaned.startswith("/")
        or ":" in cleaned
        or ".." in cleaned.split("/")
    ):
        raise ValueError(f"{label} must be a relative repository path: {value!r}")
    return cleaned


def _entry_stat(path: Path) -> os.stat_result:
    return path.lstat()


def _entry_size(path: Path) -> int:
    if path.is_symlink():
        return _entry_stat(path).st_size
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [
            name for name in dirnames if not (Path(dirpath) / name).is_symlink()
        ]
        for name in filenames:
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _unlink_if_exists(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _remove_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
    else:
        _unlink_if_exists(path)


def _acquire_prune_lock(lock_path: Path, timeout_seconds: float) -> int:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except FileExistsError:
            try:
                lock_age = time.time() - _entry_stat(lock_path).st_mtime
            except FileNotFoundError:
                continue
            if lock_age > _STALE_LOCK_SECONDS:
                _unlink_if_exists(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"cache prune lock is busy: {lock_path}") from None
            time.sleep(_LOCK_POLL_SECONDS)


def _is_temp_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = parts[-1]
    return (
        any(part in _TEMP_DIR_NAMES for part in parts)
        or name.startswith(("tmp", ".tmp"))
        or name.endswith(_TEMP_SUFFIXES)
    )


def ci_compute_cache_scope_key(
    *,
    os_name: str | None = None,
    arch: str | None = None,
    compiler_name: str = "rustc",
    compiler_version: str = "unknown",
    feature_flags: Mapping[str, FeatureValue] | Sequence[str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    """Return a stable, unique cache key for one build scope.

    The key combines the operating system, CPU architecture, compiler name and
    version, feature flags, and optional extra dimensions.  Values are
    normalized and canonicalized (sorted keys) before SHA-256 hashing, so the
    result is identical across processes and independent of flag insertion
    order.  Missing ``os_name``/``arch`` values fall back to ``platform``.
    """
    resolved_os = (os_name or platform.system() or "unknown").strip().lower()
    resolved_arch = (arch or platform.machine() or "unknown").strip().lower()
    resolved_compiler = compiler_name.strip().lower()
    resolved_version = compiler_version.strip().lower()

    flags: dict[str, str] = {}
    if isinstance(feature_flags, str):
        flag_items: Sequence[tuple[str, FeatureValue]] = ((feature_flags, True),)
    elif isinstance(feature_flags, Mapping):
        flag_items = tuple(feature_flags.items())
    elif feature_flags is not None:
        flag_items = tuple((flag, True) for flag in feature_flags)
    else:
        flag_items = ()
    for raw_key, raw_value in flag_items:
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if isinstance(raw_value, bool):
            value = "1" if raw_value else "0"
        elif raw_value is None:
            value = ""
        else:
            value = str(raw_value).strip().lower()
        flags[key] = value

    extras = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (extra or {}).items()
        if str(key).strip()
    }
    payload = {
        "os": resolved_os,
        "arch": resolved_arch,
        "compiler": resolved_compiler,
        "version": resolved_version,
        "flags": flags,
        "extra": extras,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    prefix = "-".join(
        _safe_prefix_part(part)
        for part in (resolved_os, resolved_arch, resolved_compiler)
    )
    return f"{prefix}-{digest}"


def ci_isolate_cargo_target_dir(
    profile: str,
    *,
    cache_root: Path | str,
    scope_key: str,
    env: MutableMapping[str, str] | None = None,
) -> Path:
    """Return a dedicated, created Cargo target directory for one profile.

    Each ``(scope_key, profile)`` pair maps to its own directory below
    ``cache_root/cargo``, preventing debug/test/bench artifacts from polluting
    each other.  When ``env`` is supplied, ``CARGO_TARGET_DIR`` is set to the
    returned path.  Unsafe path components are rejected rather than sanitized
    so distinct profiles cannot silently collide.
    """
    safe_profile = _validated_component(profile, "profile")
    safe_scope = _validated_component(scope_key, "scope_key")
    target = Path(cache_root).expanduser() / "cargo" / safe_scope / safe_profile
    target.mkdir(parents=True, exist_ok=True)
    if env is not None:
        env["CARGO_TARGET_DIR"] = str(target)
    return target


def ci_isolate_python_venv_paths(
    *,
    workspace: Path | str,
    cache_root: Path | str,
    scope_key: str,
    env: MutableMapping[str, str] | None = None,
    venv_name: str = ".venv",
) -> dict[str, Path]:
    """Plan isolated Python environment and artifact paths for a workspace.

    All returned paths live below ``cache_root/python/<scope_key>`` and never
    inside ``workspace``; overlapping roots are rejected.  Cache directories
    are created eagerly, while ``venv``/``python`` are only planned paths (the
    caller creates the virtual environment).  ``env`` receives ``VIRTUAL_ENV``,
    ``PIP_CACHE_DIR``, and ``PYTHONPYCACHEPREFIX`` when provided.  The
    interpreter subpath is the only platform-specific part (Windows uses
    ``Scripts/python.exe``).
    """
    resolved_workspace = Path(workspace).expanduser().resolve()
    resolved_cache = Path(cache_root).expanduser().resolve()
    if (
        resolved_workspace == resolved_cache
        or resolved_cache.is_relative_to(resolved_workspace)
        or resolved_workspace.is_relative_to(resolved_cache)
    ):
        raise ValueError(
            "cache_root must not overlap workspace: "
            f"{resolved_cache} vs {resolved_workspace}"
        )
    safe_scope = _validated_component(scope_key, "scope_key")
    if (
        not venv_name
        or venv_name in {".", ".."}
        or "/" in venv_name
        or "\\" in venv_name
    ):
        raise ValueError(f"venv_name must be a directory name: {venv_name!r}")

    base = resolved_cache / "python" / safe_scope
    venv = base / venv_name
    interpreter = (
        venv / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv / "bin" / "python"
    )
    paths = {
        "venv": venv,
        "python": interpreter,
        "pip_cache": base / "pip-cache",
        "build_cache": base / "build",
        "pytest_cache": base / "pytest-cache",
        "pycache": base / "pycache",
    }
    for key in ("pip_cache", "build_cache", "pytest_cache", "pycache"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if env is not None:
        env["VIRTUAL_ENV"] = str(venv)
        env["PIP_CACHE_DIR"] = str(paths["pip_cache"])
        env["PYTHONPYCACHEPREFIX"] = str(paths["pycache"])
    return paths


def ci_prune_stale_cache_entries(
    cache_root: Path | str,
    *,
    max_age_seconds: float = DEFAULT_CACHE_MAX_AGE_SECONDS,
    max_total_bytes: int = DEFAULT_CACHE_MAX_BYTES,
    now: float | None = None,
    lock_timeout_seconds: float = 5.0,
) -> CachePruneResult:
    """Evict expired and then oldest oversized cache entries under a lock.

    Entries are immediate children of ``cache_root`` (the internal lock file is
    excluded).  Expiry uses modification time relative to ``now`` (default:
    wall clock); afterwards the oldest entries are removed until the remaining
    recursive byte total fits ``max_total_bytes``.  An exclusive
    ``.prune.lock`` file serializes concurrent pruners, with stale locks
    reclaimed after a grace period.  Missing cache roots produce an empty
    report.
    """
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    if max_total_bytes < 0:
        raise ValueError("max_total_bytes must be non-negative")
    if lock_timeout_seconds < 0:
        raise ValueError("lock_timeout_seconds must be non-negative")

    result: CachePruneResult = {
        "removed": [],
        "removed_count": 0,
        "expired_count": 0,
        "oversized_count": 0,
        "freed_bytes": 0,
        "retained_bytes": 0,
    }
    root = Path(cache_root).expanduser()
    if not root.is_dir():
        return result

    lock_path = root / _PRUNE_LOCK_NAME
    lock_fd = _acquire_prune_lock(lock_path, lock_timeout_seconds)
    try:
        current_time = time.time() if now is None else now
        entries = sorted(
            (path for path in root.iterdir() if path.name != _PRUNE_LOCK_NAME),
            key=lambda path: path.name,
        )

        survivors: list[Path] = []
        for path in entries:
            try:
                age = current_time - _entry_stat(path).st_mtime
            except FileNotFoundError:
                continue
            if age > max_age_seconds:
                size = _entry_size(path)
                _remove_entry(path)
                result["removed"].append(path.name)
                result["removed_count"] += 1
                result["expired_count"] += 1
                result["freed_bytes"] += size
            else:
                survivors.append(path)

        sized = [
            (path, _entry_size(path), _entry_stat(path).st_mtime) for path in survivors
        ]
        total = sum(size for _, size, _ in sized)
        if total > max_total_bytes:
            for path, size, _ in sorted(
                sized, key=lambda item: (item[2], item[0].name)
            ):
                _remove_entry(path)
                result["removed"].append(path.name)
                result["removed_count"] += 1
                result["oversized_count"] += 1
                result["freed_bytes"] += size
                total -= size
                if total <= max_total_bytes:
                    break
        result["retained_bytes"] = total
        return result
    finally:
        os.close(lock_fd)
        _unlink_if_exists(lock_path)


def ci_verify_ruff_format_and_lint(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    ruff_executable: str = "ruff",
) -> tuple[CheckResult, ...]:
    """Run ``ruff format --check .`` and ``ruff check .`` at ``root``.

    Each command is reported separately; the default runner uses
    ``subprocess.run(..., check=False)`` so violations become failed checks
    instead of Python exceptions.
    """
    root_path = _resolve_root(root)
    return (
        _run_check(
            "ruff-format",
            (ruff_executable, "format", "--check", "."),
            root_path,
            runner,
        ),
        _run_check("ruff-lint", (ruff_executable, "check", "."), root_path, runner),
    )


def ci_verify_pytest_parity_suites(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
    suites: Sequence[str] | None = None,
) -> tuple[CheckResult, ...]:
    """Run core unit, generic, visualizer, and native-parity pytest suites.

    Each suite is invoked as ``<python> -m pytest -q <suite>`` with ``root`` as
    the working directory.  Pass ``suites`` to override the default suite list;
    results are returned in suite order.
    """
    root_path = _resolve_root(root)
    suite_paths = DEFAULT_PYTEST_SUITES if suites is None else tuple(suites)
    if not suite_paths:
        raise ValueError("suites must not be empty")
    python = python_executable or sys.executable
    checks: list[CheckResult] = []
    for suite in suite_paths:
        relative = _relative_path(suite, "suite")
        checks.append(
            _run_check(
                f"pytest:{relative}",
                (python, "-m", "pytest", "-q", relative),
                root_path,
                runner,
            )
        )
    return tuple(checks)


def ci_verify_rust_checks(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    cargo_executable: str = "cargo",
    crates: Sequence[str] | None = None,
) -> tuple[CheckResult, ...]:
    """Run ``cargo check`` and ``cargo clippy`` across workspace crates.

    By default both commands use ``--workspace --all-targets``; clippy denies
    warnings.  Supplying ``crates`` runs one check/clippy pair per crate with
    ``-p <crate>`` instead.
    """
    root_path = _resolve_root(root)
    check_tail = ("check", "--workspace", "--all-targets")
    clippy_tail = (
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    )
    targets: list[tuple[str, tuple[str, ...]]] = []
    if crates is None:
        targets.append(("cargo-check:workspace", check_tail))
        targets.append(("cargo-clippy:workspace", clippy_tail))
    else:
        crate_names = tuple(crates)
        if not crate_names:
            raise ValueError("crates must not be empty")
        for crate in crate_names:
            name = _validated_component(crate, "crate")
            targets.append(
                (f"cargo-check:{name}", ("check", "-p", name, "--all-targets"))
            )
            targets.append(
                (
                    f"cargo-clippy:{name}",
                    ("clippy", "-p", name, "--all-targets", "--", "-D", "warnings"),
                )
            )
    return tuple(
        _run_check(name, (cargo_executable, *tail), root_path, runner)
        for name, tail in targets
    )


def ci_verify_docs_sync_and_lint(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
) -> tuple[CheckResult, ...]:
    """Run ``tools/check_docs_sync.py`` then ``tools/lint_docs.py`` at ``root``.

    Both scripts are executed with the active Python interpreter and reported
    as independent checks.
    """
    root_path = _resolve_root(root)
    python = python_executable or sys.executable
    return (
        _run_check(
            "docs-sync",
            (python, DOCS_SYNC_SCRIPT),
            root_path,
            runner,
        ),
        _run_check("docs-lint", (python, DOCS_LINT_SCRIPT), root_path, runner),
    )


def ci_verify_clean_git_worktree(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    git_executable: str = "git",
    allowed_paths: Sequence[str] = (),
    ignore_patterns: Sequence[str] = (),
) -> CheckResult:
    """Fail when ``git status --short`` reports unexpected or temporary files.

    Untracked temp artifacts (cache directories, ``tmp*`` names, backup/log
    suffixes) are called out as temporary; all other changes are reported as
    unintended edits.  ``allowed_paths`` whitelists exact paths and their
    descendants; ``ignore_patterns`` uses :mod:`fnmatch` against normalized
    paths.  A failing ``git`` invocation is returned unchanged.
    """
    root_path = _resolve_root(root)
    result = _run_check(
        "git-worktree-clean",
        (git_executable, "status", "--short"),
        root_path,
        runner,
    )
    if not result.ok:
        return result

    allowed = tuple(_relative_path(path, "allowed_paths") for path in allowed_paths)
    patterns = tuple(ignore_patterns)
    findings: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        path = raw_path.strip().strip('"').replace("\\", "/")
        if not path:
            continue
        if path in allowed or any(path.startswith(f"{item}/") for item in allowed):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            continue
        label = "temporary file" if _is_temp_path(path) else "unexpected change"
        findings.append(f"{label}: {path}")

    if findings:
        return CheckResult(
            name=result.name,
            command=result.command,
            returncode=1,
            stdout="\n".join(findings),
            stderr=result.stderr,
        )
    if result.stdout:
        # All reported changes were allowlisted; present a clean result rather
        # than forwarding raw git output that looks like a dirty worktree.
        return CheckResult(
            name=result.name,
            command=result.command,
            returncode=0,
            stdout="",
            stderr=result.stderr,
        )
    return result


def ci_verify_rust_format_and_doc(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    cargo_executable: str = "cargo",
) -> tuple[CheckResult, ...]:
    """Run ``cargo fmt --all -- --check`` and ``cargo doc --no-deps --workspace``.

    Both commands run in ``root/native_pillars`` when that crate has a
    ``Cargo.toml``, otherwise in ``root`` itself.  The caller can redirect Cargo
    artifacts to an isolated cache directory through the injected runner, so
    this verifier never writes into the working tree.
    """
    root_path = _resolve_root(root)
    rust_root = root_path / "native_pillars"
    if not (rust_root / "Cargo.toml").is_file():
        rust_root = root_path
    return (
        _run_check(
            "cargo-fmt",
            (cargo_executable, "fmt", "--all", "--", "--check"),
            rust_root,
            runner,
        ),
        _run_check(
            "cargo-doc",
            (cargo_executable, "doc", "--no-deps", "--workspace"),
            rust_root,
            runner,
        ),
    )


def ci_verify_dependency_audit(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
) -> tuple[CheckResult, ...]:
    """Audit installed dependencies and minimal ``pyproject.toml`` metadata.

    ``pip check`` is executed through the injected runner; the metadata audit is
    synthesized in-process.  It requires ``project.license`` and
    ``project.requires-python`` and resolves each ``[project.scripts]`` target
    module with :func:`importlib.util.find_spec` (without importing the module).
    """
    root_path = _resolve_root(root)
    python = python_executable or sys.executable
    pip_check = _run_check(
        "pip-check", (python, "-m", "pip", "check"), root_path, runner
    )

    findings: list[str] = []
    pyproject = root_path / "pyproject.toml"
    if not pyproject.is_file():
        findings.append("pyproject.toml is missing")
    else:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            findings.append(f"cannot parse pyproject.toml: {error}")
        else:
            project = data.get("project")
            if not isinstance(project, Mapping):
                findings.append("missing [project] table")
            else:
                if not project.get("license"):
                    findings.append("project.license is not declared")
                if not project.get("requires-python"):
                    findings.append("project.requires-python is not declared")
                scripts = project.get("scripts") or {}
                if not isinstance(scripts, Mapping):
                    findings.append("[project.scripts] must be a table")
                else:
                    for script_name, target in sorted(scripts.items()):
                        module = str(target).split(":", 1)[0].strip()
                        if not module:
                            findings.append(
                                f"script {script_name!r} has no importable module"
                            )
                            continue
                        try:
                            spec = importlib.util.find_spec(module)
                        except (ImportError, ValueError, TypeError) as error:
                            findings.append(
                                f"script {script_name!r} module {module!r}: {error}"
                            )
                        else:
                            if spec is None:
                                findings.append(
                                    f"script {script_name!r} module {module!r} "
                                    "is not importable"
                                )

    metadata = _in_process_check(
        "metadata-audit",
        ok=not findings,
        detail="license, requires-python, and script modules are valid",
        error="\n".join(findings),
    )
    return (pip_check, metadata)


def ci_verify_doc_links(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
) -> CheckResult:
    """Check relative Markdown links in the tracked top-level docs and docs/*.md.

    Anchors, external schemes, absolute paths, ``<placeholder>`` targets, and
    fenced code blocks are ignored.  A single synthesized ``doc-links`` result
    lists up to ten broken targets; links are resolved relative to the file
    that contains them.  ``runner`` and ``python_executable`` are accepted for
    a uniform verifier interface, but this check performs no subprocess work.
    """
    del runner, python_executable
    root_path = _resolve_root(root)
    doc_paths = [
        root_path / name for name in TRACKED_DOC_NAMES if (root_path / name).is_file()
    ]
    doc_paths.extend(
        sorted(path for path in (root_path / "docs").glob("*.md") if path.is_file())
    )

    broken: list[str] = []
    scanned = 0
    for path in doc_paths:
        relative_doc = path.relative_to(root_path).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            broken.append(f"{relative_doc}: cannot read ({error})")
            continue
        scanned += 1
        in_fence = False
        for line_number, line in enumerate(text.splitlines(), 1):
            if _MARKDOWN_FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in _MARKDOWN_LINK_RE.finditer(line):
                raw_target = match.group(1).strip()
                if not raw_target:
                    continue
                target = raw_target.split()[0]
                if (
                    target.startswith(("#", "/", "http://", "https://", "mailto:"))
                    or ":" in target
                    or "<" in target
                    or ">" in target
                ):
                    continue
                relative_target = target.split("#", 1)[0]
                if relative_target and not (path.parent / relative_target).exists():
                    broken.append(f"{relative_doc}:{line_number}: {target}")

    if broken:
        shown = "; ".join(broken[:10])
        suffix = " ..." if len(broken) > 10 else ""
        return _in_process_check(
            "doc-links",
            ok=False,
            detail="",
            error=f"{len(broken)} broken relative link(s): {shown}{suffix}",
        )
    return _in_process_check(
        "doc-links",
        ok=True,
        detail=f"checked {scanned} tracked docs; all relative links resolve",
    )


def _benchmark_baseline_path(root: Path, cache_root: Path | str | None) -> Path:
    """Return a baseline file in a temp cache root outside ``root``."""
    if cache_root is None:
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        base = Path(tempfile.gettempdir()) / f"polyglot-benchmark-{digest}"
    else:
        base = Path(cache_root).expanduser()
    resolved = base.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError("benchmark cache root must be outside the workspace")
    base.mkdir(parents=True, exist_ok=True)
    return base / BENCHMARK_BASELINE_FILENAME


def _benchmark_work_units() -> int:
    """Run a fixed pure workload and return a deterministic operation count.

    Wall-clock timing is deliberately avoided: microbenchmarks that measure
    elapsed time flake easily on shared CI hosts.  Instead the median of five
    runs counts tokens, consumed parse tokens, and registry resolutions for a
    fixed source.  A regression means the same workload suddenly performs a
    different amount of work, which is reproducible across machines.
    """
    from framework_functions import get_function

    lex = get_function(1).callable
    parse = get_function(4).callable
    resolve = get_function
    total = 0
    for _ in range(BENCHMARK_REPETITIONS):
        lexed = lex(BENCHMARK_SOURCE)
        parsed = parse(BENCHMARK_SOURCE)
        total += int(lexed["token_count"])
        total += int(parsed.get("tokens_consumed", 0))
        total += resolve(1).number
    return total


def ci_verify_benchmark_regression(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
    threshold: float = DEFAULT_BENCHMARK_THRESHOLD,
    cache_root: Path | str | None = None,
) -> CheckResult:
    """Guard against benchmark regressions using deterministic work counts.

    The median of five runs of a fixed registry workload is compared with a
    baseline JSON stored below a temp cache root outside the workspace.  When
    no valid baseline exists it is recorded and a passing check with a note is
    returned.  ``threshold`` is the maximum allowed current/baseline ratio.
    Operation counts are used instead of wall-clock durations so the gate
    cannot flake because of scheduler noise.
    """
    del runner, python_executable
    root_path = _resolve_root(root)
    limit = float(threshold)
    if not limit > 0:
        raise ValueError("threshold must be positive")
    baseline_path = _benchmark_baseline_path(root_path, cache_root)
    current = int(
        statistics.median(_benchmark_work_units() for _ in range(BENCHMARK_RUNS))
    )

    baseline: int | None = None
    if baseline_path.is_file():
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            candidate = int(payload["operations"])
            if candidate > 0:
                baseline = candidate
        except OSError, KeyError, TypeError, ValueError:
            baseline = None

    if baseline is None:
        record = {
            "version": 1,
            "operations": current,
            "source": BENCHMARK_SOURCE,
            "repetitions": BENCHMARK_REPETITIONS,
            "runs": BENCHMARK_RUNS,
        }
        baseline_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        return _in_process_check(
            "benchmark-regression",
            ok=True,
            detail=(
                f"baseline recorded: operations={current}; deterministic "
                "operation counts (no wall clock)"
            ),
        )

    ratio = current / baseline
    if ratio > limit:
        return _in_process_check(
            "benchmark-regression",
            ok=False,
            detail="",
            error=(
                f"benchmark regression: current={current} baseline={baseline} "
                f"ratio={ratio:.3f} exceeds threshold={limit:.2f}"
            ),
        )
    return _in_process_check(
        "benchmark-regression",
        ok=True,
        detail=(
            f"operations={current} baseline={baseline} ratio={ratio:.3f} "
            f"threshold={limit:.2f}"
        ),
    )


def ci_verify_release_metadata(
    *,
    root: Path | str,
    runner: CommandRunner | None = None,
    python_executable: str | None = None,
) -> CheckResult:
    """Assert that packaging include lists cover every registry module.

    The check is fully in-process: it reads ``pyproject.toml``, confirms that
    each module referenced by :data:`framework_functions.FUNCTIONS` is covered
    by ``py-modules`` or ``packages.find.include`` (a pattern may match any
    package prefix), and verifies that ``tools.build_ci`` is exported by the
    wheel/sdist configuration.
    """
    del runner, python_executable
    from framework_functions import FUNCTIONS

    root_path = _resolve_root(root)
    pyproject = root_path / "pyproject.toml"
    if not pyproject.is_file():
        return _in_process_check(
            "release-metadata", ok=False, detail="", error="pyproject.toml is missing"
        )
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return _in_process_check(
            "release-metadata",
            ok=False,
            detail="",
            error=f"cannot parse pyproject.toml: {error}",
        )

    tool = data.get("tool")
    setuptools = tool.get("setuptools", {}) if isinstance(tool, Mapping) else {}
    if not isinstance(setuptools, Mapping):
        setuptools = {}
    raw_modules = setuptools.get("py-modules", ())
    packages = setuptools.get("packages", {})
    find = packages.get("find", {}) if isinstance(packages, Mapping) else {}
    raw_includes = find.get("include", ()) if isinstance(find, Mapping) else ()
    patterns = {
        str(item)
        for item in (
            *(raw_modules if isinstance(raw_modules, (list, tuple)) else ()),
            *(raw_includes if isinstance(raw_includes, (list, tuple)) else ()),
        )
    }

    def covered(module: str) -> bool:
        return any(fnmatch.fnmatchcase(module, pattern) for pattern in patterns)

    def covered_registry_module(module: str) -> bool:
        parts = module.split(".")
        return any(
            covered(".".join(parts[:length])) for length in range(len(parts), 0, -1)
        )

    registry_modules = {item.module for item in FUNCTIONS}
    missing = sorted(
        module for module in registry_modules if not covered_registry_module(module)
    )
    if not covered("tools.build_ci"):
        missing.append("tools.build_ci")
    try:
        spec = importlib.util.find_spec("tools.build_ci.framework")
    except ImportError, ValueError, TypeError:
        spec = None
    if spec is None:
        missing.append("tools.build_ci.framework (not importable)")

    if missing:
        return _in_process_check(
            "release-metadata",
            ok=False,
            detail="",
            error="wheel/sdist include lists miss: " + ", ".join(missing),
        )
    return _in_process_check(
        "release-metadata",
        ok=True,
        detail=(
            f"include lists cover {len(registry_modules)} registry module(s) "
            "and tools.build_ci"
        ),
    )


def ci_generate_release_gate_report(
    *,
    checks: Mapping[str, Sequence[CheckResult] | bool],
    blockers: Sequence[str] = (),
    required_checks: Sequence[str] = (),
) -> ReleaseGateReport:
    """Summarize gate statuses, pass/fail counts, and release blockers.

    ``checks`` maps a gate name to either an explicit boolean or the
    :class:`CheckResult` tuple produced by the verifier functions.  Every
    failed result automatically becomes a blocker; extra ``blockers`` and
    missing ``required_checks`` are appended in stable order without
    duplicates.
    """
    statuses: list[GateCheckStatus] = []
    gate_blockers: list[str] = []
    for name in sorted(checks):
        value = checks[name]
        if isinstance(value, bool):
            statuses.append(
                GateCheckStatus(name=name, ok=value, detail="explicit status")
            )
            if not value:
                gate_blockers.append(f"{name}: check failed")
            continue
        results = tuple(value)
        if not results:
            statuses.append(GateCheckStatus(name=name, ok=False, detail="no results"))
            gate_blockers.append(f"{name}: no results")
            continue
        for result in results:
            label = f"{name}:{result.name}" if result.name else name
            detail = f"exit {result.returncode}"
            first_error = next(
                (line.strip() for line in result.stderr.splitlines() if line.strip()),
                "",
            )
            if first_error:
                detail = f"{detail}; {first_error}"
            statuses.append(GateCheckStatus(name=label, ok=result.ok, detail=detail))
            if not result.ok:
                gate_blockers.append(f"{label}: exit {result.returncode}")

    for required in required_checks:
        if required not in checks:
            statuses.append(
                GateCheckStatus(
                    name=required, ok=False, detail="missing required check"
                )
            )
            gate_blockers.append(f"{required}: missing required check")

    all_blockers = tuple(dict.fromkeys((*gate_blockers, *blockers)))
    return ReleaseGateReport(checks=tuple(statuses), blockers=all_blockers)


__all__ = [
    "BENCHMARK_BASELINE_FILENAME",
    "BENCHMARK_REPETITIONS",
    "BENCHMARK_RUNS",
    "BENCHMARK_SOURCE",
    "DEFAULT_BENCHMARK_THRESHOLD",
    "DEFAULT_PYTEST_SUITES",
    "TRACKED_DOC_NAMES",
    "CachePruneResult",
    "CheckResult",
    "CommandRunner",
    "GateCheckStatus",
    "ReleaseGateReport",
    "ci_compute_cache_scope_key",
    "ci_generate_release_gate_report",
    "ci_isolate_cargo_target_dir",
    "ci_isolate_python_venv_paths",
    "ci_prune_stale_cache_entries",
    "ci_verify_benchmark_regression",
    "ci_verify_clean_git_worktree",
    "ci_verify_dependency_audit",
    "ci_verify_doc_links",
    "ci_verify_docs_sync_and_lint",
    "ci_verify_pytest_parity_suites",
    "ci_verify_release_metadata",
    "ci_verify_ruff_format_and_lint",
    "ci_verify_rust_checks",
    "ci_verify_rust_format_and_doc",
]
