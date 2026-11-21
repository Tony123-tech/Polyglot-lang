"""Injectable CI verification helpers with explicit repository roots.

All command execution goes through a caller-supplied runner that receives the
final argv, an explicit working directory, and the environment snapshot.
Tests therefore never spawn real ``pytest``, ``cargo``, ``ruff``, or ``git``
processes.  Cargo verification always installs a scoped ``CARGO_TARGET_DIR``.
Invalid API inputs raise E003 through
:class:`~compiler.frontend.errors.PolyglotError`; runner defects and tool
failures raise E004, while missing roots or artifacts raise E006.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "bv_aggregate_checks",
    "bv_check_env_allowlist",
    "bv_classify_check_result",
    "bv_classify_git_line",
    "bv_parse_command_output",
    "bv_plan_cargo_checks",
    "bv_plan_doc_checks",
    "bv_plan_git_status",
    "bv_plan_pytest_suites",
    "bv_plan_ruff_checks",
    "bv_render_check_summary",
    "bv_require_runner",
    "bv_verify_artifact_exists",
    "bv_verify_cargo",
    "bv_verify_cargo_env",
    "bv_verify_command_plan",
    "bv_verify_docs",
    "bv_verify_env_isolation",
    "bv_verify_gate_matrix",
    "bv_verify_git_worktree",
    "bv_verify_platform_support",
    "bv_verify_pytest",
    "bv_verify_ruff",
    "bv_verify_snapshot",
    "bv_verify_tool_availability",
]

CommandRunner = Callable[
    [tuple[str, ...], Path, Mapping[str, str]], subprocess.CompletedProcess[str]
]

_TEMP_DIR_NAMES = frozenset({"target", "build", "__pycache__", ".pytest_cache"})
_TEMP_SUFFIXES = (".tmp", ".log", ".bak", ".orig", ".rej", ".swp")
_PATH_ENV_KEYS = frozenset(
    {
        "CARGO_TARGET_DIR",
        "CARGO_HOME",
        "PIP_CACHE_DIR",
        "RUFF_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "PYTEST_CACHE_DIR",
    }
)
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHELL_MARKERS = frozenset({"sh", "bash", "zsh", "cmd", "powershell"})


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured verification diagnostic."""
    raise PolyglotError(code, message, phase="build-ci")


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    """Require a string, optionally allowing the empty string."""
    if not isinstance(value, str) or (not value and not allow_empty):
        _fail("E003", f"{name} must be a{' non-empty' if not allow_empty else ''} str")
    return value


def _path(value: object, name: str) -> Path:
    """Require a path-like value and expand ``~`` only."""
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    _fail("E003", f"{name} must be a str or pathlib.Path")


def _component(value: object, name: str) -> str:
    """Validate a command or path component that becomes a cache scope."""
    text = _text(value, name)
    if text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        _fail("E003", f"{name} must be a safe component: {text!r}")
    return text


def _sequence(value: object, name: str) -> list[Any]:
    """Require a non-string sequence and return a mutable copy."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail("E003", f"{name} must be a sequence")
    return list(value)


def _env(value: object, name: str = "env") -> dict[str, str]:
    """Require a mapping of string environment keys to string values."""
    if not isinstance(value, Mapping):
        _fail("E003", f"{name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_text(key, f"{name} key")] = _text(
            item, f"{name}[{key!r}]", allow_empty=True
        )
    return dict(sorted(result.items()))


def _argv(args: object, name: str = "args") -> tuple[str, ...]:
    """Require a non-empty argv sequence of non-empty strings."""
    values = tuple(_text(item, f"{name} item") for item in _sequence(args, name))
    if not values:
        _fail("E003", f"{name} must not be empty")
    return values


def _relative_spec(value: object, name: str) -> str:
    """Require a relative, traversal-free POSIX path specification."""
    text = _text(value, name).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not text:
        _fail("E003", f"{name} must be a relative path: {text!r}")
    return pure.as_posix()


def _as_text(value: object) -> str:
    """Normalize runner output to text without failing on bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _digest(payload: Any) -> str:
    """Canonical SHA-256 digest of a JSON-compatible payload."""
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("E003", f"payload is not canonicalizable: {error}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cwd(root: object, *, must_exist: bool) -> Path:
    """Resolve an explicit repository root for a command invocation."""
    directory = _path(root, "root")
    if must_exist and not directory.is_dir():
        _fail("E006", f"repository root is not a directory: {directory}")
    return directory


def _runner(runner: object) -> CommandRunner:
    """Require an explicit callable command runner."""
    if not callable(runner):
        _fail("E003", "runner must be an explicit callable")
    return runner  # type: ignore[return-value]


def _execute(
    runner: object,
    root: object,
    args: object,
    env: object,
    name: str,
) -> dict[str, Any]:
    """Run one command through the injected runner and normalize its result."""
    command_runner = _runner(runner)
    directory = _cwd(root, must_exist=True)
    argv = _argv(args)
    prepared = _env(env)
    try:
        process = command_runner(argv, directory, dict(prepared))
    except PolyglotError:
        raise
    except Exception as error:
        _fail("E004", f"runner failed for {name}: {error}")
    returncode = getattr(process, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        _fail("E004", f"runner returned no integer exit code for {name}")
    stdout = _as_text(getattr(process, "stdout", ""))
    stderr = _as_text(getattr(process, "stderr", ""))
    first_error = next(
        (line.strip() for line in stderr.splitlines() if line.strip()), ""
    )
    return {
        "name": name,
        "command": list(argv),
        "cwd": str(directory),
        "env": dict(prepared),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "first_error": first_error,
        "ok": returncode == 0,
    }


def _plan(name: str, args: Sequence[str], root: Path) -> dict[str, Any]:
    """Build one deterministic command-plan record."""
    return {"name": name, "args": list(args), "cwd": str(root)}


def _is_temporary(path: str) -> bool:
    """Whether a git status path looks like a generated temporary artifact."""
    parts = path.replace("\\", "/").lower().split("/")
    name = parts[-1]
    return (
        any(part in _TEMP_DIR_NAMES for part in parts)
        or name.startswith(("tmp", ".tmp"))
        or name.endswith(_TEMP_SUFFIXES)
    )


def bv_aggregate_checks(
    results: object, *, required: Sequence[str] = ()
) -> dict[str, Any]:
    """Aggregate normalized check outcomes with optional required names."""
    required_names = tuple(
        _text(item, "required item") for item in _sequence(required, "required")
    )
    normalized: list[dict[str, Any]] = []
    for result in _sequence(results, "results"):
        normalized.append(bv_classify_check_result(result))
    names = tuple(item["name"] for item in normalized)
    failed = tuple(item["name"] for item in normalized if not item["ok"])
    missing = tuple(name for name in required_names if name not in names)
    total = len(normalized)
    passed_count = total - len(failed)
    return {
        "passed": not failed and not missing,
        "total": total,
        "passed_count": passed_count,
        "failed_count": len(failed),
        "failed_names": list(failed),
        "missing_required": list(missing),
        "names": list(names),
        "digest": _digest([names, failed, missing]),
    }


def bv_check_env_allowlist(
    env: object,
    *,
    allowed_prefixes: Sequence[str] = (),
    forbidden_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Check environment keys against explicit allow and deny lists."""
    prepared = _env(env)
    allowed = tuple(
        _text(item, "prefix")
        for item in _sequence(allowed_prefixes, "allowed_prefixes")
    )
    forbidden = tuple(
        _text(item, "key") for item in _sequence(forbidden_keys, "forbidden_keys")
    )
    violations: list[str] = []
    for key in prepared:
        if allowed and not any(key.startswith(prefix) for prefix in allowed):
            violations.append(f"not allowed: {key}")
        if key in forbidden:
            violations.append(f"forbidden: {key}")
    return {
        "passed": not violations,
        "violations": sorted(violations),
        "checked": list(prepared),
    }


def bv_classify_check_result(
    result: object, *, name: str | None = None
) -> dict[str, Any]:
    """Normalize a command outcome mapping into a stable check record."""
    if not isinstance(result, Mapping):
        _fail("E003", "result must be a mapping")
    label = _text(name if name is not None else result.get("name"), "name")
    code = result.get("returncode", result.get("exit_code", 0))
    if isinstance(code, bool) or not isinstance(code, int):
        _fail("E003", "returncode must be an int")
    stderr = _as_text(result.get("stderr", ""))
    detail = _text(result.get("detail", ""), "detail", allow_empty=True)
    if not detail and code != 0:
        detail = next(
            (line.strip() for line in stderr.splitlines() if line.strip()),
            f"exit {code}",
        )
    return {
        "name": label,
        "ok": code == 0 and bool(result.get("ok", code == 0)),
        "returncode": code,
        "detail": detail,
        "fingerprint": _digest([label, code, detail]),
    }


def bv_classify_git_line(
    line: object,
    *,
    allowed_paths: Sequence[str] = (),
    ignore_patterns: Sequence[str] = (),
) -> dict[str, Any]:
    """Classify one ``git status --short`` line without running git."""
    text = _text(line, "line", allow_empty=True)
    if len(text) < 4:
        return {
            "path": "",
            "status": "",
            "category": "unparsed",
            "reason": "line is too short",
        }
    status = text[:2]
    raw = text[3:]
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    path = raw.strip().strip('"').replace("\\", "/")
    allowed = tuple(
        _relative_spec(item, "allowed path")
        for item in _sequence(allowed_paths, "allowed_paths")
    )
    patterns = tuple(
        _text(item, "ignore pattern")
        for item in _sequence(ignore_patterns, "ignore_patterns")
    )
    if path in allowed or any(path.startswith(f"{item}/") for item in allowed):
        category, reason = "allowed", "allowlisted path"
    elif any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
        category, reason = "ignored", "matched ignore pattern"
    elif _is_temporary(path):
        category, reason = "temporary", "temporary artifact"
    else:
        category, reason = "unexpected", "unexpected change"
    return {"path": path, "status": status, "category": category, "reason": reason}


def bv_parse_command_output(
    label: object, *, stdout: object, stderr: object, returncode: object
) -> dict[str, Any]:
    """Parse raw command output into a deterministic record."""
    name = _text(label, "label")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        _fail("E003", "returncode must be an int")
    out = _as_text(stdout)
    err = _as_text(stderr)
    lines = out.splitlines()
    errors = [line.strip() for line in err.splitlines() if line.strip()]
    return {
        "name": name,
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout_lines": lines,
        "stderr_lines": errors,
        "first_error": errors[0] if errors else "",
        "line_count": len(lines),
    }


def bv_plan_cargo_checks(
    root: object, *, crates: Sequence[str] | None = None
) -> dict[str, Any]:
    """Plan deterministic ``cargo check`` and ``cargo clippy`` commands."""
    directory = _cwd(root, must_exist=False)
    targets: list[dict[str, Any]] = []
    if crates is None:
        targets.append(
            _plan(
                "cargo-check:workspace",
                ("cargo", "check", "--workspace", "--all-targets"),
                directory,
            )
        )
        targets.append(
            _plan(
                "cargo-clippy:workspace",
                (
                    "cargo",
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ),
                directory,
            )
        )
    else:
        names = tuple(_component(item, "crate") for item in _sequence(crates, "crates"))
        if not names:
            _fail("E003", "crates must not be empty")
        for crate in names:
            targets.append(
                _plan(
                    f"cargo-check:{crate}",
                    ("cargo", "check", "-p", crate, "--all-targets"),
                    directory,
                )
            )
            targets.append(
                _plan(
                    f"cargo-clippy:{crate}",
                    (
                        "cargo",
                        "clippy",
                        "-p",
                        crate,
                        "--all-targets",
                        "--",
                        "-D",
                        "warnings",
                    ),
                    directory,
                )
            )
    return {"checks": targets, "count": len(targets), "root": str(directory)}


def bv_plan_doc_checks(
    root: object, *, python_executable: object = "python"
) -> dict[str, Any]:
    """Plan the docs-sync and docs-lint invocations for a repository root."""
    directory = _cwd(root, must_exist=False)
    python = _text(python_executable, "python_executable")
    checks = [
        _plan("docs-sync", (python, "tools/check_docs_sync.py"), directory),
        _plan("docs-lint", (python, "tools/lint_docs.py"), directory),
    ]
    return {"checks": checks, "count": len(checks), "root": str(directory)}


def bv_plan_git_status(
    root: object, *, git_executable: object = "git"
) -> dict[str, Any]:
    """Plan a single ``git status --short`` invocation."""
    directory = _cwd(root, must_exist=False)
    git = _text(git_executable, "git_executable")
    return {
        "checks": [_plan("git-worktree-clean", (git, "status", "--short"), directory)],
        "count": 1,
        "root": str(directory),
    }


def bv_plan_pytest_suites(
    root: object,
    *,
    suites: object,
    python_executable: object = "python",
) -> dict[str, Any]:
    """Plan one ``python -m pytest -q`` command per relative suite path."""
    directory = _cwd(root, must_exist=False)
    python = _text(python_executable, "python_executable")
    specs = tuple(_relative_spec(item, "suite") for item in _sequence(suites, "suites"))
    if not specs:
        _fail("E003", "suites must not be empty")
    checks = [
        _plan(
            f"pytest:{spec}",
            (python, "-m", "pytest", "-q", spec),
            directory,
        )
        for spec in specs
    ]
    return {"checks": checks, "count": len(checks), "root": str(directory)}


def bv_plan_ruff_checks(
    root: object, *, ruff_executable: object = "ruff"
) -> dict[str, Any]:
    """Plan ``ruff format --check .`` and ``ruff check .`` invocations."""
    directory = _cwd(root, must_exist=False)
    ruff = _text(ruff_executable, "ruff_executable")
    checks = [
        _plan("ruff-format", (ruff, "format", "--check", "."), directory),
        _plan("ruff-lint", (ruff, "check", "."), directory),
    ]
    return {"checks": checks, "count": len(checks), "root": str(directory)}


def bv_require_runner(runner: object) -> dict[str, Any]:
    """Validate that a real command runner was explicitly supplied."""
    command_runner = _runner(runner)
    return {
        "accepted": True,
        "kind": type(command_runner).__name__,
        "callable": True,
    }


def bv_render_check_summary(results: object) -> dict[str, Any]:
    """Render normalized check records into a stable text summary."""
    records = [bv_classify_check_result(item) for item in _sequence(results, "results")]
    lines = [
        f"CHECKS: {sum(1 for item in records if item['ok'])} passed, "
        f"{sum(1 for item in records if not item['ok'])} failed"
    ]
    for record in sorted(records, key=lambda item: item["name"]):
        state = "PASS" if record["ok"] else "FAIL"
        detail = f" ({record['detail']})" if record["detail"] else ""
        lines.append(f"  {state} {record['name']}{detail}")
    return {
        "text": "\n".join(lines),
        "lines": lines,
        "passed_count": sum(1 for item in records if item["ok"]),
        "failed_count": sum(1 for item in records if not item["ok"]),
    }


def bv_verify_artifact_exists(
    path: object, *, root: object, kind: object = "artifact"
) -> dict[str, Any]:
    """Verify that a build artifact exists below an explicit repository root."""
    directory = _cwd(root, must_exist=False)
    artifact = _path(path, "path")
    if not artifact.is_absolute():
        artifact = directory / artifact
    if not artifact.exists():
        _fail("E006", f"{kind} is missing: {artifact}")
    try:
        stat = artifact.stat()
    except OSError as error:
        _fail("E004", f"cannot stat {kind} {artifact}: {error}")
    return {
        "path": str(artifact),
        "kind": _text(kind, "kind"),
        "exists": True,
        "size_bytes": stat.st_size,
        "is_dir": artifact.is_dir(),
    }


def bv_verify_cargo(
    root: object,
    *,
    runner: object,
    cache_root: object,
    scope_key: object,
    profile: object,
    crates: Sequence[str] | None = None,
    env: object | None = None,
) -> dict[str, Any]:
    """Run planned Cargo checks with a scoped target directory.

    The caller must supply both a runner and a repository root.  Any
    pre-existing ``CARGO_TARGET_DIR`` that points outside the cache is replaced
    with the derived isolated path before the runner is invoked.
    """
    plan = bv_plan_cargo_checks(root, crates=crates)
    directory = _cwd(root, must_exist=True)
    cache = _path(cache_root, "cache_root").resolve()
    scope = _component(scope_key, "scope_key")
    name = _component(profile, "profile")
    expected = cache / "cargo" / scope / name
    prepared = {} if env is None else _env(env)
    isolation = bv_verify_env_isolation(prepared, cache_root=cache, required_keys=())
    prepared["CARGO_TARGET_DIR"] = str(expected)
    checks = [
        _execute(runner, directory, item["args"], prepared, item["name"])
        for item in plan["checks"]
    ]
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "checks": checks,
        "passed": not failed,
        "failed_names": failed,
        "cargo_target_dir": str(expected),
        "scope_key": scope,
        "profile": name,
        "env": dict(sorted(prepared.items())),
        "original_violations": isolation["violations"],
    }


def bv_verify_cargo_env(
    *,
    cache_root: object,
    scope_key: object,
    profile: object,
    env: object,
) -> dict[str, Any]:
    """Check that ``CARGO_TARGET_DIR`` points at the derived scoped directory."""
    cache = _path(cache_root, "cache_root").resolve()
    scope = _component(scope_key, "scope_key")
    name = _component(profile, "profile")
    expected = cache / "cargo" / scope / name
    prepared = _env(env)
    actual = prepared.get("CARGO_TARGET_DIR")
    violations: list[str] = []
    if actual is None:
        violations.append("CARGO_TARGET_DIR is missing")
    elif _path(actual, "CARGO_TARGET_DIR").resolve() != expected:
        violations.append(f"CARGO_TARGET_DIR is not {expected}")
    isolation = bv_verify_env_isolation(prepared, cache_root=cache, required_keys=())
    violations.extend(isolation["violations"])
    return {
        "expected": str(expected),
        "actual": actual,
        "isolated": not violations,
        "violations": sorted(set(violations)),
    }


def bv_verify_command_plan(plan: object, *, root: object) -> dict[str, Any]:
    """Validate a command plan before an injected runner is allowed to use it."""
    if not isinstance(plan, Mapping):
        _fail("E003", "plan must be a mapping")
    directory = _cwd(root, must_exist=False)
    checks = _sequence(plan.get("checks", ()), "plan checks")
    violations: list[str] = []
    commands: list[list[str]] = []
    for item in checks:
        if not isinstance(item, Mapping):
            _fail("E003", "each planned check must be a mapping")
        args = _argv(item.get("args", ()))
        cwd = _path(item.get("cwd", ""), "check cwd")
        commands.append(list(args))
        if cwd != directory:
            violations.append(f"{item.get('name', '?')}: cwd is not the root")
        if args[0] in _SHELL_MARKERS:
            violations.append(f"{item.get('name', '?')}: shell wrapper is not allowed")
    return {
        "valid": not violations,
        "violations": sorted(violations),
        "commands": commands,
        "root": str(directory),
    }


def bv_verify_docs(
    root: object,
    *,
    runner: object,
    python_executable: object = "python",
    env: object | None = None,
) -> dict[str, Any]:
    """Run the planned docs checks through an injected runner."""
    plan = bv_plan_doc_checks(root, python_executable=python_executable)
    directory = _cwd(root, must_exist=True)
    prepared = {} if env is None else _env(env)
    checks = [
        _execute(runner, directory, item["args"], prepared, item["name"])
        for item in plan["checks"]
    ]
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "checks": checks,
        "passed": not failed,
        "failed_names": failed,
        "root": str(directory),
    }


def bv_verify_env_isolation(
    env: object,
    *,
    cache_root: object,
    required_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Check that cache-related environment paths stay below *cache_root*."""
    prepared = _env(env)
    root = _path(cache_root, "cache_root").resolve()
    required = tuple(
        _text(item, "required key")
        for item in _sequence(required_keys, "required_keys")
    )
    violations: list[str] = []
    for key in required:
        if key not in prepared:
            violations.append(f"missing: {key}")
    checked: list[str] = []
    for key, value in prepared.items():
        if key not in _PATH_ENV_KEYS:
            continue
        checked.append(key)
        resolved = _path(value, key).resolve()
        if not resolved.is_relative_to(root):
            violations.append(f"{key} escapes cache root: {resolved}")
    return {
        "isolated": not violations,
        "violations": sorted(violations),
        "checked": sorted(checked),
        "cache_root": str(root),
    }


def bv_verify_gate_matrix(
    matrix: object, *, required: Sequence[str] = ()
) -> dict[str, Any]:
    """Aggregate a matrix of gate names mapped to check-record sequences."""
    if not isinstance(matrix, Mapping):
        _fail("E003", "matrix must be a mapping")
    required_names = tuple(
        _text(item, "required gate") for item in _sequence(required, "required")
    )
    gates: list[dict[str, Any]] = []
    blockers: list[str] = []
    for gate in sorted(matrix):
        summary = bv_aggregate_checks(matrix[gate])
        gates.append(
            {
                "name": gate,
                "passed": summary["passed"],
                "checks": summary["total"],
                "failed": summary["failed_names"],
            }
        )
        if not summary["passed"]:
            blockers.append(f"{gate}: checks failed")
    for name in required_names:
        if name not in matrix:
            gates.append({"name": name, "passed": False, "checks": 0, "failed": []})
            blockers.append(f"{name}: missing gate")
    return {
        "passed": not blockers,
        "gates": gates,
        "blockers": sorted(set(blockers)),
        "digest": _digest([gates, sorted(set(blockers))]),
    }


def bv_verify_git_worktree(
    root: object,
    *,
    runner: object,
    allowed_paths: Sequence[str] = (),
    ignore_patterns: Sequence[str] = (),
    env: object | None = None,
) -> dict[str, Any]:
    """Verify ``git status --short`` output through an injected runner."""
    plan = bv_plan_git_status(root)
    directory = _cwd(root, must_exist=True)
    prepared = {} if env is None else _env(env)
    outcome = _execute(
        runner,
        directory,
        plan["checks"][0]["args"],
        prepared,
        plan["checks"][0]["name"],
    )
    findings: list[dict[str, Any]] = []
    allowed: list[str] = []
    for line in outcome["stdout"].splitlines():
        classified = bv_classify_git_line(
            line, allowed_paths=allowed_paths, ignore_patterns=ignore_patterns
        )
        if classified["category"] == "allowed":
            allowed.append(classified["path"])
        elif classified["category"] in {"temporary", "unexpected"}:
            findings.append(classified)
    return {
        "check": outcome,
        "passed": outcome["ok"] and not findings,
        "findings": findings,
        "allowed": sorted(set(allowed)),
        "root": str(directory),
    }


def bv_verify_platform_support(
    platform_name: object,
    arch: object,
    *,
    supported_platforms: Sequence[str],
    supported_architectures: Sequence[str],
    strict: bool = True,
) -> dict[str, Any]:
    """Normalize a target tuple and optionally reject unsupported values."""
    name = _text(platform_name, "platform_name").strip().lower()
    architecture = _text(arch, "arch").strip().lower()
    platforms = tuple(
        _text(item, "platform").strip().lower()
        for item in _sequence(supported_platforms, "supported_platforms")
    )
    architectures = tuple(
        _text(item, "architecture").strip().lower()
        for item in _sequence(supported_architectures, "supported_architectures")
    )
    reasons: list[str] = []
    if name not in platforms:
        reasons.append(f"unsupported platform: {name}")
    if architecture not in architectures:
        reasons.append(f"unsupported architecture: {architecture}")
    if reasons and strict:
        _fail("E007", "; ".join(reasons))
    return {
        "platform": name,
        "arch": architecture,
        "supported": not reasons,
        "reasons": reasons,
        "strict": strict,
    }


def bv_verify_pytest(
    root: object,
    *,
    runner: object,
    suites: object,
    python_executable: object = "python",
    env: object | None = None,
) -> dict[str, Any]:
    """Run planned pytest suites through an injected runner."""
    plan = bv_plan_pytest_suites(
        root, suites=suites, python_executable=python_executable
    )
    directory = _cwd(root, must_exist=True)
    prepared = {} if env is None else _env(env)
    checks = [
        _execute(runner, directory, item["args"], prepared, item["name"])
        for item in plan["checks"]
    ]
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "checks": checks,
        "passed": not failed,
        "failed_names": failed,
        "suite_count": len(checks),
        "root": str(directory),
    }


def bv_verify_ruff(
    root: object,
    *,
    runner: object,
    ruff_executable: object = "ruff",
    env: object | None = None,
) -> dict[str, Any]:
    """Run planned Ruff format and lint checks through an injected runner."""
    plan = bv_plan_ruff_checks(root, ruff_executable=ruff_executable)
    directory = _cwd(root, must_exist=True)
    prepared = {} if env is None else _env(env)
    checks = [
        _execute(runner, directory, item["args"], prepared, item["name"])
        for item in plan["checks"]
    ]
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "checks": checks,
        "passed": not failed,
        "failed_names": failed,
        "root": str(directory),
    }


def bv_verify_snapshot(
    snapshot: object, *, required_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """Validate that a snapshot mapping carries all caller-required keys."""
    if not isinstance(snapshot, Mapping):
        _fail("E003", "snapshot must be a mapping")
    required = tuple(
        _text(item, "key") for item in _sequence(required_keys, "required_keys")
    )
    missing = tuple(key for key in required if key not in snapshot)
    if missing:
        _fail("E006", f"snapshot is missing keys: {', '.join(missing)}")
    keys = tuple(sorted(str(key) for key in snapshot))
    return {
        "valid": True,
        "keys": list(keys),
        "count": len(keys),
        "digest": _digest(keys),
    }


def bv_verify_tool_availability(
    tools: object, *, required: Sequence[str] = ()
) -> dict[str, Any]:
    """Verify a caller-provided tool-to-path mapping without probing the PATH."""
    if not isinstance(tools, Mapping):
        _fail("E003", "tools must be a mapping")
    prepared: dict[str, str] = {}
    for name, value in tools.items():
        key = _text(name, "tool name")
        if value is None:
            prepared[key] = ""
        else:
            prepared[key] = _text(value, f"tools[{key!r}]", allow_empty=True)
    required_names = tuple(
        _text(item, "required tool") for item in _sequence(required, "required")
    )
    missing = tuple(name for name in required_names if not prepared.get(name))
    available = tuple(sorted(name for name, value in prepared.items() if value))
    return {
        "passed": not missing,
        "available": list(available),
        "missing": list(missing),
        "checked": sorted(prepared),
    }
