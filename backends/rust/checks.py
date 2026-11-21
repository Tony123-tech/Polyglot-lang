"""Model-based Cargo checks for the Rust/PyO3 rollout.

No function in this module starts a real Cargo process.  A deterministic model
runner is used unless the caller explicitly injects one, and every runner still
receives an isolated ``CARGO_TARGET_DIR``.  Results are structured records with
stable diagnostics, fingerprints, and E001-E008 error codes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from compiler.frontend.errors import PolyglotError

__all__ = [
    "rsk_cargo_lock_check",
    "rsk_check_crate",
    "rsk_check_fingerprint",
    "rsk_check_report",
    "rsk_check_step",
    "rsk_command_plan",
    "rsk_deny_warnings",
    "rsk_diagnostic_count",
    "rsk_env_plan",
    "rsk_failure_classify",
    "rsk_feature_matrix",
    "rsk_isolation_check",
    "rsk_manifest_check",
    "rsk_model_runner",
    "rsk_no_source_mutation",
    "rsk_parse_diagnostics",
    "rsk_replay_log",
    "rsk_scaffold_check",
    "rsk_target_dirs",
    "rsk_toolchain_probe",
]

Runner = Callable[[tuple[str, ...], Path, dict[str, str]], Any]
_EDITIONS = ("2018", "2021", "2024")
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}\Z")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> NoReturn:
    """Raise a structured native-check diagnostic."""
    raise PolyglotError(code, message, line=1, col=1, phase="rust-checks")


def _text(value: object, what: str) -> str:
    """Validate one caller-supplied string."""
    if not isinstance(value, str):
        _fail("E003", f"{what} must be a str, got {type(value).__name__}")
    return value


def _path(value: object, what: str) -> Path:
    """Validate a str/Path filesystem location without requiring existence."""
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    _fail("E003", f"{what} must be a str or Path")


def _command(value: Sequence[str], what: str = "command") -> tuple[str, ...]:
    """Validate a command vector."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("E003", f"{what} must be a sequence of strings")
    result = tuple(_text(item, f"{what} argument") for item in value)
    if not result:
        _fail("E003", f"{what} must not be empty")
    return result


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    """Validate a serialized mapping, reporting malformed state as E008."""
    if not isinstance(value, Mapping):
        _fail("E008", f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _fingerprint(files: Mapping[str, str]) -> str:
    """Hash a file mapping in path-sorted canonical form."""
    canonical = "\n".join(
        f"{path}\0{_text(text, f'contents of {path}')}"
        for path, text in sorted(files.items())
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _diagnostics(stdout: str, stderr: str) -> tuple[str, ...]:
    """Extract unique ``error``/``warning`` lines from tool output."""
    found: list[str] = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            stripped = line.strip()
            if stripped.startswith(("error", "warning")) and stripped not in found:
                found.append(stripped)
    return tuple(found)


def _normalize_outcome(result: Any) -> tuple[int, str, str]:
    """Accept a model dict or a caller-supplied CompletedProcess-like object."""
    if isinstance(result, Mapping):
        payload = _mapping(result, "runner result")
        try:
            returncode = int(payload["returncode"])
        except KeyError, TypeError, ValueError:
            _fail("E008", "runner result has no integer 'returncode'")
        return (
            returncode,
            _text(payload.get("stdout", ""), "runner stdout"),
            _text(payload.get("stderr", ""), "runner stderr"),
        )
    try:
        return (
            int(result.returncode),
            _text(result.stdout or "", "runner stdout"),
            _text(result.stderr or "", "runner stderr"),
        )
    except AttributeError, TypeError, ValueError:
        _fail("E008", "runner result must be a mapping or CompletedProcess-like object")


def _selected_runner(runner: Runner | None) -> Runner:
    """Return the injected runner or the deterministic model runner."""
    if runner is None:
        return rsk_model_runner
    if not callable(runner):
        _fail("E003", "runner must be callable or None")
    return runner


# ---------------------------------------------------------------------------
# 1. rsk_model_runner
# ---------------------------------------------------------------------------


def rsk_model_runner(
    command: Sequence[str],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return deterministic Cargo-like output without launching a process."""
    argv = _command(command)
    tool = argv[0]
    if tool not in {"cargo", "rustc", "clippy-driver"}:
        _fail("E004", f"toolchain command {tool!r} is not available in the model")
    environment = dict(env or {})
    target = environment.get("CARGO_TARGET_DIR")
    if "--target-dir" in argv:
        position = argv.index("--target-dir")
        if position + 1 >= len(argv):
            _fail("E003", "--target-dir is missing a value")
        argv_target = argv[position + 1]
        if target is not None and target != argv_target:
            _fail("E003", "CARGO_TARGET_DIR disagrees with --target-dir")
        target = argv_target
    if not target:
        _fail("E003", "model runner requires an isolated CARGO_TARGET_DIR")

    subcommand = argv[1] if len(argv) > 1 else ""
    returncode = 0
    if subcommand in {"--version", "-V", "version"}:
        stdout = f"{tool} 1.90.0 (deterministic-model)\n"
        stderr = ""
    elif tool == "rustc" and subcommand == "":
        stdout = "rustc 1.90.0 (deterministic-model)\n"
        stderr = ""
    elif subcommand in {"check", "clippy", "test", "build"}:
        stdout = (
            f"    Checking polyglot_model v0.1.0 ({subcommand})\n"
            "    Finished dev profile in 0.01s (model)\n"
        )
        stderr = ""
    else:
        returncode = 2
        stdout = ""
        stderr = f"unsupported model subcommand {subcommand!r}\nerror: model failure\n"
    return {
        "command": argv,
        "cwd": "" if cwd is None else str(cwd),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "diagnostics": _diagnostics(stdout, stderr),
        "target_dir": target,
        "model": True,
    }


# ---------------------------------------------------------------------------
# 2. rsk_env_plan
# ---------------------------------------------------------------------------


def rsk_env_plan(target_dir: str | Path) -> dict[str, Any]:
    """Build the minimal deterministic environment for an isolated Cargo run."""
    directory = _path(target_dir, "target directory")
    text = str(directory)
    if not text or text == ".":
        _fail("E003", "target directory must not be empty or the current directory")
    return {
        "variables": {
            "CARGO_TARGET_DIR": text,
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1",
        },
        "target_dir": text,
        "inherits_os_environment": False,
    }


# ---------------------------------------------------------------------------
# 3. rsk_target_dirs
# ---------------------------------------------------------------------------


def rsk_target_dirs(base: str | Path) -> dict[str, Any]:
    """Return isolated target directories for check and clippy steps."""
    directory = _path(base, "base target directory")
    check = directory / "check"
    clippy = directory / "clippy"
    return {
        "base": str(directory),
        "check": str(check),
        "clippy": str(clippy),
        "isolated": check != clippy,
        "count": 2,
    }


# ---------------------------------------------------------------------------
# 4. rsk_command_plan
# ---------------------------------------------------------------------------


def rsk_command_plan(
    crate_root: str | Path,
    *,
    target_dir: str | Path | None = None,
    clippy: bool = True,
    deny_warnings: bool = False,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Construct check/clippy argv vectors with isolated target directories."""
    root = _path(crate_root, "crate root")
    extras = _command(extra_args, "extra_args") if extra_args else ()
    if "--target-dir" in extras:
        _fail("E003", "extra_args may not override --target-dir")
    base = (
        _path(target_dir, "target directory")
        if target_dir is not None
        else root / "target" / "f5-check"
    )
    dirs = rsk_target_dirs(base)
    check_command = (
        "cargo",
        "check",
        "--all-targets",
        "--target-dir",
        dirs["check"],
        *extras,
    )
    clippy_command = (
        "cargo",
        "clippy",
        "--all-targets",
        "--target-dir",
        dirs["clippy"],
        *extras,
        *(("--", "-D", "warnings") if deny_warnings else ()),
    )
    check_env = rsk_env_plan(dirs["check"])["variables"]
    clippy_env = rsk_env_plan(dirs["clippy"])["variables"]
    return {
        "crate_root": str(root),
        "base": dirs["base"],
        "targets": dirs,
        "check_command": check_command,
        "clippy_command": clippy_command,
        "check_env": check_env,
        "clippy_env": clippy_env,
        "clippy": clippy,
        "deny_warnings": deny_warnings,
        "extra_args": extras,
    }


# ---------------------------------------------------------------------------
# 5. rsk_parse_diagnostics
# ---------------------------------------------------------------------------


def rsk_parse_diagnostics(stdout: str, stderr: str = "") -> dict[str, Any]:
    """Parse tool output into unique diagnostic lines and severity counts."""
    out = _text(stdout, "stdout")
    err = _text(stderr, "stderr")
    entries = _diagnostics(out, err)
    return {
        "diagnostics": entries,
        "errors": sum(1 for item in entries if item.startswith("error")),
        "warnings": sum(1 for item in entries if item.startswith("warning")),
        "lines_scanned": len(out.splitlines()) + len(err.splitlines()),
    }


# ---------------------------------------------------------------------------
# 6. rsk_diagnostic_count
# ---------------------------------------------------------------------------


def rsk_diagnostic_count(diagnostics: Sequence[str]) -> dict[str, Any]:
    """Count diagnostics by severity and structured E-code when present."""
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        _fail("E003", "diagnostics must be a sequence of strings")
    entries = tuple(_text(item, "diagnostic") for item in diagnostics)
    codes: dict[str, int] = {}
    for entry in entries:
        match = re.search(r"\b(E00[1-8])\b", entry)
        if match is not None:
            code = match.group(1)
            codes[code] = codes.get(code, 0) + 1
    return {
        "errors": sum(1 for item in entries if item.startswith("error")),
        "warnings": sum(1 for item in entries if item.startswith("warning")),
        "total": len(entries),
        "by_code": {code: codes[code] for code in sorted(codes)},
    }


# ---------------------------------------------------------------------------
# 7. rsk_check_step
# ---------------------------------------------------------------------------


def rsk_check_step(
    name: str,
    command: Sequence[str],
    cwd: str | Path,
    env: Mapping[str, str],
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run one check step through the injected runner or the model runner."""
    step_name = _text(name, "step name")
    if not step_name.strip():
        _fail("E003", "step name must not be empty")
    argv = _command(command)
    directory = _path(cwd, "working directory")
    variables = dict(_mapping(env, "environment"))
    for key, value in variables.items():
        _text(key, "environment key")
        _text(value, f"environment value for {key}")
    selected = _selected_runner(runner)
    try:
        outcome = selected(argv, directory, variables)
    except PolyglotError:
        raise
    except Exception as error:
        _fail("E004", f"check runner {step_name!r} failed: {error}")
    returncode, stdout, stderr = _normalize_outcome(outcome)
    parsed = rsk_parse_diagnostics(stdout, stderr)
    return {
        "name": step_name,
        "command": argv,
        "cwd": str(directory),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "diagnostics": parsed["diagnostics"],
        "ok": returncode == 0,
        "target_dir": variables.get("CARGO_TARGET_DIR", ""),
    }


# ---------------------------------------------------------------------------
# 8. rsk_check_crate
# ---------------------------------------------------------------------------


def rsk_check_crate(
    crate_root: str | Path,
    *,
    target_dir: str | Path | None = None,
    clippy: bool = True,
    deny_warnings: bool = False,
    extra_args: Sequence[str] = (),
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run modeled check and optional clippy steps against an isolated crate."""
    root = _path(crate_root, "crate root")
    if not (root / "Cargo.toml").is_file():
        _fail("E006", f"no Cargo.toml found in crate root {root}")
    plan = rsk_command_plan(
        root,
        target_dir=target_dir,
        clippy=clippy,
        deny_warnings=deny_warnings,
        extra_args=extra_args,
    )
    check = rsk_check_step(
        "check",
        plan["check_command"],
        root,
        plan["check_env"],
        runner=runner,
    )
    steps = [check]
    if clippy:
        steps.append(
            rsk_check_step(
                "clippy",
                plan["clippy_command"],
                root,
                plan["clippy_env"],
                runner=runner,
            )
        )
    return {
        "ok": all(step["ok"] for step in steps),
        "crate_root": str(root),
        "target_dir": plan["base"],
        "steps": tuple(steps),
        "step_names": tuple(step["name"] for step in steps),
        "commands": {step["name"]: step["command"] for step in steps},
    }


# ---------------------------------------------------------------------------
# 9. rsk_toolchain_probe
# ---------------------------------------------------------------------------


def rsk_toolchain_probe(
    *,
    target_dir: str | Path = "target/f5-probe",
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Probe the toolchain version through the model or injected runner."""
    env = rsk_env_plan(Path(target_dir) / "probe")["variables"]
    step = rsk_check_step(
        "probe", ("cargo", "--version"), Path("."), env, runner=runner
    )
    match = re.search(r"cargo\s+([0-9]+\.[0-9]+\.[0-9]+)", step["stdout"])
    if not step["ok"] or match is None:
        _fail("E004", "cargo toolchain probe produced no version")
    return {
        "tool": "cargo",
        "version": match.group(1),
        "ok": True,
        "stdout": step["stdout"],
        "target_dir": step["target_dir"],
    }


# ---------------------------------------------------------------------------
# 10. rsk_feature_matrix
# ---------------------------------------------------------------------------


def rsk_feature_matrix(
    features: Sequence[str],
    *,
    target_dir: str | Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Check each requested Cargo feature in its own model target directory."""
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        _fail("E003", "features must be a sequence of strings")
    selected = tuple(_text(item, "feature name") for item in features)
    if not selected:
        _fail("E003", "feature matrix needs at least one feature")
    if len(set(selected)) != len(selected):
        _fail("E003", "duplicate Cargo features in matrix")
    base = _path(target_dir, "feature target directory")
    runs: list[dict[str, Any]] = []
    for feature in selected:
        directory = base / feature
        env = rsk_env_plan(directory)["variables"]
        command = (
            "cargo",
            "check",
            "--all-targets",
            "--features",
            feature,
            "--target-dir",
            str(directory),
        )
        step = rsk_check_step(feature, command, base, env, runner=runner)
        runs.append(
            {
                "feature": feature,
                "ok": step["ok"],
                "returncode": step["returncode"],
                "target_dir": step["target_dir"],
            }
        )
    return {
        "features": selected,
        "runs": tuple(runs),
        "ok": all(item["ok"] for item in runs),
        "target_dir": str(base),
        "count": len(runs),
    }


# ---------------------------------------------------------------------------
# 11. rsk_no_source_mutation
# ---------------------------------------------------------------------------


def rsk_no_source_mutation(
    files: Mapping[str, str],
    *,
    after: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify that a modeled check left the source file mapping unchanged."""
    before_map = _mapping(files, "source files before check")
    after_map = (
        _mapping(after, "source files after check") if after is not None else before_map
    )
    for label, mapping in (("before", before_map), ("after", after_map)):
        for path, text in mapping.items():
            _text(path, f"{label} path")
            _text(text, f"{label} contents of {path}")
    changed = tuple(
        sorted(
            path
            for path in set(before_map) | set(after_map)
            if before_map.get(path) != after_map.get(path)
        )
    )
    return {
        "mutated": bool(changed),
        "changed": changed,
        "before": _fingerprint(before_map),
        "after": _fingerprint(after_map),
        "file_count": len(before_map),
    }


# ---------------------------------------------------------------------------
# 12. rsk_cargo_lock_check
# ---------------------------------------------------------------------------


def rsk_cargo_lock_check(
    crate_root: str | Path,
    *,
    lock_text: str | None = None,
) -> dict[str, Any]:
    """Validate a Cargo.lock snapshot without modifying it."""
    root = _path(crate_root, "crate root")
    if lock_text is None:
        lock_path = root / "Cargo.lock"
        if not lock_path.is_file():
            _fail("E006", f"no Cargo.lock found in {root}")
        text = lock_path.read_text(encoding="utf-8")
    else:
        text = _text(lock_text, "Cargo.lock text")
    version_match = re.search(r'^version\s*=\s*"([0-9]+)"', text, re.MULTILINE)
    if version_match is None:
        _fail("E008", "Cargo.lock has no lockfile version")
    packages = text.count("[[package]]")
    if packages < 1:
        _fail("E008", "Cargo.lock has no [[package]] entries")
    return {
        "present": True,
        "version": int(version_match.group(1)),
        "packages": packages,
        "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line_count": len(text.splitlines()),
    }


# ---------------------------------------------------------------------------
# 13. rsk_manifest_check
# ---------------------------------------------------------------------------


def rsk_manifest_check(text: str) -> dict[str, Any]:
    """Parse and validate the key fields of a Cargo manifest snapshot."""
    content = _text(text, "Cargo.toml text")
    sections: list[str] = []
    values: dict[str, str] = {}
    section = ""
    for line in content.splitlines():
        stripped = line.strip()
        header = re.fullmatch(r"\[([A-Za-z0-9_.-]+)\]", stripped)
        if header is not None:
            section = header.group(1)
            sections.append(section)
            continue
        entry = re.fullmatch(r'([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"', stripped)
        if entry is not None and section == "package":
            values[entry.group(1)] = entry.group(2)
    for key in ("name", "version", "edition"):
        if key not in values:
            _fail("E008", f"Cargo.toml [package] has no {key!r} field")
    if values["edition"] not in _EDITIONS:
        _fail("E001", f"unsupported Rust edition {values['edition']!r}")
    if _VERSION_RE.fullmatch(values["version"]) is None:
        _fail("E008", f"malformed crate version {values['version']!r}")
    return {
        "name": values["name"],
        "version": values["version"],
        "edition": values["edition"],
        "sections": tuple(sections),
        "field_count": len(values),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 14. rsk_scaffold_check
# ---------------------------------------------------------------------------


def rsk_scaffold_check(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    *,
    ignore_extra: bool = False,
) -> dict[str, Any]:
    """Compare generated scaffold text against a serialized snapshot."""
    wanted = _mapping(expected, "expected scaffold")
    found = _mapping(actual, "actual scaffold")
    for label, mapping in (("expected", wanted), ("actual", found)):
        for path, text in mapping.items():
            _text(path, f"{label} path")
            _text(text, f"{label} contents of {path}")
    wanted_paths = set(wanted)
    found_paths = set(found)
    missing = tuple(sorted(wanted_paths - found_paths))
    mismatched = tuple(
        sorted(
            path for path in wanted_paths & found_paths if wanted[path] != found[path]
        )
    )
    extra = tuple(sorted(found_paths - wanted_paths))
    ok = not missing and not mismatched and (ignore_extra or not extra)
    return {
        "ok": ok,
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
        "compared": len(wanted_paths & found_paths),
        "ignore_extra": ignore_extra,
    }


# ---------------------------------------------------------------------------
# 15. rsk_deny_warnings
# ---------------------------------------------------------------------------


def rsk_deny_warnings(
    command: Sequence[str], *, enabled: bool = True
) -> dict[str, Any]:
    """Append clippy's ``-D warnings`` flags without mutating caller input."""
    argv = _command(command)
    changed = enabled and "clippy" in argv and "--" not in argv
    result = argv + (("--", "-D", "warnings") if changed else ())
    return {
        "command": result,
        "source_command": argv,
        "deny_warnings": changed,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# 16. rsk_isolation_check
# ---------------------------------------------------------------------------


def rsk_isolation_check(
    command: Sequence[str], env: Mapping[str, str]
) -> dict[str, Any]:
    """Verify that argv and environment agree on one isolated target dir."""
    argv = _command(command)
    variables = dict(_mapping(env, "environment"))
    target = variables.get("CARGO_TARGET_DIR")
    if not target:
        _fail("E003", "environment has no CARGO_TARGET_DIR")
    count = argv.count("--target-dir")
    if count > 1:
        _fail("E003", "--target-dir appears more than once")
    argv_target = None
    if count == 1:
        position = argv.index("--target-dir")
        if position + 1 >= len(argv):
            _fail("E003", "--target-dir has no value")
        argv_target = argv[position + 1]
        if argv_target != target:
            _fail("E003", f"--target-dir {argv_target!r} disagrees with env {target!r}")
    return {
        "ok": True,
        "target_dir": target,
        "argv_target": argv_target,
        "command": argv,
        "isolated": True,
    }


# ---------------------------------------------------------------------------
# 17. rsk_failure_classify
# ---------------------------------------------------------------------------


def rsk_failure_classify(step: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a failed Cargo step as E004 with a concise summary."""
    payload = _mapping(step, "check step")
    name = _text(payload.get("name", ""), "step name")
    try:
        returncode = int(payload.get("returncode", 0))
    except TypeError, ValueError:
        _fail("E008", "check step has a malformed returncode")
    stderr = _text(payload.get("stderr", ""), "step stderr")
    if returncode == 0:
        return {
            "ok": True,
            "code": None,
            "name": name,
            "summary": f"{name} succeeded",
        }
    first = next((line.strip() for line in stderr.splitlines() if line.strip()), "")
    summary = first or f"{name} exited with status {returncode}"
    _fail("E004", f"{name} failed: {summary}")


# ---------------------------------------------------------------------------
# 18. rsk_check_report
# ---------------------------------------------------------------------------


def rsk_check_report(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate structured check steps into one deterministic report."""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        _fail("E008", "steps must be a sequence of mappings")
    names: list[str] = []
    failed: list[str] = []
    diagnostics: list[str] = []
    for index, item in enumerate(steps):
        payload = _mapping(item, f"step {index}")
        name = _text(payload.get("name", ""), f"step {index} name")
        if not name:
            _fail("E008", f"step {index} has no name")
        if not isinstance(payload.get("ok"), bool):
            _fail("E008", f"step {name!r} has no boolean 'ok'")
        names.append(name)
        if not payload["ok"]:
            failed.append(name)
        entries = payload.get("diagnostics", ())
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            _fail("E008", f"step {name!r} has malformed diagnostics")
        diagnostics.extend(_text(entry, f"diagnostic of {name}") for entry in entries)
    return {
        "ok": not failed,
        "total": len(names),
        "names": tuple(names),
        "failed": tuple(failed),
        "diagnostics": tuple(diagnostics),
    }


# ---------------------------------------------------------------------------
# 19. rsk_check_fingerprint
# ---------------------------------------------------------------------------


def rsk_check_fingerprint(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash a canonical serialization of structured check steps."""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        _fail("E008", "steps must be a sequence of mappings")
    payload = [_mapping(item, "check step") for item in steps]
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        _fail("E008", f"check steps are not serializable: {error}")
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "canonical": canonical,
        "step_count": len(payload),
    }


# ---------------------------------------------------------------------------
# 20. rsk_replay_log
# ---------------------------------------------------------------------------


def rsk_replay_log(
    log: Sequence[Mapping[str, Any]],
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Replay recorded check steps through the model runner deterministically."""
    if not isinstance(log, Sequence) or isinstance(log, (str, bytes)):
        _fail("E008", "replay log must be a sequence of mappings")
    selected = _selected_runner(runner)
    matches: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for index, entry in enumerate(log):
        payload = _mapping(entry, f"replay entry {index}")
        name = _text(payload.get("name", f"step-{index}"), f"replay step {index} name")
        command = _command(payload.get("command"), f"replay step {name} command")
        env = _mapping(payload.get("env", {}), f"replay step {name} env")
        variables = {str(key): _text(value, "env value") for key, value in env.items()}
        step = rsk_check_step(name, command, Path("."), variables, runner=selected)
        expected = payload.get("expected")
        if expected is None:
            matches.append(name)
            continue
        expected_map = _mapping(expected, f"replay step {name} expected result")
        compared = ("returncode", "stdout", "stderr")
        differences = tuple(
            key
            for key in compared
            if key in expected_map and step[key] != expected_map[key]
        )
        if differences:
            mismatches.append({"name": name, "fields": differences})
        else:
            matches.append(name)
    return {
        "ok": not mismatches,
        "replayed": len(matches) + len(mismatches),
        "matches": tuple(matches),
        "mismatches": tuple(mismatches),
        "model_only": runner is None,
    }
