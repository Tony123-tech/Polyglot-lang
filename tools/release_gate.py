"""Single ordered release gate for the polyglot-lang repository.

The gate executes seven stages in a fixed order:

1. ruff format check + ruff lint;
2. in-process scaffold drift verification;
3. core pytest parity suites (plus a full ``pytest -q`` in ``--full`` mode);
4. bounded, deterministic differential-fuzz runs;
5. cargo check + clippy with an isolated ``CARGO_TARGET_DIR``;
6. docs sync + docs lint;
7. git worktree cleanliness.

Every command-running stage accepts an injected runner so unit tests never
spawn real pytest/cargo/ruff/git processes.  Fuzzing is injected separately as
``fuzz_runner(seed, case_count) -> dict``; the real helper only shells out when
``tools/diff_fuzz.py`` is present.  A missing fuzzer is a hard blocker, while
cargo may be reported as SKIP only with the explicit ``--allow-missing-cargo``
flag.  Artifacts default to the gitignored ``.artifacts/release_gate`` path and
are created only when the real fuzzer runs; build caches live under the system
temp directory, never inside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Allow both ``python tools/release_gate.py`` and ``python -m tools.release_gate``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.rust.framework import (
    ScaffoldVerification,
    rs_generate_pyo3_scaffold,
    rs_verify_scaffold_matches_generator,
)
from tools.build_ci.framework import (
    CheckResult,
    CommandRunner,
    ReleaseGateReport,
    ci_compute_cache_scope_key,
    ci_generate_release_gate_report,
    ci_isolate_cargo_target_dir,
    ci_verify_clean_git_worktree,
    ci_verify_docs_sync_and_lint,
    ci_verify_pytest_parity_suites,
    ci_verify_ruff_format_and_lint,
    ci_verify_rust_checks,
)

FuzzReport = dict[str, Any]
FuzzRunner = Callable[[int, int], FuzzReport]
FuzzPayload = FuzzReport | object
Mode = Literal["default", "quick", "full"]
Status = Literal["PASS", "FAIL", "SKIP"]

FUZZ_SCRIPT = "tools/diff_fuzz.py"
DEFAULT_ARTIFACTS = ".artifacts/release_gate"
FUZZ_CONFIG: dict[str, tuple[int, int]] = {
    "default": (1, 8),
    "quick": (3, 6),
    "full": (12, 24),
}
FUZZ_BASE_SEED = 20261120
FUZZ_SEED_STRIDE = 7919
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

SCAFFOLD_CHECK_NAME = "scaffold-drift"
SCAFFOLD_COMMAND: tuple[str, ...] = ("in-process:rs_verify_scaffold_matches_generator",)
GATE_ORDER: tuple[str, ...] = (
    "ruff",
    "scaffold",
    "pytest-core",
    "diff-fuzz",
    "cargo",
    "docs",
    "git-worktree",
)


@dataclass(frozen=True)
class GateOutcome:
    """One line of the human-readable per-check table."""

    name: str
    status: Status
    command: tuple[str, ...]
    detail: str = ""


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
    runner: CommandRunner,
) -> CheckResult:
    process = runner(args, root)
    return CheckResult(
        name=name,
        command=args,
        returncode=int(process.returncode),
        stdout=process.stdout if isinstance(process.stdout, str) else "",
        stderr=process.stderr if isinstance(process.stderr, str) else "",
    )


def _check_outcome(check: CheckResult) -> GateOutcome:
    return GateOutcome(
        name=check.name,
        status="PASS" if check.ok else "FAIL",
        command=check.command,
        detail=f"exit {check.returncode}",
    )


def _fuzz_seeds(count: int) -> tuple[int, ...]:
    return tuple(FUZZ_BASE_SEED + index * FUZZ_SEED_STRIDE for index in range(count))


def _fuzz_plan(mode: str) -> tuple[tuple[int, ...], int]:
    try:
        count, case_count = FUZZ_CONFIG[mode]
    except KeyError:
        raise ValueError(f"unknown gate mode: {mode!r}") from None
    return _fuzz_seeds(count), case_count


def _resolve_artifacts(root: Path, artifacts_dir: str | Path | None) -> Path:
    if artifacts_dir is None:
        return root / DEFAULT_ARTIFACTS
    path = Path(artifacts_dir).expanduser()
    return path if path.is_absolute() else root / path


def _allowed_artifact_paths(root: Path, artifacts: Path) -> tuple[str, ...]:
    allowed = [".artifacts", DEFAULT_ARTIFACTS]
    try:
        relative = artifacts.resolve().relative_to(root).as_posix()
    except ValueError, OSError:
        return tuple(allowed)
    if relative not in allowed:
        allowed.append(relative)
    first_part = relative.split("/", 1)[0]
    if first_part not in allowed:
        allowed.append(first_part)
    return tuple(allowed)


def _scaffold_check(verification: ScaffoldVerification) -> CheckResult:
    """Translate a structured drift report into a gate ``CheckResult``."""
    details = tuple(line for line in verification.details if line.strip())
    if verification.ok:
        return CheckResult(
            name=SCAFFOLD_CHECK_NAME,
            command=SCAFFOLD_COMMAND,
            returncode=0,
            stdout="; ".join(details) or "scaffold matches generator",
        )
    summary = "; ".join(details) or "scaffold drift detected"
    mismatched = ", ".join(verification.mismatched) or "-"
    missing = ", ".join(verification.missing) or "-"
    return CheckResult(
        name=SCAFFOLD_CHECK_NAME,
        command=SCAFFOLD_COMMAND,
        returncode=1,
        stdout=f"mismatched={mismatched} missing={missing}",
        stderr=summary,
    )


def _verify_generated_scaffold(cache_root: Path, scope_key: str) -> CheckResult:
    """Round-trip the PyO3 scaffold generator through its drift verifier.

    The repository's opt-in ``native_pillars`` crate is rustfmt-formatted, so an
    exact text comparison against the raw generator output would always report
    drift.  The gate instead materializes a fresh copy below the isolated cache
    root and verifies that copy with the same verifier, which keeps the gate
    deterministic and leaves the working tree untouched.
    """
    scaffold_root = cache_root / "scaffold" / scope_key
    generated = rs_generate_pyo3_scaffold()
    for relative, text in generated.items():
        target = scaffold_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    verification = rs_verify_scaffold_matches_generator(
        scaffold_root, generated=generated
    )
    return _scaffold_check(verification)


def _fuzz_command(
    python: str, seed: int, case_count: int, artifacts: Path
) -> tuple[str, ...]:
    return (
        python,
        FUZZ_SCRIPT,
        "--seed",
        str(seed),
        "--case-count",
        str(case_count),
        "--artifacts",
        str(artifacts),
        "--json",
    )


def _fuzz_check(
    name: str,
    command: tuple[str, ...],
    seed: int,
    case_count: int,
    payload: FuzzPayload,
) -> CheckResult:
    """Synthesize a ``CheckResult`` from a fuzzer JSON report dict."""
    if not isinstance(payload, Mapping):
        return CheckResult(
            name=name,
            command=command,
            returncode=2,
            stderr=f"fuzz runner returned {type(payload).__name__}, expected dict",
        )
    report: FuzzReport = dict(payload)
    harness_error = report.get("harness_error") or report.get("error")
    if harness_error:
        return CheckResult(
            name=name,
            command=command,
            returncode=2,
            stderr=str(harness_error),
        )
    mismatches = report.get("mismatches") or ()
    cases = report.get("cases", case_count)
    skipped = report.get("skipped", 0)
    if not report.get("passed") or mismatches:
        return CheckResult(
            name=name,
            command=command,
            returncode=1,
            stderr=(
                f"seed={seed} cases={cases} skipped={skipped} "
                f"mismatches={len(mismatches) if hasattr(mismatches, '__len__') else mismatches}"
            ),
        )
    return CheckResult(
        name=name,
        command=command,
        returncode=0,
        stdout=f"seed={seed} cases={cases} skipped={skipped}",
    )


def _run_fuzz_subprocess(
    seed: int,
    case_count: int,
    *,
    root: Path,
    python: str,
    artifacts: Path,
) -> FuzzReport:
    """Run one differential-fuzz invocation and return its JSON payload."""
    script = root / FUZZ_SCRIPT
    if not script.is_file():
        return {
            "passed": False,
            "cases": 0,
            "mismatches": [],
            "skipped": 0,
            "seed": seed,
            "harness_error": f"fuzzer script not found: {script}",
        }
    artifacts.mkdir(parents=True, exist_ok=True)
    command = _fuzz_command(python, seed, case_count, artifacts)
    process = subprocess.run(
        list(command),
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(process.stdout)
    except TypeError, ValueError:
        detail = process.stderr.strip() or process.stdout.strip() or "no output"
        return {
            "passed": False,
            "cases": 0,
            "mismatches": [],
            "skipped": 0,
            "seed": seed,
            "harness_error": f"invalid fuzzer JSON (exit {process.returncode}): {detail}",
        }
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "cases": 0,
            "mismatches": [],
            "skipped": 0,
            "seed": seed,
            "harness_error": "fuzzer JSON payload is not an object",
        }
    report: FuzzReport = dict(payload)
    if process.returncode == 2:
        report["passed"] = False
        report.setdefault(
            "harness_error",
            process.stderr.strip() or "fuzz harness error (exit 2)",
        )
    elif process.returncode not in (0, 1):
        report["passed"] = False
        report.setdefault(
            "harness_error",
            f"fuzzer exited {process.returncode}: {process.stderr.strip()}",
        )
    return report


def _make_fuzz_runner(*, root: Path, python: str, artifacts: Path) -> FuzzRunner:
    def runner(seed: int, case_count: int) -> FuzzReport:
        return _run_fuzz_subprocess(
            seed, case_count, root=root, python=python, artifacts=artifacts
        )

    return runner


def _run_fuzz_stage(
    *,
    root: Path,
    seeds: tuple[int, ...],
    case_count: int,
    python: str,
    artifacts: Path,
    fuzz_runner: FuzzRunner | None,
) -> tuple[tuple[CheckResult, ...], tuple[str, ...]]:
    checks: list[CheckResult] = []
    blockers: list[str] = []
    effective_runner = fuzz_runner
    if effective_runner is None:
        script = root / FUZZ_SCRIPT
        if not script.is_file():
            command = _fuzz_command(python, seeds[0], case_count, artifacts)
            checks.append(
                CheckResult(
                    name="diff-fuzz:missing-script",
                    command=command,
                    returncode=2,
                    stderr=f"fuzzer script not found: {script}",
                )
            )
            blockers.append(f"diff-fuzz: fuzzer script not found: {script}")
            return tuple(checks), tuple(blockers)
        effective_runner = _make_fuzz_runner(
            root=root, python=python, artifacts=artifacts
        )
    for seed in seeds:
        command = _fuzz_command(python, seed, case_count, artifacts)
        payload = effective_runner(seed, case_count)
        checks.append(
            _fuzz_check(
                f"diff-fuzz:seed-{seed}",
                command,
                seed,
                case_count,
                payload,
            )
        )
    return tuple(checks), tuple(blockers)


def _cargo_env_runner(
    runner: CommandRunner, target_dirs: Mapping[str, Path]
) -> CommandRunner:
    """Set ``CARGO_TARGET_DIR`` only around cargo invocations, then restore it."""

    def wrapped(args: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[str]:
        profile = "clippy" if len(args) > 1 and args[1] == "clippy" else "check"
        target = target_dirs[profile]
        saved = dict(os.environ)
        try:
            os.environ["CARGO_TARGET_DIR"] = str(target)
            return runner(args, cwd)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    return wrapped


def _render_check_table(outcomes: Sequence[GateOutcome]) -> str:
    lines = [f"{'CHECK':<34} {'STATUS':<6} COMMAND (DETAIL)"]
    for outcome in outcomes:
        label = outcome.name[:34]
        detail = f"  [{outcome.detail}]" if outcome.detail else ""
        lines.append(
            f"{label:<34} {outcome.status:<6} {shlex.join(outcome.command)}{detail}"
        )
    return "\n".join(lines)


def _run_gate_detailed(
    *,
    root: Path | str,
    mode: str = "quick",
    runner: CommandRunner | None = None,
    fuzz_runner: FuzzRunner | None = None,
    python: str | None = None,
    ruff: str | None = None,
    cargo: str | None = None,
    artifacts_dir: str | Path | None = None,
    allow_dirty: bool = False,
    scope_key: str | None = None,
    cache_root: str | Path | None = None,
    allow_missing_cargo: bool = False,
    cargo_available: bool | None = None,
) -> tuple[ReleaseGateReport, tuple[GateOutcome, ...]]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"root is not a directory: {root_path}")
    if mode not in FUZZ_CONFIG:
        raise ValueError(f"unknown gate mode: {mode!r}")

    command_runner = runner if runner is not None else _default_runner
    python_exe = python or sys.executable
    ruff_exe = ruff or "ruff"
    cargo_exe = cargo or "cargo"
    scope = scope_key or ci_compute_cache_scope_key(extra={"gate": mode})
    if _SAFE_COMPONENT_RE.fullmatch(scope) is None:
        raise ValueError(f"scope_key must be a safe path component: {scope!r}")
    cache = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else Path(tempfile.gettempdir()) / f"polyglot-release-gate-{scope}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    artifacts = _resolve_artifacts(root_path, artifacts_dir)

    checks: dict[str, Sequence[CheckResult] | bool] = {}
    blockers: list[str] = []
    outcomes: list[GateOutcome] = []
    required: list[str] = list(GATE_ORDER)

    # Stage 1: ruff format check and lint.
    ruff_checks = ci_verify_ruff_format_and_lint(
        root=root_path, runner=command_runner, ruff_executable=ruff_exe
    )
    checks["ruff"] = ruff_checks
    outcomes.extend(_check_outcome(check) for check in ruff_checks)

    # Stage 2: in-process scaffold drift verification.
    scaffold_check = _verify_generated_scaffold(cache, scope)
    checks["scaffold"] = (scaffold_check,)
    outcomes.append(_check_outcome(scaffold_check))

    # Stage 3: core pytest parity suites (plus full pytest in --full mode).
    # The cross-backend parity module is marked ``slow`` at module level, so
    # pyproject's default ``-m 'not slow ...'`` would collect zero tests and
    # pytest would exit 5.  Override the marker filter for that suite only.
    def pytest_runner(args: tuple[str, ...], cwd: Path):
        # ``python -m pytest`` always contains ``-m`` for the module flag, so
        # check for the marker value rather than the flag itself.
        if "tests/test_cross_backend_parity.py" in args and "slow" not in args:
            args = (*args, "-m", "slow")
        return command_runner(args, cwd)

    core_checks = ci_verify_pytest_parity_suites(
        root=root_path, runner=pytest_runner, python_executable=python_exe
    )
    checks["pytest-core"] = core_checks
    outcomes.extend(_check_outcome(check) for check in core_checks)
    if mode == "full":
        full_check = _run_check(
            "pytest-full",
            (python_exe, "-m", "pytest", "-q"),
            root_path,
            command_runner,
        )
        checks["pytest-full"] = (full_check,)
        outcomes.append(_check_outcome(full_check))
        required.append("pytest-full")

    # Stage 4: bounded deterministic differential fuzzing.
    seeds, case_count = _fuzz_plan(mode)
    fuzz_checks, fuzz_blockers = _run_fuzz_stage(
        root=root_path,
        seeds=seeds,
        case_count=case_count,
        python=python_exe,
        artifacts=artifacts,
        fuzz_runner=fuzz_runner,
    )
    checks["diff-fuzz"] = fuzz_checks
    outcomes.extend(_check_outcome(check) for check in fuzz_checks)
    blockers.extend(fuzz_blockers)

    # Stage 5: cargo check + clippy with an isolated target directory.
    if cargo_available is False:
        cargo_command: tuple[str, ...] = (cargo_exe, "--version")
        if allow_missing_cargo:
            checks["cargo [SKIP: executable unavailable]"] = True
            outcomes.append(
                GateOutcome(
                    "cargo",
                    "SKIP",
                    cargo_command,
                    "skipped: cargo executable unavailable (--allow-missing-cargo)",
                )
            )
            required.remove("cargo")
        else:
            unavailable = CheckResult(
                name="cargo:unavailable",
                command=cargo_command,
                returncode=127,
                stderr=f"cargo executable not found: {cargo_exe}",
            )
            checks["cargo"] = (unavailable,)
            outcomes.append(_check_outcome(unavailable))
            blockers.append(
                "cargo: executable not found; native checks are mandatory "
                "unless --allow-missing-cargo is supplied"
            )
    else:
        target_dirs = {
            profile: ci_isolate_cargo_target_dir(
                profile, cache_root=cache, scope_key=scope
            )
            for profile in ("check", "clippy")
        }
        # Cargo manifests live in the native_pillars crate, not at the repo
        # root; running there also keeps generated artifacts out of the source
        # tree because CARGO_TARGET_DIR is redirected to the isolated cache.
        rust_root = root_path / "native_pillars"
        if not (rust_root / "Cargo.toml").is_file():
            rust_root = root_path
        cargo_checks = ci_verify_rust_checks(
            root=rust_root,
            runner=_cargo_env_runner(command_runner, target_dirs),
            cargo_executable=cargo_exe,
        )
        checks["cargo"] = cargo_checks
        outcomes.extend(_check_outcome(check) for check in cargo_checks)

    # Stage 6: docs sync and docs lint.
    docs_checks = ci_verify_docs_sync_and_lint(
        root=root_path, runner=command_runner, python_executable=python_exe
    )
    checks["docs"] = docs_checks
    outcomes.extend(_check_outcome(check) for check in docs_checks)

    # Stage 7: git worktree cleanliness (explicit SKIP with --allow-dirty).
    git_command: tuple[str, ...] = ("git", "status", "--short")
    if allow_dirty:
        checks["git-worktree-clean [SKIP: --allow-dirty]"] = True
        outcomes.append(
            GateOutcome(
                "git-worktree-clean",
                "SKIP",
                git_command,
                "skipped via --allow-dirty",
            )
        )
        required.remove("git-worktree")
    else:
        git_check = ci_verify_clean_git_worktree(
            root=root_path,
            runner=command_runner,
            allowed_paths=_allowed_artifact_paths(root_path, artifacts),
            ignore_patterns=(".artifacts/*",),
        )
        checks["git-worktree"] = (git_check,)
        outcomes.append(_check_outcome(git_check))

    report = ci_generate_release_gate_report(
        checks=checks,
        blockers=tuple(blockers),
        required_checks=tuple(required),
    )
    return report, tuple(outcomes)


def run_gate(
    *,
    root: Path | str,
    mode: str = "quick",
    runner: CommandRunner | None = None,
    fuzz_runner: FuzzRunner | None = None,
    python: str | None = None,
    ruff: str | None = None,
    cargo: str | None = None,
    artifacts_dir: str | Path | None = None,
    allow_dirty: bool = False,
    scope_key: str | None = None,
    cache_root: str | Path | None = None,
    allow_missing_cargo: bool = False,
    cargo_available: bool | None = None,
) -> ReleaseGateReport:
    """Run the ordered release gate and return the aggregate report."""
    return _run_gate_detailed(
        root=root,
        mode=mode,
        runner=runner,
        fuzz_runner=fuzz_runner,
        python=python,
        ruff=ruff,
        cargo=cargo,
        artifacts_dir=artifacts_dir,
        allow_dirty=allow_dirty,
        scope_key=scope_key,
        cache_root=cache_root,
        allow_missing_cargo=allow_missing_cargo,
        cargo_available=cargo_available,
    )[0]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 0 when every blocking check passes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", action="store_true", help="3 fuzz seeds x 6 cases (core suites)"
    )
    mode_group.add_argument(
        "--full", action="store_true", help="12 seeds x 24 cases plus full pytest -q"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ruff", default="ruff")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="skip the git worktree check"
    )
    parser.add_argument(
        "--allow-missing-cargo",
        action="store_true",
        help="report cargo as SKIP when the executable is unavailable",
    )
    args = parser.parse_args(argv)

    mode: Mode = "full" if args.full else "quick" if args.quick else "default"
    report, outcomes = _run_gate_detailed(
        root=args.root,
        mode=mode,
        runner=_default_runner,
        fuzz_runner=None,
        python=args.python,
        ruff=args.ruff,
        cargo=args.cargo,
        artifacts_dir=args.artifacts,
        allow_dirty=args.allow_dirty,
        allow_missing_cargo=args.allow_missing_cargo,
        cargo_available=shutil.which(args.cargo) is not None,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.render())
        print()
        print(_render_check_table(outcomes))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
