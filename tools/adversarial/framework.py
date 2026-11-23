"""Frame K — adversarial generation, mutation scoring, and spec oracle.

The generators are deterministic and grammar-bounded: every emitted case is
parsed before it is returned, so adversarial means "interesting and well
formed", not random garbage.  Mutation scoring runs each injected fault in an
isolated symlink workspace under a caller-supplied root and treats a non-zero
test exit as a killed mutant.  The oracle report records exact commands and
outputs plus the limits of what those checks establish.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import PillarError
from tools.diff_fuzz import normalize_error

Span = tuple[int, int] | None
Category = Literal[
    "boundary-int", "deep-nesting", "trait-bound", "match-guard", "error-path"
]
CATEGORIES: tuple[Category, ...] = (
    "boundary-int",
    "deep-nesting",
    "trait-bound",
    "match-guard",
    "error-path",
)
BOUNDARY_VALUES: tuple[str, ...] = (
    "0",
    "1",
    "-1",
    "2147483647",
    "-2147483648",
    "9223372036854775807",
    "-9223372036854775808",
)
DEFAULT_MUTANTS: tuple[Mapping[str, str], ...] = (
    {
        "mutant_id": "formal-proof-gate",
        "path": "compiler/formal/framework.py",
        "original": "    if report.proved:\n        return",
        "replacement": "    if True:\n        return",
        "description": "accept an unproved bounded symbolic report",
    },
    {
        "mutant_id": "termination-variant-gate",
        "path": "compiler/termination/framework.py",
        "original": (
            "    proved = (\n"
            "        bool(resolved) and all(call.decreases for call in calls) and not diagnostics\n"
            "    )"
        ),
        "replacement": "    proved = True",
        "description": "accept recursive calls without decreasing variants",
    },
    {
        "mutant_id": "hard-state-safety-gate",
        "path": "runtime/hard_state/framework.py",
        "original": "    safe = not diagnostics",
        "replacement": "    safe = True",
        "description": "accept lossy schema migrations as safe",
    },
)
VERIFICATION_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff-format", ("ruff", "format", "--check", ".")),
    ("ruff-lint", ("ruff", "check", ".")),
    ("pytest", ("python", "-m", "pytest", "-q")),
    ("docs-sync", ("python", "tools/check_docs_sync.py")),
    ("docs-lint", ("python", "tools/lint_docs.py")),
)
CARGO_COMMAND: tuple[str, ...] = ("cargo", "check", "--workspace", "--all-targets")
SKIP_WORKSPACE_NAMES: frozenset[str] = frozenset(
    {".git", ".artifacts", "target", "__pycache__", ".pytest_cache", ".ruff_cache"}
)


def _fail(
    code: str,
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text=help_text or "inspect the adversarial finding and tighten the guard",
        note="tools.adversarial/frame-k",
        span=span,
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=repr)
        return items
    return repr(value)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Adversarial case generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    category: Category
    source: str
    constraints: tuple[str, ...]
    expectation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "source": self.source,
            "constraints": list(self.constraints),
            "expectation": self.expectation,
        }


@dataclass(frozen=True)
class MutationReport:
    seed: int
    count: int
    max_depth: int
    cases: tuple[AdversarialCase, ...]
    diagnostics: tuple[dict[str, str], ...]

    def digest(self) -> str:
        return _digest(
            {"seed": self.seed, "cases": [item.as_dict() for item in self.cases]}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "count": self.count,
            "max_depth": self.max_depth,
            "cases": [item.as_dict() for item in self.cases],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "digest": self.digest(),
            "caveats": (
                "generation is deterministic and grammar-bounded, not exhaustive",
                "parsing proves syntax only, not type or semantic validity",
            ),
        }


def _boundary_case(rng: random.Random, max_depth: int) -> AdversarialCase:
    value = BOUNDARY_VALUES[rng.randrange(len(BOUNDARY_VALUES))]
    depth = max(1, min(max_depth, rng.randint(1, 4)))
    source = (
        f"fn main(x: i64) -> i64 {{ return {'(' * depth}x + {value}{')' * depth}; }}"
    )
    return AdversarialCase(
        f"boundary-{value}-{depth}",
        "boundary-int",
        source,
        ("i64", f"literal={value}", f"paren-depth={depth}"),
        "compile-time scalar boundary; overflow semantics are backend-specific",
    )


def _nesting_case(rng: random.Random, max_depth: int) -> AdversarialCase:
    depth = max(2, min(max_depth, rng.randint(2, max(2, max_depth))))
    source = f"fn main(x: i64) -> i64 {{ return {'(' * depth}x{')' * depth}; }}"
    return AdversarialCase(
        f"deep-{depth}",
        "deep-nesting",
        source,
        (f"depth={depth}", "balanced-parentheses"),
        "deep expression nesting must not overflow the parser stack",
    )


def _trait_case(rng: random.Random, max_depth: int) -> AdversarialCase:
    bounds = ["Clone", "Debug", "PartialEq", "Send"]
    count = max(1, min(max_depth, rng.randint(1, 3)))
    selected = tuple(rng.sample(bounds, count))
    bound_text = " + ".join(selected)
    source = f"fn map<T: {bound_text}>(x: T) -> T {{ return x; }}"
    return AdversarialCase(
        "trait-" + "-".join(selected),
        "trait-bound",
        source,
        ("generic=T", f"bounds={bound_text}"),
        "higher-rank trait bounds must be accepted without specialization",
    )


def _guard_case(rng: random.Random, max_depth: int) -> AdversarialCase:
    operator = rng.choice((">", ">=", "<", "<="))
    threshold = BOUNDARY_VALUES[rng.randrange(3)]
    source = (
        f"fn main(x: i64) -> i64 {{ "
        f"return match x {{ n if n {operator} {threshold} => 1, _ => 0 }}; }}"
    )
    return AdversarialCase(
        f"guard-{operator}-{threshold}",
        "match-guard",
        source,
        (f"guard=n {operator} {threshold}", "exhaustive-wildcard"),
        "match guard must keep the wildcard arm reachable",
    )


def _error_case(rng: random.Random, max_depth: int) -> AdversarialCase:
    tail = rng.choice(("x / 0", "Err(E::Bad)", "x - 9223372036854775807"))
    source = f"fn main(x: i64) -> i64 {{ return {tail}; }}"
    return AdversarialCase(
        f"error-{tail.replace(' ', '-')}",
        "error-path",
        source,
        ("explicit-error-path", "structured-diagnostic-expected"),
        "error path must fail with a structured diagnostic, not a crash",
    )


_GENERATORS: dict[Category, Callable[[random.Random, int], AdversarialCase]] = {
    "boundary-int": _boundary_case,
    "deep-nesting": _nesting_case,
    "trait-bound": _trait_case,
    "match-guard": _guard_case,
    "error-path": _error_case,
}


def adv_mutate_ast_and_type_constraints(
    *,
    seed: int = 0,
    count: int = 32,
    max_depth: int = 8,
    span: Span = None,
) -> MutationReport:
    """Generate deterministic, parsed, type-aware adversarial cases."""
    if count < 0 or max_depth < 1:
        raise _fail(
            "E003", "count must be non-negative and max_depth positive", span=span
        )
    cases: list[AdversarialCase] = []
    diagnostics: list[dict[str, str]] = []
    seen: set[str] = set()
    for index in range(count):
        category = CATEGORIES[index % len(CATEGORIES)]
        rng = random.Random(f"polyglot-adversarial:{seed}:{index}")
        case = replace(
            _GENERATORS[category](rng, max_depth), case_id=f"{index:03d}-{category}"
        )
        try:
            parse(lex(case.source))
        except Exception as exc:  # generator bug must surface as a diagnostic
            diagnostics.append(
                {
                    "code": "E003",
                    "message": f"generated case {case.case_id} failed to parse",
                    "detail": str(exc),
                }
            )
            continue
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        cases.append(case)
    return MutationReport(
        seed=seed,
        count=count,
        max_depth=max_depth,
        cases=tuple(cases),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Mutation score gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    path: str
    original: str
    replacement: str
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "path": self.path,
            "original": self.original,
            "replacement": self.replacement,
            "description": self.description,
        }


@dataclass(frozen=True)
class MutantResult:
    mutant_id: str
    killed: bool
    returncode: int
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "killed": self.killed,
            "returncode": self.returncode,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class MutationScoreReport:
    score: float
    killed: int
    total: int
    threshold: float
    passed: bool
    results: tuple[MutantResult, ...]
    diagnostics: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "killed": self.killed,
            "total": self.total,
            "threshold": self.threshold,
            "passed": self.passed,
            "results": [item.as_dict() for item in self.results],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "caveats": (
                "mutation score measures test sensitivity, not correctness",
                "surviving mutants show a blind spot in the supplied suite",
                "workspaces are isolated below the caller-supplied root",
            ),
        }


def _coerce_mutants(
    values: Sequence[MutantSpec | Mapping[str, str]],
) -> tuple[MutantSpec, ...]:
    result: list[MutantSpec] = []
    for value in values:
        if isinstance(value, MutantSpec):
            result.append(value)
        elif isinstance(value, Mapping):
            required = ("mutant_id", "path", "original", "replacement")
            missing = [name for name in required if name not in value]
            if missing:
                raise _fail("E003", f"mutant spec missing {', '.join(missing)}")
            result.append(MutantSpec(**dict(value)))
        else:
            raise _fail("E003", "mutant specs must be MutantSpec or mapping values")
    return tuple(result)


def _default_runner(
    args: tuple[str, ...], cwd: Path, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _invoke_runner(
    runner: Callable[..., Any],
    args: tuple[str, ...],
    cwd: Path,
    env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    try:
        accepts_env = "env" in inspect.signature(runner).parameters
    except TypeError, ValueError:
        accepts_env = True
    process = runner(args, cwd=cwd, env=env) if accepts_env else runner(args, cwd)
    if not hasattr(process, "returncode"):
        raise _fail("E004", "mutation runner returned an invalid process result")
    return process


def _build_workspace(root: Path, workspace: Path, mutant: MutantSpec) -> Path:
    parts = Path(mutant.path).parts
    if not parts or any(part in {"..", "."} for part in parts):
        raise _fail("E003", f"invalid mutant path {mutant.path!r}")
    target = root.joinpath(*parts)
    if not target.is_file():
        raise _fail("E006", f"mutant target file not found: {target}")
    original_text = target.read_text(encoding="utf-8")
    if mutant.original not in original_text:
        raise _fail(
            "E006",
            f"mutant {mutant.mutant_id}: original text not found in {mutant.path}",
        )

    workspace.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.name in SKIP_WORKSPACE_NAMES or child.name == parts[0]:
            continue
        try:
            (workspace / child.name).symlink_to(
                child, target_is_directory=child.is_dir()
            )
        except FileExistsError:
            pass
    first = workspace / parts[0]
    if not first.exists():
        shutil.copytree(root / parts[0], first)
    mutated_file = workspace.joinpath(*parts)
    mutated_file.parent.mkdir(parents=True, exist_ok=True)
    mutated_file.write_text(
        original_text.replace(mutant.original, mutant.replacement, 1), encoding="utf-8"
    )
    return workspace


def adv_run_mutation_score_gate(
    mutants: Sequence[MutantSpec | Mapping[str, str]] | None = None,
    *,
    root: Path | str | None = None,
    workspace_root: Path | str | None = None,
    test_command: Sequence[str] = ("python", "-m", "pytest", "-q"),
    runner: Callable[..., Any] | None = None,
    threshold: float = 0.8,
    cache_root: Path | str | None = None,
    strict: bool = False,
    span: Span = None,
) -> MutationScoreReport:
    """Apply mutants in isolated workspaces and score test-suite sensitivity."""
    if not 0.0 <= threshold <= 1.0:
        raise _fail("E003", "threshold must be in [0, 1]", span=span)
    repo_root = Path(root).expanduser().resolve() if root is not None else Path.cwd()
    base = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path(cache_root).expanduser().resolve() / "mutation-workspaces"
        if cache_root is not None
        else None
    )
    if base is None:
        raise _fail(
            "E003",
            "workspace_root or cache_root is required to keep mutations isolated",
            span=span,
        )
    base.mkdir(parents=True, exist_ok=True)
    effective_runner = runner or _default_runner
    cargo_target = (
        Path(cache_root).expanduser().resolve() / "cargo-target"
        if cache_root is not None
        else None
    )
    env = dict(os.environ)
    if cargo_target is not None:
        cargo_target.mkdir(parents=True, exist_ok=True)
        env["CARGO_TARGET_DIR"] = str(cargo_target)

    specs = _coerce_mutants(
        tuple(DEFAULT_MUTANTS) if mutants is None else tuple(mutants)
    )
    results: list[MutantResult] = []
    diagnostics: list[dict[str, str]] = []
    command = tuple(test_command)
    for mutant in specs:
        workspace = base / mutant.mutant_id
        try:
            _build_workspace(repo_root, workspace, mutant)
            process = _invoke_runner(effective_runner, command, workspace, env)
            output = f"{process.stdout or ''}\n{process.stderr or ''}".strip()
            killed = int(process.returncode) != 0
            results.append(
                MutantResult(
                    mutant.mutant_id,
                    killed,
                    int(process.returncode),
                    output[-2000:],
                )
            )
        except PillarError as exc:
            diagnostics.append(
                {
                    "code": exc.code,
                    "message": f"mutant {mutant.mutant_id} could not be applied",
                    "detail": exc.diagnostic.message,
                }
            )
    total = len(results)
    killed = sum(item.killed for item in results)
    score = killed / total if total else 0.0
    passed = total > 0 and score >= threshold and not diagnostics
    if not passed and strict:
        raise _fail(
            "E002",
            f"mutation score {score:.3f} is below threshold {threshold:.3f}",
            span=span,
            reason=f"killed={killed}/{total}, application_errors={len(diagnostics)}",
            help_text="add tests that exercise the weakened checks or repair the mutant spec",
        )
    return MutationScoreReport(
        score=score,
        killed=killed,
        total=total,
        threshold=threshold,
        passed=passed,
        results=tuple(results),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Spec oracle and full verification report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleCase:
    name: str
    expected: Any
    observed: Any
    passed: bool
    basis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": _jsonable(self.expected),
            "observed": _jsonable(self.observed),
            "passed": self.passed,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class VerificationResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SpecOracleReport:
    passed: bool
    cases: tuple[OracleCase, ...]
    checks: tuple[VerificationResult, ...]
    diagnostics: tuple[dict[str, str], ...]
    caveats: tuple[str, ...]
    next_steps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cases": [item.as_dict() for item in self.cases],
            "checks": [item.as_dict() for item in self.checks],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "caveats": list(self.caveats),
            "next_steps": list(self.next_steps),
        }


def _coerce_observation(
    value: Any,
) -> tuple[Any, Any]:
    if callable(value):
        value = value()
    if isinstance(value, Mapping) and {"expected", "observed"} <= set(value):
        return value["expected"], value["observed"]
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise _fail("E003", "oracle observations must provide expected and observed values")


def _normalize_observation(value: Any) -> Any:
    if isinstance(value, BaseException | Mapping):
        return normalize_error(value)
    return _jsonable(value)


def adv_verify_spec_oracle_and_report(
    oracle: Mapping[str, Any],
    *,
    root: Path | str | None = None,
    runner: Callable[..., Any] | None = None,
    python: str = "python",
    ruff: str = "ruff",
    cargo: str = "cargo",
    cache_root: Path | str | None = None,
    include_cargo: bool = True,
    run_verification: bool = True,
    span: Span = None,
) -> SpecOracleReport:
    """Compare observations to a reference spec and run full verification.

    Each verification result keeps the exact command and output.  Cargo uses an
    isolated ``CARGO_TARGET_DIR`` below ``cache_root`` when one is supplied.
    """
    repo_root = Path(root).expanduser().resolve() if root is not None else Path.cwd()
    observations: list[OracleCase] = []
    diagnostics: list[dict[str, str]] = []
    for name, value in oracle.items():
        try:
            expected, observed = _coerce_observation(value)
            normalized_expected = _normalize_observation(expected)
            normalized_observed = _normalize_observation(observed)
            observations.append(
                OracleCase(
                    name,
                    normalized_expected,
                    normalized_observed,
                    normalized_expected == normalized_observed,
                    "caller-supplied reference specification",
                )
            )
        except PillarError as exc:
            diagnostics.append(
                {"code": exc.code, "message": name, "detail": exc.diagnostic.message}
            )

    checks: list[VerificationResult] = []
    if run_verification:
        effective_runner = runner or _default_runner
        env = dict(os.environ)
        if cache_root is not None:
            cargo_target = Path(cache_root).expanduser().resolve() / "cargo-target"
            cargo_target.mkdir(parents=True, exist_ok=True)
            env["CARGO_TARGET_DIR"] = str(cargo_target)
        commands: list[tuple[str, tuple[str, ...]]] = []
        for name, command in VERIFICATION_COMMANDS:
            resolved = tuple(
                python if part == "python" else ruff if part == "ruff" else part
                for part in command
            )
            commands.append((name, resolved))
        if include_cargo:
            commands.append(("cargo", (cargo, *CARGO_COMMAND[1:])))
        for name, command in commands:
            process = _invoke_runner(effective_runner, command, repo_root, env)
            checks.append(
                VerificationResult(
                    name=name,
                    command=command,
                    returncode=int(process.returncode),
                    stdout=process.stdout or "",
                    stderr=process.stderr or "",
                    passed=int(process.returncode) == 0,
                )
            )

    oracle_ok = bool(observations) and all(item.passed for item in observations)
    checks_ok = all(item.passed for item in checks) if run_verification else True
    passed = oracle_ok and checks_ok and not diagnostics
    caveats = (
        "bounded oracle set; passing cases do not prove full specification compliance",
        "verification commands report tool results, not undecidable properties",
        "mutation/parity scores measure sensitivity up to the declared bounds",
    )
    next_steps: list[str] = []
    if not oracle_ok:
        next_steps.append(
            "inspect divergent oracle cases and update the reference spec"
        )
    if not checks_ok:
        next_steps.extend(
            f"rerun {item.name} and repair its reported failure"
            for item in checks
            if not item.passed
        )
    if diagnostics:
        next_steps.append("repair malformed oracle observations")
    if passed:
        next_steps.append("extend adversarial cases and re-measure mutation score")
    if any(item.name == "cargo" and not item.passed for item in checks):
        next_steps.append(
            "confirm native toolchain availability and isolated target dir"
        )

    report = SpecOracleReport(
        passed=passed,
        cases=tuple(observations),
        checks=tuple(checks),
        diagnostics=tuple(diagnostics),
        caveats=caveats,
        next_steps=tuple(next_steps),
    )
    if (
        not passed
        and run_verification
        and any(item.name == "cargo" and not item.passed for item in checks)
    ):
        diagnostics.append(
            {
                "code": "E004",
                "message": "cargo verification failed",
                "detail": "see the cargo check result for exact output",
            }
        )
        return SpecOracleReport(
            passed=report.passed,
            cases=report.cases,
            checks=report.checks,
            diagnostics=tuple(diagnostics),
            caveats=report.caveats,
            next_steps=report.next_steps,
        )
    return report


__all__ = [
    "AdversarialCase",
    "MutantResult",
    "MutantSpec",
    "MutationReport",
    "MutationScoreReport",
    "OracleCase",
    "SpecOracleReport",
    "VerificationResult",
    "adv_mutate_ast_and_type_constraints",
    "adv_run_mutation_score_gate",
    "adv_verify_spec_oracle_and_report",
]
