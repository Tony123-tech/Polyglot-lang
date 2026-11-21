"""Deterministic Python/Rust-PyO3/WASM differential parity harness.

Every ``(seed, index)`` pair maps to one byte-identical source program.  The
harness normalizes outcomes into ``result``/``stdout``/``error`` records using
the same code mapping as :mod:`tools.diff_fuzz`; it compares failure modes and
normalized ``E001``–``E008`` codes rather than exception object identity.

Backend boundaries
------------------
The WASM backend specializes compile-time aggregate shapes (structs, enums,
vectors, bounded generic impls) into scalar ``i64``/``bool`` expressions before
WAT emission, so the full deterministic matrix runs on all three backends.
Constructs that genuinely require a dynamic aggregate ABI (for example an
aggregate function parameter or return value) cannot be specialized and are
marked ``wasm_supported=False``: WASM reports a structured unsupported
diagnostic and the harness records an explicit *boundary skip* rather than a
false parity bug.

Isolation and artifacts
-----------------------
Native builds use a per-case ``cache_scope`` under a caller-supplied root; WASM
instantiation is fresh per case.  Divergence artifacts (seed, source, and
normalized outcomes) are written only when a real mismatch is found.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow ``python tools/three_way_parity.py ...`` as well as package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_ci.hard import ci_persist_divergence, ci_scoped_backend_environment
from tools.diff_fuzz import normalize_error

BACKENDS = ("python", "native", "wasm")


@dataclass(frozen=True)
class ParityCase:
    """One reproducible three-backend case."""

    case_id: str
    category: str
    source: str
    function: str = "main"
    args: tuple[Any, ...] = ()
    expected_code: str | None = None
    wasm_supported: bool = True


@dataclass
class ThreeWayReport:
    """Summary of one deterministic parity run."""

    seed: int
    cases: int
    outcomes: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    skipped: dict[str, dict[str, str]] = field(default_factory=dict)
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "cases": self.cases,
            "passed": self.passed,
            "backends": list(BACKENDS),
            "outcomes": self.outcomes,
            "skipped": self.skipped,
            "mismatches": self.mismatches,
            "artifacts": self.artifacts,
            "counts": {
                "mismatches": len(self.mismatches),
                "skipped": sum(len(item) for item in self.skipped.values()),
            },
        }


def _call(namespace: Mapping[str, Any], case: ParityCase) -> Any:
    function = namespace.get(case.function)
    if not callable(function):
        raise KeyError(f"{case.function} is not exported")
    return function(*case.args)


def iter_three_way_cases(seed: int) -> Iterator[ParityCase]:
    """Yield the fixed seed-reproducible parity matrix."""
    from tools import diff_fuzz as df

    rng = random.Random(f"three-way-parity:{seed}")
    k, m = rng.randint(1, 7), rng.randint(2, 9)
    yield ParityCase(
        "scalar-add",
        "scalar",
        f"fn main(a: i64, b: i64) -> i64 {{ a + b * {k} + {m} }}",
        args=(seed % 11, 3),
    )
    yield ParityCase(
        "scalar-loop",
        "scalar",
        "fn main(n: i64) -> i64 { let mut i: i64 = 0; let mut t: i64 = 0; "
        "while i < n { t += i; i += 1; } t }",
        args=(seed % 7,),
    )
    for case_id, template in (
        ("recursive-struct", df._tpl_recursive_struct),
        ("recursive-enum", df._tpl_recursive_enum),
        ("nested-vec", df._tpl_vec_nested),
        ("trait-bounded-impl", df._tpl_traits_bounded_impl),
        ("generic-bounded-box", df._tpl_bounded_recursive_box),
    ):
        category = (
            "recursive_generics"
            if case_id.startswith(("recursive", "nested"))
            else "traits_bounds"
        )
        yield ParityCase(
            case_id,
            category,
            template(rng),
        )
    yield ParityCase(
        "diagnostic-e003-syntax",
        "diagnostics",
        "fn main(a: i64) -> i64 { a + }",
        expected_code="E003",
    )
    for case_id, template, code in (
        ("diagnostic-e001", df._tpl_diagnostic_e001, "E001"),
        ("diagnostic-e002", df._tpl_diagnostic_e002, "E002"),
        ("diagnostic-e003-arity", df._tpl_diagnostic_e003, "E003"),
        ("diagnostic-e008", df._tpl_diagnostic_e008, "E008"),
    ):
        yield ParityCase(
            case_id,
            "diagnostics",
            template(rng),
            expected_code=code,
        )


def _run_python(case: ParityCase, _root: Path | None) -> dict[str, Any]:
    from backends import get_backend
    from tools.diff_fuzz import _program_from_source

    namespace = get_backend("python", trusted=True).compile(
        _program_from_source(case.source)
    )
    return {"result": _call(namespace, case), "stdout": "", "error": None}


def _run_native(case: ParityCase, root: Path | None) -> dict[str, Any]:
    from backends.rust.compiler import NativeBackend
    from tools.diff_fuzz import _program_from_source

    cache = (root / "native-cache") if root is not None else None
    backend = NativeBackend(cache_dir=cache, trusted=True)
    namespace = backend.compile(
        _program_from_source(case.source),
        module_name=f"twp_{case.case_id.replace('-', '_')}",
        incremental=True,
        cache_scope=f"three-way:{case.case_id}",
    )
    return {"result": _call(namespace, case), "stdout": "", "error": None}


def _run_wasm(case: ParityCase, _root: Path | None) -> dict[str, Any]:
    from backends import get_backend
    from tools.diff_fuzz import _program_from_source

    namespace = get_backend("wasm", trusted=True).compile(
        _program_from_source(case.source)
    )
    return {"result": _call(namespace, case), "stdout": "", "error": None}


_RUNNERS: dict[str, Callable[[ParityCase, Path | None], dict[str, Any]]] = {
    "python": _run_python,
    "native": _run_native,
    "wasm": _run_wasm,
}


def _execute(backend: str, case: ParityCase, root: Path | None) -> dict[str, Any]:
    if backend == "native" and shutil.which("cargo") is None:
        return {
            "result": None,
            "stdout": "",
            "error": None,
            "skipped": True,
            "reason": "cargo is not available on PATH",
        }
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            raw = _RUNNERS[backend](case, root)
    except Exception as error:
        return {
            "result": None,
            "stdout": output.getvalue(),
            "error": normalize_error(error),
        }
    return {
        "result": raw.get("result"),
        "stdout": raw.get("stdout", output.getvalue()),
        "error": None if raw.get("error") is None else normalize_error(raw["error"]),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item) for key, item in sorted(value.items(), key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def run_three_way_parity(
    seed: int,
    *,
    cases: Sequence[ParityCase] | None = None,
    cache_root: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    backends: Sequence[str] = BACKENDS,
    include_native: bool = True,
) -> ThreeWayReport:
    """Execute the parity matrix and persist artifacts only on divergence."""
    matrix = list(cases) if cases is not None else list(iter_three_way_cases(seed))
    report = ThreeWayReport(seed=seed, cases=len(matrix))
    root = Path(cache_root) if cache_root is not None else None
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else None
    active = [name for name in backends if name != "native" or include_native]

    for case in matrix:
        report.outcomes[case.case_id] = {}
        report.skipped[case.case_id] = {}
        for backend in active:
            if backend == "wasm" and not case.wasm_supported:
                report.skipped[case.case_id][backend] = (
                    "WASM backend requires a scalar i64/bool ABI for this case"
                )
                continue
            outcome = _execute(backend, case, root)
            if outcome.pop("skipped", False):
                report.skipped[case.case_id][backend] = str(
                    outcome.get("reason", "backend unavailable")
                )
                continue
            report.outcomes[case.case_id][backend] = _jsonable(outcome)

        available = [name for name in active if name in report.outcomes[case.case_id]]
        if case.expected_code is not None:
            for backend in available:
                error = report.outcomes[case.case_id][backend].get("error")
                if error is None or error.get("code") != case.expected_code:
                    report.mismatches.append(
                        {
                            "case": case.case_id,
                            "backend": backend,
                            "kind": "expected-code",
                            "expected": case.expected_code,
                            "observed": error,
                        }
                    )
        for index, left_name in enumerate(available):
            for right_name in available[index + 1 :]:
                left = report.outcomes[case.case_id][left_name]
                right = report.outcomes[case.case_id][right_name]
                if all(
                    left.get(key) == right.get(key)
                    for key in ("result", "stdout", "error")
                ):
                    continue
                report.mismatches.append(
                    {
                        "case": case.case_id,
                        "backends": [left_name, right_name],
                        "kind": "outcome-divergence",
                        "left": left,
                        "right": right,
                    }
                )

    case_mismatches: dict[str, list[dict[str, Any]]] = {}
    for mismatch in report.mismatches:
        case_mismatches.setdefault(str(mismatch["case"]), []).append(mismatch)
    if artifacts is not None:
        for case in matrix:
            if case.case_id not in case_mismatches:
                continue
            written = ci_persist_divergence(
                {
                    "passed": False,
                    "mismatches": case_mismatches[case.case_id],
                },
                directory=artifacts,
                case={
                    "id": case.case_id,
                    "source": case.source,
                    "category": case.category,
                },
                seed=seed,
            )
            if written is not None:
                report.artifacts.append(str(written))
    return report


def scope_for_case(case: ParityCase, seed: int, *, root: Path | str) -> dict[str, Any]:
    """Return the isolated build environment for one backend-neutral case."""
    return ci_scoped_backend_environment(
        root, backend="python", seed=seed, case_id=case.case_id
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 0 only when every backend agrees."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="isolated native build cache (defaults to a system temp dir)",
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cache_root = args.cache_root
    if cache_root is None:
        import tempfile

        cache_root = Path(tempfile.mkdtemp(prefix="polyglot-three-way-parity-"))
    report = run_three_way_parity(
        args.seed,
        cache_root=cache_root,
        artifacts_dir=args.artifacts,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if report.passed else "FAIL"
        print(
            f"three-way parity seed={args.seed} {status} "
            f"cases={report.cases} mismatches={len(report.mismatches)} "
            f"skipped={sum(len(item) for item in report.skipped.values())}"
        )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKENDS",
    "ParityCase",
    "ThreeWayReport",
    "iter_three_way_cases",
    "main",
    "run_three_way_parity",
    "scope_for_case",
]
