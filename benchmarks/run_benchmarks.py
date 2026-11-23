#!/usr/bin/env python3
"""Compare Polyglot backends with CPython on identical workloads.

What the numbers mean:
  * ``run_ms`` comes from ``polyglot exec --result-file`` and excludes process
    startup and compilation.
  * CPython numbers are timed inside the script, excluding interpreter startup.
  * Every cell is the minimum of ``--reps`` runs.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --reps 5 --backends python
    python benchmarks/run_benchmarks.py --write benchmarks/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKLOAD_DIR = ROOT / "benchmarks" / "workloads"
CPYTHON_DIR = ROOT / "benchmarks" / "cpython"
WORKLOADS = ("fib", "loop_sum", "strings")


def _polyglot_command() -> list[str]:
    venv_bin = ROOT / ".venv" / "bin" / "polyglot"
    if venv_bin.exists():
        return [str(venv_bin)]
    return ["polyglot"]


def run_cpython(workload: str, reps: int) -> tuple[float, str]:
    """Return (min milliseconds, printed result) for the CPython baseline."""
    best: float | None = None
    result = ""
    for _ in range(reps):
        proc = subprocess.run(
            [sys.executable, str(CPYTHON_DIR / f"{workload}.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        value, millis = proc.stdout.split()
        result = value
        elapsed = float(millis)
        if best is None or elapsed < best:
            best = elapsed
    return best or 0.0, result


def run_polyglot(workload: str, backend: str, reps: int) -> tuple[float, float, str]:
    """Return (min run_ms, min compile_ms, return_value) for one backend."""
    command = _polyglot_command()
    source = WORKLOAD_DIR / f"{workload}.poly"
    best_run: float | None = None
    best_compile: float | None = None
    result = ""
    for attempt in range(reps):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = Path(tmp.name)
        started = time.perf_counter()
        proc = subprocess.run(
            command
            + ["exec", str(source), "--backend", backend, "--result-file", str(out_path)],
            capture_output=True,
            text=True,
        )
        wall = time.perf_counter() - started
        if proc.returncode != 0:
            raise RuntimeError(
                f"{workload}/{backend} failed:\n{proc.stdout}{proc.stderr}"
            )
        payload = json.loads(out_path.read_text())
        out_path.unlink(missing_ok=True)
        run_ms = payload.get("run_ms") or 0.0
        compile_ms = payload.get("compile_ms") or 0.0
        if attempt == 0 and wall > 30:
            print(
                f"  {workload}/{backend}: first run took {wall:.0f}s "
                "(native build)",
                file=sys.stderr,
            )
        result = str(payload.get("return_value"))
        if best_run is None or run_ms < best_run:
            best_run = run_ms
        if best_compile is None or compile_ms < best_compile:
            best_compile = compile_ms
    return best_run or 0.0, best_compile or 0.0, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per cell")
    parser.add_argument(
        "--backends",
        default="python,native",
        help="Comma-separated Polyglot backends to measure",
    )
    parser.add_argument(
        "--write",
        default=None,
        help="Write a markdown report to this path",
    )
    args = parser.parse_args(argv)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rows: list[tuple[str, float, dict[str, tuple[float, float, str]]]] = []
    mismatches: list[str] = []
    for workload in WORKLOADS:
        print(f"measuring {workload} ...", file=sys.stderr)
        cpython_ms, cpython_result = run_cpython(workload, args.reps)
        measured: dict[str, tuple[float, float, str]] = {}
        for backend in backends:
            measured[backend] = run_polyglot(workload, backend, args.reps)
        rows.append((workload, cpython_ms, measured))
        for backend, (run_ms, _compile_ms, value) in measured.items():
            if value != cpython_result:
                mismatches.append(f"{workload}/{backend}")
                print(
                    f"  note: {workload}/{backend} returned {value}, "
                    f"CPython returned {cpython_result}",
                    file=sys.stderr,
                )

    header = ["Workload", "CPython (ms)"]
    header += [f"Polyglot {b} (ms)" for b in backends]
    header += [f"{b} vs CPython" for b in backends]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for workload, cpython_ms, measured in rows:
        cells = [workload, f"{cpython_ms:.1f}"]
        cells += [f"{measured[b][0]:.1f}" for b in backends]
        for backend in backends:
            run_ms = measured[backend][0]
            # Sub-millisecond results make a ratio meaningless; say so instead
            # of printing a six-digit "speedup".
            if run_ms < 0.5:
                cells.append("n/a (<0.5 ms)")
            else:
                cells.append(f"{cpython_ms / run_ms:.2f}x")
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print(table)

    if args.write:
        report = [
            "# Benchmark results",
            "",
            f"- Machine: {platform.machine()} / {platform.platform()}",
            f"- Python: {platform.python_version()}",
            f"- Repetitions per cell: {args.reps} (minimum reported)",
            "- Polyglot numbers are `run_ms` from `polyglot exec`, so compilation",
            "  and interpreter startup are excluded.",
            "- CPython numbers are timed inside the script, interpreter startup excluded.",
            "- `n/a (<0.5 ms)` means the run was too short to express a ratio.",
        ]
        if mismatches:
            report.append(
                "- Return value mismatch (results are not comparable): "
                + ", ".join(mismatches)
            )
        else:
            report.append("- Every backend returned the same value as CPython.")
        report += [
            "",
            table,
            "",
            "A ratio above 1.00x means Polyglot was faster than CPython.",
            "",
        ]
        Path(args.write).write_text("\n".join(report), encoding="utf-8")
        print(f"wrote {args.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
