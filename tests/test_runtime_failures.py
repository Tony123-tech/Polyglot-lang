"""Runtime failures report a diagnostic code, a source position, and a fix hint.

Before this contract, a failed guard surfaced as whatever the host language
raised — ``ZeroDivisionError: division by zero`` from Python, ``I/O timer
expired`` style panics from Zig — and the CLI printed ``at: <unknown>`` with
``note: ZeroDivisionError``. These tests pin the replacement behaviour: every
guard raises ``RuntimeFailure`` carrying ``E002`` plus the line and column of
the failing expression, and both backends print the same first line and the
same ``at: line N, column M`` for the same program.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from backends.python.translator import RuntimeFailure, compile_rust
from compiler.core.lexer import lex
from compiler.core.parser import parse

ROOT = Path(__file__).resolve().parent.parent


def _run(source: str):
    return compile_rust(parse(lex(source)))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-m", "api.cli", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


DIVIDE_BY_ZERO = """fn main() -> i64 {
    let a: i64 = 10;
    let b: i64 = 0;
    let c: i64 = a / b;
    c
}
"""

REMAINDER_BY_ZERO = """fn main() -> i64 {
    let a: i64 = 10;
    let b: i64 = 0;
    let c: i64 = a % b;
    c
}
"""

INDEX_OUT_OF_BOUNDS = """fn main() -> i64 {
    let v: Vec<i64> = vec![1, 2, 3];
    let x: i64 = v[5];
    x
}
"""

PANIC = """fn main() -> i64 {
    panic!("boom");
    0
}
"""

ASSERT = """fn main() -> i64 {
    assert!(1 == 2);
    0
}
"""

# The position is the operator / index / macro-call token, which both backends
# read from the same AST node, so the numbers are identical by construction.
EXPECTED_LOCATION = {
    DIVIDE_BY_ZERO: (4, 20),
    REMAINDER_BY_ZERO: (4, 20),
    INDEX_OUT_OF_BOUNDS: (3, 19),
    PANIC: (2, 5),
    ASSERT: (2, 5),
}


def test_division_by_zero_raises_a_located_runtime_failure():
    namespace = _run(DIVIDE_BY_ZERO)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    failure = caught.value
    assert failure.code == "E002"
    assert str(failure) == "division by zero"
    assert (failure.line, failure.col) == (4, 20)
    assert "divisor" in failure.help_text


def test_remainder_by_zero_raises_a_located_runtime_failure():
    namespace = _run(REMAINDER_BY_ZERO)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    assert str(caught.value) == "remainder by zero"
    assert (caught.value.line, caught.value.col) == (4, 20)


def test_index_out_of_bounds_reports_the_index_and_the_length():
    namespace = _run(INDEX_OUT_OF_BOUNDS)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    failure = caught.value
    assert str(failure) == "index 5 out of bounds for length 3"
    assert (failure.line, failure.col) == (3, 19)


def test_panic_reports_the_message_and_the_call_site():
    namespace = _run(PANIC)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    failure = caught.value
    assert str(failure) == "boom"
    assert (failure.line, failure.col) == (2, 5)


def test_assert_reports_the_call_site():
    namespace = _run(ASSERT)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    assert (caught.value.line, caught.value.col) == (2, 5)


def test_assert_keeps_the_formatted_message():
    source = """fn main() -> i64 {
    let x: i64 = 3;
    assert!(x > 10, "x was {}", x);
    x
}
"""
    namespace = _run(source)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    assert str(caught.value) == "x was 3"


def test_assert_eq_reports_both_sides():
    source = """fn main() -> i64 {
    assert_eq!(1 + 1, 3);
    0
}
"""
    namespace = _run(source)
    with pytest.raises(RuntimeFailure) as caught:
        namespace["main"]()
    assert str(caught.value) == "2 != 3"


def test_floats_still_divide_by_zero_without_failing():
    source = """fn main() -> f64 {
    let a: f64 = 1.0;
    let b: f64 = 0.0;
    a / b
}
"""
    namespace = _run(source)
    assert namespace["main"]() == float("inf")


def test_cli_prints_the_runtime_failure_five_line_format(tmp_path: Path):
    source = tmp_path / "divide.poly"
    source.write_text(DIVIDE_BY_ZERO, encoding="utf-8")
    result = _run_cli("run", str(source))
    assert result.returncode == 1
    lines = [line.strip() for line in result.stderr.strip().splitlines() if line.strip()]
    assert lines[0] == "error[E002]: division by zero"
    assert lines[1] == "reason: division by zero"
    assert lines[2].startswith("help: ")
    assert lines[3] == "note: runtime failure"
    assert lines[4] == "at: line 4, column 20"


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig toolchain not available")
@pytest.mark.parametrize(
    "source",
    [DIVIDE_BY_ZERO, REMAINDER_BY_ZERO, PANIC, ASSERT],
)
def test_zig_reports_the_same_code_and_location(source: str, tmp_path: Path):
    path = tmp_path / "case.poly"
    path.write_text(source, encoding="utf-8")
    result = _run_cli("run", str(path), "--backend", "zig")
    assert result.returncode == 1
    lines = [line.strip() for line in result.stderr.strip().splitlines() if line.strip()]
    line, col = EXPECTED_LOCATION[source]
    assert lines[0].startswith("error[E002]: ")
    assert lines[1] == f"at: line {line}, column {col}"


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig toolchain not available")
@pytest.mark.parametrize("source", [DIVIDE_BY_ZERO, PANIC, ASSERT])
def test_both_backends_agree_on_the_reported_failure(source: str, tmp_path: Path):
    path = tmp_path / "case.poly"
    path.write_text(source, encoding="utf-8")
    python_run = _run_cli("run", str(path))
    zig_run = _run_cli("run", str(path), "--backend", "zig")
    python_lines = python_run.stderr.strip().splitlines()
    zig_lines = zig_run.stderr.strip().splitlines()
    assert python_lines[0] == zig_lines[0]
    assert python_lines[-1].strip() == zig_lines[-1].strip()
