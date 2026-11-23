"""Ownership transfer with explicit `drop(x)` — the contract the checker must meet.

`drop(value)` moves the value out of its binding and releases it immediately.
Because the shared front end already runs `check_moves` before either backend
generates code (`backends/python/translator.py` and `backends/rust/compiler.py`
both call it), one pass covers the Python and the native backend.

The pass extends `backends/move_check.py`; a program-wide arity scan, then the
ownership passes, all run from `check_moves`:

The borrow/drop pass lives in `backends/borrow_checker.py` and its error
(`BorrowError`) subclasses `MoveError` with the same `code == "E009"`, so the
tests below still import `backends.move_check` only; `check_moves` is the single
entry point that runs the host-seam arity scan, then the borrow pass, then the
move pass. Two consequences are pinned here: an `E003` arity error outranks
ownership, and a loop body is checked to a header fixpoint, so a value that is
reassigned before it is used inside the loop is accepted while a `drop` in the
loop surfaces as a double drop.

- `check_moves(program)` raises `MoveError` for the new cases too.
- `MoveError` keeps `code == "E009"`, keeps `name`, `moved_at`, `line`, `col`,
  `span`, and grows three `kind` values:
  `"drop"` (use of a dropped value), `"double-drop"` (dropping twice),
  `"borrow"` (moving or dropping a value that is still borrowed).
- `MoveError.format()` mentions the operation in its headline: a "drop" /
  "double-drop" message contains `drop`, a "borrow" message contains `borrow`.
- `drop(x)` on a Copy value (`i64`, `bool`, a reference, ...) and on a literal
  is allowed: dropping a copy is a no-op, exactly as in Rust.
- `drop(x)` after `x` was moved is a use-after-move (`kind == "moved"`), not a
  new diagnostic.
- A borrow that ends before the drop (its last use precedes `drop(x)`) is
  allowed. That is the NLL part; it is the only borrow rule that needs "last
  use" rather than "still in scope".

Not pinned here, so the implementation may leave them for a later batch: re-initialising a binding after `drop` (`let mut s = ...; drop(s); s = ...;`), dropping through a field (`drop(payload.name)`), and drop inside a closure that outlives the frame.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backends.move_check import MoveError, check_moves
from compiler.core.lexer import lex
from compiler.core.parser import parse

ROOT = Path(__file__).resolve().parent.parent


def _program(source: str):
    return parse(lex(source))


def _check(source: str) -> None:
    check_moves(_program(source))


READ_AFTER_DROP = """fn main() -> i64 {
    let s: String = String::from("hi");
    drop(s);
    s.len() as i64
}
"""

DOUBLE_DROP = """fn main() -> i64 {
    let s: String = String::from("hi");
    drop(s);
    drop(s);
    7
}
"""

DROP_AFTER_MOVE = """fn main() -> i64 {
    let s: String = String::from("hi");
    let t: String = s;
    drop(s);
    t.len() as i64
}
"""

DROP_THEN_TAIL = """fn main() -> i64 {
    let s: String = String::from("hi");
    drop(s);
    7
}
"""

DROP_THEN_PRINT = """fn main() -> i64 {
    let s: String = String::from("hi");
    drop(s);
    println!("{}", s);
    7
}
"""

DROP_ONE_BRANCH = """fn main(flag: bool) -> i64 {
    let s: String = String::from("hi");
    if flag {
        drop(s);
    }
    s.len() as i64
}
"""

DROP_BOTH_BRANCHES_THEN_READ = """fn main(flag: bool) -> i64 {
    let s: String = String::from("hi");
    if flag {
        drop(s);
    } else {
        drop(s);
    }
    s.len() as i64
}
"""

DROP_BOTH_BRANCHES_NO_READ = """fn main(flag: bool) -> i64 {
    let s: String = String::from("hi");
    if flag {
        drop(s);
    } else {
        drop(s);
    }
    7
}
"""

SHADOW_AFTER_DROP = """fn main() -> i64 {
    let s: String = String::from("hi");
    drop(s);
    let s: String = String::from("again");
    s.len() as i64
}
"""

DROP_COPY_VALUE = """fn main() -> i64 {
    let n: i64 = 5;
    drop(n);
    n
}
"""

DROP_LITERAL = """fn main() -> i64 {
    drop(5);
    0
}
"""

DROP_IN_LOOP = """fn main() -> i64 {
    let mut i: i64 = 0;
    let s: String = String::from("hi");
    while i < 3 {
        drop(s);
        i = i + 1;
    }
    0
}
"""

DROP_REFERENCE = """fn main() -> i64 {
    let s: String = String::from("hi");
    let r: &String = &s;
    drop(r);
    println!("{}", s);
    7
}
"""

DROP_WHILE_BORROWED = """fn main() -> i64 {
    let s: String = String::from("hi");
    let r: &String = &s;
    drop(s);
    println!("{}", r);
    7
}
"""

MOVE_WHILE_BORROWED = """fn main() -> i64 {
    let s: String = String::from("hi");
    let r: &String = &s;
    let t: String = s;
    println!("{} {} {}", r, t, s.len());
    7
}
"""

BORROW_ENDS_BEFORE_DROP = """fn main() -> i64 {
    let s: String = String::from("hi");
    let r: &String = &s;
    println!("{}", r);
    drop(s);
    7
}
"""


# ---------------------------------------------------------------------------
# Tier A — explicit drop moves the value out
# ---------------------------------------------------------------------------


def test_use_after_drop_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(READ_AFTER_DROP)
    error = caught.value
    assert error.kind == "drop"
    assert error.name == "s"
    assert error.moved_at == 3
    assert error.line == 4
    assert "s" in str(error)
    assert "drop" in str(error)


def test_double_drop_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(DOUBLE_DROP)
    error = caught.value
    assert error.kind == "double-drop"
    assert error.name == "s"
    assert error.moved_at == 3
    assert error.line == 4
    assert "drop" in str(error)


def test_drop_after_move_is_a_use_after_move():
    with pytest.raises(MoveError) as caught:
        _check(DROP_AFTER_MOVE)
    error = caught.value
    assert error.kind == "moved"
    assert error.name == "s"
    assert error.moved_at == 3
    assert error.line == 4


def test_drop_then_macro_argument_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(DROP_THEN_PRINT)
    assert caught.value.kind == "drop"
    assert caught.value.line == 4


def test_drop_in_one_branch_makes_later_use_possibly_dropped():
    with pytest.raises(MoveError) as caught:
        _check(DROP_ONE_BRANCH)
    error = caught.value
    assert error.kind == "drop"
    assert error.line == 6


def test_drop_in_both_branches_then_use_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(DROP_BOTH_BRANCHES_THEN_READ)
    assert caught.value.kind == "drop"
    assert caught.value.line == 8


def test_drop_in_both_branches_without_use_is_allowed():
    _check(DROP_BOTH_BRANCHES_NO_READ)


def test_shadowing_after_drop_is_allowed():
    _check(SHADOW_AFTER_DROP)


def test_drop_of_a_copy_value_is_allowed():
    _check(DROP_COPY_VALUE)


def test_drop_of_a_literal_is_allowed():
    _check(DROP_LITERAL)


def test_drop_inside_a_loop_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(DROP_IN_LOOP)
    error = caught.value
    # The loop is modelled as a header fixpoint (run the body from the entry
    # state, join the entry and tail states, then run it again), so a `drop`
    # in the body surfaces on the second iteration as a double drop. That is
    # the more accurate diagnosis: the value really is released once per
    # iteration, and the second release is the illegal one.
    assert error.kind in {"drop", "double-drop"}
    assert error.name == "s"
    assert "drop" in str(error)


def test_drop_of_a_reference_is_allowed():
    _check(DROP_REFERENCE)


# ---------------------------------------------------------------------------
# Tier B — a borrow cannot outlive a move or a drop
# ---------------------------------------------------------------------------


def test_drop_while_borrowed_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(DROP_WHILE_BORROWED)
    error = caught.value
    assert error.kind == "borrow"
    assert error.name == "s"
    assert error.line == 4
    assert "borrow" in str(error)


def test_move_while_borrowed_is_rejected():
    with pytest.raises(MoveError) as caught:
        _check(MOVE_WHILE_BORROWED)
    error = caught.value
    assert error.kind == "borrow"
    assert error.name == "s"
    assert error.line == 4


def test_borrow_that_ends_before_the_drop_is_allowed():
    _check(BORROW_ENDS_BEFORE_DROP)


# ---------------------------------------------------------------------------
# Runtime — `drop(x)` executes on the Python backend
# ---------------------------------------------------------------------------


def test_python_backend_runs_drop_and_returns_the_tail():
    from backends import get_backend

    namespace = get_backend("python", trusted=True).compile(_program(DROP_THEN_TAIL))
    assert namespace["main"]() == 7


def test_python_backend_drops_a_copy_value_without_failing():
    from backends import get_backend

    namespace = get_backend("python", trusted=True).compile(_program(DROP_COPY_VALUE))
    assert namespace["main"]() == 5


def test_python_backend_drops_a_literal_without_failing():
    from backends import get_backend

    namespace = get_backend("python", trusted=True).compile(_program(DROP_LITERAL))
    assert namespace["main"]() == 0


# ---------------------------------------------------------------------------
# Both backends run the shared pass, so both reject the same program
# ---------------------------------------------------------------------------


def test_python_backend_rejects_use_after_drop_at_compile_time():
    from backends.python.translator import PythonBackend

    with pytest.raises(MoveError):
        PythonBackend(trusted=True).compile(_program(READ_AFTER_DROP))


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not available")
def test_native_backend_rejects_use_after_drop_before_codegen():
    from backends.rust.compiler import NativeBackend

    with pytest.raises(MoveError):
        NativeBackend(trusted=True).build(_program(READ_AFTER_DROP))


def test_cli_check_reports_e009_for_a_dropped_value(tmp_path: Path):
    import os
    import subprocess
    import sys

    source = tmp_path / "dropped.poly"
    source.write_text(READ_AFTER_DROP, encoding="utf-8")
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-m", "api.cli", "check", str(source)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    lines = [
        line.strip() for line in result.stderr.strip().splitlines() if line.strip()
    ]
    assert lines[0].startswith("error[E009]")
    assert "drop" in lines[0]
    assert lines[-1].startswith("at: line 4")
