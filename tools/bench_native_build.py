"""Measure cold and warm native builds for a small program.

This script is diagnostic only. It does not assert timings, so CI can run it
without flaking on shared hardware.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backends.rust.compiler import NativeBackend
from lexer import lex
from parser import parse

SOURCE = """
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i64 {
    fib(15)
}
"""


def _time_build(cache_dir: Path) -> float:
    backend = NativeBackend(cache_dir=cache_dir)
    program = parse(lex(SOURCE))
    start = time.perf_counter()
    backend.compile(program)
    return time.perf_counter() - start


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="polyglot-bench-") as temp:
        cache = Path(temp) / "cache"
        cold = _time_build(cache)
        warm = _time_build(cache)
    shutil.rmtree(cache, ignore_errors=True)
    print(f"cold build: {cold:.3f}s")
    print(f"warm build: {warm:.3f}s")
    if warm > 0:
        print(f"speedup:    {cold / warm:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
