"""Unicode-safe string helpers and container extras in the bundled stdlib.

The probe returns a bit-mask score: every check adds a distinct value, so a
failure tells you which helper broke instead of just "something is off".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backends import get_backend
from include_expander import load_program

PROBE = """\
include!("std:str_utils");
include!("std:collections");
include!("std:fs");

fn probe() -> i64 {
    let mut score = 0;
    // 1: reverse a string containing an emoji (byte slicing used to corrupt it)
    if reverse_text("a😀b".to_string()) == "b😀a".to_string() { score = score + 1; }
    // 4: reverse an accented string
    if reverse_text("café".to_string()) == "éfac".to_string() { score = score + 4; }
    // 10: palindrome over an emoji string
    if is_palindrome("😀a😀".to_string()) { score = score + 10; }
    // 20: padding width counts characters, not bytes
    if pad_left("é".to_string(), 3, "0".to_string()) == "00é".to_string() { score = score + 20; }

    // 40: Counter::top_key returns the most frequent key
    let mut counter = Counter::new();
    counter.add("x".to_string());
    counter.add("y".to_string());
    counter.add("x".to_string());
    if counter.top_key().unwrap_or("".to_string()) == "x".to_string() { score = score + 40; }

    // 80: fs::read_lines drops exactly one trailing newline
    let path = "/tmp/polyglot_stdlib_unicode_test.txt";
    write_text(path.clone(), "a\\nb\\n".to_string()).unwrap();
    let lines = read_lines(path).unwrap_or([].to_vec());
    if (lines.len() as i64) == 2 { score = score + 80; }
    score
}

fn main() -> i64 {
    probe()
}
"""

EXPECTED = 155


def _compile_probe(tmp_path: Path, backend: str) -> int:
    source_path = tmp_path / "probe.poly"
    source_path.write_text(PROBE, encoding="utf-8")
    namespace = get_backend(backend, trusted=True).compile(
        load_program(str(source_path))
    )
    return namespace["main"]()


def test_stdlib_unicode_probe_python(tmp_path: Path) -> None:
    assert _compile_probe(tmp_path, "python") == EXPECTED


@pytest.mark.slow
def test_stdlib_unicode_probe_native(tmp_path: Path) -> None:
    assert _compile_probe(tmp_path, "native") == EXPECTED
