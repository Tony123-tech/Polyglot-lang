"""Standard-library include resolution for the `std:` namespace."""

from __future__ import annotations

from pathlib import Path

import pytest

import include_expander
from ast_nodes import FnDecl, StructDecl
from include_expander import (
    IncludeCycleError,
    IncludeNotFoundError,
    expand_includes,
    load_program,
)
from lexer import lex
from parser import parse


def _stdlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "stdlib"
    root.mkdir()
    monkeypatch.setattr(include_expander, "STDLIB_DIR", root)
    return root


def test_std_include_resolves_and_appends_poly(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "math.poly").write_text("fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = tmp_path / "main.poly"
    main.write_text('include!("std:math");\nfn main() -> i64 { add(1, 2) }\n')

    program = load_program(main)
    assert [item.name for item in program.items if isinstance(item, FnDecl)] == [
        "add",
        "main",
    ]


def test_std_include_keeps_explicit_poly_suffix(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "math.poly").write_text("fn one() -> i64 { 1 }\n")
    main = tmp_path / "main.poly"
    main.write_text('include!("std:math.poly");\nfn main() -> i64 { one() }\n')

    program = load_program(main)
    assert [item.name for item in program.items if isinstance(item, FnDecl)] == [
        "one",
        "main",
    ]


def test_bare_include_call_is_item_position_only(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "math.poly").write_text("fn one() -> i64 { 1 }\n")
    main = tmp_path / "main.poly"
    main.write_text('include("std:math");\nfn main() -> i64 { one() }\n')

    program = load_program(main)
    assert [item.name for item in program.items if isinstance(item, FnDecl)] == [
        "one",
        "main",
    ]

    bad = tmp_path / "bad.poly"
    bad.write_text('fn bad() -> i64 { include("std:math"); 1 }\n')
    with pytest.raises(Exception, match="item position"):
        load_program(bad)


def test_nested_std_includes_are_expanded(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "base.poly").write_text("fn base() -> i64 { 1 }\n")
    (root / "outer.poly").write_text(
        'include!("std:base");\nfn outer() -> i64 { base() + 1 }\n'
    )
    main = tmp_path / "main.poly"
    main.write_text('include!("std:outer");\nfn main() -> i64 { outer() }\n')

    names = [item.name for item in load_program(main).items if isinstance(item, FnDecl)]
    assert names == ["base", "outer", "main"]


def test_std_includes_are_deduplicated(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "base.poly").write_text("struct Value { n: i64 }\nfn base() -> i64 { 1 }\n")
    (root / "left.poly").write_text(
        'include!("std:base");\nfn left() -> i64 { base() }\n'
    )
    (root / "right.poly").write_text(
        'include!("std:base");\nfn right() -> i64 { base() }\n'
    )
    main = tmp_path / "main.poly"
    main.write_text(
        'include!("std:left");\n'
        'include!("std:right");\n'
        "fn main() -> i64 { left() + right() }\n"
    )

    program = load_program(main)
    structs = [item for item in program.items if isinstance(item, StructDecl)]
    fns = [item.name for item in program.items if isinstance(item, FnDecl)]
    assert [item.name for item in structs] == ["Value"]
    assert fns == ["base", "left", "right", "main"]


def test_regular_include_deduplication_is_still_off(tmp_path):
    (tmp_path / "lib.poly").write_text("fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = tmp_path / "main.poly"
    main.write_text(
        'include!("lib.poly");\ninclude!("lib.poly");\nfn main() -> i64 { add(2, 3) }\n'
    )
    names = [item.name for item in load_program(main).items if isinstance(item, FnDecl)]
    assert names.count("add") == 2


def test_std_include_cycle_is_rejected(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "loop.poly").write_text('include!("std:loop");\n')
    main = tmp_path / "main.poly"
    main.write_text('include!("std:loop");\n')

    with pytest.raises(IncludeCycleError) as excinfo:
        load_program(main)
    assert excinfo.value.code == "E005"
    assert "cycle" in str(excinfo.value)


@pytest.mark.parametrize(
    "path",
    [
        "std:../main.py",
        "std:sub/../../escape",
        "std:/etc/passwd",
    ],
)
def test_std_traversal_is_rejected(monkeypatch, tmp_path, path):
    _stdlib(monkeypatch, tmp_path)
    main = tmp_path / "main.poly"
    main.write_text(f'include!("{path}");\n')

    with pytest.raises(IncludeNotFoundError) as excinfo:
        load_program(main)
    error = str(excinfo.value)
    assert excinfo.value.code == "E006"
    assert "escapes" in error or "relative to" in error
    assert len([line for line in error.splitlines() if line.strip()]) == 5


def test_std_missing_module_is_rejected(monkeypatch, tmp_path):
    _stdlib(monkeypatch, tmp_path)
    main = tmp_path / "main.poly"
    main.write_text('include!("std:nonexistent");\n')

    with pytest.raises(IncludeNotFoundError) as excinfo:
        load_program(main)
    error = str(excinfo.value)
    assert "missing std module" in error
    assert len([line for line in error.splitlines() if line.strip()]) == 5


def test_expand_includes_uses_std_namespace(monkeypatch, tmp_path):
    root = _stdlib(monkeypatch, tmp_path)
    (root / "math.poly").write_text("fn one() -> i64 { 1 }\n")
    main = tmp_path / "main.poly"
    source = 'include!("std:math");\nfn main() -> i64 { one() }\n'
    main.write_text(source)

    program = expand_includes(parse(lex(source)), main)
    assert [item.name for item in program.items if isinstance(item, FnDecl)] == [
        "one",
        "main",
    ]
