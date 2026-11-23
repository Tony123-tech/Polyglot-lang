"""Packaging metadata contract tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_requires_python_matches_supported_runtime():
    project = _pyproject()["project"]
    assert project["requires-python"] == ">=3.14.8"


def test_version_matches_specification_title():
    project = _pyproject()["project"]
    title = (ROOT / "SPECIFICATION.md").read_text(encoding="utf-8").splitlines()[0]
    assert title == f"# polyglot-lang Specification v{project['version']}"


def test_console_script_points_at_cli_entry_point():
    project = _pyproject()["project"]
    assert project["scripts"]["polyglot"] == "api.cli:main"


def test_compiler_stages_ship_inside_the_package():
    # The stages used to be top-level modules listed in `py-modules`; they now
    # live in `compiler/core` and `compiler/analysis`, which the package finder
    # already includes via `compiler*`.
    modules = _pyproject()["tool"]["setuptools"]["py-modules"]
    includes = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    for moved in (
        "lexer",
        "parser",
        "ast_nodes",
        "include_expander",
        "inline_lang",
        "monomorph",
        "termination",
        "struct_order",
        "match_diagnostics",
    ):
        assert moved not in modules, f"{moved} is still a top-level module"
    assert "compiler*" in includes
    assert (ROOT / "compiler" / "core" / "lexer.py").exists()
    assert (ROOT / "compiler" / "analysis" / "monomorph.py").exists()


def test_root_directory_holds_no_compiler_stage():
    for moved in (
        "lexer",
        "parser",
        "ast_nodes",
        "include_expander",
        "inline_lang",
        "monomorph",
        "termination",
        "struct_order",
        "match_diagnostics",
    ):
        assert not (ROOT / f"{moved}.py").exists(), f"{moved}.py came back"
