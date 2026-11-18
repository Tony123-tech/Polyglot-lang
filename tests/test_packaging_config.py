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


def test_include_expander_is_shipped_as_top_level_module():
    modules = _pyproject()["tool"]["setuptools"]["py-modules"]
    assert "include_expander" in modules


def test_monomorphizer_is_shipped_as_top_level_module():
    modules = _pyproject()["tool"]["setuptools"]["py-modules"]
    assert "monomorph" in modules
