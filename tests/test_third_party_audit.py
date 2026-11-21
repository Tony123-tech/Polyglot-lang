"""Read-only automated checks backing docs/third_party_audit.md."""

from __future__ import annotations

import ast
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

# For a test under tests/, parents[1] is the workspace root. parents[2] is the
# shared parent of this workspace and the sibling checkout, so it cannot be used
# for an "outside this workspace" assertion.
WORKSPACE_ROOT = Path(__file__).parents[1]
AUDIT_TEST_FILE = Path(__file__).resolve()

DFMACRO_ROOT = Path("/Users/waika/dfmacro")
DFMACRO_INIT = DFMACRO_ROOT / "dfmacro" / "__init__.py"
DFMACRO_GENERATED = DFMACRO_ROOT / "dfmacro" / "_generated_methods.py"
DFMACRO_GENERATOR = DFMACRO_ROOT / "generate_methods.py"
DFMACRO_PYPROJECT = DFMACRO_ROOT / "pyproject.toml"
DFMACRO_EXPECTED_VERSION = "0.0.1"
DFMACRO_EXPECTED_ALL_SIZE = 13
DFMACRO_EXPECTED_EXPORTS = frozenset(
    {
        "DF",
        "ColumnExpr",
        "Literal",
        "compile_to_sql",
        "enable",
        "generate_pytest_code",
        "profile",
        "register_accessor",
        "register_backend",
        "save_profile_report",
        "validate_dataframe",
        "when",
        "write_stub_to_file",
    }
)

PYFRAME_SEARCH_ROOT = Path("/Users/waika")
PYFRAME_SEARCH_MAX_DEPTH = 3
PYFRAME_PRUNE_DIRS = frozenset({".venv", ".git", "node_modules"})
PYFRAME_CODE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".rs",
        ".c",
        ".h",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sh",
        ".go",
        ".java",
        ".rb",
    }
)
PYFRAME_MAX_FILE_BYTES = 4 * 1024 * 1024
PYFRAME_NEEDLE = "PyFrameRestart"

TERMINAL_REPO_ROOT = Path("/Users/waika/tony123-tech.github.io")
TERMINAL_APP = TERMINAL_REPO_ROOT / "pycause" / "app.js"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.name)


def _all_names(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            return [
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    return []


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _registry_method_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if (
            not isinstance(node.target, ast.Name)
            or node.target.id != "ACCESSOR_METHODS"
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        names: set[str] = set()
        for key in node.value.keys:
            if not isinstance(key, ast.Tuple) or len(key.elts) != 2:
                continue
            method = key.elts[1]
            if isinstance(method, ast.Constant) and isinstance(method.value, str):
                names.add(method.value)
        return names
    return set()


@pytest.fixture(scope="module")
def init_tree() -> ast.Module:
    if not DFMACRO_INIT.is_file():
        pytest.skip(
            "dfmacro sibling checkout is not present at the configured location"
        )
    return _parse(DFMACRO_INIT)


@pytest.fixture(scope="module")
def generated_tree() -> ast.Module:
    if not DFMACRO_INIT.is_file():
        pytest.skip(
            "dfmacro sibling checkout is not present at the configured location"
        )
    assert DFMACRO_GENERATED.is_file(), "generated methods module missing from dfmacro"
    return _parse(DFMACRO_GENERATED)


def test_dfmacro_all_size_uniqueness_and_exports(init_tree: ast.Module) -> None:
    exports = _all_names(init_tree)
    assert len(exports) == DFMACRO_EXPECTED_ALL_SIZE
    assert len(set(exports)) == len(exports), "duplicate entry in dfmacro __all__"
    assert set(exports) == DFMACRO_EXPECTED_EXPORTS


def test_dfmacro_all_exports_have_module_level_definitions(
    init_tree: ast.Module,
) -> None:
    exports = set(_all_names(init_tree))
    assert exports <= _top_level_names(init_tree), "missing export in dfmacro"


def test_dfmacro_generated_files_exist_and_define_names(
    generated_tree: ast.Module,
) -> None:
    assert DFMACRO_GENERATOR.is_file(), "generate_methods.py missing from dfmacro"
    assert DFMACRO_GENERATED.is_file(), "_generated_methods.py missing from dfmacro"
    names = _function_names(generated_tree) | _registry_method_names(generated_tree)
    assert names, "generated module defines no functions or registry method names"


def test_dfmacro_generated_module_is_a_data_table(generated_tree: ast.Module) -> None:
    assert not _function_names(generated_tree), (
        "generated module now emits def statements; update the audit parity check"
    )
    assert _registry_method_names(generated_tree), "ACCESSOR_METHODS registry is empty"


def test_dfmacro_generated_names_do_not_leak_into_all(
    init_tree: ast.Module, generated_tree: ast.Module
) -> None:
    exports = set(_all_names(init_tree))
    generated = _function_names(generated_tree) | _registry_method_names(generated_tree)
    assert not (exports & generated), "internal generated name exported via __all__"
    assert not (exports - _top_level_names(init_tree)), "missing export in dfmacro"


def test_dfmacro_metadata_declares_version_and_pandas() -> None:
    if not DFMACRO_PYPROJECT.is_file():
        pytest.skip("dfmacro pyproject.toml is not present at the configured location")
    project = tomllib.loads(DFMACRO_PYPROJECT.read_text(encoding="utf-8")).get(
        "project", {}
    )
    assert project.get("version") == DFMACRO_EXPECTED_VERSION
    dependencies = project.get("dependencies", [])
    assert any("pandas" in dep for dep in dependencies), (
        "pandas dependency not declared"
    )


def _ignore_os_error(_error: OSError) -> None:
    return None


def _bounded_source_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_ignore_os_error
    ):
        relative_dir = Path(dirpath).relative_to(root)
        depth = 0 if relative_dir == Path(".") else len(relative_dir.parts)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in PYFRAME_PRUNE_DIRS and depth < PYFRAME_SEARCH_MAX_DEPTH
        )
        for name in filenames:
            path = Path(dirpath) / name
            if path == AUDIT_TEST_FILE:
                continue
            if len(path.relative_to(root).parts) > PYFRAME_SEARCH_MAX_DEPTH:
                continue
            if path.suffix.lower() not in PYFRAME_CODE_SUFFIXES:
                continue
            yield path


def test_pyframere_start_absent_from_bounded_source_search() -> None:
    if not PYFRAME_SEARCH_ROOT.is_dir():
        pytest.skip("bounded search root is not available")
    needle = PYFRAME_NEEDLE.encode("utf-8")
    hits = 0
    for path in _bounded_source_files(PYFRAME_SEARCH_ROOT):
        try:
            if path.stat().st_size > PYFRAME_MAX_FILE_BYTES:
                continue
            if needle in path.read_bytes():
                hits += 1
        except OSError:
            continue
    assert hits == 0, "unexpected identifier occurrence in bounded source search"


def test_terminal_app_exists_outside_workspace() -> None:
    if not TERMINAL_REPO_ROOT.is_dir():
        pytest.skip("external terminal UI checkout is not present")
    assert TERMINAL_APP.is_file(), "terminal app missing at the configured path"
    assert not TERMINAL_APP.resolve().is_relative_to(WORKSPACE_ROOT.resolve())
