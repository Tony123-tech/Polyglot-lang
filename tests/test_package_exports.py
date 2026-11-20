"""Public export, version, and packaging-configuration invariants.

These tests keep the source-tree surface aligned with what a wheel ships: the
100-function registry must resolve for every entry, every declared ``__all__``
must be real, and every registry module must be covered by the setuptools
package include list.
"""

from __future__ import annotations

import fnmatch
import importlib
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_registry_resolves_all_100_callables() -> None:
    framework_functions = importlib.import_module("framework_functions")
    framework_functions.validate_registry()
    assert len(framework_functions.FUNCTIONS) == 100


def test_every_registry_module_is_included_in_the_wheel() -> None:
    config = _pyproject()
    patterns = config["tool"]["setuptools"]["packages"]["find"]["include"]
    # Check full module paths: the namespace parent "tools" itself is not
    # shipped, only the "tools.build_ci" subpackage that owns functions 81-90.
    modules = {
        item.module for item in importlib.import_module("framework_functions").FUNCTIONS
    }
    for module in sorted(modules):
        assert any(fnmatch.fnmatch(module, pattern) for pattern in patterns), module
    assert "tools.build_ci*" in patterns


@pytest.mark.parametrize(
    "module_name",
    [
        "api",
        "backends",
        "backends.python",
        "backends.rust",
        "compiler.types",
        "diagnostics",
        "docs",
        "framework_functions",
        "regression",
        "runtime",
        "runtime.hotpatch",
        "tools.build_ci",
    ],
)
def test_all_declares_real_attributes(module_name: str) -> None:
    module = importlib.import_module(module_name)
    declared = module.__all__
    assert isinstance(declared, list)
    assert len(declared) == len(set(declared))
    missing = [name for name in declared if not hasattr(module, name)]
    assert missing == [], f"{module_name}: missing exports {missing}"


def test_version_metadata_is_consistent() -> None:
    config = _pyproject()
    declared = config["project"]["version"]
    visualizer = importlib.import_module("api.visualizer_server")
    assert visualizer.VERSION == declared
    try:
        installed = version(config["project"]["name"])
    except PackageNotFoundError:
        return
    assert installed == declared


def test_public_backends_import_without_a_native_toolchain() -> None:
    backends = importlib.import_module("backends")
    assert set(backends.__all__) == {"Backend", "UntrustedCompileError", "get_backend"}
    assert backends.get_backend("python", trusted=False).name == "python"
    # Constructing the native backend does not invoke cargo; the build step does.
    native_module = importlib.import_module("backends.rust")
    assert native_module.__all__ == ["NativeBackend"]
    backend = native_module.NativeBackend(trusted=False)
    assert backend.name == "native"
    lexer = importlib.import_module("lexer")
    parser = importlib.import_module("parser")
    program = parser.parse(lexer.lex("fn package_probe() -> i64 { 1 }"))
    with pytest.raises(backends.UntrustedCompileError):
        backend.compile(program)


def test_regression_harness_reexport_matches_canonical_module() -> None:
    canonical = importlib.import_module("regression.harness")
    shim = importlib.import_module("tests.regression_harness")
    assert shim.__all__ == canonical.__all__
    for name in canonical.__all__:
        assert getattr(shim, name) is getattr(canonical, name)
