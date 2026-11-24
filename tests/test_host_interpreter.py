"""Host interpreter contract: a declared floor that something actually checks.

SPEC 9.9 names ``Python 3.14.8`` as the floor. This module pins that the floor
lives in one place (``diagnostics.host``), that the packaging metadata agrees
with it, that an older interpreter is refused with two readable lines instead
of an internal ``NameError``, and that README/SPEC/GUIDE keep stating it.

What is deliberately not pinned: what Python 3.15 does with the front end, and
whether a future compatibility layer lowers the floor. Neither can be proven
here; both would need a host-version matrix, which does not exist yet.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

from diagnostics.host import (
    MIN_LABEL,
    MIN_VERSION,
    UnsupportedInterpreterError,
    is_supported,
    require_supported_python,
)

ROOT = Path(__file__).resolve().parent.parent


def test_the_running_interpreter_is_accepted():
    # The suite only runs on a supported host; if this fails, the rest of the
    # project's conformance claims are meaningless on this machine.
    require_supported_python()
    assert is_supported()


def test_python_312_is_refused():
    with pytest.raises(UnsupportedInterpreterError) as raised:
        require_supported_python((3, 12, 4))
    assert MIN_LABEL in raised.value.message
    assert "Python 3.12.4" in raised.value.message


def test_floor_is_the_declared_patch_version_not_the_minor():
    # 3.14.0-3.14.7 are below the declared floor; the guard mirrors packaging
    # rather than accepting any 3.14.
    with pytest.raises(UnsupportedInterpreterError):
        require_supported_python((3, 14, 0))


@pytest.mark.parametrize("version", [(3, 14, 8), (3, 14, 9), (3, 15, 0)])
def test_floor_and_above_are_accepted(version):
    require_supported_python(version)
    assert is_supported(version)


def test_refusal_names_the_reason():
    with pytest.raises(UnsupportedInterpreterError) as raised:
        require_supported_python((3, 11, 9))
    assert "PEP 649" in raised.value.help_text
    assert MIN_LABEL in raised.value.help_text
    assert "README" in raised.value.help_text


def test_pyproject_requires_python_matches_the_floor():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        requires = tomllib.load(handle)["project"]["requires-python"]
    floor = re.fullmatch(r">=\s*(\d+)\.(\d+)\.(\d+)", requires.strip())
    assert floor, f"requires-python is not a plain floor: {requires!r}"
    assert tuple(int(part) for part in floor.groups()) == MIN_VERSION


def test_cli_refuses_before_parsing_arguments(monkeypatch, capsys):
    from api.cli import main
    from diagnostics import host

    def refuse(version_info=None):
        require_supported_python((3, 12, 4))

    monkeypatch.setattr(host, "require_supported_python", refuse)
    # `explain E001` would exit 0 on a supported host; a refusal must win over
    # argument parsing, so the exit code is the guard's 2 and not the command's.
    assert main(["explain", "E001"]) == 2
    err = capsys.readouterr().err
    assert "error: polyglot-lang requires Python 3.14.8" in err
    assert "help:" in err and "PEP 649" in err
    assert "Traceback" not in err


def test_spec_states_the_floor():
    spec = (ROOT / "SPECIFICATION.md").read_text(encoding="utf-8")
    assert "**9.9**" in spec
    assert "Python 3.14.8 or later" in spec
    assert "**9.9.2**" in spec


def test_readme_states_the_floor_and_the_reason():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Python 3.14.8" in readme
    assert "PEP 649" in readme


def test_guide_states_the_floor():
    guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
    assert "Python 3.14.8 or later" in guide


def test_supported_host_still_runs_a_command():
    # The guard must not cost anything on the platform the project claims:
    # `explain` is a pure-metadata path, so it cannot depend on a toolchain.
    from api.cli import main

    assert main(["explain", "E009"]) == 0
    assert sys.version_info >= MIN_VERSION
