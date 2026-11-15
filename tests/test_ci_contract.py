"""Guard against accidental removal of required CI steps."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

REQUIRED_STEPS = [
    "Lint statement leaks",
    "Lint with ruff",
    "Lint docs",
    "Check docs sync",
    "Run fast tests",
    "Run slow tests",
]


def test_required_ci_steps_are_present():
    text = WORKFLOW.read_text(encoding="utf-8")
    for step in REQUIRED_STEPS:
        assert f"name: {step}" in text, f"missing CI step: {step}"


def test_ci_runs_format_check_and_docs_sync_script():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ruff format --check ." in text
    assert "python3 tools/check_docs_sync.py" in text


def test_slow_tests_have_a_timeout_and_cache_artifact():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "timeout-minutes: 20" in text
    assert "actions/upload-artifact" in text
    assert ".pytest_cache" in text
