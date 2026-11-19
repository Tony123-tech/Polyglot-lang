"""Sandboxed meta-compiler foundations."""

from __future__ import annotations

import subprocess

import pytest

from compiler.evolution import Evaluation, EvolutionEngine, PassCandidate
from runtime.diagnostics import PillarError

CORPUS = [
    "fn main() -> i64 { 5 + 0 }",
    "fn main() -> i64 { 0 + 7 }",
    "fn main() -> i64 { 3 * 1 }",
    "fn main() -> i64 { if true { 11 } else { 12 } }",
]


def test_engine_requires_an_observation(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    with pytest.raises(PillarError) as excinfo:
        engine.propose()
    assert excinfo.value.code == "E003"


def test_engine_proposes_and_validates_a_candidate(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    engine.observe_compile_failure("error[E004]: unknown pattern")
    candidate = engine.propose()
    assert candidate.template == "fold_add_zero"
    assert candidate.parent is None
    assert "def transform" in candidate.source

    evaluation = engine.evaluate(candidate, CORPUS)
    assert evaluation.ok, evaluation.error
    assert len(evaluation.results) == len(CORPUS)
    assert all(item["expected"] == item["got"] for item in evaluation.results)
    assert evaluation.candidate_ms >= 0.0


def test_engine_promotes_versions_and_rolls_back(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    engine.observe_compile_failure("boom")
    first = engine.propose()
    evaluation = engine.evaluate(first, CORPUS)
    assert engine.promote(first, evaluation) == 1
    assert engine.current() == first

    engine.observe_profile("slow_fn", 50.0, 10.0)
    second = engine.propose()
    assert second.version == 2
    assert second.parent == 1
    evaluation = engine.evaluate(second, CORPUS)
    assert engine.promote(second, evaluation) == 2
    assert [item["version"] for item in engine.history()] == [1, 2]

    restored = engine.rollback(1)
    assert restored.version == 1
    assert engine.current().version == 1


def test_engine_rejects_a_failing_candidate(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    engine.observe_compile_failure("boom")
    candidate = PassCandidate(
        name="broken",
        version=1,
        template="broken",
        source='def transform(node):\n    raise RuntimeError("boom")\n',
    )
    evaluation = engine.evaluate(candidate, CORPUS)
    assert not evaluation.ok
    assert "boom" in (evaluation.error or "")
    with pytest.raises(PillarError) as excinfo:
        engine.promote(candidate, evaluation)
    assert excinfo.value.code == "E002"


def test_engine_times_out_hostile_candidate(tmp_path):
    engine = EvolutionEngine(tmp_path / "history", timeout=0.5)
    candidate = PassCandidate(
        name="hang",
        version=1,
        template="hang",
        source="def transform(node):\n    while True:\n        pass\n",
    )
    evaluation = engine.evaluate(candidate, CORPUS[:1])
    assert not evaluation.ok
    assert "timed out" in (evaluation.error or "")


def test_engine_rejects_syntax_errors_before_the_sandbox(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    engine.observe_compile_failure("boom")
    candidate = PassCandidate(
        name="syntax",
        version=1,
        template="syntax",
        source="def transform(node) return None\n",
    )
    with pytest.raises(PillarError) as excinfo:
        engine.evaluate(candidate, CORPUS)
    assert excinfo.value.code == "E004"


def test_missing_history_version_is_e006(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    with pytest.raises(PillarError) as excinfo:
        engine.rollback(99)
    assert excinfo.value.code == "E006"


def test_corrupt_history_pointer_is_e008(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    (history / "current.json").write_text("{not json", encoding="utf-8")
    engine = EvolutionEngine(history)
    with pytest.raises(PillarError) as excinfo:
        engine.current()
    assert excinfo.value.code == "E008"


def test_evaluation_dataclass_reports_speed(tmp_path):
    evaluation = Evaluation(ok=True, candidate_ms=1.0, baseline_ms=2.0)
    assert evaluation.faster
    assert not Evaluation(ok=False, candidate_ms=1.0, baseline_ms=2.0).faster


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef transform(node):\n    return os.getcwd()\n",
        "def transform(node):\n    return __import__('os').getcwd()\n",
        "def transform(node):\n    return node.__class__\n",
        "def transform(node):\n    return open('/etc/passwd').read()\n",
        "print('side effect')\ndef transform(node):\n    return None\n",
    ],
)
def test_unsafe_candidate_source_is_rejected(tmp_path, source):
    engine = EvolutionEngine(tmp_path / "history")
    candidate = PassCandidate("unsafe", 1, "unsafe", source)
    with pytest.raises(PillarError) as excinfo:
        engine.evaluate(candidate, CORPUS[:1])
    assert excinfo.value.code == "E003"


def test_history_is_append_only_and_writes_are_atomic(tmp_path):
    engine = EvolutionEngine(tmp_path / "history")
    engine.observe_compile_failure("boom")
    first = engine.propose()
    engine.promote(first, Evaluation(ok=True))

    changed = PassCandidate(
        name="different",
        version=1,
        template="different",
        source="def transform(node):\n    return 1\n",
    )
    with pytest.raises(PillarError) as excinfo:
        engine.promote(changed, Evaluation(ok=True))
    assert excinfo.value.code == "E008"
    assert engine.current() == first
    assert not list((tmp_path / "history").glob(".*.tmp"))
    assert list((tmp_path / "history").glob(".v*.py.*")) == []


@pytest.mark.parametrize("version", [0, -1, "1", True])
def test_invalid_history_versions_are_e003(tmp_path, version):
    engine = EvolutionEngine(tmp_path / "history")
    with pytest.raises(PillarError) as excinfo:
        engine.rollback(version)
    assert excinfo.value.code == "E003"


def test_manifest_missing_required_fields_is_e008(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    (history / "v0001.json").write_text('{"name": "partial"}', encoding="utf-8")
    engine = EvolutionEngine(history)
    with pytest.raises(PillarError) as excinfo:
        engine.history()
    assert excinfo.value.code == "E008"


def test_git_commit_stages_only_promoted_files(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("compiler.evolution.shutil.which", lambda name: "/usr/bin/git")

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("compiler.evolution.subprocess.run", fake_run)
    engine = EvolutionEngine(tmp_path / "history", use_git=True)
    engine.observe_compile_failure("boom")
    candidate = engine.propose()
    engine.promote(candidate, Evaluation(ok=True))

    add_calls = [call for call in calls if "add" in call]
    assert len(add_calls) == 1
    assert "--" in add_calls[0]
    assert {"v0001.py", "v0001.json", "current.json"} <= set(add_calls[0])
    assert "." not in add_calls[0]
