"""Contract tests for the 1,000-function rollout manifest and frontend errors."""

from __future__ import annotations

import pytest

from compiler import framework_manifest as manifest
from compiler.frontend.errors import ERROR_CODES, PolyglotError, frontend_fail
from runtime.diagnostics import PillarError


def test_manifest_has_one_thousand_gap_free_entries():
    assert len(manifest.FUNCTIONS) == 1000
    assert [entry.number for entry in manifest.FUNCTIONS] == list(range(1, 1001))
    assert [entry.id for entry in manifest.FUNCTIONS] == [
        f"{number:04d}" for number in range(1, 1001)
    ]
    manifest.validate()


def test_every_batch_is_fully_implemented_with_coverage():
    for batch in range(1, 11):
        entries = list(manifest.iter_batch(batch))
        assert len(entries) == 100
        assert all(entry.is_implemented for entry in entries)
        assert all(entry.name and entry.test_module for entry in entries)
        assert len({entry.name for entry in entries}) == 100
        assert manifest.implemented_count(batch=batch) == 100


def test_batch_modules_match_their_declared_areas():
    prefixes = {
        1: "compiler.frontend",
        2: "compiler.types",
        3: "compiler.ir",
        4: "backends.python",
        5: "backends.rust",
        6: "zig",
        7: "runtime.hotpatch",
        8: "runtime.concurrency",
        9: "diagnostics",
        10: "tools.build_ci",
    }
    for batch, prefix in prefixes.items():
        assert all(
            entry.module.startswith(prefix) for entry in manifest.iter_batch(batch)
        ), batch
    assert not any(
        entry.module.startswith("backends.zig") for entry in manifest.FUNCTIONS
    )


def test_progress_coverage_and_next_batch_are_machine_verifiable():
    summary = manifest.coverage_summary()
    assert summary == {
        "total": 1000,
        "implemented": 1000,
        "pending": 0,
        "with_test_coverage": 1000,
        "batches_completed": 10,
    }
    assert manifest.next_batch() is None
    assert manifest.implemented_count() == 1000


def test_lookup_helpers_resolve_ids_numbers_and_names():
    assert manifest.get("0001").name == "lex_source_with_spans"
    assert manifest.get(1) is manifest.get_by_number(1)
    assert manifest.get(100).id == "0100"
    assert manifest.get(101).status == "implemented"
    assert manifest.get(1000).id == "1000"
    with pytest.raises(KeyError):
        manifest.get("9999")
    with pytest.raises(KeyError):
        manifest.get_by_number(0)


def test_all_entries_resolve_to_documented_callables():
    for entry in manifest.FUNCTIONS:
        assert callable(entry.callable), entry.id
        assert entry.callable.__doc__, entry.id


def test_polyglot_error_uses_the_shared_e001_to_e008_surface():
    assert set(ERROR_CODES) == {f"E{code:03d}" for code in range(1, 9)}
    error = PolyglotError("E003", "bad token", line=2, col=5, span=(3, 7))
    assert isinstance(error, PillarError)
    assert error.code == "E003"
    assert error.line == 2
    assert error.col == 5
    assert error.span == (3, 7)
    assert error.phase == "frontend"
    rendered = error.pretty()
    assert "error[E003]: bad token" in rendered
    assert "line 2, column 5" in rendered


def test_polyglot_error_rejects_unknown_codes_and_frontend_fail_raises():
    with pytest.raises(ValueError, match="unknown diagnostic code"):
        PolyglotError("E009", "not a public code")
    with pytest.raises(PolyglotError) as caught:
        frontend_fail("E008", "malformed literal", span=(0, 3))
    assert caught.value.code == "E008"
    assert caught.value.span == (0, 3)
