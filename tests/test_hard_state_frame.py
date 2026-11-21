"""Focused tests for Frame J hard state analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.diagnostics import PillarError
from runtime.hard_state import (
    Access,
    FieldSpec,
    MemoryObject,
    PointerFact,
    Schema,
    hard_analyze_points_to_and_alias_sets,
    hard_plan_live_state_schema_migration,
    hard_quarantine_and_replay_faults,
)

SPAN = (3, 9)


def test_points_to_and_toctou_hazards_are_explicit() -> None:
    obj = MemoryObject("buf", "rust", "Vec<u8>", 64)
    left = PointerFact("p", "rust", "borrow", ("buf",), length=32)
    right = PointerFact("q", "zig", "slice", ("buf",), offset=16, length=32)

    report = hard_analyze_points_to_and_alias_sets(
        [obj],
        [left, right],
        [Access("check", "p"), Access("write", "q"), Access("use", "p")],
        span=SPAN,
    )

    assert report.alias_sets == (("p", "q"),)
    assert {hazard.kind for hazard in report.hazards} == {
        "mutable-alias",
        "toctou",
    }
    assert not report.safe
    assert "zero aliasing" in report.as_dict()["caveats"][0]

    with pytest.raises(PillarError) as raised:
        hard_analyze_points_to_and_alias_sets(
            [obj],
            [left, right],
            [Access("check", "p"), Access("write", "q")],
            strict=True,
            span=SPAN,
        )
    assert raised.value.code == "E008"
    assert raised.value.span == SPAN


def test_epoch_change_invalidates_pending_check() -> None:
    obj = MemoryObject("cell", "python", "Cell", 8)
    pointer = PointerFact("ref", "python", "reference", ("cell",))
    report = hard_analyze_points_to_and_alias_sets(
        [obj],
        [pointer],
        [Access("check", "ref"), Access("epoch", "", value=2), Access("use", "ref")],
        epoch=1,
    )

    assert report.epoch == 2
    assert [hazard.kind for hazard in report.hazards] == ["toctou"]
    assert "epoch changed" in report.hazards[0].message


def test_assignment_widens_points_to_set() -> None:
    first = MemoryObject("a", "python", "A", 8)
    second = MemoryObject("b", "python", "B", 8)
    p = PointerFact("p", "python", "reference", ("a",))
    q = PointerFact("q", "python", "reference", ("b",))

    report = hard_analyze_points_to_and_alias_sets(
        [first, second], [p, q], [Access("assign", "p", source="q")]
    )

    assert report.points_to["p"] == ("a", "b")


def test_map_jit_and_raw_ffi_are_boundary_diagnostics() -> None:
    jit_object = MemoryObject("code", "rust", "Code", 4096, mapped_jit=True)
    jit_pointer = PointerFact("code_ref", "rust", "raw", ("code",))
    alias_pointer = PointerFact("code_alias", "rust", "borrow", ("code",))
    with pytest.raises(PillarError) as raised:
        hard_analyze_points_to_and_alias_sets(
            [jit_object], [jit_pointer, alias_pointer], strict=True
        )
    assert raised.value.code == "E008"

    ffi_object = MemoryObject("foreign", "zig", "Extern", 16, model_only=False)
    ffi_pointer = PointerFact("raw", "zig", "raw", ("foreign",))
    ffi_alias = PointerFact("slice", "zig", "slice", ("foreign",))
    with pytest.raises(PillarError) as raised:
        hard_analyze_points_to_and_alias_sets(
            [ffi_object], [ffi_pointer, ffi_alias], strict=True
        )
    assert raised.value.code == "E007"


def test_missing_memory_object_is_e006() -> None:
    with pytest.raises(PillarError) as raised:
        hard_analyze_points_to_and_alias_sets(
            [], [PointerFact("p", "rust", "raw", ("missing",))]
        )
    assert raised.value.code == "E006"


def test_safe_migration_plan_has_total_mapping_and_rollback() -> None:
    old = Schema(
        1,
        "struct",
        (FieldSpec("x", "i32"), FieldSpec("name", "str")),
    )
    new = Schema(
        2,
        "struct",
        (FieldSpec("x", "i64"), FieldSpec("label", "str")),
    )

    plan = hard_plan_live_state_schema_migration(old, new, renames={"label": "name"})

    assert plan.safe and plan.atomic
    assert [step.action for step in plan.steps] == ["convert", "rename"]
    assert plan.rollback
    assert plan.digest() == plan.as_dict()["digest"]
    assert "zero-risk" not in str(plan.as_dict()["caveats"])


def test_lossy_and_ambiguous_migrations_are_e008() -> None:
    old = Schema(1, "struct", (FieldSpec("x", "i32"), FieldSpec("y", "i32")))
    lossy = Schema(2, "struct", (FieldSpec("x", "i32"),))
    with pytest.raises(PillarError) as raised:
        hard_plan_live_state_schema_migration(old, lossy, span=SPAN)
    assert raised.value.code == "E008"
    assert "required source field" in raised.value.diagnostic.reason

    ambiguous = Schema(2, "struct", (FieldSpec("a", "i32"), FieldSpec("b", "i32")))
    with pytest.raises(PillarError) as raised:
        hard_plan_live_state_schema_migration(
            old, ambiguous, mapping={"a": "x", "b": "x"}
        )
    assert raised.value.code == "E008"

    enum_old = Schema(1, "enum", (), variants=("A", "B"))
    enum_new = Schema(2, "enum", (), variants=("A",))
    with pytest.raises(PillarError) as raised:
        hard_plan_live_state_schema_migration(enum_old, enum_new)
    assert raised.value.code == "E008"


def test_allow_lossy_keeps_diagnostics_without_applying_plan() -> None:
    old = Schema(1, "struct", (FieldSpec("x", "i32"), FieldSpec("y", "i32")))
    new = Schema(2, "struct", (FieldSpec("x", "i32"),))
    plan = hard_plan_live_state_schema_migration(old, new, allow_lossy=True)

    assert not plan.safe
    assert not plan.atomic
    assert any(item["kind"] == "lossy" for item in plan.diagnostics)


def test_schema_kind_and_version_transitions_are_guarded() -> None:
    old = Schema(2, "struct", (FieldSpec("x", "i32"),))
    with pytest.raises(PillarError) as raised:
        hard_plan_live_state_schema_migration(
            old, Schema(3, "table", (FieldSpec("x", "i32"),))
        )
    assert raised.value.code == "E008"

    with pytest.raises(PillarError) as raised:
        hard_plan_live_state_schema_migration(
            old, Schema(1, "struct", (FieldSpec("x", "i32"),))
        )
    assert raised.value.code == "E008"


def test_quarantine_replay_is_deterministic_and_scoped(tmp_path: Path) -> None:
    calls = 0

    def replay(snapshot: dict[str, object]) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        return (int(snapshot["value"]) + 1,)

    capsule, report = hard_quarantine_and_replay_faults(
        "tx-1",
        {"value": 1},
        {"epoch": 4, "expected_outcomes": (2,)},
        replay=replay,
        tmp_root=tmp_path,
        span=SPAN,
    )

    assert calls == 2
    assert report.deterministic and report.matched
    assert report.outcomes == (2,)
    assert len(report.artifacts) == 1
    artifact = Path(report.artifacts[0])
    assert artifact.is_file()
    assert artifact.is_relative_to(tmp_path)
    assert capsule.digest in artifact.read_text(encoding="utf-8")
    assert capsule.jit_mode == "model"


def test_quarantine_boundaries_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(PillarError) as raised:
        hard_quarantine_and_replay_faults("tx", {}, {}, jit_mode="native", span=SPAN)
    assert raised.value.code == "E007"

    with pytest.raises(PillarError) as raised:
        hard_quarantine_and_replay_faults(
            "tx", {"raw_address": 1}, {}, model_only=False
        )
    assert raised.value.code == "E007"

    _, report = hard_quarantine_and_replay_faults(
        "tx", {"value": 1}, {}, pep667_frame=True, tmp_root=tmp_path
    )
    assert report.diagnostics[0]["code"] == "E008"
    assert "PEP 667" in report.diagnostics[0]["message"]
    assert report.artifacts
    assert all(Path(item).is_relative_to(tmp_path) for item in report.artifacts)
