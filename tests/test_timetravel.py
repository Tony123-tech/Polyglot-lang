"""Deterministic cross-FFI time-travel debugging foundations."""

from __future__ import annotations

import pytest

from debug.timetravel import (
    GraphEntry,
    ObjectGraphSnapshot,
    TimeTravelDebugger,
    _stable_digest,
    capture_python_frame,
    restore_object_graph,
    snapshot_object_graph,
)
from runtime.diagnostics import PillarError


def _record_total(state):
    state["total"] = state["x"]
    return state


def _increment_x(state):
    state["x"] = state["x"] + 1
    return state


def _accumulate(state):
    state["total"] = state["total"] + state["x"]
    return state


def _timeline() -> TimeTravelDebugger:
    debugger = TimeTravelDebugger()
    debugger.trace(
        {"x": 1, "total": 0},
        [_record_total, _increment_x, _accumulate],
        labels=["initial", "total=x", "x++", "total+=x"],
    )
    return debugger


def test_trace_records_every_checkpoint():
    debugger = _timeline()
    assert len(debugger) == 4
    assert debugger.cursor == 3
    assert [snapshot.label for snapshot in debugger.snapshots] == [
        "initial",
        "total=x",
        "x++",
        "total+=x",
    ]
    assert debugger.current().state == {"x": 2, "total": 3}


def test_step_back_and_forward_move_the_cursor():
    debugger = _timeline()
    assert debugger.step_back().state == {"x": 2, "total": 1}
    assert debugger.cursor == 2
    assert debugger.step_back().state == {"x": 1, "total": 1}
    assert debugger.step_forward().state == {"x": 2, "total": 1}
    debugger.travel_to(0)
    assert debugger.current().state == {"x": 1, "total": 0}


def test_step_past_the_ends_is_e006():
    debugger = _timeline()
    debugger.travel_to(0)
    with pytest.raises(PillarError) as excinfo:
        debugger.step_back()
    assert excinfo.value.code == "E006"

    debugger.travel_to(len(debugger) - 1)
    with pytest.raises(PillarError) as excinfo:
        debugger.step_forward()
    assert excinfo.value.code == "E006"


def test_empty_timeline_and_missing_checkpoint_are_e006():
    debugger = TimeTravelDebugger()
    with pytest.raises(PillarError) as excinfo:
        debugger.current()
    assert excinfo.value.code == "E006"
    with pytest.raises(PillarError) as excinfo:
        debugger.travel_to(0)
    assert excinfo.value.code == "E006"


def test_retroactive_edit_recomputes_all_dependent_states():
    debugger = _timeline()
    debugger.modify(0, "x", 10)
    states = [snapshot.state for snapshot in debugger.snapshots]
    assert states == [
        {"x": 10, "total": 0},
        {"x": 10, "total": 10},
        {"x": 11, "total": 10},
        {"x": 11, "total": 21},
    ]
    assert debugger.cursor == 0
    assert debugger.current().state["x"] == 10


def test_retroactive_edit_requires_recorded_transitions():
    debugger = TimeTravelDebugger()
    debugger.checkpoint({"x": 1})
    debugger.checkpoint({"x": 2})
    with pytest.raises(PillarError) as excinfo:
        debugger.modify(0, "x", 9)
    assert excinfo.value.code == "E002"


def test_modifying_an_unknown_variable_is_e006():
    debugger = _timeline()
    with pytest.raises(PillarError) as excinfo:
        debugger.modify(0, "missing", 1)
    assert excinfo.value.code == "E006"


def test_checkpoint_integrity_detects_out_of_band_edits():
    debugger = _timeline()
    snapshot = debugger.snapshots[0]
    snapshot.state["x"] = 99
    with pytest.raises(PillarError) as excinfo:
        snapshot.verify()
    assert excinfo.value.code == "E008"


def test_event_log_records_navigation_and_recomputation():
    debugger = _timeline()
    debugger.step_back()
    debugger.modify(0, "x", 5)
    kinds = [event.kind for event in debugger.events]
    assert "travel" in kinds
    assert "modify" in kinds
    assert kinds.count("recompute") == 3


def test_apply_callback_receives_the_restored_state():
    applied: list[dict] = []
    debugger = TimeTravelDebugger(apply=applied.append)
    debugger.checkpoint({"value": 1})
    debugger.checkpoint({"value": 2})
    debugger.travel_to(0)
    assert applied == [{"value": 1}]


class _Provider:
    """A fake native context with an inspectable register bank."""

    def __init__(self) -> None:
        self.registers = {"pc": 0, "sp": 64}
        self.restored: list[dict] = []

    def capture(self):
        return dict(self.registers)

    def restore(self, state):
        self.registers = dict(state)
        self.restored.append(dict(state))


def test_native_checkpoints_are_captured_and_restored():
    provider = _Provider()
    debugger = TimeTravelDebugger(native=provider)
    debugger.checkpoint({"x": 1})
    provider.registers = {"pc": 8, "sp": 32}
    debugger.checkpoint({"x": 2})

    debugger.travel_to(0)
    assert provider.registers == {"pc": 0, "sp": 64}
    debugger.travel_to(1)
    assert provider.registers == {"pc": 8, "sp": 32}
    assert provider.restored == [
        {"pc": 0, "sp": 64},
        {"pc": 8, "sp": 32},
    ]


def test_retroactive_edit_marks_native_state_stale():
    provider = _Provider()
    debugger = TimeTravelDebugger(native=provider)
    debugger.checkpoint({"x": 1}, transition=lambda state: dict(state))
    provider.registers = {"pc": 8, "sp": 64}
    debugger.checkpoint({"x": 2})
    debugger.modify(0, "x", 10)

    assert debugger.snapshots[1].stale_native
    with pytest.raises(PillarError) as excinfo:
        debugger.travel_to(1)
    assert excinfo.value.code == "E002"
    # Restoring the Python-only checkpoint is still allowed.
    debugger.travel_to(0)


def test_native_replay_recomputes_instead_of_marking_stale():
    provider = _Provider()
    debugger = TimeTravelDebugger(
        native=provider,
        native_replay=lambda state: {"pc": state["x"] * 8, "sp": 64},
    )
    debugger.checkpoint({"x": 1}, transition=lambda state: dict(state))
    debugger.checkpoint({"x": 2})
    debugger.modify(0, "x", 10)

    snapshot = debugger.snapshots[1]
    assert not snapshot.stale_native
    assert snapshot.native == {"pc": 80, "sp": 64}
    debugger.travel_to(1)
    assert provider.registers == {"pc": 80, "sp": 64}


def test_native_differential_reports_changes():
    provider = _Provider()
    debugger = TimeTravelDebugger(native=provider)
    debugger.checkpoint({"x": 1})
    provider.registers = {"pc": 8, "sp": 64, "flags": 1}
    debugger.checkpoint({"x": 2})

    diff = debugger.differential(0, 1)
    assert not diff.empty
    assert diff.changed == {"pc": (0, 8)}
    assert diff.added == {"flags": 1}
    assert "pc: 0 -> 8" in diff.render()


def test_native_differential_requires_native_checkpoints():
    debugger = _timeline()
    with pytest.raises(PillarError) as excinfo:
        debugger.differential(0, 1)
    assert excinfo.value.code == "E006"


def test_capture_python_frame_reads_locals_without_mutating_them():
    def probe():
        local_value = 7
        hidden = {"k": 1}
        captured = capture_python_frame()
        return captured, local_value, hidden

    captured, _local_value, hidden = probe()
    assert captured["__frame__"] == "probe"
    assert captured["local_value"] == 7
    assert captured["hidden"] is hidden


class Node:
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None
        self.tags: list[str] = []
        self.items: list[int] = []


def test_object_graph_snapshot_restores_identity_and_nested_state():
    first = Node(1)
    second = Node(2)
    first.next = second
    second.next = first  # cycle
    first.tags.append("old")
    shared = [1, 2]
    first.items = shared

    snapshot = snapshot_object_graph([first])

    first.value = 99
    first.next = None
    first.tags.append("new")
    shared.append(3)

    restored = restore_object_graph(snapshot)
    assert restored >= 2
    assert first.value == 1
    assert first.next is second
    assert second.next is first
    assert first.tags == ["old"]
    assert shared == [1, 2]
    assert first.items is shared


def test_object_graph_restores_dicts_and_sets():
    mapping = {"a": 1}
    seen = {"x"}
    snapshot = snapshot_object_graph([mapping, seen])
    mapping["a"] = 2
    mapping["b"] = 3
    seen.add("y")

    restore_object_graph(snapshot)
    assert mapping == {"a": 1}
    assert seen == {"x"}


def test_object_graph_metadata():
    node = Node(5)
    snapshot = snapshot_object_graph([node])
    assert snapshot.objects >= 1
    assert snapshot.roots[0] is node


def test_serializer_failure_during_checkpoint_is_e008():
    def broken(value):
        raise RuntimeError("cannot encode")

    debugger = TimeTravelDebugger(serializer=broken)
    with pytest.raises(PillarError) as excinfo:
        debugger.checkpoint({"x": 1})
    assert excinfo.value.code == "E008"
    assert len(debugger) == 0


def test_serializer_failure_during_replay_is_transactional():
    fail = {"on": False}

    def flaky(value):
        if fail["on"]:
            raise RuntimeError("later")
        return value

    debugger = TimeTravelDebugger(serializer=flaky)
    debugger.trace({"x": 1}, [_increment_x, _increment_x])
    before_states = [dict(snapshot.state) for snapshot in debugger.snapshots]
    before_events = list(debugger.events)
    before_cursor = debugger.cursor
    fail["on"] = True

    with pytest.raises(PillarError) as excinfo:
        debugger.modify(0, "x", 10)
    assert excinfo.value.code == "E008"
    assert [dict(snapshot.state) for snapshot in debugger.snapshots] == before_states
    assert list(debugger.events) == before_events
    assert debugger.cursor == before_cursor


def test_native_capture_and_restore_failures_are_e004():
    class BrokenCapture:
        def capture(self):
            raise RuntimeError("cannot capture")

        def restore(self, state):
            raise AssertionError("restore must not be called")

    debugger = TimeTravelDebugger(native=BrokenCapture())
    with pytest.raises(PillarError) as excinfo:
        debugger.checkpoint({"x": 1})
    assert excinfo.value.code == "E004"
    assert len(debugger) == 0

    class BrokenRestore(_Provider):
        def restore(self, state):
            raise RuntimeError("cannot restore")

    debugger = TimeTravelDebugger(native=BrokenRestore())
    debugger.checkpoint({"x": 1})
    with pytest.raises(PillarError) as excinfo:
        debugger.travel_to(0)
    assert excinfo.value.code == "E004"
    assert debugger.cursor == 0


def test_apply_callback_failure_is_e003():
    def broken_apply(state):
        raise RuntimeError("cannot apply")

    debugger = TimeTravelDebugger(apply=broken_apply)
    debugger.checkpoint({"x": 1})
    with pytest.raises(PillarError) as excinfo:
        debugger.travel_to(0)
    assert excinfo.value.code == "E003"


def test_failed_replay_does_not_mutate_the_timeline():
    def fail_when_large(state):
        if state["x"] > 5:
            raise RuntimeError("replay boom")
        state["x"] += 1
        return state

    debugger = TimeTravelDebugger()
    debugger.trace({"x": 1}, [_increment_x, fail_when_large])
    debugger.travel_to(1)
    originals = list(debugger.snapshots)
    before_states = [dict(snapshot.state) for snapshot in originals]
    before_digests = [snapshot.digest for snapshot in originals]
    before_events = list(debugger.events)
    before_cursor = debugger.cursor

    with pytest.raises(PillarError) as excinfo:
        debugger.modify(0, "x", 10)
    assert excinfo.value.code == "E002"
    assert all(
        snapshot is original
        for snapshot, original in zip(debugger.snapshots, originals)
    )
    assert [dict(snapshot.state) for snapshot in debugger.snapshots] == before_states
    assert [snapshot.digest for snapshot in debugger.snapshots] == before_digests
    assert list(debugger.events) == before_events
    assert debugger.cursor == before_cursor
    assert all(snapshot.verify() for snapshot in debugger.snapshots)


def test_failed_native_replay_does_not_mutate_the_timeline():
    provider = _Provider()
    fail = {"on": False}

    def replay(state):
        if fail["on"]:
            raise RuntimeError("native replay boom")
        return {"pc": state["x"], "sp": 64}

    debugger = TimeTravelDebugger(native=provider, native_replay=replay)
    debugger.trace({"x": 1}, [_increment_x, _increment_x])
    before_states = [dict(snapshot.state) for snapshot in debugger.snapshots]
    before_natives = [dict(snapshot.native) for snapshot in debugger.snapshots]
    before_events = list(debugger.events)
    fail["on"] = True

    with pytest.raises(PillarError) as excinfo:
        debugger.modify(0, "x", 10)
    assert excinfo.value.code == "E004"
    assert [dict(snapshot.state) for snapshot in debugger.snapshots] == before_states
    assert [dict(snapshot.native) for snapshot in debugger.snapshots] == before_natives
    assert list(debugger.events) == before_events


def test_stable_digest_sorts_by_key_only():
    class Opaque:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"Opaque({self.name})"

    first = {1: Opaque("a"), "1": Opaque("b")}
    second = {"1": Opaque("b"), 1: Opaque("a")}
    assert _stable_digest(first) == _stable_digest(second)


def test_invalid_graph_entries_are_e008():
    malformed = [
        ObjectGraphSnapshot((), (GraphEntry("bogus", [], []),)),
        ObjectGraphSnapshot((), (GraphEntry("list", {}, []),)),
        ObjectGraphSnapshot((), (GraphEntry("object", object(), {}),)),
    ]
    for snapshot in malformed:
        with pytest.raises(PillarError) as excinfo:
            restore_object_graph(snapshot)
        assert excinfo.value.code == "E008"


def test_frozenset_graph_members_are_restored():
    class Token:
        def __init__(self, value: int) -> None:
            self.value = value

        def __hash__(self) -> int:
            return 1

    token = Token(1)
    frozen = frozenset({token})
    snapshot = snapshot_object_graph([frozen])
    assert any(entry.kind == "frozenset" for entry in snapshot.entries)

    token.value = 99
    restored = restore_object_graph(snapshot)
    assert token.value == 1
    assert restored >= 1
