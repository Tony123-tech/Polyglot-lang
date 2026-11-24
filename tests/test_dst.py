"""Deterministic simulation testing on the polyglot tooling layer."""

from __future__ import annotations

import pytest

from runtime.diagnostics import PillarError
from tools.dst import (
    FAULTS,
    SimOp,
    VirtualClock,
    dst_bisect_and_minimize_schedule,
    dst_inject_byzantine_and_io_faults,
    dst_virtualize_clock_and_scheduler,
)


def _ops() -> list[SimOp]:
    return [
        SimOp(10, "a", "write", "hello world"),
        SimOp(10, "b", "write", "second"),
        SimOp(20, "a", "fsync"),
        SimOp(30, "b", "read", "hello world"),
    ]


def test_same_seed_reproduces_the_trace() -> None:
    first = dst_virtualize_clock_and_scheduler(_ops(), seed=7)
    second = dst_virtualize_clock_and_scheduler(_ops(), seed=7)
    assert first == second
    assert [s.now_ms for s in first.steps] == [10, 10, 20, 30]
    assert {s.node for s in first.steps[:2]} == {"a", "b"}


def test_seed_changes_only_the_tie_order() -> None:
    traces = {
        tuple(
            (s.now_ms, s.node)
            for s in dst_virtualize_clock_and_scheduler(_ops(), seed=seed).steps
        )
        for seed in range(6)
    }
    assert len(traces) >= 2
    for trace in traces:
        assert [t[0] for t in trace] == sorted(t[0] for t in trace)


def test_elapsed_time_tracks_the_last_operation() -> None:
    schedule = dst_virtualize_clock_and_scheduler(_ops(), seed=1, start_ms=0)
    assert schedule.elapsed_ms == 30
    assert schedule.steps[0].now_ms >= schedule.start_ms


def test_negative_time_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        SimOp(-1, "a", "write", "")
    assert error.value.code == "E001"


def test_operation_before_start_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        dst_virtualize_clock_and_scheduler(_ops(), seed=1, start_ms=500)
    assert error.value.code == "E001"


def test_virtual_clock_refuses_to_run_backwards() -> None:
    clock = VirtualClock(10)
    with pytest.raises(PillarError) as error:
        clock.advance_to(5)
    assert error.value.code == "E001"
    assert clock.elapsed_ms == 0


def test_injection_is_a_pure_function_of_the_seed() -> None:
    first = dst_inject_byzantine_and_io_faults(_ops(), seed=11)
    second = dst_inject_byzantine_and_io_faults(_ops(), seed=11)
    assert first.ops == second.ops
    assert first.injected == second.injected


def test_injection_drops_loses_and_duplicates() -> None:
    report = dst_inject_byzantine_and_io_faults(
        _ops(),
        seed=3,
        rates={
            "drop": 1.0,
            "duplicate": 0.0,
            "reorder": 0.0,
            "torn_write": 0.0,
            "lost_fsync": 0.0,
            "bitflip": 0.0,
        },
    )
    assert report.ops == ()
    assert report.injected["drop"] == len(_ops())


def test_duplicates_are_replayed_with_a_new_sequence() -> None:
    report = dst_inject_byzantine_and_io_faults(
        [SimOp(5, "a", "read", "x")],
        seed=3,
        rates={
            "drop": 0.0,
            "duplicate": 1.0,
            "reorder": 0.0,
            "torn_write": 0.0,
            "lost_fsync": 0.0,
            "bitflip": 0.0,
        },
    )
    assert len(report.ops) == 2
    assert report.ops[1].seq == report.ops[0].seq + 1


def test_torn_write_truncates_the_payload() -> None:
    report = dst_inject_byzantine_and_io_faults(
        [SimOp(5, "a", "write", "abcdefgh")],
        seed=3,
        rates={
            "drop": 0.0,
            "duplicate": 0.0,
            "reorder": 0.0,
            "torn_write": 1.0,
            "lost_fsync": 0.0,
            "bitflip": 0.0,
        },
    )
    assert report.ops[0].payload == "abcd"
    assert report.injected["torn_write"] == 1


def test_lost_fsync_drops_the_barrier() -> None:
    report = dst_inject_byzantine_and_io_faults(
        [SimOp(5, "a", "write", "x"), SimOp(6, "a", "fsync")],
        seed=3,
        rates={
            "drop": 0.0,
            "duplicate": 0.0,
            "reorder": 0.0,
            "torn_write": 0.0,
            "lost_fsync": 1.0,
            "bitflip": 0.0,
        },
    )
    assert [op.kind for op in report.ops] == ["write"]
    assert report.injected["lost_fsync"] == 1


def test_bitflip_corrupts_a_write() -> None:
    report = dst_inject_byzantine_and_io_faults(
        [SimOp(5, "a", "write", b"\x00\x01")],
        seed=3,
        rates={
            "drop": 0.0,
            "duplicate": 0.0,
            "reorder": 0.0,
            "torn_write": 0.0,
            "lost_fsync": 0.0,
            "bitflip": 1.0,
        },
    )
    assert report.ops[0].payload != b"\x00\x01"
    assert report.injected["bitflip"] == 1


def test_reorder_swaps_a_neighbour() -> None:
    report = dst_inject_byzantine_and_io_faults(
        [SimOp(1, "a", "read", 1), SimOp(2, "b", "read", 2)],
        seed=3,
        rates={
            "drop": 0.0,
            "duplicate": 0.0,
            "reorder": 1.0,
            "torn_write": 0.0,
            "lost_fsync": 0.0,
            "bitflip": 0.0,
        },
    )
    assert [op.node for op in report.ops] == ["b", "a"]
    assert report.injected["reorder"] == 2


def test_bad_fault_rate_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        dst_inject_byzantine_and_io_faults(_ops(), seed=1, rates={"drop": 2.0})
    assert error.value.code == "E003"


def test_unknown_fault_kind_is_rejected() -> None:
    with pytest.raises(PillarError) as error:
        dst_inject_byzantine_and_io_faults(_ops(), seed=1, rates={"meteor": 0.1})
    assert error.value.code == "E003"
    assert "meteor" in str(error.value)


def test_bisect_shrinks_to_the_single_culprit() -> None:
    ops = [SimOp(i, f"n{i}", "read", i) for i in range(16)]

    def oracle(candidate) -> bool:
        return any(op.payload == 11 for op in candidate)

    report = dst_bisect_and_minimize_schedule(ops, oracle)
    assert report.failed
    assert report.minimal == (SimOp(11, "n11", "read", 11),)
    assert report.original_length == 16
    assert not report.budget_exhausted
    assert report.shrink_ratio == pytest.approx(1 / 16)


def test_bisect_reports_a_schedule_that_does_not_fail() -> None:
    report = dst_bisect_and_minimize_schedule(_ops(), lambda candidate: False)
    assert not report.failed
    assert len(report.minimal) == len(_ops())
    assert report.probes == 1


def test_bisect_respects_the_probe_budget() -> None:
    ops = [SimOp(i, f"n{i}", "read", i) for i in range(32)]
    report = dst_bisect_and_minimize_schedule(ops, lambda candidate: True, max_probes=2)
    assert report.failed
    assert report.budget_exhausted
    assert report.probes <= 2


def test_fault_kinds_are_all_documented() -> None:
    assert set(FAULTS) == {
        "drop",
        "duplicate",
        "reorder",
        "torn_write",
        "lost_fsync",
        "bitflip",
    }
