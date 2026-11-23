"""Tests for the Elixir/OTP-inspired actor isolation and supervision subsystem.

These tests are pure Python and do not require any external toolchain.  They
cover actor isolation (deep-copy at the message boundary), mailbox semantics,
supervision strategies (one_for_one, one_for_all, rest_for_one), restart
intensity limits, crash classification, and telemetry bounds.
"""

from __future__ import annotations

import pytest

from compiler.frontend.errors import PolyglotError
from runtime.actor_otp import (
    RestartIntensity,
    Strategy,
    Supervisor,
    Telemetry,
    classify_fault,
    spawn_actor,
)

# ---------------------------------------------------------------------------
# Actor isolation and message passing
# ---------------------------------------------------------------------------


def test_spawn_actor_and_run():
    def handler(state, payload):
        return state + 1

    actor = spawn_actor(0, handler)
    actor.send("noop")
    actor.send("noop")
    n = actor.run()
    assert n == 2
    assert actor.state == 2


def test_actor_state_is_copied_at_spawn():
    original = [1, 2, 3]
    actor = spawn_actor(original, lambda s, p: s)
    original.append(99)
    assert actor.state == [1, 2, 3]


def test_actor_message_is_deep_copied():
    actor = spawn_actor([], lambda s, p: s + [p])
    original = [1, 2]
    actor.send(original)
    original.append(99)
    actor.run()
    assert actor.state == [[1, 2]]


def test_actor_send_to_dead_raises():
    actor = spawn_actor(0, lambda s, p: s)
    actor.stop()
    with pytest.raises(PolyglotError) as exc:
        actor.send("message")
    assert exc.value.code == "E001"


def test_actor_handler_exception_kills_actor():
    def bad_handler(state, payload):
        raise ValueError("boom")

    actor = spawn_actor(0, bad_handler)
    actor.send("trigger")
    with pytest.raises(ValueError):
        actor.run()
    assert not actor.alive


def test_actor_stop_discards_mailbox():
    actor = spawn_actor(0, lambda s, p: s)
    actor.send("a")
    actor.send("b")
    actor.stop()
    assert actor.mailbox.is_empty()
    assert not actor.alive


def test_mailbox_capacity_overflow():
    actor = spawn_actor(0, lambda s, p: s, mailbox_capacity=2)
    actor.send("a")
    actor.send("b")
    with pytest.raises(PolyglotError) as exc:
        actor.send("c")
    assert exc.value.code == "E001"


def test_mailbox_receive_until_preserves_rest():
    actor = spawn_actor([], lambda s, p: s)
    actor.send(1)
    actor.send(2)
    actor.send(3)
    result = actor.mailbox.receive_until(lambda x: x == 2)
    assert result is not None
    assert result[1] == 2
    # 1 and 3 should still be in the mailbox
    remaining = []
    while not actor.mailbox.is_empty():
        remaining.append(actor.mailbox.receive()[1])
    assert remaining == [1, 3]


def test_mailbox_receive_until_no_match():
    actor = spawn_actor([], lambda s, p: s)
    actor.send(1)
    actor.send(2)
    result = actor.mailbox.receive_until(lambda x: x == 99)
    assert result is None
    assert len(actor.mailbox) == 2


def test_actor_id_is_unique():
    actor1 = spawn_actor(0, lambda s, p: s)
    actor2 = spawn_actor(0, lambda s, p: s)
    assert actor1.actor_id != actor2.actor_id


def test_actor_run_max_messages():
    actor = spawn_actor(0, lambda s, p: s + 1)
    for _ in range(10):
        actor.send("tick")
    processed = actor.run(max_messages=3)
    assert processed == 3
    assert actor.state == 3
    assert len(actor.mailbox) == 7


# ---------------------------------------------------------------------------
# Supervision strategies
# ---------------------------------------------------------------------------


def _make_supervisor(strategy, max_restarts=10, window=60.0):
    sup = Supervisor("test", strategy, RestartIntensity(max_restarts, window))
    sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    sup.add_child("b", lambda: spawn_actor(0, lambda s, p: s))
    sup.add_child("c", lambda: spawn_actor(0, lambda s, p: s))
    return sup


def test_one_for_one_restarts_only_crashed():
    sup = _make_supervisor(Strategy.ONE_FOR_ONE)
    outcome = sup.restart("b", now=1.0)
    assert outcome.restarted == ("b",)
    assert not outcome.escalated


def test_one_for_all_restarts_every_child():
    sup = _make_supervisor(Strategy.ONE_FOR_ALL)
    outcome = sup.restart("b", now=1.0)
    assert set(outcome.restarted) == {"a", "b", "c"}


def test_rest_for_one_restarts_from_crashed_onward():
    sup = _make_supervisor(Strategy.REST_FOR_ONE)
    outcome = sup.restart("b", now=1.0)
    assert set(outcome.restarted) == {"b", "c"}


def test_restart_intensity_exceeded_escalates():
    sup = Supervisor("test", Strategy.ONE_FOR_ONE, RestartIntensity(2, 60.0))
    sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    sup.restart("a", now=1.0)
    sup.restart("a", now=2.0)
    with pytest.raises(PolyglotError) as exc:
        sup.restart("a", now=3.0)
    assert exc.value.code == "E004"


def test_restart_intensity_window_resets():
    sup = Supervisor("test", Strategy.ONE_FOR_ONE, RestartIntensity(2, 10.0))
    sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    sup.restart("a", now=1.0)
    sup.restart("a", now=2.0)
    # The first two restarts are within the window; a third at t=100 is
    # outside the window of the first two, so the budget resets.
    outcome = sup.restart("a", now=100.0)
    assert not outcome.escalated


def test_restart_intensity_zero_restarts():
    sup = Supervisor("test", Strategy.ONE_FOR_ONE, RestartIntensity(0, 60.0))
    sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    with pytest.raises(PolyglotError) as exc:
        sup.restart("a", now=1.0)
    assert exc.value.code == "E004"


def test_unknown_child_restart_raises():
    sup = _make_supervisor(Strategy.ONE_FOR_ONE)
    with pytest.raises(PolyglotError) as exc:
        sup.restart("unknown", now=1.0)
    assert exc.value.code == "E001"


def test_duplicate_child_raises():
    sup = Supervisor("test", Strategy.ONE_FOR_ONE, RestartIntensity(5, 60.0))
    sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    with pytest.raises(PolyglotError) as exc:
        sup.add_child("a", lambda: spawn_actor(0, lambda s, p: s))
    assert exc.value.code == "E001"


def test_strategy_from_name():
    assert Strategy.from_name("one_for_one") == Strategy.ONE_FOR_ONE
    assert Strategy.from_name("REST_FOR_ONE") == Strategy.REST_FOR_ONE
    with pytest.raises(PolyglotError) as exc:
        Strategy.from_name("invalid")
    assert exc.value.code == "E001"


# ---------------------------------------------------------------------------
# Crash classification and telemetry
# ---------------------------------------------------------------------------


def test_classify_fault_value_error():
    try:
        raise ValueError("bad input")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker1")
    assert report.code == "E001"
    assert report.actor_name == "worker1"
    assert report.boundary_safe is True


def test_classify_fault_type_error():
    try:
        raise TypeError("wrong type")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker2")
    assert report.code == "E001"


def test_classify_fault_import_error():
    try:
        raise ImportError("no module")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker3")
    assert report.code == "E006"


def test_classify_fault_memory_error():
    try:
        raise MemoryError("oom")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker4")
    assert report.code == "E004"


def test_classify_fault_boundary_violation():
    try:
        raise ValueError("attempted to access MAP_JIT region")
    except Exception as exc:
        with pytest.raises(PolyglotError) as boundary:
            classify_fault(exc, actor_name="worker5")
    assert boundary.value.code == "E007"


def test_classify_fault_recursion_error():
    try:
        raise RecursionError("deep recursion")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker6")
    assert report.code == "E002"


def test_classify_fault_preserves_polyglot_error_code():
    try:
        raise PolyglotError("E003", "api misuse")
    except Exception as exc:
        report = classify_fault(exc, actor_name="worker7")
    assert report.code == "E003"


def test_crash_report_format():
    try:
        raise ValueError("test message")
    except Exception as exc:
        report = classify_fault(exc, actor_name="actor1")
    formatted = report.format()
    assert "actor1" in formatted
    assert "E001" in formatted
    assert "test message" in formatted


def test_telemetry_bounded():
    tel = Telemetry(max_events=3)
    for i in range(5):
        tel.record("a", "event", f"detail_{i}", timestamp=float(i))
    assert len(tel) == 3
    # The oldest two events should have been dropped.
    details = [e.detail for e in tel._events]
    assert "detail_0" not in details
    assert "detail_1" not in details
    assert "detail_2" in details
    assert "detail_4" in details


def test_telemetry_events_for_actor():
    tel = Telemetry(max_events=10)
    tel.record("a", "crash", "a failed", code="E001")
    tel.record("b", "crash", "b failed", code="E002")
    tel.record("a", "restart", "a restarted")
    a_events = tel.events_for("a")
    assert len(a_events) == 2
    assert all(e.actor_name == "a" for e in a_events)


def test_telemetry_clear():
    tel = Telemetry(max_events=10)
    tel.record("a", "crash", "failed")
    assert len(tel) == 1
    tel.clear()
    assert len(tel) == 0


def test_telemetry_zero_max_raises():
    with pytest.raises(PolyglotError) as exc:
        Telemetry(max_events=0)
    assert exc.value.code == "E001"


def test_restart_intensity_exceeded_helper():
    intensity = RestartIntensity(3, 10.0)
    # 3 restarts within 10s should not exceed.
    assert not intensity.exceeded([1.0, 2.0, 3.0])
    # 4 restarts within 10s should exceed.
    assert intensity.exceeded([1.0, 2.0, 3.0, 4.0])
    # 4 restarts spread across 20s should not exceed (only 2 in the last 10s).
    assert not intensity.exceeded([1.0, 2.0, 15.0, 18.0])


def test_supervisor_as_dict():
    sup = _make_supervisor(Strategy.ONE_FOR_ONE)
    info = sup.as_dict()
    assert info["strategy"] == "one_for_one"
    assert info["children"] == ("a", "b", "c")
    assert info["alive"] == ("a", "b", "c")
