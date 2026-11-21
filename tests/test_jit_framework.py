"""Focused tests for the speculative JIT frame (Frame A)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from runtime.diagnostics import PillarError
from runtime.jit import (
    GuardSpec,
    jit_compile_hot_trace,
    jit_deoptimize_to_interpreter_frame,
    jit_emit_diagnostic_e008,
    jit_emit_type_and_bounds_guards,
    jit_invalidate_on_redefinition_or_hotpatch,
    jit_manage_executable_memory_rx,
    jit_patch_ic_and_call_sites,
    new_jit_state,
    release_executable_memory_block,
)

SOURCE = "fn add(a: i64, b: i64) -> i64 { a + b }"
SPAN = (3, 9)


def test_compile_hot_trace_counts_to_threshold_then_compiles() -> None:
    state = new_jit_state()
    records = [
        jit_compile_hot_trace(
            state,
            key="add",
            source=SOURCE,
            signature=("i64", "i64"),
            threshold=3,
            span=SPAN,
        )
        for _ in range(3)
    ]
    assert [record.state for record in records] == ["cold", "cold", "compiled"]
    assert records[-1].hotness == 3
    assert records[-1].mode == "model"
    assert state.traces["add"] is records[-1]
    assert [guard.target for guard in records[-1].guards] == [
        "add",
        "arg0",
        "arg1",
    ]


def test_compile_hot_trace_rejects_shape_change() -> None:
    state = new_jit_state()
    jit_compile_hot_trace(state, key="f", source=SOURCE, threshold=1)
    with pytest.raises(PillarError) as raised:
        jit_compile_hot_trace(state, key="f", source="fn f() -> i64 { 1 }", threshold=1)
    assert raised.value.code == "E008"


def test_compile_hot_trace_can_use_real_native_codegen() -> None:
    from backends.jit.memory import is_supported_host

    if not is_supported_host():
        pytest.skip("native JIT requires macOS on Apple Silicon")
    state = new_jit_state()
    record = jit_compile_hot_trace(
        state,
        key="native_add",
        source=SOURCE,
        signature=("i64", "i64"),
        threshold=1,
        allow_native=True,
    )
    assert record.mode == "native"
    assert record.block_id == "native_add:v1"
    assert record.module is not None
    assert callable(record.module["add"])
    assert record.module["add"](2, 3) == 5


def test_type_non_null_and_bounds_guards() -> None:
    guards = (
        GuardSpec("type", "arg0", "i64", span=SPAN),
        GuardSpec("type", "arg1", "bool", span=SPAN),
        GuardSpec("non_null", "items", span=SPAN),
        GuardSpec("bounds", "items", lower=0, upper=4, span=SPAN),
    )
    ok = jit_emit_type_and_bounds_guards(
        {"arg0": 7, "arg1": True, "items": [1, 2, 3]}, guards=guards
    )
    assert ok.passed and ok.failures == ()

    failed = jit_emit_type_and_bounds_guards(
        {"arg0": 7, "arg1": 1, "items": None}, guards=guards
    )
    assert not failed.passed
    assert [item.guard.kind for item in failed.failures] == [
        "type",
        "non_null",
        "bounds",
    ]
    assert "arg1" in failed.reason


def test_bounds_guard_out_of_range_and_empty_range_are_diagnostics() -> None:
    values = {"items": [1, 2, 3, 4]}
    report = jit_emit_type_and_bounds_guards(
        values,
        guards=(GuardSpec("bounds", "items", lower=0, upper=4),),
    )
    assert not report.passed  # len == upper is out of range

    with pytest.raises(PillarError) as raised:
        jit_emit_type_and_bounds_guards(
            values,
            guards=(GuardSpec("bounds", "items", lower=3, upper=3),),
        )
    assert raised.value.code == "E008"


def test_deoptimize_preserves_live_values_and_stack() -> None:
    state = new_jit_state()
    jit_compile_hot_trace(state, key="f", source=SOURCE, threshold=1, span=SPAN)
    report = jit_emit_type_and_bounds_guards(
        {"arg0": "wrong"}, guards=(GuardSpec("type", "arg0", "i64"),)
    )
    frame = jit_deoptimize_to_interpreter_frame(
        state,
        key="f",
        guard_report=report,
        live_values={"arg0": "wrong", "saved": 42},
        stack=[("f", 12)],
    )
    assert frame.live_values == {"arg0": "wrong", "saved": 42}
    assert frame.stack == (("f", 12),)
    assert frame.span == SPAN
    assert "arg0" in frame.reason
    assert state.traces["f"].state == "deoptimized"


def test_guard_exhaustion_is_e008() -> None:
    state = new_jit_state()
    jit_compile_hot_trace(state, key="f", source=SOURCE, threshold=1)
    failing = jit_emit_type_and_bounds_guards(
        {"x": 1}, guards=(GuardSpec("type", "x", "str"),)
    )
    state.traces["f"] = replace(state.traces["f"], deopt_count=32)
    with pytest.raises(PillarError) as raised:
        jit_deoptimize_to_interpreter_frame(
            state, key="f", guard_report=failing, live_values={"x": 1}
        )
    assert raised.value.code == "E008"
    assert "exhaustion" in str(raised.value)


def test_executable_memory_model_and_real_transitions() -> None:
    from backends.jit.memory import is_supported_host

    state = new_jit_state()
    block = jit_manage_executable_memory_rx(
        state, block_id="model", code=b"\x00" * 32, mode="model"
    )
    assert block.protection == "model (no executable mapping)"
    assert release_executable_memory_block(state, block_id="model") is True
    assert state.blocks["model"].released is True

    if not is_supported_host():
        return
    real = jit_manage_executable_memory_rx(
        state,
        block_id="real",
        code=b"\x00" * 32,
        mode="native",
        opt_in_executable_memory=True,
        span=SPAN,
    )
    assert real.region is not None
    assert real.protection == "r-x (W^X after commit)"
    assert release_executable_memory_block(state, block_id="real") is True
    assert real.region is None


def test_executable_memory_pins_and_rejects_in_place_mutation() -> None:
    state = new_jit_state()
    jit_manage_executable_memory_rx(
        state, block_id="b", code=b"abc", active_frames=["frame-1"]
    )
    with pytest.raises(PillarError) as raised:
        jit_manage_executable_memory_rx(state, block_id="b", code=b"xyz")
    assert raised.value.code == "E008"
    assert release_executable_memory_block(state, block_id="b") is False
    assert (
        release_executable_memory_block(state, block_id="b", active_frames=["frame-1"])
        is False
    )


def test_ic_patching_is_atomic_deferred_and_version_checked() -> None:
    state = new_jit_state()
    first = jit_patch_ic_and_call_sites(
        state, site_id="call@10", target="f:v1", version=1
    )
    assert first.target == "f:v1"

    pinned = jit_patch_ic_and_call_sites(
        state,
        site_id="call@10",
        target="f:v2",
        version=2,
        active_frames=["frame-7"],
    )
    assert pinned.target == "f:v1"
    assert pinned.deferred_target == "f:v2"
    assert pinned.pinned_by == frozenset({"frame-7"})

    published = jit_patch_ic_and_call_sites(
        state, site_id="call@10", target="f:v2", version=2
    )
    assert published.target == "f:v2"
    assert published.deferred_target is None

    with pytest.raises(PillarError):
        jit_patch_ic_and_call_sites(state, site_id="call@10", target="f:v0", version=0)


def test_invalidation_bumps_epoch_and_defers_pinned_blocks() -> None:
    state = new_jit_state()
    jit_compile_hot_trace(state, key="f", source=SOURCE, threshold=1)
    jit_patch_ic_and_call_sites(state, site_id="s", target="f", version=1)
    report = jit_invalidate_on_redefinition_or_hotpatch(
        state,
        key="f",
        reason="definition changed",
        hotpatch_event={"target": "f", "version": 2},
        active_frames=["frame-1"],
        span=SPAN,
    )
    assert report.epoch == 1
    assert report.new_version == 2
    assert report.invalidated_traces == ("f",)
    assert report.cleared_sites == ("s",)
    assert report.deferred_blocks == ("f:v1",)
    assert state.traces["f"].state == "invalidated"
    assert state.epochs["f"] == 1

    released = jit_invalidate_on_redefinition_or_hotpatch(
        state,
        key="f",
        reason="idle reclamation",
        released_frames=["frame-1"],
    )
    assert released.reclaimed_blocks == ("f:v1",)
    assert released.epoch == 2


def test_invalidation_rejects_mismatched_hotpatch_target() -> None:
    state = new_jit_state()
    with pytest.raises(PillarError) as raised:
        jit_invalidate_on_redefinition_or_hotpatch(
            state,
            key="f",
            reason="hotpatch",
            hotpatch_event={"target": "g"},
            span=SPAN,
        )
    assert raised.value.code == "E008"
    assert raised.value.span == SPAN


def test_e008_diagnostic_carries_span_and_message() -> None:
    error = jit_emit_diagnostic_e008("bad guard", span=SPAN, trace_key="f")
    assert isinstance(error, PillarError)
    assert error.code == "E008"
    assert error.span == SPAN
    assert "bad guard" in str(error)
