from __future__ import annotations

import builtins

import pytest

from runtime.diagnostics import PillarError
from security.sandbox import (
    DYNAMIC_CODE_SINKS,
    REFERENCE_CAVEAT,
    ZEROIZATION_CAVEAT,
    CapabilityEpoch,
    CfiTargetSpec,
    HeapRegionSpec,
    PkuDomainSpec,
    SpectreSite,
    SyscallRule,
    TaggedCapability,
    sandbox_cfi_shadow_stack_emit,
    sandbox_deterministic_fuel_meter,
    sandbox_ebpf_syscall_synthesize,
    sandbox_fat_pointer_bounds_verify,
    sandbox_mprotect_heap_isolate,
    sandbox_pku_mpk_domain_switch,
    sandbox_spectre_lfence_inject,
    sandbox_taint_flow_enforce,
    sandbox_violation_trap_recover,
    sandbox_zeroize_secret_drop,
)


def _assert_error(
    excinfo: pytest.ExceptionInfo[PillarError],
    code: str,
    span: tuple[int, int] | None = None,
    line: int | None = None,
    col: int | None = None,
) -> PillarError:
    error = excinfo.value
    assert isinstance(error, PillarError)
    assert error.code == code
    if span is not None:
        assert error.span == span
    if line is not None:
        assert error.line == line
    if col is not None:
        assert error.col == col
    return error


# 71. mprotect heap isolation ------------------------------------------------


def test_mprotect_plan_is_declarative_and_deterministic() -> None:
    plan = sandbox_mprotect_heap_isolate(
        (HeapRegionSpec("heap", 5000, "rw"), ("meta", 100, "ro")),
        page_size=4096,
        guard_pages=1,
    )
    assert REFERENCE_CAVEAT in plan.caveat
    assert plan.page_size == 4096
    assert plan.total_layout == 24576
    assert [(region.name, region.base, region.size) for region in plan.regions] == [
        ("heap", 4096, 8192),
        ("meta", 16384, 4096),
    ]
    assert plan.regions[0].guard_before == (0, 4096)
    assert plan.regions[0].guard_after == (12288, 16384)
    assert plan.regions[1].guard_after == (20480, 24576)


def test_mprotect_rejects_real_mprotect_with_span() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_mprotect_heap_isolate(
            [("heap", 4096)], span=(11, 22), line=7, col=9, execute=True
        )
    error = _assert_error(excinfo, "E007", (11, 22), 7, 9)
    assert "mprotect" in str(error)


def test_mprotect_plan_execution_boundary_uses_recorded_span() -> None:
    plan = sandbox_mprotect_heap_isolate([("heap", 4096)], span=(3, 8), line=2, col=1)
    with pytest.raises(PillarError) as excinfo:
        plan.apply()
    _assert_error(excinfo, "E007", (3, 8), 2, 1)


@pytest.mark.parametrize(
    ("regions", "code"),
    [
        ([("heap", 0)], "E001"),
        ([("heap", -1)], "E001"),
        ([("heap", 10, "rwx")], "E003"),
        ([("heap", 10), ("heap", 10)], "E003"),
    ],
)
def test_mprotect_validates_policy(regions: object, code: str) -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_mprotect_heap_isolate(regions)
    _assert_error(excinfo, code)


# 72. Declarative syscall allowlist ------------------------------------------


def test_ebpf_emits_portable_allowlist_with_deny_precedence() -> None:
    plan = sandbox_ebpf_syscall_synthesize(
        ["read", SyscallRule("write", number=1)],
        deny=[("read", "deny")],
    )
    assert plan.default_action == "deny"
    assert plan.emit() == (
        SyscallRule("read", action="deny", number=None, constraints=()),
        SyscallRule("write", action="allow", number=1, constraints=()),
    )
    assert (
        plan.digest
        == sandbox_ebpf_syscall_synthesize([("write", 1), "read"], deny=["read"]).digest
    )
    assert REFERENCE_CAVEAT in plan.caveat


def test_ebpf_real_backends_are_rejected_with_span() -> None:
    plan = sandbox_ebpf_syscall_synthesize(["read"], span=(20, 31), line=4, col=5)
    with pytest.raises(PillarError) as excinfo:
        plan.to_bpf()
    _assert_error(excinfo, "E007", (20, 31), 4, 5)
    with pytest.raises(PillarError) as excinfo:
        plan.to_seatbelt()
    _assert_error(excinfo, "E007", (20, 31), 4, 5)
    with pytest.raises(PillarError) as excinfo:
        sandbox_ebpf_syscall_synthesize(["read"], span=(1, 2), execute=True)
    _assert_error(excinfo, "E007", (1, 2))


def test_ebpf_validates_rules_and_constraints() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_ebpf_syscall_synthesize(["read", ("read", "allow", 7)])
    _assert_error(excinfo, "E002")
    with pytest.raises(PillarError) as excinfo:
        sandbox_ebpf_syscall_synthesize([("read", "allow", None, ("bad",))])
    _assert_error(excinfo, "E003")
    assert sandbox_ebpf_syscall_synthesize(
        [], deny=[{"name": "read", "constraints": ["fd>=0", "count<=4096"]}]
    ).emit() == (
        SyscallRule(
            "read",
            action="deny",
            constraints=("fd>=0", "count<=4096"),
        ),
    )


# 73. PKU / MPK domain switching --------------------------------------------


def test_pku_simulates_domain_tokens_and_transitions() -> None:
    plan = sandbox_pku_mpk_domain_switch(
        [
            PkuDomainSpec("public", 1, "rw"),
            PkuDomainSpec("secret", 2, "ro"),
        ],
        switches=[("public", "secret"), ("secret", "public")],
    )
    assert [token.render() for token in plan.tokens] == [
        "public:pkey=1:rw",
        "secret:pkey=2:ro",
    ]
    assert plan.current == "public"
    assert [transition.target for transition in plan.transitions] == [
        "secret",
        "public",
    ]
    assert plan.transitions[0].token.pkey == 2
    assert REFERENCE_CAVEAT in plan.caveat


def test_pku_rejects_real_mpk_with_span() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_pku_mpk_domain_switch(
            [("public", 0)], span=(40, 50), line=8, col=11, execute=True
        )
    error = _assert_error(excinfo, "E007", (40, 50), 8, 11)
    assert "MPK/POE" in str(error)


def test_pku_validates_keys_policy_and_transition_continuity() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_pku_mpk_domain_switch([("public", 16)])
    _assert_error(excinfo, "E001")
    with pytest.raises(PillarError) as excinfo:
        sandbox_pku_mpk_domain_switch(
            [("a", 0), ("b", 1)],
            switches=[("b", "a")],
            allowed_edges=[("a", "b")],
        )
    _assert_error(excinfo, "E002")
    with pytest.raises(PillarError) as excinfo:
        sandbox_pku_mpk_domain_switch([("a", 0)], current="missing")
    _assert_error(excinfo, "E003")


def test_pku_plan_activate_is_e007_with_recorded_span() -> None:
    plan = sandbox_pku_mpk_domain_switch([("a", 0)], span=(2, 4), line=3, col=6)
    with pytest.raises(PillarError) as excinfo:
        plan.activate()
    _assert_error(excinfo, "E007", (2, 4), 3, 6)


# 74. Fat-pointer bounds verification ---------------------------------------


def test_fat_pointer_accepts_valid_spatial_and_temporal_access() -> None:
    capability = TaggedCapability("buffer", 100, 16, offset=4, alignment=4)
    epoch = CapabilityEpoch("buffer", 1)
    result = sandbox_fat_pointer_bounds_verify(
        capability, epoch, access_size=4, span=(5, 9), line=1, col=2
    )
    assert result.result == "proved"
    assert result.address == 104
    assert result.access_size == 4
    assert result.capability is capability


def test_fat_pointer_rejects_spatial_violations() -> None:
    capability = TaggedCapability("buffer", 0, 8, offset=6)
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(
            capability, CapabilityEpoch("buffer", 1), access_size=4, span=(1, 4)
        )
    _assert_error(excinfo, "E002", (1, 4))
    misaligned = TaggedCapability("buffer", 0, 8, offset=1, alignment=4)
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(
            misaligned, CapabilityEpoch("buffer", 1), access_size=1
        )
    _assert_error(excinfo, "E001")


def test_fat_pointer_rejects_temporal_violations() -> None:
    capability = TaggedCapability("buffer", 0, 8, generation=2)
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(
            capability, CapabilityEpoch("buffer", 1), span=(10, 18), line=6, col=7
        )
    _assert_error(excinfo, "E002", (10, 18), 6, 7)
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(
            capability, CapabilityEpoch("buffer", 2, alive=False)
        )
    _assert_error(excinfo, "E002")
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(capability, CapabilityEpoch("other", 2))
    _assert_error(excinfo, "E002")


def test_fat_pointer_validates_layout_and_advance() -> None:
    cap = TaggedCapability("buffer", 0, 16, alignment=4)
    moved = cap.at(8)
    assert (moved.offset, moved.address(), moved.generation) == (8, 8, 1)
    with pytest.raises(PillarError) as excinfo:
        cap.at(17)
    _assert_error(excinfo, "E002")
    with pytest.raises(PillarError) as excinfo:
        cap.at(1)
    _assert_error(excinfo, "E001")
    with pytest.raises(PillarError) as excinfo:
        sandbox_fat_pointer_bounds_verify(
            TaggedCapability("buffer", 0, 0), CapabilityEpoch("buffer", 1)
        )
    _assert_error(excinfo, "E001")


# 75. CFI shadow-stack plan --------------------------------------------------


def test_cfi_plan_emits_deterministic_shadow_stack_slots() -> None:
    plan = sandbox_cfi_shadow_stack_emit(
        [CfiTargetSpec("main", 2, "()"), ("helper", 1)], mode="return"
    )
    assert plan.mode == "return"
    assert [slot.target for slot in plan.slots] == ["main", "main", "helper"]
    assert [slot.depth for slot in plan.slots] == [1, 2, 1]
    assert [slot.index for slot in plan.slots] == [0, 1, 2]
    assert (
        plan.digest
        == sandbox_cfi_shadow_stack_emit(
            [CfiTargetSpec("main", 2, "()"), ("helper", 1)], mode="return"
        ).digest
    )
    assert REFERENCE_CAVEAT in plan.caveat


def test_cfi_real_shadow_stack_is_rejected_with_span() -> None:
    plan = sandbox_cfi_shadow_stack_emit([("main", 1)], span=(13, 19), line=12, col=14)
    with pytest.raises(PillarError) as excinfo:
        plan.install()
    _assert_error(excinfo, "E007", (13, 19), 12, 14)
    with pytest.raises(PillarError) as excinfo:
        sandbox_cfi_shadow_stack_emit([("main", 1)], span=(0, 1), execute=True)
    _assert_error(excinfo, "E007", (0, 1))


def test_cfi_validates_mode_and_targets() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_cfi_shadow_stack_emit([("main", 1)], mode="sideways")
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        sandbox_cfi_shadow_stack_emit([("main", -1)])
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        sandbox_cfi_shadow_stack_emit([])
    _assert_error(excinfo, "E003")


# 76. Spectre barrier annotations --------------------------------------------


def test_spectre_plan_emits_barrier_annotations() -> None:
    plan = sandbox_spectre_lfence_inject(
        [SpectreSite("load", "bounds-check"), ("call", "indirect-branch")]
    )
    assert [item.barrier for item in plan.annotations] == ["lfence", "lfence"]
    assert [item.placement for item in plan.annotations] == [
        "after-bounds-check",
        "before-indirect-call",
    ]
    assert [item.index for item in plan.annotations] == [0, 1]
    assert REFERENCE_CAVEAT in plan.caveat


def test_spectre_real_barriers_are_rejected_with_span() -> None:
    plan = sandbox_spectre_lfence_inject(
        [("load", "load-value")], span=(31, 42), line=9, col=3
    )
    with pytest.raises(PillarError) as excinfo:
        plan.inject()
    _assert_error(excinfo, "E007", (31, 42), 9, 3)
    with pytest.raises(PillarError) as excinfo:
        sandbox_spectre_lfence_inject([("load", "load-value")], execute=True)
    _assert_error(excinfo, "E007")


def test_spectre_validates_pattern_and_barrier() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_spectre_lfence_inject([("load", "unknown")])
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        sandbox_spectre_lfence_inject([("load", "bounds-check")], barrier="")
    _assert_error(excinfo, "E003")


# 77. Deterministic fuel meter -----------------------------------------------


def test_fuel_meter_counts_loop_iterations_deterministically() -> None:
    meter = sandbox_deterministic_fuel_meter(3, span=(4, 7), line=2, col=2)
    snapshots = [_assert_fuel_snapshot(meter.consume(), 1, 3)]
    snapshots.append(_assert_fuel_snapshot(meter.consume(2), 3, 3))
    assert [snapshot.remaining for snapshot in snapshots] == [2, 0]
    assert meter.exhausted
    assert meter.snapshot() == snapshots[-1]


def _assert_fuel_snapshot(snapshot: object, used: int, limit: int) -> object:
    assert snapshot.used == used
    assert snapshot.limit == limit
    assert snapshot.remaining == limit - used
    assert snapshot.exhausted is (used == limit)
    return snapshot


def test_fuel_exhaustion_is_e002_and_does_not_mutate_state() -> None:
    meter = sandbox_deterministic_fuel_meter(2, span=(60, 66), line=15, col=16)
    meter.consume()
    with pytest.raises(PillarError) as excinfo:
        meter.consume(2)
    _assert_error(excinfo, "E002", (60, 66), 15, 16)
    assert (meter.used, meter.remaining) == (1, 1)
    with pytest.raises(PillarError) as excinfo:
        meter.consume(2, span=(70, 75), line=20, col=21)
    _assert_error(excinfo, "E002", (70, 75), 20, 21)


def test_fuel_zero_cost_reset_and_replay_checkpoint() -> None:
    meter = sandbox_deterministic_fuel_meter(0)
    assert meter.exhausted
    assert meter.consume(0).used == 0
    with pytest.raises(PillarError) as excinfo:
        meter.consume(1, span=(12, 15))
    _assert_error(excinfo, "E002", (12, 15))
    replay = sandbox_deterministic_fuel_meter(5, initial_used=2)
    assert replay.remaining == 3
    replay.consume(3)
    replay.reset()
    assert replay.snapshot().used == 0
    assert replay.consume(5).exhausted


def test_fuel_validates_limit_cost_and_checkpoint() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_deterministic_fuel_meter(-1)
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        sandbox_deterministic_fuel_meter(2, initial_used=3)
    _assert_error(excinfo, "E003")
    meter = sandbox_deterministic_fuel_meter(2)
    with pytest.raises(PillarError) as excinfo:
        meter.consume(True)
    _assert_error(excinfo, "E003")


# 78. Bit-level taint flow ---------------------------------------------------


def test_taint_sources_have_distinct_single_bits() -> None:
    engine = sandbox_taint_flow_enforce(span=(1, 3), line=1, col=1)
    secret = engine.source("secret", 7)
    nonce = engine.source("nonce", 2)
    assert (secret.taint, nonce.taint) == (1, 2)
    assert engine.sources == (("secret", 1), ("nonce", 2))
    assert engine.labels_for(3) == ("secret", "nonce")
    assert engine.labels_for(8) == ("bit:3",)


def test_taint_propagates_through_operators_and_projection() -> None:
    engine = sandbox_taint_flow_enforce({"secret": 4})
    secret = engine.source("secret", 5)
    result = engine.combine("add", engine.combine("mul", secret, 2), 1)
    assert (result.value, result.taint) == (11, 4)
    compared = engine.combine("eq", result, 11)
    assert (compared.value, compared.taint) == (True, 4)
    values = engine.source("secret", [10, 20])
    index = engine.source("offset", 1)
    projected = engine.project(values, index)
    assert engine.sources == (("secret", 4), ("offset", 5))
    assert (projected.value, projected.taint) == (20, 5)
    container = [1, {"nested": (values, 3)}]
    assert engine.taint_of(container) == 4


def test_taint_blocks_exec_eval_and_compile_sinks_with_e008() -> None:
    engine = sandbox_taint_flow_enforce(span=(80, 90), line=30, col=31)
    payload = engine.source("secret", "2 + 2")
    for sink in ("eval", "exec", "compile", builtins.eval, builtins.exec):
        with pytest.raises(PillarError) as excinfo:
            engine.enforce_sink(sink, payload)
        error = _assert_error(excinfo, "E008", (80, 90), 30, 31)
        assert "tainted input" in str(error)
    assert engine.enforce_sink("EVAL", "1 + 1").sink == "eval"
    assert DYNAMIC_CODE_SINKS == frozenset({"exec", "eval", "compile"})


def test_taint_rejects_unknown_sinks_and_bad_bit_encoding() -> None:
    engine = sandbox_taint_flow_enforce()
    with pytest.raises(PillarError) as excinfo:
        engine.enforce_sink("open", "1 + 1")
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        sandbox_taint_flow_enforce({"secret": 3})
    _assert_error(excinfo, "E008")
    with pytest.raises(PillarError) as excinfo:
        engine.labels_for(-1)
    _assert_error(excinfo, "E008")


def test_taint_detects_cycles_and_wraps_operator_failures() -> None:
    engine = sandbox_taint_flow_enforce()
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(PillarError) as excinfo:
        engine.taint_of(cycle)
    _assert_error(excinfo, "E008")
    with pytest.raises(PillarError) as excinfo:
        engine.combine("truediv", 1, 0, span=(2, 5), line=3, col=4)
    _assert_error(excinfo, "E001", (2, 5), 3, 4)
    with pytest.raises(PillarError) as excinfo:
        engine.combine("unknown", 1)
    _assert_error(excinfo, "E003")


# 79. Best-effort zeroization ------------------------------------------------


def test_zeroize_bytearray_in_place_with_explicit_caveat() -> None:
    secret = bytearray(b"topsecret")
    wrapper = sandbox_zeroize_secret_drop(secret, span=(6, 8), line=4, col=5)
    assert wrapper.zeroizable
    with pytest.raises(PillarError) as excinfo:
        sandbox_zeroize_secret_drop(b"immutable", require_mutable=True, span=(1, 2))
    _assert_error(excinfo, "E003", (1, 2))
    receipt = wrapper.drop()
    assert bytes(secret) == bytes(len(secret))
    assert receipt.zeroized and receipt.zeroized_bytes == 9
    assert receipt.length == 9
    assert receipt.caveat == ZEROIZATION_CAVEAT
    assert "Best-effort" in receipt.caveat
    assert wrapper.drop() is receipt


def test_zeroize_context_manager_and_writable_memoryview() -> None:
    buffer = bytearray(b"abcd")
    view = memoryview(buffer)
    with sandbox_zeroize_secret_drop(view):
        pass
    assert bytes(buffer) == bytes(4)


def test_zeroize_immutable_secret_reports_reason_without_claiming_wipe() -> None:
    wrapper = sandbox_zeroize_secret_drop("secret")
    assert not wrapper.zeroizable
    assert "immutable" in wrapper.reason
    receipt = wrapper.drop()
    assert not receipt.zeroized
    assert receipt.zeroized_bytes == 0
    assert receipt.secret_type == "str"
    assert receipt.caveat == ZEROIZATION_CAVEAT


def test_zeroize_list_and_integer_array() -> None:
    values = [1, 2, 3]
    receipt = sandbox_zeroize_secret_drop(values).drop()
    assert values == [0, 0, 0]
    assert receipt.zeroized_bytes == 3
    assert receipt.secret_type == "list"


# 80. Violation journal and recovery simulation ------------------------------


def test_violation_journal_records_and_recovers_deterministically() -> None:
    journal = sandbox_violation_trap_recover(
        [("bounds", "index outside allocation", "fallback")],
        span=(70, 75),
        line=10,
        col=11,
    )
    first = journal.record("taint", "dynamic code sink", strategy="skip")
    assert first.index == 1
    assert first.location.span == (70, 75)
    assert len(journal.pending()) == 2
    recovered = journal.recover(0)
    assert recovered.recovered
    assert "no signal handler installed" in recovered.recovery_note
    assert [entry.index for entry in journal.pending()] == [1]
    assert journal.snapshot()[0].message == "index outside allocation"


def test_violation_signal_handlers_are_rejected_with_span() -> None:
    with pytest.raises(PillarError) as excinfo:
        sandbox_violation_trap_recover(span=(100, 108), line=21, col=22, execute=True)
    error = _assert_error(excinfo, "E007", (100, 108), 21, 22)
    assert "signal trapping" in str(error)


def test_violation_journal_validates_records_and_recovery() -> None:
    journal = sandbox_violation_trap_recover()
    with pytest.raises(PillarError) as excinfo:
        journal.record("bad-kind", "message")
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        journal.record("bounds", "message", strategy="teleport")
    _assert_error(excinfo, "E003")
    with pytest.raises(PillarError) as excinfo:
        journal.recover(3)
    _assert_error(excinfo, "E006")
    journal.record("bounds", "message")
    journal.recover(0)
    with pytest.raises(PillarError) as excinfo:
        journal.recover(0)
    _assert_error(excinfo, "E003")


# Location validation across the public API ----------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda span: sandbox_mprotect_heap_isolate([("a", 1)], span=span),
        lambda span: sandbox_ebpf_syscall_synthesize(["read"], span=span),
        lambda span: sandbox_pku_mpk_domain_switch([("a", 0)], span=span),
        lambda span: sandbox_fat_pointer_bounds_verify(
            TaggedCapability("a", 0, 1), CapabilityEpoch("a", 1), span=span
        ),
        lambda span: sandbox_cfi_shadow_stack_emit([("a", 1)], span=span),
        lambda span: sandbox_spectre_lfence_inject([("a", "load-value")], span=span),
        lambda span: sandbox_deterministic_fuel_meter(1, span=span),
        lambda span: sandbox_taint_flow_enforce(span=span),
        lambda span: sandbox_zeroize_secret_drop(bytearray(b"x"), span=span),
        lambda span: sandbox_violation_trap_recover(span=span),
    ],
)
def test_every_public_api_rejects_malformed_spans(call: object) -> None:
    for bad_span in ((5,), (10, 2), (0, -1), [1, 2]):
        with pytest.raises(PillarError) as excinfo:
            call(bad_span)  # type: ignore[operator]
        _assert_error(excinfo, "E003")
