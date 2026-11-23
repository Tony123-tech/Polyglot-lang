"""FRAME 6 framework functions: registry, guards, transactions, live state.

All state is in-process (model dictionaries, temporary files, local selectors
and legacy ``HotpatchManager`` namespaces).  No test patches a foreign code
segment, dereferences a raw address, or changes a page protection flag.
"""

from __future__ import annotations

import contextlib
import selectors
import socket
from pathlib import Path
from typing import Any

import pytest

from backends.hotpatch import HotpatchManager
from compiler.core.lexer import lex
from compiler.core.parser import parse
from runtime.diagnostics import PillarError
from runtime.hotpatch import HotpatchError, RcuDomain, RollbackJournal, byte_offsets
from runtime.hotpatch.framework import (
    FrameEvalHook,
    PatchStep,
    hotpatch_advance_rcu_epoch,
    hotpatch_apply_patch_transaction,
    hotpatch_delegate_legacy_backend,
    hotpatch_emit_diagnostic_e008,
    hotpatch_guard_raw_address_toctou,
    hotpatch_migrate_reactor_registrations,
    hotpatch_registry_lookup_capability,
    hotpatch_rollback_transaction,
    hotpatch_update_frame_eval_hook,
    hotpatch_validate_target_abi,
)

BASE = """
struct Counter { value: i64 }

impl Counter {
    fn bump(&self) -> i64 { self.value + 1 }
}

fn main() -> i64 { 0 }
"""

BUMP_2 = BASE.replace("self.value + 1", "self.value + 2")

SIGNATURE = BASE.replace(
    "fn bump(&self) -> i64 { self.value + 1 }",
    "fn bump(&self, step: i64) -> i64 { self.value + step }",
)


def _program(source: str) -> Any:
    return parse(lex(source))


# ---------------------------------------------------------------------------
# 51. Registry lookup
# ---------------------------------------------------------------------------


def test_registry_lookup_by_name_number_and_boundaries() -> None:
    by_name = hotpatch_registry_lookup_capability("hotpatch_cpython_frame_eval")
    assert (by_name.number, by_name.status, by_name.active) == (
        2,
        "foundation",
        True,
    )
    assert callable(by_name.handler)
    assert by_name.model_only is False

    by_number = hotpatch_registry_lookup_capability(7)
    assert by_number.name == "hotpatch_rollback_journal_commit"
    assert by_number.active

    native = hotpatch_registry_lookup_capability(
        "hotpatch_rollback_journal_commit", backend="native"
    )
    assert native.backend == "native" and native.active
    assert native.handler is by_number.handler

    modelled = hotpatch_registry_lookup_capability("hotpatch_rcu_ast_epoch")
    assert modelled.active and modelled.model_only

    rejected = hotpatch_registry_lookup_capability("hotpatch_tls_slot_relocate")
    assert not rejected.active and rejected.handler is None
    assert rejected.code == "E007" and "rejected" in rejected.reason


def test_registry_lookup_rejects_unknown_id_and_backend() -> None:
    with pytest.raises(PillarError) as unknown:
        hotpatch_registry_lookup_capability("not_a_capability")
    assert unknown.value.code == "E006"

    with pytest.raises(PillarError) as bad_number:
        hotpatch_registry_lookup_capability(101)
    assert bad_number.value.code == "E006"

    with pytest.raises(PillarError) as bad_backend:
        hotpatch_registry_lookup_capability(1, backend="wasm")
    assert bad_backend.value.code == "E003"


# ---------------------------------------------------------------------------
# 52. ABI validation
# ---------------------------------------------------------------------------


def test_validate_target_abi_accepts_compatible_plan() -> None:
    result = hotpatch_validate_target_abi(
        "polyglot_dispatch",
        signature="i64(i64, ptr)",
        expected_signature="i64 (i64, ptr)",
        layout={"value": 8},
        expected_layout={"value": 8},
        version_token=4,
        expected_version=4,
    )
    assert result.ok and result.model_only
    assert result.signature == "i64(i64,ptr)"
    assert result.fields == ("value",)
    assert any("version token" in item for item in result.preconditions)
    assert "No executable bytes" in result.boundary


def test_validate_target_abi_rejects_drift_and_foreign_text() -> None:
    with pytest.raises(HotpatchError) as bad_symbol:
        hotpatch_validate_target_abi("bad symbol!")
    assert bad_symbol.value.code == "E008"

    with pytest.raises(HotpatchError) as bad_abi:
        hotpatch_validate_target_abi(
            "polyglot_dispatch",
            signature="i64(i64)",
            expected_signature="i64(i64, ptr)",
        )
    assert bad_abi.value.code == "E008" and "signature mismatch" in str(bad_abi.value)

    with pytest.raises(HotpatchError) as extra_field:
        hotpatch_validate_target_abi(
            "polyglot_dispatch",
            layout={"value": 8, "extra": 4},
            expected_layout={"value": 8},
        )
    assert extra_field.value.code == "E008" and "extra" in str(extra_field.value)

    with pytest.raises(HotpatchError) as stale:
        hotpatch_validate_target_abi(
            "polyglot_dispatch", version_token=1, expected_version=2
        )
    assert stale.value.code == "E008"

    with pytest.raises(HotpatchError) as foreign:
        hotpatch_validate_target_abi("polyglot_dispatch", foreign_text=True)
    assert foreign.value.code == "E007"

    grown = hotpatch_validate_target_abi(
        "polyglot_dispatch",
        layout={"value": 8, "extra": 4},
        expected_layout={"value": 8},
        allow_layout_growth=True,
    )
    assert grown.fields == ("value", "extra")


# ---------------------------------------------------------------------------
# 53. TOCTOU guard
# ---------------------------------------------------------------------------


def test_guard_raw_address_toctou_revalidates_model_cell() -> None:
    state: dict[int, dict[str, Any]] = {
        0x1000: {"value": 0xAA, "version": 3, "label": "slot"}
    }
    token = hotpatch_guard_raw_address_toctou(
        0x1000, state=state, expected_value=0xAA, expected_version=3
    )
    assert token.value == 0xAA and token.version == 3
    assert token.model_only and token.fingerprint
    assert token.still_valid()
    again = hotpatch_guard_raw_address_toctou(
        0x1000, state=state, expected_value=0xAA, expected_version=3
    )
    assert again.fingerprint == token.fingerprint


def test_guard_rejects_changed_state_stale_token_and_bad_address() -> None:
    state: dict[int, dict[str, int]] = {0x2000: {"value": 1, "version": 1}}

    def mutate() -> None:
        state[0x2000] = {"value": 2, "version": 2}

    with pytest.raises(HotpatchError) as changed:
        hotpatch_guard_raw_address_toctou(
            0x2000, state=state, expected_value=1, expected_version=1, recheck=mutate
        )
    assert changed.value.code == "E008"
    assert "between checks" in str(changed.value)

    with pytest.raises(HotpatchError) as stale:
        hotpatch_guard_raw_address_toctou(
            0x2000, state=state, expected_value=2, expected_version=1
        )
    assert "stale version" in str(stale.value)

    with pytest.raises(HotpatchError):
        hotpatch_guard_raw_address_toctou(0x2001, state=state)
    with pytest.raises(HotpatchError):
        hotpatch_guard_raw_address_toctou(0x3000, state=state)


# ---------------------------------------------------------------------------
# 54. Atomic patch transactions
# ---------------------------------------------------------------------------


def test_apply_patch_transaction_commits_all_steps() -> None:
    state: dict[str, int] = {"a": 1, "b": 2}
    steps = (
        PatchStep("a", lambda: state.__setitem__("a", 10), key="a", state=state),
        PatchStep(
            "b",
            lambda: state.__setitem__("b", 20),
            undo=lambda: state.__setitem__("b", 2),
        ),
    )
    journal = RollbackJournal()
    transaction = hotpatch_apply_patch_transaction(
        steps,
        journal=journal,
        label="commit",
        verify=lambda: state == {"a": 10, "b": 20},
    )
    assert transaction.committed and not transaction.rolled_back
    assert transaction.applied == ("a", "b")
    assert transaction.verification == "pass"
    assert state == {"a": 10, "b": 20}
    assert journal.pending() == ()


def test_apply_patch_transaction_rolls_back_failed_and_unverified_steps() -> None:
    state: dict[str, int] = {"a": 1, "b": 2}

    def broken() -> None:
        state["b"] = 99  # partial mutation before the failure
        raise RuntimeError("boom")

    steps = (
        PatchStep("a", lambda: state.__setitem__("a", 10), key="a", state=state),
        PatchStep("broken", broken, key="b", state=state),
    )
    with pytest.raises(HotpatchError) as failure:
        hotpatch_apply_patch_transaction(steps, label="fail")
    assert failure.value.code == "E008"
    assert state == {"a": 1, "b": 2}
    assert failure.value.transaction.rolled_back
    assert failure.value.transaction.verification == "fail"

    already = hotpatch_rollback_transaction(failure.value.transaction)
    assert already.already_rolled_back and already.restored == 0

    state["a"] = 1
    with pytest.raises(HotpatchError) as unverified:
        hotpatch_apply_patch_transaction(
            (PatchStep("a", lambda: state.__setitem__("a", 5), key="a", state=state),),
            verify=lambda: False,
        )
    assert "verification failed" in str(unverified.value)
    assert state == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# 55. Explicit rollback
# ---------------------------------------------------------------------------


def test_rollback_transaction_restores_committed_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"value": 0}')
    state: dict[str, int] = {"value": 0}

    def apply() -> None:
        state["value"] = 7
        path.write_text('{"value": 7}')

    def undo() -> None:
        state["value"] = 0
        path.write_text('{"value": 0}')

    transaction = hotpatch_apply_patch_transaction(
        (PatchStep("value", apply, undo),), label="file"
    )
    assert path.read_text() == '{"value": 7}'

    outcome = hotpatch_rollback_transaction(
        transaction, verify=lambda: state["value"] == 0
    )
    assert outcome.restored == 1 and outcome.committed_before
    assert outcome.verify_passed and outcome.generation == 0
    assert state == {"value": 0}
    assert path.read_text() == '{"value": 0}'

    with pytest.raises(HotpatchError) as missing:
        hotpatch_rollback_transaction("no-such-transaction")
    assert missing.value.code == "E008"


# ---------------------------------------------------------------------------
# 56. Frame-evaluation hook management
# ---------------------------------------------------------------------------


def test_frame_eval_hook_install_replace_remove_and_reuse() -> None:
    hooks: dict[str, FrameEvalHook] = {}
    code = (lambda value: value).__code__

    first = hotpatch_update_frame_eval_hook(hooks, "line_hook", code=code)
    assert first.model_only and first.tool_id == 0 and first.generation == 1
    second = hotpatch_update_frame_eval_hook(hooks, "second", code=code)
    assert second.tool_id == 1

    replaced = hotpatch_update_frame_eval_hook(
        hooks, "line_hook", action="replace", code=code, live_locals=True
    )
    assert replaced.tool_id == 0 and replaced.generation == 2
    assert (
        hotpatch_update_frame_eval_hook(hooks, "line_hook", action="query") is replaced
    )

    removed = hotpatch_update_frame_eval_hook(hooks, "line_hook", action="remove")
    assert removed is replaced and "line_hook" not in hooks
    third = hotpatch_update_frame_eval_hook(hooks, "third", code=code)
    assert third.tool_id == 0


def test_frame_eval_hook_enforces_version_and_slot_limits() -> None:
    with pytest.raises(HotpatchError) as old:
        hotpatch_update_frame_eval_hook({}, "old", python_version=(3, 11))
    assert old.value.code == "E007"

    with pytest.raises(HotpatchError) as locals_gate:
        hotpatch_update_frame_eval_hook(
            {}, "locals", python_version=(3, 12), live_locals=True
        )
    assert locals_gate.value.code == "E007"

    hooks = {
        f"hook{index}": FrameEvalHook(
            name=f"hook{index}",
            code_name="target",
            tool_id=index,
            generation=1,
            version_floor=(3, 14),
            live_locals=False,
        )
        for index in range(6)
    }
    with pytest.raises(HotpatchError) as full:
        hotpatch_update_frame_eval_hook(hooks, "extra")
    assert full.value.code == "E007"


# ---------------------------------------------------------------------------
# 57. Reactor migration
# ---------------------------------------------------------------------------


def test_migrate_reactor_registrations_preserves_events_and_interests() -> None:
    left, right = socket.socketpair()
    selector = selectors.DefaultSelector()
    result = None
    try:
        selector.register(left, selectors.EVENT_READ, "payload")
        right.send(b"ready")
        result = hotpatch_migrate_reactor_registrations(
            selector,
            pending_events=[("manual", 1)],
            source_text="prototype",
        )
        assert result.packet_loss == 0 and not result.model_only
        assert len(result.registrations) == 1
        assert result.registrations[0].data == "payload"
        assert any(
            event.origin == "ready" and event.fd == left.fileno()
            for event in result.pending_events
        )
        assert any(
            event.origin == "caller" and event.data == ("manual", 1)
            for event in result.pending_events
        )
        replacement = result.migration.new_selector
        assert replacement.get_map()[left.fileno()].data == "payload"

        result.commit()
        assert result.committed
    finally:
        if result is not None and not result.committed and not result.rolled_back:
            result.rollback()
        if result is not None:
            with contextlib.suppress(Exception):
                result.migration.new_selector.close()
        with contextlib.suppress(Exception):
            selector.close()
        left.close()
        right.close()


def test_migrate_reactor_registrations_rejects_closed_selector() -> None:
    closed = selectors.DefaultSelector()
    closed.close()
    with pytest.raises(HotpatchError) as error:
        hotpatch_migrate_reactor_registrations(closed)
    assert error.value.code == "E008"


# ---------------------------------------------------------------------------
# 58. RCU epochs
# ---------------------------------------------------------------------------


def test_advance_rcu_epoch_defers_reclamation_until_readers_exit() -> None:
    domain = RcuDomain("framework-test")
    reclaimed: list[str] = []
    domain.enter()
    try:
        with pytest.raises(HotpatchError) as timeout:
            hotpatch_advance_rcu_epoch(
                domain, retire="old", reclaim=reclaimed.append, timeout=0.01
            )
        assert timeout.value.code == "E008"
        assert domain.stats().retired == 1 and reclaimed == []
    finally:
        domain.exit()

    result = hotpatch_advance_rcu_epoch(domain, timeout=1.0)
    assert result.reclaimed == 1 and reclaimed == ["old"]
    assert result.deferred == 0 and result.active_readers == 0
    assert result.epoch == result.previous_epoch + 1
    assert result.model_only


# ---------------------------------------------------------------------------
# 59. Legacy backend delegation
# ---------------------------------------------------------------------------


def test_delegate_legacy_backend_routes_through_runtime_registry() -> None:
    manager = HotpatchManager(_program(BASE))
    counter = manager.register(manager.namespace["Counter"](10))

    prepared = hotpatch_delegate_legacy_backend(
        manager, action="prepare", new_program=_program(BUMP_2)
    )
    assert prepared.route == "runtime.hotpatch"
    assert "bump:body" in prepared.changes

    applied = hotpatch_delegate_legacy_backend(manager, new_program=_program(BUMP_2))
    assert applied.generation == 2 and applied.journal_generation >= 1
    assert applied.committed and not applied.model_only
    assert counter.bump() == 12

    with pytest.raises(HotpatchError) as rejected:
        hotpatch_delegate_legacy_backend(manager, new_program=_program(SIGNATURE))
    assert rejected.value.code == "E008"
    assert counter.bump() == 12

    rolled_back = hotpatch_delegate_legacy_backend(manager, action="rollback")
    assert rolled_back.generation == 1
    assert counter.bump() == 11


def test_delegate_legacy_backend_refuses_foreign_text_patching() -> None:
    with pytest.raises(HotpatchError) as refused:
        hotpatch_delegate_legacy_backend(object(), action="native_patch_text")
    assert refused.value.code == "E007"


# ---------------------------------------------------------------------------
# 60. E008 diagnostics
# ---------------------------------------------------------------------------


def test_emit_diagnostic_e008_carries_utf8_byte_span() -> None:
    source = "let α = hot_patch();"
    start = source.index("hot_patch")
    span = (start, start + len("hot_patch"))
    emitted: list[Any] = []

    diagnostic = hotpatch_emit_diagnostic_e008(
        "patch refused",
        span=span,
        source=source,
        reason="ABI drift",
        detail="layout changed",
        capability="hotpatch_cpython_frame_eval",
        sink=emitted.append,
    )
    assert diagnostic.code == "E008"
    assert diagnostic.span == span
    assert diagnostic.byte_span == byte_offsets(source, span)
    assert diagnostic.byte_span is not None
    assert diagnostic.byte_span.byte_start > span[0]
    assert diagnostic.excerpt == "hot_patch"
    assert diagnostic.emitted and emitted == [diagnostic]
    assert "error[E008]" in diagnostic.formatted
    assert "layout changed" in diagnostic.reason

    with pytest.raises(HotpatchError) as raised:
        hotpatch_emit_diagnostic_e008(
            "fatal", span=span, source=source, raise_error=True
        )
    assert raised.value.code == "E008" and raised.value.span == span

    refusal = hotpatch_emit_diagnostic_e008("refused", unsupported_request=True)
    assert refusal.note.startswith("unsupported-hotpatch")
