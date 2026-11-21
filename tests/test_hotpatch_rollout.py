"""Behavioral tests for framework batch 7 (hot-patching and live state).

All functions run against in-process dictionaries only.  Raw addresses are
opaque model keys, epochs are cooperative, and no executable text, page flag or
global runtime callback is touched by these tests.
"""

from __future__ import annotations

import importlib
import selectors
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from compiler.frontend.errors import PolyglotError
from runtime.hotpatch import audit, epochs, guards, tx
from runtime.hotpatch.rollout import FRAMEWORK_ROWS


@contextmanager
def _error_code(code: str) -> Iterator[None]:
    """Assert that the enclosed call raises a ``PolyglotError`` with *code*."""
    with pytest.raises(PolyglotError) as error:
        yield
    assert error.value.code == code


def _tx(state: dict | None = None):
    """Fresh draft transaction over a small model state."""
    return tx.hptx_begin("t1", state={"a": 1, "b": 2} if state is None else state)


def _prepared():
    """Prepared transaction with one staged change."""
    staged = tx.hptx_stage_change(_tx(), "a", 10)["transaction"]
    return tx.hptx_prepare(staged)["transaction"]


def _journal():
    """Two-entry commit/rollback journal from a real transaction."""
    committed = tx.hptx_commit(_prepared())
    return tx.hptx_rollback(committed["transaction"])["transaction"]["journal"]


def _domain():
    """Epoch domain with one registered reader."""
    state = epochs.hpe_domain("ast")
    return epochs.hpe_register_reader(state, "r1")["state"]


def _trail():
    """Audit trail with two model events."""
    trail = audit.hpa_trail("rollout", sinks=["jsonl"])
    for event in ("stage", "commit"):
        trail = audit.hpa_record(
            trail, event=event, actor="transaction", target="patch-1"
        )["trail"]
    return trail


# ---------------------------------------------------------------------------
# Rollout metadata
# ---------------------------------------------------------------------------


def test_rollout_metadata_has_100_resolvable_documented_callables() -> None:
    assert len(FRAMEWORK_ROWS) == 100
    assert len({name for name, _, _, _ in FRAMEWORK_ROWS}) == 100
    for name, module_name, category, test_module in FRAMEWORK_ROWS:
        assert category
        assert test_module == "tests.test_hotpatch_rollout"
        function = getattr(importlib.import_module(module_name), name)
        assert callable(function)
        assert function.__doc__
    new_rows = FRAMEWORK_ROWS[10:]
    assert len(new_rows) == 90
    assert [row[1] for row in new_rows].count("runtime.hotpatch.tx") == 25
    assert [row[1] for row in new_rows].count("runtime.hotpatch.guards") == 25
    assert [row[1] for row in new_rows].count("runtime.hotpatch.epochs") == 20
    assert [row[1] for row in new_rows].count("runtime.hotpatch.audit") == 20
    assert {row[2] for row in new_rows} == {
        "transactions",
        "guards",
        "epochs",
        "audit",
    }


def test_rollout_metadata_reuses_legacy_f6_names() -> None:
    assert [row[0] for row in FRAMEWORK_ROWS[:10]] == [
        "hotpatch_registry_lookup_capability",
        "hotpatch_validate_target_abi",
        "hotpatch_guard_raw_address_toctou",
        "hotpatch_apply_patch_transaction",
        "hotpatch_rollback_transaction",
        "hotpatch_update_frame_eval_hook",
        "hotpatch_migrate_reactor_registrations",
        "hotpatch_advance_rcu_epoch",
        "hotpatch_delegate_legacy_backend",
        "hotpatch_emit_diagnostic_e008",
    ]
    assert all(row[1] == "runtime.hotpatch.framework" for row in FRAMEWORK_ROWS[:10])


# ---------------------------------------------------------------------------
# Legacy FRAME 6 smoke coverage
# ---------------------------------------------------------------------------


def test_legacy_registry_lookup_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_registry_lookup_capability

    assert hotpatch_registry_lookup_capability("hotpatch_rcu_ast_epoch").active


def test_legacy_abi_validation_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_validate_target_abi

    assert hotpatch_validate_target_abi("polyglot_dispatch").ok


def test_legacy_address_guard_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_guard_raw_address_toctou

    token = hotpatch_guard_raw_address_toctou(
        0x1000,
        state={0x1000: {"value": 1, "version": 1}},
        expected_value=1,
        expected_version=1,
    )
    assert token.value == 1 and token.model_only


def test_legacy_patch_transaction_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_apply_patch_transaction

    assert hotpatch_apply_patch_transaction(()).committed


def test_legacy_rollback_transaction_smoke() -> None:
    from runtime.hotpatch.framework import (
        hotpatch_apply_patch_transaction,
        hotpatch_rollback_transaction,
    )

    transaction = hotpatch_apply_patch_transaction(())
    assert hotpatch_rollback_transaction(transaction).restored == 0


def test_legacy_frame_eval_hook_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_update_frame_eval_hook

    assert hotpatch_update_frame_eval_hook({}, "smoke_hook").name == "smoke_hook"


def test_legacy_reactor_migration_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_migrate_reactor_registrations

    selector = selectors.DefaultSelector()
    try:
        result = hotpatch_migrate_reactor_registrations(selector)
        assert result.packet_loss == 0
    finally:
        result.migration.new_selector.close()
        selector.close()


def test_legacy_rcu_epoch_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_advance_rcu_epoch

    assert hotpatch_advance_rcu_epoch().epoch >= 1


def test_legacy_backend_delegation_smoke() -> None:
    from backends.hotpatch import HotpatchManager
    from lexer import lex
    from parser import parse
    from runtime.hotpatch.framework import hotpatch_delegate_legacy_backend

    program = parse(lex("fn main() -> i64 { 0 }"))
    result = hotpatch_delegate_legacy_backend(
        HotpatchManager(program), action="prepare", new_program=program
    )
    assert result.route == "runtime.hotpatch"


def test_legacy_e008_diagnostic_smoke() -> None:
    from runtime.hotpatch.framework import hotpatch_emit_diagnostic_e008

    assert hotpatch_emit_diagnostic_e008("smoke").code == "E008"


# ---------------------------------------------------------------------------
# transactions (hptx_)
# ---------------------------------------------------------------------------


def test_hptx_begin_creates_independent_draft() -> None:
    record = _tx()
    assert record["status"] == "draft" and record["model_only"] is True
    assert record["state"] == {"a": 1, "b": 2}
    record["state"]["a"] = 99
    assert _tx()["state"]["a"] == 1
    with _error_code("E003"):
        tx.hptx_begin("")


def test_hptx_stage_change_records_before_image() -> None:
    result = tx.hptx_stage_change(_tx(), "a", 10)
    change = result["change"]
    assert (change["key"], change["kind"], change["old"], change["new"]) == (
        "a",
        "set",
        1,
        10,
    )
    assert change["seq"] == 1 and change["digest"]
    with _error_code("E003"):
        tx.hptx_stage_change(_tx(), "a", 10, kind="rewrite")
    with _error_code("E003"):
        tx.hptx_stage_change(result["transaction"], "a", 11)


def test_hptx_stage_many_stages_all_changes() -> None:
    result = tx.hptx_stage_many(
        _tx(), [{"key": "a", "value": 10}, {"key": "c", "value": 3, "kind": "insert"}]
    )
    assert [item["key"] for item in result["staged"]] == ["a", "c"]
    assert result["count"] == 2
    with _error_code("E003"):
        tx.hptx_stage_many(_tx(), [{"value": 1}])


def test_hptx_list_changes_sorted_by_sequence() -> None:
    staged = tx.hptx_stage_many(
        _tx(), [{"key": "c", "value": 3}, {"key": "a", "value": 10}]
    )["transaction"]
    listed = tx.hptx_list_changes(staged)
    assert listed["keys"] == ["c", "a"]
    assert [item["seq"] for item in listed["changes"]] == [1, 2]


def test_hptx_unstage_change_renumbers_remainder() -> None:
    staged = tx.hptx_stage_many(
        _tx(), [{"key": "a", "value": 10}, {"key": "b", "value": 20}]
    )["transaction"]
    result = tx.hptx_unstage_change(staged, "a")
    assert result["removed"]["key"] == "a"
    assert [item["seq"] for item in result["transaction"]["changes"]] == [1]
    with _error_code("E003"):
        tx.hptx_unstage_change(staged, "missing")


def test_hptx_preview_state_does_not_mutate_transaction() -> None:
    staged = tx.hptx_stage_change(_tx(), "a", 10)["transaction"]
    preview = tx.hptx_preview_state(staged)
    assert preview["state"]["a"] == 10
    assert staged["state"]["a"] == 1
    assert preview["applied"][0]["key"] == "a"


def test_hptx_validate_change_kind_rejects_abi_drift() -> None:
    change = tx.hptx_stage_change(_tx(), "a", 10)["change"]
    assert tx.hptx_validate_change_kind(change, current={"a": 1})["valid"]
    with _error_code("E001"):
        tx.hptx_validate_change_kind(change, current={"a": "old"})
    insert = {
        "key": "a",
        "kind": "insert",
        "old": None,
        "new": 10,
        "seq": 1,
        "had_old": True,
    }
    with _error_code("E001"):
        tx.hptx_validate_change_kind(insert, current={"a": 1})


def test_hptx_add_dependency_rejects_self_and_cycles() -> None:
    result = tx.hptx_add_dependency(_tx(), "t2")
    assert result["dependencies"] == ["t2"] and result["cycle_free"]
    with _error_code("E005"):
        tx.hptx_add_dependency(_tx(), "t1")
    with _error_code("E005"):
        tx.hptx_add_dependency(_tx(), "t3", graph={"t3": ["t1"]})


def test_hptx_detect_cycle_reports_cycle_path() -> None:
    clean = tx.hptx_detect_cycle({"a": ["b"], "b": []})
    assert not clean["has_cycle"] and clean["nodes"] == ["a", "b"]
    cyclic = tx.hptx_detect_cycle({"a": ["b"], "b": ["a"]})
    assert cyclic["has_cycle"] and set(cyclic["cycle"]) == {"a", "b"}
    assert cyclic["cyclic_nodes"] == ["a", "b"]


def test_hptx_order_by_dependency_levels() -> None:
    result = tx.hptx_order_by_dependency(["c", "a", "b"], {"b": ["a"], "c": ["b"]})
    assert result["order"] == ["a", "b", "c"]
    assert result["levels"] == [["a"], ["b"], ["c"]]
    with _error_code("E005"):
        tx.hptx_order_by_dependency(["a", "b"], {"a": ["b"], "b": ["a"]})


def test_hptx_check_preconditions_enforces_strict_mode() -> None:
    conditions = [{"key": "a", "op": "eq", "value": 1}, {"key": "z", "op": "absent"}]
    result = tx.hptx_check_preconditions({"a": 1}, conditions)
    assert result["all_ok"] and result["passed"] == 2
    with _error_code("E002"):
        tx.hptx_check_preconditions({"a": 2}, conditions, strict=True)


def test_hptx_reserve_id_skips_used_ids() -> None:
    first = tx.hptx_reserve_id("patch")
    assert first["id"] == "patch-0001"
    skipped = tx.hptx_reserve_id("patch", used=["patch-0001"])
    assert skipped["id"] == "patch-0002" and skipped["used_count"] == 1
    with _error_code("E003"):
        tx.hptx_reserve_id("patch", width=9)


def test_hptx_seal_and_unseal_toggle_mutable_state() -> None:
    staged = tx.hptx_stage_change(_tx(), "a", 10)["transaction"]
    sealed = tx.hptx_seal(staged)
    assert sealed["transaction"]["status"] == "sealed" and sealed["seal"]
    reopened = tx.hptx_unseal(sealed["transaction"])
    assert reopened["status"] == "draft" and reopened["previous_seal"]
    with _error_code("E003"):
        tx.hptx_seal(_tx())


def test_hptx_fingerprint_is_stable_and_state_sensitive() -> None:
    first = tx.hptx_fingerprint(_tx())
    second = tx.hptx_fingerprint(_tx())
    assert first["digest"] == second["digest"]
    assert first["algorithm"] == "sha256"
    changed = tx.hptx_fingerprint(_tx({"a": 7, "b": 2}))
    assert changed["digest"] != first["digest"]


def test_hptx_prepare_requires_changes() -> None:
    staged = tx.hptx_stage_change(_tx(), "a", 10)["transaction"]
    prepared = tx.hptx_prepare(staged)
    assert prepared["transaction"]["status"] == "prepared" and prepared["seal"]
    with _error_code("E003"):
        tx.hptx_prepare(_tx())


def test_hptx_commit_applies_changes_and_appends_journal() -> None:
    committed = tx.hptx_commit(_prepared())
    assert committed["state"]["a"] == 10
    assert committed["transaction"]["status"] == "committed"
    assert committed["journal_seq"] == 1 and committed["digest"]
    with _error_code("E003"):
        tx.hptx_commit(committed["transaction"])


def test_hptx_rollback_restores_before_image() -> None:
    committed = tx.hptx_commit(_prepared())
    rolled = tx.hptx_rollback(committed["transaction"])
    assert rolled["state"] == {"a": 1, "b": 2}
    assert rolled["transaction"]["status"] == "rolled_back"
    assert "a" in rolled["restored_keys"]
    with _error_code("E003"):
        tx.hptx_rollback(_tx())


def test_hptx_journal_append_and_slice() -> None:
    first = tx.hptx_journal_append(
        [], action="commit", tx_id="t1", before={"a": 1}, after={"a": 10}
    )
    second = tx.hptx_journal_append(
        first["journal"],
        action="rollback",
        tx_id="t1",
        before={"a": 10},
        after={"a": 1},
    )
    assert [item["seq"] for item in second["journal"]] == [1, 2]
    assert second["entry"]["prev"] == first["entry"]["digest"]
    assert tx.hptx_journal_slice(second["journal"], start=2)["count"] == 1
    with _error_code("E003"):
        tx.hptx_journal_append([], action="format", tx_id="t1", before={}, after={})


def test_hptx_verify_journal_rejects_corruption() -> None:
    journal = _journal()
    assert tx.hptx_verify_journal(journal)["ok"]
    broken = [dict(journal[0]), dict(journal[1])]
    broken[1]["digest"] = "tampered"
    with _error_code("E008"):
        tx.hptx_verify_journal(broken)


def test_hptx_replay_journal_honours_stop_after() -> None:
    journal = _journal()
    full = tx.hptx_replay_journal(journal)
    assert full["state"] == {"a": 1, "b": 2}
    assert full["actions"] == ["commit", "rollback"]
    partial = tx.hptx_replay_journal(journal, stop_after=1)
    assert partial["state"]["a"] == 10 and partial["applied"] == 1
    with _error_code("E003"):
        tx.hptx_replay_journal(journal, stop_after=0)


def test_hptx_diff_states_reports_add_remove_change() -> None:
    result = tx.hptx_diff_states({"a": 1, "b": 2}, {"a": 10, "c": 3})
    assert result["added"] == ["c"] and result["removed"] == ["b"]
    assert result["changed"][0]["old"] == 1
    assert result["changed"][0]["new"] == 10
    assert result["count"] == 3


def test_hptx_merge_changes_detects_conflicts() -> None:
    left = tx.hptx_stage_change(_tx(), "a", 10)["transaction"]["changes"]
    right = tx.hptx_stage_change(_tx(), "a", 20)["transaction"]["changes"]
    merged = tx.hptx_merge_changes(left, right)
    assert merged["conflicts"][0]["key"] == "a"
    assert merged["count"] == 1
    with _error_code("E003"):
        tx.hptx_merge_changes(left, right, strict=True)


def test_hptx_summarize_exposes_metadata_without_state() -> None:
    committed = tx.hptx_commit(_prepared())["transaction"]
    summary = tx.hptx_summarize(committed)
    assert summary["status"] == "committed" and summary["change_count"] == 1
    assert summary["keys"] == ["a"] and summary["journal_count"] == 1
    assert summary["digest"] and summary["model_only"] is True
    assert "state" not in summary


# ---------------------------------------------------------------------------
# guards (hpg_)
# ---------------------------------------------------------------------------


def test_hpg_abi_shape_parses_and_rejects_mismatch() -> None:
    shape = guards.hpg_abi_shape("i64 (i64, ptr)", expected="i64(i64,ptr)")
    assert shape["return_type"] == "i64" and shape["arity"] == 2
    assert shape["matches"] and shape["model_only"]
    with _error_code("E001"):
        guards.hpg_abi_shape("i64(i64)", expected="i64(i64, ptr)")
    with _error_code("E001"):
        guards.hpg_abi_shape("i64(i64")


def test_hpg_address_tag_and_check_raw_address() -> None:
    tag = guards.hpg_address_tag(0x1000, namespace="dispatch")
    assert tag["key"] == "dispatch:0x1000" and tag["model_only"]
    registry = {0x1000: {"tag": "cell", "namespace": "model"}}
    checked = guards.hpg_check_raw_address(0x1000, registry=registry, generation=1)
    assert checked["tag"] == "cell" and checked["generation"] == 1
    with _error_code("E007"):
        guards.hpg_check_raw_address(
            0x2000,
            registry={
                0x2000: {"tag": "text", "namespace": "image", "loaded_image": True}
            },
        )
    with _error_code("E006"):
        guards.hpg_check_raw_address(0x3000, registry=registry)
    with _error_code("E001"):
        guards.hpg_check_raw_address(0x1000, registry=registry, generation=2)


def test_hpg_address_guard_token_checks_generation_and_epoch() -> None:
    token = guards.hpg_address_guard_token(0x1000, generation=2, epoch=3)
    assert token["key"] == "model:0x1000" and token["digest"]
    ok = guards.hpg_address_guard_check(token, generation=2, epoch=3)
    assert ok["ok"] and not ok["reasons"]
    assert not guards.hpg_address_guard_check(token, generation=3)["ok"]
    with _error_code("E001"):
        guards.hpg_address_guard_check(token, epoch=4, strict=True)
    broken = dict(token, digest="bad")
    with _error_code("E008"):
        guards.hpg_address_guard_check(broken)


def test_hpg_address_generation_bump_chains_registry() -> None:
    registry = {0x1000: {"tag": "cell", "namespace": "model"}}
    result = guards.hpg_address_generation_bump(registry, 0x1000, by=2)
    assert result["previous"] == 1 and result["generation"] == 3
    assert result["registry"][0x1000]["generation"] == 3
    with _error_code("E006"):
        guards.hpg_address_generation_bump(registry, 0x9999)


def test_hpg_layout_compat_allows_controlled_growth() -> None:
    base = {"value": 8}
    grown = guards.hpg_layout_compat(
        {"value": 8, "extra": 4}, expected=base, allow_growth=True
    )
    assert grown["added"] == ["extra"] and grown["compatible"]
    assert guards.hpg_layout_compat(base, expected=base)["compatible"]
    with _error_code("E001"):
        guards.hpg_layout_compat({"value": 8, "extra": 4}, expected=base)
    with _error_code("E001"):
        guards.hpg_layout_compat({}, expected=base)
    with _error_code("E001"):
        guards.hpg_layout_compat({"value": 16}, expected=base)


def test_hpg_version_token_check_reports_stale_tokens() -> None:
    assert guards.hpg_version_token_check(4, expected=4)["matches"]
    stale = guards.hpg_version_token_check(3, expected=4, strict=False)
    assert not stale["matches"] and stale["model_only"]
    with _error_code("E001"):
        guards.hpg_version_token_check(3, expected=4)


def test_hpg_frame_layout_check_detects_overlap() -> None:
    layout = {
        "a": {"offset": 0, "size": 8},
        "b": {"offset": 8, "size": 4},
    }
    result = guards.hpg_frame_layout_check(layout, frame_size=16)
    assert result["total_size"] == 12 and [f["field"] for f in result["fields"]] == [
        "a",
        "b",
    ]
    with _error_code("E001"):
        guards.hpg_frame_layout_check(
            {"a": {"offset": 0, "size": 8}, "b": {"offset": 4, "size": 8}}
        )
    with _error_code("E001"):
        guards.hpg_frame_layout_check(layout, frame_size=8)


def test_hpg_frame_eval_slot_enforces_runtime_boundary() -> None:
    slot = guards.hpg_frame_eval_slot(2, version=(3, 13), tool_id=1)
    assert slot["slot"] == 2 and slot["version"] == "3.13" and slot["supported"]
    with _error_code("E007"):
        guards.hpg_frame_eval_slot(0, version=(3, 11))
    with _error_code("E003"):
        guards.hpg_frame_eval_slot(6)


def test_hpg_hook_plan_and_slot_conflict() -> None:
    plan = guards.hpg_hook_plan("bump", slot=1, callback="python_bridge")
    assert plan["digest"] and plan["installed"] is False
    assert guards.hpg_hook_plan("other", slot=1)["symbol"] == "other"
    conflicts = guards.hpg_hook_slot_conflict(
        [
            {"slot": 1, "symbol": "a", "callback": "python_bridge"},
            {"slot": 1, "symbol": "b", "callback": "python_bridge"},
        ]
    )
    assert conflicts["count"] == 1 and conflicts["conflicts"][0]["slot"] == 1
    with _error_code("E007"):
        guards.hpg_hook_plan("bump", callback="native_frame_eval")


def test_hpg_reactor_registration_guard_validates_events() -> None:
    record = {"fd": 4, "events": ["read", "write"], "callback": "on_ready"}
    checked = guards.hpg_reactor_registration_guard(record)
    assert checked["events"] == ["read", "write"] and checked["digest"]
    with _error_code("E003"):
        guards.hpg_reactor_registration_guard({**record, "events": ["hover"]})
    with _error_code("E007"):
        guards.hpg_reactor_registration_guard({**record, "backend": "native_mio"})


def test_hpg_reactor_snapshot_diff_tracks_fd_changes() -> None:
    before = [{"fd": 1, "events": ["read"], "callback": "a"}]
    after = [
        {"fd": 1, "events": ["write"], "callback": "a"},
        {"fd": 2, "events": ["read"], "callback": "b"},
    ]
    result = guards.hpg_reactor_snapshot_diff(before, after)
    assert [item["fd"] for item in result["added"]] == [2]
    assert [item["fd"] for item in result["changed"]] == [1]
    assert result["count"] == 2


def test_hpg_loaded_image_guard_refuses_rewrite() -> None:
    allowed = guards.hpg_loaded_image_guard(
        "module", operation="inspect", loaded_image=True
    )
    assert allowed["allowed"] and allowed["model_only"]
    with _error_code("E007"):
        guards.hpg_loaded_image_guard("module", operation="rewrite", loaded_image=True)


def test_hpg_native_pattern_guard_classifies_patterns() -> None:
    assert guards.hpg_native_pattern_guard("model")["classification"] == "model-safe"
    with _error_code("E007"):
        guards.hpg_native_pattern_guard("mprotect")
    with _error_code("E007"):
        guards.hpg_native_pattern_guard("unknown-native-seam")


def test_hpg_page_protection_guard_never_changes_flags() -> None:
    result = guards.hpg_page_protection_guard(address=0x1000)
    assert result["changed"] is False and result["model_only"]
    with _error_code("E007"):
        guards.hpg_page_protection_guard(protection="read-write")


def test_hpg_symbol_resolution_guard_requires_known_symbol() -> None:
    result = guards.hpg_symbol_resolution_guard(
        "dispatch", symbols=["dispatch", "bump"]
    )
    assert result["resolved"] and result["candidates"] == ["dispatch", "bump"]
    with _error_code("E006"):
        guards.hpg_symbol_resolution_guard("missing", symbols=["dispatch"])


def test_hpg_capability_guard_rejects_unsupported_backend() -> None:
    result = guards.hpg_capability_guard("hotpatch", supported=["hotpatch"])
    assert result["active"] and result["backend"] == "python"
    with _error_code("E006"):
        guards.hpg_capability_guard("other", supported=["hotpatch"])
    with _error_code("E007"):
        guards.hpg_capability_guard(
            "hotpatch", backend="native", supported=["hotpatch"]
        )
    with _error_code("E003"):
        guards.hpg_capability_guard("hotpatch", backend="wasm", supported=["hotpatch"])


def test_hpg_epoch_guard_detects_regression() -> None:
    assert guards.hpg_epoch_guard(3, current=2)["ok"]
    assert guards.hpg_epoch_guard(1, current=2)["regression"]
    with _error_code("E002"):
        guards.hpg_epoch_guard(1, current=2, strict=True)


def test_hpg_journal_guard_validates_chain() -> None:
    assert guards.hpg_journal_guard(None)["count"] == 0
    journal = _journal()
    verdict = guards.hpg_journal_guard(journal)
    assert verdict["ok"] and verdict["count"] == 2 and verdict["head"]
    with _error_code("E008"):
        guards.hpg_journal_guard([{"seq": 5, "action": "commit", "digest": "x"}])


def test_hpg_dependency_guard_finds_cycles() -> None:
    result = guards.hpg_dependency_guard("b", graph={"b": ["a"], "a": []})
    assert result["ancestors"] == ["a"] and result["cycle_free"]
    with _error_code("E005"):
        guards.hpg_dependency_guard("a", graph={"a": ["a"]})
    with _error_code("E005"):
        guards.hpg_dependency_guard("a", graph={"a": ["b"], "b": ["a"]})


def test_hpg_change_scope_guard_splits_keys() -> None:
    result = guards.hpg_change_scope_guard(
        [{"key": "model.a"}, {"key": "foreign.text"}], allowed_prefixes=["model."]
    )
    assert result["accepted"] == ["model.a"]
    assert result["rejected"] == ["foreign.text"]
    with _error_code("E003"):
        guards.hpg_change_scope_guard(
            ["foreign.text"], allowed_prefixes=["model."], strict=True
        )


def test_hpg_audit_trail_guard_normalizes_records() -> None:
    checked = guards.hpg_audit_trail_guard(
        {"seq": 2, "event": "commit", "actor": "tx", "target": "patch"}
    )
    assert checked["record"]["seq"] == 2 and checked["digest"]
    with _error_code("E008"):
        guards.hpg_audit_trail_guard({"event": "commit"})


def test_hpg_boundary_report_aggregates_decisions() -> None:
    report = guards.hpg_boundary_report(
        [
            {"name": "abi", "allowed": True, "code": "OK"},
            {"name": "image", "allowed": False, "code": "E007"},
        ]
    )
    assert (report["total"], report["allowed"], report["denied"]) == (2, 1, 1)
    assert report["codes"] == {"E007": 1, "OK": 1}
    assert report["model_only"] is True


# ---------------------------------------------------------------------------
# epochs (hpe_)
# ---------------------------------------------------------------------------


def test_hpe_domain_creates_empty_cooperative_state() -> None:
    state = epochs.hpe_domain("ast", epoch=3)
    assert state["domain"] == "ast" and state["current"] == 3
    assert state["readers"] == {} and state["model_only"] is True
    with _error_code("E003"):
        epochs.hpe_domain("")


def test_hpe_register_and_unregister_reader() -> None:
    state = epochs.hpe_register_reader(epochs.hpe_domain("ast"), "r1")["state"]
    assert state["registered"] == ["r1"]
    active = epochs.hpe_enter_read_section(state, "r1")["state"]
    with _error_code("E002"):
        epochs.hpe_unregister_reader(active, "r1")
    removed = epochs.hpe_unregister_reader(active, "r1", force=True)
    assert removed["was_active"] and removed["state"]["registered"] == []
    with _error_code("E003"):
        epochs.hpe_unregister_reader(state, "missing")


def test_hpe_enter_and_exit_read_section_track_epochs() -> None:
    entered = epochs.hpe_enter_read_section(_domain(), "r1")
    assert entered["epoch"] == 0 and entered["state"]["readers"] == {"r1": 0}
    with _error_code("E002"):
        epochs.hpe_enter_read_section(entered["state"], "r1")
    exited = epochs.hpe_exit_read_section(entered["state"], "r1")
    assert exited["observed_epoch"] == 0 and exited["state"]["readers"] == {}
    with _error_code("E003"):
        epochs.hpe_exit_read_section(exited["state"], "r1")


def test_hpe_reader_snapshot_and_min_reader_epoch() -> None:
    state = epochs.hpe_register_reader(epochs.hpe_domain("ast"), "r1")["state"]
    state = epochs.hpe_register_reader(state, "r2")["state"]
    state = epochs.hpe_enter_read_section(state, "r2")["state"]
    snapshot = epochs.hpe_reader_snapshot(state)
    assert [item["token"] for item in snapshot["readers"]] == ["r1", "r2"]
    active = epochs.hpe_reader_snapshot(state, active_only=True)
    assert [item["token"] for item in active["readers"]] == ["r2"]
    assert epochs.hpe_min_reader_epoch(state)["epoch"] == 0


def test_hpe_advance_records_history() -> None:
    result = epochs.hpe_advance(epochs.hpe_domain("ast"), steps=2)
    assert result["previous"] == 0 and result["current"] == 2
    assert result["state"]["history"][0]["to"] == 2
    with _error_code("E003"):
        epochs.hpe_advance(epochs.hpe_domain("ast"), steps=0)


def test_hpe_retire_and_retired_queue_sorted() -> None:
    state = epochs.hpe_advance(epochs.hpe_domain("ast"), steps=1)["state"]
    state = epochs.hpe_retire(state, "z")["state"]
    state = epochs.hpe_retire(state, "a")["state"]
    queue = epochs.hpe_retired_queue(state, min_epoch=1)
    assert [item["id"] for item in queue["retired"]] == ["a", "z"]
    with _error_code("E003"):
        epochs.hpe_retire(state, "a")


def test_hpe_grace_period_waits_for_reader_exit() -> None:
    state = epochs.hpe_enter_read_section(_domain(), "r1")["state"]
    state = epochs.hpe_retire(state, "old")["state"]
    blocked = epochs.hpe_grace_period(state)
    assert blocked["ready"] == []
    assert blocked["nodes"][0]["blocking"] == ["r1"]
    state = epochs.hpe_advance(state)["state"]
    assert epochs.hpe_grace_period(state)["ready"] == []
    state = epochs.hpe_exit_read_section(state, "r1")["state"]
    assert epochs.hpe_grace_period(state)["ready"] == ["old"]


def test_hpe_reclaim_ready_and_reclaim_append_callbacks() -> None:
    state = epochs.hpe_retire(_domain(), "old")["state"]
    assert epochs.hpe_reclaim_ready(state)["ids"] == ["old"]
    result = epochs.hpe_reclaim(state)
    assert result["reclaimed"][0]["id"] == "old"
    assert result["state"]["retired"] == []
    assert epochs.hpe_callback_log(result["state"], name="reclaim")["count"] == 1


def test_hpe_min_reader_epoch_reports_oldest_snapshot() -> None:
    state = epochs.hpe_domain("ast")
    state = epochs.hpe_register_reader(state, "r1")["state"]
    state = epochs.hpe_enter_read_section(state, "r1")["state"]
    advanced = epochs.hpe_advance(state, steps=3)["state"]
    result = epochs.hpe_min_reader_epoch(advanced)
    assert result["epoch"] == 0 and result["current"] == 3
    assert epochs.hpe_min_reader_epoch(epochs.hpe_domain("ast"))["epoch"] is None


def test_hpe_epoch_delta_compares_two_domains() -> None:
    before = epochs.hpe_domain("ast")
    after = epochs.hpe_advance(epochs.hpe_retire(before, "old")["state"])["state"]
    delta = epochs.hpe_epoch_delta(before, after)
    assert delta["epoch_advance"] == 1 and delta["retired_added"] == ["old"]
    assert delta["history_records"] == 1
    with _error_code("E003"):
        epochs.hpe_epoch_delta(before, epochs.hpe_domain("other"))


def test_hpe_epoch_fingerprint_and_guard_token_bind_state() -> None:
    state = epochs.hpe_domain("ast", epoch=2)
    fingerprint = epochs.hpe_epoch_fingerprint(state)
    assert fingerprint["digest"] and fingerprint["algorithm"] == "sha256"
    token = epochs.hpe_epoch_guard_token(state)
    assert token["epoch"] == 2 and token["digest"] != fingerprint["digest"]
    assert epochs.hpe_epoch_fingerprint(state)["digest"] == fingerprint["digest"]


def test_hpe_domain_merge_unifies_readers() -> None:
    left = epochs.hpe_register_reader(epochs.hpe_domain("ast"), "r1")["state"]
    right = epochs.hpe_register_reader(epochs.hpe_domain("ast"), "r2")["state"]
    merged = epochs.hpe_domain_merge(left, right)
    assert merged["state"]["registered"] == ["r1", "r2"]
    divergent = epochs.hpe_domain_merge(left, epochs.hpe_advance(right)["state"])
    assert divergent["state"]["current"] == 1 and divergent["conflicts"]
    with _error_code("E002"):
        epochs.hpe_domain_merge(
            left, epochs.hpe_advance(right)["state"], strategy="strict"
        )


def test_hpe_validate_epoch_state_rejects_malformed_state() -> None:
    verdict = epochs.hpe_validate_epoch_state(epochs.hpe_domain("ast"))
    assert verdict["ok"] and verdict["epoch"] == 0
    with _error_code("E008"):
        epochs.hpe_validate_epoch_state({"domain": "ast"})


def test_hpe_epoch_report_summarizes_domain() -> None:
    state = epochs.hpe_retire(_domain(), "old")["state"]
    report = epochs.hpe_epoch_report(state)
    assert report["domain"] == "ast" and report["active_readers"] == 0
    assert report["retired"] == 1 and report["reclaimable"] == 1
    assert report["digest"] and report["model_only"] is True


# ---------------------------------------------------------------------------
# audit (hpa_)
# ---------------------------------------------------------------------------


def test_hpa_trail_starts_empty_with_model_sinks() -> None:
    trail = audit.hpa_trail("rollout", sinks=["jsonl"])
    assert trail["entries"] == [] and trail["head"] is None
    assert trail["sinks"] == ["jsonl"] and trail["model_only"] is True
    with _error_code("E003"):
        audit.hpa_trail("")


def test_hpa_record_and_verify_chain() -> None:
    trail = _trail()
    assert [entry["event"] for entry in trail["entries"]] == ["stage", "commit"]
    assert [entry["seq"] for entry in trail["entries"]] == [1, 2]
    assert audit.hpa_verify_chain(trail)["ok"]
    with _error_code("E006"):
        audit.hpa_audit_guard(trail)
    assert audit.hpa_audit_guard(trail, required=["stage", "commit"])["ok"]


def test_hpa_record_batch_appends_in_order() -> None:
    result = audit.hpa_record_batch(
        audit.hpa_trail("batch"),
        [
            {"event": "stage", "actor": "a", "target": "t"},
            {"event": "commit", "actor": "a", "target": "t", "detail": "ok"},
        ],
    )
    assert result["count"] == 2
    assert [item["event"] for item in result["entries"]] == ["stage", "commit"]
    with _error_code("E003"):
        audit.hpa_record_batch(audit.hpa_trail("batch"), [{"event": "stage"}])


def test_hpa_entry_digest_depends_on_fields() -> None:
    first = audit.hpa_entry_digest(
        {"event": "a", "actor": "b", "target": "c", "detail": "d"}
    )
    second = audit.hpa_entry_digest(
        {"event": "a", "actor": "b", "target": "c", "detail": "e"}
    )
    assert first["digest"] != second["digest"] and first["algorithm"] == "sha256"


def test_hpa_verify_chain_rejects_tampered_digest() -> None:
    trail = _trail()
    broken = dict(trail, entries=[dict(trail["entries"][0]), dict(trail["entries"][1])])
    broken["entries"][1]["detail"] = "tampered"
    with _error_code("E008"):
        audit.hpa_verify_chain(broken)


def test_hpa_filter_entries_by_event_actor_and_prefix() -> None:
    trail = _trail()
    exact = audit.hpa_filter_entries(trail, event="commit")
    assert exact["count"] == 1 and exact["entries"][0]["event"] == "commit"
    all_events = audit.hpa_filter_entries(trail, actor="trans", prefix=True)
    assert all_events["count"] == 2


def test_hpa_entries_by_actor_and_target_group_deterministically() -> None:
    trail = _trail()
    by_actor = audit.hpa_entries_by_actor(trail)
    assert by_actor["counts"] == {"transaction": 2}
    by_target = audit.hpa_entries_by_target(trail)
    assert by_target["counts"] == {"patch-1": 2}
    assert list(by_target["targets"]) == ["patch-1"]


def test_hpa_timeline_and_counter_summary() -> None:
    trail = _trail()
    timeline = audit.hpa_timeline(trail)
    assert [(item["seq"], item["event"]) for item in timeline["events"]] == [
        (1, "stage"),
        (2, "commit"),
    ]
    summary = audit.hpa_counter_summary(trail)
    assert summary["total"] == 2 and summary["events"] == {"commit": 1, "stage": 1}


def test_hpa_fingerprint_is_canonical_and_stable() -> None:
    assert (
        audit.hpa_fingerprint(_trail())["digest"]
        == audit.hpa_fingerprint(_trail())["digest"]
    )
    other = audit.hpa_record(_trail(), event="extra", actor="a", target="t")["trail"]
    assert (
        audit.hpa_fingerprint(other)["digest"]
        != audit.hpa_fingerprint(_trail())["digest"]
    )


def test_hpa_serialize_and_deserialize_round_trip() -> None:
    serialized = audit.hpa_serialize(_trail())
    assert serialized["length"] == len(serialized["text"])
    restored = audit.hpa_deserialize(serialized["text"])
    assert restored["count"] == 2 and restored["trail"]["trail"] == "rollout"
    assert audit.hpa_fingerprint(restored["trail"])["digest"] == serialized["digest"]
    with _error_code("E008"):
        audit.hpa_deserialize("{not json")


def test_hpa_merge_trails_rebases_sequences() -> None:
    left = audit.hpa_record(
        audit.hpa_trail("rollout"), event="stage", actor="a", target="t"
    )["trail"]
    right = audit.hpa_record(
        audit.hpa_trail("rollout"), event="commit", actor="a", target="t"
    )["trail"]
    merged = audit.hpa_merge_trails(left, right)
    assert [item["seq"] for item in merged["trail"]["entries"]] == [1, 2]
    assert audit.hpa_verify_chain(merged["trail"])["ok"]
    with _error_code("E003"):
        audit.hpa_merge_trails(left, audit.hpa_trail("other"))


def test_hpa_slice_trail_rebuilds_chain() -> None:
    sliced = audit.hpa_slice_trail(_trail(), start=2, stop=3)
    assert sliced["count"] == 1 and sliced["entries"][0]["seq"] == 1
    assert sliced["entries"][0]["event"] == "commit"
    assert audit.hpa_verify_chain(sliced["trail"])["ok"]


def test_hpa_redact_changes_detail_and_digest() -> None:
    trail = audit.hpa_record(
        _trail(), event="secret", actor="a", target="t", detail="token"
    )["trail"]
    result = audit.hpa_redact(trail)
    assert result["redacted"] == 3
    assert all(entry["detail"] == "[redacted]" for entry in result["trail"]["entries"])
    assert result["trail"]["head"] != trail["head"]


def test_hpa_diff_trails_reports_added_entries() -> None:
    old = _trail()
    new = audit.hpa_record(old, event="extra", actor="a", target="t")["trail"]
    result = audit.hpa_diff_trails(old, new)
    assert result["added_count"] == 1 and result["removed_count"] == 0
    assert result["added"][0]["event"] == "extra"


def test_hpa_export_summary_is_model_only() -> None:
    summary = audit.hpa_export_summary(_trail())
    assert summary["trail"] == "rollout" and summary["event_count"] == 2
    assert summary["events"] == ["commit", "stage"]
    assert summary["sinks"] == ["jsonl"] and summary["model_only"] is True


def test_hpa_retention_prune_and_audit_guard() -> None:
    trail = _trail()
    pruned = audit.hpa_retention_prune(trail, keep=1)
    assert pruned["kept"] == 1 and pruned["removed"][0]["event"] == "stage"
    assert audit.hpa_verify_chain(pruned["trail"])["ok"]
    with _error_code("E003"):
        audit.hpa_retention_prune(trail)
    with _error_code("E006"):
        audit.hpa_audit_guard(pruned["trail"], required=["commit", "rollback"])
