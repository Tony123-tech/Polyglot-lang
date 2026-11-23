"""Disjoint splitting, generational arenas, and scoped cleanup plans.

``rsbe_arena_simulate`` is behaviour, not a description: it runs the slot and
epoch bookkeeping the emitted Rust would use, so the stale-handle tests assert
what a caller actually observes. The split tests pin the disjointness
argument and the code each violation carries (``E001`` bounds/name, ``E002``
overlap or a whole-struct borrow, ``E003`` API misuse).
"""

from __future__ import annotations

import pytest

from backends.rust.borrow_ergo import (
    rsbe_arena_plan,
    rsbe_arena_simulate,
    rsbe_scope_exit,
    rsbe_split_fields,
    rsbe_split_windows,
)
from compiler.frontend.errors import PolyglotError


def _code(fn, *args, **kwargs) -> str:
    with pytest.raises(PolyglotError) as caught:
        fn(*args, **kwargs)
    return caught.value.code


def _key(index: int, epoch: int) -> dict[str, int]:
    return {"index": index, "epoch": epoch}


# -- field splitting -----------------------------------------------------


def test_field_split_emits_one_mutable_view_per_field() -> None:
    plan = rsbe_split_fields(["left", "right"])
    assert plan["emitted"] == (
        "let left = &mut self.left;",
        "let right = &mut self.right;",
    )
    assert plan["disjoint"] is True
    assert plan["proof"].startswith("distinct fields")


def test_shared_split_emits_shared_references() -> None:
    plan = rsbe_split_fields(["left", "right"], mutable=False)
    assert plan["emitted"] == ("let left = & self.left;", "let right = & self.right;")
    assert plan["mutable"] is False


def test_split_is_refused_while_a_whole_struct_borrow_is_live() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsbe_split_fields(["left"], borrowed_whole=["all"])
    assert caught.value.code == "E002"
    assert "all" in str(caught.value)


def test_duplicate_field_views_are_api_misuse() -> None:
    assert _code(rsbe_split_fields, ["left", "left"]) == "E003"


def test_self_is_not_a_field() -> None:
    assert _code(rsbe_split_fields, ["self"]) == "E001"


def test_empty_field_list_is_api_misuse() -> None:
    assert _code(rsbe_split_fields, []) == "E003"


# -- window splitting ----------------------------------------------------


def test_adjacent_windows_cover_the_sequence_without_a_gap() -> None:
    plan = rsbe_split_windows(10, [(0, 3), (3, 7)])
    assert plan["covered"] == 7
    assert plan["gaps"] == ((7, 10),)
    assert plan["exhaustive"] is False
    assert [w["len"] for w in plan["windows"]] == [3, 4]


def test_contiguous_windows_are_exhaustive() -> None:
    plan = rsbe_split_windows(6, [(3, 6), (0, 3)])
    assert plan["exhaustive"] is True
    assert plan["gaps"] == ()
    assert plan["windows"][0]["range"] == (0, 3)


def test_overlapping_windows_are_refused_and_both_are_named() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsbe_split_windows(10, [(0, 4), (2, 6)])
    assert caught.value.code == "E002"
    assert "[0..4]" in str(caught.value) and "[2..6]" in str(caught.value)


def test_window_past_the_length_is_out_of_bounds() -> None:
    assert _code(rsbe_split_windows, 3, [(2, 5)]) == "E001"


def test_inverted_window_is_api_misuse() -> None:
    assert _code(rsbe_split_windows, 5, [(3, 3)]) == "E003"


def test_window_that_exactly_reaches_the_end_is_in_bounds() -> None:
    plan = rsbe_split_windows(5, [(0, 5)])
    assert plan["exhaustive"] is True


# -- generational arena --------------------------------------------------


def test_arena_plan_describes_the_key_and_slot_shapes() -> None:
    plan = rsbe_arena_plan(8, generations=16, item_type="Node")
    assert plan["capacity"] == 8
    assert plan["generations"] == 16
    assert plan["burned_after"] == 15
    assert any("ArenaKey" in line for line in plan["emitted"])
    assert plan["key_bytes"] == 8


def test_arena_capacity_must_be_positive() -> None:
    assert _code(rsbe_arena_plan, 0) == "E001"


def test_arena_generation_width_is_capped() -> None:
    assert _code(rsbe_arena_plan, 4, generations=33) == "E001"


def test_arena_insert_then_get_returns_the_value() -> None:
    result = rsbe_arena_simulate(
        [{"op": "insert", "value": "alpha"}, {"op": "get", "key": _key(0, 0)}]
    )
    assert [entry["outcome"] for entry in result["trace"]] == ["ok", "ok"]
    assert result["trace"][1]["value"] == "alpha"
    assert result["live"] == 1
    assert result["stale_detected"] is False


def test_handle_from_before_removal_is_stale_after_the_slot_is_reused() -> None:
    # This is the property the epoch exists for: the slot is alive again, but
    # the old key must not resolve to the new value.
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": "old"},
            {"op": "remove", "key": _key(0, 0)},
            {"op": "insert", "value": "new"},
            {"op": "get", "key": _key(0, 0)},
        ]
    )
    outcomes = [entry["outcome"] for entry in result["trace"]]
    assert outcomes == ["ok", "ok", "ok", "stale"]
    assert result["stale_detected"] is True
    stale = result["trace"][-1]
    assert stale["slot_generation"] == 1
    assert stale["alive"] is True


def test_new_key_after_reuse_resolves_to_the_new_value() -> None:
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": "old"},
            {"op": "remove", "key": _key(0, 0)},
            {"op": "insert", "value": "new"},
            {"op": "get", "key": _key(0, 1)},
        ]
    )
    assert result["trace"][-1]["outcome"] == "ok"
    assert result["trace"][-1]["value"] == "new"


def test_double_remove_is_stale() -> None:
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": "x"},
            {"op": "remove", "key": _key(0, 0)},
            {"op": "remove", "key": _key(0, 0)},
        ]
    )
    assert result["trace"][-1]["outcome"] == "stale"


def test_arena_reports_full_when_every_slot_is_alive() -> None:
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": 1},
            {"op": "insert", "value": 2},
            {"op": "insert", "value": 3},
        ],
        capacity=2,
    )
    assert [entry["outcome"] for entry in result["trace"]] == ["ok", "ok", "full"]
    assert result["live"] == 2


def test_arena_burns_a_slot_whose_epoch_is_exhausted() -> None:
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": 1},
            {"op": "remove", "key": _key(0, 0)},
            {"op": "insert", "value": 2},
            {"op": "remove", "key": _key(0, 1)},
            {"op": "insert", "value": 3},
        ],
        capacity=1,
        generations=2,
    )
    assert [entry["outcome"] for entry in result["trace"]] == [
        "ok",
        "ok",
        "ok",
        "ok",
        "burned",
    ]


def test_clear_frees_every_slot_and_invalidates_handles() -> None:
    result = rsbe_arena_simulate(
        [
            {"op": "insert", "value": 1},
            {"op": "clear"},
            {"op": "get", "key": _key(0, 0)},
        ]
    )
    assert result["live"] == 0
    assert result["trace"][-1]["outcome"] == "stale"


def test_unknown_arena_op_is_api_misuse() -> None:
    assert _code(rsbe_arena_simulate, [{"op": "resize"}]) == "E003"


def test_out_of_range_key_index_is_out_of_bounds() -> None:
    assert (
        _code(rsbe_arena_simulate, [{"op": "get", "key": _key(9, 0)}], capacity=2)
        == "E001"
    )


# -- scoped exit ---------------------------------------------------------


def test_cleanup_runs_in_reverse_acquisition_order() -> None:
    plan = rsbe_scope_exit(["open_file", "begin_tx"])
    assert plan["actions"] == ("open_file", "begin_tx")
    assert plan["drop_order"] == ("begin_tx", "open_file")
    assert "early-return" in plan["runs_on"]
    assert "unwind" in plan["runs_on"]


def test_abort_only_build_does_not_claim_unwind_cleanup() -> None:
    plan = rsbe_scope_exit(["open_file"], include_unwind=False)
    assert "unwind" not in plan["runs_on"]


def test_duplicate_cleanup_names_are_api_misuse() -> None:
    assert _code(rsbe_scope_exit, ["guard", "guard"]) == "E003"


def test_empty_cleanup_list_is_api_misuse() -> None:
    assert _code(rsbe_scope_exit, []) == "E003"
