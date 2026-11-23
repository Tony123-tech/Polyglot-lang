"""Error-context chains, cancellation-safe plans, and PyO3 boundary mappings.

Every case is a plan: the retry schedule is a list of waits, the deadline
allocator sums to the budget exactly, and the GIL plan refuses a Python-object
operation inside a released region. The codes pinned here are ``E001``
invalid name or budget, ``E002`` unsatisfiable obligation, ``E003`` API
misuse, plus the aggregated headline (the lowest collected code).
"""

from __future__ import annotations

import pytest

from backends.rust.async_interop import (
    rsai_aggregate,
    rsai_cancel_scope,
    rsai_convert_enum,
    rsai_convert_result,
    rsai_deadline_plan,
    rsai_error_chain,
    rsai_gil_plan,
    rsai_retry_plan,
)
from compiler.frontend.errors import PolyglotError


def _code(fn, *args, **kwargs) -> str:
    with pytest.raises(PolyglotError) as caught:
        fn(*args, **kwargs)
    return caught.value.code


# -- error context -------------------------------------------------------


def test_chain_keeps_every_layer_from_outermost_to_root() -> None:
    plan = rsai_error_chain(
        [
            {"operation": "write_report", "code": "E002", "message": "report is empty"},
            {"operation": "read_text", "code": "E006", "message": "missing input"},
        ]
    )
    assert plan["code"] == "E002"
    assert plan["root_code"] == "E006"
    assert plan["depth"] == 2
    assert "caused by [E006] read_text: missing input" in plan["report"]


def test_single_step_chain_is_allowed() -> None:
    plan = rsai_error_chain(
        [{"operation": "parse", "code": "E008", "message": "bad literal"}]
    )
    assert plan["depth"] == 1
    assert "caused by" not in plan["report"]


def test_chain_rejects_an_unknown_code() -> None:
    assert (
        _code(
            rsai_error_chain,
            [{"operation": "parse", "code": "E999", "message": "bad"}],
        )
        == "E003"
    )


def test_chain_rejects_an_empty_step_list() -> None:
    assert _code(rsai_error_chain, []) == "E003"


def test_chain_rejects_an_invalid_operation_name() -> None:
    assert (
        _code(
            rsai_error_chain,
            [{"operation": "read text", "code": "E002", "message": "bad"}],
        )
        == "E001"
    )


# -- aggregation ---------------------------------------------------------


def test_aggregate_reports_every_error_not_just_the_first() -> None:
    plan = rsai_aggregate(
        [
            {"code": "E002", "message": "empty report", "path": "report"},
            {"code": "E001", "message": "bad width", "path": "width"},
            {"code": "E002", "message": "missing field", "path": "header"},
        ]
    )
    assert plan["count"] == 3
    assert plan["code"] == "E001"
    assert plan["report"].splitlines()[0].startswith("error[E001] width")


def test_aggregate_collapses_exact_duplicates() -> None:
    error = {"code": "E002", "message": "same", "path": "x"}
    plan = rsai_aggregate([error, dict(error)])
    assert plan["count"] == 1


def test_aggregate_limit_truncates_the_report_but_not_the_count() -> None:
    plan = rsai_aggregate(
        [{"code": "E002", "message": f"m{i}", "path": f"p{i}"} for i in range(5)],
        limit=2,
    )
    assert plan["count"] == 5
    assert plan["shown"] == 2
    assert plan["truncated"] is True
    assert "3 more" in plan["report"]


def test_aggregate_of_nothing_is_not_a_failure() -> None:
    plan = rsai_aggregate([])
    assert plan["code"] is None
    assert plan["count"] == 0


def test_aggregate_rejects_an_unknown_code() -> None:
    assert _code(rsai_aggregate, [{"code": "E010", "message": "x"}]) == "E003"


# -- retry and deadlines -------------------------------------------------


def test_retry_schedule_is_exponential_with_a_cap() -> None:
    plan = rsai_retry_plan(4, base_ms=100, cap_ms=300)
    assert plan["attempts"] == 4
    assert plan["waits_ms"] == (100, 200, 300)
    assert plan["total_ms"] == 600


def test_retry_schedule_has_one_wait_fewer_than_attempts() -> None:
    plan = rsai_retry_plan(1)
    assert plan["waits_ms"] == ()
    assert plan["checkpoints"] == ("before-attempt-1",)


def test_retry_checkpoints_cover_attempts_and_waits() -> None:
    plan = rsai_retry_plan(3)
    assert plan["checkpoints"] == (
        "before-attempt-1",
        "before-attempt-2",
        "before-attempt-3",
        "during-wait-1",
        "during-wait-2",
    )
    assert plan["idempotent_required"] is True


def test_retry_cap_below_base_is_invalid() -> None:
    assert _code(rsai_retry_plan, 3, base_ms=100, cap_ms=50) == "E001"


def test_retry_attempts_must_be_at_least_one() -> None:
    assert _code(rsai_retry_plan, 0) == "E001"


def test_deadline_allocations_sum_to_the_budget_exactly() -> None:
    plan = rsai_deadline_plan(
        100, [{"name": "a", "min_ms": 10}, {"name": "b", "min_ms": 20, "share": 2}]
    )
    total = sum(entry["ms"] for entry in plan["allocations"])
    assert total == 100
    assert plan["slack_ms"] == 70
    assert plan["allocations"][1]["ms"] > plan["allocations"][0]["ms"]


def test_remainder_goes_to_the_earliest_step() -> None:
    plan = rsai_deadline_plan(100, [{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert [entry["ms"] for entry in plan["allocations"]] == [34, 33, 33]


def test_infeasible_budget_is_refused_instead_of_shaved() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsai_deadline_plan(10, [{"name": "a", "min_ms": 8}, {"name": "b", "min_ms": 5}])
    assert caught.value.code == "E002"
    assert "13 ms" in str(caught.value)


def test_budget_must_be_positive() -> None:
    assert _code(rsai_deadline_plan, 0, [{"name": "a"}]) == "E001"


# -- cancel scope --------------------------------------------------------


def test_cancel_scope_lists_commits_and_reverse_rollbacks() -> None:
    plan = rsai_cancel_scope(
        [
            {"name": "fetch", "commit": "apply_fetch"},
            {"name": "render", "commit": "apply_render"},
        ]
    )
    assert plan["commit_order"] == ("apply_fetch", "apply_render")
    assert plan["rollback_order"] == ("apply_render", "apply_fetch")
    assert "after apply_fetch before render" in plan["safe_cancel_windows"]


def test_step_without_a_commit_point_is_refused() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsai_cancel_scope([{"name": "mutate_in_place"}])
    assert caught.value.code == "E002"
    assert "loses work" in str(caught.value)


def test_duplicate_step_names_are_api_misuse() -> None:
    steps = [
        {"name": "a", "commit": "c1"},
        {"name": "a", "commit": "c2"},
    ]
    assert _code(rsai_cancel_scope, steps) == "E003"


# -- GIL discipline ------------------------------------------------------


def test_gil_plan_separates_released_work_from_python_object_work() -> None:
    plan = rsai_gil_plan(["compute"], ["dict_set"])
    assert plan["released"] == ("compute",)
    assert plan["requires_reacquire"] == ("dict_set",)


def test_python_object_operation_inside_a_released_region_is_refused() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsai_gil_plan(["dict_view"], [])
    assert caught.value.code == "E002"
    assert "dict_view" in str(caught.value)


def test_operation_declared_in_both_regions_is_refused() -> None:
    assert _code(rsai_gil_plan, ["compute"], ["compute"]) == "E002"


# -- PyO3 conversion -----------------------------------------------------


def test_known_result_pair_maps_to_a_value_or_an_exception() -> None:
    plan = rsai_convert_result("String", "PillarError")
    assert plan["ok_py"] == "str"
    assert plan["err_py"] == "raise PillarError"


def test_unknown_result_pair_lists_the_supported_pairs() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsai_convert_result("HashMap<String, i64>", "PillarError")
    assert caught.value.code == "E003"
    assert "Result<String, PillarError>" in str(caught.value)


def test_enum_plan_separates_unit_and_payload_variants() -> None:
    plan = rsai_convert_enum(
        "Kind", [{"name": "Empty"}, {"name": "Count", "payload": "i64"}]
    )
    assert plan["unit_variants"] == ("Empty",)
    assert plan["payload_variants"] == ("Count",)
    assert plan["emitted"][1].startswith("Kind::Count(v)")


def test_enum_variant_carrying_a_reference_is_refused() -> None:
    with pytest.raises(PolyglotError) as caught:
        rsai_convert_enum("Kind", [{"name": "Borrowed", "payload": "&String"}])
    assert caught.value.code == "E002"
    assert "outlive" in str(caught.value)


def test_enum_variant_with_an_unknown_payload_type_is_api_misuse() -> None:
    assert (
        _code(rsai_convert_enum, "Kind", [{"name": "X", "payload": "Widget"}]) == "E003"
    )


def test_enum_duplicate_variant_names_are_api_misuse() -> None:
    variants = [{"name": "A"}, {"name": "A"}]
    assert _code(rsai_convert_enum, "Kind", variants) == "E003"
