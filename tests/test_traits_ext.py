"""Coherent extension, blanket composition, and disjoint dispatch plans.

These are plan-level tests: nothing here compiles Rust or invokes Cargo. Each
case pins what the plan accepts, what it refuses, and which framework code the
refusal carries (``E001`` invalid name, ``E002`` unsatisfiable obligation,
``E003`` API misuse, ``E005`` cycle).
"""

from __future__ import annotations

import pytest

from backends.rust.traits_ext import (
    rste_compose,
    rste_fast_path,
    rste_impl_plan,
    rste_iter_adapter,
    rste_try_adapter,
)
from compiler.frontend.errors import PolyglotError


def _code(fn, *args, **kwargs) -> str:
    with pytest.raises(PolyglotError) as caught:
        fn(*args, **kwargs)
    return caught.value.code


# -- coherence: orphan rule and overlap ----------------------------------


def test_local_trait_for_foreign_type_is_allowed() -> None:
    plan = rste_impl_plan(
        [{"trait": "Show", "trait_local": True, "for_type": "f64", "type_local": False}]
    )
    assert plan["stable"] is True
    assert plan["emitted"] == ("impl Show for f64",)


def test_foreign_trait_for_local_type_is_allowed() -> None:
    plan = rste_impl_plan(
        [
            {
                "trait": "Display",
                "trait_local": False,
                "for_type": "Widget",
                "type_local": True,
            }
        ]
    )
    assert plan["concrete"][0]["for_type"] == "Widget"


def test_two_foreign_items_violate_the_orphan_rule() -> None:
    assert (
        _code(
            rste_impl_plan,
            [
                {
                    "trait": "Display",
                    "trait_local": False,
                    "for_type": "f64",
                    "type_local": False,
                }
            ],
        )
        == "E002"
    )


def test_duplicate_concrete_impl_is_a_conflict() -> None:
    impls = [
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "Widget",
            "type_local": True,
        },
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "Widget",
            "type_local": True,
        },
    ]
    assert _code(rste_impl_plan, impls) == "E002"


def test_blanket_plus_concrete_impl_is_refused_on_stable() -> None:
    impls = [
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "T",
            "type_local": True,
            "kind": "blanket",
            "bounds": ["Show"],
        },
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "Widget",
            "type_local": True,
        },
    ]
    with pytest.raises(PolyglotError) as caught:
        rste_impl_plan(impls)
    message = str(caught.value)
    assert caught.value.code == "E002"
    assert "specialisation" in message
    assert "exclusion list" in message


def test_blanket_impl_with_exclusion_keeps_the_concrete_impl_disjoint() -> None:
    # The exclusion list is the only stable-Rust way to keep both impls; the
    # plan must accept it instead of reporting an overlap.
    impls = [
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "T",
            "type_local": True,
            "kind": "blanket",
            "bounds": ["Copy"],
            "excludes": ["Widget"],
        },
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "Widget",
            "type_local": True,
        },
    ]
    plan = rste_impl_plan(impls)
    assert plan["stable"] is True
    assert plan["requires_specialization"] is False
    assert plan["blanket"][0]["excludes"] == ("Widget",)


def test_concrete_impl_with_bounds_is_api_misuse() -> None:
    impls = [
        {
            "trait": "Show",
            "trait_local": True,
            "for_type": "Widget",
            "type_local": True,
            "bounds": ["Copy"],
        }
    ]
    assert _code(rste_impl_plan, impls) == "E003"


def test_trait_name_must_be_a_valid_path() -> None:
    assert (
        _code(
            rste_impl_plan,
            [
                {
                    "trait": "Show<",
                    "trait_local": True,
                    "for_type": "Widget",
                    "type_local": True,
                }
            ],
        )
        == "E001"
    )


def test_empty_impl_set_is_api_misuse() -> None:
    assert _code(rste_impl_plan, []) == "E003"


# -- blanket composition -------------------------------------------------


def test_composition_chain_is_ordered_and_transitive() -> None:
    plan = rste_compose(
        [
            {"name": "num", "requires": [], "provides": ["Num"]},
            {"name": "fmt", "requires": ["Num"], "provides": ["Fmt"]},
            {"name": "print", "requires": ["Fmt"], "provides": ["Printable"]},
        ],
        target="Printable",
    )
    assert plan["path"] == ("num", "fmt", "print")
    assert "Num" in plan["bounds"] and "Fmt" in plan["bounds"]


def test_composition_can_start_from_an_available_bound() -> None:
    plan = rste_compose(
        [{"name": "fmt", "requires": ["Num"], "provides": ["Fmt"]}],
        available=["Num"],
        target="Fmt",
    )
    assert plan["path"] == ("fmt",)


def test_unreachable_target_reports_the_missing_bounds() -> None:
    with pytest.raises(PolyglotError) as caught:
        rste_compose(
            [{"name": "fmt", "requires": ["Num"], "provides": ["Fmt"]}], target="Fmt"
        )
    assert caught.value.code == "E002"
    assert "Num" in str(caught.value)


def test_adapter_cycle_is_a_dependency_error() -> None:
    adapters = [
        {"name": "a", "requires": ["B"], "provides": ["A"]},
        {"name": "b", "requires": ["A"], "provides": ["B"]},
    ]
    assert _code(rste_compose, adapters, target="A") == "E005"


def test_duplicate_adapter_names_are_api_misuse() -> None:
    adapters = [
        {"name": "a", "requires": [], "provides": ["A"]},
        {"name": "a", "requires": [], "provides": ["B"]},
    ]
    assert _code(rste_compose, adapters, target="A") == "E003"


def test_adapter_that_provides_nothing_is_refused() -> None:
    assert (
        _code(rste_compose, [{"name": "a", "requires": [], "provides": []}], target="A")
        == "E002"
    )


# -- fast-path dispatch --------------------------------------------------


def test_disjoint_arms_are_ordered_fastest_first() -> None:
    plan = rste_fast_path(
        [
            {"category": "primitive", "handler": "add_i64", "covers": ["i64"]},
            {"category": "concrete", "handler": "add_widget", "covers": ["Widget"]},
        ],
        fallback="add_generic",
    )
    assert plan["arms"][-1] == "_ => add_generic"
    assert plan["order"] == ("add_i64", "add_widget", "add_generic")
    assert plan["stable"] is True


def test_overlapping_arms_are_refused_and_the_shared_type_is_named() -> None:
    cases = [
        {"category": "primitive", "handler": "fast", "covers": ["i64"]},
        {"category": "concrete", "handler": "slow", "covers": ["i64"]},
    ]
    with pytest.raises(PolyglotError) as caught:
        rste_fast_path(cases, fallback="generic")
    assert caught.value.code == "E002"
    assert "i64" in str(caught.value)


def test_generic_copy_arm_requires_nightly_specialization() -> None:
    plan = rste_fast_path(
        [{"category": "copy", "handler": "copy_fast", "covers": ["T"]}],
        fallback="generic",
    )
    assert plan["requires_specialization"] is True
    assert plan["stable"] is False


def test_scalar_copy_arm_does_not_require_specialization() -> None:
    plan = rste_fast_path(
        [{"category": "copy", "handler": "scalar", "covers": ["i64"]}],
        fallback="generic",
    )
    assert plan["requires_specialization"] is False


def test_exhaustive_dispatch_without_a_fallback_is_refused() -> None:
    cases = [{"category": "concrete", "handler": "only", "covers": ["Widget"]}]
    assert _code(rste_fast_path, cases, exhaustive=True) == "E002"


def test_duplicate_handler_names_are_api_misuse() -> None:
    cases = [
        {"category": "concrete", "handler": "h", "covers": ["A"]},
        {"category": "concrete", "handler": "h", "covers": ["B"]},
    ]
    assert _code(rste_fast_path, cases) == "E003"


def test_case_covering_nothing_is_refused() -> None:
    assert (
        _code(rste_fast_path, [{"category": "concrete", "handler": "h", "covers": []}])
        == "E002"
    )


# -- iterator and GAT shapes ---------------------------------------------


def test_owned_iterator_needs_no_lifetime() -> None:
    plan = rste_iter_adapter("i64")
    assert plan["item"] == "i64"
    assert plan["lifetime"] is None
    assert plan["gat"] is False
    assert plan["emitted"][0] == "impl Iterator for Adapter {"


def test_borrowed_iterator_gets_a_lifetime_and_a_gat_shape() -> None:
    plan = rste_iter_adapter("i64", borrowed=True, name="Windows")
    assert plan["item"] == "&'a i64"
    assert plan["lifetime"] == "'a"
    assert plan["gat"] is True
    assert plan["emitted"][0] == "impl<'a> Iterator for Windows<'a> {"


def test_borrowed_fallible_iterator_nests_the_reference_in_the_result() -> None:
    plan = rste_iter_adapter(
        "i64", borrowed=True, fallible=True, error_type="PillarError"
    )
    assert plan["item"] == "Result<&'a i64, PillarError>"


def test_fallible_iterator_without_an_error_type_is_api_misuse() -> None:
    assert _code(rste_iter_adapter, "i64", fallible=True) == "E003"


def test_try_adapter_short_circuits_on_the_first_error() -> None:
    plan = rste_try_adapter("String", "PillarError")
    assert plan["item"] == "Result<String, PillarError>"
    assert plan["residual"] == "PillarError"
    assert plan["short_circuit"] == "first-error"


def test_try_adapter_rejects_an_invalid_identifier() -> None:
    assert _code(rste_try_adapter, "Result<T>", "E") == "E001"
