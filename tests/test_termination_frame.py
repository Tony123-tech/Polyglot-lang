"""Focused tests for Frame I termination and resource-bound proofs."""

from __future__ import annotations

import pytest

from compiler.termination import (
    FuelContract,
    VariantSpec,
    term_check_structural_and_lexicographic_variants,
    term_enforce_resource_fuel_contracts,
    term_infer_loop_bound_certificates,
)
from runtime.diagnostics import PillarError

SPAN = (3, 9)

TREE_SOURCE = """
fn size(t: Tree) -> i64 {
    if is_empty(t) { return 0; }
    return 1 + size(t.left);
}
"""

ACK_SOURCE = """
fn ack(m: i64, n: i64) -> i64 {
    if m == 0 { return n + 1; }
    if n == 0 { return ack(m - 1, 1); }
    return ack(m - 1, ack(m, n - 1));
}
"""

LOOP_SOURCE = """
fn main(n: i64) -> i64 {
    let total = 0;
    for i in 0..n { total += i; }
    while n > 0 { n -= 1; }
    return total;
}
"""


def test_structural_variant_proves_self_recursion() -> None:
    report = term_check_structural_and_lexicographic_variants(
        TREE_SOURCE,
        specs={
            "size": VariantSpec(
                function="size", kind="structural", parameters=("t",), span=SPAN
            )
        },
        span=SPAN,
    )

    assert report.proved
    assert len(report.calls) == 1
    assert report.calls[0].arguments == ("t.left",)
    assert report.calls[0].decreases == ("t",)
    assert report.diagnostics == ()
    assert report.digest() == report.as_dict()["digest"]


def test_lexicographic_variant_accepts_nested_and_outer_calls() -> None:
    report = term_check_structural_and_lexicographic_variants(
        ACK_SOURCE,
        specs={
            "ack": {
                "kind": "lexicographic",
                "parameters": ("m", "n"),
            }
        },
        span=SPAN,
    )

    assert report.proved
    assert [call.decreases for call in report.calls] == [("m",), ("m",), ("n",)]
    assert all(diag.code not in {"E005", "E008"} for diag in report.diagnostics)


def test_non_decreasing_recursive_call_is_e005() -> None:
    source = "fn spin(x: i64) -> i64 { return spin(x); }"
    report = term_check_structural_and_lexicographic_variants(
        source,
        specs={"spin": {"kind": "numeric", "parameters": ("x",)}},
        span=SPAN,
    )

    assert not report.proved
    assert [diag.code for diag in report.diagnostics] == ["E005"]
    assert report.diagnostics[0].span is not None


def test_unsupported_variant_arity_is_e008() -> None:
    report = term_check_structural_and_lexicographic_variants(
        ACK_SOURCE,
        specs={"ack": {"kind": "lexicographic", "parameters": ("m", "n", "extra")}},
        span=SPAN,
    )

    assert not report.proved
    assert any(diag.code == "E003" for diag in report.diagnostics)
    assert any(diag.code == "E008" for diag in report.diagnostics)


def test_unknown_variant_kind_and_missing_function_are_structured() -> None:
    with pytest.raises(PillarError) as raised:
        VariantSpec(function="f", kind="magic", parameters=("x",), span=SPAN)
    assert raised.value.code == "E003"

    with pytest.raises(PillarError) as raised:
        term_check_structural_and_lexicographic_variants(
            TREE_SOURCE,
            specs={"missing": {"kind": "numeric", "parameters": ("x",)}},
            span=SPAN,
        )
    assert raised.value.code == "E006"


def test_loop_certificates_cover_counted_and_counter_loops() -> None:
    report = term_infer_loop_bound_certificates(LOOP_SOURCE, span=SPAN)

    assert report.proved
    assert [cert.kind for cert in report.certificates] == ["counted", "counter"]
    assert report.certificates[0].iterations == "max(0, n - 0)"
    assert report.certificates[1].iterations == "max(0, n - 0) / 1"
    assert all(cert.allocations == "0" for cert in report.certificates)


def test_allocation_certificate_counts_sites_per_iteration() -> None:
    source = """
fn main(n: i64) -> i64 {
    for i in 0..4 { let a = alloc(i); let b = alloc(a); }
    return 0;
}
"""
    report = term_infer_loop_bound_certificates(source)

    assert report.proved
    cert = report.certificates[0]
    assert cert.iterations == "4"
    assert cert.allocations == "2 * 4"


def test_unbounded_loops_are_explicit_e008() -> None:
    source = """
fn main(n: i64) -> i64 {
    loop { n += 1; }
    while n > 0 { n += 1; }
}
"""
    report = term_infer_loop_bound_certificates(source, span=SPAN)

    assert not report.proved
    assert all(cert.kind == "UnboundedOrUnknown" for cert in report.certificates)
    assert [diag.code for diag in report.diagnostics] == ["E008", "E008"]
    assert any(
        "UnboundedOrUnknown is an explicit refusal" in caveat
        for caveat in report.as_dict()["caveats"]
    )


def test_fuel_contract_reports_exhaustion_cleanly() -> None:
    contract = FuelContract(
        name="recursion", steps=8, memory_bytes=1024, stack_depth=4, span=SPAN
    )

    allowed = term_enforce_resource_fuel_contracts(
        contract, steps=5, memory_bytes=512, stack_depth=3, span=SPAN
    )
    assert allowed.allowed
    assert allowed.remaining == {"steps": 3, "memory_bytes": 512, "stack_depth": 1}

    partial = term_enforce_resource_fuel_contracts(
        contract, steps=20, stack_depth=9, strict=False, span=SPAN
    )
    assert not partial.allowed
    assert partial.exhausted == ("stack_depth", "steps")
    assert partial.remaining["steps"] == 0

    with pytest.raises(PillarError) as raised:
        term_enforce_resource_fuel_contracts(contract, steps=20, span=SPAN)
    assert raised.value.code == "E008"
    assert raised.value.span == SPAN
    assert "steps" in raised.value.diagnostic.message


def test_fuel_contract_validates_limits_and_usage() -> None:
    with pytest.raises(PillarError) as raised:
        FuelContract(name="bad", steps=0, memory_bytes=1, stack_depth=1, span=SPAN)
    assert raised.value.code == "E003"

    contract = FuelContract(name="ok", steps=4, memory_bytes=4, stack_depth=4)
    with pytest.raises(PillarError) as raised:
        term_enforce_resource_fuel_contracts(contract, steps=-1, span=SPAN)
    assert raised.value.code == "E003"
