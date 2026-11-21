"""Focused tests for Frame C hard compiler/runtime edge cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from compiler.ir.framework import ir_lower_ast_to_hir, ir_lower_hir_to_mir
from compiler.ir.hard import (
    ir_schedule_raii_cleanup,
    ir_track_branch_aware_moves,
    ir_verify_dominance_and_loop_headers,
)
from compiler.types.hard import (
    types_check_coherence_and_projection,
    types_enforce_higher_rank_bounds,
    types_expand_mutually_recursive_generics,
)
from lexer import lex
from parser import parse
from runtime.diagnostics import PillarError
from runtime.hotpatch.hard import (
    hp_rcu_reader_writer_invariants,
    hp_reactor_migrate_without_loss,
    hp_revalidate_raw_address,
    hp_rollback_verified,
)
from tools.build_ci.hard import (
    ci_persist_divergence,
    ci_scoped_backend_environment,
    ci_seed_reproducible_matrix,
)

SPAN = (2, 8)


def _mir(source: str) -> Any:
    return ir_lower_hir_to_mir(ir_lower_ast_to_hir(parse(lex(source))))


def test_mutually_recursive_generics_with_alias_unification() -> None:
    report = types_expand_mutually_recursive_generics(
        {
            "Tree": ["Vec", "Leaf"],
            "Node": ["Vec"],
            "Vec": ["Node"],
            "Leaf": [],
        },
        root="Tree",
        aliases={"Node": "Tree"},
        budget=64,
        span=SPAN,
    )
    assert report.canonical_root == "Tree"
    assert report.placeholders == ("Leaf", "Tree", "Vec")
    assert any(cycle[0] == cycle[-1] for cycle in report.cycles)
    assert not report.exhausted


def test_recursive_generic_budget_is_structured() -> None:
    with pytest.raises(PillarError) as raised:
        types_expand_mutually_recursive_generics(
            {"Tree": ["Tree"]}, root="Tree", budget=2, span=SPAN
        )
    assert raised.value.code == "E002"
    assert raised.value.span == SPAN


def test_undefined_generic_reference_is_e005() -> None:
    with pytest.raises(PillarError) as raised:
        types_expand_mutually_recursive_generics({"Tree": ["Missing"]}, root="Tree")
    assert raised.value.code == "E005"


def test_trait_coherence_projection_and_higher_rank_bounds() -> None:
    report = types_check_coherence_and_projection(
        [
            {
                "name": "impl_i64",
                "trait": "Display",
                "for": "i64",
                "assoc": {"Output": "i64"},
                "bounds": ("for<'a> Fn(&'a i64)",),
                "lifetimes": ("'a",),
            },
            {
                "name": "impl_maybe",
                "trait": "Display",
                "for": "Option<i64>",
                "assoc": {"Output": "bool"},
            },
        ],
        trait="Display",
        associated={"Output": "i64"},
    )
    assert report.dispatch == {"i64": "impl_i64", "Option<i64>": "impl_maybe"}
    assert report.projections["impl_i64"] == {"Output": "i64"}
    assert report.higher_rank == ("impl_i64:a",)

    with pytest.raises(PillarError) as raised:
        types_check_coherence_and_projection(
            [
                {"name": "a", "trait": "Display", "for": "i64", "assoc": {}},
                {"name": "b", "trait": "Display", "for": "i64", "assoc": {}},
            ],
            trait="Display",
        )
    assert raised.value.code == "E001"

    assert types_enforce_higher_rank_bounds(["for<'a> Fn(&'a T)"], supplied=["'a"])[
        "required"
    ] == ["a"]
    with pytest.raises(PillarError) as raised:
        types_enforce_higher_rank_bounds(
            ["for<'a> Fn(&'a T)"], supplied=["'b"], span=SPAN
        )
    assert raised.value.code == "E002"


def test_dominance_and_loop_headers() -> None:
    module = _mir(
        "fn sum(n: i64) -> i64 { let mut i: i64 = 0; let mut t: i64 = 0; "
        "while i < n { t += i; i += 1; } t }"
    )
    report = ir_verify_dominance_and_loop_headers(module, span=SPAN)
    assert report.valid
    assert report.functions["sum"]["loop_headers"] == ["while_head_2"]
    assert report.functions["sum"]["back_edges"] == [["while_body_3", "while_head_2"]]


def test_branch_aware_moves_and_raii_cleanup() -> None:
    module = _mir("fn f(n: i64) -> i64 { let mut x: i64 = n; if n > 0 { x += 1; } x }")
    moves = ir_track_branch_aware_moves(module, span=SPAN)
    assert moves.as_dict()["ok"]
    cleanup = ir_schedule_raii_cleanup(module, span=SPAN)
    assert cleanup.zero_leak
    with pytest.raises(PillarError) as raised:
        ir_schedule_raii_cleanup(module, early_returns=["missing_block"], span=SPAN)
    assert raised.value.code == "E002"


def test_rcu_reclaim_requires_grace_period() -> None:
    report = hp_rcu_reader_writer_invariants(
        [
            {"op": "enter", "reader": "r1"},
            {"op": "retire", "item": "old-ast"},
            {"op": "advance"},
            {"op": "exit", "reader": "r1"},
            {"op": "advance"},
            {"op": "reclaim"},
        ],
        span=SPAN,
    )
    assert report.reclaimed == ("old-ast",)
    assert report.deferred == ()
    assert report.violations == ()

    with pytest.raises(PillarError) as raised:
        hp_rcu_reader_writer_invariants(
            [
                {"op": "enter", "reader": "r1"},
                {"op": "retire", "item": "old-ast"},
                {"op": "advance"},
                {"op": "reclaim"},
            ],
            span=SPAN,
        )
    assert raised.value.code == "E008"


def test_reactor_migration_preserves_events() -> None:
    report = hp_reactor_migrate_without_loss(
        ["fd-a", "fd-b"],
        ["fd-a", "fd-b", "fd-c"],
        events=[("fd-a", "readable"), ("fd-b", "writable")],
        span=SPAN,
    )
    assert report.migrated == ("fd-a", "fd-b")
    assert report.added == ("fd-c",)
    assert report.preserved[-1][0] == "fd-b"

    with pytest.raises(PillarError) as raised:
        hp_reactor_migrate_without_loss(
            ["fd-a", "fd-b"], ["fd-a"], events=[], span=SPAN
        )
    assert raised.value.code == "E008"


def test_raw_address_toctou_revalidation() -> None:
    snapshot = {"address": 4096, "value": 7, "version": 3}
    digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert (
        hp_revalidate_raw_address(
            [snapshot, dict(snapshot)], expected_digest=digest, span=SPAN
        )["checked"]
        == 2
    )

    with pytest.raises(PillarError) as raised:
        hp_revalidate_raw_address(
            [snapshot, {**snapshot, "value": 8}], expected_digest=digest, span=SPAN
        )
    assert raised.value.code == "E008"
    assert "TOCTOU" in str(raised.value)


def test_rollback_verified_restores_state() -> None:
    state: list[str] = []

    def step(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "apply": lambda: state.append(name),
            "undo": lambda: state.pop(),
        }

    assert (
        hp_rollback_verified([step("a"), step("b")], verifier=lambda _payload: True)[
            "applied"
        ]
        == 2
    )
    assert state == ["a", "b"]
    state.clear()

    with pytest.raises(PillarError) as raised:
        hp_rollback_verified(
            [step("a"), step("b")], verifier=lambda _payload: False, span=SPAN
        )
    assert raised.value.code == "E008"
    assert state == []


def test_isolated_build_environment_is_scoped() -> None:
    first = ci_scoped_backend_environment(
        Path("/tmp/polyglot-hard-tests"),
        backend="native",
        seed=77,
        case_id="recursive-generics",
    )
    second = ci_scoped_backend_environment(
        Path("/tmp/polyglot-hard-tests"),
        backend="native",
        seed=77,
        case_id="recursive-generics",
    )
    assert first == second
    assert first["env"]["CARGO_TARGET_DIR"] == str(first["target_dir"])
    assert "/tmp/polyglot-hard-tests" in str(first["cache"])
    assert (
        first["scope"]
        != ci_scoped_backend_environment(
            Path("/tmp/polyglot-hard-tests"),
            backend="wasm",
            seed=77,
            case_id="recursive-generics",
        )["scope"]
    )


def test_seed_reproducible_matrix_and_divergence_artifacts(tmp_path: Path) -> None:
    cases = [{"id": "traits", "source": "fn f() -> i64 { 1 }"}]
    matrix = ci_seed_reproducible_matrix(
        cases, seed=1234, backends=["python", "wasm", "native"]
    )
    assert [entry["backend"] for entry in matrix] == ["native", "python", "wasm"]
    assert all(entry["seed"] == 1234 for entry in matrix)
    assert matrix == ci_seed_reproducible_matrix(
        cases, seed=1234, backends=["python", "wasm", "native"]
    )

    artifacts = tmp_path / "artifacts"
    assert (
        ci_persist_divergence(
            {"passed": True, "mismatches": []},
            directory=artifacts,
            case=cases[0],
            seed=1234,
        )
        is None
    )
    assert not artifacts.exists()
    written = ci_persist_divergence(
        {"passed": False, "mismatches": [{"call": "f"}]},
        directory=artifacts,
        case=cases[0],
        seed=1234,
    )
    assert written is not None and written.exists()
    assert (artifacts / "1234-traits.poly").read_text() == cases[0]["source"]
    payload = json.loads(written.read_text())
    assert payload["seed"] == 1234
    assert payload["mismatches"] == [{"call": "f"}]
