"""Coverage tests for batch 3 of the 1,000-function framework rollout.

The tests resolve every metadata row, invoke every function with a
representative model-IR input, and exercise the diagnostic and verification
contracts.  All programs are parsed and lowered with the real frontend/IR
pipeline; no user code is executed and no backend is invoked.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from compiler.frontend.errors import PolyglotError
from compiler.ir import hir, mir, passes, verify
from compiler.ir.framework import (
    BasicBlock,
    Cfg,
    HirExpr,
    HirFunction,
    HirModule,
    HirType,
    MirFunction,
    MirInstr,
    MirModule,
    MirOperand,
    MirTerminator,
    ir_lower_ast_to_hir,
    ir_lower_hir_to_mir,
)
from compiler.ir.rollout import FRAMEWORK_ROWS

_I64 = HirType("i64")
_F64 = HirType("f64")
_BOOL = HirType("bool")
_UNIT = HirType("()")

_HIR_SOURCE = """
fn double(x: i64) -> i64 { return x * 2; }
fn main(n: i64) -> i64 {
  let total = 2 + 3 * 4;
  let flag = true && false;
  let choice = if total > 5 { double(total) } else { 0 };
  while total > 0 { total -= 1; }
  match choice { 0 => 1, 1 => 2, _ => 3 }
}
"""
_SIMPLE_SOURCE = "fn f(n: i64) -> i64 { if n > 0 { return 1; } return 0; }"

_HIR = ir_lower_ast_to_hir(_HIR_SOURCE)
_MIR = ir_lower_hir_to_mir(_HIR)
_SIMPLE_HIR = ir_lower_ast_to_hir(_SIMPLE_SOURCE)
_SIMPLE_MIR = ir_lower_hir_to_mir(_SIMPLE_HIR)
_SERIALIZED = mir.mir_serialize(_MIR)
_LOOP_CFG = _MIR.functions["main"].cfg
_ACYCLIC_CFG = _SIMPLE_MIR.functions["f"].cfg
_ENTRY = _ACYCLIC_CFG.entry

_MERGE_CFG = Cfg(
    "entry",
    {
        "entry": BasicBlock("entry", [], MirTerminator("jump", ("middle",))),
        "middle": BasicBlock("middle", [], MirTerminator("jump", ("exit",))),
        "exit": BasicBlock("exit", [], MirTerminator("return")),
    },
)


def _hir_nodes(expr: HirExpr) -> list[HirExpr]:
    """Collect a HIR tree, including nested statement and branch values."""
    nodes = [expr]
    for arg in expr.args:
        nodes.extend(_hir_nodes(arg))
    for branch in expr.branches:
        if branch.condition is not None:
            nodes.extend(_hir_nodes(branch.condition))
        nodes.extend(_hir_nodes(branch.body))
    for stmt in expr.meta.get("stmts", ()):
        if getattr(stmt, "value", None) is not None:
            nodes.extend(_hir_nodes(stmt.value))
    return nodes


_MATCH_EXPR = next(
    node for node in _hir_nodes(_HIR.functions["main"].body) if node.kind == "match"
)


def _call(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Package one representative invocation for the metadata smoke test."""
    return args, kwargs


_CALL_ARGS: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
    # 0201-0210: legacy FRAME 3 surface.
    "ir_lower_ast_to_hir": _call(_SIMPLE_SOURCE),
    "ir_lower_hir_to_mir": _call(_SIMPLE_HIR),
    "ir_fold_constant_expressions": _call(_HIR),
    "ir_eliminate_dead_blocks": _call(_SIMPLE_MIR),
    "ir_validate_control_flow_graph": _call(_SIMPLE_MIR),
    "ir_check_borrow_and_move_state": _call(_SIMPLE_MIR),
    "ir_insert_raii_drop_points": _call(_SIMPLE_MIR),
    "ir_inline_small_pure_functions": _call(_HIR),
    "ir_hoist_loop_invariant_checks": _call(_MIR),
    "ir_verify_dual_backend_invariants": _call(_SIMPLE_SOURCE),
    # 0211-0235: HIR helpers.
    "hir_alpha_rename": _call(_HIR),
    "hir_bind_locals": _call(_HIR.functions["main"].body),
    "hir_call_graph": _call(_HIR),
    "hir_collect_symbols": _call(_HIR),
    "hir_const_env": _call(_HIR.functions["main"].body),
    "hir_desugar_match": _call(_MATCH_EXPR),
    "hir_drop_scopes": _call(_HIR),
    "hir_expr_depth": _call(_HIR.functions["main"].body),
    "hir_function_summary": _call(_HIR.functions["main"]),
    "hir_lower_block": _call("let x = 1; return x + 2;"),
    "hir_lower_loops": _call(_HIR.functions["main"].body),
    "hir_lower_short_circuit": _call(_HIR.functions["main"].body),
    "hir_literal_table": _call(_HIR),
    "hir_monomorph_plan": _call(_HIR),
    "hir_module_fingerprint": _call(_HIR),
    "hir_normalize_spans": _call(_HIR.functions["main"].body),
    "hir_pattern_bindings": _call(_MATCH_EXPR),
    "hir_ref_types": _call(_HIR),
    "hir_rewrite_ops": _call(_HIR, {"+": "add"}),
    "hir_resolve_calls": _call(_HIR),
    "hir_stmt_kinds": _call(_HIR),
    "hir_substitute": _call(
        _HIR.functions["main"].body,
        {"total": HirExpr("literal", _I64, value=7)},
    ),
    "hir_type_expr": _call(_HIR.functions["main"].body),
    "hir_unify_types": _call(_I64, _F64),
    "hir_validate_module": _call(_HIR),
    # 0236-0260: MIR/CFG helpers.
    "mir_block_index": _call(_MIR),
    "mir_cfg_fingerprint": _call(_MIR),
    "mir_cfg_summary": _call(_MIR),
    "mir_clone_cfg": _call(_ACYCLIC_CFG),
    "mir_detect_cycles": _call(_LOOP_CFG),
    "mir_deserialize": _call(_SERIALIZED),
    "mir_dominators": _call(_ACYCLIC_CFG),
    "mir_empty_blocks": _call(_MIR),
    "mir_find_instr": _call(_MIR, op="const"),
    "mir_instr_histogram": _call(_MIR),
    "mir_liveness": _call(_ACYCLIC_CFG),
    "mir_loop_headers": _call(_LOOP_CFG),
    "mir_merge_blocks": _call(_MERGE_CFG, "middle"),
    "mir_operand_kinds": _call(_MIR),
    "mir_operand_types": _call(_MIR),
    "mir_phi_sources": _call(_MIR),
    "mir_predecessors": _call(_ACYCLIC_CFG),
    "mir_reads_writes": _call(_MIR),
    "mir_reachable": _call(_ACYCLIC_CFG),
    "mir_rename_block": _call(_ACYCLIC_CFG, _ENTRY, "start"),
    "mir_serialize": _call(_MIR),
    "mir_split_block": _call(_ACYCLIC_CFG, _ENTRY, 0),
    "mir_successors": _call(_ACYCLIC_CFG, _ENTRY),
    "mir_terminator_arity": _call(_MIR),
    "mir_topological_order": _call(_ACYCLIC_CFG),
    # 0261-0285: optimizer passes.
    "pass_check_borrows": _call(_MIR),
    "pass_constant_env": _call(_HIR),
    "pass_count_effects": _call(_MIR),
    "pass_dead_branch_elimination": _call(_MIR),
    "pass_dedupe_constants": _call(_MIR),
    "pass_desugar_short_circuit_hir": _call(_HIR),
    "pass_eliminate_dead_code": _call(_MIR),
    "pass_estimate_cost": _call(_HIR),
    "pass_fixpoint": _call(
        _MIR, ("propagate_copies", "fold_mir_constants"), max_rounds=2
    ),
    "pass_fold_binary": _call("+", 2, 3),
    "pass_fold_hir_constants": _call(_HIR),
    "pass_fold_mir_constants": _call(_MIR),
    "pass_fold_unary": _call("-", 7),
    "pass_hoist_guards": _call(_MIR),
    "pass_inline_pure_calls": _call(_HIR),
    "pass_insert_drops": _call(_MIR),
    "pass_merge_jump_chains": _call(_MIR),
    "pass_propagate_copies": _call(_MIR),
    "pass_remove_empty_blocks": _call(_MIR),
    "pass_remove_unreachable": _call(_MIR),
    "pass_rename_temporaries": _call(_MIR),
    "pass_simplify_phi": _call(_MIR),
    "pass_strip_metadata": _call(_MIR, ("scope",)),
    "pass_validate_cfg": _call(_MIR),
    "pass_verify_invariants": _call(_SIMPLE_SOURCE),
    # 0286-0300: verification helpers.
    "verify_backend_type_support": _call(_HIR),
    "verify_borrow_safety": _call(_MIR),
    "verify_cfg_structure": _call(_MIR),
    "verify_constant_folds": _call(_HIR),
    "verify_desugared_control_flow": _call(_MIR),
    "verify_dual_backend_invariants": _call(_SIMPLE_SOURCE),
    "verify_function_signatures": _call(_MIR),
    "verify_hir_module": _call(_HIR),
    "verify_metadata_values": _call(_MIR),
    "verify_mir_roundtrip": _call(_MIR),
    "verify_no_unreachable_blocks": _call(_MIR),
    "verify_operand_kinds": _call(_MIR),
    "verify_pass_trace": _call(
        ("pass_propagate_copies", "pass_fold_mir_constants", "pass_dedupe_constants")
    ),
    "verify_phi_placement": _call(_MIR),
    "verify_type_unification": _call(_I64, _F64),
}

_ROW_IDS = [row[0] for row in FRAMEWORK_ROWS]


@pytest.mark.parametrize(
    "name,module_name,category,test_module", FRAMEWORK_ROWS, ids=_ROW_IDS
)
def test_every_metadata_row_resolves_to_a_documented_callable(
    name: str, module_name: str, category: str, test_module: str
) -> None:
    """Every row resolves to a callable with documentation and valid metadata."""
    function = getattr(import_module(module_name), name)
    assert callable(function)
    assert function.__doc__
    assert category in {"lowering", "passes", "verification", "hir", "mir"}
    assert test_module == "tests.test_ir_rollout"


@pytest.mark.parametrize(
    "name,module_name,category,test_module", FRAMEWORK_ROWS, ids=_ROW_IDS
)
def test_every_metadata_row_is_invoked(
    name: str, module_name: str, category: str, test_module: str
) -> None:
    """Every row can be invoked with a representative deterministic input."""
    function = getattr(import_module(module_name), name)
    args, kwargs = _CALL_ARGS[name]
    result = function(*args, **kwargs)
    assert result is not None
    if module_name != "compiler.ir.framework":
        assert isinstance(result, dict)
        assert result


def test_rollout_shape_and_category_layout() -> None:
    """The metadata is a stable, gap-free, uniquely named 100-row batch."""
    assert len(FRAMEWORK_ROWS) == 100
    assert all(
        len(row) == 4 and all(isinstance(item, str) for item in row)
        for row in FRAMEWORK_ROWS
    )
    assert len(_ROW_IDS) == len(set(_ROW_IDS))
    assert [row[2] for row in FRAMEWORK_ROWS[:2]] == ["lowering", "lowering"]
    assert [row[2] for row in FRAMEWORK_ROWS[2:9]] == ["passes"] * 7
    assert FRAMEWORK_ROWS[9][2] == "verification"
    assert {row[1] for row in FRAMEWORK_ROWS[:10]} == {"compiler.ir.framework"}
    categories = {
        "compiler.ir.hir": "hir",
        "compiler.ir.mir": "mir",
        "compiler.ir.passes": "passes",
        "compiler.ir.verify": "verification",
    }
    for module_name, category in categories.items():
        rows = [row for row in FRAMEWORK_ROWS[10:] if row[1] == module_name]
        assert len(rows) == (15 if module_name.endswith("verify") else 25)
        assert all(row[2] == category for row in rows)


@pytest.mark.parametrize(
    "module,prefix,count",
    [
        (hir, "hir_", 25),
        (mir, "mir_", 25),
        (passes, "pass_", 25),
        (verify, "verify_", 15),
    ],
)
def test_public_surfaces_are_sorted_and_complete(
    module: object, prefix: str, count: int
) -> None:
    """Each new module exports exactly its documented public functions."""
    names = list(module.__all__)
    assert len(names) == count
    assert names == sorted(names)
    assert all(name.startswith(prefix) for name in names)
    exported = {
        name
        for name in dir(module)
        if name.startswith(prefix) and callable(getattr(module, name))
    }
    assert exported == set(names)


def test_hir_helpers_lower_and_analyze_deterministically() -> None:
    """HIR helpers rewrite structure without touching their input trees."""
    source = "fn f(a: bool, b: bool) -> bool { return a && b; }"
    module = ir_lower_ast_to_hir(source)
    before = hir.hir_module_fingerprint(module)["digest"]

    lowered = hir.hir_lower_short_circuit(module.functions["f"].body)
    assert lowered["rewritten"] == 1
    assert lowered["remaining"] == 0
    assert lowered["operators"] == ("&&",)
    assert hir.hir_module_fingerprint(module)["digest"] == before

    renamed = hir.hir_alpha_rename(module, prefix="renamed")
    assert renamed["count"] >= 2
    assert hir.hir_module_fingerprint(module)["digest"] == before
    assert hir.hir_module_fingerprint(renamed["module"])["digest"] != before

    call_graph = hir.hir_call_graph(_HIR)
    assert call_graph["edges"]["main"] == ("double",)
    assert call_graph["has_cycle"] is False
    assert hir.hir_validate_module(_HIR)["ok"] is True


def test_mir_helpers_round_trip_and_classify_control_flow() -> None:
    """MIR serialization and CFG helpers agree on the same structure."""
    payload = mir.mir_serialize(_MIR)
    rebuilt = mir.mir_deserialize(payload)
    assert rebuilt["functions"] == tuple(sorted(_MIR.functions))
    assert mir.mir_serialize(rebuilt["module"]) == payload
    assert mir.mir_cfg_fingerprint(_MIR)["digest"]

    assert mir.mir_topological_order(_ACYCLIC_CFG)["acyclic"] is True
    cycles = mir.mir_detect_cycles(_LOOP_CFG)
    assert cycles["has_cycle"] is True
    assert cycles["back_edges"]
    assert mir.mir_loop_headers(_LOOP_CFG)["headers"]

    merged = mir.mir_merge_blocks(_MERGE_CFG, "middle")
    assert merged["merged_into"] == "exit"
    assert "middle" not in merged["cfg"].blocks
    assert merged["cfg"].blocks["entry"].terminator.targets == ("exit",)
    assert set(_MERGE_CFG.blocks) == {"entry", "middle", "exit"}


def test_passes_fold_propagate_and_bound_the_fixpoint() -> None:
    """Structured passes rewrite copies of the input and report stable traces."""
    assert passes.pass_fold_binary("+", 2, 3)["value"] == 5
    assert passes.pass_fold_binary("==", 2, 3)["result_type"] == "bool"
    assert passes.pass_fold_unary("-", 7)["value"] == -7

    folded_hir = passes.pass_fold_hir_constants(_HIR)
    assert folded_hir["count"] >= 2
    assert folded_hir["module"] is not _HIR
    folded_mir = passes.pass_fold_mir_constants(_MIR)
    assert folded_mir["count"] >= 1
    assert folded_mir["module"] is not _MIR

    propagated = passes.pass_propagate_copies(_MIR)
    assert propagated["module"] is not _MIR
    fixed = passes.pass_fixpoint(
        _MIR, ("propagate_copies", "fold_mir_constants"), max_rounds=2
    )
    assert fixed["rounds"] >= 1
    assert fixed["trace"][0] == "propagate_copies"
    assert fixed["stable"] is True
    assert passes.pass_estimate_cost(_HIR)["score"] > 0


def test_verifiers_report_findings_and_accept_clean_inputs() -> None:
    """Verification helpers distinguish clean IR from structural findings."""
    assert verify.verify_hir_module(_HIR)["ok"] is True
    assert verify.verify_dual_backend_invariants(_SIMPLE_SOURCE)["ok"] is True
    assert verify.verify_mir_roundtrip(_MIR)["ok"] is True
    assert verify.verify_operand_kinds(_MIR)["ok"] is True
    assert verify.verify_function_signatures(_MIR)["ok"] is True
    assert verify.verify_pass_trace(_CALL_ARGS["verify_pass_trace"][0][0])["ok"] is True

    source = "fn f(a: bool, b: bool) -> bool { return a && b; }"
    module = ir_lower_ast_to_hir(source)
    assert verify.verify_constant_folds(module)["count"] == 0
    assert verify.verify_desugared_control_flow(module)["ok"] is False
    desugared = passes.pass_desugar_short_circuit_hir(module)["module"]
    assert verify.verify_desugared_control_flow(desugared)["ok"] is True

    unsupported = HirModule(
        {
            "f": HirFunction(
                "f",
                (),
                HirType("Unsupported"),
                HirExpr("block", HirType("Unsupported")),
            )
        }
    )
    backend_report = verify.verify_backend_type_support(unsupported)
    assert backend_report["ok"] is False
    assert any(issue["code"] == "E007" for issue in backend_report["issues"])
    assert verify.verify_backend_type_support(_HIR)["ok"] is True
    with pytest.raises(PolyglotError) as error:
        verify.verify_backend_type_support("not a module")
    assert error.value.code == "E003"

    trace = ("pass_dedupe_constants", "pass_fold_mir_constants")
    report = verify.verify_pass_trace(trace)
    assert report["ok"] is False
    assert any(issue["code"] == "E005" for issue in report["issues"])

    assert verify.verify_type_unification(_I64, _F64)["ok"] is True
    incompatible = verify.verify_type_unification(_I64, HirType("str"))
    assert incompatible["ok"] is False
    assert incompatible["issues"][0]["code"] == "E001"


def test_structured_diagnostics_cover_documented_codes() -> None:
    """Invalid API inputs and malformed IR raise the documented error codes."""
    with pytest.raises(PolyglotError) as error:
        hir.hir_type_expr(
            HirExpr("var", _I64, value="x"),
            env={"x": "not a type"},  # type: ignore[dict-item]
        )
    assert error.value.code == "E001"

    with pytest.raises(PolyglotError) as error:
        mir.mir_successors(_ACYCLIC_CFG, 123)  # type: ignore[arg-type]
    assert error.value.code == "E003"

    with pytest.raises(PolyglotError) as error:
        hir.hir_lower_block("fn broken {")
    assert error.value.code == "E004"

    with pytest.raises(PolyglotError) as error:
        mir.mir_topological_order(_LOOP_CFG)
    assert error.value.code == "E005"

    with pytest.raises(PolyglotError) as error:
        mir.mir_successors(_ACYCLIC_CFG, "missing")
    assert error.value.code == "E006"

    with pytest.raises(PolyglotError) as error:
        mir.mir_serialize(
            MirModule(
                {
                    "bad": MirFunction(
                        "bad",
                        Cfg(
                            "entry",
                            {
                                "entry": BasicBlock(
                                    "entry",
                                    [
                                        MirInstr(
                                            "op",
                                            args=(MirOperand("opaque", 1, _I64),),
                                        )
                                    ],
                                    MirTerminator("return"),
                                )
                            },
                        ),
                    )
                }
            )
        )
    assert error.value.code == "E007"

    with pytest.raises(PolyglotError) as error:
        mir.mir_deserialize({"format": "polyglot-mir-0", "functions": {}})
    assert error.value.code == "E008"

    with pytest.raises(PolyglotError) as error:
        hir.hir_resolve_calls(_HIR, required=("missing_fn",))
    assert error.value.code == "E006"

    with pytest.raises(PolyglotError) as error:
        passes.pass_fold_binary("+", 1, 2, left_ty="str")
    assert error.value.code == "E001"
