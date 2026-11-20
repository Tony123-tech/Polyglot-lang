"""Behavioral tests for FRAME 3 (lowering, IR, and optimization passes).

All programs are parsed with the real lexer/parser.  The tests exercise each of
the ten framework functions through small deterministic IR programs; the model
never emits backend source or executes user code.
"""

from __future__ import annotations

from compiler.ir.framework import (
    BasicBlock,
    Cfg,
    HirExpr,
    HirType,
    MirFunction,
    MirInstr,
    MirModule,
    MirOperand,
    MirTerminator,
    ir_check_borrow_and_move_state,
    ir_eliminate_dead_blocks,
    ir_fold_constant_expressions,
    ir_hoist_loop_invariant_checks,
    ir_inline_small_pure_functions,
    ir_insert_raii_drop_points,
    ir_lower_ast_to_hir,
    ir_lower_hir_to_mir,
    ir_validate_control_flow_graph,
    ir_verify_dual_backend_invariants,
)

I64 = HirType("i64")
BOOL = HirType("bool")
VEC = HirType("Vec", (I64,))


def _hir_nodes(expr: HirExpr) -> list[HirExpr]:
    """Collect an HIR expression tree including statement values."""
    nodes = [expr]
    for arg in expr.args:
        nodes.extend(_hir_nodes(arg))
    for branch in expr.branches:
        if branch.condition is not None:
            nodes.extend(_hir_nodes(branch.condition))
        nodes.extend(_hir_nodes(branch.body))
    for stmt in expr.meta.get("stmts", ()):
        if stmt.value is not None:
            nodes.extend(_hir_nodes(stmt.value))
    return nodes


def _mir(source: str) -> MirModule:
    """Parse source and lower it all the way to MIR."""
    return ir_lower_hir_to_mir(ir_lower_ast_to_hir(source))


def _first_function_end(module: MirModule) -> tuple[MirModule, str]:
    """Return a MIR module and the entry label of its sole function."""
    name = next(iter(module.functions))
    return module, module.functions[name].cfg.entry


def test_lower_ast_to_hir_infers_types_and_preserves_control_syntax() -> None:
    hir = ir_lower_ast_to_hir(
        """
fn double(x: i64) -> i64 { return x * 2; }
fn main(n: i64) -> i64 {
  let total = 2 + 3 * 4;
  let flag = true && false;
  let choice = if total > 5 { double(total) } else { 0 };
  while total > 0 { total -= 1; }
  match choice { 0 => 1, 1 => 2, _ => 3 }
}
"""
    )

    assert set(hir.functions) == {"double", "main"}
    double, main = hir.functions["double"], hir.functions["main"]
    assert main.params[0].ty == I64
    assert main.ret == I64
    assert double.pure is True
    assert main.pure is False

    statements = main.body.meta["stmts"]
    assert [stmt.name for stmt in statements[:3]] == ["total", "flag", "choice"]
    assert statements[0].ty == I64
    assert statements[1].ty == BOOL
    assert statements[0].value is not None and statements[0].value.kind == "bin"
    assert statements[1].value is not None and statements[1].value.kind == "logical_and"

    kinds = {node.kind for node in _hir_nodes(main.body)}
    assert {"if", "loop", "match", "assign"} <= kinds
    match = next(node for node in _hir_nodes(main.body) if node.kind == "match")
    assert len(match.branches) == 3
    assert match.meta["exhaustive"] is True


def test_lower_hir_to_mir_desugars_short_circuit_loops_and_matches() -> None:
    module = _mir(
        """
fn f(a: bool, b: bool, n: i64) -> i64 {
  let mut i = 0;
  while i < n { if a && b { i += 1; } }
  match n { 0 => 1, _ => 2 }
}
"""
    )
    cfg = module.functions["f"].cfg
    labels = set(cfg.blocks)

    assert any(label.startswith("and_rhs_") for label in labels)
    assert any(label.startswith("and_join_") for label in labels)
    assert any(label.startswith("while_head_") for label in labels)
    assert any(label.startswith("while_body_") for label in labels)
    assert any(label.startswith("match_arm_") for label in labels)
    assert all(block.terminator.kind for block in cfg.blocks.values())
    assert ir_validate_control_flow_graph(cfg).valid

    phis = [
        instr
        for block in cfg.blocks.values()
        for instr in block.instructions
        if instr.op == "phi"
    ]
    assert phis
    assert all("blocks" in instr.meta for instr in phis)


def test_fold_constant_expressions_evaluates_pure_expressions_only() -> None:
    hir = ir_lower_ast_to_hir("fn answer() -> i64 { return 2 + 3 * 4; }")
    folded = ir_fold_constant_expressions(hir)
    assert folded.count >= 2

    body = folded.module.functions["answer"].body
    returned = body.meta["stmts"][0].value.args[0]
    assert returned.kind == "literal"
    assert returned.value == 14

    mir_folded = ir_fold_constant_expressions(ir_lower_hir_to_mir(hir))
    constants = [
        instr
        for function in mir_folded.module.functions.values()
        for block in function.cfg.blocks.values()
        for instr in block.instructions
        if instr.op == "const" and instr.args[0].value == 14
    ]
    assert constants

    unsafe = ir_fold_constant_expressions(
        ir_lower_ast_to_hir("fn bad() -> i64 { return 1 / 0; }")
    )
    assert unsafe.count == 0


def test_eliminate_dead_blocks_prunes_orphans_and_dead_assignments() -> None:
    module = _mir("fn f() -> i64 { let x = 1; let y = 2; return 3; }")
    cfg = module.functions["f"].cfg
    cfg.blocks["orphan"] = BasicBlock(
        "orphan",
        [MirInstr("const", dest="ghost", args=(MirOperand("const", 9, I64),))],
        MirTerminator("return"),
    )

    result = ir_eliminate_dead_blocks(module)
    assert any(item.endswith(":orphan") for item in result.removed_blocks)
    assert "f:entry_1:x" in result.removed_assignments
    assert "f:entry_1:y" in result.removed_assignments
    assert set(result.module.functions["f"].cfg.blocks) == {cfg.entry}

    # The input module is left untouched.
    assert "orphan" in cfg.blocks
    assert any(instr.dest == "x" for instr in cfg.blocks[cfg.entry].instructions)


def test_validate_control_flow_graph_detects_dangling_and_unterminated_blocks() -> None:
    module = _mir(
        "fn f(n: i64) -> i64 { let mut i = 0; while i < n { i += 1; } return i; }"
    )
    valid = ir_validate_control_flow_graph(module)
    assert valid.valid
    assert not any(issue.code == "MissingLoopHeader" for issue in valid.issues)

    cfg = module.functions["f"].cfg
    cfg.blocks[cfg.entry].terminator = MirTerminator("jump", ("nowhere",))
    broken = ir_validate_control_flow_graph(module)
    assert not broken.valid
    assert any(issue.code == "DanglingTarget" for issue in broken.issues)

    cfg.blocks[cfg.entry].terminator = MirTerminator("")
    unterminated = ir_validate_control_flow_graph(module)
    assert any(issue.code == "UnterminatedBlock" for issue in unterminated.issues)


def test_check_borrow_and_move_state_flags_use_after_move() -> None:
    module = _mir("fn f(x: Vec<i64>) -> i64 { let y = x; let z = x; return 0; }")
    report = ir_check_borrow_and_move_state(module)

    assert report.ok is False
    assert any(
        issue.code == "UseAfterMove" and issue.variable == "x"
        for issue in report.issues
    )


def test_check_borrow_and_move_state_merges_branch_states() -> None:
    cfg = Cfg(
        "entry",
        {
            "entry": BasicBlock(
                "entry",
                [],
                MirTerminator(
                    "branch",
                    ("moved", "kept"),
                    cond=MirOperand("const", True, BOOL),
                ),
            ),
            "moved": BasicBlock(
                "moved",
                [MirInstr("move", dest="y", args=(MirOperand("var", "x", VEC),))],
                MirTerminator("jump", ("join",)),
            ),
            "kept": BasicBlock("kept", [], MirTerminator("jump", ("join",))),
            "join": BasicBlock(
                "join",
                [MirInstr("copy", dest="z", args=(MirOperand("var", "x", VEC),))],
                MirTerminator("return", value=MirOperand("var", "z", I64)),
            ),
        },
    )
    report = ir_check_borrow_and_move_state(MirModule({"f": MirFunction("f", cfg)}))

    assert report.ok is False
    assert any(
        issue.code == "UseAfterMove"
        and issue.block.endswith(":join")
        and "maybe_moved" in issue.message
        for issue in report.issues
    )


def test_insert_raii_drop_points_places_cleanup_on_early_return() -> None:
    module = _mir(
        """
fn f(c: bool) -> i64 {
  let v: Vec<i64> = 0;
  if c { return v; }
  return 0;
}
"""
    )
    result = ir_insert_raii_drop_points(module)

    assert len(result.insertions) == 1
    point = result.insertions[0]
    assert point.variable == "v"
    assert point.reason == "return"
    assert point.block.startswith("if_then_")

    dropped = [
        instr
        for block in result.module.functions["f"].cfg.blocks.values()
        for instr in block.instructions
        if instr.op == "drop"
    ]
    assert len(dropped) == 1
    assert dropped[0].args[0].value == "v"


def test_inline_small_pure_functions_substitutes_arguments() -> None:
    hir = ir_lower_ast_to_hir(
        "fn double(x: i64) -> i64 { return x * 2; } "
        "fn main() -> i64 { return double(5); }"
    )
    result = ir_inline_small_pure_functions(hir)

    assert result.inlined_sites == ("main:double",)
    main = result.module.functions["main"]
    nodes = _hir_nodes(main.body)
    assert not any(node.kind == "call" for node in nodes)
    assert [node.value for node in nodes if node.kind == "literal"] == [5, 2]


def test_inline_small_pure_functions_skips_recursive_callees() -> None:
    hir = ir_lower_ast_to_hir(
        """
fn countdown(n: i64) -> i64 {
  if n <= 0 { return 0; }
  return countdown(n - 1);
}
fn main() -> i64 { return countdown(3); }
"""
    )
    result = ir_inline_small_pure_functions(hir, max_cost=100)

    assert result.inlined_sites == ()
    assert any(site.endswith(":countdown") for site in result.skipped)


def test_hoist_loop_invariant_checks_uses_preheader() -> None:
    true = MirOperand("const", True, BOOL)
    cfg = Cfg(
        "entry",
        {
            "entry": BasicBlock("entry", [], MirTerminator("jump", ("head",))),
            "head": BasicBlock(
                "head",
                [
                    MirInstr("guard", meta={"name": "invariant"}),
                    MirInstr(
                        "guard",
                        args=(MirOperand("var", "t1", I64),),
                        meta={"name": "variant"},
                    ),
                ],
                MirTerminator("branch", ("body", "exit"), cond=true),
            ),
            "body": BasicBlock(
                "body",
                [
                    MirInstr(
                        "binop", dest="t1", args=(true, true), meta={"operator": "+"}
                    )
                ],
                MirTerminator("jump", ("head",)),
            ),
            "exit": BasicBlock("exit", [], MirTerminator("return")),
        },
    )
    module = MirModule({"f": MirFunction("f", cfg)})
    result = ir_hoist_loop_invariant_checks(module)

    assert result.hoisted == ("f:invariant",)
    assert result.preheaders == ("f:preheader_head",)
    new_cfg = result.module.functions["f"].cfg
    assert set(new_cfg.blocks) == {"entry", "head", "body", "exit", "preheader_head"}
    assert new_cfg.blocks["entry"].terminator.targets == ("preheader_head",)
    assert new_cfg.blocks["preheader_head"].terminator.targets == ("head",)
    assert [
        instr.meta.get("name")
        for instr in new_cfg.blocks["preheader_head"].instructions
    ] == ["invariant"]
    assert [
        instr.meta.get("name") for instr in new_cfg.blocks["head"].instructions
    ] == ["variant"]
    # The source CFG is not mutated by the pass.
    assert len(cfg.blocks["head"].instructions) == 2


def test_verify_dual_backend_invariants_accepts_and_rejects_programs() -> None:
    accepted = ir_verify_dual_backend_invariants(
        "fn f(n: i64) -> i64 { if n > 0 { 1 } else { 0 } }"
    )
    assert accepted.ok
    assert "terminated_blocks" in accepted.checked
    assert "lowered_short_circuit" in accepted.checked

    unresolved = _mir("fn f(n: i64) -> i64 { return mystery(n); }")
    unresolved_report = ir_verify_dual_backend_invariants(unresolved)
    assert not unresolved_report.ok
    assert any(issue.code == "UnresolvedCall" for issue in unresolved_report.issues)

    unsafe_cfg = Cfg(
        "entry",
        {
            "entry": BasicBlock(
                "entry",
                [MirInstr("const", dest="class", args=(MirOperand("const", 1, I64),))],
                MirTerminator("return", value=MirOperand("var", "class", I64)),
            )
        },
    )
    unsafe_report = ir_verify_dual_backend_invariants(
        MirModule({"f": MirFunction("f", unsafe_cfg)})
    )
    assert any(issue.code == "UnsafeIdentifier" for issue in unsafe_report.issues)


def test_lowering_pipeline_is_deterministic() -> None:
    source = """
fn pick(a: bool, b: bool, n: i64) -> i64 {
  let mut acc = 0;
  while acc < n { if a || b { acc += 1; } }
  match acc { 0 => 1, _ => 2 }
}
"""

    def summary(module: MirModule) -> tuple[object, ...]:
        result: list[object] = []
        for name, function in module.functions.items():
            cfg = function.cfg
            blocks = tuple(
                (
                    label,
                    tuple(instr.op for instr in block.instructions),
                    block.terminator.kind,
                    block.terminator.targets,
                )
                for label, block in sorted(cfg.blocks.items())
            )
            result.append((name, cfg.entry, blocks))
        return tuple(result)

    assert summary(_mir(source)) == summary(_mir(source))
    _, entry = _first_function_end(_mir(source))
    assert entry.startswith("entry_")
