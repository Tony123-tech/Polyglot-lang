"""Deterministic optimizer passes for the FRAME 3 rollout.

Passes consume the model HIR/MIR dataclasses from :mod:`compiler.ir.framework`
and return fresh dictionaries containing the rewritten IR plus an audit trail.
Constant folding consults only a closed table of pure operators; user code is
never executed, no backend is invoked, and no native memory is allocated.
Legacy ten-function passes are reused through their documented public surface
and adapted to the structured dictionary convention of this rollout.

Diagnostic conventions used here:

* ``E001`` – incompatible operand types in an explicit fold request;
* ``E003`` – API misuse (bad module, unknown pass, malformed options);
* ``E005`` – a cyclic CFG supplied to a pass that requires a DAG.
"""

from __future__ import annotations

import operator
import re
from dataclasses import replace
from typing import Any

from compiler.frontend.errors import PolyglotError
from compiler.ir.framework import (
    BasicBlock,
    Cfg,
    HirBranch,
    HirExpr,
    HirFunction,
    HirModule,
    HirStmt,
    MirFunction,
    MirInstr,
    MirModule,
    MirOperand,
    MirTerminator,
    ir_check_borrow_and_move_state,
    ir_eliminate_dead_blocks,
    ir_hoist_loop_invariant_checks,
    ir_inline_small_pure_functions,
    ir_insert_raii_drop_points,
    ir_validate_control_flow_graph,
    ir_verify_dual_backend_invariants,
)
from compiler.ir.hir import (
    hir_const_env,
    hir_lower_short_circuit,
    hir_module_fingerprint,
)
from compiler.ir.mir import mir_cfg_fingerprint, mir_reachable

__all__ = [
    "pass_check_borrows",
    "pass_constant_env",
    "pass_count_effects",
    "pass_dead_branch_elimination",
    "pass_dedupe_constants",
    "pass_desugar_short_circuit_hir",
    "pass_eliminate_dead_code",
    "pass_estimate_cost",
    "pass_fixpoint",
    "pass_fold_binary",
    "pass_fold_hir_constants",
    "pass_fold_mir_constants",
    "pass_fold_unary",
    "pass_hoist_guards",
    "pass_inline_pure_calls",
    "pass_insert_drops",
    "pass_merge_jump_chains",
    "pass_propagate_copies",
    "pass_remove_empty_blocks",
    "pass_remove_unreachable",
    "pass_rename_temporaries",
    "pass_simplify_phi",
    "pass_strip_metadata",
    "pass_validate_cfg",
    "pass_verify_invariants",
]

_BINARY_OPERATORS: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}
_UNARY_OPERATORS: dict[str, Any] = {
    "-": operator.neg,
    "+": operator.pos,
    "!": operator.not_,
}
_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_NUMERIC_TYPES = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "usize",
        "f32",
        "f64",
    }
)
_IMPURE_CALLEES = frozenset(
    {"print", "println", "eprint", "input", "read", "write", "panic", "rand", "now"}
)
_TEMP_RE = re.compile(r"^t([0-9]+)$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(code: str, message: str, **context: Any) -> None:
    """Raise a structured IR diagnostic with optional context keys."""
    details = ", ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
    raise PolyglotError(
        code,
        f"{message} ({details})" if details else message,
        phase="ir",
    )


def _require_mir(module: object) -> MirModule:
    """Return *module* or raise E003 for a non-MIR module."""
    if not isinstance(module, MirModule):
        _fail("E003", "expected a MirModule", value=type(module).__name__)
    return module


def _require_value(value: object, kind: type, label: str) -> Any:
    """Return a value of the required type or raise E003."""
    if not isinstance(value, kind):
        _fail("E003", f"{label} must be {kind.__name__}", value=type(value).__name__)
    return value


def _compatible(left_ty: str, right_ty: str) -> bool:
    """Whether two declared scalar types may participate in one operator."""
    if left_ty == right_ty or "?" in {left_ty, right_ty}:
        return True
    return left_ty in _NUMERIC_TYPES and right_ty in _NUMERIC_TYPES


def _result_type(op: str, left_ty: str, right_ty: str = "") -> str:
    """Choose the deterministic declared result type of a pure operator."""
    if op in _COMPARISONS:
        return "bool"
    if "f32" in {left_ty, right_ty} or "f64" in {left_ty, right_ty}:
        return "f64"
    return left_ty


def _eval_binary(op: str, left: Any, right: Any) -> tuple[bool, Any, str]:
    """Evaluate one closed-table binary operator without executing user code."""
    func = _BINARY_OPERATORS.get(op)
    if func is None:
        return False, None, "unsupported_operator"
    if op in {"&", "|", "^"} and isinstance(left, bool) != isinstance(right, bool):
        return False, None, "type_mismatch"
    try:
        return True, func(left, right), "folded"
    except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
        return False, None, type(error).__name__.lower()


def _eval_unary(op: str, value: Any) -> tuple[bool, Any, str]:
    """Evaluate one closed-table unary operator without executing user code."""
    func = _UNARY_OPERATORS.get(op)
    if func is None:
        return False, None, "unsupported_operator"
    try:
        return True, func(value), "folded"
    except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
        return False, None, type(error).__name__.lower()


def _copy_operand(operand: MirOperand) -> MirOperand:
    """Copy one immutable MIR operand."""
    return MirOperand(operand.kind, operand.value, operand.ty)


def _copy_instr(instr: MirInstr) -> MirInstr:
    """Copy one MIR instruction including metadata."""
    return MirInstr(
        instr.op,
        instr.dest,
        tuple(_copy_operand(arg) for arg in instr.args),
        dict(instr.meta),
        instr.span,
    )


def _copy_cfg(cfg: Cfg) -> Cfg:
    """Copy a CFG so a pass can rewrite it without mutating its input."""
    blocks = {
        label: BasicBlock(
            label,
            [_copy_instr(instr) for instr in block.instructions],
            MirTerminator(
                block.terminator.kind,
                tuple(block.terminator.targets),
                None
                if block.terminator.cond is None
                else _copy_operand(block.terminator.cond),
                None
                if block.terminator.value is None
                else _copy_operand(block.terminator.value),
                dict(block.terminator.meta),
            ),
            dict(block.meta),
        )
        for label, block in cfg.blocks.items()
    }
    return Cfg(cfg.entry, blocks)


def _map_hir(expr: HirExpr, transform: Any) -> HirExpr:
    """Rebuild a HIR tree bottom-up with an internal transform callback."""
    args = tuple(_map_hir(arg, transform) for arg in expr.args)
    branches = tuple(
        HirBranch(
            None if branch.condition is None else _map_hir(branch.condition, transform),
            _map_hir(branch.body, transform),
            dict(branch.meta),
        )
        for branch in expr.branches
    )
    stmts: list[HirStmt] = []
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt):
            stmts.append(
                replace(
                    stmt,
                    value=None
                    if stmt.value is None
                    else _map_hir(stmt.value, transform),
                )
            )
    meta = dict(expr.meta)
    if stmts or "stmts" in meta:
        meta["stmts"] = tuple(stmts)
    return transform(replace(expr, args=args, branches=branches, meta=meta))


def _hir_walk(expr: HirExpr) -> tuple[HirExpr, ...]:
    """Yield one HIR expression and all of its nested expressions."""
    result = [expr]
    for arg in expr.args:
        result.extend(_hir_walk(arg))
    for branch in expr.branches:
        if branch.condition is not None:
            result.extend(_hir_walk(branch.condition))
        result.extend(_hir_walk(branch.body))
    for stmt in expr.meta.get("stmts", ()):
        if isinstance(stmt, HirStmt) and stmt.value is not None:
            result.extend(_hir_walk(stmt.value))
    return tuple(result)


def _fingerprint(module: HirModule | MirModule) -> str:
    """Canonical digest for either supported IR module kind."""
    if isinstance(module, HirModule):
        return hir_module_fingerprint(module)["digest"]
    if isinstance(module, MirModule):
        return mir_cfg_fingerprint(module)["digest"]
    _fail("E003", "unsupported IR module for fingerprint", value=type(module).__name__)


def _issues(records: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """Normalize dataclass issue records into plain dictionaries."""
    return tuple(
        {
            "code": getattr(record, "code", ""),
            "block": getattr(record, "block", getattr(record, "function", "")),
            "message": getattr(record, "message", ""),
            "severity": getattr(record, "severity", "error"),
        }
        for record in records
    )


# ---------------------------------------------------------------------------
# Public passes
# ---------------------------------------------------------------------------


def pass_fold_binary(
    op: str,
    left: object,
    right: object,
    *,
    left_ty: str = "i64",
    right_ty: str = "i64",
) -> dict[str, Any]:
    """Fold one explicitly requested binary operation using the closed operator table."""
    _require_value(op, str, "operator")
    _require_value(left_ty, str, "left_ty")
    _require_value(right_ty, str, "right_ty")
    if op not in _BINARY_OPERATORS:
        _fail("E003", "unsupported binary operator", op=op)
    if not _compatible(left_ty, right_ty):
        _fail("E001", "incompatible operand types", left=left_ty, right=right_ty)
    folded, value, reason = _eval_binary(op, left, right)
    return {
        "op": op,
        "folded": folded,
        "value": value,
        "result_type": _result_type(op, left_ty, right_ty),
        "reason": reason,
    }


def pass_fold_unary(op: str, value: object, *, ty: str = "i64") -> dict[str, Any]:
    """Fold one explicitly requested unary operation using the closed operator table."""
    _require_value(op, str, "operator")
    _require_value(ty, str, "type")
    if op not in _UNARY_OPERATORS:
        _fail("E003", "unsupported unary operator", op=op)
    folded, result, reason = _eval_unary(op, value)
    return {
        "op": op,
        "folded": folded,
        "value": result,
        "result_type": "bool" if op == "!" else ty,
        "reason": reason,
    }


def pass_fold_hir_constants(module: HirModule) -> dict[str, Any]:
    """Fold literal-only HIR arithmetic and comparisons into literal nodes."""
    root = _require_value(module, HirModule, "module")
    replacements: list[str] = []

    def transform(expr: HirExpr) -> HirExpr:
        if expr.kind not in {"bin", "unop"} or len(expr.args) < 1:
            return expr
        if not all(arg.kind == "literal" for arg in expr.args):
            return expr
        op = str(expr.value)
        if expr.kind == "unop":
            folded, value, _reason = _eval_unary(op, expr.args[0].value)
        else:
            if len(expr.args) != 2:
                return expr
            folded, value, _reason = _eval_binary(
                op, expr.args[0].value, expr.args[1].value
            )
        if not folded:
            return expr
        replacements.append(f"{op}:{value!r}")
        return replace(expr, kind="literal", value=value)

    functions: dict[str, HirFunction] = {}
    for name, function in root.functions.items():
        functions[name] = replace(function, body=_map_hir(function.body, transform))
    return {
        "module": HirModule(functions, root.source),
        "replacements": tuple(replacements),
        "count": len(replacements),
    }


def pass_fold_mir_constants(module: MirModule) -> dict[str, Any]:
    """Fold MIR binary/unary instructions whose operands are known constants."""
    root = _require_mir(module)
    replacements: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label in sorted(cfg.blocks):
            block = cfg.blocks[label]
            known: dict[str, MirOperand] = {}
            instructions: list[MirInstr] = []
            for instr in block.instructions:
                args = tuple(
                    known.get(str(arg.value), arg) if arg.kind == "var" else arg
                    for arg in instr.args
                )
                current = replace(instr, args=args)
                if current.op in {"binop", "unop"} and all(
                    arg.kind == "const" for arg in current.args
                ):
                    op = str(current.meta.get("operator", ""))
                    if current.op == "unop":
                        folded, value, _reason = _eval_unary(op, current.args[0].value)
                    else:
                        folded, value, _reason = _eval_binary(
                            op, current.args[0].value, current.args[1].value
                        )
                    if folded:
                        constant = MirOperand("const", value, current.args[0].ty)
                        replacements.append(f"{name}:{label}:{op}")
                        if current.dest:
                            known[current.dest] = constant
                        instructions.append(
                            MirInstr(
                                "const",
                                current.dest,
                                (constant,),
                                dict(current.meta),
                                current.span,
                            )
                        )
                        continue
                instructions.append(current)
            block.instructions = instructions
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "replacements": tuple(replacements),
        "count": len(replacements),
    }


def pass_propagate_copies(module: MirModule) -> dict[str, Any]:
    """Propagate block-local constant copies into later instruction operands."""
    root = _require_mir(module)
    propagated: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label in sorted(cfg.blocks):
            block = cfg.blocks[label]
            known: dict[str, MirOperand] = {}
            instructions: list[MirInstr] = []
            for instr in block.instructions:
                source = instr.args[0] if instr.args else None
                const_define = (
                    instr.op == "const"
                    and instr.dest is not None
                    and source is not None
                )
                const_copy = (
                    instr.op == "copy"
                    and instr.dest is not None
                    and source is not None
                    and source.kind == "const"
                )
                if const_define or const_copy:
                    known[instr.dest] = _copy_operand(source)
                args = tuple(
                    known.get(str(arg.value), arg) if arg.kind == "var" else arg
                    for arg in instr.args
                )
                for before, after in zip(instr.args, args):
                    if before != after:
                        propagated.append(f"{name}:{label}:{before.value}")
                instructions.append(replace(instr, args=args))
            block.instructions = instructions
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "propagated": tuple(dict.fromkeys(propagated)),
        "count": len(propagated),
    }


def pass_eliminate_dead_code(module: MirModule) -> dict[str, Any]:
    """Prune unreachable blocks and dead assignments via the legacy pass."""
    root = _require_mir(module)
    result = ir_eliminate_dead_blocks(root)
    return {
        "module": result.module,
        "removed_blocks": result.removed_blocks,
        "removed_assignments": result.removed_assignments,
        "count": len(result.removed_blocks) + len(result.removed_assignments),
    }


def pass_remove_empty_blocks(module: MirModule) -> dict[str, Any]:
    """Delete instruction-free jump blocks and redirect their predecessors."""
    root = _require_mir(module)
    removed: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        while True:
            candidates = [
                label
                for label, block in sorted(cfg.blocks.items())
                if label != cfg.entry
                and not block.instructions
                and block.terminator.kind == "jump"
                and len(block.terminator.targets) == 1
                and block.terminator.targets[0] in cfg.blocks
                and block.terminator.targets[0] != label
            ]
            if not candidates:
                break
            label = candidates[0]
            target = cfg.blocks[label].terminator.targets[0]
            for block in cfg.blocks.values():
                block.terminator.targets = tuple(
                    target if item == label else item
                    for item in block.terminator.targets
                )
            del cfg.blocks[label]
            removed.append(f"{name}:{label}")
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "removed": tuple(removed),
        "count": len(removed),
    }


def pass_merge_jump_chains(module: MirModule) -> dict[str, Any]:
    """Shorten chains of empty jump blocks to their final destinations."""
    root = _require_mir(module)
    shortened: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label in sorted(cfg.blocks):
            seen: set[str] = {label}
            current = label
            while True:
                block = cfg.blocks[current]
                if block.instructions or block.terminator.kind != "jump":
                    break
                targets = block.terminator.targets
                if (
                    len(targets) != 1
                    or targets[0] not in cfg.blocks
                    or targets[0] in seen
                ):
                    break
                seen.add(targets[0])
                current = targets[0]
            if current != label:
                cfg.blocks[label].terminator.targets = (current,)
                shortened.append(f"{name}:{label}")
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "shortened": tuple(shortened),
        "count": len(shortened),
    }


def pass_rename_temporaries(module: MirModule) -> dict[str, Any]:
    """Rename numbered temporaries to a fresh contiguous per-function sequence."""
    root = _require_mir(module)
    functions: dict[str, MirFunction] = {}
    mappings: dict[str, dict[str, str]] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        existing = [
            instr.dest
            for block in cfg.blocks.values()
            for instr in block.instructions
            if instr.dest
        ]
        numbers = [
            int(match.group(1))
            for dest in existing
            if (match := _TEMP_RE.fullmatch(dest)) is not None
        ]
        start = max(numbers, default=0) + 1
        names = sorted(
            {
                instr.dest
                for block in cfg.blocks.values()
                for instr in block.instructions
                if instr.dest and _TEMP_RE.fullmatch(instr.dest)
            }
        )
        mapping = {old: f"t{start + index}" for index, old in enumerate(names)}
        for block in cfg.blocks.values():
            for instr in block.instructions:
                if instr.dest in mapping:
                    instr.dest = mapping[instr.dest]
                instr.args = tuple(
                    MirOperand(
                        "var",
                        mapping[str(arg.value)],
                        arg.ty,
                    )
                    if arg.kind == "var" and str(arg.value) in mapping
                    else arg
                    for arg in instr.args
                )
        functions[name] = replace(function, cfg=cfg)
        mappings[name] = dict(mapping)
    return {
        "module": MirModule(functions, root.source),
        "renames": mappings,
        "count": sum(len(mapping) for mapping in mappings.values()),
    }


def pass_inline_pure_calls(
    module: HirModule | MirModule, *, max_cost: int = 12
) -> dict[str, Any]:
    """Inline small pure functions through the legacy pass with structured output."""
    if not isinstance(module, HirModule | MirModule):
        _fail("E003", "expected a HirModule or MirModule")
    if not isinstance(max_cost, int) or max_cost < 0:
        _fail("E003", "max_cost must be a non-negative int")
    result = ir_inline_small_pure_functions(module, max_cost=max_cost)
    return {
        "module": result.module,
        "inlined_sites": result.inlined_sites,
        "skipped": result.skipped,
        "count": len(result.inlined_sites),
    }


def pass_hoist_guards(module: MirModule) -> dict[str, Any]:
    """Hoist loop-invariant guards into preheaders using the legacy pass."""
    root = _require_mir(module)
    result = ir_hoist_loop_invariant_checks(root)
    return {
        "module": result.module,
        "hoisted": result.hoisted,
        "preheaders": result.preheaders,
        "count": len(result.hoisted),
    }


def pass_insert_drops(module: MirModule) -> dict[str, Any]:
    """Insert deterministic RAII cleanup points using the legacy pass."""
    root = _require_mir(module)
    result = ir_insert_raii_drop_points(root)
    return {
        "module": result.module,
        "insertions": tuple(
            {
                "function": point.function,
                "block": point.block,
                "variable": point.variable,
                "reason": point.reason,
            }
            for point in result.insertions
        ),
        "count": len(result.insertions),
    }


def pass_validate_cfg(module: MirModule) -> dict[str, Any]:
    """Validate CFG structure and return issue dictionaries plus the input module."""
    root = _require_mir(module)
    result = ir_validate_control_flow_graph(root)
    issues = _issues(result.issues)
    return {
        "module": root,
        "valid": result.valid,
        "issues": issues,
        "count": len(issues),
    }


def pass_check_borrows(module: MirModule) -> dict[str, Any]:
    """Report borrow and move findings as structured dictionaries."""
    root = _require_mir(module)
    result = ir_check_borrow_and_move_state(root)
    issues = _issues(result.issues)
    return {
        "module": root,
        "ok": result.ok,
        "issues": issues,
        "states": {key: dict(value) for key, value in sorted(result.states.items())},
        "count": len(issues),
    }


def pass_verify_invariants(target: Any) -> dict[str, Any]:
    """Check shared backend invariants through the legacy verifier."""
    report = ir_verify_dual_backend_invariants(target)
    issues = _issues(report.issues)
    return {
        "ok": report.ok,
        "issues": issues,
        "checked": report.checked,
        "count": len(issues),
    }


def pass_constant_env(module: HirModule) -> dict[str, Any]:
    """Build per-function constant environments from literal let bindings."""
    root = _require_value(module, HirModule, "module")
    environments: dict[str, dict[str, Any]] = {}
    for name, function in root.functions.items():
        environments[name] = hir_const_env(function.body)["constants"]
    return {
        "constants": environments,
        "functions": tuple(sorted(environments)),
        "count": sum(len(env) for env in environments.values()),
    }


def pass_dead_branch_elimination(module: MirModule) -> dict[str, Any]:
    """Replace branches with constant conditions by unconditional jumps."""
    root = _require_mir(module)
    simplified: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label in sorted(cfg.blocks):
            terminator = cfg.blocks[label].terminator
            cond = terminator.cond
            if (
                terminator.kind != "branch"
                or cond is None
                or cond.kind != "const"
                or not isinstance(cond.value, bool)
                or len(terminator.targets) != 2
            ):
                continue
            target = terminator.targets[0] if cond.value else terminator.targets[1]
            cfg.blocks[label].terminator = MirTerminator(
                "jump", (target,), None, terminator.value, dict(terminator.meta)
            )
            simplified.append(f"{name}:{label}")
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "simplified": tuple(simplified),
        "count": len(simplified),
    }


def pass_remove_unreachable(module: MirModule) -> dict[str, Any]:
    """Drop blocks that are unreachable from each function entry."""
    root = _require_mir(module)
    removed: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        reachable = set(mir_reachable(function.cfg)["reachable"])
        blocks = {
            label: block
            for label, block in function.cfg.blocks.items()
            if label in reachable
        }
        removed.extend(
            f"{name}:{label}" for label in sorted(set(function.cfg.blocks) - reachable)
        )
        if reachable:
            functions[name] = replace(function, cfg=Cfg(function.cfg.entry, blocks))
        else:
            functions[name] = function
    return {
        "module": MirModule(functions, root.source),
        "removed": tuple(removed),
        "count": len(removed),
    }


def pass_simplify_phi(module: MirModule) -> dict[str, Any]:
    """Replace phi nodes with a single incoming value by simple copies."""
    root = _require_mir(module)
    simplified: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label in sorted(cfg.blocks):
            instructions: list[MirInstr] = []
            for instr in cfg.blocks[label].instructions:
                if instr.op == "phi" and instr.dest and instr.args:
                    first = instr.args[0]
                    if all(arg == first for arg in instr.args):
                        instructions.append(
                            MirInstr(
                                "copy",
                                instr.dest,
                                (first,),
                                dict(instr.meta),
                                instr.span,
                            )
                        )
                        simplified.append(f"{name}:{label}:{instr.dest}")
                        continue
                instructions.append(instr)
            cfg.blocks[label].instructions = instructions
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "simplified": tuple(simplified),
        "count": len(simplified),
    }


def pass_strip_metadata(module: MirModule, keys: tuple[str, ...]) -> dict[str, Any]:
    """Remove named metadata keys from blocks, instructions, and terminators."""
    root = _require_mir(module)
    if not isinstance(keys, tuple) or not all(isinstance(key, str) for key in keys):
        _fail("E003", "keys must be a tuple of strings")
    removed: list[str] = []
    functions: dict[str, MirFunction] = {}
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        for label, block in cfg.blocks.items():
            for scope, meta in (
                ("block", block.meta),
                *(("instr", instr.meta) for instr in block.instructions),
                ("terminator", block.terminator.meta),
            ):
                for key in keys:
                    if key in meta:
                        del meta[key]
                        removed.append(f"{name}:{label}:{scope}:{key}")
        functions[name] = replace(function, cfg=cfg)
    return {
        "module": MirModule(functions, root.source),
        "removed": tuple(removed),
        "count": len(removed),
    }


def pass_desugar_short_circuit_hir(module: HirModule) -> dict[str, Any]:
    """Replace HIR short-circuit operators with explicit branch expressions."""
    root = _require_value(module, HirModule, "module")
    functions: dict[str, HirFunction] = {}
    rewritten = 0
    for name, function in root.functions.items():
        result = hir_lower_short_circuit(function.body)
        functions[name] = replace(function, body=result["expr"])
        rewritten += result["rewritten"]
    return {
        "module": HirModule(functions, root.source),
        "rewritten": rewritten,
        "count": rewritten,
    }


def pass_dedupe_constants(module: MirModule) -> dict[str, Any]:
    """Hoist repeated literal operands into one named constant per distinct value."""
    root = _require_mir(module)
    functions: dict[str, MirFunction] = {}
    pools: dict[str, tuple[tuple[str, MirOperand], ...]] = {}
    deduped = 0
    for name, function in root.functions.items():
        cfg = _copy_cfg(function.cfg)
        occurrences: dict[tuple[str, str], int] = {}
        for block in cfg.blocks.values():
            for instr in block.instructions:
                for arg in instr.args:
                    if arg.kind == "const":
                        key = (repr(arg.value), arg.ty.text)
                        occurrences[key] = occurrences.get(key, 0) + 1
        repeated = {key for key, count in occurrences.items() if count > 1}
        pool: list[tuple[str, MirOperand]] = []
        used_names = {
            instr.dest
            for block in cfg.blocks.values()
            for instr in block.instructions
            if instr.dest
        }
        for index, key in enumerate(sorted(repeated), 1):
            value_repr, type_text = key
            operand = next(
                arg
                for block in cfg.blocks.values()
                for instr in block.instructions
                for arg in instr.args
                if arg.kind == "const"
                and repr(arg.value) == value_repr
                and arg.ty.text == type_text
            )
            dest = f"c{index}"
            while dest in used_names:
                dest = f"{dest}_"
            used_names.add(dest)
            pool.append((dest, operand))
        replacements = {
            (repr(operand.value), operand.ty.text): dest for dest, operand in pool
        }
        for block in cfg.blocks.values():
            for instr in block.instructions:
                rewritten: list[MirOperand] = []
                for arg in instr.args:
                    if arg.kind == "const":
                        key = (repr(arg.value), arg.ty.text)
                        if key in replacements:
                            dest = replacements[key]
                            rewritten.append(MirOperand("var", dest, arg.ty))
                            deduped += 1
                            continue
                    rewritten.append(arg)
                instr.args = tuple(rewritten)
        if cfg.entry in cfg.blocks and pool:
            entry = cfg.blocks[cfg.entry]
            entry.instructions = [
                MirInstr("const", dest, (operand,), {"deduped": True})
                for dest, operand in pool
            ] + entry.instructions
        functions[name] = replace(function, cfg=cfg)
        pools[name] = tuple((dest, operand) for dest, operand in pool)
    return {
        "module": MirModule(functions, root.source),
        "pools": pools,
        "count": deduped,
    }


def pass_count_effects(module: MirModule) -> dict[str, Any]:
    """Count modeled side effects (stores, guards, drops, and impure calls)."""
    root = _require_mir(module)
    per_function: dict[str, dict[str, int]] = {}
    for name, function in root.functions.items():
        counts = {"store": 0, "guard": 0, "drop": 0, "impure_call": 0}
        for block in function.cfg.blocks.values():
            for instr in block.instructions:
                if instr.op in counts:
                    counts[instr.op] += 1
                elif (
                    instr.op == "call"
                    and str(instr.meta.get("callee", "")) in _IMPURE_CALLEES
                ):
                    counts["impure_call"] += 1
        per_function[name] = counts
    totals = {
        kind: sum(item[kind] for item in per_function.values())
        for kind in ("store", "guard", "drop", "impure_call")
    }
    return {
        "functions": per_function,
        "totals": totals,
        "total": sum(totals.values()),
    }


def pass_fixpoint(
    module: MirModule,
    passes: tuple[str, ...],
    *,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Run named MIR passes to a structural fixed point with a bounded round count."""
    root = _require_mir(module)
    if not isinstance(passes, tuple) or not all(
        isinstance(name, str) for name in passes
    ):
        _fail("E003", "passes must be a tuple of names")
    if not isinstance(max_rounds, int) or max_rounds < 1:
        _fail("E003", "max_rounds must be a positive int")
    registry = {
        "fold_mir_constants": pass_fold_mir_constants,
        "propagate_copies": pass_propagate_copies,
        "eliminate_dead_code": pass_eliminate_dead_code,
        "remove_empty_blocks": pass_remove_empty_blocks,
        "merge_jump_chains": pass_merge_jump_chains,
        "simplify_phi": pass_simplify_phi,
        "dead_branch_elimination": pass_dead_branch_elimination,
        "remove_unreachable": pass_remove_unreachable,
        "dedupe_constants": pass_dedupe_constants,
        "rename_temporaries": pass_rename_temporaries,
        "validate_cfg": pass_validate_cfg,
    }
    for name in passes:
        if name not in registry:
            _fail("E003", "unknown pass name", name=name)
    current = root
    trace: list[str] = []
    rounds = 0
    for _round in range(max_rounds):
        before = _fingerprint(current)
        for name in passes:
            current = registry[name](current)["module"]
            trace.append(name)
        rounds += 1
        if _fingerprint(current) == before:
            break
    return {
        "module": current,
        "trace": tuple(trace),
        "rounds": rounds,
        "stable": _fingerprint(current) == before,
    }


def pass_estimate_cost(module: HirModule | MirModule) -> dict[str, Any]:
    """Estimate deterministic IR size metrics per function and for the whole module."""
    if isinstance(module, HirModule):
        functions = {
            name: {
                "nodes": len(_hir_walk(function.body)),
                "blocks": 0,
                "instructions": 0,
                "edges": 0,
                "phis": 0,
                "constants": sum(
                    1 for node in _hir_walk(function.body) if node.kind == "literal"
                ),
            }
            for name, function in module.functions.items()
        }
    elif isinstance(module, MirModule):
        functions = {}
        for name, function in module.functions.items():
            blocks = function.cfg.blocks
            functions[name] = {
                "nodes": 0,
                "blocks": len(blocks),
                "instructions": sum(
                    len(block.instructions) for block in blocks.values()
                ),
                "edges": sum(
                    len(block.terminator.targets) for block in blocks.values()
                ),
                "phis": sum(
                    1
                    for block in blocks.values()
                    for instr in block.instructions
                    if instr.op == "phi"
                ),
                "constants": sum(
                    1
                    for block in blocks.values()
                    for instr in block.instructions
                    if instr.op == "const"
                ),
            }
    else:
        _fail("E003", "expected a HirModule or MirModule")
    keys = ("nodes", "blocks", "instructions", "edges", "phis", "constants")
    totals = {key: sum(item[key] for item in functions.values()) for key in keys}
    return {
        "functions": {name: functions[name] for name in sorted(functions)},
        "totals": totals,
        "score": totals["nodes"] + totals["instructions"] + 2 * totals["phis"],
    }
