"""Deterministic MIR/CFG inspection helpers for the FRAME 3 rollout.

Every function in this module works on the behavior-free dataclasses declared
by :mod:`compiler.ir.framework` and returns plain dictionaries.  The helpers
only inspect or copy control-flow graphs: they never emit code, allocate native
memory, or execute program expressions.  Serialization uses an explicit,
versioned dictionary format so malformed IR can be rejected with E008.

Diagnostic conventions used here:

* ``E003`` – API misuse (wrong node type or malformed argument);
* ``E005`` – a cyclic CFG where an acyclic graph was required;
* ``E006`` – a referenced basic block does not exist;
* ``E007`` – an operand kind cannot be represented by the model IR format;
* ``E008`` – a serialized IR dictionary is malformed.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

from compiler.frontend.errors import PolyglotError
from compiler.ir.framework import (
    BasicBlock,
    Cfg,
    HirParam,
    HirType,
    MirFunction,
    MirInstr,
    MirModule,
    MirOperand,
    MirTerminator,
)

__all__ = [
    "mir_block_index",
    "mir_cfg_fingerprint",
    "mir_cfg_summary",
    "mir_clone_cfg",
    "mir_deserialize",
    "mir_detect_cycles",
    "mir_dominators",
    "mir_empty_blocks",
    "mir_find_instr",
    "mir_instr_histogram",
    "mir_liveness",
    "mir_loop_headers",
    "mir_merge_blocks",
    "mir_operand_kinds",
    "mir_operand_types",
    "mir_phi_sources",
    "mir_predecessors",
    "mir_reachable",
    "mir_reads_writes",
    "mir_rename_block",
    "mir_serialize",
    "mir_split_block",
    "mir_successors",
    "mir_terminator_arity",
    "mir_topological_order",
]

FORMAT = "polyglot-mir-1"
_TERMINATOR_ARITY: dict[str, tuple[int, int | None]] = {
    "jump": (1, 1),
    "branch": (2, 2),
    "switch": (1, None),
    "return": (0, 0),
    "unreachable": (0, 0),
}
_OPERAND_KINDS = frozenset({"var", "const", "unit"})


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


def _require_module(module: object) -> MirModule:
    """Return *module* or raise E003 when it is not a MIR module."""
    if not isinstance(module, MirModule):
        _fail("E003", "expected a MirModule", value=type(module).__name__)
    return module


def _require_cfg(cfg: object) -> Cfg:
    """Return *cfg* or raise E003 when it is not a control-flow graph."""
    if not isinstance(cfg, Cfg):
        _fail("E003", "expected a Cfg", value=type(cfg).__name__)
    return cfg


def _require_block(block: object) -> BasicBlock:
    """Return *block* or raise E003 when it is not a basic block."""
    if not isinstance(block, BasicBlock):
        _fail("E003", "expected a BasicBlock", value=type(block).__name__)
    return block


def _require_operand(operand: object) -> MirOperand:
    """Return *operand* or raise E003 when it is not a MIR operand."""
    if not isinstance(operand, MirOperand):
        _fail("E003", "expected a MirOperand", value=type(operand).__name__)
    return operand


def _module_from(target: MirModule | MirFunction | Cfg) -> MirModule:
    """Normalize a supported MIR target into a module."""
    if isinstance(target, MirModule):
        return target
    if isinstance(target, MirFunction):
        return MirModule({target.name: target}, "")
    if isinstance(target, Cfg):
        return MirModule({"<cfg>": MirFunction("<cfg>", target)}, "")
    _fail("E003", "unsupported MIR target", value=type(target).__name__)


def _copy_operand(operand: MirOperand) -> MirOperand:
    """Copy one immutable operand."""
    return MirOperand(operand.kind, operand.value, operand.ty)


def _copy_instr(instr: MirInstr) -> MirInstr:
    """Copy one instruction including its metadata."""
    _require_instr(instr)
    return MirInstr(
        instr.op,
        instr.dest,
        tuple(_copy_operand(arg) for arg in instr.args),
        dict(instr.meta),
        instr.span,
    )


def _require_instr(instr: object) -> MirInstr:
    """Return *instr* or raise E003 for a non-instruction value."""
    if not isinstance(instr, MirInstr):
        _fail("E003", "expected a MirInstr", value=type(instr).__name__)
    return instr


def _copy_terminator(terminator: MirTerminator) -> MirTerminator:
    """Copy one terminator including its metadata."""
    if not isinstance(terminator, MirTerminator):
        _fail("E003", "expected a MirTerminator")
    return MirTerminator(
        terminator.kind,
        tuple(terminator.targets),
        None if terminator.cond is None else _copy_operand(terminator.cond),
        None if terminator.value is None else _copy_operand(terminator.value),
        dict(terminator.meta),
    )


def _copy_block(block: BasicBlock) -> BasicBlock:
    """Copy one basic block including all instructions and its terminator."""
    root = _require_block(block)
    return BasicBlock(
        root.label,
        [_copy_instr(instr) for instr in root.instructions],
        _copy_terminator(root.terminator),
        dict(root.meta),
    )


def _copy_cfg(cfg: Cfg) -> Cfg:
    """Copy a control-flow graph without sharing mutable block state."""
    root = _require_cfg(cfg)
    return Cfg(
        root.entry, {label: _copy_block(block) for label, block in root.blocks.items()}
    )


def _successors(cfg: Cfg, label: str) -> tuple[str, ...]:
    """Return terminator targets for a known block."""
    block = cfg.blocks.get(label)
    return () if block is None else tuple(block.terminator.targets)


def _edges(cfg: Cfg) -> dict[str, tuple[str, ...]]:
    """Return successor lists for every block, sorted by label."""
    return {label: _successors(cfg, label) for label in sorted(cfg.blocks)}


def _predecessors(cfg: Cfg) -> dict[str, tuple[str, ...]]:
    """Return predecessor lists for every block, sorted by label."""
    result: dict[str, list[str]] = {label: [] for label in sorted(cfg.blocks)}
    for label in sorted(cfg.blocks):
        for target in _successors(cfg, label):
            if target in result:
                result[target].append(label)
    return {label: tuple(sorted(names)) for label, names in result.items()}


def _reachable(cfg: Cfg) -> set[str]:
    """Compute the set of labels reachable from the configured entry."""
    if cfg.entry not in cfg.blocks:
        return set()
    seen: set[str] = set()
    queue: deque[str] = deque([cfg.entry])
    while queue:
        label = queue.popleft()
        if label in seen:
            continue
        seen.add(label)
        queue.extend(
            target for target in _successors(cfg, label) if target in cfg.blocks
        )
    return seen


def _dominators(cfg: Cfg) -> dict[str, set[str]]:
    """Compute dominator sets for reachable blocks."""
    reachable = _reachable(cfg)
    if not reachable:
        return {}
    predecessors = _predecessors(cfg)
    result: dict[str, set[str]] = {label: set(reachable) for label in reachable}
    result[cfg.entry] = {cfg.entry}
    changed = True
    while changed:
        changed = False
        for label in sorted(reachable):
            if label == cfg.entry:
                continue
            common = set(reachable)
            for pred in predecessors[label]:
                if pred in reachable:
                    common &= result.get(pred, set())
            candidate = {label, *common}
            if candidate != result[label]:
                result[label] = candidate
                changed = True
    return result


def _type_record(ty: HirType) -> dict[str, Any]:
    """Encode a HirType as a plain, versioned dictionary."""
    if not isinstance(ty, HirType):
        _fail("E008", "serialized type must be a HirType")
    return {
        "__type__": "hir",
        "name": ty.name,
        "args": [_type_record(arg) for arg in ty.args],
        "mutable": bool(ty.mutable),
        "referenced": bool(ty.referenced),
    }


def _type_from_record(value: object) -> HirType:
    """Decode a versioned type dictionary into a HirType."""
    if not isinstance(value, dict) or value.get("__type__") != "hir":
        _fail("E008", "malformed serialized type")
    name = value.get("name")
    args = value.get("args")
    if not isinstance(name, str) or not name or not isinstance(args, list):
        _fail("E008", "malformed serialized type fields")
    return HirType(
        name,
        tuple(_type_from_record(arg) for arg in args),
        bool(value.get("mutable", False)),
        bool(value.get("referenced", False)),
    )


def _jsonable(value: Any) -> Any:
    """Convert compiler objects inside metadata into plain serializable values."""
    if isinstance(value, HirType):
        return _type_record(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__repr__": repr(value)}


def _unjsonable(value: Any) -> Any:
    """Restore type records from a serialized metadata value."""
    if isinstance(value, list):
        return tuple(_unjsonable(item) for item in value)
    if isinstance(value, dict):
        if value.get("__type__") == "hir":
            return _type_from_record(value)
        return {key: _unjsonable(item) for key, item in value.items()}
    return value


def _operand_record(operand: MirOperand) -> dict[str, Any]:
    """Encode one operand in the versioned MIR format."""
    root = _require_operand(operand)
    if root.kind not in _OPERAND_KINDS:
        _fail("E007", "unsupported operand kind for serialization", kind=root.kind)
    return {
        "kind": root.kind,
        "value": _jsonable(root.value),
        "type": _type_record(root.ty),
    }


def _operand_from_record(value: object) -> MirOperand:
    """Decode one operand record, rejecting malformed serialized IR."""
    if not isinstance(value, dict):
        _fail("E008", "malformed serialized operand")
    kind = value.get("kind")
    if kind not in _OPERAND_KINDS or "value" not in value or "type" not in value:
        _fail("E008", "malformed serialized operand fields")
    payload = _unjsonable(value["value"])
    if (
        isinstance(payload, tuple)
        and len(payload) == 1
        and isinstance(payload[0], dict)
    ):
        payload = payload[0].get("__repr__", payload[0])
    return MirOperand(str(kind), payload, _type_from_record(value["type"]))


# ---------------------------------------------------------------------------
# Public MIR helpers
# ---------------------------------------------------------------------------


def mir_block_index(module: MirModule) -> dict[str, Any]:
    """Index every basic block by its instruction, terminator, and edge data."""
    root = _require_module(module)
    functions: dict[str, Any] = {}
    total = 0
    for name, function in root.functions.items():
        cfg = function.cfg
        predecessors = _predecessors(cfg)
        blocks: dict[str, Any] = {}
        for label in sorted(cfg.blocks):
            block = cfg.blocks[label]
            blocks[label] = {
                "instructions": len(block.instructions),
                "terminator": block.terminator.kind,
                "targets": tuple(block.terminator.targets),
                "successors": _successors(cfg, label),
                "predecessors": predecessors.get(label, ()),
            }
        functions[name] = {"entry": cfg.entry, "blocks": blocks}
        total += len(blocks)
    return {"functions": functions, "total_blocks": total}


def mir_cfg_fingerprint(module: MirModule) -> dict[str, Any]:
    """Hash the canonical serialized form of a MIR module for parity checks."""
    root = _require_module(module)
    payload = mir_serialize(root)
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return {
        "digest": digest,
        "functions": tuple(sorted(root.functions)),
        "blocks": sum(len(fn.cfg.blocks) for fn in root.functions.values()),
    }


def mir_cfg_summary(target: MirModule | MirFunction | Cfg) -> dict[str, Any]:
    """Summarize CFG size and terminator distribution for a MIR target."""
    module = _module_from(target)
    summaries: dict[str, Any] = {}
    for name, function in module.functions.items():
        cfg = function.cfg
        terminators: dict[str, int] = {}
        instructions = 0
        edges = 0
        for label in sorted(cfg.blocks):
            block = cfg.blocks[label]
            instructions += len(block.instructions)
            terminators[block.terminator.kind] = (
                terminators.get(block.terminator.kind, 0) + 1
            )
            edges += len(block.terminator.targets)
        summaries[name] = {
            "entry": cfg.entry,
            "blocks": len(cfg.blocks),
            "instructions": instructions,
            "edges": edges,
            "branches": terminators.get("branch", 0),
            "terminators": {kind: terminators[kind] for kind in sorted(terminators)},
        }
    return {
        "functions": summaries,
        "total_blocks": sum(item["blocks"] for item in summaries.values()),
        "total_instructions": sum(item["instructions"] for item in summaries.values()),
        "total_edges": sum(item["edges"] for item in summaries.values()),
    }


def mir_successors(cfg: Cfg, label: str) -> dict[str, Any]:
    """Return the successors of one block, raising E006 for a missing label."""
    root = _require_cfg(cfg)
    if not isinstance(label, str):
        _fail("E003", "block label must be a str")
    if label not in root.blocks:
        _fail("E006", "basic block is missing", label=label)
    targets = _successors(root, label)
    block = root.blocks[label]
    return {
        "label": label,
        "successors": targets,
        "terminator": block.terminator.kind,
        "existing": tuple(target for target in targets if target in root.blocks),
        "dangling": tuple(target for target in targets if target not in root.blocks),
    }


def mir_predecessors(cfg: Cfg) -> dict[str, Any]:
    """Compute predecessor lists for all blocks in a CFG."""
    root = _require_cfg(cfg)
    predecessors = _predecessors(root)
    return {
        "predecessors": predecessors,
        "roots": tuple(label for label, parents in predecessors.items() if not parents),
        "blocks": len(root.blocks),
    }


def mir_reachable(cfg: Cfg) -> dict[str, Any]:
    """Partition CFG blocks into reachable and unreachable label sets."""
    root = _require_cfg(cfg)
    reachable = _reachable(root)
    return {
        "entry": root.entry,
        "reachable": tuple(sorted(reachable)),
        "unreachable": tuple(sorted(set(root.blocks) - reachable)),
        "count": len(reachable),
    }


def mir_dominators(cfg: Cfg) -> dict[str, Any]:
    """Compute deterministic dominator sets for all reachable blocks."""
    root = _require_cfg(cfg)
    dominators = _dominators(root)
    return {
        "entry": root.entry,
        "dominators": {
            label: tuple(sorted(dominators[label])) for label in sorted(dominators)
        },
        "count": len(dominators),
    }


def mir_empty_blocks(module: MirModule) -> dict[str, Any]:
    """List blocks with no instructions, separating entry and terminal-only blocks."""
    root = _require_module(module)
    empty: list[str] = []
    terminal_only: list[str] = []
    for name, function in root.functions.items():
        for label, block in function.cfg.blocks.items():
            if block.instructions:
                continue
            key = f"{name}:{label}"
            empty.append(key)
            if block.terminator.kind in {"return", "unreachable"}:
                terminal_only.append(key)
    return {
        "empty": tuple(sorted(empty)),
        "terminal_only": tuple(sorted(terminal_only)),
        "count": len(empty),
    }


def mir_loop_headers(cfg: Cfg) -> dict[str, Any]:
    """Find natural loop headers using dominator back edges."""
    root = _require_cfg(cfg)
    dominators = _dominators(root)
    back_edges: list[tuple[str, str]] = []
    for source in sorted(_reachable(root)):
        for target in _successors(root, source):
            if target in dominators.get(source, set()):
                back_edges.append((source, target))
    bodies: dict[str, set[str]] = {target: {target} for _, target in back_edges}
    for source, target in back_edges:
        bodies[target].add(source)
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for pred in _predecessors(root).get(node, ()):
                if pred in dominators and pred not in bodies[target]:
                    bodies[target].add(pred)
                    queue.append(pred)
    return {
        "headers": tuple(sorted(bodies)),
        "back_edges": tuple(sorted(set(back_edges))),
        "bodies": {header: tuple(sorted(bodies[header])) for header in sorted(bodies)},
    }


def mir_topological_order(cfg: Cfg) -> dict[str, Any]:
    """Return a Kahn topological order, raising E005 when the CFG has a cycle."""
    root = _require_cfg(cfg)
    reachable = _reachable(root)
    predecessors = _predecessors(root)
    indegree = {
        label: sum(1 for pred in predecessors[label] if pred in reachable)
        for label in reachable
    }
    queue = deque(sorted(label for label, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        label = queue.popleft()
        order.append(label)
        for target in sorted(set(_successors(root, label))):
            if target not in reachable:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue = deque(sorted(queue))
    if len(order) != len(reachable):
        remaining = tuple(sorted(reachable - set(order)))
        _fail("E005", "control flow graph contains a cycle", blocks=remaining)
    return {"order": tuple(order), "acyclic": True, "entry": root.entry}


def mir_detect_cycles(cfg: Cfg) -> dict[str, Any]:
    """Report back edges and cyclic components found by a deterministic DFS."""
    root = _require_cfg(cfg)
    color: dict[str, int] = {}
    stack: list[str] = []
    back_edges: set[tuple[str, str]] = set()
    cycles: set[tuple[str, ...]] = set()

    def visit(label: str) -> None:
        color[label] = 1
        stack.append(label)
        for target in sorted(set(_successors(root, label))):
            if target not in root.blocks:
                continue
            if color.get(target, 0) == 0:
                visit(target)
            elif color.get(target) == 1:
                back_edges.add((label, target))
                cycles.add(tuple(sorted(stack[stack.index(target) :])))
        stack.pop()
        color[label] = 2

    for label in sorted(root.blocks):
        if color.get(label, 0) == 0:
            visit(label)
    return {
        "cycles": tuple(sorted(cycles)),
        "back_edges": tuple(sorted(back_edges)),
        "has_cycle": bool(cycles),
    }


def mir_operand_kinds(module: MirModule) -> dict[str, Any]:
    """Count variable, constant, and unit operands used across a MIR module."""
    root = _require_module(module)
    counts: dict[str, int] = {}
    for function in root.functions.values():
        for block in function.cfg.blocks.values():
            operands = [
                _copy_operand(arg) for instr in block.instructions for arg in instr.args
            ]
            for operand in (block.terminator.cond, block.terminator.value):
                if operand is not None:
                    operands.append(operand)
            for operand in operands:
                counts[operand.kind] = counts.get(operand.kind, 0) + 1
    return {
        "counts": {kind: counts[kind] for kind in sorted(counts)},
        "total": sum(counts.values()),
        "functions": tuple(sorted(root.functions)),
    }


def mir_reads_writes(module: MirModule) -> dict[str, Any]:
    """Compute per-block variable reads and writes from instructions and terminators."""
    root = _require_module(module)
    blocks: dict[str, dict[str, tuple[str, ...]]] = {}
    all_reads: set[str] = set()
    all_writes: set[str] = set()
    for name, function in root.functions.items():
        for label, block in function.cfg.blocks.items():
            reads: set[str] = set()
            writes: set[str] = set()
            for instr in block.instructions:
                for operand in instr.args:
                    if operand.kind == "var":
                        reads.add(str(operand.value))
                if instr.dest:
                    writes.add(instr.dest)
                if instr.op == "store" and isinstance(instr.meta.get("target"), str):
                    writes.add(instr.meta["target"])
            for operand in (block.terminator.cond, block.terminator.value):
                if operand is not None and operand.kind == "var":
                    reads.add(str(operand.value))
            blocks[f"{name}:{label}"] = {
                "reads": tuple(sorted(reads)),
                "writes": tuple(sorted(writes)),
            }
            all_reads |= reads
            all_writes |= writes
    return {
        "blocks": {key: blocks[key] for key in sorted(blocks)},
        "all_reads": tuple(sorted(all_reads)),
        "all_writes": tuple(sorted(all_writes)),
    }


def mir_liveness(cfg: Cfg) -> dict[str, Any]:
    """Compute live-in and live-out variable sets by iterative dataflow."""
    root = _require_cfg(cfg)
    reachable = _reachable(root)
    live_in = {label: set[str]() for label in reachable}
    live_out = {label: set[str]() for label in reachable}
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        for label in sorted(reachable, reverse=True):
            block = root.blocks[label]
            out: set[str] = set()
            for target in block.terminator.targets:
                out |= live_in.get(target, set())
            uses: set[str] = set()
            defs: set[str] = set()
            for instr in block.instructions:
                for operand in instr.args:
                    if operand.kind == "var":
                        uses.add(str(operand.value))
                if instr.op == "store" and isinstance(instr.meta.get("target"), str):
                    uses.add(instr.meta["target"])
                if instr.dest:
                    defs.add(instr.dest)
            for operand in (block.terminator.cond, block.terminator.value):
                if operand is not None and operand.kind == "var":
                    uses.add(str(operand.value))
            new_in = uses | (out - defs)
            if new_in != live_in[label] or out != live_out[label]:
                live_in[label] = new_in
                live_out[label] = out
                changed = True
    return {
        "live_in": {label: tuple(sorted(live_in[label])) for label in sorted(live_in)},
        "live_out": {
            label: tuple(sorted(live_out[label])) for label in sorted(live_out)
        },
        "iterations": iterations,
    }


def mir_phi_sources(module: MirModule) -> dict[str, Any]:
    """Collect phi instructions and their incoming value/block pairs."""
    root = _require_module(module)
    phis: list[dict[str, Any]] = []
    for name, function in root.functions.items():
        for label, block in function.cfg.blocks.items():
            for instr in block.instructions:
                if instr.op != "phi":
                    continue
                blocks = instr.meta.get("blocks", ())
                phis.append(
                    {
                        "function": name,
                        "block": label,
                        "dest": instr.dest,
                        "values": tuple(operand.text for operand in instr.args),
                        "blocks": tuple(str(item) for item in blocks),
                    }
                )
    return {"phis": tuple(phis), "count": len(phis)}


def mir_terminator_arity(target: MirModule | MirFunction | Cfg) -> dict[str, Any]:
    """Check terminator kinds and target counts across a MIR target."""
    module = _module_from(target)
    issues: list[dict[str, str]] = []
    checked = 0
    for name, function in module.functions.items():
        for label, block in function.cfg.blocks.items():
            terminator = block.terminator
            checked += 1
            bounds = _TERMINATOR_ARITY.get(terminator.kind)
            if bounds is None:
                issues.append(
                    {
                        "code": "E003",
                        "block": f"{name}:{label}",
                        "message": f"unknown terminator {terminator.kind!r}",
                    }
                )
                continue
            low, high = bounds
            count = len(terminator.targets)
            if count < low or (high is not None and count > high):
                issues.append(
                    {
                        "code": "E003",
                        "block": f"{name}:{label}",
                        "message": f"{terminator.kind} expects {low}..{high} targets, got {count}",
                    }
                )
            if terminator.kind == "branch" and terminator.cond is None:
                issues.append(
                    {
                        "code": "E001",
                        "block": f"{name}:{label}",
                        "message": "branch has no condition",
                    }
                )
    return {"valid": not issues, "issues": tuple(issues), "checked": checked}


def mir_serialize(module: MirModule) -> dict[str, Any]:
    """Serialize a MIR module into a deterministic, JSON-compatible dictionary."""
    root = _require_module(module)
    functions: dict[str, Any] = {}
    for name, function in root.functions.items():
        cfg = function.cfg
        blocks: dict[str, Any] = {}
        for label in sorted(cfg.blocks):
            block = cfg.blocks[label]
            instructions = [
                {
                    "op": instr.op,
                    "dest": instr.dest,
                    "args": [_operand_record(arg) for arg in instr.args],
                    "meta": _jsonable(instr.meta),
                    "span": _jsonable(instr.span),
                }
                for instr in block.instructions
            ]
            terminator = block.terminator
            blocks[label] = {
                "instructions": instructions,
                "terminator": {
                    "kind": terminator.kind,
                    "targets": list(terminator.targets),
                    "cond": None
                    if terminator.cond is None
                    else _operand_record(terminator.cond),
                    "value": None
                    if terminator.value is None
                    else _operand_record(terminator.value),
                    "meta": _jsonable(terminator.meta),
                },
                "meta": _jsonable(block.meta),
            }
        functions[name] = {
            "name": function.name,
            "entry": cfg.entry,
            "params": [
                {
                    "name": param.name,
                    "type": _type_record(param.ty),
                    "span": _jsonable(param.span),
                }
                for param in function.params
            ],
            "ret": _type_record(function.ret),
            "pure": bool(function.pure),
            "recursive": bool(function.recursive),
            "meta": _jsonable(function.meta),
            "blocks": blocks,
        }
    return {"format": FORMAT, "source": root.source, "functions": functions}


def mir_deserialize(data: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a MIR module from a serialized dictionary, validating its shape."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        _fail("E008", "malformed serialized MIR format")
    functions_data = data.get("functions")
    source = data.get("source", "")
    if not isinstance(functions_data, dict) or not isinstance(source, str):
        _fail("E008", "malformed serialized MIR module")
    functions: dict[str, MirFunction] = {}
    for name, payload in functions_data.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            _fail("E008", "malformed serialized function")
        blocks_data = payload.get("blocks")
        if not isinstance(blocks_data, dict):
            _fail("E008", "malformed serialized blocks")
        blocks: dict[str, BasicBlock] = {}
        for label, block_data in blocks_data.items():
            if not isinstance(label, str) or not isinstance(block_data, dict):
                _fail("E008", "malformed serialized block")
            instrs_data = block_data.get("instructions")
            term_data = block_data.get("terminator")
            if not isinstance(instrs_data, list):
                _fail("E008", "malformed serialized instructions")
            instructions: list[MirInstr] = []
            for item in instrs_data:
                if not isinstance(item, dict) or not isinstance(item.get("op"), str):
                    _fail("E008", "malformed serialized instruction")
                args = item.get("args")
                if not isinstance(args, list):
                    _fail("E008", "malformed serialized instruction operands")
                instructions.append(
                    MirInstr(
                        item["op"],
                        item.get("dest"),
                        tuple(_operand_from_record(arg) for arg in args),
                        _unjsonable(item.get("meta", {})),
                        _unjsonable(item.get("span")),
                    )
                )
            if not isinstance(term_data, dict) or not isinstance(
                term_data.get("kind"), str
            ):
                _fail("E008", "malformed serialized terminator")
            targets = term_data.get("targets")
            if not isinstance(targets, list) or not all(
                isinstance(t, str) for t in targets
            ):
                _fail("E008", "malformed serialized terminator targets")
            cond = term_data.get("cond")
            value = term_data.get("value")
            terminator = MirTerminator(
                term_data["kind"],
                tuple(targets),
                None if cond is None else _operand_from_record(cond),
                None if value is None else _operand_from_record(value),
                _unjsonable(term_data.get("meta", {})),
            )
            blocks[label] = BasicBlock(
                label, instructions, terminator, _unjsonable(block_data.get("meta", {}))
            )
        params_data = payload.get("params")
        ret_data = payload.get("ret")
        if not isinstance(params_data, list) or ret_data is None:
            _fail("E008", "malformed serialized signature")
        params: list[HirParam] = []
        for item in params_data:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                _fail("E008", "malformed serialized parameter")
            params.append(
                HirParam(
                    item["name"],
                    _type_from_record(item.get("type")),
                    _unjsonable(item.get("span")),
                )
            )
        entry = payload.get("entry")
        if not isinstance(entry, str) or entry not in blocks:
            _fail("E008", "serialized entry block is missing")
        functions[name] = MirFunction(
            name=name,
            cfg=Cfg(entry, blocks),
            params=tuple(params),
            ret=_type_from_record(ret_data),
            pure=bool(payload.get("pure", False)),
            recursive=bool(payload.get("recursive", False)),
            meta=_unjsonable(payload.get("meta", {})),
        )
    module = MirModule(functions, source)
    return {
        "module": module,
        "functions": tuple(sorted(module.functions)),
        "blocks": sum(len(fn.cfg.blocks) for fn in module.functions.values()),
    }


def mir_clone_cfg(cfg: Cfg) -> dict[str, Any]:
    """Deep-copy a CFG so later mutations cannot affect the source graph."""
    root = _require_cfg(cfg)
    clone = _copy_cfg(root)
    return {
        "cfg": clone,
        "entry": clone.entry,
        "blocks": len(clone.blocks),
        "instructions": sum(len(block.instructions) for block in clone.blocks.values()),
    }


def mir_rename_block(cfg: Cfg, old: str, new: str) -> dict[str, Any]:
    """Rename a basic block, updating the entry and every terminator target."""
    root = _require_cfg(cfg)
    if not isinstance(old, str) or not isinstance(new, str):
        _fail("E003", "block labels must be strings")
    if old not in root.blocks:
        _fail("E006", "block to rename is missing", label=old)
    if new in root.blocks and new != old:
        _fail("E003", "target block label already exists", label=new)
    clone = _copy_cfg(root)
    if new != old:
        clone.blocks[new] = clone.blocks.pop(old)
        clone.blocks[new].label = new
    updated = 0
    for block in clone.blocks.values():
        targets = tuple(
            new if target == old else target for target in block.terminator.targets
        )
        updated += sum(
            1
            for before, after in zip(block.terminator.targets, targets)
            if before != after
        )
        block.terminator.targets = targets
    if clone.entry == old:
        clone.entry = new
    return {
        "cfg": clone,
        "entry": clone.entry,
        "renamed": new != old,
        "updated_targets": updated,
    }


def mir_split_block(cfg: Cfg, label: str, index: int) -> dict[str, Any]:
    """Split a block at an instruction index, keeping the original terminator last."""
    root = _require_cfg(cfg)
    if not isinstance(index, int):
        _fail("E003", "split index must be an int")
    block = root.blocks.get(label)
    if block is None:
        _fail("E006", "block to split is missing", label=label)
    if not 0 <= index <= len(block.instructions):
        _fail("E003", "split index is out of range", index=index)
    clone = _copy_cfg(root)
    original = clone.blocks[label]
    new_label = f"{label}_split"
    suffix = 2
    while new_label in clone.blocks:
        new_label = f"{label}_split_{suffix}"
        suffix += 1
    tail = BasicBlock(
        new_label,
        original.instructions[index:],
        original.terminator,
        dict(original.meta),
    )
    original.instructions = original.instructions[:index]
    original.terminator = MirTerminator("jump", (new_label,))
    clone.blocks[new_label] = tail
    return {
        "cfg": clone,
        "first": label,
        "second": new_label,
        "first_instructions": len(original.instructions),
        "second_instructions": len(tail.instructions),
    }


def mir_merge_blocks(cfg: Cfg, label: str) -> dict[str, Any]:
    """Merge a jump-only block into its unique successor when the CFG allows it."""
    root = _require_cfg(cfg)
    block = root.blocks.get(label)
    if block is None:
        _fail("E006", "block to merge is missing", label=label)
    if label == root.entry or block.instructions:
        _fail("E003", "block is not mergeable", label=label)
    terminator = block.terminator
    if terminator.kind != "jump" or len(terminator.targets) != 1:
        _fail("E003", "block does not end in a single jump", label=label)
    target = terminator.targets[0]
    predecessors = _predecessors(root)
    if target not in root.blocks or predecessors.get(target, ()) != (label,):
        _fail("E003", "target is not uniquely owned by the block", target=target)
    clone = _copy_cfg(root)
    successor = clone.blocks[target]
    successor.instructions = block.instructions + successor.instructions
    del clone.blocks[label]
    redirected: list[str] = []
    for current in clone.blocks.values():
        if label in current.terminator.targets:
            current.terminator.targets = tuple(
                target if item == label else item for item in current.terminator.targets
            )
            redirected.append(current.label)
    return {
        "cfg": clone,
        "merged_from": label,
        "merged_into": target,
        "instructions": len(successor.instructions),
        "updated_targets": tuple(sorted(redirected)),
    }


def mir_operand_types(module: MirModule) -> dict[str, Any]:
    """Histogram operand type spellings and list types without shared backend support."""
    root = _require_module(module)
    counts: dict[str, int] = {}
    unsupported: set[str] = set()
    for function in root.functions.values():
        for block in function.cfg.blocks.values():
            operands = [arg for instr in block.instructions for arg in instr.args]
            for operand in (block.terminator.cond, block.terminator.value):
                if operand is not None:
                    operands.append(operand)
            for operand in operands:
                ty = operand.ty
                if not isinstance(ty, HirType):
                    _fail("E001", "operand has a non-HirType type")
                counts[ty.text] = counts.get(ty.text, 0) + 1
                if not ty.backend_supported:
                    unsupported.add(ty.text)
    return {
        "types": {name: counts[name] for name in sorted(counts)},
        "unsupported": tuple(sorted(unsupported)),
        "count": sum(counts.values()),
    }


def mir_instr_histogram(module: MirModule) -> dict[str, Any]:
    """Count instruction opcodes across all functions in sorted order."""
    root = _require_module(module)
    counts: dict[str, int] = {}
    for function in root.functions.values():
        for block in function.cfg.blocks.values():
            for instr in block.instructions:
                counts[instr.op] = counts.get(instr.op, 0) + 1
    return {
        "counts": {op: counts[op] for op in sorted(counts)},
        "total": sum(counts.values()),
    }


def mir_find_instr(
    module: MirModule,
    *,
    op: str | None = None,
    dest: str | None = None,
    callee: str | None = None,
) -> dict[str, Any]:
    """Find instructions matching optional opcode, destination, and callee filters."""
    root = _require_module(module)
    for name, value in (("op", op), ("dest", dest), ("callee", callee)):
        if value is not None and not isinstance(value, str):
            _fail("E003", f"{name} filter must be a str")
    matches: list[dict[str, Any]] = []
    for name, function in root.functions.items():
        for label, block in function.cfg.blocks.items():
            for index, instr in enumerate(block.instructions):
                if op is not None and instr.op != op:
                    continue
                if dest is not None and instr.dest != dest:
                    continue
                if callee is not None and instr.meta.get("callee") != callee:
                    continue
                matches.append(
                    {
                        "function": name,
                        "block": label,
                        "index": index,
                        "op": instr.op,
                        "dest": instr.dest,
                        "callee": instr.meta.get("callee"),
                        "span": instr.span,
                    }
                )
    return {"matches": tuple(matches), "count": len(matches)}
