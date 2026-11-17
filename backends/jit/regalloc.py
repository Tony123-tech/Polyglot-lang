"""Liveness analysis and a linear-scan register allocator for the JIT.

The allocator follows the AAPCS64 register roles:

* ``X0``-``X7`` / ``D0``-``D7``  incoming arguments and return values
* ``X9``-``X15`` / ``D8``-``D15`` caller-saved temporaries
* ``X19``-``X27`` callee-saved temporaries
* ``X28`` trap-context pointer (reserved)
* ``X29`` frame pointer, ``X30`` link register

Incoming arguments are pre-coloured by the IR builder.  The allocator assigns
temporaries from the callee-saved pool ``X19``-``X27`` / ``D8``-``D15`` so a
``BL`` never clobbers a live value; ``X16``/``X17`` and ``D16``/``D17`` are
scratch registers for materialising constants and resolving parallel copies.
Values whose live intervals exceed the register file are assigned stack spill
slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backends.jit.ir import (
    BasicBlock,
    BrCond,
    IRFunction,
    Phi,
    Ret,
    Value,
)

INT_CALLEE_SAVED = tuple(f"X{i}" for i in range(19, 28))
INT_CALLER_SAVED = tuple(f"X{i}" for i in range(9, 16))
FLOAT_CALLEE_SAVED = tuple(f"D{i}" for i in range(8, 16))
FLOAT_CALLER_SAVED = tuple(f"D{i}" for i in range(8))

INT_POOL = INT_CALLEE_SAVED + INT_CALLER_SAVED
FLOAT_POOL = FLOAT_CALLEE_SAVED + FLOAT_CALLER_SAVED

SCRATCH_INT = "X16"
SCRATCH_INT2 = "X17"
SCRATCH_FLOAT = "D16"
SCRATCH_FLOAT2 = "D17"


@dataclass
class LiveInterval:
    value: Value
    start: int
    end: int
    reg: str | None = None
    spill: int | None = None

    @property
    def is_float(self) -> bool:
        return self.value.ty.is_float


@dataclass
class Allocation:
    intervals: dict[Value, LiveInterval] = field(default_factory=dict)
    spill_count: int = 0
    block_order: list[BasicBlock] = field(default_factory=list)
    block_start: dict[BasicBlock, int] = field(default_factory=dict)
    block_end: dict[BasicBlock, int] = field(default_factory=dict)
    live_in: dict[BasicBlock, set[Value]] = field(default_factory=dict)
    live_out: dict[BasicBlock, set[Value]] = field(default_factory=dict)
    instruction_index: dict[int, int] = field(default_factory=dict)

    def reg(self, value: Value) -> str | None:
        interval = self.intervals.get(value)
        return interval.reg if interval is not None else value.fixed_reg

    def spill(self, value: Value) -> int | None:
        interval = self.intervals.get(value)
        return interval.spill if interval is not None else None

    def interval(self, value: Value) -> LiveInterval | None:
        return self.intervals.get(value)

    def is_spilled(self, value: Value) -> bool:
        interval = self.intervals.get(value)
        return interval is not None and interval.spill is not None


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _block_use_def(block: BasicBlock) -> tuple[set[Value], set[Value]]:
    uses: set[Value] = set()
    defs: set[Value] = set()
    for inst in block.insts:
        if isinstance(inst, Phi):
            defs.add(inst.result)
            continue
        for value in inst.uses():
            if value not in defs:
                uses.add(value)
        for value in inst.defs():
            defs.add(value)
    # Terminators also use values (branch conditions and return operands).
    terminator = block.terminator
    terminator_uses: tuple[Value, ...] = ()
    if isinstance(terminator, BrCond):
        terminator_uses = (terminator.cond,)
    elif isinstance(terminator, Ret) and terminator.value is not None:
        terminator_uses = (terminator.value,)
    for value in terminator_uses:
        if value not in defs:
            uses.add(value)
    return uses, defs


def compute_liveness(
    function: IRFunction,
) -> tuple[
    dict[BasicBlock, set[Value]],
    dict[BasicBlock, set[Value]],
]:
    blocks = list(function.blocks)
    uses: dict[BasicBlock, set[Value]] = {}
    defs: dict[BasicBlock, set[Value]] = {}
    for block in blocks:
        uses[block], defs[block] = _block_use_def(block)

    live_in: dict[BasicBlock, set[Value]] = {block: set() for block in blocks}
    live_out: dict[BasicBlock, set[Value]] = {block: set() for block in blocks}

    changed = True
    while changed:
        changed = False
        for block in reversed(blocks):
            out: set[Value] = set()
            for succ in function.successors(block):
                out |= live_in[succ]
                # Phi operands are used on the incoming edge, not inside the
                # successor block.
                for inst in succ.insts:
                    if isinstance(inst, Phi):
                        for pred, value in inst.incoming:
                            if pred is block:
                                out.add(value)
            new_in = uses[block] | (out - defs[block])
            if out != live_out[block] or new_in != live_in[block]:
                live_out[block] = out
                live_in[block] = new_in
                changed = True
    return live_in, live_out


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def compute_intervals(
    function: IRFunction,
    live_in: dict[BasicBlock, set[Value]],
    live_out: dict[BasicBlock, set[Value]],
) -> tuple[dict[Value, LiveInterval], dict[BasicBlock, int], dict[BasicBlock, int]]:
    intervals: dict[Value, LiveInterval] = {}
    block_start: dict[BasicBlock, int] = {}
    block_end: dict[BasicBlock, int] = {}
    index = 0
    for block in function.blocks:
        block_start[block] = index
        for inst in block.insts:
            for value in inst.defs():
                interval = intervals.get(value)
                if interval is None:
                    intervals[value] = LiveInterval(value, index, index)
                else:
                    interval.start = min(interval.start, index)
                    interval.end = max(interval.end, index)
            for value in inst.uses():
                interval = intervals.get(value)
                if interval is None:
                    intervals[value] = LiveInterval(value, index, index)
                else:
                    interval.end = max(interval.end, index)
            index += 1
        # The terminator/edge occupies the next slot so phi copies have a
        # well-defined live-out position.
        block_end[block] = index
        index += 1

    # Param values are defined before the entry block.
    for param in function.params:
        interval = intervals.get(param)
        if interval is None:
            intervals[param] = LiveInterval(param, 0, 0)
        else:
            interval.start = 0

    # Values live across blocks (including loop backedges) must cover the
    # whole region in which liveness says they are live.
    changed = True
    iterations = 0
    while changed and iterations < 64:
        changed = False
        iterations += 1
        for block in function.blocks:
            start = block_start[block]
            end = block_end[block]
            for value in live_in[block] | live_out[block]:
                interval = intervals.get(value)
                if interval is None:
                    intervals[value] = LiveInterval(value, start, end)
                    changed = True
                    continue
                new_start = min(interval.start, start)
                new_end = max(interval.end, end)
                if (new_start, new_end) != (interval.start, interval.end):
                    interval.start = new_start
                    interval.end = new_end
                    changed = True
    return intervals, block_start, block_end


# ---------------------------------------------------------------------------
# Linear scan
# ---------------------------------------------------------------------------


def _reg_sort_key(reg: str) -> tuple[int, int]:
    prefix = 0 if reg.startswith("X") else 1
    return prefix, int(reg[1:])


def allocate(function: IRFunction) -> Allocation:
    live_in, live_out = compute_liveness(function)
    intervals, block_start, block_end = compute_intervals(
        function,
        live_in,
        live_out,
    )

    result = Allocation(
        intervals=intervals,
        block_order=list(function.blocks),
        block_start=block_start,
        block_end=block_end,
        live_in=live_in,
        live_out=live_out,
    )

    # Pre-coloured argument values keep their ABI registers.
    for value in function.params:
        interval = intervals.get(value)
        if interval is not None and value.fixed_reg is not None:
            interval.reg = value.fixed_reg

    int_values = [
        interval
        for interval in intervals.values()
        if not interval.value.ty.is_float
        and interval.value.ty.name != "()"
        and interval.value.fixed_reg is None
    ]
    float_values = [
        interval
        for interval in intervals.values()
        if interval.value.ty.is_float and interval.value.fixed_reg is None
    ]

    spill_index = 0
    # Keep temporaries in callee-saved registers.  Caller-saved registers are
    # used for argument setup and scratch values, so values never need to be
    # spilled around a BL merely because a call clobbers them.
    spill_index = _linear_scan(int_values, INT_CALLEE_SAVED, spill_index)
    spill_index = _linear_scan(float_values, FLOAT_CALLEE_SAVED, spill_index)
    result.spill_count = spill_index
    return result


def _linear_scan(
    intervals: list[LiveInterval],
    pool: tuple[str, ...],
    spill_base: int,
) -> int:
    free = list(pool)
    active: list[LiveInterval] = []
    spill_index = spill_base

    for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
        # Expire everything that ended before this interval starts.
        still_active: list[LiveInterval] = []
        for current in active:
            if current.end < interval.start:
                if current.reg is not None:
                    free.append(current.reg)
                    free.sort(key=_reg_sort_key)
            else:
                still_active.append(current)
        active = still_active

        if free:
            interval.reg = free.pop(0)
            active.append(interval)
            continue

        # All registers are busy: evict the interval that ends last, unless
        # the new interval is the one that ends last.
        victim = max(active, key=lambda item: item.end)
        if victim.end > interval.end:
            interval.reg = victim.reg
            victim.reg = None
            victim.spill = spill_index
            spill_index += 1
            active.remove(victim)
            active.append(interval)
        else:
            interval.spill = spill_index
            spill_index += 1
    return spill_index


# ---------------------------------------------------------------------------
# Phi parallel copies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Location:
    """Where a value lives: a register, a spill slot, or nothing."""

    reg: str | None = None
    spill: int | None = None

    def is_reg(self) -> bool:
        return self.reg is not None

    def is_spill(self) -> bool:
        return self.spill is not None


def parallel_copy(
    moves: list[tuple[Location, Location]],
    scratch: Location,
) -> list[tuple[Location, Location]]:
    """Schedule ``(source, destination)`` pairs as sequential moves.

    The returned list is safe to emit in order.  Register cycles (``a -> b``,
    ``b -> a``) are broken by routing one source through ``scratch``.  Spill
    locations never participate in cycles because each spill slot is unique.
    """
    pending = [
        (source, destination)
        for source, destination in moves
        if source != destination
    ]
    scheduled: list[tuple[Location, Location]] = []

    while pending:
        sources = {source for source, _ in pending}
        ready = [
            index
            for index, (_, destination) in enumerate(pending)
            if destination not in sources
        ]
        if ready:
            # Emit in reverse index order so the pop() indices stay valid.
            for index in reversed(ready):
                scheduled.append(pending.pop(index))
            continue

        # Every remaining destination is also a source: we are in a cycle.
        source, _destination = pending[0]
        if source == scratch:
            raise ValueError("parallel copy cycle cannot be broken: source is scratch")
        scheduled.append((source, scratch))
        pending = [
            ((scratch if current == source else current), destination)
            for current, destination in pending
        ]

    return scheduled


__all__ = [
    "FLOAT_CALLEE_SAVED",
    "FLOAT_CALLER_SAVED",
    "FLOAT_POOL",
    "INT_CALLEE_SAVED",
    "INT_CALLER_SAVED",
    "INT_POOL",
    "SCRATCH_FLOAT",
    "SCRATCH_FLOAT2",
    "SCRATCH_INT",
    "SCRATCH_INT2",
    "Allocation",
    "LiveInterval",
    "Location",
    "allocate",
    "compute_intervals",
    "compute_liveness",
    "parallel_copy",
]
