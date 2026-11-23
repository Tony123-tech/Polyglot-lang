"""A virtual clock and a reproducible scheduler.

The scheduler never reads the wall clock: an operation's ``at_ms`` is its
logical time, and ties inside the same millisecond are broken by a key derived
from ``(seed, at_ms, seq, node)``. Running the same operation list with the
same seed therefore produces byte-identical traces, which is what makes a
one-in-a-million interleaving reproducible from a single number.

Bounded model (SPEC 8.18, 10.48): operations are replayed in sorted order and
the clock only moves forward. A negative timestamp or an operation scheduled
before the clock's current instant is rejected with ``E001``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from runtime.diagnostics import PillarError


@dataclass(frozen=True)
class SimOp:
    """One simulated operation."""

    at_ms: int
    node: str
    kind: str
    payload: Any = None
    seq: int = 0

    def __post_init__(self) -> None:
        if self.at_ms < 0:
            raise PillarError(
                "E001",
                f"operation scheduled at negative time: {self.at_ms} ms",
                help_text="times are relative to the start of the simulation",
            )


@dataclass(frozen=True)
class Step:
    """An operation as it was executed, with the clock it ran at."""

    now_ms: int
    node: str
    kind: str
    payload: Any
    seq: int


@dataclass(frozen=True)
class Schedule:
    """The executed trace of a run."""

    seed: int
    steps: tuple[Step, ...]
    elapsed_ms: int = 0
    start_ms: int = 0


class VirtualClock:
    """A monotonic clock that only advances when told to."""

    def __init__(self, start_ms: int = 0) -> None:
        if start_ms < 0:
            raise PillarError("E001", "virtual clock cannot start before zero")
        self.start_ms = start_ms
        self.now_ms = start_ms

    def advance_to(self, at_ms: int) -> int:
        if at_ms < self.now_ms:
            raise PillarError(
                "E001",
                f"attempt to move the virtual clock backwards: {at_ms} < {self.now_ms}",
                reason="a simulated clock is monotonic; reordering belongs in the scheduler",
                help_text="schedule the operation at a time at or after the current instant",
            )
        self.now_ms = at_ms
        return self.now_ms

    @property
    def elapsed_ms(self) -> int:
        return self.now_ms - self.start_ms

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"VirtualClock(now_ms={self.now_ms})"


def _tie_key(seed: int, op: SimOp) -> str:
    return hashlib.sha256(
        f"{seed}:{op.at_ms}:{op.seq}:{op.node}:{op.kind}".encode()
    ).hexdigest()


def dst_virtualize_clock_and_scheduler(
    ops: Iterable[SimOp],
    *,
    seed: int,
    start_ms: int = 0,
) -> Schedule:
    """Order operations deterministically and replay them on a virtual clock.

    Same ``(ops, seed)`` always yields the same :class:`Schedule`. Different
    seeds only change the order of operations that share a timestamp, so a
    timing-sensitive bug can be bisected by varying one integer.
    """
    ordered = list(ops)
    if not ordered:
        return Schedule(seed=seed, steps=(), start_ms=start_ms)

    for op in ordered:
        if op.at_ms < start_ms:
            raise PillarError(
                "E001",
                f"operation at {op.at_ms} ms is before the simulation start {start_ms} ms",
                help_text="raise start_ms or move the operation later",
            )

    clock = VirtualClock(start_ms)
    keyed = sorted(
        ((op.at_ms, _tie_key(seed, op), index, op) for index, op in enumerate(ordered)),
    )
    steps: list[Step] = []
    for at_ms, _key, _index, op in keyed:
        now = clock.advance_to(at_ms)
        steps.append(
            Step(now_ms=now, node=op.node, kind=op.kind, payload=op.payload, seq=op.seq)
        )
    return Schedule(
        seed=seed,
        steps=tuple(steps),
        elapsed_ms=clock.elapsed_ms,
        start_ms=start_ms,
    )
