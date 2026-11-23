"""Seeded fault injection: byzantine messages and partial I/O.

Every decision is drawn from ``random.Random(seed)`` in a fixed order, so the
fault set is a pure function of ``(ops, seed, rates)``. The report names how
many faults of each kind were injected, which is what a failing seed needs in
order to be explained later.

Bounded model (SPEC 8.18, 10.48): faults apply to the operation list only.
``torn_write`` truncates a write payload, ``lost_fsync`` drops an fsync after a
write, ``bitflip`` corrupts one byte of a write payload, ``drop`` removes an
operation, ``duplicate`` replays it, and ``reorder`` swaps it with its
successor.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from runtime.diagnostics import PillarError

from .clock import SimOp

FAULTS = ("drop", "duplicate", "reorder", "torn_write", "lost_fsync", "bitflip")

# Defaults are deliberately mild: a schedule should usually pass, so the
# interesting failure is rare and worth minimising.
DEFAULT_RATES: dict[str, float] = {
    "drop": 0.05,
    "duplicate": 0.05,
    "reorder": 0.05,
    "torn_write": 0.05,
    "lost_fsync": 0.05,
    "bitflip": 0.05,
}


@dataclass(frozen=True)
class FaultReport:
    """The faulted operation list and a count per injected fault."""

    seed: int
    ops: tuple[SimOp, ...]
    injected: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.injected.values())


def _rate_table(rates: dict[str, float] | None) -> dict[str, float]:
    table = dict(DEFAULT_RATES)
    if rates:
        unknown = sorted(set(rates) - set(FAULTS))
        if unknown:
            raise PillarError(
                "E003",
                f"unknown fault kind(s): {', '.join(unknown)}",
                help_text="supported faults: " + ", ".join(FAULTS),
            )
        table.update(rates)
    for kind, rate in table.items():
        if not 0.0 <= rate <= 1.0:
            raise PillarError(
                "E003",
                f"fault rate for {kind!r} must be in [0, 1], got {rate}",
                help_text="use a probability between 0.0 and 1.0",
            )
    return table


def _corrupt(payload: Any) -> Any:
    if isinstance(payload, bytes):
        if not payload:
            return b"\x00"
        return payload[:-1] + bytes([payload[-1] ^ 0x01])
    if isinstance(payload, str):
        if not payload:
            return "?"
        return payload[:-1] + chr(ord(payload[-1]) ^ 1)
    if isinstance(payload, list):
        return payload[:-1] + [_corrupt(payload[-1])] if payload else [0]
    return payload


def _truncate(payload: Any) -> Any:
    if isinstance(payload, (bytes, str, list)):
        return payload[: len(payload) // 2]
    return payload


def dst_inject_byzantine_and_io_faults(
    ops: Iterable[SimOp],
    *,
    seed: int,
    rates: dict[str, float] | None = None,
) -> FaultReport:
    """Return a faulted copy of ``ops`` for the given seed and rates."""
    table = _rate_table(rates)
    rng = random.Random(seed)
    source = list(ops)
    result: list[SimOp] = []
    injected = {kind: 0 for kind in FAULTS}
    pending_swap = False

    for op in source:
        # Draw every roll in a fixed order so the seed fully determines faults.
        rolls = {kind: rng.random() for kind in FAULTS}

        if op.kind == "fsync" and rolls["lost_fsync"] < table["lost_fsync"]:
            injected["lost_fsync"] += 1
            continue
        if rolls["drop"] < table["drop"]:
            injected["drop"] += 1
            continue

        payload = op.payload
        if op.kind == "write" and rolls["torn_write"] < table["torn_write"]:
            payload = _truncate(payload)
            injected["torn_write"] += 1
        elif op.kind == "write" and rolls["bitflip"] < table["bitflip"]:
            payload = _corrupt(payload)
            injected["bitflip"] += 1

        current = SimOp(
            at_ms=op.at_ms,
            node=op.node,
            kind=op.kind,
            payload=payload,
            seq=op.seq,
        )
        if pending_swap and result:
            result[-1], current = current, result[-1]
            pending_swap = False
        result.append(current)
        if rolls["reorder"] < table["reorder"]:
            pending_swap = True
            injected["reorder"] += 1
        if rolls["duplicate"] < table["duplicate"]:
            result.append(
                SimOp(
                    at_ms=op.at_ms,
                    node=op.node,
                    kind=op.kind,
                    payload=payload,
                    seq=op.seq + 1,
                )
            )
            injected["duplicate"] += 1

    return FaultReport(seed=seed, ops=tuple(result), injected=injected)
