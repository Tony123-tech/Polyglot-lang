"""Shrink a failing schedule to a minimal reproducer.

This is delta debugging over the operation list: repeatedly try to remove
chunks while the failure oracle still reports the bug. The oracle is any
deterministic predicate ``ops -> bool`` that is ``True`` when the bug is
present, so the minimiser does not need to know what the bug is.

Bounded model (SPEC 8.18, 10.48): the result is minimal with respect to
chunk removal, not provably 1-minimal, and the probe budget is hard. Exhausting
it raises ``E002`` rather than looping.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from runtime.diagnostics import PillarError

from .clock import SimOp


@dataclass(frozen=True)
class BisectReport:
    """The minimised schedule and how much work it took to find it."""

    failed: bool
    minimal: tuple[SimOp, ...]
    original_length: int
    probes: int
    budget_exhausted: bool = False

    @property
    def shrink_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return len(self.minimal) / self.original_length


def dst_bisect_and_minimize_schedule(
    ops: Iterable[SimOp],
    oracle: Callable[[Sequence[SimOp]], bool],
    *,
    max_probes: int = 2_000,
) -> BisectReport:
    """Remove as many operations as possible while ``oracle`` still fails."""
    if max_probes <= 0:
        raise PillarError(
            "E003",
            "bisect probe budget must be positive",
            help_text="pass max_probes >= 1",
        )

    current = list(ops)
    probes = 1
    if not oracle(current):
        return BisectReport(
            failed=False,
            minimal=tuple(current),
            original_length=len(current),
            probes=probes,
        )

    original_length = len(current)
    chunks = 2
    while len(current) >= 2:
        if probes >= max_probes:
            return BisectReport(
                failed=True,
                minimal=tuple(current),
                original_length=original_length,
                probes=probes,
                budget_exhausted=True,
            )
        size = max(1, len(current) // chunks)
        reduced = False
        for start in range(0, len(current), size):
            candidate = current[:start] + current[start + size :]
            if not candidate:
                continue
            probes += 1
            if oracle(candidate):
                current = candidate
                chunks = max(chunks - 1, 2)
                reduced = True
                break
        if not reduced:
            if chunks >= len(current):
                break
            chunks = min(len(current), chunks * 2)

    return BisectReport(
        failed=True,
        minimal=tuple(current),
        original_length=original_length,
        probes=probes,
    )
