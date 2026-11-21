"""Hard hot-patch concurrency/rollback edge cases (Frame C, runtime half).

These deterministic models complement the real mechanisms in
:mod:`runtime.hotpatch.framework`, :mod:`runtime.hotpatch.rcu`, and
:mod:`runtime.hotpatch.reactor`.  They assert RCU grace-period discipline,
zero-loss descriptor migration, raw-address TOCTOU re-validation, and atomic
rollback on failed verification.  All failures are structured ``E008``
diagnostics; no privileged memory writes occur here.

Diagnostic map: ``E008`` for any failed invariant, ``E007`` for a platform or
capability boundary that cannot be modelled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from runtime.diagnostics import PillarError

Span = tuple[int, int] | None


def _fail(
    code: str, message: str, *, span: Span = None, reason: str | None = None
) -> PillarError:
    return PillarError(
        code,
        message,
        reason=reason or message,
        help_text="drain readers, re-check the address, or verify the patch",
        note="runtime.hotpatch/hard",
        span=span,
    )


@dataclass
class RcuHardReport:
    """Deterministic RCU event replay result."""

    reclaimed: tuple[str, ...]
    deferred: tuple[str, ...]
    epochs: tuple[int, ...]
    violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "reclaimed": list(self.reclaimed),
            "deferred": list(self.deferred),
            "epochs": list(self.epochs),
            "violations": list(self.violations),
            "ok": not self.violations,
        }


def hp_rcu_reader_writer_invariants(
    events: Sequence[Mapping[str, Any]], *, span: Span = None
) -> RcuHardReport:
    """Replay RCU transitions and reject reclamation while readers are active."""
    readers: dict[str, int] = {}
    retired: list[tuple[str, int]] = []
    reclaimed: list[str] = []
    deferred: list[str] = []
    epochs: list[int] = []
    violations: list[str] = []
    epoch = 0
    for index, event in enumerate(events):
        operation = str(event.get("op", ""))
        reader = str(event.get("reader", ""))
        if operation == "enter":
            if reader in readers:
                violations.append(f"event {index}: reader {reader!r} entered twice")
            readers[reader] = epoch
        elif operation == "exit":
            if reader not in readers:
                violations.append(f"event {index}: reader {reader!r} never entered")
            else:
                del readers[reader]
        elif operation == "retire":
            retired.append((str(event.get("item", f"item-{index}")), epoch))
        elif operation == "advance":
            epoch += 1
        elif operation == "reclaim":
            if readers:
                violations.append(
                    f"event {index}: reclaim attempted with active readers "
                    f"{sorted(readers)}"
                )
                continue
            keep: list[tuple[str, int]] = []
            for item, retired_epoch in retired:
                if retired_epoch < epoch:
                    reclaimed.append(item)
                else:
                    keep.append((item, retired_epoch))
            retired = keep
        else:
            violations.append(f"event {index}: unknown RCU operation {operation!r}")
        epochs.append(epoch)
    if violations:
        raise _fail(
            "E008",
            f"RCU invariant violation at {len(violations)} event(s)",
            span=span,
            reason=violations[0],
        )
    deferred = [item for item, _epoch in retired]
    return RcuHardReport(
        reclaimed=tuple(reclaimed),
        deferred=tuple(deferred),
        epochs=tuple(epochs),
    )


@dataclass
class ReactorHardReport:
    """Descriptor migration result with an explicit event ledger."""

    preserved: tuple[tuple[str, Any], ...]
    migrated: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "preserved": [list(item) for item in self.preserved],
            "migrated": list(self.migrated),
            "added": list(self.added),
            "removed": list(self.removed),
        }


def hp_reactor_migrate_without_loss(
    before: Sequence[str],
    after: Sequence[str],
    *,
    events: Sequence[tuple[str, Any]],
    span: Span = None,
) -> ReactorHardReport:
    """Verify descriptor migration preserves every registration and event."""
    before_set, after_set = set(before), set(after)
    duplicates = sorted({name for name in after if after.count(name) > 1})
    if duplicates:
        raise _fail(
            "E008",
            f"duplicate reactor registrations: {', '.join(duplicates)}",
            span=span,
        )
    lost = sorted(before_set - after_set)
    if lost:
        raise _fail(
            "E008",
            f"reactor migration lost descriptors: {', '.join(lost)}",
            span=span,
        )
    orphan_events = sorted({name for name, _payload in events} - after_set)
    if orphan_events:
        raise _fail(
            "E008",
            f"events reference unregistered descriptors: {', '.join(orphan_events)}",
            span=span,
        )
    return ReactorHardReport(
        preserved=tuple(events),
        migrated=tuple(sorted(before_set & after_set)),
        added=tuple(sorted(after_set - before_set)),
        removed=(),
    )


def hp_revalidate_raw_address(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    expected_digest: str,
    span: Span = None,
) -> dict[str, Any]:
    """Re-check raw-address snapshots before and after a privileged write."""
    import hashlib
    import json

    fingerprints: list[str] = []
    for snapshot in snapshots:
        try:
            address = int(snapshot["address"])
            value = int(snapshot["value"])
            version = int(snapshot["version"])
        except (KeyError, TypeError, ValueError) as error:
            raise _fail(
                "E008", "malformed raw-address snapshot", span=span, reason=str(error)
            ) from error
        if address < 0 or version < 0:
            raise _fail("E008", "raw-address snapshot has a negative field", span=span)
        payload = json.dumps(
            {"address": address, "value": value, "version": version},
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprints.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    if any(fingerprint != expected_digest for fingerprint in fingerprints):
        raise _fail(
            "E008",
            "raw-address TOCTOU re-validation failed: snapshot changed",
            span=span,
            reason="expected digest does not match at least one observation",
        )
    return {"ok": True, "checked": len(fingerprints), "digest": expected_digest}


def hp_rollback_verified(
    steps: Sequence[Mapping[str, Any]],
    *,
    verifier: Callable[[Mapping[str, Any]], bool],
    span: Span = None,
) -> dict[str, Any]:
    """Apply steps atomically, rolling back in reverse if verification fails."""
    applied: list[Mapping[str, Any]] = []
    try:
        for index, step in enumerate(steps):
            if not callable(step.get("apply")) or not callable(step.get("undo")):
                raise _fail(
                    "E008",
                    f"patch step {index} needs apply and undo callables",
                    span=span,
                )
            step["apply"]()
            applied.append(step)
        if not verifier({"steps": applied}):
            raise _fail(
                "E008", "patch verification failed", span=span, reason="atomic rollback"
            )
    except PillarError:
        for step in reversed(applied):
            step["undo"]()
        raise
    except Exception as error:
        for step in reversed(applied):
            step["undo"]()
        raise _fail(
            "E008",
            f"patch application failed: {error}",
            span=span,
            reason=str(error),
        ) from error
    return {"ok": True, "applied": len(applied), "rolled_back": 0}


__all__ = [
    "RcuHardReport",
    "ReactorHardReport",
    "hp_rcu_reader_writer_invariants",
    "hp_reactor_migrate_without_loss",
    "hp_revalidate_raw_address",
    "hp_rollback_verified",
]
