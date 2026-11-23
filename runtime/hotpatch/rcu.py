"""Read-Copy-Update epochs for live AST roots.

The domain is a cooperative, user-space RCU implementation:

* readers enter a guard that snapshots the current epoch;
* a writer publishes a new immutable root and retires the old one;
* ``synchronize()`` waits until every reader that entered before the grace
  period has exited, then runs the retired reclamation callbacks.

CPython's GIL makes the root pointer swap atomic for Python objects.  The Rust
counterpart in ``runtime/hotpatch/native`` uses ``AtomicPtr`` with the same
epoch protocol.  No kernel RCU syscall is claimed.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import compiler.core.ast_nodes as ast
from runtime.hotpatch.errors import HotpatchError
from runtime.hotpatch.events import emit


@dataclass
class _ReaderSlot:
    depth: int = 0
    epoch: int = 0


@dataclass
class _Retired:
    epoch: int
    item: Any
    reclaim: Callable[[Any], None] | None
    span: tuple[int, int] | None


@dataclass(frozen=True)
class RcuStats:
    epoch: int
    active_readers: int
    retired: int
    reclaimed: int
    grace_periods: int
    reclaim_failures: int


class RcuDomain:
    """An epoch-based reclamation domain."""

    def __init__(self, name: str = "ast") -> None:
        self.name = name
        self._condition = threading.Condition(threading.RLock())
        self._writer_lock = threading.Lock()
        self._epoch = 0
        self._readers: dict[int, _ReaderSlot] = {}
        self._retired: list[_Retired] = []
        self._reclaimed = 0
        self._grace_periods = 0
        self._reclaim_failures = 0

    # -- reader side -------------------------------------------------------

    def enter(self) -> int:
        thread = threading.get_ident()
        with self._condition:
            slot = self._readers.get(thread)
            if slot is None:
                slot = _ReaderSlot(epoch=self._epoch)
                self._readers[thread] = slot
            if slot.depth == 0:
                slot.epoch = self._epoch
            slot.depth += 1
            return slot.epoch

    def exit(self) -> None:
        thread = threading.get_ident()
        with self._condition:
            slot = self._readers.get(thread)
            if slot is None:
                raise HotpatchError(
                    "RCU reader exited without a matching enter()",
                    reason="unbalanced read guard",
                    help_text="use the read() context manager or pair enter()/exit()",
                )
            slot.depth -= 1
            if slot.depth == 0:
                del self._readers[thread]
                self._condition.notify_all()

    @contextlib.contextmanager
    def read(self) -> Iterator[int]:
        epoch = self.enter()
        try:
            yield epoch
        finally:
            self.exit()

    @property
    def active_readers(self) -> int:
        with self._condition:
            return len(self._readers)

    # -- writer side -------------------------------------------------------

    def retire(
        self,
        item: Any,
        reclaim: Callable[[Any], None] | None = None,
        *,
        span: tuple[int, int] | None = None,
    ) -> int:
        with self._condition:
            epoch = self._epoch
            self._retired.append(_Retired(epoch, item, reclaim, span))
        return epoch

    def synchronize(self, *, timeout: float | None = 5.0) -> int:
        """Wait for a grace period and reclaim every eligible item.

        Returns the number of reclaimed items.  Raises ``E008`` when readers
        keep the grace period open past ``timeout``.
        """
        with self._writer_lock:
            with self._condition:
                target = self._epoch
                self._epoch += 1
                self._grace_periods += 1
                if timeout is None:
                    self._condition.wait_for(lambda: not self._readers_before(target))
                else:
                    finished = self._condition.wait_for(
                        lambda: not self._readers_before(target), timeout=timeout
                    )
                    if not finished:
                        raise HotpatchError(
                            "RCU grace period timed out",
                            reason=(
                                f"{self._readers_before(target)} reader(s) entered "
                                f"at or before epoch {target} are still active"
                            ),
                            help_text=(
                                "hold read guards only around AST traversal and "
                                "retry with a longer timeout in tests"
                            ),
                        )
                pending = [entry for entry in self._retired if entry.epoch <= target]
                self._retired = [
                    entry for entry in self._retired if entry.epoch > target
                ]
            reclaimed = 0
            for entry in pending:
                if entry.reclaim is not None:
                    try:
                        entry.reclaim(entry.item)
                    except Exception:
                        self._reclaim_failures += 1
                        continue
                reclaimed += 1
            with self._condition:
                self._reclaimed += reclaimed
            if reclaimed:
                # Reclamation is the observable migration; ``drain`` delegates
                # here, so it emits once per grace period without double counts.
                try:
                    emit(
                        "migration",
                        reclaimed=reclaimed,
                        operation="synchronize",
                    )
                except Exception:  # noqa: S110 - telemetry is diagnostic-only
                    pass
            return reclaimed

    def _readers_before(self, target: int) -> int:
        return sum(1 for slot in self._readers.values() if slot.epoch <= target)

    def stats(self) -> RcuStats:
        with self._condition:
            return RcuStats(
                epoch=self._epoch,
                active_readers=len(self._readers),
                retired=len(self._retired),
                reclaimed=self._reclaimed,
                grace_periods=self._grace_periods,
                reclaim_failures=self._reclaim_failures,
            )

    def drain(self, *, timeout: float | None = 5.0) -> int:
        """Reclaim everything retired so far (repeat until empty)."""
        total = 0
        while True:
            with self._condition:
                if not self._retired:
                    return total
            total += self.synchronize(timeout=timeout)


class AstEpochManager:
    """RCU-protected ``Program`` root with versioned publishes."""

    def __init__(
        self,
        program: ast.Program,
        *,
        domain: RcuDomain | None = None,
        on_reclaim: Callable[[ast.Program, int], None] | None = None,
    ) -> None:
        self.domain = domain or RcuDomain("ast")
        self._root = program
        self._version = 1
        self._lock = threading.Lock()
        self._on_reclaim = on_reclaim
        self.publications = 1
        self.reclaimed = 0
        self._retired_versions: list[int] = []

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self) -> tuple[int, ast.Program]:
        """Return the current ``(version, root)`` without taking a guard."""
        with self._lock:
            return self._version, self._root

    @contextlib.contextmanager
    def read(self) -> Iterator[ast.Program]:
        with self.domain.read():
            with self._lock:
                root = self._root
            yield root

    def publish(
        self, program: ast.Program, *, span: tuple[int, int] | None = None
    ) -> int:
        with self._lock:
            old = self._root
            old_version = self._version
            self._root = program
            self._version += 1
            version = self._version
        self._retired_versions.append(old_version)
        self.publications += 1
        self.domain.retire(
            old,
            lambda retired: self._reclaim(retired, old_version),
            span=span,
        )
        try:
            emit(
                "epoch",
                epoch=version,
                version=version,
                publications=self.publications,
            )
        except Exception:  # noqa: S110 - telemetry must never break publish
            pass
        return version

    def _reclaim(self, program: ast.Program, version: int) -> None:
        self.reclaimed += 1
        if self._on_reclaim is not None:
            self._on_reclaim(program, version)

    def synchronize(self, *, timeout: float | None = 5.0) -> int:
        return self.domain.synchronize(timeout=timeout)

    def drain(self, *, timeout: float | None = 5.0) -> int:
        return self.domain.drain(timeout=timeout)

    def stats(self) -> RcuStats:
        return self.domain.stats()


def hotpatch_rcu_ast_epoch(
    program: ast.Program,
    *,
    domain: RcuDomain | None = None,
    on_reclaim: Callable[[ast.Program, int], None] | None = None,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> AstEpochManager:
    """Create an RCU epoch manager for ``program``.

    ``span``/``source`` are accepted for API uniformity and are used when the
    caller later reports a relocation failure through the same journal.
    """

    if not isinstance(program, ast.Program):
        raise HotpatchError(
            "RCU epochs manage a parsed Program root",
            span=span,
            source=source,
            reason=f"got {type(program).__name__}",
            help_text="pass the value returned by parser.parse()",
        )
    return AstEpochManager(program, domain=domain, on_reclaim=on_reclaim)


def wait_for_epoch(
    manager: AstEpochManager, version: int, timeout: float = 5.0
) -> bool:
    """Block until ``manager.version >= version`` (used by tests and reactors)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.version >= version:
            return True
        time.sleep(0.001)
    return manager.version >= version


__all__ = [
    "AstEpochManager",
    "RcuDomain",
    "RcuStats",
    "hotpatch_rcu_ast_epoch",
    "wait_for_epoch",
]
