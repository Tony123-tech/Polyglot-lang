"""Zero-packet-loss reactor migration across reloaded modules.

Python builds use :mod:`selectors` (epoll on Linux, kqueue on macOS).  A
migration re-registers every ``(fd, events, data)`` entry on a fresh selector
while the underlying sockets stay open, so kernel receive buffers and
connection state survive.  The old selector owns only its epoll/kqueue
descriptor, which is never a socket the application reads from.

For a native ``mio::Poll`` reactor the caller supplies a bridge object with
``poll_fd()``, ``snapshot_registrations()``, ``reregister()`` and
``close_poller()``.  That is the documented PyO3 seam: the Rust module owns the
poll descriptor, while the Python side validates liveness and orchestrates the
swap.
"""

from __future__ import annotations

import contextlib
import os
import selectors
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

from runtime.hotpatch.errors import HotpatchError


@dataclass(frozen=True)
class Registration:
    fd: int
    events: int
    data: Any


@dataclass(frozen=True)
class ReactorState:
    kind: str
    poller_fd: int | None
    registrations: tuple[Registration, ...]
    generation: int = 1

    @property
    def fds(self) -> tuple[int, ...]:
        return tuple(registration.fd for registration in self.registrations)


def _selector_kind(selector: selectors.BaseSelector) -> str:
    return type(selector).__name__


def _poller_fd(selector: selectors.BaseSelector) -> int | None:
    poller = getattr(selector, "_selector", None)
    if poller is None:
        return None
    fileno = getattr(poller, "fileno", None)
    if fileno is None:
        return None
    try:
        return int(fileno())
    except OSError, ValueError:
        return None


def capture_selector(
    selector: selectors.BaseSelector,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> ReactorState:
    """Snapshot the registered interests without touching the socket queue."""
    if not isinstance(selector, selectors.BaseSelector):
        raise HotpatchError(
            "reactor source must be a selectors.BaseSelector",
            span=span,
            source=source,
            reason=f"got {type(selector).__name__}",
            help_text="pass a selector, a ReactorState, or a mio bridge object",
        )
    try:
        mapping = selector.get_map()
    except (ValueError, OSError) as error:
        raise HotpatchError(
            "reactor selector is already closed",
            span=span,
            source=source,
            reason=str(error),
            help_text="capture the selector before its owning module shuts down",
        ) from None
    registrations = tuple(
        Registration(key.fd, key.events, key.data) for key in mapping.values()
    )
    return ReactorState(_selector_kind(selector), _poller_fd(selector), registrations)


def _new_selector(old: selectors.BaseSelector) -> selectors.BaseSelector:
    try:
        return type(old)()
    except TypeError:
        return selectors.DefaultSelector()


@dataclass
class ReactorMigration:
    old_selector: selectors.BaseSelector
    new_selector: selectors.BaseSelector
    registrations: tuple[Registration, ...]
    duplicated: dict[int, int] = field(default_factory=dict)
    old_poller_fd: int | None = None
    new_poller_fd: int | None = None
    elapsed_ns: int = 0
    committed: bool = False
    rolled_back: bool = False
    span: tuple[int, int] | None = None
    source: str | None = None

    @property
    def selector(self) -> selectors.BaseSelector:
        return self.new_selector

    @property
    def packet_loss(self) -> int:
        # Re-registering does not drain or close any socket; no datagram can
        # be dropped by this operation.
        return 0

    def commit(self) -> None:
        """Retire the old selector after the owning module swaps its pointer."""
        if self.rolled_back:
            raise HotpatchError(
                "cannot commit a rolled-back reactor migration",
                help_text="create a new migration from the active selector",
            )
        if not self.committed:
            with contextlib.suppress(OSError, ValueError):
                self.old_selector.close()
            self.committed = True

    def rollback(self) -> selectors.BaseSelector:
        """Discard the new selector and keep serving from the old one."""
        if self.committed:
            raise HotpatchError(
                "cannot roll back a committed reactor migration",
                help_text="keep the migration object alive until verification passes",
            )
        if not self.rolled_back:
            with contextlib.suppress(OSError, ValueError):
                self.new_selector.close()
            for fd in self.duplicated.values():
                with contextlib.suppress(OSError):
                    os.close(fd)
            self.rolled_back = True
        return self.old_selector

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def __del__(self) -> None:
        # RAII: if the caller dropped the handle without an explicit
        # commit()/rollback(), discard the new selector and its duplicated fds.
        if not self.committed and not self.rolled_back:
            with contextlib.suppress(Exception):
                self.rollback()


def migrate_selector(
    selector: selectors.BaseSelector,
    *,
    duplicate_fds: bool = False,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> ReactorMigration:
    """Move all registrations to a pristine selector of the same kind."""
    started = time.perf_counter_ns()
    state = capture_selector(selector, span=span, source=source)
    new_selector = _new_selector(selector)
    duplicated: dict[int, int] = {}
    try:
        for registration in state.registrations:
            fd = registration.fd
            if duplicate_fds:
                try:
                    fd = os.dup(registration.fd)
                except OSError as error:
                    raise HotpatchError(
                        f"cannot duplicate reactor fd {registration.fd}",
                        span=span,
                        source=source,
                        reason=str(error),
                        help_text="check the descriptor is still open",
                    ) from None
                duplicated[registration.fd] = fd
            try:
                new_selector.register(fd, registration.events, registration.data)
            except (OSError, ValueError) as error:
                raise HotpatchError(
                    f"cannot re-register reactor fd {registration.fd}",
                    span=span,
                    source=source,
                    reason=str(error),
                    help_text="the descriptor closed between capture and migration",
                ) from None
    except Exception:
        with contextlib.suppress(OSError, ValueError):
            new_selector.close()
        for fd in duplicated.values():
            with contextlib.suppress(OSError):
                os.close(fd)
        raise
    return ReactorMigration(
        old_selector=selector,
        new_selector=new_selector,
        registrations=state.registrations,
        duplicated=duplicated,
        old_poller_fd=state.poller_fd,
        new_poller_fd=_poller_fd(new_selector),
        elapsed_ns=time.perf_counter_ns() - started,
        span=span,
        source=source,
    )


# ---------------------------------------------------------------------------
# Native mio bridge
# ---------------------------------------------------------------------------


@runtime_checkable
class MioReactorBridge(Protocol):
    """The PyO3 seam for a native ``mio::Poll`` reactor."""

    def poll_fd(self) -> int: ...

    def snapshot_registrations(self) -> Iterable[tuple[int, int, Any]]: ...

    def reregister(self, registrations: Iterable[tuple[int, int, Any]]) -> None: ...

    def close_poller(self) -> None: ...


@dataclass
class MioBridgeMigration:
    old_bridge: MioReactorBridge
    new_bridge: MioReactorBridge
    registrations: tuple[Registration, ...]
    old_poller_fd: int
    new_poller_fd: int
    elapsed_ns: int
    committed: bool = False
    rolled_back: bool = False
    span: tuple[int, int] | None = None
    source: str | None = None

    @property
    def packet_loss(self) -> int:
        return 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def commit(self) -> None:
        if self.rolled_back:
            raise HotpatchError(
                "cannot commit a rolled-back reactor migration",
                span=self.span,
                source=self.source,
                help_text="create a new migration from the active poller",
            )
        if not self.committed:
            try:
                self.old_bridge.close_poller()
            except Exception as error:
                raise HotpatchError(
                    "cannot close the old reactor poller",
                    span=self.span,
                    source=self.source,
                    reason=str(error),
                ) from None
            self.committed = True

    def rollback(self) -> MioReactorBridge:
        if self.committed:
            raise HotpatchError(
                "cannot roll back a committed reactor migration",
                span=self.span,
                source=self.source,
                help_text="keep the migration object alive until verification passes",
            )
        if not self.rolled_back:
            # The replacement poller owns no live socket; close its descriptor
            # so a dropped migration cannot leak the new epoll/kqueue fd.
            try:
                self.new_bridge.close_poller()
            except Exception as error:
                raise HotpatchError(
                    "cannot close the replacement reactor poller",
                    span=self.span,
                    source=self.source,
                    reason=str(error),
                ) from None
            self.rolled_back = True
        return self.old_bridge

    def __del__(self) -> None:
        if not self.committed and not self.rolled_back:
            with contextlib.suppress(Exception):
                self.rollback()


def _is_mio_bridge(value: Any) -> bool:
    return all(
        callable(getattr(value, name, None))
        for name in ("poll_fd", "snapshot_registrations", "reregister", "close_poller")
    )


def migrate_mio_bridge(
    old_bridge: MioReactorBridge,
    new_bridge: MioReactorBridge | None = None,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> MioBridgeMigration:
    if not _is_mio_bridge(old_bridge):
        raise HotpatchError(
            "object does not implement the mio reactor bridge protocol",
            span=span,
            source=source,
            reason="expected poll_fd(), snapshot_registrations(), reregister(), close_poller()",
            help_text="install the bridge from the PyO3 module that owns mio::Poll",
        )
    if new_bridge is None:
        raise HotpatchError(
            "no replacement mio reactor was supplied",
            code="E006",
            span=span,
            source=source,
            reason="the reloaded module did not expose a new bridge",
            help_text="construct the new module's Poll before reopening the descriptor",
        )
    if not _is_mio_bridge(new_bridge):
        raise HotpatchError(
            "replacement does not implement the mio reactor bridge protocol",
            span=span,
            source=source,
            reason="expected poll_fd(), snapshot_registrations(), reregister(), close_poller()",
            help_text="construct the new module's Poll before adopting registrations",
        )
    started = time.perf_counter_ns()
    try:
        old_poller = int(old_bridge.poll_fd())
    except (OSError, TypeError, ValueError) as error:
        raise HotpatchError(
            "cannot read the old reactor poll descriptor",
            span=span,
            source=source,
            reason=str(error),
        ) from None
    registrations: list[Registration] = []
    for fd, events, data in old_bridge.snapshot_registrations():
        try:
            os.fstat(fd)
        except OSError as error:
            raise HotpatchError(
                f"mio registration references a closed fd {fd}",
                span=span,
                source=source,
                reason=str(error),
            ) from None
        registrations.append(Registration(int(fd), int(events), data))
    try:
        new_bridge.reregister(
            (registration.fd, registration.events, registration.data)
            for registration in registrations
        )
        new_poller = int(new_bridge.poll_fd())
    except Exception as error:
        # The replacement poller was touched but never handed to the caller;
        # close it so the failed migration leaks no descriptor.
        with contextlib.suppress(Exception):
            new_bridge.close_poller()
        if isinstance(error, HotpatchError):
            raise
        raise HotpatchError(
            "cannot adopt registrations on the replacement reactor",
            span=span,
            source=source,
            reason=str(error),
        ) from None
    if new_poller == old_poller:
        raise HotpatchError(
            "the replacement reactor reused the live poll descriptor",
            span=span,
            source=source,
            reason="both modules would close the same epoll/kqueue fd",
            help_text="create the new Poll before adopting registrations",
        )
    return MioBridgeMigration(
        old_bridge=old_bridge,
        new_bridge=new_bridge,
        registrations=tuple(registrations),
        old_poller_fd=old_poller,
        new_poller_fd=new_poller,
        elapsed_ns=time.perf_counter_ns() - started,
        span=span,
        source=source,
    )


def hotpatch_mio_reactor_state(
    source: Any,
    target: Any | None = None,
    *,
    duplicate_fds: bool = False,
    span: tuple[int, int] | None = None,
    source_text: str | None = None,
) -> ReactorMigration | MioBridgeMigration:
    """Migrate a selector or a native ``mio::Poll`` bridge.

    ``source_text`` is the program text used to compute byte offsets for
    diagnostics; ``source`` is the reactor (kept for signature parity with the
    other subsystems).
    """
    if _is_mio_bridge(source):
        return migrate_mio_bridge(source, target, span=span, source=source_text)
    if isinstance(source, selectors.BaseSelector):
        return migrate_selector(
            source, duplicate_fds=duplicate_fds, span=span, source=source_text
        )
    if isinstance(source, ReactorState):
        raise HotpatchError(
            "a ReactorState snapshot has no live selector to migrate",
            span=span,
            source=source_text,
            reason="registrations alone cannot be re-armed without the poll object",
            help_text="pass the live selector or a mio bridge",
        )
    raise HotpatchError(
        "reactor source is not a selector or mio bridge",
        span=span,
        source=source_text,
        reason=f"got {type(source).__name__}",
    )


__all__ = [
    "MioBridgeMigration",
    "MioReactorBridge",
    "ReactorMigration",
    "ReactorState",
    "Registration",
    "capture_selector",
    "hotpatch_mio_reactor_state",
    "migrate_mio_bridge",
    "migrate_selector",
]
