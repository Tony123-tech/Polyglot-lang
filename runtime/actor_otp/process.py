"""Elixir/BEAM-inspired isolated actor processes with message passing.

An :class:`Actor` owns its state and a :class:`Mailbox` of pending messages.
State is never shared across actors: when a message is sent, the value is
*deep-copied* (or transferred by ownership for native types), so an actor
cannot mutate another actor's state through a shared reference.  This models
BEAM's "share nothing" process isolation.

The module is deterministic and pure Python: actors do not spawn real OS
threads, and message delivery is synchronous via an explicit ``run`` method
that drains the mailbox.  A caller that wants real concurrency can wrap
actors in threads; the isolation guarantees are unchanged because state is
copied at the message boundary.

Diagnostics follow the shared ``E001``–``E008`` surface: ``E001`` for invalid
actor configuration, ``E003`` for API misuse, and ``E006`` for messages to
unknown actors.
"""

from __future__ import annotations

import copy
import itertools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from compiler.frontend.errors import PolyglotError

__all__ = [
    "Actor",
    "ActorId",
    "Mailbox",
    "ProcessRegistry",
    "spawn_actor",
]

_ActorIdCounter: Final = itertools.count(1)


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    raise PolyglotError(code, message, reason=reason)


@dataclass(frozen=True)
class ActorId:
    """A unique, opaque identifier for an actor process."""

    value: int

    def as_dict(self) -> dict[str, Any]:
        return {"actor_id": self.value}

    def __str__(self) -> str:
        return f"<Actor#{self.value}>"


def spawn_actor_id() -> ActorId:
    """Allocate a fresh :class:`ActorId`."""
    return ActorId(value=next(_ActorIdCounter))


@dataclass
class Mailbox:
    """A structured mailbox holding pending messages for an actor.

    Messages are stored as ``(sender_id, payload)`` pairs.  The mailbox
    supports selective receive: ``receive_until`` drains messages until a
    predicate matches, leaving non-matching messages in the mailbox.
    """

    _messages: deque[tuple[ActorId | None, Any]] = field(
        default_factory=deque, repr=False
    )
    _capacity: int = 0  # 0 = unbounded

    def __post_init__(self) -> None:
        if self._capacity < 0:
            _fail("E001", "mailbox capacity must be non-negative")

    def send(self, payload: Any, *, sender: ActorId | None = None) -> None:
        if self._capacity and len(self._messages) >= self._capacity:
            _fail(
                "E001",
                f"mailbox is full (capacity {self._capacity})",
                reason="the actor's mailbox overflowed; increase capacity or "
                "drain the mailbox",
            )
        self._messages.append((sender, copy.deepcopy(payload)))

    def receive(self) -> tuple[ActorId | None, Any] | None:
        if not self._messages:
            return None
        return self._messages.popleft()

    def receive_until(
        self, predicate: Callable[[Any], bool]
    ) -> tuple[ActorId | None, Any] | None:
        """Drain messages until *predicate* matches, preserving the rest.

        Returns the first matching ``(sender, payload)`` pair, or ``None``
        if the mailbox is exhausted without a match.
        """
        skipped: list[tuple[ActorId | None, Any]] = []
        result: tuple[ActorId | None, Any] | None = None
        while self._messages:
            entry = self._messages.popleft()
            if predicate(entry[1]):
                result = entry
                break
            skipped.append(entry)
        for entry in skipped:
            self._messages.appendleft(entry)
        return result

    def __len__(self) -> int:
        return len(self._messages)

    def is_empty(self) -> bool:
        return not self._messages

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": len(self._messages),
            "capacity": self._capacity,
            "empty": self.is_empty(),
        }


@dataclass
class Actor:
    """A lightweight, isolated actor process.

    The actor owns its *state* and a :class:`Mailbox`.  The ``handler``
    callable is invoked for each message as ``handler(state, payload)`` and
    returns the new state (or ``None`` to keep the current state).  State is
    never shared: the mailbox deep-copies payloads on send, and the handler
    receives the actor's own state by value.
    """

    actor_id: ActorId
    state: Any
    handler: Callable[[Any, Any], Any]
    mailbox: Mailbox = field(default_factory=Mailbox)
    _alive: bool = field(default=True, repr=False)

    @property
    def alive(self) -> bool:
        return self._alive

    def send(self, payload: Any, *, sender: ActorId | None = None) -> None:
        """Send a message to this actor (deep-copied at the boundary)."""
        if not self._alive:
            _fail(
                "E001",
                f"actor {self.actor_id} is not alive; cannot send messages",
                reason="send to a dead actor",
            )
        self.mailbox.send(payload, sender=sender)

    def run(self, *, max_messages: int = 0) -> int:
        """Drain the mailbox, invoking the handler for each message.

        Returns the number of messages processed.  If the handler raises,
        the actor is marked dead and the exception propagates (let-it-crash
        semantics; the supervisor is responsible for catching and restarting).
        """
        if not self._alive:
            return 0
        processed = 0
        while self.mailbox._messages:
            if max_messages and processed >= max_messages:
                break
            entry = self.mailbox.receive()
            assert entry is not None
            _, payload = entry
            try:
                new_state = self.handler(self.state, payload)
            except Exception:
                self._alive = False
                raise
            if new_state is not None:
                self.state = new_state
            processed += 1
        return processed

    def stop(self) -> None:
        """Mark the actor as dead.  Pending messages are discarded."""
        self._alive = False
        self.mailbox._messages.clear()

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id.value,
            "alive": self._alive,
            "mailbox": self.mailbox.as_dict(),
        }


def spawn_actor(
    state: Any,
    handler: Callable[[Any, Any], Any],
    *,
    actor_id: ActorId | None = None,
    mailbox_capacity: int = 0,
) -> Actor:
    """Spawn a new isolated actor process.

    The *handler* is called as ``handler(state, payload)`` for each message
    and must return the new state (or ``None`` to keep the current state).
    The actor's state is never shared with the caller after spawn — the
    caller should treat the actor as a black box that communicates only via
    messages.
    """
    if not callable(handler):
        _fail("E003", "actor handler must be callable")
    if mailbox_capacity < 0:
        _fail("E001", "mailbox capacity must be non-negative")
    return Actor(
        actor_id=actor_id or spawn_actor_id(),
        state=copy.deepcopy(state),
        handler=handler,
        mailbox=Mailbox(_capacity=mailbox_capacity),
    )


class ProcessRegistry:
    """A registry of live actors keyed by :class:`ActorId`."""

    def __init__(self) -> None:
        self._actors: dict[ActorId, Actor] = {}

    def register(self, actor: Actor) -> None:
        if actor.actor_id in self._actors:
            _fail("E001", f"actor {actor.actor_id} is already registered")
        self._actors[actor.actor_id] = actor

    def get(self, actor_id: ActorId) -> Actor:
        actor = self._actors.get(actor_id)
        if actor is None:
            _fail("E006", f"unknown actor {actor_id}")
        return actor

    def send(self, actor_id: ActorId, payload: Any) -> None:
        self.get(actor_id).send(payload)

    def run_all(self, *, max_rounds: int = 1) -> int:
        """Run every alive actor up to *max_rounds* mailbox drains."""
        total = 0
        for _ in range(max_rounds):
            for actor in self._actors.values():
                if actor.alive:
                    total += actor.run()
        return total

    def alive_actors(self) -> tuple[ActorId, ...]:
        return tuple(
            actor_id for actor_id, actor in self._actors.items() if actor.alive
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._actors),
            "alive": [a.value for a in self.alive_actors()],
        }
