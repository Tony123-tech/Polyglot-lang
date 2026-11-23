"""OTP-inspired supervision strategies with restart intensity limits.

OTP supervisors enforce a *restart budget*: if a child crashes more than
``max_restarts`` times within ``window_seconds``, the supervisor gives up
and escalates to its own parent (or terminates).  This module models the
three standard OTP strategies:

* ``one_for_one`` — restart only the crashed child.
* ``one_for_all``  — restart every child when any child crashes.
* ``rest_for_one`` — restart the crashed child and every child started after it.

The supervisor is deterministic: time is supplied by the caller (integer
ticks or seconds), and no wall-clock sleeping is used.  State is never shared
across actors — children are restarted from their initial state.

Diagnostics follow the shared ``E001``–``E008`` surface: ``E004`` when a
restart budget is exhausted and the supervisor escalates, ``E001`` for invalid
configuration, and ``E003`` for API misuse.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compiler.frontend.errors import PolyglotError

__all__ = [
    "RestartIntensity",
    "RestartOutcome",
    "Strategy",
    "SupervisionTree",
    "Supervisor",
]


def _fail(code: str, message: str, *, reason: str | None = None) -> None:
    raise PolyglotError(code, message, reason=reason)


class Strategy(Enum):
    """An OTP supervision strategy."""

    ONE_FOR_ONE = "one_for_one"
    ONE_FOR_ALL = "one_for_all"
    REST_FOR_ONE = "rest_for_one"

    @classmethod
    def from_name(cls, name: str) -> Strategy:
        try:
            return cls(name.lower())
        except ValueError:
            _fail("E001", f"unknown supervision strategy: {name!r}")


@dataclass(frozen=True)
class RestartIntensity:
    """The maximum number of restarts allowed within a time window."""

    max_restarts: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            _fail("E001", "max_restarts must be non-negative")
        if self.window_seconds < 0:
            _fail("E001", "window_seconds must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_restarts": self.max_restarts,
            "window_seconds": self.window_seconds,
        }

    def exceeded(self, restart_times: Sequence[float]) -> bool:
        """Return whether the restarts in *restart_times* exceed the budget.

        *restart_times* is a sequence of timestamps (in seconds) at which
        restarts occurred.  The budget is exceeded when more than
        ``max_restarts`` restarts fall within ``window_seconds`` of the most
        recent restart.
        """
        if not restart_times or self.max_restarts == 0:
            return bool(restart_times) and self.max_restarts == 0
        if len(restart_times) <= self.max_restarts:
            return False
        recent = restart_times[-self.max_restarts - 1 :]
        window_start = restart_times[-1] - self.window_seconds
        count_in_window = sum(1 for t in recent if t >= window_start)
        return count_in_window > self.max_restarts


@dataclass(frozen=True)
class RestartOutcome:
    """The outcome of a supervisor restart decision."""

    restarted: tuple[str, ...]
    escalated: bool
    reason: str
    strategy: Strategy

    def as_dict(self) -> dict[str, Any]:
        return {
            "restarted": list(self.restarted),
            "escalated": self.escalated,
            "reason": self.reason,
            "strategy": self.strategy.value,
        }


@dataclass
class _ChildSpec:
    """Internal record for a supervised child."""

    name: str
    factory: Callable[[], Any]
    initial_state: Any


@dataclass
class Supervisor:
    """An OTP-style supervisor over a set of named children.

    The supervisor tracks restart history per child and enforces a global
    :class:`RestartIntensity` budget.  When a child crashes (raises an
    exception during ``run``), the supervisor applies its *strategy* to
    determine which children to restart and whether to escalate.
    """

    name: str
    strategy: Strategy
    intensity: RestartIntensity
    children: list[_ChildSpec] = field(default_factory=list)
    _restart_times: list[float] = field(default_factory=list)
    _current_states: dict[str, Any] = field(default_factory=dict)
    _alive: dict[str, bool] = field(default_factory=dict)

    def add_child(
        self, name: str, factory: Callable[[], Any], *, initial_state: Any = None
    ) -> None:
        if any(child.name == name for child in self.children):
            _fail("E001", f"child {name!r} is already supervised")
        if not callable(factory):
            _fail("E003", f"child {name!r} factory must be callable")
        self.children.append(
            _ChildSpec(name=name, factory=factory, initial_state=initial_state)
        )
        self._current_states[name] = copy.deepcopy(initial_state)
        self._alive[name] = True

    def child_names(self) -> tuple[str, ...]:
        return tuple(child.name for child in self.children)

    def restart(self, crashed_child: str, *, now: float) -> RestartOutcome:
        """Handle a crash of *crashed_child* at time *now*.

        Applies the supervision strategy to determine which children to
        restart.  If the restart intensity budget is exceeded, the supervisor
        escalates (raises ``E004``).
        """
        if crashed_child not in self._alive:
            _fail("E001", f"unknown child {crashed_child!r}")
        self._restart_times.append(now)
        if self.intensity.exceeded(self._restart_times):
            reason = (
                f"restart intensity exceeded: {len(self._restart_times)} "
                f"restarts in the budget window"
            )
            # Reset the budget window so a fresh start is possible after
            # escalation, but still raise to signal the failure.
            self._restart_times.clear()
            _fail("E004", reason, reason="supervisor escalating to parent")

        to_restart = self._strategy_restart_set(crashed_child)
        for child in self.children:
            if child.name in to_restart:
                self._current_states[child.name] = copy.deepcopy(child.initial_state)
                self._alive[child.name] = True
            elif self.strategy == Strategy.ONE_FOR_ALL:
                self._alive[child.name] = False
        return RestartOutcome(
            restarted=tuple(to_restart),
            escalated=False,
            reason=f"strategy {self.strategy.value} restarted {', '.join(to_restart)}",
            strategy=self.strategy,
        )

    def _strategy_restart_set(self, crashed_child: str) -> list[str]:
        if self.strategy == Strategy.ONE_FOR_ONE:
            return [crashed_child]
        if self.strategy == Strategy.ONE_FOR_ALL:
            return [child.name for child in self.children]
        # rest_for_one: the crashed child and every child started after it.
        names = [child.name for child in self.children]
        start = names.index(crashed_child)
        return names[start:]

    def alive_children(self) -> tuple[str, ...]:
        return tuple(name for name, alive in self._alive.items() if alive)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "intensity": self.intensity.as_dict(),
            "children": self.child_names(),
            "alive": self.alive_children(),
            "restart_count": len(self._restart_times),
        }


@dataclass
class SupervisionTree:
    """A tree of supervisors with parent-child escalation.

    When a child supervisor exhausts its restart budget, it escalates to its
    parent supervisor by raising ``E004``.  The root supervisor has no parent
    and terminating it ends the tree.
    """

    root: Supervisor
    _children: dict[str, Supervisor] = field(default_factory=dict)

    def add_supervisor(self, parent_name: str, child: Supervisor) -> None:
        if parent_name == self.root.name:
            self._children[child.name] = child
        else:
            parent = self._children.get(parent_name)
            if parent is None:
                _fail("E006", f"unknown parent supervisor {parent_name!r}")
            # Nested children are stored on the parent's own child list.
            parent.children.append(  # type: ignore[attr-defined]
                _ChildSpec(
                    name=child.name, factory=lambda c=child: c, initial_state=child
                )
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.as_dict(),
            "supervisors": {name: s.as_dict() for name, s in self._children.items()},
        }
