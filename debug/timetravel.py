"""Deterministic, cross-FFI time-travel debugging foundations.

The debugger is a checkpoint/replay machine, not a live-frame rewriter.
Rewinding a running CPython frame or a suspended ``PyO3`` stack is not a
supported operation on any stable Python/Rust ABI, so this module never
pretends to do it.  Instead it provides:

* **Checkpoints** of Python variables and (through a provider protocol) native
  Rust state.
* **Bi-directional stepping** over the checkpoint log, with live objects only
  modified through an explicit ``apply`` callback.
* **Retroactive edits** that deterministically replay the recorded transitions
  after the edited checkpoint, so dependent future states are genuinely
  recomputed rather than guessed.
* **Native staleness tracking.**  Replaying Python transitions cannot execute
  Rust code; unless a ``native_replay`` hook is supplied, later native
  checkpoints are marked stale and restoring them raises ``E002``.

Python object graphs can be captured and restored with
:func:`snapshot_object_graph` and :func:`restore_object_graph`.  Those helpers
preserve identity for mutable containers and objects with ``__dict__``; they
do not resurrect objects that were created after the snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import types
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from runtime.diagnostics import PillarError

# ---------------------------------------------------------------------------
# Native provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NativeStateProvider(Protocol):
    """Adapter for a native execution context.

    ``capture`` must return a serializable/structured description of the
    native state.  ``restore`` is called by the debugger when travelling to a
    non-stale checkpoint.
    """

    def capture(self) -> Mapping[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class NativeDiff:
    """A key-level differential between two native checkpoints."""

    changed: Mapping[str, tuple[Any, Any]]
    added: Mapping[str, Any]
    removed: Mapping[str, Any]

    @property
    def empty(self) -> bool:
        return not (self.changed or self.added or self.removed)

    def render(self) -> str:
        parts: list[str] = []
        for name, (before, after) in sorted(self.changed.items()):
            parts.append(f"{name}: {before!r} -> {after!r}")
        for name in sorted(self.added):
            parts.append(f"+{name}")
        for name in sorted(self.removed):
            parts.append(f"-{name}")
        return ", ".join(parts) or "no native changes"


# ---------------------------------------------------------------------------
# Event log and checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    kind: str
    index: int
    detail: str = ""


def _stable_digest(state: Mapping[str, Any]) -> str:
    rendered = repr(sorted((str(key), value) for key, value in state.items()))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass
class Snapshot:
    index: int
    label: str
    state: dict[str, Any]
    native: dict[str, Any] | None = None
    stale_native: bool = False
    digest: str = ""
    frame: str | None = None

    def __post_init__(self) -> None:
        if not self.digest:
            self.digest = _stable_digest(self.state)

    def verify(self) -> bool:
        if _stable_digest(self.state) != self.digest:
            raise PillarError(
                "E008",
                f"checkpoint {self.index} was modified without replay",
                help_text="use modify() so dependents are recomputed",
            )
        return True


Transition = Callable[[MutableMapping[str, Any]], Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Debugger
# ---------------------------------------------------------------------------


class TimeTravelDebugger:
    """Checkpoint/replay debugger for mixed Python/native execution.

    Parameters
    ----------
    native:
        Optional adapter for a native execution context.  ``capture`` is
        invoked for every checkpoint and ``restore`` when travelling.
    native_replay:
        Optional pure function ``state -> native_state`` used to recompute
        native checkpoints after a retroactive edit.  Without it, affected
        native checkpoints are marked stale.
    apply:
        Optional callback invoked with a copy of a checkpoint's Python state
        after every successful travel.  Use it to synchronize live objects.
    serializer:
        Used when copying states.  Defaults to :func:`copy.deepcopy`.
    """

    def __init__(
        self,
        native: NativeStateProvider | None = None,
        *,
        native_replay: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        apply: Callable[[Mapping[str, Any]], None] | None = None,
        serializer: Callable[[Any], Any] = copy.deepcopy,
    ) -> None:
        self.native = native
        self.native_replay = native_replay
        self.apply = apply
        self.serializer = serializer
        self._snapshots: list[Snapshot] = []
        self._transitions: dict[int, Transition] = {}
        self._events: list[Event] = []
        self._cursor = -1

    # -- properties --------------------------------------------------------

    @property
    def snapshots(self) -> tuple[Snapshot, ...]:
        return tuple(self._snapshots)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    @property
    def cursor(self) -> int:
        return self._cursor

    def __len__(self) -> int:
        return len(self._snapshots)

    # -- recording ---------------------------------------------------------

    def checkpoint(
        self,
        state: Mapping[str, Any],
        *,
        label: str = "",
        transition: Transition | None = None,
        frame: str | None = None,
        capture_native: bool = True,
    ) -> Snapshot:
        """Record a checkpoint.

        ``transition`` describes how the *next* checkpoint is produced from
        this one.  It is stored on this checkpoint's index so
        :meth:`modify` can replay the gap.
        """
        index = len(self._snapshots)
        snapshot = Snapshot(
            index=index,
            label=label or f"checkpoint-{index}",
            state={str(key): self.serializer(value) for key, value in state.items()},
            native=(
                dict(self.native.capture())
                if capture_native and self.native is not None
                else None
            ),
            frame=frame,
        )
        if transition is not None:
            self._transitions[index] = transition
        self._snapshots.append(snapshot)
        self._cursor = index
        self._events.append(Event("checkpoint", index, snapshot.label))
        return snapshot

    def trace(
        self,
        initial_state: Mapping[str, Any],
        transitions: Sequence[Transition],
        *,
        labels: Sequence[str] | None = None,
    ) -> list[Snapshot]:
        """Record ``len(transitions) + 1`` checkpoints by replaying now.

        The transitions are kept so a retroactive edit can recompute every
        dependent checkpoint deterministically.
        """
        if labels is not None and len(labels) != len(transitions) + 1:
            raise PillarError(
                "E003",
                "label count does not match the number of checkpoints",
                help_text="supply one label per checkpoint, including the initial one",
            )
        state = {
            str(key): self.serializer(value) for key, value in initial_state.items()
        }
        label_for = (
            (lambda index: labels[index]) if labels is not None else (lambda index: "")
        )
        created = [
            self.checkpoint(
                state,
                label=label_for(0),
                transition=transitions[0] if transitions else None,
            )
        ]
        for index, transition in enumerate(transitions):
            state = dict(transition(copy.deepcopy(state)))
            self._transitions[index] = transition
            created.append(
                self.checkpoint(
                    state,
                    label=label_for(index + 1),
                    transition=(
                        transitions[index + 1] if index + 1 < len(transitions) else None
                    ),
                )
            )
        return created

    # -- navigation --------------------------------------------------------

    def _require_index(self, index: int) -> Snapshot:
        if not -len(self._snapshots) <= index < len(self._snapshots):
            raise PillarError(
                "E006",
                f"checkpoint {index} does not exist",
                reason=f"the timeline has {len(self._snapshots)} checkpoint(s)",
                help_text="record a checkpoint first or choose an existing index",
            )
        return self._snapshots[index]

    def current(self) -> Snapshot:
        if not self._snapshots:
            raise PillarError("E006", "the timeline is empty")
        self._cursor = max(self._cursor, 0)
        return self._snapshots[self._cursor]

    def travel_to(self, index: int, *, restore_native: bool = True) -> Snapshot:
        snapshot = self._require_index(index)
        if snapshot.stale_native and restore_native:
            raise PillarError(
                "E002",
                f"native state at checkpoint {snapshot.index} is stale",
                reason="a retroactive edit replayed Python state but not native code",
                help_text="provide native_replay=... or re-run the native provider",
            )
        if restore_native and self.native is not None and snapshot.native is not None:
            self.native.restore(copy.deepcopy(snapshot.native))
        self._cursor = snapshot.index
        self._events.append(Event("travel", snapshot.index, snapshot.label))
        if self.apply is not None:
            self.apply(copy.deepcopy(snapshot.state))
        return snapshot

    def step_back(self) -> Snapshot:
        if not self._snapshots:
            raise PillarError("E006", "the timeline is empty")
        target = self._cursor - 1
        if target < 0:
            raise PillarError(
                "E006",
                "already at the first checkpoint",
                help_text="use step_forward() or travel_to()",
            )
        return self.travel_to(target)

    def step_forward(self) -> Snapshot:
        if not self._snapshots:
            raise PillarError("E006", "the timeline is empty")
        target = self._cursor + 1
        if target >= len(self._snapshots):
            raise PillarError(
                "E006",
                "already at the last checkpoint",
                help_text="use step_back() or travel_to()",
            )
        return self.travel_to(target)

    # -- retroactive editing ----------------------------------------------

    def modify(
        self,
        index: int,
        name: str,
        value: Any,
        *,
        recompute: bool = True,
    ) -> list[Snapshot]:
        """Change a past variable and recompute every dependent checkpoint."""
        snapshot = self._require_index(index)
        if name not in snapshot.state:
            raise PillarError(
                "E006",
                f"checkpoint {snapshot.index} has no variable {name!r}",
                help_text="choose a variable captured at that checkpoint",
            )
        if not recompute:
            snapshot.state[name] = self.serializer(value)
            snapshot.digest = ""
            snapshot.__post_init__()
            self._events.append(Event("modify", snapshot.index, f"{name}={value!r}"))
            return list(self._snapshots)

        snapshot.state[name] = self.serializer(value)
        snapshot.digest = ""
        snapshot.__post_init__()
        self._events.append(Event("modify", snapshot.index, f"{name}={value!r}"))
        self._recompute_from(snapshot.index)
        return list(self._snapshots)

    def _recompute_from(self, index: int) -> None:
        for next_index in range(index + 1, len(self._snapshots)):
            transition = self._transitions.get(next_index - 1)
            if transition is None:
                raise PillarError(
                    "E002",
                    f"cannot recompute checkpoint {next_index}",
                    reason="no transition was recorded for the preceding gap",
                    help_text="record transitions with checkpoint(transition=...) or trace()",
                )
            previous = self._snapshots[next_index - 1]
            try:
                new_state = dict(transition(copy.deepcopy(previous.state)))
            except PillarError:
                raise
            except Exception as error:
                raise PillarError(
                    "E002",
                    f"replay transition failed at checkpoint {next_index}",
                    reason=f"{type(error).__name__}: {error}",
                    help_text="make transitions deterministic and total over the edited state",
                ) from error
            native_state: dict[str, Any] | None = None
            stale = False
            if self.native is not None:
                if self.native_replay is not None:
                    try:
                        native_state = dict(self.native_replay(new_state))
                    except Exception as error:
                        raise PillarError(
                            "E004",
                            f"native replay failed at checkpoint {next_index}",
                            reason=f"{type(error).__name__}: {error}",
                            help_text="provide a native_replay hook that accepts every replayed state",
                        ) from error
                else:
                    stale = True
                    native_state = copy.deepcopy(previous.native)
            old = self._snapshots[next_index]
            rebuilt = Snapshot(
                index=next_index,
                label=old.label,
                state=new_state,
                native=native_state,
                stale_native=stale,
                frame=old.frame,
            )
            self._snapshots[next_index] = rebuilt
            self._events.append(
                Event(
                    "recompute",
                    next_index,
                    "stale native state" if stale else "recomputed",
                )
            )
        self._cursor = index

    # -- native differentials ---------------------------------------------

    def differential(self, before: int, after: int) -> NativeDiff:
        left = self._require_index(before)
        right = self._require_index(after)
        if left.native is None or right.native is None:
            raise PillarError(
                "E006",
                "native differential requires two native checkpoints",
                help_text="configure a NativeStateProvider and capture native state",
            )
        changed: dict[str, tuple[Any, Any]] = {}
        added: dict[str, Any] = {}
        removed: dict[str, Any] = {}
        for key in left.native.keys() | right.native.keys():
            if key not in left.native:
                added[key] = right.native[key]
            elif key not in right.native:
                removed[key] = left.native[key]
            elif left.native[key] != right.native[key]:
                changed[key] = (left.native[key], right.native[key])
        return NativeDiff(changed=changed, added=added, removed=removed)


# ---------------------------------------------------------------------------
# Python frame and object-graph snapshots
# ---------------------------------------------------------------------------


def capture_python_frame(frame: types.FrameType | None = None) -> dict[str, Any]:
    """Capture the locals of ``frame`` (or the caller's frame).

    The values are shallow references; wrap them with ``copy.deepcopy`` if the
    graph will mutate.  Nothing is written back into the frame.
    """
    if frame is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            raise PillarError("E007", "the running interpreter exposes no Python frame")
        frame = frame.f_back
    return {
        "__frame__": frame.f_code.co_name,
        **{name: value for name, value in frame.f_locals.items()},
    }


@dataclass(frozen=True)
class GraphEntry:
    kind: str
    target: Any
    payload: Any


@dataclass(frozen=True)
class ObjectGraphSnapshot:
    roots: tuple[Any, ...]
    entries: tuple[GraphEntry, ...]
    frame: str | None = None

    @property
    def objects(self) -> int:
        return len(self.entries)


def snapshot_object_graph(
    roots: Iterable[Any],
    *,
    frame: types.FrameType | None = None,
) -> ObjectGraphSnapshot:
    """Capture an identity-preserving snapshot of Python object graphs."""
    entries: list[GraphEntry] = []
    seen: set[int] = set()
    root_tuple = tuple(roots)

    def visit(value: Any) -> None:
        if isinstance(value, (str, bytes, int, float, bool, type(None), complex)):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, list):
            contents = list(value)
            entries.append(GraphEntry("list", value, contents))
            for item in contents:
                visit(item)
            return
        if isinstance(value, dict):
            items = list(value.items())
            entries.append(GraphEntry("dict", value, items))
            for key, item in items:
                visit(key)
                visit(item)
            return
        if isinstance(value, set):
            items = set(value)
            entries.append(GraphEntry("set", value, items))
            for item in items:
                visit(item)
            return
        if isinstance(value, tuple):
            entries.append(GraphEntry("tuple", value, tuple(value)))
            for item in value:
                visit(item)
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            payload = dict(attributes)
            entries.append(GraphEntry("object", value, payload))
            for item in payload.values():
                visit(item)

    for root in root_tuple:
        visit(root)
    frame_name = frame.f_code.co_name if frame is not None else None
    # ``entries`` intentionally holds strong references: an identity-preserving
    # restore is impossible if the captured graph has been collected.
    return ObjectGraphSnapshot(
        roots=root_tuple, entries=tuple(entries), frame=frame_name
    )


def restore_object_graph(snapshot: ObjectGraphSnapshot) -> int:
    """Restore a previously captured graph in place and return object count."""
    restored = 0
    for entry in snapshot.entries:
        target = entry.target
        if entry.kind == "list":
            target[:] = entry.payload
        elif entry.kind == "dict" or entry.kind == "set":
            target.clear()
            target.update(entry.payload)
        elif entry.kind == "object":
            attributes = getattr(target, "__dict__", None)
            if attributes is None:
                continue
            attributes.clear()
            attributes.update(entry.payload)
        else:
            # Tuples cannot be restored in place; their mutable members are
            # restored by their own entries.
            continue
        restored += 1
    return restored


__all__ = [
    "Event",
    "GraphEntry",
    "NativeDiff",
    "NativeStateProvider",
    "ObjectGraphSnapshot",
    "Snapshot",
    "TimeTravelDebugger",
    "capture_python_frame",
    "restore_object_graph",
    "snapshot_object_graph",
]
