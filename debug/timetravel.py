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
    # Sort by key only.  Sorting the ``(key, value)`` tuples would compare
    # values whenever two keys stringify identically, which is neither stable
    # nor safe for opaque native values.
    items = sorted(
        ((str(key), key, value) for key, value in state.items()),
        key=lambda item: (item[0], repr(item[1])),
    )
    rendered = repr([(name, value) for name, _key, value in items])
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

    # -- structured failure helpers ----------------------------------------

    def _serialize_value(self, label: str, value: Any) -> Any:
        try:
            return self.serializer(value)
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E008",
                f"cannot serialize {label}",
                reason=f"{type(error).__name__}: {error}",
                help_text="make the serializer total over captured values",
            ) from error

    def _copy_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return {key: self.serializer(value) for key, value in state.items()}
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E008",
                "cannot copy checkpoint state",
                reason=f"{type(error).__name__}: {error}",
                help_text="make the serializer total over checkpoint values",
            ) from error

    def _copy_native_state(
        self, state: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if state is None:
            return None
        try:
            return dict(copy.deepcopy(state))
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E004",
                "cannot copy native checkpoint state",
                reason=f"{type(error).__name__}: {error}",
                help_text="native state must be structured and copyable",
            ) from error

    def _capture_native(self) -> dict[str, Any] | None:
        if self.native is None:
            return None
        try:
            captured = self.native.capture()
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E004",
                "native state capture failed",
                reason=f"{type(error).__name__}: {error}",
                help_text="make the native provider's capture() total",
            ) from error
        if not isinstance(captured, Mapping):
            raise PillarError(
                "E004",
                "native state capture did not return a mapping",
                reason=f"got {type(captured).__name__}",
                help_text="return a mapping of native key/value pairs from capture()",
            )
        return dict(captured)

    def _restore_native(self, snapshot: Snapshot) -> None:
        if self.native is None or snapshot.native is None:
            return
        try:
            payload = copy.deepcopy(snapshot.native)
        except Exception as error:
            raise PillarError(
                "E004",
                f"cannot copy native state for checkpoint {snapshot.index}",
                reason=f"{type(error).__name__}: {error}",
                help_text="native state must be structured and copyable",
            ) from error
        try:
            self.native.restore(payload)
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E004",
                f"native restore failed at checkpoint {snapshot.index}",
                reason=f"{type(error).__name__}: {error}",
                help_text="make the native provider's restore() total",
            ) from error

    def _apply_state(self, snapshot: Snapshot) -> None:
        if self.apply is None:
            return
        payload = self._copy_state(snapshot.state)
        try:
            self.apply(payload)
        except PillarError:
            raise
        except Exception as error:
            raise PillarError(
                "E003",
                f"apply callback failed at checkpoint {snapshot.index}",
                reason=f"{type(error).__name__}: {error}",
                help_text="make the apply callback total over restored states",
            ) from error

    @staticmethod
    def _refresh_digest(snapshot: Snapshot) -> None:
        snapshot.digest = ""
        snapshot.__post_init__()

    def _commit_edits(
        self,
        states: Mapping[int, dict[str, Any]],
        natives: Mapping[int, dict[str, Any] | None],
        stales: Mapping[int, bool],
        events: Sequence[Event],
        *,
        cursor: int | None,
    ) -> None:
        """Commit a fully planned edit, preserving Snapshot object identity."""
        for index, state in states.items():
            target = self._snapshots[index]
            target.state = state
            self._refresh_digest(target)
            if index in natives:
                target.native = natives[index]
            if index in stales:
                target.stale_native = stales[index]
        self._events.extend(events)
        if cursor is not None:
            self._cursor = cursor

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
        captured_state: dict[str, Any] = {}
        for key, value in state.items():
            name = str(key)
            if name in captured_state:
                raise PillarError(
                    "E003",
                    f"state keys collide after string coercion: {name!r}",
                    help_text="use distinct string keys when recording a checkpoint",
                )
            captured_state[name] = self._serialize_value(f"state key {name!r}", value)
        snapshot = Snapshot(
            index=index,
            label=label or f"checkpoint-{index}",
            state=captured_state,
            native=self._capture_native() if capture_native else None,
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
        state: dict[str, Any] = {}
        for key, value in initial_state.items():
            name = str(key)
            if name in state:
                raise PillarError(
                    "E003",
                    f"state keys collide after string coercion: {name!r}",
                    help_text="use distinct string keys when recording a trace",
                )
            state[name] = self._serialize_value(f"initial state key {name!r}", value)
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
            replay_input = self._copy_state(state)
            try:
                state = dict(transition(replay_input))
            except PillarError:
                raise
            except Exception as error:
                raise PillarError(
                    "E002",
                    f"transition failed while recording checkpoint {index + 1}",
                    reason=f"{type(error).__name__}: {error}",
                    help_text="make transitions deterministic and total",
                ) from error
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
        if restore_native:
            self._restore_native(snapshot)
        self._cursor = snapshot.index
        self._events.append(Event("travel", snapshot.index, snapshot.label))
        self._apply_state(snapshot)
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
        """Change a past variable and recompute every dependent checkpoint.

        The edit is planned in full before anything is committed, so a failed
        replay leaves the timeline, digests, cursor, and event log untouched.
        """
        snapshot = self._require_index(index)
        target_index = snapshot.index
        if name not in snapshot.state:
            raise PillarError(
                "E006",
                f"checkpoint {target_index} has no variable {name!r}",
                help_text="choose a variable captured at that checkpoint",
            )
        serialized = self._serialize_value(f"modified value {name!r}", value)
        base_state = self._copy_state(snapshot.state)
        base_state[name] = serialized

        states: dict[int, dict[str, Any]] = {target_index: base_state}
        natives: dict[int, dict[str, Any] | None] = {}
        stales: dict[int, bool] = {}
        events = [Event("modify", target_index, f"{name}={value!r}")]
        if recompute:
            (
                replayed_states,
                replayed_natives,
                replayed_stales,
                replayed_events,
            ) = self._plan_recompute(target_index, base_state)
            states.update(replayed_states)
            natives.update(replayed_natives)
            stales.update(replayed_stales)
            events.extend(replayed_events)
        self._commit_edits(states, natives, stales, events, cursor=target_index)
        return list(self._snapshots)

    def _plan_recompute(
        self, index: int, base_state: Mapping[str, Any]
    ) -> tuple[
        dict[int, dict[str, Any]],
        dict[int, dict[str, Any] | None],
        dict[int, bool],
        list[Event],
    ]:
        """Plan dependent checkpoints without mutating the recorded timeline."""
        states: dict[int, dict[str, Any]] = {}
        natives: dict[int, dict[str, Any] | None] = {}
        stales: dict[int, bool] = {}
        events: list[Event] = []
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
            if next_index - 1 == index:
                previous_state = base_state
            else:
                previous_state = states.get(next_index - 1, previous.state)
            replay_input = self._copy_state(previous_state)
            try:
                new_state = dict(transition(replay_input))
            except PillarError:
                raise
            except Exception as error:
                raise PillarError(
                    "E002",
                    f"replay transition failed at checkpoint {next_index}",
                    reason=f"{type(error).__name__}: {error}",
                    help_text="make transitions deterministic and total over the edited state",
                ) from error
            states[next_index] = new_state

            native_state: dict[str, Any] | None = None
            stale = False
            if self.native is not None:
                if self.native_replay is not None:
                    try:
                        native_state = dict(self.native_replay(new_state))
                    except PillarError:
                        raise
                    except Exception as error:
                        raise PillarError(
                            "E004",
                            f"native replay failed at checkpoint {next_index}",
                            reason=f"{type(error).__name__}: {error}",
                            help_text="provide a native_replay hook that accepts every replayed state",
                        ) from error
                else:
                    stale = True
                    native_state = self._copy_native_state(previous.native)
            natives[next_index] = native_state
            stales[next_index] = stale
            events.append(
                Event(
                    "recompute",
                    next_index,
                    "stale native state" if stale else "recomputed",
                )
            )
        return states, natives, stales, events

    def _recompute_from(self, index: int) -> None:
        snapshot = self._require_index(index)
        states, natives, stales, events = self._plan_recompute(
            snapshot.index, self._copy_state(snapshot.state)
        )
        self._commit_edits(states, natives, stales, events, cursor=snapshot.index)

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
        current = inspect.currentframe()
        try:
            if current is None or current.f_back is None:
                raise PillarError(
                    "E007", "the running interpreter exposes no Python frame"
                )
            frame = current.f_back
        finally:
            del current
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
        if isinstance(value, (set, frozenset)):
            kind = "set" if isinstance(value, set) else "frozenset"
            items = set(value)
            entries.append(GraphEntry(kind, value, items))
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


def _invalid_graph_entry(entry: GraphEntry, reason: str) -> PillarError:
    return PillarError(
        "E008",
        f"invalid object graph entry {getattr(entry, 'kind', '<unknown>')!r}",
        reason=reason,
        help_text="rebuild the snapshot with snapshot_object_graph()",
    )


def restore_object_graph(snapshot: ObjectGraphSnapshot) -> int:
    """Restore a previously captured graph in place and return object count."""
    if not isinstance(snapshot, ObjectGraphSnapshot):
        raise PillarError(
            "E008",
            "expected an ObjectGraphSnapshot",
            reason=f"got {type(snapshot).__name__}",
            help_text="pass the value returned by snapshot_object_graph()",
        )
    restored = 0
    for entry in snapshot.entries:
        if not isinstance(entry, GraphEntry):
            raise _invalid_graph_entry(entry, "entries must be GraphEntry values")
        target = entry.target
        payload = entry.payload
        if entry.kind == "list":
            if not isinstance(target, list) or not isinstance(payload, (list, tuple)):
                raise _invalid_graph_entry(
                    entry, "list entries need a list target and list/tuple payload"
                )
            target[:] = payload
        elif entry.kind == "dict":
            if not isinstance(target, dict):
                raise _invalid_graph_entry(entry, "dict entries need a dict target")
            if isinstance(payload, Mapping):
                items = list(payload.items())
            elif isinstance(payload, (list, tuple)) and all(
                isinstance(item, (list, tuple)) and len(item) == 2 for item in payload
            ):
                items = list(payload)
            else:
                raise _invalid_graph_entry(
                    entry, "dict payload must be a mapping or key/value pairs"
                )
            target.clear()
            target.update(items)
        elif entry.kind == "set":
            if not isinstance(target, set) or not isinstance(
                payload, (set, frozenset, list, tuple)
            ):
                raise _invalid_graph_entry(
                    entry, "set entries need a set target and iterable payload"
                )
            target.clear()
            target.update(payload)
        elif entry.kind == "object":
            attributes = getattr(target, "__dict__", None)
            if not isinstance(attributes, dict) or not isinstance(payload, Mapping):
                raise _invalid_graph_entry(
                    entry, "object entries need a __dict__ target and mapping payload"
                )
            attributes.clear()
            attributes.update(payload)
        elif entry.kind in {"tuple", "frozenset"}:
            # Immutable containers cannot be restored in place; their mutable
            # members are restored by their own entries.
            continue
        else:
            raise _invalid_graph_entry(entry, f"unknown entry kind {entry.kind!r}")
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
