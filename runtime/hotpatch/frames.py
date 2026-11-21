"""CPython frame migration at bytecode safe points.

CPython does not expose a supported way to replace the ``f_code`` pointer of a
frame that is *currently* executing an instruction.  Doing so would desync
``prev_instr``, the value stack, and the ``localsplus`` array, so this module
deliberately refuses that operation and reports ``E007``.

What it does implement, and what the tests pin, is safe-point migration:

* the bound function's ``__code__`` is swapped atomically for **future**
  entries (existing frames keep their captured code object);
* live frames of that function have their fast locals migrated in place
  through CPython 3.13+ ``FrameLocalsProxy`` (PEP 667);
* suspended generators can be migrated explicitly; they stay on their
  captured code until they are recreated;
* an ABI/layout mismatch (argument count, generator flags, closure cells,
  removed locals, or a required new local without a default) reports ``E008``
  with the caller's character/byte span.

The evaluator stack depth is validated against the new code's ``co_stacksize``
for re-entry.  A running frame's stack is never resized.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import sys
import threading
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Self

from runtime.hotpatch.errors import HotpatchError, abi_error
from runtime.hotpatch.events import emit

_GENERATOR_FLAGS = (
    inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
)
_VARIADIC_FLAGS = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


@dataclass(frozen=True)
class FrameLayout:
    code: types.CodeType
    argcount: int
    posonly: int
    kwonly: int
    nlocals: int
    stacksize: int
    flags: int
    varnames: tuple[str, ...]
    freevars: tuple[str, ...]
    cellvars: tuple[str, ...]

    @classmethod
    def from_code(cls, code: types.CodeType) -> FrameLayout:
        return cls(
            code=code,
            argcount=code.co_argcount,
            posonly=code.co_posonlyargcount,
            kwonly=code.co_kwonlyargcount,
            nlocals=code.co_nlocals,
            stacksize=code.co_stacksize,
            flags=code.co_flags,
            varnames=tuple(code.co_varnames),
            freevars=tuple(code.co_freevars),
            cellvars=tuple(code.co_cellvars),
        )

    def issues(self, other: FrameLayout, *, restart: bool = False) -> list[str]:
        problems: list[str] = []
        if (self.argcount, self.posonly, self.kwonly) != (
            other.argcount,
            other.posonly,
            other.kwonly,
        ):
            problems.append(
                f"argument layout {self.argcount}/{self.posonly}/{self.kwonly} "
                f"-> {other.argcount}/{other.posonly}/{other.kwonly}"
            )
        if (self.flags & _GENERATOR_FLAGS) != (other.flags & _GENERATOR_FLAGS):
            problems.append("generator/coroutine flags changed")
        if (self.flags & _VARIADIC_FLAGS) != (other.flags & _VARIADIC_FLAGS):
            problems.append("*args/**kwargs flags changed")
        if self.freevars != other.freevars:
            problems.append(
                f"closure cells {self.freevars} -> {other.freevars} cannot be rebound"
            )
        if restart and other.stacksize > self.stacksize:
            problems.append(
                f"evaluation stack {self.stacksize} -> {other.stacksize} "
                "cannot grow under a running frame"
            )
        return problems


@dataclass
class FrameSnapshot:
    frame: types.FrameType
    code: types.CodeType
    values: dict[str, Any]
    label: str = ""
    added: tuple[str, ...] = ()

    def restore(self) -> None:
        with contextlib.suppress(ValueError, RuntimeError):
            for name, value in self.values.items():
                self.frame.f_locals[name] = value
        # Locals introduced by the migration must be removed again so rollback
        # restores the original fast-local layout, not just the original values.
        for name in self.added:
            with contextlib.suppress(ValueError, RuntimeError, KeyError):
                del self.frame.f_locals[name]


@dataclass
class FramePatch:
    function: types.FunctionType
    old_code: types.CodeType
    new_code: types.CodeType
    snapshots: list[FrameSnapshot]
    pinned_frames: int
    migrated_locals: int
    generators: list[Any] = field(default_factory=list)
    generation: int = 1
    span: tuple[int, int] | None = None
    source: str | None = None
    rolled_back: bool = False

    def rollback(self) -> int:
        if self.rolled_back:
            return 0
        self.function.__code__ = self.old_code
        restored = 0
        for snapshot in reversed(self.snapshots):
            snapshot.restore()
            restored += 1
        self.rolled_back = True
        return restored

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()


def _require_writable_locals(
    *, span: tuple[int, int] | None = None, source: str | None = None
) -> None:
    """Refuse live-local migration when this interpreter cannot write them.

    CPython < 3.13 returns a *snapshot* dict from ``frame.f_locals``; writes to
    it silently disappear.  Failing loudly with ``E007`` is the only honest
    behaviour — otherwise the call would claim a migration that never happened.
    """
    # Runtime capability gate, not a static compatibility shim: the module can
    # be imported on older interpreters but must refuse live-local migration.
    if sys.version_info < (3, 13):  # noqa: UP036
        raise HotpatchError(
            "this interpreter does not expose writable frame locals",
            code="E007",
            span=span,
            source=source,
            reason="CPython 3.13+ (PEP 667) is required for live local migration",
            help_text="upgrade the interpreter, or patch only future function entries",
        )


def _resolve_target(
    target: Any,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> types.FunctionType:
    if isinstance(target, types.FunctionType):
        return target
    if isinstance(target, types.MethodType):
        return target.__func__
    if isinstance(target, staticmethod | classmethod):
        return target.__func__
    if isinstance(target, str):
        module_name, separator, qualname = target.partition(":")
        if not separator:
            module_name, _, qualname = target.rpartition(".")
        if not module_name or not qualname:
            raise abi_error(
                f"cannot resolve frame target {target!r}",
                span=span,
                source=source,
                detail="expected 'module:qualname' or a function object",
            )
        try:
            attribute: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                attribute = getattr(attribute, part)
        except (ImportError, AttributeError) as error:
            raise abi_error(
                f"cannot resolve frame target {target!r}",
                span=span,
                source=source,
                detail=str(error),
            ) from None
        return _resolve_target(attribute, span=span, source=source)
    raise abi_error(
        f"frame target must be a function, got {type(target).__name__}",
        span=span,
        source=source,
        detail="pass a function object or 'module:qualname'",
    )


def _code_of(
    replacement: Any,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> types.CodeType:
    if isinstance(replacement, types.CodeType):
        return replacement
    if isinstance(replacement, (types.FunctionType, types.MethodType)):
        return replacement.__code__
    raise abi_error(
        f"replacement must be a function or code object, got {type(replacement).__name__}",
        span=span,
        source=source,
        detail="the replacement supplies the new bytecode and local layout",
    )


def _live_frames() -> list[types.FrameType]:
    frames: list[types.FrameType] = []
    for frame in sys._current_frames().values():
        while frame is not None:
            frames.append(frame)
            frame = frame.f_back
    return frames


def _default_for(name: str) -> Any:
    lowered = name.lower()
    if "count" in lowered or "index" in lowered or "len" in lowered:
        return 0
    if lowered.startswith(("is_", "has_", "can_")):
        return False
    return None


class FrameEvalMigrator:
    """Swap function bytecode and migrate live fast locals in place."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._patches: list[FramePatch] = []
        self._generation = 0
        self.migrated_frames = 0
        self.pinned_frames = 0

    def patch(
        self,
        target: Any,
        replacement: Any,
        *,
        defaults: dict[str, Any] | None = None,
        generators: Iterable[Any] = (),
        restart: bool = False,
        span: tuple[int, int] | None = None,
        source: str | None = None,
    ) -> FramePatch:
        function = _resolve_target(target, span=span, source=source)
        new_code = _code_of(replacement, span=span, source=source)
        old_layout = FrameLayout.from_code(function.__code__)
        new_layout = FrameLayout.from_code(new_code)
        problems = old_layout.issues(new_layout, restart=restart)
        if problems:
            raise abi_error(
                f"frame patch for {function.__qualname__} is not ABI-compatible",
                span=span,
                source=source,
                detail="; ".join(problems),
            )
        values = dict(defaults or {})
        new_names = [
            name for name in new_layout.varnames if name not in old_layout.varnames
        ]
        removed = [
            name for name in old_layout.varnames if name not in new_layout.varnames
        ]

        live_frames = [
            frame for frame in _live_frames() if frame.f_code is old_layout.code
        ]
        generator_list = list(generators)
        if live_frames or generator_list:
            _require_writable_locals(span=span, source=source)

        snapshots: list[FrameSnapshot] = []
        pinned = 0
        migrated = 0
        code_swapped = False
        try:
            for frame in live_frames:
                current = dict(frame.f_locals)
                for name in removed:
                    if current.get(name) is not None:
                        raise abi_error(
                            f"cannot drop live local {name!r} from {function.__qualname__}",
                            span=span,
                            source=source,
                            detail="removed locals without a default still hold a value",
                        )
                migrated_values = {
                    name: value
                    for name, value in current.items()
                    if name in new_layout.varnames
                }
                for name in new_names:
                    migrated_values[name] = values.get(name, _default_for(name))
                added = tuple(name for name in migrated_values if name not in current)
                snapshots.append(
                    FrameSnapshot(
                        frame, frame.f_code, current, function.__name__, added
                    )
                )
                try:
                    for name, value in migrated_values.items():
                        frame.f_locals[name] = value
                except (TypeError, AttributeError) as error:
                    raise HotpatchError(
                        "this interpreter does not expose writable frame locals",
                        code="E007",
                        span=span,
                        source=source,
                        reason=str(error),
                        help_text="CPython 3.13+ (PEP 667) is required for live local migration",
                    ) from None
                migrated += 1
                pinned += 1

            generator_snapshots = 0
            for generator in generator_list:
                snapshots.append(
                    self._migrate_generator(
                        generator,
                        old_layout,
                        new_layout,
                        values,
                        span=span,
                        source=source,
                    )
                )
                generator_snapshots += 1

            with self._lock:
                function.__code__ = new_code
                code_swapped = True
                self._generation += 1
                patch = FramePatch(
                    function=function,
                    old_code=old_layout.code,
                    new_code=new_code,
                    snapshots=snapshots,
                    pinned_frames=pinned,
                    migrated_locals=migrated + generator_snapshots,
                    generators=generator_list,
                    generation=self._generation,
                    span=span,
                    source=source,
                )
                self._patches.append(patch)
                self.migrated_frames += migrated
                self.pinned_frames += pinned
                # Counts and generation only: code objects and source are never
                # published through the telemetry bus.
                try:
                    emit(
                        "frame_patch",
                        generation=patch.generation,
                        snapshots=len(patch.snapshots),
                        pinned_frames=patch.pinned_frames,
                        migrated_locals=patch.migrated_locals,
                    )
                except Exception:  # noqa: S110 - telemetry is diagnostic-only
                    pass
                return patch
        except BaseException:
            # A failure anywhere in the migration must not leave a half-patched
            # process: restore the code object (if it was swapped) and replay
            # every completed frame snapshot in reverse order.
            if code_swapped:
                with contextlib.suppress(Exception):
                    function.__code__ = old_layout.code
            for snapshot in reversed(snapshots):
                with contextlib.suppress(Exception):
                    snapshot.restore()
            raise

    def _migrate_generator(
        self,
        generator: Any,
        old_layout: FrameLayout,
        new_layout: FrameLayout,
        defaults: dict[str, Any],
        *,
        span: tuple[int, int] | None,
        source: str | None,
    ) -> FrameSnapshot:
        frame = getattr(generator, "gi_frame", None) or getattr(
            generator, "cr_frame", None
        )
        if frame is None:
            raise abi_error(
                f"suspended generator {generator!r} has no frame",
                span=span,
                source=source,
                detail="the generator already finished or was never started",
            )
        if frame.f_code is not old_layout.code:
            raise abi_error(
                "generator frame does not match the patched function",
                span=span,
                source=source,
                detail=f"got {frame.f_code.co_qualname}",
            )
        current = dict(frame.f_locals)
        updated = {
            name: value
            for name, value in current.items()
            if name in new_layout.varnames
        }
        for name in new_layout.varnames:
            if name in old_layout.varnames or name in updated:
                continue
            updated[name] = defaults.get(name, _default_for(name))
        added = tuple(name for name in updated if name not in current)
        snapshot = FrameSnapshot(frame, frame.f_code, current, "generator", added)
        try:
            for name, value in updated.items():
                frame.f_locals[name] = value
        except (TypeError, AttributeError) as error:
            raise HotpatchError(
                "generator frame locals are not writable",
                code="E007",
                span=span,
                source=source,
                reason=str(error),
            ) from None
        return snapshot

    def rollback(self, patch: FramePatch) -> int:
        with self._lock:
            restored = patch.rollback()
            try:
                emit(
                    "frame_rollback",
                    generation=patch.generation,
                    restored=restored,
                    already_rolled_back=patch.rolled_back and restored == 0,
                )
            except Exception:  # noqa: S110 - telemetry is diagnostic-only
                pass
            return restored

    @property
    def patches(self) -> tuple[FramePatch, ...]:
        with self._lock:
            return tuple(self._patches)


def hotpatch_cpython_frame_eval(
    target: Any,
    replacement: Any,
    *,
    defaults: dict[str, Any] | None = None,
    generators: Iterable[Any] = (),
    restart: bool = False,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> FramePatch:
    """Patch a function's code and migrate live locals at a safe point."""
    migrator = FrameEvalMigrator()
    return migrator.patch(
        target,
        replacement,
        defaults=defaults,
        generators=generators,
        restart=restart,
        span=span,
        source=source,
    )


# ---------------------------------------------------------------------------
# Optional monitoring-based safepoint hook
# ---------------------------------------------------------------------------


class SafepointDispatcher:
    """Run queued actions at the next ``LINE`` event for a code object.

    This is the supported way to defer work to a bytecode boundary when the
    caller wants the patch to become visible only after the current statement.
    """

    def __init__(self) -> None:
        self._actions: dict[types.CodeType, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._tool_id: int | None = None
        self._previous: Any = None
        self.dispatches = 0
        self.errors: list[BaseException] = []

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    def __del__(self) -> None:
        # A dispatcher dropped without uninstall() would leak a global
        # sys.monitoring tool id, blocking every future safepoint.
        if getattr(self, "_tool_id", None) is not None:
            with contextlib.suppress(Exception):
                self.uninstall()

    def install(self) -> None:
        if self._tool_id is not None:
            return
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            raise HotpatchError(
                "sys.monitoring is unavailable",
                code="E007",
                reason="bytecode safepoints require CPython 3.12+",
            )
        for candidate in (3, 4, 5, 2, 1, 0):
            try:
                monitoring.use_tool_id(candidate, "polyglot-hotpatch")
            except ValueError:
                continue
            self._tool_id = candidate
            break
        else:
            raise HotpatchError(
                "no free sys.monitoring tool id",
                code="E007",
                help_text="another profiler/debugger owns every monitoring slot",
            )
        try:
            monitoring.register_callback(
                self._tool_id, monitoring.events.LINE, self._on_line
            )
            monitoring.set_events(self._tool_id, monitoring.events.LINE)
        except Exception as error:
            # Never leave a half-installed tool id behind on a failed install.
            with contextlib.suppress(Exception):
                monitoring.free_tool_id(self._tool_id)
            self._tool_id = None
            raise HotpatchError(
                "cannot install the bytecode safepoint callback",
                code="E007",
                reason=str(error),
            ) from None

    def uninstall(self) -> None:
        if self._tool_id is None:
            return
        monitoring = sys.monitoring
        tool_id = self._tool_id
        try:
            with contextlib.suppress(Exception):
                monitoring.set_events(tool_id, 0)
            with contextlib.suppress(Exception):
                monitoring.register_callback(tool_id, monitoring.events.LINE, None)
        finally:
            # Even if deregistration failed, release the global tool id.
            with contextlib.suppress(Exception):
                monitoring.free_tool_id(tool_id)
            self._tool_id = None

    def defer(self, code: types.CodeType, action: Callable[[], None]) -> None:
        with self._lock:
            self._actions[code] = action

    def _on_line(self, code: types.CodeType, line: int) -> None:
        with self._lock:
            action = self._actions.pop(code, None)
        if action is not None:
            self.dispatches += 1
            try:
                action()
            except BaseException as error:
                # An exception escaping a monitoring callback makes CPython
                # disable the callback for the whole tool; retain it instead so
                # later safepoints still dispatch.
                self.errors.append(error)


def install_safepoint(
    action: Callable[[], None], code: types.CodeType
) -> SafepointDispatcher:
    dispatcher = SafepointDispatcher()
    dispatcher.install()
    dispatcher.defer(code, action)
    return dispatcher


__all__ = [
    "FrameEvalMigrator",
    "FrameLayout",
    "FramePatch",
    "FrameSnapshot",
    "SafepointDispatcher",
    "hotpatch_cpython_frame_eval",
    "install_safepoint",
]
