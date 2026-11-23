"""Frame A: JIT compilation and speculative execution.

The frame is a deterministic state machine around the existing AArch64 JIT
backend.  It deliberately separates *speculation* (hotness counters, guards,
deoptimization, inline caches) from *privileged execution* (``MAP_JIT`` pages
and W^X transitions), which live in :mod:`backends.jit.memory`.

Platform and security boundaries
---------------------------------
* Native code generation is attempted only when ``allow_native=True`` **and**
  :func:`backends.jit.memory.is_supported_host` reports macOS on arm64.
  Otherwise a fully functional model record is produced, so tests and
  cross-platform callers can exercise the whole state machine.
* Executable pages are allocated only through
  :func:`backends.jit.allocate_executable`, which uses ``MAP_JIT`` and the
  per-thread write-protect toggle.  Callers must opt in with
  ``opt_in_executable_memory=True``.  Stale blocks are unmapped only when no
  active frame pins them.
* Every malformed input, guard-exhaustion event, or codegen failure raises a
  structured ``PillarError``/``PolyglotError`` ``E008`` with a source span via
  :func:`jit_emit_diagnostic_e008`; no raw exception escapes this frame.

Diagnostic map for this module: ``E008`` for speculative/codegen failures.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from backends.jit.errors import JITError
from backends.jit.memory import allocate_executable, is_supported_host
from runtime.diagnostics import PillarError

Span = tuple[int, int] | None
ExecutionMode = Literal["model", "native"]
TraceState = Literal["cold", "compiled", "deoptimized", "invalidated"]
GuardKind = Literal["type", "non_null", "bounds"]

DEFAULT_HOTNESS_THRESHOLD = 8
MAX_GUARDS_PER_TRACE = 64
MAX_DEOPT_COUNT = 32

_LOCK_CREATION = threading.Lock()


def jit_emit_diagnostic_e008(
    message: str,
    *,
    span: Span = None,
    reason: str | None = None,
    help_text: str | None = None,
    trace_key: str | None = None,
) -> PillarError:
    """Raise a structured ``E008`` JIT diagnostic carrying ``span``."""
    note = "runtime.jit/speculative-execution"
    if trace_key:
        note += f"/{trace_key}"
    return PillarError(
        "E008",
        message,
        reason=reason or message,
        help_text=help_text
        or "recompile the trace or use the Python baseline for this construct",
        note=note,
        span=span,
    )


@dataclass(frozen=True)
class GuardSpec:
    """One speculative assumption that must hold before the fast path runs."""

    kind: GuardKind
    target: str
    expected: Any = None
    lower: int | None = None
    upper: int | None = None
    span: Span = None

    def describe(self) -> str:
        if self.kind == "type":
            return f"{self.target} is {self.expected!r}"
        if self.kind == "non_null":
            return f"{self.target} is not null"
        return f"{self.target} in [{self.lower}, {self.upper})"


@dataclass(frozen=True)
class GuardFailure:
    """A single guard that did not hold in a speculative execution."""

    guard: GuardSpec
    observed: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.guard.kind,
            "target": self.guard.target,
            "expected": self.guard.expected,
            "lower": self.guard.lower,
            "upper": self.guard.upper,
            "observed": self.observed,
            "span": list(self.guard.span) if self.guard.span else None,
        }


@dataclass(frozen=True)
class GuardReport:
    """Outcome of checking a trace's guards against concrete values."""

    passed: bool
    failures: tuple[GuardFailure, ...] = ()

    @property
    def reason(self) -> str:
        if self.passed:
            return "guards satisfied"
        return "; ".join(failure.guard.describe() for failure in self.failures)


@dataclass
class IcSite:
    """A versioned inline-cache/direct-call entry."""

    site_id: str
    target: str
    version: int
    kind: Literal["inline_cache", "direct_call", "indirect_call"] = "inline_cache"
    pinned_by: frozenset[str] = frozenset()
    deferred_target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "target": self.target,
            "version": self.version,
            "kind": self.kind,
            "pinned_by": sorted(self.pinned_by),
            "deferred_target": self.deferred_target,
        }


@dataclass
class CodeBlock:
    """Owning record for one native (or modelled) code region."""

    block_id: str
    size: int
    digest: str
    mode: ExecutionMode
    protection: str
    writes_in_progress: int = 0
    region: Any | None = None
    pinned_by: frozenset[str] = frozenset()
    released: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "size": self.size,
            "digest": self.digest,
            "mode": self.mode,
            "protection": self.protection,
            "writes_in_progress": self.writes_in_progress,
            "pinned_by": sorted(self.pinned_by),
            "released": self.released,
        }


@dataclass
class JitTraceRecord:
    """The speculative state of one hot trace/function."""

    key: str
    source: str
    signature: tuple[str, ...]
    threshold: int
    hotness: int
    version: int
    state: TraceState
    mode: ExecutionMode
    guards: tuple[GuardSpec, ...] = ()
    block_id: str | None = None
    deopt_count: int = 0
    last_deopt_reason: str | None = None
    span: Span = None
    module: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "signature": list(self.signature),
            "threshold": self.threshold,
            "hotness": self.hotness,
            "version": self.version,
            "state": self.state,
            "mode": self.mode,
            "guards": [guard.describe() for guard in self.guards],
            "block_id": self.block_id,
            "deopt_count": self.deopt_count,
            "last_deopt_reason": self.last_deopt_reason,
            "span": list(self.span) if self.span else None,
        }


@dataclass
class DeoptFrame:
    """A safe bailout target preserving live values and stack state."""

    key: str
    version: int
    reason: str
    live_values: dict[str, Any]
    stack: tuple[Any, ...]
    guard_failures: tuple[GuardFailure, ...]
    span: Span
    restored_by: str = "interpreter"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "reason": self.reason,
            "live_values": dict(self.live_values),
            "stack": list(self.stack),
            "guard_failures": [item.as_dict() for item in self.guard_failures],
            "span": list(self.span) if self.span else None,
            "restored_by": self.restored_by,
        }


@dataclass
class InvalidationReport:
    """Result of a definition-change or hotpatch invalidation."""

    key: str
    old_version: int
    new_version: int
    epoch: int
    invalidated_traces: tuple[str, ...]
    cleared_sites: tuple[str, ...]
    reclaimed_blocks: tuple[str, ...]
    deferred_blocks: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "epoch": self.epoch,
            "invalidated_traces": list(self.invalidated_traces),
            "cleared_sites": list(self.cleared_sites),
            "reclaimed_blocks": list(self.reclaimed_blocks),
            "deferred_blocks": list(self.deferred_blocks),
            "reason": self.reason,
        }


@dataclass
class JitState:
    """Process-local JIT state; one instance per isolated compilation scope."""

    traces: dict[str, JitTraceRecord] = field(default_factory=dict)
    blocks: dict[str, CodeBlock] = field(default_factory=dict)
    sites: dict[str, IcSite] = field(default_factory=dict)
    epochs: dict[str, int] = field(default_factory=dict)
    active_frames: dict[str, set[str]] = field(default_factory=dict)
    _locks: dict[str, threading.RLock] = field(default_factory=dict, repr=False)

    def lock_for(self, key: str) -> threading.RLock:
        with _LOCK_CREATION:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def as_dict(self) -> dict[str, Any]:
        return {
            "traces": {key: value.as_dict() for key, value in self.traces.items()},
            "blocks": {key: value.as_dict() for key, value in self.blocks.items()},
            "sites": {key: value.as_dict() for key, value in self.sites.items()},
            "epochs": dict(self.epochs),
        }


def new_jit_state() -> JitState:
    """Return an empty, isolated :class:`JitState`."""
    return JitState()


def _normalize_span(span: Span) -> Span:
    if span is None:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError) as error:
        raise jit_emit_diagnostic_e008(
            "invalid source span for JIT trace", reason=str(error)
        ) from error
    if start < 0 or end < start:
        raise jit_emit_diagnostic_e008(
            "source span must satisfy 0 <= start <= end", span=(start, end)
        )
    return (start, end)


def _guard_specs_from_signature(
    key: str,
    signature: Sequence[str],
    span: Span,
    *,
    bounds: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[GuardSpec, ...]:
    guards: list[GuardSpec] = [
        GuardSpec("type", key, "function", span=span),
    ]
    for index, type_name in enumerate(signature):
        guards.append(GuardSpec("type", f"arg{index}", type_name, span=span))
        if type_name in {"slice", "array", "str"}:
            guards.append(GuardSpec("non_null", f"arg{index}", span=span))
    for target, (lower, upper) in sorted((bounds or {}).items()):
        guards.append(GuardSpec("bounds", target, lower=lower, upper=upper, span=span))
    if len(guards) > MAX_GUARDS_PER_TRACE:
        raise jit_emit_diagnostic_e008(
            f"trace {key!r} needs {len(guards)} guards (limit {MAX_GUARDS_PER_TRACE})",
            span=span,
            trace_key=key,
        )
    return tuple(guards)


def jit_compile_hot_trace(
    state: JitState,
    *,
    key: str,
    source: str,
    signature: Sequence[str] = (),
    span: Span = None,
    threshold: int = DEFAULT_HOTNESS_THRESHOLD,
    bounds: Mapping[str, tuple[int, int]] | None = None,
    allow_native: bool = False,
    opt_in_executable_memory: bool = False,
) -> JitTraceRecord:
    """Record one execution of ``key`` and compile it once it becomes hot.

    The function is pure with respect to ``state`` until ``hotness`` reaches
    ``threshold``.  Past the threshold it compiles through the real JIT backend
    when explicitly allowed on a supported host; otherwise it records an
    equivalent *model* trace so speculative behaviour remains testable
    everywhere.  A substitution of the same ``key`` with a different source
    raises ``E008`` rather than silently serving stale code.
    """
    if not key.strip():
        raise jit_emit_diagnostic_e008("trace key must not be empty", span=span)
    if threshold <= 0:
        raise jit_emit_diagnostic_e008(
            "hotness threshold must be positive", span=span, trace_key=key
        )
    resolved_span = _normalize_span(span)
    guards = _guard_specs_from_signature(key, signature, resolved_span, bounds=bounds)
    with state.lock_for(key):
        existing = state.traces.get(key)
        if existing is not None:
            if existing.source != source or existing.signature != tuple(signature):
                raise jit_emit_diagnostic_e008(
                    f"trace {key!r} was redefined with a different shape",
                    span=resolved_span,
                    trace_key=key,
                    reason="use jit_invalidate_on_redefinition_or_hotpatch first",
                )
            if existing.state in {"invalidated", "deoptimized"}:
                existing = replace(
                    existing,
                    state="cold",
                    hotness=0,
                    version=existing.version + 1,
                    guards=guards,
                    span=resolved_span,
                    module=None,
                    block_id=None,
                    last_deopt_reason=None,
                )
                state.traces[key] = existing
            record = replace(existing, hotness=existing.hotness + 1)
        else:
            record = JitTraceRecord(
                key=key,
                source=source,
                signature=tuple(signature),
                threshold=threshold,
                hotness=1,
                version=state.epochs.get(key, 0) + 1,
                state="cold",
                mode="model",
                guards=guards,
                span=resolved_span,
            )
        state.traces[key] = record
        if record.hotness < record.threshold or record.state == "compiled":
            return record

        mode: ExecutionMode = "model"
        module: Any = None
        if allow_native and is_supported_host():
            try:
                from backends.jit.backend import JITBackend
                from compiler.core.lexer import lex
                from compiler.core.parser import parse

                module = JITBackend(trusted=True).compile(parse(lex(source)))
            except JITError as error:
                raise jit_emit_diagnostic_e008(
                    f"native codegen failed for trace {key!r}: {error}",
                    span=resolved_span,
                    trace_key=key,
                    reason=str(error),
                ) from error
            mode = "native"
        block_id = f"{key}:v{record.version}"
        jit_manage_executable_memory_rx(
            state,
            block_id=block_id,
            code=b"",  # JITModule owns its own MAP_JIT region
            mode=mode,
            span=resolved_span,
            opt_in_executable_memory=opt_in_executable_memory,
            region=getattr(module, "memory", None),
        )
        compiled = replace(
            record,
            state="compiled",
            mode=mode,
            module=module,
            block_id=block_id,
        )
        state.traces[key] = compiled
        return compiled


def jit_emit_type_and_bounds_guards(
    values: Mapping[str, Any],
    *,
    guards: Sequence[GuardSpec],
    span: Span = None,
) -> GuardReport:
    """Evaluate the guards for one speculative fast-path entry.

    Returns a :class:`GuardReport` instead of raising on a failed check so the
    caller can deoptimize with a precise reason.  Malformed guard definitions
    are programming errors and raise ``E008``.
    """
    failures: list[GuardFailure] = []
    for guard in guards:
        if guard.kind not in {"type", "non_null", "bounds"}:
            raise jit_emit_diagnostic_e008(
                f"unknown guard kind {guard.kind!r}", span=guard.span or span
            )
        if guard.target not in values:
            raise jit_emit_diagnostic_e008(
                f"guard target {guard.target!r} has no value",
                span=guard.span or span,
            )
        observed = values[guard.target]
        if guard.kind == "type":
            expected = guard.expected
            if expected == "i64" or expected == "u64":
                ok = isinstance(observed, int) and not isinstance(observed, bool)
            elif expected == "f64":
                ok = isinstance(observed, float)
            elif expected == "bool":
                ok = isinstance(observed, bool)
            elif expected == "str":
                ok = isinstance(observed, str)
            elif expected in {"slice", "array", "list", "tuple"}:
                ok = isinstance(observed, (list, tuple))
            elif expected == "function":
                ok = callable(observed)
            else:
                ok = type(observed).__name__ == expected
        elif guard.kind == "non_null":
            ok = observed is not None
        else:
            if guard.lower is None or guard.upper is None:
                raise jit_emit_diagnostic_e008(
                    f"bounds guard {guard.target!r} needs lower and upper bounds",
                    span=guard.span or span,
                )
            if guard.upper <= guard.lower:
                raise jit_emit_diagnostic_e008(
                    f"bounds guard {guard.target!r} has an empty range",
                    span=guard.span or span,
                )
            try:
                length = len(observed)  # type: ignore[arg-type]
            except TypeError:
                ok = False
            else:
                ok = guard.lower <= length < guard.upper
        if not ok:
            failures.append(GuardFailure(guard=guard, observed=observed))
    return GuardReport(passed=not failures, failures=tuple(failures))


def jit_deoptimize_to_interpreter_frame(
    state: JitState,
    *,
    key: str,
    guard_report: GuardReport,
    live_values: Mapping[str, Any],
    stack: Sequence[Any] = (),
    span: Span = None,
) -> DeoptFrame:
    """Bail out to the baseline interpreter without losing live state."""
    with state.lock_for(key):
        trace = state.traces.get(key)
        if trace is None:
            raise jit_emit_diagnostic_e008(
                f"cannot deoptimize unknown trace {key!r}",
                span=span,
                trace_key=key,
            )
        if guard_report.passed:
            raise jit_emit_diagnostic_e008(
                f"cannot deoptimize {key!r}: all guards passed",
                span=span,
                trace_key=key,
            )
        if trace.deopt_count >= MAX_DEOPT_COUNT:
            raise jit_emit_diagnostic_e008(
                f"guard exhaustion for trace {key!r} "
                f"({trace.deopt_count} deoptimizations)",
                span=span,
                trace_key=key,
                reason="the trace is too unstable to keep compiling",
            )
        frame = DeoptFrame(
            key=key,
            version=trace.version,
            reason=guard_report.reason,
            live_values=dict(live_values),
            stack=tuple(stack),
            guard_failures=guard_report.failures,
            span=trace.span or _normalize_span(span),
        )
        state.traces[key] = replace(
            trace,
            state="deoptimized",
            deopt_count=trace.deopt_count + 1,
            last_deopt_reason=frame.reason,
        )
        return frame


def jit_manage_executable_memory_rx(
    state: JitState,
    *,
    block_id: str,
    code: bytes,
    mode: ExecutionMode = "model",
    span: Span = None,
    opt_in_executable_memory: bool = False,
    region: Any | None = None,
    active_frames: Sequence[str] = (),
) -> CodeBlock:
    """Allocate, transition, or evict an executable code block.

    Native allocation is only attempted with ``mode='native'`` **and**
    ``opt_in_executable_memory=True`` on a supported host.  ``region`` lets a
    caller hand over an existing ``ExecutableMemory`` (as ``JITModule`` does)
    so the frame can own its lifetime without remapping it.  A block pinned by
    an active frame is never released.
    """
    resolved_span = _normalize_span(span)
    with state.lock_for(block_id):
        existing = state.blocks.get(block_id)
        if existing is not None:
            if (
                existing.mode != mode
                or existing.digest != hashlib.sha256(code).hexdigest()
            ):
                raise jit_emit_diagnostic_e008(
                    f"code block {block_id!r} changed shape in place",
                    span=resolved_span,
                    reason="allocate a new version instead of mutating a live block",
                )
            if active_frames:
                existing.pinned_by = existing.pinned_by | frozenset(active_frames)
            return existing

        size = max(1, len(code))
        digest = hashlib.sha256(code).hexdigest()
        owned: Any | None = None
        if mode == "native" and opt_in_executable_memory:
            if not is_supported_host():
                raise jit_emit_diagnostic_e008(
                    "executable JIT memory is unavailable on this host",
                    span=resolved_span,
                    reason="MAP_JIT is macOS/arm64 only (model mode still available)",
                )
            if region is None:
                try:
                    owned = allocate_executable(code)
                except Exception as error:
                    raise jit_emit_diagnostic_e008(
                        f"MAP_JIT allocation failed for block {block_id!r}: {error}",
                        span=resolved_span,
                        reason=str(error),
                    ) from error
            else:
                owned = region
            protection = "r-x (W^X after commit)"
        else:
            protection = "model (no executable mapping)"
        block = CodeBlock(
            block_id=block_id,
            size=size,
            digest=digest,
            mode=mode,
            protection=protection,
            region=owned,
            pinned_by=frozenset(active_frames),
        )
        state.blocks[block_id] = block
        return block


def release_executable_memory_block(
    state: JitState, *, block_id: str, active_frames: Sequence[str] = ()
) -> bool:
    """Release a stale block when no active frame pins it.

    Returns ``True`` when the block was released.  This helper is kept separate
    from invalidation so tests can exercise pinned-frame deferral explicitly.
    """
    with state.lock_for(block_id):
        block = state.blocks.get(block_id)
        if block is None or block.released:
            return False
        pins = block.pinned_by | frozenset(active_frames)
        if pins:
            block.pinned_by = pins
            return False
        if block.region is not None:
            close = getattr(block.region, "close", None)
            if callable(close):
                close()
        block.released = True
        block.region = None
        return True


def jit_patch_ic_and_call_sites(
    state: JitState,
    *,
    site_id: str,
    target: str,
    version: int,
    kind: Literal["inline_cache", "direct_call", "indirect_call"] = "inline_cache",
    active_frames: Sequence[str] = (),
    span: Span = None,
) -> IcSite:
    """Atomically publish a new inline-cache/call target.

    Copy-on-write under a per-site lock means readers holding an old site object
    continue safely.  If an active frame still pins the old target, the patch is
    deferred (``deferred_target``) instead of mutating live code.
    """
    if version < 0:
        raise jit_emit_diagnostic_e008(
            "call-site version must not be negative", span=span
        )
    with state.lock_for(site_id):
        existing = state.sites.get(site_id)
        site = existing or IcSite(
            site_id=site_id, target=target, version=version, kind=kind
        )
        if existing is not None and existing.version > version:
            raise jit_emit_diagnostic_e008(
                f"stale patch for {site_id!r}: version {version} < {existing.version}",
                span=span,
                reason="a newer patch was already published",
            )
        pins = frozenset(active_frames)
        if pins:
            published = replace(
                site,
                pinned_by=site.pinned_by | pins,
                deferred_target=target,
            )
        else:
            published = replace(
                site, target=target, version=version, kind=kind, deferred_target=None
            )
        state.sites[site_id] = published
        return published


def jit_invalidate_on_redefinition_or_hotpatch(
    state: JitState,
    *,
    key: str,
    new_version: int | None = None,
    reason: str,
    hotpatch_event: Mapping[str, Any] | None = None,
    active_frames: Sequence[str] = (),
    released_frames: Sequence[str] = (),
    span: Span = None,
) -> InvalidationReport:
    """Invalidate dependent code versions after a redefinition or hotpatch."""
    resolved_span = _normalize_span(span)
    if not reason.strip():
        raise jit_emit_diagnostic_e008(
            "invalidation requires a reason", span=resolved_span, trace_key=key
        )
    if hotpatch_event is not None and "target" in hotpatch_event:
        event_target = str(hotpatch_event["target"])
        if event_target != key:
            raise jit_emit_diagnostic_e008(
                f"hotpatch target {event_target!r} does not match trace {key!r}",
                span=resolved_span,
                trace_key=key,
            )
    with state.lock_for(key):
        trace = state.traces.get(key)
        old_version = trace.version if trace is not None else state.epochs.get(key, 0)
        resolved_version = new_version if new_version is not None else old_version + 1
        if resolved_version <= old_version:
            raise jit_emit_diagnostic_e008(
                f"invalidation must advance the version ({resolved_version} "
                f"<= {old_version})",
                span=resolved_span,
                trace_key=key,
            )
        epoch = state.epochs.get(key, 0) + 1
        state.epochs[key] = epoch
        invalidated: list[str] = []
        if trace is not None:
            state.traces[key] = replace(
                trace, state="invalidated", version=resolved_version, module=None
            )
            invalidated.append(key)

        cleared: list[str] = []
        for site_id, site in list(state.sites.items()):
            if site.target == key or site.deferred_target == key:
                state.sites[site_id] = replace(
                    site, version=resolved_version, deferred_target=key
                )
                cleared.append(site_id)

        reclaimed: list[str] = []
        deferred: list[str] = []
        if trace is not None and trace.block_id is not None:
            block = state.blocks.get(trace.block_id)
            if block is not None and released_frames:
                block.pinned_by = block.pinned_by - frozenset(released_frames)
            pins = frozenset(active_frames)
            if release_executable_memory_block(
                state, block_id=trace.block_id, active_frames=sorted(pins)
            ):
                reclaimed.append(trace.block_id)
            else:
                deferred.append(trace.block_id)

        return InvalidationReport(
            key=key,
            old_version=old_version,
            new_version=resolved_version,
            epoch=epoch,
            invalidated_traces=tuple(invalidated),
            cleared_sites=tuple(cleared),
            reclaimed_blocks=tuple(reclaimed),
            deferred_blocks=tuple(deferred),
            reason=reason,
        )


__all__ = [
    "CodeBlock",
    "DeoptFrame",
    "GuardFailure",
    "GuardReport",
    "GuardSpec",
    "IcSite",
    "InvalidationReport",
    "JitState",
    "JitTraceRecord",
    "jit_compile_hot_trace",
    "jit_deoptimize_to_interpreter_frame",
    "jit_emit_diagnostic_e008",
    "jit_emit_type_and_bounds_guards",
    "jit_invalidate_on_redefinition_or_hotpatch",
    "jit_manage_executable_memory_rx",
    "jit_patch_ic_and_call_sites",
    "new_jit_state",
    "release_executable_memory_block",
]
