"""FRAME 6: hot-patching and live runtime state framework functions.

The ten public functions in this module are the framework-level entry points
for capabilities 51-60.  They compose the existing hot-patch primitives
(``trampoline``, ``frames``, ``reactor``, ``rcu``, ``journal`` and ``errors``)
with the capability registry, the legacy ``backends.hotpatch`` manager and the
shared E001-E008 diagnostic pipeline.

Honest boundaries are part of the contract:

* No foreign executable text is ever rewritten.  A request that targets
  loaded-image code is refused with E007 through :func:`errors.unsupported`.
* Raw addresses are only *model state keys*.  This module never dereferences
  an arbitrary pointer and never changes a page protection flag.  The TOCTOU
  guard narrows the race inside one in-process lock; it cannot control writers
  outside that model.
* Frame-evaluation hook management is a model of CPython's ``sys.monitoring``
  tool-id slots.  It validates version and slot limits but installs no global
  callback and never changes a ``f_code`` pointer.
* RCU epochs are cooperative user-space epochs, not kernel RCU syscalls.

Every result type carries a ``model_only`` flag so callers can distinguish a
real selector migration or journal commit from a deterministic model.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import selectors
import sys
import threading
import types
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ast_nodes import Program
from backends import hotpatch as legacy_hotpatch
from runtime import capabilities
from runtime.diagnostics import PillarError
from runtime.hotpatch.errors import (
    ByteSpan,
    HotpatchError,
    abi_error,
    relocation_error,
    unsupported,
)
from runtime.hotpatch.frames import FrameLayout
from runtime.hotpatch.journal import RollbackJournal, hotpatch_rollback_journal_commit
from runtime.hotpatch.rcu import AstEpochManager, RcuDomain, RcuStats
from runtime.hotpatch.reactor import (
    Registration,
    capture_selector,
    hotpatch_mio_reactor_state,
)
from runtime.hotpatch.trampoline import parse_signature, resolve_symbol

_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$@.]*$")
_ADDRESS_MODEL: MutableMapping[int, Mapping[str, Any]] = {}
_ADDRESS_LOCK = threading.RLock()
_ADDRESS_GENERATION = [0]
_TRANSACTION_LOCK = threading.RLock()
_TRANSACTIONS: dict[str, PatchTransaction] = {}
_MISSING = object()


# ---------------------------------------------------------------------------
# 51. Registry lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityResolution:
    """The active handler and metadata for one capability registry entry."""

    name: str
    number: int
    status: str
    backend: str
    handler: Callable[..., Any] | None
    active: bool
    code: str
    reason: str
    boundary: str
    model_only: bool


def hotpatch_registry_lookup_capability(
    capability_id: str | int,
    *,
    backend: str = "python",
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> CapabilityResolution:
    """Resolve a capability ID (name or 1-based number) to its active handler.

    ``runtime.capabilities`` remains the single source of truth: this function
    reads ``BY_NAME``/``CAPABILITIES`` and calls ``resolve()`` so a capability
    with a missing adapter is reported as inactive rather than invoked.
    REJECTED entries return an inactive resolution carrying their E007 boundary;
    unknown IDs raise the registry's canonical E006 rejection.

    Model-only status is inherited from the registry: ``MODELLED`` entries set
    ``model_only=True``.  No platform primitive is touched by the lookup.
    """
    del source  # registry diagnostics use character spans only
    if backend not in ("python", "native"):
        raise PillarError(
            "E003",
            f"unknown backend {backend!r}",
            reason="capabilities are exposed on 'python' and 'native' only",
            span=span,
        )
    if isinstance(capability_id, bool) or not isinstance(capability_id, (str, int)):
        raise PillarError(
            "E006",
            f"unknown capability ID of type {type(capability_id).__name__}",
            reason="pass a capability name or a 1-based capability number",
            span=span,
        )
    if isinstance(capability_id, int):
        if not 1 <= capability_id <= len(capabilities.CAPABILITIES):
            raise PillarError(
                "E006",
                f"no capability numbered {capability_id}",
                reason="the registry defines capabilities 1..100",
                span=span,
            )
        cap = capabilities.CAPABILITIES[capability_id - 1]
    else:
        cap = capabilities.BY_NAME.get(capability_id)
        if cap is None:
            raise capabilities.reject(capability_id, backend=backend, span=span)

    handler = capabilities.resolve(cap) if cap.supported else None
    active = handler is not None
    if cap.status is capabilities.CapabilityStatus.REJECTED:
        reason = f"registry status is rejected: {cap.boundary}"
    elif handler is None:
        reason = f"no callable adapter for {cap.module}.{cap.attr}"
    else:
        reason = f"active {cap.status.value} handler resolved from {cap.module}"
    return CapabilityResolution(
        name=cap.name,
        number=cap.number,
        status=cap.status.value,
        backend=backend,
        handler=handler,
        active=active,
        code=cap.code,
        reason=reason,
        boundary=cap.boundary,
        model_only=cap.status is capabilities.CapabilityStatus.MODELLED,
    )


# ---------------------------------------------------------------------------
# 52. ABI validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbiValidation:
    """Result of a pre-patch symbol/layout/version validation."""

    symbol: str
    signature: str | None
    fields: tuple[str, ...]
    version_token: int | str | None
    preconditions: tuple[str, ...]
    ok: bool
    model_only: bool
    boundary: str


def hotpatch_validate_target_abi(
    symbol: str,
    *,
    signature: str | None = None,
    expected_signature: str | None = None,
    layout: Mapping[str, int] | None = None,
    expected_layout: Mapping[str, int] | None = None,
    version_token: int | str | None = None,
    expected_version: int | str | None = None,
    allow_layout_growth: bool = False,
    foreign_text: bool = False,
    require_resolved: bool = False,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> AbiValidation:
    """Validate symbol shape, ABI signature, layout and version before patching.

    Signature parsing and the optional ``dlsym`` probe use the real trampoline
    helpers.  Layout maps and version tokens are in-process model checks
    (``model_only=True``): no object bytes, executable text or page table is
    inspected.  ``allow_layout_growth`` permits fields appended at the end but
    never a renamed, removed or resized field.

    ``foreign_text=True`` is refused with E007: rewriting a loaded image's
    executable text is outside the documented platform boundary.  Malformed
    symbols, ABI mismatches, layout drift and stale tokens raise E008 before
    anything is mutated.
    """
    if foreign_text:
        raise unsupported("foreign executable text patching", span=span)
    if not symbol or not _SYMBOL_RE.fullmatch(symbol):
        raise abi_error(
            f"invalid hot-patch symbol {symbol!r}",
            span=span,
            source=source,
            detail="expected a C-style identifier such as 'polyglot_dispatch'",
        )
    preconditions: list[str] = [f"symbol {symbol!r} has a valid identifier shape"]

    normalised_signature: str | None = None
    if signature is not None:
        parsed = parse_signature(signature)
        normalised_signature = signature.replace(" ", "")
        preconditions.append(f"signature {normalised_signature} parses")
        if expected_signature is not None:
            expected = parse_signature(expected_signature)
            if parsed != expected:
                raise abi_error(
                    f"ABI signature mismatch for {symbol!r}",
                    span=span,
                    source=source,
                    detail=(
                        f"expected {expected_signature.replace(' ', '')}, "
                        f"got {normalised_signature}"
                    ),
                )
            preconditions.append("replacement signature matches the target")
    elif expected_signature is not None:
        parse_signature(expected_signature)
        preconditions.append("expected signature parses")

    current_fields: tuple[str, ...] = ()
    if layout is not None or expected_layout is not None:
        actual = dict(layout or {})
        expected_fields = dict(expected_layout or {})
        for name, size in actual.items():
            if not isinstance(name, str) or not _SYMBOL_RE.fullmatch(name):
                raise abi_error(
                    f"invalid layout field {name!r}",
                    span=span,
                    source=source,
                    detail="layout keys must be field identifiers",
                )
            if not isinstance(size, int) or size < 0:
                raise abi_error(
                    f"invalid size {size!r} for layout field {name!r}",
                    span=span,
                    source=source,
                    detail="field sizes are non-negative byte counts",
                )
        current_fields = tuple(actual)
        if expected_fields:
            retained = tuple(name for name in actual if name in expected_fields)
            if retained != tuple(expected_fields):
                raise abi_error(
                    f"layout field order changed for {symbol!r}",
                    span=span,
                    source=source,
                    detail=(
                        f"expected prefix {tuple(expected_fields)}, got {retained}"
                    ),
                )
            for name, size in expected_fields.items():
                if actual.get(name) != size:
                    raise abi_error(
                        f"layout field {name!r} changed size for {symbol!r}",
                        span=span,
                        source=source,
                        detail=f"expected {size} bytes, got {actual.get(name)}",
                    )
            added = tuple(name for name in actual if name not in expected_fields)
            if added and not allow_layout_growth:
                raise abi_error(
                    f"unexpected layout fields for {symbol!r}: {added}",
                    span=span,
                    source=source,
                    detail="set allow_layout_growth=True to permit appended fields",
                )
        elif not allow_layout_growth and actual:
            raise abi_error(
                f"unexpected layout fields for {symbol!r}: {current_fields}",
                span=span,
                source=source,
                detail="the expected layout was empty",
            )
        elif not allow_layout_growth and actual != expected_fields:
            raise abi_error(
                f"layout mismatch for {symbol!r}",
                span=span,
                source=source,
                detail=f"expected {expected_fields}, got {actual}",
            )
        preconditions.append(
            "layout fields match"
            if allow_layout_growth
            else "layout fields are identical"
        )

    if expected_version is not None:
        if version_token != expected_version:
            raise abi_error(
                f"stale version token for {symbol!r}",
                span=span,
                source=source,
                detail=f"expected {expected_version!r}, got {version_token!r}",
            )
        preconditions.append(f"version token {version_token!r} is current")

    if require_resolved and resolve_symbol(symbol) is None:
        raise relocation_error(
            f"symbol {symbol!r} is not bound in this process",
            span=span,
            detail="dlsym(RTLD_DEFAULT) returned NULL; this is a platform probe only",
        )

    return AbiValidation(
        symbol=symbol,
        signature=normalised_signature,
        fields=current_fields,
        version_token=version_token,
        preconditions=tuple(preconditions),
        ok=True,
        model_only=True,
        boundary=(
            "signature parsing/dlsym are real; layout and version checks are "
            "in-process model checks. No executable bytes are inspected."
        ),
    )


# ---------------------------------------------------------------------------
# 53. TOCTOU guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddressGuard:
    """A re-validated snapshot token for one model address cell."""

    address: int
    value: int
    version: int
    fingerprint: str
    generation: int
    state: MutableMapping[int, Mapping[str, Any]] = field(repr=False, compare=False)
    alignment: int = 8
    model_only: bool = True

    def still_valid(
        self,
        *,
        lock: threading.RLock | threading.Lock | None = None,
        span: tuple[int, int] | None = None,
    ) -> bool:
        """Return whether the model cell still matches the validated token."""
        try:
            hotpatch_guard_raw_address_toctou(
                self.address,
                state=self.state,
                expected_value=self.value,
                expected_version=self.version,
                lock=lock,
                alignment=self.alignment,
                span=span,
            )
        except HotpatchError:
            return False
        return True


def _target_fields(
    record: Any,
    address: int,
    *,
    span: tuple[int, int] | None,
    source: str | None,
) -> tuple[int, int]:
    if isinstance(record, Mapping):
        value = record.get("value")
        version = record.get("version", 0)
    else:
        value = getattr(record, "value", None)
        version = getattr(record, "version", 0)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not isinstance(version, int)
        or isinstance(version, bool)
    ):
        raise HotpatchError(
            f"model target at {address:#x} has no integer value/version",
            span=span,
            source=source,
            reason=f"got value={value!r}, version={version!r}",
            help_text="register a mapping or object with integer value and version",
        )
    return value, version


def _address_fingerprint(address: int, value: int, version: int) -> str:
    payload = f"{address}:{value}:{version}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def hotpatch_guard_raw_address_toctou(
    address: int,
    *,
    state: MutableMapping[int, Mapping[str, Any]] | None = None,
    expected_value: int | None = None,
    expected_version: int | None = None,
    lock: threading.RLock | threading.Lock | None = None,
    recheck: Callable[[], None] | None = None,
    alignment: int = 8,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> AddressGuard:
    """Re-validate a target cell and version token while holding one lock.

    The address is a key into an in-process model mapping; it is never
    dereferenced.  Under a single (re-entrant) lock the function reads the
    model cell twice, optionally invoking ``recheck`` between the reads, and
    requires value/version equality.  A returned :class:`AddressGuard` carries
    a deterministic fingerprint that later patch steps can re-check.

    This reduces the window for writers that use the same lock/model; it is
    explicitly model-only and does not protect against an external process, a
    real pointer, or a page-protection change.  No foreign text or raw address
    is touched.
    """
    if alignment <= 0:
        raise HotpatchError(
            "address alignment must be positive",
            span=span,
            source=source,
            reason=f"got alignment={alignment}",
        )
    if address <= 0 or address % alignment:
        raise HotpatchError(
            f"model address {address:#x} is not a positive aligned cell",
            span=span,
            source=source,
            reason=f"required a multiple of {alignment} bytes",
            help_text="pass an aligned key from the in-process dispatch model",
        )
    model = _ADDRESS_MODEL if state is None else state
    target_lock = lock if lock is not None else _ADDRESS_LOCK
    with target_lock:
        record = model.get(address)
        if record is None:
            raise HotpatchError(
                f"no model target registered at {address:#x}",
                span=span,
                source=source,
                reason="raw addresses are model keys, never dereferenced",
            )
        first_value, first_version = _target_fields(
            record, address, span=span, source=source
        )
        if expected_value is not None and first_value != expected_value:
            raise HotpatchError(
                f"target value changed at {address:#x}",
                span=span,
                source=source,
                reason=f"expected {expected_value:#x}, got {first_value:#x}",
            )
        if expected_version is not None and first_version != expected_version:
            raise HotpatchError(
                f"stale version token at {address:#x}",
                span=span,
                source=source,
                reason=f"expected {expected_version}, got {first_version}",
            )
        if recheck is not None:
            recheck()
        latest = model.get(address)
        if latest is None:
            raise HotpatchError(
                f"model target at {address:#x} vanished during re-check",
                span=span,
                source=source,
            )
        second_value, second_version = _target_fields(
            latest, address, span=span, source=source
        )
        if (second_value, second_version) != (first_value, first_version):
            raise HotpatchError(
                f"target state changed between checks at {address:#x}",
                span=span,
                source=source,
                reason=(
                    f"({first_value}, {first_version}) -> "
                    f"({second_value}, {second_version})"
                ),
                help_text="retry the guard with the newer version token",
            )
        _ADDRESS_GENERATION[0] += 1
        return AddressGuard(
            address=address,
            value=second_value,
            version=second_version,
            fingerprint=_address_fingerprint(address, second_value, second_version),
            generation=_ADDRESS_GENERATION[0],
            state=model,
            alignment=alignment,
        )


# ---------------------------------------------------------------------------
# 54/55. Atomic patch transactions and rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchStep:
    """One reversible sub-step of a hot-patch transaction."""

    name: str
    apply: Callable[[], Any]
    undo: Callable[[], Any] | None = None
    key: str | None = None
    state: MutableMapping[str, Any] | None = None


@dataclass
class PatchTransaction:
    """A committed or rolled-back transaction record with replayable undo."""

    id: str
    label: str
    applied: tuple[str, ...]
    results: tuple[Any, ...]
    generation: int
    journal_generation: int
    committed: bool
    rolled_back: bool
    verification: str
    model_only: bool = True
    undo: tuple[Callable[[], None], ...] = field(default=(), repr=False, compare=False)
    error: str | None = None


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of an explicit :func:`hotpatch_rollback_transaction` call."""

    transaction_id: str
    restored: int
    generation: int
    committed_before: bool
    already_rolled_back: bool
    verify_passed: bool
    model_only: bool = True


def _undo_for_step(
    step: PatchStep,
    *,
    span: tuple[int, int] | None,
    source: str | None,
) -> Callable[[], None]:
    if step.undo is not None:
        callback = step.undo

        def undo_callback() -> None:
            callback()

        return undo_callback
    if step.state is not None and step.key is not None:
        previous = step.state.get(step.key, _MISSING)
        target = step.state
        key = step.key

        def undo_state() -> None:
            if previous is _MISSING:
                target.pop(key, None)
            else:
                target[key] = previous

        return undo_state
    raise HotpatchError(
        f"patch step {step.name!r} has no undo plan",
        span=span,
        source=source,
        reason="supply undo= or both key= and state=",
        help_text="every sub-step must be reversible before it is applied",
    )


def hotpatch_apply_patch_transaction(
    steps: Sequence[PatchStep],
    *,
    journal: RollbackJournal | None = None,
    label: str = "hotpatch",
    guard: AddressGuard | None = None,
    verify: Callable[[], bool] | None = None,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> PatchTransaction:
    """Apply reversible steps atomically, rolling back on any failed step.

    The before-image is recorded in a :class:`RollbackJournal` obtained through
    ``runtime.hotpatch.journal``.  Undo entries are registered *before* each
    ``apply()`` call so a step that mutates state and then raises is still
    undone.  An optional :class:`AddressGuard` is re-checked before each step,
    and ``verify`` is treated as a post-check: failure rolls back and raises
    E008 with a transaction record attached as ``error.transaction``.

    The model-only flag is True because steps operate on caller-supplied
    in-process state.  No executable text or raw memory range is patched.
    """
    active_journal = hotpatch_rollback_journal_commit(journal, span=span, source=source)
    snapshot = active_journal.checkpoint(label=label, span=span, source=source)
    plan_text = "|".join(step.name for step in steps)
    digest = hashlib.blake2b(
        f"{label}:{snapshot.generation}:{plan_text}".encode(),
        digest_size=8,
    ).hexdigest()
    transaction = PatchTransaction(
        id=f"{label}:{snapshot.generation}:{digest}",
        label=label,
        applied=(),
        results=(),
        generation=0,
        journal_generation=snapshot.generation,
        committed=False,
        rolled_back=False,
        verification="pending",
    )
    with _TRANSACTION_LOCK:
        _TRANSACTIONS[transaction.id] = transaction
    undo_plan: list[Callable[[], None]] = []
    applied: list[str] = []
    results: list[Any] = []
    try:
        with _TRANSACTION_LOCK:
            for step in steps:
                undo = _undo_for_step(step, span=span, source=source)
                active_journal.record(undo, label=step.name)
                undo_plan.append(undo)
                transaction.undo = tuple(undo_plan)
                if guard is not None and not guard.still_valid(
                    lock=_ADDRESS_LOCK, span=span
                ):
                    raise HotpatchError(
                        f"guard {guard.address:#x} became stale before {step.name!r}",
                        span=span,
                        source=source,
                        reason="the validated target changed before the patch step",
                        help_text="re-run hotpatch_guard_raw_address_toctou and retry",
                    )
                results.append(step.apply())
                applied.append(step.name)
                transaction.applied = tuple(applied)
                transaction.results = tuple(results)
            passed = True if verify is None else bool(verify())
            if not passed:
                raise HotpatchError(
                    f"post-patch verification failed for {label!r}",
                    span=span,
                    source=source,
                    reason="the verifier rejected the patched state",
                    help_text="inspect the step plan and retry with a compatible body",
                )
            active_journal.commit(snapshot)
            transaction.committed = True
            transaction.generation = 1
            transaction.verification = "pass"
            return transaction
    except HotpatchError as error:
        with contextlib.suppress(Exception):
            active_journal.rollback(snapshot)
        transaction.rolled_back = True
        transaction.verification = "fail"
        transaction.error = error.diagnostic.message
        error.transaction = transaction
        raise
    except Exception as error:
        with contextlib.suppress(Exception):
            active_journal.rollback(snapshot)
        transaction.rolled_back = True
        transaction.verification = "fail"
        transaction.error = f"{type(error).__name__}: {error}"
        wrapped = HotpatchError(
            f"hot-patch transaction {label!r} failed: {error}",
            span=span,
            source=source,
            reason=transaction.error,
            help_text="state was rolled back; fix the failing step and retry",
        )
        wrapped.transaction = transaction
        raise wrapped from error


def hotpatch_rollback_transaction(
    transaction: PatchTransaction | str,
    *,
    verify: Callable[[], bool] | None = None,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> RollbackResult:
    """Replay a transaction's undo entries in reverse and verify the result.

    A transaction object or its deterministic id may be supplied.  Committed
    and already-rolled-back transactions are both accepted: a second rollback
    is a no-op, which makes failure handling idempotent.  This is an in-process
    model rollback; it restores Python state and never rewrites code bytes.
    """
    with _TRANSACTION_LOCK:
        target = (
            _TRANSACTIONS.get(transaction)
            if isinstance(transaction, str)
            else transaction
        )
    if target is None:
        raise HotpatchError(
            f"unknown hot-patch transaction {transaction!r}",
            span=span,
            source=source,
            reason="the transaction id is not in this process's undo registry",
            help_text="pass the transaction returned by apply_patch_transaction",
        )
    if target.rolled_back:
        passed = True if verify is None else bool(verify())
        return RollbackResult(
            transaction_id=target.id,
            restored=0,
            generation=target.generation,
            committed_before=target.committed,
            already_rolled_back=True,
            verify_passed=passed,
        )
    failures: list[str] = []
    with _TRANSACTION_LOCK:
        for undo in reversed(target.undo):
            try:
                undo()
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        restored = len(target.undo) - len(failures)
        committed_before = target.committed
        target.committed = False
        target.rolled_back = True
        target.generation = 0
        target.verification = "rolled-back"
    if failures:
        raise HotpatchError(
            f"rollback of {target.id!r} had failing undo callbacks",
            span=span,
            source=source,
            reason="; ".join(failures),
            help_text="inspect the undo callbacks; remaining state may still be patched",
        )
    passed = True if verify is None else bool(verify())
    if not passed:
        raise HotpatchError(
            f"post-rollback verification failed for {target.id!r}",
            span=span,
            source=source,
            reason="state still fails the caller's verifier",
        )
    return RollbackResult(
        transaction_id=target.id,
        restored=restored,
        generation=target.generation,
        committed_before=committed_before,
        already_rolled_back=False,
        verify_passed=passed,
    )


# ---------------------------------------------------------------------------
# 56. Frame evaluation hook registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameEvalHook:
    """One model slot in a CPython-style frame-evaluation hook table."""

    name: str
    code_name: str
    tool_id: int
    generation: int
    version_floor: tuple[int, int]
    live_locals: bool
    layout: FrameLayout | None = field(repr=False, compare=False, default=None)
    model_only: bool = True


def hotpatch_update_frame_eval_hook(
    hooks: MutableMapping[str, FrameEvalHook],
    name: str,
    *,
    action: str = "install",
    code: types.CodeType | str | None = None,
    python_version: tuple[int, int] | None = None,
    live_locals: bool = False,
    max_tool_ids: int = 6,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> FrameEvalHook:
    """Install, replace, query or remove a frame-evaluation hook model slot.

    ``sys.monitoring`` requires CPython 3.12+, writable frame locals require
    3.13+, and CPython offers tool ids 0..5.  Those version and slot limits are
    enforced here (E007 for an unsupported version, E008/E006 for a bad slot
    transition).  The hook table is caller-owned in-process state: this
    function installs no global monitoring callback and never rewrites a live
    ``f_code`` pointer, so ``model_only`` is always True.
    """
    if action not in {"install", "replace", "query", "remove"}:
        raise PillarError(
            "E003",
            f"unknown frame-evaluation hook action {action!r}",
            reason="use install, replace, query or remove",
            span=span,
        )
    version = tuple(python_version or sys.version_info[:2])
    if len(version) != 2 or not all(isinstance(part, int) for part in version):
        raise HotpatchError(
            "python_version must be a (major, minor) pair",
            span=span,
            source=source,
            reason=f"got {version!r}",
        )
    if version < (3, 12):
        raise HotpatchError(
            "frame-evaluation hooks require CPython 3.12 or newer",
            code="E007",
            span=span,
            source=source,
            reason=f"this interpreter is {version[0]}.{version[1]}",
            help_text="use a supported interpreter or patch future entries only",
        )
    if live_locals and version < (3, 13):
        raise HotpatchError(
            "live fast-local migration requires CPython 3.13 or newer",
            code="E007",
            span=span,
            source=source,
            reason="PEP 667 writable frame locals are unavailable",
            help_text="disable live_locals or upgrade the interpreter",
        )
    if not 1 <= max_tool_ids <= 6:
        raise HotpatchError(
            "CPython exposes at most six monitoring tool ids",
            span=span,
            source=source,
            reason=f"max_tool_ids={max_tool_ids} is outside 1..6",
        )

    current = hooks.get(name)
    if action == "query":
        if current is None:
            raise HotpatchError(
                f"frame-evaluation hook {name!r} is not installed",
                code="E006",
                span=span,
                source=source,
            )
        return current
    if action == "remove":
        if current is None:
            raise HotpatchError(
                f"frame-evaluation hook {name!r} is not installed",
                code="E006",
                span=span,
                source=source,
            )
        del hooks[name]
        return current
    if action == "install" and current is not None:
        raise HotpatchError(
            f"frame-evaluation hook {name!r} is already installed",
            span=span,
            source=source,
            reason="use action='replace' to update an installed hook",
        )
    if action == "replace" and current is None:
        raise HotpatchError(
            f"frame-evaluation hook {name!r} is not installed",
            code="E006",
            span=span,
            source=source,
            reason="use action='install' for a new hook",
        )

    layout: FrameLayout | None = None
    if isinstance(code, types.CodeType):
        layout = FrameLayout.from_code(code)
        code_name = code.co_name
    elif isinstance(code, str):
        code_name = code
    elif code is None:
        code_name = current.code_name if current is not None else name
    else:
        raise HotpatchError(
            f"hook target must be a code object or name, got {type(code).__name__}",
            span=span,
            source=source,
        )
    if current is not None:
        if current.code_name != code_name:
            raise abi_error(
                f"hook {name!r} target changed from {current.code_name!r}",
                span=span,
                source=source,
                detail=f"replacement targets {code_name!r}",
            )
        if layout is not None and current.layout is not None:
            issues = current.layout.issues(layout)
            if issues:
                raise abi_error(
                    f"hook {name!r} target layout changed",
                    span=span,
                    source=source,
                    detail="; ".join(issues),
                )
        tool_id = current.tool_id
        generation = current.generation + 1
    else:
        used = {hook.tool_id for hook in hooks.values()}
        free = [index for index in range(max_tool_ids) if index not in used]
        if not free:
            raise HotpatchError(
                "no free frame-evaluation hook slot",
                code="E007",
                span=span,
                source=source,
                reason=f"all {max_tool_ids} model tool ids are in use",
                help_text="remove a hook before installing another",
            )
        tool_id = free[0]
        generation = 1
    hook = FrameEvalHook(
        name=name,
        code_name=code_name,
        tool_id=tool_id,
        generation=generation,
        version_floor=version,
        live_locals=live_locals,
        layout=layout if layout is not None else (current.layout if current else None),
    )
    hooks[name] = hook
    return hook


# ---------------------------------------------------------------------------
# 57. Reactor registration migration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingEvent:
    """A ready event captured before a reactor swap."""

    fd: int
    events: int
    data: Any
    mask: int
    origin: str


@dataclass
class ReactorMigrationResult:
    """A reactor migration plus the events that must survive the swap."""

    kind: str
    registrations: tuple[Registration, ...]
    pending_events: tuple[PendingEvent, ...]
    packet_loss: int
    generation: int
    committed: bool
    rolled_back: bool
    model_only: bool
    boundary: str
    migration: Any = field(repr=False, compare=False)

    def commit(self) -> None:
        """Retire the old poll object after the caller swaps its pointer."""
        if not self.committed:
            self.migration.commit()
            self.committed = True

    def rollback(self) -> None:
        """Discard the replacement poll object and keep the old one active."""
        if not self.rolled_back:
            self.migration.rollback()
            self.rolled_back = True


def hotpatch_migrate_reactor_registrations(
    source: Any,
    *,
    target: Any | None = None,
    pending_events: Sequence[Any] = (),
    duplicate_fds: bool = False,
    commit: bool = False,
    span: tuple[int, int] | None = None,
    source_text: str | None = None,
) -> ReactorMigrationResult:
    """Move reactor interests to a fresh poll object without losing events.

    Live ``selectors`` and mio bridge objects are delegated to
    ``runtime.hotpatch.reactor.hotpatch_mio_reactor_state``, so descriptors and
    their kernel queues stay open.  Ready events are captured with a
    non-blocking ``select(0)`` before the swap and are returned alongside
    caller-supplied ``pending_events``; the registration set is verified on the
    replacement before the migration is offered for commit.  A mismatch rolls
    the migration back and raises E008.

    The selector path is a real in-process migration (``model_only=False``).
    No raw descriptor pointers are forged and no socket is closed; only the
    poll object is retired on ``commit()``.
    """
    captured: list[PendingEvent] = []
    state = None
    if isinstance(source, selectors.BaseSelector):
        try:
            ready = source.select(timeout=0)
        except (OSError, ValueError) as error:
            raise HotpatchError(
                "cannot capture pending reactor events",
                span=span,
                source=source_text,
                reason=str(error),
                help_text="capture readiness before closing the owning module",
            ) from None
        captured.extend(
            PendingEvent(key.fd, key.events, key.data, mask, "ready")
            for key, mask in ready
        )
        state = capture_selector(source, span=span, source=source_text)
    captured.extend(PendingEvent(-1, 0, item, 0, "caller") for item in pending_events)

    migration = hotpatch_mio_reactor_state(
        source,
        target,
        duplicate_fds=duplicate_fds,
        span=span,
        source_text=source_text,
    )
    registrations = tuple(migration.registrations)
    replacement = getattr(migration, "new_selector", None)
    if replacement is not None:
        current = replacement.get_map()
        expected = {(item.fd, item.events, item.data) for item in registrations}
        actual = {(key.fd, key.events, key.data) for key in current.values()}
        if expected != actual:
            with contextlib.suppress(Exception):
                migration.rollback()
            raise HotpatchError(
                "reactor re-registration did not preserve all interests",
                span=span,
                source=source_text,
                reason=f"expected {expected!r}, got {actual!r}",
            )
    if state is not None and len(state.registrations) != len(registrations):
        with contextlib.suppress(Exception):
            migration.rollback()
        raise HotpatchError(
            "reactor migration changed the registration count",
            span=span,
            source=source_text,
            reason=(
                f"captured {len(state.registrations)}, migrated {len(registrations)}"
            ),
        )
    result = ReactorMigrationResult(
        kind=type(migration).__name__,
        registrations=registrations,
        pending_events=tuple(captured),
        packet_loss=getattr(migration, "packet_loss", 0),
        generation=state.generation if state is not None else 1,
        committed=False,
        rolled_back=False,
        model_only=state is None,
        boundary=(
            "selectors/epoll/kqueue fds stay open; a mio bridge is a PyO3 seam, "
            "not a raw pointer operation from this module"
        ),
        migration=migration,
    )
    if commit:
        result.commit()
    return result


# ---------------------------------------------------------------------------
# 58. RCU epoch advance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RcuEpochResult:
    """Observed epoch transition and reclamation outcome."""

    previous_epoch: int
    epoch: int
    retired_before: int
    reclaimed: int
    deferred: int
    active_readers: int
    grace_periods: int
    published: bool
    model_only: bool = True
    boundary: str = "cooperative user-space RCU; no kernel RCU syscall"


def hotpatch_advance_rcu_epoch(
    manager: RcuDomain | AstEpochManager | None = None,
    *,
    retire: Any = None,
    reclaim: Callable[[Any], None] | None = None,
    timeout: float = 5.0,
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> RcuEpochResult:
    """Retire an old root and advance the epoch once readers have drained.

    An :class:`AstEpochManager` receiving a ``Program`` publishes it (versioned
    root swap); other values are retired on the domain with an optional
    reclaimer callback.  ``synchronize()`` waits for readers that entered
    before the advance and defers reclamation; a grace-period timeout is
    reported as E008 instead of freeing live state.

    This is deterministic user-space RCU (``model_only=True``): readers must
    cooperate by leaving their guards, and no kernel epoch syscall is used.
    """
    domain: RcuDomain
    published = False
    if manager is None:
        domain = RcuDomain("framework")
    elif isinstance(manager, AstEpochManager):
        domain = manager.domain
    else:
        domain = manager
    if retire is not None:
        if isinstance(manager, AstEpochManager) and isinstance(retire, Program):
            manager.publish(retire, span=span)
            published = True
        else:
            domain.retire(retire, reclaim, span=span)
    before: RcuStats = domain.stats()
    reclaimed = domain.synchronize(timeout=timeout)
    after: RcuStats = domain.stats()
    del source  # RCU diagnostics are carried by the domain's own HotpatchError
    return RcuEpochResult(
        previous_epoch=before.epoch,
        epoch=after.epoch,
        retired_before=before.retired,
        reclaimed=reclaimed,
        deferred=after.retired,
        active_readers=after.active_readers,
        grace_periods=after.grace_periods,
        published=published,
    )


# ---------------------------------------------------------------------------
# 59. Legacy backend delegation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationResult:
    """Outcome of routing a legacy backend call through runtime.hotpatch."""

    capability: str
    route: str
    action: str
    generation: int
    changes: tuple[str, ...]
    journal_generation: int
    committed: bool
    model_only: bool
    detail: str


def hotpatch_delegate_legacy_backend(
    manager: Any,
    *,
    action: str = "patch",
    new_program: Program | None = None,
    capability: str = "hotpatch_rollback_journal_commit",
    span: tuple[int, int] | None = None,
    source: str | None = None,
) -> DelegationResult:
    """Route ``backends.hotpatch`` operations through the runtime registry.

    The capability handler is resolved by :func:`hotpatch_registry_lookup_capability`
    and invoked through ``runtime.capabilities.invoke``; the rollback journal is
    therefore the runtime one.  ``prepare``, ``patch`` and ``rollback`` delegate
    to the legacy manager's in-process namespace swap while the journal records
    live objects before a patch.  Legacy rejections become structured E008
    diagnostics via :func:`hotpatch_emit_diagnostic_e008`.

    ``action='native_patch_text'`` is refused with E007 through the trampoline
    unsupported helper: loaded executable text is never modified.  The wrapper
    itself is real in-process state management (``model_only=False``), not a
    foreign code patch.
    """
    if action == "native_patch_text":
        raise unsupported("foreign executable text patching", span=span)
    if action not in {"prepare", "patch", "rollback"}:
        raise PillarError(
            "E003",
            f"unknown legacy delegation action {action!r}",
            reason="use prepare, patch, rollback or native_patch_text",
            span=span,
        )
    resolution = hotpatch_registry_lookup_capability(
        capability, span=span, source=source
    )
    if not resolution.active or resolution.handler is None:
        raise HotpatchError(
            f"capability {capability!r} is not active",
            span=span,
            source=source,
            reason=resolution.reason,
            help_text="choose an active runtime.hotpatch capability",
        )
    if action in {"prepare", "patch"} and new_program is None:
        raise HotpatchError(
            f"action {action!r} requires new_program",
            span=span,
            source=source,
        )

    changes: tuple[str, ...] = ()
    if action in {"prepare", "patch"}:
        prepare = getattr(manager, "prepare", None)
        if callable(prepare):
            plan = prepare(new_program)
        else:
            plan = legacy_hotpatch.diff_programs(
                getattr(manager, "_program", None), new_program
            )
        changes = tuple(f"{change.name}:{change.kind.value}" for change in plan.changes)

    if action == "prepare":
        return DelegationResult(
            capability=capability,
            route="runtime.hotpatch",
            action=action,
            generation=int(getattr(manager, "generation", 0)),
            changes=changes,
            journal_generation=0,
            committed=False,
            model_only=False,
            detail=f"prepared {len(changes)} change(s) via the runtime registry",
        )

    if action == "rollback":
        try:
            generation = int(manager.rollback())
        except PillarError as error:
            diagnostic = hotpatch_emit_diagnostic_e008(
                "legacy backend rollback failed",
                span=span,
                source=source,
                reason=error.diagnostic.reason,
                detail=error.diagnostic.message,
                capability=capability,
            )
            raise diagnostic.error from error
        return DelegationResult(
            capability=capability,
            route="runtime.hotpatch",
            action=action,
            generation=generation,
            changes=changes,
            journal_generation=0,
            committed=True,
            model_only=False,
            detail="legacy undo replayed with runtime capability resolution",
        )

    journal = capabilities.invoke(capability, span=span)
    live = getattr(manager, "live_objects", None)
    objects = list(live()) if callable(live) else []
    snapshot = journal.checkpoint(objects=objects, label="legacy-delegate", span=span)
    try:
        result = manager.patch(new_program)
    except PillarError as error:
        with contextlib.suppress(Exception):
            journal.rollback(snapshot)
        diagnostic = hotpatch_emit_diagnostic_e008(
            "legacy backend rejected hot-patch",
            span=_first_span(plan, error),
            source=source,
            reason=error.diagnostic.reason,
            detail=error.diagnostic.message,
            capability=capability,
        )
        raise diagnostic.error from error
    journal.commit(snapshot)
    return DelegationResult(
        capability=capability,
        route="runtime.hotpatch",
        action=action,
        generation=int(result.generation),
        changes=changes,
        journal_generation=snapshot.generation,
        committed=True,
        model_only=False,
        detail=(
            f"runtime journal captured {len(objects)} live object(s); "
            f"legacy namespace swapped {int(result.swapped)} callable(s)"
        ),
    )


def _first_span(plan: Any, error: PillarError) -> tuple[int, int] | None:
    """Prefer an incompatible change's span, then the legacy error's span."""
    for change in getattr(plan, "incompatible", ()):
        if change.span is not None:
            return change.span
    return error.span


# ---------------------------------------------------------------------------
# 60. E008 diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E008Diagnostic:
    """A structured E008 record with character and UTF-8 byte spans."""

    code: str
    message: str
    reason: str
    help_text: str
    note: str
    line: int
    col: int
    span: tuple[int, int] | None
    byte_span: ByteSpan | None
    excerpt: str
    formatted: str
    emitted: bool
    capability: str | None
    error: HotpatchError = field(repr=False, compare=False)


def hotpatch_emit_diagnostic_e008(
    message: str,
    *,
    span: tuple[int, int] | None = None,
    source: str | None = None,
    line: int = 0,
    col: int = 0,
    reason: str | None = None,
    help_text: str | None = None,
    note: str | None = None,
    detail: str | None = None,
    capability: str | None = None,
    unsupported_request: bool = False,
    sink: Callable[[E008Diagnostic], None] | None = None,
    raise_error: bool = False,
) -> E008Diagnostic:
    """Build (and optionally emit/raise) a structured E008 hot-patch failure.

    The character ``span`` is preserved verbatim while ``byte_span`` is derived
    from ``source`` by the hot-patch error layer, so non-ASCII source before the
    span still reports correct UTF-8 offsets.  ``excerpt`` is the exact source
    slice.  Setting ``unsupported_request`` annotates the diagnostic as a
    refused platform request; truly unavailable primitives are reported as E007
    by :func:`errors.unsupported`, while failed hot-patch attempts use this
    E008 surface as required by the framework contract.

    ``sink`` receives the immutable diagnostic when supplied.  With
    ``raise_error=True`` the same payload is raised as a :class:`HotpatchError`.
    """
    combined_reason = reason or (
        "an unsupported or failed hot-patch was refused without mutating live state"
        if unsupported_request
        else "a hot-patch validation, relocation or verification step failed"
    )
    if detail:
        combined_reason = f"{combined_reason}; {detail}"
    combined_note = note or (
        "unsupported-hotpatch" if unsupported_request else "HotpatchError"
    )
    if capability:
        combined_note = f"{combined_note}:{capability}"
    error = HotpatchError(
        message,
        code="E008",
        span=span,
        source=source,
        line=line,
        col=col,
        reason=combined_reason,
        help_text=help_text,
        note=combined_note,
    )
    excerpt = "" if span is None or source is None else source[span[0] : span[1]]
    diagnostic = E008Diagnostic(
        code=error.code,
        message=message,
        reason=combined_reason,
        help_text=error.help_text,
        note=combined_note,
        line=error.line,
        col=error.col,
        span=error.span,
        byte_span=error.byte_span,
        excerpt=excerpt,
        formatted=error.format(),
        emitted=sink is not None,
        capability=capability,
        error=error,
    )
    if sink is not None:
        sink(diagnostic)
    if raise_error:
        raise error
    return diagnostic


__all__ = [
    "AbiValidation",
    "AddressGuard",
    "CapabilityResolution",
    "DelegationResult",
    "E008Diagnostic",
    "FrameEvalHook",
    "PatchStep",
    "PatchTransaction",
    "PendingEvent",
    "RcuEpochResult",
    "ReactorMigrationResult",
    "RollbackResult",
    "hotpatch_advance_rcu_epoch",
    "hotpatch_apply_patch_transaction",
    "hotpatch_delegate_legacy_backend",
    "hotpatch_emit_diagnostic_e008",
    "hotpatch_guard_raw_address_toctou",
    "hotpatch_migrate_reactor_registrations",
    "hotpatch_registry_lookup_capability",
    "hotpatch_rollback_transaction",
    "hotpatch_update_frame_eval_hook",
    "hotpatch_validate_target_abi",
]
