"""Honest, dependency-free reference foundations for DSM and consensus.

The ten entry points below are *simulations* built from plain Python data
structures.  They are useful for lowering, testing, and reasoning about
distributed-shared-memory algorithms, but they deliberately do **not** claim:

* real ``userfaultfd``, Mach, ``mprotect``/signal, or any other kernel
  page-fault interception;
* real RDMA, kernel bypass, sockets, or network message passing;
* real Raft/Paxos kernel consensus, threads, or cross-process fencing.

Requesting one of those privileged backends raises ``E007`` with the exact
caller-supplied AST span.  Everything else is deterministic and local: no
clock, network, thread, file, or third-party dependency is used.  All failures
are :class:`runtime.diagnostics.PillarError` values using the project's public
E001--E008 codes.

Every public function accepts the optional keyword-only ``span``, ``line`` and
``col`` arguments so a front end can attach exact character locations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, NoReturn

from runtime.diagnostics import PillarError

__all__ = [
    "REJECTED_BACKENDS",
    "Acquire",
    "CASNode",
    "CASResult",
    "CRDTDocument",
    "CRDTField",
    "CRDTSchema",
    "CausalEvent",
    "CausalityReport",
    "FenceClaim",
    "FenceResult",
    "GCReport",
    "HLCStamp",
    "HybridLogicalClock",
    "LWWRegister",
    "LeaseGrant",
    "LogEntry",
    "MVCCBatch",
    "MVCCRead",
    "MVCCReport",
    "MVCCVersion",
    "MerkleDiff",
    "MerkleTree",
    "NewEpoch",
    "ORSet",
    "PNCounter",
    "PageAccess",
    "PageFetch",
    "PyFutureStub",
    "RaftAwaitable",
    "RaftLog",
    "Release",
    "SoftwarePageTable",
    "SplitPlan",
    "SplitReport",
    "SubPage",
    "Tick",
    "VectorClock",
    "drive_future",
    "dsm_atomic_cross_node_cas",
    "dsm_compile_struct_to_crdt",
    "dsm_distributed_gc_lease",
    "dsm_false_sharing_page_split",
    "dsm_merkle_tree_state_diff",
    "dsm_mvcc_snapshot_isolation",
    "dsm_raft_quorum_future_yield",
    "dsm_split_brain_partition_fence",
    "dsm_vector_clock_causality",
    "dsm_virtual_page_fault_fetch",
]

Span = tuple[int, int] | None

#: Privileged backends we explicitly refuse to emulate as "real" features.
REJECTED_BACKENDS: Final[dict[str, str]] = {
    "userfaultfd": "Linux userfaultfd page-fault interception needs privileged OS support",
    "mach": "Mach VM page-fault handling needs kernel ports and VM privileges",
    "mprotect": "mprotect/signal page trapping needs OS memory-protection privileges",
    "sigsegv": "SIGSEGV page-fault interception needs an OS signal handler",
    "rdma": "real RDMA needs kernel-bypass hardware and pinned memory registration",
    "kernel": "real kernel consensus is not implemented",
    "kernel-raft": "real kernel Raft consensus is not implemented",
    "kernel-paxos": "real kernel Paxos consensus is not implemented",
}

_SOFTWARE_BACKENDS: Final[frozenset[str]] = frozenset(
    {"software", "simulated", "python", "in-memory", "memory"}
)
_HASH_SIZE: Final = 16
_EMPTY_LEAF: Final = hashlib.blake2s(
    b"polyglot:dsm:empty-leaf", digest_size=_HASH_SIZE
).hexdigest()


@dataclass(frozen=True)
class _Context:
    span: Span
    line: int
    col: int

    def fail(
        self,
        code: str,
        message: str,
        *,
        reason: str | None = None,
        help_text: str | None = None,
        note: str | None = None,
    ) -> NoReturn:
        raise PillarError(
            code,
            message,
            reason=reason,
            help_text=help_text,
            note=note,
            line=self.line,
            col=self.col,
            span=self.span,
        )


def _context(span: Span, line: int, col: int) -> _Context:
    if span is not None and (
        not isinstance(span, tuple)
        or len(span) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) for part in span)
        or span[0] < 0
        or span[1] < span[0]
    ):
        raise PillarError(
            "E003",
            "invalid character span; expected (start, end) with 0 <= start <= end",
            note="InvalidSpan",
            line=line,
            col=col,
        )
    return _Context(span=span, line=line, col=col)


def _items(value: Any, name: str, ctx: _Context) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)):
        ctx.fail("E001", f"{name} must be an iterable of records, not text")
    try:
        return list(value)
    except TypeError:
        ctx.fail("E001", f"{name} must be iterable", note="InvalidInput")


def _mapping(value: Any, name: str, ctx: _Context) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        ctx.fail("E001", f"{name} must be a mapping", note="InvalidInput")
    return dict(value)


def _integer(
    value: Any, name: str, ctx: _Context, *, minimum: int | None = None
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        ctx.fail("E001", f"{name} must be an integer", note="InvalidInput")
    if minimum is not None and value < minimum:
        ctx.fail("E003", f"{name} must be >= {minimum}", note="InvalidArgument")
    return value


def _text(value: Any, name: str, ctx: _Context) -> str:
    if not isinstance(value, str) or not value:
        ctx.fail("E001", f"{name} must be a non-empty string", note="InvalidInput")
    return value


def _bytes(value: Any, name: str, ctx: _Context) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    ctx.fail("E001", f"{name} must be bytes-like or str", note="InvalidInput")


def _backend(backend: Any, allowed: frozenset[str], ctx: _Context, *, kind: str) -> str:
    if not isinstance(backend, str):
        ctx.fail("E001", "backend must be a string", note="InvalidInput")
    if backend in REJECTED_BACKENDS:
        ctx.fail(
            "E007",
            f"{backend!r} is not implemented: {REJECTED_BACKENDS[backend]}",
            reason=REJECTED_BACKENDS[backend],
            help_text="use the dependency-free software simulator provided here",
            note="PrivilegedFeatureRejected",
        )
    if backend not in allowed:
        ctx.fail(
            "E003",
            f"unknown {kind} backend {backend!r}",
            help_text=f"use one of: {', '.join(sorted(allowed))}",
            note="InvalidBackend",
        )
    return backend


# ---------------------------------------------------------------------------
# 61. Virtual page fault fetch (software page table only)
# ---------------------------------------------------------------------------


@dataclass
class SoftwarePageTable:
    """An in-memory page table: a dict lookup stands in for a fault handler."""

    pages: dict[str, bytes] = field(default_factory=dict)
    remote: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    faults: int = 0
    hits: int = 0
    trace: list[str] = field(default_factory=list)

    def install(
        self, page_id: str, data: bytes, *, ctx: _Context | None = None
    ) -> None:
        context = ctx or _context(None, 0, 0)
        self.pages[_text(page_id, "page_id", context)] = _bytes(data, "data", context)

    def publish(
        self,
        page_id: str,
        data: bytes,
        node: str,
        *,
        ctx: _Context | None = None,
    ) -> None:
        context = ctx or _context(None, 0, 0)
        self.remote[_text(page_id, "page_id", context)] = (
            _text(node, "node", context),
            _bytes(data, "data", context),
        )

    def fetch(
        self,
        page_id: str,
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> PageFetch:
        ctx = _context(span, line, col)
        pid = _text(page_id, "page_id", ctx)
        if pid in self.pages:
            self.hits += 1
            self.trace.append(f"hit:{pid}")
            return PageFetch(
                pid,
                self.pages[pid],
                "resident",
                self.faults,
                self.hits,
                tuple(self.trace),
            )
        if pid in self.remote:
            node, data = self.remote[pid]
            self.faults += 1
            self.pages[pid] = data
            self.trace.append(f"fault:{pid}<-{node}")
            return PageFetch(
                pid, data, "remote", self.faults, self.hits, tuple(self.trace)
            )
        ctx.fail(
            "E006",
            f"page {pid!r} is not installed or published by any peer",
            help_text="install the page or publish it through the remote page directory",
            note="PageNotFound",
        )


@dataclass(frozen=True)
class PageFetch:
    page_id: str
    data: bytes
    source: str
    faults: int
    hits: int
    trace: tuple[str, ...]


def dsm_virtual_page_fault_fetch(
    table: SoftwarePageTable | Mapping[str, bytes],
    page_id: str,
    *,
    remote: Mapping[str, tuple[str, bytes] | bytes] | None = None,
    backend: str = "software",
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> PageFetch:
    """Resolve a page through a software page table.

    A miss is a simulated fault: the page is copied from an explicitly
    published in-memory peer directory.  No OS page-fault handler is installed,
    and no memory protection is changed.  ``userfaultfd``, Mach, ``mprotect``
    and signal backends are rejected with ``E007``.
    """
    ctx = _context(span, line, col)
    _backend(backend, _SOFTWARE_BACKENDS, ctx, kind="page-fault")
    if isinstance(table, SoftwarePageTable):
        # A page table is deliberately stateful: a simulated fault installs the
        # page so later fetches observe a normal resident hit.
        page_table = table
    else:
        raw_pages = _mapping(table, "table", ctx)
        page_table = SoftwarePageTable()
        for pid, data in raw_pages.items():
            page_table.install(pid, data, ctx=ctx)
    if remote is not None:
        for pid, payload in _mapping(remote, "remote", ctx).items():
            if isinstance(payload, tuple) and len(payload) == 2:
                node, data = payload
            else:
                node, data = "peer", payload
            page_table.publish(pid, data, node, ctx=ctx)
    return page_table.fetch(page_id, span=span, line=line, col=col)


# ---------------------------------------------------------------------------
# 62. Struct field metadata -> CRDT lowering
# ---------------------------------------------------------------------------

CRDT_KINDS: Final[frozenset[str]] = frozenset({"lww-register", "or-set", "pn-counter"})
_KIND_ALIASES: Final[dict[str, str]] = {
    "lww": "lww-register",
    "lww_register": "lww-register",
    "lww-register": "lww-register",
    "orset": "or-set",
    "or_set": "or-set",
    "or-set": "or-set",
    "pncounter": "pn-counter",
    "pn_counter": "pn-counter",
    "pn-counter": "pn-counter",
}


@dataclass(frozen=True)
class CRDTField:
    name: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRDTSchema:
    name: str
    fields: tuple[CRDTField, ...]

    def document(self) -> CRDTDocument:
        return CRDTDocument(self)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fields": [
                {"name": item.name, "kind": item.kind, "metadata": dict(item.metadata)}
                for item in self.fields
            ],
        }


@dataclass
class LWWRegister:
    """Last-writer-wins register ordered by ``(timestamp, writer)``."""

    value: Any = None
    timestamp: int = 0
    writer: str = ""

    def set(self, value: Any, *, timestamp: int, writer: str = "") -> None:
        if (timestamp, writer) > (self.timestamp, self.writer):
            self.value = value
            self.timestamp = timestamp
            self.writer = writer

    def merge(self, other: LWWRegister) -> None:
        self.set(other.value, timestamp=other.timestamp, writer=other.writer)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "lww-register",
            "value": self.value,
            "timestamp": self.timestamp,
            "writer": self.writer,
        }


@dataclass
class ORSet:
    """Observed-remove set: tagged additions and the tombstones observed."""

    adds: set[tuple[Any, str]] = field(default_factory=set)
    removes: set[tuple[Any, str]] = field(default_factory=set)

    @property
    def value(self) -> frozenset[Any]:
        return frozenset(
            element for element, tag in self.adds if (element, tag) not in self.removes
        )

    def add(self, element: Any, tag: str) -> None:
        self.adds.add((element, tag))

    def remove(self, element: Any) -> None:
        self.removes.update(item for item in self.adds if item[0] == element)

    def merge(self, other: ORSet) -> None:
        self.adds |= other.adds
        self.removes |= other.removes

    def snapshot(self) -> dict[str, Any]:
        return {"kind": "or-set", "elements": tuple(sorted(self.value, key=repr))}


@dataclass
class PNCounter:
    """Positive/negative counter merged with per-replica maxima."""

    positives: dict[str, int] = field(default_factory=dict)
    negatives: dict[str, int] = field(default_factory=dict)

    @property
    def value(self) -> int:
        return sum(self.positives.values()) - sum(self.negatives.values())

    def increment(self, replica: str, amount: int = 1) -> None:
        self.positives[replica] = self.positives.get(replica, 0) + amount

    def decrement(self, replica: str, amount: int = 1) -> None:
        self.negatives[replica] = self.negatives.get(replica, 0) + amount

    def merge(self, other: PNCounter) -> None:
        for replica, count in other.positives.items():
            self.positives[replica] = max(self.positives.get(replica, 0), count)
        for replica, count in other.negatives.items():
            self.negatives[replica] = max(self.negatives.get(replica, 0), count)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "pn-counter",
            "value": self.value,
            "positives": dict(sorted(self.positives.items())),
            "negatives": dict(sorted(self.negatives.items())),
        }


CRDTValue = LWWRegister | ORSet | PNCounter


@dataclass
class CRDTDocument:
    """One replica of a compiled struct; merges are commutative and idempotent."""

    schema: CRDTSchema
    fields: dict[str, CRDTValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for item in self.schema.fields:
            if item.kind == "lww-register":
                self.fields[item.name] = LWWRegister()
            elif item.kind == "or-set":
                self.fields[item.name] = ORSet()
            else:
                self.fields[item.name] = PNCounter()

    def apply(
        self,
        name: str,
        operation: str,
        *,
        value: Any = None,
        tag: str | None = None,
        replica: str = "",
        writer: str | None = None,
        timestamp: int = 0,
        amount: int = 1,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> None:
        ctx = _context(span, line, col)
        if name not in self.fields:
            ctx.fail("E003", f"unknown CRDT field {name!r}", note="UnknownField")
        crdt = self.fields[name]
        if isinstance(crdt, LWWRegister):
            if operation not in {"set", "write", "assign"}:
                ctx.fail(
                    "E003", f"operation {operation!r} is invalid for a LWW-register"
                )
            crdt.set(
                value,
                timestamp=_integer(timestamp, "timestamp", ctx, minimum=0),
                writer=writer if writer is not None else replica,
            )
            return
        if isinstance(crdt, ORSet):
            try:
                hash(value)
            except TypeError:
                ctx.fail(
                    "E001", "OR-Set elements must be hashable", note="InvalidInput"
                )
            if operation == "add":
                crdt.add(value, _text(tag or replica or "tag", "tag", ctx))
            elif operation == "remove":
                crdt.remove(value)
            else:
                ctx.fail("E003", f"operation {operation!r} is invalid for an OR-Set")
            return
        if operation in {"inc", "increment"}:
            crdt.increment(
                _text(replica, "replica", ctx),
                _integer(amount, "amount", ctx, minimum=0),
            )
        elif operation in {"dec", "decrement"}:
            crdt.decrement(
                _text(replica, "replica", ctx),
                _integer(amount, "amount", ctx, minimum=0),
            )
        else:
            ctx.fail("E003", f"operation {operation!r} is invalid for a PN-Counter")

    def merge(
        self, other: CRDTDocument, *, span: Span = None, line: int = 0, col: int = 0
    ) -> None:
        ctx = _context(span, line, col)
        if self.schema != other.schema:
            ctx.fail(
                "E008",
                "CRDT schema drift: documents were compiled from different structs",
            )
        for name, crdt in self.fields.items():
            incoming = other.fields[name]
            if type(crdt) is not type(incoming):
                ctx.fail("E008", f"CRDT kind drift on field {name!r}")
            crdt.merge(incoming)

    def snapshot(self) -> dict[str, Any]:
        return {name: crdt.snapshot() for name, crdt in self.fields.items()}


def dsm_compile_struct_to_crdt(
    fields: Mapping[str, str | Mapping[str, Any]],
    *,
    name: str = "Struct",
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> CRDTSchema:
    """Lower field metadata into LWW-Register / OR-Set / PN-Counter fields.

    A field maps to a kind name (with a few aliases) or to a mapping with a
    required ``crdt``/``kind`` key; extra keys are preserved as metadata.  The
    returned schema can create replicas whose merges converge.
    """
    ctx = _context(span, line, col)
    _text(name, "name", ctx)
    compiled: list[CRDTField] = []
    for field_name, spec in _mapping(fields, "fields", ctx).items():
        _text(field_name, "field name", ctx)
        if isinstance(spec, str):
            kind, metadata = spec, {}
        elif isinstance(spec, Mapping):
            raw = dict(spec)
            kind = raw.pop("crdt", raw.pop("kind", None))
            metadata = raw
        else:
            ctx.fail(
                "E003", f"field {field_name!r} needs a kind string or metadata mapping"
            )
        if not isinstance(kind, str) or kind not in _KIND_ALIASES:
            ctx.fail(
                "E003",
                f"unknown CRDT kind {kind!r} for field {field_name!r}",
                help_text=f"use one of: {', '.join(sorted(CRDT_KINDS))}",
                note="UnknownCRDTKind",
            )
        compiled.append(CRDTField(field_name, _KIND_ALIASES[kind], metadata))
    return CRDTSchema(name=name, fields=tuple(compiled))


# ---------------------------------------------------------------------------
# 63. Single-process Raft-style log with a PyFuture-like yield
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEntry:
    index: int
    term: int
    command: Any
    acks: int
    committed: bool

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "term": self.term,
            "command": self.command,
            "acks": self.acks,
            "committed": self.committed,
        }


@dataclass
class PyFutureStub:
    """A cooperative future stand-in; there is no thread, loop, or IO behind it."""

    label: str
    result_value: Any = None
    is_done: bool = False
    is_cancelled: bool = False

    def done(self) -> bool:
        return self.is_done or self.is_cancelled

    def cancelled(self) -> bool:
        return self.is_cancelled

    def set_result(self, value: Any) -> None:
        self.result_value = value
        self.is_done = True

    def result(self) -> Any:
        return self.result_value

    def cancel(self) -> None:
        self.is_cancelled = True

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "done": self.done(),
            "cancelled": self.is_cancelled,
            "result": self.result_value,
        }


@dataclass
class RaftAwaitable:
    value: Any
    context: _Context
    label: str = "raft-commit"
    stubs: list[PyFutureStub] = field(default_factory=list)

    def __await__(self) -> Generator[PyFutureStub, None, Any]:
        stub = PyFutureStub(self.label)
        self.stubs.append(stub)
        yield stub
        if stub.cancelled():
            self.context.fail(
                "E008",
                "raft future was cancelled before the log entry committed",
                note="ConsensusDrift",
            )
        return stub.result() if stub.done() else self.value


def drive_future(
    awaitable: RaftAwaitable,
    *,
    on_yield: Callable[[PyFutureStub], None] | None = None,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> Any:
    """Drive one cooperative await step without an event loop or scheduler."""
    ctx = (
        awaitable.context
        if span is None and line == 0 and col == 0
        else _context(span, line, col)
    )
    iterator = awaitable.__await__()
    try:
        stub = next(iterator)
        for _ in range(8):
            if on_yield is not None:
                on_yield(stub)
            elif not stub.done():
                stub.set_result(awaitable.value)
            stub = iterator.send(None)
        ctx.fail("E008", "raft awaitable yielded too many times", note="ConsensusDrift")
    except StopIteration as stop:
        return stop.value


@dataclass
class RaftLog:
    """A single-process append-only log with majority arithmetic."""

    cluster_size: int
    term: int = 1
    role: str = "leader"
    entries: list[LogEntry] = field(default_factory=list)
    context: _Context = field(default_factory=lambda: _context(None, 0, 0))

    @property
    def quorum(self) -> int:
        return self.cluster_size // 2 + 1

    @property
    def commit_index(self) -> int:
        return max(
            (entry.index for entry in self.entries if entry.committed), default=0
        )

    @property
    def committed(self) -> tuple[LogEntry, ...]:
        return tuple(entry for entry in self.entries if entry.committed)

    @property
    def committed_commands(self) -> tuple[Any, ...]:
        return tuple(entry.command for entry in self.committed)

    def append(
        self,
        command: Any,
        *,
        acks: int | None = None,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> LogEntry:
        ctx = (
            self.context
            if span is None and line == 0 and col == 0
            else _context(span, line, col)
        )
        count = self.quorum if acks is None else _integer(acks, "acks", ctx, minimum=0)
        if count > self.cluster_size:
            ctx.fail("E003", "acknowledgements cannot exceed the cluster size")
        entry = LogEntry(
            index=len(self.entries) + 1,
            term=self.term,
            command=command,
            acks=count,
            committed=count >= self.quorum,
        )
        self.entries.append(entry)
        return entry

    def future(self) -> RaftAwaitable:
        return RaftAwaitable(self.committed_commands, self.context)

    def describe(self) -> dict[str, Any]:
        return {
            "cluster_size": self.cluster_size,
            "quorum": self.quorum,
            "term": self.term,
            "role": self.role,
            "commit_index": self.commit_index,
            "entries": [entry.describe() for entry in self.entries],
        }


def dsm_raft_quorum_future_yield(
    cluster_size: int,
    commands: Iterable[Any],
    *,
    term: int = 1,
    acks: int | None = None,
    backend: str = "software",
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> RaftLog:
    """Append commands to a local Raft-style log and return an awaitable log.

    ``acks`` models acknowledgements collected from a simulated cluster; a
    majority commits the entry.  ``backend='kernel'`` is rejected with ``E007``
    because this module never runs a real consensus implementation.
    """
    ctx = _context(span, line, col)
    _backend(backend, _SOFTWARE_BACKENDS | {"raft", "quorum"}, ctx, kind="consensus")
    size = _integer(cluster_size, "cluster_size", ctx, minimum=1)
    current_term = _integer(term, "term", ctx, minimum=0)
    log = RaftLog(cluster_size=size, term=current_term, context=ctx)
    for command in _items(commands, "commands", ctx):
        log.append(command, acks=acks, span=span, line=line, col=col)
    return log


# ---------------------------------------------------------------------------
# 64. Rolling Merkle tree over page digests and minimal diff
# ---------------------------------------------------------------------------


def _leaf_digest(page_id: str, data: bytes) -> str:
    payload = b"L" + page_id.encode("utf-8") + b"\x00" + data
    return hashlib.blake2s(payload, digest_size=_HASH_SIZE).hexdigest()


def _node_digest(left: str, right: str) -> str:
    return hashlib.blake2s(
        b"N" + left.encode("ascii") + right.encode("ascii"),
        digest_size=_HASH_SIZE,
    ).hexdigest()


@dataclass(frozen=True)
class MerkleTree:
    """A deterministic Merkle tree; updating a page rolls the root forward."""

    page_ids: tuple[str, ...]
    leaves: dict[str, str]
    levels: tuple[tuple[str, ...], ...]

    @property
    def root(self) -> str:
        return self.levels[-1][0]

    @classmethod
    def from_pages(
        cls,
        pages: Mapping[str, bytes],
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> MerkleTree:
        ctx = _context(span, line, col)
        raw = _mapping(pages, "pages", ctx)
        for pid in raw:
            _text(pid, "page id", ctx)
        page_ids = tuple(sorted(raw))
        if not page_ids:
            return cls((), {}, ((_EMPTY_LEAF,),))
        leaves = {
            pid: _leaf_digest(pid, _bytes(raw[pid], "page data", ctx))
            for pid in page_ids
        }
        width = 1
        while width < len(page_ids):
            width *= 2
        level = tuple(leaves[pid] for pid in page_ids) + (_EMPTY_LEAF,) * (
            width - len(page_ids)
        )
        levels = [level]
        while len(level) > 1:
            level = tuple(
                _node_digest(level[i], level[i + 1]) for i in range(0, len(level), 2)
            )
            levels.append(level)
        return cls(page_ids, leaves, tuple(levels))

    def update(
        self,
        page_id: str,
        data: bytes,
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> MerkleTree:
        """Return a new tree with one page replaced (the rolling update)."""
        ctx = _context(span, line, col)
        pid = _text(page_id, "page_id", ctx)
        payload = _bytes(data, "data", ctx)
        raw = {pid: payload}
        for old in self.page_ids:
            if old != pid:
                raw[old] = b""
        rebuilt = MerkleTree.from_pages(raw, span=span, line=line, col=col)
        # Reuse the existing digests for untouched pages so only the path from
        # the changed leaf to the root needs re-hashing.
        leaves = dict(rebuilt.leaves)
        leaves[pid] = _leaf_digest(pid, payload)
        for other in self.page_ids:
            if other != pid:
                leaves[other] = self.leaves[other]
        return replace(rebuilt, leaves=leaves, levels=_rebuild_levels(leaves))

    def proof(
        self,
        page_id: str,
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> tuple[tuple[str, bool], ...]:
        """Return ``(sibling digest, sibling_is_right)`` pairs for one page."""
        ctx = _context(span, line, col)
        if page_id not in self.page_ids:
            ctx.fail(
                "E006", f"page {page_id!r} is not in the tree", note="PageNotFound"
            )
        index = self.page_ids.index(page_id)
        path: list[tuple[str, bool]] = []
        for level in self.levels[:-1]:
            sibling = index ^ 1
            if sibling < len(level):
                path.append((level[sibling], index % 2 == 0))
            index //= 2
        return tuple(path)

    @staticmethod
    def verify(
        page_id: str,
        data: bytes,
        proof: Iterable[tuple[str, bool]],
        root: str,
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> bool:
        ctx = _context(span, line, col)
        digest = _leaf_digest(_text(page_id, "page_id", ctx), _bytes(data, "data", ctx))
        for sibling, sibling_is_right in _items(proof, "proof", ctx):
            valid_digest = (
                isinstance(sibling, str)
                and len(sibling) == 2 * _HASH_SIZE
                and all(char in "0123456789abcdefABCDEF" for char in sibling)
            )
            if not valid_digest or not isinstance(sibling_is_right, bool):
                ctx.fail("E008", "malformed Merkle proof step", note="WireFormatError")
            digest = (
                _node_digest(digest, sibling)
                if sibling_is_right
                else _node_digest(sibling, digest)
            )
        return digest == root

    def describe(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "pages": list(self.page_ids),
            "levels": [list(l) for l in self.levels],
        }


def _rebuild_levels(leaves: Mapping[str, str]) -> tuple[tuple[str, ...], ...]:
    if not leaves:
        return ((_EMPTY_LEAF,),)
    page_ids = tuple(sorted(leaves))
    width = 1
    while width < len(page_ids):
        width *= 2
    level = tuple(leaves[pid] for pid in page_ids) + (_EMPTY_LEAF,) * (
        width - len(page_ids)
    )
    levels = [level]
    while len(level) > 1:
        level = tuple(
            _node_digest(level[i], level[i + 1]) for i in range(0, len(level), 2)
        )
        levels.append(level)
    return tuple(levels)


@dataclass(frozen=True)
class MerkleDiff:
    old_root: str
    new_root: str
    changed: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    old_tree: MerkleTree
    new_tree: MerkleTree

    @property
    def equal(self) -> bool:
        return self.old_root == self.new_root

    def describe(self) -> dict[str, Any]:
        return {
            "old_root": self.old_root,
            "new_root": self.new_root,
            "changed": list(self.changed),
            "added": list(self.added),
            "removed": list(self.removed),
        }


def dsm_merkle_tree_state_diff(
    old_pages: Mapping[str, bytes],
    new_pages: Mapping[str, bytes],
    *,
    backend: str = "software",
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> MerkleDiff:
    """Hash two page maps and report only the pages whose leaf digests differ.

    This is local hashing in Python.  ``backend='rdma'`` is rejected with
    ``E007``; no real RDMA transfer or memory registration is performed.
    """
    ctx = _context(span, line, col)
    _backend(backend, _SOFTWARE_BACKENDS, ctx, kind="transfer")
    old_tree = MerkleTree.from_pages(old_pages, span=span, line=line, col=col)
    new_tree = MerkleTree.from_pages(new_pages, span=span, line=line, col=col)
    old_ids, new_ids = set(old_tree.page_ids), set(new_tree.page_ids)
    changed = tuple(
        pid
        for pid in sorted(old_ids & new_ids)
        if old_tree.leaves[pid] != new_tree.leaves[pid]
    )
    return MerkleDiff(
        old_root=old_tree.root,
        new_root=new_tree.root,
        changed=changed,
        added=tuple(sorted(new_ids - old_ids)),
        removed=tuple(sorted(old_ids - new_ids)),
        old_tree=old_tree,
        new_tree=new_tree,
    )


# ---------------------------------------------------------------------------
# 65. Hybrid logical clocks, vector clocks, and anomaly detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HLCStamp:
    physical: int
    logical: int = 0
    node: str = ""

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.physical, self.logical, self.node)

    def describe(self) -> dict[str, Any]:
        return {"physical": self.physical, "logical": self.logical, "node": self.node}


@dataclass
class HybridLogicalClock:
    physical: int
    logical: int = 0

    def tick(self) -> HLCStamp:
        self.logical += 1
        return HLCStamp(self.physical, self.logical)

    def observe(self, remote: HLCStamp) -> HLCStamp:
        if remote.physical > self.physical:
            self.physical = remote.physical
            self.logical = remote.logical + 1
        elif remote.physical == self.physical:
            self.logical = max(self.logical, remote.logical) + 1
        else:
            self.logical += 1
        return HLCStamp(self.physical, self.logical, remote.node)

    def describe(self) -> dict[str, Any]:
        return {"physical": self.physical, "logical": self.logical}


@dataclass
class VectorClock:
    counters: dict[str, int] = field(default_factory=dict)

    def tick(self, node: str, amount: int = 1) -> None:
        self.counters[node] = self.counters.get(node, 0) + amount

    def merge(self, other: VectorClock) -> None:
        for node, count in other.counters.items():
            self.counters[node] = max(self.counters.get(node, 0), count)

    def happens_before(self, other: VectorClock) -> bool:
        nodes = set(self.counters) | set(other.counters)
        left = all(self.counters.get(n, 0) <= other.counters.get(n, 0) for n in nodes)
        strict = any(self.counters.get(n, 0) < other.counters.get(n, 0) for n in nodes)
        return left and strict

    def concurrent(self, other: VectorClock) -> bool:
        return not self.happens_before(other) and not other.happens_before(self)

    def copy(self) -> VectorClock:
        return VectorClock(dict(self.counters))


@dataclass(frozen=True)
class CausalEvent:
    event_id: str
    node: str
    vector: Mapping[str, int]
    dependencies: tuple[str, ...] = ()
    hlc: HLCStamp | None = None


@dataclass(frozen=True)
class CausalityReport:
    events: tuple[CausalEvent, ...]
    final_vector: dict[str, int]
    final_hlc: HLCStamp
    happened_before: tuple[tuple[str, str], ...]
    concurrent: tuple[tuple[str, str], ...]

    def describe(self) -> dict[str, Any]:
        return {
            "events": [event.event_id for event in self.events],
            "final_vector": dict(self.final_vector),
            "final_hlc": self.final_hlc.describe(),
            "happened_before": [list(pair) for pair in self.happened_before],
            "concurrent": [list(pair) for pair in self.concurrent],
        }


def dsm_vector_clock_causality(
    events: Iterable[CausalEvent],
    *,
    physical_now: int = 1000,
    max_skew_ms: int = 10_000,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> CausalityReport:
    """Validate a causal event stream with vector clocks and an HLC.

    Duplicate ids, missing dependencies, vector regressions, dependencies not
    reflected in a vector, and implausible HLC timestamps are reported as
    ``E008`` causality anomalies with the caller's span.
    """
    ctx = _context(span, line, col)
    now = _integer(physical_now, "physical_now", ctx, minimum=0)
    skew = _integer(max_skew_ms, "max_skew_ms", ctx, minimum=0)
    processed: list[CausalEvent] = []
    seen: dict[str, tuple[str, dict[str, int]]] = {}
    vector = VectorClock()
    clock = HybridLogicalClock(now)
    for event in _items(events, "events", ctx):
        if not isinstance(event, CausalEvent):
            ctx.fail(
                "E001", "events must contain CausalEvent records", note="InvalidInput"
            )
        event_id = _text(event.event_id, "event_id", ctx)
        node = _text(event.node, "node", ctx)
        if event_id in seen:
            ctx.fail(
                "E008", f"duplicate event id {event_id!r}", note="CausalityViolation"
            )
        raw_vector = _mapping(event.vector, "vector", ctx)
        for key, count in raw_vector.items():
            _text(key, "vector node", ctx)
            _integer(count, "vector counter", ctx, minimum=0)
        own = raw_vector.get(node, 0)
        if own <= seen_node_count(seen, node):
            ctx.fail(
                "E008",
                f"non-monotonic vector clock for node {node!r}: event {event_id!r}",
                note="CausalityViolation",
            )
        for dep in _items(event.dependencies, "dependencies", ctx):
            if not isinstance(dep, str) or dep not in seen:
                ctx.fail(
                    "E008",
                    f"event {event_id!r} depends on unknown or future event {dep!r}",
                    note="CausalityViolation",
                )
            dep_vector = seen[dep][1]
            if any(
                raw_vector.get(other_node, 0) < count
                for other_node, count in dep_vector.items()
            ):
                ctx.fail(
                    "E008",
                    f"vector clock of {event_id!r} omits dependency {dep!r}",
                    note="CausalityViolation",
                )
        if event.hlc is not None:
            stamp = event.hlc
            if not isinstance(stamp, HLCStamp):
                ctx.fail("E001", "hlc must be an HLCStamp", note="InvalidInput")
            if (
                not isinstance(stamp.physical, int)
                or not isinstance(stamp.logical, int)
                or stamp.physical < 0
                or stamp.logical < 0
            ):
                ctx.fail("E008", f"malformed HLC stamp on event {event_id!r}")
            if stamp.physical > now + skew:
                ctx.fail(
                    "E008",
                    f"HLC timestamp on {event_id!r} is too far in the future",
                    note="CausalityViolation",
                )
            clock.observe(stamp)
        clock.tick()
        vector.merge(VectorClock(dict(raw_vector)))
        seen[event_id] = (node, dict(raw_vector))
        processed.append(event)

    ordered: list[tuple[str, str]] = []
    concurrent: list[tuple[str, str]] = []
    for i, left in enumerate(processed):
        for right in processed[i + 1 :]:
            lv, rv = VectorClock(dict(left.vector)), VectorClock(dict(right.vector))
            if lv.happens_before(rv):
                ordered.append((left.event_id, right.event_id))
            elif lv.concurrent(rv):
                concurrent.append((left.event_id, right.event_id))
    return CausalityReport(
        events=tuple(processed),
        final_vector=dict(vector.counters),
        final_hlc=HLCStamp(clock.physical, clock.logical),
        happened_before=tuple(ordered),
        concurrent=tuple(concurrent),
    )


def seen_node_count(seen: Mapping[str, tuple[str, dict[str, int]]], node: str) -> int:
    """Highest counter already observed for ``node`` across processed events."""
    return max((vector.get(node, 0) for _, vector in seen.values()), default=0)


# ---------------------------------------------------------------------------
# 66. Split-brain fencing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FenceClaim:
    node: str
    token: int
    epoch: int = 0


@dataclass(frozen=True)
class FenceResult:
    resource: str
    winner: FenceClaim
    stale: tuple[FenceClaim, ...]
    epoch: int

    def authorize(
        self,
        claim: FenceClaim,
        *,
        span: Span = None,
        line: int = 0,
        col: int = 0,
    ) -> FenceClaim:
        ctx = _context(span, line, col)
        if not isinstance(claim, FenceClaim):
            ctx.fail("E001", "claim must be a FenceClaim", note="InvalidInput")
        if (claim.node, claim.token, claim.epoch) != (
            self.winner.node,
            self.winner.token,
            self.winner.epoch,
        ):
            ctx.fail(
                "E008",
                f"stale fencing token {claim.token} for {self.resource!r} from {claim.node!r}",
                help_text="reload the object from its current owner and retry",
                note="StaleFenceError",
            )
        return claim

    def describe(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "winner": {"node": self.winner.node, "token": self.winner.token},
            "stale": [{"node": c.node, "token": c.token} for c in self.stale],
            "epoch": self.epoch,
        }


def _validate_claim(claim: Any, ctx: _Context) -> FenceClaim:
    if not isinstance(claim, FenceClaim):
        ctx.fail("E001", "claims must contain FenceClaim records", note="InvalidInput")
    _text(claim.node, "claim node", ctx)
    _integer(claim.token, "claim token", ctx, minimum=0)
    _integer(claim.epoch, "claim epoch", ctx, minimum=0)
    return claim


def dsm_split_brain_partition_fence(
    resource: str,
    claims: Iterable[FenceClaim],
    *,
    current: FenceClaim | None = None,
    strict: bool = True,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> FenceResult:
    """Pick the unique current owner of ``resource`` from fencing claims.

    Two nodes presenting the same ``(epoch, token)`` are a split brain and are
    rejected with ``E008``.  With ``strict=True`` (the default) any stale claim
    is also an ``E008`` failure; ``strict=False`` returns the stale claims so a
    controller can inspect the partition.
    """
    ctx = _context(span, line, col)
    key = _text(resource, "resource", ctx)
    if not isinstance(strict, bool):
        ctx.fail("E001", "strict must be a boolean", note="InvalidInput")
    if current is not None:
        _validate_claim(current, ctx)
    checked = [_validate_claim(claim, ctx) for claim in _items(claims, "claims", ctx)]
    if not checked:
        ctx.fail("E003", "at least one fencing claim is required")
    by_token: dict[tuple[int, int], str] = {}
    for claim in checked:
        marker = (claim.epoch, claim.token)
        if marker in by_token and by_token[marker] != claim.node:
            ctx.fail(
                "E008",
                f"split brain: token {claim.token} in epoch {claim.epoch} claimed by "
                f"{by_token[marker]!r} and {claim.node!r}",
                note="SplitBrainError",
            )
        by_token[marker] = claim.node

    ranked = sorted(checked, key=lambda c: (c.epoch, c.token, c.node))
    winner = ranked[-1]
    stale: list[FenceClaim] = []
    for claim in checked:
        is_renewal = current is not None and (claim.node, claim.token, claim.epoch) == (
            current.node,
            current.token,
            current.epoch,
        )
        behind_current = current is not None and (claim.epoch, claim.token) <= (
            current.epoch,
            current.token,
        )
        is_stale = claim is not winner and (
            behind_current or (claim.epoch, claim.token) < (winner.epoch, winner.token)
        )
        if claim is winner and behind_current and not is_renewal:
            is_stale = True
        if is_stale:
            stale.append(claim)
    if strict and stale:
        oldest = stale[0]
        ctx.fail(
            "E008",
            f"stale primary {oldest.node!r} holds token {oldest.token} in epoch {oldest.epoch}",
            help_text="reload the object from its current owner and retry",
            note="StaleFenceError",
        )
    return FenceResult(key, winner, tuple(stale), winner.epoch)


# ---------------------------------------------------------------------------
# 67. MVCC snapshot isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MVCCBatch:
    txn: str
    reads: tuple[str, ...] = ()
    writes: Mapping[str, Any] = field(default_factory=dict)
    commit: bool = True
    snapshot_ts: int | None = None


@dataclass(frozen=True)
class MVCCVersion:
    commit_ts: int
    txn: str
    value: Any


@dataclass(frozen=True)
class MVCCRead:
    txn: str
    key: str
    value: Any
    snapshot_ts: int


@dataclass(frozen=True)
class MVCCReport:
    reads: tuple[MVCCRead, ...]
    committed: tuple[str, ...]
    rolled_back: tuple[str, ...]
    versions: dict[str, tuple[MVCCVersion, ...]]

    def describe(self) -> dict[str, Any]:
        return {
            "reads": [
                {"txn": r.txn, "key": r.key, "value": r.value} for r in self.reads
            ],
            "committed": list(self.committed),
            "rolled_back": list(self.rolled_back),
            "versions": {
                key: [
                    {"ts": v.commit_ts, "txn": v.txn, "value": v.value}
                    for v in versions
                ]
                for key, versions in self.versions.items()
            },
        }


def dsm_mvcc_snapshot_isolation(
    batches: Iterable[MVCCBatch],
    *,
    initial: Mapping[str, Any] | None = None,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> MVCCReport:
    """Run atomic batches over a versioned store with snapshot isolation.

    Reads see the newest version committed at or before the batch's snapshot
    timestamp.  A writer whose key was committed after its snapshot aborts
    with ``E008`` (first-committer-wins).
    """
    ctx = _context(span, line, col)
    seed = {} if initial is None else _mapping(initial, "initial", ctx)
    versions: dict[str, list[MVCCVersion]] = {
        _text(key, "initial key", ctx): [MVCCVersion(0, "init", value)]
        for key, value in seed.items()
    }
    clock = 1
    reads: list[MVCCRead] = []
    committed: list[str] = []
    rolled_back: list[str] = []
    transactions: set[str] = set()

    for batch in _items(batches, "batches", ctx):
        if not isinstance(batch, MVCCBatch):
            ctx.fail(
                "E001", "batches must contain MVCCBatch records", note="InvalidInput"
            )
        txn = _text(batch.txn, "txn", ctx)
        if txn in transactions:
            ctx.fail("E003", f"duplicate transaction id {txn!r}")
        transactions.add(txn)
        if not isinstance(batch.commit, bool):
            ctx.fail("E001", "commit must be a boolean", note="InvalidInput")
        snapshot = (
            clock - 1
            if batch.snapshot_ts is None
            else _integer(batch.snapshot_ts, "snapshot_ts", ctx, minimum=0)
        )
        if snapshot >= clock:
            ctx.fail(
                "E008",
                f"snapshot timestamp {snapshot} is in the future",
                note="StateMismatchError",
            )
        writes = _mapping(batch.writes, "writes", ctx)
        for key in writes:
            _text(key, "write key", ctx)
        for key in _items(batch.reads, "reads", ctx):
            _text(key, "read key", ctx)
            visible = None
            for version in reversed(versions.get(key, [])):
                if version.commit_ts <= snapshot:
                    visible = version.value
                    break
            reads.append(MVCCRead(txn, key, visible, snapshot))

        if not batch.commit:
            rolled_back.append(txn)
            continue
        if not writes:
            committed.append(txn)
            continue
        for key in writes:
            latest = max(
                (version.commit_ts for version in versions.get(key, [])), default=-1
            )
            if latest > snapshot:
                ctx.fail(
                    "E008",
                    f"write conflict on {key!r}: committed at {latest} after snapshot {snapshot}",
                    note="StateMismatchError",
                )
        stamp = clock
        clock += 1
        for key, value in writes.items():
            versions.setdefault(key, []).append(MVCCVersion(stamp, txn, value))
        committed.append(txn)

    return MVCCReport(
        reads=tuple(reads),
        committed=tuple(committed),
        rolled_back=tuple(rolled_back),
        versions={key: tuple(history) for key, history in versions.items()},
    )


# ---------------------------------------------------------------------------
# 68. False-sharing invalidation detection and page splitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageAccess:
    page: str
    node: str
    offset: int
    size: int = 1
    write: bool = True


@dataclass(frozen=True)
class SubPage:
    page: str
    index: int
    start: int
    end: int
    writes: int
    writers: tuple[str, ...]
    hot: bool


@dataclass(frozen=True)
class SplitPlan:
    page: str
    invalidations: int
    sub_pages: tuple[SubPage, ...]

    @property
    def hot_regions(self) -> tuple[SubPage, ...]:
        return tuple(sub for sub in self.sub_pages if sub.hot)

    def describe(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "invalidations": self.invalidations,
            "sub_pages": [
                {
                    "index": s.index,
                    "range": [s.start, s.end],
                    "writers": list(s.writers),
                    "hot": s.hot,
                }
                for s in self.sub_pages
            ],
        }


@dataclass(frozen=True)
class SplitReport:
    plans: tuple[SplitPlan, ...]
    invalidations: dict[str, int]
    accesses: int

    def describe(self) -> dict[str, Any]:
        return {
            "plans": [plan.describe() for plan in self.plans],
            "invalidations": dict(self.invalidations),
            "accesses": self.accesses,
        }


def dsm_false_sharing_page_split(
    accesses: Iterable[PageAccess],
    *,
    page_size: int = 4096,
    partitions: int = 2,
    invalidation_threshold: int = 2,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> SplitReport:
    """Count cross-node write invalidations and split ping-ponged pages.

    An invalidation is a write by a different node than the previous writer of
    the same page.  Pages at or above ``invalidation_threshold`` are divided
    into deterministic equal-width sub-pages; sub-pages written by more than
    one node are flagged as hot regions.
    """
    ctx = _context(span, line, col)
    size = _integer(page_size, "page_size", ctx, minimum=1)
    parts = _integer(partitions, "partitions", ctx, minimum=1)
    threshold = _integer(
        invalidation_threshold, "invalidation_threshold", ctx, minimum=1
    )
    by_page: dict[str, list[PageAccess]] = {}
    total = 0
    for access in _items(accesses, "accesses", ctx):
        if not isinstance(access, PageAccess):
            ctx.fail(
                "E001", "accesses must contain PageAccess records", note="InvalidInput"
            )
        pid = _text(access.page, "access page", ctx)
        _text(access.node, "access node", ctx)
        offset = _integer(access.offset, "offset", ctx, minimum=0)
        width = _integer(access.size, "size", ctx, minimum=1)
        if offset + width > size:
            ctx.fail("E003", f"access at {offset}+{width} exceeds page size {size}")
        if not isinstance(access.write, bool):
            ctx.fail("E001", "write must be a boolean", note="InvalidInput")
        by_page.setdefault(pid, []).append(access)
        total += 1

    invalidations: dict[str, int] = {}
    plans: list[SplitPlan] = []
    for pid in sorted(by_page):
        writes = [a for a in by_page[pid] if a.write]
        previous: str | None = None
        count = 0
        for access in writes:
            if previous is not None and access.node != previous:
                count += 1
            previous = access.node
        invalidations[pid] = count
        if count < threshold:
            continue
        sub_pages: list[SubPage] = []
        for index in range(parts):
            start = index * size // parts
            end = (index + 1) * size // parts
            touched = [
                a for a in writes if a.offset < end and a.offset + a.size > start
            ]
            writers = tuple(sorted({a.node for a in touched}))
            sub_pages.append(
                SubPage(pid, index, start, end, len(touched), writers, len(writers) > 1)
            )
        plans.append(SplitPlan(pid, count, tuple(sub_pages)))
    return SplitReport(tuple(plans), invalidations, total)


# ---------------------------------------------------------------------------
# 69. Lease-based distributed GC with epoch reclamation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaseGrant:
    object_id: str
    holder: str
    token: int
    epoch: int
    expires_at: int

    def describe(self) -> dict[str, Any]:
        return {
            "object": self.object_id,
            "holder": self.holder,
            "token": self.token,
            "epoch": self.epoch,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class Acquire:
    object_id: str
    holder: str
    ttl: int


@dataclass(frozen=True)
class Release:
    object_id: str
    holder: str


@dataclass(frozen=True)
class Tick:
    ticks: int = 1


@dataclass(frozen=True)
class NewEpoch:
    epoch: int


@dataclass(frozen=True)
class GCReport:
    epoch: int
    now: int
    reclaimed: tuple[str, ...]
    live: dict[str, tuple[LeaseGrant, ...]]
    grants_issued: int
    expirations: int
    log: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "now": self.now,
            "reclaimed": list(self.reclaimed),
            "live": {
                key: [g.describe() for g in grants] for key, grants in self.live.items()
            },
            "grants_issued": self.grants_issued,
            "expirations": self.expirations,
            "log": list(self.log),
        }


def dsm_distributed_gc_lease(
    operations: Iterable[Acquire | Release | Tick | NewEpoch],
    *,
    epoch: int = 0,
    now: int = 0,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> GCReport:
    """Reclaim objects when their current-epoch lease refcount reaches zero.

    Leases are logical (a ``Tick`` advances a counter), tokens are monotonic
    per object, and a new epoch revokes every older grant.  An unknown or
    stale release and an epoch regression are ``E008`` boundary faults.
    """
    ctx = _context(span, line, col)
    current_epoch = _integer(epoch, "epoch", ctx, minimum=0)
    clock = _integer(now, "now", ctx, minimum=0)
    grants: dict[str, list[LeaseGrant]] = {}
    tokens: dict[str, int] = {}
    counted_expirations: set[tuple[str, int, int]] = set()
    reclaimed: list[str] = []
    log: list[str] = []
    issued = 0
    expired = 0

    def active(object_id: str) -> list[LeaseGrant]:
        return [
            grant
            for grant in grants.get(object_id, [])
            if grant.epoch == current_epoch and grant.expires_at > clock
        ]

    def reclaim() -> None:
        for object_id in sorted(grants):
            if object_id not in reclaimed and not active(object_id):
                reclaimed.append(object_id)
                log.append(f"reclaim:{object_id}:epoch:{current_epoch}")

    for op in _items(operations, "operations", ctx):
        if isinstance(op, NewEpoch):
            target = _integer(op.epoch, "epoch", ctx, minimum=0)
            if target <= current_epoch:
                ctx.fail(
                    "E008",
                    "epoch must increase monotonically",
                    note="StateMismatchError",
                )
            current_epoch = target
            log.append(f"epoch:{current_epoch}")
            reclaim()
        elif isinstance(op, Tick):
            advance = _integer(op.ticks, "ticks", ctx, minimum=0)
            clock += advance
            for object_id, history in grants.items():
                for grant in history:
                    if grant.expires_at <= clock and grant.epoch == current_epoch:
                        marker = (object_id, grant.token, grant.epoch)
                        if marker not in counted_expirations:
                            counted_expirations.add(marker)
                            expired += 1
                            log.append(f"expire:{object_id}:{grant.holder}")
            reclaim()
        elif isinstance(op, Acquire):
            object_id = _text(op.object_id, "object_id", ctx)
            holder = _text(op.holder, "holder", ctx)
            ttl = _integer(op.ttl, "ttl", ctx, minimum=1)
            if object_id in reclaimed:
                ctx.fail(
                    "E008",
                    f"object {object_id!r} was already reclaimed",
                    note="StateMismatchError",
                )
            tokens[object_id] = tokens.get(object_id, 0) + 1
            grant = LeaseGrant(
                object_id, holder, tokens[object_id], current_epoch, clock + ttl
            )
            grants.setdefault(object_id, []).append(grant)
            issued += 1
            log.append(f"acquire:{object_id}:{holder}:{grant.token}")
        elif isinstance(op, Release):
            object_id = _text(op.object_id, "object_id", ctx)
            holder = _text(op.holder, "holder", ctx)
            matching = [g for g in active(object_id) if g.holder == holder]
            if not matching:
                ctx.fail(
                    "E008",
                    f"stale or unknown lease for {object_id!r} held by {holder!r}",
                    note="LeaseExpiredError",
                )
            latest = max(matching, key=lambda g: g.token)
            grants[object_id].remove(latest)
            log.append(f"release:{object_id}:{holder}:{latest.token}")
            reclaim()
        else:
            ctx.fail(
                "E001", "operations must be Acquire, Release, Tick, or NewEpoch records"
            )

    reclaim()
    return GCReport(
        epoch=current_epoch,
        now=clock,
        reclaimed=tuple(reclaimed),
        live={
            key: tuple(sorted(active(key), key=lambda g: g.token))
            for key in sorted(grants)
            if active(key)
        },
        grants_issued=issued,
        expirations=expired,
        log=tuple(log),
    )


# ---------------------------------------------------------------------------
# 70. Atomic cross-node CAS via a two-phase coordinator simulation
# ---------------------------------------------------------------------------


@dataclass
class CASNode:
    node_id: str
    value: Any
    pending: Any = None
    ballot: int = 0
    phase: str = "idle"

    def describe(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "value": self.value,
            "ballot": self.ballot,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class CASResult:
    success: bool
    expected: Any
    new: Any
    committed: tuple[str, ...]
    failed: tuple[str, ...]
    rounds: tuple[str, ...]
    ballots: int
    nodes: tuple[CASNode, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "expected": self.expected,
            "new": self.new,
            "committed": list(self.committed),
            "failed": list(self.failed),
            "rounds": list(self.rounds),
            "ballots": self.ballots,
            "nodes": [node.describe() for node in self.nodes],
        }


def dsm_atomic_cross_node_cas(
    nodes: Mapping[str, Any] | Iterable[CASNode],
    expected: Any,
    new: Any,
    *,
    backend: str = "software",
    drift_after_prepare: Mapping[str, Any] | None = None,
    span: Span = None,
    line: int = 0,
    col: int = 0,
) -> CASResult:
    """Run a two-phase compare-and-swap across simulated nodes.

    Phase one prepares every replica if its value matches ``expected``; phase
    two commits the new value everywhere.  A value that changes between the
    two phases is consensus drift and raises ``E008`` with the caller's span.
    ``backend='kernel'`` is rejected with ``E007``.
    """
    ctx = _context(span, line, col)
    _backend(
        backend,
        _SOFTWARE_BACKENDS | {"2pc", "paxos-like"},
        ctx,
        kind="cas-coordinator",
    )
    if isinstance(nodes, Mapping):
        replicas = [
            CASNode(_text(key, "node id", ctx), value) for key, value in nodes.items()
        ]
    else:
        replicas = []
        for node in _items(nodes, "nodes", ctx):
            if not isinstance(node, CASNode):
                ctx.fail(
                    "E001", "nodes must contain CASNode records", note="InvalidInput"
                )
            replicas.append(replace(node))
    if not replicas:
        ctx.fail("E003", "at least one node is required for a cross-node CAS")
    seen: set[str] = set()
    for node in replicas:
        node_id = _text(node.node_id, "node id", ctx)
        if node_id in seen:
            ctx.fail("E003", f"duplicate node id {node_id!r}")
        seen.add(node_id)
        node.phase = "idle"
        node.ballot = 0
        node.pending = None

    drift = (
        {}
        if drift_after_prepare is None
        else _mapping(drift_after_prepare, "drift_after_prepare", ctx)
    )
    for node_id in drift:
        if node_id not in seen:
            ctx.fail("E003", f"drift references unknown node {node_id!r}")
    before = {node.node_id: node.value for node in replicas}

    failed = tuple(node.node_id for node in replicas if node.value != expected)
    if failed:
        for node in replicas:
            node.phase = "aborted"
        return CASResult(
            False, expected, new, (), failed, ("prepare",), 1, tuple(replicas)
        )

    for node in replicas:
        node.phase = "prepared"
        node.ballot = 1
        node.pending = new
    for node_id, value in drift.items():
        for node in replicas:
            if node.node_id == node_id:
                node.value = value
    if any(node.value != before[node.node_id] for node in replicas):
        ctx.fail(
            "E008",
            "consensus drift: a node changed its value between prepare and commit",
            note="ConsensusDrift",
        )
    for node in replicas:
        node.value = new
        node.pending = None
        node.phase = "committed"
    return CASResult(
        True,
        expected,
        new,
        tuple(node.node_id for node in replicas),
        (),
        ("prepare", "commit"),
        1,
        tuple(replicas),
    )
