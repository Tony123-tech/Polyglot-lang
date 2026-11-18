"""Network connections that survive actor migration.

A migration cannot move a live kernel socket: the peer keeps talking to the
source host's address.  The runtime therefore separates *connection identity*
from *connection ownership*.

* :class:`ConnectionAnchor` lives on a stable node and owns the socket, the
  sequenced inbound buffer, and the outbound replay log.
* :class:`ChannelHandle` is the small wire value that moves with the actor.  It
  names the anchor, the inbound cursor the actor has consumed, and the next
  outbound sequence it will write.
* :class:`AnchorChannel` is the actor-facing facade.  After migration it
  re-attaches to the same anchor with a higher fencing token and resumes from
  the cursor, so no byte is lost or delivered twice.

Inbound data is numbered; consumers acknowledge a cursor.  Outbound writes are
numbered and idempotent: replaying sequence *n* after a reconnect returns the
original acknowledgement instead of writing twice.  Every mutating operation
validates ``(owner, epoch, fencing)``; a stale owner gets an ``E008`` diagnostic
instead of corrupting the stream.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from distributed.diagnostics import (
    DistributedError,
    protocol_error,
    stale_fence,
    wire_error,
)
from distributed.transmute import register_type

MAX_MESSAGE_BYTES: Final = 16 * 1024 * 1024
MAX_BUFFERED: Final = 1_000_000
LENGTH: Final = struct.Struct(">I")


@dataclass
class ChannelHandle:
    """The wire form of a channel that an actor carries across nodes.

    The handle is intentionally mutable: a successful ``recv`` advances its
    inbound cursor in place, so the same object held in task arguments, journal
    operations, and actor state always names the next unconsumed message.
    """

    anchor_id: str
    owner: str = ""
    epoch: int = 0
    fencing: int = 0
    inbound: int = 0
    outbound: int = 0
    anchor_node: str = ""
    kind: str = "channel"
    closed: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "owner": self.owner,
            "epoch": self.epoch,
            "fencing": self.fencing,
            "inbound": self.inbound,
            "outbound": self.outbound,
            "anchor_node": self.anchor_node,
            "kind": self.kind,
            "closed": self.closed,
        }


register_type(ChannelHandle, name="ChannelHandle")


@dataclass(frozen=True)
class Inbound:
    seq: int
    data: bytes


@dataclass(frozen=True)
class Outbound:
    seq: int
    data: bytes


class ConnectionAnchor:
    """Owner-side durable buffer for one logical connection."""

    def __init__(
        self,
        anchor_id: str,
        *,
        node: str = "",
        owner: str = "",
        epoch: int = 0,
        fencing: int = 1,
        max_buffered: int = MAX_BUFFERED,
        max_message: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if not anchor_id:
            raise DistributedError(
                "an anchor needs a non-empty id",
                code="E003",
                note="ProtocolError",
            )
        self.anchor_id = anchor_id
        self.node = node
        self.owner = owner
        self.epoch = int(epoch)
        self.fencing = int(fencing)
        self.max_buffered = max_buffered
        self.max_message = max_message
        self.created_at = time.time()
        self.closed = False
        self._lock = threading.RLock()
        self._inbound: list[Inbound] = []
        self._inbound_next = 0
        self._inbound_acked = 0
        self._outbound: list[Outbound] = []
        self._outbound_next = 0
        self._outbound_done = 0
        self._sent: dict[int, bytes] = {}
        self.delivered = 0
        self.sent = 0
        self.dropped = 0

    # -- cursors ----------------------------------------------------------
    @property
    def inbound_cursor(self) -> int:
        return self._inbound_acked

    @property
    def inbound_next(self) -> int:
        with self._lock:
            return self._inbound_next

    @property
    def outbound_cursor(self) -> int:
        with self._lock:
            return self._outbound_done

    @property
    def outbound_next(self) -> int:
        with self._lock:
            return self._outbound_next

    def handle(self, *, owner: str | None = None) -> ChannelHandle:
        with self._lock:
            return ChannelHandle(
                anchor_id=self.anchor_id,
                owner=owner or self.owner,
                epoch=self.epoch,
                fencing=self.fencing,
                inbound=self._inbound_acked,
                outbound=self._outbound_done,
                anchor_node=self.node,
                closed=self.closed,
            )

    # -- ownership --------------------------------------------------------
    def bind(self, owner: str, *, epoch: int, fencing: int) -> None:
        """Give the anchor to ``owner`` after cutting off the previous one."""
        with self._lock:
            if fencing <= self.fencing:
                raise stale_fence(
                    f"cannot bind anchor {self.anchor_id!r} to {owner!r}: "
                    f"fencing {fencing} is not newer than {self.fencing}",
                    context={"anchor": self.anchor_id},
                )
            if epoch < self.epoch:
                raise stale_fence(
                    f"cannot bind anchor {self.anchor_id!r} to epoch {epoch}: "
                    f"current epoch is {self.epoch}",
                    context={"anchor": self.anchor_id},
                )
            self.owner = owner
            self.epoch = epoch
            self.fencing = fencing

    def _validate(
        self,
        *,
        owner: str | None,
        epoch: int | None,
        fencing: int | None,
        operation: str,
    ) -> None:
        if self.closed:
            raise protocol_error(
                f"anchor {self.anchor_id!r} is closed",
                context={"anchor": self.anchor_id, "operation": operation},
            )
        if owner is not None and owner != self.owner:
            raise stale_fence(
                f"anchor {self.anchor_id!r} is owned by {self.owner!r}, not {owner!r}",
                context={
                    "anchor": self.anchor_id,
                    "operation": operation,
                    "owner": owner,
                },
            )
        if epoch is not None and epoch != self.epoch:
            raise stale_fence(
                f"anchor {self.anchor_id!r} is at epoch {self.epoch}, not {epoch}",
                context={"anchor": self.anchor_id, "operation": operation},
            )
        if fencing is not None and fencing != self.fencing:
            raise stale_fence(
                f"anchor {self.anchor_id!r} has fencing {self.fencing}, not {fencing}",
                context={
                    "anchor": self.anchor_id,
                    "operation": operation,
                    "fencing": fencing,
                },
            )

    # -- inbound ----------------------------------------------------------
    def deliver(self, data: bytes) -> int:
        """Append one inbound message and return its sequence number."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise DistributedError(
                f"inbound data must be bytes, not {type(data).__name__}",
                code="E003",
                note="ProtocolError",
            )
        payload = bytes(data)
        if len(payload) > self.max_message:
            raise wire_error(
                f"message is too large: {len(payload)} bytes "
                f"(limit {self.max_message})",
                context={"anchor": self.anchor_id},
            )
        with self._lock:
            self._validate(
                owner=None,
                epoch=None,
                fencing=None,
                operation="deliver",
            )
            if len(self._inbound) >= self.max_buffered:
                raise DistributedError(
                    f"anchor {self.anchor_id!r} inbound buffer is full",
                    note="StateTooLargeError",
                    help_text="the consuming actor is not draining its channel",
                    context={"anchor": self.anchor_id},
                )
            seq = self._inbound_next
            self._inbound.append(Inbound(seq, payload))
            self._inbound_next += 1
            self.delivered += 1
            return seq

    def recv(
        self,
        cursor: int | None = None,
        *,
        limit: int = 64,
        owner: str | None = None,
        epoch: int | None = None,
        fencing: int | None = None,
    ) -> tuple[list[Inbound], int]:
        """Return buffered messages at or after ``cursor`` and the new cursor.

        The caller acknowledges by passing the returned cursor back.  Reading
        does not consume; :meth:`ack` does.  This makes a reconnect after a
        migration a pure replay.
        """
        with self._lock:
            self._validate(owner=owner, epoch=epoch, fencing=fencing, operation="recv")
            start = self._inbound_acked if cursor is None else int(cursor)
            if start < 0 or start > self._inbound_next:
                raise wire_error(
                    f"inbound cursor {start} is outside [0, {self._inbound_next}]",
                    context={"anchor": self.anchor_id},
                )
            pending = [item for item in self._inbound if item.seq >= start]
            return pending[:limit], (
                pending[0].seq + len(pending[:limit])
                if len(pending) > limit
                else self._inbound_next
            )

    def ack(self, cursor: int) -> None:
        """Discard inbound messages below ``cursor``."""
        with self._lock:
            if cursor < self._inbound_acked or cursor > self._inbound_next:
                raise wire_error(
                    f"ack cursor {cursor} is outside "
                    f"[{self._inbound_acked}, {self._inbound_next}]",
                    context={"anchor": self.anchor_id},
                )
            if cursor == self._inbound_acked:
                return
            self._inbound = [item for item in self._inbound if item.seq >= cursor]
            self.dropped += cursor - self._inbound_acked
            self._inbound_acked = cursor

    # -- outbound ---------------------------------------------------------
    def send(
        self,
        seq: int,
        data: bytes,
        *,
        owner: str | None = None,
        epoch: int | None = None,
        fencing: int | None = None,
    ) -> bool:
        """Record an outbound write.  Returns ``False`` for a replay."""
        payload = bytes(data)
        if len(payload) > self.max_message:
            raise wire_error(
                f"message is too large: {len(payload)} bytes "
                f"(limit {self.max_message})",
                context={"anchor": self.anchor_id},
            )
        with self._lock:
            self._validate(owner=owner, epoch=epoch, fencing=fencing, operation="send")
            if seq < 0:
                raise wire_error(f"negative outbound sequence {seq}")
            if seq > self._outbound_next:
                raise wire_error(
                    f"outbound sequence gap: got {seq}, expected at most "
                    f"{self._outbound_next}",
                    context={"anchor": self.anchor_id},
                )
            previous = self._sent.get(seq)
            if previous is not None:
                if previous != payload:
                    raise wire_error(
                        f"outbound replay of sequence {seq} carries different bytes",
                        note="StateMismatchError",
                        help_text=("idempotent writes must replay the exact payload"),
                        context={"anchor": self.anchor_id, "seq": seq},
                    )
                return False
            if len(self._outbound) >= self.max_buffered:
                raise DistributedError(
                    f"anchor {self.anchor_id!r} outbound buffer is full",
                    note="StateTooLargeError",
                    context={"anchor": self.anchor_id},
                )
            self._sent[seq] = payload
            self._outbound.append(Outbound(seq, payload))
            if seq == self._outbound_next:
                self._outbound_next += 1
            self.sent += 1
            return True

    def pending_outbound(self, *, limit: int = 64) -> list[Outbound]:
        with self._lock:
            return [item for item in self._outbound if item.seq >= self._outbound_done][
                :limit
            ]

    def complete(self, seq: int) -> None:
        """Mark every outbound message up to ``seq`` as written to the peer."""
        with self._lock:
            if seq < self._outbound_done or seq >= self._outbound_next + 1:
                raise wire_error(
                    f"outbound completion {seq} is outside "
                    f"[{self._outbound_done}, {self._outbound_next}]",
                    context={"anchor": self.anchor_id},
                )
            self._outbound_done = max(self._outbound_done, seq)
            self._outbound = [
                item for item in self._outbound if item.seq >= self._outbound_done
            ]
            for old in [s for s in self._sent if s < self._outbound_done]:
                del self._sent[old]

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._inbound.clear()
            self._outbound.clear()
            self._sent.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "anchor_id": self.anchor_id,
                "node": self.node,
                "owner": self.owner,
                "epoch": self.epoch,
                "fencing": self.fencing,
                "inbound": [item.data for item in self._inbound],
                "inbound_acked": self._inbound_acked,
                "inbound_next": self._inbound_next,
                "outbound": [(item.seq, item.data) for item in self._outbound],
                "outbound_done": self._outbound_done,
                "outbound_next": self._outbound_next,
                "sent": dict(self._sent),
                "closed": self.closed,
            }

    def restore(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.node = str(payload.get("node", self.node))
            self.owner = str(payload.get("owner", self.owner))
            self.epoch = int(payload.get("epoch", self.epoch))
            self.fencing = int(payload.get("fencing", self.fencing))
            self._inbound_acked = int(payload.get("inbound_acked", self._inbound_acked))
            self._inbound_next = int(payload.get("inbound_next", self._inbound_next))
            raw_inbound = payload.get("inbound", [])
            base = self._inbound_next - len(raw_inbound)
            self._inbound = [
                Inbound(base + index, bytes(data))
                for index, data in enumerate(raw_inbound)
            ]
            self._outbound_done = int(payload.get("outbound_done", self._outbound_done))
            self._outbound_next = int(payload.get("outbound_next", self._outbound_next))
            self._outbound = [
                Outbound(int(seq), bytes(data))
                for seq, data in payload.get("outbound", [])
            ]
            self._sent = {
                int(seq): bytes(data) for seq, data in payload.get("sent", {}).items()
            }
            self.closed = bool(payload.get("closed", self.closed))


class AnchorRegistry:
    """The node-local set of anchors plus its control-plane RPC surface."""

    def __init__(self, node: str = "local") -> None:
        self.node = node
        self._anchors: dict[str, ConnectionAnchor] = {}
        self._lock = threading.RLock()
        self._counter = 0

    # -- local API --------------------------------------------------------
    def open(
        self,
        anchor_id: str | None = None,
        *,
        owner: str = "",
        epoch: int = 0,
        fencing: int = 1,
    ) -> ConnectionAnchor:
        with self._lock:
            if anchor_id is None:
                self._counter += 1
                anchor_id = f"{self.node}:chan-{self._counter}"
            if anchor_id in self._anchors:
                raise protocol_error(
                    f"anchor {anchor_id!r} already exists on {self.node}",
                    context={"anchor": anchor_id},
                )
            anchor = ConnectionAnchor(
                anchor_id,
                node=self.node,
                owner=owner,
                epoch=epoch,
                fencing=fencing,
            )
            self._anchors[anchor_id] = anchor
            return anchor

    def get(self, anchor_id: str) -> ConnectionAnchor:
        with self._lock:
            anchor = self._anchors.get(anchor_id)
        if anchor is None:
            raise DistributedError(
                f"unknown anchor {anchor_id!r} on node {self.node}",
                code="E006",
                note="MissingAnchorError",
                help_text=(
                    "the actor handle refers to a channel anchored on another "
                    "node; route the request to the owning node"
                ),
                context={"anchor": anchor_id, "node": self.node},
            )
        return anchor

    def __contains__(self, anchor_id: str) -> bool:
        with self._lock:
            return anchor_id in self._anchors

    def anchor_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._anchors)

    def close(self, anchor_id: str) -> None:
        self.get(anchor_id).close()

    def drop(self, anchor_id: str) -> None:
        with self._lock:
            anchor = self._anchors.pop(anchor_id, None)
        if anchor is not None:
            anchor.close()

    # -- RPC surface ------------------------------------------------------
    def handle_request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if op == "open":
                anchor = self.open(
                    payload.get("anchor_id"),
                    owner=payload.get("owner", ""),
                    epoch=int(payload.get("epoch", 0)),
                    fencing=int(payload.get("fencing", 1)),
                )
                return {"ok": True, "handle": anchor.handle()}
            if op == "handle":
                return {
                    "ok": True,
                    "handle": self.get(payload["anchor_id"]).handle(
                        owner=payload.get("owner")
                    ),
                }
            if op == "bind":
                anchor = self.get(payload["anchor_id"])
                anchor.bind(
                    payload["owner"],
                    epoch=int(payload["epoch"]),
                    fencing=int(payload["fencing"]),
                )
                return {"ok": True, "handle": anchor.handle()}
            if op == "deliver":
                seq = self.get(payload["anchor_id"]).deliver(payload["data"])
                return {"ok": True, "seq": seq}
            if op == "recv":
                messages, cursor = self.get(payload["anchor_id"]).recv(
                    payload.get("cursor"),
                    limit=int(payload.get("limit", 64)),
                    owner=payload.get("owner"),
                    epoch=payload.get("epoch"),
                    fencing=payload.get("fencing"),
                )
                return {
                    "ok": True,
                    "messages": [(item.seq, item.data) for item in messages],
                    "cursor": cursor,
                }
            if op == "ack":
                self.get(payload["anchor_id"]).ack(int(payload["cursor"]))
                return {"ok": True}
            if op == "send":
                applied = self.get(payload["anchor_id"]).send(
                    int(payload["seq"]),
                    payload["data"],
                    owner=payload.get("owner"),
                    epoch=payload.get("epoch"),
                    fencing=payload.get("fencing"),
                )
                return {"ok": True, "applied": applied}
            if op == "pending":
                pending = self.get(payload["anchor_id"]).pending_outbound(
                    limit=int(payload.get("limit", 64))
                )
                return {
                    "ok": True,
                    "messages": [(item.seq, item.data) for item in pending],
                }
            if op == "complete":
                self.get(payload["anchor_id"]).complete(int(payload["seq"]))
                return {"ok": True}
            if op == "close":
                self.close(payload["anchor_id"])
                return {"ok": True}
            if op == "snapshot":
                return {
                    "ok": True,
                    "snapshot": self.get(payload["anchor_id"]).snapshot(),
                }
            if op == "restore":
                anchor = self.open(
                    payload["snapshot"]["anchor_id"],
                    owner=payload["snapshot"].get("owner", ""),
                    epoch=int(payload["snapshot"].get("epoch", 0)),
                    fencing=int(payload["snapshot"].get("fencing", 0)),
                )
                anchor.restore(payload["snapshot"])
                return {"ok": True, "handle": anchor.handle()}
            raise protocol_error(
                f"unknown anchor operation {op!r}",
                context={"node": self.node, "op": op},
            )
        except DistributedError as error:
            return {"ok": False, "error": error}


def coerce_error(value: Any) -> DistributedError | None:
    """Rebuild a structured error from its wire dictionary, if possible."""
    if isinstance(value, DistributedError):
        return value
    if not isinstance(value, dict):
        return None
    return DistributedError(
        str(value.get("message", "remote anchor operation failed")),
        code=str(value.get("code", "E008")),
        reason=value.get("reason"),
        help_text=value.get("help"),
        note=value.get("note"),
        context=value.get("context") or {},
    )


class AnchorChannel:
    """An actor-facing channel that is just a handle plus an RPC client."""

    def __init__(
        self,
        handle: ChannelHandle,
        call: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.handle = handle
        self._call = call
        self._inbound = handle.inbound
        self._outbound = handle.outbound

    # -- client plumbing --------------------------------------------------
    def _request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload.setdefault("anchor_id", self.handle.anchor_id)
        payload.setdefault("owner", self.handle.owner)
        payload.setdefault("epoch", self.handle.epoch)
        payload.setdefault("fencing", self.handle.fencing)
        payload.setdefault("anchor_node", self.handle.anchor_node)
        reply = self._call(op, payload)
        if not reply.get("ok", False):
            error = coerce_error(reply.get("error"))
            if isinstance(error, DistributedError):
                raise error
            raise protocol_error(
                f"anchor {self.handle.anchor_id!r} rejected {op}: {error}",
                context={"anchor": self.handle.anchor_id, "op": op},
            )
        return reply

    # -- data path --------------------------------------------------------
    def send(self, data: bytes) -> int:
        seq = self._outbound
        self._request("send", {"seq": seq, "data": bytes(data)})
        self._outbound += 1
        self.handle.outbound = self._outbound
        return seq

    def send_many(self, messages: list[bytes]) -> int:
        first = self._outbound
        for message in messages:
            self.send(message)
        return first

    def recv(self, *, limit: int = 64) -> list[bytes]:
        reply = self._request("recv", {"cursor": self._inbound, "limit": limit})
        messages = [bytes(data) for _seq, data in reply["messages"]]
        self._inbound = int(reply["cursor"])
        self.handle.inbound = self._inbound
        return messages

    def ack(self) -> None:
        self._request("ack", {"cursor": self._inbound})

    def pending_outbound(self, *, limit: int = 64) -> list[tuple[int, bytes]]:
        reply = self._request("pending", {"limit": limit})
        return [(int(seq), bytes(data)) for seq, data in reply["messages"]]

    def complete(self, seq: int) -> None:
        self._request("complete", {"seq": seq})
        self.handle.outbound = max(self.handle.outbound, seq)

    def rebind(self, owner: str, *, epoch: int, fencing: int) -> ChannelHandle:
        reply = self._request(
            "bind", {"owner": owner, "epoch": epoch, "fencing": fencing}
        )
        new_handle = reply["handle"]
        for field_name in (
            "owner",
            "epoch",
            "fencing",
            "inbound",
            "outbound",
            "anchor_node",
            "closed",
        ):
            setattr(self.handle, field_name, getattr(new_handle, field_name))
        self._inbound = self.handle.inbound
        self._outbound = self.handle.outbound
        return self.handle

    def close(self) -> None:
        self._request("close", {})


@dataclass
class SocketStream:
    """A length-prefixed TCP socket bound to an anchor.

    The reader thread calls ``anchor.deliver`` for every inbound frame.  Writes
    go through ``AnchorChannel`` so the actor gets the same idempotent,
    cursor-based semantics as an in-memory test channel.
    """

    sock: socket.socket
    anchor: ConnectionAnchor
    registry: AnchorRegistry
    max_message: int = MAX_MESSAGE_BYTES
    _reader: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _error: BaseException | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Bound the blocking calls so close() cannot hang forever.
        self.sock.settimeout(0.2)

    @classmethod
    def connect(
        cls,
        address: tuple[str, int],
        registry: AnchorRegistry,
        *,
        owner: str,
        epoch: int = 0,
        fencing: int = 1,
        timeout: float = 5.0,
    ) -> SocketStream:
        sock = socket.create_connection(address, timeout=timeout)
        anchor = registry.open(owner=owner, epoch=epoch, fencing=fencing)
        return cls(sock=sock, anchor=anchor, registry=registry)

    @classmethod
    def accept(
        cls,
        sock: socket.socket,
        registry: AnchorRegistry,
        *,
        owner: str,
        epoch: int = 0,
        fencing: int = 1,
    ) -> SocketStream:
        anchor = registry.open(owner=owner, epoch=epoch, fencing=fencing)
        return cls(sock=sock, anchor=anchor, registry=registry)

    def start(self) -> SocketStream:
        if self._reader is not None:
            return self
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"anchor-{self.anchor.anchor_id}",
            daemon=True,
        )
        self._reader.start()
        return self

    def _read_loop(self) -> None:
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self.sock.recv(65536)
            except TimeoutError:
                continue
            except OSError as error:
                self._error = error
                break
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= LENGTH.size:
                (length,) = LENGTH.unpack(buffer[: LENGTH.size])
                if length > self.max_message:
                    self._error = wire_error(
                        f"frame is too large: {length} bytes",
                        context={"anchor": self.anchor.anchor_id},
                    )
                    self._stop.set()
                    return
                end = LENGTH.size + length
                if len(buffer) < end:
                    break
                payload = bytes(buffer[LENGTH.size : end])
                del buffer[:end]
                try:
                    self.anchor.deliver(payload)
                except DistributedError as error:
                    self._error = error
                    self._stop.set()
                    return
        self._stop.set()

    def flush(self, *, limit: int = 64) -> int:
        pending = self.anchor.pending_outbound(limit=limit)
        written = 0
        with self._write_lock:
            for item in pending:
                frame = LENGTH.pack(len(item.data)) + item.data
                self.sock.sendall(frame)
                self.anchor.complete(item.seq + 1)
                written += 1
        return written

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        self.anchor.close()

    @property
    def error(self) -> BaseException | None:
        return self._error


__all__ = [
    "MAX_BUFFERED",
    "MAX_MESSAGE_BYTES",
    "AnchorChannel",
    "AnchorRegistry",
    "ChannelHandle",
    "ConnectionAnchor",
    "Inbound",
    "Outbound",
    "SocketStream",
    "coerce_error",
]
