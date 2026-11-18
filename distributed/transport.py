"""Cluster transports.

Two implementations share one envelope shape:

* :class:`InMemoryNetwork` -- deterministic, partitionable, and fast.  Tests use
  it to drive real message paths through the control plane.
* :class:`TCPTransport` / :func:`serve` -- length-prefixed frames over a real
  socket, used by the TCP smoke test.

Envelopes are themselves wire values, so the payload can be a state frame, a
checkpoint, a lease request, or plain application bytes.  The transport only
moves bytes and correlates replies; it never inspects actor state.
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Self

from distributed.diagnostics import (
    DistributedError,
    node_unavailable,
    protocol_error,
    wire_error,
)
from distributed.transmute import decode_state, encode_state, register_type
from distributed.wire import KIND_RPC, WireError

KIND_REPLY: Final = "reply"
KIND_PING: Final = "ping"
KIND_PONG: Final = "pong"

MAX_ENVELOPE_BYTES: Final = 64 * 1024 * 1024
LENGTH: Final = struct.Struct(">I")


@dataclass
class Envelope:
    src: str
    dst: str
    kind: str = KIND_RPC
    payload: bytes = b""
    msg_id: str = ""
    correlation_id: str = ""
    sent_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.src or not self.dst:
            raise DistributedError(
                "an envelope needs both a source and a destination",
                code="E003",
                note="ProtocolError",
            )
        if not self.msg_id:
            self.msg_id = f"{self.src}:{time.time_ns()}:{id(self):x}"
        if isinstance(self.payload, (bytearray, memoryview)):
            self.payload = bytes(self.payload)
        elif not isinstance(self.payload, bytes):
            raise DistributedError(
                f"envelope payload must be bytes, not {type(self.payload).__name__}",
                code="E003",
                note="ProtocolError",
            )

    def reply(self, payload: bytes = b"", *, kind: str = KIND_REPLY) -> Envelope:
        return Envelope(
            src=self.dst,
            dst=self.src,
            kind=kind,
            payload=payload,
            correlation_id=self.msg_id,
        )


register_type(Envelope, name="Envelope")


def encode_envelope(envelope: Envelope) -> bytes:
    return encode_state(envelope, kind=KIND_RPC)


def decode_envelope(data: bytes) -> Envelope:
    try:
        value = decode_state(data)
    except WireError as error:
        raise wire_error(str(error), reason=getattr(error, "reason", "")) from error
    if not isinstance(value, Envelope):
        raise protocol_error(
            "frame does not contain an Envelope",
            context={"type": type(value).__name__},
        )
    return value


# ---------------------------------------------------------------------------
# In-memory network
# ---------------------------------------------------------------------------


@dataclass
class _NodeBox:
    inbox: queue.Queue[Envelope] = field(default_factory=queue.Queue)
    alive: bool = True
    partitions: set[str] = field(default_factory=set)


class InMemoryNetwork:
    """A deterministic network with explicit partitions and node kills."""

    def __init__(self, *, strict_partitions: bool = False) -> None:
        self.nodes: dict[str, _NodeBox] = {}
        self.strict_partitions = strict_partitions
        self.sent = 0
        self.dropped = 0
        self.partitioned = 0
        self.log: list[Envelope] = []
        self._lock = threading.RLock()

    def register(self, node: str, *, alive: bool = True) -> None:
        with self._lock:
            if node in self.nodes:
                raise protocol_error(f"node {node!r} is already registered")
            self.nodes[node] = _NodeBox(alive=alive)

    def unregister(self, node: str) -> None:
        with self._lock:
            self.nodes.pop(node, None)

    def transport(self, node: str) -> InMemoryTransport:
        with self._lock:
            if node not in self.nodes:
                raise node_unavailable(
                    f"node {node!r} is not registered",
                    context={"node": node},
                )
        return InMemoryTransport(self, node)

    def is_alive(self, node: str) -> bool:
        with self._lock:
            box = self.nodes.get(node)
            return bool(box and box.alive)

    def nodes_alive(self) -> list[str]:
        with self._lock:
            return sorted(node for node, box in self.nodes.items() if box.alive)

    def kill(self, node: str) -> None:
        with self._lock:
            box = self.nodes.get(node)
            if box is None:
                return
            box.alive = False
            while True:
                try:
                    box.inbox.get_nowait()
                except queue.Empty:
                    break

    def revive(self, node: str) -> None:
        with self._lock:
            box = self.nodes.get(node)
            if box is not None:
                box.alive = True

    def partition(self, left: str, right: str) -> None:
        with self._lock:
            if left not in self.nodes or right not in self.nodes:
                raise node_unavailable(
                    f"cannot partition unknown node(s): {left!r}, {right!r}"
                )
            self.nodes[left].partitions.add(right)
            self.nodes[right].partitions.add(left)

    def heal(self, left: str | None = None, right: str | None = None) -> None:
        with self._lock:
            if left is None and right is None:
                for box in self.nodes.values():
                    box.partitions.clear()
                return
            if left is None or right is None:
                raise protocol_error("heal needs both node names or neither")
            self.nodes[left].partitions.discard(right)
            self.nodes[right].partitions.discard(left)

    def reachable(self, src: str, dst: str) -> bool:
        with self._lock:
            box = self.nodes.get(src)
            if box is None or not box.alive:
                return False
            other = self.nodes.get(dst)
            if other is None or not other.alive:
                return False
            return dst not in box.partitions

    def send(self, envelope: Envelope) -> bool:
        with self._lock:
            self.sent += 1
            source = self.nodes.get(envelope.src)
            target = self.nodes.get(envelope.dst)
            if source is not None and not source.alive:
                raise node_unavailable(
                    f"node {envelope.src!r} is down",
                    context={"node": envelope.src},
                )
            if target is None or not target.alive:
                raise node_unavailable(
                    f"node {envelope.dst!r} is down or unknown",
                    context={"node": envelope.dst},
                )
            if envelope.dst in source.partitions:
                self.partitioned += 1
                if self.strict_partitions:
                    raise node_unavailable(
                        f"link {envelope.src!r} -> {envelope.dst!r} is partitioned",
                        note="NetworkPartitionError",
                        context={
                            "src": envelope.src,
                            "dst": envelope.dst,
                        },
                    )
                self.dropped += 1
                return False
            self.log.append(envelope)
            target.inbox.put(envelope)
            return True

    def recv(self, node: str, *, timeout: float | None = None) -> Envelope:
        with self._lock:
            box = self.nodes.get(node)
            if box is None:
                raise node_unavailable(f"node {node!r} is not registered")
            if not box.alive:
                raise node_unavailable(f"node {node!r} is down")
        try:
            return box.inbox.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(f"no message for {node!r}") from error

    def drain(self, node: str) -> list[Envelope]:
        with self._lock:
            box = self.nodes.get(node)
            if box is None:
                return []
        messages: list[Envelope] = []
        while True:
            try:
                messages.append(box.inbox.get_nowait())
            except queue.Empty:
                return messages

    def pending(self, node: str) -> int:
        with self._lock:
            box = self.nodes.get(node)
            return 0 if box is None else box.inbox.qsize()


class InMemoryTransport:
    """A node's endpoint on an :class:`InMemoryNetwork`."""

    def __init__(self, network: InMemoryNetwork, node: str) -> None:
        self.network = network
        self.node = node
        self._backlog: list[Envelope] = []
        self.closed = False

    def send(self, envelope: Envelope) -> bool:
        if self.closed:
            raise node_unavailable(f"transport for {self.node!r} is closed")
        if envelope.src != self.node:
            envelope = Envelope(
                src=self.node,
                dst=envelope.dst,
                kind=envelope.kind,
                payload=envelope.payload,
                msg_id=envelope.msg_id,
                correlation_id=envelope.correlation_id,
                sent_at=envelope.sent_at,
            )
        return self.network.send(envelope)

    def recv(self, *, timeout: float | None = None) -> Envelope:
        if self.closed:
            raise node_unavailable(f"transport for {self.node!r} is closed")
        if self._backlog:
            return self._backlog.pop(0)
        return self.network.recv(self.node, timeout=timeout)

    def drain(self) -> list[Envelope]:
        messages, self._backlog = self._backlog, []
        messages.extend(self.network.drain(self.node))
        return messages

    def restash(self, messages: list[Envelope]) -> None:
        """Put unmatched messages back at the front of this endpoint."""
        if messages:
            self._backlog[0:0] = messages

    def call(
        self,
        envelope: Envelope,
        *,
        timeout: float = 5.0,
    ) -> Envelope:
        self.send(envelope)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                message = self.recv(timeout=remaining)
            except TimeoutError:
                break
            if message.correlation_id == envelope.msg_id and message.kind == KIND_REPLY:
                return message
            self._backlog.append(message)
        raise node_unavailable(
            f"no reply from {envelope.dst!r} for {envelope.msg_id!r}",
            note="ProtocolError",
            context={
                "src": self.node,
                "dst": envelope.dst,
                "msg_id": envelope.msg_id,
            },
        )

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# TCP transport
# ---------------------------------------------------------------------------


def _read_exact(
    sock: socket.socket,
    count: int,
    *,
    deadline: float | None = None,
) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < count:
        if deadline is not None and time.monotonic() >= deadline:
            return None
        try:
            chunk = sock.recv(count - len(chunks))
        except TimeoutError:
            continue
        if not chunk:
            return None
        chunks += chunk
    return bytes(chunks)


class TCPTransport:
    """A blocking TCP envelope transport (one peer per instance)."""

    def __init__(
        self,
        sock: socket.socket,
        node: str,
        *,
        timeout: float = 5.0,
    ) -> None:
        self.sock = sock
        self.node = node
        self.timeout = timeout
        self.closed = False
        self.sock.settimeout(timeout)

    @classmethod
    def connect(
        cls,
        address: tuple[str, int],
        node: str,
        *,
        timeout: float = 5.0,
    ) -> TCPTransport:
        sock = socket.create_connection(address, timeout=timeout)
        return cls(sock, node, timeout=timeout)

    def send(self, envelope: Envelope) -> None:
        if self.closed:
            raise node_unavailable(f"TCP transport for {self.node!r} is closed")
        data = encode_envelope(envelope)
        if len(data) > MAX_ENVELOPE_BYTES:
            raise wire_error(
                f"envelope is too large: {len(data)} bytes (limit {MAX_ENVELOPE_BYTES})"
            )
        self.sock.sendall(LENGTH.pack(len(data)) + data)

    def recv(self) -> Envelope:
        deadline = time.monotonic() + self.timeout
        header = _read_exact(self.sock, LENGTH.size, deadline=deadline)
        if header is None:
            raise node_unavailable(
                f"peer closed the connection to {self.node!r}",
                note="NodeLostError",
            )
        (length,) = LENGTH.unpack(header)
        if length > MAX_ENVELOPE_BYTES:
            raise wire_error(
                f"envelope is too large: {length} bytes (limit {MAX_ENVELOPE_BYTES})"
            )
        data = _read_exact(self.sock, length, deadline=deadline)
        if data is None:
            raise node_unavailable(
                f"peer closed mid-frame for {self.node!r}",
                note="NodeLostError",
            )
        return decode_envelope(data)

    def call(
        self,
        envelope: Envelope,
        *,
        timeout: float = 5.0,
    ) -> Envelope:
        self.send(envelope)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self.recv()
            if message.correlation_id == envelope.msg_id and message.kind == KIND_REPLY:
                return message
        raise node_unavailable(
            f"no reply from {envelope.dst!r} for {envelope.msg_id!r}",
            note="ProtocolError",
        )

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class TCPNodeServer:
    """Accept loop that decodes envelopes and calls a handler per message."""

    def __init__(
        self,
        handler: Callable[[Envelope], Envelope | None],
        *,
        node: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.handler = handler
        self.node = node
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port

    def start(self) -> TCPNodeServer:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(8)
        listener.settimeout(0.2)
        self.host, self.port = listener.getsockname()
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"tcp-server-{self.node}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, _address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            client.settimeout(0.2)
            with self._lock:
                self._clients.append(client)
            threading.Thread(
                target=self._serve_client,
                args=(client,),
                name=f"tcp-client-{self.node}",
                daemon=True,
            ).start()

    def _serve_client(self, client: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                header = _read_exact(client, LENGTH.size)
                if header is None:
                    return
                (length,) = LENGTH.unpack(header)
                if length > MAX_ENVELOPE_BYTES:
                    return
                data = _read_exact(client, length)
                if data is None:
                    return
                envelope = decode_envelope(data)
                reply = self.handler(envelope)
                if reply is not None:
                    payload = encode_envelope(reply)
                    client.sendall(LENGTH.pack(len(payload)) + payload)
        except OSError, DistributedError:
            return
        finally:
            try:
                client.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def serve(
    handler: Callable[[Envelope], Envelope | None],
    *,
    node: str,
    host: str = "127.0.0.1",
    port: int = 0,
) -> TCPNodeServer:
    return TCPNodeServer(handler, node=node, host=host, port=port).start()


__all__ = [
    "KIND_PING",
    "KIND_PONG",
    "KIND_REPLY",
    "MAX_ENVELOPE_BYTES",
    "Envelope",
    "InMemoryNetwork",
    "InMemoryTransport",
    "TCPNodeServer",
    "TCPTransport",
    "decode_envelope",
    "encode_envelope",
    "serve",
]
