"""Quantum hot-patching subsystem: trampolines, frames, reactors, RCU, journal."""

from __future__ import annotations

import ctypes
import gc
import mmap
import os
import platform
import selectors
import socket
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from lexer import lex
from parser import parse
from runtime.hotpatch import (
    CAPABILITIES,
    AstEpochManager,
    ExecutableRegion,
    FrameEvalMigrator,
    HotpatchError,
    LiveTrampoline,
    RcuDomain,
    RollbackJournal,
    byte_offsets,
    find_got_slots,
    hotpatch_cpython_frame_eval,
    hotpatch_live_dylib_trampoline,
    hotpatch_mio_reactor_state,
    hotpatch_rcu_ast_epoch,
    hotpatch_rollback_journal_commit,
    migrate_mio_bridge,
    migrate_selector,
    platform_support,
    resolve_symbol,
    span_from_line_col,
)
from runtime.hotpatch.trampoline import (
    _page_protection,
    atomic_load_ptr,
    atomic_store_ptr,
)

NATIVE = Path(__file__).resolve().parents[1] / "runtime" / "hotpatch" / "native"
ARM64 = platform.machine().lower() in {"arm64", "aarch64"}


def _program(source: str):
    return parse(lex(source))


def _machine_code(value: int) -> bytes:
    if ARM64:
        return struct.pack("<II", 0x52800000 | (value << 5), 0xD65F03C0)
    return b"\xb8" + struct.pack("<I", value) + b"\xc3"


def _callable_address(function) -> int:
    return ctypes.cast(function, ctypes.c_void_p).value or 0


def _slot_page() -> tuple[mmap.mmap, int]:
    page = mmap.mmap(-1, mmap.PAGESIZE)
    slot = ctypes.addressof((ctypes.c_void_p * 1).from_buffer(page))
    return page, slot


def _call_slot(slot: int) -> int:
    target = ctypes.c_void_p.from_address(slot).value
    return ctypes.CFUNCTYPE(ctypes.c_int64)(target)()


def _nullary(address: int) -> int:
    return ctypes.CFUNCTYPE(ctypes.c_int64)(address)()


def _close_quietly(selector) -> None:
    try:
        selector.close()
    except OSError, ValueError:
        pass


# ---------------------------------------------------------------------------
# RCU
# ---------------------------------------------------------------------------


def test_rcu_ast_epoch_reclaims_after_readers_exit():
    first = _program("fn main() -> i64 { 1 }")
    manager = hotpatch_rcu_ast_epoch(first)
    assert isinstance(manager, AstEpochManager)
    reclaimed: list[int] = []
    manager._on_reclaim = lambda _program, version: reclaimed.append(version)

    with manager.read() as root:
        assert root is first
        assert manager.publish(_program("fn main() -> i64 { 2 }")) == 2
    assert manager.synchronize(timeout=2.0) == 1
    assert reclaimed == [1]
    assert manager.stats().reclaimed == 1
    assert manager.stats().epoch >= 1


def test_rcu_timeout_reports_e008_with_span():
    manager = hotpatch_rcu_ast_epoch(_program("fn main() -> i64 { 0 }"))
    manager.domain.enter()
    try:
        with pytest.raises(HotpatchError) as info:
            manager.synchronize(timeout=0.01)
        assert info.value.code == "E008"
        assert "grace period timed out" in str(info.value)
    finally:
        manager.domain.exit()
    assert manager.synchronize(timeout=2.0) >= 0


def test_rcu_concurrent_readers_and_writer_drain():
    manager = hotpatch_rcu_ast_epoch(_program("fn main() -> i64 { 0 }"))
    stop = threading.Event()
    failures: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                with manager.read() as root:
                    assert root is not None
        except BaseException as error:  # pragma: no cover - failure path
            failures.append(error)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        for index in range(20):
            manager.publish(_program(f"fn main() -> i64 {{ {index} }}"))
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
    assert not failures
    assert manager.drain(timeout=5.0) >= 20
    assert manager.stats().reclaimed >= 20


def test_rcu_domain_is_independent_of_ast_shape():
    domain = RcuDomain("generic")
    with domain.read() as epoch:
        assert epoch == 0
    assert domain.stats().grace_periods == 0
    domain.retire(object())
    assert domain.synchronize(timeout=1.0) == 1


# ---------------------------------------------------------------------------
# Rollback journal
# ---------------------------------------------------------------------------


class _Mutable:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_journal_rolls_back_objects_buffers_and_program():
    journal = hotpatch_rollback_journal_commit()
    assert isinstance(journal, RollbackJournal)
    obj = _Mutable(1)
    buffer = bytearray(b"hello")
    program = _program("fn main() -> i64 { 1 }")
    snapshot = journal.checkpoint(
        objects=[obj], buffers=[buffer], program=program, label="unit"
    )
    obj.value = 9
    buffer[0:2] = b"HE"
    removed = program.items.pop()
    assert journal.verify(snapshot) is False
    assert journal.rollback(snapshot) == snapshot.generation
    assert obj.value == 1
    assert bytes(buffer) == b"hello"
    assert removed in program.items
    assert journal.last_rollback_ns > 0
    assert journal.rollbacks == 1


def test_journal_transaction_rolls_back_failed_verification():
    journal = RollbackJournal()
    obj = _Mutable(1)
    with (
        pytest.raises(HotpatchError) as info,
        journal.transaction(objects=[obj], verify=lambda: obj.value == 1),
    ):
        obj.value = 2
        assert not journal.verify()
    assert info.value.code == "E008"
    assert obj.value == 1
    assert journal.pending() == ()


def test_journal_rejects_values_without_restorable_state():
    journal = RollbackJournal()
    with pytest.raises(HotpatchError) as info:
        journal.checkpoint(objects=[object()])
    assert info.value.code == "E008"
    assert "no __dict__ or slots" in str(info.value)


def test_journal_deep_snapshot_isolates_nested_containers():
    journal = RollbackJournal()
    obj = _Mutable({"nested": [1]})
    journal.checkpoint(objects=[obj], deep=True)
    obj.value["nested"].append(2)
    assert obj.value == {"nested": [1, 2]}
    journal.rollback()
    assert obj.value == {"nested": [1]}


def test_byte_offsets_use_utf8_encoding():
    source = "éabcdefghij"
    span = byte_offsets(source, (1, 5))
    assert span is not None
    assert span.as_tuple() == (1, 5)
    assert (span.byte_start, span.byte_end) == (2, 6)


def test_span_helpers_clamp_invalid_ranges():
    source = "éabcdefghij"
    span = byte_offsets(source, (99, -5))
    assert span is not None
    assert span.as_tuple() == (99, -5)
    encoded_length = len(source.encode("utf-8"))
    assert (span.byte_start, span.byte_end) == (encoded_length, encoded_length)

    without_source = byte_offsets(None, (7, 3))
    assert without_source is not None
    assert without_source.as_tuple() == (7, 3)
    assert (without_source.byte_start, without_source.byte_end) == (3, 7)
    assert span_from_line_col("abc\ndé", 1, -2) == (0, 1)
    assert span_from_line_col("abc\ndé", 2, 99, width=8) == (6, 6)
    assert span_from_line_col("abc", 5, 1) is None


# ---------------------------------------------------------------------------
# CPython frame migration
# ---------------------------------------------------------------------------


def test_frame_patch_migrates_live_fastlocals():
    captured: dict[str, object] = {}
    gate = threading.Event()

    def old(x: int) -> int:
        y = x + 1
        captured["frame"] = sys._getframe()
        gate.wait(timeout=5.0)
        return y

    def new(x: int) -> int:
        _ = captured, gate
        y = x + 2
        extra = 7
        return y + extra

    thread = threading.Thread(target=old, args=(1,))
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "frame" not in captured and time.monotonic() < deadline:
            time.sleep(0.001)
        assert "frame" in captured
        frame = captured["frame"]
        assert isinstance(frame, type(sys._getframe()))
        live = dict(frame.f_locals)
        assert live["x"] == 1 and live["y"] == 2
        patch = hotpatch_cpython_frame_eval(old, new, defaults={"extra": 5})
        try:
            assert patch.pinned_frames == 1
            assert patch.migrated_locals == 1
            assert dict(frame.f_locals)["extra"] == 5
            assert old(1) == 10  # new code: (1 + 2) + 7
        finally:
            gate.set()
            thread.join(timeout=5.0)
        assert not thread.is_alive()
        # The captured frame finished on its original code object.
        assert patch.rollback() == 1
        assert old(1) == 2
    finally:
        gate.set()
        thread.join(timeout=5.0)


def test_frame_patch_migrates_suspended_generator():
    def source():
        x = 5
        yield sys._getframe()
        yield x

    def replacement():
        x = 5
        extra = 0
        yield extra
        yield x

    generator = source()
    frame = next(generator)
    try:
        migrator = FrameEvalMigrator()
        patch = migrator.patch(
            source, replacement, generators=[generator], defaults={"extra": 9}
        )
        assert patch.generators == [generator]
        assert dict(frame.f_locals)["extra"] == 9
        assert next(generator) == 5
        patch.rollback()
        assert source.__code__ is patch.old_code
    finally:
        generator.close()


def test_frame_patch_reports_e008_with_character_and_byte_spans():
    def one(x: int) -> int:
        return x

    def two(x: int, y: int) -> int:
        return x

    source = "éabcdefghij"
    with pytest.raises(HotpatchError) as info:
        hotpatch_cpython_frame_eval(one, two, span=(1, 5), source=source)
    error = info.value
    assert error.code == "E008"
    assert error.span == (1, 5)
    assert error.bytes == (2, 6)
    assert "bytes: 2..6" in str(error)


def test_frame_patch_rejects_a_removed_live_local():
    gate = threading.Event()
    captured: dict[str, object] = {}

    def old(x: int) -> int:
        keep = x + 1
        removed = 10
        captured["frame"] = sys._getframe()
        gate.wait(timeout=5.0)
        return keep + removed

    def new(x: int) -> int:
        keep = x + 1
        return keep

    thread = threading.Thread(target=old, args=(1,))
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "frame" not in captured and time.monotonic() < deadline:
            time.sleep(0.001)
        with pytest.raises(HotpatchError) as info:
            hotpatch_cpython_frame_eval(old, new)
        assert info.value.code == "E008"
        assert "removed" in str(info.value)
    finally:
        gate.set()
        thread.join(timeout=5.0)


def test_frame_patch_rolls_back_all_frames_when_a_later_generator_fails():
    gate = threading.Event()
    captured: dict[str, object] = {}

    def old(x: int) -> int:
        y = x + 1
        captured["frame"] = sys._getframe()
        gate.wait(timeout=5.0)
        return y

    def new(x: int) -> int:
        _ = captured, gate
        y = x + 2
        extra = 7
        return y + extra

    def unrelated():
        yield sys._getframe()

    bad_generator = unrelated()
    bad_frame = next(bad_generator)
    original_code = old.__code__
    thread = threading.Thread(target=old, args=(1,))
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "frame" not in captured and time.monotonic() < deadline:
            time.sleep(0.001)
        frame = captured["frame"]
        before = dict(frame.f_locals)
        assert before["y"] == 2
        assert "extra" not in before

        with pytest.raises(HotpatchError) as info:
            hotpatch_cpython_frame_eval(
                old, new, generators=[bad_generator], defaults={"extra": 5}
            )
        assert info.value.code == "E008"
        assert "generator frame does not match" in str(info.value)
        assert old.__code__ is original_code
        assert bad_frame.f_code is not original_code
        assert dict(frame.f_locals) == before
        assert "extra" not in dict(frame.f_locals)
    finally:
        gate.set()
        thread.join(timeout=5.0)
        bad_generator.close()


def test_frame_patch_reports_e007_on_interpreters_without_writable_locals(monkeypatch):
    import runtime.hotpatch.frames as frames_module

    class _SysProxy:
        _current_frames = staticmethod(sys._current_frames)
        version_info = (3, 12, 0)

    monkeypatch.setattr(frames_module, "sys", _SysProxy())

    def source():
        y = 1
        yield sys._getframe()
        yield y

    def replacement():
        y = 1
        extra = 0
        yield extra
        yield y

    generator = source()
    frame = next(generator)
    original_code = source.__code__
    try:
        with pytest.raises(HotpatchError) as info:
            FrameEvalMigrator().patch(
                source,
                replacement,
                generators=[generator],
                defaults={"extra": 9},
                span=(2, 3),
            )
        assert info.value.code == "E007"
        assert info.value.span == (2, 3)
        assert "writable frame locals" in str(info.value)
        assert source.__code__ is original_code
        assert "extra" not in dict(frame.f_locals)
    finally:
        generator.close()


def test_frame_patch_context_manager_rolls_back_on_exception():
    def source():
        x = 1
        yield sys._getframe()
        yield x

    def replacement():
        x = 1
        extra = 2
        yield extra + x
        yield x

    generator = source()
    frame = next(generator)
    original_code = source.__code__
    try:
        patch = FrameEvalMigrator().patch(
            source, replacement, generators=[generator], defaults={"extra": 9}
        )
        assert dict(frame.f_locals)["extra"] == 9
        with pytest.raises(RuntimeError, match="abort"), patch:
            raise RuntimeError("abort")
        assert patch.rolled_back
        assert source.__code__ is original_code
        assert "extra" not in dict(frame.f_locals)
    finally:
        generator.close()


def test_frame_patch_pins_the_running_frame_to_its_captured_code():
    gate = threading.Event()
    captured: dict[str, object] = {}

    def old(x: int) -> int:
        captured["frame"] = sys._getframe()
        gate.wait(timeout=5.0)
        return x + 1

    def new(x: int) -> int:
        assert captured is not None and gate is not None
        return x + 100

    thread = threading.Thread(target=old, args=(1,))
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "frame" not in captured and time.monotonic() < deadline:
            time.sleep(0.001)
        patch = hotpatch_cpython_frame_eval(old, new)
        try:
            gate.set()
            thread.join(timeout=5.0)
            # The frame finished on the old bytecode, not the new body.
            assert not thread.is_alive()
            assert old(1) == 101
        finally:
            patch.rollback()
    finally:
        gate.set()
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Safepoint dispatcher lifecycle
# ---------------------------------------------------------------------------


class _FakeMonitoring:
    def __init__(self, *, fail_register: bool = False) -> None:
        self.events = types.SimpleNamespace(LINE=1)
        self.fail_register = fail_register
        self.used: list[int] = []
        self.freed: list[int] = []
        self.registered: list[tuple[int, int]] = []
        self.events_set: list[tuple[int, int]] = []

    def use_tool_id(self, tool_id: int, name: str) -> None:
        self.used.append(tool_id)

    def register_callback(self, tool_id: int, event: int, _callback) -> None:
        if self.fail_register:
            raise RuntimeError("callback registration refused")
        self.registered.append((tool_id, event))

    def set_events(self, tool_id: int, events: int) -> None:
        self.events_set.append((tool_id, events))

    def free_tool_id(self, tool_id: int) -> None:
        self.freed.append(tool_id)


def _patch_fake_monitoring(monkeypatch, monitoring: _FakeMonitoring) -> None:
    import runtime.hotpatch.frames as frames_module

    monkeypatch.setattr(
        frames_module, "sys", types.SimpleNamespace(monitoring=monitoring)
    )


def test_safepoint_install_failure_releases_tool_id(monkeypatch):
    from runtime.hotpatch.frames import SafepointDispatcher

    monitoring = _FakeMonitoring(fail_register=True)
    _patch_fake_monitoring(monkeypatch, monitoring)
    dispatcher = SafepointDispatcher()

    with pytest.raises(HotpatchError) as info:
        dispatcher.install()
    assert info.value.code == "E007"
    assert "cannot install" in str(info.value)
    assert monitoring.used == [3]
    assert monitoring.freed == monitoring.used
    assert dispatcher._tool_id is None


def test_safepoint_dispatcher_releases_tool_id_on_drop(monkeypatch):
    from runtime.hotpatch.frames import SafepointDispatcher

    monitoring = _FakeMonitoring()
    _patch_fake_monitoring(monkeypatch, monitoring)
    dispatcher = SafepointDispatcher()
    dispatcher.install()
    assert monitoring.used == [3]
    assert dispatcher._tool_id == 3

    del dispatcher
    gc.collect()
    assert monitoring.freed == [3]


def test_safepoint_dispatcher_retains_callback_exceptions(monkeypatch):
    from runtime.hotpatch.frames import SafepointDispatcher

    monitoring = _FakeMonitoring()
    _patch_fake_monitoring(monkeypatch, monitoring)
    dispatcher = SafepointDispatcher()
    dispatcher.install()
    called: list[int] = []

    def failing_action() -> None:
        called.append(1)
        raise RuntimeError("safepoint action failed")

    dispatcher.defer(sys._getframe().f_code, failing_action)
    dispatcher._on_line(sys._getframe().f_code, 1)
    assert called == [1]
    assert dispatcher.dispatches == 1
    assert len(dispatcher.errors) == 1
    assert isinstance(dispatcher.errors[0], RuntimeError)
    dispatcher.uninstall()


# ---------------------------------------------------------------------------
# Reactor migration
# ---------------------------------------------------------------------------


def test_selector_migration_keeps_queued_udp_packets():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    selector = selectors.DefaultSelector()
    try:
        server.bind(("127.0.0.1", 0))
        server.setblocking(False)
        selector.register(server, selectors.EVENT_READ, "udp")
        client.sendto(b"queued-before-migration", server.getsockname())

        migration = migrate_selector(selector)
        assert migration.packet_loss == 0
        assert migration.registrations[0].fd == server.fileno()
        migration.commit()

        events = migration.selector.select(timeout=2.0)
        assert events
        payload, _ = server.recvfrom(64)
        assert payload == b"queued-before-migration"
        migration.selector.close()
    finally:
        _close_quietly(selector)
        server.close()
        client.close()


def test_selector_migration_can_duplicate_descriptors():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    selector = selectors.DefaultSelector()
    try:
        server.bind(("127.0.0.1", 0))
        server.setblocking(False)
        selector.register(server, selectors.EVENT_READ, "udp")
        client.sendto(b"duplicated", server.getsockname())

        migration = migrate_selector(selector, duplicate_fds=True)
        assert server.fileno() in migration.duplicated
        migration.commit()
        assert migration.selector.select(timeout=2.0)
        assert server.recvfrom(64)[0] == b"duplicated"
        migration.selector.close()
        for fd in migration.duplicated.values():
            os.close(fd)
    finally:
        _close_quietly(selector)
        server.close()
        client.close()


def test_selector_migration_rollback_keeps_the_old_selector():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    selector = selectors.DefaultSelector()
    try:
        server.bind(("127.0.0.1", 0))
        server.setblocking(False)
        selector.register(server, selectors.EVENT_READ, "udp")
        migration = migrate_selector(selector)
        assert migration.rollback() is selector
        assert migration.rolled_back
        with pytest.raises(HotpatchError):
            migration.commit()
    finally:
        _close_quietly(selector)
        server.close()


def test_reactor_rejects_a_snapshot_without_a_poller():
    with pytest.raises(HotpatchError) as info:
        hotpatch_mio_reactor_state(object())
    assert info.value.code == "E008"


class _FakeBridge:
    def __init__(self, poll_fd: int, registrations) -> None:
        self._poll_fd = poll_fd
        self._registrations = list(registrations)
        self.reregistered: list[tuple[int, int, object]] = []
        self.closed = False

    def poll_fd(self) -> int:
        return self._poll_fd

    def snapshot_registrations(self):
        return list(self._registrations)

    def reregister(self, registrations) -> None:
        self.reregistered = list(registrations)

    def close_poller(self) -> None:
        self.closed = True


def test_mio_bridge_migration_moves_interests_without_closing_sockets():
    left, right = os.pipe()
    old_fd, new_fd = os.pipe()
    try:
        old = _FakeBridge(old_fd, [(left, 1, "read")])
        new = _FakeBridge(new_fd, [])
        migration = migrate_mio_bridge(old, new)
        assert migration.registrations[0].fd == left
        assert new.reregistered == [(left, 1, "read")]
        assert migration.packet_loss == 0
        migration.commit()
        assert old.closed
        with pytest.raises(HotpatchError) as info:
            migration.rollback()
        assert info.value.code == "E008"
    finally:
        os.close(left)
        os.close(right)
        os.close(old_fd)
        os.close(new_fd)


def test_mio_bridge_rejects_a_reused_poll_descriptor():
    left, right = os.pipe()
    poll_fd, poll_fd_other = os.pipe()
    try:
        old = _FakeBridge(poll_fd, [(left, 1, "read")])
        new = _FakeBridge(poll_fd, [])
        with pytest.raises(HotpatchError) as info:
            migrate_mio_bridge(old, new)
        assert info.value.code == "E008"
    finally:
        os.close(left)
        os.close(right)
        os.close(poll_fd)
        os.close(poll_fd_other)


def test_mio_bridge_rejects_an_invalid_replacement():
    left, right = os.pipe()
    old_fd, old_fd_other = os.pipe()
    try:
        old = _FakeBridge(old_fd, [(left, 1, "read")])
        with pytest.raises(HotpatchError) as info:
            migrate_mio_bridge(old, object())
        assert info.value.code == "E008"
        assert "replacement" in str(info.value)
        assert old.closed is False
    finally:
        os.close(left)
        os.close(right)
        os.close(old_fd)
        os.close(old_fd_other)


def test_mio_bridge_rollback_closes_only_the_replacement_poller():
    left, right = os.pipe()
    old_fd, old_fd_other = os.pipe()
    new_fd, new_fd_other = os.pipe()
    try:
        old = _FakeBridge(old_fd, [(left, 1, "read")])
        new = _FakeBridge(new_fd, [])
        migration = migrate_mio_bridge(old, new)
        assert new.reregistered == [(left, 1, "read")]
        assert not old.closed and not new.closed

        assert migration.rollback() is old
        assert new.closed
        assert not old.closed
        assert migration.rolled_back
        with pytest.raises(HotpatchError):
            migration.commit()
    finally:
        os.close(left)
        os.close(right)
        os.close(old_fd)
        os.close(old_fd_other)
        os.close(new_fd)
        os.close(new_fd_other)


def test_mio_bridge_migration_raii_closes_replacement_on_drop():
    left, right = os.pipe()
    old_fd, old_fd_other = os.pipe()
    new_fd, new_fd_other = os.pipe()
    try:
        old = _FakeBridge(old_fd, [(left, 1, "read")])
        new = _FakeBridge(new_fd, [])
        migration = migrate_mio_bridge(old, new)
        del migration
        gc.collect()
        assert new.closed
        assert not old.closed
    finally:
        os.close(left)
        os.close(right)
        os.close(old_fd)
        os.close(old_fd_other)
        os.close(new_fd)
        os.close(new_fd_other)


def test_selector_migration_raii_closes_new_poller_and_duplicates():
    read_fd, write_fd = os.pipe()
    selector = selectors.DefaultSelector()
    try:
        selector.register(read_fd, selectors.EVENT_READ, "read")
        migration = migrate_selector(selector, duplicate_fds=True)
        new_selector = migration.selector
        duplicated_fd = next(iter(migration.duplicated.values()))

        del migration
        gc.collect()
        with pytest.raises((ValueError, OSError)):
            new_selector.select(0)
        with pytest.raises(OSError):
            os.fstat(duplicated_fd)
    finally:
        _close_quietly(selector)
        os.close(read_fd)
        os.close(write_fd)


# ---------------------------------------------------------------------------
# Trampolines
# ---------------------------------------------------------------------------


def test_executable_region_runs_machine_code_and_sleds():
    if not ExecutableRegion.available():
        pytest.skip("anonymous executable pages are unavailable")
    first = ExecutableRegion()
    second = ExecutableRegion()
    try:
        first.write(0, _machine_code(1))
        second.write(0, _machine_code(2))
        assert _nullary(first.address) == 1
        assert _nullary(second.address) == 2

        patch = hotpatch_live_dylib_trampoline(region=first, replacement=second.address)
        assert patch.kind == "sled"
        assert _nullary(first.address) == 2
        assert patch.rollback() is True
        assert _nullary(first.address) == 1
        assert patch.rollback() is False
    finally:
        first.close()
        second.close()


def test_trampoline_refuses_foreign_executable_text():
    with pytest.raises(HotpatchError) as info:
        hotpatch_live_dylib_trampoline(region=object(), replacement=1)
    assert info.value.code == "E007"
    assert "owned" in str(info.value).lower() or "ExecutableRegion" in str(info.value)


def test_atomic_pointer_helpers_validate_addresses_before_dereference():
    page, slot = _slot_page()
    try:
        atomic_store_ptr(slot, 0x1234)
        assert atomic_load_ptr(slot) == 0x1234

        with pytest.raises(HotpatchError) as info:
            atomic_load_ptr(0)
        assert info.value.code == "E008"
        assert "NULL" in str(info.value)

        with pytest.raises(HotpatchError) as info:
            atomic_store_ptr(slot + 1, 0x99)
        assert info.value.code == "E008"
        assert "aligned" in str(info.value)
        assert atomic_load_ptr(slot) == 0x1234

        unmapped = 0x7FFF_FFFF_F000
        if _page_protection(unmapped) is not None:  # pragma: no cover - unusual host
            pytest.skip("chosen probe address is mapped on this host")
        with pytest.raises(HotpatchError) as info:
            atomic_load_ptr(unmapped)
        assert info.value.code == "E007"
        assert "mapped" in str(info.value)
    finally:
        page.close()


def test_trampoline_slot_patch_is_atomic_and_reversible():
    page, slot = _slot_page()
    one = ctypes.CFUNCTYPE(ctypes.c_int64)(lambda: 1)
    two = ctypes.CFUNCTYPE(ctypes.c_int64)(lambda: 2)
    ctypes.c_void_p.from_address(slot).value = _callable_address(one)
    try:
        assert _call_slot(slot) == 1
        patch = hotpatch_live_dylib_trampoline(
            slot=slot, replacement=_callable_address(two)
        )
        assert patch.old_value == _callable_address(one)
        assert _call_slot(slot) == 2
        assert patch.rollback() is True
        assert _call_slot(slot) == 1
    finally:
        page.close()


def test_trampoline_pointer_swap_is_visible_to_concurrent_readers():
    page, slot = _slot_page()
    one = ctypes.CFUNCTYPE(ctypes.c_int64)(lambda: 1)
    two = ctypes.CFUNCTYPE(ctypes.c_int64)(lambda: 2)
    addresses = (_callable_address(one), _callable_address(two))
    ctypes.c_void_p.from_address(slot).value = addresses[0]
    stop = threading.Event()
    failures: list[object] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                result = _call_slot(slot)
                if result not in (1, 2):
                    failures.append(result)
        except BaseException as error:  # pragma: no cover - failure path
            failures.append(error)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(40):
            patch = hotpatch_live_dylib_trampoline(slot=slot, replacement=addresses[1])
            patch.rollback()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
        page.close()
    assert not failures


def test_abi_signature_mismatch_reports_e008_with_byte_span():
    wrong = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)(
        lambda a, b: a + b
    )
    source = "éabcdefghij"
    with pytest.raises(HotpatchError) as info:
        hotpatch_live_dylib_trampoline(
            slot=0x1000,
            replacement=wrong,
            expected_signature="i64(i64)",
            span=(1, 5),
            source=source,
        )
    error = info.value
    assert error.code == "E008"
    assert error.span == (1, 5)
    assert error.bytes == (2, 6)
    assert "bytes: 2..6" in str(error)


def test_got_discovery_finds_a_bound_symbol():
    if resolve_symbol("getpid") is None:
        pytest.skip("getpid is not resolvable")
    try:
        slots = LiveTrampoline().discover("getpid")
    except HotpatchError as error:
        pytest.skip(str(error))
    assert slots
    assert slots == find_got_slots("getpid")
    assert all(slot.symbol == "getpid" for slot in slots)
    assert all(slot.address % 8 == 0 for slot in slots)


def test_signature_helpers_round_trip():
    from runtime.hotpatch.trampoline import parse_signature, signature_text

    restype, argtypes = parse_signature("i64(i64, ptr)")
    assert restype is not None
    assert len(argtypes) == 2
    procedure = ctypes.CFUNCTYPE(restype, *argtypes)(lambda value, pointer: value)
    assert signature_text(procedure) == "i64(i64, ptr)"


# ---------------------------------------------------------------------------
# Public surface and native source contracts
# ---------------------------------------------------------------------------


def test_public_capabilities_and_platform_probe():
    assert set(CAPABILITIES) == {
        "hotpatch_live_dylib_trampoline",
        "hotpatch_cpython_frame_eval",
        "hotpatch_mio_reactor_state",
        "hotpatch_rcu_ast_epoch",
        "hotpatch_rollback_journal_commit",
    }
    support = platform_support()
    assert support["selector"] is True
    assert support["platform"] == sys.platform
    for key in ("executable_region", "got_discovery", "frame_locals"):
        assert isinstance(support[key], bool)


def test_native_crate_contains_real_unsafe_primitives():
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(NATIVE.glob("src/*.rs"))
    )
    manifest = (NATIVE / "Cargo.toml").read_text(encoding="utf-8")
    for token in (
        "mach_vm_protect",
        "pthread_jit_write_protect_np",
        "MAP_JIT",
        "AtomicUsize",
        "from_ptr",
        "mach_vm_region",
        "_dyld_image_count",
        "fn adopt_from",
        "mio::Poll",
        "pthread_jit_write_protect_supported_np",
        "sys_icache_invalidate",
        "BeforeImage",
        "synchronize",
        "E008",
        "FrameLocalsProxy",
        "validate_pointer_cell",
        "validate_mapped_range",
        "restore_snapshots",
        "is_some_and",
    ):
        assert token in sources or token in manifest, token
    assert "pyo3" in manifest and "mio" in manifest
    assert 'panic = "abort"' not in manifest
    assert (NATIVE / "build.rs").exists()


@pytest.mark.native
def test_native_extension_builds_when_requested(tmp_path):
    if os.environ.get("POLYGLOT_BUILD_NATIVE_HOTPATCH") != "1":
        pytest.skip("set POLYGLOT_BUILD_NATIVE_HOTPATCH=1 to compile the PyO3 crate")
    import shutil
    import subprocess

    target = tmp_path / "target"
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "PYO3_PYTHON": sys.executable,
    }
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=NATIVE,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = next(
        path
        for path in target.glob("release/libpolyglot_hotpatch.*")
        if path.suffix in {".dylib", ".so"}
    )
    module_path = tmp_path / "polyglot_hotpatch.so"
    shutil.copy(artifact, module_path)
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        module = importlib.import_module("polyglot_hotpatch")
        assert module.E008 == "E008"
        assert module.native_platform_support()["rcu"] is True
        region = module.ExecutableRegion()
        assert region.size >= 4096
        region.close()
        domain = module.native_rcu_domain("test")
        domain.enter()
        domain.exit()
        assert domain.synchronize(1.0) == 0
        journal = module.native_journal()
        buffer = bytearray(b"abc")
        address = ctypes.addressof((ctypes.c_char * 3).from_buffer(buffer))
        journal.capture(address, 3, "buffer")
        buffer[0] = ord("z")
        assert journal.verify() is False
        assert journal.rollback() == 1
        assert bytes(buffer) == b"abc"
        with pytest.raises(RuntimeError, match="invalid native journal range"):
            journal.capture(0x7FFF_FFFF_F000, 4, "bogus")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("polyglot_hotpatch", None)
