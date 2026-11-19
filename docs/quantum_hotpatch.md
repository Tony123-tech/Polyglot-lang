# Quantum hot-patching (`runtime/hotpatch`)

This subsystem is an opt-in research pillar. It is not imported by
`polyglot run` / `polyglot build`, is not part of the stable conformance
surface in `SPECIFICATION.md`, and refuses the operations it cannot perform
correctly instead of approximating them.

The five Python entry points share one diagnostic pipeline. Relocation, ABI,
frame-layout, reactor-validity, and undo-log failures are `E008`; unsupported
platforms or primitives are `E007`, and a missing replacement reactor is `E006`.
`HotpatchError` carries the caller's character span and, when the source text is
supplied, the exact UTF-8 byte span. The PyO3 surface mirrors the same codes in
`PyRuntimeError` text (`error[E007]` / `error[E008]`), not by raising the Python
`HotpatchError` class.

```python
from runtime.hotpatch import (
    hotpatch_live_dylib_trampoline,
    hotpatch_cpython_frame_eval,
    hotpatch_mio_reactor_state,
    hotpatch_rcu_ast_epoch,
    hotpatch_rollback_journal_commit,
)
```

## Support probe

```python
from runtime.hotpatch import CAPABILITIES, platform_support

platform_support()
# {'platform': 'darwin', 'executable_region': True, 'got_discovery': True,
#  'frame_locals': True, 'selector': True}
```

`CAPABILITIES` records the real mechanism and the deliberate boundary for each
entry point. Code that wants a specific primitive should check the probe rather
than assume the host can provide it.

## 1. Live-library trampolines

`hotpatch_live_dylib_trampoline` has two relocation paths.

* **GOT/PLT cell swap.** A bound symbol is resolved with `dlsym`. On macOS the
  Mach-O `__got` / `__la_symbol_ptr` / `__auth_got` / `__symbol_ptr` sections
  are scanned for pointer cells whose value equals that address. On Linux the
  implementation scans writable process mappings for the same value; it does
  not parse ELF `DT_JMPREL` relocations. The selected cell is widened to
  writable and rewritten with one aligned atomic word store. Threads that
  already loaded the old pointer keep executing it; the next indirect call
  observes the replacement.
* **Inline jump sled.** A `MAP_JIT` region on Apple Silicon (or an owned
  `RW -> RX` page elsewhere) receives a 16-byte AArch64 `ldr/br` sled or a
  12-byte x86-64 `mov rax, imm64; jmp rax` sled. The instruction cache is
  flushed after each write: before write protection is restored on Apple
  Silicon, and after the page is returned to RX on the `RW -> RX` path.

```python
from runtime.hotpatch import ExecutableRegion, hotpatch_live_dylib_trampoline

first = ExecutableRegion()
second = ExecutableRegion()
first.write(0, machine_code_for_1)
second.write(0, machine_code_for_2)

patch = hotpatch_live_dylib_trampoline(region=first, replacement=second.address)
assert patch.kind == "sled"
patch.rollback()  # restores the previous machine-code bytes
```

`ExecutableRegion` is the ownership token: only pages allocated by this
process are written. Foreign `__TEXT` is refused with `E007`. When a symbol has
more than one bound cell, or the ABI signature supplied through
`expected_signature` does not match the replacement, the call reports `E008`
before touching memory. Raw `slot=` addresses are checked for NULL, word
alignment, and an apparent live mapping before dereference; the caller still
owns the cell's lifetime, so a TOCTOU window remains.

## 2. CPython frame migration

`hotpatch_cpython_frame_eval` applies changes at safe points:

* the target function's `__code__` is swapped for future entries;
* frames currently executing that function keep the code object they captured,
  so `prev_instr`, the value stack, and `localsplus` cannot desynchronise;
* live fast locals are copied through CPython 3.13+ writable frame locals
  (PEP 667). On current CPython this is the write-through `frame.f_locals`
  object, not a separately exposed `FrameLocalsProxy` class;
* suspended generators can be migrated explicitly and stay on their captured
  code until recreated.

```python
patch = hotpatch_cpython_frame_eval(
    old_function,
    new_function,
    defaults={"extra": 0},
)
assert patch.pinned_frames == 1
patch.rollback()
```

Argument layout, generator/coroutine flags, closure cells, removed live locals,
and an evaluation-stack increase under a running frame are validated first.
A mismatch raises `E008` with the exact reason. If a later frame or generator
migration fails, completed frame snapshots and the function's `__code__` are
restored before the error is reported; rollback also deletes locals introduced
by the migration. An interpreter without writable frame locals raises `E007`;
the subsystem never pretends to rewrite a mid-instruction `f_code` pointer.

## 3. Reactor-state migration

`hotpatch_mio_reactor_state` replaces the poll object without closing the
application's sockets. The sockets keep their kernel queues, connection state,
and blocking mode; only the interest registrations move.

* **Python selector or mio bridge.** A pristine selector of the same kind is
  created and every `(fd, events, data)` registration is re-armed;
  `duplicate_fds=True` additionally `dup()`s descriptors when the old poller
  will be kept alive. An external native reactor instead supplies the
  `poll_fd()`, `snapshot_registrations()`, `reregister()`, and `close_poller()`
  bridge methods. Both paths are implemented in `runtime.hotpatch.reactor`, so
  the Python and PyO3 entry points return the same `ReactorMigration` or
  `MioBridgeMigration` object.
* **Direct native `mio::Poll`.** When both arguments are `NativeMioPoller`, the
  PyO3 entry point stays in Rust. The poller exposes `poll_fd`, `snapshot`,
  `adopt_from`, `wait`, and `close`; `adopt_from` re-arms `(fd, token,
  readable, writable)` registrations and, if one fails, deregisters those
  already adopted. It returns `NativeMioMigration`, whose `commit()` and
  `rollback()` close only the old or new poll descriptor, never the sockets.

```python
migration = hotpatch_mio_reactor_state(old_bridge, new_bridge)
assert migration.packet_loss == 0
migration.commit()  # closes the old poller, never the sockets
```

A migration that has been committed cannot be rolled back, and a migration
that has been rolled back cannot be committed. Both mistakes are `E008`. An
uncommitted migration is RAII-cleaned by its context manager or `__del__`: the
replacement selector/poller and any duplicated descriptors are closed. A
replacement mio bridge must implement the four bridge methods and must not
reuse the live poll descriptor; otherwise migration fails with `E008`.

## 4. RCU AST epochs

`hotpatch_rcu_ast_epoch` wraps an immutable `Program` root in a cooperative
epoch domain:

1. readers enter a guard and publish the epoch they observed;
2. a writer publishes a new root, retires the old one, and advances the
   global epoch;
3. `synchronize()` waits until every reader that entered before the advance
   has left, then runs the reclaim callbacks.

```python
manager = hotpatch_rcu_ast_epoch(program)
with manager.read() as root:
    use(root)
manager.publish(next_program)
manager.synchronize(timeout=5.0)
```

This is user-space RCU, not a kernel RCU syscall. Readers are cooperative: a
thread that never leaves its guard blocks reclamation, and the timeout is
reported as `E008` instead of silently freeing live memory. CPython's root swap
relies on the GIL; the native crate uses `AtomicUsize` epochs and a condvar.

## 5. Rollback journal

`hotpatch_rollback_journal_commit` captures before-images and replays them in
reverse order, so nested checkpoints unwind correctly.

* object attributes and slots are copied at attribute granularity;
  `deep=True` isolates nested lists/dicts/sets;
* writable buffers are captured byte-for-byte;
* parsed `Program` roots are snapshotted;
* the Python `RollbackJournal` restores captured state and then calls any
  user-supplied undo callbacks in reverse order; callback exceptions are
  collected and reported as `E008`, so rollback is **not** guaranteed to be
  callback-free. The native raw-range journal described below does not call
  user code.

```python
journal = hotpatch_rollback_journal_commit()
snapshot = journal.checkpoint(objects=[state], buffers=[payload], label="deploy")
mutate(state, payload)
assert journal.verify(snapshot) is False
journal.rollback(snapshot)
```

The native `NativeJournal` journals raw ranges, which is the correct seam for
GOT cells, JIT pages, and Rust-owned memory. Its FFI entry point refuses NULL,
empty, overflowing, and apparently unmapped ranges before reading them, but the
caller must keep the range alive for the whole checkpoint (a TOCTOU window
remains). The journal does **not** claim OS page-level copy-on-write: without
`fork`, the process has one address space. `deep=True` is the explicit way to
isolate containers.

## Native crate

`runtime/hotpatch/native` is a real PyO3 crate (`pyo3` 0.26, `libc`, `mio`)
with `unsafe` implementations of the same primitives:

* Mach-O GOT-section value scans, Linux writable-map value scans,
  `mach_vm_protect`/`mprotect`, `MAP_JIT`, `pthread_jit_write_protect_np`,
  `sys_icache_invalidate`, atomic slot stores;
* a native `RcuDomain` / `EpochRoot` with grace periods and panic-contained
  reclaimers;
* a before-image `Journal` over raw ranges;
* `NativeMioPoller` around a live `mio::Poll`;
* PEP 667 frame migration through the bound Python frame object.

Raw-address FFI primitives validate alignment and mapping before dereference.
The release profile keeps Rust's default unwind strategy so PyO3 can convert
Rust panics into Python exceptions and `catch_unwind` can contain a panicking
RCU reclaimer; abort-on-panic would break both.

The Python-visible classes are `ExecutableRegion`, `NativeTrampolinePatch`,
`NativeRcuDomain`, `NativeJournal`, `NativeMioPoller`, `NativeMioMigration`,
and `NativeFramePatch`. The manager-returning AST and object-journal paths, the
selector path, and migration of generic mio bridge objects delegate to the
Python host bridge so both backends return the same Python manager classes.
The direct two-`NativeMioPoller` path remains in Rust and returns
`NativeMioMigration`.

Build and test:

```sh
PYO3_PYTHON=$(pwd)/.venv/bin/python \
  CARGO_TARGET_DIR=$(pwd)/native_pillars/target/hotpatch \
  cargo build --manifest-path runtime/hotpatch/native/Cargo.toml

cargo test --manifest-path runtime/hotpatch/native/Cargo.toml --no-default-features
```

The default `extension-module` feature must be disabled for `cargo test` so
the test binary links against the interpreter that built the crate. Building
the extension module itself leaves CPython symbols unresolved (`-undefined
dynamic_lookup`), because the interpreter that loads the `.so` provides them.

Set `POLYGLOT_BUILD_NATIVE_HOTPATCH=1` to run the optional pytest that compiles
the crate, loads the resulting module, and exercises the native primitives.

## Tests

```sh
pytest -q tests/test_quantum_hotpatch.py
pytest -q tests/test_quantum_hotpatch.py -m native
pytest -q tests/test_hotpatch.py tests/test_capabilities.py
```
