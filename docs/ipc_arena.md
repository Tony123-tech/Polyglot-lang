# Shared-memory ring arena

`runtime/ipc/` is a memory-mapped ring buffer plus epoch-fenced borrow leases.
It exists to make "no serialisation" measurable: a consumer reads a buffer straight
out of the mapping, and the guard against use-after-reclaim is an epoch comparison
rather than a per-byte check.

## Mapping the arena

```python
from runtime.ipc import ipc_map_shared_ring_arena

arena = ipc_map_shared_ring_arena("/tmp/polyglot.ring", slots=64, slot_size=256)
arena.header()  # ArenaHeader(version=1, slots=64, slot_size=256, head=0, tail=0, epoch=0, inflight=0)
```

Header layout (little endian, 56 bytes): magic `PLYGRING`, version, slot count,
slot size, next slot, `head`, `tail`, `epoch`, `inflight`. A file whose magic
does not match is `E008`; an unsupported version is `E007`; a missing file without
`create=True` is `E006`.

## Push and pop

```python
arena.push(b"tensor bytes")  # returns the slot index, or -1 when full
lease = arena.pop()  # RingLease with a memoryview into the mapping
bytes(lease.view[:13])  # b'tensor bytes'
arena.release(lease)  # advances tail
```

- A ring of `N` slots holds `N-1` payloads; the spare slot separates `head` from
  `tail`. A full ring returns `-1` instead of overwriting.
- While a lease is outstanding `inflight > 0` and `push` returns `-1`, which
  closes the window where a producer would overwrite a buffer being read.
- `pop` is zero-copy: `lease.view.obj` is the `mmap`. `push` copies the caller's
  bytes once, which is unavoidable across an address space.
- Leases release in FIFO order; an out-of-order `release` is `E003`.

Two handles on the same file share the header, so a producer in one mapping and
a consumer in another observe the same ring — that is the case covered by
`tests/test_ipc_arena.py::test_two_mappings_share_the_same_ring`.

## Borrow leases

```python
from runtime.ipc import ipc_enforce_epoch_borrow_lease, ipc_release_borrow_lease

borrow = ipc_enforce_epoch_borrow_lease(arena, lease, holder="worker-1")
borrow.token()  # 'worker-1:0:0:…'
ipc_release_borrow_lease(arena, holder="worker-1")
```

A lease is accepted only when its epoch equals the arena epoch. After a reclaim
the epoch moves on and an old lease is refused with `E002`, which is what stops
a consumer from reading a slot that has already been handed to someone else.
One live lease per holder keeps the table unambiguous (`E003` on duplicates).

## Reclaim

```python
from runtime.ipc import ipc_reclaim_abandoned_slots

report = ipc_reclaim_abandoned_slots(arena, now_epoch=arena.epoch + 1)
report.reclaimed  # holders evicted
report.dropped_queued  # queued slots discarded
```

Reclaim is deliberately blunt: it zeros the data region, clears leases, resets
`head`/`tail`/`inflight`, and bumps the epoch. `now_epoch` must advance past the
current epoch, so the destructive step cannot happen by accident (`E002`).

## Boundaries

Single host, fixed-size byte slots, and a per-process holder table. It is not
Arrow-aware, not a tensor transport, and not a substitute for a real SPSC queue
with atomic cursors — the cursors live in a shared mapping but this implementation
does not claim lock-free multi-threaded safety.
