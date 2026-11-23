# Deterministic simulation testing

`tools/dst/` gives every simulated run a virtual clock, a seeded scheduler, and
seeded fault injection. A failing interleaving can be replayed and then shrunk
from the same seed, which is the FoundationDB/TigerBeetle trick applied to this
project's own tooling.

## Virtual clock and scheduler

```python
from tools.dst import SimOp, dst_virtualize_clock_and_scheduler

ops = [SimOp(10, "a", "write", "hello"), SimOp(10, "b", "write", "world"), SimOp(20, "a", "fsync")]
schedule = dst_virtualize_clock_and_scheduler(ops, seed=7)
schedule.steps      # Steps with now_ms, node, kind, payload, seq
schedule.elapsed_ms # 20
```

- Operations sort by `at_ms`; ties break on a SHA-256 key over
  `(seed, at_ms, seq, node, kind)`, so the order is a pure function of the seed.
- The clock is monotonic. A negative `at_ms`, or one before `start_ms`, is
  `E001`; the scheduler never rewinds time to hide an ordering mistake.

## Fault injection

```python
from tools.dst import dst_inject_byzantine_and_io_faults

report = dst_inject_byzantine_and_io_faults(ops, seed=11, rates={"drop": 0.2, "torn_write": 0.1})
report.ops       # faulted operation list
report.injected  # {'drop': 1, 'torn_write': 1, ...}
```

Six fault kinds, all drawn in a fixed order from `random.Random(seed)`:

| Fault | Effect |
| --- | --- |
| `drop` | the operation disappears |
| `duplicate` | the operation is replayed with `seq + 1` |
| `reorder` | the operation swaps with its successor |
| `torn_write` | a write payload is truncated to half |
| `lost_fsync` | an fsync after a write is dropped |
| `bitflip` | one byte of a write payload is flipped |

Rates outside `[0, 1]` and unknown fault names are `E003`.

## Shrinking a failing schedule

```python
from tools.dst import dst_bisect_and_minimize_schedule

report = dst_bisect_and_minimize_schedule(ops, oracle, max_probes=2000)
report.minimal        # the smallest prefix-preserving subset that still fails
report.probes         # how many oracle calls it took
report.shrink_ratio   # len(minimal) / len(original)
```

The oracle is any deterministic `ops -> bool` predicate. The minimiser is delta
debugging over contiguous chunks, so it is not a proof of 1-minimality; the
probe budget is hard, and exhausting it returns `budget_exhausted=True`
rather than looping.

## Boundaries

The simulation models time, ordering, and I/O faults only. There is no thread
scheduler preemption model, no address-space simulation, and no disk persistence
— a `write` is a payload in a list. It is a harness for testing the project's own
algorithms, not a substitute for a real distributed runtime.
