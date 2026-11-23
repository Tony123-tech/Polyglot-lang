# Benchmark results

- Machine: arm64 / macOS-15.6.1-arm64-arm-64bit-Mach-O
- Python: 3.14.8
- Repetitions per cell: 3 (minimum reported)
- Polyglot numbers are `run_ms` from `polyglot exec`, so compilation
  and interpreter startup are excluded.
- CPython numbers are timed inside the script, interpreter startup excluded.
- `n/a (<0.5 ms)` means the run was too short to express a ratio.
- Every backend returned the same value as CPython.

| Workload | CPython (ms) | Polyglot python (ms) | Polyglot native (ms) | python vs CPython | native vs CPython |
| --- | --- | --- | --- | --- | --- |
| fib | 10.1 | 146.9 | 0.2 | 0.07x | n/a (<0.5 ms) |
| loop_sum | 317.4 | 164.5 | 0.0 | 1.93x | n/a (<0.5 ms) |
| strings | 5.1 | 62.9 | 14.4 | 0.08x | 0.35x |

A ratio above 1.00x means Polyglot was faster than CPython.
