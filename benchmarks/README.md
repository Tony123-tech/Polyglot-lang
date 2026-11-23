# Benchmarks

Three microbenchmarks, same algorithm in Polyglot and CPython:

| Workload | What it stresses |
| --- | --- |
| `workloads/fib.poly` | Function call overhead (recursive `fib(25)`) |
| `workloads/loop_sum.poly` | Per-iteration cost in a typed `for` loop (5M iterations) |
| `workloads/strings.poly` | String allocation and `replace`/`count` churn (20k iterations) |

Run everything and rewrite the report:

```sh
python benchmarks/run_benchmarks.py --write benchmarks/RESULTS.md
```

Just the Python backend (fast, no Rust toolchain needed):

```sh
python benchmarks/run_benchmarks.py --backends python
```

## How to read the numbers

- Polyglot times are `run_ms` from `polyglot exec --result-file`: process
  startup and compilation are excluded.
- CPython times are measured inside the script; interpreter startup is excluded.
- Each cell is the minimum of `--reps` runs (default 3).
- A ratio above `1.00x` means Polyglot was faster.
- `n/a (<0.5 ms)` means the run finished too fast to express a ratio honestly;
  read the absolute millisecond column instead.

## What the results actually say

On an Apple M1 (Python 3.14.8, see `RESULTS.md` for the current run):

- **Native wins decisively where it matters most.** Recursive `fib` drops from
  ~10 ms (CPython) to under half a millisecond, and the 5M-iteration integer loop
  is effectively free. That is the whole point of the native backend.
- **The Python backend is slower than CPython on call-heavy code** (~15x on
  `fib`), because every transpiled function wraps its arguments. This is a
  known, quantified weakness, not something to paper over.
- **The Python backend can beat CPython on tight typed loops** (~2x), since loop
  variables carry static integer types and avoid per-iteration dynamic lookup.
- **Native is slower than CPython on string churn** (~0.4x) in this workload:
  each iteration clones a `String` across the FFI boundary. String-heavy code
  should stay on the Python backend for now.

## Known costs worth optimising

1. Python-backend call overhead: inlining trivial functions or emitting
   direct C-level calls would close most of the `fib` gap.
2. `std:str_utils` takes `String` by value, forcing a clone per call. An
   `&String`/borrowed overload would remove the string-churn penalty.

These are honest targets, not promises. Re-run the script after any change;
`RESULTS.md` is generated, so it always matches the code that produced it.
