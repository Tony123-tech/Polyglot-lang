# polyglot-lang

A Rust-like language with a shared lexer, parser, and AST, plus a Python backend, a native PyO3 extension backend, and an experimental compute-only WebAssembly backend.

## Why polyglot-lang?

You want Rust's syntax, type safety, and explicit error handling.
But you don't want to rewrite your whole project in Rust, or set up
`Cargo.toml` + PyO3 + maturin just to speed up one function.

`polyglot-lang` lets you write one `.poly` file:

```polyglot
fn add(a: i64, b: i64) -> i64 { a + b }
```

and choose how to run it:

- **Python backend** — `polyglot run add.poly` (no build step, no Rust)
- **Native backend** — `polyglot build add.poly` (compiles to a PyO3 extension)
- **WASM backend (experimental)** — `polyglot build add.poly --backend wasm` (writes WebAssembly text format)

Same source, same tests, same error messages. No `Cargo.toml`, no
`pyproject.toml` scaffolding.


## Quick Start

New to Rust-like syntax? See [GUIDE.md](GUIDE.md) for a step-by-step tutorial.

```sh
pip install polyglot-lang
```

Python 3.14.8 or later is required.

Create `hello.poly`:

```polyglot
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(2, 3) }
```

Run it:

```sh
polyglot run hello.poly
```

Output:

```text
5
```

Build a native extension (requires a Rust toolchain and Cargo):

```sh
polyglot build hello.poly
python -c "import hello; print(hello.add(2, 3))"
```

If you don't have a Rust toolchain, you can skip this step. The Python backend (`polyglot run`) works without it.

> **First native build takes about 2 minutes.** Subsequent builds reuse the
> cache and take a few seconds. The Python backend (`polyglot run`)
> has no build step.

## Status

| Milestone | Status | Scope |
|---|---|---|
| N0 | Done | Empty program → Cargo → load module |
| N1 | Done | Free functions, `i64`/`f64`/`bool`/`String`, `Vec`, structs with `impl` exposed as PyO3 classes, and `if` |
| N2a | Done | Native `while`, `loop`, and `match` |
| N2b | Done | `Vec` methods: `push`, `pop`, `len`, and indexing |
| N2c | Done | Multiple structs/enums as `pyclass` |
| N2d | Done | Improved caching and Rust-error mapping to source lines |
| N3 | Done | `Vec` iteration, `contains`, `sort`/`sort_by`, `map`/`filter`/`fold`, slicing, and mutation methods |
| N4 | Done | String length/iteration, split/join, replace/trim, case conversion, `parse`, `format!`, `write!`, slicing, and comparison |
| N5 | Done | Option/Result, `?`, `if let`, async free functions, match exhaustiveness warnings, and Python sandboxing |
| N6 | Done | Additional `Vec` and `String` methods, including search, slicing, rotation, and sort helpers |
| N7 | Done | Distinct `usize`/`isize` types, explicit casts, and target-width-aware boundary checks |
| N8 | Done | Collections-layer `HashMap` operations and Python/native dictionary boundaries |
| N9 | Done | Platform-defined Option/Result combinators and native Result continuations |
| N10 | Done | SPEC v0.2.7 and final conformance status |
| M10 | Done | Opt-in termination checking for `#[terminates]` functions |
| F1 | Done | Installable package, CLI, examples, and delivery documentation |
| M3 | Done | HashMap entry completion, `BTreeMap`, `HashSet`, and nested `HashMap` depth 1 |
| M4 | Done | Additional fixed-width integer types and the cross-backend cast matrix |
| M5 | Done | Send-aware `PyFuture`, async `&self` impl methods, drop-based cancellation, a `mio` reactor with timers, and an `asyncio` bridge |

**Current test status:** 11361 passed, 11 skipped, 1129 deselected (`pytest -q`, 1 xpassed); 19 passed, 12483 deselected (`pytest -q -m slow`).

## Support matrix

| Feature | Python backend | Native backend |
|---|---|---|
| Vec, String, Option, Result | ✅ | ✅ |
| HashMap (core ops) | ✅ | ✅ |
| HashMap mutable iterators (`iter_mut`, `values_mut`) | ✅ | ✅ |
| BTreeMap | ✅ | ✅ |
| HashSet | ✅ | ✅ |
| Nested HashMap (depth 1) | ✅ | ✅ |
| Integer types (i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize) | ✅ | ✅ |
| `include!` directive | ✅ | ✅ |
| Bundled `std:` standard library | ✅ | ✅ |
| Error codes (E001–E008) | ✅ | ✅ |
| `#[terminates]` checking | ✅ | ✅ |
| async free functions | ✅ (synchronous) | ✅ |
| async `&self` impl methods | ✅ | ✅ |
| cross-thread await with Send captures | n/a | ✅ |
| drop-based cancellation | n/a | ✅ |
| reactor timer wakeups | n/a | ✅ |
| `asyncio` awaiting of pending futures | n/a | ✅ |
| Box patterns | ✅ | ⬜ Deferred |
| Payload enum slices | ✅ | ⬜ Deferred |

`BTreeMap` iteration is sorted by key in both backends. `HashSet`
iteration order is unspecified; sort before comparing. `get_mut` on
`BTreeMap`/`HashSet` returns an owned copy at the native Python boundary
(see SPEC 8.10.2 and 10.20).

The experimental WASM backend is not part of this matrix. It supports only
`i64` and `bool` values, with no strings, aggregates, or host imports.

## Features

- **Shared front end:** Turns source text into tokens, a parsed program, and AST nodes.
- **Python backend:** Translates the Rust-like AST to a Python AST, then compiles and executes it.
- **Native backend:** Generates Rust source with PyO3, builds it with Cargo, and loads the resulting Python extension (`.so` on macOS and Linux, `.pyd` on Windows).
- **WASM backend (experimental):** Lowers an `i64`/`bool` compute-only subset to WebAssembly text format (`.wat`) and runs it through `wasmtime`.
- **Common backend interface:** Select a backend with `get_backend("python")`, `get_backend("native")`, or `get_backend("wasm")`. Calling `backend.compile(program)` returns a namespace dictionary.
- **CLI:** `polyglot run`, `polyglot build`, `polyglot check`, `polyglot lsp`, and the compatibility `polyglot compile` command.
- **N5:** Option and Result construction and combinators, the `?` operator, `if let`, async functions, match exhaustiveness warnings, and Python sandboxing.
- **N6:** Additional `Vec` methods (`first`, `last`, `get`, `truncate`, `rotate_*`, `fill`, `swap`, `resize`, `split_off`, `partition`, `sort_*`, `is_sorted`, and others) and `String` methods (`find`, `rfind`, `strip_*`, `split_once`, `lines`, `as_bytes`, `is_char_boundary`, and others).
- **N7:** Distinct `usize`/`isize` types, explicit `as` boundaries, range-loop variable typing, and selectable Python target-width simulation.
- **N8:** `HashMap<K, V>` construction and supported operations, typed `usize` length, dict conversion, key/value/iteration views, mutable iterators, entry insertion, and mutable access.
- **N9:** Dedicated Python/native coverage for Platform-defined Result/Option combinators and native `Result::and_then`/`or_else`.
- **M10:** `#[terminates]` enables conservative static termination checks for bounded ranges, decreasing `while` variants, and recursive descent.
- **M5:** Send-aware `PyFuture`, async `&self` impl methods, drop-based cancellation, a single-threaded `mio` reactor for timer and Unix fd wakeups, cross-thread Send awaits, and an `asyncio` bridge for pending futures.
- **Bundled stdlib:** `include!("std:...")` resolves eight pure-`.poly` modules (`math`, `algo`, `str_utils`, `collections`, `numpy`, `linalg`, `random`, `prelude`) from the installed package on both backends. `std:` includes are deduplicated; ordinary includes are still textual.
- **Module isolation:** Every Python compilation registers under a unique generated module name, so separate programs do not clobber each other in `sys.modules`.

## CLI

```sh
polyglot run <file> [--backend python|native|wasm] [--target-width N]
polyglot build <file> [--backend python|native|wasm] [--out-dir <dir>] [--incremental]
polyglot check <file> [--all-errors]
polyglot lsp
polyglot compile --backend python|native|wasm (--file <path> | --stdin)
```

- `run` compiles and executes a file and prints the result of `main()` when it returns a value. `--target-width 32` or `64` simulates the corresponding `usize`/`isize` width on the Python backend.
- `build` compiles a file. The native backend produces an extension module, the WASM backend writes a `.wat` file, and both print the artifact path. `--incremental` reuses per-function Rust fragments across native rebuilds.
- `check` parses, translates (Python backend), and runs termination checks without executing. `--all-errors` recovers from top-level syntax errors and reports every parse error it finds.
- `lsp` runs the stdio language server.
- `compile` is a compatibility command used by the HTTP executor and older tooling. It prints the exported top-level names.
- Errors use the toolchain five-line format, or six lines when a span is available. Pass `--verbose` to include a traceback.
- `--error-format text|json` selects the diagnostic format on `run`, `build`, `check`, and `compile`. JSON output is a single object with `code`, `message`, `line`, `col`, `span`, and `source`.

The Python backend has no native dependencies. The native backend requires a Rust toolchain (`cargo` and `rustc`) and PyO3-compatible build tooling. The WASM backend requires the optional `wasm` extra (`wasmtime`).

## Include directive

`include!("path.poly");` splices the items of another `.poly` file at that point. Paths are resolved relative to the including file, and includes may nest.

```polyglot
// math.poly
fn add(a: i64, b: i64) -> i64 { a + b }

// main.poly
include!("math.poly");
fn main() -> i64 { add(2, 3) }
```

- Includes are one-way; the included file does not see the includer's items.
- Cycles are rejected with `E005`, and missing files with `E006`.
- Including the same file twice is allowed and splices its items twice. This is textual inclusion: duplicate item definitions are the author's responsibility.
- `include!` is only valid at item position. Using it inside a block is a parse error.
- Absolute paths and `..` segments are allowed. There is no repository-containment policy.

The directive is expanded before code generation, so both backends see the same spliced program.

### Bundled standard library

Use the reserved `std:` namespace to include a module that ships with the package:

```polyglot
include!("std:prelude");

fn main() -> i64 {
    let xs: Vec<i64> = [3, 1, 2].to_vec();
    let sorted = bubble_sort(xs);
    let m = match from_vec([1.0, 2.0, 3.0, 4.0].to_vec(), 2, 2) {
        Ok(v) => v,
        Err(_) => zeros(0, 0),
    };
    sorted[0] + (m.trace() as i64)
}
```

The bundled modules are pure `.poly` and require no extra dependencies:

| Module | Contents |
|---|---|
| `std:math` | integer helpers, `gcd`/`lcm`/`pow_mod`/`is_prime`, and pure `f64` `sqrt`/`exp` |
| `std:algo` | sorting, searching, sums, extrema, `unique`, and occurrence counting |
| `std:str_utils` | trimming, case conversion, split/join, padding, search, and reversal |
| `std:collections` | `Stack`, `Queue`, `SetI64`, and `Counter` |
| `std:numpy` | dense `NdArray` creation, element-wise ops, reductions, and matrix products |
| `std:linalg` | `inv`, `solve`, and `determinant` via Gauss-Jordan/Gaussian elimination |
| `std:random` | deterministic Xoshiro256** `Rng` with `next_u64`/`next_f64`/ranges/shuffle |
| `std:prelude` | includes every module above |

- `include!("std:math")` appends `.poly` automatically; `include!("std:math.poly")` also works.
- `std:` paths are always resolved inside the packaged `stdlib/` directory. Missing modules and paths that escape that directory report `E006`.
- `std:` includes are deduplicated by resolved path, so `std:numpy` can be included by the program, `std:linalg`, and `std:prelude` at once without duplicating `NdArray`. Ordinary includes keep the textual duplicate-splice behaviour.
- There is no package manager. `std:` modules are part of this distribution and use only features described in the specification.

## Examples

The `examples/` directory contains runnable programs:

- `hello.poly` — function call and return value
- `fib.poly` — recursive `#[terminates]` function
- `vec_ops.poly` — Vec mutation, length, and sorting
- `string_ops.poly` — split, join, and replace
- `hashmap.poly` — HashMap insert/get/iteration
- `option_result.poly` — Option, Result, `?`, and `if let`
- `terminates.poly` — `#[terminates]` decreasing loop

Run them with:

```sh
polyglot check examples/hello.poly
polyglot run examples/hello.poly
```

## Primitive semantics

Explicit numeric types follow Rust where Python's defaults differ:

- `as` casts truncate and reinterpret integer values at the target width; `isize`/`usize` are 64-bit, `f32` rounds through single precision, and `char` uses Unicode code points.
- `/` truncates toward zero for integers; `%` keeps the dividend's sign. Float `%` uses `fmod`.
- Annotated numeric bindings, assignments, compound assignments, and returns wrap at the declared width, matching native release builds.

Overflow parity is limited to explicitly typed bindings; unannotated inference still uses Python's unbounded integers. These semantics are covered by `tests/test_native_n8_types.py`.

## Async

Native async functions return a `PyFuture` that can be awaited from Python.
The future works both with `asyncio.run` directly and inside a coroutine:

```python
import asyncio

from backends import get_backend
from lexer import lex
from parser import parse

source = """
async fn add(a: i64, b: i64) -> i64 {
    a + b
}
"""
namespace = get_backend("native", trusted=True).compile(parse(lex(source)))
assert asyncio.run(namespace["add"](2, 3)) == 5
```

The native backend uses a hand-written `PyFuture` runtime, not
`pyo3-asyncio`. Futures are Send-aware: a future created on one thread can be
awaited on another when its captured values are Send. Async `&self` impl
methods also return `PyFuture`; referenced fields are cloned into owned
locals before the future is created, so the method body does not capture
`self`.

A single reactor thread drives the native futures. Each task has a waker that
re-enqueues it for polling, timers are delivered from a deadline heap, and
Unix file-descriptor readiness uses `mio`. When Python awaits a future that is
not ready, `PyFutureIterator` returns an `asyncio` future created on the
running loop; the reactor completes it thread-safely with
`loop.call_soon_threadsafe`, so there is no busy-polling. Dropping a future or
its iterator before completion cancels it by drop semantics.

The runtime also provides internal `polyglot_sleep_ms` and Unix-only
`polyglot_readable` helpers, plus an `await_with_timeout(future, seconds)`
helper. These are runtime and test helpers, not language-level APIs.
`await_with_timeout` waits on a condition-variable waker with the GIL
released and raises `PyTimeoutError` on expiry. Async `&mut self` methods are
rejected by codegen, as are async methods that use bare `self` or capture
non-`Clone` fields.

## Sandbox

`PythonBackend(sandbox=True)` restricts `__builtins__` to a safe subset, independently of the `trusted` flag. It blocks `__import__`, `eval`, `open`, `compile`, and `globals`. The safe subset includes `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `format`, `frozenset`, `int`, `isinstance`, `issubclass`, `iter`, `len`, `list`, `map`, `max`, `min`, `next`, `object`, `print`, `range`, `repr`, `reversed`, `round`, `set`, `slice`, `sorted`, `str`, `sum`, `tuple`, `type`, `zip`, the `True`/`False`/`None` constants, standard exceptions, and `__build_class__`.

```python
from backends.python.translator import PythonBackend

namespace = PythonBackend(sandbox=True).compile(program)
```

For isolation levels A/B/C and the `X-Trusted-Token` requirement, see the
Security section in [NOTES.md](NOTES.md#security).

`trusted=False` is not a sandbox; it is a speed bump. The sandbox restricts
`__builtins__` but does not provide process isolation. Treat untrusted input
as requiring OS-level isolation, such as the subprocess or Docker levels
described in `NOTES.md` (isolation B or C), with network access disabled and
a read-only filesystem where appropriate.

## Performance

Native builds use `sccache` through `RUSTC_WRAPPER` when it is available.
Cargo's `target` directory stays within each build directory. After clearing
the cache, the first build takes about two minutes; a warm cached build takes
seconds. Nineteen slow tests, including source-line mapping, are deselected by
default.

See [NOTES.md](NOTES.md#build-performance) for detailed build-performance notes.

## Relationship to PyO3 and maturin

**Quick decision:** If you want full Rust control, wheel packaging, and
cross-platform builds, use maturin. If you want to write one Rust-like file
and get either a Python module or a native extension without writing
`Cargo.toml` or PyO3 scaffolding by hand, use `polyglot-lang`. Native builds
still run Cargo internally, and the current single-file model does not yet
expose arbitrary crate dependencies.

## Error codes

Diagnostics include a stable error code. The table below maps each code to its class and meaning.

| Code | Class | Meaning |
|---|---|---|
| `E001` | `TypeErrorForBoundary` | An implicit conversion between different integer types was attempted. Add an explicit `as` cast. See SPEC 2.4.6–2.4.7. |
| `E002` | `TerminationError` | A function annotated `#[terminates]` could not be proven to terminate. See SPEC 2.8. |
| `E003` | `TranslateError` | A generic AST-to-backend translation failure (unsupported construct, invalid pattern, etc.). Python backend only; the native backend reports `E004` or `E007`. |
| `E004` | `RustCompileError` | The generated Rust failed to compile. The message includes the original source location when available. |
| `E005` | `IncludeCycleError` | A cyclic `include!` chain was detected. The message prints the full chain. See SPEC 9.2. |
| `E006` | `IncludeNotFoundError` | An `include!` target file does not exist or is unreadable. See SPEC 9.2. |
| `E007` | `UnsupportedNativePatternError` | A source pattern cannot be represented by stable Rust. See SPEC 8.3. |
| `E008` | `InvalidIntegerParseLiteralError` | A literal integer could not be parsed for the target type. |

## Known limitations

- `is_ok`, `is_err`, `map_err`, `ok_or`, `and_then`, and `or_else` are Python-only inside let-bindings.
- `while let Some(x) = iter.next()` is out of scope because Python has no Rust-style Option-returning `next()`.
- Box patterns, enum payload slices, and nested Vec slice patterns are Python-only.
- Async `&mut self` methods are rejected by the native codegen. Async `&self`
  methods may only read fields (`self.<field>`); bare `self` use and
  non-`Clone` fields such as payload enums are rejected.
- Async is Experimental. The Python backend executes async functions
  synchronously. Native futures work with `asyncio`, but the language does not
  yet expose timer or I/O primitives: `polyglot_sleep_ms`,
  `polyglot_readable`, and `await_with_timeout` are runtime/test helpers only.
  See SPEC 8.2 and 10.34.
- The WASM backend is experimental and compute-only. It supports `i64`/`bool`
  values with no strings, aggregates, or host imports.
- `HashMap` supports only the implemented entry subset (`or_insert`, `or_insert_with`, `or_insert_with_key`, `and_modify`, `key`); other entry combinators are not yet lowered.
- `vec![...]` macro syntax is parsed as a token stream and is not lowered; use `[...].to_vec()` instead.
- `String::from(...)` is not lowered; use `"...".to_string()` instead.
- The `#[terminates]` checker accepts a **single** recursive call that
  strictly decreases one argument. Multiple recursive calls (e.g. classic
  `fib(n-1) + fib(n-2)`) are deferred and therefore rejected even though
  they terminate. Mutual recursion, loops with `break`, complex arithmetic,
  non-recursive calls, and async suspension are also deferred. See SPEC 10.22.
- `if let` is supported as a statement and as the final expression of a
  block (e.g. the tail of `main()`), but not as a sub-expression inside a
  `let` binding or a function argument. Use `match` when an `if let`
  expression is required inside another expression.
- `BTreeMap` and `HashSet` are available in both backends for the type sets in SPEC 7.2.6. Deeper composition and unsupported key/value combinations remain deferred. See SPEC 10.25 and 10.27.
- Nested `HashMap` (`HashMap<K, HashMap<K2, V2>>`) is supported at depth 1 in both backends. Deeper nesting is deferred. See SPEC 10.25.
- `get_mut` on `BTreeMap` and `HashSet` follows the same Python-boundary copy semantics as `HashMap` (Python: mutable view; native: owned copy that cannot mutate the input). See SPEC 8.10.2 and 10.20.
- The bundled `std:` modules are pure `.poly` and inherit the language subset. They are not a package manager, do not expose arbitrary Cargo dependencies, and only `std:` includes are deduplicated. See SPEC 9.2 and 10.35.

## Running tests

Run the test suite from the project root:

```sh
pytest -q
pytest -q -m slow
```

`pytest -q` deselects slow tests by default. `pytest -q -m slow` runs only
the slow tests. `make check` runs the linters, the docs-sync check, and the
fast suite.

The native backend caches build artifacts under `~/.cache/polyglot-native`. To clear the cache:

```sh
rm -rf ~/.cache/polyglot-native
```

## Architecture

`source → lexer → parser → AST → (Python AST → compile/exec | Rust + PyO3 → Cargo build → extension) → Python namespace`

## Roadmap

N0 through N10, M10, F1, and M5 are complete. M5 now includes a real native
reactor thread, timer and Unix fd wakeups, cross-thread Send awaits, and an
`asyncio` bridge. Remaining known limitations are listed under
[Known limitations](#known-limitations). The main out-of-scope items are
`while let Some(x) = iter.next()`, async `&mut self` methods, language-level
async timer/I/O primitives, and the compute-only WASM subset.

## Notes

See [NOTES.md](NOTES.md) for design decisions, per-batch history, and the
lessons recorded for each fixed bug.

## License

MIT License — see [LICENSE](LICENSE) for the full text.
