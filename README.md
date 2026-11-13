# polyglot-lang

A Rust-like language with a shared lexer, parser, and AST, plus a Python backend and a native PyO3 extension backend.

## Quick Start

```sh
pip install polyglot-lang
```

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

> **First native build takes about 2 minutes.** Subsequent builds use a
> cache and take about 3 seconds. The Python backend (`polyglot run`)
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

**Current test status:** 469 passed, 9 skipped, 7 deselected (`pytest -q`); 7 passed, 464 deselected (`pytest -q -m slow`).

## Support matrix

| Feature | Python backend | Native backend |
|---|---|---|
| Vec, String, Option, Result | ✅ | ✅ |
| HashMap (core ops) | ✅ | ✅ |
| BTreeMap | ✅ | ⬜ Deferred |
| HashSet | ✅ | ⬜ Deferred |
| Nested HashMap (depth 1) | ✅ | ⬜ Deferred |
| Integer types (i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize) | ✅ | ✅ |
| `include!` directive | ✅ | ✅ |
| Error codes (E001–E008) | ✅ | ✅ |
| `#[terminates]` checking | ✅ | ✅ |
| async free functions | ✅ (synchronous) | ✅ |
| Box patterns | ✅ | ⬜ Deferred |
| Payload enum slices | ✅ | ⬜ Deferred |

## Features

- **Shared front end:** Turns source text into tokens, a parsed program, and AST nodes.
- **Python backend:** Translates the Rust-like AST to a Python AST, then compiles and executes it.
- **Native backend:** Generates Rust source with PyO3, builds it with Cargo, and loads the resulting `.so` or `.dylib` as a Python extension.
- **Common backend interface:** Select a backend with `get_backend("python")` or `get_backend("native")`. Calling `backend.compile(program)` returns a namespace dictionary.
- **CLI:** `polyglot run`, `polyglot build`, and `polyglot check` for file-based workflows.
- **N5:** Option and Result construction and combinators, the `?` operator, `if let`, async functions, match exhaustiveness warnings, and Python sandboxing.
- **N6:** Additional `Vec` methods (`first`, `last`, `get`, `truncate`, `rotate_*`, `fill`, `swap`, `resize`, `split_off`, `partition`, `sort_*`, `is_sorted`, and others) and `String` methods (`find`, `rfind`, `strip_*`, `split_once`, `lines`, `as_bytes`, `is_char_boundary`, and others).
- **N7:** Distinct `usize`/`isize` types, explicit `as` boundaries, range-loop variable typing, and selectable Python target-width simulation.
- **N8:** `HashMap<K, V>` construction and supported operations, typed `usize` length, dict conversion, key/value/iteration views, entry insertion, and mutable access.
- **N9:** Dedicated Python/native coverage for Platform-defined Result/Option combinators and native `Result::and_then`/`or_else`.
- **M10:** `#[terminates]` enables conservative static termination checks for bounded ranges, decreasing `while` variants, and recursive descent.
- **Module isolation:** Every Python compilation registers under a unique generated module name, so separate programs do not clobber each other in `sys.modules`.

## CLI

```sh
polyglot run <file> [--backend python|native] [--target-width N]
polyglot build <file> [--backend native] [--target <triple>] [--out-dir <dir>]
polyglot check <file>
```

- `run` compiles and executes a file and prints the result of `main()` when it returns a value.
- `build` compiles a file. The native backend produces an extension module and prints its path.
- `check` parses, translates (Python backend), and runs termination checks without executing.
- Errors use the toolchain five-line format. Pass `--verbose` to include a traceback.

The Python backend has no native dependencies. The native backend requires a Rust toolchain (`cargo` and `rustc`) and PyO3-compatible build tooling.

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

Native async functions can be awaited from Python with `asyncio.run`:

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
namespace = get_backend("native").compile(parse(lex(source)))
assert asyncio.run(namespace["add"](2, 3)) == 5
```

The native backend uses a hand-written `PyFuture` runtime, not `pyo3-asyncio`. Async support is currently limited to free functions, not impl methods, and futures must not require `Send`.

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
the cache, the first build takes about two minutes; a warm second run takes
about three seconds. Four slow tests, including source-line mapping, are
deselected by default.

See [NOTES.md](NOTES.md#build-performance) for detailed build-performance notes.

## Relationship to PyO3 and maturin

**Quick decision:** If you want full Rust control, wheel packaging, and
cross-platform builds, use maturin. If you want to write one Rust-like file
and get either a Python module or a native extension without writing
`Cargo.toml` or PyO3 scaffolding by hand, use `polyglot-lang`. Native builds
still run Cargo internally, and the current single-file model does not yet
expose arbitrary crate dependencies.

## Known limitations

- `is_ok`, `is_err`, `map_err`, `ok_or`, `and_then`, and `or_else` are Python-only inside let-bindings.
- `while let Some(x) = iter.next()` is out of scope because Python has no Rust-style Option-returning `next()`.
- Box patterns, enum payload slices, and nested Vec slice patterns are Python-only.
- Async is not supported on impl methods.
- `PyFuture` is unsendable and can only be used on the thread that created it.
- Option `map` is intentionally unsupported because it collides with Vec `map` in codegen; both backends reject it with a clear error. Use `match` or `if let` instead.
- `HashMap` supports only the implemented entry subset (`or_insert`, `or_insert_with`, `or_insert_with_key`, `and_modify`, `key`); other entry combinators are not yet lowered.
- `vec![...]` macro syntax is parsed as a token stream and is not lowered; use `[...].to_vec()` instead.
- `String::from(...)` is not lowered; use `"...".to_string()` instead.
- The `#[terminates]` checker accepts only single recursive calls that decrease one argument; a function with two recursive calls (e.g. classic `fib(n-1) + fib(n-2)`) is rejected even though it terminates.
- `if let` is supported as a statement, not as an expression. Use `match` when an expression is required.s
- `BTreeMap` and `HashSet` are available in the Python backend only. Native support is deferred. See SPEC 10.27.
- Nested `HashMap` (`HashMap<K, HashMap<K2, V2>>`) is supported in the Python backend only. Native support is deferred. See SPEC 10.25.
- `get_mut` on `BTreeMap` and `HashSet` follows the same Python-boundary copy semantics as `HashMap` (Python: mutable view; native: deferred). See SPEC 8.10.2.

## Running tests

Run the test suite from the project root:

```sh
pytest
```

The native backend caches build artifacts under `~/.cache/polyglot-native`. To clear the cache:

```sh
rm -rf ~/.cache/polyglot-native
```

## Architecture

`source → lexer → parser → AST → (Python AST → compile/exec | Rust + PyO3 → Cargo build → extension) → Python namespace`

## Roadmap

N0 through N10, M10, and F1 are complete. Remaining known limitations are listed under
[Known limitations](#known-limitations), including the out-of-scope items
`while let Some(x) = iter.next()`, async impl methods, and cross-thread
`PyFuture` use.

## Notes

See [NOTES.md](NOTES.md) for design decisions, per-batch history, and the
lessons recorded for each fixed bug.

## License

MIT License — see [LICENSE](LICENSE) for the full text.
