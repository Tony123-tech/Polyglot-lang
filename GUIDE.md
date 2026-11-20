# Polyglot Language Guide

This guide is for Python developers who have never written Rust.
It starts from zero and ends with a working program that uses
Polyglot's type system, collections, and error handling.

If you already know Rust, skip to the [README](README.md) or
[SPECIFICATION](SPECIFICATION.md).

## 1. Install

```sh
pip install polyglot-lang
```

Python 3.14.8 or later is required. The Python backend has no native
dependencies. You can run `.poly` files immediately.

For the native backend, you also need a Rust toolchain:

```sh
rustc --version
cargo --version
```

If those commands work, you can compile `.poly` files to native
extensions.

The experimental Zig backend needs only the Zig toolchain:

```sh
zig version
```

If that prints a version, you can try `--backend zig`. Zig is not required for
the Python or native backends.

## 2. Your first `.poly` file

Create `hello.poly`:

```polyglot
fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn main() -> i64 {
    add(2, 3)
}
```

Run it:

```sh
polyglot run hello.poly
```

You should see:

```text
5
```

### What just happened

- `fn add(a: i64, b: i64) -> i64` declares a function named `add`
  that takes two 64-bit integers and returns a 64-bit integer.
- `fn main() -> i64` is the entry point. `polyglot run` calls `main()`
  and prints its return value.
- `add(2, 3)` calls the function. The result is `5`.

### How this differs from Python

- Types are explicit. `a: i64` means "`a` is a 64-bit integer."
- `fn` replaces `def`.
- `-> i64` replaces Python's return type hint.
- There is no `print()` in `main()`. `polyglot run` prints the return
  value of `main()` automatically.

### Splitting a program across files

Use `include!` to splice another file's items at item position:

```polyglot
// math.poly
fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

```polyglot
// hello.poly
include!("math.poly");

fn main() -> i64 {
    add(2, 3)
}
```

Paths are resolved relative to the including file. `include!` cannot be
used inside a function body, and including the same file twice splices
its items twice (there is no de-duplication).

## 3. Variables and mutability

Variables are **immutable by default**:

```polyglot
fn main() -> i64 {
    let x = 10;
    x
}
```

If you try to change `x`, you get an error. To allow mutation:

```polyglot
fn main() -> i64 {
    let mut x = 10;
    x = 20;
    x
}
```

This is the biggest difference from Python: **you must say `mut` when
you want to change a value.** The compiler enforces it.

## 4. Control flow

### If

```polyglot
fn main() -> i64 {
    let x = 5;
    if x > 0 {
        1
    } else {
        -1
    }
}
```

### While

```polyglot
fn main() -> i64 {
    let mut i = 0;
    while i < 5 {
        i = i + 1;
    }
    i
}
```

### For

```polyglot
fn main() -> i64 {
    let mut total = 0;
    for i in 0..5 {
        total = total + i;
    }
    total
}
```

`0..5` is a **range**: it yields `0, 1, 2, 3, 4`. The end is exclusive.
Use `0..=5` for an inclusive range.

## 5. Vec (Python list, but typed)

```polyglot
fn main() -> i64 {
    let mut xs = [1, 2, 3].to_vec();
    xs.push(4);
    xs.pop();
    xs.len() as i64
}
```

`[1, 2, 3].to_vec()` creates a list. `vec![1, 2, 3]` is an equivalent
shortcut, and `vec![7; 3]` creates `[7, 7, 7]` (the count is a `usize`).
Both forms work identically on the Python and native backends. `xs.push(4)`
appends. `xs.pop()` removes the last element. `xs.len()` returns a
`usize`, so you need `as i64` to return it as an `i64`.

### Why `as i64`?

`len()` returns `usize` (a platform-sized unsigned integer).
Your `main()` returns `i64`. Polyglot does not convert between them
automatically:

```text
error[E001]: type mismatch at line 5, column 7
  found:    usize
  expected: i64
  help: write `as i64` here
  note: `as` may wrap or truncate the value; check the range first
  at: line 5, column 7 (span 90..92)
```

To fix it, write `xs.len() as i64`. This is the **only** place in
Polyglot where a cast is required for a common operation, and the
error message tells you exactly what to do.

## 6. Strings

```polyglot
fn main() -> i64 {
    let s = "hello, world".to_string();
    let parts = s.split(", ");
    parts.len() as i64
}
```

`"...".to_string()` creates a string. `String::from("...")` is an equivalent
shortcut and is lowered by both backends.
`split`, `join`, `replace`, `trim`, and `contains` work like Python
strings, but they return Polyglot types (`Vec<String>`, `Option<usize>`).

## 7. Option and Result

These are the two types that replace `None` and exceptions.

### Option

```polyglot
fn find_even(xs: Vec<i64>) -> Option<i64> {
    for x in xs {
        if x % 2 == 0 {
            return Some(x);
        }
    }
    None
}

fn main() -> i64 {
    match find_even([1, 3, 4, 5].to_vec()) {
        Some(x) => x,
        None => -1,
    }
}
```

`Some(x)` wraps a value. `None` is the absence. `match` extracts the
value safely; `unwrap_or` is the shorter form when you only need a
default. `unwrap_or_else(|| expensive())` is the lazy form: the closure
runs only when the value is `None`.

### Result

```polyglot
fn parse_number(s: String) -> Result<i64, String> {
    let n = s.parse::<i64>()?;
    Ok(n)
}

fn main() -> i64 {
    match parse_number("42".to_string()) {
        Ok(n) => n,
        Err(_) => -1,
    }
}
```

`Ok(n)` is a success. `Err(e)` is a failure. The `?` operator
propagates an `Err` to the caller. `unwrap_or` returns the `Ok` value or
a default on `Err`; `unwrap_or_else(|_| fallback())` computes that default
only on `Err`.

### `if let` shortcut

```polyglot
fn main() -> i64 {
    if let Some(x) = find_even([1, 3, 4].to_vec()) {
        x
    } else {
        -1
    }
}
```

`if let` is shorter than `match` when you only care about one branch. The
builtin `Some`, `None`, `Ok`, and `Err` patterns are supported, and `if let`
can be used as a statement or as an expression (including `let` initialisers
and call arguments):

```polyglot
fn main() -> i64 {
    let y = if let Some(v) = find_even([1, 3, 4].to_vec()) { v } else { 0 };
    y
}
```

Other if-let pattern forms still raise a translation error.

### Reading and writing files

`include!("std:fs");` adds `Result`-returning file helpers. They are only
available to trusted compilation paths (`polyglot run`, `polyglot build`,
and a trusted backend instance):

```polyglot
include!("std:fs");

fn demo(path: String) -> Result<String, String> {
    write_text(path, "hello".to_string())?;
    read_text_nonempty(path)
}
```

`read_text`, `write_text`, `append_text`, and `remove` return `Result`;
`exists` returns `bool`. `read_text_or`, `write_text_if_missing`, and
`read_text_nonempty` are convenience wrappers. On the native backend an
`Err` payload is a `PyErr`, so portable code should match on `Err(_)`
instead of binding the payload as a language string. See SPEC 8.11.

## 8. When to use the native backend

The Python backend is fine for most programs. Use the native
backend when:

- You need **C-level performance** for a hot loop.
- You want to **ship a `.so`/`.pyd`** instead of Python source.
- You are already using PyO3 and want a simpler single-file alternative.

Build a native extension:

```sh
polyglot build hello.poly
```

This copies an importable extension next to `hello.poly`. The artifact
is named `hello.so` on macOS and Linux and `hello.pyd` on Windows; on
macOS, Cargo's `.dylib` output is copied to the `.so` suffix Python
imports. Use `--out-dir out/` to write it elsewhere:

```sh
polyglot build hello.poly --out-dir out/
```

The file stem determines both the artifact name and the Python module
name, so `hello.poly` becomes `import hello`. You can import it from
Python:

```python
import hello

print(hello.add(2, 3))
```

The first build takes about 2 minutes. Subsequent builds reuse the cache
under `~/.cache/polyglot-native` (created with mode `0700`) and take a few
seconds. Builds are keyed by a hash of the generated Rust, so editing the
`.poly` source produces a new cache entry. Add `--incremental` to reuse
per-function Rust fragments across edits. To reclaim disk space, remove
the cache directory:

```sh
rm -rf ~/.cache/polyglot-native
```

## 9. When to use the Zig backend (experimental)

The Zig backend is an opt-in execution mode for experimenting with Zig-style
integer semantics, threads, and vectors. It is **not** part of the
Python/native parity contract. Use it when you specifically want to try:

- saturating arithmetic that clamps instead of wrapping;
- arbitrary bit-width integers such as `u8`, `u24`, or `i19`;
- a `#[parallel]` range loop;
- `SimdN<T>` vector operations and horizontal reductions.

For normal programs, keep using the Python or native backend. The Zig lowering
covers only a small AST subset, explained below.

### Run a first Zig program

Create `sat.poly`:

```polyglot
fn main() -> i64 {
    (250 as u8 +| 20 as u8) as i64
}
```

Run it with:

```sh
polyglot run sat.poly --backend zig
```

Output:

```text
255
```

The `+|` operator adds with saturation. `250u8 + 20` would normally overflow;
instead it clamps at `255`. The sibling operators are `-|` and `*|`:

```polyglot
fn sat_sub(a: u8, b: u8) -> u8 { a -| b }
fn sat_mul(a: u8, b: u8) -> u8 { a *| b }
```

`sat_sub(10, 30)` returns `0`, and `sat_mul(20, 20)` returns `255`.

### Arbitrary bit-width integers

Any `uN` or `iN` with `N` between 0 and 65535 is accepted:

```polyglot
fn sat_u24(a: u24, b: u24) -> u24 {
    a +| b
}
```

From Python, call it through the backend:

```python
from backends import get_backend
from lexer import lex
from parser import parse

source = """
fn sat_u24(a: u24, b: u24) -> u24 { a +| b }
"""
namespace = get_backend("zig", trusted=True).compile(parse(lex(source)))
print(namespace["sat_u24"](16_777_210, 100))  # 16777215
```

Zig stores the value at its declared width. The Python `ctypes` boundary is
64-bit, so values wider than 64 bits are truncated when they cross into
Python. The lowering fallback used by non-Zig execution clamps at `i64`
bounds and is not bit-width aware.

### Parallel range loops

Add `#[parallel]` to a function with a direct range `for` loop:

```polyglot
#[parallel]
fn scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
    for i in 0..n {
        out[i] = (inp[i] * factor) + 1.0;
    }
}
```

The backend splits `0..n` into chunks and runs them on up to 16 worker threads
(capped by the available CPU count). Compile the function and call it from
Python with `ctypes` arrays:

```python
import ctypes

from backends import get_backend
from lexer import lex
from parser import parse

source = """
#[parallel]
fn scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
    for i in 0..n {
        out[i] = (inp[i] * factor) + 1.0;
    }
}
"""
namespace = get_backend("zig", trusted=True).compile(parse(lex(source)))

n = 10_000
inp = (ctypes.c_double * n)(*[float(i) for i in range(n)])
out = (ctypes.c_double * n)()

namespace["scale"](out, inp, 2.5, n)
assert out[10] == 26.0
```

Only the annotated function's direct range loop is parallelized. Nested loops
run serially. The backend does not add locks or reductions, so the loop body
must write disjoint elements or otherwise be race-free by construction.
Ordering between iterations is not guaranteed.

### SIMD vectors

`SimdN<T>` maps to Zig's `@Vector(N, T)`. The supported builtins are:

| Builtin | Meaning |
| ------- | ------- |
| `Simd::splat(x)` | Broadcast one scalar to every lane |
| `Simd::load(ptr, index, count)` | Load `count` lanes starting at `index` |
| `Simd::store(ptr, index, count, value)` | Store `count` lanes starting at `index` |
| `Simd::reduce_add(v)` | Horizontal sum |
| `Simd::reduce_mul(v)` | Horizontal product |
| `Simd::reduce_min(v)` | Horizontal minimum |
| `Simd::reduce_max(v)` | Horizontal maximum |

A four-lane dot product looks like this:

```polyglot
fn simd_dot4(a: &[f64], b: &[f64]) -> f64 {
    let va: Simd4<f64> = Simd::load(a, 0, 4);
    let vb: Simd4<f64> = Simd::load(b, 0, 4);
    Simd::reduce_add(va * vb)
}
```

The full runnable example is
[examples/zig_parallel_simd.poly](examples/zig_parallel_simd.poly). It combines
`#[parallel]`, `Simd4<f64>`, `splat`, `load`, `store`, `reduce_add`, and a
saturating `u8` return in `main()`.

### Build and cache

```sh
polyglot build sat.poly --backend zig
```

The CLI writes the shared library next to the source (as `sat.so` on macOS).
The backend caches generated Zig and the library under
`~/.cache/polyglot-zig`, keyed by a hash of the generated source. Clear it with:

```sh
rm -rf ~/.cache/polyglot-zig
```

The build artifact has C-ABI `__cabi_*` symbols and is intended to be loaded
with `ctypes`. It is not an importable Python extension module, so
`import sat` will not work.

### What the Zig backend does not support

The current lowering covers free functions, scalar parameters and returns,
`let`, assignment, `while`, direct range `for`, `return`, integer casts,
indexing, binary operations, saturating operations, arbitrary-width integers,
parallel range loops, and the SIMD builtins above. It does **not** lower
structs, enums, `if`, `match`, strings, `Vec`, `Option`, `Result`, or ordinary
user-function calls. Programs outside that subset are outside the Zig
contract; use the Python or native backend for them. See
[docs/zig.md](docs/zig.md) for the complete feature reference, SPEC 8.12 for
the normative rules, and SPEC 10.42 for the boundary entry.

## 10. Error codes

Polyglot uses stable error codes. When you see an error, look it up
in the [README's error code table](README.md#error-codes):

- `E001`: type mismatch (e.g. `usize` vs `i64`). Add an `as` cast.
- `E002`: a `#[terminates]` function could not be proven to terminate.
- `E003`: a translation error (unsupported construct).
- `E004`: the generated Rust failed to compile.
- `E005`: cyclic `include!`.
- `E006`: missing `include!` file.
- `E007`: unsupported native pattern (e.g. Box pattern).
- `E008`: invalid integer literal.

When available, a diagnostic includes a deterministic fix hint. For example,
an `E001` type mismatch adds an `as <type>` text edit. `polyglot exec
--error-format=json` writes the hint as a `fix` object (a message plus LSP-style
`edits`), and the language server sends the same payload in its
`textDocument/publishDiagnostics` notifications on open, change, and save.
Some diagnostics carry a message only, because the correct edit cannot be
derived from the diagnostic alone.

## 11. What to do when you get stuck

1. Read the error message. It usually tells you what to write.
2. Check the [SPECIFICATION](SPECIFICATION.md) for the rule.
3. Check [NOTES.md](NOTES.md) for the design decision behind the rule.
4. Run `polyglot check hello.poly` to check the program without
   running it.

## 12. Telemetry and verification

Inspect what a compile run did without changing your program:

```sh
polyglot telemetry          # phases, total time, monomorphization counts
polyglot timings            # alias for the timing section
polyglot cache              # hit rates, hits/misses by scope, rebuild reasons
polyglot hotpatch           # epoch, active capabilities, rollbacks, latest E008
polyglot telemetry --json   # machine-readable, path-redacted payload
```

The same data is available from the visualizer (`polyglot visualizer`) in the
collapsed **Telemetry** panel and from `GET /api/telemetry`. If nothing has run
yet in the current session, each section reports an explicit empty state.
Hot-patch entries report capability boundaries; they do not claim privileged
MAP_JIT, PEP 667, or raw-address patching on platforms where those paths are
modelled or refused.

Before publishing a change, run the unified gate:

```sh
python tools/release_gate.py --quick      # fast PR loop
python tools/release_gate.py --full       # extended fuzz seeds + full pytest
python tools/diff_fuzz.py --seed 42 --case-count 8 --json
```

Fuzz mismatches are written to `.artifacts/diff_fuzz/` with the exact seed and
source needed to reproduce them. Clean runs leave no files behind. Use
`--allow-dirty` on the gate only when you deliberately want to verify an
uncommitted worktree.

## 13. Next steps

- Read the [README](README.md) for the full feature list.
- Read the [SPECIFICATION](SPECIFICATION.md) if you want to know
  exactly what the language guarantees.
- Try the [examples](examples/) directory.

Note: `BTreeMap`, `HashSet`, and nested `HashMap` (depth 1) are available
in both backends for the type sets in SPEC 7.2.6–7.2.8. Deeper nesting and
unsupported key/value combinations are deferred. See SPEC 10.25 and 10.27.
