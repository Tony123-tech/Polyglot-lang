# Polyglot Language Guide

This guide is for Python developers who have never written Rust.
It starts from zero and ends with a working program that uses
Polyglot's type system, collections, and error handling.

If you already know Rust, skip to the [README](README.md) or
[SPECIFICATION](SPECIFICATION.md).

This guide covers the Python and native backends. Experimental execution
modes (the Zig backend, arbitrary bit-width integers, `#[parallel]` loops,
SIMD vectors) and the pure-Python Gleam/Elixir/Mojo models are documented in
[docs/advanced.md](docs/advanced.md).

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

The quickest start is a generated project:

```sh
polyglot new myapp
cd myapp
polyglot run main.poly
```

Or create the file by hand. Create `hello.poly`:

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

### Moving, borrowing, and dropping

Owned values behave like Rust values: passing a `String` or a `Vec` by value
*moves* it, and the old name stops working.

```polyglot
fn take(s: String) -> i64 { s.len() as i64 }

fn main() -> i64 {
    let text: String = "hi";
    take(text);
    println!("{}", text);   // error[E009]: use of moved value: `text`
    0
}
```

Two other names for the same thing:

- `&text` borrows it instead of moving it. You cannot move or `drop` a value
  while a borrow is alive, and you cannot assign to it either.
- `drop(text)` releases it early. Reading it afterwards is an error, and
  dropping it twice is an error.

Copying is free for `i64`, `f64`, `bool`, `char`, and references, so none of
this applies to them. All of it is checked before either backend generates
code, so the Python and native backends reject the same programs; the Python
backend simply ignores `drop`, because Python already owns the object. See
SPEC 8.16 for the rule and SPEC 10.55 for the limits of the model.

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

Experimental modes (saturating arithmetic, arbitrary bit-width integers,
`#[parallel]` loops, SIMD vectors) and the pure-Python Gleam/Elixir/Mojo
models are documented in [docs/advanced.md](docs/advanced.md).

## 9. Error codes

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
- `E009`: ownership — a value was used after it was moved, borrowed while being
  moved or dropped, or dropped twice. See "Moving, borrowing, and dropping" in
  section 3.

When available, a diagnostic includes a deterministic fix hint. For example,
an `E001` type mismatch adds an `as <type>` text edit. `polyglot exec
--error-format=json` writes the hint as a `fix` object (a message plus LSP-style
`edits`), and the language server sends the same payload in its
`textDocument/publishDiagnostics` notifications on open, change, and save.
Some diagnostics carry a message only, because the correct edit cannot be
derived from the diagnostic alone.

### Runtime failures

A guard that fails while the program runs uses the same five-line format and
the same code, `E002`. The message says what failed, the note says it failed
while running, and the location line points at the expression that failed:

```text
$ polyglot run div.poly
error[E002]: division by zero
  reason: division by zero
  help: check the divisor is non-zero before dividing
  note: runtime failure
  at: line 4, column 20
```

The guarded operations are integer `/` and `%` with a zero divisor,
out-of-bounds indexing of a `Vec`, array, or slice, `panic!`, `assert!`,
`assert_eq!`, and `assert_ne!`. Float division by zero is not an error: `x / 0`
is infinity and `0 / 0` is NaN, matching Rust. See SPEC 2.9.

## 10. What to do when you get stuck

1. Read the error message. It usually tells you what to write.
2. Run `polyglot explain E001` (or any code from `E001`-`E008`) to see the
   wrong and the right version side by side.
3. Check the [SPECIFICATION](SPECIFICATION.md) for the rule.
4. Check [NOTES.md](NOTES.md) for the design decision behind the rule.
5. Run `polyglot check hello.poly` to check the program without
   running it.

## 11. Common questions

**Why does `main()` return a number instead of printing?**

`polyglot run` prints the return value of `main()`, so the output of a program
is whatever `main()` returns. If you want different output, return that value.

**Why is `as i64` needed after `len()`?**

`len()` returns `usize` and `main()` returns `i64`. Polyglot never converts
between integer types implicitly. The `E001` diagnostic already tells you the
exact cast to add.

**Why can't I change a variable?**

Bindings are immutable unless declared with `let mut`. The compiler enforces
this, so an accidental reassignment is caught before the program runs.

**Which backend should I use?**

Start with the Python backend: no toolchain, fastest edit-run loop. Move to the
native backend only when a hot loop needs C-level speed, or when you want to
ship a `.so`/`.pyd` instead of Python source.

**Why does `include!` fail inside a function?**

`include!` splices items and is only allowed at item position. Move it to the
top of the file, outside any function body.

**Does `include!` de-duplicate files?**

No. Including the same file twice splices its items twice, which usually ends in
duplicate definitions.

## 12. Next steps

- Read the [README](README.md) for the full feature list.
- Read the [SPECIFICATION](SPECIFICATION.md) if you want to know
  exactly what the language guarantees.
- See [docs/native_distribution.md](docs/native_distribution.md) for how native
  wheels are built and signed in CI, and
  [docs/third_party_audit.md](docs/third_party_audit.md) for read-only third-party
  audit scope decisions.
- Try the [examples](examples/) directory.
- Read [docs/advanced.md](docs/advanced.md) for the Zig backend, SIMD, and
  maintainer tooling.

Note: `BTreeMap`, `HashSet`, and nested `HashMap` (depth 1) are available
in both backends for the type sets in SPEC 7.2.6–7.2.8. Deeper nesting and
unsupported key/value combinations are deferred. See SPEC 10.25 and 10.27.
