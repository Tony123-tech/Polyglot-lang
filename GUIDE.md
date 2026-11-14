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

The Python backend has no native dependencies. You can run `.poly`
files immediately.

For the native backend, you also need a Rust toolchain:

```sh
rustc --version
cargo --version
```

If those commands work, you can compile `.poly` files to native
extensions.

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

`[1, 2, 3].to_vec()` creates a list. `xs.push(4)` appends. `xs.pop()`
removes the last element. `xs.len()` returns a `usize`, so you need
`as i64` to return it as an `i64`.

### Why `as i64`?

`len()` returns `usize` (a platform-sized unsigned integer).
Your `main()` returns `i64`. Polyglot does not convert between them
automatically:

```
error[E001]: type mismatch at line 6, column 5
  found:    usize
  expected: i64
  help: write `as i64` here
  note: `as` may wrap or truncate the value; check the range first
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

`"...".to_string()` creates a string. (`String::from("...")` is parsed but not yet lowered; see SPEC 10.28.)
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
value safely.

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
propagates an `Err` to the caller.

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

`if let` is shorter than `match` when you only care about one variant.

Note: `if let` works as the final expression of a block, but not inside a
`let` binding. Use `match` if you need an `if let` expression in the middle
of another expression.

## 8. When to use the native backend

The Python backend is fine for most programs. Use the native
backend when:

- You need **C-level performance** for a hot loop.
- You want to **ship a `.so`/`.dylib`** instead of Python source.
- You are already using PyO3 and want a simpler single-file alternative.

Build a native extension:

```sh
polyglot build hello.poly
```

This produces a `.dylib`/`.so` next to `hello.poly`. You can import
it from Python:

```python
import hello
print(hello.add(2, 3))
```

The first build takes about 2 minutes. Subsequent builds use a cache
and take about 3 seconds.

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

## 10. What to do when you get stuck

1. Read the error message. It usually tells you what to write.
2. Check the [SPECIFICATION](SPECIFICATION.md) for the rule.
3. Check [NOTES.md](NOTES.md) for the design decision behind the rule.
4. Run `polyglot check hello.poly` to check the program without
   running it.

## 11. Next steps

- Read the [README](README.md) for the full feature list.
- Read the [SPECIFICATION](SPECIFICATION.md) if you want to know
  exactly what the language guarantees.
- Try the [examples](examples/) directory.
