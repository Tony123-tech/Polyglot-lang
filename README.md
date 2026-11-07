# polyglot-lang

Embed Rust-like syntax in Python. `polyglot-lang` provides a shared lexer, parser, and AST, with backends that either transpile programs to Python or compile them into native Python extensions.

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

**Current test status:** 252 passed, 9 skipped, 4 deselected (`pytest -q`).

## Features

- **Shared front end:** Turns source text into tokens, a parsed program, and AST nodes.
- **Python backend:** Translates the Rust-like AST to a Python AST, then compiles and executes it.
- **Native backend:** Generates Rust source with PyO3, builds it with Cargo, and loads the resulting `.so` or `.dylib` as a Python extension.
- **Common backend interface:** Select a backend with `get_backend("python")` or `get_backend("native")`. Calling `backend.compile(program)` returns a namespace dictionary.
- **N5:** Option and Result construction and combinators, the `?` operator, `if let`, async functions, match exhaustiveness warnings, and Python sandboxing.

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

- Result `is_ok`, `is_err`, `map_err`, and `ok_or` are Python-only inside let-bindings.
- `while let Some(x) = iter.next()` is out of scope because Python has no Rust-style Option-returning `next()`.
- Box patterns, enum payload slices, and nested Vec slice patterns are Python-only.
- Async is not supported on impl methods.
- `PyFuture` is unsendable and can only be used on the thread that created it.
- Option `map` is intentionally unsupported because it collides with Vec `map` in codegen.

## Requirements

- Python 3.12 or later (tested on Python 3.14)
- Rust toolchain, including `cargo`, for the native backend
- `pytest` to run the test suite

Create and activate a virtual environment, then install the test runner:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pytest
```

The Python backend does not require the Rust toolchain. Native compilation requires a supported Rust toolchain and platform setup. The native compiler applies the required PyO3 and linker settings automatically.

## Quick start

The entry points are `lex`, `parse`, and `get_backend`. The compiler returns a dictionary containing the compiled program's names.

The Python backend needs no trust opt-in. The native backend requires
`trusted=True` or `POLYGLOT_TRUSTED=1`; see [Sandbox](#sandbox) for the
distinction between the trust gate and the sandbox.

### Add function

```python
from backends import get_backend
from lexer import lex
from parser import parse

source = """
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"""
program = parse(lex(source))
namespace = get_backend("python").compile(program)

assert namespace["add"](2, 3) == 5
```

### Point struct

```python
source = """
struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x: x, y: y }
    }

    fn sum(&self) -> i64 {
        self.x + self.y
    }
}
"""

program = parse(lex(source))
# Native compilation requires explicit trust opt-in.
# See the Sandbox section for why this is separate from sandboxing.
namespace = get_backend("native", trusted=True).compile(program)
point = namespace["Point"](3, 4)

assert point.sum() == 7
```

### Sum a vector

```python
from backends import get_backend
from lexer import lex
from parser import parse

source = """
fn sum(values: Vec<i64>) -> i64 {
    let mut total: i64 = 0;

    for value in values {
        total = total + value;
    }

    total
}
"""

program = parse(lex(source))
namespace = get_backend("python").compile(program)

assert namespace["sum"]([1, 2, 3]) == 6
```

Choose `"python"` or `"native"` according to the program features and whether native compilation is desired. Native `format!` currently requires its first argument to be a string literal; code generation handles that form specially.

## Project layout

```text
polyglot-lang/
├── ast_nodes.py
├── lexer.py
├── parser.py
├── conftest.py
├── main.py
├── match_diagnostics.py
├── struct_order.py
├── _native_sources.py
├── api/
│   ├── cli.py
│   ├── config.py
│   ├── executor.py
│   ├── models.py
│   ├── server.py
│   └── worker.py
├── backends/
│   ├── base.py
│   ├── python/
│   │   ├── translator.py
│   │   └── runtime.py
│   └── rust/
│       ├── codegen.py
│       └── compiler.py
├── tools/
│   └── compile_native.py
├── tests/
│   └── test_native_*.py
├── NOTES.md
└── README.md
```

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

N0 through N5 are complete. Remaining known limitations are listed under
[Known limitations](#known-limitations), including the out-of-scope items
`while let Some(x) = iter.next()`, async impl methods, and cross-thread
`PyFuture` use.

## Notes

See [NOTES.md](NOTES.md) for design decisions, per-batch history, and the
lessons recorded for each fixed bug.

## License

MIT License — see [LICENSE](LICENSE) for the full text.
