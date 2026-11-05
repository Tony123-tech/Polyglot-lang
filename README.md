# polyglot-lang

Embed Rust-like syntax in Python. `polyglot-lang` provides a shared lexer, parser, and AST, with backends that either transpile programs to Python or compile them into native Python extensions.

## Status

| Milestone | Status | Scope |
|---|---|---|
| N0 | Done | Empty program → Cargo → load module |
| N1 | Done | Free functions, `i64`/`f64`/`bool`/`String`, `Vec`, structs with `impl` exposed as PyO3 classes, and `if` |
| N2a | Done | Native `while`, `loop`, and `match` |
| N2b | Next | `Vec` methods: `push`, `pop`, `len`, and indexing |
| N2c | Planned | Multiple structs/enums as `pyclass` |
| N2d | Planned | Improved caching and Rust-error mapping to source lines |

**Current test status:** 22 tests passing.

## Features

- **Shared front end:** Turns source text into tokens, a parsed program, and AST nodes.
- **Python backend:** Translates the Rust-like AST to a Python AST, then compiles and executes it.
- **Native backend:** Generates Rust source with PyO3, builds it with Cargo, and loads the resulting `.so` or `.dylib` as a Python extension.
- **Common backend interface:** Select a backend with `get_backend("python")` or `get_backend("native")`. Calling `backend.compile(program)` returns a namespace dictionary.

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
namespace = get_backend("native").compile(program)
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
├── backends/
│   ├── base.py
│   ├── python/
│   │   ├── translator.py
│   │   └── runtime.py
│   └── rust/
│       ├── codegen.py
│       └── compiler.py
├── tests/
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

- **N2b:** Add native `Vec` methods `push`, `pop`, and `len`, plus indexing.
- **N2c:** Support multiple structs and enums exposed as PyO3 classes.
- **N2d:** Improve the native build cache and map Rust compiler errors back to source lines.

## Notes

See [NOTES.md](NOTES.md) for additional project notes.

## License

MIT License — placeholder. Add the project license text before distributing.