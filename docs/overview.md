# polyglot-lang documentation

One-page orientation for the repository.

## What it is

`polyglot-lang` is a small Rust-like language with a shared lexer, parser,
and AST. A `.poly` file can run on a stdlib-only Python backend or compile
to a native PyO3 extension. Both backends are selected through the same
backend interface and the same CLI.

## Where to start

| Document | Use it for |
| -------- | ---------- |
| [README](../README.md) | Install, Quick Start, CLI, features, support matrix, error codes |
| [GUIDE](../GUIDE.md) | Step-by-step tutorial for Python developers |
| [SPECIFICATION](../SPECIFICATION.md) | Normative language rules, platform differences, conformance status |
| [NOTES](../NOTES.md) | Design journal: symptoms, causes, fixes, lessons, test baselines |
| [CONTRIBUTING](../CONTRIBUTING.md) | Development setup, tests, Makefile, adding error codes/examples |
| [architecture.md](architecture.md) | ASCII pipeline diagram and layer responsibilities |
| [CHANGELOG](../CHANGELOG.md) | Version history from v0.3.0 through v0.3.4 |

## The two backends

- **Python** — `polyglot run file.poly` or `polyglot run file.poly --backend python`.
  No Rust toolchain. Translates the AST to Python and executes it in an
  isolated namespace.
- **Native** — `polyglot build file.poly` or `polyglot run file.poly --backend native`.
  Generates Rust with PyO3, builds with Cargo, and loads a `.so`/`.pyd`.
  Builds are cached under `~/.cache/polyglot-native`.

## Key boundaries

- `usize`/`isize` are distinct from `i64`; every crossing needs `as`.
- Fixed-width integer casts follow Rust semantics; literal out-of-range
  casts raise `OverflowError` in the Python backend and fail native Rust
  compilation.
- Collections: `Vec`, `String`, `Option`, `Result` (core), `HashMap`,
  `BTreeMap`, `HashSet` (collections). Nested `HashMap` values are depth 1.
- `include!` is a one-way textual splice resolved relative to the
  including file; cycles are rejected with `E005`.
- Diagnostics use stable codes `E001`–`E008`, five lines by default and
  six when a source span is available. `--error-format=json` emits a
  machine-readable object.

## Verifying the repository

```sh
make check          # lint + format-check + docs-sync + fast tests
make test-slow      # native/slow suite
python3 tools/check_docs_sync.py
```

CI runs the same checks, including `ruff format --check .` and a check
that the README and SPEC 12.1 pytest baselines agree.
