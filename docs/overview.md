# polyglot-lang documentation

One-page orientation for the repository.

## What it is

`polyglot-lang` is a small Rust-like language with a shared lexer, parser,
and AST. A `.poly` file can run on a stdlib-only Python backend or compile
to a native PyO3 extension. An experimental compute-only backend emits
WebAssembly text format (`.wat`). All backends are selected through the
same interface and CLI.

Python 3.14.8 or later is required.

## Where to start

| Document | Use it for |
| -------- | ---------- |
| [README](../README.md) | Install, Quick Start, CLI, features, support matrix, error codes |
| [GUIDE](../GUIDE.md) | Step-by-step tutorial for Python developers |
| [SPECIFICATION](../SPECIFICATION.md) | Normative language rules, platform differences, conformance status |
| [NOTES](../NOTES.md) | Design journal: symptoms, causes, fixes, lessons, test baselines |
| [CONTRIBUTING](../CONTRIBUTING.md) | Development setup, tests, Makefile, adding error codes/examples |
| [architecture.md](architecture.md) | ASCII pipeline diagram and layer responsibilities |
| [visualizer.md](visualizer.md) | Interactive AST, monomorphization, diagnostics, and dual-backend UI |
| [experimental_pillars.md](experimental_pillars.md) | Opt-in hot-patching, meta-compiler, synthesis, time-travel, and transmutation foundations |
| [distributed.md](distributed.md) | Opt-in distributed actors, live migration, anchored connections, leases, and failure recovery |
| [unified_pillars.md](unified_pillars.md) | 10-pillar / 100-function capability registry, dual-backend parity rules, and the PyO3 binding scaffold |
| [CHANGELOG](../CHANGELOG.md) | Version history from v0.3.0 through the current unreleased work |

## The backends

- **Python** — `polyglot run file.poly` or `polyglot run file.poly --backend python`.
  No Rust toolchain. Translates the AST to Python and executes it in an
  isolated namespace.
- **Native** — `polyglot build file.poly` or `polyglot run file.poly --backend native`.
  Generates Rust with PyO3, builds with Cargo, and loads a `.so`/`.pyd`.
  Async futures are driven by a single-threaded reactor and integrate
  with Python `asyncio`. Builds are cached under
  `~/.cache/polyglot-native`.
- **WASM (experimental)** — `polyglot build file.poly --backend wasm`
  writes a `.wat` module for a compute-only `i64`/`bool` subset. It
  requires the optional `wasm` extra (`wasmtime`).

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
  machine-readable object. E003 is Python-only; the native backend uses
  E004 or E007 instead.
- Async is experimental. Native futures use a real waker/reactor,
  integrate with `asyncio`, and support timer wakeups. Async `&mut self`
  methods are rejected.

## Verifying the repository

```sh
make check          # lint + format-check + docs-sync + fast tests
make test-slow      # native/slow suite
python3 tools/check_docs_sync.py
```

CI runs the same checks, including `ruff format --check .` and a check
that the README and SPEC 12.1 pytest baselines agree.
