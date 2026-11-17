# Architecture overview

`polyglot-lang` compiles one `.poly` source file with two primary
interchangeable backends (Python and native PyO3), plus an experimental
compute-only WASM text backend. The front end is shared; only the
lowering and execution stages differ.

```text
                    .poly source
                         |
                         v
                 +---------------+
                 |    lexer.py   |  tokens
                 +---------------+
                         |
                         v
                 +---------------+
                 |   parser.py   |  AST (ast_nodes.py) + spans
                 +---------------+
                         |
                         v
              +---------------------+
              | include_expander.py |  splices include!("...") items
              +---------------------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
 +---------------------+      +----------------------+
 | backend: python     |      | backend: rust        |
 | translator.py       |      | codegen.py           |
 | (AST -> Python AST) |      | (AST -> Rust/PyO3)   |
 +---------------------+      +----------------------+
          |                             |
          v                             v
 +---------------------+      +----------------------+
 | compile()/exec()    |      | cargo build --release|
 | runtime.py helpers  |      | load .so / .pyd      |
 +---------------------+      +----------------------+
          |                             |
          +--------------+--------------+
                         |
                         v
              namespace dict / CLI result
```

The optional experimental WASM backend consumes the same AST and writes
WebAssembly text format (`.wat`) for an `i64`/`bool` compute-only subset.

## Layers

- **Front end:** `lexer.py`, `parser.py`, `ast_nodes.py`. Parser nodes carry
  best-effort character spans used by diagnostics.
- **Include expansion:** `include_expander.py` resolves `include!` paths
  relative to the including file, rejects cycles, and splices items before
  backend selection.
- **Python backend:** `backends/python/translator.py` converts the AST to a
  Python AST; `backends/python/runtime.py` holds stdlib-only helpers for
  casts, `Option`/`Result`, collections, and boundary checks.
- **Native backend:** `backends/rust/codegen.py` emits Rust with PyO3 and
  the async runtime (a single-threaded `mio` reactor, task wakers, timers,
  Unix fd readiness, and the `asyncio` bridge); `backends/rust/compiler.py`
  drives Cargo and loads the extension. Builds are cached under
  `~/.cache/polyglot-native` by a hash of generated Rust. Pass
  `--incremental` to keep per-function fragments in a stable scope
  directory.
- **WASM backend (experimental):** `backends/wasm/codegen.py` lowers the
  `i64`/`bool` subset to `.wat`; `wasmtime` is an optional dependency.
- **CLI/API:** `api/cli.py` implements `run`, `build`, `check`, `compile`,
  and `lsp`; `api/` also contains the optional HTTP worker/server.
- **Tooling:** `tools/` contains documentation/structure lints, a native
  compile helper, shell completions, and the native build benchmark.

## Design constraints

- One grammar and one AST for all backends.
- The Python backend is stdlib-only.
- Platform differences are explicit, documented in SPEC Section 8, and
  mirrored by Section 10 entries.
- Language semantics are not changed by a single backend; differences are
  recorded rather than silently accepted.

See [overview.md](overview.md), [../README.md](../README.md),
[../GUIDE.md](../GUIDE.md), [../SPECIFICATION.md](../SPECIFICATION.md), and
[../NOTES.md](../NOTES.md).
