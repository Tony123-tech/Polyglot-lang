# Contributing to polyglot-lang

Thanks for your interest. This document explains how to set up
the project, run tests, and submit changes.

## Project structure

See [docs/architecture.md](docs/architecture.md) for the pipeline diagram
and layer responsibilities. In short:

- `compiler/core/` — `lexer.py`, `parser.py`, `ast_nodes.py`, the shared
  front end.
- `compiler/core/include_expander.py` — `include!` expansion and include
  diagnostics; `compiler/analysis/` holds `monomorph.py`, `termination.py`,
  `struct_order.py`, `match_diagnostics.py`.
- `backends/python/` — AST-to-Python translator and runtime.
- `backends/rust/` — Rust/PyO3 codegen, async reactor runtime, and Cargo
  build driver.
- `backends/wasm/` — experimental compute-only WebAssembly text backend.
- `zig/` — experimental Zig backend (`backend.py` codegen/build/load,
  `features.py` saturating arithmetic and bit-width bounds, `lowering.py`
  Python fallback).
- `api/cli.py` — the `polyglot` command-line interface.
- `api/lsp.py` — the stdio language server.
- `tests/` — pytest suite, including native and slow tests.
- `docs/` — architecture notes and the documentation index.
- `tools/` — developer checks, completions, and helper scripts.

## Development setup

    git clone https://github.com/Tony123-tech/Polyglot-lang.git
    cd Polyglot-lang
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

Python 3.14.8 or later is required. The native backend also requires a
Rust toolchain:

    rustc --version
    cargo --version

The experimental WASM backend needs the optional `wasm` extra:

    pip install -e ".[dev,wasm]"

The experimental Zig backend needs `zig` on `PATH`:

    zig version

Zig tests are skipped when the toolchain is absent. They are not required for
the Python/native parity suite.

## Running tests

    pytest -q
    pytest -q -n auto
    pytest -q -m slow
    pytest -q tests/test_m3_collections.py
    pytest -q tests/test_zig_backend.py

Or use the Makefile:

    make test          # pytest -q
    make test-slow     # pytest -q -m slow
    make lint          # ruff check .
    make format        # ruff format .
    make format-check  # ruff format --check .
    make docs-sync     # README/SPEC test-baseline agreement
    make check         # lint + format-check + docs-sync + fast tests

Native tests compile Rust; the first run is slow, subsequent runs hit
the cache under `~/.cache/polyglot-native`. Zig tests build a dynamic library
and cache it under `~/.cache/polyglot-zig`, keyed by a hash of the generated
source. To clear both:

    rm -rf ~/.cache/polyglot-native ~/.cache/polyglot-zig

Slow tests are deselected by default via `addopts = "-m 'not slow'"`.
Run them explicitly with `make test-slow` or `pytest -q -m slow`.
The full-suite baseline recorded in `README.md` and SPEC 12.1 is the
authoritative count; update both together when tests change.

## Code style

Before every commit, run:

    ruff format .
    ruff check --fix .
    ruff format --check .
    ruff check .

- 4-space indentation
- Type annotations on public functions
- Docstrings on public API (see backends/__init__.py)
- Comments and identifiers in English

## Adding a new error code

Error codes are stable and documented. To add one:

1. Define the exception, set its `code` class attribute (for example
   `code = "E009"`), and implement `format()` in the relevant module:
   - generic translation errors: `backends/errors.py`
   - Cargo/boundary errors: `backends/rust/compiler.py`
   - termination errors: `termination.py`
   - include errors: `include_expander.py`
2. Add a row to the error-code table in `README.md`.
3. Add or update the normative clause and the Section 10 entry in
   `SPECIFICATION.md`, and bump the SPEC version per Section 11.
4. Add a matching entry to `CHANGELOG.md`.
5. Add a test that asserts the code appears exactly once and that the
   span is printed when available (see `tests/test_m7_diagnostics.py`).
6. If the code can reach the CLI, verify the default text format and
   `--error-format=json` both emit the new code.

Current code ownership: `E001` in `backends/rust/compiler.py`, `E002` in
`termination.py`, `E003`/`E007`/`E008` in `backends/errors.py`, `E004`
in `backends/rust/compiler.py`, and `E005`/`E006` in
`include_expander.py`.

## Adding an example

1. Add a runnable `examples/<name>.poly` with a `main()` function.
2. If it should be part of the documented set, update the examples list
   in `README.md` and `tests/test_f1_cli.py`.
3. Run `polyglot check examples/<name>.poly` and
   `polyglot run examples/<name>.poly` (or add a test) before opening a
   pull request.

## Commit conventions

Follow [COMMIT_GUIDE.md](COMMIT_GUIDE.md). One commit is one logical
change, and files are staged explicitly by path (never `git add -A` or
`git add .`).

    <type>(<scope>): <short description>

Types: feat, fix, refactor, test, docs, build, chore, perf, ci.
Scopes: frontend, python, rust, api, cli, tests, docs, build, ci.

## Pull request process

1. Fork and branch from main.
2. Add tests for any behaviour change.
3. Run `make check`, `pytest -q -m slow`, and `git diff --check`.
4. If you change a SPEC clause or Section 8, bump the version in
   SPECIFICATION.md and add the matching CHANGELOG.md entry.
5. Update README/GUIDE when the user-facing CLI, support matrix, or
   test baseline changes.
6. Open a PR. Describe what changed and why.

## Updating documentation

- SPEC is normative. Any behaviour change MUST update the relevant
  clause and bump the version. Section 8 changes also require a
  Section 10 entry and a CHANGELOG.md entry.
- NOTES is a design journal. Add a Symptom/Cause/Fix/Lesson entry
  for each non-trivial bug or decision. Write the entry when the bug
  is found, not after the fix: the Symptom is the part that is easy to
  lose. Name the test that now covers it.
- Any bug fixed without a covering test must say so in the entry,
  otherwise the next refactor silently reintroduces it.
- README should reflect the SPEC support matrix, the CLI, and the
  current test baseline. `make docs-sync` and the CI docs-sync check
  enforce matching pytest baselines in README and SPEC 12.1.
- GUIDE examples must run with the real CLI. Spot-check changed examples
  with `polyglot check` and `polyglot run`.
- docs/overview.md and docs/architecture.md must stay in step with the
  shipped CLI and backend layout.

## Where to start

- Native collections parity (M3): backends/rust/codegen.py.
- Integer matrix parity (M4): tests/test_m4_integers.py.
- Diagnostics and spans (M7): parser.py, backends/errors.py.
- Async reactor and asyncio bridge (M5): the runtime in
  backends/rust/codegen.py.
- Language server: api/lsp.py.
- Experimental WASM subset: backends/wasm/codegen.py.
- Experimental Zig backend: zig/backend.py, zig/features.py, and
  zig/lowering.py (see docs/zig.md).
- Shell completions: tools/completions/.
