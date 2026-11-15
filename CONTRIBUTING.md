# Contributing to polyglot-lang

Thanks for your interest. This document explains how to set up
the project, run tests, and submit changes.

## Project structure

See the README for the full file layout. In short:

- `lexer.py`, `parser.py`, `ast_nodes.py` — shared front end.
- `backends/python/` — AST-to-Python translator and runtime.
- `backends/rust/` — Rust/PyO3 codegen and Cargo build driver.
- `api/cli.py` — the `polyglot` command-line interface.
- `tests/` — pytest suite, including native and slow tests.
- `tools/` — developer checks and helper scripts.

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

## Running tests

    pytest -q
    pytest -q -n auto
    pytest -q -m slow
    pytest -q tests/test_m3_collections.py

Or use the Makefile:

    make test          # pytest -q
    make test-slow     # pytest -q -m slow
    make lint          # ruff check .
    make format        # ruff format .
    make check         # lint + format-check + docs-sync + fast tests

Native tests compile Rust; the first run is slow, subsequent runs hit
the cache under `~/.cache/polyglot-native`. To clear it:

    rm -rf ~/.cache/polyglot-native

Slow tests are deselected by default via `addopts = "-m 'not slow'"`.
Run them explicitly with `make test-slow` or `pytest -q -m slow`.

## Code style

    ruff check --fix .
    ruff check .
    ruff format .

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
4. Add a test that asserts the code appears exactly once and that the
   span is printed when available (see `tests/test_m7_diagnostics.py`).
5. If the code can reach the CLI, verify the default text format and
   `--error-format=json` both emit the new code.

## Adding an example

1. Add a runnable `examples/<name>.poly` with a `main()` function.
2. If it should be part of the documented set, update the examples list
   in `README.md` and `tests/test_f1_cli.py`.
3. Run `polyglot check examples/<name>.poly` and
   `polyglot run examples/<name>.poly` (or add a test) before opening a
   pull request.

## Commit conventions

    <type>(<scope>): <short description>

Types: feat, fix, docs, test, refactor, perf, chore
Scopes: m3, m4, m6, m7, cli, api, spec, readme

## Pull request process

1. Fork and branch from main.
2. Add tests for any behaviour change.
3. Run `make check`, `pytest -q -m slow`, and `git diff --check`.
4. If you change a SPEC clause, bump the version in SPECIFICATION.md.
5. Open a PR. Describe what changed and why.

## Updating SPECIFICATION.md and NOTES.md

- SPEC is normative. Any behaviour change MUST update the relevant
  clause and bump the version.
- NOTES is a design journal. Add a Symptom/Cause/Fix/Lesson entry
  for each non-trivial bug or decision.
- README should reflect the SPEC support matrix. The `Check docs sync`
  CI step enforces matching pytest baselines in README and SPEC 12.1.

## Where to start

- Native collections parity (M3): backends/rust/codegen.py.
- Integer matrix parity (M4): tests/test_m4_integers.py.
- Diagnostics and spans (M7): parser.py, backends/errors.py.
- Async executor (M5): _RUNTIME_PREFIX.
- Shell completions: tools/completions/.
- VS Code extension: syntax highlighting and completion for .poly files.
