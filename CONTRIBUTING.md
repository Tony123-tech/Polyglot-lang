# Contributing to polyglot-lang

Thanks for your interest. This document explains how to set up
the project, run tests, and submit changes.

## Project structure

See the README for the full file layout.

## Development setup

    git clone https://github.com/Tony123-tech/Polyglot-lang.git
    cd Polyglot-lang
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

The native backend requires a Rust toolchain:

    rustc --version
    cargo --version

## Running tests

    pytest -q
    pytest -q -n auto
    pytest -q -m slow
    pytest -q tests/test_m3_collections.py

Native tests compile Rust; first run is slow, subsequent runs
hit the cache. To clear the cache:

    rm -rf ~/.cache/polyglot-native

## Code style

    ruff check --fix .
    ruff check .

- 4-space indentation
- Type annotations on public functions
- Docstrings on public API (see backends/__init__.py)
- Comments and identifiers in English

## Commit conventions

    <type>(<scope>): <short description>

Types: feat, fix, docs, test, refactor, perf, chore
Scopes: m3, m4, m6, m7, cli, api, spec, readme

## Pull request process

1. Fork and branch from main.
2. Add tests for any behaviour change.
3. Run pytest -q, ruff check ., git diff --check.
4. If you change a SPEC clause, bump the version in SPECIFICATION.md.
5. Open a PR. Describe what changed and why.

## Updating SPECIFICATION.md and NOTES.md

- SPEC is normative. Any behaviour change MUST update the relevant
  clause and bump the version.
- NOTES is a design journal. Add a Symptom/Cause/Fix/Lesson entry
  for each non-trivial bug or decision.
- README should reflect the SPEC support matrix.

## Where to start

- Native BTreeMap/HashSet (M3): backends/rust/codegen.py.
- Async executor (M5): _RUNTIME_PREFIX.
- Shell completions: tools/completions/.
- VS Code extension: syntax highlighting and completion for .poly files.
