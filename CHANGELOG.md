# Changelog

Version history derived from SPECIFICATION.md Section 11.

## v0.3.4 — M3 remainder

- Specified `BTreeMap` and `HashSet` as collections-layer types and
  extended the `HashMap` entry API (`or_insert_with`, `and_modify`,
  `or_insert_with_key`).
- Implemented native `BTreeMap<K, V>` for `K` in `{i64, String}` and `V` in
  `{i64, String, bool, f64, Vec<i64>}`, with sorted iteration.
- Implemented native `HashSet<T>` for `T` in `{i64, String}` and nested
  `HashMap` values at depth 1.
- Added collection `get_mut` copy-at-Python-boundary semantics for
  `BTreeMap` and `HashSet`, with tests proving the input is not mutated.
- Added parity tests for sorted `BTreeMap` iteration and
  order-insensitive `HashSet` iteration.

## v0.3.3 — M4 remainder

- Added `i8`, `i16`, `i32`, `i128`, `u8`, `u16`, `u32`, `u64`, and `u128`
  to the type system alongside `i64`.
- Defined implicit-conversion boundary diagnostics across all integer
  types: `E001` with the six-line format when a span is available.
- Documented Python `as`-cast behavior: fixed-width targets wrap, while an
  out-of-range literal raises `OverflowError`; native literal casts follow
  Rust and reject out-of-range literals at compile time.
- Documented float-to-int, `char`, and `bool` cast differences between
  backends.

## v0.3.2 — M7 diagnostics

- Assigned stable codes `E001`–`E008` to boundary, termination,
  translation, Rust compile, include-cycle, include-not-found,
  unsupported-native-pattern, and invalid-literal errors.
- Added best-effort character spans to AST nodes and extended diagnostics
  to the six-line format when a span is available.
- Added `--error-format={text,json}` to the CLI; JSON payloads carry
  `code`, `message`, `line`, `col`, `span`, and `source`.

## v0.3.1 — M6 include directive

- Added the one-way `include!("path.poly");` directive.
- Includes resolve relative to the including file and may nest.
- Cyclic includes report the full chain (`E005`); missing targets report
  the resolved path and include site (`E006`).

## v0.3.0 — F1 delivery

- Added package metadata and the `polyglot` console script.
- Added `polyglot run`, `polyglot build`, and `polyglot check`.
- Added the Quick Start, the `examples/` directory, and delivery
  acceptance criteria.
- Recorded the pytest baseline in README and SPEC 12.1.
