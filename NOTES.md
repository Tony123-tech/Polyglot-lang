# NOTES.md

Rust-like syntax is embedded in Python through the shared lexer, parser, and AST provided by `polyglot-lang`. Programs can be translated from the Rust AST to Python AST and executed with `compile()` and `exec()`, or emitted as Rust source with PyO3, built with Cargo, and loaded as native Python extensions.

## N1 complete

Support for free functions, primitive types, vectors, structs with `impl`, and conditional expressions and statements was added in N1.

### Bug 1

**Symptom:** Native code generation failed for `format!` calls whose first argument was not emitted as a bare string literal, such as a `String::from(...)` expression.

**Cause:** The format-string handling path expected a literal and did not distinguish that required form from other expressions.

**Fix:** The `_format_str` helper was added to handle the supported literal format-string form explicitly.

**Lesson:** Special syntax should be validated and handled at the boundary where its constraints are known.

### Bug 2

**Symptom:** An expression such as `xs { total = ... }` could be parsed as a struct literal.

**Cause:** `_looks_like_struct_literal` inspected too few tokens after the path and treated a following block as a struct literal without confirming its contents.

**Fix:** Struct-literal recognition now requires the next token to be `:`, `,`, `}`, `..`, or empty.

**Lesson:** Ambiguous syntax should be disambiguated using the minimum structural evidence required by the grammar.

### Status after N1

- Free functions and `i64`, `f64`, `bool`, and `String` support completed.
- `Vec<T>`, structs with `impl`, and `if`/`else` support completed.
- **13 tests passed.**

## N2a complete

Native support for `while`, `loop`, and `match` was added.

### Bug 1

**Symptom:** Parameters intended to be mutable inside generated Rust functions were not mutable.

**Cause:** Rust parameters are immutable by default, while the generated function signature did not account for the parameter pattern's mutability.

**Fix:** `param_to_rust` now emits `mut` unless the pattern is already `mut`.

**Lesson:** Generated code must preserve the source language's mutability intent while also satisfying the target language's defaults.

### Bug 2

**Symptom:** Vector length and indexing expressions could use incompatible integer types in generated Rust.

**Cause:** `Vec::len()` returns `usize`, while the generated expression used `i64` for the length and index.

**Fix:** Generated length expressions use `(recv.len() as i64)`, and generated indexes use `recv[(idx) as usize]`.

**Lesson:** Target-language type differences should be handled explicitly at code-generation boundaries.

### Bug 3

**Symptom:** Block-bodied `match` arms without a trailing comma were rejected.

**Cause:** The parser required a comma after every arm, although Rust permits omitting it after a block body.

**Fix:** `parse_match` now accepts a missing comma when an arm has a block body.

**Lesson:** Parsers should model grammar variations according to syntactic context rather than applying one separator rule universally.

### Bug 4

**Symptom:** Parsing a `match` expression did not consume its opening brace.

**Cause:** The opening brace was not explicitly consumed by `parse_match`.

**Fix:** `self.expect(TokenType.LBRACE)` was added at the start of match-body parsing.

**Lesson:** Delimiters should be consumed and validated at the grammar production that owns them.

### Bug 5

**Symptom:** A block-bodied `match` arm could not be used where an expression was expected.

**Cause:** The generated arm body was not represented as a Rust block expression.

**Fix:** Block-bodied arms are now wrapped in `{ ... }`.

**Lesson:** Code generation should preserve the syntactic category required by the surrounding expression.

### Status after N2a

- Native `while`, `loop`, and `match` support completed.
- Mutability, integer casts, match separators, delimiters, and block-expression handling corrected.
- **22 tests passed.**

## N2b complete

Native vector methods `push`, `pop`, and `len`, along with indexing, were added.

### Bug 1

**Symptom:** Generated Rust bindings could trigger unused-mutability warnings for bindings that were not reassigned.

**Cause:** Mutability requirements were not inferred per binding, while mutable bindings were needed for supported assignment behavior.

**Fix:** `generate_stmt` emits `let mut` for every binding, and the generated crate includes `#![allow(unused_mut, ...)]`.

**Lesson:** When exact analysis is not available, a consistent conservative code-generation policy can be used if the resulting diagnostics are explicitly controlled.

### Bug 2

**Symptom:** Vector `pop` results and `Some(x)` / `None` patterns needed special handling in generated native code.

**Cause:** `Vec::pop()` returns `Option<T>`, and pattern emission could otherwise transform those variant patterns incorrectly.

**Fix:** PyO3's existing `Option<T>` support is used, and `pattern_to_rust` emits `Some(x)` and `None` verbatim.

**Lesson:** Existing target-language and binding-library support should be reused instead of duplicating representation logic.

### Status after N2b

- Native `Vec` methods `push`, `pop`, and `len`, plus indexing, completed.
- Mutable bindings and `Option<T>` pattern handling corrected.
- **30 tests passed.**

## N2d complete

Native test compilation was consolidated, build caching was improved, and Rust compiler errors were given a structured representation. All native test sources are collected from `SOURCES` dictionaries in `tests/test_native_*.py`, merged, and compiled once per pytest session. `conftest.py` invokes `compile_sources(sources)` from `pytest_collection_modifyitems`; `tools/compile_native.py` uses the same flow for CI pre-builds.

A SHA-256 digest of the merged source is used as the cache key. The shared cache path is `~/.cache/polyglot-native/<hash>/`. The existing `target/` directory is retained for incremental builds; only `Cargo.toml`, `.cargo/config.toml`, and `src/lib.rs` are rewritten. Native test build time was reduced from approximately 214 seconds to 0.15 seconds on a cache hit.

### Bug 1

**Symptom:** Loading the built macOS dynamic library under a filename-derived module name caused Python to look for `PyInit_libpolyglot_native` rather than the generated `PyInit_polyglot_native` symbol.

**Cause:** The dynamic-library filename prefix and the PyO3 module initialization name differ.

**Fix:** The library is loaded with `ExtensionFileLoader(mod_name="polyglot_native")`.

**Lesson:** Extension loaders and generated module symbols must agree on one canonical module name, independent of platform-specific filenames.

### Bug 2

**Symptom:** Classes created by the Python backend could fail during `@dataclass` processing because their module could not be resolved.

**Cause:** The generated namespace's module name had not been registered in `sys.modules` before dataclass processing.

**Fix:** `Translator.compile_rust` registers the namespace as `sys.modules["polyglot"]`.

**Lesson:** Dynamically generated objects may depend on normal module-registration conventions used by runtime libraries.

### Bug 3

**Symptom:** Native compiler failures were difficult to connect to structured diagnostics.

**Cause:** Cargo's standard error was exposed without parsing its available source location and error-code fields.

**Fix:** `RustCompileError` parses `error_line`, `error_col`, and `error_code`, and provides `pretty()`.

**Lesson:** Errors from external toolchains should be normalized at the integration boundary so callers can consume consistent diagnostics.

### Bug 4

**Symptom:** Native compilation could be invoked without an explicit trust decision.

**Cause:** Backend selection did not enforce the project's default-deny trust policy.

**Fix:** `UntrustedCompileError` and `Backend._check_trust()` enforce the trust gate. `get_backend(name)` reads `POLYGLOT_TRUSTED`, which defaults to `"0"`; `trusted=False` is the default, and tests opt in with `trusted=True`.

**Lesson:** Risky execution paths should require explicit opt-in, while keeping the scope of that gate clearly documented.

### Bug 5

**Symptom:** Methods without a `self` parameter, such as `fn new(...) -> Point`, could not be called as associated constructors through the generated Python class.

**Cause:** Methods without a receiver were not emitted as static methods, and `Path` expressions were not translated into attribute chains.

**Fix:** Methods without `self` are emitted as `@staticmethod`; `Point::new(x, y)` is translated to `Point.new(x, y)` through `ast.Attribute` chains in `translate_expr`.

**Lesson:** Cross-language APIs should preserve receiver and qualification semantics explicitly in both code generation and expression translation.

### Status after N2d

- Native test sources are merged and compiled once per pytest session, with the same build flow available for CI pre-builds.
- Content-addressed caching retains Cargo's `target/` artifacts for incremental builds.
- Structured Rust compiler errors and readable formatting are available.
- Dynamic-library loading, dataclass module registration, trust checking, and associated-method translation corrected.
- **30 tests passed; cache-hit build time was approximately 0.15 seconds, down from approximately 214 seconds.**

## N2c backlog

- Support multiple structs and enums exposed as `#[pyclass]` classes.

## N2d backlog (remaining)

- Map generated Rust source line numbers back to original Polyglot source lines.

## Security

`trusted=False` is not a sandbox; it is a speed bump. Operating-system-level isolation, such as a container, seccomp, and disabled network access, is required for untrusted input. The native cache directory is created with mode `0o700`. `pyo3 = "0.26"` is pinned in `Cargo.toml`.

Three isolation levels are provided through `api/`:

- **A:** In-process queue.
- **B:** Subprocess.
- **C:** Docker with `--network=none --read-only`.

HTTP requests are handled with isolation level B by default. An `X-Trusted-Token` must be provided when `trusted=true` is set.

## Roadmap

| Milestone | Status | Scope |
|---|---|---|
| N3 | Planned | Native `HashMap`, `Option`, and `Result` support |
| N4 | Planned | Native `async` / `await` support |
| N5 | Planned | Map generated `.rs` line numbers back to source lines |

## N2a pattern expansion — batch 1

The existing pattern AST nodes were wired through parsing, native Rust code generation, and Python match translation. Batch 1 covers twenty direct match-pattern forms:

1. Or-patterns.
2. Exclusive literal ranges.
3. Inclusive literal ranges.
4. Tuple patterns.
5. Nested tuple patterns.
6. Unit tuple patterns.
7. Single-element tuple patterns.
8. Struct patterns with explicit fields.
9. Struct patterns with shorthand fields.
10. Struct patterns with a rest pattern.
11. `@` binding patterns.
12. Ranges without a lower bound.
13. Ranges without an upper bound.
14. Negative literal ranges.
15. Or-patterns containing ranges.
16. Tuple patterns with identifier bindings.
17. Nested tuple binding patterns.
18. Struct field patterns with nested bindings.
19. Reference patterns with literal subpatterns.
20. Reference patterns with identifier bindings.

Single-element tuple types now accept a trailing comma and are emitted with the required Rust tuple comma. Negative literal bounds are emitted without parentheses, as required by Rust range-pattern syntax. Python match translation now handles tuple and struct tests, range and or-pattern tests, references, bindings, and bound-variable assignments.

The native test cases are stored in `tests/test_native_n2a_patterns.py` under `SOURCES` and compiled together with the existing native test sources.

### Tests in batch 1

- Feature 1: `test_pattern_or`
- Feature 2: `test_pattern_range_exclusive`
- Feature 3: `test_pattern_range_inclusive`
- Feature 4: `test_pattern_tuple`
- Feature 5: `test_pattern_nested_tuple`
- Feature 6: `test_pattern_unit_tuple`
- Feature 7: `test_pattern_single_tuple`
- Feature 8: `test_pattern_struct_explicit`
- Feature 9: `test_pattern_struct_shorthand`
- Feature 10: `test_pattern_struct_rest`
- Feature 11: `test_pattern_binding`
- Feature 12: `test_pattern_range_open_start`
- Feature 13: `test_pattern_range_open_end`
- Feature 14: `test_pattern_range_negative`
- Feature 15: `test_pattern_or_range`
- Feature 16: `test_pattern_tuple_binding`
- Feature 17: `test_pattern_nested_binding`
- Feature 18: `test_pattern_struct_nested`
- Feature 19: `test_pattern_ref_literal`
- Feature 20: `test_pattern_ref_binding`
- Cross-backend parity: `test_python_backend_pattern_parity`

## N2a pattern batch 2

`PatSlice` and `PatRest` were added to the AST and wired through the parser, Rust code generator, and Python match translator. Slice patterns support fixed-length matching, a single rest marker in any position, and `name @ ..` bindings. Native matching converts a vector scrutinee to a slice, while Python matching validates the sequence type and length before testing elements. A single-element parenthesized pattern is now treated as grouping unless a trailing comma makes it a tuple.

Native support is validated for direct vector slices, slices nested in tuples, range/ref/or patterns within slices, guards, and rest bindings. Struct-field and nested-slice combinations are covered by Python-backend tests but remain unsupported by native codegen where they would require matching a `Vec` as a slice within a Rust pattern. A tuple-style enum payload slice is skipped until payload-carrying enum values are represented by the Python runtime.

### Tests in batch 2

- Feature 21: `test_pattern_slice_fixed`
- Feature 22: `test_pattern_slice_rest_middle`
- Feature 23: `test_pattern_slice_rest_end`
- Feature 24: `test_pattern_slice_rest_start`
- Feature 25: `test_pattern_slice_all_rest`
- Feature 26: `test_pattern_slice_empty`
- Feature 27: `test_pattern_slice_literals`
- Feature 28: `test_pattern_slice_bindings`
- Feature 29: `test_pattern_slice_tuple`
- Feature 30: `test_pattern_slice_or`
- Feature 31: `test_pattern_slice_range`
- Feature 32: `test_pattern_slice_ref`
- Feature 33: `test_pattern_slice_at`
- Feature 34: `test_pattern_slice_guard`
- Feature 35: `test_pattern_slice_tuple_context`
- Feature 36: `test_pattern_slice_struct_context`
- Feature 37: `test_pattern_slice_enum_context` (skipped)
- Feature 38: `test_pattern_slice_rest_binding`
- Feature 39: `test_pattern_slice_only_rest_binding`
- Feature 40: `test_pattern_slice_nested`
- Cross-backend parity: `test_python_backend_pattern_parity`

Batch 2 validation: `pytest -v` completed with 70 passed and 1 skipped in 11.63 seconds; `ruff check --fix .` passed.

## N2a pattern batch 3

`PatBox`, `PatConst`, `PatMacro`, and `PatPath` were added to the pattern AST and connected to parser, native code generation, and Python translation. Box patterns are treated as transparent patterns by Python; native box-pattern tests are skipped because stable Rust does not support this syntax. Explicit `const NAME` patterns lower to the corresponding Rust constant path. Qualified paths are distinguished from identifier bindings, and Python path patterns compare against the resolved attribute chain. Macro patterns retain their delimiter and raw token body for code generation; Python raises `TranslateError` for unsupported pattern macros, while unresolved native macros are rejected by Cargo. Standalone `..` patterns are rejected by the parser with a source location.

### Tests in batch 3

- Feature 41: `test_pattern_box_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 42: `test_pattern_box_literal_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 43: `test_pattern_box_tuple_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 44: `test_pattern_box_nested_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 45: `test_pattern_box_binding_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 46: `test_pattern_const_bare`
- Feature 47: `test_pattern_const_path`
- Feature 48: `test_pattern_const_qualified`
- Feature 49: `test_pattern_const_binding`
- Feature 50: `test_pattern_const_tuple`
- Feature 51: `test_pattern_macro_parses[m!()]`
- Feature 52: `test_pattern_macro_parses[m!(1, 2)]`
- Feature 53: `test_pattern_macro_parses[m! { 1 }]`
- Feature 54: `test_pattern_macro_parses[m!(x @ 1..=5)]`
- Feature 55: `test_pattern_macro_nested_in_tuple`
- Feature 56: `test_pattern_path_constant`
- Feature 57: `test_pattern_path_binding_rejected`
- Feature 58: `test_pattern_path_or`
- Feature 59: `test_pattern_standalone_rest_rejected`
- Feature 60: `test_pattern_struct_rest_batch3`
- Python macro diagnostic: `test_python_pattern_macro_raises_translate_error`

Batch 3 validation: `pytest -v` completed with 87 passed and 6 skipped; `ruff check --fix .` passed.

## N2a pattern batch 4

Nested or-patterns and guards were exercised in tuple, slice, struct, and binding patterns. Rust code generation now parenthesizes an or-pattern nested under an `@` binding. The Pratt precedence for bitwise `|` was corrected relative to comparison operators so boolean guard expressions retain their intended grouping. Python translation now rejects or-pattern alternatives that bind different names, supports payload-like tuple representations for enum variant patterns, and validates struct patterns against declared fields. Wrong-arity tuple patterns remain non-matching in Python; Cargo rejects structurally incompatible tuple patterns in native code.

### Tests in batch 4

- Feature 61: `test_pattern_or_tuple`
- Feature 62: `test_pattern_tuple_or`
- Feature 63: `test_pattern_slice_or_inner`
- Feature 64: `test_pattern_struct_or_inner`
- Feature 65: `test_pattern_binding_or`
- Feature 66: `test_pattern_guard_or`
- Feature 67: `test_pattern_guard_tuple`
- Feature 68: `test_pattern_guard_slice`
- Feature 69: `test_pattern_guard_struct`
- Feature 70: `test_pattern_guard_binding`
- Feature 71: `test_pattern_guard_or_expr`
- Feature 72: `test_pattern_many_arms`
- Feature 73: `test_pattern_arm_bindings`
- Feature 74: `test_pattern_or_disjoint_bindings_rejected`
- Feature 75: `test_pattern_or_binding_consistency`
- Feature 76: `test_pattern_struct_missing_rest_rejected`
- Feature 77: `test_pattern_struct_extra_field_rejected`
- Feature 78: `test_pattern_tuple_wrong_arity_rejected` and `test_python_tuple_pattern_wrong_arity_does_not_match`
- Feature 79: `test_pattern_slice_wrong_length`
- Feature 80: `test_pattern_nested_or_tuple`
- Cross-backend parity: `test_python_nested_or_guard_parity`

Batch 4 validation: `pytest -v` completed with 112 passed and 6 skipped in 14.69 seconds; `ruff check --fix .` passed.

## N2a pattern batch 5

Parser diagnostics now reject empty or-patterns, missing `@` and box subpatterns, standalone rest patterns, and multiple slice or struct rests with source positions. Python translation checks duplicate bindings and or-pattern binding consistency, validates const paths and struct shapes, and raises `TranslateError` with the offending pattern where available. Explicit `const NAME` patterns are emitted as `self::NAME` in Rust so unresolved constants cannot silently become identifier bindings. Rust remains responsible for type checking, duplicate bindings, exhaustiveness, and other invalid native patterns; these failures are surfaced through `RustCompileError.pretty()`.

Shared match diagnostics report the first unreachable arm and first duplicate literal per match, and report a guard binding that shadows the scrutinee name. The batch tests verify diagnostics are emitted once per match expression.

### Tests in batch 5

- Feature 81: `test_error_duplicate_binding_native`
- Feature 82: `test_error_duplicate_binding_or_different_types`
- Feature 83: `test_error_rest_outside_slice`
- Feature 84: `test_error_empty_or_pattern`
- Feature 85: `test_error_at_binding_without_inner_pattern`
- Feature 86: `test_error_box_without_inner_pattern`
- Feature 87: `test_error_unresolved_const`
- Feature 88: `test_error_unsupported_macro`
- Feature 89: `test_error_two_slice_rests`
- Feature 90: `test_error_two_struct_rests`
- Feature 91: `test_warning_unreachable_arm`
- Feature 92: `test_warning_duplicate_literal_arm`
- Feature 93: `test_warning_guard_shadows_binding`
- Feature 94: `test_error_non_exhaustive_native_match`
- Feature 95: `test_error_source_location`
- Feature 96: `test_error_duplicate_binding_contains_pattern`
- Feature 97: `test_python_unsupported_pattern_has_translate_error`
- Feature 98: `test_native_compile_error_pretty_has_pattern`
- Feature 99: `test_error_type_and_message_contract`
- Feature 100: `test_match_warnings_emitted_once_per_expression`

Batch 5 validation: `pytest -v` completed with 133 passed and 8 skipped in 13.35 seconds; `ruff check --fix .` passed.