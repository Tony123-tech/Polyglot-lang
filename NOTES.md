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

## N2c batch 1

The native code generator already emits `#[pyclass]` and a module registration call for every struct declaration. A native integration test now defines two independent structs with constructors and verifies that both classes can be instantiated from Python. No code-generation change was needed.

- Covered by `tests/test_native_features.py::test_native_register_multiple_pyclasses`.
- `pytest -q`: 134 passed, 8 skipped.

## N2c batch 2

A shared stable dependency ordering now emits a referenced inner struct before its containing struct in both backends. Native and Python tests declare `Outer` before `Inner` and verify that an `Outer` instance contains a usable `Inner` instance.

- Covered by `tests/test_native_features.py::test_native_struct_field_of_struct` and `tests/test_point.py::test_python_struct_field_of_later_struct`.
- `pytest -q`: 136 passed, 8 skipped.

## N2c batch 3

Simple enums containing only unit variants are now emitted as PyO3 classes and registered with the module. Python backend variants retain the standard `Enum` representation, accessed as `Choice.A` and `Choice.B`; native variants are class attributes on the PyO3 enum class. Payload enums remain unchanged.

- Covered by `tests/test_native_features.py::test_native_unit_enum_variants` and `tests/test_point.py::test_python_unit_enum_variant_representation`.
- `pytest -q`: 138 passed, 8 skipped.

## N2c batch 4

The Python backend now represents payload enum variants as callable class attributes that produce tagged tuples, allowing `Maybe::Some(value)` and payload-pattern matching while preserving unit-variant matching. PyO3 0.26 safely exposes only the unit-enum form used here; payload enums stay internal to generated Rust and are not exported. The native exposure test is explicitly skipped with that limitation.

- Covered by `tests/test_point.py::test_python_payload_enum_representation`.
- Native limitation: `tests/test_native_features.py::test_native_payload_enum_is_exposed`.
- `pytest -q`: 139 passed, 9 skipped.

## N2c batch 5

`impl` blocks on unit enums now contribute methods to the Python enum class and are emitted as `#[pymethods] impl` blocks for native PyO3 enums. The parity tests call the method through variants in both backends. Payload enum methods remain Python-only because payload enums are not exposed as native classes.

- Covered by `tests/test_native_features.py::test_native_unit_enum_variants` and `tests/test_point.py::test_python_unit_enum_variant_representation`.
- `pytest -q`: 139 passed, 9 skipped.

## N2c batch 6

Structs with `Vec<i64>` fields are constructible from Python in both backends. PyO3 converts Python lists at the constructor boundary, and the exposed field reads back as a Python list.

- Covered by `tests/test_native_features.py::test_native_struct_with_vec_field` and `tests/test_point.py::test_python_struct_with_vec_field`.
- `pytest -q`: 141 passed, 9 skipped.

## N2c batch 7

Struct methods returning `Self` are supported by returning a cloned pyclass instance in native Rust and the cloned dataclass value in Python. Both tests verify that the returned object has equivalent fields but is not the original instance.

- Covered by `tests/test_native_features.py::test_native_struct_method_returns_self` and `tests/test_point.py::test_point_method_returns_self`.
- Python translation uses shallow `copy.copy` for Rust's `.clone()` method.
- `pytest -q`: 143 passed, 9 skipped.

## N2c batch 8

Generic struct declarations are erased to a concrete native pyclass, with each declared type parameter mapped to `i64`; the Python backend maps the corresponding field and method annotations to `int`. This is a deliberate native limitation: generic arguments are not runtime-specialized and all type parameters default to `i64`.

- Covered by `tests/test_native_features.py::test_native_generic_struct_defaults_to_i64` and `tests/test_point.py::test_python_generic_struct_defaults_type_parameter_to_int`.
- `pytest -q`: 145 passed, 9 skipped.

## N2c batch 9

The native code generator groups `impl` blocks by target and emits one Rust `impl` and one `#[pymethods]` block per class, preventing duplicate PyO3 registration scaffolding. The Python translator already combines methods from all impl blocks into the class body. Both backends now test methods defined in separate blocks.

- Covered by `tests/test_native_features.py::test_native_multiple_impl_blocks` and `tests/test_point.py::test_point_methods_across_impl_blocks`.
- `pytest -q`: 147 passed, 9 skipped.

## N2c batch 10

A combined regression program declares a struct, unit enum, and `impl` blocks for both. The native backend registers both classes and their methods; the Python backend provides matching behavior.

- Covered by `tests/test_native_features.py::test_native_struct_enum_impl_regression` and `tests/test_point.py::test_python_struct_enum_impl_program_regression`.
- `pytest -q`: 149 passed, 9 skipped.

## N2d batch 11

`RustCompileError` is now a dataclass carrying message, Rust and source locations, raw stderr, generated source, and error code. Cargo output parsing extracts `error[...]` and `--> src/lib.rs:LINE:COL`, while compatibility properties preserve `error_line`, `error_col`, and `rust_message`.

- Added direct parser tests in `tests/test_rust_compile_error.py`.
- `pytest -q`: 151 passed, 9 skipped.

## N2d batch 12

`RustCodegen` now exposes a one-based `line_map` from generated Rust line numbers to original `(line, col)` positions. `set_origin` records the current AST node, and `emit` associates generated lines with that origin. Item, struct, enum, impl, function, and statement generation establish origins.

- Covered by `tests/test_rust_compile_error.py::test_rust_codegen_records_source_origin_for_emitted_lines`.
- `pytest -q`: 152 passed, 9 skipped.

## N2d batch 13

The `RustCodegen` instance is retained through compilation, and its line map is passed into the Cargo build error parser. Rust error locations now populate `source_line` and `source_col` when an origin is available; linker errors and unmapped generated lines retain `None`.

- Covered by `tests/test_rust_compile_error.py::test_cargo_error_maps_rust_line_to_source_origin`.
- `pytest -q`: 153 passed, 9 skipped.

## N2d batch 14

Source line text is now retained from lexer tokens through `Program` and code generation. The structured compiler error receives the original line text and `pretty()` prints both the original Polyglot line/location and the offending generated Rust context. Token and Program additions are optional, preserving existing construction call sites.

- Covered by `tests/test_rust_compile_error.py::test_pretty_includes_original_and_generated_lines`.
- `pytest -q`: 154 passed, 9 skipped.

## N2d batch 15

Four native compiler errors are now exercised end to end: a return-type mismatch, assignment to an immutable local, a non-exhaustive match, and an unresolved constant pattern. The code generator now preserves local binding mutability instead of marking every binding mutable. Existing Vec mutation fixtures were updated to use `let mut`. Cargo diagnostic location parsing now selects the location following the first error header rather than a preceding warning.

- Covered by the parametrized `tests/test_rust_compile_error.py::test_native_errors_map_back_to_polyglot_source`.
- Unresolved-constant diagnostics in inline match expressions currently map to the enclosing match line.
- `pytest -q`: 158 passed, 9 skipped.

## N2d batch 16

`RustCompileError` now has a warnings list populated from Cargo warning headers when an error is also present. The build path continues to raise only when Cargo exits unsuccessfully; a warning-only successful result is explicitly tested as non-exceptional.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_captures_warnings_without_replacing_error` and `test_warning_only_cargo_result_does_not_raise`.
- `pytest -q`: 160 passed, 9 skipped.

## N2d batch 17

Only the first Cargo error is promoted into the structured `RustCompileError` fields. The complete stderr remains in `raw_stderr`, retaining additional distinct diagnostics for callers and reports.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_keeps_later_errors_in_raw_stderr`.
- `pytest -q`: 161 passed, 9 skipped.

## N2d batch 18

Linker-style errors without a `src/lib.rs` location remain valid structured errors with message and raw stderr; source and Rust line/column fields stay unset rather than fabricating a location.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_without_source_location`.
- `pytest -q`: 161 passed, 9 skipped.

## N2d batch 19

Identical Cargo error blocks are removed from retained raw stderr while distinct later errors remain available. The first error continues to populate the primary structured fields.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_deduplicates_identical_diagnostics`.
- `pytest -q`: 162 passed, 9 skipped.

## N2d batch 20

The full existing suite was rerun against the structured error type. `RustCompileError` remains a `RuntimeError` subclass, and compatibility aliases for `error_line`, `error_col`, and `rust_message` are verified.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_preserves_runtime_error_compatibility`.
- `pytest -q`: 163 passed, 9 skipped.

### Status after N2d

- Native test sources are merged and compiled once per pytest session, with the same build flow available for CI pre-builds.
- Content-addressed caching retains Cargo's `target/` artifacts for incremental builds.
- Structured Rust compiler errors and readable formatting are available.
- Dynamic-library loading, dataclass module registration, trust checking, and associated-method translation corrected.
- **30 tests passed; cache-hit build time was approximately 0.15 seconds, down from approximately 214 seconds.**

## N3 batch 21

Iteration over `Vec<String>` and `Vec<(i64, i64)>` is verified in both backends. Python `for` translation now supports tuple destructuring patterns in addition to identifier targets, matching Rust's loop binding behavior for pair elements.

- Covered by `tests/test_native_n3.py::test_native_iter_vec_string`, `test_native_iter_vec_tuple`, and `test_python_iter_vec_parity`.
- `pytest -q`: 167 passed, 9 skipped.

## N3 batch 22

`Vec::contains` now emits Rust's reference-taking `contains(&value)` call. The Python backend lowers the same method to a membership test, with native and Python cases covering present and absent values.

- Covered by `tests/test_native_n3.py::test_native_vec_contains` and `test_python_vec_contains_parity`.
- `pytest -q`: 169 passed, 9 skipped.

## N3 batch 23

`Vec::sort` now works through the existing method path. `sort_by` supports both native Rust Ordering closures and the requested numeric-subtraction shorthand, which is lowered to `.cmp()` for Rust and `functools.cmp_to_key` for Python.

- Covered by `tests/test_native_n3.py::test_native_vec_sort_and_sort_by` and `test_python_vec_sort_parity`.
- Closure parsing now treats unparenthesized closure parameters as pattern atoms so the closing pipe is not consumed as an or-pattern operator.
- `pytest -q`: 172 passed, 9 skipped.

## N3 batch 24

The `Vec.map` shorthand is supported in both backends. Native code generation consumes the vector, maps the closure, and collects a `Vec`; Python translation emits a list comprehension. The supported closure shape is one identifier parameter.

- Covered by `tests/test_native_n3.py::test_native_vec_map` and `test_python_vec_map_parity`.
- `pytest -q`: 174 passed, 9 skipped.

## N3 batch 25

The `Vec.filter` shorthand is supported in both backends. Native code uses `into_iter().filter(...).collect::<Vec<_>>()`; Python emits a filtering list comprehension. One-parameter identifier closures are supported.

- Covered by `tests/test_native_n3.py::test_native_vec_filter` and `test_python_vec_filter_parity`.
- `pytest -q`: 176 passed, 9 skipped.

## N3 batch 26

The `Vec.fold(initial, closure)` shorthand is supported in both backends. Native code lowers it to `into_iter().fold(...)`, and Python calls `functools.reduce` with the same initial value and accumulator argument order.

- Covered by `tests/test_native_n3.py::test_native_vec_fold` and `test_python_vec_fold_parity`.
- `pytest -q`: 178 passed, 9 skipped.

## N3 batch 27

Vector range indexing is supported as `&values[start..end]`. Native code returns an owned vector copy to cross the Python extension boundary; Python uses a list slice. The returned values match the requested half-open slice.

- Covered by `tests/test_native_n3.py::test_native_vec_slice` and `test_python_vec_slice_parity`.
- `pytest -q`: 180 passed, 9 skipped.

## N3 batch 28

Vector concatenation is supported through `extend` in both backends. Native code uses `Vec::extend`, and Python calls `list.extend`; direct `Vec + Vec` remains unsupported because Rust does not define that operator for vectors.

- Covered by `tests/test_native_n3.py::test_native_vec_extend` and `test_python_vec_extend_parity`.
- `pytest -q`: 182 passed, 9 skipped.

## N3 batch 29

`Vec.insert`, `remove`, `retain`, and `dedup` are supported in both backends. Native index arguments are cast to `usize`; Python `remove` is translated to `pop(index)`, and runtime helpers preserve in-place `retain` and adjacent-only `dedup` semantics.

- Covered by `tests/test_native_n3.py::test_native_vec_insert_remove_retain_dedup` and `test_python_vec_mutation_parity`.
- `pytest -q`: 187 passed, 9 skipped.

## N3 batch 30

`Vec.windows(size)` and `chunks(size)` now return `Vec<Vec<i64>>` natively and lists of lists in Python. Native slices are copied into owned vectors for PyO3 conversion. Python raises `ValueError` for nonpositive sizes, matching the native operation's invalid-size failure rather than returning misleading empty output.

- Covered by `tests/test_native_n3.py::test_native_vec_windows_and_chunks` and `test_python_vec_windows_and_chunks_parity`.
- `pytest -q`: 190 passed, 9 skipped.

## N4 batch 31

String byte length, Unicode character iteration, and UTF-8 byte iteration are supported. Python's `.len()` lowering now counts UTF-8 bytes for strings while preserving normal `len()` behavior for vectors and other containers.

- Covered by `tests/test_native_n4.py::test_native_string_len_chars_bytes` and `test_python_string_len_chars_bytes_parity`.
- Python `.bytes()` matches PyO3's `Vec<u8>` conversion by returning a `bytes` object.
- `pytest -q`: 194 passed, 9 skipped.

## N4 batch 32

String splitting returns `Vec<String>`/Python lists, and vector joining uses a string separator and returns a string in both backends.

- Covered by `tests/test_native_n4.py::test_native_string_split_join` and `test_python_string_split_join_parity`.
- `pytest -q`: 197 passed, 9 skipped.

## N4 batch 33

String replacement, trimming, prefix checks, and suffix checks are available across both backends. Native string references are passed to Rust's `&str` APIs, and Rust's borrowed trim result is converted to an owned `String` for the Python boundary.

- Covered by `tests/test_native_n4.py::test_native_string_replace_trim_prefix_suffix` and `test_python_string_replace_trim_prefix_suffix_parity`.
- `pytest -q`: 202 passed, 9 skipped.

## N4 batch 34

Unicode-aware uppercase and lowercase conversions are supported. Native `to_uppercase`/`to_lowercase` and Python `upper`/`lower` are tested with a sharp-s case whose uppercase expands to two characters.

- Covered by `tests/test_native_n4.py::test_native_string_case_conversion` and `test_python_string_case_conversion_parity`.
- `pytest -q`: 205 passed, 9 skipped.

## N4 batch 35

The parser now accepts turbofish type arguments on method calls without changing existing `MethodCall` construction. `.parse::<i64>()` lowers to Rust's typed parse with `unwrap()` and Python's `int()`. Invalid-input error behavior is backend-specific; successful parsing is verified.

- Covered by `tests/test_native_n4.py::test_native_string_parse_i64` and `test_python_string_parse_i64_parity`.
- `pytest -q`: 207 passed, 9 skipped.

## N4 batch 36

`format!` with multiple positional arguments is verified in both backends. Named format arguments are not added in this batch because the current macro parser represents them as assignment expressions rather than format-specific arguments.

- Covered by `tests/test_native_n4.py::test_native_format_multiple_arguments` and `test_python_format_multiple_arguments_parity`.
- `pytest -q`: 209 passed, 9 skipped.

## N4 batch 37

`write!` into a mutable `String` is supported for identifier targets and positional formatting arguments. Native generation brings `std::fmt::Write` into scope and unwraps the infallible String write result; Python translates the macro statement into string concatenation.

- Covered by `tests/test_native_n4.py::test_native_write_macro_into_string` and `test_python_write_macro_into_string_parity`.
- A generated `&mut` reference originally lacked required spacing; native write tests now cover the corrected syntax.
- `pytest -q`: 211 passed, 9 skipped.

## N4 batch 38

String `+` concatenation is recognized by native code generation for expressions with known string parameters, string literals, or string-returning methods, and is emitted with `format!` to avoid Rust's `String + &str` ownership requirements. Python retains native string `+`.

- Covered by `tests/test_native_n4.py::test_native_string_concatenation` and `test_python_string_concatenation_parity`.
- `pytest -q`: 213 passed, 9 skipped.

## N4 batch 39

String range slicing uses UTF-8 byte offsets in both backends. Native code returns an owned `String`; Python slices encoded bytes and decodes the result, so invalid UTF-8 boundaries raise instead of silently changing indexing semantics.

- Covered by `tests/test_native_n4.py::test_native_string_slice_uses_utf8_byte_offsets` and `test_python_string_slice_uses_utf8_byte_offsets`.
- `pytest -q`: 215 passed, 9 skipped.

## N4 batch 40

String equality and ordering use Rust's native `String` comparison and Python's lexicographic string comparison. Both backends verify less-than behavior for ASCII strings.

- Covered by `tests/test_native_n4.py::test_native_string_comparison_and_ordering` and `test_python_string_comparison_and_ordering_parity`.
- Validation is recorded after the batch test run.

## Security

`trusted=False` is not a sandbox; it is a speed bump. Operating-system-level isolation, such as a container, seccomp, and disabled network access, is required for untrusted input. The native cache directory is created with mode `0o700`. `pyo3 = "0.26"` is pinned in `Cargo.toml`.

Three isolation levels are provided through `api/`:

- **A:** In-process queue.
- **B:** Subprocess.
- **C:** Docker with `--network=none --read-only`.

HTTP requests are handled with isolation level B by default. An `X-Trusted-Token` must be provided when `trusted=true` is set.

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

## N5 test harness restoration

Replaced the malformed `tests/test_native_n5.py` implementation copy with a `SOURCES` module consumed by the session-scoped `native_ns` fixture. All native sources are merged during collection and compiled as one generated extension for the session; every source function is prefixed `n5_`. Added parity coverage for the existing Option predicates/unwrapping and Python-only Result predicates.

- Covered by `tests/test_native_n5.py::test_native_option_methods`, `test_python_option_methods_parity`, and `test_python_result_predicates`.
- `pytest -q`: 227 passed, 9 skipped.

## N5 batch 5

Added Option `and_then`, `or_else`, and `filter` in the Python runtime; native Option combinator calls use Rust's standard Option methods. Native `filter` codegen now applies the Vec iterator/collect lowering only when the receiver is known to be a Vec, avoiding the Option collision. Python filter dispatch preserves existing Vec behavior while handling Option values. Option `map` remains intentionally unsupported because the code generator's `map` lowering is reserved for Vec; both backends now raise a clear error suggesting `match` or `if let` instead of failing with a confusing runtime `TypeError`.

- Covered by `tests/test_native_n5.py::test_native_option_methods` and `test_python_option_methods_parity`.
- Existing Vec filter behavior remains covered by `tests/test_native_n3.py::test_native_vec_filter` and `test_python_vec_filter_parity`.
- `pytest -q tests/test_native_n3.py tests/test_native_n5.py`: 43 passed.

## N5 batch 6

Added Python Result `and_then`/`or_else` and Result-to-Option `ok()`/`err()`, plus Option-to-Result `ok_or()`/`ok_or_else()`. The Result `and_then`/`or_else` operations remain Python-only: native Result values in local bindings cannot safely encode Rust `Err` as a Python exception. Native `ok()`/`err()` and Option `ok_or()` are verified through expressions returning ordinary Python-compatible values. Native Result constructors now annotate their inferred error type as `PyErr` where needed, preventing inference failures in conversion expressions.

- Covered by `tests/test_native_n5.py::test_native_option_methods`, `test_python_option_methods_parity`, and `test_python_result_conversions_and_combinators`.
- `test_native_result_and_then_or_else_requires_typed_err` is explicitly skipped because native Result `Err` cannot be represented in the existing local-binding model.
- `pytest -q tests/test_native_n5.py`: 19 passed, 1 skipped.

## N5 batch 7

Implemented `?` for Result values. Native generation emits Rust's `?` operator, preserving parse errors through the PyO3 `PyResult` return. Python `_Result.__try__()` returns Ok values and raises a private sentinel for Err; functions declared to return Result catch only that sentinel and return its Result value. Python string parsing under `?` returns an internal Result while existing non-`?` parse behavior remains unchanged.

- Covered by `tests/test_native_n5.py::test_native_result_try_operator` and `test_python_result_try_operator`, including Python Err propagation.
- `pytest -q tests/test_native_n5.py`: 21 passed, 1 skipped.

## N5 batch 8

Added parser and AST support for `if let Some(binding) = expression`. Native generation emits Rust `if let`. Python binds the expression once to a temporary, checks `_Option` and `is_some()`, then binds the payload before translating the branches. Other if-let pattern forms remain unsupported and raise `TranslateError`.

- Covered by `tests/test_native_n5.py::test_native_if_let_some_and_none` and `test_python_if_let_some_and_none_parity`.
- `pytest -q tests/test_native_n5.py`: 23 passed, 1 skipped.

## N5 batch 9 — out of scope

`while let Some(x) = iter.next()` is out of scope. Native Rust iterators already produce `Option` from `next()`, but the Python backend has no corresponding iterator adapter with Rust-style Option-returning `next()`; Python's built-in iterator protocol instead raises `StopIteration`. Adding an adapter, lifetime/consumption behavior, and parity tests would be a significant feature rather than a safe lowering. The batch is explicitly skipped rather than simulating mismatched behavior.

- `tests/test_native_n5.py::test_n5_while_let_option_iterator_not_supported` records the reason.

## N5 batch 10

Extended shared match diagnostics to warn on known Option/Result scrutinees with a Some/Ok arm but no None/Err arm, respectively. Both backends pass scrutinee type information from function parameters and typed/local constructor bindings, so unrelated user enums with variants named Some do not trigger false warnings. Complete matches remain warning-free, and diagnostics do not reject the program.

- Covered by `tests/test_native_n5.py::test_option_result_match_warn_once_per_match`, which checks exactly one warning per incomplete match through both translators and confirms complete/custom-enum matches do not warn.
- `pytest -q tests/test_native_n5.py tests/test_native_n2a_patterns.py`: 127 passed, 10 skipped.
- Final `pytest -q`: 243 passed, 11 skipped.
- `pytest -q tests/test_native_n5.py`: 24 passed, 2 skipped.

Batch 5 validation: `pytest -v` completed with 133 passed and 8 skipped in 13.35 seconds; `ruff check --fix .` passed.

## Async runtime

The native backend uses hand-written `PyFuture` and `PyFutureIterator` classes in `_RUNTIME_PREFIX` in `backends/rust/codegen.py`; it does not use `pyo3-asyncio` or PyO3 experimental async support. Both classes are `#[pyclass(unsendable)]` and poll the Rust future with a noop waker; the future output is `PyResult<PyObject>`. Async free functions are emitted as `#[pyfunction] fn name(params) -> PyResult<PyFuture> { Ok(py_future(async move { ... })) }`, and await expressions emit `.await`. The tail value is converted to Python with `into_py_any(py)` inside `Python::with_gil`, fixing conversion for PyO3 0.26.

Tests: `tests/test_native_n5.py::test_native_async_add`, `tests/test_native_n5.py::test_native_async_identity`.

## Sandbox

Sandboxing is independent of the trust gate. `trusted=False` raises `UntrustedCompileError` before compilation; `sandbox=True` on `PythonBackend` replaces `__builtins__` with the restricted dictionary returned by `_sandbox_builtins()`. Compilation alone does not execute function bodies, so sandbox tests must call the compiled function to trigger blocked-name errors.

Tests: `tests/test_native_n5.py::test_sandbox_blocks_import`, `tests/test_native_n5.py::test_sandbox_blocks_eval`, `tests/test_native_n5.py::test_sandbox_blocks_open`, `tests/test_native_n5.py::test_sandbox_blocks_compile`, `tests/test_native_n5.py::test_sandbox_blocks_globals`, `tests/test_native_n5.py::test_sandbox_allows_safe_builtins`, `tests/test_native_n5.py::test_sandbox_allows_isinstance`, `tests/test_native_n5.py::test_sandbox_allows_len`, `tests/test_native_n5.py::test_sandbox_allows_exceptions`.

## Build performance

When available, `sccache` is configured through `RUSTC_WRAPPER` in `_build_env`. Cargo's `target` directory remains per-build under each build directory rather than being shared. After clearing the cache, a cold build takes about two minutes; a warm second run takes about three seconds. `test_native_errors_map_back_to_polyglot_source` is marked `slow`; `pyproject.toml` sets `addopts = "-m 'not slow'"`, deselecting four slow tests by default.

## N5 complete

N5 delivers Option and Result construction and combinators, the `?` operator, `if let`, nested Option support, async functions through the custom PyFuture runtime, match exhaustiveness warnings, and Python built-ins sandboxing. The features that remain Python-only are Result `is_ok`, `is_err`, `map_err`, and `ok_or` in let-bindings (native `Err` emits `PyErr`, which does not match the declared error type), Box patterns, enum payload slices, and nested Vec slice patterns. Option `map` remains intentionally unsupported because it collides with Vec `map`; both backends reject it with a clear error.

Out of scope: `while let Some(x) = iter.next()` because Python has no Rust-style Option-returning `next()`; async functions as impl methods; and using a `PyFuture` from a thread other than the one that created it, because `PyFuture` is unsendable.

Tests: `tests/test_native_n5.py` covers native and Python Option/Result, async, sandbox, `?`, `if let`, and match diagnostics.

`pytest -q`: 252 passed, 9 skipped, 4 deselected.

## N6 batch 1

Added Vec `is_empty`, `clear`, and `reverse`, plus String `repeat`, with equivalent Python and native lowerings. Native repeat counts are cast to `usize`; the Python backend uses string multiplication.

- Covered by `tests/test_native_n5.py::test_native_vec_clear_empty_reverse_and_string_repeat` and `tests/test_native_n5.py::test_python_vec_clear_empty_reverse_and_string_repeat`.
- `pytest -q`: 258 passed, 9 skipped, 4 failed, 7 deselected. The four failures were in unrelated inline annotation resolution and native extension loading (`_Py_IsFinalizing`); the affected Vec/String/N5 tests passed (95 passed).

## N6 batch 2

Added Vec `first`, `last`, and bounds-checked `get`, plus String `find`, with matching native and Python behavior. String search returns an Option containing the UTF-8 byte offset, and an absent match returns None.

- Covered by `tests/test_native_n5.py::test_native_vec_first_last_get_and_string_search` and `tests/test_native_n5.py::test_python_vec_first_last_get_and_string_search`.
- `pytest -q`: 260 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 3

Added Vec `truncate` and String `is_empty`, `starts_with`, and `ends_with` to both backends. Native truncation casts the signed source index to `usize`, matching Rust's behavior for negative values (no truncation).

- Covered by `tests/test_native_n5.py::test_native_vec_truncate_and_string_predicates` and `tests/test_native_n5.py::test_python_vec_truncate_and_string_predicates`.
- `pytest -q`: 262 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 4

Added String `trim_start`, `trim_end`, and `is_ascii` on both backends.

- Covered by `tests/test_native_n5.py::test_native_string_trim_start_end_and_is_ascii` and `tests/test_native_n5.py::test_python_string_trim_start_end_and_is_ascii`.
- `pytest -q`: 264 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 5

Added String `lines` and `split_whitespace`, plus Vec `sort_unstable`, on both backends. Python `lines` follows Rust's newline and CRLF terminator behavior.

- Covered by `tests/test_native_n5.py::test_native_string_lines_whitespace_and_vec_sort_unstable` and `tests/test_native_n5.py::test_python_string_lines_whitespace_and_vec_sort_unstable`.
- `pytest -q`: 266 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 6

Added String `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, and `to_ascii_uppercase` to both backends. `as_bytes` returns Python `bytes` from either backend, and the ASCII case methods leave non-ASCII characters unchanged.

- Covered by `tests/test_native_n5.py::test_native_string_bytes_boundaries_and_ascii_case` and `tests/test_native_n5.py::test_python_string_bytes_boundaries_and_ascii_case`.
- `pytest -q`: 268 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 7

Added Vec `rotate_left`, `rotate_right`, `fill`, and `swap` to both backends. Python lowering checks rotation and swap bounds rather than silently accepting indices or counts that Rust would reject.

- Covered by `tests/test_native_n5.py::test_native_vec_rotate_fill_and_swap` and `tests/test_native_n5.py::test_python_vec_rotate_fill_and_swap`.
- `pytest -q`: 270 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (51 passed).

## N6 batch 8

Added Vec `extend_from_slice`, `resize`, and `split_off` on both backends. The Python lowering preserves the receiver's retained prefix for `split_off` and does not mutate the source slice supplied to `extend_from_slice`.

- Covered by `tests/test_native_n5.py::test_native_vec_extend_from_slice_resize_and_split_off` and `tests/test_native_n5.py::test_python_vec_extend_from_slice_resize_and_split_off`.
- `pytest -q`: 272 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (53 passed).

## N6 batch 9

Added String `rfind`, `strip_prefix`, `strip_suffix`, `split_once`, and `rsplit_once` on both backends. Search positions are UTF-8 byte offsets, and split methods return an Option containing the two owned strings.

- Covered by `tests/test_native_n5.py::test_native_string_rfind_strip_and_split_once` and `tests/test_native_n5.py::test_python_string_rfind_strip_and_split_once`.
- `pytest -q`: 274 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (55 passed).

## N6 batch 10

Added Vec `partition`, `sort_by_key`, and `is_sorted` on both backends. Python lowering accepts the Rust-style `into_iter()` before `partition` and returns a pair of stable-order lists.

- Covered by `tests/test_native_n5.py::test_native_vec_partition_sort_by_key_and_is_sorted` and `tests/test_native_n5.py::test_python_vec_partition_sort_by_key_and_is_sorted`.
- `pytest -q`: 276 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (57 passed).

## N6 complete

N6 added the remaining `Vec` and `String` standard-library methods in batches 1–10. `Vec` gained `is_empty`, `clear`, `reverse`, `first`, `last`, `get`, `truncate`, `sort_unstable`, `rotate_left`, `rotate_right`, `fill`, `swap`, `extend_from_slice`, `resize`, `split_off`, `partition`, `sort_by_key`, and `is_sorted`. `String` gained `repeat`, `contains`, `find`, `rfind`, `is_empty`, `starts_with`, `ends_with`, `trim_start`, `trim_end`, `is_ascii`, `lines`, `split_whitespace`, `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, `to_ascii_uppercase`, `strip_prefix`, `strip_suffix`, `split_once`, and `rsplit_once`.

### Bug 1

**Symptom:** The Python backend returned `""` for `value.strip_suffix("")`, while native Rust returned the original string.

**Cause:** The suffix path contained a special case, `value[:len(value) - len(part)] if part else ""`, that assumed an empty suffix should strip the whole string. Rust treats both empty prefixes and empty suffixes as successful matches that remove nothing.

**Fix:** The suffix path now slices with `len(part)` unconditionally, so an empty suffix returns the original string.

**Lesson:** Empty-string edge cases in search and strip operations should follow the target operation's identity behavior rather than being special-cased.

### Review: UTF-8 byte offsets

Rust `find` and `rfind` return UTF-8 byte offsets. The Python backend obtains a character index from `str.find`/`str.rfind` and converts it through the UTF-8 length of the preceding prefix, which is correct both when the match itself is multibyte and when earlier characters are multibyte. The parity tests now cover a multibyte match (`"café"` finding `"é"` at byte 3), a later ASCII match (`"café x"` finding `"x"` at byte 6), an empty needle, and the corresponding `rfind` cases.

### Bug 2 (environment)

**Symptom:** Three inline-language tests failed with `NameError: name 'i64' is not defined`, and native extension loading failed with `_Py_IsFinalizing`.

**Cause:** The suite was run under Python 3.12 while the project declares `requires-python >= 3.14`. Before PEP 649, annotations are evaluated eagerly at definition time, so `def add(a: i64, ...)` fails before the `@inline_lang` decorator can run. The native extension-load failure was specific to that interpreter and toolchain combination.

**Fix:** The environment now matches the declared requirement (Python 3.14.8).

**Lesson:** Test on the interpreter range the project declares; annotation-evaluation changes can determine whether decorator-based tooling is reachable at all.

### Status after N6

- All four previously failing tests pass: the three inline annotation cases and the native extension-load case.
- `pytest -q`: 280 passed, 9 skipped, 7 deselected (`pytest -q -m slow`: 7 passed, 289 deselected).

## N7: HashMap

N7 adds `HashMap` support on both backends with byte-for-byte parity for the implemented surface:

- Construction: `HashMap::new()` and `HashMap<K, V>`-annotated bindings.
- Core operations: `insert`, `get`, `get_mut`, `remove`, `contains_key`, `len`, and `is_empty`.
- Views: `keys`, `values`, and `iter`, each returning an owned `Vec` so iteration can cross the Python boundary.
- Entry subset: `or_insert`, `or_insert_with`, and `and_modify`.
- Mutable slots: `*slot = value` and `*slot += value` (plus `-=`, `*=`, `/=`) for values obtained from `get_mut` or the entry API.
- Struct fields: a `HashMap` field accessed as `self.field` inside an `impl` method is recognized as a map.

The Python backend represents a Rust `&mut V` with `_MapSlot`, which records the mapping and key and writes back on mutation, and the `Entry` API with `_MapEntry`. Dispatch on both backends is driven by the existing binding-type tables (`binding_types` / `_binding_types`) plus a per-impl field-type map, so it does not depend on global inference.

### Bug 1

**Symptom:** `*x += 1` compiled even when `x` was an ordinary integer, silently ignoring the dereference.

**Cause:** The `*` unary operator lowered to its operand in the Python backend, and only map-slot expressions were expected to reach the mutation path. Nothing checked that a dereference-assignment target was actually a mutable map slot, so invalid programs produced valid-looking Python.

**Fix:** Added `_is_map_slot_expr`, which accepts only tracked slot identifiers, `get_mut(...).unwrap()`/`.expect()`, and entry `or_insert`/`or_insert_with` chains. `_translate_assign_stmt`, `_translate_compound_assign_stmt`, and closure mutation lowering now reject any other dereference target with a `TranslateError`. Slot bindings are tracked in `translate_let`, scoped per function, and extended for closure parameters during body lowering.

**Lesson:** When a source operator is erased during lowering, validate the erased operand at the lowering site. Otherwise a type error in the source becomes a silently different operation at runtime.

### Ownership and parity review

- `get` returns an owned `Option<V>` on the native side by appending `.cloned()`; `keys`, `values`, and `iter` likewise return owned vectors. Generated structs and enums already derive `Clone`, so cloned views are available for every supported value type.
- Python `get`/views return the stored Python object rather than a deep copy, matching the existing `Vec::get` behavior. Mutating a value obtained from `get` therefore aliases on the Python side instead of being rejected by the borrow checker.
- `insert` and `entry` strip an explicit `&`/`&mut` from the key argument on the native side; `get`, `get_mut`, `remove`, and `contains_key` add one. This keeps Rust's `Borrow`-based signatures working for both `&key` and bare literals.
- Iteration order is unspecified in both backends because both use hash maps. Tests sort view results whenever order would otherwise be observable.
- Map-field detection resolves the receiver through parameter types and nested struct fields, so `self.field`, `param.field`, and `param.inner.field` all dispatch as maps when their declared types are known. Unknown or unannotated receivers still fall through to generic method dispatch.

### Follow-ups

- Added `HashMap` receiver resolution for parameters and nested struct fields on both backends, sharing the same `_expr_struct_type` helper used for map detection.
- Added entry `key()` and `or_insert_with_key()`.
- Added `%=` for map-slot mutation and switched `/=`/`%=` to the Rust-compatible integer helpers.
- New coverage lives in `tests/test_native_n7_map.py` (`map_receivers`, `map_entry_key`).

### Status after N7

- `pytest -q`: 326 passed, 9 skipped, 7 deselected (`pytest -q -m slow`: 7 passed, 335 deselected).
- New parity coverage lives in `tests/test_native_n7_map.py` and includes two rejection tests for invalid dereference assignments.

## N8: Primitive type semantics

N8 closes the gap between Rust numeric semantics and Python's unbounded integers:

- `as` casts lower to `_runtime_cast`, which reproduces Rust truncation and sign reinterpretation for integer widths, `isize`/`usize` (64-bit), `f32` rounding, `char` code points, and `bool`.
- `/` and `%` lower to `_runtime_int_div`/`_runtime_int_rem`: integer division truncates toward zero and the remainder keeps the dividend's sign; float `%` uses `fmod`.
- Explicitly annotated numeric bindings, assignments, compound assignments, and returns coerce through their declared type, so overflow wraps at the same boundary as native release builds.
- Overflow parity is scoped to typed numeric bindings. Unannotated inference can still differ because Python has no fixed-width integer type.
- Tests: `tests/test_native_n8_types.py`.

## M2: Python module isolation

Each Python compilation now uses a unique generated module name (`_polyglot_generated_<n>`) and registers itself under that name in `sys.modules`. Previously every program replaced `sys.modules["polyglot"]`, so a second compilation could detach dataclasses created by the first. `RustCodegen` also records `line_map_nodes` alongside the existing line/column map so diagnostics can retain the originating AST node.

Tests: `tests/test_python_module_isolation.py` and the `line_map_nodes` case in `tests/test_rust_compile_error.py`.

## Limitations audit

Each README limitation is pinned by a test or has an explicit out-of-scope rationale:

| Limitation | Coverage |
|---|---|
| Result `is_ok`/`is_err`/`map_err`/`ok_or`/`and_then`/`or_else` are Python-only in let-bindings | `tests/test_native_n5.py` predicate/conversion tests; `tests/test_known_limitations.py::test_result_combinators_and_conversions_are_python_only` |
| `while let Some(x) = iter.next()` | `tests/test_known_limitations.py::test_while_let_next_is_a_parse_error` |
| Box patterns, enum payload slices, nested Vec slice patterns | Python behavior in `tests/test_known_limitations.py`; native lowering remains out of scope because Rust slice patterns do not apply directly to `Vec`/enum payloads |
| Async impl methods | `tests/test_known_limitations.py::test_async_impl_method_runs_in_python` and `::test_async_impl_method_native_codegen_is_sync_only` |
| `PyFuture` is unsendable | `tests/test_known_limitations.py::test_pyfuture_runtime_is_marked_unsendable` |
| Option `map` collides with Vec `map` | `tests/test_known_limitations.py` rejection tests for both backends plus `test_vec_map_still_lowers` |
| HashMap entry subset | `tests/test_native_n7_map.py` entry tests |
| HashMap views allocate and have unspecified order | `tests/test_native_n7_map.py` sorts view results where order is observable |
| HashMap dispatch requires a known type | `tests/test_native_n7_map.py` receiver tests; unknown receivers remain generic by design |
| `get_mut`/entry mutation requires an explicit dereference | `tests/test_native_n7_map.py` rejection tests |

## Milestone status

See the README Status table for the authoritative milestone status.

## N7 batch 1 — usize/isize as distinct types

**Symptom:** `Vec::len()` and index codegen silently crossed the `usize`/`i64` boundary, while Python casts ignored the selected target width.
**Cause:** Pointer-sized integers were treated as ordinary `i64` values in generated code and did not have explicit-cast runtime checks.
**Fix:** Added shared type helpers, Python/Rust boundary diagnostics, explicit cast lowering, target-width-aware Python casts, and removed implicit length/index conversions. Updated existing native test sources to use explicit `as` conversions where required.
**Lesson:** A backend must preserve source type boundaries before lowering; host integer behavior cannot stand in for the compilation target's pointer width.

- Covered by: `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`, `test_python_implicit_len_return_reports_boundary_error`, `test_native_implicit_len_return_reports_boundary_error`, `test_python_implicit_len_let_reports_boundary_error`, `test_python_i64_index_requires_explicit_cast`, `test_native_i64_index_requires_explicit_cast`, `test_python_usize_cast_overflow_raises`, `test_python_usize_cast_uses_selectable_target_width`
- `pytest -q`: 360 passed, 4 skipped, 7 deselected

## N7.1 — Function-argument boundary checks

**Symptom:** A `usize`/`i64` value could be passed to a known function parameter of the other type without an explicit cast.
**Cause:** Boundary inference checked assignments and returns but did not compare call arguments with declared function parameter types.
**Fix:** Both backends collect function signatures before lowering calls and reject mismatched known argument types with the uniform SPEC 2.4.7 diagnostic; unknown and external signatures remain unchecked.
**Lesson:** Validate type boundaries at every statically known transfer point, not just assignments and returns.

- Covered by: `tests/test_n7_usize.py::test_python_call_argument_requires_explicit_cast`, `test_native_call_argument_requires_explicit_cast`, `test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.2 — Range loop variable types

**Symptom:** Range loop variables did not retain the range endpoint's integer type, allowing implicit `usize`/`i64` mixing in the loop body.
**Cause:** Loop bindings were not included in the backend boundary type state.
**Fix:** Both backends infer the range type from its endpoint and bind the loop variable to that type; `0..Vec::len()` is `usize`, while an `i64` endpoint produces an `i64` loop variable.
**Lesson:** A loop binding inherits the type of the range, and explicit casts are required when crossing integer boundaries.

- Covered by: `tests/test_n7_usize.py::test_python_range_variable_inherits_usize_type`, `test_native_range_variable_inherits_usize_type`, `test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.3 — Receiver-aware `len()` inference

**Symptom:** Every `.len()` call was inferred as `usize`, including user-defined methods returning `i64`.
**Cause:** Type inference recognized the method name but did not inspect the receiver or known impl signatures.
**Fix:** Vec and String lengths infer as `usize`; known user methods use their declared return type; other receivers remain unknown. Python lowering also preserves a user-defined `len` method while using the runtime length helper for other receivers.
**Lesson:** Method return types depend on the receiver; a method name alone is not sufficient type evidence.

- Covered by: `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`, `test_python_user_len_method_uses_declared_return_type`, `test_native_user_len_method_uses_declared_return_type`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.4 — Selectable Python target width

**Symptom:** Python `usize`/`isize` cast checks used the host width even when simulating a different compilation target.
**Cause:** Runtime width configuration was not connected to a backend instance.
**Fix:** The Python backend defaults to host pointer width and accepts an explicit `target_width` of 32 or 64 through its constructor or `get_backend("python", target_width=...)`; the selected width is scoped to compiled casts and does not mutate global runtime configuration. This option is Python-backend-only and does not alter native compilation behavior.
**Lesson:** Cross-target simulation must be explicit and isolated per backend instance so it cannot change default or native behavior.

- Covered by: `tests/test_n7_usize.py::test_python_backend_target_width_is_instance_scoped`, `test_python_usize_cast_uses_selectable_target_width`, `test_python_usize_cast_overflow_raises`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N8 — HashMap (collections layer)

**Symptom:** HashMap behavior existed in backend code but lacked a complete, traceable collections-layer contract; its `len()` result was not checked at the `usize`/`i64` boundary.
**Cause:** HashMap had not yet been included in the normative collection type set or N7's receiver-aware length inference, and native `get_mut` references cannot outlive the by-value map argument at the PyO3 boundary.
**Fix:** Specified and tested supported key/value types and operations, added HashMap `len()` inference and boundary checks, verified Python/native dictionary conversion, and made native `get_mut` return an owned copy at the Python boundary while retaining mutable internal use.
**Lesson:** Collection semantics include both language-level operations and explicit ownership/conversion rules at the host boundary.

- Covered by: `tests/test_n8_hashmap.py::test_python_hashmap_operations_and_option_results`, `test_native_hashmap_operations_and_option_results`, `test_python_hashmap_len_and_boundary_error`, `test_native_hashmap_len_and_boundary_error`, `test_python_hashmap_views_and_iteration`, `test_native_hashmap_views_and_iteration`, `test_python_hashmap_mutable_access_and_entry`, `test_native_hashmap_mutable_access_and_entry`, `test_python_hashmap_supported_key_value_types`, `test_native_hashmap_supported_key_value_types`, `test_hashmap_operations_match_across_backends`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected

## N9 — Error-handling boundary completion

**Symptom:** The Platform-defined Result and Option combinators lacked dedicated backend-specific acceptance tests, and native Result continuations were still marked Deferred.
**Cause:** Native errors are represented as `PyErr`, while Option-to-Result conversions needed to adapt their error values to the native exception representation.
**Fix:** Added separate Python/native tests for `is_ok`, `is_err`, `map_err`, `ok_or`, and `ok_or_else`; lowered native `ok_or*` errors to `PyErr`; verified native `and_then`/`or_else` with `PyErr` continuations and made them normative.
**Lesson:** Backend-specific error representations can remain explicit while combinators preserve their documented branch behavior.

- Covered by: `tests/test_n9_result.py::test_python_result_is_ok`, `test_native_result_is_ok`, `test_python_result_is_err`, `test_native_result_is_err`, `test_python_result_map_err`, `test_native_result_map_err`, `test_python_option_ok_or`, `test_native_option_ok_or`, `test_python_option_ok_or_else`, `test_native_option_ok_or_else`, `test_python_result_and_then_and_or_else`, `test_native_result_and_then_and_or_else`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected

## N10 — SPEC v0.2.7 and conformance status

**Symptom:** SPEC v0.2.4 still described the retired N7 transitional casts, Deferred HashMap and native Result continuations, and stale acceptance-test statuses.
**Cause:** The versioned specification had not recorded the completed N7–N9 behavior or the native HashMap `get_mut` boundary limitation.
**Fix:** Applied the N7 changes in v0.2.5, un-deferred HashMap and documented its native copy limitation in v0.2.6, and recorded the N9 combinator tests and native continuations in v0.2.7; updated 12.2 to 45 Pass and 6 Partial rows and synchronized the README test baseline.
**Lesson:** Conformance status must track tested behavior and versioned platform boundaries, including accepted limitations.

- Covered by: `tests/test_n7_usize.py`, `tests/test_n8_hashmap.py`, `tests/test_n9_result.py`, `tests/test_spec_conformance.py`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 400 deselected
- `ruff check --fix . && ruff check .`: passed

## M10 batch 1 — termination checking for `#[terminates]`

**Symptom:** Annotated functions had no static check for obvious unbounded loops or recursive calls that do not decrease.
**Cause:** The parser discarded attributes and neither backend performed a termination proof before lowering.
**Fix:** Preserved the `terminates` attribute in `FnDecl`, added a shared conservative checker, and invoke it before either backend lowers the function. The checker accepts explicit finite ranges, direct lower-bounded decreasing `while` variants, and guarded integer or structural slice recursion; unsupported calls, async suspension, parameter mutation/shadowing, and unbounded loops are rejected with one identical five-line `TerminationError`.
**Lesson:** A termination measure must remain well-founded through every recursive step; conservative rejection is safer than accepting based only on a syntactic subtraction.

- Covered by: `tests/test_m10_termination.py::test_terminates_attribute_is_parsed_on_and_off`, `test_fn_decl_terminates_flag_is_preserved_in_ast`, `test_checker_ignores_unannotated_functions`, `test_python_terminating_patterns_execute`, `test_native_terminating_patterns_execute`, `test_python_unannotated_recursion_compiles_without_running`, `test_native_unannotated_recursion_compiles_without_running`, `test_python_non_terminating_patterns_rejected`, `test_native_non_terminating_patterns_rejected`, `test_termination_error_message_is_backend_identical`
- `pytest -q`: 416 passed, 4 skipped, 7 deselected
- Deferred: mutual recursion, loop with break, complex arithmetic, non-recursive calls, async suspension

The checker also conservatively rejects non-recursive calls and async functions because their termination cannot be established by the batch-1 proof subset; both remain within SPEC 10.22.
## F1 — Deliverable single-file toolchain

- Symptom: The language could be developed and tested from a source checkout, but it was not packaged as an installable tool with a stable CLI, a short Quick Start, or runnable example files for a new user.
- Cause: The repository focused on language semantics and backend milestones; `pyproject.toml`, `api/cli.py`, README onboarding, and example delivery artifacts were incomplete or oriented around internal compile workflows.
- Fix: Added package metadata and a `polyglot` console script, expanded the CLI with `run`, `build`, and `check` (while preserving the compatibility `compile` command), rewrote the Quick Start, added seven runnable examples, documented F1 in SPEC v0.3.0, and added CLI delivery tests.
- Lesson: Delivery milestones need product-facing contracts (install, entrypoint, examples, and error format) to be specified and tested like language behavior; otherwise a working compiler core remains hard to adopt.
