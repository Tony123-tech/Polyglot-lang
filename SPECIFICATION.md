# polyglot-lang Specification v0.2.2

## 0. Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document are to be interpreted as described in RFC 2119.

A requirement is marked Normative when it applies to both backends without platform-specific variation. A requirement is marked Platform-defined when Section 8 defines backend-specific behaviour. A requirement is marked Deferred when it is out of scope for the current version and tracked separately.

## 1. Positioning

- **1.1** polyglot-lang MUST be defined as a programming language, not as a Python extension.
- **1.2** The language MUST provide two backends: a Python transpilation backend and a Rust backend that emits PyO3 extension modules.
- **1.3** Language semantics MUST be defined by this specification and MUST NOT be defined by backend implementation details.
- **1.4** The language MUST be definable independently of Python. The Python backend is one execution mode of the language.
- **1.5** An official embedding API MAY be provided. Such an API MUST NOT alter core language semantics.
- **1.6** Every platform difference MUST be listed in Section 8. Silent platform differences are prohibited.
- **1.7** A single source file MUST be accepted by both backends unless it uses a feature marked Platform-defined in Section 8. Platform-defined features MUST produce an explicit error or a documented backend-specific result. They MUST NOT silently produce differing behaviour.

**Acceptance criteria**

- Each Platform-defined feature has a test asserting either an explicit error or the documented result.
- No platform difference exists outside Section 8.

## 2. Type System

- **2.1** The primitive types `i64`, `f64`, `bool`, and `String` MUST be available in both backends. Normative.
- **2.2** The container types `Vec<T>`, `Option<T>`, and `Result<T, E>` MUST be available in both backends. Normative.
- **2.3** The composite types `struct`, `enum`, and `impl` MUST be available in both backends. Normative.
- **2.4** `usize` and `isize` MUST be distinct types from `i64`. Deferred.
  - **2.4.1** The native backend MUST map `usize` and `isize` to the corresponding Rust types.
  - **2.4.2** The Python backend MUST represent `usize` and `isize` as `int` and MUST validate the platform range at the boundary.
  - **2.4.3** `Vec::len()` MUST return `usize`. In the Python backend, the returned `int` MUST represent `usize`, and values outside the platform range MUST raise `OverflowError`.
  - **2.4.4** Indexing MUST accept `usize`. An `i64` index MUST require an explicit `as usize` conversion.
  - **2.4.5 (Transitional)** Until the `usize` and `isize` work is complete, an implementation MAY represent `usize` and `isize` as `i64`. Such an implementation MUST NOT claim conformance with 2.4.1–2.4.4, and the deviation MUST be recorded in Section 10. Automatic insertion of `as i64` or `as usize` at code-generation boundaries is permitted under this clause but is not a conforming implementation of 2.4.3 or 2.4.4.
- **2.5** Additional integer types MAY be introduced after the `usize` and `isize` work is complete. Deferred.
- **2.6** Integer overflow behaviour MUST be documented per backend in Section 8.1. Cross-backend parity is not required.
- **2.7** The `as` conversion MUST follow Rust semantics in the native backend. Its Python backend behaviour MUST be documented in Section 8 until the type system work is complete. Deferred.

**Acceptance criteria**

- A parity test exists for every Normative type requirement.
- Negative-index and out-of-range tests exist for the Python backend.
- **Transitional:** while 2.4.5 is in force, the native backend is not required to reject an `i64` index without an explicit conversion. This criterion is satisfied by 8.8 boundary errors once 2.4.5 is retired.
- Every Deferred requirement has an entry in Section 10.

## 3. Mutability

- **3.1** Variables MUST be immutable by default.
- **3.2** `let mut` MUST declare a mutable binding.
- **3.3** Function parameters MUST preserve the mutability intent of the source. Generated code MUST reflect that intent.
- **3.4** The compiler MUST NOT change the declared mutability of a binding. Generated native code MAY include `#![allow(unused_mut)]`.

**Acceptance criteria**

- A parity test exists for each mutability rule.
- Generated native code preserves the declared mutability intent; this is verified by an explicit mutability-preservation test.

## 4. Pattern Matching

- **4.1** The parser MUST support the following pattern forms: or-patterns, literal ranges, tuple patterns, struct patterns, `@` bindings, reference patterns, slice patterns, rest patterns, const patterns, path patterns, and macro patterns.
  - **4.1.1** Code generation for const patterns and macro patterns is Platform-defined. See 10.18.
  - **4.1.2** All other forms listed in 4.1 MUST be supported by both backends.
- **4.2** Match exhaustiveness MUST produce a diagnostic for a known `Option` or `Result` scrutinee that is missing a `None` or `Err` arm. Exhaustiveness analysis for other types is Deferred.
  - **4.2.1** The diagnostic MUST include the scrutinee type and the missing variant(s).
  - **4.2.2** The diagnostic severity is Platform-defined: it is a warning in the Python backend and a compile error in the native backend. See 8.9.
- **4.3** Duplicate bindings and inconsistent bindings across or-pattern alternatives MUST be rejected with a source location.
- **4.4** Box patterns MUST follow Section 8.3.
- **4.5** Payload enum slices MUST follow Section 8.4.
- **4.6** A standalone `..` pattern MUST be rejected by the parser.

**Acceptance criteria**

- A numbered test exists for each supported pattern form.
- Each rejection rule has a test asserting the documented error class and source location.
- An exhaustiveness-diagnostic test asserts that the diagnostic names the scrutinee type and the missing variant(s), for both backends.

## 5. Error Handling

- **5.1** `Option<T>` and `Result<T, E>` MUST be available in both backends.
- **5.2** The `?` operator MUST have consistent propagation semantics across backends for synchronous functions. See Section 8.5.1. The concrete `Err` representation follows Section 8.5.2.
- **5.3** The behaviour of `?` inside an async function is Platform-defined. See Section 8.5.3.
- **5.4** `if let Some(binding) = expr` MUST be supported. Other if-let pattern forms MUST raise a translation error until they are implemented.
- **5.5** The concrete representation of `Err` is Platform-defined. See Section 8.5.2.
- **5.6** Native `Result::and_then` and `Result::or_else` MAY be deferred. See Section 8.5.4.
- **5.7** `Option::map` MUST remain unsupported until method dispatch is type-aware. `Vec::filter` and `Option::filter` are currently distinguished by a receiver-type check; this partial dispatch does not satisfy the condition for supporting `Option::map`, which collides with `Vec::map` at the call site.
- **5.8** The following combinators are Platform-defined:
  - **5.8.1** `Result::is_ok`, `Result::is_err`, and `Result::map_err`. See Section 8.5.5.
  - **5.8.2** `Option::ok_or` and `Option::ok_or_else`. See Section 8.5.6.

**Acceptance criteria**

- A parity test exists for synchronous `?`.
- The async behaviour of `?` is documented in Section 8.5.3.
- Each unsupported combinator has an entry in Section 10.
- A Platform-defined test exists for each of `Result::is_ok`, `Result::is_err`, `Result::map_err`, `Option::ok_or`, and `Option::ok_or_else`.

## 6. Async

- **6.1** Async MUST be marked Experimental and MUST NOT be part of the normative core of this specification. See Section 8.2.1.
- **6.2** The async execution model is Platform-defined. See Section 8.2.2.
- **6.3** The native future type MUST be documented as `unsendable`. See Section 8.2.3.
- **6.4** Async free functions MUST be supported in both backends. The native backend MUST use the language's own future type. Async `impl` methods are Deferred.
  - **6.4.1** Async free functions in the Python backend MUST execute synchronously until the async execution model is moved out of Experimental.
- **6.5** `while let Some(x) = iter.next()` is Deferred until a Rust-style `Option`-returning iterator adapter is specified.

**Acceptance criteria**

- A native async free-function test exists.
- An async embedding test exists.
- A Python async free-function test asserts synchronous execution. This test asserts backend-specific behaviour and is not a parity test; see 11.3.1.
- Each Deferred item is recorded in Section 10.

## 7. Standard Library Boundaries

- **7.1** The standard library MUST be divided into three layers:
  - core: `Vec`, `String`, `Option`, and `Result`. Normative.
  - collections: `HashMap`. Deferred.
  - platform-specific: the async executor and payload enums. See Section 8.
- **7.2** Each library function MUST be marked Normative, Platform-defined, or Deferred. The complete function table is given in Sections 7.2.1 through 7.2.6.
- **7.3** A function marked Platform-defined MUST have an entry in Section 10.
- **7.4** `HashMap` MUST be classified as a collections-layer type, not a core type.

**Acceptance criteria**

- A table of standard library functions with status markers exists.
- No Platform-defined function lacks a Section 10 entry.

### 7.2.1 `Vec<T>` (core)

| Function | Status | Reference |
|---|---|---|
| `push`, `pop`, `len` | Normative | 2.4.3 |
| index `v[i]` | Normative | requires `usize` per 2.4.4; transitional per 2.4.5 |
| `contains` | Normative | — |
| `sort`, `sort_by` | Normative | — |
| `map`, `filter`, `fold` | Normative | `filter` receiver-type check per 5.7 |
| slice `&v[a..b]` | Normative | — |
| `extend` | Normative | — |
| `insert`, `remove`, `retain`, `dedup` | Normative | — |
| `windows`, `chunks` | Normative | — |
| `is_empty`, `clear`, `reverse` | Normative | — |
| `first`, `last`, `get` | Normative | — |
| `truncate` | Normative | — |
| `sort_unstable` | Normative | — |
| `rotate_left`, `rotate_right`, `fill`, `swap` | Normative | out-of-range behaviour per 8.8 |
| `extend_from_slice`, `resize`, `split_off` | Normative | — |
| `partition`, `sort_by_key`, `is_sorted` | Normative | — |

### 7.2.2 `String` (core)

| Function | Status | Reference |
|---|---|---|
| `len` (UTF-8 bytes), `chars`, `bytes` | Normative | — |
| `split`, `join` | Normative | — |
| `replace`, `trim`, `starts_with`, `ends_with` | Normative | — |
| `to_uppercase`, `to_lowercase` | Normative | — |
| `parse::<T>` | Normative; failure behaviour Platform-defined | 8.7 |
| slice `&s[a..b]` (UTF-8 byte offsets) | Normative | 8.8 |
| `==`, `<` and other comparisons | Normative | — |
| `+` concatenation | Normative | — |
| `repeat` | Normative | negative-count behaviour per 8.8 |
| `find` | Normative | — |
| `is_empty` | Normative | — |
| `trim_start`, `trim_end`, `is_ascii` | Normative | — |
| `lines`, `split_whitespace` | Normative | — |
| `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, `to_ascii_uppercase` | Normative | — |
| `rfind`, `strip_prefix`, `strip_suffix`, `split_once`, `rsplit_once` | Normative | — |

### 7.2.3 `Option<T>` (core)

| Function | Status | Reference |
|---|---|---|
| construction `Some`, `None` | Normative | — |
| `is_some`, `is_none`, `unwrap`, `unwrap_or` | Normative | — |
| `and_then`, `or_else`, `filter` | Normative; native uses Rust standard combinators | `filter` receiver-type check per 5.7 |
| `ok_or`, `ok_or_else` | Platform-defined | 5.8.2, 8.5.6 |
| `map` | **Unsupported** | 5.7; collides with `Vec::map` until dispatch is fully type-aware |

### 7.2.4 `Result<T, E>` (core)

| Function | Status | Reference |
|---|---|---|
| construction `Ok`, `Err` | Normative | — |
| `?` operator, synchronous | Normative; `Err` representation Platform-defined | 5.2, 5.5, 8.5.1–8.5.2 |
| `?` operator, asynchronous | Platform-defined | 5.3, 8.5.3 |
| `is_ok`, `is_err`, `map_err` | Platform-defined | 5.8.1, 8.5.5 |
| `ok`, `err` | Normative | — |
| `and_then`, `or_else` (native) | **Deferred** | 5.6, 8.5.4 |
| `and_then`, `or_else` (Python) | Normative | — |

### 7.2.5 Platform-defined function behaviour

Every row MUST have an entry in Section 8 and a cross-reference in Section 10.

| Behaviour | Reference |
|---|---|
| Integer overflow | 8.1 |
| `as` conversion, Python backend | 2.7 (Deferred) |
| `parse` failure behaviour | 8.7 |
| Async execution model | 8.2 (Experimental) |
| `Err` concrete representation | 8.5.2 |
| `Result::is_ok`, `is_err`, `map_err` | 8.5.5 |
| `Option::ok_or`, `ok_or_else` | 8.5.6 |
| Box patterns | 8.3 |
| Payload enum slices | 8.4 |
| `usize` / `isize` boundary | 2.4.5, 8.6 |
| Numeric and index boundary errors | 8.8 |
| Non-exhaustive match diagnostic | 8.9 |

### 7.2.6 Collections and platform-specific types

| Type | Status | Reference |
|---|---|---|
| `HashMap` | Deferred | 7.1, 7.4 |
| Async executor (`PyFuture`) | Platform-defined | 8.2.2–8.2.3 |
| Payload enum exposure | Platform-defined (native: internal; Python: public) | 8.4.2 |

## 8. Platform Differences

This section defines the platform differences permitted by this specification. Each difference MUST have a test asserting the documented behaviour.

### 8.1 Integer overflow

- **8.1.1** Native `i64` overflow MUST follow Rust semantics: panic in debug builds and wrap in release builds.
  - **8.1.1.1** Native tests MUST assert wrap behaviour under `--release` and panic behaviour under the default debug profile, or under `--release` with `overflow-checks = true`.
- **8.1.2** Python integers MUST remain arbitrary precision.
- **8.1.3** Cross-backend numerical results MAY differ when overflow occurs.
- **8.1.4** This difference MUST be recorded in Section 10.

**Acceptance criteria**

- A native test asserts panic or wrap according to build profile.
- A Python test asserts arbitrary-precision behaviour.

### 8.2 Async execution

- **8.2.1** Async MUST be marked Experimental.
- **8.2.2** The execution model is Platform-defined. The native backend uses a non-`Send` future polled with a no-op waker; the Python backend executes async functions synchronously until a later version specifies otherwise.
- **8.2.3** The native future type MUST be `unsendable`. Cross-thread use MUST fail explicitly.

**Acceptance criteria**

- A native async test exists.
- A cross-thread rejection test exists.

### 8.3 Box patterns

- **8.3.1** Box patterns MUST be supported only by the Python backend until stable Rust supports them.
- **8.3.2** The native backend MUST reject Box patterns with an explicit error. Until a dedicated native rejection test exists, a Box pattern in native source MUST raise `RustCompileError`; this behaviour is tracked in 10.13.

**Acceptance criteria**

- A Python Box pattern test exists.
- **Partial:** a native rejection test does not yet exist. A native Box pattern is currently skipped in the test suite. See 10.13.

### 8.4 Payload enum slices

- **8.4.1** Payload enum slices MUST be supported only by the Python backend until PyO3 exposes payload enums.
- **8.4.2** Native payload enums MUST remain internal and MUST NOT be exposed as Python classes.

**Acceptance criteria**

- A Python payload slice test exists.
- A native limitation test exists.

### 8.5 Result representation

- **8.5.1** `?` propagation MUST be consistent across backends for synchronous functions.
- **8.5.2** The concrete `Err` representation is Platform-defined: the native backend uses `PyErr`; the Python backend uses its runtime `Result` object.
  - **8.5.2.1** The Python backend `Result<T, E>` MUST be a class with `Ok(value)` and `Err(error)` constructors.
- **8.5.3** The behaviour of `?` inside async functions is Platform-defined.
- **8.5.4** Native `Result::and_then` and `Result::or_else` MAY be deferred.
- **8.5.5** `Result::is_ok`, `Result::is_err`, and `Result::map_err` are Platform-defined.
  - **8.5.5.1** The Python backend MUST support them on its runtime `Result` class.
  - **8.5.5.2** The native backend MUST support them via `PyErr`.
- **8.5.6** `Option::ok_or` and `Option::ok_or_else` are Platform-defined.
  - **8.5.6.1** The Python backend MUST support them on its runtime `Option` class.
  - **8.5.6.2** The native backend MUST support them where the `Err` type can be inferred; otherwise it MUST raise `RustCompileError`.

**Acceptance criteria**

- A synchronous `?` parity test exists.
- The async `?` behaviour is documented.
- Each deferred combinator is listed in Section 10.
- A Platform-defined test exists for each of `Result::is_ok`, `Result::is_err`, `Result::map_err`, `Option::ok_or`, and `Option::ok_or_else`.

### 8.6 `usize` and `isize` boundary

- **8.6.1** `usize` and `isize` MUST be distinct language types.
- **8.6.2** The Python backend MUST represent them as `int` and MUST validate the platform range at the boundary.
- **8.6.3** Out-of-range values MUST raise `OverflowError`.

**Acceptance criteria**

- A parity test exists for `Vec::len()`.
- Range-validation tests exist for the Python backend.

### 8.7 String parsing failure

- **8.7.1** Native `parse::<i64>()` uses Rust's typed parse and MUST surface a parse failure as a `RustCompileError` at compile time when the input is statically known to be invalid, or as a runtime error otherwise.
- **8.7.2** Python `parse::<i64>()` lowers to `int()` and MUST raise `ValueError` on invalid input.
- **8.7.3** Successful parsing produces the same numeric result in both backends.
- **8.7.4** This difference MUST be recorded in Section 10.

**Acceptance criteria**

- A native test asserts the documented failure behaviour.
- A Python test asserts `ValueError` on invalid input.
- A parity test asserts identical successful results.

### 8.8 Numeric and index boundary errors

- **8.8.1** `String::repeat` with a negative count MUST raise an explicit error in both backends. Native MAY rely on Rust's behaviour; the Python backend MUST raise `ValueError`.
- **8.8.2** `Vec::rotate_left` and `Vec::rotate_right` with an out-of-range count MUST raise an explicit error in the Python backend and MUST follow Rust's panic behaviour natively.
- **8.8.3** `Vec::swap` with an out-of-range index MUST raise an explicit error in the Python backend and MUST follow Rust's panic behaviour natively.
- **8.8.4** String slicing at a non-UTF-8 boundary MUST raise in both backends; it MUST NOT silently change indexing semantics.
- **8.8.5** These differences MUST be recorded in Section 10.

**Acceptance criteria**

- A native test exists for each boundary case.
- A Python test exists for each boundary case.
- A parity test asserts that both backends raise rather than silently differing.

### 8.9 Non-exhaustive match diagnostic

- **8.9.1** A match expression that does not cover all possible values of a known `Option` or `Result` scrutinee MUST produce the diagnostic required by 4.2.
- **8.9.2** The native backend MUST follow Rust's compile-time exhaustiveness check. A non-exhaustive match in native source MUST be a `RustCompileError`, not a warning.
- **8.9.3** The Python backend MUST emit a warning and MUST raise `RuntimeError` when a non-exhaustive match falls through at runtime.
- **8.9.4** This difference MUST be recorded in Section 10.

**Acceptance criteria**

- A native test asserts `RustCompileError` on non-exhaustive `Option` / `Result`.
- A Python test asserts a warning at translation time and a `RuntimeError` at runtime on non-exhaustive `Option` / `Result`.

## 9. Toolchain and Delivery

- **9.1** The toolchain MUST include a lexer, a parser, an AST, both backends, diagnostics that map native errors to source positions, and a module mechanism.
- **9.2** The module mechanism SHOULD be a one-way inclusion directive. Full module and import syntax is Deferred.
- **9.3** Diagnostics MUST map native errors back to original source lines.
- **9.4** A command-line interface SHOULD provide a run command and a build command.
- **9.5** The README Quick Start MUST allow a new user to run a first program without reading this specification.
- **9.6** Installation SHOULD be a single package installation. The Python backend MUST have no native dependencies.
- **9.7** The test suite MUST be runnable on the platforms listed in the README. A known environment-dependent failure that is recorded in Section 12 does not invalidate an acceptance criterion; an unrecorded failure does.
  - **9.7.1** An environment-dependent failure is recorded in 12.1 with the platform and failure signature. If the signature changes, the recording MUST be updated.
- **9.8** A criterion marked **Blocked** in Section 12 MUST be re-evaluated once the test suite baseline is restored.

**Acceptance criteria**

- A command-line interface exists.
- The Quick Start is validated.
- An examples directory exists.
- The test suite status recorded in Section 12 is the authoritative source; `README.md` MUST be updated to match Section 12 before a conformance claim is made.

## 10. Known Limitations

This section is normative. Each entry MUST reference the requirement or platform difference it qualifies.

- **10.1** Integer overflow differs across backends. See 8.1.4.
- **10.2** Async is Experimental and the native future type is `unsendable`. See 8.2.3.
- **10.3** Box patterns are supported only by the Python backend. See 8.3.
- **10.4** Payload enum slices are supported only by the Python backend; native payload enums are internal. See 8.4.
- **10.5** Native `Result::and_then` and `Result::or_else` are deferred. See 8.5.4.
- **10.6** `Option::map` is unsupported until method dispatch is fully type-aware. See 5.7.
- **10.7** `while let Some(x) = iter.next()` is deferred. See 6.5.
- **10.8** Async `impl` methods are deferred. See 6.4.
- **10.9** `usize` and `isize` are not yet distinct from `i64`; the native backend currently emits automatic `as i64` / `as usize` conversions at code-generation boundaries. See 2.4.5.
- **10.10** Additional integer types beyond `i64`, `usize`, and `isize` are deferred. See 2.5.
- **10.11** The `as` conversion follows Rust semantics natively; its Python backend behaviour is not yet specified and is deferred. See 2.7.
- **10.12** `if let` forms other than `if let Some(binding) = expr` raise a translation error. See 5.4.
- **10.13** A native Box pattern rejection test does not yet exist; Box patterns are skipped in the native test suite. See 8.3.
- **10.14** `String::parse` failure behaviour differs across backends. See 8.7.
- **10.15** Negative `String::repeat` counts and out-of-range `Vec::rotate_*` / `Vec::swap` arguments raise in the Python backend and panic natively. See 8.8.
- **10.16** Non-exhaustive match produces a warning in the Python backend and a `RustCompileError` in the native backend. See 8.9.
- **10.17** `PatConst` and `PatMacro` are parsed but not emitted by either backend. See 4.1.1.
- **10.18** `Result::is_ok`, `Result::is_err`, and `Result::map_err` are Platform-defined. See 8.5.5.
- **10.19** `Option::ok_or` and `Option::ok_or_else` are Platform-defined. See 8.5.6.

## 11. Versioning

- **11.1** Any change to Section 8 MUST increase the specification version.
- **11.2** Narrowing a platform difference MUST be recorded as a version increase and MUST NOT be introduced as a silent test change.
- **11.3** No async parity test may be added before the async execution model is moved out of Experimental.
  - **11.3.1** A parity test is a test that asserts identical observable behaviour across both backends for the same input. A test that asserts backend-specific behaviour, such as the synchronous Python async test required by 6, is not a parity test.

## 12. Conformance Status

This section records the conformance status of the reference implementation against the acceptance criteria in Sections 1–9. It is normative for reporting purposes only; it does not modify any requirement.

Status values: **Pass**, **Partial**, **Fail**, **Blocked (environment)**.

### 12.1 Test suite baseline

As recorded at the latest batch, under Python 3.14.8 (the declared minimum):

- `pytest -q`: 350 passed, 4 skipped, 7 deselected.
- `pytest -q -m slow`: 7 passed.

The previously recorded failures (three inline annotation `NameError`s and one native extension-loading `_Py_IsFinalizing` failure) do not reproduce under Python 3.14.8. The suite reports 0 failed.

- **12.1.1** The numbers above are from the current repository state at the time of writing. They MUST be updated whenever the test suite changes.
- **12.1.2** A criterion marked **Blocked** is not a conformance failure; it is a dependency on the test suite baseline being restored. The baseline is now restored; Blocked criteria have been re-evaluated in 12.2 and any that remain are marked **Partial** with a note recording what is still unverified.
- **12.1.3** `README.md` MUST record the same baseline as 12.1 before a conformance claim is made. See 12.3.

### 12.2 Acceptance criteria status

**12.2.0** Summary: 33 Pass, 18 Partial, 1 Fail, 1 Blocked.

**12.2.1** **Partial** means at least one test exists and passes, but the criterion is not fully covered. The Note column MUST state what is missing.

| Section | Criterion | Status | Note |
|---|---|---|---|
| 1 | Each Platform-defined feature has a test | Partial | A parametrized native Box pattern rejection test is reported to exist in `tests/test_native_n2a_patterns.py`; exact test name pending confirmation in this session. See 10.13 |
| 1 | No platform difference outside Section 8 | Pass | — |
| 2 | Parity test per Normative type | Pass | — |
| 2 | Negative-index and out-of-range tests | Pass | Python backend covered |
| 2 | Native rejects `i64` index without conversion | Partial | 2.4.5 transitional; see 10.9. Not required while 2.4.5 is in force |
| 2 | Every Deferred requirement in Section 10 | Pass | 2.5, 2.7, 2.4 recorded |
| 3 | Parity test per mutability rule | Pass | — |
| 3 | Native code preserves declared mutability | Partial | No dedicated test name recorded in NOTES |
| 4 | Numbered test per pattern form | Pass | N2a batches 1–5 |
| 4 | Rejection tests with error class and location | Pass | — |
| 4 | Exhaustiveness-diagnostic test | Pass | Diagnostic text asserted in `tests/test_spec_conformance.py` |
| 5 | Synchronous `?` parity test | Pass | — |
| 5 | Async `?` documented | Pass | 8.5.3 |
| 5 | Unsupported combinator in Section 10 | Pass | 10.6 |
| 5 | Platform-defined test for each of `is_ok`, `is_err`, `map_err`, `ok_or`, `ok_or_else` | Partial | Tests not yet recorded |
| 6 | Native async free-function test | Pass | — |
| 6 | Async embedding test | Pass | — |
| 6 | Python async free-function test | Partial | Synchronous execution asserted indirectly |
| 6 | Deferred items in Section 10 | Pass | 10.7, 10.8 |
| 7 | Function table with status markers | Pass | 7.2.1–7.2.6 |
| 7 | No Platform-defined function lacks Section 10 entry | Pass | 10.14–10.19 |
| 8.1 | Native overflow test | Partial | Baseline restored; dedicated test not yet verified in this session |
| 8.1 | Python arbitrary-precision test | Partial | Baseline restored; dedicated test not yet verified in this session |
| 8.2 | Native async test | Pass | `tests/test_native_n5.py::test_native_async_add`, `::test_native_async_identity` |
| 8.2 | Cross-thread rejection test | Partial | `PyFuture` unsendable; explicit test name not recorded |
| 8.3 | Python Box pattern test | Pass | — |
| 8.3 | Native rejection test | Partial | A parametrized rejection test is reported to exist in `tests/test_native_n2a_patterns.py`; exact test name pending confirmation in this session. See 10.13 |
| 8.4 | Python payload slice test | Pass | — |
| 8.4 | Native limitation test | Pass | — |
| 8.5 | Synchronous `?` parity test | Pass | — |
| 8.5 | Async `?` documented | Pass | — |
| 8.5 | Deferred combinator in Section 10 | Pass | 10.5 |
| 8.5 | Platform-defined test for `Result::is_ok`, `is_err`, `map_err` | Partial | Tests not yet recorded |
| 8.5 | Platform-defined test for `Option::ok_or`, `ok_or_else` | Partial | Tests not yet recorded |
| 8.6 | `Vec::len()` parity test | Partial | Baseline restored; dedicated test not yet verified in this session |
| 8.6 | Range-validation tests | Partial | `OverflowError` behaviour not yet exercised |
| 8.7 | Native parse failure test | Pass | `tests/test_spec_conformance.py::test_native_invalid_literal_parse_fails_during_compile`, `::test_native_dynamic_parse_failure_remains_a_runtime_error` |
| 8.7 | Python `ValueError` test | Pass | `tests/test_spec_conformance.py::test_python_invalid_literal_parse_raises_value_error`, `::test_python_dynamic_parse_failure_raises_value_error` |
| 8.7 | Parity on successful parse | Pass | `tests/test_spec_conformance.py::test_parse_success_matches_both_backends` |
| 8.8 | Native boundary tests | Pass | `tests/test_spec_conformance.py::test_native_boundary_errors_are_not_silent` |
| 8.8 | Python boundary tests | Pass | `tests/test_spec_conformance.py::test_python_boundary_errors_are_not_silent` |
| 8.8 | Parity that both raise | Pass | Both parametrized boundary tests in `tests/test_spec_conformance.py` |
| 8.9 | Native non-exhaustive test | Pass | `tests/test_spec_conformance.py::test_native_non_exhaustive_known_match_is_compile_error` |
| 8.9 | Python non-exhaustive test | Pass | `tests/test_spec_conformance.py::test_python_non_exhaustive_known_match_warns_and_raises` |
| 9 | CLI exists | Pass | `api/cli.py` |
| 9 | Quick Start validated | Pass | — |
| 9 | Examples directory exists | Pass | — |
| 9 | Test status matches Section 12 | Pass | README and 12.1 both record 350 passed, 4 skipped, 7 deselected |

### 12.3 Outstanding items that block a v0.2 conformance claim

| # | Item | Priority |
|---|---|---|
| 1 | Confirm the exact test name of the parametrized native Box pattern rejection test in `tests/test_native_n2a_patterns.py`, then remove 10.13 and the 8.3 Fail note | P1 |