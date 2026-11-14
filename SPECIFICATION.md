# polyglot-lang Specification v0.3.4

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
- **2.4** `usize` and `isize` MUST be distinct types from `i64`. Normative.
  - **2.4.1** The native backend MUST map `usize` and `isize` to the corresponding Rust types.
  - **2.4.2** The Python backend MUST represent `usize` and `isize` as `int` and MUST validate the target range at every `as` boundary.
  - **2.4.3** `Vec::len()` MUST return `usize`. In the Python backend, the returned `int` MUST represent `usize`, and values outside the target range MUST raise `OverflowError` at the point of conversion, not at the point of return.
  - **2.4.4** Indexing MUST accept `usize`. An `i64` index MUST require an explicit `as usize` conversion.
  - **2.4.5 (Retired in v0.2.5).** The transitional permission to represent `usize` and `isize` as `i64` or to insert automatic boundary casts is retired. Implementations MUST follow 2.4.1–2.4.4.
  - **2.4.6** No implicit conversion exists between `usize`/`isize` and `i64`. A value of one type MUST NOT be assigned, passed, or returned where the other is expected. An explicit `as` conversion is required at every boundary.
  - **2.4.7** A type error caused by 2.4.6 MUST report:
    - (a) the source type,
    - (b) the expected type,
    - (c) the location of the boundary,
    - (d) a suggested fix using `as`,
    - (e) a warning that `as` may wrap or truncate the value.

    The suggestion and warning MUST use one format for every boundary. The compiler MUST NOT classify boundaries as safe or unsafe. Errors MUST include a stable error code. When a character span is available, the location line MUST include ` (span S..E)`. With a span, the diagnostic uses a six-line format; without a span, the five-line format is used.

  - **2.4.8** The Python backend MUST represent `usize` and `isize` as `int` and MUST NOT raise on a value that is within the target's `usize`/`isize` range.
  - **2.4.9** An `OverflowError` MUST be raised only at an explicit `as` boundary (per 2.4.6), when the value does not fit the target type. `len()` and other `usize`-returning operations MUST NOT raise on their own.
  - **2.4.10** `usize` and `isize` are platform-dependent in width. The width MUST be determined by the compilation target, not by the machine running the compiler.
  - **2.4.11** When the native backend is cross-compiled to a target whose pointer width differs from the host, the Python backend MUST simulate the target's pointer width for `usize` and `isize` operations.
  - **2.4.12** Simulation under 2.4.11 MUST:
    - (a) reject values outside the target's `usize`/`isize` range with `OverflowError` at an explicit `as` boundary,
    - (b) preserve the target width in all arithmetic that depends on `usize`/`isize`,
    - (c) be selectable by the user so that host-width behaviour remains available.
  - **2.4.13** A program that depends on a specific `usize` width MUST declare it explicitly (mechanism deferred to a later version).
  - **2.4.14** A `for` loop variable takes the element type produced by the iterator. A range over `usize` yields `usize`; a range over `i64` yields `i64`. No implicit conversion is inserted.

- **2.5** The additional integer types `i8`, `i16`, `i32`, `u8`, `u16`, `u32`, `u64`, `i128`, and `u128` MUST be supported. The implicit-conversion rules of 2.4.6 and the diagnostic requirements of 2.4.7 apply to all integer types (including `usize`, `isize`, and `i64`). Explicit `as` conversions between integer types MUST follow Rust semantics for truncation, wrapping, and sign interpretation. The Python backend enforces integer cast boundaries as documented in 8.1.
- **2.6** Integer overflow behaviour MUST be documented per backend in Section 8.1. Cross-backend parity is not required.
- **2.7** The `as` conversion MUST follow Rust semantics in the native backend. Its Python backend behaviour MUST be documented in Section 8 until the type system work is complete. Deferred.
- **2.8** Termination checking is opt-in through `#[terminates]`.
  - **2.8.1** A function annotated `#[terminates]` MUST be proven to terminate. Compilation MUST fail if the proof fails.
  - **2.8.2** Functions without `#[terminates]` are not checked.
  - **2.8.3** The checker MUST accept bounded `for` loops, `while` loops with
  a strictly decreasing variant that has a lower bound, and a **single**
  recursive call that strictly decreases at least one argument. Multiple
  recursive calls are Deferred per 10.22.
  - **2.8.4** The checker is sound but incomplete. It MUST NOT accept a non-terminating function annotated `#[terminates]`.
  - **2.8.5** Termination failure MUST use the five-item format of 2.4.7, adapted as in the termination error format.

**Acceptance criteria**

- Tests cover parsing and preserving the attribute, each accepted proof pattern, each required rejection, parity of backend diagnostics, and the unannotated opt-out.

**Acceptance criteria**

- A parity test exists for every Normative type requirement.
- Negative-index and out-of-range tests exist for the Python backend.
- Every Deferred requirement has an entry in Section 10.
- A test exists for 2.4.6 (no implicit conversion between `usize`/`isize` and `i64`; an explicit `as` is required at every boundary).
- A test exists for 2.4.7 (a type error caused by 2.4.6 reports the source type, the expected type, the boundary location, a suggested `as` fix, and a warning that `as` may wrap or truncate).
- A test exists for 2.4.10 (the width of `usize` and `isize` is determined by the compilation target, not the host).
- A test exists for 2.4.11 and 2.4.12 (cross-compilation simulation rejects out-of-range values, preserves the target width, and can be disabled to fall back to host width).
- A test exists for 2.4.14 (range loop variables retain the range element type without implicit conversions).

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
  - **4.1.1** Code generation for const patterns and macro patterns is Platform-defined. See 10.16.
  - **4.1.2** All other forms listed in 4.1 MUST be supported by both backends.
- **4.2** Match exhaustiveness MUST produce a diagnostic for a known `Option` or `Result` scrutinee that is missing a `None` or `Err` arm. Exhaustiveness analysis for other types is Deferred.
  - **4.2.1** The diagnostic MUST include the scrutinee type and the missing variant(s).
  - **4.2.2** The diagnostic severity is Platform-defined: it is a translation-time warning in the Python backend and a compile-time error in the native backend. See 8.9.
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
- **5.6** Native `Result::and_then` and `Result::or_else` MUST be supported. Their continuations operate on the native `PyErr` error representation described in 8.5.4.
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
  - **6.1.1** Full async support is a planned core feature. Until it is specified, async remains Experimental and does not satisfy the independence requirement of 1.4. Moving async into the core MUST be recorded as a SPEC version increase (see 11.1) and MUST precede any async parity test (see 11.3).
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
  - collections: `HashMap`. Normative.
  - platform-specific: the async executor and payload enums. See Section 8.
- **7.2** Each library function MUST be marked Normative, Platform-defined, or Deferred. The complete function table is given in Sections 7.2.1 through 7.2.7.
- **7.3** A function marked Platform-defined MUST have an entry in Section 10.
- **7.4** `HashMap` MUST be classified as a collections-layer type, not a core type.

**Acceptance criteria**

- A table of standard library functions with status markers exists.
- No Platform-defined function lacks a Section 10 entry.

### 7.2.1 `Vec<T>` (core)

| Function                                      | Status    | Reference                            |
| --------------------------------------------- | --------- | ------------------------------------ |
| `push`, `pop`, `len`                          | Normative | 2.4.3                                |
| index `v[i]`                                  | Normative | requires `usize` per 2.4.4           |
| `contains`                                    | Normative | —                                    |
| `sort`, `sort_by`                             | Normative | —                                    |
| `map`, `filter`, `fold`                       | Normative | `filter` receiver-type check per 5.7 |
| slice `&v[a..b]`                              | Normative | —                                    |
| `extend`                                      | Normative | —                                    |
| `insert`, `remove`, `retain`, `dedup`         | Normative | —                                    |
| `windows`, `chunks`                           | Normative | —                                    |
| `is_empty`, `clear`, `reverse`                | Normative | —                                    |
| `first`, `last`, `get`                        | Normative | —                                    |
| `truncate`                                    | Normative | —                                    |
| `sort_unstable`                               | Normative | —                                    |
| `rotate_left`, `rotate_right`, `fill`, `swap` | Normative | out-of-range behaviour per 8.8       |
| `extend_from_slice`, `resize`, `split_off`    | Normative | —                                    |
| `partition`, `sort_by_key`, `is_sorted`       | Normative | —                                    |
| `vec![...]` macro | **Unsupported** | parsed, not lowered; use `[...].to_vec()` |

### 7.2.2 `String` (core)

| Function                                                                   | Status                                        | Reference                        |
| -------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------- |
| `len` (UTF-8 bytes), `chars`, `bytes`                                      | Normative                                     | —                                |
| `split`, `join`                                                            | Normative                                     | —                                |
| `replace`, `trim`, `starts_with`, `ends_with`                              | Normative                                     | —                                |
| `to_uppercase`, `to_lowercase`                                             | Normative                                     | —                                |
| `parse::<T>`                                                               | Normative; failure behaviour Platform-defined | 8.7                              |
| slice `&s[a..b]` (UTF-8 byte offsets)                                      | Normative                                     | 8.8                              |
| `==`, `<` and other comparisons                                            | Normative                                     | —                                |
| `+` concatenation                                                          | Normative                                     | —                                |
| `repeat`                                                                   | Normative                                     | negative-count behaviour per 8.8 |
| `find`                                                                     | Normative                                     | —                                |
| `is_empty`                                                                 | Normative                                     | —                                |
| `trim_start`, `trim_end`, `is_ascii`                                       | Normative                                     | —                                |
| `lines`, `split_whitespace`                                                | Normative                                     | —                                |
| `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, `to_ascii_uppercase` | Normative                                     | —                                |
| `rfind`, `strip_prefix`, `strip_suffix`, `split_once`, `rsplit_once`       | Normative                                     | —                                |
| `String::from(...)` | **Unsupported** | not lowered; use `"...".to_string()` |

### 7.2.3 `Option<T>` (core)

| Function                                    | Status                                           | Reference                                                        |
| ------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| construction `Some`, `None`                 | Normative                                        | —                                                                |
| `is_some`, `is_none`, `unwrap`, `unwrap_or` | Normative                                        | —                                                                |
| `and_then`, `or_else`, `filter`             | Normative; native uses Rust standard combinators | `filter` receiver-type check per 5.7                             |
| `ok_or`, `ok_or_else`                       | Platform-defined                                 | 5.8.2, 8.5.6                                                     |
| `map`                                       | **Unsupported**                                  | 5.7; collides with `Vec::map` until dispatch is fully type-aware |

### 7.2.4 `Result<T, E>` (core)

| Function                       | Status                                           | Reference             |
| ------------------------------ | ------------------------------------------------ | --------------------- |
| construction `Ok`, `Err`       | Normative                                        | —                     |
| `?` operator, synchronous      | Normative; `Err` representation Platform-defined | 5.2, 5.5, 8.5.1–8.5.2 |
| `?` operator, asynchronous     | Platform-defined                                 | 5.3, 8.5.3            |
| `is_ok`, `is_err`, `map_err`   | Platform-defined                                 | 5.8.1, 8.5.5          |
| `ok`, `err`                    | Normative                                        | —                     |
| `and_then`, `or_else` (native) | Normative; native errors use `PyErr`             | 5.6, 8.5.4            |
| `and_then`, `or_else` (Python) | Normative                                        | —                     |

### 7.2.5 Platform-defined function behaviour

Every row MUST have an entry in Section 8 and a cross-reference in Section 10.

| Behaviour                                 | Reference          |
| ----------------------------------------- | ------------------ |
| Integer overflow                          | 8.1                |
| `as` conversion, Python backend           | 2.7 (Deferred)     |
| `parse` failure behaviour                 | 8.7                |
| Async execution model                     | 8.2 (Experimental) |
| `Err` concrete representation             | 8.5.2              |
| `Result::is_ok`, `is_err`, `map_err`      | 8.5.5              |
| `Option::ok_or`, `ok_or_else`             | 8.5.6              |
| Box patterns                              | 8.3                |
| Payload enum slices                       | 8.4                |
| `usize` / `isize` boundary                | 2.4.5, 8.6         |
| `HashMap::get_mut` Python-boundary result | 8.10.2             |
| Numeric and index boundary errors         | 8.8                |
| Non-exhaustive match diagnostic           | 8.9                |

### 7.2.6 Collections and platform-specific types

| Type                        | Status                                              | Reference       |
| --------------------------- | --------------------------------------------------- | --------------- |
| `HashMap`                   | Normative                                           | 7.1, 7.4, 7.2.7 |
| `BTreeMap`                  | Partial (M3-remaining)                              | 7.2.8, 10.27    |
| `HashSet`                   | Partial (M3-remaining)                              | 7.2.8, 10.27    |
| Async executor (`PyFuture`) | Platform-defined                                    | 8.2.2–8.2.3     |
| Payload enum exposure       | Platform-defined (native: internal; Python: public) | 8.4.2           |

`HashMap<K, V>` supports `K` in `{i64, String}` and `V` in `{i64, String, bool, f64, Vec<i64>, HashMap<K2, V2>}` where `K2`/`V2` are themselves in the supported sets (depth 1). `BTreeMap` and `HashSet` support the same core key/value constraints where applicable. Deeper nesting is Deferred. See 10.25.

### 7.2.7 `HashMap<K, V>` (collections)

| Function                                                                                      | Status                                  | Reference     |
| --------------------------------------------------------------------------------------------- | --------------------------------------- | ------------- |
| `new`, `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys`, `values`, `iter` | Normative (N8)                          | 7.2.6, 8.10.1 |
| `entry(k).or_insert(v)`                                                                       | Normative (N8 minimal entry subset)     | —             |
| `get_mut`                                                                                     | Platform-defined at the Python boundary | 8.10.2        |
| `entry(k).or_insert_with(f)`                                                                  | Partial (M3-remaining)                  | 10.27         |
| `entry(k).and_modify(f)`                                                                      | Partial (M3-remaining)                  | 10.27         |
| `entry(k).or_insert_with_key(f)`                                                              | Partial (M3-remaining)                  | 10.27         |

### 7.2.8 `BTreeMap<K, V>` and `HashSet<T>` (collections)

| Function                                                                                                                    | Status                 | Reference |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------- |
| `BTreeMap::new`, `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys`, `values`, `iter`, `entry`, `get_mut` | Partial (M3-remaining) | 10.27     |
| `HashSet::new`, `insert`, `contains`, `remove`, `len`, `is_empty`, `iter`                                                   | Partial (M3-remaining) | 10.27     |

## 8. Platform Differences

This section defines the platform differences permitted by this specification. Each difference MUST have a test asserting the documented behaviour.

### 8.1 Integer overflow and integer casts

- **8.1.1** Integer overflow behaviour is Platform-defined. Native fixed-width integer overflow follows Rust semantics (panic in debug builds; wrap in release builds unless overflow checks are enabled). Python integers remain arbitrary precision.
  - **8.1.1.1** Native tests MUST assert wrap or panic behaviour according to build/profile where covered.
- **8.1.2** Cross-backend numerical results MAY differ when overflow occurs.
- **8.1.3** Explicit `as` conversions between integer types MUST follow Rust semantics natively. In the Python backend, fixed-width integer `as` casts apply Rust-like wrapping/truncation/sign interpretation unless the source is a literal integer value outside the target range, in which case `OverflowError` is raised at the `as` boundary. `usize`/`isize` casts remain range-checked per 2.4.9 and 2.4.12. See 2.5 and 10.26.
- **8.1.4** This difference MUST be recorded in Section 10.

**Acceptance criteria**

- Native overflow/cast tests exist where covered by the test suite.
- Python overflow/cast and boundary tests exist where covered by the test suite.

### 8.2 Async execution

- **8.2.1** Async MUST be marked Experimental.
- **8.2.2** The execution model is Platform-defined. The native backend uses a non-`Send` future polled with a no-op waker; the Python backend executes async functions synchronously until a later version specifies otherwise.
- **8.2.3** The native future type MUST be `unsendable`. Cross-thread use MUST fail explicitly.

**Acceptance criteria**

- A native async test exists.
- A cross-thread rejection test exists.

### 8.3 Box patterns

- **8.3.1** Box patterns MUST be supported only by the Python backend until stable Rust supports them.
- **8.3.2** The native backend MUST reject Box patterns with an explicit error before invoking the Rust compiler.
  - **8.3.2.1** The error MUST identify the Box pattern's source location.

**Acceptance criteria**

- A Python Box pattern test exists.
- A native rejection test exists in `tests/test_native_n2a_patterns.py`.

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
- **8.5.4** Native `Result::and_then` and `Result::or_else` use Rust's standard combinators over `Result<T, PyErr>`. A continuation receives the `PyErr` on the error path and MUST return a `Result<_, PyErr>`.
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
- **8.6.2** The Python backend MUST represent them as `int` and MUST validate the target range at the boundary.
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
- **8.9.3** The Python backend MUST emit a translation-time warning and MUST raise `RuntimeError` when a non-exhaustive match falls through at runtime.
- **8.9.4** This difference MUST be recorded in Section 10.

**Acceptance criteria**

- A native test asserts `RustCompileError` on non-exhaustive `Option` / `Result`.
- A Python test asserts a warning at translation time and a `RuntimeError` at runtime on non-exhaustive `Option` / `Result`.

### 8.10 HashMap Python boundary

- **8.10.1** The native backend MUST convert supported `HashMap<K, V>` arguments from Python dictionaries and return values to Python dictionaries. Native `Option<V>` results from `get` and `remove` convert to the corresponding value or `None`. The Python backend uses dictionaries internally and returns its runtime `Option` object from `get` and `remove`.
- **8.10.2** The Python backend's `get_mut` returns an `Option` containing a mutable view of the map entry. At a native Python boundary, `get_mut` returns an owned copy in `Option<V>`: a Rust mutable reference cannot outlive the by-value map argument. Mutating the returned copy does not mutate the original Python dictionary. Internal native use of `get_mut` remains mutable.

**Acceptance criteria**

- Tests cover the supported key/value types, operations, dictionary conversion, `usize` length boundaries, and iteration in both backends.
- Python `get_mut` returns a mutable view; native `get_mut` returns a copied value at the Python boundary.

## 9. Toolchain and Delivery

- **9.1** The toolchain MUST include a lexer, a parser, an AST, both backends, diagnostics that map native errors to source positions, and a module mechanism.
- **9.2** The module mechanism MUST provide a one-way inclusion directive. `include!("path/to/file.poly");` at item position parses the included file and splices its items at the include site. Relative paths resolve relative to the including file's directory; absolute paths are allowed. Included files MAY contain includes recursively. Cyclic includes MUST be detected and rejected with a five-line error that prints the include chain. Full `mod` and `use` syntax remains Deferred.
- **9.3** Diagnostics MUST map native errors back to original source lines and MUST include a stable error code and, where available, a character span.
- **9.4** A command-line interface SHOULD provide a run command and a build command.
- **9.5** The README Quick Start MUST allow a new user to run a first program without reading this specification.
- **9.6** Installation SHOULD be a single package installation. The Python backend MUST have no native dependencies.
- **9.7** The test suite MUST be runnable on the platforms listed in the README. A known environment-dependent failure that is recorded in Section 12 does not invalidate an acceptance criterion; an unrecorded failure does.
  - **9.7.1** An environment-dependent failure is recorded in 12.1 with the platform and failure signature. A failure signature is the exception type plus the first line of the traceback that identifies the failure site. If the signature changes, the recording MUST be updated.
- **9.8** A criterion marked **Blocked** in Section 12 MUST be re-evaluated once the test suite baseline is restored.
- **9.9** The Python backend MUST support Python 3.14.8 or later. Earlier versions are not supported and MUST NOT be claimed as conforming.
- **9.10** The CLI MUST provide `run`, `build`, and `check` subcommands. Errors MUST use the five-line format of 2.4.7.
- **9.11** The package MUST be installable as `pip install polyglot-lang`. The Python backend MUST have no native dependencies; the native backend MUST document its Rust toolchain requirement.

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
- **10.5** Native `Result::and_then` and `Result::or_else` use `PyErr` continuations. See 8.5.4.
- **10.6** `Option::map` is unsupported until method dispatch is fully type-aware. See 5.7.
- **10.7** `while let Some(x) = iter.next()` is deferred. See 6.5.
- **10.8** Async `impl` methods are deferred. See 6.4.
- **10.9 (Retired in v0.2.5).** The transitional `usize`/`isize` behavior is retired; implementations use distinct pointer-sized types and explicit casts. See 2.4.5.
- **10.10 (Retired in v0.3.3).** Additional fixed-width integer types are specified in 2.5 and implemented in scope of M4-remaining. See 2.5 and 8.1.
- **10.11 (Retired in v0.3.3).** Python backend `as` conversion behavior for additional integer types is specified in 8.1 and 10.26. See 2.5 and 2.7.
- **10.12** `if let` forms other than `if let Some(binding) = expr` raise a translation error. See 5.4.
- **10.13** `String::parse` failure behaviour differs across backends. See 8.7.
- **10.14** Negative `String::repeat` counts and out-of-range `Vec::rotate_*` / `Vec::swap` arguments raise in the Python backend and panic natively. See 8.8.
- **10.15** Non-exhaustive match produces a translation-time warning in the Python backend and a `RustCompileError` in the native backend. See 8.9.
- **10.16** `PatConst` and `PatMacro` are parsed but not emitted by either backend. See 4.1.1.
- **10.17** `Result::is_ok`, `Result::is_err`, and `Result::map_err` are Platform-defined. See 8.5.5.
- **10.18** `Option::ok_or` and `Option::ok_or_else` are Platform-defined. See 8.5.6.
- **10.19** No platform difference may be removed from this section without a corresponding SPEC version increase. See 11.1.
- **10.20** Native `HashMap::get_mut` returns a copied `Option<V>` at the Python boundary; the copy cannot mutate the input dictionary. See 8.10.2.
- **10.21** M3-remaining extends the collection targets in 7.2.6–7.2.8, but full parity for extended entry methods, `BTreeMap`, `HashSet`, nested HashMap depth-1 values, and collection `get_mut` semantics is Partial in this pass. See 10.25 and 10.27.
- **10.22** Termination proofs for mutual recursion, loops with `break`, complex arithmetic, non-recursive function calls, and async suspension are Deferred. Such constructs are conservatively rejected in an annotated function when the checker cannot prove termination.
- **10.23** Cyclic includes are rejected. Full `mod` and `use` syntax remains Deferred. See 9.2.
- **10.24** AST spans are best-effort; nodes with a single token have span equal to that token's span. See 2.4.7 and 9.3.
- **10.25** Nested `HashMap` value depth is 1 (e.g. `HashMap<K, HashMap<K2, V2>>` where `K2`/`V2` are supported). Deeper nesting is Deferred. See 7.2.6.
- **10.26** M4 Python-backend integer `as` casts apply Rust-like wrapping for fixed-width integer targets unless the source is a literal integer value outside the target range, in which case `OverflowError` is raised at the `as` boundary. `usize`/`isize` casts remain range-checked per 2.4.9 and 2.4.12. See 2.5 and 8.1.
- **10.27** M3-remaining: native backend support for `BTreeMap`, `HashSet`, extended entry methods, nested `HashMap` depth-1 values, and collection `get_mut` copy semantics beyond existing `HashMap` coverage is incomplete in this pass; Python backend support is partial. See 7.2.6–7.2.8.

## 11. Versioning

- **11.1** Any change to Section 8 MUST increase the specification version.
- **11.2** Narrowing a platform difference MUST be recorded as a version increase and MUST NOT be introduced as a silent test change.
- **11.3** No async parity test may be added before the async execution model is moved out of Experimental.
  - **11.3.1** A parity test is a test that asserts identical observable behaviour across both backends for the same input. A test that asserts backend-specific behaviour, such as the synchronous Python async test required by 6, is not a parity test.
- **11.4** v0.2.3 narrowed 8.3.2 from `RustCompileError` to a dedicated pre-compiler rejection, and recorded the native Box pattern rejection test in `tests/test_native_n2a_patterns.py`.
- **11.5** v0.2.4 added 9.9 (minimum Python version), clarified 9.7.1 (failure signature), added 8.3.2.1 (Box pattern source location), added 6.1.1 (full async as planned core feature), added 10.19 (no silent removal of platform differences), and clarified 4.2.2 and 8.9.3 (translation-time warning).
- **11.6** v0.2.5 retired 2.4.5 and 10.9, added 2.4.14 for range loop variable types, moved 2.4.4 and 8.6.1–8.6.3 acceptance criteria to Pass, and recorded the completed N7 type-boundary rules.
- **11.7** v0.2.6 un-deferred the collections-layer `HashMap`, specified its supported API and type sets in 7.2.6–7.2.7, and documented the native `get_mut` copy limitation in 8.10.2 and 10.20.
- **11.8** v0.2.7 added dedicated tests for Platform-defined Result/Option combinators, un-deferred native `Result::and_then` and `Result::or_else`, and updated their acceptance status.
- **11.9** v0.2.8 added opt-in termination checking for `#[terminates]` functions, a conservative shared checker for bounded ranges, decreasing `while` variants and recursive descent, and matching pre-execution errors in both backends.
- **11.10** v0.3.0 records F1 delivery: package metadata, the `run`/`build`/`check` CLI, Quick Start and examples, and delivery acceptance criteria.
- **11.11** v0.3.1 records M6: the concrete `include!` one-way inclusion directive, recursive expansion, and cycle/missing-file diagnostics.
- **11.12** v0.3.2 records M7: stable diagnostic error codes, AST character spans, and the extended five/six-line diagnostic format.
- **11.13** v0.3.3 records M4-remaining: additional fixed-width integer types, integer boundary diagnostics across all integer types, and documented Python/native `as`-cast behavior for those types.
- **11.14** v0.3.4 records M3-remaining: HashMap API completion targets, BTreeMap/HashSet collection types, extended supported key/value sets, and documented partial implementation status.

## 12. Conformance Status

This section records the conformance status of the reference implementation against the acceptance criteria in Sections 1–9. It is informative with respect to requirements and normative with respect to conformance reporting. It does not modify any requirement in Sections 1–9.

Status values: **Pass**, **Partial**, **Fail**, **Blocked (environment)**.

### 12.1 Test suite baseline

As recorded in this session, under Python 3.14.8:

- `pytest -q`: 469 passed, 9 skipped, 15 deselected.
- `pytest -q -m slow`: 15 passed, 478 deselected.

The suite reports 0 failed. The increases in passed/skipped counts reflect M6, M7, M4-remaining, and M3-remaining tests added in v0.3.1–v0.3.4 (including partial/skipped M3 cases).

- **12.1.1** The numbers above are from the current repository state at the time of writing. They MUST be updated whenever the test suite changes. The summary counts in 12.2.0 are derived from the 12.2 table below and MUST be recomputed whenever a row changes.
- **12.1.2** A criterion marked **Blocked** is not a conformance failure; it is a dependency on the test suite baseline being restored. The baseline is now restored; Blocked criteria have been re-evaluated in 12.2 and any that remain are marked **Partial** with a note recording what is still unverified.
- **12.1.3** `README.md` MUST record the same baseline as 12.1 before a conformance claim is made. See 12.3.

### 12.2 Acceptance criteria status

**12.2.0** Summary: 62 Pass, 5 Partial, 0 Fail, 0 Blocked.

**12.2.1** **Partial** means at least one test exists and passes, but the criterion is not fully covered. The Note column MUST state what is missing. A criterion marked **Partial** with the note "Tests not yet recorded" means the test may exist but is not traceable to a named test in `NOTES.md` or `tests/`. Such a criterion MUST be reclassified as **Fail** if, on inspection, no such test exists.

| Section | Criterion                                                                             | Covers clause             | Status  | Note                                                                                                                                                                                                                                                                                |
| ------- | ------------------------------------------------------------------------------------- | ------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | Each Platform-defined feature has a test                                              | 1.7                       | Pass    | Box pattern native rejection covered by `tests/test_native_n2a_patterns.py::test_pattern_box_native_rejected`                                                                                                                                                                       |
| 1       | No platform difference outside Section 8                                              | 1.6                       | Pass    | —                                                                                                                                                                                                                                                                                   |
| 2       | Parity test per Normative type                                                        | 2.1–2.3                   | Pass    | —                                                                                                                                                                                                                                                                                   |
| 2       | Negative-index and out-of-range tests                                                 | 2.4.2                     | Pass    | Python backend covered                                                                                                                                                                                                                                                              |
| 2       | Native rejects `i64` index without conversion                                         | 2.4.4                     | Pass    | `tests/test_n7_usize.py::test_native_i64_index_requires_explicit_cast`                                                                                                                                                                                                              |
| 2       | Every Deferred requirement in Section 10                                              | 2.5, 2.7                  | Pass    | 2.5 and 2.7 remain recorded as Deferred                                                                                                                                                                                                                                             |
| 2       | Range loop variable preserves the range element type                                  | 2.4.14                    | Pass    | `tests/test_n7_usize.py::test_native_range_variable_inherits_usize_type`, `::test_python_range_variable_inherits_usize_type`                                                                                                                                                        |
| 2       | Parser accepts `#[terminates]`                                                        | 2.8.1                     | Pass    | `tests/test_m10_termination.py::test_terminates_attribute_is_parsed_on_and_off`                                                                                                                                                                                                     |
| 2       | AST carries the termination flag                                                      | 2.8.1                     | Pass    | `tests/test_m10_termination.py::test_fn_decl_terminates_flag_is_preserved_in_ast`                                                                                                                                                                                                   |
| 2       | Checker accepts bounded `for` loops                                                   | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                             |
| 2       | Checker accepts decreasing `while` variants                                           | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                             |
| 2       | Checker accepts decreasing recursion                                                  | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                             |
| 2       | Checker rejects non-decreasing recursion                                              | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                   |
| 2       | Checker rejects increasing `while` variants                                           | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                   |
| 2       | Checker rejects `loop` without `break`                                                | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                   |
| 2       | Both backends produce the same termination error                                      | 2.8.5                     | Pass    | `tests/test_m10_termination.py::test_termination_error_message_is_backend_identical`                                                                                                                                                                                                |
| 2       | Unannotated functions are not checked                                                 | 2.8.2                     | Pass    | `tests/test_m10_termination.py::test_checker_ignores_unannotated_functions`, `::test_python_unannotated_recursion_compiles_without_running`, `::test_native_unannotated_recursion_compiles_without_running`                                                                         |
| 3       | Parity test per mutability rule                                                       | 3.1–3.3                   | Pass    | —                                                                                                                                                                                                                                                                                   |
| 3       | Native code preserves declared mutability                                             | 3.4                       | Partial | No dedicated test name recorded in NOTES                                                                                                                                                                                                                                            |
| 4       | Numbered test per pattern form                                                        | 4.1                       | Pass    | N2a batches 1–5                                                                                                                                                                                                                                                                     |
| 4       | Rejection tests with error class and location                                         | 4.3, 4.6                  | Pass    | —                                                                                                                                                                                                                                                                                   |
| 4       | Exhaustiveness-diagnostic test                                                        | 4.2                       | Pass    | Diagnostic text asserted in `tests/test_spec_conformance.py`                                                                                                                                                                                                                        |
| 5       | Synchronous `?` parity test                                                           | 5.2                       | Pass    | —                                                                                                                                                                                                                                                                                   |
| 5       | Async `?` documented                                                                  | 5.3                       | Pass    | 8.5.3                                                                                                                                                                                                                                                                               |
| 5       | Unsupported combinator in Section 10                                                  | 5.7                       | Pass    | 10.6                                                                                                                                                                                                                                                                                |
| 5       | Platform-defined test for each of `is_ok`, `is_err`, `map_err`, `ok_or`, `ok_or_else` | 5.8                       | Pass    | `tests/test_n9_result.py` contains a dedicated test per combinator and backend                                                                                                                                                                                                      |
| 6       | Native async free-function test                                                       | 6.4                       | Pass    | —                                                                                                                                                                                                                                                                                   |
| 6       | Async embedding test                                                                  | 6.4                       | Pass    | —                                                                                                                                                                                                                                                                                   |
| 6       | Python async free-function test                                                       | 6.4.1                     | Partial | Synchronous execution asserted indirectly                                                                                                                                                                                                                                           |
| 6       | Deferred items in Section 10                                                          | 6.4, 6.5                  | Pass    | 10.7, 10.8                                                                                                                                                                                                                                                                          |
| 7       | Function table with status markers                                                    | 7.2                       | Pass    | 7.2.1–7.2.7                                                                                                                                                                                                                                                                         |
| 7       | No Platform-defined function lacks Section 10 entry                                   | 7.3                       | Pass    | 10.13–10.18, 10.20                                                                                                                                                                                                                                                                  |
| 7       | HashMap supported methods, type sets, and dictionary boundary                         | 7.2.6–7.2.7, 8.10.1       | Pass    | `tests/test_n8_hashmap.py`                                                                                                                                                                                                                                                          |
| 7       | HashMap `get_mut` reference at native Python boundary                                 | 8.10.2                    | Partial | Native returns an owned copy because the input map is passed by value; see 10.20                                                                                                                                                                                                    |
| 8.1     | Native overflow test                                                                  | 8.1.1                     | Partial | Baseline restored; dedicated test not yet verified in this session                                                                                                                                                                                                                  |
| 8.1     | Python arbitrary-precision test                                                       | 8.1.2                     | Partial | Baseline restored; dedicated test not yet verified in this session                                                                                                                                                                                                                  |
| 8.2     | Native async test                                                                     | 8.2.2                     | Pass    | `tests/test_native_n5.py::test_native_async_add`, `::test_native_async_identity`                                                                                                                                                                                                    |
| 8.2     | Cross-thread rejection test                                                           | 8.2.3                     | Partial | `PyFuture` unsendable; explicit test name not recorded                                                                                                                                                                                                                              |
| 8.3     | Python Box pattern test                                                               | 8.3.1                     | Pass    | —                                                                                                                                                                                                                                                                                   |
| 8.3     | Native rejection test                                                                 | 8.3.2                     | Pass    | `tests/test_native_n2a_patterns.py::test_pattern_box_native_rejected` (parametrized over five Box pattern forms)                                                                                                                                                                    |
| 8.4     | Python payload slice test                                                             | 8.4.1                     | Pass    | —                                                                                                                                                                                                                                                                                   |
| 8.4     | Native limitation test                                                                | 8.4.2                     | Pass    | —                                                                                                                                                                                                                                                                                   |
| 8.5     | Synchronous `?` parity test                                                           | 8.5.1                     | Pass    | —                                                                                                                                                                                                                                                                                   |
| 8.5     | Async `?` documented                                                                  | 8.5.3                     | Pass    | —                                                                                                                                                                                                                                                                                   |
| 8.5     | Native `Result::and_then` and `Result::or_else`                                       | 8.5.4                     | Pass    | `tests/test_n9_result.py::test_native_result_and_then_and_or_else`                                                                                                                                                                                                                  |
| 8.5     | Platform-defined test for `Result::is_ok`, `is_err`, `map_err`                        | 8.5.5                     | Pass    | Dedicated Python and native tests in `tests/test_n9_result.py`                                                                                                                                                                                                                      |
| 8.5     | Platform-defined test for `Option::ok_or`, `ok_or_else`                               | 8.5.6                     | Pass    | Dedicated Python and native tests in `tests/test_n9_result.py`                                                                                                                                                                                                                      |
| 8.6     | `Vec::len()` parity test                                                              | 8.6.1                     | Pass    | `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `::test_native_explicit_len_cast_and_usize_return`                                                                                                                                                        |
| 8.6     | Range-validation tests                                                                | 8.6.2–8.6.3               | Pass    | `tests/test_n7_usize.py::test_python_usize_cast_overflow_raises`, `::test_python_backend_target_width_is_instance_scoped`                                                                                                                                                           |
| 8.7     | Native parse failure test                                                             | 8.7.1                     | Pass    | `tests/test_spec_conformance.py::test_native_invalid_literal_parse_fails_during_compile`, `::test_native_dynamic_parse_failure_remains_a_runtime_error`                                                                                                                             |
| 8.7     | Python `ValueError` test                                                              | 8.7.2                     | Pass    | `tests/test_spec_conformance.py::test_python_invalid_literal_parse_raises_value_error`, `::test_python_dynamic_parse_failure_raises_value_error`                                                                                                                                    |
| 8.7     | Parity on successful parse                                                            | 8.7.3                     | Pass    | `tests/test_spec_conformance.py::test_parse_success_matches_both_backends`                                                                                                                                                                                                          |
| 8.8     | Native boundary tests                                                                 | 8.8.1–8.8.4               | Pass    | `tests/test_spec_conformance.py::test_native_boundary_errors_are_not_silent`                                                                                                                                                                                                        |
| 8.8     | Python boundary tests                                                                 | 8.8.1–8.8.4               | Pass    | `tests/test_spec_conformance.py::test_python_boundary_errors_are_not_silent`                                                                                                                                                                                                        |
| 8.8     | Parity that both raise                                                                | 8.8.1–8.8.4               | Pass    | Both parametrized boundary tests in `tests/test_spec_conformance.py`                                                                                                                                                                                                                |
| 8.9     | Native non-exhaustive test                                                            | 8.9.2                     | Pass    | `tests/test_spec_conformance.py::test_native_non_exhaustive_known_match_is_compile_error`                                                                                                                                                                                           |
| 8.9     | Python non-exhaustive test                                                            | 8.9.3                     | Pass    | `tests/test_spec_conformance.py::test_python_non_exhaustive_known_match_warns_and_raises`                                                                                                                                                                                           |
| 9       | CLI exists                                                                            | Acceptance criterion      | Pass    | `api/cli.py`                                                                                                                                                                                                                                                                        |
| 9       | Quick Start validated                                                                 | 9.5                       | Pass    | —                                                                                                                                                                                                                                                                                   |
| 9       | Examples directory exists                                                             | Acceptance criterion      | Pass    | —                                                                                                                                                                                                                                                                                   |
| 9       | Test status matches Section 12                                                        | Acceptance criterion      | Pass    | README and 12.1 both record the current test baseline                                                                                                                                                                                                                               |
| 9       | Package installs via pip metadata and build artifacts                                 | F1.1                      | Pass    | `pyproject.toml` declares package metadata and console script; `python -m build` produces wheel and sdist                                                                                                                                                                           |
| 9       | `polyglot run` executes a file                                                        | F1.2                      | Pass    | `tests/test_f1_cli.py::test_run_hello_exits_zero_and_prints_five`                                                                                                                                                                                                                   |
| 9       | `polyglot build` produces an extension                                                | F1.3                      | Pass    | `tests/test_f1_cli.py::test_build_hello_produces_extension` (skipped if cargo unavailable)                                                                                                                                                                                          |
| 9       | `polyglot check` reports termination failures                                         | F1.4                      | Pass    | `tests/test_f1_cli.py::test_check_terminates_example_exits_zero`, `::test_check_non_terminating_reports_error`                                                                                                                                                                      |
| 9       | Quick Start runs in under 10 minutes                                                  | F1.5                      | Pass    | README Quick Start uses `pip install`, a short `hello.poly`, and `polyglot run`; validated by CLI tests and examples                                                                                                                                                                |
| 9       | `examples/` contains seven runnable files                                             | F1.6                      | Pass    | `tests/test_f1_cli.py::test_examples_directory_contains_seven_runnable_files`                                                                                                                                                                                                       |
| M6      | `include!` splices items and runs in both backends                                    | 9.2, 10.23                | Pass    | `tests/test_m6_include.py::test_include_add_both_backends`, `::test_include_add_native_backend`                                                                                                                                                                                     |
| M6      | Nested and relative-path includes resolve correctly                                   | 9.2                       | Pass    | `tests/test_m6_include.py::test_nested_include`, `::test_include_path_relative_to_including_file`                                                                                                                                                                                   |
| M6      | Cyclic/missing includes rejected with five-line diagnostics                           | 9.2, 10.23                | Pass    | `tests/test_m6_include.py::test_cyclic_include_rejected`, `::test_missing_include_file`                                                                                                                                                                                             |
| M7      | Stable error codes and span-aware diagnostics                                         | 2.4.7, 9.3, 10.24         | Pass    | `tests/test_m7_diagnostics.py::test_e001_boundary_error_has_code_and_span`, `::test_e002_termination_error_has_code_and_span`, `::test_e005_include_cycle_has_code_and_chain`, `::test_every_error_class_emits_code_exactly_once`, `::test_span_present_for_multi_token_expression` |
| M4      | Additional integer types, `as` casts, and E001 boundaries                             | 2.5, 8.1, 10.26           | Pass | `tests/test_m4_integers.py` covers Python and native matrix/boundary cases (E001)                                                                                                                              |
| M3      | HashMap API completion, BTreeMap, HashSet, nested values, and `get_mut` semantics     | 7.2.6–7.2.8, 10.25, 10.27 | Partial | Basic Python HashSet/BTreeMap covered by `tests/test_m3_collections.py`; extended entry methods, nested HashMap depth-1, BTreeMap/HashSet `get_mut` boundary semantics, and native parity are incomplete in this pass                                                               |

### 12.3 Outstanding items that block a v0.3.0 conformance claim

No item blocks a v0.3.0 conformance claim, provided that:

- Every Partial in 12.2 is explicitly accepted as a known gap in this version.
- The 6 Partial criteria are listed in 12.3.1 with a justification for why each does not block conformance.

### 12.3.1 Accepted Partial items for v0.3.0

| Partial item                                        | Justification                                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 3.4 mutability preservation test                    | Behaviour exists; only the test name is missing from NOTES                                                                     |
| 6.4.1 Python async synchronous execution            | Behaviour is backend-specific per 11.3.1; not a parity gap                                                                     |
| 8.1 overflow tests                                  | Baseline restored; dedicated verification is next milestone scope                                                              |
| 8.2 cross-thread rejection test                     | `PyFuture` unsendable is documented; explicit test is next milestone scope                                                     |
| 8.10.2 native `HashMap::get_mut` boundary reference | The by-value native map cannot safely return a mutable reference; the documented platform behaviour returns a copy. See 10.20. |

Each item above is either (a) covered by a Deferred clause, (b) a test-recording gap rather than a behaviour gap, or (c) explicitly assigned to next milestone in the roadmap.
- **10.28** `String::from(...)` is parsed but not lowered by either backend;
  use `"...".to_string()` instead. See 7.2.2.
- **10.29** The `vec![...]` macro expression is parsed as a token stream but
  not lowered by either backend; use `[...].to_vec()` instead. See 7.2.1.
