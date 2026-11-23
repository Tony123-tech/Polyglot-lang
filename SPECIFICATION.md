# polyglot-lang Specification v0.3.8

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
    a strictly decreasing variant that has a lower bound, and self-recursive
    calls that each strictly decrease at least one argument under a lower-bound
    guard. Multiple self-recursive calls are accepted; mutual recursion remains
    Deferred per 10.22.
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
- **5.4** `if let` MUST be supported for the builtin `Some`, `None`, `Ok`, and `Err` patterns in both statement and expression positions (including `let` initialisers and call arguments). Other if-let pattern forms MUST raise a translation error until they are implemented.
- **5.5** The concrete representation of `Err` is Platform-defined. See Section 8.5.2.
- **5.6** Native `Result::and_then` and `Result::or_else` MUST be supported. Their continuations operate on the native `PyErr` error representation described in 8.5.4.
- **5.7** Method dispatch MUST be type-aware. `Option` and `Result` receivers—including `Some`/`None`/`Ok`/`Err` expressions, annotated bindings, `pop`/`next` results, and free calls whose declared return type is `Option` or `Result`—MUST lower `map` to the matching combinator. `Vec` receivers MUST retain collection semantics. See 7.2.1, 7.2.3, and 7.2.4.
- **5.8** The following combinators are Platform-defined:
  - **5.8.1** `Result::is_ok`, `Result::is_err`, and `Result::map_err`. See Section 8.5.5.
  - **5.8.2** `Option::ok_or` and `Option::ok_or_else`. See Section 8.5.6.

**Acceptance criteria**

- A parity test exists for synchronous `?`.
- The async behaviour of `?` is documented in Section 8.5.3.
- A parity test exists for type-aware `Option::map`, `Result::map`, and `Vec::map` dispatch.
- Each combinator marked Platform-defined or Unsupported has an entry in Section 10.
- A Platform-defined test exists for each of `Result::is_ok`, `Result::is_err`, `Result::map_err`, `Option::ok_or`, and `Option::ok_or_else`.

## 6. Async

- **6.1** Async MUST be marked Experimental and MUST NOT be part of the normative core of this specification. See Section 8.2.1.
  - **6.1.1** Full async support is a planned core feature. Until it is specified, async remains Experimental and does not satisfy the independence requirement of 1.4. Moving async into the core MUST be recorded as a SPEC version increase (see 11.1) and MUST precede any async parity test (see 11.3).
- **6.2** The async execution model is Platform-defined. See Section 8.2.2.
- **6.3** The native future type MUST be documented as Send-aware: a future whose captured values are Send MUST be usable across threads. See Section 8.2.3.
- **6.4** Async free functions MUST be supported in both backends. The native backend MUST use the language's own future type. Async `impl` methods with `&self` or `&mut self` receivers MUST be supported; by-value receivers are Deferred. For `&mut self`, the native backend MUST use an atomic borrow guard, copy mutable fields before the future runs, and commit field writes only when the future completes successfully; overlapping calls MUST be rejected.
  - **6.4.1** Async free functions in the Python backend MUST execute synchronously until the async execution model is moved out of Experimental.
- **6.5** `while let Some(x) = iter.next()` is Deferred until a Rust-style `Option`-returning iterator adapter is specified.

**Acceptance criteria**

- A native async free-function test exists.
- An async embedding test exists.
- A native async `&self` impl-method test exists.
- A native async `&mut self` impl-method test verifies the borrow guard and commit-on-success lowering.
- A Python async free-function test asserts synchronous execution. This test asserts backend-specific behaviour and is not a parity test; see 11.3.1.
- Each Deferred item is recorded in Section 10.

## 7. Standard Library Boundaries

- **7.1** The standard library MUST be divided into three layers:
  - core: `Vec`, `String`, `Option`, and `Result`. Normative.
  - collections: `HashMap`. Normative.
  - platform-specific: the async executor and payload enums. See Section 8.
- **7.2** Each library function MUST be marked Normative, Platform-defined, or Deferred. The complete function table is given in Sections 7.2.1 through 7.2.8.
- **7.3** A function marked Platform-defined MUST have an entry in Section 10.
- **7.4** `HashMap` MUST be classified as a collections-layer type, not a core type.

**Acceptance criteria**

- A table of standard library functions with status markers exists.
- No Platform-defined function lacks a Section 10 entry.

### 7.2.1 `Vec<T>` (core)

| Function                                      | Status    | Reference                                         |
| --------------------------------------------- | --------- | ------------------------------------------------- |
| `push`, `pop`, `len`                          | Normative | 2.4.3                                             |
| index `v[i]`                                  | Normative | requires `usize` per 2.4.4                        |
| `contains`                                    | Normative | —                                                 |
| `sort`, `sort_by`                             | Normative | —                                                 |
| `map`, `filter`, `fold`                       | Normative | type-aware dispatch per 5.7                       |
| slice `&v[a..b]`                              | Normative | —                                                 |
| `extend`                                      | Normative | —                                                 |
| `insert`, `remove`, `retain`, `dedup`         | Normative | —                                                 |
| `windows`, `chunks`                           | Normative | —                                                 |
| `is_empty`, `clear`, `reverse`                | Normative | —                                                 |
| `first`, `last`, `get`                        | Normative | —                                                 |
| `truncate`                                    | Normative | —                                                 |
| `sort_unstable`                               | Normative | —                                                 |
| `rotate_left`, `rotate_right`, `fill`, `swap` | Normative | out-of-range behaviour per 8.8                    |
| `extend_from_slice`, `resize`, `split_off`    | Normative | —                                                 |
| `partition`, `sort_by_key`, `is_sorted`       | Normative | —                                                 |
| `vec![...]` macro                             | Normative | lowered to the shared array AST; both forms below |
| `vec![a, b, ...]` / `vec![value; count]`      | Normative | `vec![value; count]` requires a `usize` count     |

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
| `String::from(...)`                                                        | Normative                                     | —                                |

### 7.2.3 `Option<T>` (core)

| Function                                    | Status                                           | Reference                   |
| ------------------------------------------- | ------------------------------------------------ | --------------------------- |
| construction `Some`, `None`                 | Normative                                        | —                           |
| `is_some`, `is_none`, `unwrap`, `unwrap_or` | Normative                                        | —                           |
| `unwrap_or_else`                            | Normative; the fallback closure is lazy          | —                           |
| `and_then`, `or_else`, `filter`             | Normative; native uses Rust standard combinators | type-aware dispatch per 5.7 |
| `map`                                       | Normative                                        | type-aware dispatch per 5.7 |
| `ok_or`, `ok_or_else`                       | Platform-defined                                 | 5.8.2, 8.5.6                |

### 7.2.4 `Result<T, E>` (core)

| Function                       | Status                                           | Reference                   |
| ------------------------------ | ------------------------------------------------ | --------------------------- |
| construction `Ok`, `Err`       | Normative                                        | —                           |
| `?` operator, synchronous      | Normative; `Err` representation Platform-defined | 5.2, 5.5, 8.5.1–8.5.2       |
| `?` operator, asynchronous     | Platform-defined                                 | 5.3, 8.5.3                  |
| `is_ok`, `is_err`, `map_err`   | Platform-defined                                 | 5.8.1, 8.5.5                |
| `map`                          | Normative                                        | type-aware dispatch per 5.7 |
| `unwrap_or_else`               | Normative; the fallback closure is lazy          | —                           |
| `ok`, `err`                    | Normative                                        | —                           |
| `and_then`, `or_else` (native) | Normative; native errors use `PyErr`             | 5.6, 8.5.4                  |
| `and_then`, `or_else` (Python) | Normative                                        | —                           |

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
| `usize` / `isize` boundary                | 2.4.1–2.4.4, 8.6   |
| `HashMap::get_mut` Python-boundary result | 8.10.2             |
| Numeric and index boundary errors         | 8.8                |
| Non-exhaustive match diagnostic           | 8.9                |
| `std:fs` host seam and `Err` payload      | 8.11               |

### 7.2.6 Collections and platform-specific types

| Type                        | Status                                              | Reference       |
| --------------------------- | --------------------------------------------------- | --------------- |
| `HashMap`                   | Normative                                           | 7.1, 7.4, 7.2.7 |
| `BTreeMap`                  | Normative (M3)                                      | 7.2.8, 10.25    |
| `HashSet`                   | Normative (M3)                                      | 7.2.8, 10.25    |
| Async executor (`PyFuture`) | Platform-defined                                    | 8.2.2–8.2.3     |
| Payload enum exposure       | Platform-defined (native: internal; Python: public) | 8.4.2           |

`HashMap<K, V>` supports `K` in `{i64, String}` and `V` in `{i64, String, bool, f64, Vec<i64>, HashMap<K2, V2>}` where `K2`/`V2` are themselves in the supported sets (depth 1). `BTreeMap<K, V>` supports `K` in `{i64, String}` and `V` in `{i64, String, bool, f64, Vec<i64>}`. `HashSet<T>` supports `T` in `{i64, String}`. Deeper nesting is Deferred. See 10.25.

### 7.2.7 `HashMap<K, V>` (collections)

| Function                                                                                      | Status                                  | Reference     |
| --------------------------------------------------------------------------------------------- | --------------------------------------- | ------------- |
| `new`, `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys`, `values`, `iter` | Normative (N8)                          | 7.2.6, 8.10.1 |
| `entry(k).or_insert(v)`                                                                       | Normative (N8 minimal entry subset)     | —             |
| `get_mut`                                                                                     | Platform-defined at the Python boundary | 8.10.2        |
| `entry(k).or_insert_with(f)`                                                                  | Normative (M3)                          | 7.2.6         |
| `entry(k).and_modify(f)`                                                                      | Normative (M3)                          | 7.2.6         |
| `entry(k).or_insert_with_key(f)`                                                              | Normative (M3)                          | 7.2.6         |
| `entry(k).or_default()`                                                                       | Normative                               | 7.2.6         |

### 7.2.8 `BTreeMap<K, V>` and `HashSet<T>` (collections)

| Function                                                                                                         | Status                                  | Reference |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------- | --------- |
| `BTreeMap::new`, `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys`, `values`, `iter`, `entry` | Normative (M3)                          | 7.2.6     |
| `BTreeMap::get_mut`                                                                                              | Platform-defined at the Python boundary | 8.10.2    |
| `HashSet::new`, `insert`, `contains`, `remove`, `len`, `is_empty`, `iter`                                        | Normative (M3)                          | 7.2.6     |
| `HashSet::get_mut`                                                                                               | Platform-defined at the Python boundary | 8.10.2    |

## 8. Platform Differences

This section defines the platform differences permitted by this specification. Each difference MUST have a test asserting the documented behaviour.

### 8.1 Integer overflow and integer casts

- **8.1.1** Integer overflow behaviour is Platform-defined. Native fixed-width integer overflow follows Rust semantics (panic in debug builds; wrap in release builds unless overflow checks are enabled). Python integers remain arbitrary precision.
  - **8.1.1.1** Native tests MUST assert wrap or panic behaviour according to build/profile where covered.
- **8.1.2** Cross-backend numerical results MAY differ when overflow occurs.
- **8.1.3** Explicit `as` conversions between integer types MUST follow Rust semantics natively. In the Python backend, fixed-width integer `as` casts apply Rust-like wrapping/truncation/sign interpretation unless the source is a literal integer value outside the target range, in which case `OverflowError` is raised at the `as` boundary. `usize`/`isize` casts remain range-checked per 2.4.9 and 2.4.12. See 2.5 and 10.26.
- **8.1.4** This difference MUST be recorded in Section 10.
- **8.1.5** Float-to-integer `as` casts truncate toward zero in both backends. NaN and infinities become 0. Native Rust saturates at the target type's bounds; the Python backend wraps the truncated value to the target width. The difference MUST be recorded in Section 10. See 10.28.
- **8.1.6** `char as i64` yields the Unicode scalar value in both backends. Integer-to-`char` casts are limited by Rust in the native backend: only `u8 as char` is accepted, and wider integer sources are rejected at Rust compile time. The Python backend accepts any integer in `0..=0x10FFFF` except the surrogate range `0xD800..=0xDFFF`, and raises `ValueError` otherwise. See 10.29.
- **8.1.7** `bool as i64` yields `0` or `1` in both backends. `i64 as bool` is a truthiness conversion in the Python backend (`0` is false, any non-zero value is true); the native backend rejects it at Rust compile time because Rust has no `integer as bool` cast. See 10.30.

**Acceptance criteria**

- Native overflow/cast tests exist where covered by the test suite.
- Python overflow/cast and boundary tests exist where covered by the test suite.

### 8.2 Async execution

- **8.2.1** Async MUST be marked Experimental.
- **8.2.2** The execution model is Platform-defined. The native backend drives futures on a single-threaded reactor thread backed by `mio`. Each task has a waker that enqueues it for polling; timer deadlines and Unix file-descriptor readability are delivered by the same reactor. When Python awaits a pending native future, `PyFutureIterator` yields an `asyncio` future created on the running loop, and the reactor completes it thread-safely with `loop.call_soon_threadsafe`. The native runtime also provides internal `polyglot_sleep_ms` and Unix-only `polyglot_readable` helpers; these are runtime helpers, not language-level standard-library functions. The internal `await_with_timeout` helper waits for a reactor wakeup or its deadline with the GIL released, rather than busy-polling, and raises `PyTimeoutError` on expiry. The Python backend executes async functions synchronously until a later version specifies otherwise.
- **8.2.3** The native future type MUST be Send when its captured values are Send. Cross-thread await MUST be supported for such futures. If a captured value is not Send, the native backend MUST reject the generated Rust at compile time.

**Acceptance criteria**

- A native async test exists.
- A cross-thread await test exists for Send captures.
- A native timer test demonstrates reactor wakeups without busy-polling.

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

### 8.11 `std:fs` host seam

- **8.11.1** The `std:fs` module MUST be a pure `.poly` wrapper over a small compiler-reserved host seam: `__fs_read_text(path)`, `__fs_write_text(path, contents)`, `__fs_append_text(path, contents)`, `__fs_exists(path)`, and `__fs_remove(path)`. Reads and writes return `Result<String, String>` or `Result<(), String>`; `exists` returns `bool`. Malformed host calls MUST raise a span-aware `E003` in both backends.
- **8.11.2** Host-seam calls MUST be gated by backend trust. They are available to the Python backend only when it is constructed with `trusted=True` and to the native backend only in its trusted compile path. Untrusted programs MUST NOT be able to reach the filesystem through `std:fs`.
- **8.11.3** In the Python backend an `Err` payload from `std:fs` is a raw `String`. In the native backend the language-level `Result<T, String>` is internally represented as `Result<T, PyErr>`; the payload is surfaced as a `PyErr` rather than a language string. Programs that need portable behaviour MUST match on `Err(_)` and MUST NOT propagate the payload into a language string.

**Acceptance criteria**

- A cross-backend parity test performs write/read/append/exists/remove round-trips through `std:fs` and asserts identical results.
- A test asserts `E003` with a character span for each host-seam arity violation in both backends.

### 8.12 Zig execution mode (experimental)

- **8.12.1** The Zig backend is an optional execution mode selected by `--backend zig` on `run`, `build`, `exec`, and `compile`, or by `get_backend("zig")`. It is not one of the two mandatory backends in 1.2, is not part of the Python/native parity contract, and MUST NOT be used to weaken a requirement that applies to those backends.
- **8.12.2** The Zig backend MUST require a `zig` executable on `PATH`. It generates Zig source from the shared AST, builds a dynamic library with `zig build-lib -dynamic -O ReleaseFast`, and loads the exported `__cabi_*` symbols with `ctypes`. The generated source and library MUST be cached under `~/.cache/polyglot-zig` in a directory keyed by a hash of the generated source; a cache hit MUST skip the Zig build.
- **8.12.3** The Zig build artifact is a C-ABI shared library, not a PyO3 extension module. It exposes one `export fn __cabi_<name>` wrapper per top-level function. `polyglot build --backend zig` copies the cached library next to the source. The copied artifact MUST NOT be presented as importable with Python's `import` statement.
- **8.12.4** At the `__cabi_*` boundary, integer parameters and results MUST be represented as 64-bit values (`u64` for unsigned Zig types, `i64` otherwise). Zig-native values wider than 64 bits MUST be truncated when converted at that boundary. Floating-point and pointer parameters are passed through as `f32`/`f64` and pointers; `void` results are supported.
- **8.12.5** The binary operators `+|`, `-|`, and `*|` MUST perform saturating addition, subtraction, and multiplication, clamping at the minimum or maximum value of the declared integer type instead of wrapping. They are Zig-only extensions and are not part of the portable integer contract in 2.5.
- **8.12.6** The type names `uN` and `iN` for an integer literal `N` with `0 <= N <= 65535` MUST be accepted as arbitrary bit-width integer types. Values MUST be represented at the declared Zig width. Widths above 64 bits remain subject to the truncating C-ABI rule in 8.12.4. The Python fallback lowering in `zig/lowering.py` rewrites the saturating operators to `_zig_sat_add`, `_zig_sat_sub`, and `_zig_sat_mul` and clamps at `i64` bounds; that fallback is not bit-width aware.
- **8.12.7** A `#[parallel]` annotation on a free function MUST split a direct range `for` loop in that function body into chunks executed by up to 16 worker threads, further capped by the available CPU count. Only the annotated function's direct range loop is parallelized; nested loops run serially. The implementation MUST NOT be read as providing iteration ordering, reduction, locking, or any other synchronization guarantee. A program is responsible for ensuring that parallel iterations do not race.
- **8.12.8** The type form `SimdN<T>` MUST map to Zig's `@Vector(N, T)`. The builtins `Simd::splat`, `Simd::load`, `Simd::store`, `Simd::reduce_add`, `Simd::reduce_mul`, `Simd::reduce_min`, `Simd::reduce_max`, `Simd::reduce_and`, `Simd::reduce_or`, and `Simd::reduce_xor` MUST lower to the corresponding Zig vector operations (`@splat`, slice-sourced vectors, and `@reduce`). Masking, gather/scatter, and general vector type inference are outside this execution mode.
- **8.12.9** The Zig execution mode lowers a narrow scalar subset: free functions, scalar parameters and returns, `let`, assignment, `while`, `loop`, `break`/`continue` (including labeled-block `break <value>`), direct range `for`, `return`, integer casts, indexing (with bounds-checked slice access via `_zig_at`/`_zig_at_mut`), binary operations, signed `/` and `%` lowered to `@divTrunc`/`@rem`, saturating operations, arbitrary-width integers, `#[parallel]` range loops, `Result<T, PillarError>` lowered to `PillarError!T`, the SIMD builtins, user-function calls, `if`/`else`, struct declarations and field access, the Zig intrinsics in 8.12.15–8.12.23, and `#[packed]` structs. Struct construction, enums, `match`, strings, `Vec`, and `Option` are not lowered. Programs outside this subset are outside the Zig contract and MAY fail to compile or produce behaviour that is not covered by this specification.
- **8.12.10** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.
- **8.12.11** Signed integer division and remainder MUST lower to `@divTrunc` and `@rem` respectively (truncate toward zero, matching C semantics). Unsigned division and remainder use the ordinary `/` and `%` operators.
- **8.12.12** Slice indexing `items[i]` MUST lower to a bounds-checked helper (`_zig_at` for immutable slices, `_zig_at_mut` for mutable slices) that asserts `i < items.len` before indexing. Many-item pointers (`[*]T`) have no `.len` field; reading `.len` on such a pointer MUST emit a `@compileError` directing the caller to pass the length explicitly as a separate `usize` argument.
- **8.12.13** `Result<T, PillarError>` MUST lower to the typed error union `PillarError!T` (not `anyerror!T`). `Err(E00n)`, `Err(n)` for `1 <= n <= 8`, and `Err(error.E00n)` MUST all resolve to the corresponding `PillarError` member.
- **8.12.14** A labeled block expression (`blk: { break :blk <value>; }`) MUST use an untyped label. A type annotation belongs on the variable being assigned (`const v: T = blk: { ... }`), not on the label itself; `blk: T { ... }` is rejected by Zig when the block is an if-branch.
- **8.12.15** The counting and rearrangement intrinsics `popcount!`, `clz!`, `ctz!`, `bswap!`, and `bitreverse!` MUST accept one fixed-width integer operand and lower to `@popCount`, `@clz`, `@ctz`, `@byteSwap`, and `@bitReverse`. The operand width MUST come from the declared polyglot type, with an unannotated identifier treated as `i64` and `usize`/`isize` modelled as 64-bit. `popcount!`, `clz!`, and `ctz!` MUST yield `i64`; `bswap!` and `bitreverse!` MUST yield the operand's own type so that a 64-bit result is not truncated. A float or otherwise widthless operand MUST be rejected with `E001`, a `bswap!` operand whose width is not a multiple of 8 bits MUST be rejected with `E001`, and a wrong argument count MUST be rejected with `E003`. These intrinsics are Zig-only and are not part of the portable integer contract in 2.5.
- **8.12.16** The attribute `#[packed]` on a struct declaration MUST lower that struct to a Zig `packed struct` with the declared field order and no padding. Every field MUST have a known bit width; integers (including arbitrary widths from 8.12.6) and `bool` qualify, while floats, pointers, slices, strings, and tuples MUST be rejected with `E001` because this backend does not assign them a bit layout. `#[packed]` on a generic struct MUST be rejected with `E003`. The declared field bit widths MUST sum to the reported `bitsizeof!` result.
- **8.12.17** `bitcast!(value, T)` MUST lower to `@bitCast` through an emitted helper that carries the explicit result type, and MUST require the source and target bit counts to be equal. A width or type mismatch, a pointer or slice source, and a type name with no known width MUST be rejected with `E001` before any Zig build runs. `bitsizeof!(T)` and `sizeof!(T)` MUST lower to `@bitSizeOf` and `@sizeOf` and MUST yield `i64`; a type name that does not resolve to a known type MUST be rejected with `E001`.
- **8.12.18** The attribute `#[unchecked]` on a free function MUST emit `@setRuntimeSafety(false);` as the first statement of that function body and MUST NOT affect any other function in the module. The explicit `std.debug.assert` guards emitted for bounds-checked slice access in 8.12.12 are unaffected, so `#[unchecked]` does not disable those checks. This attribute changes arithmetic behaviour from trapping to wrapping for that body and is Zig-only.
- **8.12.19** `add_with_overflow!(a, b)` and `mul_with_overflow!(a, b)` MUST lower to `@addWithOverflow`/`@mulWithOverflow` and yield `i64`: `1` when the operation overflowed the operand type, `0` otherwise. `wrapping_add!(a, b)` and `wrapping_mul!(a, b)` MUST lower to the same builtins and yield the wrapped value in the operand's own type. Both operands MUST have the same integer type; a float operand, a `bool` operand, an unknown width, and a type mismatch MUST be rejected with `E001`, and a wrong argument count MUST be rejected with `E003`. The builtin's `.{ value, flag }` tuple MUST be indexed directly, because struct construction is outside this backend's subset (8.12.9).
- **8.12.20** `rotl!(x, n)` and `rotr!(x, n)` MUST lower to `std.math.rotl`/`std.math.rotr` with the operand type as the comptime type argument, and MUST yield the operand's own type so the rotation stays inside the declared width. The count MUST be an integer and MUST be converted to the operand type; a count outside `0 .. width-1` traps on the conversion rather than rotating silently. A float operand MUST be rejected with `E001` and a wrong argument count with `E003`.
- **8.12.21** `int_from_float!(x)` MUST lower to `@intFromFloat` and yield `i64`, truncating toward zero; `float_from_int!(x)` MUST lower to `@floatFromInt` and yield `f64`. Each MUST reject the other operand family with `E001`: `int_from_float!` needs a float, `float_from_int!` needs an integer. An out-of-range `int_from_float!` or an integer too wide for an exact `f64` is the caller's responsibility and is recorded as a boundary in 10.53.
- **8.12.22** `alignof!(T)` MUST lower to `@alignOf` and yield `i64`. It MUST accept the same operands as `sizeof!` (8.12.17): builtin integer and float types, arbitrary-width integers, and declared structs. For a `#[packed]` struct the reported alignment is that of the backing integer, which is why `alignof!(Flags)` and `sizeof!(Flags)` agree for a 9-bit layout.
- **8.12.23** `memcpy_bytes!(dst, src, count)` MUST lower to `@memcpy` over `dst[0..count]` and `src[0..count]`, and `memset_bytes!(dst, value, count)` to `@memset` over `dst[0..count]`. The destination MUST be a pointer or slice (`E001` otherwise), the source of `memcpy_bytes!` likewise, the count MUST be an integer, and a wrong argument count MUST be rejected with `E003`. Equal lengths are guaranteed by construction (both views use the same `count`), but the regions MUST NOT overlap; `@memcpy` treats an overlap as an assertion failure.

**Acceptance criteria**

- `tests/test_zig_backend.py` covers saturating `+|`/`-|`/`*|`, `u8` and `u24` arithmetic, the Python fallback lowering, a `#[parallel]` range loop, SIMD vector arithmetic and reductions, signed `/`/`%` lowering to `@divTrunc`/`@rem`, bounds-checked slice indexing, labeled-block `break <value>`, `Result<T, PillarError>` → `PillarError!T`, `Err(E00n)` resolution, and the CLI `run`/`build` paths.
- The Zig tests are skipped when `zig` is not installed; the Python and native suites do not depend on the Zig toolchain.
- `examples/zig_parallel_simd.poly` combines `#[parallel]`, `Simd4<f64>`, `Simd::load`/`store`/`reduce_add`, and a saturating `u8` return.

### 8.13 Gleam-inspired statics (experimental)

- **8.13.1** `compiler/gleam_style/` is a pure-Python analysis layer that models Gleam-style total algebraic data types (ADTs), exhaustive pattern matching, bit-array pattern matching, and cross-target capability contracts. It is not part of the Python/native parity contract and MUST NOT be used to weaken a requirement that applies to those backends.
- **8.13.2** `gleam_enforce_total_adt_and_exhaustiveness` MUST validate that every variant of an ADT is covered by a `match`'s patterns. A non-exhaustive match MUST raise `E002` with a concrete missing-pattern witness per uncovered variant (for example `Some(Int)` or `None`), not a generic message. A wildcard pattern makes the match trivially exhaustive. Unknown variants in a pattern MUST raise `E001`.
- **8.13.3** Recursive ADTs (for example a `Tree<T>` whose `Node` variant references `Tree<T>`) MUST be resolved by a bounded Hindley-Milner-style walk. Cycles MUST be recorded; exceeding the resolution budget MUST raise `E002` rather than recurse without bound.
- **8.13.4** `gleam_lower_bit_array_pattern_matching` MUST parse patterns of the form `<<name:WIDTH[:TYPE][:ENDIAN]>>` with `TYPE` in `{binary, bit_string, utf8_codepoint, int, float}` and `ENDIAN` in `{big, little, native}`. A segment without an explicit width MUST be variable-width and MUST be the final segment; at most one variable-width segment is allowed.
- **8.13.5** Bit-array matching MUST bounds-validate every read against `len(data)` before extraction. A short buffer, a non-byte-aligned fixed segment, or a fixed width that is not a multiple of 8 bits MUST raise `E001`. No `ctypes.cast`, no `struct.unpack_from` without a length guard, and no raw pointer arithmetic is permitted. The total fixed width MUST be capped at 4096 bits.
- **8.13.6** `gleam_emit_cross_target_contracts` MUST keep pure functional data transformations portable across the Python, Rust/PyO3, Zig, BEAM, and JS target descriptors. A target that lacks a required capability MUST raise `E007` naming the missing capabilities; a `TargetDescriptor` MAY override or disable capabilities for a specific deployment.
- **8.13.7** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_gleam_style.py` covers exhaustive and non-exhaustive matches with concrete witnesses, wildcard handling, unknown/duplicate/empty ADTs, registry errors, recursive cycle detection and budget exhaustion, bit-array parsing and matching across endian and bounds cases, and cross-target contracts with overrides and disabled capabilities.

### 8.14 Mojo-inspired systems bridge (experimental)

- **8.14.1** `compiler/mojo_bridge/` is a pure-Python analysis layer that models Mojo-style value/origin lifetimes, contiguous buffer layouts with SIMD lane analysis, MLIR capability probing, and CPython interop. It is not part of the Python/native parity contract and MUST NOT be used to weaken a requirement that applies to those backends.
- **8.14.2** `mojo_model_value_and_origin_semantics` MUST track, per binding, a value kind in `{Copyable, Movable, BorrowOnly}` and an origin in `{Owned, Borrowed, UnsafeAnyOrigin}`. Using a moved binding MUST raise `E002`. Returning a `Borrowed` binding as owned MUST raise `E001`. Declaring or returning an `UnsafeAnyOrigin` binding without the explicit `#[unsafe_origin]` opt-in MUST raise `E007`; the widening MUST never be silent.
- **8.14.3** `mojo_lower_vectorized_kernels_and_layout` MUST describe contiguous tensor/buffer layouts (shape, dtype, strides, memory order) with `is_contiguous()` and `simd_lanes()` helpers, and MUST lower simple loop kernels into a backend-neutral plan recording loop bounds, vectorizability, lane count, and an optional MLIR-style pass list. Non-contiguous layouts and layouts too small for SIMD MUST be marked non-vectorizable with a concrete reason.
- **8.14.4** `probe_mlir_available()` MUST report `available: False` with the missing tool names when `mlir-opt` / `mlir-translate` are not on `PATH`. It MUST NOT claim native MLIR or GPU support that does not exist.
- **8.14.5** `mojo_marshal_cpython_interop` MUST bridge Python scalars and bytes to typed native values on the cold path only (`bool`→`bool`, `int`→`i64` truncated to 64 bits, `float`→`f64`, `bytes`→`[]u8`, `None`→`void`). Unsupported types MUST raise `E001` directing the caller to a typed native binding. Accessing `frame.f_locals` in a hot loop MUST raise `E007` under PEP 667. The hot-loop zero-boxing path MUST NOT be modelled by per-element `PyObject` boxing.
- **8.14.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_mojo_bridge.py` covers origin declarations, borrows, lifetime matching, moves (use-after-move, double move, moving borrowed values), origin escapes, `UnsafeAnyOrigin` without annotation, buffer layout contiguity and SIMD lanes, kernel lowering, the MLIR probe, cold-path bridging for every supported type, and the PEP 667 hot-loop guard.

### 8.15 Elixir/OTP-inspired actor runtime (experimental)

- **8.15.1** `runtime/actor_otp/` is a deterministic, pure-Python runtime model that implements Erlang/OTP-style isolated actors, supervision, and let-it-crash fault handling. It is not part of the Python/native parity contract and MUST NOT be used to weaken a requirement that applies to those backends. It MUST NOT spawn OS threads or provide real BEAM/OTP semantics.
- **8.15.2** `otp_spawn_isolated_actor_process` MUST create an actor that owns its state and a structured mailbox. Messages MUST be deep-copied at the send boundary so an actor can never mutate another actor's state through a shared reference. Sending to a dead actor MUST raise `E001`; a full bounded mailbox MUST raise `E001`.
- **8.15.3** `otp_supervise_and_restart_children` MUST implement the `one_for_one`, `one_for_all`, and `rest_for_one` strategies. `one_for_one` restarts only the crashed child; `one_for_all` restarts every child; `rest_for_one` restarts the crashed child and every child started after it. A `RestartIntensity(max_restarts, window_seconds)` budget MUST be enforced; exceeding it MUST raise `E004` and escalate rather than restart indefinitely.
- **8.15.4** `otp_let_it_crash_quarantine_and_telemetry` MUST isolate child failures into structured `E001`–`E008` fault reports with a traceback and a boundary-safety flag, and MUST record telemetry in a bounded ring buffer (default 10,000 events) that drops the oldest event when full. Telemetry MUST NOT grow unbounded or crash the host.
- **8.15.5** The actor runtime MUST respect platform boundaries. A fault handler that attempts to touch `MAP_JIT`, PEP 667 frame locals, raw addresses, TOCTOU windows, or model-only paths MUST raise `E007`. Fault isolation MUST NOT crash unrelated subsystems.
- **8.15.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_actor_otp.py` covers actor spawn and message delivery, isolation guarantees (initial-state copy and payload deep-copy), dead-actor sends, handler crashes, mailbox capacity and selective receive, all three supervision strategies, restart-intensity exhaustion and window reset, crash classification per exception class, boundary-violation rejection, and telemetry bounding and filtering.

### 8.16 Ownership and move checking

- **8.16.1** Polyglot MUST follow Rust ownership for owned values: `String`, `Vec<T>`, `HashMap`, user structs and enums are moved when used by value; integers, floats, `bool`, `char`, references and tuples of those are copied. A use of a moved binding MUST raise `E009`, naming the binding and the line of the move and/or the line of the reuse.
- **8.16.2** The check MUST run on the shared front end before either backend generates code, so the Python and native backends accept and reject the same programs. A program rejected by one backend MUST be rejected by the other with the same code, the same binding name and the same line pair. `polyglot explain E009` documents the rule for users.
- **8.16.3** Moving out of an owned container by index (for example passing `words[0]` where a `String` is expected) MUST raise `E009` with its own message, because indexing yields a place rather than a value. Cloning the element or taking a reference is the documented fix.
- **8.16.4** The first use of a moved value is reported; the pass MUST be deterministic and MUST stop at the first violation rather than emit a cascade. A diverging branch (`return`, `break`, `continue`, `panic!`) MUST NOT contribute its moves to the state after the branch, matching Rust, and an unconditional reassignment MUST make the binding live again.
- **8.16.5** The model is bounded and MUST NOT guess. Unknown types, method semantics the pass does not model, and moves through references are left to the backend. A construct the pass cannot prove MUST NOT be reported as an error.

**Acceptance criteria**

- `tests/test_move_parity.py` runs the same case table through both backends and asserts identical outcomes: allowed programs run with the same return value, rejected programs raise the same `E009` headline with the same line pair.
- `tests/test_m7_diagnostics.py` covers `E001`–`E008` formatting; `E009` uses the same five-line form and prints a character span when the node has one.

### 8.17 Cluster membership, session routing, and split brain

- **8.17.1** `cluster_discover_and_heartbeat` MUST process a heartbeat batch in `(node, seq)` order, MUST join an unknown sender as a discovered member, and MUST ignore a repeated or older `seq` from the same node, reporting it as stale.
- **8.17.2** A heartbeat whose timestamp moves backwards MUST raise `E001`; a member silent longer than `ttl_ms` MUST be evicted from the membership view, marked dead, and reported. A non-positive `ttl_ms` MUST raise `E001`.
- **8.17.3** `cluster_route_session_typed_rpc` MUST reject a message whose role or type is not the first step of the supplied protocol (`E001`), a `schema_version` outside the supported set (`E007`), and an empty cluster (`E006`). The payload MUST round-trip through `encode_frame`/`decode_frame`; a mismatch MUST raise `E004`.
- **8.17.4** Routing MUST be deterministic: the target is selected by a hash of `route_key` over the sorted live node set, and the session id by a hash of `(route_key, direction, message_type, epoch)`.
- **8.17.5** `cluster_handle_partition_and_split_brain` MUST mark the largest group that reaches `total // 2 + 1` nodes writable, fence the remaining groups at a bumped epoch, and freeze the whole cluster (no epoch change, no writable group) when no group reaches quorum. Unknown nodes, overlapping groups, and an empty cluster MUST raise `E001`/`E002`.
- **8.17.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_cluster_frontier.py` covers discovery, stale and backwards heartbeats, TTL eviction, route determinism, out-of-turn messages, unsupported codec versions, an empty cluster, majority fencing, frozen splits, overlapping partitions, and unknown nodes.

### 8.18 Deterministic simulation testing

- **8.18.1** `dst_virtualize_clock_and_scheduler` MUST order operations by logical time and break ties with a key derived from the seed, so the same operations and seed always produce the same trace.
- **8.18.2** The virtual clock MUST be monotonic. A negative timestamp or an operation scheduled before `start_ms` MUST raise `E001`.
- **8.18.3** `dst_inject_byzantine_and_io_faults` MUST draw its faults in a fixed order from a seeded generator, so the faulted operation list is a pure function of `(ops, seed, rates)`. Rates outside `[0, 1]` and unknown fault kinds MUST raise `E003`.
- **8.18.4** The supported faults MUST be `drop`, `duplicate`, `reorder`, `torn_write`, `lost_fsync`, and `bitflip`, and the report MUST count each injected fault.
- **8.18.5** `dst_bisect_and_minimize_schedule` MUST shrink a failing schedule by removing contiguous chunks while a caller-supplied deterministic oracle still fails, MUST return the probe count, and MUST stop at a hard probe budget instead of looping.
- **8.18.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_dst.py` covers seed reproducibility, seed-dependent tie order, elapsed time, rejected timestamps, per-fault effects, rate validation, culprit isolation, the non-failing case, and the probe budget.

### 8.19 Hygienic cross-backend macros

- **8.19.1** `macro_resolve_hygienic_scopes` MUST rename template binders to `name#<scope>` and resolve template free references at the definition site as `name@<scope>`.
- **8.19.2** A free name that also exists at the call site MUST be recorded as captured and MUST be qualified, so a call-site binding cannot capture it. Unbound globals MUST stay bare.
- **8.19.3** `macro_expand_typed_ast_splice` MUST fill every hole exactly once, MUST reject an arity mismatch and an unfilled hole with `E002`, and MUST reject a concrete type mismatch with `E001`. `unknown` MUST be accepted for names.
- **8.19.4** Expansion depth MUST be capped; exceeding the cap MUST raise `E002` rather than overflow the stack.
- **8.19.5** `macro_verify_backend_expansion_parity` MUST lower the expanded tree on every requested backend, compare canonical opcode streams, and raise `E004` naming the divergent backend and the first divergent opcode. An unregistered backend MUST raise `E004`.
- **8.19.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_macro_engine.py` covers binder renaming, definition-site qualification, capture reporting, bare globals, determinism, block scoping, hole filling, arity and type rejection, depth limits, parity success, divergence detection, and unfilled-hole rejection.

### 8.20 Shared-memory arena and borrow leases

- **8.20.1** `ipc_map_shared_ring_arena` MUST create a file-backed ring with a fixed header (magic, version, slots, slot size, next slot, head, tail, epoch, inflight). A foreign magic MUST raise `E008`, an unsupported version `E007`, a missing file without `create=True` `E006`, and a slot-count mismatch `E003`.
- **8.20.2** `push` MUST copy the payload into the next slot and return its index, MUST return `-1` when the ring is full or a lease is outstanding, and MUST raise `E003` for a payload larger than a slot.
- **8.20.3** `pop` MUST return a zero-copy `memoryview` into the mapping and MUST register it as in-flight until released; leases MUST release in FIFO order (`E003` otherwise).
- **8.20.4** `ipc_enforce_epoch_borrow_lease` MUST accept a lease only when its epoch equals the arena epoch, MUST raise `E002` for a stale epoch, and MUST allow one live lease per holder (`E003` on duplicates).
- **8.20.5** `ipc_reclaim_abandoned_slots` MUST require an epoch that advances past the current one (`E002` otherwise), MUST zero abandoned slot bytes, clear leases, reset the ring, and report evicted holders and dropped slots.
- **8.20.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_ipc_arena.py` covers header initialisation, push/pop round-trip, zero-copy views, empty and full rings, lease-blocked overwrite, FIFO release, oversized payloads, two mappings sharing one ring, foreign files, missing files, slot-count mismatch, lease grant/release, duplicate holders, stale epochs, reclaim effects, and zeroed bytes on disk.

### 8.21 Bootstrapped compiler core

- **8.21.1** `bootstrap_stage1_compile_subset` MUST parse a subset of integer functions with `let`, `return`, `if`/`else`, calls, and `+ - * / < > ==`, type-check it, and lower it to canonical IR whose digest is a hash of the IR text.
- **8.21.2** Syntax errors and duplicate names MUST raise `E003`; unknown variables and return-type mismatches MUST raise `E001`; calls to undefined functions MUST raise `E006`.
- **8.21.3** `parse_ir` MUST accept only the canonical shapes `emit_ir` can produce and MUST raise `E008` otherwise, so the fixed-point check cannot pass by re-printing malformed IR.
- **8.21.4** `bootstrap_verify_stage2_fixed_point` MUST re-read and re-emit the IR and MUST require the digest to repeat within a bounded number of rounds; failing to reach a fixed point MUST raise `E004`.
- **8.21.5** `bootstrap_diff_stage_ir_hashes` MUST report digest equality and, when they differ, the first differing IR line.
- **8.21.6** A change to this section MUST follow the versioning rule in 11.1 and MUST record the resulting boundary in Section 10.

**Acceptance criteria**

- `tests/test_bootstrap.py` covers canonical lowering, digest stability, syntax/type/undefined-function errors, duplicate parameters, the Zig demonstration emitter and its documented limit, IR round-tripping, the fixed point, malformed IR, digest diffing, and whitespace-insensitive parsing.

## 9. Toolchain and Delivery

- **9.1** The toolchain MUST include a lexer, a parser, an AST, both backends, diagnostics that map native errors to source positions, and a module mechanism.
- **9.2** The module mechanism MUST provide a one-way inclusion directive. `include!("path/to/file.poly");` at item position parses the included file and splices its items at the include site. Relative paths resolve relative to the including file's directory; absolute paths are allowed. Included files MAY contain includes recursively. Cyclic includes MUST be detected and rejected with a five-line error that prints the include chain. Full `mod` and `use` syntax remains Deferred.
- **9.2.1** A bundled standard-library namespace MUST be available as `std:`. `include!("std:<name>");` resolves the name inside the packaged `stdlib/` directory and appends `.poly` when the name has no suffix. The equivalent bare `include("std:<name>");` form MUST be accepted at item position. Names that are absolute, escape `stdlib/`, or do not name a module MUST be rejected with `E006`; cyclic standard-library includes MUST still report `E005`.
- **9.2.2** Standard-library includes MUST be deduplicated by resolved path across the whole program expansion. Ordinary (non-`std:`) includes remain textual and MAY splice duplicate items, preserving the pre-v0.3.8 contract.
- **9.2.3** The bundled standard library MUST consist of pure `.poly` modules: `math`, `algo`, `str_utils`, `fs`, `collections`, `numpy`, `linalg`, `random`, and `prelude`. `prelude` includes the other modules, and `numpy` may be re-included by `linalg` and `prelude` without duplicating `NdArray` or its impl blocks. The `stdlib/` package and its `.poly` files MUST ship with the wheel.
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
- **9.12** The toolchain MUST expose a stable registry of 100 numbered framework functions spanning lexing/parsing, types/monomorphization, IR/optimization, both backends, hot-patching, concurrency, diagnostics, CI gates, and docs/regression harnesses. `framework_functions.validate_registry()` MUST confirm contiguous numbering, unique names, and importable callables. See `docs/framework_functions.md`.
- **9.13** A deterministic differential fuzzer MUST generate valid programs covering recursive generic structs/enums, `vec!` element propagation, trait bounds and impls, match guards, arithmetic/boolean control flow, and E001–E008 cases; execute each case through the Python and native backends under isolated cache scopes; compare normalized values, side effects, and structured error codes; and save minimal seed/source artifacts only on mismatch. Clean runs MUST NOT create working-tree files. `tools/diff_fuzz.py --seed N --case-count M --json` MUST be reproducible for a fixed seed.
- **9.14** A unified release gate MUST run ruff format/lint, scaffold drift verification, core pytest suites, bounded fuzz seeds, isolated `cargo check`/`cargo clippy`, docs sync/lint, and a git worktree check in a fixed order, aggregate them with `ci_generate_release_gate_report`, and exit non-zero on any blocker. `--quick` and `--full` MUST select bounded and extended verification respectively.
- **9.15** The CLI and visualizer MUST expose session telemetry containing compile-phase timings, cache hits/misses by scope and rebuild reason, monomorphization specialization counts, and hot-patch capability/epoch/rollback/event status. Telemetry payloads MUST be JSON-safe, MUST report graceful empty states, and MUST NOT leak absolute local paths. Model-only hot-patch boundaries (MAP_JIT, PEP 667, raw-address TOCTOU) MUST remain visible in capability metadata.
- **9.16** Wheels and sdists MUST ship every module referenced by the 100-function registry. `tests/` stays out of wheels, so framework functions 94–100 resolve through the installable `regression.harness` package; `tests/regression_harness.py` is only a compatibility re-export. Cargo metadata MUST be complete enough for `cargo package` to run without manifest warnings, and builds MUST use isolated cache/target directories outside the workspace.
- **9.17** The fuzzer corpus MUST cover Option/Result flows, strings and floats, collections, stdlib includes, and hotpatch-shaped immutable state in addition to the categories in 9.13. Native-unsupported templates MUST be reported as skips with a reason, never as fabricated mismatches. Mismatches MUST be shrunk before artifact emission, with a bounded number of additional native builds, so saved reproducers are minimal.
- **9.18** Telemetry MUST capture hot-patch activity automatically from the RCU epoch manager, rollback journal, and frame migrator through a dependency-free runtime event bus; these events MUST be redacted and MUST NOT claim privileged MAP_JIT/PEP 667/TOCTOU behavior. Telemetry MUST support path-redacted baseline persistence and comparison outside the workspace, and MUST report cache-prune and codegen-fragment counters in its snapshot.
- **9.19** The release gate MUST additionally verify cargo formatting/docs, dependency and package metadata, Markdown link integrity, benchmark regression, and release metadata, in a defined order after the stages in 9.14. It MUST support optional parallel fuzz seeds that preserve deterministic report order and a shared warm dependency cache. CI configuration MUST run the full gate on clean worktrees and publish maturin-built native wheels with SBOMs; release-only signing and trusted publishing MUST be declared but MUST NOT be claimed as executed locally.
- **9.20** The release gate MUST include a `hard-frames-parity` stage after the core pytest suites and before bounded fuzzing. It MUST run the JIT/WASM/hard-frame suites and the three-backend parity seeds `1`, `42`, `9999`, and `2026` as separately reported checks; any mismatch, harness error, or suite failure MUST block the gate. The parity runner MUST be injectable so verification tests never compile native code.
- **9.21** The repository MUST provide research frames H–K without adding WASM or JIT behavior: `compiler.formal` (bounded symbolic execution, bounded relational refinement, infeasible-path witnesses), `compiler.termination` (declared structural/lexicographic variants, syntactic loop-bound certificates, deterministic resource-fuel contracts), `runtime.hard_state` (declared points-to/alias over-approximation, total live-state schema migration with rollback, quarantined deterministic replay), and `tools.adversarial` (deterministic grammar-bounded case generation, mutation-score gate, spec-oracle report). Every result MUST be bounded and MUST report bound exhaustion, timeouts, or `Unknown` as structured `E001`–`E008` diagnostics; no frame MAY claim an undecidable property unconditionally. `MAP_JIT`, raw-FFI, and PEP 667 dynamic-frame paths MUST remain explicit boundaries. The release gate MUST include a `research-frames` stage after `hard-frames-parity` and before bounded fuzzing that runs the H–K focused suites plus an injectable, in-process spec-oracle check; a missing or failed check MUST block the gate.

**Acceptance criteria**

- A command-line interface exists.
- The Quick Start is validated.
- An examples directory exists.
- The test suite status recorded in Section 12 is the authoritative source; `README.md` MUST be updated to match Section 12 before a conformance claim is made.

## 10. Known Limitations

This section is normative. Each entry MUST reference the requirement or platform difference it qualifies.

- **10.1** Integer overflow differs across backends. See 8.1.4.
- **10.2** Async is Experimental. Native futures are Send-aware and may be awaited across threads when their captured values are Send; non-Send captures are rejected at Rust compile time. See 8.2.3.
- **10.3** Box patterns are supported only by the Python backend. See 8.3.
- **10.4** Payload enum slices are supported only by the Python backend; native payload enums are internal. See 8.4.
- **10.5** Native `Result::and_then` and `Result::or_else` use `PyErr` continuations. See 8.5.4.
- **10.6 (Retired in v0.3.7).** `Option::map` and `Result::map` are supported through type-aware receiver dispatch. See 5.7.
- **10.7** `while let Some(x) = iter.next()` is deferred. See 6.5.
- **10.8** Native async `impl` methods with `&self` and `&mut self` receivers are supported. `&mut self` uses an atomic borrow guard, copied mutable fields, and commit-on-success; overlapping calls are rejected. By-value receivers are Deferred; bare `self` use and non-`Clone` fields are not supported. See 6.4.
- **10.9 (Retired in v0.2.5).** The transitional `usize`/`isize` behavior is retired; implementations use distinct pointer-sized types and explicit casts. See 2.4.5.
- **10.10 (Retired in v0.3.3).** Additional fixed-width integer types are specified in 2.5 and implemented in scope of M4-remaining. See 2.5 and 8.1.
- **10.11 (Retired in v0.3.3).** Python backend `as` conversion behavior for additional integer types is specified in 8.1 and 10.26. See 2.5 and 2.7.
- **10.12 (Retired in v0.3.9).** The restriction to `if let Some(binding) = expr` is lifted: the builtin `Some`, `None`, `Ok`, and `Err` patterns are supported in both statement and expression positions. Other if-let pattern forms still raise a translation error. See 5.4.
- **10.13** `String::parse` failure behaviour differs across backends. See 8.7.
- **10.14** Negative `String::repeat` counts and out-of-range `Vec::rotate_*` / `Vec::swap` arguments raise in the Python backend and panic natively. See 8.8.
- **10.15** Non-exhaustive match produces a translation-time warning in the Python backend and a `RustCompileError` in the native backend. See 8.9.
- **10.16** `PatConst` and `PatMacro` are parsed but not emitted by either backend. See 4.1.1.
- **10.17** `Result::is_ok`, `Result::is_err`, and `Result::map_err` are Platform-defined. See 8.5.5.
- **10.18** `Option::ok_or` and `Option::ok_or_else` are Platform-defined. See 8.5.6.
- **10.19** No platform difference may be removed from this section without a corresponding SPEC version increase. See 11.1.
- **10.20** Native `HashMap::get_mut` returns a copied `Option<V>` at the Python boundary; the copy cannot mutate the input dictionary. See 8.10.2.
- **10.21** M3-remaining extends the collection targets in 7.2.6–7.2.8. Native and Python support for `BTreeMap`, `HashSet`, extended entry methods, nested HashMap depth-1 values, and collection `get_mut` semantics is implemented for those types. Deeper nesting and unsupported type combinations remain Deferred. See 10.25 and 10.27.
- **10.22** Multiple self-recursive calls are accepted when each strictly decreases an argument under a lower-bound guard in scope at that call site; an `E002` points at the offending call rather than the function header. Guards from an enclosing scope are not visible inside a nested function or closure body, whose self-calls must be proved from guards inside that body. Termination proofs for mutual recursion, loops with `break`, complex arithmetic, non-recursive function calls, and async suspension are Deferred. Such constructs are conservatively rejected in an annotated function when the checker cannot prove termination.
- **10.23** Cyclic includes are rejected. Full `mod` and `use` syntax remains Deferred. See 9.2.
- **10.24** AST spans are best-effort; nodes with a single token have span equal to that token's span. See 2.4.7 and 9.3.
- **10.25** Nested `HashMap` value depth is 1 (e.g. `HashMap<K, HashMap<K2, V2>>` where `K2`/`V2` are supported). Deeper nesting is Deferred. See 7.2.6.
- **10.26** M4 Python-backend integer `as` casts apply Rust-like wrapping for fixed-width integer targets unless the source is a literal integer value outside the target range, in which case `OverflowError` is raised at the `as` boundary. `usize`/`isize` casts remain range-checked per 2.4.9 and 2.4.12. See 2.5 and 8.1.
- **10.27** Native backend support for `BTreeMap`, `HashSet`, extended entry methods, nested `HashMap` depth-1 values, and collection `get_mut` copy semantics is implemented for the types listed in 7.2.6–7.2.8. Deeper nesting and unsupported key/value combinations remain Deferred. See 7.2.6–7.2.8.
- **10.28** Float-to-integer `as` casts truncate toward zero in both backends; native saturates at the target bounds while the Python backend wraps the truncated value. See 8.1.5.
- **10.29** Integer-to-`char` `as` casts accept only `u8` natively (Rust restriction); the Python backend accepts Unicode scalar values and rejects surrogates and values above `0x10FFFF`. See 8.1.6.
- **10.30** `i64 as bool` is a truthiness conversion in the Python backend and a native compile error (Rust has no such cast). `bool as i64` is `0`/`1` in both backends. See 8.1.7.
- **10.31** `TranslateError` (E003) is raised only by the Python backend translation path. The native backend reports unsupported constructs as `UnsupportedNativePatternError` (E007) or `RustCompileError` (E004); there is no native E003 path. See 9.3.
- **10.32 (Retired in v0.3.9).** `String::from(...)` is lowered by both backends; `"...".to_string()` remains equivalent. See 7.2.2.
- **10.33 (Retired in v0.3.9).** The `vec![...]` macro is lowered by both backends to the shared array AST, including `vec![value; count]`. See 7.2.1.
- **10.34** The native reactor, its timers, and Unix fd readiness are runtime-internal. The language does not expose async timer or I/O primitives; `polyglot_sleep_ms` and `polyglot_readable` are helpers used by the runtime and tests. See 8.2.2.
- **10.35** The bundled `std:` modules are pure `.poly` and inherit the language subset in this specification. They do not add a package manager, arbitrary crate dependencies, or float math beyond the provided helpers. Only `std:` includes are deduplicated; ordinary includes still splice textually. See 9.2.
- **10.36** The experimental pillar modules (`backends/hotpatch.py`, `compiler/`, `proofs/`, `debug/`, `runtime/transmute.py`, and `runtime/hotpatch/`) are opt-in research foundations and are not part of the stable conformance surface. They confine patches to writable process GOT/PLT cells and owned executable pages or to quiescent/PEP 667 safe points, never patch foreign loaded executable text or a running frame's `f_code` mid-instruction, never rewind an opaque CPython/PyO3 stack, prove only within their documented interval/polynomial grammar, and deoptimise through a guard plus fallback rather than a nanosecond guarantee. The quantum subsystem adds real GOT/PLT, frame-local, reactor, RCU, and before-image journal primitives under those boundaries. They reuse the `E001`–`E008` diagnostics. See `docs/quantum_hotpatch.md` and `docs/experimental_pillars.md`.
- **10.37** The unified 10-pillar / 100-function surface is a capability registry (`runtime/capabilities.py`), not a claim that every requested mechanism is implementable. `runtime/capabilities.CAPABILITIES` classifies each function as `foundation`, `modelled`, or `rejected`; both backends expose the same 100 names through `capability_namespace()` and emit identical span-aware `E001`–`E008` diagnostics. Requests that require unstable interpreter internals, executable-text patching, privileged OS primitives, real accelerator dispatch, or cryptographic threshold signatures are uniformly rejected. The generated `native_pillars/` PyO3 scaffold is opt-in and is not part of `polyglot build`. See `docs/unified_pillars.md`.
- **10.38** The `std:fs` host seam is available only to trusted compilation paths, and the native `Err` payload is a `PyErr` rather than a language string. See 8.11.
- **10.39** Diagnostic fix hints are deterministic and local: an edit is emitted only when the replacement can be derived from the diagnostic itself (currently implicit `as` casts under `E001`). Other codes may carry an actionable message without a text edit, and no hint is produced when the correct fix cannot be determined. See 9.3.
- **10.40** The language server publishes at most one diagnostic per validation event and converts 0-based character offsets to LSP positions without applying UTF-16 code-unit encoding; non-BMP text before a diagnostic can shift the reported column. LSP 3.17 has no `publishDiagnosticsProvider` capability, so diagnostics are unsolicited notifications on `didOpen`, `didChange`, and `didSave`. See 9.3.
- **10.41** The `HashMap` entry API supports `or_insert`, `or_insert_with`, `or_insert_with_key`, `and_modify`, `key`, and `or_default`. `Entry::insert` is not lowered by the native backend and remains Deferred. See 7.2.7.
- **10.42** The Zig backend is an experimental, opt-in execution mode outside the Python/native parity contract. It requires a Zig toolchain, lowers only the narrow subset in 8.12.9, produces a C-ABI shared library rather than an importable Python extension, truncates Python-boundary integers to 64 bits, parallelizes only a direct range loop without ordering or synchronization guarantees, and exposes only the named SIMD builtins. See 8.12.
- **10.43** `compiler/gleam_style/` is a pure-Python analysis layer outside the Python/native parity contract. It models total ADTs with exhaustive-match witnesses, bit-array pattern matching with checked bounds, and cross-target capability contracts. It does not generate native code, does not require a Gleam or BEAM toolchain, and does not provide real Gleam compilation. See 8.13 and `docs/gleam_style.md`.
- **10.44** `runtime/actor_otp/` is a deterministic, pure-Python runtime model outside the Python/native parity contract. It provides share-nothing actors, OTP supervision strategies with restart budgets, and let-it-crash fault reports with bounded telemetry. It does not spawn OS threads, does not provide real BEAM/OTP semantics, and does not replace the `distributed/` actor runtime. See 8.15 and `docs/actor_otp.md`.
- **10.45** `compiler/mojo_bridge/` is a pure-Python analysis layer outside the Python/native parity contract. It models value/origin lifetimes, contiguous buffer layouts with SIMD lane analysis, an honest MLIR availability probe, and CPython interop with a PEP 667 hot-loop guard. It does not generate native code, does not invoke MLIR/GPU toolchains, and does not provide real Mojo compilation. See 8.14 and `docs/mojo_bridge.md`.

- **10.47** The cluster frontier is a single-process model. Membership, leases, and codecs live in memory; there is no socket layer, gossip, or persistence, and no claim of multi-host location transparency. See 8.17.
- **10.48** The DST harness models time, ordering, and six I/O/network faults over a list of operations. It has no thread preemption model, no address-space simulation, and no durable disk; the minimiser is chunk-based and not a proof of 1-minimality. See 8.18.
- **10.49** The macro engine proves hygiene, typed splicing, and opcode parity on a miniature AST with three naive emitters. It does not parse surface syntax and does not know the real backend IRs, so it is not a claim that the three compilers agree on everything. See 8.19.
- **10.50** The IPC arena is single-host with fixed-size byte slots and a per-process holder table. It does not claim lock-free multi-threaded safety, Arrow awareness, or tensor transport; `push` copies once, while `pop` is zero-copy. See 8.20.
- **10.51** Stage 2 of the bootstrap is the same Python implementation driven from printed IR, and the Zig demonstration emitter handles only return-only functions. What is verified is canonicality and digest stability, not machine-code self-hosting. See 8.21.
- **10.52** The Zig intrinsics are a bounded model of hardware operations, not a portable integer library: operand widths come from the inferred polyglot type (an unannotated identifier is `i64`, and `usize`/`isize` are 64-bit), so a program that relies on a width the backend cannot infer is rejected rather than guessed. `sizeof!` reports `@sizeOf`, which rounds the backing integer up to its ABI alignment, so an exact layout claim MUST use `bitsizeof!`. `#[unchecked]` removes Zig's runtime safety checks for one function body; it does not prove the arithmetic is in range, and it does not disable the emitted `std.debug.assert` bounds guards. See 8.12.15–8.12.18.
- **10.53** The second batch of Zig intrinsics keeps the same width model as 10.52 and adds four more limits. Overflow is reported, not avoided: `add_with_overflow!` and friends only answer whether the operand type overflowed, so a caller that wants a guarantee must still branch on the flag or use the saturating operators. Rotate counts are converted with `@intCast`, so a negative or over-wide count traps instead of wrapping the way a hand-written shift might. `int_from_float!` truncates toward zero and does not range-check; an out-of-range float is undefined behaviour in ReleaseFast, and a 64-bit integer converted with `float_from_int!` may lose precision, because the target is fixed at `f64`. `memcpy_bytes!` shares one `count` between the two views but does not detect overlap, and `Simd::reduce_and`/`or`/`xor` are bitwise, so they are not defined for float lanes. `offsetof!` is deliberately not provided: `@offsetOf` reports byte offsets, so every field of a packed struct reports `0`, and a plain struct's fields may be reordered by Zig, which would make the answer a layout claim this backend cannot honour. See 8.12.19-8.12.23.

- **10.46** Ownership checking is a bounded syntactic pass, not a borrow checker: it tracks direct local bindings with a statically known type, follows straight-line blocks, `if`/`else`, `if let`, `match` arms and loop bodies, and treats a diverging branch as unreachable. It does not model partial moves of struct fields, moves through references, `RefCell`-style interior mutability, or method semantics it does not know, and it does not add lifetime or borrow tracking. Those cases remain the native backend's responsibility, so `E009` coverage is not guaranteed to be complete across both backends for programs that rely on unmodelled constructs. See 8.16.

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
- **11.15** v0.3.5 records completion of M3-remaining and M4-remaining, include-system hardening, M7 span/JSON diagnostics hardening, and the documentation, packaging, and CI tooling pass. It adds 8.1.5–8.1.7 for float, `char`, and `bool` cast differences and 10.28–10.31 for the corresponding limitations and the unreachable native E003 path.
- **11.16** v0.3.6 records the real native async event loop: a single-threaded `mio` reactor with task wakers, timers, Unix fd readiness, and an `asyncio` bridge. It updates 6.3, 6.4, 8.2, 10.2, and 10.8 for Send-aware cross-thread awaits and supported async `&self` methods, and adds 10.34 for the runtime-internal timer/I/O helpers. Async remains Experimental.
- **11.17** v0.3.7 records type-aware `Option::map` and `Result::map` support in both backends, retires 10.6, and adds the `map` row to 7.2.4. `Vec::map` retains collection semantics under the same receiver-type dispatch rule.
- **11.18** v0.3.8 records the bundled zero-dependency standard library: the `std:` include namespace, path containment and `E006` handling, `std:`-only deduplication, the eight shipped modules, and wheel packaging for `stdlib/*.poly`. It adds 9.2.1–9.2.3 and 10.35.
- **11.19** v0.3.8 also records the opt-in experimental pillar foundations: quiescent-point hot-patching, sandboxed compiler-pass evolution, bounded certificate-checked synthesis, deterministic cross-FFI checkpoint replay, and guarded speculative specialization. They are explicitly outside the stable conformance surface; 10.36 records their safety boundaries.
- **11.20** v0.3.8+ records the unified capability registry for the ten pillars (100 functions), the dual-backend host bridge and generated PyO3 binding scaffold, and the `#[gpu]`/`#[npu]` span-aware `E007` feature gate. Parity is defined as identical names and diagnostics on both backends, not as universal availability of privileged mechanisms. See 10.37 and `docs/unified_pillars.md`.
- **11.21** v0.3.9 lowers `vec![...]` (including `vec![value; count]`) and `String::from(...)` in both backends, adds lazy `Option`/`Result::unwrap_or_else`, adds the pure-`.poly` `std:fs` module and its trusted host seam (8.11), broadens `if let` to expression positions and builtin `Ok`/`Err`/`None` patterns (5.4), adds `HashMap` `entry(k).or_default()`, extends `#[terminates]` proofs to multiple self-recursive calls with call-site `E002` spans, expands `stdlib/algo` and `stdlib/str_utils`, and adds structured fix hints plus LSP diagnostic publication. It retires 10.12, 10.32, and 10.33 and adds 10.38–10.41.
- **11.22** v0.3.9+ documents the opt-in Zig execution mode: `get_backend("zig")` and `--backend zig` on `run`, `build`, `exec`, and `compile`; Zig-style saturating arithmetic (`+|`, `-|`, `*|`); arbitrary bit-width `uN`/`iN` integers; `#[parallel]` range-loop splitting; and `SimdN<T>` vector builtins with horizontal reductions. It adds 8.12 and 10.42 and records the mode's C-ABI truncation, narrow AST subset, cache, and lack of synchronization guarantees. The experimental mode is not part of the Python/native parity contract.
- **11.23** v0.3.9++ adds 9.20 and 13.3.7 (the ordered `hard-frames-parity` release-gate stage with fixed seeds), 13.2.7 (shared WASM preflight diagnostics and compile-time aggregate specialization with explicit dynamic-ABI boundaries), and records that the four Zig SIMD/parallel C-ABI defects are fixed. The three-way parity matrix now runs all 12 cases on all three backends with zero skips.
- **11.24** v0.3.9+++ adds 9.21 and 13.3.8: research frames H–K (bounded formal methods, termination/variant proofs, hard-state migration/replay, and adversarial verification) with explicit `Unknown`/bound/timeout boundaries and no WASM or JIT changes. The release gate gains a `research-frames` stage with an injectable in-process spec-oracle runner, and the wheel package list includes `tools.adversarial*`.
- **11.25** v0.3.9++++ upgrades the Zig backend for Zig 0.17.1 and adds three pure-Python language/runtime models. The Zig changes add 8.12.11–8.12.14: signed `/` and `%` lower to `@divTrunc`/`@rem`, slice indexing is bounds-checked via `_zig_at`/`_zig_at_mut`, `Result<T, PillarError>` lowers to `PillarError!T`, and labeled-block `break <value>` uses an untyped label. The new models add 8.13 (`compiler/gleam_style/`: total ADTs with exhaustive-match witnesses, bit-array patterns with checked bounds, cross-target capability contracts), 8.14 (`compiler/mojo_bridge/`: value/origin lifetimes, contiguous layouts with SIMD lane analysis, an honest MLIR probe, PEP 667-safe CPython interop), and 8.15 (`runtime/actor_otp/`: share-nothing actors, OTP supervision with restart budgets, let-it-crash fault reports and bounded telemetry). They add 10.43–10.45 and are outside the Python/native parity contract.
- **11.26** v0.3.9+ records shared ownership checking: `backends/move_check.py` runs before both backends and raises `E009` (8.16), `stdlib` was made Unicode-safe with `read_lines` and `Counter::top_key`, the CLI gained `polyglot new` and `polyglot explain`, and `examples/` gained three task-oriented programs plus a benchmark suite. It adds 8.16, 10.46, and moves the one native diagnostic that used `E009` for an out-of-range cast literal to `E008`, restoring the E001–E008 pillar surface.
- **11.27** v0.3.9++ records the expansion frontier: `distributed/frontier.py` (8.17), `tools/dst/` (8.18), `compiler/macro_engine/` (8.19), `runtime/ipc/` (8.20), and `bootstrap/` (8.21). Each is a bounded model whose limit is recorded in 10.47–10.51, and each reuses the project's existing E001–E008 diagnostic surface rather than adding codes. They add five named test files to the fast suite and sit outside the Python/native parity contract.
- **11.28** v0.3.9+++ gives the Zig backend a Zig-exclusive surface: five integer intrinsics (8.12.15), `#[packed]` structs (8.12.16), `bitcast!` with `bitsizeof!`/`sizeof!` (8.12.17), and `#[unchecked]` functions (8.12.18), implemented in `zig/intrinsics.py` as the single authority table plus a Python mirror. The width model and the rounded `sizeof!` limit are recorded in 10.52, `tests/test_zig_special.py` adds 19 cases with a worked example in `examples/zig_intrinsics.poly`, and every rejection reuses `E001`/`E003` rather than adding a code.

- **11.29** v0.3.9++++ extends the Zig-exclusive surface with a second batch: overflow reports and wrapping arithmetic (8.12.19), rotations (8.12.20), float/integer conversions (8.12.21), `alignof!` (8.12.22), byte-buffer `@memcpy`/`@memset` intrinsics (8.12.23), and `Simd::reduce_and`/`or`/`xor` in 8.12.8. All of it lives in `zig/intrinsics.py`, keeps the width model of 10.52, adds the limit entry 10.53, and reuses `E001`/`E003`; `tests/test_zig_special.py` grows to 34 cases.

## 12. Conformance Status

This section records the conformance status of the reference implementation against the acceptance criteria in Sections 1–9. It is informative with respect to requirements and normative with respect to conformance reporting. It does not modify any requirement in Sections 1–9.

Status values: **Pass**, **Partial**, **Fail**, **Blocked (environment)**.

### 12.1 Test suite baseline

As recorded in this session, under Python 3.14.8:

- `pytest -q`: 3374 passed, 6 skipped, 31 deselected.
- `pytest -q -m slow`: 31 passed, 3380 deselected.

The four previously failing Zig SIMD/parallel C-ABI cases in `tests/test_zig_backend.py` and `tests/test_auto.py` now pass: the generated wrappers use `extern struct` vector parameters/returns, the `std` import is emitted once, and stale zero-byte dylibs are evicted and rebuilt. The counts were re-measured after the priority feature batch (`vec!`, `String::from`, lazy `unwrap_or_else`, `std:fs`, expression-position `if let`, the extended entry API, multi-recursion termination proofs, expanded stdlib helpers, and structured diagnostics/fix hints), the 100-function framework registry with its ten frame test suites, the differential fuzzer / release gate / telemetry / packaging suites, the completed 1,000-function rollout (all ten batches, 1000 functions), the speculative JIT / WASM production / hard compiler-runtime frames with their focused suites, WASM compile-time aggregate specialization with shared `E001`–`E008` preflight diagnostics, the full three-way parity matrix (12/12 cases on all three backends with zero skips), the `hard-frames-parity` release-gate stage, and the Zig-exclusive intrinsics batches (`tests/test_zig_special.py`, 34 cases), replacing the earlier stale baseline.

- **12.1.1** The numbers above are from the current repository state at the time of writing. They MUST be updated whenever the test suite changes. The summary counts in 12.2.0 are derived from the 12.2 table below and MUST be recomputed whenever a row changes.
- **12.1.2** A criterion marked **Blocked** is not a conformance failure; it is a dependency on the test suite baseline being restored. The baseline is now restored; Blocked criteria have been re-evaluated in 12.2 and any that remain are marked **Partial** with a note recording what is still unverified.
- **12.1.3** `README.md` MUST record the same baseline as 12.1 before a conformance claim is made. See 12.3.

### 12.2 Acceptance criteria status

**12.2.0** Summary: 76 Pass, 3 Partial, 0 Fail, 0 Blocked.

**12.2.1** **Partial** means at least one test exists and passes, but the criterion is not fully covered. The Note column MUST state what is missing. A criterion marked **Partial** with the note "Tests not yet recorded" means the test may exist but is not traceable to a named test in `NOTES.md` or `tests/`. Such a criterion MUST be reclassified as **Fail** if, on inspection, no such test exists.

| Section | Criterion                                                                             | Covers clause             | Status  | Note                                                                                                                                                                                                                                                                                                                                                                |
| ------- | ------------------------------------------------------------------------------------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | Each Platform-defined feature has a test                                              | 1.7                       | Pass    | Box pattern native rejection covered by `tests/test_native_n2a_patterns.py::test_pattern_box_native_rejected`                                                                                                                                                                                                                                                       |
| 1       | No platform difference outside Section 8                                              | 1.6                       | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 2       | Parity test per Normative type                                                        | 2.1–2.3                   | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 2       | Negative-index and out-of-range tests                                                 | 2.4.2                     | Pass    | Python backend covered                                                                                                                                                                                                                                                                                                                                              |
| 2       | Native rejects `i64` index without conversion                                         | 2.4.4                     | Pass    | `tests/test_n7_usize.py::test_native_i64_index_requires_explicit_cast`                                                                                                                                                                                                                                                                                              |
| 2       | Every Deferred requirement in Section 10                                              | 2.5, 2.7                  | Pass    | 2.5 and 2.7 remain recorded as Deferred                                                                                                                                                                                                                                                                                                                             |
| 2       | Range loop variable preserves the range element type                                  | 2.4.14                    | Pass    | `tests/test_n7_usize.py::test_native_range_variable_inherits_usize_type`, `::test_python_range_variable_inherits_usize_type`                                                                                                                                                                                                                                        |
| 2       | Parser accepts `#[terminates]`                                                        | 2.8.1                     | Pass    | `tests/test_m10_termination.py::test_terminates_attribute_is_parsed_on_and_off`                                                                                                                                                                                                                                                                                     |
| 2       | AST carries the termination flag                                                      | 2.8.1                     | Pass    | `tests/test_m10_termination.py::test_fn_decl_terminates_flag_is_preserved_in_ast`                                                                                                                                                                                                                                                                                   |
| 2       | Checker accepts bounded `for` loops                                                   | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                                                                                                             |
| 2       | Checker accepts decreasing `while` variants                                           | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                                                                                                             |
| 2       | Checker accepts decreasing recursion                                                  | 2.8.3                     | Pass    | `tests/test_m10_termination.py::test_python_terminating_patterns_execute`, `::test_native_terminating_patterns_execute`                                                                                                                                                                                                                                             |
| 2       | Checker rejects non-decreasing recursion                                              | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                                                                                                   |
| 2       | Checker rejects increasing `while` variants                                           | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                                                                                                   |
| 2       | Checker rejects `loop` without `break`                                                | 2.8.4                     | Pass    | `tests/test_m10_termination.py::test_python_non_terminating_patterns_rejected`, `::test_native_non_terminating_patterns_rejected`                                                                                                                                                                                                                                   |
| 2       | Both backends produce the same termination error                                      | 2.8.5                     | Pass    | `tests/test_m10_termination.py::test_termination_error_message_is_backend_identical`                                                                                                                                                                                                                                                                                |
| 2       | Unannotated functions are not checked                                                 | 2.8.2                     | Pass    | `tests/test_m10_termination.py::test_checker_ignores_unannotated_functions`, `::test_python_unannotated_recursion_compiles_without_running`, `::test_native_unannotated_recursion_compiles_without_running`                                                                                                                                                         |
| 3       | Parity test per mutability rule                                                       | 3.1–3.3                   | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 3       | Native code preserves declared mutability                                             | 3.4                       | Partial | No dedicated test name recorded in NOTES                                                                                                                                                                                                                                                                                                                            |
| 4       | Numbered test per pattern form                                                        | 4.1                       | Pass    | N2a batches 1–5                                                                                                                                                                                                                                                                                                                                                     |
| 4       | Rejection tests with error class and location                                         | 4.3, 4.6                  | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 4       | Exhaustiveness-diagnostic test                                                        | 4.2                       | Pass    | Diagnostic text asserted in `tests/test_spec_conformance.py`                                                                                                                                                                                                                                                                                                        |
| 5       | Synchronous `?` parity test                                                           | 5.2                       | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 5       | Async `?` documented                                                                  | 5.3                       | Pass    | 8.5.3                                                                                                                                                                                                                                                                                                                                                               |
| 5       | Type-aware `Option::map`, `Result::map`, and `Vec::map` dispatch                      | 5.7                       | Pass    | `tests/test_tier1_option_result_map.py`                                                                                                                                                                                                                                                                                                                             |
| 5       | Platform-defined test for each of `is_ok`, `is_err`, `map_err`, `ok_or`, `ok_or_else` | 5.8                       | Pass    | `tests/test_n9_result.py` contains a dedicated test per combinator and backend                                                                                                                                                                                                                                                                                      |
| 6       | Native async free-function test                                                       | 6.4                       | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 6       | Async embedding test                                                                  | 6.4                       | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 6       | Python async free-function test                                                       | 6.4.1                     | Partial | Synchronous execution asserted indirectly                                                                                                                                                                                                                                                                                                                           |
| 6       | Native async `&self` / `&mut self` impl-method tests                                  | 6.4                       | Pass    | `tests/test_native_n5.py::test_native_async_impl_method`; borrow-guard lowering in `tests/test_known_limitations.py::test_async_mut_self_method_native_codegen_uses_borrow_guard`                                                                                                                                                                                   |
| 6       | Deferred items in Section 10                                                          | 6.4, 6.5                  | Pass    | 10.7, 10.8 (by-value receivers)                                                                                                                                                                                                                                                                                                                                     |
| 7       | Function table with status markers                                                    | 7.2                       | Pass    | 7.2.1–7.2.8                                                                                                                                                                                                                                                                                                                                                         |
| 7       | No Platform-defined function lacks Section 10 entry                                   | 7.3                       | Pass    | 10.13–10.18, 10.20                                                                                                                                                                                                                                                                                                                                                  |
| 7       | HashMap supported methods, type sets, and dictionary boundary                         | 7.2.6–7.2.7, 8.10.1       | Pass    | `tests/test_n8_hashmap.py`                                                                                                                                                                                                                                                                                                                                          |
| 7       | HashMap `get_mut` reference at native Python boundary                                 | 8.10.2                    | Partial | Native returns an owned copy because the input map is passed by value; see 10.20                                                                                                                                                                                                                                                                                    |
| 8.1     | Native overflow test                                                                  | 8.1.1                     | Pass    | `tests/test_native_n8_types.py::test_native_overflow_wraps`                                                                                                                                                                                                                                                                                                         |
| 8.1     | Python arbitrary-precision/overflow test                                              | 8.1.2                     | Pass    | `tests/test_m4_integers.py::test_i128_u128_above_2_64_python` exercises values above 2^64; `tests/test_native_n8_types.py::test_python_overflow_wraps` covers typed wrapping                                                                                                                                                                                        |
| 8.1     | Float-to-int `as` cast documented and tested                                          | 8.1.5                     | Pass    | `tests/test_m4_integers.py::test_float_to_int_truncates_toward_zero_both_backends`, `::test_float_to_int_range_difference_is_documented`                                                                                                                                                                                                                            |
| 8.1     | `char` casts documented and tested                                                    | 8.1.6                     | Pass    | `tests/test_m4_integers.py::test_char_as_i64_both_backends`, `::test_python_integer_as_char_invalid_code_points`, `::test_native_integer_as_char_other_widths_rejected`                                                                                                                                                                                             |
| 8.1     | `bool` casts documented and tested                                                    | 8.1.7                     | Pass    | `tests/test_m4_integers.py::test_bool_as_i64_both_backends`, `::test_python_i64_as_bool`, `::test_native_i64_as_bool_rejected`                                                                                                                                                                                                                                      |
| 8.2     | Native async test                                                                     | 8.2.2                     | Pass    | `tests/test_native_n5.py::test_native_async_add`, `::test_native_async_identity`                                                                                                                                                                                                                                                                                    |
| 8.2     | Cross-thread await test for Send captures                                             | 8.2.3                     | Pass    | `tests/test_native_n5.py::test_native_async_send_add_across_threads`; main-thread parity in `::test_native_async_send_add_on_main_thread`                                                                                                                                                                                                                           |
| 8.3     | Python Box pattern test                                                               | 8.3.1                     | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 8.3     | Native rejection test                                                                 | 8.3.2                     | Pass    | `tests/test_native_n2a_patterns.py::test_pattern_box_native_rejected` (parametrized over five Box pattern forms)                                                                                                                                                                                                                                                    |
| 8.4     | Python payload slice test                                                             | 8.4.1                     | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 8.4     | Native limitation test                                                                | 8.4.2                     | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 8.5     | Synchronous `?` parity test                                                           | 8.5.1                     | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 8.5     | Async `?` documented                                                                  | 8.5.3                     | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 8.5     | Native `Result::and_then` and `Result::or_else`                                       | 8.5.4                     | Pass    | `tests/test_n9_result.py::test_native_result_and_then_and_or_else`                                                                                                                                                                                                                                                                                                  |
| 8.5     | Platform-defined test for `Result::is_ok`, `is_err`, `map_err`                        | 8.5.5                     | Pass    | Dedicated Python and native tests in `tests/test_n9_result.py`                                                                                                                                                                                                                                                                                                      |
| 8.5     | Platform-defined test for `Option::ok_or`, `ok_or_else`                               | 8.5.6                     | Pass    | Dedicated Python and native tests in `tests/test_n9_result.py`                                                                                                                                                                                                                                                                                                      |
| 8.6     | `Vec::len()` parity test                                                              | 8.6.1                     | Pass    | `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `::test_native_explicit_len_cast_and_usize_return`                                                                                                                                                                                                                                        |
| 8.6     | Range-validation tests                                                                | 8.6.2–8.6.3               | Pass    | `tests/test_n7_usize.py::test_python_usize_cast_overflow_raises`, `::test_python_backend_target_width_is_instance_scoped`                                                                                                                                                                                                                                           |
| 8.7     | Native parse failure test                                                             | 8.7.1                     | Pass    | `tests/test_spec_conformance.py::test_native_invalid_literal_parse_fails_during_compile`, `::test_native_dynamic_parse_failure_remains_a_runtime_error`                                                                                                                                                                                                             |
| 8.7     | Python `ValueError` test                                                              | 8.7.2                     | Pass    | `tests/test_spec_conformance.py::test_python_invalid_literal_parse_raises_value_error`, `::test_python_dynamic_parse_failure_raises_value_error`                                                                                                                                                                                                                    |
| 8.7     | Parity on successful parse                                                            | 8.7.3                     | Pass    | `tests/test_spec_conformance.py::test_parse_success_matches_both_backends`                                                                                                                                                                                                                                                                                          |
| 8.8     | Native boundary tests                                                                 | 8.8.1–8.8.4               | Pass    | `tests/test_spec_conformance.py::test_native_boundary_errors_are_not_silent`                                                                                                                                                                                                                                                                                        |
| 8.8     | Python boundary tests                                                                 | 8.8.1–8.8.4               | Pass    | `tests/test_spec_conformance.py::test_python_boundary_errors_are_not_silent`                                                                                                                                                                                                                                                                                        |
| 8.8     | Parity that both raise                                                                | 8.8.1–8.8.4               | Pass    | Both parametrized boundary tests in `tests/test_spec_conformance.py`                                                                                                                                                                                                                                                                                                |
| 8.9     | Native non-exhaustive test                                                            | 8.9.2                     | Pass    | `tests/test_spec_conformance.py::test_native_non_exhaustive_known_match_is_compile_error`                                                                                                                                                                                                                                                                           |
| 8.9     | Python non-exhaustive test                                                            | 8.9.3                     | Pass    | `tests/test_spec_conformance.py::test_python_non_exhaustive_known_match_warns_and_raises`                                                                                                                                                                                                                                                                           |
| 9       | CLI exists                                                                            | Acceptance criterion      | Pass    | `api/cli.py`                                                                                                                                                                                                                                                                                                                                                        |
| 9       | Quick Start validated                                                                 | 9.5                       | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 9       | Examples directory exists                                                             | Acceptance criterion      | Pass    | —                                                                                                                                                                                                                                                                                                                                                                   |
| 9       | Test status matches Section 12                                                        | Acceptance criterion      | Pass    | README and 12.1 both record the current test baseline                                                                                                                                                                                                                                                                                                               |
| 9       | Package installs via pip metadata and build artifacts                                 | F1.1                      | Pass    | `pyproject.toml` declares package metadata and console script; `python -m build` produces wheel and sdist                                                                                                                                                                                                                                                           |
| 9       | `polyglot run` executes a file                                                        | F1.2                      | Pass    | `tests/test_f1_cli.py::test_run_hello_exits_zero_and_prints_five`                                                                                                                                                                                                                                                                                                   |
| 9       | `polyglot build` produces an extension                                                | F1.3                      | Pass    | `tests/test_f1_cli.py::test_build_hello_produces_extension` (skipped if cargo unavailable)                                                                                                                                                                                                                                                                          |
| 9       | `polyglot check` reports termination failures                                         | F1.4                      | Pass    | `tests/test_f1_cli.py::test_check_terminates_example_exits_zero`, `::test_check_non_terminating_reports_error`                                                                                                                                                                                                                                                      |
| 9       | Quick Start runs in under 10 minutes                                                  | F1.5                      | Pass    | README Quick Start uses `pip install`, a short `hello.poly`, and `polyglot run`; validated by CLI tests and examples                                                                                                                                                                                                                                                |
| 9       | `examples/` contains seven runnable files                                             | F1.6                      | Pass    | `tests/test_f1_cli.py::test_examples_directory_contains_seven_runnable_files`                                                                                                                                                                                                                                                                                       |
| M6      | `include!` splices items and runs in both backends                                    | 9.2, 10.23                | Pass    | `tests/test_m6_include.py::test_include_add_both_backends`, `::test_include_add_native_backend`                                                                                                                                                                                                                                                                     |
| M6      | Nested and relative-path includes resolve correctly                                   | 9.2                       | Pass    | `tests/test_m6_include.py::test_nested_include`, `::test_include_path_relative_to_including_file`                                                                                                                                                                                                                                                                   |
| M6      | Cyclic/missing includes rejected with five-line diagnostics                           | 9.2, 10.23                | Pass    | `tests/test_m6_include.py::test_cyclic_include_rejected`, `::test_missing_include_file`                                                                                                                                                                                                                                                                             |
| M6      | Duplicate, `..`, absolute, include-only, block, and non-string includes               | 9.2                       | Pass    | `tests/test_m6_include.py::test_duplicate_include_is_allowed_and_splices_twice`, `::test_include_with_parent_directory_segments_is_allowed`, `::test_include_with_absolute_path_outside_repo_is_allowed`, `::test_include_only_file_produces_empty_program`, `::test_include_inside_block_is_parse_error`, `::test_include_with_non_string_argument_is_parse_error` |
| M6      | Bundled `std:` namespace resolves, dedups, and ships                                  | 9.2.1–9.2.3, 10.35        | Pass    | `tests/test_std_include.py` covers resolution, traversal, cycles, and dedup; `tests/test_stdlib.py` covers every module on both backends, packaging, `E006`, and `NdArray` dedup                                                                                                                                                                                    |
| M7      | Stable error codes and span-aware diagnostics                                         | 2.4.7, 9.3, 10.24         | Pass    | `tests/test_m7_diagnostics.py` covers E001–E008 span formatting, E007 native rejection, E008 literal span, E005 multi-file chain, E006 resolved path, and JSON diagnostics for E001/E002/E005                                                                                                                                                                       |
| M4      | Additional integer types, `as` casts, and E001 boundaries                             | 2.5, 8.1, 10.26           | Pass    | `tests/test_m4_integers.py` covers the full Python/native integer matrix, literal and variable casts, E001 boundaries, `i128`/`u128` above 2^64, float/char/bool casts                                                                                                                                                                                              |
| M3      | HashMap API completion, BTreeMap, HashSet, nested values, and `get_mut` semantics     | 7.2.6–7.2.8, 10.25, 10.27 | Pass    | `tests/test_m3_collections.py` covers Python and native `BTreeMap`, `HashSet`, entry methods, nested HashMap depth-1 values, sorted/unspecified iteration, and `get_mut` copy semantics                                                                                                                                                                             |

### 12.3 Outstanding items that block a v0.3.8 conformance claim

No item blocks a v0.3.8 conformance claim, provided that:

- Every Partial in 12.2 is explicitly accepted as a known gap in this version.
- The 3 Partial criteria are listed in 12.3.1 with a justification for why each does not block conformance.

### 12.3.1 Accepted Partial items for v0.3.8

| Partial item                                        | Justification                                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 3.4 mutability preservation test                    | Behaviour exists; only the test name is missing from NOTES                                                                     |
| 6.4.1 Python async synchronous execution            | Behaviour is backend-specific per 11.3.1; not a parity gap                                                                     |
| 8.10.2 native `HashMap::get_mut` boundary reference | The by-value native map cannot safely return a mutable reference; the documented platform behaviour returns a copy. See 10.20. |

Each item above is either (a) covered by a Deferred clause, (b) a test-recording gap rather than a behaviour gap, or (c) explicitly assigned to next milestone in the roadmap.

## 13. Production JIT, WASM, and hard-frame requirements

This section records the requirements for the speculative JIT frame, the WASM
production frame, and the hard compiler/runtime frames that sit above the
1,000-function rollout. These are model-and-tooling contracts: they must remain
deterministic, structured-diagnostic, and safe on every host, even when the
privileged operation they model is available only on one platform.

### 13.1 Frame A — speculative JIT (`runtime.jit`)

- **13.1.1** `jit_compile_hot_trace` MUST count executions per trace key and
  compile only at the hotness threshold; redefining a key with a different
  shape MUST raise `E008` rather than serve stale code.
- **13.1.2** Native codegen and executable-memory allocation MUST be gated on
  an explicit caller opt-in and a supported host (macOS/arm64 `MAP_JIT`). All
  other hosts and callers MUST receive a fully functional _model_ trace so the
  state machine remains testable.
- **13.1.3** Guards MUST be type, non-null, and bounds checks evaluated
  against concrete values, returning a report on failure (not an exception).
  A failed guard MUST deoptimize via `jit_deoptimize_to_interpreter_frame`,
  preserving all live values, the stack, and the source span. Guard exhaustion
  MUST produce `E008`.
- **13.1.4** Executable blocks MUST be immutable once published (W^X after
  commit) and MUST NOT be reclaimed while an active frame pins them. Inline
  caches and call sites MUST be patched copy-on-write under a per-site lock,
  deferring patches that would mutate live code.
- **13.1.5** Redefinition and hotpatch events MUST advance a version and epoch,
  invalidate dependent traces, clear call sites, and reclaim or defer code
  blocks. A hotpatch target that does not match the trace key MUST raise
  `E008`. Every diagnostic in this frame MUST carry the trace span.

### 13.2 Frame B — WASM production frame (`backends.wasm.framework`)

- **13.2.1** `wasm_lower_ir_to_module` MUST lower source, AST, HIR, or validated
  MIR to valid WAT and (where `wasmtime` is installed) a `.wasm` binary.
  Unsupported source constructs MUST raise structured `E003` diagnostics with
  a source span; a missing runtime MUST raise `E007`.
- **13.2.2** Linear memory MUST respect the `wasm32` 4 GiB ceiling (checked
  limit `2**32` by default). Field layout MUST be aligned, bump-allocated, and
  bounds-checked against a 16-byte-aligned shadow stack before any access.
- **13.2.3** The ABI MUST round-trip structs, tagged `Result`, strings, and
  slices deterministically; malformed shapes, out-of-range integers, and
  truncated payloads MUST raise structured diagnostics (`E001`/`E008`).
- **13.2.4** Host imports MUST be explicitly declared and implemented; export
  signatures MUST match the module's declared signatures. Signature drift or
  an unknown delegation target MUST raise `E001`. No ambient filesystem,
  clock, or process capability may be imported.
- **13.2.5** Trap/OOM paths MUST be reported as structured results (or `E008`
  in strict mode) without corrupting memory. Each isolated instance MUST use a
  fresh `Engine`/`Store`/`Memory` so no state leaks between runs; compiled
  modules may be cached only under a caller-supplied temp root.
- **13.2.6** `wasm_verify_dual_backend_parity` MUST compare normalized
  Python/Rust-PyO3/WASM outcomes and record WASM-unsupported constructs as
  explicit boundary skips, not false mismatches.
- **13.2.7** Before WAT emission, `wasm_preflight_validate` MUST reuse the
  shared frontend error types so the normalized `E001`–`E008` codes and messages
  are identical to the Python and native backends. `lower_wasm_aggregates` MUST
  monomorphize generics and specialize compile-time struct literals, vectors,
  enum variants, and bounded generic methods into scalar `i64`/`bool`
  expressions. Scalar locals MUST remain real WASM locals so mutation and
  loop-carried values are preserved. An aggregate function parameter, return
  value, or other dynamic aggregate ABI MUST raise structured `E003` with the
  source position instead of producing an invalid module.

### 13.3 Frame C — hard compiler/runtime frames

- **13.3.1** Mutually recursive generic expansion MUST unify aliases, detect
  cycles, and enforce a bounded specialization budget; exceeding the budget
  MUST raise `E002` and undefined references `E005`, never `RecursionError`.
- **13.3.2** Trait analysis MUST reject overlapping implementations (`E001`),
  missing associated types and unsatisfied higher-rank bounds (`E002`).
- **13.3.3** CFG verification MUST compute dominance sets, identify natural
  loop headers, and reject back edges whose header does not dominate the source
  (`E005`). Move analysis MUST be branch-aware and join states MUST agree.
- **13.3.4** RAII cleanup MUST schedule drops on all paths and reject a leak on
  an early-return or unwind path with `E002`.
- **13.3.5** Hotpatch concurrency models MUST reject reclamation while readers
  are active, descriptor loss during migration, raw-address TOCTOU (address or
  version drift), and failed verification MUST roll back atomically; all
  failures are `E008`.
- **13.3.6** Differential parity MUST be seed-reproducible, use isolated
  per-backend cache scopes and temp paths, normalize codes/failure modes rather
  than exception identities, and persist seed + source + normalized outcomes
  only on a genuine divergence.
- **13.3.7** The release gate MUST include a `hard-frames-parity` stage between
  `pytest-core` and `diff-fuzz`. It MUST run the JIT/WASM/hard-frame suites and
  the three-way parity seeds `1`, `42`, `9999`, and `2026`; each seed result is
  an independently reported check, and any mismatch or harness error blocks the
  gate. The per-seed runner MUST be injectable so unit tests never compile
  native code.
- **13.3.8** The release gate MUST include a `research-frames` stage between
  `hard-frames-parity` and `diff-fuzz`. It MUST run the H–K focused suites and
  an adversarial spec-oracle check as separately reported checks; any failed
  or missing check blocks the gate. The oracle runner MUST be injectable so
  verification tests never spawn native, JIT, or WASM code, and the gate MUST
  report that a passing oracle is bounded rather than a complete proof.

### 13.4 Platform and security caveats

- **13.4.1** `MAP_JIT` and the per-thread write-protect toggle are Apple
  Silicon/macOS specific; other hosts run model mode only. PEP 667 frame
  semantics and raw-address TOCTOU are modelled, not exercised against a live
  interpreter frame by default.
- **13.4.2** Native execution and host-import bridges are privileged paths:
  they require `trusted=True` and explicit opt-in. They are never enabled by
  untrusted input handling.
