# Changelog

Version history derived from SPECIFICATION.md Section 11.

## Unreleased — Apple Silicon JIT backend and edge-case audit

- Added a shared monomorphization pass (`monomorph.py`) that resolves generic
  functions, structs, enums, and impl blocks into concrete mangled
  declarations such as `Box_i64`, `Box_String`, and `identity_i64`, with
  deduplication and a depth-32 recursion limit that rejects infinite generic
  expansion as `E003`.
- Added trait-bound parsing (`<T: Display + Clone>`), preserved generic
  arguments on paths and struct literals, and taught the Python, native Rust,
  and JIT backends to consume fully concrete generic output.
- Added `tests/test_generics.py` covering bounds, nested arguments, turbofish
  syntax, deduplication, recursion rejection, Python/native parity for
  `Box<T>` and `identity<T>`, and the JIT's aggregate-generic limitation.
- Added a zero-dependency AArch64 JIT backend (`backends/jit/`) that lowers
  typed SSA to machine words directly, manages `MAP_JIT` executable memory,
  performs linear-scan register allocation with spilling, resolves cyclic phi
  copies, and reports division/shift traps at the Python boundary. Wired into
  `get_backend("jit")` and `polyglot run/compile --backend jit`.
- Fixed parser precedence for bitwise operators, added `^` (XOR) to the lexer,
  parser, and Python backend, and prevented a block-form `if` from capturing
  the following parenthesised expression.
- Fixed the `#[terminates]` checker so unsigned counters cannot use wrapped
  subtraction (`i >= 0`, or steps larger than one) as a termination proof.
- Hardened `include!` targets: directories and unreadable files now report
  E006 instead of raising tracebacks.
- Fixed Python/JIT drift for narrow-integer wrapping, Rust-style constant
  division/modulo folding, and comparison/logical-result typing.
- Added `tests/test_jit_backend.py` and `tests/test_edge_cases.py`; see
  `BUGS_AND_EDGE_CASES.md` for the audit details and known limitations.
- Added a zero-dependency bundled standard library under `stdlib/` with eight
  pure-`.poly` modules: `math`, `algo`, `str_utils`, `collections`, `numpy`,
  `linalg`, `random`, and `prelude`.
- Added `include!("std:<name>")` resolution against the packaged `stdlib/`
  directory, including `.poly` suffix inference, traversal rejection with
  `E006`, cycle detection with `E005`, and resolved-path deduplication for
  `std:` includes only. Ordinary includes still splice duplicates.
- Shipped `stdlib/*.poly` in the wheel and added `tests/test_stdlib.py`, which
  exercises every module on both backends, checks the inverse identity within
  `1e-9`, exact `solve` roots, the reference Xoshiro256** sequence, packaging,
  `E006`, and `NdArray` deduplication.
- Fixed parser handling of block-form `if`/`while`/`for`/`match` conditions so
  `if a < b { a } else { b }` is not misread as a struct literal.
- Fixed Python method dispatch for struct-typed bindings and fields, switched
  `.clone()` lowering to `deepcopy` for nested vectors, and preferred user
  `impl` methods over built-in collection rewrites in both backends.
- Exposed receiver-less associated functions as static methods in native
  code, lowered method `Result<T, String>` returns to `PyResult<T>`, and
  improved Python/native expression type inference for field, call, and
  `match` expressions.
- Added a dependency-free interactive visualizer (`polyglot visualizer`,
  alias `gui`) that renders the color-coded AST, summarizes and diffs generic
  versus monomorphized source, overlays E001–E008 character spans in the
  editor, and runs Python and native backends side by side with compile/run
  timings, peak RSS, exit codes, and return values.
- Added `polyglot exec` and `api/visualizer.py` / `api/visualizer_server.py`
  to support the visualizer without new runtime dependencies; user programs
  execute only in isolated `api.cli exec` subprocesses with timeout and
  process-group cleanup.
- Added `tests/test_visualizer.py` and packaged `api/static/*.html` in the
  wheel; documented the UI and its local-trusted-code security model in
  `docs/visualizer.md`.
- Added five opt-in experimental pillar foundations, documented in
  `docs/experimental_pillars.md`: quiescent-point hot-patching with live
  struct migration and atomic native symbol slots (`backends/hotpatch.py`),
  sandboxed compiler-pass synthesis with versioned history
  (`compiler/evolution.py`), bounded interval/polynomial synthesis with
  independently checked certificates and dual-backend emission
  (`proofs/synthesis.py`), deterministic Python/native checkpoint replay with
  retroactive edits and object-graph snapshots (`debug/timetravel.py`), and
  guarded background specialization with a fallback deopt path
  (`runtime/transmute.py`). Shared diagnostics reuse `E001`–`E008`.
- Fixed Python lowering of `assert!(condition, "fmt {}", ...)`: it now lowers
  to the `_assert` runtime helper with the condition and format arguments
  instead of emitting an invalid `assert()` call. The native backend continues
  to emit Rust's `assert!`.
- Extended setuptools package discovery to `compiler*`, `debug*`, `proofs*`,
  and `runtime*`, and added `tests/test_hotpatch.py`,
  `tests/test_evolution.py`, `tests/test_synthesis.py`,
  `tests/test_timetravel.py`, and `tests/test_transmute.py`.
- Added an opt-in `distributed/` runtime for live actor migration across
  nodes: a versioned CRC-checked codec for Python graphs and native
  `NativeLayout`/`NativeValue` state, checkpointable await journals, anchored
  connections with cursor replay and idempotent writes, epoch/fencing leases
  for distributed borrow checking, an in-memory partitionable network plus a
  TCP transport, and checkpoint-based recovery with `E008` rollback and
  node-loss diagnostics. See `docs/distributed.md`. Migration is cooperative
  replay, not opaque CPython/Rust stack copying; raw descriptors and pointers
  are never transported.
- Added `tests/test_distributed.py` and extended setuptools package discovery
  to `distributed*`.
- Added a unified 10-pillar / 100-function capability registry
  (`runtime/capabilities.py`). It classifies every directive function as a
  foundation, a deterministic in-process model, or an explicit rejection, and
  gives both backends the same 100-name namespace and span-aware
  `E001`–`E008` diagnostics. Rejected mechanisms include executable-text
  patching, live-frame rewriting, privileged page/syscall hooks, real
  accelerator dispatch, and threshold signatures.
- Added `distributed/dsm.py`, `security/sandbox.py`,
  `compiler/heterogeneous.py`, and `swarm/agent.py` with dependency-free
  reference models for CRDT/consensus, sandbox policy/taint/fuel, MLIR-style
  accelerator lowering with CPU fallback, and gossip/self-healing.
- Added `runtime/native_bindings.py` and the generated opt-in `native_pillars/`
  crate: one `#[pyfunction]` per capability with honest `E006`/`E007`
  diagnostics. It is not built by `polyglot build` or CI.
- Documented the capability matrix in `docs/unified_pillars.md` and aligned the
  `#[gpu]`/`#[npu]` feature gate with the accelerator boundary.
- Added a Zig backend and lowering pass, including Zig-style `errdefer` and the
  saturating `+|`, `-|`, and `*|` operator family.
- Fixed recursive generic struct and enum expansion by caching the in-progress
  instantiation before transforming variants, so `Node<T>` and `Tree<T>` reach
  a fixed point instead of exhausting the recursion limit.
- Added native async `&mut self` impl methods. Mutable fields are copied before
  the future runs, an atomic borrow guard rejects overlapping calls, and field
  writes are committed only when the future completes successfully.

## v0.3.7 — type-aware Option/Result map

- Lowered `Option::map` and `Result::map` in both backends through
  receiver-type dispatch instead of rejecting them with E003.
- Added free-function return-type tracking so calls returning `Option` or
  `Result` dispatch `map` correctly, including chained maps.
- Kept `Vec::map` collection lowering unchanged and covered the
  `Option::map` / `Result::map` / `Vec::map` split with Python and native
  parity tests.
- Retired SPEC 10.6 and added the `Result` `map` row to 7.2.4.

## v0.3.6 — real async event loop

- Fixed Python backend `match` lowering for built-in `Option` and `Result`
  patterns: `Some(x)`, `None`, `Ok(x)`, and `Err(e)` now match the runtime
  representation, including qualified paths, nested payloads, guards, and
  wildcard arms. Match scrutinees are evaluated once.
- Aligned `pyproject.toml` package version with the SPEC title and added a
  packaging test that prevents future version drift.
- Replaced the no-op waker with per-task wakers that enqueue work on a
  single-threaded native reactor backed by `mio`.
- Added timer wakeups and Unix file-descriptor readiness to the reactor,
  with internal `polyglot_sleep_ms` and `polyglot_readable` helpers for
  native tests.
- Bridged pending `PyFuture` iterations to Python `asyncio` futures and
  completed them thread-safely through the running loop.
- Reworked `await_with_timeout` to wait on a condition-variable waker with
  the GIL released instead of busy-polling.
- Documented Send-aware cross-thread awaits and updated the async
  platform-difference clauses. Async remains Experimental.

## v0.3.5 — M3/M4 completion and hardening

- Completed native `BTreeMap<K, V>` for `K` in `{i64, String}` and `V` in
  `{i64, String, bool, f64, Vec<i64>}`, with sorted iteration.
- Completed native `HashSet<T>` for `T` in `{i64, String}` and nested
  `HashMap` values at depth 1.
- Completed the extended `HashMap` entry API (`or_insert_with`,
  `and_modify`, `or_insert_with_key`) and collection `get_mut`
  copy-at-Python-boundary semantics.
- Completed the cross-backend fixed-width integer matrix and documented
  float, `char`, and `bool` cast differences.
- Hardened `include!` handling for duplicate, parent-directory, absolute,
  block-position, and non-string cases.
- Hardened span and JSON diagnostics, and recorded the unreachable native
  E003 path.
- Added the docs/CI tooling pass: `Makefile`, `docs/`, docs-sync checks,
  formatting checks, build benchmark, and `CHANGELOG.md`.

## v0.3.4 — M3 remainder

- Specified `BTreeMap` and `HashSet` as collections-layer types and
  extended the `HashMap` entry API targets (`or_insert_with`,
  `and_modify`, `or_insert_with_key`).
- Recorded the supported `BTreeMap`, `HashSet`, nested `HashMap` depth-1,
  and collection `get_mut` type sets.
- Marked native collection support as Partial pending completion; the
  implementation landed in v0.3.5.

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
