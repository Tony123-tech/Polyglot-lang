# Changelog

Version history derived from SPECIFICATION.md Section 11.

## Unreleased — Runtime failures carry a code, a position, and a fix hint

- Runtime guards now report `error[E002]: <message>` with `note: runtime failure`
  and `at: line N, column M` instead of leaking the host exception's name and
  printing `at: <unknown>`. The Python backend raises `RuntimeFailure` (exported
  from `backends.python`) from integer `/` and `%`, bounds-checked indexing,
  `panic!`, and the assert helpers; the CLI prints it in the five-line format.
  Adds SPEC 2.9.
- The Zig backend prints the same first line and the same location line through
  `_zig_fail`, `_zig_div`, `_zig_rem`, and `_zig_assert`, then exits non-zero.
  Same source, same `error[E002]` line, same `at: line N, column M`.
- Fixed: `1.0 / 0.0` raised `ZeroDivisionError` in the Python backend. Rust
  floats follow IEEE 754, so it now yields a signed infinity and `0.0 / 0.0`
  yields NaN.
- Adds `tests/test_runtime_failures.py` (16 cases) and SPEC 10.54 for what the
  guards deliberately do not cover. Fast baseline: 3390 passed.

## Unreleased — Zig overflow reports, rotations, conversions, byte buffers

- Added `add_with_overflow!` / `mul_with_overflow!` (yielding the `i64` flag) and
  `wrapping_add!` / `wrapping_mul!` (yielding the wrapped value at the operand
  width), all lowering to `@addWithOverflow` / `@mulWithOverflow`. The builtin's
  tuple is indexed directly because struct construction is outside the subset.
  Adds SPEC 8.12.19.
- Added `rotl!` / `rotr!` lowering to `std.math.rotl` / `std.math.rotr`, with the
  rotation bounded by the declared width. Adds SPEC 8.12.20.
- Added `int_from_float!` (→ `i64`, truncating toward zero) and
  `float_from_int!` (→ `f64`), each rejecting the other operand family. Adds
  SPEC 8.12.21.
- Added `alignof!` → `@alignOf`, which completes the layout trio with
  `bitsizeof!` and `sizeof!` (a 9-bit packed struct reports 2, matching
  `sizeof!`). Adds SPEC 8.12.22.
- Added `memcpy_bytes!` / `memset_bytes!` → `@memcpy` / `@memset` over
  `dst[0..count]` and `src[0..count]`, rejecting a non-pointer destination and
  stating that overlapping regions are not detected. Adds SPEC 8.12.23.
- Added `Simd::reduce_and` / `reduce_or` / `reduce_xor` → `@reduce(.And/.Or/.Xor, v)`
  to 8.12.8.
- `zig/intrinsics.py` gained `int_shape`, `require_arithmetic_operand`,
  `require_matching_operands`, `require_float_operand`, `mirror_overflow`
  (returns the wrapped value and the flag) and `mirror_rotate`.
- New tests: 15 more in `tests/test_zig_special.py` (34 total); the example
  `examples/zig_intrinsics.poly` now also checks `rotl!`, an overflow flag,
  `int_from_float!`, and `alignof!`.
- Boundary: SPEC 10.53 records that overflow is reported rather than prevented,
  rotate counts trap on `@intCast`, conversions are unchecked, `memcpy_bytes!`
  does not detect overlap, and `offsetof!` is deliberately absent.

## Unreleased — Zig-exclusive intrinsics, packed structs, unchecked bodies

- Added `zig/intrinsics.py`, the single authority table for the Zig-only
  surface: operand widths (`uN`/`iN` plus `usize`/`isize` as 64-bit), the five
  value intrinsics, the three type intrinsics, argument validation, and a
  Python mirror used to cross-check the emitted code.
- Added `popcount!`, `clz!`, `ctz!`, `bswap!`, and `bitreverse!` lowering to
  `@popCount`, `@clz`, `@ctz`, `@byteSwap`, and `@bitReverse`. Counting
  intrinsics yield `i64`; the rearrangement ones keep the operand width so a
  `u64` result is not truncated. Adds SPEC 8.12.15.
- Added `#[packed]` on struct declarations, lowering to a Zig `packed struct`
  with a computed bit total. Fields must have a known bit width; floats,
  pointers, slices, strings, tuples and generic packed structs are rejected.
  Adds SPEC 8.12.16.
- Added `bitcast!(value, T)` with equal-bit-count checking, plus
  `bitsizeof!(T)` and `sizeof!(T)` lowering to `@bitSizeOf` and `@sizeOf`
  through an emitted `_zig_bitcast` helper. Adds SPEC 8.12.17.
- Added `#[unchecked]` on free functions, emitting
  `@setRuntimeSafety(false);` as the first statement of that body only; the
  bounds-check `std.debug.assert` guards are unaffected. Adds SPEC 8.12.18.
- `ast_nodes.StructDecl` gained an `attributes` list and `parser.parse_struct`
  now receives the declaration's attributes, so `#[packed]` survives parsing.
- New tests: `tests/test_zig_special.py` (19). New example:
  `examples/zig_intrinsics.poly`. Every rejection reuses `E001`/`E003`; no new
  diagnostic codes. The width model and the rounded `sizeof!` result are
  recorded in SPEC 10.52, and 8.12.9 now lists the expanded subset.

## Unreleased — expansion frontier (cluster, DST, macros, IPC, bootstrap)

- Added `distributed/frontier.py`: `cluster_discover_and_heartbeat`,
  `cluster_route_session_typed_rpc`, and `cluster_handle_partition_and_split_brain`.
  Unknown senders are discovered, stale/backwards heartbeats are separated (the
  latter is `E001`), silent members are evicted by TTL, RPCs are routed by a
  deterministic hash through the existing `encode_frame`/`decode_frame` codec
  (`E007` for an unsupported version, `E006` for an empty cluster), and a
  partition either elects a writable quorum at a bumped epoch or freezes. Adds
  SPEC 8.17 and 10.47.
- Added `tools/dst/`: `dst_virtualize_clock_and_scheduler`,
  `dst_inject_byzantine_and_io_faults`, `dst_bisect_and_minimize_schedule`. Tie
  order is a SHA-256 key over `(seed, at_ms, seq, node, kind)`, six faults are
  drawn in a fixed order from `random.Random(seed)`, and delta debugging shrinks
  a failing schedule against a caller-supplied oracle under a hard probe budget.
  Adds SPEC 8.18 and 10.48.
- Added `compiler/macro_engine/`: `macro_resolve_hygienic_scopes`,
  `macro_expand_typed_ast_splice`, `macro_verify_backend_expansion_parity`.
  Binders become `name#scope`, template free names resolve to `name@scope` and
  are reported when a call-site name would have captured them, holes are filled
  once with `i64`/`str`/`bool` checking, and the three backends must agree on the
  lowered opcode stream (`E004` on the first divergence). Adds SPEC 8.19 and
  10.49.
- Added `runtime/ipc/`: `ipc_map_shared_ring_arena`, `ipc_enforce_epoch_borrow_lease`,
  `ipc_reclaim_abandoned_slots`. A file-backed `mmap` ring with a 56-byte header
  shares cursors across mappings, `pop` returns a zero-copy `memoryview` while a
  lease is outstanding (so a producer gets `-1` instead of overwriting), leases
  release in FIFO order, and reclaim zeroes abandoned slots and fences stale
  leases by advancing the epoch. Adds SPEC 8.20 and 10.50.
- Added `bootstrap/`: `bootstrap_stage1_compile_subset`,
  `bootstrap_verify_stage2_fixed_point`, `bootstrap_diff_stage_ir_hashes`. The
  subset front end type-checks and lowers to one-line canonical IR; `parse_ir`
  accepts only canonical shapes (`E008` otherwise) and stage 2 must reproduce the
  stage-1 digest. Adds SPEC 8.21 and 10.51.
- Added `docs/cluster.md`, `docs/dst.md`, `docs/macros.md`, `docs/ipc_arena.md`,
  and `docs/bootstrap.md`, each stating its bounded model, plus an "Expansion
  frontier" table in the README.
- Cleaned duplicate `[tool.setuptools.packages.find]` include entries
  (`tools.abi_contract*` and `tools.provenance*` appeared twice) and added
  `bootstrap*`/`tools.dst*` so the new packages install.
- New tests: `tests/test_cluster_frontier.py` (12), `tests/test_dst.py` (19),
  `tests/test_macro_engine.py` (19), `tests/test_ipc_arena.py` (18),
  `tests/test_bootstrap.py` (14).
- Fixed two lint findings left in `benchmarks/run_benchmarks.py` (missing
  `check=False` on `subprocess.run`, non-executable shebang).

## Unreleased — comprehension pass, CLI commands, native cache fix

- Split `GUIDE.md` into a beginner path and `docs/advanced.md`; the guide now
  stops at Option/Result and the native backend, while Zig, SIMD, `#[parallel]`,
  telemetry, release gates, fuzzing and the formal frames moved to the advanced
  reference.
- Added `polyglot new <dir>` (starter `main.poly` and README, refuses a
  non-empty directory) and `polyglot explain [E00n]` (meaning, wrong/right
  examples and a tip for `E001`–`E008`). `polyglot exec` and
  `polyglot compile` now expand `include!`, matching `run` and `build`.
- Hardened `stdlib`: `reverse_text`, `is_palindrome` and `pad_left`/`pad_right`
  operate on Unicode scalars instead of byte offsets, `read_lines` was added to
  `std:fs`, and `Counter::top_key` returns the most frequent key or `None`.
- Added `examples/csv_sum.poly`, `word_count.poly`, `cli_report.poly` and
  `examples/README.md`; the README now states the positioning, the three
  backend tiers (python + native parity baseline, best-effort WASM, research
  JIT/Zig) and two known limitations (move-checking parity and the call-heavy
  Python backend).
- Added `benchmarks/` with three workloads, a CPython comparison, and a
  generated `RESULTS.md` that reports `n/a` below 0.5 ms instead of a
  meaningless ratio.
- Fixed the native backend reusing artifacts it did not build: cache entries
  seeded with another build's `target/` looked complete, which turned 14
  pattern/spec-conformance tests into "DID NOT RAISE". `_is_built` now requires
  a matching `.polyglot-manifest.json`, validated by
  `tests/test_native_cache_manifest.py`.
- Added shared ownership checking: `backends/move_check.py` runs before both
  backends and raises `E009` for a moved `String`/`Vec`/struct used again, with
  the binding name and both lines. Index moves (`words[0]` into an owned
  parameter) get their own message. Adds SPEC 8.16 and 10.46. The model is
  bounded: struct-field partial moves and unmodelled method semantics stay with
  the native backend.
- Moved the native out-of-range cast literal diagnostic from `E009` to `E008`,
  restoring the documented E001-E008 pillar surface now that `E009` means
  ownership.
- Added `tests/test_move_parity.py`: 8 allowed and 12 rejected programs run
  through both backends and must agree on the outcome.
- Recorded all defects with Symptom/Cause/Fix/Lesson entries in `NOTES.md`.
- Verified baseline: 3258 passed, 6 skipped, 31 deselected (`pytest -q`);
  31 passed, 3264 deselected (`pytest -q -m slow`).

## Unreleased — Zig 0.17.1 codegen and language/runtime models

- Upgraded the Zig backend for Zig 0.17.1: signed `/` and `%` now lower to
  `@divTrunc` and `@rem` (truncate toward zero, C semantics) because Zig
  rejects signed division with the ordinary operators; slice indexing
  `items[i]` lowers to bounds-checked `_zig_at`/`_zig_at_mut` helpers with
  `std.debug.assert`; reading `.len` on a many-item pointer (`[*]T`) emits a
  `@compileError` instead of silently reading garbage; `Result<T, PillarError>`
  lowers to the typed error union `PillarError!T` (not `anyerror!T`) and
  `Err(E00n)`/`Err(n)`/`Err(error.E00n)` all resolve to the matching member;
  and labeled-block `break <value>` lowers to `break :blk_N <value>;` with an
  untyped label, suppressing the unreachable fallthrough that Zig rejects.
  Adds SPEC 8.12.11–8.12.14 and 11.25.
- Added `compiler/gleam_style/` (SPEC 8.13): total ADTs with Hindley-Milner-style
  validation, exhaustive-match checking with concrete missing-pattern witnesses
  (`E002`), bounded recursive cycle detection, structured bit-array pattern
  matching (`<<header:8, len:16, payload:binary>>`) with endianness and checked
  bounds (`E001` on short buffers, no raw pointer arithmetic), and cross-target
  capability contracts (`E007` on unsupported capabilities) across Python,
  Rust, Zig, BEAM, and JS descriptors. Adds 10.43 and
  `docs/gleam_style.md`.
- Added `runtime/actor_otp/` (SPEC 8.15): share-nothing isolated actors with
  deep-copied messages at the send boundary, structured mailboxes with
  selective receive, OTP supervision strategies (`one_for_one`, `one_for_all`,
  `rest_for_one`) with `RestartIntensity` budgets and `E004` escalation, and
  let-it-crash fault classification into structured `E001`–`E008` reports with
  a bounded telemetry ring buffer. Platform boundaries (`MAP_JIT`, PEP 667,
  raw-address TOCTOU, model-only paths) raise `E007`. Adds 10.44 and
  `docs/actor_otp.md`.
- Added `compiler/mojo_bridge/` (SPEC 8.14): value/origin semantics
  (`Copyable`/`Movable`/`BorrowOnly` × `Owned`/`Borrowed`/`UnsafeAnyOrigin`)
  with use-after-move (`E002`), origin-escape (`E001`), and non-silent unsafe
  widening (`E007`) diagnostics; contiguous buffer layouts with contiguity and
  SIMD lane analysis; backend-neutral vectorized-kernel lowering; an honest
  MLIR availability probe that reports `available: False` with the missing tool
  names; and cold-path CPython interop with a PEP 667 hot-loop `f_locals` guard.
  Adds 10.45 and `docs/mojo_bridge.md`.
- Added 128 focused tests: 11 Zig codegen tests, 37 Gleam-style tests, 34
  actor/OTP tests, and 46 Mojo-bridge tests. Updated the verified baseline to
  3146 passed, 6 skipped, 24 deselected (`pytest -q`) and 24 passed, 3152
  deselected (`pytest -q -m slow`).

## Unreleased — research frames H–K

- Added Frame H (`compiler.formal`): bounded symbolic execution, bounded
  relational refinement with injected backend runners, and minimized
  infeasible-path witnesses. Exhausted step/unroll/path/time budgets and solver
  timeouts are structured `E008`/`E004` diagnostics; `Unknown` feasibility is
  never reported as a proof.
- Added Frame I (`compiler.termination`): declared structural/lexicographic
  variants with per-call decrease checks, conservative counted/monotone
  loop-bound certificates with explicit `UnboundedOrUnknown`, and deterministic
  resource-fuel contracts charged through the sandbox meter.
- Added Frame J (`runtime.hard_state`): declared points-to/alias
  over-approximation with TOCTOU windows, total live-state schema migrations
  with rollback plans, and quarantined deterministic fault replay. `MAP_JIT`,
  raw-FFI, and PEP 667 dynamic frames remain explicit boundaries.
- Added Frame K (`tools.adversarial`): deterministic grammar-bounded case
  generation, isolated mutation-score gating, and a spec-oracle report that
  records exact commands/outputs and bounded caveats.
- Added the `research-frames` release-gate stage (14 stages total) between
  `hard-frames-parity` and `diff-fuzz`, with an injectable in-process oracle
  runner so verification never spawns native, JIT, or WASM code. Added
  `tools.adversarial*` to the wheel package list.
- Added 42 focused tests across the four research frames and release-gate
  coverage for stage ordering, oracle failures, and malformed reports.

## Unreleased — hard frames and three-way parity

- Added compile-time WASM aggregate specialization (`backends.wasm.aggregates`):
  generic programs are monomorphized, then struct literals, vectors, enum
  variants and bounded generic methods are lowered to scalar WAT expressions.
  Scalar locals remain real WASM locals so mutation and loops stay correct.
- Added shared WASM preflight validation (`backends.wasm.validation`) that reuses
  the Python/native error types, so normalized `E001`–`E008` codes and messages
  match exactly. Dynamic aggregate ABIs remain explicit `E003` boundary skips.
- Extended `tools/three_way_parity.py` with a CLI and full 12/12 parity across
  Python, Rust-PyO3, and WASM for seeds `1`, `42`, `9999`, and `2026`; no skips
  and divergence artifacts only on real mismatches.
- Added the `hard-frames-parity` release-gate stage after `pytest-core`, wiring
  the JIT/WASM/hard-frame suites plus the four parity seeds with an injectable
  per-seed runner.
- Fixed the experimental Zig SIMD/parallel C-ABI defects: vector `extern struct`
  wrappers, a single `std` import, and eviction of stale zero-byte dylibs.
- Documented the WASM aggregate boundary, the three-way CI parity stage, and the
  verified test baseline in README, SPECIFICATION, and the guides.

## Unreleased — expansion batch

- Expanded the differential fuzzer corpus with Option/Result, string/float,
  collection, async-shaped, stdlib-include, and hotpatch-state templates; closed
  the E003 native skip by exercising a shared host-call arity diagnostic; and
  added bounded mismatch shrinking so artifacts contain minimal reproducers.
- Expanded session telemetry with automatic hot-patch event capture from the RCU,
  rollback-journal, and frame-migrator subsystems, baseline save/compare support
  in the CLI and visualizer, and cache-pruning/codegen-fragment counters.
- Expanded the release gate with cargo format/doc, dependency/metadata audit,
  Markdown link checking, benchmark regression, and release-metadata stages,
  plus optional parallel fuzz seeds with a shared warm dependency cache.
- Added GitHub Actions workflows for the unified release gate and maturin-based
  native wheel builds (with SBOM and release-only signing/trusted publishing),
  and documented the distribution boundaries.
- Added a read-only third-party audit report for `dfmacro`, `PyFrameRestart`,
  and the external terminal UI, with explicit scope decisions.

## Unreleased — Zig backend feature expansion

- Expanded the experimental Zig backend into a documented opt-in execution
  mode, selected with `--backend zig` on `run`, `build`, `exec`, and `compile`
  or with `get_backend("zig")`. It generates Zig source, builds a dynamic
  library with `zig build-lib -dynamic -O ReleaseFast`, caches it under
  `~/.cache/polyglot-zig`, and loads `__cabi_*` wrappers with `ctypes`.
- Added Zig-style saturating arithmetic `+|`, `-|`, and `*|`, including
  `saturating_op`, `int_bounds`, and `wrap_arbitrary_int` helpers and a
  zero-intrusion `lower_zig_ast` fallback that rewrites the operators to
  `_zig_sat_add`, `_zig_sat_sub`, and `_zig_sat_mul`.
- Added arbitrary bit-width integer types `uN`/`iN` for `0 <= N <= 65535`,
  represented at their native Zig width. The Python C ABI exposes 64-bit
  integers, so wider values truncate at that boundary.
- Added `#[parallel]` lowering for direct range `for` loops, splitting work
  across up to 16 threads capped by the available CPU count. There are no
  ordering, reduction, locking, or synchronization guarantees; nested loops
  run serially.
- Added `SimdN<T>` lowering to `@Vector(N, T)` with `Simd::splat`,
  `Simd::load`, `Simd::store`, and `Simd::reduce_add`/`reduce_mul`/
  `reduce_min`/`reduce_max`.
- Added `examples/zig_parallel_simd.poly`, combining `#[parallel]`, SIMD
  load/store/reduction, and a saturating `u8` return, plus extended coverage
  in `tests/test_zig_backend.py` for arbitrary widths, parallel scaling, SIMD,
  the example program, and CLI `run`/`build`.
- Documented the mode in README, GUIDE, SPEC 8.12/10.42/11.22, NOTES,
  `docs/overview.md`, `docs/architecture.md`, `CONTRIBUTING.md`, and the new
  `docs/zig.md`. The Zig backend is explicitly outside the Python/native
  parity contract; it lowers a narrow AST subset, produces a C-ABI library
  rather than an importable extension, and truncates Python-boundary integers
  to 64 bits.

## Unreleased — verification, telemetry, and release readiness

- Added `tools/diff_fuzz.py`, a deterministic seed-reproducible differential
  fuzzer that compares Python and native backends under isolated cache scopes,
  covers recursive generics/traits/match guards/control flow/E001–E008 cases,
  and writes minimal mismatch artifacts only when a mismatch occurs.
- Added `tools/release_gate.py` with `--quick`/`--full`, fixed-order checks,
  isolated cargo target directories, bounded fuzz seeds, and one aggregated
  `ci_generate_release_gate_report` result.
- Added `api/telemetry.py` with a thread-safe collector/serializer, path
  redaction, compile timing/cache/monomorph/hot-patch aggregation, CLI commands
  (`telemetry`, `timings`, `cache`, `hotpatch`), and a visualizer `/api/telemetry`
  endpoint plus collapsed panel with graceful empty states.
- Fixed wheel/sdist packaging: `framework_functions`, `diagnostics`, `docs`,
  `tools.build_ci`, and the new installable `regression.harness` package are
  shipped so the 100-function registry resolves in fresh environments;
  `tests/regression_harness.py` is now a compatibility re-export. Added Cargo
  package metadata and isolated cache/build tests.

## Unreleased — 100 framework functions

- Added `framework_functions`, a stable registry for 100 numbered compiler,
  runtime, diagnostics, CI, docs, and regression-harness functions across ten
  frames, with `get_function`, `invoke`, `iter_functions`, and
  `validate_registry`.
- Added frame modules under `compiler/frontend`, `compiler/types`, `compiler/ir`,
  `backends/python`, `backends/rust`, `runtime/hotpatch`, `runtime/concurrency`,
  `diagnostics`, `tools/build_ci`, `docs`, and `tests/regression_harness.py`,
  each with dedicated tests.
- Documented the canonical list and naming notes in `docs/framework_functions.md`
  and added SPEC 9.12 for the registry contract.

## Unreleased — Priority language and tooling features (v0.3.9)

- Lowered `vec![...]` in the parser to the shared `ArrayExpr`/`RepeatArray`
  AST, covering `vec![a, b]`, `vec![value; count]`, and nested `vec!` in both
  backends.
- Lowered `String::from(...)` in both backends and fixed a double-wrapped
  `String::from(String::from(...))` in generated Rust.
- Added lazy `Option::unwrap_or_else` and `Result::unwrap_or_else`; the
  fallback closure runs only on `None`/`Err`.
- Added the pure `.poly` `std:fs` module (`read_text`, `write_text`,
  `append_text`, `exists`, `remove`, `read_text_or`, `write_text_if_missing`,
  `read_text_nonempty`) over a trusted-only host seam with a shared span-aware
  `E003` arity check in both backends. See SPEC 8.11.
- Broadened `if let` to expression positions and the builtin `Some`, `None`,
  `Ok`, and `Err` patterns in both backends.
- Added `HashMap::entry(k).or_default()` and fixed `or_insert_with_key` slot
  typing. `Entry::insert` remains deferred on the native backend.
- Extended `#[terminates]` proofs to multiple self-recursive calls with
  per-call-site lower-bound guards; `E002` now points at the offending call,
  and guards do not leak into closures or nested functions.
- Expanded `stdlib/algo` (`chunk_sum`, `running_max`, `find_index`,
  `all_equal`, `rotate_left`, `zip_sum`) and `stdlib/str_utils`
  (`trim_matches`, `replace_first`, `split_once`, `is_empty_text`,
  `capitalize`, `contains_any`, `count_lines`).
- Added deterministic `FixHint` payloads to `polyglot exec --error-format=json`
  and the diagnostic helper, plus `textDocument/publishDiagnostics` on open,
  change, and save in the language server.
- Added cross-backend parity and edge tests in
  `tests/test_priority_features_parity.py` and `tests/test_stdlib_fs.py`, and
  extended termination, CLI, LSP, diagnostics, stdlib, and visualizer tests.
- Synced SPEC v0.3.9 (Sections 5, 7, 8, 9, 10, 11, 12), README, GUIDE, NOTES,
  and the recorded test baseline.

## Unreleased — Quantum hot-patching subsystem

- Added the opt-in `runtime/hotpatch` package with five concrete mechanisms:
  atomic GOT/PLT cell swaps plus owned-page AArch64/x86-64 jump sleds, CPython
  3.13+ safe-point frame migration through PEP 667, selector and native
  `mio::Poll` registration migration, cooperative RCU epochs for AST roots,
  and before-image rollback journaling.
- Added `runtime/hotpatch/native`, a PyO3 crate with real `unsafe` primitives
  for Mach-O GOT-section scans, Linux writable-map pointer scans, `MAP_JIT`
  executable regions, instruction-cache flushes, native RCU, raw-range
  journals, and a live `mio::Poll` wrapper.
- Added `E008` diagnostics with character and UTF-8 byte spans for relocation,
  ABI, frame-layout, reactor, and journal failures; a mio bridge without a
  replacement reactor reports `E006`, and foreign executable text or an
  unsupported primitive reports `E007`. The PyO3 surface renders the matching
  `error[E007]` / `error[E008]` text in `PyRuntimeError` rather than raising the
  Python `HotpatchError` class.
- Added `tests/test_quantum_hotpatch.py` covering live local and generator
  migration, concurrency-safe pointer swaps, zero-loss UDP reactor migration,
  RCU timeouts, rollback, span formatting, and an optional crate build.
- Documented the mechanisms and honesty boundaries in
  `docs/quantum_hotpatch.md` and linked the pillar from
  `docs/experimental_pillars.md`.

## v0.3.8 — Apple Silicon JIT backend and edge-case audit

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
  `E001`–`E008` diagnostics. Rejected mechanisms include foreign executable-text
  patching, mid-instruction `f_code` rewriting, privileged page/syscall hooks,
  real accelerator dispatch, and threshold signatures.
- Added `distributed/dsm.py`, `security/sandbox.py`,
  `compiler/heterogeneous.py`, and `swarm/agent.py` with dependency-free
  reference models for CRDT/consensus, sandbox policy/taint/fuel, MLIR-style
  accelerator lowering with CPU fallback, and gossip/self-healing.
- Added `runtime/native_bindings.py` and the generated opt-in `native_pillars/`
  crate: one `#[pyfunction]` per capability with honest `E006`/`E007`
  diagnostics. It is not built by `polyglot build` or CI.
- Documented the capability matrix in `docs/unified_pillars.md` and aligned the
  `#[gpu]`/`#[npu]` feature gate with the accelerator boundary.
- Added the initial Zig backend and lowering pass, including the saturating
  `+|`, `-|`, and `*|` operator family. The later Zig feature expansion above
  adds `#[parallel]`, SIMD, arbitrary bit-width integers, and the documented
  C-ABI boundaries.
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
