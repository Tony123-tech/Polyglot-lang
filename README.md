# polyglot-lang

**Write Rust-style code in a single `.poly` file and run it with Python — no
Rust toolchain, no `Cargo.toml`, no build step.**

```sh
polyglot new myapp && cd myapp && polyglot run main.poly
```

**Who this is for:** Python developers who want explicit types, `Option`/`Result`
instead of `None` and exceptions, and predictable integer widths — without
learning Rust or maintaining a PyO3 project.

**Who this is not for (yet):** teams shipping a large production service today.
The language is young, the standard library is small, and the native story is
still settling. Use CPython, Numba or Polars when you need production maturity
this month.

What you actually get:

- Types are checked at compile time; bindings are immutable unless you write `let mut`.
- `Option` and `Result` make missing values and failures part of the signature.
- One file runs in-process through the Python backend, or compiles to a PyO3
  extension when a hot loop needs C-level speed.
- Stable error codes (`E001`–`E009`) with fix hints, `polyglot explain E001`,
  and an LSP that publishes the same diagnostics.

New here? Start with `polyglot new`, then read [GUIDE.md](GUIDE.md) and
[examples/README.md](examples/README.md).

## Backends by tier

The tier says how much you can rely on a backend. Docs, examples and the
error-code guarantees only cover tier 1.

| Tier | Backend | Default flag | Artifact | Requirements | Promise |
|------|---------|--------------|----------|--------------|---------|
| 1 | Python | `python` | Runs in-process (Python AST) | None | Parity baseline |
| 1 | Native | `native` (`build` default) | PyO3 extension (`.so`/`.pyd`) | `cargo` + `rustc` | Parity baseline |
| 2 | WASM | `wasm` | `.wat` via wasmtime | `wasmtime` extra | Best effort, compute-only |
| 3 | JIT | `jit` | AArch64 machine code | macOS, Apple Silicon | Research |
| 3 | Zig | `zig` | C-ABI dynamic library via `ctypes` | `zig` on `PATH` | Research |

Tier 1 backends share one front end and are parity-tested. Tier 2 and 3 are opt-in
experiments: they are not covered by the parity contract, they can lag behind
language changes, and problems in them are handled as research rather than
product bugs. No `Cargo.toml` or `pyproject.toml` scaffolding is needed for a
single-file program.

## Quick Start

Install:

```sh
pip install polyglot-lang
```

Python 3.14.8 or later is required. Create a project and run it:

```sh
polyglot new myapp
cd myapp
polyglot run main.poly
```

Prefer a single file with no directory? Create `hello.poly` by hand:

```polyglot
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 { add(2, 3) }
```

Run it with the Python backend (no build step):

```sh
polyglot run hello.poly
```

Output:

```text
5
```

Build a native extension (requires a Rust toolchain):

```sh
polyglot build hello.poly
python -c "import hello; print(hello.add(2, 3))"
```

The first native build takes about two minutes; warm cached builds take seconds.
The Python backend has no build step. If you lack a Rust toolchain, skip the
native build — `polyglot run` works without it.

New to Rust-like syntax? See [GUIDE.md](GUIDE.md) for a step-by-step tutorial.

## CLI reference

```sh
polyglot new <dir>                    create a starter project
polyglot explain [E001]               explain an error code (omit to list all)
polyglot run <file>        [--backend python|native|wasm|jit|zig] [--target-width N] [--error-format text|json] [--verbose]
polyglot build <file>      [--backend python|native|wasm|zig] [--out-dir <dir>] [--incremental] [--error-format text|json] [--verbose]
polyglot check <file>                  [--all-errors] [--error-format text|json] [--verbose]
polyglot lsp
polyglot compile --backend python|native|wasm|jit|zig (--file <path> | --stdin) [--trusted] [--error-format text|json] [--verbose]
polyglot exec (<file> | --stdin)       [--backend python|native|wasm|jit|zig] [--target-width N] [--result-file <path>] [--error-format text|json] [--verbose]
polyglot visualizer [--host 127.0.0.1] [--port 8765] [--no-open] [--verbose]   (alias: gui)
polyglot telemetry [--section all|telemetry|cache|hotpatch] [--json] [--color] [--save-as NAME] [--compare NAME]
polyglot timings [--json] [--color] [--save-as NAME] [--compare NAME]
polyglot cache   [--json] [--color] [--save-as NAME] [--compare NAME]
polyglot hotpatch[--json] [--color] [--save-as NAME] [--compare NAME]
```

| Command | What it does |
|---------|--------------|
| `new` | Creates a directory with a runnable `main.poly` and a short README; refuses to overwrite a non-empty directory. |
| `explain` | Prints what an error code means plus a wrong/right example; omit the code to list every code. |
| `run` | Compiles and executes a file; prints `main()`'s return value when present. `--target-width 32\|64` simulates that `usize`/`isize` width on the Python backend only. |
| `build` | Compiles only. Native emits an extension module, WASM writes a `.wat`, Zig copies a C-ABI dynamic library. All print the artifact path. `--incremental` reuses per-function Rust fragments across native rebuilds. |
| `check` | Parses, translates (Python path only — no `--backend`), and runs `#[terminates]` checks without executing. `--all-errors` reports every top-level parse error. |
| `lsp` | Starts the stdio language server (`textDocument/publishDiagnostics` on open/change/save). |
| `compile` | Compatibility command used by the HTTP executor; prints exported top-level names. |
| `exec` | Compiles and executes like `run`, prints program output only, and — with `--result-file` — writes JSON with compile/run timings, peak RSS, return value, and structured `E001`–`E009` diagnostics. |
| `visualizer` (`gui`) | Local dependency-free web UI for the AST, monomorphization diff, error spans, and side-by-side Python/native output. Binds to `127.0.0.1` by default; runs submitted code as trusted local code. |
| `telemetry`/`timings`/`cache`/`hotpatch` | Print compile-phase timings, cache hit rates, monomorphization counts, and hot-patch status. `--save-as NAME` and `--compare NAME` persist and diff baselines outside the workspace. |

Errors use the toolchain five-line format (six lines when a span is available);
pass `--verbose` for a full traceback. `--error-format text|json` is supported
on `run`, `build`, `check`, `compile`, and `exec`; JSON emits one object with
`code`, `message`, `line`, `col`, `span`, and `source`.

Backend requirements: Python — none; native — Rust toolchain (`cargo`, `rustc`)
and PyO3-compatible build tooling; WASM — optional `wasm` extra (`wasmtime`);
JIT — Apple Silicon (`MAP_JIT`); Zig — `zig` on `PATH` (`zig version`).

## Support & parity

The Python and native backends share a front end and are parity-tested for the
documented subset:

| Feature | Python | Native |
|---------|--------|--------|
| Vec, String, Option, Result | ✅ | ✅ |
| HashMap / BTreeMap / HashSet | ✅ | ✅ |
| Nested HashMap (depth 1) | ✅ | ✅ |
| Integer types (i8…u128, isize, usize) | ✅ | ✅ |
| `include!` directive; bundled `std:` library | ✅ | ✅ |
| `vec![...]`, `String::from(...)`, lazy `unwrap_or_else` | ✅ | ✅ |
| `if let` (`Some`/`None`/`Ok`/`Err`), statement & expression position | ✅ | ✅ |
| `HashMap::entry(...)` (`or_default`, `or_insert`, …) | ✅ | ✅ |
| Error codes `E001`–`E009` + structured fix hints | ✅ | ✅ |
| LSP `publishDiagnostics` on open/change/save | ✅ | ✅ |
| `#[terminates]` checking, incl. multiple self-recursive calls | ✅ | ✅ |
| `std:fs` trusted host seam | ✅ (trusted) | ✅ (trusted) |
| Async free functions | ✅ (synchronous) | ✅ |
| Async `&self` / `&mut self` impl methods | ✅ | ✅ |
| Cross-thread Send awaits, drop-based cancellation, reactor timers | ⬜ | ✅ |
| Box patterns, payload enum slices | ✅ | ⬜ Deferred |

`BTreeMap` iteration is sorted by key in both backends; `HashSet` iteration order
is unspecified — sort before comparing. `get_mut` on `BTreeMap`/`HashSet`
returns an owned copy at the native Python boundary (SPEC 8.10.2, 10.20).

### Experimental Zig backend

Opt-in (`--backend zig` on `run`, `build`, `exec`, `compile`, or
`get_backend("zig")`); not part of the Python/native parity contract. The tested
subset covers top-level functions, `let`/assignment, `while`/`loop`/range `for`,
`return`, binary and cast expressions, signed `/`/`%` (`@divTrunc`/`@rem`),
bounds-checked slice indexing, labeled-block `break <value>`,
`Result<T, PillarError>` → `PillarError!T`, saturating arithmetic, arbitrary
bit-width `uN`/`iN` (`0..=65535`), `#[parallel]` range-loop splitting, and
`SimdN<T>` builtins, user-function calls, `if`/`else`, struct declarations and
field access. Struct construction, enums, `match`, strings, `Vec`, and `Option`
are not lowered. The artifact is a C-ABI shared library (`__cabi_*` symbols)
loaded with `ctypes`, not an importable Python extension. See
[docs/zig.md](docs/zig.md) and SPEC 8.12.

Zig also carries a **Zig-exclusive surface** the other backends cannot model:
`popcount!`, `clz!`, `ctz!`, `bswap!`, `bitreverse!`, `rotl!`/`rotr!`,
`bitcast!(value, T)`, `bitsizeof!`/`sizeof!`/`alignof!`, overflow reports
(`add_with_overflow!`, `mul_with_overflow!`, `wrapping_add!`, `wrapping_mul!`),
float/integer conversions (`int_from_float!`, `float_from_int!`), byte buffers
(`memcpy_bytes!`, `memset_bytes!`), `Simd::reduce_and`/`or`/`xor`, a `#[packed]`
struct layout, and an `#[unchecked]` function that drops Zig's runtime safety
checks for that body (SPEC 8.12.15–8.12.23, boundaries in 10.52–10.53).

```polyglot
fn main() -> i64 { (250 as u8 +| 20 as u8) as i64 }
```

```sh
polyglot run sat.poly --backend zig   # prints 255
```

Zig caches generated Zig and its dynamic library under `~/.cache/polyglot-zig`,
keyed by a hash of the generated source. `polyglot build --backend zig` copies
the cached library next to the source (`.so` on macOS).

### Experimental language & runtime models

Pure-Python modules that need no native toolchain: Gleam-style total ADTs with
exhaustive-match witnesses, Elixir/OTP-style supervised actors with restart
budgets, and Mojo-style origin tracking with lane analysis. They are analysis and
runtime helpers, not new compilers, and they sit outside the Python/native parity
contract.

These are research surfaces, not product features. Full reference, runnable
examples and their tests live in [docs/advanced.md](docs/advanced.md).

### Expansion frontier

Five bounded engineering models, each with a documented limit rather than a
claim:

| Surface | What it proves | Docs |
|---------|----------------|------|
| Cluster membership, session routing, split brain | Discovery with TTL eviction, session-typed RPC through the versioned wire codec, quorum/fence decisions | [docs/cluster.md](docs/cluster.md) |
| Deterministic simulation testing | Virtual clock, seeded tie-breaking, six seeded faults, chunk-based shrinking from one seed | [docs/dst.md](docs/dst.md) |
| Hygienic macro engine | `name#scope` binders, definition-site free names, typed holes, cross-backend opcode parity | [docs/macros.md](docs/macros.md) |
| Shared-memory ring arena | `mmap` ring with zero-copy `memoryview` reads, FIFO leases, epoch-fenced borrow leases, blunt reclaim | [docs/ipc_arena.md](docs/ipc_arena.md) |
| Bootstrapped compiler core | Subset front end to canonical IR, IR re-read to a fixed-point digest, stage diff | [docs/bootstrap.md](docs/bootstrap.md) |

`tests/test_cluster_frontier.py`, `tests/test_dst.py`, `tests/test_macro_engine.py`,
`tests/test_ipc_arena.py`, and `tests/test_bootstrap.py` cover them; all are
deterministic and run in the fast suite.

## Error codes (E001–E009)

Every diagnostic carries a stable code:

| Code | Raised by | Meaning |
|------|-----------|---------|
| `E001` | `TypeErrorForBoundary` | Implicit conversion between different integer types; add an explicit `as` cast. SPEC 2.4.6–2.4.7. |
| `E002` | `TerminationError` | `#[terminates]` function has no supported termination proof. SPEC 2.8. |
| `E003` | `TranslateError` | AST-to-backend translation failure (unsupported construct, invalid pattern). Python backend only; native reports `E004` or `E007`. |
| `E004` | `RustCompileError` | Generated Rust failed to compile; original source location included when available. |
| `E005` | `IncludeCycleError` | Cyclic `include!` chain; the full chain is printed. SPEC 9.2. |
| `E006` | `IncludeNotFoundError` | `include!` target missing or unreadable; `std:` paths escaping `stdlib/` also report this. SPEC 9.2. |
| `E007` | `UnsupportedNativePatternError` | Source pattern cannot be represented by stable Rust (also reserved `#[gpu]`/`#[npu]`). SPEC 8.3. |
| `E008` | `InvalidIntegerParseLiteralError`, `InvalidCastError` | A literal integer could not be parsed, or a literal cannot be represented in the type it is cast to. |
| `E009` | `MoveError` | An owned value (`String`, `Vec`, struct, enum) is used after it was moved; the message names the binding and both lines. SPEC 8.16. |

Fix hints are local and deterministic (currently an edit only for implicit `as`
casts under `E001`). The language server publishes at most one diagnostic per
validation event and does not apply UTF-16 code-unit encoding to positions
(SPEC 10.39, 10.40).

## Platform & runtime boundaries

### Build & test cache isolation

Native builds set an isolated `CARGO_TARGET_DIR` per build/check scope (Cargo's
`target` directory never escapes the build directory; `sccache` is used via
`RUSTC_WRAPPER` when available). The native backend caches under
`~/.cache/polyglot-native`; the Zig backend caches under
`~/.cache/polyglot-zig`. Test and fuzz tooling (`tools/diff_fuzz.py`,
`tools/three_way_parity.py`, `runtime/causal_replay/isolation.py`) runs each
case under a scoped temporary directory so parallel runs do not race.

Clear both caches:

```sh
rm -rf ~/.cache/polyglot-native ~/.cache/polyglot-zig
```

### Privileged / model-only mechanisms

The speculative JIT (`runtime.jit`) composes with the Apple Silicon `MAP_JIT`
backend to track hot traces, emit type/null/bounds guards, deoptimize without
losing live state, manage W^X code blocks, and report `E008` with spans.
`MAP_JIT`, PEP 667 frame locals, raw addresses, and TOCTOU windows are **model
only unless explicitly enabled on a supported host**. The experimental pillar
modules (`compiler/gleam_style/`, `runtime/actor_otp/`, `compiler/mojo_bridge/`)
and the research frames H–K (`compiler.formal`, `compiler.termination`,
`runtime.hard_state`, `tools.adversarial`) do not generate native code, do not
invoke MLIR/GPU toolchains, and do not touch `MAP_JIT`, PEP 667 frame locals, or
raw addresses; requests needing those mechanisms are rejected with `E007`.
Distributed shared memory, OS-enforced sandboxing, GPU/NPU offloading, and agent
swarms are modelled, not privileged hardware mechanisms. `#[gpu]` and `#[npu]`
are reserved but unsupported — both backends reject them with `E007` and an
attribute span before lowering.

### Sandbox

`PythonBackend(sandbox=True)` restricts `__builtins__` to a safe subset
(independent of the `trusted` flag), blocking `__import__`, `eval`, `open`,
`compile`, and `globals`. `trusted=False` is a speed bump, not a sandbox — there
is no process isolation. Treat untrusted input as needing OS-level isolation
(subprocess or Docker; see NOTES.md isolation levels B/C). The visualizer binds
to `127.0.0.1` and executes submitted code as trusted local code; it is a
developer tool, not a sandbox.

## Verification & reporting

```sh
pytest -q
pytest -q -m slow
pytest -q tests/test_zig_backend.py     # requires zig on PATH
pytest -q tests/test_zig_special.py     # Zig-exclusive intrinsics
pytest -q tests/test_gleam_style.py
pytest -q tests/test_actor_otp.py
pytest -q tests/test_mojo_bridge.py
```

- `pytest -q` deselects slow tests by default. `make check` runs ruff, the
  docs-sync check, and the fast suite.
- `python tools/diff_fuzz.py --seed 42 --case-count 8 --json` runs a
  deterministic differential fuzzer across the Python and native backends under
  isolated cache scopes, shrinking mismatches and writing artifacts only on a
  real mismatch.
- `python tools/three_way_parity.py --seed 2026 --json` runs a 12-case matrix
  on Python, Rust-PyO3, and WASM with per-case isolated caches. Seeds `1`, `42`,
  `9999`, `2026` are release-gate checks.
- `python tools/release_gate.py --quick` (or `--full`, `--jobs N`) runs the
  unified 14-stage release gate (lint, drift, pytest, hard-frames-parity,
  research-frames, fuzz, cargo checks, docs, distribution, benchmarks) and emits
  one pass/fail report; the research runner is injectable so verification never
  spawns native or JIT code.

**Current test status:** 3374 passed, 6 skipped, 31 deselected (`pytest -q`);
31 passed, 3380 deselected (`pytest -q -m slow`).

The Zig backend was upgraded for Zig 0.17.1 (signed `/`/`%` to `@divTrunc`/`@rem`,
bounds-checked slice indexing, labeled-block `break <value>`, and
`Result<T, PillarError>` → `PillarError!T`), and three pure-Python language/
runtime models were added: `compiler/gleam_style/`, `runtime/actor_otp/`, and
`compiler/mojo_bridge/`.

## Architecture

```text
source → lexer → parser → AST →
  Python AST → compile/exec  |
  Rust + PyO3 → Cargo build → extension  |
  Zig → zig build-lib → ctypes           |
  WAT → wasmtime                          |
  → Python namespace
```

Select a backend with `get_backend("python")`, `get_backend("native")`,
`get_backend("wasm")`, `get_backend("jit")`, or `get_backend("zig")`; each
`backend.compile(program)` returns a namespace dictionary.

## Known limitations

- `#[gpu]` and `#[npu]` annotations are rejected with `E007` before lowering.
- `while let Some(x) = iter.next()` and language-level async timer/I/O
  primitives are out of scope. `polyglot_sleep_ms`, `polyglot_readable`, and
  `await_with_timeout` are runtime/test helpers only (SPEC 8.2, 10.34).
- Box patterns, enum payload slices, and nested Vec slice patterns are
  Python-only; `is_ok`/`is_err`/`map_err`/`ok_or`/`and_then`/`or_else` are
  Python-only inside let-bindings.
- The Python backend executes async functions synchronously; native futures
  work with `asyncio` but expose no language-level timer/I/O primitives.
- WASM is compute-only (`i64`/`bool`, no strings, aggregates, or host imports).
- Zig is experimental and opt-in; its narrow AST subset and C-ABI boundary are
  not covered by the Python/native parity contract. `#[parallel]` splits only a
  direct range `for` into up to 16 worker threads with no ordering/reduction
  guarantees; nested loops run serially.
- SIMD is limited to `SimdN<T>` and `splat`/`load`/`store`/`reduce_*` builtins
  (no masking, gather/scatter, or general vector type inference).
- The lock-order proof accepts an explicit finite set of acquisition edges;
  cycles are rejected with an `E002` witness path (does not prove
  deadlock-freedom for arbitrary code).
- Distributed migration is cooperative replay, not opaque stack copying
  (see docs/distributed.md).
- Ownership is checked by one shared pass (`backends/move_check.py`, SPEC 8.16),
  so a moved `String`/`Vec`/struct is rejected with `E009` on both backends
  before either one generates code. The model is bounded: struct-field partial
  moves, moves through references and unmodelled method semantics are still left
  to the native backend, so a program can occasionally pass `--backend python`
  and fail rustc. `polyglot explain E009` shows the fix.
- **The Python backend is slow on call-heavy code.** Transpiled functions wrap
  their arguments, so recursive `fib` runs about 15x slower than CPython, while a
  tight typed `for` loop can beat CPython. Numbers and method:
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md), reproduced with
  `python benchmarks/run_benchmarks.py`.

## Notes & license

See [NOTES.md](NOTES.md) for design decisions, per-batch history, and bug
lessons (including the Security section for isolation levels and
`X-Trusted-Token`). See [SPECIFICATION.md](SPECIFICATION.md) for the full
language specification.

MIT License — see [LICENSE](LICENSE).
