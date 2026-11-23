# polyglot-lang

**polyglot-lang** (v0.3.8) is a Rust-like language with one shared front end
(lexer, parser, AST) and several backends: a Python transpiler with no native
dependencies, a native Rust/PyO3 extension backend, an experimental WebAssembly
backend, an experimental Apple-Silicon JIT backend, and an experimental Zig
backend. It also ships pure-Python language and runtime models (Gleam-style
ADTs, OTP actors, Mojo-style origin tracking) for static analysis and runtime
verification.

Write one `.poly` file and choose how to run it:

```polyglot
fn add(a: i64, b: i64) -> i64 { a + b }
```

## Backends at a glance

| Backend | Default flag | Artifact | Requirements | Contract |
|---------|--------------|----------|--------------|----------|
| Python | `python` | Runs in-process (Python AST) | None | Parity baseline |
| Native | `native` (`build` default) | PyO3 extension (`.so`/`.pyd`) | `cargo` + `rustc` | Parity baseline |
| WASM | `wasm` | `.wat` via wasmtime | `wasmtime` extra | Experimental, compute-only |
| JIT | `jit` | AArch64 machine code | macOS, Apple Silicon | Experimental |
| Zig | `zig` | C-ABI dynamic library via `ctypes` | `zig` on `PATH` | Experimental |

Python and native are parity-tested for the documented language subset. WASM,
JIT, and Zig are opt-in experiments and are **not** part of that parity
contract. No `Cargo.toml` or `pyproject.toml` scaffolding is needed for a
single-file program.

## Quick Start

Install:

```sh
pip install polyglot-lang
```

Python 3.14.8 or later is required. Create `hello.poly`:

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
| `run` | Compiles and executes a file; prints `main()`'s return value when present. `--target-width 32\|64` simulates that `usize`/`isize` width on the Python backend only. |
| `build` | Compiles only. Native emits an extension module, WASM writes a `.wat`, Zig copies a C-ABI dynamic library. All print the artifact path. `--incremental` reuses per-function Rust fragments across native rebuilds. |
| `check` | Parses, translates (Python path only — no `--backend`), and runs `#[terminates]` checks without executing. `--all-errors` reports every top-level parse error. |
| `lsp` | Starts the stdio language server (`textDocument/publishDiagnostics` on open/change/save). |
| `compile` | Compatibility command used by the HTTP executor; prints exported top-level names. |
| `exec` | Compiles and executes like `run`, prints program output only, and — with `--result-file` — writes JSON with compile/run timings, peak RSS, return value, and structured `E001`–`E008` diagnostics. |
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
| Error codes `E001`–`E008` + structured fix hints | ✅ | ✅ |
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
`SimdN<T>` builtins. Structs, enums, `if`, `match`, strings, `Vec`, `Option`,
and ordinary user-function calls are not lowered. The artifact is a C-ABI shared
library (`__cabi_*` symbols) loaded with `ctypes`, not an importable Python
extension. See [docs/zig.md](docs/zig.md) and SPEC 8.12.

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

Pure-Python modules, no native toolchain, outside the Python/native parity
contract:

| Module | Inspired by | Provides | Guide |
|--------|-------------|----------|-------|
| `compiler/gleam_style/` | Gleam | Total ADTs, exhaustive-match witnesses (`E002`), bit-array patterns with checked bounds, cross-target capability contracts | [docs/gleam_style.md](docs/gleam_style.md) |
| `runtime/actor_otp/` | Elixir/OTP | Share-nothing actors, deep-copied messages, `one_for_one`/`one_for_all`/`rest_for_one` supervision with restart budgets, let-it-crash `E001`–`E008` reports, bounded telemetry | [docs/actor_otp.md](docs/actor_otp.md) |
| `compiler/mojo_bridge/` | Mojo | `Copyable`/`Movable`/`BorrowOnly` value kinds, `Owned`/`Borrowed`/`UnsafeAnyOrigin` lifetimes, SIMD lane analysis, MLIR availability probe, PEP 667-safe CPython interop | [docs/mojo_bridge.md](docs/mojo_bridge.md) |

```python
from compiler.gleam_style import AdtDef, Variant, check_exhaustive

option = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
check_exhaustive(option, [("Some", ("x",))])
# error[E002]: non-exhaustive match on 'Option'; missing patterns: None
```

## Error codes (E001–E008)

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
| `E008` | `InvalidIntegerParseLiteralError` | A literal integer could not be parsed for the target type. |

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

**Current test status:** 3224 passed, 5 skipped, 24 deselected (`pytest -q`);
24 passed, 3152 deselected (`pytest -q -m slow`).

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

## Notes & license

See [NOTES.md](NOTES.md) for design decisions, per-batch history, and bug
lessons (including the Security section for isolation levels and
`X-Trusted-Token`). See [SPECIFICATION.md](SPECIFICATION.md) for the full
language specification.

MIT License — see [LICENSE](LICENSE).
