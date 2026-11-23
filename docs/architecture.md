# Architecture overview

`polyglot-lang` compiles one `.poly` source file with two primary
interchangeable backends (Python and native PyO3), plus experimental
compute-only WASM and Zig backends. The front end is shared; only the
lowering and execution stages differ.

```text
                    .poly source
                         |
                         v
                 +---------------+
                 |    lexer.py   |  tokens
                 +---------------+
                         |
                         v
                 +---------------+
                 |   parser.py   |  AST (ast_nodes.py) + spans
                 +---------------+
                         |
                         v
              +---------------------+
              | include_expander.py |  splices include!("...") items
              +---------------------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
 +---------------------+      +----------------------+
 | backend: python     |      | backend: rust        |
 | translator.py       |      | codegen.py           |
 | (AST -> Python AST) |      | (AST -> Rust/PyO3)   |
 +---------------------+      +----------------------+
          |                             |
          v                             v
 +---------------------+      +----------------------+
 | compile()/exec()    |      | cargo build --release|
 | runtime.py helpers  |      | load .so / .pyd      |
 +---------------------+      +----------------------+
          |                             |
          +--------------+--------------+
                         |
                         v
              namespace dict / CLI result
```

The optional experimental WASM backend consumes the same AST and writes
WebAssembly text format (`.wat`) for an `i64`/`bool` compute-only subset.

The optional experimental Zig backend also consumes the same AST. `ZigCodegen`
emits Zig source for a narrow scalar subset, `zig build-lib -dynamic -O
ReleaseFast` builds a dynamic library, and `ctypes` loads the `__cabi_*`
wrappers. It is a separate execution mode, not a third parity backend.

## Layers

- **Front end:** `lexer.py`, `parser.py`, `ast_nodes.py`. Parser nodes carry
  best-effort character spans used by diagnostics.
- **Include expansion:** `include_expander.py` resolves `include!` paths
  relative to the including file, rejects cycles, and splices items before
  backend selection.
- **Python backend:** `backends/python/translator.py` converts the AST to a
  Python AST; `backends/python/runtime.py` holds stdlib-only helpers for
  casts, `Option`/`Result`, collections, and boundary checks.
- **Native backend:** `backends/rust/codegen.py` emits Rust with PyO3 and
  the async runtime (a single-threaded `mio` reactor, task wakers, timers,
  Unix fd readiness, and the `asyncio` bridge); `backends/rust/compiler.py`
  drives Cargo and loads the extension. Builds are cached under
  `~/.cache/polyglot-native` by a hash of generated Rust. Pass
  `--incremental` to keep per-function fragments in a stable scope
  directory.
- **WASM backend (experimental):** `backends/wasm/codegen.py` lowers the
  `i64`/`bool` subset to `.wat`; `wasmtime` is an optional dependency.
- **Zig backend (experimental):** `zig/backend.py` holds `ZigBackend` and
  `ZigCodegen`, `zig/features.py` implements saturating arithmetic and
  arbitrary bit-width bounds, and `zig/lowering.py` rewrites saturating
  operators for the Python fallback. The backend caches under
  `~/.cache/polyglot-zig` and loads a C-ABI dynamic library through
  `ctypes`. Signed `/` and `%` lower to `@divTrunc`/`@rem`, slice indexing is
  bounds-checked, and `Result<T, PillarError>` lowers to `PillarError!T`. See
  [zig.md](zig.md).
- **Language & runtime models (experimental):**
  `compiler/gleam_style/` provides total ADTs with exhaustive-match witnesses,
  bit-array pattern matching, and cross-target capability contracts; see
  [gleam_style.md](gleam_style.md). `runtime/actor_otp/` provides share-nothing
  actors, OTP supervision strategies with restart budgets, and let-it-crash
  fault reports with bounded telemetry; see [actor_otp.md](actor_otp.md).
  `compiler/mojo_bridge/` provides value/origin lifetimes, buffer layouts with
  SIMD lane analysis, an honest MLIR availability probe, and CPython interop
  with a PEP 667 hot-loop guard; see [mojo_bridge.md](mojo_bridge.md). These
  are pure-Python models: they do not generate code or invoke native toolchains.
- **CLI/API:** `api/cli.py` implements `run`, `build`, `check`, `compile`,
  `exec`, `visualizer`/`gui`, and `lsp`; `api/` also contains the optional
  HTTP worker/server and the dependency-free visualizer
  (`visualizer.py`, `visualizer_server.py`, and `static/visualizer.html`).
- **Experimental pillars (opt-in):** `backends/hotpatch.py`,
  `compiler/evolution.py`, `proofs/synthesis.py`, `debug/timetravel.py`,
  `runtime/transmute.py`, and `runtime/hotpatch/` (atomic GOT/PLT swaps, PEP 667
  safe-point frame migration, and reactor/RCU/journal primitives; see
  [quantum_hotpatch.md](quantum_hotpatch.md)) are research foundations that are
  not imported by the normal CLI path. The same layer also contains
  `distributed/` (actors, leases, migration, DSM), `security/sandbox.py`,
  `compiler/heterogeneous.py`, and `swarm/agent.py`; `runtime/capabilities.py`
  registers their 100-function surface. See
  [experimental_pillars.md](experimental_pillars.md) and
  [unified_pillars.md](unified_pillars.md).
- **Tooling:** `tools/` contains documentation/structure lints, a native
  compile helper, shell completions, and the native build benchmark.

## Design constraints

- One grammar and one AST for all backends.
- The Python backend is stdlib-only.
- Platform differences are explicit, documented in SPEC Section 8, and
  mirrored by Section 10 entries.
- Language semantics are not changed by a single backend; differences are
  recorded rather than silently accepted.
- The Zig execution mode is an opt-in experiment with its own narrow subset
  and C-ABI boundary. It does not extend the Python/native parity contract.
- The Gleam/OTP/Mojo models are analysis and runtime layers only. They never
  generate native code, invoke MLIR/GPU toolchains, or touch `MAP_JIT`,
  PEP 667 frame locals, or raw addresses.

See [overview.md](overview.md), [../README.md](../README.md),
[../GUIDE.md](../GUIDE.md), [../SPECIFICATION.md](../SPECIFICATION.md), and
[../NOTES.md](../NOTES.md).
