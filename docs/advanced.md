# Polyglot Advanced Features

This page collects the opt-in and maintainer-facing parts of Polyglot: the
experimental Zig backend (saturating arithmetic, arbitrary bit-width integers,
`#[parallel]` loops, SIMD vectors), the pure-Python Gleam/Elixir/Mojo models,
telemetry, and the verification gates.

None of it is required to write a normal Polyglot program. If you are new to
the language, read [GUIDE.md](../GUIDE.md) first.

## Zig backend (experimental)

The Zig backend is an opt-in execution mode for experimenting with Zig-style
integer semantics, threads, and vectors. It is **not** part of the
Python/native parity contract. Use it when you specifically want to try:

- saturating arithmetic that clamps instead of wrapping;
- arbitrary bit-width integers such as `u8`, `u24`, or `i19`;
- a `#[parallel]` range loop;
- `SimdN<T>` vector operations and horizontal reductions.

For normal programs, keep using the Python or native backend. The Zig lowering
covers only a small AST subset, explained below.

### Run a first Zig program

Create `sat.poly`:

```polyglot
fn main() -> i64 {
    (250 as u8 +| 20 as u8) as i64
}
```

Run it with:

```sh
polyglot run sat.poly --backend zig
```

Output:

```text
255
```

The `+|` operator adds with saturation. `250u8 + 20` would normally overflow;
instead it clamps at `255`. The sibling operators are `-|` and `*|`:

```polyglot
fn sat_sub(a: u8, b: u8) -> u8 { a -| b }
fn sat_mul(a: u8, b: u8) -> u8 { a *| b }
```

`sat_sub(10, 30)` returns `0`, and `sat_mul(20, 20)` returns `255`.

### Arbitrary bit-width integers

Any `uN` or `iN` with `N` between 0 and 65535 is accepted:

```polyglot
fn sat_u24(a: u24, b: u24) -> u24 {
    a +| b
}
```

From Python, call it through the backend:

```python
from backends import get_backend
from lexer import lex
from parser import parse

source = """
fn sat_u24(a: u24, b: u24) -> u24 { a +| b }
"""
namespace = get_backend("zig", trusted=True).compile(parse(lex(source)))
print(namespace["sat_u24"](16_777_210, 100))  # 16777215
```

Zig stores the value at its declared width. The Python `ctypes` boundary is
64-bit, so values wider than 64 bits are truncated when they cross into
Python. The lowering fallback used by non-Zig execution clamps at `i64`
bounds and is not bit-width aware.

### Parallel range loops

Add `#[parallel]` to a function with a direct range `for` loop:

```polyglot
#[parallel]
fn scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
    for i in 0..n {
        out[i] = (inp[i] * factor) + 1.0;
    }
}
```

The backend splits `0..n` into chunks and runs them on up to 16 worker threads
(capped by the available CPU count). Compile the function and call it from
Python with `ctypes` arrays:

```python
import ctypes

from backends import get_backend
from lexer import lex
from parser import parse

source = """
#[parallel]
fn scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
    for i in 0..n {
        out[i] = (inp[i] * factor) + 1.0;
    }
}
"""
namespace = get_backend("zig", trusted=True).compile(parse(lex(source)))

n = 10_000
inp = (ctypes.c_double * n)(*[float(i) for i in range(n)])
out = (ctypes.c_double * n)()

namespace["scale"](out, inp, 2.5, n)
assert out[10] == 26.0
```

Only the annotated function's direct range loop is parallelized. Nested loops
run serially. The backend does not add locks or reductions, so the loop body
must write disjoint elements or otherwise be race-free by construction.
Ordering between iterations is not guaranteed.

### SIMD vectors

`SimdN<T>` maps to Zig's `@Vector(N, T)`. The supported builtins are:

| Builtin | Meaning |
| ------- | ------- |
| `Simd::splat(x)` | Broadcast one scalar to every lane |
| `Simd::load(ptr, index, count)` | Load `count` lanes starting at `index` |
| `Simd::store(ptr, index, count, value)` | Store `count` lanes starting at `index` |
| `Simd::reduce_add(v)` | Horizontal sum |
| `Simd::reduce_mul(v)` | Horizontal product |
| `Simd::reduce_min(v)` | Horizontal minimum |
| `Simd::reduce_max(v)` | Horizontal maximum |

A four-lane dot product looks like this:

```polyglot
fn simd_dot4(a: &[f64], b: &[f64]) -> f64 {
    let va: Simd4<f64> = Simd::load(a, 0, 4);
    let vb: Simd4<f64> = Simd::load(b, 0, 4);
    Simd::reduce_add(va * vb)
}
```

The full runnable example is
[examples/zig_parallel_simd.poly](../examples/zig_parallel_simd.poly). It combines
`#[parallel]`, `Simd4<f64>`, `splat`, `load`, `store`, `reduce_add`, and a
saturating `u8` return in `main()`.

### Zig-exclusive intrinsics

Eight calls exist only on this backend, each lowering onto a Zig builtin:

| Call | Builtin | Result |
| --- | --- | --- |
| `popcount!(x)` | `@popCount` | `i64` set-bit count |
| `clz!(x)` / `ctz!(x)` | `@clz` / `@ctz` | `i64` zeros within the operand width |
| `bswap!(x)` / `bitreverse!(x)` | `@byteSwap` / `@bitReverse` | operand's own type |
| `bitcast!(x, T)` | `@bitCast` | `T`, equal bit count required |
| `bitsizeof!(T)` / `sizeof!(T)` | `@bitSizeOf` / `@sizeOf` | `i64` bits / bytes |

```polyglot
#[packed]
struct FrameHeader { version: u3, flags: u5, compressed: bool }

fn swapped(x: u32) -> u32 { bswap!(x) }
fn header_bits() -> i64 { bitsizeof!(FrameHeader) }  // 9
fn float_pattern(x: f32) -> u32 { bitcast!(x, u32) } // 1.0 -> 0x3F800000
```

`#[unchecked]` on a free function emits `@setRuntimeSafety(false)` for that
body, so its arithmetic wraps instead of trapping; the bounds-check
`std.debug.assert` guards stay. Widths come from the inferred polyglot type
(`usize`/`isize` are 64-bit, an unannotated identifier is `i64`), and a width
that cannot be inferred is `E001` rather than a guess. `sizeof!` reports
`@sizeOf`, which rounds up to the backing integer's alignment; use
`bitsizeof!` for an exact count.

The runnable example is
[examples/zig_intrinsics.poly](../examples/zig_intrinsics.poly):

```sh
polyglot run examples/zig_intrinsics.poly --backend zig   # prints 255
```

### Build and cache

```sh
polyglot build sat.poly --backend zig
```

The CLI writes the shared library next to the source (as `sat.so` on macOS).
The backend caches generated Zig and the library under
`~/.cache/polyglot-zig`, keyed by a hash of the generated source. Clear it with:

```sh
rm -rf ~/.cache/polyglot-zig
```

The build artifact has C-ABI `__cabi_*` symbols and is intended to be loaded
with `ctypes`. It is not an importable Python extension module, so
`import sat` will not work.

### What the Zig backend does not support

The current lowering covers free functions, scalar parameters and returns,
`let`, assignment, `while`, `loop` with `break`/`continue` (including labeled
blocks with `break <value>`), direct range `for`, `return`, integer casts,
indexing with bounds-checked slice access, binary operations, signed `/` and `%`
(lowered to `@divTrunc`/`@rem`), saturating operations, arbitrary-width
integers, `Result<T, PillarError>` lowered to `PillarError!T`, parallel range
loops, the SIMD builtins above, user-function calls, `if`/`else`, struct
declarations and field access, the intrinsics above, and `#[packed]` structs.
It does **not** lower struct construction, enums, `match`, strings, `Vec`, or
`Option`. Programs
outside that subset are outside the Zig contract; use the Python or native
backend for them. See [docs/zig.md](zig.md) for the complete feature
reference, SPEC 8.12 for the normative rules, and SPEC 10.42 / 10.52 for the
boundary entries.

### Language and runtime models (experimental)

Three pure-Python modules model the strongest ideas from other systems
languages. They run inside `polyglot-lang` itself, need no external toolchain,
and are outside the Python/native parity contract. Read them as analysis and
runtime helpers, not as new compilers.

| Model                   | Inspired by | When to reach for it                                                       | Guide |
| ----------------------- | ----------- | -------------------------------------------------------------------------- | ----- |
| `compiler/gleam_style/` | Gleam       | You want exhaustive `match` with concrete missing-pattern errors, or to match fixed binary protocol headers safely | [docs/gleam_style.md](gleam_style.md) |
| `runtime/actor_otp/`    | Elixir/OTP  | You want isolated actors, supervisor restart trees with budgets, and bounded crash telemetry | [docs/actor_otp.md](actor_otp.md) |
| `compiler/mojo_bridge/` | Mojo        | You want ownership/borrow lifetimes, contiguous buffer layout analysis, or an honest MLIR capability check | [docs/mojo_bridge.md](mojo_bridge.md) |

An exhaustive match with a concrete witness:

```python
from compiler.gleam_style import AdtDef, Variant, check_exhaustive

option = AdtDef("Option", (Variant("Some", ("Int",)), Variant("None")))
check_exhaustive(option, [("Some", ("x",))])
# error[E002]: non-exhaustive match on 'Option'; missing patterns: None
```

A supervised actor with a restart budget:

```python
from runtime.actor_otp import RestartIntensity, Strategy, Supervisor, spawn_actor

sup = Supervisor("root", Strategy.ONE_FOR_ONE, RestartIntensity(5, 60.0))
sup.add_child("worker", lambda: spawn_actor(0, lambda s, p: s + 1 if p else s))
assert sup.restart("worker", now=1.0).restarted == ("worker",)
```

An origin escape rejected:

```python
from compiler.mojo_bridge import Origin, OriginEnv, ValueKind, check_origin_escape

env = OriginEnv()
env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
check_origin_escape(env, "ref", return_as_owned=True)
# error[E001]: borrowed binding 'ref' cannot be returned as owned
```

Each module has a dedicated test file that runs without a Zig, Gleam, Elixir,
or MLIR toolchain:

```sh
pytest -q tests/test_gleam_style.py tests/test_actor_otp.py tests/test_mojo_bridge.py
```

## Telemetry and verification

Inspect what a compile run did without changing your program:

```sh
polyglot telemetry          # phases, total time, monomorphization counts
polyglot timings            # alias for the timing section
polyglot cache              # hit rates, hits/misses, cache prunes, rebuild reasons
polyglot hotpatch           # epoch, active capabilities, rollbacks, latest E008
polyglot telemetry --json   # machine-readable, path-redacted payload
polyglot telemetry --save-as before   # store a baseline outside the workspace
polyglot telemetry --compare before  # show timing/cache/monomorph deltas
```

The same data is available from the visualizer (`polyglot visualizer`) in the
collapsed **Telemetry** panel and from `GET /api/telemetry`, including a baseline
comparison selector. Hot-patch activity is captured automatically when the RCU
epoch manager publishes a new AST, the rollback journal commits or rolls back,
or the frame migrator patches a live function. If nothing has run yet, each
section reports an explicit empty state. Hot-patch entries report capability
boundaries; they do not claim privileged MAP_JIT, PEP 667, or raw-address
patching on platforms where those paths are modelled or refused.

Before publishing a change, run the unified gate:

```sh
python tools/release_gate.py --quick             # fast PR loop
python tools/release_gate.py --full --jobs 4     # extended fuzz seeds + full pytest
python tools/diff_fuzz.py --seed 42 --case-count 8 --json
python tools/three_way_parity.py --seed 2026 --json
```

The `hard-frames-parity` gate stage runs the JIT/WASM/hard-frame suites together
with the three-way parity seeds `1`, `42`, `9999`, and `2026`. The WASM backend
monomorphizes generics and specializes compile-time structs, vectors, enum
variants and bounded methods into scalar WAT before execution, while sharing the
Python/native `E001`–`E008` preflight diagnostics. Constructs that need a real
dynamic aggregate ABI (for example, an aggregate function parameter) are reported
as explicit boundary skips instead of false mismatch artifacts. Use
`--backend wasm` on a program that stays within the scalar/compile-time subset;
`--backend auto`, `--backend python`, and `--backend native` remain available.

The `research-frames` gate stage runs the bounded H–K suites plus an in-process
spec-oracle check. Use these APIs directly when investigating a boundary:

```python
from compiler.formal import formal_sym_exec_bounded_paths, formal_require_bounded_proof
from compiler.termination import term_infer_loop_bound_certificates
from runtime.hard_state import hard_plan_live_state_schema_migration
from tools.adversarial import adv_mutate_ast_and_type_constraints

# Bound exhaustion, timeouts, and Unknown feasibility are explicit.
report = formal_sym_exec_bounded_paths(source, max_unroll=4, strict=False)
if not report.complete:
    ...
```

Every research result is bounded: pass `strict=True` to raise the relevant
`E001`–`E008` diagnostic on exhaustion, or inspect the returned report for
`complete=False`, `UnboundedOrUnknown`, or `Unknown` feasibility. Structural
subterm fields, lexicographic components, points-to sets, and migration maps
are all caller-declared; the frames never infer a global termination, aliasing,
or migration proof. `MAP_JIT`, raw FFI, and PEP 667 dynamic frames are recorded
as explicit boundaries rather than silently modeled.

Beyond the original checks, the gate verifies cargo format/docs, package and
dependency metadata, Markdown links, benchmark regression, and release metadata.
`--jobs N` runs independent fuzz seeds concurrently while keeping the report in
seed order; each case still gets a deterministic isolated scope and the seeds
share a warm dependency cache.

Fuzz mismatches are shrunk and written to `.artifacts/diff_fuzz/` with the exact
seed and source needed to reproduce them. Clean runs leave no files behind. Use
`--allow-dirty` on the gate only when you deliberately want to verify an
uncommitted worktree. CI runs `--full` without that flag.
