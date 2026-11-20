# Zig backend (experimental)

The Zig backend is an opt-in execution mode for experimenting with Zig-style
integer semantics, threads, and vectors. It is **not** one of the two
mandatory backends and is **not** part of the Python/native parity contract.
For normal programs, use the Python or native backend.

- Normative rules: [SPECIFICATION.md](../SPECIFICATION.md) Section 8.12
- Boundary entry: SPEC 10.42
- Tutorial: [GUIDE.md](../GUIDE.md#9-when-to-use-the-zig-backend-experimental)
- Tests: `tests/test_zig_backend.py`
- Example: [examples/zig_parallel_simd.poly](../examples/zig_parallel_simd.poly)

## Requirements and selection

The backend requires a `zig` executable on `PATH`:

```sh
zig version
```

Select it through the CLI:

```sh
polyglot run file.poly --backend zig
polyglot build file.poly --backend zig
polyglot exec file.poly --backend zig
polyglot compile --backend zig --file file.poly
```

or through the backend registry:

```python
from backends import get_backend
from lexer import lex
from parser import parse

namespace = get_backend("zig", trusted=True).compile(parse(lex(source)))
```

`polyglot check` does not take `--backend`; it always uses the Python
translation path. The `zig` backend is not available there.

The Zig backend implements the same trust gate as the other backends:
`ZigBackend(trusted=False)` raises `UntrustedCompileError` on `compile()` and
`build()`. The CLI passes `trusted=True`, matching `run` and `build` for the
other native backends.

## How compilation works

1. The shared lexer and parser produce the normal AST.
2. `zig/backend.py` emits Zig source with `ZigCodegen`.
3. `zig build-lib -dynamic -O ReleaseFast -femit-bin=<lib>` builds a dynamic
   library.
4. `ctypes.CDLL` loads the library. Each top-level function is exposed through
   an `export fn __cabi_<name>` wrapper.

Generated source and the library are cached under:

```text
~/.cache/polyglot-zig/<sha256-prefix>/
```

The cache key is a SHA-256 prefix of the generated Zig source. A cache hit
skips the Zig build. The cache directory is created with mode `0700`.

`polyglot build` copies the cached library next to the source. On macOS the
copied artifact uses the `.so` suffix. It is a **C-ABI library**, not a PyO3
extension: load it with `ctypes`, not `import`.

```python
import ctypes

lib = ctypes.CDLL("./sat.so")
lib.__cabi_main.restype = ctypes.c_int64
print(lib.__cabi_main())
```

## Supported subset

The current lowering covers:

| Construct | Status | Notes |
| --------- | ------ | ----- |
| Free functions | ✅ | Top-level `fn` only |
| Scalar parameters and returns | ✅ | Integer, `f32`/`f64`, `bool`, pointers |
| `let` and assignment | ✅ | `mut` and immutable bindings |
| `while` | ✅ | Standard condition loop |
| Direct range `for` | ✅ | `start..end` and `start..=end` |
| `return` and block tails | ✅ | Function result |
| Integer casts | ✅ | `as` to integer types |
| Indexing | ✅ | Pointer/slice indexing |
| Binary operations | ✅ | Arithmetic, comparison, logical |
| Saturating operators | ✅ | `+`, `-`, `*` with a trailing pipe |
| Arbitrary-width integers | ✅ | `uN`/`iN`, `0 <= N <= 65535` |
| `#[parallel]` range loops | ✅ | Direct loop in the annotated function |
| `SimdN<T>` and SIMD builtins | ✅ | See below |
| Structs, enums, `if`, `match` | ⬜ | Not lowered |
| Strings, `Vec`, `Option`, `Result` | ⬜ | Not lowered |
| Ordinary user-function calls | ⬜ | Only the SIMD builtins are special-cased |
| Float casts | ⬜ | `as` lowering uses integer casts |

Programs outside this subset are outside the Zig contract. Use the Python or
native backend for them.

## Saturating arithmetic

`+|`, `-|`, and `*|` clamp at the bounds of the declared integer type:

```polyglot
fn sat_add(a: u8, b: u8) -> u8 { a +| b }
fn sat_sub(a: u8, b: u8) -> u8 { a -| b }
fn sat_mul(a: u8, b: u8) -> u8 { a *| b }
```

| Call | Result |
| ---- | ------ |
| `sat_add(250, 20)` | `255` |
| `sat_sub(10, 30)` | `0` |
| `sat_mul(20, 20)` | `255` |

The operators are Zig-only extensions. The Python/native contract still uses
ordinary wrapping/overflow rules.

The Python fallback lowering in `zig/lowering.py` rewrites the operators to
`_zig_sat_add`, `_zig_sat_sub`, and `_zig_sat_mul`. Those helpers clamp at
`i64` bounds and are not bit-width aware.

## Arbitrary bit-width integers

`uN` and `iN` are accepted for `0 <= N <= 65535`:

```polyglot
fn sat_u24(a: u24, b: u24) -> u24 {
    a +| b
}
```

Zig stores the value at the declared width. For example, `u24` has the range
`0..=16_777_215`, and `sat_u24(16_777_210, 100)` returns `16_777_215`.

At the Python boundary every integer becomes a 64-bit `ctypes` value (`u64` for
unsigned types, `i64` otherwise). Widths above 64 bits are truncated. This is
an ABI boundary, not a change to the Zig-side value.

## `#[parallel]` range loops

```polyglot
#[parallel]
fn scale(out: &mut [f64], inp: &[f64], factor: f64, n: usize) {
    for i in 0..n {
        out[i] = (inp[i] * factor) + 1.0;
    }
}
```

The backend splits the direct range loop into chunks and runs them on up to 16
worker threads, further capped by the available CPU count. It joins all spawned
threads before returning.

Important boundaries:

- Only a direct range `for` in the annotated function is parallelized.
- Nested loops run serially.
- There is no built-in reduction, lock, ordering, or synchronization guarantee.
- Iterations must write disjoint elements or otherwise be race-free by
  construction.
- Parameters and explicitly typed locals are copied into the worker context.
  Pointer/slice arguments still point at the caller's memory.

Save it as `scale.poly` and call it from Python with `ctypes` arrays:

```python
import ctypes
from pathlib import Path

from backends import get_backend
from lexer import lex
from parser import parse

program = parse(lex(Path("scale.poly").read_text(encoding="utf-8")))
namespace = get_backend("zig", trusted=True).compile(program)

n = 10_000
inp = (ctypes.c_double * n)(*[float(i) for i in range(n)])
out = (ctypes.c_double * n)()

namespace["scale"](out, inp, 2.5, n)
assert out[10] == 26.0
```

## SIMD

`SimdN<T>` maps to Zig's `@Vector(N, T)`. The `N` must be a literal in the type
name, for example `Simd4<f64>` or `Simd8<f32>`.

| Builtin | Lowering | Meaning |
| ------- | -------- | ------- |
| `Simd::splat(x)` | `@splat(x)` | Broadcast one scalar to every lane |
| `Simd::load(ptr, index, count)` | `ptr[index..][0..count].*` | Load lanes |
| `Simd::store(ptr, index, count, value)` | `ptr[index..][0..count].* = value` | Store lanes |
| `Simd::reduce_add(v)` | `@reduce(.Add, v)` | Horizontal sum |
| `Simd::reduce_mul(v)` | `@reduce(.Mul, v)` | Horizontal product |
| `Simd::reduce_min(v)` | `@reduce(.Min, v)` | Horizontal minimum |
| `Simd::reduce_max(v)` | `@reduce(.Max, v)` | Horizontal maximum |

Example:

```polyglot
fn simd_dot4(a: &[f64], b: &[f64]) -> f64 {
    let va: Simd4<f64> = Simd::load(a, 0, 4);
    let vb: Simd4<f64> = Simd::load(b, 0, 4);
    Simd::reduce_add(va * vb)
}
```

There is no masking, gather/scatter, or general vector type inference.

The runnable example combines SIMD, `#[parallel]`, and saturating arithmetic:

```polyglot
#[parallel]
fn simd_scale_add(out: &mut [f64], inp: &[f64], scale: f64, bias: f64, chunks: usize) {
    let v_scale: Simd4<f64> = Simd::splat(scale);
    let v_bias: Simd4<f64> = Simd::splat(bias);
    for c in 0..chunks {
        let idx: usize = c * 4;
        let v: Simd4<f64> = Simd::load(inp, idx, 4);
        let res: Simd4<f64> = (v * v_scale) + v_bias;
        Simd::store(out, idx, 4, res);
    }
}

fn main() -> i64 {
    (250 as u8 +| 20 as u8) as i64
}
```

Run `main()` with:

```sh
polyglot run examples/zig_parallel_simd.poly --backend zig
```

## Tests

`tests/test_zig_backend.py` is skipped when `zig` is not installed. It covers:

- saturating `+|`, `-|`, and `*|` on `u8` and `u24`;
- the Python fallback lowering pass;
- a `#[parallel]` scaling loop over `ctypes` arrays;
- SIMD `splat`/`load`/`store` arithmetic;
- SIMD horizontal reduction on the example program;
- CLI `run` and `build` with `--backend zig`.

Run it directly with:

```sh
pytest -q tests/test_zig_backend.py
```

The rest of the Python and native test suites does not require a Zig
toolchain.

## Boundaries

- Experimental and opt-in; not part of the Python/native parity contract.
- Requires `zig` on `PATH`.
- Narrow AST subset; unsupported constructs are not covered.
- C-ABI artifact; not importable as a Python extension.
- Python-boundary integers are 64-bit; wider values truncate.
- `#[parallel]` provides no ordering, reduction, or synchronization.
- SIMD is limited to the named builtins above.
- The Python fallback lowering clamps at `i64` bounds and is not bit-width
  aware.
