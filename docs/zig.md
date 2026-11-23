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
| `loop` and `break`/`continue` | ✅ | Labeled-block `break <value>` supported |
| Direct range `for` | ✅ | `start..end` and `start..=end` |
| `return` and block tails | ✅ | Function result |
| Signed `/` and `%` | ✅ | Lowered to `@divTrunc` / `@rem` (truncate toward zero) |
| Slice bounds checks | ✅ | `items[i]` lowers to `_zig_at` with `std.debug.assert` |
| `Result<T, PillarError>` | ✅ | Lowers to `PillarError!T`; `Err(E00n)` → `error.E00n` |
| Integer casts | ✅ | `as` to integer types |
| Indexing | ✅ | Pointer/slice indexing |
| Binary operations | ✅ | Arithmetic, comparison, logical |
| Saturating operators | ✅ | `+`, `-`, `*` with a trailing pipe |
| Arbitrary-width integers | ✅ | `uN`/`iN`, `0 <= N <= 65535` |
| `#[parallel]` range loops | ✅ | Direct loop in the annotated function |
| `SimdN<T>` and SIMD builtins | ✅ | See below |
| User-function calls, `if`/`else` | ✅ | Top-level `fn` calls with scalar arguments |
| Struct declarations and field access | ✅ | `#[packed]` gives a `packed struct` |
| Zig intrinsics | ✅ | `popcount!`, `clz!`, `ctz!`, `bswap!`, `bitreverse!` |
| `bitcast!`, `bitsizeof!`, `sizeof!` | ✅ | Bit-level reinterpretation and layout queries |
| `#[unchecked]` functions | ✅ | `@setRuntimeSafety(false)` for that body |
| Struct construction, enums, `match` | ⬜ | Not lowered |
| Strings, `Vec`, `Option` | ⬜ | Not lowered |
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

## Integer division and remainder

Zig rejects `/` and `%` on signed integers. The backend lowers them to the
truncating builtins so signed arithmetic keeps C-like semantics:

| Source | Lowering |
| ------ | -------- |
| `a / b` (signed) | `@divTrunc(a, b)` |
| `a % b` (signed) | `@rem(a, b)` |

`@divTrunc` truncates toward zero, so `-7 / 2` is `-3` (not `-4` as floor
division would give), and `@rem` produces a signed remainder.

## Bounds-checked slice indexing

`items[i]` on a slice lowers to a call to `_zig_at` (immutable) or
`_zig_at_mut` (mutable), which asserts `i < items.len` before indexing:

```zig
fn _zig_at(comptime T: type, items: []const T, index: usize) T {
    std.debug.assert(index < items.len);
    return items[index];
}
```

Many-item pointers (`[*]T`, produced by `&[T]` parameters) have no `.len`
field in Zig. Reading `.len` on such a pointer emits a `@compileError` with a
clear message — pass the length explicitly as a separate `usize` argument,
matching the `#[parallel]` convention.

## Structured error unions

`Result<T, PillarError>` lowers to the typed error union `PillarError!T`
(not `anyerror!T`), and `Err(E00n)` / `Err(n)` / `Err(error.E00n)` all resolve
to the corresponding `PillarError` member:

```polyglot
fn fail() -> Result<i64, PillarError> { Err(E003) }
```

emits:

```zig
fn fail() PillarError!i64 {
    return error.E003;
}
```

## Labeled-block breaks

`if cond { break N; } else { M }` lowers to a labeled block expression with
`break :blk_N N;`. The label is always untyped (`blk_N: { ... }`) because a
type annotation belongs on the variable being assigned
(`const v: i64 = blk: { ... }`), not on the label itself.

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

## Zig intrinsics

Zig exposes hardware-level operations the other backends do not model. They are
available as macro-style calls and lower 1:1 onto Zig builtins:

| Call | Zig builtin | Result |
|------|-------------|--------|
| `popcount!(x)` | `@popCount` | `i64` number of set bits |
| `clz!(x)` | `@clz` | `i64` leading zeros within the operand width |
| `ctz!(x)` | `@ctz` | `i64` trailing zeros within the operand width |
| `bswap!(x)` | `@byteSwap` | operand's own type, bytes reversed |
| `bitreverse!(x)` | `@bitReverse` | operand's own type, bits reversed |
| `bitcast!(x, T)` | `@bitCast` | `T`, same bit count required |
| `bitsizeof!(T)` | `@bitSizeOf` | `i64` exact bit count |
| `sizeof!(T)` | `@sizeOf` | `i64` byte count, alignment included |

```polyglot
fn density(x: u32) -> i64 { popcount!(x) }        // 0b1011 -> 3
fn lz(x: u8) -> i64 { clz!(x) }                    // 0 -> 8
fn swap32(x: u32) -> u32 { bswap!(x) }             // 0x12345678 -> 0x78563412
fn rev8(x: u8) -> u8 { bitreverse!(x) }            // 1 -> 128
fn float_bits(x: f32) -> u32 { bitcast!(x, u32) }  // 1.0 -> 0x3F800000
```

The operand width comes from the polyglot type, so `u8` and `u64` answers
differ for `clz!`. An unannotated identifier is treated as `i64`; annotate it
with `let` when the width matters. A float operand is rejected with `E001`
(use `bitcast!`), and `bswap!` requires a byte-sized width. The counting
intrinsics widen to `i64`; `bswap!` and `bitreverse!` keep the operand type so a
`u64` result is not truncated.

## Bit casts

`bitcast!(value, T)` reinterprets the bit pattern and requires equal bit counts:

```polyglot
fn bits(x: f32) -> u32 { bitcast!(x, u32) }
fn back(bits: u32) -> f32 { bitcast!(bits, f32) }
```

Mismatched widths (`u8` to `u32`), pointer or slice sources, and unknown types
are `E001` before Zig runs. The backend emits a small generic helper
(`_zig_bitcast`) so the cast has an explicit result type at the call site.

## Packed structs

`#[packed]` declares a Zig `packed struct`: no padding, declaration order
preserved, size equal to the sum of field widths.

```polyglot
#[packed]
struct FrameHeader { version: u3, flags: u5, compressed: bool }

bitsizeof!(FrameHeader)  // 9 bits
sizeof!(FrameHeader)     // 2 bytes
```

Rules:

- every field must have a known bit width: integers (including arbitrary
  widths like `u3`) and `bool`; floats are `E001`, because this backend will
  not guess a layout for them;
- pointers, slices, strings and tuples have no bit width and are `E001`;
- `#[packed]` on a generic struct is `E003` — the bit width would depend on a
  comptime parameter;
- `bitsizeof!` is exact, while `sizeof!` rounds up to the backing integer's
  ABI alignment (`u3+u5+bool` → 2 bytes, `u12+u12` → 4 bytes). Report the
  exact number with `bitsizeof!`.

The generated source carries a comment with the computed bit total, so a layout
change shows up in the emitted Zig as well as in the tests.

## `#[unchecked]` functions

`#[unchecked]` opts a function body out of Zig's runtime safety checks:

```polyglot
#[unchecked]
fn fast_sum(a: i64, b: i64) -> i64 { a + b }
```

The backend emits `@setRuntimeSafety(false);` as the first statement of that
function only. Overflow and integer-range checks in that body become wrapping
behaviour instead of a trap; everything else in the module stays checked, and
the explicit `std.debug.assert` guards from `_zig_at` are unaffected. Do not use
it to silence a real error — it is for hot loops where the bounds were already
proven at the call site.

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
- CLI `run` and `build` with `--backend zig`;
- signed `/` and `%` lowering to `@divTrunc` / `@rem`;
- bounds-checked slice indexing via `_zig_at` / `_zig_at_mut`;
- labeled-block `break <value>` lowering;
- `Result<T, PillarError>` → `PillarError!T` and `Err(E00n)` resolution.

`tests/test_zig_special.py` covers the Zig-exclusive surface: `popcount!`,
`clz!`, `ctz!`, `bswap!`, `bitreverse!` (cross-checked against a Python
mirror), `bitcast!` in both directions including width-mismatch rejection,
`#[packed]` bit/byte sizes with field-type rejection, `bitsizeof!`/`sizeof!`,
and `#[unchecked]`. The worked example is
[examples/zig_intrinsics.poly](../examples/zig_intrinsics.poly).

Run them directly with:

```sh
pytest -q tests/test_zig_backend.py tests/test_zig_special.py
```

The rest of the Python and native test suites does not require a Zig
toolchain.

## Boundaries

- Experimental and opt-in; not part of the Python/native parity contract.
- The intrinsics model operand widths from the inferred polyglot type: an
  unannotated identifier is `i64`, and `usize`/`isize` are 64-bit. A width the
  backend cannot infer is reported (`E001`) rather than guessed.
- Requires `zig` on `PATH`.
- Narrow AST subset; unsupported constructs are not covered.
- C-ABI artifact; not importable as a Python extension.
- Python-boundary integers are 64-bit; wider values truncate.
- `#[parallel]` provides no ordering, reduction, or synchronization.
- SIMD is limited to the named builtins above.
- The Python fallback lowering clamps at `i64` bounds and is not bit-width
  aware.
