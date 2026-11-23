# Mojo-inspired systems bridge (experimental)

`compiler/mojo_bridge/` brings Mojo's signature strengths to `polyglot-lang`
as pure-Python analysis passes:

- **Explicit value/origin semantics** — `Copyable`/`Movable`/`BorrowOnly`
  values with `Owned`/`Borrowed` origins, use-after-move detection (`E002`),
  origin-escape detection (`E001`), and an auditable `UnsafeAnyOrigin` escape
  hatch (`E007`) that is never silent.
- **Contiguous layouts and SIMD lowering** — tensor/buffer layout descriptions
  with contiguity checks, SIMD lane computation, and backend-neutral kernel
  lowering plans.
- **Honest MLIR capability reporting** — `probe_mlir_available()` reports
  `available: False` with the missing tool names when no MLIR/GPU toolchain is
  present, instead of pretending to have native MLIR support.
- **CPython interop with PEP 667 guards** — cold-path Python→native bridging
  for scalars and bytes, plus a hot-loop `f_locals` guard (`E007`).

These modules are **analysis models**. They do not generate native code, do
not invoke MLIR/GPU toolchains, and are not part of the Python/native parity
contract.

- Normative rules: [SPECIFICATION.md](../SPECIFICATION.md) Section 8.14
- Boundary entry: SPEC 10.45
- Tests: `tests/test_mojo_bridge.py`

## Requirements

None beyond Python 3.14.8.

```python
from compiler.mojo_bridge import (
    Origin,
    OriginEnv,
    ValueKind,
    BufferLayout,
    LayoutOrder,
    lower_vectorized_kernel,
    probe_mlir_available,
    bridge_python_value,
    check_pep_667_safe,
)
```

## Value kinds and origins

Every binding has a **value kind** and an **origin**:

| Value kind    | Meaning                                                   |
| ------------- | --------------------------------------------------------- |
| `COPYABLE`    | Owns its value; may be copied freely                      |
| `MOVABLE`     | Owns its value; may be moved but not copied implicitly    |
| `BORROW_ONLY` | Does not own its value; may only be borrowed              |

| Origin             | Meaning                                                            |
| ------------------ | ------------------------------------------------------------------ |
| `OWNED`            | Owns its value; may be moved or borrowed                           |
| `BORROWED`         | References a value owned elsewhere; may not be moved or returned as owned |
| `UNSAFE_ANY_ORIGIN`| Explicit escape hatch widening a reference's origin                |

### Declaring and borrowing

```python
from compiler.mojo_bridge import Origin, OriginEnv, ValueKind, check_borrow

env = OriginEnv()
env.declare("x", ValueKind.COPYABLE, Origin.OWNED)
env.declare("y", ValueKind.MOVABLE, Origin.OWNED)

binding = check_borrow(env, "x")
assert binding.origin == Origin.OWNED
```

A borrow can be constrained to a specific lifetime id:

```python
env.declare("z", ValueKind.COPYABLE, Origin.OWNED, lifetime_id=7)
check_borrow(env, "z", expected_lifetime=7)  # ok
check_borrow(env, "z", expected_lifetime=8)  # error[E001]: lifetime mismatch
```

### Moving

```python
env.move("y")
check_borrow(env, "y")  # error[E002]: borrow of moved value
env.move("y")  # error[E002]: use of moved value
```

Moving a `BORROW_ONLY` or `BORROWED` binding raises `E002` — a borrowed
reference does not own its value.

### Origin escapes

Returning a borrowed binding as owned would let the caller outlive the borrow:

```python
env = OriginEnv()
env.declare("ref", ValueKind.BORROW_ONLY, Origin.BORROWED)
check_origin_escape(env, "ref", return_as_owned=True)
# error[E001]: borrowed binding 'ref' cannot be returned as owned
check_origin_escape(env, "ref", return_as_owned=False)  # ok
```

### UnsafeAnyOrigin

`UnsafeAnyOrigin` is never silent. Declaring it without the `#[unsafe_origin]`
annotation raises `E007`:

```python
env.declare("unsafe", ValueKind.COPYABLE, Origin.UNSAFE_ANY_ORIGIN)
# error[E007]: binding 'unsafe' declares UnsafeAnyOrigin without the
# #[unsafe_origin] annotation
```

Returning an unsafe-origin binding also requires an explicit `allow_unsafe`
flag, mirroring the annotation.

### Scopes

`OriginEnv` is a stack of scopes. `push_scope()` / `pop_scope()` create and
destroy nested scopes; lookups search from the innermost scope outward.
Popping the root scope raises `E003`.

## Buffer layouts and SIMD

A `BufferLayout` describes a contiguous tensor/buffer:

```python
from compiler.mojo_bridge import BufferLayout, LayoutOrder

layout = BufferLayout(shape=(128, 256), dtype="f32", dtype_size=4)
assert layout.is_contiguous() is True
assert layout.element_count == 128 * 256
assert layout.byte_size == 128 * 256 * 4
assert layout.simd_lanes() == 64
```

### Contiguity

A layout is contiguous when its strides match the canonical row-major or
column-major traversal of its shape. A transposed view is not contiguous:

```python
transposed = BufferLayout(
    shape=(128, 256), dtype="f32", dtype_size=4, strides=(4, 128 * 4)
)
assert transposed.is_contiguous() is False
```

### SIMD lanes

`simd_lanes()` returns the largest power-of-two divisor of the element count,
capped at 64. A 1-element layout yields 1 lane (no SIMD benefit).

## Kernel lowering

`lower_vectorized_kernel` produces a **backend-neutral** plan, not native
code:

```python
from compiler.mojo_bridge import lower_vectorized_kernel

kernel = lower_vectorized_kernel("axpy", layout)
assert kernel.vectorizable is True
assert kernel.simd_lanes > 1
```

The returned `KernelLowering` records:

| Field           | Meaning                                                    |
| --------------- | ---------------------------------------------------------- |
| `loop_bounds`   | Per-dimension iteration bounds                             |
| `vectorizable`  | Whether the loop can use SIMD                              |
| `simd_lanes`    | Lane count (1 when not vectorizable)                       |
| `layout`        | The source `BufferLayout`                                  |
| `passes`        | Caller-supplied MLIR-style pass pipeline names             |
| `reason`        | Human-readable justification                               |

Non-contiguous layouts and layouts too small for SIMD are marked
`vectorizable=False` with a concrete reason, and callers fall back to scalar
loops.

## MLIR capability probe

```python
from compiler.mojo_bridge import probe_mlir_available

result = probe_mlir_available()
# {"available": False, "reason": "MLIR tools not found on PATH: mlir-opt, mlir-translate",
#  "missing_tools": ["mlir-opt", "mlir-translate"], "model_only": True}
```

The probe never claims native MLIR support that does not exist. When the tools
are absent — the common case — callers get an explicit fallback signal.

## CPython interop

### Cold-path bridging

`bridge_python_value` converts a Python value to a typed native value for the
cold boundary-crossing path:

| Python value | Native type |
| ------------ | ----------- |
| `None`       | `void`      |
| `bool`       | `bool`      |
| `int`        | `i64` (truncated to 64 bits) |
| `float`      | `f64`       |
| `bytes` / `bytearray` | `[]u8` |

```python
from compiler.mojo_bridge import bridge_python_value

result = bridge_python_value(42)
assert result.native_type == "i64"
assert result.value == 42
```

Unsupported types (`str`, `list`, `dict`, ...) raise `E001` with a message
directing the caller to a typed native binding for hot paths.

### Hot-loop guard (PEP 667)

Reading `frame.f_locals` in a hot loop and mutating the returned dict is
forbidden under PEP 667 (Python 3.13+). `check_pep_667_safe` rejects such
patterns statically:

```python
from compiler.mojo_bridge import check_pep_667_safe

check_pep_667_safe("f_locals in hot loop")
# error[E007]: PEP 667 violation: 'f_locals in hot loop' suggests f_locals
# access in a hot loop
```

The hot-loop path is explicitly **not** modelled here. Callers that need
zero-boxing hot loops must use typed native values (`ctypes` buffers or PyO3
`pyclass`), never per-element `PyObject` boxing.

## Tests

`tests/test_mojo_bridge.py` covers:

- origin declarations, borrows, lifetime matching, and scope push/pop;
- move semantics (use-after-move, double move, moving borrowed values);
- origin escapes (borrowed-as-owned) and `UnsafeAnyOrigin` without annotation;
- buffer layouts (contiguity, row/column major, SIMD lanes, invalid shapes);
- kernel lowering (vectorizable, non-contiguous, small layouts, bound
  mismatches);
- the MLIR probe's structured output;
- cold-path bridging for every supported type and unsupported-type rejection;
- PEP 667 hot-loop guard and normal-context acceptance.

Run it directly with:

```sh
pytest -q tests/test_mojo_bridge.py
```