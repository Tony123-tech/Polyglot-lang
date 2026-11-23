# Gleam-inspired statics (experimental)

`compiler/gleam_style/` brings three of Gleam's signature strengths to
`polyglot-lang` as pure-Python analysis passes:

- **Total ADTs with mandatory exhaustiveness** — a `match` that does not cover
  every variant is an `E002` error with a *concrete* missing-pattern witness
  (`Some(Int)`, `None`), not a vague "match is not exhaustive".
- **Structured bit-array matching** — `<<header:8, len:16, payload:binary>>`
  patterns with endianness and checked bounds validation instead of unchecked
  pointer casts.
- **Cross-target contracts** — pure functional data transformations stay
  portable across Python, Rust/PyO3, Zig, and optional BEAM/JS targets, with
  explicit capability checks (`E007`) when a target lacks a feature.

These modules are **analysis and runtime models**. They do not generate code,
do not touch native toolchains, and do not require `zig`, `cargo`, or a BEAM
installation. They are not part of the Python/native parity contract.

- Normative rules: [SPECIFICATION.md](../SPECIFICATION.md) Section 8.13
- Boundary entry: SPEC 10.43
- Tests: `tests/test_gleam_style.py`

## Requirements

None beyond Python 3.14.8. Every module is pure Python and deterministic.

```python
from compiler.gleam_style import (
    AdtDef,
    Variant,
    check_exhaustive,
    check_total_adt,
    parse_bit_pattern,
    match_bit_array,
    Target,
    Capability,
    emit_contract,
)
```

## Total ADTs and exhaustive matching

An ADT is a named set of variants, each optionally carrying payload types:

```python
from compiler.gleam_style import AdtDef, Variant, check_exhaustive

option = AdtDef(
    "Option",
    (
        Variant("Some", ("Int",)),
        Variant("None"),
    ),
)

result = check_exhaustive(option, [("Some", ("x",)), ("None", ())])
assert result.exhaustive is True
```

A missing variant raises `E002` with a concrete witness:

```python
check_exhaustive(option, [("Some", ("x",))])
# error[E002]: non-exhaustive match on 'Option'; missing patterns: None
```

A wildcard makes the match trivially exhaustive:

```python
check_exhaustive(option, [("Some", ("x",))], has_wildcard=True)
```

### Recursive ADTs and HM-style validation

`check_total_adt` performs a Hindley-Milner-style walk over an ADT's variants,
expanding type aliases and detecting recursive cycles. Recursion is bounded by
a budget; exceeding it is an `E002` diagnostic, never unbounded recursion:

```python
from compiler.gleam_style import check_total_adt

tree = AdtDef(
    "Tree",
    (Variant("Leaf", ("Int",)), Variant("Node", ("Tree", "Tree"))),
    generics=("Int",),
)
report = check_total_adt(tree, bindings={"Tree": "Tree", "Int": "Int"})
assert report["total"] is True
assert report["cycles"]  # Node -> Tree -> Node is recorded
```

### Registry

`AdtRegistry` declares several ADTs up front and checks matches by name:

```python
from compiler.gleam_style import AdtRegistry

registry = AdtRegistry()
registry.register(AdtDef("Bool", (Variant("True"), Variant("False"))))
registry.check_match("Bool", [("True", ()), ("False", ())])
```

## Bit-array pattern matching

A pattern is wrapped in `<<...>>` and contains comma-separated segments of the
form `name:WIDTH[:TYPE][:ENDIAN]`:

| Segment form          | Meaning                                                      |
| --------------------- | ------------------------------------------------------------ |
| `header:8`            | 8-bit integer (type defaults to `int` when a width is given)  |
| `len:16:little`       | 16-bit little-endian integer                                 |
| `payload:binary`      | Variable-width: consumes the remaining buffer                 |
| `value:32:float:big`  | 32-bit big-endian float                                      |
| `cp:8:utf8_codepoint` | One UTF-8 code point                                         |

```python
from compiler.gleam_style import parse_bit_pattern, match_bit_array

pattern = parse_bit_pattern("<<header:8, len:16, payload:binary>>")
result = match_bit_array(pattern, bytes([0xAB, 0x00, 0x05, 1, 2, 3, 4, 5]))
assert result.bindings["header"] == 0xAB
assert result.bindings["len"] == 5
assert result.bindings["payload"] == bytes([1, 2, 3, 4, 5])
```

### Safety properties

- Every read is **bounds-validated against `len(data)` before extraction**. A
  short buffer raises `E001` (size mismatch), never an out-of-bounds read.
- No `ctypes.cast`, no `struct.unpack_from` without a length guard, and no
  raw pointer arithmetic.
- Fixed segments must be byte-aligned and a multiple of 8 bits; a 12-bit
  `int` segment raises `E001`.
- At most one variable-width segment is allowed, and it must be last.
- The total fixed width is capped at 4096 bits to prevent unbounded work.

## Cross-target contracts

`emit_contract` validates that a pure transformation's required capabilities
are supported by a target, raising `E007` with the missing capability names:

```python
from compiler.gleam_style import Capability, Target, emit_contract

emit_contract("adt_test", {Capability.ADT}, Target.PYTHON)  # ok
emit_contract("actor_test", {Capability.OTP_ACTOR}, Target.BEAM)  # ok
emit_contract("actor_test", {Capability.OTP_ACTOR}, Target.PYTHON)
# error[E007]: contract 'actor_test' is not supported on target python;
# missing capabilities: otp_actor
```

### Capability matrix

| Capability         | Python | Rust | Zig | BEAM | JS |
| ------------------ | :----: | :--: | :-: | :--: | :-: |
| `adt`              |   ✅   |  ✅  | ✅  |  ✅  | ✅ |
| `exhaustive_match` |   ✅   |  ✅  | ✅  |  ✅  | ✅ |
| `bit_array`        |   ✅   |  ⬜  | ⬜  |  ✅  | ✅ |
| `simd`             |   ⬜   |  ⬜  | ✅  |  ⬜  | ⬜ |
| `otp_actor`        |   ⬜   |  ⬜  | ⬜  |  ✅  | ⬜ |
| `supervision`      |   ⬜   |  ⬜  | ⬜  |  ✅  | ⬜ |
| `error_unions`     |   ✅   |  ✅  | ✅  |  ✅  | ✅ |
| `c_interop`        |   ⬜   |  ✅  | ✅  |  ⬜  | ⬜ |
| `python_interop`   |   ✅   |  ✅  | ⬜  |  ⬜  | ⬜ |

A `TargetDescriptor` can override or disable capabilities for a specific
deployment:

```python
from compiler.gleam_style import Capability, Target, TargetDescriptor

desc = TargetDescriptor(target=Target.ZIG, disabled=frozenset({Capability.SIMD}))
```

## Tests

`tests/test_gleam_style.py` covers:

- exhaustive and non-exhaustive matches, including concrete missing-pattern
  witnesses with payload types;
- wildcard matches, unknown variants, duplicate variants, and empty ADTs;
- `AdtRegistry` registration, lookup, and duplicate/unknown errors;
- recursive ADT cycle detection and budget exhaustion;
- bit-array parsing (defaults, malformed syntax, zero width, over-wide
  patterns) and matching (big/little endian, variable payloads, short buffers,
  non-byte-aligned segments);
- cross-target contracts, capability overrides, and disabled capabilities.

Run it directly with:

```sh
pytest -q tests/test_gleam_style.py
```