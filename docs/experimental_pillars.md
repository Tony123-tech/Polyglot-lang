# Experimental engineering pillars

These modules are opt-in research foundations. They are **not** imported by
the normal `polyglot run` / `polyglot build` path, are not covered by the
stable conformance surface in `SPECIFICATION.md`, and deliberately stop at
the point where a guarantee would be dishonest.

The shared diagnostics in `runtime/diagnostics.py` reuse the public
`E001`–`E008` codes instead of inventing a parallel namespace.

| Code | Meaning in the pillar modules |
| ---- | ----------------------------- |
| `E001` | type, layout, or ABI boundary is not hot-swappable |
| `E002` | proof, termination, or deterministic-replay obligation failed |
| `E003` | AST, API, or call-sequence misuse |
| `E004` | candidate pass or backend specializer failed |
| `E006` | missing snapshot, history version, symbol, or export |
| `E007` | unsupported platform/native feature |
| `E008` | digest, manifest, or serialized-state corruption |

## 1. Hot-patching (`backends/hotpatch.py`)

`diff_programs(old, new)` classifies item changes (`BODY`, `SIGNATURE`,
`STATE_LAYOUT`, `CONSTANT`, `ADDED`, `REMOVED`, `INCOMPATIBLE`) independently
of source locations. `HotpatchManager` compiles the new program, swaps Python
code objects and class dictionaries **in place**, migrates registered live
instances to a new struct layout, and keeps an undo snapshot so `rollback()`
restores code, class surfaces, and object state.

```python
from backends.hotpatch import HotpatchManager
from lexer import lex
from parser import parse

manager = HotpatchManager(parse(lex(old_source)))
counter = manager.register(manager.namespace["Counter"](10))
manager.patch(parse(lex(new_source)), field_defaults={"label": 0})
counter.bump()  # runs the new body on the same object identity
```

Safety boundaries:

- A patch is applied only at a quiescent point by default. `execution()` marks
  a region in which live frames must not be rewritten; suspended coroutines
  finish on the code object they captured.
- Signature changes, removed fields, and changed generics are rejected with
  `E001`. New fields need a type-derived zero value or an explicit
  `field_defaults` entry.
- `NativeSymbolTable` swaps ABI-compatible `ctypes` callbacks through an
  atomic indirection slot. It does **not** rewrite the text of a loaded
  `.so`/`.dylib`; `patch_text()` reports `E007` instead.
- The `mio`/`PyFuture` reactor is preserved because the manager never replaces
  the namespace or recreates live objects.

## 2. Self-synthesizing compiler (`compiler/evolution.py`)

`EvolutionEngine` records compile failures and runtime profiles, proposes a
candidate pass from a small auditable template library, validates it in an
isolated subprocess against a corpus, and promotes accepted candidates into a
content-addressed version history.

```python
from compiler.evolution import EvolutionEngine

engine = EvolutionEngine(".polyglot-history")
engine.observe_compile_failure(cargo_stderr)
candidate = engine.propose()  # template-generated Python source
evaluation = engine.evaluate(candidate, corpus)  # isolated subprocess
if evaluation.ok:
    engine.promote(candidate, evaluation)
engine.rollback(1)
```

Safety boundaries:

- The running translator/codegen is never rewritten in place. A candidate is a
  normal Python file with a `transform(node)` entry point; promotion is an
  explicit, auditable step.
- The sandbox compares observable `main()` results before and after the pass
  and enforces a timeout, so a hostile or broken candidate cannot hang or
  silently change program output.
- History versions store the source, digest, parent, timings, and corpus
  results. A digest mismatch or a corrupt `current.json` reports `E008`.

## 3. Formal synthesis (`proofs/synthesis.py`)

The sound core is a controlled invariant language plus a small checker:

- `interpret_spec(text)` parses `function`, `assume ... between`, `ensure`,
  and `for all ...` lines into a `SynthesisSpec`.
- integer expressions are checked with an `i64` interval domain and a
  polynomial-identity normalizer;
- candidates are enumerated from a finite grammar and accepted only when the
  postcondition is **proved** under the declared ranges, never by sampling;
- `ProofCertificate` is independently re-checked by `check_certificate()`;
- `emit_source`, `emit_program`, and `emit_function_ast` embed an
  `assert!(postcondition)` runtime guard and refuse to emit a candidate that
  does not match its certificate.

```python
from proofs.synthesis import compile_both, emit_program, synthesize

spec = interpret_spec("""
function add(a: i64, b: i64)
assume a is between 0 and 100
assume b is between 0 and 100
ensure result == a + b
""")
candidate, certificate = synthesize(spec)
program = emit_program(spec, candidate, certificate)
assert compile_both(program)["python"]["add"](2, 3) == 5
```

`verify_bounds()` proves literal array indices in straight-line code and marks
dynamic indices `unknown`. `verify_lock_order(LockOrder(edges))` rejects a
cyclic lock-acquisition order with an `E002` cycle witness rather than
claiming deadlock freedom for arbitrary programs. Callers may provide a source
span as `LockOrder(edges, span=(start, end))`; the diagnostic preserves it.

Safety boundaries:

- Unrestricted natural language, arbitrary loops, heap aliasing, and
  "100% verified for every possible specification" are out of scope. The
  grammar and the interval/polynomial fragment define the proved envelope.
- Out-of-range candidate intervals are rejected; a proof is never inferred
  from interval overlap alone.
- The emitted `assert!` checks the postcondition at runtime, so a backend or a
  hand edit that breaks the proof fails loudly. The Python backend lowers it
  to the `_assert` runtime helper; the native backend emits Rust's `assert!`.

## 4. Cross-FFI time travel (`debug/timetravel.py`)

`TimeTravelDebugger` is a checkpoint/replay machine for mixed Python/native
execution. `trace(initial, transitions)` records one checkpoint per state,
`step_back()` / `step_forward()` navigate, and `modify(index, name, value)`
edits a past checkpoint and deterministically replays every recorded
transition after it.

```python
from debug.timetravel import TimeTravelDebugger

debugger = TimeTravelDebugger()
debugger.trace({"x": 1, "total": 0}, [record_total, increment_x, accumulate])
debugger.step_back()
debugger.modify(0, "x", 10)  # dependent future states are recomputed
```

`snapshot_object_graph()` captures an identity-preserving graph of objects,
lists, dicts, and sets; `restore_object_graph()` rewrites them in place.
`capture_python_frame()` copies frame locals without writing to the live
frame. A `NativeStateProvider` supplies `capture()` / `restore()` for native
registers or heap snapshots, and `differential(a, b)` reports key-level
native changes.

Safety boundaries:

- Live CPython frames and suspended `PyO3` stacks are not rewound. State is
  restored through checkpoint replay and an explicit `apply` callback.
- Replaying Python transitions cannot execute Rust code. Unless a pure
  `native_replay(state)` hook is supplied, native checkpoints after a
  retroactive edit are marked stale and restoring them reports `E002`.
- Object graphs created after a snapshot are not resurrected.

## 5. Speculative transmutation (`runtime/transmute.py`)

`SpeculativeTransmuter` profiles exact argument types, asks a pluggable
`Specializer` to compile a faster path in a background thread, installs it
atomically, and guards each call with a tuple comparison. A guard miss
deoptimises to the fallback and continues correctly.

```python
from runtime.transmute import PolyglotSpecializer, transmute, polyglot_type

specializer = PolyglotSpecializer(
    lambda name, sig: f"fn {name}(x: {polyglot_type(sig.types[0])}) -> i64 {{ x + 1 }}",
    backend="native",
)
fn = transmute(lambda x: x + 1, specializer, name="inc", threshold=3)
```

`PolyglotSpecializer` drives the project's JIT or native backend from
generated `.poly` source. The fallback remains authoritative throughout.

Safety boundaries:

- Compilation is off the calling thread's critical path; in-flight calls keep
  using the callable they resolved.
- Deoptimisation is a guard check plus a fallback call. It is fast but is not
  "nanoseconds"; `DEOPT_COST_MODEL` records the actual model.
- Specializer failures are contained (`E004`) and only surface when the caller
  opts into `raise_on_compile_error=True`. After `max_deopts`, the layer
  disables specialisation instead of thrashing.
- Live frames are not rewritten during a hot swap.

## 6. Quantum hot-patching (`runtime/hotpatch`)

The second-generation hot-patch pillar, documented in full at
[`docs/quantum_hotpatch.md`](quantum_hotpatch.md), implements five mechanisms
with a real Rust/PyO3 counterpart:

- atomic GOT/PLT pointer swaps and owned-page AArch64/x86-64 jump sleds;
- CPython 3.13+ fast-local migration through PEP 667 at a safe point;
- selector and native `mio::Poll` registration migration with the sockets kept
  open;
- cooperative user-space RCU epochs for immutable AST roots;
- before-image attribute/buffer journaling with reverse-order rollback.

The first-generation boundaries still hold: foreign `__TEXT` is never patched,
a running frame is never moved to a new `f_code` mid-instruction, and OS
page-level copy-on-write is not claimed without `fork`. Relocation, ABI, and
journal failures are `E008` with character and byte spans; unavailable
primitives are `E007`.

## Tests

```sh
pytest -q tests/test_hotpatch.py tests/test_evolution.py tests/test_synthesis.py \
  tests/test_timetravel.py tests/test_transmute.py tests/test_quantum_hotpatch.py
pytest -q -m slow   # includes the native-backend synthesis/transmutation paths
```
