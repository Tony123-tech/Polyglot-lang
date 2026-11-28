# NOTES.md

## Quick index

- N1–N2d: Core language (functions, control flow, structs)
- N3–N4: Vec and String stdlib
- N5: Option/Result, async, sandbox
- N6: Remaining Vec/String methods
- N7: usize/isize distinct types
- N8: HashMap
- N9: Result/Option combinators
- N10: SPEC v0.2.7 conformance
- M10: Termination checking
- M5: Async extensions (Send-aware PyFuture, async impl methods, cancellation, timeout)
- F1: Deliverable CLI
- Zig (experimental): saturating ops, arbitrary-width integers, `#[parallel]`, SIMD, C-ABI boundaries, intrinsics, `#[packed]`, `#[unchecked]`, overflow reports/rotations/conversions/buffers
- Runtime failures: located `E002` guards on both backends, IEEE 754 float division
- Ownership: explicit `drop(x)`, borrow-vs-move conflicts, bounded NLL, loop fixpoint
- Rust ergonomics: trait coherence/composition/dispatch plans, arena handles, async boundaries
- Repository root cleanup: stage modules into `compiler/core/` and `compiler/analysis/`, shims removed, string-embedded imports swept
- Host interpreter: 3.14.8 floor checked at the CLI door, refusal as `error:`/`help:`, PEP 649 reason
- Parser/translator/codegen: `if let` as a tail expression (10 bugs)
- Priority features: `vec!`, `String::from`, lazy fallbacks, `std:fs`, expression-position `if let`, entry API, termination, diagnostics
- Bug log convention: every defect found gets its own `Symptom` / `Cause` / `Fix` / `Lesson` entry here before or alongside the fix commit
- Latest fixes: `check_program` import order, UTF-8 string indexing in `stdlib`, native cache manifest validation

**Latest baseline:** 3520 passed, 5 skipped, 31 deselected (fast suite, after the demo smoke test, the Linux `RTLD_DEFAULT` fix, the GOT-scan restriction and the backend-registry split); slow suite 31 passed, 3520 deselected.

**Baseline ledger.** The counts quoted further down are the numbers printed by the
run that closed each batch, oldest first: 3258 (comprehension pass) → 3340
(priority features) → 3390 (runtime failure reporting, 11.30) → 3411 (explicit
`drop` and borrow conflicts, 11.31) → 3498 (Rust ergonomics, 11.32) → 3511
(host-interpreter floor, 12.1) → 3512 (console-script path fix, 12.2), 3513 (xdist parallelization and pinned `scalar-add` parity) → 3518 (demo smoke test, `RTLD_DEFAULT` platform value, gate failure logs) → 3519 (GOT scan restricted to readable
file-backed mappings) → 3520 (`backends.base` import-graph probe). Two sections
still quote 3258 because the later section copied the earlier figure instead of
re-running the suite; read each quoted count as the close of its own batch, not as
the current total.

The earlier 3511 / 6 figure was explained wrongly. The missing test was not a
toolchain skip: `tests/test_f1_cli.py::test_console_script_entry_point` skipped
whenever `shutil.which("polyglot")` returned `None`, which depends on whether the
virtualenv's `bin` directory happened to be on `PATH` — true for `make check` with
an activated venv (3512 / 5), false for `.venv/bin/python -m pytest` run from a
plain shell (3511 / 6). A baseline that changes with how you invoke pytest is not a
baseline. The test now resolves the script from `sysconfig.get_path("scripts")` and
falls back to `PATH`, so the install either has the script or does not, independent
of activation.

Rust-like syntax is embedded in Python through the shared lexer, parser, and AST provided by `polyglot-lang`. Programs can be translated from the Rust AST to Python AST and executed with `compile()` and `exec()`, or emitted as Rust source with PyO3, built with Cargo, and loaded as native Python extensions.

## Repository root cleanup — imports that are not import statements

**Symptom:** Moving the nine stage modules into `compiler/core/` and
`compiler/analysis/` and deleting the root shims looked mechanical: an AST codemod
rewrote 308 real `import` statements in 126 files with zero parse errors, and the
targeted tests passed. The first full suite after the shim removal reported 12
failures, and four of them were not import problems in the static sense at all:
`compiler/evolution.py` could no longer self-compile a candidate pass, the
regression harness reported `RuntimeError: lexer/parser are unavailable` and
`monomorph=unavailable` for every probe, and `test_package_exports` could not
import `lexer` at all.

**Cause:** Three classes of reference survive an AST rewrite. First, *source text
that is later imported*: `evolution.py` stores its auditable pass templates and its
runner script as string literals, writes them to disk, and imports them back, so
`from ast_nodes import Binary` sits inside a triple-quoted block the codemod never
sees. Second, *dynamic imports*: `regression/harness.py` resolves stages through
`importlib.import_module("lexer")` with a fallback that reports "unavailable", and
`runtime/capabilities.py` stores the stage module name as a string field, so a stale
name silently downgrades capability probes to `status: unavailable` instead of
failing loudly. Third, *documentation samples*, which are the first thing a reader
copy-pastes. A fourth class was pure path arithmetic: `include_expander` resolved
`Path(__file__).parent / "stdlib"`, and `tools/lint_statement_leaks.py` read
`ROOT / "parser.py"`, which after the move was the three-line shim — the guard
would have stayed green while inspecting nothing.

**Fix:** The string sites now name the packaged path (`compiler.core.ast_nodes`,
`compiler.analysis.monomorph`), the capability entries store
`compiler.analysis.*`, the harness imports `compiler.core.*`, and the doc samples in
`docs/advanced.md`, `docs/zig.md` and `docs/experimental_pillars.md` show the path a
reader can actually run. The evolution allowlist became
`_ALLOWED_IMPORT_MODULES` and accepts both `ast_nodes` and
`compiler.core.ast_nodes`, because a pass written before the move is still a valid
auditable transform; rejecting it would have been a behaviour change disguised as a
rename. `STDLIB_DIR` uses `parents[2]`, and the statement-leak guard reads the real
parser.

**Lesson:** "The codemod parsed everything and touched 308 statements" is a
statement about syntax, not about references. Anything that turns a name into an
import later — templates that get written to disk, `import_module`, a config field,
a doc sample — needs a separate sweep, and the sweep should be a full test run plus
a grep for the old names inside strings, not a review of the diff. The failures that
matter most here were the quiet ones: a guard reading a shim and a capability probe
reporting `unavailable` both look like success from the outside.

## Host interpreter — a requirement nothing actually checked

**Symptom:** SPEC 9.9 said "Python 3.14.8 or later", `pyproject.toml` said
`requires-python = ">=3.14.8"`, and README repeated it. None of that had any
effect on a running interpreter. On 3.12 the suite failed with
`NameError: name 'i64' is not defined` from three inline-language tests, with no
message pointing at the host version, and a user on 3.11 would have seen the same
thing and reasonably concluded the language was broken rather than the interpreter.

**Cause:** The front end depends on PEP 649 deferred annotation evaluation, which
became the default in 3.14. Before it, annotations are evaluated at definition time,
so `def add(a: i64) -> i64` raises before the `@inline_lang` decorator is reached —
the failure is in annotation evaluation, not in the interpreter's ability to run the
transpiler. Declaring a floor in metadata only refuses at `pip install` time; an
already-installed package, a source checkout, or a `python -m` run never hits it.

**Fix:** `diagnostics/host.py` holds the floor once (`MIN_VERSION`, plus
`require_supported_python` and `is_supported`), and `api/cli.py::main` calls it before
`build_parser()`, exiting 2 with one `error:` line naming both versions and one
`help:` line naming PEP 649 and the README section. SPEC 9.9.1 states the reason and
the condition for lowering it (a compatibility layer plus a recorded host-version
matrix); 9.9.2 states the refusal contract, including that it may not surface as a
traceback. README has a `Host interpreter` section above Quick Start. 10.57 states
what is *not* claimed: no 3.11–3.13 run is recorded, and the refusal is the only part
that a test proves.

**Lesson:** A version floor that only exists in metadata is a comment. The useful
question is not "did we document the requirement" but "what does the first line of
output say when the requirement is violated". Two lines at the door cost less than
one `NameError` three layers deep — and the boundary stays honest, because the
refusal is tested while the wider range is not.

## Rust ergonomics — plans that answer a decision, not code that pretends

**Goal:** cover the three things Rust programmers ask for most — coherent trait
extension without orphan-rule walls, splitting a value into disjoint mutable
views instead of cloning it, and async/PyO3 boundaries that neither lose error
context nor leak work on cancellation — in a form that can be tested without
Cargo.

**The shape of the answer.** Each capability is a *plan*: a function that
validates its input, returns a deterministic description (emitted headers,
arm order, cleanup order, schedule), and raises `PolyglotError` with the
framework codes when the input is incoherent. That is a deliberate choice over
generating Rust source, because the value of each feature is a decision — do my
impls cohere, are my dispatch arms disjoint, is my cleanup order correct —
and a decision is cheaper to assert than a ninety-line `unsafe` block that no
test compiles. It also keeps `cargo check`/`cargo clippy` out of the loop,
which is why `backends/rust/` needed no new Cargo scaffolding: importing it
costs 0.09s and builds nothing.

**Where each rule comes from.** `rste_impl_plan` encodes the orphan rule
(`impl` is legal when the crate owns the trait *or* the type) and refuses a
blanket impl plus a concrete impl unless the concrete type is in the blanket's
explicit exclusion list — with the flag `requires_specialization=False` only
when the plan really is stable-Rust reachable. `rste_fast_path` refuses to
infer coverage: `covers` is explicit, two arms sharing a type is `E002`, and a
generic `Copy` arm reports `stable=False`. `rsbe_arena_simulate` runs the
protocol rather than describing it, so "a stale key is refused after the slot
is reused" is observed in a trace, not asserted in prose. `rsai_gil_plan`
refuses a Python-object operation in a released region, which is the PEP 667
hazard `allow_threads` creates, instead of documenting it in a comment.

**What is not claimed.** No generated code, no measured speedup, no soundness
guarantee for the emitted Rust, and no trait solving: associated-type equality,
lifetime bounds, and auto-trait leakage are outside the plan, so a plan it
accepts can still be rejected by rustc. All five limits are written down in
SPEC 10.56 rather than left to the reader.

### Bug 1 — the emitted split binding was not valid Rust

**Symptom:** the smoke run of `rsbe_split_fields(["a", "b"])` printed
`('let &mut a = &mut self.a;', 'let &mut b = &mut self.b;')`.

**Cause:** the pattern was copied from a slice-window split, where
`let &mut w = &mut xs[0..3];` is correct because the right-hand side is a
`&mut [T]` and the pattern destructures the reference. For a field access the
right-hand side is already a place of type `&mut T`, so the extra `&mut` in the
pattern makes the binding a `&mut &mut T` — the plan was emitting code that
does not compile, in a batch whose whole point is that the emitted strings are
trustworthy.

**Fix:** field views emit `let a = &mut self.a;`, and the window views keep
their destructuring pattern because that one is correct.

**Lesson:** an emitted string is a claim about Rust. A plan that prints
uncompilable code is worse than one that prints nothing, because the string is
what a reader copies.

## Explicit release and borrow conflicts — one ownership pass, five states

**Goal:** make `drop(x)` a real operation with Rust semantics instead of a name
that raised `NameError`, and make the ownership pass reject the four mistakes a
move checker alone cannot see: use after release, double release, use after
move, and a move or assignment while a value is borrowed — with the same code
and the same line pair on the Python and native backends.

**Where the code lives.** `backends/borrow_checker.py` holds the borrow/drop
pass, and `backends/move_check.py::check_moves` is the single entry point that
runs it. That placement was argued, not assumed: `backends/rust/__init__.py`'s
first line imports `NativeBackend`, so a checker under `backends/rust/` would be
unreachable from the Python backend, and the Python backend is the one that has
to produce the same diagnostic. The class names and messages keep Rust
vocabulary (`BorrowError`, `kind`, `moved_at`), which was the compromise.

**Five states, four `kind` values.** A binding is `alive`, `moved`, or
`dropped`, and the pass records the line where each transition happened so the
diagnostic can point at both ends. `kind` says which rule fired:
`"moved"` (use or drop after a move), `"drop"` (use after release),
`"double-drop"` (a second release), `"borrow"` (a move, drop, or assignment to a
borrowed value). `BorrowError` subclasses `MoveError` and keeps `code = "E009"`,
so `pytest.raises(MoveError)`, the CLI's error formatter, and SPEC 8.16 all agree
that ownership is one language rule rather than two.

**Why the borrow pass runs first.** Moving a borrowed value is rejected by both
passes; which one reports it decides the message. Reporting `use of moved value`
would describe the symptom, while the borrow is the cause, so the borrow pass
runs first and reports `cannot move \`s\` because it is borrowed by \`r\``
(the shape of rustc's `E0505`). A move that leaves nothing borrowed falls
through to the move pass unchanged.

**Bounded NLL.** A borrow is dead once its holder is never read again, so
`let r = &s; println!("{}", r); drop(s);` is legal. The model is one dictionary
of last-read lines per function, gathered by a generic AST walk, and a live
borrow is "last read after this line". Borrows that escape (into a returned
value, a field, a closure) are outside it, which is the honest direction:
the pass would rather miss a violation than reject a correct program.

**Loops are a fixpoint, not a single pass.** The earlier shape of the check -
run the body once, then compare the tail state against the entry state and
report every binding that went from `alive` to `moved`/`dropped` - is wrong for
the most common loop in the repository:

```polyglot
let mut mixer = seed;
while i < 4 {
    mixer = _wadd(mixer, 0x9E3779B97F4A7C15);   // reassigned first
    state.push(_splitmix(mixer));               // then read
    i = i + 1;
}
```

The body now runs from the entry state, the entry and tail states are joined
(`dropped` over `moved` over `alive`), and the body runs again from the joined
state. Only the second pass can prove a violation, because only it starts
every iteration where the first one ended.

### Bug 1 — every call looked like it produced an owned value

**Symptom:** the full suite failed during collection with
`error[E009]: use of moved value: \`mixer\`` at `stdlib/random.poly`, a file with
no moves in it. `_wadd` returns `u64`.

**Cause:** `_is_owned` answered `True` for any `Call` except `drop`. `mixer`
is declared without an annotation (`let mut mixer = seed`), so the pass learned
its owned-ness from the initializer, saw a call, and marked it owned. The next
read of `mixer` was then a use of a moved value — a false positive produced by
an over-approximation, not by the program.

**Fix:** owned-ness follows a declared return type. `check_program_borrows`
collects every `FnDecl`/method name with its return type and hands it to the
per-function checker; `_is_owned` marks a call owned only when that return type
is known and non-`Copy`. An unknown callee is treated as non-owning, and
reassignment clears owned-ness when the new value is `Copy`.

**Lesson:** an ownership pass reads types or it guesses. Guessing "this call
returns something owned" turns every numeric helper in the stdlib into a move.

### Bug 2 — the arity error lost to the ownership error

**Symptom:** `tests/test_diff_fuzz.py` failed on the seed-2026 E003 template
`__fs_read_text(p, p)`: the diagnostic was `use of moved value: \`p\`` instead of
`__fs_read_text expects 1 argument, got 2`, and the Python/wasm parity case
diverged because the Python backend reported `E009` while wasm reported `E003`.

**Cause:** two ordering mistakes. `check_moves` ran before the translator's
host-seam arity check, and `RustCodegen.generate` called the borrow pass
directly, so the native path skipped the shared entry point entirely. Passing
`p` twice is genuinely a use-after-move — the call is what is wrong, and
signature errors outrank ownership.

**Fix:** a program-wide arity scan runs first inside `check_moves`, using the
same `HOST_CALL_ARITIES` table and the same span, and `RustCodegen.generate`
calls `check_moves` instead of the borrow pass so no backend can take a
different order.

**Lesson:** "the check is shared" is only true if every entry point is the
`shared` one. The duplicate call site looked harmless because it was
idempotent, and it was - it was just in the wrong place.

### Bug 3 — the headline described the pass, not the mistake

**Symptom:** `polyglot check` printed
`error[E009]: borrow check failed in \`main\`` for a use-after-drop; the test
that pinned "the first line names the violation" failed.

**Cause:** `format()` put the function name in the headline and the specific
reason on the `reason:` line, which is the opposite of the five-line contract
in the rest of the project: the headline is the message, the `note:` carries
the context.

**Fix:** the headline is the reason (`error[E009]: use of dropped value: \`s\``),
and the function name moved to `note: ownership and borrow rules enforced in
\`main\` before lowering`.

**Lesson:** a shared error formatter is a contract. Reusing `E009` while
inventing a new headline shape extends the surface that every CLI test has to
pin.

**What is deliberately not pinned.** `tests/test_ownership_drop.py` does not
pin three adjacent behaviours, because the model does not claim them yet:
reassigning after a drop, dropping a projection such as `drop(payload.name)`,
and a `drop` inside a closure. They are
listed in the module docstring so a future change is a decision rather than an
accident.

## Runtime failures — a located guard instead of a host exception

**Goal:** make a failure that happens *while* a program runs look like every
other diagnostic: a code, a message, and the position of the expression that
failed. Before this batch, four different failures on the same program printed
`at: <unknown>` and a `note:` naming whatever Python class raised.

**What the probe found.** Running real programs through the CLI (not the
lowering functions) showed the actual shapes:

```text
% polyglot run /tmp/rz.poly
error: division by zero
  note: ZeroDivisionError
  at: <unknown>
```

`v[5]` on a three-element `Vec` printed `IndexError: list index out of range`,
and `panic!`/`assert!` printed the same generic help line. Two probes looked
like defects and were **not**: `panic("boom")` without the `!` is an ordinary
call to an undefined function, so `NameError` is correct there, and `u8 250 +
20` returning 14 is the wrapping behaviour of SPEC 2.4.9. Both were dropped
after checking the source rather than reporting them.

**Design choice: reuse `E002`.** The public code set stops at `E008` and there
is no runtime-specific code, so a guard failure reuses `E002` ("semantic proof
or termination obligation") — a guard that fails at run time means the
obligation the compiler assumed could not be discharged. No new code, no
parallel namespace.

**Position source: the AST node, not the exception.** `_translate_binary_raw`
and the `Index` branch already hold the node, so `/`, `%`, and indexing emit
`_div(left, right, line, col)`, `_mod(...)`, and `_at(container, index, line,
col)`; the macro branch appends `line=`/`col=` keywords to `panic!` and the
assert family. The CLI needed almost nothing: it already read `.line`/`.col`,
so giving the exception object those attributes fixed `at: <unknown>`. The only
CLI change is the head line (`error[E002]: …` when the object declares
`runtime_failure = True`) and `note: runtime failure`.

**Watch out: `assert!` used to raise `AssertionError`.** Three tests and one
framework test expected that, and `vec_sort([])` expected `IndexError` because
the *user* code indexed an empty vec. All four were updated to the new
contract rather than keeping a compatibility inheritance, because a guard that
ghosts its own position is the bug being fixed.

**Bug found while writing the float test — `1.0 / 0.0` raised.** Rust floats
follow IEEE 754 (`x/0` is a signed infinity, `0/0` is NaN); Python raises. The
helper now catches the zero case and returns the IEEE result, which also
removes a silent Python/parity divergence.

### Bug 1 — float division by zero raised instead of returning infinity

**Symptom:** `fn main() -> f64 { 1.0 / 0.0 }` exited 1 with
`error: division by zero` / `note: ZeroDivisionError`. A native Rust build
returns `inf`.
**Cause:** `_runtime_int_div` fell through to Python's `left / right` for any
non-integer operand, and Python raises for floats too.
**Fix:** catch `ZeroDivisionError` for the float path, return
`copysign(inf, left) * copysign(1.0, right)`, and NaN for `0/0`.
**Lesson:** "integer-only guard" was an assumption about the host, not a
checked fact. The parity suite had no float-division-by-zero case, so nothing
failed.

### Bug 3 — the documented bounds check never existed, and an out-of-range read segfaults

**Symptom:** `fn pick(items: &[i64]) -> i64 { items[5] }` on a three-element
buffer printed nothing and died with `Segmentation fault: 11` (exit 139).
SPEC 8.12.12 and `docs/zig.md` both claimed `_zig_at`/`_zig_at_mut` helpers that
`std.debug.assert(i < len)`.
**Cause:** two separate gaps. `&[i64]` lowers to the many-item pointer
`[*]const i64`, which has no `.len`, so the claimed helper could not have
compiled against a real parameter; and the library is built with
`-O ReleaseFast`, which turns Zig's own bounds check off. The claim had lived in
the docs since 11.25 without a failing test to contradict it, because the
existing "bounds-checked indexing" test only read in-range.
**Fix:** the claim is withdrawn rather than papered over. SPEC 8.12.12 now states
that indexing lowers to raw Zig access, that a `[*]T` parameter carries no
length, and that an out-of-range read is undefined behaviour; 10.54 records the
asymmetry, `docs/zig.md` documents the "pass the length and compare" pattern,
and the stale ✅ row is now ❌ with the release-mode reason. Guarding reads
through a helper was tried and reverted: it cannot compile for `[*]T`.
**Lesson:** a ✅ in a feature table needs a test that would fail if the feature
were absent. In-range reads never prove a bounds check.

### Bug 2 — the generated Zig print broke the build

**Symptom:** every `--backend zig` failure printed
`error: Zig build-lib failed:` with no detail.
**Cause:** the helper block is a Python triple-quoted string, so `\n` inside
the Zig format string became a real newline and produced an invalid Zig string
literal — a compile error, not a runtime failure, which is why the real cause
was swallowed.
**Fix:** escape the newline as `\\n` in the Python source; the failure path
then prints the two intended lines.
**Lesson:** generated-source helpers need a smoke build for the *failure* path
too, not just the happy path.

## Zig backend — an opt-in execution mode

**Goal:** Explore Zig as an alternative code generator without changing the
shared front end, the Python/native parity contract, or the stable CLI
behaviour. The backend is selected with `get_backend("zig")` or `--backend zig`
on `run`, `build`, `exec`, and `compile`.

### Mechanics

`ZigCodegen` walks the shared AST and emits Zig source. Each top-level function
gets a normal Zig `fn` plus an `export fn __cabi_<name>` wrapper. The wrapper
converts C-ABI integer parameters to the declared Zig type, calls the function,
and converts the result back. `ZigBackend` hashes the generated source, reuses
a cached dynamic library when the hash matches, and otherwise runs:

```text
zig build-lib <src> -dynamic -O ReleaseFast -femit-bin=<lib>
```

`ctypes.CDLL` loads the library, and `compile()` returns a namespace whose
callables are the `__cabi_*` symbols. The cache lives under
`~/.cache/polyglot-zig`; the directory is created with mode `0700`. The
artifact is a C-ABI library, not a PyO3 extension, so `polyglot build` cannot
produce something importable with `import <stem>`.

### Saturating arithmetic

Zig already has `+|`, `-|`, and `*|`, so the backend emits them directly.
`zig/features.py` provides the shared Python-side model:

- `saturating_op(left, right, op, type_name)` computes the raw result and
  clamps it to `int_bounds(type_name)`.
- `int_bounds` understands both standard widths and arbitrary `uN`/`iN`.
- `wrap_arbitrary_int` truncates a value to a declared width.

`zig/lowering.py` is a separate, zero-intrusion fallback. It deep-copies the
AST and rewrites saturating binary nodes into calls to `_zig_sat_add`,
`_zig_sat_sub`, and `_zig_sat_mul`, which the Python backend can inject with
`inject_zig_runtime`. That fallback clamps at `i64` bounds; it is deliberately
not bit-width aware.

### Arbitrary bit-width integers

`parse_arbitrary_int_type` accepts `uN`/`iN` for `0 <= N <= 65535`, and
`ZigCodegen` emits the same type name to Zig. Zig stores the value at that
width. The Python boundary is the limiting factor: every integer becomes a
64-bit `ctypes` value (`u64` for unsigned types, `i64` otherwise), so values
wider than 64 bits are truncated when they cross back into Python. The spec
records this as a platform boundary rather than pretending the ABI preserves
the full width.

### `#[parallel]` lowering

For a `#[parallel]` function, the backend copies the parameters and function
locals into a generated context struct, then splits each direct range `for`
loop into chunks. It spawns up to `min(16, CPU count)` workers, each running a
generated `worker_N` function over a half-open chunk, and joins every spawned
thread before the loop completes.

The important limits are structural:

- only a direct range loop in the annotated function is parallelized;
- nested loops and loops inside other statements run serially;
- a value copy means scalar writes do not escape, while pointer/slice
  arguments still alias the caller's memory;
- there are no locks, reductions, ordering guarantees, or race detection.

This is enough for the tested scaling pattern, not a general parallel runtime.

### SIMD lowering

`SimdN<T>` is recognized syntactically and maps to `@Vector(N, T)`. The builtin
calls lower as follows:

| Source | Zig |
| ------ | --- |
| `Simd::splat(x)` | `@splat(x)` |
| `Simd::load(ptr, i, n)` | `ptr[i..][0..n].*` |
| `Simd::store(ptr, i, n, v)` | `ptr[i..][0..n].* = v` |
| `Simd::reduce_add(v)` | `@reduce(.Add, v)` |
| `Simd::reduce_mul(v)` | `@reduce(.Mul, v)` |
| `Simd::reduce_min(v)` | `@reduce(.Min, v)` |
| `Simd::reduce_max(v)` | `@reduce(.Max, v)` |

The length must be a literal in the type name, and there is no masking,
gather/scatter, or general vector inference.

### Honest boundaries

The tested surface is intentionally narrow: free functions, scalar parameters
and returns, `let`, assignment, `while`, direct range `for`, `return`, integer
casts, indexing, binary operations, saturating operations, arbitrary-width
integers, `#[parallel]` range loops, and the SIMD builtins. Structs, enums,
`if`, `match`, strings, `Vec`, `Option`, `Result`, and ordinary user-function
calls are not lowered. In particular, the emitter has no general call lowering;
the tested programs either use the SIMD builtins or avoid calls. Unsupported
constructs are outside the contract and are not guaranteed to be rejected with
a diagnostic, so they must not be presented as working features.

### Tests and lesson

`tests/test_zig_backend.py` skips when `zig` is unavailable and covers
saturating arithmetic, arbitrary widths, the fallback lowering, parallel
scaling, SIMD arithmetic, horizontal reductions, the runnable example, and
the CLI run/build paths.

**Lesson:** An alternative backend is useful as an experiment only when its
boundaries are explicit. Reusing the AST keeps the front end honest, while a
C-ABI wrapper, a narrow lowering subset, and a separate cache make it clear
that the Zig mode is not a drop-in replacement for the parity-tested backends.

## Experimental pillars — drawing the honesty line

The five opt-in pillars (`backends/hotpatch.py`, `compiler/evolution.py`,
`proofs/synthesis.py`, `debug/timetravel.py`, `runtime/transmute.py`) were
built as architectural foundations, not as claims that the impossible
guarantees are reachable. These boundaries drove the design:

- **Mid-instruction frame/stack rewriting.** CPython does not expose a portable
  way to replace `f_code` while a frame is executing between instructions, and
  a suspended PyO3 stack cannot be rewritten through a stable ABI.
  First-generation hot-patching therefore applies at quiescent points and
  preserves object identity; time travel is deterministic checkpoint/replay,
  not live rewind. The second-generation subsystem adds future-entry `__code__`
  swaps and PEP 667 fast-local migration at safe points, but still refuses
  mid-instruction `f_code` replacement.
- **Foreign executable-text patching.** Loaded `.so`/`.dylib` text is not
  patched. First-generation native hot-swapping goes through an atomic
  `NativeSymbolTable` indirection, which is ABI-safe and testable; foreign text
  patching is rejected with `E007`. The second-generation subsystem permits
  only owned GOT/PLT cells and process-owned jump sleds.
- **Natural language to proofs.** A controlled invariant grammar is parsed
  into symbolic conditions and checked with an i64 interval domain plus a
  polynomial normalizer. Candidates are enumerated from a finite grammar and
  accepted only when the obligation is proved. Interval overlap alone never
  counts as equality — that unsound rule was caught while testing `a + b`
  against `2 * a`.
- **Nanosecond deoptimisation.** The guard is a tuple comparison followed by
  the Python fallback. `DEOPT_COST_MODEL` records the real model; compilation
  stays off the critical path and in-flight calls keep their resolved
  callable.

The shared `runtime/diagnostics.py` maps pillar failures onto the public
`E001`–`E008` codes instead of adding a second error namespace. The full API
and boundary list live in `docs/experimental_pillars.md`.

## Quantum hot-patching — second-generation boundaries

`runtime/hotpatch` is a new opt-in subsystem, not a replacement for
`backends/hotpatch.py`. It implements real primitives that the first
generation only exposed at quiescent points:

- GOT/PLT cells are swapped with one aligned atomic word after temporarily
  widening the page, and owned `MAP_JIT` pages accept AArch64/x86-64 jump
  sleds. Foreign `__TEXT` is still refused with `E007`.
- Future entries of a function get the new `__code__`; live frames keep their
  captured code and have fast locals migrated through PEP 667. A
  mid-instruction `f_code` rewrite remains impossible and is refused.
- A fresh selector or `mio::Poll` adopts the old interest registry while the
  sockets stay open, so queued datagrams and connections are not lost.
- RCU is cooperative user-space epoch reclamation, not a kernel RCU syscall;
  a reader that never leaves its guard produces an `E008` timeout.
- The journal captures attribute/buffer before-images and replays them in
  reverse order. It does not claim OS page-level copy-on-write without
  `fork`; `deep=True` is the explicit container-isolation mode.

The Rust/PyO3 crate in `runtime/hotpatch/native` mirrors these mechanisms with
`unsafe` Mach/Mach-O, `libc`, and `mio` code. Native failures render matching
`error[E007]` / `error[E008]` text with character and byte labels in
`PyRuntimeError`; they do not raise the Python `HotpatchError` class. Details and
build/test commands are in `docs/quantum_hotpatch.md`.

## Doc sync — `if let` expression-position limit

> **Note (superseded):** This entry records an earlier README wording
> fix. The subsequent "Parser/translator/codegen — `if let` as a tail
> expression" entry changed the behavior: `if let` is now supported as a
> tail expression in both backends, not just documented as such.

**Symptom:** README said "`if let` is supported as a statement, not as an
expression", but `if let` works as the tail expression of `main()` in both
backends. The GUIDE.md example uses exactly this form.

**Cause:** The parser supports `if let` only at the final expression
position of a block. It rejects `if let` as a sub-expression inside a
`let` binding or a function argument. README's wording conflated these two
"expression positions".

**Fix:** README now says "supported as a statement and as the final
expression of a block, but not as a sub-expression inside a `let` binding
or a function argument". GUIDE adds a three-line note after the example.

**Lesson:** "Expression position" is not a single concept. Parser support
must be described by concrete syntactic contexts (statement / tail
expression / sub-expression), not by a statement-vs-expression dichotomy.

## Priority features — one lowering path, honest boundaries

**Symptom:** The spec, README, and GUIDE still said `vec![...]` and
`String::from(...)` were parsed but not lowered, `if let` worked only at a
block tail, the entry API lacked `or_default()`, `#[terminates]` reported the
function header rather than the offending recursive call, and the CLI/LSP
had no fix hints or diagnostic publication.

**Cause:** The front end and the two backends had accumulated separate,
partially overlapping paths for the same syntax. `vec!` was left as a token
stream; `String::from` produced a doubled `String::from(String::from(...))`
in generated Rust; `if let` lowering only handled a `Some` pattern at a tail
position; the `std:fs` host seam had no shared arity table; and the
diagnostic payload was built independently by each tool.

**Fix:** `vec![...]` now lowers in the parser into the existing `ArrayExpr`
and `RepeatArray` nodes so both backends share one path. `String::from`
lowers to a single `String::from` in Rust and the equivalent Python call.
`Option`/`Result::unwrap_or_else` are lazy in both backends. `std:fs` is a
pure `.poly` wrapper over a trusted-only host seam with a shared `E003`
arity check and span. Expression-position `if let` uses the shared pattern
machinery for `Some`/`None`/`Ok`/`Err`. Entry `or_default()` and a slot
handling fix for `or_insert_with_key` were added. The termination checker
accumulates lower-bound guards per call site, resets them inside closures
and nested functions, and reports `E002` on the offending call. `FixHint`
and span/range helpers are shared by the CLI JSON and LSP
`publishDiagnostics`. SPEC 5.4, 7.2.x, 8.11, 9.2.3, 10.x, and 11.21 plus
README, GUIDE, and CHANGELOG were updated together.

**Lesson:** When a feature is implemented in a shared front-end node, the
docs should describe that one path rather than backend-specific workarounds.
Where a real difference remains — native `Err` payloads are `PyErr`, LSP
positions are character-based, only local fix edits are safe — record it as
an explicit limitation instead of implying parity.

## 100 framework functions — registry over scattered entry points

**Symptom:** The 100-function request numbered Frame 8 as 71–80 but listed
only 72–80, and spelled function 4 `parse_ Pratt_expr_with_precedence` with an
embedded space. There was also no single place to discover or invoke the
requested functions, and each subsystem had its own entry-point conventions.

**Cause:** The existing `runtime.capabilities` registry covers a different
100-function pillar directive, so it could not be reused for the compiler,
IR, backend, diagnostics, CI, and documentation functions in this request.
Without a canonical mapping, callers would depend on module paths instead of
stable numbers and names.

**Fix:** Added `framework_functions` with a 1–100 registry, frame titles,
`get_function`, `invoke`, `iter_functions`, and `validate_registry`. Each frame
lives in its own module (`compiler/frontend`, `compiler/types`, `compiler/ir`,
`backends/python`, `backends/rust`, `runtime/hotpatch`, `runtime/concurrency`,
`diagnostics`, `tools/build_ci`, `docs`, `tests/regression_harness`) with
dedicated tests. Function 4 was normalized to `parse_pratt_expr_with_precedence`;
slot 71 was filled with a real `diag_create_diagnostic_context` constructor so
all ten frames have ten functions without renaming any supplied entry.

**Lesson:** A numeric contract is only useful if it is validated. The registry
test asserts contiguous numbering, unique names, and importable callables, so
a missing or duplicated implementation fails as a normal test rather than as
an integration surprise. Platform-sensitive functions (GIL, MAP_JIT, PEP 667,
raw-address TOCTOU) return structured models and report their boundaries
instead of claiming privileged behaviour.

## Hard frames and three-way parity — specialization without silent drift

**Symptom:** WASM supported only scalar `i64`/`bool` programs, so nine of the
twelve deterministic parity cases were honest skips. Diagnostic wording also
risked drifting between backends, and scalar locals were briefly substituted
with their initializer constants during aggregate lowering, which broke loops.

**Cause:** The WAT emitter is a scalar stack machine, while the parity corpus
uses structs, recursive enums, nested vectors, bounded generics and shared
diagnostics. Lowering aggregates at runtime would have required a second linear
memory compiler; lowering every scalar binding as a constant would have erased
mutation. Diagnostics needed to reuse the existing typed error classes rather
than be reworded in the WASM backend.

**Fix:** `wasm_preflight_validate` runs the shared typed checks before lowering;
`lower_wasm_aggregates` monomorphizes generics and specializes only compile-time
aggregate shapes, keeping scalar bindings as real WASM locals. Genuine dynamic
aggregate ABIs raise structured `E003` and become explicit boundary skips. The
parity harness now runs all 12 cases on all three backends for seeds `1`, `42`,
`9999`, and `2026` with zero skips; a synthetic unsupported case still verifies
the skip path. The `hard-frames-parity` gate stage wires the frame suites and
per-seed checks with an injectable runner, so unit tests never compile native
code.

**Lesson:** Deterministic parity is strongest when boundary narrowing is
explicit and shared diagnostics are reused. Compile-time specialization can
widen a backend's surface without duplicating its runtime, but scalar mutation
must stay in the emitted program.

## Expansion batch — close the loop, then widen it

**Symptom:** The first verification layer proved the tools worked but left
honest gaps: one diagnostic template was skipped natively, mismatch artifacts
were full generated programs rather than minimal reproducers, hot-patch
telemetry depended on callers remembering to emit events, the gate did not cover
formatting/docs/metadata/links/benchmarks, and distribution and third-party
surfaces were only described in conversation.

**Cause:** Each gap sat at a different seam. Native E003 needed a diagnostic
both backends already share, so the unknown-macro template was replaced with a
host-call arity error. Shrinking needed a bounded compare function under the
existing artifact writer. Hot-patch events needed a dependency-free bus in
`runtime/hotpatch` so `api.telemetry` could subscribe without an import cycle.
Gate breadth needed the same injected-runner pattern as the original stages.
CI and native packaging needed configuration that stays honest about what is
executed locally versus on GitHub Actions.

**Fix:** Expanded the fuzzer corpus and added bounded shrinking; subscribed
telemetry to real RCU/journal/frame events; added baseline save/compare and two
new counters; appended five gate stages and optional parallel seeds; added
release-gate and maturin wheel workflows with SBOM and release-only signing;
and documented the third-party audit scope. All new state stays in temp/cache
roots or the gitignored `.artifacts/` directory.

**Lesson:** A verification system matures when it stops relying on manual
discipline: events are captured at the source, reproducers are minimized, gates
cover the whole release surface, and external scope decisions are written down
rather than implied.

## Verification, telemetry, and release readiness — isolated by construction

**Symptom:** Comparing the two backends by hand only found mismatches that a
developer happened to run, and "all checks pass" meant remembering eight
separate commands. Packaging looked healthy in the source tree but the built
wheel omitted several modules that the 100-function registry imports, and
telemetry data had no place to surface in the CLI or visualizer.

**Cause:** There was no deterministic program generator with a stable seed, no
single ordered gate that reused the FRAME 9 verifiers, and the setuptools
package list predated the new framework packages. `tests/` is intentionally
absent from wheels, but the registry pointed functions 94–100 at it, so a fresh
install could not resolve all 100 callables. Telemetry helpers produced data
only when a caller happened to ask for it.

**Fix:** Added a template-based fuzzer whose per-case RNG derives from
`(seed, index)`, a gate that runs checks in a documented order and aggregates
them with `ci_generate_release_gate_report`, and a session telemetry collector
wired into the CLI and visualizer. Moved FRAME 10 harness implementations to
the installable `regression.harness` package and kept the old test import as a
re-export. Added `diagnostics`, `docs`, `framework_functions`, and
`tools.build_ci` to the wheel/sdist package list, plus Cargo package metadata
and cache-isolation tests.

**Lesson:** A verification story is only as strong as its isolation. Fuzzer
artifacts live under the gitignored `.artifacts/` directory and are written
only on mismatch; cargo and Python caches get scope-derived directories outside
the workspace; telemetry payloads are redacted before serialization. Privileged
mechanisms remain labelled as models or refusals — MAP_JIT owns pages only,
PEP 667 covers future entries, and raw-address checks do not remove TOCTOU.

## N1 complete

Support for free functions, primitive types, vectors, structs with `impl`, and conditional expressions and statements was added in N1.

### Bug 1

**Symptom:** Native code generation failed for `format!` calls whose first argument was not emitted as a bare string literal, such as a `String::from(...)` expression.

**Cause:** The format-string handling path expected a literal and did not distinguish that required form from other expressions.

**Fix:** The `_format_str` helper was added to handle the supported literal format-string form explicitly.

**Lesson:** Special syntax should be validated and handled at the boundary where its constraints are known.

### Bug 2

**Symptom:** An expression such as `xs { total = ... }` could be parsed as a struct literal.

**Cause:** `_looks_like_struct_literal` inspected too few tokens after the path and treated a following block as a struct literal without confirming its contents.

**Fix:** Struct-literal recognition now requires the next token to be `:`, `,`, `}`, `..`, or empty.

**Lesson:** Ambiguous syntax should be disambiguated using the minimum structural evidence required by the grammar.

### Status after N1

- Free functions and `i64`, `f64`, `bool`, and `String` support completed.
- `Vec<T>`, structs with `impl`, and `if`/`else` support completed.
- **13 tests passed.**

## N2a complete

Native support for `while`, `loop`, and `match` was added.

### Bug 1

**Symptom:** Parameters intended to be mutable inside generated Rust functions were not mutable.

**Cause:** Rust parameters are immutable by default, while the generated function signature did not account for the parameter pattern's mutability.

**Fix:** `param_to_rust` now emits `mut` unless the pattern is already `mut`.

**Lesson:** Generated code must preserve the source language's mutability intent while also satisfying the target language's defaults.

### Bug 2

**Symptom:** Vector length and indexing expressions could use incompatible integer types in generated Rust.

**Cause:** `Vec::len()` returns `usize`, while the generated expression used `i64` for the length and index.

**Fix:** Generated length expressions use `(recv.len() as i64)`, and generated indexes use `recv[(idx) as usize]`.

**Lesson:** Target-language type differences should be handled explicitly at code-generation boundaries.

### Bug 3

**Symptom:** Block-bodied `match` arms without a trailing comma were rejected.

**Cause:** The parser required a comma after every arm, although Rust permits omitting it after a block body.

**Fix:** `parse_match` now accepts a missing comma when an arm has a block body.

**Lesson:** Parsers should model grammar variations according to syntactic context rather than applying one separator rule universally.

### Bug 4

**Symptom:** Parsing a `match` expression did not consume its opening brace.

**Cause:** The opening brace was not explicitly consumed by `parse_match`.

**Fix:** `self.expect(TokenType.LBRACE)` was added at the start of match-body parsing.

**Lesson:** Delimiters should be consumed and validated at the grammar production that owns them.

### Bug 5

**Symptom:** A block-bodied `match` arm could not be used where an expression was expected.

**Cause:** The generated arm body was not represented as a Rust block expression.

**Fix:** Block-bodied arms are now wrapped in `{ ... }`.

**Lesson:** Code generation should preserve the syntactic category required by the surrounding expression.

### Status after N2a

- Native `while`, `loop`, and `match` support completed.
- Mutability, integer casts, match separators, delimiters, and block-expression handling corrected.
- **22 tests passed.**

## N2b complete

Native vector methods `push`, `pop`, and `len`, along with indexing, were added.

### Bug 1

**Symptom:** Generated Rust bindings could trigger unused-mutability warnings for bindings that were not reassigned.

**Cause:** Mutability requirements were not inferred per binding, while mutable bindings were needed for supported assignment behavior.

**Fix:** `generate_stmt` emits `let mut` for every binding, and the generated crate includes `#![allow(unused_mut, ...)]`.

**Lesson:** When exact analysis is not available, a consistent conservative code-generation policy can be used if the resulting diagnostics are explicitly controlled.

### Bug 2

**Symptom:** Vector `pop` results and `Some(x)` / `None` patterns needed special handling in generated native code.

**Cause:** `Vec::pop()` returns `Option<T>`, and pattern emission could otherwise transform those variant patterns incorrectly.

**Fix:** PyO3's existing `Option<T>` support is used, and `pattern_to_rust` emits `Some(x)` and `None` verbatim.

**Lesson:** Existing target-language and binding-library support should be reused instead of duplicating representation logic.

### Status after N2b

- Native `Vec` methods `push`, `pop`, and `len`, plus indexing, completed.
- Mutable bindings and `Option<T>` pattern handling corrected.
- **30 tests passed.**

## N2d complete

Native test compilation was consolidated, build caching was improved, and Rust compiler errors were given a structured representation. All native test sources are collected from `SOURCES` dictionaries in `tests/test_native_*.py`, merged, and compiled once per pytest session. `conftest.py` invokes `compile_sources(sources)` from `pytest_collection_modifyitems`; `tools/compile_native.py` uses the same flow for CI pre-builds.

A SHA-256 digest of the merged source is used as the cache key. The shared cache path is `~/.cache/polyglot-native/<hash>/`. The existing `target/` directory is retained for incremental builds; only `Cargo.toml`, `.cargo/config.toml`, and `src/lib.rs` are rewritten. Native test build time was reduced from approximately 214 seconds to 0.15 seconds on a cache hit.

### Bug 1

**Symptom:** Loading the built macOS dynamic library under a filename-derived module name caused Python to look for `PyInit_libpolyglot_native` rather than the generated `PyInit_polyglot_native` symbol.

**Cause:** The dynamic-library filename prefix and the PyO3 module initialization name differ.

**Fix:** The library is loaded with `ExtensionFileLoader(mod_name="polyglot_native")`.

**Lesson:** Extension loaders and generated module symbols must agree on one canonical module name, independent of platform-specific filenames.

### Bug 2

**Symptom:** Classes created by the Python backend could fail during `@dataclass` processing because their module could not be resolved.

**Cause:** The generated namespace's module name had not been registered in `sys.modules` before dataclass processing.

**Fix:** `Translator.compile_rust` registers the namespace as `sys.modules["polyglot"]`.

**Lesson:** Dynamically generated objects may depend on normal module-registration conventions used by runtime libraries.

### Bug 3

**Symptom:** Native compiler failures were difficult to connect to structured diagnostics.

**Cause:** Cargo's standard error was exposed without parsing its available source location and error-code fields.

**Fix:** `RustCompileError` parses `error_line`, `error_col`, and `error_code`, and provides `pretty()`.

**Lesson:** Errors from external toolchains should be normalized at the integration boundary so callers can consume consistent diagnostics.

### Bug 4

**Symptom:** Native compilation could be invoked without an explicit trust decision.

**Cause:** Backend selection did not enforce the project's default-deny trust policy.

**Fix:** `UntrustedCompileError` and `Backend._check_trust()` enforce the trust gate. `get_backend(name)` reads `POLYGLOT_TRUSTED`, which defaults to `"0"`; `trusted=False` is the default, and tests opt in with `trusted=True`.

**Lesson:** Risky execution paths should require explicit opt-in, while keeping the scope of that gate clearly documented.

### Bug 5

**Symptom:** Methods without a `self` parameter, such as `fn new(...) -> Point`, could not be called as associated constructors through the generated Python class.

**Cause:** Methods without a receiver were not emitted as static methods, and `Path` expressions were not translated into attribute chains.

**Fix:** Methods without `self` are emitted as `@staticmethod`; `Point::new(x, y)` is translated to `Point.new(x, y)` through `ast.Attribute` chains in `translate_expr`.

**Lesson:** Cross-language APIs should preserve receiver and qualification semantics explicitly in both code generation and expression translation.

## N2c batch 1

The native code generator already emits `#[pyclass]` and a module registration call for every struct declaration. A native integration test now defines two independent structs with constructors and verifies that both classes can be instantiated from Python. No code-generation change was needed.

- Covered by `tests/test_native_features.py::test_native_register_multiple_pyclasses`.
- `pytest -q`: 134 passed, 8 skipped.

## N2c batch 2

A shared stable dependency ordering now emits a referenced inner struct before its containing struct in both backends. Native and Python tests declare `Outer` before `Inner` and verify that an `Outer` instance contains a usable `Inner` instance.

- Covered by `tests/test_native_features.py::test_native_struct_field_of_struct` and `tests/test_point.py::test_python_struct_field_of_later_struct`.
- `pytest -q`: 136 passed, 8 skipped.

## N2c batch 3

Simple enums containing only unit variants are now emitted as PyO3 classes and registered with the module. Python backend variants retain the standard `Enum` representation, accessed as `Choice.A` and `Choice.B`; native variants are class attributes on the PyO3 enum class. Payload enums remain unchanged.

- Covered by `tests/test_native_features.py::test_native_unit_enum_variants` and `tests/test_point.py::test_python_unit_enum_variant_representation`.
- `pytest -q`: 138 passed, 8 skipped.

## N2c batch 4

The Python backend now represents payload enum variants as callable class attributes that produce tagged tuples, allowing `Maybe::Some(value)` and payload-pattern matching while preserving unit-variant matching. PyO3 0.26 safely exposes only the unit-enum form used here; payload enums stay internal to generated Rust and are not exported. The native exposure test is explicitly skipped with that limitation.

- Covered by `tests/test_point.py::test_python_payload_enum_representation`.
- Native limitation: `tests/test_native_features.py::test_native_payload_enum_is_exposed`.
- `pytest -q`: 139 passed, 9 skipped.

## N2c batch 5

`impl` blocks on unit enums now contribute methods to the Python enum class and are emitted as `#[pymethods] impl` blocks for native PyO3 enums. The parity tests call the method through variants in both backends. Payload enum methods remain Python-only because payload enums are not exposed as native classes.

- Covered by `tests/test_native_features.py::test_native_unit_enum_variants` and `tests/test_point.py::test_python_unit_enum_variant_representation`.
- `pytest -q`: 139 passed, 9 skipped.

## N2c batch 6

Structs with `Vec<i64>` fields are constructible from Python in both backends. PyO3 converts Python lists at the constructor boundary, and the exposed field reads back as a Python list.

- Covered by `tests/test_native_features.py::test_native_struct_with_vec_field` and `tests/test_point.py::test_python_struct_with_vec_field`.
- `pytest -q`: 141 passed, 9 skipped.

## N2c batch 7

Struct methods returning `Self` are supported by returning a cloned pyclass instance in native Rust and the cloned dataclass value in Python. Both tests verify that the returned object has equivalent fields but is not the original instance.

- Covered by `tests/test_native_features.py::test_native_struct_method_returns_self` and `tests/test_point.py::test_point_method_returns_self`.
- Python translation uses shallow `copy.copy` for Rust's `.clone()` method.
- `pytest -q`: 143 passed, 9 skipped.

## N2c batch 8

Generic struct declarations are erased to a concrete native pyclass, with each declared type parameter mapped to `i64`; the Python backend maps the corresponding field and method annotations to `int`. This is a deliberate native limitation: generic arguments are not runtime-specialized and all type parameters default to `i64`.

- Covered by `tests/test_native_features.py::test_native_generic_struct_defaults_to_i64` and `tests/test_point.py::test_python_generic_struct_defaults_type_parameter_to_int`.
- `pytest -q`: 145 passed, 9 skipped.

## N2c batch 9

The native code generator groups `impl` blocks by target and emits one Rust `impl` and one `#[pymethods]` block per class, preventing duplicate PyO3 registration scaffolding. The Python translator already combines methods from all impl blocks into the class body. Both backends now test methods defined in separate blocks.

- Covered by `tests/test_native_features.py::test_native_multiple_impl_blocks` and `tests/test_point.py::test_point_methods_across_impl_blocks`.
- `pytest -q`: 147 passed, 9 skipped.

## N2c batch 10

A combined regression program declares a struct, unit enum, and `impl` blocks for both. The native backend registers both classes and their methods; the Python backend provides matching behavior.

- Covered by `tests/test_native_features.py::test_native_struct_enum_impl_regression` and `tests/test_point.py::test_python_struct_enum_impl_program_regression`.
- `pytest -q`: 149 passed, 9 skipped.

## N2d batch 11

`RustCompileError` is now a dataclass carrying message, Rust and source locations, raw stderr, generated source, and error code. Cargo output parsing extracts `error[...]` and `--> src/lib.rs:LINE:COL`, while compatibility properties preserve `error_line`, `error_col`, and `rust_message`.

- Added direct parser tests in `tests/test_rust_compile_error.py`.
- `pytest -q`: 151 passed, 9 skipped.

## N2d batch 12

`RustCodegen` now exposes a one-based `line_map` from generated Rust line numbers to original `(line, col)` positions. `set_origin` records the current AST node, and `emit` associates generated lines with that origin. Item, struct, enum, impl, function, and statement generation establish origins.

- Covered by `tests/test_rust_compile_error.py::test_rust_codegen_records_source_origin_for_emitted_lines`.
- `pytest -q`: 152 passed, 9 skipped.

## N2d batch 13

The `RustCodegen` instance is retained through compilation, and its line map is passed into the Cargo build error parser. Rust error locations now populate `source_line` and `source_col` when an origin is available; linker errors and unmapped generated lines retain `None`.

- Covered by `tests/test_rust_compile_error.py::test_cargo_error_maps_rust_line_to_source_origin`.
- `pytest -q`: 153 passed, 9 skipped.

## N2d batch 14

Source line text is now retained from lexer tokens through `Program` and code generation. The structured compiler error receives the original line text and `pretty()` prints both the original Polyglot line/location and the offending generated Rust context. Token and Program additions are optional, preserving existing construction call sites.

- Covered by `tests/test_rust_compile_error.py::test_pretty_includes_original_and_generated_lines`.
- `pytest -q`: 154 passed, 9 skipped.

## N2d batch 15

Four native compiler errors are now exercised end to end: a return-type mismatch, assignment to an immutable local, a non-exhaustive match, and an unresolved constant pattern. The code generator now preserves local binding mutability instead of marking every binding mutable. Existing Vec mutation fixtures were updated to use `let mut`. Cargo diagnostic location parsing now selects the location following the first error header rather than a preceding warning.

- Covered by the parametrized `tests/test_rust_compile_error.py::test_native_errors_map_back_to_polyglot_source`.
- Unresolved-constant diagnostics in inline match expressions currently map to the enclosing match line.
- `pytest -q`: 158 passed, 9 skipped.

## N2d batch 16

`RustCompileError` now has a warnings list populated from Cargo warning headers when an error is also present. The build path continues to raise only when Cargo exits unsuccessfully; a warning-only successful result is explicitly tested as non-exceptional.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_captures_warnings_without_replacing_error` and `test_warning_only_cargo_result_does_not_raise`.
- `pytest -q`: 160 passed, 9 skipped.

## N2d batch 17

Only the first Cargo error is promoted into the structured `RustCompileError` fields. The complete stderr remains in `raw_stderr`, retaining additional distinct diagnostics for callers and reports.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_keeps_later_errors_in_raw_stderr`.
- `pytest -q`: 161 passed, 9 skipped.

## N2d batch 18

Linker-style errors without a `src/lib.rs` location remain valid structured errors with message and raw stderr; source and Rust line/column fields stay unset rather than fabricating a location.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_without_source_location`.
- `pytest -q`: 161 passed, 9 skipped.

## N2d batch 19

Identical Cargo error blocks are removed from retained raw stderr while distinct later errors remain available. The first error continues to populate the primary structured fields.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_deduplicates_identical_diagnostics`.
- `pytest -q`: 162 passed, 9 skipped.

## N2d batch 20

The full existing suite was rerun against the structured error type. `RustCompileError` remains a `RuntimeError` subclass, and compatibility aliases for `error_line`, `error_col`, and `rust_message` are verified.

- Covered by `tests/test_rust_compile_error.py::test_rust_compile_error_preserves_runtime_error_compatibility`.
- `pytest -q`: 163 passed, 9 skipped.

### Status after N2d

- Native test sources are merged and compiled once per pytest session, with the same build flow available for CI pre-builds.
- Content-addressed caching retains Cargo's `target/` artifacts for incremental builds.
- Structured Rust compiler errors and readable formatting are available.
- Dynamic-library loading, dataclass module registration, trust checking, and associated-method translation corrected.
- **30 tests passed; cache-hit build time was approximately 0.15 seconds, down from approximately 214 seconds.**

## N3 batch 21

Iteration over `Vec<String>` and `Vec<(i64, i64)>` is verified in both backends. Python `for` translation now supports tuple destructuring patterns in addition to identifier targets, matching Rust's loop binding behavior for pair elements.

- Covered by `tests/test_native_n3.py::test_native_iter_vec_string`, `test_native_iter_vec_tuple`, and `test_python_iter_vec_parity`.
- `pytest -q`: 167 passed, 9 skipped.

## N3 batch 22

`Vec::contains` now emits Rust's reference-taking `contains(&value)` call. The Python backend lowers the same method to a membership test, with native and Python cases covering present and absent values.

- Covered by `tests/test_native_n3.py::test_native_vec_contains` and `test_python_vec_contains_parity`.
- `pytest -q`: 169 passed, 9 skipped.

## N3 batch 23

`Vec::sort` now works through the existing method path. `sort_by` supports both native Rust Ordering closures and the requested numeric-subtraction shorthand, which is lowered to `.cmp()` for Rust and `functools.cmp_to_key` for Python.

- Covered by `tests/test_native_n3.py::test_native_vec_sort_and_sort_by` and `test_python_vec_sort_parity`.
- Closure parsing now treats unparenthesized closure parameters as pattern atoms so the closing pipe is not consumed as an or-pattern operator.
- `pytest -q`: 172 passed, 9 skipped.

## N3 batch 24

The `Vec.map` shorthand is supported in both backends. Native code generation consumes the vector, maps the closure, and collects a `Vec`; Python translation emits a list comprehension. The supported closure shape is one identifier parameter.

- Covered by `tests/test_native_n3.py::test_native_vec_map` and `test_python_vec_map_parity`.
- `pytest -q`: 174 passed, 9 skipped.

## N3 batch 25

The `Vec.filter` shorthand is supported in both backends. Native code uses `into_iter().filter(...).collect::<Vec<_>>()`; Python emits a filtering list comprehension. One-parameter identifier closures are supported.

- Covered by `tests/test_native_n3.py::test_native_vec_filter` and `test_python_vec_filter_parity`.
- `pytest -q`: 176 passed, 9 skipped.

## N3 batch 26

The `Vec.fold(initial, closure)` shorthand is supported in both backends. Native code lowers it to `into_iter().fold(...)`, and Python calls `functools.reduce` with the same initial value and accumulator argument order.

- Covered by `tests/test_native_n3.py::test_native_vec_fold` and `test_python_vec_fold_parity`.
- `pytest -q`: 178 passed, 9 skipped.

## N3 batch 27

Vector range indexing is supported as `&values[start..end]`. Native code returns an owned vector copy to cross the Python extension boundary; Python uses a list slice. The returned values match the requested half-open slice.

- Covered by `tests/test_native_n3.py::test_native_vec_slice` and `test_python_vec_slice_parity`.
- `pytest -q`: 180 passed, 9 skipped.

## N3 batch 28

Vector concatenation is supported through `extend` in both backends. Native code uses `Vec::extend`, and Python calls `list.extend`; direct `Vec + Vec` remains unsupported because Rust does not define that operator for vectors.

- Covered by `tests/test_native_n3.py::test_native_vec_extend` and `test_python_vec_extend_parity`.
- `pytest -q`: 182 passed, 9 skipped.

## N3 batch 29

`Vec.insert`, `remove`, `retain`, and `dedup` are supported in both backends. Native index arguments are cast to `usize`; Python `remove` is translated to `pop(index)`, and runtime helpers preserve in-place `retain` and adjacent-only `dedup` semantics.

- Covered by `tests/test_native_n3.py::test_native_vec_insert_remove_retain_dedup` and `test_python_vec_mutation_parity`.
- `pytest -q`: 187 passed, 9 skipped.

## N3 batch 30

`Vec.windows(size)` and `chunks(size)` now return `Vec<Vec<i64>>` natively and lists of lists in Python. Native slices are copied into owned vectors for PyO3 conversion. Python raises `ValueError` for nonpositive sizes, matching the native operation's invalid-size failure rather than returning misleading empty output.

- Covered by `tests/test_native_n3.py::test_native_vec_windows_and_chunks` and `test_python_vec_windows_and_chunks_parity`.
- `pytest -q`: 190 passed, 9 skipped.

## N4 batch 31

String byte length, Unicode character iteration, and UTF-8 byte iteration are supported. Python's `.len()` lowering now counts UTF-8 bytes for strings while preserving normal `len()` behavior for vectors and other containers.

- Covered by `tests/test_native_n4.py::test_native_string_len_chars_bytes` and `test_python_string_len_chars_bytes_parity`.
- Python `.bytes()` matches PyO3's `Vec<u8>` conversion by returning a `bytes` object.
- `pytest -q`: 194 passed, 9 skipped.

## N4 batch 32

String splitting returns `Vec<String>`/Python lists, and vector joining uses a string separator and returns a string in both backends.

- Covered by `tests/test_native_n4.py::test_native_string_split_join` and `test_python_string_split_join_parity`.
- `pytest -q`: 197 passed, 9 skipped.

## N4 batch 33

String replacement, trimming, prefix checks, and suffix checks are available across both backends. Native string references are passed to Rust's `&str` APIs, and Rust's borrowed trim result is converted to an owned `String` for the Python boundary.

- Covered by `tests/test_native_n4.py::test_native_string_replace_trim_prefix_suffix` and `test_python_string_replace_trim_prefix_suffix_parity`.
- `pytest -q`: 202 passed, 9 skipped.

## N4 batch 34

Unicode-aware uppercase and lowercase conversions are supported. Native `to_uppercase`/`to_lowercase` and Python `upper`/`lower` are tested with a sharp-s case whose uppercase expands to two characters.

- Covered by `tests/test_native_n4.py::test_native_string_case_conversion` and `test_python_string_case_conversion_parity`.
- `pytest -q`: 205 passed, 9 skipped.

## N4 batch 35

The parser now accepts turbofish type arguments on method calls without changing existing `MethodCall` construction. `.parse::<i64>()` lowers to Rust's typed parse with `unwrap()` and Python's `int()`. Invalid-input error behavior is backend-specific; successful parsing is verified.

- Covered by `tests/test_native_n4.py::test_native_string_parse_i64` and `test_python_string_parse_i64_parity`.
- `pytest -q`: 207 passed, 9 skipped.

## N4 batch 36

`format!` with multiple positional arguments is verified in both backends. Named format arguments are not added in this batch because the current macro parser represents them as assignment expressions rather than format-specific arguments.

- Covered by `tests/test_native_n4.py::test_native_format_multiple_arguments` and `test_python_format_multiple_arguments_parity`.
- `pytest -q`: 209 passed, 9 skipped.

## N4 batch 37

`write!` into a mutable `String` is supported for identifier targets and positional formatting arguments. Native generation brings `std::fmt::Write` into scope and unwraps the infallible String write result; Python translates the macro statement into string concatenation.

- Covered by `tests/test_native_n4.py::test_native_write_macro_into_string` and `test_python_write_macro_into_string_parity`.
- A generated `&mut` reference originally lacked required spacing; native write tests now cover the corrected syntax.
- `pytest -q`: 211 passed, 9 skipped.

## N4 batch 38

String `+` concatenation is recognized by native code generation for expressions with known string parameters, string literals, or string-returning methods, and is emitted with `format!` to avoid Rust's `String + &str` ownership requirements. Python retains native string `+`.

- Covered by `tests/test_native_n4.py::test_native_string_concatenation` and `test_python_string_concatenation_parity`.
- `pytest -q`: 213 passed, 9 skipped.

## N4 batch 39

String range slicing uses UTF-8 byte offsets in both backends. Native code returns an owned `String`; Python slices encoded bytes and decodes the result, so invalid UTF-8 boundaries raise instead of silently changing indexing semantics.

- Covered by `tests/test_native_n4.py::test_native_string_slice_uses_utf8_byte_offsets` and `test_python_string_slice_uses_utf8_byte_offsets`.
- `pytest -q`: 215 passed, 9 skipped.

## N4 batch 40

String equality and ordering use Rust's native `String` comparison and Python's lexicographic string comparison. Both backends verify less-than behavior for ASCII strings.

- Covered by `tests/test_native_n4.py::test_native_string_comparison_and_ordering` and `test_python_string_comparison_and_ordering_parity`.
- Validation is recorded after the batch test run.

## Security

`trusted=False` is not a sandbox; it is a speed bump. Operating-system-level isolation, such as a container, seccomp, and disabled network access, is required for untrusted input. The native cache directory is created with mode `0o700`. `pyo3 = "0.26"` is pinned in `Cargo.toml`.

Three isolation levels are provided through `api/`:

- **A:** In-process queue.
- **B:** Subprocess.
- **C:** Docker with `--network=none --read-only`.

HTTP requests are handled with isolation level B by default. An `X-Trusted-Token` must be provided when `trusted=true` is set.

## N2a pattern expansion — batch 1

The existing pattern AST nodes were wired through parsing, native Rust code generation, and Python match translation. Batch 1 covers twenty direct match-pattern forms:

1. Or-patterns.
2. Exclusive literal ranges.
3. Inclusive literal ranges.
4. Tuple patterns.
5. Nested tuple patterns.
6. Unit tuple patterns.
7. Single-element tuple patterns.
8. Struct patterns with explicit fields.
9. Struct patterns with shorthand fields.
10. Struct patterns with a rest pattern.
11. `@` binding patterns.
12. Ranges without a lower bound.
13. Ranges without an upper bound.
14. Negative literal ranges.
15. Or-patterns containing ranges.
16. Tuple patterns with identifier bindings.
17. Nested tuple binding patterns.
18. Struct field patterns with nested bindings.
19. Reference patterns with literal subpatterns.
20. Reference patterns with identifier bindings.

Single-element tuple types now accept a trailing comma and are emitted with the required Rust tuple comma. Negative literal bounds are emitted without parentheses, as required by Rust range-pattern syntax. Python match translation now handles tuple and struct tests, range and or-pattern tests, references, bindings, and bound-variable assignments.

The native test cases are stored in `tests/test_native_n2a_patterns.py` under `SOURCES` and compiled together with the existing native test sources.

### Tests in batch 1

- Feature 1: `test_pattern_or`
- Feature 2: `test_pattern_range_exclusive`
- Feature 3: `test_pattern_range_inclusive`
- Feature 4: `test_pattern_tuple`
- Feature 5: `test_pattern_nested_tuple`
- Feature 6: `test_pattern_unit_tuple`
- Feature 7: `test_pattern_single_tuple`
- Feature 8: `test_pattern_struct_explicit`
- Feature 9: `test_pattern_struct_shorthand`
- Feature 10: `test_pattern_struct_rest`
- Feature 11: `test_pattern_binding`
- Feature 12: `test_pattern_range_open_start`
- Feature 13: `test_pattern_range_open_end`
- Feature 14: `test_pattern_range_negative`
- Feature 15: `test_pattern_or_range`
- Feature 16: `test_pattern_tuple_binding`
- Feature 17: `test_pattern_nested_binding`
- Feature 18: `test_pattern_struct_nested`
- Feature 19: `test_pattern_ref_literal`
- Feature 20: `test_pattern_ref_binding`
- Cross-backend parity: `test_python_backend_pattern_parity`

## N2a pattern batch 2

`PatSlice` and `PatRest` were added to the AST and wired through the parser, Rust code generator, and Python match translator. Slice patterns support fixed-length matching, a single rest marker in any position, and `name @ ..` bindings. Native matching converts a vector scrutinee to a slice, while Python matching validates the sequence type and length before testing elements. A single-element parenthesized pattern is now treated as grouping unless a trailing comma makes it a tuple.

Native support is validated for direct vector slices, slices nested in tuples, range/ref/or patterns within slices, guards, and rest bindings. Struct-field and nested-slice combinations are covered by Python-backend tests but remain unsupported by native codegen where they would require matching a `Vec` as a slice within a Rust pattern. A tuple-style enum payload slice is skipped until payload-carrying enum values are represented by the Python runtime.

### Tests in batch 2

- Feature 21: `test_pattern_slice_fixed`
- Feature 22: `test_pattern_slice_rest_middle`
- Feature 23: `test_pattern_slice_rest_end`
- Feature 24: `test_pattern_slice_rest_start`
- Feature 25: `test_pattern_slice_all_rest`
- Feature 26: `test_pattern_slice_empty`
- Feature 27: `test_pattern_slice_literals`
- Feature 28: `test_pattern_slice_bindings`
- Feature 29: `test_pattern_slice_tuple`
- Feature 30: `test_pattern_slice_or`
- Feature 31: `test_pattern_slice_range`
- Feature 32: `test_pattern_slice_ref`
- Feature 33: `test_pattern_slice_at`
- Feature 34: `test_pattern_slice_guard`
- Feature 35: `test_pattern_slice_tuple_context`
- Feature 36: `test_pattern_slice_struct_context`
- Feature 37: `test_pattern_slice_enum_context` (skipped)
- Feature 38: `test_pattern_slice_rest_binding`
- Feature 39: `test_pattern_slice_only_rest_binding`
- Feature 40: `test_pattern_slice_nested`
- Cross-backend parity: `test_python_backend_pattern_parity`

Batch 2 validation: `pytest -v` completed with 70 passed and 1 skipped in 11.63 seconds; `ruff check --fix .` passed.

## N2a pattern batch 3

`PatBox`, `PatConst`, `PatMacro`, and `PatPath` were added to the pattern AST and connected to parser, native code generation, and Python translation. Box patterns are treated as transparent patterns by Python; native box-pattern tests are skipped because stable Rust does not support this syntax. Explicit `const NAME` patterns lower to the corresponding Rust constant path. Qualified paths are distinguished from identifier bindings, and Python path patterns compare against the resolved attribute chain. Macro patterns retain their delimiter and raw token body for code generation; Python raises `TranslateError` for unsupported pattern macros, while unresolved native macros are rejected by Cargo. Standalone `..` patterns are rejected by the parser with a source location.

### Tests in batch 3

- Feature 41: `test_pattern_box_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 42: `test_pattern_box_literal_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 43: `test_pattern_box_tuple_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 44: `test_pattern_box_nested_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 45: `test_pattern_box_binding_native` (skipped for native; exercised by `test_python_box_pattern_parity`)
- Feature 46: `test_pattern_const_bare`
- Feature 47: `test_pattern_const_path`
- Feature 48: `test_pattern_const_qualified`
- Feature 49: `test_pattern_const_binding`
- Feature 50: `test_pattern_const_tuple`
- Feature 51: `test_pattern_macro_parses[m!()]`
- Feature 52: `test_pattern_macro_parses[m!(1, 2)]`
- Feature 53: `test_pattern_macro_parses[m! { 1 }]`
- Feature 54: `test_pattern_macro_parses[m!(x @ 1..=5)]`
- Feature 55: `test_pattern_macro_nested_in_tuple`
- Feature 56: `test_pattern_path_constant`
- Feature 57: `test_pattern_path_binding_rejected`
- Feature 58: `test_pattern_path_or`
- Feature 59: `test_pattern_standalone_rest_rejected`
- Feature 60: `test_pattern_struct_rest_batch3`
- Python macro diagnostic: `test_python_pattern_macro_raises_translate_error`

Batch 3 validation: `pytest -v` completed with 87 passed and 6 skipped; `ruff check --fix .` passed.

## N2a pattern batch 4

Nested or-patterns and guards were exercised in tuple, slice, struct, and binding patterns. Rust code generation now parenthesizes an or-pattern nested under an `@` binding. The Pratt precedence for bitwise `|` was corrected relative to comparison operators so boolean guard expressions retain their intended grouping. Python translation now rejects or-pattern alternatives that bind different names, supports payload-like tuple representations for enum variant patterns, and validates struct patterns against declared fields. Wrong-arity tuple patterns remain non-matching in Python; Cargo rejects structurally incompatible tuple patterns in native code.

### Tests in batch 4

- Feature 61: `test_pattern_or_tuple`
- Feature 62: `test_pattern_tuple_or`
- Feature 63: `test_pattern_slice_or_inner`
- Feature 64: `test_pattern_struct_or_inner`
- Feature 65: `test_pattern_binding_or`
- Feature 66: `test_pattern_guard_or`
- Feature 67: `test_pattern_guard_tuple`
- Feature 68: `test_pattern_guard_slice`
- Feature 69: `test_pattern_guard_struct`
- Feature 70: `test_pattern_guard_binding`
- Feature 71: `test_pattern_guard_or_expr`
- Feature 72: `test_pattern_many_arms`
- Feature 73: `test_pattern_arm_bindings`
- Feature 74: `test_pattern_or_disjoint_bindings_rejected`
- Feature 75: `test_pattern_or_binding_consistency`
- Feature 76: `test_pattern_struct_missing_rest_rejected`
- Feature 77: `test_pattern_struct_extra_field_rejected`
- Feature 78: `test_pattern_tuple_wrong_arity_rejected` and `test_python_tuple_pattern_wrong_arity_does_not_match`
- Feature 79: `test_pattern_slice_wrong_length`
- Feature 80: `test_pattern_nested_or_tuple`
- Cross-backend parity: `test_python_nested_or_guard_parity`

Batch 4 validation: `pytest -v` completed with 112 passed and 6 skipped in 14.69 seconds; `ruff check --fix .` passed.

## N2a pattern batch 5

Parser diagnostics now reject empty or-patterns, missing `@` and box subpatterns, standalone rest patterns, and multiple slice or struct rests with source positions. Python translation checks duplicate bindings and or-pattern binding consistency, validates const paths and struct shapes, and raises `TranslateError` with the offending pattern where available. Explicit `const NAME` patterns are emitted as `self::NAME` in Rust so unresolved constants cannot silently become identifier bindings. Rust remains responsible for type checking, duplicate bindings, exhaustiveness, and other invalid native patterns; these failures are surfaced through `RustCompileError.pretty()`.

Shared match diagnostics report the first unreachable arm and first duplicate literal per match, and report a guard binding that shadows the scrutinee name. The batch tests verify diagnostics are emitted once per match expression.

### Tests in batch 5

- Feature 81: `test_error_duplicate_binding_native`
- Feature 82: `test_error_duplicate_binding_or_different_types`
- Feature 83: `test_error_rest_outside_slice`
- Feature 84: `test_error_empty_or_pattern`
- Feature 85: `test_error_at_binding_without_inner_pattern`
- Feature 86: `test_error_box_without_inner_pattern`
- Feature 87: `test_error_unresolved_const`
- Feature 88: `test_error_unsupported_macro`
- Feature 89: `test_error_two_slice_rests`
- Feature 90: `test_error_two_struct_rests`
- Feature 91: `test_warning_unreachable_arm`
- Feature 92: `test_warning_duplicate_literal_arm`
- Feature 93: `test_warning_guard_shadows_binding`
- Feature 94: `test_error_non_exhaustive_native_match`
- Feature 95: `test_error_source_location`
- Feature 96: `test_error_duplicate_binding_contains_pattern`
- Feature 97: `test_python_unsupported_pattern_has_translate_error`
- Feature 98: `test_native_compile_error_pretty_has_pattern`
- Feature 99: `test_error_type_and_message_contract`
- Feature 100: `test_match_warnings_emitted_once_per_expression`

## N5 test harness restoration

Replaced the malformed `tests/test_native_n5.py` implementation copy with a `SOURCES` module consumed by the session-scoped `native_ns` fixture. All native sources are merged during collection and compiled as one generated extension for the session; every source function is prefixed `n5_`. Added parity coverage for the existing Option predicates/unwrapping and Python-only Result predicates.

- Covered by `tests/test_native_n5.py::test_native_option_methods`, `test_python_option_methods_parity`, and `test_python_result_predicates`.
- `pytest -q`: 227 passed, 9 skipped.

## N5 batch 5

Added Option `and_then`, `or_else`, and `filter` in the Python runtime; native Option combinator calls use Rust's standard Option methods. Native `filter` codegen now applies the Vec iterator/collect lowering only when the receiver is known to be a Vec, avoiding the Option collision. Python filter dispatch preserves existing Vec behavior while handling Option values. Option `map` remains intentionally unsupported because the code generator's `map` lowering is reserved for Vec; both backends now raise a clear error suggesting `match` or `if let` instead of failing with a confusing runtime `TypeError`.

- Covered by `tests/test_native_n5.py::test_native_option_methods` and `test_python_option_methods_parity`.
- Existing Vec filter behavior remains covered by `tests/test_native_n3.py::test_native_vec_filter` and `test_python_vec_filter_parity`.
- `pytest -q tests/test_native_n3.py tests/test_native_n5.py`: 43 passed.

## N5 batch 6

Added Python Result `and_then`/`or_else` and Result-to-Option `ok()`/`err()`, plus Option-to-Result `ok_or()`/`ok_or_else()`. The Result `and_then`/`or_else` operations remain Python-only: native Result values in local bindings cannot safely encode Rust `Err` as a Python exception. Native `ok()`/`err()` and Option `ok_or()` are verified through expressions returning ordinary Python-compatible values. Native Result constructors now annotate their inferred error type as `PyErr` where needed, preventing inference failures in conversion expressions.

- Covered by `tests/test_native_n5.py::test_native_option_methods`, `test_python_option_methods_parity`, and `test_python_result_conversions_and_combinators`.
- `test_native_result_and_then_or_else_requires_typed_err` is explicitly skipped because native Result `Err` cannot be represented in the existing local-binding model.
- `pytest -q tests/test_native_n5.py`: 19 passed, 1 skipped.

## N5 batch 7

Implemented `?` for Result values. Native generation emits Rust's `?` operator, preserving parse errors through the PyO3 `PyResult` return. Python `_Result.__try__()` returns Ok values and raises a private sentinel for Err; functions declared to return Result catch only that sentinel and return its Result value. Python string parsing under `?` returns an internal Result while existing non-`?` parse behavior remains unchanged.

- Covered by `tests/test_native_n5.py::test_native_result_try_operator` and `test_python_result_try_operator`, including Python Err propagation.
- `pytest -q tests/test_native_n5.py`: 21 passed, 1 skipped.

## N5 batch 8

Added parser and AST support for `if let Some(binding) = expression`. Native generation emits Rust `if let`. Python binds the expression once to a temporary, checks `_Option` and `is_some()`, then binds the payload before translating the branches. Other if-let pattern forms remain unsupported and raise `TranslateError`.

- Covered by `tests/test_native_n5.py::test_native_if_let_some_and_none` and `test_python_if_let_some_and_none_parity`.
- `pytest -q tests/test_native_n5.py`: 23 passed, 1 skipped.

## N5 batch 9 — out of scope

`while let Some(x) = iter.next()` is out of scope. Native Rust iterators already produce `Option` from `next()`, but the Python backend has no corresponding iterator adapter with Rust-style Option-returning `next()`; Python's built-in iterator protocol instead raises `StopIteration`. Adding an adapter, lifetime/consumption behavior, and parity tests would be a significant feature rather than a safe lowering. The batch is explicitly skipped rather than simulating mismatched behavior.

- `tests/test_native_n5.py::test_n5_while_let_option_iterator_not_supported` records the reason.

## N5 batch 10

Extended shared match diagnostics to warn on known Option/Result scrutinees with a Some/Ok arm but no None/Err arm, respectively. Both backends pass scrutinee type information from function parameters and typed/local constructor bindings, so unrelated user enums with variants named Some do not trigger false warnings. Complete matches remain warning-free, and diagnostics do not reject the program.

- Covered by `tests/test_native_n5.py::test_option_result_match_warn_once_per_match`, which checks exactly one warning per incomplete match through both translators and confirms complete/custom-enum matches do not warn.
- `pytest -q tests/test_native_n5.py tests/test_native_n2a_patterns.py`: 127 passed, 10 skipped.
- Final `pytest -q`: 243 passed, 11 skipped.
- `pytest -q tests/test_native_n5.py`: 24 passed, 2 skipped.

Batch 5 validation: `pytest -v` completed with 133 passed and 8 skipped in 13.35 seconds; `ruff check --fix .` passed.

## Async runtime

The native backend uses hand-written `PyFuture` and `PyFutureIterator` classes in `_RUNTIME_PREFIX` in `backends/rust/codegen.py`; it does not use `pyo3-asyncio` or PyO3 experimental async support. Both are regular `#[pyclass]` types. The boxed future is `Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>`, stored in a `Mutex<Option<...>>` so the pyclass is both `Send` and `Sync`, and `py_future` requires `Send`. Async free functions are emitted as `#[pyfunction] fn name(params) -> PyResult<PyFuture> { Ok(py_future(async move { ... })) }`, and await expressions emit `.await`. The tail value is converted to Python with `into_py_any(py)` inside `Python::with_gil`, fixing conversion for PyO3 0.26.

A single-threaded reactor thread backed by `mio` drives the futures. Each task has a waker that enqueues it for polling; timer deadlines use a binary heap; Unix file-descriptor readiness is registered with the reactor. When Python awaits a pending future, `PyFutureIterator` returns an `asyncio` future created on the running loop and the reactor completes it thread-safely with `loop.call_soon_threadsafe`. The internal `await_with_timeout` helper waits on a condition-variable waker with the GIL released and raises `PyTimeoutError` on expiry; it no longer busy-polls. `polyglot_sleep_ms` and Unix-only `polyglot_readable` are runtime/test helpers, not language-level APIs.

Async `&self` impl methods use the same lowering. Before `async move`, referenced fields are cloned into owned locals and `self.<field>` is aliased to those locals, because `&self` is not `Send`. Async `&mut self` methods are also supported: `PolyglotBorrowGuard::acquire_mut` is an atomic guard that rejects overlapping calls, mutable fields are cloned before the future runs, and writes are committed only when the future completes successfully. By-value receivers, bare `self` uses, and non-`Clone` payload-enum fields are not supported. Cancellation is drop-based: dropping a `PyFuture` or its iterator drops the boxed future without another poll.

Tests: `tests/test_native_n5.py::test_native_async_add`, `::test_native_async_identity`, `::test_native_async_send_add_across_threads`, `::test_native_async_impl_method`, `::test_native_async_timer_wakes_without_busy_waiting`, `::test_native_async_timers_run_concurrently`, `::test_native_await_with_timeout_value`, `::test_native_await_with_timeout_times_out`, `::test_native_dropping_future_iterator_cancels_before_poll`.

## Sandbox

Sandboxing is independent of the trust gate. `trusted=False` raises `UntrustedCompileError` before compilation; `sandbox=True` on `PythonBackend` replaces `__builtins__` with the restricted dictionary returned by `_sandbox_builtins()`. Compilation alone does not execute function bodies, so sandbox tests must call the compiled function to trigger blocked-name errors.

Tests: `tests/test_native_n5.py::test_sandbox_blocks_import`, `tests/test_native_n5.py::test_sandbox_blocks_eval`, `tests/test_native_n5.py::test_sandbox_blocks_open`, `tests/test_native_n5.py::test_sandbox_blocks_compile`, `tests/test_native_n5.py::test_sandbox_blocks_globals`, `tests/test_native_n5.py::test_sandbox_allows_safe_builtins`, `tests/test_native_n5.py::test_sandbox_allows_isinstance`, `tests/test_native_n5.py::test_sandbox_allows_len`, `tests/test_native_n5.py::test_sandbox_allows_exceptions`.

## Build performance

When available, `sccache` is configured through `RUSTC_WRAPPER` in `_build_env`. Cargo's `target` directory remains per-build under each build directory rather than being shared. After clearing the cache, a cold build takes about two minutes; a warm cached build takes seconds. `test_native_errors_map_back_to_polyglot_source` is marked `slow`; `pyproject.toml` sets `addopts = "-m 'not slow'"`, deselecting 22 slow tests by default.

## N5 complete

N5 delivers Option and Result construction and combinators, the `?` operator, `if let`, nested Option support, async functions through the custom PyFuture runtime, match exhaustiveness warnings, and Python built-ins sandboxing. The features that remain Python-only are Result `is_ok`, `is_err`, `map_err`, and `ok_or` in let-bindings (native `Err` emits `PyErr`, which does not match the declared error type), Box patterns, enum payload slices, and nested Vec slice patterns. Option `map` remains intentionally unsupported because it collides with Vec `map`; both backends reject it with a clear error.

Out of scope: `while let Some(x) = iter.next()` because Python has no Rust-style Option-returning `next()`. The M5 async work is documented below; async `&mut self` methods are now supported natively with the borrow guard and commit-on-success lowering.

> **Superseded by v0.3.6:** The original N5 note said the timeout helper was polling-based and that external wakeups/event-loop integration were unsupported. The v0.3.6 reactor replaced the no-op waker with real task wakers, a `mio` reactor, timer and Unix fd wakeups, and an `asyncio` bridge. See "M5: real async event loop".

Tests: `tests/test_native_n5.py` covers native and Python Option/Result, async, sandbox, `?`, `if let`, and match diagnostics.

`pytest -q`: 252 passed, 9 skipped, 4 deselected.

## N6 batch 1

Added Vec `is_empty`, `clear`, and `reverse`, plus String `repeat`, with equivalent Python and native lowerings. Native repeat counts are cast to `usize`; the Python backend uses string multiplication.

- Covered by `tests/test_native_n5.py::test_native_vec_clear_empty_reverse_and_string_repeat` and `tests/test_native_n5.py::test_python_vec_clear_empty_reverse_and_string_repeat`.
- `pytest -q`: 258 passed, 9 skipped, 4 failed, 7 deselected. The four failures were in unrelated inline annotation resolution and native extension loading (`_Py_IsFinalizing`); the affected Vec/String/N5 tests passed (95 passed).

## N6 batch 2

Added Vec `first`, `last`, and bounds-checked `get`, plus String `find`, with matching native and Python behavior. String search returns an Option containing the UTF-8 byte offset, and an absent match returns None.

- Covered by `tests/test_native_n5.py::test_native_vec_first_last_get_and_string_search` and `tests/test_native_n5.py::test_python_vec_first_last_get_and_string_search`.
- `pytest -q`: 260 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 3

Added Vec `truncate` and String `is_empty`, `starts_with`, and `ends_with` to both backends. Native truncation casts the signed source index to `usize`, matching Rust's behavior for negative values (no truncation).

- Covered by `tests/test_native_n5.py::test_native_vec_truncate_and_string_predicates` and `tests/test_native_n5.py::test_python_vec_truncate_and_string_predicates`.
- `pytest -q`: 262 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 4

Added String `trim_start`, `trim_end`, and `is_ascii` on both backends.

- Covered by `tests/test_native_n5.py::test_native_string_trim_start_end_and_is_ascii` and `tests/test_native_n5.py::test_python_string_trim_start_end_and_is_ascii`.
- `pytest -q`: 264 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 5

Added String `lines` and `split_whitespace`, plus Vec `sort_unstable`, on both backends. Python `lines` follows Rust's newline and CRLF terminator behavior.

- Covered by `tests/test_native_n5.py::test_native_string_lines_whitespace_and_vec_sort_unstable` and `tests/test_native_n5.py::test_python_string_lines_whitespace_and_vec_sort_unstable`.
- `pytest -q`: 266 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 6

Added String `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, and `to_ascii_uppercase` to both backends. `as_bytes` returns Python `bytes` from either backend, and the ASCII case methods leave non-ASCII characters unchanged.

- Covered by `tests/test_native_n5.py::test_native_string_bytes_boundaries_and_ascii_case` and `tests/test_native_n5.py::test_python_string_bytes_boundaries_and_ascii_case`.
- `pytest -q`: 268 passed, 9 skipped, 4 failed, 7 deselected. The same three inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error remained.

## N6 batch 7

Added Vec `rotate_left`, `rotate_right`, `fill`, and `swap` to both backends. Python lowering checks rotation and swap bounds rather than silently accepting indices or counts that Rust would reject.

- Covered by `tests/test_native_n5.py::test_native_vec_rotate_fill_and_swap` and `tests/test_native_n5.py::test_python_vec_rotate_fill_and_swap`.
- `pytest -q`: 270 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (51 passed).

## N6 batch 8

Added Vec `extend_from_slice`, `resize`, and `split_off` on both backends. The Python lowering preserves the receiver's retained prefix for `split_off` and does not mutate the source slice supplied to `extend_from_slice`.

- Covered by `tests/test_native_n5.py::test_native_vec_extend_from_slice_resize_and_split_off` and `tests/test_native_n5.py::test_python_vec_extend_from_slice_resize_and_split_off`.
- `pytest -q`: 272 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (53 passed).

## N6 batch 9

Added String `rfind`, `strip_prefix`, `strip_suffix`, `split_once`, and `rsplit_once` on both backends. Search positions are UTF-8 byte offsets, and split methods return an Option containing the two owned strings.

- Covered by `tests/test_native_n5.py::test_native_string_rfind_strip_and_split_once` and `tests/test_native_n5.py::test_python_string_rfind_strip_and_split_once`.
- `pytest -q`: 274 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (55 passed).

## N6 batch 10

Added Vec `partition`, `sort_by_key`, and `is_sorted` on both backends. Python lowering accepts the Rust-style `into_iter()` before `partition` and returns a pair of stable-order lists.

- Covered by `tests/test_native_n5.py::test_native_vec_partition_sort_by_key_and_is_sorted` and `tests/test_native_n5.py::test_python_vec_partition_sort_by_key_and_is_sorted`.
- `pytest -q`: 276 passed, 9 skipped, 4 failed, 7 deselected in the available Python 3.12 environment. The failures were the recurring inline annotation `NameError`s and native `_Py_IsFinalizing` extension-loading error; the targeted N5 tests passed (57 passed).

## N6 complete

N6 added the remaining `Vec` and `String` standard-library methods in batches 1–10. `Vec` gained `is_empty`, `clear`, `reverse`, `first`, `last`, `get`, `truncate`, `sort_unstable`, `rotate_left`, `rotate_right`, `fill`, `swap`, `extend_from_slice`, `resize`, `split_off`, `partition`, `sort_by_key`, and `is_sorted`. `String` gained `repeat`, `contains`, `find`, `rfind`, `is_empty`, `starts_with`, `ends_with`, `trim_start`, `trim_end`, `is_ascii`, `lines`, `split_whitespace`, `as_bytes`, `is_char_boundary`, `to_ascii_lowercase`, `to_ascii_uppercase`, `strip_prefix`, `strip_suffix`, `split_once`, and `rsplit_once`.

### Bug 1

**Symptom:** The Python backend returned `""` for `value.strip_suffix("")`, while native Rust returned the original string.

**Cause:** The suffix path contained a special case, `value[:len(value) - len(part)] if part else ""`, that assumed an empty suffix should strip the whole string. Rust treats both empty prefixes and empty suffixes as successful matches that remove nothing.

**Fix:** The suffix path now slices with `len(part)` unconditionally, so an empty suffix returns the original string.

**Lesson:** Empty-string edge cases in search and strip operations should follow the target operation's identity behavior rather than being special-cased.

### Review: UTF-8 byte offsets

Rust `find` and `rfind` return UTF-8 byte offsets. The Python backend obtains a character index from `str.find`/`str.rfind` and converts it through the UTF-8 length of the preceding prefix, which is correct both when the match itself is multibyte and when earlier characters are multibyte. The parity tests now cover a multibyte match (`"café"` finding `"é"` at byte 3), a later ASCII match (`"café x"` finding `"x"` at byte 6), an empty needle, and the corresponding `rfind` cases.

### Bug 2 (environment)

**Symptom:** Three inline-language tests failed with `NameError: name 'i64' is not defined`, and native extension loading failed with `_Py_IsFinalizing`.

**Cause:** The suite was run under Python 3.12 while the project declares `requires-python >= 3.14`. Before PEP 649, annotations are evaluated eagerly at definition time, so `def add(a: i64, ...)` fails before the `@inline_lang` decorator can run. The native extension-load failure was specific to that interpreter and toolchain combination.

**Fix:** The environment now matches the declared requirement (Python 3.14.8).

**Lesson:** Test on the interpreter range the project declares; annotation-evaluation changes can determine whether decorator-based tooling is reachable at all.

### Status after N6

- All four previously failing tests pass: the three inline annotation cases and the native extension-load case.
- `pytest -q`: 280 passed, 9 skipped, 7 deselected (`pytest -q -m slow`: 7 passed, 289 deselected).

## N7: HashMap

N7 adds `HashMap` support on both backends with byte-for-byte parity for the implemented surface:

- Construction: `HashMap::new()` and `HashMap<K, V>`-annotated bindings.
- Core operations: `insert`, `get`, `get_mut`, `remove`, `contains_key`, `len`, and `is_empty`.
- Views: `keys`, `values`, and `iter`, each returning an owned `Vec` so iteration can cross the Python boundary.
- Mutable iterators: `iter_mut` and `values_mut` yield mutable views, so `*value = ...` and `*value += ...` write back to the map.
- Entry subset: `or_insert`, `or_insert_with`, and `and_modify`.
- Mutable slots: `*slot = value` and `*slot += value` (plus `-=`, `*=`, `/=`) for values obtained from `get_mut` or the entry API.
- Struct fields: a `HashMap` field accessed as `self.field` inside an `impl` method is recognized as a map.

The Python backend represents a Rust `&mut V` with `_MapSlot`, which records the mapping and key and writes back on mutation, and the `Entry` API with `_MapEntry`. Dispatch on both backends is driven by the existing binding-type tables (`binding_types` / `_binding_types`) plus a per-impl field-type map, so it does not depend on global inference.

### Bug 1

**Symptom:** `*x += 1` compiled even when `x` was an ordinary integer, silently ignoring the dereference.

**Cause:** The `*` unary operator lowered to its operand in the Python backend, and only map-slot expressions were expected to reach the mutation path. Nothing checked that a dereference-assignment target was actually a mutable map slot, so invalid programs produced valid-looking Python.

**Fix:** Added `_is_map_slot_expr`, which accepts only tracked slot identifiers, `get_mut(...).unwrap()`/`.expect()`, and entry `or_insert`/`or_insert_with` chains. `_translate_assign_stmt`, `_translate_compound_assign_stmt`, and closure mutation lowering now reject any other dereference target with a `TranslateError`. Slot bindings are tracked in `translate_let`, scoped per function, and extended for closure parameters during body lowering.

**Lesson:** When a source operator is erased during lowering, validate the erased operand at the lowering site. Otherwise a type error in the source becomes a silently different operation at runtime.

### Ownership and parity review

- `get` returns an owned `Option<V>` on the native side by appending `.cloned()`; `keys`, `values`, and `iter` likewise return owned vectors. Generated structs and enums already derive `Clone`, so cloned views are available for every supported value type.
- Python `get`/views return the stored Python object rather than a deep copy, matching the existing `Vec::get` behavior. Mutating a value obtained from `get` therefore aliases on the Python side instead of being rejected by the borrow checker.
- `insert` and `entry` strip an explicit `&`/`&mut` from the key argument on the native side; `get`, `get_mut`, `remove`, and `contains_key` add one. This keeps Rust's `Borrow`-based signatures working for both `&key` and bare literals.
- Iteration order is unspecified in both backends because both use hash maps. Tests sort view results whenever order would otherwise be observable.
- Map-field detection resolves the receiver through parameter types and nested struct fields, so `self.field`, `param.field`, and `param.inner.field` all dispatch as maps when their declared types are known. Unknown or unannotated receivers still fall through to generic method dispatch.

### Follow-ups

- Added `HashMap` receiver resolution for parameters and nested struct fields on both backends, sharing the same `_expr_struct_type` helper used for map detection.
- Added `iter_mut` and `values_mut`: the Python backend returns `_MapSlot` views, while the native backend iterates with `.iter_mut()`/`.values_mut()`; loop variables are tracked as mutable slots so deref assignments write through.
- Added entry `key()` and `or_insert_with_key()`.
- Added `%=` for map-slot mutation and switched `/=`/`%=` to the Rust-compatible integer helpers.
- New coverage lives in `tests/test_native_n7_map.py` (`map_receivers`, `map_entry_key`, `map_iter_mut`, `map_values_mut`).

### Status after N7

- `pytest -q`: 326 passed, 9 skipped, 7 deselected (`pytest -q -m slow`: 7 passed, 335 deselected).
- New parity coverage lives in `tests/test_native_n7_map.py` and includes two rejection tests for invalid dereference assignments.

## N8: Primitive type semantics

N8 closes the gap between Rust numeric semantics and Python's unbounded integers:

- `as` casts lower to `_runtime_cast`, which reproduces Rust truncation and sign reinterpretation for integer widths, `isize`/`usize` (64-bit), `f32` rounding, `char` code points, and `bool`.
- `/` and `%` lower to `_runtime_int_div`/`_runtime_int_rem`: integer division truncates toward zero and the remainder keeps the dividend's sign; float `%` uses `fmod`.
- Explicitly annotated numeric bindings, assignments, compound assignments, and returns coerce through their declared type, so overflow wraps at the same boundary as native release builds.
- Overflow parity is scoped to typed numeric bindings. Unannotated inference can still differ because Python has no fixed-width integer type.
- Tests: `tests/test_native_n8_types.py`.

## M2: Python module isolation

Each Python compilation now uses a unique generated module name (`_polyglot_generated_<n>`) and registers itself under that name in `sys.modules`. Previously every program replaced `sys.modules["polyglot"]`, so a second compilation could detach dataclasses created by the first. `RustCodegen` also records `line_map_nodes` alongside the existing line/column map so diagnostics can retain the originating AST node.

Tests: `tests/test_python_module_isolation.py` and the `line_map_nodes` case in `tests/test_rust_compile_error.py`.

## Limitations audit

Each README limitation is pinned by a test or has an explicit out-of-scope rationale:

| Limitation                                                                                     | Coverage                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Result `is_ok`/`is_err`/`map_err`/`ok_or`/`and_then`/`or_else` are Python-only in let-bindings | `tests/test_native_n5.py` predicate/conversion tests; `tests/test_known_limitations.py::test_result_combinators_and_conversions_are_python_only`                    |
| `while let Some(x) = iter.next()`                                                              | `tests/test_known_limitations.py::test_while_let_next_is_a_parse_error`                                                                                             |
| Box patterns, enum payload slices, nested Vec slice patterns                                   | Python behavior in `tests/test_known_limitations.py`; native lowering remains out of scope because Rust slice patterns do not apply directly to `Vec`/enum payloads |
| Async `&mut self` methods use an atomic borrow guard, copied fields, and commit-on-success | `tests/test_known_limitations.py::test_async_mut_self_method_native_codegen_uses_borrow_guard`; `tests/test_native_n5.py::test_native_async_impl_method` covers the supported `&self` path |
| Async timer/I/O helpers are runtime-only; no language-level async timer or I/O primitives           | `tests/test_native_n5.py::test_native_async_timer_wakes_without_busy_waiting`, `::test_native_async_timers_run_concurrently`, `::test_native_await_with_timeout_value`, `::test_native_await_with_timeout_times_out` |
| WASM backend is experimental and compute-only                                                 | `tests/test_wasm_backend.py` covers the `i64`/`bool` subset and unsupported-node rejection                                                                          |
| Option `map` collides with Vec `map`                                                           | `tests/test_known_limitations.py` rejection tests for both backends plus `test_vec_map_still_lowers`                                                                |
| HashMap entry subset                                                                           | `tests/test_native_n7_map.py` entry tests                                                                                                                           |
| HashMap views allocate and have unspecified order                                              | `tests/test_native_n7_map.py` sorts view results where order is observable                                                                                          |
| HashMap dispatch requires a known type                                                         | `tests/test_native_n7_map.py` receiver tests; unknown receivers remain generic by design                                                                            |
| `get_mut`/entry mutation requires an explicit dereference                                      | `tests/test_native_n7_map.py` rejection tests                                                                                                                       |

## Milestone status

See the README Status table for the authoritative milestone status.

## N7 batch 1 — usize/isize as distinct types

**Symptom:** `Vec::len()` and index codegen silently crossed the `usize`/`i64` boundary, while Python casts ignored the selected target width.
**Cause:** Pointer-sized integers were treated as ordinary `i64` values in generated code and did not have explicit-cast runtime checks.
**Fix:** Added shared type helpers, Python/Rust boundary diagnostics, explicit cast lowering, target-width-aware Python casts, and removed implicit length/index conversions. Updated existing native test sources to use explicit `as` conversions where required.
**Lesson:** A backend must preserve source type boundaries before lowering; host integer behavior cannot stand in for the compilation target's pointer width.

- Covered by: `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`, `test_python_implicit_len_return_reports_boundary_error`, `test_native_implicit_len_return_reports_boundary_error`, `test_python_implicit_len_let_reports_boundary_error`, `test_python_i64_index_requires_explicit_cast`, `test_native_i64_index_requires_explicit_cast`, `test_python_usize_cast_overflow_raises`, `test_python_usize_cast_uses_selectable_target_width`
- `pytest -q`: 360 passed, 4 skipped, 7 deselected

## N7.1 — Function-argument boundary checks

**Symptom:** A `usize`/`i64` value could be passed to a known function parameter of the other type without an explicit cast.
**Cause:** Boundary inference checked assignments and returns but did not compare call arguments with declared function parameter types.
**Fix:** Both backends collect function signatures before lowering calls and reject mismatched known argument types with the uniform SPEC 2.4.7 diagnostic; unknown and external signatures remain unchecked.
**Lesson:** Validate type boundaries at every statically known transfer point, not just assignments and returns.

- Covered by: `tests/test_n7_usize.py::test_python_call_argument_requires_explicit_cast`, `test_native_call_argument_requires_explicit_cast`, `test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.2 — Range loop variable types

**Symptom:** Range loop variables did not retain the range endpoint's integer type, allowing implicit `usize`/`i64` mixing in the loop body.
**Cause:** Loop bindings were not included in the backend boundary type state.
**Fix:** Both backends infer the range type from its endpoint and bind the loop variable to that type; `0..Vec::len()` is `usize`, while an `i64` endpoint produces an `i64` loop variable.
**Lesson:** A loop binding inherits the type of the range, and explicit casts are required when crossing integer boundaries.

- Covered by: `tests/test_n7_usize.py::test_python_range_variable_inherits_usize_type`, `test_native_range_variable_inherits_usize_type`, `test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.3 — Receiver-aware `len()` inference

**Symptom:** Every `.len()` call was inferred as `usize`, including user-defined methods returning `i64`.
**Cause:** Type inference recognized the method name but did not inspect the receiver or known impl signatures.
**Fix:** Vec and String lengths infer as `usize`; known user methods use their declared return type; other receivers remain unknown. Python lowering also preserves a user-defined `len` method while using the runtime length helper for other receivers.
**Lesson:** Method return types depend on the receiver; a method name alone is not sufficient type evidence.

- Covered by: `tests/test_n7_usize.py::test_python_explicit_len_cast_and_usize_return`, `test_native_explicit_len_cast_and_usize_return`, `test_python_user_len_method_uses_declared_return_type`, `test_native_user_len_method_uses_declared_return_type`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N7.4 — Selectable Python target width

**Symptom:** Python `usize`/`isize` cast checks used the host width even when simulating a different compilation target.
**Cause:** Runtime width configuration was not connected to a backend instance.
**Fix:** The Python backend defaults to host pointer width and accepts an explicit `target_width` of 32 or 64 through its constructor or `get_backend("python", target_width=...)`; the selected width is scoped to compiled casts and does not mutate global runtime configuration. This option is Python-backend-only and does not alter native compilation behavior.
**Lesson:** Cross-target simulation must be explicit and isolated per backend instance so it cannot change default or native behavior.

- Covered by: `tests/test_n7_usize.py::test_python_backend_target_width_is_instance_scoped`, `test_python_usize_cast_uses_selectable_target_width`, `test_python_usize_cast_overflow_raises`
- `pytest -q`: 367 passed, 4 skipped, 7 deselected

## N8 — HashMap (collections layer)

**Symptom:** HashMap behavior existed in backend code but lacked a complete, traceable collections-layer contract; its `len()` result was not checked at the `usize`/`i64` boundary.
**Cause:** HashMap had not yet been included in the normative collection type set or N7's receiver-aware length inference, and native `get_mut` references cannot outlive the by-value map argument at the PyO3 boundary.
**Fix:** Specified and tested supported key/value types and operations, added HashMap `len()` inference and boundary checks, verified Python/native dictionary conversion, and made native `get_mut` return an owned copy at the Python boundary while retaining mutable internal use.
**Lesson:** Collection semantics include both language-level operations and explicit ownership/conversion rules at the host boundary.

- Covered by: `tests/test_n8_hashmap.py::test_python_hashmap_operations_and_option_results`, `test_native_hashmap_operations_and_option_results`, `test_python_hashmap_len_and_boundary_error`, `test_native_hashmap_len_and_boundary_error`, `test_python_hashmap_views_and_iteration`, `test_native_hashmap_views_and_iteration`, `test_python_hashmap_mutable_access_and_entry`, `test_native_hashmap_mutable_access_and_entry`, `test_python_hashmap_supported_key_value_types`, `test_native_hashmap_supported_key_value_types`, `test_hashmap_operations_match_across_backends`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected

## N9 — Error-handling boundary completion

**Symptom:** The Platform-defined Result and Option combinators lacked dedicated backend-specific acceptance tests, and native Result continuations were still marked Deferred.
**Cause:** Native errors are represented as `PyErr`, while Option-to-Result conversions needed to adapt their error values to the native exception representation.
**Fix:** Added separate Python/native tests for `is_ok`, `is_err`, `map_err`, `ok_or`, and `ok_or_else`; lowered native `ok_or*` errors to `PyErr`; verified native `and_then`/`or_else` with `PyErr` continuations and made them normative.
**Lesson:** Backend-specific error representations can remain explicit while combinators preserve their documented branch behavior.

- Covered by: `tests/test_n9_result.py::test_python_result_is_ok`, `test_native_result_is_ok`, `test_python_result_is_err`, `test_native_result_is_err`, `test_python_result_map_err`, `test_native_result_map_err`, `test_python_option_ok_or`, `test_native_option_ok_or`, `test_python_option_ok_or_else`, `test_native_option_ok_or_else`, `test_python_result_and_then_and_or_else`, `test_native_result_and_then_and_or_else`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected

## N10 — SPEC v0.2.7 and conformance status

**Symptom:** SPEC v0.2.4 still described the retired N7 transitional casts, Deferred HashMap and native Result continuations, and stale acceptance-test statuses.
**Cause:** The versioned specification had not recorded the completed N7–N9 behavior or the native HashMap `get_mut` boundary limitation.
**Fix:** Applied the N7 changes in v0.2.5, un-deferred HashMap and documented its native copy limitation in v0.2.6, and recorded the N9 combinator tests and native continuations in v0.2.7; updated 12.2 to 45 Pass and 6 Partial rows and synchronized the README test baseline.
**Lesson:** Conformance status must track tested behavior and versioned platform boundaries, including accepted limitations.

- Covered by: `tests/test_n7_usize.py`, `tests/test_n8_hashmap.py`, `tests/test_n9_result.py`, `tests/test_spec_conformance.py`
- `pytest -q`: 396 passed, 4 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 400 deselected
- `ruff check --fix . && ruff check .`: passed

## M10 batch 1 — termination checking for `#[terminates]`

**Symptom:** Annotated functions had no static check for obvious unbounded loops or recursive calls that do not decrease.
**Cause:** The parser discarded attributes and neither backend performed a termination proof before lowering.
**Fix:** Preserved the `terminates` attribute in `FnDecl`, added a shared conservative checker, and invoke it before either backend lowers the function. The checker accepts explicit finite ranges, direct lower-bounded decreasing `while` variants, and guarded integer or structural slice recursion; unsupported calls, async suspension, parameter mutation/shadowing, and unbounded loops are rejected with one identical five-line `TerminationError`.
**Lesson:** A termination measure must remain well-founded through every recursive step; conservative rejection is safer than accepting based only on a syntactic subtraction.

- Covered by: `tests/test_m10_termination.py::test_terminates_attribute_is_parsed_on_and_off`, `test_fn_decl_terminates_flag_is_preserved_in_ast`, `test_checker_ignores_unannotated_functions`, `test_python_terminating_patterns_execute`, `test_native_terminating_patterns_execute`, `test_python_unannotated_recursion_compiles_without_running`, `test_native_unannotated_recursion_compiles_without_running`, `test_python_non_terminating_patterns_rejected`, `test_native_non_terminating_patterns_rejected`, `test_termination_error_message_is_backend_identical`
- `pytest -q`: 416 passed, 4 skipped, 7 deselected
- Deferred: mutual recursion, loop with break, complex arithmetic, non-recursive calls, async suspension

The checker also conservatively rejects non-recursive calls and async functions because their termination cannot be established by the batch-1 proof subset; both remain within SPEC 10.22.

## F1 — Deliverable single-file toolchain

- Symptom: The language could be developed and tested from a source checkout, but it was not packaged as an installable tool with a stable CLI, a short Quick Start, or runnable example files for a new user.
- Cause: The repository focused on language semantics and backend milestones; `pyproject.toml`, `api/cli.py`, README onboarding, and example delivery artifacts were incomplete or oriented around internal compile workflows.
- Fix: Added package metadata and a `polyglot` console script, expanded the CLI with `run`, `build`, and `check` (while preserving the compatibility `compile` command), rewrote the Quick Start, added seven runnable examples, documented F1 in SPEC v0.3.0, and added CLI delivery tests.
- Lesson: Delivery milestones need product-facing contracts (install, entrypoint, examples, and error format) to be specified and tested like language behavior; otherwise a working compiler core remains hard to adopt.

## M6 — Module system (one-way include)

- Symptom: Programs could not compose source from multiple files; each file had to be self-contained.
- Cause: The module mechanism was deferred and there was no parser/loader support for file inclusion.
- Fix: Added `IncludeDecl` parsing for `include!("path/to/file.poly");` at item position, implemented recursive include expansion with cycle detection and path resolution relative to the including file, and wired file-based CLI commands (`run`, `build`, `check`) through `load_program`.
- Lesson: A one-way include directive is a small, explicit composition mechanism that fits the single-file/native-acceleration model without introducing visibility or module semantics.

- Covered by: `tests/test_m6_include.py::test_include_add_both_backends`, `test_include_add_native_backend` (native), `test_nested_include`, `test_cyclic_include_rejected`, `test_missing_include_file`, `test_include_path_relative_to_including_file`
- `pytest -q`: 453 passed, 11 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 464 deselected

## M7 — Diagnostics upgrade

- Symptom: Diagnostics lacked stable error codes and character spans, making tooling and regression tracking harder.
- Cause: Error classes used ad hoc formatting and AST nodes only carried line/column locations.
- Fix: Added `span` to AST nodes (best-effort), assigned stable codes (E001–E008) to boundary, termination, translation, Rust compile, include, unsupported-pattern, and invalid-literal errors, and extended diagnostics to the six-line format when a span is available.
- Lesson: Error codes and spans should be added as structured metadata early so format changes do not require heuristic parsing later.

- Covered by: `tests/test_m7_diagnostics.py::test_e001_boundary_error_has_code_and_span`, `test_e002_termination_error_has_code_and_span`, `test_e005_include_cycle_has_code_and_chain`, `test_every_error_class_emits_code_exactly_once`, `test_span_present_for_multi_token_expression`
- `pytest -q`: 453 passed, 11 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 464 deselected

## M4-remaining — Additional integer types

- Symptom: Only the N7 integer subset was available, limiting parity with Rust numeric types and explicit casts.
- Cause: Additional fixed-width integer types and their boundary/cast semantics were deferred after usize/isize work.
- Fix: Added integer literal suffix parsing, extended Python runtime cast helpers for i8/i16/i32/u8/u16/u32/u64/i128/u128, generalized implicit integer-boundary checks (E001), and implemented Python `as`-cast behavior with Rust-like wrapping except for literal out-of-range values which raise `OverflowError` at the cast boundary. Native support is partially leveraged through existing type handling.
- Lesson: Cross-backend integer semantics must distinguish compile-time literal diagnostics from runtime cast behavior, and that rule must be documented explicitly.

- Covered by: `tests/test_m4_integers.py::test_casts_i8_i64_succeed_python`, `test_casts_i8_i64_succeed_native` (native), `test_literal_out_of_range_raises_overflow_python`, `test_literal_wraps_native` (native), `test_implicit_conversion_boundary_error_e001`, `test_cross_backend_matrix_python`
- `pytest -q`: 453 passed, 11 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 464 deselected
- Deferred/partial: full native boundary/cast parity validation for the entire integer matrix in this pass; see SPEC 10.26.

## M3-remaining — HashMap API completion

- Symptom: Collections support stopped at the N8 core HashMap subset and lacked BTreeMap/HashSet and extended entry/nesting semantics required by M3.
- Cause: Entry combinators, additional collection types, and nested value support were deferred pending follow-on implementation.
- Fix: Added initial Python-backend support for `HashSet` and `BTreeMap` construction/basic operations and wired runtime helpers; added test scaffolding for the required M3 scenarios. Full Python/native parity for extended entry methods, nested HashMap depth-1 round trips, and BTreeMap/HashSet `get_mut` boundary semantics remains incomplete.
- Lesson: Collection APIs with backend-specific mutation/boundary models need incremental vertical slices (type + ops + boundary tests) per backend to avoid broad partial implementations.

- Covered by (partial): `tests/test_m3_collections.py::test_hashset_insert_contains_remove_len_iter_python`, `test_btreemap_sorted_iteration_python`; other required M3 cases are recorded as skipped/partial pending implementation
- `pytest -q`: 453 passed, 11 skipped, 7 deselected
- `pytest -q -m slow`: 7 passed, 464 deselected
- Blocker: native BTreeMap/HashSet/extended entry and nested collection value support, and full Python parity for all specified M3 cases, were not completed in one pass; recorded as Partial in SPEC 10.27 and 12.2.

## Parser/translator/codegen — `if let` as a tail expression

**Symptom:** A function whose tail expression is `if let Some(x) = v { x }
else { -1 }` returned `None` in the Python backend and failed with
`mismatched types` in the native backend. Ten related bugs surfaced:

1. `parse_block` put `For`/`While`/`Loop` into `block.tail`.
2. `parse_expr` allowed those nodes as binary operands, producing
   `Binary(For, -, 1)` when `-1` followed a `for` loop.
3. `parse_block` put `IfExpr` with statements in its body into `tail`.
4. `translate_expr`'s `IfExpr` branch read only `then_block.tail`,
   silently dropping `return` statements in `then_block.stmts`.
5. `translate_expr` had no `IfLet` branch.
6. `parse_block` put every `IfLet` into `stmts`, so it never reached
   `translate_expr`.
7. `opt { x }` in `if let Some(x) = opt { ... }` was misparsed as a
   struct literal, so `parse_block` saw `else` instead of `{`.
8. `_iflet_match` was not injected into the compiled namespace.
9. `_iflet_match` used `runtime._Option` for `isinstance`, but compiled
   code uses `translator._Option`. Two distinct classes.
10. `RustCodegen.expr_to_rust` had no `IfLet` branch and fell back to
    `/* expr IfLet */`, producing an empty `main` body.

**Cause:** Statement-only AST nodes leaked into expression positions at
multiple points. Runtime helpers were injected into the namespace by
hand and one was missed. `_Option` exists in both `translator.py` and
`runtime.py` as separate classes. `RustCodegen` treated `if let` as a
statement-only node.

**Fix:** Guards at every expression-entry point in `parser.py`
(`parse_block`, `parse_expr`, `_looks_like_struct_literal`); a
`_parsing_iflet_value` flag; `_ifexpr_has_stmts` and `_iflet_has_tail`
helpers; `_translate_iflet_expr` and `_iflet_match`;
`ns["_iflet_match"]`; duck typing in `_iflet_match`; and
`generate_block_body` dispatch for `IfLet` tail directly to
`_emit_if_let`.

**Lesson:** Every statement-only node needs an explicit guard at every
expression-entry point, in _both_ backends. Rust `if let` is an
expression, so the Rust codegen can emit it directly as a tail — but
it needs an explicit branch to do so. The cross-backend parity test
caught bug 10, which the Python-only tests would have missed.

**Test baseline:** 469 passed, 9 skipped, 15 deselected; slow
15 passed, 478 deselected.

**Technical debt:** `_Option` is defined twice (`translator.py:856`,
`runtime.py:330`). Long-term this should be unified. The current duck
typing in `_iflet_match` is a workaround, not a fix.

## polyglot build — module name vs filename mismatch

**Symptom:** `polyglot build hello.poly` produces `hello.dylib`, but the
exported init symbol is `PyInit_polyglot_native`. `import hello` fails;
the module must be loaded as `polyglot_native`.

**Cause:** The native backend emits a single `#[pymodule]` named
`polyglot_native` regardless of the source filename. The output filename
is derived from the source file, so filename and module name diverge.

**Fix (test):** `test_built_extension_loads_and_runs` loads via
`ExtensionFileLoader("polyglot_native", path)` + `exec_module()`. This
makes the test pass and removes the deprecated `load_module()` call.

**Open issue (resolved):** README Quick Start said `import hello`, which
did not work. Either the native backend should emit `PyInit_<filename>` per
source file, or README should document the `polyglot_native` loading
pattern. Tracked for a follow-up.

**Resolution (2026-11-14):** `polyglot build X.poly` now names the PyO3
module after the source stem, so the artifact is `X.<suffix>` and its init
symbol is `PyInit_X`. On macOS the copy step maps the Cargo `.dylib`
suffix to `.so` (the suffix Python imports), while the cache keeps the
original `.dylib`. `import X` therefore works after `polyglot build`, and
`NativeBackend.compile(program)` without a module name still defaults to
`polyglot_native` for programmatic callers. Regression coverage:
`test_build_output_importable_by_stem`.

**Test baseline:** 469 passed, 9 skipped, 17 deselected; slow
17 passed, 478 deselected.

**Lesson:** The filename and the exported module symbol are separate
identities. Tests that load the artifact must use the symbol name, not
the filename. Documentation must state which one is user-visible.

## M3-remaining completion — native BTreeMap, HashSet, entry methods

**Symptom:** SPEC 7.2.6-7.2.8 marked `BTreeMap`, `HashSet`, and the
`HashMap` entry extensions as Partial, and the native backend emitted
`HashMap` only. `tests/test_m3_collections.py` contained skip
placeholders for the deferred cases.

**Cause:** The initial collections milestone implemented the minimal
`HashMap` API and left the remaining surface for a later pass. Native
map dispatch was keyed on a single `_is_hashmap_expr` predicate, so there
was no place to encode `BTreeMap` or `HashSet` method lowering.

**Fix:** Replaced the predicate with `_map_kind`, which returns
`HashMap`, `BTreeMap`, or `HashSet` from bindings, struct fields,
`HashMap::new`-style constructors, and depth-1 nested lookups. The
native backend now emits:
`BTreeMap` construction/insert/get/remove/contains_key/len/is_empty/
keys/values/iter/entry with sorted iteration;
`HashSet` construction/insert/contains/remove/len/is_empty/iter; and
`entry().or_insert_with`, `and_modify`, `or_insert_with_key`.
Added `polyglot_hashset_get_mut` because `HashSet` has no std `get_mut`;
the Python boundary returns an owned copy, matching the documented
`Option<V>` semantics. On the Python side, `btreemap_insert` and
`btreemap_remove` now return runtime `Option` objects to match Rust's
`Option<V>` return type, and `hashset_get_mut` was added.
Tests cover both backends, shuffled `BTreeMap` insertion order, sorted
`BTreeMap` iteration, order-insensitive `HashSet` iteration (sort before
comparing), nested `HashMap` depth 1, and `get_mut` copy semantics.

**Lesson:** A single boolean predicate is too weak to dispatch a family
of collection types. Encoding the collection kind as a small enum-like
string made the Rust lowering table explicit and kept the Python and
native method sets easy to compare.

## M4-remaining completion — integer matrix and documented cast rules

**Symptom:** The additional fixed-width integer types existed, but the
cross-backend matrix was incomplete. Native literal casts out of range
failed with Rust errors such as `literal out of range for i8`, while the
M4 tests expected wrapping. Python `-1 as u8` was parsed as
`-(1 as u8)` and therefore did not trigger the literal-range check.

**Cause:** Rust infers an unsuffixed integer literal from the cast
target, so an out-of-range literal is a compile error rather than a
run-time wrap. The parser gives unary minus lower precedence than `as`,
so a negative literal needs parentheses before `as`. `cast_i64` also
lacked the `literal_out_of_range` keyword used by the other fixed-width
cast helpers.

**Fix:** `cast_i64` now delegates to `cast_int` like the other helpers.
The M4 tests parenthesise negative literals, expect `OverflowError` for
Python literal out-of-range casts, and expect `RustCompileError` for
native literal out-of-range casts. A native variable-cast matrix asserts
that run-time casts wrap. SPEC 8.1.5-8.1.7 records the exact
float-to-int, `char`, and `bool` rules: float casts truncate toward zero
(native saturates, Python wraps after truncation); `char as i64` yields
the scalar value, native integer-to-`char` accepts only `u8`, and Python
rejects surrogates and code points above `0x10FFFF`; `bool as i64` is
0/1 in both backends, while `i64 as bool` is Python truthiness and a
native compile error. `i128`/`u128` values above `2^64` are covered.

**Lesson:** Literal casts and variable casts are different operations:
literals are typed by their context and can be rejected at compile time,
while variables perform a run-time conversion. Tests must state which
one they exercise.

## M6 include hardening — duplicate, `..`, block, and non-string cases

**Symptom:** Include support covered the happy path, cycles, and missing
files, but not duplicate includes, `..`/absolute paths, include-only
files, block position, or non-string arguments. `include!` inside a
block leaked through to a generic `unknown macro include!` translation
error (E003) instead of a clear parse error.

**Cause:** The parser only recognised `include!` in `parse_item`; in
expression position it fell through to the generic macro parser.

**Fix:** The macro parser now rejects `include` with
`include! is only allowed at item position, not inside a block`.
Tests document that:
- duplicate includes are allowed and splice items twice (intended
  textual-inclusion behaviour, no de-duplication);
- relative `..` paths and absolute paths outside the repository are
  allowed (no containment policy is proposed, because the language has
  no sandbox model for source files);
- a file containing only includes produces an empty program and exits 0;
- `include!` inside a block and `include!(123)` are parse errors.
README and GUIDE document the directive.

**Lesson:** A textual include is not a module system. Defining duplicate
and path behaviour through tests avoids accidental de-duplication or
path-containment changes later.

## M7 diagnostics hardening — spans, JSON, and unreachable E003

**Symptom:** Some node types had no span, several error classes had no
test proving the span appears, and there was no machine-readable
diagnostic format.

**Cause:** Spans were assigned ad hoc by individual parser branches.
Boundary checks did not consistently pass a span. The CLI printed only
the text format.

**Fix:** Added `_fill_node_spans`, which walks the AST bottom-up and
derives a span from child spans when a node lacks one; keyword
statements (`return`, `break`, `continue`) get explicit spans.
Boundary checks in the Python translator and native codegen now pass
node spans, and include errors carry the include-site span. The CLI
gained `--error-format={text,json}` with `code`, `message`, `line`,
`col`, `span`, and `source`. `RustCompileError` from an invalid integer
parse literal now preserves code `E008` and the literal's span.
Tests cover span presence for E001-E008, E008 literal text, E006
resolved path and include site, a three-file E005 chain in both the
message and note lines, and JSON output for E001/E002/E005.

**E003 reachability:** `TranslateError` (E003) is raised only by the
Python translator. The native backend reports unsupported constructs as
`UnsupportedNativePatternError` (E007) or `RustCompileError` (E004).
There is no native E003 path to test; this is recorded in SPEC 10.31
and the M7 test asserts the native Box-pattern rejection is E007.

**Span coverage:** Before this pass, spans were already present for
literals, identifiers, paths, binary/unary/cast expressions, calls,
method calls, fields, indexes, array/tuple/struct literals, blocks,
`if`, `if let`, and `match` expressions. `_fill_node_spans` now derives
spans bottom-up for the remaining container nodes (`MatchArm`, `For`,
`While`, `Loop`, `Closure`, and statement wrappers with spanned
children), and `return`/`break`/`continue` receive explicit keyword
spans. Type and pattern nodes intentionally remain span-less; they are
not diagnostic boundaries.

**Lesson:** Spans must be derived for every node, not just the node
types that happened to need them. A JSON diagnostic contract is only
useful if every error path carries the same five fields.

## Repository tooling — docs sync, formatting, docs/, benchmark, CI contract

**Symptom:** README and SPEC 12.1 could drift, formatting was not
enforced in CI, there was no local task runner, and the architecture
was only described in prose.

**Fix:** Added `tools/check_docs_sync.py` (wired into CI and
`make check`), `ruff format --check .` in CI, a `Makefile`, a `docs/`
directory with an ASCII architecture pipeline and a one-page overview,
`CHANGELOG.md` for v0.3.0-v0.3.4, `tools/bench_native_build.py`
(cold/warm build timing, no timing assertions), and
`tests/test_ci_contract.py` to guard required CI step names. Packaging
tests assert `requires-python >= 3.14.8` and the
`polyglot = "api.cli:main"` console script.

**Test changes:** No existing test was deleted or skipped. M4 tests were
updated only where the documented semantics changed: negative literals
are parenthesised, Python literal out-of-range casts expect
`OverflowError`, and native literal out-of-range casts expect
`RustCompileError` (matching Rust's literal typing). Include errors now
carry spans, but still occupy five printed lines because the span is
appended to the location line.

## M5: Async extensions

### PyFuture was not Send

**Symptom:** A future created on one thread could not be awaited on another, and the limitation audit pinned `#[pyclass(unsendable)]` for both runtime types.

**Cause:** The boxed future type and the `py_future` bound had no `Send` requirement. Removing `unsendable` from a `#[pyclass]` also requires the type to be `Sync`, which the bare `Option` field was not.

**Fix:** Changed the boxed future to `Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>`, stored it in `Mutex<Option<...>>` so the pyclass is `Send + Sync`, removed `unsendable`, and added main-thread and cross-thread await tests. The stale limitation test was updated in a separate `test(rust)` commit because amending was not allowed.

**Lesson:** A `#[pyclass]` that is not `unsendable` must be both `Send` and `Sync`; interior mutability is the minimal way to keep a consumed `Option<BoxFuture>` while satisfying both. Limitation audits must move with the limitation.

### Async impl methods were silently synchronous

**Symptom:** An `async fn` inside an `impl` block was emitted as a plain synchronous Rust method, so native code returned the value directly instead of a `PyFuture`.

**Cause:** `generate_fn` applied the async lowering only when `not as_method`; the method path dropped the `async` keyword without an error.

**Fix:** Async `&self` methods now return `PyResult<PyFuture>` and wrap the body in `py_future(async move { ... })`. Referenced fields are cloned into owned locals and `self.<field>` is aliased to them before the move. Async `&mut self` methods are also supported: `PolyglotBorrowGuard::acquire_mut` rejects overlapping calls, all fields are cloned before `async move`, and writes are committed only when `__async_res.is_ok()`. By-value receivers, bare `self` uses, and non-`Clone` payload-enum fields remain unsupported. The known-limitations test now pins the borrow-guard lowering, and a native async-method test covers the supported paths.

**Lesson:** A silent synchronous fallback hides the absence of async semantics. Unsupported receivers should fail at translation time with a clear message rather than producing code with different behavior.

### Real async event loop (v0.3.6)

**Symptom:** Native futures were polled with a no-op waker, so a future that returned `Pending` could only make progress when Python called `PyFutureIterator.__next__` again. Timers busy-waited, and Python `asyncio` could not await a pending native future without blocking.

**Cause:** The runtime had no source of external wakeups. `PyFutureIterator` had no bridge to the running `asyncio` loop, and `await_with_timeout` polled in a loop with `std::thread::sleep`.

**Fix:** Added a single-threaded reactor in `backends/rust/codegen.py` built on `mio`. Each task has a per-task waker that enqueues it for polling. The reactor owns a deadline heap for timers and registers Unix file descriptors for readiness. `PyFutureIterator` now creates an `asyncio` future on the running loop when a poll returns `Pending`, and the reactor completes it with `loop.call_soon_threadsafe`. `await_with_timeout` waits on a condition-variable waker with the GIL released. The generated `Cargo.toml` adds `mio` with `os-poll` and `os-ext`.

**Lesson:** A no-op waker is only sufficient for futures that are always ready. Real async needs a wake source, a task queue, and an explicit bridge to the host event loop; otherwise "async" is just polling.

### Status

M5 is complete. Commit 1 also removed the thread-affinity limitation; commit 3 added drop-based cancellation. v0.3.6 replaced the no-op waker with the real reactor, timers, fd readiness, and the `asyncio` bridge described above.

- `pytest -q`: 907 passed, 5 skipped, 19 deselected.
- `pytest -q -m slow`: 19 passed, 912 deselected.
- `pytest -q tests/test_native_n5.py`: 62 passed, 2 deselected.
- `pytest -q tests/test_known_limitations.py`: 16 passed.

## M-T0a: Python match on Option and Result

**Symptom:** In the Python backend, `match Some(1) { Some(x) => x, None => 0 }` took the `None` arm, a bare `None` pattern raised `NameError: name 'None_' is not defined`, and `Ok`/`Err` patterns produced `int(None)` errors. Native matching already worked because the code generator emits real Rust patterns.

**Cause:** The Python pattern lowerer treated every `PatEnum` payload as a tagged tuple (`(variant, payload)`) and lowered `PatPath` through the user-path helper. Built-in Option/Result values are instances of the internal `_Option`/`_Result` runtime classes, so neither representation applied.

**Fix:** `_pattern_test_bindings` now recognises the built-in variants `Some`, `None`, `Ok`, and `Err`, including qualified forms such as `Option::Some` and `Result::Err`. It emits `isinstance(..., _Option)`/`_Result` plus `is_some`/`is_none`/`is_ok`/`is_err`, binds payloads through `unwrap`/`unwrap_err`, and applies nested payload patterns recursively. Wrong arity is rejected with a translation error. Bare variant names are treated as the prelude variants unless the scrutinee type is a known user enum; user enums can always qualify their patterns. Match scrutinees are now evaluated once, in both expression and statement position, instead of once per test.

**Tests:** `tests/test_native_n5.py::test_python_option_result_match_patterns`, `::test_native_option_result_match_patterns`, and `::test_python_option_result_match_arity_errors` cover direct construction, typed `None`, wildcards, qualified patterns, nested payloads, function-call scrutinees, guards, and arity rejection. The GUIDE Option/Result `match` examples were restored.

**Lesson:** Built-in generic types and user enums need different runtime representations. Pattern lowering must branch on the built-in constructors before falling back to the user-enum tuple encoding.

## M-T0b: package version drift

**Symptom:** `pyproject.toml` declared version `0.3.0` while `SPECIFICATION.md` was already at v0.3.6.

**Fix:** Set the package version to `0.3.6` and add `tests/test_packaging_config.py::test_version_matches_specification_title`, which compares the package version with the SPEC title. The test fails on future drift in either direction.

**Lesson:** Generated or manually recorded version numbers need an executable check; prose alone cannot keep two files aligned.

## M-T1–M-T3: bundled standard library

**Goal:** Ship a zero-dependency standard library as pure `.poly` modules and make it reachable from any source file without a package manager.

**Resolution:** `include_expander.py` treats the `std:` prefix as a reserved namespace. `resolve_std_include` resolves names against `Path(__file__).resolve().parent / "stdlib"`, appends `.poly` when the name has no suffix, rejects absolute names and paths that escape the stdlib root with `E006`, and reports missing modules with `E006`. Bare `include("std:...")` is accepted at item position for compatibility with the directive's call syntax; it remains a parse error inside a block.

**Deduplication:** Only resolved `std:` paths enter the standard-library dedup set, and only after a successful expansion. That keeps genuine cycles as `E005` while allowing `std:numpy` to be included by the program, `std:linalg`, and `std:prelude` without duplicating `NdArray` or its impl blocks. Ordinary include paths are not deduplicated; the existing duplicate-splice contract is unchanged.

**Modules:** `math` (integer helpers plus pure-`.poly` Newton/Taylor `sqrt_f64`/`exp_f64`), `algo` (sorts, searches, reductions, `unique`), `str_utils`, `collections` (`Stack`, `Queue`, `SetI64`, `Counter`), `numpy` (dense `NdArray`), `linalg` (Gauss-Jordan inverse and Gaussian-elimination solve), `random` (Xoshiro256** seeded by SplitMix64), and `prelude`. All use only language constructs that both backends support: `[...].to_vec()`, `"...".to_string()`, `match` instead of `Option::map`, explicit `as` casts, and `format!` instead of native `String + String`.

**Backend fixes found by the stdlib:** The parser now applies Rust's rule that a `{` in an `if`/`while`/`for`/`if let`/`match` scrutinee starts the body, not a struct literal, so `if a < b { a } else { b }` parses correctly. The Python translator infers struct types for unannotated `let` bindings and struct fields, dispatches user `impl` methods before built-in `Vec`/`String` rewrites, and uses `deepcopy` for `.clone()` so nested vectors do not alias. The native codegen prefers user methods, emits `#[staticmethod]` for receiver-less associated functions, lowers method `Result<T, String>` returns to `PyResult<T>`, and infers types through field, call, and `match` expressions.

**Packaging and tests:** `stdlib` is included in `[tool.setuptools.packages.find]`, and `stdlib/*.poly` is declared as package data. `tests/test_std_include.py` covers resolution, traversal, cycles, and deduplication. `tests/test_stdlib.py` compiles every module on the Python and native backends, checks the math invariants (`A.matmul(&A.inv()?) ≈ I` within `1e-9` and exact `solve` roots), verifies the reference RNG sequence, and covers `E006` and wheel metadata.

**Lesson:** A textual include mechanism needs an explicit namespace and an explicit dedup policy once it starts shipping shared definitions. Keeping dedup limited to `std:` preserves the documented behaviour of ordinary includes while making a prelude safe to include anywhere.

## Frames H–K: sound bounded approximations, never universal proofs

**Symptom:** Termination, aliasing, schema migration, and refinement are undecidable or environment-dependent in general. A naive "verifier" either overclaims or silently drops the hard cases, which is worse than refusing them.

**Cause:** The interesting facts are not in the program alone. A structural variant must be declared and checked at each recursive call; a points-to set is only sound as a declared over-approximation; a total field map and a rollback plan are properties of the caller's schema pair; a mutation score measures test sensitivity, not correctness.

**Fix:** Each frame exposes an explicit finite model and reports its boundary instead of hiding it. Frame H uses a bounded symbolic IR with an interval solver: exhaustive step/unroll/path/timeout limits produce `E008`/`E004`, `Unknown` feasibility is never reported as proved, and infeasible-path witnesses preserve symbolic atoms so the conflict is genuinely minimized. Frame I accepts caller-declared variants, infers only counted/monotone loop bounds, and labels everything else `UnboundedOrUnknown`; fuel is charged through the sandbox's replayable meter. Frame J computes points-to sets with range-overlap and TOCTOU windows, rejects lossy or ambiguous migrations with `E008`, and writes capsules only under a caller-supplied temp root; `MAP_JIT`, raw FFI, and PEP 667 frames are recorded as hazards or explicit boundaries. Frame K generates deterministic parsed cases, isolates mutations in symlink workspaces, and records exact commands and outputs while stating that passing is bounded.

Several defects were caught by the focused suites during construction: the parser wraps `return`, loops, and assignments in `ExprStmt`, so Frame H had to lower those wrappers; negating a guard needs the complement operator, not the swapped-operand form; nested branch/loop states had to propagate through a mutable child list or dead guards disappeared; rendered strings could not be re-minimized, so symbolic atoms are retained off the wire; `digest()` recursed through `as_dict()` in Frames I/J until it hashed a payload without the digest; and hazard detection needs two aliasing pointers to be meaningful.

**Gate wiring:** The `research-frames` stage runs the four focused suites and an injectable in-process spec oracle between `hard-frames-parity` and `diff-fuzz`. The default oracle evaluates real frame functions without spawning native, JIT, or WASM code, and all caches stay under caller-supplied roots so `git status --short` remains clean.

**Lesson:** A verifier earns trust by naming exactly what it did not prove. Bounded results with explicit `Unknown`, bound, and timeout diagnostics compose into a release gate; unconditional claims about undecidable properties do not.

## Build performance — APFS zero-copy target seeding for isolated native builds

**Symptom:** Default single-threaded `pytest -q` took over 21 minutes and exhausted temporary disk space because every isolated test `cache_dir` cold-compiled `pyo3` and `mio` from scratch.

**Cause:** `_seed_incremental_target` only ran for `incremental=True` builds and only searched inside `self.cache_dir`, missing prebuilt dependencies in `~/.cache/polyglot-native`.

**Fix:** Extended `_seed_incremental_target` to include `~/.cache/polyglot-native` as a fallback seed root and invoked it during standard `NativeBackend._build()` when `build_dir/target` is absent, cloning prebuilt dependencies via APFS `cp -Rc` (or `--reflink=auto`).

**Lesson:** Content-addressed isolation still benefits from read-only dependency seeding; copy-on-write cloning cuts full test suite runtime from ~21m 53s to ~3m 06s without duplicating disk footprint.

## Zig backend expansion — user calls, `if / else`, and struct field inference

**Symptom:** Zig backend programs could not call user functions with narrow integer parameters (`u8`/`usize`), failed when `if` appeared without `else` in statement position, and mis-inferred `f64` struct fields as `i64` (emitting `@intCast` instead of `@intFromFloat`).

**Cause:** `ZigCodegen` did not record function parameter types or struct field types during `_scan`, emitted expression-position `if` blocks without an `else` arm in statement position, and lacked `Binary` float propagation into `_emit_cast`.

**Fix:** Recorded `_fn_params` and `_struct_fields` in `_scan`, added `_coerce_call_arg` for narrow integer arguments, lowered `else`-less `IfExpr` as Zig statement blocks, and propagated `f32`/`f64` through `Field` and `Binary` inference.

**Lesson:** Even when structs stay internal to Zig (non-C-ABI), accurate field and parameter type tables are essential so casts and calls emit valid Zig builtins.

- Covered by `tests/test_zig_backend.py::test_zig_user_fn_calls_if_else_and_struct_fields`.

## Comprehension pass — bugs found while making the project teachable

Scope: while splitting `GUIDE.md` into a beginner path plus `docs/advanced.md`,
adding `polyglot new` / `polyglot explain`, hardening `stdlib`, and writing
task-oriented examples, four defects surfaced that no existing test caught.
They are recorded here with the same layout used by the N-series entries.

### Bug 1 — `check_program` used before it was imported

**Symptom:** Every `polyglot run` failed immediately with
`error: name 'check_program' is not defined`, including
`polyglot run examples/vec_ops.poly`.

**Cause:** Three files (`backends/python/translator.py`,
`backends/rust/codegen.py`, `backends/wasm/validation.py`) carried duplicated
blocks left by a repeated scripted edit: `self._proven_terminating =
check_program(program)` appeared above the local import that defines
`check_program`, the import line itself was written
`check_function, check_program, check_program`, and the WASM validator
referenced a `program` name that only existed in a different scope.

**Fix:** One local import at the first use site, duplicated statements removed,
and the WASM preflight now computes `proven = check_program(program)` once and
passes it into `_validate_item` instead of relying on an enclosing name.

**Lesson:** A scripted edit across several files needs an import-time smoke test
(`polyglot run` on a known example), not just a syntax check per file.

### Bug 2 — `reverse_text` indexed UTF-8 strings by byte

**Symptom:** `reverse_text("café")` raised
`'utf-8' codec can't decode byte 0x80` on the Python backend and returned
garbage on the native backend; `reverse_text("a😀b")` was worse.

**Cause:** `stdlib/str_utils.poly` built the result with `&text[i..i+1]`,
which slices at byte offsets. Any code point outside ASCII straddles several
bytes, so the slice is not a `char` and the conversion fails.

**Fix:** The function now indexes `text.chars()` and appends with
`format!("{}{}", out, chars[i])`. `is_palindrome` compares characters instead
of bytes, and `pad_left` / `pad_right` treat the requested width as a count of
Unicode scalars with a comment stating that choice.

**Lesson:** In a `String` type backed by UTF-8, byte offsets are never a
substitute for character positions. `tests/test_stdlib_unicode.py` now guards
this with a bit-mask probe so each behaviour can be checked in isolation on
both backends.

### Bug 3 — the native cache reused artifacts it did not build

**Symptom:** 14 cases in `tests/test_native_n2a_patterns.py` and
`tests/test_spec_conformance.py` reported "DID NOT RAISE" where a
`RustCompileError` was required — for example a `match` arm binding the same
name twice, which `rustc` rejects with `E0416`. The same compile reported
`reused: True` and exposed a namespace containing only `n8_valid_len`, a
function from a different program.

**Cause:** `_seed_incremental_target` clones another build's `target/`
directory into a fresh cache entry to keep `pyo3` and `mio` warm (see the
APFS seeding section above). That directory can already contain
`libpolyglot_native.dylib`, and `_is_built` treated the presence of any
extension file as "this source is compiled". A program that had never been
built was therefore loaded from someone else's module, so no compile error
could ever occur.

**Fix:** `_is_built` now requires `<build_dir>/.polyglot-manifest.json` with a
`module` equal to the requested module name, a `source_hash` equal to the
cache entry name, and an `extension` that actually exists in the entry.
`_build` writes that manifest after a successful cargo run;
`_build_incremental` already wrote an equivalent one.
`tests/test_native_cache_manifest.py` pins the four rejection cases and the
one acceptance case.

**Lesson:** A content-addressed cache must verify its own identity from
metadata it wrote; inferring "done" from the shape of files on disk is exactly
the assumption that seeding breaks.

### Bug 4 — the fix for Bug 3 broke a caller that tolerates a missing artifact

**Symptom:** `tests/test_rust_compile_error.py::test_warning_only_cargo_result_does_not_raise`
failed with `RuntimeError: No extension module found`.

**Cause:** The manifest was written by calling `_find_extension`, which raises
when cargo exits successfully without emitting a library — a state the
existing warning-only test intentionally produces.

**Fix:** The manifest step uses the quiet lookup and writes nothing when no
extension exists; such an entry simply rebuilds next time.

**Lesson:** A verification step added after an existing success path must not
tighten that path's contract for callers that already relied on it.

### Status after this pass

- Full fast suite: **3258 passed, 6 skipped, 31 deselected.**
- Native pattern and spec-conformance failures: 14 → 0.
- All three bugs are covered by tests that fail if the behaviour returns.

## Expansion frontier — five bounded models, no overclaiming

The challenge asked for five surfaces: cluster/location-transparent actors,
deterministic simulation testing, a hygienic cross-backend macro engine, a
zero-copy shared-memory arena, and a self-hosted compiler core. Each landed as a
model with a written limit (SPEC 10.47–10.51) instead of a claim, because each
full version needs infrastructure this repository does not have (sockets and
persistence, a real preemption model, a surface macro syntax, a second compiler
generation).

**Why models still earn their place:** they make the *decision logic* testable.
Quorum election, epoch fencing, seeded tie-breaking, capture avoidance, FIFO
release and digest fixed points are all properties you can break with a small
change, and now each has a named test file. When the heavier version arrives, the
tests describe the behaviour it must keep.

**Deliberate choices:**

- The cluster module reuses `Cluster`, `LeaseManager` (`bump_epoch`/`fencing`),
  `wire.encode_frame`/`decode_frame` and `DistributedError` rather than building
  a second transport. Heartbeat state is kept on the cluster so two calls form
  one timeline.
- Freezing on a split is the default. A minority that keeps accepting writes is
  the failure mode split-brain names, so no quorum means no writable group and
  no epoch change.
- DST ties break on a hash of the seed rather than `random.shuffle`, so the same
  seed reproduces the same trace across processes and Python versions.
- Reclaim is whole-arena and requires an epoch that advances. Being blunt and
  explicit is better than a per-slot free list that can be wrong silently.
- Stage 2 re-reads only canonical IR shapes. If it accepted anything, the
  fixed-point check could pass by re-printing garbage that proves nothing.

### Bug 1 — `Sorted` comparisons on dataclass payloads

**Symptom:** the first scheduler draft used `sorted(key=(at_ms, tie_key))` with
the operation object as the last tuple element. Python compares tuples
element-wise, so equal keys reached the `SimOp` dataclass and raised
`TypeError`.

**Cause:** the sort key carried the object itself instead of a total order over
keys.

**Fix:** include the original index before the operation, making the tuple
totally ordered without ever comparing payloads.

**Lesson:** a deterministic scheduler needs a *total* key, not a mostly-unique
one.

### Bug 2 — `PillarError` has no `context` keyword

**Symptom:** five DST tests failed with
`PillarError.__init__() got an unexpected keyword argument 'context'`.

**Cause:** the signature in `runtime/diagnostics.py` is
`(code, message, *, reason, help_text, note, line, col, span)`; the DST module
invented a `context=` argument from memory of a different codebase.

**Fix:** dropped the keyword and moved the extra detail into the message.

**Lesson:** read the diagnostic constructor before writing new call sites; the
five-line format is the contract, not a suggestion.

### Bug 3 — the ring holds `N-1`, and the test assumed `N`

**Symptom:** `test_full_ring_refuses_the_push` expected the second push into a
three-slot ring to fail; it succeeded.

**Cause:** the classic ring-buffer invariant: with `head == tail` meaning empty,
one slot must stay unused to distinguish empty from full.

**Fix:** the test now pushes `N-1` payloads before expecting `-1`, and the
docstring in `arena.push` states the invariant.

**Lesson:** document capacity as `slots - 1` at the API, or every caller will
assume `slots`.

### Bug 4 — exported memoryviews block `mmap.close()`

**Symptom:** `BufferError: cannot close exported pointers exist` when a test
closed an arena with an outstanding `RingLease`.

**Cause:** `pop` returned a `memoryview` over the mapping and nothing tracked it;
`mmap.close()` refuses while any view is exported.

**Fix:** the arena keeps a list of live views, drops each on `release`, and
`close` releases them all first.

**Lesson:** zero-copy hands out a lifetime; whoever hands it out must own the
release bookkeeping.

### Bug 5 — the IR reader assumed one top-level form

**Symptom:** a two-function program failed stage 2 with
`IR has trailing tokens after the program`.

**Cause:** `parse_ir` read a single root and then demanded end-of-input, but
`emit_ir` writes one line per function.

**Fix:** read top-level forms until the token stream is exhausted.

**Lesson:** a canonical printer and its reader must share the same notion of
"program"; one-form and many-form are different grammars.

### Bug 6 — the subset tokenizer had no `=`

**Symptom:** `let sum = x + y;` failed with `unexpected character at offset 34: ' '`.

**Cause:** the token regex listed `==` but not the single `=` used by `let`, so
`re.match` failed at the space before it and reported the whitespace.

**Fix:** add `=` to the token alternation; the parser already required it.

**Lesson:** a tokenizer error pointing at whitespace usually means the *next*
character is unrecognised.

**Baseline after this batch:** 3340 passed, 6 skipped, 31 deselected (fast
suite); the five new test files add 82 tests and no new error codes.

## Ownership checking (E009) — parity by construction

The README listed "move checking is a parity gap" as a known limitation: the
same `.poly` file could pass on the Python backend and fail on the native
backend with rustc's "use of moved value". Closing that gap needs the rule in
one place, not in two backends.

**Design:** `backends/move_check.py` walks the shared AST between parsing and
code generation. `TranslateError`-style formatting is reused: one `E009` with
the binding name, the line it moved on, and the line that reused it, plus a
span when the node has one. `PythonBackend.translate_program` and
`NativeBackend.build` both call it, so `run`, `check`, `exec`, `compile` and
`build` all agree. The pass is bounded: it models locals with a statically
known type and reports only what it can prove; unknown types and unmodelled
method semantics are left to backend, as SPEC 8.16.5 requires.

### Bug 1 — branching on a diverging branch produced false positives

**Symptom:** Four shipped `stdlib` modules were rejected by the new pass:
`algo` (`bubble_sort`), `random` (`shuffle`), `str_utils` (`pad_left`), and
`prelude` (which includes them). Every message pointed at a `return` inside a
guard followed by a later use of the same binding.

**Cause:** the pass merged the state of both `if` branches unconditionally, so
`if n < 2 { return out; }` marked `out` moved on the fall-through path even
though that branch never reaches the code after the `if`. Rust does not work
that way: a diverging path contributes nothing to the join.

**Fix:** `terminates_block` / `terminates_expr` recognise `return`, `break`,
`continue`, `panic!` and `if`/`match` where every arm diverges. A diverging
branch merges as the entry state, a diverging loop body leaves the exit state
untouched, and statements after a diverging statement are treated as
unreachable. All four modules then passed, which matches rustc.

### Bug 2 — array-literal `vec![...]` hid the element type

**Symptom:** `let words = vec![String::from("a")]; take(words[0]);` was
accepted by the pass even though the native backend rejects it with
"cannot move out of index".

**Cause:** the parser lowers `vec![a, b]` to an `ArrayExpr`, and the checker's
element-type helper only understood `Vec<T>` generics, so `[String]` produced
no element type and `is_copy(None)` stayed quiet.

**Fix:** `_element_type` understands the `[T]` form as well. The parity table
now pins the index case to `cannot move out of index of 'words'`.

### Bug 3 — the pass could not be imported by the installed CLI

**Symptom:** `polyglot run` failed with `No module named 'move_check'` while
`pytest` passed, because the editable install ships only the modules listed in
`pyproject.toml`.

**Cause:** the module lived at the repository root, which is neither a
`py-modules` entry nor a package.

**Fix:** it moved to `backends/move_check.py` and is imported as
`backends.move_check`. `pyproject.toml` stays untouched.

### Status after this pass

- `tests/test_move_parity.py`: 8 allowed programs (run on both backends) and
  12 rejected ones that must raise the same `E009` headline and line pair.
- Full fast suite: **3258 passed, 6 skipped, 31 deselected**; slow suite
  **31 passed**.

## Zig-exclusive surface — intrinsics, packed layout, unchecked bodies

The Zig backend is explicitly not one of the two parity backends (SPEC 8.12.1),
which makes it the right place to carry capabilities the Python and native
backends have no way to model: hardware bit operations, an exact bit layout, and
a per-body safety switch. Eight named intrinsics and two attributes landed
against real `zig 0.17.1` compilation rather than a mock.

**Why a table instead of inline checks:** every intrinsic has four facts that
must stay consistent — the Zig builtin it maps to, the operand widths it
accepts, the result type it produces, and the diagnostic to raise when the
operand is wrong. `zig/intrinsics.py` holds them once, and the backend, the
tests and the Python mirror all read from it. The mirror (`mirror()`) computes
the same answer in Python, so the tests compare Zig's result against an
independent implementation instead of against a constant someone typed.

**Deliberate choices:**

- Counting intrinsics widen to `i64`; `bswap!`/`bitreverse!` keep the operand
  type. Widening a `u64` byte swap would truncate the top half, and narrowing a
  count would lose the meaning.
- Widths come from the inferred polyglot type, with `usize`/`isize` modelled at
  64 bits. An unannotated identifier is `i64` rather than "whatever the machine
  is". Guessing would produce answers that change with the platform.
- `#[packed]` refuses floats outright instead of inventing a layout. A packed
  struct is a claim about bits; a byte-order-independent bit count for `f64`
  would be a claim this backend cannot back up.
- `#[unchecked]` sets `@setRuntimeSafety(false)` as the first statement of its
  own body, so the blast radius is one function. The emitted `std.debug.assert`
  bounds guards stay, because `#[unchecked]` is about arithmetic overflow, not
  about removing checks the caller was promised.

### Bug 1 — the `bswap` mirror reversed the wrong way

**Symptom:** `mirror("bswap", 0x12345678, "u32")` returned `0x12345678`, i.e.
its own input, so the cross-check agreed with any Zig result at all for
palindromic-looking cases and silently passed.

**Cause:** the mirror reversed the integer's *bits* over the declared width
instead of reversing the byte order. For a 4-byte value that reverse-twice
sequence is the identity, which is exactly why it looked plausible.

**Fix:** split the value into `width // 8` bytes big-endian, reverse the byte
sequence, and re-assemble big-endian. `0x12345678` now yields `0x78563412`, and
the Zig test asserts that value rather than the input.

**Lesson:** a mirror that can only return its input is not a check. The test had
to be rewritten so the expected value differs from the argument.

### Bug 2 — `sizeof!` returned 4 for a 24-bit struct

**Symptom:** `sizeof!(Two)` for `struct Two { a: u12, b: u12 }` returned 4,
while `bitsizeof!` said 24. The first test expected 3.

**Cause:** `@sizeOf` reports the size of the *backing integer* after alignment,
and Zig rounds a 24-bit integer up to its ABI alignment of 4 bytes. Nothing was
wrong with the lowering; the expectation was.

**Fix:** the test now expects 4 and states why, and the docs and SPEC 10.52
split the two semantics: exact layout claims use `bitsizeof!`, byte-size claims
use `sizeof!` and accept the rounding. Without that split, a caller comparing
`sizeof!` against a hand-summed bit count would see a contradiction the
backend never promised to resolve.

**Lesson:** when a query wraps a platform rule, assert the platform rule in the
test and name it in the docs, rather than adjusting the implementation until it
matches an assumption.

## Zig-exclusive surface, second batch — overflow, rotation, conversion, buffers

The first batch gave the Zig backend a bit-level and layout surface. The second
batch fills the gaps that first batch left in the *arithmetic* story, and adds
the two things the subset table had admitted were missing: float/integer
conversion and byte-buffer memory operations.

**What landed:** `add_with_overflow!`/`mul_with_overflow!` (flag as `i64`),
`wrapping_add!`/`wrapping_mul!` (wrapped value at the operand width), `rotl!`/
`rotr!`, `int_from_float!`/`float_from_int!`, `alignof!`, `memcpy_bytes!`/
`memset_bytes!`, and `Simd::reduce_and`/`or`/`xor`. All of them were probed with
`zig test` on 0.17.1 before any lowering was written, so the emitted shapes
(`@addWithOverflow(...)[1]`, `std.math.rotl(u32, x, @as(u32, @intCast(n)))`,
`@memcpy(dst[0..n], src[0..n])`) are the ones the compiler actually accepts.

**Deliberate choices:**

- The overflow report splits into two names instead of returning a struct.
  `@addWithOverflow` produces a `.{ value, flag }` tuple, and struct
  construction is not lowered (8.12.9), so a two-field return type would have
  forced a half-finished struct feature into an intrinsics batch. Two entry
  points that index the tuple keep the batch honest and small.
- Three arithmetic modes now exist side by side: checked `+` (traps), wrapping
  (`wrapping_add!` or `#[unchecked]`), and saturating (`+|`). Naming the middle
  one explicitly is the point: before this batch, "wrap instead of trap" was
  only reachable by dropping safety for a whole function body.
- `rotl!` casts the count with `@intCast`, so an out-of-range count traps. The
  alternative — masking the count modulo width — hides a caller bug behind a
  plausible answer.
- `alignof!` was added instead of `offsetof!`. That is not an arbitrary
  preference: probing showed `@offsetOf` on a `packed struct` reports `0` for
  every field (the fields share one backing integer) and on a plain struct the
  fields may be reordered. An `offsetof!` would therefore answer `0` or a
  non-declaration order, and both are wrong for the question "where is this
  field". `offsetof!` is skipped and the reason is written into SPEC 10.53.
- `memcpy_bytes!` takes one `count` and builds both views from it, so the two
  lengths cannot disagree; overlap is documented instead of detected.

### Bug 1 — the examples runner and the parity suite each had their own marker list

**Symptom:** the first batch's new example was picked up by the slow
cross-backend parity suite as a Python/native case, because that suite's list of
"Zig-only" markers still named only `+|`/`SimdN<`/`#[parallel]`.

**Cause:** two tests kept two copies of the same list. The fast-suite failure
(`zig_intrinsics.poly is not listed in examples/README.md`) was the visible half;
the slow-suite misclassification was the half that would only appear in a slow
run.

**Fix:** `zig/features.py` now owns `ZIG_ONLY_SOURCE_MARKERS` and
`is_zig_only_source()`, and both `tests/test_f1_cli.py` and
`tests/test_cross_backend_parity.py` call it.

**Lesson:** when two test files need the same "which programs does this backend
own" answer, the list belongs next to the backend, not next to one test.

### Bug 2 — a rotate expectation was wrong, not the lowering

**Symptom:** the first `zig test` probe failed on `rotl(0x12345678, 5)`, and the
value the test expected was `0x91A2B3C0`.

**Cause:** the expectation was arithmetic I had done in my head; the real answer
is `0x468ACF02` (the low five bits `0x18` land at the top). The lowering was
already correct.

**Fix:** recompute against the mirror formula, then assert the mirror value in
the test instead of a hand-computed constant. `mirror_rotate` exists so the test
and the emitted code cannot both be wrong in the same way.

**Lesson:** a failing probe is evidence about *someone's* arithmetic, not
necessarily the compiler's. Cross-check before editing the lowering.

## The Linux CI segfault — `RTLD_DEFAULT` is not -2 everywhere

**Symptom:** the ubuntu release gate failed at `pytest-full ... [exit -11]` while
the macOS job on the same commit passed, and the rest of the gate was green. The
fault handler (added hours earlier) named the frame:

```
Fatal Python error: Segmentation fault
  File "runtime/hotpatch/trampoline.py", line 577 in resolve_symbol
  File "tests/test_quantum_hotpatch.py", line 972 in test_got_discovery_finds_a_bound_symbol
```

**Cause:** `RTLD_DEFAULT` was written as `ctypes.c_void_p(-2)`, which is what dyld
uses for "search every loaded image". glibc spells it `NULL` and treats *any*
other handle as a `link_map *`, so `dlsym(0xfffffffffffffffe, "getpid")` reads
that address and dies. It is not an error the caller can observe: there is no
return value to test, no `errno`, no exception — the interpreter is gone. Verified
directly on x86_64 Linux: `c_void_p(0)` returns the address, `c_void_p(-2)`
segfaults. macOS never runs the branch that fails, which is why a one-line
constant could pass 3,513 tests locally and still kill the Linux job.

**Fix:** the handle is now chosen per platform (`-2 if _DARWIN else 0`), with the
reason written next to it, and `test_rtld_default_carries_the_platform_value`
asserts the value on either platform so the constant cannot silently drift back.

**Symptom, second round:** the gate ran again with that fix and `pytest-full`
still ended at `exit -11` — but one frame further along, inside the scan that was
only reachable now that `dlsym` returned:

```
Fatal Python error: Segmentation fault
  File "runtime/hotpatch/trampoline.py", line 657 in _elf_got_slots
  File "runtime/hotpatch/trampoline.py", line 627 in find_got_slots
  File "runtime/hotpatch/trampoline.py", line 813 in discover
```

**Cause:** the Linux fallback scanned *every* writable mapping from
`/proc/self/maps` and read 8 bytes at each address with
`ctypes.c_void_p.from_address`. Heap, stack, `[vvar]` and anonymous regions have
no PLT cells in them, and a writable mapping that is not readable (or one a
native library has already released) faults on the first read. Two things were
wrong at once: the scan read memory it had no business reading, and it filtered on
`w` without checking `r`.

**Fix:** the scan now requires a readable, writable, file-backed mapping whose
path is not a bracketed pseudo-file, and skips anything above a 96 MiB budget, so
it touches the data segments of loaded objects only — which is the only place a
bound GOT cell can live. On the Linux clone that is 800 KiB across 9 mappings,
14 ms, versus every writable page in the process before. `_read_proc_maps` now
carries the readable bit, `_page_protection` uses it instead of assuming
`PROT_READ`, and `test_elf_scan_touches_only_readable_file_backed_mappings` stubs
the mapping table against a real heap buffer so the rule is tested on macOS too.

**Lesson:** two of these three failures were invisible from the platform the tests
normally run on, and the only reason either was found is that the gate was changed
to keep the failing check's full output and print the fault handler's own block.
An `exit -11` with a 25-line tail of addresses would still be an open bug. The
deeper lesson is about the diagnostic itself: "scan all writable memory" is a
sentence that sounds bounded and is not — the bounds were never written down, so
nothing could check them.

## pylint 9.19: one artifact, and nine import cycles that were real

**Symptom:** `pylint . -j 4` rates the tree 9.19/10 and prints a hundred-plus
`R0801` (duplicate-code) blocks plus ten `R0401` (cyclic-import) blocks, every
last one headed `debug/timetravel.py:1:0`. That file is a small debug helper and
duplicates none of it.

**Cause:** both checks are graph-level. `duplicate-code` compares the modules it
has seen against each other and `cyclic-import` walks the aggregated import graph;
both finalise after the per-file pass, so the message is attached to whichever
module happens to be current when they fire — the last one processed. Verified by
running `pylint --disable=all --enable=R0801 diagnostics/format.py
diagnostics/telemetry.py`: the two blocks come back under
`diagnostics/telemetry.py:1:0`, and the actual pair only appears inside the body
of the message (`==diagnostics.format:[81:95]` /
`==diagnostics.telemetry:[112:126]`). The header file is not the duplicate and
`debug/timetravel.py` is not the culprit.

**Fix:** no duplicate block was touched. Seven of the ten cycles came from one
place: `backends/base.py` held `get_backend()`, whose body imports each concrete
backend, while every backend imports `BackendBase` from `base`. The imports were
already lazy, so nothing failed at runtime, but the module graph carried a cycle
per backend. `get_backend()` now lives in `backends/registry.py`, `base` imports
nothing from the backends at any level, and `backends/__init__.py` still exports
the factory, so `from backends import get_backend` is unchanged. The only in-tree
caller of `from backends.base import get_backend` was `tests/test_n7_usize.py`.
R0401: 10 → 3.

**Lesson:** pylint counts a function-local import as a module edge, which is why
"the import is lazy, so there is no cycle" — true for the importer — still reports
as `R0401`. The three that remain have that same shape inside a single backend
family (`move_check ↔ borrow_checker`, `rust.compiler ↔ rust.codegen`,
`wasm.codegen ↔ wasm.aggregates`): one side owns an exception class, the other
imports it late. Clearing them means relocating those classes to standalone
modules — churn with no runtime benefit, so they stay, documented rather than
silenced with a disable.

**Duplicate code, deferred.** The `R0801` pairs are 5–15 line per-module helpers
(`_text`/`_integer`/`_fail`, the `"i8"…"u128"` width tables, `__all__` re-export
lists) plus the HIR/MIR twins. They exist because each module keeps its own
validators; merging them is a cross-module refactor with a wide blast radius and
is not something a lint pass can do. Recorded here so the rating is not read as an
unfinished cleanup.

First batch merged: the four literal scanners (`raw_string_prefix`,
`string_literal_end`, `char_literal_end`, `token_end`) now live in
`compiler/frontend/lexeme.py` and are imported by both `frontend/lexing.py` and
`frontend/framework.py`. The two copies had already drifted — the framework one
returned `(start, False)` for a call not starting at a quote, the lexing one
scanned to end of file — and both are on the stricter contract now. The same
round made `api.visualizer.line_col_to_offset` public and deleted the
byte-identical `_offset_for_line_col` from `api/visualizer_server.py`.

Second batch merged: the recovery machinery (`RecoveringLexer`,
`RecoveryLexError`, `dispatch_one`, `recovery_stop`) moved to
`compiler/frontend/lexer_recovery.py`. The two copies differed only in the
exception class they raised, so the class is shared and both modules catch it.
That leaves one block between `lexing.py` and `framework.py` — the eight-line
recovery loop, whose per-error payload differs (`_byte_offset` records vs
`_make_record`) — and it is left alone deliberately.

## The incremental-rebuild test was timing a warm scope

**Symptom:** `pytest -q` failed
`tests/test_native_skeleton.py::test_incremental_rebuild_only_recompiles_changed_function`
with "incremental rebuild burned 6.69s of CPU vs 6.01s for the initial build" —
the rebuild cost more than the build it was meant to beat — while the same test
passed on its own.

**Cause:** the test pinned `cache_scope="incremental-test"`, so every run reused
one cargo `target` directory. A per-run `module_name` kept the *artifact*
distinct, but not the compile cache: once the scope was warm, the initial build
was already as cheap as an incremental one and the 0.75× budget stopped
separating them. Under xdist load the two costs landed within noise of each
other.

**Fix:** the scope is per-run (`incremental-<module_name>`), so the first build is
a cold crate build and the second reuses the `target` it just populated. Still a
child-CPU ratio, not wall clock.

**Lesson:** a timing assertion is only meaningful when the state it measures is
fresh. `cache_scope` reads like test isolation, but it isolates artifacts, not
the compiler's cache.

## Bounded expansion batches — syntax, diagnostics, and native lowering

The earlier note that `while let Some(x) = iter.next()` was out of scope is now
historical: `while let PATTERN = EXPR` is paired in both backends. Python uses a
loop over an `Option`-returning helper, so it no longer depends on Python's
`StopIteration` protocol; native emits Rust's own `while let`.

The same batch adds `let else`, struct update `rest`, structural type aliases, a
single simple `use`, inline `mod`, and labelled `break`/`continue`. Python
lowers non-innermost labels through flags so the control transfer reaches the
named loop; native emits Rust labels. External module files, aliases, globs,
alias generics, and tuple-pattern rewrites stay out of scope.

Standard-library coverage grows in paired steps: Vec gains drain/extract-if,
resize/fill-from-closure, comparator searches and partition point, and
equal-length `swap_with_slice`; String gains in-place-style mutators through
Python rebinding plus iterator materializers; BTreeMap gains first/pop/range and
HashSet gains `is_subset`; `HashMap::entry(k).insert(v)` is emulated natively
with `Occupied`/`Vacant` rather than calling Rust's unstable entry insertion.
Native `extract_if` needs the two-argument `(range, predicate)` form, and
`rsplit_inclusive` is emitted as reversed `split_inclusive` output.

Diagnostics move too. `E009` joins the registered public code table with errno
22 and the ownership-fix hint, mirrored in `zig/abi.py` and the ABI contract.
`PatMacro` is rejected by one shared `E007` helper in both backends, while
`PatConst` lowers to a constant path. The language server validates with
recovering parsing so it publishes every syntax error from one document, and it
converts UTF-16 positions only at the LSP boundary; internal spans stay Python
character offsets. A shared flow pass warns once per block for unreachable code
after `return`.

The located runtime guard now covers an out-of-bounds assignment target on the
Python backend, including compound assignment. That guard is deliberately not
added to native codegen: doing it honestly would require widening fallible
functions to `PyResult` throughout the generated crate, so native assignment
still relies on Rust's bounds panic. That asymmetry is recorded in SPEC 10.54
rather than hidden behind a claimed parity test.

The native codegen batch emits a small, explicit attribute set (`inline`,
`cold`, `must_use` on `Result` functions, and validated `repr(transparent)`),
adds local dead-code allows to generated helpers, lowers immutable `&String` to
`&str` and immutable `&Vec<T>` to `&[T]`, and keeps an owned-`Vec` PyO3 wrapper
because generic slices cannot be extracted from Python. Literal comparisons are
folded to `true`/`false`; no general constant propagation is implied.
