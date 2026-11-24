# Glossary

The vocabulary this repository uses for its own machinery. Each entry says what
the term means *here*, and points at the clause or file that defines it, because
several of these words mean something looser outside this project.

## Backends and tiers

**Tier 1.** The two backends the documentation, examples and error-code
guarantees cover: the Python backend and the native (Rust/PyO3) backend. They
share one front end and are parity-tested. **Tier 2** is the experimental
compute-only WebAssembly backend (`i64`/`bool` only). **Tier 3** is research:
the JIT backend and the Zig backend. Tier 2 and 3 are opt-in, can lag behind
language changes, and are not covered by the parity contract. Table:
`README.md` → "Backends by tier".

**Parity (對等).** Two Tier 1 backends agreeing about a program. It is narrower
than "the same program works everywhere": SPEC 8.16.2 promises that when both
backends reject a program, the code, the binding and the line pair agree, and
10.58 says everything a single backend decides on its own (rustc's `E004`/`E007`,
the Python backend running what native refuses) sits outside that promise.

**Guarded failure.** One of the operations SPEC 2.9.1 lists — integer `/` and `%`
by zero, Python-backend indexing out of bounds, and `panic!`/`assert!` — failing
at run time. It must carry a public code, a position and a hint, and must never
leak the host exception's class name (2.9.3). Runtime failures reuse `E002`.

**Error codes `E001`–`E009`.** Public codes are `E001`–`E008`. `E009` is the
ownership diagnostic and is deliberately outside the public table
(`runtime/diagnostics.ERROR_CODES`): it names a rule violation the front end can
prove before any backend runs, not a failure of a guarded operation. Before 11.26
`E009` meant an out-of-range cast literal; that case moved to `E008` (2.9.2).

## Front end

**Span.** The `span` field on an AST node: a `(line, column)` tuple where the
construct starts, carried from the lexer through the parser so diagnostics and
the visualizer can point at source. It is not a byte range.

**Inline language.** Writing a Rust-like `fn` body directly under a Python
`def` and letting a decorator compile it
(`compiler/core/inline_lang.py`). The decorator reads the function's *source*,
rewrites the `def` header into a `fn` item and parses it, which is why parameter
annotations must be present but never need to be resolvable objects.

**Monomorphization.** Expanding generic definitions into concrete instances.
`compiler/analysis/monomorph.py` does this on the shared AST, with
`MAX_MONOMORPH_DEPTH` bounding recursion, so both Tier 1 backends see the same
concrete program.

**Termination checking.** `#[terminates]` asks the compiler to prove a function
always returns. `compiler/analysis/termination.py` does that with a ranking
argument and refuses conservatively: calls across functions and `async` are
rejected rather than assumed terminating (SPEC 10.22).

## Ownership analysis

**Move checking.** The bounded, flow-sensitive pass in `backends/move_check.py`
(`check_moves`) that reports one `E009` with the binding name and both lines. It
runs on the shared front end, before either backend generates code, so the two
Tier 1 backends cannot disagree about a move (8.16).

**Bounded NLL.** What this project has instead of Rust's region inference. Borrow
death is approximated by the holder's last read line inside the current
function, and only direct local bindings are modelled; a borrow that escapes
into a return value, a struct field, a closure or a stored reference is outside
the model. It is called *bounded* because the approximation stops where proof
stops — the pass loses an error rather than inventing one (10.55).

## Documents

**Normative.** A rule a change MUST satisfy, stated in `SPECIFICATION.md`. A
section marked **roadmap, not shipped behaviour** (for example Section 13) is
normative about a *future* contract only; the current tree is not required to
meet it, and a reader should not test the tree against it.

**Deferred.** An entry in `NOTES.md`'s bug log or a clause that records what is
deliberately *not* done, so the limitation is discoverable instead of implied.
The house style is Symptom / Cause / Fix / Lesson, written when the defect is
found rather than after the fix.

**Baseline.** The test counts quoted in `README.md`, SPEC 12.1 and `NOTES.md`.
`tools/check_docs_sync.py` compares them, so a suite that changes size must
update all three in the same change.

**Capability status.** `runtime/capabilities.py` labels each of the
framework-function probes `foundation`, `modelled` or `rejected`. `foundation`
means the adapter exists in this tree; `modelled` means the behaviour is
described but not enforced; `rejected` states a boundary and why it holds.
