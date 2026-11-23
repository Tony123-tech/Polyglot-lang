# Rust ergonomics: plans instead of boilerplate

Three problems Rust programmers keep hitting are modelled in
`backends/rust/` as **plans**: deterministic descriptions of what Rust code
would look like and what rustc would reject. They generate no Rust source and
invoke no Cargo, so they are importable and testable anywhere the toolchain
itself is installed.

Why plans and not generated code: the point of each of these is a *decision*
(does my impl set cohere, are my dispatch arms disjoint, is this cleanup order
correct), and a decision is easier to assert than a 90-line `unsafe` block.
Nothing here claims a performance property. The limits are in SPEC 10.56 and
the requirements are in SPEC 8.22.

## Trait plans (`backends.rust.traits_ext`)

### Coherent extension

Implementing a trait for a type you do not own is legal exactly when your crate
owns the trait or the type. The plan says which, and refuses the rest before
anyone writes the impl:

```python
from backends.rust.traits_ext import rste_impl_plan

rste_impl_plan(
    [
        {"trait": "Show", "trait_local": True, "for_type": "f64", "type_local": False},
    ]
)["emitted"]
# ('impl Show for f64',)
```

Two foreign items raise `E002` with the orphan-rule explanation, and a
duplicate concrete impl raises `E002` naming both positions.

### Blanket composition

An adapter chain is a set of `requires`/`provides` edges. `rste_compose` walks
them in declaration order and returns the path plus the single `where` clause a
caller would otherwise hand-write:

```python
from backends.rust.traits_ext import rste_compose

plan = rste_compose(
    [
        {"name": "num", "requires": [], "provides": ["Num"]},
        {"name": "fmt", "requires": ["Num"], "provides": ["Fmt"]},
        {"name": "print", "requires": ["Fmt"], "provides": ["Printable"]},
    ],
    target="Printable",
)
plan["path"]  # ('num', 'fmt', 'print')
```

A cycle raises `E005`; an unreachable target raises `E002` naming the bounds
that are missing.

### Fast-path dispatch, honestly flagged

Disjointness is checked against the `covers` list each arm declares — the plan
trusts coverage arguments instead of guessing them, which is the only way a
dispatch stays correct as arms are added:

```python
from backends.rust.traits_ext import rste_fast_path

plan = rste_fast_path(
    [
        {"category": "primitive", "handler": "add_i64", "covers": ["i64"]},
        {"category": "concrete", "handler": "add_widget", "covers": ["Widget"]},
    ],
    fallback="add_generic",
)
plan["arms"]  # ('i64 => add_i64', 'Widget => add_widget', '_ => add_generic')
plan["requires_specialization"]  # False
```

A generic `Copy` arm (`covers=["T"]`) sets `requires_specialization=True` and
`stable=False`, because that arm can only exist on nightly `specialization`. The
plan never reports such a dispatch as stable. Two arms covering a shared type
raise `E002` naming that type.

### Iterator and `Try` shapes

`rste_iter_adapter` and `rste_try_adapter` emit the associated-type shape,
including the lifetime binder a borrowed adapter needs and the residual type a
`Try` adapter carries:

```python
rste_iter_adapter("i64", borrowed=True, fallible=True, error_type="PillarError")["item"]
# "Result<&'a i64, PillarError>"
```

## Borrow splitting (`backends.rust.borrow_ergo`)

### Two mutable fields, one struct

```python
from backends.rust.borrow_ergo import rsbe_split_fields

rsbe_split_fields(["left", "right"])["emitted"]
# ('let left = &mut self.left;', 'let right = &mut self.right;')
```

A split while a whole-struct borrow is live raises `E002` — that is the case
where the alternative is cloning the whole value or reaching for
`Rc<RefCell<..>>`.

### Disjoint slice windows

`rsbe_split_windows` validates half-open ranges and returns the covered count
and the gaps, which is the argument a reviewer wants:

```python
plan = rsbe_split_windows(10, [(0, 3), (3, 7)])
plan["covered"]  # 7
plan["gaps"]  # ((7, 10),)
```

Overlap raises `E002` naming both ranges; out-of-bounds raises `E001`.

### Generational arenas

`rsbe_arena_plan` describes `ArenaKey { index, epoch }` and the slot header;
`rsbe_arena_simulate` actually runs the protocol, so the stale-handle property
is testable rather than asserted:

```python
from backends.rust.borrow_ergo import rsbe_arena_simulate

result = rsbe_arena_simulate(
    [
        {"op": "insert", "value": "old"},
        {"op": "remove", "key": {"index": 0, "epoch": 0}},
        {"op": "insert", "value": "new"},
        {"op": "get", "key": {"index": 0, "epoch": 0}},
    ]
)
[entry["outcome"] for entry in result["trace"]]  # ['ok', 'ok', 'ok', 'stale']
```

The slot is alive again when the last `get` runs, and the old key still
resolves to nothing. A slot whose generation reaches the configured limit is
reported `burned` instead of wrapping.

### Scoped cleanup

`rsbe_scope_exit` plans a guard whose `Drop` runs the actions in reverse
acquisition order, so an early return and an unwind get the same cleanup:

```python
rsbe_scope_exit(["open_file", "begin_tx"])["drop_order"]  # ('begin_tx', 'open_file')
```

## Async and PyO3 boundaries (`backends.rust.async_interop`)

### Error context that is kept

`map_err` chains throw away the layer that explains the failure.
`rsai_error_chain` keeps every layer and maps each one onto the `E001`-`E008`
surface:

```text
error[E002]: report is empty
  in: write_report
  caused by [E006] read_text: missing input
```

`rsai_aggregate` is the non-short-circuiting counterpart: it dedupes, sorts
deterministically, reports every failure, and picks the lowest code as the
headline so a caller that prints only one code still sees the most structural
problem.

### Cancellation-safe schedules

- `rsai_retry_plan` returns `attempts - 1` waits, the capped exponential
  schedule, and the checkpoints where a cancellation is observed. It states
  plainly that a cancellation during an attempt is safe only when that attempt
  is idempotent.
- `rsai_deadline_plan` allocates a budget so the sum is exactly the budget,
  gives every step its minimum, and raises `E002` when the minimums exceed the
  budget instead of silently shaving a step.
- `rsai_cancel_scope` refuses a step with no commit point (`E002`), lists
  commit order and reverse rollback order, and names the safe cancellation
  windows.

### GIL discipline and typed conversions

`rsai_gil_plan` refuses a Python-object operation inside an `allow_threads`
region (`E002`), which is the boundary PEP 667 makes observable — a dict view
mutated while the GIL is released is a race, not a convenience. The conversion
helpers are closed tables on purpose: `rsai_convert_result` lists the `Result`
pairs this boundary knows, and `rsai_convert_enum` refuses a variant carrying a
reference because a `&T` payload cannot outlive the call.

## What these plans do not do

They do not solve traits, they do not evaluate associated-type equality, and
they do not detect a data race a caller introduces in some other way. Disjoint
splits are disjoint *by argument*: distinct fields and non-overlapping ranges,
not projections or nested collections. `rsbe_arena_simulate` is evidence for the
algorithm, not for emitted Rust. And no number in a retry or deadline plan is a
latency claim. See SPEC 10.56.
