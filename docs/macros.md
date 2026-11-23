# Hygienic cross-backend macros

`compiler/macro_engine/` provides the three pieces a macro system needs before
it can be trusted across backends: hygiene, typed splicing, and a parity check
over the lowered opcodes.

The AST is a miniature one (`ast.py`): literals, variables, `add`, calls,
`let`, `if`, `return`, blocks, and `Hole` for a macro parameter.

## Hygiene

```python
from compiler.macro_engine import Let, Hole, Var, macro_resolve_hygienic_scopes

template = Let("tmp", Hole(0), Var("limit"))
expansion = macro_resolve_hygienic_scopes(
    template, scope_id=7, call_site_names=("limit",)
)
expansion.template  # Let('tmp#7', Hole(0), Var('limit@7'))
expansion.binders  # (('tmp', 'tmp#7'),)
expansion.captured  # ('limit',)
```

- Names the template binds are renamed to `name#<scope>`.
- Names the template references resolve at the definition site, `name@<scope>`.
- A free name that also exists at the call site is recorded in `captured`, which
  is the report you want when auditing a macro for accidental capture.
- Unknown globals stay bare; the pass does not invent bindings.

## Typed splicing

```python
from compiler.macro_engine import macro_expand_typed_ast_splice

result = macro_expand_typed_ast_splice(
    Let("tmp#7", Hole(0), Hole(1)),
    (LitInt(3), Var("counter")),
    param_types=("i64", "unknown"),
)
result.holes_filled  # 2
```

- Arity must match the declared parameter list, and every hole must be filled
  exactly once (`E002`).
- `i64`, `str`, `bool` are checked; `unknown` is accepted for names because the
  surrounding checker owns their type. A concrete mismatch is `E001`, and
  `add` on non-`i64` operands is also `E001`.
- `depth`/`max_depth` turn runaway recursive expansion into `E002` instead of a
  stack overflow.

## Cross-backend parity

```python
from compiler.macro_engine import macro_verify_backend_expansion_parity

report = macro_verify_backend_expansion_parity(expanded.ast)
report.parity  # True
report.opcodes  # canonical sequence, e.g. (('const.int', '1'), ('return',))
report.digests  # per-backend digest of the emitted stream
```

Each backend lowers the same expanded tree to a canonical opcode stream; any
difference is `E004` with the first divergent opcode index. The three emitters
in `parity.py` are deliberately simple textual models. The harness proves that
the *project* checks parity, not that Python, Rust, and Zig agree on everything
(SPEC 10.49).

## Boundaries

The engine proves hygiene and typed splicing on a miniature AST. It does not
parse surface syntax, does not know the real backend IRs, and does not attempt
macros over statements the mini AST cannot express. That is why the deliverable
is a harness with honest limits rather than a claim of a production macro
system.
