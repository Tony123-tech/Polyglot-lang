# Bootstrapped compiler core

`bootstrap/` compiles a clean subset from source to canonical IR as "stage 1",
re-reads that IR and re-emits it as "stage 2", and requires the two digests to
match. That is the fixed-point property a self-hosting compiler needs from its
own intermediate form.

## The subset

```polyglot
fn add(x, y) -> i64 {
    let sum = x + y;
    return sum;
}
```

Supported: `fn name(a, b) -> i64|bool { ... }`, `let`, `return`, `if`/`else`,
calls between functions, integers, and `+ - * / < > ==`. Anything else is a
syntax error (`E003`) with a plain explanation instead of a parser trace.

## Stage 1

```python
from bootstrap import bootstrap_stage1_compile_subset

artifact = bootstrap_stage1_compile_subset(source)
artifact.ir  # '(fn add (params x y) (ret i64) (block (let sum (add (var x) (var y))) (return (var sum))))'
artifact.digest  # sha256 of the IR text
artifact.zig  # optional Zig rendering when emit_zig_text=True
```

The type checker runs first: unknown variables and return mismatches are `E001`,
calls to undefined functions are `E006`, duplicate parameters/functions and
syntax errors are `E003`. The IR is canonical: one line per function, fixed
spacing, no source positions, so equal programs produce equal text.

## Stage 2 fixed point

```python
from bootstrap import bootstrap_verify_stage2_fixed_point

report = bootstrap_verify_stage2_fixed_point(artifact)
report.stable  # True
report.rounds  # 1
report.stage2_digest  # == artifact.digest
```

`parse_ir` accepts only the shapes `emit_ir` can produce; anything else is
`E008`. The check therefore cannot pass by re-printing garbage, and a printer
that stops being idempotent fails with `E004` instead of claiming self-hosting.

## Diffing two stages

```python
from bootstrap import bootstrap_diff_stage_ir_hashes

diff = bootstrap_diff_stage_ir_hashes(stage1, stage2)
diff.equal  # True
diff.first_diff_line  # None, or 1-based line of the first divergence
```

## Boundaries

Stage 2 is the same Python implementation driven from the printed IR, not a
separate binary produced by stage 1, and the Zig emitter handles only
return-only functions (`E007` otherwise). What is verified is IR canonicality and
digest stability — the property that has to hold *before* a real second
generation can be attempted. It is not a claim of machine-code self-hosting.
