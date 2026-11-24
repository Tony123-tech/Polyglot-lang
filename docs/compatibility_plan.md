# Python 3.14.8 compatibility plan (design note, not implemented)

This is a design note, not a schedule. Nothing here is shipped: the CLI refuses
anything below 3.14.8 today, no 3.11–3.13 run is recorded, and the compatibility
layer described below does not exist (SPEC 9.9.1, 10.57). It is written so that
the floor is a visible decision rather than a wall, and so the next person to
pick this up starts from evidence instead of starting over.

## What the floor actually buys

The gate lives in `diagnostics/host.py` (`MIN_VERSION = (3, 14, 8)`,
`require_supported_python`) and `api/cli.py::main` calls it before it parses
arguments, so an older interpreter gets one `error:` line and one `help:` line
instead of a traceback.

Reading the code, the floor rests on a single mechanism: **when annotations are
evaluated**. Under PEP 649 (3.14 default) `def add(a: i64) -> i64` does not
resolve `i64` at definition time. On 3.11–3.13 it does, so the module fails with
`NameError: name 'i64' is not defined` before the inline-language decorator runs.

What matters is that the decorator never needs the annotations as *objects*:
`compiler/core/inline_lang.py` reads the decorated function's source with
`inspect.getsource`, and `_format_params`/`_format_return` take
`ast.unparse` of the annotation nodes. It rebuilds a `fn` header as text and
parses that. No entry point calls `typing.get_type_hints`, and nothing imports
`annotationlib`; the only reflective use of annotations in the tree is
`backends/hotpatch.py` copying `__annotations__` dictionaries verbatim, which is
version-independent.

So the honest summary is: the floor is about *evaluation timing*, not about
3.14-only syntax or library features. That makes a lower floor plausible, and
also explains why the refusal is a one-line check rather than a compatibility
shim.

## Options

| Option | What it is | Cost | Risk |
| --- | --- | --- | --- |
| A. Keep 3.14.8 only | Status quo, refusal documented in README/SPEC 9.9.2 | None | Every reader on 3.11–3.13 is turned away, including people who only want to read the source |
| B. Future-import shim | Require `from __future__ import annotations` in modules that declare inline `fn` bodies, then lower the floor | Low: one documented rule, one test, plus a real version matrix | A module that forgets the import fails at import time with `NameError`; the rule is easy to miss because the failure is in the user's file, not in the CLI |
| C. Import-hook rewrite | A loader that rewrites `def` headers into decorator calls before execution, so annotations never exist | High: a custom finder/loader in the import path, plus debugging surface | Every traceback, `inspect.getsource` call and tool that reads source now depends on the rewrite behaving; `inline_lang` *itself* reads source, so the hook and the decorator would have to agree |
| D. Annotation backport | Vendor deferred-annotation semantics for 3.11–3.13 | Very high: patching evaluation timing of the host interpreter is not something a library can do safely | Fragile against any interpreter update; would make the project's behaviour differ from the host it runs on |

B is the only option whose failure mode is boring: a missing import statement in
a user module. C changes the import machinery that the whole project is built on
top of, and D is not a library-level operation at all.

## If B is attempted, in this order

1. **Measure before changing anything.** Create 3.12 and 3.13 environments, run
   `pytest -q -m core_only`, and write down every failure with its message. The
   claim "the floor only exists because of annotations" is currently an
   inference from reading the code, not a measured fact; `backends/hotpatch.py`
   copying annotations and any framework decorator that inspects signatures are
   exactly the places where that inference could be wrong.
2. **Prototype the shim, not the floor.** Add `from __future__ import annotations`
   to the modules that declare inline `fn` bodies, run the core-only set on the
   old interpreter, and record what still fails. Only a clean run justifies
   touching `MIN_VERSION`.
3. **Lower the floor in one place, with the matrix attached.** `MIN_VERSION`,
   `pyproject.toml`'s `requires-python`, SPEC 9.9.1, README's Host interpreter
   section and the changelog move together. The revision of 9.9.1 must name the
   interpreter versions actually exercised, because "works on 3.11" without a
   recorded run is the same mistake 10.57 already admits to.
4. **Keep the refusal path tested.** `tests/test_host_interpreter.py` proves the
   CLI rejects an unsupported interpreter; a lowered floor needs the same proof
   for the new one, plus a test that asserts the future-import rule is present
   in the modules that need it.

## What would make the floor stay

- A dependency on `annotationlib` or `typing` helpers that only exist in 3.14,
   or on 3.14-only syntax.
- A measured 3.11–3.13 failure that the future-import shim cannot reach — for
   example a decorator that turns annotations into runtime types before
   `inline_lang` sees the source.
- A native, WASM or Zig path that fails for a reason unrelated to annotations on
   the older interpreter. Lowering the Python-side floor while a backend silently
   breaks would trade one honest refusal for three dishonest partial passes.

Until one of those is measured, the floor is a declared requirement, and the
documentation says so in the same words as SPEC 9.9.1 and 10.57. A reader on
3.12 should expect the CLI's refusal, not a partial pass.
