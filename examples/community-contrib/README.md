# Community examples

Programs contributed by people other than the original author. They live in a
subdirectory so the top-level `examples/` set stays curated: `examples/*.poly` is
checked by `tests/test_f1_cli.py` as the documented starting set, while anything
here is community material.

## Current examples

| File | What it shows |
| --- | --- |
| `times_table.poly` | A `while` loop, a mutable binding, and `println!` with three placeholders |

Run one:

```sh
polyglot check examples/community-contrib/times_table.poly
polyglot run examples/community-contrib/times_table.poly
```

## Rules for a contributed example

1. It must run on the Python backend with no toolchain installed:
   `polyglot check` and `polyglot run` both succeed on a machine without
   `cargo`, `rustc` or `zig`. Examples that need a toolchain belong under
   `examples/` only if the maintainers ask for them, or in your own PR comment.
2. First line is a `//` comment saying what the program demonstrates — not what
   it is called.
3. File name is `snake_case.poly`, and the program has a `main()` that returns
   `i64`.
4. No `include!("std:...")` of a module you added in the same PR: community
   examples use the shipped `stdlib` and the shipped language surface.
5. Add a row to the table above. A file that is not listed there is not
   discoverable, and the pull request will be asked to add it.
6. Commit with `docs(examples): add <file> showing <feature>` and stage by path
   (`git add examples/community-contrib/<file>.poly`), per
   [COMMIT_GUIDE.md](../../COMMIT_GUIDE.md).

If an example is wrong about the language — it compiles but teaches the wrong
thing — that is a documentation bug: open an issue against the example rather
than a PR that quietly rewrites it.
