# Examples

Every file here runs with the Python backend, no toolchain required:

```sh
polyglot run examples/hello.poly
```

## Start here

| File | What it shows |
| --- | --- |
| `hello.poly` | Smallest possible program: one function and `main()` |
| `vec_ops.poly` | Creating, mutating, sorting and indexing a `Vec` |
| `option_result.poly` | `Option`, `Result`, `?` and `if let` |
| `string_ops.poly` | String creation, splitting, joining and searching |
| `fib.poly` | Recursion and a `for` loop |
| `hashmap.poly` | `HashMap` key/value storage |
| `terminates.poly` | A `#[terminates]` function proven total by the checker |

## Things people actually build

| File | What it does |
| --- | --- |
| `csv_sum.poly` | Writes a small CSV, reads it back, sums a numeric column with `Result` |
| `word_count.poly` | Counts word frequencies with two parallel `Vec`s |
| `cli_report.poly` | A tiny report tool: file stats, line count, longest line |

## Experimental backends

| File | What it shows |
| --- | --- |
| `zig_parallel_simd.poly` | `#[parallel]` loops and `Simd4<f64>` on the Zig backend |

The Zig example needs a Zig toolchain and is documented in
[docs/advanced.md](../docs/advanced.md).
