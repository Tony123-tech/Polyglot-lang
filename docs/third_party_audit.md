# Third-party audit (read-only surface review)

This report records a bounded, read-only inspection of three third-party surfaces
referenced by this project. All findings were produced by parsing files on disk
(AST or plain text); no third-party module was imported and no external file was
written. The automated checks live in `tests/test_third_party_audit.py`.

## 1. dfmacro surface audit

| Item | Observed value |
| --- | --- |
| Checkout (module path) | `/Users/waika/dfmacro` — separate sibling repository, outside this workspace |
| Package entry point | `dfmacro/__init__.py` inside that checkout |
| Declared version | `0.0.1` (`pyproject.toml`, `[project].version`) |
| Installed metadata in this workspace venv | Not installed; the declared version is therefore the authoritative value for this audit |
| Build backend | `setuptools.build_meta` (`setuptools>=68`) |
| `__all__` size | 13 entries, no duplicates |
| Missing-export check | PASS — all 13 declared names are defined at module scope in `dfmacro/__init__.py` |
| Generated-method files | `generate_methods.py` and `dfmacro/_generated_methods.py` both present |
| Hard dependency | `pandas>=1.5` |
| Optional dependencies | `polars`, `dask[dataframe]`, `duckdb`, `pyarrow`, plus test/build/rust extras |
| Checkout state | Dirty worktree: three modified tracked files. Individual user edits are intentionally not enumerated here. |
| Scope decision | **Read-only** — inspected on disk only; no changes made |

The 13 declared exports are:

`DF`, `ColumnExpr`, `Literal`, `compile_to_sql`, `enable`, `generate_pytest_code`,
`profile`, `register_accessor`, `register_backend`, `save_profile_report`,
`validate_dataframe`, `when`, `write_stub_to_file`.

`dfmacro/__init__.py` has 17 AST-visible top-level names. Besides the 13 exports,
it also binds the internal names `PackageNotFoundError`, `_version`, `__all__`,
and `load_ipython_extension`; those are implementation details and are correctly
absent from `__all__`.

Because `pandas` is a hard dependency and is not installed in this workspace's
venv, the package was **not imported**. All conclusions below come from
`ast.parse` over the source files.

## 2. Generated-method parity check (AST, no pandas import)

Method:

1. Parse `dfmacro/__init__.py`; collect the string literals in the single
   top-level `__all__` assignment and all AST-visible module-level names
   (imports, functions, classes, assignments).
2. Parse `dfmacro/_generated_methods.py`; collect true `FunctionDef` /
   `AsyncFunctionDef` nodes anywhere in the file, and parse the top-level
   `ACCESSOR_METHODS` dictionary literal. Each key is a `(namespace, method)`
   tuple; the second element is treated as a generated accessor method name.
3. Compare both name sets with `__all__`. Nothing is executed or imported.

Actual result of the inspection performed for this report:

| Check | Result |
| --- | --- |
| `__all__` entries | 13 |
| Duplicate entries in `__all__` | 0 |
| Top-level names in `__init__.py` | 17 |
| `__all__` names missing from top-level definitions | **0** |
| Python `def` nodes in `_generated_methods.py` | **0** |
| `ACCESSOR_METHODS` registry entries | 175 |
| Distinct generated accessor method names | 164 |
| Generated method names also present in `__all__` | 0 |
| Generated names also present in `__init__.py` top-level names | 0 |

The registry covers six accessor namespaces: `cat` (12 entries), `dt` (45),
`list` (22), `num` (39), `str` (51), and `struct` (6). Some method names recur
in different namespaces (for example `ceil`, `contains`, `floor`, `get`, `head`,
`join`, `len`, `normalize`, `round`, `slice`, `tail`), which is why 175 entries
map to 164 distinct names; this is expected, not drift.

Distinct generated method names (sorted, as observed):

```text
abs add add_categories arccos arcsin arctan as_ordered as_unit as_unordered capitalize casefold categories
ceil center clip codes components contains contains_all contains_any cosh count date day
day_name dayofweek dayofyear days days_in_month decode degrees div drop_nulls encode endswith eq
eval exp exp2 extract extractall field fields fill_null find first floor floordiv
fullmatch gather ge get get_categories get_dummies gt head hour index is_finite is_infinite
is_leap_year is_month_end is_month_start is_nan is_null is_year_end is_year_start isalnum isalpha isdigit islower isocalendar
isupper join json_encode last le len ljust log log10 log1p lower lstrip
lt match max mean median microsecond microseconds milliseconds min minute mod month
month_name mul nanosecond nanoseconds ne normalize ordered pad partition pow quarter radians
remove_categories remove_unused_categories removeprefix removesuffix rename_categories rename_fields reorder_categories repeat replace reverse rfind rindex
rjust round round_sig_figs rpartition rstrip second seconds set_categories sign sinh slice sort
split sqrt startswith std strftime strip sub sum tail tanh time timetz
title to_period to_pydatetime to_pytimedelta total_seconds translate tz tz_convert tz_localize unique unit unnest
upper var week weekday with_fields wrap year zfill
```

**Drift found:** none with respect to the declared exports — every `__all__`
entry resolves at module scope, with no duplicates. Two structural details are
worth recording explicitly so they are not mistaken for drift:

- `_generated_methods.py` contains **no Python function definitions**. It is a
  generated data table (`ACCESSOR_METHODS`, typed by `AccessorMethodSpec`) that
  is consumed by `dfmacro/expr.py` and `dfmacro/compiler/accessor.py`. The
  automated check therefore counts the registry's method names as the generated
  method-name set and additionally records that the real `def` count is zero.
- None of those internal accessor names appears in `__all__` or among the
  module-level names of `__init__.py`. That separation is by design: the public
  API and the accessor dispatch table are different surfaces.

## 3. PyFrameRestart

No code implementing or referencing `PyFrameRestart` exists to audit.

Search actually performed for this report:

- Root: `/Users/waika`
- Bounded depth: `maxdepth 3` (files whose path has at most three components
  below the root)
- Pruned directory names: `.venv`, `.git`, `node_modules`
- Fast pre-filter: source-like suffixes only (`.py`, `.pyi`, `.rs`, `.c`, `.h`,
  `.js`, `.ts`, `.jsx`, `.tsx`, `.sh`, `.go`, `.java`, `.rb`) and a 4 MiB
  per-file size cap; raw bytes searched for the literal identifier
- The audit test file itself is excluded because it necessarily contains the
  search term as a constant; prose documentation (`.md`/`.txt`) is excluded from
  the source-code assertion and assessed separately below
- A prior, wider pass (max depth 4) also found no implementation.

Result: **zero matches in source code**. Honest caveat: a literal-string sweep
that includes prose finds only self-referential documentation — two
cross-references (`README.md:204`, `CHANGELOG.md:20`) plus the occurrences in
this report itself. None is code, and none points at a third-party
implementation. The automated test asserts absence from source files, excludes
prose documentation from that assertion, and the expected state remains "no
code present".

## 4. Terminal UI (`?api=terminal`)

| Item | Observed value |
| --- | --- |
| Path | `/Users/waika/tony123-tech.github.io/pycause/app.js` (present) |
| Location | Outside the workspace root `/Users/waika/polyglot-lang`; `is_relative_to` check returned false. Note: for a test under `tests/`, `Path(__file__).parents[2]` is `/Users/waika`, the shared parent of both checkouts, not the workspace root. The automated check therefore uses `parents[1]`, the actual workspace root; a literal `parents[2]` assertion would be false for a sibling path and would be incorrect. |
| Sibling worktree state | Dirty: a modified tracked file and untracked paths, including `pycause/`. No edits were inspected or made. |
| Scope decision | **Out-of-scope** for code changes; existence checked only, contents not read or modified |

Telemetry for terminal-style diagnostics was instead added to **this** project's
CLI and visualizer (`api/cli`, `api/telemetry`, and the visualizer surface; see
`tests/test_telemetry.py` and `tests/test_visualizer_telemetry.py`). The external
UI repository is not part of this audit's change surface.

## 5. Scope decisions

| Target | Decision | Rationale | Changes made |
| --- | --- | --- | --- |
| `dfmacro` checkout | Read-only | Separate sibling repo with its own user modifications | None |
| Generated-method parity | Read-only | AST/text inspection avoids the pandas dependency and any code generation | None |
| `PyFrameRestart` | Out-of-scope | No implementation or usage exists; only self-referential doc mentions | None |
| Terminal UI (`pycause/app.js`) | Out-of-scope | Outside this workspace; dirty sibling worktree | None |
| This repository's audit report and tests | In-scope | Two new files only | `docs/third_party_audit.md`, `tests/test_third_party_audit.py` |

**Statement:** no external files were modified. No repository other than
`/Users/waika/polyglot-lang` was written to, and within this repository only the
two files listed above were created. The automated checks are read-only and do
not import third-party packages or embed absolute paths in telemetry output.
