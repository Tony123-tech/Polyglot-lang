# Interactive AST and monomorphization visualizer

`polyglot visualizer` (alias `polyglot gui`) starts a local web UI for
inspecting the front end and running the same program on both primary
backends. It uses only the Python standard library: no FastAPI, no uvicorn,
and no JavaScript dependencies.

```sh
polyglot visualizer
# http://127.0.0.1:8765/

polyglot visualizer --port 9000 --no-open
polyglot visualizer --host 0.0.0.0   # only on a trusted network
```

## What it shows

| Panel | Contents |
| --- | --- |
| Source | Editable `.poly` buffer. `⌘`/`Ctrl` + `Enter` runs both backends. |
| Monomorphization summary | Chips such as `Box → Box_i64` and `identity → identity_i64`; click a chip to jump to the concrete declaration. |
| AST | Collapsible tree with color-coded categories: functions, structs, enums, impl blocks, generics, types, patterns, literals, statements, and expressions. Click a node to select its source span; double-click to expand or collapse. |
| Monomorphization inspector | Side-by-side generic source and fully concrete output. Added and removed lines are tinted. |
| Diagnostics | E001–E008, syntax errors, and backend failures. Selecting a diagnostic highlights its exact character span in the editor. |
| Python backend | Console output, compile time, run time, peak RSS, exit code, and return value. |
| Native backend | The same metrics from the Rust + PyO3 extension. |

Analysis is debounced while typing. Changing the source marks previous run
results as stale; they are replaced on the next **Run both**.

## Architecture

```text
browser (embedded HTML/CSS/JS)
   │  POST /api/analyze          POST /api/run
   ▼                              ▼
api/visualizer_server.py     subprocess: python -m api.cli exec
   │                              ├── Python backend
   ▼                              └── Rust/PyO3 backend (cargo)
api/visualizer.py
   ├── lexer + parser → JSON AST
   ├── compiler/analysis/monomorph.py → concrete source printer
   └── E001–E009 diagnostic + span extraction
```

- `api/visualizer.py` is read-only. It parses, monomorphizes, serializes the
  AST, prints concrete source, and derives character spans from structured
  diagnostics.
- `api/visualizer_server.py` serves the page and API with
  `ThreadingHTTPServer`. `/api/run` starts `api.cli exec` in a new process
  group with a timeout, so user programs never run inside the server.
- `api/cli.py exec` writes timing, peak RSS (`resource.getrusage`), return
  value, and diagnostics to `--result-file`. Program stdout is kept separate
  from metadata.
- `api/static/visualizer.html` contains the entire frontend. The AST tree is
  built lazily, mono output uses a line diff, and editor highlights are drawn
  on a mirror layer behind a transparent textarea.

## API

| Method | Path | Body / result |
| --- | --- | --- |
| `GET` | `/` | The visualizer page. |
| `GET` | `/api/health` | Version, Python version, cargo availability, backend list. |
| `POST` | `/api/analyze` | `{"source": "..."}` → `{ok, original, monomorphized, diagnostics, timing}`. |
| `POST` | `/api/run` | `{"source": "...", "backends": ["python", "native"], "timeout": 180}` → `{ok, results, duration_ms}`. |

`/api/run` validates the backend allowlist and clamps timeouts to 1–600
seconds. Bodies larger than 2 MiB and sources larger than 512 KiB are
rejected. A timed-out backend is killed by process group and reported as
`timed_out: true` with an explanatory diagnostic.

The analysis route is available even when the source has errors: parse and
monomorphization failures return `ok: false` with structured diagnostics
instead of an HTTP error. Malformed requests return HTTP 400.

## Security model

The visualizer is a developer tool for trusted local code:

- It binds to `127.0.0.1` by default. `--host` is an explicit opt-in.
- `/api/analyze` never executes user code.
- `/api/run` executes code in a child Python process with a timeout, but
  that is crash isolation, not an OS sandbox. Do not expose it to untrusted
  networks.
- The UI labels the run path as trusted local execution.

For isolation levels A/B/C, see the security notes in
[NOTES.md](../NOTES.md#security).

## Tests

`tests/test_visualizer.py` covers:

- AST serialization and category coverage.
- Generic instantiation summaries and concrete source rendering.
- Syntax and monomorphization diagnostics with spans.
- Static file serving, health, analyze, and run endpoints.
- Unknown-backend validation and timeout handling.
- `api.cli exec` timing, memory, return values, and coded diagnostics.
- Package data for `api/static/*.html`.

Run just this slice with:

```sh
pytest -q tests/test_visualizer.py
```
