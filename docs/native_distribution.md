# Native distribution (CI configuration)

## Two separate distributions

The root project (`polyglot-lang`, currently 0.3.8) is a **pure-Python**
setuptools build: `pyproject.toml` at the repository root uses
`setuptools.build_meta` and ships Python modules/packages only. Installing that
wheel does **not** compile or bundle the PyO3 backend.

The Rust/PyO3 backend lives in the `native_pillars/` crate (`polyglot_pillars`,
PyO3 0.26, `cdylib`, `publish = false`). To use it today you either need a
**local Rust toolchain** (cargo) available so the backend can be compiled from
source, or you install a maturin-built `polyglot-pillars` wheel produced by the
native wheels workflow and made available to users. There is no automatic
fallback to a prebuilt native module in the pure-Python root wheel.

`native_pillars/pyproject.toml` is a separate, minimal **maturin** build
configuration (`polyglot-pillars` 0.1.0, `requires-python = ">=3.14"`,
`features = ["pyo3/extension-module"]`, `module-name = "polyglot_pillars"`,
`bindings = "pyo3"`, `strip = true`). It is not read by the root setuptools
build. Linux `manylinux` compatibility is intentionally **not** stored in that
file: the workflow passes `--compatibility linux` only on Linux, because that
flag is invalid for macOS and Windows builds.

## Workflows (CI-only)

Everything below is **CI-only configuration**. None of these steps run during a
local `pytest`, `ruff`, or release-gate invocation in this repository.

### `.github/workflows/release-gate.yml`

- Triggers: push to `main`, pull request, manual `workflow_dispatch`.
- Matrix: `macos-latest` and `ubuntu-latest`, Python 3.14, stable Rust, with
  `Swatinem/rust-cache` scoped to `native_pillars`.
- Installs `.[dev]` with pip and runs `python tools/release_gate.py --full`
  with a 45-minute timeout. `--allow-dirty` is deliberately omitted: CI
  starts from a clean checkout, so the worktree-cleanliness stage is meaningful.
- On failure, uploads `.artifacts/` as
  `release-gate-artifacts-<os>-py<version>` (fuzzer/gate reports).

The existing `.github/workflows/ci.yml` keeps all of its fast checks (ruff,
docs checks, pytest, slow pytest on macOS/Python 3.14), adds a `gate` job that
runs `python tools/release_gate.py --quick`, and gained a header comment
pointing at this workflow and the native wheels workflow. The full gate remains
the dedicated workflow above.

### `.github/workflows/native-wheels.yml`

- Triggers: manual `workflow_dispatch` and `release` (`published`).
- Build matrix (Python 3.14, stable Rust, maturin via pip):
  - `macos-latest` / `aarch64-apple-darwin`
  - `macos-latest` / `x86_64-apple-darwin`
  - `ubuntu-latest` / `x86_64-unknown-linux-gnu` (adds `--compatibility linux`)
  - `windows-latest` / `x86_64-pc-windows-msvc`
- Each job runs `maturin build --release --out dist --target ...` from
  `native_pillars`, then generates a CycloneDX SBOM with
  `cyclonedx-py environment --output-format JSON --output-file sbom.json`
  (`cyclonedx-py` is the PyPI alias distribution for `cyclonedx-bom`).
- Artifacts per matrix entry: `wheels-<platform>`, `sbom-<platform>`, and (on
  release events only) `signatures-<platform>`.

### SBOM and signing

Each build uploads its JSON SBOM alongside the wheel. Signing uses
`sigstore/gh-action-sigstore-python@v3` with `id-token: write` on the build job;
it is guarded by `if: github.event_name == 'release'`. **Manual
`workflow_dispatch` runs build wheels and SBOMs but are never signed.** On
release runs, the v3 action writes `<wheel>.sigstore.json` bundles next to the
wheels; those bundles are uploaded as `signatures-<platform>` artifacts. The
action's `release-signing-artifacts` input is disabled here, so no release
assets or repository `contents: write` permission are needed.

### Trusted publishing

The final `publish` job also runs only on `release` events, after all builds
succeed. It downloads the `wheels-*` artifacts into `dist/` and calls
`pypa/gh-action-pypi-publish@release/v1` with **OIDC trusted publishing**
(`permissions: id-token: write`, no API token or secret in the workflow). The
job targets a `pypi` GitHub environment; the repository owner must configure a
matching trusted publisher in PyPI project settings (repository, workflow
`native-wheels.yml`, environment `pypi`). No secret handling is performed in
any workflow file. The PyPI publish action has PEP 740 attestations enabled by
default, independently of the Sigstore step above.

## Scope of what was actually done

This change only adds/edits CI and packaging configuration plus this document.
**No multi-platform wheels were built locally, no signing took place, and no
real PyPI publish occurred.** Local verification was limited to parsing the
workflow YAML and checking the diff; the workflows themselves have not been
executed here.
