# Native hot-patching primitives

A PyO3 extension module implementing the Rust side of
`runtime/hotpatch`. See [`docs/quantum_hotpatch.md`](../../../docs/quantum_hotpatch.md)
for the guarantees and the deliberate boundaries.

## Build

```sh
PYO3_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python \
  CARGO_TARGET_DIR=$(git rev-parse --show-toplevel)/native_pillars/target/hotpatch \
  cargo build
```

The built artifact is `libpolyglot_hotpatch.dylib` on macOS or
`libpolyglot_hotpatch.so` on Linux. Copy or symlink it to
`polyglot_hotpatch.so` on `PYTHONPATH` to import it.

On macOS the module intentionally leaves CPython symbols unresolved
(`-undefined dynamic_lookup`, supplied by `build.rs` for a plain `cargo
build`). The interpreter that loads the module provides those symbols; linking
a second libpython into the extension would create two interpreters.

## Test

```sh
PYO3_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python \
  CARGO_TARGET_DIR=$(git rev-parse --show-toplevel)/native_pillars/target/hotpatch \
  cargo test --no-default-features
```

`--no-default-features` turns off `extension-module`, which lets the test
binary link against the configured interpreter. The tests exercise:

- owned executable regions and real machine-code sleds;
- atomic pointer-cell swaps and alignment rejection;
- value-scan GOT discovery for a bound libc symbol (Mach-O GOT-like sections
  or Linux writable mappings), not a full ELF relocation walk;
- RCU grace periods, reclamation, and unbalanced exits;
- before-image rollback, commit, and nested checkpoint order;
- `mio::Poll` migration with a datagram queued before the swap.

The Python suite compiles this crate only when
`POLYGLOT_BUILD_NATIVE_HOTPATCH=1` is set.
