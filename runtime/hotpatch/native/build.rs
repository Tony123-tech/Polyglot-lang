// macOS extension modules resolve CPython symbols from the interpreter that
// loads them; they must not link a second copy of libpython.  PyO3 0.26 leaves
// these flags to the build tool, so an explicit cargo build needs them too.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
