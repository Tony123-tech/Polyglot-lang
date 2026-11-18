//! Structured `E008` diagnostics for the native hot-patching primitives.
//!
//! The Python package raises `HotpatchError` (a `PillarError`) so callers see
//! one diagnostic pipeline.  The PyO3 surface mirrors that with a runtime
//! error whose text starts with `error[E008]` and, when a span is supplied,
//! prints both the character range and the byte range.

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// The relocation/ABI/journal failure code shared with the Python package.
pub const E008: &str = "E008";
/// The unsupported-platform/native-feature code.
pub const E007: &str = "E007";

pub fn formatted(
    code: &str,
    message: impl AsRef<str>,
    reason: impl AsRef<str>,
    span: Option<(usize, usize)>,
) -> String {
    let mut text = format!("error[{}]: {}", code, message.as_ref());
    if let Some((start, end)) = span {
        // Python callers work in characters; the native surface receives the
        // same range and reports it verbatim as bytes when the caller used
        // `byte_offsets()`.  Both labels are printed so the range cannot be
        // mistaken for a half-open character slice.
        text.push_str(&format!(
            "\n  span: {start}..{end} (characters)\n  bytes: {start}..{end}"
        ));
    }
    text.push_str(&format!("\n  reason: {}", reason.as_ref()));
    text
}

pub fn hotpatch_error(
    message: impl AsRef<str>,
    reason: impl AsRef<str>,
    span: Option<(usize, usize)>,
) -> PyErr {
    PyRuntimeError::new_err(formatted(E008, message, reason, span))
}

pub fn unsupported(
    message: impl AsRef<str>,
    reason: impl AsRef<str>,
    span: Option<(usize, usize)>,
) -> PyErr {
    PyRuntimeError::new_err(formatted(E007, message, reason, span))
}
