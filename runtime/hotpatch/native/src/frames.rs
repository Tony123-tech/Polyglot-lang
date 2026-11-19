//! Native CPython frame migration at a bytecode safe point.
//!
//! CPython 3.13+ exposes `FrameLocalsProxy` (PEP 667), which lets a holder of
//! the GIL read and write the fast locals of a *live* frame.  That is the
//! supported seam this module uses from Rust:
//!
//! * the target function's `__code__` is replaced for future entries;
//! * every live frame of that function has its fast locals copied into the
//!   new layout and written back through the proxy;
//! * suspended generators are migrated explicitly (they keep their captured
//!   code until recreated);
//! * argument, closure, generator, and stack-layout mismatches are rejected
//!   with `E008` and the caller's span.
//!
//! Replacing the `f_code` pointer of a frame that is *currently executing* an
//! instruction is deliberately refused (`E007`): `prev_instr`, the value
//! stack, and `localsplus` would desynchronise.  Existing frames are pinned to
//! the code object they started with.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::error::{hotpatch_error, unsupported};

const CO_VARARGS: i64 = 0x4;
const CO_VARKEYWORDS: i64 = 0x8;
const CO_GENERATOR: i64 = 0x20;
const CO_COROUTINE: i64 = 0x80;
const CO_ASYNC_GENERATOR: i64 = 0x200;
const GENERATOR_MASK: i64 = CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR;
const VARIADIC_MASK: i64 = CO_VARARGS | CO_VARKEYWORDS;

static GENERATION: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
struct Layout {
    argcount: usize,
    posonly: usize,
    kwonly: usize,
    // `co_nlocals` is implied by `varnames`/`freevars` for the checks we need.
    #[allow(dead_code)]
    nlocals: usize,
    stacksize: usize,
    flags: i64,
    varnames: Vec<String>,
    freevars: Vec<String>,
}

impl Layout {
    fn from_code(code: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            argcount: code.getattr("co_argcount")?.extract()?,
            posonly: code.getattr("co_posonlyargcount")?.extract()?,
            kwonly: code.getattr("co_kwonlyargcount")?.extract()?,
            nlocals: code.getattr("co_nlocals")?.extract()?,
            stacksize: code.getattr("co_stacksize")?.extract()?,
            flags: code.getattr("co_flags")?.extract()?,
            varnames: code.getattr("co_varnames")?.extract()?,
            freevars: code.getattr("co_freevars")?.extract()?,
        })
    }

    fn issues(&self, other: &Layout, restart: bool) -> Vec<String> {
        let mut issues = Vec::new();
        if (self.argcount, self.posonly, self.kwonly)
            != (other.argcount, other.posonly, other.kwonly)
        {
            issues.push(format!(
                "argument layout {}/{}/{} -> {}/{}/{}",
                self.argcount,
                self.posonly,
                self.kwonly,
                other.argcount,
                other.posonly,
                other.kwonly
            ));
        }
        if (self.flags & GENERATOR_MASK) != (other.flags & GENERATOR_MASK) {
            issues.push("generator/coroutine flags changed".to_string());
        }
        if (self.flags & VARIADIC_MASK) != (other.flags & VARIADIC_MASK) {
            issues.push("*args/**kwargs flags changed".to_string());
        }
        if self.freevars != other.freevars {
            issues.push(format!(
                "closure cells {:?} -> {:?} cannot be rebound",
                self.freevars, other.freevars
            ));
        }
        if restart && other.stacksize > self.stacksize {
            issues.push(format!(
                "evaluation stack {} -> {} cannot grow under a running frame",
                self.stacksize, other.stacksize
            ));
        }
        issues
    }
}

struct FrameSnapshot {
    frame: Py<PyAny>,
    values: Vec<(String, Py<PyAny>)>,
    added: Vec<String>,
}

fn code_of<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if object.hasattr("co_code")? {
        return Ok(object.clone());
    }
    if object.hasattr("__code__")? {
        return object.getattr("__code__");
    }
    Err(hotpatch_error(
        format!(
            "frame replacement must be a function or code object, got {}",
            object
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        ),
        "the replacement supplies the new bytecode and local layout",
        None,
    ))
}

fn resolve_function<'py>(target: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if target.hasattr("__code__")? {
        return Ok(target.clone());
    }
    if let Ok(text) = target.extract::<String>() {
        let (module_name, qualname) = text
            .split_once(':')
            .or_else(|| text.rsplit_once('.'))
            .ok_or_else(|| {
                hotpatch_error(
                    format!("cannot resolve frame target {text:?}"),
                    "expected 'module:qualname' or a function object",
                    None,
                )
            })?;
        let module = target.py().import(module_name)?;
        let mut attribute = module.into_any();
        for part in qualname.split('.') {
            attribute = attribute.getattr(part)?;
        }
        return Ok(attribute);
    }
    Err(hotpatch_error(
        format!(
            "frame target must be a function, got {}",
            target
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        ),
        "pass a function object or 'module:qualname'",
        None,
    ))
}

fn default_value(py: Python<'_>, name: &str) -> Py<PyAny> {
    let lowered = name.to_lowercase();
    if lowered.contains("count") || lowered.contains("index") || lowered.contains("len") {
        return 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any()
            .unbind();
    }
    if lowered.starts_with("is_") || lowered.starts_with("has_") || lowered.starts_with("can_") {
        return PyBool::new(py, false).to_owned().into_any().unbind();
    }
    py.None()
}

fn migrate_frame(
    py: Python<'_>,
    frame: &Bound<'_, PyAny>,
    old: &Layout,
    new: &Layout,
    defaults: &HashMap<String, Py<PyAny>>,
    span: Option<(usize, usize)>,
) -> PyResult<(FrameSnapshot, usize)> {
    let locals = frame.getattr("f_locals")?;
    let mut current: Vec<(String, Py<PyAny>)> = Vec::new();
    for item in locals.call_method0("items")?.try_iter()? {
        let item = item?;
        let name: String = item.get_item(0)?.extract()?;
        let value: Py<PyAny> = item.get_item(1)?.extract()?;
        current.push((name, value));
    }
    let lookup: HashMap<&str, &Py<PyAny>> = current
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();

    let mut removed = Vec::new();
    for name in &old.varnames {
        if !new.varnames.contains(name) {
            if let Some(value) = lookup.get(name.as_str()) {
                if !value.bind(py).is_none() {
                    removed.push(name.clone());
                }
            }
        }
    }
    if !removed.is_empty() {
        return Err(hotpatch_error(
            format!("cannot drop live local(s) {removed:?}"),
            "removed locals still hold a value; preserve them in the new layout",
            span,
        ));
    }

    let mut migrated: Vec<(String, Py<PyAny>)> = Vec::new();
    let mut added = Vec::new();
    for name in &new.varnames {
        if let Some(value) = lookup.get(name.as_str()) {
            migrated.push((name.clone(), value.clone_ref(py)));
        } else if !old.varnames.contains(name) {
            migrated.push((
                name.clone(),
                defaults
                    .get(name)
                    .map(|value| value.clone_ref(py))
                    .unwrap_or_else(|| default_value(py, name)),
            ));
            added.push(name.clone());
        }
    }

    let snapshot = FrameSnapshot {
        frame: frame.clone().unbind(),
        values: current,
        added,
    };
    for (name, value) in &migrated {
        locals.set_item(name, value.bind(py)).map_err(|error| {
            unsupported(
                "this interpreter does not expose writable frame locals",
                format!("{error}; CPython 3.13+ (PEP 667) is required for live local migration"),
                span,
            )
        })?;
    }
    Ok((snapshot, migrated.len()))
}

/// Replay completed frame snapshots in reverse order after a failed patch.
fn restore_snapshots(py: Python<'_>, snapshots: &[FrameSnapshot]) {
    for snapshot in snapshots.iter().rev() {
        let Ok(locals) = snapshot.frame.bind(py).getattr("f_locals") else {
            continue;
        };
        for (name, value) in &snapshot.values {
            let _ = locals.set_item(name, value.bind(py));
        }
        for name in &snapshot.added {
            let _ = locals.call_method1("__delitem__", (name,));
        }
    }
}

fn collect_frames(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    let sys = py.import("sys")?;
    let current = sys.call_method0("_current_frames")?;
    let mut frames = Vec::new();
    for entry in current.call_method0("values")?.try_iter()? {
        let mut cursor = entry?;
        loop {
            frames.push(cursor.clone());
            let back = cursor.getattr("f_back")?;
            if back.is_none() {
                break;
            }
            cursor = back;
        }
    }
    Ok(frames)
}

fn generator_frame<'py>(generator: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    for attribute in ["gi_frame", "cr_frame", "ag_frame"] {
        if let Ok(frame) = generator.getattr(attribute) {
            if !frame.is_none() {
                return Ok(frame);
            }
        }
    }
    Err(hotpatch_error(
        "suspended generator has no frame",
        "the generator already finished or was never started",
        None,
    ))
}

/// Patch a function's code object and migrate live frame locals.
#[allow(clippy::too_many_arguments)]
fn patch_function(
    py: Python<'_>,
    target: &Bound<'_, PyAny>,
    replacement: &Bound<'_, PyAny>,
    defaults: Option<HashMap<String, Py<PyAny>>>,
    generators: Option<Vec<Py<PyAny>>>,
    restart: bool,
    span: Option<(usize, usize)>,
) -> PyResult<NativeFramePatch> {
    let function = resolve_function(target)?;
    let old_code = function.getattr("__code__")?;
    let new_code = code_of(replacement)?;
    let old_layout = Layout::from_code(&old_code)?;
    let new_layout = Layout::from_code(&new_code)?;
    let issues = old_layout.issues(&new_layout, restart);
    if !issues.is_empty() {
        return Err(hotpatch_error(
            format!(
                "frame patch for {} is not ABI-compatible",
                function
                    .getattr("__qualname__")
                    .and_then(|name| name.extract::<String>())
                    .unwrap_or_else(|_| "<anonymous>".to_string())
            ),
            issues.join("; "),
            span,
        ));
    }

    let defaults = defaults.unwrap_or_default();
    let generators = generators.unwrap_or_default();
    let mut snapshots = Vec::new();
    let mut migrated_locals = 0usize;
    let mut pinned_frames = 0usize;

    // All mutations happen inside one fallible block.  If anything fails
    // part-way (a later generator, for example), replay every snapshot so a
    // rejected patch cannot leave half-migrated frames behind.  The handle is
    // constructed only after the block succeeds, so `snapshots` stays available
    // for rollback on the error path.
    let outcome: PyResult<usize> = (|| -> PyResult<usize> {
        for frame in collect_frames(py)? {
            let code = frame.getattr("f_code")?;
            if !code.is(&old_code) {
                continue;
            }
            let (snapshot, count) = migrate_frame(
                py,
                &frame,
                &old_layout,
                &new_layout,
                &defaults,
                span,
            )?;
            snapshots.push(snapshot);
            migrated_locals += count;
            pinned_frames += 1;
        }

        for generator in &generators {
            let generator = generator.bind(py);
            let frame = generator_frame(generator).map_err(|error| {
                hotpatch_error(
                    "generator frame does not match the patched function",
                    error.to_string(),
                    span,
                )
            })?;
            let code = frame.getattr("f_code")?;
            if !code.is(&old_code) {
                return Err(hotpatch_error(
                    "generator frame does not match the patched function",
                    format!(
                        "got {}",
                        code.getattr("co_qualname")
                            .and_then(|name| name.extract::<String>())
                            .unwrap_or_else(|_| "<anonymous>".to_string())
                    ),
                    span,
                ));
            }
            let (snapshot, count) = migrate_frame(
                py,
                &frame,
                &old_layout,
                &new_layout,
                &defaults,
                span,
            )?;
            snapshots.push(snapshot);
            migrated_locals += count;
        }

        function.setattr("__code__", &new_code)?;
        Ok(GENERATION.fetch_add(1, Ordering::SeqCst) + 1)
    })();

    match outcome {
        Ok(generation) => Ok(NativeFramePatch {
            function: function.unbind(),
            old_code: old_code.unbind(),
            new_code: new_code.unbind(),
            snapshots,
            pinned_frames,
            migrated_locals,
            generation,
            rolled_back: false,
        }),
        Err(error) => {
            let _ = function.setattr("__code__", &old_code);
            restore_snapshots(py, &snapshots);
            Err(error)
        }
    }
}

/// The handle returned by [`hotpatch_cpython_frame_eval`].
#[pyclass(module = "polyglot_hotpatch")]
pub struct NativeFramePatch {
    function: Py<PyAny>,
    old_code: Py<PyAny>,
    new_code: Py<PyAny>,
    snapshots: Vec<FrameSnapshot>,
    #[pyo3(get)]
    pinned_frames: usize,
    #[pyo3(get)]
    migrated_locals: usize,
    #[pyo3(get)]
    generation: usize,
    rolled_back: bool,
}

#[pymethods]
impl NativeFramePatch {
    fn rollback(&mut self, py: Python<'_>) -> PyResult<usize> {
        if self.rolled_back {
            return Ok(0);
        }
        self.function
            .bind(py)
            .setattr("__code__", self.old_code.bind(py))?;
        let mut restored = 0usize;
        for snapshot in self.snapshots.iter().rev() {
            let frame = snapshot.frame.bind(py);
            let locals = frame.getattr("f_locals")?;
            for (name, value) in &snapshot.values {
                locals.set_item(name, value.bind(py))?;
            }
            for name in &snapshot.added {
                let _ = locals.call_method1("__delitem__", (name,));
            }
            restored += 1;
        }
        self.rolled_back = true;
        Ok(restored)
    }

    #[getter]
    fn active(&self) -> bool {
        !self.rolled_back
    }

    #[getter]
    fn old_code(&self, py: Python<'_>) -> Py<PyAny> {
        self.old_code.clone_ref(py)
    }

    #[getter]
    fn new_code(&self, py: Python<'_>) -> Py<PyAny> {
        self.new_code.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeFramePatch(generation={}, pinned={}, migrated={})",
            self.generation, self.pinned_frames, self.migrated_locals
        )
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<Bound<'_, PyAny>>,
        _exc: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if exc_type.is_some() {
            self.rollback(py)?;
        }
        Ok(())
    }
}

/// Native implementation of `hotpatch_cpython_frame_eval`.
#[pyfunction]
#[pyo3(signature = (target, replacement, defaults=None, generators=None, restart=false, span=None))]
pub fn hotpatch_cpython_frame_eval(
    py: Python<'_>,
    target: Bound<'_, PyAny>,
    replacement: Bound<'_, PyAny>,
    defaults: Option<HashMap<String, Py<PyAny>>>,
    generators: Option<Vec<Py<PyAny>>>,
    restart: bool,
    span: Option<(usize, usize)>,
) -> PyResult<NativeFramePatch> {
    patch_function(
        py,
        &target,
        &replacement,
        defaults,
        generators,
        restart,
        span,
    )
}

/// Pure helper used by tests: copy a frame's locals into a plain dict.
#[pyfunction]
pub fn native_frame_locals<'py>(frame: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(frame.py());
    for item in frame
        .getattr("f_locals")?
        .call_method0("items")?
        .try_iter()?
    {
        let item = item?;
        output.set_item(item.get_item(0)?, item.get_item(1)?)?;
    }
    Ok(output)
}
