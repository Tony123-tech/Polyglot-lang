//! PyO3 surface for the quantum hot-patching subsystem.
//!
//! The module exposes real native primitives:
//!
//! * [`ExecutableRegion`] - owned executable pages with AArch64/x86_64 sleds;
//! * [`native_patch_slot`] / [`NativeTrampolinePatch`] - atomic GOT-cell swaps;
//! * [`NativeRcuDomain`] - epoch reclamation that can retire Python objects;
//! * [`NativeJournal`] - before-image raw-range snapshots;
//! * [`NativeMioPoller`] - a live `mio::Poll` plus its interest registry;
//! * [`hotpatch_cpython_frame_eval`] - PEP 667 fast-local migration.
//!
//! `hotpatch_rcu_ast_epoch`, `hotpatch_rollback_journal_commit`, and the
//! selector branch of `hotpatch_mio_reactor_state` delegate to the Python host
//! bridge so both backends return exactly the same manager objects.  The
//! native mio branch is used when both arguments are [`NativeMioPoller`]s.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

mod error;
mod frames;
mod journal;
mod rcu;
mod reactor;
mod trampoline;

use error::{hotpatch_error, unsupported};
use frames::{hotpatch_cpython_frame_eval, native_frame_locals, NativeFramePatch};
use journal::Journal;
use rcu::RcuDomain;
use reactor::Reactor;
use trampoline::{ExecutableRegion, GotSlotInfo};

fn io_error(operation: &str, error: impl std::fmt::Display, span: Option<(usize, usize)>) -> PyErr {
    hotpatch_error(operation, error.to_string(), span)
}

// ---------------------------------------------------------------------------
// Executable regions and trampolines
// ---------------------------------------------------------------------------

/// An executable page range owned by this process.
#[pyclass(module = "polyglot_hotpatch", name = "ExecutableRegion")]
pub struct PyExecutableRegion {
    inner: ExecutableRegion,
}

#[pymethods]
impl PyExecutableRegion {
    #[new]
    #[pyo3(signature = (size=4096))]
    fn new(size: usize) -> PyResult<Self> {
        ExecutableRegion::new(size)
            .map(|inner| Self { inner })
            .map_err(|reason| unsupported("cannot allocate an executable region", reason, None))
    }

    #[getter]
    fn address(&self) -> usize {
        self.inner.address()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn is_jit(&self) -> bool {
        self.inner.is_jit()
    }

    fn write(&mut self, offset: usize, data: Vec<u8>) -> PyResult<()> {
        self.inner
            .write(offset, &data)
            .map_err(|reason| hotpatch_error("cannot write machine code", reason, None))
    }

    fn read(&self, py: Python<'_>, offset: usize, size: usize) -> PyResult<Py<PyBytes>> {
        let data = self
            .inner
            .read(offset, size)
            .map_err(|reason| hotpatch_error("cannot read machine code", reason, None))?;
        Ok(PyBytes::new(py, &data).unbind())
    }

    fn close(&mut self) {
        self.inner.close();
    }

    fn __int__(&self) -> usize {
        self.inner.address()
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) {
        self.inner.close();
    }
}

/// A handle for a GOT-cell or sled patch.
#[pyclass(module = "polyglot_hotpatch", name = "NativeTrampolinePatch")]
pub struct NativeTrampolinePatch {
    #[pyo3(get)]
    slot: u64,
    #[pyo3(get)]
    old_value: u64,
    #[pyo3(get)]
    new_value: u64,
    #[pyo3(get)]
    symbol: String,
    #[pyo3(get)]
    image: String,
    #[pyo3(get)]
    section: String,
    #[pyo3(get)]
    kind: String,
    region: Option<Py<PyExecutableRegion>>,
    offset: usize,
    original: Option<Vec<u8>>,
    active: bool,
}

#[pymethods]
impl NativeTrampolinePatch {
    #[getter]
    fn active(&self) -> bool {
        self.active
    }

    /// Restore the previous cell value or the original machine code.
    fn rollback(&mut self, py: Python<'_>) -> PyResult<bool> {
        if !self.active {
            return Ok(false);
        }
        if let Some(region) = &self.region {
            let original = self.original.clone().unwrap_or_default();
            region
                .bind(py)
                .borrow_mut()
                .inner
                .write(self.offset, &original)
                .map_err(|reason| hotpatch_error("cannot roll back a sled", reason, None))?;
        } else {
            trampoline::patch_pointer(self.slot as usize, self.old_value).map_err(|reason| {
                hotpatch_error("cannot roll back a pointer patch", reason, None)
            })?;
        }
        self.active = false;
        Ok(true)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeTrampolinePatch(kind={}, slot={:#x}, old={:#x}, new={:#x})",
            self.kind, self.slot, self.old_value, self.new_value
        )
    }
}

fn replacement_address(py: Python<'_>, replacement: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(value) = replacement.extract::<u64>() {
        return Ok(value);
    }
    if replacement.hasattr("restype")? {
        let ctypes = py.import("ctypes")?;
        let cast = ctypes
            .getattr("cast")?
            .call1((replacement, ctypes.getattr("c_void_p")?))?;
        return cast.getattr("value")?.extract::<u64>().map_err(|_| {
            hotpatch_error("replacement has no address", "NULL function pointer", None)
        });
    }
    Err(unsupported(
        "a Python callable replacement needs the host bridge",
        "pass a CFUNCTYPE instance, an integer address, or call the Python implementation",
        None,
    ))
}

fn verify_signature(
    py: Python<'_>,
    replacement: &Bound<'_, PyAny>,
    expected: &str,
) -> PyResult<()> {
    let helper = py
        .import("runtime.hotpatch.trampoline")?
        .getattr("_signature_matches")?;
    let matches: bool = helper.call1((replacement, expected))?.extract()?;
    if matches {
        Ok(())
    } else {
        Err(hotpatch_error(
            "replacement ABI signature does not match the expected signature",
            format!("expected {}", expected.replace(' ', "")),
            None,
        ))
    }
}

fn delegate_trampoline(
    py: Python<'_>,
    symbol: Option<String>,
    replacement: Option<Bound<'_, PyAny>>,
    slot: Option<u64>,
    region: Option<Bound<'_, PyAny>>,
    offset: usize,
    signature: Option<String>,
    expected_signature: Option<String>,
    library: Option<String>,
    span: Option<(usize, usize)>,
    source: Option<String>,
) -> PyResult<Py<PyAny>> {
    let module = py.import("runtime.hotpatch.trampoline")?;
    let function = module.getattr("hotpatch_live_dylib_trampoline")?;
    let kwargs = PyDict::new(py);
    if let Some(symbol) = symbol {
        kwargs.set_item("symbol", symbol)?;
    }
    if let Some(replacement) = replacement {
        kwargs.set_item("replacement", replacement)?;
    }
    if let Some(slot) = slot {
        kwargs.set_item("slot", slot)?;
    }
    if let Some(region) = region {
        kwargs.set_item("region", region)?;
    }
    kwargs.set_item("offset", offset)?;
    if let Some(signature) = signature {
        kwargs.set_item("signature", signature)?;
    }
    if let Some(expected) = expected_signature {
        kwargs.set_item("expected_signature", expected)?;
    }
    if let Some(library) = library {
        kwargs.set_item("library", library)?;
    }
    if let Some(span) = span {
        kwargs.set_item("span", span)?;
    }
    if let Some(source) = source {
        kwargs.set_item("source", source)?;
    }
    Ok(function.call((), Some(&kwargs))?.unbind())
}

/// Patch a bound GOT cell or write an inline sled into an owned region.
#[pyfunction]
#[pyo3(signature = (symbol=None, replacement=None, *, slot=None, region=None, offset=0, signature=None, expected_signature=None, library=None, span=None, source=None))]
#[allow(clippy::too_many_arguments)]
pub fn hotpatch_live_dylib_trampoline(
    py: Python<'_>,
    symbol: Option<String>,
    replacement: Option<Bound<'_, PyAny>>,
    slot: Option<u64>,
    region: Option<Py<PyExecutableRegion>>,
    offset: usize,
    signature: Option<String>,
    expected_signature: Option<String>,
    library: Option<String>,
    span: Option<(usize, usize)>,
    source: Option<String>,
) -> PyResult<Py<PyAny>> {
    let Some(replacement) = replacement else {
        return Err(hotpatch_error(
            "no replacement was supplied",
            "pass a callable, CFUNCTYPE instance, or absolute address",
            span,
        ));
    };
    // Python callables are wrapped through the host bridge, which owns the
    // ctypes trampoline allocations and their keep-alive references.
    if !replacement.hasattr("restype")? && replacement.extract::<u64>().is_err() {
        return delegate_trampoline(
            py,
            symbol,
            Some(replacement),
            slot,
            region.map(|region| region.into_bound(py).into_any()),
            offset,
            signature,
            expected_signature,
            library,
            span,
            source,
        );
    }
    if let Some(expected) = expected_signature.as_deref() {
        if replacement.hasattr("restype")? {
            verify_signature(py, &replacement, expected)?;
        }
    }
    let address = replacement_address(py, &replacement)?;

    if let Some(region) = region {
        let sled = trampoline::encode_jump(address)
            .map_err(|reason| unsupported("inline trampolines are unavailable", reason, span))?;
        let original = region
            .bind(py)
            .borrow()
            .inner
            .read(offset, sled.len())
            .map_err(|reason| hotpatch_error("cannot read the sled target", reason, span))?;
        region
            .bind(py)
            .borrow_mut()
            .inner
            .write(offset, &sled)
            .map_err(|reason| hotpatch_error("cannot write the sled", reason, span))?;
        let patch = NativeTrampolinePatch {
            slot: region.bind(py).borrow().inner.address() as u64 + offset as u64,
            old_value: 0,
            new_value: address,
            symbol: "region".to_string(),
            image: String::new(),
            section: String::new(),
            kind: "sled".to_string(),
            region: Some(region),
            offset,
            original: Some(original),
            active: true,
        };
        return Ok(Py::new(py, patch)?.into_any());
    }

    let (target_slot, info) = if let Some(slot) = slot {
        (slot, None)
    } else if let Some(symbol) = symbol {
        let slots = trampoline::find_got_slots(&symbol).map_err(|reason| {
            hotpatch_error(format!("no GOT/PLT slot for {symbol:?}"), reason, span)
        })?;
        let filtered: Vec<GotSlotInfo> = slots
            .into_iter()
            .filter(|info| {
                library
                    .as_deref()
                    .map_or(true, |library| info.image.contains(library))
            })
            .collect();
        if filtered.is_empty() {
            return Err(hotpatch_error(
                format!("no GOT/PLT slot found for {symbol:?}"),
                "the symbol is not bound through a writable pointer cell",
                span,
            ));
        }
        let mut addresses = filtered.iter().map(|info| info.address).collect::<Vec<_>>();
        addresses.dedup();
        if addresses.len() > 1 {
            return Err(hotpatch_error(
                format!("symbol {symbol:?} has multiple GOT slots"),
                format!("candidates: {addresses:?}"),
                span,
            ));
        }
        (addresses[0], filtered.into_iter().next())
    } else {
        return Err(hotpatch_error(
            "no symbol, slot, or region was supplied",
            "specify where the trampoline should live",
            span,
        ));
    };
    let old = trampoline::patch_pointer(target_slot as usize, address)
        .map_err(|reason| hotpatch_error("cannot patch the pointer cell", reason, span))?;
    let patch = NativeTrampolinePatch {
        slot: target_slot,
        old_value: old,
        new_value: address,
        symbol: info
            .as_ref()
            .map(|info| info.symbol.clone())
            .unwrap_or_else(|| "slot".to_string()),
        image: info
            .as_ref()
            .map(|info| info.image.clone())
            .unwrap_or_default(),
        section: info
            .as_ref()
            .map(|info| info.section.clone())
            .unwrap_or_default(),
        kind: "pointer".to_string(),
        region: None,
        offset: 0,
        original: None,
        active: true,
    };
    Ok(Py::new(py, patch)?.into_any())
}

/// Patch an aligned pointer cell directly.
#[pyfunction]
pub fn native_patch_slot(slot: u64, value: u64) -> PyResult<u64> {
    trampoline::patch_pointer(slot as usize, value)
        .map_err(|reason| hotpatch_error("cannot patch the pointer cell", reason, None))
}

/// Read an aligned pointer cell.
#[pyfunction]
pub fn native_read_pointer(slot: u64) -> PyResult<u64> {
    if slot == 0 {
        return Err(hotpatch_error(
            "cannot read a NULL pointer cell",
            "NULL",
            None,
        ));
    }
    if slot as usize % std::mem::align_of::<usize>() != 0 {
        return Err(hotpatch_error(
            "pointer cell is not word-aligned",
            format!("{slot:#x} is not 8-byte aligned"),
            None,
        ));
    }
    // SAFETY: the caller owns the cell; this is the same contract as
    // `atomic_load_ptr` in the Python trampoline module.
    Ok(unsafe { trampoline::load_pointer(slot as usize) })
}

/// Discover bound GOT/PLT cells for `symbol`.
#[pyfunction]
pub fn native_find_got_slots(symbol: String) -> PyResult<Vec<(u64, String, String, String)>> {
    let slots = trampoline::find_got_slots(&symbol).map_err(|reason| {
        hotpatch_error(format!("no GOT/PLT slot for {symbol:?}"), reason, None)
    })?;
    Ok(slots
        .into_iter()
        .map(|info| (info.address, info.image, info.segment, info.section))
        .collect())
}

/// Encode the host architecture's absolute jump sled.
#[pyfunction]
pub fn native_encode_jump(py: Python<'_>, target: u64) -> PyResult<Py<PyBytes>> {
    let sled = trampoline::encode_jump(target)
        .map_err(|reason| unsupported("inline trampolines are unavailable", reason, None))?;
    Ok(PyBytes::new(py, &sled).unbind())
}

// ---------------------------------------------------------------------------
// RCU
// ---------------------------------------------------------------------------

/// A native RCU domain that can retire Python objects.
#[pyclass(module = "polyglot_hotpatch", name = "NativeRcuDomain")]
pub struct NativeRcuDomain {
    domain: Arc<RcuDomain>,
    retired_python: Mutex<Vec<(usize, Py<PyAny>)>>,
    reclaimed_python: AtomicUsize,
}

#[pymethods]
impl NativeRcuDomain {
    #[new]
    #[pyo3(signature = (name="ast"))]
    fn new(name: &str) -> Self {
        Self {
            domain: Arc::new(RcuDomain::new(name)),
            retired_python: Mutex::new(Vec::new()),
            reclaimed_python: AtomicUsize::new(0),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.domain.name.clone()
    }

    #[getter]
    fn epoch(&self) -> usize {
        self.domain.stats().epoch
    }

    /// Enter a read-side critical section on the calling thread.
    fn enter(&self) -> usize {
        self.domain.enter_raw()
    }

    /// Exit a read-side critical section on the calling thread.
    fn exit(&self) -> PyResult<()> {
        self.domain
            .exit_raw()
            .map_err(|reason| hotpatch_error("unbalanced RCU read guard", reason, None))
    }

    /// Retire a Python value; it is dropped after the grace period.
    fn retire(&self, value: Py<PyAny>) -> usize {
        let epoch = self.domain.stats().epoch;
        self.retired_python
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push((epoch, value));
        epoch
    }

    /// Wait for a grace period and drop eligible Python values.
    #[pyo3(signature = (timeout=5.0))]
    fn synchronize(&self, _py: Python<'_>, timeout: Option<f64>) -> PyResult<usize> {
        let target = self.domain.stats().epoch;
        let limit = timeout.map(|seconds| {
            if seconds.is_finite() && seconds >= 0.0 {
                Duration::from_secs_f64(seconds)
            } else {
                Duration::ZERO
            }
        });
        let reclaimed = self
            .domain
            .synchronize(limit)
            .map_err(|reason| hotpatch_error("RCU grace period timed out", reason, None))?;
        let dropped = {
            let mut retired = self
                .retired_python
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut keep = Vec::with_capacity(retired.len());
            let mut dropped = Vec::new();
            for (epoch, value) in std::mem::take(&mut *retired) {
                if epoch <= target {
                    dropped.push(value);
                } else {
                    keep.push((epoch, value));
                }
            }
            *retired = keep;
            dropped
        };
        let count = dropped.len();
        // Dropping `Py<PyAny>` may release interpreter resources; the GIL is
        // held by the caller for this whole method.
        drop(dropped);
        self.reclaimed_python.fetch_add(count, Ordering::Relaxed);
        Ok(reclaimed + count)
    }

    /// Retire every pending value (repeated grace periods until empty).
    #[pyo3(signature = (timeout=5.0))]
    fn drain(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<usize> {
        let mut total = 0;
        loop {
            let remaining = self
                .retired_python
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .len();
            if remaining == 0 && self.domain.stats().retired == 0 {
                return Ok(total);
            }
            total += self.synchronize(py, timeout)?;
        }
    }

    /// `(epoch, active_readers, retired, reclaimed, grace_periods)`.
    fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let stats = self.domain.stats();
        let python_retired = self
            .retired_python
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len();
        (
            stats.epoch,
            stats.active_readers,
            stats.retired + python_retired,
            stats.reclaimed + self.reclaimed_python.load(Ordering::Relaxed),
            stats.grace_periods,
        )
    }

    fn __repr__(&self) -> String {
        let stats = self.stats();
        format!(
            "NativeRcuDomain(name={:?}, epoch={}, readers={}, retired={})",
            self.domain.name, stats.0, stats.1, stats.2
        )
    }
}

/// Create a native RCU domain.
#[pyfunction]
#[pyo3(signature = (name="ast"))]
pub fn native_rcu_domain(name: &str) -> NativeRcuDomain {
    NativeRcuDomain::new(name)
}

/// Python host bridge for AST epoch management (returns `AstEpochManager`).
#[pyfunction]
#[pyo3(signature = (program, *, span=None, source=None))]
pub fn hotpatch_rcu_ast_epoch(
    py: Python<'_>,
    program: Bound<'_, PyAny>,
    span: Option<(usize, usize)>,
    source: Option<String>,
) -> PyResult<Py<PyAny>> {
    let module = py.import("runtime.hotpatch.rcu")?;
    let kwargs = PyDict::new(py);
    if let Some(span) = span {
        kwargs.set_item("span", span)?;
    }
    if let Some(source) = source {
        kwargs.set_item("source", source)?;
    }
    Ok(module
        .getattr("hotpatch_rcu_ast_epoch")?
        .call((program,), Some(&kwargs))?
        .unbind())
}

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------

/// A native before-image journal over raw memory ranges.
#[pyclass(module = "polyglot_hotpatch", name = "NativeJournal")]
pub struct NativeJournal {
    inner: Journal,
}

#[pymethods]
impl NativeJournal {
    #[new]
    fn new() -> Self {
        Self {
            inner: Journal::new(),
        }
    }

    /// Capture `[address, address + size)` before mutating it.
    #[pyo3(signature = (address, size, label=""))]
    fn capture(&self, address: u64, size: usize, label: &str) -> PyResult<()> {
        // SAFETY: the caller registers a live range; the Python helper
        // `runtime.hotpatch.journal` derives addresses from buffers it keeps
        // alive for the duration of the checkpoint.
        unsafe { self.inner.capture(address as usize, size, label) }
            .map(|_| ())
            .map_err(|reason| hotpatch_error("cannot capture a before-image", reason, None))
    }

    fn rollback(&self) -> usize {
        self.inner.rollback()
    }

    fn commit(&self) -> usize {
        self.inner.commit()
    }

    fn verify(&self) -> bool {
        self.inner.verify()
    }

    fn digest(&self) -> u64 {
        self.inner.digest()
    }

    #[getter]
    fn pending(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn captured_bytes(&self) -> usize {
        self.inner.captured_bytes()
    }

    #[getter]
    fn commits(&self) -> usize {
        self.inner.commits()
    }

    #[getter]
    fn rollbacks(&self) -> usize {
        self.inner.rollbacks()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeJournal(pending={}, bytes={}, commits={}, rollbacks={})",
            self.inner.len(),
            self.inner.captured_bytes(),
            self.inner.commits(),
            self.inner.rollbacks()
        )
    }
}

/// Create a native journal.
#[pyfunction]
pub fn native_journal() -> NativeJournal {
    NativeJournal::new()
}

/// Python host bridge for object/buffer journaling.
#[pyfunction]
#[pyo3(signature = (journal=None, *, span=None, source=None))]
pub fn hotpatch_rollback_journal_commit(
    py: Python<'_>,
    journal: Option<Bound<'_, PyAny>>,
    span: Option<(usize, usize)>,
    source: Option<String>,
) -> PyResult<Py<PyAny>> {
    if let Some(journal) = &journal {
        if let Ok(native) = journal.extract::<Py<NativeJournal>>() {
            return Ok(native.into_any());
        }
    }
    let module = py.import("runtime.hotpatch.journal")?;
    let kwargs = PyDict::new(py);
    if let Some(journal) = journal {
        kwargs.set_item("journal", journal)?;
    }
    if let Some(span) = span {
        kwargs.set_item("span", span)?;
    }
    if let Some(source) = source {
        kwargs.set_item("source", source)?;
    }
    Ok(module
        .getattr("hotpatch_rollback_journal_commit")?
        .call((), Some(&kwargs))?
        .unbind())
}

// ---------------------------------------------------------------------------
// mio reactor
// ---------------------------------------------------------------------------

/// A live `mio::Poll` plus its interest registry.
#[pyclass(module = "polyglot_hotpatch", name = "NativeMioPoller", unsendable)]
pub struct NativeMioPoller {
    inner: Reactor,
}

#[pymethods]
impl NativeMioPoller {
    #[new]
    fn new() -> PyResult<Self> {
        Reactor::new()
            .map(|inner| Self { inner })
            .map_err(|reason| io_error("cannot create a mio poller", reason, None))
    }

    #[getter]
    fn poll_fd(&self) -> Option<i64> {
        self.inner.poll_fd().map(|fd| fd as i64)
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn register(&mut self, fd: i64, token: u64, readable: bool, writable: bool) -> PyResult<()> {
        self.inner
            .register(fd as i32, token, readable, writable)
            .map_err(|reason| io_error("cannot register a reactor descriptor", reason, None))
    }

    fn deregister(&mut self, fd: i64) -> PyResult<()> {
        self.inner
            .deregister(fd as i32)
            .map_err(|reason| io_error("cannot deregister a reactor descriptor", reason, None))
    }

    /// `[(fd, token, readable, writable), ...]`
    fn snapshot(&self) -> Vec<(i64, u64, bool, bool)> {
        self.inner
            .snapshot()
            .into_iter()
            .map(|registration| {
                (
                    registration.fd as i64,
                    registration.token,
                    registration.readable,
                    registration.writable,
                )
            })
            .collect()
    }

    /// Adopt every registration from `other`, leaving its sockets open.
    fn adopt_from(&mut self, other: PyRef<'_, NativeMioPoller>) -> PyResult<usize> {
        self.inner
            .adopt_from(&other.inner)
            .map_err(|reason| io_error("cannot adopt reactor registrations", reason, None))
    }

    /// Poll for readiness; `timeout_ms=None` blocks.
    fn wait(&mut self, timeout_ms: Option<u64>) -> PyResult<Vec<(u64, bool, bool)>> {
        let readiness = self
            .inner
            .wait(timeout_ms.map(Duration::from_millis))
            .map_err(|reason| io_error("reactor poll failed", reason, None))?;
        Ok(readiness
            .into_iter()
            .map(|event| (event.token, event.readable, event.writable))
            .collect())
    }

    /// Close the poll descriptor; registered sockets stay open.
    fn close(&mut self) {
        self.inner.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeMioPoller(fd={:?}, registrations={})",
            self.inner.poll_fd(),
            self.inner.snapshot().len()
        )
    }
}

/// The result of a native poller migration.
#[pyclass(module = "polyglot_hotpatch", name = "NativeMioMigration", unsendable)]
pub struct NativeMioMigration {
    old_poller: Option<Py<NativeMioPoller>>,
    new_poller: Py<NativeMioPoller>,
    #[pyo3(get)]
    registrations: Vec<(i64, u64, bool, bool)>,
    #[pyo3(get)]
    old_poller_fd: i64,
    #[pyo3(get)]
    new_poller_fd: i64,
    committed: bool,
    rolled_back: bool,
}

#[pymethods]
impl NativeMioMigration {
    #[getter]
    fn packet_loss(&self) -> usize {
        0
    }

    #[getter]
    fn committed(&self) -> bool {
        self.committed
    }

    #[getter]
    fn rolled_back(&self) -> bool {
        self.rolled_back
    }

    /// Close the old poller.  No socket is closed.
    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.rolled_back {
            return Err(hotpatch_error(
                "cannot commit a rolled-back reactor migration",
                "create a new migration from the active poller",
                None,
            ));
        }
        if !self.committed {
            if let Some(old) = self.old_poller.take() {
                old.bind(py).borrow_mut().close();
            }
            self.committed = true;
        }
        Ok(())
    }

    /// Discard the new poller; the old one still owns every registration.
    fn rollback(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.committed {
            return Err(hotpatch_error(
                "cannot roll back a committed reactor migration",
                "keep the migration object alive until verification passes",
                None,
            ));
        }
        self.new_poller.bind(py).borrow_mut().close();
        self.rolled_back = true;
        Ok(())
    }
}

/// Migrate a native poller or delegate to the Python selector bridge.
#[pyfunction]
#[pyo3(signature = (source, target=None, *, duplicate_fds=false, span=None, source_text=None))]
pub fn hotpatch_mio_reactor_state(
    py: Python<'_>,
    source: Bound<'_, PyAny>,
    target: Option<Bound<'_, PyAny>>,
    duplicate_fds: bool,
    span: Option<(usize, usize)>,
    source_text: Option<String>,
) -> PyResult<Py<PyAny>> {
    let old_native = source.extract::<Py<NativeMioPoller>>();
    let new_native = target
        .as_ref()
        .and_then(|target| target.extract::<Py<NativeMioPoller>>().ok());
    if let (Ok(old), Some(new)) = (old_native, new_native) {
        let (entries, old_fd, new_fd) = {
            let old_ref = old.bind(py).borrow();
            let new_ref = new.bind(py).borrow();
            let old_fd = old_ref.inner.poll_fd().ok_or_else(|| {
                hotpatch_error("old poller is closed", "no poll descriptor", span)
            })?;
            let new_fd = new_ref.inner.poll_fd().ok_or_else(|| {
                hotpatch_error("new poller is closed", "no poll descriptor", span)
            })?;
            if old_fd == new_fd {
                return Err(hotpatch_error(
                    "the replacement reactor reused the live poll descriptor",
                    "both modules would close the same epoll/kqueue fd",
                    span,
                ));
            }
            let entries: Vec<(i64, u64, bool, bool)> = old_ref
                .inner
                .snapshot()
                .into_iter()
                .map(|registration| {
                    (
                        registration.fd as i64,
                        registration.token,
                        registration.readable,
                        registration.writable,
                    )
                })
                .collect();
            drop(new_ref);
            let mut new_ref = new.bind(py).borrow_mut();
            new_ref
                .inner
                .adopt_from(&old_ref.inner)
                .map_err(|reason| io_error("cannot adopt reactor registrations", reason, span))?;
            (entries, old_fd as i64, new_fd as i64)
        };
        return Ok(Py::new(
            py,
            NativeMioMigration {
                old_poller: Some(old),
                new_poller: new,
                registrations: entries,
                old_poller_fd: old_fd,
                new_poller_fd: new_fd,
                committed: false,
                rolled_back: false,
            },
        )?
        .into_any());
    }
    // The selector path (and any bridge object) is implemented once, in
    // Python, so both backends return identical migration handles.
    let module = py.import("runtime.hotpatch.reactor")?;
    let kwargs = PyDict::new(py);
    if let Some(target) = target {
        kwargs.set_item("target", target)?;
    }
    kwargs.set_item("duplicate_fds", duplicate_fds)?;
    if let Some(span) = span {
        kwargs.set_item("span", span)?;
    }
    if let Some(source_text) = source_text {
        kwargs.set_item("source_text", source_text)?;
    }
    Ok(module
        .getattr("hotpatch_mio_reactor_state")?
        .call((source,), Some(&kwargs))?
        .unbind())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Report which native primitives are usable in this process.
#[pyfunction]
fn native_platform_support(py: Python<'_>) -> PyResult<HashMap<String, bool>> {
    let mut support = HashMap::new();
    support.insert(
        "executable_region".to_string(),
        ExecutableRegion::new(1).is_ok(),
    );
    support.insert(
        "got_discovery".to_string(),
        trampoline::resolve_symbol("malloc").is_some(),
    );
    let frame_locals = py
        .import("sys")
        .and_then(|sys| sys.getattr("version_info"))
        .and_then(|version| {
            let major: u8 = version.getattr("major")?.extract()?;
            let minor: u8 = version.getattr("minor")?.extract()?;
            Ok((major, minor))
        })
        .map(|(major, minor)| major > 3 || (major == 3 && minor >= 13))
        .unwrap_or(false);
    support.insert("frame_locals".to_string(), frame_locals);
    support.insert("rcu".to_string(), true);
    support.insert("journal".to_string(), true);
    support.insert("mio".to_string(), Reactor::new().is_ok());
    Ok(support)
}

#[pymodule]
fn polyglot_hotpatch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExecutableRegion>()?;
    m.add_class::<NativeTrampolinePatch>()?;
    m.add_class::<NativeRcuDomain>()?;
    m.add_class::<NativeJournal>()?;
    m.add_class::<NativeMioPoller>()?;
    m.add_class::<NativeMioMigration>()?;
    m.add_class::<NativeFramePatch>()?;

    m.add_function(wrap_pyfunction!(hotpatch_live_dylib_trampoline, m)?)?;
    m.add_function(wrap_pyfunction!(hotpatch_cpython_frame_eval, m)?)?;
    m.add_function(wrap_pyfunction!(hotpatch_mio_reactor_state, m)?)?;
    m.add_function(wrap_pyfunction!(hotpatch_rcu_ast_epoch, m)?)?;
    m.add_function(wrap_pyfunction!(hotpatch_rollback_journal_commit, m)?)?;
    m.add_function(wrap_pyfunction!(native_patch_slot, m)?)?;
    m.add_function(wrap_pyfunction!(native_read_pointer, m)?)?;
    m.add_function(wrap_pyfunction!(native_find_got_slots, m)?)?;
    m.add_function(wrap_pyfunction!(native_encode_jump, m)?)?;
    m.add_function(wrap_pyfunction!(native_rcu_domain, m)?)?;
    m.add_function(wrap_pyfunction!(native_journal, m)?)?;
    m.add_function(wrap_pyfunction!(native_frame_locals, m)?)?;
    m.add_function(wrap_pyfunction!(native_platform_support, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("E008", error::E008)?;
    m.add("E007", error::E007)?;
    Ok(())
}
