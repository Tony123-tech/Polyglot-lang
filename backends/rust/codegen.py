"""
AST -> Rust source code generator for the native backend.

Supports (v2 native, N2b):
- free functions with #[pyfunction]
- basic types: integers, floats, bool, String, &str
- Vec<T> with push / pop / len / index
- struct + #[pyclass]
- impl methods + #[pymethods] / #[new]
- expressions & control flow: if / while / for / loop / match
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass

from ast_nodes import (
    ArrayExpr,
    Assign,
    Await,
    Binary,
    Block,
    Break,
    Call,
    Cast,
    Closure,
    CompoundAssign,
    ConstDecl,
    Continue,
    EnumDecl,
    Expr,
    ExprStmt,
    Field,
    FnDecl,
    FnParam,
    For,
    Ident,
    IfExpr,
    IfLet,
    ImplBlock,
    Index,
    Item,
    ItemStmt,
    LetStmt,
    LitBool,
    LitChar,
    LitFloat,
    LitInt,
    LitStr,
    Loop,
    MacroCall,
    MatchArm,
    MatchExpr,
    MethodCall,
    ModDecl,
    Node,
    OptionNone,
    OptionSome,
    Paren,
    PatBinding,
    PatBox,
    PatConst,
    PatEnum,
    Path,
    PatIdent,
    PatLiteral,
    PatMacro,
    PatOr,
    PatPath,
    PatRange,
    PatRef,
    PatRest,
    PatSlice,
    PatStruct,
    Pattern,
    PatTuple,
    PatWildcard,
    Program,
    Range,
    RepeatArray,
    ResultErr,
    ResultOk,
    Return,
    Stmt,
    StructDecl,
    StructLiteral,
    TraitDecl,
    Try,
    TupleExpr,
    Type,
    TypeAlias,
    TypeArray,
    TypeName,
    TypeRef,
    TypeSlice,
    TypeTuple,
    TypeUnit,
    Unary,
    UseDecl,
    While,
    is_i64_type,
    is_isize_type,
    is_usize_type,
)
from backends.errors import (
    HOST_CALL_ARITIES,
    InvalidIntegerParseLiteralError,
    TranslateError,
    UnsupportedNativePatternError,
    check_host_call_arity,
)
from match_diagnostics import warn_match_issues
from monomorph import monomorphize
from runtime.feature_gates import validate_program_features
from struct_order import order_struct_declarations


class InvalidCastError(RuntimeError):
    def __init__(self, message, line, col):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_unit_type(ty: Type) -> bool:
    return isinstance(ty, TypeUnit) or (
        isinstance(ty, TypeTuple) and len(ty.elems) == 0
    )


def _is_valid_i64_literal(value: str) -> bool:
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        return False
    negative = value.startswith("-")
    digits = value.lstrip("+-").lstrip("0") or "0"
    limit = "9223372036854775808" if negative else "9223372036854775807"
    return len(digits) < len(limit) or (len(digits) == len(limit) and digits <= limit)


_RUNTIME_PREFIX = r"""
use std::future::Future;
use pyo3::IntoPyObjectExt;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use std::collections::{BinaryHeap, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, OnceLock};
use std::time::{Duration, Instant};

// Compiler-reserved host seam for std:fs. These helpers return the
// language-level Result<T, String> so user code can propagate Err(String)
// directly; the generated function boundary converts that to PyResult<T>.
#[allow(dead_code)]
fn __fs_py_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            c if c == quote => { out.push('\\'); out.push(c); }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[allow(dead_code)]
fn __fs_error(operation: &str, path: &str, error: &std::io::Error) -> String {
    format!("{} {}: {}", operation, __fs_py_repr(path), error)
}

#[allow(dead_code)]
fn __fs_read_text(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| __fs_error("read", &path, &e))
}

#[allow(dead_code)]
fn __fs_write_text(path: String, contents: String) -> Result<(), String> {
    fs::write(&path, contents).map_err(|e| __fs_error("write", &path, &e))
}

#[allow(dead_code)]
fn __fs_append_text(path: String, contents: String) -> Result<(), String> {
    use std::io::Write;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(contents.as_bytes()))
        .map_err(|e| __fs_error("append", &path, &e))
}

#[allow(dead_code)]
fn __fs_exists(path: String) -> bool {
    std::path::Path::new(&path).exists()
}

#[allow(dead_code)]
fn __fs_remove(path: String) -> Result<(), String> {
    fs::remove_file(&path).map_err(|e| __fs_error("remove", &path, &e))
}

// Number of times a native future has been polled. Tests reset this to prove
// that a pending future makes progress through the reactor instead of being
// busy-polled from Python.
static POLYGLOT_ASYNC_POLLS: AtomicU64 = AtomicU64::new(0);

fn polyglot_async_poll_reset() {
    POLYGLOT_ASYNC_POLLS.store(0, AtomicOrdering::SeqCst);
}

fn polyglot_async_poll_count() -> i64 {
    POLYGLOT_ASYNC_POLLS.load(AtomicOrdering::SeqCst) as i64
}

type PolyglotBoxFuture =
    Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>;

struct PolyglotTask {
    id: u64,
    future: Mutex<PolyglotBoxFuture>,
    result: Mutex<Option<PyResult<PyObject>>>,
    waiter: Mutex<Option<Py<PyAny>>>,
    poll_lock: Mutex<()>,
    done: AtomicBool,
    scheduled: AtomicBool,
    active: AtomicBool,
    woken: AtomicBool,
    cancelled: AtomicBool,
}

impl PolyglotTask {
    fn take_result(&self) -> Option<PyResult<PyObject>> {
        self.result.lock().unwrap().take()
    }

    fn waker(self: &Arc<Self>) -> Waker {
        unsafe {
            let raw = RawWaker::new(
                Arc::into_raw(self.clone()) as *const (),
                &POLYGLOT_TASK_WAKER_VTABLE,
            );
            Waker::from_raw(raw)
        }
    }

    fn notify_python(&self) {
        let waiter = self.waiter.lock().unwrap().take();
        if let Some(waiter) = waiter {
            Python::with_gil(|py| {
                let future = waiter.bind(py);
                let loop_obj = future.call_method0("get_loop").ok();
                let set_result = future.getattr("set_result").ok();
                if let (Some(loop_obj), Some(set_result)) = (loop_obj, set_result) {
                    // Python futures must be resolved on their own event-loop
                    // thread, so schedule the callback thread-safely.
                    let _ = loop_obj.call_method1(
                        "call_soon_threadsafe",
                        (set_result, py.None()),
                    );
                }
            });
        }
    }
}

fn polyglot_wake_task(task: &Arc<PolyglotTask>) {
    // Before the task is handed to the reactor the wake is recorded locally.
    // This closes the race between the first synchronous poll and activation.
    if task.active.load(AtomicOrdering::SeqCst) {
        polyglot_reactor().enqueue(task);
    } else {
        task.woken.store(true, AtomicOrdering::SeqCst);
    }
}

unsafe fn polyglot_task_waker_clone(data: *const ()) -> RawWaker {
    Arc::increment_strong_count(data as *const PolyglotTask);
    RawWaker::new(data, &POLYGLOT_TASK_WAKER_VTABLE)
}

unsafe fn polyglot_task_waker_wake(data: *const ()) {
    let task = Arc::from_raw(data as *const PolyglotTask);
    polyglot_wake_task(&task);
}

unsafe fn polyglot_task_waker_wake_by_ref(data: *const ()) {
    Arc::increment_strong_count(data as *const PolyglotTask);
    let task = Arc::from_raw(data as *const PolyglotTask);
    polyglot_wake_task(&task);
}

unsafe fn polyglot_task_waker_drop(data: *const ()) {
    drop(Arc::from_raw(data as *const PolyglotTask));
}

static POLYGLOT_TASK_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    polyglot_task_waker_clone,
    polyglot_task_waker_wake,
    polyglot_task_waker_wake_by_ref,
    polyglot_task_waker_drop,
);

struct PolyglotTimer {
    deadline: Instant,
    waker: Waker,
}

impl PartialEq for PolyglotTimer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for PolyglotTimer {}

impl PartialOrd for PolyglotTimer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PolyglotTimer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap becomes a min-heap by deadline.
        other.deadline.cmp(&self.deadline)
    }
}

#[cfg(unix)]
struct PolyglotIoSlot {
    token: mio::Token,
    fd: std::os::fd::RawFd,
    ready: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

struct PolyglotReactorState {
    next_id: u64,
    tasks: HashMap<u64, Arc<PolyglotTask>>,
    queue: VecDeque<u64>,
    timers: BinaryHeap<PolyglotTimer>,
    #[cfg(unix)]
    next_io_token: usize,
    #[cfg(unix)]
    io: HashMap<mio::Token, Arc<PolyglotIoSlot>>,
}

struct PolyglotReactor {
    poll: Mutex<mio::Poll>,
    registry: mio::Registry,
    wake_handle: mio::Waker,
    state: Mutex<PolyglotReactorState>,
}

static POLYGLOT_REACTOR: OnceLock<Arc<PolyglotReactor>> = OnceLock::new();

fn polyglot_reactor() -> Arc<PolyglotReactor> {
    POLYGLOT_REACTOR
        .get_or_init(PolyglotReactor::start)
        .clone()
}

impl PolyglotReactor {
    fn start() -> Arc<Self> {
        let poll = mio::Poll::new().expect("failed to create async reactor");
        let registry = poll
            .registry()
            .try_clone()
            .expect("failed to clone async reactor registry");
        let wake_handle = mio::Waker::new(&registry, mio::Token(0))
            .expect("failed to create async reactor waker");
        let reactor = Arc::new(Self {
            poll: Mutex::new(poll),
            registry,
            wake_handle,
            state: Mutex::new(PolyglotReactorState {
                next_id: 0,
                tasks: HashMap::new(),
                queue: VecDeque::new(),
                timers: BinaryHeap::new(),
                #[cfg(unix)]
                next_io_token: 1,
                #[cfg(unix)]
                io: HashMap::new(),
            }),
        });
        let runner = reactor.clone();
        std::thread::Builder::new()
            .name("polyglot-async".to_string())
            .spawn(move || runner.run())
            .expect("failed to spawn async reactor");
        reactor
    }

    fn add_task(&self, future: PolyglotBoxFuture) -> Arc<PolyglotTask> {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        let task = Arc::new(PolyglotTask {
            id,
            future: Mutex::new(future),
            result: Mutex::new(None),
            waiter: Mutex::new(None),
            poll_lock: Mutex::new(()),
            done: AtomicBool::new(false),
            scheduled: AtomicBool::new(false),
            active: AtomicBool::new(false),
            woken: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        });
        state.tasks.insert(id, task.clone());
        task
    }

    fn remove_task(&self, id: u64) {
        self.state.lock().unwrap().tasks.remove(&id);
    }

    fn enqueue(&self, task: &Arc<PolyglotTask>) {
        if !task.scheduled.swap(true, AtomicOrdering::SeqCst) {
            self.state.lock().unwrap().queue.push_back(task.id);
            let _ = self.wake_handle.wake();
        }
    }

    fn add_timer(&self, deadline: Instant, waker: Waker) {
        self.state
            .lock()
            .unwrap()
            .timers
            .push(PolyglotTimer { deadline, waker });
        let _ = self.wake_handle.wake();
    }

    fn fire_timers(&self) {
        loop {
            let now = Instant::now();
            let waker = {
                let mut state = self.state.lock().unwrap();
                match state.timers.peek() {
                    Some(timer) if timer.deadline <= now => {
                        state.timers.pop().map(|timer| timer.waker)
                    }
                    _ => None,
                }
            };
            match waker {
                Some(waker) => waker.wake(),
                None => break,
            }
        }
    }

    fn poll_task(&self, task: &Arc<PolyglotTask>) {
        if task.cancelled.load(AtomicOrdering::SeqCst) {
            self.remove_task(task.id);
            return;
        }
        let waker = task.waker();
        let mut cx = Context::from_waker(&waker);
        let polled = {
            let _guard = task.poll_lock.lock().unwrap();
            let mut future = task.future.lock().unwrap();
            POLYGLOT_ASYNC_POLLS.fetch_add(1, AtomicOrdering::SeqCst);
            future.as_mut().poll(&mut cx)
        };
        if let Poll::Ready(result) = polled {
            task.done.store(true, AtomicOrdering::SeqCst);
            *task.result.lock().unwrap() = Some(result);
            self.remove_task(task.id);
            task.notify_python();
        }
    }

    #[cfg(unix)]
    fn register_io(
        &self,
        fd: std::os::fd::RawFd,
        waker: Waker,
    ) -> Option<Arc<PolyglotIoSlot>> {
        let (token, slot) = {
            let mut state = self.state.lock().unwrap();
            let token = mio::Token(state.next_io_token);
            state.next_io_token += 1;
            let slot = Arc::new(PolyglotIoSlot {
                token,
                fd,
                ready: AtomicBool::new(false),
                waker: Mutex::new(Some(waker)),
            });
            state.io.insert(token, slot.clone());
            (token, slot)
        };
        let mut source = mio::unix::SourceFd(&fd);
        match self
            .registry
            .register(&mut source, token, mio::Interest::READABLE)
        {
            Ok(()) => Some(slot),
            Err(_) => {
                self.state.lock().unwrap().io.remove(&token);
                None
            }
        }
    }

    #[cfg(unix)]
    fn deregister_io(&self, token: mio::Token) {
        let slot = self.state.lock().unwrap().io.remove(&token);
        if let Some(slot) = slot {
            let mut source = mio::unix::SourceFd(&slot.fd);
            let _ = self.registry.deregister(&mut source);
        }
    }

    fn run(self: Arc<Self>) {
        let mut events = mio::Events::with_capacity(64);
        loop {
            let ready: Vec<u64> = {
                let mut state = self.state.lock().unwrap();
                state.queue.drain(..).collect()
            };
            for id in ready {
                let task = {
                    let state = self.state.lock().unwrap();
                    state.tasks.get(&id).cloned()
                };
                if let Some(task) = task {
                    task.scheduled.store(false, AtomicOrdering::SeqCst);
                    self.poll_task(&task);
                }
            }

            self.fire_timers();

            let timeout = {
                let state = self.state.lock().unwrap();
                if !state.queue.is_empty() {
                    Some(Duration::ZERO)
                } else {
                    state.timers.peek().map(|timer| {
                        timer
                            .deadline
                            .saturating_duration_since(Instant::now())
                    })
                }
            };

            if timeout == Some(Duration::ZERO) {
                continue;
            }

            let mut poll = self.poll.lock().unwrap();
            let _ = poll.poll(&mut events, timeout);

            #[cfg(unix)]
            for event in events.iter() {
                if event.token() == mio::Token(0) {
                    continue;
                }
                let slot = {
                    let state = self.state.lock().unwrap();
                    state.io.get(&event.token()).cloned()
                };
                if let Some(slot) = slot {
                    slot.ready.store(true, AtomicOrdering::SeqCst);
                    let waker = slot.waker.lock().unwrap().take();
                    if let Some(waker) = waker {
                        waker.wake();
                    }
                }
            }
        }
    }
}

struct PolyglotSleep {
    delay: Duration,
    deadline: Option<Instant>,
}

impl Future for PolyglotSleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let now = Instant::now();
        if let Some(deadline) = this.deadline {
            if now >= deadline {
                return Poll::Ready(());
            }
        } else {
            let deadline = now + this.delay;
            this.deadline = Some(deadline);
            polyglot_reactor().add_timer(deadline, cx.waker().clone());
        }
        Poll::Pending
    }
}

fn polyglot_sleep_ms(ms: i64) -> impl Future<Output = ()> + Send {
    let millis = ms.max(0) as u64;
    PolyglotSleep {
        delay: Duration::from_millis(millis),
        deadline: None,
    }
}

#[cfg(unix)]
struct PolyglotReadable {
    fd: std::os::fd::RawFd,
    slot: Option<Arc<PolyglotIoSlot>>,
}

#[cfg(unix)]
impl Future for PolyglotReadable {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if let Some(slot) = &this.slot {
            if slot.ready.swap(false, AtomicOrdering::SeqCst) {
                polyglot_reactor().deregister_io(slot.token);
                this.slot = None;
                return Poll::Ready(());
            }
            *slot.waker.lock().unwrap() = Some(cx.waker().clone());
            return Poll::Pending;
        }
        match polyglot_reactor().register_io(this.fd, cx.waker().clone()) {
            Some(slot) => {
                this.slot = Some(slot);
                Poll::Pending
            }
            None => Poll::Ready(()),
        }
    }
}

#[cfg(unix)]
fn polyglot_readable(fd: i64) -> impl Future<Output = ()> + Send {
    PolyglotReadable {
        fd: fd as std::os::fd::RawFd,
        slot: None,
    }
}

struct PolyglotSyncWake {
    flag: Mutex<bool>,
    condvar: Condvar,
}

unsafe fn polyglot_sync_waker_clone(data: *const ()) -> RawWaker {
    Arc::increment_strong_count(data as *const PolyglotSyncWake);
    RawWaker::new(data, &POLYGLOT_SYNC_WAKER_VTABLE)
}

unsafe fn polyglot_sync_waker_wake(data: *const ()) {
    let wake = Arc::from_raw(data as *const PolyglotSyncWake);
    *wake.flag.lock().unwrap() = true;
    wake.condvar.notify_all();
}

unsafe fn polyglot_sync_waker_wake_by_ref(data: *const ()) {
    Arc::increment_strong_count(data as *const PolyglotSyncWake);
    let wake = Arc::from_raw(data as *const PolyglotSyncWake);
    *wake.flag.lock().unwrap() = true;
    wake.condvar.notify_all();
}

unsafe fn polyglot_sync_waker_drop(data: *const ()) {
    drop(Arc::from_raw(data as *const PolyglotSyncWake));
}

static POLYGLOT_SYNC_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    polyglot_sync_waker_clone,
    polyglot_sync_waker_wake,
    polyglot_sync_waker_wake_by_ref,
    polyglot_sync_waker_drop,
);

fn polyglot_sync_waker(wake: Arc<PolyglotSyncWake>) -> Waker {
    unsafe {
        let raw = RawWaker::new(
            Arc::into_raw(wake) as *const (),
            &POLYGLOT_SYNC_WAKER_VTABLE,
        );
        Waker::from_raw(raw)
    }
}

// M3: HashSet has no `get_mut` in std. At the Python boundary the value is
// an owned copy anyway, so return the present element unchanged.
fn polyglot_hashset_get_mut<T: Clone + Eq + std::hash::Hash>(
    set: &HashSet<T>,
    value: &T,
) -> Option<T> {
    if set.contains(value) {
        Some(value.clone())
    } else {
        None
    }
}

#[pyclass]
pub struct PyFuture {
    future: Mutex<Option<PolyglotBoxFuture>>,
}

#[pymethods]
impl PyFuture {
    fn __await__(mut slf: PyRefMut<'_, Self>) -> PyResult<Py<PyFutureIterator>> {
        let py = slf.py();
        let future = slf.future.lock().unwrap().take();
        Py::new(
            py,
            PyFutureIterator {
                inner: Mutex::new(PolyglotIteratorInner {
                    future,
                    task: None,
                }),
            },
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
}

struct PolyglotIteratorInner {
    future: Option<PolyglotBoxFuture>,
    task: Option<Arc<PolyglotTask>>,
}

/// Cancellation is drop-based: dropping a `PyFuture` before it is awaited, or
/// dropping the iterator before it is exhausted, drops (or unregisters) the
/// boxed future so it is never polled again.
#[pyclass]
pub struct PyFutureIterator {
    inner: Mutex<PolyglotIteratorInner>,
}

impl Drop for PyFutureIterator {
    fn drop(&mut self) {
        if let Ok(inner) = self.inner.get_mut() {
            if let Some(task) = inner.task.take() {
                task.cancelled.store(true, AtomicOrdering::SeqCst);
                polyglot_reactor().remove_task(task.id);
            }
        }
    }
}

fn polyglot_future_result(result: PyResult<PyObject>) -> PyResult<PyObject> {
    match result {
        Ok(value) => Err(pyo3::exceptions::PyStopIteration::new_err(value)),
        Err(err) => Err(err),
    }
}

fn polyglot_pending_waiter(
    py: Python<'_>,
    task: &Arc<PolyglotTask>,
) -> PyResult<PyObject> {
    if let Some(result) = task.take_result() {
        return polyglot_future_result(result);
    }

    let mut waiter_guard = task.waiter.lock().unwrap();
    if waiter_guard.is_some() {
        // Python is asking for another iteration while the waiter future is
        // still pending; yielding to the loop is the only sane response.
        return Ok(py.None());
    }

    let waiter = py
        .import("asyncio")
        .and_then(|module| module.call_method0("get_running_loop"))
        .and_then(|loop_obj| loop_obj.call_method0("create_future"));
    let waiter = match waiter {
        Ok(waiter) => {
            // Mark the future as yielded via `yield from`, matching what
            // `Future.__await__` does. Without this flag asyncio treats the
            // value as a plain `yield` and refuses to wait on the task.
            let _ = waiter.setattr("_asyncio_future_blocking", true);
            waiter.unbind()
        }
        // Without an asyncio loop, fall back to cooperative yielding. The
        // caller can keep calling __next__ until the future is ready.
        Err(_) => return Ok(py.None()),
    };
    *waiter_guard = Some(waiter.clone_ref(py));
    drop(waiter_guard);

    // The reactor may have completed the task while the waiter was being
    // installed. Check once more so that wakeups are never lost.
    if let Some(result) = task.take_result() {
        return polyglot_future_result(result);
    }
    Ok(waiter)
}

#[pymethods]
impl PyFutureIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyObject> {
        let py = slf.py();
        let mut inner = slf.inner.lock().unwrap();

        if let Some(task) = inner.task.as_ref() {
            if let Some(result) = task.take_result() {
                return polyglot_future_result(result);
            }
            return polyglot_pending_waiter(py, task);
        }

        let Some(future) = inner.future.take() else {
            return Err(pyo3::exceptions::PyStopIteration::new_err(()));
        };

        let task = polyglot_reactor().add_task(future);
        let waker = task.waker();
        let mut cx = Context::from_waker(&waker);
        let polled = {
            let _guard = task.poll_lock.lock().unwrap();
            let mut future = task.future.lock().unwrap();
            POLYGLOT_ASYNC_POLLS.fetch_add(1, AtomicOrdering::SeqCst);
            future.as_mut().poll(&mut cx)
        };

        match polled {
            Poll::Ready(result) => {
                task.done.store(true, AtomicOrdering::SeqCst);
                *task.result.lock().unwrap() = Some(result);
                polyglot_reactor().remove_task(task.id);
                let result = task
                    .take_result()
                    .expect("future result disappeared after completion");
                polyglot_future_result(result)
            }
            Poll::Pending => {
                inner.task = Some(task.clone());
                task.active.store(true, AtomicOrdering::SeqCst);
                if task.woken.swap(false, AtomicOrdering::SeqCst) {
                    polyglot_reactor().enqueue(&task);
                }
                polyglot_pending_waiter(py, &task)
            }
        }
    }
}

fn py_future<F>(future: F) -> PyFuture
where
    F: Future<Output = PyResult<PyObject>> + Send + 'static,
{
    PyFuture {
        future: Mutex::new(Some(Box::pin(future))),
    }
}

pyo3::create_exception!(polyglot_native, PyTimeoutError, pyo3::exceptions::PyException);

/// A future that never becomes ready. Internal helper used to exercise the
/// timeout path without an event loop or external I/O.
fn polyglot_pending_future() -> PyFuture {
    py_future(std::future::pending::<PyResult<PyObject>>())
}

struct PolyglotBorrowGuard {
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl PolyglotBorrowGuard {
    fn acquire_mut(
        state: &std::sync::Arc<std::sync::atomic::AtomicU8>,
        type_name: &str,
    ) -> pyo3::PyResult<Self> {
        if state
            .compare_exchange(
                0,
                2,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{} is already borrowed by an active async method",
                type_name
            )));
        }
        Ok(Self {
            state: state.clone(),
        })
    }
}

impl Drop for PolyglotBorrowGuard {
    fn drop(&mut self) {
        self.state.store(0, std::sync::atomic::Ordering::Release);
    }
}


/// Poll `future` until it is ready or `seconds` elapse.
///
/// Readiness is delivered by the reactor waker to a condition variable, so the
/// helper sleeps until the future makes progress (or the deadline expires)
/// instead of repeatedly polling. The GIL is released while waiting. Dropping
/// the future on timeout or error provides drop-based cancellation.
fn await_with_timeout(
    py: Python<'_>,
    future: PyFuture,
    seconds: f64,
) -> PyResult<PyObject> {
    let mut fut = match future.future.lock().unwrap().take() {
        Some(fut) => fut,
        None => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "future has already been consumed",
            ))
        }
    };
    let seconds = if seconds.is_finite() && seconds > 0.0 {
        seconds
    } else {
        0.0
    };
    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let wake = Arc::new(PolyglotSyncWake {
        flag: Mutex::new(false),
        condvar: Condvar::new(),
    });
    let waker = polyglot_sync_waker(wake.clone());
    let mut cx = Context::from_waker(&waker);
    loop {
        *wake.flag.lock().unwrap() = false;
        POLYGLOT_ASYNC_POLLS.fetch_add(1, AtomicOrdering::SeqCst);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(value)) => return Ok(value),
            Poll::Ready(Err(e)) => return Err(e),
            Poll::Pending => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(PyTimeoutError::new_err(format!(
                        "future did not complete within {seconds} seconds"
                    )));
                }
                let remaining = deadline - now;
                let timed_out = py.allow_threads(|| {
                    let flag = wake.flag.lock().unwrap();
                    if *flag {
                        return false;
                    }
                    let (_flag, result) = wake
                        .condvar
                        .wait_timeout(flag, remaining)
                        .expect("async waiter mutex poisoned");
                    result.timed_out()
                });
                if timed_out {
                    return Err(PyTimeoutError::new_err(format!(
                        "future did not complete within {seconds} seconds"
                    )));
                }
            }
        }
    }
}
"""


class RustCodegen:
    """Generate Rust source (with PyO3) from a Polyglot Program."""

    def __init__(self, module_name: str = "polyglot_native") -> None:
        self.module_name = module_name
        self.lines: list[str] = []
        self.indent_level = 0
        self.exported_fns: list[str] = []
        self.exported_classes: list[str] = []
        self.struct_names: set[str] = set()
        self.pyclass_names: set[str] = set()
        self.generic_struct_names: set[str] = set()
        self.generic_type_names: set[str] = set()
        self.string_bindings: set[str] = set()
        self.vec_bindings: set[str] = set()
        self.binding_types: dict[str, str] = {}
        self._function_signatures: dict[str, list[str | None]] = {}
        self._function_return_types: dict[str, str | None] = {}
        self._method_return_types: dict[tuple[str, str], str | None] = {}
        self.struct_field_types: dict[str, dict[str, str]] = {}
        self._current_impl_type: str | None = None
        self.line_map: dict[int, tuple[int, int]] = {}
        self.line_map_nodes: dict[int, Node] = {}
        self.source_line_text_map: dict[int, str] = {}
        self.source_lines: dict[int, str] = {}
        self._origin: tuple[int, int] | None = None
        self._origin_node: Node | None = None
        self._in_pymethods = False
        self._result_types: tuple[str, str] | None = None
        self._current_return_type: str | None = None
        self._self_field_aliases: dict[str, str] | None = None
        self._payload_enum_names: set[str] = set()
        self.borrow_state_types: set[str] = set()
        # Content-addressed top-level fragments. ``fragments`` maps a stable
        # key such as ``fn:add`` to the Rust text that implements it.
        self.fragments: dict[str, str] = {}
        self.fragment_hashes: dict[str, str] = {}
        self.fragment_spans: dict[str, tuple[int, int]] = {}
        self.header_source = ""

    # ---- helpers ----------------------------------------------------------

    def emit(self, line: str = "") -> None:
        self.lines.append("    " * self.indent_level + line)
        if self._origin is not None:
            self.line_map[len(self.lines)] = self._origin
            if self._origin_node is not None:
                self.line_map_nodes[len(self.lines)] = self._origin_node
            source_text = self.source_lines.get(self._origin[0])
            if source_text is not None:
                self.source_line_text_map[len(self.lines)] = source_text

    def set_origin(self, node: Node) -> None:
        self._origin = (node.line, node.col)
        self._origin_node = node

    def indent(self) -> None:
        self.indent_level += 1

    def dedent(self) -> None:
        self.indent_level = max(0, self.indent_level - 1)

    # ---- fragments --------------------------------------------------------

    def _begin_fragment(self) -> int:
        """Return the line index where the next top-level fragment starts."""
        return len(self.lines)

    def _end_fragment(self, key: str, start: int) -> None:
        """Record ``self.lines[start:]`` as a named, content-addressed fragment."""
        unique = key
        suffix = 2
        while unique in self.fragment_spans:
            unique = f"{key}#{suffix}"
            suffix += 1
        end = len(self.lines)
        text = "\n".join(self.lines[start:end])
        self.fragment_spans[unique] = (start, end)
        self.fragments[unique] = text
        self.fragment_hashes[unique] = hashlib.sha256(text.encode()).hexdigest()[:16]

    def _item_fragment_key(self, item: Item) -> str:
        if isinstance(item, FnDecl):
            return f"fn:{item.name}"
        if isinstance(item, StructDecl):
            return f"struct:{item.name}"
        if isinstance(item, ImplBlock):
            return f"impl:{self.type_to_rust(item.ty)}"
        if isinstance(item, EnumDecl):
            return f"enum:{item.name}"
        if isinstance(item, ConstDecl):
            return f"const:{item.name}"
        if isinstance(item, UseDecl):
            return f"use:{item.path}"
        if isinstance(item, ModDecl):
            return f"mod:{item.name}"
        if isinstance(item, TypeAlias):
            return f"type:{item.name}"
        if isinstance(item, TraitDecl):
            return f"trait:{item.name}"
        return type(item).__name__

    def generate(self, program: Program) -> str:
        validate_program_features(program)
        program = monomorphize(program)
        self.lines = []
        self.fragments = {}
        self.fragment_hashes = {}
        self.fragment_spans = {}
        self.header_source = ""
        self.exported_fns = []
        self.exported_classes = []
        self.struct_names = set()
        self.pyclass_names = set()
        self.generic_struct_names = set()
        self.generic_type_names = set()
        self.string_bindings = set()
        self.vec_bindings = set()
        self.binding_types = {}
        self._function_signatures = {}
        self._function_return_types = {}
        self._method_return_types = {}
        self.struct_field_types = {}
        self._current_impl_type = None
        self.line_map = {}
        self.line_map_nodes = {}
        self.source_line_text_map = {}
        self.source_lines = program.source_lines
        self._origin = None
        self._origin_node = None
        self.indent_level = 0
        self._in_pymethods = False
        self._result_types = None
        self._current_return_type = None
        self._self_field_aliases = None
        self._payload_enum_names = set()
        self.borrow_state_types = set()

        for item in program.items:
            if isinstance(item, StructDecl):
                self.struct_names.add(item.name)
                self.pyclass_names.add(item.name)
                self.struct_field_types[item.name] = {
                    name: ty.name
                    for name, ty, _is_pub in item.fields
                    if isinstance(ty, TypeName)
                }
                if item.generics:
                    self.generic_struct_names.add(item.name)
                    self.generic_type_names.update(
                        param.name for param in item.generics
                    )
            elif isinstance(item, FnDecl):
                self._function_signatures[item.name] = [
                    self._binding_type_name(param.ty) for param in item.params
                ]
                self._function_return_types[item.name] = self._binding_type_name(
                    item.ret
                )
            elif isinstance(item, ImplBlock):
                target = self._binding_type_name(item.ty)
                if target is not None:
                    for method in item.items:
                        if isinstance(method, FnDecl):
                            self._method_return_types[(target, method.name)] = (
                                self._binding_type_name(method.ret)
                            )
                            if (
                                method.is_async
                                and self._async_receiver_kind(method) == "mut"
                            ):
                                self.borrow_state_types.add(target)
            elif isinstance(item, EnumDecl) and all(
                variant.fields is None for variant in item.variants
            ):
                self.pyclass_names.add(item.name)
            elif isinstance(item, EnumDecl):
                # Payload enums are generated without a Clone derive, so an
                # async &self method cannot clone such a field.
                self._payload_enum_names.add(item.name)

        # Suppress warnings from aggressive `mut` / stub bodies.
        self.emit("#![allow(unused_mut, unused_variables, dead_code, unused_imports)]")
        self.emit()
        self.header_source = "\n".join(self.lines)

        runtime_start = self._begin_fragment()
        self.emit("use pyo3::prelude::*;")
        self.emit("use std::collections::{BTreeMap, HashMap, HashSet};")
        self.emit("use std::fmt::Write as FmtWrite;")
        self.emit()
        for line in _RUNTIME_PREFIX.strip().splitlines():
            self.emit(line)
        self.emit()
        self._end_fragment("runtime", runtime_start)

        structs = [item for item in program.items if isinstance(item, StructDecl)]
        for struct in order_struct_declarations(structs):
            start = self._begin_fragment()
            self.generate_struct(struct)
            self._end_fragment(f"struct:{struct.name}", start)
        emitted_impls: set[str] = set()
        for item in program.items:
            if not isinstance(item, StructDecl):
                if isinstance(item, ImplBlock):
                    target = self.type_to_rust(item.ty)
                    if target in emitted_impls:
                        continue
                    impls = [
                        candidate
                        for candidate in program.items
                        if isinstance(candidate, ImplBlock)
                        and self.type_to_rust(candidate.ty) == target
                    ]
                    start = self._begin_fragment()
                    self.generate_impls(impls)
                    self._end_fragment(f"impl:{target}", start)
                    emitted_impls.add(target)
                else:
                    start = self._begin_fragment()
                    self.generate_item(item)
                    self._end_fragment(self._item_fragment_key(item), start)

        start = self._begin_fragment()
        self.emit()
        self.emit("#[pymodule]")
        self.emit(f"fn {self.module_name}(m: &Bound<'_, PyModule>) -> PyResult<()> {{")
        self.indent()
        for name in self.exported_fns:
            self.emit(f"m.add_function(wrap_pyfunction!({name}, m)?)?;")
        for name in self.exported_classes:
            self.emit(f"m.add_class::<{name}>()?;")
        self.emit("Ok(())")
        self.dedent()
        self.emit("}")
        self._end_fragment(f"module:{self.module_name}", start)

        return "\n".join(self.lines)

    # ---- items ------------------------------------------------------------

    def generate_item(self, item: Item) -> None:
        self.set_origin(item)
        if isinstance(item, FnDecl):
            self.generate_fn(item, as_method=False)
        elif isinstance(item, StructDecl):
            self.generate_struct(item)
        elif isinstance(item, EnumDecl):
            self.generate_enum(item)
        elif isinstance(item, ImplBlock):
            self.generate_impl(item)
        elif isinstance(item, ConstDecl):
            self.generate_const(item)
        elif isinstance(item, UseDecl):
            self.emit(f"use {item.path};")
        elif isinstance(item, ModDecl):
            if item.items is not None:
                self.emit(f"mod {item.name} {{")
                self.indent()
                for sub in item.items:
                    self.generate_item(sub)
                self.dedent()
                self.emit("}")
        elif isinstance(item, TypeAlias):
            self.emit(f"type {item.name} = {self.type_to_rust(item.ty)};")
        elif isinstance(item, TraitDecl):
            pass

    def _async_receiver_kind(self, fn: FnDecl) -> str | None:
        """Classify the receiver of `fn` as ``shared``, ``mut`` or ``value``."""
        if not fn.params:
            return None
        pattern = fn.params[0].pattern
        if (
            isinstance(pattern, PatRef)
            and isinstance(pattern.inner, PatIdent)
            and pattern.inner.name == "self"
        ):
            return "mut" if pattern.mutable else "shared"
        if isinstance(pattern, PatIdent) and pattern.name == "self":
            return "value"
        return None

    def _collect_async_self_usage(
        self,
        node: Node | list | tuple | None,
        fields: set[str],
        bare: list[Node],
    ) -> None:
        """Collect ``self.<field>`` reads and bare ``self`` uses in ``node``."""
        if isinstance(node, Field):
            if isinstance(node.receiver, Ident) and node.receiver.name == "self":
                fields.add(node.name)
            else:
                self._collect_async_self_usage(node.receiver, fields, bare)
            return
        if isinstance(node, Ident):
            if node.name == "self":
                bare.append(node)
            return
        if is_dataclass(node):
            for info in dataclass_fields(node):
                self._collect_async_self_usage(getattr(node, info.name), fields, bare)
        elif isinstance(node, (list, tuple)):
            for item in node:
                self._collect_async_self_usage(item, fields, bare)

    def _prepare_async_mut_fields(self, fn: FnDecl) -> dict[str, str]:
        struct_name = self._current_impl_type
        if not struct_name:
            return {}
        fields = self.struct_field_types.get(struct_name, {})
        return {name: f"__self_mut_{name}" for name in fields}

    def _prepare_async_self_fields(self, fn: FnDecl) -> dict[str, str]:
        """Return field aliases to clone before an async ``&self`` body runs."""
        fields: set[str] = set()
        bare: list[Node] = []
        if fn.body is not None:
            self._collect_async_self_usage(fn.body, fields, bare)
        if bare:
            raise TranslateError(
                f"async method '{fn.name}' uses `self` directly; "
                "async &self methods may only access fields such as "
                "`self.<field>`"
            )
        field_types = self.struct_field_types.get(self._current_impl_type or "", {})
        aliases: dict[str, str] = {}
        for name in sorted(fields):
            ty = field_types.get(name)
            if ty is None:
                raise TranslateError(
                    f"async method '{fn.name}' reads field '{name}', but the "
                    "native backend cannot prove the field type is Clone"
                )
            if ty in self._payload_enum_names:
                raise TranslateError(
                    f"async method '{fn.name}' reads field '{name}' whose type "
                    f"'{ty}' is not Clone; async &self methods cannot capture it"
                )
            aliases[name] = f"__self_{name}"
        return aliases

    def generate_fn(self, fn: FnDecl, *, as_method: bool) -> None:
        from termination import check_function

        check_function(fn)
        self.set_origin(fn)
        previous_string_bindings = self.string_bindings
        self.string_bindings = set(previous_string_bindings)
        previous_binding_types = self.binding_types
        self.binding_types = {
            param.pattern.name: name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            for name in [self._binding_type_name(param.ty)]
            if name is not None
        }
        previous_vec_bindings = self.vec_bindings
        self.vec_bindings = set(previous_vec_bindings)
        self.string_bindings.update(
            param.pattern.name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            and isinstance(param.ty, TypeName)
            and param.ty.name == "String"
        )
        self.vec_bindings.update(
            param.pattern.name
            for param in fn.params
            if isinstance(param.pattern, PatIdent)
            and isinstance(param.ty, TypeName)
            and param.ty.name == "Vec"
        )
        previous_result_types = self._result_types
        previous_return_type = self._current_return_type
        self._current_return_type = (
            self._binding_type_name(fn.ret) if fn.ret is not None else None
        )
        if (
            fn.ret is not None
            and isinstance(fn.ret, TypeName)
            and fn.ret.name == "Result"
            and len(fn.ret.args) == 2
        ):
            self._result_types = (
                self.type_to_rust(fn.ret.args[0]),
                self.type_to_rust(fn.ret.args[1]),
            )
        else:
            self._result_types = None

        is_ctor = as_method and fn.name == "new"

        if not as_method:
            self.emit("#[pyfunction]")
            self.exported_fns.append(fn.name)
        elif is_ctor:
            self.emit("#[new]")
        elif not any(self._is_self_param(p) for p in fn.params):
            # Associated function without a receiver, e.g. `Vec::new()`-style
            # constructors on user structs: expose it as a static method.
            self.emit("#[staticmethod]")

        params = [self.param_to_rust(p, as_method=as_method) for p in fn.params]
        param_str = ", ".join(params)

        ret = ""
        if fn.ret is not None:
            if (
                isinstance(fn.ret, TypeName)
                and fn.ret.name == "Result"
                and len(fn.ret.args) == 2
            ):
                ok_ty = self.type_to_rust(fn.ret.args[0])
                ret = f" -> PyResult<{ok_ty}>"
            else:
                ret = f" -> {self.type_to_rust(fn.ret)}"
        elif is_ctor:
            ret = " -> Self"

        unsafe_prefix = "unsafe " if fn.is_unsafe else ""

        receiver_kind = self._async_receiver_kind(fn) if as_method else None
        if fn.is_async and receiver_kind == "value":
            raise TranslateError(
                f"async method '{fn.name}' takes `self` by value; async methods "
                "must take `&self`, `&mut self`, or no receiver"
            )

        if fn.is_async:
            aliases: dict[str, str] = {}
            if receiver_kind == "shared" and fn.body is not None:
                aliases = self._prepare_async_self_fields(fn)
            elif receiver_kind == "mut" and fn.body is not None:
                aliases = self._prepare_async_mut_fields(fn)
                other_params = [
                    p for p in param_str.split(", ") if "&mut self" not in p
                ]
                sig_parts = [
                    "slf: pyo3::Py<Self>",
                    "py: pyo3::Python<'_>",
                    *other_params,
                ]
                param_str = ", ".join(sig_parts)

            self.emit(f"fn {fn.name}({param_str}) -> PyResult<PyFuture> {{")
            self.indent()
            previous_aliases = self._self_field_aliases
            self._self_field_aliases = aliases

            if receiver_kind == "mut":
                impl_name = self._current_impl_type or "Self"
                self.emit(
                    "let __borrow_state = slf.borrow(py).__polyglot_borrow_state.clone();"
                )
                self.emit(
                    f'let __borrow_guard = PolyglotBorrowGuard::acquire_mut(&__borrow_state, "{impl_name}")?;'
                )
                self.emit("{")
                self.indent()
                self.emit("let __this = slf.borrow(py);")
                for name, alias in aliases.items():
                    self.emit(f"let mut {alias} = __this.{name}.clone();")
                self.dedent()
                self.emit("}")
            else:
                for name, alias in aliases.items():
                    self.emit(f"let {alias} = self.{name}.clone();")

            self.emit("Ok(py_future(async move {")
            self.indent()
            if receiver_kind == "mut":
                self.emit("let _guard = __borrow_guard;")
                self.emit("let __async_res = async {")
                self.indent()

            if fn.body is not None:
                self.generate_block_body(
                    fn.body,
                    as_fn_body=True,
                    async_wrap=True,
                )
            else:
                self.emit("Err(pyo3::exceptions::PyRuntimeError::new_err(")
                self.emit('    "async function has no body",')
                self.emit("))")

            if receiver_kind == "mut":
                self.dedent()
                self.emit("}.await;")
                self.emit("if __async_res.is_ok() {")
                self.indent()
                self.emit("pyo3::Python::with_gil(|py| {")
                self.indent()
                self.emit("let mut __this = slf.borrow_mut(py);")
                for name, alias in aliases.items():
                    self.emit(f"__this.{name} = {alias};")
                self.dedent()
                self.emit("});")
                self.dedent()
                self.emit("}")
                self.emit("__async_res")

            self.dedent()
            self.emit("}))")
            self._self_field_aliases = previous_aliases
            self.dedent()
            self.emit("}")
            self.emit()
        else:
            self.emit(f"{unsafe_prefix}fn {fn.name}({param_str}){ret} {{")
            self.indent()
            if fn.body is not None:
                self.generate_block_body(fn.body, as_fn_body=True)
                if (
                    fn.ret is not None
                    and not _is_unit_type(fn.ret)
                    and not fn.body.stmts
                    and fn.body.tail is None
                ):
                    self.emit("unreachable!()")
            else:
                self.emit("unreachable!()")
            self.dedent()
            self.emit("}")
            self.emit()
        self.string_bindings = previous_string_bindings
        self.vec_bindings = previous_vec_bindings
        self.binding_types = previous_binding_types
        self._result_types = previous_result_types
        self._current_return_type = previous_return_type

    def generate_struct(self, s: StructDecl) -> None:
        self.set_origin(s)
        self.exported_classes.append(s.name)
        self.emit("#[pyclass]")
        self.emit("#[derive(Clone, Debug)]")
        self.emit(f"struct {s.name} {{")
        self.indent()
        for name, ty, _is_pub in s.fields:
            self.emit("#[pyo3(get, set)]")
            self.emit(f"{name}: {self.type_to_rust(ty)},")
        if s.name in self.borrow_state_types:
            self.emit(
                "__polyglot_borrow_state: std::sync::Arc<std::sync::atomic::AtomicU8>,"
            )
        self.dedent()
        self.emit("}")
        self.emit()

    def generate_enum(self, e: EnumDecl) -> None:
        self.set_origin(e)
        is_unit_enum = all(variant.fields is None for variant in e.variants)
        if is_unit_enum:
            self.exported_classes.append(e.name)
            self.emit("#[pyclass(eq, eq_int)]")
            self.emit("#[derive(Debug, PartialEq, Eq, Clone)]")
        self.emit(f"enum {e.name} {{")
        self.indent()
        for v in e.variants:
            if v.fields is None:
                self.emit(f"{v.name},")
            else:
                fields = ", ".join(self.type_to_rust(t) for t in v.fields)
                self.emit(f"{v.name}({fields}),")
        self.dedent()
        self.emit("}")
        self.emit()

    def generate_impl(self, impl: ImplBlock) -> None:
        self.generate_impls([impl])

    def generate_impls(self, impls: list[ImplBlock]) -> None:
        if not impls:
            return
        self.set_origin(impls[0])
        target = self.type_to_rust(impls[0].ty)
        use_pymethods = target in self.pyclass_names

        if use_pymethods:
            self.emit("#[pymethods]")
            self._in_pymethods = True

        previous_impl_type = self._current_impl_type
        self._current_impl_type = (
            impls[0].ty.name if isinstance(impls[0].ty, TypeName) else None
        )
        self.emit(f"impl {target} {{")
        self.indent()
        for impl in impls:
            for item in impl.items:
                if isinstance(item, FnDecl):
                    self.generate_fn(item, as_method=True)
        self.dedent()
        self.emit("}")
        self.emit()
        self._current_impl_type = previous_impl_type
        self._in_pymethods = False

    def generate_const(self, c: ConstDecl) -> None:
        ty = self.type_to_rust(c.ty) if c.ty else "_"
        val = self.expr_to_rust(c.value)
        keyword = "static" if c.is_static else "const"
        self.emit(f"{keyword} {c.name}: {ty} = {val};")

    # ---- statements / blocks ----------------------------------------------

    def generate_block_body(
        self,
        block: Block,
        *,
        as_fn_body: bool = False,
        async_wrap: bool = False,
    ) -> None:
        for stmt in block.stmts:
            self.generate_stmt(stmt)
        if block.tail is not None:
            self.set_origin(block.tail)
            if as_fn_body and isinstance(block.tail, IfLet):
                self._check_type_boundary(
                    self._expr_type(block.tail),
                    self._current_return_type,
                    block.tail,
                )
                self._emit_if_let(block.tail)
                return
            if as_fn_body:
                self._check_type_boundary(
                    self._expr_type(block.tail),
                    self._current_return_type,
                    block.tail,
                )
            tail = (
                self._return_expr_to_rust(block.tail)
                if as_fn_body
                else self.expr_to_rust(block.tail)
            )
            if async_wrap:
                self.emit(f"pyo3::Python::with_gil(|py| {tail}.into_py_any(py))")
            else:
                self.emit(tail)
        elif as_fn_body:
            if async_wrap:
                self.emit("pyo3::Python::with_gil(|py| ().into_py_any(py))")
            elif not block.stmts:
                self.emit("()")

    def generate_stmt(self, stmt: Stmt) -> None:
        self.set_origin(stmt)
        if isinstance(stmt, LetStmt):
            self.emit_let(stmt)
        elif isinstance(stmt, ExprStmt):
            expr = stmt.expr
            if isinstance(expr, IfExpr):
                self._emit_if(expr, as_stmt=True)
            elif isinstance(expr, IfLet):
                self._emit_if_let(expr)
            elif isinstance(expr, While):
                self._emit_while(expr)
            elif isinstance(expr, For):
                self._emit_for(expr)
            elif isinstance(expr, Loop):
                self._emit_loop(expr)
            elif isinstance(expr, MatchExpr):
                self._emit_match(expr, as_stmt=True)
            elif isinstance(expr, Assign):
                if isinstance(expr.target, Ident):
                    self._check_type_boundary(
                        self._expr_type(expr.value),
                        self.binding_types.get(expr.target.name),
                        expr.value,
                    )
                self.emit(
                    f"{self.expr_to_rust(expr.target)} = "
                    f"{self.expr_to_rust(expr.value)};"
                )
            elif isinstance(expr, CompoundAssign):
                self.emit(
                    f"{self.expr_to_rust(expr.target)} {expr.op} "
                    f"{self.expr_to_rust(expr.value)};"
                )
            elif isinstance(expr, Return):
                if expr.value is None:
                    self.emit("return;")
                else:
                    self._check_type_boundary(
                        self._expr_type(expr.value),
                        self._current_return_type,
                        expr.value,
                    )
                    self.emit(f"return {self._return_expr_to_rust(expr.value)};")
            elif isinstance(expr, Break):
                self.emit("break;")
            elif isinstance(expr, Continue):
                self.emit("continue;")
            elif isinstance(expr, Block):
                self.emit("{")
                self.indent()
                self.generate_block_body(expr)
                self.dedent()
                self.emit("}")
            else:
                self.emit(f"{self.expr_to_rust(expr)};")
        elif isinstance(stmt, ItemStmt):
            self.generate_item(stmt.item)
        else:
            self.emit(f"/* unsupported stmt {type(stmt).__name__} */")

    def infer_type(self, expr: Expr) -> str | None:
        if isinstance(expr, MethodCall):
            receiver_type = self._expr_type(expr.receiver)
            if expr.method == "len":
                if receiver_type in {"Vec", "String", "HashMap", "BTreeMap", "HashSet"}:
                    return "usize"
                return self._method_return_types.get(
                    (receiver_type, expr.method),
                )
            if expr.method in {"get", "get_mut"} and self._map_kind(expr.receiver):
                return "Option"
            if expr.method == "unwrap" and isinstance(expr.receiver, MethodCall):
                inner = expr.receiver
                if inner.method == "get":
                    kind = self._map_kind(inner.receiver)
                    if kind in {"HashMap", "BTreeMap"}:
                        return kind
            return self._method_return_types.get(
                (receiver_type, expr.method),
            )
        if isinstance(expr, Cast) and isinstance(expr.ty, TypeName):
            return expr.ty.name
        if isinstance(expr, Range):
            return (self._expr_type(expr.end) if expr.end is not None else None) or (
                self._expr_type(expr.start) if expr.start is not None else None
            )
        if isinstance(expr, Binary):
            left_type = self._expr_type(expr.left)
            right_type = self._expr_type(expr.right)
            if left_type is not None and right_type is not None:
                self._check_type_boundary(
                    right_type,
                    left_type,
                    expr,
                )
            return left_type or right_type
        return None

    def _adapt_host_result(self, expr: Expr, emitted: str) -> str:
        """Convert a bare std:fs host result to the generated PyResult type.

        A function returning ``Result<T, String>`` compiles to
        ``PyResult<T>``. Explicit Ok/Err nodes already emit PyResult, but a
        host call returns the language-level Result; convert only those bare
        calls here.
        """
        if (
            self._current_return_type == "Result"
            and isinstance(expr, Call)
            and isinstance(expr.func, Ident)
            and expr.func.name in HOST_CALL_ARITIES
        ):
            return (
                f"({emitted}).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))"
            )
        return emitted

    def _return_expr_to_rust(self, expr: Expr) -> str:
        emitted = self.expr_to_rust(expr)
        emitted = self._adapt_host_result(expr, emitted)
        if (
            self._current_return_type == "Option"
            and isinstance(expr, MethodCall)
            and expr.method == "get_mut"
            and self._map_kind(expr.receiver) in {"HashMap", "BTreeMap"}
        ):
            return f"({emitted}).map(|value| (*value).clone())"
        return emitted

    def emit_cast(self, node: Cast) -> str:
        target = self.type_to_rust(node.ty)
        source = self._expr_type(node.expr)
        if target == "bool":
            if source != "bool":
                raise InvalidCastError(
                    f"cannot cast `{source or 'unknown'}` as `bool`",
                    node.line,
                    node.col,
                )
        elif target == "char" and source != "u8":
            raise InvalidCastError(
                f"only `u8` can be cast as `char`, not `{source or 'unknown'}`",
                node.line,
                node.col,
            )
        return f"({self.expr_to_rust(node.expr)} as {target})"

    def _emit_usize_arg(self, expr: Expr) -> str:
        self._check_type_boundary(self._expr_type(expr), "usize", expr)
        return self.expr_to_rust(expr)

    def _expr_type(self, expr: Expr) -> str | None:
        inferred = self.infer_type(expr)
        if inferred is not None:
            return inferred
        if isinstance(expr, Ident):
            return self.binding_types.get(expr.name)
        if isinstance(expr, StructLiteral):
            return expr.path.split("::")[-1]
        if isinstance(expr, Field):
            receiver_type = self._expr_type(expr.receiver)
            return self.struct_field_types.get(receiver_type or "", {}).get(
                expr.name,
            )
        if isinstance(expr, Call):
            if isinstance(expr.func, Ident):
                return self._function_return_types.get(expr.func.name)
            if isinstance(expr.func, Path) and len(expr.func.segments) == 1:
                return self._function_return_types.get(expr.func.segments[0])
        if isinstance(expr, Call) and isinstance(expr.func, Path):
            segments = expr.func.segments
            if len(segments) >= 2:
                target = segments[-2]
                if target == "Self" and self._current_impl_type is not None:
                    target = self._current_impl_type
                return self._method_return_types.get((target, segments[-1]))
        if isinstance(expr, MatchExpr):
            known: list[str] = []
            for arm in expr.arms:
                arm_type = self._expr_type(arm.body)
                if arm_type is not None and arm_type not in known:
                    known.append(arm_type)
            if len(known) == 1:
                return known[0]
        return None

    def _check_call_arguments(self, call: Call) -> None:
        if isinstance(call.func, Ident):
            name = call.func.name
        elif isinstance(call.func, Path):
            name = call.func.segments[-1]
        else:
            return
        check_host_call_arity(
            name,
            call.args,
            line=call.line,
            col=call.col,
            span=call.span,
        )
        parameter_types = self._function_signatures.get(name)
        if parameter_types is None:
            return
        for argument, expected_type in zip(call.args, parameter_types):
            self._check_type_boundary(
                self._expr_type(argument),
                expected_type,
                argument,
            )

    def _check_type_boundary(
        self,
        source_type: str | None,
        expected_type: str | None,
        node: Node,
    ) -> None:
        if source_type is None or expected_type is None:
            return
        # M4: implicit conversion between different integer types requires `as`
        _INT_TYPE_NAMES = {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "isize",
        }
        if source_type in _INT_TYPE_NAMES and expected_type in _INT_TYPE_NAMES:
            if source_type != expected_type:
                from backends.rust.compiler import TypeErrorForBoundary

                raise TypeErrorForBoundary(
                    source_type,
                    expected_type,
                    node.line,
                    node.col,
                    self.source_lines.get(node.line),
                    span=getattr(node, "span", None),
                )
            return
        source_ty = TypeName(node.line, node.col, source_type)
        expected_ty = TypeName(node.line, node.col, expected_type)
        source_is_pointer_sized = is_usize_type(source_ty) or is_isize_type(source_ty)
        expected_is_pointer_sized = is_usize_type(expected_ty) or is_isize_type(
            expected_ty
        )
        if (
            source_is_pointer_sized
            and is_i64_type(expected_ty)
            or expected_is_pointer_sized
            and is_i64_type(source_ty)
            or (
                source_is_pointer_sized
                and expected_is_pointer_sized
                and source_type != expected_type
            )
        ):
            from backends.rust.compiler import TypeErrorForBoundary

            raise TypeErrorForBoundary(
                source_type,
                expected_type,
                node.line,
                node.col,
                self.source_lines.get(node.line),
                span=getattr(node, "span", None),
            )

    def emit_let(self, stmt: LetStmt) -> None:
        mut = "mut " if stmt.mutable else ""
        pat = self.pattern_to_rust(stmt.pattern)
        ty = f": {self.type_to_rust(stmt.ty)}" if stmt.ty else ""
        inferred_type = self._expr_type(stmt.value) if stmt.value is not None else None
        expected_type = self._binding_type_name(stmt.ty)
        self._check_type_boundary(inferred_type, expected_type, stmt)
        previous_result_types = self._result_types
        if (
            stmt.ty is not None
            and isinstance(stmt.ty, TypeName)
            and stmt.ty.name == "Result"
            and len(stmt.ty.args) == 2
        ):
            self._result_types = (
                self.type_to_rust(stmt.ty.args[0]),
                self.type_to_rust(stmt.ty.args[1]),
            )
        if stmt.value is not None:
            self.emit(f"let {mut}{pat}{ty} = {self.expr_to_rust(stmt.value)};")
            if isinstance(stmt.pattern, PatIdent):
                binding_type = (
                    expected_type
                    or inferred_type
                    or self._expr_variant_type(stmt.value)
                )
                if binding_type is not None:
                    self.binding_types[stmt.pattern.name] = binding_type
            if isinstance(stmt.pattern, PatIdent) and self._is_string_expr(stmt.value):
                self.string_bindings.add(stmt.pattern.name)
            if isinstance(stmt.pattern, PatIdent) and (
                isinstance(stmt.ty, TypeName)
                and stmt.ty.name == "Vec"
                or self._is_vec_expr(stmt.value)
            ):
                self.vec_bindings.add(stmt.pattern.name)
        else:
            self.emit(f"let {mut}{pat}{ty};")
            if isinstance(stmt.pattern, PatIdent) and expected_type is not None:
                self.binding_types[stmt.pattern.name] = expected_type
        self._result_types = previous_result_types

    def _emit_if(self, ifx: IfExpr, *, as_stmt: bool) -> None:
        self.emit(f"if {self.expr_to_rust(ifx.cond)} {{")
        self.indent()
        self.generate_block_body(ifx.then_block)
        self.dedent()
        branch = ifx.else_branch
        while isinstance(branch, IfExpr):
            self.emit(f"}} else if {self.expr_to_rust(branch.cond)} {{")
            self.indent()
            self.generate_block_body(branch.then_block)
            self.dedent()
            branch = branch.else_branch
        if isinstance(branch, Block):
            self.emit("} else {")
            self.indent()
            self.generate_block_body(branch)
            self.dedent()
            self.emit("}")
        else:
            self.emit("}")

    def _emit_if_let(self, ifx: IfLet) -> None:
        self.emit(
            f"if let {self.pattern_to_rust(ifx.pattern)} = "
            f"{self.expr_to_rust(ifx.value)} {{"
        )
        self.indent()
        self.generate_block_body(ifx.then_block)
        self.dedent()
        if isinstance(ifx.else_branch, Block):
            self.emit("} else {")
            self.indent()
            self.generate_block_body(ifx.else_branch)
            self.dedent()
        elif isinstance(ifx.else_branch, IfLet):
            self.emit("} else {")
            self.indent()
            self._emit_if_let(ifx.else_branch)
            self.dedent()
        elif isinstance(ifx.else_branch, IfExpr):
            self.emit("} else {")
            self.indent()
            self._emit_if(ifx.else_branch, as_stmt=True)
            self.dedent()
        self.emit("}")

    def _emit_while(self, w: While) -> None:
        self.emit(f"while {self.expr_to_rust(w.cond)} {{")
        self.indent()
        self.generate_block_body(w.body)
        self.dedent()
        self.emit("}")

    def _emit_for(self, f: For) -> None:
        pat = self.pattern_to_rust(f.pattern)
        self.emit(f"for {pat} in {self.expr_to_rust(f.iter)} {{")
        self.indent()
        if isinstance(f.pattern, PatIdent):
            name = f.pattern.name
            previous_type = self.binding_types.get(name)
            loop_type = self._expr_type(f.iter)
            if loop_type is None:
                self.binding_types.pop(name, None)
            else:
                self.binding_types[name] = loop_type
            try:
                self.generate_block_body(f.body)
            finally:
                if previous_type is None:
                    self.binding_types.pop(name, None)
                else:
                    self.binding_types[name] = previous_type
        else:
            self.generate_block_body(f.body)
        self.dedent()
        self.emit("}")

    def _emit_loop(self, lp: Loop) -> None:
        self.emit("loop {")
        self.indent()
        self.generate_block_body(lp.body)
        self.dedent()
        self.emit("}")

    def _emit_match(self, m: MatchExpr, *, as_stmt: bool) -> None:
        warn_match_issues(
            m,
            scrutinee_type=self._match_scrutinee_type(m),
        )
        scrutinee = self.expr_to_rust(m.scrutinee)
        if any(isinstance(arm.pattern, PatSlice) for arm in m.arms):
            scrutinee = f"({scrutinee}).as_slice()"
        self.emit(f"match {scrutinee} {{")
        self.indent()
        for arm in m.arms:
            self._emit_match_arm(arm)
        self.dedent()
        self.emit("}")

    def _emit_match_arm(self, arm: MatchArm) -> None:
        pat = self.pattern_to_rust(arm.pattern)
        guard = ""
        if arm.guard is not None:
            guard = f" if {self.expr_to_rust(arm.guard)}"
        body = arm.body
        if isinstance(body, Block):
            self.emit(f"{pat}{guard} => {{")
            self.indent()
            self.generate_block_body(body)
            self.dedent()
            self.emit("}")
        else:
            self.emit(f"{pat}{guard} => {self.expr_to_rust(body)},")

    def _format_str(self, e: Expr) -> str:
        """Emit a bare string literal for the first arg of format!/println!."""
        if isinstance(e, LitStr):
            escaped = e.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return self.expr_to_rust(e)

    # ---- expressions ------------------------------------------------------

    def expr_to_rust(self, e: Expr) -> str:
        if isinstance(e, LitInt):
            return e.raw
        if isinstance(e, LitFloat):
            return e.raw
        if isinstance(e, LitStr):
            escaped = e.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'String::from("{escaped}")'
        if isinstance(e, LitChar):
            return f"'{e.value}'"
        if isinstance(e, LitBool):
            return "true" if e.value else "false"
        if isinstance(e, Ident):
            return e.name
        if isinstance(e, Path):
            return "::".join(e.segments)
        if isinstance(e, Paren):
            return f"({self.expr_to_rust(e.inner)})"
        if isinstance(e, OptionSome):
            return f"Some({self.expr_to_rust(e.value)})"
        if isinstance(e, OptionNone):
            return "None"
        if isinstance(e, ResultOk):
            if self._result_types is not None:
                ok_ty, _err_ty = self._result_types
                return f"Ok::<{ok_ty}, PyErr>({self.expr_to_rust(e.value)})"
            return f"Ok::<_, PyErr>({self.expr_to_rust(e.value)})"
        if isinstance(e, ResultErr):
            if self._result_types is not None:
                ok_ty, _err_ty = self._result_types
                return (
                    f"Err::<{ok_ty}, PyErr>("
                    f"pyo3::exceptions::PyRuntimeError::new_err("
                    f"{self.expr_to_rust(e.value)}))"
                )
            return (
                f"Err::<i64, PyErr>(pyo3::exceptions::PyRuntimeError::new_err("
                f"{self.expr_to_rust(e.value)}))"
            )
        if isinstance(e, Unary):
            if (
                e.op == "&"
                and isinstance(e.operand, Index)
                and isinstance(e.operand.index, Range)
            ):
                return self.expr_to_rust(e.operand)
            if (
                e.op == "-"
                and isinstance(e.operand, Cast)
                and isinstance(e.operand.ty, TypeName)
                and e.operand.ty.name in {"isize", "usize"}
            ):
                return self.emit_cast(
                    Cast(
                        e.operand.line,
                        e.operand.col,
                        Unary(e.line, e.col, "-", e.operand.expr),
                        e.operand.ty,
                    )
                )
            op = "&mut " if e.op == "&mut" else e.op
            return f"({op}{self.expr_to_rust(e.operand)})"
        if isinstance(e, Binary):
            if e.op == "+" and (
                self._is_string_expr(e.left) or self._is_string_expr(e.right)
            ):
                return (
                    f'format!("{{}}{{}}", {self.expr_to_rust(e.left)}, '
                    f"{self.expr_to_rust(e.right)})"
                )
            return f"({self.expr_to_rust(e.left)} {e.op} {self.expr_to_rust(e.right)})"
        if isinstance(e, Assign):
            return f"({self.expr_to_rust(e.target)} = {self.expr_to_rust(e.value)})"
        if isinstance(e, CompoundAssign):
            return (
                f"({self.expr_to_rust(e.target)} {e.op} {self.expr_to_rust(e.value)})"
            )
        if isinstance(e, Call):
            self._check_call_arguments(e)
            # String literals already lower to String::from("..."). Wrapping
            # that again would emit String::from(String::from("...")).
            if (
                isinstance(e.func, Path)
                and e.func.segments == ["String", "from"]
                and len(e.args) == 1
                and isinstance(e.args[0], LitStr)
            ):
                return self.expr_to_rust(e.args[0])
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            return f"{self.expr_to_rust(e.func)}({args})"
        if isinstance(e, MethodCall):
            recv = self.expr_to_rust(e.receiver)
            if self._is_user_method_call(e):
                args = ", ".join(self.expr_to_rust(a) for a in e.args)
                return f"{recv}.{e.method}({args})"
            kind = self._map_kind(e.receiver)
            if kind in {"HashMap", "BTreeMap"}:
                if e.method == "insert" and len(e.args) == 2:
                    return (
                        f"{recv}.insert("
                        f"{self._owned_arg_to_rust(e.args[0])}, "
                        f"{self.expr_to_rust(e.args[1])})"
                    )
                if e.method == "get" and len(e.args) == 1:
                    return f"{recv}.get({self._ref_arg_to_rust(e.args[0])}).cloned()"
                if e.method == "get_mut" and len(e.args) == 1:
                    return f"{recv}.get_mut({self._ref_arg_to_rust(e.args[0])})"
                if e.method == "remove" and len(e.args) == 1:
                    return f"{recv}.remove({self._ref_arg_to_rust(e.args[0])})"
                if e.method == "contains_key" and len(e.args) == 1:
                    return f"{recv}.contains_key({self._ref_arg_to_rust(e.args[0])})"
                if e.method == "entry" and len(e.args) == 1:
                    return f"{recv}.entry({self._owned_arg_to_rust(e.args[0])})"
                if e.method == "keys" and not e.args:
                    return f"{recv}.keys().cloned().collect::<Vec<_>>()"
                if e.method == "into_keys" and not e.args:
                    return f"{recv}.into_iter().map(|(key, _)| key).collect::<Vec<_>>()"
                if e.method == "into_values" and not e.args:
                    return (
                        f"{recv}.into_iter().map(|(_, value)| value)"
                        ".collect::<Vec<_>>()"
                    )
                if e.method == "values" and not e.args:
                    return f"{recv}.values().cloned().collect::<Vec<_>>()"
                if e.method == "iter" and not e.args:
                    return (
                        f"{recv}.iter()"
                        ".map(|(key, value)| "
                        "(key.clone(), value.clone()))"
                        ".collect::<Vec<_>>()"
                    )
                if e.method == "iter_mut" and not e.args:
                    return f"{recv}.iter_mut()"
                if e.method == "values_mut" and not e.args:
                    return f"{recv}.values_mut()"
                if e.method == "drain" and not e.args:
                    return f"{recv}.drain().collect::<Vec<_>>()"
                if e.method == "extend" and len(e.args) == 1:
                    other = self.expr_to_rust(e.args[0])
                    return f"{recv}.extend({other}.into_iter())"
                if e.method == "get_key_value" and len(e.args) == 1:
                    key = self._ref_arg_to_rust(e.args[0])
                    return (
                        f"{recv}.get_key_value({key})"
                        ".map(|(key, value)| (key.clone(), value.clone()))"
                    )
                if e.method == "remove_entry" and len(e.args) == 1:
                    key = self._ref_arg_to_rust(e.args[0])
                    return f"{recv}.remove_entry({key})"
            if kind == "HashSet":
                if e.method == "insert" and len(e.args) == 1:
                    return f"{recv}.insert({self._owned_arg_to_rust(e.args[0])})"
                if e.method == "contains" and len(e.args) == 1:
                    return f"{recv}.contains({self._ref_arg_to_rust(e.args[0])})"
                if e.method == "remove" and len(e.args) == 1:
                    return f"{recv}.remove({self._ref_arg_to_rust(e.args[0])})"
                if e.method == "get_mut" and len(e.args) == 1:
                    return (
                        "polyglot_hashset_get_mut("
                        f"&({recv}), {self._ref_arg_to_rust(e.args[0])})"
                    )
                if e.method == "iter" and not e.args:
                    return f"{recv}.iter().cloned().collect::<Vec<_>>()"
            if self._is_map_entry_expr(e.receiver):
                if e.method == "key" and not e.args:
                    return f"{recv}.key().clone()"
                if e.method == "or_insert_with_key" and len(e.args) == 1:
                    return f"{recv}.or_insert_with_key({self.expr_to_rust(e.args[0])})"
                if e.method == "or_insert" and len(e.args) == 1:
                    return f"{recv}.or_insert({self.expr_to_rust(e.args[0])})"
                if e.method == "or_insert_with" and len(e.args) == 1:
                    return f"{recv}.or_insert_with({self.expr_to_rust(e.args[0])})"
                if e.method == "or_default" and not e.args:
                    return f"{recv}.or_default()"
                if e.method == "and_modify" and len(e.args) == 1:
                    return f"{recv}.and_modify({self.expr_to_rust(e.args[0])})"
            if e.method == "ok_or" and len(e.args) == 1:
                error = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.ok_or_else(|| "
                    "pyo3::exceptions::PyRuntimeError::new_err("
                    f"{error}))"
                )
            if e.method == "ok_or_else" and len(e.args) == 1:
                factory = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.ok_or_else(|| "
                    "pyo3::exceptions::PyRuntimeError::new_err("
                    f"({factory})()))"
                )
            if e.method == "map_err" and len(e.args) == 1:
                return f"{recv}.map_err({self.expr_to_rust(e.args[0])})"
            if e.method == "unwrap" and not e.args:
                return f"{recv}.unwrap()"
            if e.method == "unwrap_err" and not e.args:
                return f"{recv}.unwrap_err()"
            if e.method == "expect" and len(e.args) == 1:
                message = self.expr_to_rust(e.args[0])
                return f"{recv}.expect(&{message})"
            if e.method == "expect_err" and len(e.args) == 1:
                message = self.expr_to_rust(e.args[0])
                return f"{recv}.expect_err(&{message})"
            if (
                e.method == "is_ok_and"
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Result"
            ):
                return f"{recv}.is_ok_and({self.expr_to_rust(e.args[0])})"
            if (
                e.method == "is_err_and"
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Result"
            ):
                return f"{recv}.is_err_and({self.expr_to_rust(e.args[0])})"
            if (
                e.method == "map_or"
                and len(e.args) == 2
                and self._option_result_type(e.receiver) == "Result"
            ):
                first = self.expr_to_rust(e.args[0])
                second = self.expr_to_rust(e.args[1])
                return f"{recv}.map_or({first}, {second})"
            if (
                e.method == "and"
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Result"
            ):
                return f"{recv}.and({self.expr_to_rust(e.args[0])})"
            if (
                e.method == "or"
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Result"
            ):
                return f"{recv}.or({self.expr_to_rust(e.args[0])})"
            if e.method == "unwrap_or" and len(e.args) == 1:
                return f"{recv}.unwrap_or({self.expr_to_rust(e.args[0])})"
            if e.method == "unwrap_or_else" and len(e.args) == 1:
                return f"{recv}.unwrap_or_else({self.expr_to_rust(e.args[0])})"
            if (
                e.method == "iter"
                and not e.args
                and self._option_result_type(e.receiver) == "Option"
            ):
                return f"{recv}.iter().cloned().collect::<Vec<_>>()"
            if (
                e.method in ("and", "or", "xor")
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Option"
            ):
                return f"{recv}.{e.method}({self.expr_to_rust(e.args[0])})"
            if (
                e.method
                in (
                    "insert",
                    "get_or_insert",
                    "get_or_insert_with",
                    "is_some_and",
                    "is_none_or",
                )
                and len(e.args) == 1
                and self._option_result_type(e.receiver) == "Option"
            ):
                return f"{recv}.{e.method}({self.expr_to_rust(e.args[0])})"
            if (
                e.method in ("map_or", "map_or_else")
                and len(e.args) == 2
                and self._option_result_type(e.receiver) == "Option"
            ):
                first = self.expr_to_rust(e.args[0])
                second = self.expr_to_rust(e.args[1])
                return f"{recv}.{e.method}({first}, {second})"
            if e.method == "chars" and not e.args:
                return f"{recv}.chars().collect::<Vec<char>>()"
            if e.method == "char_indices" and not e.args:
                return (
                    f"{recv}.char_indices()"
                    ".map(|(index, character)| (index as i64, character))"
                    ".collect::<Vec<_>>()"
                )
            if (
                e.method == "rev"
                and not e.args
                and isinstance(e.receiver, MethodCall)
                and e.receiver.method == "chars"
            ):
                inner = self.expr_to_rust(e.receiver.receiver)
                return f"{inner}.chars().rev().collect::<Vec<char>>()"
            if e.method == "rev" and not e.args:
                return f"{recv}.iter().rev().cloned().collect::<Vec<_>>()"
            if (
                e.method == "skip"
                and len(e.args) == 1
                and self._is_vec_expr(e.receiver)
            ):
                arg = self.expr_to_rust(e.args[0])
                return f"{recv}.iter().skip({arg}).cloned().collect::<Vec<_>>()"
            if e.method == "enumerate" and not e.args and self._is_vec_expr(e.receiver):
                return (
                    f"{recv}.iter().enumerate()"
                    ".map(|(index, value)| (index as i64, value.clone()))"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "zip" and len(e.args) == 1 and self._is_vec_expr(e.receiver):
                other = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.iter().zip({other}.iter())"
                    ".map(|(left, right)| (left.clone(), right.clone()))"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "flatten" and not e.args and self._is_vec_expr(e.receiver):
                return f"{recv}.iter().flatten().cloned().collect::<Vec<_>>()"
            if (
                e.method == "chain"
                and len(e.args) == 1
                and self._is_vec_expr(e.receiver)
            ):
                other = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.iter().chain({other}.iter()).cloned().collect::<Vec<_>>()"
                )
            if (
                e.method == "step_by"
                and len(e.args) == 1
                and self._is_vec_expr(e.receiver)
            ):
                step = self.expr_to_rust(e.args[0])
                return f"{recv}.iter().step_by({step}).cloned().collect::<Vec<_>>()"
            if (
                e.method == "split_first"
                and not e.args
                and self._is_vec_expr(e.receiver)
            ):
                return (
                    f"{recv}.split_first().map(|(first, rest)| (*first, rest.to_vec()))"
                )
            if (
                e.method == "split_last"
                and not e.args
                and self._is_vec_expr(e.receiver)
            ):
                return f"{recv}.split_last().map(|(last, rest)| (*last, rest.to_vec()))"
            if e.method == "bytes" and not e.args:
                return f"{recv}.bytes().collect::<Vec<u8>>()"
            if e.method == "parse" and not e.args:
                target = (
                    self.type_to_rust(e.generic_args[0]) if e.generic_args else "i64"
                )
                if (
                    target == "i64"
                    and isinstance(e.receiver, LitStr)
                    and not _is_valid_i64_literal(e.receiver.value)
                ):
                    raise InvalidIntegerParseLiteralError(
                        e.receiver.value,
                        e.receiver.line,
                        e.receiver.col,
                        getattr(e.receiver, "span", None),
                    )
                return f"{recv}.parse::<{target}>().unwrap()"
            if e.method == "split" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.split(&{separator}).map(String::from).collect::<Vec<_>>()"
                )
            if e.method in ("splitn", "rsplitn") and len(e.args) == 2:
                count = self.expr_to_rust(e.args[0])
                separator = self.expr_to_rust(e.args[1])
                return (
                    f"{recv}.{e.method}({count}, &{separator})"
                    ".map(String::from).collect::<Vec<_>>()"
                )
            if e.method == "rsplit" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.rsplit(&{separator}).map(String::from).collect::<Vec<_>>()"
                )
            if e.method == "split_terminator" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.split_terminator(&{separator})"
                    ".map(String::from).collect::<Vec<_>>()"
                )
            if e.method == "rsplit_terminator" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.rsplit_terminator(&{separator})"
                    ".map(String::from).collect::<Vec<_>>()"
                )
            if e.method in ("matches", "rmatches") and len(e.args) == 1:
                pattern = self.expr_to_rust(e.args[0])
                return (
                    f"{recv}.{e.method}(&{pattern})"
                    ".map(String::from).collect::<Vec<_>>()"
                )
            if e.method == "replacen" and len(e.args) == 3:
                old = self.expr_to_rust(e.args[0])
                new = self.expr_to_rust(e.args[1])
                count = self.expr_to_rust(e.args[2])
                return f"{recv}.replacen(&{old}, &{new}, {count})"
            if e.method == "join" and len(e.args) == 1:
                separator = self.expr_to_rust(e.args[0])
                return f"{recv}.join(&{separator})"
            if e.method == "trim" and not e.args:
                return f"{recv}.trim().to_string()"
            if e.method in ("trim_start", "trim_end") and not e.args:
                return f"{recv}.{e.method}().to_string()"
            if e.method == "trim_matches" and len(e.args) == 1:
                return (
                    f"{recv}.trim_matches({self.expr_to_rust(e.args[0])}).to_string()"
                )
            if (
                e.method in ("trim_start_matches", "trim_end_matches")
                and len(e.args) == 1
            ):
                return f"{recv}.{e.method}({self.expr_to_rust(e.args[0])}).to_string()"
            if e.method == "is_ascii" and not e.args:
                return f"{recv}.is_ascii()"
            if e.method == "lines" and not e.args:
                return (
                    f"{recv}.lines().map(|line| line.to_string()).collect::<Vec<_>>()"
                )
            if e.method == "split_whitespace" and not e.args:
                return (
                    f"{recv}.split_whitespace()"
                    ".map(|part| part.to_string())"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "sort_unstable" and not e.args:
                return f"{recv}.sort_unstable()"
            if e.method == "as_bytes" and not e.args:
                return f"{recv}.as_bytes().to_vec()"
            if e.method == "is_char_boundary" and len(e.args) == 1:
                return f"{recv}.is_char_boundary({self._emit_usize_arg(e.args[0])})"
            if e.method == "is_alphabetic" and not e.args:
                return f"{recv}.is_alphabetic()"
            if e.method == "is_numeric" and not e.args:
                return f"{recv}.is_numeric()"
            if e.method == "is_alphanumeric" and not e.args:
                return f"{recv}.is_alphanumeric()"
            if e.method == "is_whitespace" and not e.args:
                return f"{recv}.is_whitespace()"
            if e.method == "is_uppercase" and not e.args:
                return f"{recv}.is_uppercase()"
            if e.method == "is_lowercase" and not e.args:
                return f"{recv}.is_lowercase()"
            if e.method == "to_digit" and len(e.args) == 1:
                radix = self.expr_to_rust(e.args[0])
                return f"{recv}.to_digit({radix} as u32)"
            if e.method == "is_ascii_alphabetic" and not e.args:
                return f"{recv}.is_ascii_alphabetic()"
            if e.method == "is_ascii_digit" and not e.args:
                return f"{recv}.is_ascii_digit()"
            if e.method == "is_ascii_alphanumeric" and not e.args:
                return f"{recv}.is_ascii_alphanumeric()"
            if e.method == "is_ascii_whitespace" and not e.args:
                return f"{recv}.is_ascii_whitespace()"
            if e.method == "is_ascii_uppercase" and not e.args:
                return f"{recv}.is_ascii_uppercase()"
            if e.method == "is_ascii_lowercase" and not e.args:
                return f"{recv}.is_ascii_lowercase()"
            if e.method == "is_ascii_punctuation" and not e.args:
                return f"{recv}.is_ascii_punctuation()"
            if e.method == "len_utf8" and not e.args:
                return f"{recv}.len_utf8()"
            if e.method == "len_utf16" and not e.args:
                return f"{recv}.len_utf16()"
            if e.method == "is_control" and not e.args:
                return f"{recv}.is_control()"
            if e.method == "is_ascii_hexdigit" and not e.args:
                return f"{recv}.is_ascii_hexdigit()"
            if e.method == "eq_ignore_ascii_case" and len(e.args) == 1:
                return f"{recv}.eq_ignore_ascii_case(&{self.expr_to_rust(e.args[0])})"
            if e.method == "is_ascii_control" and not e.args:
                return f"{recv}.is_ascii_control()"
            if e.method in ("to_ascii_lowercase", "to_ascii_uppercase") and not e.args:
                return f"{recv}.{e.method}()"
            if e.method == "replace" and len(e.args) == 2:
                old = self.expr_to_rust(e.args[0])
                new = self.expr_to_rust(e.args[1])
                return f"{recv}.replace(&{old}, &{new})"
            if e.method in ("starts_with", "ends_with") and len(e.args) == 1:
                arg = self.expr_to_rust(e.args[0])
                return f"{recv}.{e.method}(&{arg})"
            if (
                e.method == "map"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
            ):
                option_kind = self._option_result_type(e.receiver)
                if option_kind is not None:
                    return f"{recv}.map({self.expr_to_rust(e.args[0])})"
                return (
                    f"{recv}.into_iter().map("
                    f"{self.expr_to_rust(e.args[0])})"
                    ".collect::<Vec<_>>()"
                )
            if (
                e.method == "filter"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
                and self._is_vec_expr(e.receiver)
            ):
                return (
                    f"{recv}.into_iter().filter("
                    f"{self.expr_to_rust(e.args[0])})"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "fold" and len(e.args) == 2:
                return (
                    f"{recv}.into_iter().fold("
                    f"{self.expr_to_rust(e.args[0])}, "
                    f"{self.expr_to_rust(e.args[1])})"
                )
            if e.method == "partition" and len(e.args) == 1:
                return f"{recv}.partition({self.expr_to_rust(e.args[0])})"
            if e.method == "len" and not e.args:
                return f"{recv}.len()"
            if e.method == "is_sorted" and not e.args:
                return f"{recv}.is_sorted()"
            if e.method in ("first", "last") and not e.args:
                return f"{recv}.{e.method}().cloned()"
            if e.method == "get" and len(e.args) == 1:
                return f"{recv}.get({self._emit_usize_arg(e.args[0])}).cloned()"
            if e.method == "truncate" and len(e.args) == 1:
                return f"{recv}.truncate({self._emit_usize_arg(e.args[0])})"
            if e.method in ("rotate_left", "rotate_right") and len(e.args) == 1:
                return f"{recv}.{e.method}({self._emit_usize_arg(e.args[0])})"
            if e.method == "fill" and len(e.args) == 1:
                return f"{recv}.fill({self.expr_to_rust(e.args[0])})"
            if e.method == "swap" and len(e.args) == 2:
                return (
                    f"{recv}.swap({self._emit_usize_arg(e.args[0])}, "
                    f"{self._emit_usize_arg(e.args[1])})"
                )
            if e.method == "extend_from_slice" and len(e.args) == 1:
                return f"{recv}.extend_from_slice(&{self.expr_to_rust(e.args[0])})"
            if e.method == "resize" and len(e.args) == 2:
                return (
                    f"{recv}.resize({self._emit_usize_arg(e.args[0])}, "
                    f"{self.expr_to_rust(e.args[1])})"
                )
            if e.method == "split_off" and len(e.args) == 1:
                return f"{recv}.split_off({self._emit_usize_arg(e.args[0])})"
            if (
                e.method == "split_at"
                and len(e.args) == 1
                and self._is_string_expr(e.receiver)
            ):
                index = self._emit_usize_arg(e.args[0])
                return (
                    f"{{ let (left, right) = {recv}.split_at({index}); "
                    "(left.to_string(), right.to_string()) }"
                )
            if e.method == "split_at" and len(e.args) == 1:
                index = self._emit_usize_arg(e.args[0])
                return (
                    f"{{ let (left, right) = {recv}.split_at({index}); "
                    "(left.to_vec(), right.to_vec()) }"
                )
            if e.method == "find" and len(e.args) == 1:
                return f"{recv}.find(&{self.expr_to_rust(e.args[0])})"
            if e.method == "binary_search" and len(e.args) == 1:
                return f"{recv}.binary_search(&{self.expr_to_rust(e.args[0])})"
            if e.method == "rfind" and len(e.args) == 1:
                return f"{recv}.rfind(&{self.expr_to_rust(e.args[0])})"
            if e.method == "position" and len(e.args) == 1:
                return (
                    f"{recv}.clone().into_iter().position("
                    f"{self.expr_to_rust(e.args[0])})"
                    ".map(|index| index as i64)"
                )
            if e.method == "any" and len(e.args) == 1:
                return f"{recv}.clone().into_iter().any({self.expr_to_rust(e.args[0])})"
            if e.method == "all" and len(e.args) == 1:
                return f"{recv}.clone().into_iter().all({self.expr_to_rust(e.args[0])})"
            if e.method == "min" and not e.args:
                return f"{recv}.iter().min().cloned()"
            if e.method == "max" and not e.args:
                return f"{recv}.iter().max().cloned()"
            if e.method == "sum" and not e.args:
                return f"{recv}.iter().copied().sum::<i64>()"
            if e.method == "product" and not e.args:
                return f"{recv}.iter().copied().product::<i64>()"
            if e.method in ("strip_prefix", "strip_suffix") and len(e.args) == 1:
                return (
                    f"{recv}.{e.method}(&{self.expr_to_rust(e.args[0])})"
                    ".map(|part| part.to_string())"
                )
            if e.method in ("split_once", "rsplit_once") and len(e.args) == 1:
                return (
                    f"{recv}.{e.method}(&{self.expr_to_rust(e.args[0])})"
                    ".map(|(left, right)| "
                    "(left.to_string(), right.to_string()))"
                )
            if e.method == "is_empty" and not e.args:
                return f"{recv}.is_empty()"
            if e.method == "repeat" and len(e.args) == 1:
                return f"{recv}.repeat({self._emit_usize_arg(e.args[0])})"
            if e.method == "contains" and len(e.args) == 1:
                arg = self.expr_to_rust(e.args[0])
                return f"{recv}.contains(&{arg})"
            if e.method == "insert" and len(e.args) == 2:
                index = self._emit_usize_arg(e.args[0])
                value = self.expr_to_rust(e.args[1])
                return f"{recv}.insert({index}, {value})"
            if e.method == "remove" and len(e.args) == 1:
                index = self._emit_usize_arg(e.args[0])
                return f"{recv}.remove({index})"
            if e.method in ("windows", "chunks") and len(e.args) == 1:
                size = self._emit_usize_arg(e.args[0])
                return (
                    f"{recv}.{e.method}({size})"
                    ".map(|window| window.to_vec())"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "chunks_exact" and len(e.args) == 1:
                size = self._emit_usize_arg(e.args[0])
                return (
                    f"{recv}.chunks_exact({size})"
                    ".map(|chunk| chunk.to_vec())"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "rchunks" and len(e.args) == 1:
                size = self._emit_usize_arg(e.args[0])
                return (
                    f"{recv}.rchunks({size})"
                    ".map(|chunk| chunk.to_vec())"
                    ".collect::<Vec<_>>()"
                )
            if e.method == "rchunks_exact" and len(e.args) == 1:
                size = self._emit_usize_arg(e.args[0])
                return (
                    f"{recv}.rchunks_exact({size})"
                    ".map(|chunk| chunk.to_vec())"
                    ".collect::<Vec<_>>()"
                )
            if (
                e.method == "sort_by"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
                and len(e.args[0].params) == 2
                and isinstance(e.args[0].body, Binary)
                and e.args[0].body.op == "-"
            ):
                left_pat = e.args[0].params[0][0]
                right_pat = e.args[0].params[1][0]
                body = e.args[0].body
                if (
                    isinstance(left_pat, PatIdent)
                    and isinstance(right_pat, PatIdent)
                    and isinstance(body.left, Ident)
                    and isinstance(body.right, Ident)
                    and body.left.name == left_pat.name
                    and body.right.name == right_pat.name
                ):
                    return (
                        f"{recv}.sort_by(|{left_pat.name}, "
                        f"{right_pat.name}| {left_pat.name}.cmp("
                        f"{right_pat.name}))"
                    )
            if (
                e.method == "sort_by_key"
                and len(e.args) == 1
                and isinstance(e.args[0], Closure)
            ):
                return f"{recv}.sort_by_key({self.expr_to_rust(e.args[0])})"
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            return f"{self.expr_to_rust(e.receiver)}.{e.method}({args})"
        if isinstance(e, Field):
            if (
                self._self_field_aliases is not None
                and isinstance(e.receiver, Ident)
                and e.receiver.name == "self"
                and e.name in self._self_field_aliases
            ):
                return self._self_field_aliases[e.name]
            return f"{self.expr_to_rust(e.receiver)}.{e.name}"
        if isinstance(e, Index):
            recv = self.expr_to_rust(e.receiver)
            if isinstance(e.index, Range):
                start = (
                    self.expr_to_rust(e.index.start)
                    if e.index.start is not None
                    else ""
                )
                end = self.expr_to_rust(e.index.end) if e.index.end is not None else ""
                op = "..=" if e.index.inclusive else ".."
                if self._is_string_expr(e.receiver):
                    return f"{recv}[{start}{op}{end}].to_string()"
                return f"{recv}[{start}{op}{end}].to_vec()"
            self._check_type_boundary(
                self._expr_type(e.index),
                "usize",
                e.index,
            )
            idx = self.expr_to_rust(e.index)
            return f"{recv}[{idx}]"
        if isinstance(e, TupleExpr):
            elems = ", ".join(self.expr_to_rust(x) for x in e.elems)
            if len(e.elems) == 1:
                return f"({elems},)"
            return f"({elems})"
        if isinstance(e, ArrayExpr):
            elems = ", ".join(self.expr_to_rust(x) for x in e.elems)
            return f"vec![{elems}]"
        if isinstance(e, RepeatArray):
            return f"vec![{self.expr_to_rust(e.elem)}; {self.expr_to_rust(e.count)}]"
        if isinstance(e, StructLiteral):
            name = e.path.split("::")[-1]
            field_items = [f"{k}: {self.expr_to_rust(v)}" for k, v in e.fields]
            if e.rest is None and name in self.borrow_state_types:
                field_items.append(
                    "__polyglot_borrow_state: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0))"
                )
            fields = ", ".join(field_items)
            rest = ""
            if e.rest is not None:
                rest = f", ..{self.expr_to_rust(e.rest)}"
            return f"{name} {{ {fields}{rest} }}"
        if isinstance(e, IfExpr):
            then = self._block_as_expr(e.then_block)
            else_part = "()"
            if isinstance(e.else_branch, Block):
                else_part = self._block_as_expr(e.else_branch)
            elif e.else_branch is not None:
                else_part = self.expr_to_rust(e.else_branch)
            return (
                f"(if {self.expr_to_rust(e.cond)} {{ {then} }} else {{ {else_part} }})"
            )
        if isinstance(e, IfLet):
            then = self._block_as_expr(e.then_block)
            else_part = "()"
            if isinstance(e.else_branch, Block):
                else_part = self._block_as_expr(e.else_branch)
            elif e.else_branch is not None:
                else_part = self.expr_to_rust(e.else_branch)
            return (
                f"(if let {self.pattern_to_rust(e.pattern)} = "
                f"{self.expr_to_rust(e.value)} {{ {then} }} "
                f"else {{ {else_part} }})"
            )
        if isinstance(e, MatchExpr):
            warn_match_issues(
                e,
                scrutinee_type=self._match_scrutinee_type(e),
            )
            arms = []
            for arm in e.arms:
                pat = self.pattern_to_rust(arm.pattern)
                guard = ""
                if arm.guard is not None:
                    guard = f" if {self.expr_to_rust(arm.guard)}"
                if isinstance(arm.body, Block):
                    body = f"{{ {self._block_as_expr(arm.body)} }}"
                else:
                    body = self.expr_to_rust(arm.body)
                arms.append(f"{pat}{guard} => {body}")
            scrutinee = self.expr_to_rust(e.scrutinee)
            if any(isinstance(arm.pattern, PatSlice) for arm in e.arms):
                scrutinee = f"({scrutinee}).as_slice()"
            return f"(match {scrutinee} {{ {', '.join(arms)} }})"
        if isinstance(e, Loop):
            return "/* loop-expr not supported as value */"
        if isinstance(e, While):
            return "/* while-expr not supported as value */"
        if isinstance(e, For):
            return "/* for-expr not supported as value */"
        if isinstance(e, Closure):
            params = []
            for pat, ty in e.params:
                p = self.pattern_to_rust(pat)
                if ty is not None:
                    p = f"{p}: {self.type_to_rust(ty)}"
                params.append(p)
            body = (
                self._block_as_expr(e.body)
                if isinstance(e.body, Block)
                else self.expr_to_rust(e.body)
            )
            ret = ""
            if e.ret is not None:
                ret = f" -> {self.type_to_rust(e.ret)}"
            return f"|{', '.join(params)}|{ret} {{ {body} }}"
        if isinstance(e, Range):
            start = self.expr_to_rust(e.start) if e.start is not None else ""
            end = self.expr_to_rust(e.end) if e.end is not None else ""
            op = "..=" if e.inclusive else ".."
            return f"({start}{op}{end})"
        if isinstance(e, Cast):
            return self.emit_cast(e)
        if isinstance(e, Await):
            return f"({self.expr_to_rust(e.expr)}).await"
        if isinstance(e, Try):
            if isinstance(e.expr, MethodCall) and e.expr.method == "parse":
                recv = self.expr_to_rust(e.expr.receiver)
                target = (
                    self.type_to_rust(e.expr.generic_args[0])
                    if e.expr.generic_args
                    else "i64"
                )
                return f"{recv}.parse::<{target}>()?"
            return f"{self.expr_to_rust(e.expr)}?"
        if isinstance(e, Return):
            if e.value is None:
                return "return"
            value = self.expr_to_rust(e.value)
            value = self._adapt_host_result(e.value, value)
            return f"return {value}"
        if isinstance(e, Break):
            if e.value is not None:
                return f"break {self.expr_to_rust(e.value)}"
            return "break"
        if isinstance(e, Continue):
            return "continue"
        if isinstance(e, MacroCall):
            if isinstance(e.args, str):
                return f"{e.path}!({e.args})"
            if e.path in ("println", "print", "eprintln", "eprint", "format"):
                if not e.args:
                    return f"{e.path}!()"
                fmt_str = self._format_str(e.args[0])
                rest = ", ".join(self.expr_to_rust(a) for a in e.args[1:])
                if rest:
                    return f"{e.path}!({fmt_str}, {rest})"
                return f"{e.path}!({fmt_str})"
            if e.path == "write":
                if len(e.args) < 2:
                    return "write!().unwrap()"
                target = self.expr_to_rust(e.args[0])
                format_string = self._format_str(e.args[1])
                rest = ", ".join(self.expr_to_rust(arg) for arg in e.args[2:])
                args = f"{target}, {format_string}"
                if rest:
                    args += f", {rest}"
                return f"write!({args}).unwrap()"
            args = ", ".join(self.expr_to_rust(a) for a in e.args)
            if e.path == "vec":
                return f"vec![{args}]"
            return f"{e.path}!({args})"
        if isinstance(e, Block):
            return f"{{ {self._block_as_expr(e)} }}"
        return f"/* expr {type(e).__name__} */"

    def _is_string_expr(self, expr: Expr) -> bool:
        if isinstance(expr, LitStr):
            return True
        if isinstance(expr, Ident):
            return expr.name in self.string_bindings
        if isinstance(expr, MethodCall):
            return expr.method in {
                "replace",
                "trim",
                "to_uppercase",
                "to_lowercase",
                "parse",
            }
        if isinstance(expr, MacroCall):
            return expr.path == "format"
        return False

    def _is_vec_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Ident):
            return expr.name in self.vec_bindings
        if isinstance(expr, (ArrayExpr, RepeatArray)):
            return True
        if isinstance(expr, MacroCall):
            return expr.path == "vec"
        if isinstance(expr, Paren):
            return self._is_vec_expr(expr.inner)
        return False

    def _match_scrutinee_type(self, match: MatchExpr) -> str | None:
        if isinstance(match.scrutinee, Ident):
            return self.binding_types.get(match.scrutinee.name)
        return self._expr_variant_type(match.scrutinee)

    def _option_result_type(self, expr: Expr) -> str | None:
        if isinstance(expr, Ident):
            kind = self.binding_types.get(expr.name)
            return kind if kind in {"Option", "Result"} else None
        if isinstance(expr, Paren):
            return self._option_result_type(expr.inner)
        kind = self._expr_variant_type(expr)
        return kind if kind in {"Option", "Result"} else None

    def _expr_variant_type(self, expr: Expr) -> str | None:
        if isinstance(expr, (OptionSome, OptionNone)):
            return "Option"
        if isinstance(expr, (ResultOk, ResultErr)):
            return "Result"
        if isinstance(expr, Call) and isinstance(expr.func, Ident):
            return self._function_return_types.get(expr.func.name)
        if (
            isinstance(expr, Call)
            and isinstance(expr.func, Path)
            and len(expr.func.segments) == 1
        ):
            return self._function_return_types.get(expr.func.segments[0])
        if isinstance(expr, MethodCall) and expr.method in {"pop", "next"}:
            return "Option"
        if isinstance(expr, MethodCall) and expr.method in {
            "map",
            "map_err",
            "and_then",
            "or_else",
            "filter",
        }:
            return self._option_result_type(expr.receiver)
        if isinstance(expr, Path) and expr.segments[:1] in (
            ["HashMap"],
            ["BTreeMap"],
            ["HashSet"],
        ):
            return expr.segments[0]
        if isinstance(expr, Call) and isinstance(expr.func, Path):
            segments = expr.func.segments
            if (
                len(segments) == 2
                and segments[1] == "new"
                and segments[0] in {"HashMap", "BTreeMap", "HashSet"}
            ):
                return segments[0]
        return None

    def _binding_type_name(self, ty: Type | None) -> str | None:
        if isinstance(ty, TypeRef):
            return self._binding_type_name(ty.inner)
        if isinstance(ty, TypeName):
            return ty.name
        return None

    def _expr_struct_type(self, expr: Expr) -> str | None:
        if isinstance(expr, Ident):
            if expr.name == "self" and self._current_impl_type is not None:
                return self._current_impl_type
            return self.binding_types.get(expr.name)
        if isinstance(expr, Paren):
            return self._expr_struct_type(expr.inner)
        if isinstance(expr, Field):
            owner = self._expr_struct_type(expr.receiver)
            if owner is None:
                return None
            return self.struct_field_types.get(owner, {}).get(expr.name)
        return None

    def _is_user_method_call(self, e: MethodCall) -> bool:
        """True when ``e`` calls a method declared in an ``impl`` block."""
        receiver_type = self._expr_struct_type(e.receiver) or self._expr_type(
            e.receiver
        )
        return (
            receiver_type is not None
            and (receiver_type, e.method) in self._method_return_types
        )

    def _is_hashmap_expr(self, expr: Expr) -> bool:
        return self._map_kind(expr) == "HashMap"

    def _is_btreemap_expr(self, expr: Expr) -> bool:
        return self._map_kind(expr) == "BTreeMap"

    def _is_hashset_expr(self, expr: Expr) -> bool:
        return self._map_kind(expr) == "HashSet"

    def _map_kind(self, expr: Expr) -> str | None:
        """Return the collection kind of an expression, if known."""
        if isinstance(expr, Paren):
            return self._map_kind(expr.inner)
        if isinstance(expr, Ident):
            kind = self.binding_types.get(expr.name)
            return kind if kind in {"HashMap", "BTreeMap", "HashSet"} else None
        if isinstance(expr, Field):
            owner = self._expr_struct_type(expr.receiver)
            if owner is None:
                return None
            kind = self.struct_field_types.get(owner, {}).get(expr.name)
            return kind if kind in {"HashMap", "BTreeMap", "HashSet"} else None
        if isinstance(expr, Call) and isinstance(expr.func, Path):
            segments = expr.func.segments
            if (
                len(segments) == 2
                and segments[1] == "new"
                and segments[0] in {"HashMap", "BTreeMap", "HashSet"}
            ):
                return segments[0]
        if (
            isinstance(expr, MethodCall)
            and expr.method == "unwrap"
            and isinstance(expr.receiver, MethodCall)
            and expr.receiver.method == "get"
        ):
            # A map lookup returns the nested map value (depth 1).
            kind = self._map_kind(expr.receiver.receiver)
            return kind if kind in {"HashMap", "BTreeMap"} else None
        return None

    def _is_map_entry_expr(self, expr: Expr) -> bool:
        if isinstance(expr, Paren):
            return self._is_map_entry_expr(expr.inner)
        if isinstance(expr, MethodCall):
            if expr.method == "entry":
                return self._map_kind(expr.receiver) in {"HashMap", "BTreeMap"}
            if expr.method == "and_modify":
                return self._is_map_entry_expr(expr.receiver)
        return False

    def _ref_arg_to_rust(self, expr: Expr) -> str:
        if isinstance(expr, Unary) and expr.op in ("&", "&mut"):
            return f"&({self.expr_to_rust(expr.operand)})"
        return f"&({self.expr_to_rust(expr)})"

    def _owned_arg_to_rust(self, expr: Expr) -> str:
        if isinstance(expr, Unary) and expr.op in ("&", "&mut"):
            return self.expr_to_rust(expr.operand)
        return self.expr_to_rust(expr)

    def _block_as_expr(self, block: Block) -> str:
        parts: list[str] = []
        for stmt in block.stmts:
            if isinstance(stmt, LetStmt):
                mut = "mut "
                pat = self.pattern_to_rust(stmt.pattern)
                ty = f": {self.type_to_rust(stmt.ty)}" if stmt.ty else ""
                if stmt.value is not None:
                    parts.append(
                        f"let {mut}{pat}{ty} = {self.expr_to_rust(stmt.value)};"
                    )
                else:
                    parts.append(f"let {mut}{pat}{ty};")
            elif isinstance(stmt, ExprStmt):
                parts.append(f"{self.expr_to_rust(stmt.expr)};")
        if block.tail is not None:
            parts.append(self.expr_to_rust(block.tail))
        elif not parts:
            parts.append("()")
        return " ".join(parts)

    # ---- types & patterns -------------------------------------------------

    def type_to_rust(self, ty: Type | None) -> str:
        if ty is None:
            return "()"
        if isinstance(ty, TypeName):
            name = ty.name
            if name in self.generic_type_names:
                return "i64"
            if name == "str":
                name = "String"
            if name in self.generic_struct_names:
                return name
            if ty.args:
                args = ", ".join(self.type_to_rust(a) for a in ty.args)
                if name == "Vec":
                    return f"Vec<{args}>"
                if name == "Result" and len(ty.args) == 2:
                    # Language Result<T, String> is represented as
                    # Result<T, PyErr> in generated Rust; the exposed function
                    # boundary uses PyResult<T>.
                    return f"Result<{self.type_to_rust(ty.args[0])}, pyo3::PyErr>"
                return f"{name}<{args}>"
            return name
        if isinstance(ty, TypeRef):
            inner = ty.inner
            if isinstance(inner, TypeName) and inner.name == "str":
                mut = "mut " if ty.mutable else ""
                return f"&{mut}str"
            mut = "mut " if ty.mutable else ""
            return f"&{mut}{self.type_to_rust(inner)}"
        if isinstance(ty, TypeSlice):
            return f"[{self.type_to_rust(ty.inner)}]"
        if isinstance(ty, TypeArray):
            return f"[{self.type_to_rust(ty.inner)}; {self.expr_to_rust(ty.size)}]"
        if isinstance(ty, TypeTuple):
            elems = ", ".join(self.type_to_rust(t) for t in ty.elems)
            if len(ty.elems) == 1:
                elems += ","
            return f"({elems})"
        if isinstance(ty, TypeUnit):
            return "()"
        return "()"

    def _is_self_param(self, p: FnParam) -> bool:
        pattern = p.pattern
        if isinstance(pattern, PatIdent):
            return pattern.name == "self"
        if isinstance(pattern, PatRef):
            inner = pattern.inner
            return isinstance(inner, PatIdent) and inner.name == "self"
        return False

    def param_to_rust(self, p: FnParam, *, as_method: bool = False) -> str:
        pat = self.pattern_to_rust(p.pattern)

        if isinstance(p.pattern, PatIdent) and p.pattern.name == "self":
            return "self"
        if isinstance(p.pattern, PatRef):
            inner = p.pattern.inner
            if isinstance(inner, PatIdent) and inner.name == "self":
                return "&mut self" if p.pattern.mutable else "&self"

        if p.ty is None:
            return pat

        if not (isinstance(p.pattern, PatIdent) and p.pattern.mutable):
            pat = f"mut {pat}"
        return f"{pat}: {self.type_to_rust(p.ty)}"

    def pattern_to_rust(self, pat: Pattern) -> str:
        if isinstance(pat, PatIdent):
            mut = "mut " if pat.mutable else ""
            return f"{mut}{pat.name}"
        if isinstance(pat, PatWildcard):
            return "_"
        if isinstance(pat, PatRef):
            mut = "mut " if pat.mutable else ""
            return f"&{mut}{self.pattern_to_rust(pat.inner)}"
        if isinstance(pat, PatLiteral):
            val = pat.value
            if isinstance(val, LitBool):
                return "true" if val.value else "false"
            if isinstance(val, Unary) and val.op == "-":
                return f"-{self._pattern_bound_to_rust(val.operand)}"
            raw = getattr(val, "raw", None)
            if raw is not None:
                return raw
            value = getattr(val, "value", None)
            if value is not None:
                return repr(value)
            return str(val)
        if isinstance(pat, PatTuple):
            elems = ", ".join(self.pattern_to_rust(p) for p in pat.elems)
            if len(pat.elems) == 1:
                elems += ","
            return f"({elems})"
        if isinstance(pat, PatStruct):
            struct_short_name = pat.path.split("::")[-1]
            fields = ", ".join(
                f"{name}: {self.pattern_to_rust(field_pattern)}"
                for name, field_pattern in pat.fields
            )
            if pat.rest or struct_short_name in self.borrow_state_types:
                fields = f"{fields}, .." if fields else ".."
            return f"{pat.path} {{ {fields} }}"
        if isinstance(pat, PatEnum):
            args = ", ".join(self.pattern_to_rust(a) for a in pat.args)
            if args:
                return f"{pat.path}({args})"
            return pat.path
        if isinstance(pat, PatOr):
            return " | ".join(self.pattern_to_rust(alt) for alt in pat.alts)
        if isinstance(pat, PatRange):
            start = (
                self._pattern_bound_to_rust(pat.start) if pat.start is not None else ""
            )
            end = self._pattern_bound_to_rust(pat.end) if pat.end is not None else ""
            op = "..=" if pat.inclusive else ".."
            return f"{start}{op}{end}"
        if isinstance(pat, PatBinding):
            inner = self.pattern_to_rust(pat.inner)
            if isinstance(pat.inner, PatOr):
                inner = f"({inner})"
            return f"{pat.name} @ {inner}"
        if isinstance(pat, PatBox):
            raise UnsupportedNativePatternError(
                "Box patterns are supported only by the Python backend",
                pat.line,
                pat.col,
            )
        if isinstance(pat, PatConst):
            return pat.path if "::" in pat.path else f"self::{pat.path}"
        if isinstance(pat, PatPath):
            return pat.path
        if isinstance(pat, PatMacro):
            return f"{pat.path}!{pat.delimiter[0]}{pat.tokens}{pat.delimiter[1]}"
        if isinstance(pat, PatSlice):
            elems = ", ".join(
                self.pattern_to_rust(
                    elem.inner if isinstance(elem, PatRef) else elem,
                )
                for elem in pat.elems
            )
            return f"[{elems}]"
        if isinstance(pat, PatRest):
            return ".."
        return "_"

    def _pattern_bound_to_rust(self, bound: Expr) -> str:
        if isinstance(bound, Unary) and bound.op == "-":
            return f"-{self._pattern_bound_to_rust(bound.operand)}"
        raw = getattr(bound, "raw", None)
        if raw is not None:
            return raw
        if isinstance(bound, LitChar):
            return f"'{bound.value}'"
        if isinstance(bound, LitBool):
            return "true" if bound.value else "false"
        return self.expr_to_rust(bound)
