//! Cooperative epoch-based reclamation for immutable AST roots.
//!
//! This is user-space RCU, not a kernel RCU syscall.  Readers publish the
//! epoch they entered in a per-thread slot; a writer publishes a new root,
//! retires the old one, flips the global epoch, and waits until every reader
//! that entered before the flip has left its guard.  Reclamation callbacks run
//! after that grace period.
//!
//! The protocol is intentionally small enough to audit:
//!
//! 1. `enter()` records the current epoch in a thread-local slot and increments
//!    a nesting depth;
//! 2. `retire()` attaches a `FnOnce` reclaimer to the current epoch;
//! 3. `synchronize()` advances the epoch and blocks until no active reader is
//!    still at or below the previous epoch, then drains eligible reclaimers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

struct ReaderSlot {
    depth: AtomicUsize,
    epoch: AtomicUsize,
}

impl ReaderSlot {
    fn new(epoch: usize) -> Self {
        Self {
            depth: AtomicUsize::new(0),
            epoch: AtomicUsize::new(epoch),
        }
    }
}

struct Retired {
    epoch: usize,
    reclaim: Box<dyn FnOnce() + Send + 'static>,
}

struct Readers {
    slots: HashMap<ThreadId, Arc<ReaderSlot>>,
}

/// A snapshot of one domain's counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct RcuStats {
    pub epoch: usize,
    pub active_readers: usize,
    pub retired: usize,
    pub reclaimed: usize,
    pub grace_periods: usize,
    pub reclaim_failures: usize,
}

/// A cooperative RCU domain.
pub struct RcuDomain {
    pub name: String,
    epoch: AtomicUsize,
    readers: Mutex<Readers>,
    condvar: Condvar,
    retired: Mutex<Vec<Retired>>,
    writer: Mutex<()>,
    reclaimed: AtomicUsize,
    grace_periods: AtomicUsize,
    reclaim_failures: AtomicUsize,
}

impl RcuDomain {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            epoch: AtomicUsize::new(0),
            readers: Mutex::new(Readers {
                slots: HashMap::new(),
            }),
            condvar: Condvar::new(),
            retired: Mutex::new(Vec::new()),
            writer: Mutex::new(()),
            reclaimed: AtomicUsize::new(0),
            grace_periods: AtomicUsize::new(0),
            reclaim_failures: AtomicUsize::new(0),
        }
    }

    fn lock_readers(&self) -> std::sync::MutexGuard<'_, Readers> {
        self.readers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn lock_retired(&self) -> std::sync::MutexGuard<'_, Vec<Retired>> {
        self.retired
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn lock_writer(&self) -> std::sync::MutexGuard<'_, ()> {
        self.writer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn slot_for(&self, thread: ThreadId) -> Arc<ReaderSlot> {
        let mut readers = self.lock_readers();
        readers
            .slots
            .entry(thread)
            .or_insert_with(|| Arc::new(ReaderSlot::new(self.epoch.load(Ordering::Acquire))))
            .clone()
    }

    /// Enter a read-side critical section on the current thread.
    pub fn enter(&self) -> ReadGuard<'_> {
        let slot = self.slot_for(std::thread::current().id());
        let previous = slot.depth.fetch_add(1, Ordering::AcqRel);
        if previous == 0 {
            slot.epoch
                .store(self.epoch.load(Ordering::Acquire), Ordering::Release);
        }
        ReadGuard { domain: self, slot }
    }

    fn exit(&self, slot: &Arc<ReaderSlot>) {
        if slot.depth.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut readers = self.lock_readers();
            if slot.depth.load(Ordering::Acquire) == 0 {
                let thread = std::thread::current().id();
                if readers
                    .slots
                    .get(&thread)
                    .map_or(false, |current| Arc::ptr_eq(current, slot))
                {
                    readers.slots.remove(&thread);
                }
                self.condvar.notify_all();
            }
        }
    }

    /// Enter without a guard (used by the PyO3 bridge, which pairs calls).
    pub fn enter_raw(&self) -> usize {
        let slot = self.slot_for(std::thread::current().id());
        let previous = slot.depth.fetch_add(1, Ordering::AcqRel);
        if previous == 0 {
            let epoch = self.epoch.load(Ordering::Acquire);
            slot.epoch.store(epoch, Ordering::Release);
            return epoch;
        }
        slot.epoch.load(Ordering::Acquire)
    }

    /// Exit a read-side critical section entered with [`enter_raw`].
    pub fn exit_raw(&self) -> Result<(), String> {
        let thread = std::thread::current().id();
        let slot = {
            let readers = self.lock_readers();
            readers.slots.get(&thread).cloned()
        };
        let Some(slot) = slot else {
            return Err("RCU reader exited without a matching enter".to_string());
        };
        self.exit(&slot);
        Ok(())
    }

    pub fn active_readers(&self) -> usize {
        self.lock_readers()
            .slots
            .values()
            .filter(|slot| slot.depth.load(Ordering::Acquire) > 0)
            .count()
    }

    /// Retire a value with an explicit reclamation callback.
    pub fn retire<F>(&self, reclaim: F) -> usize
    where
        F: FnOnce() + Send + 'static,
    {
        let epoch = self.epoch.load(Ordering::Acquire);
        self.lock_retired().push(Retired {
            epoch,
            reclaim: Box::new(reclaim),
        });
        epoch
    }

    /// Wait for a grace period and run every eligible reclaimer.
    pub fn synchronize(&self, timeout: Option<Duration>) -> Result<usize, String> {
        let _writer = self.lock_writer();
        let target = self.epoch.load(Ordering::Acquire);
        self.epoch.store(target + 1, Ordering::Release);
        self.grace_periods.fetch_add(1, Ordering::Relaxed);
        let deadline = timeout.map(|limit| Instant::now() + limit);
        let mut readers = self.lock_readers();
        while readers.slots.values().any(|slot| {
            slot.depth.load(Ordering::Acquire) > 0 && slot.epoch.load(Ordering::Acquire) <= target
        }) {
            match deadline {
                None => {
                    readers = self
                        .condvar
                        .wait(readers)
                        .unwrap_or_else(|error| error.into_inner());
                }
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        let stuck = readers
                            .slots
                            .values()
                            .filter(|slot| {
                                slot.depth.load(Ordering::Acquire) > 0
                                    && slot.epoch.load(Ordering::Acquire) <= target
                            })
                            .count();
                        drop(readers);
                        return Err(format!(
                            "RCU grace period timed out with {stuck} reader(s) \
                             still active at or before epoch {target}"
                        ));
                    }
                    let (guard, _) = self
                        .condvar
                        .wait_timeout(readers, deadline - now)
                        .unwrap_or_else(|error| error.into_inner());
                    readers = guard;
                }
            }
        }
        drop(readers);
        let pending: Vec<Retired> = {
            let mut retired = self.lock_retired();
            let mut pending = Vec::new();
            let mut remaining = Vec::with_capacity(retired.len());
            for entry in std::mem::take(&mut *retired) {
                if entry.epoch <= target {
                    pending.push(entry);
                } else {
                    remaining.push(entry);
                }
            }
            *retired = remaining;
            pending
        };
        let mut reclaimed = 0usize;
        for entry in pending {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(entry.reclaim));
            match result {
                Ok(()) => reclaimed += 1,
                Err(_) => {
                    self.reclaim_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.reclaimed.fetch_add(reclaimed, Ordering::Relaxed);
        Ok(reclaimed)
    }

    pub fn stats(&self) -> RcuStats {
        RcuStats {
            epoch: self.epoch.load(Ordering::Acquire),
            active_readers: self.active_readers(),
            retired: self.lock_retired().len(),
            reclaimed: self.reclaimed.load(Ordering::Relaxed),
            grace_periods: self.grace_periods.load(Ordering::Relaxed),
            reclaim_failures: self.reclaim_failures.load(Ordering::Relaxed),
        }
    }

    pub fn drain(&self, timeout: Option<Duration>) -> Result<usize, String> {
        let mut total = 0;
        loop {
            if self.lock_retired().is_empty() {
                return Ok(total);
            }
            total += self.synchronize(timeout)?;
        }
    }
}

/// A read-side critical section; exiting it may release a grace period.
pub struct ReadGuard<'a> {
    domain: &'a RcuDomain,
    slot: Arc<ReaderSlot>,
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.domain.exit(&self.slot);
    }
}

/// An RCU-protected `Arc` root with monotonic versions.
pub struct EpochRoot<T> {
    root: Mutex<Arc<T>>,
    version: AtomicUsize,
    domain: Arc<RcuDomain>,
}

impl<T: Send + Sync + 'static> EpochRoot<T> {
    pub fn new(value: T) -> Self {
        Self {
            root: Mutex::new(Arc::new(value)),
            version: AtomicUsize::new(1),
            domain: Arc::new(RcuDomain::new("ast-root")),
        }
    }

    pub fn version(&self) -> usize {
        self.version.load(Ordering::Acquire)
    }

    pub fn snapshot(&self) -> (usize, Arc<T>) {
        let root = self.root.lock().unwrap_or_else(|error| error.into_inner());
        (self.version.load(Ordering::Acquire), Arc::clone(&root))
    }

    pub fn read(&self) -> RootRead<'_, T> {
        let guard = self.domain.enter();
        let root = self.root.lock().unwrap_or_else(|error| error.into_inner());
        RootRead {
            _guard: guard,
            version: self.version.load(Ordering::Acquire),
            value: Arc::clone(&root),
        }
    }

    pub fn publish(&self, value: T) -> usize {
        let old = {
            let mut root = self.root.lock().unwrap_or_else(|error| error.into_inner());
            let previous = std::mem::replace(&mut *root, Arc::new(value));
            self.version.fetch_add(1, Ordering::AcqRel);
            previous
        };
        self.domain.retire(move || drop(old));
        self.version.load(Ordering::Acquire)
    }

    pub fn domain(&self) -> &Arc<RcuDomain> {
        &self.domain
    }

    pub fn synchronize(&self, timeout: Option<Duration>) -> Result<usize, String> {
        self.domain.synchronize(timeout)
    }
}

/// A root read that keeps the grace period open until dropped.
pub struct RootRead<'a, T> {
    _guard: ReadGuard<'a>,
    pub version: usize,
    pub value: Arc<T>,
}

impl<T> std::ops::Deref for RootRead<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn guard_blocks_reclamation_until_exit() {
        let domain = Arc::new(RcuDomain::new("test"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let reader = {
            let domain = Arc::clone(&domain);
            thread::spawn(move || {
                let _guard = domain.enter();
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        };
        entered_rx.recv().unwrap();

        let reclaimed = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&reclaimed);
        domain.retire(move || flag.store(true, Ordering::SeqCst));

        let syncer = {
            let domain = Arc::clone(&domain);
            thread::spawn(move || domain.synchronize(Some(Duration::from_secs(2))))
        };
        thread::sleep(Duration::from_millis(50));
        assert!(!reclaimed.load(Ordering::SeqCst));
        release_tx.send(()).unwrap();
        reader.join().unwrap();
        assert_eq!(syncer.join().unwrap().unwrap(), 1);
        assert!(reclaimed.load(Ordering::SeqCst));
    }

    #[test]
    fn epoch_root_publishes_and_reclaims() {
        let root = EpochRoot::new(String::from("one"));
        assert_eq!(root.version(), 1);
        {
            let read = root.read();
            assert_eq!(&*read.value, "one");
            assert_eq!(root.publish(String::from("two")), 2);
            // The reader still sees the snapshot it captured.
            assert_eq!(&*read.value, "one");
        }
        assert_eq!(root.synchronize(Some(Duration::from_secs(2))).unwrap(), 1);
        assert_eq!(&*root.snapshot().1, "two");
        assert_eq!(root.domain().stats().reclaimed, 1);
    }

    #[test]
    fn unbalanced_exit_is_reported() {
        let domain = RcuDomain::new("test");
        assert!(domain.exit_raw().is_err());
        domain.enter_raw();
        assert!(domain.exit_raw().is_ok());
    }
}
