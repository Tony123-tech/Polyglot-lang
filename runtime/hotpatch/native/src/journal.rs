//! Sub-millisecond before-image journaling for hot-patch rollback.
//!
//! A checkpoint copies the *bytes that are about to change* into a compact
//! before-image buffer.  Rollback replays the images in reverse order, so
//! nested patches unwind correctly.  The journal never allocates during
//! rollback and never calls back into Python, which keeps the failure path
//! panic-free with respect to interpreter state.
//!
//! The Python side journals object attributes and buffers; this native
//! implementation journals raw ranges.  That is the correct seam for GOT
//! cells, JIT pages, and Rust-owned state, where an object graph snapshot
//! would not describe the modified memory.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A single captured range.
#[derive(Clone, Debug)]
pub struct BeforeImage {
    pub address: usize,
    pub before: Vec<u8>,
    pub label: String,
}

/// A thread-safe before-image log.
pub struct Journal {
    entries: Mutex<Vec<BeforeImage>>,
    commits: AtomicUsize,
    rollbacks: AtomicUsize,
    captured_bytes: AtomicUsize,
}

impl Default for Journal {
    fn default() -> Self {
        Self::new()
    }
}

impl Journal {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            commits: AtomicUsize::new(0),
            rollbacks: AtomicUsize::new(0),
            captured_bytes: AtomicUsize::new(0),
        }
    }

    fn lock_entries(&self) -> std::sync::MutexGuard<'_, Vec<BeforeImage>> {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Capture `[address, address + size)` before a patch mutates it.
    ///
    /// # Safety
    /// The caller must ensure the range is mapped and readable for the whole
    /// lifetime of the checkpoint.  Capturing an unmapped range is undefined
    /// behaviour; the PyO3 wrapper only accepts ranges the caller registered.
    pub unsafe fn capture(
        &self,
        address: usize,
        size: usize,
        label: impl Into<String>,
    ) -> Result<usize, String> {
        if address == 0 {
            return Err("cannot journal a NULL range".to_string());
        }
        if size == 0 {
            return Err("cannot journal an empty range".to_string());
        }
        let end = address
            .checked_add(size)
            .ok_or_else(|| "journal range overflows the address space".to_string())?;
        let mut before = vec![0u8; size];
        std::ptr::copy_nonoverlapping(address as *const u8, before.as_mut_ptr(), size);
        let mut entries = self.lock_entries();
        entries.push(BeforeImage {
            address,
            before,
            label: label.into(),
        });
        self.captured_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(end)
    }

    /// Restore every captured range in reverse order and empty the log.
    pub fn rollback(&self) -> usize {
        let entries = {
            let mut entries = self.lock_entries();
            std::mem::take(&mut *entries)
        };
        let restored = entries.len();
        for entry in entries.iter().rev() {
            // SAFETY: `capture` documented that the range stays mapped; the
            // before-image has exactly the captured length.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    entry.before.as_ptr(),
                    entry.address as *mut u8,
                    entry.before.len(),
                );
            }
        }
        if restored > 0 {
            self.rollbacks.fetch_add(1, Ordering::Relaxed);
        }
        restored
    }

    /// Accept the patch: discard the before-images without touching memory.
    pub fn commit(&self) -> usize {
        let discarded = {
            let mut entries = self.lock_entries();
            let count = entries.len();
            entries.clear();
            count
        };
        self.commits.fetch_add(1, Ordering::Relaxed);
        discarded
    }

    /// True when every captured range still matches its before-image.
    pub fn verify(&self) -> bool {
        let entries = self.lock_entries();
        entries.iter().all(|entry| {
            // SAFETY: see `capture`; the caller keeps the range alive.
            let current = unsafe {
                std::slice::from_raw_parts(entry.address as *const u8, entry.before.len())
            };
            current == entry.before.as_slice()
        })
    }

    /// FNV-1a over the labels and before-images, for diagnostics.
    pub fn digest(&self) -> u64 {
        let entries = self.lock_entries();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for entry in entries.iter() {
            for byte in entry.label.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
            for byte in &entry.before {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        hash
    }

    pub fn len(&self) -> usize {
        self.lock_entries().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn captured_bytes(&self) -> usize {
        self.captured_bytes.load(Ordering::Relaxed)
    }

    pub fn commits(&self) -> usize {
        self.commits.load(Ordering::Relaxed)
    }

    pub fn rollbacks(&self) -> usize {
        self.rollbacks.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollback_restores_external_mutations() {
        let mut buffer = vec![1u8, 2, 3, 4];
        let address = buffer.as_mut_ptr() as usize;
        let journal = Journal::new();
        // SAFETY: `buffer` lives until rollback and the range is exact.
        unsafe {
            journal.capture(address, buffer.len(), "vec").unwrap();
        }
        buffer[0] = 9;
        buffer[3] = 8;
        assert!(!journal.verify());
        assert_eq!(journal.rollback(), 1);
        assert_eq!(buffer, vec![1, 2, 3, 4]);
        assert!(journal.verify());
        assert_eq!(journal.rollbacks(), 1);
    }

    #[test]
    fn commit_discards_before_images() {
        let mut buffer = vec![5u8; 8];
        let journal = Journal::new();
        // SAFETY: as above.
        unsafe {
            journal
                .capture(buffer.as_mut_ptr() as usize, 8, "vec")
                .unwrap();
        }
        buffer[0] = 6;
        assert_eq!(journal.commit(), 1);
        assert!(journal.is_empty());
        assert_eq!(journal.rollback(), 0);
        assert_eq!(buffer[0], 6);
        assert_eq!(journal.commits(), 1);
    }

    #[test]
    fn reverse_order_unwinds_nested_checkpoints() {
        let mut buffer = vec![0u8; 4];
        let address = buffer.as_mut_ptr() as usize;
        let journal = Journal::new();
        // SAFETY: the vec outlives both captures and the rollback.
        unsafe {
            journal.capture(address, 4, "outer").unwrap();
            buffer[0] = 1;
            journal.capture(address, 4, "inner").unwrap();
            buffer[0] = 2;
        }
        assert_eq!(journal.rollback(), 2);
        // The inner image restored 1, then the outer image restored 0.
        assert_eq!(buffer[0], 0);
    }

    #[test]
    fn rejects_null_and_empty_ranges() {
        let journal = Journal::new();
        // SAFETY: rejected before any read.
        assert!(unsafe { journal.capture(0, 4, "null") }.is_err());
        assert!(unsafe { journal.capture(4096, 0, "empty") }.is_err());
    }
}
