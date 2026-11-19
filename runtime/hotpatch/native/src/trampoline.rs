//! Atomic GOT/PLT pointer swaps and owned-page jump sleds.
//!
//! The native module implements the same two relocation paths as the Python
//! package, but without ctypes:
//!
//! * [`patch_pointer`] rewrites an aligned pointer cell with one atomic word
//!   store after temporarily widening the page protection;
//! * [`ExecutableRegion`] owns a page range (``MAP_JIT`` on Apple Silicon,
//!   RW -> RX + ``mprotect`` elsewhere) and accepts inline jump sleds.
//!
//! Foreign executable text is never rewritten: only memory this process owns
//! through [`ExecutableRegion`] receives machine code.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use libc::{c_uint, size_t};

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const PROT_EXEC: c_int = 0x4;
const VM_PROT_READ: c_int = 0x1;
const VM_PROT_WRITE: c_int = 0x2;
const VM_PROT_COPY: c_int = 0x10;
const VM_REGION_BASIC_INFO_64: c_int = 9;
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
const LC_SEGMENT_64: u32 = 0x19;

#[cfg(target_os = "macos")]
const MAP_ANON: c_int = libc::MAP_ANON;
#[cfg(not(target_os = "macos"))]
const MAP_ANON: c_int = libc::MAP_ANONYMOUS;

#[cfg(target_os = "macos")]
extern "C" {
    static mach_task_self_: c_uint;
    fn mach_vm_protect(
        task: c_uint,
        address: u64,
        size: u64,
        set_maximum: c_int,
        new_protection: c_int,
    ) -> c_int;
    fn mach_vm_region(
        task: c_uint,
        address: *mut u64,
        size: *mut u64,
        flavor: c_int,
        info: *mut c_int,
        info_count: *mut c_uint,
        object_name: *mut c_uint,
    ) -> c_int;
    fn pthread_jit_write_protect_np(enabled: c_int);
    fn pthread_jit_write_protect_supported_np() -> c_int;
    fn sys_icache_invalidate(start: *mut c_void, length: size_t);

    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_header(index: u32) -> *const c_void;
    fn _dyld_get_image_vmaddr_slide(index: u32) -> isize;
    fn _dyld_get_image_name(index: u32) -> *const c_char;
}

#[cfg(not(target_os = "macos"))]
extern "C" {
    fn __clear_cache(start: *mut c_char, end: *mut c_char);
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VmRegionBasicInfo64 {
    protection: c_int,
    max_protection: c_int,
    inheritance: c_int,
    shared: c_int,
    reserved: c_int,
    offset: u64,
    behavior: c_int,
    user_wired_count: u16,
}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 {
        4096
    } else {
        value as usize
    }
}

fn errno() -> i32 {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: reading the thread-local errno value.
        unsafe { *libc::__error() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // SAFETY: reading the thread-local errno value.
        unsafe { *libc::__errno_location() }
    }
}

/// Best-effort query of the protection bits for the page containing `address`.
pub fn page_protection(address: usize) -> Option<c_int> {
    let start = address & !(page_size() - 1);
    #[cfg(target_os = "macos")]
    {
        let mut region_address = start as u64;
        let mut region_size = 0u64;
        let mut info = VmRegionBasicInfo64::default();
        let mut count =
            (std::mem::size_of::<VmRegionBasicInfo64>() / std::mem::size_of::<c_int>()) as c_uint;
        let mut object_name = 0u32;
        // SAFETY: every out-parameter points at a live, correctly sized local.
        let result = unsafe {
            mach_vm_region(
                mach_task_self_,
                &mut region_address,
                &mut region_size,
                VM_REGION_BASIC_INFO_64,
                &mut info as *mut VmRegionBasicInfo64 as *mut c_int,
                &mut count,
                &mut object_name,
            )
        };
        if result == 0 {
            Some(info.protection)
        } else {
            None
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for (mapped_start, mapped_end, _path, writable) in read_proc_maps() {
            if mapped_start <= start && start < mapped_end {
                return Some(PROT_READ | if writable { PROT_WRITE } else { 0 });
            }
        }
        None
    }
}

/// A pending protection restoration produced by [`make_writable`].
pub struct ProtectionGuard {
    start: usize,
    span: usize,
    protection: c_int,
    mach: bool,
}

impl ProtectionGuard {
    pub fn restore(self) {
        if self.protection == 0 {
            return;
        }
        if self.mach {
            // SAFETY: the range was previously protected with these bits.
            unsafe {
                mach_vm_protect(
                    mach_task_self_,
                    self.start as u64,
                    self.span as u64,
                    0,
                    self.protection,
                );
            }
        } else {
            // SAFETY: `mprotect` accepts page-aligned addresses; the original
            // bits were observed before widening the range.
            unsafe {
                libc::mprotect(self.start as *mut c_void, self.span, self.protection);
            }
        }
    }
}

/// Temporarily make the page(s) covering `[address, address + size)` writable.
pub fn make_writable(address: usize, size: usize) -> Result<ProtectionGuard, String> {
    let page = page_size();
    let start = address & !(page - 1);
    let span = size + (address - start);
    let original = page_protection(start).unwrap_or(PROT_READ);
    // SAFETY: the caller established that the range is mapped; `mprotect`
    // widens the permission on existing mappings only.
    let result = unsafe { libc::mprotect(start as *mut c_void, span, PROT_READ | PROT_WRITE) };
    if result == 0 {
        return Ok(ProtectionGuard {
            start,
            span,
            protection: original,
            mach: false,
        });
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: same range, COW-private remap for `__DATA_CONST` pages.
        let result = unsafe {
            mach_vm_protect(
                mach_task_self_,
                start as u64,
                span as u64,
                0,
                VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY,
            )
        };
        if result == 0 {
            return Ok(ProtectionGuard {
                start,
                span,
                protection: original,
                mach: true,
            });
        }
    }
    Err(format!(
        "cannot make page {start:#x} writable (errno {})",
        errno()
    ))
}

/// Load an aligned pointer cell.
///
/// # Safety
/// `slot` must point at a mapped, aligned machine word owned by the process.
pub unsafe fn load_pointer(slot: usize) -> u64 {
    AtomicUsize::from_ptr(slot as *mut usize).load(Ordering::SeqCst) as u64
}

/// Validate an FFI-supplied pointer cell before any raw dereference.
///
/// PyO3 callers hand us arbitrary integers.  Dereferencing an unmapped address
/// would segfault the interpreter, so confirm the page is live and the address
/// is word-aligned first.  This narrows (but cannot eliminate) the TOCTOU
/// window inherent to raw-address APIs; embedders still own the lifetime.
fn validate_pointer_cell(slot: usize) -> Result<(), String> {
    if slot == 0 {
        return Err("cannot access a NULL pointer cell".to_string());
    }
    if !slot.is_multiple_of(std::mem::align_of::<usize>()) {
        return Err(format!("pointer cell {slot:#x} is not word-aligned"));
    }
    if page_protection(slot).is_none() {
        return Err(format!(
            "pointer cell {slot:#x} is not in a mapped page"
        ));
    }
    Ok(())
}

/// Safe entry point used by the PyO3 bridge to read a pointer cell.
pub fn read_pointer(slot: usize) -> Result<u64, String> {
    validate_pointer_cell(slot)?;
    // SAFETY: validate_pointer_cell confirmed the cell is mapped and aligned.
    Ok(unsafe { load_pointer(slot) })
}

/// Validate every page covering `[address, address + size)` before a raw copy.
///
/// Callers still own the range lifetime; this only refuses obviously invalid
/// FFI addresses (NULL, overflow, unmapped pages) instead of segfaulting.
pub fn validate_mapped_range(address: usize, size: usize) -> Result<(), String> {
    if address == 0 {
        return Err("cannot access a NULL memory range".to_string());
    }
    if size == 0 {
        return Err("cannot access an empty memory range".to_string());
    }
    let end = address
        .checked_add(size)
        .ok_or_else(|| "memory range overflows the address space".to_string())?;
    let page = page_size();
    let mut cursor = address & !(page - 1);
    loop {
        if page_protection(cursor).is_none() {
            return Err(format!(
                "range {address:#x}..{end:#x} includes an unmapped page at {cursor:#x}"
            ));
        }
        if cursor >= end - 1 {
            return Ok(());
        }
        cursor += page;
    }
}

/// Atomically store a pointer and return the previous value.
///
/// The store is a single aligned machine-word write; a thread executing the
/// previous target continues there, while the next indirect call observes the
/// new target.
pub fn patch_pointer(slot: usize, value: u64) -> Result<u64, String> {
    validate_pointer_cell(slot)?;
    // SAFETY: validate_pointer_cell confirmed the cell is mapped and aligned;
    // reading before widening the page is safe on read-only data.
    let old = unsafe { load_pointer(slot) };
    let guard = make_writable(slot, std::mem::size_of::<usize>())?;
    // SAFETY: the guard made the containing page writable.
    unsafe {
        AtomicUsize::from_ptr(slot as *mut usize).store(value as usize, Ordering::SeqCst);
    }
    guard.restore();
    Ok(old)
}

pub fn resolve_symbol(name: &str) -> Option<u64> {
    let c_name = CString::new(name).ok()?;
    // SAFETY: `c_name` is a valid NUL-terminated string.
    let address = unsafe { libc::dlsym(RTLD_DEFAULT, c_name.as_ptr()) };
    if address.is_null() {
        None
    } else {
        Some(address as u64)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn fixed_name(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// A bound pointer cell discovered in a Mach-O GOT-like section.
#[derive(Clone, Debug)]
pub struct GotSlotInfo {
    pub address: u64,
    pub symbol: String,
    pub image: String,
    pub segment: String,
    pub section: String,
}

#[cfg(target_os = "macos")]
type MachSection = (String, String, u64, u64);
#[cfg(target_os = "macos")]
type MachImage = (String, usize, isize, Vec<MachSection>);

#[cfg(target_os = "macos")]
fn mach_o_images() -> Vec<MachImage> {
    let mut images = Vec::new();
    // SAFETY: dyld's image list is process-wide and stable while the GIL is
    // held; every header is validated against the Mach-O 64-bit magic.
    unsafe {
        let count = _dyld_image_count();
        for index in 0..count {
            let header = _dyld_get_image_header(index);
            if header.is_null() {
                continue;
            }
            let header_bytes = std::slice::from_raw_parts(header as *const u8, 32);
            if read_u32(header_bytes, 0) != 0xFEED_FACF {
                continue;
            }
            let ncmds = read_u32(header_bytes, 16) as usize;
            let mut offset = 32usize;
            let mut sections = Vec::new();
            // Cap the walk so a corrupt load-command chain cannot run off the
            // mapped Mach-O header image.
            for _ in 0..ncmds.min(1024) {
                if offset + 8 > 4096 {
                    break;
                }
                let command_bytes = std::slice::from_raw_parts(header.cast::<u8>().add(offset), 8);
                let command = read_u32(command_bytes, 0);
                let command_size = read_u32(command_bytes, 4) as usize;
                if command == LC_SEGMENT_64 {
                    if command_size < 72 || offset + command_size > 4096 {
                        break;
                    }
                    let segment = std::slice::from_raw_parts(header.cast::<u8>().add(offset), 72);
                    let nsects = read_u32(segment, 64) as usize;
                    let mut section_offset = offset + 72;
                    for _ in 0..nsects.min(512) {
                        if section_offset + 80 > offset + command_size {
                            break;
                        }
                        let section =
                            std::slice::from_raw_parts(header.cast::<u8>().add(section_offset), 80);
                        let sectname = fixed_name(&section[0..16]);
                        let segname = fixed_name(&section[16..32]);
                        let address = read_u64(section, 32);
                        let size = read_u64(section, 40);
                        sections.push((segname, sectname, address, size));
                        section_offset += 80;
                    }
                }
                if command_size == 0 {
                    break;
                }
                offset += command_size;
            }
            let name = _dyld_get_image_name(index);
            let path = if name.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(name)
                    .to_string_lossy()
                    .into_owned()
            };
            images.push((
                path,
                header as usize,
                _dyld_get_image_vmaddr_slide(index),
                sections,
            ));
        }
    }
    images
}

#[cfg(target_os = "macos")]
fn mach_o_got_slots(symbol: &str, target: u64) -> Vec<GotSlotInfo> {
    const GOT_SECTIONS: [&str; 4] = ["__got", "__la_symbol_ptr", "__auth_got", "__symbol_ptr"];
    let mut slots = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (path, _header, slide, sections) in mach_o_images() {
        for (segment, section, address, size) in sections {
            if !GOT_SECTIONS.contains(&section.as_str()) {
                continue;
            }
            let base = (slide as i64 + address as i64) as usize;
            let count = (size as usize / 8).min(1_000_000);
            for index in 0..count {
                let slot = base + index * 8;
                // SAFETY: the section is mapped by dyld; the scan stays inside
                // its declared `size`.
                let value = unsafe { ptr::read_volatile(slot as *const usize) } as u64;
                if value == target && seen.insert(slot) {
                    slots.push(GotSlotInfo {
                        address: slot as u64,
                        symbol: symbol.to_string(),
                        image: path.clone(),
                        segment: segment.clone(),
                        section: section.clone(),
                    });
                }
            }
        }
    }
    slots
}

#[cfg(not(target_os = "macos"))]
fn proc_maps() -> Vec<(usize, usize, String, bool)> {
    let mut entries = Vec::new();
    if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
        for line in maps.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 2 {
                continue;
            }
            let mut range = fields[0].split('-');
            let (Some(start), Some(end)) = (range.next(), range.next()) else {
                continue;
            };
            let (Ok(start), Ok(end)) = (
                usize::from_str_radix(start, 16),
                usize::from_str_radix(end, 16),
            ) else {
                continue;
            };
            let permissions = fields[1];
            let path = fields.get(5).copied().unwrap_or("").to_string();
            entries.push((start, end, path, permissions.contains('w')));
        }
    }
    entries
}

#[cfg(not(target_os = "macos"))]
fn elf_got_slots(symbol: &str, target: u64) -> Vec<GotSlotInfo> {
    let mut slots = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (start, end, path, writable) in proc_maps() {
        if !writable || !path.starts_with('/') {
            continue;
        }
        let mut slot = (start + 7) & !7;
        while slot + 8 <= end && slots.len() < 64 {
            // SAFETY: the scan is bounded by a writable mapping that the
            // kernel reports as live.
            let value = unsafe { ptr::read_volatile(slot as *const usize) } as u64;
            if value == target && seen.insert(slot) {
                slots.push(GotSlotInfo {
                    address: slot as u64,
                    symbol: symbol.to_string(),
                    image: path.clone(),
                    segment: String::new(),
                    section: String::new(),
                });
            }
            slot += 8;
        }
    }
    slots
}

/// Find already-bound GOT/PLT cells whose value is `symbol`.
pub fn find_got_slots(symbol: &str) -> Result<Vec<GotSlotInfo>, String> {
    let target = resolve_symbol(symbol)
        .ok_or_else(|| format!("symbol `{symbol}` is not resolvable in this process"))?;
    #[cfg(target_os = "macos")]
    {
        let slots = mach_o_got_slots(symbol, target);
        if slots.is_empty() {
            Err(format!(
                "no bound GOT/PLT slot found for `{symbol}` (the symbol may be \
                 lazily bound or resolved through chained fixups)"
            ))
        } else {
            Ok(slots)
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let slots = elf_got_slots(symbol, target);
        if slots.is_empty() {
            Err(format!(
                "no bound GOT/PLT slot found for `{symbol}` (the symbol may be \
                 lazily bound)"
            ))
        } else {
            Ok(slots)
        }
    }
}

/// Encode an absolute jump sled for the host architecture.
pub fn encode_jump(target: u64) -> Result<Vec<u8>, String> {
    #[cfg(target_arch = "aarch64")]
    {
        // ldr x16, #8 ; br x16 ; .quad target
        let mut code = Vec::with_capacity(16);
        code.extend_from_slice(&0x5800_0050u32.to_le_bytes());
        code.extend_from_slice(&0xD61F_0200u32.to_le_bytes());
        code.extend_from_slice(&target.to_le_bytes());
        Ok(code)
    }
    #[cfg(target_arch = "x86_64")]
    {
        // mov rax, imm64 ; jmp rax
        let mut code = Vec::with_capacity(12);
        code.extend_from_slice(&[0x48, 0xB8]);
        code.extend_from_slice(&target.to_le_bytes());
        code.extend_from_slice(&[0xFF, 0xE0]);
        Ok(code)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = target;
        Err("inline trampolines support only x86_64 and AArch64".to_string())
    }
}

fn flush_instruction_cache(address: usize, length: usize) {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: the range belongs to an executable region owned by us.
        unsafe { sys_icache_invalidate(address as *mut c_void, length) };
    }
    #[cfg(not(target_os = "macos"))]
    {
        // SAFETY: same requirement as above; `__clear_cache` touches the
        // instruction cache for the given address range.
        unsafe {
            __clear_cache(address as *mut c_char, (address + length) as *mut c_char);
        }
    }
}

/// A page range owned by this process that may contain machine code.
pub struct ExecutableRegion {
    address: usize,
    size: usize,
    jit: bool,
    closed: bool,
}

impl ExecutableRegion {
    pub fn new(size: usize) -> Result<Self, String> {
        let page = page_size();
        let size = size.max(1).div_ceil(page) * page;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // SAFETY: anonymous, private mapping with no file descriptor.
            let address = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    libc::MAP_PRIVATE | MAP_ANON | libc::MAP_JIT,
                    -1,
                    0,
                )
            };
            if address != libc::MAP_FAILED {
                // SAFETY: the JIT primitive is available on this platform.
                let supported = unsafe { pthread_jit_write_protect_supported_np() } != 0;
                return Ok(Self {
                    address: address as usize,
                    size,
                    jit: supported,
                    closed: false,
                });
            }
        }
        // SAFETY: as above, without MAP_JIT.
        let address = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                libc::MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "cannot allocate an executable page (errno {})",
                errno()
            ));
        }
        // SAFETY: the mapping is live and page-aligned.
        let result = unsafe { libc::mprotect(address, size, PROT_READ | PROT_EXEC) };
        if result != 0 {
            // SAFETY: the mapping was just created by us.
            unsafe { libc::munmap(address, size) };
            return Err("cannot mark the page executable".to_string());
        }
        Ok(Self {
            address: address as usize,
            size,
            jit: false,
            closed: false,
        })
    }

    pub fn address(&self) -> usize {
        self.address
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_jit(&self) -> bool {
        self.jit
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn check(&self, offset: usize, length: usize) -> Result<(), String> {
        if self.closed {
            return Err("executable region is closed".to_string());
        }
        if offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(format!(
                "write of {length} bytes at offset {offset} exceeds {} bytes",
                self.size
            ));
        }
        Ok(())
    }

    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), String> {
        self.check(offset, data.len())?;
        let destination = (self.address + offset) as *mut u8;
        if self.jit {
            // SAFETY: the region was mapped writable and executable with
            // MAP_JIT; the write-protect toggle is per-thread.
            unsafe {
                pthread_jit_write_protect_np(0);
                ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
                flush_instruction_cache(destination as usize, data.len());
                pthread_jit_write_protect_np(1);
            }
            return Ok(());
        }
        // SAFETY: the mapping is live; widen it to writable, copy, then make
        // it executable again.
        unsafe {
            if libc::mprotect(
                self.address as *mut c_void,
                self.size,
                PROT_READ | PROT_WRITE,
            ) != 0
            {
                return Err("cannot reopen the executable page for writing".to_string());
            }
            ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
            flush_instruction_cache(destination as usize, data.len());
            if libc::mprotect(
                self.address as *mut c_void,
                self.size,
                PROT_READ | PROT_EXEC,
            ) != 0
            {
                return Err("cannot restore execute permission".to_string());
            }
        }
        Ok(())
    }

    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, String> {
        self.check(offset, length)?;
        let mut buffer = vec![0u8; length];
        // SAFETY: bounds were checked against the owned mapping.
        unsafe {
            ptr::copy_nonoverlapping(
                (self.address + offset) as *const u8,
                buffer.as_mut_ptr(),
                length,
            );
        }
        Ok(buffer)
    }

    pub fn close(&mut self) {
        if !self.closed && self.address != 0 {
            // SAFETY: the mapping was created by `new` and is not used after
            // `close`.
            unsafe { libc::munmap(self.address as *mut c_void, self.size) };
            self.address = 0;
            self.closed = true;
        }
    }
}

impl Drop for ExecutableRegion {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_encoding_has_expected_shape() {
        let sled = encode_jump(0x1234_5678_9ABC_DEF0).expect("encoding");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(sled.len(), 16);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(sled.len(), 12);
        assert!(sled
            .windows(8)
            .any(|window| window == 0x1234_5678_9ABC_DEF0u64.to_le_bytes()));
    }

    #[test]
    fn pointer_patch_is_atomic_and_reversible() {
        let cell = AtomicUsize::new(0x11);
        let slot = &cell as *const AtomicUsize as usize;
        let old = patch_pointer(slot, 0x22).expect("patch");
        assert_eq!(old, 0x11);
        assert_eq!(cell.load(Ordering::SeqCst), 0x22);
        assert_eq!(patch_pointer(slot, old).expect("restore"), 0x22);
        assert_eq!(cell.load(Ordering::SeqCst), 0x11);
    }

    #[test]
    fn pointer_patch_rejects_unaligned_cells() {
        let cell = AtomicUsize::new(0);
        let slot = &cell as *const AtomicUsize as usize + 1;
        assert!(patch_pointer(slot, 1).is_err());
    }

    #[cfg(target_arch = "aarch64")]
    fn machine_code(value: u32) -> Vec<u8> {
        let mut code = Vec::new();
        code.extend_from_slice(&(0x5280_0000 | (value << 5)).to_le_bytes());
        code.extend_from_slice(&0xD65F_03C0u32.to_le_bytes());
        code
    }

    #[cfg(target_arch = "x86_64")]
    fn machine_code(value: u32) -> Vec<u8> {
        let mut code = vec![0xB8];
        code.extend_from_slice(&value.to_le_bytes());
        code.push(0xC3);
        code
    }

    type Nullary = extern "C" fn() -> i64;

    fn call(address: usize) -> i64 {
        // SAFETY: the caller only invokes regions containing a valid Nullary.
        let function: Nullary = unsafe { std::mem::transmute(address) };
        function()
    }

    #[test]
    fn owned_region_executes_and_accepts_sleds() {
        let mut first = ExecutableRegion::new(4096).expect("region");
        let mut second = ExecutableRegion::new(4096).expect("region");
        first.write(0, &machine_code(1)).expect("write");
        second.write(0, &machine_code(2)).expect("write");
        assert_eq!(call(first.address()), 1);
        assert_eq!(call(second.address()), 2);

        let original = first.read(0, 16).expect("read");
        let sled = encode_jump(second.address() as u64).expect("sled");
        first.write(0, &sled).expect("sled write");
        assert_eq!(call(first.address()), 2);
        first.write(0, &original).expect("restore");
        assert_eq!(call(first.address()), 1);
    }

    #[test]
    fn resolution_finds_libc_symbols() {
        assert!(resolve_symbol("malloc").is_some());
        assert!(resolve_symbol("polyglot_symbol_that_does_not_exist").is_none());
    }

    #[test]
    fn got_discovery_finds_the_bound_getpid_cell() {
        let slots = find_got_slots("getpid").expect("bound symbol");
        assert!(!slots.is_empty());
        assert!(slots.iter().all(|slot| slot.symbol == "getpid"));
    }
}
