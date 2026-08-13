//! Misc-character-device registration with the full `struct file_operations`
//! layout, plus user-copy helpers.
//!
//! ABI notes:
//! - `struct file_operations` is `__randomize_layout` in the kernel, but the
//!   running kernel was built with `CONFIG_RANDSTRUCT_NONE=y` (the distro
//!   default — RANDSTRUCT breaks the module ABI), so the layout is exactly
//!   the header order below. The struct is defined in full so the kernel
//!   never reads past our copy when it probes later fields (e.g. `.release`
//!   on close). A compile-time size assertion against the field count guards
//!   the most likely drift.
//! - `struct miscdevice` layout comes from `include/linux/miscdevice.h`; the
//!   `list_head` member is real storage because `misc_register` links the
//!   device into the kernel's misc list.

use crate::ffi;
use core::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};

// ---------------------------------------------------------------------------
// Opaque kernel types referenced by file_operations
// ---------------------------------------------------------------------------

macro_rules! opaque {
    ($($name:ident),+ $(,)?) => {
        $(
            #[repr(C)]
            pub struct $name {
                _private: [u8; 0],
            }
        )+
    };
}

opaque!(
    File,
    Inode,
    Kiocb,
    IovIter,
    IoCompBatch,
    DirContext,
    PollTable,
    VmArea,
    FileLock,
    FileLease,
    PipeInode,
    SeqFile,
    UringCmd,
    VmAreaDesc,
);

// ---------------------------------------------------------------------------
// struct file_operations (include/linux/fs.h)
// ---------------------------------------------------------------------------

/// `struct file_operations` — full layout, in header order
/// (CONFIG_RANDSTRUCT_NONE). Every member is a nullable function pointer
/// except `owner` and `fop_flags`; leave unused members `None` (NULL).
#[repr(C)]
pub struct FileOperations {
    /// `struct module *owner` — set to `&rykernel_shim::__this_module`
    /// (the `ModuleOpaque` from the crate root).
    pub owner: *const crate::ModuleOpaque,
    /// `fop_flags_t fop_flags` (`typedef unsigned int`).
    pub fop_flags: u32,
    pub llseek: Option<unsafe extern "C" fn(*mut File, i64, c_int) -> i64>,
    pub read: Option<unsafe extern "C" fn(*mut File, *mut c_void, usize, *mut i64) -> isize>,
    pub write: Option<unsafe extern "C" fn(*mut File, *const c_void, usize, *mut i64) -> isize>,
    pub read_iter: Option<unsafe extern "C" fn(*mut Kiocb, *mut IovIter) -> isize>,
    pub write_iter: Option<unsafe extern "C" fn(*mut Kiocb, *mut IovIter) -> isize>,
    pub iopoll: Option<unsafe extern "C" fn(*mut Kiocb, *mut IoCompBatch, c_uint) -> c_int>,
    pub iterate_shared: Option<unsafe extern "C" fn(*mut File, *mut DirContext) -> c_int>,
    pub poll: Option<unsafe extern "C" fn(*mut File, *mut PollTable) -> u32>,
    pub unlocked_ioctl: Option<unsafe extern "C" fn(*mut File, c_uint, c_ulong) -> c_long>,
    pub compat_ioctl: Option<unsafe extern "C" fn(*mut File, c_uint, c_ulong) -> c_long>,
    pub mmap: Option<unsafe extern "C" fn(*mut File, *mut VmArea) -> c_int>,
    pub open: Option<unsafe extern "C" fn(*mut Inode, *mut File) -> c_int>,
    pub flush: Option<unsafe extern "C" fn(*mut File, *mut c_void) -> c_int>,
    pub release: Option<unsafe extern "C" fn(*mut Inode, *mut File) -> c_int>,
    pub fsync: Option<unsafe extern "C" fn(*mut File, i64, i64, c_int) -> c_int>,
    pub fasync: Option<unsafe extern "C" fn(c_int, *mut File, c_int) -> c_int>,
    pub lock: Option<unsafe extern "C" fn(*mut File, c_int, *mut FileLock) -> c_int>,
    pub get_unmapped_area:
        Option<unsafe extern "C" fn(*mut File, c_ulong, c_ulong, c_ulong, c_ulong) -> c_ulong>,
    pub check_flags: Option<unsafe extern "C" fn(c_int) -> c_int>,
    pub flock: Option<unsafe extern "C" fn(*mut File, c_int, *mut FileLock) -> c_int>,
    pub splice_write:
        Option<unsafe extern "C" fn(*mut PipeInode, *mut File, *mut i64, usize, c_uint) -> isize>,
    pub splice_read:
        Option<unsafe extern "C" fn(*mut File, *mut i64, *mut PipeInode, usize, c_uint) -> isize>,
    pub splice_eof: Option<unsafe extern "C" fn(*mut File)>,
    pub setlease: Option<unsafe extern "C" fn(*mut File, c_int, *mut FileLease, *mut *mut c_void) -> c_int>,
    pub fallocate: Option<unsafe extern "C" fn(*mut File, c_int, i64, i64) -> c_long>,
    pub show_fdinfo: Option<unsafe extern "C" fn(*mut SeqFile, *mut File)>,
    pub copy_file_range: Option<unsafe extern "C" fn(*mut File, i64, *mut File, i64, usize, c_uint) -> isize>,
    pub remap_file_range: Option<unsafe extern "C" fn(*mut File, i64, *mut File, i64, i64, c_uint) -> i64>,
    pub fadvise: Option<unsafe extern "C" fn(*mut File, i64, i64, c_int) -> c_int>,
    pub uring_cmd: Option<unsafe extern "C" fn(*mut UringCmd, c_uint) -> c_int>,
    pub uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut UringCmd, *mut IoCompBatch, c_uint) -> c_int>,
    pub mmap_prepare: Option<unsafe extern "C" fn(*mut VmAreaDesc) -> c_int>,
}

// owner(8) + fop_flags(4 + 4 pad) + 32 fn pointers * 8 = 272 bytes.
// (CONFIG_MMU is set on every x86-64 kernel, so mmap_capabilities is absent.)
const _: () = assert!(core::mem::size_of::<FileOperations>() == 272);

// Plain data record read by the kernel; raw pointers and fn pointers are
// not `Sync` by default, but the table is immutable after init.
unsafe impl Sync for FileOperations {}

impl FileOperations {
    /// An all-NULL operations table — the base for a `static` definition
    /// using struct-update syntax:
    ///
    /// ```ignore
    /// // Illustrative: FileOperations and the dev_* handlers are module
    /// // items, not doctest-visible, and the table feeds the kernel's
    /// // file_operations, so this is not a compilable host snippet.
    /// static FOPS: FileOperations = FileOperations {
    ///     owner: &rykernel_shim::__this_module,
    ///     read: Some(dev_read),
    ///     write: Some(dev_write),
    ///     unlocked_ioctl: Some(dev_ioctl),
    ///     ..FileOperations::empty()
    /// };
    /// ```
    pub const fn empty() -> Self {
        Self {
            owner: core::ptr::null(),
            fop_flags: 0,
            llseek: None,
            read: None,
            write: None,
            read_iter: None,
            write_iter: None,
            iopoll: None,
            iterate_shared: None,
            poll: None,
            unlocked_ioctl: None,
            compat_ioctl: None,
            mmap: None,
            open: None,
            flush: None,
            release: None,
            fsync: None,
            fasync: None,
            lock: None,
            get_unmapped_area: None,
            check_flags: None,
            flock: None,
            splice_write: None,
            splice_read: None,
            splice_eof: None,
            setlease: None,
            fallocate: None,
            show_fdinfo: None,
            copy_file_range: None,
            remap_file_range: None,
            fadvise: None,
            uring_cmd: None,
            uring_cmd_iopoll: None,
            mmap_prepare: None,
        }
    }
}

// ---------------------------------------------------------------------------
// struct miscdevice (include/linux/miscdevice.h)
// ---------------------------------------------------------------------------

/// `struct miscdevice` — the record passed to `misc_register`.
#[repr(C)]
pub struct MiscDeviceRaw {
    pub minor: c_int,
    pub name: *const c_char,
    pub fops: *const FileOperations,
    /// `struct list_head` — real storage; `misc_register` links this in.
    pub list: [usize; 2],
    pub parent: *const c_void,
    pub this_device: *const c_void,
    pub groups: *const c_void,
    pub nodename: *const c_char,
    pub mode: u16,
}

/// A misc character device (`/dev/<name>`), backed by the provided fops.
///
/// The device record must live at a stable address for the kernel's whole
/// registration lifetime — store it in a `static`.
pub struct MiscDevice {
    raw: MiscDeviceRaw,
}

impl MiscDevice {
    /// Build a device record for a static. `name` must be NUL-terminated.
    pub const fn new(name: &'static [u8], fops: &'static FileOperations, mode: u16) -> Self {
        Self {
            raw: MiscDeviceRaw {
                // MISC_DYNAMIC_MINOR is 255, not -1: misc_register() checks
                // `minor == 255` for dynamic allocation and rejects
                // `minor > 255` with -EINVAL. Using -1 (0xFFFFFFFF) would
                // silently take the fixed-minor path and fail with -EBUSY.
                minor: 255,
                name: name.as_ptr() as *const c_char,
                fops,
                list: [0; 2],
                parent: core::ptr::null(),
                this_device: core::ptr::null(),
                groups: core::ptr::null(),
                nodename: core::ptr::null(),
                mode,
            },
        }
    }

    /// Register the device (wraps `misc_register`). Returns `Ok` on success
    /// or the negative errno from the kernel.
    pub fn register(&'static self) -> Result<(), c_int> {
        let ret = unsafe { ffi::misc_register(&self.raw as *const MiscDeviceRaw as *const c_void) };
        if ret < 0 {
            Err(ret)
        } else {
            Ok(())
        }
    }

    /// Deregister the device (wraps `misc_deregister`).
    pub fn deregister(&'static self) {
        unsafe { ffi::misc_deregister(&self.raw as *const MiscDeviceRaw as *const c_void) };
    }
}

unsafe impl Sync for MiscDevice {}

// ---------------------------------------------------------------------------
// User-copy helpers
// ---------------------------------------------------------------------------

/// Copy `n` bytes from kernel memory to user memory. Returns `Err(EFAULT)`
/// if any byte could not be copied.
pub unsafe fn copy_to_user(dst: *mut c_void, src: *const u8, n: usize) -> Result<(), c_int> {
    if n == 0 {
        return Ok(());
    }
    let left = unsafe { ffi::_copy_to_user(dst, src as *const c_void, n as c_ulong) };
    if left == 0 {
        Ok(())
    } else {
        Err(ffi::EFAULT)
    }
}

/// Copy `n` bytes from user memory to kernel memory. Returns `Err(EFAULT)`
/// if any byte could not be copied.
pub unsafe fn copy_from_user(dst: *mut u8, src: *const c_void, n: usize) -> Result<(), c_int> {
    if n == 0 {
        return Ok(());
    }
    let left = unsafe { ffi::_copy_from_user(dst as *mut c_void, src, n as c_ulong) };
    if left == 0 {
        Ok(())
    } else {
        Err(ffi::EFAULT)
    }
}
