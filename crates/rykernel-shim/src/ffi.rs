//! `extern "C"` declarations of kernel symbols exported to modules.
//!
//! Every symbol here is exported by the kernel to loadable modules
//! (`EXPORT_SYMBOL*` / `EXPORT_SYMBOL_GPL*`); `misc_register` etc. are
//! exported GPL. The signatures follow the kernel's C prototypes exactly.
//! `__user` pointers are plain pointers — the annotation is compile-time-only
//! in C and has no ABI effect.
//!
//! Rustdoc does not render documentation for items inside `extern` blocks, so
//! the per-symbol notes below are `//` comments.

use core::ffi::{c_char, c_int, c_uint, c_ulong, c_void};

// int _printk(const char *fmt, ...) — the symbol behind the printk macro.
// The kernel does NOT export `printk` itself; that is a macro.
// Use the `rykernel_shim::printk!` macro rather than calling this directly.
unsafe extern "C" {
    pub fn _printk(fmt: *const c_char, ...) -> c_int;
}

// void *__kmalloc_noprof(size_t size, gfp_t flags)  (GFP_KERNEL = 0xCC0 on 6.x/7.x)
// Kernel >= 7.0: the kmalloc() inline in slab.h lowers to __kmalloc_noprof()
// (or __kmalloc_large_noprof() for large sizes); the _noprof suffix comes
// from the allocation-profiling rework. On 6.x the exported symbol was
// `kmalloc` — adjust if you build against an older kernel.
unsafe extern "C" {
    pub fn __kmalloc_noprof(size: usize, flags: c_uint) -> *mut c_void;
}

// void kfree(const void *objp)
unsafe extern "C" {
    pub fn kfree(ptr: *const c_void);
}

// time64_t ktime_get_real_seconds(void) — the kernel's wall clock, seconds
// since the Unix epoch. Exported to modules (EXPORT_SYMBOL) and safe to
// call from any context that can sleep (module init/exit included). Use the
// `rykernel_shim::ktime_get_real_seconds` wrapper rather than calling this
// directly.
unsafe extern "C" {
    pub fn ktime_get_real_seconds() -> i64;
}

// unsigned long _copy_to_user(void __user *to, const void *from, unsigned long n)
// The C `copy_to_user()` macro wraps `_copy_to_user`; the underscore symbol
// is the one actually exported to modules.
// Returns the number of bytes that could NOT be copied.
unsafe extern "C" {
    pub fn _copy_to_user(to: *mut c_void, from: *const c_void, n: c_ulong) -> c_ulong;
}

// unsigned long _copy_from_user(void *to, const void __user *from, unsigned long n)
// Returns the number of bytes that could NOT be copied.
unsafe extern "C" {
    pub fn _copy_from_user(to: *mut c_void, from: *const c_void, n: c_ulong) -> c_ulong;
}

// int misc_register(struct miscdevice *misc)
unsafe extern "C" {
    pub fn misc_register(misc: *const c_void) -> c_int;
}

// void misc_deregister(struct miscdevice *misc)
unsafe extern "C" {
    pub fn misc_deregister(misc: *const c_void);
}

// int mutex_lock_interruptible(struct mutex *lock) — kept for modules that
// want interruptible sleeps; the shim's own SpinLock needs no kernel symbols.
unsafe extern "C" {
    pub fn mutex_lock_interruptible(lock: *mut c_void) -> c_int;
}

// void mutex_unlock(struct mutex *lock)
unsafe extern "C" {
    pub fn mutex_unlock(lock: *mut c_void);
}

/// Kernel's `-ERESTARTSYS` errno (returned by interruptible waits).
pub const ERESTARTSYS: c_int = -512;
/// Kernel's `-EFAULT` errno.
pub const EFAULT: c_int = -14;
/// Kernel's `-ENOMEM` errno.
pub const ENOMEM: c_int = -12;
/// Kernel's `-ENOTTY` errno (unknown ioctl).
pub const ENOTTY: c_int = -25;
