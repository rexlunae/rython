//! rykernel-shim — a minimal, C-free runtime for out-of-tree Linux kernel
//! modules written in plain Rust.
//!
//! This crate is the "shim" layer for the raw-FFI kernel target: it declares
//! the kernel symbols a module needs (`_printk`, `kmalloc`/`kfree`,
//! `misc_register`, the copy helpers, …), provides a kmalloc-backed global
//! allocator, a panic handler, a spinlock, and a `MiscDevice` registration
//! helper with the full `struct file_operations` layout.
//!
//! It deliberately does **not** depend on rust-for-linux or a `CONFIG_RUST`
//! kernel: everything here is plain `#![no_std]` Rust plus `extern "C"`
//! declarations of symbols every kernel exports to modules
//! (`EXPORT_SYMBOL*`). Build with:
//!
//! ```text
//! cargo build --release --target x86_64-unknown-none --manifest-path crates/rykernel-shim/Cargo.toml
//! ```
//!
//! (see `rypip --kernel-module` for the full ld/kbuild recipe).

#![cfg_attr(target_os = "none", no_std)]

pub mod ffi;
pub mod sync;
pub mod device;
pub mod alloc;

// The module's own `struct module`, emitted by kbuild's modpost as
// `__this_module` in `{name}.mod.c` — the Rust equivalent of `THIS_MODULE`
// in C. Only its address is ever used (e.g. `FileOperations::owner`).
unsafe extern "C" {
    pub static __this_module: ModuleOpaque;
}

/// Opaque stand-in for `struct module`. Never dereferenced by the shim.
#[repr(C)]
pub struct ModuleOpaque {
    _private: u8,
}

/// Kernel panic handler. Prints a fixed message with `_printk` and then
/// spins forever — there is no way out of a kernel panic.
///
/// Deliberately avoids `core::fmt`: formatting the panic info would pull in
/// core's panic-message machinery, whose precompiled rlib objects carry
/// GOTPCREL relocations the kernel module loader rejects. The message text
/// is static; that is enough to say what happened.
///
/// Only defined for freestanding targets (`x86_64-unknown-none`): on host
/// builds (where this crate may be cargo-checked as part of the workspace)
/// the standard library provides the panic handler.
#[cfg(target_os = "none")]
#[panic_handler]
fn kernel_panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        ffi::_printk(b"kernel panic: module fault, spinning\0".as_ptr() as *const core::ffi::c_char);
    }
    loop {
        core::hint::spin_loop();
    }
}

/// `printk!` — the Rust equivalent of the C `printk` macro (which is a macro
/// around `_printk` in the kernel). The format string follows the kernel's C
/// format parser: `%ld` conversions, `%%` for a literal percent sign.
#[macro_export]
macro_rules! printk {
    ($fmt:literal) => {
        unsafe {
            $crate::ffi::_printk(
                concat!($fmt, "\0").as_ptr() as *const core::ffi::c_char
            )
        }
    };
    ($fmt:literal, $($arg:expr),+ $(,)?) => {
        unsafe {
            $crate::ffi::_printk(
                concat!($fmt, "\0").as_ptr() as *const core::ffi::c_char,
                $($arg),+
            )
        }
    };
}

/// Wall-clock seconds since the Unix epoch, read from the kernel's own
/// clock (`ktime_get_real_seconds`, exported to modules).
///
/// This is the canonical "kernel resource" for rython's kernel-module FFI:
/// kernel-module Python can import it (`from rykernel_shim import
/// ktime_get_real_seconds`) and rypip lowers the call to a direct call into
/// this crate, which calls the kernel symbol directly. No user-space
/// equivalent exists — this is a resource only the kernel has.
pub fn ktime_get_real_seconds() -> i64 {
    unsafe { ffi::ktime_get_real_seconds() }
}
