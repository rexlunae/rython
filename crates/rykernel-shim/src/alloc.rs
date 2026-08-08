//! kmalloc-backed global allocator — the kernel-context equivalent of the
//! system allocator. Enables `alloc` (String, Vec, …) inside a module.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_uint;

use crate::ffi;

/// `GFP_KERNEL` for the running kernel family (6.x/7.x, x86-64).
pub const GFP_KERNEL: c_uint = 0xCC0;

/// Global allocator backed by `kmalloc`/`kfree`.
pub struct KernelAllocator;

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { ffi::__kmalloc_noprof(layout.size(), GFP_KERNEL) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { ffi::kfree(ptr as *const core::ffi::c_void) }
    }
}

/// Install as the crate's global allocator:
///
/// ```rust
/// #[global_allocator]
/// static ALLOCATOR: rykernel_shim::alloc::KernelAllocator =
///     rykernel_shim::alloc::KernelAllocator;
/// ```
pub static ALLOCATOR: KernelAllocator = KernelAllocator;
