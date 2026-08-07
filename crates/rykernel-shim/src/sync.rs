//! A tiny kernel-context spinlock, implemented with atomics only.
//!
//! The kernel's own `struct mutex`/`raw_spinlock_t` layouts are
//! config-dependent and not worth re-deriving in Rust; for the short,
//! non-sleeping critical sections a device shim needs (ring-buffer moves,
//! counter updates) a spinlock is the honest primitive. It uses no kernel
//! symbols, so it works on every kernel.

use core::sync::atomic::{AtomicBool, Ordering};

/// A simple test-and-set spinlock.
pub struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    /// Create an unlocked spinlock (usable in `static` contexts).
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Acquire the lock, spinning until it is free.
    pub fn lock(&self) -> SpinGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinGuard { lock: self }
    }

    /// Try to acquire without spinning.
    pub fn try_lock(&self) -> Option<SpinGuard<'_>> {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| SpinGuard { lock: self })
    }
}

/// RAII guard that releases the lock on drop.
pub struct SpinGuard<'a> {
    lock: &'a SpinLock,
}

impl Drop for SpinGuard<'_> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

unsafe impl Send for SpinLock {}
unsafe impl Sync for SpinLock {}
