//! Python threading module — thread management and synchronization on
//! `std::thread` / `std::sync`.
//!
//! Every object here is a HANDLE with Python's reference semantics:
//! cloning shares the underlying thread/lock/event (the codegen clones
//! thread arguments into the spawned closure, so passing a Lock or Event
//! to a worker shares it exactly as CPython does; plain containers follow
//! rython's usual value semantics — the §12 ledger divergence).
//!
//! Divergences (documented in docs/spec.md §12):
//! - CPython's interpreter waits for non-daemon threads at exit; rython
//!   joins a started, never-joined thread when its LAST handle drops (at
//!   latest, end of `main`). Daemon threads detach and die with the
//!   process, as in CPython.
//! - An unhandled exception in a thread prints CPython's header and final
//!   exception line but no traceback frames (rython has no frames).
//! - `start()`/`join()` misuse panics with CPython's RuntimeError message
//!   (their Python signatures return values, so the `Result` channel is
//!   not available — the `time.sleep` precedent); Lock/RLock `release()`
//!   errors are catchable `RuntimeError`s.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::PyException;

/// Live started-and-unfinished non-main threads. active_count() adds 1
/// for the main thread, matching CPython's enumeration.
static ACTIVE_THREADS: AtomicI64 = AtomicI64::new(0);
/// CPython names threads "Thread-N (target)" with N starting at 1.
static THREAD_COUNTER: AtomicI64 = AtomicI64::new(0);

type ThreadBody = Box<dyn FnOnce() + Send + 'static>;

struct ThreadState {
    body: Option<ThreadBody>,
    handle: Option<std::thread::JoinHandle<()>>,
    started: bool,
}

struct ThreadInner {
    state: Mutex<ThreadState>,
    finished: Arc<AtomicBool>,
    name: String,
    daemon: bool,
}

/// threading.Thread — created by the codegen's `threading.Thread(target=,
/// args=)` lowering via [`Thread::new`]; user code drives it with
/// `start()` / `join()` / `is_alive()`.
#[derive(Clone)]
pub struct Thread {
    inner: Arc<ThreadInner>,
}

impl Thread {
    /// `target_name` is the Python target function's name, used only for
    /// CPython's "Thread-N (target)" naming.
    pub fn new<F: FnOnce() + Send + 'static>(target_name: &str, daemon: bool, body: F) -> Thread {
        let n = THREAD_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
        Thread {
            inner: Arc::new(ThreadInner {
                state: Mutex::new(ThreadState {
                    body: Some(Box::new(body)),
                    handle: None,
                    started: false,
                }),
                finished: Arc::new(AtomicBool::new(false)),
                name: format!("Thread-{} ({})", n, target_name),
                daemon,
            }),
        }
    }

    /// Python `Thread.start()`. Panics with CPython's RuntimeError on a
    /// second start ("threads can only be started once").
    pub fn start(&self) {
        let mut st = self.inner.state.lock().unwrap();
        if st.started {
            // Release the lock before unwinding so the Drop-join sees a
            // healthy mutex.
            drop(st);
            // Verified against python3: RuntimeError('threads can only be started once')
            panic!(
                "{}",
                PyException::new("RuntimeError", "threads can only be started once")
            );
        }
        st.started = true;
        let body = st.body.take().expect("unstarted thread has a body");
        let finished = Arc::clone(&self.inner.finished);
        ACTIVE_THREADS.fetch_add(1, Ordering::SeqCst);
        /// Runs on normal completion AND unwind, so `is_alive()` and
        /// `active_count()` stay truthful when the body raises.
        struct Cleanup {
            finished: Arc<AtomicBool>,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                self.finished.store(true, Ordering::SeqCst);
                ACTIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let spawned = std::thread::Builder::new()
            .name(self.inner.name.clone())
            .spawn(move || {
                let _cleanup = Cleanup { finished };
                body();
            });
        match spawned {
            Ok(handle) => st.handle = Some(handle),
            Err(e) => {
                ACTIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
                panic!(
                    "{}",
                    PyException::new("RuntimeError", format!("can't start new thread: {}", e))
                );
            }
        }
    }

    /// Python `Thread.join()`. Panics with CPython's RuntimeError when the
    /// thread was never started; a repeated join returns immediately (as
    /// CPython's does once the thread has finished).
    pub fn join(&self) {
        let handle = {
            let mut st = self.inner.state.lock().unwrap();
            if !st.started {
                drop(st);
                // Verified against python3: RuntimeError('cannot join thread before it is started')
                panic!(
                    "{}",
                    PyException::new("RuntimeError", "cannot join thread before it is started")
                );
            }
            st.handle.take()
        };
        if let Some(h) = handle {
            // A panicked body already reported itself; join still succeeds
            // (CPython's join returns normally after a thread's exception).
            let _ = h.join();
        }
    }

    /// Python `Thread.is_alive()`: true between start() and the body's end.
    pub fn is_alive(&self) -> bool {
        let st = self.inner.state.lock().unwrap();
        st.started && !self.inner.finished.load(Ordering::SeqCst)
    }
}

impl Drop for ThreadInner {
    fn drop(&mut self) {
        // The last handle joins a still-running non-daemon thread — the
        // closest std-Rust equivalent of CPython's interpreter-exit join
        // (Rust would otherwise kill it when main returns).
        if self.daemon {
            return;
        }
        // Tolerate a poisoned mutex (a panic elsewhere must not turn into
        // a second panic inside Drop — that aborts).
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(h) = state.handle.take() {
            let _ = h.join();
        }
    }
}

/// The unhandled-exception reporter the codegen wraps thread bodies with.
/// CPython prints "Exception in thread NAME:" plus a traceback; rython has
/// no frames, so the header and the final exception line are kept.
pub fn report_thread_exception(e: &PyException) {
    eprintln!("Exception in thread {}:", current_thread().name);
    eprintln!("{}", e);
}

/// threading.current_thread() — a lightweight view carrying the `.name`
/// attribute ("MainThread" on the main thread, the "Thread-N (target)"
/// name inside spawned ones).
pub struct CurrentThread {
    pub name: String,
}

pub fn current_thread() -> CurrentThread {
    let name = match std::thread::current().name() {
        // Rust's main thread is named "main"; CPython calls it MainThread.
        Some("main") | None => "MainThread".to_string(),
        Some(other) => other.to_string(),
    };
    CurrentThread { name }
}

/// threading.active_count() — the main thread plus live started threads.
pub fn active_count() -> i64 {
    ACTIVE_THREADS.load(Ordering::SeqCst) + 1
}

/// threading.get_ident() — the current thread's opaque integer identity
/// (urllib3's connectionpool keys per-thread state on it). CPython
/// promises only uniqueness among live threads; a stable per-thread hash
/// of the Rust ThreadId satisfies that.
pub fn get_ident() -> i64 {
    use core::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    (h.finish() & 0x7fff_ffff_ffff_ffff) as i64
}

struct LockInner {
    locked: Mutex<bool>,
    cv: Condvar,
}

/// threading.Lock — non-reentrant; release() from any thread, as CPython.
#[derive(Clone)]
pub struct Lock {
    inner: Arc<LockInner>,
}

/// threading.Lock() — the constructor function (Python spells types and
/// callables the same; the struct lives in the type namespace).
#[allow(non_snake_case)]
pub fn Lock() -> Lock {
    Lock {
        inner: Arc::new(LockInner {
            locked: Mutex::new(false),
            cv: Condvar::new(),
        }),
    }
}

impl Lock {
    /// Python `lock.acquire()` — blocks, returns True.
    pub fn acquire(&self) -> Result<bool, PyException> {
        let mut locked = self.inner.locked.lock().unwrap();
        while *locked {
            locked = self.inner.cv.wait(locked).unwrap();
        }
        *locked = true;
        Ok(true)
    }

    /// Python `lock.release()` — RuntimeError on an unlocked lock.
    pub fn release(&self) -> Result<(), PyException> {
        let mut locked = self.inner.locked.lock().unwrap();
        if !*locked {
            // Verified against python3: RuntimeError('release unlocked lock')
            return Err(PyException::new("RuntimeError", "release unlocked lock"));
        }
        *locked = false;
        self.inner.cv.notify_one();
        Ok(())
    }

    /// Python `lock.locked()`.
    pub fn locked(&self) -> bool {
        *self.inner.locked.lock().unwrap()
    }

    /// `with lock:` — acquire now, release when the guard drops (Python's
    /// `__enter__`/`__exit__`, exception-safe through unwinding `?`).
    pub fn py_guard(&self) -> Result<LockReleaseGuard, PyException> {
        self.acquire()?;
        Ok(LockReleaseGuard { lock: self.clone() })
    }
}

/// RAII half of `with lock:` — releases on Drop.
pub struct LockReleaseGuard {
    lock: Lock,
}

impl Drop for LockReleaseGuard {
    fn drop(&mut self) {
        let _ = self.lock.release();
    }
}

struct RLockState {
    owner: Option<std::thread::ThreadId>,
    count: u64,
}

struct RLockInner {
    state: Mutex<RLockState>,
    cv: Condvar,
}

/// threading.RLock — reentrant; only the owning thread may release.
#[derive(Clone)]
pub struct RLock {
    inner: Arc<RLockInner>,
}

/// threading.RLock() — constructor function.
#[allow(non_snake_case)]
pub fn RLock() -> RLock {
    RLock {
        inner: Arc::new(RLockInner {
            state: Mutex::new(RLockState {
                owner: None,
                count: 0,
            }),
            cv: Condvar::new(),
        }),
    }
}

impl RLock {
    /// Python `rlock.acquire()` — reentrant for the owning thread.
    pub fn acquire(&self) -> Result<bool, PyException> {
        let me = std::thread::current().id();
        let mut st = self.inner.state.lock().unwrap();
        if st.owner == Some(me) {
            st.count += 1;
            return Ok(true);
        }
        while st.owner.is_some() {
            st = self.inner.cv.wait(st).unwrap();
        }
        st.owner = Some(me);
        st.count = 1;
        Ok(true)
    }

    /// Python `rlock.release()` — RuntimeError from a non-owner.
    pub fn release(&self) -> Result<(), PyException> {
        let me = std::thread::current().id();
        let mut st = self.inner.state.lock().unwrap();
        if st.owner != Some(me) {
            // Verified against python3: RuntimeError('cannot release un-acquired lock')
            return Err(PyException::new(
                "RuntimeError",
                "cannot release un-acquired lock",
            ));
        }
        st.count -= 1;
        if st.count == 0 {
            st.owner = None;
            self.inner.cv.notify_one();
        }
        Ok(())
    }

    /// `with rlock:` — see [`Lock::py_guard`].
    pub fn py_guard(&self) -> Result<RLockReleaseGuard, PyException> {
        self.acquire()?;
        Ok(RLockReleaseGuard { lock: self.clone() })
    }
}

/// RAII half of `with rlock:` — releases on Drop.
pub struct RLockReleaseGuard {
    lock: RLock,
}

impl Drop for RLockReleaseGuard {
    fn drop(&mut self) {
        let _ = self.lock.release();
    }
}

struct EventInner {
    set: Mutex<bool>,
    cv: Condvar,
}

/// threading.Event — a one-bit flag threads wait on.
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

/// threading.Event() — constructor function.
#[allow(non_snake_case)]
pub fn Event() -> Event {
    Event {
        inner: Arc::new(EventInner {
            set: Mutex::new(false),
            cv: Condvar::new(),
        }),
    }
}

impl Event {
    /// Python `event.is_set()`.
    pub fn is_set(&self) -> bool {
        *self.inner.set.lock().unwrap()
    }

    /// Python `event.set()` — wakes every waiter.
    pub fn set(&self) {
        let mut set = self.inner.set.lock().unwrap();
        *set = true;
        self.inner.cv.notify_all();
    }

    /// Python `event.clear()` — `&mut self` only because the transpiler's
    /// syntactic mutability analysis lists `clear` as mutating (list/dict
    /// clear); the shared inner state needs no exclusivity.
    pub fn clear(&mut self) {
        *self.inner.set.lock().unwrap() = false;
    }

    /// Python `event.wait()` — blocks until set, returns True.
    pub fn wait(&self) -> Result<bool, PyException> {
        let mut set = self.inner.set.lock().unwrap();
        while !*set {
            set = self.inner.cv.wait(set).unwrap();
        }
        Ok(true)
    }
}

struct SemaphoreInner {
    value: Mutex<i64>,
    cv: Condvar,
}

/// threading.Semaphore — a counter that blocks acquire() at zero.
#[derive(Clone)]
pub struct Semaphore {
    inner: Arc<SemaphoreInner>,
}

/// threading.Semaphore(value) — constructor function; the codegen supplies
/// CPython's default of 1 for the zero-argument spelling. Panics with
/// CPython's ValueError on a negative initial value (the constructor's
/// Python signature has no error channel).
#[allow(non_snake_case)]
pub fn Semaphore(value: i64) -> Semaphore {
    if value < 0 {
        // Verified against python3: ValueError('semaphore initial value must be >= 0')
        panic!(
            "{}",
            PyException::new("ValueError", "semaphore initial value must be >= 0")
        );
    }
    Semaphore {
        inner: Arc::new(SemaphoreInner {
            value: Mutex::new(value),
            cv: Condvar::new(),
        }),
    }
}

impl Semaphore {
    /// Python `sem.acquire()` — blocks at zero, returns True.
    pub fn acquire(&self) -> Result<bool, PyException> {
        let mut value = self.inner.value.lock().unwrap();
        while *value == 0 {
            value = self.inner.cv.wait(value).unwrap();
        }
        *value -= 1;
        Ok(true)
    }

    /// Python `sem.release()`.
    pub fn release(&self) -> Result<(), PyException> {
        let mut value = self.inner.value.lock().unwrap();
        *value += 1;
        self.inner.cv.notify_one();
        Ok(())
    }

    /// `with sem:` — see [`Lock::py_guard`].
    pub fn py_guard(&self) -> Result<SemaphoreReleaseGuard, PyException> {
        self.acquire()?;
        Ok(SemaphoreReleaseGuard { sem: self.clone() })
    }
}

/// RAII half of `with sem:` — releases on Drop.
pub struct SemaphoreReleaseGuard {
    sem: Semaphore,
}

impl Drop for SemaphoreReleaseGuard {
    fn drop(&mut self) {
        let _ = self.sem.release();
    }
}

// ---------------------------------------------------------------------------
// threading.local (issue #356)
// ---------------------------------------------------------------------------

/// Each `threading.local()` object's identity. A counter, not an address:
/// an address is reused once the object is freed, and a new object must
/// never see a dead one's per-thread attributes.
static NEXT_LOCAL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

std::thread_local! {
    /// This thread's attribute bags, one per live `threading.local()`
    /// object, keyed by the object's identity.
    static LOCAL_ATTRS: core::cell::RefCell<
        std::collections::HashMap<u64, crate::PyDict<String, crate::PyValue>>,
    > = core::cell::RefCell::new(std::collections::HashMap::new());
}

struct LocalId(u64);

impl Drop for LocalId {
    /// The last handle is gone: this thread's bag goes with it. (Another
    /// thread's bag for the object is freed when that thread exits.)
    fn drop(&mut self) {
        let _ = LOCAL_ATTRS.try_with(|m| m.borrow_mut().remove(&self.0));
    }
}

/// `threading.local()`: an object whose attributes are created at run time
/// (`tl.count = 0`) and are PER THREAD — every thread starts with none and
/// sees only its own. Values are boxed (`PyValue`), since the attribute set
/// and each attribute's type are decided by the program as it runs. A
/// handle clone is the same object (Python reference semantics).
#[derive(Clone)]
pub struct Local {
    id: Arc<LocalId>,
}

/// `threading.local()` — constructor function.
pub fn local() -> Local {
    Local {
        id: Arc::new(LocalId(NEXT_LOCAL_ID.fetch_add(1, Ordering::Relaxed))),
    }
}

/// The type under its Python name, so `from threading import local`
/// brings in both the constructor function and the type: a construction
/// through the imported name renders `local::new()`.
#[allow(non_camel_case_types)]
pub type local = Local;

impl Default for Local {
    fn default() -> Self {
        local()
    }
}

impl Local {
    /// `local()` — a new object with no attributes in any thread.
    pub fn new() -> Local {
        local()
    }

    fn missing(name: &str) -> PyException {
        PyException::new(
            "AttributeError",
            format!("'_thread._local' object has no attribute '{name}'"),
        )
    }

    /// `tl.name` — this thread's value, or CPython's AttributeError.
    pub fn py_getattr(&self, name: &str) -> Result<crate::PyValue, PyException> {
        LOCAL_ATTRS.with(|m| {
            m.borrow()
                .get(&self.id.0)
                .and_then(|bag| bag.get(name).cloned())
                .ok_or_else(|| Self::missing(name))
        })
    }

    /// `tl.name = value` — for this thread only.
    pub fn py_setattr<V: Into<crate::PyValue>>(&self, name: &str, value: V) {
        let value = value.into();
        LOCAL_ATTRS.with(|m| {
            m.borrow_mut()
                .entry(self.id.0)
                .or_default()
                .insert(name.to_string(), value);
        });
    }

    /// `hasattr(tl, "name")` — in this thread.
    pub fn py_hasattr(&self, name: &str) -> bool {
        LOCAL_ATTRS.with(|m| m.borrow().get(&self.id.0).is_some_and(|bag| bag.contains_key(name)))
    }

    /// `del tl.name` — this thread's binding, or CPython's AttributeError.
    pub fn py_delattr(&self, name: &str) -> Result<(), PyException> {
        LOCAL_ATTRS.with(|m| {
            m.borrow_mut()
                .get_mut(&self.id.0)
                .and_then(|bag| bag.shift_remove(name))
                .map(|_| ())
                .ok_or_else(|| Self::missing(name))
        })
    }
}

impl core::fmt::Debug for Local {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "<_thread._local object at 0x{:012x}>", Arc::as_ptr(&self.id) as usize)
    }
}

impl crate::PyDisplay for Local {
    fn py_display(&self) -> String {
        format!("{:?}", self)
    }
}

#[cfg(test)]
mod local_tests {
    use super::*;

    #[test]
    fn attributes_are_per_thread_and_per_object() {
        let tl = local();
        assert!(!tl.py_hasattr("n"));
        tl.py_setattr("n", 1i64);
        assert_eq!(tl.py_getattr("n").unwrap(), crate::PyValue::Int(1));
        // A clone is the same object.
        assert!(tl.clone().py_hasattr("n"));
        // Another object has its own attributes.
        assert!(!local().py_hasattr("n"));
        // Another thread starts empty and does not disturb this one.
        let other = tl.clone();
        std::thread::spawn(move || {
            assert!(!other.py_hasattr("n"));
            other.py_setattr("n", 2i64);
        })
        .join()
        .unwrap();
        assert_eq!(tl.py_getattr("n").unwrap(), crate::PyValue::Int(1));
        tl.py_delattr("n").unwrap();
        let err = tl.py_getattr("n").unwrap_err();
        assert_eq!(err.message, "'_thread._local' object has no attribute 'n'");
        assert!(tl.py_delattr("n").is_err());
    }
}
