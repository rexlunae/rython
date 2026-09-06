//! Python's per-module import lock, for the generated `__module_init__`
//! of every crate module (issue #333; Devin reviews on #336 and #338).
//!
//! CPython runs a module's body once, under a lock keyed by the module:
//! the importing thread re-entering (an import cycle) continues with the
//! partially-initialized module; another thread blocks until the body has
//! finished; a body that raises leaves no module behind, so the next
//! import runs it again; and a wait cycle across threads (thread A owns
//! `a` and waits for `b`, thread B owns `b` and waits for `a`) is detected
//! (`_DeadlockError`) and the later importer proceeds with the partial
//! module instead of blocking forever. This type is that lock: one static
//! per module, a process-wide wait graph for the deadlock check, a
//! condition variable for the waiters, and an RAII guard so a body that
//! unwinds resets the module to NOT STARTED and wakes the waiters.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::ThreadId;

#[derive(Clone, Debug)]
enum State {
    NotStarted,
    Running(ThreadId),
    Done,
    /// The body raised after touching a module static (a value the
    /// process cannot re-initialize): the module stays failed and every
    /// later import raises the same exception — loud, where CPython would
    /// run the body again with fresh globals.
    Failed(crate::PyException),
}

/// The lock of one module (a `static` in the generated module).
pub struct ModuleInitLock {
    state: Mutex<State>,
    cv: Condvar,
}

/// The process-wide wait graph: which lock each blocked thread waits on,
/// and which thread owns each running lock (CPython's `_blocking_on`).
/// Always taken AFTER a module's own state lock.
struct WaitGraph {
    waiting_on: HashMap<ThreadId, usize>,
    owners: HashMap<usize, ThreadId>,
}

fn graph() -> &'static Mutex<WaitGraph> {
    static GRAPH: OnceLock<Mutex<WaitGraph>> = OnceLock::new();
    GRAPH.get_or_init(|| {
        Mutex::new(WaitGraph {
            waiting_on: HashMap::new(),
            owners: HashMap::new(),
        })
    })
}

impl WaitGraph {
    /// Would `me` waiting for the lock owned by `owner` close a cycle —
    /// is `owner` (transitively) waiting for a lock `me` owns?
    fn would_deadlock(&self, me: ThreadId, owner: ThreadId) -> bool {
        let mut cur = owner;
        for _ in 0..self.owners.len() + 1 {
            let Some(lock) = self.waiting_on.get(&cur) else {
                return false;
            };
            let Some(next) = self.owners.get(lock) else {
                return false;
            };
            if *next == me {
                return true;
            }
            cur = *next;
        }
        false
    }
}

/// What `enter` found.
pub enum ModuleInitEntry {
    /// The body has run: nothing to do.
    Done,
    /// The body is running on this thread (an import cycle), or a wait for
    /// it would deadlock across threads: continue with the module's
    /// partial state, as CPython does.
    Cycle,
    /// This caller runs the body; `finish` (or the drop) settles the state.
    Run(ModuleInitGuard),
    /// The body failed before and cannot run again faithfully: the same
    /// exception, raised again.
    Failed(crate::PyException),
}

/// The running body's guard: `finish(ok)` marks the module DONE (or NOT
/// STARTED on failure, so the next import retries); a guard dropped
/// without `finish` — the body unwound — resets to NOT STARTED too, and
/// wakes the waiters either way.
pub struct ModuleInitGuard {
    lock: &'static ModuleInitLock,
    finished: bool,
}

impl ModuleInitLock {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(State::NotStarted),
            cv: Condvar::new(),
        }
    }

    fn key(&'static self) -> usize {
        self as *const Self as usize
    }

    /// Enter the module's body: run it, skip it, or wait for it.
    pub fn enter(&'static self) -> ModuleInitEntry {
        let me = std::thread::current().id();
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            match &*st {
                State::Done => return ModuleInitEntry::Done,
                State::Failed(e) => return ModuleInitEntry::Failed(e.clone()),
                State::Running(owner) if *owner == me => return ModuleInitEntry::Cycle,
                State::Running(owner) => {
                    let owner = *owner;
                    let mut g = graph().lock().unwrap_or_else(|p| p.into_inner());
                    if g.would_deadlock(me, owner) {
                        return ModuleInitEntry::Cycle;
                    }
                    g.waiting_on.insert(me, self.key());
                    drop(g);
                    st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
                    graph()
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .waiting_on
                        .remove(&me);
                }
                State::NotStarted => {
                    *st = State::Running(me);
                    graph()
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .owners
                        .insert(self.key(), me);
                    return ModuleInitEntry::Run(ModuleInitGuard {
                        lock: self,
                        finished: false,
                    });
                }
            }
        }
    }

    fn settle(&'static self, state: State) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        *st = state;
        graph()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .owners
            .remove(&self.key());
        self.cv.notify_all();
    }
}

impl Default for ModuleInitLock {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleInitGuard {
    /// The body finished: DONE when it succeeded, NOT STARTED when it
    /// raised (the next import runs it again).
    pub fn finish(mut self, ok: bool) {
        self.finished = true;
        self.lock
            .settle(if ok { State::Done } else { State::NotStarted });
    }

    /// The body raised after touching a module static, which the process
    /// cannot re-initialize: the module stays FAILED, and every later
    /// import raises `err` again (CPython would run the body again with
    /// fresh globals — a re-run here would see the first attempt's
    /// values, so the failure is kept loud instead).
    pub fn poison(mut self, err: crate::PyException) {
        self.finished = true;
        self.lock.settle(State::Failed(err));
    }
}

impl Drop for ModuleInitGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.lock.settle(State::NotStarted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn a_second_thread_waits_for_the_body_and_then_finds_it_done() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let ModuleInitEntry::Run(guard) = LOCK.enter() else {
            panic!("first entry runs");
        };
        let ran2 = ran.clone();
        let waiter = std::thread::spawn(move || {
            let entry = LOCK.enter();
            // The body has finished by the time the waiter is released.
            assert_eq!(ran2.load(Ordering::SeqCst), 1);
            assert!(matches!(entry, ModuleInitEntry::Done));
        });
        std::thread::sleep(Duration::from_millis(100));
        ran.store(1, Ordering::SeqCst);
        guard.finish(true);
        waiter.join().unwrap();
    }

    #[test]
    fn the_owning_thread_re_entering_is_a_cycle_and_a_failure_resets() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter() else {
            panic!("first entry runs");
        };
        assert!(matches!(LOCK.enter(), ModuleInitEntry::Cycle));
        guard.finish(false);
        // The failed body is NOT STARTED again: the next import runs it.
        let ModuleInitEntry::Run(guard) = LOCK.enter() else {
            panic!("a failed module runs again");
        };
        guard.finish(true);
        assert!(matches!(LOCK.enter(), ModuleInitEntry::Done));
    }

    #[test]
    fn a_poisoned_module_raises_the_same_error_on_every_later_import() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter() else {
            panic!("first entry runs");
        };
        guard.poison(crate::PyException::new("RuntimeError", "first attempt fails"));
        for _ in 0..2 {
            let ModuleInitEntry::Failed(e) = LOCK.enter() else {
                panic!("a poisoned module never runs again");
            };
            assert_eq!(e.message, "first attempt fails");
        }
    }

    #[test]
    fn a_dropped_guard_resets_the_module_and_wakes_the_waiters() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter() else {
            panic!("first entry runs");
        };
        let waiter = std::thread::spawn(move || matches!(LOCK.enter(), ModuleInitEntry::Run(_)));
        std::thread::sleep(Duration::from_millis(100));
        drop(guard); // the body unwound
        assert!(waiter.join().unwrap(), "the waiter runs the body after the unwind");
    }

    #[test]
    fn a_wait_cycle_across_threads_lets_the_later_importer_proceed() {
        static A: ModuleInitLock = ModuleInitLock::new();
        static B: ModuleInitLock = ModuleInitLock::new();
        // Thread 1 owns A and waits for B; thread 2 owns B and asks for A:
        // a cross-thread cycle — thread 2 proceeds (CPython's
        // _DeadlockError path), then thread 1 finds B done.
        let t1 = std::thread::spawn(|| {
            let ModuleInitEntry::Run(ga) = A.enter() else { panic!() };
            std::thread::sleep(Duration::from_millis(100));
            let b = B.enter();
            ga.finish(true);
            matches!(b, ModuleInitEntry::Done)
        });
        std::thread::sleep(Duration::from_millis(20));
        let t2 = std::thread::spawn(|| {
            let ModuleInitEntry::Run(gb) = B.enter() else { panic!() };
            std::thread::sleep(Duration::from_millis(150));
            let a = A.enter();
            gb.finish(true);
            matches!(a, ModuleInitEntry::Cycle)
        });
        assert!(t2.join().unwrap(), "the later importer proceeds with the partial module");
        assert!(t1.join().unwrap(), "the first importer finds the other module done");
    }
}
