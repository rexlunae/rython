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
//! unwinds settles the module (NOT STARTED, or FAILED when a re-run could
//! not be faithful) and wakes the waiters.

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

/// The running body's guard: `finish(outcome)` marks the module DONE on
/// `Ok`; on `Err` it is NOT STARTED again when the module can retry (the
/// next import runs the body afresh, as CPython does after dropping a
/// failed import) and FAILED otherwise. A guard dropped without `finish`
/// — the body unwound — settles the same way (FAILED with an ImportError
/// naming the reason when it cannot retry), and wakes the waiters either
/// way.
pub struct ModuleInitGuard {
    lock: &'static ModuleInitLock,
    retry: bool,
    finished: bool,
}

/// The exception a module that cannot retry raises after its body
/// unwound (a panic, not a Python exception, so there is none to keep).
fn unwound_error() -> crate::PyException {
    crate::PyException::new(
        "ImportError",
        "the module body panicked on its first import after initializing a module \
         value; it cannot run again",
    )
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

    /// Enter the module's body: run it, skip it, or wait for it. `retry`
    /// is the module's policy after a failed body: a module with no
    /// static state of its own runs again on the next import (CPython
    /// drops a failed import); one holding a static the process cannot
    /// re-initialize (a promoted value, a mutable global) stays FAILED,
    /// and every later import raises again — loud, where a re-run would
    /// see the first attempt's values.
    pub fn enter(&'static self, retry: bool) -> ModuleInitEntry {
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
                        retry,
                        finished: false,
                    });
                }
            }
        }
    }

    /// Whether a thread is blocked waiting for this module's body (the
    /// tests' deterministic "the waiter is waiting" signal).
    #[cfg(test)]
    fn is_waited_on(&'static self) -> bool {
        let key = self.key();
        graph()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .waiting_on
            .values()
            .any(|k| *k == key)
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
    /// The body finished: DONE on `Ok`; on `Err`, NOT STARTED when the
    /// module retries (the next import runs the body again) and FAILED
    /// with that exception otherwise (every later import raises it
    /// again).
    pub fn finish(mut self, outcome: Result<(), crate::PyException>) {
        self.finished = true;
        self.lock.settle(match outcome {
            Ok(()) => State::Done,
            Err(_) if self.retry => State::NotStarted,
            Err(e) => State::Failed(e),
        });
    }
}

impl Drop for ModuleInitGuard {
    fn drop(&mut self) {
        if !self.finished {
            // The body unwound: the retry policy decides, as for a raise.
            self.lock.settle(if self.retry {
                State::NotStarted
            } else {
                State::Failed(unwound_error())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;

    /// Block until a thread is waiting for `lock`'s body: the tests'
    /// ordering signal, a state the lock records rather than a delay.
    fn until_waited_on(lock: &'static ModuleInitLock) {
        while !lock.is_waited_on() {
            std::thread::yield_now();
        }
    }

    #[test]
    fn a_second_thread_waits_for_the_body_and_then_finds_it_done() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let ModuleInitEntry::Run(guard) = LOCK.enter(true) else {
            panic!("first entry runs");
        };
        let ran2 = ran.clone();
        let waiter = std::thread::spawn(move || {
            let entry = LOCK.enter(true);
            // The body has finished by the time the waiter is released.
            assert_eq!(ran2.load(Ordering::SeqCst), 1);
            assert!(matches!(entry, ModuleInitEntry::Done));
        });
        until_waited_on(&LOCK);
        ran.store(1, Ordering::SeqCst);
        guard.finish(Ok(()));
        waiter.join().unwrap();
    }

    #[test]
    fn the_owning_thread_re_entering_is_a_cycle_and_a_failure_resets() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter(true) else {
            panic!("first entry runs");
        };
        assert!(matches!(LOCK.enter(true), ModuleInitEntry::Cycle));
        guard.finish(Err(crate::PyException::new("RuntimeError", "first attempt fails")));
        // The failed body is NOT STARTED again: the next import runs it.
        let ModuleInitEntry::Run(guard) = LOCK.enter(true) else {
            panic!("a failed module runs again");
        };
        guard.finish(Ok(()));
        assert!(matches!(LOCK.enter(true), ModuleInitEntry::Done));
    }

    #[test]
    fn a_module_that_cannot_retry_raises_the_same_error_on_every_later_import() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter(false) else {
            panic!("first entry runs");
        };
        guard.finish(Err(crate::PyException::new("RuntimeError", "first attempt fails")));
        for _ in 0..2 {
            let ModuleInitEntry::Failed(e) = LOCK.enter(false) else {
                panic!("a failed module that cannot retry never runs again");
            };
            assert_eq!(e.message, "first attempt fails");
        }
    }

    #[test]
    fn a_dropped_guard_resets_the_module_and_wakes_the_waiters() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter(true) else {
            panic!("first entry runs");
        };
        let waiter =
            std::thread::spawn(move || matches!(LOCK.enter(true), ModuleInitEntry::Run(_)));
        until_waited_on(&LOCK);
        drop(guard); // the body unwound
        assert!(waiter.join().unwrap(), "the waiter runs the body after the unwind");
    }

    #[test]
    fn a_dropped_guard_of_a_module_that_cannot_retry_leaves_it_failed() {
        static LOCK: ModuleInitLock = ModuleInitLock::new();
        let ModuleInitEntry::Run(guard) = LOCK.enter(false) else {
            panic!("first entry runs");
        };
        let waiter = std::thread::spawn(move || match LOCK.enter(false) {
            ModuleInitEntry::Failed(e) => e.message,
            _ => panic!("the waiter finds the module failed"),
        });
        until_waited_on(&LOCK);
        drop(guard); // the body unwound after initializing a static
        assert!(waiter.join().unwrap().contains("cannot run again"));
        assert!(matches!(LOCK.enter(false), ModuleInitEntry::Failed(_)));
    }

    #[test]
    fn a_wait_cycle_across_threads_lets_the_later_importer_proceed() {
        static A: ModuleInitLock = ModuleInitLock::new();
        static B: ModuleInitLock = ModuleInitLock::new();
        // Thread 1 owns A and waits for B; thread 2 owns B and asks for A:
        // a cross-thread cycle — thread 2 proceeds (CPython's
        // _DeadlockError path), then thread 1 finds B done. Each step
        // waits for the state the previous one establishes.
        let (a_running, a_running_seen) = mpsc::channel();
        let (go_wait_for_b, go_wait_for_b_seen) = mpsc::channel();
        let t1 = std::thread::spawn(move || {
            let ModuleInitEntry::Run(ga) = A.enter(true) else { panic!() };
            a_running.send(()).unwrap();
            go_wait_for_b_seen.recv().unwrap();
            let b = B.enter(true);
            ga.finish(Ok(()));
            matches!(b, ModuleInitEntry::Done)
        });
        a_running_seen.recv().unwrap();
        let (b_running, b_running_seen) = mpsc::channel();
        let (go_ask_for_a, go_ask_for_a_seen) = mpsc::channel();
        let t2 = std::thread::spawn(move || {
            let ModuleInitEntry::Run(gb) = B.enter(true) else { panic!() };
            b_running.send(()).unwrap();
            go_ask_for_a_seen.recv().unwrap();
            let a = A.enter(true);
            gb.finish(Ok(()));
            matches!(a, ModuleInitEntry::Cycle)
        });
        b_running_seen.recv().unwrap();
        go_wait_for_b.send(()).unwrap();
        until_waited_on(&B);
        go_ask_for_a.send(()).unwrap();
        assert!(t2.join().unwrap(), "the later importer proceeds with the partial module");
        assert!(t1.join().unwrap(), "the first importer finds the other module done");
    }
}
