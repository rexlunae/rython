//! Python asyncio module — a thin mapping onto the tokio runtime.
//!
//! rython programs with async code are driven by the runtime the generated
//! BINARY crate links (tokio behind the `async-tokio` feature); the entry
//! point already runs inside the runtime, so `asyncio.run(coro)` drives a
//! coroutine by awaiting it directly (Python's create-a-fresh-loop
//! semantics collapse to "run on the current loop"), and `asyncio.sleep`
//! suspends the current task on tokio's timer.
//!
//! Compiling this module requires the `async-tokio` feature. The rest of
//! the asyncio surface (gather, create_task, queues, ...) is not modeled:
//! the transpiler rejects those calls loudly rather than approximating
//! them.

use core::future::Future;

/// asyncio.run(coro): drive a coroutine to completion on the current
/// runtime. The transpiler lowers every call site to `.await?` (or plain
/// `.await` inside the entry main), so the coroutine's own Result unwraps
/// exactly like any other awaited async call.
pub async fn run<F: Future>(coro: F) -> <F as Future>::Output {
    coro.await
}

/// asyncio.run(coro) called from SYNCHRONOUS code: CPython creates a fresh
/// event loop and blocks until the coroutine finishes (issue #463).
///
/// rython's async programs already run under the tokio runtime, and a
/// blocking wait INSIDE a running runtime panics, so this picks the right
/// strategy at run time:
///
/// - on a runtime thread, it hands the future to a throwaway blocking
///   worker thread with its own current-thread runtime, and joins it (the
///   caller blocks, the nested runtime is never entered from inside an
///   async context, and no `Handle::block_on` panic can occur);
/// - off any runtime, it builds a runtime and blocks on the future
///   directly, which is what CPython does with a fresh loop.
///
/// The future must be `Send + 'static` because it crosses the thread
/// boundary; generated calls satisfy that (they move owned arguments).
pub fn run_blocking<F>(coro: F) -> <F as Future>::Output
where
    F: Future + Send + 'static,
    <F as Future>::Output: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        // Already inside a runtime: build the future on a worker thread
        // that owns its own runtime, then join.
        Ok(_) => std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("asyncio.run: building a blocking runtime")
                        .block_on(coro)
                })
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        }),
        // No runtime: this IS the thread that should block.
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("asyncio.run: building a runtime")
            .block_on(coro),
    }
}

/// asyncio.sleep(secs): suspend the current task for `secs` seconds
/// (float, like Python).
pub async fn sleep(secs: f64) {
    tokio::time::sleep(std::time::Duration::from_secs_f64(secs)).await;
}

/// asyncio.current_task() is not modeled; a loud conversion error names
/// the call. (Kept as documentation of the boundary rather than a stub.)
#[cfg(test)]
mod tests {
    // asyncio::run and sleep are exercised end-to-end by the rypip async
    // integration tests; a direct unit test needs a running runtime.
    use super::*;

    #[tokio::test]
    async fn run_awaits_the_coroutine() {
        async fn double(x: i64) -> i64 {
            x * 2
        }
        assert_eq!(run(double(21)).await, 42);
    }

    #[tokio::test]
    async fn sleep_suspends() {
        sleep(0.001).await;
    }

    // `run_blocking` has TWO branches, and the nested one is the tricky
    // one: a plain `block_on` inside a running runtime panics, which is why
    // the runtime-thread case hands the future to a worker thread with its
    // own current-thread runtime. Both branches need a pin (Devin review on
    // #466).
    #[test]
    fn run_blocking_off_any_runtime_builds_one() {
        // No runtime on this thread: the direct branch.
        let out = run_blocking(async { 7i64 });
        assert_eq!(out, 7);
    }

    #[test]
    fn run_blocking_inside_a_runtime_uses_the_worker_thread() {
        // On a runtime thread: the worker-thread branch. Reaching this
        // without a panic IS the assertion — a `block_on` in the running
        // context would panic with "Cannot start a runtime from within a
        // runtime".
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let got = rt.block_on(async { run_blocking(async { 21i64 * 2 }) });
        assert_eq!(got, 42);
    }

    #[test]
    fn run_blocking_propagates_the_coroutine_result() {
        // The coroutine's own Output crosses the boundary unchanged --
        // including a Result, which is what generated `asyncio.run(...)?`
        // sites rely on.
        let ok: Result<i64, crate::PyException> = run_blocking(async { Ok(5i64) });
        assert_eq!(ok.expect("no error"), 5);
        let err: Result<i64, crate::PyException> =
            run_blocking(async { Err(crate::runtime_error("boom")) });
        assert!(err.is_err(), "the error must propagate, not be swallowed");
    }

    #[test]
    fn run_blocking_runs_the_coroutine_exactly_once() {
        // BOTH branches must run the body exactly once (no double-poll, no
        // drop). Each branch is exercised in its own condition, because the
        // branch is chosen at run time from `Handle::try_current()` (Devin
        // review on #466).
        use std::sync::atomic::{AtomicUsize, Ordering};
        static OFF_RUNTIME: AtomicUsize = AtomicUsize::new(0);
        static ON_RUNTIME: AtomicUsize = AtomicUsize::new(0);

        // (a) off any runtime -> the direct branch.
        let _ = run_blocking(async {
            OFF_RUNTIME.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(OFF_RUNTIME.load(Ordering::SeqCst), 1, "direct branch");

        // (b) ON a runtime -> the worker-thread branch.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        rt.block_on(async {
            let _ = run_blocking(async {
                ON_RUNTIME.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert_eq!(ON_RUNTIME.load(Ordering::SeqCst), 1, "worker-thread branch");
    }
}
