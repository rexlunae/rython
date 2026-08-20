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
}
