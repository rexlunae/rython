//! Python time module implementation
//!
//! Wall-clock and monotonic time plus sleep, matching Python's time
//! module API for the commonly generated calls.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The process-wide monotonic origin: Python only promises that
/// monotonic()/perf_counter() differences are meaningful, so anchoring at
/// first use is conformant.
static MONOTONIC_ORIGIN: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

/// time.time(): seconds since the Unix epoch as a float.
pub fn time() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64(),
        // A pre-1970 system clock yields a negative timestamp, as in Python.
        Err(e) => -e.duration().as_secs_f64(),
    }
}

/// time.time_ns(): nanoseconds since the Unix epoch.
pub fn time_ns() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i64,
        Err(e) => -(e.duration().as_nanos() as i64),
    }
}

/// time.sleep(seconds). Negative durations raise ValueError in Python;
/// this fails loudly the same way.
pub fn sleep(seconds: f64) {
    if seconds < 0.0 {
        panic!(
            "{}",
            crate::PyException::new("ValueError", "sleep length must be non-negative")
        );
    }
    std::thread::sleep(Duration::from_secs_f64(seconds));
}

/// time.monotonic(): a clock that cannot go backwards; only differences
/// are meaningful.
pub fn monotonic() -> f64 {
    MONOTONIC_ORIGIN.elapsed().as_secs_f64()
}

/// time.perf_counter(): the highest-resolution monotonic clock available.
pub fn perf_counter() -> f64 {
    MONOTONIC_ORIGIN.elapsed().as_secs_f64()
}
