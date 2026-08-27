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

/// time.ctime(): the current time as a platform string, like Python's
/// (requests' digest auth seeds the nonce with it).
pub fn ctime() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = now.div_euclid(86400);
    let secs = now.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    const WD: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MO: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    // The weekday of the epoch day (1970-01-01) was Thursday.
    let wd = WD[days.rem_euclid(7) as usize];
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {}",
        wd,
        MO[(month - 1) as usize],
        day,
        hour,
        minute,
        second,
        year
    )
}

/// Days-to-civil-date (Howard Hinnant's algorithm), for time.ctime().
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}
