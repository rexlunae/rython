//! Timezone-sensitive timestamp tests, in their OWN test binary.
//!
//! Python's naive datetime.timestamp() uses LOCAL time, so these
//! assertions depend on the process timezone and must pin TZ=UTC. That
//! mutation cannot live in the shared python_semantics binary: cargo
//! runs a binary's tests as parallel THREADS, and setenv concurrent
//! with any other thread's getenv (std::env::temp_dir, libc's own TZ
//! reads) is a data race on glibc — the reason set_var is unsafe.
//! Here the mutation runs exactly once, via Once, before either test
//! reaches libc's time functions, and no other test shares the
//! process.

use stdpython::datetime::datetime;

static PIN_UTC: std::sync::Once = std::sync::Once::new();

fn pin_utc() {
    // POSIX tzset(), so localtime_r/mktime re-read TZ; the libc crate
    // does not bind it.
    #[cfg(unix)]
    unsafe extern "C" {
        fn tzset();
    }
    PIN_UTC.call_once(|| unsafe {
        std::env::set_var("TZ", "UTC");
        #[cfg(unix)]
        tzset();
    });
}

#[test]
fn datetime_timestamps_round_trip_and_handle_pre_epoch() {
    pin_utc();
    // fromtimestamp(-1) is 1969-12-31 23:59:59 — the old code wrapped
    // negatives into a panic via Duration::from_secs_f64.
    let d = datetime::fromtimestamp(-1.0).unwrap();
    assert_eq!(d.date_component().year, 1969);
    assert_eq!(d.time_component().second, 59);
    // python3 under TZ=UTC:
    // datetime(2026, 7, 23, 12, 0).timestamp() == 1784808000.0
    let d = datetime::new(2026, 7, 23, Some(12), Some(0), Some(0), None).unwrap();
    assert_eq!(d.timestamp(), 1784808000.0);
    // Round trip.
    let back = datetime::fromtimestamp(1784808000.0).unwrap();
    assert_eq!(back.date_component().day, 23);
    assert_eq!(back.time_component().hour, 12);
}

#[test]
fn timestamp_one_second_before_epoch_is_minus_one() {
    pin_utc();
    // mktime returns -1 BOTH as its error value and as the valid result
    // for this exact moment; the disambiguation must return -1.0 here
    // (python3 under TZ=UTC: datetime(1969, 12, 31, 23, 59, 59)
    // .timestamp() == -1.0).
    let d = datetime::new(1969, 12, 31, Some(23), Some(59), Some(59), None).unwrap();
    assert_eq!(d.timestamp(), -1.0);
}
