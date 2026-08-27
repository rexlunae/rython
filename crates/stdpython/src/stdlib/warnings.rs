//! Python `warnings` module (issue #111).
//!
//! The HOOKS (filter state + `warn`) are separated from the OUTPUT, in the
//! same spirit as Rust's `log` facade: `warn` consults the filter action,
//! then dispatches to an installed OUTPUT HOOK. With no hook installed the
//! hooks still run — `warn` is a no-op (like `log` with no logger
//! backend). The `std` tier installs a stderr printer by default (CPython
//! writes warnings to stderr); the alloc tier has no default output.
//! `set_warning_output` installs or replaces the hook, and
//! `reset_warning_output` restores the feature default.
//!
//! Signatures mirror Python's parameter lists so the keyword-argument
//! mapping in call.rs can slot arguments by name; every parameter is
//! `Option` so omitted trailing parameters fill with `None` and present
//! ones wrap in `Some(...)` uniformly. The `category`/`source`/`registry`
//! parameters are CLASSES or objects in Python; rython cannot pass a class
//! as a value, so they are `Option<()>` and ignored — a call that passes a
//! class still needs the class to be a value, which codegen rejects
//! loudly.

use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Filter actions (Python's warning actions, as an int for the atomic).
// ---------------------------------------------------------------------------

const ACTION_ALWAYS: usize = 0;
const ACTION_IGNORE: usize = 1;
const ACTION_ERROR: usize = 2;
const ACTION_DEFAULT: usize = 3;
const ACTION_ONCE: usize = 4;
const ACTION_MODULE: usize = 5;
const ACTION_MESSAGE: usize = 6;

/// The current default filter action (the most recent simplefilter/
/// filterwarnings wins; category/message/module filters are unrepresentable
/// because categories are classes, so the action is the whole filter).
static DEFAULT_ACTION: AtomicUsize = AtomicUsize::new(ACTION_DEFAULT);

fn action_from_str(action: &str) -> usize {
    match action {
        "always" => ACTION_ALWAYS,
        "ignore" => ACTION_IGNORE,
        "error" => ACTION_ERROR,
        "default" => ACTION_DEFAULT,
        "once" => ACTION_ONCE,
        "module" => ACTION_MODULE,
        "message" => ACTION_MESSAGE,
        _ => ACTION_DEFAULT,
    }
}

fn should_emit() -> bool {
    // "ignore" suppresses; "error" turns the warning into an exception,
    // which rython does not raise here (documented) — also suppressed so
    // the program does not diverge into printing what Python would raise.
    !matches!(
        DEFAULT_ACTION.load(Ordering::Relaxed),
        ACTION_IGNORE | ACTION_ERROR
    )
}

// ---------------------------------------------------------------------------
// Output hook (the separable OUTPUT half of the facade).
// ---------------------------------------------------------------------------

/// The output hook: (message, filename, lineno). Stored as a raw pointer
/// so a `static` can hold it; only ever written with real fn pointers via
/// `set_warning_output`, so reading it back and calling it is sound.
static OUTPUT: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Install a warning OUTPUT hook. Subsequent `warn`/`warn_explicit` calls
/// route through it; the filter hooks run regardless.
pub fn set_warning_output(hook: fn(&str, &str, i64)) {
    OUTPUT.store(hook as *const () as *mut (), Ordering::Relaxed);
}

/// Restore the feature default output (stderr on std, no-op on alloc).
pub fn reset_warning_output() {
    OUTPUT.store(ptr::null_mut(), Ordering::Relaxed);
}

/// The std default output: CPython writes warnings to stderr.
#[cfg(feature = "std")]
fn default_output(message: &str, filename: &str, lineno: i64) {
    if filename.is_empty() {
        eprintln!("warning: {}", message);
    } else {
        eprintln!("{}:{}: warning: {}", filename, lineno, message);
    }
}

fn emit(message: &str, filename: &str, lineno: i64) {
    let ptr = OUTPUT.load(Ordering::Relaxed);
    if !ptr.is_null() {
        // The pointer was only ever written by set_warning_output with a
        // real fn pointer, so reinterpreting it is sound; a raw-pointer-to-
        // fn-pointer `as` cast is not allowed, hence the transmute.
        let hook: fn(&str, &str, i64) = unsafe { core::mem::transmute(ptr) };
        hook(message, filename, lineno);
        return;
    }
    #[cfg(feature = "std")]
    default_output(message, filename, lineno);
    #[cfg(not(feature = "std"))]
    let _ = (message, filename, lineno); // no output installed: no-op
}

// ---------------------------------------------------------------------------
// The warnings API (the HOOK half).
// ---------------------------------------------------------------------------

/// `warnings.warn(message, category=..., stacklevel=..., source=...)` —
/// run the filter hook, then dispatch to the output hook.
pub fn warn(
    message: Option<&str>,
    _category: Option<()>,
    _stacklevel: Option<i64>,
    _source: Option<()>,
) {
    if !should_emit() {
        return;
    }
    if let Some(message) = message {
        emit(message, "", 0);
    }
}

/// `warnings.warn_explicit(message, category, filename, lineno, ...)`.
pub fn warn_explicit(
    message: Option<&str>,
    _category: Option<()>,
    filename: Option<&str>,
    lineno: Option<i64>,
    _module: Option<&str>,
    _registry: Option<()>,
    _module_globals: Option<()>,
    _source: Option<()>,
) {
    if !should_emit() {
        return;
    }
    match (message, filename, lineno) {
        (Some(message), Some(filename), Some(lineno)) => emit(message, filename, lineno),
        (Some(message), _, _) => emit(message, "", 0),
        _ => {}
    }
}

/// `warnings.simplefilter(action, category=..., module=..., lineno=...,
/// append=...)` — set the default filter action (the hook layer).
pub fn simplefilter(
    action: Option<&str>,
    _category: Option<()>,
    _module: Option<&str>,
    _lineno: Option<i64>,
    _append: Option<bool>,
) {
    if let Some(action) = action {
        DEFAULT_ACTION.store(action_from_str(action), Ordering::Relaxed);
    }
}

/// `warnings.filterwarnings(...)` — same: sets the default action.
pub fn filterwarnings(
    action: Option<&str>,
    _message: Option<&str>,
    _category: Option<()>,
    _module: Option<&str>,
    _lineno: Option<i64>,
    _append: Option<bool>,
) {
    if let Some(action) = action {
        DEFAULT_ACTION.store(action_from_str(action), Ordering::Relaxed);
    }
}

/// `warnings.resetwarnings()` — restore the default action and output.
pub fn resetwarnings() {
    DEFAULT_ACTION.store(ACTION_DEFAULT, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    // The output hook and DEFAULT_ACTION are PROCESS-GLOBAL, and every
    // test here mutates them: the parallel test harness races them (a
    // sibling's reset_warning_output mid-warn reads as CAPTURED == 0).
    // They serialize on this core-only spin guard — std::sync::Mutex
    // would fail the no_std --all-targets build.
    use core::sync::atomic::AtomicBool;
    static SERIAL: AtomicBool = AtomicBool::new(false);
    struct SerialGuard;
    fn serial() -> SerialGuard {
        while SERIAL.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
        SerialGuard
    }
    impl Drop for SerialGuard {
        fn drop(&mut self) {
            SERIAL.store(false, Ordering::Release);
        }
    }

    #[test]
    fn hooks_run_without_an_output_hook() {
        let _serial = serial();
        // The hooks work with no output installed: warn is a no-op, and
        // the filter action still suppresses.
        reset_warning_output();
        DEFAULT_ACTION.store(ACTION_DEFAULT, Ordering::Relaxed);
        warn(Some("hi"), None, None, None); // no-op, no panic

        simplefilter(Some("ignore"), None, None, None, None);
        warn(Some("suppressed"), None, None, None);
        DEFAULT_ACTION.store(ACTION_DEFAULT, Ordering::Relaxed);
    }

    #[test]
    fn warn_routes_through_the_output_hook() {
        let _serial = serial();
        use core::sync::atomic::AtomicUsize;
        static CAPTURED: AtomicUsize = AtomicUsize::new(0);
        fn capture(message: &str, _file: &str, _line: i64) {
            CAPTURED.store(message.len(), Ordering::Relaxed);
        }
        set_warning_output(capture);
        DEFAULT_ACTION.store(ACTION_DEFAULT, Ordering::Relaxed);
        warn(Some("hello"), None, None, None);
        assert_eq!(CAPTURED.load(Ordering::Relaxed), 5);
        reset_warning_output();
    }

    #[test]
    fn ignore_action_suppresses_the_hook() {
        let _serial = serial();
        use core::sync::atomic::AtomicUsize;
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn count(_m: &str, _f: &str, _l: i64) {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        set_warning_output(count);
        DEFAULT_ACTION.store(ACTION_IGNORE, Ordering::Relaxed);
        warn(Some("x"), None, None, None);
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);
        reset_warning_output();
        DEFAULT_ACTION.store(ACTION_DEFAULT, Ordering::Relaxed);
    }

    #[test]
    fn filter_calls_set_the_action() {
        let _serial = serial();
        simplefilter(Some("ignore"), None, None, None, Some(true));
        assert_eq!(DEFAULT_ACTION.load(Ordering::Relaxed), ACTION_IGNORE);
        filterwarnings(Some("always"), None, None, None, None, None);
        assert_eq!(DEFAULT_ACTION.load(Ordering::Relaxed), ACTION_ALWAYS);
        resetwarnings();
        assert_eq!(DEFAULT_ACTION.load(Ordering::Relaxed), ACTION_DEFAULT);
    }
}
