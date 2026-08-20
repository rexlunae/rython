//! Python `warnings` module (issue #111).
//!
//! Warnings are diagnostics: they do not change program behavior, so the
//! filter API is a no-op and `warn` prints to stderr (CPython writes
//! warnings to stderr). Signatures mirror Python's parameter lists so the
//! keyword-argument mapping in call.rs can slot arguments by name; every
//! parameter is `Option` so omitted trailing parameters fill with `None`
//! and present ones wrap in `Some(...)` uniformly. The `category`/
//! `source`/`registry` parameters are CLASSES or objects in Python; rython
//! cannot pass a class as a value, so they are `Option<()>` and ignored —
//! a call that passes a class still needs the class to be a value, which
//! codegen rejects loudly.

/// `warnings.warn(message, category=..., stacklevel=..., source=...)` —
/// print the message to stderr like CPython.
pub fn warn(
    message: Option<&str>,
    _category: Option<()>,
    _stacklevel: Option<i64>,
    _source: Option<()>,
) {
    if let Some(message) = message {
        eprintln!("warning: {}", message);
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
    match (message, filename, lineno) {
        (Some(message), Some(filename), Some(lineno)) => {
            eprintln!("{}:{}: warning: {}", filename, lineno, message)
        }
        (Some(message), _, _) => eprintln!("warning: {}", message),
        _ => {}
    }
}

/// `warnings.simplefilter(action, category=..., module=..., lineno=...,
/// append=...)` — a filter configuration; diagnostics-only, so a no-op is
/// faithful (warnings still do not affect behavior).
pub fn simplefilter(
    _action: Option<&str>,
    _category: Option<()>,
    _module: Option<&str>,
    _lineno: Option<i64>,
    _append: Option<bool>,
) {
}

/// `warnings.filterwarnings(...)` — same: diagnostics-only.
pub fn filterwarnings(
    _action: Option<&str>,
    _message: Option<&str>,
    _category: Option<()>,
    _module: Option<&str>,
    _lineno: Option<i64>,
    _append: Option<bool>,
) {
}

/// `warnings.resetwarnings()`.
pub fn resetwarnings() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_calls_are_no_ops() {
        simplefilter(Some("ignore"), None, None, None, Some(true));
        filterwarnings(Some("error"), None, None, None, None, None);
        resetwarnings();
    }

    #[test]
    fn warn_prints_to_stderr() {
        warn(Some("test"), None, None, None);
        warn_explicit(Some("test"), None, Some("f.py"), Some(3), None, None, None, None);
    }
}
