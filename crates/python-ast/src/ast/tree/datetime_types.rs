//! The datetime-module runtime types the compiler knows, as a typed enum.
//!
//! Python attribute names arrive from CPython's parser as strings, so ONE
//! string comparison at the AST boundary is unavoidable — but it happens
//! exactly once, in [`DatetimeType::from_name`]. Every consumer (the
//! import item registry, the constructor lowering) works with the enum,
//! so the set of known types has a single source of truth instead of
//! parallel string lists that can drift (the ThreadingType precedent).

/// A class of the stdpython `datetime` runtime module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatetimeType {
    Date,
    DateTime,
    Time,
    Timedelta,
    Timezone,
}

impl DatetimeType {
    /// Parse a Python identifier (an attribute or imported name) at the
    /// AST boundary. The caller is responsible for having established that
    /// the name resolves against the `datetime` module.
    pub(crate) fn from_name(name: &str) -> Option<DatetimeType> {
        Some(match name {
            "date" => DatetimeType::Date,
            "datetime" => DatetimeType::DateTime,
            "time" => DatetimeType::Time,
            "timedelta" => DatetimeType::Timedelta,
            "timezone" => DatetimeType::Timezone,
            _ => return None,
        })
    }
}
