//! The threading-module runtime types the compiler knows, as a typed enum.
//!
//! Python attribute names arrive from CPython's parser as strings, so ONE
//! string comparison at the AST boundary is unavoidable — but it happens
//! exactly once, in [`ThreadingType::from_name`]. Every consumer (the
//! annotation mapping, the `with lock:` classifier, the `threading.Thread`
//! and `threading.Semaphore` call lowerings, the import item registry)
//! works with the enum, so the set of known types has a single source of
//! truth instead of parallel string lists that can drift.

use proc_macro2::TokenStream;
use quote::quote;

/// A type of the stdpython `threading` runtime module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadingType {
    Thread,
    Lock,
    RLock,
    Event,
    Semaphore,
}

impl ThreadingType {
    /// Parse a Python identifier (an attribute or imported name) at the
    /// AST boundary. The caller is responsible for having established that
    /// the name resolves against the `threading` module.
    pub(crate) fn from_name(name: &str) -> Option<ThreadingType> {
        match name {
            "Thread" => Some(ThreadingType::Thread),
            "Lock" => Some(ThreadingType::Lock),
            "RLock" => Some(ThreadingType::RLock),
            "Event" => Some(ThreadingType::Event),
            "Semaphore" => Some(ThreadingType::Semaphore),
            _ => None,
        }
    }

    /// Whether `with obj:` over this type must lower to the runtime's
    /// acquire/release RAII guard (`py_guard()`). Thread and Event are not
    /// context managers in Python.
    pub(crate) fn is_sync_guard(self) -> bool {
        matches!(
            self,
            ThreadingType::Lock | ThreadingType::RLock | ThreadingType::Semaphore
        )
    }

    /// The type's Python name — the enum is the authority even where a
    /// string is ultimately stored (the local_types annotation records).
    pub(crate) fn name(self) -> &'static str {
        match self {
            ThreadingType::Thread => "Thread",
            ThreadingType::Lock => "Lock",
            ThreadingType::RLock => "RLock",
            ThreadingType::Event => "Event",
            ThreadingType::Semaphore => "Semaphore",
        }
    }

    /// The type's path in the stdpython runtime, for annotations.
    pub(crate) fn rust_path(self) -> TokenStream {
        match self {
            ThreadingType::Thread => quote!(threading::Thread),
            ThreadingType::Lock => quote!(threading::Lock),
            ThreadingType::RLock => quote!(threading::RLock),
            ThreadingType::Event => quote!(threading::Event),
            ThreadingType::Semaphore => quote!(threading::Semaphore),
        }
    }
}
