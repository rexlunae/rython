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
    /// `threading.local()`: a per-thread bag of run-time attributes
    /// (issue #356); attribute access on it is dynamic (`py_getattr`).
    Local,
}

impl ThreadingType {
    /// The imported constructor hidden by a compiler-renamed local. This
    /// classifies a refusal, never a construction: Python calls the local
    /// value, whereas a Rust module path could accidentally use the import.
    pub(crate) fn shadowed_constructor(
        func: &crate::ExprType,
        symbols: &crate::SymbolTableScopes,
    ) -> Option<Self> {
        let mut original = func.clone();
        let name = match &mut original {
            crate::ExprType::Name(n) => n,
            crate::ExprType::Attribute(a) => match a.value.as_mut() {
                crate::ExprType::Name(n) => n,
                _ => return None,
            },
            _ => return None,
        };
        let prefix = crate::ast::tree::rename::shadowing_local_ident("");
        name.id = name.id.strip_prefix(&prefix)?.to_string();
        Self::constructor(&original, symbols)
    }

    /// Resolve a constructor through its actual import binding, including
    /// `import threading as th` and `from threading import Thread as Worker`.
    /// Inference and lowering must agree on the same constructor.
    pub(crate) fn constructor(
        func: &crate::ExprType,
        symbols: &crate::SymbolTableScopes,
    ) -> Option<Self> {
        use crate::{ExprType, StdModule, SymbolTableNode};
        match func {
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::ImportFrom(i))
                    if i.level == 0
                        && StdModule::from_name(&i.module) == Some(StdModule::Threading) =>
                {
                    Self::from_name(&i.defining_name(&n.id))
                }
                _ => None,
            },
            ExprType::Attribute(a) => {
                let ExprType::Name(n) = a.value.as_ref() else { return None };
                let module = match symbols.get(&n.id) {
                    Some(SymbolTableNode::Import(i)) => i.names.iter().any(|alias| {
                        alias.asname.as_deref().unwrap_or(&alias.name) == n.id
                            && StdModule::from_name(&alias.name) == Some(StdModule::Threading)
                    }),
                    Some(SymbolTableNode::Alias(module)) => {
                        StdModule::from_name(module) == Some(StdModule::Threading)
                    }
                    _ => false,
                };
                module.then(|| Self::from_name(&a.attr)).flatten()
            }
            _ => None,
        }
    }

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
            "local" => Some(ThreadingType::Local),
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
            ThreadingType::Local => "local",
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
            ThreadingType::Local => quote!(threading::Local),
        }
    }
}

/// The supported lifecycle methods of a typed Thread handle. One boundary
/// for method names, shared by result inference and call lowering.
#[derive(Clone, Copy)]
pub(crate) enum ThreadMethod {
    Start,
    Join,
    IsAlive,
}

impl ThreadMethod {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "start" => Some(Self::Start),
            "join" => Some(Self::Join),
            "is_alive" => Some(Self::IsAlive),
            _ => None,
        }
    }

    pub(crate) fn result_type(self) -> crate::TypeInfo {
        match self {
            Self::Start | Self::Join => crate::TypeInfo::Custom(quote!(())),
            Self::IsAlive => crate::TypeInfo::Bool,
        }
    }

    pub(crate) fn receiver_is_thread(ty: &crate::TypeInfo) -> bool {
        match ty {
            crate::TypeInfo::Threading(ThreadingType::Thread) => true,
            crate::TypeInfo::Option(inner) => Self::receiver_is_thread(inner),
            _ => false,
        }
    }
}
