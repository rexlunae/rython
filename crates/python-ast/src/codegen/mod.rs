//! Code generation for Python ASTs.

use std::fmt::Debug;

pub mod python_options;
pub use python_options::*;

/// Reexport the CodeGen from to_tokenstream
pub use to_tokenstream::CodeGen;

/// A type to track the context of code generation.
#[derive(Clone, Debug)]
pub enum CodeGenContext {
    Module(String),
    /// Inside a class body; the payload is the class name, so method
    /// lowering can resolve `self` (receiver typing, `self.method()` calls)
    /// against the class's definition in the symbol table.
    Class(String),
    /// Inside a method body being emitted as a trait item (trait-based
    /// inheritance). `class` is the class whose struct `self` is; `generic`
    /// is true for trait DEFAULT bodies (where `Self` is the trait's generic
    /// implementor, so field access goes through generated accessors) and
    /// false for override bodies in a concrete `impl Trait for Struct`
    /// (where `self` is the concrete struct, so field access is direct).
    Trait {
        class: String,
        generic: bool,
        /// When an override originally defined by another class is re-emitted
        /// for a derived class, `super()` inside it must resolve to the
        /// ORIGINAL definer's base (not the derived class's base). `None`
        /// when the method belongs to the enclosing class itself.
        super_target: Option<String>,
        /// Force `&mut self` on the emitted method signature even when the
        /// method's own body does not mutate: the owning trait's signature
        /// widened because SOME definition in the hierarchy mutates, so
        /// every impl (and every default) must match.
        force_mut_self: bool,
    },
    /// Inside a function body. Carries the enclosing class name (if any) so
    /// `self` still resolves in method bodies — the codegen context otherwise
    /// loses the Class marker at the function boundary, and function bodies
    /// must not be confused with module scope (module-level-only constructs
    /// like rust.bind declarations are rejected there).
    Function {
        class: Option<String>,
    },
    Async(Box<CodeGenContext>),
    /// Directly inside a loop body. `has_else` is true when the loop carries
    /// a Python `else` clause, in which case `break` statements must also set
    /// the `__rython_broke` flag the loop lowering declares.
    Loop {
        has_else: bool,
        parent: Box<CodeGenContext>,
    },
    /// Inside a `try` block body, which lowers to a closure returning
    /// `Result<(), PyException>`; `raise` (and failed `assert`) here lower
    /// to `return Err(...)` so the except handlers can catch them.
    TryBlock {
        parent: Box<CodeGenContext>,
    },
    /// Inside an `except` handler body, where the caught exception is in
    /// scope as `__rython_exc`. `parent` is the context the try statement
    /// itself appears in (handler code runs outside the try's closure).
    ExceptHandler {
        parent: Box<CodeGenContext>,
    },
}

impl CodeGenContext {
    /// Whether code generated here runs inside a try-block closure, so a
    /// raised exception can `return Err(...)` to be caught by its handlers.
    pub fn in_try_block(&self) -> bool {
        match self {
            CodeGenContext::TryBlock { .. } => true,
            CodeGenContext::ExceptHandler { parent } | CodeGenContext::Loop { parent, .. } => {
                parent.in_try_block()
            }
            CodeGenContext::Async(inner) => inner.in_try_block(),
            _ => false,
        }
    }

    /// Whether a `break`/`continue` generated here would have to cross a
    /// try-block closure boundary to reach its loop. Walking outward, the
    /// first `Loop` means the statement binds to a real Rust loop; hitting
    /// `TryBlock` first means the loop is outside the closure, so the
    /// statement must be threaded out as a `PyFlow` signal instead.
    pub fn break_crosses_try_closure(&self) -> bool {
        match self {
            CodeGenContext::Loop { .. } => false,
            CodeGenContext::TryBlock { .. } => true,
            CodeGenContext::ExceptHandler { parent } => parent.break_crosses_try_closure(),
            CodeGenContext::Async(inner) => inner.break_crosses_try_closure(),
            _ => false,
        }
    }

    /// Whether the loop a `break` here targets carries a Python `else`
    /// clause, so the break must also set the loop's `__rython_broke` flag.
    pub fn break_target_has_else(&self) -> bool {
        match self {
            CodeGenContext::Loop { has_else, .. } => *has_else,
            CodeGenContext::TryBlock { parent } | CodeGenContext::ExceptHandler { parent } => {
                parent.break_target_has_else()
            }
            CodeGenContext::Async(inner) => inner.break_target_has_else(),
            _ => false,
        }
    }

    /// Whether code generated here runs inside an except handler, i.e. the
    /// caught exception is in scope as `__rython_exc` (for bare `raise`).
    pub fn in_except_handler(&self) -> bool {
        match self {
            CodeGenContext::ExceptHandler { .. } => true,
            CodeGenContext::TryBlock { parent } | CodeGenContext::Loop { parent, .. } => {
                parent.in_except_handler()
            }
            CodeGenContext::Async(inner) => inner.in_except_handler(),
            _ => false,
        }
    }

    /// The context for a nested function definition's body: exception scopes
    /// don't cross function boundaries (a `raise` in a nested function can't
    /// return out of the enclosing try's closure).
    pub fn strip_exception_scopes(self) -> CodeGenContext {
        match self {
            CodeGenContext::TryBlock { parent } | CodeGenContext::ExceptHandler { parent } => {
                parent.strip_exception_scopes()
            }
            other => other,
        }
    }

    /// True inside a FUNCTION body (module and class level are false): a
    /// nested function definition there is a CLOSURE in Python — rython's
    /// closures do not capture the enclosing scope (the closure-capture
    /// divergence), so the definition lowers to a no-op. Trait-method
    /// bodies (incl. re-emitted overrides) are function bodies too.
    pub fn is_function_body(&self) -> bool {
        match self {
            CodeGenContext::Function { .. } | CodeGenContext::Trait { .. } => true,
            CodeGenContext::Async(inner) => inner.is_function_body(),
            CodeGenContext::Loop { parent, .. }
            | CodeGenContext::TryBlock { parent }
            | CodeGenContext::ExceptHandler { parent } => parent.is_function_body(),
            _ => false,
        }
    }

    /// The name of the class whose method body this context sits inside, if
    /// any — the class `self` refers to.
    pub fn enclosing_class_name(&self) -> Option<&str> {
        match self {
            CodeGenContext::Class(name) => Some(name),
            CodeGenContext::Trait { class, .. } => Some(class),
            CodeGenContext::Function { class } => class.as_deref(),
            CodeGenContext::Async(inner) => inner.enclosing_class_name(),
            CodeGenContext::Loop { parent, .. }
            | CodeGenContext::TryBlock { parent }
            | CodeGenContext::ExceptHandler { parent } => parent.enclosing_class_name(),
            _ => None,
        }
    }

    /// Whether this context sits inside a trait DEFAULT body, where `self`
    /// is the trait's generic implementor and field access must go through
    /// the generated accessors (`self.f()`, `self.base().f`).
    pub fn in_generic_trait(&self) -> bool {
        match self {
            CodeGenContext::Trait { generic, .. } => *generic,
            CodeGenContext::Async(inner) => inner.in_generic_trait(),
            CodeGenContext::Loop { parent, .. }
            | CodeGenContext::TryBlock { parent }
            | CodeGenContext::ExceptHandler { parent } => parent.in_generic_trait(),
            _ => false,
        }
    }
}
