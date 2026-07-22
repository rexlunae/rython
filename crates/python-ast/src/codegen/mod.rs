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
    Class,
    Function,
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
    TryBlock { parent: Box<CodeGenContext> },
    /// Inside an `except` handler body, where the caught exception is in
    /// scope as `__rython_exc`. `parent` is the context the try statement
    /// itself appears in (handler code runs outside the try's closure).
    ExceptHandler { parent: Box<CodeGenContext> },
}

impl CodeGenContext {
    /// Whether code generated here runs inside a try-block closure, so a
    /// raised exception can `return Err(...)` to be caught by its handlers.
    pub fn in_try_block(&self) -> bool {
        match self {
            CodeGenContext::TryBlock { .. } => true,
            CodeGenContext::ExceptHandler { parent }
            | CodeGenContext::Loop { parent, .. } => parent.in_try_block(),
            CodeGenContext::Async(inner) => inner.in_try_block(),
            _ => false,
        }
    }

    /// Whether code generated here runs inside an except handler, i.e. the
    /// caught exception is in scope as `__rython_exc` (for bare `raise`).
    pub fn in_except_handler(&self) -> bool {
        match self {
            CodeGenContext::ExceptHandler { .. } => true,
            CodeGenContext::TryBlock { parent }
            | CodeGenContext::Loop { parent, .. } => parent.in_except_handler(),
            CodeGenContext::Async(inner) => inner.in_except_handler(),
            _ => false,
        }
    }

    /// The context for a nested function definition's body: exception scopes
    /// don't cross function boundaries (a `raise` in a nested function can't
    /// return out of the enclosing try's closure).
    pub fn strip_exception_scopes(self) -> CodeGenContext {
        match self {
            CodeGenContext::TryBlock { parent }
            | CodeGenContext::ExceptHandler { parent } => parent.strip_exception_scopes(),
            other => other,
        }
    }
}
