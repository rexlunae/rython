use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableNode,
    SymbolTableScopes,
};

/// Raise statement (raise [exception [from cause]])
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Raise {
    /// The exception to raise (optional - bare raise re-raises current exception)
    pub exc: Option<ExprType>,
    /// The cause of the exception (optional - used with 'from' clause)
    pub cause: Option<ExprType>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Raise {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract exc (optional)
        let exc: Option<ExprType> = if let Ok(exc_attr) = ob.getattr("exc") {
            if exc_attr.is_none() {
                None
            } else {
                Some(exc_attr.extract()?)
            }
        } else {
            None
        };
        
        // Extract cause (optional)
        let cause: Option<ExprType> = if let Ok(cause_attr) = ob.getattr("cause") {
            if cause_attr.is_none() {
                None
            } else {
                Some(cause_attr.extract()?)
            }
        } else {
            None
        };
        
        Ok(Raise {
            exc,
            cause,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for Raise {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for Raise {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = if let Some(exc) = self.exc {
            exc.find_symbols(symbols)
        } else {
            symbols
        };
        
        if let Some(cause) = self.cause {
            cause.find_symbols(symbols)
        } else {
            symbols
        }
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let exc_tokens = match self.exc {
            Some(exc) => {
                let mut tokens =
                    exception_value(&exc, ctx.clone(), options.clone(), symbols.clone())?;
                if let Some(cause) = self.cause {
                    // `raise X from Y`: keep the cause visible in the message
                    // rather than dropping it.
                    let cause_tokens = cause.to_rust(ctx.clone(), options, symbols)?;
                    tokens = quote! {
                        {
                            let mut __rython_raised = #tokens;
                            __rython_raised.message =
                                format!("{} [from {}]", __rython_raised.message, #cause_tokens);
                            __rython_raised
                        }
                    };
                }
                tokens
            }
            None => {
                // Bare `raise` re-raises the exception the enclosing except
                // handler caught. Outside a handler (`_onerror_reraise` —
                // pip's misc, an rmtree onerror callback) the active
                // exception is unknown: a generic re-raise (documented
                // divergence).
                if !ctx.in_except_handler() {
                    options.definition_warnings.borrow_mut().push(
                        "bare `raise` outside an except handler re-raises a generic \
                         exception (the active exception is unmodeled)"
                            .to_string(),
                    );
                    quote!(PyException::new("RuntimeError", "bare raise"))
                } else {
                    quote!(__rython_exc.clone())
                }
            }
        };

        // Functions return Result<T, PyException>, so raising is returning
        // Err: inside a try block it returns out of the block's Result
        // closure to be caught by the handlers, and anywhere else it
        // propagates out of the function, as in Python.
        Ok(quote!(return Err(#exc_tokens)))
    }
}

/// Names that look like Python exception classes, so `raise Name` /
/// `raise Name(...)` can construct a PyException carrying that class name.
/// Anything else is treated as an expression already producing a
/// PyException value (e.g. a variable bound by `except ... as e`).
/// Every BUILTIN exception class name the compiler recognizes — ONE
/// list, the compile-time mirror of stdpython's runtime hierarchy
/// (`direct_exception_parent`). Consumers: `raise` lowering (below),
/// class_def's exception-class detection, type_ctx's annotation typing,
/// and the PyValue boxability checks — previously several of those kept
/// their own partial copies, which had drifted (KeyboardInterrupt bases
/// unrecognized by class_def; the File*Error family missing from the
/// boxability lists).
pub const BUILTIN_EXCEPTION_NAMES: &[&str] = &[
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BaseExceptionGroup",
    "BufferError",
    "DeprecationWarning",
    "EOFError",
    "EnvironmentError",
    "Exception",
    "ExceptionGroup",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "PermissionError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
];

/// Whether a name is a BUILTIN exception class — the fixed set only, no
/// naming heuristic (annotation typing and boxability must not absorb
/// user classes that merely end in "Error").
pub fn is_builtin_exception_name(name: &str) -> bool {
    BUILTIN_EXCEPTION_NAMES.contains(&name)
}

/// CPython's stdlib exception ALIASES: `socket.timeout` IS TimeoutError
/// and the socket error family IS OSError (Python ≥3.10 aliases the
/// class objects), so raises and except-clauses through these names
/// canonicalize to the builtin the runtime's hierarchy walk knows
/// (urllib3's pyopenssl `raise timeout(...)` under `from socket import
/// timeout` — issue #137).
pub(crate) fn stdlib_exception_canonical(module: &str, name: &str) -> Option<&'static str> {
    match (module, name) {
        ("socket", "timeout") => Some("TimeoutError"),
        ("socket", "error" | "gaierror" | "herror") => Some("OSError"),
        // The ssl exception family is in stdpython's runtime hierarchy
        // under its own names (SSLError IS-A OSError), so the canonical
        // form is the bare name; CertificateError is CPython's alias of
        // SSLCertVerificationError. Verified against python3.
        ("ssl", "SSLError") => Some("SSLError"),
        ("ssl", "SSLZeroReturnError") => Some("SSLZeroReturnError"),
        ("ssl", "SSLWantReadError") => Some("SSLWantReadError"),
        ("ssl", "SSLWantWriteError") => Some("SSLWantWriteError"),
        ("ssl", "SSLSyscallError") => Some("SSLSyscallError"),
        ("ssl", "SSLEOFError") => Some("SSLEOFError"),
        ("ssl", "SSLCertVerificationError" | "CertificateError") => {
            Some("SSLCertVerificationError")
        }
        _ => None,
    }
}

/// A canonical stdlib-MODULE exception name (the terminal of an alias
/// chain that never passes through an ImportFrom — urllib3's
/// `BaseSSLError = ssl.SSLError` registers `Alias("SSLError")`, and
/// "SSLError" itself has no symbol entry). The runtime hierarchy knows
/// these names directly.
pub(crate) fn stdlib_module_exception_name(name: &str) -> Option<&'static str> {
    match name {
        "SSLError" => Some("SSLError"),
        "SSLZeroReturnError" => Some("SSLZeroReturnError"),
        "SSLWantReadError" => Some("SSLWantReadError"),
        "SSLWantWriteError" => Some("SSLWantWriteError"),
        "SSLSyscallError" => Some("SSLSyscallError"),
        "SSLEOFError" => Some("SSLEOFError"),
        "SSLCertVerificationError" | "CertificateError" => Some("SSLCertVerificationError"),
        _ => None,
    }
}

/// Resolve a NAME bound by `from <stdlib module> import <exc> [as alias]`
/// to its canonical builtin exception, when the (module, name) pair is a
/// known stdlib exception alias. With `options`, a SIBLING-module import
/// (`from .connection import BaseSSLError` — urllib3) also resolves,
/// through the defining module's own exception-alias assign.
pub(crate) fn imported_exception_alias(
    name: &str,
    symbols: &SymbolTableScopes,
    options: Option<&crate::PythonOptions>,
) -> Option<&'static str> {
    // An aliased import registers the asname as an Alias hop to the
    // canonical name (`from socket import timeout as SocketTimeout`):
    // follow it before reading the ImportFrom.
    let mut current = name.to_string();
    for _ in 0..8 {
        match symbols.get(&current) {
            Some(crate::SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(crate::SymbolTableNode::ImportFrom(ifm)) => {
                let canonical = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(current.as_str()))
                    .map(|a| a.name.as_str())
                    .unwrap_or(current.as_str());
                if let Some(c) = stdlib_exception_canonical(&ifm.module, canonical) {
                    return Some(c);
                }
                // A sibling module's exception-alias binding.
                if let Some(options) = options {
                    let path = ifm.resolved_module_path(options);
                    return crate::ast::tree::module::module_def_exception_alias(
                        options, &path, canonical,
                    );
                }
                return None;
            }
            // An alias chain can terminate at a canonical stdlib-module
            // exception name with no symbol entry of its own
            // (`BaseSSLError = ssl.SSLError` → Alias("SSLError")).
            _ => return stdlib_module_exception_name(&current),
        }
    }
    None
}

pub fn is_exception_class_name(name: &str) -> bool {
    is_builtin_exception_name(name)
        // The naming convention covers user-defined exception classes
        // (`IDNAError`, `MyWarning`): `raise Name(...)` constructs a
        // PyException carrying the class name.
        || name.ends_with("Error")
        || name.ends_with("Exception")
        || name.ends_with("Warning")
}

/// Lower the raised expression to a PyException value: `Name(...)` and bare
/// `Name` forms that look like exception classes construct one carrying the
/// class name (so handlers can match on it); any other expression is
/// assumed to already be a PyException.
fn exception_value(
    exc: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    match exc {
        ExprType::Call(call) => {
            if let ExprType::Name(name) = call.func.as_ref() {
                if let Some(kind) = imported_exception_alias(&name.id, &symbols, Some(&options)) {
                    // `raise timeout(...)` under `from socket import
                    // timeout`: the canonical builtin (TimeoutError).
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        _ => {
                            let arg =
                                call.args[0].clone().to_rust(ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                    };
                    return Ok(quote!(PyException::new(#kind, #msg)));
                }
                if is_exception_class_name(&name.id)
                    || resolved_is_exception_class(&name.id, &options, &symbols)
                {
                    let kind = &name.id;
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        1 => {
                            let arg = call.args[0].clone().to_rust(ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                        _ => {
                            let args: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = call
                                .args
                                .iter()
                                .map(|a| {
                                    a.clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )
                                })
                                .collect();
                            let args = args?;
                            let fmt = vec!["{}"; args.len()].join(", ");
                            quote!(format!(#fmt, #(#args),*))
                        }
                    };
                    return Ok(quote!(PyException::new(#kind, #msg)));
                }
            }
            let tokens = exc.clone().to_rust(ctx, options, symbols)?;
            Ok(quote!(#tokens))
        }
        ExprType::Name(name)
            if imported_exception_alias(&name.id, &symbols, Some(&options)).is_some() =>
        {
            let kind = imported_exception_alias(&name.id, &symbols, Some(&options)).unwrap();
            Ok(quote!(PyException::new(#kind, String::new())))
        }
        ExprType::Name(name)
            if is_exception_class_name(&name.id)
                || resolved_is_exception_class(&name.id, &options, &symbols) =>
        {
            let kind = &name.id;
            Ok(quote!(PyException::new(#kind, String::new())))
        }
        other => {
            let tokens = other.clone().to_rust(ctx, options, symbols)?;
            Ok(quote!(#tokens))
        }
    }
}

/// Whether a name resolves (in the current symbol table) to a class that
/// IS an exception class by its bases — `InvalidCodepointContext` inherits
/// `IDNAError`, which follows the `*Error` convention, even though its own
/// name does not. Resolves imported classes through the defining module.
fn resolved_is_exception_class(
    name: &str,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match symbols.get(name) {
        Some(SymbolTableNode::ClassDef(c)) => crate::is_exception_class(c),
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            if options.module_defs.contains_key(&path) {
                crate::module_class_def(&options, &path, name)
                    .is_some_and(|(c, _)| crate::is_exception_class(&c))
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_raise, "raise ValueError('error')", "test.py");
}