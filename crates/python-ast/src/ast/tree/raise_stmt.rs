use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;

/// An exception-message argument, rendered for `format!("{}", ...)`:
/// Python's str() is py_display, not Rust's Display — a class INSTANCE,
/// Option, or boxed value in a message formats through py_display (the
/// generated class PyDisplay impl routes __str__/__repr__/the object
/// repr — round 34), where a raw `format!` would demand a Rust Display
/// the type does not have (E0277).
pub(crate) fn message_arg(
    expr: &crate::ExprType,
    ctx: crate::CodeGenContext,
    options: crate::PythonOptions,
    symbols: crate::SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // The type check borrows ctx BEFORE the render move (infer_type now
    // takes the CodeGenContext — Directive 2). A GENERIC parameter keeps
    // the raw format arg (its concrete type is bound elsewhere); a
    // concrete class instance/Option/boxed value needs py_display.
    let needs_display = !matches!(expr, crate::ExprType::Name(n)
        if options.param_type_vars.contains_key(&n.id))
        && matches!(
            crate::infer_type(Some(&ctx), expr, &options, &symbols),
            crate::TypeInfo::Class(_)
                | crate::TypeInfo::Option(_)
                | crate::TypeInfo::PyValue
                | crate::TypeInfo::PyValueMember(_)
                | crate::TypeInfo::PyObject
        );
    let rendered = expr.clone().to_rust(ctx, options.clone(), symbols.clone())?;
    if needs_display {
        Ok(quote!(py_display(&(#rendered))))
    } else {
        Ok(rendered)
    }
}

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
                    // `raise X from None` — CPython sets the cause to None
                    // (no cause text at all); the `None` literal cannot
                    // format either (E0277 ×17 in the corpus). Other
                    // causes keep the documented §12.3 folding.
                    if crate::is_none_expr(&cause) {
                        // fall through with the plain message
                    } else {
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
    // The OSError connection family and the remaining warning types:
    // absent for years (this registry had drifted from the runtime's
    // builtin_exceptions tree, which has had them all along), surfacing
    // as an E0432 on connection.py's `BrokenPipeError = BrokenPipeError`
    // re-export shim (issue #137 round 16).
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "BytesWarning",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "DeprecationWarning",
    "EncodingWarning",
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
    "ImportWarning",
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
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
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
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
];

/// The BUILTIN exception ALIASES — names bound to another class object,
/// so their `BuiltinException` variant is the CANONICAL name's
/// (`EnvironmentError`/`IOError` ARE `OSError`; the runtime's
/// `from_name` maps them). The round-52 fast path must not emit
/// `BuiltinException::EnvironmentError` — no such variant exists.
fn is_builtin_alias_name(name: &str) -> bool {
    matches!(name, "EnvironmentError" | "IOError")
}

/// The `BuiltinException` variant ident for a NON-ALIAS builtin name
/// (`ValueError` → `ValueError`): the round-52 fast path for a literal
/// `except ValueError:` clause — the runtime compares the raised
/// exception's discriminant against the variant and its ancestor slice
/// (no string walk). None for aliases (their variant is the canonical
/// name's) and for non-builtins (user classes stay on the string path).
pub(crate) fn builtin_exception_variant(name: &str) -> Option<String> {
    if !is_builtin_exception_name(name) || is_builtin_alias_name(name) {
        return None;
    }
    Some(crate::exception_tree::variant_ident(name))
}

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
            // `raise ssl.SSLError(...)` — the dotted stdlib exception
            // spelling (urllib3's pyopenssl): canonicalize like the bare
            // name, constructing the tagged PyException (the runtime
            // module has no SSLError item to call).
            if let ExprType::Attribute(attr) = call.func.as_ref()
                && let ExprType::Name(m) = attr.value.as_ref()
                && let Some(kind) = stdlib_exception_canonical(&m.id, &attr.attr)
            {
                let msg = match call.args.len() {
                    0 => quote!(String::new()),
                    _ => {
                        let arg = message_arg(&call.args[0], ctx, options, symbols)?;
                        quote!(format!("{}", #arg))
                    }
                };
                return Ok(quote!(PyException::new(#kind, #msg)));
            }
            if let ExprType::Name(name) = call.func.as_ref() {
                if let Some(kind) = imported_exception_alias(&name.id, &symbols, Some(&options)) {
                    // `raise timeout(...)` under `from socket import
                    // timeout`: the canonical builtin (TimeoutError).
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        _ => {
                            let arg = message_arg(&call.args[0], ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                    };
                    return Ok(quote!(PyException::new(#kind, #msg)));
                }
                // Round 82: a raise of a name imported from an EXTERNAL
                // module (`raise ResponseNotReady()` — http.client, which
                // urllib3 imports; `raise RemoteDisconnected()`) that is
                // NOT resolvable as an in-crate class: the exception model
                // is string-tagged, so the class NAME is the exception's
                // runtime value — construct the tagged PyException. The
                // previous fall-through dropped the raise to the boxed
                // None (the call into an external module), silently
                // replacing a raised exception with a returned value
                // (E0308 `PyException | PyValue` on every handler).
                if crate::ast::tree::import::resolves_to_external_import(
                    &name.id,
                    &options,
                    &symbols,
                ) {
                    let kind = &name.id;
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        1 => {
                            let arg = message_arg(&call.args[0], ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                        _ => {
                            let args: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = call
                                .args
                                .iter()
                                .map(|a| {
                                    message_arg(
                                        a,
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
                if is_exception_class_name(&name.id)
                    || resolved_is_exception_class(&name.id, &options, &symbols)
                {
                    let kind = &name.id;
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        1 => {
                            let arg = message_arg(&call.args[0], ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                        _ => {
                            let args: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = call
                                .args
                                .iter()
                                .map(|a| {
                                    message_arg(
                                        a,
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
            let Some(key) = crate::module_defs_key(options, &path) else {
                return false;
            };
            crate::module_class_def(&options, key, name).is_some_and(|(c, _)| {
                crate::is_exception_class(&c)
            })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_raise, "raise ValueError('error')", "test.py");
}