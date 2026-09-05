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
    // A runtime-ambiguous alias (`try: R = Root / except: R = Exception`)
    // names no one class: loud (Devin review on #330).
    let alias_name = match exc {
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) => Some(n.id.as_str()),
            _ => None,
        },
        ExprType::Name(n) => Some(n.id.as_str()),
        _ => None,
    };
    if let Some(msg) = alias_name.and_then(|n| ambiguous_alias_refusal(n, &options)) {
        return Ok(quote!(compile_error!(#msg)));
    }
    match exc {
        ExprType::Call(call) => {
            // An IN-CRATE exception class with a modeled __init__: render
            // the class's own message and carry its field stores as attrs
            // (bank's InsufficientFunds — round 99). Runs FIRST (before
            // the message builders move ctx/options/symbols below).
            // The class resolves locally or through its import (`from
            // .errors import MyError`): the defining module's ClassDef
            // and symbols drive the model and the ancestor chain.
            if let ExprType::Name(name) = call.func.as_ref()
                && let Some((cls, class_symbols)) =
                    crate::ast::tree::call::resolve_construction_class(&name.id, &symbols, &options)
                && crate::is_exception_class(&cls)
            {
                return exception_construction(
                    &cls,
                    &class_symbols,
                    call,
                    ctx,
                    options,
                    symbols,
                );
            }
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
                    // The kind is the class's canonical name: `raise
                    // R(...)` under `R = Root` raises a Root (Devin
                    // review on #330).
                    let kind = canonical_exception_class(&name.id, &symbols, &options)
                        .map(|(n, _)| n)
                        .unwrap_or_else(|| name.id.clone());
                    // An in-crate class's ancestor chain is class
                    // metadata: attached whether or not its __init__ is
                    // modeled (`class MyError(ValueError): pass` is
                    // caught by `except ValueError:`).
                    let ancestors = crate::ast::tree::call::resolve_construction_class(
                        &name.id, &symbols, &options,
                    )
                    .map(|(cls, class_symbols)| {
                        exception_ancestor_tokens(&cls, &class_symbols, &options)
                    })
                    .transpose()?;
                    // An IN-CRATE exception class with a modeled __init__
                    // (`class InsufficientFunds(BankError)` whose __init__
                    // calls super().__init__(f"need {needed}, have
                    // {available}") and stores self.needed/self.available
                    // — bank, round 99): render the __init__'s own message
                    // (CPython's exact text) and carry the field stores as
                    // attrs. Falls back to the positional message when the
                    // class is not resolvable in this module.
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
                    return Ok(match ancestors {
                        Some(ancestors) => quote!(PyException::new_with_attrs_and_ancestors(
                            #kind, #msg, vec![], vec![#(#ancestors),*]
                        )),
                        None => quote!(PyException::new(#kind, #msg)),
                    });
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
        // `raise MyError` (the class, no call) for an in-crate class: the
        // ancestor chain attaches here too.
        ExprType::Name(name)
            if crate::ast::tree::call::resolve_construction_class(&name.id, &symbols, &options)
                .is_some_and(|(c, _)| crate::is_exception_class(&c)) =>
        {
            let (cls, class_symbols) = crate::ast::tree::call::resolve_construction_class(
                &name.id, &symbols, &options,
            )
            .expect("checked by the guard");
            // The class's own name (`raise R` under `R = Root` raises a
            // Root — Devin review on #330).
            let kind = &cls.name;
            let ancestors = exception_ancestor_tokens(&cls, &class_symbols, &options)?;
            Ok(quote!(PyException::new_with_attrs_and_ancestors(
                #kind, String::new(), vec![], vec![#(#ancestors),*]
            )))
        }
        ExprType::Name(name)
            if is_exception_class_name(&name.id)
                || resolved_is_exception_class(&name.id, &options, &symbols) =>
        {
            // `raise Base` under `Base = ValueError` raises a ValueError.
            let kind = canonical_exception_class(&name.id, &symbols, &options)
                .map(|(n, _)| n)
                .unwrap_or_else(|| name.id.clone());
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
    // The one construction-class resolver: a local class, an import, an
    // alias, or a module-level rebinding (`R = Root` — Devin review on
    // #330).
    crate::ast::tree::call::resolve_construction_class(name, symbols, options)
        .is_some_and(|(c, _)| crate::is_exception_class(&c))
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_raise, "raise ValueError('error')", "test.py");
}
/// What an exception `__init__` passes to `super().__init__`: nothing,
/// one message expression, or its own `*args` (the variadic forwarder).
pub(crate) enum SuperMessage {
    Empty,
    /// The arguments as written, in the initializer's namespace: bound to
    /// the next initializer of the chain, or — at `BaseException` — one
    /// is the message, two or more make `str(e)` the args tuple's repr,
    /// and a keyword is a TypeError.
    Args {
        args: alloc_vec::Vec<crate::ExprType>,
        keywords: alloc_vec::Vec<(String, crate::ExprType)>,
    },
    /// `super().__init__(*args)` — with `**kwargs` too when `kwargs`.
    Forwarded { kwargs: bool },
}
mod alloc_vec {
    pub use std::vec::Vec;
}

/// Classify an exception `__init__` body into the model — the
/// `super().__init__` call (at most one: a second would replace the
/// first's `args`, refused) and the `self.<field> = <param>` stores, the
/// LAST store of a field winning as in Python — or the refusal message
/// for the first statement the model does not run. Walks the body
/// through the one statement visitor; a modeled statement has no
/// bodies, and an unmodeled one stops the walk at its head.
/// The model of an exception `__init__` body: the `super().__init__`
/// call, the field stores as they stand at the END of the body (the
/// attrs), and as they stood AT the super call (what a `self.<field>`
/// read in the message means — a later store does not rewrite the
/// message; Devin review on #330).
pub(crate) struct InitModel<'a> {
    pub super_init: Option<SuperMessage>,
    pub fields: Vec<(String, &'a str)>,
    pub fields_at_super: Vec<(String, &'a str)>,
    /// The stores AFTER the super call (they overwrite what a base's
    /// `__init__` stored — the chain's execution order).
    pub fields_after_super: Vec<(String, &'a str)>,
}

pub(crate) fn classify_exception_init<'a>(
    cls: &crate::ClassDef,
    body: &'a [crate::Statement],
    params: &[&'a str],
    vararg: Option<&str>,
    kwarg: Option<&str>,
) -> Result<InitModel<'a>, String> {
    use crate::ast::tree::visit::{Descend, Flow, is_self, walk_stmts};
    use crate::{ExprType, StatementType};
    let mut super_init: Option<SuperMessage> = None;
    let mut fields: Vec<(String, &'a str)> = Vec::new(); // (field, param)
    let mut fields_at_super: Vec<(String, &'a str)> = Vec::new();
    let mut fields_after_super: Vec<(String, &'a str)> = Vec::new();
    let mut refusal: Option<String> = None;
    walk_stmts(body, Descend::OwnScope, &mut |stmt| {
        let modeled = match &stmt.statement {
            StatementType::Assign(a) => {
                if let [ExprType::Attribute(attr)] = a.targets.as_slice()
                    && is_self(&attr.value)
                    && let ExprType::Name(v) = &a.value
                    && let Some(param) = params.iter().find(|p| **p == v.id)
                {
                    if let Some(existing) = fields.iter_mut().find(|(f, _)| *f == attr.attr) {
                        existing.1 = param;
                    } else {
                        fields.push((attr.attr.clone(), param));
                    }
                    if super_init.is_some() {
                        if let Some(existing) =
                            fields_after_super.iter_mut().find(|(f, _)| *f == attr.attr)
                        {
                            existing.1 = param;
                        } else {
                            fields_after_super.push((attr.attr.clone(), param));
                        }
                    }
                    true
                } else {
                    false
                }
            }
            StatementType::Expr(e) => match &e.value {
                ExprType::Call(sc) => {
                    let is_super_init = match sc.func.as_ref() {
                        ExprType::Attribute(attr) if attr.attr == "__init__" => {
                            matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "super")
                                || matches!(attr.value.as_ref(), ExprType::Call(c2)
                                    if matches!(c2.func.as_ref(), ExprType::Name(n) if n.id == "super"))
                        }
                        _ => false,
                    };
                    if is_super_init {
                        if super_init.is_some() {
                            refusal = Some(format!(
                                "rython: `{}.__init__` (line {}) calls `super().__init__` a \
                                 second time, which replaces the exception's args; rython's \
                                 one-message model refuses to pick either call silently — \
                                 keep one call",
                                cls.name,
                                stmt.lineno.unwrap_or(0)
                            ));
                            return Flow::Stop;
                        }
                        // The forwarder: `super().__init__(*args)` or
                        // `(*args, **kwargs)` — the initializer's OWN
                        // variadics by name; `*other` is not the raise's
                        // arguments (Devin review on #330).
                        let splats = sc.args.iter().any(|a| matches!(a, ExprType::Starred(_)))
                            || sc.keywords.iter().any(|k| k.arg.is_none());
                        let forwards_own = matches!(sc.args.as_slice(), [ExprType::Starred(st)]
                                if matches!(st.value.as_ref(), ExprType::Name(n) if Some(n.id.as_str()) == vararg))
                            && sc.keywords.iter().all(|k| {
                                k.arg.is_none()
                                    && matches!(&k.value, ExprType::Name(n) if Some(n.id.as_str()) == kwarg)
                            });
                        fields_at_super = fields.clone();
                        if forwards_own {
                            super_init = Some(SuperMessage::Forwarded {
                                kwargs: sc.keywords.iter().any(|k| k.arg.is_none()),
                            });
                        } else if splats {
                            refusal = Some(format!(
                                "rython: `{}.__init__` calls `super().__init__` with a starred \
                                 argument that is not its own `*args`/`**kwargs`: the message \
                                 would not be the raise's argument, and rython does not model \
                                 the unpacking; pass one message expression",
                                cls.name
                            ));
                            return Flow::Stop;
                        } else {
                            // A keyword binds to a user-defined base's
                            // parameter; BaseException.__init__ takes
                            // none (a TypeError, loud at the site).
                            super_init = Some(if sc.args.is_empty() && sc.keywords.is_empty() {
                                SuperMessage::Empty
                            } else {
                                SuperMessage::Args {
                                    args: sc.args.clone(),
                                    keywords: sc
                                        .keywords
                                        .iter()
                                        .map(|k| (k.arg.clone().expect("splats refused"), k.value.clone()))
                                        .collect(),
                                }
                            });
                        }
                    }
                    is_super_init
                }
                ExprType::Constant(c) => matches!(&c.0, Some(litrs::Literal::String(_))),
                _ => false,
            },
            StatementType::Pass => true,
            _ => false,
        };
        if modeled {
            Flow::Skip
        } else {
            refusal = Some(format!(
                "rython: `{}.__init__` (line {}) has a statement beyond `self.<field> = \
                 <param>` stores and the `super().__init__(<message>)` call: rython models \
                 an exception class's construction as its message and its stored fields, \
                 so that statement would not run at `raise {}(...)`; rython refuses to \
                 silently drop it. Move the logic to the raise site, or store the value \
                 as a field",
                cls.name,
                stmt.lineno.unwrap_or(0),
                cls.name
            ));
            Flow::Stop
        }
    });
    match refusal {
        Some(msg) => Err(msg),
        None => Ok(InitModel { super_init, fields, fields_at_super, fields_after_super }),
    }
}

/// The ONE construction of an in-crate exception class, wherever the
/// call sits — a `raise`, an `isinstance` probe, a value position
/// (`err = MyError("x")`, `str(MyError("x"))`): the modeled `__init__`
/// (`exception_class_raise`) when the class has one to model, else the
/// generic construction — the kind is the class's own name, the message
/// the one positional argument or empty (two or more would be the args
/// tuple's repr: refused), the ancestor chain attached (Devin review on
/// #330: the value position built `PyException::new(<ident>, ...)`,
/// which never compiled).
pub(crate) fn exception_construction(
    cls: &crate::ClassDef,
    class_symbols: &SymbolTableScopes,
    call: &crate::Call,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if let Some(tokens) = exception_class_raise(
        cls,
        call,
        ctx.clone(),
        options.clone(),
        symbols.clone(),
        class_symbols,
    )? {
        return Ok(tokens);
    }
    // No initializer chain to model: the arguments are `args` — the same
    // once-evaluation and rendering as a chain's end.
    if !call.keywords.is_empty() {
        let msg = format!(
            "rython: `{}()` takes no keyword arguments (BaseException.__init__ accepts none)",
            cls.name
        );
        return Ok(quote!(compile_error!(#msg)));
    }
    if call.args.iter().any(|a| matches!(a, ExprType::Starred(_))) {
        let msg = format!(
            "rython: `raise {}(*...)`: a starred argument to an exception constructor is \
             not modeled; pass the arguments explicitly",
            cls.name
        );
        return Ok(quote!(compile_error!(#msg)));
    }
    let mut st = Construction {
        ctx,
        options,
        symbols,
        prelude: Vec::new(),
        call_positional: call.args.clone(),
        runs_code: false,
    };
    st.runs_code = call.args.iter().any(|a| st.may_run_code(a));
    finish_construction(st, cls, class_symbols, call.args.clone(), Vec::new())
}

/// The ancestor chain of an in-crate exception class, base-most last
/// (`InsufficientFunds` → `["BankError", "Exception"]`): the bases
/// resolved through the symbol table, ending at the first builtin. Class
/// metadata, attached to EVERY construction of the class — a modeled
/// `__init__` or none (`class MyError(ValueError): pass` must be caught
/// by `except ValueError:`; Devin review on #330).
pub(crate) fn exception_ancestors(
    cls: &crate::ClassDef,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(exception_mro(cls, symbols, options)?
        .into_iter()
        .skip(1)
        .map(|(name, _)| name)
        .collect())
}

/// A class's method resolution order as Python computes it (C3
/// linearization over EVERY base, left to right — `class Both(A, B)` is
/// caught by `except B:` too; following the first base alone dropped
/// every other branch; Devin review on #330): the class first, then each
/// ancestor by its canonical name with its definition and scope when the
/// crate defines it. A builtin base is a leaf (the runtime expands its
/// own MRO from the interpreter table); a base the crate does not define
/// and that is not a builtin ends its branch. An inconsistent hierarchy
/// (`A(Exception)`, `B(A)`, `C(A, B)` — a TypeError at class creation in
/// CPython) is an error: the class does not exist there, so nothing here
/// may pretend it does (Devin review on #330).
pub(crate) fn exception_mro(
    cls: &crate::ClassDef,
    scope: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<Vec<(String, Option<(crate::ClassDef, crate::SymbolTableScopes)>)>, Box<dyn std::error::Error>> {
    type Entry = (String, Option<(crate::ClassDef, crate::SymbolTableScopes)>);
    /// A builtin base's own linearization from the interpreter table
    /// (`ValueError, Exception, BaseException, object`), so the merge
    /// sees the constraints CPython sees (`class C(Exception,
    /// ValueError)` has no consistent MRO — Devin review on #330).
    fn builtin_seq(name: &str) -> Result<Vec<Entry>, Box<dyn std::error::Error>> {
        Ok(match builtin_exception_mro(name)? {
            Some(mro) => mro.iter().map(|n| (n.clone(), None)).collect(),
            None => vec![(name.to_string(), None)],
        })
    }
    fn linearize(
        name: String,
        def: Option<(crate::ClassDef, crate::SymbolTableScopes)>,
        options: &crate::PythonOptions,
        active: &mut Vec<String>,
        direct_builtins: &mut std::collections::HashSet<String>,
    ) -> Result<Vec<Entry>, Box<dyn std::error::Error>> {
        let Some((cls, scope)) = def else {
            return builtin_seq(&name);
        };
        // A base that re-enters the chain being linearized is a cycle
        // (`class B(A)` then `A = B`: the name-keyed resolution turns
        // back on itself): loud, never a partial MRO (Devin review on
        // #330).
        if active.contains(&name) {
            return Err(format!(
                "class `{}`: cyclic inheritance through {} — a base re-enters the chain \
                 being linearized, so no MRO exists; rython refuses to invent one",
                name,
                active.join(" -> ")
            )
            .into());
        }
        if active.len() > 64 {
            return Err(format!(
                "class `{}`: the base chain is deeper than 64 classes; rython refuses to \
                 linearize it",
                name
            )
            .into());
        }
        active.push(name.clone());
        // Each base by its canonical name, with its definition: a crate
        // class; a builtin (its canonical MRO head, recorded as a direct
        // builtin base); a stdlib-module spelling (`ssl.SSLError`); an
        // exception-named base the crate does not define (an external
        // package's — the branch ends there, matched by name at runtime);
        // `object` adds nothing. Any other base cannot take a place in
        // the MRO: loud, never dropped.
        let mut bases: Vec<Entry> = Vec::new();
        for b in &cls.bases {
            match b {
                crate::ExprType::Name(n) if n.id == "object" => {}
                crate::ExprType::Name(n) => match canonical_exception_class(&n.id, &scope, options) {
                    Some((base, Some(def))) => bases.push((base, Some(def))),
                    Some((base, None)) => {
                        let head = builtin_exception_mro(&base)?
                            .and_then(|m| m.first().cloned())
                            .unwrap_or(base);
                        direct_builtins.insert(head.clone());
                        bases.push((head, None));
                    }
                    None => {
                        return Err(format!(
                            "class `{}` inherits from `{}`, which is not an exception class \
                             the model can place in its MRO (an ordinary class, an unresolved \
                             import, or a name bound to more than one class at runtime)",
                            cls.name, n.id
                        )
                        .into());
                    }
                },
                crate::ExprType::Attribute(a) => {
                    let base = match a.value.as_ref() {
                        crate::ExprType::Name(m) => stdlib_exception_canonical(&m.id, &a.attr)
                            .map(str::to_string)
                            .unwrap_or_else(|| a.attr.clone()),
                        _ => a.attr.clone(),
                    };
                    if !is_exception_class_name(&base) {
                        return Err(format!(
                            "class `{}` inherits from a dotted base `{}` that is no exception \
                             class the model knows; rython cannot place it in the MRO",
                            cls.name, base
                        )
                        .into());
                    }
                    let head = builtin_exception_mro(&base)?
                        .and_then(|m| m.first().cloned())
                        .unwrap_or(base);
                    direct_builtins.insert(head.clone());
                    bases.push((head, None));
                }
                _ => {
                    return Err(format!(
                        "class `{}` has a base expression rython cannot place in its MRO",
                        cls.name
                    )
                    .into());
                }
            }
        }
        let mut seqs: Vec<Vec<Entry>> = Vec::new();
        for (n, d) in &bases {
            seqs.push(linearize(n.clone(), d.clone(), options, active, direct_builtins)?);
        }
        active.pop();
        seqs.push(bases.clone());
        let mut out: Vec<Entry> = vec![(name, Some((cls, scope)))];
        // C3 merge: take the first head that is in no other sequence's
        // tail; none such is an inconsistent hierarchy.
        loop {
            seqs.retain(|s| !s.is_empty());
            if seqs.is_empty() {
                break;
            }
            let pick = seqs.iter().find_map(|s| {
                let head = &s[0].0;
                let in_tail = seqs.iter().any(|t| t[1..].iter().any(|(n, _)| n == head));
                (!in_tail).then(|| s[0].clone())
            });
            match pick {
                Some(entry) => {
                    for s in seqs.iter_mut() {
                        if !s.is_empty() && s[0].0 == entry.0 {
                            s.remove(0);
                        }
                    }
                    out.push(entry);
                }
                None => {
                    let bases: Vec<&str> = bases.iter().map(|(n, _)| n.as_str()).collect();
                    return Err(format!(
                        "class `{}`: cannot create a consistent method resolution order (MRO) \
                         for bases {} — CPython raises TypeError at class creation, so the \
                         class never exists; reorder or drop the bases",
                        out[0].0,
                        bases.join(", ")
                    )
                    .into());
                }
            }
        }
        Ok(out)
    }
    let mut direct_builtins: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut active: Vec<String> = Vec::new();
    let full = linearize(
        cls.name.clone(),
        Some((cls.clone(), scope.clone())),
        options,
        &mut active,
        &mut direct_builtins,
    )?;
    // The recorded chain: the crate's classes and the builtins they
    // derive from DIRECTLY — the runtime expands each builtin's own MRO
    // from the interpreter table, so the tail (`Exception, BaseException,
    // object`) is not repeated here.
    Ok(full
        .into_iter()
        .filter(|(name, def)| def.is_some() || direct_builtins.contains(name))
        .collect())
}

/// The refusal for an exception name that is a runtime-ambiguous alias
/// (bound to more than one class at module level, or to a class only
/// inside control flow that may leave it unbound), or None.
pub(crate) fn ambiguous_alias_refusal(name: &str, options: &PythonOptions) -> Option<String> {
    crate::ast::tree::class_def::is_runtime_ambiguous_alias(name, &options.this_module_path).then(|| {
        format!(
            "rython: `{name}` is bound to more than one class at module level, or to a class \
             only inside a `try:`, an `if`, a loop, or a `with` the conversion cannot fold \
             (so it may be unbound), so which exception class it names is decided at \
             runtime; rython refuses to follow one branch silently — bind the name once at \
             module level"
        )
    })
}

pub(crate) fn canonical_exception_class(
    name: &str,
    scope: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(String, Option<(crate::ClassDef, SymbolTableScopes)>)> {
    if crate::ast::tree::class_def::is_runtime_ambiguous_alias(name, &options.this_module_path) {
        return None;
    }
    if let Some((cls, cls_scope)) =
        crate::ast::tree::call::resolve_construction_class(name, scope, options)
    {
        return Some((cls.name.clone(), Some((cls, cls_scope))));
    }
    if let Some(builtin) = imported_exception_alias(name, scope, Some(options)) {
        return Some((builtin.to_string(), None));
    }
    // A local alias chain (`Base = ValueError`) ending at a builtin's
    // own name.
    let mut current = name.to_string();
    for _ in 0..8 {
        match scope.get(&current) {
            Some(SymbolTableNode::Alias(canonical)) => current = canonical.clone(),
            Some(SymbolTableNode::Assign { value: ExprType::Name(v), .. }) => {
                current = v.id.clone()
            }
            _ => break,
        }
    }
    is_exception_class_name(&current).then_some((current, None))
}

/// The ancestor chain as tokens for `new_with_attrs_and_ancestors`.
pub(crate) fn exception_ancestor_tokens(
    cls: &crate::ClassDef,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<Vec<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
    Ok(exception_ancestors(cls, symbols, options)?
        .iter()
        .map(|a| quote::quote!((#a) . to_string ()))
        .collect())
}

/// A temporary of the construction (bound once in the prelude).
fn is_exc_temp(e: &crate::ExprType) -> bool {
    matches!(e, crate::ExprType::Name(n) if n.id.starts_with("__rython_exc_"))
}

/// The temporary's name for an evaluated argument (`arg`) or a message
/// part (`m`) of one level of the initializer chain: the raise's own
/// level keeps the short names, a base's level is prefixed.
fn exc_temp(level: usize, kind: &str, k: usize) -> String {
    if level == 0 {
        format!("__rython_exc_{kind}{k}")
    } else {
        format!("__rython_exc_l{level}_{kind}{k}")
    }
}

/// A stored attribute of the construction: (field, its value in the
/// raise site's namespace, the class and parameter that stored it).
type Attr = (String, crate::ExprType, String, String);

/// `over` replaces same-named fields of `base` in place, new ones append.
fn overlay(mut base: Vec<Attr>, over: impl IntoIterator<Item = Attr>) -> Vec<Attr> {
    for a in over {
        match base.iter_mut().find(|b| b.0 == a.0) {
            Some(b) => *b = a,
            None => base.push(a),
        }
    }
    base
}

/// A site error is a `compile_error!` at the construction (the inner
/// Err); the outer Err is a conversion failure.
type Modeled<T> = Result<Result<T, String>, Box<dyn std::error::Error>>;

/// The state one construction threads through the initializer chain:
/// the prelude (temporaries bound once, in evaluation order), the
/// options carrying every temporary's type, the raise's own positional
/// arguments (what `BaseException.__new__` records when no initializer
/// of the chain calls `super().__init__`), and whether ANY expression of
/// the construction runs code (then a bare name is captured before it).
struct Construction {
    ctx: crate::CodeGenContext,
    options: crate::PythonOptions,
    symbols: crate::SymbolTableScopes,
    prelude: Vec<proc_macro2::TokenStream>,
    call_positional: Vec<crate::ExprType>,
    runs_code: bool,
}

/// Whether an expression MAY run code by its shape alone: a call, an
/// attribute read (a property getter, unless the receiver is `self` —
/// the modeled field reads), a subscript (`__getitem__`), a walrus, an
/// await or a yield, anywhere in it. Constants, names, and the pure
/// combinators over them (an f-string, an arithmetic or a comparison, a
/// literal collection) are not code of their own.
fn may_run_code_by_shape(e: &crate::ExprType) -> bool {
    use crate::ExprType;
    crate::ast::tree::visit::any_expr_for(e, crate::ast::tree::visit::Descend::All, |x| {
        matches!(
            x,
            ExprType::Call(_)
                | ExprType::Subscript(_)
                | ExprType::NamedExpr(_)
                | ExprType::Await(_)
                | ExprType::Yield(_)
                | ExprType::YieldFrom(_)
        ) || matches!(x, ExprType::Attribute(a) if !crate::ast::tree::visit::is_self(&a.value))
    })
}

impl Construction {
    /// Whether a site expression may run code: by shape, except that an
    /// attribute read is a plain field read (no code) when its receiver's
    /// class is known and declares no property getter of that name — the
    /// one authority attribute lowering routes property reads by.
    fn may_run_code(&self, e: &crate::ExprType) -> bool {
        use crate::ExprType;
        crate::ast::tree::visit::any_expr_for(e, crate::ast::tree::visit::Descend::All, |x| {
            match x {
                ExprType::Call(_)
                | ExprType::Subscript(_)
                | ExprType::NamedExpr(_)
                | ExprType::Await(_)
                | ExprType::Yield(_)
                | ExprType::YieldFrom(_) => true,
                ExprType::Attribute(a) => {
                    match crate::receiver_class_for_read(&a.value, &self.ctx, &self.symbols, &self.options) {
                        Some((class, class_symbols)) => {
                            class.has_property_getter(&a.attr, &class_symbols, &self.options)
                        }
                        None => !crate::ast::tree::visit::is_self(&a.value),
                    }
                }
                _ => false,
            }
        })
    }

    /// Bind `expr` to a fresh temporary, its type recorded for the reads.
    fn bind_temp(
        &mut self,
        name: String,
        expr: &crate::ExprType,
        clone: bool,
    ) -> Result<crate::ExprType, Box<dyn std::error::Error>> {
        let ident = proc_macro2::Ident::new(&name, proc_macro2::Span::call_site());
        let tokens =
            expr.clone().to_rust(self.ctx.clone(), self.options.clone(), self.symbols.clone())?;
        // A captured PLACE — a bare name, a field read — is cloned (the
        // site may read it again; a field cannot be moved out); a
        // captured combinator is an owned value already.
        let clone = clone
            && matches!(expr, crate::ExprType::Name(_) | crate::ExprType::Attribute(_));
        if clone {
            self.prelude.push(quote::quote!(let #ident = (#tokens).clone();));
        } else {
            self.prelude.push(quote::quote!(let #ident = #tokens;));
        }
        let ty = crate::infer_type(Some(&self.ctx), expr, &self.options, &self.symbols);
        if !matches!(ty, crate::TypeInfo::PyObject) {
            let mut name_types = (*self.options.name_types).clone();
            name_types.insert(name.clone(), ty);
            self.options.name_types = std::rc::Rc::new(name_types);
        }
        Ok(crate::ExprType::Name(crate::ast::tree::name::Name { id: name }))
    }

    /// Whether the expression at position `k` of `exprs` must be captured
    /// (evaluated into its temporary) before a later expression could
    /// rebind a name it reads or run effects it should precede: a later
    /// one of these runs code, or (`chain_after`: the expressions are a
    /// level's arguments, with the chain's messages still to run) any
    /// expression of the construction does (Devin review on #330: a
    /// composite argument observed a later mutation).
    fn captures(&self, exprs: &[crate::ExprType], k: usize, chain_after: bool) -> bool {
        let e = &exprs[k];
        // An expression that runs code is bound once anyway (an owned
        // value, no clone); a capture is for the pure ones — a name that
        // a later expression could rebind, a combinator that could raise
        // (`1 // 0`) before a later expression's effects (Devin review on
        // #330). A constant, `None`, `self`, and a temporary need none.
        !is_exc_temp(e)
            && !self.may_run_code(e)
            && !matches!(e, crate::ExprType::Constant(_) | crate::ExprType::NoneType(_))
            && !matches!(e, crate::ExprType::Name(n) if n.id == "self")
            && ((chain_after && self.runs_code) || exprs[k + 1..].iter().any(|e| self.may_run_code(e)))
    }
}

/// Whether a `super().__init__` argument of any initializer up the chain
/// may run code (by shape — the initializer's namespace has no site
/// types): a message that calls a function can rebind a global a
/// bare-name argument reads, so the name is captured first (Devin review
/// on #330).
fn construction_runs_code(chain: &[(crate::ClassDef, SymbolTableScopes)]) -> bool {
    let trivial = |e: &crate::ExprType| !may_run_code_by_shape(e);
    chain.iter().any(|(c, _)| {
        let Some(init) = c.init_method() else { return false };
        let params: Vec<&str> = init
            .args
            .posonlyargs
            .iter()
            .chain(init.args.args.iter())
            .skip(1)
            .chain(init.args.kwonlyargs.iter())
            .map(|a| a.arg.as_str())
            .collect();
        let vararg = init.args.vararg.as_ref().map(|a| a.arg.as_str());
        let kwarg = init.args.kwarg.as_ref().map(|a| a.arg.as_str());
        match classify_exception_init(c, &init.body, &params, vararg, kwarg) {
            Ok(InitModel { super_init: Some(SuperMessage::Args { args, keywords }), .. }) => {
                args.iter().any(|a| !trivial(a)) || keywords.iter().any(|(_, v)| !trivial(v))
            }
            _ => false,
        }
    })
}

/// One level of the initializer chain: the first `__init__` among
/// `chain` (the MRO from this level on) bound to `args`/`keywords` —
/// site-namespace expressions — with `attrs_in` the attribute state on
/// entry (the stores a subclass made before its `super().__init__`).
/// Returns what `BaseException.__init__` finally receives (`str(e)`'s
/// arguments, site namespace) and the attribute state on exit. A
/// user-defined base's `__init__` RUNS here, at the `super().__init__`
/// call, with the call's arguments bound to its parameters — its
/// message, its stores, its own super call (Devin review on #330: the
/// super call was taken for BaseException's, dropping the base's body).
/// `via` names the class whose super call produced `args`, for the
/// messages.
fn model_init_level(
    st: &mut Construction,
    cls: &crate::ClassDef,
    chain: &[(crate::ClassDef, SymbolTableScopes)],
    args: Vec<crate::ExprType>,
    keywords: Vec<(Option<String>, crate::ExprType)>,
    attrs_in: Vec<Attr>,
    level: usize,
    via: Option<&str>,
) -> Modeled<(Vec<crate::ExprType>, Vec<Attr>)> {
    use crate::ExprType;
    use std::collections::HashMap;
    macro_rules! site {
        ($($arg:tt)*) => { return Ok(Err(format!($($arg)*))) };
    }
    let Some(pos) = chain.iter().position(|(c, _)| c.init_method().is_some()) else {
        // BaseException.__init__: `args` is what it receives; it takes
        // no keyword (a TypeError in CPython).
        if !keywords.is_empty() {
            match via {
                Some(owner) => site!(
                    "rython: `{owner}.__init__` calls `super().__init__` with a keyword \
                     argument, which BaseException.__init__ does not accept (a TypeError in \
                     CPython)"
                ),
                None => site!(
                    "rython: `{}()` takes no keyword arguments (BaseException.__init__ \
                     accepts none)",
                    cls.name
                ),
            }
        }
        if args.iter().any(|a| matches!(a, ExprType::Starred(_))) {
            site!(
                "rython: `raise {}(*...)`: a starred argument to an exception constructor \
                 is not modeled; pass the arguments explicitly",
                cls.name
            );
        }
        return Ok(Ok((args, attrs_in)));
    };
    let (owner, owner_scope) = &chain[pos];
    let rest = &chain[pos + 1..];
    let init = owner.init_method().expect("the owner defines __init__");
    if init.args.vararg.is_some() || init.args.kwarg.is_some() {
        return model_variadic_level(st, cls, owner, init, rest, args, keywords, attrs_in, level);
    }
    // The positional parameters are the positional-only ones followed by
    // the ordinary ones (CPython's `posonlyargs ++ args`); the receiver
    // is the first of that sequence wherever it sits, and a
    // positional-only parameter cannot be passed by keyword.
    let combined: Vec<&crate::ast::tree::arguments::Parameter> =
        init.args.posonlyargs.iter().chain(init.args.args.iter()).collect();
    let positional: Vec<&str> = combined.iter().skip(1).map(|a| a.arg.as_str()).collect();
    let positional_only: Vec<&str> =
        init.args.posonlyargs.iter().skip(1).map(|a| a.arg.as_str()).collect();
    let kwonly: Vec<&str> = init.args.kwonlyargs.iter().map(|a| a.arg.as_str()).collect();
    // Defaults: positional defaults align to the tail of the positional
    // parameters, keyword-only defaults to their parameters.
    let mut defaults: HashMap<&str, &ExprType> = HashMap::new();
    let n_all = combined.len();
    let n_def = init.args.defaults.len();
    if n_def > n_all {
        site!("rython: `{}.__init__` has more defaults than parameters", owner.name);
    }
    for (i, d) in init.args.defaults.iter().enumerate() {
        let param = combined[n_all - n_def + i];
        if i + (n_all - n_def) > 0 {
            defaults.insert(param.arg.as_str(), d.as_ref());
        }
    }
    for (param, d) in init.args.kwonlyargs.iter().zip(init.args.kw_defaults.iter()) {
        if let Some(d) = d.as_deref() {
            defaults.insert(param.arg.as_str(), d);
        }
    }
    // Bind the call: (param → the argument expression), in source order.
    let mut given: Vec<(&str, ExprType)> = Vec::new();
    let n_positional = args.len();
    for (i, a) in args.into_iter().enumerate() {
        if matches!(a, ExprType::Starred(_)) {
            site!(
                "rython: `raise {}(*...)`: a starred argument to an exception constructor \
                 is not modeled; pass the arguments explicitly",
                cls.name
            );
        }
        let Some(p) = positional.get(i) else {
            site!(
                "rython: `{}()` takes {} positional argument(s) but {} were given",
                owner.name,
                positional.len(),
                n_positional
            );
        };
        given.push((p, a));
    }
    for (name, value) in keywords {
        let Some(name) = name else {
            site!(
                "rython: `raise {}(**...)`: a keyword splat to an exception constructor is \
                 not modeled; pass the arguments explicitly",
                cls.name
            );
        };
        if positional_only.iter().any(|p| *p == name) {
            site!(
                "rython: `{}()` got some positional-only arguments passed as keyword \
                 arguments: '{}'",
                owner.name,
                name
            );
        }
        let Some(param) = positional.iter().chain(kwonly.iter()).find(|p| **p == name) else {
            site!("rython: `{}()` got an unexpected keyword argument '{}'", owner.name, name);
        };
        if given.iter().any(|(p, _)| *p == name) {
            site!("rython: `{}()` got multiple values for argument '{}'", owner.name, name);
        }
        given.push((param, value));
    }
    // Every parameter's substitution at the site: a temporary for an
    // evaluated argument (bound once, in source order — a property read,
    // a call, a walrus, an f-string run once), the expression itself for
    // a constant or a bare name — unless a LATER expression of the
    // construction runs code, which could rebind the name before the
    // message and the attrs read it (`E(counter, bump())`, or a message
    // that calls `bump()`): then the name is captured in source order
    // too, a clone (Devin review on #330).
    // The __init__ body — `self.<field> = <param>` stores and the
    // `super().__init__(...)` call; a docstring and `pass` are inert;
    // anything else would not run at the site — classified through the
    // one statement visitor.
    let params: Vec<&str> = positional.iter().chain(kwonly.iter()).copied().collect();
    let InitModel { super_init, fields, fields_at_super, fields_after_super } =
        match classify_exception_init(owner, &init.body, &params, None, None) {
            Ok(model) => model,
            Err(msg) => return Ok(Err(msg)),
        };
    // The parameters the model reads: a field store, a name in the super
    // call's arguments, a `self.<field>` read there.
    let mut read_by_model: Vec<&str> = fields.iter().map(|(_, p)| *p).collect();
    if let Some(SuperMessage::Args { args, keywords }) = &super_init {
        for m0 in args.iter().chain(keywords.iter().map(|(_, v)| v)) {
            crate::ast::tree::visit::walk_expr(m0, &mut |e| {
                if let ExprType::Name(n) = e
                    && let Some(p) = params.iter().find(|p| **p == n.id)
                {
                    read_by_model.push(p);
                }
                if let ExprType::Attribute(attr) = e
                    && crate::ast::tree::visit::is_self(&attr.value)
                    && let Some((_, p)) = fields_at_super.iter().find(|(f, _)| *f == attr.attr)
                {
                    read_by_model.push(p);
                }
            });
        }
    }
    let exprs: Vec<ExprType> = given.iter().map(|(_, a)| a.clone()).collect();
    let mut substitution: HashMap<&str, ExprType> = HashMap::new();
    for (k, (param, arg)) in given.iter().enumerate() {
        let captured = st.captures(&exprs, k, true);
        // Bound once: an expression that may run code (a call, a property
        // read, a subscript, a walrus), or one captured before a later
        // expression that may; a constant, a name, or a pure combinator
        // over them (an f-string, an arithmetic) is otherwise read in
        // place.
        if !captured && !st.may_run_code(arg) {
            // Every argument is evaluated, in source order, whether or
            // not the model reads it: a bare name the message and the
            // fields ignore is still read once (`raise E(undefined_name)`
            // is a NameError in CPython, an unresolved name in rustc
            // here), and an ignored pure combinator still runs (`raise
            // E(1 // 0)` is the ZeroDivisionError, not E — Devin review
            // on #330).
            if !read_by_model.contains(param) && !is_exc_temp(arg) {
                match arg {
                    ExprType::Constant(_) | ExprType::NoneType(_) => {}
                    ExprType::Name(n) => {
                        let ident = crate::safe_ident(&n.id);
                        st.prelude.push(quote::quote!(let _ = &#ident;));
                    }
                    other => {
                        let tokens = other.clone().to_rust(
                            st.ctx.clone(),
                            st.options.clone(),
                            st.symbols.clone(),
                        )?;
                        st.prelude.push(quote::quote!(let _ = #tokens;));
                    }
                }
            }
            substitution.insert(param, arg.clone());
            continue;
        }
        let temp = st.bind_temp(exc_temp(level, "arg", k), arg, captured)?;
        substitution.insert(param, temp);
    }
    if level == 0 {
        // What `BaseException.__new__` records as `args` when no
        // initializer calls super: the raise's positional arguments, as
        // bound here (a temporary stands for an evaluated one).
        st.call_positional = given[..n_positional].iter().map(|(p, _)| substitution[p].clone()).collect();
    }
    for param in positional.iter().chain(kwonly.iter()) {
        if substitution.contains_key(param) {
            continue;
        }
        let Some(d) = defaults.get(param) else {
            site!("rython: `{}()` missing a required argument: '{}'", owner.name, param);
        };
        let constant = matches!(d, ExprType::Constant(_) | ExprType::NoneType(_))
            || matches!(d, ExprType::UnaryOp(u) if matches!(u.operand.as_ref(), ExprType::Constant(_)));
        if !constant {
            site!(
                "rython: `{}.__init__`'s default for '{}' is not a constant: CPython \
                 evaluates a default once at definition time, which re-rendering it at \
                 each raise cannot reproduce; pass the argument explicitly, or make the \
                 default a constant",
                owner.name,
                param
            );
        }
        substitution.insert(param, (*d).clone());
    }
    let stored = |list: &[(String, &str)]| -> Vec<Attr> {
        list.iter()
            .map(|(f, p)| (f.clone(), substitution[p].clone(), owner.name.clone(), p.to_string()))
            .collect()
    };
    let Some(super_init) = super_init else {
        // No super call at all: `BaseException.__new__` still records
        // the raise's positional arguments as `args`, so `str(e)` is the
        // one positional argument, or empty for none — more than one is
        // the tuple's repr, refused; the chain's remaining initializers
        // never run.
        let msg = match st.call_positional.len() {
            0 => Vec::new(),
            1 => vec![st.call_positional[0].clone()],
            n => site!(
                "rython: `{}.__init__` never calls `super().__init__`, so `str(e)` is the \
                 repr of the {} positional arguments as a tuple, which rython's \
                 one-message exception model does not reproduce; call \
                 `super().__init__(<message>)`",
                owner.name,
                n
            ),
        };
        return Ok(Ok((msg, overlay(attrs_in, stored(&fields)))));
    };
    // The attribute state at the super call: what a `self.<field>` read
    // in the message means (a later store does not rewrite it).
    let attrs_at_super = overlay(attrs_in, stored(&fields_at_super));
    let (next_args, next_keywords): (Vec<ExprType>, Vec<(Option<String>, ExprType)>) =
        match super_init {
            SuperMessage::Empty => (Vec::new(), Vec::new()),
            SuperMessage::Args { args, keywords } => {
                let mut next_args = Vec::new();
                for m0 in &args {
                    match rewrite_super_arg(st, owner, owner_scope, m0, &params, &substitution, &attrs_at_super)? {
                        Ok(m) => next_args.push(m),
                        Err(msg) => return Ok(Err(msg)),
                    }
                }
                let mut next_keywords = Vec::new();
                for (name, v0) in &keywords {
                    match rewrite_super_arg(st, owner, owner_scope, v0, &params, &substitution, &attrs_at_super)? {
                        Ok(v) => next_keywords.push((Some(name.clone()), v)),
                        Err(msg) => return Ok(Err(msg)),
                    }
                }
                (next_args, next_keywords)
            }
            // `super().__init__(*args)` names a `*args` this initializer
            // does not have: refused by the classifier.
            SuperMessage::Forwarded { .. } => unreachable!("forwarding needs a vararg"),
        };
    let (msg, attrs_after_super) = match model_init_level(
        st,
        cls,
        rest,
        next_args,
        next_keywords,
        attrs_at_super,
        level + 1,
        Some(&owner.name),
    )? {
        Ok(out) => out,
        Err(msg) => return Ok(Err(msg)),
    };
    // The stores after the super call overwrite what the base stored.
    Ok(Ok((msg, overlay(attrs_after_super, stored(&fields_after_super)))))
}

/// A `super().__init__` argument rewritten from the initializer's
/// namespace to the site's, evaluated once: a lambda or a comprehension
/// is refused (bindings of its own the rewrite cannot model);
/// `self.<field>` means the value the field held at the super call (a
/// store of this initializer before the call, or of a subclass before
/// its own); every parameter means its substitution — one pre-order pass
/// through the mutable visitor (a replaced node is a leaf, never
/// rewritten again, so a caller name that matches another parameter
/// binds ONCE — Devin review on #330); every other name the argument
/// reads must be bound at the site as the `__init__` sees it.
fn rewrite_super_arg(
    st: &Construction,
    owner: &crate::ClassDef,
    owner_scope: &SymbolTableScopes,
    m0: &crate::ExprType,
    params: &[&str],
    substitution: &std::collections::HashMap<&str, crate::ExprType>,
    attrs_at_super: &[Attr],
) -> Modeled<crate::ExprType> {
    use crate::ExprType;
    use crate::ast::tree::visit::{Descend, any_expr_for, is_self, walk_expr, walk_expr_mut};
    macro_rules! site {
        ($($arg:tt)*) => { return Ok(Err(format!($($arg)*))) };
    }
    if any_expr_for(m0, Descend::All, |e| {
        matches!(
            e,
            ExprType::Lambda(_)
                | ExprType::ListComp(_)
                | ExprType::SetComp(_)
                | ExprType::DictComp(_)
                | ExprType::GeneratorExp(_)
        )
    }) {
        site!(
            "rython: `{}.__init__`'s message holds a lambda or a comprehension, whose own \
             bindings the raise-site rewrite cannot model; compute the message in a local \
             first",
            owner.name
        );
    }
    // The names the argument reads from the initializer's own scope —
    // neither a parameter nor a modeled `self.<field>` read; a bare
    // `self` (the exception under construction) or a field no store
    // before the super call set is refused.
    let mut free: Vec<String> = Vec::new();
    let mut self_attr_reads = 0usize;
    let mut self_reads = 0usize;
    let mut bad_field: Option<String> = None;
    walk_expr(m0, &mut |e| match e {
        ExprType::Attribute(attr) if is_self(&attr.value) => {
            self_attr_reads += 1;
            if !attrs_at_super.iter().any(|(f, ..)| *f == attr.attr) && bad_field.is_none() {
                bad_field = Some(attr.attr.clone());
            }
        }
        ExprType::Name(n) if n.id == "self" => self_reads += 1,
        ExprType::Name(n) if !params.contains(&n.id.as_str()) && !free.contains(&n.id) => {
            free.push(n.id.clone());
        }
        _ => {}
    });
    if let Some(f) = bad_field {
        site!(
            "rython: `{}.__init__`'s message reads `self.{f}`, which no `self.{f} = <param>` \
             store before the `super().__init__` call sets: rython models the message from \
             the stored fields and refuses to guess the value; store the field first, or \
             pass the value as an argument",
            owner.name
        );
    }
    if self_reads > self_attr_reads {
        site!(
            "rython: `{}.__init__`'s message reads `self` (the exception under \
             construction), which the raise-site rewrite cannot model; pass the value as \
             an argument",
            owner.name
        );
    }
    for name in free {
        let at_def = format!("{:?}", owner_scope.module_get(&name));
        let at_site = format!("{:?}", st.symbols.get(&name));
        let local = st.options.name_types.contains_key(&name)
            && owner_scope.module_get(&name).is_some();
        if at_def != at_site || local {
            site!(
                "rython: `{}.__init__`'s message reads `{}`, which at this raise site is not \
                 the binding the `__init__` sees (a local, or another module's global of \
                 the same name): rython renders the message at the raise site and refuses \
                 to silently read the wrong one; pass the value as an argument, or store it \
                 as a field",
                owner.name,
                name
            );
        }
    }
    let mut m = m0.clone();
    walk_expr_mut(&mut m, &mut |e| {
        if let ExprType::Attribute(attr) = e
            && is_self(&attr.value)
            && let Some((_, value, ..)) = attrs_at_super.iter().find(|(f, ..)| *f == attr.attr)
        {
            *e = value.clone();
        } else if let ExprType::Name(n) = e
            && let Some(sub) = substitution.get(n.id.as_str())
        {
            *e = sub.clone();
        }
    });
    Ok(Ok(m))
}

/// A level whose `__init__` is variadic (`*args` / `**kwargs`): nothing
/// binds by name, so the body may only forward to
/// `super().__init__(*args)` (a docstring and `pass` aside) — the named
/// parameters before `*args` take the first arguments (evaluated once,
/// dropped), the forwarded slice is the rest, handed to the next
/// initializer of the chain; a keyword is forwarded with `**kwargs`,
/// swallowed by a `**kwargs` the call does not forward, and a TypeError
/// (loud) without one.
#[allow(clippy::too_many_arguments)]
fn model_variadic_level(
    st: &mut Construction,
    cls: &crate::ClassDef,
    owner: &crate::ClassDef,
    init: &crate::FunctionDef,
    rest: &[(crate::ClassDef, SymbolTableScopes)],
    args: Vec<crate::ExprType>,
    keywords: Vec<(Option<String>, crate::ExprType)>,
    attrs_in: Vec<Attr>,
    level: usize,
) -> Modeled<(Vec<crate::ExprType>, Vec<Attr>)> {
    use crate::ExprType;
    macro_rules! site {
        ($($arg:tt)*) => { return Ok(Err(format!($($arg)*))) };
    }
    let vararg = init.args.vararg.as_ref().map(|a| a.arg.as_str());
    let kwarg = init.args.kwarg.as_ref().map(|a| a.arg.as_str());
    let named: Vec<&str> = init
        .args
        .posonlyargs
        .iter()
        .chain(init.args.args.iter())
        .skip(1)
        .map(|a| a.arg.as_str())
        .collect();
    if !init.args.kwonlyargs.is_empty() {
        site!(
            "rython: `{}.__init__` mixes `*args` with keyword-only parameters, which the \
             forwarding model does not bind; name the parameters, or drop `*args`",
            owner.name
        );
    }
    let model = match classify_exception_init(owner, &init.body, &named, vararg, kwarg) {
        Ok(model) => model,
        Err(msg) => return Ok(Err(msg)),
    };
    if !model.fields.is_empty() {
        site!(
            "rython: `{}.__init__` stores a named parameter as a field while forwarding \
             `*args`, which the forwarding model does not bind; name every parameter, or \
             drop `*args`",
            owner.name
        );
    }
    if let Some(SuperMessage::Args { .. } | SuperMessage::Empty) = model.super_init {
        site!(
            "rython: `{}.__init__` takes `*args` but does not forward them to \
             `super().__init__(*args)`: the message would not be the raise's argument; \
             forward the arguments, or name the parameters",
            owner.name
        );
    }
    if kwarg.is_none() && !keywords.is_empty() {
        site!(
            "rython: `{}()` takes no keyword arguments (BaseException.__init__ accepts none)",
            cls.name
        );
    }
    if args.iter().any(|a| matches!(a, ExprType::Starred(_))) {
        site!(
            "rython: `raise {}(*...)`: a starred argument to an exception constructor is \
             not modeled; pass the arguments explicitly",
            cls.name
        );
    }
    if args.len() < named.len() {
        site!(
            "rython: `{}()` missing a required argument: '{}'",
            owner.name,
            named[args.len()]
        );
    }
    if keywords.iter().any(|(name, _)| name.is_none()) {
        site!(
            "rython: `raise {}(**...)`: a keyword splat to an exception constructor is \
             not modeled; pass the arguments explicitly",
            cls.name
        );
    }
    // EVERY incoming expression — the named parameters' arguments, the
    // forwarded slice, the keywords — is bound once in CALL order before
    // any is forwarded or dropped: `E(f(), ignored=g())` runs f() then
    // g() (Devin review on #330: a swallowed keyword ran before a
    // forwarded positional). An expression that may run code, or one a
    // later expression's effects could disturb, is its temporary; a
    // constant or a name stays in place.
    let exprs: Vec<ExprType> =
        args.iter().cloned().chain(keywords.iter().map(|(_, v)| v.clone())).collect();
    let mut bound: Vec<ExprType> = Vec::with_capacity(exprs.len());
    for (k, e) in exprs.iter().enumerate() {
        let captured = st.captures(&exprs, k, true);
        if captured || st.may_run_code(e) {
            bound.push(st.bind_temp(exc_temp(level, "arg", k), e, captured)?);
        } else {
            bound.push(e.clone());
        }
    }
    // A bound expression the chain never reads is still evaluated once
    // (a temporary already was; a bare name is read; a pure combinator
    // runs — it may raise; a constant needs nothing).
    let drop = |st: &mut Construction, e: &ExprType| -> Result<(), Box<dyn std::error::Error>> {
        match e {
            ExprType::Constant(_) | ExprType::NoneType(_) => {}
            ExprType::Name(n) if is_exc_temp(e) => {
                let _ = n;
            }
            ExprType::Name(n) => {
                let ident = crate::safe_ident(&n.id);
                st.prelude.push(quote::quote!(let _ = &#ident;));
            }
            other => {
                let tokens =
                    other.clone().to_rust(st.ctx.clone(), st.options.clone(), st.symbols.clone())?;
                st.prelude.push(quote::quote!(let _ = #tokens;));
            }
        }
        Ok(())
    };
    let n_args = args.len();
    if level == 0 {
        st.call_positional = bound[..n_args].to_vec();
    }
    if model.super_init.is_none() {
        // No super call: `BaseException.__new__` records the raise's
        // positional arguments (the named ones included) as `args`; a
        // base's forwarded arguments are evaluated once and dropped.
        let msg = match st.call_positional.len() {
            0 => Vec::new(),
            1 => vec![st.call_positional[0].clone()],
            n => site!(
                "rython: `{}.__init__` never calls `super().__init__`, so `str(e)` is the \
                 repr of the {} positional arguments as a tuple, which rython's \
                 one-message exception model does not reproduce; call \
                 `super().__init__(<message>)`",
                owner.name,
                n
            ),
        };
        let read_by_msg = |e: &ExprType| msg.iter().any(|m| m == e);
        for e in &bound {
            if !read_by_msg(e) {
                drop(st, e)?;
            }
        }
        return Ok(Ok((msg, attrs_in)));
    }
    // The named parameters' arguments are dropped (the body forwards only
    // `*args`); so is a keyword `**kwargs` swallows.
    for e in &bound[..named.len()] {
        drop(st, e)?;
    }
    let forwarded: Vec<ExprType> = bound[named.len()..n_args].to_vec();
    let forwards_kwargs = matches!(model.super_init, Some(SuperMessage::Forwarded { kwargs: true }));
    let mut next_keywords = Vec::new();
    for ((name, _), value) in keywords.iter().zip(bound[n_args..].iter()) {
        if forwards_kwargs {
            next_keywords.push((name.clone(), value.clone()));
        } else {
            drop(st, value)?;
        }
    }
    model_init_level(st, cls, rest, forwarded, next_keywords, attrs_in, level + 1, Some(&owner.name))
}

/// The construction's tokens from what `BaseException.__init__` receives
/// and the stored attrs: each message part evaluated once (a temporary
/// unless it is a constant or a name — a bare name captured when a later
/// part runs code), `str(e)` as CPython sets it (the one argument, empty
/// for none, the args tuple's repr for two or more), `repr(e)` recording
/// every part's repr, the attrs boxed, the ancestor chain attached.
fn finish_construction(
    mut st: Construction,
    cls: &crate::ClassDef,
    class_symbols: &SymbolTableScopes,
    msg_exprs: Vec<crate::ExprType>,
    attrs: Vec<Attr>,
) -> Result<proc_macro2::TokenStream, Box<dyn std::error::Error>> {
    use crate::ExprType;
    let mut parts: Vec<ExprType> = Vec::new();
    for (i, m) in msg_exprs.iter().enumerate() {
        let captured = st.captures(&msg_exprs, i, false);
        if !captured && !st.may_run_code(m) {
            parts.push(m.clone());
        } else {
            parts.push(st.bind_temp(exc_temp(0, "m", i), m, captured)?);
        }
    }
    let mut part_tokens: Vec<proc_macro2::TokenStream> = Vec::new();
    for p in &parts {
        part_tokens.push(p.clone().to_rust(st.ctx.clone(), st.options.clone(), st.symbols.clone())?);
    }
    let msg = match parts.as_slice() {
        [] => quote::quote!(String::new()),
        [one] => {
            // The display wrapping the message builder applies (a class
            // instance, an Option, a boxed value — a temporary's type is
            // recorded in the options).
            let m = message_arg(one, st.ctx.clone(), st.options.clone(), st.symbols.clone())?;
            quote::quote!(format!("{}", #m))
        }
        many => {
            let fmt = format!("({})", vec!["{}"; many.len()].join(", "));
            quote::quote!(format!(#fmt, #(stdpython::PyRepr::py_repr(&(#part_tokens))),*))
        }
    };
    let args_repr = quote::quote!(vec![#(stdpython::PyRepr::py_repr(&(#part_tokens))),*]);
    let kind = &cls.name;
    let mut attr_pairs: Vec<proc_macro2::TokenStream> = Vec::new();
    for (f, value, owner, param) in &attrs {
        // The attrs are BOXED values: a class instance (`raise
        // EmptyPoolError(self, ...)` storing `self.pool = pool` —
        // urllib3's connectionpool) has no box, and dropping the field
        // would make `e.pool` a silent AttributeError; refused at the
        // site instead. A temporary's type lives in the options (the
        // site binding), so an evaluated construction or a captured name
        // is seen too (Devin review on #330).
        let instance = crate::ast::tree::visit::is_self(value)
            || matches!(
                crate::infer_type(Some(&st.ctx), value, &st.options, &st.symbols),
                crate::TypeInfo::Class(_)
            );
        if instance {
            let msg = format!(
                "rython: `{owner}.__init__` stores its parameter `{param}` as `self.{f}`, and \
                 the argument here is a class instance, which an exception's boxed attrs \
                 cannot hold; rython refuses to silently drop the field. Store a plain \
                 value (a name, an id) instead"
            );
            return Ok(quote::quote!(compile_error!(#msg)));
        }
        let boxed = if crate::is_none_expr(value) {
            quote::quote!(stdpython :: PyValue :: None_)
        } else {
            let a = value.clone().to_rust(st.ctx.clone(), st.options.clone(), st.symbols.clone())?;
            // A temporary is read by the message too: clone it into the box.
            if is_exc_temp(value) {
                quote::quote!(stdpython :: PyValue :: from ((#a).clone()))
            } else {
                quote::quote!(stdpython :: PyValue :: from (#a))
            }
        };
        attr_pairs.push(quote::quote!((#f . to_string (), #boxed)));
    }
    let ancestor_tokens = exception_ancestor_tokens(cls, class_symbols, &st.options)?;
    let construct = quote::quote! {
        stdpython :: PyException :: new_with_attrs_and_ancestors (
            #kind , #msg , vec ! [#(#attr_pairs),*] ,
            vec ! [#(#ancestor_tokens),*]
        ) . with_args_repr (#args_repr)
    };
    Ok(if st.prelude.is_empty() {
        construct
    } else {
        let prelude = st.prelude;
        quote::quote!({ #(#prelude)* #construct })
    })
}

/// A construction of an in-crate exception class through its
/// initializer CHAIN: the first `__init__` up the MRO runs with the
/// call's arguments bound the way CPython binds a call (positionals in
/// order, keywords by name, a missing parameter to its constant
/// default); its `super().__init__(...)` runs the NEXT `__init__` of the
/// chain the same way, down to `BaseException.__init__`, whose arguments
/// are `str(e)` and `repr(e)`; the `self.<field> = <param>` stores of
/// every level, in execution order, are the attrs. The model is exactly
/// those stores and super calls; any other statement is a
/// `compile_error!` at the site. Every evaluated expression is bound
/// once, in evaluation order (a bare name captured before a later
/// expression that runs code). None when `cls` is not an exception class
/// of the crate.
pub(crate) fn exception_class_raise(
    cls: &crate::ClassDef,
    call: &crate::Call,
    ctx: crate::CodeGenContext,
    options: crate::PythonOptions,
    symbols: crate::SymbolTableScopes,
    class_symbols: &crate::SymbolTableScopes,
) -> Result<Option<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
    if !crate::is_exception_class(cls) {
        return Ok(None);
    }
    // The chain: the MRO's crate classes, `cls` first (Python's MRO: a
    // `class Child(Base): pass` constructs through Base's `__init__` —
    // the kind stays Child and the ancestors are Child's; Devin review
    // on #330).
    let chain: Vec<(crate::ClassDef, SymbolTableScopes)> = exception_mro(cls, class_symbols, &options)?
        .into_iter()
        .filter_map(|(_, def)| def)
        .collect();
    let mut st = Construction {
        ctx,
        options,
        symbols,
        prelude: Vec::new(),
        call_positional: call.args.clone(),
        runs_code: false,
    };
    st.runs_code = call.args.iter().any(|a| st.may_run_code(a))
        || call.keywords.iter().any(|k| st.may_run_code(&k.value))
        || construction_runs_code(&chain);
    let keywords: Vec<(Option<String>, crate::ExprType)> =
        call.keywords.iter().map(|k| (k.arg.clone(), k.value.clone())).collect();
    let (msg_exprs, attrs) =
        match model_init_level(&mut st, cls, &chain, call.args.clone(), keywords, Vec::new(), 0, None)? {
            Ok(out) => out,
            Err(msg) => return Ok(Some(quote::quote!(compile_error!(#msg)))),
        };
    // What `BaseException.__new__` records as `args` when no initializer
    // calls super is the raise's own arguments — as bound at the site
    // (a level-0 temporary stands for an evaluated one).
    Ok(Some(finish_construction(st, cls, class_symbols, msg_exprs, attrs)?))
}

/// A modeled exception field's type: the exception class's __init__
/// stores `self.<field> = <param>` where the param's annotation types it
/// (`self.needed = needed` with `needed: int` → Int — bank's e.needed,
/// round 99). None for an unmodeled read.
pub(crate) fn exception_field_type(
    cls: &crate::ClassDef,
    field: &str,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<Option<crate::TypeInfo>, Box<dyn std::error::Error>> {
    // Walk the class's MRO: the field may be defined on a BASE (an
    // `except BankError as e` catching an InsufficientFunds whose
    // __init__ stores the field — Devin review on #328), local or
    // imported, named directly or through an alias, on any branch of a
    // multiple inheritance — the one linearization, each hop's
    // annotation resolved in its own scope (Devin review on #330).
    let chain: Vec<(crate::ClassDef, SymbolTableScopes)> = exception_mro(cls, symbols, options)?
        .into_iter()
        .filter_map(|(_, def)| def)
        .collect();
    let Some((param, init, c_scope)) = chain_field_store(&chain, field) else {
        return Ok(None);
    };
    // The param's annotation types the field — a positional-only,
    // positional, or keyword-only parameter.
    let Some(param) = init
        .args
        .posonlyargs
        .iter()
        .chain(init.args.args.iter())
        .chain(init.args.kwonlyargs.iter())
        .find(|p| p.arg == param)
    else {
        return Ok(None);
    };
    if let Some(ann) = param.annotation.as_ref()
        && let Some(t) = crate::resolve_alias_typeinfo(ann, c_scope, options)
    {
        return Ok(Some(t));
    }
    // An unannotated param: no type — the reader refuses loudly rather
    // than guessing (an Int guess made a str field's read a false
    // AttributeError).
    Ok(Some(crate::TypeInfo::PyObject))
}

/// The store that gives `field` its final value over the initializer
/// chain, in the chain's EXECUTION order — the order the construction
/// records the attrs in (`model_init_level`): the first `__init__` up
/// the MRO runs; its stores AFTER its `super().__init__` win, else the
/// base's `__init__` (which ran at that call) decides the same way, else
/// this initializer's stores before the call; with no super call the
/// chain ends here. `(the parameter stored, the initializer, its scope)`
/// (Devin review on #330: the first store's annotation typed a field the
/// last store, or the base's, had set to another type).
fn chain_field_store<'a>(
    chain: &'a [(crate::ClassDef, SymbolTableScopes)],
    field: &str,
) -> Option<(&'a str, &'a crate::FunctionDef, &'a SymbolTableScopes)> {
    let pos = chain.iter().position(|(c, _)| c.init_method().is_some())?;
    let (c, c_scope) = &chain[pos];
    let init = c.init_method()?;
    let params: Vec<&str> = init
        .args
        .posonlyargs
        .iter()
        .chain(init.args.args.iter())
        .skip(1)
        .chain(init.args.kwonlyargs.iter())
        .map(|a| a.arg.as_str())
        .collect();
    let vararg = init.args.vararg.as_ref().map(|a| a.arg.as_str());
    let kwarg = init.args.kwarg.as_ref().map(|a| a.arg.as_str());
    let find = |list: &[(String, &'a str)]| list.iter().find(|(f, _)| f == field).map(|(_, p)| *p);
    match classify_exception_init(c, &init.body, &params, vararg, kwarg) {
        Ok(model) => {
            if let Some(p) = find(&model.fields_after_super) {
                return Some((p, init, c_scope));
            }
            if model.super_init.is_some() {
                if let Some(found) = chain_field_store(&chain[pos + 1..], field) {
                    return Some(found);
                }
                find(&model.fields_at_super).map(|p| (p, init, c_scope))
            } else {
                find(&model.fields).map(|p| (p, init, c_scope))
            }
        }
        // An unmodeled body (the construction is refused at every site):
        // the last store of the field, for the read's type.
        Err(_) => {
            use crate::ast::tree::visit::{Descend, Flow, is_self, walk_stmts};
            let mut last: Option<&'a str> = None;
            walk_stmts(&init.body, Descend::OwnScope, &mut |stmt| {
                if let crate::StatementType::Assign(a) = &stmt.statement
                    && let [ExprType::Attribute(attr)] = a.targets.as_slice()
                    && attr.attr == field
                    && is_self(&attr.value)
                    && let ExprType::Name(v) = &a.value
                {
                    last = Some(v.id.as_str());
                }
                Flow::Continue
            });
            last.map(|p| (p, init, c_scope))
        }
    }
}

/// A BUILTIN exception's method resolution order, from the live
/// interpreter (`exception_tree::dump_builtin_exception_tree` — the same
/// dump the runtime's `BUILTIN_EXCEPTION_MRO` table is generated from):
/// `[the class's canonical name, then each ancestor]`, or None for a
/// name that is no builtin exception. One authority for the hierarchy
/// the issubclass fold walks — a hand-typed parent map drifted from the
/// generated table on day one (44 of 84 names missing, so
/// `issubclass(FileNotFoundError, OSError)` folded to `false`; Devin
/// review on #328 / the evaluation on issue #137).
/// An interpreter that cannot be run is an error, never an empty
/// hierarchy — an empty map would fold every `issubclass` of a builtin
/// pair to `false` and strip every alias, silently (Devin review on
/// #330).
pub(crate) fn builtin_exception_mro(
    name: &str,
) -> Result<Option<&'static [String]>, Box<dyn std::error::Error>> {
    static TREE: std::sync::OnceLock<
        Result<std::collections::HashMap<String, Vec<String>>, String>,
    > = std::sync::OnceLock::new();
    let tree = TREE
        .get_or_init(|| {
            crate::exception_tree::dump_builtin_exception_tree()
                .map(|(_, _, entries)| entries.into_iter().collect())
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| {
            format!(
                "rython: the builtin exception hierarchy comes from the live interpreter \
                 (`python3`), which could not be run: {e}"
            )
        })?;
    Ok(tree.get(name).map(|v| v.as_slice()))
}
