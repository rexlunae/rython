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
                    });
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
            let ancestors = exception_ancestor_tokens(&cls, &class_symbols, &options);
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
    Expr(crate::ExprType),
    Forwarded,
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
                            super_init = Some(SuperMessage::Forwarded);
                        } else if splats {
                            refusal = Some(format!(
                                "rython: `{}.__init__` calls `super().__init__` with a starred \
                                 argument that is not its own `*args`/`**kwargs`: the message \
                                 would not be the raise's argument, and rython does not model \
                                 the unpacking; pass one message expression",
                                cls.name
                            ));
                            return Flow::Stop;
                        } else if sc.args.len() > 1 || !sc.keywords.is_empty() {
                            // `BaseException.__init__(a, b)` sets `args`
                            // to the tuple and `str(e)` to ITS repr
                            // (`('a', 'b')`), which the one-message model
                            // does not render.
                            refusal = Some(format!(
                                "rython: `{}.__init__` calls `super().__init__` with more \
                                 than one argument: `str(e)` is then the repr of the args \
                                 tuple, which rython's one-message exception model does not \
                                 reproduce; pass one message and store the rest as fields",
                                cls.name
                            ));
                            return Flow::Stop;
                        } else {
                            super_init = Some(match sc.args.first() {
                                None => SuperMessage::Empty,
                                Some(m) => SuperMessage::Expr(m.clone()),
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
        None => Ok(InitModel { super_init, fields, fields_at_super }),
    }
}

/// A raise of a class whose `__init__` is variadic (`*args` / `**kwargs`):
/// nothing binds by name, so the body may only forward to
/// `super().__init__(*args)` (a docstring and `pass` aside); the message
/// is then the raise's one positional argument per `BaseException`, or
/// empty for none; a keyword (BaseException takes none — a TypeError in
/// CPython) and two or more positionals (the args tuple's repr) are loud.
fn variadic_exception_raise(
    cls: &crate::ClassDef,
    init: &crate::FunctionDef,
    call: &crate::Call,
    ctx: crate::CodeGenContext,
    options: crate::PythonOptions,
    symbols: crate::SymbolTableScopes,
    class_symbols: &crate::SymbolTableScopes,
) -> Result<Option<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
    let site_error = |msg: String| -> Result<Option<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
        Ok(Some(quote::quote!(compile_error!(#msg))))
    };
    let vararg = init.args.vararg.as_ref().map(|a| a.arg.as_str());
    let kwarg = init.args.kwarg.as_ref().map(|a| a.arg.as_str());
    // The NAMED parameters before `*args` (`__init__(self, prefix,
    // *args)`) take the first arguments; the forwarded slice is the rest
    // (Devin review on #330). A keyword-only parameter or a field store
    // of a named one is not modeled here: loud.
    let named: Vec<&str> = init
        .args
        .posonlyargs
        .iter()
        .chain(init.args.args.iter())
        .skip(1)
        .map(|a| a.arg.as_str())
        .collect();
    if !init.args.kwonlyargs.is_empty() {
        return site_error(format!(
            "rython: `{}.__init__` mixes `*args` with keyword-only parameters, which the \
             forwarding model does not bind; name the parameters, or drop `*args`",
            cls.name
        ));
    }
    let model = match classify_exception_init(cls, &init.body, &named, vararg, kwarg) {
        Ok(model) => model,
        Err(msg) => return site_error(msg),
    };
    if !model.fields.is_empty() {
        return site_error(format!(
            "rython: `{}.__init__` stores a named parameter as a field while forwarding \
             `*args`, which the forwarding model does not bind; name every parameter, \
             or drop `*args`",
            cls.name
        ));
    }
    if let Some(SuperMessage::Expr(_) | SuperMessage::Empty) = model.super_init {
        return site_error(format!(
            "rython: `{}.__init__` takes `*args` but does not forward them to \
             `super().__init__(*args)`: the message would not be the raise's argument; \
             forward the arguments, or name the parameters",
            cls.name
        ));
    }
    if !call.keywords.is_empty() {
        return site_error(format!(
            "rython: `{}()` takes no keyword arguments (BaseException.__init__ accepts \
             none)",
            cls.name
        ));
    }
    if call.args.iter().any(|a| matches!(a, crate::ExprType::Starred(_))) {
        return site_error(format!(
            "rython: `raise {}(*...)`: a starred argument to an exception constructor \
             is not modeled; pass the arguments explicitly",
            cls.name
        ));
    }
    if call.args.len() < named.len() {
        return site_error(format!(
            "rython: `{}()` missing a required argument: '{}'",
            cls.name,
            named[call.args.len()]
        ));
    }
    // The named parameters' arguments are evaluated once, in order, and
    // dropped (the body forwards only `*args`).
    let mut prelude: Vec<proc_macro2::TokenStream> = Vec::new();
    for a in &call.args[..named.len()] {
        if matches!(a, crate::ExprType::Constant(_)) {
            continue;
        }
        let tokens = a.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        prelude.push(quote::quote!(let _ = #tokens;));
    }
    let msg = match &call.args[named.len()..] {
        [] => quote::quote!(String::new()),
        [one] => {
            let m = message_arg(one, ctx, options.clone(), symbols)?;
            quote::quote!(format!("{}", #m))
        }
        many => {
            return site_error(format!(
                "rython: `{}.__init__` forwards its {} positional arguments to \
                 `super().__init__`, so `str(e)` is their tuple's repr, which rython's \
                 one-message exception model does not reproduce; pass one message",
                cls.name,
                many.len()
            ));
        }
    };
    let kind = &cls.name;
    let ancestor_tokens = exception_ancestor_tokens(cls, class_symbols, &options);
    let construct = quote::quote! {
        stdpython :: PyException :: new_with_attrs_and_ancestors (
            #kind , #msg , vec ! [] , vec ! [#(#ancestor_tokens),*]
        )
    };
    Ok(Some(if prelude.is_empty() {
        construct
    } else {
        quote::quote!({ #(#prelude)* #construct })
    }))
}

/// The class whose `__init__` a construction of `cls` runs — `cls`
/// itself or the first base up the chain defining one — with that
/// class's scope; None when no class of the chain defines one.
pub(crate) fn exception_init_owner(
    cls: &crate::ClassDef,
    scope: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    exception_mro(cls, scope, options)
        .into_iter()
        .filter_map(|(_, def)| def)
        .find(|(c, _)| c.init_method().is_some())
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
    let kind = &cls.name;
    if !call.keywords.is_empty() {
        let msg = format!(
            "rython: `{}()` takes no keyword arguments (BaseException.__init__ accepts none)",
            cls.name
        );
        return Ok(quote!(compile_error!(#msg)));
    }
    let msg = match call.args.as_slice() {
        [] => quote!(String::new()),
        [one] => {
            let m = message_arg(one, ctx, options.clone(), symbols)?;
            quote!(format!("{}", #m))
        }
        many => {
            // `str(e)` is `str(e.args)`: the tuple's repr — each
            // argument's repr, parenthesized and comma-joined
            // (`(<Pool object>, 'Pool is closed.')`; the default object
            // repr drops CPython's address, the documented §12.3
            // divergence). The old comma-joined display was a silent
            // divergence (the CI transcript on #330).
            let mut parts: Vec<TokenStream> = Vec::new();
            for a in many {
                let tokens = a.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                parts.push(quote!(stdpython::PyRepr::py_repr(&(#tokens))));
            }
            let fmt = format!("({})", vec!["{}"; parts.len()].join(", "));
            quote!(format!(#fmt, #(#parts),*))
        }
    };
    let ancestors = exception_ancestor_tokens(cls, class_symbols, &options);
    Ok(quote!(PyException::new_with_attrs_and_ancestors(
        #kind, #msg, vec![], vec![#(#ancestors),*]
    )))
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
) -> Vec<String> {
    exception_mro(cls, symbols, options)
        .into_iter()
        .skip(1)
        .map(|(name, _)| name)
        .collect()
}

/// A class's method resolution order as Python computes it (C3
/// linearization over EVERY base, left to right — `class Both(A, B)` is
/// caught by `except B:` too; following the first base alone dropped
/// every other branch; Devin review on #330): the class first, then each
/// ancestor by its canonical name with its definition and scope when the
/// crate defines it. A builtin base is a leaf (the runtime expands its
/// own MRO from the interpreter table); a base the crate does not define
/// and that is not a builtin ends its branch. An inconsistent hierarchy
/// (a TypeError at class creation in CPython) falls back to the
/// left-to-right depth-first order.
pub(crate) fn exception_mro(
    cls: &crate::ClassDef,
    scope: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Vec<(String, Option<(crate::ClassDef, crate::SymbolTableScopes)>)> {
    type Entry = (String, Option<(crate::ClassDef, crate::SymbolTableScopes)>);
    fn linearize(
        name: String,
        def: Option<(crate::ClassDef, crate::SymbolTableScopes)>,
        options: &crate::PythonOptions,
        depth: usize,
    ) -> Vec<Entry> {
        let Some((cls, scope)) = def else {
            return vec![(name, None)];
        };
        if depth > 32 {
            return vec![(name, Some((cls, scope)))];
        }
        // Each base by its canonical name, with its definition.
        let bases: Vec<Entry> = cls
            .bases
            .iter()
            .filter_map(|b| match b {
                crate::ExprType::Name(n) => canonical_exception_class(&n.id, &scope, options),
                _ => None,
            })
            .collect();
        let mut seqs: Vec<Vec<Entry>> = bases
            .iter()
            .map(|(n, d)| linearize(n.clone(), d.clone(), options, depth + 1))
            .collect();
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
                    // Inconsistent: depth-first, left to right, deduped.
                    for s in seqs.iter() {
                        for e in s {
                            if !out.iter().any(|(n, _)| *n == e.0) {
                                out.push(e.clone());
                            }
                        }
                    }
                    break;
                }
            }
        }
        out
    }
    linearize(cls.name.clone(), Some((cls.clone(), scope.clone())), options, 0)
}

/// The refusal for an exception name that is a runtime-ambiguous alias
/// (bound to more than one class at module level), or None.
pub(crate) fn ambiguous_alias_refusal(name: &str, options: &PythonOptions) -> Option<String> {
    crate::ast::tree::class_def::is_runtime_ambiguous_alias(name, &options.this_module_path).then(|| {
        format!(
            "rython: `{name}` is bound to more than one class at module level (a \
             `try:`/`except:` or an `if` the conversion cannot fold), so which exception \
             class it names is decided at runtime; rython refuses to follow one branch \
             silently — bind the name once"
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
) -> Vec<proc_macro2::TokenStream> {
    exception_ancestors(cls, symbols, options)
        .iter()
        .map(|a| quote::quote!((#a) . to_string ()))
        .collect()
}

/// A raise of an in-crate exception class whose `__init__` the model
/// covers: the class's own message (the `super().__init__(<message>)`
/// argument, rendered at the raise site with the parameters bound to the
/// call's arguments) and its field stores as attrs, plus the ancestor
/// chain. The model is exactly `self.<field> = <param>` stores and the
/// message; any other statement is a `compile_error!` at the raise site.
///
/// Arguments bind the way CPython binds a call: positionals to the
/// positional parameters in order, keywords by name to a positional or a
/// keyword-only parameter, a missing parameter to its default. Every
/// bound argument that is not a bare name or a constant is evaluated
/// ONCE, in source order, into a typed temporary the message and the
/// attrs both read (a property read or a call runs once — Devin review
/// on #330); a default is modeled only when it is a constant (a
/// definition-time value with an identity or an effect cannot be
/// re-rendered at each raise). What the model cannot bind is loud at the
/// site. None when the class has no `__init__` parameters to model (the
/// generic message-only construction, which still attaches the
/// ancestors).
pub(crate) fn exception_class_raise(
    cls: &crate::ClassDef,
    call: &crate::Call,
    ctx: crate::CodeGenContext,
    options: crate::PythonOptions,
    symbols: crate::SymbolTableScopes,
    class_symbols: &crate::SymbolTableScopes,
) -> Result<Option<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
    use crate::ExprType;
    use std::collections::HashMap;
    if !crate::is_exception_class(cls) {
        return Ok(None);
    }
    // The `__init__` that runs is the first one up the chain (Python's
    // MRO): a `class Child(Base): pass` constructs through Base's — the
    // kind stays Child and the ancestors are Child's (Devin review on
    // #330).
    let Some((owner, owner_scope)) = exception_init_owner(cls, class_symbols, &options) else {
        return Ok(None);
    };
    let init = owner.init_method().expect("the owner defines __init__");
    // The positional parameters are the positional-only ones followed by
    // the ordinary ones (CPython's `posonlyargs ++ args`); the receiver
    // is the first of that sequence wherever it sits, and a
    // positional-only parameter cannot be passed by keyword.
    let combined: Vec<&crate::ast::tree::arguments::Parameter> =
        init.args.posonlyargs.iter().chain(init.args.args.iter()).collect();
    let positional: Vec<&str> = combined.iter().skip(1).map(|a| a.arg.as_str()).collect();
    let positional_only: Vec<&str> = init
        .args
        .posonlyargs
        .iter()
        .skip(1)
        .map(|a| a.arg.as_str())
        .collect();
    let kwonly: Vec<&str> = init.args.kwonlyargs.iter().map(|a| a.arg.as_str()).collect();
    let site_error = |msg: String| -> Result<Option<proc_macro2::TokenStream>, Box<dyn std::error::Error>> {
        Ok(Some(quote::quote!(compile_error!(#msg))))
    };
    // A VARIADIC `__init__(self, *args, **kwargs)` binds nothing by name:
    // modeled when its body only forwards to `super().__init__(*args)`
    // (idna's IDNAError), the message then being the raise's one
    // positional argument — the BaseException rule below; any other
    // statement in it is the same refusal as an unmodeled body.
    if init.args.vararg.is_some() || init.args.kwarg.is_some() {
        return variadic_exception_raise(cls, init, call, ctx, options, symbols, class_symbols);
    }
    // Defaults: positional defaults align to the tail of the positional
    // parameters, keyword-only defaults to their parameters.
    let mut defaults: HashMap<&str, &ExprType> = HashMap::new();
    let n_all = combined.len();
    let n_def = init.args.defaults.len();
    if n_def > n_all {
        return Ok(None);
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
    let mut given: Vec<(&str, &ExprType)> = Vec::new();
    for (i, a) in call.args.iter().enumerate() {
        if matches!(a, ExprType::Starred(_)) {
            return site_error(format!(
                "rython: `raise {}(*...)`: a starred argument to an exception constructor \
                 is not modeled; pass the arguments explicitly",
                cls.name
            ));
        }
        let Some(p) = positional.get(i) else {
            return site_error(format!(
                "rython: `{}()` takes {} positional argument(s) but {} were given",
                cls.name,
                positional.len(),
                call.args.len()
            ));
        };
        given.push((p, a));
    }
    for kw in &call.keywords {
        let Some(name) = kw.arg.as_deref() else {
            return site_error(format!(
                "rython: `raise {}(**...)`: a keyword splat to an exception constructor is \
                 not modeled; pass the arguments explicitly",
                cls.name
            ));
        };
        if positional_only.contains(&name) {
            return site_error(format!(
                "rython: `{}()` got some positional-only arguments passed as keyword \
                 arguments: '{}'",
                cls.name, name
            ));
        }
        if !positional.contains(&name) && !kwonly.contains(&name) {
            return site_error(format!(
                "rython: `{}()` got an unexpected keyword argument '{}'",
                cls.name, name
            ));
        }
        if given.iter().any(|(p, _)| *p == name) {
            return site_error(format!(
                "rython: `{}()` got multiple values for argument '{}'",
                cls.name, name
            ));
        }
        given.push((name, &kw.value));
    }
    // Every parameter's substitution at the raise site: a temporary for
    // an evaluated argument, the expression itself for a bare name or a
    // constant, the constant default otherwise.
    let mut prelude: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut msg_options = options.clone();
    let mut name_types = (*msg_options.name_types).clone();
    let mut substitution: HashMap<&str, ExprType> = HashMap::new();
    let evaluated = |a: &ExprType| !matches!(a, ExprType::Name(_) | ExprType::Constant(_));
    for (k, (param, arg)) in given.iter().enumerate() {
        // A bare name is read in place — unless a LATER argument runs
        // code, which could rebind it before the message and the attrs
        // read it (`E(counter, bump())`): then it is captured in source
        // order too, a clone (the raise site may read it again — an
        // `isinstance(E(x, f()), ...)` construction; Devin review on
        // #330).
        let captured_name = matches!(arg, ExprType::Name(_))
            && given[k + 1..].iter().any(|(_, later)| evaluated(later));
        if !evaluated(arg) && !captured_name {
            substitution.insert(param, (*arg).clone());
            continue;
        }
        let temp = format!("__rython_exc_arg{k}");
        let ident = proc_macro2::Ident::new(&temp, proc_macro2::Span::call_site());
        let tokens = (*arg).clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if captured_name {
            prelude.push(quote::quote!(let #ident = (#tokens).clone();));
        } else {
            prelude.push(quote::quote!(let #ident = #tokens;));
        }
        let ty = crate::infer_type(Some(&ctx), arg, &options, &symbols);
        if !matches!(ty, crate::TypeInfo::PyObject) {
            name_types.insert(temp.clone(), ty);
        }
        substitution.insert(param, ExprType::Name(crate::ast::tree::name::Name { id: temp }));
    }
    for param in positional.iter().chain(kwonly.iter()) {
        if substitution.contains_key(param) {
            continue;
        }
        let Some(d) = defaults.get(param) else {
            return site_error(format!(
                "rython: `{}()` missing a required argument: '{}'",
                cls.name, param
            ));
        };
        let constant = matches!(d, ExprType::Constant(_) | ExprType::NoneType(_))
            || matches!(d, ExprType::UnaryOp(u) if matches!(u.operand.as_ref(), ExprType::Constant(_)));
        if !constant {
            return site_error(format!(
                "rython: `{}.__init__`'s default for '{}' is not a constant: CPython \
                 evaluates a default once at definition time, which re-rendering it at \
                 each raise cannot reproduce; pass the argument explicitly, or make the \
                 default a constant",
                cls.name, param
            ));
        }
        substitution.insert(param, (*d).clone());
    }
    msg_options.name_types = std::rc::Rc::new(name_types);
    // The __init__ body — `self.<field> = <param>` stores and the
    // `super().__init__(<message>)` call; a docstring and `pass` are
    // inert; anything else would not run at the raise site — classified
    // through the one statement visitor (a compound statement is
    // unmodeled at its head, so its bodies are never entered).
    let params: Vec<&str> = positional.iter().chain(kwonly.iter()).copied().collect();
    let InitModel { super_init, fields, fields_at_super } =
        match classify_exception_init(cls, &init.body, &params, None, None) {
            Ok(model) => model,
            Err(msg) => return site_error(msg),
        };
    // The message, as CPython sets `str(e)`: the `super().__init__`
    // argument; the empty string for the zero-argument call (a class
    // that stores fields and calls `super().__init__()` keeps its fields
    // — Devin review on #330); with no super call at all,
    // `BaseException.__new__` still records the call's positional
    // arguments as `args`, so `str(e)` is the one positional argument,
    // or empty for none — more than one is the tuple's repr, refused.
    let msg_expr: Option<ExprType> = match super_init {
        Some(SuperMessage::Empty) => None,
        Some(SuperMessage::Expr(m)) => Some(m),
        // `super().__init__(*args)` cannot appear here: the variadic
        // initializer took the branch above, and a non-variadic body
        // naming `*args` is refused by the classifier.
        Some(SuperMessage::Forwarded) => unreachable!("forwarding needs a vararg"),
        None => match call.args.len() {
            0 => None,
            1 => Some(ExprType::Name(crate::ast::tree::name::Name {
                id: positional[0].to_string(),
            })),
            _ => {
                return site_error(format!(
                    "rython: `{}.__init__` never calls `super().__init__`, so `str(e)` is \
                     the repr of the {} positional arguments as a tuple, which rython's \
                     one-message exception model does not reproduce; call \
                     `super().__init__(<message>)`",
                    cls.name,
                    call.args.len()
                ));
            }
        },
    };
    // Every argument is evaluated, in source order, whether or not the
    // model reads it: a bare name bound to a parameter the message and
    // the fields ignore is still read once (`raise E(undefined_name)`
    // is a NameError before the construction in CPython, an unresolved
    // name in rustc here — Devin review on #330).
    let mut read_by_model: Vec<&str> = fields.iter().map(|(_, p)| *p).collect();
    if let Some(m0) = &msg_expr {
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
    for (param, arg) in &given {
        if let ExprType::Name(n) = arg
            && !read_by_model.contains(param)
            && matches!(substitution.get(param), Some(ExprType::Name(s)) if s.id == n.id)
        {
            let ident = crate::safe_ident(&n.id);
            prelude.push(quote::quote!(let _ = &#ident;));
        }
    }
    let msg = match msg_expr {
        None => quote::quote!(String::new()),
        Some(mut m) => {
            // The rewrite binds names by the __init__'s OWN scope: a
            // lambda or a comprehension in the message would bind names
            // of its own, which the parameter substitution must not touch
            // and the raise site cannot re-bind; refused.
            if crate::ast::tree::visit::any_expr_for(&m, crate::ast::tree::visit::Descend::All, |e| {
                matches!(
                    e,
                    ExprType::Lambda(_)
                        | ExprType::ListComp(_)
                        | ExprType::SetComp(_)
                        | ExprType::DictComp(_)
                        | ExprType::GeneratorExp(_)
                )
            }) {
                return site_error(format!(
                    "rython: `{}.__init__`'s message holds a lambda or a comprehension, \
                     whose own bindings the raise-site rewrite cannot model; compute the \
                     message in a local first",
                    cls.name
                ));
            }
            // The message at the raise site: a `self.<field>` read means
            // the parameter the field stores, and each parameter means
            // its substitution — one pre-order pass through the mutable
            // visitor; a node the pass replaces is a leaf (a name, a
            // constant, a temporary), never rewritten again, so a caller
            // name that happens to match another parameter binds ONCE
            // (`E(b, "second")` for `__init__(self, a, b)` reads the
            // caller's `b` — Devin review on #330).
            crate::ast::tree::visit::walk_expr_mut(&mut m, &mut |e| {
                if let ExprType::Attribute(attr) = e
                    && crate::ast::tree::visit::is_self(&attr.value)
                    && let Some((_, param)) = fields_at_super.iter().find(|(f, _)| *f == attr.attr)
                {
                    *e = ExprType::Name(crate::ast::tree::name::Name { id: param.to_string() });
                }
                if let ExprType::Name(n) = e
                    && let Some(sub) = substitution.get(n.id.as_str())
                {
                    *e = sub.clone();
                }
            });
            // The message renders at the raise site, so every name it
            // still reads — a module global of the defining module, a
            // builtin — must be bound there exactly as the `__init__`
            // sees it; a caller local or a same-named global of another
            // module would silently change the message (Devin review on
            // #330). Refused otherwise.
            // A bare-name ARGUMENT inlined for a parameter is the raise
            // site's own name, read there by design.
            let site_names: Vec<&str> = substitution
                .values()
                .filter_map(|v| match v {
                    ExprType::Name(n) => Some(n.id.as_str()),
                    _ => None,
                })
                .collect();
            let mut free: Vec<String> = Vec::new();
            crate::ast::tree::visit::walk_expr(&m, &mut |e| {
                if let ExprType::Name(n) = e
                    && !n.id.starts_with("__rython_exc_arg")
                    && !site_names.contains(&n.id.as_str())
                    && !free.contains(&n.id)
                {
                    free.push(n.id.clone());
                }
            });
            for name in free {
                let at_def = format!("{:?}", owner_scope.module_get(&name));
                let at_site = format!("{:?}", symbols.get(&name));
                // A raise-site LOCAL of the name (a parameter or a local
                // store — the function's typed names) shadows the global.
                let local = options.name_types.contains_key(&name)
                    && owner_scope.module_get(&name).is_some();
                if at_def != at_site || local {
                    return site_error(format!(
                        "rython: `{}.__init__`'s message reads `{}`, which at this raise \
                         site is not the binding the `__init__` sees (a local, or another \
                         module's global of the same name): rython renders the message at \
                         the raise site and refuses to silently read the wrong one; pass \
                         the value as an argument, or store it as a field",
                        cls.name, name
                    ));
                }
            }
            let msg_tok = m.to_rust(ctx.clone(), msg_options.clone(), symbols.clone())?;
            quote::quote!(format!("{}", #msg_tok))
        }
    };
    // The attrs: each field's substitution, boxed.
    let kind = &cls.name;
    let mut attr_pairs: Vec<proc_macro2::TokenStream> = Vec::new();
    for (f, param) in &fields {
        let value = &substitution[param];
        // The attrs are BOXED values: a class instance (`raise
        // EmptyPoolError(self, ...)` storing `self.pool = pool` —
        // urllib3's connectionpool) has no box, and dropping the field
        // would make `e.pool` a silent AttributeError; refused at the
        // site instead.
        // A temporary's type lives in msg_options (the raise-site
        // binding), so an evaluated construction or a captured name is
        // seen too (Devin review on #330).
        let instance = crate::ast::tree::visit::is_self(value)
            || matches!(
                crate::infer_type(Some(&ctx), value, &msg_options, &symbols),
                crate::TypeInfo::Class(_)
            );
        if instance {
            return site_error(format!(
                "rython: `{}.__init__` stores its parameter `{}` as `self.{}`, and the \
                 argument here is a class instance, which an exception's boxed attrs \
                 cannot hold; rython refuses to silently drop the field. Store a plain \
                 value (a name, an id) instead",
                cls.name, param, f
            ));
        }
        let boxed = if crate::is_none_expr(value) {
            quote::quote!(stdpython :: PyValue :: None_)
        } else {
            let a = value.clone().to_rust(ctx.clone(), msg_options.clone(), symbols.clone())?;
            // A temporary is read by the message too: clone it into the box.
            if matches!(value, ExprType::Name(n) if n.id.starts_with("__rython_exc_arg")) {
                quote::quote!(stdpython :: PyValue :: from ((#a).clone()))
            } else {
                quote::quote!(stdpython :: PyValue :: from (#a))
            }
        };
        attr_pairs.push(quote::quote!((#f . to_string (), #boxed)));
    }
    let ancestor_tokens = exception_ancestor_tokens(cls, class_symbols, &options);
    let construct = quote::quote! {
        stdpython :: PyException :: new_with_attrs_and_ancestors (
            #kind , #msg , vec ! [#(#attr_pairs),*] ,
            vec ! [#(#ancestor_tokens),*]
        )
    };
    if prelude.is_empty() {
        Ok(Some(construct))
    } else {
        Ok(Some(quote::quote!({ #(#prelude)* #construct })))
    }
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
) -> Option<crate::TypeInfo> {
    // Walk the class's MRO: the field may be defined on a BASE (an
    // `except BankError as e` catching an InsufficientFunds whose
    // __init__ stores the field — Devin review on #328), local or
    // imported, named directly or through an alias, on any branch of a
    // multiple inheritance — the one linearization, each hop's
    // annotation resolved in its own scope (Devin review on #330).
    for (_, def) in exception_mro(cls, symbols, options) {
        let Some((c, c_scope)) = def else { continue };
        if let Some(init) = c.init_method() {
            for stmt in &init.body {
                if let crate::StatementType::Assign(a) = &stmt.statement
                    && let [ExprType::Attribute(attr)] = a.targets.as_slice()
                    && attr.attr == field
                    && let ExprType::Name(r) = attr.value.as_ref()
                    && r.id == "self"
                    && let ExprType::Name(v) = &a.value
                {
                    // The param's annotation types the field — a
                    // positional-only, positional, or keyword-only
                    // parameter.
                    let param = init
                        .args
                        .posonlyargs
                        .iter()
                        .chain(init.args.args.iter())
                        .chain(init.args.kwonlyargs.iter())
                        .find(|p| p.arg == v.id)?;
                    if let Some(ann) = param.annotation.as_ref()
                        && let Some(t) = crate::resolve_alias_typeinfo(ann, &c_scope, options)
                    {
                        return Some(t);
                    }
                    // An unannotated param: no type — the reader refuses
                    // loudly rather than guessing (an Int guess made a
                    // str field's read a false AttributeError).
                    return Some(crate::TypeInfo::PyObject);
                }
            }
        }
    }
    None
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
