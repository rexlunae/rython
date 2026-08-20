//! The systematic decorator registry.
//!
//! Python decorators are compile-time definition transformers — the same
//! role Rust's attributes and derive macros play. rython consumes the
//! decorator EXPRESSION at conversion time and rewrites the definition it
//! decorates. Because rython's model cannot pass function objects (the
//! documented function-as-value divergence, issue #122), only a FIXED,
//! compiler-known set of structural decorators is supported; any other
//! decorator is a loud conversion error, never silently dropped.
//!
//! One parser (`parse_decorator`) handles every spelling form (`Name`,
//! `Call`, `functools.` attribute) for every consumer:
//!
//! - `@classmethod` / `@staticmethod` — method shape (no receiver; a
//!   classmethod drops its first parameter).
//! - `@functools.lru_cache(maxsize=N)` / `@functools.cache` — cache the
//!   function's results in a static `PyLruCache`.
//! - `@dataclass` — synthesize `__init__` from annotated class fields.
//! - The decorator-FACTORY expression `name = lru_cache(maxsize=N)(fn)` —
//!   the same cache rewrite applied to an expression (issue #127).
//!
//! Every variant carries a human `describe()` for the "unsupported
//! decorator" error, so the message is consistent across consumers.

use proc_macro2::TokenStream;
use quote::quote;

use crate::ast::tree::function_def::MethodDecorator;
use crate::{ExprType, SymbolTableScopes};

/// A supported, compiler-known decorator.
#[derive(Clone, Debug, PartialEq)]
pub enum Decorator {
    /// `@classmethod` — an associated function with no receiver; the
    /// first parameter (cls) is dropped.
    ClassMethod,
    /// `@staticmethod` — an associated function with no receiver.
    StaticMethod,
    /// `@functools.lru_cache(maxsize=N)` / `@functools.cache`. The spec is
    /// `Some(Some(n))` for a bounded LRU, `Some(None)` for an unbounded
    /// cache, `None` for "not present" (callers default it).
    Cache(Option<Option<i64>>),
    /// `@dataclass` (with optional `(frozen=..., slots=...)` args, all
    /// no-ops in the value model) — synthesizes `__init__` from annotated
    /// class fields.
    DataClass,
}

impl Decorator {
    /// The stable name shown in "unsupported decorator" errors.
    pub fn describe(&self) -> &'static str {
        match self {
            Decorator::ClassMethod => "classmethod",
            Decorator::StaticMethod => "staticmethod",
            Decorator::Cache(_) => "functools.lru_cache",
            Decorator::DataClass => "dataclass",
        }
    }

    /// The `@classmethod`/`@staticmethod`/cache shape a FUNCTION definition
    /// takes, for the function codegen. `None` for decorators that do not
    /// apply to functions.
    pub fn as_method_decorator(&self) -> Option<MethodDecorator> {
        match self {
            Decorator::ClassMethod => Some(MethodDecorator::ClassMethod),
            Decorator::StaticMethod => Some(MethodDecorator::StaticMethod),
            Decorator::Cache(spec) => Some(MethodDecorator::Cache(*spec)),
            Decorator::DataClass => None,
        }
    }

    /// The lru_cache maxsize spec, for callers that only care about
    /// caching (`None` when this is not a cache decorator).
    pub fn cache_spec(&self) -> Option<Option<i64>> {
        match self {
            Decorator::Cache(spec) => *spec,
            _ => None,
        }
    }
}

/// The one decorator parser. Accepts `Name`, `Call`, and `functools.`
/// attribute spellings of the supported set; anything else is a loud
/// error with a consistent message. `None` means "no decorators".
pub fn parse_decorator(
    decorators: &[ExprType],
) -> Result<Option<Decorator>, Box<dyn std::error::Error>> {
    let unsupported = |what: &str| -> Box<dyn std::error::Error> {
        format!(
            "decorator `{}` is not supported yet (only functools.lru_cache, \
             functools.cache, classmethod, staticmethod, and dataclass are); \
             rython refuses to silently ignore it",
            what
        )
        .into()
    };
    // `functools.lru_cache` / `functools.cache` / bare `lru_cache`/`cache`.
    let name_of = |e: &ExprType| -> Option<String> {
        match e {
            ExprType::Name(n) => Some(n.id.clone()),
            ExprType::Attribute(a) => match a.value.as_ref() {
                ExprType::Name(m) if m.id == "functools" => Some(a.attr.clone()),
                _ => None,
            },
            _ => None,
        }
    };
    match decorators {
        [] => Ok(None),
        [single] => {
            let (base, call) = match single {
                ExprType::Call(c) => (name_of(c.func.as_ref()), Some(c)),
                other => (name_of(other), None),
            };
            match (base.as_deref(), call) {
                (Some("classmethod"), None) => Ok(Some(Decorator::ClassMethod)),
                (Some("staticmethod"), None) => Ok(Some(Decorator::StaticMethod)),
                (Some("cache"), None) => Ok(Some(Decorator::Cache(Some(None)))),
                (Some("cache"), Some(c)) if c.args.is_empty() && c.keywords.is_empty() => {
                    Ok(Some(Decorator::Cache(Some(None))))
                }
                (Some("lru_cache"), None) => Ok(Some(Decorator::Cache(Some(Some(128))))),
                (Some("lru_cache"), Some(c)) => {
                    let maxsize = match (c.args.as_slice(), c.keywords.as_slice()) {
                        ([], []) => None,
                        ([e], []) => Some(e.clone()),
                        ([], [kw]) if kw.arg.as_deref() == Some("maxsize") => {
                            Some(kw.value.clone())
                        }
                        _ => {
                            return Err(
                                "lru_cache() takes at most a single maxsize argument"
                                    .to_string()
                                    .into(),
                            )
                        }
                    };
                    match maxsize {
                        None => Ok(Some(Decorator::Cache(Some(Some(128))))),
                        Some(m) if crate::is_none_expr(&m) => {
                            Ok(Some(Decorator::Cache(Some(None))))
                        }
                        Some(ExprType::Constant(c))
                            if matches!(&c.0, Some(litrs::Literal::Integer(_))) =>
                        {
                            let lit = c.0.clone().expect("matched integer");
                            let n: i64 = match &lit {
                                litrs::Literal::Integer(i) => i
                                    .value()
                                    .ok_or("lru_cache maxsize out of range")?,
                                _ => unreachable!(),
                            };
                            Ok(Some(Decorator::Cache(Some(Some(n)))))
                        }
                        Some(_) => Err(
                            "lru_cache maxsize must be an integer literal or None"
                                .to_string()
                                .into(),
                        ),
                    }
                }
                // `@dataclass` / `@dataclass(frozen=True, slots=True)` —
                // the args are accepted and treated as no-ops (the Rust
                // struct is already value-semantics).
                (Some("dataclass"), None) => Ok(Some(Decorator::DataClass)),
                (Some("dataclass"), Some(_)) => Ok(Some(Decorator::DataClass)),
                _ => Err(unsupported(&format!("{:?}", single))),
            }
        }
        many => Err(unsupported(&format!("{:?}", many[0]))),
    }
}

/// Whether the decorator expression is the `@dataclass` spelling
/// (used by class codegen before full parsing).
pub fn is_dataclass_decorator(d: &ExprType) -> bool {
    match d {
        ExprType::Name(n) => n.id == "dataclass",
        ExprType::Call(c) => {
            matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "dataclass")
        }
        _ => false,
    }
}

/// Render a decorator expression back as Rust tokens (used for the
/// synthesized-wrapper decorator in the lru_cache-factory rewrite).
pub fn decorator_to_tokens(d: &Decorator) -> TokenStream {
    match d {
        Decorator::ClassMethod => quote!(classmethod),
        Decorator::StaticMethod => quote!(staticmethod),
        Decorator::Cache(spec) => match spec {
            Some(None) => quote!(cache),
            Some(Some(n)) => quote!(lru_cache(#n)),
            None => quote!(lru_cache),
        },
        Decorator::DataClass => quote!(dataclass),
    }
}

/// Issue #127: the decorator-FACTORY-as-expression form
/// `name = lru_cache(maxsize=N)(fn)` (charset_normalizer's
/// `cached_mess_ratio = lru_cache(maxsize=None)(mess_ratio)`). Returns a
/// synthesized cached-wrapper FunctionDef (decorated with the same cache
/// spec, body calls fn) when the assignment value is exactly that shape
/// and fn resolves to a known function (same module or cross-module,
/// issue #123). None for any other assignment.
pub fn try_lru_cache_factory(
    assign: &crate::Assign,
    options: Option<&crate::PythonOptions>,
    symbols: &SymbolTableScopes,
) -> Option<crate::FunctionDef> {
    let [crate::ExprType::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    // value = Call(decorator_call, [fn_name])
    let crate::ExprType::Call(outer) = &assign.value else {
        return None;
    };
    if outer.args.len() != 1 || !outer.keywords.is_empty() {
        return None;
    }
    let crate::ExprType::Name(fn_name) = outer.args.first()? else {
        return None;
    };
    // The decorator call: lru_cache(...) / cache(...) / bare Name.
    let decorator = match parse_decorator(std::slice::from_ref(outer.func.as_ref())) {
        Ok(Some(d)) => d,
        _ => return None,
    };
    let Decorator::Cache(_) = decorator else {
        return None;
    };
    // Resolve fn's FunctionDef: same module, or cross-module when options
    // are available (issue #123).
    let fdef = match symbols.get(&fn_name.id) {
        Some(crate::SymbolTableNode::FunctionDef(f)) => f.clone(),
        Some(crate::SymbolTableNode::ImportFrom(i)) => {
            let options = options?;
            let path = i.resolved_module_path(&options);
            if options.module_defs.contains_key(&path) {
                let (f, _) = crate::module_function_def(&options, &path, &fn_name.id)?;
                f
            } else {
                return None;
            }
        }
        _ => return None,
    };
    let param_names: Vec<String> = fdef
        .args
        .posonlyargs
        .iter()
        .chain(fdef.args.args.iter())
        .chain(fdef.args.kwonlyargs.iter())
        .map(|p| p.arg.clone())
        .collect();
    // body: return fn(param, param, ...)
    let call_args: Vec<crate::ExprType> = param_names
        .iter()
        .map(|p| crate::ExprType::Name(crate::ast::tree::name::Name { id: p.clone() }))
        .collect();
    let call = crate::ExprType::Call(crate::Call {
        func: Box::new(crate::ExprType::Name(crate::ast::tree::name::Name {
            id: fn_name.id.clone(),
        })),
        args: call_args,
        keywords: Vec::new(),
    });
    let body = vec![crate::Statement {
        statement: crate::StatementType::Return(Some(crate::ast::tree::expression::Expr {
            value: call,
            ctx: None,
            lineno: None,
            col_offset: None,
            end_lineno: None,
            end_col_offset: None,
        })),
        lineno: None,
        col_offset: None,
        end_lineno: None,
        end_col_offset: None,
    }];
    // Reuse the original decorator expression (lru_cache(maxsize=None)) so
    // the synthesized function gets exactly the same cache spec.
    Some(crate::FunctionDef {
        name: target.id.clone(),
        args: fdef.args.clone(),
        body,
        decorator_list: vec![outer.func.as_ref().clone()],
        returns: fdef.returns.clone(),
    })
}
