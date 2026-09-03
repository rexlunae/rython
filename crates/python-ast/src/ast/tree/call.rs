use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Keyword, PythonOptions, SymbolTableNode, SymbolTableScopes,
    extract_required_attr,
};
use crate::ast::tree::std_module::variant as rt_variant;

/// The type names isinstance() lowers against (a PyValue predicate or a
/// static verdict exists for each). ONE list for the three consumers —
/// the alias resolver, the single-Name target, and the tuple-of-types
/// target — which previously kept three copies, one of them missing
/// `frozenset` (a `MySet = frozenset` alias then failed to resolve and
/// the check silently lowered to `false`).
const ISINSTANCE_TARGET_NAMES: &[&str] = &[
    "int",
    "float",
    "str",
    "bool",
    "bytes",
    "bytearray",
    "PathLike",
    "BinaryIO",
    "tuple",
    "dict",
    "list",
    "set",
    "frozenset",
    "Mapping",
    "Iterable",
    "Sequence",
];

/// Runtime-module functions that return `Result<T, PyException>` because
/// they can raise like their Python counterparts. A call through a module
/// path (`math.sqrt(x)`, `json.loads(s)`) into one of these threads `?`
/// (see `propagates_exceptions`), so the exception stays catchable instead
/// of surfacing as a rustc type error in the generated crate. The set must
/// mirror the Result-returning `python_function!` blocks in stdpython.
const FALLIBLE_STDLIB_FN: &[&str] = &[
    // math: domain/range errors and overflow.
    "sqrt", "pow", "log", "log2", "log10", "log1p", "asin", "acos", "acosh", "atanh",
    "factorial", "fmod", "remainder", "ldexp",
    // json: parse errors.
    "loads",
    // glob: filesystem access can fail.
    "glob", "rglob", "iglob",
    // os: entropy source can fail (os.urandom raises OSError).
    "urandom",
    // socket.socket() rejects unknown families/kinds with OSError.
    "socket",
    // socket.getaddrinfo raises gaierror (an OSError) on resolution failure.
    "getaddrinfo",
    // socket.SocketIO / ssl.MemoryBIO construction raises
    // NotImplementedError: makefile()-over-wrapped-socket and TLS-in-TLS
    // BIOs are not modeled by the runtime (documented divergence) — the
    // constructors are loud, catchable stubs.
    "SocketIO",
    "MemoryBIO",
    // urllib.request.urlopen raises URLError/HTTPError.
    "urlopen",
];

/// Issue #111: keyword-argument signatures of stdpython runtime functions
/// (module root, function, positional parameter names in order). Calls
/// render through these signatures: keywords map to their slots, and
/// omitted trailing parameters fill with `None` — Python's warnings API is
/// the flagship (its category parameters are CLASSES, which rython cannot
/// pass as values, so those slots stay generic).
pub(crate) const RUNTIME_KEYWORD_SIGNATURES: &[(&str, &str, &[&str])] = &[
    (
        "warnings",
        "simplefilter",
        &["action", "category", "module", "lineno", "append"],
    ),
    (
        "warnings",
        "filterwarnings",
        &["action", "message", "category", "module", "lineno", "append"],
    ),
    (
        "warnings",
        "warn",
        &["message", "category", "stacklevel", "source"],
    ),
    (
        "warnings",
        "warn_explicit",
        &[
            "message",
            "category",
            "filename",
            "lineno",
            "module",
            "registry",
            "module_globals",
            "source",
        ],
    ),
];

/// Strip a trailing `?` from a rendered call (`f()?` → `f()`), so the `?`
/// can be re-applied AFTER an `.await`: an async function call renders with
/// `?` (exceptions propagate), but the operator must unwrap the awaited
/// Result, not the future. Mirrors the Await node's reordering.
///
/// The rendered value may be a BLOCK — a class construction lowers to
/// `{ prelude Klass::new(..)? }` (issue #229) — so the trailing `?` can
/// sit INSIDE a brace group. A block whose last statement ends with `?`
/// strips there: the block still evaluates to the (now un-unwrapped)
/// value, exactly like the bare-call form.
pub(crate) fn strip_trailing_question(tokens: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    use proc_macro2::{Delimiter, Group, TokenTree};

    fn strip_last(stream: proc_macro2::TokenStream) -> Option<proc_macro2::TokenStream> {
        let mut trees: Vec<TokenTree> = stream.into_iter().collect();
        match trees.last() {
            Some(TokenTree::Punct(p)) if p.as_char() == '?' => {
                trees.pop();
                Some(trees.into_iter().collect())
            }
            _ => None,
        }
    }

    if let Some(stripped) = strip_last(tokens.clone()) {
        return stripped;
    }
    // A single `{ ... }` block whose last statement ends with `?`: strip
    // inside the braces and keep the block (its prelude statements, if
    // any, still run).
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    if trees.len() == 1
        && let TokenTree::Group(g) = &trees[0]
        && g.delimiter() == Delimiter::Brace
        && let Some(inner) = strip_last(g.stream())
    {
        let mut block = Group::new(Delimiter::Brace, inner);
        block.set_span(g.span());
        return quote!(#block);
    }
    tokens.clone()
}

/// A Name resolving (through ImportFrom re-export chains) to a module-level
/// LITERAL constant (`DEFAULT_POOLSIZE = 10` — requests/adapters, used as a
/// dropped DEFAULT in sessions.py's `HTTPAdapter()` call): render the
/// constant's VALUE tokens — the call site does not import the name. Only
/// literal values inline (anything else keeps the bare name).
fn resolve_constant_name(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<TokenStream> {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    // The crate module the chain last crossed into (None while still in
    // the starting scope).
    let mut defining_module: Option<Vec<String>> = None;
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let path = ifm.resolved_module_path(options);
                let Some(key) = crate::module_defs_key(options, &path) else {
                    return None;
                };
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[key];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
                defining_module = Some(key.to_vec());
            }
            Some(SymbolTableNode::Assign { value, .. }) => {
                // Literal-only: the value must be a constant the caller
                // can inline without the defining module's context.
                if crate::ast::tree::module::const_static_type(value).is_some() {
                    let rendered = value.clone().to_rust(
                        CodeGenContext::Module("constant".to_string()),
                        options.clone(),
                        syms.clone(),
                    );
                    return rendered.ok();
                }
                // A COMPUTED module constant of ANOTHER crate module
                // (`timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT` — urllib3's
                // connectionpool, whose default is util/timeout.py's
                // `_DEFAULT_TIMEOUT = _TYPE_DEFAULT.token`): the defining
                // module emits it as a promoted LazyLock static (the same
                // authority its own reads consult), so a caller that does
                // not import the name reads the static by crate path.
                if let Some(path) = &defining_module
                    && crate::ast::tree::module::module_promoted_static_names(options, path)
                        .contains(&current)
                {
                    let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
                    let ident = crate::safe_ident(&current);
                    return Some(quote!((*crate #(::#segs)* :: #ident).clone()));
                }
                return None;
            }
            _ => return None,
        }
    }
    None
}

/// Follow a compat builtin-alias import chain (`from .compat import
/// builtin_str` where compat does `builtin_str = str`): the name resolves to
/// one of the builtin type names (str/bytes/int/float/bool), which the call
/// re-dispatches to. None when the chain does not end in a builtin alias.
fn resolve_builtin_alias(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<String> {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let path = ifm.resolved_module_path(options);
                let Some(key) = crate::module_defs_key(options, &path) else {
                    return None;
                };
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[key];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
            }
            Some(SymbolTableNode::Assign { value, .. }) => {
                if let ExprType::Name(b) = value {
                    let b = b.id.as_str();
                    // A SELF-alias (`str = str` — requests' compat) would
                    // re-dispatch to the same name forever.
                    if b == current {
                        return None;
                    }
                    if crate::ast::tree::assign::is_builtin_scalar_name(b) {
                        return Some(b.to_string());
                    }
                }
                return None;
            }
            _ => return None,
        }
    }
    None
}

/// Follow an ImportFrom re-export chain (`from .compat import OrderedDict`
/// where compat does `from collections import OrderedDict`) through the
/// generated crate's modules; returns the terminal STDPYTHON
/// (module root, item name) the chain ends in (`("collections",
/// "OrderedDict")`), or None when the chain does not end in a stdpython
/// module. The caller checks `stdpython_module_class(root, name)` — the
/// re-exported item may be a FUNCTION (requests' compat re-exports
/// `urlparse` from urllib.parse; calls must NOT lower as constructions).
fn stdpython_reexport_chain(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(String, String)> {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let path = ifm.resolved_module_path(options);
                let Some(key) = crate::module_defs_key(options, &path) else {
                    // Terminal hop: the import binds into a stdpython (or
                    // external) module — `current` is the item name there.
                    let root = ifm.module.split('.').next().unwrap_or("").to_string();
                    return crate::is_stdpython_module(&root).then(|| (root, current.clone()));
                };
                // Re-export chain: hop into the defining module's scope.
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[key];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
            }
            _ => return None,
        }
    }
    None
}

/// The urllib.parse function an imported Name callee resolves to
/// (`urlparse`, `urlsplit`, ...), following the same re-export chain as
/// stdpython_reexport_chain (requests' compat re-exports them from
/// `urllib.parse`). Returns the canonical runtime item name, or None
/// when the name does not resolve to a urllib.parse function.
fn urllib_parse_fn(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<&'static str> {
    const PARSE_FNS: &[&str] = &[
        "urlparse",
        "urlsplit",
        "urlunparse",
        "urljoin",
        "urlencode",
        "quote",
        "quote_plus",
        "unquote",
        "unquote_plus",
        "urldefrag",
    ];
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                // Direct `from urllib.parse import urlparse`: the item
                // may be ALIASED (`import urlparse as up`). Owned copy so
                // the borrow of the scope ends before the return.
                let root = ifm.module.split('.').next().unwrap_or("");
                let canonical: String = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                if crate::StdModule::from_name(&root) == Some(crate::StdModule::Urllib) {
                    if let Some(f) = PARSE_FNS.iter().copied().find(|f| *f == canonical.as_str()) {
                        return Some(f);
                    }
                }
                // Re-export chain through a sibling module (requests'
                // compat): hop into the defining module's scope.
                let path = ifm.resolved_module_path(options);
                let Some(key) = crate::module_defs_key(options, &path) else {
                    return None;
                };
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[key];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
            }
            // `import urllib.parse as p` then `p.urlparse(...)` is an
            // Attribute callee, not a Name — handled elsewhere.
            _ => return None,
        }
    }
    None
}

/// datetime constructors: `date` / `datetime` / `timedelta`. Shared by the
/// `from datetime import ...` Name path (`date(2025, 1, 1)`) and the
/// module-qualified attribute path (`datetime.date(...)` — urllib3's
/// connection module). Arguments map against the Python signatures and lower
/// to the runtime `::new` constructors (Option-typed defaulted parameters);
/// `date`/`datetime` validate and propagate with `?`. Returns Ok(None) when
/// `name` is not one of the three constructors.
fn render_datetime_ctor(
    name: &str,
    args: &[crate::ExprType],
    keywords: &[crate::Keyword],
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    let Some(dt) = crate::DatetimeType::from_name(name) else {
        return Ok(None);
    };
    // time/timezone construct through stdpython_class (`time::new(...)`),
    // not through the field-decomposed constructor below.
    if matches!(
        dt,
        crate::DatetimeType::Time | crate::DatetimeType::Timezone
    ) {
        return Ok(None);
    }
    let (params, required): (&[&str], usize) = match dt {
        crate::DatetimeType::Date => (&["year", "month", "day"], 3),
        crate::DatetimeType::DateTime => (
            &[
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "microsecond",
            ],
            3,
        ),
        crate::DatetimeType::Timedelta => (
            &[
                "days",
                "seconds",
                "microseconds",
                "milliseconds",
                "minutes",
                "hours",
                "weeks",
            ],
            0,
        ),
        // Filtered out above (they construct through stdpython_class).
        crate::DatetimeType::Time | crate::DatetimeType::Timezone => unreachable!(),
    };
    if args.len() > params.len() {
        return Err(format!(
            "{}() takes at most {} arguments ({} given)",
            name,
            params.len(),
            args.len()
        )
        .into());
    }
    let mut slots: Vec<Option<crate::ExprType>> = vec![None; params.len()];
    for (i, arg) in args.iter().enumerate() {
        slots[i] = Some(arg.clone());
    }
    // A `*spread` positional (`datetime(*date[:6],
    // tzinfo=timezone.utc)` — pip's vendored cachecontrol
    // heuristics): the spread's element types are dynamic (the
    // spreaded value is a boxed PyValue), so the constructor
    // cannot be statically decomposed. The whole construction
    // is dropped and replaced with now() — the call site's
    // surrounding logic (a cache-expiry heuristic) still works
    // off a plausible date (the spread-argument divergence).
    if args.iter().any(|a| matches!(a, ExprType::Starred(_)))
        || keywords.iter().any(|kw| kw.arg.is_none())
    {
        options.definition_warnings.borrow_mut().push(format!(
            "{}() with a `*spread`/`**spread` argument is dropped; the \
             construction lowers to now()/today()/zero (the spread's \
             element types are dynamic, issue #130)",
            name
        ));
        return Ok(Some(match dt {
            crate::DatetimeType::DateTime => quote!(stdpython::datetime::datetime::now()),
            crate::DatetimeType::Date => quote!(stdpython::datetime::date::today()),
            crate::DatetimeType::Timedelta => quote!(stdpython::datetime::timedelta::new(
                None, None, None, None, None, None, None
            )),
            crate::DatetimeType::Time | crate::DatetimeType::Timezone => unreachable!(),
        }));
    }
    for kw in keywords {
        // `tzinfo=` (e.g. `datetime(*date[:6],
        // tzinfo=timezone.utc)` — pip's vendored
        // cachecontrol) is dropped: rython's datetime is
        // naive, so attaching a timezone is a no-op (the same
        // model as the `replace(tzinfo=None)` tolerance).
        if kw.arg.as_deref() == Some("tzinfo") {
            options.definition_warnings.borrow_mut().push(
                "datetime(...) `tzinfo=` keyword is dropped \
                 (rython's datetime is naive)"
                    .to_string(),
            );
            continue;
        }
        let idx = kw
            .arg
            .as_deref()
            .and_then(|k| params.iter().position(|p| *p == k));
        match idx {
            Some(i) if slots[i].is_none() => slots[i] = Some(kw.value.clone()),
            Some(i) => {
                return Err(format!(
                    "{}() got multiple values for argument '{}'",
                    name, params[i]
                )
                .into());
            }
            None => {
                return Err(format!(
                    "{}() got an unexpected keyword argument '{}'",
                    name,
                    kw.arg.as_deref().unwrap_or("**kwargs")
                )
                .into());
            }
        }
    }
    let mut rendered = Vec::new();
    for (i, slot) in slots.iter().enumerate() {
        let tok = match slot {
            Some(e) => {
                let v = e.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                if i < required {
                    v
                } else {
                    quote!(Some(#v))
                }
            }
            None if i < required => {
                return Err(format!(
                    "{}() missing required argument: '{}'",
                    name, params[i]
                )
                .into());
            }
            None => quote!(None),
        };
        rendered.push(tok);
    }
    let ident = crate::safe_ident(name);
    let call = quote!(stdpython::datetime::#ident::new(#(#rendered),*));
    // timedelta::new is infallible; date/datetime validate.
    Ok(Some(if name == "timedelta" {
        call
    } else {
        quote!(#call?)
    }))
}

impl Call {
    /// Issue #111: render a call to a runtime-module function whose
    /// signature is known — keywords map to parameter slots by name and
    /// omitted trailing parameters fill with `None` (so `warnings.warn(
    /// "x")` and `warnings.simplefilter("ignore", append=True)` both
    /// render). Returns None when the callee is not a signed runtime fn.
    fn render_runtime_signature(
        &self,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
        // Qualified (`warnings.warn(msg)`) or a Name from-imported from a
        // stdpython module (`from warnings import warn; warn(msg)` —
        // charset_normalizer's legacy.py, round 55): both resolve against
        // the signed runtime signature. A Name callee bound by a
        // from-import carries the module in the symbol.
        let (root, attr) = match self.func.as_ref() {
            ExprType::Attribute(a) => match crate::ast::tree::call::root_name(&a.value) {
                Some(root) => (root.to_string(), a.attr.clone()),
                None => return Ok(None),
            },
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let root = i.module.split('.').next().unwrap_or("").to_string();
                    if !crate::is_stdpython_module(&root) {
                        return Ok(None);
                    }
                    // The canonical item name: the from-import may ALIAS
                    // (`from warnings import warn as w`).
                    let canonical = i
                        .names
                        .iter()
                        .find(|a| a.asname.as_deref() == Some(&n.id))
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| n.id.clone());
                    (root, canonical)
                }
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        if !crate::is_stdpython_module(&root) {
            return Ok(None);
        }
        let Some((_, _, params)) = RUNTIME_KEYWORD_SIGNATURES
            .iter()
            .find(|(r, f, _)| *r == root && *f == attr)
        else {
            return Ok(None);
        };
        let mut slots: Vec<Option<ExprType>> = params.iter().map(|_| None).collect();
        for (i, arg) in self.args.iter().enumerate() {
            if i >= slots.len() {
                return Err(format!(
                    "`{attr}()` takes at most {} argument(s) ({} given)",
                    params.len(),
                    self.args.len() + self.keywords.len()
                )
                .into());
            }
            slots[i] = Some(arg.clone());
        }
        for kw in &self.keywords {
            let Some(name) = kw.arg.as_deref() else {
                continue;
            };
            // md5/sha's usedforsecurity keyword is a FIPS policy flag —
            // ignored (requests' digest auth).
            if name == "usedforsecurity"
                && matches!(attr.as_str(), "md5" | "sha1" | "sha256" | "sha512")
            {
                continue;
            }
            let Some(pos) = params.iter().position(|p| *p == name) else {
                return Err(format!(
                    "`{attr}()` got an unexpected keyword argument '{name}'"
                )
                .into());
            };
            if slots[pos].is_some() {
                return Err(format!(
                    "`{attr}()` got multiple values for argument '{name}'"
                )
                .into());
            }
            slots[pos] = Some(kw.value.clone());
        }
        let mut rendered_args = Vec::new();
        for (i, slot) in slots.iter().enumerate() {
            // The warning CATEGORY and SOURCE parameters of
            // `warnings.warn`/`warn_explicit`/`simplefilter`/
            // `filterwarnings` are warning-class VALUES — classes cannot be
            // runtime values in rython. They lower as None (the warning
            // fires unconditionally; documented divergence).
            let is_warning_class_slot = matches!(
                attr.as_str(),
                "warn" | "warn_explicit" | "simplefilter" | "filterwarnings"
            ) && matches!(params.get(i).copied(), Some("category" | "source"));
            match slot {
                // The signed runtime signatures take Option for every
                // parameter: a present argument wraps in Some, an omitted
                // one fills None.
                Some(expr) if is_warning_class_slot => {
                    options.definition_warnings.borrow_mut().push(format!(
                        "warnings.{attr}() `{}` (a warning-class value) is dropped: \
                         classes cannot be runtime values in rython (the \
                         class-as-value divergence)",
                        params[i]
                    ));
                    rendered_args.push(quote!(None));
                }
                Some(expr) => {
                    let rendered =
                        expr.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    // The signed runtime signatures take `Option<&str>`
                    // for the MESSAGE/ACTION params, `Option<i64>` for
                    // stacklevel/lineno: a string LITERAL renders as
                    // `Some("...")` (Option<&str>); an owned String
                    // (`warn(format!(...))` — charset_normalizer's
                    // legacy.py) needs `.as_str()` to coerce into the
                    // Option<&str> slot. Numeric params (stacklevel,
                    // lineno) must NOT get `.as_str()`.
                    let is_str_param = params
                        .get(i)
                        .is_some_and(|p| matches!(*p, "message" | "action" | "filename" | "module"));
                    let wrapped = if is_str_param
                        && !matches!(expr, crate::ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::String(_))))
                    {
                        quote!(Some((#rendered).as_str()))
                    } else {
                        quote!(Some(#rendered))
                    };
                    rendered_args.push(wrapped);
                }
                None => rendered_args.push(quote!(None)),
            }
        }
        let name = self
            .func
            .clone()
            .to_rust(ctx, options, symbols)?;
        Ok(Some(quote!(#name(#(#rendered_args),*))))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub func: Box<ExprType>,
    pub args: Vec<ExprType>,
    pub keywords: Vec<Keyword>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Call {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let func: ExprType = extract_required_attr(&ob, "func", "function call expression")?;
        let args: Vec<ExprType> = extract_required_attr(&ob, "args", "function call arguments")?;
        let keywords: Vec<Keyword> =
            extract_required_attr(&ob, "keywords", "function call keywords")?;

        Ok(Call {
            func: Box::new(func),
            args,
            keywords,
        })
    }
}

/// Is this callee functools.partial — via `from functools import
/// partial` (the symbol table maps the bare name to the functools
/// import) or the `functools.partial` attribute spelling? A user
/// function named partial shadows the import in the symbol table and
/// does not match.
fn is_partial_target(func: &ExprType, symbols: &SymbolTableScopes) -> bool {
    match func {
        ExprType::Name(n) => {
            n.id == "partial"
                && matches!(
                    symbols.get(&n.id),
                    Some(SymbolTableNode::ImportFrom(i))
                        if crate::StdModule::from_name(&i.module)
                            == Some(crate::StdModule::Functools)
                )
        }
        ExprType::Attribute(attr) => {
            attr.attr == "partial"
                && matches!(attr.value.as_ref(), ExprType::Name(m)
                    if crate::StdModule::from_name(&m.id)
                        == Some(crate::StdModule::Functools))
        }
        _ => false,
    }
}

/// Resolve a call target to a numpy function name, when the call is on the
/// numpy module (`np.foo(...)`, `numpy.foo(...)`, `np.linalg.inv(...)`) or
/// on a name imported from it (`from numpy import array; array(...)`).
/// Returns the Python-side function name (`"linalg.inv"` for the nested
/// form). A user definition shadowing the import wins in the symbol table,
/// like the other module dispatches in this file.
fn numpy_target(func: &ExprType, symbols: &SymbolTableScopes) -> Option<String> {
    match func {
        ExprType::Attribute(attr) => match attr.value.as_ref() {
            ExprType::Attribute(inner) => {
                if let ExprType::Name(m) = inner.value.as_ref() {
                    if crate::is_numpy_alias(&m.id) {
                        // `linalg` is the one modeled submodule. Any other
                        // (`np.random.rand`) used to lower to a bare
                        // `np::random::rand` path and fail in rustc; route
                        // it to the numpy lowering so it is refused by name
                        // at conversion time (issue #204).
                        return Some(format!("{}.{}", inner.attr, attr.attr));
                    }
                }
                None
            }
            ExprType::Name(m) if crate::is_numpy_alias(&m.id) => Some(attr.attr.clone()),
            _ => None,
        },
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::ImportFrom(import))
                if crate::StdModule::from_name(
                    import.module.split('.').next().unwrap_or("")
                ) == Some(crate::StdModule::Numpy) =>
            {
                let name = n.id.clone();
                Some(if import.module == "numpy.linalg" {
                    format!("linalg.{name}")
                } else {
                    name
                })
            }
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// numpy call lowering
// ---------------------------------------------------------------------------

/// Render one positional argument.
type NpCtx = (CodeGenContext, PythonOptions, SymbolTableScopes, bool);

/// Whether a numpy function's array arguments are BORROWED by the runtime
/// (`&NdArray` params): the reduction family, the unary ufuncs, and the
/// binary ufuncs all read their operands and build fresh outputs, so the
/// generated call passes `&a` instead of `a.clone()` (a full 64MB memcpy
/// per argument per call — issue #200/#220 follow-up). Functions that
/// CONSUME or reshape their inputs (concatenate, reshape, vstack, ...)
/// keep by-value parameters and the clone spelling.
fn numpy_borrows_arrays(plain_name: &str) -> bool {
    matches!(
        plain_name,
        // reductions (1-arg)
        "sum" | "prod" | "mean" | "max" | "min" | "all" | "any" | "argmax" | "argmin"
        // unary elementwise ufuncs
            | "abs" | "negative" | "sqrt" | "exp" | "expm1" | "log" | "log1p" | "log2"
            | "log10" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "arcsin"
            | "arccos" | "arctan" | "ceil" | "floor" | "square" | "reciprocal" | "sign"
            | "isfinite" | "isinf" | "isnan" | "logical_not" | "where"
        // binary elementwise ufuncs (Into<BinaryOperand<'a>> borrows)
            | "add" | "subtract" | "multiply" | "divide" | "floor_divide" | "mod"
            | "remainder" | "power" | "maximum" | "minimum" | "equal" | "not_equal"
            | "less" | "less_equal" | "greater" | "greater_equal" | "bitwise_and"
            | "bitwise_or" | "bitwise_xor" | "logical_and" | "logical_or" | "logical_xor"
    )
}

fn np_render(expr: &ExprType, ctx: &NpCtx) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let tokens = expr
        .clone()
        .to_rust(ctx.0.clone(), ctx.1.clone(), ctx.2.clone())?;
    // Array-taking numpy functions BORROW their arguments (`&NdArray`
    // params — numpy_borrows_arrays): the generated call passes `&a`, and
    // Python value semantics keep `a` usable after the call with no
    // copy (a full 64MB memcpy per argument per call — issue #200/#220
    // follow-up).
    //
    // For the by-value functions (reshape/concatenate/vstack/...) a PLACE
    // argument still needs a clone so `a = reshape(a, ...); b = a[0]`
    // compiles. A temporary has no name to survive the move, so cloning
    // it only copies an array nobody can observe again
    // (`np.sum(np.multiply(x, x))` — issue #200).
    if ctx.3 {
        return Ok(quote!(&(#tokens)));
    }
    match expr {
        ExprType::Name(_) | ExprType::Attribute(_) | ExprType::Subscript(_) => {
            Ok(quote!((#tokens).clone()))
        }
        _ => Ok(quote!((#tokens))),
    }
}

/// Render an ARRAY-LIST argument (`np.concatenate([p, q], ...)`).
///
/// A list literal lowers to `vec![p, q]`, which MOVES each element — so a
/// second `np.concatenate([p, q], ...)` on the same arrays failed to
/// compile (issue #201). Cloning the whole `vec!` (what np_render does)
/// is too late; each element needs its own clone, which is also the right
/// value semantics: Python's list holds references, rython's holds copies.
fn np_render_array_list(
    expr: &ExprType,
    ctx: &NpCtx,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if let ExprType::List(items) = expr {
        let elems: Result<Vec<TokenStream>, Box<dyn std::error::Error>> =
            items.iter().map(|e| np_render(e, ctx)).collect();
        let elems = elems?;
        return Ok(quote!(vec![#(#elems),*]));
    }
    np_render(expr, ctx)
}

fn np_kw<'a>(keywords: &'a [Keyword], name: &str) -> Option<&'a ExprType> {
    keywords
        .iter()
        .find(|k| k.arg.as_deref() == Some(name))
        .map(|k| &k.value)
}

fn np_has_kw(keywords: &[Keyword], name: &str) -> bool {
    keywords.iter().any(|k| k.arg.as_deref() == Some(name))
}

/// Reject any keyword argument (other than the named ones, if any).
fn np_no_extra_kw(
    fname: &str,
    keywords: &[Keyword],
    allowed: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    for k in keywords {
        if let Some(arg) = k.arg.as_deref() {
            if !allowed.contains(&arg) {
                return Err(format!(
                    "np.{fname}() got an unexpected keyword argument '{arg}' (rython's \
                     numpy subset supports: {})",
                    allowed.join(", ")
                )
                .into());
            }
        }
    }
    Ok(())
}

/// numpy functions whose runtime form returns `Result<_, PyException>`.
///
/// These are the operations CPython raises from — a broadcast mismatch, a
/// singular matrix, a reduction with no identity over an empty array — so
/// the call site propagates with `?` and the exception is catchable
/// instead of aborting the process (issue #205). The OPERATOR spellings
/// (`a + b`) still panic: the operator traits have no fallible form
/// (spec §12.2).
fn np_is_fallible(name: &str) -> bool {
    matches!(
        name,
        // Binary ufuncs: broadcasting can raise ValueError.
        "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "floor_divide"
            | "mod"
            | "remainder"
            | "power"
            | "maximum"
            | "minimum"
            | "equal"
            | "not_equal"
            | "less"
            | "less_equal"
            | "greater"
            | "greater_equal"
            | "bitwise_and"
            | "bitwise_or"
            | "bitwise_xor"
            | "logical_and"
            | "logical_or"
            | "logical_xor"
            // Reductions with no identity raise on an empty array.
            | "max"
            | "min"
            | "argmax"
            | "argmin"
            // np.where broadcasts its two branches.
            | "where"
            // A singular matrix raises LinAlgError.
            | "linalg.inv"
            | "linalg.solve"
    )
}

/// Is this expression PROVABLY a 1-D array?
///
/// numpy's `np.dot` returns a scalar for 1-D x 1-D and an array otherwise;
/// rython has one static type per expression, so the vector case is routed
/// to `vdot` (which returns f64) when — and only when — both operands can
/// be shown 1-D from the creation call itself (issue #206). Anything not
/// provable keeps the array-returning `dot`, which prints identically; the
/// check never guesses, so it can only turn a build error into working
/// code, never a right answer into a wrong one.
fn np_is_1d(expr: &ExprType, options: &PythonOptions, symbols: &SymbolTableScopes) -> bool {
    match expr {
        // A local whose recorded assignment is itself provably 1-D.
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::Assign { value, .. }) => np_is_1d(&value, options, symbols),
            _ => false,
        },
        ExprType::Call(call) => {
            let Some(fname) = numpy_target(call.func.as_ref(), symbols) else {
                return false;
            };
            match fname.as_str() {
                // Always 1-D regardless of arguments.
                "linspace" | "arange" => true,
                // 1-D exactly when the shape argument is a scalar, not a
                // tuple/list of dimensions.
                "zeros" | "ones" | "empty" | "full" => call
                    .args
                    .first()
                    .is_some_and(|s| !matches!(s, ExprType::Tuple(_) | ExprType::List(_))),
                // A flat list literal of numbers.
                "array" | "asarray" => call.args.first().is_some_and(|a| match a {
                    ExprType::List(items) => !items
                        .iter()
                        .any(|e| matches!(e, ExprType::List(_) | ExprType::Tuple(_))),
                    _ => false,
                }),
                // Elementwise results keep their operand's rank.
                "ravel" => true,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Is this expression a float literal (possibly under unary +/-)?
fn np_is_float_literal(expr: &ExprType) -> bool {
    match expr {
        ExprType::Constant(c) => matches!(c.0, Some(litrs::Literal::Float(_))),
        ExprType::UnaryOp(u) => np_is_float_literal(&u.operand),
        _ => false,
    }
}

/// Is this expression an int literal (possibly under unary +/-)?
fn np_is_int_literal(expr: &ExprType) -> bool {
    match expr {
        ExprType::Constant(c) => matches!(c.0, Some(litrs::Literal::Integer(_))),
        ExprType::UnaryOp(u) => np_is_int_literal(&u.operand),
        _ => false,
    }
}

/// Is this expression the None literal?
fn np_is_none(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Constant(c) if c.0.is_none())
}

/// Is this expression a bool literal?
fn np_is_bool_literal(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Constant(c) if matches!(c.0, Some(litrs::Literal::Bool(_))))
}

/// Map a `dtype=` keyword value (`np.int64`, `"float32"`, ...) to the
/// `numpy::Dtype` path token.
fn np_dtype_tokens(expr: &ExprType) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let name = match expr {
        ExprType::Attribute(attr) => {
            if !matches!(attr.value.as_ref(), ExprType::Name(m) if crate::is_numpy_alias(&m.id))
            {
                return Err(format!(
                    "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                     or a string like \"float64\" (got an unsupported expression)"
                )
                .into());
            }
            attr.attr.clone()
        }
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(s)) => s.value().to_string(),
            _ => {
                return Err(
                    "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                     or a string like \"float64\""
                        .to_string()
                        .into(),
                );
            }
        },
        _ => {
            return Err(
                "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                 or a string like \"float64\""
                    .to_string()
                    .into(),
            );
        }
    };
    let variant = match name.as_str() {
        "float64" => "Float64",
        "float32" => "Float32",
        "int64" => "Int64",
        "int32" => "Int32",
        "bool_" | "bool" => "Bool",
        _ => {
            return Err(format!(
                "unsupported numpy dtype '{name}' (rython's numpy subset supports \
                 float64, float32, int64, int32, bool_)"
            )
            .into());
        }
    };
    // An IDENT, not the `&str`: interpolating the string would render a
    // string literal (`numpy::Dtype::"Int64"`), which is not valid Rust.
    let variant = crate::safe_ident(variant);
    Ok(quote!(numpy::Dtype::#variant))
}

/// The `numpy::` path prefix for a given numpy function name (handles the
/// `linalg.*` nested functions).
fn np_path(name: &str) -> (String, TokenStream) {
    if let Some(linalg_fn) = name.strip_prefix("linalg.") {
        let ident = crate::safe_ident(linalg_fn);
        (linalg_fn.to_string(), quote!(numpy::linalg::#ident))
    } else {
        let ident = crate::safe_ident(name);
        (name.to_string(), quote!(numpy::#ident))
    }
}

/// A readable source-like spelling of a (short) expression chain for -W
/// messages: `self.items.append` — Name/Attribute chains join with dots, a
/// call renders as `f(...)`, and anything else falls back to the Debug form
/// (terminal). The -W channel is for humans; a raw AST Debug dump is not a
/// diagnostic (issue #209).
pub(crate) fn expr_chain_spelling(e: &ExprType) -> String {
    match e {
        ExprType::Name(n) => n.id.clone(),
        ExprType::Attribute(a) => {
            format!("{}.{}", expr_chain_spelling(&a.value), a.attr)
        }
        ExprType::Call(c) => format!("{}(...)", expr_chain_spelling(c.func.as_ref())),
        _ => format!("{:?}", e),
    }
}

/// Lower `np.<fname>(args, kwargs)` onto the stdpython numpy module.
#[allow(clippy::too_many_lines)]
fn lower_numpy_call(
    fname: &str,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let borrows = numpy_borrows_arrays(fname.strip_prefix("linalg.").unwrap_or(&fname));
    let npc = (ctx, options, symbols, borrows);
    let fname = fname.to_string();
    // An unmodeled numpy SUBMODULE (`np.random.rand`) is refused before
    // anything tries to build a Rust path out of the dotted name — that
    // used to emit a bare `np::random::rand` and fail in rustc (issue
    // #204), and the ident builder cannot represent the dot at all.
    if fname.contains('.') && !fname.starts_with("linalg.") {
        let submodule = fname.split('.').next().unwrap_or(&fname);
        return Err(format!(
            "np.{submodule} is not supported by rython's numpy subset (the only \
             modeled numpy submodule is np.linalg, with inv, det and solve); \
             rython refuses to emit a call it cannot reproduce"
        )
        .into());
    }
    let (plain_name, path) = np_path(&fname);

    // np.float64(x) / np.int64(x) / np.float32(x) / np.int32(x) / np.bool_(x)
    // are dtype CASTS (numpy's scalar types).
    let cast: Option<&str> = match plain_name.as_str() {
        "float64" => Some("f64"),
        "float32" => Some("f32"),
        "int64" => Some("i64"),
        "int32" => Some("i32"),
        _ => None,
    };
    if let Some(ty) = cast {
        np_no_extra_kw(&plain_name, keywords, &[])?;
        if args.len() != 1 {
            return Err(format!(
                "np.{}() takes exactly 1 argument ({} given)",
                plain_name,
                args.len()
            )
            .into());
        }
        let a = np_render(&args[0], &npc)?;
        // `#ty` is a &str; parse it into an actual Rust type token so the
        // cast emits `as f64`, not `as "f64"`.
        let ty_tokens: proc_macro2::TokenStream = ty
            .parse()
            .unwrap_or_else(|_| panic!("numpy cast type '{ty}' is not valid Rust"));
        return Ok(quote!((#a) as #ty_tokens));
    }
    if plain_name == "bool_" {
        np_no_extra_kw(&plain_name, keywords, &[])?;
        if args.len() != 1 {
            return Err(
                format!("np.bool_() takes exactly 1 argument ({} given)", args.len()).into(),
            );
        }
        let a = np_render(&args[0], &npc)?;
        return Ok(quote!((#a).py_bool()));
    }

    // Construction helpers with shape arguments: shape tuples (2, 3) render
    // as Rust tuples and convert via IntoShape; `dtype=` maps to Dtype.
    let shape_fns: &[&str] = &["zeros", "ones", "empty", "full"];
    if shape_fns.contains(&plain_name.as_str()) {
        np_no_extra_kw(&plain_name, keywords, &["dtype"])?;
        let ndtype = np_kw(keywords, "dtype").map(np_dtype_tokens).transpose()?;
        match plain_name.as_str() {
            "zeros" | "ones" | "empty" => {
                if args.len() != 1 {
                    return Err(format!(
                        "np.{}() takes exactly 1 positional argument (shape), got {}",
                        plain_name,
                        args.len()
                    )
                    .into());
                }
                let shape = np_render(&args[0], &npc)?;
                let dtype = ndtype.unwrap_or(quote!(numpy::Dtype::Float64));
                return Ok(quote!(#path(#shape, #dtype)));
            }
            _ => {
                // full: fill value picks the Rust function (int → full_i,
                // float → full, bool → full_bool); dtype= is not supported
                // yet (it would fight the fill type, like numpy's promotion).
                if ndtype.is_some() {
                    return Err(
                        "np.full(..., dtype=...) is not supported in rython's numpy subset \
                         (the dtype follows the fill value)"
                            .to_string()
                            .into(),
                    );
                }
                if args.len() != 2 {
                    return Err(format!(
                        "np.full() takes exactly 2 arguments (shape, fill_value), got {}",
                        args.len()
                    )
                    .into());
                }
                let shape = np_render(&args[0], &npc)?;
                let fill = np_render(&args[1], &npc)?;
                return if np_is_bool_literal(&args[1]) {
                    Ok(quote!(numpy::full_bool(#shape, #fill)))
                } else if np_is_int_literal(&args[1]) {
                    Ok(quote!(numpy::full_i(#shape, #fill)))
                } else {
                    Ok(quote!(numpy::full(#shape, (#fill) as f64)))
                };
            }
        }
    }

    // Dispatch on the FULL name (fname) so the `linalg.*` arms fire; the
    // plain arms use `plain_name`, which equals fname for non-linalg calls.
    match fname.as_str() {
        "arange" => {
            np_no_extra_kw("arange", keywords, &[])?;
            let n = args.len();
            if !(1..=3).contains(&n) {
                return Err(format!("np.arange() takes 1 to 3 arguments ({} given)", n).into());
            }
            let any_float = args.iter().any(np_is_float_literal);
            let any_int = args.iter().any(np_is_int_literal);
            if any_float && any_int {
                return Err(
                    "np.arange() mixing int and float bounds is not supported in rython's \
                     numpy subset; use all-int or all-float literals"
                        .to_string()
                        .into(),
                );
            }
            let rendered: Vec<TokenStream> = args
                .iter()
                .map(|a| np_render(a, &npc))
                .collect::<Result<_, _>>()?;
            if any_float {
                let (a, b, c) = (
                    rendered
                        .first()
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(0.0)),
                    rendered
                        .get(1)
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(0.0)),
                    rendered
                        .get(2)
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(1.0)),
                );
                return Ok(if n == 1 {
                    quote!(numpy::arange_f(#a))
                } else {
                    quote!(numpy::arange_f3(#a, #b, #c))
                });
            }
            let (a, b, c) = (
                rendered
                    .first()
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(0i64)),
                rendered
                    .get(1)
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(1i64)),
                rendered
                    .get(2)
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(1i64)),
            );
            Ok(if n == 1 {
                quote!(numpy::arange(#a))
            } else {
                quote!(numpy::arange3(#a, #b, #c))
            })
        }

        "linspace" => {
            np_no_extra_kw("linspace", keywords, &["num", "endpoint"])?;
            if let Some(endpoint) = np_kw(keywords, "endpoint") {
                if !matches!(endpoint, ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::Bool(litrs::BoolLit::True))))
                {
                    return Err(
                        "np.linspace(..., endpoint=False) is not supported in rython's \
                         numpy subset (endpoint defaults to True)"
                            .to_string()
                            .into(),
                    );
                }
            }
            if !(2..=3).contains(&args.len()) {
                return Err(format!(
                    "np.linspace() takes 2 or 3 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let start = np_render(&args[0], &npc)?;
            let stop = np_render(&args[1], &npc)?;
            let num = match np_kw(keywords, "num") {
                Some(n) => Some(np_render(n, &npc)?),
                None if args.len() == 3 => Some(np_render(&args[2], &npc)?),
                None => None,
            };
            let num = match num {
                Some(t) => quote!((#t) as i64),
                None => quote!(50i64),
            };
            Ok(quote!(numpy::linspace((#start) as f64, (#stop) as f64, #num)))
        }

        "eye" | "identity" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path((#a) as i64)))
        }

        "dtype" => {
            np_no_extra_kw("dtype", keywords, &[])?;
            if args.len() != 1 {
                return Err(
                    format!("np.dtype() takes exactly 1 argument ({} given)", args.len()).into(),
                );
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(numpy::dtype(#a)))
        }

        "set_backend" => {
            np_no_extra_kw("set_backend", keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.set_backend() takes exactly 1 argument ({} given)",
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            // set_backend_by_name errors with a plain String (unknown
            // backend name); surface it as a raised RuntimeError so the
            // generated Result<_, PyException> functions can `?` it. The
            // extra `&` lets both &str literals and String locals coerce.
            Ok(quote!(numpy::set_backend_by_name(&#a)
                .map_err(|e| PyException::new("RuntimeError", e))?))
        }

        "sum" | "prod" | "mean" | "max" | "min" | "all" | "any" | "argmax" | "argmin" => {
            if np_has_kw(keywords, "axis") {
                return Err(format!(
                    "np.{}(a, axis=...) is not supported in rython's numpy subset \
                     (full reductions only)",
                    plain_name
                )
                .into());
            }
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            if np_is_fallible(&fname) {
                return Ok(quote!(#path(#a)?));
            }
            Ok(quote!(#path(#a)))
        }

        "std" | "var" => {
            np_no_extra_kw(&plain_name, keywords, &["ddof"])?;
            if args.len() != 1 {
                // numpy's second POSITIONAL parameter is `axis`, not `ddof`
                // — `np.std(a, 1)` is a per-axis reduction there. Binding it
                // to ddof made the same call mean something else with no
                // diagnostic (issue #196), so the positional form is refused
                // and ddof is keyword-only.
                return Err(format!(
                    "np.{}() takes exactly 1 positional argument ({} given); numpy's \
                     second positional parameter is `axis`, which is not supported in \
                     rython's numpy subset — pass ddof as a keyword (np.{}(a, ddof=1))",
                    plain_name,
                    args.len(),
                    plain_name
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            let ddof = match np_kw(keywords, "ddof") {
                Some(d) => Some(np_render(d, &npc)?),
                None => None,
            };
            let ddof = match ddof {
                Some(t) => quote!((#t) as f64),
                None => quote!(0.0),
            };
            Ok(quote!(#path(#a, #ddof)))
        }

        "clip" => {
            np_no_extra_kw("clip", keywords, &["min", "max"])?;
            let lo = np_kw(keywords, "min").or_else(|| args.get(1));
            let hi = np_kw(keywords, "max").or_else(|| args.get(2));
            if args.is_empty() || lo.is_none() || hi.is_none() {
                return Err(
                    "np.clip() needs an array plus min and max bounds (either both \
                     positional or min=/max= keywords); None means unbounded"
                        .to_string()
                        .into(),
                );
            }
            let a = np_render(&args[0], &npc)?;
            let lo = match lo {
                Some(e) if np_is_none(e) => quote!(None),
                Some(e) => {
                    let t = np_render(e, &npc)?;
                    quote!(Some((#t) as f64))
                }
                None => quote!(None),
            };
            let hi = match hi {
                Some(e) if np_is_none(e) => quote!(None),
                Some(e) => {
                    let t = np_render(e, &npc)?;
                    quote!(Some((#t) as f64))
                }
                None => quote!(None),
            };
            Ok(quote!(numpy::clip(#a, #lo, #hi)))
        }

        "where" => {
            np_no_extra_kw("where", keywords, &[])?;
            if args.len() != 3 {
                return Err(
                    "np.where(cond, a, b) needs exactly 3 arguments in rython's numpy \
                     subset (the single-argument index form is not supported)"
                        .to_string()
                        .into(),
                );
            }
            let (c, a, b) = (
                np_render(&args[0], &npc)?,
                np_render(&args[1], &npc)?,
                np_render(&args[2], &npc)?,
            );
            Ok(quote!(numpy::where_(&#c, &#a, &#b)?))
        }

        "concatenate" => {
            np_no_extra_kw("concatenate", keywords, &["axis"])?;
            if args.is_empty() || args.len() > 2 {
                return Err(format!(
                    "np.concatenate() takes 1 or 2 positional arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let arrays = np_render_array_list(&args[0], &npc)?;
            // numpy's `axis` is the second POSITIONAL parameter as well as a
            // keyword — `np.concatenate([p, q], 1)` is ordinary numpy.
            let axis = match np_kw(keywords, "axis").or_else(|| args.get(1)) {
                Some(a) => np_render(a, &npc)?,
                None => quote!(0i64),
            };
            Ok(quote!(numpy::concatenate(#arrays, (#axis) as i64)))
        }

        // Plain 1-arg pass-throughs.
        "array" | "asarray" | "ravel" | "transpose" | "sort" | "argsort" | "negative" | "abs"
        | "square" | "sign" | "isfinite" | "isinf" | "isnan" | "logical_not" | "sqrt" | "exp"
        | "log" | "log2" | "log10" | "sin" | "cos" | "tan" | "arcsin" | "arccos" | "arctan"
        | "sinh" | "cosh" | "tanh" | "floor" | "ceil" | "reciprocal" | "expm1" | "log1p" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }

        // np.reshape(a, shape) — the shape tuple converts via IntoShape.
        "reshape" => {
            np_no_extra_kw("reshape", keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.reshape() takes exactly 2 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let (a, s) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            Ok(quote!(numpy::reshape(#a, #s)))
        }

        // Plain 2-arg pass-throughs.
        "add" | "subtract" | "multiply" | "divide" | "floor_divide" | "mod" | "remainder"
        | "power" | "maximum" | "minimum" | "equal" | "not_equal" | "less" | "less_equal"
        | "greater" | "greater_equal" | "bitwise_and" | "bitwise_or" | "bitwise_xor"
        | "logical_and" | "logical_or" | "logical_xor" | "matmul" | "dot" | "vdot" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.{}() takes exactly 2 arguments ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let (a, b) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            // np.dot on two provably 1-D operands is numpy's inner product,
            // which returns a SCALAR — `vdot` is that function here.
            if plain_name == "dot"
                && np_is_1d(&args[0], &npc.1, &npc.2)
                && np_is_1d(&args[1], &npc.1, &npc.2)
            {
                return Ok(quote!(numpy::vdot(#a, #b)));
            }
            let name = if plain_name == "mod" {
                "mod_"
            } else {
                plain_name.as_str()
            };
            let path = crate::safe_ident(name);
            if np_is_fallible(&fname) {
                return Ok(quote!(numpy::#path(#a, #b)?));
            }
            Ok(quote!(numpy::#path(#a, #b)))
        }

        "vstack" | "hstack" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render_array_list(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }

        // linalg.inv / linalg.det / linalg.solve
        "linalg.inv" | "linalg.det" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.linalg.{}() takes exactly 1 argument ({} given)",
                    plain_name.trim_start_matches("linalg."),
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            if np_is_fallible(&fname) {
                return Ok(quote!(#path(#a)?));
            }
            Ok(quote!(#path(#a)))
        }
        "linalg.solve" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.linalg.solve() takes exactly 2 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let (a, b) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            Ok(quote!(#path(#a, #b)?))
        }

        other => Err(format!(
            "np.{other}() is not supported by rython's numpy subset. Supported: \
             array, zeros, ones, full, empty, arange, linspace, eye, identity, dtype, \
             reshape, ravel, transpose, concatenate, vstack, hstack, clip, where, sort, \
             argsort, sum, prod, mean, max, min, std, var, all, any, argmax, argmin, \
             dot, matmul, vdot, set_backend, the ufuncs (add, subtract, multiply, \
             divide, floor_divide, mod, power, sqrt, exp, log, sin, cos, ...), the \
             comparisons (equal, not_equal, less, greater, ...), the dtype casts \
             (np.float64, np.int64, np.bool_, ...), and np.linalg.{{inv,det,solve}}. \
             The accepted surface is docs/spec.md §10.6."
        )
        .into()),
    }
}

/// Whether a name resolves to a class (via aliases and re-export chains)
/// — for the statically-decidable class isinstance.
fn is_class_target(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    depth: usize,
) -> bool {
    if depth > 16 {
        return false;
    }
    match symbols.get(name) {
        Some(SymbolTableNode::ClassDef(_)) => true,
        Some(SymbolTableNode::Alias(canonical)) => {
            is_class_target(canonical, symbols, options, depth + 1)
        }
        Some(SymbolTableNode::ImportFrom(_)) => {
            // Any imported name (`Morsel` from http.cookiejar, or a
            // resolvable in-package class) is class-like — the isinstance
            // check is statically false for rython's boxed values.
            true
        }
        _ => false,
    }
}

/// Resolve the class behind a construction call (`Point(args)`), through
/// aliases (`TimeoutSauce` → `Timeout`) and re-export chains (`util` →
/// `.timeout`), returning the ClassDef with its defining module's symbols.
pub(crate) fn resolve_construction_class(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    resolve_construction_class_depth(name, symbols, options, 0)
}

fn resolve_construction_class_depth(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    depth: usize,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    if depth > 16 {
        return None;
    }
    match symbols.get(name) {
        Some(SymbolTableNode::ClassDef(c)) => Some((c.clone(), symbols.clone())),
        Some(SymbolTableNode::Alias(canonical)) => {
            resolve_construction_class_depth(canonical, symbols, options, depth + 1)
        }
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            if options.module_defs.contains_key(&path) {
                if name == "RLResolver" {
                }
                // The DEFINING module's name for the class: an alias
                // (`from urllib3.util import Timeout as TimeoutSauce`)
                // binds the ORIGINAL name there.
                let defining = i
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(name))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| name.to_string());
                crate::resolve_imported_class(options, &path, &defining, 0)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Whether a method call on this ATTRIBUTE receiver is DROPPED by the
/// boxed-receiver rule (the dynamic-method divergence) — the exact
/// condition the call lowering uses, shared with the RETURN lowering (a
/// dropped call in return position must not emit `Ok(PyValue::None_)`
/// in a typed fn — round 80). Protocol methods survive (the runtime
/// forwards them); module members and resolvable receivers do not drop.
/// Whether a NAME is bound to a call into an EXTERNAL module, which
/// lowered to the boxed None (`conn = h2.connection.H2Connection(...)`).
/// A merely-unknown PyValue-typed name keeps its calls.
pub(crate) fn name_is_dropped_external_value(
    e: &crate::ExprType,
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> bool {
    let crate::ExprType::Name(n) = e else {
        return false;
    };
    let Some(SymbolTableNode::Assign {
        value: ExprType::Call(c),
        ..
    }) = symbols.get(&n.id)
    else {
        return false;
    };
    match c.func.as_ref() {
        crate::ExprType::Attribute(a) => crate::ast::tree::attribute::external_module_root(
            &a.value, symbols, options,
        )
        .is_some(),
        crate::ExprType::Name(f) => {
            crate::ast::tree::import::resolves_to_external_import(&f.id, options, symbols)
        }
        _ => false,
    }
}

pub(crate) fn boxed_receiver_method_dropped(
    attr: &crate::Attribute,
    ctx: &crate::CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> bool {
    if !crate::ast::tree::attribute::receiver_is_pyvalue(&attr.value, ctx, symbols, options) {
        return false;
    }
    matches!(attr.value.as_ref(), ExprType::Attribute(_))
        || crate::ast::tree::attribute::receiver_call_is_external_drop(
            &attr.value,
            symbols,
            options,
        )
        || (!crate::ast::tree::attribute::pyvalue_protocol_method(&attr.attr)
            && (name_is_dropped_external_value(&attr.value, symbols, options)
                || crate::ast::tree::attribute::receiver_is_boxed_positively(
                    &attr.value,
                    symbols,
                    options,
                )))
}

impl<'a> CodeGen for Call {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // typing-module calls are compile-time-only: TypeVar, Protocol,
        // TypeAlias, runtime_checkable, Literal, ... exist only for
        // the type system (annotations are strings under `from __future__
        // import annotations`), so a call lowers to nothing — the same
        // treatment the `from typing import ...` import already gets.
        // `cast(T, value)` is special: it IS a runtime identity (the value
        // passes through unchanged), so it lowers to its VALUE argument —
        // an empty lowering would break `cookie_jar = cast("CookieJar",
        // prep._cookies)` into `cookie_jar = ;` (requests' auth).
        if let ExprType::Name(n) = self.func.as_ref() {
            if let Some(SymbolTableNode::ImportFrom(i)) = symbols.get(&n.id) {
                if crate::AnnotationModule::from_name(i.module.split('.').next().unwrap_or(""))
                    == Some(crate::AnnotationModule::Typing)
                {
                    if n.id == "cast" && self.args.len() == 2 {
                        return self.args[1].clone().to_rust(ctx, options, symbols);
                    }
                    return Ok(TokenStream::new());
                }
            }
        }
        // `typing.cast(T, value)` — the MODULE-QUALIFIED form of the
        // runtime-identity cast (`typing.cast(ProxyConfig,
        // self.proxy_config)` — urllib3's _connect_tls_proxy): same
        // lowering as the imported `cast` name above — the value passes
        // through unchanged (round 94 — without it the call fell to the
        // external-module drop and the local became `PyValue::None_`,
        // breaking every field read on it, E0609).
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && attr.attr == "cast"
            && self.args.len() == 2
            && matches!(attr.value.as_ref(), ExprType::Name(m) if crate::is_typing(&m.id))
        {
            return self.args[1].clone().to_rust(ctx, options, symbols);
        }
        // A compat builtin ALIAS used as a callee (`builtin_str = str` —
        // requests/compat, called as `builtin_str(x)` in models.py): the
        // alias resolves through its defining module to the builtin name,
        // so the call re-dispatches exactly like the builtin (`str(x)`).
        if let ExprType::Name(n) = self.func.as_ref()
            && let Some(builtin) = resolve_builtin_alias(&n.id, &symbols, &options)
        {
            let mut c = self.clone();
            c.func = Box::new(ExprType::Name(crate::ast::tree::name::Name {
                id: builtin,
            }));
            return c.to_rust(ctx, options, symbols);
        }
        // `threading.Thread(target=f, args=(...))` — the constructor takes
        // a CALLABLE, which rython cannot pass as a value: the target is
        // resolved statically and the thread body synthesized at conversion
        // time (the functools.partial model).
        if let Some(tokens) = lower_threading_thread(&self, &ctx, &options, &symbols)? {
            return Ok(tokens);
        }
        // `threading.Semaphore()` — CPython's default initial value is 1;
        // the runtime constructor takes the value explicitly.
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && crate::ThreadingType::from_name(&attr.attr) == Some(crate::ThreadingType::Semaphore)
            && matches!(attr.value.as_ref(), ExprType::Name(n)
                if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Threading))
            && !module_name_shadowed(crate::StdModule::Threading.name(), &symbols)
            && self.args.is_empty()
            && self.keywords.is_empty()
        {
            return Ok(quote!(threading::Semaphore(1)));
        }
        // A keyword call through a functools.partial-bound name: the
        // generated closure has no named parameters, so the keyword would
        // be silently dropped and the call mis-arity'd — loud at
        // conversion (the callable-as-value divergence, issue #122,
        // tracks the general model; botocore's
        // `self._action(attempts=attempts)` is the real-world shape).
        if let ExprType::Name(n) = self.func.as_ref()
            && !self.keywords.is_empty()
            && symbols.get(&n.id).is_some_and(|s| {
                matches!(
                    s,
                    SymbolTableNode::Assign {
                        value: ExprType::Call(c),
                        ..
                    } if is_partial_target(c.func.as_ref(), &symbols)
                )
            })
        {
            return Err("keyword call through a functools.partial-bound name is not \
                        supported yet (the callable-as-value divergence, issue #122)"
                .to_string()
                .into());
        }
        // Calls to functions that return Result<T, PyException> get `?` so
        // exceptions propagate to the caller (or an enclosing try block),
        // as in Python: user-defined functions (known from the symbol
        // table), names imported from user modules, and the Result-returning
        // stdpython builtins.
        let propagates_exceptions = match self.func.as_ref() {
            ExprType::Name(name) => {
                matches!(name.id.as_str(), "int" | "float" | "chr")
                    || match symbols.get(&name.id) {
                        Some(SymbolTableNode::FunctionDef(_)) => true,
                        Some(SymbolTableNode::ImportFrom(import)) => {
                            let root = import.module.split('.').next().unwrap_or("");
                            !crate::is_stdpython_module(root)
                        }
                        // `from pylev import wf as w` — an aliased import of
                        // a user-module function propagates exactly like the
                        // unaliased spelling.
                        Some(SymbolTableNode::Alias(canonical)) => {
                            matches!(
                                symbols.get(canonical),
                                Some(SymbolTableNode::ImportFrom(import))
                                    if !crate::is_stdpython_module(
                                        import.module.split('.').next().unwrap_or("")
                                    )
                            )
                        }
                        // A name bound to functools.partial(f, ...) is a
                        // closure returning f's Result: propagate.
                        Some(SymbolTableNode::Assign {
                            value: ExprType::Call(c),
                            ..
                        }) => is_partial_target(c.func.as_ref(), &symbols),
                        _ => false,
                    }
            }
            // `pylev.wf(...)` / `p: pylev` aliases: a call through a module
            // path into a transpiled (non-runtime) module returns
            // Result<T, PyException>. Runtime modules keep their own
            // lowering (time.monotonic() returns f64, no `?`) — except the
            // ones that return Result because they can raise like Python
            // (math.sqrt, math.pow, json.loads, ...); those thread `?`
            // exactly like the fallible builtins (issue #82 makes the math
            // family reachable from transpiled code).
            ExprType::Attribute(attr) => {
                let root = crate::ast::tree::call::root_name(&attr.value);
                match root {
                    Some(root) if !crate::is_stdpython_module(&root) => {
                        crate::ast::tree::attribute::is_module_path_chain(&attr.value, &symbols, &options)
                    }
                    Some(root) if crate::is_stdpython_module(&root) => {
                        FALLIBLE_STDLIB_FN.contains(&attr.attr.as_str())
                    }
                    _ => false,
                }
            }
            _ => false,
        };

        // rust.bind / rust.c_bind names: calls lower to direct calls into
        // the bound crate, with type-directed conversions. This comes before
        // every builtin handler — a binding shadows the builtin of the same
        // name exactly as a user function would.
        if let ExprType::Name(n) = self.func.as_ref() {
            if let Some(SymbolTableNode::RustBinding(spec)) = symbols.get(&n.id) {
                let spec: crate::RustModuleSpec = spec.into();
                return lower_rust_binding_call(
                    &spec,
                    None,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }
            // A name imported from a Rust module (`from crc32c import crc32c`,
            // optionally aliased): the symbol carries the single-function spec.
            if let Some(SymbolTableNode::RustModule(spec)) = symbols.get(&n.id) {
                return lower_rust_binding_call(
                    spec,
                    None,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }
        }
        // Calls into a Rust module: `crc32c.crc32c(data, 0)` — the module
        // name (or its alias) resolves through the symbol table to a
        // RustModule symbol.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            if let ExprType::Name(root) = attr.value.as_ref() {
                // Follow `import crc32c as c` aliases to the canonical name.
                let module_symbol = match symbols.get(&root.id) {
                    Some(crate::SymbolTableNode::Alias(canonical)) => {
                        symbols.get(canonical).and_then(|s| match s {
                            crate::SymbolTableNode::RustModule(spec) => Some(spec.clone()),
                            _ => None,
                        })
                    }
                    Some(crate::SymbolTableNode::RustModule(spec)) => Some(spec.clone()),
                    _ => None,
                };
                if let Some(spec) = module_symbol {
                    // Validate the function exists; the call itself reuses
                    // the spec (single-function from-imports, or the full
                    // module spec for module imports).
                    spec.get_fn(&attr.attr).ok_or_else(|| {
                        format!(
                            "`{}` is not a bound function of Rust module `{}` (bound: {})",
                            attr.attr,
                            root.id,
                            spec.fns
                                .iter()
                                .map(|f| f.fn_name.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                    return lower_rust_binding_call(
                        &spec,
                        Some(&attr.attr),
                        &self.args,
                        &self.keywords,
                        &ctx,
                        &options,
                        &symbols,
                    );
                }
            }
        }
        // A rust.bind declaration used as a bare expression (not assigned) is
        // a loud error: bindings must be module-level assignments.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            if matches!(attr.value.as_ref(), ExprType::Name(m) if m.id == "rust")
                && matches!(attr.attr.as_str(), "bind" | "c_bind")
            {
                return Err("rust.bind(...) must be assigned to a name at module level"
                    .to_string()
                    .into());
            }
        }

        // A method call on a receiver that is a BOXED-PyValue FIELD CHAIN
        // (`(self._response_mut().body).close()` — the emscripten response
        // where `body` is a PyValue field) OR a PyValue-typed NAME
        // (`conn.data_to_send()` where conn is a boxed h2 connection —
        // issue #137 round 22): the method has no static shape — the call
        // lowers to the boxed None (dynamic-method divergence). The boxed
        // value's own PROTOCOL surface stays exempt on name receivers —
        // is_*/as_*/py_* and the rewrite-table names the later pipeline
        // lowers (decode, encode, split, strip, ...) resolve normally.
        // One definition, shared with the read-side drop in attribute.rs.
        // A NAME receiver drops only on the PRECISE pattern: the name is
        // bound to a call into an EXTERNAL module, which lowered to the
        // boxed None (`conn = h2.connection.H2Connection(...)`, `log =
        // logging.getLogger(...)`). A merely-unknown PyValue-typed name
        // (a socket, a generic parameter, a class) keeps its calls — the
        // TypeInfo::PyValue signal alone means "unknown", not "boxed".

        if let ExprType::Attribute(attr) = self.func.as_ref()
            && crate::boxed_receiver_method_dropped(attr, &ctx, &symbols, &options)
        {
            options.definition_warnings.borrow_mut().push(format!(
                "`{}.{}(...)` is dropped: the receiver is a boxed PyValue \
                 (dynamic-method divergence)",
                expr_chain_spelling(&attr.value), attr.attr
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }
        // Calls into an EXTERNAL module (`ssl.SSLContext(...)`,
        // `socket.socket(...)`, `logging.getLogger(...)` — stdlib rython
        // does not model, or a non-vendored dependency) lower to the boxed
        // None with a warning (documented divergence: the module has no
        // runtime in the generated crate). Calls into generated sibling
        // modules and stdpython are untouched.
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && let Some(root) =
                crate::ast::tree::attribute::external_module_root(&attr.value, &symbols, &options)
        {
            options.definition_warnings.borrow_mut().push(format!(
                "`{}.{}(...)` is dropped: the module `{}` is external to the generated \
                 crate (external-module divergence)",
                root, attr.attr, root
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }
        // A name imported from an external module via `from X import name`
        // (`from logging import getLogger`, `from zlib import ...`), or
        // re-exported through a sibling chain (`from .compat import
        // urlparse` where compat re-exports urllib.parse — requests): the
        // call drops the same way. Exception-class names and stdpython /
        // sibling-module imports are untouched.
        if let ExprType::Name(n) = self.func.as_ref()
            && !crate::ast::tree::raise_stmt::is_exception_class_name(&n.id)
            && !crate::ast::tree::import::import_from_python_module(&n.id, &symbols, &options)
            && (crate::ast::tree::attribute::external_module_root(
                &ExprType::Name(crate::ast::tree::name::Name {
                    id: n.id.clone(),
                }),
                &symbols,
                &options,
            )
            .is_some()
                || crate::ast::tree::import::resolves_to_external_import(
                    &n.id,
                    &options,
                    &symbols,
                )
                || crate::ast::tree::import::import_dropped_stdpython_item(
                    &n.id, &symbols,
                ))
        {
            options.definition_warnings.borrow_mut().push(format!(
                "`{}(...)` is dropped: `{}` is imported from a module that is \
                 external to the generated crate (external-module divergence)",
                n.id, n.id
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // The I/O builtins have no no_std lowering (stdpython gates them
        // behind std): fail at conversion time with the reason, rather than
        // let the generated crate fail with a bare unresolved-name error. A
        // user definition of the same name shadows the builtin as usual.
        if options.no_std {
            if let ExprType::Name(n) = self.func.as_ref() {
                if matches!(n.id.as_str(), "print" | "input" | "open")
                    && symbols.get(&n.id).is_none()
                {
                    return Err(format!(
                        "`{}()` requires OS I/O, which the no_std profile does not \
                         provide; remove the call or convert without the no_std \
                         profile",
                        n.id
                    )
                    .into());
                }
            }
        }

        // Fallible numpy METHODS (`a.max()` on an empty array) propagate
        // the same catchable exception their function spelling does.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            if matches!(attr.attr.as_str(), "max" | "min" | "argmax" | "argmin")
                && self.args.is_empty()
                && self.keywords.is_empty()
                && crate::ast::tree::type_ctx::is_ndarray_expr(&attr.value, &options, &symbols)
            {
                let recv = attr.value.as_ref().clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                let m = crate::safe_ident(&attr.attr);
                return Ok(quote!((#recv).#m()?));
            }
        }

        // `a.astype(np.int64)` — a METHOD on an array whose argument is a
        // dtype. The runtime method exists; only the argument needed
        // mapping, since `np.int64` is otherwise a cast CALL and not a
        // value (issue #204).
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            if attr.attr == "astype"
                && self.args.len() == 1
                && self.keywords.is_empty()
                && crate::ast::tree::type_ctx::is_ndarray_expr(&attr.value, &options, &symbols)
            {
                let recv = attr.value.as_ref().clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                let dtype = np_dtype_tokens(&self.args[0])?;
                return Ok(quote!((#recv).astype(#dtype)));
            }
        }

        // numpy subset calls (`np.foo(...)`, `numpy.foo(...)`, and
        // `from numpy import ...` names) map onto the stdpython numpy
        // module at compile time: the runtime has one Rust function per
        // arity/type combination (no overloading, no default arguments),
        // shape tuples need converting, and dtype= keywords need mapping.
        // Anything unsupported fails HERE with a clear message instead of
        // as a cryptic error in the generated crate.
        if let Some(numpy_fn) = numpy_target(self.func.as_ref(), &symbols) {
            return lower_numpy_call(&numpy_fn, &self.args, &self.keywords, ctx, options, symbols);
        }

        // Multi-argument range() maps to the arity-specific runtime
        // functions (Rust has no overloading); the 3-argument form can
        // raise ValueError on a zero step, hence `?`. A user-defined
        // `range` shadows the builtin and skips this mapping.
        if let ExprType::Name(n) = self.func.as_ref() {
            if n.id == "range"
                && symbols.get("range").is_none()
                && self.keywords.is_empty()
                && matches!(self.args.len(), 1 | 2 | 3)
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    // Every range argument is an index/step: `range(len(x))`
                    // passes usize, so coerce to i64 here rather than rely on
                    // the runtime generic (which fails to resolve for
                    // expression arguments of unknown type).
                    rendered.push(crate::render_typed(
                        arg,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        Some(crate::TypeInfo::Int),
                    )?);
                }
                return Ok(match rendered.len() {
                    1 => quote!(range(#(#rendered),*)),
                    2 => {
                        let (a, b) = (&rendered[0], &rendered[1]);
                        quote!(range_start_stop(#a, #b))
                    }
                    _ => {
                        let (a, b, c) = (&rendered[0], &rendered[1], &rendered[2]);
                        quote!(range_start_stop_step(#a, #b, #c)?)
                    }
                });
            }
        }

        // `object()` — a unique sentinel (`self._body_position = object()` —
        // requests' models): the boxed None (unique objects have no
        // analogue — the sentinel divergence). `memoryview(...)` — the
        // buffer-view builtin — a boxed value (annotations already resolve
        // memoryview to PyValue).
        if let ExprType::Name(n) = self.func.as_ref()
            && matches!(n.id.as_str(), "object" | "memoryview")
            && symbols.get(&n.id).is_none()
        {
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // Builtins with keyword variants or by-reference runtime shapes:
        // min/max (key=, default=, n-ary), sorted (key=, reverse=),
        // enumerate (start=), pow (3-arg modular), and the by-reference
        // len/repr/reversed. Each spelling maps to its runtime variant; a
        // user definition of the same name shadows the builtin, and
        // unknown or duplicate keywords are loud errors, as Python raises
        // TypeError for them.
        if let ExprType::Name(n) = self.func.as_ref() {
            let bname = n.id.as_str();
            if matches!(
                bname,
                "min"
                    | "max"
                    | "sorted"
                    | "enumerate"
                    | "pow"
                    | "len"
                    | "repr"
                    | "reversed"
                    | "frozenset"
                    | "map"
                    | "filter"
                    | "list"
                    | "isinstance"
                    | "hash"
                    | "print"
                    | "open"
                    | "round"
                    | "divmod"
                    | "bytes"
                    | "str"
                    | "getattr"
                    | "hasattr"
                    | "setattr"
                    | "type"
                    | "set"
                    | "bytearray"
                    | "iter"
                    | "tuple"
                    | "next"
                    | "id"
            ) && (symbols.get(bname).is_none()
                // An import of a BUILTIN-CLASS self-alias (`from .compat
                // import str` where compat does `str = str` — requests'
                // py2 shim): the self-alias emits no runtime item, so the
                // name still IS the builtin — dispatch to the builtin arm
                // (the generic import path would render a dangling static
                // read, `(*str).clone()(...)`).
                || crate::ast::tree::module::import_binds_builtin_self_alias(
                    bname,
                    &symbols,
                    &options,
                ))
                // A loop element shadowing the builtin (`for filter in ...:
                // filter(**kwargs)` — botocore's docs client): the call
                // through the untyped element is dropped at the generic
                // fallback — do not treat it as the builtin. A local of
                // the same name shadows the builtin the same way.
                && !options.called_params.contains(bname)
                && !options.local_types.contains_key(bname)
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    // Builtin args are borrowed or copied by the runtime;
                    // render them plain — clone-on-reuse is only inserted
                    // for user-function calls, whose params are owned.
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let unexpected = |kw: Option<&str>| -> Box<dyn std::error::Error> {
                    format!(
                        "{}() got an unexpected or duplicate keyword argument '{}'",
                        bname,
                        kw.unwrap_or("**kwargs")
                    )
                    .into()
                };
                match bname {
                    "min" | "max" => {
                        let mut key = None;
                        let mut default = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("default") if default.is_none() => {
                                    default = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err(format!("{}() expected at least 1 argument", bname).into());
                        }
                        if rendered.len() >= 2 {
                            if key.is_some() || default.is_some() {
                                return Err(format!(
                                    "{}() with multiple positional values and keywords \
                                     is not supported yet; pass a list instead",
                                    bname
                                )
                                .into());
                            }
                            // Python min(a, b, c) folds pairwise; ties keep
                            // the earlier argument.
                            let two = format_ident!("{}2", bname);
                            let mut acc = rendered[0].clone();
                            for next in &rendered[1..] {
                                acc = quote!(#two(#acc, #next));
                            }
                            return Ok(acc);
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        return Ok(match (key, default) {
                            (None, None) => {
                                let f = format_ident!("{}", bname);
                                quote!(#f(&(#a))?)
                            }
                            (Some(k), None) => {
                                let k = render_key_fn(&k, &self.args[0], &ctx, &options, &symbols)?;
                                let f = format_ident!("{}_key", bname);
                                quote!(#f(&(#a), #k)?)
                            }
                            (None, Some(d)) => {
                                let d = render(d)?;
                                let f = format_ident!("{}_default", bname);
                                quote!(#f(&(#a), #d))
                            }
                            (Some(k), Some(d)) => {
                                let k = render_key_fn(&k, &self.args[0], &ctx, &options, &symbols)?;
                                let d = render(d)?;
                                let f = format_ident!("{}_key_default", bname);
                                quote!(#f(&(#a), #k, #d))
                            }
                        });
                    }
                    "sorted" => {
                        let mut key = None;
                        let mut reverse = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("reverse") if reverse.is_none() => {
                                    reverse = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.len() != 1 {
                            return Err("sorted() takes exactly one positional argument"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        // A (K, V)-pair element whose value type has NO
                        // ordering (a user class — Item has no `__lt__`):
                        // CPython sorts tuples lexicographically, so the
                        // values compare ONLY on a key tie and raise
                        // TypeError there. `sorted_pairs` sorts by key and
                        // panics CPython's TypeError text on a tie — exact
                        // for unique keys (dict items), loud otherwise
                        // (round 99; the idiom corpus's report).
                        let elem_is_unordered_pair = key.is_none()
                            && matches!(
                                crate::infer_type(Some(&ctx), self.args.first().unwrap(), &options, &symbols),
                                crate::TypeInfo::Vec(e) if matches!(
                                    &*e, crate::TypeInfo::Tuple(ts)
                                        if ts.len() == 2 && matches!(ts[1], crate::TypeInfo::Class(_))
                                )
                            );
                        return Ok(match (key, reverse) {
                            (None, None) if elem_is_unordered_pair => {
                                quote!(sorted_pairs(&(#a)))
                            }
                            (None, None) => quote!(sorted(&(#a))),
                            (Some(k), None) => {
                                let k = render_key_fn(&k, &self.args[0], &ctx, &options, &symbols)?;
                                quote!(sorted_key(&(#a), #k))
                            }
                            (None, Some(r)) => {
                                let r = render(r)?;
                                quote!(sorted_reverse(&(#a), #r))
                            }
                            (Some(k), Some(r)) => {
                                let k = render_key_fn(&k, &self.args[0], &ctx, &options, &symbols)?;
                                let r = render(r)?;
                                quote!(sorted_key_reverse(&(#a), #k, #r))
                            }
                        });
                    }
                    "enumerate" => {
                        let mut start = self.args.get(1).cloned();
                        if self.args.len() > 2 {
                            return Err("enumerate() takes at most 2 arguments".to_string().into());
                        }
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("start") if start.is_none() => start = Some(kw.value.clone()),
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err("enumerate() expected an iterable".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(match start {
                            None => quote!(enumerate(#a)),
                            Some(s) => {
                                let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(enumerate_start(#a, #s))
                            }
                        });
                    }
                    "pow" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        return match rendered.as_slice() {
                            [b, e] => Ok(quote!(pow(#b, #e))),
                            [b, e, m] => Ok(quote!(pow_mod(#b, #e, #m)?)),
                            _ => Err("pow() takes 2 or 3 arguments".to_string().into()),
                        };
                    }
                    // round(x) -> int (half-even), round(x, n) -> float.
                    // The first argument is always numeric: coerce to f64
                    // so round(3) works; the ndigits argument stays i64.
                    "round" => {
                        // round(x, ndigits=N): the ndigits keyword maps to
                        // the second positional argument.
                        let kw_ndigits = self.keywords.len() == 1
                            && self.keywords[0].arg.as_deref() == Some("ndigits");
                        if !self.keywords.is_empty() && !kw_ndigits {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        let render_float = |e: &crate::ExprType| {
                            crate::render_typed(
                                e,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::Float),
                            )
                        };
                        let second = if self.args.len() == 2 {
                            Some(&self.args[1])
                        } else if kw_ndigits {
                            Some(&self.keywords[0].value)
                        } else {
                            None
                        };
                        return match (self.args.len(), second) {
                            (1, None) => {
                                let a = render_float(&self.args[0])?;
                                Ok(quote!(round(#a)))
                            }
                            (_, Some(n)) => {
                                let a = render_float(&self.args[0])?;
                                let n = crate::render_typed(
                                    n,
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                    Some(crate::TypeInfo::Int),
                                )?;
                                Ok(quote!(round_digits(#a, #n)))
                            }
                            _ => Err("round() takes 1 or 2 arguments".to_string().into()),
                        };
                    }
                    // divmod(a, b) lowers to the stdpython helper, whose
                    // floor-division and modulus steps can raise
                    // ZeroDivisionError: the Result it returns propagates
                    // with `?` like the other fallible builtins.
                    "divmod" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        return match rendered.as_slice() {
                            [a, b] => Ok(quote!(divmod(#a, #b)?)),
                            _ => Err("divmod() takes exactly 2 arguments".to_string().into()),
                        };
                    }
                    // isinstance is statically decidable in a typed
                    // lowering: it becomes the constant true/false when the
                    // argument's type is known (annotation or literal), and
                    // a loud error when it is not.
                    "isinstance" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if self.args.len() != 2 {
                            return Err("isinstance() takes exactly 2 arguments"
                                .to_string()
                                .into());
                        }
                        // The RESIDUAL variant of a specialized function
                        // (specialize.rs): its axis parameter is only ever
                        // bound to types OUTSIDE the tested set, so its
                        // isinstance checks are false by construction —
                        // silently, unlike the divergences below.
                        if let ExprType::Name(n) = &self.args[0]
                            && options.residual_fold_false.contains(&n.id)
                        {
                            return Ok(quote!(false));
                        }
                        // An isinstance test on an INFERRED-GENERIC
                        // parameter in a shape the specializer does not
                        // cover (a non-if-test use, a second tested
                        // parameter, defaults/varargs, a method): there is
                        // no runtime type to dispatch on and no variant to
                        // fold in — the documented class-as-value
                        // divergence, false with a warning naming the
                        // specializable shape.
                        if let ExprType::Name(n) = &self.args[0]
                            && options.param_type_vars.contains_key(&n.id)
                        {
                            options.definition_warnings.borrow_mut().push(format!(
                                "isinstance({0}, ...) on an inferred-generic \
                                 parameter lowers to false (the class-as-value \
                                 divergence). rython specializes a module \
                                 function whose unannotated parameter is tested \
                                 only in plain `if isinstance({0}, T):` \
                                 statements with builtin or class targets; \
                                 restructure into that shape, or annotate `{0}`",
                                n.id
                            ));
                            return Ok(quote!(false));
                        }
                        // Exception-class isinstance: `isinstance(e,
                        // LookupError)` where e is a caught exception tests
                        // the PyException's name string — the same match
                        // except handlers use (charset_normalizer's codec
                        // fallback). The target may be an exception class
                        // NAME even though classes aren't values.
                        let is_exc_class = |name: &str| -> bool {
                            crate::ast::tree::raise_stmt::is_exception_class_name(name)
                                || match symbols.get(name) {
                                    Some(SymbolTableNode::ClassDef(c)) => {
                                        crate::is_exception_class(c)
                                    }
                                    _ => false,
                                }
                        };
                        if let ExprType::Name(_n) = &self.args[0]
                            && let ExprType::Name(t) = &self.args[1]
                            && is_exc_class(&t.id)
                        {
                            let arg = self.args[0].clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            let kind = &t.id;
                            return Ok(quote!((#arg).matches(#kind)));
                        }
                        // `isinstance(v, type(x))`: `type(...)` of a
                        // statically-known class instance resolves to that
                        // class (issue #134 — charset_normalizer's codec
                        // fallback checks `type(self)`). Decidable exactly
                        // like the direct class-target form below: true
                        // when v is typed as the same class, otherwise the
                        // documented class-as-value divergence (false),
                        // warned when v's type is unknown.
                        if let ExprType::Call(c) = &self.args[1]
                            && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "type")
                            && c.args.len() == 1
                        {
                            let inner_class = match &c.args[0] {
                                ExprType::Name(n) if n.id == "self" => ctx
                                    .enclosing_class_name()
                                    .map(str::to_string),
                                ExprType::Name(n) => {
                                    // A user-class instance resolves through
                                    // name_types. A BUILTIN type name (int,
                                    // str, ...) acts as its own class name —
                                    // detected via the single authoritative
                                    // annotation mapping rather than a
                                    // parallel string list.
                                    match options.name_types.get(&n.id) {
                                        Some(crate::TypeInfo::Class(cname)) => {
                                            Some(cname.clone())
                                        }
                                        _ => crate::ast::tree::type_ctx::is_builtin_type_annotation(&ExprType::Name(n.clone()))
                                            .then(|| n.id.clone()),
                                    }
                                }
                                _ => None,
                            };
                            if let Some(cname) = inner_class {
                                // A ROOT-typed value (hierarchy.rs) answers
                                // by its runtime variant, not by the static
                                // fold below (Devin review on #319): the
                                // one registry test the class-target form
                                // uses.
                                if let ExprType::Name(n) = &self.args[0]
                                    && let Some(crate::TypeInfo::Class(c)) =
                                        options.name_types.get(&n.id)
                                    && crate::ast::tree::hierarchy::is_polymorphic_root(c)
                                {
                                    let arg = self.args[0].clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )?;
                                    let target = crate::ast::tree::hierarchy::canonical_class_name(&cname, &symbols);
                                    return Ok(root_isinstance_test(c, &target, &arg, &symbols));
                                }
                                // isinstance also accepts subclasses of the
                                // resolved class: walk the inheritance tree.
                                let same_class = matches!(
                                    &self.args[0],
                                    ExprType::Name(n)
                                        if options.name_types.get(&n.id).is_some_and(|ty| {
                                            matches!(
                                                ty,
                                                crate::TypeInfo::Class(cc)
                                                    if crate::ast::tree::class_def::ClassDef
                                                        ::class_extends(cc, &cname, &symbols)
                                            )
                                        })
                                );
                                if !same_class {
                                    options.definition_warnings.borrow_mut().push(format!(
                                        "isinstance(x, type(self)) with x not statically \
                                         typed as `{cname}` lowers to false (the \
                                         class-as-value divergence)"
                                    ));
                                }
                                return Ok(quote!(#same_class));
                            }
                        }
                        // A NON-exception class target (`isinstance(other,
                        // CompatibleFamillyRange)`, or an alias/import of one
                        // like `TimeoutSauce`): statically decidable in
                        // rython's value model through the INHERITANCE TREE —
                        // true when the first argument's class is the target
                        // or transitively inherits from it (`isinstance(dog,
                        // Animal)` with dog: Dog is true, like CPython's
                        // subclass check), false otherwise. A value whose
                        // type is unknown cannot dispatch dynamically (the
                        // class-as-value divergence) — that case is false
                        // WITH a warning, never silently.
                        if let ExprType::Name(t) = &self.args[1]
                            && is_class_target(&t.id, &symbols, &options, 0)
                        {
                            let arg_class = match &self.args[0] {
                                ExprType::Name(n) => {
                                    match options.name_types.get(&n.id) {
                                        Some(crate::TypeInfo::Class(c)) => {
                                            Some(c.clone())
                                        }
                                        _ => None,
                                    }
                                }
                                _ => None,
                            };
                            // A ROOT-typed name (hierarchy.rs) holds any
                            // class of the subtree: a target inside the
                            // subtree is a RUNTIME variant test (the
                            // generated predicate), an ancestor is true,
                            // anything else is false — exact, never a fold
                            // of the static type as if it were the runtime
                            // one (the idiom corpus's shapes: a Square in a
                            // `list[Shape]` answered false to `isinstance(s,
                            // Rect)`). A leaf class IS its struct, so the
                            // class-tree fold is exact there.
                            if let Some(c) = &arg_class
                                && crate::ast::tree::hierarchy::is_polymorphic_root(c)
                            {
                                // The registry knows the class's own name:
                                // an alias target (`C = Circle`, `import
                                // Rect as R`) resolves to it first.
                                let target = crate::ast::tree::hierarchy::canonical_class_name(&t.id, &symbols);
                                let arg = self.args[0].clone().to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                return Ok(root_isinstance_test(c, &target, &arg, &symbols));
                            }
                            let result = match &arg_class {
                                Some(c) => {
                                    crate::ast::tree::class_def::ClassDef::class_extends(
                                        c, &t.id, &symbols,
                                    )
                                }
                                None => {
                                    options.definition_warnings.borrow_mut().push(
                                        format!(
                                            "isinstance(x, {}) with x not statically \
                                             typed as a class lowers to false (the \
                                             class-as-value divergence)",
                                            t.id
                                        ),
                                    );
                                    false
                                }
                            };
                            return Ok(quote!(#result));
                        }
                        // A TUPLE of class targets on a ROOT-typed value
                        // (`isinstance(s, (Circle, Rect))` — Devin review
                        // on #319): the OR of the registry's test per
                        // element — an ancestor makes the whole check true,
                        // a subtree class contributes its variant test, an
                        // unrelated class nothing.
                        if let ExprType::Name(n) = &self.args[0]
                            && let Some(crate::TypeInfo::Class(c)) = options.name_types.get(&n.id)
                            && crate::ast::tree::hierarchy::is_polymorphic_root(c)
                            && let ExprType::Tuple(tup) = &self.args[1]
                            && !tup.elts.is_empty()
                            && tup.elts.iter().all(|e| {
                                matches!(e, ExprType::Name(t) if is_class_target(&t.id, &symbols, &options, 0))
                            })
                        {
                            let arg = self.args[0].clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            let mut tests: Vec<TokenStream> = Vec::new();
                            for e in &tup.elts {
                                let ExprType::Name(t) = e else { unreachable!() };
                                let target = crate::ast::tree::hierarchy::canonical_class_name(&t.id, &symbols);
                                let test = root_isinstance_test(c, &target, &arg, &symbols);
                                let flat = test.to_string();
                                if flat == "true" {
                                    return Ok(quote!(true));
                                }
                                if flat != "false" {
                                    tests.push(test);
                                }
                            }
                            if tests.is_empty() {
                                return Ok(quote!(false));
                            }
                            return Ok(quote!((#(#tests)||*)));
                        }
                        // Resolve a name that aliases a TUPLE of builtin
                        // type names (`basestring = (str, bytes)` in
                        // requests' compat): the elements, each resolved
                        // through the same alias/import machinery. Returns
                        // None when the name is not a tuple-of-types alias.
                        fn resolve_type_tuple(
                            id: &str,
                            options: &PythonOptions,
                            symbols: &SymbolTableScopes,
                        ) -> Option<Vec<String>> {
                            let tuple_value: Vec<ExprType> = match symbols.get(id) {
                                Some(SymbolTableNode::Assign { value, .. }) => match value {
                                    ExprType::Tuple(t) => t.elts.clone(),
                                    _ => return None,
                                },
                                Some(SymbolTableNode::Alias(canonical)) => {
                                    return resolve_type_tuple(canonical, options, symbols)
                                }
                                Some(SymbolTableNode::ImportFrom(i)) => {
                                    let path = i.resolved_module_path(options);
                                    let module = options.module_defs.get(&path)?;
                                    let module: &crate::Module = module;
                                    let syms =
                                        module.clone().find_symbols(SymbolTableScopes::new());
                                    return resolve_type_tuple(id, options, &syms);
                                }
                                _ => return None,
                            };
                            let mut names = Vec::new();
                            for elt in tuple_value {
                                let ExprType::Name(n) = elt else {
                                    return None;
                                };
                                let resolved = resolve_type_name(&n.id, options, symbols)
                                    .unwrap_or_else(|| n.id.clone());
                                names.push(resolved);
                            }
                            Some(names)
                        }

                        // Resolve a non-builtin type NAME through symbols:
                        // `builtin_str = str` (requests' compat alias), an
                        // import alias, or an imported name (`from .compat
                        // import builtin_str`) yields the underlying builtin
                        // type name. Returns None when the name is not a
                        // statically-known type.
                        fn resolve_type_name(
                            id: &str,
                            options: &PythonOptions,
                            symbols: &SymbolTableScopes,
                        ) -> Option<String> {
                            resolve_type_name_depth(id, options, symbols, 0)
                        }
                        fn resolve_type_name_depth(
                            id: &str,
                            options: &PythonOptions,
                            symbols: &SymbolTableScopes,
                            depth: usize,
                        ) -> Option<String> {
                            if depth > 16 {
                                return None;
                            }
                            match symbols.get(id) {
                                Some(SymbolTableNode::Assign { value, .. }) => {
                                    match value {
                                        ExprType::Name(n)
                                            if ISINSTANCE_TARGET_NAMES
                                                .contains(&n.id.as_str()) =>
                                        {
                                            Some(n.id.clone())
                                        }
                                        _ => None,
                                    }
                                }
                                Some(SymbolTableNode::Alias(canonical)) => {
                                    // A self-aliasing re-export
                                    // (`from .connection import ProxyConfig
                                    // as ProxyConfig` — urllib3) would
                                    // recurse forever; the alias is a
                                    // no-op.
                                    if canonical == id {
                                        None
                                    } else {
                                        resolve_type_name_depth(canonical, options, symbols, depth + 1)
                                    }
                                }
                                // An imported name: resolve it through the
                                // DEFINING module's symbol table, where the
                                // alias assignment lives.
                                Some(SymbolTableNode::ImportFrom(i)) => {
                                    let path = i.resolved_module_path(options);
                                    if options.module_defs.contains_key(&path) {
                                        let module =
                                            options.module_defs.get(&path)?;
                                        let module: &crate::Module = module;
                                        let syms = module
                                            .clone()
                                            .find_symbols(SymbolTableScopes::new());
                                        resolve_type_name_depth(id, options, &syms, depth + 1)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            }
                        }
                        let targets = match &self.args[1] {
                            // A single type name (or a name resolving to one,
                            // or a TUPLE alias like requests' `basestring =
                            // (str, bytes)`).
                            ExprType::Name(t) => {
                                let id = resolve_type_name(&t.id, &options, &symbols)
                                    .unwrap_or_else(|| t.id.clone());
                                if ISINSTANCE_TARGET_NAMES.contains(&id.as_str()) {
                                    vec![id]
                                } else if let Some(names) =
                                    resolve_type_tuple(&t.id, &options, &symbols)
                                {
                                    names
                                } else if is_class_target(&t.id, &symbols, &options, 0)
                                    || crate::ast::tree::raise_stmt::is_exception_class_name(&id)
                                    // A BUILTIN class not in symbols
                                    // (`memoryview` — urllib3's
                                    // `isinstance(result, memoryview)`):
                                    // no PyValue predicate — statically
                                    // false, the class-as-value
                                    // divergence.
                                    || matches!(
                                        id.as_str(),
                                        "memoryview" | "bytearray" | "bytes" | "str"
                                    )
                                {
                                    vec![id]
                                } else if options.param_type_vars.contains_key(&t.id)
                                    || options.name_types.contains_key(&t.id)
                                    || options.local_types.contains_key(&t.id)
                                {
                                    // A TYPE-CLASS PARAMETER (`isinstance(
                                    // value, expected_type)` where
                                    // `expected_type: type[_T]` — pip's
                                    // direct_url): a class held as a value
                                    // has no predicate. rython's typed
                                    // model guarantees the value IS the
                                    // expected type, so the check is
                                    // statically true (the class-as-value
                                    // divergence; the guard's else-branch
                                    // raise never fires).
                                    options.definition_warnings.borrow_mut().push(format!(
                                        "isinstance with a type-class parameter \
                                         (`{:?}`) is statically true: rython's typed \
                                         model guarantees the value's type (the \
                                         class-as-value divergence)",
                                        self.args[0]
                                    ));
                                    return Ok(quote!(true));
                                } else {
                                    return Err(format!(
                                        "isinstance() second argument must be int, float, \
                                         str, bool, bytes, bytearray, tuple, or a tuple of \
                                         those (got `{:?}`); classes are not supported yet",
                                        t
                                    )
                                    .into());
                                }
                            }
                            // A tuple of type names: `isinstance(x, (bytearray, bytes))`
                            // — the common "accept either of these" check. Each
                            // element must be a statically-known type name.
                            ExprType::Tuple(tup) => {
                                let mut names = Vec::new();
                                for elt in &tup.elts {
                                    match elt {
                                        ExprType::Name(t) => {
                                            let id = resolve_type_name(&t.id, &options, &symbols)
                                                .unwrap_or_else(|| t.id.clone());
                                            if ISINSTANCE_TARGET_NAMES
                                                .contains(&id.as_str())
                                            {
                                                names.push(id);
                                            } else if is_class_target(&t.id, &symbols, &options, 0)
                                                || crate::ast::tree::raise_stmt::
                                                    is_exception_class_name(&id)
                                            {
                                                // A CLASS target in the tuple
                                                // (`isinstance(e, (BaseSSLError,
                                                // CertificateError))` — exception
                                                // classes in urllib3): no PyValue
                                                // predicate, so it contributes
                                                // nothing (the check is statically
                                                // false for rython's boxed values,
                                                // the class-as-value divergence).
                                                continue;
                                            } else {
                                                return Err(format!(
                                                    "isinstance() tuple-of-types element must \
                                                     be int, float, str, bool, bytes, or \
                                                     bytearray (got `{:?}`)",
                                                    t
                                                )
                                                .into());
                                            }
                                        }
                                        other => {
                                            return Err(format!(
                                                "isinstance() tuple-of-types element must be \
                                                 int, float, str, bool, bytes, or bytearray \
                                                 (got `{:?}`)",
                                                other
                                            )
                                            .into());
                                        }
                                    }
                                }
                                names
                            }
                            other => {
                                // `typing.Mapping` / `cookielib.CookieJar`
                                // — an attribute-path class target,
                                // tolerated (the PyValue dispatch has no
                                // predicate for it, so the check is
                                // statically false in rython's value
                                // model).
                                if let ExprType::Attribute(a) = other {
                                    vec![a.attr.clone()]
                                } else if matches!(other, ExprType::Call(_)) {
                                    // `isinstance(other, type(self))` — a
                                    // runtime CLASS reference (pip's
                                    // CacheablePageContent): no predicate —
                                    // the check is statically false (the
                                    // class-as-value divergence).
                                    vec![]
                                } else {
                                    return Err(format!(
                                        "isinstance() second argument must be int, float, \
                                         str, bool, bytes, bytearray, or a tuple of those \
                                         (got `{:?}`); classes are not supported yet",
                                        other
                                    )
                                    .into());
                                }
                            }
                        };
                        // A StrOrBytes- or PyValue-typed argument (issue
                        // #121): isinstance is a RUNTIME dispatch
                        // (is_bytes / is_str / is_int / ...), not a static
                        // constant — the branch body narrows the name.
                        if let ExprType::Name(n) = &self.args[0]
                            && options.name_types.get(&n.id).is_some_and(|t| {
                                matches!(
                                    t,
                                    crate::TypeInfo::StrOrBytes | crate::TypeInfo::PyValue
                                )
                            })
                        {
                            let arg = self.args[0].clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            // The runtime predicate per resolved target.
                            let method_of = |id: &str| -> Option<&'static str> {
                                match id {
                                    "str" => Some("is_str"),
                                    "int" => Some("is_int"),
                                    "float" => Some("is_float"),
                                    "bool" => Some("is_bool"),
                                    "bytes" | "bytearray" => Some("is_bytes"),
                                    "tuple" => Some("is_tuple"),
                                    _ => None,
                                }
                            };
                            let mut methods: Vec<TokenStream> = Vec::new();
                            for t in &targets {
                                // Class-typed targets (PathLike, BinaryIO,
                                // ...) have no PyValue predicate: a boxed
                                // value never holds a class instance, so
                                // they contribute nothing to the check.
                                let Some(m) = method_of(t) else {
                                    continue;
                                };
                                let m = crate::safe_ident(m);
                                methods.push(quote!((#arg).#m()));
                            }
                            // `isinstance(x, PathLike)` on a boxed value is
                            // always false in rython's value model.
                            if methods.is_empty() {
                                return Ok(quote!(false));
                            }
                            // Single target: one predicate. Multiple
                            // DISTINCT targets (`(str, bytes)`) — only
                            // possible on a PyValue — OR together.
                            return Ok(if methods.len() == 1 {
                                methods.pop().unwrap()
                            } else {
                                let joined = methods.iter();
                                quote!(#(#joined)||*)
                            });
                        }
                        // The Python type name each target lowers to;
                        // bytes and bytearray BOTH lower to Vec<u8> (the
                        // `bytes | bytearray` union is a single Rust type),
                        // so a tuple containing either matches a bytes
                        // argument (the local_types map records the union
                        // annotation as "bytes").
                        let actual: Option<String> = match &self.args[0] {
                            ExprType::Name(n) => options.local_types.get(&n.id).cloned(),
                            // The shared map keeps this comparison in
                            // lockstep with the local_types PRODUCER; the
                            // "str" fallback for unmodeled literal types is
                            // this site's own historical behavior.
                            lit => crate::ast::tree::function_def::simple_expr_type(lit).map(
                                |ty| {
                                    crate::ast::tree::function_def::rust_type_to_py_name(&ty)
                                        .unwrap_or("str")
                                        .to_string()
                                },
                            ),
                        };
                        let Some(actual) = actual else {
                            // An expression whose type is not statically
                            // known (`isinstance(req.url, bytes)` where req
                            // is an external-class parameter): rython's
                            // static model cannot decide it, and there is no
                            // runtime type to dispatch on — the check is
                            // statically false (documented divergence, the
                            // class-as-value family).
                            return Ok(quote!(false));
                        };
                        // True when the argument's type is one of the target
                        // types (bool is a subclass of int in Python).
                        let result = targets.iter().any(|t| {
                            actual == *t
                                || (actual == "bool" && t == "int")
                                || (actual == "bytes" && t == "bytearray")
                                || (actual == "bytearray" && t == "bytes")
                        });
                        return Ok(if result { quote!(true) } else { quote!(false) });
                    }
                    // The by-reference builtins: their runtime functions
                    // borrow, and Python's calls never consume the value.
                    "print" => {
                        // print builds on py_display (Python's str
                        // semantics: True, 1e+16, unquoted strings) — the
                        // Display fallback would silently diverge.
                        let mut sep = None;
                        let mut end = None;
                        let mut flush = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("sep") if sep.is_none() => sep = Some(kw.value.clone()),
                                Some("end") if end.is_none() => end = Some(kw.value.clone()),
                                Some("flush") if flush.is_none() => flush = Some(kw.value.clone()),
                                Some("file") => {
                                    return Err("print(file=...) is not supported: \
                                                generated code writes to stdout only"
                                        .to_string()
                                        .into());
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        // sep=None / end=None mean the defaults in Python.
                        let sep = sep.filter(|s| !crate::is_none_expr(s));
                        let end = end.filter(|e| !crate::is_none_expr(e));
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        if sep.is_none() && end.is_none() && flush.is_none() {
                            match rendered.as_slice() {
                                [] => return Ok(quote!(println!())),
                                // A BYTES argument prints its CPython form
                                // (`b'ab'`), not the int-list the blanket
                                // Vec<T> display renders (issue #137): route
                                // through the runtime's verified
                                // py_bytes_repr.
                                [a] => {
                                    if crate::ast::tree::call::receiver_is_bytes_like(
                                        &self.args[0],
                                        &options,
                                        &symbols,
                                    ) {
                                        let runtime = crate::safe_ident(&options.stdpython);
                                        return Ok(quote!(print(&(#runtime::py_bytes_repr(&(#a))))));
                                    }
                                    return Ok(quote!(print(&(#a))));
                                }
                                _ => {}
                            }
                        }
                        let sep = match sep {
                            Some(s) => render(s)?,
                            None => quote!(" "),
                        };
                        let end = match end {
                            Some(e) => render(e)?,
                            None => quote!("\n"),
                        };
                        // print(end="") with no arguments still needs an
                        // element type for the empty parts slice.
                        let parts = if rendered.is_empty() {
                            quote!(&[] as &[&str])
                        } else {
                            quote!(&[#(py_display(&(#rendered))),*])
                        };
                        return Ok(match flush {
                            None => quote!(print_parts(#parts, #sep, #end)),
                            Some(f) => {
                                let f = render(f)?;
                                quote!(print_parts_flush(#parts, #sep, #end, #f))
                            }
                        });
                    }
                    "open" => {
                        // The runtime signature takes mode as an Option;
                        // the arity split wraps it here. Text modes only —
                        // the `encoding` keyword (pip's sitecustomize
                        // write) is accepted — rython's text mode is
                        // always UTF-8, so a non-UTF-8 encoding is the
                        // documented divergence; newline/binary spellings
                        // stay loud.
                        if self
                            .keywords
                            .iter()
                            .any(|k| {
                                k.arg
                                    .as_deref()
                                    .is_some_and(|a| a != "encoding" && a != "errors")
                            })
                        {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        return Ok(match rendered.as_slice() {
                            [p] => quote!(open(&(#p), None::<&str>)?),
                            [p, m] => quote!(open(&(#p), Some(#m))?),
                            _ => {
                                return Err("open() takes 1 or 2 arguments (path and mode)"
                                    .to_string()
                                    .into());
                            }
                        });
                    }
                    "len" | "repr" | "reversed" | "hash" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(format!("{}() takes exactly one argument", bname).into());
                        }
                        let f = format_ident!("{}", bname);
                        let a = &rendered[0];
                        // len() returns the runtime's usize length, but
                        // Python ints are i64 everywhere else (range(),
                        // indexing, arithmetic); cast so `len(s) + 1` and
                        // `xs[len(xs) - 1]` stay in i64 land.
                        if bname == "len" {
                            return Ok(quote!(#f(&(#a)) as i64));
                        }
                        return Ok(quote!(#f(&(#a))));
                    }
                    // iter(x): Python's iterator factory. rython's value
                    // model makes every value already iterable at its
                    // natural position (a for-loop over the value), so the
                    // factory is the identity — the argument boxes to a
                    // PyValue (urllib3's `chunks = iter(body)` in
                    // request.py's body_to_chunks).
                    "id" => {
                        // id(x): Python's object identity. rython's values
                        // are owned Rust structs with no stable CPython-like
                        // address; the identity lowers to the value's
                        // address cast to i64 — unique per value while it
                        // lives, matching the repr uses (urllib3's
                        // `f"<{self} at {id(self):#x}>"`). The exact number
                        // is not CPython's — the identity divergence.
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err("id() takes exactly one argument".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(quote!((&(#a)) as *const _ as i64));
                    }
                    // iter(x): Python's iterator factory. rython's value
                    // model makes every value already iterable at its
                    // natural position (a for-loop over the value), so the
                    // factory is the identity — the argument boxes to a
                    // PyValue (urllib3's `chunks = iter(body)` in
                    // request.py's body_to_chunks).
                    "iter" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() == 2 {
                            // Issue #155: iter(callable, sentinel) is
                            // supported only as a for-loop iterable, where
                            // it desugars to a call-until-sentinel loop
                            // (for_stmt.rs). As a bare value it would need
                            // an iterator object, which the value model
                            // does not have.
                            return Err(
                                "iter(callable, sentinel) is only supported as a \
                                 for-loop iterable (`for x in iter(f, sentinel):`), \
                                 where it lowers to a call-until-sentinel loop \
                                 (issue #155)"
                                    .to_string()
                                    .into(),
                            );
                        }
                        if rendered.len() != 1 {
                            return Err("iter() takes exactly one argument".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(quote!(PyValue::from(#a)));
                    }
                    // Dynamic attribute access on a boxed/foreign value
                    // (getattr/hasattr/setattr — requests' __getstate__,
                    // urllib3's ssl probing): rython's static model cannot
                    // look names up at runtime. The documented divergence:
                    // getattr(obj, name, default) lowers to the DEFAULT
                    // (boxed PyValue::None_ without one) — the lookup is
                    // dropped; hasattr(obj, name) is statically false
                    // (nothing is known about the value's members); and
                    // setattr(obj, name, v) is a no-op. All through the -W
                    // channel, never silent.
                    "getattr" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() < 2 || rendered.len() > 3 {
                            return Err("getattr() takes 2 or 3 arguments (object, name[, default])"
                                .to_string()
                                .into());
                        }
                        // getattr on a STDLIB MODULE with a LITERAL name
                        // resolves statically — the version-probing idiom
                        // (`getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN",
                        // 0x80000)` — urllib3's ssl_.py): the runtime item
                        // when the module has it, else the default (this
                        // mirrors the static import-guard decision).
                        if let (
                            ExprType::Name(m),
                            ExprType::Constant(c),
                        ) = (&self.args[0], &self.args[1])
                            && crate::ast::tree::import::is_stdpython_module(&m.id)
                            && let Some(litrs::Literal::String(slit)) = &c.0
                        {
                            let item = slit.value();
                            if crate::ast::tree::import::stdpython_module_item(&m.id, item)
                            {
                                let module = crate::safe_ident(&m.id);
                                let name = crate::safe_ident(item);
                                return Ok(quote!(#module::#name));
                            }
                            if let Some(d) = rendered.get(2) {
                                options.definition_warnings.borrow_mut().push(format!(
                                    "getattr({}, \"{}\", default): the runtime module \
                                     has no such item — statically the default",
                                    m.id, item
                                ));
                                return Ok(d.clone());
                            }
                        }
                        options.definition_warnings.borrow_mut().push(
                            "getattr(obj, name[, default]) is dropped; the default is \
                             returned (dynamic attribute lookup is unmodeled — the \
                             external-object divergence)"
                                .to_string(),
                        );
                        return Ok(match rendered.get(2) {
                            Some(d) => d.clone(),
                            None => quote!(stdpython::PyValue::None_),
                        });
                    }
                    "hasattr" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 2 {
                            return Err("hasattr() takes exactly 2 arguments (object, name)"
                                .to_string()
                                .into());
                        }
                        options.definition_warnings.borrow_mut().push(
                            "hasattr(obj, name) is statically false: rython's typed \
                             model knows nothing about the value's dynamic members \
                             (the external-object divergence)"
                                .to_string(),
                        );
                        return Ok(quote!(false));
                    }
                    "setattr" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 3 {
                            return Err("setattr() takes exactly 3 arguments (object, name, value)"
                                .to_string()
                                .into());
                        }
                        options.definition_warnings.borrow_mut().push(
                            "setattr(obj, name, value) is dropped: rython's typed \
                             model cannot write dynamic members (the external-object \
                             divergence)"
                                .to_string(),
                        );
                        return Ok(quote!(stdpython::PyValue::None_));
                    }
                    // `type(x)` — Python's class object. rython cannot
                    // represent classes as values, but the COMMON pattern
                    // `type(self).__name__` (a class name string for
                    // repr/error messages — urllib3, requests) resolves
                    // statically: `type(self)` inside a method is the
                    // enclosing class's name as a string literal. Any other
                    // `type(...)` call boxes as PyValue (the class-as-value
                    // divergence).
                    "type" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        // The 3-argument form `type(name, bases, dict)`
                        // CREATES a class at runtime (boto3's
                        // collection.py) — the class-as-value divergence:
                        // boxed PyValue.
                        if rendered.len() == 3 {
                            options.definition_warnings.borrow_mut().push(
                                "type(name, bases, dict) — runtime class creation — \
                                 lowers as the boxed PyValue (classes cannot be \
                                 runtime values in rython)"
                                    .to_string(),
                            );
                            return Ok(quote!(stdpython::PyValue::None_));
                        }
                        if rendered.len() != 1 {
                            return Err("type() takes exactly one argument".to_string().into());
                        }
                        if let Some(enclosing) = ctx.enclosing_class_name()
                            && matches!(
                                self.args.first(),
                                Some(ExprType::Name(n)) if n.id == "self"
                            )
                        {
                            let name = crate::safe_ident(enclosing);
                            return Ok(quote!(stringify!(#name).to_string()));
                        }
                        options.definition_warnings.borrow_mut().push(
                            "type(...) lowers as the boxed PyValue (classes cannot \
                             be runtime values in rython — the class-as-value \
                             divergence)"
                                .to_string(),
                        );
                        return Ok(quote!(stdpython::PyValue::None_));
                    }
                    // `set()` / `bytearray()` — constructors whose empty
                    // form has no inferable element type: the boxed None
                    // (the empty-set divergence); a non-empty `set([...])`
                    // keeps the runtime call.
                    "set" | "bytearray" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.is_empty() {
                            options.definition_warnings.borrow_mut().push(format!(
                                "an empty {}() lowers as the boxed None (the \
                                 empty-set divergence)",
                                bname
                            ));
                            return Ok(quote!(stdpython::PyValue::None_));
                        }
                        // A BOXED/unknown argument (`set(self._container.
                        // keys())` — urllib3's HTTPHeaderDict.keys, where
                        // the keys() member of the boxed dict is an
                        // unmodeled value): the set's content is unknown —
                        // the boxed None (the empty-set divergence family).
                        // The ident spelling (not the string literal —
                        // round 36 fixes the old `"set"(...)` string-call).
                        if rendered.iter().any(|r| {
                            r.to_string().contains("PyValue::None_")
                                || r.to_string().contains("PyValue :: None_")
                        }) {
                            options.definition_warnings.borrow_mut().push(format!(
                                "{}() over an unmodeled value lowers as the boxed \
                                 None (the set-content divergence)",
                                bname
                            ));
                            return Ok(quote!(stdpython::PyValue::None_));
                        }
                        let bname = crate::safe_ident(&bname);
                        return Ok(quote!(#bname(#(#rendered),*)));
                    }
                    "frozenset" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            // An EMPTY `frozenset()` (`excluded_params or
                            // frozenset()` — botocore's EndpointProvider):
                            // the element type is unknowable — a boxed
                            // None (the empty-set divergence).
                            options.definition_warnings.borrow_mut().push(
                                "an empty frozenset() lowers as the boxed None (the \
                                 empty-set divergence)"
                                    .to_string(),
                            );
                            return Ok(quote!(stdpython::PyValue::None_));
                        }
                        let a = &rendered[0];
                        return Ok(quote!(frozenset(#a)));
                    }
                    // map/filter dispatch on the FUNCTION argument's shape:
                    // lambdas are plain closures, while user-defined
                    // functions return Result and route through the
                    // fallible variants so their exceptions propagate.
                    "map" | "filter" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        // A lambda function argument's parameter is the
                        // iterable's element — typed like a `key=` lambda.
                        // `map(f, a, b)`: one parameter per iterable, in
                        // order.
                        let rendered = match self.args.first() {
                            Some(f @ ExprType::Lambda(_)) if self.args.len() >= 2 => {
                                let iterables: Vec<&ExprType> = self.args[1..].iter().collect();
                                let mut r = rendered.clone();
                                r[0] = render_lambda_over(f, &iterables, &ctx, &options, &symbols)?;
                                r
                            }
                            _ => rendered,
                        };
                        let fallible = matches!(self.args.first(), Some(ExprType::Name(f))
                            if matches!(symbols.get(&f.id), Some(SymbolTableNode::FunctionDef(_))));
                        if bname == "filter" {
                            if rendered.len() != 2 {
                                return Err("filter() takes a function and an iterable"
                                    .to_string()
                                    .into());
                            }
                            let (f, xs) = (&rendered[0], &rendered[1]);
                            // filter(None, xs) keeps the truthy elements.
                            if self.args.first().is_some_and(crate::is_none_expr) {
                                return Ok(quote!(filter_truthy(#xs)));
                            }
                            return Ok(if fallible {
                                quote!(filter_fallible(#f, #xs)?)
                            } else {
                                quote!(filter(#f, #xs))
                            });
                        }
                        return match rendered.as_slice() {
                            [f, xs] => {
                                // `map(str.lower, xs)` — an UNBOUND
                                // builtin-str method as the function
                                // argument (urllib3's request():
                                // `"content-type" in map(str.lower,
                                // headers.keys())`): the class-as-value
                                // model has no `str.lower` value (E0609)
                                // — the function lowers to a closure
                                // applying the bound method.
                                let unbound = match self.args.first() {
                                    Some(ExprType::Attribute(ub))
                                        if matches!(
                                            ub.value.as_ref(),
                                            ExprType::Name(n) if n.id == "str"
                                        ) && crate::ast::tree::type_ctx::StrMethod::from_name(&ub.attr)
                                            .is_some_and(|m| m.takes_only_receiver()) =>
                                    {
                                        Some(crate::safe_ident(&ub.attr))
                                    }
                                    _ => None,
                                };
                                let f = match unbound {
                                    Some(m) => quote!(|__rython_x| (__rython_x).#m()),
                                    None => f.clone(),
                                };
                                Ok(if fallible {
                                    quote!(map_fallible(#f, #xs)?)
                                } else {
                                    quote!(map(#f, #xs))
                                })
                            }
                            [f, a, b] => {
                                if fallible {
                                    return Err("map() over two iterables with a user-defined \
                                         function is not supported yet; use a lambda"
                                        .to_string()
                                        .into());
                                }
                                Ok(quote!(map2(#f, #a, #b)))
                            }
                            _ => Err("map() takes a function and 1-2 iterables"
                                .to_string()
                                .into()),
                        };
                    }
                    "list" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err("list() requires an iterable argument in rython (an \
                                 empty list has no inferable element type; use [])"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        return Ok(quote!(list(#a)));
                    }
                    // tuple(x): Python's tuple factory. rython's value
                    // model boxes tuples, so the factory is the boxed
                    // argument (urllib3's `context["socket_options"] =
                    // tuple(socket_opts)` in poolmanager.py).
                    "tuple" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err("tuple() requires an iterable argument in rython"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        return Ok(quote!(PyValue::from(#a)));
                    }
                    // next(iterable[, default]): Python's iterator advance.
                    // rython's eager generator model collects a generator
                    // body into a Vec, so next(vec) returns the FIRST
                    // element (the first yield) — or raises StopIteration
                    // when the generator produced nothing (requests'
                    // sessions.py: `r._next = next(self.resolve_redirects(
                    // ..., yield_requests=True))` inside try/except
                    // StopIteration). A default argument lowers to the
                    // default value instead of raising.
                    "next" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() < 1 || rendered.len() > 2 {
                            return Err("next() takes 1 or 2 arguments (iterator[, default])"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        let default = rendered.get(1);
                        return Ok(match default {
                            Some(d) => quote!({
                                let mut __rython_next_iter = (#a).into_iter();
                                match __rython_next_iter.next() {
                                    Some(__rython_next_v) => __rython_next_v,
                                    None => (#d),
                                }
                            }),
                            None => quote!({
                                let mut __rython_next_iter = (#a).into_iter();
                                __rython_next_iter
                                    .next()
                                    .ok_or_else(|| {
                                        PyException::new("StopIteration", String::new())
                                    })?
                            }),
                        });
                    }
                    // bytes(x): the byte representation. On a str|bytes
                    // union this extracts the bytes branch (idna's
                    // `label_bytes = bytes(label)` after the isinstance
                    // check); on a String it is UTF-8 bytes; on bytes it
                    // is the identity.
                    "bytes" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err("bytes() takes exactly 1 argument".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(quote!((#a).into_bytes_like()));
                    }
                    // str(x): Python display (py_display); str(bytes,
                    // encoding=...): decode bytes with the codec.
                    "str" => {
                        let mut encoding = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("encoding") if encoding.is_none() => {
                                    encoding = Some(kw.value.clone())
                                }
                                // The errors keyword (str(bytes, enc,
                                // errors="replace")) is accepted — the
                                // codec layer always decodes strictly.
                                Some("errors") => {}
                                other => return Err(unexpected(other)),
                            }
                        }
                        match (rendered.len(), encoding) {
                            // str(x): the runtime str() (PyToString bound —
                            // works for generic inferred params), EXCEPT a
                            // class instance, Option, or boxed value, whose
                            // str() is Python's DISPLAY (__str__/__repr__/
                            // the object repr — round 34): those route
                            // through py_display.
                            (1, None) => {
                                let a = &rendered[0];
                                // A GENERIC parameter keeps the runtime
                                // str() (PyToString bound — the generic
                                // machinery adds it); a concrete class
                                // instance, Option, or boxed value routes
                                // through py_display (round 34).
                                let generic = matches!(
                                    &self.args[0],
                                    crate::ExprType::Name(n)
                                        if options.param_type_vars.contains_key(&n.id)
                                );
                                if !generic
                                    && matches!(
                                        crate::infer_type(Some(&ctx), 
                                            &self.args[0],
                                            &options,
                                            &symbols
                                        ),
                                        crate::TypeInfo::Class(_)
                                            | crate::TypeInfo::Option(_)
                                            | crate::TypeInfo::PyValue
                                            | crate::TypeInfo::PyValueMember(_)
                                            | crate::TypeInfo::PyObject
                                    )
                                {
                                    return Ok(quote!(py_display(&(#a))));
                                }
                                return Ok(quote!(str(#a)));
                            }
                            // str(bytes, encoding=...) — decode the bytes
                            // with the codec (charset_normalizer's
                            // `str((...), encoding=encoding_iana)`).
                            (1, Some(enc)) => {
                                let a = &rendered[0];
                                let enc = enc.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::decode_by_name(&(#a), #enc)?
                                ));
                            }
                            (2, _) => {
                                let (a, enc) = (&rendered[0], &rendered[1]);
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::decode_by_name(&(#a), #enc)?
                                ));
                            }
                            // str(bytes, encoding, errors) — the errors
                            // argument is accepted (strict is Python's
                            // default; the codec layer always decodes
                            // strictly, so non-strict values diverge).
                            (3, _) => {
                                let (a, enc) = (&rendered[0], &rendered[1]);
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::decode_by_name(&(#a), #enc)?
                                ));
                            }
                            _ => {
                                return Err("str() takes at most 3 arguments"
                                    .to_string()
                                    .into())
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        // datetime constructors imported via `from datetime import ...`, or
        // module-qualified (`datetime.date(...)` / `datetime.datetime(...)` /
        // `datetime.timedelta(...)` — urllib3's connection module): calls
        // resolve their positional and keyword arguments against the Python
        // signatures and lower to the runtime ::new constructors
        // (Option-typed for the defaulted parameters). date/datetime
        // validate and propagate with `?`.
        let datetime_name: Option<&str> = match self.func.as_ref() {
            ExprType::Name(n) => {
                let from_datetime = matches!(
                    symbols.get(&n.id),
                    Some(SymbolTableNode::ImportFrom(import))
                        if crate::StdModule::from_name(&import.module)
                            == Some(crate::StdModule::Datetime)
                );
                if from_datetime {
                    Some(n.id.as_str())
                } else {
                    None
                }
            }
            ExprType::Attribute(a) => {
                // `datetime.date(...)`: the receiver is the stdlib module,
                // not shadowed by a user binding.
                let is_datetime_module = matches!(
                    a.value.as_ref(),
                    ExprType::Name(n)
                        if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Datetime)
                ) && !crate::module_name_shadowed(crate::StdModule::Datetime.name(), &symbols);
                if is_datetime_module {
                    Some(a.attr.as_str())
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(name) = datetime_name {
            if let Some(tokens) = render_datetime_ctor(
                name,
                &self.args,
                &self.keywords,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )? {
                return Ok(tokens);
            }
        }

        // itertools functions imported via `from itertools import ...`:
        // keyword spellings (initial=, repeat=, fillvalue=, key=) map to
        // arity-specific runtime variants, and iterable arguments are
        // borrowed (the runtime takes slices; Python calls never consume).
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_itertools = matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::ImportFrom(import))
                    if crate::StdModule::from_name(&import.module)
                        == Some(crate::StdModule::Itertools)
            );
            let handled = matches!(
                n.id.as_str(),
                "accumulate"
                    | "product"
                    | "zip_longest"
                    | "groupby"
                    | "pairwise"
                    | "combinations"
                    | "combinations_with_replacement"
                    | "permutations"
                    | "starmap"
                    | "takewhile"
                    | "dropwhile"
                    | "filterfalse"
            );
            if from_itertools && handled {
                let name = n.id.as_str();
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let render =
                    |e: crate::ExprType| e.to_rust(ctx.clone(), options.clone(), symbols.clone());
                let kw_of = |allowed: &[&str]| -> Result<
                    Vec<Option<crate::ExprType>>,
                    Box<dyn std::error::Error>,
                > {
                    let mut out: Vec<Option<crate::ExprType>> = vec![None; allowed.len()];
                    for kw in &self.keywords {
                        let idx = kw
                            .arg
                            .as_deref()
                            .and_then(|k| allowed.iter().position(|a| *a == k));
                        match idx {
                            Some(i) if out[i].is_none() => out[i] = Some(kw.value.clone()),
                            _ => {
                                return Err(format!(
                                    "{}() got an unexpected or duplicate keyword \
                                     argument '{}'",
                                    name,
                                    kw.arg.as_deref().unwrap_or("**kwargs")
                                )
                                .into());
                            }
                        }
                    }
                    Ok(out)
                };
                match name {
                    "accumulate" => {
                        let kws = kw_of(&["initial"])?;
                        let initial = kws.into_iter().next().unwrap();
                        let (xs, func) = match rendered.as_slice() {
                            [xs] => (xs.clone(), None),
                            [xs, f] => (xs.clone(), Some(f.clone())),
                            _ => {
                                return Err("accumulate() takes 1 or 2 positional \
                                            arguments"
                                    .to_string()
                                    .into());
                            }
                        };
                        return Ok(match (func, initial) {
                            (None, None) => { let f = crate::safe_ident(rt_variant::ACCUMULATE_SUM); quote!(#f(&(#xs))) },
                            (Some(f), None) => { let v = crate::safe_ident(rt_variant::ACCUMULATE_FUNC); quote!(#v(&(#xs), #f)) },
                            (None, Some(init)) => {
                                let init = render(init)?;
                                { let f = crate::safe_ident(rt_variant::ACCUMULATE_SUM_INITIAL); quote!(#f(&(#xs), #init)) }
                            }
                            (Some(f), Some(init)) => {
                                let init = render(init)?;
                                { let v = crate::safe_ident(rt_variant::ACCUMULATE_FUNC_INITIAL); quote!(#v(&(#xs), #f, #init)) }
                            }
                        });
                    }
                    "product" => {
                        let kws = kw_of(&["repeat"])?;
                        let repeat = kws.into_iter().next().unwrap();
                        if let Some(r) = repeat {
                            if rendered.len() != 1 {
                                return Err("product(iterable, repeat=n) takes one \
                                            iterable"
                                    .to_string()
                                    .into());
                            }
                            let r = render(r)?;
                            let xs = &rendered[0];
                            return match r.to_string().as_str() {
                                "2" => { let f = crate::safe_ident(rt_variant::PRODUCT_REPEAT2); Ok(quote!(#f(&(#xs)))) },
                                "3" => { let f = crate::safe_ident(rt_variant::PRODUCT_REPEAT3); Ok(quote!(#f(&(#xs)))) },
                                other => Err(format!(
                                    "product() repeat must be the literal 2 or 3 \
                                     (tuple arity is a compile-time shape); got {}",
                                    other
                                )
                                .into()),
                            };
                        }
                        return match rendered.as_slice() {
                            [a, b] => { let f = crate::safe_ident(rt_variant::PRODUCT2); Ok(quote!(#f(&(#a), &(#b)))) },
                            [a, b, c] => { let f = crate::safe_ident(rt_variant::PRODUCT3); Ok(quote!(#f(&(#a), &(#b), &(#c)))) },
                            _ => Err("product() supports 2 or 3 iterables, or one \
                                      iterable with repeat=2/3"
                                .to_string()
                                .into()),
                        };
                    }
                    "zip_longest" => {
                        let kws = kw_of(&["fillvalue"])?;
                        let fill = kws.into_iter().next().unwrap();
                        if rendered.len() != 2 {
                            // A dynamic-N `zip_longest(*rows, fillvalue=
                            // "")` (`pip`'s misc table formatting): the
                            // spread cannot be unpacked statically — the
                            // call is dropped (an empty iterable; the
                            // dynamic-arity divergence).
                            if self
                                .args
                                .iter()
                                .any(|a| matches!(a, ExprType::Starred(_)))
                            {
                                options.definition_warnings.borrow_mut().push(
                                    "zip_longest with a `*` spread is dropped (an empty \
                                     iterable; the dynamic-arity divergence)"
                                        .to_string(),
                                );
                                return Ok(quote!(Vec::<stdpython::PyValue>::new()));
                            }
                            return Err("zip_longest() supports exactly 2 iterables"
                                .to_string()
                                .into());
                        }
                        let (a, b) = (&rendered[0], &rendered[1]);
                        return Ok(match fill {
                            Some(v) => {
                                let v = render(v)?;
                                { let f = crate::safe_ident(rt_variant::ZIP_LONGEST_FILL); quote!(#f(&(#a), &(#b), #v)) }
                            }
                            None => quote!(zip_longest(&(#a), &(#b))),
                        });
                    }
                    "groupby" => {
                        let kws = kw_of(&["key"])?;
                        let mut key = kws.into_iter().next().unwrap();
                        if rendered.len() == 2 && key.is_none() {
                            // A POSITIONAL key (`groupby(strings, lambda s:
                            // s[0] == first[0])` — pygments' regexopt): the
                            // second argument is the key function.
                            key = Some(self.args[1].clone());
                        }
                        if rendered.is_empty() || rendered.len() > 2 {
                            return Err("groupby() takes one iterable and a key".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(match key {
                            Some(f) => {
                                let f = render(f)?;
                                { let v = crate::safe_ident(rt_variant::GROUPBY_KEY); quote!(#v(&(#xs), #f)) }
                            }
                            None => quote!(groupby(&(#xs))),
                        });
                    }
                    "pairwise" => {
                        kw_of(&[])?;
                        if rendered.len() != 1 {
                            return Err("pairwise() takes one iterable".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(quote!(pairwise(&(#xs))));
                    }
                    "combinations" | "combinations_with_replacement" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err(format!("{}() takes an iterable and r", name).into());
                        }
                        let f = format_ident!("{}", name);
                        let (xs, r) = (&rendered[0], &rendered[1]);
                        // Negative r raises ValueError, hence the `?`.
                        return Ok(quote!(#f(&(#xs), #r)?));
                    }
                    "permutations" => {
                        kw_of(&[])?;
                        return match rendered.as_slice() {
                            [xs] => Ok(quote!(permutations(&(#xs), None)?)),
                            [xs, r] => Ok(quote!(permutations(&(#xs), Some(#r))?)),
                            _ => Err("permutations() takes an iterable and optional r"
                                .to_string()
                                .into()),
                        };
                    }
                    "starmap" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err("starmap() takes a function and an iterable"
                                .to_string()
                                .into());
                        }
                        let (f, xs) = (&rendered[0], &rendered[1]);
                        return Ok(quote!(starmap(#f, &(#xs))));
                    }
                    // takewhile/dropwhile/filterfalse: the runtime takes
                    // (iterable, predicate) — Python's (predicate,
                    // iterable) order is swapped at the call site (urllib3's
                    // `takewhile(lambda x: ..., reversed(...))`).
                    "takewhile" | "dropwhile" | "filterfalse" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err(format!("{}() takes a predicate and an iterable", name)
                                .to_string()
                                .into());
                        }
                        let (pred, xs) = (&rendered[0], &rendered[1]);
                        let ident = crate::safe_ident(name);
                        return Ok(quote!(#ident(#xs, #pred)));
                    }
                    _ => unreachable!(),
                }
            }
        }

        // urllib.parse functions (round 55): `urlparse(url)`,
        // `urlsplit(url)`, `urljoin(base, url)`, `urlunparse(parts)`,
        // `urlencode(query, doseq=...)`, `quote(s, safe=...)`,
        // `quote_plus/unquote/unquote_plus(s)`, `urldefrag(url)`.
        // Reachable directly (`from urllib.parse import urlparse`) or via
        // a re-export chain (requests' compat re-exports them). The
        // runtime functions take string-like args (PyValue/String/&str
        // all work through AsStrLike) and return Result — the call
        // renders with `?` so exceptions propagate like Python.
        if let ExprType::Name(n) = self.func.as_ref()
            && let Some(fname) = urllib_parse_fn(&n.id, &symbols, &options)
        {
            let mut rendered = Vec::new();
            for arg in &self.args {
                rendered.push(arg.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            let mut doseq = quote!(false);
            let mut safe: Option<TokenStream> = None;
            for kw in &self.keywords {
                match (kw.arg.as_deref(), fname) {
                    (Some("doseq"), "urlencode") => {
                        doseq = kw.value.clone().to_rust(
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?;
                    }
                    (Some("safe"), "quote") | (Some("safe"), "quote_plus") => {
                        safe = Some(kw.value.clone().to_rust(
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?);
                    }
                    (Some(other), _) => {
                        return Err(format!(
                            "{}() got an unexpected keyword argument '{}'",
                            fname, other
                        )
                        .into());
                    }
                    (None, _) => {
                        return Err(format!("{}() does not accept **kwargs", fname).into());
                    }
                }
            }
            let f = crate::safe_ident(fname);
            let p = quote!(stdpython::urllib::parse::#f);
            return match fname {
                // The 6-part sequence: Python accepts a list or tuple of
                // str-or-None. A literal sequence renders as a boxed
                // tuple so the runtime can extract the six components;
                // a dynamic value passes through boxed.
                "urlunparse" => {
                    if rendered.len() != 1 || !self.keywords.is_empty() {
                        return Err(
                            "urlunparse() takes exactly one sequence argument".to_string().into(),
                        );
                    }
                    let parts = match self.args.first() {
                        Some(crate::ExprType::List(l)) => l.clone(),
                        Some(crate::ExprType::Tuple(t)) => t.elts.clone(),
                        _ => {
                            // A dynamic sequence: pass the boxed value.
                            let arg = &rendered[0];
                            return Ok(quote!(#p(&(PyValue::from(#arg)))?));
                        }
                    };
                    if parts.len() != 6 {
                        return Err(
                            "urlunparse() requires a 6-element sequence".to_string().into(),
                        );
                    }
                    let mut boxed = Vec::new();
                    for part in parts {
                        if crate::ast::tree::function_def::is_none_expr(&part) {
                            boxed.push(quote!(stdpython::PyValue::None_));
                        } else {
                            let r =
                                part.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                            boxed.push(quote!(PyValue::from(#r)));
                        }
                    }
                    Ok(quote!(#p(&(PyValue::from(vec![#(#boxed),*])))?))
                }
                "urlencode" => {
                    if rendered.is_empty() || rendered.len() > 1 {
                        return Err(
                            "urlencode() takes exactly one query argument".to_string().into(),
                        );
                    }
                    let q = &rendered[0];
                    Ok(quote!(#p(&(#q), #doseq)?))
                }
                "quote" => {
                    if rendered.is_empty() || rendered.len() > 1 {
                        return Err("quote() takes exactly one string argument".to_string().into());
                    }
                    let s = &rendered[0];
                    match safe {
                        Some(safe) => Ok(quote!(#p(&(#s), Some(&(#safe)))?)),
                        None => Ok(quote!(#p(&(#s), None)?)),
                    }
                }
                "quote_plus" => {
                    if rendered.is_empty() || rendered.len() > 1 || safe.is_some() {
                        return Err(
                            "quote_plus() takes exactly one string argument".to_string().into(),
                        );
                    }
                    let s = &rendered[0];
                    Ok(quote!(#p(&(#s))?))
                }
                "urljoin" => {
                    if rendered.len() != 2 {
                        return Err("urljoin() takes a base and a url".to_string().into());
                    }
                    let (base, url) = (&rendered[0], &rendered[1]);
                    Ok(quote!(#p(&(#base), &(#url))?))
                }
                _ => {
                    if rendered.len() != 1 {
                        return Err(format!("{}() takes exactly one string argument", fname)
                            .to_string()
                            .into());
                    }
                    let s = &rendered[0];
                    Ok(quote!(#p(&(#s))?))
                }
            };
        }

        // `from json import dumps; dumps(pyvalue, ...)` (charset_normalizer's
        // models.py — `dumps(self.__dict__, ensure_ascii=True, indent=4)`):
        // json does not dispatch from import, so the call would fall to the
        // generic path and mismatch the `&JSONValue` signature. The runtime
        // converts the boxed value (pyvalue_to_json) — round 55.
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_json = match symbols.get(&n.id) {
                Some(SymbolTableNode::ImportFrom(i)) =>
                    crate::StdModule::from_name(&i.module) == Some(crate::StdModule::Json),
                Some(SymbolTableNode::Alias(canonical)) => {
                    matches!(
                        symbols.get(canonical),
                        Some(SymbolTableNode::ImportFrom(i))
                            if crate::StdModule::from_name(&i.module)
                                == Some(crate::StdModule::Json)
                    )
                }
                _ => false,
            };
            if from_json && n.id == "dumps" {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let mut indent: Option<TokenStream> = None;
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        // ensure_ascii/check_circular/allow_nan/sort_keys:
                        // rython's encoder is already ASCII-safe; the
                        // remaining options are accepted and ignored (the
                        // documented divergence — CPython's exact float
                        // formatting and key sorting are unmodeled).
                        Some("indent") => {
                            indent = Some(kw.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?);
                        }
                        Some(_) | None => {}
                    }
                }
                let obj = match rendered.as_slice() {
                    [obj] => obj.clone(),
                    _ => {
                        return Err("json.dumps() takes exactly one object argument"
                            .to_string()
                            .into());
                    }
                };
                let indent = match indent {
                    Some(i) => quote!(Some(#i)),
                    None => quote!(None),
                };
                let p = quote!(stdpython::json::dumps_pyvalue);
                return Ok(quote!(#p(#obj, #indent)?));
            }
        }

        // functools.partial over a STATICALLY-KNOWN function: the
        // signature comes from the symbol table, so the call lowers to a
        // move closure binding the leading arguments, with the remaining
        // parameters keeping their Python names. The closure returns the
        // function's Result directly; calls through the bound name get
        // `?` (see propagates_exceptions). Dynamic first arguments have
        // no signature to consult and are a loud error.
        if is_partial_target(self.func.as_ref(), &symbols) {
            if let ExprType::Name(f) = &self.args[0]
                && let Some(SymbolTableNode::FunctionDef(fdef)) = symbols.get(&f.id)
            {
                let params: Vec<String> = fdef
                    .args
                    .posonlyargs
                    .iter()
                    .chain(fdef.args.args.iter())
                    .chain(fdef.args.kwonlyargs.iter())
                    .map(|p| p.arg.clone())
                    .collect();
                let bound_n = self.args.len() - 1;
                if bound_n > params.len() {
                    return Err(format!(
                        "functools.partial: `{}` takes {} argument(s), but {} were bound",
                        f.id,
                        params.len(),
                        bound_n
                    )
                    .into());
                }
                let mut bound = Vec::new();
                for (bi, arg) in self.args[1..].iter().enumerate() {
                    // A CLASS bound to a `type[...]`-annotated parameter
                    // (`functools.partial(_default_key_normalizer,
                    // PoolKey)` — urllib3's poolmanager, where key_class:
                    // type[PoolKey]): classes cannot be runtime values —
                    // the boxed None (the callables-as-data divergence).
                    let param = params.get(bi).and_then(|p| {
                        fdef.args
                            .posonlyargs
                            .iter()
                            .chain(fdef.args.args.iter())
                            .find(|pp| &pp.arg == p)
                    });
                    let is_type_param = param.is_some_and(|p| {
                        p.annotation.as_deref().is_some_and(|a| {
                            crate::ast::tree::arguments::is_type_annotation(a)
                                || matches!(
                                    a,
                                    crate::ExprType::Subscript(s)
                                        if matches!(s.value.as_ref(), crate::ExprType::Name(n) if n.id == "type")
                                )
                        })
                    });
                    if is_type_param && crate::is_class_value_expr(arg, &symbols) {
                        options.definition_warnings.borrow_mut().push(format!(
                            "functools.partial: class `{}` bound to a `type`-annotated \
                             parameter lowers to the boxed None (classes cannot be \
                             runtime values in rython)",
                            arg.clone()
                                .to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )
                                .map(|t| t.to_string())
                                .unwrap_or_else(|_| "<arg>".to_string())
                        ));
                        bound.push(quote!(stdpython::PyValue::None_));
                    } else {
                        bound.push(
                            arg.clone()
                                .to_rust(ctx.clone(), options.clone(), symbols.clone())?,
                        );
                    }
                }
                // Keyword bindings (`partial(f, a, k=v)`): bind the named
                // parameters too. They must be in the tail (a Rust call
                // cannot place a positional argument after a named one) and
                // must not collide with positionally-bound parameters.
                let mut kw_bindings: Vec<(String, TokenStream)> = Vec::new();
                for kw in &self.keywords {
                    let Some(kname) = &kw.arg else {
                        return Err(
                            "functools.partial with a `**d` spread is not supported"
                                .to_string()
                                .into(),
                        );
                    };
                    let idx = params
                        .iter()
                        .position(|p| p == kname)
                        .ok_or_else(|| {
                            format!(
                                "functools.partial: `{}` has no parameter `{}`",
                                f.id, kname
                            )
                        })?;
                    if idx < bound_n {
                        return Err(format!(
                            "functools.partial: `{}` is bound both positionally and by \
                             keyword",
                            kname
                        )
                        .into());
                    }
                    let value = kw
                        .value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    kw_bindings.push((kname.clone(), value));
                }
                // The call emits arguments in the CALLEE'S DECLARED
                // ORDER — for each declared parameter: its positional
                // value, else its keyword value, else the closure
                // parameter. Python's keyword bindings may bind any
                // subset in any order
                // (`partial(delay_exponential, base=base,
                // growth_factor=growth_factor)` leaves only `attempts`
                // unbound — botocore's retryhandler); emitting in the
                // callee's order is what makes the Rust call valid. (The
                // previous spelling placed keyword-bound values as
                // `ident: value` call arguments — not Rust — and required
                // them to form a suffix of the tail.)
                let fident = crate::safe_ident(&f.id);
                let mut call_args: Vec<TokenStream> = Vec::with_capacity(params.len());
                let mut closure_params: Vec<proc_macro2::Ident> = Vec::new();
                for (i, p) in params.iter().enumerate() {
                    if i < bound.len() {
                        call_args.push(bound[i].clone());
                        continue;
                    }
                    if let Some((_, value)) = kw_bindings.iter().find(|(k, _)| k == p) {
                        call_args.push(value.clone());
                        continue;
                    }
                    let ident = crate::safe_ident(p);
                    closure_params.push(ident.clone());
                    call_args.push(quote!(#ident));
                }
                return Ok(quote!(
                    move |#(#closure_params),*| #fident(#(#call_args),*)
                ));
            }
            // Any OTHER target — a class (`partial(AWSHTTPResponse,
            // status_tuple=...)` — botocore's AWSConnection), a
            // module-path (`partial(select.select, ...)` — urllib3's
            // wait.py), or an external/imported name — has no
            // statically-known signature: the callable-value divergence
            // (issue #122) — the partial is dropped with a warning, and
            // calls through the bound name lower to a plain call (dynamic
            // dispatch).
            options.definition_warnings.borrow_mut().push(format!(
                "functools.partial over a non-local function is dropped (the \
                 callable-as-value divergence, issue #122)"
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // functools/heapq/copy/textwrap/re functions: their runtime shapes
        // borrow (or mutably borrow) arguments, reduce() splits by arity,
        // and the re functions validate patterns at runtime (hence `?`).
        // Handled for both `from X import f; f(...)` and `import X;
        // X.f(...)` spellings. Which modules route here is a StdModule
        // property (dispatches_from_import / dispatches_qualified) — this
        // block previously carried three inline copies of the list.
        {
            let dispatches_qualified = |name: &str| {
                crate::StdModule::from_name(name).is_some_and(|m| m.dispatches_qualified())
            };
            let target: Option<(String, Option<&'static str>, String)> = match self.func.as_ref() {
                ExprType::Name(n) => {
                    // An ALIASED import (`from re import compile as
                    // re_compile` — charset_normalizer's constant.py):
                    // the symbol table binds the alias to the canonical
                    // name. The dispatch arms match the ORIGINAL name,
                    // but the generated call must use the BOUND name
                    // (only `re_compile` is in scope — the alias
                    // re-export; rendering `compile` would be E0425).
                    let resolved: Option<(String, String)> = match symbols.get(&n.id) {
                        Some(SymbolTableNode::ImportFrom(import)) => {
                            let canonical = import
                                .names
                                .iter()
                                .find(|a| a.asname.as_deref() == Some(&n.id))
                                .map(|a| a.name.clone())
                                .unwrap_or_else(|| n.id.clone());
                            Some((import.module.clone(), canonical))
                        }
                        Some(SymbolTableNode::Alias(canonical)) => {
                            match symbols.get(canonical) {
                                Some(SymbolTableNode::ImportFrom(import)) => {
                                    Some((import.module.clone(), canonical.clone()))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    match resolved {
                        Some((module, canonical))
                            if crate::StdModule::from_name(&module)
                                .is_some_and(|m| m.dispatches_from_import()) =>
                        {
                            Some((canonical, None, n.id.clone()))
                        }
                        _ => None,
                    }
                }
                ExprType::Attribute(attr) => {
                    // The module name may be an ALIAS (`import json as
                    // _json` — urllib3): follow it to the runtime module.
                    let root = match attr.value.as_ref() {
                        ExprType::Name(m) => {
                            if dispatches_qualified(&m.id)
                                && !module_name_shadowed(&m.id, &symbols)
                            {
                                m.id.clone()
                            } else {
                                match symbols.get(&m.id) {
                                    Some(SymbolTableNode::Alias(canonical)) => {
                                        canonical.clone()
                                    }
                                    _ => m.id.clone(),
                                }
                            }
                        }
                        _ => String::new(),
                    };
                    match crate::StdModule::from_name(&root) {
                        Some(module)
                            if module.dispatches_qualified()
                                && !module_name_shadowed(&root, &symbols) =>
                        {
                            Some((attr.attr.clone(), Some(module.name()), attr.attr.clone()))
                        }
                        _ => None,
                    }
                },
                _ => None,
            };
            let known = target.as_ref().is_some_and(|(f, _, _)| {
                matches!(
                    f.as_str(),
                    "reduce"
                        | "heappush"
                        | "heappop"
                        | "heapify"
                        | "heappushpop"
                        | "heapreplace"
                        | "nlargest"
                        | "nsmallest"
                        | "copy"
                        | "deepcopy"
                        | "dedent"
                        | "indent"
                        | "search"
                        | "match"
                        | "fullmatch"
                        | "findall"
                        | "finditer"
                        | "sub"
                        | "split"
                        | "compile"
                        | "md5"
                        | "sha1"
                        | "sha256"
                        | "sha512"
                        | "wrap"
                        | "fill"
                        | "reader"
                        | "writer"
                        | "StringIO"
                        | "BytesIO"
                        | "BufferedRWPair"
                        | "BufferedReader"
                        | "BufferedWriter"
                        | "TextIOWrapper"
                        | "DEFAULT_BUFFER_SIZE"
                )
            });
            if let (Some((fname, module_prefix, render_name)), true) = (target, known) {
                // wrap/fill accept width=, the re functions accept
                // flags= (and sub also count=); everything else takes no
                // keywords.
                let is_re_fn = matches!(
                    fname.as_str(),
                    "search" | "match" | "fullmatch" | "findall" | "finditer" | "sub" | "split" | "compile"
                );
                let mut width_kw: Option<crate::ExprType> = None;
                let mut flags_kw: Option<crate::ExprType> = None;
                let mut count_kw: Option<crate::ExprType> = None;
                let mut maxsplit_kw: Option<crate::ExprType> = None;
                let mut _usedforsecurity_kw: Option<crate::ExprType> = None;
                for kw in &self.keywords {
                    let slot = match kw.arg.as_deref() {
                        Some("width") if matches!(fname.as_str(), "wrap" | "fill") => &mut width_kw,
                        Some("flags") if is_re_fn => &mut flags_kw,
                        Some("count") if fname == "sub" => &mut count_kw,
                        Some("maxsplit") if fname == "split" => &mut maxsplit_kw,
                        // md5/sha's usedforsecurity is a FIPS policy flag —
                        // ignored (requests' digest auth).
                        Some("usedforsecurity")
                            if matches!(fname.as_str(), "md5" | "sha1" | "sha256" | "sha512") =>
                        {
                            &mut _usedforsecurity_kw
                        }
                        // md5/sha: a `*args` / `**kwargs` SPREAD
                        // (`hashlib.md5(*args, **kwargs)` — botocore's
                        // compat.get_md5, where both are empty in practice):
                        // the dynamic args are dropped (documented
                        // divergence).
                        None
                            if matches!(fname.as_str(), "md5" | "sha1" | "sha256" | "sha512") =>
                        {
                            &mut _usedforsecurity_kw
                        }
                        // deepcopy's `memo=` keyword (issue #154:
                        // `copy.deepcopy(params, memo=_ForgetfulDict())` —
                        // boto3's dynamodb transform) is dropped: rython's
                        // value model has no shared references, so every
                        // deepcopy already produces fresh objects — the
                        // forgetful-memo behavior. A REAL memo's
                        // dedup-within-one-call (shared refs inside `params`
                        // mapping to ONE new object) is not reproduced;
                        // that's the §12.3 aliasing divergence, surfaced
                        // through -W.
                        Some("memo") if fname == "deepcopy" => {
                            options.definition_warnings.borrow_mut().push(
                                "deepcopy(memo=...) is dropped: rython's value \
                                 semantics copy everything fresh (the forgetful-memo \
                                 behavior); a real memo's shared-reference dedup is \
                                 not reproduced (issue #154, the aliasing divergence)"
                                    .to_string(),
                            );
                            continue;
                        }
                        // csv reader/writer: a `**d` SPREAD of the class-level
                        // dialect defaults (`csv.reader(self.stream,
                        // **self.defaults)` — distlib's CSVReader, where
                        // CSVBase.defaults IS the excel default dialect):
                        // the spread's keys are dynamic at this lowering —
                        // dropped (the dialect-options divergence).
                        None if matches!(fname.as_str(), "reader" | "writer") => {
                            options.definition_warnings.borrow_mut().push(
                                "csv.reader/writer `**`-spread dialect options are \
                                 dropped; the excel default dialect is used \
                                 (documented divergence)"
                                    .to_string(),
                            );
                            continue;
                        }
                        _ => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                fname,
                                kw.arg.as_deref().unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    };
                    if slot.is_some() {
                        return Err(format!(
                            "{}() got multiple values for a keyword argument",
                            fname
                        )
                        .into());
                    }
                    *slot = Some(kw.value.clone());
                }
                // Python re flags lower to inline flag letters: single
                // constants or |-combinations of re.IGNORECASE/I,
                // re.MULTILINE/M, re.DOTALL/S. Anything else is loud.
                fn flag_letters(
                    symbols: &crate::SymbolTableScopes,
                    e: &crate::ExprType,
                ) -> Result<String, Box<dyn std::error::Error>> {
                    let name_of = |id: &str| -> Result<String, Box<dyn std::error::Error>> {
                        match id {
                            "IGNORECASE" | "I" => Ok("i".to_string()),
                            "MULTILINE" | "M" => Ok("m".to_string()),
                            "DOTALL" | "S" => Ok("s".to_string()),
                            // re.ASCII disables Unicode matching — rython's
                            // regex stays Unicode — a documented divergence
                            // (pip's cmdoptions). re.UNICODE is the DEFAULT
                            // (Unicode matching) — a no-op.
                            "ASCII" | "A" | "UNICODE" | "U" => Ok(String::new()),
                            other => Err(format!(
                                "unsupported re flag `{}`; supported: IGNORECASE,                                  MULTILINE, DOTALL (and | combinations)",
                                other
                            )
                            .into()),
                        }
                    };
                    match e {
                        ExprType::Attribute(a)
                            if matches!(a.value.as_ref(), ExprType::Name(m)
                                if crate::StdModule::from_name(&m.id) == Some(crate::StdModule::Re)
                                    && !module_name_shadowed(crate::StdModule::Re.name(), symbols)) =>
                        {
                            name_of(&a.attr)
                        }
                        ExprType::Name(n) => name_of(&n.id),
                        // `0` — no flags (the conditional `0 if ... else
                        // re.IGNORECASE` body).
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::Integer(_))) =>
                        {
                            let lit = c.0.clone().expect("matched integer");
                            let n: i64 = match &lit {
                                litrs::Literal::Integer(i) => i.value().ok_or(
                                    "re flags integer out of range",
                                )?,
                                _ => unreachable!(),
                            };
                            if n == 0 {
                                Ok(String::new())
                            } else {
                                Err(format!("unsupported re flags value {n}").into())
                            }
                        }
                        ExprType::BinOp(b)
                            if matches!(b.op, crate::ast::tree::bin_ops::BinOps::BitOr) =>
                        {
                            Ok(format!(
                                "{}{}",
                                flag_letters(symbols, &b.left)?,
                                flag_letters(symbols, &b.right)?
                            ))
                        }
                        // A CONDITIONAL flags expression (`flags=0 if
                        // case_sensitive else re.IGNORECASE` — rich's
                        // Text.highlight_words): both branches must resolve
                        // to the same flag set — take the non-empty one
                        // (0 = no flags).
                        ExprType::IfExp(i) => {
                            let body = flag_letters(symbols, &i.body)?;
                            let orelse = flag_letters(symbols, &i.orelse)?;
                            if body == orelse {
                                Ok(body)
                            } else if body.is_empty() {
                                Ok(orelse)
                            } else if orelse.is_empty() {
                                Ok(body)
                            } else {
                                Err("conditional re flags with different flag \
                                     sets are not supported"
                                    .to_string()
                                    .into())
                            }
                        }
                        other => Err(format!(
                            "unsupported re flags expression `{:?}`; use re.IGNORECASE,                              re.MULTILINE, re.DOTALL, or | combinations of them",
                            other
                        )
                        .into()),
                    }
                }
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                // The heap mutators take their first argument by &mut, so
                // it must be lowered as a PLACE: `heappush(rows[i], v)`
                // through the Load path would clone the element and the
                // push would silently vanish (the same clone-mutation bug
                // fixed for mutating methods on subscripted receivers).
                // rendered[0] becomes the full mutable-borrow expression:
                // py_index_mut already yields &mut for subscripts, names
                // take a fresh &mut.
                let heap_mutator = crate::ast::tree::scope::HEAPQ_FIRST_ARG_MUTATORS
                    .contains(&fname.as_str());
                if heap_mutator {
                    if let Some(first) = self.args.first() {
                        rendered[0] = if matches!(first, ExprType::Subscript(_)) {
                            crate::ast::tree::subscript::subscript_receiver_place(
                                first,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?
                        } else {
                            let v = &rendered[0];
                            quote!(&mut (#v))
                        };
                    }
                }
                let qual = |name: &str| {
                    // An aliased from-import (`compile as re_compile`):
                    // the canonical name is in the known list, but only
                    // the BOUND name is in scope — render it instead.
                    let f = if name == fname && render_name != fname {
                        crate::safe_ident(&render_name)
                    } else {
                        crate::safe_ident(name)
                    };
                    match module_prefix {
                        Some(m) => {
                            let m = format_ident!("{}", m);
                            quote!(#m::#f)
                        }
                        None => quote!(#f),
                    }
                };
                let arity = |expected: &str| -> Box<dyn std::error::Error> {
                    format!("{}() takes {} arguments", fname, expected).into()
                };
                return match (fname.as_str(), rendered.as_slice()) {
                    ("reduce", [f, xs]) => {
                        let p = qual("reduce");
                        Ok(quote!(#p(#f, &(#xs))?))
                    }
                    ("reduce", [f, xs, init]) => {
                        let p = qual(rt_variant::REDUCE_INITIAL);
                        Ok(quote!(#p(#f, &(#xs), #init)))
                    }
                    ("reduce", _) => Err(arity("2 or 3")),
                    ("heappush", [h, x]) => {
                        let p = qual("heappush");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heappop", [h]) => {
                        let p = qual("heappop");
                        Ok(quote!(#p(#h)?))
                    }
                    ("heapify", [h]) => {
                        let p = qual("heapify");
                        Ok(quote!(#p(#h)))
                    }
                    ("heappushpop", [h, x]) => {
                        let p = qual("heappushpop");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heapreplace", [h, x]) => {
                        let p = qual("heapreplace");
                        Ok(quote!(#p(#h, #x)?))
                    }
                    ("nlargest" | "nsmallest", [n_arg, xs]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(#n_arg, &(#xs))))
                    }
                    ("copy" | "deepcopy", [x]) | ("copy" | "deepcopy", [x, _]) => {
                        // The copy/deepcopy protocol's `memo` argument is
                        // dropped (`copy.deepcopy(self._mapping, memo)` —
                        // botocore's ConfigValueStore.__deepcopy__): the
                        // protocol is unmodeled.
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#x))))
                    }
                    ("dedent", [s]) => {
                        let p = qual("dedent");
                        Ok(quote!(#p(&(#s))))
                    }
                    // wrap/fill: width by position, keyword, or Python's
                    // default of 70. They validate width, hence `?`.
                    ("wrap" | "fill", [t]) | ("wrap" | "fill", [t, _]) => {
                        let p = qual(&fname);
                        let width = match (rendered.get(1), width_kw) {
                            (Some(_), Some(_)) => {
                                return Err(format!(
                                    "{}() got multiple values for argument 'width'",
                                    fname
                                )
                                .into());
                            }
                            (Some(w), None) => quote!(#w),
                            (None, Some(w)) => {
                                let w = w.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(#w)
                            }
                            (None, None) => quote!(70),
                        };
                        Ok(quote!(#p(&(#t), #width)?))
                    }
                    (
                        "compile",
                        [pat, ..],
                    ) => {
                        if rendered.len() > 2 {
                            return Err("compile() takes at most 2 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 1 && flags_kw.is_some() {
                            return Err("compile() got multiple values for argument 'flags'"
                                .to_string()
                                .into());
                        }
                        let flags = match (self.args.get(1), flags_kw) {
                            (Some(e), None) => flag_letters(&symbols, e)?,
                            (None, Some(e)) => flag_letters(&symbols, &e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        // Escape unescaped literal braces for the Rust
                        // engine (see the search/match arm).
                        let pat = match (&self.args[0], &pat) {
                            (ExprType::Constant(c), _)
                                if let Some(litrs::Literal::String(slit)) = &c.0 =>
                            {
                                let escaped = escape_regex_braces(&slit.value());
                                quote!(#escaped)
                            }
                            _ => pat.clone(),
                        };
                        let p = qual("compile");
                        Ok(quote!(#p(&(#pat), #flags)?))
                    }
                    (
                        "search" | "match" | "fullmatch" | "findall" | "finditer",
                        [pat, text, ..],
                    ) => {
                        if rendered.len() > 3 {
                            return Err(format!(
                                "{}() takes at most 3 positional arguments",
                                fname
                            )
                            .into());
                        }
                        if rendered.len() > 2 && flags_kw.is_some() {
                            return Err(format!(
                                "{}() got multiple values for argument 'flags'",
                                fname
                            )
                            .into());
                        }
                        let flags = match (self.args.get(2), flags_kw) {
                            (Some(e), None) => flag_letters(&symbols, e)?,
                            (None, Some(e)) => flag_letters(&symbols, &e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        // findall's result SHAPE depends on the pattern's
                        // capture-group count (strings for 0-1 groups,
                        // tuples beyond), so a literal pattern is compiled
                        // here at conversion time to pick the variant —
                        // which also surfaces bad patterns before the
                        // program ever runs. Non-literal patterns keep the
                        // string shape; 2+ groups there stay a loud
                        // runtime error.
                        let mut target = fname.clone();
                        let mut pat_lit: Option<String> = None;
                        if let ExprType::Constant(c) = &self.args[0] {
                            if let Some(litrs::Literal::String(slit)) = &c.0 {
                                // Python's regex treats an unescaped `{` that
                                // does not form a quantifier as a literal;
                                // Rust's regex crate treats it as a
                                // repetition start (`{(.*?)}` — botocore's
                                // serialize): escape the braces for the
                                // Rust engine, on EVERY entry point that
                                // takes a pattern.
                                let escaped = escape_regex_braces(&slit.value());
                                // findall's result SHAPE depends on the
                                // pattern's capture-group count (strings for
                                // 0-1 groups, tuples beyond), so it is also
                                // compiled here to pick the variant — which
                                // also surfaces bad patterns before the
                                // program ever runs. 2+ groups there stay a
                                // loud conversion error.
                                if fname == "findall" {
                                    let re = regex::Regex::new(&escaped).map_err(|e| {
                                        format!(
                                            "re.findall(): cannot compile pattern {:?}: {} \
                                             (the regex engine does not support Python's \
                                             backreferences or lookarounds)",
                                            slit.value(),
                                            e
                                        )
                                    })?;
                                    match re.captures_len() - 1 {
                                        0 | 1 => {}
                                        2 => target = rt_variant::FINDALL2.to_string(),
                                        3 => target = rt_variant::FINDALL3.to_string(),
                                        n => {
                                            return Err(format!(
                                                "re.findall() with {} capture groups is \
                                                 not supported yet (at most 3)",
                                                n
                                            )
                                            .into());
                                        }
                                    }
                                }
                                pat_lit = Some(escaped);
                            }
                        }
                        let p = qual(&target);
                        let pat = match &pat_lit {
                            Some(escaped) => quote!(#escaped),
                            None => pat.clone(),
                        };
                        Ok(quote!(#p(&(#pat), &(#text), #flags)?))
                    }
                    // re.split(pattern, string, maxsplit=0, flags=0):
                    // the THIRD positional is maxsplit, unlike the other
                    // re functions where it is flags.
                    ("split", [pat, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("split() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 2 && maxsplit_kw.is_some() {
                            return Err("split() got multiple values for argument 'maxsplit'"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 3 && flags_kw.is_some() {
                            return Err("split() got multiple values for argument 'flags'"
                                .to_string()
                                .into());
                        }
                        let maxsplit = match (rendered.get(2), maxsplit_kw) {
                            (Some(m), None) => quote!(#m),
                            (None, Some(m)) => {
                                let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(#m)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match (self.args.get(3), flags_kw) {
                            (Some(e), None) => flag_letters(&symbols, e)?,
                            (None, Some(e)) => flag_letters(&symbols, &e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        // Escape unescaped literal braces for the Rust
                        // engine (see the search/match arm).
                        let pat = match (&self.args[0], &pat) {
                            (ExprType::Constant(c), _)
                                if let Some(litrs::Literal::String(slit)) = &c.0 =>
                            {
                                let escaped = escape_regex_braces(&slit.value());
                                quote!(#escaped)
                            }
                            _ => pat.clone(),
                        };
                        let p = qual("split");
                        Ok(quote!(#p(&(#pat), &(#text), #maxsplit, #flags)?))
                    }
                    ("sub", [pat, repl, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("sub() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 3 && count_kw.is_some() {
                            return Err("sub() got multiple values for argument 'count'"
                                .to_string()
                                .into());
                        }
                        let count = match (rendered.get(3), count_kw) {
                            (Some(c), None) => quote!(#c),
                            (None, Some(c)) => {
                                let c = c.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(#c)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match flags_kw {
                            Some(e) => flag_letters(&symbols, &e)?,
                            None => String::new(),
                        };
                        // Escape unescaped literal braces for the Rust
                        // engine (see the search/match arm).
                        let pat = match (&self.args[0], &pat) {
                            (ExprType::Constant(c), _)
                                if let Some(litrs::Literal::String(slit)) = &c.0 =>
                            {
                                let escaped = escape_regex_braces(&slit.value());
                                quote!(#escaped)
                            }
                            _ => pat.clone(),
                        };
                        let p = qual("sub");
                        Ok(quote!(#p(&(#pat), &(#repl), &(#text), #count, #flags)?))
                    }
                    // hashlib constructors: with initial data, or the
                    // empty + update() idiom.
                    ("md5" | "sha1" | "sha256" | "sha512", [data]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#data))))
                    }

                    // io.StringIO: arity split — the seeded form starts
                    // with the cursor at 0, as in Python.
                    ("StringIO", []) => {
                        let p = qual("StringIO");
                        Ok(quote!(#p()))
                    }
                    ("StringIO", [initial]) => {
                        let p = qual(rt_variant::STRINGIO_SEEDED);
                        Ok(quote!(#p(&(#initial))))
                    }
                    // io.BytesIO: arity split like StringIO — the seeded
                    // form starts with the cursor at 0, as in Python.
                    ("BytesIO", []) => {
                        let p = qual("BytesIO");
                        Ok(quote!(#p()))
                    }
                    ("BytesIO", [initial]) => {
                        let p = qual(rt_variant::BYTESIO_SEEDED);
                        Ok(quote!(#p(&(#initial))))
                    }
                    // io.BufferedRWPair/BufferedReader/BufferedWriter/
                    // TextIOWrapper: buffered file-object wrappers (urllib3's
                    // ssltransport.makefile) — no rython equivalent — the
                    // boxed PyValue (the file-object divergence).
                    ("BufferedRWPair", _) | ("BufferedReader", _) | ("BufferedWriter", _)
                    | ("TextIOWrapper", _) => {
                        options.definition_warnings.borrow_mut().push(
                            "io.{}(...) lowers as the boxed PyValue (buffered \
                             file-object wrappers are unmodeled — the \
                             file-object divergence)"
                                .to_string(),
                        );
                        Ok(quote!(stdpython::PyValue::None_))
                    }
                    // io.DEFAULT_BUFFER_SIZE: a module constant — the
                    // buffering default — the boxed None (the constant is
                    // unmodeled; buffering is a no-op on the boxed wrapper).
                    ("DEFAULT_BUFFER_SIZE", _) => {
                        options.definition_warnings.borrow_mut().push(
                            "io.DEFAULT_BUFFER_SIZE lowers to the boxed None (the \
                             constant is unmodeled — buffering is a no-op on the \
                             boxed file-object divergence)"
                                .to_string(),
                        );
                        Ok(quote!(stdpython::PyValue::None_))
                    }
                    // csv.writer(f) borrows the file mutably for the
                    // writer's lifetime (scope analysis marks f mut).
                    ("writer", [f]) => {
                        let p = qual("writer");
                        Ok(quote!(#p(&mut (#f))))
                    }
                    ("reader", [lines]) => {
                        let p = qual("reader");
                        Ok(quote!(#p(&(#lines))?))
                    }
                    ("md5" | "sha1" | "sha256" | "sha512", []) => {
                        let p = qual(crate::ast::tree::std_module::hashlib_new_variant(&fname)
                            .expect("the arm above names exactly the registry algos"));
                        Ok(quote!(#p()))
                    }
                    ("indent", [s, prefix]) => {
                        let p = qual("indent");
                        Ok(quote!(#p(&(#s), &(#prefix))))
                    }
                    (
                        "heappush" | "heappushpop" | "heapreplace" | "indent" | "nlargest"
                        | "nsmallest",
                        _,
                    ) => Err(arity("2")),
                    _ => Err(arity("the documented number of")),
                };
            }
        }

        // Constructing a class instance: `Point(args)` lowers to
        // `Point::new(args)?`, with arguments resolved against __init__'s
        // signature (minus self) so keywords and defaults follow Python
        // call semantics. A derived class without its own __init__ uses the
        // first __init__ on its MRO (its synthesized constructor forwards
        // to it).
        //
        // An imported class name (`from .animals import Dog`) resolves
        // through the DEFINING module's AST (options.module_defs): the
        // import binding is just a name, but construction must map
        // arguments against the class's real __init__ signature, and an
        // inherited __init__ must resolve through the defining module's
        // base chain, not the importer's scope.
        // `type(self)(args)` — the class-object construction (`result =
        // type(self)(maybe_constructable)` — urllib3's
        // HTTPHeaderDict.__ror__): CPython constructs a new instance of
        // the runtime class. Lower exactly like `{Class}(args)` — rebuild
        // the call with the class name as the callee and run the same
        // construction lowering (full signature mapping).
        if let ExprType::Call(inner) = self.func.as_ref()
            && matches!(inner.func.as_ref(), ExprType::Name(f) if f.id == "type")
            && inner.args.len() == 1
            && matches!(
                inner.args.first(),
                Some(ExprType::Name(a)) if a.id == "self"
            )
            && let Some(enclosing) = ctx.enclosing_class_name()
        {
            let mut rewritten = self.clone();
            rewritten.func = Box::new(ExprType::Name(crate::Name {
                id: enclosing.to_string(),
            }));
            return rewritten.to_rust(ctx, options, symbols);
        }
        if let ExprType::Name(n) = self.func.as_ref() {
            if n.id == "RLResolver" {
            }
            let resolved = resolve_construction_class(&n.id, &symbols, &options);
            if n.id == "RLResolver" {
            }
            if let Some((class, class_symbols)) = resolved {
                // The CONSTRUCTED type's Rust name: for a @classmethod's
                // `cls(...)` the receiver identifier `cls` is not a Rust
                // type in scope — the class's OWN name is (urllib3's
                // Retry.from_int).
                // A LOCAL alias names the class it stands for (`Dial = Knob;
                // Dial()` — Devin review on #321); an imported name is bound
                // as written.
                let cname = match symbols.get(&n.id) {
                    Some(crate::SymbolTableNode::ClassDef(c)) => {
                        crate::safe_ident(&c.name)
                    }
                    Some(crate::SymbolTableNode::Alias(_))
                    | Some(crate::SymbolTableNode::Assign { value: ExprType::Name(_), .. }) => {
                        // Only a class THIS module defines: an import alias
                        // (`Timeout as TimeoutSauce` — requests' adapters)
                        // is bound under the alias by its `use`.
                        let canonical =
                            crate::ast::tree::hierarchy::canonical_class_name(&n.id, &symbols);
                        if matches!(symbols.get(&canonical), Some(crate::SymbolTableNode::ClassDef(_))) {
                            crate::safe_ident(&canonical)
                        } else {
                            crate::safe_ident(&n.id)
                        }
                    }
                    _ => crate::safe_ident(&n.id),
                };
                // An EXCEPTION class (`SSLError(e)` — urllib3's
                // connectionpool): exceptions are string-tagged PyException
                // values, not structs. Lower exactly like a `raise`
                // statement's exception expression — PyException::new with
                // the formatted message — so the value flows into `raise
                // new_e` and `except` matches by name.
                if crate::is_exception_class(&class) {
                    let msg = match self.args.len() {
                        0 => quote!(String::new()),
                        1 => {
                            let arg = self.args[0].clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            quote!(format!("{}", #arg))
                        }
                        _ => {
                            let args: Result<Vec<TokenStream>, Box<dyn std::error::Error>> =
                                self.args
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
                    return Ok(quote!(PyException::new(#cname, #msg)));
                }
                // A starred argument (`HTTPBasicAuth(*auth)` — requests)
                // spreads a tuple; the signature mapping cannot know the
                // arity, so the collection passes positionally (the
                // spread divergence, issue #122-family).
                if self
                    .args
                    .iter()
                    .any(|a| matches!(a, ExprType::Starred(_)))
                {
                    let mut args = Vec::new();
                    for arg in &self.args {
                        args.push(arg.clone().to_rust(
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                        )?);
                    }
                    return Ok(quote!(#cname::new(#(#args),*)?));
                }
                match class.method_on_mro_with_options("__init__", &class_symbols, &options) {
                    Some(init) => {
                        let mut sig = init.clone();
                        crate::strip_self(&mut sig.args);
                        // An ARGUMENT-MISMATCHED constructor (`RLResolver(
                        // provider, reporter)` where the alias resolved to
                        // the WRONG class's 10-parameter __init__ — pip's
                        // resolvelib, where a local class named `Resolver`
                        // shadows the imported alias's canonical): the
                        // arguments lower positionally (the
                        // dynamic-dispatch divergence) instead of failing.
                        let mapped = map_call_arguments_inner(
                            &sig,
                            &self.args,
                            &self.keywords,
                            &ctx,
                            &options,
                            // Argument expressions render in the caller's
                            // scope; dropped-default constants resolve in
                            // the defining module's scope.
                            &symbols,
                            Some(&class.name),
                            Some(&class_symbols),
                        );
                        let MappedArguments { prelude, args } = match mapped {
                            Ok(m) => m,
                            Err(_) => {
                                options.definition_warnings.borrow_mut().push(format!(
                                    "constructor `{}` argument mapping failed; the \
                                     arguments lower positionally (the \
                                     dynamic-dispatch divergence)",
                                    n.id
                                ));
                                let mut args = Vec::new();
                                for arg in &self.args {
                                    args.push(arg.clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )?);
                                }
                                for kw in &self.keywords {
                                    args.push(kw.value.clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )?);
                                }
                                MappedArguments {
                                    prelude: TokenStream::new(),
                                    args,
                                }
                            }
                        };
                        // A SHARED class (shared.rs) constructs behind
                        // `PyRef`; a shared ROOT's value is its sum type
                        // holding the reference.
                        if crate::ast::tree::shared::is_shared(&class.name) {
                            let shared = quote!(stdpython::PyRef::new(#cname::new(#(#args),*)?));
                            if crate::ast::tree::hierarchy::is_polymorphic_root(&class.name) {
                                let any = crate::ast::tree::hierarchy::any_ident(&class.name);
                                return Ok(quote!({ #prelude #any::from(#shared) }));
                            }
                            return Ok(quote!({ #prelude #shared }));
                        }
                        return Ok(quote!({ #prelude #cname::new(#(#args),*)? }));
                    }
                    None => {
                        // The class has NO own __init__ and its base's
                        // __init__ is not resolvable (an imported base whose
                        // module isn't converted/reachable — boto3's
                        // ResourceShapeDocumenter, order-dependent): lower
                        // the kwargs positionally (the dynamic-dispatch
                        // divergence) — the base __init__ stores are
                        // unmodeled.
                        if !self.args.is_empty() || !self.keywords.is_empty() {
                            let mut args = Vec::new();
                            for arg in &self.args {
                                args.push(arg.clone().to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?);
                            }
                            for kw in &self.keywords {
                                args.push(kw.value.clone().to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?);
                            }
                            options.definition_warnings.borrow_mut().push(format!(
                                "class `{}` has no resolvable __init__ (an imported \
                                 base's constructor is unmodeled); the arguments lower \
                                 positionally (documented divergence)",
                                n.id
                            ));
                            return Ok(quote!(#cname::new(#(#args),*)?));
                        }
                        return Ok(quote!(#cname::new()?));
                    }
                }
            }
            // A stdpython class (`from collections import OrderedDict`
            // — `OrderedDict()` in requests' sessions/structures, or a
            // re-export chain through a sibling module — `from
            // .compat import OrderedDict` where compat re-exports
            // collections'): stdpython classes aren't in module_defs,
            // so the signature mapping above cannot resolve; lower to
            // the runtime `::new` constructor with the arguments passed
            // positionally (the stdpython-construction divergence).
            // A from-imported stdlib FUNCTION (`from socket import
            // getdefaulttimeout` — urllib3's util/timeout) is a direct
            // call, not a `::new` construction; the socket module's
            // items are functions except the constants.
            let stdpython_fn = matches!(symbols.get(&n.id),
                Some(crate::SymbolTableNode::ImportFrom(ifm))
                    if crate::StdModule::from_name(&ifm.module)
                        == Some(crate::StdModule::Socket)
                        && matches!(
                            n.id.as_str(),
                            "getdefaulttimeout" | "setdefaulttimeout" | "gethostname"
                        ));
            // `stdpython_class`: the from-imported (or re-exported) item is
            // a CLASS in the runtime — construction lowers to `X::new(...)`.
            // Function items (urlparse, quote, re.compile, warnings.warn,
            // json.dumps, ...) are plain calls: treating them as classes
            // produced `urlparse::new(...)` (E0433 — a function used as a
            // module path) at every requests/urllib3/charset_normalizer call
            // site (round 55). Direct from-imports check the module's class
            // registry; re-export chains (requests' compat re-exports
            // urllib.parse's functions) check the terminal item.
            let stdpython_class = !stdpython_fn
                && match symbols.get(&n.id) {
                    Some(crate::SymbolTableNode::ImportFrom(ifm)) => {
                        let root = ifm.module.split('.').next().unwrap_or("");
                        if crate::is_stdpython_module(root) {
                            // The imported name may be an ALIAS (`from
                            // socket import socket as socket_cls`): the
                            // registry key is the ORIGINAL item name.
                            let canonical = ifm
                                .names
                                .iter()
                                .find(|a| a.asname.as_deref() == Some(&n.id))
                                .map(|a| a.name.as_str())
                                .unwrap_or(&n.id);
                            crate::ast::tree::import::stdpython_module_class(root, canonical)
                        } else {
                            stdpython_reexport_chain(&n.id, &symbols, &options).is_some_and(
                                |(module, name)| {
                                    crate::ast::tree::import::stdpython_module_class(
                                        &module, &name,
                                    )
                                },
                            )
                        }
                    }
                    Some(crate::SymbolTableNode::Alias(canonical)) => {
                        stdpython_reexport_chain(canonical, &symbols, &options).is_some_and(
                            |(module, name)| {
                                crate::ast::tree::import::stdpython_module_class(&module, &name)
                            },
                        )
                    }
                    _ => false,
                };
            if stdpython_class {
                let cname = crate::safe_ident(&n.id);
                let mut args = Vec::new();
                for arg in &self.args {
                    args.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                for kw in &self.keywords {
                    args.push(kw.value.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                if args.is_empty() {
                    return Ok(quote!(#cname::new()));
                }
                return Ok(quote!(#cname::new(#(#args),*)));
            }
        }

        // `module.Class(args)` where `module` is a SIBLING module and
        // `Class` is a class it defines (`sessions.Session()` — requests'
        // api.py, `from . import sessions`): the module-path call must
        // lower to the class constructor `Session::new(args)?`, not
        // `sessions::Session(args)` (E0423 — a struct used as a value).
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && let ExprType::Name(receiver) = attr.value.as_ref()
            && !crate::module_name_shadowed(&receiver.id, &symbols)
            && let Some(crate::SymbolTableNode::ImportFrom(ifm)) = symbols.get(&receiver.id)
            && ifm.level > 0
        {
            let mut mod_path = ifm.resolved_module_path(&options);
            mod_path.push(receiver.id.clone());
            if options.module_defs.contains_key(&mod_path)
                && let Some((_class, _cs)) =
                    crate::module_class_def(&options, &mod_path, &attr.attr)
            {
                // The FULL crate path: `sessions.Session()` renders as
                // `crate::requests::sessions::Session::new()` — a bare
                // `Session` is only in scope if the module imports the
                // class itself, which it may not (`from . import
                // sessions` binds the MODULE, requests' api.py).
                let cname = crate::safe_ident(&attr.attr);
                let path_parts = mod_path.iter().map(|p| crate::safe_ident(p));
                let mut args = Vec::new();
                for arg in &self.args {
                    args.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                for kw in &self.keywords {
                    args.push(kw.value.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                if args.is_empty() {
                    return Ok(quote!(crate::#(#path_parts)::*::#cname::new()?));
                }
                return Ok(quote!(crate::#(#path_parts)::*::#cname::new(#(#args),*)?));
            }
        }

        // `Class.method(args)` where `method` is a @classmethod/
        // @staticmethod (issue #117): an ASSOCIATED call — the class name
        // is a type, so the call lowers to `Class::method(args)` with the
        // class reference (cls) dropped from the callee signature. The
        // class may be IMPORTED (`Retry.from_int(...)` — requests/adapters
        // uses urllib3's Retry): resolve through the defining module, the
        // same path construction calls use.
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && let ExprType::Name(receiver) = attr.value.as_ref()
        {
            // Same-module class, or an imported one resolved through its
            // defining module.
            let resolved = match symbols.get(&receiver.id) {
                Some(crate::SymbolTableNode::ClassDef(c)) => {
                    Some((c.clone(), symbols.clone()))
                }
                _ => resolve_construction_class(&receiver.id, &symbols, &options),
            };

            if let Some((class, class_symbols)) = resolved
                && let Some(method) = class.method_on_mro(&attr.attr, &class_symbols)
                && (matches!(method.decorator_list.as_slice(), [ExprType::Name(n)] if n.id == "classmethod")
                    || matches!(method.decorator_list.as_slice(), [ExprType::Name(n)] if n.id == "staticmethod"))
            {
                let mut sig = method.clone();
                if matches!(method.decorator_list.as_slice(), [ExprType::Name(n)] if n.id == "classmethod")
                {
                    // Drop the class-reference parameter (cls/self).
                    if !sig.args.posonlyargs.is_empty() {
                        sig.args.posonlyargs.remove(0);
                    } else if !sig.args.args.is_empty() {
                        sig.args.args.remove(0);
                    }
                }
                let MappedArguments { prelude, args } = map_call_arguments(
                    &sig,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                let cname = crate::safe_ident(&receiver.id);
                let method_name = crate::safe_ident(&attr.attr);
                return Ok(quote!({ #prelude #cname::#method_name(#(#args),*)? }));
            }
            // An UNBOUND-method call (`RequestMethods.__init__(self,
            // headers)` — urllib3's connectionpool, where HTTPConnectionPool
            // extends RequestMethods and explicitly initializes the mixin):
            // the class name is a type and the instance is the first
            // argument — the call lowers to the associated `Class::method(
            // self, ...)`, with the receiver kept in the args (Python binds
            // it to the first parameter). The generated method is
            // `pub(crate) fn __init__(&mut self, ...)`, so passing `self`
            // positionally works. Re-resolves the class (the first arm
            // moved `resolved`); only when the first arm's classmethod/
            // staticmethod guard did NOT match (it returns on success).
            if let Some((class, class_symbols)) = (match symbols.get(&receiver.id) {
                    Some(crate::SymbolTableNode::ClassDef(c)) => {
                        Some((c.clone(), symbols.clone()))
                    }
                    _ => resolve_construction_class(&receiver.id, &symbols, &options),
                })
                && let Some(method) = class.method_on_mro(&attr.attr, &class_symbols)
                && attr.attr == "__init__"
                && !method.decorator_list.iter().any(|d| {
                    matches!(d, ExprType::Name(n) if n.id == "classmethod" || n.id == "staticmethod")
                })
            {
                let MappedArguments { prelude, args } = map_call_arguments(
                    &method,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                let cname = crate::safe_ident(&receiver.id);
                let method_name = crate::safe_ident(&attr.attr);
                return Ok(quote!({ #prelude #cname::#method_name(#(#args),*)? }));
            }
        }

        // Python methods whose Rust inherent namesakes have DIFFERENT
        // semantics (or the wrong shape) are rewritten here; methods with no
        // Rust conflict resolve through the stdpython PyListOps/PyStrOps
        // traits without any rewriting.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            // `super().m(args)` — calls the DIRECT base's implementation of
            // `m` on the embedded base part: `self.__rython_base.m(args)`
            // (inherent context) or `self.base().m(args)` (generic trait
            // default). Argument mapping uses the base's signature, exactly
            // like any other user-class method call. The receiver is
            // `super()` — a bare call — not a bare name.
            let is_super = match attr.value.as_ref() {
                ExprType::Name(n) => n.id == "super",
                ExprType::Call(c) => {
                    matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "super")
                        && c.args.is_empty()
                        && c.keywords.is_empty()
                }
                _ => false,
            };
            if is_super {
                let Some(enclosing) = ctx.enclosing_class_name() else {
                    return Err("super() used outside a method".to_string().into());
                };
                // The class whose base `super()` targets: the class whose
                // override body we are emitting (the super_target for a
                // re-emitted override), or the enclosing class itself.
                let super_owner = match &ctx {
                    CodeGenContext::Trait {
                        super_target: Some(definer),
                        ..
                    } => definer.clone(),
                    _ => enclosing.to_string(),
                };
                let class = match symbols.get(&super_owner) {
                    Some(SymbolTableNode::ClassDef(c)) => c.clone(),
                    // A CROSS-MODULE definer (a re-emitted override whose
                    // defining class was imported — `SOCKSHTTPSConnectionPool`
                    // re-emitting `HTTPSConnectionPool`'s method, urllib3's
                    // contrib/socks): the class resolves through the
                    // defining module, not the current scope.
                    Some(SymbolTableNode::ImportFrom(i)) => {
                        let path = i.resolved_module_path(&options);
                        match crate::resolve_imported_class(&options, &path, &super_owner, 0) {
                            Some((c, _)) => c,
                            None => {
                                return Err(format!(
                                    "super() used outside a class method (`{}` is not a class)",
                                    super_owner
                                )
                                .into());
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "super() used outside a class method (`{}` is not a class)",
                            super_owner
                        )
                        .into());
                    }
                };
                let base = match class.base_class(&symbols) {
                    Some(b) => b,
                    None => {
                        // `super()` in a class with NO structural base
                        // (BaseAdapter, or a typing/metadata base like
                        // HTTPHeaderDict(MutableMapping)) resolves to
                        // object. A no-argument __init__ call is a no-op;
                        // any other call lowers to a call on `self` — the
                        // base implementation is unmodeled, so the method
                        // resolves to the class's own (a documented
                        // divergence; Python would use the base's).
                        if attr.attr == "__init__" && self.args.is_empty() && self.keywords.is_empty()
                        {
                            return Ok(quote!(()));
                        }
                        // Round 88: `super().__init__(args)` against an
                        // UNMODELED base (urllib3's `_HTTPConnection`
                        // inherits `http.client.HTTPConnection` — an
                        // external class whose constructor would set up
                        // the socket rython cannot model) must NOT fall to
                        // the self-call below: that renders a
                        // SELF-RECURSIVE `self.__init__(raw args)` call
                        // against the class's OWN 8-parameter signature
                        // with 5 args (E0061). The external constructor is
                        // unmodeled — a no-op with the definition warning,
                        // the same documented divergence as the
                        // base-method-unmodeled path below.
                        if attr.attr == "__init__" {
                            options.definition_warnings.borrow_mut().push(format!(
                                "super().__init__(...) in `{}` is dropped: the base's \
                                 constructor is unmodeled (an external or \
                                 non-structural base)",
                                class.name
                            ));
                            return Ok(quote!(stdpython::PyValue::None_));
                        }
                        let mut args = Vec::new();
                        for arg in &self.args {
                            args.push(arg.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?);
                        }
                        for kw in &self.keywords {
                            args.push(kw.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?);
                        }
                        let method_name = crate::safe_ident(&attr.attr);
                        // The `?` is essential: the recursive call is
                        // fallible (every user method returns Result), and
                        // WITHOUT it the Result leaks into the caller's
                        // value and every later `.attr`/method on it fails
                        // (urllib3's `httplib_response =
                        // super().getresponse()` then
                        // `httplib_response.msg` — E0609 on
                        // `Result<HTTPResponse, PyException>`).
                        return Ok(quote!(self.#method_name(#(#args),*)?));
                    }
                };
                let method = match base.method_on_mro(&attr.attr, &symbols) {
                    Some(m) => m,
                    None => {
                        // `super().__init__(*args, **kwargs)` where NO class
                        // in the base chain defines `__init__` (the chain
                        // bottoms out in an external base —
                        // optparse.OptionParser — pip's ConfigOptionParser):
                        // the external constructor is unmodeled — a no-op
                        // (documented divergence). Any OTHER unresolvable
                        // base method (`super().cert_verify(...)` — pip's
                        // InsecureHTTPAdapter, where the vendored base's
                        // method is unmodeled) drops the same way.
                        options.definition_warnings.borrow_mut().push(format!(
                            "super().{}(...) in `{}` is dropped: the base `{}`'s method \
                             is unmodeled",
                            attr.attr, class.name, base.name
                        ));
                        return Ok(quote!(stdpython::PyValue::None_));
                    }
                };
                let mut sig = method;
                crate::strip_self(&mut sig.args);
                let MappedArguments { prelude, args } = map_call_arguments(
                    &sig,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                // `super().__init__(...)` stays on the embedded base struct:
                // __init__ is a constructor, not a virtual method — it must
                // write the base's fields through the base struct (`new`
                // runs the chain one level at a time, so the base part is
                // the real receiver).
                if attr.attr == "__init__" {
                    // The embedded base of `super_owner` is reached from
                    // `self` by walking to `super_owner`'s struct, then one
                    // more level for its base. In a generic trait default
                    // the base accessor reaches the direct base of the
                    // generic Self — the MUTABLE accessor when the callee
                    // mutates self (`super().bump()` must not borrow the
                    // shared base and then fail to mutate).
                    let mutates_receiver =
                        base.method_needs_mut_self(&attr.attr, &symbols, &options);
                    let receiver = if ctx.in_generic_trait() {
                        if mutates_receiver {
                            quote!(self.base_mut())
                        } else {
                            quote!(self.base())
                        }
                    } else if super_owner == enclosing {
                        quote!(self.__rython_base)
                    } else {
                        // Re-emitted override: walk from the derived struct
                        // to the definer's struct, then one level deeper for
                        // the definer's own embedded base.
                        let enclosing_class = match symbols.get(enclosing) {
                            Some(SymbolTableNode::ClassDef(c)) => c.clone(),
                            _ => {
                                return Err(format!(
                                    "super() in `{}`: enclosing class `{}` is not a class",
                                    super_owner, enclosing
                                )
                                .into());
                            }
                        };
                        let chain = enclosing_class.base_chain(&symbols);
                        let depth = chain
                            .iter()
                            .position(|c| c.name == super_owner)
                            .map_or(0, |d| d + 1);
                        let chain_tokens = crate::base_field_chain(depth);
                        quote!(self #chain_tokens)
                    };
                    let method_name = crate::safe_ident(&attr.attr);
                    return Ok(quote!({ #prelude (#receiver).#method_name(#(#args),*)? }));
                }
                // Non-init `super().m(...)`: dispatch through the DEFINER's
                // super trampoline with the plain derived `self`, so the
                // ancestor's original body runs with `Self` = the most-
                // derived type and nested `self.x()` keeps resolving to the
                // derived class's override (CPython's MRO). The trampoline
                // is a uniquely-named trait default no override can
                // intercept — see ClassDef::emit_trait.
                let definer = base
                    .base_chain(&symbols)
                    .into_iter()
                    .find(|c| c.methods().any(|mm| mm.name == attr.attr))
                    .expect("super(): method_on_mro found the method, so its definer exists");
                let definer_trait =
                    crate::safe_ident(&format!("{}Trait", definer.name));
                let helper = crate::safe_ident(&format!("__rython_super_{}", attr.attr));
                return Ok(quote!({ #prelude <Self as #definer_trait>::#helper(self, #(#args),*)? }));
            }
            // A method call on a receiver whose class is known — `self`
            // inside a method, or a name assigned a construction — resolves
            // against the class's MRO FIRST, so a user-defined method named
            // like a builtin (`get`, `pop`, ...) is not rewritten out from
            // under the class (and an inherited method keeps its `?` and
            // keyword/default mapping). Calls propagate exceptions (`?`)
            // and map keywords/defaults like any user function call.
            if let Some((class, class_symbols)) =
                receiver_class(&attr.value, &ctx, &symbols, &options)
            {
                eprintln!("R99GATE {}.{} -> {}", match attr.value.as_ref() { crate::ExprType::Name(n) => format!("Name({})", n.id), crate::ExprType::Attribute(a) => format!("Attr(.{})", a.attr), crate::ExprType::Call(_) => "Call".into(), _ => "other".into() }, attr.attr, class.name);
                if let Some(method) =
                    class.method_on_mro_with_options(&attr.attr, &class_symbols, &options)
                {
                    if method.name == "__init__" && class.init_method().is_none() {
                        return Err(format!(
                            "`self.__init__(...)` calling an inherited `__init__` is not \
                             supported; use `super().__init__(...)`"
                        )
                        .into());
                    }
                    let mut sig = method;
                    // A @classmethod's first parameter is `cls` (not
                    // `self`): it is dropped once here. A @staticmethod has
                    // NO receiver — no parameter is dropped. An instance
                    // method's receiver is dropped by strip_self below,
                    // which removes the leading positional parameter
                    // whatever its name — Python binds the instance to the
                    // first parameter, so boto3's `factory_self` is a
                    // receiver too.
                    let is_static = sig.decorator_list.iter().any(|d| {
                        matches!(d, ExprType::Name(n) if n.id == "staticmethod")
                    });
                    let is_classmethod = sig.decorator_list.iter().any(|d| {
                        matches!(d, ExprType::Name(n) if n.id == "classmethod")
                    });
                    if is_classmethod {
                        if !sig.args.posonlyargs.is_empty() {
                            sig.args.posonlyargs.remove(0);
                        } else if !sig.args.args.is_empty() {
                            sig.args.args.remove(0);
                        }
                    } else if !is_static {
                        crate::strip_self(&mut sig.args);
                    }
                    // An ABSTRACT-STUB call: the resolved method is a
                    // raise-only NotImplementedError stub whose signature
                    // has MORE parameters than the call supplies
                    // (`self._do_modeled_error_parse(response, shape)` —
                    // botocore's parsers, where the BASE stub takes an
                    // extra `parsed` that only derived overrides lack):
                    // the stub cannot be invoked at this arity — the call
                    // is dropped (the abstract protocol is unmodeled;
                    // documented divergence).
                    let supplied = self.args.len() + self.keywords.len();
                    let stub_params = sig.args.posonlyargs.len() + sig.args.args.len()
                        + sig.args.kwonlyargs.len()
                        + usize::from(sig.args.vararg.is_some())
                        + usize::from(sig.args.kwarg.is_some());
                    // A call through a DYNAMIC field sharing a method's
                    // name (`self._s3_addressing_handler(request=request,
                    // **kwargs)` — botocore's S3EndpointSetter, where the
                    // field is registered externally and never stored in
                    // __init__): the class has no stored field of this
                    // name, so the KEYWORD-style call cannot be the
                    // zero-parameter method — the external field is
                    // unmodeled — the call is dropped (external-field
                    // divergence).
                    // The resolved method must be a PROPERTY-LIKE descriptor
                    // (`@CachedProperty`, `@property`, `@functools.cached_
                    // property` — botocore's `_s3_addressing_handler`, a
                    // zero-parameter property whose VALUE is a callable): a
                    // plain method with a keyword call (`c.bump(amount=2)`)
                    // is a normal keyword-argument call, never dropped.
                    let is_property_descriptor = sig.decorator_list.iter().any(|d| {
                        match d {
                            ExprType::Name(n) => matches!(
                                n.id.as_str(),
                                "property" | "CachedProperty" | "cached_property"
                            ),
                            ExprType::Attribute(a) => a.attr == "cached_property",
                            _ => false,
                        }
                    });
                    if is_property_descriptor
                        && !self.keywords.is_empty()
                        && class
                            .infer_fields(&class_symbols, &options)
                            .map(|fields| {
                                !fields.iter().any(|(name, _)| name == &attr.attr)
                            })
                            .unwrap_or(false)
                    {
                        options.definition_warnings.borrow_mut().push(format!(
                            "call through the external field `{}.{}` with keyword \
                             arguments is dropped (the field is registered externally \
                             and is unmodeled)",
                            class.name, attr.attr
                        ));
                        return Ok(quote!(stdpython::PyValue::None_));
                    }
                    // The stub call can only be DROPPED when the missing
                    // arguments are NOT defaultable — a stub whose missing
                    // params all have defaults (or Option annotations) can
                    // be mapped to the full-arity call, which dispatches
                    // VIRTUALLY to the most-derived override
                    // (`self.read(len(b))` in BaseHTTPResponse.readinto
                    // where HTTPResponse overrides read — urllib3, round
                    // 79: dropping it boxed the bytes result as None).
                    if supplied < stub_params
                        && crate::ast::tree::call::is_notimpl_stub(&sig)
                        && !stub_missing_args_defaultable(&sig, supplied)
                    {
                        options.definition_warnings.borrow_mut().push(format!(
                            "call to the abstract stub `{}.{}` with {} argument(s) is \
                             dropped (the abstract method protocol is unmodeled; the \
                             call returns a boxed None)",
                            class.name, attr.attr, supplied
                        ));
                        return Ok(quote!(stdpython::PyValue::None_));
                    }
                    let MappedArguments { prelude, args } = map_call_arguments_inner(
                        &sig,
                        &self.args,
                        &self.keywords,
                        &ctx,
                        &options,
                        // ARGUMENT expressions render in the CALLER's scope
                        // (an argument can be a call into the caller's
                        // module — `merge_setting(request.headers,
                        // self.headers, dict_class=CaseInsensitiveDict)` in
                        // requests' prepare_request, where p is a models.py
                        // PreparedRequest): the inner callee resolves in
                        // sessions.py, not models.py.
                        &symbols,
                        None,
                        // Dropped DEFAULT constants resolve in the defining
                        // module's scope.
                        Some(&class_symbols),
                    )?;
                    // A @staticmethod or @classmethod called through an
                    // INSTANCE binds no receiver in Python (`w.helper(1)`,
                    // `w.make(3)`): lower as the ASSOCIATED call
                    // `Class::method(args)` — method-call syntax against
                    // the value cannot resolve an associated fn (E0599),
                    // and a classmethod's cls is the class itself, never
                    // the instance.
                    if is_static || is_classmethod {
                        let cname = crate::safe_ident(&class.name);
                        let mname = crate::safe_ident(&attr.attr);
                        return Ok(quote!({ #prelude #cname::#mname(#(#args),*)? }));
                    }
                    // A field receiver (`self.inner`, `self.items`) renders
                    // in place flavor when the callee mutates the receiver
                    // (`self.inner_mut().bump()`), because the load flavor
                    // clones and the mutation would silently vanish. Read
                    // only callees may read through the load form.
                    let mutates_receiver =
                        class.method_needs_mut_self(&attr.attr, &class_symbols, &options);
                    // A receiver that is, or is a chain rooted at, a
                    // narrowed root-typed name: the place from the mutable
                    // view (`s.center.bump()` — Devin review on #319).
                    let narrowed_class_chain =
                        crate::ast::tree::attribute::chain_root_is_narrowed_class(&attr.value, &options);
                    // A field chain rooted at a SHARED instance (`a.center
                    // .bump()`): the place through the mutable borrow, or
                    // the call lands on the read's clone of the field.
                    let shared_chain = crate::ast::tree::attribute::chain_root_is_shared_instance(
                        &attr.value, &ctx, &symbols, &options,
                    );
                    let receiver =
                        if mutates_receiver
                            && (crate::ast::tree::attribute::chain_root_is_self(&attr.value)
                                || narrowed_class_chain
                                || shared_chain)
                        {
                            // The WHOLE chain renders as a place:
                            // `self.outer.inner.bump()` goes through
                            // `self.outer_mut().inner_mut().bump()`, not the
                            // cloning load accessors.
                            crate::ast::tree::attribute::to_rust_place_expr(
                                &attr.value,
                                &ctx,
                                &options,
                                &symbols,
                                false,
                            )?
                        } else {
                            attr.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?
                        };
                    // Issue #137's Option-aware access, the known-class
                    // method path: a receiver that is an Option-typed
                    // value (`self.timeout.connect_timeout()` where the
                    // field is `Timeout | None`) unwraps before the call —
                    // CPython's AttributeError-on-None as a loud §12.2
                    // panic. Computed BEFORE attr.value is moved above.
                    let receiver = if let Some(_inner) =
                        crate::ast::tree::attribute::receiver_option_inner(
                            &attr.value,
                            &ctx,
                            &symbols,
                            &options,
                        )
                    {
                        let mname = attr.attr.clone();
                        // A SHARED class mutates through the borrow: the
                        // Option unwrap clones the reference either way.
                        let shared = crate::ast::tree::shared::is_shared(&class.name);
                        if mutates_receiver && !shared {
                            quote!((#receiver).as_mut().unwrap_or_else(|| {
                                panic!(
                                    "AttributeError: 'NoneType' object has no attribute '{}'",
                                    #mname
                                )
                            }))
                        } else if crate::ast::tree::attribute::narrowed_name_read(&attr.value, &options) {
                            receiver
                        } else {
                            quote!((#receiver).clone().unwrap_or_else(|| {
                                panic!(
                                    "AttributeError: 'NoneType' object has no attribute '{}'",
                                    #mname
                                )
                            }))
                        }
                    } else {
                        receiver
                    };
                    // A SHARED class's value is a `PyRef` (shared.rs): the
                    // call borrows the one object — mutably when the
                    // method mutates. `self` inside the class is the
                    // struct itself; a shared ROOT's sum type borrows in
                    // its delegators.
                    let shared_borrow = crate::ast::tree::shared::is_shared(&class.name)
                        && !crate::ast::tree::hierarchy::is_polymorphic_root(&class.name)
                        && !crate::ast::tree::visit::is_self(attr.value.as_ref());
                    let receiver = if shared_borrow {
                        if mutates_receiver {
                            quote!((#receiver).borrow_mut())
                        } else {
                            quote!((#receiver).borrow())
                        }
                    } else {
                        receiver
                    };
                    let method_name = crate::safe_ident(&attr.attr);
                    if shared_borrow {
                        // The ARGUMENTS evaluate before the borrow, as
                        // Python evaluates them before the method body
                        // runs: bound to locals in source order, so an
                        // argument reading or mutating the same object
                        // (`a.set(a.value)`, `q.note(q.drain())`) completes
                        // first (Devin review on #321). Then the borrow
                        // ends WITH the call: bound to a local, the
                        // `Ref`/`RefMut` temporary drops at the `let` (a
                        // tail expression's temporary would live to the
                        // enclosing statement's end — `print(q.drain(),
                        // len(queues[0].items))` reads the same object
                        // next, as Python's left-to-right evaluation may).
                        let arg_names: Vec<proc_macro2::Ident> = (0..args.len())
                            .map(|i| format_ident!("__rython_sarg{}", i))
                            .collect();
                        return Ok(quote!({
                            #prelude
                            #(let #arg_names = #args;)*
                            let __rython_call = (#receiver).#method_name(#(#arg_names),*)?;
                            __rython_call
                        }));
                    }
                    return Ok(quote!({ #prelude (#receiver).#method_name(#(#args),*)? }));
                }
            }
            // A method call on a PyValue-typed SELF-FIELD
            // (`self._associated_futures.add(future)` — s3transfer's
            // TransferCoordinator, where a `set()` field lowers as a boxed
            // PyValue): the boxed value has no statically-known methods, so
            // the call is a no-op with a warning — the bookkeeping the set
            // performs is unmodeled (documented divergence).
            if crate::ast::tree::call::receiver_is_pyvalue_self_field(
                &attr.value, &ctx, &symbols, &options,
            ) {
                options.definition_warnings.borrow_mut().push(format!(
                    "call to `self.{}(...)` on a boxed PyValue field is dropped (the \
                     boxed value's methods are unmodeled)",
                    attr.attr
                ));
                return Ok(quote!(()));
            }
            // A mutating method on a subscripted receiver must go through
            // the PLACE lowering: `xs[0].append(v)` has to mutate the real
            // element, where the Load lowering (py_index) yields a clone
            // and the write silently vanishes. The same holds for a
            // `self.<field>` receiver in a generic trait default, where the
            // load form (`self.items()`) clones the field: the mutable
            // accessor (`self.items_mut()`) keeps the write on the real
            // field.
            let mutating_self_field = (ctx.in_generic_trait()
                && matches!(attr.value.as_ref(), ExprType::Attribute(_))
                && crate::ast::tree::attribute::chain_root_is_self(&attr.value))
                // A container field of a NARROWED root-typed name
                // (`s.tags.append(..)` — Devin review on #319): the place
                // from the mutable view, or the push lands on the read
                // view's clone.
                || (matches!(attr.value.as_ref(), ExprType::Attribute(_))
                    && crate::ast::tree::attribute::chain_root_is_narrowed_class(&attr.value, &options))
                // A container field of a SHARED instance (`a.items.append(x)`
                // where `a = accounts[k]` — Devin review on #321): the place
                // through the mutable borrow, never the read's clone.
                || crate::ast::tree::attribute::chain_root_is_shared_instance(
                    &attr.value, &ctx, &symbols, &options,
                );
            // Issue #137's Option-aware access, the CALL side: a method
            // call through an Option-typed receiver (`conn.close()` where
            // conn is `BaseHTTPConnection | None` — urllib3's
            // _get_conn) unwraps it first. A &mut-taking method uses the
            // mutable unwrap; a &self method clones. CPython raises
            // AttributeError on a None receiver — a loud §12.2 panic here.
            // Computed BEFORE attr.value is moved below.
            let option_receiver =
                crate::ast::tree::attribute::receiver_option_inner(&attr.value, &ctx, &symbols, &options);
            let receiver = if (matches!(attr.value.as_ref(), ExprType::Subscript(_))
                || mutating_self_field)
                && crate::ast::tree::scope::mutates_receiver(&attr.attr)
            {
                if let ExprType::Attribute(_) = attr.value.as_ref() {
                    // The whole receiver chain renders as a place:
                    // `self.inner.items.append(v)` mutates through
                    // `self.inner_mut().items`, not a clone of the field.
                    crate::ast::tree::attribute::to_rust_place_expr(
                        &attr.value,
                        &ctx,
                        &options,
                        &symbols,
                        false,
                    )?
                } else {
                    crate::ast::tree::subscript::subscript_receiver_place(
                        attr.value.as_ref(),
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?
                }
            } else {
                attr.value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?
            };
            let receiver = if let Some(_inner) = option_receiver {
                if crate::ast::tree::scope::mutates_receiver(&attr.attr) {
                    quote!((#receiver).as_mut().unwrap())
                } else if crate::ast::tree::attribute::narrowed_name_read(&attr.value, &options) {
                    receiver
                } else {
                    quote!((#receiver).clone().unwrap())
                }
            } else {
                receiver
            };

            // A str method on a receiver that POSITIVELY infers the boxed
            // PyValue (`context["scheme"].lower()` where context is
            // `dict[str, Any]` — urllib3's poolmanager: the subscript read
            // yields PyValue): dispatch on the runtime member — Str ->
            // the operation; anything else -> CPython's AttributeError
            // panic (§12.2). The blanket PyStrOps needs AsRef<str>, which
            // PyValue does not satisfy (E0599 "trait bounds not
            // satisfied"). Only a POSITIVE PyValue inference qualifies —
            // PyObject (unknown) keeps the plain method, loud in rustc if
            // the member is boxed.
            if matches!(attr.attr.as_str(), "lower" | "upper" | "strip")
                && matches!(
                    crate::infer_type(Some(&ctx), &attr.value, &options, &symbols),
                    crate::TypeInfo::PyValue | crate::TypeInfo::PyValueMember(_)
                )
            {
                let m = match attr.attr.as_str() {
                    "lower" => quote!(py_boxed_lower),
                    "upper" => quote!(py_boxed_upper),
                    _ => quote!(py_boxed_strip),
                };
                return Ok(quote!((#receiver).#m()));
            }

            // An UNBOUND builtin-str method applied to its receiver
            // (`str.title(header)` — urllib3's SKIPPABLE_HEADERS
            // titlecasing): Python's `str.m(s)` is `s.m()`. The
            // class-as-value model has no `str.title` attribute — the
            // builtin `str` lowers to the runtime str() fn item, and the
            // method call on it fails its PyStrOps bound (E0599) or the
            // attribute read fails (E0609) — so the call lowers to the
            // bound method on the argument. Only the zero-arg-beyond-
            // receiver str methods qualify (`str.join(sep, xs)` is the
            // two-argument bound form).
            if matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "str")
                && self.args.len() == 1
                && crate::ast::tree::type_ctx::StrMethod::from_name(&attr.attr)
                    .is_some_and(|m| m.takes_only_receiver())
            {
                let m = crate::safe_ident(&attr.attr);
                let arg = self.args[0].clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                return Ok(quote!((#arg).#m()));
            }

            // String-keyed dicts (from a literal or a `dict[str, V]`
            // annotation) take String keys in py_setdefault/py_pop and
            // &String in py_get/py_get_default/py_contains; literal `"a"`
            // keys are owned at the call site so the generic impls apply.
            // An OPTION-wrapped receiver (`request_context.pop("scheme")`
            // where request_context is `dict[str, Any] | None` — urllib3's
            // poolmanager: the call path unwraps the Option first, round
            // 63) is the same dict once unwrapped — the key owning sees
            // through the Option (round 66).
            let string_keyed_dict = matches!(
                attr.value.as_ref(),
                ExprType::Name(n)
                    if (matches!(
                        options.name_types.get(&n.id),
                        Some(crate::TypeInfo::Dict(k, _))
                            if matches!(**k, crate::TypeInfo::String)
                    ) || matches!(
                        options.name_types.get(&n.id),
                        Some(crate::TypeInfo::Option(inner))
                            if matches!(
                                &**inner,
                                crate::TypeInfo::Dict(k, _)
                                    if matches!(**k, crate::TypeInfo::String)
                            )
                    ))
            );
            // The receiver's (k, v) pair for dict methods whose ARGUMENT
            // must match the element types (`dict.update(other)` — the
            // stdpython PyDictOps method takes the other dict by value):
            // the pair comes from the dict (or Option-wrapped dict) type
            // the receiver's name holds (round 88).
            let dict_receiver_kv: Option<(crate::TypeInfo, crate::TypeInfo)> =
                match crate::infer_type(Some(&ctx), &attr.value, &options, &symbols) {
                    crate::TypeInfo::Dict(k, v) => Some(((*k).clone(), (*v).clone())),
                    crate::TypeInfo::Option(inner) => match &*inner {
                        crate::TypeInfo::Dict(k, v) => Some(((**k).clone(), (**v).clone())),
                        _ => None,
                    },
                    _ => None,
                };

            // list.sort(): in-place, stable, with Python's keyword-only
            // key=/reverse=. Vec's inherent sort demands a total order
            // (rejecting floats), so every shape routes through the
            // PySort variants, which share sorted()'s NaN-loud comparator
            // and run key exactly once per element.
            if attr.attr == "sort" {
                if !self.args.is_empty() {
                    return Err("sort() takes no positional arguments".to_string().into());
                }
                let mut key = None;
                let mut reverse = None;
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("key") if key.is_none() => key = Some(kw.value.clone()),
                        Some("reverse") if reverse.is_none() => reverse = Some(kw.value.clone()),
                        other => {
                            return Err(format!(
                                "sort() got an unexpected or duplicate keyword \
                                 argument '{}'",
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let render =
                    |e: crate::ExprType| e.to_rust(ctx.clone(), options.clone(), symbols.clone());
                return Ok(match (key, reverse) {
                    (None, None) => quote!((#receiver).py_sort()),
                    (None, Some(r)) => {
                        let r = render(r)?;
                        quote!((#receiver).py_sort_reverse(#r))
                    }
                    (Some(k), None) => {
                        let k = render_key_fn(&k, &attr.value, &ctx, &options, &symbols)?;
                        quote!((#receiver).py_sort_key(#k))
                    }
                    (Some(k), Some(r)) => {
                        let k = render_key_fn(&k, &attr.value, &ctx, &options, &symbols)?;
                        let r = render(r)?;
                        quote!((#receiver).py_sort_key_reverse(#k, #r))
                    }
                });
            }

            // replace(...) with datetime-family keywords: one lowering
            // through the PyReplace trait, whose receiver impls
            // (datetime/date/time) each validate their own field set and
            // raise Python's exact TypeError for foreign fields. Only
            // keyword spellings route here — bare/positional replace
            // stays the plain method call (str.replace among others).
            if attr.attr == "replace" && !self.keywords.is_empty() {
                const FIELDS: [&str; 7] = [
                    "year",
                    "month",
                    "day",
                    "hour",
                    "minute",
                    "second",
                    "microsecond",
                ];
                // `replace(tzinfo=None)` (botocore's compat
                // get_current_datetime) strips the timezone — rython's
                // datetime is naive, so removing tz is a no-op; drop the
                // keyword rather than erroring.
                let keywords: Vec<crate::Keyword> = self
                    .keywords
                    .iter()
                    .filter(|kw| {
                        !(kw.arg.as_deref() == Some("tzinfo")
                            && crate::is_none_expr(&kw.value))
                    })
                    .cloned()
                    .collect();
                let keywords = if keywords.len() == self.keywords.len() {
                    std::borrow::Cow::Borrowed(&self.keywords)
                } else {
                    std::borrow::Cow::Owned(keywords)
                };
                if keywords
                    .iter()
                    .all(|kw| kw.arg.as_deref().is_some_and(|a| FIELDS.contains(&a)))
                {
                    let mut slots: [Option<crate::ExprType>; 7] = Default::default();
                    if self.args.len() > FIELDS.len() {
                        return Err("replace() takes at most 7 arguments".to_string().into());
                    }
                    for (i, arg) in self.args.iter().enumerate() {
                        slots[i] = Some(arg.clone());
                    }
                    for kw in keywords.iter() {
                        let name = kw.arg.as_deref().expect("checked above");
                        let idx = FIELDS.iter().position(|f| *f == name).expect("checked");
                        if slots[idx].is_some() {
                            return Err(format!(
                                "replace() got multiple values for argument '{}'",
                                name
                            )
                            .into());
                        }
                        slots[idx] = Some(kw.value.clone());
                    }
                    let mut inits = Vec::new();
                    for (idx, slot) in slots.into_iter().enumerate() {
                        if let Some(e) = slot {
                            let field = crate::safe_ident(FIELDS[idx]);
                            let v = e.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                            inits.push(quote!(#field: Some(#v)));
                        }
                    }
                    return Ok(quote!(
                        (#receiver).py_replace(ReplaceArgs {
                            #(#inits,)*
                            ..ReplaceArgs::default()
                        })?
                    ));
                }
                // A keyword outside the datetime field set: an EXTERNAL
                // object's replace (`signature.replace(parameters=...)` —
                // an inspect.Signature, botocore's docs method): the
                // external object's method is unmodeled — a dropped call
                // with a warning (external-object divergence).
                let bad = keywords
                    .iter()
                    .find(|kw| !kw.arg.as_deref().is_some_and(|a| FIELDS.contains(&a)))
                    .and_then(|kw| kw.arg.clone())
                    .unwrap_or_else(|| "**kwargs".to_string());
                options.definition_warnings.borrow_mut().push(format!(
                    "replace({}) on an external object is dropped (the external \
                     object's method is unmodeled)",
                    bad
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }

            // str.split / str.rsplit take sep and maxsplit by position or
            // keyword, with sep=None (or absent) meaning whitespace mode.
            // Normalized here so every spelling maps to the right runtime
            // variant; unknown or duplicate keywords are loud errors, as
            // Python raises TypeError for them.
            if matches!(attr.attr.as_str(), "split" | "rsplit")
                // A NON-str receiver's split with FOREIGN keywords
                // (`text.split(allow_blank=True)` — a rich Text, pip's
                // exceptions): not the str split — fall through to the
                // generic path (the keywords lower positionally).
                && !self
                    .keywords
                    .iter()
                    .any(|k| !matches!(k.arg.as_deref(), Some("sep" | "maxsplit")))
                // A MODULE-PATH receiver (`os.path.split(...)` — the
                // runtime module function, not a str method): skip the
                // rewrite so the module-path call renders.
                && !crate::ast::tree::attribute::is_module_path_chain(
                    &attr.value,
                    &symbols,
                    &options,
                )
            {
                if self.args.len() > 2 {
                    return Err(format!(
                        "{}() takes at most 2 arguments ({} given)",
                        attr.attr,
                        self.args.len()
                    )
                    .into());
                }
                let mut sep = self.args.first().cloned();
                let mut maxsplit = self.args.get(1).cloned();
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("sep") => {
                            if sep.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'sep'",
                                    attr.attr
                                )
                                .into());
                            }
                            sep = Some(kw.value.clone());
                        }
                        Some("maxsplit") => {
                            if maxsplit.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'maxsplit'",
                                    attr.attr
                                )
                                .into());
                            }
                            maxsplit = Some(kw.value.clone());
                        }
                        other => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                attr.attr,
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let is_rsplit = attr.attr == "rsplit";
                let sep = sep.filter(|s| !crate::is_none_expr(s));
                return Ok(match (sep, maxsplit) {
                    (None, None) => quote!((#receiver).py_split_whitespace()),
                    (None, Some(m)) => {
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_whitespace_maxsplit(#m))
                        } else {
                            quote!((#receiver).py_split_whitespace_maxsplit(#m))
                        }
                    }
                    (Some(s), None) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit(&(#s))?)
                        } else {
                            quote!((#receiver).py_split(&(#s))?)
                        }
                    }
                    (Some(s), Some(m)) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_maxsplit(&(#s), #m)?)
                        } else {
                            quote!((#receiver).py_split_maxsplit(&(#s), #m)?)
                        }
                    }
                });
            }

            // str.format on a LITERAL template translates to format! at
            // conversion time: auto-numbering, {0} positions, {name}
            // keywords, {{ escaping, and format specs all map — and any
            // spec Rust cannot reproduce exactly is a loud conversion
            // error, never approximated output.
            if attr.attr == "format" {
                let Some(template) =
                    str_format_template(attr.value.as_ref(), Some(&ctx), &symbols, &options)
                else {
                    // The dynamic-format divergence: a template the
                    // conversion cannot see (a parameter, a field stored
                    // from one) — the call is dropped as the boxed None,
                    // with the warning.
                    options.definition_warnings.borrow_mut().push(
                        "str.format on a non-literal template is dropped (the \
                         dynamic-format divergence)"
                            .to_string(),
                    );
                    return Ok(quote!(stdpython::PyValue::None_));
                };
                return lower_str_format(
                    &template,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }

            // The remaining builtin methods are positional-only in Python;
            // a keyword here would be silently dropped by the positional
            // pattern match below, so fall through to the generic path,
            // which rejects keywords without a resolvable signature.
            // Exception: bytes/str.decode(enc, errors="strict") — the
            // errors keyword is accepted (strict is Python's default; the
            // stdpython codec layer always decodes strictly, so a non-
            // strict errors value is the documented divergence).
            let decode_errors_kw = attr.attr == "decode"
                && self.keywords.len() == 1
                && self.keywords[0].arg.as_deref() == Some("errors");
            // md5(x, usedforsecurity=False): the FIPS policy keyword is
            // ignored.
            let hash_kw = matches!(attr.attr.as_str(), "md5" | "sha1" | "sha256" | "sha512")
                && self.keywords.len() == 1
                && self.keywords[0].arg.as_deref() == Some("usedforsecurity");
            if !self.keywords.is_empty() && !decode_errors_kw && !hash_kw {
                // fall through
            } else {
                let mut rendered_args = Vec::new();
                for arg in &self.args {
                    // A SHARED class's reference stored into a container
                    // (`self.audit.append(acct)`) and read again later is
                    // another reference to the one object: clone it.
                    let shared_name_reused = matches!(arg, ExprType::Name(n)
                        if options.use_counts.get(&n.id).copied().unwrap_or(0) > 1
                            && matches!(options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Class(c)) if crate::ast::tree::shared::is_shared(c)));
                    let rendered = arg.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    rendered_args.push(if shared_name_reused {
                        quote!(Clone::clone(&(#rendered)))
                    } else {
                        rendered
                    });
                }
                match (attr.attr.as_str(), rendered_args.as_slice()) {
                    // list.append(x) pushes one element; Vec::append (inherent)
                    // concatenates another Vec — silently different.
                    ("append", [value]) => {
                        // A str literal appended to a Vec<String> local
                        // (`lines.append("\r\n")` — urllib3's
                        // render_headers) is a &'static str; the Vec holds
                        // owned Strings, so the literal owns at the push
                        // site (mirroring the String-name store rule).
                        if let crate::TypeInfo::Vec(inner) =
                            crate::infer_type(Some(&ctx), &attr.value, &options, &symbols)
                            && matches!(*inner, crate::TypeInfo::String)
                            && matches!(
                                self.args.first(),
                                Some(ExprType::Constant(c))
                                    if matches!(&c.0, Some(litrs::Literal::String(_)))
                            )
                        {
                            return Ok(quote!((#receiver).push((#value).to_string())));
                        }
                        return Ok(quote!((#receiver).push(#value)));
                    }
                    // list.extend(x) with a PyValue/heterogeneous argument
                    // (botocore's `retryable_exceptions.extend(
                    // retry_exception)` where retry_exception is the
                    // boxed return of _extract_retryable_exception — round
                    // 33): the runtime's py_extend reads the boxed
                    // tuple's members into the Vec. A Vec<String> receiver
                    // takes the strings path, a boxed-element receiver
                    // (PyValue, or PyObject from the empty-literal
                    // divergence — `[]` lowers to Vec<PyValue>) the
                    // members path. A typed Vec arg keeps the inherent
                    // extend, element-converting for a boxed-element
                    // receiver (`exceptions.extend([ChecksumError])` —
                    // Vec<PyValue> + Vec<String>).
                    ("extend", [value]) => {
                        let recv_ty = crate::infer_type(Some(&ctx), &attr.value, &options, &symbols);
                        if let crate::TypeInfo::Vec(inner) = &recv_ty {
                            let arg_ty = crate::infer_type(Some(&ctx), &self.args[0], &options, &symbols);
                            let runtime = crate::safe_ident(&options.stdpython);
                            let boxed = matches!(
                                **inner,
                                crate::TypeInfo::PyValue | crate::TypeInfo::PyObject
                            );
                            match &arg_ty {
                                crate::TypeInfo::Vec(_) => {
                                    if boxed {
                                        return Ok(quote!(
                                            (#receiver).extend(
                                                (#value).into_iter().map(stdpython::PyValue::from)
                                            )
                                        ));
                                    }
                                    return Ok(quote!((#receiver).extend(#value)));
                                }
                                _ => {
                                    if matches!(**inner, crate::TypeInfo::String) {
                                        return Ok(quote!(
                                            #runtime::py_extend_strings(&mut (#receiver), &(#value))?
                                        ));
                                    }
                                    return Ok(quote!(
                                        #runtime::py_extend_values(&mut (#receiver), &(#value))?
                                    ));
                                }
                            }
                        }
                    }
                    // list.count(x): the PyListOps method takes a reference.
                    ("count", [value]) => {
                        return Ok(quote!((#receiver).count(&(#value))));
                    }
                    // File-object and csv.Writer methods return Result (I/O
                    // can fail; Python raises): thread `?`. The threading
                    // (acquire/release/wait) and socket (accept/getsockname/
                    // getpeername) object methods return Result the same way
                    // — lock release and socket state errors are catchable
                    // Python exceptions.
                    ("read", [])
                    | ("readline", [])
                    | ("readlines", [])
                    | ("close", [])
                    | ("getvalue", [])
                    | ("acquire", [])
                    | ("release", [])
                    | ("wait", [])
                    | ("accept", [])
                    | ("getsockname", [])
                    | ("getpeername", []) => {
                        let m = crate::safe_ident(&attr.attr);
                        return Ok(quote!((#receiver).#m()?));
                    }
                    // Socket methods with arguments: all Result-returning
                    // (network I/O raises OSError kinds); byte payloads pass
                    // by reference (the runtime takes AsRef<[u8]>), so a
                    // buffer survives its send.
                    ("connect", [a]) | ("bind", [a]) => {
                        let m = crate::safe_ident(&attr.attr);
                        return Ok(quote!((#receiver).#m(#a)?));
                    }
                    ("listen", [n]) | ("recv", [n]) | ("recvfrom", [n]) => {
                        let m = crate::safe_ident(&attr.attr);
                        return Ok(quote!((#receiver).#m(#n)?));
                    }
                    // Python accepts int or float seconds; the runtime takes
                    // f64 (the coercion is exact for any plausible timeout).
                    ("settimeout", [t]) => {
                        return Ok(quote!((#receiver).settimeout((#t) as f64)?));
                    }
                    ("send", [d]) | ("sendall", [d]) => {
                        let m = crate::safe_ident(&attr.attr);
                        return Ok(quote!((#receiver).#m(&(#d))?));
                    }
                    ("sendto", [d, a]) => {
                        return Ok(quote!((#receiver).sendto(&(#d), #a)?));
                    }
                    ("write", [d]) => {
                        return Ok(quote!((#receiver).write(&(#d))?));
                    }
                    ("writelines", [l]) => {
                        return Ok(quote!((#receiver).writelines(&(#l))?));
                    }
                    ("writerow", [r]) => {
                        // writerow([]) (an empty record) still needs an
                        // element type for the slice.
                        if r.to_string() == "vec ! []" {
                            return Ok(quote!((#receiver).writerow(&[] as &[&str])?));
                        }
                        return Ok(quote!((#receiver).writerow(&(#r))?));
                    }
                    ("writerows", [r]) => {
                        return Ok(quote!((#receiver).writerows(&(#r))?));
                    }
                    // re Match: m.group() is m.group(0); Rust can't overload.
                    ("group", []) => {
                        return Ok(quote!((#receiver).group(0)));
                    }
                    // m.group("name") for (?P<name>...) groups: Rust can't
                    // overload on the argument type, so the string spelling
                    // routes to group_name. Numeric group(i) falls through to
                    // the plain method call.
                    ("group", [g]) if g.to_string().starts_with('"') => {
                        return Ok(quote!((#receiver).group_name(#g)));
                    }
                    // m.span(i) — the group-index form (Python's optional
                    // argument): Rust can't overload on arity, so the
                    // indexed spelling routes to span_group; m.span() falls
                    // through to the plain (0-arg) span().
                    ("span", [g]) => {
                        return Ok(quote!((#receiver).span_group(#g)));
                    }
                    // str.encode() / encode("utf-8"): UTF-8 bytes, which is
                    // exactly what Rust strings hold. ascii and punycode
                    // (RFC 3492) go through the stdpython codec layer.
                    // Only fires when the receiver is a string-like VALUE
                    // (a str literal, a str-typed name, or StrOrBytes) —
                    // `idna.encode(name)` (a module function) must fall
                    // through to the generic path.
                    ("encode", [])
                        if crate::ast::tree::call::receiver_is_str_like(
                            &attr.value, &options, &symbols,
                        ) =>
                    {
                        return Ok(quote!((#receiver).as_bytes().to_vec()));
                    }
                    ("encode", [enc])
                        if crate::ast::tree::call::receiver_is_str_like(
                            &attr.value, &options, &symbols,
                        ) =>
                    {
                        // A `self.CONST` encoding argument
                        // (`value.encode(self.DEFAULT_ENCODING)` — botocore's
                        // serialize, where DEFAULT_ENCODING = 'utf-8' is a
                        // class constant): resolve the constant's literal.
                        let codec = if let ExprType::Attribute(a) = self.args.first().unwrap()
                            && crate::ast::tree::visit::is_self(a.value.as_ref())
                            && let Some(enclosing) = ctx.enclosing_class_name()
                            && let Some(SymbolTableNode::ClassDef(class)) = symbols.get(enclosing)
                            && let Some(lit) = class.body.iter().find_map(|s| match &s.statement {
                                crate::StatementType::Assign(assign)
                                    if assign.targets.len() == 1
                                        && matches!(&assign.targets[0], ExprType::Name(n) if n.id == a.attr) =>
                                {
                                    match &assign.value {
                                        ExprType::Constant(c) => match &c.0 {
                                            Some(litrs::Literal::String(s)) => {
                                                Some(s.value().to_string())
                                            }
                                            _ => None,
                                        },
                                        _ => None,
                                    }
                                }
                                _ => None,
                            })
                        {
                            lit
                        } else {
                            enc.to_string().trim_matches('"').to_string()
                        };
                        match codec.as_str() {
                            // `utf8` (no hyphen) is Python's accepted
                            // spelling (botocore's signers).
                            "utf-8" | "utf8" => {
                                return Ok(quote!((#receiver).as_bytes().to_vec()));
                            }
                            "ascii" => {
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::encode_ascii(#receiver)?
                                ));
                            }
                            // `host.encode("idna")` — a VALIDATION call whose
                            // result is discarded (urllib3's
                            // create_connection): idna encoding is a
                            // no-op here (the exception check is the
                            // point).
                            "idna" => {
                                return Ok(quote!((#receiver).as_bytes().to_vec()));
                            }
                            "punycode" => {
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::encode_punycode(#receiver)
                                ));
                            }
                            // latin-1: each character is its code point.
                            // ISO-8859-15 (Latin-9) differs from latin-1 in
                            // only a few code points (urllib3's emscripten
                            // fetch); treated as latin-1 — a documented
                            // divergence.
                            "latin1" | "latin-1" | "iso-8859-1" | "iso-8859-15"
                            | "ISO-8859-15" | "ISO-8859-1" => {
                                let runtime = crate::safe_ident(&options.stdpython);
                                return Ok(quote!(
                                    #runtime::stdlib::codec::encode_latin1(#receiver)?
                                ));
                            }
                            other => {
                                return Err(format!(
                                    "str.encode({}): only utf-8, ascii, punycode, and \
                                     latin-1 are supported",
                                    other
                                )
                                .into());
                            }
                        }
                    }
                    // decode on a receiver WITHOUT a statically-known bytes
                    // type — an unannotated parameter (`T: PyDecode`, the
                    // isinstance-residual morphs of issue #161) or a boxed
                    // PyValue — dispatches through the PyDecode trait at
                    // runtime. Defaults are Python's (utf-8, strict).
                    ("decode", args @ ([] | [_] | [_, _]))
                        if crate::ast::tree::call::root_name(&attr.value)
                            .is_some_and(|root| {
                                options.param_method_params.contains(root)
                                    || matches!(
                                        options.name_types.get(root),
                                        Some(crate::TypeInfo::PyValue)
                                    )
                            }) =>
                    {
                        let enc = args
                            .first()
                            .map(|e| quote!(#e))
                            .unwrap_or_else(|| quote!("utf-8"));
                        let errors = args
                            .get(1)
                            .map(|e| quote!(#e))
                            .unwrap_or_else(|| quote!("strict"));
                        return Ok(quote!(
                            (#receiver).py_decode(#enc, #errors)?
                        ));
                    }
                    // bytes.decode(enc, errors) with BOTH positional
                    // arguments (`path.decode(filesystem_encoding,
                    // 'replace')` — botocore configloader): the PyDecode
                    // lowering ('replace' follows CPython for utf-8; other
                    // non-strict errors values decode strictly — the
                    // documented decode divergence). A bytes-typed
                    // receiver qualifies alongside the str-like ones
                    // (Vec<u8> implements the trait directly).
                    ("decode", [enc, errors])
                        if crate::ast::tree::call::receiver_is_str_like(
                            &attr.value, &options, &symbols,
                        ) || matches!(
                            &*attr.value,
                            ExprType::Name(n) if matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Bytes)
                            )
                        ) =>
                    {
                        return Ok(quote!(
                            (#receiver).py_decode(#enc, #errors)?
                        ));
                    }
                    // bytes.decode("utf-8"|"ascii"|"punycode"): the codec
                    // layer in stdpython (Rust strings ARE utf-8, so the
                    // bytes→String conversion is the codec's job).
                    ("decode", [enc])
                        if crate::ast::tree::call::receiver_is_str_like(
                            &attr.value, &options, &symbols,
                        ) =>
                    {
                        // A literal codec name dispatches at conversion
                        // time; a runtime name (a parameter) goes through
                        // the codec registry dispatch.
                        let codec = enc.to_string().trim_matches('"').to_string();
                        let runtime = crate::safe_ident(&options.stdpython);
                        if matches!(
                            codec.as_str(),
                            "utf-8" | "ascii" | "punycode"
                        ) {
                            let f = crate::safe_ident(match codec.as_str() {
                                "utf-8" | "utf8" => "decode_utf8",
                                "ascii" => "decode_ascii",
                                _ => "decode_punycode",
                            });
                            return Ok(quote!(
                                #runtime::stdlib::codec::#f(&(#receiver))?
                            ));
                        }
                        // Runtime codec name: dispatch in the runtime.
                        return Ok(quote!(
                            #runtime::stdlib::codec::decode_by_name(&(#receiver), #enc)?
                        ));
                    }
                    ("decode", []) => {
                        let runtime = crate::safe_ident(&options.stdpython);
                        return Ok(quote!(#runtime::stdlib::codec::decode_utf8(&(#receiver))?));
                    }
                    // bytes sep.join(parts) — `b"".join(data_parts)`
                    // (urllib3's chunked response assembly): the bytes
                    // twin of str join, through the runtime's bytes
                    // surface. Only when the receiver is genuinely a
                    // bytes value; a str receiver keeps the PyStrOps path.
                    ("join", [parts])
                        if crate::ast::tree::call::receiver_is_bytes_like(
                            &attr.value,
                            &options,
                            &symbols,
                        ) =>
                    {
                        let runtime = crate::safe_ident(&options.stdpython);
                        return Ok(quote!(#runtime::bytes_join(&(#receiver), &(#parts))));
                    }
                    // list.pop() returns the last element or raises IndexError
                    // (Vec::pop returns an Option). A GENERIC receiver (an
                    // unannotated parameter with a PyPop bound, issue #109
                    // M2) has no inherent pop: route through the trait.
                    ("pop", []) => {
                        let generic = crate::ast::tree::call::root_name(&attr.value)
                            .is_some_and(|root| options.param_method_params.contains(root));
                        if generic {
                            return Ok(quote!((#receiver).py_pop(-1)?));
                        }
                        return Ok(quote! {
                            (#receiver).pop().ok_or_else(|| {
                                PyException::new("IndexError", "pop from empty list")
                            })?
                        });
                    }
                    // pop with an argument dispatches by receiver through the
                    // PyPop trait: list.pop(i) by index (IndexError), dict.pop(k)
                    // by key (KeyError).
                    ("pop", [arg]) => {
                        if string_keyed_dict {
                            let key = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote!((#receiver).py_pop(#key)?));
                        }
                        return Ok(quote!((#receiver).py_pop(#arg)?));
                    }
                    ("pop", [key, default]) => {
                        if string_keyed_dict {
                            let key = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote!((#receiver).py_pop_default(#key, #default)));
                        }
                        return Ok(quote!((#receiver).py_pop_default(#key, #default)));
                    }
                    // dict.get never raises: value-or-None (an Option), or the
                    // provided default. IndexMap's inherent get returns a
                    // borrowed Option, so both forms map to py_ versions.
                    // A USER-CLASS receiver defines no `get` — the
                    // collections.abc mixin provides it via `__getitem__`
                    // (`response.headers.get("Retry-After")` where
                    // HTTPHeaderDict subclasses MutableMapping — urllib3):
                    // synthesize the mixin exactly, catching KeyError only
                    // (any other exception propagates, Python's behavior).
                    ("get", [key]) => {
                        if let Some((class, class_symbols)) =
                            crate::receiver_class(&attr.value, &ctx, &symbols, &options)
                            && let Some(method) =
                                class.method_on_mro("__getitem__", &class_symbols)
                            && crate::ast::tree::class_def::class_has_mapping_abc_base(
                                &class,
                                &class_symbols,
                            )
                        {
                            let call = crate::dunder_method_call(
                                &method,
                                &receiver,
                                std::slice::from_ref(&self.args[0]),
                                false,
                                &ctx,
                                &options,
                                &symbols,
                            )?;
                            return Ok(quote! {
                                match #call {
                                    Ok(__rython_v) => Some(__rython_v),
                                    Err(__rython_e)
                                        if __rython_e.matches("KeyError") => None,
                                    Err(__rython_e) => return Err(__rython_e),
                                }
                            });
                        }
                        if string_keyed_dict {
                            let key = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote!((#receiver).py_get(&(#key))));
                        }
                        return Ok(quote!((#receiver).py_get(&(#key))));
                    }
                    ("get", [key, default]) => {
                        if let Some((class, class_symbols)) =
                            crate::receiver_class(&attr.value, &ctx, &symbols, &options)
                            && class.method_on_mro("__getitem__", &class_symbols).is_some()
                            && crate::ast::tree::class_def::class_has_mapping_abc_base(
                                &class,
                                &class_symbols,
                            )
                        {
                            // A None (or Option-typed) DEFAULT (`headers.get(
                            // name, default=None)` — urllib3's getheader)
                            // makes the result an OPTION: the Ok arm must
                            // wrap (`Ok(v) => Some(v)`), or the arms mix
                            // String and Option (round 61b).
                            let default_is_none = crate::is_none_expr(&self.args[1])
                                || crate::expr_yields_option_ctx(
                                    &self.args[1],
                                    &ctx,
                                    &options,
                                    &symbols,
                                );
                            // Round 83: an OPTION-typed default renders as the
                            // Option ITSELF — the Some-wrapped Ok arm matches
                            // it, and the empty case IS the fallback (Python's
                            // `headers.get(k, default)` returns the Option
                            // default when the key is absent). The Option→
                            // concrete coercion (the round-83 unwrap) must NOT
                            // fire here — it would unwrap the fallback to the
                            // member and break the arm match (`Option<String> |
                            // String`, getheader ×3 in urllib3).
                            let default = if default_is_none {
                                self.args[1]
                                    .clone()
                                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?
                            } else {
                                crate::render_typed(
                                    &self.args[1],
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                    Some(crate::TypeInfo::String),
                                )?
                            };
                            let ok_arm = if default_is_none {
                                quote!(Ok(__rython_v) => Some(__rython_v))
                            } else {
                                quote!(Ok(__rython_v) => __rython_v)
                            };
                            // Python evaluates RECEIVER, KEY, then DEFAULT
                            // — each exactly once, and a failure in an
                            // earlier expression prevents the later ones
                            // (Devin review on #267: binding the default
                            // first reordered the side effects, and the
                            // bare call repeated nothing but evaluated
                            // receiver+key after the default). Bind all
                            // three up front, then invoke __getitem__.
                            let key_arg = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote! {
                                {
                                    let __rython_recv = (#receiver).clone();
                                    let __rython_key = #key_arg;
                                    let __rython_default = #default;
                                    match __rython_recv.__getitem__(__rython_key) {
                                        #ok_arm,
                                        Err(__rython_e)
                                            if __rython_e.matches("KeyError") =>
                                                __rython_default,
                                        Err(__rython_e) => return Err(__rython_e),
                                    }
                                }
                            });
                        }
                        if string_keyed_dict {
                            let key = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote!((#receiver).py_get_default(&(#key), #default)));
                        }
                        return Ok(quote!((#receiver).py_get_default(&(#key), #default)));
                    }
                    // Views materialize as Vecs in insertion order.
                    ("keys", []) => {
                        return Ok(quote!((#receiver).py_keys()));
                    }
                    // dict.update(other) — the stdpython PyDictOps method
                    // takes the other dict BY VALUE (`other: PyDict<K, V>`),
                    // and an OPTION-typed argument (`headers.update(
                    // self.proxy_headers)` — a `Mapping[str, str] | None`
                    // field, urllib3's urlopen/_make_request) must coerce
                    // via the round-83 Option→concrete match: Python's
                    // update(None) is a TypeError, and the loud panic is
                    // the honest model (round 88).
                    ("update", [other]) => {
                        if let Some((k, v)) = dict_receiver_kv.clone() {
                            let arg = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::Dict(
                                    Box::new(k),
                                    Box::new(v),
                                )),
                            )?;
                            return Ok(quote!((#receiver).update(#arg)));
                        }
                        return Ok(quote!((#receiver).update(#other)));
                    }
                    ("values", []) => {
                        return Ok(quote!((#receiver).py_values()));
                    }
                    ("items", []) => {
                        return Ok(quote!((#receiver).py_items()));
                    }
                    // dict/list/set .clear() mutates in place, so it must
                    // go through the PLACE-flavored receiver computed
                    // above. Without this arm the call fell to the generic
                    // fallback, which re-renders the receiver in LOAD
                    // flavor — in a trait default that is the cloning
                    // accessor (`self.regs()`), and the clear silently
                    // vanished.
                    ("clear", [])
                        if !crate::ast::tree::attribute::is_module_path_chain(
                            &attr.value,
                            &symbols,
                            &options,
                        ) =>
                    {
                        return Ok(quote!((#receiver).clear()));
                    }
                    ("setdefault", [key, default]) => {
                        if string_keyed_dict {
                            let key = crate::render_typed(
                                &self.args[0],
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::String),
                            )?;
                            return Ok(quote!((#receiver).py_setdefault(#key, #default)));
                        }
                        return Ok(quote!((#receiver).py_setdefault(#key, #default)));
                    }
                    // list.remove(x) removes by VALUE and raises ValueError;
                    // Vec::remove removes by index — silently different. A
                    // MODULE-PATH receiver (`os.remove(path)` — requests'
                    // utils) is the runtime function, not a list method.
                    ("remove", [value])
                        if !crate::ast::tree::attribute::is_module_path_chain(
                            &attr.value,
                            &symbols,
                            &options,
                        ) => {
                        // The receiver and argument are each evaluated ONCE,
                        // receiver first (CPython evaluates the primary +
                        // attribute, then the argument). The previous shape
                        // spliced the receiver twice and the argument inside
                        // the position closure, so a side-effecting receiver
                        // (`grid[which()].remove(2)`) ran twice and the
                        // argument once per element scanned (issue #80).
                        return Ok(quote! {
                            {
                                let __rython_recv = &mut (#receiver);
                                let __rython_val = #value;
                                let __rython_pos = __rython_recv
                                    .iter()
                                    .position(|__rython_e| __rython_e == &__rython_val)
                                    .ok_or_else(|| {
                                        PyException::new(
                                            "ValueError",
                                            "list.remove(x): x not in list",
                                        )
                                    })?;
                                __rython_recv.remove(__rython_pos);
                            }
                        });
                    }
                    // list.insert follows Python index rules (negative counts
                    // from the end, out-of-range clamps); Vec::insert takes a
                    // usize and panics past len. The `?` propagates the
                    // IndexError a bounded deque raises at its maxlen
                    // (issue #82).
                    ("insert", [idx, value]) => {
                        // A str literal inserted into a Vec<String> local
                        // (`output.insert(0, "")` — urllib3's
                        // _remove_path_dot_segments) is a &'static str; the
                        // Vec holds owned Strings, so the literal owns at the
                        // insert site.
                        if let crate::TypeInfo::Vec(inner) =
                            crate::infer_type(Some(&ctx), &attr.value, &options, &symbols)
                            && matches!(*inner, crate::TypeInfo::String)
                            && matches!(
                                self.args.get(1),
                                Some(ExprType::Constant(c))
                                    if matches!(&c.0, Some(litrs::Literal::String(_)))
                            )
                        {
                            return Ok(quote!((#receiver).py_insert(#idx, (#value).to_string())?));
                        }
                        return Ok(quote!((#receiver).py_insert(#idx, #value)?));
                    }
                    // partition/rpartition raise ValueError on an empty
                    // separator, so the calls take `?`.
                    ("partition", [sep]) => {
                        return Ok(quote!((#receiver).partition(&(#sep))?));
                    }
                    ("rpartition", [sep]) => {
                        return Ok(quote!((#receiver).rpartition(&(#sep))?));
                    }
                    // strip family with a chars argument (the no-arg forms
                    // resolve through PyStrOps directly).
                    ("strip", [chars]) => {
                        return Ok(quote!((#receiver).py_strip_chars(&(#chars))));
                    }
                    ("lstrip", [chars]) => {
                        return Ok(quote!((#receiver).py_lstrip_chars(&(#chars))));
                    }
                    ("rstrip", [chars]) => {
                        return Ok(quote!((#receiver).py_rstrip_chars(&(#chars))));
                    }
                    // ljust/rjust: the optional fillchar selects the py_ form
                    // (space by default).
                    ("ljust", [width]) => {
                        return Ok(quote!((#receiver).py_ljust(#width, " ")?));
                    }
                    ("ljust", [width, fill]) => {
                        return Ok(quote!((#receiver).py_ljust(#width, &(#fill))?));
                    }
                    ("rjust", [width]) => {
                        return Ok(quote!((#receiver).py_rjust(#width, " ")?));
                    }
                    ("rjust", [width, fill]) => {
                        return Ok(quote!((#receiver).py_rjust(#width, &(#fill))?));
                    }
                    // str.find returns -1 when absent; str::find an Option.
                    ("find", [needle]) => {
                        return Ok(quote!((#receiver).py_find(&(#needle))));
                    }
                    // A COMPILED-REGEX static receiver (`_TARGET_RE.match(
                    // target)` — the module static holds `re.compile(...)`,
                    // typed as the runtime Regex): the anchored matching
                    // dispatches through the runtime's PyRegexOps (py_match
                    // anchors at the start, py_search anywhere, py_fullmatch
                    // requires the whole text). Without the arm the call
                    // emitted `.r#match()` on the static's value — E0599
                    // (no such method on a boxed PyValue). The MODULE name
                    // resolves through the StdModule registry.
                    ("match" | "search" | "fullmatch", [text]) => {
                        if crate::ast::tree::call::root_name(&attr.value).is_some_and(|root| {
                            matches!(
                                symbols.get(&root),
                                Some(crate::SymbolTableNode::Assign {
                                    value: crate::ExprType::Call(c),
                                    ..
                                }) if matches!(c.func.as_ref(), crate::ExprType::Attribute(a)
                                    if a.attr == "compile"
                                        && matches!(a.value.as_ref(), crate::ExprType::Name(n)
                                            if crate::StdModule::from_name(&n.id)
                                                == Some(crate::StdModule::Re)))
                            )
                        }) {
                            let m = match attr.attr.as_str() {
                                "search" => quote!(py_search),
                                "fullmatch" => quote!(py_fullmatch),
                                _ => quote!(py_match),
                            };
                            // An Option-typed TEXT argument (`host: str |
                            // None` whose None-ness was already tested —
                            // urllib3's _normalize_host) unwraps with the
                            // same loud NoneType panic the Option-receiver
                            // path uses; CPython would raise TypeError on
                            // an actual None here, and the unwrap fires
                            // only when the flow contradicts the guard.
                            let text_arg = if crate::expr_yields_option_ctx(
                                &self.args[0],
                                &ctx,
                                &options,
                                &symbols,
                            ) {
                                // CPython raises TypeError for a None text
                                // argument: re.compile("a").match(None) ->
                                // "expected string or bytes-like object,
                                // got 'NoneType'".
                                quote!(&((#text).clone().unwrap_or_else(|| {
                                    panic!(
                                        "TypeError: expected string or bytes-like object, got 'NoneType'"
                                    )
                                })))
                            } else {
                                quote!(&(#text))
                            };
                            return Ok(quote!((#receiver).#m(#text_arg)));
                        }
                    }
                    _ => {}
                }
                // Issue #121: bytes methods on a name narrowed to Vec<u8>
                // (the bytes branch of a str|bytes union) — ASCII byte-wise
                // semantics, matching Python's bytes methods.
                let narrowed_bytes = crate::ast::tree::call::root_name(&attr.value)
                    .is_some_and(|root| {
                        options
                            .narrowed_names
                            .get(root)
                            .is_some_and(|t| matches!(t, crate::TypeInfo::Bytes))
                    });
                if narrowed_bytes {
                    match attr.attr.as_str() {
                        "lower" => {
                            return Ok(quote!((#receiver).lower()));
                        }
                        "startswith" => {
                            if let Some(arg) = self.args.first() {
                                let arg = arg.clone().to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                return Ok(quote!((#receiver).startswith(&(#arg))));
                            }
                        }
                        "endswith" => {
                            if let Some(arg) = self.args.first() {
                                let arg = arg.clone().to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                return Ok(quote!((#receiver).endswith(&(#arg))));
                            }
                        }
                        "isascii" => {
                            return Ok(quote!((#receiver).isascii()));
                        }
                        _ => {}
                    }
                }
            }
        }

        // A call to a SPECIALIZED function (specialize.rs): dispatch to
        // the variant matching the axis argument's static type — Python's
        // first-true-test order through the inheritance tree — or to the
        // `__any` residual for a type outside the tested set. An argument
        // whose type is not statically known cannot be dispatched: loud
        // conversion error, never a silently-wrong branch.
        if let ExprType::Name(callee_name) = self.func.as_ref()
            && let Some(spec) = options.specialized_fns.get(&callee_name.id).cloned()
        {
            if !self.keywords.is_empty() {
                return Err(format!(
                    "`{}` is specialized on its isinstance-tested parameter; \
                     calls take positional arguments only",
                    callee_name.id
                )
                .into());
            }
            let type_info_py_name = |t: &crate::TypeInfo| -> Option<String> {
                match t {
                    crate::TypeInfo::Int => Some("int".into()),
                    crate::TypeInfo::Float => Some("float".into()),
                    crate::TypeInfo::Bool => Some("bool".into()),
                    crate::TypeInfo::String | crate::TypeInfo::StrRef => {
                        Some("str".into())
                    }
                    crate::TypeInfo::Bytes => Some("bytes".into()),
                    _ => None,
                }
            };
            // A BOXED argument (PyValue — a heterogeneous-union value, a
            // call whose mixed returns unified to the box) has no static
            // type to dispatch on; when the dynamic router exists, the
            // dispatch happens at RUNTIME through it instead.
            let arg_is_boxed = |arg: &ExprType| -> bool {
                match arg {
                    ExprType::Name(n) => matches!(
                        options.name_types.get(&n.id),
                        Some(crate::TypeInfo::PyValue)
                    ),
                    ExprType::Call(c) => matches!(
                        crate::ast::tree::type_ctx::call_return_typeinfo(
                            c,
                            Some(&symbols),
                            Some(&options),
                        ),
                        Some(crate::TypeInfo::PyValue)
                    ),
                    _ => false,
                }
            };
            // The static (py type name, is_class) of an axis argument.
            let classify = |arg: &ExprType| -> (Option<String>, bool) {
                match arg {
                    ExprType::Name(n) => match options.name_types.get(&n.id) {
                        Some(crate::TypeInfo::Class(c)) => (Some(c.clone()), true),
                        Some(t) => (type_info_py_name(t), false),
                        None => (options.local_types.get(&n.id).cloned(), false),
                    },
                    // A call argument: a constructor
                    // (`describe(Dog("rex"))`) is an instance of its
                    // class; a known function resolves through its
                    // return type.
                    ExprType::Call(c) => {
                        let ctor = matches!(
                            c.func.as_ref(),
                            ExprType::Name(f)
                                if matches!(
                                    symbols.get(&f.id),
                                    Some(SymbolTableNode::ClassDef(_))
                                )
                        );
                        if ctor {
                            let ExprType::Name(f) = c.func.as_ref() else {
                                unreachable!()
                            };
                            (Some(f.id.clone()), true)
                        } else {
                            match crate::ast::tree::type_ctx::call_return_typeinfo(
                                c,
                                Some(&symbols),
                                Some(&options),
                            ) {
                                Some(crate::TypeInfo::Class(cn)) => (Some(cn), true),
                                Some(t) => (type_info_py_name(&t), false),
                                None => (None, false),
                            }
                        }
                    }
                    lit => (
                        crate::ast::tree::function_def::simple_expr_type(lit)
                            .and_then(|ty| {
                                crate::ast::tree::function_def::rust_type_to_py_name(&ty)
                                    .map(str::to_string)
                            }),
                        false,
                    ),
                }
            };
            // Type every axis argument.
            struct AxisSite<'a> {
                axis: &'a crate::ast::tree::specialize::SpecAxis,
                boxed: bool,
                py_ty: Option<String>,
                is_class: bool,
            }
            let mut sites: Vec<AxisSite> = Vec::new();
            for axis in &spec.axes {
                let Some(arg) = self.args.get(axis.index) else {
                    return Err(format!(
                        "`{}` needs at least {} positional argument(s)",
                        callee_name.id,
                        axis.index + 1
                    )
                    .into());
                };
                let boxed = arg_is_boxed(arg);
                let (py_ty, is_class) =
                    if boxed { (None, false) } else { classify(arg) };
                sites.push(AxisSite { axis, boxed, py_ty, is_class });
            }
            // An argument with NO statically-known type (a local reassigned
            // through untyped calls — botocore configloader's
            // `path = os.path.expandvars(path)` before `_unicode_path(path)`,
            // issue #161) dispatches at runtime through the router exactly
            // like a boxed one: `impl Into<Enum>` resolves From for the
            // argument's actual concrete type (or fails loudly at build
            // when that type is outside the tested set).
            if sites.iter().any(|s| s.boxed || s.py_ty.is_none()) {
                let Some(router) = &spec.router else {
                    // Round 54: a boxed/unknown axis argument with no
                    // dynamic router (an unannotated non-axis parameter
                    // blocked planning) — the isinstance dispatch cannot
                    // run, so the call DROPS loudly (the dynamic-dispatch
                    // divergence) instead of failing the whole module:
                    // requests' `_validate_header_part(header, name, 0)`
                    // where `name` comes from an untyped tuple
                    // destructure. The warning names the rewrite.
                    options.definition_warnings.borrow_mut().push(format!(
                        "`{}(...)` is dropped: its isinstance-dispatch axis argument \
                         is a boxed or statically-unknown value and no dynamic router \
                         could be planned (an unannotated non-axis parameter or an \
                         underivable morph return type); annotate the argument or \
                         the callee's non-axis parameters to keep the dispatch \
                         (the dynamic-dispatch divergence)",
                        callee_name.id
                    ));
                    return Ok(quote!(stdpython::PyValue::None_));
                };
                // Route the whole call through the router: each axis
                // parameter is `impl Into<Enum>`, so a boxed argument
                // passes unadorned (From<PyValue>, first-true-test
                // routing), a static argument matching a variant passes
                // as a plain value (From<T>), and a static argument
                // OUTSIDE the tested set boxes so it lands in `Other` —
                // the residual arm for that axis, exactly Python.
                use crate::ast::tree::specialize::{
                    axis_dispatch_suffix, py_id_boxable, RouterReturn,
                };
                let orig = crate::safe_ident(&callee_name.id);
                let mut rendered = Vec::new();
                for (i, arg) in self.args.iter().enumerate() {
                    let a = arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    let Some(site) =
                        sites.iter().find(|s| s.axis.index == i)
                    else {
                        rendered.push(a);
                        continue;
                    };
                    // Boxed and statically-unknown arguments both pass
                    // unadorned: the router's `impl Into<Enum>` does the
                    // routing (From<PyValue> tests variants in order;
                    // From<T> lands a concrete value directly).
                    if site.boxed || site.py_ty.is_none() {
                        rendered.push(a);
                        continue;
                    }
                    let Some(py_ty) = &site.py_ty else {
                        unreachable!("unknown-typed sites take the unadorned path above")
                    };
                    if axis_dispatch_suffix(site.axis, py_ty, site.is_class)
                        .is_some()
                        || site.is_class
                    {
                        // A class inside the tested subtree has its own
                        // From impl; one outside it cannot box — loud
                        // below via the From-less build if ever reached,
                        // so keep classes on the plain path only when a
                        // variant exists.
                        if site.is_class
                            && axis_dispatch_suffix(site.axis, py_ty, true)
                                .is_none()
                        {
                            return Err(format!(
                                "cannot dispatch `{}`: argument {} is a class \
                                 outside the isinstance-tested set, which has \
                                 no boxed representation for runtime routing",
                                callee_name.id,
                                i + 1
                            )
                            .into());
                        }
                        rendered.push(a);
                    } else if py_id_boxable(py_ty) {
                        rendered.push(quote!(stdpython::PyValue::from(#a)));
                    } else {
                        return Err(format!(
                            "cannot dispatch `{}`: argument {} has type `{}`, \
                             outside the isinstance-tested set, with no boxed \
                             representation for runtime routing",
                            callee_name.id,
                            i + 1,
                            py_ty
                        )
                        .into());
                    }
                }
                let call = quote!(#orig(#(#rendered),*));
                // An output-enum router (diverging morph returns):
                // Python's value here is the union — the boxed PyValue —
                // so the result converts on the way out. Members that
                // cannot box (a class) have no Python-faithful landing
                // at a boxed call site: loud.
                let boxes = match &router.ret {
                    RouterReturn::Unified(_) => None,
                    RouterReturn::Enum { members, .. } => {
                        if !members.iter().all(|m| py_id_boxable(m)) {
                            return Err(format!(
                                "cannot dispatch `{}` on a boxed value: its \
                                 morphs' return types include a class, so the \
                                 result has no boxed representation; annotate \
                                 the argument with a concrete type",
                                callee_name.id
                            )
                            .into());
                        }
                        Some(())
                    }
                };
                return Ok(match (boxes, propagates_exceptions) {
                    (None, true) => quote!((#call)?),
                    (None, false) => call,
                    (Some(()), true) => {
                        quote!(stdpython::PyValue::from((#call)?))
                    }
                    (Some(()), false) => {
                        return Err(format!(
                            "cannot dispatch `{}` on a boxed value in a \
                             context that does not propagate exceptions",
                            callee_name.id
                        )
                        .into());
                    }
                });
            }
            // Fully static: each axis picks its variant (or its "any"
            // residual), joined into the morph's suffix.
            let mut suffixes: Vec<String> = Vec::new();
            for site in &sites {
                let Some(py_ty) = &site.py_ty else {
                    return Err(format!(
                        "cannot dispatch the call to `{}`: the type of its \
                         isinstance-dispatched argument is not statically known — \
                         annotate the value (or the enclosing parameter) so the \
                         converter can pick the right specialization",
                        callee_name.id
                    )
                    .into());
                };
                suffixes.push(
                    crate::ast::tree::specialize::axis_dispatch_suffix(
                        site.axis, py_ty, site.is_class,
                    )
                    .unwrap_or("any")
                    .to_lowercase(),
                );
            }
            let suffix = suffixes.join("_");
            let mangled = crate::safe_ident(&crate::ast::tree::specialize::mangled_name(&callee_name.id, &suffix));
            let mut rendered = Vec::new();
            for arg in &self.args {
                rendered.push(arg.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            // A morph for a polymorphic ROOT (hierarchy.rs) takes the
            // root's sum type: a concrete struct of the subtree converts
            // on the way in.
            for site in &sites {
                if site.is_class
                    && let Some(py_ty) = &site.py_ty
                    && let Some(variant) = crate::ast::tree::specialize::axis_dispatch_suffix(site.axis, py_ty, true)
                    && crate::ast::tree::hierarchy::is_polymorphic_root(variant)
                    && let Some(a) = rendered.get_mut(site.axis.index)
                {
                    let inner = a.clone();
                    *a = quote!((#inner).into());
                }
            }
            let call = quote!(#mangled(#(#rendered),*));
            return Ok(if propagates_exceptions {
                quote!((#call)?)
            } else {
                call
            });
        }

        // Keyword arguments and omitted defaulted parameters resolve
        // against the callee's signature: keywords map to their parameter
        // positions and missing parameters fill from their default values,
        // matching Python call semantics. Without a known signature,
        // keywords would silently become misordered positional arguments —
        // that is a loud conversion error instead.
        // Issue #111: runtime-module functions with known signatures (the
        // warnings family) accept keyword arguments and omitted trailing
        // parameters — render through the signature's slots.
        if let Some(call) =
            self.render_runtime_signature(ctx.clone(), options.clone(), symbols.clone())?
        {
            return Ok(call);
        }

        let callee = match self.func.as_ref() {
            ExprType::Name(n) => {
                match symbols.get(&n.id) {
                Some(SymbolTableNode::FunctionDef(f)) => Some(f.clone()),
                // Issue #123: an IMPORTED function (`from pip._internal.
                // locations import get_scheme`) resolves through the
                // defining module's AST, with that module's symbol table —
                // the same cross-module lookup classes use. This unlocks
                // keyword arguments on imported functions and, via
                // `module_function_def`, return-annotation typing.
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let path = i.resolved_module_path(&options);
                    if options.module_defs.contains_key(&path) {
                        crate::module_function_def(&options, &path, &n.id).map(|(f, _)| f)
                    } else {
                        None
                    }
                }
                _ => None,
                }
            },
            _ => None,
        };
        if let Some(callee_def) = &callee {
            let pos_param_count = callee_def.args.posonlyargs.len() + callee_def.args.args.len();
            let has_optional_params = callee_def
                .args
                .posonlyargs
                .iter()
                .chain(callee_def.args.args.iter())
                .chain(callee_def.args.kwonlyargs.iter())
                .any(|p| {
                    p.annotation
                        .as_deref()
                        .is_some_and(crate::is_optional_annotation)
                });
            // A **kwargs or *args callee always routes through
            // map_call_arguments, which packs the extras into the boxed
            // PyDict / Vec<PyValue> (issue #120).
            let needs_mapping = !self.keywords.is_empty()
                || !callee_def.args.kwonlyargs.is_empty()
                || self.args.len() < pos_param_count
                || has_optional_params
                || callee_def.args.kwarg.is_some()
                || callee_def.args.vararg.is_some();
            if needs_mapping {
                let MappedArguments { prelude, args } = map_call_arguments(
                    callee_def,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                let name = self.func.to_rust(ctx, options, symbols)?;
                let call = quote!({ #prelude #name(#(#args),*) });
                return Ok(if propagates_exceptions {
                    // Parenthesize before `?`: a bare `{...}?` in statement
                    // position is not a valid expression statement (the
                    // block's tail value mismatches `()`), so `f(a=1)` on
                    // its own line failed to build (Devin review on #103,
                    // F9). `({...})?` is valid both as a statement and as
                    // an operand in an assignment/return.
                    quote!((#call)?)
                } else {
                    call
                });
            }
        } else if !self.keywords.is_empty() {
            // An unknown callee. A name that exists as a Python variable
            // (an assignment or import — e.g. a dynamically-imported module
            // member: `decoder = importlib.import_module(...).Klass`) has
            // no signature to map keywords against: lower the keyword
            // VALUES as trailing positional arguments (documented
            // divergence, dynamic dispatch). A completely unknown name is
            // likely a typo — stay loud.
            let known_var = match self.func.as_ref() {
                ExprType::Name(_n) => {
                    // A callable VALUE (`hook(hook_data, **kwargs)` — a
                    // hook function stored in a dict, requests/hooks.py)
                    // has no signature: the keywords lower positionally
                    // (the callable-as-value divergence, issue #122).
                    true
                }
                // A method call on a receiver (`proxy_manager.
                // connection_from_host(**host_params, ...)` — a PyValue
                // from a dict; `self.poolmanager...` — a field;
                // `session.request(method=..., **kwargs)` — a with-target
                // bound to an external class instance): the keywords lower
                // positionally (dynamic dispatch divergence).
                ExprType::Attribute(_) => true,
                // A call on a CALL result (`codecs.getincrementaldecoder(
                // ...)(errors="replace")`): the keywords lower positionally.
                ExprType::Call(_) => true,
                // A call through a DICT-LOOKUP result (`SERIALIZERS[
                // protocol_name](timestamp_precision=...)` — botocore's
                // serialize): the keywords lower positionally.
                ExprType::Subscript(_) => true,
                _ => false,
            };
            if !known_var {
                return Err(format!(
                    "keyword arguments require the callee's signature, and `{}` is not \
                     a function defined in this module; pass the arguments positionally",
                    self.func
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                        .map(|t| t.to_string())
                        .unwrap_or_else(|_| "<callee>".to_string())
                )
                .into());
            }
        }

        // A duck-typed user-method call on an unannotated parameter (M3)
        // returns Result (the generated Has* trait methods do): capture
        // whether to thread `?` before self.func is moved below.
        let duck_question = match self.func.as_ref() {
            ExprType::Attribute(attr) => crate::ast::tree::call::root_name(&attr.value)
                .is_some_and(|root| {
                    options
                        .duck_methods_on_params
                        .get(root)
                        .is_some_and(|methods| methods.contains(&attr.attr))
                }),
            _ => false,
        };

        // A call through a CALLED PARAMETER or loop element (`callback(
        // bytes_transferred=...)` — s3transfer's invoke_progress_callbacks,
        // where `callback` iterates an unannotated `callbacks` parameter):
        // the callable-as-value divergence (#122) — the call is dropped
        // (the boxed value is not callable in rython).
        if let ExprType::Name(callee_name) = self.func.as_ref()
            && options.called_params.contains(&callee_name.id)
        {
            options.definition_warnings.borrow_mut().push(format!(
                "call through callable value `{}` is dropped (the \
                 callable-as-value divergence, issue #122)",
                callee_name.id
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // A call through a SELF member that is neither a method nor a
        // field of the receiver's class (`self._tunnel()` — urllib3's
        // connection.connect, where _tunnel is a method INHERITED FROM
        // the external http.client base; the dispatch already ruled out
        // the class's own methods, and the field walk found nothing): the
        // member is an unmodeled callable VALUE — drop the call loudly
        // (the callable-as-value divergence, #122), exactly like the
        // called-parameter arm above.
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && let Some((class, class_symbols)) =
                crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            && !class.methods().any(|m| m.name == attr.attr)
            && (crate::ast::tree::attribute::class_field_access(
                &attr.value,
                &attr.attr,
                &ctx,
                &symbols,
                &options,
            )
            .is_none()
                // A FIELD the model types as the boxed PyValue (the
                // runtime-modeled accessor — `self._tunnel()` returns a
                // boxed value, not a callable).
                || class.infer_fields(&class_symbols, &options).ok().is_some_and(
                    |fields| {
                        fields
                            .iter()
                            .find(|(n, _)| *n == attr.attr)
                            .is_some_and(|(_, t)| {
                                matches!(
                                    t,
                                    crate::TypeInfo::PyValue | crate::TypeInfo::PyObject
                                )
                            })
                    },
                ))
        {
            options.definition_warnings.borrow_mut().push(format!(
                "call to `{}.{}(...)` is dropped: the member is neither a \
                 method nor a field of `{}` (the member is an unmodeled \
                 callable value — the callable-as-value divergence)",
                crate::ast::tree::call::expr_chain_spelling(&attr.value),
                attr.attr,
                class.name
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // A call through a VALUE the model cannot hold as a callable
        // (the callable-as-value divergence, #122): the callee's inferred
        // type is a boxed value / string / Option — or the callee is
        // itself a CALL whose result is called (`pool_cls(scheme=...)`
        // where pool_cls is the class-name string read from
        // pool_classes_by_scheme) — drop the call loudly, exactly like
        // the called-parameter arm above.
        let value_callee = match self.func.as_ref() {
            ExprType::Call(_) => true,
            e => {
                let mut drop = matches!(
                    crate::infer_type(Some(&ctx), e, &options, &symbols),
                    crate::TypeInfo::PyValue
                        | crate::TypeInfo::String
                        | crate::TypeInfo::Option(_)
                        | crate::TypeInfo::PyValueMember(_)
                );
                // A local assigned from a CONTAINER subscript
                // (`pool_cls = self.pool_classes_by_scheme[scheme]` —
                // urllib3's _new_pool, where the PyDict<String, String>
                // field holds CLASS NAMES): the local carries a container
                // value (a class name or a boxed member) — calling it is
                // the callable-as-value drop.
                if !drop
                    && let ExprType::Name(n) = e
                    && let Some(SymbolTableNode::Assign {
                        value: ExprType::Subscript(_),
                        ..
                    }) = symbols.get(&n.id)
                {
                    drop = true;
                }
                drop
            }
        };
        if value_callee {
            options.definition_warnings.borrow_mut().push(format!(
                "call through value `{}` is dropped (callables cannot be \
                 runtime values in rython — the callable-as-value divergence)",
                expr_chain_spelling(self.func.as_ref())
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // A call through a SIBLING-MODULE member that is NOT a module-level
        // function (`http2_probe.acquire_and_get(...)` — urllib3's
        // connection.py, where probe.py's acquire_and_get is a module-level
        // BOUND-METHOD alias `acquire_and_get = _HTTP2_PROBE_CACHE.
        // acquire_and_get`, not a `def`): the member is a callable VALUE —
        // rython cannot hold it, so the generated module has no `pub fn`
        // for it (E0425). Drop the call (the callable-as-value divergence).
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && crate::ast::tree::attribute::is_module_path_chain(
                &attr.value,
                &symbols,
                &options,
            )
            && let Some(module_path) = crate::ast::tree::call::module_path_of(&attr.value, &symbols, &options)
            && options.module_defs.contains_key(&module_path)
            && crate::module_function_def(&options, &module_path, &attr.attr).is_none()
            && crate::module_class_def(&options, &module_path, &attr.attr).is_none()
        {
            options.definition_warnings.borrow_mut().push(format!(
                "call through module member `{}.{}` is dropped (the member is a \
                 callable VALUE — a bound method or alias, not a module-level \
                 function; the callable-as-value divergence, issue #122)",
                module_path.last().map(|s| s.as_str()).unwrap_or(""),
                attr.attr
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }

        // A qualified STDPYTHON-CLASS construction (`collections.deque()`
        // — urllib3's response.py): the runtime item is a struct, so the
        // path call must go through its `::new` constructor exactly like
        // the from-import spelling does.
        if let ExprType::Attribute(attr) = self.func.as_ref()
            && let ExprType::Name(m) = attr.value.as_ref()
            && crate::StdModule::from_name(&m.id) == Some(crate::StdModule::Collections)
            && matches!(attr.attr.as_str(), "deque" | "OrderedDict" | "defaultdict")
        {
            let module = crate::safe_ident(&m.id);
            let cname = crate::safe_ident(&attr.attr);
            let mut args = Vec::new();
            for arg in &self.args {
                args.push(arg.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?);
            }
            return Ok(quote!(#module::#cname::new(#(#args),*)));
        }

        let name = self
            .func
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        let mut all_args = Vec::new();

        // When the callee is a user function defined in this module, the
        // direct-call path (simple signature, matching arity, no mapping)
        // still needs param-aware lowering: coerce to the parameter's
        // annotated type and clone non-Copy names that are reused later
        // (`f(x); g(x)` — Python shares by reference, Rust moves).
        let pos_params: Vec<&crate::Parameter> = match &callee {
            Some(f) => f
                .args
                .posonlyargs
                .iter()
                .chain(f.args.args.iter())
                .collect(),
            None => Vec::new(),
        };

        // Add positional arguments
        // (captured before the consuming loop so asyncio::sleep can re-render
        // its first argument as a float)
        let first_arg_expr = self.args.first().cloned();
        // The second argument's inferred type, captured before the consuming
        // loop (os.setenv's Option-vs-plain value routing).
        let second_arg_option = self.args.get(1).map(|a| {
            matches!(
                crate::infer_type(Some(&ctx), a, &options, &symbols),
                crate::TypeInfo::Option(_)
            )
        });
        for (i, arg) in self.args.into_iter().enumerate() {
            let rust_arg = if let Some(param) = pos_params.get(i) {
                // A CALLABLE parameter (`dict_class: type`): the argument
                // (a class name) lowers to its NAME STRING — the callee
                // cannot CALL through it (the callable-as-value drop at
                // the callee's own call sites), but the string is the
                // class object's runtime value (round 33 design).
                if param
                    .annotation
                    .as_deref()
                    .is_some_and(crate::ast::tree::arguments::is_type_annotation)
                {
                    if crate::is_class_value_expr(&arg, &symbols) {
                        let ExprType::Name(n) = &arg else {
                            unreachable!("is_class_value_expr matched a Name");
                        };
                        let name = n.id.clone();
                        quote!(#name.to_string())
                    } else {
                        quote!(stdpython::PyValue::None_)
                    }
                } else if crate::is_class_value_expr(&arg, &symbols)
                    && param.annotation.as_deref().and_then(crate::call_arg_expected_type)
                        .is_some_and(|t| {
                            let s = t.to_rust_type().to_string();
                            s == "stdpython :: PyValue" || s == "PyValue"
                        })
                {
                    // A CLASS passed to a PyValue-typed parameter
                    // (`merge_setting(..., CaseInsensitiveDict)` — requests'
                    // sessions.py, where dict_class: type maps to the boxed
                    // PyValue): the name string, boxed — a class as a value
                    // IS its name (round 33 design).
                    let ExprType::Name(n) = &arg else {
                        unreachable!("is_class_value_expr matched a Name");
                    };
                    let name = n.id.clone();
                    quote!(stdpython::PyValue::from(#name.to_string()))
                } else {
                    let expected = param
                        .annotation
                        .as_deref()
                        .and_then(crate::call_arg_expected_type)
                        // Round 86: the same symbols-aware fallback the
                        // mapped-call fill uses — an annotation the
                        // syntax-only mapping cannot see (a module-level
                        // alias resolving to the boxed PyValue:
                        // `_TYPE_TIMEOUT = Union[float, str, None]` —
                        // urllib3's `resolve_default_timeout(timeout)`
                        // into a `_TYPE_TIMEOUT` param) resolves through
                        // `resolve_alias_typeinfo`, so an OPTION-typed
                        // argument coerces (`Option<f64> → PyValue` via
                        // the Some/None match — Python's None passes
                        // through as the boxed None).
                        .or_else(|| {
                            arg_expected_fallback(param, &arg, &ctx, &symbols, &options)
                        });
                    let rendered = crate::render_typed_reused(
                        &arg,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        expected,
                    )?;
                    // An unannotated (inferred) parameter takes OWNED string
                    // literals: the recursion fixpoint needs
                    // `T: PyAdd<T, Output = T>`, which only String (not a
                    // `&'static str` literal) satisfies for self-adding params.
                    if param.annotation.is_none()
                        && matches!(
                            &arg,
                            ExprType::Constant(c)
                                if matches!(&c.0, Some(litrs::Literal::String(_)))
                        )
                    {
                        quote!((#rendered).to_string())
                    } else {
                        rendered
                    }
                }
            } else {
                // A CLASS NAME passed as a call argument without a resolved
                // signature (`merge_setting(request.headers, self.headers,
                // CaseInsensitiveDict)` — requests' sessions.py): classes
                // as values lower to their NAME STRINGS (round 33 design).
                if crate::is_class_value_expr(&arg, &symbols) {
                    let ExprType::Name(n) = &arg else {
                        unreachable!("is_class_value_expr matched a Name");
                    };
                    let name = n.id.clone();
                    quote!(#name.to_string())
                } else {
                    // The reuse-clone (round 99): an unknown callee (a
                    // runtime trait method like `"-".join(words)` — no
                    // FunctionDef signature) renders through the SAME
                    // clone-on-reuse renderer user calls use, so a value
                    // read again later (`len(words)` after the join moved
                    // it — the idiom corpus's main) clones at the earlier
                    // read. expected=None: no coercion, just the clone.
                    crate::render_typed_reused(
                        &arg,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        None,
                    )?
                }
            };
            all_args.push(rust_arg);
        }

        // Add keyword arguments
        for keyword in self.keywords {
            let rust_kw = keyword.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_kw);
        }

        // Check if we're in an async context and if the function being called is async
        let call_expr = quote!(#name(#(#all_args),*));

        // Check if this function returns a Result that should be unwrapped
        let name_str = format!("{}", name);

        // os.setenv(name, value) with an OPTIONAL value (requests'
        // set_environ — `os.environ[name] = value` where the value is a
        // narrowed Option<String>): route through setenv_opt, which
        // unwraps the Option (None is a no-op, matching Python's
        // `if value is None: return` guard). A non-Option value keeps
        // the plain setenv.
        if name_str == "os :: setenv" && all_args.len() == 2 {
            let first = &all_args[0];
            let second = &all_args[1];
            return Ok(if second_arg_option == Some(true) {
                quote!(os::setenv_opt(#first, #second))
            } else {
                quote!(os::setenv(#first, #second))
            });
        }

        // datetime.strptime parses and validates, so it raises ValueError
        // like Python; propagate rather than hand back a bare Result.
        if name_str.ends_with(":: strptime") {
            return Ok(quote!(#call_expr?));
        }
        // `subprocess :: run` and `os :: execv` are NOT here: the if-else
        // chain below handles both with dedicated arms before this branch
        // is consulted, so listing them was dead (the same name in two
        // dispatch structures of one function).
        let needs_unwrap = matches!(
            name_str.as_str(),
            "subprocess :: run_with_env"
                | "subprocess :: check_call"
                | "subprocess :: check_output"
                | "os :: getcwd"
                | "os :: chdir"
                | "os :: path :: abspath"
        );

        // Special handling for subprocess.run and os.execv with fallback for compatibility
        let final_call = if name_str == "json :: dumps" {
            // json.dumps takes its object by reference; Python code passes
            // it by value (`json.dumps(parsed)`), so borrow the first
            // argument here instead of failing with a mismatched type.
            if let Some((first, rest)) = all_args.split_first() {
                quote!(#name(&(#first), #(#rest),*))
            } else {
                quote!(#name)
            }
        } else if name_str == "asyncio :: run" {
            // asyncio.run(coro): drive the coroutine on the CURRENT runtime
            // (rython's entry point already runs under tokio), so the call
            // lowers to awaiting the future and unwrapping its Result. The
            // coroutine argument renders with a trailing `?` (calls to user
            // async functions propagate exceptions), which must apply to
            // the awaited Result, not the future — strip it like the Await
            // node does. In a synchronous context rustc rejects the `.await`
            // loudly.
            if let Some((first, rest)) = all_args.split_first() {
                let inner = strip_trailing_question(first);
                quote!(asyncio::run(#inner, #(#rest),*).await?)
            } else {
                quote!(#call_expr.await?)
            }
        } else if name_str == "asyncio :: sleep" {
            // asyncio.sleep(secs): suspend on tokio's timer. The argument
            // is a float in Python (`asyncio.sleep(1)` is valid too), so
            // coerce it like round() does. The enclosing `await` expression
            // (Await node) appends the `.await` — Python requires
            // `await asyncio.sleep(...)`, exactly like CPython's
            // "coroutine never awaited" warning when it is missing.
            if let Some(first) = first_arg_expr {
                let coerced = crate::render_typed(
                    &first,
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                    Some(crate::TypeInfo::Float),
                )?;
                if all_args.len() > 1 {
                    quote!(#name(#coerced, #(#all_args[1..]),*))
                } else {
                    quote!(#name(#coerced))
                }
            } else {
                quote!(#call_expr)
            }
        } else if name_str == "socket :: socket" && all_args.len() == 3 {
            // socket.socket(family, type, proto) — the 3-argument spelling
            // (the getaddrinfo loop passes the resolved proto). The runtime's
            // 2-arg `socket()` keeps the common spelling; the proto-validating
            // variant carries the third argument.
            quote!(socket::socket3(#(#all_args),*)?)
        } else if propagates_exceptions {
            quote!(#call_expr?)
        } else if name_str == "subprocess :: run" {
            // Try mixed_args version first, fallback to regular version
            if all_args.len() >= 2 {
                let args_param = &all_args[0];
                let cwd_param = &all_args[1];
                // Convert args to Vec<String> to avoid lifetime issues, then pass owned strings
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    let cwd_str = #cwd_param;
                    subprocess::run(args_vec, Some(&cwd_str)).unwrap()
                })
            } else {
                let args_param = &all_args[0];
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    subprocess::run(args_vec, None).unwrap()
                })
            }
        } else if name_str == "os :: execv" {
            // Convert to Vec<&str> for compatibility with standard execv function
            let program_param = &all_args[0];
            let args_param = &all_args[1];
            quote!({
                let program_str: String = (#program_param).clone();
                let args_owned: Vec<String> = #args_param;
                let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                os::execv(&program_str, args_vec).unwrap()
            })
        } else if duck_question {
            // The generated Has* trait method returns Result: exceptions
            // propagate like any user-class method call.
            quote!(#call_expr?)
        } else if needs_unwrap {
            quote!(#call_expr.unwrap())
        } else {
            call_expr
        };

        // `.await` is added only by an explicit `await` expression (the Await
        // node), mirroring Python: calling an async function without await
        // does not implicitly run it. The old behavior appended `.await` to
        // any call whose name started with "a" in async contexts, which broke
        // calls like abs(x).
        Ok(final_call)
    }
}

/// Lower a call to a bound function: a direct call into the bound crate,
/// with type-directed conversions between rython's Python types and the
/// declared Rust signature. Works for both declaration-style (`rust.bind`)
/// and import-style (`import crc32c`) bindings.
fn lower_rust_binding_call(
    spec: &crate::RustModuleSpec,
    fn_name: Option<&str>,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // rust.bind declarations always bind exactly one function; import-style
    // specs carry one (from-import) or many (module). Attribute calls name
    // the callee; bare-name calls require a single-function spec.
    let fspec = match fn_name {
        Some(name) => spec.get_fn(name).ok_or_else(|| {
            format!(
                "`{name}` is not a bound function of crate `{}`",
                spec.crate_name
            )
        })?,
        None => {
            if spec.fns.len() != 1 {
                return Err(format!(
                    "internal error: RustModule spec with multiple functions in a \
                     bare-name call (bound: {})",
                    spec.fns
                        .iter()
                        .map(|f| f.fn_name.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .into());
            }
            &spec.fns[0]
        }
    };

    if !keywords.is_empty() {
        return Err(format!(
            "keyword arguments in a call to bound function `{}::{}` are not \
             supported yet; pass the arguments positionally",
            spec.crate_name, fspec.fn_name
        )
        .into());
    }
    if args.len() != fspec.args.len() {
        return Err(format!(
            "bound function `{}::{}` takes {} argument(s), but {} were given",
            spec.crate_name,
            fspec.fn_name,
            fspec.args.len(),
            args.len()
        )
        .into());
    }

    let mut converted = Vec::new();
    for ((_, ty), arg) in fspec.args.iter().zip(args.iter()) {
        let rendered = arg
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        converted.push(convert_rust_bind_arg(rendered, ty)?);
    }

    let crate_ident = format_ident!("{}", spec.crate_name.replace('-', "_"));
    let fn_ident = format_ident!("{}", fspec.fn_name);
    let call = quote!(#crate_ident::#fn_ident(#(#converted),*));
    let call = if fspec.unsafe_call {
        quote!(unsafe { #call })
    } else {
        call
    };
    convert_rust_bind_ret(call, fspec.returns.as_deref())
}

/// Convert a Python-side argument to the declared Rust parameter type.
/// rython's Python types are fixed: int is i64, float is f64, str is String,
/// bytes is Vec<u8>, bool is bool — so every conversion is deterministic.
fn convert_rust_bind_arg(
    tokens: TokenStream,
    ty: &str,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let converted = match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize" => {
            let t: TokenStream = ty.parse().expect("validated rust type");
            quote!(#tokens as #t)
        }
        "f64" => tokens,
        "f32" => quote!(#tokens as f32),
        "bool" => tokens,
        // AsRef bridges both spellings: String and &'static str (literals and
        // module constants) both produce &str.
        "&str" => quote!(#tokens.as_ref()),
        // From is an identity impl for String and a conversion for &str.
        "String" => quote!(String::from(#tokens)),
        "&[u8]" => quote!(#tokens.as_ref()),
        // From is identity for Vec<u8> and converts from byte literals.
        "Vec<u8>" => quote!(Vec::from(#tokens)),
        "*const u8" => quote!(#tokens.as_ptr()),
        "*mut u8" => quote!(#tokens.as_mut_ptr()),
        other => return Err(format!("rust.bind: unsupported parameter type `{}`", other).into()),
    };
    Ok(converted)
}

/// Convert a bound call's result to the Python-side type. `()`/`void`/absent
/// returns stay bare so statement-position calls keep a plain `fn();` shape.
fn convert_rust_bind_ret(
    call: TokenStream,
    ret: Option<&str>,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let converted = match ret {
        None | Some("()") | Some("void") => call,
        Some("i64") | Some("i32") | Some("u64") | Some("u32") | Some("i16") | Some("u16")
        | Some("i8") | Some("u8") | Some("isize") | Some("usize") => {
            quote!(#call as i64)
        }
        Some("f64") => call,
        Some("f32") => quote!(#call as f64),
        Some("bool") => call,
        Some("String") => call,
        Some("&str") => quote!(#call.to_string()),
        Some("Vec<u8>") => call,
        Some(other) => return Err(format!("rust.bind: unsupported return type `{}`", other).into()),
    };
    Ok(converted)
}

/// The statically-known template of a `str.format` RECEIVER — the one
/// authority for which receivers the format lowering renders: a literal,
/// a module-level string constant, a self-field stored from a literal or
/// a class constant, a class constant (possibly imported). `None` is the
/// dynamic-template divergence: the lowering drops the call as the boxed
/// None, and the type side (type_ctx.rs) types the result the same way,
/// never as a String the lowering does not produce. Devin review on
/// #318.
pub(crate) fn str_format_template(
    receiver: &ExprType,
    ctx: Option<&CodeGenContext>,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<String> {
    Some(match receiver {
        ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_))) => {
            match &c.0 {
                Some(litrs::Literal::String(s)) => s.value().to_string(),
                _ => unreachable!(),
            }
        }
        // A MODULE-level constant template
        // (`LARGE_SECTION_MESSAGE.format(...)` — botocore's
        // restdoc): the constant's literal value.
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::Assign {
                value: ExprType::Constant(c),
                ..
            }) if matches!(&c.0, Some(litrs::Literal::String(_))) => {
                match &c.0 {
                    Some(litrs::Literal::String(s)) => s.value().to_string(),
                    _ => unreachable!(),
                }
            }
            _ => {
                    return None;
            }
        },
        // A SELF-FIELD template (`self.default_endpoint.format(
        // service=..., region=...)` — botocore's
        // Client._assume_endpoint, where the field stores
        // `default_endpoint or self.DEFAULT_ENDPOINT`): resolve
        // the field's stored value to a string template.
        ExprType::Attribute(a)
            if crate::ast::tree::visit::is_self(a.value.as_ref()) =>
        {
            let Some(enclosing) = ctx.and_then(|c| c.enclosing_class_name()) else {
                    return None;
            };
            let Some(SymbolTableNode::ClassDef(class)) = symbols.get(enclosing)
            else {
                    return None;
            };
            // The class's class-level string constants
            // (`self.DEFAULT_ENDPOINT` reads).
            let class_const = |attr: &str| -> Option<String> {
                class.body.iter().find_map(|s| match &s.statement {
                    crate::StatementType::Assign(assign)
                        if assign.targets.len() == 1
                            && matches!(
                                &assign.targets[0],
                                ExprType::Name(n) if n.id == attr
                            ) =>
                    {
                        match &assign.value {
                            ExprType::Constant(c) => match &c.0 {
                                Some(litrs::Literal::String(s)) => {
                                    Some(s.value().to_string())
                                }
                                _ => None,
                            },
                            _ => None,
                        }
                    }
                    _ => None,
                })
            };
            // The field's store value in __init__
            // (`self.<field> = <value>`).
            let store_value = class.init_method().and_then(|init| {
                init.body.iter().find_map(|s| match &s.statement {
                    crate::StatementType::Assign(assign)
                        if assign.targets.len() == 1
                            && matches!(
                                &assign.targets[0],
                                ExprType::Attribute(t)
                                    if crate::ast::tree::visit::is_self(t.value.as_ref()) && t.attr == a.attr
                            ) =>
                    {
                        Some(&assign.value)
                    }
                    _ => None,
                })
            });
            let Some(v) = store_value else {
                    return None;
            };
            match template_from_expr(v, &class_const) {
                Some(t) => t,
                None => {
                    // A self-field template whose stored value
                    // is a PARAMETER (`self.text_format.format(
                    // task=task)` — rich's TextColumn, where
                    // __init__ stores the text_format
                    // parameter): the template is dynamic at
                    // conversion time — the format call is
                    // dropped (the dynamic-format divergence).
                    return None;
                }
            }
        }
        // A CLASS-CONSTANT template (`ResponseError.SPECIFIC_ERROR
        // .format(...)` — urllib3's exceptions): the class's
        // class-level string constants are metadata, but the
        // template is statically known.
        ExprType::Attribute(a) => {
            let ExprType::Name(class) = a.value.as_ref() else {
                    return None;
            };
            // The class may be imported (`from .exceptions import
            // ResponseError` in urllib3/util/retry.py): resolve
            // through the defining module.
            let class_def = match symbols.get(&class.id) {
                Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let path = i.resolved_module_path(options);
                    crate::resolve_imported_class(options, &path, &class.id, 0)
                        .map(|(c, _)| c)
                }
                _ => None,
            };
            let Some(c) = class_def else {
                    return None;
            };
            let Some(assign) = c.body.iter().find_map(|s| match &s.statement {
                crate::StatementType::Assign(assign)
                    if assign.targets.len() == 1
                        && matches!(&assign.targets[0], ExprType::Name(n) if n.id == a.attr) =>
                {
                    Some(assign)
                }
                _ => None,
            }) else {

                    return None;
            };
            match &assign.value {
                ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))) =>
                {
                    match &c.0 {
                        Some(litrs::Literal::String(s)) => s.value().to_string(),
                        _ => unreachable!(),
                    }
                }
                _ => {
                    return None;
                }
            }
        }
        _ => {
            // A RUNTIME template (`template.format(service=...,
            // region=...)` — botocore's regions, where the
            // template is a parameter): the template cannot be
            // checked at conversion time — the format call is
            // dropped (the dynamic-format divergence).
            return None;
        }
    })
}

/// Lower a literal `template.format(args...)` call to a Rust `format!`.
///
/// Every argument (used or not) is evaluated exactly once, in Python's
/// order, into a local binding — Python evaluates unused arguments too.
/// Used bindings are referenced from the format string by name; unused
/// ones bind to `_` so no warning fires. Errors mirror Python's:
/// mixing auto and manual numbering, out-of-range indices, and missing
/// keywords are conversion-time failures.
fn lower_str_format(
    template: &str,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    use crate::pyformat::{FieldRef, Piece, parse_template, translate_format_spec};

    let pieces = match parse_template(template) {
        Ok(p) => p,
        Err(e) => {
            // A template with fields this lowering cannot express
            // (`{data[installer][name]}` — pip's user agent): attribute/
            // index access inside fields is unmodeled — the format call is
            // dropped (documented divergence).
            options.definition_warnings.borrow_mut().push(format!(
                "str.format({:?}) is dropped: {}",
                template, e
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }
    };

    // A `*t` SPREAD argument (`"{}.{}".format(*sys.version_info)` — pip's
    // locations): the spread supplies the positional fields beyond the
    // explicit arguments — render the spread once per remaining field
    // (the spread-argument divergence).
    let mut args: Vec<&ExprType> = args.iter().collect();
    if let Some(pos) = args.iter().position(|a| matches!(a, ExprType::Starred(_))) {
        let field_count = pieces
            .iter()
            .filter(|p| matches!(p, Piece::Field { arg: FieldRef::Auto, .. }))
            .count();
        let explicit = args
            .iter()
            .filter(|a| !matches!(a, ExprType::Starred(_)))
            .count();
        if field_count > explicit {
            let ExprType::Starred(st) = args.remove(pos) else {
                unreachable!()
            };
            for _ in explicit..field_count {
                args.insert(pos, &st.value);
            }
        }
    }

    // A `**d` SPREAD with STATICALLY-KNOWN keys (`...format(**dict(opts,
    // links=...))` — pip's req_command): the dict's entries resolve as
    // format keywords by name. A spread with dynamic keys stays a loud
    // error.
    let mut resolved_keywords: Vec<crate::Keyword> = Vec::new();
    for kw in keywords {
        let Some(_name) = &kw.arg else {
            let mut entries: Vec<(String, ExprType)> = Vec::new();
            match &kw.value {
                ExprType::Name(n) => {
                    if let Some(SymbolTableNode::Assign {
                        value: ExprType::Dict(d),
                        ..
                    }) = symbols.get(&n.id)
                    {
                        for (k, v) in d.keys.iter().zip(d.values.iter()) {
                            if let Some(ExprType::Constant(c)) = k
                                && let Some(litrs::Literal::String(sv)) = &c.0
                            {
                                entries.push((sv.value().to_string(), (*v).clone()));
                            }
                        }
                    }
                }
                ExprType::Call(c)
                    if matches!(c.func.as_ref(), ExprType::Name(f) if f.id == "dict") =>
                {
                    // The first argument may be a dict LITERAL, or a NAME
                    // bound to one (`dict(opts, links=...)` where
                    // `opts = {"name": ...}` — pip's req_command).
                    let first_dict: Option<&crate::Dict> = match c.args.first() {
                        Some(ExprType::Dict(d)) => Some(d),
                        Some(ExprType::Name(n)) => {
                            match symbols.get(&n.id) {
                                Some(SymbolTableNode::Assign {
                                    value: ExprType::Dict(d),
                                    ..
                                }) => Some(d),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    if let Some(d) = first_dict {
                        for (k, v) in d.keys.iter().zip(d.values.iter()) {
                            if let Some(ExprType::Constant(c)) = k
                                && let Some(litrs::Literal::String(sv)) = &c.0
                            {
                                entries.push((sv.value().to_string(), (*v).clone()));
                            }
                        }
                    }
                    for kk in &c.keywords {
                        if let Some(kname) = &kk.arg {
                            entries.push((kname.clone(), kk.value.clone()));
                        }
                    }
                }
                _ => {}
            }
            if entries.is_empty() {
                return Err("str.format with **kwargs is not supported yet"
                    .to_string()
                    .into());
            }
            for (ename, evalue) in entries {
                resolved_keywords.push(crate::Keyword {
                    arg: Some(ename),
                    value: evalue,
                    ..Default::default()
                });
            }
            continue;
        };
        resolved_keywords.push(kw.clone());
    }

    let keywords = &resolved_keywords;
    // Resolve each field to an argument slot and build the format string.
    let mut fmt = String::new();
    let mut used_positions: std::collections::HashSet<usize> = Default::default();
    let mut used_names: std::collections::HashSet<String> = Default::default();
    let mut auto_next = 0usize;
    let mut saw_auto = false;
    let mut saw_manual = false;
    let mut field_bindings: Vec<TokenStream> = Vec::new();
    for piece in &pieces {
        match piece {
            Piece::Literal(text) => {
                fmt.push_str(&text.replace('{', "{{").replace('}', "}}"));
            }
            Piece::Field {
                arg,
                conversion,
                spec,
            } => {
                let index_name = match arg {
                    FieldRef::Auto => {
                        saw_auto = true;
                        let i = auto_next;
                        auto_next += 1;
                        if i >= args.len() {
                            return Err(format!(
                                "str.format: not enough positional arguments (field {} of \
                                 template {:?})",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Index(i) => {
                        saw_manual = true;
                        if *i >= args.len() {
                            return Err(format!(
                                "str.format: replacement index {} out of range for \
                                 template {:?}",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(*i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Name(name) => {
                        if !keywords
                            .iter()
                            .any(|k| k.arg.as_deref() == Some(name.as_str()))
                        {
                            return Err(format!(
                                "str.format: template {:?} refers to {:?}, which is not \
                                 among the keyword arguments",
                                template, name
                            )
                            .into());
                        }
                        used_names.insert(name.clone());
                        format!("__rython_fmt_{}", name)
                    }
                };
                if saw_auto && saw_manual {
                    return Err(
                        "str.format: cannot switch between automatic field numbering \
                         and manual field specification"
                            .to_string()
                            .into(),
                    );
                }
                let is_repr = matches!(conversion, Some('r') | Some('a'));
                let lowering =
                    translate_format_spec(spec).map_err(|e| format!("str.format: {}", e))?;
                if is_repr {
                    // Python's !r renders the repr STRING and applies the
                    // spec to it; Rust's `{:?}` would print its own Debug
                    // form ("ab" with double quotes) and diverge.
                    let crate::pyformat::SpecLowering::Inline(suffix) = lowering else {
                        return Err("str.format: numeric presentation types cannot combine \
                                    with !r/!a (Python applies the spec to the repr string \
                                    and raises)"
                            .to_string()
                            .into());
                    };
                    let fld = format!("__rython_fld{}", field_bindings.len());
                    let src = crate::safe_ident(&index_name);
                    let ident = crate::safe_ident(&fld);
                    field_bindings.push(quote!(let #ident = repr(&(#src));));
                    if suffix.is_empty() {
                        fmt.push_str(&format!("{{{}}}", fld));
                    } else {
                        fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                    }
                    continue;
                }
                match lowering {
                    crate::pyformat::SpecLowering::Inline(suffix) => {
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", index_name));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", index_name, suffix));
                        }
                    }
                    // The operand coerces or converts per-field (one
                    // argument may be reused with different specs), via a
                    // field-local binding referencing the argument's.
                    crate::pyformat::SpecLowering::CastF64(suffix) => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(let #ident = (#src) as f64;));
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", fld));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                        }
                    }
                    crate::pyformat::SpecLowering::IntRadix {
                        fill,
                        align,
                        plus,
                        alternate,
                        zero,
                        width,
                        radix,
                    } => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(
                            let #ident = py_int_radix_format(
                                #src, #fill, #align, #plus, #alternate, #zero, #width, #radix,
                            );
                        ));
                        fmt.push_str(&format!("{{{}}}", fld));
                    }
                    // Python's general float format: the runtime renders
                    // the significant digits; fill/align/width apply after.
                    crate::pyformat::SpecLowering::GeneralFloat { precision, suffix } => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(
                            let #ident = py_format_g((#src) as f64, #precision);
                        ));
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", fld));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                        }
                    }
                    // The `,` thousands separator: the runtime groups the
                    // integer's digits.
                    crate::pyformat::SpecLowering::GroupedInt => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(let #ident = py_grouped_int(#src);));
                        fmt.push_str(&format!("{{{}}}", fld));
                    }
                }
            }
        }
    }

    // Bindings: every argument evaluates exactly once, in order.
    let mut bindings = TokenStream::new();
    for (i, arg) in args.iter().enumerate() {
        let value = (*arg)
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_positions.contains(&i) {
            let ident = crate::safe_ident(&format!("__rython_fmt{}", i));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }
    for kw in keywords {
        let name = kw.arg.as_deref().unwrap_or_default();
        let value = kw
            .value
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_names.contains(name) {
            let ident = crate::safe_ident(&format!("__rython_fmt_{}", name));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }

    for fb in field_bindings {
        bindings.extend(fb);
    }

    Ok(quote!({
        #bindings
        format!(#fmt)
    }))
}

/// Extract a statically-known string TEMPLATE from an expression, for
/// str.format lowering: a string literal, an `x or y` chain (the first
/// branch that yields a literal — `default_endpoint or
/// self.DEFAULT_ENDPOINT`, botocore's Client), or a `self.CONST` read of
/// a class-level string constant.
fn template_from_expr(
    v: &ExprType,
    class_const: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    match v {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(s)) => Some(s.value().to_string()),
            _ => None,
        },
        ExprType::BoolOp(b) => b
            .values
            .iter()
            .find_map(|bv| template_from_expr(bv, class_const)),
        ExprType::Attribute(inner) => match inner.value.as_ref() {
            ExprType::Name(n) if n.id == "self" => class_const(&inner.attr),
            _ => None,
        },
        _ => None,
    }
}

/// Escape `{`/`}` in a Python regex pattern for Rust's regex crate: Python
/// treats an unescaped brace as literal, Rust treats `{` as a repetition
/// start (`{(.*?)}` — botocore's serialize). Valid `{n}`, `{n,}`, `{n,m}`
/// repetitions and already-escaped braces pass through.
fn escape_regex_braces(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                out.push(c);
                if i + 1 < chars.len() {
                    out.push(chars[i + 1]);
                    i += 1;
                }
            }
            '{' => {
                let mut j = i + 1;
                while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == ',') {
                    j += 1;
                }
                let is_repetition = j < chars.len()
                    && chars[j] == '}'
                    && chars[i + 1..j].iter().any(|ch| ch.is_ascii_digit());
                if is_repetition {
                    while i <= j {
                        out.push(chars[i]);
                        i += 1;
                    }
                    continue;
                }
                out.push('\\');
                out.push(c);
            }
            '}' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Whether a function is a raise-only NotImplementedError stub — an
/// abstract method (`_do_modeled_error_parse` — botocore's parsers).
/// Whether a stub call's MISSING positional arguments can all be filled
/// by the argument-mapping (each has a Python default or is
/// Option-annotated): such a call lowers to the full-arity invocation
/// (virtual dispatch), so it must NOT be dropped. Only a stub whose
/// missing params are REQUIRED (unmappable — botocore's extra `parsed`)
/// is dropped.
fn stub_missing_args_defaultable(sig: &crate::FunctionDef, supplied: usize) -> bool {
    let posonly = sig.args.posonlyargs.len();
    let pos = posonly + sig.args.args.len();
    if supplied >= pos {
        return true;
    }
    let pos_defaulted = sig.args.defaults.len().min(pos);
    let first_defaulted = pos - pos_defaulted;
    (supplied..pos).all(|i| {
        if i >= first_defaulted {
            return true;
        }
        let p = if i < posonly {
            &sig.args.posonlyargs[i]
        } else {
            &sig.args.args[i - posonly]
        };
        p.annotation
            .as_deref()
            .is_some_and(crate::is_optional_annotation)
    })
}


fn is_notimpl_stub(f: &crate::FunctionDef) -> bool {
    f.body.len() == 1
        && matches!(
            &f.body[0].statement,
            crate::StatementType::Raise(_)
        )
}

/// The class of a method-call receiver, when it is statically known:
/// `self` inside a class's method body, or a local/module name whose
/// (symbol-table-recorded) assignment constructs a known class. Unknown
/// receivers return None and fall through to the generic lowering — where
/// a genuine user-method call fails to compile (loud), never silently
/// drops exception propagation.
/// Whether `expr` is a `super` reference: the bare Name (defensive; Python
/// has no `super.foo` form) or the real shape, a `super()` Call. The
/// receiver of `super().m(...)`.
fn is_super_reference(expr: &ExprType) -> bool {
    match expr {
        ExprType::Name(n) => n.id == "super",
        ExprType::Call(c) => {
            c.args.is_empty()
                && c.keywords.is_empty()
                && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "super")
        }
        _ => false,
    }
}

pub(crate) fn receiver_class(
    recv: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    let (class_name, class_symbols) = match recv {
        ExprType::Name(n) if n.id == "self" => {
            (ctx.enclosing_class_name()?.to_string(), symbols.clone())
        }
        // `super()` inside a method resolves to the enclosing class's base,
        // so `super().m(...)` (and mut analysis for `super().__init__`)
        // resolve against the base class. Python parses `super()` as a
        // Call node — `super` is a function — so the receiver must match
        // the Call shape, not a bare Name.
        recv if is_super_reference(recv) => {
            let class_name = ctx.enclosing_class_name()?;
            let class = match symbols.get(class_name) {
                Some(SymbolTableNode::ClassDef(c)) => c.clone(),
                _ => return None,
            };
            let base = class.base_class(symbols)?;
            return Some((base, symbols.clone()));
        }
        ExprType::Name(n) => {
            // A local bound to a call (`c = make()`, `c = Counter()`): the
            // class the call produces, through the one authority below.
            let named = match symbols.get(&n.id) {
                Some(SymbolTableNode::Assign {
                    value: ExprType::Call(call),
                    ..
                }) => match call.func.as_ref() {
                    ExprType::Name(_) => named_call_class(call, symbols, options),
                    _ => None,
                },
                _ => None,
            };
            match named {
                Some(resolved) => resolved,
                // Otherwise the analysis's recorded type is the receiver's
                // class: a TYPED PARAMETER (`def f(c: C): return c.x`), or
                // a local fetched from a container (`item = self.items.get(
                // name)` — `Item | None`, narrowed by its guard), so a
                // method call on it is the user-method call with its `?`
                // and, for a SHARED class, the borrow.
                None => match options.name_types.get(&n.id) {
                    Some(crate::TypeInfo::Class(cname)) => (cname.clone(), symbols.clone()),
                    Some(crate::TypeInfo::Option(inner)) => match inner.as_ref() {
                        crate::TypeInfo::Class(cname) => (cname.clone(), symbols.clone()),
                        _ => return None,
                    },
                    _ => return None,
                },
            }
        }
        // Composition: `self.field.method()` resolves through the owner
        // class's field types. `field_class` yields the field's class
        // NAME, resolved to its ClassDef below.
        ExprType::Attribute(attr) => {
            let (owner, owner_symbols) = receiver_class(&attr.value, ctx, symbols, options)?;
            let field = owner.field_class(&attr.attr, &owner_symbols, options)?;
            (field, owner_symbols)
        }
        // A METHOD CALL as the receiver (`self.proxy().host` — urllib3's
        // connection_from_url, where proxy() returns a ProxyConfig whose
        // host field is Option-typed): the double-wrap family's remaining
        // shape — resolve the method's return class through the
        // receiver's class (round 58).
        ExprType::Call(call) => match call.func.as_ref() {
            // A CONSTRUCTOR or factory call as the receiver itself
            // (`Shape().area()`, `make().run()`): the same class the local
            // bound to that call would have (the idiom corpus's shapes).
            ExprType::Name(_) => named_call_class(call, symbols, options)?,
            ExprType::Attribute(attr) => {
                let (owner, owner_symbols) =
                    receiver_class(&attr.value, ctx, symbols, options)?;
                // The callee may be a real METHOD returning a class
                // instance, or a FIELD ACCESSOR (`self.proxy()` where
                // proxy is a ProxyConfig-typed field — urllib3): the
                // accessor returns the field's class.
                let class = match owner
                    .method_on_mro_with_options(&attr.attr, &owner_symbols, options)
                    .and_then(|m| m.return_class_name(options))
                {
                    Some(class) => class,
                    None => match owner.field_class(&attr.attr, &owner_symbols, options) {
                        Some(class) => class,
                        // A method returning an OPTION of a class
                        // (`inv.find("bolt").label()` — find is
                        // `Item | None`): the read unwraps the Option
                        // (the AttributeError-on-None machinery), so the
                        // receiver's class is the INNER one (round 99).
                        None => {
                            match crate::call_return_typeinfo(
                                call,
                                Some(symbols),
                                Some(options),
                            ) {
                                Some(crate::TypeInfo::Option(inner)) => match *inner {
                                    crate::TypeInfo::Class(cname) => cname,
                                    _ => return None,
                                },
                                _ => return None,
                            }
                        }
                    },
                };
                (class, owner_symbols)
            }
            _ => return None,
        },
        // Any other receiver the inferrer types as a class (a subscript
        // into a container of a class — `self.accounts[src].withdraw(n)`,
        // the idiom corpus's bank — or an Option of one): the one type
        // authority resolves it.
        other => match crate::infer_type(Some(ctx), other, options, symbols) {
            crate::TypeInfo::Class(cname) => (cname, symbols.clone()),
            crate::TypeInfo::Option(inner) => match *inner {
                crate::TypeInfo::Class(cname) => (cname, symbols.clone()),
                _ => return None,
            },
            _ => return None,
        },
    };
    receiver_class_tail(&class_name, class_symbols, options)
}

/// The class a NAME-callee call produces, with the symbol table its name
/// resolves in: a local factory (`make()` where `def make() -> Counter`,
/// or the unannotated lazy-singleton getter, issue #189) through the
/// function's return; an IMPORTED factory (`parse_url(url)` — urllib3,
/// from .util) through the defining module the same way (round 58: the
/// double-wrap family — `Some(u.host)` nested because the local's class
/// was never resolved; Devin review on #264: KEEP the defining module's
/// symbol table, its return annotation names classes THERE); otherwise a
/// constructor, local or imported (the tail resolves an imported class
/// through its defining module). `None` when the callee is a function
/// without a class-typed return. One authority for a local bound to the
/// call AND for the call used directly as a receiver.
fn named_call_class(
    call: &crate::Call,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(String, SymbolTableScopes)> {
    let ExprType::Name(cn) = call.func.as_ref() else {
        return None;
    };
    // An ALIASED import (`from pkg.shapes import Shape as S`) binds the
    // local name to the canonical one, which carries the ImportFrom: the
    // defining module knows only the canonical name.
    let (name, node) = match symbols.get(&cn.id) {
        Some(SymbolTableNode::Alias(canonical)) => (canonical.clone(), symbols.get(canonical)),
        other => (cn.id.clone(), other),
    };
    let fdef = match node {
        Some(SymbolTableNode::FunctionDef(f)) => Some((f.clone(), symbols.clone())),
        Some(SymbolTableNode::ImportFrom(ifm)) => {
            // The module key authority (`module_defs_key`) covers the
            // package's own root-qualified spelling (`from pkg.session
            // import make` in a src-layout sdist, keyed ["session"]).
            let path = ifm.resolved_module_path(options);
            let key = crate::module_defs_key(options, &path);
            // An imported CLASS constructor: the class itself, with its
            // defining module's symbols (the same key).
            if let Some(key) = key
                && let Some((class, class_symbols)) = crate::module_class_def(options, key, &name)
            {
                return Some((class.name, class_symbols));
            }
            key.and_then(|key| crate::module_function_def(options, key, &name))
        }
        _ => None,
    };
    match fdef {
        Some((f, f_symbols)) => f.return_class_name(options).map(|class| (class, f_symbols)),
        None => Some((name, symbols.clone())),
    }
}

/// Render a `key=` function for a builtin over `iterable`. A LAMBDA's
/// parameter is the iterable's element (`key=lambda s: s.area()` over a
/// `list[Shape]`), typed through the same binder authority the
/// for-statement and the comprehension use, so the body's method calls
/// resolve their receiver's class — and emit the `?` a user method's
/// Result needs (the idiom corpus's shapes: an untyped `s` left `s.area()`
/// a Result inside the key closure, E0277). Anything else renders as is.
fn render_key_fn(
    key: &ExprType,
    iterable: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    render_lambda_over(key, &[iterable], ctx, options, symbols)
}

/// Render a function argument whose lambda's parameters are, in order,
/// the elements of `iterables` (`map(lambda x, y: ..., xs, ys)` binds `x`
/// to xs's element and `y` to ys's). Each parameter is a fresh binding
/// whether or not its element type is known.
fn render_lambda_over(
    f: &ExprType,
    iterables: &[&ExprType],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if let ExprType::Lambda(lam) = f {
        let elems: Vec<Option<crate::TypeInfo>> = iterables
            .iter()
            .map(|it| {
                crate::ast::tree::type_ctx::iterable_element_type(&crate::infer_type(
                    Some(ctx),
                    it,
                    options,
                    symbols,
                ))
            })
            .collect();
        let scope = crate::lambda_scope(lam, &elems, options);
        return f.clone().to_rust(ctx.clone(), scope, symbols.clone());
    }
    f.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())
}

/// Resolve a class NAME to its ClassDef (and the defining module's symbol
/// table): same-module, or an imported class through its defining module.
pub(crate) fn receiver_class_tail(
    class_name: &str,
    class_symbols: SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    match class_symbols.get(class_name) {
        Some(SymbolTableNode::ClassDef(c)) => Some((c.clone(), class_symbols)),
        // An imported class (`from .animals import Dog`): the binding is a
        // name, so resolve through the DEFINING module's AST, where the
        // class (and its base chain) is declared — with the defining
        // module's symbol table, so `Dog`'s base `Animal` resolves there.
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            if options.module_defs.contains_key(&path) {
                crate::module_class_def(options, &path, class_name)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The attribute-READ path's receiver resolution: [`receiver_class`] plus
/// the FACTORY-LOCAL shapes (`x = self._make()` / `x = imported_factory(
/// ...)` whose return annotation names the class). The attribute path
/// needs these so PROPERTY reads and base-chain field walks route
/// correctly on such locals (`timeout_obj.connect_timeout` — urllib3,
/// whose `_get_timeout` returns `Timeout`; issue #137's E0615 cluster).
/// The METHOD-DISPATCH path deliberately keeps the conservative
/// [`receiver_class`]: resolving a callee's full signature from a
/// factory-inferred receiver perturbs dropped-default inlining (E0425
/// regressions) with no compensating gain.
pub(crate) fn receiver_class_for_read(
    recv: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    if let Some(r) = receiver_class(recv, ctx, symbols, options) {
        return Some(r);
    }
    // A DIRECT factory CALL as the receiver (`parse_url(url).netloc` —
    // a property of the return class, or `Url(...).url` — a property of
    // the CONSTRUCTED class): the same resolution as the NAME-assign
    // factory, for the call in place instead of a local. The
    // conservative receiver_class keeps method-call receivers only; the
    // property surface needs this shape.
    if let ExprType::Call(call) = recv
        && let ExprType::Name(cn) = call.func.as_ref()
    {
        // A CLASS CONSTRUCTION — the class itself (local, or imported:
        // `Url(...).url` where Url comes from another module).
        if matches!(symbols.get(&cn.id), Some(SymbolTableNode::ClassDef(_))) {
            return receiver_class_tail(&cn.id, symbols.clone(), options);
        }
        if let Some(SymbolTableNode::ImportFrom(i)) = symbols.get(&cn.id) {
            let path = i.resolved_module_path(options);
            // An IMPORTED CLASS construction.
            if options.module_defs.contains_key(&path)
                && crate::module_class_def(options, &path, &cn.id).is_some()
            {
                return receiver_class_tail(&cn.id, symbols.clone(), options);
            }
            // An IMPORTED factory (`parse_url(url)`).
            let (f, f_symbols) = crate::module_function_def(options, &path, &cn.id)?;
            let class_name = f.return_class_name(options)?;
            return receiver_class_tail(&class_name, f_symbols, options);
        }
    }
    let ExprType::Name(n) = recv else {
        return None;
    };
    let Some(SymbolTableNode::Assign {
        value: ExprType::Call(call),
        ..
    }) = symbols.get(&n.id)
    else {
        return None;
    };
    let class_name = match call.func.as_ref() {
        // A SELF-METHOD factory: `c = self._make()` — the class comes
        // from the method's return annotation. Only the bare `self`
        // receiver: a deeper receiver would recurse through arbitrary
        // assign chains (cycle risk).
        ExprType::Attribute(attr)
            if crate::ast::tree::visit::is_self(attr.value.as_ref()) =>
        {
            let (owner, _) = receiver_class(&attr.value, ctx, symbols, options)?;
            let method = owner.methods().find(|m| m.name == attr.attr)?;
            method.return_class_name(options)?
        }
        // A SUPER-method factory: `r = super().make()` — an override that
        // assigns the base's result (a method reads its own override's
        // base-chain members afterwards). The base's method (not the
        // enclosing class's own, which does not define it) provides the
        // return class.
        ExprType::Attribute(attr)
            if matches!(
                attr.value.as_ref(),
                ExprType::Call(c)
                    if matches!(c.func.as_ref(), ExprType::Name(s) if s.id == "super")
            ) =>
        {
            let class_name = ctx.enclosing_class_name()?;
            let crate::SymbolTableNode::ClassDef(cls) = symbols.get(class_name)? else {
                return None;
            };
            let base = cls.base_chain(symbols).into_iter().find(|c| {
                c.methods().any(|m| m.name == attr.attr)
            })?;
            let method = base.methods().find(|m| m.name == attr.attr)?;
            method.return_class_name(options)?
        }
        // An IMPORTED factory (`parsed_url = parse_url(url)` — urllib3,
        // whose parse_url is imported from util.url): resolve the function
        // through its defining module and take its return class. The
        // return annotation names classes in the DEFINING module, so the
        // tail resolves against its symbol table (the same rule the
        // conservative receiver_class follows for imported factories).
        ExprType::Name(cn) => {
            let Some(SymbolTableNode::ImportFrom(i)) = symbols.get(&cn.id) else {
                return None;
            };
            let path = i.resolved_module_path(options);
            let (f, f_symbols) = crate::module_function_def(options, &path, &cn.id)?;
            let class_name = f.return_class_name(options)?;
            return receiver_class_tail(&class_name, f_symbols, options);
        }
        _ => return None
    };
    receiver_class_tail(&class_name, symbols.clone(), options)
}

/// Whether `recv` is a `self.<field>` expression whose field's inferred
/// Rust type is the boxed `stdpython::PyValue` (a `set()` bookkeeping
/// field, an external-object field, ...). Method calls on such receivers
/// lower as no-ops (the boxed value's methods are unmodeled).
pub(crate) fn receiver_is_pyvalue_self_field(
    recv: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Attribute(attr) = recv else {
        return false;
    };
    if !crate::ast::tree::visit::is_self(attr.value.as_ref()) {
        return false;
    }
    let Some((class, class_symbols)) = receiver_class(&attr.value, ctx, symbols, options) else {
        return false;
    };
    class
        .infer_fields(&class_symbols, options)
        .ok()
        .and_then(|fields| {
            fields
                .iter()
                .find(|(name, _)| name == &attr.attr)
                .map(|(_, ty)| matches!(ty, crate::TypeInfo::PyValue))
        })
        .unwrap_or(false)
}

/// Resolve a call's arguments against the callee's signature, in Python's
/// order: positionals fill left to right, keywords map by name, missing
/// parameters take their default values, and every mismatch Python would
/// raise a TypeError for is a conversion-time error.
///
/// Returned arguments are in parameter order, as Rust needs; when keyword
/// arguments make that differ from Python's source evaluation order, the
/// prelude binds each argument to a temp in source order first (CPython
/// evaluates positionals left to right, then keywords left to right).
struct MappedArguments {
    prelude: TokenStream,
    args: Vec<TokenStream>,
}

/// Defaults are evaluated ONCE at def time by CPython; rython inlines them
/// at each call site. A scalar constant (literal, None/True/False, or a
/// tuple of those) is safe to inline; anything else is a loud conversion
/// error rather than a silent divergence — a mutable container would also
/// be SHARED across calls by CPython, which owned Rust values cannot
/// express (issue #80).
fn check_default_constant(
    default: &ExprType,
    fname: &str,
    param: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let scalar = |e: &ExprType| -> bool {
        matches!(e, ExprType::Constant(_))
            || matches!(
                e,
                ExprType::Name(n) if matches!(n.id.as_str(), "None" | "True" | "False")
            )
            // A signed constant (`-1` — a common default like urllib3's
            // `output_buffer_limit: int = -1`).
            || matches!(
                e,
                ExprType::UnaryOp(u) if matches!(u.operand.as_ref(), ExprType::Constant(_))
            )
            // A constant BINARY expression (`2**16` — urllib3's
            // `stream(amt: int | None = 2**16)`, `8 * MB` — s3transfer's
            // TransferConfig): a compile-time scalar. Operands may be
            // constants OR module-level Name constants (`MB = 1024*1024`).
            || matches!(
                e,
                ExprType::BinOp(op)
                    if (matches!(op.left.as_ref(), ExprType::Constant(_))
                        || matches!(op.left.as_ref(), ExprType::Name(_)))
                        && (matches!(op.right.as_ref(), ExprType::Constant(_))
                            || matches!(op.right.as_ref(), ExprType::Name(_)))
            )
    };
    match default {
        ExprType::Constant(_) => Ok(()),
        // A signed constant (`-1`) — a scalar default.
        ExprType::UnaryOp(u) if matches!(u.operand.as_ref(), ExprType::Constant(_)) => Ok(()),
        // A constant binary expression (`2**16`, `8 * MB`).
        ExprType::BinOp(op)
            if (matches!(op.left.as_ref(), ExprType::Constant(_))
                || matches!(op.left.as_ref(), ExprType::Name(_)))
                && (matches!(op.right.as_ref(), ExprType::Constant(_))
                    || matches!(op.right.as_ref(), ExprType::Name(_))) =>
        {
            Ok(())
        }
        ExprType::UnaryOp(u) if matches!(u.operand.as_ref(), ExprType::Constant(_)) => Ok(()),
        // A CLASS-REFERENCE default (`executor_cls=concurrent.futures.
        // ThreadPoolExecutor` — s3transfer): a module-path attribute naming
        // a class — safe to re-evaluate (the class is a static).
        ExprType::Attribute(_) => Ok(()),
        // A NAME default (a module-level constant like urllib3's
        // `DEFAULT_ALLOWED_METHODS`) renders as a read of the static — safe
        // to re-evaluate at every call site. Python's call-by-reference
        // defaults are still a divergence only for the mutable-container
        // arms below (issue #80).
        ExprType::Name(_) => Ok(()),
        // A `field(default_factory=...)` marker (a @dataclass field default,
        // urllib3's EmscriptenRequest.headers): the FACTORY creates a fresh
        // container per call, which rython's inline-empty default matches
        // exactly — the shared-mutable-default divergence does NOT apply.
        // fill renders the typed empty container from the annotation.
        ExprType::Call(c)
            if c.keywords
                .iter()
                .any(|k| k.arg.as_deref() == Some("default_factory"))
                && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "field") =>
        {
            Ok(())
        }
        // A CLASS-CONSTRUCTION default (`highlighter=ReprHighlighter()` —
        // rich's Console, pip's vendored console): the default is a fresh
        // construction per call site, so the shared-mutable-default
        // divergence cannot arise — accepted (documented).
        ExprType::Call(_) => Ok(()),
        // A LAMBDA default (`replacefunc=lambda x: x` — pygments'
        // _replace_special): a callable as a default value — the
        // function-as-value divergence (#122). The default is dropped; the
        // parameter lowers as a called-param (its call sites drop as
        // no-ops).
        ExprType::Lambda(_) => Ok(()),
        ExprType::Tuple(t) if t.elts.iter().all(&scalar) => Ok(()),
        ExprType::Tuple(t)
            if t.elts
                .iter()
                .any(|e| matches!(e, ExprType::List(_) | ExprType::Dict(_) | ExprType::Set(_))) =>
        {
            Err(format!(
                "parameter `{param}` of `{fname}()` has a mutable default; CPython \
                 evaluates defaults once at def time and SHARES the single container \
                 across all calls, which rython's owned-value model cannot express. \
                 Pass the container explicitly at every call site instead"
            )
            .into())
        }
        ExprType::List(_) | ExprType::Dict(_) | ExprType::Set(_) => Err(format!(
            "parameter `{param}` of `{fname}()` has a mutable default; CPython \
             evaluates defaults once at def time and SHARES the single container \
             across all calls, which rython's owned-value model cannot express. \
             Pass the container explicitly at every call site instead"
        )
        .into()),
        _ => Err(format!(
            "parameter `{param}` of `{fname}()` has a non-constant default; CPython \
             evaluates defaults once at def time, but rython would re-evaluate this \
             expression at every call site. Use a constant default instead (issue #80)"
        )
        .into()),
    }
}

/// Whether a name is a user-defined symbol (assignment, function, class,
/// or rust.bind) rather than an imported stdlib module. Module
/// intercepts (`re.search`, `csv.reader`, ...) and module-path lowering
/// (`sys.argv`, `np.dot`) must defer to user definitions: `re = my_thing;
/// re.search(...)` calls the user's object, not the re module (issue #80).
/// Imports (plain or from-import) are module-like by construction — and so
/// is an aliased import (`import numpy as np` binds `np` as a module
/// alias, not a user value; a later `np = ...` reassignment replaces the
/// alias in the symbol table, which still shadows correctly — Devin review
/// on #103).
/// Whether an attribute-call RECEIVER is a string-like VALUE (a str
/// literal, a str-typed local, or a StrOrBytes/PyValue boxed union): the
/// str.encode/decode builtin dispatch applies only to these. A module
/// receiver (`idna.encode(name)` — a module function) falls through to the
/// generic path.
pub(crate) fn receiver_is_str_like(
    receiver: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match receiver {
        ExprType::Constant(c) => matches!(&c.0, Some(litrs::Literal::String(_))),
        ExprType::Name(n) => {
            options
                .local_types
                .get(&n.id)
                .is_some_and(|t| {
                    let s = t.to_string();
                    s.contains("String") || s.contains("str")
                })
                || options
                    .name_types
                    .get(&n.id)
                    .is_some_and(|t| {
                        matches!(
                            t,
                            crate::TypeInfo::StrOrBytes
                                | crate::TypeInfo::String
                                | crate::TypeInfo::StrRef
                                | crate::TypeInfo::PyValue
                        )
                    })
                || options
                    .narrowed_names
                    .get(&n.id)
                    .is_some_and(|t| {
                        matches!(
                            t,
                            crate::TypeInfo::StrOrBytes
                                | crate::TypeInfo::String
                                | crate::TypeInfo::StrRef
                        )
                    })
                || crate::module_name_shadowed(&n.id, symbols)
        }
        _ => false,
    }
}

pub(crate) fn module_name_shadowed(name: &str, symbols: &SymbolTableScopes) -> bool {
    matches!(
        symbols.get(name),
        Some(SymbolTableNode::Assign { .. })
            | Some(SymbolTableNode::FunctionDef(_))
            | Some(SymbolTableNode::ClassDef(_))
            | Some(SymbolTableNode::RustBinding(_))
    )
}

/// `threading.Thread(target=f, args=(...), daemon=...)`: callables are not
/// values in rython, so the constructor's target is resolved statically (a
/// plain function name) and the thread body is synthesized as a closure at
/// conversion time — the same model functools.partial uses. Name-elements
/// of args are CLONED into the closure: shared-identity runtime objects
/// (locks, events, sockets) keep sharing through the clone, containers
/// copy (rython's value-semantics ledger divergence, exactly as ordinary
/// call arguments behave). Returns Ok(None) when the call is not a
/// threading.Thread construction.
fn lower_threading_thread(
    call: &Call,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    let is_thread_name =
        |name: &str| crate::ThreadingType::from_name(name) == Some(crate::ThreadingType::Thread);
    let is_thread = match call.func.as_ref() {
        ExprType::Attribute(attr) => {
            is_thread_name(&attr.attr)
                && matches!(attr.value.as_ref(), ExprType::Name(n)
                    if crate::StdModule::from_name(&n.id)
                        == Some(crate::StdModule::Threading))
                && !module_name_shadowed(crate::StdModule::Threading.name(), symbols)
        }
        ExprType::Name(n) => {
            is_thread_name(&n.id)
                && matches!(
                    symbols.get("Thread"),
                    Some(SymbolTableNode::ImportFrom(i))
                        if crate::StdModule::from_name(&i.module)
                            == Some(crate::StdModule::Threading)
                )
        }
        _ => false,
    };
    if !is_thread {
        return Ok(None);
    }
    if !call.args.is_empty() {
        return Err(
            "threading.Thread(...): positional arguments are not supported yet; pass \
             target= and args= as keywords"
                .to_string()
                .into(),
        );
    }
    let mut target: Option<String> = None;
    let mut thread_args: Vec<ExprType> = Vec::new();
    let mut daemon = false;
    for kw in &call.keywords {
        match kw.arg.as_deref() {
            Some("target") => match &kw.value {
                ExprType::Name(n) => target = Some(n.id.clone()),
                _ => {
                    return Err(
                        "threading.Thread(target=...): the target must be a plain \
                         function name — callables are not runtime values in rython"
                            .to_string()
                            .into(),
                    );
                }
            },
            Some("args") => match &kw.value {
                ExprType::Tuple(t) => thread_args = t.elts.clone(),
                ExprType::List(l) => thread_args = l.clone(),
                _ => {
                    return Err(
                        "threading.Thread(args=...): the arguments must be a tuple or \
                         list literal, so they can be bound at conversion time"
                            .to_string()
                            .into(),
                    );
                }
            },
            Some("daemon") => match &kw.value {
                // The parser represents True/False as bool Constants; the
                // Name spelling covers synthesized/re-entered ASTs.
                ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::Bool(b)) if b.value()) =>
                {
                    daemon = true
                }
                ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::Bool(b)) if !b.value()) =>
                {
                    daemon = false
                }
                ExprType::Name(n) if n.id == "True" => daemon = true,
                ExprType::Name(n) if n.id == "False" => daemon = false,
                _ => {
                    return Err(
                        "threading.Thread(daemon=...): only the literal True/False is \
                         supported"
                            .to_string()
                            .into(),
                    );
                }
            },
            other => {
                return Err(format!(
                    "threading.Thread keyword `{}` is not supported yet (supported: \
                     target=, args=, daemon=); rython refuses to silently ignore it",
                    other.unwrap_or("**kwargs")
                )
                .into());
            }
        }
    }
    let Some(target) = target else {
        return Err(
            "threading.Thread(...) requires target= (a thread with no target does \
             nothing)"
                .to_string()
                .into(),
        );
    };
    // The synthesized body call lowers through the normal machinery, so
    // argument marshalling and exception propagation match a direct call.
    let inner = Call {
        func: Box::new(ExprType::Name(crate::ast::tree::name::Name {
            id: target.clone(),
        })),
        args: thread_args.clone(),
        keywords: Vec::new(),
    };
    let body = inner.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
    // Clone captured names OUTSIDE the move closure, so the caller's
    // bindings stay usable after start().
    let mut clones = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for arg in &thread_args {
        if let ExprType::Name(n) = arg {
            if !matches!(n.id.as_str(), "True" | "False" | "None") && seen.insert(n.id.clone()) {
                let ident = crate::safe_ident(&n.id);
                clones.push(quote!(let #ident = (#ident).clone();));
            }
        }
    }
    let target_name = target.as_str();
    Ok(Some(quote! {
        threading::Thread::new(#target_name, #daemon, {
            #(#clones)*
            move || {
                let __rython_thread_body = || -> Result<(), PyException> {
                    let _ = #body;
                    Ok(())
                };
                if let Err(e) = __rython_thread_body() {
                    threading::report_thread_exception(&e);
                }
            }
        })
    }))
}

/// The name at the root of a dotted expression chain (`os` in `os.path`,
/// A ROOT-typed value's `isinstance` against ONE class target, by the
/// hierarchy registry (the single authority every target form — a class
/// name, an element of a tuple, `type(self)` — consults): the root itself
/// or an ancestor is true, a class of the subtree is the runtime variant
/// test on the sum type, anything else is false.
pub(crate) fn root_isinstance_test(
    root: &str,
    target: &str,
    arg: &TokenStream,
    symbols: &SymbolTableScopes,
) -> TokenStream {
    // The root itself or an ANCESTOR: the crate-wide registry answers
    // (an ancestor of a root is a root whose subtree holds it — an
    // imported root, or one whose base is imported, is not in this
    // module's symbols; Devin review on #319), with the class tree
    // for a same-module ancestor that is no root.
    if target == root
        || crate::ast::tree::hierarchy::in_subtree_by_name(root, target)
        || crate::ast::tree::class_def::ClassDef::class_extends(root, target, symbols)
    {
        return quote!(true);
    }
    if crate::ast::tree::hierarchy::in_subtree_by_name(target, root) {
        let is_fn = format_ident!("__rython_is_{}", target);
        return quote!((#arg).#is_fn());
    }
    quote!(false)
}

/// `np` in `np.linalg.inv`), for module-vs-value resolution.
pub(crate) fn root_name(expr: &ExprType) -> Option<&str> {
    match expr {
        ExprType::Name(n) => Some(&n.id),
        ExprType::Attribute(a) => root_name(&a.value),
        _ => None,
    }
}

/// The CRATE module path a receiver name resolves to (`http2_probe` from
/// `from .http2 import probe as http2_probe` → ["urllib3", "http2",
/// "probe"]), for the module-member-call drop: a call through a member
/// that is not a module-level function/class cannot render. None when the
/// name is not a sibling submodule import.
pub(crate) fn module_path_of(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<Vec<String>> {
    let ExprType::Name(n) = expr else { return None };
    if crate::module_name_shadowed(&n.id, symbols) {
        return None;
    }
    // Follow alias chains (`from .http2 import probe as http2_probe`
    // registers "http2_probe" as Alias("probe")).
    let mut current = n.id.clone();
    for _ in 0..16 {
        match symbols.get(&current) {
            Some(crate::SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(crate::SymbolTableNode::ImportFrom(ifm)) => {
                let mut path = ifm.resolved_module_path(options);
                // The canonical (unaliased) name is the actual module —
                // relative (`from .http2 import probe`) or absolute
                // (`from urllib3.contrib import pyopenssl`).
                path.push(
                    ifm.names
                        .iter()
                        .find(|a| a.asname.as_deref() == Some(&current))
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| current.clone()),
                );
                return Some(path);
            }
            _ => return None,
        }
    }
    None
}

/// The CRATE module path a DOTTED CHAIN resolves to (`util.ssl_` from
/// `from .. import util` in urllib3's pyopenssl → ["urllib3", "util",
/// "ssl_"]), for the module-member-read drop: reading a member the
/// generated module never defines lowers to the boxed None. The chain's
/// ROOT name resolves through its import; each following segment that is
/// itself a module of the crate extends the path. None when the chain is
/// not a sibling-module path.
pub(crate) fn module_path_of_chain(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<Vec<String>> {
    // Collect the dotted parts root-first: `util.ssl_` → ["util", "ssl_"].
    let mut parts = Vec::new();
    let mut cur = expr;
    loop {
        match cur {
            ExprType::Attribute(a) => {
                parts.push(a.attr.clone());
                cur = &a.value;
            }
            ExprType::Name(n) => {
                parts.push(n.id.clone());
                break;
            }
            _ => return None,
        }
    }
    parts.reverse();
    // Resolve the ROOT name through its import, then extend with every
    // segment that is itself a module of the crate.
    let root = ExprType::Name(crate::ast::tree::name::Name {
        id: parts.remove(0),
    });
    let mut path = module_path_of(&root, symbols, options)?;
    for part in parts {
        let mut sub = path.clone();
        sub.push(part);
        if options.module_defs.contains_key(&sub) {
            path = sub;
        } else {
            break;
        }
    }
    Some(path)
}

/// The MODULE path a dotted chain names (`botocore.httpsession` →
/// ["botocore", "httpsession"]), for in-crate module member resolution.
pub(crate) fn dotted_module_path(expr: &ExprType) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut cur = expr;
    loop {
        match cur {
            ExprType::Attribute(a) => {
                parts.push(a.attr.clone());
                cur = &a.value;
            }
            ExprType::Name(n) => {
                parts.push(n.id.clone());
                break;
            }
            _ => return None,
        }
    }
    parts.reverse();
    Some(parts)
}

/// Round 84/86 (the generics directive): the argument-side expected-type
/// fallback for an annotation the syntax-only mapping cannot see. A CLASS
/// name (`conn: BaseHTTPConnection` — a TYPE_CHECKING Protocol stub,
/// round 84) or a module-level ALIAS resolving to the boxed PyValue
/// (`_TYPE_TIMEOUT = Union[float, str, None]` — urllib3's
/// `resolve_default_timeout(timeout)` into a `_TYPE_TIMEOUT` param,
/// round 86) resolves through the symbols-aware authority — the same
/// resolution the parameter's Rust type used — so an OPTION-typed
/// argument coerces (`Option<f64> → PyValue` via the Some/None match,
/// Python's None passing through as the boxed None). Gated on the
/// OPTION-typed argument: a plain class-instance argument must keep the
/// pre-existing raw render (a loud rustc mismatch), not box through
/// `PyValue::from` (no such From for a class — an E0277 shift at the
/// `err: _TYPE_TIMEOUT` sites). Skips an annotation the syntax-only
/// mapping ALREADY answers (`str` → call_arg_expected_type deliberately
/// returns None so literals pass as &str into the `impl Into<String>`
/// parameter).
fn arg_expected_fallback(
    param: &crate::Parameter,
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Option<crate::TypeInfo> {
    let ann = param.annotation.as_deref()?;
    if crate::annotation_type_info(ann).is_some() {
        return None;
    }
    // A NARROWED name's read already unwraps to the inner value
    // (`(conn).clone().unwrap()` — the `if conn and
    // is_connection_dropped(conn)` chain, round 81): the tokens are the
    // member, not the Option — wrapping them in the Option→PyValue match
    // would match on a non-Option (E0308). Same principle as the
    // render_typed narrowed-names handling.
    if matches!(expr, ExprType::Name(n)
        if options.narrowed_names.contains_key(&n.id))
    {
        return None;
    }
    // The argument must be an OPTION-typed value: an inferred Option, a
    // None-then-assigned NAME (`conn` — urllib3's urlopen, whose
    // infer_type answers the boxed PyValue while the BINDING is
    // Option<PyValue>), or a CALL whose callee returns an Option
    // (`Timeout.resolve_default_timeout(timeout)` — a classmethod whose
    // `-> float | None` return infer_type cannot see; `self.proxy()` — a
    // `Proxy | None` property accessor — round 86).
    let arg_is_option = matches!(
        crate::infer_type(Some(&ctx), expr, options, symbols),
        crate::TypeInfo::Option(_)
    ) || matches!(expr, ExprType::Name(n)
        if options.optional_names.contains(&n.id))
        || matches!(expr, ExprType::Call(c)
            if call_returns_option(c, ctx, symbols, options));
    if !arg_is_option {
        return None;
    }
    let resolved = crate::resolve_alias_typeinfo(ann, symbols, options)?;
    // For a BOXED-PyValue slot, the Option's INNER must be boxable
    // (`PyValue::from(inner)` must exist — int/str/bytes/tuple/Vec/
    // PyValue ...): an `Option<Class>` argument (`Option<Retry>` —
    // urllib3's `retries` union) cannot box (no `From<Class>`), so the
    // fallback must not fire — the raw mismatch stays loud instead of
    // shifting to an E0277. A CONCRETE slot (a class-annotated parameter
    // — charset's `fallback_specified: CharsetMatch`) coerces any inner
    // via the Option→concrete match-unwrap (the round-83 loud
    // unhandled-None panic).
    if matches!(resolved, crate::TypeInfo::PyValue) {
        let inner_boxable = match crate::infer_type(Some(&ctx), expr, options, symbols) {
            crate::TypeInfo::Option(inner) => {
                crate::ast::tree::type_ctx::is_boxable_value_type(&inner)
            }
            _ => true,
        };
        if !inner_boxable {
            return None;
        }
    }
    Some(resolved)
}

/// Whether a CALL's callee resolves to an Option-returning function:
/// a NAME callee (call_return_typeinfo), or a CLASS-QUALIFIED callee
/// (`Timeout.resolve_default_timeout(...)` — a classmethod/staticmethod
/// on a same-module class whose `-> T | None` return annotation
/// infer_type cannot see — round 86).
fn call_returns_option(
    call: &crate::Call,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> bool {
    let return_is_option = |m: crate::FunctionDef| -> bool {
        m.returns.as_deref().is_some_and(|r| {
            matches!(
                crate::resolve_alias_typeinfo(r, symbols, options),
                Some(crate::TypeInfo::Option(_))
            )
        })
    };
    match call.func.as_ref() {
        ExprType::Name(_) => matches!(
            crate::call_return_typeinfo(call, Some(symbols), Some(options)),
            Some(crate::TypeInfo::Option(_))
        ),
        ExprType::Attribute(attr) => {
            let ExprType::Name(recv) = attr.value.as_ref() else {
                return false;
            };
            // `self.proxy()` — a `Proxy | None` property accessor on the
            // ENCLOSING class (urllib3's
            // `connection_requires_http_tunnel(self.proxy(), ...)`).
            if recv.id == "self"
                && let Some((class, class_symbols)) =
                    receiver_class(&attr.value, ctx, symbols, options)
            {
                return class
                    .method_on_mro_with_options(&attr.attr, &class_symbols, options)
                    .is_some_and(return_is_option);
            }
            // Same-module class, or an IMPORTED one (`Timeout` from
            // .util.timeout — urllib3's `Timeout::resolve_default_timeout`
            // classmethod) resolved through its defining module.
            let class = match symbols.get(&recv.id) {
                Some(crate::SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                _ => crate::resolve_construction_class(&recv.id, symbols, options)
                    .map(|(c, _)| c),
            };
            let Some(class) = class else {
                return false;
            };
            class
                .method_on_mro_with_options(&attr.attr, symbols, options)
                .is_some_and(return_is_option)
        }
        _ => false,
    }
}

fn map_call_arguments(
    func: &crate::FunctionDef,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<MappedArguments, Box<dyn std::error::Error>> {
    map_call_arguments_inner(func, args, keywords, ctx, options, symbols, None, None)
}

/// [`map_call_arguments`] with the CONSTRUCTED class's name (when the call
/// is a class construction `Retry(...)` / `Retry::new(...)`): dropped
/// defaults that reference CLASS-BODY constants (`DEFAULT_ALLOWED_METHODS`
/// — urllib3's Retry) resolve through the class, even at module level
/// where there is no enclosing-class context.
///
/// `default_symbols` is the DEFINING module's symbol scope for a call on an
/// IMPORTED class's method/constructor (`p.prepare(...)` — requests'
/// sessions.py, where p is a models.py PreparedRequest): dropped-DEFAULT
/// constants resolve there, while the ARGUMENT expressions render in the
/// caller's `symbols` scope (an argument can itself be a call into the
/// caller's module — `merge_setting(request.headers, self.headers,
/// dict_class=CaseInsensitiveDict)` — which must resolve its callee in the
/// caller's scope).
fn map_call_arguments_inner(
    func: &crate::FunctionDef,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    constructed_class: Option<&str>,
    default_symbols: Option<&SymbolTableScopes>,
) -> Result<MappedArguments, Box<dyn std::error::Error>> {
    let fname = &func.name;
    // Optional-annotated parameters take Option values: the Option-slot
    // lowering wraps plain arguments in Some, passes None and
    // already-Option values (dict.get, another optional name, an
    // Optional-returning call) through unwrapped, and handles conditional
    // arms independently.
    let fill = |param: &crate::Parameter,
                expr: &ExprType|
     -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A `field(default_factory=...)` marker default renders as the
        // typed EMPTY container from the parameter's annotation: `dict[str,
        // str]` → `PyDict::<String, String>::from([])`, `list[T]` → the
        // empty Vec, `set[T]` → the empty set. The factory semantics (fresh
        // container per call) match rython's inline-empty exactly.
        if crate::is_field_factory_call(expr) {
            if let Some(ann) = param.annotation.as_deref() {
                if let Some(t) = crate::annotation_type_info(ann) {
                    match t {
                        crate::TypeInfo::Dict(k, v)
                            if !matches!(*k, crate::TypeInfo::PyObject)
                                && !matches!(*v, crate::TypeInfo::PyObject) =>
                        {
                            let k = k.to_rust_type();
                            let v = v.to_rust_type();
                            return Ok(quote!(PyDict::<#k, #v>::from([])));
                        }
                        crate::TypeInfo::Vec(inner)
                            if !matches!(*inner, crate::TypeInfo::PyObject) =>
                        {
                            let t = inner.to_rust_type();
                            return Ok(quote!(Vec::<#t>::new()));
                        }
                        _ => {}
                    }
                }
            }
            return Ok(quote!(Default::default()));
        }
        let optional = param
            .annotation
            .as_deref()
            .is_some_and(crate::is_optional_annotation);
        // A cross-module constant name used as a dropped DEFAULT
        // (`HTTPAdapter()` — sessions.py, whose __init__ defaults reference
        // adapters.py's `DEFAULT_POOLSIZE = 10`): the call site does not
        // import the name, so inline the constant's VALUE. Resolve in the
        // DEFINING module's scope when the callee is an imported class's
        // method (the caller may not even bind the constant).
        if let ExprType::Name(n) = expr
            && let Some(v) = resolve_constant_name(
                &n.id,
                default_symbols.unwrap_or(symbols),
                &options,
            )
        {
            return Ok(v);
        }
        // A CLASS-BODY computed constant used as a dropped DEFAULT inside
        // the defining class (`Retry::new(...)` from from_int, whose
        // __init__ defaults reference `DEFAULT_ALLOWED_METHODS =
        // frozenset([...])` — a class-level LazyLock static, not a module
        // name): deref-clone the class static. The name is not in the
        // symbol table (class bodies register only the class), so resolve
        // through the class being CONSTRUCTED first (a `Retry(...)` call
        // from ANOTHER class's method — requests' adapters.py — must
        // resolve Retry's constants, not the enclosing HTTPAdapter's) —
        // or the ENCLOSING class's ClassDef for non-construction calls
        // (`map_call_arguments_inner` with no constructed_class).
        if let ExprType::Name(n) = expr
            && let Some(class_name) = constructed_class
                .or(ctx.enclosing_class_name())
            && let Some(class) = crate::resolve_class_referenced(
                &class_name,
                default_symbols.unwrap_or(symbols),
                &options,
            )
            && class.body.iter().any(|bs| {
                matches!(&bs.statement, crate::StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Name(t) if t.id == n.id)
                        && crate::ast::tree::module::const_static_type(&a.value).is_some())
            })
        {
            let ident = crate::safe_ident(&n.id);
            let class_ident = crate::safe_ident(&class_name);
            return Ok(quote!((*#class_ident::#ident).clone()));
        }
        if let ExprType::Name(n) = expr
            && let Some(class_name) = constructed_class
                .or(ctx.enclosing_class_name())
            && let Some(class) = crate::resolve_class_referenced(
                &class_name,
                default_symbols.unwrap_or(symbols),
                &options,
            )
            && class.body.iter().any(|bs| {
                matches!(&bs.statement, crate::StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Name(t) if t.id == n.id)
                        && crate::ast::tree::module::const_static_type(&a.value).is_none()
                        && crate::class_body_computed_constant(&a.value))
            })
        {
            // The computed constant's associated ACCESSOR (the LazyLock
            // static itself lives at module level in the DEFINING module —
            // issue #137).
            let ident = crate::safe_ident(&n.id);
            let class_ident = crate::safe_ident(&class_name);
            return Ok(quote!(#class_ident::#ident()));
        }
        // A CALLABLE parameter (`dict_class: type = OrderedDict` —
        // requests' sessions): a CLASS argument lowers to its NAME STRING
        // (the class object's runtime value — the class-as-value model,
        // round 33); any other callable (a function) cannot be a runtime
        // value and lowers to the boxed None (the callable-as-value
        // divergence).
        if param
            .annotation
            .as_deref()
            .is_some_and(crate::ast::tree::arguments::is_type_annotation)
        {            if crate::is_class_value_expr(expr, symbols) {
                if let ExprType::Name(n) = expr {
                    let name = n.id.clone();
                    return Ok(quote!(#name.to_string()));
                }
            }
            options.definition_warnings.borrow_mut().push(format!(
                "callable argument for `{}` (a `type`-annotated parameter) lowers to \
                 the boxed None (callables cannot be runtime values in rython)",
                param.arg
            ));
            return Ok(quote!(stdpython::PyValue::None_));
        }
        // An UNANNOTATED parameter with a CLASS default (`dict_class =
        // OrderedDict` — requests' merge_setting, where dict_class has no
        // annotation): the inlined default is the class object, whose
        // rython value is its NAME STRING (round 33). Without this, the
        // default renders as a bare struct name — E0423 (the struct lives
        // in the type namespace). Only a CLASS: a stdpython-module item
        // must be a class per the registry (OrderedDict), and a local
        // name must resolve to a ClassDef — function defaults stay the
        // callable-as-value drop (a bare function name is E0425, loud).
        if param.annotation.is_none()
            && let ExprType::Name(n) = expr
            && match symbols.get(&n.id) {
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let root = i.module.split('.').next().unwrap_or("");
                    crate::ast::tree::import::stdpython_module_class(root, &n.id)
                        || crate::ast::tree::import::stdpython_module_class(
                            root,
                            &i.names
                                .iter()
                                .find(|a| a.asname.as_deref() == Some(&n.id))
                                .map(|a| a.name.as_str())
                                .unwrap_or(&n.id),
                        )
                }
                Some(SymbolTableNode::ClassDef(_))
                | Some(SymbolTableNode::Alias(_)) => crate::is_class_value_expr(expr, symbols),
                _ => false,
            }
        {
            let name = n.id.clone();
            return Ok(quote!(#name.to_string()));
        }
        if optional {
            // An `X | None` parameter whose X has no Rust type
            // (`headers: ValidHTTPHeaderSource | None` — urllib3's
            // HTTPHeaderDict) lowers the PARAM to the plain boxed PyValue,
            // not an Option — wrapping the argument in Some would
            // mismatch. The None default boxes as PyValue::None_, and a
            // present argument coerces to the boxed value (a PyValue
            // passes through; a class instance cannot be boxed and stays a
            // loud rustc error).
            // The `X | None` UNION form only (not `Optional[T]` — that
            // always lowers to a real Option), and only when X resolves to
            // the boxed PyValue.
            if param.annotation.as_deref().is_some_and(|ann| {
                matches!(
                    ann,
                    ExprType::BinOp(op)
                        if matches!(op.op, crate::BinOps::BitOr)
                            && (crate::is_none_expr(&op.left)
                                || crate::is_none_expr(&op.right))
                ) && matches!(
                    // The annotation's alias lives in the CALLEE's module
                    // (`headers: ValidHTTPHeaderSource | None` where
                    // ValidHTTPHeaderSource is defined in _collections.py —
                    // a construction `HTTPHeaderDict(headers)` from
                    // _request_methods.py): resolve through the
                    // defining module's symbols (default_symbols), not the
                    // caller's scope where the alias is absent (round 94 —
                    // the mismatch left the OPTION-typed argument
                    // uncoerced, raw against the boxed param).
                    crate::resolve_alias_typeinfo(
                        ann,
                        default_symbols.unwrap_or(symbols),
                        &options,
                    ),
                    Some(crate::TypeInfo::PyValue)
                )
            }) {
                if crate::is_none_expr(expr) {
                    return Ok(quote!(stdpython::PyValue::None_));
                }
                // A present argument: an OPTION-typed value (a None-
                // stored local like `conn` — urllib3's urlopen, whose
                // `_put_conn(conn: BaseHTTPConnection | None)` param
                // boxes to PyValue) unwraps to the boxed value with
                // Python's None passing through as PyValue::None_ — the
                // boxed slot IS the None-able value (round 84). A plain
                // PyValue passes through (the param IS the boxed value),
                // while a class instance stays a loud mismatch (the boxed
                // value cannot hold one) — the coercion would only move
                // the error kind, not fix it.
                let arg_is_option = matches!(expr, ExprType::Name(n)
                    if options.optional_names.contains(&n.id))
                    || matches!(
                        crate::infer_type(Some(&ctx), expr, &options, &symbols),
                        crate::TypeInfo::Option(_)
                    );
                return crate::render_typed_reused(
                    expr,
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                    if arg_is_option {
                        Some(crate::TypeInfo::PyValue)
                    } else {
                        None
                    },
                );
            }
            // Round 81 (the generics directive): an `X | None`-annotated
            // parameter whose X is a CONCRETE member (`cert_reqs: int |
            // None` → `Option<i64>`) receives a BOXED argument
            // (`resolve_cert_reqs(...)` — the callee's return was boxed
            // by the None-mixing inference). The inner converts via the
            // reverse From<PyValue> impls: `Some((arg).into())` — loud
            // TypeError panic on a wrong member, never a silent
            // placeholder (rustc's own suggestion is exactly the
            // `.into()`). The bare-PyValue case wraps here so the wrap
            // and the conversion happen in one pass (lower_optional_value
            // would Some-wrap the raw PyValue, leaving `Some(PyValue)`
            // against `Option<i64>`).
            let inner_expected = param.annotation.as_deref().and_then(|ann| {
                match crate::call_arg_expected_type(ann) {
                    Some(crate::TypeInfo::Option(inner))
                        if crate::ast::tree::type_ctx::is_boxable_value_type(&inner) =>
                    {
                        Some(inner)
                    }
                    _ => None,
                }
            });
            let arg_infers = crate::ast::tree::type_ctx::infer_type(None, expr, &options, &symbols);
            // A BOXED argument: the inferrer's PyValue, or PyObject
            // ("no answer") that the boxed-receiver read-side drop
            // recognizes — an ATTRIBUTE read on a boxed receiver
            // (`conn.host` where conn is `PyValue` — urllib3's
            // _url_from_connection) lowers to `PyValue::None_`; its
            // argument is boxed, so it converts the same way. A CALL into
            // a same-module function whose RESOLVED return is the boxed
            // PyValue (`resolve_cert_reqs(...)` — a generic callee the
            // inferrer cannot see through) is boxed the same way.
            let arg_is_boxed = matches!(
                arg_infers,
                crate::TypeInfo::PyValue | crate::TypeInfo::PyValueMember(_)
            ) || (matches!(arg_infers, crate::TypeInfo::PyObject)
                && (matches!(expr, ExprType::Call(c)
                    // A call into a same-module function whose RESOLVED
                    // return is the boxed PyValue (`resolve_cert_reqs(...)`
                    // — a generic callee the inferrer cannot see through).
                    // The resolved return is the authority: the codegen
                    // renders `Result<PyValue>` from it, so the argument is
                    // boxed and converts like one.
                    if matches!(
                        crate::ast::tree::type_ctx::call_return_typeinfo(
                            c, Some(&symbols), Some(&options),
                        ),
                        Some(crate::TypeInfo::PyValue | crate::TypeInfo::PyObject)
                    ) || {
                        match c.func.as_ref() {
                            ExprType::Name(callee) => match symbols.get(&callee.id) {
                                Some(crate::SymbolTableNode::FunctionDef(f)) => f
                                .resolved_return_type(&symbols, &options)
                                .is_some_and(|t| {
                                    let s = t.to_string();
                                    s.contains("PyValue") || s.contains("PyObject")
                                }),
                                _ => false,
                            },
                            _ => false,
                        }
                    })
                // A NAME receiver with POSITIVE boxing evidence (`conn.host`
                // where conn is a boxed PyValue param — urllib3's
                // _url_from_connection): the read-side drop fires, so the
                // read lowers to `PyValue::None_` and the argument is boxed.
                // The SAME positive-evidence predicate the read-side drop
                // uses (`receiver_is_boxed_positively` — PyValue, never the
                // PyObject "no answer", which is what round 24's regression
                // keyed on) — so a FIELD-CHAIN receiver
                // (`httplib_response.reason` — a REAL `Option<String>`
                // field read resolving through the base) stays uncoerced:
                // its value is the typed field, and coercing would corrupt
                // it.
                || matches!(expr, ExprType::Attribute(a)
                    if matches!(a.value.as_ref(), ExprType::Name(_))
                        && crate::ast::tree::attribute::receiver_is_boxed_positively(
                            &a.value, &symbols, &options,
                        ))));
            if let (Some(inner), true) = (&inner_expected, arg_is_boxed) {
                let tokens = expr
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                // An unboxable inner (a class slot — no From<PyValue>) is
                // NOT coercible: the raw mismatch stays loud (round 99 —
                // the class-annotation slot widened the slot set, so this
                // is reachable; a panic here would crash the conversion).
                let Some(inner_ty) = crate::ast::tree::type_ctx::coerce_tokens(
                    quote!(__rython_v),
                    &crate::TypeInfo::PyValue,
                    inner,
                ) else {
                    return Ok(tokens);
                };
                return Ok(quote!(Some({
                    let __rython_v = #tokens;
                    #inner_ty
                })));
            }
            // An OPTION-wrapped boxed argument (`Option<PyValue> →
            // Option<i64>` — a `.get(key, None)` call into a concrete
            // optional slot): map the conversion over the Option — None
            // passes through, Some converts loudly.
            let optional_tokens =
                crate::lower_optional_value(expr, ctx.clone(), options.clone(), symbols.clone())?;
            if let (Some(inner), crate::TypeInfo::Option(inner_arg)) =
                (&inner_expected, &arg_infers)
            {
                if matches!(inner_arg.as_ref(), crate::TypeInfo::PyValue) {
                    if let Some(coerced) = crate::ast::tree::type_ctx::coerce_tokens(
                        optional_tokens.clone(),
                        &crate::TypeInfo::Option(Box::new(crate::TypeInfo::PyValue)),
                        &crate::TypeInfo::Option(inner.clone()),
                    ) {
                        return Ok(coerced);
                    }
                }
            }
            Ok(optional_tokens)
        } else {
            // Type-aware lowering: coerce the argument to the parameter's
            // annotated type (usize → i64, i64 → f64) and clone non-Copy
            // names that are reused later (Python shares by reference;
            // Rust moves).
            let expected = param
                .annotation
                .as_deref()
                .and_then(crate::call_arg_expected_type)
                // A bare CLASS-ANNOTATED parameter (`item: Item` —
                // annotation_type_info answers None for class names):
                // the slot is the class, so a DERIVED argument coerces
                // through the generated `From<Derived> for Base` (round
                // 99 — the idiom corpus's `add(perishable)`).
                .or_else(|| {
                    let ann = param.annotation.as_deref()?;
                    let crate::ExprType::Name(cn) = ann else {
                        return None;
                    };
                    let is_class = matches!(
                        symbols.get(&cn.id),
                        Some(crate::SymbolTableNode::ClassDef(_))
                    ) || crate::resolve_class_referenced(&cn.id, symbols, options)
                        .is_some();
                    if is_class {
                        Some(crate::TypeInfo::Class(cn.id.clone()))
                    } else {
                        None
                    }
                })
                .or_else(|| arg_expected_fallback(param, expr, &ctx, symbols, &options))
                .or_else(|| {
                    // A None-defaulted unannotated parameter whose VALUE is
                    // used in the callee (`retryable_exceptions=None`
                    // stored into a field — botocore's
                    // MaxAttemptsDecorator, round 33): the callee types it
                    // as the boxed PyValue, so call-site arguments —
                    // including the dropped None default, which boxes to
                    // PyValue::None_ — coerce to the boxed value.
                    if param.annotation.is_none()
                        && crate::param_has_none_default(param, func)
                        && crate::name_read_as_value(&param.arg, &func.body)
                    {
                        Some(crate::TypeInfo::PyValue)
                    } else {
                        None
                    }
                });
            crate::render_typed_reused(
                expr,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                expected,
            )
        }
    };

    let pos_params: Vec<&crate::Parameter> = func
        .args
        .posonlyargs
        .iter()
        .chain(func.args.args.iter())
        .collect();
    let n = pos_params.len();
    // A *args callee (an exception `__init__(self, *args, **kwargs)`)
    // accepts extra positionals: they collect into the vararg slot.
    let vararg_param = func.args.vararg.as_ref();
    if args.len() > n && vararg_param.is_none() {
        return Err(format!(
            "{}() takes {} positional argument(s) but {} were given",
            fname,
            n,
            args.len()
        )
        .into());
    }

    // Rendered values in Python's source evaluation order (positionals,
    // then keywords), for the temp prelude when reordering is needed.
    let mut eval_order: Vec<TokenStream> = Vec::new();
    let total = n + func.args.kwonlyargs.len();
    // Which eval_order index each parameter slot was filled from; None
    // means the slot holds an inlined (constant) default.
    let mut slot_temp: Vec<Option<usize>> = vec![None; total];

    let mut slots: Vec<Option<TokenStream>> = vec![None; n];
    // Issue #120: extra positionals for a *args callee, boxed one by one
    // (PyValue-yielding values pass through), plus `*t` spreads forwarded
    // into the vector in source order. Each records its eval_order index
    // so the keyword-reorder path can reference the temps.
    let mut vararg_extras: Vec<(usize, TokenStream, bool)> = Vec::new(); // (temp, value, is_spread)
    // `*t` positional spreads and `**d` keyword spreads collected
    // separately: only the POSITIONAL spreads can fill missing params.
    let mut spreads: Vec<TokenStream> = Vec::new();
    let mut kw_spreads: Vec<TokenStream> = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        // A `*t` SPREAD argument. To a *args callee it FORWARDS into the
        // vararg vector (`g(*args)` passthrough — the elements box via
        // PyValue::from, an identity for an already-boxed Vec<PyValue>).
        // Otherwise (`build_netloc(*host_port)` — pip's package_finder)
        // the elements cannot be unpacked statically — the spread value
        // fills each missing positional parameter (the spread-argument
        // divergence).
        if let ExprType::Starred(st) = arg {
            let value = st
                .value
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            eval_order.push(value.clone());
            if vararg_param.is_some() && i >= n {
                vararg_extras.push((eval_order.len() - 1, value, true));
            } else {
                spreads.push(value);
            }
            continue;
        }
        if i >= n {
            // Extra positionals collect into the *args slot, boxed.
            let value = crate::render_typed(
                arg,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                Some(crate::TypeInfo::PyValue),
            )?;
            eval_order.push(value.clone());
            vararg_extras.push((eval_order.len() - 1, value, false));
            continue;
        }
        let value = fill(pos_params[i], arg)?;
        eval_order.push(value.clone());
        slot_temp[i] = Some(eval_order.len() - 1);
        slots[i] = Some(value);
    }

    let mut kwonly_slots: Vec<Option<TokenStream>> = vec![None; func.args.kwonlyargs.len()];
    // Issue #120: a **kwargs callee collects the extra keyword arguments
    // (and any `**d` spreads) into a boxed PyDict<String, PyValue>, passed
    // as the final argument.
    let kwarg_param = func.args.kwarg.as_ref();
    let mut extra_kwargs: Vec<(String, TokenStream)> = Vec::new();
    let mut extra_kwarg_temps: Vec<usize> = Vec::new();
    for kw in keywords {
        let Some(kw_name) = &kw.arg else {
            // `**d` spread: merges into the callee's **kwargs when it has
            // one; otherwise the dict passes as a trailing argument (the
            // extra-keywords divergence — `self.request(..., **kwargs)`
            // where request has no **kwargs, requests/sessions.py).
            let spread_expr = &kw.value;
            // A STATIC-KEY spread: the spread is a Name bound to a dict
            // LITERAL with string-literal keys — its entries bind to the
            // callee's named parameters BY KEY (`self._account_id_set_
            // without_credentials(**credentials_kwargs)` — boto3's
            // Session, where credentials_kwargs is a local dict literal
            // holding exactly the callee's keyword-only parameters).
            let mut bound_any = false;
            if let ExprType::Name(spread_name) = spread_expr
                && let Some(SymbolTableNode::Assign {
                    value: ExprType::Dict(d),
                    ..
                }) = symbols.get(&spread_name.id)
                && d.keys.iter().all(|k| {
                    matches!(
                        k,
                        Some(ExprType::Constant(c))
                            if matches!(&c.0, Some(litrs::Literal::String(_)))
                    )
                })
            {
                for (k, v) in d.keys.iter().zip(d.values.iter()) {
                    let Some(ExprType::Constant(c)) = k else { continue };
                    let Some(litrs::Literal::String(s)) = &c.0 else { continue };
                    let key = s.value().to_string();
                    if let Some(idx) = pos_params.iter().position(|p| &p.arg == &key) {
                        if slots[idx].is_some() {
                            return Err(format!(
                                "{}() got multiple values for argument `{}`",
                                fname, key
                            )
                            .into());
                        }
                        let value = fill(pos_params[idx], v)?;
                        eval_order.push(value.clone());
                        slot_temp[idx] = Some(eval_order.len() - 1);
                        slots[idx] = Some(value);
                        bound_any = true;
                    } else if let Some(kidx) =
                        func.args.kwonlyargs.iter().position(|p| &p.arg == &key)
                    {
                        if kwonly_slots[kidx].is_some() {
                            return Err(format!(
                                "{}() got multiple values for argument `{}`",
                                fname, key
                            )
                            .into());
                        }
                        let value = fill(&func.args.kwonlyargs[kidx], v)?;
                        eval_order.push(value.clone());
                        slot_temp[n + kidx] = Some(eval_order.len() - 1);
                        kwonly_slots[kidx] = Some(value);
                        bound_any = true;
                    } else if kwarg_param.is_some() {
                        let value = crate::render_typed(
                            v,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::PyValue),
                        )?;
                        eval_order.push(value.clone());
                        extra_kwarg_temps.push(eval_order.len() - 1);
                        extra_kwargs.push((key, value));
                        bound_any = true;
                    } else {
                        return Err(format!(
                            "{}() got an unexpected keyword argument `{}`",
                            fname, key
                        )
                        .into());
                    }
                }
                // The static spread's entries were bound above; do NOT
                // also merge the whole dict into **kwargs (that would
                // duplicate them).
                if bound_any {
                    continue;
                }
            }
            let value = kw
                .value
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            eval_order.push(value.clone());
            kw_spreads.push(value);
            continue;
        };
        if let Some(idx) = pos_params.iter().position(|p| &p.arg == kw_name) {
            let value = fill(pos_params[idx], &kw.value)?;
            if idx < func.args.posonlyargs.len() {
                return Err(format!(
                    "{}(): parameter `{}` is positional-only and cannot be passed by keyword",
                    fname, kw_name
                )
                .into());
            }
            if slots[idx].is_some() {
                return Err(
                    format!("{}() got multiple values for argument `{}`", fname, kw_name).into(),
                );
            }
            eval_order.push(value.clone());
            slot_temp[idx] = Some(eval_order.len() - 1);
            slots[idx] = Some(value);
        } else if let Some(idx) = func.args.kwonlyargs.iter().position(|p| &p.arg == kw_name) {
            let value = fill(&func.args.kwonlyargs[idx], &kw.value)?;
            if kwonly_slots[idx].is_some() {
                return Err(
                    format!("{}() got multiple values for argument `{}`", fname, kw_name).into(),
                );
            }
            eval_order.push(value.clone());
            slot_temp[n + idx] = Some(eval_order.len() - 1);
            kwonly_slots[idx] = Some(value);
        } else if let Some(_kp) = kwarg_param {
            // An extra keyword lands in **kwargs, boxed.
            let value = crate::render_typed(
                &kw.value,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                Some(crate::TypeInfo::PyValue),
            )?;
            eval_order.push(value.clone());
            extra_kwarg_temps.push(eval_order.len() - 1);
            extra_kwargs.push((kw_name.clone(), value));
        } else {
            return Err(format!(
                "{}() got an unexpected keyword argument `{}`",
                fname, kw_name
            )
            .into());
        }
    }

    // Defaults align with the tail of the positional parameter list.
    let default_offset = n - func.args.defaults.len();
    for i in 0..n {
        if slots[i].is_none() {
            if i >= default_offset {
                let default = &func.args.defaults[i - default_offset];
                check_default_constant(default, fname, &pos_params[i].arg)?;
                slots[i] = Some(fill(pos_params[i], default)?);
            } else if !spreads.is_empty() || kw_spreads.len() == 1 {
                // A `**d` SPREAD supplies the missing positional params
                // (`self._resolve_endpoint(**resolve_endpoint_kwargs)` —
                // botocore's args.py): the dict holds the args by name, but
                // rython cannot index it statically — the dict itself
                // passes as each missing argument (documented divergence).
                if spreads.len() == 1 {
                    slots[i] = Some(spreads[0].clone());
                } else if kw_spreads.len() == 1 {
                    slots[i] = Some(kw_spreads[0].clone());
                }
            } else {
                return Err(format!(
                    "{}() missing required argument `{}`",
                    fname, pos_params[i].arg
                )
                .into());
            }
        }
    }
    for (i, param) in func.args.kwonlyargs.iter().enumerate() {
        if kwonly_slots[i].is_none() {
            match func.args.kw_defaults.get(i).and_then(|d| d.as_ref()) {
                Some(default) => {
                    check_default_constant(default, fname, &param.arg)?;
                    kwonly_slots[i] = Some(fill(param, default)?);
                }
                None => {
                    return Err(format!(
                        "{}() missing required keyword-only argument `{}`",
                        fname, param.arg
                    )
                    .into());
                }
            }
        }
    }

    // Issue #120: the *args vector — plain extras as vec![..] elements,
    // spreads extended in source order (PyValue::from is the identity for
    // an already-boxed forwarded Vec<PyValue>). Empty when the call has
    // no extras: the callee's parameter still needs its vector.
    let build_vararg = |items: &[(TokenStream, bool)]| -> TokenStream {
        if items.iter().all(|(_, is_spread)| !is_spread) {
            let vals: Vec<&TokenStream> = items.iter().map(|(v, _)| v).collect();
            quote!(vec![#(#vals),*])
        } else {
            let mut stmts =
                quote!(let mut __rython_varargs: Vec<stdpython::PyValue> = Vec::new(););
            for (v, is_spread) in items {
                if *is_spread {
                    stmts.extend(quote!(__rython_varargs.extend(
                        (#v).into_iter().map(stdpython::PyValue::from)
                    );));
                } else {
                    stmts.extend(quote!(__rython_varargs.push(#v);));
                }
            }
            quote!({ #stmts __rython_varargs })
        }
    };

    // Parameter-ordered VALUES, kept in two lists so both emission paths
    // below assemble the signature order — [positional, *args, kwonly,
    // **kwargs] — without index arithmetic across the inserted vararg
    // element (Devin review on PR #157: the old flat list shifted the
    // keyword-only defaults by one and appended the vector last).
    let pos_values: Vec<TokenStream> = slots
        .into_iter()
        .map(|s| s.expect("all argument slots filled"))
        .collect();
    let kwonly_values: Vec<TokenStream> = kwonly_slots
        .into_iter()
        .map(|s| s.expect("all argument slots filled"))
        .collect();

    let mut final_slots: Vec<TokenStream> = Vec::with_capacity(total + 2);
    final_slots.extend(pos_values.iter().cloned());
    // The *args slot (collected extra positionals) sits between the
    // positional params and the keyword-only params.
    if vararg_param.is_some() {
        let items: Vec<(TokenStream, bool)> = vararg_extras
            .iter()
            .map(|(_, v, s)| (v.clone(), *s))
            .collect();
        final_slots.push(build_vararg(&items));
    }
    final_slots.extend(kwonly_values.iter().cloned());
    // Issue #120: append the **kwargs dict — explicit extra keywords boxed
    // in PyValue::from, then `**d` spreads merged in source order. When
    // keywords reorder the emission, the element values are bound to temps
    // in the prelude (their eval_order index is recorded), so the dict
    // references the temps instead of re-rendering the expressions.
    let kw_expr = if kwarg_param.is_some() {
        let pairs = if keywords.is_empty() {
            Vec::new()
        } else {            extra_kwarg_temps
                .iter()
                .zip(extra_kwargs.iter())
                .map(|(ti, (name, _))| {
                    let tid = format_ident!("__rython_arg_{}", ti);
                    quote!((#name.to_string(), #tid))
                })
                .collect::<Vec<_>>()
        };
        Some(if kw_spreads.is_empty() {
            quote!(PyDict::<String, PyValue>::from([#(#pairs),*]))
        } else {
            quote!({
                let mut __rython_kw = PyDict::<String, PyValue>::from([#(#pairs),*]);
                #( __rython_kw.update(#kw_spreads); )*
                __rython_kw
            })
        })
    } else {
        None
    };
    if let Some(kw) = &kw_expr {
        final_slots.push(kw.clone());
    }

    if keywords.is_empty() {
        // No reordering: the parameter-ordered emission is already the
        // source order, so emit the arguments directly (no prelude).
        return Ok(MappedArguments {
            prelude: TokenStream::new(),
            args: final_slots,
        });
    }

    // Keywords reorder the emission: bind every argument to a temp in
    // source order, then reference the temps in parameter order.
    let mut prelude = TokenStream::new();
    for (i, value) in eval_order.iter().enumerate() {
        let tid = format_ident!("__rython_arg_{}", i);
        prelude.extend(quote!(let #tid = #value;));
    }
    // Signature order: positional temps, the *args vector, keyword-only
    // temps, the **kwargs dict — mirroring the non-reorder layout above.
    // A slot with no temp holds a constant default (verified by
    // `check_default_constant`), so inlining it in parameter position
    // cannot reorder or duplicate side effects.
    let temp_or = |i: usize, value: &TokenStream| -> TokenStream {
        match slot_temp[i] {
            Some(ti) => {
                let tid = format_ident!("__rython_arg_{}", ti);
                quote!(#tid)
            }
            None => value.clone(),
        }
    };
    let mut args: Vec<TokenStream> = Vec::with_capacity(total + 2);
    for (i, v) in pos_values.iter().enumerate() {
        args.push(temp_or(i, v));
    }
    // The *args vector references its prelude temps (each collected
    // extra was bound in source order).
    if vararg_param.is_some() {
        let items: Vec<(TokenStream, bool)> = vararg_extras
            .iter()
            .map(|(ti, _, s)| {
                let tid = format_ident!("__rython_arg_{}", ti);
                (quote!(#tid), *s)
            })
            .collect();
        args.push(build_vararg(&items));
    }
    for (i, v) in kwonly_values.iter().enumerate() {
        args.push(temp_or(n + i, v));
    }
    if let Some(kw) = &kw_expr {
        args.push(kw.clone());
    }
    Ok(MappedArguments { prelude, args })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_of_function() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(a=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let code = result
            .to_rust(CodeGenContext::Module("test".to_string()), options, symbols)
            .unwrap()
            .to_string();
        assert!(
            code.contains("let __rython_arg_0 = 9 ; foo (__rython_arg_0)"),
            "generated: {}",
            code
        );
    }

    #[test]
    fn unknown_keyword_argument_is_a_conversion_error() {
        // Python raises TypeError for foo(b=9) when foo has no parameter b;
        // silently passing it positionally would be wrong.
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(b=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let err = result
            .to_rust(CodeGenContext::Module("test".to_string()), options, symbols)
            .expect_err("unexpected keyword must not convert");
        assert!(
            format!("{}", err).contains("unexpected keyword"),
            "error: {}",
            err
        );
    }
}

/// Route a SUBSCRIPT/`in`/dunder operation to a user-class's own dunder
/// method through the FULL call-argument mapping — the same mapping a
/// normal call receives — so arguments coerce to the method's declared
/// parameter types (`x[key]` where `__getitem__(self, key: Any)` boxes
/// the key, a str-typed key owns literals, ...). This is the §7
/// mapping-protocol slice: the class's own method IS Python's behavior,
/// including its exceptions and any case-insensitivity. `fallible` adds
/// the `?` (subscript reads and `in` propagate; the get-synthesis
/// matches the Result itself to catch KeyError).
pub(crate) fn dunder_method_call(
    method: &crate::FunctionDef,
    recv: &TokenStream,
    args: &[ExprType],
    fallible: bool,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut sig = method.clone();
    // The receiver parameter (`self`) is bound to the receiver
    // expression, never an argument — the same strip the ordinary
    // self-method call path applies (strip_self).
    crate::strip_self(&mut sig.args);
    let MappedArguments { prelude, args } =
        map_call_arguments(&sig, args, &[], ctx, options, symbols)?;
    let mname = crate::safe_ident(&method.name);
    if fallible {
        Ok(quote!({ #prelude (#recv).#mname(#(#args),*)? }))
    } else {
        Ok(quote!({ #prelude (#recv).#mname(#(#args),*) }))
    }
}

/// Whether a dunder method has a WELL-TYPED first argument — a concrete
/// (non-`Any`/`object`/unannotated) annotation after the receiver. The
/// mapping-protocol routing (subscripts, `in`, get-synthesis) only fires
/// for these: an `Any`-typed dunder (`RecentlyUsedContainer.
/// __setitem__(self, key: Any, value: Any)` — urllib3) cannot coerce
/// the call's arguments either, so routing would merely swap one loud
/// rustc error for another — the pre-existing py_index path stays.
pub(crate) fn dunder_method_well_typed(method: &crate::FunctionDef) -> bool {
    let mut args = method.args.clone();
    crate::strip_self(&mut args);
    let first = args.posonlyargs.first().or_else(|| args.args.first());
    let Some(p) = first else {
        // Zero parameters after the receiver: a subscript with an index
        // cannot fill it — the py_index fallback (loud) is more honest.
        return false;
    };
    match p.annotation.as_deref() {
        Some(ExprType::Name(n)) => !matches!(n.id.as_str(), "Any" | "object" | "None"),
        Some(_) => true,
        None => false,
    }
}

/// Whether a receiver is a BYTES value (`b"..."`, a Vec<u8>-typed name) —
/// the bytes twin of `receiver_is_str_like`: bytes methods (join, ...)
/// dispatch to the runtime's bytes surface.
pub(crate) fn receiver_is_bytes_like(
    receiver: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match receiver {
        ExprType::Constant(c) => matches!(
            &c.0,
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_))
        ),
        ExprType::Name(n) => {
            options
                .local_types
                .get(&n.id)
                .is_some_and(|t| t.contains("Vec < u8 >") || t.contains("Vec<u8>"))
                || options
                    .name_types
                    .get(&n.id)
                    .is_some_and(|t| matches!(t, crate::TypeInfo::Bytes))
        }
        ExprType::Call(c) => {
            // A `.to_vec()` of a bytes literal (`b"ab".to_vec()`).
            matches!(c.func.as_ref(), ExprType::Attribute(a)
                if a.attr == "to_vec"
                    && matches!(a.value.as_ref(), ExprType::Constant(cc)
                        if matches!(&cc.0, Some(litrs::Literal::Byte(_))
                            | Some(litrs::Literal::ByteString(_)))))
                // A call whose return annotation is bytes
                // (`print(join_sep([...]))` where join_sep -> bytes).
                || crate::call_return_typeinfo(c, Some(symbols), Some(options))
                    .is_some_and(|t| matches!(t, crate::TypeInfo::Bytes))
        }
        _ => false,
    }
}
