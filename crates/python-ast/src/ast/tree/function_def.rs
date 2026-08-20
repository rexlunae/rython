use tracing::debug;
use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};
use crate::ast::tree::statement::PyStatementTrait;

use crate::{
    CodeGen, CodeGenContext, ExprType, Object, ParameterList, PythonOptions, Statement,
    StatementType, SymbolTableNode, SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: ParameterList,
    pub body: Vec<Statement>,
    pub decorator_list: Vec<ExprType>,
    /// The function's return annotation (`-> int`), if present.
    pub returns: Option<Box<ExprType>>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for FunctionDef {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let name: String = ob.getattr("name")?.extract()?;
        let args: ParameterList = ob.getattr("args")?.extract()?;
        let body: Vec<Statement> = ob.getattr("body")?.extract()?;

        // Extract decorator_list as Vec<ExprType>
        let decorator_list: Vec<ExprType> = ob.getattr("decorator_list")?.extract().unwrap_or_default();

        // Extract the return annotation, if any.
        let returns: Option<Box<ExprType>> = match ob.getattr("returns") {
            Ok(r) if !r.is_none() => r.extract().ok().map(Box::new),
            _ => None,
        };

        Ok(FunctionDef {
            name,
            args,
            body,
            decorator_list,
            returns,
        })
    }
}

impl PyStatementTrait for FunctionDef {
}

/// One add_argument spec collected at conversion time.
struct ArgparseSpec {
    name: String,
    kind: &'static str, // "Str" | "Int" | "Float" | "StoreTrue"
    default: Option<ExprType>,
    help: Option<String>,
}

/// The argparse rewrite plan for a function body: parser-building
/// statements to drop, the parse_args assignment to replace, and the
/// literal specs. ArgumentParser/add_argument/parse_args are evaluated
/// HERE, at conversion time — only literal specs can shape the typed
/// namespace struct, so anything dynamic is a loud error.
struct ArgparseRewrite {
    skip: std::collections::HashSet<usize>,
    parse_index: usize,
    args_var: String,
    prog: Option<String>,
    description: Option<String>,
    specs: Vec<ArgparseSpec>,
}

fn literal_str(e: &ExprType) -> Option<String> {
    match e {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(s)) => Some(s.value().to_string()),
            _ => None,
        },
        _ => None,
    }
}

fn scan_argparse(
    body: &[Statement],
) -> Result<Option<ArgparseRewrite>, Box<dyn std::error::Error>> {
    // Find `<var> = argparse.ArgumentParser(...)`.
    let mut parser: Option<(usize, String, Option<String>, Option<String>)> = None;
    for (i, stmt) in body.iter().enumerate() {
        let StatementType::Assign(assign) = &stmt.statement else {
            continue;
        };
        let ExprType::Call(call) = &assign.value else {
            continue;
        };
        let ExprType::Attribute(attr) = call.func.as_ref() else {
            continue;
        };
        let is_ctor = attr.attr == "ArgumentParser"
            && matches!(attr.value.as_ref(), ExprType::Name(m) if m.id == "argparse");
        if !is_ctor {
            continue;
        }
        let [ExprType::Name(target)] = assign.targets.as_slice() else {
            return Err("argparse.ArgumentParser must be assigned to a plain name".into());
        };
        if !call.args.is_empty() {
            return Err(
                "argparse.ArgumentParser: pass prog=/description= by keyword".into(),
            );
        }
        let mut prog = None;
        let mut description = None;
        for kw in &call.keywords {
            let value = literal_str(&kw.value).ok_or_else(|| {
                format!(
                    "argparse.ArgumentParser: {} must be a string literal (the parser \
                     is evaluated at conversion time)",
                    kw.arg.as_deref().unwrap_or("argument")
                )
            })?;
            match kw.arg.as_deref() {
                Some("prog") => prog = Some(value),
                Some("description") => description = Some(value),
                other => {
                    return Err(format!(
                        "argparse.ArgumentParser keyword '{}' is not supported yet",
                        other.unwrap_or("**kwargs")
                    )
                    .into())
                }
            }
        }
        parser = Some((i, target.id.clone(), prog, description));
        break;
    }
    let Some((ctor_index, pvar, prog, description)) = parser else {
        return Ok(None);
    };

    // Collect `<pvar>.add_argument(...)` statements and the final
    // `<args> = <pvar>.parse_args()`.
    let mut skip = std::collections::HashSet::from([ctor_index]);
    let mut specs = Vec::new();
    let mut parse: Option<(usize, String)> = None;
    for (i, stmt) in body.iter().enumerate().skip(ctor_index + 1) {
        let call_on_parser = |call: &crate::Call| -> Option<String> {
            let ExprType::Attribute(attr) = call.func.as_ref() else {
                return None;
            };
            match attr.value.as_ref() {
                ExprType::Name(m) if m.id == pvar => Some(attr.attr.clone()),
                _ => None,
            }
        };
        // A bare call statement surfaces as Expr(Call) or Call
        // depending on the extraction path; normalize.
        let stmt_call: Option<&crate::Call> = match &stmt.statement {
            StatementType::Call(c) => Some(c),
            StatementType::Expr(e) => match &e.value {
                ExprType::Call(c) => Some(c),
                _ => None,
            },
            _ => None,
        };
        match &stmt.statement {
            _ if stmt_call.is_some_and(|c| call_on_parser(c) == Some("add_argument".into())) => {
                let call = stmt_call.expect("checked");
                if parse.is_some() {
                    return Err("add_argument after parse_args is not supported".into());
                }
                let [name_expr] = call.args.as_slice() else {
                    return Err(
                        "add_argument takes exactly one name (short aliases are not \
                         supported yet)"
                            .into(),
                    );
                };
                let name = literal_str(name_expr)
                    .ok_or("add_argument: the name must be a string literal")?;
                if name.starts_with('-') && !name.starts_with("--") {
                    return Err(format!(
                        "add_argument: short option '{}' is not supported yet; use the \
                         --long form",
                        name
                    )
                    .into());
                }
                let mut kind: Option<&'static str> = None;
                let mut default = None;
                let mut help = None;
                let mut store_true = false;
                for kw in &call.keywords {
                    match kw.arg.as_deref() {
                        Some("type") => {
                            kind = Some(match &kw.value {
                                ExprType::Name(n) if n.id == "int" => "Int",
                                ExprType::Name(n) if n.id == "float" => "Float",
                                ExprType::Name(n) if n.id == "str" => "Str",
                                _ => {
                                    return Err(format!(
                                        "add_argument('{}'): type must be int, float, \
                                         or str",
                                        name
                                    )
                                    .into())
                                }
                            });
                        }
                        Some("default") => default = Some(kw.value.clone()),
                        Some("help") => {
                            help = Some(literal_str(&kw.value).ok_or_else(|| {
                                format!(
                                    "add_argument('{}'): help must be a string literal",
                                    name
                                )
                            })?)
                        }
                        Some("action") => match literal_str(&kw.value).as_deref() {
                            Some("store_true") => store_true = true,
                            _ => {
                                return Err(format!(
                                    "add_argument('{}'): only action=\"store_true\" is \
                                     supported",
                                    name
                                )
                                .into())
                            }
                        },
                        other => {
                            return Err(format!(
                                "add_argument('{}'): keyword '{}' is not supported yet",
                                name,
                                other.unwrap_or("**kwargs")
                            )
                            .into())
                        }
                    }
                }
                let kind = if store_true {
                    if kind.is_some() || default.is_some() {
                        return Err(format!(
                            "add_argument('{}'): store_true takes neither type nor \
                             default",
                            name
                        )
                        .into());
                    }
                    "StoreTrue"
                } else {
                    kind.unwrap_or("Str")
                };
                let is_positional = !name.starts_with('-');
                if is_positional && default.is_some() {
                    return Err(format!(
                        "add_argument('{}'): defaults on positionals are not supported",
                        name
                    )
                    .into());
                }
                if !is_positional && !store_true && default.is_none() {
                    return Err(format!(
                        "add_argument('{}'): a value-taking option needs default= (its \
                         Python default None cannot inhabit a typed field)",
                        name
                    )
                    .into());
                }
                specs.push(ArgparseSpec {
                    name,
                    kind,
                    default,
                    help,
                });
                skip.insert(i);
            }
            StatementType::Assign(assign) => {
                if let ExprType::Call(call) = &assign.value {
                    if call_on_parser(call) == Some("parse_args".into()) {
                        if !call.args.is_empty() || !call.keywords.is_empty() {
                            return Err("parse_args with arguments is not supported".into());
                        }
                        let [ExprType::Name(t)] = assign.targets.as_slice() else {
                            return Err("parse_args must be assigned to a plain name".into());
                        };
                        parse = Some((i, t.id.clone()));
                    } else if call_on_parser(call).is_some() {
                        return Err(format!(
                            "argparse parser `{}`: only add_argument and parse_args \
                             are supported",
                            pvar
                        )
                        .into());
                    }
                }
            }
            _ if stmt_call.is_some_and(|c| call_on_parser(c).is_some()) => {
                return Err(format!(
                    "argparse parser `{}`: only add_argument and parse_args are \
                     supported",
                    pvar
                )
                .into());
            }
            _ => {}
        }
    }
    let Some((parse_index, args_var)) = parse else {
        return Err("argparse.ArgumentParser built but parse_args() never assigned".into());
    };
    Ok(Some(ArgparseRewrite {
        skip,
        parse_index,
        args_var,
        prog,
        description,
        specs,
    }))
}

/// Emit the parse_args replacement: a namespace struct typed from the
/// specs, the run_parser call, and the destructuring assignment into
/// the (hoisted) namespace variable.
fn lower_parse_args(
    rw: &ArgparseRewrite,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    use quote::format_ident;
    let mut fields = Vec::new();
    let mut field_types = Vec::new();
    let mut spec_tokens = Vec::new();
    let mut accessors = Vec::new();
    for spec in &rw.specs {
        let dest = spec.name.trim_start_matches('-').replace('-', "_");
        fields.push(crate::safe_ident(&dest));
        let (fty, kind, accessor) = match spec.kind {
            "Int" => (quote!(i64), quote!(Int), format_ident!("into_int")),
            "Float" => (quote!(f64), quote!(Float), format_ident!("into_float")),
            "StoreTrue" => (quote!(bool), quote!(StoreTrue), format_ident!("into_flag")),
            _ => (quote!(String), quote!(Str), format_ident!("into_str")),
        };
        field_types.push(fty);
        accessors.push(accessor);
        let default = match &spec.default {
            None => quote!(None),
            Some(e) => {
                let d = e
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                // Coerce literal defaults onto the declared type
                // (default=1 with type=float is valid Python).
                match spec.kind {
                    "Int" => quote!(Some(argparse::ParsedValue::Int((#d) as i64))),
                    "Float" => quote!(Some(argparse::ParsedValue::Float((#d) as f64))),
                    _ => quote!(Some(argparse::ParsedValue::Str((#d).to_string()))),
                }
            }
        };
        let name = &spec.name;
        let help = match &spec.help {
            Some(h) => quote!(Some(#h)),
            None => quote!(None),
        };
        spec_tokens.push(quote!(argparse::ArgSpec {
            name: #name,
            kind: argparse::ArgKind::#kind,
            default: #default,
            help: #help,
        }));
    }
    let prog = match &rw.prog {
        Some(p) => quote!(Some(#p)),
        None => quote!(None),
    };
    let description = match &rw.description {
        Some(d) => quote!(Some(#d)),
        None => quote!(None),
    };
    let args_var = crate::safe_ident(&rw.args_var);
    Ok(quote! {
        #[allow(non_camel_case_types)]
        struct __ArgparseArgs {
            #(#fields: #field_types,)*
        }
        let mut __parsed = argparse::run_parser(
            #prog,
            #description,
            &[#(#spec_tokens),*],
        )?
        .into_iter();
        #args_var = __ArgparseArgs {
            #(#fields: __parsed.next().expect("one value per spec").#accessors(),)*
        }
    })
}

/// The cache discipline a functools cache decorator asks for: None
/// means uncached; Some(None) is unbounded (functools.cache or
/// lru_cache(maxsize=None)); Some(Some(n)) is a bounded LRU (Python's
/// bare @lru_cache default is 128). ANY other decorator is a loud
/// error: silently ignoring a decorator converts a program into a
/// different one.
fn parse_cache_decorator(
    decorators: &[ExprType],
) -> Result<Option<Option<i64>>, Box<dyn std::error::Error>> {
    let unsupported = |what: &str| -> Box<dyn std::error::Error> {
        format!(
            "decorator `{}` is not supported yet (only functools.lru_cache and \
             functools.cache are); rython refuses to silently ignore it",
            what
        )
        .into()
    };
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
                (Some("cache"), None) => Ok(Some(None)),
                (Some("cache"), Some(c)) if c.args.is_empty() && c.keywords.is_empty() => {
                    Ok(Some(None))
                }
                (Some("lru_cache"), None) => Ok(Some(Some(128))),
                (Some("lru_cache"), Some(c)) => {
                    let maxsize = match (c.args.as_slice(), c.keywords.as_slice()) {
                        ([], []) => return Ok(Some(Some(128))),
                        ([e], []) => e.clone(),
                        ([], [kw]) if kw.arg.as_deref() == Some("maxsize") => {
                            kw.value.clone()
                        }
                        _ => {
                            return Err(
                                "lru_cache() takes at most a single maxsize argument"
                                    .to_string()
                                    .into(),
                            )
                        }
                    };
                    if crate::is_none_expr(&maxsize) {
                        return Ok(Some(None));
                    }
                    match &maxsize {
                        ExprType::Constant(c) => match &c.0 {
                            Some(litrs::Literal::Integer(i)) => {
                                let n: i64 = i.value().ok_or("maxsize out of range")?;
                                Ok(Some(Some(n)))
                            }
                            _ => Err("lru_cache maxsize must be an integer literal or None"
                                .to_string()
                                .into()),
                        },
                        _ => Err("lru_cache maxsize must be an integer literal or None"
                            .to_string()
                            .into()),
                    }
                }
                _ => Err(unsupported(&format!("{:?}", single))),
            }
        }
        many => Err(unsupported(&format!("{:?}", many[0]))),
    }
}

impl CodeGen for FunctionDef {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        symbols.insert(
            self.name.clone(),
            SymbolTableNode::FunctionDef(self.clone()),
        );
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut streams = TokenStream::new();
        let fn_name = crate::safe_ident(&self.name);

        // Issue #115: `global x` declares module scope. Reads resolve to
        // the module statics; WRITES need mutable module state, which
        // rython does not model (module-level reassignment lowers to
        // __module_init__ locals invisible to functions) — loud error
        // naming the fix instead of a rustc surprise.
        if let Some(name) = global_write_error(&self.body) {
            return Err(format!(
                "writing to module-level name `{name}` from function `{}` is not supported \
                 (issue #115): rython has no mutable module state visible to functions; \
                 keep the mutation inside one function, or restructure to avoid the \
                 global",
                self.name
            )
            .into());
        }

        // An argparse parser in the body is evaluated at conversion time:
        // its statements vanish and parse_args becomes a typed struct.
        let argparse_rewrite = scan_argparse(&self.body)?;
        let effective_body: Vec<Statement> = match &argparse_rewrite {
            None => self.body.clone(),
            Some(rw) => self
                .body
                .iter()
                .enumerate()
                .filter(|(i, _)| !rw.skip.contains(i))
                .map(|(_, s)| s.clone())
                .collect(),
        };
        // The parse_args statement's position within the filtered body.
        let argparse_parse_at: Option<usize> = argparse_rewrite.as_ref().map(|rw| {
            (0..rw.parse_index)
                .filter(|i| !rw.skip.contains(i))
                .count()
        });

        // functools cache decorators rewrite the whole definition below;
        // any OTHER decorator is a loud error (see parse_cache_decorator).
        let cache_spec = parse_cache_decorator(&self.decorator_list)?;
        if cache_spec.is_some() && options.no_std {
            return Err(format!(
                "@lru_cache on `{}` needs a global Mutex, which the no_std \
                 profile does not provide",
                self.name
            )
            .into());
        }

        // The Python convention is that functions that begin with a single underscore,
        // it's private. Otherwise, it's public. We formalize that by default.
        // Trait items carry no visibility modifier at all (they are public
        // through the trait); only inherent methods get one.
        let visibility = if matches!(&ctx, CodeGenContext::Trait { .. }) {
            quote!()
        } else if self.name.starts_with("_") && !self.name.starts_with("__") {
            quote!()  // private, no visibility modifier
        } else if self.name.starts_with("__") && self.name.ends_with("__") {
            quote!(pub(crate))  // dunder methods are crate-visible
        } else {
            quote!(pub)  // regular methods are public
        };

        // A nested function body is a fresh exception scope: a `raise` in it
        // cannot return out of an enclosing try block's closure.
        let ctx = ctx.strip_exception_scopes();

        let is_async = match ctx.clone() {
            CodeGenContext::Async(_) => {
                quote!(async)
            }
            _ => quote!(),
        };

        // Local assignments participate in name resolution for the body:
        // `p = Point(...)` makes `p`'s class known to method-call lowering.
        let mut symbols = symbols;
        for s in &self.body {
            symbols = s.clone().find_symbols(symbols);
        }

        // A `def` in a class body whose first parameter is `self` is a
        // method: `self` becomes the Rust receiver instead of a parameter —
        // `&mut self` when the method stores through `self`, directly or by
        // calling another method of the class that does. In a Trait context
        // the method is emitted as a trait item (a default in the class's
        // trait, or an override in an ancestor's trait's impl).
        let is_method = matches!(
            &ctx,
            CodeGenContext::Class(_) | CodeGenContext::Trait { .. }
        ) && self
            .args
            .posonlyargs
            .first()
            .or(self.args.args.first())
            .is_some_and(|p| p.arg == "self");
        let mut render_args = self.args.clone();
        let method_mutates_self = is_method
            && match &ctx {
                CodeGenContext::Class(class_name)
                | CodeGenContext::Trait { class: class_name, .. } => {
                    match symbols.get(class_name) {
                        Some(crate::SymbolTableNode::ClassDef(c)) => {
                            c.method_needs_mut_self(&self.name, &symbols, &options)
                        }
                        _ => false,
                    }
                }
                _ => false,
            };
        if is_method {
            crate::strip_self(&mut render_args);
        }

        // A cached function's arguments form the cache KEY, so every
        // parameter needs a hashable, nameable type: int, bool, or str
        // (floats are not hashable in Rust — Python would cache them,
        // which rython cannot reproduce, so it refuses loudly).
        let cache_key: Option<Vec<(proc_macro2::Ident, TokenStream)>> = match cache_spec {
            None => None,
            Some(_) => {
                if is_method {
                    return Err(format!(
                        "@lru_cache on method `{}` is not supported yet",
                        self.name
                    )
                    .into());
                }
                if !self.args.posonlyargs.is_empty()
                    || !self.args.kwonlyargs.is_empty()
                    || self.args.vararg.is_some()
                    || self.args.kwarg.is_some()
                {
                    return Err(format!(
                        "@lru_cache on `{}`: only plain positional parameters are \
                         supported",
                        self.name
                    )
                    .into());
                }
                let mut key = Vec::new();
                for p in &self.args.args {
                    let ty = match p.annotation.as_deref() {
                        Some(ExprType::Name(n)) if n.id == "int" => quote!(i64),
                        Some(ExprType::Name(n)) if n.id == "bool" => quote!(bool),
                        Some(ExprType::Name(n)) if n.id == "str" => quote!(String),
                        _ => {
                            return Err(format!(
                                "@lru_cache on `{}`: parameter `{}` must be annotated \
                                 int, bool, or str (the arguments form the cache key)",
                                self.name, p.arg
                            )
                            .into());
                        }
                    };
                    key.push((crate::safe_ident(&p.arg), ty));
                }
                Some(key)
            }
        };

        // Python variables are function-scoped: hoist every assigned name to
        // a declaration here so assignments inside nested blocks (if/loop/
        // try bodies) store into the same variable instead of creating a
        // shadowing binding. Scope analysis decides which declarations need
        // `mut` (mirroring rustc's rules, so the generated code carries
        // neither unused_mut warnings nor missing-mut errors), and which
        // parameters must be rebound as mutable locals (Rust parameters are
        // immutable; Python's are ordinary variables).
        let mut param_names: Vec<String> = render_args
            .args
            .iter()
            .chain(render_args.posonlyargs.iter())
            .chain(render_args.kwonlyargs.iter())
            .map(|p| p.arg.clone())
            .chain(render_args.vararg.iter().map(|p| p.arg.clone()))
            .chain(render_args.kwarg.iter().map(|p| p.arg.clone()))
            .collect();
        if is_method {
            // The receiver is initialized like a parameter, but it is never
            // rebound (`let mut self = self` is not legal Rust); its
            // mutations select `&mut self` above instead.
            param_names.push("self".to_string());
        }
        // The resolver makes class knowledge authoritative for method
        // calls: `c.bump()` marks `c` mutable when bump takes &mut self,
        // and a read-only user method shadowing a builtin mutator name
        // (`pop`, `update`, ...) does NOT force a spurious `mut`.
        let scope = crate::analyze_scope_with(
            &effective_body,
            &param_names,
            &crate::class_call_resolver(&ctx, &symbols, &options),
        );
        if is_method {
            param_names.pop();
        }
        let forced_mut_self = matches!(
            &ctx,
            CodeGenContext::Trait {
                force_mut_self: true,
                ..
            }
        );
        let receiver = if is_method {
            if forced_mut_self || method_mutates_self || scope.needs_mut.contains("self") {
                quote!(&mut self,)
            } else {
                quote!(&self,)
            }
        } else {
            quote!()
        };
        // Optional names (assigned None on some path, or parameters with an
        // Optional annotation) are visible to every assignment in the body:
        // their non-None stores wrap in Some.
        let mut options = options;
        // Names managed by this function's prologue: hoisted assignments
        // plus mutable parameters. A `for`-loop target on one of these
        // lowers to a store into the hoisted binding, never a shadowing
        // fresh binding (issue #80).
        options.hoisted_names = std::rc::Rc::new(
            scope
                .assigned
                .iter()
                .chain(scope.needs_mut.iter())
                .cloned()
                .collect(),
        );
        // Only the targets whose value is observed after the loop store
        // into the hoisted binding; the rest keep fresh per-loop bindings
        // (issue #80).
        options.leaked_loop_targets = std::rc::Rc::new(scope.leaked_loop_targets.clone());
        {
            let mut optional = scope.optional.clone();
            for p in self
                .args
                .posonlyargs
                .iter()
                .chain(self.args.args.iter())
                .chain(self.args.kwonlyargs.iter())
            {
                if let Some(ann) = p.annotation.as_deref() {
                    if crate::is_optional_annotation(ann) {
                        optional.insert(p.arg.clone());
                    }
                }
            }
            options.optional_names = std::rc::Rc::new(optional);
            options.clone_str_attribute_returns =
                matches!(self.returns.as_deref(), Some(ExprType::Name(n)) if n.id == "str");
            // Locals whose only known type is a string literal (`label =
            // "fine"`): they lower to `&'static str`, which a `-> str`
            // return must own. Reuse the literal-local inference, keyed by
            // the same `&'static str` type it records.
            if options.clone_str_attribute_returns {
                let mut locals = std::collections::HashMap::new();
                collect_local_types(&effective_body, &mut locals);
                options.str_literal_locals = std::rc::Rc::new(
                    locals
                        .into_iter()
                        .filter(|(_, ty)| ty.to_string() == quote!(&'static str).to_string())
                        .map(|(name, _)| name)
                        .collect(),
                );
            }
            // Issue #110: a name bound to a string literal and later
            // REBOUND by a String-producing expression (`out = ""; out +=
            // "x"`) must be owned from the first assignment
            // (`out = "".to_string()`), or the binding stays &'static str
            // and the String rebind mismatches at build time.
            let mut literal_bindings: std::collections::HashMap<String, bool> =
                std::collections::HashMap::new();
            let mut rebound: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            scan_str_rebindings(&effective_body, &mut literal_bindings, &mut rebound);
            options.owned_str_literals = std::rc::Rc::new(
                literal_bindings
                    .into_iter()
                    .filter(|(name, _)| rebound.contains(name))
                    .map(|(name, _)| name)
                    .collect(),
            );
        }
        // Statically-known local types for isinstance(): parameter
        // annotations plus literal assignments, as Python type names.
        {
            let mut known: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for param in self
                .args
                .args
                .iter()
                .chain(self.args.posonlyargs.iter())
                .chain(self.args.kwonlyargs.iter())
            {
                if let Some(ExprType::Name(ann)) = param.annotation.as_deref() {
                    if matches!(ann.id.as_str(), "int" | "float" | "str" | "bool") {
                        known.insert(param.arg.clone(), ann.id.clone());
                    }
                }
            }
            let mut literal_types = std::collections::HashMap::new();
            collect_local_types(&self.body, &mut literal_types);
            for (name, ty) in literal_types {
                let py = match ty.to_string().as_str() {
                    "i64" => "int",
                    "f64" => "float",
                    "bool" => "bool",
                    s if s.contains("str") || s.contains("String") => "str",
                    _ => continue,
                };
                // A literal assignment overrides nothing: annotations win.
                known.entry(name).or_insert_with(|| py.to_string());
            }
            options.local_types = std::rc::Rc::new(known);
        }
        // Type-aware lowering context for the body: read-use counts (for
        // clone-on-reuse), inferred name types, and empty-container types
        // pinned by later use. Annotation-derived types win over
        // assignment-inferred ones, matching local_types above.
        {
            let mut info = crate::analyze_function_types(&effective_body);
            for p in self
                .args
                .args
                .iter()
                .chain(self.args.posonlyargs.iter())
                .chain(self.args.kwonlyargs.iter())
            {
                if let Some(ann) = p.annotation.as_deref() {
                    // Scalar annotations map directly; container
                    // annotations (`list[float]`, `dict[str, int]`,
                    // `Optional[str]`) arrive as Subscript expressions.
                    match ann {
                        ExprType::Name(n) => match n.id.as_str() {
                            "int" => {
                                info.name_types.insert(p.arg.clone(), crate::TypeInfo::Int);
                            }
                            "float" => {
                                info.name_types.insert(p.arg.clone(), crate::TypeInfo::Float);
                            }
                            "bool" => {
                                info.name_types.insert(p.arg.clone(), crate::TypeInfo::Bool);
                            }
                            "str" => {
                                info.name_types.insert(p.arg.clone(), crate::TypeInfo::String);
                            }
                            "bytes" => {
                                info.name_types.insert(p.arg.clone(), crate::TypeInfo::Bytes);
                            }
                            // A bare `list`/`dict`/... annotation has no
                            // element/key type: the generated Rust would be
                            // `xs: list` — invalid — so fail loudly at
                            // conversion time instead of at rustc.
                            "list" | "List" | "dict" | "Dict" | "tuple" | "Tuple" | "set"
                            | "Set" | "Optional" => {
                                return Err(format!(
                                    "parameter `{}` is annotated `{}`, which has no element/\
                                     key type; use a subscripted annotation like \
                                     `list[float]` or `dict[str, int]`",
                                    p.arg, n.id
                                )
                                .into());
                            }
                            _ => {}
                        },
                        other => {
                            if let Some(t) = crate::annotation_type_info(other) {
                                info.name_types.insert(p.arg.clone(), t);
                            } else if crate::python_annotation_to_rust_type(other).is_none() {
                                // The annotation is genuinely unsupported
                                // (e.g. a custom type): fail loudly at
                                // conversion time instead of emitting
                                // invalid Rust that rustc rejects.
                                return Err(format!(
                                    "parameter `{}` has an unsupported annotation `{}`",
                                    p.arg,
                                    crate::annotation_display(other)
                                )
                                .into());
                            }
                        }
                    }
                }
            }
            options.use_counts = std::rc::Rc::new(info.use_counts);
            options.name_types = std::rc::Rc::new(info.name_types);
            options.empty_pinned = std::rc::Rc::new(info.empty_pinned);
        }
        // Empty-container pinning needs the parameter annotations above,
        // so re-run the pin pass now that name_types knows the params
        // (`xs.append(x[0])` can resolve x's element type from its
        // `list[float]` annotation).
        {
            let mut info = crate::FunctionTypeInfo {
                use_counts: options.use_counts.as_ref().clone(),
                name_types: options.name_types.as_ref().clone(),
                empty_pinned: options.empty_pinned.as_ref().clone(),
            };
            crate::pin_empty_containers(&effective_body, &mut info);
            options.use_counts = std::rc::Rc::new(info.use_counts);
            options.name_types = std::rc::Rc::new(info.name_types);
            options.empty_pinned = std::rc::Rc::new(info.empty_pinned);
        }
        // Issue #79's cheap guard: reject aliasing shapes rython cannot
        // model (`b = a` on a container that is later mutated, and passing
        // a container to a function that mutates it) at conversion time
        // instead of silently diverging or leaving the report to rustc.
        crate::check_aliasing(
            &effective_body,
            &symbols,
            &options.name_types,
            &options.use_counts,
        )?;
        // Issue #109, M1: parameter type inference. Unannotated parameters
        // (other than `self`) get trait-bound generic signatures derived
        // from their uses — `def add(a, b): return a + b` becomes
        // `fn add<A, B>(a: A, b: B) -> Result<<A as PyAdd<B>>::Output, ...>
        // where A: PyAdd<B>`. The `impl Into<PyObject>` dead end is gone:
        // an unannotated parameter either infers or fails loudly here.
        let inferred_signature = {
            let is_method = matches!(
                &ctx,
                CodeGenContext::Class(_) | CodeGenContext::Trait { .. }
            ) && self
                .args
                .posonlyargs
                .first()
                .or(self.args.args.first())
                .is_some_and(|p| p.arg == "self");
            let unannotated: Vec<String> = self
                .args
                .posonlyargs
                .iter()
                .chain(self.args.args.iter())
                .chain(self.args.kwonlyargs.iter())
                .filter(|p| p.arg != "self" && p.annotation.is_none())
                .map(|p| p.arg.clone())
                .collect();
            if unannotated.is_empty() {
                // No inferred parameters: the body's calls are still
                // checked against callee bounds (M5, call-site
                // satisfiability) — the inference collector only walks
                // functions with unannotated parameters.
                crate::check_call_sites(
                    &effective_body,
                    &symbols,
                    &options.name_types,
                    &options,
                )
                .map_err(|e| format!("function `{}`: {}", self.name, e))?;
                crate::InferredSignature::default()
            } else if is_method {
                // M1 covers free functions; methods (and __init__) need
                // concrete parameter types — a loud error beats the old
                // uncallable `impl Into<PyObject>`.
                return Err(format!(
                    "parameter(s) `{}` of method `{}` are unannotated; annotate them \
                     (issue #109, M1 infers free functions only; method parameters are \
                     inferred in a later milestone)",
                    unannotated.join("`, `"),
                    self.name
                )
                .into());
            } else {
                crate::infer_unannotated_signature(
                    &effective_body,
                    &unannotated,
                    &options.name_types,
                    &options.use_counts,
                    &symbols,
                    &options,
                )
                .map_err(|e| {
                    format!("function `{}`: {}", self.name, e)
                })?
            }
        };
        // Thread the inferred type variables into parameter rendering
        // (Parameter::to_rust emits `a: A` instead of the dead
        // `impl Into<PyObject>`), and the stdlib-method-bound parameters
        // into method-call dispatch (M2).
        options.param_type_vars = std::rc::Rc::new(inferred_signature.param_types.clone());
        options.param_method_params =
            std::rc::Rc::new(inferred_signature.method_params.clone());
        options.duck_methods_on_params =
            std::rc::Rc::new(inferred_signature.duck_methods_on_params.clone());
        // str parameters arrive as impl Into<String>; convert them to owned
        // Strings up front so the body works with a concrete type.
        let str_params: std::collections::HashSet<&str> = self
            .args
            .args
            .iter()
            .chain(self.args.posonlyargs.iter())
            .chain(self.args.kwonlyargs.iter())
            .filter(|p| {
                matches!(
                    p.annotation.as_deref(),
                    Some(ExprType::Name(n)) if n.id == "str"
                )
            })
            .map(|p| p.arg.as_str())
            .collect();
        let mut streams_prologue = TokenStream::new();
        for name in &param_names {
            let ident = crate::safe_ident(name);
            if str_params.contains(name.as_str()) {
                if scope.needs_mut.contains(name) {
                    streams_prologue.extend(quote!(let mut #ident: String = #ident.into();));
                } else {
                    streams_prologue.extend(quote!(let #ident: String = #ident.into();));
                }
            } else if scope.needs_mut.contains(name) {
                streams_prologue.extend(quote!(let mut #ident = #ident;));
            }
        }
        for name in &scope.assigned {
            let ident = crate::safe_ident(name);
            if scope.needs_mut.contains(name) {
                if scope.closure_captured_uninit.contains(name) {
                    // The name is captured by a generated closure (try body,
                    // or finally-guarded handler/else body) while possibly
                    // uninitialized: a bare `let mut x;` would be rejected
                    // by rustc's E0381. Default-initialize so the capture
                    // is legal; the real value is stored on the happy path.
                    streams_prologue.extend(quote!(let mut #ident = Default::default();));
                } else {
                    streams_prologue.extend(quote!(let mut #ident;));
                }
            } else {
                streams_prologue.extend(quote!(let #ident;));
            }
        }
        streams.extend(streams_prologue);

        // A leading docstring is emitted as doc comments below; skip it here
        // so it isn't also emitted as a statement.
        let body_start = if self.get_docstring().is_some() { 1 } else { 0 };
        // Body statements render in a Function context. rython's
        // module-level-only constructs (rust.bind declarations) are rejected
        // inside functions; nothing in the lowerings inspects the outer
        // context (try/loop/async scopes are pushed explicitly below it),
        // except the enclosing class name, which the Function context carries
        // so `self` keeps resolving in method bodies. Trait method bodies
        // keep the Trait context so field access through the generated
        // accessors stays active.
        let body_ctx = match &ctx {
            CodeGenContext::Trait {
                class,
                generic,
                super_target,
                force_mut_self,
            } => CodeGenContext::Trait {
                class: class.clone(),
                generic: *generic,
                super_target: super_target.clone(),
                force_mut_self: *force_mut_self,
            },
            _ => CodeGenContext::Function {
                class: ctx.enclosing_class_name().map(str::to_string),
            },
        };
        for (i, s) in effective_body.iter().enumerate().skip(body_start) {
            if Some(i) == argparse_parse_at {
                let rw = argparse_rewrite.as_ref().expect("index implies rewrite");
                streams.extend(lower_parse_args(
                    rw,
                    &ctx,
                    &options,
                    &symbols,
                )?);
                streams.extend(quote!(;));
                continue;
            }
            streams.extend(
                s.clone()
                    .to_rust(body_ctx.clone(), options.clone(), symbols.clone())?,
            );
            streams.extend(quote!(;));
        }

        // Every generated function returns Result<T, PyException> so raised
        // exceptions propagate across function boundaries the way Python's
        // do: call sites append `?`, and an uncaught exception surfaces at
        // the entry point. T is the resolved Python return type (unit when
        // there is none).
        // A bare `-> list` / `-> dict` return annotation would silently
        // resolve to unit and then fail at rustc on the body's real value:
        // fail loudly at conversion time instead.
        if let Some(ann) = self.returns.as_deref()
            && let ExprType::Name(n) = ann
            && matches!(
                n.id.as_str(),
                "list" | "List" | "dict" | "Dict" | "tuple" | "Tuple" | "set" | "Set"
            )
        {
            return Err(format!(
                "return annotation `{}` has no element/key type; use a subscripted \
                 annotation like `list[float]` or `dict[str, int]`",
                n.id
            )
            .into());
        }
        let return_type = if inferred_signature.is_generic() && self.returns.is_none() {
            // The inferred generic return (issue #109, M1): a parameter's
            // variable, a conversion result, or an associated Output. Only
            // when every path returns — a fall-through body returns unit.
            // An explicit return annotation always wins.
            if guarantees_return(&self.body) {
                match &inferred_signature.return_type {
                    Some(ty) => quote!(-> Result<#ty, PyException>),
                    None => {
                        return Err(
                            "could not infer this function's return type from its \
                             unannotated parameters; add a return annotation \
                             (issue #109, M1)"
                                .to_string()
                                .into(),
                        )
                    }
                }
            } else if inferred_signature.return_type.as_ref().is_some_and(|ty| {
                // A return type that references a type variable (a loop
                // element, an associated Output) cannot coexist with a
                // fall-through path (Python returns None there) — loud,
                // never a rustc mismatch at build time. A CONCRETE partial
                // return (e.g. `if c: return 1`) keeps the old Result<()>
                // shape.
                let s = ty.to_string();
                let tokens: Vec<&str> = s
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|t| !t.is_empty())
                    .collect();
                inferred_signature.type_params.iter().any(|p| {
                    let p = p.to_string();
                    tokens.iter().any(|t| *t == p)
                })
            }) {
                // Some path returns an inferred generic value while
                // another can fall through (Python returns None there) —
                // the value cannot be both that type and unit. Loud,
                // never a rustc mismatch at build time (issue #109, M2).
                return Err(
                    "this function returns an inferred value on some path but can \
                     fall through without one (Python would return None); annotate \
                     its return type (issue #109, M2)"
                        .to_string()
                        .into(),
                );
            } else {
                quote!(-> Result<(), PyException>)
            }
        } else {
            match self.resolved_return_type() {
                Some(ty) => quote!(-> Result<#ty, PyException>),
                None => quote!(-> Result<(), PyException>),
            }
        };

        // A body that can fall off the end implicitly returns None: give the
        // generated block an Ok(()) tail. Bodies that return (or raise) on
        // every path end with `return`/`return Err`, which need no tail.
        if !guarantees_return(&self.body) {
            streams.extend(quote!(Ok(())));
        }

        // A cached function wraps its ORIGINAL body in an inner fn: the
        // outer fn consults a static LRU keyed on the argument tuple,
        // computes through the inner fn on a miss, and stores the clone.
        // Recursive calls in the body resolve to the OUTER item, so
        // recursion hits the cache, exactly like Python's wrapper.
        // @lru_cache keys must stay concrete (int/bool/str): an inferred
        // generic parameter cannot be hashed into a static key, so it is a
        // loud error (issue #109).
        if cache_spec.is_some() && inferred_signature.is_generic() {
            return Err(
                "functools.lru_cache on a function with unannotated parameters is not \
                 supported: the cache key needs a concrete type. Annotate the \
                 parameters (issue #109)"
                    .to_string()
                    .into(),
            );
        }
        let streams = if let (Some(maxsize), Some(key)) = (cache_spec, cache_key.as_ref()) {
            let maxsize_tokens = match maxsize {
                None => quote!(None),
                Some(n) => quote!(Some(#n as usize)),
            };
            let key_types: Vec<&TokenStream> = key.iter().map(|(_, t)| t).collect();
            let key_names: Vec<&proc_macro2::Ident> = key.iter().map(|(n, _)| n).collect();
            let ret = match self.resolved_return_type() {
                Some(ty) => quote!(#ty),
                None => quote!(()),
            };
            // str parameters arrive as impl Into<String>; normalize them
            // before building the key (the inner fn takes concrete String).
            let mut outer_rebinds = TokenStream::new();
            for (p, (name, _)) in self.args.args.iter().zip(key.iter()) {
                if matches!(p.annotation.as_deref(), Some(ExprType::Name(n)) if n.id == "str")
                {
                    outer_rebinds.extend(quote!(let #name: String = #name.into();));
                }
            }
            quote! {
                #outer_rebinds
                static __LRU_CACHE: std::sync::LazyLock<
                    std::sync::Mutex<PyLruCache<(#(#key_types,)*), #ret>>,
                > = std::sync::LazyLock::new(|| {
                    std::sync::Mutex::new(PyLruCache::new(#maxsize_tokens))
                });
                if let Some(__hit) = __LRU_CACHE
                    .lock()
                    .expect("lru_cache mutex poisoned")
                    .get(&(#((#key_names).clone(),)*))
                {
                    return Ok(__hit);
                }
                #[allow(non_snake_case)]
                fn __lru_uncached(#(#key_names: #key_types),*) -> Result<#ret, PyException> {
                    #streams
                }
                let __lru_value = __lru_uncached(#((#key_names).clone()),*)?;
                __LRU_CACHE
                    .lock()
                    .expect("lru_cache mutex poisoned")
                    .put((#((#key_names).clone(),)*), __lru_value.clone());
                Ok(__lru_value)
            }
        } else {
            streams
        };

        // A definition-time unsatisfiability warning (M5) never blocks
        // conversion; report it through the -W channel (drained by the
        // transpiler) and fold it into the #[deprecated] note.
        if let Some(dw) = &inferred_signature.definition_warning {
            options
                .definition_warnings
                .borrow_mut()
                .push(format!("function `{}`: {}", self.name, dw));
        }
        // Lossy conversions are silent semantic changes callers may not want
        // — surface them as a compiler warning at every call site outside the
        // generated crate via a single #[deprecated] note (the standard
        // mechanism for user-defined warnings). An item can carry only one
        // #[deprecated] attribute, so all notes are folded into it.
        let lossy_warning = if options.lossy_warnings {
            let mut notes = self.lossy_conversion_notes();
            if let Some(dw) = &inferred_signature.definition_warning {
                notes.push(dw.clone());
            }
            if notes.is_empty() {
                quote!()
            } else {
                let note = notes.join("; ");
                quote!(#[deprecated(note = #note)])
            }
        } else {
            quote!()
        };

        let generic_header = inferred_signature.generic_header();
        let where_clause = inferred_signature.where_clause();

        // Render the parameter list now: the inference pass has populated
        // options.param_type_vars, so unannotated parameters render as
        // their inferred type variables (`a: A`).
        let parameters = render_args
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        let function = if let Some(docstring) = self.get_docstring() {
            // Convert docstring to Rust doc comments
            let doc_lines: Vec<_> = docstring
                .lines()
                .map(|line| {
                    if line.trim().is_empty() {
                        quote! { #[doc = ""] }
                    } else {
                        let doc_line = format!("{}", line);
                        quote! { #[doc = #doc_line] }
                    }
                })
                .collect();

            quote! {
                #(#doc_lines)*
                #lossy_warning
                #visibility #is_async fn #fn_name #generic_header(#receiver #parameters) #return_type #where_clause {
                    #streams
                }
            }
        } else {
            quote! {
                #lossy_warning
                #visibility #is_async fn #fn_name #generic_header(#receiver #parameters) #return_type #where_clause {
                    #streams
                }
            }
        };

        debug!("function: {}", function);
        Ok(function)
    }
}

/// Collect every `return` statement's value (None for a bare `return`)
/// from a statement list, recursing into nested control-flow bodies but not
/// into nested function or class definitions.
fn collect_returns<'a>(body: &'a [Statement], out: &mut Vec<Option<&'a ExprType>>) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Return(value) => {
                out.push(value.as_ref().map(|e| &e.value));
            }
            StatementType::If(s) => {
                collect_returns(&s.body, out);
                collect_returns(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_returns(&s.body, out);
                collect_returns(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_returns(&s.body, out);
                collect_returns(&s.orelse, out);
            }
            StatementType::With(s) => collect_returns(&s.body, out),
            StatementType::AsyncWith(s) => collect_returns(&s.body, out),
            StatementType::AsyncFor(s) => collect_returns(&s.body, out),
            StatementType::Try(s) => {
                collect_returns(&s.body, out);
                for handler in &s.handlers {
                    collect_returns(&handler.body, out);
                }
                collect_returns(&s.orelse, out);
                collect_returns(&s.finalbody, out);
            }
            // Nested defs/classes have their own return scopes; everything
            // else contains no return statements we care about.
            _ => {}
        }
    }
}

/// Map an expression to an obviously-inferable Rust type, if any.
pub(crate) fn simple_expr_type(expr: &ExprType) -> Option<TokenStream> {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Some(quote!(i64)),
            Some(litrs::Literal::Float(_)) => Some(quote!(f64)),
            Some(litrs::Literal::Bool(_)) => Some(quote!(bool)),
            // A string constant lowers to a &'static str literal.
            Some(litrs::Literal::String(_)) => Some(quote!(&'static str)),
            _ => None,
        },
        ExprType::JoinedStr(_) => Some(quote!(String)),
        // `"sep".join(iterable)` — and join on any string literal — yields
        // an owned String (str::join / PyStrOps::join). Common idiom
        // (version strings, path joining); the method table omits join for
        // the unannotated-parameter bound, but concrete receivers work.
        ExprType::Call(call) => {
            if let ExprType::Attribute(a) = call.func.as_ref()
                && a.attr == "join"
                && matches!(
                    a.value.as_ref(),
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_)))
                )
            {
                Some(quote!(String))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Issue #110: names bound to a string literal and later rebound by a
/// non-literal (an aug-assign target, or a second assignment whose value
/// is not a string literal). `literal_bindings` maps name → whether the
/// (first) binding was a bare string literal; `rebound` collects names
/// stored to by an aug-assign or a non-literal assignment.
pub(crate) fn scan_str_rebindings(
    body: &[Statement],
    literal_bindings: &mut std::collections::HashMap<String, bool>,
    rebound: &mut std::collections::HashSet<String>,
) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Assign(a) => {
                if let [ExprType::Name(name)] = a.targets.as_slice() {
                    let is_literal = matches!(
                        &a.value,
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::String(_)))
                    );
                    literal_bindings
                        .entry(name.id.clone())
                        .and_modify(|first| *first |= is_literal)
                        .or_insert(is_literal);
                    if !is_literal {
                        rebound.insert(name.id.clone());
                    }
                }
            }
            StatementType::AugAssign(a) => {
                if let ExprType::Name(name) = &a.target {
                    rebound.insert(name.id.clone());
                }
            }
            StatementType::If(s) => {
                scan_str_rebindings(&s.body, literal_bindings, rebound);
                scan_str_rebindings(&s.orelse, literal_bindings, rebound);
            }
            StatementType::For(s) => {
                scan_str_rebindings(&s.body, literal_bindings, rebound);
                scan_str_rebindings(&s.orelse, literal_bindings, rebound);
            }
            StatementType::While(s) => {
                scan_str_rebindings(&s.body, literal_bindings, rebound);
                scan_str_rebindings(&s.orelse, literal_bindings, rebound);
            }
            StatementType::Try(t) => {
                scan_str_rebindings(&t.body, literal_bindings, rebound);
                for h in &t.handlers {
                    scan_str_rebindings(&h.body, literal_bindings, rebound);
                }
                scan_str_rebindings(&t.orelse, literal_bindings, rebound);
                scan_str_rebindings(&t.finalbody, literal_bindings, rebound);
            }
            StatementType::With(w) => scan_str_rebindings(&w.body, literal_bindings, rebound),
            StatementType::AsyncWith(w) => scan_str_rebindings(&w.body, literal_bindings, rebound),
            _ => {}
        }
    }
}

/// Issue #115: a module-level name WRITTEN from this function — a `global
/// x` declaration whose name is an Assign/AugAssign target in the same
/// body. Returns the offending name.
fn global_write_error(body: &[Statement]) -> Option<String> {
    let mut declared: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut written: std::collections::HashSet<String> = std::collections::HashSet::new();
    for stmt in body {
        match &stmt.statement {
            StatementType::Global(names) => {
                declared.extend(names.iter().cloned());
            }
            StatementType::Assign(a) => {
                for target in &a.targets {
                    if let ExprType::Name(n) = target {
                        written.insert(n.id.clone());
                    }
                }
            }
            StatementType::AugAssign(a) => {
                if let ExprType::Name(n) = &a.target {
                    written.insert(n.id.clone());
                }
            }
            _ => {}
        }
    }
    declared.into_iter().find(|n| written.contains(n))
}

/// Collect `name = <simply-typed constant>` assignments (recursing into
/// control-flow bodies) so returns of those names can be inferred too.
pub(crate) fn collect_local_types(
    body: &[Statement],
    out: &mut std::collections::HashMap<String, TokenStream>,
) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Assign(assign) => {
                if let [ExprType::Name(name)] = assign.targets.as_slice() {
                    if let Some(ty) = simple_expr_type(&assign.value) {
                        out.insert(name.id.clone(), ty);
                    }
                }
            }
            StatementType::If(s) => {
                collect_local_types(&s.body, out);
                collect_local_types(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_local_types(&s.body, out);
                collect_local_types(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_local_types(&s.body, out);
                collect_local_types(&s.orelse, out);
            }
            StatementType::With(s) => collect_local_types(&s.body, out),
            _ => {}
        }
    }
}

/// Whether an annotation expression means `None` (`-> None` marks a
/// procedure): the parser may surface it as the NoneType variant, a
/// valueless constant, or the bare name `None`.
pub(crate) fn is_none_expr(ann: &ExprType) -> bool {
    match ann {
        ExprType::NoneType(_) => true,
        ExprType::Constant(c) => c.0.is_none(),
        ExprType::Name(name) => name.id == "None",
        _ => false,
    }
}

/// Whether an expression already lowers to an `Option` value, so a store
/// into an optional-tracked name (or an Optional parameter slot) must NOT
/// wrap it in `Some` — double-wrapping turns an absent value into
/// `Some(None)`, and a later `is None` check silently answers wrongly.
pub(crate) fn expr_yields_option(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match expr {
        // A name that itself holds an Option (assigned None on some path,
        // or an Optional-annotated parameter).
        ExprType::Name(name) => options.optional_names.contains(&name.id),
        ExprType::Call(call) => match call.func.as_ref() {
            // dict.get(k) lowers to py_get, which returns Option<V>.
            ExprType::Attribute(attr) => attr.attr == "get" && call.args.len() == 1,
            // A user function annotated `-> Optional[T]` generates
            // `Result<Option<T>, PyException>`; the call site's `?` strips
            // only the Result layer, leaving an Option.
            ExprType::Name(name) => match symbols.get(&name.id) {
                Some(SymbolTableNode::FunctionDef(f)) => f
                    .returns
                    .as_deref()
                    .is_some_and(crate::is_optional_annotation),
                _ => false,
            },
            _ => false,
        },
        // A conditional yields an Option when either arm does (None counts):
        // the arms unify to one type, so an Option arm makes the whole
        // expression an Option. A plain-vs-Option mix fails to compile —
        // loud, never silent.
        ExprType::IfExp(e) => {
            let arm = |x: &ExprType| {
                crate::is_none_expr(x) || expr_yields_option(x, options, symbols)
            };
            arm(&e.body) || arm(&e.orelse)
        }
        _ => false,
    }
}

/// Lower an expression destined for an Option slot (a store into an
/// optional-tracked name, or an Optional-annotated parameter): values that
/// already yield an Option (and None itself) pass through, plain values
/// wrap in `Some`, and conditionals wrap each arm independently — so
/// `x if c else None` becomes `if c { Some(x) } else { None }` instead of
/// burying the None arm inside `Some(...)`.
pub(crate) fn lower_optional_value(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // Conditionals recurse per arm FIRST: even when one arm makes the whole
    // expression Option-typed (e.g. an `else None`), the other arm may be a
    // plain value that still needs its Some wrap.
    if let ExprType::IfExp(e) = expr {
        let test =
            crate::condition_to_rust(&e.test, ctx.clone(), options.clone(), symbols.clone())?;
        let body = lower_optional_value(&e.body, ctx.clone(), options.clone(), symbols.clone())?;
        let orelse = lower_optional_value(&e.orelse, ctx, options, symbols)?;
        return Ok(quote!(if #test { #body } else { #orelse }));
    }
    if is_none_expr(expr) || expr_yields_option(expr, &options, &symbols) {
        return expr.clone().to_rust(ctx, options, symbols);
    }
    let tokens = expr.clone().to_rust(ctx, options, symbols)?;
    Ok(quote!(Some(#tokens)))
}

/// Best-effort Python-source rendering of an annotation expression, for
/// warning messages.
pub(crate) fn annotation_display(ann: &ExprType) -> String {
    match ann {
        ExprType::Name(name) => name.id.clone(),
        ExprType::Constant(c) => c.to_string(),
        _ => "<annotation>".to_string(),
    }
}

/// Whether a statement list is guaranteed to return a value on every
/// control-flow path: its final statement is a `return <value>`, an
/// `if`/`else` whose branches both guarantee a return, or a diverging
/// `raise`. Loops and other constructs may fall through, so they never
/// guarantee a return.
pub(crate) fn guarantees_return(body: &[Statement]) -> bool {
    match body.last().map(|stmt| &stmt.statement) {
        Some(StatementType::Return(Some(_))) => true,
        Some(StatementType::If(s)) => {
            !s.orelse.is_empty() && guarantees_return(&s.body) && guarantees_return(&s.orelse)
        }
        // `raise` lowers to `return Err(...)`, which terminates the path.
        Some(StatementType::Raise(_)) => true,
        // A try guarantees a return when its no-exception path does (the
        // body, or the else clause the body falls into) and every handler
        // does too — or when the finally clause returns unconditionally.
        // Unhandled exceptions exit via Err, which also terminates.
        Some(StatementType::Try(t)) => {
            let normal_path = if t.orelse.is_empty() {
                guarantees_return(&t.body)
            } else {
                guarantees_return(&t.body) || guarantees_return(&t.orelse)
            };
            let handlers = t.handlers.iter().all(|h| guarantees_return(&h.body));
            (normal_path && handlers) || guarantees_return(&t.finalbody)
        }
        _ => false,
    }
}

impl FunctionDef {
    /// The return type the generated Rust function actually carries, if any.
    ///
    /// Inference from the body comes first (it reflects the type the body
    /// actually produces — e.g. a string literal is a &'static str even
    /// under a `-> str` annotation); an explicit annotation with a known
    /// Rust mapping is the fallback for bodies inference can't see through.
    /// Both require the body to return on every path: a fall-through path
    /// yields `()`, which no concrete annotation can type. `-> None` and
    /// unmappable annotations yield None.
    ///
    /// Tools generating call-through code (e.g. PyO3 wrappers) must use this
    /// same method so their signatures match the generated function.
    pub fn resolved_return_type(&self) -> Option<TokenStream> {
        // A bare `str` annotation is authoritative: the inferred type for a
        // literal-returning body (`&'static str`) is a Rust literal artifact,
        // not the Python type, and the mismatch breaks every call site
        // (`return self.speak()` where speak returns a literal would hand
        // `&str` to a `-> str` context). The annotation is the contract.
        if guarantees_return(&self.body)
            && matches!(
                self.returns.as_deref(),
                Some(ExprType::Name(n)) if n.id == "str"
            )
        {
            return Some(quote!(String));
        }
        let annotated = if guarantees_return(&self.body) {
            self.returns.as_deref().and_then(|ann| {
                if is_none_expr(ann) {
                    None
                } else {
                    crate::python_annotation_to_rust_type(ann)
                }
            })
        } else {
            None
        };
        self.inferred_return_type().or(annotated)
    }

    /// The Python-source text of a return annotation the generated function
    /// does not honor: the body can fall through (implicitly returning
    /// None), so the generated function returns `()` no matter what the
    /// annotation claims. This frequently marks a bug in the Python source
    /// — the author declared a return type but not every path returns one —
    /// so it must be surfaced, not silently reproduced.
    pub fn ignored_return_annotation(&self) -> Option<String> {
        let ann = self.returns.as_deref()?;
        if is_none_expr(ann) || guarantees_return(&self.body) {
            return None;
        }
        Some(annotation_display(ann))
    }

    /// Human-readable notes for every lossy conversion this function's
    /// signature underwent. These become the #[deprecated] note on the
    /// generated function, and conversion tools report them to the user.
    pub fn lossy_conversion_notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        let dropped = self.dropped_default_parameters();
        if !dropped.is_empty() {
            notes.push(format!(
                "rython: Python default value(s) for parameter(s) `{}` were dropped \
                 (Rust has no default arguments); every argument must be passed explicitly",
                dropped.join("`, `")
            ));
        }
        if let Some(ann) = self.ignored_return_annotation() {
            notes.push(format!(
                "rython: the `-> {}` return annotation was ignored because the function \
                 body does not return a value on every path; the generated function \
                 returns `()` where Python would implicitly return None",
                ann
            ));
        }
        notes
    }

    /// Names of parameters whose Python default values cannot be carried
    /// into the generated Rust signature (Rust has no default arguments).
    /// Used to attach a call-site warning to the generated function and to
    /// let tools report the loss during conversion.
    pub fn dropped_default_parameters(&self) -> Vec<String> {
        let mut dropped = Vec::new();
        let defaults_offset = self
            .args
            .args
            .len()
            .saturating_sub(self.args.defaults.len());
        for arg in &self.args.args[defaults_offset..] {
            dropped.push(arg.arg.clone());
        }
        for (i, arg) in self.args.kwonlyargs.iter().enumerate() {
            if self.args.kw_defaults.get(i).is_some_and(Option::is_some) {
                dropped.push(arg.arg.clone());
            }
        }
        dropped
    }

    /// Infer a return type when the function is guaranteed to return on
    /// every control-flow path AND every return value in the body maps to
    /// the same simple type — either directly (a constant or f-string) or
    /// via a local variable assigned a constant. Partial/conditional
    /// returns (which implicitly return None on the fall-through path),
    /// mixed types, and uninferable values all yield None so the function
    /// stays unannotated, as before.
    pub fn inferred_return_type(&self) -> Option<TokenStream> {
        // A function that can fall off the end must not get a concrete
        // return annotation: the implicit tail is `()`.
        if !guarantees_return(&self.body) {
            return None;
        }

        let mut returns = Vec::new();
        collect_returns(&self.body, &mut returns);

        let mut locals = std::collections::HashMap::new();
        collect_local_types(&self.body, &mut locals);
        // Issue #110: a string-literal local that is later REBOUND by a
        // String (`out = ""; out += "x"`) is owned from its first
        // assignment, so its type is String, not &'static str.
        let mut literal_bindings: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        let mut rebound: std::collections::HashSet<String> = std::collections::HashSet::new();
        scan_str_rebindings(&self.body, &mut literal_bindings, &mut rebound);

        let mut inferred: Option<TokenStream> = None;
        for ret in &returns {
            let value = (*ret)?; // a bare `return` means the type is unit
            let ty = match value {
                ExprType::Name(name) => {
                    let t = locals.get(&name.id)?.clone();
                    if t.to_string() == quote!(&'static str).to_string()
                        && rebound.contains(&name.id)
                    {
                        quote!(String)
                    } else {
                        t
                    }
                }
                other => simple_expr_type(other)?,
            };
            match &inferred {
                None => inferred = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                _ => return None,
            }
        }
        inferred
    }
}

impl FunctionDef {
    fn get_docstring(&self) -> Option<String> {
        if self.body.is_empty() {
            return None;
        }
        
        let expr = self.body[0].clone();
        match expr.statement {
            StatementType::Expr(e) => match e.value {
                ExprType::Constant(c) => {
                    let raw_string = c.to_string();
                    // Clean up the docstring for Rust documentation
                    Some(self.format_docstring(&raw_string))
                },
                _ => None,
            },
            _ => None,
        }
    }
    
    fn format_docstring(&self, raw: &str) -> String {
        // Remove surrounding quotes
        let content = raw.trim_matches('"');
        
        // Split into lines and clean up Python-style indentation
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return String::new();
        }
        
        // First line is usually the summary
        let mut formatted = vec![lines[0].trim().to_string()];
        
        if lines.len() > 1 {
            // Add empty line after summary if there are more lines
            if !lines[0].trim().is_empty() && !lines[1].trim().is_empty() {
                formatted.push(String::new());
            }
            
            // Process remaining lines, cleaning up indentation
            for line in lines.iter().skip(1) {
                let cleaned = line.trim();
                if cleaned.starts_with("Args:") {
                    formatted.push(String::new());
                    formatted.push("# Arguments".to_string());
                } else if cleaned.starts_with("Returns:") {
                    formatted.push(String::new());
                    formatted.push("# Returns".to_string());
                } else if cleaned.starts_with("Example:") {
                    formatted.push(String::new());
                    formatted.push("# Examples".to_string());
                } else if cleaned.starts_with(">>>") {
                    // Convert Python examples to Rust doc test format
                    formatted.push(format!("```rust"));
                    formatted.push(format!("// {}", cleaned));
                } else if !cleaned.is_empty() {
                    formatted.push(cleaned.to_string());
                }
            }
            
            // Close any open code blocks
            if content.contains(">>>") {
                formatted.push("```".to_string());
            }
        }
        
        formatted.join("\n")
    }
}

impl Object for FunctionDef {}
