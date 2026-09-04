use tracing::debug;
use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};
use crate::ast::tree::statement::PyStatementTrait;
use crate::ast::tree::visit::{
    self, Descend, Flow, any_expr_in, stmt_all_exprs, stmt_targets, walk_stmts,
};

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
pub(crate) struct ArgparseSpec {
    name: String,
    /// The short alias of `add_argument("-c", "--contents", ...)`
    /// (issue #118 — certifi's __main__); None otherwise.
    short: Option<String>,
    kind: &'static str, // "Str" | "Int" | "Float" | "StoreTrue"
    default: Option<ExprType>,
    help: Option<String>,
}

/// The argparse rewrite plan for a function body: parser-building
/// statements to drop, the parse_args assignment to replace, and the
/// literal specs. ArgumentParser/add_argument/parse_args are evaluated
/// HERE, at conversion time — only literal specs can shape the typed
/// namespace struct, so anything dynamic is a loud error.
pub(crate) struct ArgparseRewrite {
    pub(crate) skip: std::collections::HashSet<usize>,
    pub(crate) parse_index: usize,
    pub(crate) args_var: String,
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

pub(crate) fn scan_argparse(
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
            && matches!(attr.value.as_ref(), ExprType::Name(m)
                if crate::StdModule::from_name(&m.id) == Some(crate::StdModule::Argparse));
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
                // One name (`"count"`, `"--verbose"`), or a short+long
                // alias pair (`"-c", "--contents"` — certifi, issue #118).
                let (short, name) = match call.args.as_slice() {
                    [name_expr] => {
                        let name = literal_str(name_expr)
                            .ok_or("add_argument: the name must be a string literal")?;
                        if name.starts_with('-') && !name.starts_with("--") {
                            return Err(format!(
                                "add_argument: short option '{}' needs a --long \
                                 alias (the long name is the namespace attribute)",
                                name
                            )
                            .into());
                        }
                        (None, name)
                    }
                    [short_expr, long_expr] => {
                        let short = literal_str(short_expr)
                            .ok_or("add_argument: the name must be a string literal")?;
                        let long = literal_str(long_expr)
                            .ok_or("add_argument: the name must be a string literal")?;
                        if !short.starts_with('-')
                            || short.starts_with("--")
                            || !long.starts_with("--")
                        {
                            return Err(format!(
                                "add_argument('{}', '{}'): a two-name argument must \
                                 be a -short, --long alias pair",
                                short, long
                            )
                            .into());
                        }
                        (Some(short), long)
                    }
                    _ => {
                        return Err(
                            "add_argument takes one name, or a -short, --long alias \
                             pair"
                                .into(),
                        );
                    }
                };
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
                    short,
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
pub(crate) fn lower_parse_args(
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
        let short = match &spec.short {
            Some(s) => quote!(Some(#s)),
            None => quote!(None),
        };
        spec_tokens.push(quote!(argparse::ArgSpec {
            name: #name,
            short: #short,
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
/// What a function's decorators ask for, parsed through the systematic
/// decorator registry (decorator.rs): `@classmethod`/`@staticmethod`
/// (issue #117) make the method an ASSOCIATED function — no receiver; a
/// classmethod additionally drops its first parameter (cls/self — the
/// class reference). Anything else is a loud error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MethodDecorator {
    None,
    Cache(Option<Option<i64>>),
    ClassMethod,
    StaticMethod,
}

/// Parse a FUNCTION's decorator list through the one registry. Only the
/// method-shape and cache decorators apply to functions; a `@dataclass`
/// (or anything unknown) is a loud error here (class codegen handles
/// `@dataclass`).
fn parse_method_decorator(
    decorators: &[ExprType],
    symbols: &SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<MethodDecorator, Box<dyn std::error::Error>> {
    // A bare-Name decorator bound to a LOCAL FunctionDef (same module or
    // imported from a sibling module of the crate — `@_text_content`,
    // `@instance_cache` — botocore, `@with_cleanup` — pip): a hand-rolled
    // wrapper (caching, text extraction, cleanup) — unmodeled; the
    // decorated function lowers directly (documented divergence).
    let is_local_fn = |name: &str| -> bool {
        match symbols.get(name) {
            Some(SymbolTableNode::FunctionDef(_)) => true,
            Some(SymbolTableNode::ImportFrom(i)) => {
                let path = i.resolved_module_path(options);
                // `module_defs` is keyed RELATIVE to the package root for
                // src-layout packages (pip/urllib3), while an absolute
                // import (`from pip._internal.cli.req_command import ...`)
                // resolves to a root-qualified path — match both forms.
                let Some(key) = crate::module_defs_key(options, &path) else {
                    return false;
                };
                options.module_defs.get(key).is_some_and(|m| {
                    let m: &crate::Module = m;
                    m.raw.body.iter().any(|s| {
                        matches!(
                            &s.statement,
                            crate::StatementType::FunctionDef(f) if f.name == name
                        )
                    })
                })
            }
            _ => false,
        }
    };
    if decorators.iter().any(|d| {
        match d {
            ExprType::Name(n) => is_local_fn(&n.id),
            // A CALL to a local decorator factory (`@retry(stop_after_delay
            // =3, wait=0.5)` — pip's misc): the factory's wrapper is
            // unmodeled — the decorated function lowers directly.
            ExprType::Call(c) => match c.func.as_ref() {
                ExprType::Name(n) => is_local_fn(&n.id),
                _ => false,
            },
            _ => false,
        }
    }) {
        return Ok(MethodDecorator::None);
    }
    match crate::parse_decorator(decorators)? {
        None => Ok(MethodDecorator::None),
        // Issue #181: the singledispatch family is fused at MODULE level.
        // Reaching here means the definition is not a module-level one —
        // nested in a function, a class, or a conditional block — where
        // the family cannot be assembled. Say that, rather than the
        // generic "does not apply" message.
        Some(d @ (crate::Decorator::SingleDispatch | crate::Decorator::Register { .. })) => {
            Err(format!(
                "`{}` is only supported on a MODULE-LEVEL definition: rython fuses the \
                 generic and its registrations into one dispatching function, which it \
                 can only do when the whole family sits at module level",
                d.describe()
            )
            .into())
        }
        Some(d) => d.as_method_decorator().ok_or_else(|| {
            format!(
                "decorator `{}` does not apply to a function definition (only \
                 functools.lru_cache, functools.cache, classmethod, and staticmethod do)",
                d.describe()
            )
            .into()
        }),
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
        // A module function registered for isinstance specialization
        // (specialize.rs) renders as its variants + residual instead of
        // one generic definition. Methods and nested defs never register.
        let result = if !options.rendering_specialization
            && matches!(ctx, CodeGenContext::Module(_))
            && options.specialized_fns.contains_key(&self.name)
        {
            let spec = options.specialized_fns.get(&self.name).unwrap().clone();
            self.render_specializations(spec, ctx, options, symbols)
        } else {
            self.to_rust_inner(ctx, options, symbols)
        };
        return result;
    }
}

impl FunctionDef {
    /// Emit the monomorphized variants of an isinstance-dispatched
    /// function (specialize.rs): one definition per tested type with the
    /// axis parameter ANNOTATED as that type — the isinstance checks fold
    /// to constants through the inheritance tree and dead arms are pruned
    /// before rendering — plus the `__any` residual, whose axis stays
    /// generic and whose tested arms are removed at the AST level.
    fn render_specializations(
        self,
        spec: crate::ast::tree::specialize::SpecializedFn,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut out = TokenStream::new();
        let mut vopts = options.clone();
        vopts.rendering_specialization = true;
        use crate::ast::tree::specialize::{
            fold_morph_body, mangled_name, target_typeinfo, SpecializedFn,
        };
        // One morph per assignment in the cartesian product over axes:
        // per tested BUILTIN type and per concrete class in the tested
        // subtrees (a Cat argument gets describe_cat with `x: Cat`,
        // keeping Cat's own overrides — Rust structs have no subtyping
        // to flow through an Animal-typed variant), plus Any per axis;
        // the all-Any assignment is the residual.
        let assignments = spec.morph_assignments();
        for assignment in &assignments {
            let suffix = SpecializedFn::assignment_suffix(assignment);
            let mut variant = self.clone();
            variant.name = mangled_name(&self.name, &suffix);
            let mut seeds: Vec<(String, crate::TypeInfo)> = Vec::new();
            let mut unassigned: Vec<String> = Vec::new();
            for (axis, a) in spec.axes.iter().zip(assignment) {
                match a {
                    Some(target) => {
                        variant.args.args[axis.index].annotation = Some(Box::new(
                            crate::ExprType::Name(crate::ast::tree::name::Name {
                                id: target.suffix().to_string(),
                            }),
                        ));
                        if let Some(ti) = target_typeinfo(target) {
                            seeds.push((axis.name.clone(), ti));
                        }
                    }
                    None => unassigned.push(axis.name.clone()),
                }
            }
            variant.body = fold_morph_body(&self.body, &spec, assignment, &symbols);
            // The original carries no return annotation (its tested
            // params were unannotated); the folded morph needs one so
            // the annotated machinery types the returns. An Any axis has
            // no seed — derivation then fails for returns involving it,
            // and the ordinary inference machinery takes over.
            if variant.returns.is_none() {
                variant.returns = crate::ast::tree::specialize::derive_return_annotation(
                    &variant.body,
                    &seeds,
                    &options,
                    &symbols,
                )
                .map(Box::new);
            }
            crate::ast::tree::specialize::underscore_unused_params(&mut variant);
            let mut mopts = vopts.clone();
            if !unassigned.is_empty() {
                // Any leftover isinstance render on an Any axis (a shape
                // the fold does not cover) lowers to false — the residual
                // semantics for that axis.
                let mut fold = options.residual_fold_false.as_ref().clone();
                fold.extend(unassigned);
                mopts.residual_fold_false = std::rc::Rc::new(fold);
            }
            out.extend(variant.to_rust(ctx.clone(), mopts, symbols.clone())?);
        }

        // ---- The dynamic router ----
        // A closed-world argument enum (one variant per morph, plus a
        // boxed Other) and a function under the ORIGINAL Python name that
        // matches on it: runtime dispatch over the compile-time morphs,
        // for boxed values and for Rust callers with runtime-varying
        // data. Emitted only when every morph derived the same return
        // type (planned at detection time).
        if let Some(router) = &spec.router
            && options.with_std_python
        {
            use crate::ast::tree::specialize::{
                py_id_boxable, py_type_tokens, to_pascal, RouterReturn, SpecTarget,
            };
            use quote::format_ident;
            let orig_ident = crate::safe_ident(&self.name);
            let enum_idents: Vec<proc_macro2::Ident> = router
                .enum_names
                .iter()
                .map(|n| crate::safe_ident(n))
                .collect();
            let variant_ident = |t: &SpecTarget| -> proc_macro2::Ident {
                crate::safe_ident(&match t {
                    SpecTarget::Builtin(b) => to_pascal(b),
                    SpecTarget::Class(c) => c.clone(),
                })
            };

            // The non-axis parameters pass through positionally: an int
            // extra is `n: i64`, a str extra keeps the annotated-fn shape
            // `s: impl Into<String>` so string literals still call
            // directly, and morphs (whose annotated str parameters take
            // the same shape) receive the generic value unchanged. Each
            // AXIS parameter takes `impl Into<{F}ArgN>`, and the arm
            // forwards its binding v1/v2/...
            let extra_param_ty = |id: &str| -> TokenStream {
                match id {
                    "str" => quote!(impl Into<String>),
                    other => py_type_tokens(other),
                }
            };
            let mut sig_params = TokenStream::new();
            let mut forward_args: Vec<TokenStream> = Vec::new();
            // Which axis (if any) each forwarded argument is, in order.
            let mut forward_axis: Vec<Option<usize>> = Vec::new();
            let mut axis_arg_idents: Vec<proc_macro2::Ident> = Vec::new();
            for (i, p) in self.args.args.iter().enumerate() {
                if let Some(k) = spec.axes.iter().position(|a| a.index == i) {
                    let ident = crate::safe_ident(&p.arg);
                    let e = &enum_idents[k];
                    sig_params.extend(quote!(#ident: impl Into<#e>,));
                    axis_arg_idents.push(ident);
                    let v = format_ident!("v{}", k + 1);
                    forward_args.push(quote!(#v));
                    forward_axis.push(Some(k));
                    continue;
                }
                let (_, name, id) = router
                    .extra_params
                    .iter()
                    .find(|(idx, _, _)| *idx == i)
                    .expect("plan_router covered every non-axis parameter");
                let ident = crate::safe_ident(name);
                let ty = extra_param_ty(id);
                sig_params.extend(quote!(#ident: #ty,));
                forward_args.push(quote!(#ident));
                forward_axis.push(None);
            }

            // The return shape: the unified type, or the OUTPUT enum
            // (one variant per distinct morph return type, From<T> per
            // member) — a runtime-dispatched result then lands as a
            // matchable value, and as a boxed PyValue when every member
            // is boxable (Python's `str | int` union).
            let (ret_ty, out_enum_stream, wrap_result) = match &router.ret {
                RouterReturn::Unified(id) => {
                    (py_type_tokens(id), TokenStream::new(), false)
                }
                RouterReturn::Enum { name, members } => {
                    let out_ident = crate::safe_ident(name);
                    let mut out_variants = TokenStream::new();
                    let mut out_from = TokenStream::new();
                    let mut pv_arms = TokenStream::new();
                    for m in members {
                        let var = crate::safe_ident(&if py_id_boxable(m) {
                            to_pascal(m)
                        } else {
                            m.clone()
                        });
                        let ty = py_type_tokens(m);
                        out_variants.extend(quote!(#var(#ty),));
                        out_from.extend(quote! {
                            impl From<#ty> for #out_ident {
                                fn from(v: #ty) -> Self {
                                    #out_ident::#var(v)
                                }
                            }
                        });
                        pv_arms.extend(quote! {
                            #out_ident::#var(v) => stdpython::PyValue::from(v),
                        });
                    }
                    // Every member boxable → the result converts to the
                    // boxed PyValue, which is exactly Python's union
                    // return — generated call sites consume it that way.
                    let pv_impl = if members.iter().all(|m| py_id_boxable(m)) {
                        quote! {
                            impl From<#out_ident> for stdpython::PyValue {
                                fn from(v: #out_ident) -> Self {
                                    match v { #pv_arms }
                                }
                            }
                        }
                    } else {
                        TokenStream::new()
                    };
                    let out_doc = format!(
                        "The result of `{}`'s dynamic router: one variant \
                         per distinct morph return type.",
                        self.name
                    );
                    let stream = quote! {
                        #[doc = #out_doc]
                        #[derive(Clone)]
                        pub enum #out_ident {
                            #out_variants
                        }
                        #out_from
                        #pv_impl
                    };
                    (quote!(#out_ident), stream, true)
                }
            };

            // One argument enum PER AXIS, each with its variants plus
            // `Other(PyValue)`, From<T> per variant, and the boxed
            // From<PyValue> routing.
            let mut axis_enum_streams = TokenStream::new();
            for (k, axis) in spec.axes.iter().enumerate() {
                let enum_ident = &enum_idents[k];
                let mut enum_variants = TokenStream::new();
                let mut from_impls = TokenStream::new();
                let mut boxed_arms = TokenStream::new();
                let mut has_bool_morph = false;
                let mut has_int_morph = false;
                for target in axis.variants() {
                    let var_ident = variant_ident(&target);
                    let ty = py_type_tokens(target.suffix());
                    enum_variants.extend(quote!(#var_ident(#ty),));
                    from_impls.extend(quote! {
                        impl From<#ty> for #enum_ident {
                            fn from(v: #ty) -> Self {
                                #enum_ident::#var_ident(v)
                            }
                        }
                    });
                    if let SpecTarget::Builtin(b) = &target {
                        match b.as_str() {
                            "bool" => has_bool_morph = true,
                            "int" => has_int_morph = true,
                            _ => {}
                        }
                        let pv_variant = crate::safe_ident(&to_pascal(b));
                        boxed_arms.extend(quote! {
                            stdpython::PyValue::#pv_variant(v) =>
                                #enum_ident::#var_ident(v),
                        });
                        if b == "str" {
                            // Convenience for Rust callers with &str
                            // literals.
                            from_impls.extend(quote! {
                                impl From<&str> for #enum_ident {
                                    fn from(v: &str) -> Self {
                                        #enum_ident::#var_ident(v.to_string())
                                    }
                                }
                            });
                        }
                    }
                }
                // bool ⊂ int is handled at DETECTION time: an int-tested
                // axis always carries a bool morph of its own
                // (detect_specializable), so a boxed bool routes to a
                // genuine bool-typed body and `str(x)` renders
                // True/False exactly like CPython.
                debug_assert!(
                    has_bool_morph || !has_int_morph,
                    "an int morph implies a bool morph (detect_specializable)"
                );
                let enum_doc = format!(
                    "The dispatch argument for `{}`'s parameter `{}`: one \
                     variant per compile-time morph, `Other` for \
                     everything else.",
                    self.name, axis.name
                );
                axis_enum_streams.extend(quote! {
                    #[doc = #enum_doc]
                    #[derive(Clone)]
                    pub enum #enum_ident {
                        #enum_variants
                        Other(stdpython::PyValue),
                    }
                    #from_impls
                    impl #enum_ident {
                        /// Route a BOXED runtime value to its morph.
                        pub fn from_py_value(v: stdpython::PyValue) -> Self {
                            match v {
                                #boxed_arms
                                other => #enum_ident::Other(other),
                            }
                        }
                    }
                    impl From<stdpython::PyValue> for #enum_ident {
                        fn from(v: stdpython::PyValue) -> Self {
                            Self::from_py_value(v)
                        }
                    }
                });
            }

            // The dispatch match: one arm per morph assignment (the
            // cartesian product covers every combination, so the match
            // is exhaustive without a wildcard).
            let arm_body = |call: TokenStream| -> TokenStream {
                if wrap_result {
                    quote!(Ok(#ret_ty::from((#call)?)))
                } else {
                    call
                }
            };
            let mut match_arms = TokenStream::new();
            for assignment in &assignments {
                let suffix = SpecializedFn::assignment_suffix(assignment);
                let morph = crate::safe_ident(&mangled_name(&self.name, &suffix));
                let pats: Vec<TokenStream> = assignment
                    .iter()
                    .enumerate()
                    .map(|(k, a)| {
                        let e = &enum_idents[k];
                        let v = format_ident!("v{}", k + 1);
                        match a {
                            Some(t) => {
                                let var = variant_ident(t);
                                quote!(#e::#var(#v))
                            }
                            None => quote!(#e::Other(#v)),
                        }
                    })
                    .collect();
                // A morph for a polymorphic ROOT (hierarchy.rs) takes the
                // root's sum type; the axis enum's variant holds the
                // concrete struct — it converts on the way in.
                let args: Vec<TokenStream> = forward_args
                    .iter()
                    .zip(forward_axis.iter())
                    .map(|(fa, ax)| match ax {
                        Some(k) => match &assignment[*k] {
                            Some(SpecTarget::Class(c))
                                if crate::ast::tree::hierarchy::is_polymorphic_root(c) =>
                            {
                                quote!((#fa).into())
                            }
                            _ => fa.clone(),
                        },
                        None => fa.clone(),
                    })
                    .collect();
                let call = arm_body(quote!(#morph(#(#args),*)));
                if pats.len() == 1 {
                    let p = &pats[0];
                    match_arms.extend(quote!(#p => #call,));
                } else {
                    match_arms.extend(quote!((#(#pats),*) => #call,));
                }
            }
            let scrutinee = if axis_arg_idents.len() == 1 {
                let a = &axis_arg_idents[0];
                quote!(#a.into())
            } else {
                let parts = axis_arg_idents.iter().map(|a| quote!(#a.into()));
                quote!((#(#parts),*))
            };

            let router_doc = format!(
                "Dynamic router for `{}`: dispatches runtime-typed \
                 argument(s) to the compile-time morphs, in Python's \
                 first-true-test order per parameter.",
                self.name
            );
            out.extend(quote! {
                #axis_enum_streams
                #out_enum_stream
                #[doc = #router_doc]
                pub fn #orig_ident(
                    #sig_params
                ) -> Result<#ret_ty, PyException> {
                    match #scrutinee {
                        #match_arms
                    }
                }
            });
        }
        Ok(out)
    }

    fn to_rust_inner(
        self,
        ctx: CodeGenContext,
        options: PythonOptions,
        symbols: SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut streams = TokenStream::new();
        let fn_name = crate::safe_ident(&self.name);

        // Issue #119: a MODULE-level `__getattr__` / `__dir__` implements
        // the module attribute protocol (PEP 562) — a dynamic fallback for
        // attribute reads that rython cannot model (module attributes
        // resolve statically at conversion time). Lowering the definition
        // as an ordinary function produces dead code that misstates the
        // module's behavior, so the definition is a loud error naming the
        // dunder and the fix.
        let is_module_ctx = matches!(&ctx, CodeGenContext::Module(_))
            || matches!(
                &ctx,
                CodeGenContext::Async(inner)
                    if matches!(inner.as_ref(), CodeGenContext::Module(_))
            );
        if is_module_ctx && (self.name == "__getattr__" || self.name == "__dir__") {
            return Err(format!(
                "module-level `{}` implements the module attribute protocol \
                 (PEP 562), which is not supported yet: rython resolves module \
                 attributes statically at conversion time, so the dynamic \
                 fallback could never run; define or import the module's \
                 attributes explicitly and remove the `{}` definition — rython \
                 refuses to silently ignore it (issue #119)",
                self.name, self.name
            )
            .into());
        }

        // A `@typing.overload` STUB (`def f(x: int = ...) -> int: ...`) is
        // compile-time metadata: its `...` defaults and `...` body are
        // placeholders, never runtime code. Skip it entirely — call sites
        // resolve the real implementation below the stubs (see
        // module_function_def), and emitting the stub would fail on the
        // Ellipsis defaults.
        if self.decorator_list.iter().any(|d| match d {
            crate::ExprType::Name(n) => n.id == "overload",
            crate::ExprType::Attribute(a) => a.attr == "overload",
            _ => false,
        }) {
            return Ok(TokenStream::new());
        }

        // Issue #115: `global x` declares module scope. Reads resolve to
        // the module statics. A written global whose module binding
        // qualified as a MUTABLE static (module.rs's
        // module_global_mutable_names: a single scalar/None module store)
        // writes THROUGH the static — py_global_write — with no warning.
        // A write to any OTHER global keeps the documented divergence: the
        // write is a no-op, surfaced through the -W channel.
        if let Some(name) = global_write_error(&self.body)
            && !options.mutable_statics.contains_key(&name)
        {
            options.definition_warnings.borrow_mut().push(format!(
                "function `{}` writes to module-level name `{name}`: the write is \
                 dropped (the §5.1 boxed-global divergence — rython has no \
                 mutable module state visible to functions)",
                self.name
            ));
        }
        // The `global` names this function declares that ARE mutable
        // statics: its stores route through py_global_write, and their
        // bindings are the module statics — never hoisted locals.
        let fn_mutable_globals: std::collections::HashSet<String> =
            collect_global_decls(&self.body)
                .into_iter()
                .filter(|n| options.mutable_statics.contains_key(n))
                .collect();
        // Issue #112: `del name` lowers to a no-op, which is faithful ONLY
        // while the name is never referenced afterwards — this pass makes
        // any such use a loud error (and a reassignment/import clears the
        // deletion, like Python's `del x; x = 1`).
        check_deleted_names(&self.body)
            .map_err(|e| format!("function `{}`: {}", self.name, e))?;

        // An argparse parser in the body is evaluated at conversion time:
        // its statements vanish and parse_args becomes a typed struct.
        let argparse_rewrite = scan_argparse(&self.body)?;
        let mut effective_body: Vec<Statement> = match &argparse_rewrite {
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
        // @classmethod/@staticmethod change the method shape; any OTHER
        // decorator is a loud error (see parse_method_decorator).
        let decorator = parse_method_decorator(&self.decorator_list, &symbols, &options)?;
        let mut cache_spec = match decorator {
            MethodDecorator::Cache(spec) => spec,
            _ => None,
        };
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
        // A MODULE-level `_name` function (`_wrap_proxy_error` — urllib3's
        // connection.py) is still imported by SIBLING modules
        // (`use crate::urllib3::connection::_wrap_proxy_error`), so it must
        // be crate-visible; a class method's underscore-prefix privacy
        // stays (methods are reached through the trait or the receiver).
        // The module's own context reaches module-level functions as
        // Module (or Async(Module)) — not `Function { class: None }` —
        // so those contexts take the same module-level path.
        let is_module_level = matches!(&ctx, CodeGenContext::Function { class: None })
            || matches!(&ctx, CodeGenContext::Module(_))
            || matches!(
                &ctx,
                CodeGenContext::Async(inner)
                    if matches!(inner.as_ref(), CodeGenContext::Module(_))
            );
        let visibility = if matches!(&ctx, CodeGenContext::Trait { .. }) {
            quote!()
        } else if is_module_level && self.name.starts_with("_") && !self.name.starts_with("__") {
            quote!(pub(crate))  // module-level: sibling modules import it
        } else if is_module_level {
            quote!(pub)
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

        // A @classmethod's body references the dropped class parameter
        // (`cls.DEFAULT`, `cls(...)` — urllib3's Retry.from_int): bind `cls`
        // to the enclosing class's ClassDef so attribute reads resolve to
        // class constants and calls resolve to construction — the class
        // reference is a compile-time value (issue #117).
        if matches!(decorator, MethodDecorator::ClassMethod)
            && let Some(class_name) = ctx.enclosing_class_name()
            && let Some(crate::SymbolTableNode::ClassDef(class)) =
                symbols.get(class_name)
        {
            symbols.insert(
                "cls".to_string(),
                crate::SymbolTableNode::ClassDef((*class).clone()),
            );
        }

        // A `def` in a class body is an instance method: its first
        // positional parameter is the RECEIVER — `self` becomes the Rust
        // receiver instead of a parameter, `&mut self` when the method
        // stores through `self`, directly or by calling another method of
        // the class that does. Python binds the instance to the first
        // parameter whatever its name (boto3's `factory_self`), so ANY
        // leading positional parameter counts; a method with no parameters
        // is a static-style function callable via `self` (a silent
        // divergence). In a Trait context the method is emitted as a trait
        // item (a default in the class's trait, or an override in an
        // ancestor's trait's impl).
        let is_class_method = matches!(decorator, MethodDecorator::ClassMethod);
        let is_static_method = matches!(decorator, MethodDecorator::StaticMethod);
        // @classmethod/@staticmethod are ASSOCIATED functions: no receiver
        // (issue #117). A classmethod's first parameter is the class
        // reference (cls/self) and is dropped.
        let is_method = !is_class_method
            && !is_static_method
            && matches!(
                &ctx,
                CodeGenContext::Class(_) | CodeGenContext::Trait { .. }
            )
            && (self.args.posonlyargs.first().is_some() || self.args.args.first().is_some());
        let mut render_args = self.args.clone();
        if is_class_method {
            // Drop the class-reference parameter.
            if !render_args.posonlyargs.is_empty() {
                render_args.posonlyargs.remove(0);
            } else if !render_args.args.is_empty() {
                render_args.args.remove(0);
            }
        }
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
            // Python binds the instance to the first parameter whatever its
            // name (boto3's `factory_self`); the Rust receiver is always
            // `self`, and codegen special-cases that literal name, so body
            // references to the original parameter are rewritten first
            // (issue #132).
            let receiver_name = self
                .args
                .posonlyargs
                .first()
                .or(self.args.args.first())
                .map(|p| p.arg.clone());
            if let Some(r) = receiver_name {
                if r != "self" {
                    effective_body = super::rename::rename_receiver_in_body(
                        &effective_body,
                        &r,
                        "self",
                    )
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                }
            }
        }

        // A cached function's arguments form the cache KEY, so every
        // parameter needs a hashable, nameable type: int, bool, or str
        // (floats are not hashable in Rust — Python would cache them,
        // which rython cannot reproduce, so it refuses loudly).
        let mut cache_key: Option<Vec<(proc_macro2::Ident, TokenStream)>> = match cache_spec {
            None => None,
            Some(_) => {
                // A cached METHOD (`@lru_cache_weakref(maxsize=...)` on
                // `_resolve_endpoint` — botocore's EndpointProvider): the
                // cache is a performance optimization — the method lowers
                // UNCACHED with a warning (documented divergence).
                if is_method {
                    options.definition_warnings.borrow_mut().push(format!(
                        "@lru_cache on method `{}` is dropped (caching is a \
                         performance optimization; the method lowers uncached)",
                        self.name
                    ));
                    None
                } else {
                    if !self.args.posonlyargs.is_empty()
                        || !self.args.kwonlyargs.is_empty()
                        || self.args.vararg.is_some()
                        || self.args.kwarg.is_some()
                    {
                        // A cached function with *args/**kwargs
                        // (`func_with_weakref(weakref_to_self, *args,
                        // **kwargs)` — botocore's lru_cache_weakref): the
                        // varargs cannot form the cache key — the cache is
                        // dropped (the function lowers uncached;
                        // documented divergence).
                        options.definition_warnings.borrow_mut().push(format!(
                            "@lru_cache on `{}` with varargs is dropped (the cache \
                             cannot key on varargs; the function lowers uncached)",
                            self.name
                        ));
                        None
                    } else {
                        let mut key = Vec::new();
                        let mut key_ok = true;
                        for p in &self.args.args {
                        let ty = match p.annotation.as_deref() {
                            Some(ExprType::Name(n)) if n.id == "int" => quote!(i64),
                            Some(ExprType::Name(n)) if n.id == "bool" => quote!(bool),
                            Some(ExprType::Name(n)) if n.id == "str" => quote!(String),
                            // float keys use the PyFloatKey wrapper: Python
                            // semantics (-0.0 == 0.0, NaN never hits) differ
                            // from Rust's f64 Hash/Eq.
                            Some(ExprType::Name(n)) if n.id == "float" => {
                                quote!(stdpython::stdlib::functools::PyFloatKey)
                            }
                            // Optional keys: `x: str | None` caches on the
                            // Option (charset_normalizer's lg_inclusion).
                            // python_annotation_to_rust_type ALREADY returns
                            // the full Option<T> — wrapping it again nested
                            // Option<Option<T>> (round 75).
                            Some(ann)
                                if crate::is_optional_annotation(ann) =>
                            {
                                crate::python_annotation_to_rust_type(ann)
                                    .unwrap_or_else(|| quote!(Option<String>))
                            }
                            _ => {
                                // A cache key of an unsupported type
                                // (`cacheable_page: CacheablePageContent` —
                                // pip's with_cached_index_content): the
                                // cache cannot key on it — dropped (the
                                // function lowers uncached; documented
                                // divergence).
                                options.definition_warnings.borrow_mut().push(format!(
                                    "@lru_cache on `{}`: parameter `{}` is not a valid \
                                     cache key (must be int, bool, str, float, or \
                                     Optional); the cache is dropped",
                                    self.name, p.arg
                                ));
                                key_ok = false;
                                break;
                            }
                        };
                        key.push((crate::safe_ident(&p.arg), ty));
                    }
                    if key_ok { Some(key) } else { None }
                }
            }
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
        // Issue #115: this function may write exactly the mutable statics
        // its own `global` statements declare; every other scope's
        // grant (the module's all-of-them default in particular) does not
        // apply inside this body.
        options.scope_global_writables = std::rc::Rc::new(fn_mutable_globals.clone());
        // Names managed by this function's prologue: hoisted assignments
        // plus mutable parameters. A `for`-loop target on one of these
        // lowers to a store into the hoisted binding, never a shadowing
        // fresh binding (issue #80). A `global`-declared mutable static is
        // NOT a local — its binding is the module static (issue #115).
        options.hoisted_names = std::rc::Rc::new(
            scope
                .assigned
                .iter()
                .chain(scope.needs_mut.iter())
                .filter(|n| !fn_mutable_globals.contains(*n))
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
                    // An "optional" annotation (`Optional[T]`, `T | None`)
                    // makes the parameter an Option slot. EXCEPT a union
                    // whose members resolve to the boxed PyValue — `int |
                    // str | None` (urllib3's cert_reqs) AND class-member
                    // unions `Retry | bool | int | None` (urllib3's
                    // retries): the box already contains None, so the name
                    // is PyValue, not Option, and must not be Some-wrapped
                    // (rounds 40/41). The symbol-aware alias resolver is
                    // the authority — it maps `Optional[bool | str]`,
                    // `bool | str | None` AND `Retry | bool | int | None`
                    // to PyValue for exactly this reason, while `Retry |
                    // None` resolves to Option<Retry>.
                    if crate::is_optional_annotation(ann)
                        && !matches!(
                            crate::resolve_alias_typeinfo(ann, &symbols, &options),
                            Some(crate::TypeInfo::PyValue)
                        )
                    {
                        optional.insert(p.arg.clone());
                    }
                }
            }
            options.optional_names = std::rc::Rc::new(optional);
            options.clone_str_attribute_returns =
                matches!(self.returns.as_deref(), Some(ExprType::Name(n)) if n.id == "str");
            // Issue #222's self-field half: the resolved return type (this
            // same chain the signature below consults) decides whether an
            // ATTRIBUTE return must clone out of the shared receiver — a
            // non-Copy field read moved into `Ok(..)` would leave `&self`
            // (E0507). Copy returns (int/float/bool) need no clone.
            options.clone_field_returns = self
                .resolved_return_type_in(&symbols, &options, ctx.enclosing_class_name())
                .is_some_and(|ty| {
                    let ty = ty.to_string();
                    !(ty == "i64"
                        || ty == "f64"
                        || ty == "bool"
                        || ty == "usize"
                        || ty == "()"
                        || ty.is_empty())
                });
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
                        .filter(|(_, ty)| matches!(ty, crate::TypeInfo::StrRef))
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
                    if crate::ast::tree::assign::is_builtin_scalar_name(&ann.id) {
                        known.insert(param.arg.clone(), ann.id.clone());
                    }
                    // A bare threading type name (`lock: Lock` under
                    // `from threading import Lock`): record the dotted
                    // form, so `with lock:` on the parameter lowers to
                    // the RAII guard (the with-statement classifier
                    // consults local_types).
                    if let Some(t) = crate::ThreadingType::from_name(&ann.id)
                        && matches!(
                            symbols.get(&ann.id),
                            Some(crate::SymbolTableNode::ImportFrom(i))
                                if crate::StdModule::from_name(&i.module)
                                    == Some(crate::StdModule::Threading)
                        )
                    {
                        known.insert(param.arg.clone(), format!("threading.{}", t.name()));
                    }
                }
                // A `T | None` param annotation (`release_conn: bool |
                // None` — urllib3): record the dotted form so infer_type
                // resolves the name to an Option (a local assigned from
                // it is itself an Option binding — round 45).
                if let Some(ExprType::BinOp(op)) = param.annotation.as_deref()
                    && matches!(op.op, crate::BinOps::BitOr)
                {
                    // `bool | None` — one side is a Name, the other is the
                    // None literal (a Constant node). Record the dotted
                    // form so infer_type resolves the name to an Option.
                    let name_side: Option<&ExprType> = if crate::is_none_expr(&op.right) {
                        Some(op.left.as_ref())
                    } else if crate::is_none_expr(&op.left) {
                        Some(op.right.as_ref())
                    } else {
                        None
                    };
                    if let Some(ExprType::Name(n)) = name_side {
                        known.insert(param.arg.clone(), format!("{} | None", n.id));
                    }
                }
                // A dotted threading annotation (`lock: threading.Lock`):
                // same recording as above.
                if let Some(ExprType::Attribute(ann)) = param.annotation.as_deref()
                    && matches!(ann.value.as_ref(), ExprType::Name(m)
                        if crate::StdModule::from_name(&m.id)
                            == Some(crate::StdModule::Threading))
                    && let Some(t) = crate::ThreadingType::from_name(&ann.attr)
                {
                    known.insert(param.arg.clone(), format!("threading.{}", t.name()));
                }
                // `bytes | bytearray` (a same-Rust-type union) is the "raw
                // sequence" idiom: record it as bytes so isinstance checks
                // against either name decide true (charset_normalizer).
                if let Some(ExprType::BinOp(op)) = param.annotation.as_deref()
                    && matches!(op.op, crate::BinOps::BitOr)
                {
                    let mut names = Vec::new();
                    for side in [&op.left, &op.right] {
                        if let ExprType::Name(n) = side.as_ref()
                            && matches!(n.id.as_str(), "bytes" | "bytearray")
                        {
                            names.push(n.id.clone());
                        }
                    }
                    if names.len() == 2 {
                        known.insert(param.arg.clone(), "bytes".to_string());
                    }
                }
            }
            let mut literal_types = std::collections::HashMap::new();
            collect_local_types(&self.body, &mut literal_types);
            for (name, ty) in literal_types {
                let Some(py) = typeinfo_to_py_name(&ty) else {
                    continue;
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
            // The body analysis sees the PARAMETERS' annotated types (the
            // one annotation authority), so a local bound to an expression
            // over a parameter (`bigs = [s for s in shapes if ...]`) types
            // from it; the full parameter seeding below still runs last
            // and wins.
            let analysis_options = {
                let mut seeded = options.clone();
                let mut names = seeded.name_types.as_ref().clone();
                for p in self
                    .args
                    .args
                    .iter()
                    .chain(self.args.posonlyargs.iter())
                    .chain(self.args.kwonlyargs.iter())
                {
                    if let Some(ann) = p.annotation.as_deref()
                        && let Some(t) = crate::resolve_alias_typeinfo(ann, &symbols, &options)
                            .or_else(|| crate::annotation_type_info(ann))
                    {
                        names.entry(p.arg.clone()).or_insert(t);
                    }
                }
                seeded.name_types = std::rc::Rc::new(names);
                seeded
            };
            let mut info = crate::analyze_function_types_with_class(
                &effective_body,
                Some(&analysis_options),
                Some(&symbols),
                ctx.enclosing_class_name(),
            );
            // None-assigned locals (and locals seeded from Option fields
            // by the class-aware analysis) are Option bindings: the
            // access lowering unwraps them (issue #137's Option-aware
            // access), so the codegen's optional set must include them,
            // not just the Optional-annotated PARAMETERS.
            if !info.optional_names.is_empty() {
                let mut merged: std::collections::HashSet<String> =
                    (*options.optional_names).clone();
                merged.extend(info.optional_names.iter().cloned());
                options.optional_names = std::rc::Rc::new(merged);
            }
            // Annotated names (annotated locals) join the annotation-
            // derived set: the round-84 Option-unwrap guard consults them
            // (an annotated local's PyValue-ness is authoritative — a
            // `X | None` union that boxes has its None inside the box).
            if !info.annotated_names.is_empty() {
                let mut merged: std::collections::HashSet<String> =
                    (*options.annotated_names).clone();
                merged.extend(info.annotated_names.iter().cloned());
                options.annotated_names = std::rc::Rc::new(merged);
            }
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
                            // `object` — a value of unknown type: the boxed
                            // heterogeneous value. EXCEPT a comparison
                            // dunder (__eq__/__lt__/...) of a SHARED class:
                            // its other parameter compares two shared
                            // instances — the PyRef class, not the box
                            // (PyValue has no class member — records's
                            // Version.__eq__(self, other: object), round
                            // 99).
                            "object" | "Any"
                                if matches!(
                                    self.name.as_str(),
                                    "__eq__" | "__ne__" | "__lt__" | "__le__"
                                        | "__gt__" | "__ge__"
                                ) && ctx.enclosing_class_name().is_some_and(|c| {
                                    crate::ast::tree::shared::is_shared(c)
                                }) =>
                            {
                                let class = ctx.enclosing_class_name().expect("checked").to_string();
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::Class(class));
                            }
                            "object" | "Any" => {
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            }
                            // A bare `list`/`dict`/... annotation has no
                            // element/key type (`properties: dict` — a
                            // NamedTuple field, botocore's
                            // RuleSetEndpoint): the parameter lowers as a
                            // boxed PyValue (the unannotated fallback) —
                            // the field inference matches.
                            "list" | "List" | "dict" | "Dict" | "tuple" | "Tuple" | "set"
                            | "Set" | "Optional" => {
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            }
                            // A CLASS-typed parameter (`def f(c: C)` —
                            // requests' sessions): record the class so
                            // receiver resolution (property reads/setter
                            // stores on the parameter) can route to the
                            // class's methods. A module-level TYPE ALIAS
                            // annotation (`value: _TYPE_FIELD_VALUE` where
                            // `_TYPE_FIELD_VALUE = Union[str, bytes]` —
                            // urllib3's fields) resolves FIRST through the
                            // symbols-aware authority — the alias name is
                            // not a class, and the recorded type must agree
                            // with the parameter's actual Rust type (round
                            // 93 — the alias-as-Class entry left the
                            // PyValue-typed local's stores uncoerced,
                            // `value = py_mod(...)?` raw against PyValue).
                            _ => {
                                let cname = n.id.clone();
                                if let Some(t) = crate::resolve_alias_typeinfo(
                                    &ExprType::Name(n.clone()),
                                    &symbols,
                                    &options,
                                ) {
                                    info.name_types.insert(p.arg.clone(), t);
                                } else if symbols.get(&cname).is_some() {
                                    info.name_types.insert(
                                        p.arg.clone(),
                                        crate::TypeInfo::Class(cname),
                                    );
                                }
                            }
                        },
                        other => {
                            if crate::is_none_expr(other) {
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::Option(Box::new(crate::TypeInfo::PyObject)));
                            } else if matches!(other, ExprType::Subscript(sub)
                                if matches!(sub.value.as_ref(), ExprType::Name(n)
                                    if matches!(n.id.as_str(), "type" | "Type")))
                            {
                                // `type[X]` / `Type[X]` — a CLASS held as a
                                // value (`expected_type: type[_T]` — pip's
                                // direct_url): no rython value equivalent —
                                // a boxed PyValue (the class-as-value
                                // divergence).
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            } else if let Some(t) = crate::annotation_type_info(other) {
                                info.name_types.insert(p.arg.clone(), t);
                            } else if crate::is_str_bytes_union(other) {
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::StrOrBytes);
                            } else if let Some(t) =
                                crate::resolve_alias_typeinfo(other, &symbols, &options)
                            {
                                // A module-level TYPE ALIAS
                                // (`list[CoherenceMatches]`, charset_normalizer).
                                info.name_types.insert(p.arg.clone(), t);
                            } else if matches!(other, ExprType::BinOp(op)
                                if matches!(op.op, crate::BinOps::BitOr))
                            {
                                // An unresolvable UNION (`exc_info: ExcInfo |
                                // BaseException` — pip's misc): the
                                // parameter is a boxed PyValue.
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            } else if crate::is_optional_annotation(other) {
                                // `T | None` with an unresolvable inner
                                // (`load_only: Kind | None` where Kind is a
                                // NewType — pip's Configuration): the
                                // parameter is a boxed PyValue.
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            } else if matches!(other, ExprType::Attribute(_))
                                && !crate::resolve_alias_typeinfo(other, &symbols, &options)
                                    .is_some()
                            {
                                // An EXTERNAL-MODULE dotted annotation
                                // (`std_handle: wintypes.HANDLE` — rich's
                                // _win32_console, a Windows-only ctypes
                                // module): a foreign object type — the
                                // parameter is a boxed PyValue (the
                                // external-type divergence).
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            } else if matches!(other, ExprType::Constant(c)
                                if matches!(&c.0, Some(litrs::Literal::String(_))))
                            {
                                // A STRING-LITERAL annotation (`"IO[str]"` —
                                // rich's _win32_console, a forward
                                // reference): unresolvable at conversion
                                // time — the parameter is a boxed PyValue
                                // (the external-type divergence).
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
                            } else if matches!(other, ExprType::Subscript(sub)
                                if matches!(sub.value.as_ref(), ExprType::Name(n)
                                    if matches!(n.id.as_str(), "dict" | "Dict")))
                            {
                                // A dict-generic annotation with an
                                // unresolvable element (`Dict[int, None]` —
                                // rich's control.py): a boxed
                                // PyDict<String, PyValue>.
                                info.name_types
                                    .insert(p.arg.clone(), crate::TypeInfo::PyValue);
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
            // Issue #120: the **kwargs parameter is a boxed heterogeneous
            // dict (`PyDict<String, PyValue>`): extra keyword arguments
            // pack into it at call sites, and the body reads/contains/
            // copies it like any dict.
            if let Some(kwarg) = &self.args.kwarg {
                info.name_types
                    .insert(kwarg.arg.clone(), crate::TypeInfo::Dict(
                        Box::new(crate::TypeInfo::String),
                        Box::new(crate::TypeInfo::PyValue),
                    ));
            }
            // Issue #120: the *args parameter is the boxed heterogeneous
            // list (`Vec<PyValue>`): extra positional arguments pack into
            // it at call sites; len/index/iterate yield PyValue.
            if let Some(vararg) = &self.args.vararg {
                info.name_types.insert(
                    vararg.arg.clone(),
                    crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue)),
                );
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
                annotated_names: std::collections::HashSet::new(),
                optional_names: std::collections::HashSet::new(),
            };
            crate::pin_empty_containers(&effective_body, &mut info, Some(&symbols), Some(&options));
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
        // A None-defaulted unannotated parameter can only ever hold None
        // (its generic type variable would be un-inferable from None at
        // call sites) — it gets the concrete Option<()> type and takes no
        // part in inference (issue #117).
        let none_defaulted: std::collections::HashSet<String> = self
            .args
            .posonlyargs
            .iter()
            .chain(self.args.args.iter())
            .chain(self.args.kwonlyargs.iter())
            .filter(|p| crate::param_has_none_default(p, &self))
            .map(|p| p.arg.clone())
            .collect();
        let mut inferred_signature = {
            // @classmethod/@staticmethod are associated functions (no
            // receiver, class ref dropped) — NOT methods, so unannotated
            // parameters are inferred like free functions (issue #117).
            let is_method = matches!(
                &ctx,
                CodeGenContext::Class(_) | CodeGenContext::Trait { .. }
            ) && !is_class_method
                && !is_static_method
                && (self.args.posonlyargs.first().is_some()
                    || self.args.args.first().is_some());
            let unannotated: Vec<String> = self
                .args
                .posonlyargs
                .iter()
                .chain(self.args.args.iter())
                .chain(self.args.kwonlyargs.iter())
                // A classmethod's first parameter is the class reference
                // (cls/self) — dropped from the signature, so it takes no
                // part in inference (issue #117). An instance method's
                // first parameter is the RECEIVER — also dropped, whatever
                // its name (boto3's `factory_self`).
                .enumerate()
                .filter(|(i, p)| {
                    !(is_class_method && *i == 0)
                        && !(is_method && *i == 0)
                        && p.arg != "self"
                        && p.annotation.is_none()
                        && !none_defaulted.contains(&p.arg)
                })
                .map(|(_, p)| p.arg.clone())
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
                // A kwargs-only (or fully-annotated) function still calls
                // through loop elements over non-param iterables
                // (`for filter in self.X: filter(**kwargs)` — botocore's
                // docs client): collect those called-param names so the
                // call sites lower as dropped no-ops.
                let mut sig = crate::InferredSignature::default();
                sig.called_params =
                    crate::collect_called_params(&effective_body, &symbols, &options);
                // A parameter annotated bare `type` (`dict_class: type =
                // OrderedDict` — requests' sessions) is a CALLABLE: calls
                // through it drop (the callable-as-value divergence) — the
                // codegen cannot hold a class/function as a value.
                for p in self
                    .args
                    .posonlyargs
                    .iter()
                    .chain(self.args.args.iter())
                    .chain(self.args.kwonlyargs.iter())
                {
                    if p.annotation
                        .as_deref()
                        .is_some_and(crate::ast::tree::arguments::is_type_annotation)
                    {
                        sig.called_params.insert(p.arg.clone());
                    }
                }
                sig
            } else if is_method {
                // An unannotated method parameter lowers as boxed PyValue
                // (with a warning) instead of failing: __init__ fields are
                // typed by the class field inference, and other methods'
                // unannotated params are duck-typed values (boto3's
                // document_actions(section), s3transfer's ReadFileChunk).
                options.definition_warnings.borrow_mut().push(format!(
                    "method `{}` has unannotated parameter(s) `{}`; they lower as \
                     boxed PyValue",
                    self.name,
                    unannotated.join("`, `")
                ));
                let mut sig = crate::InferredSignature::default();
                for p in unannotated {
                    sig.param_types.insert(p, quote!(stdpython::PyValue));
                }
                sig
            } else {
                crate::infer_unannotated_signature(
                    &effective_body,
                    &unannotated,
                    &options.name_types,
                    &options.use_counts,
                    &symbols,
                    &options,
                    &self.name,
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
        // None-defaulted unannotated parameters are concrete Option<()> —
        // nothing but None can ever be stored in them (issue #117) —
        // UNLESS the parameter's value is genuinely used
        // (`self._retryable_exceptions = retryable_exceptions` —
        // botocore's MaxAttemptsDecorator, round 33): it then carries a
        // real Python value (a boxed exception-name list) and types as
        // the boxed PyValue, so call sites pass values and the dropped
        // None default boxes to PyValue::None_.
        let mut final_param_types = inferred_signature.param_types.clone();
        for name in &none_defaulted {
            if crate::name_read_as_value(name, &effective_body) {
                final_param_types.insert(name.clone(), quote!(stdpython::PyValue));
            } else {
                final_param_types.insert(name.clone(), quote!(Option<()>));
            }
        }
        // A FREE function's value-pinned parameters (inferred boxed
        // PyValue — issue #161) render `impl Into<stdpython::PyValue>`
        // with a boxing prologue, so callers pass plain values exactly
        // like Python. Methods and trait bodies keep the bare PyValue
        // (impl-Trait parameters are not legal in trait signatures).
        let pyvalue_into_params: std::collections::HashSet<String> = if matches!(
            &ctx,
            CodeGenContext::Class(_) | CodeGenContext::Trait { .. }
        ) {
            std::collections::HashSet::new()
        } else {
            final_param_types
                .iter()
                .filter(|(_, ty)| ty.to_string() == "stdpython :: PyValue")
                .map(|(name, _)| name.clone())
                .collect()
        };

        options.pyvalue_into_params = std::rc::Rc::new(pyvalue_into_params);
        options.param_type_vars = std::rc::Rc::new(final_param_types);
        // An INFERRED String return needs the same literal-owning
        // treatment as an annotated `-> str`: `return "pos"` in a generic
        // function whose return type unified to String must own the
        // &'static str (to_string), or the declared Result<String>
        // mismatches at build time.
        if inferred_signature
            .return_type
            .as_ref()
            .is_some_and(|t| t.to_string() == "String")
            && !options.clone_str_attribute_returns
        {
            options.clone_str_attribute_returns = true;
            let mut locals = std::collections::HashMap::new();
            collect_local_types(&effective_body, &mut locals);
            options.str_literal_locals = std::rc::Rc::new(
                locals
                    .into_iter()
                    .filter(|(_, ty)| matches!(ty, crate::TypeInfo::StrRef))
                    .map(|(name, _)| name)
                    .collect(),
            );
        }
        options.param_method_params =
            std::rc::Rc::new(inferred_signature.method_params.clone());
        options.duck_methods_on_params =
            std::rc::Rc::new(inferred_signature.duck_methods_on_params.clone());
        // A nested function name (`def KD(s, d)` inside __init__ —
        // requests' auth) is a CLOSURE in Python; rython's closures do not
        // capture the enclosing scope, so the definition drops (statement.rs)
        // and CALLS through the name drop too — add the names to
        // called_params so the call sites lower as no-ops. Their VALUE
        // reads (`hash_utf8 = sha256_utf8`) box to the boxed None
        // (value_callables).
        let mut value_callables = std::collections::HashSet::new();
        for nested in crate::nested_function_names(&self.body) {
            inferred_signature.called_params.insert(nested.clone());
            value_callables.insert(nested);
        }
        // A `type`-annotated callable parameter is the same: calls drop
        // (called_params) and value reads box.
        for p in self
            .args
            .posonlyargs
            .iter()
            .chain(self.args.args.iter())
            .chain(self.args.kwonlyargs.iter())
        {
            if p.annotation
                .as_deref()
                .is_some_and(crate::ast::tree::arguments::is_type_annotation)
            {
                value_callables.insert(p.arg.clone());
            }
        }
        options.value_callables = std::rc::Rc::new(value_callables);
        options.called_params =
            std::rc::Rc::new(inferred_signature.called_params.clone());
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
            } else if options.pyvalue_into_params.contains(name) {
                // Value-pinned parameters arrive as impl
                // Into<stdpython::PyValue>; box them up front (issue #161).
                if scope.needs_mut.contains(name) {
                    streams_prologue.extend(
                        quote!(let mut #ident: stdpython::PyValue = #ident.into();),
                    );
                } else {
                    streams_prologue.extend(
                        quote!(let #ident: stdpython::PyValue = #ident.into();),
                    );
                }
            } else if scope.needs_mut.contains(name) {
                streams_prologue.extend(quote!(let mut #ident = #ident;));
            }
        }
        for name in &scope.assigned {
            // `_` hoists like any name: safe_ident maps it to a real
            // identifier (readable, like Python's `_`), so the old
            // wildcard skip would leave tuple-destructure stores
            // (`(scheme, _, host, port, _) = parse_url(...)` — urllib3)
            // without a binding.
            // A `global`-declared mutable static has no local binding: its
            // stores go through py_global_write (issue #115).
            if fn_mutable_globals.contains(name) {
                continue;
            }
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

        // Generator lowering (issue #122-family): a body with `yield`
        // builds a Vec and returns it — the closest rython can get to a
        // generator (Python's `for chunk in cut_sequence_chunks(...)`
        // iterates the returned list just the same). The element type
        // comes from the `Generator[T, ...]` return annotation or the
        // first yielded value.
        let returns_generator_annotation = matches!(
            self.returns.as_deref(),
            Some(ExprType::Subscript(sub)) if match sub.value.as_ref() {
                ExprType::Name(n) => matches!(n.id.as_str(), "Generator" | "Iterator"),
                ExprType::Attribute(a) => {
                    matches!(a.value.as_ref(), ExprType::Name(m) if crate::is_typing(&m.id))
                        && matches!(a.attr.as_str(), "Generator" | "Iterator")
                }
                _ => false,
            }
        );
        // A `yield` used as a VALUE (`x = yield v`, `(yield v) or 1`) is
        // the generator's send channel; the list lowering has no such
        // channel, and `push(v)` in its place would evaluate to unit —
        // silently a different program. Refuse.
        if let Some(stmt) = expression_position_yield(&effective_body) {
            return Err(format!(
                "`yield` used as a value (`x = yield v`, `(yield v) or d`) in `{}` at line {}: \
                 rython lowers a generator to build-and-return-a-list, which has no send \
                 channel, so the expression would silently evaluate to (). Put the yield on \
                 its own statement, or rewrite the generator to build and return a list.",
                self.name,
                stmt.lineno.unwrap_or(0)
            )
            .into());
        }
        let gen_elt = if crate::body_has_yields(&effective_body)
            // An abstract generator STUB (`def stream(...) ->
            // typing.Iterator[bytes]: raise NotImplementedError()` —
            // urllib3's BaseHTTPResponse) has no yields, but its
            // annotation still decides the signature the trait
            // declaration carries; otherwise overriding generators'
            // Vec returns are E0053 against a () trait method.
            || returns_generator_annotation
        {
            // Even when the element type cannot be resolved (a
            // `typing.Iterator[str]` annotation), the generator must still
            // lower — Vec<_> infers from the pushes.
            generator_element_type(self.returns.as_deref(), &effective_body, &options, &symbols)
                .or(Some(crate::TypeInfo::PyObject))
        } else {
            None
        };
        // An UNRESOLVED yield type (a bare `yield` — @contextmanager — or
        // a value the inference cannot type) boxes: the placeholder `_`
        // is not legal in item signatures (E0121), and the boxed PyValue
        // is the honest element for an unknown yield.
        let gen_boxed = matches!(gen_elt, Some(crate::TypeInfo::PyObject));
        let gen_elt_tokens = gen_elt.as_ref().map(|elt| {
            if gen_boxed {
                quote!(stdpython::PyValue)
            } else {
                elt.to_rust_type()
            }
        });
        if let Some(t) = &gen_elt_tokens {
            streams.extend(quote!(let mut __rython_gen: Vec<#t> = Vec::new();));
        }

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
        // A function whose return type is the boxed PyValue: `return None`
        // lowers to `PyValue::None_` and other returns wrap in
        // PyValue::from (the None-mixing unification — botocore's
        // docs.client._allowlist_generate_presigned_url). Set BEFORE the
        // body statements render (they clone options per statement).
        // Issue #133: the flag must agree with WHICHEVER inference writes
        // the signature below — resolved_return_type for annotated/simple
        // functions, the collector unification for the generic path
        // (previously only the former was consulted, so a generic
        // `Result<PyValue, _>` signature carried unwrapped `Ok(val)`
        // bodies — E0308 on the issue's flagify shape).
        // Round 85 (the return-type directive): when the function's
        // return IS an Option (annotated `-> T | None`, or the inferred
        // `T | None` unification below), the return site must NOT box —
        // plain returns Some-wrap, None stays the empty member, and the
        // fall-through tail is Ok(None).
        let resolved = self.resolved_return_type_in(&symbols, &options, ctx.enclosing_class_name());
        let return_is_option = self
            .returns
            .as_deref()
            // The symbols-aware authority (round 85): the syntax-only
            // annotation_type_info cannot see a quoted class name
            // (`Optional["CharsetMatch"]` — charset_normalizer's best,
            // where the signature resolves via resolve_alias_typeinfo) —
            // the flag must agree with the signature.
            .and_then(|ann| crate::resolve_alias_typeinfo(ann, &symbols, &options))
            .is_some_and(|t| matches!(t, crate::TypeInfo::Option(_)))
            || (self.returns.is_none()
                && (inferred_signature
                    .return_type
                    .as_ref()
                    .is_some_and(|t| t.to_string().starts_with("Option"))
                    || matches!(
                        &resolved,
                        Some(ty) if ty.to_string().starts_with("Option")
                    )));
        options.fn_return_is_pyvalue = !return_is_option
            && (matches!(
                &resolved,
                Some(ty) if ty.to_string() == "stdpython :: PyValue"
            ) || (self.returns.is_none()
                // A unified return exists whenever the collector ran — for a
                // generic signature AND for one whose parameters pinned
                // concrete (a value-pinned PyValue, issue #161); a method's
                // synthesized signature never carries one, so methods keep
                // their own path.
                && matches!(
                    &inferred_signature.return_type,
                    Some(ty) if ty.to_string() == "stdpython :: PyValue"
                ))
                // The NON-generic disconnect (annotated params, mixed
                // `return 1`/`return None` literal bodies — the issue's pick
                // shape): the generic collector never ran (no unannotated
                // params), resolved_return_type has no answer — box.
                || (self.returns.is_none()
                    && !inferred_signature.is_generic()
                    && resolved.is_none()
                    && literal_returns_need_boxing(&self.body)));

        // A `-> T | None` function's resolved return type is an Option:
        // plain (non-Option) returns wrap in Some at the return site
        // (statement.rs) — Python returns the bare value where None is
        // only one of the possible results (urllib3's `_normalize_host`
        // returns both `host.lower()` and `host`, a `str | None` path).
        // Round 85 extends this to the INFERRED `T | None` returns.
        options.fn_return_is_option = return_is_option;

        // Round 81 (the generics directive): a CONCRETE typed return
        // (`-> Vec<u8>`, `-> i64` ...) whose value arrives as a boxed
        // PyValue (a dropped external call stored in a local —
        // DeflateDecoder's `decompressed`, a boxed member read) converts
        // at the return site via the reverse From<PyValue> impls. The
        // boxed-PyValue and Option returns are NOT "typed" here — they
        // have their own return-site shapes (PyValue::from / Some wrap).
        options.fn_return_typed = self
            .resolved_return_type_in(&symbols, &options, ctx.enclosing_class_name())
            .and_then(|ts| {
                let s = ts.to_string();
                let t = if s == "i64" {
                    Some(crate::TypeInfo::Int)
                } else if s == "f64" {
                    Some(crate::TypeInfo::Float)
                } else if s == "bool" {
                    Some(crate::TypeInfo::Bool)
                } else if s == "std :: string :: String" || s == "String" {
                    Some(crate::TypeInfo::String)
                } else if s == "std :: vec :: Vec < u8 >" || s == "Vec < u8 >" {
                    Some(crate::TypeInfo::Bytes)
                } else {
                    None
                };
                t.filter(|_| !options.fn_return_is_pyvalue && !options.fn_return_is_option)
            })
            // A declared return naming a polymorphic ROOT (hierarchy.rs):
            // the slot is the sum type, and the return site converts a
            // subtree struct into it.
            .or_else(|| {
                self.returns
                    .as_deref()
                    .and_then(|ann| crate::resolve_alias_typeinfo(ann, &symbols, &options))
                    .filter(|t| {
                        matches!(t, crate::TypeInfo::Class(r)
                            if crate::ast::tree::hierarchy::is_polymorphic_root(r))
                            && !options.fn_return_is_pyvalue
                            && !options.fn_return_is_option
                    })
            });

        // A `-> List[Union[...]]` return whose element resolves to the
        // boxed PyValue (`_seg_N` in idna's uts46data: `List[Union[
        // Tuple[int, str], Tuple[int, str, str]]]`): a RETURNING list
        // literal must box each element (`PyValue::from((0, "3"))`) — the
        // literal alone sees only homogeneous 2-tuples and infers
        // `Vec<(i64, &str)>`, mismatching the annotation. The return
        // statement threads this in (round 57).
        options.fn_return_list_elt = std::rc::Rc::new(
            self.returns.as_deref().and_then(|ann| {
                match crate::annotation_type_info(ann) {
                    // A `-> list[str]` return's list LITERAL must own its
                    // string literals (`return ["a", "b"]` — charset's
                    // unicode_range_languages): the literal alone infers
                    // Vec<&'static str>, mismatching Vec<String> (round
                    // 78 — the same forced-elt mechanism round 57 added
                    // for the boxed-element case).
                    Some(crate::TypeInfo::Vec(inner))
                        if matches!(*inner, crate::TypeInfo::String) =>
                    {
                        Some(crate::TypeInfo::String)
                    }
                    Some(crate::TypeInfo::Vec(inner))
                        if matches!(*inner, crate::TypeInfo::PyValue) =>
                    {
                        Some((*inner).clone())
                    }
                    _ => None,
                }
            }),
        );

        // Issue #125: thread narrowed-Option state through the body. After
        // `if x is not None: <body> else: <else>` where BOTH branches leave
        // x holding a non-None value, x is non-None for the rest of the
        // function (Python narrows it; the hoisted binding stays Option, so
        // every later read must unwrap). A name assigned None on some path
        // drops out of the narrowed set again.
        let mut narrowed: std::collections::HashMap<String, crate::TypeInfo> =
            std::collections::HashMap::new();
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
            let mut stmt_options = options.clone();
            stmt_options.narrowed_names = std::rc::Rc::new(narrowed.clone());
            if gen_elt.is_some() {
                stmt_options.generator_collector =
                    std::rc::Rc::new(Some("__rython_gen".to_string()));
                stmt_options.generator_boxes = gen_boxed;
            }
            streams.extend(
                s.clone()
                    .to_rust(body_ctx.clone(), stmt_options, symbols.clone())?,
            );
            streams.extend(quote!(;));
            crate::update_narrowed_after_statement(s, &mut narrowed, &options);
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
        // A GENERATOR returns its collected list (`Generator[str, ...]`
        // → Vec<String>), overriding any other inference.
        let return_type = if let Some(t) = &gen_elt_tokens {
            quote!(-> Result<Vec<#t>, PyException>)
        } else if (inferred_signature.is_generic()
            || inferred_signature.return_type.is_some())
            && self.returns.is_none()
        {
            // Inference ran and owns the unannotated return type: a
            // generic signature, or a non-generic one whose parameters
            // pinned concrete while the collector still unified the
            // returns (issue #161's value-pinned PyValue params). A
            // method's synthesized signature has neither, so methods keep
            // the resolved_return_type path.
            // The inferred generic return (issue #109, M1): a parameter's
            // variable, a conversion result, or an associated Output. Only
            // when every path returns — a fall-through body returns unit.
            // An explicit return annotation always wins.
            if guarantees_return(&self.body) {
                match &inferred_signature.return_type {
                    Some(ty) => quote!(-> Result<#ty, PyException>),
                    None => {
                        // A raise-only stub (`raise NotImplementedError`
                        // — a base-class method that subclasses override,
                        // s3transfer's DownloadOutputManager): the return
                        // is a boxed PyValue (the override's type is
                        // unknown).
                        if self.body.iter().any(|s| {
                            matches!(&s.statement, crate::StatementType::Raise(_))
                        }) {
                            quote!(-> Result<stdpython::PyValue, PyException>)
                        } else {
                            return Err(
                                "could not infer this function's return type from its \
                                 unannotated parameters; add a return annotation \
                                 (issue #109, M1)"
                                    .to_string()
                                    .into(),
                            )
                        }
                    }
                }
            } else if inferred_signature.return_type.as_ref().is_some_and(|ty| {
                // A return type that references a type variable (a loop
                // element, an associated Output) cannot coexist with a
                // fall-through path (Python returns None there) — loud,
                // never a rustc mismatch at build time. A CONCRETE partial
                // return (e.g. `if c: return 1`) keeps the old Result<()>
                // shape. Round 85 (the return-type directive): an
                // OPTION return is EXEMPT — the fall-through None is the
                // Option's empty member (`for x in p: return x` returns
                // `B | None` = Option<B>), exactly the model the directive
                // prescribes; the signature below carries the Option.
                let s = ty.to_string();
                !s.starts_with("Option")
                    && {
                        let tokens: Vec<&str> = s
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|t| !t.is_empty())
                            .collect();
                        inferred_signature.type_params.iter().any(|p| {
                            let p = p.to_string();
                            tokens.iter().any(|t| *t == p)
                        })
                    }
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
            } else if matches!(
                &inferred_signature.return_type,
                Some(ty) if ty.to_string() == "stdpython :: PyValue"
            ) {
                // A fall-through path with the BOXED-PyValue return (the
                // None-mixing unification — `return [ChecksumError]` +
                // `return exceptions` + fall-through, botocore's
                // retryhandler): None is already one of the boxed value's
                // members, so the type stands and the fall-through renders
                // PyValue::None_ (issue #122 step 3).
                quote!(-> Result<stdpython::PyValue, PyException>)
            } else if let Some(ty) = &inferred_signature.return_type
                && ty.to_string().starts_with("Option")
            {
                // Round 85: an inferred `Option<T>` return with a
                // fall-through path — the None IS the fall-through (the
                // directive: a function that can return exactly `T | None`
                // returns Option<T>, and the caller decides what to do
                // with the None). The fall-through tail below renders the
                // None member.
                quote!(-> Result<#ty, PyException>)
            } else {
                quote!(-> Result<(), PyException>)
            }
        } else {
            match self.resolved_return_type_in(&symbols, &options, ctx.enclosing_class_name()) {
                Some(ty) => {
                    // A SHARED class method returning its OWN class
                    // (`parse -> "Version"` on the PyRef-shared Version —
                    // records, round 99): the instance is held behind
                    // PyRef, so the signature says PyRef<Version>, the
                    // same wrap the constructor and the field types use.
                    let enclosing_shared = ctx
                        .enclosing_class_name()
                        .is_some_and(|c| crate::ast::tree::shared::is_shared(c));
                    let ty_s = ty.to_string();
                    let is_own_class = ty_s == "Self"
                        || Some(ty_s.replace(' ', "").as_str())
                            == ctx.enclosing_class_name();
                    let ty = if enclosing_shared && is_own_class {
                        quote!(stdpython::PyRef<#ty>)
                    } else {
                        ty
                    };
                    quote!(-> Result<#ty, PyException>)
                }
                // Mixed literal returns (`return 1` / `return None` under
                // annotated params) box to PyValue; the body statements
                // were rendered with fn_return_is_pyvalue set above, so
                // the signature must agree (issue #133).
                None if self.returns.is_none()
                    && literal_returns_need_boxing(&self.body) =>
                {
                    quote!(-> Result<stdpython::PyValue, PyException>)
                }
                None => quote!(-> Result<(), PyException>),
            }
        };

        // A body that can fall off the end implicitly returns None: give the
        // generated block an Ok(()) tail. Bodies that return (or raise) on
        // every path end with `return`/`return Err`, which need no tail.
        // A GENERATOR ends by returning its collected list — inside the
        // function's Result, like every return.
        if gen_elt.is_some() {
            streams.extend(quote!(return Ok(__rython_gen);));
        } else if !guarantees_return(&self.body) {
            if options.fn_return_is_pyvalue {
                streams.extend(quote!(Ok(PyValue::None_)));
            } else if options.fn_return_is_option {
                // Round 85: an Option-returning function's fall-through
                // path is Python's None — the Option's empty member.
                streams.extend(quote!(Ok(None)));
            } else {
                streams.extend(quote!(Ok(())));
            }
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
            // A cached function with GENERIC (unannotated) parameters
            // (`func_with_weakref(weakref_to_self, *args, **kwargs)` under
            // `@functools.lru_cache(...)` — botocore's lru_cache_weakref):
            // the cache cannot key on the inferred types — the cache is
            // dropped (the function lowers uncached; documented
            // divergence).
            options.definition_warnings.borrow_mut().push(format!(
                "functools.lru_cache on `{}` with generic parameters is dropped (the \
                 cache cannot key on inferred types; the function lowers uncached)",
                self.name
            ));
            cache_spec = None;
            cache_key = None;
        }
        let streams = if let (Some(maxsize), Some(key)) = (cache_spec, cache_key.as_ref()) {
            let maxsize_tokens = match maxsize {
                None => quote!(None),
                Some(n) => quote!(Some(#n as usize)),
            };
            let key_types: Vec<&TokenStream> = key.iter().map(|(_, t)| t).collect();
            let key_names: Vec<&proc_macro2::Ident> = key.iter().map(|(n, _)| n).collect();
            // The __lru_uncached fn takes the RAW parameter types (float
            // stays f64 for the wrapped call); only the cache KEY tuple
            // wraps floats in PyFloatKey.
            let uncached_types: Vec<TokenStream> = key
                .iter()
                .zip(self.args.args.iter())
                .map(|((_, ty), p)| {
                    if matches!(p.annotation.as_deref(), Some(ExprType::Name(n)) if n.id == "float")
                        && ty.to_string().contains("PyFloatKey")
                    {
                        crate::python_annotation_to_rust_type(
                            p.annotation.as_deref().expect("float annotated"),
                        )
                        .unwrap_or_else(|| quote!(f64))
                    } else {
                        quote!(#ty)
                    }
                })
                .collect();
            // The cache KEY tuple: floats wrap in PyFloatKey (Python float
            // semantics); everything else is the raw value cloned. The
            // __lru_uncached call always passes the raw values.
            let key_vals: Vec<TokenStream> = key
                .iter()
                .zip(self.args.args.iter())
                .map(|((name, ty), p)| {
                    let ident = name;
                    if matches!(p.annotation.as_deref(), Some(ExprType::Name(n)) if n.id == "float")
                        && ty.to_string().contains("PyFloatKey")
                    {
                        quote!(stdpython::stdlib::functools::PyFloatKey(#ident))
                    } else {
                        quote!(#ident.clone())
                    }
                })
                .collect();
            let ret = match self.resolved_return_type_in(&symbols, &options, ctx.enclosing_class_name()) {
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
                    .get(&(#(#key_vals,)*))
                {
                    return Ok(__hit);
                }
                #[allow(non_snake_case)]
                fn __lru_uncached(#(#key_names: #uncached_types),*) -> Result<#ret, PyException> {
                    #streams
                }
                let __lru_value = __lru_uncached(#((#key_names).clone()),*)?;
                __LRU_CACHE
                    .lock()
                    .expect("lru_cache mutex poisoned")
                    .put((#(#key_vals,)*), __lru_value.clone());
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
        // NOT in a Trait context: `#[deprecated]` is not legal on trait
        // methods in impl blocks (issue #137: urllib3's inherited-method
        // bodies) — the -W channel still reports those notes.
        let lossy_warning = if options.lossy_warnings
            && !matches!(&ctx, CodeGenContext::Trait { .. })
        {
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
        // A comparison dunder of a SHARED class (`__eq__(self, other:
        // object)` — records's Version, round 99): the object-typed
        // parameter compares two shared instances — retype it as the
        // class so the signature says PyRef<Version> (the body's field
        // reads already borrow through the PyRef).
        let mut render_args = render_args;
        if matches!(self.name.as_str(), "__eq__" | "__ne__" | "__lt__" | "__le__" | "__gt__" | "__ge__")
            && ctx.enclosing_class_name().is_some_and(|c| crate::ast::tree::shared::is_shared(c))
        {
            let class = ctx.enclosing_class_name().expect("checked").to_string();
            for p in &mut render_args.args {
                let is_object_any = matches!(
                    p.annotation.as_deref(),
                    Some(ExprType::Name(n))
                        if matches!(n.id.as_str(), "object" | "Any")
                );
                if is_object_any {
                    p.annotation = Some(Box::new(ExprType::Name(crate::Name {
                        id: class.clone(),
                    })));
                }
            }
        }
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
/// into nested function or class definitions (their own return scopes).
fn collect_returns<'a>(body: &'a [Statement], out: &mut Vec<Option<&'a ExprType>>) {
    walk_stmts(body, Descend::SkipDefs, &mut |stmt| {
        if let StatementType::Return(value) = &stmt.statement {
            out.push(value.as_ref().map(|e| &e.value));
        }
        Flow::Continue
    });
}

/// Whether an unannotated-return function whose return values are all
/// LITERALS mixes kinds only the boxed PyValue can hold: a `return None`
/// (or bare `return`, or a possible fall-through — Python returns None
/// there) alongside a value literal, or value literals of two different
/// types (`return 1` / `return "x"`). Any non-literal return value bails
/// to false — the collector-based generic inference owns those shapes,
/// and a single consistent literal kind stays concrete (issue #133: the
/// annotated-parameter `pick(flag: bool)` shape, whose `return 1` /
/// `return None` mix previously rendered against a `Result<(), _>`
/// signature).
fn literal_returns_need_boxing(body: &[Statement]) -> bool {
    let mut returns = Vec::new();
    collect_returns(body, &mut returns);
    let mut kinds: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let mut has_none = !guarantees_return(body);
    for r in &returns {
        match r {
            None => has_none = true,
            Some(e) if is_none_expr(e) => has_none = true,
            Some(ExprType::Constant(c)) => {
                kinds.insert(match &c.0 {
                    Some(litrs::Literal::Integer(_)) => "int",
                    Some(litrs::Literal::Float(_)) => "float",
                    Some(litrs::Literal::Bool(_)) => "bool",
                    Some(litrs::Literal::String(_)) => "str",
                    _ => return false,
                });
            }
            _ => return false,
        }
    }
    !kinds.is_empty() && (has_none || kinds.len() > 1)
}

/// The names of NESTED function definitions inside a statement list
/// (recursing through control-flow bodies but not into nested defs/classes).
/// These are CLOSURES in Python; rython's closures do not capture the
/// enclosing scope (the closure-capture divergence), so the definitions
/// drop (statement.rs) and calls through the names drop too.
pub(crate) fn nested_function_names(body: &[crate::Statement]) -> Vec<String> {
    let mut out = Vec::new();
    walk_stmts(body, Descend::SkipDefs, &mut |stmt| {
        if let StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) =
            &stmt.statement
        {
            out.push(f.name.clone());
        }
        Flow::Continue
    });
    out
}

/// Map an expression to an obviously-inferable Rust type, if any.
/// Whether a [`crate::TypeInfo`] can stand as an inferred return type
/// (issue #222). The whitelist is deliberately narrow: only types whose
/// `to_rust_type()` is a faithful, self-contained rendering of what the
/// body actually produces.
///
/// Refused, and why:
/// - `PyObject` — the inferrer's "no answer"; rendering it would be a
///   guess, and a wrong return type is worse than an absent one.
/// - `StrRef` — a `&'static str` is a Rust literal artifact, not the
///   Python type; the literal rules earlier in the chain own that case
///   (`resolved_return_type` documents why `-> str` means `String`).
/// - everything else (`Option`, `Borrowed`, `PyValueMember`, `Range`,
///   `NdArray`, `PyValue`, ...) — either owned by a dedicated rule or
///   dependent on context this function does not have.
fn renderable_return_typeinfo(t: &crate::TypeInfo) -> bool {
    match t {
        crate::TypeInfo::Int
        | crate::TypeInfo::Float
        | crate::TypeInfo::Bool
        | crate::TypeInfo::String
        | crate::TypeInfo::Bytes
        | crate::TypeInfo::Class(_) => true,
        // The boxed PyValue IS a concrete answer (`return tuple(x)` —
        // round 33's botocore retryhandler): a guaranteed boxed return
        // declares Result<PyValue, _>, and the body already emits the
        // boxed values. Only the NO-ANSWER PyObject keeps refusing.
        crate::TypeInfo::PyValue => true,
        crate::TypeInfo::Vec(e) => renderable_return_typeinfo(e),
        crate::TypeInfo::Dict(k, v) => {
            renderable_return_typeinfo(k) && renderable_return_typeinfo(v)
        }
        crate::TypeInfo::Tuple(ts) => ts.iter().all(renderable_return_typeinfo),
        _ => false,
    }
}

/// The TypeInfo twin of [`simple_expr_type`] — the field-inference layer
/// (issue #137's review: `infer_fields` carries TypeInfo, not tokens, so
/// field types are structural and the coercion layers can match on them).
pub(crate) fn simple_expr_typeinfo(expr: &ExprType) -> Option<crate::TypeInfo> {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Some(crate::TypeInfo::Int),
            Some(litrs::Literal::Float(_)) => Some(crate::TypeInfo::Float),
            Some(litrs::Literal::Bool(_)) => Some(crate::TypeInfo::Bool),
            // A string constant lowers to a &'static str literal (the
            // field layer converts to owned String where fields need it).
            Some(litrs::Literal::String(_)) => Some(crate::TypeInfo::StrRef),
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => {
                Some(crate::TypeInfo::Bytes)
            }
            _ => None,
        },
        ExprType::JoinedStr(_) => Some(crate::TypeInfo::String),
        // `"sep".join(iterable)` — yields an owned String.
        ExprType::Call(call) => {
            if let ExprType::Attribute(a) = call.func.as_ref()
                && a.attr == "join"
                && matches!(
                    a.value.as_ref(),
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_)))
                )
            {
                Some(crate::TypeInfo::String)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn simple_expr_type(expr: &ExprType) -> Option<TokenStream> {    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Some(quote!(i64)),
            Some(litrs::Literal::Float(_)) => Some(quote!(f64)),
            Some(litrs::Literal::Bool(_)) => Some(quote!(bool)),
            // A string constant lowers to a &'static str literal.
            Some(litrs::Literal::String(_)) => Some(quote!(&'static str)),
            // A bytes literal (`b""`, `b"x"` — urllib3's
            // `self._first_try_data = b""`) lowers to Vec<u8>.
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => {
                Some(quote!(Vec<u8>))
            }
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
/// stored to by an aug-assign or a non-literal assignment. Recurses
/// through control flow but not into nested defs/classes (their locals
/// are their own).
pub(crate) fn scan_str_rebindings(
    body: &[Statement],
    literal_bindings: &mut std::collections::HashMap<String, bool>,
    rebound: &mut std::collections::HashSet<String>,
) {
    walk_stmts(body, Descend::SkipDefs, &mut |stmt| {
        if let StatementType::Assign(a) = &stmt.statement
            && let [ExprType::Name(name)] = a.targets.as_slice()
        {
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
        } else if let StatementType::AugAssign(a) = &stmt.statement
            && let ExprType::Name(name) = &a.target
        {
            rebound.insert(name.id.clone());
        }
        Flow::Continue
    });
}

/// Issue #112: a `del name` whose name is referenced afterwards (or whose
/// deletion is conditional) cannot lower to a no-op — the value would
/// still be readable where Python raises NameError. Loud error; a
/// reassignment or import clears the deletion (Python rebinds).
pub(crate) fn check_deleted_names(body: &[Statement]) -> Result<(), String> {
    let mut deleted = std::collections::HashSet::new();
    scan_deleted_names(body, &mut deleted)
}

type Deleted = std::collections::HashSet<String>;

/// The deletion state on the ways out of a statement sequence: the
/// fall-through state (None when every path returns or raises), the
/// states at each `break` and each `continue` (the enclosing loop merges
/// them: a continue into the next iteration, a break past the else
/// clause), and the union of the state at EVERY statement boundary inside
/// — where an exception may leave from, and what a `finally` may see.
struct DeletedPaths {
    falls: Option<Deleted>,
    breaks: Vec<Deleted>,
    continues: Vec<Deleted>,
    anywhere: Deleted,
}

impl DeletedPaths {
    /// Take the states that leave `nested` for an outer statement — its
    /// breaks and continues (an enclosing loop's business), and everything
    /// it reached — and hand back its fall-through.
    fn absorb(&mut self, nested: DeletedPaths) -> Option<Deleted> {
        self.breaks.extend(nested.breaks);
        self.continues.extend(nested.continues);
        self.anywhere.extend(nested.anywhere);
        nested.falls
    }
}

/// The union of the continuing states, None when none continues.
fn merge_deleted(states: Vec<Option<Deleted>>) -> Option<Deleted> {
    let mut out: Option<Deleted> = None;
    for state in states.into_iter().flatten() {
        out.get_or_insert_with(Deleted::new).extend(state);
    }
    out
}

/// A read of a deleted name in `exprs` is the loud issue-#112 error.
fn check_deleted_reads(deleted: &Deleted, exprs: &[&ExprType]) -> Result<(), String> {
    if let Some(name) = deleted
        .iter()
        .find(|name| exprs.iter().any(|e| crate::expr_references(e, name)))
    {
        return Err(format!(
            "`del {name}` then a use of `{name}`: rython cannot unbind a \
             binding (the value would still be readable where Python raises \
             NameError). Remove the del, or reassign `{name}` first (issue \
             #112)"
        ));
    }
    Ok(())
}

/// One pass over `body` from the incoming deletion state in `deleted`,
/// leaving the fall-through state there. Deletion follows control flow,
/// not the source order (Devin review on #323): each branch starts from
/// the incoming state, and the state after the statement is the UNION of
/// the paths that continue through it — deleted on any continuing path is
/// deleted, so a conditional deletion stays loud, a rebinding on one path
/// does not revive the name for its sibling (`del x; if c: x = 2; else:
/// print(x)`), and a path that returns or raises contributes nothing to
/// what follows (`del x; if c: x = 2; else: return 0; return x` is
/// fine). A loop body is scanned a second time from the merged state, so
/// a read that precedes the `del` in the next iteration is seen too; a
/// `break` skips the else clause. A `try`'s handlers and `finally` start
/// from every state the body reached, since an exception may leave from
/// any statement boundary. Nested defs and classes are their own scopes.
fn scan_deleted_names(body: &[Statement], deleted: &mut Deleted) -> Result<(), String> {
    let paths = scan_deleted_paths(body, deleted)?;
    *deleted = paths.falls.unwrap_or_default();
    Ok(())
}

fn scan_deleted_paths(body: &[Statement], incoming: &Deleted) -> Result<DeletedPaths, String> {
    let mut deleted = incoming.clone();
    let mut out = DeletedPaths {
        falls: None,
        breaks: Vec::new(),
        continues: Vec::new(),
        anywhere: incoming.clone(),
    };
    for stmt in body {
        out.anywhere.extend(deleted.iter().cloned());
        // A read of a deleted name: the statement's own expressions (a
        // definition's header included), and a store target that reads
        // its base (`x[i] = v`, `x.a = v`); a name / tuple target only
        // binds.
        let reads: Vec<&ExprType> = visit::stmt_exprs(stmt)
            .into_iter()
            .chain(stmt_targets(stmt).into_iter().filter(|t| {
                !matches!(
                    t,
                    ExprType::Name(_) | ExprType::Tuple(_) | ExprType::List(_) | ExprType::Starred(_)
                )
            }))
            .collect();
        check_deleted_reads(&deleted, &reads)?;
        // A definition binds its name here; its body is its own scope.
        match &stmt.statement {
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                deleted.remove(&f.name);
                continue;
            }
            StatementType::ClassDef(c) => {
                deleted.remove(&c.name);
                continue;
            }
            _ => {}
        }
        // A store rebinds the name: an assignment / aug-assign / `with
        // ... as` target, an import. A `for` target is bound only when
        // the loop body runs: the loop arm below rebinds it on the
        // body-entry state, not on the zero-iteration path.
        let is_for = matches!(stmt.statement, StatementType::For(_) | StatementType::AsyncFor(_));
        if !is_for {
            for target in stmt_targets(stmt) {
                deleted.retain(|name| !visit::target_binds(target, name));
            }
        }
        match &stmt.statement {
            StatementType::Import(i) => {
                for alias in &i.names {
                    deleted.remove(&alias.name);
                }
            }
            StatementType::ImportFrom(i) => {
                for alias in &i.names {
                    deleted.remove(&alias.name);
                    if let Some(asname) = &alias.asname {
                        deleted.remove(asname);
                    }
                }
            }
            StatementType::Delete(targets) => {
                for target in targets {
                    if let ExprType::Name(n) = target {
                        deleted.insert(n.id.clone());
                    }
                }
            }
            // The path ends here: nothing after it in this body runs.
            StatementType::Return(_) | StatementType::Raise(_) => {
                out.anywhere.extend(deleted);
                return Ok(out);
            }
            // The path leaves for the enclosing loop, carrying its state.
            StatementType::Break => {
                out.anywhere.extend(deleted.iter().cloned());
                out.breaks.push(deleted);
                return Ok(out);
            }
            StatementType::Continue => {
                out.anywhere.extend(deleted.iter().cloned());
                out.continues.push(deleted);
                return Ok(out);
            }
            _ => {}
        }
        // The paths through the statement's bodies.
        let next = match &stmt.statement {
            StatementType::If(i) => {
                let then = out.absorb(scan_deleted_paths(&i.body, &deleted)?);
                // No else: the fall-through path keeps the incoming state.
                let other = if i.orelse.is_empty() {
                    Some(deleted.clone())
                } else {
                    out.absorb(scan_deleted_paths(&i.orelse, &deleted)?)
                };
                merge_deleted(vec![then, other])
            }
            StatementType::For(_) | StatementType::AsyncFor(_) | StatementType::While(_) => {
                let bodies = visit::stmt_bodies(stmt);
                let (loop_body, orelse) = (bodies[0], bodies[1]);
                // Zero iterations keep the incoming state (a `for`
                // target stays unbound); an iteration enters the body
                // with the target rebound, and a later one re-enters it
                // from the merged state (a fall-through or a `continue`).
                let mut entering = deleted.clone();
                for target in stmt_targets(stmt) {
                    entering.retain(|name| !visit::target_binds(target, name));
                }
                let first = scan_deleted_paths(loop_body, &entering)?;
                let mut again = entering.clone();
                if let Some(falls) = &first.falls {
                    again.extend(falls.iter().cloned());
                }
                for state in &first.continues {
                    again.extend(state.iter().cloned());
                }
                for target in stmt_targets(stmt) {
                    again.retain(|name| !visit::target_binds(target, name));
                }
                // A `while` re-evaluates its test each iteration.
                if matches!(stmt.statement, StatementType::While(_)) {
                    check_deleted_reads(&again, &visit::stmt_exprs(stmt))?;
                }
                let second = scan_deleted_paths(loop_body, &again)?;
                // The loop's completion (no break): the incoming state
                // (zero iterations) and the merged iteration states.
                let mut completed = again.clone();
                completed.extend(deleted.iter().cloned());
                if let Some(falls) = &second.falls {
                    completed.extend(falls.iter().cloned());
                }
                for state in &second.continues {
                    completed.extend(state.iter().cloned());
                }
                // A break lands after the statement, skipping the else
                // clause; the else clause runs after a completion.
                let mut exits: Vec<Option<Deleted>> = first
                    .breaks
                    .iter()
                    .chain(second.breaks.iter())
                    .map(|b| Some(b.clone()))
                    .collect();
                out.anywhere.extend(first.anywhere);
                out.anywhere.extend(second.anywhere);
                exits.push(out.absorb(scan_deleted_paths(orelse, &completed)?));
                merge_deleted(exits)
            }
            StatementType::With(w) => out.absorb(scan_deleted_paths(&w.body, &deleted)?),
            StatementType::AsyncWith(w) => out.absorb(scan_deleted_paths(&w.body, &deleted)?),
            StatementType::Try(t) => {
                // An exception may leave the body from any statement
                // boundary: a handler starts from every state the body
                // reached, with its `as` name rebound; the else clause
                // from the body's fall-through. `finally` runs on EVERY
                // way out, per path: the continuing state, each pending
                // break and continue (the finally's own state is what
                // leaves — a `finally: del x` on a break path deletes
                // `x` past the loop), and, for the reads only, every
                // state a return or raise inside may leave from. A
                // finally that itself returns, raises, breaks, or
                // continues overrides the pending exit.
                let body = scan_deleted_paths(&t.body, &deleted)?;
                let at_handler = body.anywhere.clone();
                let mut reached = at_handler.clone();
                let mut pending_breaks = body.breaks;
                let mut pending_continues = body.continues;
                out.anywhere.extend(body.anywhere);
                let mut exits: Vec<Option<Deleted>> = vec![];
                let body_falls = body.falls;
                for handler in &t.handlers {
                    // `except E as e` binds `e` for the handler's body
                    // and UNBINDS it on every way out of the handler
                    // (Python's implicit `del e`), so a later read of
                    // `e` is Python's NameError.
                    let mut entry = at_handler.clone();
                    if let Some(name) = &handler.name {
                        entry.remove(name);
                    }
                    let mut handled = scan_deleted_paths(&handler.body, &entry)?;
                    if let Some(name) = &handler.name {
                        if let Some(falls) = &mut handled.falls {
                            falls.insert(name.clone());
                        }
                        for state in handled.breaks.iter_mut().chain(handled.continues.iter_mut()) {
                            state.insert(name.clone());
                        }
                        handled.anywhere.insert(name.clone());
                    }
                    reached.extend(handled.anywhere.iter().cloned());
                    out.anywhere.extend(handled.anywhere);
                    pending_breaks.extend(handled.breaks);
                    pending_continues.extend(handled.continues);
                    exits.push(handled.falls);
                }
                if let Some(falls) = &body_falls {
                    let orelse = scan_deleted_paths(&t.orelse, falls)?;
                    reached.extend(orelse.anywhere.iter().cloned());
                    out.anywhere.extend(orelse.anywhere);
                    pending_breaks.extend(orelse.breaks);
                    pending_continues.extend(orelse.continues);
                    exits.push(orelse.falls);
                }
                let continuing = merge_deleted(exits);
                if t.finalbody.is_empty() {
                    out.breaks.extend(pending_breaks);
                    out.continues.extend(pending_continues);
                    continuing
                } else {
                    let next = match &continuing {
                        Some(state) => out.absorb(scan_deleted_paths(&t.finalbody, state)?),
                        None => None,
                    };
                    for state in &pending_breaks {
                        if let Some(after) = out.absorb(scan_deleted_paths(&t.finalbody, state)?) {
                            out.breaks.push(after);
                        }
                    }
                    for state in &pending_continues {
                        if let Some(after) = out.absorb(scan_deleted_paths(&t.finalbody, state)?) {
                            out.continues.push(after);
                        }
                    }
                    // The return and raise paths: the finally's reads
                    // are checked from every state the try reached; its
                    // fall-through goes nowhere.
                    out.absorb(scan_deleted_paths(&t.finalbody, &reached)?);
                    next
                }
            }
            _ => {
                // Any other body-carrying form: every body from the
                // incoming state, the union afterwards.
                let bodies = visit::stmt_bodies(stmt);
                if bodies.is_empty() {
                    Some(deleted.clone())
                } else {
                    let mut exits = vec![Some(deleted.clone())];
                    for nested in bodies {
                        exits.push(out.absorb(scan_deleted_paths(nested, &deleted)?));
                    }
                    merge_deleted(exits)
                }
            }
        };
        match next {
            Some(state) => deleted = state,
            None => return Ok(out),
        }
    }
    out.anywhere.extend(deleted.iter().cloned());
    out.falls = Some(deleted);
    Ok(out)
}

/// Issue #115: every name this function's body declares `global`,
/// recursing through control flow but NOT into nested defs (each def is
/// its own scope with its own declarations).
fn collect_global_decls(body: &[Statement]) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    walk_stmts(body, Descend::SkipDefs, &mut |stmt| {
        if let StatementType::Global(names) = &stmt.statement {
            out.extend(names.iter().cloned());
        }
        Flow::Continue
    });
    out
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
    // `global __doc__` + docstring mutation (requests' status_codes._init)
    // is metadata — tolerated.
    declared
        .into_iter()
        .find(|n| written.contains(n) && n != "__doc__")
}

/// Collect `name = <simply-typed constant>` assignments (recursing into
/// control-flow bodies) so returns of those names can be inferred too.
/// Collect the types of local names assigned in a statement list: annotated
/// locals and simple-literal locals. `symbols`/`options` enable CALL-return
/// resolution (`proxy = parse_url(...)` → the callee's `-> Url`), so a
/// field stored from the local (`self.proxy = proxy` — urllib3's
/// ProxyManager) gets the class struct type.
/// Map a rendered Rust type to the Python type name `local_types`
/// records. ONE map: the producer (the local_types seeding below) and the
/// consumer (call.rs's isinstance comparison against local_types) MUST
/// yield the same name for the same type — they previously kept separate
/// copies whose fallbacks disagreed.
/// The TypeInfo twin of [`rust_type_to_py_name`] — the local-types layer
/// carries TypeInfo (issue #137's review), so the python-name mapping
/// matches structurally instead of re-parsing rendered tokens.
pub(crate) fn typeinfo_to_py_name(ty: &crate::TypeInfo) -> Option<&'static str> {
    Some(match ty {
        crate::TypeInfo::Int => "int",
        crate::TypeInfo::Float => "float",
        crate::TypeInfo::Bool => "bool",
        crate::TypeInfo::Bytes => "bytes",
        crate::TypeInfo::StrRef | crate::TypeInfo::String => "str",
        _ => return None,
    })
}

pub(crate) fn rust_type_to_py_name(ty: &proc_macro2::TokenStream) -> Option<&'static str> {
    let s = ty.to_string();
    Some(match s.as_str() {
        "i64" => "int",
        "f64" => "float",
        "bool" => "bool",
        "Vec < u8 >" => "bytes",
        _ if s.contains("str") || s.contains("String") => "str",
        _ => return None,
    })
}

/// The obviously-inferable types of a body's locals, by name: an
/// annotated or literal-valued assignment, a container literal, a local
/// import, a bare annotated name — recursing through control flow but not
/// into nested defs/classes (their locals are their own).
pub(crate) fn collect_local_types(
    body: &[Statement],
    out: &mut std::collections::HashMap<String, crate::TypeInfo>,
) {
    walk_stmts(body, Descend::SkipDefs, &mut |stmt| {
        if let StatementType::Assign(assign) = &stmt.statement {
            if let [ExprType::Name(name)] = assign.targets.as_slice() {
                // An annotated local (`flags: int = _character_flags(c)`)
                // types from the annotation.
                if let Some(ann) = assign.annotation.as_ref()
                    && let Some(t) = crate::annotation_type_info(ann)
                {
                    out.insert(name.id.clone(), t);
                } else if let Some(ty) = simple_expr_typeinfo(&assign.value) {
                    out.insert(name.id.clone(), ty);
                } else {
                    // A CONTAINER literal local types like the literal
                    // lowering (issue #180): a dict gets String keys and
                    // boxes heterogeneous-but-boxable values (`{
                    // 'ProviderType': 'sso', 'Credentials': {...}}` —
                    // botocore), a list gets its concrete element type.
                    // Without this, an unannotated function returning
                    // such a local collapses to unit and every use of
                    // the returned container breaks.
                    match &assign.value {
                        ExprType::Dict(d) if !d.keys.is_empty() => {
                            if let crate::TypeInfo::Dict(k, v) =
                                crate::syntactic_type(&assign.value)
                            {
                                let k = if matches!(
                                    *k,
                                    crate::TypeInfo::StrRef | crate::TypeInfo::PyObject
                                ) {
                                    crate::TypeInfo::String
                                } else {
                                    *k
                                };
                                let v = if matches!(*v, crate::TypeInfo::PyObject) {
                                    crate::TypeInfo::PyValue
                                } else {
                                    *v
                                };
                                out.insert(
                                    name.id.clone(),
                                    crate::TypeInfo::Dict(Box::new(k), Box::new(v)),
                                );
                            }
                        }
                        ExprType::List(l) if !l.is_empty() => {
                            if let crate::TypeInfo::Vec(elt) =
                                crate::syntactic_type(&assign.value)
                            {
                                if !matches!(*elt, crate::TypeInfo::PyObject) {
                                    out.insert(name.id.clone(), crate::TypeInfo::Vec(elt));
                                }
                            }
                        }
                        // An EMPTY container local (`allowed = {}` then
                        // `allowed[alg] = ...` — pip's Hashes): the
                        // boxed heterogeneous container (the element
                        // types are unknowable at the store).
                        ExprType::Dict(d) if d.keys.is_empty() => {
                            out.insert(
                                name.id.clone(),
                                crate::TypeInfo::Dict(
                                    Box::new(crate::TypeInfo::String),
                                    Box::new(crate::TypeInfo::PyValue),
                                ),
                            );
                        }
                        ExprType::List(l) if l.is_empty() => {
                            out.insert(
                                name.id.clone(),
                                crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue)),
                            );
                        }
                        _ => {}
                    }
                }
            }
        } else if let StatementType::Import(im) = &stmt.statement {
            // A local IMPORT (`import keyring` inside __init__, then
            // `self.keyring = keyring` — pip's KeyRingPythonProvider): a
            // module object — a boxed value.
            for a in &im.names {
                out.insert(a.name.clone(), crate::TypeInfo::PyValue);
            }
        } else if let StatementType::AnnotatedName { name, annotation } = &stmt.statement
            && let Some(t) = crate::annotation_type_info(annotation)
        {
            // A bare annotated local (`key: str` / `value: str` — urllib3's
            // ssl_match_hostname): types the name for downstream use
            // (dnsnames.append(value) pins Vec<String>).
            out.insert(name.clone(), t);
        }
        Flow::Continue
    });
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
/// Whether `expr`'s VALUE is already an Option — `expr_yields_option`
/// plus a FIELD READ whose field is Option-typed (`destination_scheme =
/// parsed_url.scheme`, `ca_cert_dir=self.ca_cert_dir` — urllib3): the
/// field lowers to its accessor, which returns the Option, so a store or
/// argument wrap would nest (`Some(parsed_url.scheme)` ->
/// Option<Option<String>>, the retrospective's R2 double-wrap family —
/// ~70 rustc errors at the round-57 sweep). Resolves the receiver's
/// class (self fields, typed params, and factory-assigned locals like
/// `u = parse_url(url)` all route through receiver_class) and consults
/// the field type. The ctx is used only for `self` receivers.
pub(crate) fn expr_yields_option_ctx(
    expr: &ExprType,
    ctx: &crate::CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    if expr_yields_option(expr, options, symbols) {
        return true;
    }
    // A `typing.cast(T, value)` call is a runtime identity: it yields
    // exactly what its VALUE yields (`proxy_config = cast(ProxyConfig,
    // self.proxy_config)` — a `ProxyConfig | None` field read — must
    // pass the Option through the Option-slot store, or the wrap nests
    // into Option<Option<ProxyConfig>>, round 95).
    if let ExprType::Call(call) = expr
        && call.args.len() == 2
        && (matches!(
            call.func.as_ref(),
            ExprType::Name(n)
                if n.id == "cast"
                    && matches!(
                        symbols.get(&n.id),
                        Some(crate::SymbolTableNode::ImportFrom(i))
                            if crate::AnnotationModule::from_name(
                                i.module.split('.').next().unwrap_or("")
                            ) == Some(crate::AnnotationModule::Typing)
                    )
        ) || matches!(
            call.func.as_ref(),
            ExprType::Attribute(attr)
                if attr.attr == "cast"
                    && matches!(attr.value.as_ref(), ExprType::Name(m) if crate::is_typing(&m.id))
        ))
    {
        return expr_yields_option_ctx(&call.args[1], ctx, options, symbols);
    }
    // Whether a class (or its BASE chain) has an Option-typed field
    // `name` — a `self.<field>` read or accessor call of an inherited
    // field (`self._tunnel_host` in a derived method whose struct embeds
    // the defining base) yields the Option either way.
    let class_field_is_option =
        |name: &str, class: &crate::ClassDef, class_symbols: &SymbolTableScopes| -> bool {
            class
                .base_chain_with_options(class_symbols, options)
                .iter()
                .any(|c| {
                    c.infer_fields(class_symbols, options)
                        .ok()
                        .is_some_and(|fields| {
                            fields.iter().any(|(n, t)| {
                                *n == name && matches!(t, crate::TypeInfo::Option(_))
                            })
                        })
                })
        };
    // A FIELD READ of a BOXED self-referential field (`node.left` where
    // the slot is Option<Box<Node>> — the corpus's contains, round 99):
    // the read yields the Option<Node> the boxing derefs to — a store
    // into the boxed slot re-boxes it. Scoped to the boxed-self-ref shape
    // so the broad field-read passes through stay unchanged.
    if let ExprType::Attribute(attr) = expr
        && let Some(c) = crate::TypeInfo::enum_receiver_class(
            &attr.value, Some(ctx), options, symbols,
        )
        && crate::TypeInfo::enum_receiver_class(&attr.value, Some(ctx), options, symbols)
            .and_then(|rc| crate::resolve_class_referenced(&rc, symbols, options))
            .and_then(|class| {
                class
                    .infer_fields(symbols, options)
                    .ok()
                    .and_then(|fields| {
                        fields
                            .iter()
                            .find(|(name, _)| name == &attr.attr)
                            .map(|(_, t)| t.clone())
                    })
            })
            .is_some_and(|t| {
                matches!(
                    t,
                    crate::TypeInfo::Option(ref inner)
                        if matches!(
                            **inner,
                            crate::TypeInfo::Class(ref fc) if *fc == c
                        )
                )
            })
    {
        return true;
    }
    // A CONDITIONAL whose arms yield Options (`node = node.left if key <
    // node.key else node.right` — the idiom corpus's contains, round 99):
    // CPython's result is one of the arms — an Option when EITHER is.
    if let ExprType::IfExp(e) = expr {
        return expr_yields_option_ctx(&e.body, ctx, options, symbols)
            || expr_yields_option_ctx(&e.orelse, ctx, options, symbols);
    }
    // A SELF-FIELD ACCESSOR CALL (`self._tunnel_host()` — the field's
    // generated getter): an Option-typed field's accessor returns the
    // Option, so the call yields one — a store into an Option local must
    // pass it through, never Some-wrap (`server_hostname =
    // self._tunnel_host` where the local was widened to Option — round
    // 70). The accessor's name is the field's.
    if let ExprType::Call(call) = expr
        && let ExprType::Attribute(attr) = call.func.as_ref()
        && let Some((class, class_symbols)) =
            crate::receiver_class(&attr.value, ctx, symbols, options)
        && class_field_is_option(&attr.attr, &class, &class_symbols)
    {
        return true;
    }
    // A SELF-METHOD call returning an Option (`item = self.find(name)` —
    // a `-> Optional[Item]` finder the caller narrows with an early-exit
    // guard): the result IS the Option — a store into an Option local
    // must pass it through, never Some-wrap (the corpus's take() would
    // nest `Some(Option<Item>)`, round 97). The field-accessor arm above
    // covers OPTION FIELDS; this covers METHODS whose return annotation
    // is an Option.
    // A METHOD CALL whose declared return is optional (`self.find(k)`,
    // `board.find(k)` on a class-typed local — the corpus's ledger): the
    // receiver's class resolves through the read path (`self` through
    // the enclosing class), the method on its MRO, the annotation in the
    // class's scope.
    if let ExprType::Call(call) = expr
        && let ExprType::Attribute(attr) = call.func.as_ref()
        && let Some((owner, owner_symbols)) =
            crate::receiver_class(&attr.value, ctx, symbols, options)
        && owner
            .method_on_mro(&attr.attr, &owner_symbols)
            .and_then(|m| m.returns.as_deref().cloned())
            .is_some_and(|r| {
                matches!(
                    crate::resolve_alias_typeinfo(&r, &owner_symbols, options),
                    Some(crate::TypeInfo::Option(_))
                )
            })
    {
        return true;
    }
    // itself an Option — the mapping protocol's get returns None when the
    // key is absent (`headers.get(name, default=None)` — urllib3's
    // getheader). The get lowering's OWN Some-wrap makes the value the
    // union member, so a return/store of it must pass the Option through,
    // never Some-wrap again.
    if let ExprType::Call(call) = expr
        && let ExprType::Attribute(attr) = call.func.as_ref()
        && attr.attr == "get"
        && call.args.len() == 2
        && (crate::is_none_expr(&call.args[1])
            || crate::expr_yields_option_ctx(&call.args[1], ctx, options, symbols))
    {
        return true;
    }
    let ExprType::Attribute(attr) = expr else {
        return false;
    };
    // A PROPERTY read whose getter's return annotation is `T | None`
    // (`self.url` where `@property def url(self) -> str | None` —
    // urllib3's HTTPResponse.url) yields the Option: the property's VALUE
    // is the union member, never a plain member. The READ-flavored
    // receiver resolution (not the conservative receiver_class) so a
    // local assigned from a SELF-METHOD factory (`timeout_obj =
    // self._get_timeout()` — round 87) resolves its class: receiver_class
    // hard-returns None on that Assign shape (an attribute callee), and
    // the property read would double-wrap `Some(timeout_obj.read_timeout()?)`
    // instead of passing the Option through.
    if let Some((class, class_symbols)) =
        crate::receiver_class_for_read(&attr.value, ctx, symbols, options)
        && class
            .base_chain_with_options(&class_symbols, options)
            .iter()
            .any(|c| {
                c.methods().any(|m| {
                    m.name == attr.attr
                        && m.decorator_list.iter().any(|d| match d {
                            ExprType::Name(n) => n.id == "property",
                            ExprType::Attribute(a) => a.attr == "property",
                            _ => false,
                        })
                        && m.returns
                            .as_deref()
                            .is_some_and(crate::is_optional_annotation)
                })
            })
    {
        return true;
    }
    // An OPTION-typed FIELD read on a class-resolved receiver
    // (`self.proxy.host` — Url's `host` field is `str | None`, the
    // proxy field resolves through the ProxyManager class table; round
    // 89's `super().connection_from_host(self.proxy.host, ...)`): the
    // field read yields the Option (the accessor returns it), so an
    // Option-slot argument must pass it through — `Some(self.proxy().host)`
    // would double it into `Option<Option<String>>`. The property arm
    // above covers accessor METHODS; this arm covers plain FIELDS on any
    // receiver (the self-field accessor-call arm above covers the
    // `self._tunnel_host()` shape).
    if let Some((class, class_symbols)) =
        crate::receiver_class_for_read(&attr.value, ctx, symbols, options)
        && class
            .base_chain_with_options(&class_symbols, options)
            .iter()
            .any(|c| {
                c.infer_fields(&class_symbols, options)
                    .ok()
                    .is_some_and(|fields| {
                        fields.iter().any(|(n, t)| {
                            *n == attr.attr && matches!(t, crate::TypeInfo::Option(_))
                        })
                    })
            })
    {
        return true;
    }
    // The receiver may be an OPTION-CLASS-typed LOCAL (`proxy_config`
    // where `proxy_config = cast(ProxyConfig, self.proxy_config)` — the
    // walk seeds it Option<ProxyConfig>, and the attr-read lowering
    // unwraps the Option on the read, so `X.assert_fingerprint` is a
    // field of the CLASS — round 95): receiver_class_for_read cannot
    // resolve an Option-wrapped name, so look the inner class's field
    // table up directly.
    if let ExprType::Name(n) = attr.value.as_ref()
        && let Some(crate::TypeInfo::Option(inner)) = options.name_types.get(&n.id)
        && let crate::TypeInfo::Class(cname) = &**inner
        && let Some(class) = crate::resolve_class_referenced(cname, symbols, options)
        && class
            .base_chain_with_options(symbols, options)
            .iter()
            .any(|c| {
                c.infer_fields(symbols, options)
                    .ok()
                    .is_some_and(|fields| {
                        fields.iter().any(|(fn_, t)| {
                            *fn_ == attr.attr && matches!(t, crate::TypeInfo::Option(_))
                        })
                    })
            })
    {
        return true;
    }
    let Some((class, class_symbols)) =
        crate::receiver_class(&attr.value, ctx, symbols, options)
        .or_else(|| {
            // `self.proxy().host` — the receiver is a METHOD CALL whose
            // return annotation is `T | None` (an Option-typed property
            // accessor, `@property def proxy(self) -> Proxy | None`): the
            // receiver is the Option of T, and the field read is T's
            // field. Resolve the inner class through the method's
            // annotation.
            if let ExprType::Call(call) = attr.value.as_ref()
                && let ExprType::Attribute(fn_attr) = call.func.as_ref()
                && crate::ast::tree::visit::is_self(fn_attr.value.as_ref())
                && let Some(class_name) = ctx.enclosing_class_name()
                && let Some(crate::SymbolTableNode::ClassDef(owner)) = symbols.get(class_name)
                && let Some(method) = owner.method_on_mro(&fn_attr.attr, symbols)
                && let Some(ann) = method.returns.as_deref()
                && crate::is_optional_annotation(ann)
                && let Some(inner) = match ann {
                    ExprType::BinOp(op) if crate::is_none_expr(&op.right) => {
                        Some(op.left.as_ref())
                    }
                    ExprType::BinOp(op) if crate::is_none_expr(&op.left) => {
                        Some(op.right.as_ref())
                    }
                    _ => None,
                }
                && let ExprType::Name(inner_name) = inner
            {
                crate::receiver_class_tail(&inner_name.id, symbols.clone(), options)
            } else {
                None
            }
        })
    else {
        return false;
    };
    class_field_is_option(&attr.attr, &class, &class_symbols)
}

pub(crate) fn expr_yields_option(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match expr {
        // A name that itself holds an Option (assigned None on some path,
        // an Optional-annotated parameter, or a local whose INFERRED type
        // is an Option — `netloc = self.netloc()?` in a Url method, where
        // netloc() returns `Result<Option<String>, _>`; the `?` strips the
        // Result layer, leaving an Option).
        ExprType::Name(name) => {
            options.optional_names.contains(&name.id)
                || matches!(
                    options.name_types.get(&name.id),
                    Some(crate::TypeInfo::Option(_))
                )
        }
        ExprType::Call(call) => match call.func.as_ref() {
            // dict.get(k) lowers to py_get, which returns Option<V>; the
            // TWO-argument form with a None default (`headers.get(name,
            // default=None)` — urllib3's getheader/get_redirect_location)
            // is Option too (round 61b).
            ExprType::Attribute(attr) => {
                attr.attr == "get"
                    && (call.args.len() == 1
                        || (call.args.len() == 2 && crate::is_none_expr(&call.args[1])))
            }
            // A user function annotated `-> Optional[T]` generates
            // `Result<Option<T>, PyException>`; the call site's `?` strips
            // only the Result layer, leaving an Option. An IMPORTED
            // function (`from .utils import unicode_range` —
            // charset_normalizer's cd.py, whose callee returns `str |
            // None`) resolves the defining FunctionDef through the module
            // defs and checks the same annotation — without it a store
            // Some-wraps the already-Option result (Option<Option<T>>,
            // round 75).
            ExprType::Name(name) => match symbols.get(&name.id) {
                Some(SymbolTableNode::FunctionDef(f)) => f
                    .returns
                    .as_deref()
                    .is_some_and(crate::is_optional_annotation)
                    // Round 85 (the return-type directive): an UNANNOTATED
                    // callee whose body can return exactly `T | None`
                    // infers an `Option<T>` return — the caller's store
                    // analysis must see the Option so the value is an
                    // Option binding (narrowing, Option-access unwraps,
                    // and the Option→concrete coercions all apply).
                    || (f.returns.is_none()
                        && f.inferred_return_typeinfo(symbols, options).is_some_and(
                            |t| matches!(t, crate::TypeInfo::Option(_)),
                        )),
                Some(SymbolTableNode::ImportFrom(ifm))
                    if !crate::ast::tree::import::is_stdpython_module(&ifm.module) =>
                {
                    let path = ifm.resolved_module_path(options);
                    let Some(key) = crate::module_defs_key(options, &path) else {
                        return false;
                    };
                    options.module_defs.get(key).is_some_and(|m| {
                        m.raw.body.iter().any(|s| {
                            matches!(&s.statement, crate::StatementType::FunctionDef(f)
                                if f.name == name.id
                                    && f.returns
                                        .as_deref()
                                        .is_some_and(crate::is_optional_annotation))
                        })
                    })
                }
                // socket.getdefaulttimeout/setdefaulttimeout return the
                // default as `float | None` (Option<f64>) — the runtime
                // free functions' shape (urllib3's Timeout module).
                Some(SymbolTableNode::ImportFrom(ifm))
                    if crate::StdModule::from_name(&ifm.module)
                        == Some(crate::StdModule::Socket)
                        && matches!(
                            name.id.as_str(),
                            "getdefaulttimeout" | "setdefaulttimeout"
                        ) =>
                {
                    true
                }
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
        // `a and b` / `a or b` yield an Option when an operand is an
        // Option: the operand-returning fold returns the Option (or
        // wraps the plain arm), so the BoolOp's value IS an Option — a
        // store into an Option slot passes it through instead of
        // double-wrapping (`self.ca_certs = ca_certs and
        // expanduser(ca_certs)` — urllib3).
        ExprType::BoolOp(bo) => bo
            .values
            .iter()
            .any(|v| expr_yields_option(v, options, symbols)),
        _ => false,
    }
}

/// Whether an expression already yields a boxed PyValue (issue #121): a
/// PyValue-annotated name or a call returning one. Such values store into
/// PyValue slots through unchanged — wrapping again would nest PyValue
/// inside PyValue.
pub(crate) fn expr_yields_pyvalue(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    match expr {
        ExprType::Name(name) => {
            // A narrowed read converts to the member type (String, i64,
            // ...) — no longer a PyValue. A name narrowed back to PyValue
            // itself (the else of a compound isinstance test) still is.
            match options.narrowed_names.get(&name.id) {
                Some(t) => matches!(t, crate::TypeInfo::PyValue),
                None => options
                    .name_types
                    .get(&name.id)
                    .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue)),
            }
        }
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(name) => match symbols.get(&name.id) {
                Some(SymbolTableNode::FunctionDef(f)) => f.returns.as_deref().is_some_and(|r| {
                    crate::annotation_type_info(r)
                        .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue))
                }),
                _ => false,
            },
            _ => false,
        },
        _ => false,
    }
}

/// Whether a function body contains a `yield`/`yield from` anywhere in
/// its own scope (control-flow nested, and inside a larger expression —
/// `x = yield v`, `f((yield))`; a nested def's yields make THAT function
/// the generator): the body is a GENERATOR and lowers to
/// build-and-return-a-list (issue #122-family).
pub(crate) fn body_has_yields(body: &[Statement]) -> bool {
    any_expr_in(body, Descend::OwnScope, |e| {
        matches!(e, ExprType::Yield(_) | ExprType::YieldFrom(_))
    })
}

/// The first statement of `body` (nested defs and lambdas excluded — a
/// yielding lambda is its own generator) that uses a `yield` / `yield
/// from` as a VALUE rather than as its own statement: `x = yield v`,
/// `(yield v) or 1`, `f((yield))`, `yield (yield v)`. That is the
/// generator's send channel, which the build-and-return-a-list lowering
/// cannot model.
fn expression_position_yield(body: &[Statement]) -> Option<&Statement> {
    let is_yield = |e: &ExprType| matches!(e, ExprType::Yield(_) | ExprType::YieldFrom(_));
    let own = Descend::OwnScope;
    let mut found = None;
    visit::any_stmt(body, own, |s| {
        let hit = match &s.statement {
            // A statement-level yield: only a yield NESTED in its value
            // counts.
            StatementType::Expr(e) if is_yield(&e.value) => visit::subexprs_for(&e.value, own)
                .into_iter()
                .any(|sub| visit::any_expr_for(sub, own, is_yield)),
            _ => stmt_all_exprs(s)
                .into_iter()
                .any(|e| visit::any_expr_for(e, own, is_yield)),
        };
        if hit {
            found = Some(s);
        }
        hit
    });
    found
}

/// The element type of a generator body: from the `Generator[T, ...]` /
/// `Iterator[T]` return annotation (Python's convention), else the first
/// yielded value's inferred type.
pub(crate) fn generator_element_type(
    returns: Option<&ExprType>,
    body: &[Statement],
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<crate::TypeInfo> {
    // Both the bare and the typing-qualified spellings, with the yield
    // type as a tuple's head (`Generator[bytes, None, None]`) or alone
    // (`typing.Generator[bytes]` — urllib3's response.stream).
    if let Some(ann) = returns
        && let ExprType::Subscript(sub) = ann
        && match sub.value.as_ref() {
            ExprType::Name(n) => matches!(n.id.as_str(), "Generator" | "Iterator"),
            ExprType::Attribute(a) => {
                matches!(a.value.as_ref(), ExprType::Name(m) if crate::is_typing(&m.id))
                    && matches!(a.attr.as_str(), "Generator" | "Iterator")
            }
            _ => false,
        }
        && let crate::SubscriptKind::Index(elt) = &sub.kind
    {
        let first = match elt.as_ref() {
            ExprType::Tuple(t) => t.elts.first(),
            other => Some(other),
        };
        if let Some(first) = first {
            if let Some(t) = crate::resolve_alias_typeinfo(first, symbols, options) {
                return Some(t);
            }
            if let Some(t) = crate::annotation_type_info(first) {
                return Some(t);
            }
        }
    }
    first_yield_type(body, options, symbols)
}

/// The inferred type of the first `yield <value>` (in source order,
/// anywhere in the function's own scope) whose value infers to a
/// concrete type.
fn first_yield_type(
    body: &[Statement],
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<crate::TypeInfo> {
    let mut found = None;
    any_expr_in(body, Descend::OwnScope, |e| {
        if let ExprType::Yield(y) = e
            && let Some(v) = y.value.as_ref()
        {
            let t = crate::infer_type(None, v, options, symbols);
            if !matches!(t, crate::TypeInfo::PyObject) {
                found = Some(t);
                return true;
            }
        }
        false
    });
    found
}

/// Whether `param` is a None-defaulted, unannotated parameter of `func`
/// (issue #117): the default is the None literal and there is no
/// annotation to pin a type. Such parameters are concrete `Option<()>`
/// unless their value is used (round 33), in which case they carry the
/// boxed `PyValue`. Shared by the callee's signature generation and the
/// call site's argument coercion, so the two cannot disagree.
pub(crate) fn param_has_none_default(param: &crate::Parameter, func: &crate::FunctionDef) -> bool {
    if param.annotation.is_some() {
        return false;
    }
    let pos_params: Vec<&crate::Parameter> = func
        .args
        .posonlyargs
        .iter()
        .chain(func.args.args.iter())
        .collect();
    let from = func.args.posonlyargs.len() + func.args.args.len() - func.args.defaults.len();
    if let Some(pos) = pos_params.iter().position(|q| q.arg == param.arg) {
        return pos >= from
            && func
                .args
                .defaults
                .get(pos - from)
                .is_some_and(|d| crate::is_none_expr(d));
    }
    let kw = func
        .args
        .kwonlyargs
        .iter()
        .zip(func.args.kw_defaults.iter())
        .find(|(q, _)| q.arg == param.arg);
    kw.is_some_and(|(_, d)| d.as_deref().is_some_and(|d| crate::is_none_expr(d)))
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
    boxed: bool,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // Conditionals recurse per arm FIRST: even when one arm makes the whole
    // expression Option-typed (e.g. an `else None`), the other arm may be a
    // plain value that still needs its Some wrap.
    if let ExprType::IfExp(e) = expr {
        let test =
            crate::condition_to_rust(&e.test, ctx.clone(), options.clone(), symbols.clone())?;
        let body = lower_optional_value(&e.body, ctx.clone(), options.clone(), symbols.clone(), boxed)?;
        let orelse = lower_optional_value(&e.orelse, ctx, options, symbols, boxed)?;
        return Ok(quote!(if #test { #body } else { #orelse }));
    }
    if is_none_expr(expr) || expr_yields_option(expr, &options, &symbols) {
        // An Option-valued NAME read more than once takes the reuse-clone
        // like any other non-Copy name (`same(ta, tb)` then `same(ta, ..)`
        // — the corpus's ledger, an `Option<PyRef<Tag>>` local).
        return crate::render_reused(expr, ctx, options, symbols);
    }
    // A NAME whose recorded type is itself an Option (`host =
    // _normalize_host(...)` where the callee returns `str | None` —
    // urllib3's parse_url) passes through an Option slot unwrapped: the
    // value already IS the Option. The runtime Option-slot machinery
    // tracks None-assigned names and Optional params in optional_names,
    // but a local assigned from an Option-returning CALL only lands in
    // name_types (round 47) — wrapping it in Some would nest.
    // A NAME whose recorded type is itself an Option (`host =
    // _normalize_host(...)` where the callee returns `str | None` —
    // parse_url; `server_hostname: str = self.host` widened by a later
    // Option-valued store) passes through an Option slot unwrapped: the
    // value already IS the Option. Consult name_types DIRECTLY — a name
    // with an annotation in local_types would otherwise report its
    // annotated (plain) type and wrap again.
    if let ExprType::Name(n) = expr
        && matches!(
            options.name_types.get(&n.id),
            Some(crate::TypeInfo::Option(_))
        )
    {
        return crate::render_reused(expr, ctx, options, symbols);
    }
    // A FIELD read whose field is Option-typed (`ca_cert_dir=
    // self.ca_cert_dir` — urllib3's _ssl_wrap_socket call sites): the
    // field lowers to its accessor call, which already RETURNS the
    // Option — wrapping it in Some would nest (`Some(self.ca_cert_dir())`
    // -> Option<Option<String>>, the retrospective's R2 double-wrap
    // family). The ctx-aware predicate resolves the receiver's class
    // (self fields, typed params, factory-assigned locals).
    if crate::expr_yields_option_ctx(expr, &ctx, &options, &symbols) {
        let tokens = expr.clone().to_rust(ctx.clone(), options, symbols)?;
        // The pass-through MOVES the Option out of the receiver: a
        // `self.<field>` read borrows `&self` (E0507 — `headers =
        // self.headers` where the field is Option<IndexMap>, urllib3's
        // _request_methods), and a field read on a composed receiver
        // (`self.proxy.host` — the ProxyManager super call, round 90)
        // borrows the composed object the same way. Clone the value out —
        // the Python object is shared by reference, so the clone is the
        // faithful copy.
        if matches!(expr, ExprType::Attribute(_)) {
            // A boxed slot (round 99): the read's Option<Box<Class>>
            // map-derefs to Option<Class> — the slot's type.
            let read = if boxed {
                quote!((#tokens).clone().map(| b | * b))
            } else {
                quote!((#tokens).clone())
            };
            return Ok(read);
        }
        let read = if boxed {
            quote!((#tokens).map(| b | * b))
        } else {
            tokens
        };
        return Ok(read);
    }
    let tokens = expr.clone().to_rust(ctx, options, symbols)?;
    // A string LITERAL lowers to `&'static str`; an Option<String> slot
    // owns it (`pick("x")` where the parameter is `str | None`) — the
    // same ownership the `-> str` return path applies (issue #137's
    // and/or round).
    if matches!(expr, ExprType::Constant(c)
        if matches!(&c.0, Some(litrs::Literal::String(_))))
    {
        return Ok(quote!(Some((#tokens).to_string())));
    }
    // A SELF-REFERENTIAL boxed slot (the annotation is
    // `Optional[Node]` inside Node — round 99, E0072): the binding holds
    // Option<Box<Node>> and the plain-value store wraps in
    // Some(Box::new(...)).
    if boxed {
        return Ok(quote!(Some(Box::new((#tokens).clone()))));
    }
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
        // A `with` body's return IS the function's return (context
        // managers only intercept exceptions, never returns — requests'
        // api.request; issue #137).
        Some(StatementType::With(s)) => guarantees_return(&s.body),
        Some(StatementType::AsyncWith(s)) => guarantees_return(&s.body),
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
    /// Whether this method is a property SETTER (`@x.setter def x(...)` —
    /// the `setter` decorator spelling). Only the setter half of a
    /// getter/setter pair gets the distinct Rust name `{name}_set`; the
    /// getter (`@property def x`) keeps its name.
    pub fn is_property_setter(&self) -> bool {
        self.decorator_list.iter().any(|d| match d {
            ExprType::Attribute(a) => {
                a.attr == "setter"
                    && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == self.name)
            }
            _ => false,
        })
    }

    /// The class a factory function returns, for receiver resolution
    /// (`history_recorder = get_global_history_recorder()` then
    /// `.record(...)` — botocore's client.py). The `-> ClassName`
    /// annotation when it names a type; else — for the unannotated
    /// lazy-singleton shape (issue #189) — the single class-instance
    /// module global every return reads. None when neither applies.
    pub fn return_class_name(&self, options: &crate::PythonOptions) -> Option<String> {
        if let Some(ann) = self.returns.as_ref() {
            return match ann.as_ref() {
                ExprType::Name(r) => Some(r.id.clone()),
                _ => None,
            };
        }
        // Unannotated: every return must be the same Class-kind mutable
        // static (a bare/implicit None return bails — the receiver could
        // be None, which no class type represents).
        let mut returns = Vec::new();
        collect_returns(&self.body, &mut returns);
        let mut class: Option<String> = None;
        for ret in &returns {
            let value = (*ret)?;
            let ExprType::Name(name) = value else {
                return None;
            };
            match options.mutable_statics.get(&name.id) {
                Some(crate::MutableGlobalKind::Class { class: c }) => {
                    if class.is_some() && class.as_deref() != Some(c.as_str()) {
                        return None;
                    }
                    class = Some(c.clone());
                }
                _ => return None,
            }
        }
        class
    }

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
    pub fn resolved_return_type(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<TokenStream> {
        self.resolved_return_type_in(symbols, options, None)
    }

    /// [`resolved_return_type`] with the ENCLOSING CLASS (issue #222): a
    /// method whose returns are calls to other methods of its own class
    /// (`return self._retries()`) can only be typed when the class is
    /// known. Callers inside the lowering pass `ctx.enclosing_class_name()`;
    /// class-less callers use the plain [`resolved_return_type`].
    pub fn resolved_return_type_in(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
        self_class: Option<&str>,
    ) -> Option<TokenStream> {
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
                    crate::python_annotation_to_rust_type(ann).or_else(|| {
                        // Round 82 (the generics directive): an EXTERNAL-module
                        // class annotation (`-> ssl.SSLSocket`,
                        // `-> logging.StreamHandler` — the function returns a
                        // real object rython cannot model) resolves to the
                        // boxed PyValue through the symbols-aware authority —
                        // the same external-object divergence the parameter
                        // and field sides use. The symbols-FREE
                        // python_annotation_to_rust_type above cannot see the
                        // import (`ssl` → external), so it returned None and
                        // the function silently typed `()` while its body
                        // returned a value — every caller of the return then
                        // mismatched (the `() | PyValue` family).
                        crate::resolve_alias_typeinfo(ann, symbols, options)
                            .map(|t| t.to_rust_type())
                            .or_else(|| {
                        // A user-class annotation (`-> Scheme`): the type is
                        // the class name rendered as a Rust ident — the same
                        // path parameters use. A bare Name that is NOT a
                        // class would fail loudly later at the call site;
                        // mapping it here makes `-> Scheme` + `return
                        // Scheme::new(...)` line up.
                        match ann {
                            ExprType::Name(n)
                                if !crate::ast::tree::assign::is_builtin_scalar_name(&n.id) =>
                            {
                                if n.id == "Self" {
                                    // typing.Self (`def __enter__(self) ->
                                    // Self` — requests' models): the
                                    // enclosing class — the Rust `Self`
                                    // impl-keyword is exactly that (the
                                    // safe_ident `Self_` escaping is wrong
                                    // inside an impl block).
                                    Some(quote!(Self))
                                } else if let Some(SymbolTableNode::ImportFrom(ifm)) =
                                    symbols.get(&n.id)
                                {
                                    let runtime_item =
                                        crate::ast::tree::module::module_def_has_runtime_item(
                                            options,
                                            &ifm.resolved_module_path(options),
                                            &n.id,
                                        );
                                    if runtime_item {
                                        let ident = crate::safe_ident(&n.id);
                                        Some(quote!(#ident))
                                    } else {
                                        // An imported name with NO runtime item
                                        // (`-> BaseHTTPConnection` — a
                                        // TYPE_CHECKING-only Protocol stub in
                                        // urllib3's _base_connection): the boxed
                                        // PyValue.
                                        Some(quote!(stdpython::PyValue))
                                    }
                                } else if matches!(
                                    symbols.get(&n.id),
                                    Some(SymbolTableNode::Assign { value, .. })
                                        if crate::ast::tree::type_ctx::is_typevar_call(value)
                                ) {
                                    // `-> T` where `T = TypeVar("T")`
                                    // (urllib3's http2 _LockedObject):
                                    // the boxed PyValue, matching the
                                    // parameter-position lowering — a
                                    // bare `T` names nothing in Rust.
                                    Some(quote!(stdpython::PyValue))
                                } else {
                                    let ident = crate::safe_ident(&n.id);
                                    Some(quote!(#ident))
                                }
                            }
                            _ => None,
                        }
                        })
                    })
                }
            })
        } else {
            None
        };
        // A @classmethod whose body returns `cls(...)`: the call constructs
        // an instance of the enclosing class (`def make(cls, v): return
        // cls(v)`), so the return type is Self. The simple-shape inferrer
        // below cannot type a cls-call, and without this rule the signature
        // collapses to unit and every use of the constructed value breaks.
        if guarantees_return(&self.body)
            && self.decorator_list.iter().any(|d| {
                matches!(d, ExprType::Name(n) if n.id == "classmethod")
            })
            && self.returns_cls_construction()
        {
            return Some(quote!(Self));
        }
        // The ANNOTATION is the contract and wins over the body's inferred
        // type: a `-> float | None` getter whose body returns a plain
        // literal (`return 0.5` — urllib3's Timeout.read_timeout) must
        // type as `Option<f64>` (the return-site Some-wrap and the
        // fn_return_is_option flag already agree), NOT the body's `f64` —
        // the inferred-first ordering let a plain return silently shrink
        // the annotated Option and every caller of the union typed the
        // bare member (round 87).
        annotated
            .or_else(|| self.inferred_return_type(options))
            .or_else(|| self.boxed_list_return_type(symbols, options))
            .or_else(|| self.unified_return_type(symbols, options))
            .or_else(|| self.module_path_call_return_type(symbols, options))
            .or_else(|| self.self_method_call_return_type(self_class, symbols, options))
            .or_else(|| {
                self.self_field_return_type(self_class, symbols, options)
                    .map(|t| t.to_rust_type())
            })
    }

    /// An unannotated METHOD whose returns are all reads of fields of its
    /// own class on `self` (`return self.scheme` — issue #222's deferred
    /// self-field half): the field's inferred type, from the same
    /// `infer_fields` table the struct declaration uses, so the signature
    /// and the struct agree by construction. END of the chain; all
    /// returns must agree. The MOVE side is separate: a non-Copy field
    /// read in a `return` clones out of the shared receiver
    /// (statement.rs), or the value would leave `&self` (E0507).
    fn self_field_return_type(
        &self,
        self_class: Option<&str>,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<crate::TypeInfo> {
        if !guarantees_return(&self.body) {
            return None;
        }
        let mut returns = Vec::new();
        crate::ast::tree::specialize::collect_return_exprs(&self.body, &mut returns);
        if returns.is_empty() {
            return None;
        }
        let class_name = self_class?;
        let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(class_name) else {
            return None;
        };
        let fields = class.infer_fields(symbols, options).ok()?;
        // A one-step local indirection (`box = self.scheme; return box` —
        // issue #222's local half): resolve the local's single assignment
        // in the method body and type through it. Deeper chains stay out
        // of scope (the local-type collector's own bucket).
        let assigned_self_field = |name: &str| -> Option<crate::TypeInfo> {
            let mut found: Option<crate::TypeInfo> = None;
            for s in &self.body {
                let crate::StatementType::Assign(a) = &s.statement else {
                    continue;
                };
                let [crate::ExprType::Name(n)] = a.targets.as_slice() else {
                    continue;
                };
                if n.id != name {
                    continue;
                }
                let crate::ExprType::Attribute(attr) = &a.value else {
                    continue;
                };
                if !crate::ast::tree::visit::is_self(attr.value.as_ref()) {
                    continue;
                }
                found = fields
                    .iter()
                    .find(|(f, _)| *f == attr.attr)
                    .map(|(_, ty)| ty.clone());
            }
            found
        };
        let mut unified: Option<crate::TypeInfo> = None;
        for r in &returns {
            let ty = match r {
                ExprType::Attribute(attr) => {
                    let ExprType::Name(recv) = attr.value.as_ref() else {
                        return None;
                    };
                    if recv.id != "self" {
                        return None;
                    }
                    fields
                        .iter()
                        .find(|(name, _)| *name == attr.attr)
                        .map(|(_, ty)| ty.clone())?
                }
                ExprType::Name(n) => assigned_self_field(&n.id)?,
                _ => return None,
            };
            match &unified {
                None => unified = Some(ty),
                Some(prev) if prev == &ty => {}
                _ => return None,
            }
        }
        unified
    }

    /// An unannotated function whose returns are all CALLS to a
    /// crate-module function by path (`return helper.parse(s)` — issue
    /// #222: the ~11 module-function-call sites in urllib3): the callee's
    /// return annotation, resolved ALIAS-AWARE in its DEFINING module, so
    /// the signature compiles against the body's `Ok(path::call(...)?)`.
    /// Sits at the END of the chain — it can only replace a signature
    /// that would otherwise be unit. All returns must name the same
    /// type; an annotation that resolves to no answer (PyValue/Any) or a
    /// `-> None` callee declines, never guesses.
    fn module_path_call_return_type(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<TokenStream> {
        if !guarantees_return(&self.body) {
            return None;
        }
        let mut returns = Vec::new();
        crate::ast::tree::specialize::collect_return_exprs(&self.body, &mut returns);
        if returns.is_empty() {
            return None;
        }
        let mut unified: Option<TokenStream> = None;
        for r in &returns {
            let ExprType::Call(call) = r else {
                return None;
            };
            let ExprType::Attribute(attr) = call.func.as_ref() else {
                return None;
            };
            let ExprType::Name(recv) = attr.value.as_ref() else {
                return None;
            };
            if recv.id == "self" {
                return None;
            }
            let path = crate::ast::tree::call::module_path_of_chain(
                &ExprType::Name(recv.clone()),
                symbols,
                options,
            )?;
            let (f, defining_symbols) =
                crate::module_function_def(options, &path, &attr.attr)?;
            let ann = f.returns.as_deref()?;
            if crate::is_none_expr(ann) {
                return None;
            }
            let ti = crate::resolve_alias_typeinfo(ann, &defining_symbols, options)?;
            if !renderable_return_typeinfo(&ti) {
                return None;
            }
            let ty = ti.to_rust_type();
            match &unified {
                None => unified = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                _ => return None,
            }
        }
        unified
    }

    /// An unannotated METHOD whose returns are all calls to methods of
    /// its own class on `self` (`return self._retries()` — issue #222:
    /// the ~8 self-method-call sites in urllib3): the callee method's
    /// return annotation, or — when the callee is itself unannotated —
    /// its own all-returns unification (ONE level deep; the recursion
    /// never re-enters the self rules, so mutual recursion between
    /// methods cannot spin). END of the chain; all returns must agree;
    /// a `-> None` callee declines (the unit signature is already
    /// correct for it).
    fn self_method_call_return_type(
        &self,
        self_class: Option<&str>,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<TokenStream> {
        if !guarantees_return(&self.body) {
            return None;
        }
        let mut returns = Vec::new();
        crate::ast::tree::specialize::collect_return_exprs(&self.body, &mut returns);
        if returns.is_empty() {
            return None;
        }
        let class_name = self_class?;
        let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(class_name) else {
            return None;
        };
        let mut unified: Option<TokenStream> = None;
        for r in &returns {
            let ExprType::Call(call) = r else {
                return None;
            };
            let ExprType::Attribute(attr) = call.func.as_ref() else {
                return None;
            };
            let ExprType::Name(recv) = attr.value.as_ref() else {
                return None;
            };
            if recv.id != "self" {
                return None;
            }
            let method = class.method_on_mro(&attr.attr, symbols)?;
            let ty = if let Some(ann) = method.returns.as_deref() {
                if crate::is_none_expr(ann) {
                    return None;
                }
                let ti = crate::resolve_alias_typeinfo(ann, symbols, options)?;
                if !renderable_return_typeinfo(&ti) {
                    return None;
                }
                ti.to_rust_type()
            } else {
                method.unified_return_type(symbols, options)?
            };
            match &unified {
                None => unified = Some(ty),
                Some(prev) if prev.to_string() == ty.to_string() => {}
                _ => return None,
            }
        }
        unified
    }

    /// The `Vec<stdpython::PyValue>` return type for an unannotated
    /// function whose returns are all LIST LITERALS of constants that the
    /// list lowering element-boxes (a mix of boxable kinds — issue #130's
    /// `[1, "a"]`), plus possibly empty lists (`vec![]` coerces). This
    /// mirrors the `ExprType::List` lowering's boxing decision so the
    /// SIGNATURE agrees with the rendered body — previously the boxed
    /// `vec![PyValue::from(1), ...]` returns rendered against a
    /// `Result<(), _>` signature (issue #133). Anything else — a
    /// concrete-element list, a non-constant element, a bare/None return,
    /// a possible fall-through — bails to None (unchanged behavior).
    fn boxed_list_return_type(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<TokenStream> {
        if self.returns.is_some() || !guarantees_return(&self.body) {
            return None;
        }
        let mut returns = Vec::new();
        collect_returns(&self.body, &mut returns);
        let mut any_boxed = false;
        for r in &returns {
            let Some(ExprType::List(l)) = r else {
                return None;
            };
            if l.is_empty() {
                continue;
            }
            let mut distinct: Vec<crate::TypeInfo> = Vec::new();
            let mut expected = crate::TypeInfo::PyObject;
            for li in l {
                if !matches!(li, ExprType::Constant(_)) {
                    return None;
                }
                let t = crate::infer_type(None, li, options, symbols);
                if matches!(t, crate::TypeInfo::PyObject) {
                    continue;
                }
                if !distinct.contains(&t) {
                    distinct.push(t.clone());
                }
                expected = crate::unify(expected, t);
            }
            if distinct.len() > 1
                && matches!(expected, crate::TypeInfo::PyObject)
                && distinct.iter().all(crate::is_boxable_value_type)
            {
                any_boxed = true;
            } else {
                return None;
            }
        }
        any_boxed.then(|| quote!(Vec<stdpython::PyValue>))
    }

    /// Whether any `return` in this body is `return cls(...)` — a call to
    /// the classmethod's own first parameter (the class reference).
    fn returns_cls_construction(&self) -> bool {
        let mut returns = Vec::new();
        collect_returns(&self.body, &mut returns);
        returns.iter().any(|r| {
            r.as_ref()
                .is_some_and(|e| matches!(e, ExprType::Call(c) if matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "cls")))
        })
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
    /// The LAST-RESORT return type (issue #222): the type every `return`
    /// in the body agrees on, per the general expression inferrer.
    ///
    /// Deliberately the final link in `resolved_return_type`'s chain, so it
    /// can only turn a signature that would have collapsed to `()` into a
    /// concrete one — it never overrides a type an earlier rule derived.
    /// That matters because the collapse is not a cosmetic wart: the body
    /// still emits `Ok(<value>)`, so `-> Result<(), PyException>` is code
    /// that cannot compile. Widening here strictly reduces the set of
    /// functions that lower to something rustc rejects.
    ///
    /// Two guards keep it honest. Every return must infer the SAME type —
    /// disagreement falls through to the existing literal-boxing rule
    /// rather than picking a winner. And the type must be one this
    /// whitelist can render faithfully: `PyObject` means the inferrer had
    /// no answer, and `StrRef` is a literal artifact that earlier rules
    /// already own, so both are refused rather than guessed at.
    fn unified_return_type(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<TokenStream> {
        self.inferred_return_typeinfo(symbols, options)
            .map(|t| t.to_rust_type())
    }

    /// The TypeInfo-level return unification (the None-mixing fold behind
    /// [`unified_return_type`]), exposed for the CALLER side: a caller
    /// that stores or passes an unannotated callee's result must learn
    /// the inferred `Option<T>` so the Option-aware machinery (narrowing,
    /// Option-access unwraps, the Option→concrete coercions) applies —
    /// the round-85 return-type directive ("a function that can return
    /// T | None returns Option<T>; the caller decides what to do with the
    /// None"). Round 85: a SINGLE concrete T plus the None path — exactly
    /// `T | None` — is `Option<T>`; a boxed or unrenderable T keeps the
    /// boxed PyValue (the box already contains None); disagreeing types
    /// with a None path box.
    pub(crate) fn inferred_return_typeinfo(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<crate::TypeInfo> {
        // Cycle guard: `infer_type` on a return may consult
        // `call_return_typeinfo` (a recursive callee — `def f(x): return
        // f(x-1)`) which consults this again — the same thread_local
        // pattern the alias resolver uses.
        crate::ast::tree::type_ctx::resolving_return(
            self as *const FunctionDef as usize,
            || None,
            || self.inferred_return_typeinfo_inner(symbols, options),
        )
    }

    fn inferred_return_typeinfo_inner(
        &self,
        symbols: &crate::SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<crate::TypeInfo> {
        let mut returns = Vec::new();
        crate::ast::tree::specialize::collect_return_exprs(&self.body, &mut returns);
        if returns.is_empty() {
            return None;
        }
        // A fall-through path (or a bare `return`) yields Python's None:
        // `return [ChecksumError]` + `return exceptions` + fall-through
        // (botocore's _extract_retryable_exception) is
        // `list[str] | list[Any] | None` — no static type — the boxed
        // PyValue, exactly the None-mixing unification the literal
        // partial-return rule already applies (issue #122 step 3).
        // Without a None path, disagreeing returns keep refusing.
        let mut has_none = !guarantees_return(&self.body)
            || self
                .body
                .iter()
                .any(|s| matches!(&s.statement, crate::StatementType::Return(None)));
        let mut unified: Option<crate::TypeInfo> = None;
        for r in &returns {
            // A `return None` literal is the None path (its infer_type
            // is PyObject — "no answer" — which would otherwise look like
            // a disagreeing type and box the fold).
            if crate::is_none_expr(r) {
                has_none = true;
                continue;
            }
            let t = match crate::infer_type(None, r, options, symbols) {
                // A string-LITERAL return infers `&'static str` (StrRef);
                // the codegen returns an owned String — normalize so the
                // `T | None` fold types `return "s"` + `return None` as
                // `Option<String>`, matching the collector's literal
                // handling (round 85).
                crate::TypeInfo::StrRef => crate::TypeInfo::String,
                t => t,
            };
            match &unified {
                None => unified = Some(t),
                Some(prev) if *prev == t => {}
                Some(_) if has_none => {
                    return Some(crate::TypeInfo::PyValue);
                }
                Some(_) => return None,
            }
        }
        match (unified, has_none) {
            // No None path: the type survives ONLY when renderable — an
            // unrenderable (PyObject "no answer") return still refuses
            // rather than guessing (the #224 discipline).
            (Some(t), false) if renderable_return_typeinfo(&t) => Some(t),
            // A None path makes the function heterogeneous by definition.
            // Round 85 (the return-type directive): a SINGLE concrete type
            // T plus None — exactly `T | None` — returns `Option<T>` (the
            // caller decides what to do with the None). A boxed or
            // unrenderable T keeps the boxed PyValue (the box already
            // contains None).
            (Some(t), true) => {
                if renderable_return_typeinfo(&t)
                    && !matches!(t, crate::TypeInfo::PyValue | crate::TypeInfo::PyObject)
                {
                    Some(crate::TypeInfo::Option(Box::new(t)))
                } else {
                    Some(crate::TypeInfo::PyValue)
                }
            }
            _ => None,
        }
    }

    /// The BODY-VISIBLE Rust type of an annotated parameter, or None when
    /// the parameter is absent, unannotated, or its annotation has no
    /// concrete Rust mapping (issue #222).
    ///
    /// Driven by the annotation rather than by `options.param_type_vars`
    /// deliberately: that map is populated on `to_rust`'s per-function
    /// options clone, so callers that ask for a signature from outside the
    /// lowering (the PyO3 wrapper generator) would see an empty map and
    /// compute a DIFFERENT return type than the function they wrap. The
    /// annotation gives every caller the same answer.
    ///
    /// A `str` parameter arrives as `impl Into<String>` and is converted by
    /// the body prologue, so what a `return` sees is the owned `String`.
    /// An unannotated parameter is left alone: its type may be an inferred
    /// generic, and that path is owned by `inferred_signature`.
    fn annotated_param_type(&self, name: &str) -> Option<crate::TypeInfo> {
        let param = self
            .args
            .posonlyargs
            .iter()
            .chain(self.args.args.iter())
            .chain(self.args.kwonlyargs.iter())
            .find(|p| p.arg == name)?;
        let ann = param.annotation.as_deref()?;
        if matches!(ann, ExprType::Name(n) if n.id == "str") {
            return Some(crate::TypeInfo::String);
        }
        crate::annotation_type_info(ann)
    }

    pub fn inferred_return_type(&self, options: &crate::PythonOptions) -> Option<TokenStream> {
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

        let mut inferred: Option<crate::TypeInfo> = None;
        for ret in &returns {
            let value = (*ret)?; // a bare `return` means the type is unit
            let ty = match value {
                ExprType::Name(name) => match locals.get(&name.id) {
                    Some(t) => {
                        let t = t.clone();
                        if matches!(t, crate::TypeInfo::StrRef)
                            && rebound.contains(&name.id)
                        {
                            crate::TypeInfo::String
                        } else {
                            t
                        }
                    }
                    // Issue #189: a class-instance module global reads as
                    // the INSTANCE (the Option is the static's
                    // representation), so `return HISTORY_RECORDER` types
                    // the function as the class.
                    None => match options.mutable_statics.get(&name.id) {
                        Some(crate::MutableGlobalKind::Class { class }) => {
                            crate::TypeInfo::Class(class.clone())
                        }
                        // Issue #222: a returned PARAMETER is not a local,
                        // so it used to fall straight through to `None` and
                        // the signature collapsed to unit while the body
                        // still emitted `Ok(x)` — code that cannot compile.
                        // An ANNOTATED parameter's type is known here.
                        _ => match self.annotated_param_type(&name.id) {
                            Some(t) => t,
                            None => return None,
                        },
                    },
                },
                other => crate::simple_expr_typeinfo(other)?,
            };
            match &inferred {
                None => inferred = Some(ty),
                Some(prev) if prev == &ty => {}
                _ => return None,
            }
        }
        inferred.map(|t| t.to_rust_type())
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
                    // The Ellipsis sentinel is NOT a docstring: a Protocol
                    // stub `def f(...) -> None: ...` must not get a bogus
                    // `#![doc = "\0RYTHON_ELLIPSIS"]` from its `...` body.
                    if c.0
                        .as_ref()
                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                    {
                        return None;
                    }
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
