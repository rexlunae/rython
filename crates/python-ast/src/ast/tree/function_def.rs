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
                options.module_defs.get(&path).is_some_and(|m| {
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
        thread_local! {
            static FN_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let depth = FN_DEPTH.with(|d| d.get());
        if depth > 50 && depth % 10 == 0 {
        }
        FN_DEPTH.with(|d| d.set(depth + 1));
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
        FN_DEPTH.with(|d| d.set(depth));
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
            let mut axis_arg_idents: Vec<proc_macro2::Ident> = Vec::new();
            for (i, p) in self.args.args.iter().enumerate() {
                if let Some(k) = spec.axes.iter().position(|a| a.index == i) {
                    let ident = crate::safe_ident(&p.arg);
                    let e = &enum_idents[k];
                    sig_params.extend(quote!(#ident: impl Into<#e>,));
                    axis_arg_idents.push(ident);
                    let v = format_ident!("v{}", k + 1);
                    forward_args.push(quote!(#v));
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
                let call = arm_body(quote!(#morph(#(#forward_args),*)));
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
        // the module statics; WRITES need mutable module state, which
        // rython does not model (module-level reassignment lowers to
        // __module_init__ locals invisible to functions) — the writes are
        // no-ops, surfaced through the -W channel (issue #115, a documented
        // divergence; the read side still resolves the module static).
        if let Some(name) = global_write_error(&self.body) {
            options.definition_warnings.borrow_mut().push(format!(
                "function `{}` writes to module-level name `{name}`: the write is \
                 dropped (issue #115 — rython has no mutable module state visible \
                 to functions)",
                self.name
            ));
        }
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
                            Some(ann)
                                if crate::is_optional_annotation(ann) =>
                            {
                                let inner = crate::python_annotation_to_rust_type(ann)
                                    .unwrap_or_else(|| quote!(String));
                                quote!(Option<#inner>)
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
                            Some(crate::SymbolTableNode::ImportFrom(i)) if i.module == "threading"
                        )
                    {
                        known.insert(param.arg.clone(), format!("threading.{}", t.name()));
                    }
                }
                // A dotted threading annotation (`lock: threading.Lock`):
                // same recording as above.
                if let Some(ExprType::Attribute(ann)) = param.annotation.as_deref()
                    && matches!(ann.value.as_ref(), ExprType::Name(m) if m.id == "threading")
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
                let Some(py) = rust_type_to_py_name(&ty) else {
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
            let mut info = crate::analyze_function_types(&effective_body, Some(&options), Some(&symbols));
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
                            // heterogeneous value.
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
                            // class's methods.
                            _ => {
                                let cname = n.id.clone();
                                if symbols.get(&cname).is_some() {
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
            .filter(|p| p.annotation.is_none())
            .filter(|p| {
                let pos = self
                    .args
                    .posonlyargs
                    .iter()
                    .chain(self.args.args.iter())
                    .position(|q| q.arg == p.arg);
                let from = self.args.posonlyargs.len() + self.args.args.len()
                    - self.args.defaults.len();
                match pos {
                    Some(i) if i >= from => self
                        .args
                        .defaults
                        .get(i - from)
                        .is_some_and(|d| crate::is_none_expr(d)),
                    _ => {
                        let kw = self
                            .args
                            .kwonlyargs
                            .iter()
                            .zip(self.args.kw_defaults.iter())
                            .find(|(q, _)| q.arg == p.arg);
                        kw.is_some_and(|(_, d)| {
                            d.as_deref().is_some_and(crate::is_none_expr)
                        })
                    }
                }
            })
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
        // nothing but None can ever be stored in them (issue #117).
        let mut final_param_types = inferred_signature.param_types.clone();
        for name in &none_defaulted {
            final_param_types.insert(name.clone(), quote!(Option<()>));
        }
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
                    .filter(|(_, ty)| {
                        ty.to_string() == quote!(&'static str).to_string()
                    })
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
            } else if scope.needs_mut.contains(name) {
                streams_prologue.extend(quote!(let mut #ident = #ident;));
            }
        }
        for name in &scope.assigned {
            // Python's `_` discard target (`(scheme, _, host, port, _) =
            // parse_url(...)` — idna/urllib3): a wildcard, never a hoisted
            // binding — `let mut _;` is not legal Rust.
            if name == "_" {
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
        let gen_elt = if crate::body_has_yields(&effective_body) {
            // Even when the element type cannot be resolved (a
            // `typing.Iterator[str]` annotation), the generator must still
            // lower — Vec<_> infers from the pushes.
            generator_element_type(self.returns.as_deref(), &effective_body, &options, &symbols)
                .or(Some(crate::TypeInfo::PyObject))
        } else {
            None
        };
        if let Some(elt) = &gen_elt {
            let t = elt.to_rust_type();
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
        // A function whose resolved return type is the boxed PyValue:
        // `return None` lowers to `PyValue::None_` and other returns wrap
        // in PyValue::from (the None-mixing unification — botocore's
        // docs.client._allowlist_generate_presigned_url). Set BEFORE the
        // body statements render (they clone options per statement).
        options.fn_return_is_pyvalue = matches!(
            self.resolved_return_type(&symbols, &options),
            Some(ref ty) if ty.to_string() == "stdpython :: PyValue"
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
        let return_type = if let Some(elt) = &gen_elt {
            let t = elt.to_rust_type();
            quote!(-> Result<Vec<#t>, PyException>)
        } else if inferred_signature.is_generic() && self.returns.is_none() {
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
            match self.resolved_return_type(&symbols, &options) {
                Some(ty) => quote!(-> Result<#ty, PyException>),
                None => quote!(-> Result<(), PyException>),
            }
        };

        // A body that can fall off the end implicitly returns None: give the
        // generated block an Ok(()) tail. Bodies that return (or raise) on
        // every path end with `return`/`return Err`, which need no tail.
        // A GENERATOR ends by returning its collected list.
        if gen_elt.is_some() {
            streams.extend(quote!(return __rython_gen;));
        } else if !guarantees_return(&self.body) {
            if options.fn_return_is_pyvalue {
                streams.extend(quote!(Ok(PyValue::None_)));
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
            let ret = match self.resolved_return_type(&symbols, &options) {
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

/// The names of NESTED function definitions inside a statement list
/// (recursing through control-flow bodies but not into nested defs/classes).
/// These are CLOSURES in Python; rython's closures do not capture the
/// enclosing scope (the closure-capture divergence), so the definitions
/// drop (statement.rs) and calls through the names drop too.
pub(crate) fn nested_function_names(body: &[crate::Statement]) -> Vec<String> {
    use crate::StatementType as ST;
    fn scan(stmts: &[crate::Statement], out: &mut Vec<String>) {
        for s in stmts {
            match &s.statement {
                ST::FunctionDef(f) | ST::AsyncFunctionDef(f) => out.push(f.name.clone()),
                ST::If(i) => {
                    scan(&i.body, out);
                    scan(&i.orelse, out);
                }
                ST::While(w) => {
                    scan(&w.body, out);
                    scan(&w.orelse, out);
                }
                ST::For(f) => {
                    scan(&f.body, out);
                    scan(&f.orelse, out);
                }
                ST::AsyncFor(f) => {
                    scan(&f.body, out);
                    scan(&f.orelse, out);
                }
                ST::With(w) => scan(&w.body, out),
                ST::AsyncWith(w) => scan(&w.body, out),
                ST::Try(t) => {
                    scan(&t.body, out);
                    scan(&t.orelse, out);
                    scan(&t.finalbody, out);
                    for h in &t.handlers {
                        scan(&h.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    scan(body, &mut out);
    out
}

/// Map an expression to an obviously-inferable Rust type, if any.
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

/// Issue #112: a `del name` whose name is referenced afterwards (or whose
/// deletion is conditional) cannot lower to a no-op — the value would
/// still be readable where Python raises NameError. Loud error; a
/// reassignment or import clears the deletion (Python rebinds).
pub(crate) fn check_deleted_names(body: &[Statement]) -> Result<(), String> {
    let mut deleted: std::collections::HashSet<String> = std::collections::HashSet::new();
    let check = |stmt: &Statement, deleted: &mut std::collections::HashSet<String>| -> Result<(), String> {
        let walk_expr = |expr: &ExprType, deleted: &std::collections::HashSet<String>| -> Result<(), String> {
            for name in deleted {
                if crate::expr_references(expr, name) {
                    return Err(format!(
                        "`del {name}` then a use of `{name}`: rython cannot unbind a \
                         binding (the value would still be readable where Python raises \
                         NameError). Remove the del, or reassign `{name}` first (issue \
                         #112)"
                    ));
                }
            }
            Ok(())
        };
        match &stmt.statement {
            StatementType::Delete(targets) => {
                for target in targets {
                    if let ExprType::Name(n) = target {
                        deleted.insert(n.id.clone());
                    }
                }
                Ok(())
            }
            StatementType::Assign(a) => {
                for target in &a.targets {
                    if let ExprType::Name(n) = target {
                        deleted.remove(&n.id);
                    }
                }
                walk_expr(&a.value, deleted)?;
                for target in &a.targets {
                    walk_expr(target, deleted)?;
                }
                Ok(())
            }
            StatementType::AugAssign(a) => {
                if let ExprType::Name(n) = &a.target {
                    deleted.remove(&n.id);
                }
                walk_expr(&a.target, deleted)?;
                walk_expr(&a.value, deleted)
            }
            StatementType::Import(i) => {
                for alias in &i.names {
                    deleted.remove(&alias.name);
                }
                Ok(())
            }
            StatementType::ImportFrom(i) => {
                for alias in &i.names {
                    deleted.remove(&alias.name);
                    if let Some(asname) = &alias.asname {
                        deleted.remove(asname);
                    }
                }
                Ok(())
            }
            StatementType::For(f) => {
                // The loop target rebinds each iteration.
                deleted.remove(&loop_target_name(&f.target));
                walk_expr(&f.iter, deleted)?;
                walk_expr(&f.target, deleted)?;
                check_deleted_names(&f.body)?;
                check_deleted_names(&f.orelse)?;
                Ok(())
            }
            StatementType::If(s) => {
                walk_expr(&s.test, deleted)?;
                check_deleted_names(&s.body)?;
                check_deleted_names(&s.orelse)?;
                Ok(())
            }
            StatementType::While(s) => {
                walk_expr(&s.test, deleted)?;
                check_deleted_names(&s.body)?;
                check_deleted_names(&s.orelse)?;
                Ok(())
            }
            StatementType::Try(t) => {
                check_deleted_names(&t.body)?;
                for h in &t.handlers {
                    check_deleted_names(&h.body)?;
                }
                check_deleted_names(&t.orelse)?;
                check_deleted_names(&t.finalbody)?;
                Ok(())
            }
            StatementType::With(w) => check_deleted_names(&w.body),
            StatementType::AsyncWith(w) => check_deleted_names(&w.body),
            StatementType::AsyncFor(f) => {
                deleted.remove(&loop_target_name(&f.target));
                walk_expr(&f.iter, deleted)?;
                walk_expr(&f.target, deleted)?;
                check_deleted_names(&f.body)?;
                check_deleted_names(&f.orelse)?;
                Ok(())
            }
            StatementType::Expr(e) => walk_expr(&e.value, deleted),
            StatementType::Return(Some(e)) => walk_expr(&e.value, deleted),
            StatementType::Return(None) => Ok(()),
            StatementType::Call(c) => {
                walk_expr(&c.func, deleted)?;
                for arg in &c.args {
                    walk_expr(arg, deleted)?;
                }
                for kw in &c.keywords {
                    walk_expr(&kw.value, deleted)?;
                }
                Ok(())
            }
            StatementType::Assert { test, msg } => {
                walk_expr(test, deleted)?;
                if let Some(m) = msg {
                    walk_expr(m, deleted)?;
                }
                Ok(())
            }
            StatementType::Raise(r) => {
                if let Some(exc) = &r.exc {
                    walk_expr(exc, deleted)?;
                }
                if let Some(cause) = &r.cause {
                    walk_expr(cause, deleted)?;
                }
                Ok(())
            }
            // Nested functions/classes have their own scope.
            StatementType::FunctionDef(_)
            | StatementType::AsyncFunctionDef(_)
            | StatementType::ClassDef(_)
            | StatementType::Global(_)
            | StatementType::Nonlocal(_)
            | StatementType::AnnotatedName { .. }
            | StatementType::Pass
            | StatementType::Break
            | StatementType::Continue
            | StatementType::Unimplemented(_) => Ok(()),
        }
    };
    for stmt in body {
        check(stmt, &mut deleted)?;
    }
    Ok(())
}

/// The name bound by a for-loop target (single-name targets only).
fn loop_target_name(target: &ExprType) -> String {
    match target {
        ExprType::Name(n) => n.id.clone(),
        _ => String::new(),
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

pub(crate) fn collect_local_types(
    body: &[Statement],
    out: &mut std::collections::HashMap<String, TokenStream>,
) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Assign(assign) => {
                if let [ExprType::Name(name)] = assign.targets.as_slice() {
                    // An annotated local (`flags: int = _character_flags(c)`)
                    // types from the annotation.
                    if let Some(ann) = assign.annotation.as_ref()
                        && let Some(t) = crate::annotation_type_info(ann)
                    {
                        out.insert(name.id.clone(), t.to_rust_type());
                    } else if let Some(ty) = simple_expr_type(&assign.value) {
                        out.insert(name.id.clone(), ty);
                    } else {
                        // An EMPTY container local (`allowed = {}` then
                        // `allowed[alg] = ...` — pip's Hashes): the boxed
                        // heterogeneous container (the element types are
                        // unknowable at the store).
                        match &assign.value {
                            ExprType::Dict(d) if d.keys.is_empty() => {
                                out.insert(name.id.clone(), quote!(PyDict<String, stdpython::PyValue>));
                            }
                            ExprType::List(l) if l.is_empty() => {
                                out.insert(name.id.clone(), quote!(Vec<stdpython::PyValue>));
                            }
                            _ => {}
                        }
                    }
                }
            }
            // A local IMPORT (`import keyring` inside __init__, then
            // `self.keyring = keyring` — pip's KeyRingPythonProvider): a
            // module object — a boxed value.
            StatementType::Import(im) => {
                for a in &im.names {
                    out.insert(a.name.clone(), quote!(stdpython::PyValue));
                }
            }
            // A bare annotated local (`key: str` / `value: str` — urllib3's
            // ssl_match_hostname): types the name for downstream use
            // (dnsnames.append(value) pins Vec<String>).
            StatementType::AnnotatedName { name, annotation } => {
                if let Some(t) = crate::annotation_type_info(annotation) {
                    out.insert(name.clone(), t.to_rust_type());
                }
            }
            StatementType::Try(s) => {
                collect_local_types(&s.body, out);
                for h in &s.handlers {
                    collect_local_types(&h.body, out);
                }
                collect_local_types(&s.orelse, out);
                collect_local_types(&s.finalbody, out);
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

/// Whether a function body contains a `yield`/`yield from` anywhere
/// (control-flow nested included): the body is a GENERATOR and lowers to
/// build-and-return-a-list (issue #122-family).
pub(crate) fn body_has_yields(body: &[Statement]) -> bool {
    body.iter().any(|s| match &s.statement {
        StatementType::Expr(e) => matches!(
            e.value,
            ExprType::Yield(_) | ExprType::YieldFrom(_)
        ),
        StatementType::If(s) => body_has_yields(&s.body) || body_has_yields(&s.orelse),
        StatementType::While(s) => body_has_yields(&s.body),
        StatementType::For(s) => body_has_yields(&s.body) || body_has_yields(&s.orelse),
        StatementType::With(s) => body_has_yields(&s.body),
        StatementType::Try(s) => {
            body_has_yields(&s.body)
                || s.handlers.iter().any(|h| body_has_yields(&h.body))
                || body_has_yields(&s.orelse)
                || body_has_yields(&s.finalbody)
        }
        _ => false,
    })
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
    if let Some(ann) = returns
        && let ExprType::Subscript(sub) = ann
        && matches!(
            sub.value.as_ref(),
            ExprType::Name(n) if matches!(n.id.as_str(), "Generator" | "Iterator")
        )
        && let crate::SubscriptKind::Index(elt) = &sub.kind
        && let ExprType::Tuple(t) = elt.as_ref()
        && let Some(first) = t.elts.first()
    {
        if let Some(t) = crate::resolve_alias_typeinfo(first, symbols, options) {
            return Some(t);
        }
        if let Some(t) = crate::annotation_type_info(first) {
            return Some(t);
        }
    }
    first_yield_type(body, options, symbols)
}

/// The inferred type of the first `yield <value>` in a body.
fn first_yield_type(
    body: &[Statement],
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<crate::TypeInfo> {
    for s in body {
        match &s.statement {
            StatementType::Expr(e) => {
                if let ExprType::Yield(y) = &e.value
                    && let Some(v) = y.value.as_ref()
                {
                    let t = crate::infer_type(v, options, symbols);
                    if !matches!(t, crate::TypeInfo::PyObject) {
                        return Some(t);
                    }
                }
            }
            StatementType::If(s) => {
                if let Some(t) = first_yield_type(&s.body, options, symbols) {
                    return Some(t);
                }
                if let Some(t) = first_yield_type(&s.orelse, options, symbols) {
                    return Some(t);
                }
            }
            StatementType::For(f) => {
                if let Some(t) = first_yield_type(&f.body, options, symbols) {
                    return Some(t);
                }
            }
            StatementType::While(w) => {
                if let Some(t) = first_yield_type(&w.body, options, symbols) {
                    return Some(t);
                }
            }
            _ => {}
        }
    }
    None
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
                                } else {
                                    let ident = crate::safe_ident(&n.id);
                                    Some(quote!(#ident))
                                }
                            }
                            _ => None,
                        }
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
        self.inferred_return_type().or(annotated)
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
