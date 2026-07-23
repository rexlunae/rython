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

        // The Python convention is that functions that begin with a single underscore,
        // it's private. Otherwise, it's public. We formalize that by default.
        let visibility = if self.name.starts_with("_") && !self.name.starts_with("__") {
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
        // calling another method of the class that does.
        let is_method = matches!(&ctx, CodeGenContext::Class(_))
            && self
                .args
                .posonlyargs
                .first()
                .or(self.args.args.first())
                .is_some_and(|p| p.arg == "self");
        let mut render_args = self.args.clone();
        let method_mutates_self = is_method
            && match &ctx {
                CodeGenContext::Class(class_name) => match symbols.get(class_name) {
                    Some(crate::SymbolTableNode::ClassDef(c)) => {
                        c.method_needs_mut_self(&self.name, &symbols)
                    }
                    _ => false,
                },
                _ => false,
            };
        if is_method {
            crate::strip_self(&mut render_args);
        }

        let parameters = render_args
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

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
            &self.body,
            &param_names,
            &crate::class_call_resolver(&ctx, &symbols),
        );
        if is_method {
            param_names.pop();
        }
        let receiver = if is_method {
            if method_mutates_self || scope.needs_mut.contains("self") {
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
        }
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
                streams_prologue.extend(quote!(let mut #ident;));
            } else {
                streams_prologue.extend(quote!(let #ident;));
            }
        }
        streams.extend(streams_prologue);

        // A leading docstring is emitted as doc comments below; skip it here
        // so it isn't also emitted as a statement.
        let body_start = if self.get_docstring().is_some() { 1 } else { 0 };
        for s in self.body.iter().skip(body_start) {
            streams.extend(
                s.clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?,
            );
            streams.extend(quote!(;));
        }

        // Every generated function returns Result<T, PyException> so raised
        // exceptions propagate across function boundaries the way Python's
        // do: call sites append `?`, and an uncaught exception surfaces at
        // the entry point. T is the resolved Python return type (unit when
        // there is none).
        let return_type = match self.resolved_return_type() {
            Some(ty) => quote!(-> Result<#ty, PyException>),
            None => quote!(-> Result<(), PyException>),
        };

        // A body that can fall off the end implicitly returns None: give the
        // generated block an Ok(()) tail. Bodies that return (or raise) on
        // every path end with `return`/`return Err`, which need no tail.
        if !guarantees_return(&self.body) {
            streams.extend(quote!(Ok(())));
        }

        // Lossy conversions are silent semantic changes callers may not want
        // — surface them as a compiler warning at every call site outside the
        // generated crate via a single #[deprecated] note (the standard
        // mechanism for user-defined warnings). An item can carry only one
        // #[deprecated] attribute, so all notes are folded into it.
        let lossy_warning = if options.lossy_warnings {
            let notes = self.lossy_conversion_notes();
            if notes.is_empty() {
                quote!()
            } else {
                let note = notes.join("; ");
                quote!(#[deprecated(note = #note)])
            }
        } else {
            quote!()
        };

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
                #visibility #is_async fn #fn_name(#receiver #parameters) #return_type {
                    #streams
                }
            }
        } else {
            quote! {
                #lossy_warning
                #visibility #is_async fn #fn_name(#receiver #parameters) #return_type {
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
        _ => None,
    }
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
fn annotation_display(ann: &ExprType) -> String {
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

        let mut inferred: Option<TokenStream> = None;
        for ret in &returns {
            let value = (*ret)?; // a bare `return` means the type is unit
            let ty = match value {
                ExprType::Name(name) => locals.get(&name.id)?.clone(),
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
