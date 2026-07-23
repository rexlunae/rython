//! Struct-based class lowering.
//!
//! A Python class lowers to a Rust struct plus an inherent impl block:
//!
//! - Instance attributes become struct fields, inferred from the `self.attr`
//!   assignments in `__init__` (from annotated parameters, literals, or
//!   construction of another known class).
//! - `__init__` lowers as an ordinary method taking `&mut self`, and a
//!   synthesized `new(...)` constructor default-initializes the struct and
//!   runs it; `ClassName(...)` call sites lower to `ClassName::new(...)?`.
//! - Methods lower as inherent methods; the receiver is `&self`, or
//!   `&mut self` when the method stores through `self` (directly or by
//!   calling another method of the class that does).
//!
//! Unsupported class constructs — inheritance, class-level statements,
//! attributes whose types can't be inferred — are conversion-time errors,
//! never silently dropped: lowering that diverges from Python must fail
//! loudly.

use proc_macro2::TokenStream;
use pyo3::FromPyObject;
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, FunctionDef, Name, PythonOptions, Statement,
    StatementType, SymbolTableNode, SymbolTableScopes,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct ClassDef {
    pub name: String,
    pub bases: Vec<Name>,
    pub keywords: Vec<String>,
    pub body: Vec<Statement>,
}

impl ClassDef {
    /// The class's `__init__` method, if it defines one.
    pub fn init_method(&self) -> Option<&FunctionDef> {
        self.methods().find(|m| m.name == "__init__")
    }

    /// The methods defined directly on the class, in source order.
    pub fn methods(&self) -> impl Iterator<Item = &FunctionDef> {
        self.body.iter().filter_map(|s| match &s.statement {
            StatementType::FunctionDef(f) => Some(f),
            _ => None,
        })
    }

    /// The class of the value stored in field `attr`, when the field holds
    /// an instance of another known class (composition): inferred from the
    /// `__init__` stores, either a direct construction or an
    /// annotated parameter whose annotation names a class.
    pub(crate) fn field_class(
        &self,
        attr: &str,
        symbols: &SymbolTableScopes,
    ) -> Option<String> {
        let init = self.init_method()?;
        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        let store = stores.iter().find(|s| s.attr == attr)?;
        let class_name = match store.value {
            ExprType::Call(call) => match call.func.as_ref() {
                ExprType::Name(n) => n.id.clone(),
                _ => return None,
            },
            ExprType::Name(n) => {
                let param = init
                    .args
                    .posonlyargs
                    .iter()
                    .chain(init.args.args.iter())
                    .chain(init.args.kwonlyargs.iter())
                    .find(|p| p.arg == n.id)?;
                match param.annotation.as_deref() {
                    Some(ExprType::Name(ann)) => ann.id.clone(),
                    _ => return None,
                }
            }
            _ => return None,
        };
        match symbols.get(&class_name) {
            Some(SymbolTableNode::ClassDef(_)) => Some(class_name),
            _ => None,
        }
    }

    /// Whether `method` mutates `self` — directly (attribute stores,
    /// mutating container methods on `self.attr`) or transitively through
    /// a call that bases at `self`: another method of this class
    /// (`self.helper()`) or a mutating method of a composed field's class
    /// (`self.inner.bump()`).
    pub(crate) fn method_needs_mut_self(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
    ) -> bool {
        let mut visited = std::collections::HashSet::new();
        self.method_mut_inner(method, symbols, &mut visited)
    }

    fn method_mut_inner(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        visited: &mut std::collections::HashSet<(String, String)>,
    ) -> bool {
        // A cycle in the call graph resolves optimistically: the mutation,
        // if real, is found on the acyclic part of some path.
        if !visited.insert((self.name.clone(), method.to_string())) {
            return false;
        }
        let Some(m) = self.methods().find(|m| m.name == method) else {
            return false;
        };
        let params = method_param_names(m);
        if crate::analyze_scope(&m.body, &params)
            .needs_mut
            .contains("self")
        {
            return true;
        }
        let ctx = CodeGenContext::Class(self.name.clone());
        let mut needs = false;
        crate::for_each_call(&m.body, &mut |call| {
            if needs {
                return;
            }
            if let ExprType::Attribute(attr) = call.func.as_ref() {
                if crate::chain_base_name(&attr.value) == Some("self") {
                    if let Some(class) = crate::receiver_class(&attr.value, &ctx, symbols) {
                        if class.method_mut_inner(&attr.attr, symbols, visited) {
                            needs = true;
                        }
                    }
                }
            }
        });
        needs
    }
}

/// All parameter names of a method, `self` included.
fn method_param_names(m: &FunctionDef) -> Vec<String> {
    m.args
        .posonlyargs
        .iter()
        .chain(m.args.args.iter())
        .chain(m.args.kwonlyargs.iter())
        .map(|p| p.arg.clone())
        .chain(m.args.vararg.iter().map(|p| p.arg.clone()))
        .chain(m.args.kwarg.iter().map(|p| p.arg.clone()))
        .collect()
}

/// A `self.attr = value` assignment found in `__init__`, used for field
/// inference.
struct FieldStore<'a> {
    attr: String,
    value: &'a ExprType,
}

/// Collect `self.attr = ...` stores anywhere in a body (recursing into
/// control flow), in first-store order.
fn collect_field_stores<'a>(body: &'a [Statement], out: &mut Vec<FieldStore<'a>>) {
    for stmt in body {
        match &stmt.statement {
            StatementType::Assign(assign) => {
                for target in &assign.targets {
                    if let ExprType::Attribute(attr) = target {
                        if matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                            out.push(FieldStore {
                                attr: attr.attr.clone(),
                                value: &assign.value,
                            });
                        }
                    }
                }
            }
            StatementType::If(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::With(s) => collect_field_stores(&s.body, out),
            StatementType::Try(s) => {
                collect_field_stores(&s.body, out);
                for h in &s.handlers {
                    collect_field_stores(&h.body, out);
                }
                collect_field_stores(&s.orelse, out);
                collect_field_stores(&s.finalbody, out);
            }
            _ => {}
        }
    }
}

impl CodeGen for ClassDef {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        symbols.insert(self.name.clone(), SymbolTableNode::ClassDef(self.clone()));
        symbols
    }

    fn to_rust(
        self,
        _ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let class_name = crate::safe_ident(&self.name);

        // Inheritance changes method resolution and construction in ways a
        // plain struct can't reproduce; failing loudly beats generating
        // something that behaves differently. `object` is every class's
        // implicit base, so naming it changes nothing.
        let real_bases: Vec<&str> = self
            .bases
            .iter()
            .map(|b| b.id.as_str())
            .filter(|b| *b != "object")
            .collect();
        if !real_bases.is_empty() {
            return Err(format!(
                "class `{}` uses inheritance (base{} {}), which is not supported yet: \
                 classes lower to plain Rust structs",
                self.name,
                if real_bases.len() == 1 { "" } else { "s" },
                real_bases.join(", "),
            )
            .into());
        }

        // Only methods (plus a docstring and `pass`) are supported in class
        // bodies. Class-level assignments (class attributes) would need a
        // shared-state story; erroring is the loud option.
        let body_start = if self.get_docstring().is_some() { 1 } else { 0 };
        for stmt in self.body.iter().skip(body_start) {
            match &stmt.statement {
                StatementType::FunctionDef(_) | StatementType::Pass => {}
                StatementType::AsyncFunctionDef(f) => {
                    return Err(format!(
                        "async method `{}.{}` is not supported yet",
                        self.name, f.name
                    )
                    .into());
                }
                other => {
                    let kind = match other {
                        StatementType::Assign(_) | StatementType::AugAssign(_) => {
                            "a class attribute assignment"
                        }
                        StatementType::ClassDef(_) => "a nested class",
                        StatementType::Import(_) | StatementType::ImportFrom(_) => "an import",
                        _ => "a statement",
                    };
                    return Err(format!(
                        "class `{}` contains {} at class level, which is not supported \
                         yet: only methods, a docstring, and `pass` lower",
                        self.name, kind,
                    )
                    .into());
                }
            }
        }

        // The synthesized constructor occupies `new` in the inherent impl;
        // a user method with that name would be a confusing duplicate-item
        // compile error instead of a conversion-time one.
        if self.methods().any(|m| m.name == "new") {
            return Err(format!(
                "class `{}` defines a method named `new`, which collides with the \
                 constructor synthesized for `{}(...)` call sites; rename the method",
                self.name, self.name
            )
            .into());
        }

        // ---- Field inference from __init__ ----
        let mut fields: Vec<(String, TokenStream)> = Vec::new();
        if let Some(init) = self.init_method() {
            // Types known for names in the __init__ body: annotated
            // parameters first, then simply-typed locals.
            let mut name_types: std::collections::HashMap<String, TokenStream> =
                std::collections::HashMap::new();
            crate::collect_local_types(&init.body, &mut name_types);
            for p in init
                .args
                .posonlyargs
                .iter()
                .chain(init.args.args.iter())
                .chain(init.args.kwonlyargs.iter())
            {
                if let Some(ann) = p.annotation.as_deref() {
                    // Mirror Parameter::to_rust: a `str` parameter becomes an
                    // owned String local via the prologue. A parameter
                    // annotated with a known class types the field as that
                    // class's struct (composition).
                    let ty = if matches!(ann, ExprType::Name(n) if n.id == "str") {
                        Some(quote!(String))
                    } else if let ExprType::Name(n) = ann {
                        match symbols.get(&n.id) {
                            Some(SymbolTableNode::ClassDef(_)) => {
                                let ident = crate::safe_ident(&n.id);
                                Some(quote!(#ident))
                            }
                            _ => crate::python_annotation_to_rust_type(ann),
                        }
                    } else {
                        crate::python_annotation_to_rust_type(ann)
                    };
                    if let Some(ty) = ty {
                        name_types.insert(p.arg.clone(), ty);
                    }
                }
            }

            let mut stores = Vec::new();
            collect_field_stores(&init.body, &mut stores);
            for store in &stores {
                let ty = infer_field_type(store.value, &name_types, &symbols);
                match ty {
                    Some(ty) => {
                        match fields.iter().find(|(name, _)| *name == store.attr) {
                            None => fields.push((store.attr.clone(), ty)),
                            Some((_, prev)) if prev.to_string() == ty.to_string() => {}
                            Some((_, prev)) => {
                                return Err(format!(
                                    "attribute `self.{}` of class `{}` is assigned \
                                     conflicting types ({} and {}); a struct field needs \
                                     one type",
                                    store.attr, self.name, prev, ty
                                )
                                .into());
                            }
                        }
                    }
                    None => {
                        return Err(format!(
                            "cannot infer a type for attribute `self.{}` of class `{}`: \
                             assign it from an annotated __init__ parameter, a literal, \
                             or a constructed class instance (None-valued attributes are \
                             not supported yet)",
                            store.attr, self.name
                        )
                        .into());
                    }
                }
            }
        }

        let field_defs: Vec<TokenStream> = fields
            .iter()
            .map(|(name, ty)| {
                let ident = crate::safe_ident(name);
                quote!(pub #ident: #ty)
            })
            .collect();

        // ---- Methods ----
        let method_ctx = CodeGenContext::Class(self.name.clone());
        let mut methods_stream = TokenStream::new();
        for stmt in self.body.iter().skip(body_start) {
            if let StatementType::FunctionDef(_) = &stmt.statement {
                methods_stream.extend(stmt.clone().to_rust(
                    method_ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
        }

        // ---- Synthesized constructor ----
        // Python constructs with `ClassName(args)`: default-initialize the
        // struct and run __init__ on it. Call sites lower to
        // `ClassName::new(args)?` (see Call::to_rust).
        let constructor = match self.init_method() {
            Some(init) => {
                if init.args.vararg.is_some() || init.args.kwarg.is_some() {
                    return Err(format!(
                        "`{}.__init__` takes *args/**kwargs, which is not supported yet",
                        self.name
                    )
                    .into());
                }
                let mut params = init.args.clone();
                strip_self(&mut params);
                let param_names: Vec<_> = params
                    .posonlyargs
                    .iter()
                    .chain(params.args.iter())
                    .chain(params.kwonlyargs.iter())
                    .map(|p| crate::safe_ident(&p.arg))
                    .collect();
                let rendered = params.to_rust(
                    method_ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                quote! {
                    pub fn new(#rendered) -> Result<Self, PyException> {
                        let mut __rython_self = Self::default();
                        __rython_self.__init__(#(#param_names),*)?;
                        Ok(__rython_self)
                    }
                }
            }
            None => quote! {
                pub fn new() -> Result<Self, PyException> {
                    Ok(Self::default())
                }
            },
        };

        let docs = match self.get_docstring() {
            Some(docstring) => {
                let doc_lines: Vec<_> = docstring
                    .lines()
                    .map(|line| {
                        let doc_line = line.to_string();
                        quote! { #[doc = #doc_line] }
                    })
                    .collect();
                quote!(#(#doc_lines)*)
            }
            None => quote!(),
        };

        Ok(quote! {
            #docs
            #[derive(Clone, Default)]
            pub struct #class_name {
                #(#field_defs),*
            }
            impl #class_name {
                #constructor
                #methods_stream
            }
        })
    }
}

/// Remove the leading `self` parameter from a method's parameter list.
pub(crate) fn strip_self(args: &mut crate::ParameterList) {
    if args
        .posonlyargs
        .first()
        .is_some_and(|p| p.arg == "self")
    {
        args.posonlyargs.remove(0);
    } else if args.args.first().is_some_and(|p| p.arg == "self") {
        args.args.remove(0);
    }
}

/// Infer the struct field type for a value stored into `self.attr`.
fn infer_field_type(
    value: &ExprType,
    name_types: &std::collections::HashMap<String, TokenStream>,
    symbols: &SymbolTableScopes,
) -> Option<TokenStream> {
    match value {
        ExprType::Name(n) => name_types.get(&n.id).cloned(),
        // A constructed instance of a known class types the field as that
        // class's struct.
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::ClassDef(_)) => {
                    let ident = crate::safe_ident(&n.id);
                    Some(quote!(#ident))
                }
                _ => None,
            },
            _ => None,
        },
        other => match crate::simple_expr_type(other) {
            // String literals are owned in fields; the store side converts
            // (see Assign).
            Some(ty) if ty.to_string() == "& 'static str" => Some(quote!(String)),
            other => other,
        },
    }
}

impl ClassDef {
    fn get_docstring(&self) -> Option<String> {
        if self.body.is_empty() {
            return None;
        }

        let expr = self.body[0].clone();
        match expr.statement {
            StatementType::Expr(e) => match e.value {
                ExprType::Constant(c) => {
                    let raw_string = c.to_string();
                    Some(self.format_docstring(&raw_string))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn format_docstring(&self, raw: &str) -> String {
        let content = raw.trim_matches('"');
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return String::new();
        }

        let mut formatted = vec![lines[0].trim().to_string()];

        if lines.len() > 1 {
            if !lines[0].trim().is_empty() && !lines[1].trim().is_empty() {
                formatted.push(String::new());
            }
            for line in lines.iter().skip(1) {
                let cleaned = line.trim();
                if !cleaned.is_empty() {
                    formatted.push(cleaned.to_string());
                }
            }
        }

        formatted.join("\n")
    }
}
