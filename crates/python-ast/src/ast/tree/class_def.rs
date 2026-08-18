//! Struct-and-trait class lowering.
//!
//! A Python class lowers to a Rust struct plus impl blocks:
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
//! Single inheritance (bases defined in the same module) lowers to a
//! trait-per-class scheme:
//!
//! - A class that has a base, or is used as a base, emits `trait {Name}Trait`
//!   with accessors for its fields (`fn f(&self) -> T`, `fn f_mut(&mut self)
//!   -> &mut T`), a `base()`/`base_mut()` accessor pair for its embedded base
//!   struct, and its own methods as default bodies. Each derived class
//!   implements every ancestor trait (accessors walk the embedded
//!   `__rython_base` chain; overridden methods replace the default).
//!   `self.helper()` calls on an instance resolve through the traits, so a
//!   call into an inherited method dispatches to the most-derived impl —
//!   Python's method resolution — while `super().helper()` pins the call to
//!   the direct base's implementation.
//! - A derived struct embeds its direct base as `pub __rython_base: Base`,
//!   so every ancestor's fields stay reachable (`self.__rython_base.f`, or
//!   `self.base().f` inside a generic trait default). `super().__init__(...)`
//!   lowers to `self.__rython_base.__init__(...)`, and a class without its
//!   own `__init__` synthesizes a forwarder so construction still runs the
//!   first `__init__` on the MRO.
//!
//! Unsupported class constructs — multiple inheritance, unknown/imported
//! bases, class-level statements, attributes whose types can't be inferred —
//! are conversion-time errors, never silently dropped: lowering that
//! diverges from Python must fail loudly.

use proc_macro2::TokenStream;
use pyo3::FromPyObject;
use quote::{format_ident, quote};

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

    /// The class's real base (the first base that is not `object`), resolved
    /// through the symbol table to its `ClassDef`. None for a base-less
    /// class or a base that is not a known class in this module.
    pub(crate) fn base_class(&self, symbols: &SymbolTableScopes) -> Option<ClassDef> {
        let base = self
            .bases
            .iter()
            .find(|b| b.id != "object")?;
        match symbols.get(&base.id) {
            Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
            _ => None,
        }
    }

    /// The class itself followed by every ancestor, nearest base first.
    /// Returns just `[self]` when the class has no base.
    pub(crate) fn base_chain(&self, symbols: &SymbolTableScopes) -> Vec<ClassDef> {
        let mut chain = vec![self.clone()];
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(self.name.clone());
        while let Some(base) = chain.last().and_then(|c| c.base_class(symbols)) {
            // A cyclic base (`class A(A)` after the name was rebound, so
            // the class resolves to itself) must terminate: Python looks
            // the base up in the OUTER scope and errors when it finds the
            // class being defined, but the symbol table can only see the
            // rebound name. Stop before the chain grows forever; emit_class
            // reports the cycle as a conversion error.
            if !seen.insert(base.name.clone()) {
                break;
            }
            chain.push(base);
        }
        chain
    }

    /// The class name that closes an inheritance cycle (`class A(A)`), or
    /// None for a valid chain. Walks bases like base_chain but reports the
    /// repeat instead of silently stopping.
    pub(crate) fn base_cycle(&self, symbols: &SymbolTableScopes) -> Option<String> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut cur = Some(self.clone());
        while let Some(c) = cur {
            if !seen.insert(c.name.clone()) {
                return Some(c.name.clone());
            }
            cur = c.base_class(symbols);
        }
        None
    }

    /// The first method named `name` found walking the MRO (the class
    /// itself, then its base, then the base's base, ...).
    pub(crate) fn method_on_mro(&self, name: &str, symbols: &SymbolTableScopes) -> Option<FunctionDef> {
        self.base_chain(symbols)
            .into_iter()
            .find_map(|c| c.methods().find(|m| m.name == name).cloned())
    }

    /// Whether `attr` is a field assigned somewhere in this class's own
    /// `__init__`.
    fn owns_field(&self, attr: &str) -> bool {
        let Some(init) = self.init_method() else {
            return false;
        };
        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        stores.iter().any(|s| s.attr == attr)
    }

    /// Which class in the MRO chain owns the field `attr`: 0 for this class,
    /// 1 for its direct base, etc. None when no class in the chain assigns
    /// the field. `self.attr` where attr is owned by an ancestor resolves
    /// through the ancestor's embedded struct.
    ///
    /// The owner is the class whose STRUCT physically holds the field — the
    /// DEEPEST ancestor that assigns it. A derived class's own stores of a
    /// base-owned field are filtered out of its struct (see `to_rust`), so
    /// the field lives at the topmost assigner in the chain; `rposition`
    /// finds that one. `.position` (nearest assigner) would report a
    /// depth-0 owner for a field the struct no longer has, making
    /// `self.n = n` in a subclass's own `__init__` write to a nonexistent
    /// field.
    pub(crate) fn field_owner_depth(&self, attr: &str, symbols: &SymbolTableScopes) -> Option<usize> {
        self.base_chain(symbols)
            .iter()
            .rposition(|c| c.owns_field(attr))
    }

    /// The class of the value stored in field `attr`, when the field holds
    /// an instance of another known class (composition): inferred from the
    /// `__init__` stores, either a direct construction or an
    /// annotated parameter whose annotation names a class. Walks the MRO so
    /// a base class's composed field resolves from a derived method.
    pub(crate) fn field_class(
        &self,
        attr: &str,
        symbols: &SymbolTableScopes,
    ) -> Option<String> {
        let chain = self.base_chain(symbols);
        let owner = chain.iter().find(|c| c.owns_field(attr))?;
        let init = owner.init_method()?;
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
        options: &PythonOptions,
    ) -> bool {
        // Trait signatures widen to `&mut self` when ANY definition in the
        // hierarchy mutates (overrides re-emit into the root's trait), so
        // call sites must borrow the receiver mutably to match. The
        // module-level precompute is authoritative when present.
        let root = self
            .base_chain(symbols)
            .into_iter()
            .find(|c| c.methods().any(|mm| mm.name == method));
        if let Some(root) = root
            && options
                .trait_mut_self
                .get(&root.name)
                .is_some_and(|s| s.contains(method))
        {
            return true;
        }
        // Fallback (no module precompute): the root's own body mutating, or
        // a chain class's override mutating.
        let mut visited = std::collections::HashSet::new();
        for c in self.base_chain(symbols) {
            if c.methods().any(|mm| mm.name == method)
                && c.method_mut_inner(method, symbols, &mut visited, options)
            {
                return true;
            }
        }
        false
    }

    /// Whether this class's OWN definition of `method` (not any override)
    /// mutates self — the module-level trait-signature widening precompute
    /// uses this on every defining class.
    pub(crate) fn own_method_mutates(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> bool {
        let mut visited = std::collections::HashSet::new();
        self.method_mut_inner(method, symbols, &mut visited, options)
    }

    fn method_mut_inner(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        visited: &mut std::collections::HashSet<(String, String)>,
        options: &PythonOptions,
    ) -> bool {
        // A cycle in the call graph resolves optimistically: the mutation,
        // if real, is found on the acyclic part of some path.
        if !visited.insert((self.name.clone(), method.to_string())) {
            return false;
        }
        let Some(m) = self.method_on_mro(method, symbols) else {
            return false;
        };
        let params = method_param_names(&m);
        let ctx = CodeGenContext::Class(self.name.clone());
        // The same resolver-backed analysis codegen uses, threading the
        // visited set through recursive method resolution (RefCell because
        // the resolver is a shared Fn).
        let visited = std::cell::RefCell::new(visited);
        let resolve = |call: &crate::Call| -> Option<bool> {
            let ExprType::Attribute(attr) = call.func.as_ref() else {
                return None;
            };
            let (class, class_symbols) =
                crate::receiver_class(&attr.value, &ctx, symbols, options)?;
            if class.method_on_mro(&attr.attr, &class_symbols).is_none() {
                return None;
            }
            Some(class.method_mut_inner(
                &attr.attr,
                &class_symbols,
                &mut **visited.borrow_mut(),
                options,
            ))
        };
        crate::analyze_scope_with(&m.body, &params, &resolve)
            .needs_mut
            .contains("self")
    }

    /// Infer the struct field list from the `self.attr = ...` stores in
    /// `__init__`. Stores are typed from annotated `__init__` parameters,
    /// simply-typed locals, and constructions of known classes; a store that
    /// cannot be typed is a loud error (a silently dropped attribute would
    /// diverge from Python). Includes stores that belong to a BASE class's
    /// fields; callers subtract those when they want the class's OWN fields.
    pub(crate) fn infer_fields(
        &self,
        symbols: &SymbolTableScopes,
    ) -> Result<Vec<(String, TokenStream)>, Box<dyn std::error::Error>> {
        let mut fields: Vec<(String, TokenStream)> = Vec::new();
        let Some(init) = self.init_method() else {
            return Ok(fields);
        };
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
            let ty = infer_field_type(store.value, &name_types, symbols);
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
        Ok(fields)
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

        // ---- Base resolution ----
        // Single inheritance only. `object` is every class's implicit base,
        // so naming it changes nothing. The base must be a class defined in
        // this module: the embedded-struct + trait scheme cannot represent an
        // imported or builtin base faithfully.
        let real_bases: Vec<&str> = self
            .bases
            .iter()
            .map(|b| b.id.as_str())
            .filter(|b| *b != "object")
            .collect();
        if real_bases.len() > 1 {
            return Err(format!(
                "class `{}` uses multiple inheritance (bases {}), which is not \
                 supported: only single inheritance lowers",
                self.name,
                real_bases.join(", "),
            )
            .into());
        }
        let base: Option<ClassDef> = match real_bases.first() {
            None => None,
            Some(base_name) => match symbols.get(base_name) {
                Some(SymbolTableNode::ClassDef(c)) => {
                    // A base that resolves to an already-visited class in
                    // this class's own chain is a cycle (`class A(A)` —
                    // Python looks the base up in the outer scope, which
                    // the symbol table can only see as the rebound name).
                    // The embedded-struct scheme cannot represent it; fail
                    // loudly instead of emitting an infinitely-sized
                    // struct or looping forever in base_chain.
                    if let Some(cycle) = self.base_cycle(&symbols) {
                        return Err(format!(
                            "class `{}` cannot inherit from `{}`: cyclic inheritance",
                            self.name, cycle,
                        )
                        .into());
                    }
                    Some(c.clone())
                }
                _ => {
                    return Err(format!(
                        "class `{}` inherits from `{}`, which is not a class defined \
                         in this module; imported bases and built-in bases are not \
                         supported yet",
                        self.name, base_name,
                    )
                    .into());
                }
            },
        };
        let in_hierarchy = base.is_some() || options.hierarchy_classes.contains(&self.name);

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

        // ---- Field inference ----
        // A derived class's OWN fields are its __init__ stores minus the
        // stores that belong to a base class's fields (those write into the
        // embedded base struct).
        let own_stores = self.infer_fields(&symbols)?;
        let base_owned: std::collections::HashSet<String> = match &base {
            Some(b) => b
                .base_chain(&symbols)
                .iter()
                .filter_map(|c| c.infer_fields(&symbols).ok())
                .flat_map(|f| f.into_iter().map(|(n, _)| n))
                .collect(),
            None => std::collections::HashSet::new(),
        };
        let fields: Vec<(String, TokenStream)> = own_stores
            .into_iter()
            .filter(|(name, _)| !base_owned.contains(name))
            .collect();

        // The embedded base struct occupies `__rython_base`; a user field of
        // the same name would collide with it.
        if base.is_some() && fields.iter().any(|(name, _)| name == "__rython_base") {
            return Err(format!(
                "class `{}` assigns an attribute named `__rython_base`, which \
                 collides with the embedded base struct generated for inheritance; \
                 rename the attribute",
                self.name
            )
            .into());
        }

        let mut field_defs: Vec<TokenStream> = fields
            .iter()
            .map(|(name, ty)| {
                let ident = crate::safe_ident(name);
                quote!(pub #ident: #ty)
            })
            .collect();
        if let Some(b) = &base {
            let b_ident = crate::safe_ident(&b.name);
            field_defs.push(quote!(pub __rython_base: #b_ident));
        }

        // ---- Trait-machinery name guards (hierarchy classes only) ----
        // The generated accessors (`base`, `base_mut`, `<field>_mut`) and the
        // trait name (`{Class}Trait`) must not collide with user code.
        if in_hierarchy {
            for m in self.methods() {
                if matches!(m.name.as_str(), "base" | "base_mut") {
                    return Err(format!(
                        "class `{}` defines a method named `{}`, which collides with \
                         the base accessor generated for inheritance; rename the method",
                        self.name, m.name
                    )
                    .into());
                }
                if let Some(field) = m.name.strip_suffix("_mut") {
                    if fields.iter().any(|(fname, _)| fname == field) {
                        return Err(format!(
                            "class `{}` defines a method named `{}`, which collides with \
                             the mutable accessor generated for field `{}`; rename the \
                             method or the field",
                            self.name, m.name, field
                        )
                        .into());
                    }
                }
            }
            let trait_name_str = format!("{}Trait", self.name);
            if symbols.get(&trait_name_str).is_some() {
                return Err(format!(
                    "class `{}` is part of an inheritance hierarchy, so its trait is \
                     named `{}`, which is already taken by another class in this module; \
                     rename one of them",
                    self.name, trait_name_str
                )
                .into());
            }
        }

        // ---- Methods ----
        let method_ctx = CodeGenContext::Class(self.name.clone());
        let methods: Vec<FunctionDef> = self
            .body
            .iter()
            .skip(body_start)
            .filter_map(|s| match &s.statement {
                StatementType::FunctionDef(f) => Some(f.clone()),
                _ => None,
            })
            .collect();
        let mut methods_stream = TokenStream::new();
        for m in &methods {
            methods_stream.extend(
(*m).clone()
                    .to_rust(method_ctx.clone(), options.clone(), symbols.clone())?,
            );
        }

        // ---- Synthesized constructor ----
        // Python constructs with `ClassName(args)`: default-initialize the
        // struct and run the first __init__ on the MRO. Call sites lower to
        // `ClassName::new(args)?` (see Call::to_rust).
        let mro_init = self.method_on_mro("__init__", &symbols);
        if let Some(init) = mro_init.as_ref()
            && (init.args.vararg.is_some() || init.args.kwarg.is_some())
        {
            return Err(format!(
                "`{}.__init__` takes *args/**kwargs, which is not supported yet",
                self.name
            )
            .into());
        }
        let constructor = match mro_init.as_ref() {
            Some(init) => {
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
        // A derived class without its own __init__ forwards to the first
        // __init__ on the MRO, so `new` runs it on the embedded base part.
        // A base chain with no __init__ anywhere needs no forwarder (`new`
        // is just the default struct).
        let init_forwarder = if self.init_method().is_none() {
            match (&base, mro_init.as_ref()) {
                (Some(_), Some(init)) => {
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
                        pub(crate) fn __init__(&mut self, #rendered) -> Result<(), PyException> {
                            self.__rython_base.__init__(#(#param_names),*)?;
                            Ok(())
                        }
                    }
                }
                _ => quote!(),
            }
        } else {
            quote!()
        };

        // ---- Trait machinery (inheritance hierarchies only) ----
        let trait_stream = if in_hierarchy {
            self.emit_trait(&base, &fields, &methods, &options, &symbols)?
        } else {
            quote!()
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
            #trait_stream
            impl #class_name {
                #constructor
                #init_forwarder
                #methods_stream
            }
        })
    }
}

impl ClassDef {
    /// Emit the trait-based inheritance machinery for a class that has a
    /// base or is used as a base:
    ///
    /// - `trait {Name}Trait` — `base()`/`base_mut()` accessors (when the
    ///   class has a base), field accessors `fn f(&self) -> T` /
    ///   `fn f_mut(&mut self) -> &mut T`, and the class's own (non-override)
    ///   methods as default bodies written against the accessors.
    /// - `impl {Name}Trait for {Name}` — the accessor bodies.
    /// - For every ancestor, `impl {Ancestor}Trait for {Name}` — the
    ///   ancestor's field accessors (walking the embedded `__rython_base`
    ///   chain) and the class's overrides of the ancestor's methods, written
    ///   against the concrete struct.
    fn emit_trait(
        &self,
        base: &Option<ClassDef>,
        fields: &[(String, TokenStream)],
        methods: &[FunctionDef],
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let class_name = crate::safe_ident(&self.name);
        let trait_name = format_ident!("{}Trait", self.name);

        // ---- Own trait ----
        let mut own_accessor_decls = TokenStream::new();
        if let Some(b) = base {
            let b_ident = crate::safe_ident(&b.name);
            own_accessor_decls.extend(quote! {
                fn base(&self) -> & #b_ident;
                fn base_mut(&mut self) -> &mut #b_ident;
            });
        }
        for (fname, fty) in fields {
            let f = crate::safe_ident(fname);
            let f_mut = format_ident!("{}_mut", fname);
            own_accessor_decls.extend(quote! {
                fn #f(&self) -> #fty;
                fn #f_mut(&mut self) -> &mut #fty;
            });
        }
        // A method the class defines that no ancestor defines is a NEW
        // method: it lives as a default in this class's own trait.
        let own_methods: Vec<&FunctionDef> = methods
            .iter()
            .filter(|m| {
                m.name != "__init__"
                    && base
                        .as_ref()
                        .map_or(true, |b| b.method_on_mro(&m.name, symbols).is_none())
            })
            .collect();
        let mut own_method_defaults = TokenStream::new();
        for m in &own_methods {
            let trait_ctx = CodeGenContext::Trait {
                class: self.name.clone(),
                generic: true,
                super_target: None,
                force_mut_self: options
                    .trait_mut_self
                    .get(&self.name)
                    .is_some_and(|s| s.contains(&m.name)),
            };
            own_method_defaults.extend(
(*m).clone()
                    .to_rust(trait_ctx, options.clone(), symbols.clone())?,
            );
        }

        let mut own_impl_body = TokenStream::new();
        if let Some(b) = base {
            let b_ident = crate::safe_ident(&b.name);
            own_impl_body.extend(quote! {
                fn base(&self) -> & #b_ident {
                    &self.__rython_base
                }
                fn base_mut(&mut self) -> &mut #b_ident {
                    &mut self.__rython_base
                }
            });
        }
        for (fname, fty) in fields {
            let f = crate::safe_ident(fname);
            let f_mut = format_ident!("{}_mut", fname);
            own_impl_body.extend(quote! {
                fn #f(&self) -> #fty {
                    self.#f.clone()
                }
                fn #f_mut(&mut self) -> &mut #fty {
                    &mut self.#f
                }
            });
        }

        // The per-class trait is PUBLIC (inherited methods are called
        // cross-module: the struct is `pub` and re-exported, so the traits
        // carrying its methods must be nameable wherever the struct is) and
        // declares the direct base's trait as a supertrait. Trait default
        // bodies are generic over `Self: {Name}Trait` only, so a new method
        // that calls an inherited method (`def bar(self): self.foo()` where
        // foo lives on the base) resolves `foo` through the supertrait
        // bound; ancestor methods are not on the concrete `Self` otherwise.
        let supertrait = base.as_ref().map(|b| {
            let b_trait = format_ident!("{}Trait", b.name);
            quote!(: #b_trait)
        });
        let own_trait = quote! {
            pub trait #trait_name #supertrait {
                #own_accessor_decls
                #own_method_defaults
            }
            impl #trait_name for #class_name {
                #own_impl_body
            }
        };

        // ---- Ancestor impls ----
        // For each ancestor (root first), implement ITS trait: field
        // accessors reaching through `self.__rython_base` (repeated per
        // level), plus this class's overrides of the methods that ancestor
        // defined, written against the concrete struct.
        let mut ancestor_impls = TokenStream::new();
        let chain = self.base_chain(symbols);
        for (depth, ancestor) in chain.iter().enumerate().skip(1).rev() {
            let ancestor_trait = format_ident!("{}Trait", ancestor.name);
            let mut chain_tokens = TokenStream::new();
            for _ in 0..depth {
                chain_tokens.extend(quote!(.__rython_base));
            }
            let a_fields = ancestor.infer_fields(symbols)?;
            let mut accessor_impls = TokenStream::new();
            // The ancestor's own base accessors, if it has a base: from the
            // derived struct, its base struct is one level deeper.
            if let Some(a_base) = ancestor.base_class(symbols) {
                let ab_ident = crate::safe_ident(&a_base.name);
                let mut base_self = quote!(self);
                base_self.extend(chain_tokens.clone());
                accessor_impls.extend(quote! {
                    fn base(&self) -> & #ab_ident {
                        &#base_self.__rython_base
                    }
                    fn base_mut(&mut self) -> &mut #ab_ident {
                        &mut #base_self.__rython_base
                    }
                });
            }
            for (fname, fty) in &a_fields {
                let f = crate::safe_ident(fname);
                let f_mut = format_ident!("{}_mut", fname);
                // `self.__rython_base[.__rython_base]*` reaches the
                // ancestor's struct from the derived struct.
                let mut accessor_self = quote!(self);
                accessor_self.extend(chain_tokens.clone());
                accessor_impls.extend(quote! {
                    fn #f(&self) -> #fty {
                        #accessor_self.#f.clone()
                    }
                    fn #f_mut(&mut self) -> &mut #fty {
                        &mut #accessor_self.#f
                    }
                });
            }
            // Overrides: for each TRAIT MEMBER of the ancestor (its own
            // methods that are not themselves overrides of ITS base — those
            // live in a higher trait), the NEAREST definition in the derived
            // chain (self … down to the ancestor, exclusive) wins. A
            // definition by the ancestor itself is the trait default, so
            // only strictly-lower definers need re-emission against this
            // class's struct — and `super()` inside such a re-emitted
            // override must target the DEFINER's base, not this class's base.
            let a_base = ancestor.base_class(symbols);
            let ancestor_members: Vec<&FunctionDef> = ancestor
                .methods()
                .filter(|am| {
                    am.name != "__init__"
                        && a_base.as_ref().map_or(true, |b| {
                            b.method_on_mro(&am.name, symbols).is_none()
                        })
                })
                .collect();
            let mut override_stream = TokenStream::new();
            for am in &ancestor_members {
                let mut definer: Option<FunctionDef> = None;
                let mut definer_name: Option<String> = None;
                for c in chain.iter() {
                    if c.name == ancestor.name {
                        break;
                    }
                    if let Some(m) = c.methods().find(|m| m.name == am.name) {
                        definer = Some(m.clone());
                        definer_name = Some(c.name.clone());
                        break;
                    }
                }
                if let (Some(m), Some(dname)) = (definer, definer_name) {
                    let trait_ctx = CodeGenContext::Trait {
                        class: self.name.clone(),
                        generic: false,
                        super_target: Some(dname),
                        force_mut_self: options
                            .trait_mut_self
                            .get(&ancestor.name)
                            .is_some_and(|s| s.contains(&am.name)),
                    };
                    override_stream.extend(
                        m.to_rust(trait_ctx, options.clone(), symbols.clone())?,
                    );
                }
            }
            ancestor_impls.extend(quote! {
                impl #ancestor_trait for #class_name {
                    #accessor_impls
                    #override_stream
                }
            });
        }

        Ok(quote!(#own_trait #ancestor_impls))
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
