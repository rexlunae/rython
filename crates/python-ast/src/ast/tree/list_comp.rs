use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
    PyAttributeExtractor, extract_list,
};

/// List comprehension (e.g., [x ** 2 for x in range(10) if x % 2 == 0])
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ListComp {
    /// The element expression being computed
    pub elt: Box<ExprType>,
    /// The generators (for clauses)
    pub generators: Vec<Comprehension>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Set comprehension (e.g., {x for x in range(10) if x % 2 == 0})
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SetComp {
    /// The element expression being computed
    pub elt: Box<ExprType>,
    /// The generators (for clauses)
    pub generators: Vec<Comprehension>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Generator expression (e.g., (x for x in range(10) if x % 2 == 0))
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeneratorExp {
    /// The element expression being computed
    pub elt: Box<ExprType>,
    /// The generators (for clauses)
    pub generators: Vec<Comprehension>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Dictionary comprehension (e.g., {k: v for k, v in items.items()})
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DictComp {
    /// The key expression being computed
    pub key: Box<ExprType>,
    /// The value expression being computed
    pub value: Box<ExprType>,
    /// The generators (for clauses)
    pub generators: Vec<Comprehension>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// A comprehension generator (for x in iter if condition)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Comprehension {
    /// The target variable(s) (e.g., x in "for x in range(10)")
    pub target: ExprType,
    /// The iterable expression (e.g., range(10) in "for x in range(10)")
    pub iter: ExprType,
    /// The conditions (if clauses)
    pub ifs: Vec<ExprType>,
    /// Whether this is an async comprehension
    pub is_async: bool,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ListComp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the element expression
        let elt = ob.extract_attr_with_context("elt", "list comprehension element")?;
        let elt: ExprType = elt.extract()?;
        
        // Extract generators
        let generators: Vec<Comprehension> = extract_list(&ob, "generators", "list comprehension generators")?;
        
        Ok(ListComp {
            elt: Box::new(elt),
            generators,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for SetComp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the element expression
        let elt = ob.extract_attr_with_context("elt", "set comprehension element")?;
        let elt: ExprType = elt.extract()?;
        
        // Extract generators
        let generators: Vec<Comprehension> = extract_list(&ob, "generators", "set comprehension generators")?;
        
        Ok(SetComp {
            elt: Box::new(elt),
            generators,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for GeneratorExp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the element expression
        let elt = ob.extract_attr_with_context("elt", "generator expression element")?;
        let elt: ExprType = elt.extract()?;
        
        // Extract generators
        let generators: Vec<Comprehension> = extract_list(&ob, "generators", "generator expression generators")?;
        
        Ok(GeneratorExp {
            elt: Box::new(elt),
            generators,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DictComp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the key expression
        let key = ob.extract_attr_with_context("key", "dict comprehension key")?;
        let key: ExprType = key.extract()?;
        
        // Extract the value expression
        let value = ob.extract_attr_with_context("value", "dict comprehension value")?;
        let value: ExprType = value.extract()?;
        
        // Extract generators
        let generators: Vec<Comprehension> = extract_list(&ob, "generators", "dict comprehension generators")?;
        
        Ok(DictComp {
            key: Box::new(key),
            value: Box::new(value),
            generators,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Comprehension {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract target
        let target = ob.extract_attr_with_context("target", "comprehension target")?;
        let target: ExprType = target.extract()?;
        
        // Extract iter
        let iter = ob.extract_attr_with_context("iter", "comprehension iter")?;
        let iter: ExprType = iter.extract()?;
        
        // Extract ifs (list of conditions)
        let ifs: Vec<ExprType> = extract_list(&ob, "ifs", "comprehension conditions").unwrap_or_default();
        
        // Extract is_async
        let is_async: bool = ob.getattr("is_async")?.extract().unwrap_or(false);
        
        Ok(Comprehension {
            target,
            iter,
            ifs,
            is_async,
        })
    }
}

impl Node for ListComp {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for SetComp {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for GeneratorExp {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for DictComp {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

/// Lower a comprehension's generator clauses into nested `for` loops around
/// `inner`, binding each generator's real target name (so the element and
/// condition expressions can reference it) and applying `if` guards with
/// `continue`. Generators nest left-to-right, matching Python's evaluation
/// order, and later generators may reference earlier targets.
///
/// `options` is the OUTER scope; each generator's iterable, target and
/// filters render in the prefix scope that binds exactly the targets
/// before / through it (`comprehension_prefix_scopes`).
///
/// The names a comprehension's `if` FILTERS prove non-None for the
/// element: the element runs only after every guard passed, so a name a
/// filter keeps is a plain value there. Two filter shapes narrow: the
/// `x is not None` compare `narrowings_from_test` sees, and a filter
/// that IS a bare Option-typed name (`{r for r in detected_ranges if r}`
/// — charset_normalizer's alphabets: detected_ranges is `list[str |
/// None]`, and the guard's Option-truthiness model (a None member is
/// falsy, so the guard continues on it) makes the element read the
/// inner String — the comprehension must insert Strings, not Options).
/// Only Option-typed names narrow (mirroring the statement guard rules).
fn comprehension_filter_narrowings(
    generators: &[crate::Comprehension],
    options: &PythonOptions,
) -> Vec<(String, crate::TypeInfo)> {
    let mut out: Vec<(String, crate::TypeInfo)> = Vec::new();
    for generator in generators {
        for if_expr in &generator.ifs {
            let mut names: Vec<String> = Vec::new();
            if let ExprType::Name(n) = if_expr {
                names.push(n.id.clone());
            }
            for (name, _) in crate::narrowings_from_test(if_expr, options) {
                names.push(name);
            }
            for name in names {
                if let Some(crate::TypeInfo::Option(inner)) = options.name_types.get(&name)
                    && !out.iter().any(|(seen, _)| *seen == name)
                {
                    out.push((name.clone(), (**inner).clone()));
                }
            }
        }
    }
    out
}

/// Apply a name's non-None narrowing to a comprehension element scope,
/// the way the if-statement installs its body narrowing (reads of the
/// name unwrap, and the name_types entry becomes the inner type so the
/// element's own typing agrees).
fn apply_comprehension_narrowing(
    scope: &mut PythonOptions,
    name: &str,
    inner: crate::TypeInfo,
) {
    // The narrowed READ of an Option name unwraps only when the name is
    // still recorded as Option-typed (name.rs consults optional_names):
    // a comprehension target may not have been, so mark it — the guard
    // proved the None case skipped.
    let mut optional = scope.optional_names.as_ref().clone();
    optional.insert(name.to_string());
    scope.optional_names = std::rc::Rc::new(optional);
    let mut narrowed = scope.narrowed_names.as_ref().clone();
    narrowed.insert(name.to_string(), inner.clone());
    scope.narrowed_names = std::rc::Rc::new(narrowed);
    let mut types = scope.name_types.as_ref().clone();
    types.insert(name.to_string(), inner);
    scope.name_types = std::rc::Rc::new(types);
}

fn build_comprehension_loops(
    generators: &[Comprehension],
    inner: TokenStream,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let scopes = crate::comprehension_prefix_scopes(generators, Some(ctx), options, symbols);
    let mut acc = inner;
    for (i, generator) in generators.iter().enumerate().rev() {
        // Generator i's iterable sees the targets bound BEFORE it; its
        // target and filters see its own binding too (the prefix scopes).
        let iter_options = &scopes[i];
        let scope = &scopes[i + 1];
        let target = generator
            .target
            .clone()
            .to_rust(ctx.clone(), scope.clone(), symbols.clone())?;
        // The iterable takes the reuse-clone a for-statement's does: the
        // loop consumes it, and a name read again later (`sum(s.area()
        // for s in shapes)` then `max(shapes, ...)` — the idiom corpus's
        // shapes) would otherwise be a use after move (E0382).
        // An ALL-CONSTANT tuple iterable (`for k in (35, 36, 80)` — the
        // idiom corpus's tree, round 99) iterates as an array of the
        // element tokens, the same rule the for-statement lowering uses
        // (E0277: the Rust tuple is not IntoIterator).
        let iter_expr = match &generator.iter {
            ExprType::Tuple(t)
                if t.elts.iter().all(|e| {
                    matches!(e, ExprType::Constant(_) | ExprType::UnaryOp(_))
                }) =>
            {
                let mut elts = Vec::with_capacity(t.elts.len());
                for elt in &t.elts {
                    let tok = elt.clone().to_rust(
                        ctx.clone(),
                        iter_options.clone(),
                        symbols.clone(),
                    )?;
                    if matches!(
                        elt,
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::String(_)))
                    ) {
                        elts.push(quote!((#tok).to_string()));
                    } else {
                        elts.push(tok);
                    }
                }
                quote!([#(#elts),*])
            }
            _ => {
                let mut it = crate::render_reused(
                    &generator.iter,
                    ctx.clone(),
                    iter_options.clone(),
                    symbols.clone(),
                )?;
                // A DICT-typed iterable (`for w in counts` in a
                // comprehension — text_stats's starts-with-q, round 99):
                // Python iterates the KEYS.
                if matches!(
                    crate::infer_type(
                        Some(ctx),
                        &generator.iter,
                        iter_options,
                        symbols
                    ),
                    crate::TypeInfo::Dict(_, _)
                ) {
                    it = quote!(#it . py_keys ());
                }
                // A STRING-typed iterable (`all(_.isupper() for _ in
                // buf)` — charset_normalizer's md.py, which iterates its
                // accumulated `_buffer: str`): Python iterates
                // one-character strings and the element analysis types
                // them String — iterate the chars mapped back to
                // one-char Strings (a raw String is not IntoIterator).
                if matches!(
                    crate::infer_type(
                        Some(ctx),
                        &generator.iter,
                        iter_options,
                        symbols
                    ),
                    crate::TypeInfo::String | crate::TypeInfo::StrRef
                ) {
                    it = quote!(
                        #it . chars () . map (| __rython_char | __rython_char . to_string ())
                    );
                }
                it
            }
        };
        let conditions: Result<Vec<_>, _> = generator
            .ifs
            .iter()
            .map(|if_expr| {
                // The filter lowers truthiness through the SAME authority
                // the if-statement uses (`condition_to_rust` → the
                // `(#tokens).is_truthy()` contract — Directive 5): the
                // raw `!(w.strip())` applied `!` to a String (E0600 in
                // the idiom corpus's `[w.strip() for w in ... if
                // w.strip()]`).
                crate::condition_to_rust(
                    if_expr,
                    ctx.clone(),
                    scope.clone(),
                    symbols.clone(),
                )
            })
            .collect();
        let conditions = conditions?;
        let guard = if conditions.is_empty() {
            quote!()
        } else {
            quote! { if !( #((#conditions))&&* ) { continue; } }
        };
        acc = quote! {
            for #target in #iter_expr {
                #guard
                #acc
            }
        };
    }
    Ok(acc)
}

impl CodeGen for ListComp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process the element and generators
        let symbols = (*self.elt).clone().find_symbols(symbols);
        self.generators.into_iter().fold(symbols, |acc, generator| {
            let acc = generator.target.find_symbols(acc);
            let acc = generator.iter.find_symbols(acc);
            generator.ifs.into_iter().fold(acc, |acc, if_expr| if_expr.find_symbols(acc))
        })
    }

        fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // The element type an ANNOTATED binding threads in (`pets:
        // list[Animal] = [Cat(..), Dog(..)]` — the sibling-class pushes
        // coerce to the hierarchy root, round 114): without it a
        // heterogeneous Vec cannot even name its element.
                // `X.__subclasses__()` — the DETECTOR-REGISTRY shape
        // (`[md_class() for md_class in MessDetectorPlugin.__subclasses__()]`
        // — charset_normalizer's mess_ratio, round 114): the registry
        // enumerates the class's KNOWN subclasses — statically, from the
        // hierarchy index — and the element (a no-argument construction of
        // the bound target) renders per subclass, coerced to the root the
        // binding's annotation names. Python's `__subclasses__` is live
        // metadata; the static enumeration is the registry's only
        // representable form — subclasses defined outside the converted
        // modules are invisible (a divergence on the ledger), and a
        // non-construction element stays on the general path below.
        {
            let registry_shape = self.generators.len() == 1
                && {
                    let iter_is_subclasses = match &self.generators[0].iter {
                        ExprType::Call(c) => matches!(
                            c.func.as_ref(),
                            ExprType::Attribute(a)
                                if a.attr == "__subclasses__"
                                    && matches!(
                                        a.value.as_ref(),
                                        ExprType::Name(b)
                                            if matches!(
                                                symbols.get(&b.id),
                                                Some(crate::SymbolTableNode::ClassDef(_))
                                            )
                                    )
                        ),
                        _ => false,
                    };
                    iter_is_subclasses
                }
                && match self.elt.as_ref() {
                    ExprType::Call(c) => matches!(c.func.as_ref(), ExprType::Name(t)
                        if match &self.generators[0].target {
                            ExprType::Name(tn) => tn.id == t.id,
                            _ => false,
                        }),
                    _ => false,
                };
            if registry_shape {
                let base_name = match &self.generators[0].iter {
                    ExprType::Call(c) => match c.func.as_ref() {
                        ExprType::Attribute(a) => match a.value.as_ref() {
                            ExprType::Name(b) => b.id.clone(),
                            _ => unreachable!("checked above"),
                        },
                        _ => unreachable!("checked above"),
                    },
                    _ => unreachable!("checked above"),
                };
                let members: Vec<String> =
                    crate::ast::tree::hierarchy::subtree(&options, &base_name)
                        .map(|v| {
                            v.iter()
                                .skip(1)
                                .map(|m| m.name.clone())
                                .collect()
                        })
                        .unwrap_or_default();
                let elt_expected = (*options.forced_list_elt).clone();
                let mut pushes = TokenStream::new();
                for member in &members {
                    let construction = ExprType::Call(crate::Call {
                        func: Box::new(ExprType::Name(crate::Name {
                            id: member.clone(),
                        })),
                        args: Vec::new(),
                        keywords: Vec::new(),
                    });
                    let rendered = match &elt_expected {
                        Some(ty) => crate::render_typed(
                            &construction,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(ty.clone()),
                        )?,
                        None => construction.to_rust(ctx.clone(), options.clone(), symbols.clone())?,
                    };
                    pushes.extend(quote!(__rython_comp.push(#rendered);));
                }
                let vec_ty = match &elt_expected {
                    Some(ty) => ty.to_rust_type(),
                    None => quote!(_),
                };
                return Ok(quote! {
                    {
                        let mut __rython_comp: Vec<#vec_ty> = Vec::new();
                        #pushes
                        __rython_comp
                    }
                });
            }
        }
        let mut scope = crate::comprehension_scope(&self.generators, Some(&ctx), &options, &symbols);
        for (name, inner) in comprehension_filter_narrowings(&self.generators, &scope) {
            apply_comprehension_narrowing(&mut scope, &name, inner);
        }
        let elt_expected = (*options.forced_list_elt).clone();
        let elt = match &elt_expected {
            Some(ty) => crate::render_typed(
                &self.elt,
                ctx.clone(),
                scope.clone(),
                symbols.clone(),
                Some(ty.clone()),
            )?,
            None => (*self.elt).clone().to_rust(ctx.clone(), scope.clone(), symbols.clone())?,
        };
        let loops = build_comprehension_loops(
            &self.generators,
            quote! { __rython_comp.push(#elt); },
            &ctx,
            &options,
            &symbols,
        )?;
        let vec_ty = match &elt_expected {
            Some(ty) => ty.to_rust_type(),
            None => quote!(_),
        };
        Ok(quote! {
            {
                let mut __rython_comp: Vec<#vec_ty> = Vec::new();
                #loops
                __rython_comp
            }
        })
    }
}

impl CodeGen for SetComp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process the element and generators
        let symbols = (*self.elt).clone().find_symbols(symbols);
        self.generators.into_iter().fold(symbols, |acc, generator| {
            let acc = generator.target.find_symbols(acc);
            let acc = generator.iter.find_symbols(acc);
            generator.ifs.into_iter().fold(acc, |acc, if_expr| if_expr.find_symbols(acc))
        })
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // The element renders in the full scope, with every name an `if`
        // FILTER proved non-None narrowed to its inner (see
        // comprehension_filter_narrowings).
        let mut scope = crate::comprehension_scope(&self.generators, Some(&ctx), &options, &symbols);
        for (name, inner) in comprehension_filter_narrowings(&self.generators, &scope) {
            apply_comprehension_narrowing(&mut scope, &name, inner);
        }
        let elt = (*self.elt).clone().to_rust(ctx.clone(), scope.clone(), symbols.clone())?;
        let loops = build_comprehension_loops(
            &self.generators,
            quote! { __rython_comp.insert(#elt); },
            &ctx,
            &options,
            &symbols,
        )?;
        Ok(quote! {
            {
                let mut __rython_comp = std::collections::HashSet::new();
                #loops
                __rython_comp
            }
        })
    }
}

impl CodeGen for GeneratorExp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process the element and generators
        let symbols = (*self.elt).clone().find_symbols(symbols);
        self.generators.into_iter().fold(symbols, |acc, generator| {
            let acc = generator.target.find_symbols(acc);
            let acc = generator.iter.find_symbols(acc);
            generator.ifs.into_iter().fold(acc, |acc, if_expr| if_expr.find_symbols(acc))
        })
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Generator expressions are lowered eagerly (like a list
        // comprehension) and then turned back into an iterator; Python's lazy
        // evaluation is not modeled yet.
        // The element renders in the full scope, with every name an `if`
        // FILTER proved non-None narrowed to its inner (see
        // comprehension_filter_narrowings).
        let mut scope = crate::comprehension_scope(&self.generators, Some(&ctx), &options, &symbols);
        for (name, inner) in comprehension_filter_narrowings(&self.generators, &scope) {
            apply_comprehension_narrowing(&mut scope, &name, inner);
        }
        let elt = (*self.elt).clone().to_rust(ctx.clone(), scope.clone(), symbols.clone())?;
        let loops = build_comprehension_loops(
            &self.generators,
            quote! { __rython_comp.push(#elt); },
            &ctx,
            &options,
            &symbols,
        )?;
        Ok(quote! {
            {
                let mut __rython_comp = Vec::new();
                #loops
                __rython_comp.into_iter()
            }
        })
    }
}

impl CodeGen for DictComp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process the key, value and generators
        let symbols = (*self.key).clone().find_symbols(symbols);
        let symbols = (*self.value).clone().find_symbols(symbols);
        self.generators.into_iter().fold(symbols, |acc, generator| {
            let acc = generator.target.find_symbols(acc);
            let acc = generator.iter.find_symbols(acc);
            generator.ifs.into_iter().fold(acc, |acc, if_expr| if_expr.find_symbols(acc))
        })
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let scope = crate::comprehension_scope(&self.generators, Some(&ctx), &options, &symbols);
        let key = (*self.key).clone().to_rust(ctx.clone(), scope.clone(), symbols.clone())?;
        let value = (*self.value).clone().to_rust(ctx.clone(), scope.clone(), symbols.clone())?;
        let loops = build_comprehension_loops(
            &self.generators,
            quote! { __rython_comp.insert(#key, #value); },
            &ctx,
            &options,
            &symbols,
        )?;
        // PyDict, like dict literals: comprehension-built dicts preserve
        // insertion order too.
        Ok(quote! {
            {
                let mut __rython_comp = PyDict::new();
                #loops
                __rython_comp
            }
        })
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests might need additional AST node implementations
    // create_parse_test!(test_simple_listcomp, "[x for x in range(5)]", "test.py");
    // create_parse_test!(test_listcomp_with_condition, "[x for x in range(10) if x % 2 == 0]", "test.py");
}