use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableNode,
    SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Assign {
    pub targets: Vec<ExprType>,
    pub value: ExprType,
    pub type_comment: Option<String>,
    /// The annotation of an annotated assignment (`x: list[float] = []`),
    /// carried so type analysis can honor it for empty-container pinning
    /// (the error message at the empty literal suggests exactly this form —
    /// the annotation must not be discarded, Devin review on #103).
    pub annotation: Option<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Assign {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let targets: Vec<ExprType> = ob
            .getattr("targets")
            .map_err(|e| extraction_failure("assignment targets", &ob, e))?
            .extract()
            .map_err(|e| extraction_failure("assignment targets", &ob, e))?;

        let python_value = ob.getattr("value").map_err(|e| extraction_failure("value", &ob, e))?;

        let value = python_value.extract().map_err(|e| extraction_failure("python_value", &ob, e))?;

        Ok(Assign {
            targets: targets,
            value: value,
            type_comment: None,
            annotation: None,
        })
    }
}

impl<'a> CodeGen for Assign {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        let mut position = 0;
        for target in self.targets {
            // Only add symbols for Name assignments, not for Attribute assignments
            if let ExprType::Name(name) = target {
                // A rust.bind(...) / rust.c_bind(...) declaration: record the
                // parsed binding so call sites can lower to direct calls.
                // Parse failures surface later, in to_rust, which re-parses
                // and reports them loudly.
                if let ExprType::Call(call) = &self.value {
                    if crate::is_rust_bind_call(&self.value) {
                        if let Ok(spec) = crate::parse_rust_bind(call) {
                            symbols.insert(name.id, SymbolTableNode::RustBinding(spec));
                        }
                        continue;
                    }
                }
                symbols.insert(
                    name.id,
                    SymbolTableNode::Assign {
                        position: position,
                        value: self.value.clone(),
                    },
                );
            }
            // Could also handle other target types here if needed
            position += 1;
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // rust.bind / rust.c_bind declarations are compile-time-only: the
        // assignment emits nothing (the binding lives in the symbol table).
        // Everything about the declaration is validated here, loudly.
        if crate::is_rust_bind_call(&self.value) {
            if !matches!(ctx, CodeGenContext::Module(_)) {
                return Err("rust.bind declarations must be at module level, not inside \
                            a function or class"
                    .to_string()
                    .into());
            }
            if self.targets.len() != 1 || !matches!(self.targets[0], ExprType::Name(_)) {
                return Err("rust.bind must be assigned to exactly one name"
                    .to_string()
                    .into());
            }
            if let ExprType::Call(call) = &self.value {
                crate::parse_rust_bind(call)?;
            }
            return Ok(TokenStream::new());
        }

        let value_is_none_early = crate::is_none_expr(&self.value);
        let value_yields_option = crate::expr_yields_option(&self.value, &options, &symbols);
        let value_expr = self.value.clone();
        let mut value = self
            .value
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        // `x = []` / `x = {}` with a pinned element type (from a later
        // append/insert/indexed-store/use, or a later typed assignment):
        // render the empty literal with explicit types so the store does
        // not poison the binding with an unconstrained Vec<_>/PyDict<_, _>
        // that rustc then rejects at the first use. An empty literal that
        // was never pinned is a loud conversion-time error (issue #77)
        // rather than a cryptic rustc mismatch inside generated code.
        if let [ExprType::Name(name)] = self.targets.as_slice() {
            let empty_literal = match &value_expr {
                ExprType::List(l) if l.is_empty() => true,
                ExprType::Dict(d) if d.keys.is_empty() => true,
                _ => false,
            };
            if empty_literal {
                // name_types carries the FINAL per-name type: later typed
                // assignments and pinning uses both refine it, so an empty
                // store is rendered against the binding's final type.
                let pinned = options.name_types.get(&name.id).cloned();
                match pinned {
                    Some(crate::TypeInfo::Vec(inner)) if !matches!(*inner, crate::TypeInfo::PyObject) => {
                        let t = inner.to_rust_type();
                        value = quote!(Vec::<#t>::new());
                    }
                    Some(crate::TypeInfo::Dict(k, v))
                        if !matches!(*k, crate::TypeInfo::PyObject)
                            && !matches!(*v, crate::TypeInfo::PyObject) =>
                    {
                        let k = k.to_rust_type();
                        let v = v.to_rust_type();
                        value = quote!(PyDict::<#k, #v>::from([]));
                    }
                    Some(_) | None => {
                        return Err(format!(
                            "empty container literal has no inferable element type; \
                             annotate the variable (e.g. `{}: list[float] = []` or \
                             `{}: dict[str, int] = {{}}`) or add a use that pins \
                             the type (`{}.append(...)`, `{}[k] = v`)",
                            name.id, name.id, name.id, name.id
                        )
                        .into());
                    }
                }
            }
        }

        // Render one assignment for a single target. Python variables are
        // function-scoped, so name targets are declared once (hoisted to a
        // `let mut` at the top of the enclosing function/module scope by the
        // scope's code generator) and every assignment is a plain store —
        // emitting `let mut` per assignment would create a fresh shadowing
        // binding inside nested blocks, silently dropping the store.
        // A name that holds an Option (assigned None on some path) wraps
        // its non-None stores in Some, so both arms unify to Option<T> —
        // unless the value is already an Option (dict.get, another optional
        // name, an Optional-returning call), which stores through unchanged.
        let value_is_none = value_is_none_early;
        // A string literal stored into an attribute becomes an owned String:
        // struct fields hold String (Python strings are owned values), while
        // the literal itself is a &'static str.
        let value_is_str_literal = matches!(
            &value_expr,
            ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_)))
        );
        let render_one = |target: &ExprType,
                          value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            // An attribute store target renders in place flavor: in a
            // generic trait default, `self.f = v` must store through the
            // mutable accessor (`*self.f_mut() = v`) rather than the load
            // accessor (which clones).
            let target_code = match target {
                ExprType::Attribute(attr) => crate::ast::tree::attribute::to_rust_place(
                    &attr.value,
                    &attr.attr,
                    &ctx,
                    &options,
                    &symbols,
                    true,
                )?,
                _ => target
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?,
            };
            Ok(match target {
                ExprType::Name(name) => {
                    if !value_is_none
                        && !value_yields_option
                        && options.optional_names.contains(&name.id)
                    {
                        quote!(#target_code = Some(#value);)
                    } else {
                        quote!(#target_code = #value;)
                    }
                }
                // Destructuring assignment to the hoisted names.
                ExprType::Tuple(_) => quote!((#target_code) = #value;),
                ExprType::Attribute(_) if value_is_str_literal => {
                    quote!(#target_code = (#value).to_string();)
                }
                _ => quote!(#target_code = #value;),
            })
        };

        // Subscript stores don't go through the Load-position lowering
        // (which reads via py_index): `x[i] = v` follows Python index rules
        // through py_set_index — negatives from the end, catchable
        // IndexError for lists, insert-or-overwrite for dicts.
        let render_subscript_store = |sub: &crate::Subscript,
                                      value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            // The receiver must be a PLACE (nested subscripts thread
            // through py_index_mut): the Load lowering would clone, and the
            // store would silently land on the clone.
            let receiver = crate::subscript_receiver_place(
                &sub.value,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?;
            // `os.environ[k] = v` routes through `os::setenv`: os.environ
            // is an IMMUTABLE module static (a live view of the process
            // environment) and cannot be borrowed mutably for py_set_index.
            // The static's set-index impls (Environ: PySetIndex) cover
            // receivers that are real values (`e = os.environ`), but the
            // direct module-path store must go through the function.
            let is_os_environ = matches!(
                sub.value.as_ref(),
                ExprType::Attribute(a)
                    if a.attr == "environ"
                        && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "os")
                        && !crate::ast::tree::call::root_name(&a.value)
                            .is_some_and(|root| crate::module_name_shadowed(root, &symbols))
            );
            match &sub.kind {
                crate::SubscriptKind::Index(index) => {
                    // String-keyed dicts store `&str` indexes through
                    // py_set_index(String, V), so a &str literal index is
                    // owned at the store site.
                    let string_keyed = matches!(
                        sub.value.as_ref(),
                        ExprType::Name(n)
                            if matches!(
                                options.name_types.get(&n.id),
                                Some(crate::TypeInfo::Dict(k, _))
                                    if matches!(**k, crate::TypeInfo::String)
                            )
                    );
                    let index = if string_keyed {
                        crate::render_typed(
                            index,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?
                    } else {
                        index
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?
                    };
                    if is_os_environ {
                        return Ok(quote!(os::setenv(#index, #value);));
                    }
                    Ok(quote!({
                        let __rython_val = #value;
                        (#receiver).py_set_index(#index, __rython_val)?;
                    }))
                }
                crate::SubscriptKind::Slice { .. } => Err(
                    "slice assignment (`x[a:b] = ...`) is not yet supported"
                        .to_string()
                        .into(),
                ),
            }
        };

        let render = |target: &ExprType,
                      value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            match target {
                ExprType::Subscript(sub) => render_subscript_store(sub, value),
                _ => render_one(target, value),
            }
        };

        if self.targets.len() == 1 {
            // A store into an optional-tracked name goes through the
            // Option-slot lowering, which passes Option values through,
            // wraps plain values in Some, and handles conditional arms
            // independently (`x if c else None`).
            if let ExprType::Name(name) = &self.targets[0]
                && options.optional_names.contains(&name.id)
            {
                let target_code = self.targets[0].clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                let value = crate::lower_optional_value(
                    &value_expr,
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                return Ok(quote!(#target_code = #value;));
            }
            render(&self.targets[0], &value)
        } else {
            // Chained assignment (`a = b = expr`): Python evaluates the value
            // once and assigns it to each target in turn.
            // A container literal would break the aliasing semantics: Python
            // shares ONE object across all targets, while the lowering must
            // clone per target — later mutations through one name would
            // silently diverge. That is a loud conversion error, not a
            // silent divergence (issue #80).
            if is_container_literal(&value_expr) {
                return Err(
                    "chained assignment to a container literal (`a = b = [...]`) \
                     cannot preserve Python's shared aliasing: rython would give \
                     each target a separate copy. Assign the literal to one name \
                     first, then copy it to the others"
                        .to_string()
                        .into(),
                );
            }
            let mut stream = quote!(let __rython_chain = #value;);
            for target in &self.targets {
                stream.extend(render(target, &quote!(__rython_chain.clone()))?);
            }
            Ok(stream)
        }
    }
}

fn is_container_literal(expr: &ExprType) -> bool {
    matches!(
        expr,
        ExprType::List(_)
            | ExprType::Dict(_)
            | ExprType::Set(_)
            | ExprType::ListComp(_)
            | ExprType::DictComp(_)
            | ExprType::SetComp(_)
    )
}
