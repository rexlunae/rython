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
        let value_is_none_early = crate::is_none_expr(&self.value);
        let value_yields_option = crate::expr_yields_option(&self.value, &options, &symbols);
        let value_expr = self.value.clone();
        let value = self
            .value
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

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
            let target_code =
                target
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
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
            match &sub.kind {
                crate::SubscriptKind::Index(index) => {
                    let index = index
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    Ok(quote!((#receiver).py_set_index(#index, #value)?;))
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
            if let ExprType::Name(name) = &self.targets[0] {
                if options.optional_names.contains(&name.id) {
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
            }
            render(&self.targets[0], &value)
        } else {
            // Chained assignment (`a = b = expr`): Python evaluates the value
            // once and assigns it to each target in turn.
            let mut stream = quote!(let __rython_chain = #value;);
            for target in &self.targets {
                stream.extend(render(target, &quote!(__rython_chain.clone()))?);
            }
            Ok(stream)
        }
    }
}
