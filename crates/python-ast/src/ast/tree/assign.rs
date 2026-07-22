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
        let value = self
            .value
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        // Render one assignment for a single target. Python variables are
        // function-scoped, so name targets are declared once (hoisted to a
        // `let mut` at the top of the enclosing function/module scope by the
        // scope's code generator) and every assignment is a plain store —
        // emitting `let mut` per assignment would create a fresh shadowing
        // binding inside nested blocks, silently dropping the store.
        // Class bodies are the exception: they aren't hoisted scopes.
        let in_class = matches!(ctx, CodeGenContext::Class);
        let render_one = |target: &ExprType,
                          value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            let target_code =
                target
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            Ok(match target {
                ExprType::Name(_) if in_class => quote!(let mut #target_code = #value;),
                ExprType::Name(_) => quote!(#target_code = #value;),
                // Destructuring assignment to the hoisted names.
                ExprType::Tuple(_) => quote!((#target_code) = #value;),
                _ => quote!(#target_code = #value;),
            })
        };

        if self.targets.len() == 1 {
            render_one(&self.targets[0], &value)
        } else {
            // Chained assignment (`a = b = expr`): Python evaluates the value
            // once and assigns it to each target in turn.
            let mut stream = quote!(let __rython_chain = #value;);
            for target in &self.targets {
                stream.extend(render_one(target, &quote!(__rython_chain.clone()))?);
            }
            Ok(stream)
        }
    }
}
