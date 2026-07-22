use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableNode,
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
            .expect(
                ob.error_message("<unknown>", "error getting assignment targets")
                    .as_str(),
            )
            .extract()
            .expect("extracting assignment targets");

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
        let mut target_streams = Vec::new();
        
        // Convert each target to Rust code
        for target in self.targets {
            let target_code = target.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            target_streams.push(target_code);
        }
        
        let value = self.value.to_rust(ctx, options, symbols)?;
        
        // For single target assignment
        if target_streams.len() == 1 {
            let target = &target_streams[0];
            // Check if this is a new variable declaration or reassignment
            // For now, we'll use `let` for new declarations
            Ok(quote!(let #target = #value;))
        } else {
            // For multiple assignment targets like: a, b = 1, 2
            // Use tuple destructuring in Rust
            Ok(quote! {
                let (#(#target_streams),*) = #value;
            })
        }
    }
}
