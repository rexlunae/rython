use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, ParameterList, PyAttributeExtractor
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Lambda {
    pub args: ParameterList,
    pub body: Box<ExprType>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Lambda {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let args = ob.extract_attr_with_context("args", "lambda arguments")?;
        let body = ob.extract_attr_with_context("body", "lambda body")?;
        
        let args = args.extract().map_err(|e| extraction_failure("getting lambda arguments", &ob, e))?;
        let body = body.extract().map_err(|e| extraction_failure("getting lambda body", &ob, e))?;
        
        Ok(Lambda {
            args,
            body: Box::new(body),
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(Lambda { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for Lambda {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Closure parameters must be bare names — `impl Trait` (which typed
        // function parameters use) is illegal in closure position; Rust
        // infers closure parameter types.
        let params: Vec<_> = self
            .args
            .args
            .iter()
            .map(|param| crate::safe_ident(&param.arg))
            .collect();
        let body = self.body.to_rust(ctx, options, symbols)?;

        Ok(quote! {
            |#(#params),*| #body
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_lambda, "lambda x: x + 1", "lambda_test.py");
    create_parse_test!(test_lambda_with_args, "lambda x, y: x * y", "lambda_test.py");
    create_parse_test!(test_lambda_no_args, "lambda: 42", "lambda_test.py");
}