use proc_macro2::{TokenStream, TokenTree};
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
        let body = wrap_fallible_body(self.body.to_rust(ctx, options, symbols)?);

        Ok(quote! {
            |#(#params),*| #body
        })
    }
}

/// A lambda body is a plain Rust closure, but fallible expressions lower to
/// `?` (`x // 2`, `int(s)`, a keyword call to a user function, ...), which
/// is only legal in a function or closure returning `Result`. When the
/// lowered body contains `?`, wrap it in an immediately-invoked
/// Result-returning closure and unwrap: the exception becomes a loud panic
/// at the lambda's call site (with the exception's message) instead of a
/// rustc error. An enclosing Python `try`/`except` cannot catch it — a
/// rython lambda is a plain closure with nowhere to route the Err, where
/// CPython would let the caller catch — so loud-and-clear beats failing to
/// compile (issues #75-family, Devin review on #103).
fn wrap_fallible_body(body: TokenStream) -> TokenStream {
    // The scan reaches into groups: a `?` inside a comprehension block or
    // a parenthesized call (`lambda s: [f(x) for x in s]`) is as fallible
    // as one at the top level, and was left unwrapped in a non-Result
    // closure before (a rustc error).
    fn contains_question(stream: TokenStream) -> bool {
        stream.into_iter().any(|tt| match tt {
            TokenTree::Punct(p) => p.as_char() == '?',
            TokenTree::Group(g) => contains_question(g.stream()),
            _ => false,
        })
    }
    if !contains_question(body.clone()) {
        return body;
    }
    quote! {{
        (|| -> Result<_, PyException> { Ok(#body) })()
            .unwrap_or_else(|e| panic!("{}", e))
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_lambda, "lambda x: x + 1", "lambda_test.py");
    create_parse_test!(test_lambda_with_args, "lambda x, y: x * y", "lambda_test.py");
    create_parse_test!(test_lambda_no_args, "lambda: 42", "lambda_test.py");
}