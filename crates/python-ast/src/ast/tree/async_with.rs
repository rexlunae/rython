use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, Statement, SymbolTableScopes,
    extract_list,
};

/// Async with statement (async with context as var: ...)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AsyncWith {
    /// The with items (context managers)
    pub items: Vec<WithItem>,
    /// The body of the with statement
    pub body: Vec<Statement>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// A with item (context_expr as optional_vars)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WithItem {
    /// The context expression (the thing being entered)
    pub context_expr: ExprType,
    /// Optional variable to bind the context to
    pub optional_vars: Option<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for AsyncWith {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract items (list of withitem objects)
        let items: Vec<WithItem> = extract_list(&ob, "items", "async with items")?;
        
        // Extract body
        let body: Vec<Statement> = extract_list(&ob, "body", "async with body")?;
        
        Ok(AsyncWith {
            items,
            body,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for WithItem {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract context_expr
        let context_expr: ExprType = ob.getattr("context_expr")?.extract()?;
        
        // Extract optional_vars (optional)
        let optional_vars: Option<ExprType> = if let Ok(vars_attr) = ob.getattr("optional_vars") {
            if vars_attr.is_none() {
                None
            } else {
                Some(vars_attr.extract()?)
            }
        } else {
            None
        };
        
        Ok(WithItem {
            context_expr,
            optional_vars,
        })
    }
}

impl Node for AsyncWith {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for AsyncWith {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process items and body
        let symbols = self.items.into_iter().fold(symbols, |acc, item| {
            let acc = item.context_expr.find_symbols(acc);
            if let Some(vars) = item.optional_vars {
                vars.find_symbols(acc)
            } else {
                acc
            }
        });
        self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // The statement's own binding mark, recorded at the top of the
        // body where Python binds the target (Devin review on #338,
        // round 9); cleared for the body's statements.
        let mut options = options;
        let target_bind = options
            .loop_target_bind
            .take()
            .map(|(word, mask)| quote!(__rython_bind__(#word, #mask);));
        // Evaluate each context manager and bind its `as` target, mirroring
        // the synchronous `with` lowering (async __aenter__/__aexit__
        // protocol semantics are not modeled yet).
        let mut item_tokens = Vec::new();
        for item in self.items {
            let context_expr =
                item.context_expr
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            match item.optional_vars {
                Some(vars) => {
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    item_tokens.push(quote! { let mut #target = #context_expr; });
                }
                None => {
                    item_tokens.push(quote! { let _ = #context_expr; });
                }
            }
        }

        let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self.body.into_iter()
            .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let mut body_tokens = body_tokens?;
        if let Some(bind) = target_bind {
            body_tokens.insert(0, bind);
        }

        Ok(quote! {
            {
                #(#item_tokens)*
                #(#body_tokens;)*
            }
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_async_with, "async with context:\n    pass", "test.py");
}