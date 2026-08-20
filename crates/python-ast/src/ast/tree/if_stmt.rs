use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor, extract_list
};

use super::Statement;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct If {
    pub test: ExprType,
    pub body: Vec<Statement>,
    pub orelse: Vec<Statement>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for If {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let test = ob.extract_attr_with_context("test", "if test condition")?;
        let test = test
            .extract()
            .map_err(|e| crate::extraction_failure("if condition", &ob, e))?;
        
        let body: Vec<Statement> = extract_list(&ob, "body", "if body statements")?;
        let orelse: Vec<Statement> = extract_list(&ob, "orelse", "if else statements")?;
        
        Ok(If {
            test,
            body,
            orelse,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(If { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for If {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = self.test.find_symbols(symbols);
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        self.orelse.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Regular if statement handling; the test is a condition position,
        // so Python truthiness applies.
        let test =
            crate::condition_to_rust(&self.test, ctx.clone(), options.clone(), symbols.clone())?;

        // Issue #125: `if x is not None:` narrows x to its inner type inside
        // the body — reads unwrap (Name::to_rust consults narrowed_names),
        // and the comprehension/iteration over x sees the inner element
        // type. Any other test narrows nothing.
        let mut body_options = options.clone();
        if let Some((narrowed, inner)) = crate::narrowing_from_test(&self.test, &options) {
            let mut narrowed_names = options.narrowed_names.as_ref().clone();
            narrowed_names.insert(narrowed.clone());
            body_options.narrowed_names = std::rc::Rc::new(narrowed_names);
            if let Some(inner) = inner {
                let mut name_types = options.name_types.as_ref().clone();
                name_types.insert(narrowed.clone(), inner);
                body_options.name_types = std::rc::Rc::new(name_types);
            }
        }

        let body_stmts: Result<Vec<_>, _> = self
            .body
            .into_iter()
            .map(|stmt| stmt.to_rust(ctx.clone(), body_options.clone(), symbols.clone()))
            .collect();
        let body_stmts = body_stmts?;
        
        if self.orelse.is_empty() {
            Ok(quote! {
                if #test {
                    #(#body_stmts;)*
                }
            })
        } else {
            let else_stmts: Result<Vec<_>, _> = self.orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let else_stmts = else_stmts?;
            
            Ok(quote! {
                if #test {
                    #(#body_stmts;)*
                } else {
                    #(#else_stmts;)*
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_if, "if x > 5:\n    print('big')", "if_test.py");
    create_parse_test!(test_if_else, "if x > 5:\n    print('big')\nelse:\n    print('small')", "if_test.py");
    create_parse_test!(test_if_elif, "if x > 10:\n    print('huge')\nelif x > 5:\n    print('big')\nelse:\n    print('small')", "if_test.py");
}