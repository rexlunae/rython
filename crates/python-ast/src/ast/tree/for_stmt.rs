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
pub struct For {
    pub target: ExprType,
    pub iter: ExprType,
    pub body: Vec<Statement>,
    pub orelse: Vec<Statement>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for For {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let target = ob.extract_attr_with_context("target", "for loop target")?;
        let iter = ob.extract_attr_with_context("iter", "for loop iterator")?;
        
        let target = target
            .extract()
            .map_err(|e| crate::extraction_failure("for loop target", &ob, e))?;
        let iter = iter
            .extract()
            .map_err(|e| crate::extraction_failure("for loop iterator", &ob, e))?;
        
        let body: Vec<Statement> = extract_list(&ob, "body", "for body statements")?;
        let orelse: Vec<Statement> = extract_list(&ob, "orelse", "for else statements")?;
        
        Ok(For {
            target,
            iter,
            body,
            orelse,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(For { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for For {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = self.target.find_symbols(symbols);
        let symbols = self.iter.find_symbols(symbols);
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        self.orelse.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let target = self.target.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        let iter = self.iter.to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        let has_else = !self.orelse.is_empty();
        // Break-tracking is only needed when the else clause could be skipped
        // by a break belonging to this loop; otherwise the flag would be
        // declared mutable but never written.
        let tracks_break = has_else && crate::loop_body_has_direct_break(&self.body);
        // Body statements compile inside a Loop context so `break` can honor
        // the else clause; the else clause itself is outside the loop.
        let body_ctx = crate::CodeGenContext::Loop {
            has_else: tracks_break,
            parent: Box::new(ctx.clone()),
        };
        let body_stmts: Result<Vec<_>, _> = self.body
            .into_iter()
            .map(|stmt| stmt.to_rust(body_ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let body_stmts = body_stmts?;

        if !has_else {
            Ok(quote! {
                for #target in #iter {
                    #(#body_stmts;)*
                }
            })
        } else {
            // Python's for/else: the else clause runs iff the loop finished
            // without hitting `break` (which sets __rython_broke).
            let else_stmts: Result<Vec<_>, _> = self.orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let else_stmts = else_stmts?;

            if tracks_break {
                Ok(quote! {
                    {
                        let mut __rython_broke = false;
                        for #target in #iter {
                            #(#body_stmts;)*
                        }
                        if !__rython_broke {
                            #(#else_stmts;)*
                        }
                    }
                })
            } else {
                // No break can skip the else clause; run it unconditionally.
                Ok(quote! {
                    {
                        for #target in #iter {
                            #(#body_stmts;)*
                        }
                        #(#else_stmts;)*
                    }
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_simple_for, "for x in range(10):\n    print(x)", "for_test.py");
    create_parse_test!(test_for_else, "for x in range(10):\n    print(x)\nelse:\n    print('done')", "for_test.py");
    create_parse_test!(test_for_list, "for item in [1, 2, 3]:\n    print(item)", "for_test.py");
}