use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor
};

/// A subscript's bracket contents: a plain index or a slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SubscriptKind {
    Index(Box<ExprType>),
    Slice {
        lower: Option<Box<ExprType>>,
        upper: Option<Box<ExprType>>,
        step: Option<Box<ExprType>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Subscript {
    pub value: Box<ExprType>,
    pub kind: SubscriptKind,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Subscript {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyTypeMethods;

        let value = ob.extract_attr_with_context("value", "subscript value")?;
        let value = value.extract().map_err(|e| extraction_failure("getting subscript value", &ob, e))?;

        let slice_attr = ob.extract_attr_with_context("slice", "subscript slice")?;
        let slice_type: String = slice_attr
            .get_type()
            .name()
            .and_then(|n| n.extract())
            .map_err(|e| extraction_failure("subscript slice type", &ob, e))?;

        let kind = if slice_type == "Slice" {
            let bound = |name: &str| -> PyResult<Option<Box<ExprType>>> {
                match slice_attr.getattr(name) {
                    Ok(v) if !v.is_none() => Ok(Some(Box::new(v.extract().map_err(
                        |e| extraction_failure("slice bound", &ob, e),
                    )?))),
                    _ => Ok(None),
                }
            };
            SubscriptKind::Slice {
                lower: bound("lower")?,
                upper: bound("upper")?,
                step: bound("step")?,
            }
        } else {
            let index = slice_attr
                .extract()
                .map_err(|e| extraction_failure("getting subscript slice", &ob, e))?;
            SubscriptKind::Index(Box::new(index))
        };

        Ok(Subscript {
            value: Box::new(value),
            kind,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(Subscript { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for Subscript {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let value = self.value.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        match self.kind {
            // Python index rules via PyIndex: negatives from the end, a
            // catchable IndexError/KeyError instead of a Rust panic.
            SubscriptKind::Index(index) => {
                let index = index.to_rust(ctx, options, symbols)?;
                Ok(quote! { (#value).py_index(#index)? })
            }
            // Slices clamp and never raise.
            SubscriptKind::Slice { lower, upper, step } => {
                let bound = |b: Option<Box<ExprType>>| -> Result<TokenStream, Box<dyn std::error::Error>> {
                    Ok(match b {
                        Some(e) => {
                            let t = e.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                            quote!(Some(#t))
                        }
                        None => quote!(None),
                    })
                };
                let lower = bound(lower)?;
                let upper = bound(upper)?;
                let step = bound(step)?;
                Ok(quote! { (#value).py_slice(#lower, #upper, #step) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_list_subscript, "a[0]", "subscript_test.py");
    create_parse_test!(test_dict_subscript, "d['key']", "subscript_test.py");
    create_parse_test!(test_nested_subscript, "matrix[i][j]", "subscript_test.py");
}