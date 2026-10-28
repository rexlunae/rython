use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, extract_list
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Set {
    pub elts: Vec<ExprType>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Set {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let elts: Vec<ExprType> = extract_list(&ob, "elts", "set elements")?;
        
        Ok(Set {
            elts,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(Set { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for Set {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let elements: Result<Vec<_>, _> = self.elts
            .clone()
            .into_iter()
            .map(|elt| elt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
            .collect();

        let elements = elements?;

        // PYTHON HAS ONE INTEGER TYPE (issue #137): a bare `{3, 1, 2}`
        // leaves the element type to rustc, whose `{integer}` fallback is
        // i32 -- a type no other rython construct uses (integers are i64)
        // and one the set operations reject (`i32` is not a set member
        // type, so `{3, 1, 2} & nums` was an E0277). Spelling `i64` makes
        // the literal the same set CPython has. Only fires when EVERY
        // element is an integer constant, so a mixed or float set keeps
        // its existing (loud) behaviour.
        if self.elts.iter().all(elt_is_int_constant)
            && !self.elts.is_empty()
            && matches!(
                crate::infer_type(None, &ExprType::Set(self.clone()), &options, &symbols),
                crate::TypeInfo::HashSet(inner) if matches!(*inner, crate::TypeInfo::Int)
            )
        {
            return Ok(quote! {
                std::collections::HashSet::<i64>::from([#(#elements),*])
            });
        }

        Ok(quote! {
            std::collections::HashSet::from([#(#elements),*])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_empty_set, "set()", "set_test.py");
    create_parse_test!(test_simple_set, "{1, 2, 3}", "set_test.py");
    create_parse_test!(test_set_with_variables, "{x, y, z}", "set_test.py");
}

/// Whether an expression is an INTEGER literal constant (`1`, `-3`). A
/// negative integer parses as a unary minus over a constant, so both
/// shapes count.
fn elt_is_int_constant(e: &ExprType) -> bool {
    match e {
        ExprType::Constant(c) => matches!(&c.0, Some(litrs::Literal::Integer(_))),
        ExprType::UnaryOp(u) => {
            matches!(u.op, crate::Ops::USub | crate::Ops::UAdd) && elt_is_int_constant(&u.operand)
        }
        _ => false,
    }
}
