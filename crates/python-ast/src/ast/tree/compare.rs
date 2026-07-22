use proc_macro2::TokenStream;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, extraction_failure, err_from, CodeGen, CodeGenContext, CompareNotYetImplemented, ExprType,
    PythonOptions, SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Compares {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,

    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Compare {
    ops: Vec<Compares>,
    left: Box<ExprType>,
    comparators: Vec<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Compare {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("ob: {}", dump(&ob, None)?);

        // Python allows for multiple comparators, rust we only supports one, so we have to rewrite the comparison a little.
        let ops_bound: Vec<Bound<PyAny>> = ob
            .getattr("ops")
            .map_err(|e| extraction_failure("comparison operators", &ob, e))?
            .extract()
            .map_err(|e| extraction_failure("comparison operators", &ob, e))?;

        let mut op_list = Vec::new();

        for op in ops_bound.iter() {
            let op_type = op
                .get_type()
                .name()
                .map_err(|e| extraction_failure("comparison operator type", &ob, e))?;

            let op_type_str: String = op_type.extract()?;
            let op = match op_type_str.as_str() {
                "Eq" => Compares::Eq,
                "NotEq" => Compares::NotEq,
                "Lt" => Compares::Lt,
                "LtE" => Compares::LtE,
                "Gt" => Compares::Gt,
                "GtE" => Compares::GtE,
                "Is" => Compares::Is,
                "IsNot" => Compares::IsNot,
                "In" => Compares::In,
                "NotIn" => Compares::NotIn,

                _ => {
                    tracing::debug!("Found unknown Compare with type: {}", op_type_str);
                    Compares::Unknown
                }
            };
            op_list.push(op);
        }

        let left = ob.getattr("left").map_err(|e| extraction_failure("left", &ob, e))?;

        let comparators = ob.getattr("comparators").map_err(|e| extraction_failure("comparators", &ob, e))?;
        tracing::debug!(
            "left: {}, comparators: {}",
            dump(&left, None)?,
            dump(&comparators, None)?
        );

        let left = left.extract().map_err(|e| extraction_failure("getting binary operator operand", &ob, e))?;
        let comparators: Vec<ExprType> = comparators
            .extract()
            .map_err(|e| extraction_failure("comparators", &ob, e))?;

        tracing::debug!(
            "left: {:?}, comparators: {:?}, op: {:?}",
            left,
            comparators,
            op_list
        );

        return Ok(Compare {
            ops: op_list,
            left: Box::new(left),
            comparators: comparators,
        });
    }
}

impl CodeGen for Compare {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut outer_ts = TokenStream::new();
        // Python chains comparisons pairwise: `a < b < c` means
        // `a < b && b < c`, so each comparator becomes the left operand of
        // the next comparison.
        let mut left = self
            .left
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        let ops = self.ops.clone();
        let comparators = self.comparators.clone();

        let mut index = 0;
        for op in ops.iter() {
            let comparator = comparators
                .get(index)
                .ok_or("comparison has more operators than comparators")?
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let tokens = match op {
                Compares::Eq => quote!((#left) == (#comparator)),
                Compares::NotEq => quote!((#left) != (#comparator)),
                Compares::Lt => quote!((#left) < (#comparator)),
                Compares::LtE => quote!((#left) <= (#comparator)),
                Compares::Gt => quote!((#left) > (#comparator)),
                Compares::GtE => quote!((#left) >= (#comparator)),
                Compares::Is => quote!(&#left == &#comparator),
                Compares::IsNot => quote!(&#left != &#comparator),
                // Python `in` dispatches on the container: substring for
                // strings, key lookup for dicts, element lookup for
                // sequences. The stdpython PyContains trait models that.
                Compares::In => quote!((#comparator).py_contains(&(#left))),
                Compares::NotIn => quote!(!(#comparator).py_contains(&(#left))),

                _ => return Err(err_from(CompareNotYetImplemented(self)).into()),
            };

            index += 1;
            left = comparator;

            outer_ts.extend(tokens);
            if index < ops.len() {
                outer_ts.extend(quote!( && ));
            }
        }
        Ok(outer_ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_eq() {
        let options = PythonOptions::default();
        let result = crate::parse("1 == 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //info!("{}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }

    #[test]
    fn test_complex_compare() {
        let options = PythonOptions::default();
        let result = crate::parse("1 < a > 6", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //info!("{}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }
}
