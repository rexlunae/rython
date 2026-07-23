use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, extraction_failure, err_from, BoolOpNotYetImplemented, CodeGen, CodeGenContext, ExprType,
    PythonOptions, SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BoolOps {
    And,
    Or,
    Unknown,
}

impl<'a, 'py> FromPyObject<'a, 'py> for BoolOps {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let op_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("boolean operator type", &ob, e))?;

        let op_type_str: String = op_type.extract()?;
        let op = match op_type_str.as_str() {
            "And" => BoolOps::And,
            "Or" => BoolOps::Or,
            _ => {
                tracing::debug!("Found unknown BoolOp {:?}", op_type_str);
                BoolOps::Unknown
            }
        };

        Ok(op)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoolOp {
    pub op: BoolOps,
    /// All operands: Python collapses `a and b and c` into one BoolOp node
    /// with three values.
    pub values: Vec<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for BoolOp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("ob: {}", dump(&ob, None)?);
        let op = ob.getattr("op").map_err(|e| extraction_failure("op", &ob, e))?;

        let op_type = op
            .get_type()
            .name()
            .map_err(|e| extraction_failure("boolean operator type", &ob, e))?;

        let values = ob.getattr("values").map_err(|e| extraction_failure("values", &ob, e))?;

        tracing::debug!("BoolOps values: {}", dump(&values, None)?);

        let values: Vec<ExprType> = values.extract().map_err(|e| extraction_failure("getting values from BoolOp", &ob, e))?;

        let op_type_str: String = op_type.extract()?;
        let op = match op_type_str.as_str() {
            "And" => BoolOps::And,
            "Or" => BoolOps::Or,

            _ => {
                tracing::debug!("Found unknown BoolOp {:?}", op);
                BoolOps::Unknown
            }
        };

        tracing::debug!("values: {:?}, op: {:?}/{:?}", values, op_type, op);

        return Ok(BoolOp { op, values });
    }
}

impl<'a> CodeGen for BoolOp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Python's boolean operators return operands, not booleans; for now we
        // approximate with Rust's short-circuiting operators, folding every
        // operand (a BoolOp node can carry more than two).
        let mut rendered = Vec::new();
        for value in self.values.clone() {
            rendered.push(value.to_rust(ctx.clone(), options.clone(), symbols.clone())?);
        }

        match self.op {
            BoolOps::Or => {
                // Special case for a trailing `or None`: drop it to avoid the
                // type mismatch with `|| None`.
                if let Some(last) = rendered.last() {
                    if last.to_string().trim() == "None" && rendered.len() == 2 {
                        let first = &rendered[0];
                        return Ok(quote!(#first));
                    }
                }
                Ok(quote!(#((#rendered))||*))
            }
            BoolOps::And => Ok(quote!(#((#rendered))&&*)),

            _ => Err(err_from(BoolOpNotYetImplemented(self)).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and() {
        let options = PythonOptions::default();
        let result = crate::parse("1 and 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //tracing::info!("{}", result.to_rust().unwrap());

        let code = result
            .to_rust(
                CodeGenContext::Module("test_case".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
        tracing::info!("module: {:?}", code);
    }

    #[test]
    fn test_or() {
        let options = PythonOptions::default();
        let result = crate::parse("1 or 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //tracing::info!("{}", result);

        let code = result
            .to_rust(
                CodeGenContext::Module("test_case".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
        tracing::info!("module: {:?}", code);
    }
}
