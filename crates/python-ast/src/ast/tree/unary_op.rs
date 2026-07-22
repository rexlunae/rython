use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{extraction_failure,     dump, err_from, CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    UnaryOpNotYetImplemented,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Ops {
    Invert,
    Not,
    UAdd,
    USub,

    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnaryOp {
    op: Ops,
    operand: Box<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for UnaryOp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();

        tracing::debug!("ob: {}", dump(&ob, None)?);
        let op = ob
            .as_unbound()
            .getattr(py, "op")
            .map_err(|e| extraction_failure("unary operator", &ob, e))?;

        let bound_op = op.bind(py);
        let op_type = bound_op
            .get_type()
            .name()
            .map_err(|e| extraction_failure("unary operator type", &ob, e))?;

        let operand = ob
            .as_unbound()
            .getattr(py, "operand")
            .map_err(|e| extraction_failure("unary operand", &ob, e))?;

        let op = match op_type.extract::<String>()?.as_str() {
            "Invert" => Ops::Invert,
            "Not" => Ops::Not,
            "UAdd" => Ops::UAdd,
            "USub" => Ops::USub,
            _ => {
                tracing::debug!("{:?}", op);
                Ops::Unknown
            }
        };

        tracing::debug!("operand: {}", dump(&operand.bind(py), None)?);
        let bound_op = operand.bind(py);
        let operand = ExprType::extract(bound_op.as_borrowed())
            .map_err(|e| extraction_failure("unary operator operand", &ob, e))?;

        return Ok(UnaryOp {
            op: op,
            operand: Box::new(operand),
        });
    }
}

impl CodeGen for UnaryOp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let operand = self.operand.clone().to_rust(ctx, options, symbols)?;
        match self.op {
            Ops::Invert | Ops::Not => Ok(quote!(!#operand)),
            // Rust has no unary plus; Python's `+x` is the identity for
            // numbers, so emit the operand alone (parenthesized).
            Ops::UAdd => Ok(quote!((#operand))),
            Ops::USub => Ok(quote!(-#operand)),
            _ => Err(err_from(UnaryOpNotYetImplemented(self)).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not() {
        let options = PythonOptions::default();
        let result = crate::parse("not True", "test").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //tracing::info!("{}", result);

        let code = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
        tracing::info!("module: {:?}", code);
    }
}
