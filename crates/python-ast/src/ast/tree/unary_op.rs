use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes, UnaryOpNotYetImplemented,
    dump, err_from, extraction_failure,
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
    pub op: Ops,
    pub operand: Box<ExprType>,
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
        // A BOXED-PyValue operand (`-e.partial` — urllib3's response.py
        // _error_catcher, where `e.partial` is an exception attribute
        // boxed to None by the dynamic-attribute divergence): Rust has no
        // unary minus on PyValue — USub routes through py_neg (a TypeError
        // panic for unmodeled operands, matching the PySub contract).
        // Computed before `options`/`symbols` are moved below.
        let operand_is_boxed = matches!(
            crate::infer_type(&self.operand, &options, &symbols),
            crate::TypeInfo::PyValue | crate::TypeInfo::PyObject
        );
        let operand = self.operand.clone().to_rust(ctx, options, symbols)?;
        match self.op {
            // `~x` is Rust's bitwise complement, but `not x` is a
            // TRUTHINESS test: `not 5` is False, where `!5i64` is -6.
            Ops::Invert => Ok(quote!(!#operand)),
            Ops::Not => Ok(quote!(!(#operand).is_truthy())),
            // Rust has no unary plus; Python's `+x` is the identity for
            // numbers, so emit the operand alone (parenthesized).
            Ops::UAdd => Ok(quote!((#operand))),
            Ops::USub => {
                if operand_is_boxed {
                    Ok(quote!((#operand).py_neg()))
                } else {
                    Ok(quote!(-#operand))
                }
            }
            _ => Err(err_from(UnaryOpNotYetImplemented(self)).into()),
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
