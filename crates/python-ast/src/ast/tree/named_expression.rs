use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes};

/// A keyword argument, gnerally used in function calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NamedExpr {
    pub left: Box<ExprType>,
    pub right: Box<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for NamedExpr {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // ast.NamedExpr stores its operands as `target` and `value`.
        let left = ob.getattr("target")?.extract::<ExprType>()?;
        let right = ob.getattr("value")?.extract::<ExprType>()?;
        Ok(NamedExpr {
            left: Box::new(left),
            right: Box::new(right),
        })
    }
}

impl CodeGen for NamedExpr {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // `(x := value)` — the walrus: bind the target, then evaluate to
        // it, as a block so the assignment is legal in expression
        // position (`if (seek := getattr(...)) is not None:`).
        //
        // When the target is HOISTED (a function/module scope recorded it
        // as a store — scope.rs), the walrus renders as a STORE into the
        // hoisted binding so the value stays visible to the enclosing
        // scope (`if data_to_send := conn.data_to_send():` reads
        // data_to_send in the if body — urllib3's http2). In a scope that
        // did not hoist it, the old block-let form is kept (a local).
        let target = self
            .left
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        // Hoisted check uses the PLAIN python name (the rendered form may
        // be keyword-escaped, e.g. `r#match`).
        let hoisted = match self.left.as_ref() {
            ExprType::Name(n) => options.hoisted_names.contains(&n.id),
            _ => false,
        };
        let right = self.right.clone().to_rust(ctx, options, symbols)?;
        Ok(if hoisted {
            quote!({ #target = #right; #target })
        } else {
            quote!({ let #target = #right; #target })
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Constant, ExprType, Name};
    use litrs::*;

    #[test]
    fn test_named_expression() {
        let named_expression = NamedExpr {
            left: Box::new(ExprType::Name(Name {
                id: "a".to_string(),
            })),
            right: Box::new(ExprType::Constant(Constant(Some(Literal::Integer(
                IntegerLit::parse("1".to_string()).unwrap(),
            ))))),
        };
        let rust = named_expression
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                PythonOptions::default(),
                SymbolTableScopes::new(),
            )
            .unwrap();
        assert_eq!(rust.to_string(), "{ let a = 1 ; a }");
    }
}
