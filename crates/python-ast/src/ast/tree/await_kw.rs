use proc_macro2::TokenStream;
use pyo3::{Borrowed, PyAny, PyResult, FromPyObject, prelude::PyAnyMethods};
use quote::quote;

use crate::{CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
//#[pyo3(transparent)]
pub struct Await {
    pub value: Box<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Await {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = ob.getattr("value").expect("Await.value");
        Ok(Await {
            value: Box::new(value.extract().expect("Await.value")),
        })
    }
}

impl<'a> CodeGen for Await {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        _ctx: Self::Context,
        _options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let value = self
            .value
            .to_rust(_ctx, _options, _symbols)
            .expect("Failed to convert async function to rust");
        Ok(quote!(#value.await))
    }
}
