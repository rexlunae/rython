use proc_macro2::TokenStream;
use pyo3::{Borrowed, PyAny, PyResult, FromPyObject, prelude::PyAnyMethods};
use quote::quote;

use crate::{extraction_failure, CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
//#[pyo3(transparent)]
pub struct Await {
    pub value: Box<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Await {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = ob.getattr("value").map_err(|e| extraction_failure("Await.value", &ob, e))?;
        Ok(Await {
            value: Box::new(value.extract().map_err(|e| extraction_failure("Await.value", &ob, e))?),
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
        let value = self.value.to_rust(_ctx, _options, _symbols)?;
        // A call to a user async function lowers to `f(...)?` — but the `?`
        // must apply to the awaited Result, not the future: reorder to
        // `f(...).await?`.
        let rendered = value.to_string();
        if let Some(inner) = rendered.trim_end().strip_suffix('?') {
            let inner: TokenStream = inner
                .parse()
                .map_err(|e| format!("re-parsing awaited call: {:?}", e))?;
            return Ok(quote!(#inner.await?));
        }
        Ok(quote!(#value.await))
    }
}
