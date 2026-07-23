use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
    PyAttributeExtractor,
};

/// Starred expression for unpacking (*args)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Starred {
    /// The expression being unpacked
    pub value: Box<ExprType>,
    /// Context (Load, Store, etc.) - not used in Rust generation
    pub ctx: Option<String>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Starred {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the value being starred
        let value = ob.extract_attr_with_context("value", "starred expression value")?;
        let value: ExprType = value.extract()?;
        
        // Extract context (Load, Store, etc.) - optional
        let ctx = ob.getattr("ctx").ok().and_then(|ctx_obj| {
            ctx_obj.get_type().name().ok().and_then(|name| name.extract().ok())
        });
        
        Ok(Starred {
            value: Box::new(value),
            ctx,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for Starred {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for Starred {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        (*self.value).clone().find_symbols(symbols)
    }

    fn to_rust(
        self,
        _ctx: Self::Context,
        _options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Emitting the bare value would pass the whole collection as ONE
        // argument/element — silently different from unpacking it.
        Err(
            "starred unpacking (`*expr`) is not supported yet: it would \
             silently pass the collection as a single value instead of \
             spreading its elements. Spell the elements out explicitly."
                .to_string()
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests will likely fail until full starred expression support is added
    // create_parse_test!(test_starred_args, "*args", "test.py");
    // create_parse_test!(test_starred_in_call, "func(*args)", "test.py");
}