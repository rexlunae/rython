use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
};

/// Yield expression (yield value)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Yield {
    /// The value being yielded (optional)
    pub value: Option<Box<ExprType>>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Yield from expression (yield from iterable)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct YieldFrom {
    /// The iterable being yielded from
    pub value: Box<ExprType>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Yield {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract value (optional)
        let value: Option<Box<ExprType>> = if let Ok(value_attr) = ob.getattr("value") {
            if value_attr.is_none() {
                None
            } else {
                Some(Box::new(value_attr.extract()?))
            }
        } else {
            None
        };
        
        Ok(Yield {
            value,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for YieldFrom {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract value
        let value: ExprType = ob.getattr("value")?.extract()?;
        
        Ok(YieldFrom {
            value: Box::new(value),
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for Yield {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for YieldFrom {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for Yield {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        if let Some(value) = self.value {
            (*value).find_symbols(symbols)
        } else {
            symbols
        }
    }

    fn to_rust(
        self,
        _ctx: Self::Context,
        _options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A generator silently becoming a plain function is exactly the kind
        // of divergence that must fail loudly instead.
        Err(
            "generators (`yield`) are not supported yet: the function would \
             silently evaluate a single value instead of producing a \
             generator. Rewrite it to build and return a list."
                .to_string()
                .into(),
        )
    }
}

impl CodeGen for YieldFrom {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        (*self.value).find_symbols(symbols)
    }

    fn to_rust(
        self,
        _ctx: Self::Context,
        _options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        Err(
            "generators (`yield from`) are not supported yet: the function \
             would silently evaluate the iterable once instead of delegating \
             to it. Rewrite it to build and return a list."
                .to_string()
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_yield, "def gen(): yield 42", "test.py");
}