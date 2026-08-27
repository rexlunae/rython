use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use serde::{Deserialize, Serialize};

use crate::{CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes};
use quote::quote;

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
    fn lineno(&self) -> Option<usize> {
        self.lineno
    }
    fn col_offset(&self) -> Option<usize> {
        self.col_offset
    }
    fn end_lineno(&self) -> Option<usize> {
        self.end_lineno
    }
    fn end_col_offset(&self) -> Option<usize> {
        self.end_col_offset
    }
}

impl Node for YieldFrom {
    fn lineno(&self) -> Option<usize> {
        self.lineno
    }
    fn col_offset(&self) -> Option<usize> {
        self.col_offset
    }
    fn end_lineno(&self) -> Option<usize> {
        self.end_lineno
    }
    fn end_col_offset(&self) -> Option<usize> {
        self.end_col_offset
    }
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
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Generator lowering: a `yield x` inside a function whose body the
        // function codegen rewrote becomes `push(x)` on the collector Vec
        // (issue #122-family — generators build-and-return a list).
        if let Some(collector) = options.generator_collector.as_ref() {
            let collector = proc_macro2::Ident::new(collector, proc_macro2::Span::call_site());
            let boxes = options.generator_boxes;
            let value = match self.value.as_ref() {
                Some(v) => v.clone().to_rust(ctx, options, symbols)?,
                None if boxes => {
                    // A bare `yield` (the @contextmanager shape) yields
                    // None — the boxed None in a PyValue collector.
                    return Ok(quote!(#collector . push (stdpython::PyValue::None_)));
                }
                None => return Ok(quote!(#collector)),
            };
            if boxes {
                return Ok(quote!(#collector . push (stdpython::PyValue::from(#value))));
            }
            return Ok(quote!(#collector . push (#value)));
        }
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
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Generator lowering: `yield from xs` extends the collector Vec.
        // Outside a generator (a `__iter__` that is ONLY `yield from xs`,
        // charset_normalizer's CharsetMatches) it lowers to returning the
        // collection directly.
        if let Some(collector) = options.generator_collector.as_ref() {
            let collector = proc_macro2::Ident::new(collector, proc_macro2::Span::call_site());
            let boxes = options.generator_boxes;
            let value = self.value.to_rust(ctx, options.clone(), symbols)?;
            if boxes {
                // A boxed collector: each yielded-from element boxes too
                // (a concrete inner Vec fails Into<PyValue> loudly when
                // the element cannot box).
                return Ok(quote!(#collector . extend (
                    (#value).into_iter().map(stdpython::PyValue::from)
                )));
            }
            return Ok(quote!(#collector . extend (#value)));
        }
        let value = self.value.to_rust(ctx, options, symbols)?;
        Ok(proc_macro2::TokenStream::from_iter(
            std::iter::once(proc_macro2::TokenTree::Ident(proc_macro2::Ident::new(
                "return",
                proc_macro2::Span::call_site(),
            )))
            .chain(value.into_iter()),
        ))
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_yield, "def gen(): yield 42", "test.py");
}
