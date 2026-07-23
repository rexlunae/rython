use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
};

/// Raise statement (raise [exception [from cause]])
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Raise {
    /// The exception to raise (optional - bare raise re-raises current exception)
    pub exc: Option<ExprType>,
    /// The cause of the exception (optional - used with 'from' clause)
    pub cause: Option<ExprType>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Raise {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract exc (optional)
        let exc: Option<ExprType> = if let Ok(exc_attr) = ob.getattr("exc") {
            if exc_attr.is_none() {
                None
            } else {
                Some(exc_attr.extract()?)
            }
        } else {
            None
        };
        
        // Extract cause (optional)
        let cause: Option<ExprType> = if let Ok(cause_attr) = ob.getattr("cause") {
            if cause_attr.is_none() {
                None
            } else {
                Some(cause_attr.extract()?)
            }
        } else {
            None
        };
        
        Ok(Raise {
            exc,
            cause,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for Raise {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for Raise {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let symbols = if let Some(exc) = self.exc {
            exc.find_symbols(symbols)
        } else {
            symbols
        };
        
        if let Some(cause) = self.cause {
            cause.find_symbols(symbols)
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
        let exc_tokens = match self.exc {
            Some(exc) => {
                let mut tokens =
                    exception_value(&exc, ctx.clone(), options.clone(), symbols.clone())?;
                if let Some(cause) = self.cause {
                    // `raise X from Y`: keep the cause visible in the message
                    // rather than dropping it.
                    let cause_tokens = cause.to_rust(ctx.clone(), options, symbols)?;
                    tokens = quote! {
                        {
                            let mut __rython_raised = #tokens;
                            __rython_raised.message =
                                format!("{} [from {}]", __rython_raised.message, #cause_tokens);
                            __rython_raised
                        }
                    };
                }
                tokens
            }
            None => {
                // Bare `raise` re-raises the exception the enclosing except
                // handler caught (a runtime error outside a handler, as in
                // Python).
                if !ctx.in_except_handler() {
                    return Err(
                        "bare `raise` outside an except handler has no exception to re-raise"
                            .to_string()
                            .into(),
                    );
                }
                quote!(__rython_exc.clone())
            }
        };

        // Functions return Result<T, PyException>, so raising is returning
        // Err: inside a try block it returns out of the block's Result
        // closure to be caught by the handlers, and anywhere else it
        // propagates out of the function, as in Python.
        Ok(quote!(return Err(#exc_tokens)))
    }
}

/// Names that look like Python exception classes, so `raise Name` /
/// `raise Name(...)` can construct a PyException carrying that class name.
/// Anything else is treated as an expression already producing a
/// PyException value (e.g. a variable bound by `except ... as e`).
fn is_exception_class_name(name: &str) -> bool {
    matches!(
        name,
        "Exception"
            | "BaseException"
            | "ArithmeticError"
            | "AssertionError"
            | "AttributeError"
            | "BufferError"
            | "EOFError"
            | "FileExistsError"
            | "FileNotFoundError"
            | "FloatingPointError"
            | "ImportError"
            | "IndentationError"
            | "IndexError"
            | "InterruptedError"
            | "IsADirectoryError"
            | "KeyError"
            | "KeyboardInterrupt"
            | "LookupError"
            | "MemoryError"
            | "ModuleNotFoundError"
            | "NameError"
            | "NotADirectoryError"
            | "NotImplementedError"
            | "OSError"
            | "OverflowError"
            | "PermissionError"
            | "RecursionError"
            | "ReferenceError"
            | "RuntimeError"
            | "StopAsyncIteration"
            | "StopIteration"
            | "SyntaxError"
            | "SystemError"
            | "SystemExit"
            | "TabError"
            | "TimeoutError"
            | "TypeError"
            | "UnboundLocalError"
            | "UnicodeDecodeError"
            | "UnicodeEncodeError"
            | "UnicodeError"
            | "ValueError"
            | "ZeroDivisionError"
    ) || name.ends_with("Error")
        || name.ends_with("Exception")
        || name.ends_with("Warning")
}

/// Lower the raised expression to a PyException value: `Name(...)` and bare
/// `Name` forms that look like exception classes construct one carrying the
/// class name (so handlers can match on it); any other expression is
/// assumed to already be a PyException.
fn exception_value(
    exc: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    match exc {
        ExprType::Call(call) => {
            if let ExprType::Name(name) = call.func.as_ref() {
                if is_exception_class_name(&name.id) {
                    let kind = &name.id;
                    let msg = match call.args.len() {
                        0 => quote!(String::new()),
                        1 => {
                            let arg = call.args[0].clone().to_rust(ctx, options, symbols)?;
                            quote!(format!("{}", #arg))
                        }
                        _ => {
                            let args: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = call
                                .args
                                .iter()
                                .map(|a| {
                                    a.clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )
                                })
                                .collect();
                            let args = args?;
                            let fmt = vec!["{}"; args.len()].join(", ");
                            quote!(format!(#fmt, #(#args),*))
                        }
                    };
                    return Ok(quote!(PyException::new(#kind, #msg)));
                }
            }
            let tokens = exc.clone().to_rust(ctx, options, symbols)?;
            Ok(quote!(#tokens))
        }
        ExprType::Name(name) if is_exception_class_name(&name.id) => {
            let kind = &name.id;
            Ok(quote!(PyException::new(#kind, String::new())))
        }
        other => {
            let tokens = other.clone().to_rust(ctx, options, symbols)?;
            Ok(quote!(#tokens))
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_raise, "raise ValueError('error')", "test.py");
}