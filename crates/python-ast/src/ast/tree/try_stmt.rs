use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, Statement, SymbolTableScopes,
    extract_list,
};

/// Try statement (try/except/else/finally)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Try {
    /// The main body of the try block
    pub body: Vec<Statement>,
    /// Exception handlers (except clauses)
    pub handlers: Vec<ExceptHandler>,
    /// Optional else clause body (executed when no exception occurs)
    pub orelse: Vec<Statement>,
    /// Optional finally clause body (always executed)
    pub finalbody: Vec<Statement>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Exception handler (except clause)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExceptHandler {
    /// The exception type to catch (None means catch all)
    pub exception_type: Option<ExprType>,
    /// Variable name to bind the exception to (optional)
    pub name: Option<String>,
    /// Body of the except clause
    pub body: Vec<Statement>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Try {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract body
        let body: Vec<Statement> = extract_list(&ob, "body", "try body")?;
        
        // Extract handlers
        let handlers: Vec<ExceptHandler> = extract_list(&ob, "handlers", "try handlers")?;
        
        // Extract orelse (optional)
        let orelse: Vec<Statement> = extract_list(&ob, "orelse", "try orelse").unwrap_or_default();
        
        // Extract finalbody (optional)
        let finalbody: Vec<Statement> = extract_list(&ob, "finalbody", "try finalbody").unwrap_or_default();
        
        Ok(Try {
            body, 
            handlers,
            orelse,
            finalbody,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ExceptHandler {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract exception type (optional)
        let exception_type: Option<ExprType> = if let Ok(type_attr) = ob.getattr("type") {
            if type_attr.is_none() {
                None
            } else {
                Some(type_attr.extract()?)
            }
        } else {
            None
        };
        
        // Extract name (optional)
        let name: Option<String> = if let Ok(name_attr) = ob.getattr("name") {
            if name_attr.is_none() {
                None
            } else {
                Some(name_attr.extract()?)
            }
        } else {
            None
        };
        
        // Extract body
        let body: Vec<Statement> = extract_list(&ob, "body", "except handler body")?;
        
        Ok(ExceptHandler {
            exception_type,
            name,
            body,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for Try {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for ExceptHandler {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for Try {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process body, handlers, orelse, and finalbody
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        let symbols = self.handlers.into_iter().fold(symbols, |acc, handler| {
            let symbols = handler.body.into_iter().fold(acc, |acc, stmt| stmt.find_symbols(acc));
            if let Some(exception_type) = handler.exception_type {
                exception_type.find_symbols(symbols)
            } else {
                symbols
            }
        });
        let symbols = self.orelse.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        self.finalbody.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // The try body runs inside an immediately-invoked closure returning
        // Result<(), PyException>; `raise` (and failed `assert`) inside it
        // lower to `return Err(...)`.
        let body_ctx = CodeGenContext::TryBlock {
            parent: Box::new(ctx.clone()),
        };
        let try_body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
            .body
            .into_iter()
            .map(|stmt| stmt.to_rust(body_ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let try_body_tokens = try_body_tokens?;

        // Handler bodies run outside the closure (their exceptions are not
        // caught by this try), with the caught exception in scope.
        let handler_ctx = CodeGenContext::ExceptHandler {
            parent: Box::new(ctx.clone()),
        };
        let mut arms: Vec<TokenStream> = Vec::new();
        let mut has_catch_all = false;
        for handler in self.handlers {
            let guard = match &handler.exception_type {
                None => None,
                Some(t) => exception_match_guard(t)?,
            };
            let bind = match &handler.name {
                Some(name) => {
                    let ident = crate::safe_ident(name);
                    quote! {
                        #[allow(unused_variables, unused_mut)]
                        let mut #ident = __rython_exc.clone();
                    }
                }
                None => quote!(),
            };
            let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = handler
                .body
                .into_iter()
                .map(|stmt| stmt.to_rust(handler_ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let body_tokens = body_tokens?;
            match guard {
                Some(g) => arms.push(quote! {
                    Err(__rython_exc) if #g => { #bind #(#body_tokens;)* }
                }),
                None => {
                    has_catch_all = true;
                    arms.push(quote! {
                        Err(__rython_exc) => { #bind #(#body_tokens;)* }
                    });
                    break; // later handlers are unreachable, as in Python
                }
            }
        }

        // Else clause: runs only when the body completed without raising;
        // its own exceptions are not caught by this try's handlers.
        let else_tokens = if !self.orelse.is_empty() {
            let else_body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .orelse
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let else_body_tokens = else_body_tokens?;
            quote! { #(#else_body_tokens;)* }
        } else {
            quote!()
        };

        let finally_tokens = if !self.finalbody.is_empty() {
            let finally_body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .finalbody
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let finally_body_tokens = finally_body_tokens?;
            quote! { #(#finally_body_tokens;)* }
        } else {
            quote!()
        };

        // An exception no handler matched propagates: to the enclosing try's
        // closure when there is one, otherwise it aborts like an uncaught
        // Python exception. The finally body still runs first.
        if !has_catch_all {
            let reraise = if ctx.in_try_block() {
                quote!(return Err(__rython_exc);)
            } else {
                quote!(panic!("{}", __rython_exc);)
            };
            arms.push(quote! {
                Err(__rython_exc) => { #finally_tokens #reraise }
            });
        }

        Ok(quote! {
            {
                #[allow(unreachable_code)]
                let __rython_try_result: std::result::Result<(), PyException> = (|| {
                    #(#try_body_tokens;)*
                    Ok(())
                })();
                match __rython_try_result {
                    Ok(()) => { #else_tokens }
                    #(#arms)*
                }
                #finally_tokens
            }
        })
    }
}

/// The match guard testing whether the caught exception matches an except
/// clause's type expression: a name (`except ValueError`), a dotted name
/// (`except os.error` — matched by its final attribute), or a tuple of
/// either (`except (ValueError, TypeError)`).
fn exception_match_guard(
    exception_type: &ExprType,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    match exception_type {
        ExprType::Name(name) => {
            let n = &name.id;
            Ok(Some(quote!(__rython_exc.matches(#n))))
        }
        ExprType::Attribute(attr) => {
            let n = &attr.attr;
            Ok(Some(quote!(__rython_exc.matches(#n))))
        }
        ExprType::Tuple(tuple) => {
            let mut guards = Vec::new();
            for elt in &tuple.elts {
                match exception_match_guard(elt)? {
                    Some(g) => guards.push(g),
                    None => return Ok(None),
                }
            }
            if guards.is_empty() {
                Ok(None)
            } else {
                Ok(Some(quote!(#(#guards)||*)))
            }
        }
        other => Err(format!(
            "unsupported exception type in except clause: {:?} (use a name, \
             dotted name, or tuple of names)",
            other
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_try, "try:\n    pass\nexcept:\n    pass", "test.py");
}