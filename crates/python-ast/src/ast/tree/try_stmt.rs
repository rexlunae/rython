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
        // The try body runs inside an immediately-invoked closure; `raise`
        // (and failed `assert`) inside it lower to `return Err(...)`. When
        // the body contains function-level returns, the closure's Ok value
        // is a ControlFlow carrying the returned value out (Break) or
        // marking normal completion (Continue).
        let has_return = crate::body_contains_function_return(&self.body);
        let body_for_guarantee = self.body.clone();
        let body_ctx = CodeGenContext::TryBlock {
            parent: Box::new(ctx.clone()),
        };
        let try_body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
            .body
            .into_iter()
            .map(|stmt| stmt.to_rust(body_ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let try_body_tokens = try_body_tokens?;

        // A return that broke out of any of the closures below runs the
        // finally body, then returns from the function — re-wrapped as
        // another Break when this try is itself inside an enclosing try's
        // closure.
        let break_return = if ctx.in_try_block() {
            quote!(return Ok(std::ops::ControlFlow::Break(__rython_ret));)
        } else {
            quote!(return Ok(__rython_ret);)
        };

        let has_finally = !self.finalbody.is_empty();
        let finally_tokens = if has_finally {
            let finally_body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .finalbody
                .clone()
                .into_iter()
                .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
                .collect();
            let finally_body_tokens = finally_body_tokens?;
            quote! { #(#finally_body_tokens;)* }
        } else {
            quote!()
        };

        // Handler bodies run outside the try closure (their exceptions are
        // not caught by this try), with the caught exception in scope. When
        // a finally clause exists, each handler body runs in its own
        // closure so a return or raise inside it still executes the finally
        // body before leaving the function, as Python requires.
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
            let arm_body = lower_finally_guarded_body(
                handler.body,
                handler_ctx.clone(),
                &options,
                &symbols,
                has_finally,
                &finally_tokens,
                &break_return,
                "handler body terminates on every path",
            )?;
            match guard {
                Some(g) => arms.push(quote! {
                    Err(__rython_exc) if #g => { #bind #arm_body }
                }),
                None => {
                    has_catch_all = true;
                    arms.push(quote! {
                        Err(__rython_exc) => { #bind #arm_body }
                    });
                    break; // later handlers are unreachable, as in Python
                }
            }
        }

        // Else clause: runs only when the body completed without raising;
        // its own exceptions are not caught by this try's handlers — but a
        // return or raise in it must still run the finally body first.
        let else_tokens = if !self.orelse.is_empty() {
            lower_finally_guarded_body(
                self.orelse,
                ctx.clone(),
                &options,
                &symbols,
                has_finally,
                &finally_tokens,
                &break_return,
                "else clause terminates on every path",
            )?
        } else {
            quote!()
        };

        // When the try body terminates on every path (return/raise), the
        // completed-normally arm is provably dead — mark it unreachable so
        // the surrounding function (which emits no fall-through tail when
        // all paths terminate) still typechecks.
        let ok_arm_body = if crate::guarantees_return(&body_for_guarantee) {
            quote!(unreachable!("try body terminates on every path"))
        } else {
            else_tokens
        };

        // An exception no handler matched propagates as an Err — to the
        // enclosing try's closure when there is one, otherwise out of the
        // function, as in Python. The finally body still runs first.
        if !has_catch_all {
            arms.push(quote! {
                Err(__rython_exc) => { #finally_tokens return Err(__rython_exc); }
            });
        }

        if has_return {
            Ok(quote! {
                {
                    #[allow(unreachable_code)]
                    let __rython_try_result: std::result::Result<
                        std::ops::ControlFlow<_>,
                        PyException,
                    > = (|| {
                        #(#try_body_tokens;)*
                        Ok(std::ops::ControlFlow::Continue(()))
                    })();
                    match __rython_try_result {
                        Ok(std::ops::ControlFlow::Break(__rython_ret)) => {
                            #finally_tokens
                            #break_return
                        }
                        Ok(std::ops::ControlFlow::Continue(())) => { #ok_arm_body }
                        #(#arms)*
                    }
                    #finally_tokens
                }
            })
        } else {
            Ok(quote! {
                {
                    #[allow(unreachable_code)]
                    let __rython_try_result: std::result::Result<(), PyException> = (|| {
                        #(#try_body_tokens;)*
                        Ok(())
                    })();
                    match __rython_try_result {
                        Ok(()) => { #ok_arm_body }
                        #(#arms)*
                    }
                    #finally_tokens
                }
            })
        }
    }
}

/// Lower an except-handler or else-clause body. Without a finally clause
/// the statements run inline. With one, the body runs in its own closure —
/// like the try body — so a `return` (threaded out as ControlFlow::Break)
/// or a raise (an Err) still executes the finally body before leaving the
/// function, as Python guarantees.
#[allow(clippy::too_many_arguments)]
fn lower_finally_guarded_body(
    body: Vec<Statement>,
    base_ctx: CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    has_finally: bool,
    finally_tokens: &TokenStream,
    break_return: &TokenStream,
    unreachable_note: &str,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if !has_finally {
        let tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = body
            .into_iter()
            .map(|stmt| stmt.to_rust(base_ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let tokens = tokens?;
        return Ok(quote! { #(#tokens;)* });
    }

    let guarantees = crate::guarantees_return(&body);
    let has_ret = crate::body_contains_function_return(&body);
    let inner_ctx = CodeGenContext::TryBlock {
        parent: Box::new(base_ctx),
    };
    let tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = body
        .into_iter()
        .map(|stmt| stmt.to_rust(inner_ctx.clone(), options.clone(), symbols.clone()))
        .collect();
    let tokens = tokens?;

    let completed_arm = if guarantees {
        quote!(unreachable!(#unreachable_note))
    } else {
        quote!()
    };

    if has_ret {
        Ok(quote! {
            #[allow(unreachable_code)]
            let __rython_inner: std::result::Result<
                std::ops::ControlFlow<_>,
                PyException,
            > = (|| {
                #(#tokens;)*
                Ok(std::ops::ControlFlow::Continue(()))
            })();
            match __rython_inner {
                Ok(std::ops::ControlFlow::Break(__rython_ret)) => {
                    #finally_tokens
                    #break_return
                }
                Ok(std::ops::ControlFlow::Continue(())) => { #completed_arm }
                Err(__rython_reraise) => {
                    #finally_tokens
                    return Err(__rython_reraise);
                }
            }
        })
    } else {
        Ok(quote! {
            #[allow(unreachable_code)]
            let __rython_inner: std::result::Result<(), PyException> = (|| {
                #(#tokens;)*
                Ok(())
            })();
            match __rython_inner {
                Ok(()) => { #completed_arm }
                Err(__rython_reraise) => {
                    #finally_tokens
                    return Err(__rython_reraise);
                }
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