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
        let body_has_import = try_body_contains_import(&self.body);
        let symbols = self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc));
        let symbols = self.handlers.into_iter().fold(symbols, |acc, handler| {
            // A bare `except ImportError:` handler is DROPPED at render
            // time (rython's imports are static — the fallback can never
            // run): its body must not register symbols either. urllib3's
            // ssl_.py has `try: from ssl import PROTOCOL_TLS ... except
            // ImportError: PROTOCOL_TLS = 2` — if the fallback's Assign
            // registered, it would shadow the try body's ImportFrom and
            // the name would render as a bare value instead of the
            // external-import boxed None. The ImportError-family TUPLE
            // spelling drops the same way for an import-attempt body.
            if is_bare_import_error(&handler.exception_type)
                && (matches!(
                    handler.exception_type,
                    Some(ExprType::Name(_))
                ) || body_has_import)
            {
                return acc;
            }
            // The except-bound name (`except IncompleteRead as e:`) is a
            // runtime PyException object; mark it so attribute reads on it
            // can lower to the boxed None (dynamic-attribute divergence)
            // instead of emitting a field that does not exist.
            let mut acc = acc;
            if let Some(name) = &handler.name {
                // The caught exception CLASS (the except clause's name),
                // when it is a plain class name — the modeled-field reads
                // resolve through it (round 99).
                let caught = match &handler.exception_type {
                    Some(ExprType::Name(n)) => Some(n.id.clone()),
                    _ => None,
                };
                acc.insert(name.clone(), crate::SymbolTableNode::ExceptBinding(caught));
            }
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
        // is a PyFlow carrying the returned value out (Return), a
        // break/continue signal, or normal completion (Normal).
        let has_return = crate::body_contains_function_return(&self.body);
        // A `break`/`continue` in the try body targets a loop OUTSIDE the
        // body's closure, so it cannot be emitted as a Rust jump — it is
        // threaded out as a PyFlow signal and replayed below.
        let body_escapes = crate::body_breaks_outward(&self.body);
        // Handler and else bodies run inline UNLESS there is a finally
        // clause, which wraps them in their own closure; a break there
        // would escape that closure with no signal path back. Refuse at
        // conversion time rather than emit Rust that cannot compile.
        if !self.finalbody.is_empty() {
            let where_ = if self.handlers.iter().any(|h| crate::body_breaks_outward(&h.body)) {
                Some("except handler")
            } else if crate::body_breaks_outward(&self.orelse) {
                Some("else clause")
            } else {
                None
            };
            if let Some(where_) = where_ {
                return Err(format!(
                    "`break`/`continue` in a try statement's {} is not supported when the \
                     statement also has a `finally` clause; move the loop control out of the \
                     handler, or drop the finally clause",
                    where_
                )
                .into());
            }
        }
        let body_for_guarantee = self.body.clone();
        let body_has_import = try_body_contains_import(&self.body);
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
        // closure. The carried value was already converted by the Return
        // statement inside the closure (round 81's `.into()` fires there,
        // where the value's inference is known).
        let break_return = if ctx.in_try_block() {
            quote!(return Ok(PyFlow::Return(__rython_ret));)
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
        // (static guard, dynamic value, bind, arm body) per handler. A
        // DYNAMIC handler (`except self._retryable_exceptions:` — round
        // 33) switches the whole statement to the lazy if-chain form
        // below; static-only statements keep the match-arm guards.
        let mut entries: Vec<(Option<TokenStream>, Option<TokenStream>, TokenStream, TokenStream)> =
            Vec::new();
        let mut any_dynamic = false;
        for handler in self.handlers {
            // A bare `except ImportError:` handler (`try: from tokenize
            // import detect_encoding / except ImportError:` — distlib's
            // compat.py): rython's imports are STATIC — a module either
            // exists in the crate/stdlib or it is assumed present — so the
            // fallback can never run and its body (often Python-2-era
            // compatibility code) would fail to convert. The handler is
            // dropped (the documented static-imports divergence). The
            // ImportError-family TUPLE spelling (`except (ImportError,
            // AttributeError): ssl = None` — urllib3's connection.py)
            // drops the same way, but only when the try body is actually
            // an import attempt.
            if is_bare_import_error(&handler.exception_type)
                && (matches!(
                    handler.exception_type,
                    Some(ExprType::Name(_))
                ) || body_has_import)
            {
                options.definition_warnings.borrow_mut().push(
                    "`except ImportError:` handler is dropped: rython's \
                     imports are static, so the fallback can never run"
                        .to_string(),
                );
                continue;
            }
            let dynamic = match &handler.exception_type {
                None => None,
                Some(t) => dynamic_exception_value(t, &ctx, &options, &symbols)?,
            };
            if dynamic.is_some() {
                any_dynamic = true;
            }
            let guard = match &handler.exception_type {
                None => None,
                Some(t) if dynamic.is_none() => {
                    exception_match_guard(t, &symbols, &options)?
                }
                Some(_) => None,
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
            // The handler body renders with a PER-HANDLER symbol view:
            // the merged table's ExceptBinding for the bound name may
            // carry a LATER handler's class (both bank handlers bind
            // `as e` — find_symbols keeps the last insertion). Insert
            // THIS handler's class so the modeled-field reads resolve
            // (round 99).
            let mut handler_symbols = symbols.clone();
            if let Some(name) = &handler.name {
                let caught = match &handler.exception_type {
                    Some(ExprType::Name(n)) => Some(n.id.clone()),
                    _ => None,
                };
                handler_symbols
                    .insert(name.clone(), crate::SymbolTableNode::ExceptBinding(caught));
            }
            let arm_body = lower_finally_guarded_body(
                handler.body,
                handler_ctx.clone(),
                &options,
                &handler_symbols,
                has_finally,
                &finally_tokens,
                &break_return,
                "handler body terminates on every path",
            )?;
            if guard.is_none() && dynamic.is_none() {
                has_catch_all = true;
                entries.push((None, None, bind, arm_body));
                break; // later handlers are unreachable, as in Python
            }
            entries.push((guard, dynamic, bind, arm_body));
        }

        // A DYNAMIC handler (a boxed, runtime-valued except type —
        // `except self._retryable_exceptions:`): the whole handler list
        // becomes a lazy if-chain inside ONE Err arm, so each guard is
        // evaluated in order only once an exception propagates — and a
        // dynamic guard's `?` (its TypeError — a non-catchable except
        // value, CPython's "catching classes that do not inherit from
        // BaseException is not allowed") is a real raise exactly when
        // CPython raises it. Match-arm guards cannot thread `?`, hence
        // the switch.
        if any_dynamic {
            let mut chain = quote!(#finally_tokens return Err(__rython_exc););
            for (guard, dynamic, bind, body) in entries.into_iter().rev() {
                match (guard, dynamic) {
                    (_, Some(value)) => {
                        chain = quote! {
                            if __rython_exc.matches_value(&(#value))? { #bind #body }
                            else { #chain }
                        };
                    }
                    (Some(g), None) => {
                        chain = quote! {
                            if #g { #bind #body }
                            else { #chain }
                        };
                    }
                    // The catch-all handler (the loop broke on it): it is
                    // the innermost else — nothing falls through it.
                    (None, None) => {
                        chain = quote!({ #bind #body });
                    }
                }
            }
            arms.push(quote! {
                Err(__rython_exc) => { #chain }
            });
        } else {
            // Static-only handlers: each arm carries its own guard; a
            // catch-all handler (the loop broke on it) is the unguarded
            // arm.
            for (guard, dynamic, bind, body) in entries {
                debug_assert!(dynamic.is_none());
                match guard {
                    Some(g) => arms.push(quote! {
                        Err(__rython_exc) if #g => { #bind #body }
                    }),
                    None => arms.push(quote! {
                        Err(__rython_exc) => { #bind #body }
                    }),
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
        // function, as in Python. The finally body still runs first. (The
        // dynamic if-chain carries its own fall-through as the innermost
        // else, so nothing is added here for it.)
        if !has_catch_all && !any_dynamic {
            arms.push(quote! {
                Err(__rython_exc) => { #finally_tokens return Err(__rython_exc); }
            });
        }

        if has_return || body_escapes {
            // The Return arm carries a value, so the parameter needs a
            // type; a body with no `return` never constructs one, so pin
            // it to () rather than leave it uninferable.
            let flow_type = if has_return {
                quote!(PyFlow<_>)
            } else {
                quote!(PyFlow<()>)
            };
            let return_arm = if has_return {
                quote! {
                    Ok(PyFlow::Return(__rython_ret)) => {
                        #finally_tokens
                        #break_return
                    }
                }
            } else {
                quote! { Ok(PyFlow::Return(_)) => unreachable!("try body has no return"), }
            };
            // Replay a signalled break/continue at the try statement's own
            // position, AFTER the finally clause — Python's ordering. If
            // this try is itself inside another try's closure, the signal
            // is re-raised outward instead of becoming a Rust loop jump.
            let (break_arm, continue_arm) = if body_escapes {
                let replay_break = if ctx.break_crosses_try_closure() {
                    quote!(return Ok(PyFlow::Break);)
                } else if ctx.break_target_has_else() {
                    quote!({ __rython_broke = true; break; })
                } else {
                    quote!(break;)
                };
                let replay_continue = if ctx.break_crosses_try_closure() {
                    quote!(return Ok(PyFlow::Continue);)
                } else {
                    quote!(continue;)
                };
                (
                    quote! { Ok(PyFlow::Break) => { #finally_tokens #replay_break } },
                    quote! { Ok(PyFlow::Continue) => { #finally_tokens #replay_continue } },
                )
            } else {
                (
                    quote! { Ok(PyFlow::Break) => unreachable!("try body has no break"), },
                    quote! { Ok(PyFlow::Continue) => unreachable!("try body has no continue"), },
                )
            };
            Ok(quote! {
                {
                    #[allow(unreachable_code)]
                    let __rython_try_result: std::result::Result<
                        #flow_type,
                        PyException,
                    > = (|| {
                        #(#try_body_tokens;)*
                        Ok(PyFlow::Normal)
                    })();
                    match __rython_try_result {
                        #return_arm
                        #break_arm
                        #continue_arm
                        Ok(PyFlow::Normal) => { #ok_arm_body }
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
/// like the try body — so a `return` (threaded out as PyFlow::Return)
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
                PyFlow<_>,
                PyException,
            > = (|| {
                #(#tokens;)*
                Ok(PyFlow::Normal)
            })();
            match __rython_inner {
                Ok(PyFlow::Return(__rython_ret)) => {
                    #finally_tokens
                    #break_return
                }
                // A break/continue in a closure-wrapped handler or else
                // clause is rejected at conversion time, so these are
                // structurally unreachable.
                Ok(PyFlow::Break) => unreachable!("handler body has no break"),
                Ok(PyFlow::Continue) => unreachable!("handler body has no continue"),
                Ok(PyFlow::Normal) => { #completed_arm }
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
/// Whether an except clause is the dead fallback of an import attempt —
/// catching `ImportError` (urllib3's `except ImportError: brotli = None`)
/// or an ImportError-family tuple (`except (ImportError, AttributeError):
/// ssl = None` — connection.py, where a missing ssl module raises either).
/// Such handlers are dead under rython's static imports: the import either
/// resolves statically or the name is unmodeled, never raised at runtime.
/// The bare `ImportError` spelling is dropped unconditionally (the
/// established divergence); the TUPLE spelling only when the try body
/// actually contains an import (so a runtime `except AttributeError`
/// fallback elsewhere is never dropped).
pub(crate) fn is_bare_import_error(exception_type: &Option<ExprType>) -> bool {
    match exception_type {
        Some(ExprType::Name(n)) if n.id == "ImportError" => true,
        Some(ExprType::Tuple(t)) => {
            !t.elts.is_empty()
                && t.elts.iter().all(|e| matches!(e, ExprType::Name(n)
                    if matches!(n.id.as_str(), "ImportError" | "AttributeError")))
        }
        _ => false,
    }
}

/// Whether the try body's statements contain an import (the import-attempt
/// pattern that an ImportError-family handler guards).
pub(crate) fn try_body_contains_import(body: &[crate::Statement]) -> bool {
    body.iter().any(|s| {
        matches!(
            s.statement,
            crate::StatementType::Import(_) | crate::StatementType::ImportFrom(_)
        ) || matches!(&s.statement, crate::StatementType::Try(t)
            if try_body_contains_import(&t.body))
    })
}

/// A BOXED, runtime-valued exception type in an except clause
/// (`except self._retryable_exceptions:` — botocore's retryhandler,
/// round 33): the class-as-value model keeps the catchable name list in
/// a PyValue (a Str member, or a Tuple of Str members — lists box as
/// tuples), so the handler cannot use a static `matches` guard; it
/// lowers to the runtime's `matches_value`, evaluated lazily when the
/// exception propagates, exactly as CPython evaluates the except
/// expression then (a non-catchable value raises the TypeError there).
///
/// Statically-known class names — and module-root dotted aliases like
/// `socket.timeout`, which the static arm canonicalizes — stay on the
/// static path; anything else in except position stays a loud conversion
/// error.
fn dynamic_exception_value(
    exception_type: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    // A SELF-field whose type the class table boxes:
    // `self._retryable_exceptions` (assigned a boxed tuple, or None).
    // infer_type cannot see this — field reads infer as PyObject — so
    // consult the field table first.
    if let ExprType::Attribute(attr) = exception_type
        && let ExprType::Name(n) = attr.value.as_ref()
        && n.id == "self"
        && let Some(class_name) = ctx.enclosing_class_name()
        && let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(class_name)
        && let Ok(fields) = class.infer_fields(symbols, options)
        && let Some((_, ty)) = fields.iter().find(|(name, _)| *name == attr.attr)
    {
        let boxed = crate::ast::tree::type_ctx::type_contains_pyvalue(ty)
            || matches!(ty, crate::TypeInfo::PyObject | crate::TypeInfo::Option(_));
        if boxed {
            return exception_type
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())
                .map(Some);
        }
    }
    match crate::infer_type(Some(&ctx), exception_type, options, symbols) {
        crate::TypeInfo::PyValue => exception_type
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())
            .map(Some),
        _ => Ok(None),
    }
}

fn exception_match_guard(
    exception_type: &ExprType,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> Result<Option<TokenStream>, Box<dyn std::error::Error>> {
    match exception_type {
        ExprType::Name(name) => {
            // A stdlib exception ALIAS (`except SocketTimeout:` under
            // `from socket import timeout as SocketTimeout` — urllib3):
            // canonicalize to the builtin, matching the raise side
            // (issue #137).
            let n = crate::ast::tree::raise_stmt::imported_exception_alias(
                &name.id,
                symbols,
                Some(options),
            )
            .map(str::to_string)
            .unwrap_or_else(|| name.id.clone());
            // Round 52: a literal BUILTIN clause (`except ValueError:`)
            // lowers to the discriminant comparison — the class name is
            // a source literal and the runtime knows its variant and
            // ancestor slice statically. No string walk per clause.
            if let Some(ident) = crate::ast::tree::raise_stmt::builtin_exception_variant(&n) {
                let ident = crate::safe_ident(&ident);
                return Ok(Some(quote!(__rython_exc.matches_builtin(BuiltinException::#ident))));
            }
            Ok(Some(quote!(__rython_exc.matches(#n))))
        }
        ExprType::Attribute(attr) => {
            // `except socket.timeout:` — the dotted spelling of the same
            // stdlib alias canonicalizes identically.
            let n = match attr.value.as_ref() {
                ExprType::Name(m) => {
                    crate::ast::tree::raise_stmt::stdlib_exception_canonical(
                        &m.id, &attr.attr,
                    )
                    .map(str::to_string)
                    .unwrap_or_else(|| attr.attr.clone())
                }
                _ => attr.attr.clone(),
            };
            if let Some(ident) = crate::ast::tree::raise_stmt::builtin_exception_variant(&n) {
                let ident = crate::safe_ident(&ident);
                return Ok(Some(quote!(__rython_exc.matches_builtin(BuiltinException::#ident))));
            }
            Ok(Some(quote!(__rython_exc.matches(#n))))
        }
        ExprType::Tuple(tuple) => {
            let mut guards = Vec::new();
            for elt in &tuple.elts {
                match exception_match_guard(elt, symbols, options)? {
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