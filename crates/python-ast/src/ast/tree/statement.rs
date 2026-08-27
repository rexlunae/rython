use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    Assign, AsyncFor, AsyncWith, AugAssign, Call, ClassDef, CodeGen, CodeGenContext, Expr,
    ExprType, For, FunctionDef, If, Import, ImportFrom, Node, PythonOptions, Raise,
    StatementNotYetImplemented, SymbolTableScopes, Try, While, With, dump, err_from,
    extraction_failure,
};

use tracing::debug;

use serde::{Deserialize, Serialize};

/// AST node types that can be used as a statement implement this type.
pub trait PyStatementTrait: Clone + PartialEq {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Statement {
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
    pub statement: StatementType,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Statement {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
            statement: StatementType::extract(ob)?,
        })
    }
}

impl Node for Statement {
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

impl CodeGen for Statement {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        self.statement.clone().find_symbols(symbols)
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let (lineno, col_offset) = (self.lineno, self.col_offset);
        let (end_lineno, end_col_offset) = (self.end_lineno, self.end_col_offset);
        let result = self
            .statement
            .clone()
            .to_rust(ctx, options, symbols)
            .map_err(|e| {
                let location = crate::SourceLocation::with_span(
                    "<module>",
                    lineno,
                    col_offset.map(|c| c + 1),
                    end_lineno,
                    end_col_offset,
                );
                Box::<dyn std::error::Error>::from(crate::codegen_error(
                    location,
                    crate::format_error_chain(e.as_ref()),
                    "",
                ))
            });
        result
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StatementType {
    AsyncFunctionDef(FunctionDef),
    Assert {
        test: Box<ExprType>,
        msg: Option<Box<ExprType>>,
    },
    Assign(Assign),
    AugAssign(AugAssign),
    Break,
    Continue,
    ClassDef(ClassDef),
    Call(Call),
    Pass,
    Return(Option<Expr>),
    Import(Import),
    ImportFrom(ImportFrom),
    Expr(Expr),
    FunctionDef(FunctionDef),
    If(If),
    For(For),
    While(While),
    Try(Try),
    AsyncWith(AsyncWith),
    AsyncFor(AsyncFor),
    Raise(Raise),
    With(With),
    /// `global a, b` — declares module-level names (issue #115). Reads of
    /// the names resolve to module statics; WRITES from a function are a
    /// loud error (rython has no mutable module state).
    Global(Vec<String>),
    /// `nonlocal a, b` — a nested-function binding directive (rich's
    /// traceback IPython hooks). rython's closures do not capture outer
    /// function scopes, so the declaration has no runtime effect — a
    /// no-op (the closure-capture divergence).
    Nonlocal(Vec<String>),
    /// A bare annotated declaration (`x: int` — no value). At module/class
    /// level this is a dataclass-style field declaration; inside functions
    /// it declares nothing at runtime (lowered as a no-op). Carried so
    /// `@dataclass` can synthesize `__init__` from the class body.
    AnnotatedName {
        name: String,
        annotation: ExprType,
    },
    /// `del xs[i]` / `del d[k]` — removes an element (issue #112). Index
    /// targets lower through py_pop; `del name` and `del a.b` are loud
    /// errors (unbinding is not representable in the value model).
    Delete(Vec<ExprType>),

    Unimplemented(String),
}

impl<'a, 'py> FromPyObject<'a, 'py> for StatementType {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let ob_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("statement type", &ob, e))?;

        debug!("statement...ob_type: {}...{}", ob_type, dump(&ob, Some(4))?);
        match ob_type.extract::<String>()?.as_str() {
            "AsyncFunctionDef" => Ok(StatementType::AsyncFunctionDef(
                FunctionDef::extract(ob)
                    .map_err(|e| extraction_failure("async function definition", &ob, e))?,
            )),
            "Assign" => {
                let assignment =
                    Assign::extract(ob).map_err(|e| extraction_failure("assignment", &ob, e))?;
                Ok(StatementType::Assign(assignment))
            }
            "AnnAssign" => {
                // An annotated assignment (`x: int = 5`) is an ordinary
                // assignment with a type annotation we carry on the Assign
                // node (so empty-container pinning can honor it); a bare
                // annotation (`x: int`) declares nothing at runtime — but
                // it IS a dataclass field declaration at class level, so it
                // is carried as AnnotatedName rather than dropped.
                let value = ob
                    .getattr("value")
                    .map_err(|e| extraction_failure("annotated assignment value", &ob, e))?;
                if value.is_none() {
                    let target = ob
                        .getattr("target")
                        .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?
                        .extract()
                        .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?;
                    let annotation = ob
                        .getattr("annotation")
                        .ok()
                        .filter(|a| !a.is_none())
                        .map(|a| a.extract())
                        .transpose()
                        .map_err(|e| {
                            extraction_failure("annotated assignment annotation", &ob, e)
                        })?;
                    if let (ExprType::Name(n), Some(annotation)) = (target, annotation) {
                        return Ok(StatementType::AnnotatedName {
                            name: n.id,
                            annotation,
                        });
                    }
                    return Ok(StatementType::Pass);
                }
                let target = ob
                    .getattr("target")
                    .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("annotated assignment target", &ob, e))?;
                let value = value
                    .extract()
                    .map_err(|e| extraction_failure("annotated assignment value", &ob, e))?;
                let annotation = ob
                    .getattr("annotation")
                    .ok()
                    .filter(|a| !a.is_none())
                    .map(|a| a.extract())
                    .transpose()
                    .map_err(|e| extraction_failure("annotated assignment annotation", &ob, e))?;
                Ok(StatementType::Assign(Assign {
                    targets: vec![target],
                    value,
                    type_comment: None,
                    annotation,
                }))
            }
            "AugAssign" => {
                let aug_assignment = AugAssign::extract(ob)
                    .map_err(|e| extraction_failure("augmented assignment", &ob, e))?;
                Ok(StatementType::AugAssign(aug_assignment))
            }
            "Assert" => {
                let test: ExprType = ob
                    .getattr("test")
                    .map_err(|e| extraction_failure("assert condition", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("assert condition", &ob, e))?;
                let msg: Option<Box<ExprType>> = match ob.getattr("msg") {
                    Ok(m) if !m.is_none() => Some(Box::new(
                        m.extract()
                            .map_err(|e| extraction_failure("assert message", &ob, e))?,
                    )),
                    _ => None,
                };
                Ok(StatementType::Assert {
                    test: Box::new(test),
                    msg,
                })
            }
            "Pass" => Ok(StatementType::Pass),
            "Call" => {
                let value = ob
                    .getattr("value")
                    .map_err(|e| extraction_failure("call statement value", &ob, e))?;
                let call = Call::extract(value.as_borrowed())
                    .map_err(|e| extraction_failure("call statement", &ob, e))?;
                debug!("call: {:?}", call);
                Ok(StatementType::Call(call))
            }
            "ClassDef" => Ok(StatementType::ClassDef(
                ClassDef::extract(ob)
                    .map_err(|e| extraction_failure("class definition", &ob, e))?,
            )),
            "Continue" => Ok(StatementType::Continue),
            "Break" => Ok(StatementType::Break),
            "FunctionDef" => Ok(StatementType::FunctionDef(
                FunctionDef::extract(ob)
                    .map_err(|e| extraction_failure("function definition", &ob, e))?,
            )),
            "Import" => Ok(StatementType::Import(
                Import::extract(ob).map_err(|e| extraction_failure("import", &ob, e))?,
            )),
            "ImportFrom" => Ok(StatementType::ImportFrom(
                ImportFrom::extract(ob).map_err(|e| extraction_failure("from-import", &ob, e))?,
            )),
            "Expr" => {
                let expr = ob
                    .extract()
                    .map_err(|e| extraction_failure("expression statement", &ob, e))?;
                Ok(StatementType::Expr(expr))
            }
            "Global" => {
                let names = ob
                    .getattr("names")
                    .map_err(|e| extraction_failure("global names", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("global names", &ob, e))?;
                Ok(StatementType::Global(names))
            }
            "Nonlocal" => {
                let names = ob
                    .getattr("names")
                    .map_err(|e| extraction_failure("nonlocal names", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("nonlocal names", &ob, e))?;
                Ok(StatementType::Nonlocal(names))
            }
            "Delete" => {
                let targets = ob
                    .getattr("targets")
                    .map_err(|e| extraction_failure("delete targets", &ob, e))?
                    .extract()
                    .map_err(|e| extraction_failure("delete targets", &ob, e))?;
                Ok(StatementType::Delete(targets))
            }
            "Return" => {
                tracing::debug!("return expression: {}", dump(&ob, None)?);
                // Extract the return value from the Return statement's 'value' field
                let return_value = if let Ok(value_attr) = ob.getattr("value") {
                    if value_attr.is_none() {
                        // Bare 'return' statement - create a NoneType Expr
                        Some(Expr {
                            value: crate::tree::ExprType::NoneType(crate::tree::Constant(None)),
                            ctx: None,
                            lineno: ob.lineno(),
                            col_offset: ob.col_offset(),
                            end_lineno: ob.end_lineno(),
                            end_col_offset: ob.end_col_offset(),
                        })
                    } else {
                        // Return with actual expression - extract as ExprType then wrap in Expr
                        let expr_value: crate::tree::ExprType = value_attr
                            .extract()
                            .map_err(|e| extraction_failure("return value", &ob, e))?;
                        Some(Expr {
                            value: expr_value,
                            ctx: None,
                            lineno: ob.lineno(),
                            col_offset: ob.col_offset(),
                            end_lineno: ob.end_lineno(),
                            end_col_offset: ob.end_col_offset(),
                        })
                    }
                } else {
                    None
                };
                Ok(StatementType::Return(return_value))
            }
            "If" => {
                let if_stmt =
                    If::extract(ob).map_err(|e| extraction_failure("if statement", &ob, e))?;
                Ok(StatementType::If(if_stmt))
            }
            "For" => {
                let for_stmt =
                    For::extract(ob).map_err(|e| extraction_failure("for loop", &ob, e))?;
                Ok(StatementType::For(for_stmt))
            }
            "While" => {
                let while_stmt =
                    While::extract(ob).map_err(|e| extraction_failure("while loop", &ob, e))?;
                Ok(StatementType::While(while_stmt))
            }
            "Try" => {
                let try_stmt =
                    Try::extract(ob).map_err(|e| extraction_failure("try statement", &ob, e))?;
                Ok(StatementType::Try(try_stmt))
            }
            "AsyncWith" => {
                let async_with_stmt = AsyncWith::extract(ob)
                    .map_err(|e| extraction_failure("async with statement", &ob, e))?;
                Ok(StatementType::AsyncWith(async_with_stmt))
            }
            "AsyncFor" => {
                let async_for_stmt = AsyncFor::extract(ob)
                    .map_err(|e| extraction_failure("async for loop", &ob, e))?;
                Ok(StatementType::AsyncFor(async_for_stmt))
            }
            "Raise" => {
                let raise_stmt = Raise::extract(ob)
                    .map_err(|e| extraction_failure("raise statement", &ob, e))?;
                Ok(StatementType::Raise(raise_stmt))
            }
            "With" => {
                let with_stmt =
                    With::extract(ob).map_err(|e| extraction_failure("with statement", &ob, e))?;
                Ok(StatementType::With(with_stmt))
            }
            other => Err(extraction_failure(
                "statement",
                &ob,
                format!("the `{}` statement is not yet supported by rython", other),
            )),
        }
    }
}

/// A Python `return` lowered for the current context: inside a try-block
/// closure it signals out via PyFlow so the try lowering can run the
/// finally body and re-return; elsewhere it returns Ok directly.
fn return_tokens(ctx: &CodeGenContext, value: TokenStream) -> TokenStream {
    if ctx.in_try_block() {
        quote!(return Ok(PyFlow::Return(#value)))
    } else {
        quote!(return Ok(#value))
    }
}

/// Whether a statement list contains a function-level `return` anywhere —
/// looking through control flow (including nested trys and their handlers)
/// but not into nested function or class definitions. The try lowering uses
/// this to pick its closure's carrier type: bodies with returns thread the
/// returned value out through PyFlow.
/// Does this statement list contain a `break`/`continue` that targets a
/// loop OUTSIDE the list? A loop nested *within* the list owns its own
/// breaks, so its body is not searched — but its `else` clause is, since
/// a break there targets the enclosing loop, as in Python. Nested
/// function and class bodies are separate scopes and never searched.
pub fn body_breaks_outward(body: &[Statement]) -> bool {
    body.iter().any(|stmt| match &stmt.statement {
        StatementType::Break | StatementType::Continue => true,
        StatementType::If(s) => body_breaks_outward(&s.body) || body_breaks_outward(&s.orelse),
        // A loop captures breaks in its BODY; only its else clause can
        // break outward.
        StatementType::For(s) => body_breaks_outward(&s.orelse),
        StatementType::While(s) => body_breaks_outward(&s.orelse),
        StatementType::AsyncFor(s) => body_breaks_outward(&s.orelse),
        StatementType::Try(s) => {
            body_breaks_outward(&s.body)
                || s.handlers.iter().any(|h| body_breaks_outward(&h.body))
                || body_breaks_outward(&s.orelse)
                || body_breaks_outward(&s.finalbody)
        }
        StatementType::With(s) => body_breaks_outward(&s.body),
        StatementType::AsyncWith(s) => body_breaks_outward(&s.body),
        _ => false,
    })
}

pub fn body_contains_function_return(body: &[Statement]) -> bool {
    body.iter().any(|stmt| match &stmt.statement {
        StatementType::Return(_) => true,
        StatementType::If(s) => {
            body_contains_function_return(&s.body) || body_contains_function_return(&s.orelse)
        }
        StatementType::For(s) => {
            body_contains_function_return(&s.body) || body_contains_function_return(&s.orelse)
        }
        StatementType::While(s) => {
            body_contains_function_return(&s.body) || body_contains_function_return(&s.orelse)
        }
        StatementType::AsyncFor(s) => {
            body_contains_function_return(&s.body) || body_contains_function_return(&s.orelse)
        }
        StatementType::Try(s) => {
            body_contains_function_return(&s.body)
                || s.handlers
                    .iter()
                    .any(|h| body_contains_function_return(&h.body))
                || body_contains_function_return(&s.orelse)
                || body_contains_function_return(&s.finalbody)
        }
        StatementType::With(s) => body_contains_function_return(&s.body),
        StatementType::AsyncWith(s) => body_contains_function_return(&s.body),
        _ => false,
    })
}

/// Whether a loop body contains a `break` that belongs to that loop —
/// looking through `if`/`try`/`with` blocks but not into nested loops
/// (whose breaks are their own) or nested definitions. Loops with an `else`
/// clause only need break-tracking machinery when this is true.
pub fn loop_body_has_direct_break(body: &[Statement]) -> bool {
    body.iter().any(|stmt| match &stmt.statement {
        StatementType::Break => true,
        StatementType::If(s) => {
            loop_body_has_direct_break(&s.body) || loop_body_has_direct_break(&s.orelse)
        }
        StatementType::Try(s) => {
            loop_body_has_direct_break(&s.body)
                || s.handlers
                    .iter()
                    .any(|h| loop_body_has_direct_break(&h.body))
                || loop_body_has_direct_break(&s.orelse)
                || loop_body_has_direct_break(&s.finalbody)
        }
        StatementType::With(s) => loop_body_has_direct_break(&s.body),
        StatementType::AsyncWith(s) => loop_body_has_direct_break(&s.body),
        _ => false,
    })
}

impl CodeGen for StatementType {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        match self {
            StatementType::Assign(a) => a.find_symbols(symbols),
            StatementType::AugAssign(a) => a.find_symbols(symbols),
            StatementType::ClassDef(c) => c.find_symbols(symbols),
            StatementType::FunctionDef(f) => f.find_symbols(symbols),
            // Async functions register like ordinary ones, so call sites
            // know they return Result and append `?` (before `.await`).
            StatementType::AsyncFunctionDef(f) => f.find_symbols(symbols),
            StatementType::Import(i) => i.find_symbols(symbols),
            StatementType::ImportFrom(i) => i.find_symbols(symbols),
            StatementType::Expr(e) => e.find_symbols(symbols),
            StatementType::If(i) => i.find_symbols(symbols),
            StatementType::For(f) => f.find_symbols(symbols),
            StatementType::While(w) => w.find_symbols(symbols),
            StatementType::Try(t) => t.find_symbols(symbols),
            StatementType::AsyncWith(aw) => aw.find_symbols(symbols),
            StatementType::AsyncFor(af) => af.find_symbols(symbols),
            StatementType::Raise(r) => r.find_symbols(symbols),
            StatementType::With(w) => w.find_symbols(symbols),
            _ => symbols,
        }
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        match self {
            StatementType::AsyncFunctionDef(s) => {
                let func_def = s.to_rust(Self::Context::Async(Box::new(ctx)), options, symbols)?;
                Ok(quote!(#func_def))
            }
            StatementType::Assert { test, msg } => {
                let test_tokens =
                    crate::condition_to_rust(&test, ctx.clone(), options.clone(), symbols.clone())?;
                let msg_tokens = match msg {
                    Some(m) => {
                        let m = m.to_rust(ctx.clone(), options, symbols)?;
                        quote!(format!("{}", #m))
                    }
                    None => quote!(String::new()),
                };
                // A failed assert raises AssertionError. Functions return
                // Result<T, PyException>, so raising is returning Err: it is
                // caught by an enclosing try's closure or propagates out of
                // the function, as in Python.
                Ok(quote! {
                    if !(#test_tokens) {
                        return Err(PyException::new("AssertionError", #msg_tokens));
                    }
                })
            }
            StatementType::Assign(a) => a.to_rust(ctx, options, symbols),
            StatementType::AugAssign(a) => a.to_rust(ctx, options, symbols),
            StatementType::Break => {
                // A break whose loop lies outside an enclosing try-block
                // closure cannot be a Rust `break` here — it would escape
                // the closure. Signal it out instead; the try lowering
                // replays it after the finally clause, as Python orders it.
                if ctx.break_crosses_try_closure() {
                    return Ok(if ctx.break_target_has_else() {
                        quote! {{ __rython_broke = true; return Ok(PyFlow::Break); }}
                    } else {
                        quote! {return Ok(PyFlow::Break);}
                    });
                }
                // Inside a loop that has an `else` clause, breaking must also
                // record that the loop did not complete normally.
                if matches!(ctx, Self::Context::Loop { has_else: true, .. }) {
                    Ok(quote! {{ __rython_broke = true; break; }})
                } else {
                    Ok(quote! {break;})
                }
            }
            StatementType::Call(c) => c.to_rust(ctx, options, symbols),
            StatementType::ClassDef(c) => c.to_rust(ctx, options, symbols),
            StatementType::Continue => {
                if ctx.break_crosses_try_closure() {
                    Ok(quote! {return Ok(PyFlow::Continue);})
                } else {
                    Ok(quote! {continue;})
                }
            }
            StatementType::Pass => Ok(quote! {}),
            StatementType::FunctionDef(s) => {
                if ctx.is_function_body() {
                    // A NESTED function definition (a closure in Python):
                    // rython's closures do not capture the enclosing
                    // function's scope (the closure-capture divergence), so
                    // the definition is a no-op — calls through the name
                    // drop (function_def.rs adds it to called_params).
                    options.definition_warnings.borrow_mut().push(format!(
                        "nested function `{}` is dropped: rython's closures do not \
                         capture the enclosing scope (the closure-capture divergence)",
                        s.name
                    ));
                    Ok(TokenStream::new())
                } else {
                    s.to_rust(ctx, options, symbols)
                }
            }
            StatementType::Import(s) => s.to_rust(ctx, options, symbols),
            StatementType::ImportFrom(s) => s.to_rust(ctx, options, symbols),
            StatementType::Expr(s) => s.to_rust(ctx, options, symbols),
            // Functions return Result<T, PyException>; a Python return wraps
            // its value in Ok (bare return / return None yield Ok(())).
            // Inside a try block's closure, a return must first break out of
            // the closure: it becomes Ok(PyFlow::Return(value)), which
            // the try lowering turns back into a function return — after
            // running the finally body, as Python requires.
            StatementType::Return(None) => {
                // A bare `return` in a PyValue-returning function returns
                // the boxed None (Python's implicit None); otherwise unit.
                if options.fn_return_is_pyvalue {
                    Ok(return_tokens(&ctx, quote!(PyValue::None_)))
                } else {
                    Ok(return_tokens(&ctx, quote!(())))
                }
            }
            StatementType::Return(Some(e)) => {
                // A `return None` in a PyValue-returning function is the
                // boxed None (the None-mixing unification), whichever AST
                // shape the parser surfaced None as (issue #133: the
                // annotated-path repro surfaces it as the NAME `None`, not
                // the NoneType variant); a plain-None function returns the
                // unit value.
                let value = if options.fn_return_is_pyvalue && crate::is_none_expr(&e.value) {
                    quote!(PyValue::None_)
                } else if matches!(e.value, ExprType::NoneType(_)) {
                    quote!(())
                } else if options.fn_return_is_pyvalue {
                    // A PyValue-returning function wraps its other returns
                    // (the identity From passes already-boxed values).
                    let tokens =
                        e.clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    quote!(PyValue::from(#tokens))
                } else {
                    let tokens = e.clone().to_rust(ctx.clone(), options.clone(), symbols)?;
                    // A `-> str` function's return value must be an owned
                    // String (the annotation is authoritative — see
                    // FunctionDef::resolved_return_type). An attribute chain
                    // reads a String field through the shared receiver:
                    // clone it out (a bare field read would move out of
                    // &self and not compile). A string literal lowers to
                    // `&'static str`; own it with to_string. Python strings
                    // are immutable, so both reproduce Python's semantics
                    // exactly.
                    if options.clone_str_attribute_returns {
                        match &e.value {
                            ExprType::Attribute(_) => quote!((#tokens).clone()),
                            ExprType::Constant(c)
                                if matches!(&c.0, Some(litrs::Literal::String(_))) =>
                            {
                                quote!((#tokens).to_string())
                            }
                            ExprType::Name(n) if options.str_literal_locals.contains(&n.id) => {
                                quote!((#tokens).to_string())
                            }
                            _ => tokens,
                        }
                    } else {
                        tokens
                    }
                };
                Ok(return_tokens(&ctx, value))
            }
            StatementType::If(i) => i.to_rust(ctx, options, symbols),
            StatementType::For(f) => f.to_rust(ctx, options, symbols),
            StatementType::While(w) => w.to_rust(ctx, options, symbols),
            StatementType::Try(t) => t.to_rust(ctx, options, symbols),
            StatementType::AsyncWith(aw) => aw.to_rust(ctx, options, symbols),
            StatementType::AsyncFor(af) => af.to_rust(ctx, options, symbols),
            StatementType::Raise(r) => r.to_rust(ctx, options, symbols),
            StatementType::With(w) => w.to_rust(ctx, options, symbols),
            // `global a, b` declares module scope — a no-op here: reads
            // resolve to the module statics, and writes are rejected at
            // conversion time (issue #115).
            StatementType::Global(_) => Ok(quote! {}),
            StatementType::Nonlocal(_) => Ok(quote! {}),
            // A bare annotated declaration (`x: int`) declares nothing at
            // runtime: the annotation only types the name (dataclass-style
            // field declarations are consumed by the class codegen; inside
            // a function the annotation types later assignments).
            StatementType::AnnotatedName { .. } => Ok(quote! {}),
            // `del xs[i]` / `del d[k]`: Python removes the element at the
            // index (negative from the end, IndexError/KeyError when
            // missing) — the runtime's py_pop already implements exactly
            // that; the returned element is discarded. `del name` and
            // `del a.b` are loud errors: unbinding a name or removing a
            // struct field is not representable in rython's value model.
            StatementType::Delete(targets) => {
                let mut stmts = Vec::new();
                for target in targets {
                    match target {
                        ExprType::Subscript(sub) => {
                            let receiver = crate::subscript_receiver_place(
                                &sub.value,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            match &sub.kind {
                                crate::SubscriptKind::Index(index) => {
                                    let idx = index.clone().to_rust(
                                        ctx.clone(),
                                        options.clone(),
                                        symbols.clone(),
                                    )?;
                                    // A string-literal KEY is owned (dict
                                    // keys normalize to String).
                                    let idx = if matches!(
                                        index.as_ref(),
                                        ExprType::Constant(c)
                                            if matches!(
                                                &c.0,
                                                Some(litrs::Literal::String(_))
                                            )
                                    ) {
                                        quote!((#idx).to_string())
                                    } else {
                                        idx
                                    };
                                    // py_pop takes the index BY VALUE.
                                    stmts.push(quote!((#receiver).py_pop(#idx)?;));
                                }
                                crate::SubscriptKind::Slice {
                                    lower, upper, step, ..
                                } => {
                                    // `del xs[:]` — a FULL slice (all bounds
                                    // None) clears the container in place
                                    // (`del self._buffer[:]` — botocore's
                                    // AWSConnection._send_output): the
                                    // runtime's clear (Python's `xs[:] = []`).
                                    // An explicit step of 1 is the same
                                    // operation.
                                    let step_is_one =
                                        crate::ast::tree::subscript::is_step_one(step.as_deref());
                                    if lower.is_none()
                                        && upper.is_none()
                                        && (step.is_none() || step_is_one)
                                    {
                                        stmts.push(quote!((#receiver).clear();));
                                    } else if !step_is_one && step.is_some() {
                                        // An extended-slice delete (`del
                                        // xs[a:b:c]`, c != 0) removes the
                                        // selected slots in place.
                                        let lo_tok = match lower {
                                            Some(e) => {
                                                let t = e.clone().to_rust(
                                                    ctx.clone(),
                                                    options.clone(),
                                                    symbols.clone(),
                                                )?;
                                                quote!(Some(#t))
                                            }
                                            None => quote!(None),
                                        };
                                        let up_tok = match upper {
                                            Some(e) => {
                                                let t = e.clone().to_rust(
                                                    ctx.clone(),
                                                    options.clone(),
                                                    symbols.clone(),
                                                )?;
                                                quote!(Some(#t))
                                            }
                                            None => quote!(None),
                                        };
                                        let st_tok = step.clone().unwrap().to_rust(
                                            ctx.clone(),
                                            options.clone(),
                                            symbols.clone(),
                                        )?;
                                        stmts.push(quote!(
                                            (#receiver).py_slice_delete_step(#lo_tok, #up_tok, #st_tok)?;
                                        ));
                                    } else {
                                        // A BOUNDED slice delete (`del
                                        // xs[start:end]`) removes a range of
                                        // elements in place (issue #153).
                                        // Bounds clamp like reads: negatives
                                        // count from the end, out-of-range
                                        // clamps to the edges.
                                        let lo_tok = match lower {
                                            Some(e) => {
                                                let t = e.clone().to_rust(
                                                    ctx.clone(),
                                                    options.clone(),
                                                    symbols.clone(),
                                                )?;
                                                quote!(Some(#t))
                                            }
                                            None => quote!(None),
                                        };
                                        let up_tok = match upper {
                                            Some(e) => {
                                                let t = e.clone().to_rust(
                                                    ctx.clone(),
                                                    options.clone(),
                                                    symbols.clone(),
                                                )?;
                                                quote!(Some(#t))
                                            }
                                            None => quote!(None),
                                        };
                                        stmts.push(quote!(
                                            (#receiver).py_slice_delete(#lo_tok, #up_tok);
                                        ));
                                    }
                                }
                            }
                        }
                        ExprType::Name(_) => {
                            // `del name` unbinds the binding. Lowered to a
                            // no-op: behaviorally identical as long as the
                            // name is not referenced afterwards, which the
                            // check_deleted_names pass enforces loudly
                            // (issue #112).
                        }
                        ExprType::Attribute(a) => {
                            // `del obj.attr` on a NON-self object (`del
                            // newmod.newmod, ...` — pygments' module
                            // proxy cleanup, where newmod is a module
                            // object from _automodule): dynamic module
                            // machinery — a no-op with a warning (the
                            // module-object / class-as-value divergence).
                            // `del self.field` remains a loud error (a
                            // real struct-member removal).
                            if !matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                                options.definition_warnings.borrow_mut().push(format!(
                                    "del of `{}` attribute is dropped (removing an \
                                     attribute from a non-self object is unmodeled — \
                                     the module-object/class-as-value divergence)",
                                    a.attr
                                ));
                            } else {
                                return Err(format!(
                                    "del with an attribute target (removing a field) is not \
                                     supported: class fields are struct members and cannot be \
                                     removed (issue #112)"
                                )
                                .into());
                            }
                        }
                        _ => {
                            return Err("del with this target shape is not supported (issue #112)"
                                .to_string()
                                .into());
                        }
                    }
                }
                Ok(quote!(#(#stmts)*))
            }
            _ => {
                let error = err_from(StatementNotYetImplemented(self));
                Err(error.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_pass_statement() {
        let statement = StatementType::Pass;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), true);
    }

    #[test]
    fn check_break_statement() {
        let statement = StatementType::Break;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), false);
    }

    #[test]
    fn check_continue_statement() {
        let statement = StatementType::Continue;
        let options = PythonOptions::default();
        let tokens = statement.clone().to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );

        debug!("statement: {:?}, tokens: {:?}", statement, tokens);
        assert_eq!(tokens.unwrap().is_empty(), false);
    }

    #[test]
    fn return_with_nothing() {
        let tree = crate::parse("return", "<none>").unwrap();
        assert_eq!(tree.raw.body.len(), 1);
        assert_eq!(
            tree.raw.body[0].statement,
            StatementType::Return(Some(Expr {
                value: crate::tree::ExprType::NoneType(crate::tree::Constant(None)),
                lineno: Some(1),
                col_offset: Some(0),
                end_lineno: Some(1),
                end_col_offset: Some(6),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn return_with_expr() {
        let lit = litrs::Literal::Integer(litrs::IntegerLit::parse(String::from("8")).unwrap());
        let tree = crate::parse("return 8", "<none>").unwrap();
        assert_eq!(tree.raw.body.len(), 1);
        assert_eq!(
            tree.raw.body[0].statement,
            StatementType::Return(Some(Expr {
                value: crate::tree::ExprType::Constant(crate::tree::Constant(Some(lit))),
                lineno: Some(1),
                col_offset: Some(0),
                end_lineno: Some(1),
                end_col_offset: Some(8),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn does_module_compile() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "#test comment
def foo():
    continue
    pass
",
            "test_case",
        )
        .unwrap();
        tracing::info!("{:?}", result);
        let code = result.to_rust(
            CodeGenContext::Module("".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        tracing::info!("module: {:?}", code);
    }
}
