use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, Statement, SymbolTableNode,
    SymbolTableScopes, extract_list, WithItem,
};

/// The exception NAME of a `self.assertRaises(Exc)` context manager — the
/// `with self.assertRaises(Exc):` form (issue #334). None when the shape is
/// not that CM or the exception is not a statically-known name (a tuple of
/// exceptions, or a dynamic `self.failureException`).
pub(crate) fn assert_raises_cm_name(expr: &ExprType) -> Option<String> {
    let ExprType::Call(c) = expr else {
        return None;
    };
    let ExprType::Attribute(a) = c.func.as_ref() else {
        return None;
    };
    if a.attr != "assertRaises" || c.args.len() != 1 {
        return None;
    }
    match &c.args[0] {
        ExprType::Name(n) => Some(n.id.clone()),
        ExprType::Attribute(ea) => Some(ea.attr.clone()),
        _ => None,
    }
}

/// Whether a `with` expression is the unittest `self.subTest(...)` context
/// manager.
fn is_subtest_cm(expr: &ExprType) -> bool {
    let ExprType::Call(c) = expr else {
        return false;
    };
    let ExprType::Attribute(a) = c.func.as_ref() else {
        return false;
    };
    a.attr == "subTest" && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self")
}

/// Whether a `with` body contains a control transfer that would have to
/// escape the closure its lowering wraps it in.
fn body_escapes_a_closure(body: &[Statement]) -> bool {
    crate::ast::tree::visit::any_stmt(
        body,
        crate::ast::tree::visit::Descend::SkipDefs,
        |s| {
            matches!(
                s.statement,
                crate::StatementType::Return(_)
                    | crate::StatementType::Break
                    | crate::StatementType::Continue
            )
        },
    )
}

fn has_subtest_repr(ty: &crate::TypeInfo) -> bool {
    match ty {
        crate::TypeInfo::Int
        | crate::TypeInfo::BigInt
        | crate::TypeInfo::Float
        | crate::TypeInfo::Bool
        | crate::TypeInfo::StrRef
        | crate::TypeInfo::String
        | crate::TypeInfo::Complex
        | crate::TypeInfo::PyValue => true,
        crate::TypeInfo::Vec(inner)
        | crate::TypeInfo::PyTuple(inner)
        | crate::TypeInfo::Option(inner)
        | crate::TypeInfo::PyValueMember(inner)
        | crate::TypeInfo::Borrowed(inner) => has_subtest_repr(inner),
        crate::TypeInfo::Dict(key, value) => {
            has_subtest_repr(key) && has_subtest_repr(value)
        }
        crate::TypeInfo::Tuple(items) => {
            !items.is_empty() && items.len() <= 8 && items.iter().all(has_subtest_repr)
        }
        _ => false,
    }
}

/// Whether a with-item's context expression is a threading synchronization
/// object (Lock/RLock/Semaphore) — constructed inline
/// (`with threading.Lock():`), a name assigned from such a construction,
/// or a PARAMETER annotated as one (`def crit(lock: threading.Lock):` —
/// the exact pass-a-lock-to-a-worker pattern; local_types records the
/// annotation). These have REAL `__enter__`/`__exit__` semantics
/// (acquire/release), so the with-statement must lower to the RAII guard
/// instead of the plain binding.
fn is_threading_sync_call(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let is_sync_name =
        |name: &str| crate::ThreadingType::from_name(name).is_some_and(|t| t.is_sync_guard());
    match expr {
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Attribute(attr) => {
                is_sync_name(&attr.attr)
                    && matches!(attr.value.as_ref(), ExprType::Name(n)
                        if crate::StdModule::from_name(&n.id)
                            == Some(crate::StdModule::Threading))
                    && !crate::module_name_shadowed(crate::StdModule::Threading.name(), symbols)
            }
            ExprType::Name(n) => {
                is_sync_name(&n.id)
                    && matches!(
                        symbols.get(&n.id),
                        Some(SymbolTableNode::ImportFrom(i))
                            if crate::StdModule::from_name(&i.module)
                                == Some(crate::StdModule::Threading)
                    )
            }
            _ => false,
        },
        ExprType::Name(n) => {
            // An annotated parameter (or annotated local) recorded in
            // local_types as "threading.Lock"/"threading.RLock"/
            // "threading.Semaphore".
            let annotated_sync = options.local_types.get(&n.id).is_some_and(|py| {
                py.strip_prefix("threading.").is_some_and(is_sync_name)
            });
            annotated_sync
                || matches!(
                    symbols.get(&n.id),
                    Some(SymbolTableNode::Assign { value: ExprType::Call(c), .. })
                        if is_threading_sync_call(&ExprType::Call(c.clone()), symbols, options)
                )
        }
        _ => false,
    }
}

/// Regular with statement (with context as var: ...)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct With {
    /// The with items (context managers)
    pub items: Vec<WithItem>,
    /// The body of the with statement
    pub body: Vec<Statement>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for With {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract items (list of withitem objects)
        let items: Vec<WithItem> = extract_list(&ob, "items", "with items")?;
        
        // Extract body
        let body: Vec<Statement> = extract_list(&ob, "body", "with body")?;
        
        Ok(With {
            items,
            body,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl Node for With {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl CodeGen for With {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        // Process items and body
        let symbols = self.items.into_iter().fold(symbols, |acc, item| {
            let acc = item.context_expr.find_symbols(acc);
            if let Some(vars) = item.optional_vars {
                vars.find_symbols(acc)
            } else {
                acc
            }
        });
        self.body.into_iter().fold(symbols, |acc, stmt| stmt.find_symbols(acc))
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // `with self.subTest(...):` catches a body exception as a subtest
        // failure and lets the test method continue. A closure models this
        // boundary; control-flow transfers that would escape it are refused.
        if self
            .items
            .iter()
            .any(|item| is_subtest_cm(&item.context_expr))
        {
            if self.items.len() != 1 || self.items[0].optional_vars.is_some() {
                return Err(format!(
                    "`with self.subTest(...)` at line {} must be the only context manager \
                     and cannot bind an `as` target; rython refuses to silently skip \
                     subtest failure handling",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            if body_escapes_a_closure(&self.body) {
                return Err(format!(
                    "`with self.subTest(...)` at line {} containing `return`, `break`, or \
                     `continue` is not supported yet; move that control flow outside the \
                     subTest body; rython refuses to silently change its target",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            let ExprType::Call(subtest_call) = &self.items[0].context_expr else {
                unreachable!("is_subtest_cm only matches calls");
            };
            if !subtest_call.args.is_empty()
                || subtest_call
                    .keywords
                    .iter()
                    .any(|keyword| keyword.arg.is_none() || keyword.arg.as_deref() == Some("msg"))
            {
                return Err(format!(
                     "`self.subTest` at line {} supports named context parameters only; positional \
                      messages, `msg=`, and `**kwargs` are not supported yet; use \
                      `name=value`; \
                     rython refuses to silently change subtest reporting",
                    self.lineno.unwrap_or(0)
                )
                .into());
            }
            let mut context_values = Vec::new();
            for keyword in &subtest_call.keywords {
                let name = keyword
                    .arg
                    .as_ref()
                    .expect("unpacked kwargs rejected above");
                let ty = crate::infer_type(Some(&ctx), &keyword.value, &options, &symbols);
                if !has_subtest_repr(&ty) {
                    return Err(format!(
                        "`self.subTest({name}=...)` at line {} has a value without a \
                         supported Python repr; use a scalar or supported container; \
                         rython refuses to silently change subtest reporting",
                        keyword.lineno.or(self.lineno).unwrap_or(0)
                    )
                    .into());
                }
                let value = keyword.value.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                context_values.push(quote! {
                    (#name.to_string(), stdpython::PyRepr::py_repr(&(#value)))
                });
            }
            let context = quote!(vec![#(#context_values),*]);
            let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .body
                .iter()
                .map(|stmt| {
                    stmt.clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                })
                .collect();
            let body_tokens = body_tokens?;
            let tail = if crate::ast::tree::function_def::guarantees_return(&self.body) {
                quote!()
            } else {
                quote!(Ok(()))
            };
            return Ok(quote! {
                stdpython::unittest::run_subtest(#context, || -> Result<(), PyException> {
                    #(#body_tokens;)*
                    #tail
                })?;
            });
        }
        // `with self.assertRaises(Exc):` — the unittest context-manager form
        // (issue #334). The body runs inside a closure whose `Result` is
        // asserted to have raised `Exc` (matching `assert_raises`): a raised
        // `Exc` passes, no raise is an AssertionError, a different exception
        // propagates. A body with `return`/`break`/`continue` would have to
        // escape the closure, so that shape keeps the plain (loud-drop)
        // lowering instead.
        if self.items.len() == 1
            && self.items[0].optional_vars.is_none()
            && !body_escapes_a_closure(&self.body)
            && let Some(exc) = assert_raises_cm_name(&self.items[0].context_expr)
        {
            let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self
                .body
                .iter()
                .map(|stmt| {
                    stmt.clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                })
                .collect();
            let body_tokens = body_tokens?;
            let exc_lit = quote!(#exc);
            // A body that RAISES unconditionally (`raise E(...)` as its
            // last statement) already ends in `return Err(..)`, which is
            // the closure's exit: appending `Ok(())` after it is dead code
            // and makes rustc see the closure return `()`, not
            // `Result<(), PyException>` (issue #457 - the reported E0308
            // lands on the ENCLOSING method, because `guarantees_return`
            // unwraps a `with` into its body and so withholds the
            // method's own tail). Only a body that can fall through needs
            // the tail.
            let tail = if crate::ast::tree::function_def::guarantees_return(&self.body) {
                quote!()
            } else {
                quote!(Ok(()))
            };
            return Ok(quote! {
                unittest::assert_raises(#exc_lit, || -> Result<(), PyException> {
                    #(#body_tokens;)*
                    #tail
                })?;
            });
        }
        // Evaluate each context manager and bind its `as` target (or a
        // throwaway binding when there is none, so side effects still run).
        // The general __enter__/__exit__ protocol is not modeled yet
        // (Rust's Drop at end of block approximates __exit__ cleanup), with
        // one REAL implementation: threading Lock/RLock/Semaphore context
        // expressions lower to the runtime's RAII guard — acquire at entry,
        // release when the guard drops, exception-safe through unwinding
        // `?`, exactly Python's with-lock discipline.
        let mut item_tokens = Vec::new();
        for (index, item) in self.items.into_iter().enumerate() {
            let is_sync = is_threading_sync_call(&item.context_expr, &symbols, &options);
            // A text file's methods all take `&self` (the handle is a
            // shared reference, like Python's), so its binding is never
            // `mut`.
            let is_file = crate::ast::tree::type_ctx::is_file_typeinfo(&crate::infer_type(
                Some(&ctx),
                &item.context_expr,
                &options,
                &symbols,
            ));
            let context_expr =
                item.context_expr
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            match item.optional_vars {
                Some(vars) if is_sync => {
                    // CPython binds the target to __enter__'s return (True
                    // for locks) — a shape with no honest lowering here.
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    return Err(format!(
                        "`with <lock> as {}:` is not supported yet (a lock's __enter__ \
                         returns True, not the lock); use `with <lock>:` and name the \
                         lock outside the statement",
                        target
                    )
                    .into());
                }
                Some(vars) => {
                    // The item's target is bound right after its context
                    // expression, before the next item's runs: its marks go
                    // there (Devin review on #338, round 12).
                    let bind = crate::ast::tree::import::binds_for(
                        options.stmt_binds.as_deref(),
                        crate::ast::tree::visit::target_names(&vars).into_iter(),
                    );
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    if is_file {
                        item_tokens.push(quote! { let #target = #context_expr; #bind });
                    } else {
                        item_tokens.push(quote! { let mut #target = #context_expr; #bind });
                    }
                }
                None if is_sync => {
                    let guard = crate::safe_ident(&format!("__rython_with_guard_{}", index));
                    item_tokens.push(quote! { let #guard = (#context_expr).py_guard()?; });
                }
                None => {
                    item_tokens.push(quote! { let _ = #context_expr; });
                }
            }
        }

        let body_tokens: Result<Vec<TokenStream>, Box<dyn std::error::Error>> = self.body.into_iter()
            .map(|stmt| stmt.to_rust(ctx.clone(), options.clone(), symbols.clone()))
            .collect();
        let body_tokens = body_tokens?;

        Ok(quote! {
            {
                #(#item_tokens)*
                #(#body_tokens;)*
            }
        })
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here - currently commented out as they need full AST infrastructure
    // create_parse_test!(test_simple_with, "with context:\n    pass", "test.py");
}