use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, Statement, SymbolTableNode,
    SymbolTableScopes, extract_list, WithItem,
};

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
                    let target = vars.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    item_tokens.push(quote! { let mut #target = #context_expr; });
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