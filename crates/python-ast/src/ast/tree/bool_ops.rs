use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    dump, extraction_failure, err_from, BoolOpNotYetImplemented, CodeGen, CodeGenContext, ExprType,
    PythonOptions, SymbolTableScopes,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum BoolOps {
    And,
    Or,
    Unknown,
}

impl<'a, 'py> FromPyObject<'a, 'py> for BoolOps {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let op_type = ob
            .get_type()
            .name()
            .map_err(|e| extraction_failure("boolean operator type", &ob, e))?;

        let op_type_str: String = op_type.extract()?;
        let op = match op_type_str.as_str() {
            "And" => BoolOps::And,
            "Or" => BoolOps::Or,
            _ => {
                tracing::debug!("Found unknown BoolOp {:?}", op_type_str);
                BoolOps::Unknown
            }
        };

        Ok(op)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoolOp {
    pub op: BoolOps,
    /// All operands: Python collapses `a and b and c` into one BoolOp node
    /// with three values.
    pub values: Vec<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for BoolOp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        tracing::debug!("ob: {}", dump(&ob, None)?);
        let op = ob.getattr("op").map_err(|e| extraction_failure("op", &ob, e))?;

        let op_type = op
            .get_type()
            .name()
            .map_err(|e| extraction_failure("boolean operator type", &ob, e))?;

        let values = ob.getattr("values").map_err(|e| extraction_failure("values", &ob, e))?;

        tracing::debug!("BoolOps values: {}", dump(&values, None)?);

        let values: Vec<ExprType> = values.extract().map_err(|e| extraction_failure("getting values from BoolOp", &ob, e))?;

        let op_type_str: String = op_type.extract()?;
        let op = match op_type_str.as_str() {
            "And" => BoolOps::And,
            "Or" => BoolOps::Or,

            _ => {
                tracing::debug!("Found unknown BoolOp {:?}", op);
                BoolOps::Unknown
            }
        };

        tracing::debug!("values: {:?}, op: {:?}/{:?}", values, op_type, op);

        return Ok(BoolOp { op, values });
    }
}

impl<'a> CodeGen for BoolOp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Python's boolean operators return OPERANDS, not booleans. The
        // operand-returning forms below reproduce that exactly when the
        // operands' types can be unified; anything else keeps the
        // `&&`/`||` approximation, which fails loudly in rustc (§12.1)
        // rather than silently returning a bool where Python returns a
        // value (the ca_certs-and-expanduser shape — urllib3).
        let mut rendered = Vec::new();
        for value in self.values.clone() {
            rendered.push(value.to_rust(ctx.clone(), options.clone(), symbols.clone())?);
        }

        match self.op {
            BoolOps::Or => {
                // `a or None` yields the Option-model None when `a` is
                // falsy — dropping the None silently returned the falsy
                // value instead (`0 or None` must be None, not 0).
                if let Some(last) = rendered.last() {
                    if last.to_string().trim() == "None" && rendered.len() == 2 {
                        let first = &rendered[0];
                        return Ok(quote!({
                            let __rython_or = #first;
                            if (__rython_or).is_truthy() { Some(__rython_or) } else { None }
                        }));
                    }
                }
                Ok(fold(
                    &self.values,
                    &rendered,
                    BoolOps::Or,
                    &ctx,
                    &options,
                    &symbols,
                ))
            }
            BoolOps::And => Ok(fold(
                &self.values,
                &rendered,
                BoolOps::And,
                &ctx,
                &options,
                &symbols,
            )),

            _ => Err(err_from(BoolOpNotYetImplemented(self)).into()),
        }
    }
}

/// Fold the rendered operands left-to-right with Python's
/// operand-returning semantics: `a and b = if truthy(a) { b } else {
/// a }`, `a or b = if truthy(a) { a } else { b }`. When exactly one
/// operand is `Option<T>` and the other is `T`, the `T` arm wraps in
/// `Some` so the if-else arms agree (`ca_certs and
/// expanduser(ca_certs)` — Option<String> and String). Any other mix
/// (bool and str, two different types) falls back to the `&&`/`||`
/// approximation — loud in rustc, never a silent operand-vs-bool swap.
/// A chained BoolOp folds pairwise; a fold whose inner result the outer
/// check cannot unify stays `&&`/`||`.
fn fold(
    values: &[crate::ExprType],
    rendered: &[TokenStream],
    op: BoolOps,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TokenStream {
    fn fold_at(
        values: &[crate::ExprType],
        rendered: &[TokenStream],
        op: BoolOps,
        i: usize,
        ctx: &CodeGenContext,
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> TokenStream {
        if i == rendered.len() - 1 {
            return rendered[i].clone();
        }
        let first = &rendered[i];
        let rest = fold_at(values, rendered, op, i + 1, ctx, options, symbols);
        let a = crate::infer_type(&values[i], options, symbols);
        let b = crate::infer_type(&values[i + 1], options, symbols);
        use crate::TypeInfo as T;
        // Whether the operand can hold the Option's inner type: the
        // concrete match (`ca and x` where both are str), a STRING
        // literal (`ca and "fixed"` — the literal infers StrRef, not
        // String), or an UNKNOWN type (`ca and expanduser(ca)` — the
        // call's return infers PyObject because method/module-call
        // returns are unresolved, but the rendered expression IS the
        // inner type). An unknown operand falls back to the Option arm
        // and lets rustc judge: `Some(#rest)` into the Option<T> context
        // is loud if #rest is not T — the same loudness the && fallback
        // would have, but correct for the unifiable case.
        fn inner_matches(inner: &T, other: &T) -> bool {
            inner == other
                || matches!(other, T::PyObject)
                || (matches!(inner, T::String) && matches!(other, T::StrRef))
                || (matches!(inner, T::Bytes) && matches!(other, T::StrRef))
        }
        match (&a, &b) {
            // a: Option<T>, b: T — the falsy arm holds the Option, the
            // truthy arm wraps the plain value.
            (T::Option(inner), b) if inner_matches(inner, b) => {
                if op == BoolOps::And {
                    let wrapped = some_arm(&values[i + 1], quote!(#rest));
                    quote!({
                        let __rython_and = #first;
                        if (__rython_and).is_truthy() { #wrapped } else { __rython_and }
                    })
                } else {
                    let wrapped = some_arm(&values[i + 1], quote!(#rest));
                    quote!({
                        let __rython_or = #first;
                        if (__rython_or).is_truthy() { __rython_or } else { #wrapped }
                    })
                }
            }
            // a: T, b: Option<T> — the truthy arm holds the plain value
            // and wraps it; the falsy arm is already Option.
            (a, T::Option(inner)) if inner_matches(inner, a) => {
                if op == BoolOps::And {
                    let wrapped = some_arm(&values[i], quote!(__rython_and));
                    quote!({
                        let __rython_and = #first;
                        if (__rython_and).is_truthy() { #rest } else { #wrapped }
                    })
                } else {
                    let wrapped = some_arm(&values[i], quote!(__rython_or));
                    quote!({
                        let __rython_or = #first;
                        if (__rython_or).is_truthy() { #wrapped } else { #rest }
                    })
                }
            }
            _ => {
                if op == BoolOps::And {
                    quote!((#first) && (#rest))
                } else {
                    quote!((#first) || (#rest))
                }
            }
        }
    }
    fold_at(values, rendered, op, 0, ctx, options, symbols)
}

/// Wrap a plain arm in `Some(...)`. A string LITERAL lowers to
/// `&'static str`, but an Option<String> slot owns its string — the
/// literal must be owned at the wrap (`scheme or "http"` where scheme is
/// `str | None` — urllib3): the same ownership the optional-store path
/// applies. Any other operand wraps as-is.
fn some_arm(expr: &crate::ExprType, tokens: TokenStream) -> TokenStream {
    if matches!(expr, crate::ExprType::Constant(c)
        if matches!(&c.0, Some(litrs::Literal::String(_))))
    {
        quote!(Some((#tokens).to_string()))
    } else {
        quote!(Some(#tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and() {
        let options = PythonOptions::default();
        let result = crate::parse("1 and 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //tracing::info!("{}", result.to_rust().unwrap());

        let code = result
            .to_rust(
                CodeGenContext::Module("test_case".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
        tracing::info!("module: {:?}", code);
    }

    #[test]
    fn test_or() {
        let options = PythonOptions::default();
        let result = crate::parse("1 or 2", "test_case.py").unwrap();
        tracing::info!("Python tree: {:?}", result);
        //tracing::info!("{}", result);

        let code = result
            .to_rust(
                CodeGenContext::Module("test_case".to_string()),
                options,
                SymbolTableScopes::new(),
            )
            .unwrap();
        tracing::info!("module: {:?}", code);
    }
}
