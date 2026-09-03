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
        // infer_type cannot see `self.<field>` (it has no class context),
        // so a self-field operand's Option-ness is resolved through the
        // class table's field type (`self.path or "/"` — urllib3's Url:
        // the field is Option<String>; the fold's Option arm needs to
        // know, or the operand-returning semantics fall to `||`).
        let a = fold_operand_type(&values[i], ctx, options, symbols);
        let b = fold_operand_type(&values[i + 1], ctx, options, symbols);
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
                // CONTAINER-typed pairs that unify (`headers or {}` where
                // headers is `Mapping[str, str] | None` and the empty-dict
                // literal infers Dict(PyObject, PyObject) — urllib3's
                // RequestMethods): the literal's element types are
                // unknown, but the Option's inner anchors them. Round 62:
                // unify() is the same compatibility relation the rest of
                // the codebase uses; a boxed-PyValue result is excluded
                // (a PyValue operand does not Some-wrap into Option<T>).
                || !matches!(
                    crate::ast::tree::type_ctx::unify(inner.clone(), other.clone()),
                    T::PyObject | T::PyValue
                )
        }
        match (&a, &b) {
            // a: Option<T>, b: T — the falsy arm holds the Option, the
            // truthy arm wraps the plain value.
            (T::Option(inner), b) if inner_matches(inner, b) => {
                // The truthy arm's operand is rendered against the UNWRAPPED
                // inner: `ca_certs and os.path.expanduser(ca_certs)` passes
                // the STRING (the inner value) to expanduser, never the
                // Option (CPython: falsy None → None, truthy string →
                // expanduser(string)). Re-render the operand with the Option
                // name narrowed to its inner type so the call argument
                // reads `(#name).clone().unwrap()` (round 48). When the
                // narrowing fires the truthy arm re-reads the NAME, so the
                // fold's bind must CLONE (the move would poison it) — the
                // bind holds the Option for the truthiness test and the
                // falsy arm; the name stays owned for the inner read.
                let (narrowed_rest, bind_clone) = match narrow_option_operand(
                    &values[i],
                    &values[i + 1],
                    inner,
                    ctx,
                    options,
                    symbols,
                ) {
                    Some(tokens) => (tokens, true),
                    None => (quote!(#rest), false),
                };
                let bound = if bind_clone {
                    quote!(let __rython_and = (#first).clone();)
                } else {
                    quote!(let __rython_and = #first;)
                };
                if op == BoolOps::And {
                    let wrapped = some_arm(&values[i + 1], narrowed_rest);
                    quote!({
                        #bound
                        if (__rython_and).is_truthy() { #wrapped } else { __rython_and }
                    })
                } else if matches!(b, T::PyObject)
                    // A NAME-typed Option operand (`scheme or "http"` —
                    // a `str | None` parameter): keep the round-43
                    // Option-producing fold (the result feeds Option
                    // slots and `-> T | None` returns). Only a
                    // SELF-FIELD Option operand (`self.path or "/"` —
                    // urllib3's Url, whose `-> str` property needs the
                    // plain value) unwraps-or-defaults to T (round 48).
                    || matches!(values[i], crate::ExprType::Name(_))
                {
                    // UNKNOWN other operand or a NAME operand: keep the
                    // Option-producing fold (rustc judges the Some-wrap).
                    let wrapped = some_arm(&values[i + 1], narrowed_rest);
                    quote!({
                        let __rython_or = #first;
                        if (__rython_or).is_truthy() { __rython_or } else { #wrapped }
                    })
                } else {
                    // `Option<T> or T` with a CONCRETE other operand
                    // (`self.path or "/"` — urllib3's Url): Python's
                    // result is never None (None is falsy, so the
                    // concrete default wins) — UNWRAP the Some to the
                    // inner value and default to the operand. A
                    // truthy Some("") returns the empty string (the
                    // documented Option-truthiness gap: CPython "" is
                    // falsy and would take the default). A string
                    // literal default is owned (String, not &str).
                    let default = if matches!(&values[i + 1], crate::ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_))))
                    {
                        quote!((#rest).to_string())
                    } else {
                        quote!(#rest)
                    };
                    // A SELF-FIELD operand reads through the shared
                    // receiver — the match must CLONE it (a bare
                    // `match self.path` moves out of `&self`, E0507).
                    // A NAME operand is excluded above (stays the
                    // Option-producing fold).
                    let bound = quote!(let __rython_field = (#first).clone(););
                    quote!({
                        #bound
                        match __rython_field {
                            Some(__rython_inner) => __rython_inner,
                            None => #default,
                        }
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
            // a: bool, b: boxed — `redirect and
            // response.get_redirect_location()` (urllib3's poolmanager,
            // where the call returns a boxed PyValue): Python returns the
            // SECOND operand when the first is truthy, else the first.
            // The `&&` fallback would type the result as bool — poisoning
            // every downstream use (`urljoin(url, redirect_location)` —
            // round 55: the boxed value must survive, or the real
            // urljoin call fails on a `&bool` arg). Only a DEFINITELY
            // boxed operand qualifies: an UNKNOWN (PyObject) operand may
            // be a bool-returning call (`bom_or_sig_available and
            // should_strip_sig_or_bom(...)` — charset_normalizer), which
            // must stay `&&` (the local is `bool`; boxing would break
            // every `== &false` use).
            (T::Bool, T::PyValue) | (T::PyValue, T::Bool) => {
                // Python `a and b` returns a when a is falsy, else b;
                // `a or b` returns a when a is truthy, else b — in BOTH
                // orders the FIRST operand decides and the SECOND is the
                // alternative. The round-55 fold special-cased the
                // bool-first order and SWAPPED the arms for the
                // value-first order (`x and True` returned x on a truthy
                // x instead of True — the retrospective's shipped
                // wrong-semantics finding on #260). Order-independent:
                // `and` -> truthy ? rest : first; `or` -> truthy ?
                // first : rest.
                if op == BoolOps::And {
                    quote!({
                        let __rython_and = #first;
                        if (__rython_and).is_truthy() { PyValue::from(#rest) } else { PyValue::from((__rython_and).clone()) }
                    })
                } else {
                    quote!({
                        let __rython_or = #first;
                        if (__rython_or).is_truthy() { PyValue::from((__rython_or).clone()) } else { PyValue::from(#rest) }
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

/// The fold's operand type: [`infer_type`] plus the cases it cannot see —
/// a SELF-FIELD read (`self.path` — the field's class-table type; urllib3's
/// Url stores `Option<String>` fields, and `self.path or "/"` must fold
/// with the Option arm, not fall to `||`) and a NAME whose Option-ness
/// lives only in `optional_names` (`conn = None` then `conn = ...` —
/// urllib3's _get_conn: the recorded None assignment infers PyObject, but
/// the scope analysis tracks the binding as Option, so `conn or
/// self._new_conn()` must fold with the Option arm — round 62). The inner
/// type resolves from the recorded name type when it is an Option; an
/// unknown inner (PyObject) still folds — the Some-wrap lets rustc judge.
pub(crate) fn fold_operand_type(
    expr: &crate::ExprType,
    ctx: &crate::CodeGenContext,
    options: &crate::PythonOptions,
    symbols: &crate::SymbolTableScopes,
) -> crate::TypeInfo {
    let inferred = crate::infer_type(Some(&ctx), expr, options, symbols);
    if !matches!(inferred, crate::TypeInfo::PyObject) {
        return inferred;
    }
    if let crate::ExprType::Name(n) = expr
        && options.optional_names.contains(&n.id)
    {
        let inner = match options.name_types.get(&n.id) {
            Some(crate::TypeInfo::Option(inner)) => (**inner).clone(),
            _ => crate::TypeInfo::PyObject,
        };
        return crate::TypeInfo::Option(Box::new(inner));
    }
    let crate::ExprType::Attribute(attr) = expr else {
        return inferred;
    };
    if !matches!(attr.value.as_ref(), crate::ExprType::Name(r) if r.id == "self") {
        return inferred;
    }
    let field_ty = Some(crate::infer_type(Some(ctx), expr, options, symbols));
    if field_ty.as_ref().is_some_and(|t| matches!(t, crate::TypeInfo::Option(_))) {
        // The inner type from the field's TypeInfo, so the fold's
        // inner_matches can unify with the other operand (`self.path or
        // "/"` — Option<String> and a &str literal).
        let inner = if field_ty.as_ref().is_some_and(|t| matches!(t, crate::TypeInfo::Option(inner) if matches!(**inner, crate::TypeInfo::String))) {
            crate::TypeInfo::String
        } else if field_ty.as_ref().is_some_and(|t| matches!(t, crate::TypeInfo::Option(inner) if matches!(**inner, crate::TypeInfo::Int))) {
            crate::TypeInfo::Int
        } else if field_ty.as_ref().is_some_and(|t| matches!(t, crate::TypeInfo::Option(inner) if matches!(**inner, crate::TypeInfo::Float))) {
            crate::TypeInfo::Float
        } else if field_ty.as_ref().is_some_and(|t| matches!(t, crate::TypeInfo::Option(inner) if matches!(**inner, crate::TypeInfo::Bool))) {
            crate::TypeInfo::Bool
        } else {
            crate::TypeInfo::PyObject
        };
        crate::TypeInfo::Option(Box::new(inner))
    } else {
        inferred
    }
}

/// Re-render the truthy operand of an Option fold with the Option NAME
/// narrowed to its INNER type, so references to it read the unwrapped
/// value: `ca_certs and os.path.expanduser(ca_certs)` must pass the
/// STRING to expanduser, not the Option (round 48). Fires only when the
/// Option operand is a NAME (the narrowed binding the inner read needs)
/// and the operand actually references it; otherwise the plain render
/// stands (the Option-pass-through fallback, loud in rustc when wrong).
fn narrow_option_operand(
    option_operand: &crate::ExprType,
    other_operand: &crate::ExprType,
    inner: &crate::TypeInfo,
    ctx: &crate::CodeGenContext,
    options: &crate::PythonOptions,
    symbols: &crate::SymbolTableScopes,
) -> Option<TokenStream> {
    let crate::ExprType::Name(n) = option_operand else {
        return None;
    };
    if !crate::expr_references(other_operand, &n.id) {
        return None;
    }
    let mut narrowed = options.narrowed_names.as_ref().clone();
    // Narrow to the OPTION itself: the narrowed_names match has no
    // explicit Option arm, so the read falls to the `clone().unwrap()`
    // default — exactly the Option-inner unwrap the fold's truthy arm
    // needs (`expanduser(ca_certs)` receives the STRING, never the
    // Option; the Option-slot is unwrapped at the read).
    narrowed.insert(n.id.clone(), crate::TypeInfo::Option(Box::new((*inner).clone())));
    let mut narrowed_options = options.clone();
    narrowed_options.narrowed_names = std::rc::Rc::new(narrowed);
    other_operand
        .clone()
        .to_rust(ctx.clone(), narrowed_options, symbols.clone())
        .ok()
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
