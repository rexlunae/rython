use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IfExp {
    pub test: Box<ExprType>,
    pub body: Box<ExprType>,
    pub orelse: Box<ExprType>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for IfExp {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let test = ob.extract_attr_with_context("test", "if expression test")?;
        let body = ob.extract_attr_with_context("body", "if expression body")?;
        let orelse = ob.extract_attr_with_context("orelse", "if expression orelse")?;
        
        let test = test.extract().map_err(|e| extraction_failure("getting if expression test", &ob, e))?;
        let body = body.extract().map_err(|e| extraction_failure("getting if expression body", &ob, e))?;
        let orelse = orelse.extract().map_err(|e| extraction_failure("getting if expression orelse", &ob, e))?;
        
        Ok(IfExp {
            test: Box::new(test),
            body: Box::new(body),
            orelse: Box::new(orelse),
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(IfExp { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for IfExp {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let test =
            crate::condition_to_rust(&self.test, ctx.clone(), options.clone(), symbols.clone())?;
        // `x if r is not None else y` — the ternary's TRUE branch is the
        // narrowed scope (the same narrowing an if-statement applies to
        // its body): `r.encoding if r is not None else None` reads the
        // unwrapped field in the true branch, and a compound test narrows
        // every `is not None` conjunct it contains (`r.language if r is
        // not None and r.language != "Unknown" else ""` —
        // charset_normalizer's legacy.detect, round 99). Any other test
        // narrows nothing.
        let mut body_options = options.clone();
        let narrowings = crate::narrowings_from_test(&self.test, &options);
        for (narrowed, inner) in &narrowings {
            let mut narrowed_names = body_options.narrowed_names.as_ref().clone();
            let target = inner.clone().unwrap_or(crate::TypeInfo::StrOrBytes);
            narrowed_names.insert(narrowed.clone(), target);
            body_options.narrowed_names = std::rc::Rc::new(narrowed_names);
            if let Some(inner) = inner {
                let mut name_types = body_options.name_types.as_ref().clone();
                name_types.insert(narrowed.clone(), inner.clone());
                body_options.name_types = std::rc::Rc::new(name_types);
            }
        }
        // The ternary's VALUE TYPE follows its branches: when the test
        // narrows an optional name, the true branch reads the UNWRAPPED
        // inner (concrete) value, so the else decides the Option-ness —
        // `r.encoding if r is not None else None` is Option<String> (the
        // true branch must Some-wrap; a bare `String`-vs-`None` if/else is
        // E0308), while `r.language if r is not None and ... else ""` is a
        // plain String whose else literal must be OWNED (a `String`-vs-`&str`
        // if/else is E0308) — charset_normalizer's legacy.detect, round 99.
        let body_reads_narrowed = !narrowings.is_empty() && crate::ast::tree::visit::any_expr(
            &self.body,
            |e| {
                matches!(e, crate::ExprType::Name(n) if narrowings.iter().any(|(name, _)| *name == n.id))
            },
        );
        // The true branch's inferred type IN the narrowed scope, computed
        // once (before the body's to_rust moves the narrowed options): the
        // Some-wrap decision must not double-wrap a branch that is ALREADY
        // Option (`maybe(x) if x is not None else None` where maybe
        // returns `-> Optional[T]` — a returned None would become
        // Some(None), which is not None for a later `is None` check), and
        // the string-else ownership applies only to a String branch.
        let body_ty = if body_reads_narrowed {
            crate::infer_type(Some(&ctx), &self.body, &body_options, &symbols)
        } else {
            crate::TypeInfo::PyObject
        };
        let body_is_concrete = !matches!(
            body_ty,
            crate::TypeInfo::Option(_)
                | crate::TypeInfo::PyObject
                | crate::TypeInfo::PyValue
                | crate::TypeInfo::PyValueMember(_)
        );
        let body_is_string = matches!(
            body_ty,
            crate::TypeInfo::String | crate::TypeInfo::StrRef | crate::TypeInfo::StrOrBytes
        );
        // Captured before the orelse is moved by the to_rust below.
        let orelse_is_string_literal = matches!(
            self.orelse.as_ref(),
            crate::ExprType::Constant(c)
                if matches!(&c.0, Some(litrs::Literal::String(_)))
        );
        let orelse_is_none = crate::is_none_expr(&self.orelse);
        let body = self
            .body
            .to_rust(ctx.clone(), body_options, symbols.clone())?;
        let body = if body_reads_narrowed && orelse_is_none && body_is_concrete {
            // True branch is the concrete inner value; the ternary is the
            // value-or-None the Python spells: Option(inner). An ALREADY
            // Option-typed branch (an Option-returning callee on the
            // narrowed value) and an UNKNOWN-typed branch keep their shape
            // — wrapping either would double-wrap (`Some(None)` is not
            // None) or hide a later-stage error.
            quote!(Some(#body))
        } else {
            body
        };
        let mut orelse = self.orelse.to_rust(ctx, options, symbols)?;
        if body_reads_narrowed && orelse_is_string_literal && body_is_string {
            orelse = quote!((#orelse).to_string());
        }

        Ok(quote! {
            if #test { #body } else { #orelse }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_if_expression, "x if condition else y", "if_exp_test.py");
    create_parse_test!(test_nested_if_expression, "a if b else c if d else e", "if_exp_test.py");
}