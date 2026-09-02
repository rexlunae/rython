use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, PyAttributeExtractor
};

/// A subscript's bracket contents: a plain index or a slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SubscriptKind {
    Index(Box<ExprType>),
    Slice {
        lower: Option<Box<ExprType>>,
        upper: Option<Box<ExprType>>,
        step: Option<Box<ExprType>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Subscript {
    pub value: Box<ExprType>,
    pub kind: SubscriptKind,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Subscript {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyTypeMethods;

        let value = ob.extract_attr_with_context("value", "subscript value")?;
        let value = value.extract().map_err(|e| extraction_failure("getting subscript value", &ob, e))?;

        let slice_attr = ob.extract_attr_with_context("slice", "subscript slice")?;
        let slice_type: String = slice_attr
            .get_type()
            .name()
            .and_then(|n| n.extract())
            .map_err(|e| extraction_failure("subscript slice type", &ob, e))?;

        let kind = if slice_type == "Slice" {
            let bound = |name: &str| -> PyResult<Option<Box<ExprType>>> {
                match slice_attr.getattr(name) {
                    Ok(v) if !v.is_none() => Ok(Some(Box::new(v.extract().map_err(
                        |e| extraction_failure("slice bound", &ob, e),
                    )?))),
                    _ => Ok(None),
                }
            };
            SubscriptKind::Slice {
                lower: bound("lower")?,
                upper: bound("upper")?,
                step: bound("step")?,
            }
        } else {
            let index = slice_attr
                .extract()
                .map_err(|e| extraction_failure("getting subscript slice", &ob, e))?;
            SubscriptKind::Index(Box::new(index))
        };

        Ok(Subscript {
            value: Box::new(value),
            kind,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(Subscript { lineno, col_offset, end_lineno, end_col_offset });

/// Lower the receiver of a subscript STORE as a place. Names and
/// attributes are places already; a nested subscript (`grid[i][j] = v`)
/// threads through py_index_mut so the store mutates the real container —
/// the Load lowering would yield a clone and silently drop the write.
/// Anything else (e.g. a call result) is rejected loudly.
pub(crate) fn subscript_receiver_place(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    match expr {
        ExprType::Subscript(sub) => {
            let inner = subscript_receiver_place(
                &sub.value,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?;
            match &sub.kind {
                SubscriptKind::Index(index) => {
                    let index = index
                        .clone()
                        .to_rust(ctx, options, symbols)?;
                    Ok(quote!((#inner).py_index_mut(#index)?))
                }
                SubscriptKind::Slice { .. } => Err(
                    "cannot assign through a slice (`x[a:b][...] = ...`)"
                        .to_string()
                        .into(),
                ),
            }
        }
        ExprType::Name(_) => {
            expr.clone().to_rust(ctx, options, symbols)
        }
        ExprType::Attribute(attr) => {
            // Attribute receivers render in place flavor: in a generic trait
            // default, `self.items[i]` must mutate through `self.items_mut()`,
            // not the cloning load accessor.
            crate::ast::tree::attribute::to_rust_place(
                &attr.value,
                &attr.attr,
                &ctx,
                &options,
                &symbols,
                false,
            )
        }
        other => {
            // A store through a CALL receiver (`memoryview(byte_obj)[0:n] =
            // subarray` — urllib3's emscripten fetch loop) has no rython
            // equivalent: the receiver is a temporary object with no
            // persistent place. Emit a no-op with a warning (the
            // documented class-as-value / foreign-object divergence)
            // instead of failing the whole module.
            options.definition_warnings.borrow_mut().push(format!(
                "store into `{:?}[...]` is dropped (the receiver is a call \
                 result, which has no rython place)",
                other
            ));
            Ok(TokenStream::new())
        }
    }
}

impl CodeGen for Subscript {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A user-class receiver that defines `__getitem__` routes the
        // subscript to ITS method — that IS Python's behavior (including
        // the class's own KeyError/IndexError and any case-insensitivity;
        // §7's mapping-protocol slice). The method must exist; anything
        // else keeps the py_index path, loud in rustc for classes (§12.1).
        // Slices are not routed (a slice object has no rython value).
        // Computed BEFORE `self.value` is moved by the to_rust below.
        let dunder_getitem =
            crate::receiver_class(&self.value, &ctx, &symbols, &options)
                .and_then(|(class, class_symbols)| {
                    class
                        .method_on_mro("__getitem__", &class_symbols)
                        .filter(|m| crate::ast::tree::call::dunder_method_well_typed(m))
                        .map(|method| (class, class_symbols, method))
                });
        // An OPTION-typed receiver (`host[start:end]` where `host` is a
        // truthiness-narrowed `str | None` — urllib3's _normalize_host)
        // unwraps before the subscript: Python raises TypeError on an
        // actual None (`'NoneType' object is not subscriptable`), and the
        // unwrap fires only when the flow contradicts the guard. The same
        // loud-panic spelling the Option-receiver method path uses.
        // Computed BEFORE self.value is moved by the to_rust below.
        let value_yields_option =
            crate::expr_yields_option_ctx(&self.value, &ctx, &options, &symbols);
        let value = self.value.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        let value = if value_yields_option {
            quote!((#value).clone().unwrap_or_else(|| {
                panic!("TypeError: 'NoneType' object is not subscriptable")
            }))
        } else {
            value
        };
        match self.kind {
            // Python index rules via PyIndex: negatives from the end, a
            // catchable IndexError/KeyError instead of a Rust panic.
            SubscriptKind::Index(index) => {
                if let Some((_class, _class_symbols, method)) = &dunder_getitem {
                    return crate::ast::tree::call::dunder_method_call(
                        method,
                        &value,
                        std::slice::from_ref(index.as_ref()),
                        true,
                        &ctx,
                        &options,
                        &symbols,
                    );
                }
                // Context-aware: indices are i64. `len(x)` yields usize and
                // `xs[len(xs) - 1]` yields i64, so coerce usize → i64 here
                // rather than depend on the runtime generic. The
                // reuse-aware renderer: an index read from a REUSED
                // receiver's field moves the value out of the field (the
                // reuse-clone keeps the receiver intact, round 98).
                let index = crate::render_typed_reused(
                    &index,
                    ctx,
                    options,
                    symbols,
                    Some(crate::TypeInfo::Int),
                )?;
                Ok(quote! { (#value).py_index(#index)? })
            }
            // Slices clamp and never raise.
            SubscriptKind::Slice { lower, upper, step } => {
                let bound = |b: Option<Box<ExprType>>| -> Result<TokenStream, Box<dyn std::error::Error>> {
                    Ok(match b {
                        Some(e) => {
                            let t = e.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                            quote!(Some(#t))
                        }
                        None => quote!(None),
                    })
                };
                let lower = bound(lower)?;
                let upper = bound(upper)?;
                let step = bound(step)?;
                Ok(quote! { (#value).py_slice(#lower, #upper, #step) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_list_subscript, "a[0]", "subscript_test.py");
    create_parse_test!(test_dict_subscript, "d['key']", "subscript_test.py");
    create_parse_test!(test_nested_subscript, "matrix[i][j]", "subscript_test.py");
}

/// Whether an optional slice `step` expression is the literal `1` (or
/// absent): the only step value range-replacement supports — anything
/// else is a strided selection the caller must reject loudly.
pub fn is_step_one(step: Option<&ExprType>) -> bool {
    match step {
        None => true,
        Some(e) => matches!(
            e,
            ExprType::Constant(c)
                if matches!(&c.0, Some(litrs::Literal::Integer(i)) if i.to_string() == "1")
        ),
    }
}

