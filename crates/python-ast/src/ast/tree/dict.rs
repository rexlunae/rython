use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes,
    Node, impl_node_with_positions, extract_list
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dict {
    pub keys: Vec<Option<ExprType>>,
    pub values: Vec<ExprType>,
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Dict {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let keys: Vec<Option<ExprType>> = extract_list(&ob, "keys", "dictionary keys")?;
        let values: Vec<ExprType> = extract_list(&ob, "values", "dictionary values")?;
        
        Ok(Dict {
            keys,
            values,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl_node_with_positions!(Dict { lineno, col_offset, end_lineno, end_col_offset });

impl CodeGen for Dict {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut pairs = Vec::new();
        let mut spreads: Vec<TokenStream> = Vec::new();
        // Type-aware dict lowering: infer the key/value types across the
        // literal and coerce each pair to them, so `{1: 'a', 2: b}`
        // becomes PyDict<i64, String> with the string literal owned.
        // Issue #121: a store into a `dict[str, Any]` name forces the
        // value type to the boxed PyValue, so mixed values wrap.
        let mut k_expected = crate::TypeInfo::PyObject;
        let mut v_expected = crate::TypeInfo::PyObject;
        let mut k_distinct: Vec<crate::TypeInfo> = Vec::new();
        let mut v_distinct: Vec<crate::TypeInfo> = Vec::new();
        let forced_kv = options.dict_forced_kv.as_ref().clone();
        for (key, value) in self.keys.iter().zip(self.values.iter()) {
            if let Some(k) = key {
                let kt = crate::infer_type(k, &options, &symbols);
                if !matches!(kt, crate::TypeInfo::PyObject) {
                    if !k_distinct.contains(&kt) {
                        k_distinct.push(kt.clone());
                    }
                    k_expected = crate::unify(k_expected, kt);
                }
            }
            let vt = crate::infer_type(value, &options, &symbols);
            if !matches!(vt, crate::TypeInfo::PyObject) {
                if !v_distinct.contains(&vt) {
                    v_distinct.push(vt.clone());
                }
                v_expected = crate::unify(v_expected, vt);
            }
        }
        // A forced key/value type overrides the mixed-type check: the
        // assignment target dictates the element type.
        if forced_kv.is_none()
            && k_distinct.len() > 1
            && matches!(k_expected, crate::TypeInfo::PyObject)
        {
            let kinds = k_distinct
                .iter()
                .map(|d| d.display())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "dict literal mixes incompatible key types ({kinds}); keys must \
                 share a common type"
            )
            .into());
        }
        if forced_kv.is_none()
            && v_distinct.len() > 1
            && matches!(v_expected, crate::TypeInfo::PyObject)
        {
            // All values are TUPLES of strings of different lengths
            // (`{100: ("continue",), 101: ("switching_protocols",), 103:
            // ("processing", "early-hints"), ...}` — requests' status
            // codes): unify to Vec<String>.
            let all_str_tuples = v_distinct.iter().all(|t| match t {
                crate::TypeInfo::Tuple(ts) => {
                    !ts.is_empty()
                        && ts.iter().all(|e| {
                            matches!(
                                e,
                                crate::TypeInfo::StrRef | crate::TypeInfo::String
                            )
                        })
                }
                _ => false,
            });
            if !all_str_tuples {
                // A HETEROGENEOUS value set that is still BOXABLE
                // (Optional + bool + str... — urllib3's socks_options
                // dict: `socks_version` (Optional), `rdns` (bool)):
                // the values box into the heterogeneous PyValue, matching
                // `dict[str, Any]` lowering (issue #121).
                let boxable = v_distinct.iter().all(|t| match t {
                    crate::TypeInfo::Option(_)
                    | crate::TypeInfo::Bool
                    | crate::TypeInfo::Int
                    | crate::TypeInfo::Float
                    | crate::TypeInfo::String
                    | crate::TypeInfo::StrRef
                    | crate::TypeInfo::Bytes
                    | crate::TypeInfo::PyValue
                    // A container value (`exclude_input: underlying_
                    // operation_members` — a list — boto3's collection
                    // docs): boxed too. A CLASS instance value
                    // (`"handlers": [RichPipStreamHandler(...)]` — pip's
                    // logging dictConfig): boxed too.
                    | crate::TypeInfo::Vec(_)
                    | crate::TypeInfo::Dict(_, _)
                    | crate::TypeInfo::Tuple(_)
                    | crate::TypeInfo::Class(_) => true,
                    _ => false,
                });
                if !boxable {
                    let kinds = v_distinct
                        .iter()
                        .map(|d| d.display())
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "dict literal mixes incompatible value types ({kinds}); values \
                         must share a common type"
                    )
                    .into());
                }
                v_expected = crate::TypeInfo::PyValue;
            }
        }
        let k_expected = if matches!(k_expected, crate::TypeInfo::PyObject) {
            None
        } else {
            // Dict keys are String, not &'static str: literal `"a"` keys
            // must match `dict[str, V]` annotations (IndexMap<String, V>)
            // and survive past the literal's lifetime. PyIndex<&str> and
            // PyIndexMut<&str> impls keep `d["a"]` reads working on
            // String-keyed dicts.
            Some(match k_expected {
                crate::TypeInfo::StrRef => crate::TypeInfo::String,
                other => other,
            })
        };
        let v_expected = if let Some((_, fv)) = forced_kv.clone() {
            Some(fv)
        } else if matches!(v_expected, crate::TypeInfo::PyObject) {
            None
        } else {
            Some(v_expected)
        };

        for (key, value) in self.keys.iter().zip(self.values.iter()) {
            match key {
                Some(k) => {
                    let key_tokens = crate::render_typed(
                        k,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        k_expected.clone(),
                    )?;
                    let value_tokens = crate::render_typed(
                        value,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        v_expected.clone(),
                    )?;
                    pairs.push(quote! { (#key_tokens, #value_tokens) });
                }
                None => {
                    // `{**other}`: a spread merges the other dict's entries.
                    let spread = value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    spreads.push(spread);
                }
            }
        }

        // A spread merges via PyDict::update — the key/value types come
        // from the literal pairs, or from the first spread's own type when
        // the literal is all spreads (`{**a, **b}`). A literal without
        // spreads keeps the plain PyDict::from inference.
        let keys = pairs.iter();
        if spreads.is_empty() {
            // PyDict (an insertion-ordered map) rather than HashMap: Python
            // dicts preserve insertion order, and keys()/items()/iteration
            // must match.
            Ok(quote! {
                PyDict::from([#(#keys),*])
            })
        } else {
            let (k_ty, v_ty) = if let Some((fk, fv)) = forced_kv.clone() {
                (fk.to_rust_type(), fv.to_rust_type())
            } else if let (Some(k), Some(v)) = (k_expected.clone(), v_expected.clone())
                && !pairs.is_empty()
            {
                (k.to_rust_type(), v.to_rust_type())
            } else if let Some(k) = k_expected.clone()
                && !pairs.is_empty()
            {
                // A typed key but an UNKNOWN value (`{**scheme, 'name':
                // self._strip_sig_prefix(...)}` — botocore's regions): the
                // value boxes as PyValue.
                (k.to_rust_type(), quote!(stdpython::PyValue))
            } else if let Some(first) = self
                    .keys
                    .iter()
                    .zip(self.values.iter())
                    .find(|(k, _)| k.is_none())
                    .map(|(_, v)| v)
                    && let crate::TypeInfo::Dict(k, v) =
                        crate::infer_type(first, &options, &symbols)
                {
                    (k.to_rust_type(), v.to_rust_type())
                } else {
                    // An all-spread literal with no statically-known types
                    // (`{**request_dict['context'].get('signing', {}),
                    // **signing_context}` — botocore's utils): a boxed
                    // PyDict<String, PyValue> (the boxed-dict divergence).
                    (quote!(String), quote!(stdpython::PyValue))
                };
            Ok(quote! {
                {
                    let mut __rython_dict = PyDict::<#k_ty, #v_ty>::from([#(#keys),*]);
                    #( __rython_dict.update(#spreads); )*
                    __rython_dict
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_parse_test;

    create_parse_test!(test_empty_dict, "{}", "dict_test.py");
    create_parse_test!(test_simple_dict, "{'a': 1, 'b': 2}", "dict_test.py");
    create_parse_test!(test_dict_with_variables, "{x: y, z: w}", "dict_test.py");
}