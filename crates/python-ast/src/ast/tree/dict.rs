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
        // Type-aware dict lowering: infer the key/value types across the
        // literal and coerce each pair to them, so `{1: 'a', 2: b}`
        // becomes PyDict<i64, String> with the string literal owned.
        let mut k_expected = crate::TypeInfo::PyObject;
        let mut v_expected = crate::TypeInfo::PyObject;
        let mut k_distinct: Vec<crate::TypeInfo> = Vec::new();
        let mut v_distinct: Vec<crate::TypeInfo> = Vec::new();
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
        if k_distinct.len() > 1 && matches!(k_expected, crate::TypeInfo::PyObject) {
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
        if v_distinct.len() > 1 && matches!(v_expected, crate::TypeInfo::PyObject) {
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
        let k_expected = if matches!(k_expected, crate::TypeInfo::PyObject) {
            None
        } else {
            Some(k_expected)
        };
        let v_expected = if matches!(v_expected, crate::TypeInfo::PyObject) {
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
                    // `{**other}` has no direct HashMap-literal equivalent;
                    // fail the conversion rather than emitting invalid Rust.
                    return Err("dictionary unpacking (`{**other}`) is not yet supported \
                                in dict literals"
                        .to_string()
                        .into());
                }
            }
        }
        
        // PyDict (an insertion-ordered map) rather than HashMap: Python
        // dicts preserve insertion order, and keys()/items()/iteration
        // must match.
        Ok(quote! {
            PyDict::from([#(#pairs),*])
        })
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