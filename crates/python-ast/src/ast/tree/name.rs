use proc_macro2::TokenStream;
use pyo3::{FromPyObject, PyErr};
use quote::quote;

use crate::{CodeGen, CodeGenContext, IsIdentifier, PythonOptions, SymbolTableScopes};

use serde::{Deserialize, Serialize};

/// Identifiers represent valid Python identifiers.
#[derive(Clone, Debug, Default, Eq, FromPyObject, Hash, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Identifier(String);

impl TryFrom<&str> for Identifier {
    type Error = PyErr;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.isidentifier()? {
            Ok(Identifier(value.to_string()))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNameError, _>(format!(
                "Invalid Identifier: {}",
                String::from(value)
            )))
        }
    }
}

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

impl Into<String> for Identifier {
    fn into(self) -> String {
        self.0
    }
}

/// Names are Python identifiers, separated by '.'
#[derive(Clone, Debug, Default, Eq, FromPyObject, Hash, PartialEq, Serialize, Deserialize)]
pub struct Name {
    pub id: String,
}

impl TryFrom<&str> for Name {
    type Error = PyErr;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let parts = s.split('.');
        tracing::debug!("parts: {:?}", parts);

        let mut v = Vec::new();
        for part in parts {
            let ident = Identifier::try_from(part)?;
            v.push(String::from(ident.as_ref()));
        }

        Ok(Name { id: v.join(".") })
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        self.id.as_str()
    }
}

impl Into<String> for Name {
    fn into(self) -> String {
        self.id
    }
}

impl CodeGen for Name {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        _ctx: Self::Context,
        options: Self::Options,
        _symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Handle dotted names (like "os.path") by converting them to Rust module paths
        if self.id.contains('.') {
            let parts: Vec<&str> = self.id.split('.').collect();
            let idents: Vec<_> = parts.iter().map(|part| crate::safe_ident(part)).collect();
            Ok(quote!(#(#idents)::*))
        } else {
            let name = crate::safe_ident(&self.id);
            // Issue #125: a name narrowed by `if x is not None:` (or by an
            // if/else whose branches both leave x non-None) still holds an
            // Option at runtime — the binding is hoisted once. Every READ
            // must unwrap it: Python's value IS the inner value in the
            // narrowed region. clone() keeps the read non-consuming so the
            // name stays usable (the hoisted binding is reused).
            // Issue #121: a `str | bytes` union narrowed by isinstance
            // reads the concrete branch (String via as_str, Vec<u8> via
            // as_bytes) — a runtime conversion, not an unwrap.
            if let Some(target) = options.narrowed_names.get(&self.id) {
                return Ok(match target {
                    crate::TypeInfo::StrOrBytes => quote!((#name)),
                    crate::TypeInfo::String | crate::TypeInfo::StrRef => {
                        quote!((#name).as_str().unwrap().to_string())
                    }
                    crate::TypeInfo::Bytes => {
                        quote!((#name).as_bytes().unwrap().to_vec())
                    }
                    // Issue #121: a boxed PyValue narrowed by isinstance
                    // reads the concrete member via the PyValue accessors
                    // (as_int/as_str/as_tuple...), a runtime conversion.
                    crate::TypeInfo::PyValueMember(inner) => match inner.as_ref() {
                        crate::TypeInfo::Int => quote!((#name).as_int().unwrap()),
                        crate::TypeInfo::Float => quote!((#name).as_float().unwrap()),
                        crate::TypeInfo::Bool => quote!((#name).as_bool().unwrap()),
                        crate::TypeInfo::String | crate::TypeInfo::StrRef => {
                            quote!((#name).as_str().unwrap().to_string())
                        }
                        crate::TypeInfo::Bytes => {
                            quote!((#name).as_bytes().unwrap().to_vec())
                        }
                        // A narrowed tuple member (isinstance(x, tuple))
                        // reads as the element vector; indexing and len
                        // then operate on it.
                        crate::TypeInfo::Vec(_) => {
                            quote!((#name).as_tuple().unwrap().clone())
                        }
                        _ => quote!((#name)),
                    },
                    // A PyValue narrowed to itself (e.g. the else of a
                    // compound `isinstance(x, T) and ...`) reads bare.
                    crate::TypeInfo::PyValue => quote!((#name)),
                    _ => quote!((#name).clone().unwrap()),
                });
            }
            // A module-level value promoted to a LazyLock static (module.rs):
            // a static does not auto-deref in value/borrow position (only as
            // a method receiver), so every READ clones the deref'd value.
            if options.promoted_statics.contains(&self.id) {
                return Ok(quote!((*#name).clone()));
            }
            // A CALLABLE name read as a VALUE (`hash_utf8 = sha256_utf8` —
            // requests' auth, where sha256_utf8 is a dropped nested
            // function): the callable-as-value divergence — the read lowers
            // to the boxed None.
            if options.value_callables.contains(&self.id)
            {
                options.definition_warnings.borrow_mut().push(format!(
                    "callable `{}` read as a value lowers to the boxed None \
                     (the callable-as-value divergence, issue #122)",
                    self.id
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            Ok(quote!(#name))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_name_works() {
        let name = Name::try_from("this.symbol");
        assert!(name.is_ok());
    }

    #[test]
    fn bad_name_works() {
        let name = Name::try_from("this.0symbol");
        assert!(name.is_err());
    }
}
