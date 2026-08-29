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
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Handle dotted names (like "os.path") by converting them to Rust module paths
        if self.id.contains('.') {
            let parts: Vec<&str> = self.id.split('.').collect();
            let idents: Vec<_> = parts.iter().map(|part| crate::safe_ident(part)).collect();
            Ok(quote!(#(#idents)::*))
        } else {
            // Issue #223: a bare reference to a MONOMORPHIZED function
            // (`map(yield_lines, xs)` — the name as a value, not a call).
            //
            // Only the morphs (`f_str`, `f_any`, ...) and, when one could
            // be planned, the dynamic router carry the original name. With
            // no router there is nothing for this reference to resolve to,
            // and emitting the bare ident produced a dangling name that
            // surfaced as an unexplained E0425 in the generated crate.
            // Fail here instead, naming the construct — the value-path twin
            // of the guard call.rs raises for a call that cannot dispatch.
            //
            // A static CALL never reaches this: call.rs emits the mangled
            // morph and returns before rendering the callee name.
            if let Some(spec) = options.specialized_fns.get(&self.id)
                && spec.router.is_none()
            {
                return Err(format!(
                    "`{0}` is used as a value, but it is monomorphized over its \
                     isinstance-dispatched parameter(s) and no dynamic router could \
                     be planned for it (a morph's return type could not be derived, \
                     or a non-axis parameter lacks a concrete annotation), so the \
                     name exists only as its per-type morphs; annotate the \
                     parameter(s) and return type, or call `{0}` directly rather \
                     than passing it as a value",
                    self.id
                )
                .into());
            }
            // A CLASS NAME read as a VALUE (`[ChecksumError]` — a bare
            // name in a container, comparison, or argument position):
            // classes as values lower to their NAME STRINGS (round 33
            // design — the exception model is string-tagged). The raw
            // Python spelling, matching what the raise side emits.
            if matches!(symbols.get(&self.id), Some(crate::SymbolTableNode::ClassDef(_)))
                || matches!(symbols.get(&self.id), Some(crate::SymbolTableNode::Alias(c))
                    if matches!(symbols.get(c), Some(crate::SymbolTableNode::ClassDef(_))))
                || (matches!(symbols.get(&self.id), Some(crate::SymbolTableNode::ImportFrom(_)))
                    && crate::resolve_class_referenced(&self.id, &symbols, &options).is_some())
            {
                let name = self.id.clone();
                return Ok(quote!(#name.to_string()));
            }
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
            // Issue #115: a `global`-written module value is a MUTABLE
            // static (`static name: Mutex<T>`); every read locks and clones
            // through the helper — the guard drops inside py_global_read,
            // so two reads in one statement never deadlock. Names that a
            // function binds WITHOUT `global` never enter mutable_statics
            // (module.rs disqualifies them), so a bare read here is always
            // the module global, as in Python.
            if let Some(kind) = options.mutable_statics.get(&self.id) {
                let global_ref = kind.static_ref(&name);
                // Issue #189: a class-instance global's VALUE read is the
                // INSTANCE — the Option is the static's representation, and
                // reading while None is a loud runtime panic (§12.2). `is
                // None` compares read the Option instead (compare.rs).
                if let crate::MutableGlobalKind::Class { .. } = kind {
                    let msg =
                        format!("module global `{}` read while None (issue #189)", self.id);
                    return Ok(quote!(
                        stdpython::py_global_read(#global_ref).expect(#msg)
                    ));
                }
                return Ok(quote!(stdpython::py_global_read(#global_ref)));
            }
            // A module-level value promoted to a LazyLock static (module.rs):
            // a static does not auto-deref in value/borrow position (only as
            // a method receiver), so every READ clones the deref'd value.
            if options.promoted_statics.contains(&self.id) {
                return Ok(quote!((*#name).clone()));
            }
            // A name IMPORTED from a sibling module where it is a promoted
            // static (`from .constant import _THAI` — the import brings the
            // LazyLock static into scope): reads deref-clone the same way.
            // Resolve the ImportFrom to its defining module and consult the
            // shared promotion map (module.rs's `module_promoted_static_names`)
            // so the read matches the static the defining module emitted.
            // Follows alias chains (`from .constant import _THAI as T` —
            // the symbol for T is Alias("_THAI"), which resolves to the
            // ImportFrom).
            let mut sym = symbols.get(&self.id).cloned();
            let mut hops = 0;
            while let Some(crate::SymbolTableNode::Alias(target)) = &sym {
                if hops > 16 {
                    break;
                }
                sym = symbols.get(target).cloned();
                hops += 1;
            }
            if let Some(crate::SymbolTableNode::ImportFrom(ifm)) = sym {
                let path = ifm.resolved_module_path(&options);
                // The ssl version constants are LazyLock statics (see
                // attribute.rs `needs_deref`): a `from ssl import
                // OPENSSL_VERSION` binding (urllib3's util/ssl_.py) reads
                // as the plain &str / i64 / tuple value, not the static.
                // Both ssl backends expose the same LazyLock shape.
                if path.len() == 1
                    && path[0] == "ssl"
                    && matches!(
                        ifm.names
                            .iter()
                            .find(|a| a.asname.as_deref() == Some(self.id.as_str()))
                            .map(|a| a.name.as_str())
                            .unwrap_or(self.id.as_str()),
                        "OPENSSL_VERSION" | "OPENSSL_VERSION_NUMBER" | "OPENSSL_VERSION_INFO"
                    )
                {
                    return Ok(quote!((*#name).clone()));
                }
                if options.module_defs.contains_key(&path) {
                    // The canonical name in the defining module (the alias
                    // target when this name is an asname binding).
                    let canonical = ifm
                        .names
                        .iter()
                        .find(|a| a.asname.as_deref() == Some(self.id.as_str()))
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| self.id.clone());
                    if crate::ast::tree::module::module_promoted_static_names(
                        &options, &path,
                    )
                    .contains(&canonical)
                    {
                        return Ok(quote!((*#name).clone()));
                    }
                }
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
            // The builtin `NotImplemented` singleton (`return NotImplemented`
            // in `__eq__` fallbacks — requests' structures, urllib3's
            // collections): the comparison sentinel — a boxed None
            // (rython's comparisons return bool; the sentinel has no
            // analogue — documented divergence).
            if self.id == "NotImplemented" && symbols.get("NotImplemented").is_none() {
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // A name imported from an EXTERNAL module (`from ssl import
            // CERT_REQUIRED` — urllib3's ssl_.py) read as a VALUE: the
            // import has no runtime item, so the read lowers to the boxed
            // None (external-module divergence, the same model call.rs and
            // attribute.rs use for external imports).
            if crate::ast::tree::import::resolves_to_external_import(
                &self.id,
                &options,
                &symbols,
            ) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`{}` is dropped: it is imported from a module that is \
                     external to the generated crate (external-module divergence)",
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
