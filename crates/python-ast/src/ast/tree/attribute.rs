use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes, extraction_failure,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
//#[pyo3(transparent)]
pub struct Attribute {
    pub value: Box<ExprType>,
    pub attr: String,
    ctx: String,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Attribute {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = ob
            .getattr("value")
            .map_err(|e| extraction_failure("Attribute.value", &ob, e))?;
        let attr = ob
            .getattr("attr")
            .map_err(|e| extraction_failure("Attribute.attr", &ob, e))?;
        let ctx = ob
            .getattr("ctx")
            .map_err(|e| extraction_failure("attribute context", &ob, e))?
            .get_type()
            .name()
            .map_err(|e| extraction_failure("attribute context type", &ob, e))?;
        Ok(Attribute {
            value: Box::new(
                value
                    .extract()
                    .map_err(|e| extraction_failure("Attribute.value", &ob, e))?,
            ),
            attr: attr
                .extract()
                .map_err(|e| extraction_failure("Attribute.attr", &ob, e))?,
            ctx: ctx.to_string(),
        })
    }
}

impl<'a> CodeGen for Attribute {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // A Rust-module attribute (`crc32c.crc32c` where `crc32c` was
        // `import`ed from a rython.toml binding) is a path into the bound
        // crate — never a field access. The crate name comes from the spec
        // so aliased imports (`import crc32c as c`) still emit the real
        // crate path.
        if let ExprType::Name(root) = self.value.as_ref() {
            let module_spec = match symbols.get(&root.id) {
                Some(crate::SymbolTableNode::Alias(canonical)) => {
                    symbols.get(canonical).and_then(|s| match s {
                        crate::SymbolTableNode::RustModule(spec) => Some(spec.clone()),
                        _ => None,
                    })
                }
                Some(crate::SymbolTableNode::RustModule(spec)) => Some(spec.clone()),
                _ => None,
            };
            if let Some(spec) = module_spec {
                let crate_ident = crate::safe_ident(&spec.crate_name.replace('-', "_"));
                let attr = crate::safe_ident(&self.attr);
                return Ok(quote!(#crate_ident::#attr));
            }
        }
        // If a user binding shadows the root name (e.g. a variable named
        // `re`), the attribute is a field access on that value — never a
        // stdlib module path. Computed before `self.value` is moved below.
        let root_shadowed = crate::ast::tree::call::root_name(&self.value)
            .is_some_and(|root| crate::module_name_shadowed(root, &symbols));
        let value_tokens = self.value.to_rust(ctx, options, symbols)?;
        let value_str = value_tokens.to_string();
        let attr = crate::safe_ident(&self.attr);

        // Determine if this is a module access or a field/method access
        // Module names are typically lowercase and match Python stdlib modules
        // `np`/`numpy` cover the numpy module (import numpy as np lowers to
        // `use stdpython::numpy as np`, making np a real Rust path).
        let is_module_access = !root_shadowed
            && matches!(
                value_str.as_str(),
                "sys" | "os" | "subprocess" | "json" | "urllib" | "xml" | "asyncio" |
            "time" | "math" | "random" | "heapq" | "functools" | "textwrap" | "itertools" | "re" | "hashlib" | "csv" | "io" |
            // `datetime` covers both the runtime module and the datetime
            // TYPE from `from datetime import datetime` — either way the
            // attribute is a path item (datetime::strptime, datetime::now),
            // never a field on a value.
            "datetime" |
            "numpy" | "np" |
            "os :: path" | "os::path" | // for nested modules
            "numpy :: linalg" | "np :: linalg" | "numpy::linalg" | "np::linalg" // np.linalg.inv
            );

        if is_module_access {
            // Use :: for module access (Python's sys.executable becomes sys::executable)
            // Special handling for LazyLock static variables that need
            // dereferencing. os::environ is NOT here: it is a live-view
            // unit struct whose methods auto-ref.
            let needs_deref = matches!(
                (value_str.as_str(), self.attr.as_str()),
                ("sys", "executable") | ("sys", "argv")
            );

            if needs_deref {
                // Wrap dereferenced values in parentheses to ensure correct precedence
                // This prevents *sys::executable.to_string() and ensures (*sys::executable).to_string()
                Ok(quote!((*#value_tokens::#attr)))
            } else {
                Ok(quote!(#value_tokens::#attr))
            }
        } else {
            // Use . for field/method access (Python's obj.field becomes obj.field)
            Ok(quote!(#value_tokens.#attr))
        }
    }
}
