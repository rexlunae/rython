use proc_macro2::TokenStream;
use pyo3::{Borrowed, PyAny, FromPyObject, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{extraction_failure, CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
//#[pyo3(transparent)]
pub struct Attribute {
    value: Box<ExprType>,
    attr: String,
    ctx: String,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Attribute {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = ob.getattr("value").map_err(|e| extraction_failure("Attribute.value", &ob, e))?;
        let attr = ob.getattr("attr").map_err(|e| extraction_failure("Attribute.attr", &ob, e))?;
        let ctx = ob
            .getattr("ctx")
            .map_err(|e| extraction_failure("attribute context", &ob, e))?
            .get_type()
            .name()
            .map_err(|e| extraction_failure("attribute context type", &ob, e))?;
        Ok(Attribute {
            value: Box::new(value.extract().map_err(|e| extraction_failure("Attribute.value", &ob, e))?),
            attr: attr.extract().map_err(|e| extraction_failure("Attribute.attr", &ob, e))?,
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
        let value_tokens = self.value.to_rust(ctx, options, symbols)?;
        let value_str = value_tokens.to_string();
        let attr = crate::safe_ident(&self.attr);
        
        // Determine if this is a module access or a field/method access
        // Module names are typically lowercase and match Python stdlib modules
        let is_module_access = matches!(value_str.as_str(), 
            "sys" | "os" | "subprocess" | "json" | "urllib" | "xml" | "asyncio" |
            "os :: path" | "os::path" // for nested modules
        );
        
        if is_module_access {
            // Use :: for module access (Python's sys.executable becomes sys::executable)
            // Special handling for LazyLock static variables that need dereferencing
            let needs_deref = matches!((value_str.as_str(), self.attr.as_str()), 
                ("sys", "executable") | ("sys", "argv") | ("os", "environ")
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
