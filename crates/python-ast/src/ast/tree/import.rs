use proc_macro2::TokenStream;
use pyo3::FromPyObject;
use quote::quote;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{CodeGen, CodeGenContext, PythonOptions, SymbolTableNode, SymbolTableScopes};

/// Python stdlib modules that the stdpython runtime crate provides. Imports
/// of these resolve under the runtime crate; anything else is assumed to be
/// a sibling module of the generated crate.
pub(crate) fn is_stdpython_module(name: &str) -> bool {
    matches!(
        name,
        "os" | "sys"
            | "re"
            | "io"
            | "argparse"
            | "json"
            | "math"
            | "random"
            | "datetime"
            | "time"
            | "collections"
            | "itertools"
            | "functools"
            | "heapq"
            | "copy"
            | "textwrap"
            | "hashlib"
            | "csv"
            | "glob"
            | "pathlib"
            | "tempfile"
            | "subprocess"
            | "string"
            | "sysconfig"
            | "venv"
            | "numpy"
    )
}

/// Runtime modules that only exist on stdpython's std tier: they touch the
/// OS (or, for math, std's float intrinsics), so the no_std profile has
/// nothing to lower them to. json/string/collections/itertools live on the
/// alloc tier and stay importable.
pub(crate) fn is_std_only_module(name: &str) -> bool {
    matches!(
        name,
        "os" | "sys"
            | "re"
            | "io"
            | "argparse"
            | "math"
            | "random"
            | "datetime"
            | "time"
            | "glob"
            | "pathlib"
            | "tempfile"
            | "subprocess"
            | "sysconfig"
            | "venv"
            | "numpy"
    )
}

/// The conversion-time error for a std-tier import under the no_std
/// profile. Failing here beats failing later with an unresolved-name error
/// in the generated crate.
fn std_only_import_error(module: &str) -> Box<dyn std::error::Error> {
    format!(
        "`import {}` requires stdpython's std tier (it needs the OS), which the \
         no_std profile does not provide; remove the import or convert without \
         the no_std profile",
        module
    )
    .into()
}

#[derive(Clone, Debug, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct Alias {
    pub name: String,
    pub asname: Option<String>,
}

#[derive(Clone, Debug, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct Import {
    pub names: Vec<Alias>,
}

/// An Import (or FromImport) statement causes 2 things to occur:
/// 1. Declares the imported object within the existing scope.
/// 2. Causes the referenced module to be compiled into the program (only once).

impl CodeGen for Import {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        for alias in self.names.iter() {
            symbols.insert(alias.name.clone(), SymbolTableNode::Import(self.clone()));
            if let Some(a) = alias.asname.clone() {
                symbols.insert(a, SymbolTableNode::Alias(alias.name.clone()))
            }
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        mut symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut tokens = TokenStream::new();
        for alias in self.names.iter() {
            // `import rython` is not a runtime module: rust.bind declarations
            // are compile-time-only. The from-import spelling is required so
            // the declaration syntax stays explicit.
            if alias.name == "rython" || alias.name.starts_with("rython.") {
                return Err("`import rython` is not supported; use \
                            `from rython import rust` for compile-time Rust bindings"
                    .to_string()
                    .into());
            }
            // Import of a Rust module: the import is compile-time-only — the
            // name resolves through the symbol table; no `use` is emitted
            // (the crate is a dependency, not a sibling module). Registering
            // the module symbol lets attribute call lowering find it.
            if let Some(spec) = options.rust_modules.get(&alias.name) {
                let module_symbol = crate::SymbolTableNode::RustModule(spec.clone());
                symbols.insert(alias.name.clone(), module_symbol);
                if let Some(asname) = &alias.asname {
                    symbols.insert(
                        asname.clone(),
                        crate::SymbolTableNode::Alias(alias.name.clone()),
                    );
                }
                tokens.extend(TokenStream::new());
                continue;
            }
            if options.no_std {
                let root = alias.name.split('.').next().unwrap_or(&alias.name);
                if is_std_only_module(root) {
                    return Err(std_only_import_error(&alias.name));
                }
            }
            // Check if this is a Python standard library module that needs special handling
            let rust_import = match alias.name.as_str() {
                // `import numpy as np` (and `import numpy.linalg as np`) is
                // THE canonical numpy spelling. numpy IS a path under the
                // runtime crate (stdpython::numpy), so the alias resolves
                // as a proper `use` — unlike the glob-provided modules
                // below. The alias import also makes `np.linalg.inv(...)`
                // work through the nested path.
                "numpy" | "numpy.linalg" => {
                    let runtime = crate::safe_ident(&options.stdpython);
                    match &alias.asname {
                        None => {
                            if alias.name == "numpy" {
                                // `import numpy` — the name comes from the
                                // `use stdpython::*` glob re-export.
                                quote! {}
                            } else {
                                quote! {
                                    use #runtime::numpy;
                                }
                            }
                        }
                        Some(asname) => {
                            let asname = crate::safe_ident(asname);
                            quote! {
                                use #runtime::numpy as #asname;
                            }
                        }
                    }
                }
                // Runtime-provided modules are already in scope through
                // `use stdpython::*` (each is re-exported at the crate
                // root), so the import lowers to nothing — a bare
                // `use math;` would not even resolve.
                name if is_stdpython_module(name) => {
                    if let Some(asname) = &alias.asname {
                        // The module can't be re-bound with `use` (it's a
                        // glob-imported name, not a path), so aliasing
                        // would silently leave the alias undefined.
                        return Err(format!(
                            "`import {} as {}`: aliasing runtime modules is not \
                             supported yet; use `import {}`",
                            name, asname, name
                        )
                        .into());
                    }
                    quote! {}
                }
                // Python stdlib modules that don't have direct Rust equivalents
                "urllib" | "xml" | "asyncio" => {
                    // These will be provided by the stdpython runtime
                    // Generate a comment instead of a use statement
                    quote! {
                        // Python module '{}' will be provided by stdpython runtime
                    }
                }
                "os.path" => {
                    quote! {
                        // Python os.path module will be provided by stdpython runtime
                    }
                }
                _ => {
                    // Handle other imports normally
                    let names = if alias.name.contains('.') {
                        let parts: Vec<&str> = alias.name.split('.').collect();
                        let idents: Vec<_> =
                            parts.iter().map(|part| crate::safe_ident(part)).collect();
                        quote!(#(#idents)::*)
                    } else {
                        let single_name = crate::safe_ident(&alias.name);
                        quote!(#single_name)
                    };

                    match &alias.asname {
                        None => {
                            quote! {use #names;}
                        }
                        Some(n) => {
                            let name = crate::safe_ident(n);
                            quote! {use #names as #name;}
                        }
                    }
                }
            };

            tokens.extend(rust_import);
        }
        debug!("context: {:?}", ctx);
        debug!("options: {:?}", options);
        debug!("tokens: {}", tokens);
        Ok(tokens)
    }
}

#[derive(Clone, Debug, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct ImportFrom {
    pub module: String,
    pub names: Vec<Alias>,
    pub level: usize,
}

impl CodeGen for ImportFrom {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        for alias in self.names.iter() {
            symbols.insert(
                alias.name.clone(),
                SymbolTableNode::ImportFrom(self.clone()),
            );
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        mut symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        debug!("ctx: {:?}", ctx);
        // annotations map to Rust types directly, so the import itself
        // lowers to nothing.
        if self.module.split('.').next() == Some("typing") {
            return Ok(TokenStream::new());
        }

        // `from rython import rust` is compile-time-only: rust.bind
        // declarations lower to nothing and the module never exists at
        // runtime. Only `rust` is importable — anything else is a mistake
        // worth a loud error, not a silent no-op.
        if self.module == "rython" {
            if self.names.len() == 1
                && self.names[0].name == "rust"
                && self.names[0].asname.is_none()
            {
                return Ok(TokenStream::new());
            }
            return Err(
                "only `from rython import rust` is supported (compile-time Rust \
                 bindings); aliasing or importing other names does not exist"
                    .to_string()
                    .into(),
            );
        }

        // `from <rust-module> import <fn> [as <alias>]`: the functions are
        // compile-time bindings into a Rust crate; the import lowers to
        // nothing and registers each name in the symbol table so call
        // lowering resolves them. An unknown name is a loud error (the
        // stub/inferred signature is the source of truth).
        if let Some(spec) = options.rust_modules.get(&self.module) {
            for alias in self.names.iter() {
                if alias.name == "*" {
                    return Err(format!(
                        "`from {} import *`: wildcard imports of Rust modules are \
                         not supported; import the names explicitly",
                        self.module
                    )
                    .into());
                }
                let bind_name = alias.asname.clone().unwrap_or_else(|| alias.name.clone());
                match spec.get_fn(&alias.name) {
                    Some(fspec) => {
                        let mut spec_for_binding = spec.clone();
                        spec_for_binding.fns = vec![fspec.clone()];
                        symbols.insert(
                            bind_name,
                            crate::SymbolTableNode::RustModule(spec_for_binding),
                        );
                    }
                    None => {
                        return Err(format!(
                            "`from {} import {}`: `{}` is not a bound function of \
                             crate `{}` (bound: {})",
                            self.module,
                            alias.name,
                            alias.name,
                            spec.crate_name,
                            spec.fns
                                .iter()
                                .map(|f| f.fn_name.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                        .into());
                    }
                }
            }
            return Ok(TokenStream::new());
        }

        if options.no_std {
            let root = self.module.split('.').next().unwrap_or(&self.module);
            if is_std_only_module(root) {
                return Err(std_only_import_error(&self.module));
            }
        }

        // `from X import y` must bring `y` into scope; previously this
        // emitted nothing and later uses of `y` were undefined. `use` paths
        // can't resolve through glob imports, so anchor the path explicitly:
        // stdlib modules live under the stdpython runtime crate, and
        // anything else is assumed to be a sibling module of the generated
        // crate. Wildcard imports map to glob uses.
        let parts: Vec<&str> = self
            .module
            .split('.')
            .filter(|part| !part.is_empty())
            .collect();
        let module_path: Vec<_> = parts.iter().map(|part| crate::safe_ident(part)).collect();
        let root = if parts
            .first()
            .is_some_and(|first| is_stdpython_module(first))
        {
            let runtime = crate::safe_ident(&options.stdpython);
            quote!(#runtime)
        } else {
            quote!(crate)
        };

        let mut tokens = TokenStream::new();
        for alias in self.names.iter() {
            // functools.partial/lru_cache/cache have no runtime symbols:
            // partial lowers to a closure at each call site, and the
            // cache decorators rewrite the function definition itself,
            // so the imports emit nothing (an uncalled bare reference is
            // then a loud unresolved-name error).
            if self.module == "functools"
                && matches!(alias.name.as_str(), "partial" | "lru_cache" | "cache")
            {
                continue;
            }
            if alias.name == "*" {
                tokens.extend(quote! { use #root #(::#module_path)*::*; });
                continue;
            }
            // Some runtime functions split into arity/keyword-specific
            // variants (accumulate with initial=, product with repeat=,
            // ...); importing the Python name brings its variants along so
            // the call lowering can pick one. For these names, the BASE
            // import is allow(unused_imports) too: the lowering may rewrite
            // every call site to a variant (accumulate/product always are),
            // orphaning the bare name through no fault of the source
            // Python. Names without variants keep the plain import, so a
            // genuinely unused `from itertools import pairwise` still
            // surfaces as the source weakness it is.
            let variants: &[&str] = match (self.module.as_str(), alias.name.as_str()) {
                ("itertools", "accumulate") => &[
                    "accumulate_sum",
                    "accumulate_func",
                    "accumulate_sum_initial",
                    "accumulate_func_initial",
                ],
                ("itertools", "product") => {
                    &["product2", "product3", "product_repeat2", "product_repeat3"]
                }
                ("itertools", "zip_longest") => &["zip_longest_fill"],
                ("itertools", "groupby") => &["groupby_key"],
                ("functools", "reduce") => &["reduce_initial"],
                ("re", "findall") => &["findall2", "findall3"],
                ("io", "StringIO") => &["StringIO_seeded"],
                ("hashlib", "md5") => &["md5_new"],
                ("hashlib", "sha1") => &["sha1_new"],
                ("hashlib", "sha256") => &["sha256_new"],
                ("hashlib", "sha512") => &["sha512_new"],
                _ => &[],
            };

            let name = crate::safe_ident(&alias.name);
            let import = match &alias.asname {
                None if variants.is_empty() => {
                    quote! { use #root #(::#module_path)*::#name; }
                }
                None => quote! {
                    #[allow(unused_imports)]
                    use #root #(::#module_path)*::#name;
                },
                Some(asname) => {
                    let asname = crate::safe_ident(asname);
                    quote! { use #root #(::#module_path)*::#name as #asname; }
                }
            };
            tokens.extend(import);

            for variant in variants {
                let v = crate::safe_ident(variant);
                tokens.extend(quote! {
                    #[allow(unused_imports)]
                    use #root #(::#module_path)*::#v;
                });
            }
        }
        Ok(tokens)
    }
}
