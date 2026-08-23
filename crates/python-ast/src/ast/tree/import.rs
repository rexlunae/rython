use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
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
            | "warnings"
            | "numpy"
            // asyncio lives on the tokio-backed `async-tokio` stdpython
            // feature; generated async binaries enable it.
            | "asyncio"
    )
}

/// Whether an imported name RESOLVES to an EXTERNAL module's item (a
/// re-export chain ending in `from urllib.parse import urlparse` — requests'
/// compat, where urllib is external): no runtime item exists behind the
/// chain, so the use drops.
/// Whether a name is imported (directly or through a chain) from a
/// vendored `[python-modules]` dependency — such names are NOT external
/// (the dep is compiled into the crate).
pub(crate) fn import_from_python_module(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let Some(SymbolTableNode::ImportFrom(ifm)) = symbols.get(name) else {
        return false;
    };
    let root = ifm.module.split('.').next().unwrap_or("");
    options.python_modules.contains(root)
}

pub(crate) fn resolves_to_external_import(
    name: &str,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    // Only meaningful when the whole crate is known (multi-module
    // conversion): a single-module conversion may import any sibling.
    if options.module_defs.len() <= 1 {
        return false;
    }
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    // The package path of the CURRENT module in the chain, so its RELATIVE
    // imports resolve against the right context (resolve_imported_class's
    // model): options.module_path for the caller, then the defining
    // module's package path at each hop.
    let mut module_path = options.module_path.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let mut ctx = options.clone();
                ctx.module_path = module_path.clone();
                let path = ifm.resolved_module_path(&ctx);
                if options.module_defs.contains_key(&path) {
                    // A re-export chain: hop into the defining module.
                    let is_package = options.module_defs.keys().any(|k| {
                        k.len() > path.len() && k[..path.len()] == path[..]
                    });
                    module_path = if is_package {
                        path.clone()
                    } else {
                        path[..path.len().saturating_sub(1)].to_vec()
                    };
                    let defining = ifm
                        .names
                        .iter()
                        .find(|a| a.asname.as_deref() == Some(&current))
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| current.clone());
                    let module = &options.module_defs[&path];
                    let module: &crate::Module = module;
                    syms = module.clone().find_symbols(SymbolTableScopes::new());
                    current = defining;
                } else {
                    // The terminal hop: external when the module is neither
                    // stdpython nor a vendored python-module dep.
                    let root = ifm.module.split('.').next().unwrap_or("");
                    // `collections.abc` is the compile-time-only typing
                    // abstraction (Mapping, Iterable, ...): its names have
                    // no runtime items anywhere — external.
                    if ifm.module == "collections.abc" {
                        return true;
                    }
                    return !is_stdpython_module(root)
                        && !options.python_modules.contains(root);
                }
            }
            _ => return false,
        }
    }
    false
}

/// Whether an imported name is a TYPE-NAME TUPLE alias (`basestring =
/// (str, bytes)` — requests' compat): consumed by isinstance resolution at
/// conversion time, never a runtime value. Follows ImportFrom re-export
/// chains through the generated crate.
fn is_type_name_tuple_alias(
    name: &str,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::Assign { value, .. }) => {
                let crate::ExprType::Tuple(t) = value else {
                    return false;
                };
                // Every element must be a TYPE NAME (basestring = (str,
                // bytes)) — not an arbitrary runtime name: requests'
                // `_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR,
                // _VALID_HEADER_VALUE_RE_STR)` is a runtime tuple of
                // compiled-regex statics, imported and indexed at runtime.
                return t.elts.iter().all(|e| matches!(e, crate::ExprType::Name(n)
                    if matches!(
                        n.id.as_str(),
                        "str" | "bytes" | "bytearray" | "int" | "float" | "bool"
                            | "object" | "None" | "Any" | "Union"
                    )));
            }
            Some(SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let path = ifm.resolved_module_path(options);
                if !options.module_defs.contains_key(&path) {
                    return false;
                }
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[&path];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
            }
            _ => return false,
        }
    }
    false
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
            | "asyncio"
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
            // `import a.b.c` binds the ROOT name `a` in Python (a later
            // statement can reference `a.b`). Register the root too, so
            // module-chain resolution works for submodule attribute calls
            // (`import h2.config` — urllib3's http2: `h2.config.
            // H2Configuration(...)`).
            if let Some(root) = alias.name.split('.').next() {
                if !root.is_empty() && root != &alias.name {
                    symbols.insert(root.to_string(), SymbolTableNode::Import(self.clone()));
                }
            }
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
            // Import of a vendored Python module (`[python-modules]` in
            // rython.toml): it's a sibling module of the generated crate.
            // Attribute calls lower to `crate::textlib::fn` paths, so the
            // plain spelling emits no `use` — a `use crate::textlib;` in
            // the binary would collide with the `mod textlib;` declaration
            // that brings the sibling module into the bin crate. The
            // aliased spelling still needs the import to bind the alias
            // (`use crate::textlib as t;` — a different name, no clash).
            {
                let root = alias.name.split('.').next().unwrap_or(&alias.name);
                if options.python_modules.contains(root) {
                    if let Some(asname) = &alias.asname {
                        let names = if alias.name.contains('.') {
                            let parts: Vec<&str> = alias.name.split('.').collect();
                            let idents: Vec<_> =
                                parts.iter().map(|part| crate::safe_ident(part)).collect();
                            quote!(#(#idents)::*)
                        } else {
                            let single_name = crate::safe_ident(&alias.name);
                            quote!(#single_name)
                        };
                        let name = crate::safe_ident(asname);
                        tokens.extend(quote! {use crate::#names as #name;});
                    }
                    continue;
                }
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
                // `use math;` would not even resolve. An ALIASED import
                // (`import time as t`, `import json as _json`) binds the
                // alias as a real path (`use stdpython::time as t;`) so
                // `t::monotonic()` / `_json::loads()` resolve — the same
                // spelling numpy's alias arm uses.
                name if is_stdpython_module(name) => {
                    if let Some(asname) = &alias.asname {
                        let runtime = crate::safe_ident(&options.stdpython);
                        let module = crate::safe_ident(name);
                        let asname = crate::safe_ident(asname);
                        quote! {
                            use #runtime::#module as #asname;
                        }
                    } else {
                        quote! {}
                    }
                }
                // Python stdlib modules that don't have direct Rust equivalents
                "urllib" | "xml" => {
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
                    // A sibling module of the generated crate resolves via
                    // `use crate::...`; an EXTERNAL module (stdlib rython
                    // does not model — ssl, socket, logging, http, typing,
                    // codecs, types, ... — or a third-party dep that is not
                    // vendored) has no generated item, so the import lowers
                    // to nothing with a warning (documented divergence:
                    // the module's runtime functionality is unavailable;
                    // uses of its names become loud errors or boxed drops).
                    let path: Vec<String> =
                        alias.name.split('.').map(|s| s.to_string()).collect();
                    let is_sibling = options
                        .module_defs
                        .contains_key(&path)
                        || options.python_modules.contains(
                            &path.first().cloned().unwrap_or_default(),
                        )
                        // Single-module conversions only know the module
                        // itself (module_defs.len() == 1): assume any other
                        // non-stdpython import is a crate sibling (the
                        // module_defs check is authoritative only when the
                        // whole crate is known).
                        || options.module_defs.len() <= 1;
                    if !is_sibling {
                        options.definition_warnings.borrow_mut().push(format!(
                            "import `{}` is dropped: the module is not part of the \
                             generated crate nor the stdpython runtime \
                             (external-module divergence)",
                            alias.name
                        ));
                        quote! {}
                    } else {
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
                                quote! {use crate::#names;}
                            }
                            Some(n) => {
                                let name = crate::safe_ident(n);
                                quote! {use crate::#names as #name;}
                            }
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportFrom {
    /// The dotted module being imported FROM. `from . import x` and
    /// `from . import` (relative imports with no module part) have
    /// module = None in Python's AST — extracted as "" here so the
    /// resolved path is just the current package.
    pub module: String,
    pub names: Vec<Alias>,
    pub level: usize,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ImportFrom {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let module: Option<String> = ob
            .getattr("module")
            .map_err(|e| crate::extraction_failure("ImportFrom module", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom module", &ob, e))?;
        let names: Vec<Alias> = ob
            .getattr("names")
            .map_err(|e| crate::extraction_failure("ImportFrom names", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom names", &ob, e))?;
        let level: usize = ob
            .getattr("level")
            .map_err(|e| crate::extraction_failure("ImportFrom level", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom level", &ob, e))?;
        Ok(ImportFrom {
            module: module.unwrap_or_default(),
            names,
            level,
        })
    }
}

impl ImportFrom {
    /// The module path this import resolves to inside the generated crate:
    /// for a relative import, the current module path (cut by `level`) plus
    /// the dotted module; for an absolute import, the dotted module itself.
    /// Key into `options.module_defs` to reach the defining module's AST.
    pub(crate) fn resolved_module_path(&self, options: &PythonOptions) -> Vec<String> {
        let parts: Vec<&str> = self.module.split('.').filter(|p| !p.is_empty()).collect();
        if self.level > 0 {
            let cur = &options.module_path;
            // A relative import with more leading dots than the current
            // package depth reaches above the crate root; saturate so the
            // caller gets an empty (or root-level) prefix instead of a
            // usize underflow panic. `ImportFrom::to_rust` reports the
            // clean "reaches above the crate root" error for the user.
            let cut = (cur.len() + 1).saturating_sub(self.level);
            cur[..cut]
                .iter()
                .map(|s| s.as_str())
                .chain(parts.iter().copied())
                .map(|s| s.to_string())
                .collect()
        } else {
            parts.iter().map(|s| s.to_string()).collect()
        }
    }
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
            // `from pylev import wf as w`: the alias resolves to the
            // canonical name so call lowering propagates exceptions and
            // attribute access treats it as the imported value. A SELF-alias
            // (`from ._base_connection import ProxyConfig as ProxyConfig` —
            // urllib3's re-export) must NOT overwrite the ImportFrom symbol:
            // resolve_imported_class follows the chain through ImportFrom,
            // and an Alias-to-self would loop.
            if let Some(asname) = &alias.asname {
                if asname != &alias.name {
                    symbols.insert(asname.clone(), SymbolTableNode::Alias(alias.name.clone()));
                }
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
        // `from __future__ import ...` is a compiler directive, not a
        // runtime import: the future flags (annotations, generators, ...)
        // are either already the language's default behavior or have no
        // Rust analogue, so the statement lowers to nothing (a `use
        // crate::__future__::...` would be an unresolved import).
        if self.module == "__future__" {
            return Ok(TokenStream::new());
        }

        // `from dataclasses import dataclass` (and field, ...): the
        // decorator is CONSUMED at conversion time by the class codegen
        // (synthesized __init__), so the import is a no-op — a `use
        // crate::dataclasses::...` would be an unresolved import. Other
        // dataclasses names are the same: nothing from the module exists
        // at runtime in the generated crate.
        if self.module == "dataclasses" {
            return Ok(TokenStream::new());
        }

        // `from rython import rust` — compile-time Rust bindings.
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
        // Relative imports (`from .x import y`, `from ..x import y`, level
        // > 0) resolve against the CURRENT module's package path
        // (`options.module_path`, set by the converter; empty at the crate
        // root). level 1 is this package, level 2 its parent, and so on;
        // the resolved path always stays inside the generated crate, since
        // the package and all its submodules are compiled into it.
        let base_parts: Vec<_> = if self.level > 0 {
            let cur = &options.module_path;
            if self.level > cur.len() + 1 {
                return Err(format!(
                    "`from {} import ...` (level {}): relative import goes above \
                     the crate root",
                    self.module, self.level
                )
                .into());
            }
            let cut = cur.len() + 1 - self.level;
            cur[..cut].iter().map(|p| crate::safe_ident(p)).collect()
        } else {
            Vec::new()
        };
        let module_path: Vec<_> = parts.iter().map(|part| crate::safe_ident(part)).collect();
        let root = if self.level > 0 {
            quote!(crate)
        } else if parts
            .first()
            .is_some_and(|first| is_stdpython_module(first))
        {
            let runtime = crate::safe_ident(&options.stdpython);
            quote!(#runtime)
        } else {
            quote!(crate)
        };

        // An EXTERNAL module (stdlib rython does not model — logging, ssl,
        // socket, http, codecs, types, importlib, ... — or a non-vendored
        // dependency) has no generated items: the import lowers to nothing
        // with a warning (documented divergence). Its names still resolve in
        // the symbol table, so annotations map them to the boxed PyValue and
        // runtime calls drop. `collections.abc` is the typing-abstraction
        // submodule (Mapping, Iterable, ...): also compile-time-only.
        // Relative imports always target sibling modules of the crate.
        let resolved_path = self.resolved_module_path(&options);
        let first_part = parts.first().copied();
        // The external check is only meaningful when the whole crate is
        // known (multi-module conversions populate module_defs): a
        // single-module conversion (len == 1) only knows the module itself,
        // so any absolute non-stdpython import is assumed to be a crate
        // sibling (`from helpers import util`).
        let external = self.level == 0
            && options.module_defs.len() > 1
            && !matches!(first_part, Some(p) if is_stdpython_module(p))
            && !options.module_defs.contains_key(&resolved_path)
            && !options
                .python_modules
                .contains(&first_part.unwrap_or("").to_string());
        if external || self.module == "collections.abc" {            options.definition_warnings.borrow_mut().push(format!(
                "`from {} import ...` is dropped: the module is not part of the \
                 generated crate nor the stdpython runtime \
                 (external-module divergence)",
                self.module
            ));
            return Ok(TokenStream::new());
        }

        let mut tokens = TokenStream::new();
        // Deduplicate trait imports across the aliases of ONE ImportFrom:
        // `from .connectionpool import HTTPConnectionPool,
        // HTTPSConnectionPool` — both classes share the ancestor trait
        // `ConnectionPoolTrait`, so the bring-along would emit it twice
        // (E0252 — duplicate import).
        let mut seen_traits: std::collections::HashSet<String> =
            std::collections::HashSet::new();
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
            // A TYPE-NAME TUPLE alias (`basestring = (str, bytes)` —
            // requests' compat): consumed by isinstance resolution at
            // conversion time, never a runtime value — the import emits
            // nothing (a `pub use crate::...::basestring` would fail: the
            // value is a module-init local, not a static).
            if is_type_name_tuple_alias(&alias.name, &options, &symbols) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` is a type-name tuple alias \
                     (typing-only; consumed by isinstance resolution)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // A name that RE-EXPORTS from an EXTERNAL module (`from
            // urllib.parse import urlparse` in requests' compat — urllib is
            // external): no runtime item exists behind the chain, so the
            // use drops (calls through the name lower to the boxed None).
            if resolves_to_external_import(&alias.name, &options, &symbols) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` re-exports from an external module \
                     (no runtime item; the import is dropped)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            if alias.name == "*" {
                let visibility = if self.level > 0 { quote!(pub) } else { quote!() };
                tokens.extend(quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::*; });
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
            // Relative imports re-export from a sibling module: Python
            // treats imported names as module attributes (the package
            // `__init__.py` re-export pattern), so they lower to `pub use`
            // — callers reach `textlib.double` through the re-export chain.
            // Absolute imports of user modules stay plain `use`: callers
            // import from the defining module directly.
            let visibility = if self.level > 0 { quote!(pub) } else { quote!() };
            let import = match &alias.asname {
                None if variants.is_empty() => {
                    quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::#name; }
                }
                None => quote! {
                    #[allow(unused_imports)]
                    #visibility use #root #(::#base_parts)* #(::#module_path)*::#name;
                },
                Some(asname) => {
                    let asname = crate::safe_ident(asname);
                    quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::#name as #asname; }
                }
            };
            tokens.extend(import);

            for variant in variants {
                let v = crate::safe_ident(variant);
                tokens.extend(quote! {
                    #[allow(unused_imports)]
                    use #root #(::#base_parts)* #(::#module_path)*::#v;
                });
            }

            // A hierarchy class imported from another module of the
            // generated crate carries its methods on traits (`{Name}Trait`
            // plus ancestors'), NOT on the struct — Rust method resolution
            // needs those traits IN SCOPE at the call site, so the import
            // brings them along: `from .animals import Dog` also imports
            // `AnimalTrait`, and `d.get()` resolves. Only classes that
            // lower with the trait machinery have traits; functions and
            // plain structs get none (the per-module map is empty for
            // them).
            let import_module_path = self.resolved_module_path(&options);
            if options.module_defs.contains_key(&import_module_path)
                && let Some(traits) =
                    crate::module_class_traits(&options, &import_module_path).get(&alias.name)
            {
                for trait_name in traits {
                    if !seen_traits.insert(trait_name.clone()) {
                        continue;
                    }
                    let t = crate::safe_ident(trait_name);
                    tokens.extend(quote! {
                        #[allow(unused_imports)]
                        use #root #(::#base_parts)* #(::#module_path)*::#t;
                    });
                }
            }
        }
        Ok(tokens)
    }
}
