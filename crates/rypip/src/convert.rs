//! Conversion of a discovered Python package into a Cargo crate: one Rust
//! module per Python module, an optional binary entry point, and optional
//! PyO3 bindings so the crate can be imported from Python again.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use proc_macro2::TokenStream;
use python_ast::{
    parse_enhanced, python_annotation_to_rust_type, safe_ident, CodeGen, CodeGenContext,
    PythonOptions, StatementType, SymbolTableScopes,
};
use quote::quote;
use rust_format::{Formatter, RustFmt};

use crate::package::{PyModule, PyPackage};

/// Lint allowances for generated code: transpiled Python legitimately
/// produces unused imports/variables and similar noise, and the generated
/// crate must still build under a consumer's `-D warnings`.
const GENERATED_ALLOWS: &str = "#![allow(unused_imports, unused_variables, unused_mut, dead_code, unreachable_code, non_snake_case)]\n";

/// Options controlling crate generation.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Add PyO3 bindings (a `python` cargo feature, cdylib output, and a
    /// #[pymodule] exposing bindable functions).
    pub pyo3: bool,
    /// Path to the stdpython runtime crate the generated crate depends on.
    pub stdpython_path: Option<PathBuf>,
}

/// A converted crate on disk.
#[derive(Debug)]
pub struct ConvertedCrate {
    pub root: PathBuf,
    pub name: String,
    /// Whether a binary entry point (src/main.rs) was generated.
    pub has_binary: bool,
}

/// Convert `package` into a Cargo crate under `out_dir`.
pub fn convert(package: &PyPackage, out_dir: &Path, opts: &ConvertOptions) -> Result<ConvertedCrate> {
    let src_dir = out_dir.join("src");
    fs::create_dir_all(&src_dir)
        .with_context(|| format!("creating {}", src_dir.display()))?;

    let entry_file = package.entry_module().map(|m| m.file.clone());

    // Transpile every module.
    let mut transpiled: Vec<(&PyModule, String)> = Vec::new();
    for module in &package.modules {
        let code = transpile(module)?;
        transpiled.push((module, code));
    }

    // Parent -> children map for `pub mod` declarations. The entry module
    // still gets a lib-side module (harmless), except a dedicated
    // `__main__.py`, which is bin-only by convention.
    let mut children: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
    for (module, _) in &transpiled {
        if module.is_init || is_dunder_main(module) {
            continue;
        }
        let (parent, name) = module.path.split_at(module.path.len() - 1);
        children
            .entry(parent.to_vec())
            .or_default()
            .push(name[0].clone());
        // Intermediate packages ensure their ancestors know about them.
        for depth in 1..parent.len() + 1 {
            let (ancestor, child) = parent.split_at(depth - 1);
            let list = children.entry(ancestor.to_vec()).or_default();
            if !list.contains(&child[0].to_string()) {
                list.push(child[0].clone());
            }
        }
    }

    // Write module files. The lint allowances lead each file (they're inner
    // attributes), then the transpiled code (which may itself start with
    // inner doc attributes), then the `pub mod` declarations.
    for (module, code) in &transpiled {
        if is_dunder_main(module) {
            continue; // handled as the binary below
        }
        let is_root = module.path.is_empty();
        let decls = mod_decls(&children, &module.path, module.is_init || is_root);
        let allows = if is_root { GENERATED_ALLOWS } else { "" };
        let contents = format!("{}{}\n{}", allows, code, decls);
        let file = module_file_path(&src_dir, module);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&file, format_rust(&contents))
            .with_context(|| format!("writing {}", file.display()))?;
    }

    // Ensure lib.rs exists even when the package has no root __init__.py.
    let lib_rs = src_dir.join("lib.rs");
    if !lib_rs.exists() {
        let decls = mod_decls(&children, &[], true);
        fs::write(&lib_rs, format_rust(&format!("{}{}", GENERATED_ALLOWS, decls)))?;
    }

    // PyO3 bindings.
    if opts.pyo3 {
        let bindings = generate_bindings(package, &transpiled)?;
        fs::write(src_dir.join("python_api.rs"), format_rust(&bindings))?;
        let mut lib = fs::read_to_string(&lib_rs)?;
        lib.push_str("\n#[cfg(feature = \"python\")]\nmod python_api;\n");
        fs::write(&lib_rs, format_rust(&lib))?;
    }

    // Binary entry point.
    let mut has_binary = false;
    if let Some(entry_file) = &entry_file {
        let (entry, code) = transpiled
            .iter()
            .find(|(m, _)| &m.file == entry_file)
            .expect("entry module was transpiled");
        // The bin target declares the sibling modules itself so the entry
        // module's `use crate::...` imports resolve within the bin crate.
        // Order: lint allowances, entry code (may start with inner doc
        // attributes), then the sibling mod declarations.
        let decls = if !is_dunder_main(entry) && entry.path.len() == 1 {
            // Exclude the entry module's own name from the bin-side decls.
            let mut decls = String::new();
            if let Some(kids) = children.get(&Vec::new()) {
                for kid in kids {
                    if Some(kid) != entry.path.first() {
                        decls.push_str(&format!("mod {};\n", kid));
                    }
                }
            }
            decls
        } else {
            mod_decls(&children, &[], true).replace("pub mod", "mod")
        };
        let main_contents = format!("{}{}\n{}", GENERATED_ALLOWS, code, decls);
        fs::write(src_dir.join("main.rs"), format_rust(&main_contents))?;
        has_binary = true;
    }

    write_cargo_toml(package, out_dir, opts, has_binary)?;

    Ok(ConvertedCrate {
        root: out_dir.to_path_buf(),
        name: package.name.clone(),
        has_binary,
    })
}

fn is_dunder_main(module: &PyModule) -> bool {
    module.path.last().map(String::as_str) == Some("__main__")
}

/// A clean package-relative filename for the parser: it derives a module
/// identifier from the filename, and absolute temp paths contain characters
/// that aren't valid in identifiers.
fn parse_filename(module: &PyModule) -> String {
    if module.is_init {
        if module.path.is_empty() {
            "__init__.py".to_string()
        } else {
            format!("{}/__init__.py", module.path.join("/"))
        }
    } else {
        format!("{}.py", module.path.join("/"))
    }
}

/// Transpile one Python module to Rust source text.
fn transpile(module: &PyModule) -> Result<String> {
    let ast = parse_enhanced(&module.source, parse_filename(module))
        .map_err(|e| anyhow::anyhow!("{} ({})", e, module.file.display()))?;
    let symbols = ast.clone().find_symbols(SymbolTableScopes::new());
    let module_name = module
        .path
        .last()
        .cloned()
        .unwrap_or_else(|| "lib".to_string());
    let tokens = ast
        .to_rust(
            CodeGenContext::Module(module_name),
            PythonOptions::default(),
            symbols,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "compiling {}: {}",
                module.file.display(),
                python_ast::format_error_chain(e.as_ref())
            )
        })?;
    Ok(tokens.to_string())
}

/// `pub mod child;` declarations for a container module.
fn mod_decls(
    children: &BTreeMap<Vec<String>, Vec<String>>,
    at: &[String],
    _is_container: bool,
) -> String {
    let mut out = String::new();
    if let Some(kids) = children.get(at) {
        let mut sorted = kids.clone();
        sorted.sort();
        sorted.dedup();
        for kid in sorted {
            out.push_str(&format!("pub mod {};\n", kid));
        }
    }
    out
}

/// Where a module's Rust file lives within src/.
fn module_file_path(src_dir: &Path, module: &PyModule) -> PathBuf {
    if module.path.is_empty() {
        return src_dir.join("lib.rs");
    }
    if module.is_init {
        let mut dir = src_dir.to_path_buf();
        for part in &module.path {
            dir = dir.join(part);
        }
        return dir.join("mod.rs");
    }
    let (dirs, name) = module.path.split_at(module.path.len() - 1);
    let mut dir = src_dir.to_path_buf();
    for part in dirs {
        dir = dir.join(part);
    }
    dir.join(format!("{}.rs", name[0]))
}

/// Generate the PyO3 bindings module: wrappers for every function whose
/// signature is expressible in concrete Rust types.
fn generate_bindings(package: &PyPackage, transpiled: &[(&PyModule, String)]) -> Result<String> {
    let mut wrappers: Vec<TokenStream> = Vec::new();
    let mut registrations: Vec<TokenStream> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for (module, _) in transpiled {
        if is_dunder_main(module) {
            continue;
        }
        let ast = parse_enhanced(&module.source, parse_filename(module))
            .map_err(|e| anyhow::anyhow!("{} ({})", e, module.file.display()))?;

        for stmt in &ast.raw.body {
            let StatementType::FunctionDef(func) = &stmt.statement else {
                continue;
            };
            if func.name.starts_with('_') {
                continue;
            }
            match bindable_signature(func) {
                Some((params, arg_names, ret)) => {
                    let name = safe_ident(&func.name);
                    let path: Vec<_> = module.path.iter().map(|p| safe_ident(p)).collect();
                    let call = quote!(crate::#(#path::)*#name(#(#arg_names),*));
                    let ret_tokens = match &ret {
                        Some(ty) => quote!(-> #ty),
                        None => quote!(),
                    };
                    let body = match &ret {
                        Some(_) => quote!(#call),
                        None => quote!(#call;),
                    };
                    wrappers.push(quote! {
                        #[pyfunction]
                        fn #name(#(#params),*) #ret_tokens {
                            #body
                        }
                    });
                    registrations.push(quote! {
                        m.add_function(wrap_pyfunction!(#name, m)?)?;
                    });
                }
                None => skipped.push(format!("{}.{}", module.path.join("."), func.name)),
            }
        }
    }

    if wrappers.is_empty() {
        bail!(
            "no functions with bindable signatures found; annotate parameters \
             with int/float/str/bool types to expose them to Python{}",
            if skipped.is_empty() {
                String::new()
            } else {
                format!(" (skipped: {})", skipped.join(", "))
            }
        );
    }

    let module_name = safe_ident(&package.name);
    let skipped_note = if skipped.is_empty() {
        String::new()
    } else {
        format!(
            "//! Skipped (signature not expressible in concrete Rust types yet): {}\n",
            skipped.join(", ")
        )
    };
    let bindings = quote! {
        use pyo3::prelude::*;

        #(#wrappers)*

        #[pymodule]
        fn #module_name(m: &Bound<'_, PyModule>) -> PyResult<()> {
            #(#registrations)*
            Ok(())
        }
    };
    Ok(format!(
        "//! PyO3 bindings generated by rypip.\n{}{}",
        skipped_note, bindings
    ))
}

/// If the function's signature maps to concrete Rust types, return
/// (parameter tokens, argument names, return type tokens).
#[allow(clippy::type_complexity)]
fn bindable_signature(
    func: &python_ast::FunctionDef,
) -> Option<(Vec<TokenStream>, Vec<TokenStream>, Option<TokenStream>)> {
    // Keep to plain positional parameters.
    if func.args.vararg.is_some()
        || !func.args.kwonlyargs.is_empty()
        || !func.args.posonlyargs.is_empty()
        || !func.args.defaults.is_empty()
        || func.args.kwarg.is_some()
    {
        return None;
    }
    let mut params = Vec::new();
    let mut names = Vec::new();
    for param in &func.args.args {
        let annotation = param.annotation.as_deref()?;
        let ty = python_annotation_to_rust_type(annotation)?;
        let name = safe_ident(&param.arg);
        params.push(quote!(#name: #ty));
        names.push(quote!(#name));
    }
    // Return type: body inference first (mirrors the generated signature),
    // then the annotation; a function with neither binds as returning unit.
    let ret = func.inferred_return_type().or_else(|| {
        func.returns
            .as_deref()
            .and_then(python_annotation_to_rust_type)
    });
    Some((params, names, ret))
}

/// Write the generated crate's Cargo.toml.
fn write_cargo_toml(
    package: &PyPackage,
    out_dir: &Path,
    opts: &ConvertOptions,
    _has_binary: bool,
) -> Result<()> {
    let stdpython_path = resolve_stdpython_path(opts)?;
    let mut toml = format!(
        "# Generated by rypip from a Python package. Edit freely.\n\
         [package]\n\
         name = \"{name}\"\n\
         version = \"{version}\"\n\
         edition = \"2021\"\n\n\
         [dependencies]\n\
         stdpython = {{ path = \"{stdpython}\" }}\n",
        name = package.name,
        version = package.version,
        stdpython = stdpython_path.display().to_string().replace('\\', "/"),
    );
    if opts.pyo3 {
        toml.push_str(
            "pyo3 = { version = \"0.29\", features = [\"extension-module\"], optional = true }\n\n\
             [features]\n\
             python = [\"dep:pyo3\"]\n\n\
             [lib]\n\
             crate-type = [\"lib\", \"cdylib\"]\n",
        );
    }
    fs::write(out_dir.join("Cargo.toml"), toml)?;
    // Keep the generated crate out of any enclosing workspace.
    let manifest = out_dir.join("Cargo.toml");
    let mut text = fs::read_to_string(&manifest)?;
    text.push_str("\n[workspace]\n");
    fs::write(manifest, text)?;
    Ok(())
}

/// Locate the stdpython crate the generated code depends on: an explicit
/// option, the RYPIP_STDPYTHON_PATH environment variable, or the copy that
/// ships alongside this tool's own source tree.
fn resolve_stdpython_path(opts: &ConvertOptions) -> Result<PathBuf> {
    if let Some(path) = &opts.stdpython_path {
        return path
            .canonicalize()
            .with_context(|| format!("stdpython path {} not found", path.display()));
    }
    if let Ok(env_path) = std::env::var("RYPIP_STDPYTHON_PATH") {
        return PathBuf::from(&env_path)
            .canonicalize()
            .with_context(|| format!("RYPIP_STDPYTHON_PATH {} not found", env_path));
    }
    let built_in = Path::new(env!("CARGO_MANIFEST_DIR")).join("../stdpython");
    built_in.canonicalize().context(
        "cannot locate the stdpython runtime crate; pass --stdpython <path> or set RYPIP_STDPYTHON_PATH",
    )
}

/// Format generated Rust; fall back to the unformatted text if rustfmt is
/// unavailable or rejects the input.
fn format_rust(source: &str) -> String {
    RustFmt::default()
        .format_str(source)
        .unwrap_or_else(|_| source.to_string())
}
