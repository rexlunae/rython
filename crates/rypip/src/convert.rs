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
/// crate must still build under a consumer's `-D warnings`. `deprecated` is
/// allowed *within* the generated crate because rython uses #[deprecated]
/// notes to warn about lossy conversions (e.g. dropped parameter defaults) —
/// internal call sites are the faithfully-transpiled Python, while external
/// consumers still get the warning at their call sites.
/// Rustc lints the generated code interacts with. Most surface genuine
/// weaknesses in the source Python — unused imports and variables, dead and
/// unreachable code, dead stores (a None seed that is never read),
/// non-snake-case names, calls to lossily-converted (#[deprecated])
/// functions — so they are NOT suppressed by default: surfacing them is
/// part of the point of the tooling. The generated crate's lint posture
/// follows the warning mode: warn leaves rustc's default (warnings at
/// build time), deny promotes them to hard errors, allow suppresses them.
const GENERATED_LINTS: &str = "unused_imports, unused_variables, unused_mut, unused_assignments, dead_code, unreachable_code, non_snake_case, non_upper_case_globals, deprecated, noop_method_call";

fn generated_lint_attrs(mode: WarningMode) -> String {
    match mode {
        WarningMode::Warn => String::new(),
        WarningMode::Deny => format!("#![deny({})]\n", GENERATED_LINTS),
        WarningMode::Allow => format!("#![allow({})]\n", GENERATED_LINTS),
    }
}

/// The crate-level attribute for the no_std profile. Only the crate root
/// carries it; each module brings its own alloc imports (emitted by the
/// module lowering itself).
fn no_std_attr(opts: &ConvertOptions) -> &'static str {
    if opts.no_std { "#![no_std]\n" } else { "" }
}

/// How lossy-conversion warnings are treated during conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum WarningMode {
    /// Report warnings and bake #[deprecated] notes into the generated
    /// code; the generated crate keeps rustc's default lint warnings, which
    /// surface source-Python weaknesses at build time (the default).
    #[default]
    Warn,
    /// Promote warnings to errors: fail the conversion if any conversion is
    /// lossy, and deny the surfaced lints in the generated crate so its
    /// build fails on them.
    Deny,
    /// Suppress warnings entirely — nothing reported, no #[deprecated]
    /// notes, and the surfaced lints are allowed in the generated crate.
    Allow,
}

/// Options controlling crate generation.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Add PyO3 bindings (a `python` cargo feature, cdylib output, and a
    /// #[pymodule] exposing bindable functions).
    pub pyo3: bool,
    /// Path to the stdpython runtime crate the generated crate depends on.
    pub stdpython_path: Option<PathBuf>,
    /// How lossy-conversion warnings are treated.
    pub warnings: WarningMode,
    /// Generate a `#![no_std]` crate on stdpython's alloc tier (no OS
    /// dependency). Python constructs that need the OS — print/input/open,
    /// os/datetime/random/… imports, `__main__` blocks — fail the
    /// conversion loudly instead of surfacing later as build errors in the
    /// generated crate.
    pub no_std: bool,
}

/// A converted crate on disk.
#[derive(Debug)]
pub struct ConvertedCrate {
    pub root: PathBuf,
    pub name: String,
    /// Whether a binary entry point (src/main.rs) was generated.
    pub has_binary: bool,
    /// Human-readable warnings about lossy conversions (e.g. dropped
    /// parameter defaults). These are also baked into the generated code as
    /// #[deprecated] notes so consumers see them at their call sites.
    pub warnings: Vec<String>,
}

/// Convert `package` into a Cargo crate under `out_dir`.
pub fn convert(package: &PyPackage, out_dir: &Path, opts: &ConvertOptions) -> Result<ConvertedCrate> {
    if opts.no_std && opts.pyo3 {
        bail!("PyO3 bindings require std (pyo3 links the Python runtime); drop one of --pyo3 / --no-std");
    }
    let src_dir = out_dir.join("src");
    fs::create_dir_all(&src_dir)
        .with_context(|| format!("creating {}", src_dir.display()))?;

    let entry_file = package.entry_module().map(|m| m.file.clone());

    // Transpile every module, collecting lossy-conversion warnings.
    let mut transpiled: Vec<(&PyModule, String)> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    for module in &package.modules {
        let code = transpile(module, &mut warnings, opts)?;
        transpiled.push((module, code));
    }
    // Bindings are generated before files are written so their warnings
    // (e.g. forced Python-side renames) participate in the warning mode.
    let bindings_text = if opts.pyo3 {
        Some(generate_bindings(package, &transpiled, &mut warnings)?)
    } else {
        None
    };

    match opts.warnings {
        WarningMode::Deny if !warnings.is_empty() => bail!(
            "lossy conversion (warnings denied):\n  {}",
            warnings.join("\n  ")
        ),
        WarningMode::Allow => warnings.clear(),
        _ => {}
    }

    // Parent -> children map for `pub mod` declarations. The entry module
    // still gets a lib-side module (harmless), except a dedicated
    // `__main__.py`, which is bin-only by convention. Non-root __init__
    // modules register too: a sub-package whose only file is __init__.py
    // must still be declared by its parent or its code is silently dropped.
    let mut children: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
    for (module, _) in &transpiled {
        if module.path.is_empty() || is_dunder_main(module) {
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
        let allows = if is_root {
            format!("{}{}", no_std_attr(opts), generated_lint_attrs(opts.warnings))
        } else {
            String::new()
        };
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
        fs::write(
            &lib_rs,
            format_rust(&format!(
                "{}{}{}",
                no_std_attr(opts),
                generated_lint_attrs(opts.warnings),
                decls
            )),
        )?;
    }

    // PyO3 bindings.
    if let Some(bindings) = &bindings_text {
        fs::write(src_dir.join("python_api.rs"), format_rust(bindings))?;
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
        let main_contents = format!(
            "{}{}\n{}",
            generated_lint_attrs(opts.warnings),
            code,
            decls
        );
        fs::write(src_dir.join("main.rs"), format_rust(&main_contents))?;
        has_binary = true;
    }

    write_cargo_toml(package, out_dir, opts, has_binary)?;

    Ok(ConvertedCrate {
        root: out_dir.to_path_buf(),
        name: package.name.clone(),
        has_binary,
        warnings,
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

/// Transpile one Python module to Rust source text, appending
/// lossy-conversion warnings (which are also baked into the generated code
/// as #[deprecated] notes, unless the warning mode suppresses them).
fn transpile(
    module: &PyModule,
    warnings: &mut Vec<String>,
    opts: &ConvertOptions,
) -> Result<String> {
    let mode = opts.warnings;
    let ast = parse_enhanced(&module.source, parse_filename(module))
        .map_err(|e| anyhow::anyhow!("{} ({})", e, module.file.display()))?;

    for stmt in &ast.raw.body {
        if let StatementType::FunctionDef(func) = &stmt.statement {
            for note in func.lossy_conversion_notes() {
                warnings.push(format!(
                    "{}: function `{}`: {}",
                    parse_filename(module),
                    func.name,
                    note.trim_start_matches("rython: "),
                ));
            }
        }
    }

    let symbols = ast.clone().find_symbols(SymbolTableScopes::new());
    let module_name = module
        .path
        .last()
        .cloned()
        .unwrap_or_else(|| "lib".to_string());
    let options = PythonOptions {
        lossy_warnings: mode != WarningMode::Allow,
        no_std: opts.no_std,
        ..Default::default()
    };
    let tokens = ast
        .to_rust(
            CodeGenContext::Module(module_name),
            options,
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
/// signature is expressible in concrete Rust types. Wrapper identifiers are
/// qualified by module path so same-named functions in different modules
/// don't collide in the flat bindings file; the Python-visible name stays
/// the bare function name when it is unique across the package, and falls
/// back to the qualified name (with a conversion warning — it's a visible
/// rename) when it isn't.
fn generate_bindings(
    package: &PyPackage,
    transpiled: &[(&PyModule, String)],
    warnings: &mut Vec<String>,
) -> Result<String> {
    type Signature = (Vec<TokenStream>, Vec<TokenStream>, Option<TokenStream>);
    let mut candidates: Vec<(&PyModule, python_ast::FunctionDef, Signature)> = Vec::new();
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
                Some(sig) => candidates.push((module, func.clone(), sig)),
                None => skipped.push(format!("{}.{}", module.path.join("."), func.name)),
            }
        }
    }

    let mut name_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, func, _) in &candidates {
        *name_counts.entry(func.name.as_str()).or_default() += 1;
    }

    let mut wrappers: Vec<TokenStream> = Vec::new();
    let mut registrations: Vec<TokenStream> = Vec::new();
    let mut collisions: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (module, func, (params, arg_names, ret)) in &candidates {
        let bare = func.name.as_str();
        let qualified = if module.path.is_empty() {
            bare.to_string()
        } else {
            format!("{}_{}", module.path.join("_"), bare)
        };
        let wrapper_name = safe_ident(&qualified);
        let target = safe_ident(bare);
        let path: Vec<_> = module.path.iter().map(|p| safe_ident(p)).collect();
        let call = quote!(crate::#(#path::)*#target(#(#arg_names),*));
        // Generated functions return Result<T, PyException>; the wrapper
        // maps a raised exception onto the corresponding real Python
        // exception class (From<PyException> for PyErr in stdpython).
        let ret_tokens = match ret {
            Some(ty) => quote!(-> pyo3::PyResult<#ty>),
            None => quote!(-> pyo3::PyResult<()>),
        };
        let body = quote!(#call.map_err(pyo3::PyErr::from));
        // Keep the bare Python-visible name when it's unambiguous; a
        // package-wide duplicate keeps the qualified name (registering two
        // same-named functions would silently shadow one of them).
        let py_name = if name_counts[bare] == 1 {
            quote!(#[pyo3(name = #bare)])
        } else {
            collisions.entry(bare).or_default().push(qualified.clone());
            quote!()
        };
        wrappers.push(quote! {
            #[pyfunction]
            #py_name
            fn #wrapper_name(#(#params),*) #ret_tokens {
                #body
            }
        });
        registrations.push(quote! {
            m.add_function(wrap_pyfunction!(#wrapper_name, m)?)?;
        });
    }

    for (bare, qualified) in &collisions {
        warnings.push(format!(
            "python bindings: {} modules define a function named `{}`; they are \
             exposed to Python under module-qualified names (`{}`) because a \
             module cannot hold two same-named functions",
            qualified.len(),
            bare,
            qualified.join("`, `"),
        ));
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
    // Keep to plain positional parameters without defaults: *args/**kwargs
    // and positional-only parameters add generated parameters this simple
    // wrapper doesn't model, and defaulted parameters would lose their
    // Python-side optionality (a #[pyo3(signature = ...)] attribute could
    // lift that restriction later).
    if func.args.vararg.is_some()
        || func.args.kwarg.is_some()
        || !func.args.kwonlyargs.is_empty()
        || !func.args.posonlyargs.is_empty()
        || !func.args.defaults.is_empty()
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
    // The wrapper's return type must be exactly what the generated function
    // carries — resolved_return_type is the single source of truth (it gates
    // annotations on all-paths-return, so a function that can fall through
    // binds as returning unit, matching the generated `()`).
    let ret = func.resolved_return_type();
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
    // The no_std profile pins stdpython to its alloc tier: no OS, no libc,
    // suitable for embedded/wasm targets.
    let stdpython_dep = if opts.no_std {
        format!(
            "stdpython = {{ path = \"{}\", default-features = false, features = [\"alloc\"] }}",
            stdpython_path.display().to_string().replace('\\', "/"),
        )
    } else {
        format!(
            "stdpython = {{ path = \"{}\" }}",
            stdpython_path.display().to_string().replace('\\', "/"),
        )
    };
    let mut toml = format!(
        "# Generated by rypip from a Python package. Edit freely.\n\
         [package]\n\
         name = \"{name}\"\n\
         version = \"{version}\"\n\
         edition = \"2021\"\n\n\
         [dependencies]\n\
         {stdpython_dep}\n",
        name = package.name,
        version = package.version,
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
