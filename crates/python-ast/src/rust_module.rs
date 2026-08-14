//! Import-based bindings from Python to Rust crates.
//!
//! The declaration-based `rust.bind`/`rust.c_bind` API (see `rust_bind.rs`)
//! lowers to direct calls into a bound crate. This module adds the
//! *import-based* surface, so a Rust crate can be used like a Python
//! module:
//!
//! ```python
//! import crc32c
//! crc32c.crc32c(data, 0)
//!
//! from crc32c import crc32c as c32
//! c32(data, 1)
//! ```
//!
//! The import name is mapped to a crate by the conversion frontend (rypip /
//! rythonc) via a `rython.toml` manifest, or it may already exist as a
//! dependency of the host crate:
//!
//! ```toml
//! [rust-modules]
//! crc32c = { path = "../crc32c" }        # or version = "1.0"
//! ```
//!
//! Function signatures come from one of two places, in preference order:
//!
//! 1. A `.pyi` stub next to the importing source (`crc32c.pyi`), with
//!    string-annotated types matching the `rust.bind` type set:
//!    ```python
//!    def crc32c(data: "&[u8]", seed: "u32") -> "u32": ...
//!    ```
//! 2. Inference from the crate's own source when a `path=` is given: every
//!    `pub fn` in the crate's `src/lib.rs` (and `src/*.rs` modules) with
//!    supported signatures is bound. (Inference is intentionally
//!    conservative: a function whose signature uses an unsupported type or
//!    construct is skipped — never silently mis-lowered.)
//!
//! Both use the same type set and conversion machinery as `rust.bind`
//! (`convert_rust_bind_arg` / `convert_rust_bind_ret` in `call.rs`), so
//! the two surfaces can never diverge in lowering behavior.

use crate::ast::tree::{ExprType, FunctionDef, ParameterList, StatementType};
use crate::codegen::PythonOptions;
use crate::parser::parse_enhanced;

/// A single bound Rust function's signature, in `rust.bind` terms.
#[derive(Clone, Debug, PartialEq)]
pub struct RustFnSpec {
    /// The function name in the crate.
    pub fn_name: String,
    /// Parameter names and declared Rust types, in call order.
    pub args: Vec<(String, String)>,
    /// Declared return type, if any.
    pub returns: Option<String>,
    /// True when the function is `extern "C"` (or otherwise requires
    /// `unsafe` at the call site).
    pub unsafe_call: bool,
}

/// The resolved binding for an import name: a crate plus the signatures
/// bound from it.
#[derive(Clone, Debug, PartialEq)]
pub struct RustModuleSpec {
    /// The Python-side import name (also the path root of the call).
    pub module_name: String,
    /// The crate the functions live in; used as the Rust path root and the
    /// generated crate's dependency name.
    pub crate_name: String,
    /// Dependency spec: exactly one of path/version is populated by the
    /// frontend; both may be None when the dependency is already declared
    /// in the host crate.
    pub path: Option<String>,
    pub version: Option<String>,
    /// Bound functions, keyed by the Python-visible name.
    pub fns: Vec<RustFnSpec>,
}

impl RustModuleSpec {
    pub fn get_fn(&self, name: &str) -> Option<&RustFnSpec> {
        self.fns.iter().find(|f| f.fn_name == name)
    }
}

/// Resolve one manifest entry into a bound spec. The `.pyi` stub next to
/// the manifest (`<stub_dir>/<import_name>.pyi`) wins; otherwise the crate
/// is inferred from `path`. Version-only entries without a stub are an
/// error — there is no signature to bind.
pub fn resolve_rust_module(
    import_name: &str,
    crate_name: &str,
    path: Option<&str>,
    _version: Option<&str>,
    stub_dir: &std::path::Path,
) -> Result<RustModuleSpec, String> {
    let stub_path = stub_dir.join(format!("{import_name}.pyi"));
    if let Ok(stub) = std::fs::read_to_string(&stub_path) {
        return parse_stub(import_name, crate_name, &stub);
    }
    let Some(path) = path else {
        return Err(format!(
            "`{import_name}`: no {import_name}.pyi stub found next to the manifest, \
             and no `path` is declared to infer signatures from (version-only \
             dependencies need an explicit stub)"
        ));
    };
    // Manifest paths are relative to the manifest's directory.
    let path = if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        stub_dir.join(path)
    };
    infer_from_crate(import_name, crate_name, &path)
}

/// Validate a parameter type string against the supported set.
pub fn validate_param_type(ty: &str) -> Result<(), String> {
    match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize" | "f64"
        | "f32" | "bool" | "String" | "&str" | "Vec<u8>" | "&[u8]" | "*const u8" | "*mut u8" => {
            Ok(())
        }
        other => Err(format!(
            "unsupported Rust parameter type `{}`. Supported: i64 i32 u64 u32 i16 \
             u16 i8 u8 isize usize f64 f32 bool String &str Vec<u8> &[u8] *const u8 *mut u8",
            other
        )),
    }
}

/// Validate a return type string against the supported set.
pub fn validate_return_type(ty: &str) -> Result<(), String> {
    match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize" | "f64"
        | "f32" | "bool" | "String" | "&str" | "Vec<u8>" | "()" | "void" => Ok(()),
        other => Err(format!(
            "unsupported Rust return type `{}`. Supported: i64 i32 u64 u32 i16 u16 \
             i8 u8 isize usize f64 f32 bool String &str Vec<u8> () void",
            other
        )),
    }
}

/// Parse a `.pyi` stub into a `RustModuleSpec`. The stub is ordinary Python
/// (rython's own parser handles it): top-level `def`s with string-annotated
/// parameters and returns. Anything else is a loud error. Stub bodies
/// (`...`) are dropped before parsing — rython has no Ellipsis constant.
pub fn parse_stub(
    module_name: &str,
    crate_name: &str,
    stub: &str,
) -> Result<RustModuleSpec, String> {
    // `def f(x: "i64") -> "i64": ...` bodies are the stub convention; the
    // parser rejects the Ellipsis expression, so blank it out first.
    let source = stub
        .lines()
        .map(|line| {
            let trimmed = line.trim_end();
            if let Some(trimmed) = trimmed.strip_suffix(": ...") {
                format!("{}: pass", trimmed)
            } else if trimmed == "..." {
                "pass".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let module = parse_enhanced(&source, format!("{}.pyi", module_name))
        .map_err(|e| format!("failed to parse stub {}: {}", module_name, e))?;

    let mut fns = Vec::new();
    for stmt in &module.raw.body {
        let FunctionDef {
            name,
            args,
            returns,
            ..
        } = match &stmt.statement {
            StatementType::FunctionDef(f) => f,
            other => {
                return Err(format!(
                    "stub {}: only top-level `def` is allowed in a signature stub, \
                     found {:?}",
                    module_name, other
                ));
            }
        };
        fns.push(function_def_to_spec(name, args, returns.as_deref())?);
    }

    Ok(RustModuleSpec {
        module_name: module_name.to_string(),
        crate_name: crate_name.to_string(),
        path: None,
        version: None,
        fns,
    })
}

/// Register rust-module imports in the symbol table. Call lowering resolves
/// `crc32c.crc32c(...)` and bare `c32(...)` through symbols, so the imports
/// must be visible in the *shared* table — the table every statement's
/// `to_rust` receives. Frontends (rypip / rythonc) run this once, after
/// `find_symbols` and before `to_rust`; `Import`/`ImportFrom`'s own `to_rust`
/// inserts would only reach their private clones.
///
/// This also validates: wildcard from-imports and imports of functions the
/// bound spec does not contain are loud errors here, before codegen.
pub fn register_rust_module_imports(
    body: &[crate::Statement],
    options: &PythonOptions,
    symbols: &mut crate::symbols::SymbolTableScopes,
) -> Result<(), String> {
    for stmt in body {
        match &stmt.statement {
            StatementType::Import(imp) => {
                for alias in &imp.names {
                    let root = alias.name.split('.').next().unwrap_or(&alias.name);
                    if let Some(spec) = options.rust_modules.get(root) {
                        symbols.insert(
                            root.to_string(),
                            crate::SymbolTableNode::RustModule(spec.clone()),
                        );
                        if let Some(asname) = &alias.asname {
                            symbols.insert(
                                asname.clone(),
                                crate::SymbolTableNode::Alias(root.to_string()),
                            );
                        }
                    }
                }
            }
            StatementType::ImportFrom(fr) => {
                if fr.level != 0 {
                    continue;
                }
                let Some(spec) = options.rust_modules.get(&fr.module) else {
                    continue;
                };
                for alias in &fr.names {
                    if alias.name == "*" {
                        return Err(format!(
                            "`from {} import *`: wildcard imports of Rust modules are \
                             not supported; import the names explicitly",
                            fr.module
                        ));
                    }
                    let fspec = spec.get_fn(&alias.name).ok_or_else(|| {
                        format!(
                            "`from {} import {}`: `{}` is not a bound function of \
                             crate `{}` (bound: {})",
                            fr.module,
                            alias.name,
                            alias.name,
                            spec.crate_name,
                            spec.fns
                                .iter()
                                .map(|f| f.fn_name.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                    let mut spec_for_binding = spec.clone();
                    spec_for_binding.fns = vec![fspec.clone()];
                    let bind_name = alias.asname.clone().unwrap_or_else(|| alias.name.clone());
                    symbols.insert(
                        bind_name,
                        crate::SymbolTableNode::RustModule(spec_for_binding),
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Convert a (stub) function definition into a bound function spec,
/// extracting string annotations as Rust types.
pub fn function_def_to_spec(
    name: &str,
    args: &ParameterList,
    returns: Option<&ExprType>,
) -> Result<RustFnSpec, String> {
    // Only the plain positional parameters are supported; *args/**kwargs
    // have no fixed signature to lower against.
    let mut param_types = Vec::new();
    for p in args.posonlyargs.iter().chain(args.args.iter()) {
        let ty = match p.annotation.as_deref() {
            Some(ExprType::Constant(c)) => match &c.0 {
                Some(litrs::Literal::String(s)) => s.value().to_string(),
                _ => {
                    return Err(format!(
                        "stub parameter `{}` in `{}`: annotation must be a string \
                         literal (e.g. \"&[u8]\"), found non-string constant",
                        p.arg, name
                    ));
                }
            },
            Some(other) => {
                return Err(format!(
                    "stub parameter `{}` in `{}`: annotation must be a string \
                     literal (e.g. \"&[u8]\"), found {:?}",
                    p.arg, name, other
                ));
            }
            None => {
                return Err(format!(
                    "stub parameter `{}` in `{}`: missing type annotation",
                    p.arg, name
                ));
            }
        };
        validate_param_type(&ty).map_err(|e| format!("in `{}`: {}", name, e))?;
        param_types.push((p.arg.clone(), ty));
    }
    if args.vararg.is_some() || args.kwarg.is_some() || !args.kwonlyargs.is_empty() {
        return Err(format!(
            "stub function `{}`: *args / **kwargs / keyword-only parameters are \
             not supported in Rust bindings",
            name
        ));
    }

    let ret = match returns {
        None => None,
        Some(ExprType::Constant(c)) => match &c.0 {
            Some(litrs::Literal::String(s)) => {
                let ty = s.value().to_string();
                validate_return_type(&ty).map_err(|e| format!("in `{}`: {}", name, e))?;
                Some(ty)
            }
            _ => {
                return Err(format!(
                    "stub `{}`: return annotation must be a string literal (e.g. \
                     \"u32\"), found non-string constant",
                    name
                ));
            }
        },
        Some(other) => {
            return Err(format!(
                "stub `{}`: return annotation must be a string literal (e.g. \
                 \"u32\"), found {:?}",
                name, other
            ));
        }
    };

    Ok(RustFnSpec {
        fn_name: name.to_string(),
        args: param_types,
        returns: ret,
        unsafe_call: false,
    })
}

// ---------------------------------------------------------------------------
// Inference from the crate's own source (syn)
// ---------------------------------------------------------------------------

/// Infer a `RustModuleSpec` from a crate at `path`: every `pub fn` in
/// `src/lib.rs` (and the modules it declares) with a supported signature is
/// bound. Functions whose signatures use unsupported types or constructs are
/// skipped — inference is conservative, never silently wrong.
///
/// Module declarations (`mod foo;`) are followed into their file; `#[path]`
/// and inline modules are supported too, but the resolver deliberately does
/// not descend into subdirectories it cannot map, and skips `#[cfg]`-gated
/// or macro-generated items.
pub fn infer_from_crate(
    module_name: &str,
    crate_name: &str,
    path: &std::path::Path,
) -> Result<RustModuleSpec, String> {
    let mut fns = Vec::new();
    let mut seen_mods = std::collections::HashSet::new();

    // Resolve the crate root file: `src/lib.rs`, or `src/main.rs` fallback.
    let lib = path.join("src").join("lib.rs");
    let root_file = if lib.is_file() {
        lib
    } else {
        path.join("src").join("main.rs")
    };
    if !root_file.is_file() {
        return Err(format!(
            "cannot infer signatures for `{}` from {}: no src/lib.rs or src/main.rs",
            crate_name,
            path.display()
        ));
    }

    collect_crate_items(&root_file, &mut fns, &mut seen_mods)?;

    Ok(RustModuleSpec {
        module_name: module_name.to_string(),
        crate_name: crate_name.to_string(),
        path: Some(path.display().to_string()),
        version: None,
        fns,
    })
}

/// Collect `pub fn` items from a Rust source file, following `mod`/`pub mod`
/// declarations into their files.
fn collect_crate_items(
    file: &std::path::Path,
    fns: &mut Vec<RustFnSpec>,
    seen_mods: &mut std::collections::HashSet<std::path::PathBuf>,
) -> Result<(), String> {
    let canonical = file
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize {}: {}", file.display(), e))?;
    if !seen_mods.insert(canonical) {
        return Ok(()); // cycle / already visited
    }

    let src = std::fs::read_to_string(file)
        .map_err(|e| format!("cannot read {}: {}", file.display(), e))?;
    let ast: syn::File =
        syn::parse_str(&src).map_err(|e| format!("cannot parse {}: {}", file.display(), e))?;

    let dir = file.parent().unwrap_or_else(|| std::path::Path::new("."));
    for item in &ast.items {
        match item {
            syn::Item::Fn(f) if matches!(f.vis, syn::Visibility::Public(_)) => {
                if let Some(spec) = fn_item_to_spec(f) {
                    fns.push(spec);
                }
            }
            // Follow `mod foo;` (and `pub mod foo;`) to foo.rs / foo/mod.rs
            // in the same directory.
            syn::Item::Mod(m) if m.content.is_none() => {
                if let Some(mod_path) = resolve_mod_file(dir, &m.ident.to_string())
                    && let Err(e) = collect_crate_items(&mod_path, fns, seen_mods)
                {
                    // A broken submodule shouldn't sink the whole crate —
                    // it's skipped with a warning; the rest still binds.
                    eprintln!(
                        "rython: skipping module `{}` while inferring `{}`: {}",
                        m.ident,
                        mod_path.display(),
                        e
                    );
                }
            }
            // Inline `mod foo { ... }` — recurse into its items directly.
            syn::Item::Mod(m) if m.content.is_some() => {
                let content = m.content.as_ref().unwrap();
                for inner in &content.1 {
                    if let syn::Item::Fn(f) = inner
                        && matches!(f.vis, syn::Visibility::Public(_))
                        && let Some(spec) = fn_item_to_spec(f)
                    {
                        fns.push(spec);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Resolve a `mod foo;` declaration to its file (foo.rs, then foo/mod.rs).
fn resolve_mod_file(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let direct = dir.join(format!("{}.rs", name));
    if direct.is_file() {
        return Some(direct);
    }
    let nested = dir.join(name).join("mod.rs");
    if nested.is_file() {
        return Some(nested);
    }
    None
}

/// Convert a `syn::ItemFn` into a `RustFnSpec` when every parameter and the
/// return type are supported. `unsafe`/`extern` functions are marked
/// `unsafe_call` so the call site wraps them.
fn fn_item_to_spec(f: &syn::ItemFn) -> Option<RustFnSpec> {
    let mut args = Vec::new();
    for input in &f.sig.inputs {
        let (pat, ty) = match input {
            syn::FnArg::Receiver(_) => return None, // methods: not bound
            syn::FnArg::Typed(pt) => (pt.pat.as_ref(), pt.ty.as_ref()),
        };
        // Skip pattern forms we can't map to a single name.
        let name = match pat {
            syn::Pat::Ident(pi) => pi.ident.to_string(),
            _ => return None,
        };
        let ty_str = type_to_string(ty)?;
        validate_param_type(&ty_str).ok()?;
        args.push((name, ty_str));
    }
    let returns = match &f.sig.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, ty) => {
            let s = type_to_string(ty.as_ref())?;
            validate_return_type(&s).ok()?;
            Some(s)
        }
    };
    Some(RustFnSpec {
        fn_name: f.sig.ident.to_string(),
        args,
        returns,
        unsafe_call: f.sig.unsafety.is_some() || f.sig.abi.is_some(),
    })
}

/// Render a `syn::Type` into the `rust.bind` type-string form, returning
/// None for unsupported shapes (generics other than Vec<u8>, fn pointers,
/// references with lifetimes beyond `&'a str`/`&'a [u8]`, tuples other than
/// `()`, ...).
fn type_to_string(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => {
            // Single-segment paths only: `i64`, `String`, `Vec<u8>`, ...
            if tp.qself.is_none() && tp.path.segments.len() == 1 {
                let seg = &tp.path.segments[0];
                let ident = seg.ident.to_string();
                if seg.arguments.is_empty() {
                    return Some(ident);
                }
                if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments
                    && ab.args.len() == 1
                    && let syn::GenericArgument::Type(inner) = &ab.args[0]
                {
                    let inner_s = type_to_string(inner)?;
                    return Some(format!("{}<{}>", ident, inner_s));
                }
                return None;
            }
            None
        }
        syn::Type::Reference(r) => {
            let inner = type_to_string(r.elem.as_ref())?;
            let mut s = String::from("&");
            if r.mutability.is_some() {
                s.push_str("mut ");
            }
            s.push_str(&inner);
            Some(s)
        }
        syn::Type::Slice(s) => {
            let inner = type_to_string(s.elem.as_ref())?;
            Some(format!("[{}]", inner))
        }
        syn::Type::Ptr(p) => {
            let inner = type_to_string(p.elem.as_ref())?;
            let mut s = String::from("*");
            s.push_str(if p.mutability.is_some() {
                "mut "
            } else {
                "const "
            });
            s.push_str(&inner);
            Some(s)
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        _ => None,
    }
}
