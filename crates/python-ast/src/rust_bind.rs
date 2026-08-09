//! Compile-time bindings from Python to Rust crates.
//!
//! `from rython import rust` gives Python source a way to call functions in
//! any Rust crate directly — no runtime, no interpreter, no PyO3. The
//! binding is a *declaration*, evaluated entirely at conversion time:
//!
//! ```python
//! from rython import rust
//!
//! crc = rust.bind(
//!     "crc32c",                # crate name (dependency of the generated crate)
//!     "crc32c",                # function to call
//!     args=[("data", "&[u8]"), ("seed", "u32")],
//!     returns="u32",
//!     version="2",
//! )
//! ```
//!
//! The assignment lowers to nothing; a later `crc(...)` call lowers to a
//! direct `crc32c::crc32c(...)` call with type-directed conversions between
//! rython's Python types (i64, f64, bool, String, Vec<u8>) and the declared
//! Rust signature. `rust.c_bind(...)` is the same, but the call is wrapped
//! in `unsafe { }` and targets a C-ABI (`extern "C"`) function, mirroring
//! ctypes.
//!
//! Supported parameter types: `i64 i32 u64 u32 i16 u16 i8 u8 isize usize
//! f64 f32 bool String &str Vec<u8> &[u8] *const u8 *mut u8`. Supported
//! return types: the same integer/float/bool set plus `String &str
//! Vec<u8> ()` (and `void`). Anything else is a loud conversion error —
//! never a silently wrong cast.

use crate::tree::{Call, ExprType};

/// A parsed `rust.bind(...)` / `rust.c_bind(...)` declaration.
#[derive(Clone, Debug)]
pub struct RustBindSpec {
    /// Crate the function lives in. Used as the path root of the call and
    /// as the `[dependencies]` entry of the generated crate.
    pub crate_name: String,
    /// Function to call within the crate.
    pub fn_name: String,
    /// Parameter names and their declared Rust types, in call order.
    pub args: Vec<(String, String)>,
    /// Declared return type, if any.
    pub returns: Option<String>,
    /// True for `rust.c_bind` (C ABI): calls are wrapped in `unsafe { }`.
    pub c_abi: bool,
    /// Dependency spec: exactly one of `path`/`version` is required, so the
    /// generated crate's Cargo.toml can declare the dependency without
    /// guessing. `rypip` reads these; other frontends (python-mod) ignore
    /// them and expect the dependency in the host crate's Cargo.toml.
    pub path: Option<String>,
    pub version: Option<String>,
}

/// Whether an expression is a `rust.bind(...)` / `rust.c_bind(...)` call.
pub fn is_rust_bind_call(expr: &ExprType) -> bool {
    match expr {
        ExprType::Call(call) => matches!(bind_kind(&call.func), Some(_)),
        _ => false,
    }
}

/// `Some(false)` for `rust.bind`, `Some(true)` for `rust.c_bind`, None for
/// any other callee.
fn bind_kind(func: &ExprType) -> Option<bool> {
    match func {
        ExprType::Attribute(attr) => match attr.value.as_ref() {
            ExprType::Name(m) if m.id == "rust" => match attr.attr.as_str() {
                "bind" => Some(false),
                "c_bind" => Some(true),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// Parse a `rust.bind`/`rust.c_bind` call into its spec. Only call this
/// when `is_rust_bind_call` matched. Errors are conversion-time and loud:
/// a malformed declaration is a bug in the Python source, never something
/// to paper over.
pub fn parse_rust_bind(call: &Call) -> Result<RustBindSpec, String> {
    let c_abi = bind_kind(&call.func)
        .ok_or_else(|| "internal error: parse_rust_bind on a non-binding call".to_string())?;

    if call.args.len() < 2 {
        return Err(format!(
            "rust.{}() needs the crate name and function name as its first two \
             positional arguments, e.g. rust.{}(\"mylib\", \"add\", args=[...])",
            if c_abi { "c_bind" } else { "bind" },
            if c_abi { "c_bind" } else { "bind" }
        ));
    }
    if call.args.len() > 2 {
        return Err(format!(
            "rust.{}() takes 2 positional arguments (crate name and function \
             name); {} were given. Pass the signature as keyword arguments: \
             args=[(\"name\", \"type\"), ...], returns=\"type\"",
            if c_abi { "c_bind" } else { "bind" },
            call.args.len()
        ));
    }

    let crate_name = string_literal(&call.args[0], "crate name")?;
    let fn_name = string_literal(&call.args[1], "function name")?;
    if crate_name.is_empty() {
        return Err("rust.bind: the crate name cannot be empty".to_string());
    }
    if fn_name.is_empty() {
        return Err("rust.bind: the function name cannot be empty".to_string());
    }

    let mut args: Vec<(String, String)> = Vec::new();
    let mut returns: Option<String> = None;
    let mut path: Option<String> = None;
    let mut version: Option<String> = None;

    for kw in &call.keywords {
        match kw.arg.as_deref() {
            Some("args") => {
                let list = match &kw.value {
                    ExprType::List(items) => items,
                    _ => {
                        return Err(
                            "rust.bind: the `args=` keyword must be a list of \
                             (\"name\", \"type\") tuples"
                                .to_string(),
                        )
                    }
                };
                for item in list {
                    let (name, ty) = match item {
                        ExprType::Tuple(t) if t.elts.len() == 2 => {
                            (string_literal(&t.elts[0], "parameter name")?, string_literal(
                                &t.elts[1],
                                "parameter type",
                            )?)
                        }
                        _ => {
                            return Err(
                                "rust.bind: every entry of `args=` must be a \
                                 (\"name\", \"type\") tuple"
                                    .to_string(),
                            )
                        }
                    };
                    if name.is_empty() {
                        return Err("rust.bind: parameter names cannot be empty".to_string());
                    }
                    validate_param_type(&ty)?;
                    args.push((name, ty));
                }
            }
            Some("returns") => {
                let ty = string_literal(&kw.value, "return type")?;
                validate_return_type(&ty)?;
                returns = Some(ty);
            }
            Some("path") => path = Some(string_literal(&kw.value, "path")?),
            Some("version") => version = Some(string_literal(&kw.value, "version")?),
            other => {
                return Err(format!(
                    "rust.bind: unknown keyword argument `{}` (supported: args, \
                     returns, path, version)",
                    other.unwrap_or("**kwargs")
                ))
            }
        }
    }

    match (path.is_some(), version.is_some()) {
        (true, true) => {
            return Err(
                "rust.bind: give either `path=` or `version=` for the crate \
                 dependency, not both"
                    .to_string(),
            )
        }
        (false, false) => {
            return Err(format!(
                "rust.bind: `{}` needs `path=` or `version=` so the generated \
                 crate can declare its dependency on `{}`",
                fn_name, crate_name
            ))
        }
        _ => {}
    }

    Ok(RustBindSpec {
        crate_name,
        fn_name,
        args,
        returns,
        c_abi,
        path,
        version,
    })
}

/// Every `rust.bind` declaration in a module's top-level statements — the
/// only place bindings are allowed. Used by rypip to inject the `[dependencies]`.
pub fn rust_bind_specs(body: &[crate::Statement]) -> Vec<RustBindSpec> {
    let mut specs = Vec::new();
    for stmt in body {
        if let crate::StatementType::Assign(a) = &stmt.statement {
            if let ExprType::Call(call) = &a.value {
                if is_rust_bind_call(&a.value) {
                    if let Ok(spec) = parse_rust_bind(call) {
                        specs.push(spec);
                    }
                }
            }
        }
    }
    specs
}

fn validate_param_type(ty: &str) -> Result<(), String> {
    match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize"
        | "f64" | "f32" | "bool" | "String" | "&str" | "Vec<u8>" | "&[u8]" | "*const u8"
        | "*mut u8" => Ok(()),
        other => Err(format!(
            "rust.bind: unsupported parameter type `{}`. Supported: i64 i32 u64 u32 i16 \
             u16 i8 u8 isize usize f64 f32 bool String &str Vec<u8> &[u8] *const u8 *mut u8",
            other
        )),
    }
}

fn validate_return_type(ty: &str) -> Result<(), String> {
    match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize"
        | "f64" | "f32" | "bool" | "String" | "&str" | "Vec<u8>" | "()" | "void" => Ok(()),
        other => Err(format!(
            "rust.bind: unsupported return type `{}`. Supported: i64 i32 u64 u32 i16 u16 \
             i8 u8 isize usize f64 f32 bool String &str Vec<u8> () void",
            other
        )),
    }
}

/// Extract the value of a string-literal constant expression.
fn string_literal(expr: &ExprType, what: &str) -> Result<String, String> {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(s)) => Ok(s.value().to_string()),
            _ => Err(format!(
                "rust.bind: the {} must be a string literal, not {}",
                what,
                expr_display(expr)
            )),
        },
        _ => Err(format!(
            "rust.bind: the {} must be a string literal, not {}",
            what,
            expr_display(expr)
        )),
    }
}

/// Best-effort source-ish rendering of an expression for error messages.
fn expr_display(expr: &ExprType) -> String {
    match expr {
        ExprType::Constant(c) => c.to_string(),
        ExprType::Name(n) => n.id.clone(),
        _ => "<expression>".to_string(),
    }
}
