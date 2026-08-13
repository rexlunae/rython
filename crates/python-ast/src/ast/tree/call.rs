use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

use crate::{
    extract_required_attr, CodeGen, CodeGenContext, ExprType, Keyword, PythonOptions,
    SymbolTableNode, SymbolTableScopes,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Call {
    pub func: Box<ExprType>,
    pub args: Vec<ExprType>,
    pub keywords: Vec<Keyword>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Call {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let func: ExprType = extract_required_attr(&ob, "func", "function call expression")?;
        let args: Vec<ExprType> = extract_required_attr(&ob, "args", "function call arguments")?;
        let keywords: Vec<Keyword> = extract_required_attr(&ob, "keywords", "function call keywords")?;
        
        Ok(Call {
            func: Box::new(func),
            args,
            keywords,
        })
    }
}

/// Is this callee functools.partial — via `from functools import
/// partial` (the symbol table maps the bare name to the functools
/// import) or the `functools.partial` attribute spelling? A user
/// function named partial shadows the import in the symbol table and
/// does not match.
fn is_partial_target(func: &ExprType, symbols: &SymbolTableScopes) -> bool {    match func {
        ExprType::Name(n) => {
            n.id == "partial"
                && matches!(
                    symbols.get(&n.id),
                    Some(SymbolTableNode::ImportFrom(i)) if i.module == "functools"
                )
        }
        ExprType::Attribute(attr) => {
            attr.attr == "partial"
                && matches!(attr.value.as_ref(), ExprType::Name(m) if m.id == "functools")
        }
        _ => false,
    }
}

/// Resolve a call target to a numpy function name, when the call is on the
/// numpy module (`np.foo(...)`, `numpy.foo(...)`, `np.linalg.inv(...)`) or
/// on a name imported from it (`from numpy import array; array(...)`).
/// Returns the Python-side function name (`"linalg.inv"` for the nested
/// form). A user definition shadowing the import wins in the symbol table,
/// like the other module dispatches in this file.
fn numpy_target(func: &ExprType, symbols: &SymbolTableScopes) -> Option<String> {
    match func {
        ExprType::Attribute(attr) => match attr.value.as_ref() {
            ExprType::Attribute(inner) => {
                if let ExprType::Name(m) = inner.value.as_ref() {
                    if (m.id == "numpy" || m.id == "np") && inner.attr == "linalg" {
                        return Some(format!("linalg.{}", attr.attr));
                    }
                }
                None
            }
            ExprType::Name(m) if m.id == "numpy" || m.id == "np" => Some(attr.attr.clone()),
            _ => None,
        },
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::ImportFrom(import))
                if import.module == "numpy" || import.module == "numpy.linalg" =>
            {
                let name = n.id.clone();
                Some(if import.module == "numpy.linalg" {
                    format!("linalg.{name}")
                } else {
                    name
                })
            }
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// numpy call lowering
// ---------------------------------------------------------------------------

/// Render one positional argument.
type NpCtx = (CodeGenContext, PythonOptions, SymbolTableScopes);

fn np_render(expr: &ExprType, ctx: &NpCtx) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let tokens = expr
        .clone()
        .to_rust(ctx.0.clone(), ctx.1.clone(), ctx.2.clone())?;
    // The numpy runtime functions take their array/scalar arguments BY
    // VALUE (no borrows, unlike the Python-operator traits), while Python
    // value semantics let a variable be re-used freely after the call.
    // Clone every argument so `a = np.sum(x); b = np.mean(x)` still
    // compiles. Every generated value type (NdArray, i64, f64, bool,
    // String, Vec, Option, tuples) is Clone, so this is always safe.
    Ok(quote!((#tokens).clone()))
}

fn np_kw<'a>(keywords: &'a [Keyword], name: &str) -> Option<&'a ExprType> {
    keywords.iter().find(|k| k.arg.as_deref() == Some(name)).map(|k| &k.value)
}

fn np_has_kw(keywords: &[Keyword], name: &str) -> bool {
    keywords.iter().any(|k| k.arg.as_deref() == Some(name))
}

/// Reject any keyword argument (other than the named ones, if any).
fn np_no_extra_kw(
    fname: &str,
    keywords: &[Keyword],
    allowed: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    for k in keywords {
        if let Some(arg) = k.arg.as_deref() {
            if !allowed.contains(&arg) {
                return Err(format!(
                    "np.{fname}() got an unexpected keyword argument '{arg}' (rython's \
                     numpy subset supports: {})",
                    allowed.join(", ")
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Is this expression a float literal (possibly under unary +/-)?
fn np_is_float_literal(expr: &ExprType) -> bool {
    match expr {
        ExprType::Constant(c) => matches!(c.0, Some(litrs::Literal::Float(_))),
        ExprType::UnaryOp(u) => np_is_float_literal(&u.operand),
        _ => false,
    }
}

/// Is this expression an int literal (possibly under unary +/-)?
fn np_is_int_literal(expr: &ExprType) -> bool {
    match expr {
        ExprType::Constant(c) => matches!(c.0, Some(litrs::Literal::Integer(_))),
        ExprType::UnaryOp(u) => np_is_int_literal(&u.operand),
        _ => false,
    }
}

/// Is this expression the None literal?
fn np_is_none(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Constant(c) if c.0.is_none())
}

/// Is this expression a bool literal?
fn np_is_bool_literal(expr: &ExprType) -> bool {
    matches!(expr, ExprType::Constant(c) if matches!(c.0, Some(litrs::Literal::Bool(_))))
}

/// Map a `dtype=` keyword value (`np.int64`, `"float32"`, ...) to the
/// `numpy::Dtype` path token.
fn np_dtype_tokens(expr: &ExprType) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let name = match expr {
        ExprType::Attribute(attr) => {
            if !matches!(attr.value.as_ref(), ExprType::Name(m) if m.id == "np" || m.id == "numpy") {
                return Err(format!(
                    "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                     or a string like \"float64\" (got an unsupported expression)"
                )
                .into());
            }
            attr.attr.clone()
        }
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::String(s)) => s.value().to_string(),
            _ => {
                return Err(
                    "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                     or a string like \"float64\""
                        .to_string()
                        .into(),
                )
            }
        },
        _ => {
            return Err(
                "dtype= must be one of np.float64/np.float32/np.int64/np.int32/np.bool_ \
                 or a string like \"float64\""
                    .to_string()
                    .into(),
            )
        }
    };
    let variant = match name.as_str() {
        "float64" => "Float64",
        "float32" => "Float32",
        "int64" => "Int64",
        "int32" => "Int32",
        "bool_" | "bool" => "Bool",
        _ => {
            return Err(format!(
                "unsupported numpy dtype '{name}' (rython's numpy subset supports \
                 float64, float32, int64, int32, bool_)"
            )
            .into())
        }
    };
    Ok(quote!(numpy::Dtype::#variant))
}

/// The `numpy::` path prefix for a given numpy function name (handles the
/// `linalg.*` nested functions).
fn np_path(name: &str) -> (String, TokenStream) {
    if let Some(linalg_fn) = name.strip_prefix("linalg.") {
        let ident = crate::safe_ident(linalg_fn);
        (linalg_fn.to_string(), quote!(numpy::linalg::#ident))
    } else {
        let ident = crate::safe_ident(name);
        (name.to_string(), quote!(numpy::#ident))
    }
}

/// Lower `np.<fname>(args, kwargs)` onto the stdpython numpy module.
#[allow(clippy::too_many_lines)]
fn lower_numpy_call(
    fname: &str,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let npc = (ctx, options, symbols);
    let fname = fname.to_string();
    let (plain_name, path) = np_path(&fname);

    // np.float64(x) / np.int64(x) / np.float32(x) / np.int32(x) / np.bool_(x)
    // are dtype CASTS (numpy's scalar types).
    let cast: Option<&str> = match plain_name.as_str() {
        "float64" => Some("f64"),
        "float32" => Some("f32"),
        "int64" => Some("i64"),
        "int32" => Some("i32"),
        _ => None,
    };
    if let Some(ty) = cast {
        np_no_extra_kw(&plain_name, keywords, &[])?;
        if args.len() != 1 {
            return Err(format!(
                "np.{}() takes exactly 1 argument ({} given)",
                plain_name,
                args.len()
            )
            .into());
        }
        let a = np_render(&args[0], &npc)?;
        // `#ty` is a &str; parse it into an actual Rust type token so the
        // cast emits `as f64`, not `as "f64"`.
        let ty_tokens: proc_macro2::TokenStream = ty
            .parse()
            .unwrap_or_else(|_| panic!("numpy cast type '{ty}' is not valid Rust"));
        return Ok(quote!((#a) as #ty_tokens));
    }
    if plain_name == "bool_" {
        np_no_extra_kw(&plain_name, keywords, &[])?;
        if args.len() != 1 {
            return Err(format!(
                "np.bool_() takes exactly 1 argument ({} given)",
                args.len()
            )
            .into());
        }
        let a = np_render(&args[0], &npc)?;
        return Ok(quote!((#a).py_bool()));
    }

    // Construction helpers with shape arguments: shape tuples (2, 3) render
    // as Rust tuples and convert via IntoShape; `dtype=` maps to Dtype.
    let shape_fns: &[&str] = &["zeros", "ones", "empty", "full"];
    if shape_fns.contains(&plain_name.as_str()) {
        np_no_extra_kw(&plain_name, keywords, &["dtype"])?;
        let ndtype = np_kw(keywords, "dtype").map(np_dtype_tokens).transpose()?;
        match plain_name.as_str() {
            "zeros" | "ones" | "empty" => {
                if args.len() != 1 {
                    return Err(format!(
                        "np.{}() takes exactly 1 positional argument (shape), got {}",
                        plain_name,
                        args.len()
                    )
                    .into());
                }
                let shape = np_render(&args[0], &npc)?;
                let dtype = ndtype.unwrap_or(quote!(numpy::Dtype::Float64));
                return Ok(quote!(#path(#shape, #dtype)));
            }
            _ => {
                // full: fill value picks the Rust function (int → full_i,
                // float → full, bool → full_bool); dtype= is not supported
                // yet (it would fight the fill type, like numpy's promotion).
                if ndtype.is_some() {
                    return Err(
                        "np.full(..., dtype=...) is not supported in rython's numpy subset \
                         (the dtype follows the fill value)"
                            .to_string()
                            .into(),
                    );
                }
                if args.len() != 2 {
                    return Err(format!(
                        "np.full() takes exactly 2 arguments (shape, fill_value), got {}",
                        args.len()
                    )
                    .into());
                }
                let shape = np_render(&args[0], &npc)?;
                let fill = np_render(&args[1], &npc)?;
                return if np_is_bool_literal(&args[1]) {
                    Ok(quote!(numpy::full_bool(#shape, #fill)))
                } else if np_is_int_literal(&args[1]) {
                    Ok(quote!(numpy::full_i(#shape, #fill)))
                } else {
                    Ok(quote!(numpy::full(#shape, (#fill) as f64)))
                };
            }
        }
    }

    // Dispatch on the FULL name (fname) so the `linalg.*` arms fire; the
    // plain arms use `plain_name`, which equals fname for non-linalg calls.
    match fname.as_str() {
        "arange" => {
            np_no_extra_kw("arange", keywords, &[])?;
            let n = args.len();
            if !(1..=3).contains(&n) {
                return Err(format!(
                    "np.arange() takes 1 to 3 arguments ({} given)",
                    n
                )
                .into());
            }
            let any_float = args.iter().any(np_is_float_literal);
            let any_int = args.iter().any(np_is_int_literal);
            if any_float && any_int {
                return Err(
                    "np.arange() mixing int and float bounds is not supported in rython's \
                     numpy subset; use all-int or all-float literals"
                        .to_string()
                        .into(),
                );
            }
            let rendered: Vec<TokenStream> = args
                .iter()
                .map(|a| np_render(a, &npc))
                .collect::<Result<_, _>>()?;
            if any_float {
                let (a, b, c) = (
                    rendered
                        .first()
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(0.0)),
                    rendered
                        .get(1)
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(0.0)),
                    rendered
                        .get(2)
                        .map(|t| quote!((#t) as f64))
                        .unwrap_or(quote!(1.0)),
                );
                return Ok(if n == 1 {
                    quote!(numpy::arange_f(#a))
                } else {
                    quote!(numpy::arange_f3(#a, #b, #c))
                });
            }
            let (a, b, c) = (
                rendered
                    .first()
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(0i64)),
                rendered
                    .get(1)
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(1i64)),
                rendered
                    .get(2)
                    .map(|t| quote!((#t) as i64))
                    .unwrap_or(quote!(1i64)),
            );
            Ok(if n == 1 {
                quote!(numpy::arange(#a))
            } else {
                quote!(numpy::arange3(#a, #b, #c))
            })
        }

        "linspace" => {
            np_no_extra_kw("linspace", keywords, &["num", "endpoint"])?;
            if let Some(endpoint) = np_kw(keywords, "endpoint") {
                if !matches!(endpoint, ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::Bool(litrs::BoolLit::True)))) {
                    return Err(
                        "np.linspace(..., endpoint=False) is not supported in rython's \
                         numpy subset (endpoint defaults to True)"
                            .to_string()
                            .into(),
                    );
                }
            }
            if !(2..=3).contains(&args.len()) {
                return Err(format!(
                    "np.linspace() takes 2 or 3 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let start = np_render(&args[0], &npc)?;
            let stop = np_render(&args[1], &npc)?;
            let num = match np_kw(keywords, "num") {
                Some(n) => Some(np_render(n, &npc)?),
                None if args.len() == 3 => Some(np_render(&args[2], &npc)?),
                None => None,
            };
            let num = match num {
                Some(t) => quote!((#t) as i64),
                None => quote!(50i64),
            };
            Ok(quote!(numpy::linspace((#start) as f64, (#stop) as f64, #num)))
        }

        "eye" | "identity" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path((#a) as i64)))
        }

        "dtype" => {
            np_no_extra_kw("dtype", keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.dtype() takes exactly 1 argument ({} given)",
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(numpy::dtype(#a)))
        }

        "set_backend" => {
            np_no_extra_kw("set_backend", keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.set_backend() takes exactly 1 argument ({} given)",
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(numpy::set_backend_by_name(#a)?))
        }

        "sum" | "prod" | "mean" | "max" | "min" | "all" | "any" | "argmax" | "argmin" => {
            if np_has_kw(keywords, "axis") {
                return Err(format!(
                    "np.{}(a, axis=...) is not supported in rython's numpy subset \
                     (full reductions only)",
                    plain_name
                )
                .into());
            }
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }

        "std" | "var" => {
            np_no_extra_kw(&plain_name, keywords, &["ddof"])?;
            if !(1..=2).contains(&args.len()) {
                return Err(format!(
                    "np.{}() takes 1 or 2 arguments ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            let ddof = match np_kw(keywords, "ddof") {
                Some(d) => Some(np_render(d, &npc)?),
                None if args.len() == 2 => Some(np_render(&args[1], &npc)?),
                None => None,
            };
            let ddof = match ddof {
                Some(t) => quote!((#t) as f64),
                None => quote!(0.0),
            };
            Ok(quote!(#path(#a, #ddof)))
        }

        "clip" => {
            np_no_extra_kw("clip", keywords, &["min", "max"])?;
            let lo = np_kw(keywords, "min").or_else(|| args.get(1));
            let hi = np_kw(keywords, "max").or_else(|| args.get(2));
            if args.is_empty() || lo.is_none() || hi.is_none() {
                return Err(
                    "np.clip() needs an array plus min and max bounds (either both \
                     positional or min=/max= keywords); None means unbounded"
                        .to_string()
                        .into(),
                );
            }
            let a = np_render(&args[0], &npc)?;
            let lo = match lo {
                Some(e) if np_is_none(e) => quote!(None),
                Some(e) => {
                    let t = np_render(e, &npc)?;
                    quote!(Some((#t) as f64))
                }
                None => quote!(None),
            };
            let hi = match hi {
                Some(e) if np_is_none(e) => quote!(None),
                Some(e) => {
                    let t = np_render(e, &npc)?;
                    quote!(Some((#t) as f64))
                }
                None => quote!(None),
            };
            Ok(quote!(numpy::clip(#a, #lo, #hi)))
        }

        "where" => {
            np_no_extra_kw("where", keywords, &[])?;
            if args.len() != 3 {
                return Err(
                    "np.where(cond, a, b) needs exactly 3 arguments in rython's numpy \
                     subset (the single-argument index form is not supported)"
                        .to_string()
                        .into(),
                );
            }
            let (c, a, b) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?, np_render(&args[2], &npc)?);
            Ok(quote!(numpy::where_(#c, #a, #b)))
        }

        "concatenate" => {
            np_no_extra_kw("concatenate", keywords, &["axis"])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.concatenate() takes exactly 1 positional argument ({} given)",
                    args.len()
                )
                .into());
            }
            let arrays = np_render(&args[0], &npc)?;
            let axis = match np_kw(keywords, "axis") {
                Some(a) => np_render(a, &npc)?,
                None => quote!(0i64),
            };
            Ok(quote!(numpy::concatenate(#arrays, (#axis) as i64)))
        }

        // Plain 1-arg pass-throughs.
        "array" | "asarray" | "ravel" | "transpose" | "sort" | "argsort" | "negative"
        | "abs" | "square" | "sign" | "isfinite" | "isinf" | "isnan" | "logical_not"
        | "sqrt" | "exp" | "log" | "log2" | "log10" | "sin" | "cos" | "tan" | "arcsin"
        | "arccos" | "arctan" | "sinh" | "cosh" | "tanh" | "floor" | "ceil" | "reciprocal"
        | "expm1" | "log1p" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }

        // np.reshape(a, shape) — the shape tuple converts via IntoShape.
        "reshape" => {
            np_no_extra_kw("reshape", keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.reshape() takes exactly 2 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let (a, s) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            Ok(quote!(numpy::reshape(#a, #s)))
        }

        // Plain 2-arg pass-throughs.
        "add" | "subtract" | "multiply" | "divide" | "floor_divide" | "mod" | "remainder"
        | "power" | "maximum" | "minimum" | "equal" | "not_equal" | "less" | "less_equal"
        | "greater" | "greater_equal" | "bitwise_and" | "bitwise_or" | "bitwise_xor"
        | "logical_and" | "logical_or" | "logical_xor" | "matmul" | "dot" | "vdot" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.{}() takes exactly 2 arguments ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let (a, b) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            let name = if plain_name == "mod" { "mod_" } else { plain_name.as_str() };
            let path = crate::safe_ident(name);
            Ok(quote!(numpy::#path(#a, #b)))
        }

        "vstack" | "hstack" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.{}() takes exactly 1 argument ({} given)",
                    plain_name,
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }

        // linalg.inv / linalg.det / linalg.solve
        "linalg.inv" | "linalg.det" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 1 {
                return Err(format!(
                    "np.linalg.{}() takes exactly 1 argument ({} given)",
                    plain_name.trim_start_matches("linalg."),
                    args.len()
                )
                .into());
            }
            let a = np_render(&args[0], &npc)?;
            Ok(quote!(#path(#a)))
        }
        "linalg.solve" => {
            np_no_extra_kw(&plain_name, keywords, &[])?;
            if args.len() != 2 {
                return Err(format!(
                    "np.linalg.solve() takes exactly 2 arguments ({} given)",
                    args.len()
                )
                .into());
            }
            let (a, b) = (np_render(&args[0], &npc)?, np_render(&args[1], &npc)?);
            Ok(quote!(#path(#a, #b)))
        }

        other => Err(format!(
            "np.{other}() is not supported by rython's numpy subset. Supported: \
             array, zeros, ones, full, empty, arange, linspace, eye, identity, dtype, \
             reshape, ravel, transpose, concatenate, vstack, hstack, clip, where, sort, \
             argsort, sum, prod, mean, max, min, std, var, all, any, argmax, argmin, \
             dot, matmul, vdot, set_backend, the ufuncs (add, subtract, multiply, \
             divide, floor_divide, mod, power, sqrt, exp, log, sin, cos, ...), the \
             comparisons (equal, not_equal, less, greater, ...), the dtype casts \
             (np.float64, np.int64, np.bool_, ...), and np.linalg.{{inv,det,solve}}. \
             See the numpy README for details."
        )
        .into()),
    }
}

impl<'a> CodeGen for Call {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // Calls to functions that return Result<T, PyException> get `?` so
        // exceptions propagate to the caller (or an enclosing try block),
        // as in Python: user-defined functions (known from the symbol
        // table), names imported from user modules, and the Result-returning
        // stdpython builtins.
        let propagates_exceptions = match self.func.as_ref() {
            ExprType::Name(name) => {
                matches!(name.id.as_str(), "int" | "float" | "chr")
                    || match symbols.get(&name.id) {
                        Some(SymbolTableNode::FunctionDef(_)) => true,
                        Some(SymbolTableNode::ImportFrom(import)) => {
                            let root = import.module.split('.').next().unwrap_or("");
                            !crate::is_stdpython_module(root)
                        }
                        // A name bound to functools.partial(f, ...) is a
                        // closure returning f's Result: propagate.
                        Some(SymbolTableNode::Assign { value: ExprType::Call(c), .. }) => {
                            is_partial_target(c.func.as_ref(), &symbols)
                        }
                        _ => false,
                    }
            }
            _ => false,
        };

        // rust.bind / rust.c_bind names: calls lower to direct calls into
        // the bound crate, with type-directed conversions. This comes before
        // every builtin handler — a binding shadows the builtin of the same
        // name exactly as a user function would.
        if let ExprType::Name(n) = self.func.as_ref() {
            if let Some(SymbolTableNode::RustBinding(spec)) = symbols.get(&n.id) {
                return lower_rust_binding_call(
                    spec,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }
        }
        // A rust.bind declaration used as a bare expression (not assigned) is
        // a loud error: bindings must be module-level assignments.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            if matches!(attr.value.as_ref(), ExprType::Name(m) if m.id == "rust")
                && matches!(attr.attr.as_str(), "bind" | "c_bind")
            {
                return Err("rust.bind(...) must be assigned to a name at module level"
                    .to_string()
                    .into());
            }
        }

        // The I/O builtins have no no_std lowering (stdpython gates them
        // behind std): fail at conversion time with the reason, rather than
        // let the generated crate fail with a bare unresolved-name error. A
        // user definition of the same name shadows the builtin as usual.
        if options.no_std {
            if let ExprType::Name(n) = self.func.as_ref() {
                if matches!(n.id.as_str(), "print" | "input" | "open")
                    && symbols.get(&n.id).is_none()
                {
                    return Err(format!(
                        "`{}()` requires OS I/O, which the no_std profile does not \
                         provide; remove the call or convert without the no_std \
                         profile",
                        n.id
                    )
                    .into());
                }
            }
        }

        // numpy subset calls (`np.foo(...)`, `numpy.foo(...)`, and
        // `from numpy import ...` names) map onto the stdpython numpy
        // module at compile time: the runtime has one Rust function per
        // arity/type combination (no overloading, no default arguments),
        // shape tuples need converting, and dtype= keywords need mapping.
        // Anything unsupported fails HERE with a clear message instead of
        // as a cryptic error in the generated crate.
        if let Some(numpy_fn) = numpy_target(self.func.as_ref(), &symbols) {
            return lower_numpy_call(
                &numpy_fn,
                &self.args,
                &self.keywords,
                ctx,
                options,
                symbols,
            );
        }

        // Multi-argument range() maps to the arity-specific runtime
        // functions (Rust has no overloading); the 3-argument form can
        // raise ValueError on a zero step, hence `?`. A user-defined
        // `range` shadows the builtin and skips this mapping.
        if let ExprType::Name(n) = self.func.as_ref() {
            if n.id == "range"
                && symbols.get("range").is_none()
                && self.keywords.is_empty()
                && matches!(self.args.len(), 1 | 2 | 3)
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    // Every range argument is an index/step: `range(len(x))`
                    // passes usize, so coerce to i64 here rather than rely on
                    // the runtime generic (which fails to resolve for
                    // expression arguments of unknown type).
                    rendered.push(crate::render_typed(
                        arg,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                        Some(crate::TypeInfo::Int),
                    )?);
                }
                return Ok(match rendered.len() {
                    1 => quote!(range(#(#rendered),*)),
                    2 => {
                        let (a, b) = (&rendered[0], &rendered[1]);
                        quote!(range_start_stop(#a, #b))
                    }
                    _ => {
                        let (a, b, c) = (&rendered[0], &rendered[1], &rendered[2]);
                        quote!(range_start_stop_step(#a, #b, #c)?)
                    }
                });
            }
        }

        // Builtins with keyword variants or by-reference runtime shapes:
        // min/max (key=, default=, n-ary), sorted (key=, reverse=),
        // enumerate (start=), pow (3-arg modular), and the by-reference
        // len/repr/reversed. Each spelling maps to its runtime variant; a
        // user definition of the same name shadows the builtin, and
        // unknown or duplicate keywords are loud errors, as Python raises
        // TypeError for them.
        if let ExprType::Name(n) = self.func.as_ref() {
            let bname = n.id.as_str();
            if matches!(
                bname,
                "min" | "max" | "sorted" | "enumerate" | "pow" | "len" | "repr"
                    | "reversed" | "frozenset" | "map" | "filter" | "list"
                    | "isinstance" | "hash" | "print" | "open" | "round"
                    | "divmod"
            ) && symbols.get(bname).is_none()
            {
                let mut rendered = Vec::new();
                for arg in &self.args {
                    // Builtin args are borrowed or copied by the runtime;
                    // render them plain — clone-on-reuse is only inserted
                    // for user-function calls, whose params are owned.
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let unexpected = |kw: Option<&str>| -> Box<dyn std::error::Error> {
                    format!(
                        "{}() got an unexpected or duplicate keyword argument '{}'",
                        bname,
                        kw.unwrap_or("**kwargs")
                    )
                    .into()
                };
                match bname {
                    "min" | "max" => {
                        let mut key = None;
                        let mut default = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("default") if default.is_none() => {
                                    default = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err(
                                format!("{}() expected at least 1 argument", bname).into()
                            );
                        }
                        if rendered.len() >= 2 {
                            if key.is_some() || default.is_some() {
                                return Err(format!(
                                    "{}() with multiple positional values and keywords \
                                     is not supported yet; pass a list instead",
                                    bname
                                )
                                .into());
                            }
                            // Python min(a, b, c) folds pairwise; ties keep
                            // the earlier argument.
                            let two = format_ident!("{}2", bname);
                            let mut acc = rendered[0].clone();
                            for next in &rendered[1..] {
                                acc = quote!(#two(#acc, #next));
                            }
                            return Ok(acc);
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        return Ok(match (key, default) {
                            (None, None) => {
                                let f = format_ident!("{}", bname);
                                quote!(#f(&(#a))?)
                            }
                            (Some(k), None) => {
                                let k = render(k)?;
                                let f = format_ident!("{}_key", bname);
                                quote!(#f(&(#a), #k)?)
                            }
                            (None, Some(d)) => {
                                let d = render(d)?;
                                let f = format_ident!("{}_default", bname);
                                quote!(#f(&(#a), #d))
                            }
                            (Some(k), Some(d)) => {
                                let k = render(k)?;
                                let d = render(d)?;
                                let f = format_ident!("{}_key_default", bname);
                                quote!(#f(&(#a), #k, #d))
                            }
                        });
                    }
                    "sorted" => {
                        let mut key = None;
                        let mut reverse = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("key") if key.is_none() => key = Some(kw.value.clone()),
                                Some("reverse") if reverse.is_none() => {
                                    reverse = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.len() != 1 {
                            return Err("sorted() takes exactly one positional argument"
                                .to_string()
                                .into());
                        }
                        let a = &rendered[0];
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        return Ok(match (key, reverse) {
                            (None, None) => quote!(sorted(&(#a))),
                            (Some(k), None) => {
                                let k = render(k)?;
                                quote!(sorted_key(&(#a), #k))
                            }
                            (None, Some(r)) => {
                                let r = render(r)?;
                                quote!(sorted_reverse(&(#a), #r))
                            }
                            (Some(k), Some(r)) => {
                                let k = render(k)?;
                                let r = render(r)?;
                                quote!(sorted_key_reverse(&(#a), #k, #r))
                            }
                        });
                    }
                    "enumerate" => {
                        let mut start = self.args.get(1).cloned();
                        if self.args.len() > 2 {
                            return Err("enumerate() takes at most 2 arguments"
                                .to_string()
                                .into());
                        }
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("start") if start.is_none() => {
                                    start = Some(kw.value.clone())
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        if rendered.is_empty() {
                            return Err("enumerate() expected an iterable".to_string().into());
                        }
                        let a = &rendered[0];
                        return Ok(match start {
                            None => quote!(enumerate(#a)),
                            Some(s) => {
                                let s =
                                    s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                quote!(enumerate_start(#a, #s))
                            }
                        });
                    }
                    "pow" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(
                                self.keywords[0].arg.as_deref(),
                            ));
                        }
                        return match rendered.as_slice() {
                            [b, e] => Ok(quote!(pow(#b, #e))),
                            [b, e, m] => Ok(quote!(pow_mod(#b, #e, #m)?)),
                            _ => Err("pow() takes 2 or 3 arguments".to_string().into()),
                        };
                    }
                    // round(x) -> int (half-even), round(x, n) -> float.
                    // The first argument is always numeric: coerce to f64
                    // so round(3) works; the ndigits argument stays i64.
                    "round" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        let render_float = |e: &crate::ExprType| {
                            crate::render_typed(
                                e,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                                Some(crate::TypeInfo::Float),
                            )
                        };
                        return match self.args.len() {
                            1 => {
                                let a = render_float(&self.args[0])?;
                                Ok(quote!(round(#a)))
                            }
                            2 => {
                                let a = render_float(&self.args[0])?;
                                let n = crate::render_typed(
                                    &self.args[1],
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                    Some(crate::TypeInfo::Int),
                                )?;
                                Ok(quote!(round_digits(#a, #n)))
                            }
                            _ => Err("round() takes 1 or 2 arguments".to_string().into()),
                        };
                    }
                    // divmod(a, b) lowers to the stdpython helper, whose
                    // floor-division and modulus steps can raise
                    // ZeroDivisionError: the Result it returns propagates
                    // with `?` like the other fallible builtins.
                    "divmod" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        return match rendered.as_slice() {
                            [a, b] => Ok(quote!(divmod(#a, #b)?)),
                            _ => Err("divmod() takes exactly 2 arguments".to_string().into()),
                        };
                    }
                    // isinstance is statically decidable in a typed
                    // lowering: it becomes the constant true/false when the
                    // argument's type is known (annotation or literal), and
                    // a loud error when it is not.
                    "isinstance" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if self.args.len() != 2 {
                            return Err("isinstance() takes exactly 2 arguments"
                                .to_string()
                                .into());
                        }
                        let target = match &self.args[1] {
                            ExprType::Name(t)
                                if matches!(
                                    t.id.as_str(),
                                    "int" | "float" | "str" | "bool"
                                ) =>
                            {
                                t.id.clone()
                            }
                            other => {
                                return Err(format!(
                                    "isinstance() second argument must be int, float, \
                                     str, or bool (got `{:?}`); tuples of types are not \
                                     supported yet",
                                    other
                                )
                                .into());
                            }
                        };
                        let actual: Option<String> = match &self.args[0] {
                            ExprType::Name(n) => options.local_types.get(&n.id).cloned(),
                            lit => crate::ast::tree::function_def::simple_expr_type(lit)
                                .map(|ty| match ty.to_string().as_str() {
                                    "i64" => "int".to_string(),
                                    "f64" => "float".to_string(),
                                    "bool" => "bool".to_string(),
                                    _ => "str".to_string(),
                                }),
                        };
                        let Some(actual) = actual else {
                            return Err(format!(
                                "isinstance(): the type of `{:?}` is not statically \
                                 known; annotate it (or assign it a literal) so the \
                                 check can be decided at conversion time",
                                self.args[0]
                            )
                            .into());
                        };
                        // bool is a subclass of int in Python.
                        let result = actual == target
                            || (actual == "bool" && target == "int");
                        return Ok(if result { quote!(true) } else { quote!(false) });
                    }
                    // The by-reference builtins: their runtime functions
                    // borrow, and Python's calls never consume the value.
                    "print" => {
                        // print builds on py_display (Python's str
                        // semantics: True, 1e+16, unquoted strings) — the
                        // Display fallback would silently diverge.
                        let mut sep = None;
                        let mut end = None;
                        let mut flush = None;
                        for kw in &self.keywords {
                            match kw.arg.as_deref() {
                                Some("sep") if sep.is_none() => sep = Some(kw.value.clone()),
                                Some("end") if end.is_none() => end = Some(kw.value.clone()),
                                Some("flush") if flush.is_none() => {
                                    flush = Some(kw.value.clone())
                                }
                                Some("file") => {
                                    return Err("print(file=...) is not supported: \
                                                generated code writes to stdout only"
                                        .to_string()
                                        .into());
                                }
                                other => return Err(unexpected(other)),
                            }
                        }
                        // sep=None / end=None mean the defaults in Python.
                        let sep = sep.filter(|s| !crate::is_none_expr(s));
                        let end = end.filter(|e| !crate::is_none_expr(e));
                        let render = |e: crate::ExprType| {
                            e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                        };
                        if sep.is_none() && end.is_none() && flush.is_none() {
                            match rendered.as_slice() {
                                [] => return Ok(quote!(println!())),
                                [a] => return Ok(quote!(print(&(#a)))),
                                _ => {}
                            }
                        }
                        let sep = match sep {
                            Some(s) => render(s)?,
                            None => quote!(" "),
                        };
                        let end = match end {
                            Some(e) => render(e)?,
                            None => quote!("\n"),
                        };
                        // print(end="") with no arguments still needs an
                        // element type for the empty parts slice.
                        let parts = if rendered.is_empty() {
                            quote!(&[] as &[&str])
                        } else {
                            quote!(&[#(py_display(&(#rendered))),*])
                        };
                        return Ok(match flush {
                            None => quote!(print_parts(#parts, #sep, #end)),
                            Some(f) => {
                                let f = render(f)?;
                                quote!(print_parts_flush(#parts, #sep, #end, #f))
                            }
                        });
                    }
                    "open" => {
                        // The runtime signature takes mode as an Option;
                        // the arity split wraps it here. Text modes only —
                        // encoding/newline/binary spellings are loud.
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        return Ok(match rendered.as_slice() {
                            [p] => quote!(open(&(#p), None::<&str>)?),
                            [p, m] => quote!(open(&(#p), Some(#m))?),
                            _ => {
                                return Err(
                                    "open() takes 1 or 2 arguments (path and mode)"
                                        .to_string()
                                        .into(),
                                )
                            }
                        });
                    }
                    "len" | "repr" | "reversed" | "hash" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                format!("{}() takes exactly one argument", bname).into()
                            );
                        }
                        let f = format_ident!("{}", bname);
                        let a = &rendered[0];
                        return Ok(quote!(#f(&(#a))));
                    }
                    "frozenset" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                "frozenset() requires an iterable argument in rython \
                                 (an empty frozenset has no inferable element type)"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let a = &rendered[0];
                        return Ok(quote!(frozenset(#a)));
                    }
                    // map/filter dispatch on the FUNCTION argument's shape:
                    // lambdas are plain closures, while user-defined
                    // functions return Result and route through the
                    // fallible variants so their exceptions propagate.
                    "map" | "filter" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        let fallible = matches!(self.args.first(), Some(ExprType::Name(f))
                            if matches!(symbols.get(&f.id), Some(SymbolTableNode::FunctionDef(_))));
                        if bname == "filter" {
                            if rendered.len() != 2 {
                                return Err("filter() takes a function and an iterable"
                                    .to_string()
                                    .into());
                            }
                            let (f, xs) = (&rendered[0], &rendered[1]);
                            // filter(None, xs) keeps the truthy elements.
                            if self
                                .args
                                .first()
                                .is_some_and(crate::is_none_expr)
                            {
                                return Ok(quote!(filter_truthy(#xs)));
                            }
                            return Ok(if fallible {
                                quote!(filter_fallible(#f, #xs)?)
                            } else {
                                quote!(filter(#f, #xs))
                            });
                        }
                        return match rendered.as_slice() {
                            [f, xs] => Ok(if fallible {
                                quote!(map_fallible(#f, #xs)?)
                            } else {
                                quote!(map(#f, #xs))
                            }),
                            [f, a, b] => {
                                if fallible {
                                    return Err(
                                        "map() over two iterables with a user-defined \
                                         function is not supported yet; use a lambda"
                                            .to_string()
                                            .into(),
                                    );
                                }
                                Ok(quote!(map2(#f, #a, #b)))
                            }
                            _ => Err("map() takes a function and 1-2 iterables"
                                .to_string()
                                .into()),
                        };
                    }
                    "list" => {
                        if !self.keywords.is_empty() {
                            return Err(unexpected(self.keywords[0].arg.as_deref()));
                        }
                        if rendered.len() != 1 {
                            return Err(
                                "list() requires an iterable argument in rython (an \
                                 empty list has no inferable element type; use [])"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let a = &rendered[0];
                        return Ok(quote!(list(#a)));
                    }
                    _ => unreachable!(),
                }
            }
        }

        // datetime constructors imported via `from datetime import ...`:
        // date/datetime/timedelta calls resolve their positional and
        // keyword arguments against the Python signatures and lower to the
        // runtime ::new constructors (Option-typed for the defaulted
        // parameters). date/datetime validate and propagate with `?`.
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_datetime = matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::ImportFrom(import))
                    if import.module == "datetime"
            );
            if from_datetime && matches!(n.id.as_str(), "date" | "datetime" | "timedelta") {
                let (params, required): (&[&str], usize) = match n.id.as_str() {
                    "date" => (&["year", "month", "day"], 3),
                    "datetime" => (
                        &["year", "month", "day", "hour", "minute", "second", "microsecond"],
                        3,
                    ),
                    _ => (
                        &[
                            "days",
                            "seconds",
                            "microseconds",
                            "milliseconds",
                            "minutes",
                            "hours",
                            "weeks",
                        ],
                        0,
                    ),
                };
                if self.args.len() > params.len() {
                    return Err(format!(
                        "{}() takes at most {} arguments ({} given)",
                        n.id,
                        params.len(),
                        self.args.len()
                    )
                    .into());
                }
                let mut slots: Vec<Option<crate::ExprType>> = vec![None; params.len()];
                for (i, arg) in self.args.iter().enumerate() {
                    slots[i] = Some(arg.clone());
                }
                for kw in &self.keywords {
                    let idx = kw
                        .arg
                        .as_deref()
                        .and_then(|k| params.iter().position(|p| *p == k));
                    match idx {
                        Some(i) if slots[i].is_none() => slots[i] = Some(kw.value.clone()),
                        Some(i) => {
                            return Err(format!(
                                "{}() got multiple values for argument '{}'",
                                n.id, params[i]
                            )
                            .into());
                        }
                        None => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                n.id,
                                kw.arg.as_deref().unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let mut rendered = Vec::new();
                for (i, slot) in slots.iter().enumerate() {
                    let tok = match slot {
                        Some(e) => {
                            let v = e.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            if i < required { v } else { quote!(Some(#v)) }
                        }
                        None if i < required => {
                            return Err(format!(
                                "{}() missing required argument: '{}'",
                                n.id, params[i]
                            )
                            .into());
                        }
                        None => quote!(None),
                    };
                    rendered.push(tok);
                }
                let ty = crate::safe_ident(&n.id);
                let call = quote!(#ty::new(#(#rendered),*));
                // timedelta::new is infallible; date/datetime validate.
                return Ok(if n.id == "timedelta" { call } else { quote!(#call?) });
            }
        }

        // itertools functions imported via `from itertools import ...`:
        // keyword spellings (initial=, repeat=, fillvalue=, key=) map to
        // arity-specific runtime variants, and iterable arguments are
        // borrowed (the runtime takes slices; Python calls never consume).
        if let ExprType::Name(n) = self.func.as_ref() {
            let from_itertools = matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::ImportFrom(import))
                    if import.module == "itertools"
            );
            let handled = matches!(
                n.id.as_str(),
                "accumulate"
                    | "product"
                    | "zip_longest"
                    | "groupby"
                    | "pairwise"
                    | "combinations"
                    | "combinations_with_replacement"
                    | "permutations"
                    | "starmap"
            );
            if from_itertools && handled {
                let name = n.id.as_str();
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                let render = |e: crate::ExprType| {
                    e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                };
                let kw_of = |allowed: &[&str]| -> Result<
                    Vec<Option<crate::ExprType>>,
                    Box<dyn std::error::Error>,
                > {
                    let mut out: Vec<Option<crate::ExprType>> = vec![None; allowed.len()];
                    for kw in &self.keywords {
                        let idx = kw
                            .arg
                            .as_deref()
                            .and_then(|k| allowed.iter().position(|a| *a == k));
                        match idx {
                            Some(i) if out[i].is_none() => out[i] = Some(kw.value.clone()),
                            _ => {
                                return Err(format!(
                                    "{}() got an unexpected or duplicate keyword \
                                     argument '{}'",
                                    name,
                                    kw.arg.as_deref().unwrap_or("**kwargs")
                                )
                                .into());
                            }
                        }
                    }
                    Ok(out)
                };
                match name {
                    "accumulate" => {
                        let kws = kw_of(&["initial"])?;
                        let initial = kws.into_iter().next().unwrap();
                        let (xs, func) = match rendered.as_slice() {
                            [xs] => (xs.clone(), None),
                            [xs, f] => (xs.clone(), Some(f.clone())),
                            _ => {
                                return Err("accumulate() takes 1 or 2 positional \
                                            arguments"
                                    .to_string()
                                    .into());
                            }
                        };
                        return Ok(match (func, initial) {
                            (None, None) => quote!(accumulate_sum(&(#xs))),
                            (Some(f), None) => quote!(accumulate_func(&(#xs), #f)),
                            (None, Some(init)) => {
                                let init = render(init)?;
                                quote!(accumulate_sum_initial(&(#xs), #init))
                            }
                            (Some(f), Some(init)) => {
                                let init = render(init)?;
                                quote!(accumulate_func_initial(&(#xs), #f, #init))
                            }
                        });
                    }
                    "product" => {
                        let kws = kw_of(&["repeat"])?;
                        let repeat = kws.into_iter().next().unwrap();
                        if let Some(r) = repeat {
                            if rendered.len() != 1 {
                                return Err("product(iterable, repeat=n) takes one \
                                            iterable"
                                    .to_string()
                                    .into());
                            }
                            let r = render(r)?;
                            let xs = &rendered[0];
                            return match r.to_string().as_str() {
                                "2" => Ok(quote!(product_repeat2(&(#xs)))),
                                "3" => Ok(quote!(product_repeat3(&(#xs)))),
                                other => Err(format!(
                                    "product() repeat must be the literal 2 or 3 \
                                     (tuple arity is a compile-time shape); got {}",
                                    other
                                )
                                .into()),
                            };
                        }
                        return match rendered.as_slice() {
                            [a, b] => Ok(quote!(product2(&(#a), &(#b)))),
                            [a, b, c] => Ok(quote!(product3(&(#a), &(#b), &(#c)))),
                            _ => Err("product() supports 2 or 3 iterables, or one \
                                      iterable with repeat=2/3"
                                .to_string()
                                .into()),
                        };
                    }
                    "zip_longest" => {
                        let kws = kw_of(&["fillvalue"])?;
                        let fill = kws.into_iter().next().unwrap();
                        if rendered.len() != 2 {
                            return Err("zip_longest() supports exactly 2 iterables"
                                .to_string()
                                .into());
                        }
                        let (a, b) = (&rendered[0], &rendered[1]);
                        return Ok(match fill {
                            Some(v) => {
                                let v = render(v)?;
                                quote!(zip_longest_fill(&(#a), &(#b), #v))
                            }
                            None => quote!(zip_longest(&(#a), &(#b))),
                        });
                    }
                    "groupby" => {
                        let kws = kw_of(&["key"])?;
                        let key = kws.into_iter().next().unwrap();
                        if rendered.len() != 1 {
                            return Err("groupby() takes one iterable".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(match key {
                            Some(f) => {
                                let f = render(f)?;
                                quote!(groupby_key(&(#xs), #f))
                            }
                            None => quote!(groupby(&(#xs))),
                        });
                    }
                    "pairwise" => {
                        kw_of(&[])?;
                        if rendered.len() != 1 {
                            return Err("pairwise() takes one iterable".to_string().into());
                        }
                        let xs = &rendered[0];
                        return Ok(quote!(pairwise(&(#xs))));
                    }
                    "combinations" | "combinations_with_replacement" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err(
                                format!("{}() takes an iterable and r", name).into()
                            );
                        }
                        let f = format_ident!("{}", name);
                        let (xs, r) = (&rendered[0], &rendered[1]);
                        // Negative r raises ValueError, hence the `?`.
                        return Ok(quote!(#f(&(#xs), #r)?));
                    }
                    "permutations" => {
                        kw_of(&[])?;
                        return match rendered.as_slice() {
                            [xs] => Ok(quote!(permutations(&(#xs), None)?)),
                            [xs, r] => Ok(quote!(permutations(&(#xs), Some(#r))?)),
                            _ => Err("permutations() takes an iterable and optional r"
                                .to_string()
                                .into()),
                        };
                    }
                    "starmap" => {
                        kw_of(&[])?;
                        if rendered.len() != 2 {
                            return Err("starmap() takes a function and an iterable"
                                .to_string()
                                .into());
                        }
                        let (f, xs) = (&rendered[0], &rendered[1]);
                        return Ok(quote!(starmap(#f, &(#xs))));
                    }
                    _ => unreachable!(),
                }
            }
        }

        // functools.partial over a STATICALLY-KNOWN function: the
        // signature comes from the symbol table, so the call lowers to a
        // move closure binding the leading arguments, with the remaining
        // parameters keeping their Python names. The closure returns the
        // function's Result directly; calls through the bound name get
        // `?` (see propagates_exceptions). Dynamic first arguments have
        // no signature to consult and are a loud error.
        if is_partial_target(self.func.as_ref(), &symbols) {
            if !self.keywords.is_empty() {
                return Err("functools.partial with keyword arguments is not \
                            supported yet; bind the arguments positionally"
                    .to_string()
                    .into());
            }
            let Some(ExprType::Name(f)) = self.args.first() else {
                return Err("functools.partial requires a function defined in this \
                            module as its first argument"
                    .to_string()
                    .into());
            };
            let Some(SymbolTableNode::FunctionDef(fdef)) = symbols.get(&f.id) else {
                return Err(format!(
                    "functools.partial: `{}` is not a function defined in this \
                     module (only statically-known functions can be bound)",
                    f.id
                )
                .into());
            };
            let params: Vec<String> =
                fdef.args.args.iter().map(|p| p.arg.clone()).collect();
            let bound_n = self.args.len() - 1;
            if bound_n > params.len() {
                return Err(format!(
                    "functools.partial: `{}` takes {} argument(s), but {} were bound",
                    f.id,
                    params.len(),
                    bound_n
                )
                .into());
            }
            let mut bound = Vec::new();
            for arg in &self.args[1..] {
                bound.push(arg.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            let rest: Vec<_> = params[bound_n..]
                .iter()
                .map(|p| crate::safe_ident(p))
                .collect();
            let fident = crate::safe_ident(&f.id);
            return Ok(quote!(move |#(#rest),*| #fident(#(#bound,)* #(#rest),*)));
        }

        // functools/heapq/copy/textwrap/re functions: their runtime shapes
        // borrow (or mutably borrow) arguments, reduce() splits by arity,
        // and the re functions validate patterns at runtime (hence `?`).
        // Handled for both `from X import f; f(...)` and `import X;
        // X.f(...)` spellings.
        {
            let target: Option<(String, Option<&'static str>)> = match self.func.as_ref() {
                ExprType::Name(n) => match symbols.get(&n.id) {
                    Some(SymbolTableNode::ImportFrom(import))
                        if matches!(
                            import.module.as_str(),
                            "functools" | "heapq" | "copy" | "textwrap" | "re" | "hashlib"
                                | "csv" | "io"
                        ) =>
                    {
                        Some((n.id.clone(), None))
                    }
                    _ => None,
                },
                ExprType::Attribute(attr) => match attr.value.as_ref() {
                    ExprType::Name(m)
                        if matches!(
                            m.id.as_str(),
                            "functools" | "heapq" | "textwrap" | "re" | "hashlib" | "csv"
                                | "io"
                        )
                        && !module_name_shadowed(&m.id, &symbols) =>
                    {
                        let module: &'static str = match m.id.as_str() {
                            "functools" => "functools",
                            "heapq" => "heapq",
                            "re" => "re",
                            "hashlib" => "hashlib",
                            "csv" => "csv",
                            "io" => "io",
                            _ => "textwrap",
                        };
                        Some((attr.attr.clone(), Some(module)))
                    }
                    _ => None,
                },
                _ => None,
            };
            let known = target.as_ref().is_some_and(|(f, _)| {
                matches!(
                    f.as_str(),
                    "reduce"
                        | "heappush"
                        | "heappop"
                        | "heapify"
                        | "heappushpop"
                        | "heapreplace"
                        | "nlargest"
                        | "nsmallest"
                        | "copy"
                        | "deepcopy"
                        | "dedent"
                        | "indent"
                        | "search"
                        | "match"
                        | "fullmatch"
                        | "findall"
                        | "finditer"
                        | "sub"
                        | "split"
                        | "md5"
                        | "sha1"
                        | "sha256"
                        | "sha512"
                        | "wrap"
                        | "fill"
                        | "reader"
                        | "writer"
                        | "StringIO"
                )
            });
            if let (Some((fname, module_prefix)), true) = (target, known) {
                // wrap/fill accept width=, the re functions accept
                // flags= (and sub also count=); everything else takes no
                // keywords.
                let is_re_fn = matches!(
                    fname.as_str(),
                    "search" | "match" | "fullmatch" | "findall" | "finditer" | "sub"
                        | "split"
                );
                let mut width_kw: Option<crate::ExprType> = None;
                let mut flags_kw: Option<crate::ExprType> = None;
                let mut count_kw: Option<crate::ExprType> = None;
                let mut maxsplit_kw: Option<crate::ExprType> = None;
                for kw in &self.keywords {
                    let slot = match kw.arg.as_deref() {
                        Some("width")
                            if matches!(fname.as_str(), "wrap" | "fill") =>
                        {
                            &mut width_kw
                        }
                        Some("flags") if is_re_fn => &mut flags_kw,
                        Some("count") if fname == "sub" => &mut count_kw,
                        Some("maxsplit") if fname == "split" => &mut maxsplit_kw,
                        _ => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                fname,
                                kw.arg.as_deref().unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    };
                    if slot.is_some() {
                        return Err(format!(
                            "{}() got multiple values for a keyword argument",
                            fname
                        )
                        .into());
                    }
                    *slot = Some(kw.value.clone());
                }
                // Python re flags lower to inline flag letters: single
                // constants or |-combinations of re.IGNORECASE/I,
                // re.MULTILINE/M, re.DOTALL/S. Anything else is loud.
                fn flag_letters(
                    symbols: &crate::SymbolTableScopes,
                    e: &crate::ExprType,
                ) -> Result<String, Box<dyn std::error::Error>> {
                    let name_of = |id: &str| -> Result<String, Box<dyn std::error::Error>> {
                        match id {
                            "IGNORECASE" | "I" => Ok("i".to_string()),
                            "MULTILINE" | "M" => Ok("m".to_string()),
                            "DOTALL" | "S" => Ok("s".to_string()),
                            other => Err(format!(
                                "unsupported re flag `{}`; supported: IGNORECASE,                                  MULTILINE, DOTALL (and | combinations)",
                                other
                            )
                            .into()),
                        }
                    };
                    match e {
                        ExprType::Attribute(a)
                            if matches!(a.value.as_ref(), ExprType::Name(m)
                                if m.id == "re" && !module_name_shadowed("re", symbols)) =>
                        {
                            name_of(&a.attr)
                        }
                        ExprType::Name(n) => name_of(&n.id),
                        ExprType::BinOp(b)
                            if matches!(b.op, crate::ast::tree::bin_ops::BinOps::BitOr) =>
                        {
                            Ok(format!(
                                "{}{}",
                                flag_letters(symbols, &b.left)?,
                                flag_letters(symbols, &b.right)?
                            ))
                        }
                        other => Err(format!(
                            "unsupported re flags expression `{:?}`; use re.IGNORECASE,                              re.MULTILINE, re.DOTALL, or | combinations of them",
                            other
                        )
                        .into()),
                    }
                }
                let mut rendered = Vec::new();
                for arg in &self.args {
                    rendered.push(arg.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?);
                }
                // The heap mutators take their first argument by &mut, so
                // it must be lowered as a PLACE: `heappush(rows[i], v)`
                // through the Load path would clone the element and the
                // push would silently vanish (the same clone-mutation bug
                // fixed for mutating methods on subscripted receivers).
                // rendered[0] becomes the full mutable-borrow expression:
                // py_index_mut already yields &mut for subscripts, names
                // take a fresh &mut.
                let heap_mutator = matches!(
                    fname.as_str(),
                    "heappush" | "heappop" | "heapify" | "heappushpop" | "heapreplace"
                );
                if heap_mutator {
                    if let Some(first) = self.args.first() {
                        rendered[0] = if matches!(first, ExprType::Subscript(_)) {
                            crate::ast::tree::subscript::subscript_receiver_place(
                                first,
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?
                        } else {
                            let v = &rendered[0];
                            quote!(&mut (#v))
                        };
                    }
                }
                let qual = |name: &str| {
                    let f = crate::safe_ident(name);
                    match module_prefix {
                        Some(m) => {
                            let m = format_ident!("{}", m);
                            quote!(#m::#f)
                        }
                        None => quote!(#f),
                    }
                };
                let arity = |expected: &str| -> Box<dyn std::error::Error> {
                    format!("{}() takes {} arguments", fname, expected).into()
                };
                return match (fname.as_str(), rendered.as_slice()) {
                    ("reduce", [f, xs]) => {
                        let p = qual("reduce");
                        Ok(quote!(#p(#f, &(#xs))?))
                    }
                    ("reduce", [f, xs, init]) => {
                        let p = qual("reduce_initial");
                        Ok(quote!(#p(#f, &(#xs), #init)))
                    }
                    ("reduce", _) => Err(arity("2 or 3")),
                    ("heappush", [h, x]) => {
                        let p = qual("heappush");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heappop", [h]) => {
                        let p = qual("heappop");
                        Ok(quote!(#p(#h)?))
                    }
                    ("heapify", [h]) => {
                        let p = qual("heapify");
                        Ok(quote!(#p(#h)))
                    }
                    ("heappushpop", [h, x]) => {
                        let p = qual("heappushpop");
                        Ok(quote!(#p(#h, #x)))
                    }
                    ("heapreplace", [h, x]) => {
                        let p = qual("heapreplace");
                        Ok(quote!(#p(#h, #x)?))
                    }
                    ("nlargest" | "nsmallest", [n_arg, xs]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(#n_arg, &(#xs))))
                    }
                    ("copy" | "deepcopy", [x]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#x))))
                    }
                    ("dedent", [s]) => {
                        let p = qual("dedent");
                        Ok(quote!(#p(&(#s))))
                    }
                    // wrap/fill: width by position, keyword, or Python's
                    // default of 70. They validate width, hence `?`.
                    ("wrap" | "fill", [t]) | ("wrap" | "fill", [t, _]) => {
                        let p = qual(&fname);
                        let width = match (rendered.get(1), width_kw) {
                            (Some(_), Some(_)) => {
                                return Err(format!(
                                    "{}() got multiple values for argument 'width'",
                                    fname
                                )
                                .into());
                            }
                            (Some(w), None) => quote!(#w),
                            (None, Some(w)) => {
                                let w = w.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#w)
                            }
                            (None, None) => quote!(70),
                        };
                        Ok(quote!(#p(&(#t), #width)?))
                    }
                    (
                        "search" | "match" | "fullmatch" | "findall" | "finditer",
                        [pat, text, ..],
                    ) => {
                        if rendered.len() > 3 {
                            return Err(format!(
                                "{}() takes at most 3 positional arguments",
                                fname
                            )
                            .into());
                        }
                        if rendered.len() > 2 && flags_kw.is_some() {
                            return Err(format!(
                                "{}() got multiple values for argument 'flags'",
                                fname
                            )
                            .into());
                        }
                        let flags = match (self.args.get(2), flags_kw) {
                            (Some(e), None) => flag_letters(&symbols, e)?,
                            (None, Some(e)) => flag_letters(&symbols, &e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        // findall's result SHAPE depends on the pattern's
                        // capture-group count (strings for 0-1 groups,
                        // tuples beyond), so a literal pattern is compiled
                        // here at conversion time to pick the variant —
                        // which also surfaces bad patterns before the
                        // program ever runs. Non-literal patterns keep the
                        // string shape; 2+ groups there stay a loud
                        // runtime error.
                        let mut target = fname.clone();
                        if fname == "findall" {
                            if let ExprType::Constant(c) = &self.args[0] {
                                if let Some(litrs::Literal::String(slit)) = &c.0 {
                                    let pattern = slit.value();
                                    let re = regex::Regex::new(&pattern).map_err(|e| {
                                        format!(
                                            "re.findall(): cannot compile pattern {:?}: {} \
                                             (the regex engine does not support Python's \
                                             backreferences or lookarounds)",
                                            pattern, e
                                        )
                                    })?;
                                    match re.captures_len() - 1 {
                                        0 | 1 => {}
                                        2 => target = "findall2".to_string(),
                                        3 => target = "findall3".to_string(),
                                        n => {
                                            return Err(format!(
                                                "re.findall() with {} capture groups is \
                                                 not supported yet (at most 3)",
                                                n
                                            )
                                            .into());
                                        }
                                    }
                                }
                            }
                        }
                        let p = qual(&target);
                        Ok(quote!(#p(&(#pat), &(#text), #flags)?))
                    }
                    // re.split(pattern, string, maxsplit=0, flags=0):
                    // the THIRD positional is maxsplit, unlike the other
                    // re functions where it is flags.
                    ("split", [pat, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("split() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 2 && maxsplit_kw.is_some() {
                            return Err(
                                "split() got multiple values for argument 'maxsplit'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        if rendered.len() > 3 && flags_kw.is_some() {
                            return Err(
                                "split() got multiple values for argument 'flags'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let maxsplit = match (rendered.get(2), maxsplit_kw) {
                            (Some(m), None) => quote!(#m),
                            (None, Some(m)) => {
                                let m = m.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#m)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match (self.args.get(3), flags_kw) {
                            (Some(e), None) => flag_letters(&symbols, e)?,
                            (None, Some(e)) => flag_letters(&symbols, &e)?,
                            (None, None) => String::new(),
                            _ => unreachable!(),
                        };
                        let p = qual("split");
                        Ok(quote!(#p(&(#pat), &(#text), #maxsplit, #flags)?))
                    }
                    ("sub", [pat, repl, text, ..]) => {
                        if rendered.len() > 4 {
                            return Err("sub() takes at most 4 positional arguments"
                                .to_string()
                                .into());
                        }
                        if rendered.len() > 3 && count_kw.is_some() {
                            return Err(
                                "sub() got multiple values for argument 'count'"
                                    .to_string()
                                    .into(),
                            );
                        }
                        let count = match (rendered.get(3), count_kw) {
                            (Some(c), None) => quote!(#c),
                            (None, Some(c)) => {
                                let c = c.to_rust(
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                )?;
                                quote!(#c)
                            }
                            (None, None) => quote!(0),
                            _ => unreachable!(),
                        };
                        let flags = match flags_kw {
                            Some(e) => flag_letters(&symbols, &e)?,
                            None => String::new(),
                        };
                        let p = qual("sub");
                        Ok(quote!(#p(&(#pat), &(#repl), &(#text), #count, #flags)?))
                    }
                    // hashlib constructors: with initial data, or the
                    // empty + update() idiom.
                    ("md5" | "sha1" | "sha256" | "sha512", [data]) => {
                        let p = qual(&fname);
                        Ok(quote!(#p(&(#data))))
                    }
                    // io.StringIO: arity split — the seeded form starts
                    // with the cursor at 0, as in Python.
                    ("StringIO", []) => {
                        let p = qual("StringIO");
                        Ok(quote!(#p()))
                    }
                    ("StringIO", [initial]) => {
                        let p = qual("StringIO_seeded");
                        Ok(quote!(#p(&(#initial))))
                    }
                    // csv.writer(f) borrows the file mutably for the
                    // writer's lifetime (scope analysis marks f mut).
                    ("writer", [f]) => {
                        let p = qual("writer");
                        Ok(quote!(#p(&mut (#f))))
                    }
                    ("reader", [lines]) => {
                        let p = qual("reader");
                        Ok(quote!(#p(&(#lines))?))
                    }
                    ("md5" | "sha1" | "sha256" | "sha512", []) => {
                        let p = qual(&format!("{}_new", fname));
                        Ok(quote!(#p()))
                    }
                    ("indent", [s, prefix]) => {
                        let p = qual("indent");
                        Ok(quote!(#p(&(#s), &(#prefix))))
                    }
                    ("heappush" | "heappushpop" | "heapreplace" | "indent"
                        | "nlargest" | "nsmallest", _) => Err(arity("2")),
                    _ => Err(arity("the documented number of")),
                };
            }
        }

        // Constructing a class instance: `Point(args)` lowers to
        // `Point::new(args)?`, with arguments resolved against __init__'s
        // signature (minus self) so keywords and defaults follow Python
        // call semantics.
        if let ExprType::Name(n) = self.func.as_ref() {
            if let Some(SymbolTableNode::ClassDef(c)) = symbols.get(&n.id) {
                let cname = crate::safe_ident(&n.id);
                match c.init_method() {
                    Some(init) => {
                        if init.args.vararg.is_some() || init.args.kwarg.is_some() {
                            return Err(format!(
                                "`{}.__init__` takes *args/**kwargs, which is not \
                                 supported yet",
                                n.id
                            )
                            .into());
                        }
                        let mut sig = init.clone();
                        crate::strip_self(&mut sig.args);
                        let MappedArguments { prelude, args } = map_call_arguments(
                            &sig,
                            &self.args,
                            &self.keywords,
                            &ctx,
                            &options,
                            &symbols,
                        )?;
                        return Ok(quote!({ #prelude #cname::new(#(#args),*)? }));
                    }
                    None => {
                        if !self.args.is_empty() || !self.keywords.is_empty() {
                            return Err(format!(
                                "{}() takes no arguments: the class defines no __init__",
                                n.id
                            )
                            .into());
                        }
                        return Ok(quote!(#cname::new()?));
                    }
                }
            }
        }

        // Python methods whose Rust inherent namesakes have DIFFERENT
        // semantics (or the wrong shape) are rewritten here; methods with no
        // Rust conflict resolve through the stdpython PyListOps/PyStrOps
        // traits without any rewriting.
        if let ExprType::Attribute(attr) = self.func.as_ref() {
            // A method call on a receiver whose class is known — `self`
            // inside a method, or a name assigned a construction — resolves
            // against the class's own methods FIRST, so a user-defined
            // method named like a builtin (`get`, `pop`, ...) is not
            // rewritten out from under the class. Calls propagate
            // exceptions (`?`) and map keywords/defaults like any user
            // function call.
            if let Some(class) = receiver_class(&attr.value, &ctx, &symbols) {
                if let Some(method) = class.methods().find(|m| m.name == attr.attr).cloned() {
                    if method.args.vararg.is_some() || method.args.kwarg.is_some() {
                        return Err(format!(
                            "`{}.{}` takes *args/**kwargs, which is not supported yet",
                            class.name, method.name
                        )
                        .into());
                    }
                    let mut sig = method;
                    crate::strip_self(&mut sig.args);
                    let MappedArguments { prelude, args } = map_call_arguments(
                        &sig,
                        &self.args,
                        &self.keywords,
                        &ctx,
                        &options,
                        &symbols,
                    )?;
                    let receiver = attr.value.clone().to_rust(
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    let method_name = crate::safe_ident(&attr.attr);
                    return Ok(quote!({ #prelude (#receiver).#method_name(#(#args),*)? }));
                }
            }
            // A mutating method on a subscripted receiver must go through
            // the PLACE lowering: `xs[0].append(v)` has to mutate the real
            // element, where the Load lowering (py_index) yields a clone
            // and the write silently vanishes.
            let receiver = if matches!(attr.value.as_ref(), ExprType::Subscript(_))
                && crate::ast::tree::scope::MUTATING_METHODS.contains(&attr.attr.as_str())
            {
                crate::ast::tree::subscript::subscript_receiver_place(
                    attr.value.as_ref(),
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?
            } else {
                attr.value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())?
            };

            // String-keyed dicts (from a literal or a `dict[str, V]`
            // annotation) take String keys in py_setdefault/py_pop and
            // &String in py_get/py_get_default/py_contains; literal `"a"`
            // keys are owned at the call site so the generic impls apply.
            let string_keyed_dict = matches!(
                attr.value.as_ref(),
                ExprType::Name(n)
                    if matches!(
                        options.name_types.get(&n.id),
                        Some(crate::TypeInfo::Dict(k, _))
                            if matches!(**k, crate::TypeInfo::String)
                    )
            );

            // list.sort(): in-place, stable, with Python's keyword-only
            // key=/reverse=. Vec's inherent sort demands a total order
            // (rejecting floats), so every shape routes through the
            // PySort variants, which share sorted()'s NaN-loud comparator
            // and run key exactly once per element.
            if attr.attr == "sort" {
                if !self.args.is_empty() {
                    return Err("sort() takes no positional arguments".to_string().into());
                }
                let mut key = None;
                let mut reverse = None;
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("key") if key.is_none() => key = Some(kw.value.clone()),
                        Some("reverse") if reverse.is_none() => {
                            reverse = Some(kw.value.clone())
                        }
                        other => {
                            return Err(format!(
                                "sort() got an unexpected or duplicate keyword \
                                 argument '{}'",
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let render = |e: crate::ExprType| {
                    e.to_rust(ctx.clone(), options.clone(), symbols.clone())
                };
                return Ok(match (key, reverse) {
                    (None, None) => quote!((#receiver).py_sort()),
                    (None, Some(r)) => {
                        let r = render(r)?;
                        quote!((#receiver).py_sort_reverse(#r))
                    }
                    (Some(k), None) => {
                        let k = render(k)?;
                        quote!((#receiver).py_sort_key(#k))
                    }
                    (Some(k), Some(r)) => {
                        let k = render(k)?;
                        let r = render(r)?;
                        quote!((#receiver).py_sort_key_reverse(#k, #r))
                    }
                });
            }

            // replace(...) with datetime-family keywords: one lowering
            // through the PyReplace trait, whose receiver impls
            // (datetime/date/time) each validate their own field set and
            // raise Python's exact TypeError for foreign fields. Only
            // keyword spellings route here — bare/positional replace
            // stays the plain method call (str.replace among others).
            if attr.attr == "replace" && !self.keywords.is_empty() {
                const FIELDS: [&str; 7] = [
                    "year", "month", "day", "hour", "minute", "second", "microsecond",
                ];
                if self
                    .keywords
                    .iter()
                    .all(|kw| kw.arg.as_deref().is_some_and(|a| FIELDS.contains(&a)))
                {
                    let mut slots: [Option<crate::ExprType>; 7] = Default::default();
                    if self.args.len() > FIELDS.len() {
                        return Err("replace() takes at most 7 arguments".to_string().into());
                    }
                    for (i, arg) in self.args.iter().enumerate() {
                        slots[i] = Some(arg.clone());
                    }
                    for kw in &self.keywords {
                        let name = kw.arg.as_deref().expect("checked above");
                        let idx = FIELDS.iter().position(|f| *f == name).expect("checked");
                        if slots[idx].is_some() {
                            return Err(format!(
                                "replace() got multiple values for argument '{}'",
                                name
                            )
                            .into());
                        }
                        slots[idx] = Some(kw.value.clone());
                    }
                    let mut inits = Vec::new();
                    for (idx, slot) in slots.into_iter().enumerate() {
                        if let Some(e) = slot {
                            let field = crate::safe_ident(FIELDS[idx]);
                            let v =
                                e.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                            inits.push(quote!(#field: Some(#v)));
                        }
                    }
                    return Ok(quote!(
                        (#receiver).py_replace(ReplaceArgs {
                            #(#inits,)*
                            ..ReplaceArgs::default()
                        })?
                    ));
                }
                // A keyword outside the field set: Python's message, at
                // conversion time (str.replace takes no keywords here).
                let bad = self
                    .keywords
                    .iter()
                    .find(|kw| {
                        !kw.arg.as_deref().is_some_and(|a| FIELDS.contains(&a))
                    })
                    .and_then(|kw| kw.arg.clone())
                    .unwrap_or_else(|| "**kwargs".to_string());
                return Err(format!(
                    "'{}' is an invalid keyword argument for replace()",
                    bad
                )
                .into());
            }

            // str.split / str.rsplit take sep and maxsplit by position or
            // keyword, with sep=None (or absent) meaning whitespace mode.
            // Normalized here so every spelling maps to the right runtime
            // variant; unknown or duplicate keywords are loud errors, as
            // Python raises TypeError for them.
            if matches!(attr.attr.as_str(), "split" | "rsplit") {
                if self.args.len() > 2 {
                    return Err(format!(
                        "{}() takes at most 2 arguments ({} given)",
                        attr.attr,
                        self.args.len()
                    )
                    .into());
                }
                let mut sep = self.args.first().cloned();
                let mut maxsplit = self.args.get(1).cloned();
                for kw in &self.keywords {
                    match kw.arg.as_deref() {
                        Some("sep") => {
                            if sep.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'sep'",
                                    attr.attr
                                )
                                .into());
                            }
                            sep = Some(kw.value.clone());
                        }
                        Some("maxsplit") => {
                            if maxsplit.is_some() {
                                return Err(format!(
                                    "{}() got multiple values for argument 'maxsplit'",
                                    attr.attr
                                )
                                .into());
                            }
                            maxsplit = Some(kw.value.clone());
                        }
                        other => {
                            return Err(format!(
                                "{}() got an unexpected keyword argument '{}'",
                                attr.attr,
                                other.unwrap_or("**kwargs")
                            )
                            .into());
                        }
                    }
                }
                let is_rsplit = attr.attr == "rsplit";
                let sep = sep.filter(|s| !crate::is_none_expr(s));
                return Ok(match (sep, maxsplit) {
                    (None, None) => quote!((#receiver).py_split_whitespace()),
                    (None, Some(m)) => {
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_whitespace_maxsplit(#m))
                        } else {
                            quote!((#receiver).py_split_whitespace_maxsplit(#m))
                        }
                    }
                    (Some(s), None) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit(&(#s))?)
                        } else {
                            quote!((#receiver).py_split(&(#s))?)
                        }
                    }
                    (Some(s), Some(m)) => {
                        let s = s.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        let m = m.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        if is_rsplit {
                            quote!((#receiver).py_rsplit_maxsplit(&(#s), #m)?)
                        } else {
                            quote!((#receiver).py_split_maxsplit(&(#s), #m)?)
                        }
                    }
                });
            }

            // str.format on a LITERAL template translates to format! at
            // conversion time: auto-numbering, {0} positions, {name}
            // keywords, {{ escaping, and format specs all map — and any
            // spec Rust cannot reproduce exactly is a loud conversion
            // error, never approximated output.
            if attr.attr == "format" {
                let template = match attr.value.as_ref() {
                    ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_))) =>
                    {
                        match &c.0 {
                            Some(litrs::Literal::String(s)) => s.value().to_string(),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        return Err(
                            "str.format on a non-literal template is not supported yet: \
                             the template must be a string literal so the fields can be \
                             checked at conversion time"
                                .to_string()
                                .into(),
                        );
                    }
                };
                return lower_str_format(
                    &template,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                );
            }

            // The remaining builtin methods are positional-only in Python;
            // a keyword here would be silently dropped by the positional
            // pattern match below, so fall through to the generic path,
            // which rejects keywords without a resolvable signature.
            if !self.keywords.is_empty() {
                // fall through
            } else {
            let mut rendered_args = Vec::new();
            for arg in &self.args {
                rendered_args.push(arg.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?);
            }
            match (attr.attr.as_str(), rendered_args.as_slice()) {
                // list.append(x) pushes one element; Vec::append (inherent)
                // concatenates another Vec — silently different.
                ("append", [value]) => {
                    return Ok(quote!((#receiver).push(#value)));
                }
                // list.count(x): the PyListOps method takes a reference.
                ("count", [value]) => {
                    return Ok(quote!((#receiver).count(&(#value))));
                }
                // File-object and csv.Writer methods return Result (I/O
                // can fail; Python raises): thread `?`.
                ("read", []) | ("readline", []) | ("readlines", []) | ("close", [])
                | ("getvalue", []) => {
                    let m = crate::safe_ident(&attr.attr);
                    return Ok(quote!((#receiver).#m()?));
                }
                ("write", [d]) => {
                    return Ok(quote!((#receiver).write(&(#d))?));
                }
                ("writelines", [l]) => {
                    return Ok(quote!((#receiver).writelines(&(#l))?));
                }
                ("writerow", [r]) => {
                    // writerow([]) (an empty record) still needs an
                    // element type for the slice.
                    if r.to_string() == "vec ! []" {
                        return Ok(quote!((#receiver).writerow(&[] as &[&str])?));
                    }
                    return Ok(quote!((#receiver).writerow(&(#r))?));
                }
                ("writerows", [r]) => {
                    return Ok(quote!((#receiver).writerows(&(#r))?));
                }
                // re Match: m.group() is m.group(0); Rust can't overload.
                ("group", []) => {
                    return Ok(quote!((#receiver).group(0)));
                }
                // m.group("name") for (?P<name>...) groups: Rust can't
                // overload on the argument type, so the string spelling
                // routes to group_name. Numeric group(i) falls through to
                // the plain method call.
                ("group", [g]) if g.to_string().starts_with('"') => {
                    return Ok(quote!((#receiver).group_name(#g)));
                }
                // str.encode() / encode("utf-8"): UTF-8 bytes, which is
                // exactly what Rust strings hold.
                ("encode", []) => {
                    return Ok(quote!((#receiver).as_bytes().to_vec()));
                }
                ("encode", [enc]) => {
                    if enc.to_string().trim_matches('"') != "utf-8" {
                        return Err(format!(
                            "str.encode({}): only \"utf-8\" is supported",
                            enc
                        )
                        .into());
                    }
                    return Ok(quote!((#receiver).as_bytes().to_vec()));
                }
                // list.pop() returns the last element or raises IndexError
                // (Vec::pop returns an Option).
                ("pop", []) => {
                    return Ok(quote! {
                        (#receiver).pop().ok_or_else(|| {
                            PyException::new("IndexError", "pop from empty list")
                        })?
                    });
                }
                // pop with an argument dispatches by receiver through the
                // PyPop trait: list.pop(i) by index (IndexError), dict.pop(k)
                // by key (KeyError).
                ("pop", [arg]) => {
                    if string_keyed_dict {
                        let key = crate::render_typed(
                            &self.args[0],
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        return Ok(quote!((#receiver).py_pop(#key)?));
                    }
                    return Ok(quote!((#receiver).py_pop(#arg)?));
                }
                ("pop", [key, default]) => {
                    if string_keyed_dict {
                        let key = crate::render_typed(
                            &self.args[0],
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        return Ok(quote!((#receiver).py_pop_default(#key, #default)));
                    }
                    return Ok(quote!((#receiver).py_pop_default(#key, #default)));
                }
                // dict.get never raises: value-or-None (an Option), or the
                // provided default. IndexMap's inherent get returns a
                // borrowed Option, so both forms map to py_ versions.
                ("get", [key]) => {
                    if string_keyed_dict {
                        let key = crate::render_typed(
                            &self.args[0],
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        return Ok(quote!((#receiver).py_get(&(#key))));
                    }
                    return Ok(quote!((#receiver).py_get(&(#key))));
                }
                ("get", [key, default]) => {
                    if string_keyed_dict {
                        let key = crate::render_typed(
                            &self.args[0],
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        return Ok(quote!((#receiver).py_get_default(&(#key), #default)));
                    }
                    return Ok(quote!((#receiver).py_get_default(&(#key), #default)));
                }
                // Views materialize as Vecs in insertion order.
                ("keys", []) => {
                    return Ok(quote!((#receiver).py_keys()));
                }
                ("values", []) => {
                    return Ok(quote!((#receiver).py_values()));
                }
                ("items", []) => {
                    return Ok(quote!((#receiver).py_items()));
                }
                ("setdefault", [key, default]) => {
                    if string_keyed_dict {
                        let key = crate::render_typed(
                            &self.args[0],
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?;
                        return Ok(quote!((#receiver).py_setdefault(#key, #default)));
                    }
                    return Ok(quote!((#receiver).py_setdefault(#key, #default)));
                }
                // list.remove(x) removes by VALUE and raises ValueError;
                // Vec::remove removes by index — silently different.
                ("remove", [value]) => {
                    // The receiver and argument are each evaluated ONCE,
                    // receiver first (CPython evaluates the primary +
                    // attribute, then the argument). The previous shape
                    // spliced the receiver twice and the argument inside
                    // the position closure, so a side-effecting receiver
                    // (`grid[which()].remove(2)`) ran twice and the
                    // argument once per element scanned (issue #80).
                    return Ok(quote! {
                        {
                            let __rython_recv = &mut (#receiver);
                            let __rython_val = #value;
                            let __rython_pos = __rython_recv
                                .iter()
                                .position(|__rython_e| __rython_e == &__rython_val)
                                .ok_or_else(|| {
                                    PyException::new(
                                        "ValueError",
                                        "list.remove(x): x not in list",
                                    )
                                })?;
                            __rython_recv.remove(__rython_pos);
                        }
                    });
                }
                // list.insert follows Python index rules (negative counts
                // from the end, out-of-range clamps); Vec::insert takes a
                // usize and panics past len.
                ("insert", [idx, value]) => {
                    return Ok(quote!((#receiver).py_insert(#idx, #value)));
                }
                // partition/rpartition raise ValueError on an empty
                // separator, so the calls take `?`.
                ("partition", [sep]) => {
                    return Ok(quote!((#receiver).partition(&(#sep))?));
                }
                ("rpartition", [sep]) => {
                    return Ok(quote!((#receiver).rpartition(&(#sep))?));
                }
                // strip family with a chars argument (the no-arg forms
                // resolve through PyStrOps directly).
                ("strip", [chars]) => {
                    return Ok(quote!((#receiver).py_strip_chars(&(#chars))));
                }
                ("lstrip", [chars]) => {
                    return Ok(quote!((#receiver).py_lstrip_chars(&(#chars))));
                }
                ("rstrip", [chars]) => {
                    return Ok(quote!((#receiver).py_rstrip_chars(&(#chars))));
                }
                // ljust/rjust: the optional fillchar selects the py_ form
                // (space by default).
                ("ljust", [width]) => {
                    return Ok(quote!((#receiver).py_ljust(#width, " ")?));
                }
                ("ljust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_ljust(#width, &(#fill))?));
                }
                ("rjust", [width]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, " ")?));
                }
                ("rjust", [width, fill]) => {
                    return Ok(quote!((#receiver).py_rjust(#width, &(#fill))?));
                }
                // str.find returns -1 when absent; str::find an Option.
                ("find", [needle]) => {
                    return Ok(quote!((#receiver).py_find(&(#needle))));
                }
                _ => {}
            }
            }
        }

        // Keyword arguments and omitted defaulted parameters resolve
        // against the callee's signature: keywords map to their parameter
        // positions and missing parameters fill from their default values,
        // matching Python call semantics. Without a known signature,
        // keywords would silently become misordered positional arguments —
        // that is a loud conversion error instead.
        let callee = match self.func.as_ref() {
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::FunctionDef(f)) => Some(f.clone()),
                _ => None,
            },
            _ => None,
        };
        if let Some(callee_def) = &callee {
            let simple_signature =
                callee_def.args.vararg.is_none() && callee_def.args.kwarg.is_none();
            let pos_param_count =
                callee_def.args.posonlyargs.len() + callee_def.args.args.len();
            let has_optional_params = callee_def
                .args
                .posonlyargs
                .iter()
                .chain(callee_def.args.args.iter())
                .chain(callee_def.args.kwonlyargs.iter())
                .any(|p| {
                    p.annotation
                        .as_deref()
                        .is_some_and(crate::is_optional_annotation)
                });
            let needs_mapping = !self.keywords.is_empty()
                || !callee_def.args.kwonlyargs.is_empty()
                || self.args.len() < pos_param_count
                || has_optional_params;
            if simple_signature && needs_mapping {
                let MappedArguments { prelude, args } = map_call_arguments(
                    callee_def,
                    &self.args,
                    &self.keywords,
                    &ctx,
                    &options,
                    &symbols,
                )?;
                let name = self.func.to_rust(ctx, options, symbols)?;
                let call = quote!({ #prelude #name(#(#args),*) });
                return Ok(if propagates_exceptions {
                    quote!(#call?)
                } else {
                    call
                });
            }
            if !simple_signature && !self.keywords.is_empty() {
                return Err(format!(
                    "keyword arguments in a call to `{}` are not supported yet: \
                     its signature takes *args/**kwargs",
                    callee_def.name
                )
                .into());
            }
        } else if !self.keywords.is_empty() {
            return Err(format!(
                "keyword arguments require the callee's signature, and `{}` is not \
                 a function defined in this module; pass the arguments positionally",
                self.func
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map(|t| t.to_string())
                    .unwrap_or_else(|_| "<callee>".to_string())
            )
            .into());
        }

        let name = self.func.to_rust(ctx.clone(), options.clone(), symbols.clone())?;

        let mut all_args = Vec::new();

        // When the callee is a user function defined in this module, the
        // direct-call path (simple signature, matching arity, no mapping)
        // still needs param-aware lowering: coerce to the parameter's
        // annotated type and clone non-Copy names that are reused later
        // (`f(x); g(x)` — Python shares by reference, Rust moves).
        let pos_params: Vec<&crate::Parameter> = match &callee {
            Some(f) => f
                .args
                .posonlyargs
                .iter()
                .chain(f.args.args.iter())
                .collect(),
            None => Vec::new(),
        };

        // Add positional arguments
        for (i, arg) in self.args.into_iter().enumerate() {
            let rust_arg = if let Some(param) = pos_params.get(i) {
                let expected = param
                    .annotation
                    .as_deref()
                    .and_then(crate::call_arg_expected_type);
                crate::render_typed_reused(
                    &arg,
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                    expected,
                )?
            } else {
                arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?
            };
            all_args.push(rust_arg);
        }
        
        // Add keyword arguments
        for keyword in self.keywords {
            let rust_kw = keyword.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_kw);
        }
        
        // Check if we're in an async context and if the function being called is async
        let call_expr = quote!(#name(#(#all_args),*));
        
        // Check if this function returns a Result that should be unwrapped
        let name_str = format!("{}", name);

        // datetime.strptime parses and validates, so it raises ValueError
        // like Python; propagate rather than hand back a bare Result.
        if name_str.ends_with(":: strptime") {
            return Ok(quote!(#call_expr?));
        }
        let needs_unwrap = matches!(name_str.as_str(), 
            "subprocess :: run" | "subprocess :: run_with_env" | "subprocess :: check_call" | 
            "subprocess :: check_output" | "os :: getcwd" | "os :: chdir" | "os :: execv" |
            "os :: path :: abspath"
        );
        
        // Special handling for subprocess.run and os.execv with fallback for compatibility
        let final_call = if propagates_exceptions {
            quote!(#call_expr?)
        } else if name_str == "subprocess :: run" {
            // Try mixed_args version first, fallback to regular version
            if all_args.len() >= 2 {
                let args_param = &all_args[0];
                let cwd_param = &all_args[1];
                // Convert args to Vec<String> to avoid lifetime issues, then pass owned strings
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    let cwd_str = #cwd_param;
                    subprocess::run(args_vec, Some(&cwd_str)).unwrap()
                })
            } else {
                let args_param = &all_args[0];
                quote!({
                    let args_owned: Vec<String> = #args_param;
                    let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                    subprocess::run(args_vec, None).unwrap()
                })
            }
        } else if name_str == "os :: execv" {
            // Convert to Vec<&str> for compatibility with standard execv function
            let program_param = &all_args[0];
            let args_param = &all_args[1];
            quote!({
                let program_str: String = (#program_param).clone();
                let args_owned: Vec<String> = #args_param;
                let args_vec: Vec<&str> = args_owned.iter().map(|s| s.as_str()).collect();
                os::execv(&program_str, args_vec).unwrap()
            })
        } else if needs_unwrap {
            quote!(#call_expr.unwrap())
        } else {
            call_expr
        };
        
        // `.await` is added only by an explicit `await` expression (the Await
        // node), mirroring Python: calling an async function without await
        // does not implicitly run it. The old behavior appended `.await` to
        // any call whose name started with "a" in async contexts, which broke
        // calls like abs(x).
        Ok(final_call)
    }
}

/// Lower a call to a `rust.bind`/`rust.c_bind` name: a direct call into the
/// bound crate, with type-directed conversions between rython's Python types
/// and the declared Rust signature.
fn lower_rust_binding_call(
    spec: &crate::RustBindSpec,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if !keywords.is_empty() {
        return Err(format!(
            "keyword arguments in a call to bound function `{}::{}` are not \
             supported yet; pass the arguments positionally",
            spec.crate_name, spec.fn_name
        )
        .into());
    }
    if args.len() != spec.args.len() {
        return Err(format!(
            "bound function `{}::{}` takes {} argument(s), but {} were given",
            spec.crate_name,
            spec.fn_name,
            spec.args.len(),
            args.len()
        )
        .into());
    }

    let mut converted = Vec::new();
    for ((_, ty), arg) in spec.args.iter().zip(args.iter()) {
        let rendered = arg.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        converted.push(convert_rust_bind_arg(rendered, ty)?);
    }

    let crate_ident = format_ident!("{}", spec.crate_name.replace('-', "_"));
    let fn_ident = format_ident!("{}", spec.fn_name);
    let call = quote!(#crate_ident::#fn_ident(#(#converted),*));
    let call = if spec.c_abi {
        quote!(unsafe { #call })
    } else {
        call
    };
    convert_rust_bind_ret(call, spec.returns.as_deref())
}

/// Convert a Python-side argument to the declared Rust parameter type.
/// rython's Python types are fixed: int is i64, float is f64, str is String,
/// bytes is Vec<u8>, bool is bool — so every conversion is deterministic.
fn convert_rust_bind_arg(
    tokens: TokenStream,
    ty: &str,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let converted = match ty {
        "i64" | "i32" | "u64" | "u32" | "i16" | "u16" | "i8" | "u8" | "isize" | "usize" => {
            let t: TokenStream = ty.parse().expect("validated rust type");
            quote!(#tokens as #t)
        }
        "f64" => tokens,
        "f32" => quote!(#tokens as f32),
        "bool" => tokens,
        // AsRef bridges both spellings: String and &'static str (literals and
        // module constants) both produce &str.
        "&str" => quote!(#tokens.as_ref()),
        // From is an identity impl for String and a conversion for &str.
        "String" => quote!(String::from(#tokens)),
        "&[u8]" => quote!(#tokens.as_ref()),
        // From is identity for Vec<u8> and converts from byte literals.
        "Vec<u8>" => quote!(Vec::from(#tokens)),
        "*const u8" => quote!(#tokens.as_ptr()),
        "*mut u8" => quote!(#tokens.as_mut_ptr()),
        other => {
            return Err(format!("rust.bind: unsupported parameter type `{}`", other).into())
        }
    };
    Ok(converted)
}

/// Convert a bound call's result to the Python-side type. `()`/`void`/absent
/// returns stay bare so statement-position calls keep a plain `fn();` shape.
fn convert_rust_bind_ret(
    call: TokenStream,
    ret: Option<&str>,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let converted = match ret {
        None | Some("()") | Some("void") => call,
        Some("i64") | Some("i32") | Some("u64") | Some("u32") | Some("i16") | Some("u16")
        | Some("i8") | Some("u8") | Some("isize") | Some("usize") => {
            quote!(#call as i64)
        }
        Some("f64") => call,
        Some("f32") => quote!(#call as f64),
        Some("bool") => call,
        Some("String") => call,
        Some("&str") => quote!(#call.to_string()),
        Some("Vec<u8>") => call,
        Some(other) => {
            return Err(format!("rust.bind: unsupported return type `{}`", other).into())
        }
    };
    Ok(converted)
}

/// Lower a literal `template.format(args...)` call to a Rust `format!`.
///
/// Every argument (used or not) is evaluated exactly once, in Python's
/// order, into a local binding — Python evaluates unused arguments too.
/// Used bindings are referenced from the format string by name; unused
/// ones bind to `_` so no warning fires. Errors mirror Python's:
/// mixing auto and manual numbering, out-of-range indices, and missing
/// keywords are conversion-time failures.
fn lower_str_format(
    template: &str,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    use crate::pyformat::{parse_template, translate_format_spec, FieldRef, Piece};

    let pieces = parse_template(template).map_err(|e| format!("str.format: {}", e))?;

    for kw in keywords {
        if kw.arg.is_none() {
            return Err("str.format with **kwargs is not supported yet".to_string().into());
        }
    }

    // Resolve each field to an argument slot and build the format string.
    let mut fmt = String::new();
    let mut used_positions: std::collections::HashSet<usize> = Default::default();
    let mut used_names: std::collections::HashSet<String> = Default::default();
    let mut auto_next = 0usize;
    let mut saw_auto = false;
    let mut saw_manual = false;
    let mut field_bindings: Vec<TokenStream> = Vec::new();
    for piece in &pieces {
        match piece {
            Piece::Literal(text) => {
                fmt.push_str(&text.replace('{', "{{").replace('}', "}}"));
            }
            Piece::Field { arg, conversion, spec } => {
                let index_name = match arg {
                    FieldRef::Auto => {
                        saw_auto = true;
                        let i = auto_next;
                        auto_next += 1;
                        if i >= args.len() {
                            return Err(format!(
                                "str.format: not enough positional arguments (field {} of \
                                 template {:?})",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Index(i) => {
                        saw_manual = true;
                        if *i >= args.len() {
                            return Err(format!(
                                "str.format: replacement index {} out of range for \
                                 template {:?}",
                                i, template
                            )
                            .into());
                        }
                        used_positions.insert(*i);
                        format!("__rython_fmt{}", i)
                    }
                    FieldRef::Name(name) => {
                        if !keywords
                            .iter()
                            .any(|k| k.arg.as_deref() == Some(name.as_str()))
                        {
                            return Err(format!(
                                "str.format: template {:?} refers to {:?}, which is not \
                                 among the keyword arguments",
                                template, name
                            )
                            .into());
                        }
                        used_names.insert(name.clone());
                        format!("__rython_fmt_{}", name)
                    }
                };
                if saw_auto && saw_manual {
                    return Err(
                        "str.format: cannot switch between automatic field numbering \
                         and manual field specification"
                            .to_string()
                            .into(),
                    );
                }
                let is_repr = matches!(conversion, Some('r') | Some('a'));
                let lowering =
                    translate_format_spec(spec).map_err(|e| format!("str.format: {}", e))?;
                if is_repr {
                    // Python's !r renders the repr STRING and applies the
                    // spec to it; Rust's `{:?}` would print its own Debug
                    // form ("ab" with double quotes) and diverge.
                    let crate::pyformat::SpecLowering::Inline(suffix) = lowering else {
                        return Err("str.format: numeric presentation types cannot combine \
                                    with !r/!a (Python applies the spec to the repr string \
                                    and raises)"
                            .to_string()
                            .into());
                    };
                    let fld = format!("__rython_fld{}", field_bindings.len());
                    let src = crate::safe_ident(&index_name);
                    let ident = crate::safe_ident(&fld);
                    field_bindings.push(quote!(let #ident = repr(&(#src));));
                    if suffix.is_empty() {
                        fmt.push_str(&format!("{{{}}}", fld));
                    } else {
                        fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                    }
                    continue;
                }
                match lowering {
                    crate::pyformat::SpecLowering::Inline(suffix) => {
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", index_name));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", index_name, suffix));
                        }
                    }
                    // The operand coerces or converts per-field (one
                    // argument may be reused with different specs), via a
                    // field-local binding referencing the argument's.
                    crate::pyformat::SpecLowering::CastF64(suffix) => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(let #ident = (#src) as f64;));
                        if suffix.is_empty() {
                            fmt.push_str(&format!("{{{}}}", fld));
                        } else {
                            fmt.push_str(&format!("{{{}:{}}}", fld, suffix));
                        }
                    }
                    crate::pyformat::SpecLowering::IntRadix {
                        fill,
                        align,
                        plus,
                        alternate,
                        zero,
                        width,
                        radix,
                    } => {
                        let fld = format!("__rython_fld{}", field_bindings.len());
                        let src = crate::safe_ident(&index_name);
                        let ident = crate::safe_ident(&fld);
                        field_bindings.push(quote!(
                            let #ident = py_int_radix_format(
                                #src, #fill, #align, #plus, #alternate, #zero, #width, #radix,
                            );
                        ));
                        fmt.push_str(&format!("{{{}}}", fld));
                    }
                }
            }
        }
    }

    // Bindings: every argument evaluates exactly once, in order.
    let mut bindings = TokenStream::new();
    for (i, arg) in args.iter().enumerate() {
        let value = arg.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_positions.contains(&i) {
            let ident = crate::safe_ident(&format!("__rython_fmt{}", i));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }
    for kw in keywords {
        let name = kw.arg.as_deref().unwrap_or_default();
        let value = kw
            .value
            .clone()
            .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
        if used_names.contains(name) {
            let ident = crate::safe_ident(&format!("__rython_fmt_{}", name));
            bindings.extend(quote!(let #ident = #value;));
        } else {
            bindings.extend(quote!(let _ = #value;));
        }
    }

    for fb in field_bindings {
        bindings.extend(fb);
    }

    Ok(quote!({
        #bindings
        format!(#fmt)
    }))
}

/// The class of a method-call receiver, when it is statically known:
/// `self` inside a class's method body, or a local/module name whose
/// (symbol-table-recorded) assignment constructs a known class. Unknown
/// receivers return None and fall through to the generic lowering — where
/// a genuine user-method call fails to compile (loud), never silently
/// drops exception propagation.
pub(crate) fn receiver_class(
    recv: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
) -> Option<crate::ClassDef> {
    let class_name = match recv {
        ExprType::Name(n) if n.id == "self" => ctx.enclosing_class_name()?.to_string(),
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::Assign { value: ExprType::Call(call), .. }) => {
                match call.func.as_ref() {
                    ExprType::Name(cn) => cn.id.clone(),
                    _ => return None,
                }
            }
            _ => return None,
        },
        // Composition: `self.field.method()` resolves through the owner
        // class's field types.
        ExprType::Attribute(attr) => {
            let owner = receiver_class(&attr.value, ctx, symbols)?;
            owner.field_class(&attr.attr, symbols)?
        }
        _ => return None,
    };
    match symbols.get(&class_name) {
        Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
        _ => None,
    }
}

/// Resolve a call's arguments against the callee's signature, in Python's
/// order: positionals fill left to right, keywords map by name, missing
/// parameters take their default values, and every mismatch Python would
/// raise a TypeError for is a conversion-time error.
///
/// Returned arguments are in parameter order, as Rust needs; when keyword
/// arguments make that differ from Python's source evaluation order, the
/// prelude binds each argument to a temp in source order first (CPython
/// evaluates positionals left to right, then keywords left to right).
struct MappedArguments {
    prelude: TokenStream,
    args: Vec<TokenStream>,
}

/// Defaults are evaluated ONCE at def time by CPython; rython inlines them
/// at each call site. A scalar constant (literal, None/True/False, or a
/// tuple of those) is safe to inline; anything else is a loud conversion
/// error rather than a silent divergence — a mutable container would also
/// be SHARED across calls by CPython, which owned Rust values cannot
/// express (issue #80).
fn check_default_constant(
    default: &ExprType,
    fname: &str,
    param: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let scalar = |e: &ExprType| -> bool {
        matches!(e, ExprType::Constant(_))
            || matches!(
                e,
                ExprType::Name(n) if matches!(n.id.as_str(), "None" | "True" | "False")
            )
    };
    match default {
        ExprType::Constant(_) => Ok(()),
        ExprType::Name(n) if matches!(n.id.as_str(), "None" | "True" | "False") => Ok(()),
        ExprType::Tuple(t) if t.elts.iter().all(&scalar) => Ok(()),
        ExprType::Tuple(t)
            if t.elts
                .iter()
                .any(|e| matches!(e, ExprType::List(_) | ExprType::Dict(_) | ExprType::Set(_))) =>
        {
            Err(format!(
                "parameter `{param}` of `{fname}()` has a mutable default; CPython \
                 evaluates defaults once at def time and SHARES the single container \
                 across all calls, which rython's owned-value model cannot express. \
                 Pass the container explicitly at every call site instead"
            )
            .into())
        }
        ExprType::List(_) | ExprType::Dict(_) | ExprType::Set(_) => Err(format!(
            "parameter `{param}` of `{fname}()` has a mutable default; CPython \
             evaluates defaults once at def time and SHARES the single container \
             across all calls, which rython's owned-value model cannot express. \
             Pass the container explicitly at every call site instead"
        )
        .into()),
        _ => Err(format!(
            "parameter `{param}` of `{fname}()` has a non-constant default; CPython \
             evaluates defaults once at def time, but rython would re-evaluate this \
             expression at every call site. Use a constant default instead (issue #80)"
        )
        .into()),
    }
}

/// Whether a name is a user-defined symbol (assignment, function, class,
/// or rust.bind) rather than an imported stdlib module. Module
/// intercepts (`re.search`, `csv.reader`, ...) and module-path lowering
/// (`sys.argv`, `np.dot`) must defer to user definitions: `re = my_thing;
/// re.search(...)` calls the user's object, not the re module (issue #80).
/// Imports (plain or from-import) are module-like by construction — and so
/// is an aliased import (`import numpy as np` binds `np` as a module
/// alias, not a user value; a later `np = ...` reassignment replaces the
/// alias in the symbol table, which still shadows correctly — Devin review
/// on #103).
pub(crate) fn module_name_shadowed(name: &str, symbols: &SymbolTableScopes) -> bool {
    matches!(
        symbols.get(name),
        Some(SymbolTableNode::Assign { .. })
            | Some(SymbolTableNode::FunctionDef(_))
            | Some(SymbolTableNode::ClassDef(_))
            | Some(SymbolTableNode::RustBinding(_))
    )
}

/// The name at the root of a dotted expression chain (`os` in `os.path`,
/// `np` in `np.linalg.inv`), for module-vs-value resolution.
pub(crate) fn root_name(expr: &ExprType) -> Option<&str> {
    match expr {
        ExprType::Name(n) => Some(&n.id),
        ExprType::Attribute(a) => root_name(&a.value),
        _ => None,
    }
}

fn map_call_arguments(
    func: &crate::FunctionDef,
    args: &[ExprType],
    keywords: &[Keyword],
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Result<MappedArguments, Box<dyn std::error::Error>> {
    let fname = &func.name;
    // Optional-annotated parameters take Option values: the Option-slot
    // lowering wraps plain arguments in Some, passes None and
    // already-Option values (dict.get, another optional name, an
    // Optional-returning call) through unwrapped, and handles conditional
    // arms independently.
    let fill = |param: &crate::Parameter,
                expr: &ExprType|
     -> Result<TokenStream, Box<dyn std::error::Error>> {
        let optional = param
            .annotation
            .as_deref()
            .is_some_and(crate::is_optional_annotation);
        if optional {
            crate::lower_optional_value(expr, ctx.clone(), options.clone(), symbols.clone())
        } else {
            // Type-aware lowering: coerce the argument to the parameter's
            // annotated type (usize → i64, i64 → f64) and clone non-Copy
            // names that are reused later (Python shares by reference;
            // Rust moves).
            let expected = param
                .annotation
                .as_deref()
                .and_then(crate::call_arg_expected_type);
            crate::render_typed_reused(
                expr,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
                expected,
            )
        }
    };

    let pos_params: Vec<&crate::Parameter> = func
        .args
        .posonlyargs
        .iter()
        .chain(func.args.args.iter())
        .collect();
    let n = pos_params.len();
    if args.len() > n {
        return Err(format!(
            "{}() takes {} positional argument(s) but {} were given",
            fname,
            n,
            args.len()
        )
        .into());
    }

    // Rendered values in Python's source evaluation order (positionals,
    // then keywords), for the temp prelude when reordering is needed.
    let mut eval_order: Vec<TokenStream> = Vec::new();
    let total = n + func.args.kwonlyargs.len();
    // Which eval_order index each parameter slot was filled from; None
    // means the slot holds an inlined (constant) default.
    let mut slot_temp: Vec<Option<usize>> = vec![None; total];

    let mut slots: Vec<Option<TokenStream>> = vec![None; n];
    for (i, arg) in args.iter().enumerate() {
        let value = fill(pos_params[i], arg)?;
        eval_order.push(value.clone());
        slot_temp[i] = Some(eval_order.len() - 1);
        slots[i] = Some(value);
    }

    let mut kwonly_slots: Vec<Option<TokenStream>> = vec![None; func.args.kwonlyargs.len()];
    for kw in keywords {
        let Some(kw_name) = &kw.arg else {
            return Err(format!(
                "**kwargs unpacking in a call to {}() is not supported",
                fname
            )
            .into());
        };
        if let Some(idx) = pos_params.iter().position(|p| &p.arg == kw_name) {
            let value = fill(pos_params[idx], &kw.value)?;
            if idx < func.args.posonlyargs.len() {
                return Err(format!(
                    "{}(): parameter `{}` is positional-only and cannot be passed by keyword",
                    fname, kw_name
                )
                .into());
            }
            if slots[idx].is_some() {
                return Err(
                    format!("{}() got multiple values for argument `{}`", fname, kw_name).into(),
                );
            }
            eval_order.push(value.clone());
            slot_temp[idx] = Some(eval_order.len() - 1);
            slots[idx] = Some(value);
        } else if let Some(idx) = func.args.kwonlyargs.iter().position(|p| &p.arg == kw_name) {
            let value = fill(&func.args.kwonlyargs[idx], &kw.value)?;
            if kwonly_slots[idx].is_some() {
                return Err(
                    format!("{}() got multiple values for argument `{}`", fname, kw_name).into(),
                );
            }
            eval_order.push(value.clone());
            slot_temp[n + idx] = Some(eval_order.len() - 1);
            kwonly_slots[idx] = Some(value);
        } else {
            return Err(format!(
                "{}() got an unexpected keyword argument `{}`",
                fname, kw_name
            )
            .into());
        }
    }

    // Defaults align with the tail of the positional parameter list.
    let default_offset = n - func.args.defaults.len();
    for i in 0..n {
        if slots[i].is_none() {
            if i >= default_offset {
                let default = &func.args.defaults[i - default_offset];
                check_default_constant(default, fname, &pos_params[i].arg)?;
                slots[i] = Some(fill(pos_params[i], default)?);
            } else {
                return Err(format!(
                    "{}() missing required argument `{}`",
                    fname, pos_params[i].arg
                )
                .into());
            }
        }
    }
    for (i, param) in func.args.kwonlyargs.iter().enumerate() {
        if kwonly_slots[i].is_none() {
            match func.args.kw_defaults.get(i).and_then(|d| d.as_ref()) {
                Some(default) => {
                    check_default_constant(default, fname, &param.arg)?;
                    kwonly_slots[i] = Some(fill(param, default)?);
                }
                None => {
                    return Err(format!(
                        "{}() missing required keyword-only argument `{}`",
                        fname, param.arg
                    )
                    .into());
                }
            }
        }
    }

    let mut final_slots: Vec<TokenStream> = Vec::with_capacity(total);
    for s in slots.into_iter().chain(kwonly_slots) {
        final_slots.push(s.expect("all argument slots filled"));
    }

    if keywords.is_empty() {
        // No reordering: the parameter-ordered emission is already the
        // source order, so emit the arguments directly (no prelude).
        return Ok(MappedArguments {
            prelude: TokenStream::new(),
            args: final_slots,
        });
    }

    // Keywords reorder the emission: bind every argument to a temp in
    // source order, then reference the temps in parameter order.
    let mut prelude = TokenStream::new();
    for (i, value) in eval_order.iter().enumerate() {
        let tid = format_ident!("__rython_arg_{}", i);
        prelude.extend(quote!(let #tid = #value;));
    }
    let args = (0..total)
        .map(|i| match slot_temp[i] {
            Some(ti) => {
                let tid = format_ident!("__rython_arg_{}", ti);
                quote!(#tid)
            }
            // A default fills the slot: it is a constant (verified by
            // `check_default_constant`), so inlining it in parameter
            // position cannot reorder or duplicate side effects.
            None => final_slots[i].clone(),
        })
        .collect();
    Ok(MappedArguments { prelude, args })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_of_function() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(a=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let code = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                symbols,
            )
            .unwrap()
            .to_string();
        assert!(
            code.contains("let __rython_arg_0 = 9 ; foo (__rython_arg_0)"),
            "generated: {}",
            code
        );
    }

    #[test]
    fn unknown_keyword_argument_is_a_conversion_error() {
        // Python raises TypeError for foo(b=9) when foo has no parameter b;
        // silently passing it positionally would be wrong.
        let options = PythonOptions::default();
        let result = crate::parse(
            "def foo(a = 7):
    pass

foo(b=9)",
            "test.py",
        )
        .unwrap();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let err = result
            .to_rust(
                CodeGenContext::Module("test".to_string()),
                options,
                symbols,
            )
            .expect_err("unexpected keyword must not convert");
        assert!(
            format!("{}", err).contains("unexpected keyword"),
            "error: {}",
            err
        );
    }
}
