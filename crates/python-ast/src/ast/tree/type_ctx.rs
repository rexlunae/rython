//! Type-aware expression lowering: knows the type context of each
//! expression and inserts the manual conversions Rust needs.
//!
//! Python is dynamically typed; the generated Rust is statically typed, so
//! identical-looking expressions can produce values of different Rust types
//! depending on how they were computed (`"a"` is `&'static str` while
//! `f() -> str` is `String`; `len(x)` is `usize` while `range()` wants
//! `i64`). Rather than changing what literals and calls produce (which
//! makes every generated value more expensive), this module tracks a small
//! lattice of the types codegen actually emits, infers the type of an
//! expression bottom-up, and — when the context demands a different type —
//! wraps the rendered tokens in the minimal conversion:
//!
//! - `&str` → `String` via `.to_string()`
//! - `String` → `&str` via `.as_str()`
//! - `usize` → `i64` via `.try_into().unwrap()` (loud on overflow)
//! - `i64` → `f64` via `as f64` (only in all-numeric unification)
//!
//! The per-function analysis in [`analyze_function_types`] additionally
//! tracks how often each name is read, so move-prone positions (call
//! arguments, container elements) can clone a value that is reused later —
//! Rust's move semantics otherwise consume the variable on first use.

use proc_macro2::TokenStream;
use quote::quote;
use std::collections::{HashMap, HashSet};

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, Statement, StatementType, SymbolTableNode,
    SymbolTableScopes,
};

/// The Rust types codegen produces for Python expressions, at the
/// granularity needed to insert coercions.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeInfo {
    /// `i64`
    Int,
    /// `f64`
    Float,
    /// `bool`
    Bool,
    /// `&str` — string literals lower to `&'static str`
    StrRef,
    /// `String` — computed strings (calls, f-strings, concatenation)
    String,
    /// `Vec<u8>`
    Bytes,
    /// `Vec<T>`
    Vec(Box<TypeInfo>),
    /// `PyDict<K, V>`
    Dict(Box<TypeInfo>, Box<TypeInfo>),
    /// a Rust tuple
    Tuple(Vec<TypeInfo>),
    /// `Option<T>`
    Option(Box<TypeInfo>),
    /// `PyRange`
    Range,
    /// `numpy::NdArray`
    /// `numpy::NdArray`
    NdArray,
    /// `stdpython::StrOrBytes` — the `str | bytes` heterogeneous union
    /// (issue #121): a value that is either a String or Vec<u8>, narrowed
    /// by isinstance checks.
    StrOrBytes,
    /// `stdpython::PyValue` — the boxed heterogeneous value (issue #121):
    /// any wider union (`bool | str | None`, `tuple[...] | str | None`,
    /// `int | str | None`, ...) or `Any`. isinstance checks dispatch at
    /// runtime and narrow the branch to a concrete member.
    PyValue,
    /// A PyValue narrowed by isinstance to one of its member types. Only
    /// ever appears as a narrowed_names target: reads convert via the
    /// PyValue accessors (`as_int().unwrap()`, `as_str().unwrap()`, ...).
    PyValueMember(Box<TypeInfo>),
    /// an instance of a class defined in this module (by class name); not
    /// Copy, so reused values must be cloned at each move-prone use
    Class(String),
    /// a borrow (`&[T]`, `&str`) from iteration or indexing
    Borrowed(Box<TypeInfo>),
    PyObject,
}

impl TypeInfo {
    /// Whether a value of this type can be copied implicitly by Rust.
    pub fn is_copy(&self) -> bool {
        match self {
            TypeInfo::Int | TypeInfo::Float | TypeInfo::Bool => true,
            TypeInfo::Tuple(ts) => ts.iter().all(|t| t.is_copy()),
            TypeInfo::Option(inner) => inner.is_copy(),
            _ => false,
        }
    }

    /// Render the Rust type name, for typed empty containers
    /// (`Vec::<f64>::new()`, `PyDict::<String, i64>::from([])`).
    pub fn to_rust_type(&self) -> TokenStream {
        match self {
            TypeInfo::Int => quote!(i64),
            TypeInfo::Float => quote!(f64),
            TypeInfo::Bool => quote!(bool),
            TypeInfo::StrRef => quote!(&'static str),
            TypeInfo::String => quote!(String),
            TypeInfo::Bytes => quote!(Vec<u8>),
            TypeInfo::Vec(inner) => {
                let t = inner.to_rust_type();
                quote!(Vec<#t>)
            }
            TypeInfo::Dict(k, v) => {
                let k = k.to_rust_type();
                let v = v.to_rust_type();
                quote!(PyDict<#k, #v>)
            }
            TypeInfo::Tuple(ts) => {
                let ts = ts.iter().map(|t| t.to_rust_type());
                quote!((#(#ts),*))
            }
            TypeInfo::Option(inner) => {
                let t = inner.to_rust_type();
                quote!(Option<#t>)
            }
            TypeInfo::Range => quote!(PyRange),
            TypeInfo::NdArray => quote!(numpy::NdArray),
            TypeInfo::StrOrBytes => quote!(stdpython::StrOrBytes),
            TypeInfo::PyValue => quote!(stdpython::PyValue),
            // A narrowed PyValue member still holds the boxed value at
            // runtime; only reads convert.
            TypeInfo::PyValueMember(_) => quote!(stdpython::PyValue),
            TypeInfo::Class(name) => {
                let ident = crate::safe_ident(name);
                quote!(#ident)
            }
            TypeInfo::Borrowed(inner) => {
                let t = inner.to_rust_type();
                quote!(&#t)
            }
            TypeInfo::PyObject => quote!(_),
        }
    }

    /// Human-readable name for diagnostics.
    pub fn display(&self) -> String {
        match self {
            TypeInfo::Int => "int".into(),
            TypeInfo::Float => "float".into(),
            TypeInfo::Bool => "bool".into(),
            TypeInfo::StrRef | TypeInfo::String => "str".into(),
            TypeInfo::Bytes => "bytes".into(),
            TypeInfo::Vec(_) => "list".into(),
            TypeInfo::Dict(_, _) => "dict".into(),
            TypeInfo::Tuple(_) => "tuple".into(),
            TypeInfo::Option(_) => "Optional".into(),
            TypeInfo::Range => "range".into(),
            TypeInfo::NdArray => "ndarray".into(),
            TypeInfo::StrOrBytes => "str | bytes".into(),
            TypeInfo::PyValue => "any".into(),
            TypeInfo::PyValueMember(_) => "any member".into(),
            TypeInfo::Class(name) => name.clone(),
            TypeInfo::Borrowed(_) => "borrowed".into(),
            TypeInfo::PyObject => "unknown".into(),
        }
    }
}

/// Wrap already-rendered tokens in the conversion that takes a value of
/// `from` to a value of `to`. Returns `None` when no conversion is needed
/// or possible.
pub fn coerce_tokens(tokens: TokenStream, from: &TypeInfo, to: &TypeInfo) -> Option<TokenStream> {
    if from == to {
        return Some(tokens);
    }
    match (from, to) {
        // &str → String: string literals in String-typed contexts.
        (TypeInfo::StrRef, TypeInfo::String) => Some(quote!((#tokens).to_string())),
        // String → &str: computed keys/args into &str-typed containers.
        // The String temporary lives until the end of the enclosing
        // statement, so borrowing it in an argument/index position is safe.
        (TypeInfo::String, TypeInfo::StrRef) => Some(quote!((#tokens).as_str())),
        // i64 → f64: only in all-numeric unification (mixed int/float
        // literal lists). Python's int is arbitrary precision, so this is
        // lossy above 2^53; accepted for numeric lists because it is the
        // only way to compile them at all, and the alternative (rustc
        // error) is no more informative.
        (TypeInfo::Int, TypeInfo::Float) => Some(quote!((#tokens) as f64)),
        // Anything → PyValue (issue #121): a value stored into a boxed
        // union / Any slot wraps in PyValue::from (None via From<()>).
        (_, TypeInfo::PyValue) => Some(quote!(PyValue::from((#tokens)))),
        // Anything → StrOrBytes (issue #121): the str | bytes union's
        // heterogeneous slot converts via its From impls (&str, String,
        // &[u8], Vec<u8>).
        (_, TypeInfo::StrOrBytes) => Some(quote!(stdpython::StrOrBytes::from((#tokens)))),
        _ => None,
    }
}

/// Infer the Rust type an expression will produce, bottom-up, from syntax
/// plus the per-function annotation/assignment maps.
pub fn infer_type(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TypeInfo {
    // Cycle guard: a name whose recorded assignment references itself
    // (`label_bytes = label_bytes[lo:]`) would recurse forever through
    // Name → Assign value → Subscript → value Name → ... (idna/core.py).
    thread_local! {
        static INFER_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let d = INFER_DEPTH.with(|c| c.get());
    if d > 64 {
        return TypeInfo::PyObject;
    }
    INFER_DEPTH.with(|c| c.set(d + 1));
    let result = infer_type_inner(expr, options, symbols);
    INFER_DEPTH.with(|c| c.set(d));
    return result;
}

fn infer_type_inner(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TypeInfo {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => TypeInfo::Int,
            Some(litrs::Literal::Float(_)) => TypeInfo::Float,
            Some(litrs::Literal::Bool(_)) => TypeInfo::Bool,
            Some(litrs::Literal::String(_)) => TypeInfo::StrRef,
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => TypeInfo::Bytes,
            None => TypeInfo::PyObject, // None literal
            _ => TypeInfo::PyObject,
        },
        ExprType::Name(n) => {
            // 1. The per-function annotation map (params + literal assigns).
            if let Some(py) = options.local_types.get(&n.id) {
                return py_type(py);
            }
            // 2. Our richer per-function inference map.
            if let Some(t) = options.name_types.get(&n.id) {
                return t.clone();
            }
            // 3. The symbol table's recorded assignment.
            if let Some(SymbolTableNode::Assign { value, .. }) = symbols.get(&n.id) {
                return infer_type(value, options, symbols);
            }
            TypeInfo::PyObject
        }
        ExprType::List(l) => {
            let mut elt = TypeInfo::PyObject;
            for e in l {
                let t = infer_type(e, options, symbols);
                if !matches!(t, TypeInfo::PyObject) {
                    elt = unify(elt, t);
                }
            }
            TypeInfo::Vec(Box::new(elt))
        }
        ExprType::Dict(d) => {
            let mut k = TypeInfo::PyObject;
            let mut v = TypeInfo::PyObject;
            for (key, value) in d.keys.iter().zip(d.values.iter()) {
                let kt = match key {
                    Some(key) => infer_type(key, options, symbols),
                    None => TypeInfo::PyObject, // `**d` unpacking
                };
                let vt = infer_type(value, options, symbols);
                if !matches!(kt, TypeInfo::PyObject) {
                    k = unify(k, kt);
                }
                if !matches!(vt, TypeInfo::PyObject) {
                    v = unify(v, vt);
                }
            }
            TypeInfo::Dict(Box::new(k), Box::new(v))
        }
        ExprType::Tuple(t) => TypeInfo::Tuple(
            t.elts
                .iter()
                .map(|e| infer_type(e, options, symbols))
                .collect(),
        ),
        ExprType::JoinedStr(_) => TypeInfo::String,
        ExprType::FormattedValue(_) => TypeInfo::String,
        ExprType::BinOp(op) => {
            let l = infer_type(&op.left, options, symbols);
            let r = infer_type(&op.right, options, symbols);
            match op.op {
                crate::BinOps::Add => {
                    if is_stringy(&l) && is_stringy(&r) {
                        TypeInfo::String
                    } else if is_numeric(&l) && is_numeric(&r) {
                        numeric_join(&l, &r)
                    } else {
                        TypeInfo::PyObject
                    }
                }
                crate::BinOps::Div => TypeInfo::Float, // Python true division
                crate::BinOps::Sub | crate::BinOps::Mult | crate::BinOps::Pow => {
                    if is_numeric(&l) && is_numeric(&r) {
                        numeric_join(&l, &r)
                    } else {
                        TypeInfo::PyObject
                    }
                }
                crate::BinOps::MatMult => TypeInfo::NdArray,
                _ => {
                    if is_numeric(&l) && is_numeric(&r) {
                        numeric_join(&l, &r)
                    } else {
                        TypeInfo::PyObject
                    }
                }
            }
        }
        ExprType::UnaryOp(u) => match u.op {
            crate::Ops::Not => TypeInfo::Bool,
            crate::Ops::USub | crate::Ops::UAdd => infer_type(&u.operand, options, symbols),
            _ => TypeInfo::PyObject,
        },
        ExprType::BoolOp(_) => TypeInfo::Bool,
        ExprType::Compare(_) => TypeInfo::Bool,
        ExprType::IfExp(i) => {
            // The branches must agree for a useful inference.
            let a = infer_type(&i.body, options, symbols);
            let b = infer_type(&i.orelse, options, symbols);
            if a == b {
                a
            } else if is_numeric(&a) && is_numeric(&b) {
                TypeInfo::Float
            } else if is_stringy(&a) && is_stringy(&b) {
                TypeInfo::String
            } else {
                TypeInfo::PyObject
            }
        }
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) => match builtin_call_type(&n.id) {
                Some(t) => t,
                None => match symbols.get(&n.id) {
                    // A class-construction call produces an instance of the
                    // class (not Copy: reused instances must be cloned at
                    // each move-prone use, matching Python's aliasing).
                    Some(crate::SymbolTableNode::ClassDef(_)) => TypeInfo::Class(n.id.clone()),
                    _ => TypeInfo::PyObject,
                },
            },
            ExprType::Attribute(attr) => {
                // numpy functions produce arrays (`np.sum`, `numpy.mean`).
                let on_numpy = matches!(
                    attr.value.as_ref(),
                    ExprType::Name(n) if crate::is_numpy_alias(&n.id)
                );
                match attr.attr.as_str() {
                    "get" | "pop" | "setdefault" => TypeInfo::PyObject,
                    _ if on_numpy => TypeInfo::NdArray,
                    _ => TypeInfo::PyObject,
                }
            }
            _ => TypeInfo::PyObject,
        },
        ExprType::Subscript(sub) => match infer_type(&sub.value, options, symbols) {
            TypeInfo::Vec(inner) => *inner,
            TypeInfo::Dict(_, v) => *v,
            TypeInfo::Borrowed(inner) => match *inner {
                TypeInfo::Vec(e) => *e,
                TypeInfo::String => TypeInfo::StrRef,
                other => other,
            },
            _ => TypeInfo::PyObject,
        },
        ExprType::ListComp(_) => TypeInfo::Vec(Box::new(TypeInfo::PyObject)),
        ExprType::DictComp(_) => {
            TypeInfo::Dict(Box::new(TypeInfo::PyObject), Box::new(TypeInfo::PyObject))
        }
        ExprType::Starred(s) => infer_type(&s.value, options, symbols),
        _ => TypeInfo::PyObject,
    }
}

/// The return TypeInfo of a builtin call, when statically known. ONE map
/// for the two inference paths (infer_type and syntactic_type) that
/// previously kept byte-identical copies: len()/count() must agree with
/// the `as i64` codegen emission everywhere, or an empty container pins
/// to different element types on different paths.
fn builtin_call_type(name: &str) -> Option<TypeInfo> {
    Some(match name {
        // len()/count() lower to `len(&x) as i64` — Python ints are i64
        // everywhere (range(), indexing, arithmetic), so the inferred
        // type must be Int, not the runtime's usize.
        "len" | "count" => TypeInfo::Int,
        "range" => TypeInfo::Range,
        "str" | "repr" | "format" => TypeInfo::String,
        "int" => TypeInfo::Int,
        "float" => TypeInfo::Float,
        "bool" => TypeInfo::Bool,
        "list" => TypeInfo::Vec(Box::new(TypeInfo::PyObject)),
        "dict" => TypeInfo::Dict(Box::new(TypeInfo::PyObject), Box::new(TypeInfo::PyObject)),
        _ => return None,
    })
}

fn py_type(py: &str) -> TypeInfo {
    match py {
        "int" => TypeInfo::Int,
        "float" => TypeInfo::Float,
        "bool" => TypeInfo::Bool,
        "str" => TypeInfo::String,
        "bytes" => TypeInfo::Bytes,
        _ => TypeInfo::PyObject,
    }
}

fn is_numeric(t: &TypeInfo) -> bool {
    matches!(t, TypeInfo::Int | TypeInfo::Float)
}

fn is_stringy(t: &TypeInfo) -> bool {
    matches!(t, TypeInfo::StrRef | TypeInfo::String)
}

/// Join two element types for a container, unifying compatible kinds.
/// Recurses into containers: `Vec<i64>` unifies with `Vec<f64>` to
/// `Vec<f64>`, and an untyped `Vec<_>` unifies with `Vec<f64>` to
/// `Vec<f64>`.
pub fn unify(a: TypeInfo, b: TypeInfo) -> TypeInfo {
    if a == b {
        return a;
    }
    match (&a, &b) {
        (TypeInfo::Vec(x), TypeInfo::Vec(y)) => {
            TypeInfo::Vec(Box::new(unify((**x).clone(), (**y).clone())))
        }
        (TypeInfo::Dict(k1, v1), TypeInfo::Dict(k2, v2)) => TypeInfo::Dict(
            Box::new(unify((**k1).clone(), (**k2).clone())),
            Box::new(unify((**v1).clone(), (**v2).clone())),
        ),
        (TypeInfo::Option(x), TypeInfo::Option(y)) => {
            TypeInfo::Option(Box::new(unify((**x).clone(), (**y).clone())))
        }
        (TypeInfo::Borrowed(x), TypeInfo::Borrowed(y)) => {
            TypeInfo::Borrowed(Box::new(unify((**x).clone(), (**y).clone())))
        }
        (TypeInfo::Tuple(x), TypeInfo::Tuple(y)) if x.len() == y.len() => TypeInfo::Tuple(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| unify(a.clone(), b.clone()))
                .collect(),
        ),
        _ => {
            if is_numeric(&a) && is_numeric(&b) {
                return TypeInfo::Float;
            }
            if is_stringy(&a) && is_stringy(&b) {
                return TypeInfo::String;
            }
            // A boxed PyValue absorbs any other type (it can hold any of
            // them): `dict[str, Any]` pinned against a str-valued store
            // stays PyValue-valued instead of degrading to PyObject.
            if matches!(a, TypeInfo::PyValue) {
                return TypeInfo::PyValue;
            }
            if matches!(b, TypeInfo::PyValue) {
                return TypeInfo::PyValue;
            }
            if matches!(a, TypeInfo::PyObject) {
                return b;
            }
            if matches!(b, TypeInfo::PyObject) {
                return a;
            }
            TypeInfo::PyObject
        }
    }
}

fn numeric_join(a: &TypeInfo, b: &TypeInfo) -> TypeInfo {
    if matches!(a, TypeInfo::Float) || matches!(b, TypeInfo::Float) {
        TypeInfo::Float
    } else {
        TypeInfo::Int
    }
}

/// Render an expression, then coerce it to the expected type when the
/// inference says a conversion is needed.
pub fn render_typed(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
    expected: Option<TypeInfo>,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // A class name used as a VALUE (`merge_setting(..., dict_class=OrderedDict)`
    // — requests' sessions): classes are compile-time types, not runtime
    // values (the classes-as-values divergence), so the value lowers to the
    // boxed None. This is a value-position renderer — callees and type
    // positions never come through here.
    if let ExprType::Name(n) = expr
        && (matches!(symbols.get(&n.id), Some(SymbolTableNode::ClassDef(_)))
            || matches!(
                symbols.get(&n.id),
                Some(SymbolTableNode::Alias(c))
                    if matches!(symbols.get(c), Some(SymbolTableNode::ClassDef(_)))
            )
            // An IMPORTED class (`from .structures import
            // CaseInsensitiveDict` — a class used as a value argument):
            // resolve through the defining module.
            || crate::ast::tree::call::resolve_construction_class(&n.id, &symbols, &options).is_some())
    {
        options.definition_warnings.borrow_mut().push(format!(
            "class `{}` used as a value lowers to the boxed None (classes cannot be \
             runtime values in rython)",
            n.id
        ));
        return Ok(quote!(stdpython::PyValue::None_));
    }
    let tokens = expr
        .clone()
        .to_rust(ctx, options.clone(), symbols.clone())?;
    let Some(expected) = expected else {
        return Ok(tokens);
    };
    // A None literal destined for a boxed slot IS the boxed None — the
    // unit fallback below would render a bare `None` token that only
    // coincidentally compiles outside expression position.
    if matches!(expected, TypeInfo::PyValue) && crate::is_none_expr(expr) {
        return Ok(quote!(stdpython::PyValue::None_));
    }
    let actual = infer_type(expr, &options, &symbols);
    match coerce_tokens(tokens.clone(), &actual, &expected) {
        Some(coerced) => Ok(coerced),
        None => {
            // No conversion available. Leave the tokens alone: rustc
            // reports the mismatch against generated code (the
            // pre-existing behaviour) — inventing a conversion for an
            // unknown type would be worse than a compile error.
            let _ = tokens.clone();
            Ok(tokens)
        }
    }
}

/// Render an expression in a move-prone position (a call argument or a
/// container element). Names that are read more than once in the enclosing
/// function are cloned so Rust's move semantics do not consume them —
/// Python shares by reference. Method receivers are NOT rendered through
/// this wrapper, so `xs.pop(); xs.pop()` still mutates the same vector.
pub fn render_reused(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let tokens = expr
        .clone()
        .to_rust(ctx, options.clone(), symbols.clone())?;
    if let ExprType::Name(n) = expr {
        let uses = options.use_counts.get(&n.id).copied().unwrap_or(0);
        if uses > 1 {
            let t = infer_type(expr, &options, &symbols);
            // An inferred (unannotated) parameter is not statically Copy —
            // the reuse-clone rule adds `T: Clone` for it, so clone it
            // here too (a generic value would otherwise be moved into the
            // call while still being used).
            let inferred_param = options.param_type_vars.contains_key(&n.id);
            if !t.is_copy() && (!matches!(t, TypeInfo::PyObject) || inferred_param) {
                return Ok(quote!((#tokens).clone()));
            }
        }
    }
    Ok(tokens)
}

/// [`render_typed`] + [`render_reused`] in one pass: coerce to the
/// expected type, then clone non-Copy names that are reused. Used for
/// call arguments where the callee's signature is known.
pub fn render_typed_reused(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
    expected: Option<TypeInfo>,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let tokens = render_typed(expr, ctx, options.clone(), symbols.clone(), expected)?;
    if let ExprType::Name(n) = expr {
        let uses = options.use_counts.get(&n.id).copied().unwrap_or(0);
        if uses > 1 {
            let t = infer_type(expr, &options, &symbols);
            // See render_reused: an inferred parameter is not statically
            // Copy, so clone it at the call site (its `T: Clone` bound is
            // the reuse-clone rule's).
            let inferred_param = options.param_type_vars.contains_key(&n.id);
            if !t.is_copy() && (!matches!(t, TypeInfo::PyObject) || inferred_param) {
                return Ok(quote!((#tokens).clone()));
            }
        }
    }
    Ok(tokens)
}

/// Expected type for a CALL ARGUMENT from the callee's parameter
/// annotation. Skips `str`: `str` params lower to `impl Into<String>`,
/// which accepts `&'static str` directly — coercing literals to `String`
/// would only add `.to_string()` noise. Container and numeric params get
/// the real expected type (a `Vec<String>` param still needs owned
/// strings, but those come from the list-literal lowering).
/// Whether a name annotation is one of Python's builtin types whose
/// `type(...)` object acts as its own class (`type(int)` IS `int`). This
/// boundary function is the SINGLE source for that name set: bare
/// container names are otherwise loud errors (§3.2), so routing through
/// [`annotation_type_info`] would silently drop them while wrongly
/// accepting `object`/`Any` and exception names.
pub fn is_builtin_type_annotation(ann: &ExprType) -> bool {
    match ann {
        ExprType::Name(n) => matches!(
            n.id.as_str(),
            "int"
                | "float"
                | "bool"
                | "str"
                | "bytes"
                | "bytearray"
                | "list"
                | "tuple"
                | "set"
                | "dict"
                | "frozenset"
        ),
        _ => false,
    }
}

/// Whether a Rust-side element type can live inside the boxed
/// heterogeneous container (`Vec<stdpython::PyValue>` /
/// `PyDict<PyValue, PyValue>` / `PySet<PyValue>`): exactly the types
/// [`PyValue`](stdpython::PyValue) has variants or `From` impls for.
/// Class instances, dicts-as-elements, sets-as-elements, ranges and
/// ndarrays have no boxed variant and stay refused loudly (issue #130).
pub fn is_boxable_value_type(t: &TypeInfo) -> bool {
    matches!(
        t,
        TypeInfo::Int
            | TypeInfo::Float
            | TypeInfo::Bool
            | TypeInfo::String
            | TypeInfo::StrRef
            | TypeInfo::Bytes
            | TypeInfo::Tuple(_)
            | TypeInfo::Option(_)
            | TypeInfo::Vec(_)
            // A nested DICT value (`{'ProviderType': 'sso', 'Credentials':
            // {...}}` — botocore's credentials.py): the mixed-value dict
            // widens to PyDict<String, PyValue> and the nested dict boxes
            // via PyValue::from(PyDict) (issue #180).
            | TypeInfo::Dict(_, _)
            | TypeInfo::StrOrBytes
            | TypeInfo::PyValue
    )
}

pub fn call_arg_expected_type(ann: &ExprType) -> Option<TypeInfo> {
    let t = annotation_type_info(ann)?;
    if matches!(t, TypeInfo::String) {
        None
    } else {
        Some(t)
    }
}

/// Map a Python type annotation expression to the [`TypeInfo`] of the Rust
/// type codegen produces for it. Used to derive the expected type of a
/// call argument from the callee's parameter annotation.
pub fn annotation_type_info(ann: &ExprType) -> Option<TypeInfo> {
    // `T | None` (and `None | T`) is Option<T>; the inner type resolves
    // through the same mapping. A union of two non-None members that map
    // to the same TypeInfo (bytes | bytearray) is that type. `str | bytes`
    // is the StrOrBytes heterogeneous union; any other union whose members
    // are all boxable (int/float/bool/str/bytes/tuple/Literal/Any/None) is
    // the boxed PyValue (issue #121).
    if let ExprType::BinOp(op) = ann
        && matches!(op.op, crate::BinOps::BitOr)
    {
        if crate::is_str_bytes_union(ann) {
            return Some(TypeInfo::StrOrBytes);
        }
        let members = crate::union_members(ann);
        if crate::is_none_expr(&op.left) {
            if let Some(t) = annotation_type_info(&op.right) {
                // A boxed PyValue already contains None (`bool | str |
                // None` is PyValue, not Option<PyValue>).
                if matches!(t, TypeInfo::PyValue) {
                    return Some(t);
                }
                return Some(TypeInfo::Option(Box::new(t)));
            }
            return boxable_union(members);
        }
        if crate::is_none_expr(&op.right) {
            if let Some(t) = annotation_type_info(&op.left) {
                if matches!(t, TypeInfo::PyValue) {
                    return Some(t);
                }
                return Some(TypeInfo::Option(Box::new(t)));
            }
            return boxable_union(members);
        }
        let l = annotation_type_info(&op.left);
        let r = annotation_type_info(&op.right);
        if let (Some(l), Some(r)) = (l, r)
            && l == r
        {
            return Some(l);
        }
        return boxable_union(members);
    }
    match ann {
        ExprType::Name(n) => match n.id.as_str() {
            "int" => Some(TypeInfo::Int),
            "float" => Some(TypeInfo::Float),
            "bool" => Some(TypeInfo::Bool),
            "str" => Some(TypeInfo::String),
            "bytes" => Some(TypeInfo::Bytes),
            // ssl.TLSVersion is an IntEnum: plain ints in the runtime.
            "TLSVersion" => Some(TypeInfo::Int),
            // `Any` (typing.Any) and `object`: a value of unknown type —
            // the boxed heterogeneous value.
            "Any" | "object" => Some(TypeInfo::PyValue),
            // Builtin exception names (`BaseException | None` — the
            // context-manager protocol) box as PyValue: the canonical
            // list lives with the raise lowering (this arm was one of two
            // drifted 33-name copies). types-module classes and
            // `memoryview` (a builtin buffer class — urllib3's
            // `readinto(b: bytearray | memoryview[int])`) box too.
            "TracebackType" | "FrameType" | "CodeType" | "memoryview" => Some(TypeInfo::PyValue),
            other if crate::ast::tree::raise_stmt::is_builtin_exception_name(other) => {
                Some(TypeInfo::PyValue)
            }
            _ => None,
        },
        ExprType::Subscript(sub) => match sub.value.as_ref() {
            ExprType::Name(n) => match n.id.as_str() {
                // `memoryview[int]` — the builtin buffer class subscript.
                "memoryview" => Some(TypeInfo::PyValue),
                "list" | "List" => {
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        Some(TypeInfo::Vec(Box::new(annotation_type_info(elt)?)))
                    } else {
                        None
                    }
                }
                "set" | "Set" | "frozenset" => {
                    // A `set[T]` annotation in the syntax-only pass: the
                    // element resolves in the symbols-aware pass.
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        Some(TypeInfo::Vec(Box::new(annotation_type_info(elt)?)))
                    } else {
                        None
                    }
                }
                "Mapping" => {
                    if let crate::SubscriptKind::Index(kv) = &sub.kind
                        && let ExprType::Tuple(t) = kv.as_ref()
                        && let [k, v] = t.elts.as_slice()
                    {
                        Some(TypeInfo::Dict(
                            Box::new(annotation_type_info(k)?),
                            Box::new(annotation_type_info(v)?),
                        ))
                    } else {
                        None
                    }
                }
                "Optional" => {
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        let inner = annotation_type_info(elt)?;
                        // Optional[bool | str] is the boxed PyValue (which
                        // already contains None), not Option<PyValue>.
                        if matches!(inner, TypeInfo::PyValue) {
                            Some(inner)
                        } else {
                            Some(TypeInfo::Option(Box::new(inner)))
                        }
                    } else {
                        None
                    }
                }
                "tuple" | "Tuple" => {
                    if let crate::SubscriptKind::Index(elt) = &sub.kind
                        && let ExprType::Tuple(t) = elt.as_ref()
                    {
                        // `tuple[T, ...]` — a variadic tuple → Vec<T>.
                        if t.elts.len() == 2
                            && matches!(
                                &t.elts[1],
                                ExprType::Constant(c)
                                    if c.0
                                        .as_ref()
                                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                            )
                        {
                            let inner = annotation_type_info(&t.elts[0])?;
                            return Some(TypeInfo::Vec(Box::new(inner)));
                        }
                        let mut infos = Vec::with_capacity(t.elts.len());
                        for e in &t.elts {
                            infos.push(annotation_type_info(e)?);
                        }
                        Some(TypeInfo::Tuple(infos))
                    } else {
                        None
                    }
                }
                "Literal" => Some(TypeInfo::PyValue),
                "dict" | "Dict" => {
                    if let crate::SubscriptKind::Index(kv) = &sub.kind
                        && let ExprType::Tuple(t) = kv.as_ref()
                        && let [k, v] = t.elts.as_slice()
                    {
                        return Some(TypeInfo::Dict(
                            Box::new(annotation_type_info(k)?),
                            Box::new(annotation_type_info(v)?),
                        ));
                    }
                    None
                }
                _ => None,
            },
            // `typing.Mapping[K, V]` lowers like the bare name.
            ExprType::Attribute(a)
                if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
                    && a.attr == "Mapping" =>
            {
                if let crate::SubscriptKind::Index(kv) = &sub.kind
                    && let ExprType::Tuple(t) = kv.as_ref()
                    && let [k, v] = t.elts.as_slice()
                {
                    Some(TypeInfo::Dict(
                        Box::new(annotation_type_info(k)?),
                        Box::new(annotation_type_info(v)?),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}

/// Whether a union member can live inside the boxed PyValue — the single
/// implementation lives in arguments.rs (`is_pyvalue_boxable_member`);
/// this module previously kept a near-verbatim clone that had drifted
/// from it in both directions.

/// A union of boxable members with no single Rust type becomes the boxed
/// PyValue. Returns None when a member is not boxable.
fn boxable_union(members: Option<Vec<&ExprType>>) -> Option<TypeInfo> {
    let members = members?;
    if members.len() < 2 {
        return None;
    }
    if members.iter().all(|m| crate::is_pyvalue_boxable_member(m)) {
        Some(TypeInfo::PyValue)
    } else {
        None
    }
}

/// Per-function type analysis: read-use counts, inferred name types, and
/// pinned types for empty containers.
#[derive(Clone, Debug, Default)]
pub struct FunctionTypeInfo {
    /// How many times each name is READ (not assigned) in the function.
    pub use_counts: HashMap<String, usize>,
    /// Inferred type of each local name (annotations win; then the type of
    /// the last literal/container assignment).
    pub name_types: HashMap<String, TypeInfo>,
    /// Names whose type came from an EXPLICIT annotation: a later plain
    /// assignment must not downgrade the annotated type (`host_params:
    /// dict[str, Any] = {}` then `host_params = {...}` keeps the boxed
    /// PyValue value type, issue #121).
    pub annotated_names: HashSet<String>,
    /// Names assigned an empty `[]`/`{}` literal whose element type was
    /// pinned by a later use; maps to the pinned container type.
    pub empty_pinned: HashMap<String, TypeInfo>,
}

/// Walk a statement list, counting reads and collecting `name = expr`
/// assignments with inferable types, then pin empty-container types from
/// later use. `options` lets the pin resolve decorator-factory callables
/// (`cached_mess_ratio = lru_cache(...)(mess_ratio)`) to their return type
/// (issue #127).
pub fn analyze_function_types(
    body: &[Statement],
    options: Option<&PythonOptions>,
    symbols: Option<&SymbolTableScopes>,
) -> FunctionTypeInfo {
    let mut info = FunctionTypeInfo::default();
    for stmt in body {
        analyze_statement_types(stmt, &mut info, options, symbols);
    }
    pin_empty_containers(body, &mut info, symbols, options);
    info
}

fn analyze_statement_types(
    stmt: &Statement,
    info: &mut FunctionTypeInfo,
    options: Option<&PythonOptions>,
    symbols: Option<&SymbolTableScopes>,
) {
    match &stmt.statement {
        // A bare annotated local (`key: str` — urllib3's
        // ssl_match_hostname): the annotation types the name.
        StatementType::AnnotatedName { name, annotation } => {
            if let Some(t) = crate::annotation_type_info(annotation)
                .or_else(|| crate::resolve_alias_typeinfo(annotation, symbols?, options?))
            {
                info.name_types.insert(name.clone(), t);
                info.annotated_names.insert(name.clone());
            }
        }
        StatementType::Assign(assign) => {
            // Record `name = <inferable expr>` (syntactic only, unless an
            // annotation pins it — an annotated assignment's annotation
            // wins, so `xs: list[float] = []` pins the empty literal).
            if let [ExprType::Name(name)] = assign.targets.as_slice() {
                let mut t = match assign
                    .annotation
                    .as_ref()
                    .and_then(crate::annotation_type_info)
                    .or_else(|| {
                        assign
                            .annotation
                            .as_ref()
                            .and_then(|a| crate::resolve_alias_typeinfo(a, symbols?, options?))
                    })
                    .or_else(|| {
                        // A module-level TYPE ALIAS annotation
                        // (`filtered_results: CoherenceMatches = []` —
                        // charset_normalizer): resolve through symbols.
                        assign
                            .annotation
                            .as_ref()
                            .and_then(|a| crate::resolve_alias_typeinfo(a, symbols?, options?))
                    }) {
                    // An annotation pins the type outright.
                    Some(ann) => ann,
                    // Unparseable annotation: a call to a known function
                    // resolves through its (alias-aware) return type
                    // (`chunk_languages = cached_coherence_ratio(...)` →
                    // Vec<(String, f64)>), else the value's syntactic type
                    // (still pinable by later use).
                    None => match &assign.value {
                        ExprType::Call(c) => call_return_typeinfo(c, symbols, options)
                            .unwrap_or_else(|| syntactic_type(&assign.value)),
                        _ => syntactic_type(&assign.value),
                    },
                };
                // Dict keys normalize to String (matches literal lowering
                // and `dict[str, V]` annotations); empty dicts and lists
                // are remembered for pinning from later use.
                t = match t {
                    TypeInfo::Dict(k, v) if matches!(*k, TypeInfo::StrRef) => {
                        TypeInfo::Dict(Box::new(TypeInfo::String), v)
                    }
                    other => other,
                };
                let annotated = assign.annotation.is_some();
                if annotated {
                    info.annotated_names.insert(name.id.clone());
                }
                // Annotations win: a plain assignment must not downgrade a
                // name whose type an annotation pinned.
                if !matches!(t, TypeInfo::PyObject)
                    && (annotated || !info.annotated_names.contains(&name.id))
                {
                    info.name_types.insert(name.id.clone(), t.clone());
                    // Empty container: remember it to pin from later use.
                    if is_empty_container(&assign.value) {
                        info.empty_pinned.insert(name.id.clone(), t);
                    }
                }
            }
            // Count reads in targets (a[i] = v reads a and i) and value.
            count_expr_reads(&assign.value, info);
            for target in &assign.targets {
                count_target_reads(target, info);
            }
        }
        StatementType::AugAssign(a) => {
            count_expr_reads(&a.target, info);
            count_expr_reads(&a.value, info);
        }
        StatementType::Expr(e) => count_expr_reads(&e.value, info),
        StatementType::Return(Some(e)) => count_expr_reads(&e.value, info),
        StatementType::If(s) => {
            count_expr_reads(&s.test, info);
            for b in &s.body {
                analyze_statement_types(b, info, options, symbols);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        StatementType::While(s) => {
            count_expr_reads(&s.test, info);
            for b in &s.body {
                analyze_statement_types(b, info, options, symbols);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        StatementType::For(s) => {
            count_expr_reads(&s.iter, info);
            count_target_reads(&s.target, info);
            for b in &s.body {
                analyze_statement_types(b, info, options, symbols);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        StatementType::With(s) => {
            for item in &s.items {
                count_expr_reads(&item.context_expr, info);
                if let Some(vars) = &item.optional_vars {
                    count_target_reads(vars, info);
                }
            }
            for b in &s.body {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        StatementType::Try(s) => {
            for b in &s.body {
                analyze_statement_types(b, info, options, symbols);
            }
            for handler in &s.handlers {
                if let Some(t) = &handler.exception_type {
                    count_expr_reads(t, info);
                }
                if let Some(name) = &handler.name {
                    info.use_counts.remove(name); // bound by except, not read
                }
                for b in &handler.body {
                    analyze_statement_types(b, info, options, symbols);
                }
            }
            for b in &s.orelse {
                analyze_statement_types(b, info, options, symbols);
            }
            for b in &s.finalbody {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        StatementType::FunctionDef(f) => {
            // A nested function is a new scope: its locals do not belong to
            // the enclosing function's type analysis. BUT its ANNOTATED
            // parameter types are usable for empty-container pinning — a
            // nested closure `def inner(x: float): md_ratios.append(x)`
            // pins the outer `md_ratios = []` to Vec<f64> (charset_
            // normalizer's from_bytes). Record them and recurse the body
            // so captured uses of enclosing empty containers resolve.
            for p in f
                .args
                .posonlyargs
                .iter()
                .chain(f.args.args.iter())
                .chain(f.args.kwonlyargs.iter())
            {
                if let Some(ann) = p.annotation.as_deref()
                    && let Some(t) = crate::annotation_type_info(ann)
                {
                    info.name_types.insert(p.arg.clone(), t);
                }
            }
            for b in &f.body {
                analyze_statement_types(b, info, options, symbols);
            }
        }
        _ => {}
    }
}

fn is_empty_container(expr: &ExprType) -> bool {
    matches!(expr, ExprType::List(l) if l.is_empty())
        || matches!(expr, ExprType::Dict(d) if d.keys.is_empty())
}

/// The syntactic type of an expression, ignoring the analysis maps (used
/// while building them).
pub(crate) fn syntactic_type(expr: &ExprType) -> TypeInfo {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => TypeInfo::Int,
            Some(litrs::Literal::Float(_)) => TypeInfo::Float,
            Some(litrs::Literal::Bool(_)) => TypeInfo::Bool,
            Some(litrs::Literal::String(_)) => TypeInfo::StrRef,
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => TypeInfo::Bytes,
            _ => TypeInfo::PyObject,
        },
        ExprType::JoinedStr(_) | ExprType::FormattedValue(_) => TypeInfo::String,
        ExprType::List(l) => {
            let mut elt = TypeInfo::PyObject;
            for e in l {
                let t = syntactic_type(e);
                elt = unify(elt, t);
            }
            TypeInfo::Vec(Box::new(elt))
        }
        ExprType::Dict(d) => {
            let mut k = TypeInfo::PyObject;
            let mut v = TypeInfo::PyObject;
            for (key, value) in d.keys.iter().zip(d.values.iter()) {
                if let Some(key) = key {
                    k = unify(k, syntactic_type(key));
                }
                v = unify(v, syntactic_type(value));
            }
            TypeInfo::Dict(Box::new(k), Box::new(v))
        }
        // Builtin calls: empty-container pinning resolves `xs.append(len(s))`
        // through here (resolve_type's fallback), so len()/count() must
        // agree with the `as i64` codegen emission — a list pinned only from
        // appended lengths is Vec<i64>, not Vec<usize>.
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) => builtin_call_type(&n.id).unwrap_or(TypeInfo::PyObject),
            _ => TypeInfo::PyObject,
        },
        _ => TypeInfo::PyObject,
    }
}

/// Pin the element/key types of empty containers from a later use. Called
/// with the full statement list AFTER `name_types` is populated so that
/// `append(d[k])` can resolve `d`'s type too.
pub fn pin_empty_containers(
    body: &[Statement],
    info: &mut FunctionTypeInfo,
    symbols: Option<&SymbolTableScopes>,
    options: Option<&PythonOptions>,
) {
    let mut suggested: HashMap<String, TypeInfo> = HashMap::new();
    for stmt in body {
        collect_use_suggestions(stmt, info, symbols, options, &mut suggested);
    }
    for (name, t) in suggested {
        if info.empty_pinned.contains_key(&name) {
            // Unify with any existing (annotated) type: an annotated
            // `result: list[str] = []` must not be clobbered by a use
            // suggestion whose element type is still unknown.
            let final_t = match info.name_types.get(&name) {
                Some(existing) => unify(existing.clone(), t),
                None => t,
            };
            info.name_types.insert(name.clone(), final_t.clone());
            info.empty_pinned.insert(name, final_t);
        }
    }
}

fn collect_use_suggestions(
    stmt: &Statement,
    info: &FunctionTypeInfo,
    symbols: Option<&SymbolTableScopes>,
    options: Option<&PythonOptions>,
    out: &mut HashMap<String, TypeInfo>,
) {
    match &stmt.statement {
        StatementType::Assign(assign) => {
            // name.append(e) / name.insert(i, e) / name.extend(iter)
            if let [ExprType::Name(_target)] = assign.targets.as_slice()
                && let ExprType::Call(call) = &assign.value
                && let ExprType::Attribute(attr) = call.func.as_ref()
                && let ExprType::Name(recv) = attr.value.as_ref()
                && info.empty_pinned.contains_key(&recv.id)
            {
                match attr.attr.as_str() {
                    "append" | "push" => {
                        if let Some(arg) = call.args.first() {
                            let t = resolve_type(arg, info, symbols, options);
                            // An UNKNOWN element (`parts.append(part)` where
                            // part is an external call result — s3transfer):
                            // box it (Vec<PyValue> divergence) — but only
                            // when the name has NO annotated/existing type,
                            // which must win over the unknown suggestion
                            // (`result: list[float] = []` then
                            // `result.append(x[j])` keeps Vec<f64>).
                            let t = if matches!(t, TypeInfo::PyObject)
                                && !info.name_types.contains_key(&recv.id)
                            {
                                TypeInfo::PyValue
                            } else {
                                t
                            };
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), t.clone()))
                                .or_insert(TypeInfo::Vec(Box::new(t)));
                        }
                    }
                    "insert" => {
                        if let Some(arg) = call.args.get(1) {
                            let t = resolve_type(arg, info, symbols, options);
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), t.clone()))
                                .or_insert(TypeInfo::Vec(Box::new(t)));
                        }
                    }
                    _ => {}
                }
            }
            // d[k] = v pins dict key/value; v[i] = e pins list element.
            for target in &assign.targets {
                if let ExprType::Subscript(sub) = target
                    && let ExprType::Name(recv) = sub.value.as_ref()
                    && info.empty_pinned.contains_key(&recv.id)
                {
                    let v = resolve_type(&assign.value, info, symbols, options);
                    if let crate::SubscriptKind::Index(idx) = &sub.kind {
                        // Keys normalize to String, matching dict literals
                        // and `dict[str, V]` annotations. An UNKNOWN key
                        // (`modeled_action.name` — an attribute on a foreign
                        // object, boto3's document_actions) is a String in
                        // practice.
                        let k_raw = resolve_type(idx, info, symbols, options);
                        let key_unknown = matches!(k_raw, TypeInfo::PyObject);
                        let val_unknown = matches!(v, TypeInfo::PyObject);
                        let k = match k_raw {
                            TypeInfo::StrRef | TypeInfo::PyObject => TypeInfo::String,
                            other => other,
                        };
                        // An EXISTING container type wins over a suggestion
                        // whose key or value is unknown: `d: dict[int, int]
                        // = {}` followed by `d[i] = i` (the loop variable is
                        // untyped) must stay `dict[int, int]` — the unknown
                        // suggestion would otherwise downgrade the
                        // annotation to a boxed dict (issue #163; unify
                        // lets PyValue absorb Int). Unknown components only
                        // type an as-yet-UNTYPED container (boto3's
                        // document_actions).
                        if info.name_types.contains_key(&recv.id) && (key_unknown || val_unknown) {
                            return;
                        }
                        let ty = TypeInfo::Dict(Box::new(k), Box::new(v));
                        // An UNKNOWN value type (`modeled_actions[
                        // modeled_action.name] = modeled_action` where the
                        // value is a foreign object — boto3's
                        // document_actions): box the value.
                        let ty = match ty {
                            TypeInfo::Dict(k, v) if matches!(*v, TypeInfo::PyObject) => {
                                TypeInfo::Dict(k, Box::new(TypeInfo::PyValue))
                            }
                            other => other,
                        };
                        out.entry(recv.id.clone())
                            .and_modify(|e| *e = unify(e.clone(), ty.clone()))
                            .or_insert(ty);
                    }
                }
            }
        }
        StatementType::Expr(e) => {
            // `"; ".join(parts)` — a str join pins its ARGUMENT (the list)
            // to Vec<String>, even when the receiver is a literal.
            if let ExprType::Call(call) = &e.value
                && let ExprType::Attribute(attr) = call.func.as_ref()
                && attr.attr == "join"
                && let Some(arg) = call.args.first()
                && let ExprType::Name(arg_name) = arg
                && info.empty_pinned.contains_key(&arg_name.id)
            {
                let suggestion = TypeInfo::Vec(Box::new(TypeInfo::String));
                out.entry(arg_name.id.clone())
                    .and_modify(|t| *t = unify(t.clone(), suggestion.clone()))
                    .or_insert(suggestion);
            }
            if let ExprType::Call(call) = &e.value
                && let ExprType::Attribute(attr) = call.func.as_ref()
                && let ExprType::Name(recv) = attr.value.as_ref()
                && info.empty_pinned.contains_key(&recv.id)
            {
                match attr.attr.as_str() {
                    // d.get(k) read pins the key type.
                    "get" => {
                        if let Some(arg) = call.args.first() {
                            let k = match resolve_type(arg, info, symbols, options) {
                                TypeInfo::StrRef => TypeInfo::String,
                                other => other,
                            };
                            let t = TypeInfo::Dict(Box::new(k), Box::new(TypeInfo::PyObject));
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), t.clone()))
                                .or_insert(t);
                        }
                    }
                    // Expression-statement appends/inserts:
                    // `xs.append(v)` pins xs's element type.
                    "append" | "push" => {
                        if let Some(arg) = call.args.first() {
                            let t = resolve_type(arg, info, symbols, options);
                            // An UNKNOWN element (`parts.append(part)` where
                            // part is an external call result): box it — but
                            // only when the name has no annotated/existing
                            // type (which must win).
                            let t = if matches!(t, TypeInfo::PyObject)
                                && !info.name_types.contains_key(&recv.id)
                            {
                                TypeInfo::PyValue
                            } else {
                                t
                            };
                            let suggestion = TypeInfo::Vec(Box::new(t));
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), suggestion.clone()))
                                .or_insert(suggestion);
                        }
                    }
                    // `xs.extend(ys)` pins xs's element type to ys's.
                    "extend" => {
                        if let Some(arg) = call.args.first() {
                            let t = match resolve_type(arg, info, symbols, options) {
                                TypeInfo::Vec(e) => *e,
                                other => other,
                            };
                            let suggestion = TypeInfo::Vec(Box::new(t));
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), suggestion.clone()))
                                .or_insert(suggestion);
                        }
                    }
                    "insert" => {
                        if let Some(arg) = call.args.get(1) {
                            let t = resolve_type(arg, info, symbols, options);
                            let suggestion = TypeInfo::Vec(Box::new(t));
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), suggestion.clone()))
                                .or_insert(suggestion);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    // Recurse into control flow.
    match &stmt.statement {
        StatementType::If(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
            for b in &s.orelse {
                collect_use_suggestions(b, info, symbols, options, out);
            }
        }
        StatementType::While(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
        }
        StatementType::For(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
        }
        StatementType::With(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
        }
        // `return "; ".join(parts)` — a str join in return position pins its
        // list argument to Vec<String> (urllib3's fields.py `_render_parts`).
        StatementType::Return(Some(e)) => {
            if let ExprType::Call(call) = &e.value
                && let ExprType::Attribute(attr) = call.func.as_ref()
                && attr.attr == "join"
                && let Some(arg) = call.args.first()
                && let ExprType::Name(arg_name) = arg
                && info.empty_pinned.contains_key(&arg_name.id)
            {
                let suggestion = TypeInfo::Vec(Box::new(TypeInfo::String));
                out.entry(arg_name.id.clone())
                    .and_modify(|t| *t = unify(t.clone(), suggestion.clone()))
                    .or_insert(suggestion);
            }
        }
        StatementType::Try(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
            for h in &s.handlers {
                for b in &h.body {
                    collect_use_suggestions(b, info, symbols, options, out);
                }
            }
        }
        // A NESTED function captures enclosing names (Python closure
        // semantics): `md_ratios.append(x)` inside a nested def pins the
        // outer `md_ratios = []` (charset_normalizer's from_bytes). The
        // nested function's OWN locals must not pollute the enclosing
        // analysis, but a use of an enclosing empty container is exactly
        // the pin we want.
        StatementType::FunctionDef(f) => {
            for b in &f.body {
                collect_use_suggestions(b, info, symbols, options, out);
            }
        }
        _ => {}
    }
}

/// Resolve a name's type through the name_types map (falling back to
/// syntactic inference for expressions that don't need the map).
/// Resolve a function's return annotation to a TypeInfo, following
/// module-level TYPE ALIASES (`CoherenceMatches = List[CoherenceMatch]` in
/// models.py) and imported aliases — `cached_coherence_ratio(...) ->
/// CoherenceMatches` must pin `cd_ratios` to Vec<Vec<(String, f64)>>, not
/// stay an opaque class-name ident (charset_normalizer).
pub fn resolve_alias_typeinfo(
    ann: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<TypeInfo> {
    thread_local! {
        static RA_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let d = RA_DEPTH.with(|c| c.get());
    // A self-referential alias (`JsonType = ... | Sequence["JsonType"]`)
    // recurses through the chain; the boxed PyValue is the correct
    // resolution for the cycle.
    if d > 64 {
        return Some(TypeInfo::PyValue);
    }
    RA_DEPTH.with(|c| c.set(d + 1));
    let result = resolve_alias_typeinfo_inner(ann, symbols, options);
    RA_DEPTH.with(|c| c.set(d));
    return result;
}

fn resolve_alias_typeinfo_inner(
    ann: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<TypeInfo> {
    match ann {
        // `T | None`: Option of the alias-resolved inner type
        // (`list[CharsetMatch] | None`). A boxed PyValue already contains
        // None (`CertType = ... | None` → PyValue, not Option<PyValue>).
        ExprType::BinOp(op) if matches!(op.op, crate::BinOps::BitOr) => {
            if crate::is_none_expr(&op.left) {
                if let Some(t) = resolve_alias_typeinfo(&op.right, symbols, options) {
                    if matches!(t, TypeInfo::PyValue) {
                        return Some(t);
                    }
                    return Some(TypeInfo::Option(Box::new(t)));
                }
                return None;
            }
            if crate::is_none_expr(&op.right) {
                if let Some(t) = resolve_alias_typeinfo(&op.left, symbols, options) {
                    if matches!(t, TypeInfo::PyValue) {
                        return Some(t);
                    }
                    return Some(TypeInfo::Option(Box::new(t)));
                }
                return None;
            }
            // A general union with a class member (`int | Retry`): resolve
            // both sides; distinct members box into PyValue.
            let l = resolve_alias_typeinfo(&op.left, symbols, options);
            let r = resolve_alias_typeinfo(&op.right, symbols, options);
            if let (Some(l), Some(r)) = (l, r) {
                if l == r {
                    return Some(l);
                }
                return Some(TypeInfo::PyValue);
            }
            annotation_type_info(ann)
        }
        // `_t.TimeoutType` — a module-path attribute (`import requests.
        // _types as _t`): resolve the alias name in the imported module's
        // scope (requests/_types.py defines its TypeAliases under
        // `if TYPE_CHECKING:`, which the emitter skips but find_symbols
        // still records).
        ExprType::Attribute(attr) => {
            // `typing.Any` — the typing module is never in module_defs;
            // Any is the boxed value directly.
            if attr.attr == "Any"
                && matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "typing")
            {
                return Some(TypeInfo::PyValue);
            }
            let ExprType::Name(module) = attr.value.as_ref() else {
                // A NESTED module chain (`OpenSSL.SSL.Connection` — a
                // pyOpenSSL class annotation, urllib3's WrappedSocket): the
                // root is an external import — a boxed value.
                if crate::root_name(&attr.value).is_some_and(|r| r != "typing") {
                    return Some(TypeInfo::PyValue);
                }
                return annotation_type_info(ann);
            };
            // A SELF-module reference (`connection._TYPE_SOCKET_OPTIONS`
            // inside urllib3/connection.py): the attribute is a name in the
            // CURRENT module's symbols — resolve it there. An Import symbol
            // (`socket.socket` — `import socket` registers `socket`) is NOT
            // a local type name: the module-name loop below resolves it
            // (external module → PyValue).
            if symbols.get(&attr.attr).is_some()
                && !symbols.get(&attr.attr).is_some_and(|s| {
                    matches!(
                        s,
                        crate::SymbolTableNode::ImportFrom(_) | crate::SymbolTableNode::Import(_)
                    )
                })
            {
                return resolve_alias_typeinfo(
                    &ExprType::Name(crate::ast::tree::name::Name {
                        id: attr.attr.clone(),
                    }),
                    symbols,
                    options,
                );
            }
            // The module name may itself be an ALIAS (`from . import
            // _types as _t` registers `_t` → Alias("_types")): follow the
            // chain to the Import/ImportFrom, then resolve the attribute
            // name in the imported module's scope.
            let mut module_name = module.id.clone();
            let mut hops = 0;
            let path: Vec<String> = loop {
                if hops > 16 {
                    return None;
                }
                hops += 1;
                match symbols.get(&module_name) {
                    // A name shadowed by a try/except fallback
                    // (`ssl = None` after `import ssl` — urllib3): the
                    // module is external — a boxed value.
                    Some(SymbolTableNode::Assign { value, .. }) if crate::is_none_expr(value) => {
                        return Some(TypeInfo::PyValue);
                    }
                    Some(SymbolTableNode::Alias(canonical)) => {
                        module_name = canonical.clone();
                    }
                    // An import from a module outside the crate
                    // (`ssl.SSLContext` — urllib3): an external class — a
                    // boxed value.
                    Some(SymbolTableNode::Import(i))
                        if !options.module_defs.contains_key(
                            &i.names
                                .first()
                                .map(|a| {
                                    a.name.split('.').map(|s| s.to_string()).collect::<Vec<_>>()
                                })
                                .unwrap_or_default(),
                        ) =>
                    {
                        return Some(TypeInfo::PyValue);
                    }
                    Some(SymbolTableNode::Import(i)) => {
                        let name = i.names.first()?.name.clone();
                        break name.split('.').map(|s| s.to_string()).collect();
                    }
                    Some(SymbolTableNode::ImportFrom(i)) => {
                        let mut path = i.resolved_module_path(options);
                        // `from . import X` — the name is an alias, not
                        // part of the module path (`from . import _types
                        // as _t` → requests/_types.py); `from .util
                        // import connection` — the imported name IS the
                        // submodule (`connection._TYPE_SOCKET_OPTIONS` →
                        // urllib3/util/connection.py).
                        if let Some(alias) = i.names.iter().find(|a| {
                            a.name == module_name
                                || a.asname.as_deref() == Some(module_name.as_str())
                        }) {
                            path.push(alias.name.clone());
                        } else if i.module.is_empty() && i.names.len() == 1 {
                            path.push(i.names[0].name.clone());
                        }
                        break path;
                    }
                    _ => return annotation_type_info(ann),
                }
            };
            let module = options.module_defs.get(&path)?;
            let module: &crate::Module = module;
            let syms = module.clone().find_symbols(SymbolTableScopes::new());
            let r = resolve_alias_typeinfo(
                &ExprType::Name(crate::ast::tree::name::Name {
                    id: attr.attr.clone(),
                }),
                &syms,
                options,
            );
            r
        }
        // A STRING annotation (`Sequence["JsonType"]` — a forward
        // reference): resolve it as the name (requests/_types.py).
        ExprType::Constant(c) => {
            if let Some(litrs::Literal::String(s)) = &c.0 {
                let text = s.value().to_string();
                if !text.is_empty()
                    && text
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                {
                    return resolve_alias_typeinfo(
                        &ExprType::Name(crate::ast::tree::name::Name { id: text }),
                        symbols,
                        options,
                    );
                }
            }
            annotation_type_info(ann)
        }
        // A bare name: a builtin scalar, or an alias/import chain.
        ExprType::Name(n) => match symbols.get(&n.id) {
            // An ALIAS (`from ._base_connection import ProxyConfig as
            // ProxyConfig` — a self-aliasing re-export): follow to the
            // canonical name (the depth guard breaks cycles) — UNLESS the
            // alias came from an aliased EXTERNAL import whose canonical
            // name a later local class shadows (`_HttplibHTTPResponse`
            // vs. urllib3's own HTTPResponse): the annotation means the
            // external class — a boxed value — not the local one (which
            // would make the field self-recursive, E0072).
            Some(SymbolTableNode::Alias(canonical)) => {
                if crate::ast::tree::module::aliased_external_import(&n.id, options) {
                    return Some(TypeInfo::PyValue);
                }
                resolve_alias_typeinfo(
                    &ExprType::Name(crate::ast::tree::name::Name {
                        id: canonical.clone(),
                    }),
                    symbols,
                    options,
                )
            }
            Some(SymbolTableNode::Assign { value, .. }) => {
                // A TypeVar (`_DT = TypeVar("_DT")`) is a compile-time
                // generic — boxed when it appears in a union.
                if is_typevar_call(value) {
                    return Some(TypeInfo::PyValue);
                }
                // A NewType alias (`RecordPath = NewType("RecordPath",
                // str)` — pip's wheel): the str base.
                if let ExprType::Call(c) = value
                    && matches!(c.func.as_ref(), ExprType::Name(f) if f.id == "NewType")
                {
                    return Some(TypeInfo::String);
                }
                resolve_alias_typeinfo(value, symbols, options)
            }
            // An import from the `typing`/`typing_extensions` modules
            // (`from typing_extensions import Buffer` — requests/_types.py)
            // is a boxed value.
            Some(SymbolTableNode::ImportFrom(i))
                if matches!(i.module.as_str(), "typing" | "typing_extensions") =>
            {
                Some(TypeInfo::PyValue)
            }
            // An import from a module outside the crate (`CookieJar` from
            // http.cookiejar): an external class — a boxed value.
            Some(SymbolTableNode::ImportFrom(i))
                if !options
                    .module_defs
                    .contains_key(&i.resolved_module_path(options)) =>
            {
                Some(TypeInfo::PyValue)
            }
            Some(SymbolTableNode::ImportFrom(i)) => {
                let path = i.resolved_module_path(options);
                let module = options.module_defs.get(&path)?;
                let module: &crate::Module = module;
                let syms = module.clone().find_symbols(SymbolTableScopes::new());
                match syms.get(&n.id) {
                    Some(SymbolTableNode::Assign { value, .. }) => {
                        if is_typevar_call(value) {
                            return Some(TypeInfo::PyValue);
                        }
                        // A NewType alias (`NormalizedName =
                        // NewType("NormalizedName", str)` — pip's
                        // packaging): the str base.
                        if let ExprType::Call(c) = value
                            && matches!(c.func.as_ref(), ExprType::Name(f) if f.id == "NewType")
                        {
                            return Some(TypeInfo::String);
                        }
                        resolve_alias_typeinfo(value, &syms, options)
                    }
                    // An imported class (`from urllib3.util.retry import
                    // Retry`): the struct ident. A class that is only a
                    // TYPE_CHECKING stub in its own module (`if TYPE_CHECKING:
                    // class BaseHTTPConnection(Protocol)` — urllib3's
                    // _base_connection) is never generated: the annotation
                    // resolves to the boxed PyValue instead of a bare name
                    // that would not exist in the crate.
                    Some(SymbolTableNode::ClassDef(_)) => {
                        if crate::ast::tree::module::module_def_has_runtime_item(
                            options, &path, &n.id,
                        ) {
                            Some(TypeInfo::Class(n.id.clone()))
                        } else {
                            Some(TypeInfo::PyValue)
                        }
                    }
                    // A RE-EXPORT (`from .connection import ProxyConfig`
                    // where connection.py does `from ._base_connection
                    // import ProxyConfig` — urllib3): follow the chain in
                    // the DEFINING module's scope.
                    Some(SymbolTableNode::ImportFrom(_)) | Some(SymbolTableNode::Alias(_)) => {
                        resolve_alias_typeinfo(
                            &ExprType::Name(crate::ast::tree::name::Name { id: n.id.clone() }),
                            &syms,
                            options,
                        )
                    }
                    _ => None,
                }
            }
            // A user-defined class name (`list[CharsetMatch]`): the struct
            // ident, the same path parameters use.
            Some(SymbolTableNode::ClassDef(_)) => Some(TypeInfo::Class(n.id.clone())),
            _ => annotation_type_info(ann),
        },
        // A container generic whose ELEMENT may be an alias: rebuild with
        // the resolved element type.
        ExprType::Subscript(sub) => {
            let container = match sub.value.as_ref() {
                // Bare `Iterable[...]` / `IO[Any]` / `Sequence[...]` etc. —
                // typing generics tolerated inside a boxed PyValue union.
                ExprType::Name(n)
                    if matches!(
                        n.id.as_str(),
                        "Union"
                            | "IO"
                            | "Iterable"
                            | "Sequence"
                            | "Iterator"
                            | "Generator"
                            | "Callable"
                            | "SupportsRead"
                            | "SupportsItems"
                            | "Mapping"
                            | "MutableMapping"
                            | "Type"
                            | "Optional"
                            | "Literal"
                            | "Any"
                    ) =>
                {
                    return Some(TypeInfo::PyValue);
                }
                // A subscripted name that is a real symbol (`Morsel[
                // dict[str, str]]` — an imported class generic from
                // http.cookiejar) is a boxed value — except the container
                // generics, which resolve through the second match below.
                ExprType::Name(n)
                    if symbols.get(&n.id).is_some()
                        && !matches!(
                            n.id.as_str(),
                            "list"
                                | "List"
                                | "tuple"
                                | "Tuple"
                                | "dict"
                                | "Dict"
                                | "set"
                                | "Set"
                                | "Optional"
                        ) =>
                {
                    return Some(TypeInfo::PyValue);
                }
                ExprType::Name(n) => n.id.as_str(),
                ExprType::Attribute(a) if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing") =>
                {
                    match a.attr.as_str() {
                        "List" => "list",
                        "Tuple" => "tuple",
                        "Dict" => "dict",
                        "Set" => "set",
                        // Other typing generics (`IO[Any]`,
                        // `Iterable[bytes | str]`, `Union[...]`,
                        // `Callable[...]`) are tolerated inside a boxed
                        // PyValue union (urllib3's `_TYPE_BODY`).
                        "Union" | "IO" | "Iterable" | "Callable" | "SupportsRead"
                        | "SupportsItems" | "Mapping" | "MutableMapping" | "Optional"
                        | "Literal" | "Any" | "Sequence" | "Iterator" | "Generator" | "Type"
                        | "ClassVar" | "Collection" | "Container" => {
                            return Some(TypeInfo::PyValue);
                        }
                        _ => return annotation_type_info(ann),
                    }
                }
                // A MODULE-PATH attribute container (`_t.SupportsRead[
                // str | bytes]`, `_t.DataType` — requests/_types.py
                // aliases): resolve through the module.
                ExprType::Attribute(a) => {
                    let ExprType::Name(module) = a.value.as_ref() else {
                        return annotation_type_info(ann);
                    };
                    // An EXTERNAL module (`queue.LifoQueue[typing.Any]` —
                    // urllib3's ConnectionPool.pool): a boxed value.
                    if let Some(crate::SymbolTableNode::Import(i)) = symbols.get(&module.id)
                        && !options.module_defs.contains_key(
                            &i.names
                                .first()
                                .map(|al| {
                                    al.name
                                        .split('.')
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default(),
                        )
                    {
                        return Some(TypeInfo::PyValue);
                    }
                    let resolved_attr = resolve_alias_typeinfo(
                        &ExprType::Name(crate::ast::tree::name::Name { id: a.attr.clone() }),
                        symbols,
                        options,
                    );
                    match resolved_attr {
                        // A container-typed alias: rebuild the container
                        // (the resolved TypeInfo's rust type).
                        Some(t) => return Some(t),
                        None => return annotation_type_info(ann),
                    }
                }
                _ => return annotation_type_info(ann),
            };
            match (container, &sub.kind) {
                ("list" | "List", crate::SubscriptKind::Index(elt)) => Some(TypeInfo::Vec(
                    Box::new(resolve_alias_typeinfo(elt, symbols, options)?),
                )),
                ("tuple" | "Tuple", crate::SubscriptKind::Index(elt)) => {
                    if let ExprType::Tuple(t) = elt.as_ref() {
                        let mut infos = Vec::with_capacity(t.elts.len());
                        for e in &t.elts {
                            infos.push(resolve_alias_typeinfo(e, symbols, options)?);
                        }
                        Some(TypeInfo::Tuple(infos))
                    } else {
                        annotation_type_info(ann)
                    }
                }
                ("dict" | "Dict", crate::SubscriptKind::Index(kv)) => {
                    if let ExprType::Tuple(t) = kv.as_ref()
                        && let [k, v] = t.elts.as_slice()
                    {
                        Some(TypeInfo::Dict(
                            Box::new(resolve_alias_typeinfo(k, symbols, options)?),
                            Box::new(resolve_alias_typeinfo(v, symbols, options)?),
                        ))
                    } else {
                        annotation_type_info(ann)
                    }
                }
                // A `set[T]` annotation (`would_be_installed:
                // set[NormalizedName]` — pip's check): the set lowers as a
                // Vec of the element type (set-ops on Vecs compile; the
                // set semantics are the documented divergence).
                ("set" | "Set" | "frozenset", crate::SubscriptKind::Index(elt)) => Some(
                    TypeInfo::Vec(Box::new(resolve_alias_typeinfo(elt, symbols, options)?)),
                ),
                _ => annotation_type_info(ann),
            }
        }
        _ => annotation_type_info(ann),
    }
}

/// Resolve a Call expression's return TypeInfo through the callee's
/// (alias-aware) return annotation: a FunctionDef (`mess_ratio -> float`)
/// or a decorator-factory assignment (`cached_coherence_ratio =
/// lru_cache(...)(coherence_ratio)`). Cross-module callees resolve their
/// annotation in the DEFINING module's scope, so `-> CoherenceMatches`
/// follows the alias chain where it was written (charset_normalizer).
/// Returns None for unknown callees.
pub fn call_return_typeinfo(
    call: &crate::Call,
    symbols: Option<&SymbolTableScopes>,
    options: Option<&PythonOptions>,
) -> Option<TypeInfo> {
    let ExprType::Name(callee) = call.func.as_ref() else {
        return None;
    };
    let symbols = symbols?;
    let options = options?;
    // Resolve the callee to (FunctionDef, its defining module's symbols):
    // same module, cross-module via ImportFrom, or the function behind a
    // decorator-factory assignment.
    let fn_name = match symbols.get(&callee.id) {
        Some(SymbolTableNode::FunctionDef(_)) => callee.id.clone(),
        // A class-construction call (`d = Dog("rex")`) produces an instance
        // of the class — this is what lets isinstance fold through the
        // inheritance tree for constructor-typed locals.
        Some(SymbolTableNode::ClassDef(_)) => {
            return Some(TypeInfo::Class(callee.id.clone()));
        }
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            let (f, _) = crate::module_function_def(options, &path, &callee.id)?;
            let ann = f.returns.as_deref()?;
            return resolve_alias_typeinfo(ann, &module_symbols(options, &path), options);
        }
        Some(SymbolTableNode::Assign { value, .. }) => {
            // `cached_mess_ratio = lru_cache(...)(mess_ratio)`: resolve the
            // underlying fn's name.
            let ExprType::Call(outer) = value else {
                return None;
            };
            let ExprType::Name(fn_name) = outer.args.first()? else {
                return None;
            };
            fn_name.id.clone()
        }
        _ => return None,
    };
    match symbols.get(&fn_name) {
        Some(SymbolTableNode::FunctionDef(f)) => {
            let ann = f.returns.as_deref()?;
            resolve_alias_typeinfo(ann, symbols, options)
        }
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            let (f, _) = crate::module_function_def(options, &path, &fn_name)?;
            let ann = f.returns.as_deref()?;
            resolve_alias_typeinfo(ann, &module_symbols(options, &path), options)
        }
        _ => None,
    }
}

/// The symbol table of a module in options.module_defs ("" root).
/// Is `value` a `TypeVar(...)` construction (`T = typing.TypeVar("T")`)?
/// A TypeVar is a compile-time generic with no runtime item — every
/// annotation position that meets one lowers to the boxed PyValue.
pub(crate) fn is_typevar_call(value: &ExprType) -> bool {
    matches!(
        value,
        ExprType::Call(c)
            if matches!(c.func.as_ref(), ExprType::Name(f) if f.id == "TypeVar")
                || matches!(c.func.as_ref(), ExprType::Attribute(a) if a.attr == "TypeVar")
    )
}

fn module_symbols(options: &PythonOptions, path: &[String]) -> SymbolTableScopes {
    match options.module_defs.get(path) {
        Some(module) => {
            let module: &crate::Module = module;
            module.clone().find_symbols(SymbolTableScopes::new())
        }
        None => SymbolTableScopes::new(),
    }
}

fn resolve_type(
    expr: &ExprType,
    info: &FunctionTypeInfo,
    symbols: Option<&SymbolTableScopes>,
    options: Option<&PythonOptions>,
) -> TypeInfo {
    thread_local! {
        static RT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    let d = RT_DEPTH.with(|c| c.get());
    if d > 200 && d % 50 == 0 {}
    RT_DEPTH.with(|c| c.set(d + 1));
    let result = resolve_type_inner(expr, info, symbols, options);
    RT_DEPTH.with(|c| c.set(d));
    return result;
}

fn resolve_type_inner(
    expr: &ExprType,
    info: &FunctionTypeInfo,
    symbols: Option<&SymbolTableScopes>,
    options: Option<&PythonOptions>,
) -> TypeInfo {
    match expr {
        ExprType::Name(n) => info
            .name_types
            .get(&n.id)
            .cloned()
            .unwrap_or_else(|| syntactic_type(expr)),
        // A call to a known function resolves through its return
        // annotation: `md_ratios.append(cached_mess_ratio(...))` pins the
        // element type from the cached fn's `-> float` (charset_normalizer).
        ExprType::Call(call) => {
            if let Some(t) = crate::call_return_typeinfo(call, symbols, options) {
                return t;
            }
            // Fall back to the plain (non-alias) return-annotation path
            // for functions whose annotation maps directly.
            if let ExprType::Name(callee) = call.func.as_ref()
                && let (Some(symbols), Some(options)) = (symbols, options)
                && let Some(SymbolTableNode::FunctionDef(f)) = symbols.get(&callee.id)
                && let Some(ty) = f.resolved_return_type(symbols, options)
            {
                return ty_to_typeinfo(&ty);
            }
            syntactic_type(expr)
        }
        // `xs[i]` resolves through the receiver's recorded type so
        // `out.append(xs[0])` pins `out`'s element type from a
        // `list[float]` parameter.
        ExprType::Subscript(sub) => {
            if let ExprType::Name(n) = sub.value.as_ref()
                && let Some(t) = info.name_types.get(&n.id)
            {
                return match t {
                    TypeInfo::Vec(inner) => (**inner).clone(),
                    TypeInfo::Dict(_, v) => (**v).clone(),
                    other => other.clone(),
                };
            }
            syntactic_type(expr)
        }
        _ => syntactic_type(expr),
    }
}

/// Map a resolved Rust type token (from a function's return annotation) to
/// a TypeInfo.
fn ty_to_typeinfo(ty: &TokenStream) -> TypeInfo {
    let s = ty.to_string();
    if s.contains("i64") {
        TypeInfo::Int
    } else if s.contains("f64") {
        TypeInfo::Float
    } else if s.contains("bool") {
        TypeInfo::Bool
    } else if s.contains("String") || s.contains("str") {
        TypeInfo::String
    } else if s.contains("Vec < u8 >") || s.contains("Vec<u8>") {
        TypeInfo::Bytes
    } else {
        TypeInfo::PyObject
    }
}

fn count_expr_reads(expr: &ExprType, info: &mut FunctionTypeInfo) {
    match expr {
        ExprType::Name(n) => {
            *info.use_counts.entry(n.id.clone()).or_insert(0) += 1;
        }
        ExprType::Attribute(a) => count_expr_reads(&a.value, info),
        ExprType::Call(c) => {
            count_expr_reads(&c.func, info);
            for a in &c.args {
                count_expr_reads(a, info);
            }
            for k in &c.keywords {
                count_expr_reads(&k.value, info);
            }
        }
        ExprType::BinOp(b) => {
            count_expr_reads(&b.left, info);
            count_expr_reads(&b.right, info);
        }
        ExprType::BoolOp(b) => {
            for v in &b.values {
                count_expr_reads(v, info);
            }
        }
        ExprType::Compare(c) => {
            count_expr_reads(&c.left, info);
            for r in &c.comparators {
                count_expr_reads(r, info);
            }
        }
        ExprType::UnaryOp(u) => count_expr_reads(&u.operand, info),
        ExprType::IfExp(i) => {
            count_expr_reads(&i.test, info);
            count_expr_reads(&i.body, info);
            count_expr_reads(&i.orelse, info);
        }
        ExprType::Subscript(s) => {
            count_expr_reads(&s.value, info);
            match &s.kind {
                crate::SubscriptKind::Index(i) => count_expr_reads(i, info),
                crate::SubscriptKind::Slice { lower, upper, step } => {
                    for b in [lower, upper, step].into_iter().flatten() {
                        count_expr_reads(b, info);
                    }
                }
            }
        }
        ExprType::Tuple(t) => {
            for e in &t.elts {
                count_expr_reads(e, info);
            }
        }
        ExprType::List(l) => {
            for e in l {
                count_expr_reads(e, info);
            }
        }
        ExprType::Dict(d) => {
            for (k, v) in d.keys.iter().zip(d.values.iter()) {
                if let Some(k) = k {
                    count_expr_reads(k, info);
                }
                count_expr_reads(v, info);
            }
        }
        ExprType::Set(s) => {
            for e in &s.elts {
                count_expr_reads(e, info);
            }
        }
        ExprType::ListComp(lc) => {
            count_expr_reads(&lc.elt, info);
            for g in &lc.generators {
                count_expr_reads(&g.iter, info);
            }
        }
        ExprType::Starred(s) => count_expr_reads(&s.value, info),
        ExprType::JoinedStr(js) => {
            for v in &js.values {
                count_expr_reads(v, info);
            }
        }
        ExprType::FormattedValue(fv) => count_expr_reads(&fv.value, info),
        ExprType::Lambda(l) => count_expr_reads(&l.body, info),
        ExprType::Yield(y) => {
            if let Some(v) = &y.value {
                count_expr_reads(v, info);
            }
        }
        ExprType::YieldFrom(yf) => count_expr_reads(&yf.value, info),
        ExprType::Await(a) => count_expr_reads(&a.value, info),
        _ => {}
    }
}

/// Reads inside an assignment TARGET: `xs[i] = v` reads `xs` and `i` but
/// not `xs` as a whole-value read (the store mutates in place).
fn count_target_reads(target: &ExprType, info: &mut FunctionTypeInfo) {
    match target {
        ExprType::Name(_) => {} // the store target itself: no read
        ExprType::Subscript(s) => {
            count_expr_reads(&s.value, info);
            match &s.kind {
                crate::SubscriptKind::Index(i) => count_expr_reads(i, info),
                crate::SubscriptKind::Slice { lower, upper, step } => {
                    for b in [lower, upper, step].into_iter().flatten() {
                        count_expr_reads(b, info);
                    }
                }
            }
        }
        ExprType::Attribute(a) => count_expr_reads(&a.value, info),
        ExprType::Tuple(t) => {
            for e in &t.elts {
                count_target_reads(e, info);
            }
        }
        _ => count_expr_reads(target, info),
    }
}

/// Whether a name is referenced anywhere inside a statement list (used for
/// unused loop-index detection).
pub fn name_referenced_in(body: &[Statement], name: &str) -> bool {
    body.iter().any(|s| statement_references(s, name))
}

fn statement_references(stmt: &Statement, name: &str) -> bool {
    match &stmt.statement {
        StatementType::Assign(a) => {
            expr_references(&a.value, name) || a.targets.iter().any(|t| target_references(t, name))
        }
        StatementType::AugAssign(a) => {
            expr_references(&a.target, name) || expr_references(&a.value, name)
        }
        StatementType::Expr(e) => expr_references(&e.value, name),
        StatementType::Return(r) => r
            .as_ref()
            .map(|e| expr_references(&e.value, name))
            .unwrap_or(false),
        // Assert and raise read their expressions too: a loop index used
        // only there is still a real reference (Devin review on #103).
        StatementType::Assert { test, msg } => {
            expr_references(test, name)
                || msg
                    .as_ref()
                    .map(|m| expr_references(m, name))
                    .unwrap_or(false)
        }
        StatementType::Raise(r) => {
            r.exc
                .as_ref()
                .map(|e| expr_references(e, name))
                .unwrap_or(false)
                || r.cause
                    .as_ref()
                    .map(|e| expr_references(e, name))
                    .unwrap_or(false)
        }
        StatementType::If(s) => {
            expr_references(&s.test, name)
                || s.body.iter().any(|b| statement_references(b, name))
                || s.orelse.iter().any(|b| statement_references(b, name))
        }
        StatementType::While(s) => {
            expr_references(&s.test, name)
                || s.body.iter().any(|b| statement_references(b, name))
                || s.orelse.iter().any(|b| statement_references(b, name))
        }
        StatementType::For(s) => {
            expr_references(&s.iter, name)
                || target_references(&s.target, name)
                || s.body.iter().any(|b| statement_references(b, name))
                || s.orelse.iter().any(|b| statement_references(b, name))
        }
        StatementType::With(s) => {
            s.items
                .iter()
                .any(|i| expr_references(&i.context_expr, name))
                || s.body.iter().any(|b| statement_references(b, name))
        }
        StatementType::Try(s) => {
            s.body.iter().any(|b| statement_references(b, name))
                || s.handlers.iter().any(|h| {
                    h.exception_type
                        .as_ref()
                        .map(|t| expr_references(t, name))
                        .unwrap_or(false)
                        || h.body.iter().any(|b| statement_references(b, name))
                })
                || s.orelse.iter().any(|b| statement_references(b, name))
                || s.finalbody.iter().any(|b| statement_references(b, name))
        }
        _ => false,
    }
}

fn target_references(target: &ExprType, name: &str) -> bool {
    match target {
        ExprType::Name(n) => n.id == name,
        ExprType::Subscript(s) => {
            expr_references(&s.value, name)
                || match &s.kind {
                    crate::SubscriptKind::Index(i) => expr_references(i, name),
                    crate::SubscriptKind::Slice { lower, upper, step } => [lower, upper, step]
                        .into_iter()
                        .flatten()
                        .any(|b| expr_references(b, name)),
                }
        }
        ExprType::Tuple(t) => t.elts.iter().any(|e| target_references(e, name)),
        _ => expr_references(target, name),
    }
}

pub(crate) fn expr_references(expr: &ExprType, name: &str) -> bool {
    match expr {
        ExprType::Name(n) => n.id == name,
        ExprType::Attribute(a) => expr_references(&a.value, name),
        ExprType::Call(c) => {
            expr_references(&c.func, name)
                || c.args.iter().any(|a| expr_references(a, name))
                || c.keywords.iter().any(|k| expr_references(&k.value, name))
        }
        ExprType::BinOp(b) => expr_references(&b.left, name) || expr_references(&b.right, name),
        ExprType::BoolOp(b) => b.values.iter().any(|v| expr_references(v, name)),
        ExprType::Compare(c) => {
            expr_references(&c.left, name) || c.comparators.iter().any(|r| expr_references(r, name))
        }
        ExprType::UnaryOp(u) => expr_references(&u.operand, name),
        ExprType::IfExp(i) => {
            expr_references(&i.test, name)
                || expr_references(&i.body, name)
                || expr_references(&i.orelse, name)
        }
        ExprType::Subscript(s) => {
            expr_references(&s.value, name)
                || match &s.kind {
                    crate::SubscriptKind::Index(i) => expr_references(i, name),
                    crate::SubscriptKind::Slice { lower, upper, step } => [lower, upper, step]
                        .into_iter()
                        .flatten()
                        .any(|b| expr_references(b, name)),
                }
        }
        ExprType::Tuple(t) => t.elts.iter().any(|e| expr_references(e, name)),
        ExprType::List(l) => l.iter().any(|e| expr_references(e, name)),
        ExprType::Dict(d) => d.keys.iter().zip(d.values.iter()).any(|(k, v)| {
            k.as_ref()
                .map(|k| expr_references(k, name))
                .unwrap_or(false)
                || expr_references(v, name)
        }),
        ExprType::Set(s) => s.elts.iter().any(|e| expr_references(e, name)),
        ExprType::ListComp(lc) => {
            expr_references(&lc.elt, name)
                || lc.generators.iter().any(|g| expr_references(&g.iter, name))
        }
        ExprType::JoinedStr(js) => js.values.iter().any(|v| expr_references(v, name)),
        ExprType::FormattedValue(fv) => expr_references(&fv.value, name),
        ExprType::Lambda(l) => expr_references(&l.body, name),
        ExprType::Starred(s) => expr_references(&s.value, name),
        // `yield x` / `yield from xs` / `await f(x)` — the yielded or
        // awaited expression is a real reference (a loop index used only
        // in a yield body must not lower to `_`; response.py's `for x in
        // chunks[1:-1]: yield x + b"\n"`).
        ExprType::Yield(y) => y
            .value
            .as_ref()
            .map(|v| expr_references(v, name))
            .unwrap_or(false),
        ExprType::YieldFrom(yf) => expr_references(&yf.value, name),
        ExprType::Await(a) => expr_references(&a.value, name),
        _ => false,
    }
}
