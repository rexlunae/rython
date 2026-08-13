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
use std::collections::HashMap;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, Statement, StatementType,
    SymbolTableNode, SymbolTableScopes,
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
    /// `usize` — `len()`, `count()`
    Usize,
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
    NdArray,
    /// a borrow (`&[T]`, `&str`) from iteration or indexing
    Borrowed(Box<TypeInfo>),
    /// anything we cannot or choose not to track
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
            TypeInfo::Usize => quote!(usize),
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
            TypeInfo::Usize => "usize".into(),
            TypeInfo::Bytes => "bytes".into(),
            TypeInfo::Vec(_) => "list".into(),
            TypeInfo::Dict(_, _) => "dict".into(),
            TypeInfo::Tuple(_) => "tuple".into(),
            TypeInfo::Option(_) => "Optional".into(),
            TypeInfo::Range => "range".into(),
            TypeInfo::NdArray => "ndarray".into(),
            TypeInfo::Borrowed(_) => "borrowed".into(),
            TypeInfo::PyObject => "unknown".into(),
        }
    }
}

/// Wrap already-rendered tokens in the conversion that takes a value of
/// `from` to a value of `to`. Returns `None` when no conversion is needed
/// or possible.
pub fn coerce_tokens(
    tokens: TokenStream,
    from: &TypeInfo,
    to: &TypeInfo,
) -> Option<TokenStream> {
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
        // usize → i64: len() results into index/range positions. Loud on
        // the (theoretical) overflow rather than wrapping.
        (TypeInfo::Usize, TypeInfo::Int) => {
            Some(quote!((#tokens).try_into().unwrap()))
        }
        // i64 → f64: only in all-numeric unification (mixed int/float
        // literal lists). Python's int is arbitrary precision, so this is
        // lossy above 2^53; accepted for numeric lists because it is the
        // only way to compile them at all, and the alternative (rustc
        // error) is no more informative.
        (TypeInfo::Int, TypeInfo::Float) => Some(quote!((#tokens) as f64)),
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
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => TypeInfo::Int,
            Some(litrs::Literal::Float(_)) => TypeInfo::Float,
            Some(litrs::Literal::Bool(_)) => TypeInfo::Bool,
            Some(litrs::Literal::String(_)) => TypeInfo::StrRef,
            Some(litrs::Literal::Byte(_)) => TypeInfo::Bytes,
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
            crate::Ops::USub | crate::Ops::UAdd => {
                infer_type(&u.operand, options, symbols)
            }
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
            ExprType::Name(n) => match n.id.as_str() {
                "len" | "count" => TypeInfo::Usize,
                "range" => TypeInfo::Range,
                "str" | "repr" | "format" => TypeInfo::String,
                "int" => TypeInfo::Int,
                "float" => TypeInfo::Float,
                "bool" => TypeInfo::Bool,
                "list" => TypeInfo::Vec(Box::new(TypeInfo::PyObject)),
                "dict" => TypeInfo::Dict(
                    Box::new(TypeInfo::PyObject),
                    Box::new(TypeInfo::PyObject),
                ),
                _ => TypeInfo::PyObject,
            },
            ExprType::Attribute(attr) => {
                // numpy functions produce arrays (`np.sum`, `numpy.mean`).
                let on_numpy = matches!(
                    attr.value.as_ref(),
                    ExprType::Name(n) if n.id == "np" || n.id == "numpy"
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
        ExprType::DictComp(_) => TypeInfo::Dict(
            Box::new(TypeInfo::PyObject),
            Box::new(TypeInfo::PyObject),
        ),
        ExprType::Starred(s) => infer_type(&s.value, options, symbols),
        _ => TypeInfo::PyObject,
    }
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
        (TypeInfo::Tuple(x), TypeInfo::Tuple(y)) if x.len() == y.len() => {
            TypeInfo::Tuple(
                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| unify(a.clone(), b.clone()))
                    .collect(),
            )
        }
        _ => {
            if is_numeric(&a) && is_numeric(&b) {
                return TypeInfo::Float;
            }
            if is_stringy(&a) && is_stringy(&b) {
                return TypeInfo::String;
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
    let tokens = expr
        .clone()
        .to_rust(ctx, options.clone(), symbols.clone())?;
    let Some(expected) = expected else {
        return Ok(tokens);
    };
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
            if !t.is_copy() && !matches!(t, TypeInfo::PyObject) {
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
            if !t.is_copy() && !matches!(t, TypeInfo::PyObject) {
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
    match ann {
        ExprType::Name(n) => match n.id.as_str() {
            "int" => Some(TypeInfo::Int),
            "float" => Some(TypeInfo::Float),
            "bool" => Some(TypeInfo::Bool),
            "str" => Some(TypeInfo::String),
            "bytes" => Some(TypeInfo::Bytes),
            _ => None,
        },
        ExprType::Subscript(sub) => match sub.value.as_ref() {
            ExprType::Name(n) => match n.id.as_str() {
                "list" | "List" => {
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        Some(TypeInfo::Vec(Box::new(annotation_type_info(elt)?)))
                    } else {
                        None
                    }
                }
                "Optional" => {
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        Some(TypeInfo::Option(Box::new(annotation_type_info(elt)?)))
                    } else {
                        None
                    }
                }
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
            _ => None,
        },
        _ => None,
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
    /// Names assigned an empty `[]`/`{}` literal whose element type was
    /// pinned by a later use; maps to the pinned container type.
    pub empty_pinned: HashMap<String, TypeInfo>,
}

/// Walk a statement list, counting reads and collecting `name = expr`
/// assignments with inferable types, then pin empty-container types from
/// later use.
pub fn analyze_function_types(body: &[Statement]) -> FunctionTypeInfo {
    let mut info = FunctionTypeInfo::default();
    for stmt in body {
        analyze_statement_types(stmt, &mut info);
    }
    pin_empty_containers(body, &mut info);
    info
}

fn analyze_statement_types(stmt: &Statement, info: &mut FunctionTypeInfo) {
    match &stmt.statement {
        StatementType::Assign(assign) => {
            // Record `name = <inferable expr>` (syntactic only, unless an
            // annotation pins it — an annotated assignment's annotation
            // wins, so `xs: list[float] = []` pins the empty literal).
            if let [ExprType::Name(name)] = assign.targets.as_slice() {
                let mut t = match assign
                    .annotation
                    .as_ref()
                    .and_then(crate::annotation_type_info)
                {
                    // An annotation pins the type outright.
                    Some(ann) => ann,
                    // Unparseable annotation: fall back to the value's
                    // syntactic type (still pinable by later use).
                    None => syntactic_type(&assign.value),
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
                if !matches!(t, TypeInfo::PyObject) {
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
                analyze_statement_types(b, info);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info);
            }
        }
        StatementType::While(s) => {
            count_expr_reads(&s.test, info);
            for b in &s.body {
                analyze_statement_types(b, info);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info);
            }
        }
        StatementType::For(s) => {
            count_expr_reads(&s.iter, info);
            count_target_reads(&s.target, info);
            for b in &s.body {
                analyze_statement_types(b, info);
            }
            for b in &s.orelse {
                analyze_statement_types(b, info);
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
                analyze_statement_types(b, info);
            }
        }
        StatementType::Try(s) => {
            for b in &s.body {
                analyze_statement_types(b, info);
            }
            for handler in &s.handlers {
                if let Some(t) = &handler.exception_type {
                    count_expr_reads(t, info);
                }
                if let Some(name) = &handler.name {
                    info.use_counts.remove(name); // bound by except, not read
                }
                for b in &handler.body {
                    analyze_statement_types(b, info);
                }
            }
            for b in &s.orelse {
                analyze_statement_types(b, info);
            }
            for b in &s.finalbody {
                analyze_statement_types(b, info);
            }
        }
        StatementType::FunctionDef(_) => {
            // A nested function is a new scope: its locals do not belong to
            // the enclosing function's type analysis. (Captured reads of
            // outer names are rare and would only tune clone-on-reuse.)
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
fn syntactic_type(expr: &ExprType) -> TypeInfo {
    match expr {
        ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => TypeInfo::Int,
            Some(litrs::Literal::Float(_)) => TypeInfo::Float,
            Some(litrs::Literal::Bool(_)) => TypeInfo::Bool,
            Some(litrs::Literal::String(_)) => TypeInfo::StrRef,
            Some(litrs::Literal::Byte(_)) => TypeInfo::Bytes,
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
        _ => TypeInfo::PyObject,
    }
}

/// Pin the element/key types of empty containers from a later use. Called
/// with the full statement list AFTER `name_types` is populated so that
/// `append(d[k])` can resolve `d`'s type too.
pub fn pin_empty_containers(body: &[Statement], info: &mut FunctionTypeInfo) {
    let mut suggested: HashMap<String, TypeInfo> = HashMap::new();
    for stmt in body {
        collect_use_suggestions(stmt, info, &mut suggested);
    }
    for (name, t) in suggested {
        if info.empty_pinned.contains_key(&name) {
            info.name_types.insert(name.clone(), t.clone());
            info.empty_pinned.insert(name, t);
        }
    }
}

fn collect_use_suggestions(
    stmt: &Statement,
    info: &FunctionTypeInfo,
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
                            let t = resolve_type(arg, info);
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), t.clone()))
                                .or_insert(TypeInfo::Vec(Box::new(t)));
                        }
                    }
                    "insert" => {
                        if let Some(arg) = call.args.get(1) {
                            let t = resolve_type(arg, info);
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
                    let v = resolve_type(&assign.value, info);
                    if let crate::SubscriptKind::Index(idx) = &sub.kind {
                        // Keys normalize to String, matching dict literals
                        // and `dict[str, V]` annotations.
                        let k = match resolve_type(idx, info) {
                            TypeInfo::StrRef => TypeInfo::String,
                            other => other,
                        };
                        let ty = TypeInfo::Dict(Box::new(k), Box::new(v));
                        out.entry(recv.id.clone())
                            .and_modify(|e| *e = unify(e.clone(), ty.clone()))
                            .or_insert(ty);
                    }
                }
            }
        }
        StatementType::Expr(e) => {
            if let ExprType::Call(call) = &e.value
                && let ExprType::Attribute(attr) = call.func.as_ref()
                && let ExprType::Name(recv) = attr.value.as_ref()
                && info.empty_pinned.contains_key(&recv.id)
            {
                match attr.attr.as_str() {
                    // d.get(k) read pins the key type.
                    "get" => {
                        if let Some(arg) = call.args.first() {
                            let k = match resolve_type(arg, info) {
                                TypeInfo::StrRef => TypeInfo::String,
                                other => other,
                            };
                            let t = TypeInfo::Dict(
                                Box::new(k),
                                Box::new(TypeInfo::PyObject),
                            );
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), t.clone()))
                                .or_insert(t);
                        }
                    }
                    // Expression-statement appends/inserts:
                    // `xs.append(v)` pins xs's element type.
                    "append" | "push" => {
                        if let Some(arg) = call.args.first() {
                            let t = resolve_type(arg, info);
                            let suggestion = TypeInfo::Vec(Box::new(t));
                            out.entry(recv.id.clone())
                                .and_modify(|e| *e = unify(e.clone(), suggestion.clone()))
                                .or_insert(suggestion);
                        }
                    }
                    // `xs.extend(ys)` pins xs's element type to ys's.
                    "extend" => {
                        if let Some(arg) = call.args.first() {
                            let t = match resolve_type(arg, info) {
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
                            let t = resolve_type(arg, info);
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
                collect_use_suggestions(b, info, out);
            }
            for b in &s.orelse {
                collect_use_suggestions(b, info, out);
            }
        }
        StatementType::While(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, out);
            }
        }
        StatementType::For(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, out);
            }
        }
        StatementType::With(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, out);
            }
        }
        StatementType::Try(s) => {
            for b in &s.body {
                collect_use_suggestions(b, info, out);
            }
            for h in &s.handlers {
                for b in &h.body {
                    collect_use_suggestions(b, info, out);
                }
            }
        }
        _ => {}
    }
}

/// Resolve a name's type through the name_types map (falling back to
/// syntactic inference for expressions that don't need the map).
fn resolve_type(expr: &ExprType, info: &FunctionTypeInfo) -> TypeInfo {
    match expr {
        ExprType::Name(n) => info
            .name_types
            .get(&n.id)
            .cloned()
            .unwrap_or_else(|| syntactic_type(expr)),
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
            expr_references(&a.value, name)
                || a.targets.iter().any(|t| target_references(t, name))
        }
        StatementType::AugAssign(a) => {
            expr_references(&a.target, name) || expr_references(&a.value, name)
        }
        StatementType::Expr(e) => expr_references(&e.value, name),
        StatementType::Return(r) => {
            r.as_ref().map(|e| expr_references(&e.value, name)).unwrap_or(false)
        }
        // Assert and raise read their expressions too: a loop index used
        // only there is still a real reference (Devin review on #103).
        StatementType::Assert { test, msg } => {
            expr_references(test, name)
                || msg.as_ref().map(|m| expr_references(m, name)).unwrap_or(false)
        }
        StatementType::Raise(r) => {
            r.exc.as_ref().map(|e| expr_references(e, name)).unwrap_or(false)
                || r.cause.as_ref().map(|e| expr_references(e, name)).unwrap_or(false)
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
            s.items.iter().any(|i| expr_references(&i.context_expr, name))
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
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        [lower, upper, step]
                            .into_iter()
                            .flatten()
                            .any(|b| expr_references(b, name))
                    }
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
                || c.keywords
                    .iter()
                    .any(|k| expr_references(&k.value, name))
        }
        ExprType::BinOp(b) => {
            expr_references(&b.left, name) || expr_references(&b.right, name)
        }
        ExprType::BoolOp(b) => b.values.iter().any(|v| expr_references(v, name)),
        ExprType::Compare(c) => {
            expr_references(&c.left, name)
                || c.comparators.iter().any(|r| expr_references(r, name))
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
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        [lower, upper, step]
                            .into_iter()
                            .flatten()
                            .any(|b| expr_references(b, name))
                    }
                }
        }
        ExprType::Tuple(t) => t.elts.iter().any(|e| expr_references(e, name)),
        ExprType::List(l) => l.iter().any(|e| expr_references(e, name)),
        ExprType::Dict(d) => d
            .keys
            .iter()
            .zip(d.values.iter())
            .any(|(k, v)| {
                k.as_ref().map(|k| expr_references(k, name)).unwrap_or(false)
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
        _ => false,
    }
}
