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
    CodeGen, CodeGenContext, ExprType, PythonOptions, Statement, StatementType,
    SymbolTableNode, SymbolTableScopes,
};

/// The Rust types codegen produces for Python expressions, at the
/// granularity needed to insert coercions. `PartialEq` is manual:
/// `proc_macro2::TokenStream` (the [`TypeInfo::Custom`] payload) has no
/// structural equality, so that variant compares by its rendered spelling.
#[derive(Clone, Debug)]
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
    /// `std::collections::HashSet<T>` — `set[T]` / `frozenset[T]`
    /// annotations: set literals lower to HashSet, so an annotated set is
    /// that type, not a Vec (the two resolvers disagreed here; the
    /// generated structs are the arbiter — urllib3's PoolKey fields are
    /// `Option<HashSet<(String, String)>>`).
    HashSet(Box<TypeInfo>),
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
    /// `threading::<Name>` — a threading-module runtime handle annotation
    /// (`ready: threading.Event`, `lock: threading.Lock`): a real shared
    /// handle in the runtime, not a boxed PyValue.
    Threading(crate::ThreadingType),
    /// `socket::Socket` — the `socket.socket` annotation.
    Socket,
    /// `type[X]` / `Type[X]` — a CLASS value. rython cannot hold classes
    /// as values (the callables-as-data divergence): the tolerated opaque
    /// type is `Option<()>`.
    ClassValue,
    /// A runtime type with no structural meaning to the coercion
    /// machinery — the exact Rust type, rendered verbatim
    /// (`datetime::timedelta` — the datetime.timedelta struct). The
    /// field/coercion layers treat it like any other TypeInfo; equality
    /// is structural on the tokens (identical spellings compare equal).
    Custom(TokenStream),
    PyObject,
}

impl PartialEq for TypeInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TypeInfo::Custom(a), TypeInfo::Custom(b)) => {
                a.to_string() == b.to_string()
            }
            _ => {
                // Structural comparison of the remaining variants via
                // Debug (every variant except Custom is structurally
                // comparable; Debug renders them faithfully).
                format!("{:?}", self) == format!("{:?}", other)
            }
        }
    }
}

/// Whether a type mentions an OWNED heap value anywhere inside — PyValue,
/// String/&str, or bytes — the structural twin of the old
/// `tokens.contains("PyValue" | "String" | "Vec < u8 >")` sniffing used by
/// the clone-safe field-read rule: such a field clones out of `&self`.
pub(crate) fn type_mentions_heap(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::PyValue
        | TypeInfo::PyValueMember(_)
        | TypeInfo::String
        | TypeInfo::StrRef
        | TypeInfo::Bytes
        | TypeInfo::Custom(_) => true,
        TypeInfo::Vec(inner)
        | TypeInfo::Option(inner)
        | TypeInfo::HashSet(inner)
        | TypeInfo::Borrowed(inner) => type_mentions_heap(inner),
        TypeInfo::Dict(k, v) => type_mentions_heap(k) || type_mentions_heap(v),
        TypeInfo::Tuple(ts) => ts.iter().any(type_mentions_heap),
        _ => false,
    }
}

/// Whether a type mentions an UNKNOWN (`PyObject`) element anywhere inside
/// (`Dict(PyObject, PyObject)` — an empty-container literal typed without
/// an anchor). Such a type renders `PyDict<_, _>` / `Vec<_>`, which is
/// E0121 in an item signature — field inference must fall back rather
/// than emit it (round 62).
pub(crate) fn type_mentions_pyobject(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::PyObject => true,
        TypeInfo::Vec(inner)
        | TypeInfo::Option(inner)
        | TypeInfo::HashSet(inner)
        | TypeInfo::Borrowed(inner) => type_mentions_pyobject(inner),
        TypeInfo::Dict(k, v) => type_mentions_pyobject(k) || type_mentions_pyobject(v),
        TypeInfo::Tuple(ts) => ts.iter().any(type_mentions_pyobject),
        _ => false,
    }
}

/// Whether a type mentions the boxed heterogeneous value ANYWHERE inside
/// (`Vec<PyValue>`, `PyDict<String, PyValue>`, `Option<PyValue>`, ...).
/// The structural twin of the old `tokens.to_string().contains("PyValue")`
/// sniffing — a boxed-containing field types the whole value as dynamic
/// (the boxed-receiver drop, the poisoned-BinOp rule, the clone-safe
/// field rule).
pub(crate) fn type_contains_pyvalue(t: &TypeInfo) -> bool {
    match t {
        TypeInfo::PyValue | TypeInfo::PyValueMember(_) => true,
        TypeInfo::Vec(inner)
        | TypeInfo::Option(inner)
        | TypeInfo::HashSet(inner)
        | TypeInfo::Borrowed(inner) => type_contains_pyvalue(inner),
        TypeInfo::Dict(k, v) => type_contains_pyvalue(k) || type_contains_pyvalue(v),
        TypeInfo::Tuple(ts) => ts.iter().any(type_contains_pyvalue),
        _ => false,
    }
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
            TypeInfo::HashSet(inner) => {
                let t = inner.to_rust_type();
                quote!(std::collections::HashSet<#t>)
            }
            TypeInfo::Dict(k, v) => {
                let k = k.to_rust_type();
                let v = v.to_rust_type();
                quote!(PyDict<#k, #v>)
            }
            TypeInfo::Tuple(ts) => {
                // A Rust 1-tuple needs the trailing comma (`(i64,)`); the
                // naive repetition renders `(i64)`, which is just i64.
                if ts.len() == 1 {
                    let only = ts[0].to_rust_type();
                    return quote!((#only,));
                }
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
            TypeInfo::Threading(t) => t.rust_path(),
            TypeInfo::Socket => quote!(socket::Socket),
            TypeInfo::ClassValue => quote!(Option<()>),
            TypeInfo::Custom(t) => t.clone(),
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
            TypeInfo::HashSet(_) => "set".into(),
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
            TypeInfo::Threading(t) => t.name().into(),
            TypeInfo::Socket => "socket".into(),
            TypeInfo::ClassValue => "type".into(),
            TypeInfo::Custom(_) => "custom".into(),
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
        // i64 → f64: only in all-numeric unification (mixed int/float
        // literal lists). Python's int is arbitrary precision, so this is
        // lossy above 2^53; accepted for numeric lists because it is the
        // only way to compile them at all, and the alternative (rustc
        // error) is no more informative.
        (TypeInfo::Int, TypeInfo::Float) => Some(quote!((#tokens) as f64)),
        // Option<PyValue> → PyValue (issue #137 round 27): reading an
        // OPTIONAL boxed field where the bare boxed value is wanted.
        // Python's None IS `PyValue::None_`, so the empty case is not a
        // loss of information — it is the same value spelled the other
        // way. Placed before the general `_ → PyValue` arm, which would
        // otherwise try `PyValue::from(Option<PyValue>)` and find no impl.
        (TypeInfo::Option(inner), TypeInfo::PyValue)
            if matches!(**inner, TypeInfo::PyValue) =>
        {
            Some(quote!((#tokens).unwrap_or(stdpython::PyValue::None_)))
        }
        // Option<T> → PyValue: the same, for a typed optional — the inner
        // value boxes, and the empty case is Python's None.
        (TypeInfo::Option(inner), TypeInfo::PyValue) => {
            let inner_boxed = coerce_tokens(quote!(__rython_v), inner, &TypeInfo::PyValue)
                .unwrap_or_else(|| quote!(PyValue::from(__rython_v)));
            Some(quote!(
                match (#tokens) {
                    Some(__rython_v) => #inner_boxed,
                    None => stdpython::PyValue::None_,
                }
            ))
        }
        // Round 81: an OPTION-wrapped boxed value into an Option of a
        // CONCRETE member (`Option<PyValue> → Option<i64>` — a boxed
        // `cert_reqs` call result stored through an optional slot): map
        // the conversion over the Option — None passes through (Python's
        // None IS the empty case), Some converts loudly. Must come BEFORE
        // the generic `T → Option` arm below: a `from_ty` of
        // `Option<PyValue>` satisfies that arm's `from_ty != PyValue`
        // guard, and its recursion (`Option<PyValue> → i64`) finds no
        // conversion, returning None and dropping the map.
        (TypeInfo::Option(from_inner), TypeInfo::Option(to_inner))
            if matches!(**from_inner, TypeInfo::PyValue)
                && matches!(
                    **to_inner,
                    TypeInfo::Int
                        | TypeInfo::Float
                        | TypeInfo::Bool
                        | TypeInfo::String
                        | TypeInfo::Bytes
                ) =>
        {
            let coerced =
                coerce_tokens(quote!(__rython_v), &TypeInfo::PyValue, to_inner)?;
            Some(quote!((#tokens).map(|__rython_v| #coerced)))
        }
        // Round 83 (the generics directive): an OPTION-typed value into an
        // OPTION-typed slot of a DIFFERENT inner type (`Option<i64> →
        // Option<f64>`, `Option<A> → Option<B>` — urllib3's urlopen
        // headers): map the inner conversion — None passes through
        // (Python's None IS the empty case), Some converts. Must come
        // BEFORE the generic `T → Option` arm below, which would
        // otherwise Some-wrap the recursion and turn the empty case into
        // a panic instead of a pass-through None.
        (TypeInfo::Option(from_inner), TypeInfo::Option(to_inner))
            if !matches!(**from_inner, TypeInfo::PyValue) =>
        {
            let coerced = coerce_tokens(quote!(__rython_v), from_inner, to_inner)?;
            Some(quote!((#tokens).map(|__rython_v| #coerced)))
        }
        // T → Option<U> (issue #137 round 27): a concrete value stored
        // into an OPTIONAL slot. Round 23 gave fields an `Option<T>` type
        // when a None store joined a typed one — the declare-then-fill
        // idiom — but nothing taught the stores to wrap, so every one of
        // them landed as a bare `T` against an `Option<T>` target. Wrapping
        // is exact: Python's value is present, so `Some` is what it means.
        (from_ty, TypeInfo::Option(inner)) if from_ty != &TypeInfo::PyValue => {
            let coerced = coerce_tokens(tokens, from_ty, inner)?;
            Some(quote!(Some(#coerced)))
        }
        // Round 81 (the generics directive): a BOXED value into an
        // OPTION-typed slot (`Some(resolve_cert_reqs(...)?)` against
        // `Option<i64>` — urllib3's create_urllib3_context): the inner
        // converts via the reverse From<PyValue> impls (loud TypeError
        // panic on a wrong member — Python fails at use, rython at the
        // conversion), and the present value wraps in Some — Python's
        // optional slot holds the value. A `PyValue | None` union with a
        // None default is the None_ member, never an Option, so the
        // empty case does not arise here.
        (TypeInfo::PyValue, TypeInfo::Option(inner)) => {
            let coerced = coerce_tokens(tokens, &TypeInfo::PyValue, inner)?;
            Some(quote!(Some(#coerced)))
        }
        // Round 81 (the generics directive): a boxed value into a CONCRETE
        // typed slot — `PyValue → i64/f64/bool/String/Vec<u8>` through the
        // reverse `From<PyValue>` impls (round 80). The value was boxed
        // from a concrete member by the None-mixing inference; the
        // conversion recovers it, and a wrong member is a LOUD TypeError
        // panic (`value_member_panic`): Python fails at use, rython at the
        // conversion — never a silent placeholder. rustc's own suggestions
        // at these sites are exactly this `.into()` (urllib3's
        // py_set_index(key, ...) String keys, `port: i64` params,
        // `is_verified: bool` field stores).
        (TypeInfo::PyValue, TypeInfo::Int)
        | (TypeInfo::PyValue, TypeInfo::Float)
        | (TypeInfo::PyValue, TypeInfo::Bool)
        | (TypeInfo::PyValue, TypeInfo::String)
        | (TypeInfo::PyValue, TypeInfo::Bytes) => {
            Some(quote!((#tokens).into()))
        }
        // Round 83 (the generics directive): an OPTION-typed value into a
        // CONCRETE slot (`Option<Vec<u8>> → Vec<u8>` — a `bytes | None`
        // field read passed to a `bytes`-annotated parameter — urllib3's
        // `self.decompress(self._data)` where DeflateDecoder's `_data`
        // widens to Option when a None store joins): the inner converts,
        // and the None case is a LOUD panic — Python fails at use on a
        // None value, rython at the conversion (§12.2), mirroring the
        // return-site `Option<PyValue>` map (round 81). Excludes the
        // PyValue targets (the Option→PyValue arms above keep the empty
        // case as Python's None), StrOrBytes (a union slot), and Option
        // targets (an Option→Option join passes None through, not a
        // panic — that direction stays a rustc error until the `.map`
        // conversion is added).
        (TypeInfo::Option(inner), to_ty)
            if !matches!(
                to_ty,
                TypeInfo::PyValue | TypeInfo::StrOrBytes | TypeInfo::Option(_)
            ) =>
        {
            let coerced = coerce_tokens(quote!(__rython_v), inner, to_ty)?;
            Some(quote!(
                match (#tokens) {
                    Some(__rython_v) => #coerced,
                    None => panic!(
                        "rython: an optional value was None where a concrete value was required (Python would fail at use, rython at the conversion)"
                    ),
                }
            ))
        }
        // Anything → PyValue (issue #121): a value stored into a boxed
        // union / Any slot wraps in PyValue::from (None via From<()>).
        (_, TypeInfo::PyValue) => Some(quote!(PyValue::from((#tokens)))),
        // Anything → StrOrBytes (issue #121): the str | bytes union's
        // heterogeneous slot converts via its From impls (&str, String,
        // &[u8], Vec<u8>).
        (_, TypeInfo::StrOrBytes) => {
            Some(quote!(stdpython::StrOrBytes::from((#tokens))))
        }
        _ => None,
    }
}

/// Is this expression an `NdArray`?
///
/// `infer_type` consults the per-function `name_types` map before the
/// symbol table, and that map does not record numpy locals, so a plain
/// `a = np.array(...)` name inferred as something else and the numpy
/// attribute/method lowerings never fired (issues #197, #204). This falls
/// back to the recorded assignment, which is where a numpy local's type
/// actually lives.
pub fn is_ndarray_expr(
    expr: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    if matches!(infer_type(expr, options, symbols), TypeInfo::NdArray) {
        return true;
    }
    match expr {
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::Assign { value, .. }) => {
                matches!(infer_type(&value, options, symbols), TypeInfo::NdArray)
            }
            _ => false,
        },
        _ => false,
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
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => {
                TypeInfo::Bytes
            }
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
            // 3. A MUTABLE module global: its static's value type is the
            // name's type — a Boxed static is the boxed PyValue, so a
            // method call on the global drops like any boxed receiver
            // (`_CACERT_CTX.__exit__(...)` — certifi's atexit cleanup).
            // Checked BEFORE the symbol-table Assign below: the global's
            // recorded initializer (`_CACERT_CTX = None`) is not its type.
            if let Some(kind) = options.mutable_statics.get(&n.id) {
                match kind {
                    crate::MutableGlobalKind::Boxed
                    | crate::MutableGlobalKind::Computed { boxed: true } => {
                        return TypeInfo::PyValue;
                    }
                    crate::MutableGlobalKind::Str => return TypeInfo::String,
                    _ => {}
                }
            }
            // 4. The symbol table's recorded assignment.
            if let Some(SymbolTableNode::Assign { value, .. }) = symbols.get(&n.id) {
                return infer_type(value, options, symbols);
            }
            // A CLASS NAME read as a VALUE (`[ChecksumError]`,
            // `EXCEPTION_MAP['k']` — botocore's retryhandler): classes as
            // values lower to their NAME STRINGS — the exception model is
            // string-tagged, so the name is the class object's only
            // runtime-relevant data (round 33 design). Direct construction
            // and isinstance/except never read the name as a value, so
            // they are unaffected. A same-module ClassDef, an alias of
            // one, or an IMPORTED class (resolved through its defining
            // module) all qualify; an imported FUNCTION does not.
            if matches!(symbols.get(&n.id), Some(SymbolTableNode::ClassDef(_)))
                || matches!(symbols.get(&n.id), Some(SymbolTableNode::Alias(c))
                    if matches!(symbols.get(c), Some(SymbolTableNode::ClassDef(_))))
                || (matches!(symbols.get(&n.id), Some(SymbolTableNode::ImportFrom(_)))
                    && crate::resolve_class_referenced(&n.id, symbols, options).is_some())
            {
                return TypeInfo::String;
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
                    } else if matches!(l, TypeInfo::Bytes) && matches!(r, TypeInfo::Bytes) {
                        // bytes + bytes is bytes (`x + b"c"` — the runtime
                        // py_add for Vec<u8> pairs concatenate): typing it
                        // keeps later display/repr through the bytes path
                        // (issue #137's bytes-display round).
                        TypeInfo::Bytes
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
            // The ITERATOR builtins carry their argument's element type
            // through (issue #222), so they are typed before the
            // name-only table below, which cannot see arguments.
            ExprType::Name(n)
                if iterator_builtin_type(&n.id, call, options, symbols).is_some() =>
            {
                iterator_builtin_type(&n.id, call, options, symbols)
                    .expect("just checked")
            }
            ExprType::Name(n) => match builtin_call_type(&n.id) {
                Some(t) => t,
                None => match symbols.get(&n.id) {
                    // A class-construction call produces an instance of the
                    // class (not Copy: reused instances must be cloned at
                    // each move-prone use, matching Python's aliasing).
                    Some(crate::SymbolTableNode::ClassDef(_)) => {
                        TypeInfo::Class(n.id.clone())
                    }
                    // A SAME-MODULE FUNCTION call resolves its return
                    // annotation through the same authority the import
                    // path uses (round 81: `resolve_cert_reqs(...)` — a
                    // `-> PyValue`-resolving callee — was invisible here,
                    // so the boxed argument never coerced into the
                    // `Option<i64>` slot it feeds). An IMPORTED callee
                    // stays PyObject: resolving its concrete class return
                    // (`proxy_from_url(...)` → `ProxyManager` in
                    // requests, which imports urllib3) would type the
                    // local as the class and make the generated crate
                    // reference a type it does not import.
                    Some(crate::SymbolTableNode::FunctionDef(_)) => {
                        call_return_typeinfo(call, Some(symbols), Some(options))
                            .unwrap_or(TypeInfo::PyObject)
                    }
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
            // An OPTION-wrapped base (`request_context["scheme"]` where
            // request_context is `dict[str, Any] | None` — urllib3's
            // poolmanager): the read unwraps the Option, so the element
            // type is the dict's value (round 64 — the boxed-str-method
            // dispatch keys off this).
            TypeInfo::Option(inner) => match *inner {
                TypeInfo::Dict(_, v) => *v,
                _ => TypeInfo::PyObject,
            },
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

/// The return TypeInfo of a builtin call, when statically known. ONE map
/// for the two inference paths (infer_type and syntactic_type) that
/// previously kept byte-identical copies: len()/count() must agree with
/// the `as i64` codegen emission everywhere, or an empty container pins
/// to different element types on different paths.
/// The element type an iterable expression yields, when it is knowable.
/// Only the shapes whose element type is genuinely carried: a `Vec`, and
/// a `range` (whose elements are Python ints). A string is deliberately
/// absent — iterating one yields single-character strings, which is a
/// different type from the receiver and not what any caller here wants.
fn iterable_element_type(t: &TypeInfo) -> Option<TypeInfo> {
    match t {
        TypeInfo::Vec(e) => Some((**e).clone()),
        TypeInfo::Range => Some(TypeInfo::Int),
        TypeInfo::Borrowed(inner) => iterable_element_type(inner),
        _ => None,
    }
}

/// The declared return type of a function referenced BY NAME (`map(double,
/// xs)` — the `double`), resolved through the symbol table. Only an
/// annotated module function answers; anything else leaves `map` untyped
/// rather than guessing at the element type.
fn named_fn_return_type(
    e: &ExprType,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<TypeInfo> {
    let ExprType::Name(n) = e else { return None };
    match symbols.get(&n.id) {
        Some(SymbolTableNode::FunctionDef(f)) => {
            resolve_alias_typeinfo(f.returns.as_deref()?, symbols, options)
        }
        Some(SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            let (f, _) = crate::module_function_def(options, &path, &n.id)?;
            resolve_alias_typeinfo(
                f.returns.as_deref()?,
                &module_symbols(options, &path),
                options,
            )
        }
        _ => None,
    }
}

/// The ITERATOR builtins, typed from their arguments (issue #222).
///
/// Each rule mirrors what the lowering actually emits, not merely what
/// Python means — the inferred type has to be the type of the rendered
/// Rust expression or the signature and the body disagree:
///
/// - `sorted(xs)`   -> `stdpython::sorted(&[T]) -> Vec<T>`
/// - `filter(f, xs)` -> `filter_fallible(f, Vec<T>) -> Result<Vec<T>, _>`
/// - `map(f, xs)`   -> `map_fallible(f, Vec<T>) -> Result<Vec<U>, _>`
/// - `list(x)`      -> `list(x) -> Vec<L::Item>`
///
/// `None` whenever the element type is not knowable — an untyped
/// iterable, or a `map` over a callable this cannot resolve (a lambda, a
/// bound method like `str.strip`). The caller then falls back to the
/// name-only table, so `list` keeps its boxed-element answer.
fn iterator_builtin_type(
    name: &str,
    call: &crate::Call,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<TypeInfo> {
    let elem_of = |e: &ExprType| iterable_element_type(&infer_type(e, options, symbols));
    match name {
        // sorted/filter preserve the element type; the iterable is the
        // last positional argument (`sorted(xs)`, `filter(pred, xs)`).
        "sorted" => Some(TypeInfo::Vec(Box::new(elem_of(call.args.first()?)?))),
        "filter" | "list" | "reversed" => {
            Some(TypeInfo::Vec(Box::new(elem_of(call.args.last()?)?)))
        }
        // map's element type is the CALLABLE's return type, not the
        // iterable's. Two arguments only: `map(f, a, b)` lowers through
        // map2 and is left to the name-only table.
        "map" if call.args.len() == 2 => {
            let f = call.args.first()?;
            // The iterable must still be a real iterable — an untyped one
            // means the lowering itself is on shaky ground.
            elem_of(call.args.last()?)?;
            Some(TypeInfo::Vec(Box::new(named_fn_return_type(
                f, options, symbols,
            )?)))
        }
        _ => None,
    }
}

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
        // `tuple(x)` always yields a tuple — the boxed value (the call's
        // lowering is PyValue::from(x); round 33's
        // `return tuple(retryable_exceptions)` in botocore's retryhandler
        // types the function boxed, so its callers see a PyValue).
        "tuple" => TypeInfo::PyValue,
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
        // An `np.ndarray` annotation names the runtime array type; without
        // this a `a: np.ndarray` local inferred PyObject and the numpy
        // attribute/method lowerings never recognized it (issue #197).
        "ndarray" | "np.ndarray" | "numpy.ndarray" => TypeInfo::NdArray,
        // A `T | None` annotation stored in local_types (`release_conn:
        // bool | None` — urllib3): the name holds an Option — the inner
        // type resolves through the same mapping (round 45).
        _ if py.ends_with(" | None") => {
            let inner = py.trim_end_matches(" | None").trim();
            match inner {
                "int" => TypeInfo::Option(Box::new(TypeInfo::Int)),
                "float" => TypeInfo::Option(Box::new(TypeInfo::Float)),
                "bool" => TypeInfo::Option(Box::new(TypeInfo::Bool)),
                "str" => TypeInfo::Option(Box::new(TypeInfo::String)),
                "bytes" => TypeInfo::Option(Box::new(TypeInfo::Bytes)),
                _ => TypeInfo::PyObject,
            }
        }
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

/// Lower a bare BUILTIN class name in VALUE position (`basestring =
/// (str, bytes)`, `{bytes: ..., str: ...}` — requests' compat/
/// _internal_utils): the builtin classes are class objects too, and the
/// class-as-value model names them by their name string (round 33).
/// Only when the name is NOT shadowed by a user binding (`str = "s"`, a
/// `def str(...)`, a class named `str`) — a self-assignment (`str =
/// str`, dropped as a no-op in module.rs's fold_static_import_trys) or
/// an import of one leaves the name meaning the builtin. VALUE-position
/// only: annotations never route through this (a `xs: list` parameter
/// keeps its own lowering), so it lives in the value renderers (tuple
/// elements, render_typed), not in Name::to_rust.
pub(crate) fn builtin_class_value(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<TokenStream> {
    let ExprType::Name(n) = expr else {
        return None;
    };
    if !crate::ast::tree::assign::is_builtin_class_name(&n.id) {
        return None;
    }
    let unshadowed = match symbols.get(&n.id) {
        None => true,
        Some(_) => {
            crate::ast::tree::module::import_binds_builtin_self_alias(&n.id, symbols, options)
        }
    };
    if !unshadowed {
        return None;
    }
    let name = n.id.clone();
    Some(quote!(#name.to_string()))
}

pub fn render_typed(
    expr: &ExprType,
    ctx: CodeGenContext,
    options: PythonOptions,
    symbols: SymbolTableScopes,
    expected: Option<TypeInfo>,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // A class name used as a VALUE (`[ChecksumError]`, `merge_setting(
    // ..., dict_class=OrderedDict)` — requests' sessions): classes as
    // values lower to their NAME STRINGS — the exception model is
    // string-tagged, so the name is the class object's only
    // runtime-relevant data (round 33 design; the old model dropped the
    // value to boxed None). This is a value-position renderer — callees
    // and type positions never come through here; direct construction
    // and isinstance/except are static and unaffected.
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
        // The RAW Python name — the exception model matches on it (a
        // raised `ChecksumError` is `PyException::new("ChecksumError",
        // ...)`, so a class VALUE must carry the same spelling, not the
        // mangled Rust ident).
        let name = n.id.clone();
        return Ok(quote!(#name.to_string()));
    }
    // A bare BUILTIN class name in value position (`{bytes: ...,
    // str: ...}` dict keys — requests' _internal_utils): the builtin
    // classes are class values too — their name strings (round 56).
    if let Some(tokens) = builtin_class_value(expr, &symbols, &options) {
        return Ok(tokens);
    }
    let tokens = expr
        .clone()
        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
    let tokens = match self_field_move_clone(expr, &ctx, &options, &symbols) {
        // A self-field read of a clone-safe field in a MOVE position
        // (every render_typed caller is one): clone out of the shared
        // receiver — a bare read would move out of `&self` (E0507).
        // Clone-safe = Copy scalars (no clone needed, excluded below),
        // immutable values (String, Vec<u8>), or the boxed PyValue
        // whose Arc shares dict/tuple members and value-copies the
        // immutable scalars — Python reference semantics either way.
        // MUTABLE containers (Vec<T>, PyDict, class instances) are NOT
        // cloned: a clone would silently break aliasing, so the E0507
        // stays loud (issue #79's discipline).
        Some(clone) => clone,
        None => tokens,
    };
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
    // Round 81: a NARROWED name read already converts to the member type
    // (name.rs: `(x).as_bytes().unwrap().to_vec()` for a Bytes-narrowed
    // boxed value, `(x).as_str().unwrap().to_string()` for String). The
    // recorded type (PyValue) is the PRE-narrowing union; coerce_tokens
    // against the member target would re-wrap the already-member tokens
    // (`(Vec<u8>).into()` — a redundant E0282/identity). The narrowed
    // target IS the actual type the tokens carry.
    let actual = match expr {
        ExprType::Name(n) => match options.narrowed_names.get(&n.id) {
            Some(crate::TypeInfo::StrOrBytes) => crate::TypeInfo::StrOrBytes,
            Some(crate::TypeInfo::String) | Some(crate::TypeInfo::StrRef) => {
                crate::TypeInfo::String
            }
            Some(crate::TypeInfo::Bytes) => crate::TypeInfo::Bytes,
            Some(crate::TypeInfo::PyValueMember(inner)) => (**inner).clone(),
            Some(t) => t.clone(),
            None => actual,
        },
        _ => actual,
    };
    // Round 83 (the generics directive): an OPTION-typed value into a
    // CONCRETE slot (`self.decompress(self._data)` where the field is
    // `bytes | None` — urllib3's DeflateDecoder, whose None stores
    // widen the field to Option<Vec<u8>>): `infer_type` answers PyObject
    // for attribute reads ("no answer"), but the ctx-aware predicate
    // resolves the Option through the class table — coerce from the
    // Option shape so the read unwraps with the loud §12.2 panic
    // (Python fails at use on a None value, rython at the conversion —
    // mirroring the return site). The inner conversion reuses the same
    // coercion: identity for the matching member, and a genuine inner
    // mismatch stays a loud rustc error. Excludes Option/StrOrBytes
    // slots — the empty case is their legitimate value. Round 84: a
    // PYVALUE slot also coerce — a None-then-assigned name whose
    // binding is `Option<PyValue>` (urllib3's `conn`) into a
    // PyValue-annotated parameter (`_prepare_proxy(conn:
    // BaseHTTPConnection)` — the TYPE_CHECKING stub resolves to the
    // boxed value): `Option<PyValue> → PyValue` unwraps via
    // `unwrap_or(PyValue::None_)` — Python's None passes through
    // exactly, no panic needed.
    let actual = if crate::ast::tree::function_def::expr_yields_option_ctx(
        expr, &ctx, &options, &symbols,
    ) && !matches!(expr, ExprType::Name(n)
        // An ANNOTATED name's PyValue/PyObject answer is authoritative:
        // its annotation resolved to the boxed value (`body: _TYPE_BODY |
        // None` — urllib3's urlopen param; `chunks: Iterable[bytes] |
        // None` — the annotated local), where the None is INSIDE the box,
        // so the fabricated Option-unwrap must not fire on it. Only an
        // UNANNOTATED None-stored local (`conn = None` then `conn = ...`
        // — declare-then-fill) has the Option binding the unwrap targets.
        if options.local_types.contains_key(&n.id)
            || options.annotated_names.contains(&n.id))
        && !matches!(expected, crate::TypeInfo::Option(_) | crate::TypeInfo::StrOrBytes)
        && (matches!(actual, crate::TypeInfo::PyObject)
            || (matches!(actual, crate::TypeInfo::PyValue)
                && matches!(expected, crate::TypeInfo::PyValue)))
    {
        crate::TypeInfo::Option(Box::new(expected.clone()))
    } else {
        actual
    };
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

/// Whether `expr` is a `self.<field>` read whose field type is
/// clone-safe (see [`render_typed`]): Copy scalars return None (no clone
/// needed); immutable values and the Arc-sharing PyValue return the
/// clone; mutable containers return None so the E0507 stays loud.
pub(crate) fn self_field_move_clone(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> Option<TokenStream> {
    let ExprType::Attribute(attr) = expr else {
        return None;
    };
    if !matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self") {
        return None;
    }
    let (class, class_symbols) = crate::receiver_class(&attr.value, ctx, symbols, options)?;
    let fields = class.infer_fields(&class_symbols, options).ok()?;
    let ty = fields
        .iter()
        .find(|(name, _)| *name == attr.attr)
        .map(|(_, ty)| ty.clone())?;
    let clone_safe = crate::ast::tree::type_ctx::type_mentions_heap(&ty);
    if !clone_safe {
        return None;
    }
    let tokens = expr
        .clone()
        .to_rust(ctx.clone(), options.clone(), symbols.clone())
        .ok()?;
    Some(quote!((#tokens).clone()))
}

/// The reuse-clone decision for a move-prone expression: a NAME read
/// more than once, or an ATTRIBUTE read whose NON-SELF receiver name is
/// read more than once (`item.name` for the key and the value — the
/// idiom corpus's `self.items[item.name] = item`). The attribute case
/// walks to the base name; a `self` receiver has its own clone machinery
/// (the accessor calls and self_field_read_clone) and is skipped here.
pub(crate) fn reuse_root_name(expr: &ExprType) -> Option<String> {
    match expr {
        ExprType::Name(n) => Some(n.id.clone()),
        ExprType::Attribute(a) => {
            let mut cur = a.value.as_ref();
            loop {
                match cur {
                    ExprType::Name(n) => {
                        break if n.id == "self" { None } else { Some(n.id.clone()) };
                    }
                    ExprType::Attribute(a2) => cur = a2.value.as_ref(),
                    _ => break None,
                }
            }
        }
        _ => None,
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
    if let Some(root) = reuse_root_name(expr) {
        let uses = options.use_counts.get(&root).copied().unwrap_or(0);
        if uses > 1 {
            let t = infer_type(expr, &options, &symbols);
            // Round 92: clone whenever the name is not statically Copy —
            // INCLUDING an inferrer-unknown (PyObject) name, which the
            // old gate excluded. A local bound from a SELF-METHOD call
            // whose return the inferrer cannot see (`data =
            // self._read(amt)` — the read family in urllib3's response)
            // is PyObject to infer_type while its ACTUAL binding is
            // `Vec<u8>` — skipping the clone moved the value into the
            // first call and every later read borrowed a moved value
            // (E0382, exposed once the compare fix let the loop
            // type-check). `.clone()` compiles on every generated type
            // (Copy types clone via Copy; classes derive Clone).
            if !t.is_copy() {
                // A CLASS-typed name (a local holding an instance —
                // `timeout_obj` from `self._get_timeout()`): the bare
                // `(#tokens).clone()` would resolve to the class's OWN
                // `clone` method when it defines one (urllib3's Timeout
                // does) — a REAL semantic call, where Python just re-reads
                // the variable. The reuse-clone is rython's ownership
                // artifact and must be Rust std Clone, invoked through the
                // trait so the inherent method cannot shadow it —
                // `Clone::clone(&x)` never names the concrete type, so a
                // TYPE_CHECKING-only class stub (rendering as PyValue)
                // stays valid (round 88).
                if matches!(t, TypeInfo::Class(_)) {
                    return Ok(quote!(Clone::clone(&(#tokens))));
                }
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
    // The pre-coercion render: render_typed may ADAPT the tokens
    // (`.into()` / `.to_string()` / `as f64`) — a clone wrapped around an
    // adapted form loses the adaptation's inference anchor
    // (`((key).into()).clone()` — E0282: the Into target is
    // unconstrained inside the clone, round 98). A clone only applies to
    // an UN-adapted read; an adapted expression is a fresh value whose
    // source-consumption is the pre-existing shape.
    let raw = expr
        .clone()
        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
    let tokens = render_typed(expr, ctx, options.clone(), symbols.clone(), expected)?;
    let adapted = raw.to_string() != tokens.to_string();
    // The ONE adaptation a clone cannot wrap is `.into()`: the Into
    // target is unconstrained inside the clone (`((key).into()).clone()`
    // — E0282, round 98). Every OTHER adaptation wraps fine — the
    // round-92 boxing (`PyValue::from((x))` → the box is a fresh value)
    // NEEDS the clone or the move returns.
    let adapted_is_into = adapted && tokens.to_string().contains("into");
    if let Some(root) = reuse_root_name(expr) {
        let uses = options.use_counts.get(&root).copied().unwrap_or(0);
        if uses > 1 && !adapted_is_into {
            let t = infer_type(expr, &options, &symbols);
            // See render_reused: clone whenever the name is not statically
            // Copy — INCLUDING inferrer-unknown (PyObject) names, whose
            // actual binding may be any non-Copy value (`data =
            // self._read(amt)` — the moved-value E0382s, round 92). A
            // CLASS-typed name's reuse-clone is the qualified std Clone —
            // never the class's own `clone` method (round 88).
            if !t.is_copy() {
                // See render_reused: a CLASS-typed name's reuse-clone must
                // be the trait-qualified std Clone (`Clone::clone(&x)`),
                // never the class's own `clone` method (round 88).
                if matches!(t, TypeInfo::Class(_)) {
                    return Ok(quote!(Clone::clone(&(#tokens))));
                }
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
    // A STRING-LITERAL annotation (`verify: "bool | str | None"` —
    // requests' adapters.py writes quoted annotations): re-parse the
    // string's content as the real expression, like typing.get_type_hints
    // (round 56). Otherwise the string Constant maps to TypeInfo::String
    // and every use of the parameter breaks.
    let unquoted = crate::ast::tree::arguments::unquote_annotation(ann);
    let ann: &ExprType = unquoted.as_ref().unwrap_or(ann);
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
            // `offsets: range` — the builtin range class as a type
            // annotation (charset_normalizer's cut_sequence_chunks): the
            // runtime PyRange (the same type `range(...)` calls infer).
            "range" => Some(TypeInfo::Range),
            // `bytearray` is a bytes-like type: same Vec<u8> lowering as
            // `bytes` (the tokens resolver always mapped it; the core
            // drifted — a `b: bytearray` parameter rendered a bare
            // `bytearray` ident that rustc rejected).
            "bytes" | "bytearray" => Some(TypeInfo::Bytes),
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
            "TracebackType" | "FrameType" | "CodeType" | "memoryview" => {
                Some(TypeInfo::PyValue)
            }
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
                    // `set[T]` / `frozenset[T]` — set literals lower to
                    // HashSet, so the annotated type is HashSet<T> (the
                    // tokens resolver always said so; the syntax-only pass
                    // drifted to Vec and the generated structs are the
                    // arbiter — urllib3's PoolKey fields compile as
                    // `Option<HashSet<(String, String)>>`).
                    if let crate::SubscriptKind::Index(elt) = &sub.kind {
                        Some(TypeInfo::HashSet(Box::new(annotation_type_info(elt)?)))
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
                // `Union[A, B, ...]` — the subscript spelling of `A | B`
                // (idna's `List[Union[Tuple[int, str], Tuple[int, str,
                // str]]]` — the _seg table annotations): members resolve
                // through the same mapping, a None member makes
                // Option<T>, and a boxable member mix widens to the boxed
                // PyValue — exactly the BinOp-union semantics.
                "Union" => {
                    let members = match &sub.kind {
                        crate::SubscriptKind::Index(i) => match i.as_ref() {
                            crate::ExprType::Tuple(t) => t.elts.clone(),
                            single => vec![single.clone()],
                        },
                        _ => return None,
                    };
                    if members.is_empty() {
                        return None;
                    }
                    let non_none: Vec<&crate::ExprType> =
                        members.iter().filter(|m| !crate::is_none_expr(m)).collect();
                    if non_none.len() != members.len() {
                        // Union[X, None] — Optional[X] when X resolves.
                        if let Some(inner) = (non_none.len() == 1)
                            .then(|| annotation_type_info(non_none[0]))
                            .flatten()
                        {
                            if matches!(inner, TypeInfo::PyValue) {
                                return Some(inner);
                            }
                            return Some(TypeInfo::Option(Box::new(inner)));
                        }
                    }
                    let resolved: Option<Vec<TypeInfo>> =
                        members.iter().map(|m| annotation_type_info(m)).collect();
                    if let Some(resolved) = resolved {
                        if resolved.iter().any(|t| matches!(t, TypeInfo::PyValue)) {
                            return Some(TypeInfo::PyValue);
                        }
                        if !resolved.is_empty() && resolved.iter().all(is_boxable_value_type) {
                            return Some(TypeInfo::PyValue);
                        }
                    }
                    None
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
                // `type[X]` / `Type[X]` — a CLASS value: rython cannot
                // hold classes as values (the callables-as-data
                // divergence); the tolerated opaque type is Option<()>.
                "type" | "Type" => Some(TypeInfo::ClassValue),
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
                if matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
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
            // `typing.List[...]` / `typing.Dict[...]` / `typing.Set[...]` /
            // `typing.Tuple[...]` / `typing.Optional[...]` /
            // `typing.Literal[...]` — the typing-module spellings of the
            // bare generics (urllib3's `_TYPE_VERSION_INFO =
            // typing.Tuple[int, int, int, str, int]`, NamedTuple call-form
            // fields `typing.Optional[str]`). One definition, so the
            // tokens resolver and the TypeInfo resolver cannot drift.
            ExprType::Attribute(a)
                if matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
                    && matches!(
                        a.attr.as_str(),
                        "List" | "Dict" | "Set" | "FrozenSet" | "Tuple" | "Optional"
                            | "Literal" | "Union"
                    ) =>
            {
                let container = match a.attr.as_str() {
                    "List" => "list",
                    "Dict" => "dict",
                    "Set" | "FrozenSet" => "set",
                    "Tuple" => "tuple",
                    "Optional" => "Optional",
                    "Union" => "Union",
                    _ => "Literal",
                };
                if let crate::SubscriptKind::Index(elt) = &sub.kind {
                    annotation_type_info(&ExprType::Subscript(crate::Subscript {
                        value: Box::new(ExprType::Name(crate::ast::tree::name::Name {
                            id: container.to_string(),
                        })),
                        kind: crate::SubscriptKind::Index(Box::new((**elt).clone())),
                        lineno: sub.lineno,
                        col_offset: sub.col_offset,
                        end_lineno: sub.end_lineno,
                        end_col_offset: sub.end_col_offset,
                    }))
                } else {
                    None
                }
            }
            _ => None,
        },
        // `threading.Event` / `socket.socket` / `numpy.ndarray` /
        // `typing.Any` attribute annotations: concrete runtime types (or
        // the boxed value for Any), known syntactically — one definition,
        // so the tokens resolver and the TypeInfo resolver cannot drift.
        ExprType::Attribute(attr) => {
            if let ExprType::Name(n) = attr.value.as_ref() {
                if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Threading) {
                    return crate::ThreadingType::from_name(&attr.attr).map(TypeInfo::Threading);
                }
                if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Socket)
                    && attr.attr == "socket"
                {
                    return Some(TypeInfo::Socket);
                }
                if crate::is_typing(&n.id) && attr.attr == "Any" {
                    return Some(TypeInfo::PyValue);
                }
                if crate::is_numpy_alias(&n.id) {
                    return match attr.attr.as_str() {
                        "ndarray" => Some(TypeInfo::NdArray),
                        "float64" | "float32" => Some(TypeInfo::Float),
                        "int64" | "int32" => Some(TypeInfo::Int),
                        "bool_" => Some(TypeInfo::Bool),
                        _ => None,
                    };
                }
            }
            None
        }
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
    /// Names assigned None on some path (or seeded from an Option-typed
    /// self-field by the class-aware analysis): Option bindings whose
    /// access lowering unwraps (issue #137's Option-aware access).
    pub optional_names: HashSet<String>,
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
    analyze_function_types_with_class(body, options, symbols, None)
}

/// [`analyze_function_types`] with the ENCLOSING CLASS (issue #137's
/// Option-aware access): a local assigned from a field of the method's
/// own class (`resp_options = self._response_options` — urllib3) types
/// from the field table — a `Timeout | None` field makes the local an
/// Option, so later access through it unwraps like any Option receiver.
/// Seeded only for locals the plain analysis could not type.
pub fn analyze_function_types_with_class(
    body: &[Statement],
    options: Option<&PythonOptions>,
    symbols: Option<&SymbolTableScopes>,
    self_class: Option<&str>,
) -> FunctionTypeInfo {
    let mut info = analyze_function_types_inner(body, options, symbols);
    if let Some(class_name) = self_class
        && let (Some(options), Some(symbols)) = (options, symbols)
        && let Some(crate::SymbolTableNode::ClassDef(class)) = symbols.get(class_name)
        && let Ok(_) = class.infer_fields(symbols, options)
    {
        // Recurse into nested bodies: the Option-widening stores sit
        // inside `if`/`with` blocks (`server_hostname = self._tunnel_host`
        // inside `if self._tunnel_host is not None:`).
        fn walk_stmts(
            stmts: &[crate::Statement],
            class: &crate::ClassDef,
            info: &mut FunctionTypeInfo,
            options: &PythonOptions,
            symbols: &SymbolTableScopes,
        ) {
            for stmt in stmts {
                let inner: Option<&[crate::Statement]> = match &stmt.statement {
                    crate::StatementType::If(s) => Some(s.body.as_slice()),
                    crate::StatementType::While(s) => Some(s.body.as_slice()),
                    crate::StatementType::For(s) => Some(s.body.as_slice()),
                    crate::StatementType::With(s) => Some(s.body.as_slice()),
                    crate::StatementType::Try(s) => Some(s.body.as_slice()),
                    _ => None,
                };
                if let Some(inner) = inner {
                    walk_stmts(inner, class, info, options, symbols);
                }
                let crate::StatementType::Assign(a) = &stmt.statement else {
                    continue;
                };
                let [crate::ExprType::Name(n)] = a.targets.as_slice() else {
                    continue;
                };
                if info.optional_names.contains(&n.id) {
                    continue;
                }
                // A name the plain analysis already typed as OPTION stays
                // as it is; a PLAIN-typed name may still be WIDENED by an
                // Option-typed field store below (`server_hostname: str =
                // self.host` then `server_hostname = self._tunnel_host` —
                // the Python value becomes None-able; the annotation was a
                // hint, not a constraint).
                let already_option = matches!(
                    info.name_types.get(&n.id),
                    Some(crate::TypeInfo::Option(_))
                );
                // A `typing.cast(T, value)` assignment (`proxy_config =
                // typing.cast(ProxyConfig, self.proxy_config)` — urllib3's
                // _connect_tls_proxy): the cast is a runtime identity, so
                // the VALUE's shape seeds the local exactly like the
                // direct form (round 95 — without it the cast-assigned
                // local stayed unknown and the Option-field reads on it
                // never unwrapped, E0609).
                let cast_value = match &a.value {
                    crate::ExprType::Call(call)
                        if call.args.len() == 2
                            && (matches!(
                                call.func.as_ref(),
                                crate::ExprType::Name(n)
                                    if n.id == "cast"
                                        && matches!(
                                            symbols.get(&n.id),
                                            Some(crate::SymbolTableNode::ImportFrom(i))
                                                if crate::AnnotationModule::from_name(
                                                    i.module.split('.').next().unwrap_or("")
                                                ) == Some(crate::AnnotationModule::Typing)
                                        )
                            ) || matches!(
                                call.func.as_ref(),
                                crate::ExprType::Attribute(attr)
                                    if attr.attr == "cast"
                                        && matches!(
                                            attr.value.as_ref(),
                                            crate::ExprType::Name(m)
                                                if crate::is_typing(&m.id)
                                        )
                            )) =>
                    {
                        Some(&call.args[1])
                    }
                    _ => None,
                };
                let value_ref = cast_value.unwrap_or(&a.value);
                match value_ref {
                    // `request_context = self._merge_pool_kwargs(
                    // pool_kwargs)` — a local assigned from a SELF-METHOD
                    // CALL whose callee returns a dict (`-> dict[str,
                    // typing.Any]` — the poolmanager): the local is the
                    // callee's return type. NARROW: only Dict-returning
                    // callees seed the local — typing every self-method
                    // return was a wash in round 44 (conn locals exposed
                    // close-on-Option and From<Option<String>> cascades),
                    // but a Dict-typed local is exactly what the
                    // subscript-store lowering needs (string_keyed/
                    // pyvalue_valued ownership, round 46) and only the
                    // py_set_index/key handling changes for it.
                    crate::ExprType::Call(call) => {
                        if info.name_types.contains_key(&n.id) {
                            continue;
                        }
                        let crate::ExprType::Attribute(attr) = call.func.as_ref() else {
                            continue;
                        };
                        if !matches!(
                            attr.value.as_ref(),
                            crate::ExprType::Name(r) if r.id == "self"
                        ) {
                            continue;
                        }
                        if let Some(method) = class.method_on_mro(&attr.attr, symbols)
                            && let Some(ann) = method.returns.as_deref()
                            && let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
                            && matches!(t, crate::TypeInfo::Dict(_, _))
                        {
                            info.name_types.insert(n.id.clone(), t);
                        }
                        // Round 87: a CLASS-returning self-method call
                        // (`timeout_obj = self._get_timeout()` — urllib3's
                        // `-> Timeout` callee) seeds the local with the
                        // class, so a later property read on it
                        // (`timeout_obj.read_timeout` — an
                        // `Option<f64>`-returning accessor) resolves its
                        // receiver's class. The Dict arm above was the
                        // narrow survivor of round 44's wash (typing EVERY
                        // self-method return exposed close/From cascades);
                        // the CLASS arm is safe — the local IS the
                        // instance, and the property arm below consumes it.
                        if let Some(method) = class.method_on_mro(&attr.attr, symbols)
                            && let Some(ann) = method.returns.as_deref()
                            && let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
                            && matches!(t, crate::TypeInfo::Class(_))
                        {
                            info.name_types.insert(n.id.clone(), t);
                        }
                        // An OPTION-of-CLASS-returning self-method call
                        // (`item = self.find(name)` — a `-> Optional[Item]`
                        // finder whose result the caller narrows with an
                        // early-exit guard, the idiom corpus's take()):
                        // seed the local as the Option BINDING — name_types
                        // AND optional_names — so the `is None`-guard
                        // narrowing fires and later field reads unwrap the
                        // class (the corpus's four `Option<Item>` errors).
                        // The round-44 wash came from typing EVERY
                        // self-method return; an Option<Class> local is the
                        // narrow shape the guard narrowing consumes.
                        if let Some(method) = class.method_on_mro(&attr.attr, symbols)
                            && let Some(ann) = method.returns.as_deref()
                            && let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
                            && matches!(
                                &t,
                                crate::TypeInfo::Option(inner)
                                    if matches!(**inner, crate::TypeInfo::Class(_))
                            )
                        {
                            info.optional_names.insert(n.id.clone());
                            info.name_types.insert(n.id.clone(), t);
                        }
                    }
                    crate::ExprType::Attribute(attr) => {
                        // A SELF-field read (`resp_options =
                        // self._response_options` — the enclosing class's
                        // own field table). The field may live on a BASE
                        // whose struct is embedded (`self._tunnel_host` in
                        // a derived method): walk the chain.
                        let self_read = matches!(
                            attr.value.as_ref(),
                            crate::ExprType::Name(r) if r.id == "self"
                        );
                        let field_ty = if self_read {
                            class
                                .base_chain(symbols)
                                .iter()
                                .find_map(|c| {
                                    c.infer_fields(symbols, options).ok().and_then(|fs| {
                                        fs.iter()
                                            .find(|(name, _)| *name == attr.attr)
                                            .map(|(_, ty)| ty.clone())
                                    })
                                })
                        } else if let Some((owner, owner_symbols)) =
                            crate::receiver_class_for_read(
                                &attr.value,
                                &crate::CodeGenContext::Module(String::new()),
                                symbols,
                                options,
                            )
                        {
                            // A field of ANOTHER object whose class
                            // resolves (`destination_scheme =
                            // parsed_url.scheme` where parsed_url is a
                            // factory local — the same Option seeding as
                            // the self-field arm; without it the local
                            // stays untyped and an Option-slot argument
                            // double-wraps `Some(destination_scheme)`).
                            owner
                                .infer_fields(&owner_symbols, options)
                                .ok()
                                .and_then(|fs| {
                                    fs.iter()
                                        .find(|(name, _)| *name == attr.attr)
                                        .map(|(_, ty)| ty.clone())
                                })
                        } else {
                            None
                        };
                        // An OPTION field widens the local — including a
                        // name the plain analysis typed PLAIN
                        // (`server_hostname: str = self.host` then
                        // `server_hostname = self._tunnel_host`: the Python
                        // value becomes None-able — the annotation was a
                        // hint). name_types only — the local IS the Option
                        // (its stores are already Option and must not
                        // Some-wrap again), and reads unwrap through the
                        // Option receiver lowering.
                        if let Some(ty) = field_ty
                            && matches!(ty, crate::TypeInfo::Option(_))
                            && !already_option
                        {
                            // The REAL field type (`Option<ProxyConfig>` —
                            // the `proxy_config = cast(ProxyConfig,
                            // self.proxy_config)` local, whose reads must
                            // resolve the inner class's fields) — not the
                            // unknown placeholder: the local IS the field's
                            // Option, and an Option<Class> inner lets the
                            // receiver resolution and the Option-slot
                            // coercions see through it (round 95).
                            info.name_types.insert(n.id.clone(), ty);
                        }
                        // Round 87: a PROPERTY read on a class-resolved
                        // receiver (`read_timeout = timeout_obj.read_timeout`
                        // — urllib3's `@property def read_timeout(self) ->
                        // float | None` on the Timeout instance the
                        // class-seeding arm typed): the getter's return
                        // annotation types the local, so an Option getter
                        // makes it an Option binding (the `_raise_timeout(
                        // e, url, read_timeout)` argument then coerces
                        // instead of going in raw). The receiver's class
                        // comes from the WALK's own seeding
                        // (`info.name_types` — a local the plain analysis
                        // left PyObject) as well as the plain analysis.
                        let info_receiver_class = match attr.value.as_ref() {
                            crate::ExprType::Name(rn) => match info.name_types.get(&rn.id) {
                                Some(crate::TypeInfo::Class(c)) => {
                                    crate::receiver_class_tail(c, symbols.clone(), options)
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some((owner, owner_symbols)) = info_receiver_class
                            .or_else(|| {
                                crate::receiver_class_for_read(
                                    &attr.value,
                                    &crate::CodeGenContext::Module(String::new()),
                                    symbols,
                                    options,
                                )
                            })
                            && !already_option
                            && let Some(inner) =
                                owner.method_on_mro(&attr.attr, symbols).and_then(|m| {
                                    let is_property = m
                                        .decorator_list
                                        .iter()
                                        .any(|d| match d {
                                            crate::ExprType::Name(n) => n.id == "property",
                                            crate::ExprType::Attribute(a) => {
                                                a.attr == "property"
                                            }
                                            _ => false,
                                        });
                                    if !is_property {
                                        return None;
                                    }
                                    m.returns.as_deref().and_then(|r| {
                                        match crate::resolve_alias_typeinfo(
                                            r, &owner_symbols, options,
                                        ) {
                                            Some(crate::TypeInfo::Option(inner)) => Some(*inner),
                                            _ => None,
                                        }
                                    })
                                })
                        {
                            info.name_types.insert(
                                n.id.clone(),
                                crate::TypeInfo::Option(Box::new(inner)),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        walk_stmts(body, class, &mut info, options, symbols);
    }
    info
}

fn analyze_function_types_inner(
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
                if crate::is_none_expr(&assign.value) {
                    info.optional_names.insert(name.id.clone());
                }
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
                    })
                {
                    // An annotation pins the type outright.
                    Some(ann) => ann,
                    // Unparseable annotation: a call to a known function
                    // resolves through its (alias-aware) return type
                    // (`chunk_languages = cached_coherence_ratio(...)` →
                    // Vec<(String, f64)>), else the value's syntactic type
                    // (still pinable by later use).
                    None => match &assign.value {
                        ExprType::Call(c) => {
                            call_return_typeinfo(c, symbols, options)
                                .unwrap_or_else(|| syntactic_type(&assign.value))
                        }
                        // A BINOP value needs the context-aware inferrer:
                        // `y = x + b"c"` is bytes (bytes + bytes), which
                        // the context-free syntactic_type cannot see (its
                        // operands are names). The two paths agree
                        // everywhere else; this is the one shape where
                        // only infer_type has the operand types.
                        ExprType::BinOp(_) => match (options, symbols) {
                            (Some(options), Some(symbols)) => {
                                infer_type(&assign.value, options, symbols)
                            }
                            _ => syntactic_type(&assign.value),
                        },
                        // A CONTAINER literal likewise needs the
                        // context-aware inferrer: a dict/list of CLASS
                        // VALUES (`pool_classes_by_scheme = {"http":
                        // HTTPConnectionPool}` — urllib3's poolmanager)
                        // types Dict(String, String) through the
                        // class-as-value rule (round 33) — the same
                        // answer the literal LOWERING gives — where
                        // syntactic_type would see untyped elements and
                        // the module static boxes the whole dict.
                        _ if crate::ast::tree::assign::is_container_literal(&assign.value) => match (options, symbols)
                        {
                            (Some(options), Some(symbols)) => {
                                infer_type(&assign.value, options, symbols)
                            }
                            _ => syntactic_type(&assign.value),
                        },
                        // A NAME value (`release_this_conn =
                        // release_conn` where the param is `bool | None`
                        // — urllib3's urlopen): the context-aware
                        // inferrer resolves the Option through the
                        // param's annotation, so the local is tracked as
                        // an Option binding and later plain stores
                        // Some-wrap (round 45).
                        ExprType::Name(_) => match (options, symbols) {
                            (Some(options), Some(symbols)) => {
                                infer_type(&assign.value, options, symbols)
                            }
                            _ => syntactic_type(&assign.value),
                        },
                        _ => syntactic_type(&assign.value),
                    },
                };
                // Dict keys normalize to String (matches literal lowering
                // and `dict[str, V]` annotations); VALUES of string
                // literals likewise normalize (`headers_ = {"Accept":
                // "*/*"}` — a `-> Mapping[str, str]` local — so the local
                // types Dict(String, String), agreeing with the literal
                // lowering's owned values, round 87); empty dicts and lists
                // are remembered for pinning from later use.
                t = match t {
                    TypeInfo::Dict(k, v)
                        if matches!(*k, TypeInfo::StrRef) || matches!(*v, TypeInfo::StrRef) =>
                    {
                        TypeInfo::Dict(
                            Box::new(if matches!(*k, TypeInfo::StrRef) {
                                TypeInfo::String
                            } else {
                                (*k).clone()
                            }),
                            Box::new(if matches!(*v, TypeInfo::StrRef) {
                                TypeInfo::String
                            } else {
                                (*v).clone()
                            }),
                        )
                    }
                    // A string-literal LIST element (`ks = ["Retry-After"]`
                    // — the literal lowering owns the element, so the local
                    // types Vec(String) in agreement, round 87).
                    TypeInfo::Vec(inner) if matches!(*inner, TypeInfo::StrRef) => {
                        TypeInfo::Vec(Box::new(TypeInfo::String))
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
                    // A local assigned from an OPTION-typed value
                    // (`release_this_conn = release_conn` where the param
                    // is `bool | None` — urllib3's urlopen) is itself an
                    // Option binding: later plain stores must Some-wrap
                    // through the Option-slot path (round 45).
                    if matches!(t, TypeInfo::Option(_)) {
                        info.optional_names.insert(name.id.clone());
                    }
                    // Empty container: remember it to pin from later use.
                    if is_empty_container(&assign.value) {
                        info.empty_pinned.insert(name.id.clone(), t);
                    }
                }
            } else if let [ExprType::Tuple(targets)] = assign.targets.as_slice() {
                // A TUPLE-target destructure whose RHS is a call returning a
                // typed tuple (`(body, content_type) = encode_multipart_
                // formdata(...)` — `-> tuple[bytes, str]`, urllib3's
                // RequestMethods.request): the per-element types seed the
                // locals so a later literal store into the String slot owns
                // its &str literal (round 46). Only seeds names the plain
                // analysis left untyped.
                if let ExprType::Call(call) = &assign.value
                    && let Some(crate::TypeInfo::Tuple(infos)) =
                        crate::call_return_typeinfo(call, symbols, options)
                    && targets.elts.len() == infos.len()
                {
                    for (t, slot) in targets.elts.iter().zip(infos.iter()) {
                        if let ExprType::Name(n) = t
                            && !info.name_types.contains_key(&n.id)
                            && !matches!(slot, TypeInfo::PyObject)
                        {
                            info.name_types.insert(n.id.clone(), slot.clone());
                        }
                    }
                }
                // A tuple-target store of ALL-None literals (`auth, host,
                // port = None, None, None` — urllib3's parse_url): each
                // element name becomes an Option binding, mirroring the
                // single-name None store above. Without this the names stay
                // PyObject and a later `host = _normalize_host(...)` value
                // double-wraps when passed to an Option slot (round 47).
                if let ExprType::Tuple(vt) = &assign.value
                    && vt.elts.len() == targets.elts.len()
                    && vt.elts.iter().all(crate::is_none_expr)
                {
                    for t in &targets.elts {
                        if let ExprType::Name(n) = t {
                            info.optional_names.insert(n.id.clone());
                        }
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
        StatementType::Raise(r) => {
            // A raise's exception expression READS its operands
            // (`raise KeyError(name)` after `find(name)` moved name — the
            // idiom corpus's take): uncounted, the reuse-clone never fired
            // on the earlier move and the raise borrowed a moved value
            // (E0382, round 98).
            if let Some(e) = &r.exc {
                count_expr_reads(e, info);
            }
            if let Some(e) = &r.cause {
                count_expr_reads(e, info);
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
            Some(litrs::Literal::Byte(_)) | Some(litrs::Literal::ByteString(_)) => {
                TypeInfo::Bytes
            }
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
                        if info.name_types.contains_key(&recv.id)
                            && (key_unknown || val_unknown)
                        {
                            return;
                        }
                        let ty = TypeInfo::Dict(Box::new(k), Box::new(v));
                        // An UNKNOWN value type (`modeled_actions[
                        // modeled_action.name] = modeled_action` where the
                        // value is a foreign object — boto3's
                        // document_actions): box the value.
                        let ty = match ty {
                            TypeInfo::Dict(k, v)
                                if matches!(*v, TypeInfo::PyObject) =>
                            {
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

/// The bare container name of a Subscript's value (`Optional` for both
/// `Optional[...]` and `typing.Optional[...]`, `dict` for `Dict[...]` /
/// `typing.Dict[...]`). None for non-container values.
fn subscript_container_name(sub: &crate::Subscript) -> Option<String> {
    match sub.value.as_ref() {
        ExprType::Name(n) => Some(n.id.clone()),
        ExprType::Attribute(a)
            if matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id)) =>
        {
            Some(a.attr.clone())
        }
        _ => None,
    }
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
            // `str | bytes` (and `str | bytes | bytearray`) is the
            // StrOrBytes heterogeneous union, not a boxed PyValue — the
            // syntax core knew this, the alias-aware arm drifted (issue
            // #137's systemic review).
            if crate::is_str_bytes_union(ann) {
                return Some(TypeInfo::StrOrBytes);
            }
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
                && matches!(attr.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
            {
                return Some(TypeInfo::PyValue);
            }
            // `threading.Event` / `socket.socket` attribute annotations:
            // concrete runtime handles — resolved BEFORE the module loop
            // (threading/socket are external modules, which would box
            // them; the tokens resolver always mapped them, so the
            // authority must too).
            if let ExprType::Name(n) = attr.value.as_ref() {
                if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Threading) {
                    if let Some(t) = crate::ThreadingType::from_name(&attr.attr) {
                        return Some(TypeInfo::Threading(t));
                    }
                }
                if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Socket)
                    && attr.attr == "socket"
                {
                    return Some(TypeInfo::Socket);
                }
            }
            let ExprType::Name(module) = attr.value.as_ref() else {
                // A NESTED module chain (`OpenSSL.SSL.Connection` — a
                // pyOpenSSL class annotation, urllib3's WrappedSocket): the
                // root is an external import — a boxed value.
                if crate::root_name(&attr.value)
                    .is_some_and(|r| !crate::is_typing(&r))
                {
                    return Some(TypeInfo::PyValue);
                }
                return annotation_type_info(ann);
            };
            // numpy annotations name RUNTIME types, not external classes.
            // `numpy` is not in module_defs, so the module loop below used
            // to fall through to "external import → boxed PyValue": a
            // function annotated `-> np.ndarray` had its result boxed, and
            // every use of the local failed in rustc (issue #203).
            if crate::is_numpy_alias(&module.id) {
                return match attr.attr.as_str() {
                    "ndarray" => Some(TypeInfo::NdArray),
                    "float64" | "float32" => Some(TypeInfo::Float),
                    "int64" | "int32" => Some(TypeInfo::Int),
                    "bool_" => Some(TypeInfo::Bool),
                    _ => None,
                };
            }
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
                        crate::SymbolTableNode::ImportFrom(_)
                            | crate::SymbolTableNode::Import(_)
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
                    Some(SymbolTableNode::Assign { value, .. })
                        if crate::is_none_expr(value) =>
                    {
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
                                    a.name
                                        .split('.')
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
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
                if !text.is_empty() && text.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_') {
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
                if matches!(
                    crate::AnnotationModule::from_name(&i.module),
                    Some(
                        crate::AnnotationModule::Typing
                            | crate::AnnotationModule::TypingExtensions
                    )
                ) =>
            {
                Some(TypeInfo::PyValue)
            }
            // An import from a module outside the crate (`CookieJar` from
            // http.cookiejar): an external class — a boxed value.
            Some(SymbolTableNode::ImportFrom(i))
                if !options.module_defs.contains_key(&i.resolved_module_path(options)) =>
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
                            options,
                            &path,
                            &n.id,
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
                            &ExprType::Name(crate::ast::tree::name::Name {
                                id: n.id.clone(),
                            }),
                            &syms,
                            options,
                        )
                    }
                    _ => None,
                }
            }
            // A user-defined class name (`list[CharsetMatch]`): the struct
            // ident, the same path parameters use.
            Some(SymbolTableNode::ClassDef(_)) => {
                Some(TypeInfo::Class(n.id.clone()))
            }
            _ => annotation_type_info(ann),
        },
        // A container generic whose ELEMENT may be an alias: rebuild with
        // the resolved element type.
        ExprType::Subscript(sub) => {
            // `Optional[T]` / `typing.Optional[T]` — an OPTION of the
            // inner type, NOT a boxed value: a NamedTuple field annotated
            // `typing.Optional[str]` (`Url` — urllib3) is an
            // `Option<String>` field; boxing it made every `self.scheme =
            // scheme` store in the synthesized __init__ a
            // `PyValue`-vs-`Option<String>` mismatch (round 46). The
            // inner resolves through the same alias-aware path; an
            // unresolvable inner stays the pre-existing loud fallback
            // (rustc fails the field type).
            if let (Some(c), crate::SubscriptKind::Index(inner)) =
                (subscript_container_name(sub), &sub.kind)
                && matches!(c.as_str(), "Optional")
            {
                return resolve_alias_typeinfo(inner, symbols, options).map(|t| {
                    if matches!(t, TypeInfo::PyValue) {
                        // A boxed inner already contains None.
                        t
                    } else {
                        TypeInfo::Option(Box::new(t))
                    }
                });
            }
            let container = match sub.value.as_ref() {
                // Bare `Iterable[...]` / `IO[Any]` / `Sequence[...]` etc. —
                // typing generics tolerated inside a boxed PyValue union.
                ExprType::Name(n)
                    if matches!(
                        n.id.as_str(),
                        "Union" | "IO" | "Iterable" | "Sequence" | "Iterator" | "Generator"
                            | "Callable" | "SupportsRead" | "SupportsItems"
                            | "MutableMapping" | "Optional" | "Literal" | "Any"
                    ) =>
                {
                    return Some(TypeInfo::PyValue);
                }
                // A subscripted name that is a real symbol (`Morsel[
                // dict[str, str]]` — an imported class generic from
                // http.cookiejar) is a boxed value — except the container
                // generics, which resolve through the second match below
                // (`Mapping` and `type`/`Type` included: both have
                // structural forms in the syntax core).
                ExprType::Name(n)
                    if symbols.get(&n.id).is_some()
                        && !matches!(
                            n.id.as_str(),
                            "list" | "List" | "tuple" | "Tuple" | "dict" | "Dict" | "set"
                                | "Set" | "Optional" | "Mapping" | "type" | "Type"
                        ) =>
                {
                    return Some(TypeInfo::PyValue);
                }
                ExprType::Name(n) => n.id.as_str(),
                ExprType::Attribute(a)
                    if matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id)) =>
                {
                    match a.attr.as_str() {
                        "List" => "list",
                        "Tuple" => "tuple",
                        "Dict" => "dict",
                        "Set" => "set",
                        "Mapping" => "Mapping",
                        "Type" => "type",
                        // Other typing generics (`IO[Any]`,
                        // `Iterable[bytes | str]`, `Union[...]`,
                        // `Callable[...]`) are tolerated inside a boxed
                        // PyValue union (urllib3's `_TYPE_BODY`).
                        "Union" | "IO" | "Iterable" | "Callable" | "SupportsRead"
                        | "SupportsItems" | "MutableMapping" | "Optional"
                        | "Literal" | "Any" | "Sequence" | "Iterator" | "Generator"
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
                        &ExprType::Name(crate::ast::tree::name::Name {
                            id: a.attr.clone(),
                        }),
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
                        // `tuple[T, ...]` — a variadic tuple → Vec<T>
                        // (the syntax core handles it; the alias-aware arm
                        // drifted and resolved the Ellipsis to None —
                        // issue #137's systemic review).
                        if t.elts.len() == 2
                            && matches!(
                                &t.elts[1],
                                ExprType::Constant(c)
                                    if c.0
                                        .as_ref()
                                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                            )
                        {
                            return Some(TypeInfo::Vec(Box::new(resolve_alias_typeinfo(
                                &t.elts[0], symbols, options,
                            )?)));
                        }
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
                // HashSet of the element type — set literals generate
                // HashSet, so annotated sets are that type (the generated
                // structs are the arbiter; the earlier Vec answer drifted
                // from the tokens resolver).
                ("set" | "Set" | "frozenset", crate::SubscriptKind::Index(elt)) => {
                    Some(TypeInfo::HashSet(Box::new(resolve_alias_typeinfo(
                        elt, symbols, options,
                    )?)))
                }
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
        // An ATTRIBUTE callee on a CLASS-typed receiver (`inv.find(name)`
        // where inv = Inventory() — the idiom corpus's find): resolve the
        // method through the receiver's class MRO and take its return
        // annotation — the call LOWERING already resolves this shape (the
        // `?` is emitted), so the type side must agree or the Option
        // machinery (narrowing, the receiver unwrap) never sees it
        // (round 99).
        let ExprType::Attribute(attr) = call.func.as_ref() else {
            return None;
        };
        let (symbols, options) = (symbols?, options?);
        let ExprType::Name(recv) = attr.value.as_ref() else {
            return None;
        };
        let Some(crate::TypeInfo::Class(cname)) = options.name_types.get(&recv.id) else {
            return None;
        };
        let class = crate::resolve_class_referenced(cname, symbols, options)?;
        let method = class.method_on_mro(&attr.attr, symbols)?;
        let ann = method.returns.as_deref()?;
        return resolve_alias_typeinfo(ann, symbols, options);
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
            // module_defs is keyed RELATIVE to the package root for
            // src-layout packages, while the resolved path may carry the
            // root-qualified prefix — the same normalization the import
            // lowering uses (round 77: charset_normalizer's
            // encoding_unicode_range — an imported `str | None`-returning
            // callee stayed untyped, so the local never entered
            // optional_names and the is-not-None narrowing never fired).
            let path = crate::module_defs_key(options, &path)?;
            let (f, _) = crate::module_function_def(options, path, &callee.id)?;
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
            // Round 85 (the return-type directive): an UNANNOTATED callee
            // whose body can return exactly `T | None` infers an
            // `Option<T>` return — the caller must learn it so the
            // Option-aware machinery applies (the caller decides what to
            // do with the None; a concrete use without handling it is the
            // loud Option→concrete panic). The annotation is authoritative
            // when present.
            if let Some(ann) = f.returns.as_deref() {
                resolve_alias_typeinfo(ann, symbols, options)
            } else {
                f.inferred_return_typeinfo(symbols, options)
            }
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
    if d > 200 && d % 50 == 0 {
    }
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

/// Whether a name is READ AS A VALUE anywhere in the body. A pure None
/// check (`x is None` / `x is not None`) does NOT count — issue #117's
/// None-defaulted unannotated parameters are otherwise the concrete
/// `Option<()>` (nothing but None can be stored in them), and a store
/// target is not a read either. A read in any other position means the
/// parameter genuinely carries a Python value, so it must be typed to
/// hold one (the boxed `PyValue` — round 33's `retryable_exceptions=None`
/// in botocore's MaxAttemptsDecorator, stored into a field and later
/// matched against in `except self._retryable_exceptions:`).
pub(crate) fn name_read_as_value(name: &str, stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| statement_reads_name_as_value(name, s))
}

fn statement_reads_name_as_value(name: &str, s: &Statement) -> bool {
    use crate::StatementType as ST;
    match &s.statement {
        ST::Expr(e) => expr_reads_name_as_value(name, &e.value),
        ST::Assign(a) => {
            expr_reads_name_as_value(name, &a.value)
                || a.targets
                    .iter()
                    .any(|t| target_reads_name_as_value(name, t))
        }
        ST::AugAssign(a) => {
            expr_reads_name_as_value(name, &a.target)
                || expr_reads_name_as_value(name, &a.value)
        }
        ST::Return(Some(e)) => expr_reads_name_as_value(name, &e.value),
        ST::Assert { test, msg } => {
            expr_reads_name_as_value(name, test)
                || msg
                    .as_ref()
                    .is_some_and(|m| expr_reads_name_as_value(name, m))
        }
        ST::Raise(r) => {
            r.exc.as_ref().is_some_and(|e| expr_reads_name_as_value(name, e))
                || r.cause.as_ref().is_some_and(|e| expr_reads_name_as_value(name, e))
        }
        ST::If(i) => {
            expr_reads_name_as_value(name, &i.test)
                || i.body.iter().any(|b| statement_reads_name_as_value(name, b))
                || i.orelse.iter().any(|b| statement_reads_name_as_value(name, b))
        }
        ST::While(w) => {
            expr_reads_name_as_value(name, &w.test)
                || w.body.iter().any(|b| statement_reads_name_as_value(name, b))
                || w.orelse.iter().any(|b| statement_reads_name_as_value(name, b))
        }
        ST::For(f) => {
            expr_reads_name_as_value(name, &f.iter)
                || f.body.iter().any(|b| statement_reads_name_as_value(name, b))
                || f.orelse.iter().any(|b| statement_reads_name_as_value(name, b))
        }
        ST::With(w) => {
            w.items.iter().any(|item| {
                expr_reads_name_as_value(name, &item.context_expr)
                    || item
                        .optional_vars
                        .as_ref()
                        .is_some_and(|v| target_reads_name_as_value(name, v))
            }) || w.body.iter().any(|b| statement_reads_name_as_value(name, b))
        }
        ST::Try(t) => {
            t.body.iter().any(|b| statement_reads_name_as_value(name, b))
                || t.orelse.iter().any(|b| statement_reads_name_as_value(name, b))
                || t.finalbody.iter().any(|b| statement_reads_name_as_value(name, b))
                || t.handlers.iter().any(|h| {
                    h.exception_type
                        .as_ref()
                        .is_some_and(|et| expr_reads_name_as_value(name, et))
                        || h.body.iter().any(|b| statement_reads_name_as_value(name, b))
                })
        }
        _ => false,
    }
}

fn target_reads_name_as_value(name: &str, t: &ExprType) -> bool {
    match t {
        ExprType::Name(_) => false, // the store target itself: no read
        ExprType::Subscript(s) => {
            expr_reads_name_as_value(name, &s.value)
                || match &s.kind {
                    crate::SubscriptKind::Index(i) => expr_reads_name_as_value(name, i),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        [lower, upper, step]
                            .into_iter()
                            .flatten()
                            .any(|b| expr_reads_name_as_value(name, b))
                    }
                }
        }
        ExprType::Attribute(a) => expr_reads_name_as_value(name, &a.value),
        ExprType::Tuple(tuple) => tuple
            .elts
            .iter()
            .any(|e| target_reads_name_as_value(name, e)),
        _ => expr_reads_name_as_value(name, t),
    }
}

fn expr_reads_name_as_value(name: &str, e: &ExprType) -> bool {
    match e {
        ExprType::Name(n) => n.id == name,
        ExprType::Attribute(a) => expr_reads_name_as_value(name, &a.value),
        ExprType::Call(c) => {
            expr_reads_name_as_value(name, &c.func)
                || c.args.iter().any(|a| expr_reads_name_as_value(name, a))
                || c.keywords
                    .iter()
                    .any(|k| expr_reads_name_as_value(name, &k.value))
        }
        ExprType::BinOp(b) => {
            expr_reads_name_as_value(name, &b.left)
                || expr_reads_name_as_value(name, &b.right)
        }
        ExprType::BoolOp(b) => b
            .values
            .iter()
            .any(|v| expr_reads_name_as_value(name, v)),
        ExprType::Compare(c) => {
            // `x is None` / `x is not None` is a presence check, not a
            // value read — the Option<()> parameter's only legal use.
            let none_check = matches!(c.left.as_ref(), ExprType::Name(n) if n.id == name)
                && c.ops
                    .iter()
                    .zip(c.comparators.iter())
                    .all(|(op, rhs)| {
                        matches!(op, crate::Compares::Is | crate::Compares::IsNot)
                            && crate::is_none_expr(rhs)
                    });
            if none_check {
                return false;
            }
            expr_reads_name_as_value(name, &c.left)
                || c.comparators
                    .iter()
                    .any(|r| expr_reads_name_as_value(name, r))
        }
        ExprType::UnaryOp(u) => expr_reads_name_as_value(name, &u.operand),
        ExprType::IfExp(i) => {
            expr_reads_name_as_value(name, &i.test)
                || expr_reads_name_as_value(name, &i.body)
                || expr_reads_name_as_value(name, &i.orelse)
        }
        ExprType::Subscript(s) => {
            expr_reads_name_as_value(name, &s.value)
                || match &s.kind {
                    crate::SubscriptKind::Index(i) => expr_reads_name_as_value(name, i),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        [lower, upper, step]
                            .into_iter()
                            .flatten()
                            .any(|b| expr_reads_name_as_value(name, b))
                    }
                }
        }
        ExprType::Tuple(t) => t.elts.iter().any(|e| expr_reads_name_as_value(name, e)),
        ExprType::List(l) => l.iter().any(|e| expr_reads_name_as_value(name, e)),
        ExprType::Dict(d) => {
            d.keys
                .iter()
                .flatten()
                .any(|k| expr_reads_name_as_value(name, k))
                || d.values.iter().any(|v| expr_reads_name_as_value(name, v))
        }
        ExprType::Set(s) => s.elts.iter().any(|e| expr_reads_name_as_value(name, e)),
        ExprType::ListComp(lc) => {
            expr_reads_name_as_value(name, &lc.elt)
                || lc.generators
                    .iter()
                    .any(|g| expr_reads_name_as_value(name, &g.iter))
        }
        ExprType::Starred(s) => expr_reads_name_as_value(name, &s.value),
        ExprType::JoinedStr(js) => js
            .values
            .iter()
            .any(|v| expr_reads_name_as_value(name, v)),
        ExprType::FormattedValue(fv) => expr_reads_name_as_value(name, &fv.value),
        ExprType::Lambda(l) => expr_reads_name_as_value(name, &l.body),
        ExprType::Yield(y) => y
            .value
            .as_ref()
            .is_some_and(|v| expr_reads_name_as_value(name, v)),
        ExprType::YieldFrom(yf) => expr_reads_name_as_value(name, &yf.value),
        ExprType::Await(a) => expr_reads_name_as_value(name, &a.value),
        _ => false,
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
        // `del d[k]` reads the subscript's key (`for key in none_keys:
        // del merged_setting[key]` — requests' merge_setting): without
        // this arm the loop-target analysis declared `key` unused and
        // lowered the target to `_` while the body's `py_pop(key)` still
        // referenced it (E0425 in the generated crate).
        StatementType::Delete(targets) => {
            targets.iter().any(|t| expr_references(t, name))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, PythonOptions, StatementType, SymbolTableScopes};

    /// Parse `def f(x: <ann>): pass` and return the annotation ExprType.
    fn annotation_of(ann: &str) -> ExprType {
        let src = format!("def f(x: {ann}):\n    pass\n");
        let module = parse(&src, "ann.py").expect("parse failed");
        let StatementType::FunctionDef(f) = &module.raw.body[0].statement else {
            panic!("expected a function def");
        };
        f.args.args[0]
            .annotation
            .as_deref()
            .expect("annotation")
            .clone()
    }

    /// The single type authority (issue #137's systemic review of rounds
    /// 38–47): for every leaf annotation, the symbols-aware resolver with
    /// an EMPTY symbol table must agree with the syntax-only core — the
    /// alias/import layers may only ADD resolution on top, never change
    /// the leaf answer. This pins the one definition so a future arm added
    /// to one resolver but not the other is loud at the source, not in a
    /// corpus measurement several rounds later.
    #[test]
    fn alias_resolver_agrees_with_syntax_core_on_leaves() {
        let empty = SymbolTableScopes::new();
        let options = PythonOptions::default();
        for ann in [
            "int",
            "float",
            "bool",
            "str",
            "bytes",
            "bytearray",
            "bytes | bytearray",
            "str | bytes",
            "Any",
            "object",
            "Literal[3]",
            "list[int]",
            "List[str]",
            "set[int]",
            "Set[str]",
            "frozenset[int]",
            "dict[str, int]",
            "Dict[str, bool]",
            "tuple[int, str]",
            "tuple[int, ...]",
            "tuple[int]",
            "Mapping[str, int]",
            "typing.Mapping[str, int]",
            "typing.List[int]",
            "typing.Dict[str, int]",
            "typing.Tuple[int, str]",
            "typing.Set[int]",
            "Optional[int]",
            "typing.Optional[str]",
            "int | None",
            "None | str",
            "bool | str | None",
            "int | str",
            "threading.Event",
            "threading.Lock",
            "socket.socket",
            "type[BaseException]",
            "Type[BaseException]",
            "memoryview",
            "memoryview[int]",
        ] {
            let core = annotation_type_info(&annotation_of(ann));
            let alias = resolve_alias_typeinfo(&annotation_of(ann), &empty, &options);
            assert_eq!(
                alias, core,
                "resolver disagreement on `{ann}`: alias-aware gave {alias:?}, \
                 syntax core gave {core:?}",
            );
        }
    }

    /// The fixed drift points, pinned to the corpus-verified answers:
    /// `set[T]` is HashSet<T> (set literals generate HashSet — urllib3's
    /// PoolKey fields are Option<HashSet<(String, String)>>) and a 1-tuple
    /// renders with the trailing comma.
    #[test]
    fn set_and_one_tuple_render_corpus_verified_types() {
        let empty = SymbolTableScopes::new();
        let options = PythonOptions::default();
        let set = resolve_alias_typeinfo(&annotation_of("frozenset[tuple[str, str]]"), &empty, &options)
            .expect("frozenset must resolve");
        assert_eq!(
            set.to_rust_type().to_string().replace(' ', ""),
            "std::collections::HashSet<(String,String)>",
            "frozenset[tuple[str, str]] must be HashSet (PoolKey-verified)"
        );
        // `tuple[int]` (a 1-tuple ANNOTATION) was never supported by
        // either resolver (both require a tuple of elements); the 1-tuple
        // rendering bug lived in to_rust_type for TypeInfo::Tuple built
        // elsewhere (tuple-literal inference) — pinned directly.
        let one = TypeInfo::Tuple(vec![TypeInfo::Int]);
        assert_eq!(
            one.to_rust_type().to_string().replace(' ', ""),
            "(i64,)",
            "a 1-tuple must render with the trailing comma, not (i64)"
        );
        let class = annotation_type_info(&annotation_of("type[BaseException]"))
            .expect("type[X] resolves");
        assert_eq!(
            class.to_rust_type().to_string().replace(' ', ""),
            "Option<()>",
            "type[X] is the tolerated opaque class marker"
        );
    }
}

#[cfg(test)]
mod round81_token_format_tests {
    use super::*;
    #[test]
    fn token_format_strings() {
        let q = quote!(Vec<u8>);
        let s = q.to_string();
        eprintln!("Vec<u8> to_string = [{s}]");
        assert!(s.contains("u8"), "sanity");
    }
}
