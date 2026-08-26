//! The module defines Python-syntax arguments and maps them into Rust-syntax versions.
use proc_macro2::TokenStream;
use pyo3::{Borrowed, Bound, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{
    CodeGen, CodeGenContext, ExprType, Node, PythonOptions, SymbolTableScopes,
};

/// A complete argument representation that can hold any Python expression.
/// This replaces the limited Arg enum to support all argument types.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Argument {
    /// The argument expression (can be any valid Python expression)
    pub value: ExprType,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// An argument value that can be any expression.
/// This replaces the old limited Arg enum.
pub type Arg = ExprType;

/// A function parameter definition with optional type annotation and default value.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Parameter {
    /// Parameter name
    pub arg: String,
    /// Optional type annotation
    pub annotation: Option<Box<ExprType>>,
    /// Optional type comment (deprecated Python feature)
    pub type_comment: Option<String>,
    /// Position information
    pub lineno: Option<usize>,
    pub col_offset: Option<usize>,
    pub end_lineno: Option<usize>,
    pub end_col_offset: Option<usize>,
}

/// Comprehensive function arguments structure supporting all Python argument types.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Arguments {
    /// Positional-only parameters (before / in Python 3.8+)
    pub posonlyargs: Vec<Parameter>,
    /// Regular positional parameters
    pub args: Vec<Parameter>,
    /// Variable positional parameter (*args)
    pub vararg: Option<Parameter>,
    /// Keyword-only parameters (after * or *args)
    pub kwonlyargs: Vec<Parameter>,
    /// Default values for keyword-only parameters (None = required)
    pub kw_defaults: Vec<Option<Box<ExprType>>>,
    /// Variable keyword parameter (**kwargs)
    pub kwarg: Option<Parameter>,
    /// Default values for regular positional parameters
    pub defaults: Vec<Box<ExprType>>,
}


/// Function call arguments supporting all Python call patterns.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CallArguments {
    /// Positional arguments
    pub args: Vec<ExprType>,
    /// Keyword arguments
    pub keywords: Vec<crate::Keyword>,
}

// Implementation for new Argument struct
impl<'a, 'py> FromPyObject<'a, 'py> for Argument {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract the expression value
        let value: ExprType = ob.extract()?;
        
        Ok(Self {
            value,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

impl CodeGen for Argument {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        self.value.to_rust(ctx, options, symbols)
    }
}

// Implementation for Parameter struct
impl<'a, 'py> FromPyObject<'a, 'py> for Parameter {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let arg: String = ob.getattr("arg")?.extract()?;
        
        // Extract optional annotation
        let annotation = if let Ok(ann) = ob.getattr("annotation") {
            if ann.is_none() {
                None
            } else {
                Some(Box::new(ann.extract()?))
            }
        } else {
            None
        };
        
        // Extract optional type comment
        let type_comment = if let Ok(tc) = ob.getattr("type_comment") {
            if tc.is_none() {
                None
            } else {
                Some(tc.extract()?)
            }
        } else {
            None
        };
        
        Ok(Self {
            arg,
            annotation,
            type_comment,
            lineno: ob.lineno(),
            col_offset: ob.col_offset(),
            end_lineno: ob.end_lineno(),
            end_col_offset: ob.end_col_offset(),
        })
    }
}

/// Whether an annotation means "optional": `Optional[T]` or a union with
/// None (`T | None`). Optional-annotated names hold an Option, and stores
/// into them wrap in Some.
pub(crate) fn is_optional_annotation(ann: &ExprType) -> bool {
    match ann {
        ExprType::Subscript(sub) => {
            matches!(sub.value.as_ref(), ExprType::Name(n) if n.id == "Optional")
        }
        ExprType::BinOp(op) if matches!(op.op, crate::BinOps::BitOr) => {
            crate::is_none_expr(&op.left) || crate::is_none_expr(&op.right)
        }
        _ => false,
    }
}

/// Whether an annotation is the Python `str` type name.
pub fn is_str_annotation(ann: &ExprType) -> bool {
    matches!(ann, ExprType::Name(n) if n.id == "str")
}

/// Whether an annotation is a bytes-like type name (`bytes`/`bytearray`).
pub fn is_bytes_annotation(ann: &ExprType) -> bool {
    matches!(ann, ExprType::Name(n) if matches!(n.id.as_str(), "bytes" | "bytearray"))
}

/// Collect every member of a `|` union chain (left-associative in the
/// Python AST: `str | bytes | bytearray` is `(str | bytes) | bytearray`).
/// Returns the leaf members in order, or None when the expression is not a
/// union.
pub fn union_members(ann: &ExprType) -> Option<Vec<&ExprType>> {
    match ann {
        ExprType::BinOp(op) if matches!(op.op, crate::BinOps::BitOr) => {
            let mut members = union_members(&op.left)?;
            members.push(&op.right);
            Some(members)
        }
        _ => Some(vec![ann]),
    }
}

/// Whether a union annotation is exactly the supported StrOrBytes pair:
/// one `str` member and one-or-more bytes-like members (`str | bytes`,
/// `str | bytes | bytearray`), with no None (None makes it Optional).
pub fn is_str_bytes_union(ann: &ExprType) -> bool {
    let members = match union_members(ann) {
        Some(m) if m.len() >= 2 => m,
        _ => return false,
    };
    if members.iter().any(|m| crate::is_none_expr(m)) {
        return false;
    }
    let has_str = members.iter().any(|m| is_str_annotation(m));
    let has_bytes = members.iter().any(|m| is_bytes_annotation(m));
    let all_known = members.iter().all(|m| is_str_annotation(m) || is_bytes_annotation(m));
    has_str && has_bytes && all_known
}

/// Whether a union member can live inside the boxed PyValue: the primitive
/// value types, tuples of them, Literal constants, Any, or None. Used by
/// `python_annotation_to_rust_type` to decide when a wider union maps to
/// the boxed heterogeneous value.
pub fn is_pyvalue_boxable_member(ann: &ExprType) -> bool {
    if crate::is_none_expr(ann) {
        return true;
    }
    match ann {
        ExprType::Name(n) => {
            matches!(
                n.id.as_str(),
                "int" | "float" | "str" | "bool" | "bytes" | "bytearray" | "Any" | "memoryview"
                    // PathLike (os.PathLike) unions like `str | bytes |
                    // PathLike`: only the str/bytes members are real values in
                    // rython; the member is tolerated so file paths flow
                    // through the boxed PyValue (AsStrLike).
                    | "PathLike"
                    | "BinaryIO"
                    // types-module classes (`TracebackType | None` — the
                    // context-manager protocol): boxed values.
                    | "TracebackType" | "FrameType" | "CodeType"
            )
                // Builtin exception names (`BaseException | None`):
                // exceptions are boxed values (PyException), so a union
                // with one is the boxed PyValue. The canonical list lives
                // with the raise lowering.
                || crate::ast::tree::raise_stmt::is_builtin_exception_name(&n.id)
        }
        ExprType::Subscript(sub) => {
            match sub.value.as_ref() {
                ExprType::Name(n) => matches!(
                    n.id.as_str(),
                    "tuple" | "Tuple" | "Literal" | "list" | "List" | "IO" | "Iterable"
                        | "Union" | "Callable" | "SupportsRead" | "SupportsItems"
                        | "Mapping" | "Dict" | "Set" | "Sequence" | "MutableMapping" | "Collection" | "Container"
                        | "Generator" | "Iterator" | "Type" | "Optional" | "Any"
                        | "memoryview"
                ),
                // `typing.Sequence[...]` / `typing.Iterable[...]` — the
                // typing-module spelling of the same generics (urllib3's
                // `dict[str, T] | typing.Sequence[tuple[str, T]]`).
                ExprType::Attribute(a) => {
                    matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
                        && matches!(
                            a.attr.as_str(),
                            "Tuple" | "List" | "Dict" | "Set" | "Sequence" | "Iterable"
                                | "Iterator" | "Generator" | "Mapping" | "MutableMapping"
                                | "Callable" | "Union" | "Optional" | "Literal" | "Any"
                                | "IO" | "SupportsRead" | "SupportsItems" | "Type" | "Collection" | "Container"
                        )
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Map a Python type annotation to a Rust type, when the mapping is known.
/// `int`/`float`/`str`/`bool`/`bytes` map to concrete Rust types, and
/// `list[T]`/`dict[K, V]`/`set[T]` map to the corresponding std containers
/// when their element annotations map too. `Optional[T]` / `T | None` map
/// to `Option<T>`.
/// Whether an annotation is the bare `type` marker — a CALLABLE/class
/// parameter (`dict_class: type = OrderedDict` — requests' sessions).
/// rython cannot hold callables as values (the callables-as-data
/// divergence): the parameter lowers to a boxed PyValue, its arguments
/// lower to the boxed None, and calls through it drop (function_def.rs /
/// map_call_arguments / Parameter::to_rust).
pub(crate) fn is_type_annotation(annotation: &ExprType) -> bool {
    matches!(annotation, ExprType::Name(n) if n.id == "type")
}

pub fn python_annotation_to_rust_type(annotation: &ExprType) -> Option<TokenStream> {    match annotation {
        // T | None (and None | T) is Option<T>; a union whose members map
        // to the SAME Rust type (`bytes | bytearray` → Vec<u8>) is that
        // type. `str | bytes` (and `str | bytes | bytearray`) is the
        // StrOrBytes heterogeneous union (issue #121). Any other union has
        // no single Rust type — the caller reports the unsupported
        // annotation.
        ExprType::BinOp(op) if matches!(op.op, crate::BinOps::BitOr) => {
            let inner = if crate::is_none_expr(&op.left) {
                op.right.as_ref()
            } else if crate::is_none_expr(&op.right) {
                op.left.as_ref()
            } else {
                // All members map to the same Rust type?
                let members = crate::union_members(annotation).unwrap_or_default();
                let mut all_same: Option<TokenStream> = None;
                let mut same = true;
                for m in &members {
                    match python_annotation_to_rust_type(m) {
                        Some(t) if all_same.is_none() => all_same = Some(t),
                        Some(t) if all_same.as_ref().is_some_and(|p| p.to_string() == t.to_string()) => {}
                        _ => {
                            same = false;
                            break;
                        }
                    }
                }
                if same && let Some(t) = all_same {
                    return Some(t);
                }
                // str | bytes — the StrOrBytes heterogeneous union.
                if crate::is_str_bytes_union(annotation) {
                    return Some(quote!(stdpython::StrOrBytes));
                }
                // Any other union whose members are all boxable (issue
                // #121: `bool | str | None`, `tuple[str, str] | str |
                // None`, `int | str | None`, ...) is the boxed
                // heterogeneous value; isinstance narrows at runtime.
                if members
                    .iter()
                    .all(|m| crate::is_pyvalue_boxable_member(m))
                {
                    return Some(quote!(stdpython::PyValue));
                }
                return None;
            };
            // `T | None` where the inner union maps to one type is
            // Option<T>. A wider inner union maps to the boxed PyValue,
            // which ALREADY contains None (`bool | str | None` is PyValue,
            // not Option<PyValue>).
            let inner_tokens = python_annotation_to_rust_type(inner);
            if let Some(t) = inner_tokens {
                if t.to_string() == quote!(stdpython::PyValue).to_string() {
                    return Some(t);
                }
                return Some(quote!(Option<#t>));
            }
            // `str | bytes | None` → Option<StrOrBytes>.
            if crate::is_str_bytes_union(inner) {
                return Some(quote!(Option<stdpython::StrOrBytes>));
            }
            if crate::union_members(inner).is_some_and(|ms| {
                !ms.is_empty() && ms.iter().all(|m| crate::is_pyvalue_boxable_member(m))
            }) {
                return Some(quote!(stdpython::PyValue));
            }
            return None;
        }
        _ => {}
    }
    match annotation {
        ExprType::Name(name) => match name.id.as_str() {
            "int" => Some(quote!(i64)),
            "float" => Some(quote!(f64)),
            "str" => Some(quote!(String)),
            "bool" => Some(quote!(bool)),
            "bytes" | "bytearray" => Some(quote!(Vec<u8>)),
            // `Any` (typing.Any) and `object`: a value of unknown type —
            // the boxed heterogeneous value.
            "Any" | "object" => Some(quote!(stdpython::PyValue)),
            _ => None,
        },
        // numpy scalar type annotations: np.float64 → f64, np.int32 → i32,
        // ... (and ndarray → the numpy NdArray type). threading/socket
        // object annotations map to their runtime types, so a worker
        // parameter (`ready: threading.Event`, `srv: socket.socket`) is a
        // real shared handle rather than a boxed PyValue.
        ExprType::Attribute(attr) => {
            if let ExprType::Name(n) = attr.value.as_ref() {
                if n.id == "threading" {
                    return crate::ThreadingType::from_name(&attr.attr).map(|t| t.rust_path());
                }
                if n.id == "socket" && attr.attr == "socket" {
                    return Some(quote!(socket::Socket));
                }
            }
            let is_np = matches!(attr.value.as_ref(), ExprType::Name(n) if crate::is_numpy_alias(&n.id));
            if !is_np {
                return None;
            }
            match attr.attr.as_str() {
                "ndarray" => Some(quote!(numpy::NdArray)),
                "float64" => Some(quote!(f64)),
                "float32" => Some(quote!(f32)),
                "int64" => Some(quote!(i64)),
                "int32" => Some(quote!(i32)),
                "bool_" => Some(quote!(bool)),
                _ => None,
            }
        }
        // Subscripted generics over known element types: list[int] and
        // friends map to the concrete Rust containers codegen produces for
        // the corresponding literals.
        ExprType::Subscript(sub) => {
            let container = match sub.value.as_ref() {
                ExprType::Name(n) => n.id.as_str(),
                // `typing.Mapping[K, V]` — the typing-prefixed generic
                // lowers like the bare name.
                ExprType::Attribute(a)
                    if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing") =>
                {
                    match a.attr.as_str() {
                        "Mapping" => "Mapping",
                        "Dict" => "dict",
                        "List" => "list",
                        "Set" => "set",
                        "Optional" => "Optional",
                        "Tuple" => "tuple",
                        "Literal" => "Literal",
                        _ => return None,
                    }
                }
                _ => return None,
            };
            match (&sub.kind, container) {
                // `type[X]` / `Type[X]` is a CLASS, which rython cannot
                // pass as a value — tolerated as an opaque Option<()> so
                // the definition compiles (a call passing an actual class
                // is the documented class-as-value divergence).
                (crate::SubscriptKind::Index(_), "type" | "Type") => {
                    Some(quote!(Option<()>))
                }
                (crate::SubscriptKind::Index(elt), "Optional") => {
                    let inner = python_annotation_to_rust_type(elt)?;
                    // Optional[bool | str] is the boxed PyValue (which
                    // already contains None), not Option<PyValue>.
                    if inner.to_string() == quote!(stdpython::PyValue).to_string() {
                        return Some(inner);
                    }
                    Some(quote!(Option<#inner>))
                }
                // `tuple[T1, T2, ...]` maps to a Rust tuple; `tuple[T, ...]`
                // (a variadic tuple) maps to Vec<T>; `Literal[X]` (a
                // constant union member) is a boxed PyValue.
                (crate::SubscriptKind::Index(elt), "tuple" | "Tuple") => {
                    if let ExprType::Tuple(t) = elt.as_ref() {
                        // `tuple[int, ...]`: one element + Ellipsis = a
                        // variadic tuple → Vec<T>.
                        if t.elts.len() == 2
                            && matches!(
                                &t.elts[1],
                                ExprType::Constant(c)
                                    if c.0
                                        .as_ref()
                                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                            )
                        {
                            let inner = python_annotation_to_rust_type(&t.elts[0])?;
                            return Some(quote!(Vec<#inner>));
                        }
                        let mut tys = Vec::with_capacity(t.elts.len());
                        for e in &t.elts {
                            tys.push(python_annotation_to_rust_type(e)?);
                        }
                        if tys.len() == 1 {
                            let only = &tys[0];
                            Some(quote!((#only,)))
                        } else {
                            Some(quote!((#(#tys),*)))
                        }
                    } else {
                        None
                    }
                }
                (_, "Literal") => Some(quote!(stdpython::PyValue)),
                (crate::SubscriptKind::Index(elt), "list") => {
                    let inner = python_annotation_to_rust_type(elt)?;
                    Some(quote!(Vec<#inner>))
                }
                (crate::SubscriptKind::Index(elt), "set" | "frozenset") => {
                    let inner = python_annotation_to_rust_type(elt)?;
                    Some(quote!(std::collections::HashSet<#inner>))
                }
                (crate::SubscriptKind::Index(kv), "dict") => {
                    // dict[K, V] parses as a subscript with a tuple index.
                    // PyDict is the insertion-ordered map dict literals
                    // lower to.
                    if let ExprType::Tuple(t) = kv.as_ref() {
                        if let [k, v] = t.elts.as_slice() {
                            let k = python_annotation_to_rust_type(k)?;
                            let v = python_annotation_to_rust_type(v)?;
                            return Some(quote!(PyDict<#k, #v>));
                        }
                    }
                    None
                }
                // `typing.Mapping[K, V]` / `Mapping[K, V]` (a read-only
                // dict view in Python) lowers to PyDict — dict literals and
                // dict.get/contains all work unchanged.
                (crate::SubscriptKind::Index(kv), "Mapping") => {
                    if let ExprType::Tuple(t) = kv.as_ref() {
                        if let [k, v] = t.elts.as_slice() {
                            let k = python_annotation_to_rust_type(k)?;
                            let v = python_annotation_to_rust_type(v)?;
                            return Some(quote!(PyDict<#k, #v>));
                        }
                    }
                    None
                }
                _ => None,
            }
        }
        _ => None,
    }
}

impl CodeGen for Parameter {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {

        let param_name = crate::safe_ident(&self.arg);

        // Generate type annotation if present
        if let Some(annotation) = self.annotation {
            // A str parameter accepts anything convertible to String, so
            // call sites can pass &str literals as well as owned Strings;
            // the function prologue converts it (`let s: String = s.into()`).
            if matches!(&*annotation, ExprType::Name(n) if n.id == "str") {
                return Ok(quote!(#param_name: impl Into<String>));
            }
            // A bare `type` annotation (`dict_class: type = OrderedDict` —
            // requests' sessions): a callable/class — rython cannot hold
            // callables as values (the callables-as-data divergence), so
            // the parameter is a boxed PyValue.
            if crate::ast::tree::arguments::is_type_annotation(&annotation) {
                return Ok(quote!(#param_name: stdpython::PyValue));
            }
            // A `None`-only annotation (`cookiejar: None = None`): nothing
            // but None can ever be stored.
            if crate::is_none_expr(&annotation) {
                return Ok(quote!(#param_name: Option<()>));
            }
            // Known Python types map to concrete Rust types; a module-level
            // TYPE ALIAS (`CoherenceMatches = List[CoherenceMatch]`) or an
            // alias in another module resolves through symbols
            // (charset_normalizer). Anything else falls back to rendering
            // the annotation expression (e.g. a user-defined class name).
            let rust_type = match python_annotation_to_rust_type(&annotation) {
                Some(mapped) => mapped,
                None => {
                    if let Some(t) = crate::resolve_alias_typeinfo(&annotation, &symbols, &options)
                    {
                        t.to_rust_type()
                    } else {
                        annotation.to_rust(ctx, options, symbols)?
                    }
                }
            };
            Ok(quote!(#param_name: #rust_type))
        } else {
            // An unannotated parameter: the per-function inference pass
            // (issue #109, M1) gives it a type-variable name from its uses
            // (`def add(a, b): return a + b` → `a: A`). The old
            // `impl Into<PyObject>` fallback is gone: no ordinary rython
            // value satisfies it, so such functions converted but were
            // uncallable. If no variable was inferred, the function
            // generator already failed loudly with the reason.
            match options.param_type_vars.get(&self.arg) {
                // A value-pinned free-function parameter (inferred boxed
                // PyValue — issue #161): `impl Into<stdpython::PyValue>`,
                // boxed by the function prologue, so call sites pass plain
                // values (String, bytes, an already-boxed PyValue) exactly
                // like Python.
                Some(_) if options.pyvalue_into_params.contains(&self.arg) => {
                    Ok(quote!(#param_name: impl Into<stdpython::PyValue>))
                }
                Some(tv) => Ok(quote!(#param_name: #tv)),
                // No type var (the constructor synthesis renders __init__
                // params, or an unannotated method param): a boxed PyValue
                // fallback — the parameter's value is unknown (documented
                // divergence, issue #109).
                None => Ok(quote!(#param_name: stdpython::PyValue)),
            }
        }
    }
}

// Implementation for Arguments struct
impl<'a, 'py> FromPyObject<'a, 'py> for Arguments {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Extract each field with proper error handling
        let posonlyargs: Vec<Parameter> = ob.getattr("posonlyargs")?.extract().unwrap_or_default();
        let args: Vec<Parameter> = ob.getattr("args")?.extract().unwrap_or_default();
        
        let vararg = if let Ok(va) = ob.getattr("vararg") {
            if va.is_none() { None } else { Some(va.extract()?) }
        } else { None };
        
        let kwonlyargs: Vec<Parameter> = ob.getattr("kwonlyargs")?.extract().unwrap_or_default();
        
        // Handle kw_defaults which can contain None values
        let kw_defaults = if let Ok(kw_def) = ob.getattr("kw_defaults") {
            let defaults_list: Vec<Bound<PyAny>> = kw_def.extract().unwrap_or_default();
            let mut processed_defaults = Vec::new();
            for default in defaults_list {
                if default.is_none() {
                    processed_defaults.push(None);
                } else {
                    processed_defaults.push(Some(Box::new(default.extract()?)));
                }
            }
            processed_defaults
        } else {
            Vec::new()
        };
        
        let kwarg = if let Ok(kw) = ob.getattr("kwarg") {
            if kw.is_none() { None } else { Some(kw.extract()?) }
        } else { None };
        
        let defaults_raw: Vec<ExprType> = ob.getattr("defaults")?.extract().unwrap_or_default();
        let defaults = defaults_raw.into_iter().map(Box::new).collect();
        
        Ok(Self {
            posonlyargs,
            args,
            vararg,
            kwonlyargs,
            kw_defaults,
            kwarg,
            defaults,
        })
    }
}

impl CodeGen for Arguments {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        let mut params = Vec::new();
        
        // Process positional-only arguments
        for arg in self.posonlyargs {
            let param = arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            params.push(param);
        }
        
        // Process regular positional arguments. Defaulted parameters lower
        // to plain required parameters: Rust has no default arguments, and
        // the old Option<T> wrapping neither type-checked against bodies
        // that use the parameter directly nor matched call sites (which
        // never wrapped values in Some). Callers that omit the argument
        // fail to compile either way; callers that pass it now work.
        for arg in self.args {
            let param = arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            params.push(param);
        }
        
        // Process *args (issue #120): a boxed heterogeneous list —
        // callers pack extra positional arguments into PyValue::from
        // values (mirroring **kwargs below). The body reads it like any
        // list: len/index/iterate yield PyValue, and `callee(*args)`
        // forwards the vector.
        if let Some(vararg) = self.vararg {
            let vararg_name = crate::safe_ident(&vararg.arg);
            params.push(quote!(#vararg_name: Vec<stdpython::PyValue>));
        }
        
        // Process keyword-only arguments. Like positional defaults above,
        // these lower to plain required parameters.
        for arg in self.kwonlyargs {
            let param = arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            params.push(param);
        }
        
        // Process **kwargs (issue #120): a boxed heterogeneous dict —
        // callers pack extra keyword arguments into PyValue::from values.
        if let Some(kwarg) = self.kwarg {
            let kwarg_name = crate::safe_ident(&kwarg.arg);
            params.push(quote!(#kwarg_name: PyDict<String, stdpython::PyValue>));
        }
        
        Ok(quote!(#(#params),*))
    }
}


// Implementation for CallArguments
impl<'a, 'py> FromPyObject<'a, 'py> for CallArguments {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let args: Vec<ExprType> = ob.getattr("args")?.extract().unwrap_or_default();
        let keywords: Vec<crate::Keyword> = ob.getattr("keywords")?.extract().unwrap_or_default();
        
        Ok(Self { args, keywords })
    }
}

impl CodeGen for CallArguments {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> std::result::Result<TokenStream, Box<dyn std::error::Error>> {
        let mut all_args = Vec::new();
        
        // Add positional arguments
        for arg in self.args {
            let rust_arg = arg.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_arg);
        }
        
        // Add keyword arguments
        for keyword in self.keywords {
            let rust_kw = keyword.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            all_args.push(rust_kw);
        }
        
        Ok(quote!(#(#all_args),*))
    }
}


// Node trait implementations for position tracking
impl Node for Argument {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}

impl Node for Parameter {
    fn lineno(&self) -> Option<usize> { self.lineno }
    fn col_offset(&self) -> Option<usize> { self.col_offset }
    fn end_lineno(&self) -> Option<usize> { self.end_lineno }
    fn end_col_offset(&self) -> Option<usize> { self.end_col_offset }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, CodeGenContext, ExprType, PythonOptions, SymbolTableScopes};
    use test_log::test;

    #[test]
    fn test_simple_function_call() {
        let code = "func(1, 2, 3)";
        let result = parse(code, "test.py").unwrap();
        
        // Generate Rust code
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function call with positional arguments
    }

    #[test]
    fn test_keyword_arguments() {
        // Keywords resolve against the callee's signature and land in
        // parameter order.
        let code = "def func(a, b):\n    pass\n\nfunc(b=2, a=1)";
        let result = parse(code, "test.py").unwrap();

        let options = PythonOptions::default();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap().to_string();
        assert!(rust_code.contains("let __rython_arg_0 = 2 ; let __rython_arg_1 = 1 ; func (__rython_arg_1 , __rython_arg_0)"), "generated: {}", rust_code);
    }

    #[test]
    fn test_mixed_arguments() {
        let code = "def func(a, b, c, d):\n    pass\n\nfunc(1, 2, d=4, c=3)";
        let result = parse(code, "test.py").unwrap();

        let options = PythonOptions::default();
        let symbols = result.clone().find_symbols(SymbolTableScopes::new());
        let rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap().to_string();
        assert!(rust_code.contains("let __rython_arg_0 = 1 ; let __rython_arg_1 = 2 ; let __rython_arg_2 = 4 ; let __rython_arg_3 = 3 ; func (__rython_arg_0 , __rython_arg_1 , __rython_arg_3 , __rython_arg_2)"), "generated: {}", rust_code);
    }

    #[test]
    fn test_function_with_defaults() {
        let code = r#"
def func(a, b=2, c=3):
    pass
        "#;
        let result = parse(code, "test.py").unwrap();
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function with optional parameters
    }

    #[test]
    fn test_function_with_varargs() {
        let code = r#"
def func(a, *args):
    pass
        "#;
        let result = parse(code, "test.py").unwrap();
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function with variable arguments
    }

    #[test]
    fn test_function_with_kwargs() {
        let code = r#"
def func(a, **kwargs):
    pass
        "#;
        let result = parse(code, "test.py").unwrap();
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function with keyword arguments dict
    }

    #[test]
    fn test_complex_function_signature() {
        let code = r#"
def func(a, b=2, *args, c, d=4, **kwargs):
    pass
        "#;
        let result = parse(code, "test.py").unwrap();
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function with all argument types
    }

    #[test]
    fn test_keyword_only_arguments() {
        let code = r#"
def func(a, *, b, c=3):
    pass
        "#;
        let result = parse(code, "test.py").unwrap();
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let _rust_code = result.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        // Should generate function with keyword-only arguments
    }

    #[test]
    fn test_argument_unpacking_call() {
        // Note: This would require additional AST node support for Starred expressions
        let code = "func(*args, **kwargs)";
        let result = parse(code, "test.py");
        
        match result {
            Ok(ast) => {
                let options = PythonOptions::default();
                let symbols = SymbolTableScopes::new();
                let rust_code = ast.to_rust(
                    CodeGenContext::Module("test".to_string()),
                    options,
                    symbols,
                );
                
                match rust_code {
                    Ok(_code) => { /* Code generation succeeded as expected */ },
                    Err(_e) => { /* Expected error for unimplemented feature */ },
                }
            }
            Err(_e) => { /* Parse error expected for unimplemented features */ },
        }
    }

    #[test]
    fn test_arg_with_constant() {
        // Test that Arg (now ExprType) works with constants
        use litrs::Literal;
        let literal = Literal::parse("42").unwrap().into_owned();
        let constant = crate::Constant(Some(literal));
        let arg: Arg = ExprType::Constant(constant);
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let rust_code = arg.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        assert!(rust_code.to_string().contains("42"));
    }

    #[test]
    fn test_arg_with_name() {
        // Test that Arg (now ExprType) works with name expressions
        let name_expr = ExprType::Name(crate::Name {
            id: "variable".to_string(),
        });
        let arg: Arg = name_expr;
        
        let options = PythonOptions::default();
        let symbols = SymbolTableScopes::new();
        let rust_code = arg.to_rust(
            CodeGenContext::Module("test".to_string()),
            options,
            symbols,
        ).unwrap();
        
        assert!(rust_code.to_string().contains("variable"));
    }
}
