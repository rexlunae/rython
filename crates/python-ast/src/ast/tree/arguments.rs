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
    /// Optional type annotation — EVALUATED: a quoted annotation is the
    /// expression it spells (the parser bridge evaluates it once).
    pub annotation: Option<Box<ExprType>>,
    /// The ORIGINAL text of a quoted annotation, for the one reader that
    /// wants the text rather than the expression: the Rust stub loader
    /// (`.pyi` stubs spell Rust types as strings — `"&[u8]"`, `"u32"`).
    #[serde(default)]
    pub quoted_source: Option<String>,
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
        
        let mut quoted_source = None;
        // Extract optional annotation. A QUOTED annotation (`err:
        // "Optional[MyError]"`, `x: "str"`) is evaluated HERE, at the
        // parser bridge — the one place every reader of a parameter's
        // annotation inherits from (the signature, the body's name
        // types, the local type pass, optional-name seeding, call sites,
        // field inference, dunder routing), so no consumer can see the
        // raw string (Devin review on #342, rounds 4 and 7).
        let annotation = if let Ok(ann) = ob.getattr("annotation") {
            if ann.is_none() {
                None
            } else {
                let raw: ExprType = ann.extract()?;
                quoted_source = quoted_annotation_text(&raw);
                Some(Box::new(unquote_annotation(&raw).unwrap_or(raw)))
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
            quoted_source,
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
            // `Optional[...]` and `typing.Optional[...]` (the latter from
            // NamedTuple call-form fields — urllib3's Url).
            matches!(sub.value.as_ref(), ExprType::Name(n) if n.id == "Optional")
                || matches!(sub.value.as_ref(), ExprType::Attribute(a)
                    if a.attr == "Optional"
                        && matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id)))
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
                    matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
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

/// A STRING-LITERAL annotation (`verify: "bool | str | None"` — requests'
/// adapters.py writes its annotations as quoted strings; CPython's
/// `typing.get_type_hints` evaluates them): re-parse the string's content
/// as a Python expression so the annotation authorities (type_ctx's
/// `annotation_type_info`, Parameter::to_rust) see the real union instead
/// of a bare string Constant. Returns None when the annotation is not a
/// string literal or the content cannot be parsed (round 56).
impl Parameter {
    /// The parameter's annotation as EVALUATED: a quoted annotation
    /// (`err: "Optional[MyError]"`) is the expression it spells. The
    /// parser bridge (`FromPyObject for Parameter`) evaluates it once
    /// at construction, so `annotation` already holds the expression
    /// and every reader agrees; this accessor names that contract for
    /// the readers that want it spelled out, and evaluates again only
    /// for a Parameter built elsewhere (the class synthesizers) with a
    /// quoted form (Devin review on #342, rounds 4 and 7).
    pub(crate) fn evaluated_annotation(&self) -> Option<ExprType> {
        let ann = self.annotation.as_deref()?;
        Some(unquote_annotation(ann).unwrap_or_else(|| ann.clone()))
    }
}

/// The text of a string-literal annotation (`"Optional[MyError]"`), None
/// for any other annotation.
pub(crate) fn quoted_annotation_text(annotation: &ExprType) -> Option<String> {
    let ExprType::Constant(c) = annotation else {
        return None;
    };
    let Some(litrs::Literal::String(s)) = &c.0 else {
        return None;
    };
    Some(s.value().to_string())
}

pub(crate) fn unquote_annotation(annotation: &ExprType) -> Option<ExprType> {
    let ExprType::Constant(c) = annotation else {
        return None;
    };
    let Some(litrs::Literal::String(s)) = &c.0 else {
        return None;
    };
    let text = s.value();
    // `typing.Tuple[str, str] | str | None` parses as one expression. A
    // bare `x: T` statement carries the annotation on AnnotatedName.
    let module = crate::parse(&format!("x: {text}\n"), "<annotation>").ok()?;
    let body = &module.raw.body;
    let stmt = body.first()?;
    let crate::StatementType::AnnotatedName { annotation, .. } = &stmt.statement else {
        return None;
    };
    Some(annotation.clone())
}

/// Whether a union member names an exception class: a builtin exception
/// name, an imported stdlib alias (`SocketTimeout`), a class of the crate
/// the exception closure holds (`is_exception_class` — the one C3
/// authority: its ancestry through the crate's classes, the builtin
/// exceptions and the documented `*Error`/`*Exception`/`*Warning`
/// convention, §8.1), or a name the conversion cannot resolve at all that
/// follows the convention (an external `BaseSSLError` — the raise model's
/// rule for an unknown name).
pub(crate) fn is_exception_class_member(
    member: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    is_exception_class_member_within(member, symbols, options, &mut Vec::new())
}

/// `is_exception_class_member` with the names already followed through
/// `Assign`/`Alias` hops: a cycle (`A = B; B = A` — Devin review on
/// #342, round 7) ends the walk as "not an exception class" instead of
/// overflowing the stack, the same closure the alias resolvers keep.
fn is_exception_class_member_within(
    member: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    followed: &mut Vec<String>,
) -> bool {
    match member {
        ExprType::Name(n) => {
            if followed.iter().any(|seen| seen == &n.id) {
                return false;
            }
            followed.push(n.id.clone());
            if let Some((class, _)) =
                crate::ast::tree::call::resolve_construction_class(&n.id, symbols, options)
            {
                return crate::is_exception_class(&class);
            }
            // A name BOUND in scope as a value (`NetworkError = 1`) is
            // that value, whatever its spelling: the naming convention
            // applies to genuinely unresolved names only (Devin review
            // on #342, round 6). A binding to another name or attribute
            // (`NetworkError = requests.ConnectionError`) is judged by
            // what it names.
            match symbols.get(&n.id) {
                Some(crate::SymbolTableNode::Assign { value, .. }) => {
                    return matches!(value, ExprType::Name(_) | ExprType::Attribute(_))
                        && !crate::expr_references(value, &n.id)
                        && is_exception_class_member_within(value, symbols, options, followed);
                }
                Some(crate::SymbolTableNode::Alias(target)) if target != &n.id => {
                    let target = ExprType::Name(crate::Name { id: target.clone() });
                    return is_exception_class_member_within(&target, symbols, options, followed);
                }
                _ => {}
            }
            crate::ast::tree::raise_stmt::is_builtin_exception_name(&n.id)
                || crate::ast::tree::raise_stmt::imported_exception_alias(
                    &n.id,
                    symbols,
                    Some(options),
                )
                .is_some()
                || crate::ast::tree::raise_stmt::is_exception_class_name(&n.id)
        }
        ExprType::Attribute(a) => {
            let ExprType::Name(m) = a.value.as_ref() else {
                return false;
            };
            crate::ast::tree::raise_stmt::stdlib_exception_canonical(&m.id, &a.attr).is_some()
        }
        _ => false,
    }
}

/// The members of a union annotation in ANY supported spelling — PEP 604
/// `A | B | None`, `Union[A, B]` / `typing.Union[A, B]`, `Optional[A]` /
/// `typing.Optional[A]` (also nested: `Optional[Union[A, B]]`) — as the
/// non-None members plus whether None is a member. None for anything
/// that is not a union (a bare name, a container subscript).
pub(crate) fn union_annotation_members(ann: &ExprType) -> Option<(Vec<&ExprType>, bool)> {
    match ann {
        ExprType::BinOp(op) if matches!(op.op, crate::BinOps::BitOr) => {
            let mut members = Vec::new();
            let mut has_none = false;
            for m in union_members(ann)? {
                if crate::is_none_expr(m) {
                    has_none = true;
                } else if let Some((inner, inner_none)) = union_annotation_members(m) {
                    members.extend(inner);
                    has_none |= inner_none;
                } else {
                    members.push(m);
                }
            }
            Some((members, has_none))
        }
        ExprType::Subscript(sub) => {
            let container = match sub.value.as_ref() {
                ExprType::Name(n) => n.id.as_str(),
                ExprType::Attribute(a)
                    if matches!(a.value.as_ref(), ExprType::Name(m) if crate::is_typing(&m.id)) =>
                {
                    a.attr.as_str()
                }
                _ => return None,
            };
            let crate::SubscriptKind::Index(index) = &sub.kind else {
                return None;
            };
            let elements: Vec<&ExprType> = match (container, index.as_ref()) {
                ("Union", ExprType::Tuple(t)) => t.elts.iter().collect(),
                ("Union", single) => vec![single],
                ("Optional", single) => vec![single],
                _ => return None,
            };
            let mut members = Vec::new();
            let mut has_none = container == "Optional";
            for m in elements {
                if crate::is_none_expr(m) {
                    has_none = true;
                } else if let Some((inner, inner_none)) = union_annotation_members(m) {
                    members.extend(inner);
                    has_none |= inner_none;
                } else {
                    members.push(m);
                }
            }
            Some((members, has_none))
        }
        _ => None,
    }
}

/// The type of a union annotation (any spelling `union_annotation_members`
/// reads) whose members are ALL exception classes (optionally with None):
/// the runtime's one exception type, `PyException`, or
/// `Option<PyException>` with a None member. Any other union (a boxable or
/// class member, or no exception member) is None. ONE rule for the
/// signature (Parameter::to_rust), the body's name types
/// (function_def.rs) and the alias resolver, so a parameter's reads see
/// the type its signature declares.
pub(crate) fn exception_union_typeinfo(
    annotation: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<crate::TypeInfo> {
    let (members, has_none) = union_annotation_members(annotation)?;
    if members.is_empty()
        || !members
            .iter()
            .all(|m| is_exception_class_member(m, symbols, options))
    {
        return None;
    }
    Some(exception_typeinfo(has_none))
}

/// The runtime's exception type as a TypeInfo: `PyException`, or
/// `Option<PyException>` for a None-able slot.
pub(crate) fn exception_typeinfo(optional: bool) -> crate::TypeInfo {
    let exception = crate::TypeInfo::Custom(quote!(PyException));
    if optional {
        crate::TypeInfo::Option(Box::new(exception))
    } else {
        exception
    }
}

/// Whether a TypeInfo is the runtime's exception type (the `Custom`
/// payload the exception-union rule produces).
pub(crate) fn is_exception_typeinfo(t: &crate::TypeInfo) -> bool {
    matches!(t, crate::TypeInfo::Custom(tokens) if tokens.to_string() == "PyException")
}

pub fn python_annotation_to_rust_type(annotation: &ExprType) -> Option<TokenStream> {
    // ONE annotation authority (issue #137's systemic review of rounds
    // 38–47): the leaf mapping lives in `annotation_type_info` (with the
    // symbols-aware alias/import resolution layered on top in
    // `resolve_alias_typeinfo`), and this function is a thin wrapper —
    // `TypeInfo.to_rust_type()` is the only path from a type to tokens.
    // The previous body duplicated the union/subscript/name mapping as
    // tokens and drifted from the TypeInfo side (`set[T]` was Vec<T> here
    // vs HashSet<T> there; `tuple[int]` rendered `(i64)` instead of
    // `(i64,)`). The generated structs are the arbiter, and they compile
    // as the TypeInfo answers.
    crate::annotation_type_info(annotation).map(|t| t.to_rust_type())
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
            // A STRING-LITERAL annotation (`verify: "bool | str | None"` —
            // requests' adapters.py quotes its annotations): re-parse the
            // content so the checks below see the real expression (round
            // 56), like typing.get_type_hints.
            let annotation: ExprType =
                unquote_annotation(&annotation).unwrap_or(*annotation);
            // A str parameter accepts anything convertible to String, so
            // call sites can pass &str literals as well as owned Strings;
            // the function prologue converts it (`let s: String = s.into()`).
            if matches!(&annotation, ExprType::Name(n) if n.id == "str") {
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
            // A union of exception classes only (`err: BaseSSLError |
            // OSError | SocketTimeout` — urllib3's _raise_timeout): the
            // exception model has one runtime type, so the parameter IS a
            // PyException (an Option of one with a None member) — a caught
            // exception passes straight in, and `isinstance(err, X)` tests
            // its kind like an except clause. Decided FIRST: the
            // syntax-only mapping boxes a builtin exception member.
            if let Some(t) = exception_union_typeinfo(&annotation, &symbols, &options) {
                let rust_type = t.to_rust_type();
                return Ok(quote!(#param_name: #rust_type));
            }
            let rust_type = match python_annotation_to_rust_type(&annotation) {
                Some(mapped) => mapped,
                None => {
                    if let Some(t) = crate::resolve_alias_typeinfo(&annotation, &symbols, &options)
                    {
                        t.to_rust_type()
                    } else if let ExprType::BinOp(op) = &annotation
                        && matches!(op.op, crate::BinOps::BitOr)
                        && let Some(members) = crate::union_members(&annotation)
                        && !members.is_empty()
                        && members.iter().all(|m| {
                            crate::is_pyvalue_boxable_member(m)
                                || is_exception_class_member(m, &symbols, &options)
                        })
                    {
                        // A union MIXING exception classes with boxable
                        // members: exceptions have no boxed representation,
                        // so the parameter boxes and an exception argument
                        // stays a loud mismatch. Checked only AFTER the
                        // direct mapping, so `str | bytes` still lowers to
                        // StrOrBytes.
                        quote!(stdpython::PyValue)
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
