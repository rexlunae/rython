//! Struct-and-trait class lowering.
//!
//! A Python class lowers to a Rust struct plus impl blocks:
//!
//! - Instance attributes become struct fields, inferred from the `self.attr`
//!   assignments in `__init__` (from annotated parameters, literals, or
//!   construction of another known class).
//! - `__init__` lowers as an ordinary method taking `&mut self`, and a
//!   synthesized `new(...)` constructor default-initializes the struct and
//!   runs it; `ClassName(...)` call sites lower to `ClassName::new(...)?`.
//! - Methods lower as inherent methods; the receiver is `&self`, or
//!   `&mut self` when the method stores through `self` (directly or by
//!   calling another method of the class that does).
//!
//! Single inheritance (bases defined in the same module) lowers to a
//! trait-per-class scheme:
//!
//! - A class that has a base, or is used as a base, emits `trait {Name}Trait`
//!   with accessors for its fields (`fn f(&self) -> T`, `fn f_mut(&mut self)
//!   -> &mut T`), a `base()`/`base_mut()` accessor pair for its embedded base
//!   struct, and its own methods as default bodies. Each derived class
//!   implements every ancestor trait (accessors walk the embedded
//!   `__rython_base` chain; overridden methods replace the default).
//!   `self.helper()` calls on an instance resolve through the traits, so a
//!   call into an inherited method dispatches to the most-derived impl —
//!   Python's method resolution — while `super().helper()` pins the call to
//!   the direct base's implementation.
//! - A derived struct embeds its direct base as `pub __rython_base: Base`,
//!   so every ancestor's fields stay reachable (`self.__rython_base.f`, or
//!   `self.base().f` inside a generic trait default). `super().__init__(...)`
//!   lowers to `self.__rython_base.__init__(...)`, and a class without its
//!   own `__init__` synthesizes a forwarder so construction still runs the
//!   first `__init__` on the MRO.
//!
//! Unsupported class constructs — multiple inheritance, unknown/imported
//! bases, class-level statements, attributes whose types can't be inferred —
//! are conversion-time errors, never silently dropped: lowering that
//! diverges from Python must fail loudly.

use proc_macro2::TokenStream;
use pyo3::FromPyObject;
use quote::{format_ident, quote};

use crate::{
    Assign, CodeGen, CodeGenContext, ExprType, FunctionDef, PythonOptions, Statement,
    StatementType, SymbolTableNode, SymbolTableScopes,
};
use pyo3::{Borrowed, PyAny, PyResult};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ClassDef {
    pub name: String,
    /// The base expressions. Only simple same-module `Name` bases lower
    /// (the embedded-struct + trait scheme); a dotted base (`queue.Queue`)
    /// or any other expression is a loud error at codegen, not a silent
    /// drop. Extracted as ExprType so the parse does not crash on them.
    pub bases: Vec<ExprType>,
    /// Class keywords (`metaclass=...`, `**kwargs`). Python's AST stores
    /// these as keyword objects with an optional `arg` and a value
    /// expression — extracting them as plain strings crashed the parse.
    /// Codegen decides per keyword: `metaclass=abc.ABCMeta` is a lossy
    /// no-op (abstract-method enforcement has no Rust analogue), anything
    /// else is a loud error.
    pub keywords: Vec<ClassKeyword>,
    /// Class decorators (`@dataclass`, `@functools.lru_cache`, ...).
    /// Currently only `dataclass` (with or without `(frozen=..., ...)`
    /// args) is consumed — it synthesizes `__init__` from annotated
    /// fields. Any other decorator is a loud error at codegen.
    pub decorator_list: Vec<ExprType>,
    pub body: Vec<Statement>,
}

/// One class-definition keyword: the `arg` name (None for `**kwargs`) and
/// the value expression.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassKeyword {
    pub arg: Option<String>,
    pub value: ExprType,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ClassKeyword {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;
        let arg: Option<String> = ob
            .getattr("arg")
            .map_err(|e| crate::extraction_failure("class keyword arg", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class keyword arg", &ob, e))?;
        let value: ExprType = ob
            .getattr("value")
            .map_err(|e| crate::extraction_failure("class keyword value", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class keyword value", &ob, e))?;
        Ok(ClassKeyword { arg, value })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ClassDef {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;
        let name: String = ob
            .getattr("name")
            .map_err(|e| crate::extraction_failure("class name", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class name", &ob, e))?;
        let bases: Vec<ExprType> = ob
            .getattr("bases")
            .map_err(|e| crate::extraction_failure("class bases", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class bases", &ob, e))?;
        let keywords: Vec<ClassKeyword> = ob
            .getattr("keywords")
            .map_err(|e| crate::extraction_failure("class keywords", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class keywords", &ob, e))?;
        let body: Vec<Statement> = ob
            .getattr("body")
            .map_err(|e| crate::extraction_failure("class body", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class body", &ob, e))?;
        let decorator_list: Vec<ExprType> = ob
            .getattr("decorator_list")
            .map_err(|e| crate::extraction_failure("class decorators", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("class decorators", &ob, e))?;
        Ok(ClassDef {
            name,
            bases,
            keywords,
            decorator_list,
            body,
        })
    }
}

/// Whether a base-class NAME is METADATA — a builtin type, the
/// Enum/TypedDict family, `object`, or the `type` metaclass — whose
/// construction rython cannot model: the class lowers as a plain struct
/// (the class-as-value divergence). ONE predicate: the plain-struct
/// check and the real-base filtering previously kept two copies that had
/// drifted (`bytearray` in one, `type` in the other).
pub(crate) fn is_metadata_base_name(id: &str) -> bool {
    matches!(
        id,
        "str" | "bytes" | "bytearray" | "int" | "float" | "bool" | "list" | "dict"
            | "tuple" | "set" | "object" | "TypedDict"
            // `type` — a METACLASS base (`class LexerMeta(type)` —
            // pygments' lexer): a metaclass is a class factory, which
            // rython cannot express as a value.
            | "type"
    ) || is_enum_base_name(id)
}

/// The Enum-family base names — also consulted by the class-body walk
/// (enum MEMBERS are metadata, not struct fields).
pub(crate) fn is_enum_base_name(id: &str) -> bool {
    matches!(id, "Enum" | "IntEnum" | "Flag" | "IntFlag" | "StrEnum")
}

/// Whether a class is an exception class: its name matches the exception
/// naming convention (`*Error`, `*Exception`, `*Warning`) — the same
/// heuristic `raise` uses to construct PyException values — or one of its
/// bases does. A custom exception inheriting a builtin (`IDNAError(UnicodeError)`)
/// or another custom exception (`IDNABidiError(IDNAError)`) is an exception
/// class too. Lowered as a marker struct; the runtime matches exceptions by
/// name string, so the class carries no data.
pub fn is_exception_class(class: &ClassDef) -> bool {
    // The canonical predicate from the raise lowering: the builtin set
    // plus the naming convention. (The convention alone previously missed
    // classes inheriting KeyboardInterrupt/SystemExit/StopIteration/
    // GeneratorExit — the builtin names outside the *Error/*Exception/
    // *Warning shape.)
    let is_exception = crate::ast::tree::raise_stmt::is_exception_class_name;
    if is_exception(&class.name) {
        return true;
    }
    class.bases.iter().any(|b| match b {
        ExprType::Name(n) => is_exception(&n.id),
        _ => false,
    })
}

impl ClassDef {
    /// Whether this class is decorated `@dataclass` (with or without
    /// `(frozen=..., slots=...)` arguments) — via the systematic decorator
    /// registry (decorator.rs).
    pub fn is_dataclass(&self) -> bool {
        self.decorator_list
            .iter()
            .any(|d| crate::is_dataclass_decorator(d))
    }

    /// Whether this class is a `typing.NamedTuple` subclass (base
    /// `typing.NamedTuple` or bare `NamedTuple`, including the CALL form
    /// `typing.NamedTuple("Url", [("scheme", T), ...])`): the annotated
    /// fields are field metadata, and construction takes them as arguments
    /// — the same shape a @dataclass synthesizes (urllib3's ProxyConfig,
    /// _WrappedAndVerifiedSocket, Url).
    pub fn is_namedtuple(&self) -> bool {        self.bases.iter().any(|b| match b {
            ExprType::Attribute(a) => {
                a.attr == "NamedTuple"
                    && matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
            }
            ExprType::Name(n) => n.id == "NamedTuple",
            // The CALL form (`typing.NamedTuple("Url", [...])`): the field
            // list rides in the call's second argument.
            ExprType::Call(c) => match c.func.as_ref() {
                ExprType::Attribute(a) => {
                    a.attr == "NamedTuple"
                        && matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
                }
                ExprType::Name(n) => n.id == "NamedTuple",
                _ => false,
            },
            _ => false,
        })
    }

    /// The NamedTuple field list, whether from the attribute base (annotated
    /// class-level fields) or the CALL form (`typing.NamedTuple("Url",
    /// [("scheme", T), ...])` — the second argument). Returns None for the
    /// call form without a list, or for non-NamedTuple classes.
    pub(crate) fn namedtuple_call_fields(&self) -> Option<Vec<(String, ExprType)>> {
        if !self.is_namedtuple() {
            return None;
        }
        for b in &self.bases {
            if let ExprType::Call(c) = b
                && let Some(list) = c.args.get(1)
                && let ExprType::List(l) = list
            {
                let mut fields = Vec::new();
                for elt in l {
                    if let ExprType::Tuple(t) = elt
                        && let [ExprType::Constant(c0), ann] = t.elts.as_slice()
                        && let Some(litrs::Literal::String(s)) = &c0.0
                    {
                        fields.push((s.value().to_string(), ann.clone()));
                    }
                }
                if !fields.is_empty() {
                    return Some(fields);
                }
            }
        }
        None
    }

    /// Whether the class lowers as a PLAIN STRUCT due to a metadata base:
    /// a builtin (`str`, `bytes`, `int`, ...) or `object`/`Enum`-family
    /// base — the base's construction is unmodeled (botocore's
    /// ClientConfigString(str)). `__new__` and `super().__new__` are then
    /// dropped.
    fn is_metadata_struct(&self) -> bool {
        self.bases.iter().any(|b| match b {
            ExprType::Name(n) => is_metadata_base_name(&n.id),
            _ => false,
        })
    }

    /// Synthesize the `@dataclass` `__init__` from the annotated class-level
    /// fields and PREPEND it to the body. Idempotent: a class that already
    /// has an `__init__` (or has already been synthesized) is untouched.
    /// Field order = declaration order; a field with a default (`count: int
    /// = 0`) becomes a defaulted parameter. Runs from BOTH find_symbols
    /// (so call sites see a constructed class) and to_rust (so the emitted
    /// class carries the method).
    pub(crate) fn synthesize_dataclass_init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // A `typing.NamedTuple` subclass synthesizes the same __init__ from
        // its annotated fields (the NamedTuple base is metadata).
        let is_nt = self.is_namedtuple();
        if (!self.is_dataclass() && !is_nt) || self.init_method().is_some() {
            return Ok(());
        }
        let mut params: Vec<crate::Parameter> = Vec::new();
        let mut defaults: Vec<Box<ExprType>> = Vec::new();
        let mut stores: Vec<Statement> = Vec::new();
        // The CALL form (`typing.NamedTuple("Url", [...])`) carries its
        // fields in the base expression, not as annotated class-level
        // statements — seed the params from there. Defaults come from the
        // class's `__new__` signature when present (urllib3's Url): Python
        // constructs NamedTuples through `__new__(cls, scheme=None, ...)`,
        // and the omitted-argument call sites rely on those defaults.
        if let Some(fields) = self.namedtuple_call_fields() {
            // The __new__ signature's DEFAULTS by parameter name: Python
            // defaults are positionally aligned with the args (defaults[0]
            // belongs to args[len(args)-len(defaults)]), so zip them.
            let new_defaults: Vec<(String, ExprType)> = self
                .methods()
                .find(|m| m.name == "__new__")
                .map(|m| {
                    let names: Vec<String> = m
                        .args
                        .posonlyargs
                        .iter()
                        .chain(m.args.args.iter())
                        .filter(|p| p.arg != "cls")
                        .map(|p| p.arg.clone())
                        .collect();
                    let skip = names.len().saturating_sub(m.args.defaults.len());
                    names
                        .into_iter()
                        .enumerate()
                        // A REQUIRED field (index below the default offset)
                        // takes no default: mapping it to `defaults[0]`
                        // (via saturating_sub) made required fields
                        // optional and misaligned every default.
                        .filter_map(|(i, n)| {
                            if i < skip {
                                None
                            } else {
                                m.args.defaults.get(i - skip).map(|d| (n, (**d).clone()))
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let new_required: Vec<String> = self
                .methods()
                .find(|m| m.name == "__new__")
                .map(|m| {
                    let names: Vec<String> = m
                        .args
                        .posonlyargs
                        .iter()
                        .chain(m.args.args.iter())
                        .filter(|p| p.arg != "cls")
                        .map(|p| p.arg.clone())
                        .collect();
                    let skip = names.len().saturating_sub(m.args.defaults.len());
                    names
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i < skip)
                        .map(|(_, n)| n)
                        .collect()
                })
                .unwrap_or_default();
            for (i, (name, ann)) in fields.into_iter().enumerate() {
                // A field whose default rides in __new__: the __new__
                // parameter's default applies at the same position.
                if let Some((_, default)) =
                    new_defaults.iter().find(|(n, _)| *n == name)
                {
                    defaults.push(Box::new(default.clone()));
                } else if new_required.iter().any(|n| *n == name) {
                    // __new__ has the param WITHOUT a default: required.
                } else if i < new_defaults.len() {
                    // No __new__ at all, or field beyond __new__'s params:
                    // NamedTuple fields default to None in practice.
                    defaults.push(Box::new(ExprType::NoneType(
                        crate::ast::tree::constant::Constant(None),
                    )));
                }
                params.push(crate::Parameter {
                    arg: name.clone(),
                    annotation: Some(Box::new(ann)),
                    type_comment: None,
                    lineno: None,
                    col_offset: None,
                    end_lineno: None,
                    end_col_offset: None,
                });
                stores.push(dataclass_store_stmt(&name));
            }
        }
        for stmt in &self.body {
            match &stmt.statement {
                StatementType::AnnotatedName { name, annotation } => {
                    params.push(crate::Parameter {
                        arg: name.clone(),
                        annotation: Some(Box::new(annotation.clone())),
                        type_comment: None,
                        lineno: None,
                        col_offset: None,
                        end_lineno: None,
                        end_col_offset: None,
                    });
                    stores.push(dataclass_store_stmt(name));
                }
                // A defaulted field (`count: int = 0`) arrives as an Assign
                // with an annotation; its value is the default.
                StatementType::Assign(a) => {
                    if let [ExprType::Name(n)] = a.targets.as_slice()
                        && let Some(ann) = &a.annotation
                    {
                        params.push(crate::Parameter {
                            arg: n.id.clone(),
                            annotation: Some(Box::new(ann.clone())),
                            type_comment: None,
                            lineno: None,
                            col_offset: None,
                            end_lineno: None,
                            end_col_offset: None,
                        });
                        // A `field(default_factory=dict)` default (urllib3's
                        // EmscriptenRequest.headers): keep the `field(...)`
                        // CALL as a marker default — check_default_constant
                        // accepts it (a factory creates a FRESH container per
                        // call, which rython's inline-empty exactly matches,
                        // unlike a shared mutable default), and fill renders
                        // the typed empty container from the annotation.
                        defaults.push(Box::new(a.value.clone()));
                        stores.push(dataclass_store_stmt(&n.id));
                    } else if let [ExprType::Name(_)] = a.targets.as_slice() {
                        // A class-level CONSTANT on a dataclass-shaped class
                        // (`_hash_url_fragment_re = re.compile(...)` — pip's
                        // LinkHash): metadata, not a field — ignored.
                    } else {
                        return Err(format!(
                            "class `{}` is a @dataclass with a class-level \
                             assignment that is not an annotated field",
                            self.name
                        )
                        .into());
                    }
                }
                _ => {}
            }
        }
        if params.is_empty() {
            if is_nt {
                // A NamedTuple with no fields is still constructible with no
                // args — emit the empty __init__ (no params) below.
            } else {
                return Err(format!(
                    "class `{}` is a @dataclass with no annotated fields; \
                     annotate at least one field (`x: int`)",
                    self.name
                )
                .into());
            }
        }
        let init = FunctionDef {
            name: "__init__".to_string(),
            args: crate::Arguments {
                posonlyargs: Vec::new(),
                // `self` first, like every method; strip_self removes it
                // where callers build the `new(...)` signature.
                args: std::iter::once(crate::Parameter {
                    arg: "self".to_string(),
                    annotation: None,
                    type_comment: None,
                    lineno: None,
                    col_offset: None,
                    end_lineno: None,
                    end_col_offset: None,
                })
                .chain(params)
                .collect(),
                vararg: None,
                kwonlyargs: Vec::new(),
                kw_defaults: Vec::new(),
                kwarg: None,
                defaults,
            },
            body: stores,
            decorator_list: Vec::new(),
            returns: None,
        };
        // Insert the synthesized __init__ AFTER the docstring (if any):
        // prepending it at index 0 would push the docstring down, so
        // get_docstring (which looks at body[0]) would miss it and the
        // docstring statement would hit the class-body error arm.
        let insert_at = if self.get_docstring().is_some() { 1 } else { 0 };
        self.body.insert(
            insert_at,
            Statement {
                statement: StatementType::FunctionDef(init),
                lineno: None,
                col_offset: None,
                end_lineno: None,
                end_col_offset: None,
            },
        );
        Ok(())
    }

    /// The class's `__init__` method, if it defines one.
    pub fn init_method(&self) -> Option<&FunctionDef> {
        self.methods().find(|m| m.name == "__init__")
    }

    /// The property getter/setter PAIRS on this class: Python's
    /// `@property def x` + `@x.setter def x` lower to TWO plain methods
    /// with the SAME Python name — Rust forbids same-name methods
    /// (E0428). The setter gets a distinct Rust name `x_set` (the
    /// property-read/set divergence keeps the getter as a plain method
    /// `x`). Returns the setter's Python name → the setter FunctionDef.
    pub fn property_setters(&self) -> std::collections::HashMap<String, FunctionDef> {
        let mut out = std::collections::HashMap::new();
        let methods: Vec<&FunctionDef> = self.methods().collect();
        for (i, m) in methods.iter().enumerate() {
            let is_setter = m.decorator_list.iter().any(|d| match d {
                ExprType::Attribute(a) => {
                    a.attr == "setter"
                        && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == m.name)
                }
                _ => false,
            });
            if is_setter {
                // The GETTER must exist for this to be a pair: the setter
                // alone (a `@x.setter` with no preceding `@property`) is
                // not a valid Python property anyway.
                let has_getter = methods.iter().enumerate().any(|(j, g)| {
                    j != i && g.name == m.name && {
                        let is_getter = g.decorator_list.iter().any(|d| match d {
                            ExprType::Name(n) => n.id == "property",
                            ExprType::Attribute(a) => a.attr == "property",
                            _ => false,
                        });
                        is_getter
                    }
                });
                if has_getter {
                    out.insert(m.name.clone(), (*m).clone());
                }
            }
        }
        out
    }

    /// Whether the class defines a property getter named `name` (a
    /// `@property def name` or the getter half of a pair) — used to route
    /// attribute READS to the getter method call.
    pub fn has_property_getter(
        &self,
        name: &str,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> bool {
        // A property defined on a BASE class is a property of the derived
        // class too (`self.host` on HTTPSConnection, whose `host`
        // property HTTPConnection defines): the read routes to the getter
        // call either way. The chain follows imported bases.
        self.base_chain_with_options(symbols, options).iter().any(|c| {
            c.methods().any(|m| {
                m.name == name
                    && m.decorator_list.iter().any(|d| match d {
                        ExprType::Name(n) => n.id == "property",
                        ExprType::Attribute(a) => a.attr == "property",
                        _ => false,
                    })
            })
        })
    }

    /// The RUST name a method is emitted under: a property SETTER in a
    /// getter/setter pair lowers as `{name}_set` (Rust forbids same-name
    /// methods — E0428); everything else keeps its Python name. Callers
    /// (attribute read/store routing, super trampolines) must use the
    /// SAME name so call sites match the definition.
    pub fn emitted_method_name(&self, m: &FunctionDef) -> String {
        if m.is_property_setter() && self.property_setters().contains_key(&m.name) {
            format!("{}_set", m.name)
        } else {
            m.name.clone()
        }
    }

    /// Whether a method with this PYTHON name is a property setter (needs
    /// `{name}_set` at call sites).
    pub fn is_property_setter(&self, name: &str) -> bool {
        self.property_setters().contains_key(name)
    }

    /// The methods defined directly on the class, in source order.
    /// The methods defined directly on the class, in source order —
    /// EXCLUDING `@typing.overload` stubs (their `...` bodies and default
    /// placeholders are compile-time metadata; the real implementation
    /// follows them, urllib3's `_ssl_io_loop`).
    pub fn methods(&self) -> impl Iterator<Item = &FunctionDef> {
        self.body.iter().filter_map(|s| match &s.statement {
            StatementType::FunctionDef(f) => {
                let is_overload = f.decorator_list.iter().any(|d| match d {
                    ExprType::Name(n) => n.id == "overload",
                    ExprType::Attribute(a) => a.attr == "overload",
                    _ => false,
                });
                if is_overload {
                    None
                } else {
                    Some(f)
                }
            }
            _ => None,
        })
    }

    /// The class's real base (the first base that is not `object`), resolved
    /// through the symbol table to its `ClassDef`. None for a base-less
    /// class or a base that is not a known class in this module.
    pub(crate) fn base_class(&self, symbols: &SymbolTableScopes) -> Option<ClassDef> {
        let base = self
            .bases
            .iter()
            .find(|b| matches!(b, ExprType::Name(n) if n.id != "object"))?;
        let ExprType::Name(n) = base else {
            return None;
        };
        match symbols.get(&n.id) {
            Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
            _ => None,
        }
    }

    /// An options-aware base chain: `base_class` cannot follow IMPORTED
    /// bases (no options), so this resolves them through module_defs —
    /// PoolManager(RequestMethods) with `request` inherited cross-module.
    pub(crate) fn base_class_with_options(
        &self,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<ClassDef> {
        let base = self
            .bases
            .iter()
            .find(|b| matches!(b, ExprType::Name(n) if n.id != "object"))?;
        let ExprType::Name(n) = base else {
            return None;
        };
        match symbols.get(&n.id) {
            Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
            Some(SymbolTableNode::ImportFrom(i)) => {
                let path = i.resolved_module_path(options);
                crate::module_class_def(options, &path, &n.id)
                    .map(|(c, _)| c)
            }
            _ => None,
        }
    }

    /// Options-aware MRO method lookup (imported bases resolved).
    pub(crate) fn method_on_mro_with_options(
        &self,
        name: &str,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<FunctionDef> {
        let mut chain = vec![self.clone()];
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(self.name.clone());
        while let Some(base) = chain.last().and_then(|c| {
            c.base_class_with_options(symbols, options)
        }) {
            if !seen.insert(base.name.clone()) {
                break;
            }
            chain.push(base);
        }
        chain
            .into_iter()
            .find_map(|c| c.methods().find(|m| m.name == name).cloned())
    }

    /// The class itself followed by every ancestor, nearest base first.
    /// Returns just `[self]` when the class has no base.
    /// The base chain FOLLOWING IMPORTED BASES, each ancestor paired with
    /// the symbol table of its DEFINING module (the scope its trait's
    /// accessor types and method sets were declared in). Mirrors the
    /// options-aware `base` resolution in to_rust: aliases follow to their
    /// canonical unless the canonical is shadowed by the deriving class
    /// itself (the external-alias pattern), and imported bases resolve
    /// through module_defs — an unresolvable base ends the chain
    /// (external: metadata).
    pub(crate) fn cross_module_chain(
        &self,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Vec<(ClassDef, SymbolTableScopes, crate::PythonOptions, Option<Vec<String>>)> {
        // Each entry carries the scope its class's TRAIT was declared in:
        // the defining module's symbols AND an options clone whose
        // module_path/this_module_path point there, so relative imports
        // and annotations in that module's field types resolve exactly as
        // they did when the trait was emitted. `Some(path)` marks a
        // CROSS-MODULE ancestor (its module path, for the type-visibility
        // glob use at the emit site).
        let mut chain = vec![(self.clone(), symbols.clone(), options.clone(), None)];
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(self.name.clone());
        loop {
            let (last, last_syms, last_opts, last_path) = chain.last().unwrap();
            let Some(base_name) = last.bases.iter().find_map(|b| match b {
                ExprType::Name(n)
                    if n.id != "object" && !is_metadata_base_name(&n.id) =>
                {
                    Some(n.id.clone())
                }
                _ => None,
            }) else {
                break;
            };
            let resolve_import = |i: &crate::ImportFrom, name: &str| {
                let path = i.resolved_module_path(last_opts);
                crate::resolve_imported_class_with_path(options, &path, name, 0)
                    .map(|(c, s, defining)| {
                        let mut o = options.clone();
                        // The package context: an __init__ module IS its
                        // own package (mirrors resolve_imported_class).
                        let is_package = options.module_defs.keys().any(|k| {
                            k.len() > defining.len() && k[..defining.len()] == defining[..]
                        });
                        o.module_path = if is_package {
                            defining.clone()
                        } else {
                            defining[..defining.len().saturating_sub(1)].to_vec()
                        };
                        o.this_module_path = defining.clone();
                        (c, s, o, Some(defining))
                    },
                )
            };
            // A base resolved LOCALLY within an already-imported module
            // (ConnectionPool inside connectionpool.py, reached from
            // SOCKS) stays in that module: it inherits the path.
            let next = match last_syms.get(&base_name) {
                Some(SymbolTableNode::ClassDef(c)) => Some((
                    c.clone(),
                    last_syms.clone(),
                    last_opts.clone(),
                    last_path.clone(),
                )),
                Some(SymbolTableNode::Alias(canonical)) => match last_syms.get(canonical) {
                    Some(SymbolTableNode::ImportFrom(i)) => resolve_import(i, canonical),
                    Some(SymbolTableNode::ClassDef(c)) if c.name != last.name => Some((
                        c.clone(),
                        last_syms.clone(),
                        last_opts.clone(),
                        last_path.clone(),
                    )),
                    _ => None,
                },
                Some(SymbolTableNode::ImportFrom(i)) => resolve_import(i, &base_name),
                _ => None,
            };
            let Some(entry) = next else {
                break;
            };
            if !seen.insert(entry.0.name.clone()) {
                break;
            }
            chain.push(entry);
        }
        chain
    }

    pub(crate) fn base_chain(&self, symbols: &SymbolTableScopes) -> Vec<ClassDef> {
        self.base_chain_impl(symbols, |c, s| c.base_class(s))
    }

    /// An options-aware base chain: the plain chain cannot follow IMPORTED
    /// bases (no options to resolve the module), so a chain that crosses a
    /// module boundary (`PoolManager(RequestMethods)` with the field
    /// stored in the imported base — the field-walk, the property check,
    /// and the Option-ness resolution need it) stops at the boundary and
    /// misses the ancestor's fields (E0615/E0609 in generic trait
    /// defaults). The two share one walk so their shapes cannot drift.
    pub(crate) fn base_chain_with_options(
        &self,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Vec<ClassDef> {
        self.base_chain_impl(symbols, |c, s| c.base_class_with_options(s, options))
    }

    fn base_chain_impl(
        &self,
        symbols: &SymbolTableScopes,
        next: impl Fn(&ClassDef, &SymbolTableScopes) -> Option<ClassDef>,
    ) -> Vec<ClassDef> {
        let mut chain = vec![self.clone()];
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(self.name.clone());
        while let Some(base) = chain.last().and_then(|c| next(c, symbols)) {
            // A cyclic base (`class A(A)` after the name was rebound, so
            // the class resolves to itself) must terminate: Python looks
            // the base up in the OUTER scope and errors when it finds the
            // class being defined, but the symbol table can only see the
            // rebound name. Stop before the chain grows forever; emit_class
            // reports the cycle as a conversion error.
            if !seen.insert(base.name.clone()) {
                break;
            }
            chain.push(base);
        }
        chain
    }

    /// THE conversion-time inheritance-tree lookup: whether `child` names
    /// a class that IS `ancestor` or transitively inherits from it.
    /// `isinstance(x, C)` folds through this — `isinstance(dog, Animal)`
    /// is true because Dog's base chain contains Animal, exactly like
    /// CPython's subclass check. Non-class names answer false.
    pub(crate) fn class_extends(
        child: &str,
        ancestor: &str,
        symbols: &SymbolTableScopes,
    ) -> bool {
        if child == ancestor {
            // Reflexive even when the name isn't resolvable in scope (an
            // annotated parameter of an imported class).
            return true;
        }
        match symbols.get(child) {
            Some(crate::SymbolTableNode::ClassDef(c)) => c
                .base_chain(symbols)
                .iter()
                .any(|a| a.name == ancestor),
            _ => false,
        }
    }

    /// `class_extends` by the hierarchy registry alone (no symbol table in
    /// hand — the coercion authority): whether `child` is in `ancestor`'s
    /// subtree.
    pub(crate) fn extends_by_name(child: &str, ancestor: &str) -> bool {
        child == ancestor || crate::ast::tree::hierarchy::in_subtree_by_name(child, ancestor)
    }

    /// The class name that closes an inheritance cycle (`class A(A)`), or
    /// None for a valid chain. Walks bases like base_chain but reports the
    /// repeat instead of silently stopping.
    pub(crate) fn base_cycle(&self, symbols: &SymbolTableScopes) -> Option<String> {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut cur = Some(self.clone());
        while let Some(c) = cur {
            if !seen.insert(c.name.clone()) {
                return Some(c.name.clone());
            }
            cur = c.base_class(symbols);
        }
        None
    }

    /// The first method named `name` found walking the MRO (the class
    /// itself, then its base, then the base's base, ...).
    pub(crate) fn method_on_mro(&self, name: &str, symbols: &SymbolTableScopes) -> Option<FunctionDef> {
        self.base_chain(symbols)
            .into_iter()
            .find_map(|c| c.methods().find(|m| m.name == name).cloned())
    }

    /// Whether `attr` is a field assigned somewhere in this class's own
    /// `__init__`.
    pub(crate) fn owns_field(&self, attr: &str) -> bool {
        // EVERY method's stores, not just `__init__`'s: an attribute
        // first assigned elsewhere is still a field of this class (issue
        // #137 round 23 taught `infer_fields` the same thing), and the two
        // must agree. When they disagree the field exists on the struct
        // and in the trait's accessors, but the rewrite that routes
        // `self.x` through `self.x()` inside a generic trait default does
        // not fire — the body then reads the accessor METHOD as a value
        // (E0615) or a field the generic `Self` has not got (E0609).
        self.methods().any(|m| {
            let mut stores = Vec::new();
            collect_field_stores(&m.body, &mut stores);
            stores.iter().any(|s| s.attr == attr)
        })
    }

    /// Whether `infer_fields` puts `attr` on THIS class's struct.
    ///
    /// `owns_field` scans STORES, which is only half of what makes a field.
    /// The other half is round 23's external-base READ synthesis: an
    /// attribute the class reads but never assigns, when its base is
    /// external to the generated crate and unmodeled (urllib3's
    /// `HTTPConnection(_HTTPConnection)` reading `self.port`). Those land
    /// on the struct and in the trait's accessors exactly like stored
    /// fields, so the accessor rewrite has to see them too — otherwise a
    /// generic trait default reads the accessor METHOD as a value (E0615).
    ///
    /// This asks `infer_fields` rather than re-deriving its conditions.
    /// A hand-copied gate is how the two field sets drift apart in the
    /// first place: an earlier cut of this missed that `infer_fields`
    /// yields nothing for a class with no `__init__`, and so claimed a
    /// field the struct did not have — the same disagreement as the bug,
    /// pointing the other way.
    pub(crate) fn has_inferred_field(
        &self,
        attr: &str,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> bool {
        self.infer_fields(symbols, options)
            .map(|fields| fields.iter().any(|(f, _)| f == attr))
            .unwrap_or(false)
    }

    /// Which class in the MRO chain owns the field `attr`: 0 for this class,
    /// 1 for its direct base, etc. None when no class in the chain assigns
    /// the field. `self.attr` where attr is owned by an ancestor resolves
    /// through the ancestor's embedded struct.
    ///
    /// The owner is the class whose STRUCT physically holds the field — the
    /// DEEPEST ancestor that assigns it. A derived class's own stores of a
    /// base-owned field are filtered out of its struct (see `to_rust`), so
    /// the field lives at the topmost assigner in the chain; `rposition`
    /// finds that one. `.position` (nearest assigner) would report a
    /// depth-0 owner for a field the struct no longer has, making
    /// `self.n = n` in a subclass's own `__init__` write to a nonexistent
    /// field.
    pub(crate) fn field_owner_depth(
        &self,
        attr: &str,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Option<usize> {
        // Ownership is STORES or the external-base read synthesis: both put
        // the field on that class's struct and in its trait's accessors, so
        // both have to answer here or the accessor rewrite misses one kind
        // (issue #137 round 25). The synthesis can sit on an ANCESTOR —
        // urllib3's HTTPSConnection reads `self.host`, which HTTPConnection
        // synthesized from its own unmodeled stdlib base — so it is part of
        // the chain walk, not a check on the receiver's class alone.
        self.base_chain_with_options(symbols, options)
            .iter()
            .rposition(|c| {
                c.owns_field(attr) || c.has_inferred_field(attr, symbols, options)
            })
    }

    /// The class of the value stored in field `attr`, when the field holds
    /// an instance of another known class (composition): inferred from the
    /// `__init__` stores, either a direct construction or an
    /// annotated parameter whose annotation names a class. Walks the MRO so
    /// a base class's composed field resolves from a derived method.
    pub(crate) fn field_class(
        &self,
        attr: &str,
        symbols: &SymbolTableScopes,
        options: &crate::PythonOptions,
    ) -> Option<String> {
        let chain = self.base_chain(symbols);
        let owner = chain.iter().find(|c| c.owns_field(attr))?;
        let init = owner.init_method()?;
        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        // Prefer a store whose value is a CLASS CONSTRUCTION
        // (`self.headers = HTTPHeaderDict(headers)` — urllib3's
        // HTTPResponse.__init__, where the OTHER store assigns the
        // external `_TYPE_HEADERS` param): the constructed class is the
        // field's real type, where the param annotation may name an
        // alias that resolves to nothing (round 61b — the Mapping.get
        // fallback and the boxed-field families).
        let store = stores
            .iter()
            .filter(|s| s.attr == attr)
            .find(|s| {
                matches!(
                    &s.value,
                    ExprType::Call(c)
                        if matches!(c.func.as_ref(), ExprType::Name(_))
                )
            })
            .or_else(|| stores.iter().find(|s| s.attr == attr))?;
        let class_name = match store.value {
            ExprType::Call(call) => match call.func.as_ref() {
                ExprType::Name(n) => Some(n.id.clone()),
                _ => None,
            },
            ExprType::Name(n) => {
                let param = init
                    .args
                    .posonlyargs
                    .iter()
                    .chain(init.args.args.iter())
                    .chain(init.args.kwonlyargs.iter())
                    .find(|p| p.arg == n.id);
                // A store from a PARAM (`self.proxy = proxy` where proxy
                // is an annotated __init__ parameter): the annotation names
                // the class (or the non-None side of a `T | None` union).
                let class_name = match param {
                    Some(param) => match param.annotation.as_deref() {
                        Some(ExprType::Name(ann)) => Some(ann.id.clone()),
                        // An OPTIONAL class annotation (`headers:
                        // HTTPHeaderDict | None` — HTTPResponse): the class is
                        // the non-None side of the union.
                        Some(ExprType::BinOp(b)) if crate::is_none_expr(&b.left) => {
                            match b.right.as_ref() {
                                ExprType::Name(ann) => Some(ann.id.clone()),
                                _ => None,
                            }
                        }
                        Some(ExprType::BinOp(b)) if crate::is_none_expr(&b.right) => {
                            match b.left.as_ref() {
                                ExprType::Name(ann) => Some(ann.id.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                    // A store from a LOCAL (`proxy = parse_url(...);
                    // self.proxy = proxy` — urllib3's ProxyManager.__init__,
                    // where the factory local is later field-stored): the
                    // local's single assignment is a factory CALL whose
                    // return annotation names the class (round 90 — without
                    // it the `self.proxy` receiver never resolves and the
                    // Option-field reads double-wrap).
                    None => init.body.iter().find_map(|s| {
                        let crate::StatementType::Assign(a) = &s.statement else {
                            return None;
                        };
                        let [crate::ExprType::Name(tn)] = a.targets.as_slice() else {
                            return None;
                        };
                        if tn.id != n.id {
                            return None;
                        }
                        let crate::ExprType::Call(call) = &a.value else {
                            return None;
                        };
                        let crate::ExprType::Name(callee) = call.func.as_ref() else {
                            return None;
                        };
                        let f = match symbols.get(&callee.id) {
                            Some(crate::SymbolTableNode::FunctionDef(f)) => f.clone(),
                            // An IMPORTED factory (`parse_url` from
                            // .util.url): resolve the FunctionDef through
                            // its defining module.
                            Some(crate::SymbolTableNode::ImportFrom(i)) => {
                                let path = i.resolved_module_path(options);
                                if !options.module_defs.contains_key(&path) {
                                    return None;
                                }
                                crate::module_function_def(options, &path, &callee.id)
                                    .map(|(f, _)| f)?
                            }
                            _ => return None,
                        };
                        match f.returns.as_deref() {
                            Some(ExprType::Name(ann)) => Some(ann.id.clone()),
                            _ => None,
                        }
                    }),
                };
                let Some(class_name) = class_name else {
                    return None;
                };
                match symbols.get(&class_name) {
                    Some(SymbolTableNode::ClassDef(_)) => Some(class_name),
                    // An IMPORTED class field (`self.headers = HTTPHeaderDict(
                    // headers)` where HTTPHeaderDict comes from
                    // `from ._collections import HTTPHeaderDict` — urllib3's
                    // HTTPResponse): resolve through the defining module.
                    Some(SymbolTableNode::ImportFrom(_)) => {
                        if crate::resolve_class_referenced(&class_name, symbols, options).is_some()
                        {
                            Some(class_name)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => return None,
        };
        let class_name = class_name?;
        match symbols.get(&class_name) {
            Some(SymbolTableNode::ClassDef(_)) => Some(class_name),
            // An IMPORTED class field (`self.headers = HTTPHeaderDict(
            // headers)` where HTTPHeaderDict comes from
            // `from ._collections import HTTPHeaderDict` — urllib3's
            // HTTPResponse): resolve through the defining module.
            Some(SymbolTableNode::ImportFrom(_)) => {
                if crate::resolve_class_referenced(&class_name, symbols, options).is_some() {
                    Some(class_name)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Whether `method` mutates `self` — directly (attribute stores,
    /// mutating container methods on `self.attr`) or transitively through
    /// a call that bases at `self`: another method of this class
    /// (`self.helper()`) or a mutating method of a composed field's class
    /// (`self.inner.bump()`).
    pub(crate) fn method_needs_mut_self(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> bool {
        // Trait signatures widen to `&mut self` when ANY definition in the
        // hierarchy mutates (overrides re-emit into the root's trait), so
        // call sites must borrow the receiver mutably to match. The
        // module-level precompute is authoritative when present. The
        // precompute keys by the TOPMOST class in the chain that defines
        // the method (the trait owner — the first class in the chain that
        // defines it, i.e. the last in base_chain's self→ancestor order),
        // so the lookup must use the same key: a middle class redefining
        // the method must still find the entry recorded under the root,
        // or the widening is lost and call sites emit read-only borrows
        // the widened trait does not accept.
        let root = self
            .base_chain(symbols)
            .into_iter()
            .rev()
            .find(|c| c.methods().any(|mm| mm.name == method));
        if let Some(root) = root.as_ref()
            && options
                .trait_mut_self
                .get(&root.name)
                .is_some_and(|s| s.contains(method))
        {
            return true;
        }
        // Cross-module widening: `trait_mut_self` is computed per module in
        // `Module::to_rust`, so a class imported from another module of the
        // same crate has no entry in the CURRENT module's table — yet its
        // trait was widened (and emitted) in the DEFINING module, possibly
        // by a sibling class this module never sees. When the local table
        // has no entry for the root, the class is not a hierarchy class of
        // the current module; recompute the precompute over the shared
        // cross-module ASTs so call sites borrow mutably to match.
        if let Some(root) = root.as_ref()
            && !options.module_defs.is_empty()
            && crate::module_widens_method_cached(options, &root.name, method)
        {
            return true;
        }
        // Fallback (no module precompute): the root's own body mutating, or
        // a chain class's override mutating.
        let mut visited = std::collections::HashSet::new();
        for c in self.base_chain(symbols) {
            if c.methods().any(|mm| mm.name == method)
                && c.method_mut_inner(method, symbols, &mut visited, options)
            {
                return true;
            }
        }
        false
    }

    /// Whether this class's OWN definition of `method` (not any override)
    /// mutates self — the module-level trait-signature widening precompute
    /// uses this on every defining class.
    pub(crate) fn own_method_mutates(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> bool {
        let mut visited = std::collections::HashSet::new();
        self.method_mut_inner(method, symbols, &mut visited, options)
    }

    fn method_mut_inner(
        &self,
        method: &str,
        symbols: &SymbolTableScopes,
        visited: &mut std::collections::HashSet<(String, String)>,
        options: &PythonOptions,
    ) -> bool {
        // A cycle in the call graph resolves optimistically: the mutation,
        // if real, is found on the acyclic part of some path.
        if !visited.insert((self.name.clone(), method.to_string())) {
            return false;
        }
        let Some(m) = self.method_on_mro(method, symbols) else {
            return false;
        };
        let params = method_param_names(&m);
        let ctx = CodeGenContext::Class(self.name.clone());
        // The same resolver-backed analysis codegen uses, threading the
        // visited set through recursive method resolution (RefCell because
        // the resolver is a shared Fn).
        let visited = std::cell::RefCell::new(visited);
        let resolve = |call: &crate::Call| -> Option<bool> {
            let ExprType::Attribute(attr) = call.func.as_ref() else {
                return None;
            };
            let (class, class_symbols) =
                crate::receiver_class(&attr.value, &ctx, symbols, options)?;
            if class.method_on_mro(&attr.attr, &class_symbols).is_none() {
                return None;
            }
            Some(class.method_mut_inner(
                &attr.attr,
                &class_symbols,
                &mut **visited.borrow_mut(),
                options,
            ))
        };
        // The receiver is the FIRST parameter whatever its name (issue
        // #132): a method whose stores go through `factory_self` needs
        // `&mut self` exactly like one storing through `self`, and the
        // renamed body at codegen time will say so too.
        let receiver_name = m
            .args
            .posonlyargs
            .first()
            .or(m.args.args.first())
            .map(|p| p.arg.clone())
            .unwrap_or_else(|| "self".to_string());
        if crate::analyze_scope_with(&m.body, &params, &resolve)
            .needs_mut
            .contains(receiver_name.as_str())
        {
            return true;
        }
        // The fetch-writeback (round 99): a mutation through a local
        // whose provenance resolves to a `self.<field>` container slot
        // (`item = self.find(name)`; `item.qty -= qty`) writes the slot —
        // the method needs `&mut self` even though no python-level store
        // touches `self` (the generated py_set_index is invisible to the
        // scope analysis).
        let ctx = CodeGenContext::Class(self.name.clone());
        Self::body_has_container_writeback_checked(&m.body, &ctx, symbols, options)
    }

    /// Whether any statement in `body` mutates a field THROUGH a
    /// fetch-local whose provenance resolves to a `self.<field>` slot
    /// (the write-back's &mut-self trigger, round 99).
    fn body_has_container_writeback_checked(
        body: &[crate::Statement],
        ctx: &CodeGenContext,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> bool {
        fn walk(
            stmts: &[crate::Statement],
            ctx: &CodeGenContext,
            symbols: &SymbolTableScopes,
            options: &PythonOptions,
        ) -> bool {
            for stmt in stmts {
                let targets: Vec<&ExprType> = match &stmt.statement {
                    crate::StatementType::AugAssign(a) => vec![&a.target],
                    crate::StatementType::Assign(a) => a.targets.iter().collect(),
                    crate::StatementType::If(s) => {
                        if walk(&s.body, ctx, symbols, options)
                            || walk(&s.orelse, ctx, symbols, options)
                        {
                            return true;
                        }
                        continue;
                    }
                    crate::StatementType::For(s) => {
                        if walk(&s.body, ctx, symbols, options)
                            || walk(&s.orelse, ctx, symbols, options)
                        {
                            return true;
                        }
                        continue;
                    }
                    crate::StatementType::While(s) => {
                        if walk(&s.body, ctx, symbols, options)
                            || walk(&s.orelse, ctx, symbols, options)
                        {
                            return true;
                        }
                        continue;
                    }
                    crate::StatementType::Try(s) => {
                        if walk(&s.body, ctx, symbols, options) {
                            return true;
                        }
                        for h in &s.handlers {
                            if walk(&h.body, ctx, symbols, options) {
                                return true;
                            }
                        }
                        if walk(&s.orelse, ctx, symbols, options)
                            || walk(&s.finalbody, ctx, symbols, options)
                        {
                            return true;
                        }
                        continue;
                    }
                    crate::StatementType::With(s) => {
                        if walk(&s.body, ctx, symbols, options) {
                            return true;
                        }
                        continue;
                    }
                    _ => continue,
                };
                for t in targets {
                    if let ExprType::Attribute(attr) = t
                        && let ExprType::Name(n) = attr.value.as_ref()
                        && n.id != "self"
                        && crate::ast::tree::fetch_provenance::fetch_provenance(
                            &n.id, ctx, options, symbols,
                        )
                        .is_some()
                    {
                        return true;
                    }
                }
            }
            false
        }
        walk(body, ctx, symbols, options)
    }

    /// Infer the struct field list from the `self.attr = ...` stores in
    /// `__init__`. Stores are typed from annotated `__init__` parameters,
    /// simply-typed locals, and constructions of known classes; a store that
    /// cannot be typed is a loud error (a silently dropped attribute would
    /// diverge from Python). Includes stores that belong to a BASE class's
    /// fields; callers subtract those when they want the class's OWN fields.
    pub(crate) fn infer_fields(
        &self,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<Vec<(String, crate::TypeInfo)>, Box<dyn std::error::Error>> {
        let mut fields: Vec<(String, crate::TypeInfo)> = Vec::new();
        let Some(init) = self.init_method() else {
            return Ok(fields);
        };
        // Types known for names in the __init__ body: annotated
        // parameters first, then simply-typed locals.
        let mut name_types: std::collections::HashMap<String, crate::TypeInfo> =
            std::collections::HashMap::new();
        crate::collect_local_types(&init.body, &mut name_types);
        // Locals assigned from a CALL (`proxy = parse_url(...)` — urllib3's
        // ProxyManager) resolve their type from the callee's return
        // annotation, so a field stored from the local (`self.proxy =
        // proxy`) gets the class struct type.
        for stmt in &init.body {
            if let crate::StatementType::Assign(a) = &stmt.statement
                && let [ExprType::Name(n)] = a.targets.as_slice()
                && !name_types.contains_key(&n.id)
                && let ExprType::Call(call) = &a.value
                && let Some(t) =
                    crate::call_return_typeinfo(call, Some(&symbols), Some(&options))
            {
                name_types.insert(n.id.clone(), t);
            }
            // A local assigned a CONDITIONAL (`hashes_from_link = {} if
            // link_hash is None else link_hash.as_dict()` — pip's Link):
            // record it through the IfExp's branches so a field stored
            // from the local (`self._hashes = hashes_from_link`) types.
            if let crate::StatementType::Assign(a) = &stmt.statement
                && let [ExprType::Name(n)] = a.targets.as_slice()
                && !name_types.contains_key(&n.id)
                && let ExprType::IfExp(e) = &a.value
            {
                let ty = infer_field_type(&e.body, &name_types, symbols, options, &self.name)
                    .or_else(|| {
                        infer_field_type(&e.orelse, &name_types, symbols, options, &self.name)
                    });
                if let Some(t) = ty {
                    name_types.insert(n.id.clone(), t);
                }
            }
        }
        for p in init
            .args
            .posonlyargs
            .iter()
            .chain(init.args.args.iter())
            .chain(init.args.kwonlyargs.iter())
        {
            if let Some(ann) = p.annotation.as_deref() {
                // Mirror Parameter::to_rust: a `str` parameter becomes an
                // owned String local via the prologue. A parameter
                // annotated with a known class types the field as that
                // class's struct (composition).
                let ty = if matches!(ann, ExprType::Name(n) if n.id == "str") {
                    Some(crate::TypeInfo::String)
                } else if !matches!(ann, ExprType::Name(_)) {
                    if p.arg == "dist" {
                    }
                    // A union/container/alias annotation
                    // (`None | connection._TYPE_SOCKET_OPTIONS`,
                    // `tuple[str, int] | None`): resolve alias-aware.
                    let r = crate::resolve_alias_typeinfo(ann, symbols, options)
                        .or_else(|| crate::annotation_type_info(ann))
                        .or_else(|| {
                            if p.arg == "dist" {
                            }
                            // `T | None` where T is a CLASS (`load_only:
                            // Kind | None` — pip's Configuration): an
                            // Option of the class struct.
                            if crate::is_optional_annotation(ann) {
                                let inner = match ann {
                                    ExprType::BinOp(op) if crate::is_none_expr(&op.left) => {
                                        op.right.as_ref()
                                    }
                                    ExprType::BinOp(op) if crate::is_none_expr(&op.right) => {
                                        op.left.as_ref()
                                    }
                                    _ => return None,
                                };
                                match inner {
                                    ExprType::Name(n) => match symbols.get(&n.id) {
                                        Some(SymbolTableNode::ClassDef(_)) => Some(
                                            crate::TypeInfo::Option(Box::new(
                                                crate::TypeInfo::Class(n.id.clone()),
                                            )),
                                        ),
                                        Some(SymbolTableNode::ImportFrom(i)) => {
                                            let path = i.resolved_module_path(options);
                                            if crate::module_class_def(options, &path, &n.id)
                                                .is_some()
                                                || crate::resolve_imported_class(
                                                    options, &path, &n.id, 0,
                                                )
                                                .is_some()
                                            {
                                                Some(crate::TypeInfo::Option(Box::new(
                                                    crate::TypeInfo::Class(n.id.clone()),
                                                )))
                                            } else {
                                                None
                                            }
                                        }
                                        // A NewType alias (`Kind =
                                        // NewType("Kind", str)` — pip's
                                        // configuration): the base type.
                                        Some(SymbolTableNode::Assign {
                                            value: ExprType::Call(c),
                                            ..
                                        })
                                            if matches!(
                                                c.func.as_ref(),
                                                ExprType::Name(fnm) if fnm.id == "NewType"
                                            ) =>
                                        {
                                            Some(crate::TypeInfo::Option(Box::new(
                                                crate::TypeInfo::String,
                                            )))
                                        }
                                        _ => None,
                                    },
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        });
                    r
                } else if let ExprType::Name(n) = ann {
                    if p.arg == "dist" {
                    }
                    match symbols.get(&n.id) {
                        Some(SymbolTableNode::ClassDef(_)) => {
                            Some(crate::TypeInfo::Class(n.id.clone()))
                        }
                        // An IMPORTED class (`from .models import
                        // PreparedRequest` — requests/cookies.py): resolve
                        // through the defining module.
                        // A NewType alias (`canonical_name:
                        // NormalizedName` — pip's LinkEvaluator) resolves
                        // to its str base.
                        Some(SymbolTableNode::Assign {
                            value: ExprType::Call(c),
                            ..
                        }) if matches!(c.func.as_ref(), ExprType::Name(fnm) if fnm.id == "NewType") =>
                        {
                            Some(crate::TypeInfo::String)
                        }
                        Some(SymbolTableNode::ImportFrom(_)) => {
                            // An IMPORTED CLASS annotation that cannot be
                            // resolved (`dist: BaseDistribution` — pip's
                            // AlreadyInstalledCandidate) boxes as PyValue
                            // (the boxed-union divergence).
                            crate::resolve_alias_typeinfo(ann, symbols, options)
                                .or_else(|| Some(crate::TypeInfo::PyValue))
                        }
                        Some(SymbolTableNode::Alias(_))
                        // A TYPE-ALIAS name (`data: _TYPE_FIELD_VALUE` —
                        // urllib3's RequestField, `typing.Union[str,
                        // bytes]`): resolve the alias value.
                        | Some(SymbolTableNode::Assign { .. }) => {
                            crate::resolve_alias_typeinfo(ann, symbols, options)
                        }
                        _ => crate::annotation_type_info(ann).or_else(|| {
                            // A bare container annotation (`properties: dict`
                            // — a NamedTuple field, botocore's
                            // RuleSetEndpoint): the parameter itself lowers
                            // as a boxed PyValue (the unannotated
                            // fallback) — the field matches.
                            match ann {
                                ExprType::Name(n)
                                    if matches!(
                                        n.id.as_str(),
                                        "dict"
                                            | "list"
                                            | "set"
                                            | "frozenset"
                                            | "tuple"
                                            | "Mapping"
                                    ) =>
                                {
                                    Some(crate::TypeInfo::PyValue)
                                }
                                _ => None,
                            }
                        }),
                    }
                } else {
                    crate::annotation_type_info(ann)
                };
                if let Some(ty) = ty {
                    name_types.insert(p.arg.clone(), ty);
                }
            }
            // An UNANNOTATED __init__ parameter stored into a field
            // (`self._event_emitter = event_emitter` — botocore's
            // ClientArgsCreator): the field type is a boxed PyValue (the
            // parameter's value is unknown).
            else {
                name_types.insert(p.arg.clone(), crate::TypeInfo::PyValue);
            }
        }
        // Issue #120: the `**kwargs` parameter is a boxed heterogeneous
        // dict (`PyDict<String, PyValue>`); a field stored from it
        // (`self.conn_kw = conn_kw` — urllib3's ConnectionPool) takes the
        // same type.
        if let Some(kwarg) = &init.args.kwarg {
            name_types.insert(
                kwarg.arg.clone(),
                crate::TypeInfo::Dict(
                    Box::new(crate::TypeInfo::String),
                    Box::new(crate::TypeInfo::PyValue),
                ),
            );
        }
        // The `*args` parameter collects extra positionals as a boxed
        // heterogeneous Vec (`self._args = args` — s3transfer's
        // FunctionContainer).
        if let Some(vararg) = &init.args.vararg {
            name_types.insert(
                vararg.arg.clone(),
                crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue)),
            );
        }

        // Class-level annotated declarations (`config: dict[str, Any]`)
        // pin field types for stores whose value cannot be inferred
        // (`self.config = {}`).
        let mut class_annotations: std::collections::HashMap<String, crate::TypeInfo> =
            std::collections::HashMap::new();
        for stmt in &self.body {
            let annotated = match &stmt.statement {
                StatementType::AnnotatedName { name, annotation } => {
                    Some((name.clone(), annotation))
                }
                // An ANNOTATED ASSIGN (`_connect_callback:
                // typing.Callable[..., None] | None = None`): the
                // annotation pins the field type.
                StatementType::Assign(a) => {
                    if a.targets.len() == 1
                        && let ExprType::Name(n) = &a.targets[0]
                        && let Some(ann) = &a.annotation
                    {
                        Some((n.id.clone(), ann))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some((name, annotation)) = annotated {
                let ty = crate::annotation_type_info(annotation)
                    .or_else(|| crate::resolve_alias_typeinfo(annotation, symbols, options))
                    .or_else(|| {
                        // A dict-generic annotation with an unresolvable
                        // element (`dict[Kind, list[tuple[str,
                        // RawConfigParser]]]` — pip's Configuration, where
                        // Kind is a NewType and RawConfigParser is
                        // external): a boxed PyDict<String, PyValue>.
                        match annotation {
                            ExprType::Subscript(sub)
                                if matches!(
                                    sub.value.as_ref(),
                                    ExprType::Name(n)
                                        if matches!(n.id.as_str(), "dict" | "Dict")
                                ) =>
                            {
                                Some(crate::TypeInfo::Dict(
                                    Box::new(crate::TypeInfo::String),
                                    Box::new(crate::TypeInfo::PyValue),
                                ))
                            }
                            _ => None,
                        }
                    });
                if let Some(ty) = ty {
                    class_annotations.insert(name.clone(), ty);
                }
            }
        }

        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        // A store to a PROPERTY name (`self.retries = retries` where
        // retries is `@property` + `@retries.setter` — urllib3's
        // BaseHTTPResponse): Python's assignment invokes the SETTER method,
        // not a field write — the store must not create a struct field.
        stores.retain(|s| !self.is_property_setter(&s.attr));
        // An ANNOTATED store pins the field type (issue #121): a later
        // plain store of the same attribute (`self.headers: dict[str, str |
        // None] = {}` then `self.headers = dict(headers)` — urllib3's
        // RequestField) adopts the annotated type instead of conflicting.
        let mut annotated_fields: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for store in &stores {
            if store.annotation.is_some() {
                annotated_fields.insert(store.attr.clone());
            }
        }
        for store in &stores {
            // An explicit annotation pins the field type (`self._families:
            // frozenset[str] = frozenset(...)`, `self._last_latin_character:
            // CharInfo | None = None`); the value's shape infers it
            // otherwise.
            let ty = store.annotation.and_then(|a| {
                // A module-level TYPE ALIAS (`self._languages:
                // CoherenceMatches = languages`) resolves through symbols;
                // a class name maps to the struct ident; `T | None` wraps
                // in Option.
                let alias_ty = crate::resolve_alias_typeinfo(a, symbols, options);
                let ty_info = |t: &ExprType| -> Option<crate::TypeInfo> {
                    crate::annotation_type_info(t).or_else(|| match t {
                        ExprType::Name(n)
                            if matches!(
                                symbols.get(&n.id),
                                Some(SymbolTableNode::ClassDef(_))
                            ) =>
                        {
                            Some(crate::TypeInfo::Class(n.id.clone()))
                        }
                        _ => None,
                    })
                };
                if let Some(t) = alias_ty {
                    // The whole annotation resolved (aliases, Option<T>,
                    // containers) — use it as-is.
                    Some(t)
                } else if crate::is_optional_annotation(a) {
                    let inner = match a {
                        ExprType::BinOp(op) if crate::is_none_expr(&op.left) => op.right.as_ref(),
                        ExprType::BinOp(op) if crate::is_none_expr(&op.right) => op.left.as_ref(),
                        _ => return None,
                    };
                    let inner = ty_info(inner)?;
                    Some(crate::TypeInfo::Option(Box::new(inner)))
                } else {
                    ty_info(a)
                        .or_else(|| {
                            // A dict-generic annotation with an unresolvable
                            // element (`dict[Kind, list[tuple[str,
                            // RawConfigParser]]]` — pip's Configuration):
                            // a boxed PyDict<String, PyValue>.
                            match a {
                                ExprType::Subscript(sub)
                                    if matches!(
                                        sub.value.as_ref(),
                                        ExprType::Name(n)
                                            if matches!(n.id.as_str(), "dict" | "Dict")
                                    ) =>
                                {
                                    Some(crate::TypeInfo::Dict(
                                        Box::new(crate::TypeInfo::String),
                                        Box::new(crate::TypeInfo::PyValue),
                                    ))
                                }
                                _ => None,
                            }
                        })
                        .or_else(|| {
                            // A union of CLASSES
                            // (`InsecureCacheControlAdapter |
                            // InsecureHTTPAdapter` — pip's PipSession):
                            // the field is one of them — a boxed PyValue
                            // (the two-concrete-classes divergence).
                            crate::union_members(a).and_then(|ms| {
                                if !ms.is_empty()
                                    && ms.iter().all(|m| {
                                        matches!(m, ExprType::Name(n)
                                            if matches!(
                                                symbols.get(&n.id),
                                                Some(SymbolTableNode::ClassDef(_))
                                            ))
                                    })
                                {
                                    Some(crate::TypeInfo::PyValue)
                                } else {
                                    None
                                }
                            })
                        })
                }
            })
            .or_else(|| {
                class_annotations.get(&store.attr).cloned()
            })
            .or_else(|| {
                infer_field_type(store.value, &name_types, symbols, options, &self.name)
            });
            match ty {
                Some(ty) => {
                    match fields.iter().find(|(name, _)| *name == store.attr) {
                        None => fields.push((store.attr.clone(), ty)),
                        Some((_, prev)) if prev == &ty => {}
                        Some((_, prev))
                            if annotated_fields.contains(&store.attr) => {
                            // The attribute has an ANNOTATED store: the
                            // annotation's type wins over the value-shape
                            // inference of later plain stores (issue #121 —
                            // the same annotated-names rule as locals).
                        }
                        // A CONCRETE class store wins over a boxed-PyValue
                        // store (`self.headers = headers` where headers is a
                        // Mapping-union parameter — PyValue — then
                        // `self.headers = HTTPHeaderDict(headers)` — the
                        // concrete class — urllib3's BaseHTTPResponse): the
                        // real class is the field type. Symmetric: an
                        // earlier concrete class beats a LATER PyValue from
                        // an unannotated param (`self._session_ua_creator =
                        // UserAgentString(...)` then `= ua_creator` —
                        // botocore's ClientArgsCreator).
                        // Round 83: a None store joins a CONCRETE field
                        // type (`self._data = b""` in __init__, then
                        // `self._data = None` in decompress — urllib3's
                        // DeflateDecoder, whose `_data` is `bytes | None`):
                        // the field widens to Option<T> — the same
                        // declare-then-fill idiom the method-store join
                        // uses. The None store then lowers to the None
                        // member, and reads unwrap loudly where a concrete value is
                        // required (the round-83 Option→concrete coercion).
                        Some((_, prev))
                            if crate::is_none_expr(store.value)
                                && !matches!(prev, crate::TypeInfo::Option(_))
                                && !matches!(prev, crate::TypeInfo::PyValue) =>
                        {
                            let idx = fields
                                .iter()
                                .position(|(name, _)| name == &store.attr)
                                .unwrap();
                            fields[idx] = (
                                store.attr.clone(),
                                crate::TypeInfo::Option(Box::new(prev.clone())),
                            );
                        }
                        Some((_, prev))
                            if (matches!(prev, crate::TypeInfo::PyValue)
                                && !matches!(ty, crate::TypeInfo::PyValue))
                                || (matches!(ty, crate::TypeInfo::PyValue)
                                    && !matches!(prev, crate::TypeInfo::PyValue)) =>
                        {
                            let idx = fields
                                .iter()
                                .position(|(name, _)| name == &store.attr)
                                .unwrap();
                            let winner = if matches!(prev, crate::TypeInfo::PyValue) {
                                ty.clone()
                            } else {
                                prev.clone()
                            };
                            fields[idx] = (store.attr.clone(), winner);
                        }
                        Some((_, prev)) => {
                            // TWO different CONCRETE classes in different
                            // branches (`self._response_handler =
                            // ResourceHandler(...)` vs `RawHandler(...)` —
                            // boto3's ServiceAction): the field is a boxed
                            // PyValue (the callable is duck-dispatched).
                            if !matches!(prev, crate::TypeInfo::PyValue)
                                && !matches!(ty, crate::TypeInfo::PyValue)
                                && prev != &ty
                            {
                                let idx = fields
                                    .iter()
                                    .position(|(name, _)| name == &store.attr)
                                    .unwrap();
                                fields[idx] = (store.attr.clone(), crate::TypeInfo::PyValue);
                            } else {
                                return Err(format!(
                                    "attribute `self.{}` of class `{}` is assigned \
                                     conflicting types ({} and {}); a struct field needs \
                                     one type",
                                    store.attr, self.name, prev.display(), ty.display()
                                )
                                .into());
                            }
                        }
                    }
                }
                None => {
                    // A LATER store whose value cannot be inferred, when an
                    // EARLIER store already pinned the field (an annotated
                    // declaration — `self._trusted_host_adapter: A | B`
                    // then `self._trusted_host_adapter = insecure_adapter`
                    // — pip's PipSession): keep the pinned type.
                    if fields.iter().any(|(name, _)| *name == store.attr) {
                        continue;
                    }
                    return Err(format!(
                        "cannot infer a type for attribute `self.{}` of class `{}`: \
                         assign it from an annotated __init__ parameter, a literal, \
                         a constructed class instance, or an explicit attribute \
                         annotation (None-valued attributes are not supported yet)",
                        store.attr, self.name
                    )
                    .into());
                }
            }
        }

        // Issue #137 round 23: an attribute FIRST assigned OUTSIDE
        // `__init__` (`self.sock = self._new_conn()` in urllib3's
        // connect(), `self.proxy_is_verified = False` in its proxy path)
        // is a real Python attribute — attributes are created on
        // assignment, wherever that assignment lives — but only
        // `__init__` was scanned, so the struct had no field and every
        // read failed (E0609, 400 errors in urllib3's crate).
        //
        // Each other method contributes its stores, typed against its OWN
        // locals: a sibling method's local sharing the name must never
        // type this field. An uninferable value takes the boxed PyValue
        // rather than failing the conversion — the value is genuinely
        // unknown here, and the boxed reads/writes carry the existing
        // dynamic divergences. `__init__` still wins every name it knows.
        let mut method_field_stores: std::collections::BTreeMap<String, Vec<ObservedStore>> =
            std::collections::BTreeMap::new();
        for method in self.methods() {
            if method.name == "__init__" {
                continue;
            }
            let mut method_types: std::collections::HashMap<String, crate::TypeInfo> =
                std::collections::HashMap::new();
            crate::collect_local_types(&method.body, &mut method_types);
            let mut method_stores = Vec::new();
            collect_field_stores(&method.body, &mut method_stores);
            for store in method_stores {
                // A store to a PROPERTY name invokes the setter, not a
                // field write (same contract as the __init__ pass).
                if self.is_property_setter(&store.attr) {
                    continue;
                }
                if fields.iter().any(|(name, _)| *name == store.attr) {
                    // Round 83: a None store in a METHOD joins a
                    // __init__-declared CONCRETE field (`self._data = b""`
                    // in __init__, then `self._data = None` in decompress
                    // — urllib3's DeflateDecoder, whose `_data` is
                    // `bytes | None`): the field widens to Option<T>, the
                    // same declare-then-fill idiom the __init__ store join
                    // uses. The None store then lowers to the None member,
                    // and reads unwrap loudly where a concrete value is
                    // required (the round-83 Option→concrete coercion).
                    if crate::is_none_expr(store.value) {
                        let idx = fields
                            .iter()
                            .position(|(name, _)| name == &store.attr)
                            .unwrap();
                        let prev = fields[idx].1.clone();
                        if !matches!(prev, crate::TypeInfo::Option(_))
                            && !matches!(prev, crate::TypeInfo::PyValue)
                        {
                            fields[idx] = (
                                store.attr.clone(),
                                crate::TypeInfo::Option(Box::new(prev)),
                            );
                        }
                    }
                    continue;
                }
                // `self.last = self.value` — a store FROM another
                // attribute carries that attribute's type. Without this
                // the value reads as unknown and the new field boxes,
                // colliding with the typed field it was copied from.
                let from_sibling_field = match store.value {
                    ExprType::Attribute(a)
                        if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") =>
                    {
                        fields
                            .iter()
                            .find(|(f, _)| *f == a.attr)
                            .map(|(_, t)| t.clone())
                            .or_else(|| class_annotations.get(&a.attr).cloned())
                    }
                    _ => None,
                };
                let observed = if crate::is_none_expr(store.value) {
                    // A `None` store is the DECLARE half of Python's
                    // ubiquitous declare-then-fill idiom, not a type.
                    ObservedStore::NoneLiteral
                } else if let Some(t) = from_sibling_field {
                    ObservedStore::Typed(t)
                } else {
                    match infer_field_type(
                        store.value,
                        &method_types,
                        symbols,
                        options,
                        &self.name,
                    ) {
                        Some(t) => ObservedStore::Typed(t),
                        None => ObservedStore::Unknown,
                    }
                };
                method_field_stores
                    .entry(store.attr.clone())
                    .or_default()
                    .push(observed);
            }
        }

        // JOIN each method-assigned attribute's stores into ONE field
        // type. Python's `self.sock = None` in one method and `self.sock
        // = self._new_conn()` in another describe a single attribute
        // whose value is a socket OR None — `Option<Socket>` in Rust, and
        // nothing else is correct: typing it `Socket` breaks the None
        // store, typing it `PyValue` throws away the socket. Stores that
        // disagree on a CONCRETE type, or any store whose value cannot be
        // typed, fall back to the boxed value (the dynamic-attribute
        // divergence) rather than picking one arbitrarily.
        for (attr, observed) in method_field_stores {
            if fields.iter().any(|(name, _)| *name == attr) {
                continue;
            }
            let has_none = observed
                .iter()
                .any(|o| matches!(o, ObservedStore::NoneLiteral));
            let any_unknown = observed
                .iter()
                .any(|o| matches!(o, ObservedStore::Unknown));
            let concrete: Vec<crate::TypeInfo> = observed
                .iter()
                .filter_map(|o| match o {
                    ObservedStore::Typed(t) => Some(t.clone()),
                    _ => None,
                })
                .collect();
            let mut deduped: Vec<crate::TypeInfo> = Vec::new();
            for t in concrete {
                if !deduped.contains(&t) {
                    deduped.push(t);
                }
            }
            // A confident join — every store agreed on one concrete
            // type — WINS over a declared annotation: the annotation is a
            // first preference, and what the class actually stores is the
            // better evidence when the two disagree. An inconclusive join
            // (no concrete store, disagreeing stores, or a value rython
            // cannot read) falls back to the annotation, then to the
            // boxed value.
            let joined = match (any_unknown, deduped.as_slice()) {
                (false, [only]) => {
                    // An already-optional type absorbs the None store.
                    Some(if !has_none || matches!(only, crate::TypeInfo::Option(_)) {
                        only.clone()
                    } else {
                        crate::TypeInfo::Option(Box::new(only.clone()))
                    })
                }
                _ => None,
            };
            let ty = joined
                .or_else(|| class_annotations.get(&attr).cloned())
                .unwrap_or_else(|| crate::TypeInfo::PyValue);
            fields.push((attr, ty));
        }

        // An attribute this class READS but never assigns, with no
        // annotation to declare it, belongs to a BASE rython does not
        // model: urllib3's `HTTPConnection(http.client.HTTPConnection)`
        // reads `self.host`, `self.port` and `self.timeout`, which the
        // stdlib base owns. Python finds them on the base at runtime; the
        // generated struct had no field at all, so every read failed to
        // compile.
        //
        // They become BOXED fields — their shape is genuinely unknown
        // here — and the degradation is LOUD: the -W channel records that
        // the base is unmodeled, so nobody mistakes the boxed None for
        // the value the real base would have supplied. Only classes with
        // an UNMODELED base qualify: in a fully modeled hierarchy a read
        // of a never-assigned attribute is a real AttributeError, and
        // papering over it would be exactly the silent difference the
        // project forbids.
        let names_a_base = self.bases.iter().any(|b| {
            matches!(b, ExprType::Name(n) if n.id != "object" && !is_metadata_base_name(&n.id))
        });
        if names_a_base && self.base_class_with_options(symbols, options).is_none() {
            let mut reads = std::collections::BTreeSet::new();
            for method in self.methods() {
                collect_self_attr_reads(&method.body, &mut reads);
            }
            for name in reads {
                if fields.iter().any(|(f, _)| *f == name) {
                    continue;
                }
                // A dunder is the introspection protocol (`self.__class__`,
                // `self.__name__`), not a data attribute; a method or
                // property of this class is not a field either.
                if name.starts_with("__") {
                    continue;
                }
                if self.methods().any(|m| m.name == name) || self.is_property_setter(&name) {
                    continue;
                }
                if let Some(ty) = class_annotations.get(&name) {
                    fields.push((name, ty.clone()));
                    continue;
                }
                let message = format!(
                    "attribute `self.{}` of class `{}` is never assigned in the class: \
                     it belongs to a base that is external to the generated crate and \
                     is not modeled, so it lowers to a BOXED field that nothing \
                     populates (the external-base divergence)",
                    name, self.name
                );
                let mut warnings = options.definition_warnings.borrow_mut();
                if !warnings.iter().any(|w| *w == message) {
                    warnings.push(message);
                }
                drop(warnings);
                fields.push((name, crate::TypeInfo::PyValue));
            }
        }
        Ok(fields)
    }

    /// A class's OWN fields: its `__init__` stores minus the stores that
    /// belong to a base class's fields (those write into the embedded base
    /// struct). This is the layout `to_rust` gives the struct and the
    /// accessors `emit_trait` declares, so every consumer must use the same
    /// subtraction — an ancestor that re-assigns a field its own base owns
    /// (e.g. `class Dog(Animal): def __init__(self): self.name = ...` with
    /// no super().__init__) must not emit an accessor for it in its trait
    /// impl: the field physically lives in the ancestor's base struct.
    pub(crate) fn own_fields(
        &self,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<Vec<(String, crate::TypeInfo)>, Box<dyn std::error::Error>> {
        let own_stores = self.infer_fields(symbols, options)?;
        let base_owned: std::collections::HashSet<String> = self
            .base_class(symbols)
            .map(|b| {
                b.base_chain(symbols)
                    .iter()
                    .filter_map(|c| c.infer_fields(symbols, options).ok())
                    .flat_map(|f| f.into_iter().map(|(n, _)| n))
                    .collect()
            })
            .unwrap_or_default();
        Ok(own_stores
            .into_iter()
            .filter(|(name, _)| !base_owned.contains(name))
            .collect())
    }
}

/// Qualify CROSS-MODULE type names in rendered tokens: an accessor type
/// computed in an ancestor's defining module (`Option<Url>` — urllib3's
/// connection.py) names classes bare, which the DERIVING module need not
/// import. Each uppercase ident that positively resolves in the defining
/// module — a class defined there, or one it imports from another crate
/// module — is rewritten to its `crate::<module>::<Name>` path; anything
/// unresolved (primitives, std/stdpython types, enum variants) passes
/// through, as does any ident already behind a `::`.
pub(crate) fn qualify_cross_module_types(
    tokens: TokenStream,
    definer_path: &[String],
    a_syms: &SymbolTableScopes,
    a_opts: &PythonOptions,
    options: &PythonOptions,
) -> TokenStream {
    // Names BOUND inside the stream (let-bindings and everything in
    // parameter groups): a lowercase module-level match must not shadow
    // them — locals win in Python.
    let mut bound: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_bound_idents(&tokens, &mut bound);
    qualify_tokens(tokens, definer_path, a_syms, a_opts, options, &bound)
}

fn collect_bound_idents(tokens: &TokenStream, out: &mut std::collections::HashSet<String>) {
    use proc_macro2::TokenTree;
    let mut prev_word: Option<String> = None;
    let mut prev_prev: Option<String> = None;
    for tt in tokens.clone() {
        match &tt {
            TokenTree::Group(g) => {
                // Parameter groups and bodies alike: any ident inside a
                // group that could bind (over-collection is safe — it
                // only suppresses qualification).
                collect_bound_idents(&g.stream(), out);
                prev_prev = None;
                prev_word = None;
            }
            TokenTree::Ident(i) => {
                let w = i.to_string();
                // `let x`, `let mut x`, `for x` bind; a bare `mut` (the
                // `&mut T` position) does not.
                if prev_word.as_deref() == Some("let")
                    || prev_word.as_deref() == Some("for")
                    || (prev_word.as_deref() == Some("mut")
                        && prev_prev.as_deref() == Some("let"))
                {
                    out.insert(w.clone());
                }
                prev_prev = prev_word.take();
                prev_word = Some(w);
            }
            _ => {
                prev_prev = None;
                prev_word = None;
            }
        }
    }
}

fn qualify_tokens(
    tokens: TokenStream,
    definer_path: &[String],
    a_syms: &SymbolTableScopes,
    a_opts: &PythonOptions,
    options: &PythonOptions,
    bound: &std::collections::HashSet<String>,
) -> TokenStream {
    use proc_macro2::{TokenTree, Spacing};
    let mut out: Vec<TokenTree> = Vec::new();
    let mut after_path_sep = false;
    let mut prev_colon_joint = false;
    let mut prev_blocker = false;
    for tt in tokens {
        match tt {
            TokenTree::Group(g) => {
                let inner = qualify_tokens(
                    g.stream(),
                    definer_path,
                    a_syms,
                    a_opts,
                    options,
                    bound,
                );
                let mut ng = proc_macro2::Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.push(TokenTree::Group(ng));
                after_path_sep = false;
                prev_colon_joint = false;
                prev_blocker = false;
            }
            TokenTree::Punct(p) => {
                if p.as_char() == ':' {
                    if prev_colon_joint {
                        after_path_sep = true;
                        prev_colon_joint = false;
                    } else if p.spacing() == Spacing::Joint {
                        prev_colon_joint = true;
                    } else {
                        after_path_sep = false;
                        prev_colon_joint = false;
                    }
                } else {
                    after_path_sep = false;
                    prev_colon_joint = false;
                }
                // A field/method access position: the following ident is
                // a MEMBER, never a module path root.
                prev_blocker = p.as_char() == '.';
                out.push(TokenTree::Punct(p));
            }
            TokenTree::Ident(id) => {
                let name = id.to_string();
                // Declaration positions (`fn close`, `let conn`, …) and
                // member accesses never qualify.
                let blocked = after_path_sep || prev_blocker;
                // `mut` is NOT a blocker: `&mut Type` positions must still
                // qualify (let-mut locals are handled by the bound set).
                prev_blocker = matches!(
                    name.as_str(),
                    "fn" | "let" | "for" | "impl" | "trait" | "struct" | "mod" | "use"
                        | "pub"
                );
                // Class names are CapWords, possibly private-prefixed
                // (`_ResponseOptions` — urllib3's _base_connection).
                let is_capword = name
                    .trim_start_matches('_')
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_uppercase());
                let target: Option<Vec<String>> = if blocked
                    || matches!(
                        name.as_str(),
                        "Self" | "Some" | "None" | "Ok" | "Err" | "Default" | "self"
                    ) {
                    None
                } else if is_capword {
                    // A root's SUM TYPE (`AnyContentDecoder` — hierarchy.rs)
                    // lives beside its root class.
                    let root_of_any = name
                        .strip_prefix("Any")
                        .filter(|r| crate::ast::tree::hierarchy::is_polymorphic_root(r));
                    if crate::ast::tree::module::module_class_traits(a_opts, definer_path)
                        .contains_key(&name)
                        || crate::module_class_def(a_opts, definer_path, &name).is_some()
                        || root_of_any.is_some_and(|r| {
                            crate::module_class_def(a_opts, definer_path, r).is_some()
                        })
                    {
                        Some(definer_path.to_vec())
                    } else {
                        match a_syms.get(&name) {
                            // Follow RE-EXPORT chains to the class's real
                            // module (`from .connection import ProxyConfig`
                            // where connection.py re-exports it from
                            // _base_connection — urllib3).
                            Some(SymbolTableNode::ImportFrom(i)) => {
                                let p = i.resolved_module_path(a_opts);
                                options
                                    .module_defs
                                    .contains_key(&p)
                                    .then(|| {
                                        crate::resolve_imported_class_with_path(
                                            a_opts, &p, &name, 0,
                                        )
                                        .map(|(_, _, terminal)| terminal)
                                    })
                                    .flatten()
                            }
                            _ => None,
                        }
                    }
                } else if !bound.contains(&name) {
                    // A lowercase MODULE-LEVEL item of the defining module
                    // (`_close_pool_connections(...)`, the `log` static —
                    // urllib3's connectionpool, referenced from re-emitted
                    // override bodies). Locals win in Python, so anything
                    // bound in this stream stays untouched.
                    match a_syms.get(&name) {
                        Some(SymbolTableNode::FunctionDef(_))
                        | Some(SymbolTableNode::Assign { .. }) => {
                            Some(definer_path.to_vec())
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match target {
                    Some(path) => {
                        let segs: Vec<_> =
                            path.iter().map(|p| crate::safe_ident(p)).collect();
                        let ident = crate::safe_ident(&name);
                        out.extend(quote!(crate #(::#segs)* :: #ident));
                    }
                    None => out.push(TokenTree::Ident(id)),
                }
                after_path_sep = false;
                prev_colon_joint = false;
            }
            other => {
                out.push(other);
                after_path_sep = false;
                prev_colon_joint = false;
                prev_blocker = false;
            }
        }
    }
    out.into_iter().collect()
}

/// The comparable SIGNATURE of the first `fn` item in rendered tokens:
/// (parameter tokens, return-type tokens), whitespace-normalized. Used to
/// decide whether a cross-module override agrees with its trait's
/// declaration — Python allows covariant overrides (`_new_conn() ->
/// socks.socksocket` over `-> socket.socket`), Rust trait impls do not.
pub(crate) fn fn_signature_key(tokens: &TokenStream) -> Option<(String, String)> {
    use proc_macro2::{Delimiter, TokenTree};
    let mut iter = tokens.clone().into_iter().peekable();
    // Find `fn <name>`.
    while let Some(tt) = iter.next() {
        if matches!(&tt, TokenTree::Ident(i) if i == "fn") {
            let _name = iter.next()?;
            // Optional generics `<...>` then the parameter group.
            let mut params: Option<String> = None;
            let mut ret = String::new();
            let mut in_ret = false;
            for tt in iter.by_ref() {
                match &tt {
                    TokenTree::Group(g)
                        if g.delimiter() == Delimiter::Parenthesis && params.is_none() =>
                    {
                        params = Some(g.stream().to_string());
                    }
                    TokenTree::Group(g)
                        if g.delimiter() == Delimiter::Brace && params.is_some() =>
                    {
                        break;
                    }
                    _ if params.is_some() => {
                        if in_ret {
                            ret.push_str(&tt.to_string());
                            ret.push(' ');
                        } else if matches!(&tt, TokenTree::Punct(p) if p.as_char() == '>') {
                            // The `->`'s closing half (the '-' preceded).
                            in_ret = true;
                        }
                    }
                    _ => {}
                }
            }
            let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
            return Some((norm(&params?), norm(&ret)));
        }
    }
    None
}

/// A `self.field = field` statement for a dataclass-synthesized __init__.
fn dataclass_store_stmt(field: &str) -> Statement {    let name = || ExprType::Name(crate::ast::tree::name::Name { id: field.to_string() });
    let attr = ExprType::Attribute(crate::ast::tree::attribute::Attribute {
        value: Box::new(ExprType::Name(crate::ast::tree::name::Name {
            id: "self".to_string(),
        })),
        attr: field.to_string(),
        ctx: String::new(),
    });
    Statement {
        lineno: None,
        col_offset: None,
        end_lineno: None,
        end_col_offset: None,
        statement: StatementType::Assign(Assign {
            targets: vec![attr],
            value: name(),
            type_comment: None,
            annotation: None,
        }),
    }
}

/// All parameter names of a method, `self` included.
fn method_param_names(m: &FunctionDef) -> Vec<String> {    m.args
        .posonlyargs
        .iter()
        .chain(m.args.args.iter())
        .chain(m.args.kwonlyargs.iter())
        .map(|p| p.arg.clone())
        .chain(m.args.vararg.iter().map(|p| p.arg.clone()))
        .chain(m.args.kwarg.iter().map(|p| p.arg.clone()))
        .collect()
}

/// A `self.attr = value` assignment found in `__init__`, used for field
/// inference.
struct FieldStore<'a> {
    attr: String,
    value: &'a ExprType,
    /// The store's annotation (`self.x: list[float] = ...`), which pins
    /// the field type when the value cannot be inferred.
    annotation: Option<&'a ExprType>,
}

/// Collect `self.attr = ...` stores anywhere in a body (recursing into
/// control flow), in first-store order.
fn collect_field_stores<'a>(body: &'a [Statement], out: &mut Vec<FieldStore<'a>>) {
    for stmt in body {
        match &stmt.statement {
            // A bare ANNOTATED declaration (`self._trusted_host_adapter:
            // InsecureCacheControlAdapter | InsecureHTTPAdapter` — pip's
            // PipSession): pins the field type even though nothing is
            // stored on that statement.
            StatementType::AnnotatedName { name, annotation } => {
                let name = name.to_string();
                if let Some(attr) = name.strip_prefix("self.") {
                    out.push(FieldStore {
                        attr: attr.to_string(),
                        value: &crate::ExprType::NoneType(
                            crate::ast::tree::constant::Constant(None),
                        ),
                        annotation: Some(annotation),
                    });
                }
            }
            StatementType::Assign(assign) => {
                for target in &assign.targets {
                    if let ExprType::Attribute(attr) = target {
                        if matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                            out.push(FieldStore {
                                attr: attr.attr.clone(),
                                value: &assign.value,
                                annotation: assign.annotation.as_ref(),
                            });
                        }
                    }
                }
            }            StatementType::If(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::For(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::While(s) => {
                collect_field_stores(&s.body, out);
                collect_field_stores(&s.orelse, out);
            }
            StatementType::With(s) => collect_field_stores(&s.body, out),
            StatementType::Try(s) => {
                collect_field_stores(&s.body, out);
                for h in &s.handlers {
                    collect_field_stores(&h.body, out);
                }
                collect_field_stores(&s.orelse, out);
                collect_field_stores(&s.finalbody, out);
            }
            _ => {}
        }
    }
}

impl CodeGen for ClassDef {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(mut self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        // Synthesize the dataclass __init__ BEFORE the class enters the
        // symbol table: call sites resolve the class through this clone,
        // and a @dataclass without __init__ must look constructed.
        if let Err(_e) = self.synthesize_dataclass_init() {
            symbols.insert(
                self.name.clone(),
                SymbolTableNode::ClassDef(self),
            );
            return symbols;
        }
        symbols.insert(self.name.clone(), SymbolTableNode::ClassDef(self));
        symbols
    }

    fn to_rust(
        mut self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // `self` is consumed by value and bound mutably so the dataclass
        // synthesis can prepend the generated __init__ to the body.
        let class_name = crate::safe_ident(&self.name);

        // ---- Exception classes ----
        // A class inheriting a builtin exception (or another custom
        // exception) is lowered as a MARKER: the runtime models exceptions
        // as string-tagged PyException values (raise/except match by name),
        // so the class has no data or methods to carry — only its name.
        // This keeps `raise IDNAError("msg")` / `except IDNAError` working
        // for the ubiquitous custom-exception pattern (requests'
        // RequestException, urllib3's HTTPError, idna's IDNAError).
        // Exception classes with *args/**kwargs __init__ (idna) lower
        // through here without tripping the variadic-parameter guard.
        if is_exception_class(&self) {
            // A real doc ATTRIBUTE: interpolating a String into quote!
            // yields a string-literal token — `""` in item position, a
            // parse error in the generated crate.
            let doc = self.get_docstring().map(|d| quote!(#[doc = #d]));
            return Ok(quote! {
                #doc
                #[allow(dead_code)]
                pub struct #class_name;
            });
        }

        // A typing.Protocol subclass (or a class whose base resolves to a
        // typing import): type-only — its methods are stubs, so it lowers
        // as an empty marker struct like an exception class. The base may
        // be a Subscript (`Protocol[_T_co]`) which otherwise errors.
        let is_protocol = self.decorator_list.iter().any(|d| {
            matches!(
                d,
                ExprType::Name(n) if n.id == "runtime_checkable"
            )
        }) || self.bases.iter().any(|b| match b {
            ExprType::Name(n) => n.id == "Protocol",
            ExprType::Subscript(s) => {
                matches!(s.value.as_ref(), ExprType::Name(n) if n.id == "Protocol")
            }
            _ => false,
        });
        if is_protocol {
            // Same string-vs-attribute distinction as the exception arm.
            let doc = self.get_docstring().map(|d| quote!(#[doc = #d]));
            return Ok(quote! {
                #doc
                #[allow(dead_code)]
                pub struct #class_name;
            });
        }

        // ---- Class keywords (`metaclass=...`, `total=...`, `**kwargs`) ----
        // `metaclass=abc.ABCMeta` is the Protocol/ABC idiom: the metaclass
        // only enforces abstract-method instantiation at runtime, which has
        // no Rust analogue — lowering the class as a plain class keeps the
        // data and methods, so it is a LOSSY no-op (surfaced through the
        // -W channel, never silent). Any metaclass NAME (`metaclass=
        // LexerMeta` — pygments' Lexer) is the same: a class factory as a
        // value — metadata. `total=...` on a TypedDict is the
        // all-fields-optional marker — also metadata (the class is a plain
        // struct). Any other class keyword changes class
        // creation itself and is a loud error.
        for kw in &self.keywords {
            let is_metaclass = kw.arg.as_deref() == Some("metaclass")
                && (matches!(
                    &kw.value,
                    ExprType::Attribute(a)
                        if matches!(a.value.as_ref(), ExprType::Name(n)
                            if crate::AnnotationModule::from_name(&n.id)
                                == Some(crate::AnnotationModule::Abc))
                            && a.attr == "ABCMeta"
                ) || matches!(&kw.value, ExprType::Name(_)));
            // TypedDict's `total=` keyword (`class _VersionReplace(
            // TypedDict, total=False)` — pip's packaging.version): the
            // optionality marker is metadata — dropped.
            let is_total = kw.arg.as_deref() == Some("total");
            if is_metaclass {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: `metaclass=...` is dropped (metaclass \
                     machinery has no Rust analogue); the class lowers as a \
                     plain class",
                    self.name
                ));
            } else if is_total {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: `total=...` (TypedDict) is dropped; the class lowers \
                     as a plain struct",
                    self.name
                ));
            } else {
                let what = kw
                    .arg
                    .as_deref()
                    .map(|a| format!("`{a}=...`"))
                    .unwrap_or_else(|| "`**kwargs`".to_string());
                return Err(format!(
                    "class `{}` uses the class keyword {}, which is not supported \
                     (only `metaclass=abc.ABCMeta` is accepted, as a lossy no-op)",
                    self.name, what
                )
                .into());
            }
        }

        // ---- Decorators ----
        // `@dataclass` (with or without `(frozen=..., slots=...)` args)
        // synthesizes `__init__` from the annotated class-level fields.
        // Any other class decorator changes class creation and is a loud
        // error, never silently dropped — through the systematic registry.
        let is_dataclass = self.is_dataclass();
        for d in &self.decorator_list {
            // A SAME-PACKAGE imported wrapper decorator (`@rich_repr` —
            // rich's color.py, imported from `._vendor.rich.repr`):
            // auto-generates __rich_repr__ from the class's own method —
            // a repr metadata wrapper; the class lowers with its own
            // methods (the local-wrapper divergence, #131). A SAME-MODULE
            // decorator function (`@auto` — rich's repr.py __main__
            // demo) is the same shape.
            if matches!(d, ExprType::Name(n)
                if matches!(
                    symbols.get(&n.id),
                    Some(crate::SymbolTableNode::ImportFrom(_))
                        | Some(crate::SymbolTableNode::FunctionDef(_))
                ))
            {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: decorator `{}` (a local wrapper function) \
                     is dropped; the class lowers directly (documented divergence)",
                    self.name,
                    match d {
                        ExprType::Name(n) => n.id.clone(),
                        _ => String::new(),
                    }
                ));
                continue;
            }
            match crate::parse_decorator(std::slice::from_ref(d))? {
                Some(crate::Decorator::DataClass) => {}
                // A metadata class decorator (`@functools.total_ordering` —
                // pip's Link): no runtime effect in the lowered struct.
                Some(crate::Decorator::Property) => {}
                Some(other) => {
                    return Err(format!(
                        "class `{}` uses the decorator `{}`, which is not supported \
                         on a class (only `@dataclass` is accepted, as a synthesized \
                         __init__)",
                        self.name,
                        other.describe(),
                    )
                    .into());
                }
                None => {}
            }
        }
        if is_dataclass && self.init_method().is_some() {
            // A dataclass-SHAPED class with a REAL __init__
            // (`InstallationCandidate` — pip's models, a frozen-style
            // class using object.__setattr__): the real constructor wins
            // (the synthesis is idempotent and leaves it untouched); the
            // annotated fields still type the struct.
            let _ = ();
        }
        // Synthesize the dataclass __init__ from the annotated fields and
        // PREPEND it to the body: every downstream consumer (init_method,
        // method_on_mro, infer_fields, the constructor emission) sees it as
        // a real method, so the rest of the lowering is unchanged. Each
        // field becomes a parameter; a field with a default (`count: int =
        // 0`) becomes a defaulted parameter; the body stores each
        // `self.field = field`. (find_symbols already ran the same
        // synthesis on the symbol-table clone so call sites resolve the
        // constructed class; this run covers the emitted class itself.)
        self.synthesize_dataclass_init()?;

        // ---- Base resolution ----
        // Single inheritance only. `object` is every class's implicit base,
        // so naming it changes nothing. The base must be a class defined in
        // this module: the embedded-struct + trait scheme cannot represent an
        // imported or builtin base faithfully.
        // A base that is not a simple same-module Name (a dotted
        // `queue.Queue`, a call, ...) cannot lower in the embedded-struct +
        // trait scheme — loud error, never a silent drop. `typing.NamedTuple`
        // is the exception: it is field metadata, not a real base — the
        // annotated fields lower like a dataclass (urllib3's ProxyConfig).
        // A `typing.*` base (Generic[T], MutableMapping[K, V], Protocol,
        // ...) is metadata, not a structural base. An EXTERNAL-module
        // attribute base (`io.RawIOBase` — urllib3's emscripten
        // _ReadStream) is likewise unresolvable: metadata — the class
        // lowers as a plain struct (the class-as-value divergence).
        let external_attr_base = |b: &ExprType| -> bool {
            // ANY dotted-attribute base — external or in a sibling module
            // of the crate (`requests.Session` — pip's vendored requests,
            // `importlib.metadata.Distribution`): a foreign module's class
            // the embedded-struct scheme cannot represent — metadata.
            if matches!(b, ExprType::Attribute(_))
                && crate::root_name(b).is_some()
            {
                return true;
            }
            let external_import = |sym: &SymbolTableNode| -> bool {
                match sym {
                    SymbolTableNode::Import(i) => !options.module_defs.contains_key(
                        &i.names
                            .first()
                            .map(|al| {
                                al.name
                                    .split('.')
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default(),
                    ),
                    SymbolTableNode::ImportFrom(i) => !options
                        .module_defs
                        .contains_key(&i.resolved_module_path(&options)),
                    _ => false,
                }
            };
            if let ExprType::Attribute(_) = b {
                // The chain ROOT (`importlib.metadata.Distribution` —
                // pip's WheelDistribution): a dotted import whose module
                // is not in the crate — either the root name or the full
                // dotted chain is the registered import symbol.
                if let Some(root) = crate::root_name(b) {
                    if symbols
                        .get(root)
                        .is_some_and(|s| external_import(s))
                    {
                        return true;
                    }
                }
                // `import importlib.metadata` registers only a dotted
                // PREFIX of the chain: check every prefix
                // ("importlib", "importlib.metadata", ...) for an external
                // import symbol.
                if let Some(parts) = crate::ast::tree::call::dotted_module_path(b) {
                    for i in (1..=parts.len()).rev() {
                        let key = parts[..i].join(".");
                        if let Some(s) = symbols.get(&key)
                            && external_import(s)
                        {
                            return true;
                        }
                    }
                }
                false
            } else {
                false
            }
        };
        if let Some(_bad) = self.bases.iter().find(|b| {
            !matches!(b, ExprType::Name(_))
                && !is_typing_base(b)
                && !external_attr_base(b)
        }) {
            return Err(format!(
                "class `{}` inherits from a base rython cannot lower (only single \
                 inheritance from classes defined in this module is supported); \
                 restructure the class hierarchy (issue: the PyPI sweep)",
                self.name
            )
            .into());
        }
        // `object`, `Enum`/`IntEnum`/`Flag` (and `typing.NamedTuple`,
        // filtered above) are metadata, not structural bases: the class
        // lowers as a plain struct (urllib3's `_Sentinel(Enum)`,
        // charset_normalizer's `CoherenceMatch(TypedDict)`).
        let is_metadata_base = |id: &str| is_metadata_base_name(id);
        let real_bases: Vec<&str> = self
            .bases
            .iter()
            .filter_map(|b| match b {
                ExprType::Name(n) if !is_metadata_base(&n.id) => Some(n.id.as_str()),
                _ => None,
            })
            .collect();
        if real_bases.len() > 1 {
            // MULTIPLE same-module mixin bases (`PreparedRequest(
            // RequestEncodingMixin, RequestHooksMixin)` — requests): the
            // embedded-struct scheme keeps the FIRST base; the remaining
            // mixins' methods are NOT inherited (the documented divergence,
            // issue #122-family). Single-inheritance classes are unchanged.
            let _ = real_bases; // first base used below; extras dropped
        }
        let base: Option<ClassDef> = match real_bases.first() {
            None => None,
            Some(base_name) => match symbols.get(base_name) {
                // An ALIAS base (`from http.client import HTTPConnection
                // as _HTTPConnection`): follow to the canonical name.
                Some(SymbolTableNode::Alias(canonical)) => {
                    match symbols.get(canonical) {
                        Some(SymbolTableNode::ImportFrom(_)) => {
                            let path = match symbols.get(canonical) {
                                Some(SymbolTableNode::ImportFrom(i)) => {
                                    i.resolved_module_path(&options)
                                }
                                _ => Vec::new(),
                            };
                            crate::resolve_imported_class(&options, &path, canonical, 0)
                                .map(|(c, _)| c)
                        }
                        // The canonical name can be SHADOWED by the class
                        // being defined (`from http.client import
                        // HTTPConnection as _HTTPConnection` then `class
                        // HTTPConnection(_HTTPConnection)` — urllib3): the
                        // alias meant the import, not the later class, so
                        // a self-resolution is the shadowed EXTERNAL base
                        // — metadata (the inherited behavior is the
                        // documented divergence), never a self-supertrait
                        // and an infinitely-sized embedded struct.
                        Some(SymbolTableNode::ClassDef(c)) if c.name != self.name => {
                            Some(c.clone())
                        }
                        _ => None,
                    }
                }
                // An IMPORTED base (`from http.cookiejar import CookieJar` —
                // requests' RequestsCookieJar): resolve through the defining
                // module; an UNRESOLVABLE external base (a stdlib class
                // rython has no lowering for) is treated as metadata — the
                // class lowers as a plain struct, and the inherited
                // behavior is the documented divergence.
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let path = i.resolved_module_path(&options);
                    match crate::resolve_imported_class(&options, &path, base_name, 0) {
                        Some((c, _)) => Some(c),
                        None => None,
                    }
                }
                Some(SymbolTableNode::ClassDef(c)) => {
                    // A base that resolves to an already-visited class in
                    // this class's own chain is a cycle (`class A(A)` —
                    // Python looks the base up in the outer scope, which
                    // the symbol table can only see as the rebound name).
                    // The embedded-struct scheme cannot represent it; fail
                    // loudly instead of emitting an infinitely-sized
                    // struct or looping forever in base_chain.
                    if let Some(cycle) = self.base_cycle(&symbols) {
                        return Err(format!(
                            "class `{}` cannot inherit from `{}`: cyclic inheritance",
                            self.name, cycle,
                        )
                        .into());
                    }
                    Some(c.clone())
                }
                // A base bound to a `namedtuple(...)` /
                // `typing.NamedTuple(...)` CALL (`_ServiceContext =
                // namedtuple('ServiceContext', [...])` — boto3's utils):
                // inheriting a namedtuple — the namedtuple's fields are
                // unmodeled, so the base is metadata; the class lowers as a
                // plain struct (the namedtuple-base divergence).
                Some(SymbolTableNode::Assign {
                    value: ExprType::Call(call),
                    ..
                }) if matches!(call.func.as_ref(), ExprType::Name(n) if n.id == "namedtuple")
                    || matches!(call.func.as_ref(), ExprType::Attribute(a)
                        if a.attr == "NamedTuple"
                            && matches!(a.value.as_ref(), ExprType::Name(n)
                                if crate::is_typing(&n.id))) =>
                {
                    None
                }
                _ => {
                    return Err(format!(
                        "class `{}` inherits from `{}`, which is not a class defined \
                         in this module; imported bases and built-in bases are not \
                         supported yet",
                        self.name, base_name,
                    )
                    .into());
                }
            },
        };
        let in_hierarchy = base.is_some() || options.hierarchy_classes.contains(&self.name);

        // Only methods (plus a docstring, `pass`, and — for a @dataclass —
        // the annotated field declarations) are supported in class bodies.
        // Class-level assignments (class attributes) would need a
        // shared-state story; erroring is the loud option.
        let body_start = if self.get_docstring().is_some() { 1 } else { 0 };
        // Class-level LITERAL constants (`FIRST_MEMBER = 0`,
        // `SWALLOW_DATA = 2` — urllib3's GzipDecoderState): emitted as
        // `impl X { pub const NAME: T = v; }` so class-attribute reads
        // (`X.NAME`, rendered `X::NAME` — attribute.rs) resolve.
        let mut class_constants = TokenStream::new();
        // Class-level COMPUTED constants (`DEFAULT_ALLOWED_METHODS =
        // frozenset(["HEAD", ...])` — urllib3's Retry): not literal, so a
        // plain const cannot hold them; emitted as
        // `impl X { pub static NAME: LazyLock<T> = ...; }` — the same
        // LazyLock model module-level promotion uses. Reads inside class
        // methods (dropped-default inlining, `cls.NAME`) deref-clone the
        // static (attribute.rs / call.rs consult the same shape).
        let mut class_lazylock_constants = TokenStream::new();
        let mut class_const_accessors = TokenStream::new();
        for stmt in self.body.iter().skip(body_start) {
            match &stmt.statement {
                StatementType::FunctionDef(_) | StatementType::Pass => {}
                // A class-level literal constant assignment (int/float/
                // bool/string): an associated const, not a struct field.
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && let ExprType::Name(n) = &a.targets[0]
                        && let Some(ty) = crate::ast::tree::module::const_static_type(&a.value) =>
                {
                    let ident = crate::safe_ident(&n.id);
                    let value = a
                        .value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    class_constants.extend(quote!(pub const #ident: #ty = #value;));
                }
                // A class-level COMPUTED constant: single-store name
                // assigned a non-literal value (frozenset/dict/list/set
                // literal or a constructor call). Python resolves these at
                // class-definition time and they are READ as
                // class-attributes; the LazyLock static keeps them
                // importable inside method defaults. Values that reference
                // module state (a plain Name/attribute read) stay dropped
                // (the class-as-value divergence) — only literal-built
                // values are promoted.
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && let ExprType::Name(n) = &a.targets[0]
                        && crate::ast::tree::module::const_static_type(&a.value).is_none()
                        && class_body_computed_constant(&a.value) =>
                {
                    // The static lives at MODULE level under a
                    // class-mangled name — associated statics are not
                    // legal Rust (issue #137: urllib3's
                    // RequestMethods._encode_url_methods) — with the
                    // VALUE's inferred type when one exists (a frozenset
                    // of literals is a concrete set the boxed PyValue
                    // cannot hold); the PyValue wrap only as a fallback.
                    let ident = crate::class_const_static_ident(&self.name, &n.id);
                    let rhs = a
                        .value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                    let stripped =
                        crate::ast::tree::call::strip_trailing_question(&rhs);
                    let value_tokens = if stripped.to_string() != rhs.to_string() {
                        quote!(match #stripped {
                            Ok(__rython_v) => __rython_v,
                            Err(__rython_e) => panic!(
                                "class-level `{}` initialization failed: {}",
                                stringify!(#ident),
                                __rython_e
                            ),
                        })
                    } else {
                        stripped
                    };
                    let ti = crate::infer_type(Some(&ctx), &a.value, &options, &symbols);
                    let concrete = !matches!(ti, crate::TypeInfo::PyObject)
                        && !crate::ast::tree::module::type_contains_uninferred(&ti);
                    let (ty, init) = if concrete {
                        (ti.to_rust_type(), value_tokens)
                    } else {
                        (
                            quote!(stdpython::PyValue),
                            quote!(stdpython::PyValue::from(#value_tokens)),
                        )
                    };
                    class_lazylock_constants.extend(quote! {
                        pub static #ident: std::sync::LazyLock<#ty> =
                            std::sync::LazyLock::new(|| #init);
                    });
                    // The associated ACCESSOR keeps `Class::NAME`-shaped
                    // reads working — importable through the class alone,
                    // no static import needed at cross-module call sites.
                    let accessor = crate::safe_ident(&n.id);
                    class_const_accessors.extend(quote! {
                        #[allow(non_snake_case)]
                        pub fn #accessor() -> #ty {
                            (*#ident).clone()
                        }
                    });
                }
                // A string-literal Expr anywhere in the body (a docstring
                // placed after a class constant — botocore's TokenSigner):
                // metadata, no runtime effect.
                StatementType::Expr(e)
                    if matches!(
                        &e.value,
                        ExprType::Constant(c)
                            if matches!(&c.0, Some(litrs::Literal::String(_)))
                    ) => {}
                // `__slots__` / `__attrs__ = (...)` — class metadata
                // declarations with no runtime effect in rython (fields
                // are struct members; `__attrs__` only drives pickling).
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(
                            &a.targets[0],
                            ExprType::Name(n) if matches!(n.id.as_str(), "__slots__" | "__attrs__")
                        ) => {}
                // An ENUM member (`class _Sentinel(Enum): not_passed =
                // auto()`) is metadata: the members are sentinel values,
                // not struct fields (urllib3).
                StatementType::Assign(_)
                    if self.bases.iter().any(|b| {
                        matches!(b, ExprType::Name(n) if is_enum_base_name(&n.id))
                    }) => {}
                // A class-level CONSTANT assignment (`_encode_url_methods
                // = {...}`, `default_port = port_by_scheme["https"]`, a
                // method alias `getheaders = getlist`): metadata the struct
                // cannot express (the class lowers as a plain struct with
                // its fields). Reads of these attributes fail at rustc —
                // the documented class-as-value divergence.
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Name(_)) => {}
                // A class-level VERSION-/PLATFORM-GATED block (`if
                // sys.version_info < ...:` — urllib3's HTTPConnection;
                // `if sys.platform.startswith('java'): ... else: ...` —
                // distlib's ResourceFinder): the constants and
                // version-conditional method definitions inside are
                // metadata — tolerated. AugAssigns (`CONTENT_DECODERS +=
                // ["br"]` under `if brotli is not None:` — urllib3's
                // BaseHTTPResponse) are class-level constant mutations,
                // also metadata.
                StatementType::If(i)
                    if class_level_metadata_body(&i.body)
                        && class_level_metadata_body(&i.orelse) => {}
                // A bare `name: T` declaration is a @dataclass field; for a
                // non-dataclass class it is a plain annotation (no runtime
                // effect) — allowed either way.
                StatementType::AnnotatedName { .. } => {}
                // A defaulted @dataclass field (`count: int = 0`) arrives as
                // an annotated Assign; the dataclass synthesis consumed it
                // into the generated __init__ BEFORE this loop ran (it
                // prepends the init and keeps the field declarations in the
                // body), so what remains here is the declaration the
                // synthesis could not consume — only valid for dataclasses.
                StatementType::Assign(a)
                    if is_dataclass && a.annotation.is_some() =>
                {
                    if let [ExprType::Name(_)] = a.targets.as_slice() {
                        // consumed by the synthesis; nothing to emit here
                    } else {
                        return Err(format!(
                            "class `{}` is a @dataclass with a class-level \
                             assignment that is not an annotated field",
                            self.name
                        )
                        .into());
                    }
                }
                // A class-level CONSTANT on a dataclass-shaped class
                // (`_hash_url_fragment_re = re.compile(...)` — pip's
                // LinkHash): class metadata, no runtime effect in the
                // lowered struct — dropped.
                StatementType::Assign(a)
                    if is_dataclass
                        && a.annotation.is_none()
                        && a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Name(_)) => {}
                // A class-level SUBSCRIPT store (`TYPE_CHECKER[
                // "package_name"] = ...` — pip's PipOption): mutating a
                // class-level constant dict — metadata, no runtime effect
                // in the lowered struct — dropped.
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Subscript(_)) => {}
                // A class-level ATTRIBUTE assignment (`all.__doc__ =
                // ResourceCollection.all.__doc__` — boto3's
                // CollectionManager): docstring/metadata wiring on methods
                // (or a class-level constant store via attribute), with no
                // runtime effect in the lowered plain struct — dropped.
                StatementType::Assign(a)
                    if a.targets.len() == 1
                        && matches!(&a.targets[0], ExprType::Attribute(_)) =>
                {
                    options.definition_warnings.borrow_mut().push(format!(
                        "class `{}`: the class-level attribute assignment \
                         (target `{:?}`) is dropped (method/class metadata has \
                         no runtime effect in the lowered struct)",
                        self.name, a.targets[0]
                    ));
                }                StatementType::AsyncFunctionDef(f) => {
                    return Err(format!(
                        "async method `{}.{}` is not supported yet",
                        self.name, f.name
                    )
                    .into());
                }
                other => {
                    let kind = match other {
                        StatementType::Assign(_) | StatementType::AugAssign(_) => {
                            "a class attribute assignment"
                        }
                        StatementType::ClassDef(_) => "a nested class",
                        StatementType::Import(_) | StatementType::ImportFrom(_) => "an import",
                        _ => "a statement",
                    };
                    return Err(format!(
                        "class `{}` contains {} at class level, which is not supported \
                         yet: only methods, a docstring, and `pass` lower",
                        self.name, kind,
                    )
                    .into());
                }
            }
        }

        // The synthesized constructor occupies `new` in the inherent impl;
        // a user method with that name would be a confusing duplicate-item
        // compile error instead of a conversion-time one — unless it's an
        // unused instance method (urllib3's `Retry.new(self, **kw)`), which
        // is skipped with a warning (the constructor wins).
        if let Some(m) = self.methods().find(|m| m.name == "new") {
            let used = m.args.args.iter().any(|p| p.arg != "self");
            if !used {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: the method `new(self, **kw)` is dropped (the \
                     synthesized constructor for `{}(...)` call sites occupies \
                     the `new` name)",
                    self.name, self.name
                ));
            } else {
                return Err(format!(
                    "class `{}` defines a method named `new`, which collides with the \
                     constructor synthesized for `{}(...)` call sites; rename the method",
                    self.name, self.name
                )
                .into());
            }
        }

        // ---- Field inference ----
        // A derived class's OWN fields are its __init__ stores minus the
        // stores that belong to a base class's fields (those write into the
        // embedded base struct).
        let fields = self.own_fields(&symbols, &options)?;

        // The embedded base struct occupies `__rython_base`; a user field of
        // the same name would collide with it.
        if base.is_some() && fields.iter().any(|(name, _)| name == "__rython_base") {
            return Err(format!(
                "class `{}` assigns an attribute named `__rython_base`, which \
                 collides with the embedded base struct generated for inheritance; \
                 rename the attribute",
                self.name
            )
            .into());
        }

        let mut field_defs: Vec<TokenStream> = fields
            .iter()
            .map(|(name, ti)| {
                let ident = crate::safe_ident(name);
                let ty = ti.to_rust_type();
                quote!(pub #ident: #ty)
            })
            .collect();
        if let Some(b) = &base {
            let b_ident = crate::safe_ident(&b.name);
            field_defs.push(quote!(pub __rython_base: #b_ident));
        }

        // ---- Trait-machinery name guards (hierarchy classes only) ----
        // The generated accessors (`base`, `base_mut`, `<field>_mut`) and the
        // trait name (`{Class}Trait`) must not collide with user code.
        if in_hierarchy {
            // A FIELD named `base`/`base_mut` collides with the embedded-base
            // accessors the trait declares when the class has a base — two
            // identically named trait items (E0428) instead of a clean error.
            if base.is_some()
                && let Some((fname, _)) = fields
                    .iter()
                    .find(|(name, _)| matches!(name.as_str(), "base" | "base_mut"))
            {
                // A field named `base`/`base_mut` on a derived class
                // (`ExtrasCandidate.base` — pip's resolvelib): the field's
                // trait ACCESSORS are skipped (direct field accesses still
                // compile in concrete method bodies; generic-context access
                // to this field is the documented divergence).
                options.definition_warnings.borrow_mut().push(format!(
                    "field `{}` of `{}` collides with the generated base accessor; \
                     its trait accessors are skipped (direct accesses still work)",
                    fname, self.name
                ));
            }
            for m in self.methods() {
                if matches!(m.name.as_str(), "base" | "base_mut") {
                    return Err(format!(
                        "class `{}` defines a method named `{}`, which collides with \
                         the base accessor generated for inheritance; rename the method",
                        self.name, m.name
                    )
                    .into());
                }
                if let Some(field) = m.name.strip_suffix("_mut") {
                    if fields.iter().any(|(fname, _)| fname == field) {
                        return Err(format!(
                            "class `{}` defines a method named `{}`, which collides with \
                             the mutable accessor generated for field `{}`; rename the \
                             method or the field",
                            self.name, m.name, field
                        )
                        .into());
                    }
                }
            }
            let trait_name_str = format!("{}Trait", self.name);
            if symbols.get(&trait_name_str).is_some() {
                return Err(format!(
                    "class `{}` is part of an inheritance hierarchy, so its trait is \
                     named `{}`, which is already taken by another class in this module; \
                     rename one of them",
                    self.name, trait_name_str
                )
                .into());
            }
        }

        // ---- Methods ----
        let method_ctx = CodeGenContext::Class(self.name.clone());
        let methods: Vec<FunctionDef> = self
            .body
            .iter()
            .skip(body_start)
            .filter_map(|s| match &s.statement {
                StatementType::FunctionDef(f) => {
                    // A metadata-struct class's `__new__` (`class
                    // ClientConfigString(str): def __new__(...)` —
                    // botocore) is dropped entirely (the base construction
                    // is unmodeled): exclude it from the method list so the
                    // trait emission does not re-render it.
                    if f.name == "__new__" && self.is_metadata_struct() {
                        None
                    } else {
                        Some(f.clone())
                    }
                }
                _ => None,
            })
            .collect();
        let mut methods_stream = TokenStream::new();
        for m in &methods {
            // A NamedTuple's `__new__` is the constructor: it calls
            // `super().__new__(cls, ...)` and returns the instance, which
            // the synthesized __init__ (from the fields) replaces — the
            // __new__ body's normalization (urllib3's Url) is a documented
            // divergence. Skip it rather than emit a broken `super()` call.
            if m.name == "__new__" && self.is_namedtuple() {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: `__new__` is dropped (NamedTuple construction \
                     lowers through the synthesized __init__)",
                    self.name
                ));
                continue;
            }
            // A metadata-base class's `__new__` (`class ClientConfigString(
            // str): def __new__(cls, value=None): return super().__new__(
            // cls, value)` — botocore): the class lowers as a plain struct,
            // so `__new__` is dropped (the base construction is unmodeled).
            if m.name == "__new__" && self.is_metadata_struct() {
                options.definition_warnings.borrow_mut().push(format!(
                    "class `{}`: `__new__` is dropped (the metadata base's \
                     construction is unmodeled)",
                    self.name
                ));
                continue;
            }
            // A user `new(self, **kw)` method (urllib3's Retry): dropped
            // above (the constructor owns the `new` name) — don't emit it.
            if m.name == "new" && !m.args.args.iter().any(|p| p.arg != "self") {
                continue;
            }
            // A property SETTER in a getter/setter pair emits under the
            // distinct Rust name `{name}_set` (Rust forbids same-name
            // methods): clone the FunctionDef with the renamed method.
            let mut emitted = (*m).clone();
            if self.is_property_setter(&m.name) {
                emitted.name = self.emitted_method_name(m);
            }
            methods_stream.extend(
                emitted.to_rust(method_ctx.clone(), options.clone(), symbols.clone())?,
            );
        }

        // ---- Synthesized constructor ----
        // Python constructs with `ClassName(args)`: default-initialize the
        // struct and run the first __init__ on the MRO. Call sites lower to
        // `ClassName::new(args)?` (see Call::to_rust).
        let mro_init = self.method_on_mro("__init__", &symbols);
        let constructor = match mro_init.as_ref() {
            Some(init) => {
                let mut params = init.args.clone();
                strip_self(&mut params);
                let param_names: Vec<_> = params
                    .posonlyargs
                    .iter()
                    .chain(params.args.iter())
                    .chain(params.kwonlyargs.iter())
                    .map(|p| crate::safe_ident(&p.arg))
                    .collect();
                let rendered = params.to_rust(
                    method_ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                quote! {
                    pub fn new(#rendered) -> Result<Self, PyException> {
                        let mut __rython_self = Self::default();
                        __rython_self.__init__(#(#param_names),*)?;
                        Ok(__rython_self)
                    }
                }
            }
            None => quote! {
                pub fn new() -> Result<Self, PyException> {
                    Ok(Self::default())
                }
            },
        };
        // A derived class without its own __init__ forwards to the first
        // __init__ on the MRO, so `new` runs it on the embedded base part.
        // A base chain with no __init__ anywhere needs no forwarder (`new`
        // is just the default struct).
        let init_forwarder = if self.init_method().is_none() {
            match (&base, mro_init.as_ref()) {
                (Some(_), Some(init)) => {
                    let mut params = init.args.clone();
                    strip_self(&mut params);
                    let param_names: Vec<_> = params
                        .posonlyargs
                        .iter()
                        .chain(params.args.iter())
                        .chain(params.kwonlyargs.iter())
                        .map(|p| crate::safe_ident(&p.arg))
                        .collect();
                    let rendered = params.to_rust(
                        method_ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    quote! {
                        pub(crate) fn __init__(&mut self, #rendered) -> Result<(), PyException> {
                            self.__rython_base.__init__(#(#param_names),*)?;
                            Ok(())
                        }
                    }
                }
                _ => quote!(),
            }
        } else {
            quote!()
        };

        // ---- Trait machinery (inheritance hierarchies only) ----
        let trait_stream = if in_hierarchy {
            self.emit_trait(&base, &fields, &methods, &options, &symbols)?
        } else if options.with_std_python
            && crate::ast::tree::module::class_subclassed_crate_wide(&self.name, &options)
        {
            // A plain-struct class subclassed only CROSS-MODULE
            // (urllib3's RequestMethods): the subclass modules' ancestor
            // impls and supertrait bounds name `{Name}Trait`, so the
            // trait must exist — but ACCESSOR-ONLY. Flipping the full
            // machinery on (methods as trait defaults) re-routes the
            // subclasses' inherited-call inference, measured at ~2,700
            // errors in issue #137 round 18; the class's methods stay
            // inherent and dispatch exactly as before.
            self.emit_accessor_trait(&fields)
        } else {
            quote!()
        };

        let docs = match self.get_docstring() {
            Some(docstring) => {
                let doc_lines: Vec<_> = docstring
                    .lines()
                    .map(|line| {
                        let doc_line = line.to_string();
                        quote! { #[doc = #doc_line] }
                    })
                    .collect();
                quote!(#(#doc_lines)*)
            }
            None => quote!(),
        };

        // ---- The type-level inheritance tree ----
        // One `impl PyInherits<Ancestor> for Class` per entry in the base
        // chain (reflexive included): generic Rust code can bound on
        // Python ancestry (`fn pet<T: PyInherits<Animal>>(x: T)`), and the
        // entries are derived from the SAME base_chain the conversion-time
        // `class_extends` lookup walks, so the two trees cannot drift.
        let inherits_tree = if options.with_std_python {
            let entries = self.base_chain(&symbols).into_iter().map(|a| {
                let ancestor = crate::safe_ident(&a.name);
                quote!(impl PyInherits<#ancestor> for #class_name {})
            });
            quote!(#(#entries)*)
        } else {
            quote!()
        };

        // A class with `__len__` participates in the len() protocol:
        // `len(x)` lowers to `stdpython::len(&x)` bound on `Len`.
        // stdpython's len() is infallible like Rust's, so a raising
        // `__len__` becomes a loud abort (the §12.2 raise-in-infallible
        // divergence) — never a silent 0.
        let len_impl = if options.with_std_python
            && self.methods().any(|m| m.name == "__len__")
        {
            quote! {
                impl stdpython::Len for #class_name {
                    fn len(&self) -> usize {
                        match self.__len__() {
                            Ok(n) => n as usize,
                            Err(e) => panic!("{}", e),
                        }
                    }
                }
            }
        } else {
            quote!()
        };

        // An instance is never None (only None is None in Python), so `x
        // is None` on a class-typed value lowers through PyIsNone to a
        // constant false — the same never-None contract the scalar types
        // carry (stdpython's never_none! macro).
        let is_none_impl = if options.with_std_python {
            quote!(impl stdpython::PyIsNone for #class_name {
                fn py_is_none(&self) -> bool {
                    false
                }
            })
        } else {
            quote!()
        };

        // The INHERENT base accessors: the derived struct's `base()` /
        // `base_mut()` reach its own embedded `__rython_base`. Inherent
        // methods take precedence over the trait ones, so a concrete
        // receiver's `self.base()` cannot be ambiguous — the derived class
        // implements BOTH its own trait and every ancestor trait, and each
        // declares a `base` accessor with a different return type (E0034 —
        // urllib3's HTTPSConnectionPool). Generic trait-default bodies keep
        // resolving through their own trait's declaration.
        let base_inherent_accessors = if let Some(b) = base {
            let b_ident = crate::safe_ident(&b.name);
            quote! {
                pub(crate) fn base(&self) -> & #b_ident {
                    &self.__rython_base
                }
                pub(crate) fn base_mut(&mut self) -> &mut #b_ident {
                    &mut self.__rython_base
                }
            }
        } else {
            quote!()
        };

        // str(x)/print(x)/f-string `{x}` on a class INSTANCE route
        // through py_display (Python's str, not Rust's Display — round
        // 34's display cluster). CPython's str() calls __str__ (falling
        // back to __repr__ when only that is defined), then the default
        // object repr `<module.ClassName object at 0x...>` — the address
        // is nondeterministic (CPython's own output varies run to run),
        // so the repr drops it (§12.3 cosmetic divergence). A
        // __str__/__repr__ that raises becomes a loud abort (the §12.2
        // raise-in-infallible divergence — the display surface is
        // infallible, like len()).
        let display_impl = if options.with_std_python && !is_exception_class(&self) {
            let defines_dunder = |name: &str| -> bool {
                self.methods().any(|m| m.name == name)
                    || self
                        .base_chain(&symbols)
                        .iter()
                        .any(|a| a.methods().any(|m| m.name == name))
            };
            let display_expr = if defines_dunder("__str__") {
                quote!(self.__str__().unwrap_or_else(|e| panic!("{}", e)))
            } else if defines_dunder("__repr__") {
                quote!(self.__repr__().unwrap_or_else(|e| panic!("{}", e)))
            } else {
                let module = options.module_path.join(".");
                let class_display = if module.is_empty() {
                    self.name.clone()
                } else {
                    format!("{module}.{}", self.name)
                };
                quote!(format!("<{} object>", #class_display))
            };
            // repr(obj) is `__repr__` alone (str falls back to it, not the
            // other way round); a class without one reprs as the default
            // object form. Every class carries PyRepr so a container of
            // instances prints (`print(sorted(shapes, key=...))`) and the
            // hierarchy sum type can delegate.
            let repr_expr = if defines_dunder("__repr__") {
                quote!(self.__repr__().unwrap_or_else(|e| panic!("{}", e)))
            } else {
                let module = options.module_path.join(".");
                let class_display = if module.is_empty() {
                    self.name.clone()
                } else {
                    format!("{module}.{}", self.name)
                };
                quote!(format!("<{} object>", #class_display))
            };
            quote! {
                impl stdpython::PyDisplay for #class_name {
                    fn py_display(&self) -> String {
                        #display_expr
                    }
                }
                impl stdpython::PyRepr for #class_name {
                    fn py_repr(&self) -> String {
                        #repr_expr
                    }
                }
            }
        } else {
            quote!()
        };

        // A polymorphic ROOT's sum type (hierarchy.rs), after the class.
        let any_enum = self.emit_any_enum(in_hierarchy, &symbols, &options)?;

        Ok(quote! {
            #docs
            #[derive(Clone, Default)]
            pub struct #class_name {
                #(#field_defs),*
            }
            // A class instance is never None (CPython: only None is None),
            // so `x is None` on an instance lowers through PyIsNone to
            // false — same contract the PyInherits tree carries for
            // ancestry bounds.
            #is_none_impl
            // Class-level COMPUTED constants live at module scope under
            // class-mangled names: associated statics are not legal Rust
            // (issue #137).
            #class_lazylock_constants
            #inherits_tree
            #len_impl
            #display_impl
            #trait_stream
            impl #class_name {
                #class_constants
                #class_const_accessors
                #constructor
                #init_forwarder
                #base_inherent_accessors
                #methods_stream
            }
            #any_enum
        })
    }
}

impl ClassDef {
    /// Emit the trait-based inheritance machinery for a class that has a
    /// base or is used as a base:
    ///
    /// - `trait {Name}Trait` — `base()`/`base_mut()` accessors (when the
    ///   class has a base), field accessors `fn f(&self) -> T` /
    ///   `fn f_mut(&mut self) -> &mut T`, and the class's own (non-override)
    ///   methods as default bodies written against the accessors.
    /// - `impl {Name}Trait for {Name}` — the accessor bodies.
    /// - For every ancestor, `impl {Ancestor}Trait for {Name}` — the
    ///   ancestor's field accessors (walking the embedded `__rython_base`
    ///   chain) and the class's overrides of the ancestor's methods, written
    ///   against the concrete struct.
    /// The accessor-only trait for a class subclassed only cross-module
    /// (see the call site): field accessors, no method defaults, no base
    /// accessors (a class with a base is `in_hierarchy` and takes the
    /// full machinery instead). The declarations mirror the subclass-side
    /// ancestor impls, which iterate the same `own_fields` in this
    /// module's scope — the two sides cannot drift.
    fn emit_accessor_trait(&self, fields: &[(String, crate::TypeInfo)]) -> TokenStream {
        let class_name = crate::safe_ident(&self.name);
        let trait_name = format_ident!("{}Trait", self.name);
        let mut decls = TokenStream::new();
        let mut impls = TokenStream::new();
        for (fname, fti) in fields {
            let f = crate::safe_ident(fname);
            let f_mut = format_ident!("{}_mut", fname);
            let fty = fti.to_rust_type();
            decls.extend(quote! {
                fn #f(&self) -> #fty;
                fn #f_mut(&mut self) -> &mut #fty;
            });
            impls.extend(quote! {
                fn #f(&self) -> #fty {
                    self.#f.clone()
                }
                fn #f_mut(&mut self) -> &mut #fty {
                    &mut self.#f
                }
            });
        }
        quote! {
            pub trait #trait_name {
                #decls
            }
            impl #trait_name for #class_name {
                #impls
            }
        }
    }

    fn emit_trait(
        &self,
        base: &Option<ClassDef>,
        fields: &[(String, crate::TypeInfo)],
        methods: &[FunctionDef],
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let class_name = crate::safe_ident(&self.name);
        let trait_name = format_ident!("{}Trait", self.name);

        // The per-class trait is PUBLIC (inherited methods are called
        // cross-module: the struct is `pub` and re-exported, so the traits
        // carrying its methods must be nameable wherever the struct is) and
        // declares the direct base's trait as a supertrait. Computed here,
        // BEFORE the accessor declarations: when the supertrait is present
        // it already carries `base`/`base_mut`, so re-declaring them in the
        // derived trait would make every `self.base()` on a receiver
        // implementing both traits ambiguous (E0034 — urllib3's
        // HTTPSConnectionPool implements HTTPConnectionPoolTrait AND
        // HTTPSConnectionPoolTrait; both used to declare `base`).
        let chain = self.cross_module_chain(symbols, options);
        let supertrait = base
            .as_ref()
            .filter(|b| chain.get(1).is_some_and(|(c, _, _, _)| c.name == b.name))
            .map(|b| {
                let b_trait = format_ident!("{}Trait", b.name);
                // A CROSS-MODULE base's trait is named by its crate path
                // (this module imports the STRUCT, not its trait —
                // `PoolManagerTrait: crate::_request_methods::
                // RequestMethodsTrait`, urllib3).
                match chain.get(1).and_then(|(_, _, _, p)| p.as_ref()) {
                    Some(path) => {
                        let segs: Vec<_> =
                            path.iter().map(|p| crate::safe_ident(p)).collect();
                        quote!(: crate #(::#segs)* :: #b_trait)
                    }
                    None => quote!(: #b_trait),
                }
            });

        // ---- Own trait ----
        let mut own_accessor_decls = TokenStream::new();
        if let Some(b) = base {
            let b_ident = crate::safe_ident(&b.name);
            own_accessor_decls.extend(quote! {
                fn base(&self) -> & #b_ident;
                fn base_mut(&mut self) -> &mut #b_ident;
            });
        }
        for (fname, fty) in fields {
            // A field colliding with the base accessors (`base`/`base_mut`
            // on a derived class) has no trait accessors.
            if base.is_some() && matches!(fname.as_str(), "base" | "base_mut") {
                continue;
            }
            let f = crate::safe_ident(fname);
            let f_mut = format_ident!("{}_mut", fname);
            let fty = fty.to_rust_type();
            own_accessor_decls.extend(quote! {
                fn #f(&self) -> #fty;
                fn #f_mut(&mut self) -> &mut #fty;
            });
        }
        // A method the class defines that no ancestor defines is a NEW
        // method: it lives as a default in this class's own trait.
        let own_methods: Vec<&FunctionDef> = methods
            .iter()
            .filter(|m| {
                m.name != "__init__"
                    && base
                        .as_ref()
                        .map_or(true, |b| b.method_on_mro(&m.name, symbols).is_none())
            })
            .collect();
        let mut own_method_defaults = TokenStream::new();
        for m in &own_methods {
            let trait_ctx = CodeGenContext::Trait {
                class: self.name.clone(),
                generic: true,
                super_target: None,
                force_mut_self: options
                    .trait_mut_self
                    .get(&self.name)
                    .is_some_and(|s| s.contains(&m.name)),
            };
            // A property SETTER emits under `{name}_set` (same rename as
            // the impl loop above).
            let mut emitted = (*m).clone();
            if self.is_property_setter(&m.name) {
                emitted.name = self.emitted_method_name(m);
            }
            own_method_defaults.extend(
                emitted.to_rust(trait_ctx, options.clone(), symbols.clone())?,
            );
        }

        // ---- Super trampolines ----
        // `super().m(...)` must run the ancestor's ORIGINAL body with the
        // DERIVED self, so nested `self.x()` calls inside it keep dispatching
        // through the most-derived class — CPython resolves them on the
        // original object's MRO. Calling the body on the embedded base
        // (`self.__rython_base.m(...)`) pins the receiver to the ancestor and
        // silently resolves those nested calls to the ancestor's versions
        // (an overridden `self.speak()` called from `Animal.describe` would
        // emit D:... instead of D:woof).
        //
        // Each method the class defines gets a uniquely-named DEFAULT in its
        // own trait whose body is the GENERIC rendering of the original body
        // (identical to the trait-default rendering). The unique name means
        // no override can intercept it, so
        // `<Self as {Class}Trait>::__rython_super_{m}(self)` always runs the
        // class's original body with `Self` = the most-derived type — and
        // nested `self.x()` resolves through the trait bound to the derived
        // class's override, exactly like Python. This covers BOTH lowering
        // paths (inherent methods and re-emitted overrides), since the call
        // site always passes the plain derived `self`.
        let mut super_trampolines = TokenStream::new();
        for m in methods.iter().filter(|m| m.name != "__init__") {
            if m.name.starts_with("__rython_super_") {
                return Err(format!(
                    "class `{}` defines a method named `{}`, which collides with the \
                     generated super() trampoline namespace; rename the method",
                    self.name, m.name
                )
                .into());
            }
            // The trait signature widens to `&mut self` when ANY definition
            // in the hierarchy mutates; the trampoline must match the
            // (widened) default's receiver or call sites borrowing mutably
            // would not type-check. Keyed by the ROOT (topmost definer),
            // like the default-body widening above.
            let root = self
                .base_chain(symbols)
                .into_iter()
                .rev()
                .find(|c| c.methods().any(|mm| mm.name == m.name));
            let force_mut_self = root
                .as_ref()
                .and_then(|r| options.trait_mut_self.get(&r.name))
                .is_some_and(|s| s.contains(&m.name));
            let mut helper = (*m).clone();
            let emitted_name = self.emitted_method_name(m);
            helper.name = format!("__rython_super_{}", emitted_name);
            let helper_ctx = CodeGenContext::Trait {
                class: self.name.clone(),
                generic: true,
                super_target: None,
                force_mut_self,
            };
            super_trampolines.extend(
                helper.to_rust(helper_ctx, options.clone(), symbols.clone())?,
            );
        }

        let mut own_impl_body = TokenStream::new();
        if let Some(b) = base {
            let b_ident = crate::safe_ident(&b.name);
            own_impl_body.extend(quote! {
                fn base(&self) -> & #b_ident {
                    &self.__rython_base
                }
                fn base_mut(&mut self) -> &mut #b_ident {
                    &mut self.__rython_base
                }
            });
        }
        for (fname, fty) in fields {
            if base.is_some() && matches!(fname.as_str(), "base" | "base_mut") {
                continue;
            }
            let f = crate::safe_ident(fname);
            let f_mut = format_ident!("{}_mut", fname);
            let fty = fty.to_rust_type();
            own_impl_body.extend(quote! {
                fn #f(&self) -> #fty {
                    self.#f.clone()
                }
                fn #f_mut(&mut self) -> &mut #fty {
                    &mut self.#f
                }
            });
        }

        // Trait default bodies are generic over `Self: {Name}Trait` only,
        // so a new method that calls an inherited method (`def bar(self):
        // self.foo()` where foo lives on the base) resolves `foo` through
        // the supertrait bound; ancestor methods are not on the concrete
        // `Self` otherwise. A default body that formats `self` in an
        // exception message (`raise ClosedPoolError(self)` — urllib3)
        // lowers through py_display, which needs `Self: PyDisplay`; the
        // concrete class always carries the generated impl (round 34),
        // so the bound is satisfiable by every implementor (round 41).
        let display_bound = if options.with_std_python {
            quote!(where Self: stdpython::PyDisplay)
        } else {
            quote!()
        };
        let own_trait = quote! {
            pub trait #trait_name #supertrait #display_bound {
                #own_accessor_decls
                #own_method_defaults
                #super_trampolines
            }
            impl #trait_name for #class_name {
                #own_impl_body
            }
        };

        // ---- Ancestor impls ----
        // For each ancestor (root first), implement ITS trait: field
        // accessors reaching through `self.__rython_base` (repeated per
        // level), plus this class's overrides of the methods that ancestor
        // defined, written against the concrete struct.
        //
        // The chain follows IMPORTED bases (`SOCKSConnection(
        // HTTPConnection)` with the base in ..connection — urllib3's
        // contrib modules), carrying each ancestor's DEFINING-module
        // symbol table: its field-accessor types and method sets must
        // resolve in the module that declared its trait, or the impl's
        // signatures disagree with the trait's (E0053) and its local
        // type names don't resolve here (E0425).
        let mut ancestor_impls = TokenStream::new();
        for (depth, (ancestor, a_syms_owned, a_opts_owned, a_path)) in
            chain.iter().enumerate().skip(1).rev()
        {
            let a_syms = a_syms_owned;
            let a_opts = a_opts_owned;
            let ancestor_trait = format_ident!("{}Trait", ancestor.name);
            let mut chain_tokens = TokenStream::new();
            for _ in 0..depth {
                chain_tokens.extend(quote!(.__rython_base));
            }
            // The ancestor's OWN fields (its stores minus its own base's
            // stores — see own_fields): an ancestor that re-assigns a field
            // its base owns declares no accessor for it, because the field
            // physically lives in the ancestor's base struct and the
            // ancestor's trait only declares accessors for fields it owns.
            let a_fields = ancestor.own_fields(a_syms, a_opts)?;
            let mut accessor_impls = TokenStream::new();
            // The ancestor's own base accessors, if it has a base (the
            // NEXT chain entry — the chain already resolved imported
            // bases): from the derived struct, its base struct is one
            // level deeper.
            if let Some((a_base, _, _, _)) = chain.get(depth + 1) {
                let ab_ident = crate::safe_ident(&a_base.name);
                let mut base_self = quote!(self);
                base_self.extend(chain_tokens.clone());
                accessor_impls.extend(quote! {
                    fn base(&self) -> & #ab_ident {
                        &#base_self.__rython_base
                    }
                    fn base_mut(&mut self) -> &mut #ab_ident {
                        &mut #base_self.__rython_base
                    }
                });
            }
            for (fname, fti) in &a_fields {
                let f = crate::safe_ident(fname);
                let f_mut = format_ident!("{}_mut", fname);
                let fty = fti.to_rust_type();
                // `self.__rython_base[.__rython_base]*` reaches the
                // ancestor's struct from the derived struct.
                let mut accessor_self = quote!(self);
                accessor_self.extend(chain_tokens.clone());
                accessor_impls.extend(quote! {
                    fn #f(&self) -> #fty {
                        #accessor_self.#f.clone()
                    }
                    fn #f_mut(&mut self) -> &mut #fty {
                        &mut #accessor_self.#f
                    }
                });
            }
            // Overrides: for each TRAIT MEMBER of the ancestor (its own
            // methods that are not themselves overrides of ITS base — those
            // live in a higher trait), the NEAREST definition in the derived
            // chain (self … down to the ancestor, exclusive) wins. A
            // definition by the ancestor itself is the trait default, so
            // only strictly-lower definers need re-emission against this
            // class's struct — and `super()` inside such a re-emitted
            // override must target the DEFINER's base, not this class's base.
            let a_base = chain.get(depth + 1);
            let ancestor_members: Vec<&FunctionDef> = ancestor
                .methods()
                .filter(|am| {
                    am.name != "__init__"
                        && a_base.map_or(true, |(b, b_syms, _, _)| {
                            b.method_on_mro(&am.name, b_syms).is_none()
                        })
                })
                .collect();
            let mut override_stream = TokenStream::new();
            for am in &ancestor_members {
                let mut definer: Option<FunctionDef> = None;
                let mut definer_name: Option<String> = None;
                let mut definer_syms: Option<&SymbolTableScopes> = None;
                let mut definer_opts: Option<&crate::PythonOptions> = None;
                for (c, c_syms, c_opts, _) in chain.iter() {
                    if c.name == ancestor.name {
                        break;
                    }
                    if let Some(m) = c.methods().find(|m| m.name == am.name) {
                        definer = Some(m.clone());
                        definer_name = Some(c.name.clone());
                        // An override defined by an INTERMEDIATE imported
                        // ancestor renders in ITS module's scope.
                        definer_syms = Some(c_syms);
                        definer_opts = Some(c_opts);
                        break;
                    }
                }
                if let (Some(m), Some(dname), Some(dsyms), Some(dopts)) =
                    (definer, definer_name, definer_syms, definer_opts)
                {
                    let trait_ctx = CodeGenContext::Trait {
                        class: self.name.clone(),
                        generic: false,
                        super_target: Some(dname),
                        force_mut_self: options
                            .trait_mut_self
                            .get(&ancestor.name)
                            .is_some_and(|s| s.contains(&am.name)),
                    };
                    // A property SETTER in a pair emits under `{name}_set`
                    // — the DEFINING class's pair (the name in the
                    // hierarchy is the pair's Python name).
                    let mut emitted = m.clone();
                    if self.is_property_setter(&am.name) {
                        emitted.name = self.emitted_method_name(&am);
                    }
                    let rendered =
                        emitted.to_rust(trait_ctx, dopts.clone(), dsyms.clone())?;
                    // An override must agree with the trait's declared
                    // signature: Python's covariant overrides
                    // (`_new_conn() -> socks.socksocket` over the base's
                    // `-> socket.socket` — urllib3's SOCKS classes; the
                    // same-module `decompress`/property-setter shapes) have
                    // no Rust trait lowering. A disagreeing override is
                    // DROPPED — the base implementation is what runs (the
                    // documented divergence) — never a mismatched impl
                    // (E0053/E0050/E0049).
                    {
                        // Rendered as the ANCESTOR's own method in its own
                        // scope — only the signature is compared.
                        let ref_ctx = CodeGenContext::Trait {
                            class: ancestor.name.clone(),
                            generic: false,
                            super_target: None,
                            force_mut_self: options
                                .trait_mut_self
                                .get(&ancestor.name)
                                .is_some_and(|s| s.contains(&am.name)),
                        };
                        let reference = (*am)
                            .clone()
                            .to_rust(ref_ctx, a_opts.clone(), a_syms.clone())?;
                        if fn_signature_key(&rendered) != fn_signature_key(&reference) {
                            options.definition_warnings.borrow_mut().push(format!(
                                "`{}.{}` overrides `{}.{}` with a different \
                                 signature; the override is dropped and the base \
                                 implementation runs (covariant-override \
                                 divergence)",
                                self.name, am.name, ancestor.name, am.name
                            ));
                            continue;
                        }
                    }
                    override_stream.extend(rendered);
                }
            }
            // A CROSS-MODULE ancestor's accessor types and re-emitted
            // bodies name classes of its defining module bare: qualify
            // them to their crate paths (this module need not import
            // them).
            let (accessor_impls, override_stream) = match a_path {
                Some(path) => (
                    qualify_cross_module_types(
                        accessor_impls,
                        path,
                        a_syms,
                        a_opts,
                        options,
                    ),
                    qualify_cross_module_types(
                        override_stream,
                        path,
                        a_syms,
                        a_opts,
                        options,
                    ),
                ),
                None => (accessor_impls, override_stream),
            };
            // A CROSS-MODULE ancestor's trait is named by its crate path
            // (this module need not import it).
            let trait_path = match a_path {
                Some(path) => {
                    let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
                    quote!(crate #(::#segs)* :: #ancestor_trait)
                }
                None => quote!(#ancestor_trait),
            };
            ancestor_impls.extend(quote! {
                impl #trait_path for #class_name {
                    #accessor_impls
                    #override_stream
                }
            });
        }

        Ok(quote!(#own_trait #ancestor_impls))
    }
}

/// Remove the RECEIVER — the first positional parameter of an instance
/// method — from a method's parameter list. Python binds the instance to
/// the FIRST parameter whatever its name (boto3 names it `factory_self`;
/// most code names it `self`), so any leading positional parameter is the
/// receiver, not a call-site argument.
pub(crate) fn strip_self(args: &mut crate::ParameterList) {
    if !args.posonlyargs.is_empty() {
        args.posonlyargs.remove(0);
    } else if !args.args.is_empty() {
        args.args.remove(0);
    }
}

/// Whether an expression is a CLASS REFERENCE (`HTTPConnectionPool` — a
/// Name resolving to a class, or an imported class name): class values have
/// no rython runtime equivalent (the classes-as-values divergence).
pub(crate) fn is_class_value_expr(value: &ExprType, symbols: &SymbolTableScopes) -> bool {    match value {
        ExprType::Name(n) => match symbols.get(&n.id) {
            Some(SymbolTableNode::ClassDef(_)) => true,
            Some(SymbolTableNode::Alias(canonical)) => {
                matches!(symbols.get(canonical), Some(SymbolTableNode::ClassDef(_)))
            }
            Some(SymbolTableNode::ImportFrom(_)) => true,
            _ => false,
        },
        _ => false,
    }
}

/// Infer the struct field type for a value stored into `self.attr`.
/// Returns a TypeInfo — the single type authority (issue #137's review of
/// rounds 38–47): field types are structural, so the coercion layers can
/// match on them instead of re-parsing rendered tokens.
fn infer_field_type(
    value: &ExprType,
    name_types: &std::collections::HashMap<String, crate::TypeInfo>,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    class_name: &str,
) -> Option<crate::TypeInfo> {
    match value {
        ExprType::Name(n) => name_types.get(&n.id).cloned().or_else(|| {
            // A module-level constant (`self.flags = _LATIN`, where
            // `_LATIN: int = 1` may live in another module): follow
            // Assign/ImportFrom chains to the constant's value.
            let const_type = |value: &ExprType| {
                crate::simple_expr_typeinfo(value).map(|t| {
                    // String literals are owned in FIELDS (the store side
                    // converts; the literal itself is a &'static str).
                    if matches!(t, crate::TypeInfo::StrRef) {
                        crate::TypeInfo::String
                    } else {
                        t
                    }
                })
            };
            match symbols.get(&n.id) {
                Some(SymbolTableNode::Assign { value, .. }) => const_type(value).or_else(|| {
                    // A dict of CLASSES (`pool_classes_by_scheme = {"http":
                    // HTTPConnectionPool, ...}` — urllib3's PoolManager):
                    // class values are their NAME STRINGS (round 33), so
                    // the dict is PyDict<String, String>. A dict holding
                    // CALLABLES (`key_fn_by_scheme = {"http":
                    // functools.partial(...)}` — callables cannot be
                    // runtime values) is the boxed PyDict (documented
                    // divergence).
                    if let ExprType::Dict(d) = value {
                        let all_classes = d.values.iter().all(|v| crate::is_class_value_expr(v, symbols));
                        let has_callable = d
                            .values
                            .iter()
                            .any(|v| matches!(v, ExprType::Call(_) | ExprType::Lambda(_)));
                        if all_classes {
                            Some(crate::TypeInfo::Dict(
                                Box::new(crate::TypeInfo::String),
                                Box::new(crate::TypeInfo::String),
                            ))
                        } else if has_callable {
                            Some(crate::TypeInfo::Dict(
                                Box::new(crate::TypeInfo::String),
                                Box::new(crate::TypeInfo::PyValue),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    // A sentinel constant (`NOT_SET = object()` — botocore's
                    // model): a unique object — a boxed PyValue.
                    match value {
                        ExprType::Call(c)
                            if matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "object") =>
                        {
                            Some(crate::TypeInfo::PyValue)
                        }
                        _ => None,
                    }
                })
                .or_else(|| {
                    // A local assigned a conditional
                    // (`hashes_from_link = {} if link_hash is None else
                    // link_hash.as_dict()` — pip's Link): infer through
                    // the IfExp's branches.
                    match value {
                        ExprType::IfExp(e) => {
                            infer_field_type(&e.body, name_types, symbols, options, class_name)
                                .or_else(|| {
                                    infer_field_type(
                                        &e.orelse,
                                        name_types,
                                        symbols,
                                        options,
                                        class_name,
                                    )
                                })
                        }
                        // A local assigned an UNRESOLVABLE CALL
                        // (`handle = GetStdHandle(STDOUT)` — rich's
                        // LegacyWindowsTerm, where GetStdHandle is a
                        // Windows-API wrapper): a foreign object — a boxed
                        // PyValue (the external-object divergence).
                        ExprType::Call(_) => Some(crate::TypeInfo::PyValue),
                        _ => None,
                    }
                }),
                // A CALLABLE value (`self.header_formatter =
                // format_multipart_header_param` — urllib3's RequestField):
                // a function reference held as data has no rython value
                // equivalent — a boxed PyValue (documented divergence).
                Some(SymbolTableNode::FunctionDef(_)) => {
                    Some(crate::TypeInfo::PyValue)
                }
                // A MODULE held as a value (`import keyring` inside
                // __init__, then `self.keyring = keyring` — pip's
                // KeyRingPythonProvider): an external module object — a
                // boxed PyValue (external-object divergence).
                Some(SymbolTableNode::Import(_)) => {
                    Some(crate::TypeInfo::PyValue)
                }
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let path = i.resolved_module_path(options);
                    let module = options.module_defs.get(&path)?;
                    let module: &crate::Module = module;
                    let syms = module.clone().find_symbols(SymbolTableScopes::new());
                    match syms.get(&n.id) {
                        Some(SymbolTableNode::Assign { value, .. }) => const_type(value).or_else(
                            || {
                                // An imported DICT constant
                                // (`self._event_aliases = EVENT_ALIASES`
                                // — botocore's hooks): a boxed
                                // PyDict<String, PyValue>.
                                match value {
                                    ExprType::Dict(_) => Some(crate::TypeInfo::Dict(
                                        Box::new(crate::TypeInfo::String),
                                        Box::new(crate::TypeInfo::PyValue),
                                    )),
                                    _ => None,
                                }
                            },
                        ),
                        _ => None,
                    }
                }
                // An UNRESOLVABLE local (`self.transport = t` where
                // `t = tcls(...)` and `tcls` is itself a local bound to a
                // class — distlib's ServerProxy): a boxed PyValue (the
                // unknown-local / class-as-value divergence).
                _ => Some(crate::TypeInfo::PyValue),
            }
        }),
        // A constructed instance of a known class types the field as that
        // class's struct.
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) if n.id == "bool" => Some(crate::TypeInfo::Bool),
            // A `cast(T, ...)` typing no-op (`self.frames = cast(List[str],
            // spinner["frames"])[:]` — rich's Spinner): the cast's FIRST
            // argument is the annotation — `List[str]` → `Vec<String>`.
            ExprType::Name(n) if n.id == "cast" => {
                call.args.first().and_then(|ann| match ann {
                    ExprType::Name(sn) => match sn.id.as_str() {
                        "float" => Some(crate::TypeInfo::Float),
                        "int" => Some(crate::TypeInfo::Int),
                        "str" => Some(crate::TypeInfo::String),
                        "bool" => Some(crate::TypeInfo::Bool),
                        _ => None,
                    },
                    ExprType::Subscript(sub)
                        if matches!(sub.value.as_ref(), ExprType::Name(sn)
                            if matches!(sn.id.as_str(), "List" | "list")) =>
                    {
                        match &sub.kind {
                            crate::SubscriptKind::Index(elt) => {
                                let t = crate::annotation_type_info(elt)?;
                                if matches!(t, crate::TypeInfo::String) {
                                    Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::String)))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                })
            }
            // A builtin scalar conversion call (`int(x)`, `float(x)`,
            // `str(x)`) types the field as that scalar (urllib3's
            // emscripten `self.timeout = int(1000 * timeout)`).
            ExprType::Name(n)
                if matches!(n.id.as_str(), "int" | "float" | "str") =>
            {
                Some(match n.id.as_str() {
                    "int" => crate::TypeInfo::Int,
                    "float" => crate::TypeInfo::Float,
                    _ => crate::TypeInfo::String,
                })
            }
            // `set(...)` / `frozenset(...)` of a generator of str
            // (`frozenset(h.lower() for h in ...)` — urllib3's Retry):
            // a String set. An EMPTY or unknown-element set (`set()` —
            // s3transfer's TransferCoordinator._associated_futures, filled
            // with duck-typed future objects) is bookkeeping with no
            // statically-known element: a boxed PyValue (documented
            // divergence — set bookkeeping fields are unmodeled).
            ExprType::Name(n)
                if matches!(n.id.as_str(), "set" | "frozenset") =>
            {
                let has_str_generator = call.args.iter().any(|a| {
                    matches!(a, ExprType::GeneratorExp(g)
                        if matches!(g.elt.as_ref(), ExprType::Call(c)
                            if matches!(c.func.as_ref(), ExprType::Attribute(at) if at.attr == "lower")))
                });
                if has_str_generator {
                    Some(crate::TypeInfo::HashSet(Box::new(crate::TypeInfo::String)))
                } else {
                    Some(crate::TypeInfo::PyValue)
                }
            }
            // `OrderedDict()` / `defaultdict()` / `dict()` — a map field
            // (urllib3's RecentlyUsedContainer._container): the boxed
            // PyDict, matching `dict[str, Any]` lowering.
            ExprType::Name(n)
                if matches!(n.id.as_str(), "dict" | "OrderedDict" | "defaultdict") =>
            {
                Some(crate::TypeInfo::Dict(
                    Box::new(crate::TypeInfo::String),
                    Box::new(crate::TypeInfo::PyValue),
                ))
            }
            // A threading lock (`RLock()`, `Lock()`, `threading.RLock()`) —
            // a stdlib object with no rython equivalent; `with self.lock:`
            // only evaluates the receiver (the __enter__/__exit__ protocol
            // is unmodeled), so the field is unit.
            ExprType::Name(n)
                if matches!(n.id.as_str(), "RLock" | "Lock" | "Semaphore") =>
            {
                Some(crate::TypeInfo::Tuple(vec![]))
            }
            ExprType::Attribute(a)
                if matches!(a.attr.as_str(), "RLock" | "Lock" | "Semaphore")
                    && matches!(a.value.as_ref(), ExprType::Name(n)
                        if crate::StdModule::from_name(&n.id)
                            == Some(crate::StdModule::Threading)) =>
            {
                Some(crate::TypeInfo::Tuple(vec![]))
            }
            // `datetime.timedelta(...)` — the stdpython timedelta struct.
            ExprType::Attribute(a)
                if crate::DatetimeType::from_name(&a.attr)
                    == Some(crate::DatetimeType::Timedelta)
                    && matches!(a.value.as_ref(), ExprType::Name(n)
                        if crate::StdModule::from_name(&n.id)
                            == Some(crate::StdModule::Datetime)) =>
            {
                Some(crate::TypeInfo::Custom(quote!(datetime::timedelta)))
            }
            // `self.sock = self._new_conn()` — a call to THIS class's own
            // method (urllib3's connect()): the method's return
            // annotation types the field, the same contract issue #123
            // gave imported functions. Without it a method-assigned field
            // falls back to the boxed PyValue and every typed use of it
            // mismatches.
            ExprType::Attribute(a)
                if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") =>
            {
                let Some(SymbolTableNode::ClassDef(cls)) = symbols.get(class_name) else {
                    return None;
                };
                let method = cls.methods().find(|m| m.name == a.attr)?;
                let ann = method.returns.as_deref()?;
                if crate::is_none_expr(ann) {
                    return None;
                }
                crate::resolve_alias_typeinfo(ann, symbols, options)
            }
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::ClassDef(_)) => {
                    Some(crate::TypeInfo::Class(n.id.clone()))
                }
                // An imported class (`from urllib3.util.retry import
                // Retry` → `self.max_retries = Retry(0, ...)`) resolves
                // through the defining module, following RE-EXPORT chains
                // (`from .connection import ProxyConfig` where connection.py
                // re-exports it from ._base_connection — urllib3).
                Some(SymbolTableNode::ImportFrom(i)) => {
                    let path = i.resolved_module_path(options);
                    if crate::module_class_def(options, &path, &n.id).is_some()
                        || crate::resolve_imported_class(options, &path, &n.id, 0).is_some()
                    {
                        Some(crate::TypeInfo::Class(n.id.clone()))
                    } else {
                        // A call to an IMPORTED function whose return type
                        // cannot be resolved — either no return annotation,
                        // or a re-export chain that `module_function_def`
                        // cannot follow (`from botocore import xform_name` —
                        // boto3's ResourceCollection): the field is a boxed
                        // PyValue (cross-module return-typing divergence,
                        // #123).
                        crate::call_return_typeinfo(call, Some(symbols), Some(options))
                            .or_else(|| Some(crate::TypeInfo::PyValue))
                    }
                }
                // A module-level function: its return annotation types the
                // field (`self.punct = is_punctuation(character)`). An
                // unannotated same-module function also boxes the field
                // (PyValue) rather than failing the module.
                _ => crate::call_return_typeinfo(call, Some(symbols), Some(options))
                    .or_else(|| Some(crate::TypeInfo::PyValue)),
            },
            // A boolean predicate call (`character.isprintable()`,
            // `s.isascii()`) types the field bool.
            ExprType::Attribute(a)
                if a.attr.starts_with("is") && a.attr.len() > 2 => Some(crate::TypeInfo::Bool),
            // A `.copy()` of a module-level dict (`self.key_fn_by_scheme =
            // key_fn_by_scheme.copy()` — urllib3's PoolManager): the field
            // takes the copied dict's type (a boxed PyDict for the
            // class/callable-valued config dicts). NOT a MODULE call
            // (`copy.copy(x)` — the `copy` module import).
            ExprType::Attribute(a)
                if a.attr == "copy"
                    && crate::root_name(&a.value).is_none_or(|r| {
                        !matches!(
                            symbols.get(r),
                            Some(
                                crate::SymbolTableNode::Import(_)
                                    | crate::SymbolTableNode::ImportFrom(_)
                            )
                        )
                    }) =>
            {
                if let ExprType::Name(n) = a.value.as_ref()
                    && let Some(SymbolTableNode::Assign { value, .. }) = symbols.get(&n.id)
                    && let ExprType::Dict(d) = value
                    && d.values.iter().all(|v| {
                        crate::is_class_value_expr(v, symbols)
                            || matches!(v, ExprType::Call(_))
                            || matches!(v, ExprType::Lambda(_))
                    })
                {
                    Some(crate::TypeInfo::Dict(
                        Box::new(crate::TypeInfo::String),
                        Box::new(crate::TypeInfo::PyValue),
                    ))
                } else {
                    infer_field_type(&a.value, name_types, symbols, options, class_name)
                }
            }
            // `copy.copy(x)` / `copy.deepcopy(x)` — the argument's type
            // (copy preserves the type: `self._store = copy.copy(
            // session_vars)` — botocore's SessionVarDict).
            ExprType::Attribute(a)
                if matches!(a.attr.as_str(), "copy" | "deepcopy")
                    && crate::root_name(&a.value).is_some_and(|r| {
                        crate::StdModule::from_name(&r) == Some(crate::StdModule::Copy)
                    })
                    && call.args.len() == 1 =>
            {
                infer_field_type(
                    &call.args[0],
                    name_types,
                    symbols,
                    options,
                    class_name,
                )
            }
            // A STRING-method call on an expression (`normalize_host(host,
            // ...).lower()`, `name.strip()`, `x.replace(a, b)`): the field
            // is a String (urllib3's ConnectionPool._tunnel_host).
            ExprType::Attribute(a)
                if matches!(
                    a.attr.as_str(),
                    "lower" | "upper" | "strip" | "lstrip" | "rstrip" | "replace"
                        | "title" | "casefold" | "split" | "rsplit" | "join"
                        | "capitalize" | "swapcase"
                ) =>
            {
                Some(crate::TypeInfo::String)
            }
            // An ASSOCIATED call (`Retry.from_int(max_retries)`) types the
            // field as the class (same-module or imported).
            ExprType::Attribute(a) => {
                // A `js.*` foreign-object call (`js.globalThis.Worker.new(
                // ...)` — pyodide, urllib3's emscripten fetch): a JS
                // interop object with no rython equivalent — a boxed value.
                if crate::root_name(&a.value)
                    .is_some_and(|r| r == "js")
                {
                    return Some(crate::TypeInfo::PyValue);
                }
                // A method call on a CALL RESULT rooted in an external
                // module (`zstd.ZstdDecompressor().decompressobj()` —
                // urllib3's ZstdDecoder): the chain's root is a foreign
                // module, so the intermediate result and the method's
                // result are both foreign objects — a boxed PyValue.
                // (The direct `mod.fn()` shape is the import_sym branch
                // below; this is its CHAINED twin, whose receiver is a
                // Call the dotted-key walk cannot follow.)
                if let ExprType::Call(inner) = a.value.as_ref()
                    && let Some(root) = crate::root_name(inner.func.as_ref())
                    && let Some(sym) = symbols.get(root)
                {
                    // `import x as y` binds y as an ALIAS of the import —
                    // follow it to the Import node (import.rs's
                    // registration shape).
                    let sym = match sym {
                        SymbolTableNode::Alias(canonical) => {
                            symbols.get(canonical).unwrap_or(sym)
                        }
                        other => other,
                    };
                    let external = match sym {
                        SymbolTableNode::Import(i) => i
                            .names
                            .first()
                            .map(|al| {
                                al.name
                                    .split('.')
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                            })
                            .is_some_and(|path: Vec<String>| {
                                !options.module_defs.contains_key(&path)
                            }),
                        SymbolTableNode::ImportFrom(i) => !options
                            .module_defs
                            .contains_key(&i.resolved_module_path(options)),
                        // `try: import x except (...):: x = None` — the
                        // Assign(None) fallback shadows the import (the
                        // tuple-handler shape registers the store): the
                        // chained call is still external.
                        SymbolTableNode::Assign { value, .. }
                            if crate::is_none_expr(value) =>
                        {
                            true
                        }
                        _ => false,
                    };
                    if external {
                        return Some(crate::TypeInfo::PyValue);
                    }
                }
                // A stdlib MODULE call (`zlib.decompressobj()`,
                // `hashlib.md5()`, `brotli.Decompressor()`,
                // `OpenSSL.SSL.Context(...)`) — a foreign object with no
                // rython equivalent — a boxed value. The root is an
                // EXTERNAL module (or a try/except fallback `brotli = None`
                // shadowing one — urllib3's response decoders; NOT an
                // in-crate class import like `Retry.from_int`).
                // Find the import symbol: `import OpenSSL.SSL` registers
                // "OpenSSL.SSL", so check the full dotted chain, not just
                // the root name.
                let import_sym = {
                    let mut cur: Option<&ExprType> = Some(&a.value);
                    let mut key = String::new();
                    for _ in 0..6 {
                        match cur {
                            Some(ExprType::Attribute(inner)) => {
                                if !key.is_empty() {
                                    key = format!("{}.{}", inner.attr, key);
                                } else {
                                    key = inner.attr.clone();
                                }
                                cur = Some(&inner.value);
                            }
                            Some(ExprType::Name(n)) => {
                                key = if key.is_empty() {
                                    n.id.clone()
                                } else {
                                    format!("{}.{}", n.id, key)
                                };
                                break;
                            }
                            _ => break,
                        }
                    }
                    if !key.is_empty() {
                        symbols.get(&key)
                    } else {
                        None
                    }
                };
                if let Some(sym) = import_sym {
                    let external = match sym {
                        SymbolTableNode::Import(i) => {
                            let path = i
                                .names
                                .first()
                                .map(|al| {
                                    al.name
                                        .split('.')
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            !options.module_defs.contains_key(&path)
                        }
                        SymbolTableNode::ImportFrom(i) => {
                            !options
                                .module_defs
                                .contains_key(&i.resolved_module_path(options))
                        }
                        // `import brotlicffi as brotli` — the alias resolves
                        // to the Import symbol (urllib3's response decoders
                        // with the dropped-ImportError fallback): follow it
                        // to the canonical name's Import/ImportFrom.
                        SymbolTableNode::Alias(canonical) => {
                            let mut hops = 0;
                            let mut cur = symbols.get(canonical);
                            let mut external = false;
                            loop {
                                if hops > 16 {
                                    break;
                                }
                                match cur {
                                    Some(SymbolTableNode::Alias(next)) => {
                                        cur = symbols.get(next);
                                    }
                                    Some(SymbolTableNode::Import(i)) => {
                                        let path = i
                                            .names
                                            .first()
                                            .map(|al| {
                                                al.name
                                                    .split('.')
                                                    .map(|s| s.to_string())
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default();
                                        external = !options.module_defs.contains_key(&path);
                                        break;
                                    }
                                    Some(SymbolTableNode::ImportFrom(i)) => {
                                        external = !options
                                            .module_defs
                                            .contains_key(&i.resolved_module_path(options));
                                        break;
                                    }
                                    _ => break,
                                }
                                hops += 1;
                            }
                            external
                        }
                        // `try: import brotli except: brotli = None` — the
                        // Assign(None) fallback shadows the import.
                        SymbolTableNode::Assign { value, .. }
                            if crate::is_none_expr(value) =>
                        {
                            true
                        }
                        _ => false,
                    };
                    if external {
                        return Some(crate::TypeInfo::PyValue);
                    }
                    // An IN-CRATE module receiver: a module-function call
                    // (`botocore.session.get_session()` — boto3's Session,
                    // where `import botocore.session` registers the dotted
                    // name): the function's return annotation types the
                    // field; an unannotated function boxes it (PyValue).
                    let path = match sym {
                        SymbolTableNode::Import(i) => i
                            .names
                            .first()
                            .map(|al| {
                                al.name
                                    .split('.')
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                            }),
                        SymbolTableNode::ImportFrom(i) => {
                            Some(i.resolved_module_path(options))
                        }
                        _ => None,
                    };
                    if let Some(path) = path
                        && options.module_defs.contains_key(&path)
                        && let Some((f, _)) =
                            crate::module_function_def(options, &path, &a.attr)
                    {
                        if let Some(ann) = f.returns.as_deref()
                            && let Some(t) = crate::resolve_alias_typeinfo(ann, symbols, options)
                        {
                            return Some(t);
                        }
                        return Some(crate::TypeInfo::PyValue);
                    }
                    // An IN-CRATE module CLASS construction
                    // (`botocore.httpsession.URLLib3Session(timeout=...)` —
                    // botocore's utils): resolve the class through the
                    // module path built from the dotted receiver chain.
                    if let Some(path) = crate::dotted_module_path(&a.value)
                        && options.module_defs.contains_key(&path)
                        && (crate::module_class_def(options, &path, &a.attr).is_some()
                            || crate::resolve_imported_class(options, &path, &a.attr, 0)
                                .is_some())
                    {
                        return Some(crate::TypeInfo::Class(a.attr.clone()));
                    }
                }
                // An EXTERNAL-module-rooted receiver whose dotted chain is
                // NOT a registered import symbol (`crt_checksums.XXHash.
                // new_xxhash64()` — botocore's httpchecksum, where only
                // `crt_checksums` is registered): a foreign object — a
                // boxed PyValue.
                if let Some(root) = crate::root_name(&a.value)
                    && let Some(sym) = symbols.get(root)
                {
                    let external = match sym {
                        SymbolTableNode::Import(i) => {
                            let path = i
                                .names
                                .first()
                                .map(|al| {
                                    al.name
                                        .split('.')
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            !options.module_defs.contains_key(&path)
                        }
                        SymbolTableNode::ImportFrom(i) => !options
                            .module_defs
                            .contains_key(&i.resolved_module_path(options)),
                        // `try: import crt_checksums except: crt_checksums =
                        // None` — the fallback shadows the import
                        // (botocore's httpchecksum).
                        SymbolTableNode::Assign { value, .. }
                            if crate::is_none_expr(value) =>
                        {
                            true
                        }
                        _ => false,
                    };
                    if external {
                        return Some(crate::TypeInfo::PyValue);
                    }
                }
                // A method call on a PyValue-typed RECEIVER (`ssl_context.
                // wrap_bio(...)` where `ssl_context: ssl.SSLContext` — a
                // boxed external param, urllib3's SSLTransport): the result
                // is likewise a boxed value. Same for a BOXED-CONTAINER
                // receiver (`kwargs.pop('status_tuple')` where `kwargs` is a
                // **kwargs PyDict — botocore's AWSHTTPResponse): the member
                // access result is a boxed value.
                if let ExprType::Name(recv) = a.value.as_ref()
                    && name_types
                        .get(&recv.id)
                        .is_some_and(|t| crate::ast::tree::type_ctx::type_contains_pyvalue(t))
                {
                    return Some(crate::TypeInfo::PyValue);
                }
                // A SELF-method call (`self._init_length(...)` — urllib3's
                // emscripten response): the method's return annotation types
                // the field.
                if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                    let Some(SymbolTableNode::ClassDef(class)) = symbols.get(class_name) else {
                        return None;
                    };
                    let Some(m) = class.method_on_mro(&a.attr, symbols) else {
                        return None;
                    };
                    let ann = m.returns.as_deref()?;
                    return crate::resolve_alias_typeinfo(ann, symbols, options);
                }
                // A method call on a SELF-FIELD (`self._boto3_session.client(
                // ...)` — boto3's ServiceDocumenter): the field is a boxed
                // PyValue, so the call result is too.
                if let ExprType::Attribute(inner) = a.value.as_ref()
                    && matches!(inner.value.as_ref(), ExprType::Name(n) if n.id == "self")
                {
                    return Some(crate::TypeInfo::PyValue);
                }
                // A method call on a chain ROOTED in an associated call or
                // a construction of a known class (`UserAgentString.
                // from_environment().with_client_config(...)` — botocore's
                // BaseClient): the chain result is the class instance.
                if let ExprType::Call(inner) = a.value.as_ref() {
                    // An EXTERNAL-module-rooted call chain
                    // (`crt_checksums.XXHash.new_xxhash64()` — botocore's
                    // httpchecksum, where crt_checksums is a conditional
                    // import): a foreign object — a boxed PyValue.
                    if let Some(root) = crate::root_name(&a.value) {
                        let external = match symbols.get(root) {
                            Some(crate::SymbolTableNode::Import(i)) => {
                                let path = i
                                    .names
                                    .first()
                                    .map(|al| {
                                        al.name
                                            .split('.')
                                            .map(|s| s.to_string())
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();
                                !options.module_defs.contains_key(&path)
                            }
                            Some(crate::SymbolTableNode::ImportFrom(i)) => !options
                                .module_defs
                                .contains_key(&i.resolved_module_path(options)),
                            _ => false,
                        };
                        if external {
                            return Some(crate::TypeInfo::PyValue);
                        }
                    }
                    let class_ref = match inner.func.as_ref() {
                        // `Class(...)` — a construction.
                        ExprType::Name(cn) => Some(cn),
                        // `Class.method(...)` — an associated call.
                        ExprType::Attribute(ia) => match ia.value.as_ref() {
                            ExprType::Name(cn) => Some(cn),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(cn) = class_ref
                        && let Some(class_ty) = match symbols.get(&cn.id) {
                            Some(SymbolTableNode::ClassDef(_)) => {
                                Some(crate::TypeInfo::Class(cn.id.clone()))
                            }
                            Some(SymbolTableNode::ImportFrom(i)) => {
                                let path = i.resolved_module_path(options);
                                if crate::module_class_def(options, &path, &cn.id).is_some()
                                    || crate::resolve_imported_class(options, &path, &cn.id, 0)
                                        .is_some()
                                {
                                    Some(crate::TypeInfo::Class(cn.id.clone()))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    {
                        // A FIELD read on the construction (`TempDirectory(
                        // ...).path` — pip's VenvBuildEnvironment): the
                        // class's annotated field type.
                        if let Some(class) = match symbols.get(&cn.id) {
                            Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                            Some(SymbolTableNode::ImportFrom(i)) => {
                                let path = i.resolved_module_path(options);
                                crate::module_class_def(options, &path, &cn.id)
                                    .map(|(c, _)| c)
                                    .or_else(|| {
                                        crate::resolve_imported_class(options, &path, &cn.id, 0)
                                            .map(|(c, _)| c)
                                    })
                            }
                            _ => None,
                        } {
                            if let Some(field_ty) = class.body.iter().find_map(|s| {
                                match &s.statement {
                                    crate::StatementType::AnnotatedName {
                                        name, annotation, ..
                                    } if name == &a.attr => {
                                        crate::annotation_type_info(annotation)
                                    }
                                    // A @property accessor read on the
                                    // construction (`TempDirectory(...).path
                                    // ` — pip's VenvBuildEnvironment): the
                                    // property's return annotation.
                                    crate::StatementType::FunctionDef(f)
                                        if f.name == a.attr
                                            && f.decorator_list.iter().any(|d| {
                                                matches!(d, ExprType::Name(n) if n.id == "property")
                                                    || matches!(
                                                        d,
                                                        ExprType::Attribute(at)
                                                            if at.attr == "property"
                                                    )
                                            }) =>
                                    {
                                        f.returns
                                            .as_deref()
                                            .and_then(crate::annotation_type_info)
                                    }
                                    _ => None,
                                }
                            }) {
                                return Some(field_ty);
                            }
                        }
                        return Some(class_ty);
                    }
                }
                let ExprType::Name(class) = a.value.as_ref() else {
                    return None;
                };
                match symbols.get(&class.id) {
                    Some(SymbolTableNode::ClassDef(_)) => {
                        Some(crate::TypeInfo::Class(a.attr.clone()))
                    }
                    Some(SymbolTableNode::ImportFrom(i)) => {
                        let path = i.resolved_module_path(options);
                        // TWO shapes: the receiver is the CLASS itself
                        // (`UserAgentString.from_environment()` — an
                        // associated call, class = `class.id`), or the
                        // receiver is a SUBMODULE with the class as the
                        // attribute (`functions.Functions()` — jmespath's
                        // TreeInterpreter, class = `a.attr` in path+sub).
                        if crate::module_class_def(options, &path, &class.id).is_some() {
                            Some(crate::TypeInfo::Class(class.id.clone()))
                        } else {
                            let mut p2 = path.clone();
                            p2.push(class.id.clone());
                            if crate::module_class_def(options, &path, &a.attr).is_some()
                                || crate::module_class_def(options, &p2, &a.attr).is_some()
                            {
                                Some(crate::TypeInfo::Class(a.attr.clone()))
                            } else {
                                None
                            }
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        },
        // A conditional (`x if cond else y`): either arm's type (the
        // literal False arm in `is_punctuation(c) if printable else
        // False`).
        ExprType::IfExp(e) => infer_field_type(&e.body, name_types, symbols, options, class_name)
            .or_else(|| infer_field_type(&e.orelse, name_types, symbols, options, class_name)),
        // A None store (`self.current_buffer = None` — urllib3's emscripten
        // fetch, later filled with a JS value): the field is the boxed
        // PyValue, which ABSORBS None (`self.x = None` → PyValue::None_).
        // A documented divergence: a plain None-only attribute is a boxed
        // value instead of a typed Option. None arrives as NoneType,
        // Constant(None), or the bare Name "None".
        other if crate::is_none_expr(other) => Some(crate::TypeInfo::PyValue),
        // An ATTRIBUTE READ off a call result (`self.type =
        // urlparse(self._r.url).scheme` — requests' cookies.MockRequest,
        // issue #137): a dynamic member of a foreign object — a boxed
        // PyValue (the external-object divergence).
        ExprType::Attribute(a) if matches!(a.value.as_ref(), ExprType::Call(_)) => {
            Some(crate::TypeInfo::PyValue)
        }
        // A list comprehension of foreign objects (`self._decoders =
        // [_get_decoder(e) for e in ...]` — urllib3's MultiDecoder): the
        // element type is a boxed PyValue.
        ExprType::ListComp(_) => Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue))),
        // Logical combinations (`self.common_cjk = self.is_cjk and
        // character in COMMON_CJK_CHARACTERS`, `not x`) are bool — UNLESS
        // a branch is a boxed value (`excluded_params or frozenset()` —
        // botocore's EndpointProvider, where the param is PyValue): the
        // combination takes the boxed value. Round 62 extends the operand
        // view: `x or y` returns an OPERAND (never a bool), so an
        // Option-typed operand keeps the Option (`self.headers = headers
        // or {}` where headers is `Mapping[str, str] | None` — urllib3's
        // RequestMethods: the fold's Option arm Some-wraps the default,
        // so the field is Option<PyDict>, matching the store). The
        // operands are typed through the SAME fold_operand_type the fold
        // uses, so the field always matches the fold's output; unknown
        // operands keep the Bool fallback (`flag and check()` — both
        // bool, exactly the round-55 shape).
        ExprType::BoolOp(b) => {
            let field_ctx = crate::CodeGenContext::Class(class_name.to_string());
            // Operand typing must be CONTEXT-INDEPENDENT: infer_fields is
            // consulted from many codegen phases (trait generation, the
            // struct, method bodies) whose function-scoped options differ.
            // A NAME operand resolves through the caller's explicit
            // name_types map (the __init__ parameter/local types) before
            // any options-based inference — fold_operand_type alone would
            // type `headers` (a `Mapping[str, str] | None` __init__
            // parameter) as PyObject under module-level options but
            // Option<PyDict> under the method's, making the field type
            // depend on who asked.
            let operand_ty = |v: &crate::ExprType| -> crate::TypeInfo {
                if let crate::ExprType::Name(n) = v {
                    if let Some(t) = name_types.get(&n.id) {
                        return t.clone();
                    }
                }
                crate::ast::tree::bool_ops::fold_operand_type(
                    v, &field_ctx, options, symbols,
                )
            };
            let tys: Vec<crate::TypeInfo> = b.values.iter().map(operand_ty).collect();
            if tys
                .iter()
                .any(|t| matches!(t, crate::TypeInfo::PyValue))
            {
                return Some(crate::TypeInfo::PyValue);
            }
            let mut has_option = false;
            let mut plain: Vec<crate::TypeInfo> = Vec::new();
            for t in tys {
                match t {
                    crate::TypeInfo::Option(inner) => {
                        has_option = true;
                        plain.push((*inner).clone());
                    }
                    other => plain.push(other),
                }
            }
            let Some(mut u) = plain.pop() else {
                return Some(crate::TypeInfo::Bool);
            };
            for t in plain {
                u = crate::ast::tree::type_ctx::unify(u, t);
            }
            // A unified result containing an UNKNOWN element (`Dict(
            // PyObject, PyObject)` — the empty-dict literal typed against
            // an unknown Option inner) renders `PyDict<_, _>`, which is
            // E0121 in a field signature: fall back to Bool (the fold's
            // Option arm still Some-wraps; the mismatch stays loud where
            // it cannot type). Only fully-known types become fields.
            if crate::ast::tree::type_ctx::type_mentions_pyobject(&u) {
                return Some(crate::TypeInfo::Bool);
            }
            if has_option && !matches!(u, crate::TypeInfo::PyObject) {
                Some(crate::TypeInfo::Option(Box::new(u)))
            } else if !matches!(u, crate::TypeInfo::PyObject) {
                Some(u)
            } else {
                Some(crate::TypeInfo::Bool)
            }
        }
        ExprType::UnaryOp(_) => Some(crate::TypeInfo::Bool),
        // A comparison (`x in ys`, `a == b`) is a bool field
        // (`self.safe = character in COMMON_SAFE_ASCII_CHARACTERS`).
        ExprType::Compare(_) => Some(crate::TypeInfo::Bool),
        // A BinOp with a STRING-literal operand (`get_indentation() * " "`
        // — pip's spinner): a string repetition — String.
        ExprType::BinOp(b) => {
            let is_str_lit = |e: &ExprType| -> bool {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            };
            if is_str_lit(&b.left) || is_str_lit(&b.right) {
                Some(crate::TypeInfo::String)
            } else {
                // A SHIFT/BITWISE BinOp (`1 << bit_no` — rich's _Bit
                // descriptor): integer-only — i64.
                if matches!(
                    b.op,
                    crate::BinOps::LShift
                        | crate::BinOps::RShift
                        | crate::BinOps::BitOr
                        | crate::BinOps::BitXor
                        | crate::BinOps::BitAnd
                ) {
                    Some(crate::TypeInfo::Int)
                } else {
                    // A BinOp over a boxed/foreign operand (`1 + len(archive)`
                    // where `archive = self.loader.archive` is a PyValue —
                    // distlib's ZipResourceFinder): the result is a boxed
                    // PyValue (a PyValue operand poisons the whole BinOp).
                    let poisoned = [&b.left, &b.right].iter().any(|e| {
                        infer_field_type(e, name_types, symbols, options, class_name)
                            .is_some_and(|t| crate::ast::tree::type_ctx::type_contains_pyvalue(&t))
                    });
                    if poisoned {
                        Some(crate::TypeInfo::PyValue)
                    } else {
                        None
                    }
                }
            }
        }
        // A CLASS-CONSTANT read (`self._state = GzipDecoderState.FIRST_MEMBER`
        // — urllib3's response decoders): the class's class-level constants
        // are metadata (the class lowers as a plain struct), but the read
        // value is an int in practice.
        ExprType::Attribute(a) => {
            // A read through a getattr(...) result (`getattr(ParameterType,
            // parameter_type.lower()).value` — botocore's
            // ParameterDefinition): an enum member read via dynamic
            // attribute lookup — a boxed PyValue.
            if let ExprType::Call(c) = a.value.as_ref()
                && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "getattr")
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A chain rooted in an external-module CALL
            // (`urllib.parse.urlsplit(url).netloc` — pip's PackageIndex):
            // a foreign object — a boxed value. The callee chain's dotted
            // PREFIXES are checked against external import symbols
            // (`import urllib.parse` registers only "urllib.parse").
            if let ExprType::Call(c) = a.value.as_ref()
                && let Some(parts) = crate::dotted_module_path(c.func.as_ref())
                && (0..parts.len()).any(|i| {
                    let key = parts[..i + 1].join(".");
                    let sym = symbols.get(&key);
                    match sym {
                        Some(SymbolTableNode::Import(im)) => !options.module_defs.contains_key(
                            &im.names
                                .first()
                                .map(|al| {
                                    al.name
                                        .split('.')
                                        .map(|s| s.to_string())
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default(),
                        ),
                        Some(SymbolTableNode::ImportFrom(ifm)) => !options
                            .module_defs
                            .contains_key(&ifm.resolved_module_path(options)),
                        _ => false,
                    }
                })
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A field read on a CLASS CONSTRUCTION (`TempDirectory(...)
            // .path` — pip's VenvBuildEnvironment): the class's annotated
            // field or @property type.
            if let ExprType::Call(inner) = a.value.as_ref()
                && let Some(cn) = match inner.func.as_ref() {
                    ExprType::Name(cn) => Some(cn),
                    _ => None,
                }
                && let Some(class) = (match symbols.get(&cn.id) {
                    Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                    Some(SymbolTableNode::ImportFrom(i)) => {
                        let path = i.resolved_module_path(options);
                        crate::module_class_def(options, &path, &cn.id)
                            .map(|(c, _)| c)
                            .or_else(|| {
                                crate::resolve_imported_class(options, &path, &cn.id, 0)
                                    .map(|(c, _)| c)
                            })
                    }
                    _ => None,
                })
            {
                if let Some(field_ty) = class.body.iter().find_map(|s| match &s.statement {
                    crate::StatementType::AnnotatedName {
                        name, annotation, ..
                    } if name == &a.attr => {
                        crate::annotation_type_info(annotation)
                    }
                    crate::StatementType::FunctionDef(f)
                        if f.name == a.attr
                            && f.decorator_list.iter().any(|d| {
                                matches!(d, ExprType::Name(n) if n.id == "property")
                                    || matches!(
                                        d,
                                        ExprType::Attribute(at) if at.attr == "property"
                                    )
                            }) =>
                    {
                        f.returns
                            .as_deref()
                            .and_then(crate::annotation_type_info)
                    }
                    _ => None,
                }) {
                    return Some(field_ty);
                }
                return Some(crate::TypeInfo::Class(cn.id.clone()));
            }
            // A dotted chain rooted in an EXTERNAL module read as a VALUE
            // (`self._mfa_prompter = getpass.getpass` — botocore's
            // AssumeRoleCredentialFetcher): a module function/callable
            // reference held as data has no rython value equivalent — a
            // boxed PyValue (callable-as-value divergence, issue #122).
            if let Some(root) = crate::root_name(&a.value)
                && matches!(
                    symbols.get(root),
                    Some(
                        crate::SymbolTableNode::Import(_)
                            | crate::SymbolTableNode::ImportFrom(_)
                    )
                )
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // An attribute read on an UNKNOWN local (`context.bin_path`
            // where `context = env.ensure_directories(...)` — pip's venv):
            // a boxed PyValue (the external-object divergence).
            if let ExprType::Name(n) = a.value.as_ref()
                && !name_types.contains_key(&n.id)
                && symbols.get(&n.id).is_none()
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A BOUND-METHOD read on a self-field SUBSCRIPT
            // (`self.get = self._entries[-1].get` — rich's ThemeStack): a
            // callable held as data — a boxed PyValue (callable-as-value
            // divergence, issue #122).
            if let ExprType::Subscript(s) = a.value.as_ref()
                && matches!(s.value.as_ref(), ExprType::Attribute(t)
                    if matches!(t.value.as_ref(), ExprType::Name(n) if n.id == "self"))
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A FIELD read on a local typed as a CLASS (`scheme.scripts`
            // where `scheme = get_scheme(...)` returns the Scheme class —
            // pip's Prefix): resolve the field type from the class's
            // annotated fields. The class may be defined in ANOTHER module
            // of the crate (the local's type came from a cross-module
            // return annotation) — search the module defs.
            if let ExprType::Name(recv) = a.value.as_ref()
                && let Some(crate::TypeInfo::Class(class_name2)) = name_types.get(&recv.id)
                && let Some(c) = (match symbols.get(class_name2) {
                    Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                    _ => options.module_defs.values().find_map(|m| {
                        let m: &crate::Module = m;
                        m.raw.body.iter().find_map(|s| match &s.statement {
                            crate::StatementType::ClassDef(c) if &c.name == class_name2 => {
                                Some(c.clone())
                            }
                            _ => None,
                        })
                    }),
                })
            {
                let field_ty = c.body.iter().find_map(|s| match &s.statement {
                    crate::StatementType::AnnotatedName { name, annotation } if name == &a.attr => {
                        crate::annotation_type_info(annotation).or_else(|| {
                            crate::resolve_alias_typeinfo(annotation, symbols, options)
                        })
                    }
                    crate::StatementType::FunctionDef(f)
                        if f.name == a.attr
                            && f.decorator_list.iter().any(|d| {
                                matches!(d, ExprType::Name(n) if n.id == "property")
                                    || matches!(
                                        d,
                                        ExprType::Attribute(at) if at.attr == "property"
                                    )
                            }) =>
                    {
                        f.returns
                            .as_deref()
                            .and_then(crate::annotation_type_info)
                    }
                    _ => None,
                });
                if field_ty.is_some() {
                    return field_ty;
                }
                // A class-typed receiver's member that is NOT a resolvable
                // field (`session.resume_retries` where PipSession's
                // attribute is dynamic — pip's Downloader): a boxed
                // PyValue (external-member divergence).
                return Some(crate::TypeInfo::PyValue);
            }
            // A SELF-FIELD chain (`self._options.custom_functions` —
            // jmespath's TreeInterpreter): the member of a class-typed
            // self-field is a boxed PyValue (the member's type is not
            // statically known at this depth).
            if let ExprType::Attribute(inner) = a.value.as_ref()
                && matches!(inner.value.as_ref(), ExprType::Name(n) if n.id == "self")
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // An attribute chain rooted in a PyValue-typed local
            // (`self._resource.meta.client` where `resource` is an
            // unannotated param — boto3's BaseDocumenter): the member is a
            // boxed PyValue (external-object divergence).
            if let Some(recv) = crate::root_name(&a.value)
                && name_types
                    .get(recv)
                    .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue))
            {
                return Some(crate::TypeInfo::PyValue);
            }
            if let ExprType::Name(class) = a.value.as_ref()
                && let Some(SymbolTableNode::ClassDef(c)) = symbols.get(&class.id)
                && c.body.iter().any(|s| {
                    matches!(
                        &s.statement,
                        crate::StatementType::Assign(assign)
                            if assign.targets.len() == 1
                                && matches!(&assign.targets[0], ExprType::Name(n) if n.id == a.attr)
                    )
                })
            {
                Some(crate::TypeInfo::Int)
            } else {
                None
            }
        }
        // A dict-literal / comparison / class-constant arm is above; here
        // add a MODULE-DICT lookup (`self.protocol = _openssl_versions[
        // protocol]` — urllib3's PyOpenSSLContext, where the dict is typed
        // `dict[int, int]`): the value type.
        ExprType::Subscript(s) => {
            if let ExprType::Name(dict) = s.value.as_ref()
                && let Some(SymbolTableNode::Assign { value, .. }) = symbols.get(&dict.id)
                && let ExprType::Dict(d) = value
            {
                // A module dict lookup (`self.protocol = _openssl_versions[
                // protocol]` — urllib3's PyOpenSSLContext): the value type
                // is the dict's ANNOTATION's value type when present
                // (`dict[int, int]`), else the literal's int-ness.
                let all_int = d.values.iter().all(|v| {
                    matches!(v, ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::Integer(_))))
                });
                if all_int {
                    return Some(crate::TypeInfo::Int);
                }
            }
            // An annotated module dict lookup with unresolvable elements
            // (`self.protocol = _openssl_versions[protocol]` where the
            // values are OpenSSL constants — urllib3's PyOpenSSLContext):
            // the field is a boxed PyValue (external-class divergence).
            if let ExprType::Name(dict) = s.value.as_ref()
                && matches!(
                    symbols.get(&dict.id),
                    Some(crate::SymbolTableNode::Assign { .. })
                )
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A subscript READ on a PyValue-typed local or parameter
            // (`self._base_default_config = default_config_data['base']` —
            // botocore's DefaultConfigResolver, where default_config_data
            // is an unannotated __init__ param): the member of a boxed
            // value is a boxed value.
            if let ExprType::Name(dict) = s.value.as_ref()
                && name_types
                    .get(&dict.id)
                    .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue))
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A subscript READ on a Vec-typed local (`self.capacity =
            // ident[4]` where `ident = self._read("16B")` is a bytes
            // buffer — pip's ELFFile): the member is the element type
            // (bytes/str index into Vec<u8> → u8, Vec<String> → String).
            if let ExprType::Name(dict) = s.value.as_ref()
                && let Some(t) = name_types.get(&dict.id)
                && let Some(inner) = vec_element_type(t)
            {
                return Some(inner);
            }
            // A subscript READ on a SELF-FIELD dict
            // (`self._path_to_urls = self._paths_to_urls[path]` — pip's
            // sources): the member of a field dict — a boxed value.
            if let ExprType::Attribute(attr) = s.value.as_ref()
                && matches!(attr.value.as_ref(), ExprType::Name(n) if n.id == "self")
            {
                return Some(crate::TypeInfo::PyValue);
            }
            // A subscript READ on any other local (`self.capacity =
            // ident[4]` where `ident = self._read("16B")` — pip's
            // ELFFile, a bytes buffer whose local type analysis does not
            // reach the try-block assignment): the member of an unknown
            // local — a boxed PyValue (unknown-local divergence).
            if let ExprType::Name(_) = s.value.as_ref() {
                return Some(crate::TypeInfo::PyValue);
            }
            // A subscript SLICE of a `cast(T, ...)` call (`self.frames =
            // cast(List[str], spinner["frames"])[:]` — rich's Spinner):
            // the cast's annotation types the whole store.
            if let ExprType::Call(c) = s.value.as_ref()
                && matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "cast")
                && let Some(ann) = c.args.first()
                && let ExprType::Subscript(sub) = ann
                && matches!(sub.value.as_ref(), ExprType::Name(sn)
                    if matches!(sn.id.as_str(), "List" | "list"))
                && let crate::SubscriptKind::Index(elt) = &sub.kind
            {
                let t = crate::annotation_type_info(elt);
                if matches!(t, Some(crate::TypeInfo::String)) {
                    return Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::String)));
                }
            }
            None
        }
        // A SET LITERAL of strings (`self._feature_ids =
        // {'CREDENTIALS_PROFILE_LOGIN', 'CREDENTIALS_LOGIN'}` — botocore's
        // LoginProvider): a String set. A set of unknown elements boxes as
        // PyValue (the set-bookkeeping divergence).
        ExprType::Set(s) => {
            if s.elts.iter().all(|e| {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            }) {
                Some(crate::TypeInfo::HashSet(Box::new(crate::TypeInfo::String)))
            } else {
                Some(crate::TypeInfo::PyValue)
            }
        }
        // An EMPTY dict/list store (`self._method_cache = {}` — jmespath's
        // Visitor, later `.get`/`[k] = v`): a boxed PyDict<String, PyValue>
        // / Vec<PyValue> (the element types are unknowable at the store).
        ExprType::Dict(d) if d.keys.is_empty() => {
            Some(crate::TypeInfo::Dict(Box::new(crate::TypeInfo::String), Box::new(crate::TypeInfo::PyValue)))
        }
        // A NON-EMPTY dict literal (`self._context = {'special_shape_types':
        // {}}` — botocore's ShapeDocumenter): a boxed PyDict<String,
        // PyValue> (the element types are not resolved at field-inference
        // depth; the boxed-dict divergence).
        ExprType::Dict(_) => Some(crate::TypeInfo::Dict(Box::new(crate::TypeInfo::String), Box::new(crate::TypeInfo::PyValue))),
        // A dict COMPREHENSION (`{tag: idx for idx, tag in ...}` — pip's
        // CandidateEvaluator): a boxed PyDict<String, PyValue>.
        ExprType::DictComp(_) => Some(crate::TypeInfo::Dict(Box::new(crate::TypeInfo::String), Box::new(crate::TypeInfo::PyValue))),
        // A tuple of string literals (`self._previous_requirement_header =
        // ("", "")` — pip's RequirementPreparer): a Vec<String> (the
        // all-str-tuple rule). A heterogeneous tuple boxes as PyValue.
        ExprType::Tuple(t) => {
            if t.elts.iter().all(|e| {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            }) {
                Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::String)))
            } else {
                Some(crate::TypeInfo::PyValue)
            }
        }
        ExprType::List(l) if l.is_empty() => Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue))),
        // A list of string literals (`self.sections = ['title', 'client',
        // ...]` — boto3's ServiceDocumenter): Vec<String>.
        ExprType::List(l)
            if !l.is_empty()
                && l.iter().all(|e| {
                    matches!(e, ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_))))
                }) =>
        {
            Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::String)))
        }
        // A NON-EMPTY list whose elements all infer to one concrete type
        // (`self._visited_profiles = [self._profile_name]` — botocore's
        // AssumeRoleProvider): Vec<that type>. An unresolvable element
        // (a PyValue-typed self-field read) boxes the Vec as PyValue
        // elements (the empty-list divergence).
        ExprType::List(l) if !l.is_empty() => {
            let mut elt_ty: Option<crate::TypeInfo> = None;
            let mut unknown = false;
            for e in l {
                let t = crate::infer_type(None, e, options, symbols);
                if matches!(t, crate::TypeInfo::PyObject) {
                    unknown = true;
                    break;
                }
                match &elt_ty {
                    None => elt_ty = Some(t),
                    Some(prev) if prev == &t => {}
                    _ => {
                        unknown = true;
                        break;
                    }
                }
            }
            if unknown {
                Some(crate::TypeInfo::Vec(Box::new(crate::TypeInfo::PyValue)))
            } else {
                elt_ty.map(|t| crate::TypeInfo::Vec(Box::new(t)))
            }
        }
        other => crate::simple_expr_typeinfo(other).map(|t| {
            // Fields own their strings: a `self._value = "5"` store makes
            // a String field, not a &'static str (the store side converts).
            if matches!(t, crate::TypeInfo::StrRef) {
                crate::TypeInfo::String
            } else {
                t
            }
        }),
    }
}

/// Whether a class-level statement is pure metadata (a method definition,
/// a class-constant assignment to a bare name, an AugAssign to a bare
/// name, `pass`, or a nested gated block): tolerated inside a
/// version-/platform-gated class-level `if` (urllib3's HTTPConnection,
/// distlib's ResourceFinder). Anything else (an executable statement) is
/// not class-level metadata.
fn class_level_metadata_body(stmts: &[crate::Statement]) -> bool {
    stmts.iter().all(|s| {
        matches!(&s.statement, crate::StatementType::Pass)
            || matches!(&s.statement, crate::StatementType::FunctionDef(_))
            || matches!(&s.statement, crate::StatementType::If(_))
            || matches!(&s.statement, crate::StatementType::Assign(a)
                if a.targets.len() == 1
                    && matches!(&a.targets[0], ExprType::Name(_)))
            || matches!(&s.statement, crate::StatementType::AugAssign(a)
                if matches!(&a.target, ExprType::Name(_)))
    })
}

/// Whether a class-body assignment VALUE is a literal-built computed
/// constant the class-LazyLock promotion can hold: a frozenset/set/list/
/// dict literal (or a call constructing one — `frozenset([...])`), with no
/// reference to module state (no bare Name/Attribute reads — those depend
/// on the module scope the class-LazyLock cannot see). urllib3's Retry:
/// `DEFAULT_ALLOWED_METHODS = frozenset(["HEAD", "GET", ...])`.
pub(crate) fn class_body_computed_constant(value: &crate::ExprType) -> bool {
    match value {
        ExprType::Call(c) => {
            // `frozenset(...)` / `set(...)` / `list(...)` / `tuple(...)`
            // of literal elements (or a dict/list/set literal directly).
            let is_collector = match c.func.as_ref() {
                ExprType::Name(n) => matches!(
                    n.id.as_str(),
                    "frozenset" | "set" | "list" | "tuple" | "dict"
                ),
                _ => false,
            };
            is_collector
                && c.args.iter().all(|a| match a {
                    ExprType::List(l) => l.iter().all(expr_is_literal),
                    ExprType::Set(s) => s.elts.iter().all(expr_is_literal),
                    ExprType::Tuple(t) => t.elts.iter().all(expr_is_literal),
                    ExprType::Dict(d) => {
                        d.keys.iter().flatten().all(expr_is_literal)
                            && d.values.iter().all(expr_is_literal)
                    }
                    _ => false,
                })
        }
        ExprType::List(l) => l.iter().all(expr_is_literal),
        ExprType::Set(s) => s.elts.iter().all(expr_is_literal),
        ExprType::Tuple(t) => t.elts.iter().all(expr_is_literal),
        ExprType::Dict(d) => {
            d.keys.iter().flatten().all(expr_is_literal)
                && d.values.iter().all(expr_is_literal)
        }
        _ => false,
    }
}

fn expr_is_literal(e: &crate::ExprType) -> bool {
    matches!(e, crate::ExprType::Constant(_))
        || matches!(e, crate::ExprType::UnaryOp(u)
            if matches!(u.operand.as_ref(), crate::ExprType::Constant(_)))
}

/// Whether a type-token string is a `Vec<T>` and, if so, the inner type
/// as a token stream (`Vec<u8>` → `u8`, `Vec<stdpython::PyValue>` →
/// `stdpython::PyValue`).
fn vec_element_type(t: &crate::TypeInfo) -> Option<crate::TypeInfo> {
    // Structural, not string-parsed (issue #137's review).
    match t {
        crate::TypeInfo::Vec(inner) => Some((**inner).clone()),
        _ => None,
    }
}

/// Whether a base expression is a `typing.*` construct (Generic[T],
/// MutableMapping[K, V], Protocol, NamedTuple, ...): metadata, not a
/// structural base — the class lowers as a plain struct.
fn is_typing_base(b: &ExprType) -> bool {
    match b {
        // A SUBSCRIPTED base — a generic (`ContextManager[None]` — pip's
        // BuildEnvironment, where the generic is imported from
        // collections.abc): the type parameter is metadata, not a
        // structural base.
        ExprType::Subscript(_) => true,
        // `typing.NamedTuple` is the field-metadata base (urllib3's
        // ProxyConfig); other `typing.*` names are Generic/Protocol/
        // MutableMapping metadata.
        ExprType::Attribute(a) => {
            matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
        }
        // The CALL form of a NamedTuple base (`typing.NamedTuple("Url",
        // [("scheme", T), ...])` — urllib3's Url) is also field metadata.
        ExprType::Call(c) => match c.func.as_ref() {
            ExprType::Attribute(a) => {
                a.attr == "NamedTuple"
                    && matches!(a.value.as_ref(), ExprType::Name(n) if crate::is_typing(&n.id))
            }
            ExprType::Name(n) => n.id == "NamedTuple",
            _ => false,
        },
        // Bare typing generics (`MutableMapping[...]`, `Generic[T]`,
        // `Protocol`, `NamedTuple`, `TypedDict`).
        ExprType::Name(n)
            if matches!(
                n.id.as_str(),
                "MutableMapping" | "Mapping" | "Generic" | "Protocol" | "NamedTuple"
                    | "TypedDict" | "Iterator" | "Iterable" | "Sequence" | "Callable"
            ) =>
        {
            true
        }
        _ => false,
    }
}

/// Whether the class (or a structural ancestor) declares a
/// `MutableMapping`/`Mapping` ABC base — `typing.MutableMapping[str,
/// str]`, `collections.abc.Mapping[K, V]`, or the bare names. That is
/// what provides the mixin METHODS Python inherits (`get`, `pop`, ...,
/// implemented through `__getitem__`/`__setitem__`/`__delitem__` —
/// HTTPHeaderDict(typing.MutableMapping[str, str]) in urllib3): the
/// call-side synthesis that reproduces them must gate on it, or a plain
/// `__getitem__`-only class would silently gain methods CPython raises
/// AttributeError for (the mapping-protocol slice, §7).
pub(crate) fn class_has_mapping_abc_base(
    class: &ClassDef,
    symbols: &SymbolTableScopes,
) -> bool {
    fn base_is_mapping(b: &ExprType) -> bool {
        let tail = match b {
            // `typing.MutableMapping[str, str]` / `collections.abc.Mapping[K, V]`
            ExprType::Subscript(s) => s.value.as_ref(),
            other => other,
        };
        match tail {
            ExprType::Attribute(a) => {
                matches!(a.attr.as_str(), "MutableMapping" | "Mapping")
                    && matches!(a.value.as_ref(), ExprType::Name(n)
                        if crate::is_typing(&n.id)
                            || crate::StdModule::from_name(&n.id)
                                == Some(crate::StdModule::Collections)
                            || n.id == "collections.abc")
            }
            ExprType::Name(n) => matches!(n.id.as_str(), "MutableMapping" | "Mapping"),
            _ => false,
        }
    }
    class
        .base_chain(symbols)
        .iter()
        .any(|c| c.bases.iter().any(base_is_mapping))
}

impl ClassDef {
    fn get_docstring(&self) -> Option<String> {
        if self.body.is_empty() {
            return None;
        }

        let expr = self.body[0].clone();
        match expr.statement {
            StatementType::Expr(e) => match e.value {
                ExprType::Constant(c) => {
                    // The Ellipsis sentinel is not a docstring (a class
                    // whose first body statement is `...`).
                    if c.0
                        .as_ref()
                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                    {
                        return None;
                    }
                    let raw_string = c.to_string();
                    Some(self.format_docstring(&raw_string))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn format_docstring(&self, raw: &str) -> String {
        let content = raw.trim_matches('"');
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return String::new();
        }

        let mut formatted = vec![lines[0].trim().to_string()];

        if lines.len() > 1 {
            if !lines[0].trim().is_empty() && !lines[1].trim().is_empty() {
                formatted.push(String::new());
            }
            for line in lines.iter().skip(1) {
                let cleaned = line.trim();
                if !cleaned.is_empty() {
                    formatted.push(cleaned.to_string());
                }
            }
        }

        formatted.join("\n")
    }
}

/// One observed `self.x = ...` store, for the whole-class attribute JOIN
/// (issue #137 round 23): a `None` literal declares the attribute without
/// typing it, a typed value contributes its Rust type (compared by its
/// rendered spelling), and a value whose shape rython cannot read forces
/// the boxed fallback.
#[derive(Debug)]
enum ObservedStore {
    NoneLiteral,
    Typed(crate::TypeInfo),
    Unknown,
}

/// Every `self.X` attribute READ in a body, recursing through control
/// flow and nested expressions (issue #137 round 23). Used to find the
/// attributes a class uses but never assigns — the ones its base owns.
///
/// Expression and statement shapes this does not model contribute
/// NOTHING, deliberately: a missed read leaves the status quo (the
/// attribute stays unknown and the generated crate fails loudly on it),
/// which is always safer than synthesizing a field the class does not
/// really have.
fn collect_self_attr_reads(
    body: &[Statement],
    out: &mut std::collections::BTreeSet<String>,
) {
    fn walk_expr(e: &ExprType, out: &mut std::collections::BTreeSet<String>) {
        match e {
            ExprType::Attribute(a) => {
                if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "self") {
                    out.insert(a.attr.clone());
                }
                walk_expr(&a.value, out);
            }
            ExprType::Call(c) => {
                walk_expr(&c.func, out);
                for a in &c.args {
                    walk_expr(a, out);
                }
                for kw in &c.keywords {
                    walk_expr(&kw.value, out);
                }
            }
            ExprType::BinOp(op) => {
                walk_expr(&op.left, out);
                walk_expr(&op.right, out);
            }
            ExprType::BoolOp(op) => {
                for v in &op.values {
                    walk_expr(v, out);
                }
            }
            ExprType::UnaryOp(op) => walk_expr(&op.operand, out),
            ExprType::Compare(cmp) => {
                walk_expr(&cmp.left, out);
                for c in &cmp.comparators {
                    walk_expr(c, out);
                }
            }
            ExprType::IfExp(e) => {
                walk_expr(&e.test, out);
                walk_expr(&e.body, out);
                walk_expr(&e.orelse, out);
            }
            ExprType::NamedExpr(e) => walk_expr(&e.right, out),
            ExprType::Dict(d) => {
                for k in d.keys.iter().flatten() {
                    walk_expr(k, out);
                }
                for v in &d.values {
                    walk_expr(v, out);
                }
            }
            ExprType::Set(s) => {
                for e in &s.elts {
                    walk_expr(e, out);
                }
            }
            ExprType::List(elts) => {
                for e in elts {
                    walk_expr(e, out);
                }
            }
            ExprType::Tuple(t) => {
                for e in &t.elts {
                    walk_expr(e, out);
                }
            }
            ExprType::Subscript(sub) => {
                walk_expr(&sub.value, out);
                match &sub.kind {
                    crate::SubscriptKind::Index(i) => walk_expr(i, out),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        for o in [lower, upper, step].into_iter().flatten() {
                            walk_expr(o, out);
                        }
                    }
                }
            }
            ExprType::Starred(s) => walk_expr(&s.value, out),
            ExprType::Await(e) => walk_expr(&e.value, out),
            ExprType::Yield(y) => {
                if let Some(v) = &y.value {
                    walk_expr(v, out);
                }
            }
            ExprType::YieldFrom(y) => walk_expr(&y.value, out),
            ExprType::FormattedValue(f) => walk_expr(&f.value, out),
            ExprType::JoinedStr(j) => {
                for v in &j.values {
                    walk_expr(v, out);
                }
            }
            _ => {}
        }
    }
    for stmt in body {
        match &stmt.statement {
            StatementType::Expr(e) => walk_expr(&e.value, out),
            StatementType::Call(c) => walk_expr(&ExprType::Call(c.clone()), out),
            StatementType::Return(Some(e)) => walk_expr(&e.value, out),
            StatementType::Assign(a) => {
                walk_expr(&a.value, out);
                for t in &a.targets {
                    walk_expr(t, out);
                }
            }
            StatementType::AugAssign(a) => {
                walk_expr(&a.target, out);
                walk_expr(&a.value, out);
            }
            StatementType::If(i) => {
                walk_expr(&i.test, out);
                collect_self_attr_reads(&i.body, out);
                collect_self_attr_reads(&i.orelse, out);
            }
            StatementType::While(w) => {
                walk_expr(&w.test, out);
                collect_self_attr_reads(&w.body, out);
                collect_self_attr_reads(&w.orelse, out);
            }
            StatementType::For(f) => {
                walk_expr(&f.iter, out);
                collect_self_attr_reads(&f.body, out);
                collect_self_attr_reads(&f.orelse, out);
            }
            StatementType::AsyncFor(f) => {
                walk_expr(&f.iter, out);
                collect_self_attr_reads(&f.body, out);
                collect_self_attr_reads(&f.orelse, out);
            }
            StatementType::With(w) => collect_self_attr_reads(&w.body, out),
            StatementType::AsyncWith(w) => collect_self_attr_reads(&w.body, out),
            StatementType::Try(t) => {
                collect_self_attr_reads(&t.body, out);
                for h in &t.handlers {
                    collect_self_attr_reads(&h.body, out);
                }
                collect_self_attr_reads(&t.orelse, out);
                collect_self_attr_reads(&t.finalbody, out);
            }
            StatementType::FunctionDef(f) | StatementType::AsyncFunctionDef(f) => {
                collect_self_attr_reads(&f.body, out);
            }
            _ => {}
        }
    }
}

impl ClassDef {
    /// The sum type of a polymorphic ROOT (hierarchy.rs): `enum AnyShape {
    /// Shape(Shape), Circle(Circle), ... }` with one variant per class in
    /// the root's subtree, plus everything a root-typed slot needs —
    /// `From` per variant (and per nested root), the root's accessors and
    /// every method of its MRO dispatching by `match` to the variant's own
    /// implementation, the runtime traits (`PyDisplay`, `PyRepr`,
    /// `PyIsNone`, `PyInherits`), a `Default` (the root variant), and the
    /// `isinstance` predicates and narrowing views (`__rython_is_X`,
    /// `__rython_as_X`) the lowering emits instead of the constant fold.
    /// Empty for a class that is not a root.
    pub(crate) fn emit_any_enum(
        &self,
        in_hierarchy: bool,
        symbols: &SymbolTableScopes,
        options: &PythonOptions,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        use crate::ast::tree::hierarchy as h;
        let Some(variants) = h::subtree(options, &self.name) else {
            return Ok(quote!());
        };
        if !options.with_std_python {
            return Ok(quote!());
        }
        let any = h::any_ident(&self.name);
        let root_ident = crate::safe_ident(&self.name);
        let vnames: Vec<proc_macro2::Ident> =
            variants.iter().map(|v| crate::safe_ident(&v.name)).collect();
        let vpaths: Vec<TokenStream> = variants.iter().map(h::variant_path).collect();

        // ---- The type, its conversions, and the runtime traits ----
        let from_impls = vnames.iter().zip(vpaths.iter()).map(|(vn, vp)| {
            quote! {
                impl std::convert::From<#vp> for #any {
                    fn from(v: #vp) -> #any { #any::#vn(v) }
                }
            }
        });
        // A NESTED root in the subtree (Rect inside Shape's): its own sum
        // type converts variant by variant.
        let nested_from = variants.iter().skip(1).filter(|v| h::subtree(options, &v.name).is_some()).map(|v| {
            let inner_any = h::any_ident(&v.name);
            let inner_any_path = match &v.module_path {
                None => quote!(#inner_any),
                Some(path) => {
                    let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
                    quote!(crate #(::#segs)* :: #inner_any)
                }
            };
            let members: Vec<proc_macro2::Ident> = h::subtree(options, &v.name)
                .map(|s| s.iter().map(|m| crate::safe_ident(&m.name)).collect())
                .unwrap_or_default();
            quote! {
                impl std::convert::From<#inner_any_path> for #any {
                    fn from(v: #inner_any_path) -> #any {
                        match v { #(#inner_any_path::#members(x) => #any::#members(x)),* }
                    }
                }
            }
        });
        let ancestors: Vec<proc_macro2::Ident> = self
            .base_chain(symbols)
            .iter()
            .map(|a| crate::safe_ident(&a.name))
            .collect();
        let runtime_impls = quote! {
            impl Default for #any {
                fn default() -> Self { #any::#root_ident(#root_ident::default()) }
            }
            impl stdpython::PyIsNone for #any {
                fn py_is_none(&self) -> bool { false }
            }
            #(impl PyInherits<#ancestors> for #any {})*
            impl stdpython::PyDisplay for #any {
                fn py_display(&self) -> String {
                    match self { #(#any::#vnames(v) => v.py_display()),* }
                }
            }
            impl stdpython::PyRepr for #any {
                fn py_repr(&self) -> String {
                    match self { #(#any::#vnames(v) => v.py_repr()),* }
                }
            }
        };

        // ---- isinstance predicates and narrowing views ----
        let mut views = TokenStream::new();
        for v in variants.iter().skip(1) {
            let is_fn = format_ident!("__rython_is_{}", v.name);
            let as_fn = format_ident!("__rython_as_{}", v.name);
            let members: Vec<proc_macro2::Ident> = match h::subtree(options, &v.name) {
                Some(s) => s.iter().map(|m| crate::safe_ident(&m.name)).collect(),
                None => vec![crate::safe_ident(&v.name)],
            };
            let (view_ty, arms) = match h::subtree(options, &v.name) {
                Some(_) => {
                    let inner_any = h::any_ident(&v.name);
                    let inner_any_path = match &v.module_path {
                        None => quote!(#inner_any),
                        Some(path) => {
                            let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
                            quote!(crate #(::#segs)* :: #inner_any)
                        }
                    };
                    (
                        inner_any_path.clone(),
                        quote!(#(#any::#members(x) => Some(#inner_any_path::#members(x.clone()))),*),
                    )
                }
                None => {
                    let vp = h::variant_path(v);
                    let vn = crate::safe_ident(&v.name);
                    (vp, quote!(#any::#vn(x) => Some(x.clone())))
                }
            };
            views.extend(quote! {
                pub fn #is_fn(&self) -> bool {
                    matches!(self, #(#any::#members(_))|*)
                }
                pub fn #as_fn(&self) -> Option<#view_ty> {
                    match self { #arms, _ => None }
                }
            });
        }

        // ---- Accessors and method delegators, per class of the MRO ----
        let chain = self.cross_module_chain(symbols, options);
        let mut inherent = TokenStream::new();
        let mut trait_impls = TokenStream::new();
        let mut seen_methods: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut seen_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
        // A method is a member of the trait of the class that FIRST
        // defines it (the topmost definer in the chain): an override
        // forwards under that ancestor's trait, never under the
        // overrider's (E0407). The chain runs root-first, so the last
        // index defining a name is its definer.
        let skip_method = |m: &FunctionDef| {
            m.name == "__init__"
                || m.decorator_list.iter().any(|d| {
                    matches!(d, ExprType::Name(n) if n.id == "staticmethod" || n.id == "classmethod")
                })
        };
        let mut definer: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, (anc, _, _, _)) in chain.iter().enumerate() {
            for m in anc.methods() {
                if skip_method(m) {
                    continue;
                }
                let name = if anc.is_property_setter(&m.name) {
                    anc.emitted_method_name(m)
                } else {
                    m.name.clone()
                };
                definer.insert(name, i);
            }
        }
        for (depth, (ancestor, a_syms, a_opts, a_path)) in chain.iter().enumerate() {
            let a_trait = format_ident!("{}Trait", ancestor.name);
            let trait_path = match a_path {
                Some(path) => {
                    let segs: Vec<_> = path.iter().map(|p| crate::safe_ident(p)).collect();
                    quote!(crate #(::#segs)* :: #a_trait)
                }
                None => quote!(#a_trait),
            };
            // Does this class's trait carry its METHODS (the full
            // machinery) or only its accessors? Same rule the emission
            // uses: same-module, the module's hierarchy set; cross-module,
            // the defining module's trait registry.
            let full_trait = if depth == 0 {
                in_hierarchy
            } else {
                match a_path {
                    None => options.hierarchy_classes.contains(&ancestor.name),
                    Some(path) => crate::ast::tree::module::module_class_traits(a_opts, path)
                        .contains_key(&ancestor.name),
                }
            };
            let has_trait = full_trait
                || (a_path.is_none()
                    && crate::ast::tree::module::class_subclassed_crate_wide(&ancestor.name, options))
                || a_path.is_some();
            let mut fwd = TokenStream::new();
            // base()/base_mut(): the trait's required accessors for a class
            // with a base — trait-qualified on the variant, since the
            // variant's INHERENT base() is its own base, not this one's.
            if let Some(b) = ancestor.base_class_with_options(a_syms, a_opts) {
                let b_ident = crate::safe_ident(&b.name);
                fwd.extend(quote! {
                    fn base(&self) -> &#b_ident {
                        match self { #(#any::#vnames(v) => <#vpaths as #trait_path>::base(v)),* }
                    }
                    fn base_mut(&mut self) -> &mut #b_ident {
                        match self { #(#any::#vnames(v) => <#vpaths as #trait_path>::base_mut(v)),* }
                    }
                });
            }
            let fields = ancestor.own_fields(a_syms, a_opts)?;
            for (fname, fty) in &fields {
                if ancestor.base_class_with_options(a_syms, a_opts).is_some()
                    && matches!(fname.as_str(), "base" | "base_mut")
                {
                    continue;
                }
                if !seen_fields.insert(fname.clone()) {
                    continue;
                }
                let f = crate::safe_ident(fname);
                let f_mut = format_ident!("{}_mut", fname);
                // A cross-module ancestor's field type names classes of
                // ITS module bare: qualify to crate paths (this module
                // need not import them) — the same rule the derived
                // class's own ancestor impls apply.
                let ty = match a_path {
                    Some(path) => qualify_cross_module_types(
                        fty.to_rust_type(),
                        path,
                        a_syms,
                        a_opts,
                        options,
                    ),
                    None => fty.to_rust_type(),
                };
                inherent.extend(quote! {
                    pub fn #f(&self) -> #ty {
                        match self { #(#any::#vnames(v) => v.#f()),* }
                    }
                    pub fn #f_mut(&mut self) -> &mut #ty {
                        match self { #(#any::#vnames(v) => v.#f_mut()),* }
                    }
                });
                fwd.extend(quote! {
                    fn #f(&self) -> #ty { #any::#f(self) }
                    fn #f_mut(&mut self) -> &mut #ty { #any::#f_mut(self) }
                });
            }
            for m in ancestor.methods() {
                if skip_method(m) {
                    continue;
                }
                let mut emitted = m.clone();
                if ancestor.is_property_setter(&m.name) {
                    emitted.name = ancestor.emitted_method_name(m);
                }
                // The inherent delegator takes the most-derived rendering
                // (first seen); the trait forwarder goes under the
                // definer's trait, rendered in the definer's own scope.
                let first_seen = seen_methods.insert(emitted.name.clone());
                let is_definer = definer.get(&emitted.name) == Some(&depth);
                if !first_seen && !is_definer {
                    continue;
                }
                let force_mut_self = options
                    .trait_mut_self
                    .get(&ancestor.name)
                    .is_some_and(|s| s.contains(&m.name));
                let trait_ctx = CodeGenContext::Trait {
                    class: ancestor.name.clone(),
                    generic: false,
                    super_target: None,
                    force_mut_self,
                };
                let rendered = emitted.to_rust(trait_ctx, a_opts.clone(), a_syms.clone())?;
                let rendered = match a_path {
                    Some(path) => qualify_cross_module_types(rendered, path, a_syms, a_opts, options),
                    None => rendered,
                };
                let Some((head, name, args)) = h::split_fn(&rendered) else {
                    continue;
                };
                let call_args = quote!(#(#args),*);
                let fwd_args = quote!(#(, #args)*);
                if first_seen {
                    // Dispatch through the DEFINER's trait when it carries
                    // the method: every variant implements it with the
                    // trait's one signature (a conforming override is
                    // re-emitted there; a non-conforming one falls to the
                    // default, the covariant-override divergence the
                    // trait path already documents). A variant's inherent
                    // override may take different parameters (`urlopen(
                    // method, url, redirect, **kw)` on PoolManager vs the
                    // RequestMethods signature — urllib3), which the
                    // uniform arm cannot call.
                    let arms: Vec<TokenStream> = vnames
                        .iter()
                        .zip(vpaths.iter())
                        .map(|(vn, vp)| {
                            let call = if full_trait && is_definer {
                                quote!(<#vp as #trait_path>::#name(v #fwd_args))
                            } else {
                                quote!(v.#name(#call_args))
                            };
                            quote!(#any::#vn(v) => #call)
                        })
                        .collect();
                    inherent.extend(quote! {
                        pub #head {
                            match self { #(#arms),* }
                        }
                    });
                }
                if full_trait && is_definer {
                    fwd.extend(quote! {
                        #head { #any::#name(self #fwd_args) }
                    });
                }
            }
            if has_trait {
                trait_impls.extend(quote! {
                    impl #trait_path for #any { #fwd }
                });
            }
        }
        Ok(quote! {
            #[derive(Clone)]
            pub enum #any { #(#vnames(#vpaths)),* }
            #(#from_impls)*
            #(#nested_from)*
            #runtime_impls
            impl #any {
                #views
                #inherent
            }
            #trait_impls
        })
    }
}
