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

/// Whether a class is an exception class: its name matches the exception
/// naming convention (`*Error`, `*Exception`, `*Warning`) — the same
/// heuristic `raise` uses to construct PyException values — or one of its
/// bases does. A custom exception inheriting a builtin (`IDNAError(UnicodeError)`)
/// or another custom exception (`IDNABidiError(IDNAError)`) is an exception
/// class too. Lowered as a marker struct; the runtime matches exceptions by
/// name string, so the class carries no data.
pub fn is_exception_class(class: &ClassDef) -> bool {
    let convention = |n: &str| {
        n.ends_with("Error") || n.ends_with("Exception") || n.ends_with("Warning")
    };
    if convention(&class.name) {
        return true;
    }
    class.bases.iter().any(|b| match b {
        ExprType::Name(n) => convention(&n.id),
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
                    && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
            }
            ExprType::Name(n) => n.id == "NamedTuple",
            // The CALL form (`typing.NamedTuple("Url", [...])`): the field
            // list rides in the call's second argument.
            ExprType::Call(c) => match c.func.as_ref() {
                ExprType::Attribute(a) => {
                    a.attr == "NamedTuple"
                        && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
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
            ExprType::Name(n) => matches!(
                n.id.as_str(),
                "str" | "bytes" | "bytearray" | "int" | "float" | "bool" | "list" | "dict"
                    | "tuple" | "set" | "object" | "Enum" | "IntEnum" | "Flag" | "IntFlag"
                    | "StrEnum" | "TypedDict"
            ),
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
                        .filter_map(|(i, n)| {
                            m.args.defaults
                                .get(i.saturating_sub(skip))
                                .map(|d| (n, (**d).clone()))
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
    pub fn has_property_getter(&self, name: &str) -> bool {
        self.methods().any(|m| {
            m.name == name
                && m.decorator_list.iter().any(|d| match d {
                    ExprType::Name(n) => n.id == "property",
                    ExprType::Attribute(a) => a.attr == "property",
                    _ => false,
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
    pub(crate) fn base_chain(&self, symbols: &SymbolTableScopes) -> Vec<ClassDef> {
        let mut chain = vec![self.clone()];
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(self.name.clone());
        while let Some(base) = chain.last().and_then(|c| c.base_class(symbols)) {
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
    fn owns_field(&self, attr: &str) -> bool {
        let Some(init) = self.init_method() else {
            return false;
        };
        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        stores.iter().any(|s| s.attr == attr)
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
    pub(crate) fn field_owner_depth(&self, attr: &str, symbols: &SymbolTableScopes) -> Option<usize> {
        self.base_chain(symbols)
            .iter()
            .rposition(|c| c.owns_field(attr))
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
    ) -> Option<String> {
        let chain = self.base_chain(symbols);
        let owner = chain.iter().find(|c| c.owns_field(attr))?;
        let init = owner.init_method()?;
        let mut stores = Vec::new();
        collect_field_stores(&init.body, &mut stores);
        let store = stores.iter().find(|s| s.attr == attr)?;
        let class_name = match store.value {
            ExprType::Call(call) => match call.func.as_ref() {
                ExprType::Name(n) => n.id.clone(),
                _ => return None,
            },
            ExprType::Name(n) => {
                let param = init
                    .args
                    .posonlyargs
                    .iter()
                    .chain(init.args.args.iter())
                    .chain(init.args.kwonlyargs.iter())
                    .find(|p| p.arg == n.id)?;
                match param.annotation.as_deref() {
                    Some(ExprType::Name(ann)) => ann.id.clone(),
                    _ => return None,
                }
            }
            _ => return None,
        };
        match symbols.get(&class_name) {
            Some(SymbolTableNode::ClassDef(_)) => Some(class_name),
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
        crate::analyze_scope_with(&m.body, &params, &resolve)
            .needs_mut
            .contains("self")
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
    ) -> Result<Vec<(String, TokenStream)>, Box<dyn std::error::Error>> {
        let mut fields: Vec<(String, TokenStream)> = Vec::new();
        let Some(init) = self.init_method() else {
            return Ok(fields);
        };
        // Types known for names in the __init__ body: annotated
        // parameters first, then simply-typed locals.
        let mut name_types: std::collections::HashMap<String, TokenStream> =
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
                name_types.insert(n.id.clone(), t.to_rust_type());
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
                    Some(quote!(String))
                } else if !matches!(ann, ExprType::Name(_)) {
                    if p.arg == "dist" {
                    }
                    // A union/container/alias annotation
                    // (`None | connection._TYPE_SOCKET_OPTIONS`,
                    // `tuple[str, int] | None`): resolve alias-aware.
                    let r = crate::resolve_alias_typeinfo(ann, symbols, options)
                        .map(|t| t.to_rust_type())
                        .or_else(|| crate::python_annotation_to_rust_type(ann))
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
                                        Some(SymbolTableNode::ClassDef(_)) => {
                                            let ident = crate::safe_ident(&n.id);
                                            Some(quote!(Option<#ident>))
                                        }
                                        Some(SymbolTableNode::ImportFrom(i)) => {
                                            let path = i.resolved_module_path(options);
                                            if crate::module_class_def(options, &path, &n.id)
                                                .is_some()
                                                || crate::resolve_imported_class(
                                                    options, &path, &n.id, 0,
                                                )
                                                .is_some()
                                            {
                                                let ident = crate::safe_ident(&n.id);
                                                Some(quote!(Option<#ident>))
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
                                            Some(quote!(Option<String>))
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
                            let ident = crate::safe_ident(&n.id);
                            Some(quote!(#ident))
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
                            Some(quote!(String))
                        }
                        Some(SymbolTableNode::ImportFrom(_)) => {
                            // An IMPORTED CLASS annotation that cannot be
                            // resolved (`dist: BaseDistribution` — pip's
                            // AlreadyInstalledCandidate) boxes as PyValue
                            // (the boxed-union divergence).
                            crate::resolve_alias_typeinfo(ann, symbols, options)
                                .map(|t| t.to_rust_type())
                                .or_else(|| Some(quote!(stdpython::PyValue)))
                        }
                        Some(SymbolTableNode::Alias(_))
                        // A TYPE-ALIAS name (`data: _TYPE_FIELD_VALUE` —
                        // urllib3's RequestField, `typing.Union[str,
                        // bytes]`): resolve the alias value.
                        | Some(SymbolTableNode::Assign { .. }) => {
                            crate::resolve_alias_typeinfo(ann, symbols, options)
                                .map(|t| t.to_rust_type())
                        }
                        _ => crate::python_annotation_to_rust_type(ann).or_else(|| {
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
                                    Some(quote!(stdpython::PyValue))
                                }
                                _ => None,
                            }
                        }),
                    }
                } else {
                    crate::python_annotation_to_rust_type(ann)
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
                name_types.insert(p.arg.clone(), quote!(stdpython::PyValue));
            }
        }
        // Issue #120: the `**kwargs` parameter is a boxed heterogeneous
        // dict (`PyDict<String, PyValue>`); a field stored from it
        // (`self.conn_kw = conn_kw` — urllib3's ConnectionPool) takes the
        // same type.
        if let Some(kwarg) = &init.args.kwarg {
            name_types.insert(kwarg.arg.clone(), quote!(PyDict<String, PyValue>));
        }
        // The `*args` parameter collects extra positionals as a boxed
        // heterogeneous Vec (`self._args = args` — s3transfer's
        // FunctionContainer).
        if let Some(vararg) = &init.args.vararg {
            name_types.insert(vararg.arg.clone(), quote!(Vec<stdpython::PyValue>));
        }

        // Class-level annotated declarations (`config: dict[str, Any]`)
        // pin field types for stores whose value cannot be inferred
        // (`self.config = {}`).
        let mut class_annotations: std::collections::HashMap<String, TokenStream> =
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
                let ty = crate::python_annotation_to_rust_type(annotation)
                    .or_else(|| {
                        crate::annotation_type_info(annotation).map(|t| t.to_rust_type())
                    })
                    .or_else(|| {
                        crate::resolve_alias_typeinfo(annotation, symbols, options)
                            .map(|t| t.to_rust_type())
                    })
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
                                Some(quote!(PyDict<String, PyValue>))
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
                let alias_ty = crate::resolve_alias_typeinfo(a, symbols, options)
                    .map(|t| t.to_rust_type());
                let ty_tokens = |t: &ExprType| -> Option<TokenStream> {
                    crate::python_annotation_to_rust_type(t).or_else(|| match t {
                        ExprType::Name(n)
                            if matches!(
                                symbols.get(&n.id),
                                Some(SymbolTableNode::ClassDef(_))
                            ) =>
                        {
                            let ident = crate::safe_ident(&n.id);
                            Some(quote!(#ident))
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
                    let inner = ty_tokens(inner)?;
                    Some(quote!(Option<#inner>))
                } else {
                    ty_tokens(a)
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
                                    Some(quote!(PyDict<String, PyValue>))
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
                                    Some(quote!(stdpython::PyValue))
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
                        Some((_, prev)) if prev.to_string() == ty.to_string() => {}
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
                        Some((_, prev))
                            if (prev.to_string() == "stdpython :: PyValue"
                                && ty.to_string() != "stdpython :: PyValue")
                                || (ty.to_string() == "stdpython :: PyValue"
                                    && prev.to_string() != "stdpython :: PyValue") =>
                        {
                            let idx = fields
                                .iter()
                                .position(|(name, _)| name == &store.attr)
                                .unwrap();
                            let winner = if prev.to_string() == "stdpython :: PyValue" {
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
                            if prev.to_string() != "stdpython :: PyValue"
                                && ty.to_string() != "stdpython :: PyValue"
                                && prev.to_string() != ty.to_string()
                            {
                                let idx = fields
                                    .iter()
                                    .position(|(name, _)| name == &store.attr)
                                    .unwrap();
                                fields[idx] = (
                                    store.attr.clone(),
                                    quote!(stdpython::PyValue),
                                );
                            } else {
                                return Err(format!(
                                    "attribute `self.{}` of class `{}` is assigned \
                                     conflicting types ({} and {}); a struct field needs \
                                     one type",
                                    store.attr, self.name, prev, ty
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
    ) -> Result<Vec<(String, TokenStream)>, Box<dyn std::error::Error>> {
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
            let doc = self
                .get_docstring()
                .map(|d| format!("#[doc = \"{}\"]\n", d.replace('"', "\\\"")))
                .unwrap_or_default();
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
            let doc = self
                .get_docstring()
                .map(|d| format!("#[doc = \"{}\"]\n", d.replace('"', "\\\"")))
                .unwrap_or_default();
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
                        if matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "abc")
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
        let is_metadata_base = |id: &str| {
            matches!(
                id,
                "object" | "Enum" | "IntEnum" | "Flag" | "IntFlag" | "StrEnum" | "TypedDict"
                    // `str` (a str-subclass wrapper like botocore's
                    // ClientConfigString): metadata — the class lowers as a
                    // plain struct (the class-as-value divergence).
                    | "str" | "bytes" | "int" | "float" | "bool" | "list" | "dict" | "tuple"
                    | "set"
                    // `type` — a METACLASS base (`class LexerMeta(type)` —
                    // pygments' lexer): a metaclass is a class factory,
                    // which rython cannot express as a value — metadata;
                    // the class lowers as a plain struct (the
                    // class-as-value divergence).
                    | "type"
            )
        };
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
                        Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
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
                            && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")) =>
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
                    let ident = crate::safe_ident(&n.id);
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
                    class_lazylock_constants.extend(quote! {
                        pub static #ident: std::sync::LazyLock<stdpython::PyValue> =
                            std::sync::LazyLock::new(|| stdpython::PyValue::from(#value_tokens));
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
                        matches!(b, ExprType::Name(n)
                            if matches!(n.id.as_str(), "Enum" | "IntEnum" | "Flag" | "IntFlag" | "StrEnum"))
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
            .map(|(name, ty)| {
                let ident = crate::safe_ident(name);
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

        Ok(quote! {
            #docs
            #[derive(Clone, Default)]
            pub struct #class_name {
                #(#field_defs),*
            }
            #trait_stream
            impl #class_name {
                #class_constants
                #class_lazylock_constants
                #constructor
                #init_forwarder
                #methods_stream
            }
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
    fn emit_trait(
        &self,
        base: &Option<ClassDef>,
        fields: &[(String, TokenStream)],
        methods: &[FunctionDef],
        options: &PythonOptions,
        symbols: &SymbolTableScopes,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let class_name = crate::safe_ident(&self.name);
        let trait_name = format_ident!("{}Trait", self.name);

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
            own_impl_body.extend(quote! {
                fn #f(&self) -> #fty {
                    self.#f.clone()
                }
                fn #f_mut(&mut self) -> &mut #fty {
                    &mut self.#f
                }
            });
        }

        // The per-class trait is PUBLIC (inherited methods are called
        // cross-module: the struct is `pub` and re-exported, so the traits
        // carrying its methods must be nameable wherever the struct is) and
        // declares the direct base's trait as a supertrait. Trait default
        // bodies are generic over `Self: {Name}Trait` only, so a new method
        // that calls an inherited method (`def bar(self): self.foo()` where
        // foo lives on the base) resolves `foo` through the supertrait
        // bound; ancestor methods are not on the concrete `Self` otherwise.
        let supertrait = base.as_ref().map(|b| {
            let b_trait = format_ident!("{}Trait", b.name);
            quote!(: #b_trait)
        });
        let own_trait = quote! {
            pub trait #trait_name #supertrait {
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
        let mut ancestor_impls = TokenStream::new();
        let chain = self.base_chain(symbols);
        for (depth, ancestor) in chain.iter().enumerate().skip(1).rev() {
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
            let a_fields = ancestor.own_fields(symbols, options)?;
            let mut accessor_impls = TokenStream::new();
            // The ancestor's own base accessors, if it has a base: from the
            // derived struct, its base struct is one level deeper.
            if let Some(a_base) = ancestor.base_class(symbols) {
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
            for (fname, fty) in &a_fields {
                let f = crate::safe_ident(fname);
                let f_mut = format_ident!("{}_mut", fname);
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
            let a_base = ancestor.base_class(symbols);
            let ancestor_members: Vec<&FunctionDef> = ancestor
                .methods()
                .filter(|am| {
                    am.name != "__init__"
                        && a_base.as_ref().map_or(true, |b| {
                            b.method_on_mro(&am.name, symbols).is_none()
                        })
                })
                .collect();
            let mut override_stream = TokenStream::new();
            for am in &ancestor_members {
                let mut definer: Option<FunctionDef> = None;
                let mut definer_name: Option<String> = None;
                for c in chain.iter() {
                    if c.name == ancestor.name {
                        break;
                    }
                    if let Some(m) = c.methods().find(|m| m.name == am.name) {
                        definer = Some(m.clone());
                        definer_name = Some(c.name.clone());
                        break;
                    }
                }
                if let (Some(m), Some(dname)) = (definer, definer_name) {
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
                    override_stream.extend(
                        emitted.to_rust(trait_ctx, options.clone(), symbols.clone())?,
                    );
                }
            }
            ancestor_impls.extend(quote! {
                impl #ancestor_trait for #class_name {
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
fn infer_field_type(
    value: &ExprType,
    name_types: &std::collections::HashMap<String, TokenStream>,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
    class_name: &str,
) -> Option<TokenStream> {
    match value {
        ExprType::Name(n) => name_types.get(&n.id).cloned().or_else(|| {
            // A module-level constant (`self.flags = _LATIN`, where
            // `_LATIN: int = 1` may live in another module): follow
            // Assign/ImportFrom chains to the constant's value.
            let const_type = |value: &ExprType| {
                crate::simple_expr_type(value).map(|t| {
                    if t.to_string() == "& 'static str" {
                        quote!(String)
                    } else {
                        t
                    }
                })
            };
            match symbols.get(&n.id) {
                Some(SymbolTableNode::Assign { value, .. }) => const_type(value).or_else(|| {
                    // A dict of CLASSES (`pool_classes_by_scheme = {"http":
                    // HTTPConnectionPool, ...}` — urllib3's PoolManager) or
                    // CALLABLES (`key_fn_by_scheme = {"http":
                    // functools.partial(...)}`): class/callable values have
                    // no rython value equivalent — the dict is the boxed
                    // PyDict (documented divergence).
                    if let ExprType::Dict(d) = value
                        && d.values.iter().all(|v| {
                            crate::is_class_value_expr(v, symbols)
                                || matches!(v, ExprType::Call(_))
                                || matches!(v, ExprType::Lambda(_))
                        })
                    {
                        Some(quote!(PyDict<String, PyValue>))
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
                            Some(quote!(stdpython::PyValue))
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
                        ExprType::Call(_) => Some(quote!(stdpython::PyValue)),
                        _ => None,
                    }
                }),
                // A CALLABLE value (`self.header_formatter =
                // format_multipart_header_param` — urllib3's RequestField):
                // a function reference held as data has no rython value
                // equivalent — a boxed PyValue (documented divergence).
                Some(SymbolTableNode::FunctionDef(_)) => {
                    Some(quote!(stdpython::PyValue))
                }
                // A MODULE held as a value (`import keyring` inside
                // __init__, then `self.keyring = keyring` — pip's
                // KeyRingPythonProvider): an external module object — a
                // boxed PyValue (external-object divergence).
                Some(SymbolTableNode::Import(_)) => {
                    Some(quote!(stdpython::PyValue))
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
                                    ExprType::Dict(_) => {
                                        Some(quote!(PyDict<String, PyValue>))
                                    }
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
                _ => Some(quote!(stdpython::PyValue)),
            }
        }),
        // A constructed instance of a known class types the field as that
        // class's struct.
        ExprType::Call(call) => match call.func.as_ref() {
            ExprType::Name(n) if n.id == "bool" => Some(quote!(bool)),
            // A `cast(T, ...)` typing no-op (`self.frames = cast(List[str],
            // spinner["frames"])[:]` — rich's Spinner): the cast's FIRST
            // argument is the annotation — `List[str]` → `Vec<String>`.
            ExprType::Name(n) if n.id == "cast" => {
                call.args.first().and_then(|ann| match ann {
                    ExprType::Name(sn) => match sn.id.as_str() {
                        "float" => Some(quote!(f64)),
                        "int" => Some(quote!(i64)),
                        "str" => Some(quote!(String)),
                        "bool" => Some(quote!(bool)),
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
                                    Some(quote!(Vec<String>))
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
                    "int" => quote!(i64),
                    "float" => quote!(f64),
                    _ => quote!(String),
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
                    Some(quote!(std::collections::HashSet<String>))
                } else {
                    Some(quote!(stdpython::PyValue))
                }
            }
            // `OrderedDict()` / `defaultdict()` / `dict()` — a map field
            // (urllib3's RecentlyUsedContainer._container): the boxed
            // PyDict, matching `dict[str, Any]` lowering.
            ExprType::Name(n)
                if matches!(n.id.as_str(), "dict" | "OrderedDict" | "defaultdict") =>
            {
                Some(quote!(PyDict<String, PyValue>))
            }
            // A threading lock (`RLock()`, `Lock()`, `threading.RLock()`) —
            // a stdlib object with no rython equivalent; `with self.lock:`
            // only evaluates the receiver (the __enter__/__exit__ protocol
            // is unmodeled), so the field is unit.
            ExprType::Name(n)
                if matches!(n.id.as_str(), "RLock" | "Lock" | "Semaphore") =>
            {
                Some(quote!(()))
            }
            ExprType::Attribute(a)
                if matches!(a.attr.as_str(), "RLock" | "Lock" | "Semaphore")
                    && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "threading") =>
            {
                Some(quote!(()))
            }
            // `datetime.timedelta(...)` — the stdpython timedelta struct.
            ExprType::Attribute(a)
                if a.attr == "timedelta"
                    && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "datetime") =>
            {
                Some(quote!(datetime::timedelta))
            }
            ExprType::Name(n) => match symbols.get(&n.id) {
                Some(SymbolTableNode::ClassDef(_)) => {
                    let ident = crate::safe_ident(&n.id);
                    Some(quote!(#ident))
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
                        let ident = crate::safe_ident(&n.id);
                        Some(quote!(#ident))
                    } else {
                        // A call to an IMPORTED function whose return type
                        // cannot be resolved — either no return annotation,
                        // or a re-export chain that `module_function_def`
                        // cannot follow (`from botocore import xform_name` —
                        // boto3's ResourceCollection): the field is a boxed
                        // PyValue (cross-module return-typing divergence,
                        // #123).
                        crate::call_return_typeinfo(call, Some(symbols), Some(options))
                            .map(|t| t.to_rust_type())
                            .or_else(|| Some(quote!(stdpython::PyValue)))
                    }
                }
                // A module-level function: its return annotation types the
                // field (`self.punct = is_punctuation(character)`). An
                // unannotated same-module function also boxes the field
                // (PyValue) rather than failing the module.
                _ => crate::call_return_typeinfo(call, Some(symbols), Some(options))
                    .map(|t| t.to_rust_type())
                    .or_else(|| Some(quote!(stdpython::PyValue))),
            },
            // A boolean predicate call (`character.isprintable()`,
            // `s.isascii()`) types the field bool.
            ExprType::Attribute(a)
                if a.attr.starts_with("is") && a.attr.len() > 2 => Some(quote!(bool)),
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
                    Some(quote!(PyDict<String, PyValue>))
                } else {
                    infer_field_type(&a.value, name_types, symbols, options, class_name)
                }
            }
            // `copy.copy(x)` / `copy.deepcopy(x)` — the argument's type
            // (copy preserves the type: `self._store = copy.copy(
            // session_vars)` — botocore's SessionVarDict).
            ExprType::Attribute(a)
                if matches!(a.attr.as_str(), "copy" | "deepcopy")
                    && crate::root_name(&a.value).is_some_and(|r| r == "copy")
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
                Some(quote!(String))
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
                    return Some(quote!(stdpython::PyValue));
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
                        return Some(quote!(stdpython::PyValue));
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
                            && let Some(t) = crate::python_annotation_to_rust_type(ann)
                        {
                            return Some(t);
                        }
                        return Some(quote!(stdpython::PyValue));
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
                        let ident = crate::safe_ident(&a.attr);
                        return Some(quote!(#ident));
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
                        return Some(quote!(stdpython::PyValue));
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
                        .is_some_and(|t| t.to_string().contains("PyValue"))
                {
                    return Some(quote!(stdpython::PyValue));
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
                    return crate::resolve_alias_typeinfo(ann, symbols, options)
                        .map(|t| t.to_rust_type());
                }
                // A method call on a SELF-FIELD (`self._boto3_session.client(
                // ...)` — boto3's ServiceDocumenter): the field is a boxed
                // PyValue, so the call result is too.
                if let ExprType::Attribute(inner) = a.value.as_ref()
                    && matches!(inner.value.as_ref(), ExprType::Name(n) if n.id == "self")
                {
                    return Some(quote!(stdpython::PyValue));
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
                            return Some(quote!(stdpython::PyValue));
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
                                let ident = crate::safe_ident(&cn.id);
                                Some(quote!(#ident))
                            }
                            Some(SymbolTableNode::ImportFrom(i)) => {
                                let path = i.resolved_module_path(options);
                                if crate::module_class_def(options, &path, &cn.id).is_some()
                                    || crate::resolve_imported_class(options, &path, &cn.id, 0)
                                        .is_some()
                                {
                                    let ident = crate::safe_ident(&cn.id);
                                    Some(quote!(#ident))
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
                        if let Some(class) = (match symbols.get(&cn.id) {
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
                        }) {
                            if let Some(field_ty) = class.body.iter().find_map(|s| {
                                match &s.statement {
                                    crate::StatementType::AnnotatedName {
                                        name, annotation, ..
                                    } if name == &a.attr => {
                                        crate::python_annotation_to_rust_type(annotation)
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
                                            .and_then(crate::python_annotation_to_rust_type)
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
                        let ident = crate::safe_ident(&a.attr);
                        Some(quote!(#ident))
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
                            let ident = crate::safe_ident(&class.id);
                            Some(quote!(#ident))
                        } else {
                            let mut p2 = path.clone();
                            p2.push(class.id.clone());
                            if crate::module_class_def(options, &path, &a.attr).is_some()
                                || crate::module_class_def(options, &p2, &a.attr).is_some()
                            {
                                let ident = crate::safe_ident(&a.attr);
                                Some(quote!(#ident))
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
        other if crate::is_none_expr(other) => Some(quote!(stdpython::PyValue)),
        // A list comprehension of foreign objects (`self._decoders =
        // [_get_decoder(e) for e in ...]` — urllib3's MultiDecoder): the
        // element type is a boxed PyValue.
        ExprType::ListComp(_) => Some(quote!(Vec<stdpython::PyValue>)),
        // Logical combinations (`self.common_cjk = self.is_cjk and
        // character in COMMON_CJK_CHARACTERS`, `not x`) are bool — UNLESS
        // a branch is a boxed value (`excluded_params or frozenset()` —
        // botocore's EndpointProvider, where the param is PyValue): the
        // combination takes the boxed value.
        ExprType::BoolOp(b) => {
            if b.values.iter().any(|v| {
                infer_field_type(v, name_types, symbols, options, class_name)
                    .is_some_and(|t| t.to_string().contains("PyValue"))
            }) {
                Some(quote!(stdpython::PyValue))
            } else {
                Some(quote!(bool))
            }
        }
        ExprType::UnaryOp(_) => Some(quote!(bool)),
        // A comparison (`x in ys`, `a == b`) is a bool field
        // (`self.safe = character in COMMON_SAFE_ASCII_CHARACTERS`).
        ExprType::Compare(_) => Some(quote!(bool)),
        // A BinOp with a STRING-literal operand (`get_indentation() * " "`
        // — pip's spinner): a string repetition — String.
        ExprType::BinOp(b) => {
            let is_str_lit = |e: &ExprType| -> bool {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            };
            if is_str_lit(&b.left) || is_str_lit(&b.right) {
                Some(quote!(String))
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
                    Some(quote!(i64))
                } else {
                    // A BinOp over a boxed/foreign operand (`1 + len(archive)`
                    // where `archive = self.loader.archive` is a PyValue —
                    // distlib's ZipResourceFinder): the result is a boxed
                    // PyValue (a PyValue operand poisons the whole BinOp).
                    let poisoned = [&b.left, &b.right].iter().any(|e| {
                        infer_field_type(e, name_types, symbols, options, class_name)
                            .is_some_and(|t| t.to_string().contains("PyValue"))
                    });
                    if poisoned {
                        Some(quote!(stdpython::PyValue))
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
                return Some(quote!(stdpython::PyValue));
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
                return Some(quote!(stdpython::PyValue));
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
                        crate::python_annotation_to_rust_type(annotation)
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
                            .and_then(crate::python_annotation_to_rust_type)
                    }
                    _ => None,
                }) {
                    return Some(field_ty);
                }
                let ident = crate::safe_ident(&cn.id);
                return Some(quote!(#ident));
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
                return Some(quote!(stdpython::PyValue));
            }
            // An attribute read on an UNKNOWN local (`context.bin_path`
            // where `context = env.ensure_directories(...)` — pip's venv):
            // a boxed PyValue (the external-object divergence).
            if let ExprType::Name(n) = a.value.as_ref()
                && !name_types.contains_key(&n.id)
                && symbols.get(&n.id).is_none()
            {
                return Some(quote!(stdpython::PyValue));
            }
            // A BOUND-METHOD read on a self-field SUBSCRIPT
            // (`self.get = self._entries[-1].get` — rich's ThemeStack): a
            // callable held as data — a boxed PyValue (callable-as-value
            // divergence, issue #122).
            if let ExprType::Subscript(s) = a.value.as_ref()
                && matches!(s.value.as_ref(), ExprType::Attribute(t)
                    if matches!(t.value.as_ref(), ExprType::Name(n) if n.id == "self"))
            {
                return Some(quote!(stdpython::PyValue));
            }
            // A FIELD read on a local typed as a CLASS (`scheme.scripts`
            // where `scheme = get_scheme(...)` returns the Scheme class —
            // pip's Prefix): resolve the field type from the class's
            // annotated fields. The class may be defined in ANOTHER module
            // of the crate (the local's type came from a cross-module
            // return annotation) — search the module defs.
            if let ExprType::Name(recv) = a.value.as_ref()
                && let Some(ty) = name_types.get(&recv.id)
                && let Some(class_name2) = ty.to_string().trim().split_whitespace().next()
                && let Some(c) = (match symbols.get(class_name2) {
                    Some(SymbolTableNode::ClassDef(c)) => Some(c.clone()),
                    _ => options.module_defs.values().find_map(|m| {
                        let m: &crate::Module = m;
                        m.raw.body.iter().find_map(|s| match &s.statement {
                            crate::StatementType::ClassDef(c) if c.name == class_name2 => {
                                Some(c.clone())
                            }
                            _ => None,
                        })
                    }),
                })
            {
                let field_ty = c.body.iter().find_map(|s| match &s.statement {
                    crate::StatementType::AnnotatedName { name, annotation } if name == &a.attr => {
                        crate::python_annotation_to_rust_type(annotation)
                            .or_else(|| {
                                crate::resolve_alias_typeinfo(annotation, symbols, options)
                                    .map(|t| t.to_rust_type())
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
                            .and_then(crate::python_annotation_to_rust_type)
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
                return Some(quote!(stdpython::PyValue));
            }
            // A SELF-FIELD chain (`self._options.custom_functions` —
            // jmespath's TreeInterpreter): the member of a class-typed
            // self-field is a boxed PyValue (the member's type is not
            // statically known at this depth).
            if let ExprType::Attribute(inner) = a.value.as_ref()
                && matches!(inner.value.as_ref(), ExprType::Name(n) if n.id == "self")
            {
                return Some(quote!(stdpython::PyValue));
            }
            // An attribute chain rooted in a PyValue-typed local
            // (`self._resource.meta.client` where `resource` is an
            // unannotated param — boto3's BaseDocumenter): the member is a
            // boxed PyValue (external-object divergence).
            if let Some(recv) = crate::root_name(&a.value)
                && name_types
                    .get(recv)
                    .is_some_and(|t| t.to_string() == "stdpython :: PyValue")
            {
                return Some(quote!(stdpython::PyValue));
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
                Some(quote!(i64))
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
                    return Some(quote!(i64));
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
                return Some(quote!(stdpython::PyValue));
            }
            // A subscript READ on a PyValue-typed local or parameter
            // (`self._base_default_config = default_config_data['base']` —
            // botocore's DefaultConfigResolver, where default_config_data
            // is an unannotated __init__ param): the member of a boxed
            // value is a boxed value.
            if let ExprType::Name(dict) = s.value.as_ref()
                && name_types
                    .get(&dict.id)
                    .is_some_and(|t| t.to_string().contains("PyValue"))
            {
                return Some(quote!(stdpython::PyValue));
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
                return Some(quote!(stdpython::PyValue));
            }
            // A subscript READ on any other local (`self.capacity =
            // ident[4]` where `ident = self._read("16B")` — pip's
            // ELFFile, a bytes buffer whose local type analysis does not
            // reach the try-block assignment): the member of an unknown
            // local — a boxed PyValue (unknown-local divergence).
            if let ExprType::Name(_) = s.value.as_ref() {
                return Some(quote!(stdpython::PyValue));
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
                    return Some(quote!(Vec<String>));
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
                Some(quote!(std::collections::HashSet<String>))
            } else {
                Some(quote!(stdpython::PyValue))
            }
        }
        // An EMPTY dict/list store (`self._method_cache = {}` — jmespath's
        // Visitor, later `.get`/`[k] = v`): a boxed PyDict<String, PyValue>
        // / Vec<PyValue> (the element types are unknowable at the store).
        ExprType::Dict(d) if d.keys.is_empty() => {
            Some(quote!(PyDict<String, stdpython::PyValue>))
        }
        // A NON-EMPTY dict literal (`self._context = {'special_shape_types':
        // {}}` — botocore's ShapeDocumenter): a boxed PyDict<String,
        // PyValue> (the element types are not resolved at field-inference
        // depth; the boxed-dict divergence).
        ExprType::Dict(_) => Some(quote!(PyDict<String, stdpython::PyValue>)),
        // A dict COMPREHENSION (`{tag: idx for idx, tag in ...}` — pip's
        // CandidateEvaluator): a boxed PyDict<String, PyValue>.
        ExprType::DictComp(_) => Some(quote!(PyDict<String, stdpython::PyValue>)),
        // A tuple of string literals (`self._previous_requirement_header =
        // ("", "")` — pip's RequirementPreparer): a Vec<String> (the
        // all-str-tuple rule). A heterogeneous tuple boxes as PyValue.
        ExprType::Tuple(t) => {
            if t.elts.iter().all(|e| {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            }) {
                Some(quote!(Vec<String>))
            } else {
                Some(quote!(stdpython::PyValue))
            }
        }
        ExprType::List(l) if l.is_empty() => Some(quote!(Vec<stdpython::PyValue>)),
        // A list of string literals (`self.sections = ['title', 'client',
        // ...]` — boto3's ServiceDocumenter): Vec<String>.
        ExprType::List(l)
            if !l.is_empty()
                && l.iter().all(|e| {
                    matches!(e, ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_))))
                }) =>
        {
            Some(quote!(Vec<String>))
        }
        // A NON-EMPTY list whose elements all infer to one concrete type
        // (`self._visited_profiles = [self._profile_name]` — botocore's
        // AssumeRoleProvider): Vec<that type>. An unresolvable element
        // (a PyValue-typed self-field read) boxes the Vec as PyValue
        // elements (the empty-list divergence).
        ExprType::List(l) if !l.is_empty() => {
            let mut elt_ty: Option<proc_macro2::TokenStream> = None;
            let mut unknown = false;
            for e in l {
                let t = crate::infer_type(e, options, symbols);
                if matches!(t, crate::TypeInfo::PyObject) {
                    unknown = true;
                    break;
                }
                let r = t.to_rust_type();
                match &elt_ty {
                    None => elt_ty = Some(r),
                    Some(prev) if prev.to_string() == r.to_string() => {}
                    _ => {
                        unknown = true;
                        break;
                    }
                }
            }
            if unknown {
                Some(quote!(Vec<stdpython::PyValue>))
            } else {
                elt_ty.map(|t| quote!(Vec<#t>))
            }
        }
        other => match crate::simple_expr_type(other) {
            // String literals are owned in fields; the store side converts
            // (see Assign).
            Some(ty) if ty.to_string() == "& 'static str" => Some(quote!(String)),
            other => other,
        },
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
fn vec_element_type(t: &TokenStream) -> Option<TokenStream> {
    let s = t.to_string();
    let inner = s.strip_prefix("Vec <")?.strip_suffix('>')?;
    // `u8`, `String`, `stdpython :: PyValue` — re-quote via the token
    // parser (the TokenStream Display inserts spaces around punctuation).
    inner.replace(" :: ", "::").parse::<TokenStream>().ok()
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
            matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
        }
        // The CALL form of a NamedTuple base (`typing.NamedTuple("Url",
        // [("scheme", T), ...])` — urllib3's Url) is also field metadata.
        ExprType::Call(c) => match c.func.as_ref() {
            ExprType::Attribute(a) => {
                a.attr == "NamedTuple"
                    && matches!(a.value.as_ref(), ExprType::Name(n) if n.id == "typing")
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
        // A builtin-container base (`set[tuple[str, str]]` — urllib3's
        // HTTPHeaderDictItemView) is also metadata: the class lowers as a
        // plain struct, not a container subclass.
        ExprType::Subscript(sub) => match sub.value.as_ref() {
            ExprType::Name(n)
                if matches!(n.id.as_str(), "set" | "frozenset" | "list" | "dict" | "tuple") =>
            {
                true
            }
            other => is_typing_base(other),
        },
        _ => false,
    }
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
