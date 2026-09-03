use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::{extraction_failure, 
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableNode,
    SymbolTableScopes,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Assign {
    pub targets: Vec<ExprType>,
    pub value: ExprType,
    pub type_comment: Option<String>,
    /// The annotation of an annotated assignment (`x: list[float] = []`),
    /// carried so type analysis can honor it for empty-container pinning
    /// (the error message at the empty literal suggests exactly this form —
    /// the annotation must not be discarded, Devin review on #103).
    pub annotation: Option<ExprType>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Assign {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let targets: Vec<ExprType> = ob
            .getattr("targets")
            .map_err(|e| extraction_failure("assignment targets", &ob, e))?
            .extract()
            .map_err(|e| extraction_failure("assignment targets", &ob, e))?;

        let python_value = ob.getattr("value").map_err(|e| extraction_failure("value", &ob, e))?;

        let value = python_value.extract().map_err(|e| extraction_failure("python_value", &ob, e))?;

        Ok(Assign {
            targets: targets,
            value: value,
            type_comment: None,
            annotation: None,
        })
    }
}

/// The builtin SCALAR type names an alias assignment can declare
/// (`builtin_str = str` — requests' compat). ONE predicate: five sites
/// (the module-init skips, the static-promotion skip, and the two
/// `pub type` emissions) previously each spelled the list, and the
/// Rust-type mapping existed twice — a partial edit would have emitted a
/// `pub type` AND promoted a static of the same name (E0428), or skipped
/// the store without ever declaring the alias (E0412).
pub(crate) fn is_builtin_scalar_name(name: &str) -> bool {
    // The scalar subset of the builtin CLASS names: the ones whose alias
    // assignment maps to a Rust scalar type. Derived from
    // `is_builtin_class_name` so the two name sets cannot drift.
    is_builtin_class_name(name)
        && matches!(name, "str" | "bytes" | "bytearray" | "int" | "float" | "bool")
}

/// The builtin CLASS names a value position can reference as the class
/// OBJECT (`basestring = (str, bytes)`, `{bytes: ..., str: ...}` —
/// requests' compat/_internal_utils; isinstance targets). Under the
/// class-as-value model a class object's runtime value is its NAME
/// STRING (round 33), and the builtin classes are class objects too —
/// `str`/`bytes`/... in a tuple, dict key, or argument lower to
/// `"str".to_string()` / `"bytes".to_string()`. ONE list for the value
/// renderer (name.rs), the from-import self-alias handling (import.rs),
/// and the builtin-call dispatch (call.rs) — a partial edit would leave
/// a bare builtin name in generated code (E0425) or shadow a user
/// binding.
pub(crate) fn is_builtin_class_name(name: &str) -> bool {
    matches!(
        name,
        "str" | "bytes" | "bytearray" | "int" | "float" | "bool" | "list" | "dict"
            | "tuple" | "set" | "frozenset" | "object" | "type"
    )
}

/// When `value` is a builtin-scalar type NAME, the Rust type its alias
/// declaration maps to; None otherwise.
pub(crate) fn builtin_scalar_alias_type(value: &ExprType) -> Option<TokenStream> {
    let ExprType::Name(n) = value else {
        return None;
    };
    Some(match n.id.as_str() {
        "str" => quote!(String),
        "bytes" | "bytearray" => quote!(Vec<u8>),
        "int" => quote!(i64),
        "float" => quote!(f64),
        "bool" => quote!(bool),
        _ => return None,
    })
}

impl<'a> CodeGen for Assign {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        // Issue #127: `name = lru_cache(maxsize=N)(fn)` — the decorator
        // FACTORY applied as an expression. Register `name` as the
        // synthesized cached function so later `name(...)` call sites
        // resolve against its (fn's) signature. (The codegen emits the
        // synthesized function instead of the assignment; see to_rust.)
        // find_symbols has no options, so the cross-module resolution
        // (issue #123) can only happen when the factory's fn is defined in
        // this module; the codegen pass has options and covers the rest.
        if let Some(synth) =
            crate::try_lru_cache_factory(&self, None, &symbols)
        {
            symbols.insert(
                synth.name.clone(),
                SymbolTableNode::FunctionDef(synth),
            );
            return symbols;
        }
        let mut position = 0;
        for target in self.targets {
            // Only add symbols for Name assignments, not for Attribute assignments
            if let ExprType::Name(name) = target {
                // A rust.bind(...) / rust.c_bind(...) declaration: record the
                // parsed binding so call sites can lower to direct calls.
                // Parse failures surface later, in to_rust, which re-parses
                // and reports them loudly.
                if let ExprType::Call(call) = &self.value {
                    if crate::is_rust_bind_call(&self.value) {
                        if let Ok(spec) = crate::parse_rust_bind(call) {
                            symbols.insert(name.id, SymbolTableNode::RustBinding(spec));
                        }
                        continue;
                    }
                }
                // A LATER rebinding of a class name (urllib3's
                // `if not ssl: HTTPSConnection = DummyConnection` fallback)
                // must not overwrite the ClassDef symbol: the class is the
                // compile-time type, and its own methods resolve `super()`
                // through this symbol. The runtime store is a no-op (see
                // to_rust) — a documented classes-as-values divergence.
                if matches!(symbols.get(&name.id), Some(SymbolTableNode::ClassDef(_))) {
                    continue;
                }
                // A module-level stdlib EXCEPTION ALIAS through the dotted
                // module attribute (`BaseSSLError = ssl.SSLError` —
                // urllib3): register the target as an Alias of the
                // canonical exception name so raise/except guards
                // canonicalize; the store itself emits nothing (classes
                // cannot be runtime values — documented divergence).
                if let ExprType::Attribute(attr) = &self.value
                    && let ExprType::Name(m) = attr.value.as_ref()
                    && let Some(canonical) =
                        crate::ast::tree::raise_stmt::stdlib_exception_canonical(
                            &m.id, &attr.attr,
                        )
                {
                    symbols.insert(name.id, SymbolTableNode::Alias(canonical.to_string()));
                    continue;
                }
                // A module-level CLASS ALIAS (`VerifiedHTTPSConnection =
                // HTTPSConnection`) registers the name as an Alias so
                // construction/isinstance/type resolution follows it; the
                // assignment itself emits nothing (classes cannot be
                // runtime values — documented divergence).
                if let ExprType::Name(n) = &self.value
                    && matches!(
                        symbols.get(&n.id),
                        Some(SymbolTableNode::ClassDef(_))
                            | Some(SymbolTableNode::Alias(_))
                            | Some(SymbolTableNode::ImportFrom(_))
                    )
                {
                    symbols.insert(name.id, SymbolTableNode::Alias(n.id.clone()));
                    continue;
                }
                symbols.insert(
                    name.id,
                    SymbolTableNode::Assign {
                        position: position,
                        value: self.value.clone(),
                    },
                );
            }
            // Could also handle other target types here if needed
            position += 1;
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // rust.bind / rust.c_bind declarations are compile-time-only: the
        // assignment emits nothing (the binding lives in the symbol table).
        // Everything about the declaration is validated here, loudly.
        if crate::is_rust_bind_call(&self.value) {
            if !matches!(ctx, CodeGenContext::Module(_)) {
                return Err("rust.bind declarations must be at module level, not inside \
                            a function or class"
                    .to_string()
                    .into());
            }
            if self.targets.len() != 1 || !matches!(self.targets[0], ExprType::Name(_)) {
                return Err("rust.bind must be assigned to exactly one name"
                    .to_string()
                    .into());
            }
            if let ExprType::Call(call) = &self.value {
                crate::parse_rust_bind(call)?;
            }
            return Ok(TokenStream::new());
        }

        // A module-level TYPE ALIAS: `builtin_str = str` / `bytes = bytes`
        // (requests' compat) binds a name to a builtin TYPE. rython cannot
        // hold a type as a value (the classes-as-values divergence), but
        // the assignment is consumed by isinstance resolution at conversion
        // time. Emit a `pub type` alias so re-exports (`from .compat import
        // builtin_str`) resolve to a real item, mapping the builtin name to
        // its Rust type.
        if matches!(ctx, CodeGenContext::Module(_))
            && self.targets.len() == 1
            && let ExprType::Name(target) = &self.targets[0]
            && let Some(ty) = builtin_scalar_alias_type(&self.value)
        {
            let ident = crate::safe_ident(&target.id);
            return Ok(quote! {
                #[allow(dead_code)]
                pub type #ident = #ty;
            });
        }

        // Issue #127: `name = lru_cache(maxsize=N)(fn)` — the decorator
        // factory applied as an expression. Emit the synthesized cached
        // function (the same @lru_cache machinery a decorated definition
        // gets) instead of storing a callable value, which the value model
        // cannot represent. find_symbols registered `name` as the function,
        // so call sites already resolve; the assignment itself is consumed.
        if let Some(synth) =
            crate::try_lru_cache_factory(&self, Some(&options), &symbols)
        {
            return synth.to_rust(ctx, options, symbols);
        }

        // A store whose target is a CLASS NAME (urllib3's
        // `if not ssl: HTTPSConnection = DummyConnection` fallback) or a
        // CLASS ALIAS (`VerifiedHTTPSConnection = HTTPSConnection`) is a
        // classes-as-values divergence: classes are compile-time types, not
        // runtime values, so the rebinding has no Rust analogue. Emit
        // nothing — loudly, never silently (the -W channel carries it).
        if self.targets.len() == 1
            && let ExprType::Name(target) = &self.targets[0]
            && matches!(
                symbols.get(&target.id),
                Some(SymbolTableNode::ClassDef(_)) | Some(SymbolTableNode::Alias(_))
            )
        {
            options.definition_warnings.borrow_mut().push(format!(
                "assignment to class name `{}` is dropped (classes cannot be \
                 runtime values in rython)",
                target.id
            ));
            return Ok(TokenStream::new());
        }
        // A CLASS NAME stored as a VALUE (`HTTPConnectionPool.ConnectionCls
        // = EmscriptenHTTPConnection` — urllib3's emscripten inject, or
        // `X.attr = SomeClass`): classes cannot be runtime values (the
        // classes-as-values divergence) — the store drops.
        if crate::is_class_value_expr(&self.value, &symbols) {
            options.definition_warnings.borrow_mut().push(format!(
                "assignment of a class (`{:?}`) as a value is dropped (classes \
                 cannot be runtime values in rython)",
                self.value
            ));
            return Ok(TokenStream::new());
        }

        let value_is_none_early = crate::is_none_expr(&self.value);
        let value_yields_option =
            crate::expr_yields_option_ctx(&self.value, &ctx, &options, &symbols);
        let value_expr = self.value.clone();
        // Issue #121: a dict literal stored into a `dict[str, Any]` name
        // (whose value type is the boxed PyValue) forces the literal's
        // value type so mixed values wrap in PyValue::from per element.
        let mut value_options = options.clone();
        if let [ExprType::Name(name)] = self.targets.as_slice()
            && let Some(crate::TypeInfo::Dict(k, v)) = options.name_types.get(&name.id)
        {
            // Issue #121: a dict literal stored into a `dict[K, V]` name
            // forces the literal's element types — mixed values wrap in
            // PyValue::from, and an all-spread literal (`{**a, **b}`)
            // takes its key/value types from the annotation.
            if matches!(**v, crate::TypeInfo::PyValue)
                || matches!(&value_expr, ExprType::Dict(d) if d.keys.iter().any(|k| k.is_none()))
            {
                value_options.dict_forced_kv =
                    std::rc::Rc::new(Some(((**k).clone(), (**v).clone())));
            }
        }
        let mut value = self
            .value
            .clone()
            .to_rust(ctx.clone(), value_options, symbols.clone())?;

        // `x = []` / `x = {}` with a pinned element type (from a later
        // append/insert/indexed-store/use, or a later typed assignment):
        // render the empty literal with explicit types so the store does
        // not poison the binding with an unconstrained Vec<_>/PyDict<_, _>
        // that rustc then rejects at the first use. An empty literal that
        // was never pinned is a loud conversion-time error (issue #77)
        // rather than a cryptic rustc mismatch inside generated code.
        if let [ExprType::Name(name)] = self.targets.as_slice() {
            let empty_literal = match &value_expr {
                ExprType::List(l) if l.is_empty() => true,
                ExprType::Dict(d) if d.keys.is_empty() => true,
                _ => false,
            };
            if empty_literal {
                let is_empty_dict = matches!(&value_expr, ExprType::Dict(d) if d.keys.is_empty());
                // name_types carries the FINAL per-name type: later typed
                // assignments and pinning uses both refine it, so an empty
                // store is rendered against the binding's final type.
                // An Optional name (`xs: list[str] | None = None`, later
                // `xs = []` on the None path — charset_normalizer's
                // `cp_isolation`) renders the INNER typed empty container;
                // the optional-store wrap (assigning into an optional slot)
                // adds the Some.
                let pinned = options.name_types.get(&name.id).cloned();
                let pinned = match &pinned {
                    Some(crate::TypeInfo::Option(inner)) => Some((**inner).clone()),
                    other => other.clone(),
                };
                match pinned {
                    // A PyValue-typed name (`data: _t.DataType` storing
                    // `{}`): a generic empty dict — the store wraps it.
                    Some(crate::TypeInfo::PyValue) if is_empty_dict => {
                        value = quote!(PyDict::<String, PyValue>::from([]));
                    }
                    Some(crate::TypeInfo::Vec(inner)) if !matches!(*inner, crate::TypeInfo::PyObject) => {
                        let t = inner.to_rust_type();
                        value = quote!(Vec::<#t>::new());
                    }
                    Some(crate::TypeInfo::Dict(k, v))
                        if !matches!(*k, crate::TypeInfo::PyObject)
                            && !matches!(*v, crate::TypeInfo::PyObject) =>
                    {
                        let k = k.to_rust_type();
                        let v = v.to_rust_type();
                        value = quote!(PyDict::<#k, #v>::from([]));
                    }
                    // A dict pinned with a KNOWN key but an UNKNOWN value
                    // (`modeled_actions[modeled_action.name] = ...` where
                    // the value is a foreign-class object — boto3's
                    // document_actions): the value boxes into PyValue.
                    Some(crate::TypeInfo::Dict(k, _))
                        if !matches!(*k, crate::TypeInfo::PyObject) =>
                    {
                        let k = k.to_rust_type();
                        value = quote!(PyDict::<#k, stdpython::PyValue>::from([]));
                    }
                    pinned => {
                        // An empty container with an UNRESOLVABLE or absent
                        // element type (an unannotated param/local, a
                        // module-level value flowing across functions, or a
                        // PyObject-pinned local that is later reassigned
                        // from a foreign value — boto3's
                        // underlying_operation_members): the boxed
                        // heterogeneous container (documented divergence).
                        let unresolved = !options
                            .name_types
                            .contains_key(&name.id)
                            && !options.param_type_vars.contains_key(&name.id)
                            || matches!(&pinned, Some(crate::TypeInfo::PyObject))
                            // A name pinned to a SCALAR while storing an
                            // empty container (`empty_value = []` and later
                            // `empty_value = ''` — botocore's paginate):
                            // genuinely mixed — the boxed container
                            // (documented divergence).
                            || matches!(&pinned, Some(t)
                                if !matches!(
                                    t,
                                    crate::TypeInfo::Vec(_)
                                        | crate::TypeInfo::Dict(_, _)
                                        | crate::TypeInfo::PyObject
                                        | crate::TypeInfo::Option(_)
                                ))
                            || matches!(&pinned, Some(crate::TypeInfo::Vec(inner))
                                if matches!(**inner, crate::TypeInfo::PyObject))
                            || matches!(&pinned, Some(crate::TypeInfo::Dict(k, _))
                                if matches!(**k, crate::TypeInfo::PyObject));
                        if unresolved {
                            options.definition_warnings.borrow_mut().push(format!(
                                "`{} = []`/`{{}}` has no inferable element type; \
                                 lowering as Vec<PyValue>/PyDict<String, PyValue> \
                                 (documented divergence)",
                                name.id
                            ));
                            value = if is_empty_dict {
                                quote!(PyDict::<String, PyValue>::from([]))
                            } else {
                                quote!(Vec::<stdpython::PyValue>::new())
                            };
                        } else {
                            return Err(format!(
                                "empty container literal has no inferable element type; \
                                 annotate the variable (e.g. `{}: list[float] = []` or \
                                 `{}: dict[str, int] = {{}}`) or add a use that pins \
                                 the type (`{}.append(...)`, `{}[k] = v`)",
                                name.id, name.id, name.id, name.id
                            )
                            .into());
                        }
                    }
                }
            }
        }

        // Render one assignment for a single target. Python variables are
        // function-scoped, so name targets are declared once (hoisted to a
        // `let mut` at the top of the enclosing function/module scope by the
        // scope's code generator) and every assignment is a plain store —
        // emitting `let mut` per assignment would create a fresh shadowing
        // binding inside nested blocks, silently dropping the store.
        // A name that holds an Option (assigned None on some path) wraps
        // its non-None stores in Some, so both arms unify to Option<T> —
        // unless the value is already an Option (dict.get, another optional
        // name, an Optional-returning call), which stores through unchanged.
        let value_is_none = value_is_none_early;
        // A string literal stored into an attribute becomes an owned String:
        // struct fields hold String (Python strings are owned values), while
        // the literal itself is a &'static str.
        let value_is_str_literal = matches!(
            &value_expr,
            ExprType::Constant(c) if matches!(&c.0, Some(litrs::Literal::String(_)))
        );
        // A None store into a PyValue-typed FIELD (`self.current_buffer =
        // None` — urllib3's emscripten fetch, later filled with a JS value):
        // the boxed value absorbs None (`PyValue::None_`). Only wraps when
        // the attribute's class field type is PyValue; Option-typed fields
        // keep plain `None`.
        let attr_field_is_pyvalue = |target: &ExprType| -> bool {
            let ExprType::Attribute(attr) = target else {
                return false;
            };
            let Some((class, class_symbols)) =
                crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            else {
                return false;
            };
            // The OWNER class's field type (the struct's ground truth):
            // HTTPSConnection's `is_verified` is owned by base
            // HTTPConnection (bool), and the derived class's own
            // infer_fields must not report it as a boxed PyValue (its
            // methods store the boxed member read into the inherited
            // field). The BASE-MOST class in the chain that assigns the
            // field is where the struct declares it.
            let chain = class.base_chain_with_options(&class_symbols, &options);
            let owner = chain
                .iter()
                .rev()
                .find(|c| c.owns_field(&attr.attr))
                .cloned()
                .unwrap_or_else(|| class.clone());
            let owner_symbols = if owner.name == class.name {
                class_symbols
            } else {
                // The owner's defining module's symbols, when it is a
                // same-crate class (the base chain resolves it).
                match symbols.get(&owner.name) {
                    Some(crate::SymbolTableNode::ClassDef(_)) => symbols.clone(),
                    _ => class_symbols,
                }
            };
            owner
                .infer_fields(&owner_symbols, &options)
                .ok()
                .and_then(|fields| {
                    fields
                        .iter()
                        .find(|(name, _)| name == &attr.attr)
                        .map(|(_, ty)| matches!(ty, crate::TypeInfo::PyValue))
                })
                .unwrap_or(false)
        };
        // A store into an OPTION-typed FIELD (`self.chunk_left =
        // self.chunk_left - amt`, `self._start_connect =
        // time.monotonic()` — urllib3): Python's `int | None` slot absorbs
        // any int/float; the Rust slot is Option<T>, so a non-None,
        // non-Option value wraps in Some (mirroring the optional_names
        // name-store rule above). A value that already yields an Option
        // (another optional field, dict.get, an Optional-returning call)
        // stores through unchanged — wrapping again would nest.
        let boxed_self_ref_field = |target: &ExprType| -> bool {
            let ExprType::Attribute(attr) = target else {
                return false;
            };
            let Some(cls) = ctx.enclosing_class_name() else {
                return false;
            };
            let ExprType::Name(recv) = attr.value.as_ref() else {
                return false;
            };
            if recv.id != "self" {
                return false;
            }
            matches!(
                crate::infer_type(Some(&ctx), target, &options, &symbols),
                crate::TypeInfo::Option(inner)
                    if matches!(*inner, crate::TypeInfo::Class(ref c) if c.as_str() == cls)
            )
        };
        let attr_field_is_option = |target: &ExprType| -> bool {
            let ExprType::Attribute(attr) = target else {
                return false;
            };
            let Some((class, class_symbols)) =
                crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            else {
                return false;
            };
            // The BASE-MOST owner's field type (the struct's ground
            // truth), like the concrete-type helper: a derived class's own
            // infer_fields may report a boxed PyValue where the inherited
            // field is `Option<T>`.
            let chain = class.base_chain_with_options(&class_symbols, &options);
            let owner = chain
                .iter()
                .rev()
                .find(|c| c.owns_field(&attr.attr))
                .cloned()
                .unwrap_or_else(|| class.clone());
            let owner_symbols = if owner.name == class.name {
                class_symbols
            } else {
                symbols.clone()
            };
            owner
                .infer_fields(&owner_symbols, &options)
                .ok()
                .and_then(|fields| {
                    fields
                        .iter()
                        .find(|(name, _)| name == &attr.attr)
                        .map(|(_, ty)| matches!(ty, crate::TypeInfo::Option(_)))
                })
                .unwrap_or(false)
        };
        // Round 81 (the generics directive): the CONCRETE member type of a
        // field, when the field's class-table type is a boxable concrete
        // member (Int/Float/Bool/String/Bytes) rather than the boxed
        // PyValue or an Option. A store of a boxed value into such a field
        // converts via `.into()`; anything else (unknown field type,
        // unboxable container) returns None so the plain store keeps its
        // loud rustc error.
        //
        // The type comes from the class that OWNS the field (`field_class`
        // walks the base chain): HTTPSConnection's `is_verified` is owned
        // by base HTTPConnection (bool), while the DERIVED class's own
        // `infer_fields` may report PyValue (a boxed member stored into
        // it in a derived method). The owner's type is the struct's
        // ground truth.
        let attr_field_concrete_type = |target: &ExprType| -> Option<crate::TypeInfo> {
            let ExprType::Attribute(attr) = target else {
                return None;
            };
            let Some((class, class_symbols)) =
                crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            else {
                return None;
            };
            // The OWNER class: the BASE-MOST class in the chain that
            // assigns the field (its struct declares it) — HTTPSConnection
            // inherits `is_verified` from HTTPConnection, and the derived
            // class's own infer_fields may report a boxed PyValue where
            // the struct says bool. Fall back to the receiver class when
            // the field is not inherited.
            let chain = class.base_chain_with_options(&class_symbols, &options);
            let owner_class = chain
                .iter()
                .rev()
                .find(|c| c.owns_field(&attr.attr))
                .cloned()
                .unwrap_or_else(|| class.clone());
            let owner_symbols = if owner_class.name == class.name {
                class_symbols
            } else {
                symbols.clone()
            };
            owner_class
                .infer_fields(&owner_symbols, &options)
                .ok()
                .and_then(|fields| {
                    fields
                        .iter()
                        .find(|(name, _)| name == &attr.attr)
                        .map(|(_, ty)| ty.clone())
                })
                .filter(|t| {
                    matches!(
                        t,
                        crate::TypeInfo::Int
                            | crate::TypeInfo::Float
                            | crate::TypeInfo::Bool
                            | crate::TypeInfo::String
                            | crate::TypeInfo::Bytes
                    )
                })
        };
        // A container of string literals stored into a field whose inferred
        // type is an OWNED String container (issue #229: `self.items =
        // ["kept"]` — the field is Vec<String> but the literal renders
        // `vec!["kept"]`, a Vec<&str>). The field-inference rules type a
        // list/set of all-string-literal elements as Vec<String> /
        // HashSet<String>; the store side owns the elements, mirroring the
        // scalar string rule above.
        let attr_field_string_container = |target: &ExprType| -> Option<bool> {
            let ExprType::Attribute(attr) = target else {
                return None;
            };
            let Some((class, class_symbols)) =
                crate::receiver_class(&attr.value, &ctx, &symbols, &options)
            else {
                return None;
            };
            class
                .infer_fields(&class_symbols, &options)
                .ok()
                .and_then(|fields| {
                    fields
                        .iter()
                        .find(|(name, _)| name == &attr.attr)
                        .and_then(|(_, ty)| match ty {
                            crate::TypeInfo::Vec(inner)
                                if matches!(**inner, crate::TypeInfo::String) =>
                            {
                                Some(false)
                            }
                            crate::TypeInfo::HashSet(inner)
                                if matches!(**inner, crate::TypeInfo::String) =>
                            {
                                Some(true)
                            }
                            _ => None,
                        })
                })
        };
        // Render a list/set literal whose elements are ALL string constants
        // with each element owned (a set literal keeps its HashSet::from
        // shape — a HashSet<String>), or None when the value is not that
        // shape. Tuples are deliberately NOT owned here: the field rules
        // model an all-str tuple as Vec<String>, and a Vec displays as a
        // Python list — `('a', 'b')` would silently become `['a', 'b']`.
        // The tuple store stays its pre-existing loud rustc error.
        let owned_str_container_value = |expr: &ExprType| -> Option<TokenStream> {
            fn str_const(e: &ExprType) -> bool {
                matches!(e, ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::String(_))))
            }
            let (elts, is_set) = match expr {
                ExprType::List(l) => (l.as_slice(), false),
                ExprType::Set(s) => (s.elts.as_slice(), true),
                _ => return None,
            };
            if elts.is_empty() || !elts.iter().all(str_const) {
                return None;
            }
            let owned: Result<Vec<_>, _> = elts
                .iter()
                .map(|e| {
                    e.clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                        .map(|tok| quote!((#tok).to_string()))
                })
                .collect();
            let owned = owned.ok()?;
            if is_set {
                Some(quote!(std::collections::HashSet::from([#(#owned),*])))
            } else {
                Some(quote!(vec![#(#owned),*]))
            }
        };
        let render_one = |target: &ExprType,
                          value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            // Issue #115: a store to a `global`-declared name whose module
            // binding is a MUTABLE static writes THROUGH the static —
            // py_global_write(&name, v) — instead of binding a local. Only
            // scopes that own the binding take this path: module scope, or
            // a function declaring the name `global` (scope_global_writables);
            // a plain assignment elsewhere stays a local, as in Python.
            if let ExprType::Name(name) = target
                && options.scope_global_writables.contains(&name.id)
                && let Some(kind) = options.mutable_statics.get(&name.id)
            {
                let ident = crate::safe_ident(&name.id);
                let stored = if let crate::MutableGlobalKind::Class { class } = kind {
                    // Issue #189: the class-instance global holds exactly
                    // None and the detected class construction. The call's
                    // `?` propagates from the enclosing scope's Result.
                    if value_is_none_early {
                        quote!(None)
                    } else if matches!(
                        &value_expr,
                        ExprType::Call(c)
                            if matches!(
                                c.func.as_ref(),
                                ExprType::Name(f) if f.id == *class
                            )
                    ) {
                        quote!(Some(#value))
                    } else {
                        return Err(format!(
                            "`global {}` stores a value that is neither `None` nor \
                             a `{}` instance; the class-instance module global \
                             holds exactly that (issue #189)",
                            name.id, class
                        )
                        .into());
                    }
                } else if kind.boxed() {
                    if value_is_none_early {
                        quote!(stdpython::PyValue::None_)
                    } else if crate::expr_yields_pyvalue(&value_expr, &options, &symbols) {
                        quote!(#value)
                    } else if matches!(
                        &value_expr,
                        ExprType::List(_) | ExprType::Dict(_) | ExprType::Set(_)
                            | ExprType::ListComp(_) | ExprType::DictComp(_) | ExprType::SetComp(_)
                    ) || matches!(
                        &value_expr,
                        ExprType::Call(c)
                            if matches!(
                                c.func.as_ref(),
                                ExprType::Name(f) if matches!(
                                    symbols.get(&f.id),
                                    Some(crate::SymbolTableNode::ClassDef(_))
                                )
                            )
                    ) {
                        // A container or class instance has no PyValue
                        // representation — the store cannot round-trip
                        // through the boxed global (issue #189 scope).
                        // At MODULE scope (init-time control flow — the
                        // emscripten `_fetcher = _StreamingFetcher()`
                        // branch, behind an always-false worker check)
                        // the store warns and stores None instead of
                        // failing the conversion: the read sites stay
                        // compilable and the -W channel carries the
                        // divergence. A `global`-declared FUNCTION store
                        // keeps the loud conversion error.
                        if matches!(ctx, crate::CodeGenContext::Module(_)) {
                            options.definition_warnings.borrow_mut().push(format!(
                                "module-level store of a container or class \
                                 instance into the boxed global `{}` is \
                                 dropped (None is stored): the value has no \
                                 boxed representation (the §12 boxed-global \
                                 divergence)",
                                name.id
                            ));
                            quote!(stdpython::PyValue::None_)
                        } else {
                            return Err(format!(
                                "`global {}` stores a value with no boxed \
                                 representation (a container or class instance); \
                                 a BOXED mutable module global holds scalars, \
                                 strings, and None — rython refuses to silently \
                                 ignore the write (issue #189)",
                                name.id
                            )
                            .into());
                        }
                    } else {
                        quote!(stdpython::PyValue::from(#value))
                    }
                } else if matches!(kind, crate::MutableGlobalKind::Str) && value_is_str_literal {
                    // A String static stores literals owned.
                    quote!((#value).to_string())
                } else {
                    quote!(#value)
                };
                let global_ref = kind.static_ref(&ident);
                return Ok(quote!(stdpython::py_global_write(#global_ref, #stored);));
            }
            // An attribute store target renders in place flavor: in a
            // generic trait default, `self.f = v` must store through the
            // mutable accessor (`*self.f_mut() = v`) rather than the load
            // accessor (which clones).
            // A store into a polymorphic ROOT's slot (hierarchy.rs) —
            // `self.inner = Inner(0)` where Inner has subclasses — takes
            // the sum type: a struct of the subtree converts, the rule
            // `coerce_tokens` applies to an argument or a return.
            let root_slot_value = if let ExprType::Attribute(_) = target
                && let crate::TypeInfo::Class(slot) =
                    crate::infer_type(Some(&ctx), target, &options, &symbols)
                && crate::ast::tree::hierarchy::is_polymorphic_root(&slot)
                && let crate::TypeInfo::Class(from) =
                    crate::infer_type(Some(&ctx), &value_expr, &options, &symbols)
                && (from == slot
                    || crate::ast::tree::class_def::ClassDef::extends_by_name(&from, &slot))
            {
                Some(quote!((#value).into()))
            } else {
                None
            };
            let value = root_slot_value.as_ref().unwrap_or(value);
            let target_code = match target {
                ExprType::Attribute(attr) => crate::ast::tree::attribute::to_rust_place(
                    &attr.value,
                    &attr.attr,
                    &ctx,
                    &options,
                    &symbols,
                    true,
                )?,
                // Destructuring targets are places too: `self.x, self.y = v`
                // in a generic trait default must store through the mutable
                // accessors (`*self.x_mut(), *self.y_mut()`), not through
                // clones of the fields.
                ExprType::Tuple(tuple) => {
                    // A destructuring target containing a SLICE subscript
                    // (`self._left[left:right], self._right[left:right] =
                    // [start], [end]` — pip's lazy_wheel): the slice store
                    // has no rython lowering — the whole assignment drops
                    // (documented divergence).
                    if tuple.elts.iter().any(|e| {
                        matches!(e, ExprType::Subscript(s)
                            if matches!(s.kind, crate::SubscriptKind::Slice { .. }))
                    }) {
                        options.definition_warnings.borrow_mut().push(
                            "a destructuring assignment with a slice target is dropped \
                             (no rython equivalent)"
                                .to_string(),
                        );
                        return Ok(TokenStream::new());
                    }
                    let mut elts = Vec::with_capacity(tuple.elts.len());
                    for elt in &tuple.elts {
                        elts.push(crate::ast::tree::attribute::to_rust_place_expr(
                            elt, &ctx, &options, &symbols, true,
                        )?);
                    }
                    // A single-element target is still a TUPLE (`x, = f()`):
                    // the trailing comma is what makes it one, so emit `(x,)`
                    // — `(x)` would be a parenthesized place and the
                    // destructuring assignment would not type-check against
                    // the one-element tuple value.
                    if elts.len() == 1 {
                        let only = &elts[0];
                        quote!((#only,))
                    } else {
                        quote!((#(#elts),*))
                    }
                }
                _ => {
                    // A plain name target is a STORE into the hoisted
                    // binding — never an unwrap, even when the name is
                    // narrowed (issue #125): the binding stays Option<T>
                    // and the store wraps in Some below. Render with the
                    // narrowed set cleared so Name::to_rust emits the bare
                    // binding.
                    if matches!(target, ExprType::Name(_)) {
                        let mut store_options = options.clone();
                        store_options.narrowed_names =
                            std::rc::Rc::new(std::collections::HashMap::new());
                        target
                            .clone()
                            .to_rust(ctx.clone(), store_options, symbols.clone())?
                    } else {
                        target
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?
                    }
                }
            };
            // Issue #123 family: `self.path = path` followed by a later
            // read of `path` (pip's Prefix stores the parameter, then
            // passes it to get_scheme). Python shares by reference; the
            // Rust move would poison the later use (E0382). Cloning the
            // store is faithful ONLY for values Python cannot mutate
            // (str/bytes) — mutable containers keep the move, so aliasing
            // stays loud (issue #79). The boxed PyValue clones too (its
            // Arc shares dict/tuple members and value-copies the
            // immutable scalars — Python reference semantics, round 80);
            // without the clone a reused boxed name is moved out from
            // under its later reads (`context = ssl_context` then
            // `elif ssl_context is None` — urllib3's
            // _ssl_wrap_socket_and_match_hostname).
            let stored_name_needs_clone = matches!(&value_expr, ExprType::Name(n)
                if options.use_counts.get(&n.id).copied().unwrap_or(0) > 1
                    && matches!(
                        crate::ast::tree::type_ctx::infer_type(None, &value_expr, &options, &symbols),
                        crate::TypeInfo::String
                            | crate::TypeInfo::Bytes
                            | crate::TypeInfo::PyValue
                    ));
            Ok(match target {
                ExprType::Name(name) => {
                    // Issue #121: a name holding a boxed PyValue (wider
                    // union or Any) wraps its stores — `PyValue::from(v)`,
                    // None as `PyValue::None_`. A value that already yields
                    // a PyValue stores through unchanged. A VALUE-PINNED
                    // parameter (`path = os.path.expandvars(path)` — issue
                    // #161) is PyValue in the body the same way.
                    if options
                        .name_types
                        .get(&name.id)
                        .is_some_and(|t| matches!(t, crate::TypeInfo::PyValue))
                        || options.pyvalue_into_params.contains(&name.id)
                    {
                        if value_is_none_early {
                            quote!(#target_code = PyValue::None_;)
                        } else if crate::expr_yields_pyvalue(&value_expr, &options, &symbols) {
                            // A REUSED boxed name stores a CLONE (the
                            // Arc copy — Python reference semantics); the
                            // bare store would move the value out from
                            // under its later reads (`context =
                            // ssl_context` then `elif ssl_context is
                            // None` — urllib3's
                            // _ssl_wrap_socket_and_match_hostname).
                            if stored_name_needs_clone {
                                quote!(#target_code = (#value).clone();)
                            } else {
                                quote!(#target_code = #value;)
                            }
                        } else {
                            quote!(#target_code = PyValue::from(#value);)
                        }
                    } else if !value_is_none
                        && !value_yields_option
                        && (options.optional_names.contains(&name.id)
                            || options
                                .name_types
                                .get(&name.id)
                                .is_some_and(|t| matches!(t, crate::TypeInfo::Option(_))))
                    {
                        // A plain value into an OPTION-typed local
                        // (optional_names, or name_types widened by an
                        // Option-valued store — `server_hostname: str =
                        // self.host` then `server_hostname =
                        // self._tunnel_host`: the Python local became
                        // None-able) wraps in Some. A str LITERAL owns
                        // itself at the wrap (`Some("lit".to_string())` —
                        // the Option<String> slot, never Option<&str>).
                        if value_is_str_literal {
                            quote!(#target_code = Some((#value).to_string());)
                        } else {
                            quote!(#target_code = Some(#value);)
                        }
                    } else if value_is_str_literal
                        && options.owned_str_literals.contains(&name.id)
                    {
                        // Issue #110: a string-literal binding that is later
                        // rebound by a String (`out += "x"`) must be owned
                        // from the start.
                        quote!(#target_code = (#value).to_string();)
                    } else if value_is_str_literal
                        && options
                            .name_types
                            .get(&name.id)
                            .is_some_and(|t| matches!(t, crate::TypeInfo::String))
                    {
                        // A str literal stored into a STRING-typed NAME
                        // (`method = "GET"` where the parameter is `str`
                        // and the prologue bound `let mut method: String =
                        // method.into()` — urllib3's urlopen): the literal
                        // is a &'static str and the binding is owned, so
                        // the store owns it. A name whose type is
                        // StrRef (`&'static str` — a literal-only local)
                        // keeps the bare store.
                        quote!(#target_code = (#value).to_string();)
                    } else if let Some(clone) = self_field_read_clone(&value_expr, &ctx, &options, &symbols) {
                        // Issue #222's local half: `box = self.scheme` —
                        // a non-Copy field read moved into a LOCAL leaves
                        // `&self` (E0507). Python's objects are references,
                        // so the clone reproduces the shared value. Only
                        // immutable field types (str/bytes) clone — mutable
                        // containers keep the move so aliasing stays loud
                        // (issue #79), mirroring stored_name_needs_clone.
                        quote!(#target_code = #clone;)
                    } else if stored_name_needs_clone {
                        // A REUSED immutable/boxed name stored into a NAME
                        // target (`context = ssl_context` then `elif
                        // ssl_context is None` — urllib3's
                        // _ssl_wrap_socket_and_match_hostname): the bare
                        // store moves the value out from under its later
                        // reads (E0382). The clone is the Arc/Python
                        // reference copy — faithful for str/bytes and the
                        // boxed PyValue (issue #123 family, round 81).
                        quote!(#target_code = (#value).clone();)
                    } else {
                        quote!(#target_code = #value;)
                    }
                }
                // Destructuring assignment to the hoisted names. `target_code`
                // is already the parenthesized element list.
                ExprType::Tuple(tuple_target) => {
                    // A tuple TARGET with a tuple-literal VALUE of the same
                    // arity (`(body, content_type) = (urlencode(fields),
                    // "application/x-www-form-urlencoded")` — urllib3's
                    // request): each value element lands in the slot of the
                    // corresponding target NAME, so a str literal into a
                    // String-typed slot owns itself (`content_type` is
                    // String from the `(Vec<u8>, String)` return of
                    // encode_multipart_formdata). Only fires when a slot
                    // pairing actually needs typing; otherwise the plain
                    // whole-tuple store below keeps its exact shape.
                    let needs_slot_typing = matches!(&value_expr, ExprType::Tuple(vt)
                        if vt.elts.len() == tuple_target.elts.len()
                            && tuple_target.elts.iter().zip(vt.elts.iter()).any(
                                |(t, v)| matches!(t, ExprType::Name(n)
                                    if matches!(options.name_types.get(&n.id),
                                        Some(crate::TypeInfo::String))
                                        && matches!(v, ExprType::Constant(c)
                                            if matches!(&c.0, Some(litrs::Literal::String(_))))
                                )
                            ));
                    if needs_slot_typing {
                        let ExprType::Tuple(vt) = &value_expr else {
                            unreachable!("just checked");
                        };
                        let mut rendered = Vec::with_capacity(vt.elts.len());
                        for (t, v) in tuple_target.elts.iter().zip(vt.elts.iter()) {
                            // Only the STR-LITERAL-INTO-STRING-SLOT pairings
                            // re-render typed; every other element keeps the
                            // plain render so unrelated slots (a dropped
                            // call landing as PyValue::None_ into a
                            // StrOrBytes-typed body slot) stay exactly as
                            // before.
                            let is_owned_pair = matches!(t, ExprType::Name(n)
                                if matches!(options.name_types.get(&n.id),
                                    Some(crate::TypeInfo::String))
                                    && matches!(v, ExprType::Constant(c)
                                        if matches!(&c.0, Some(litrs::Literal::String(_)))));
                            if is_owned_pair {
                                rendered.push(crate::render_typed(
                                    v,
                                    ctx.clone(),
                                    options.clone(),
                                    symbols.clone(),
                                    Some(crate::TypeInfo::String),
                                )?);
                            } else {
                                rendered.push(
                                    v.clone()
                                        .to_rust(ctx.clone(), options.clone(), symbols.clone())?,
                                );
                            }
                        }
                        quote!(#target_code = (#(#rendered),*);)
                    } else {
                        quote!(#target_code = #value;)
                    }
                }
                // A None store into a PyValue-typed field (`self.current_buffer
                // = None`) wraps in PyValue::None_ (the boxed value absorbs
                // None); Option-typed fields keep the plain None store.
                ExprType::Attribute(_) if value_is_none_early && attr_field_is_pyvalue(target) => {
                    quote!(#target_code = PyValue::None_;)
                }
                // A non-None value stored into a PyValue-typed FIELD
                // (`self.box: Any = "abc"` — urllib3's Any-annotated
                // fields) wraps in PyValue::from, mirroring the Name-slot
                // rule: the field holds the boxed union, the value is a
                // concrete member. A value that already yields a PyValue
                // stores through unchanged; a container literal or class
                // construction has NO boxed representation (the same
                // exclusions the global-store path makes — those keep
                // their pre-existing loud rustc error).
                ExprType::Attribute(_)
                    if !value_is_none_early
                        && attr_field_is_pyvalue(target)
                        && !crate::expr_yields_pyvalue(&value_expr, &options, &symbols)
                        && !matches!(
                            &value_expr,
                            ExprType::List(_)
                                | ExprType::Dict(_)
                                | ExprType::Set(_)
                                | ExprType::ListComp(_)
                                | ExprType::DictComp(_)
                                | ExprType::SetComp(_)
                                | ExprType::Tuple(_)
                        ) =>
                {
                    quote!(#target_code = PyValue::from(#value);)
                }
                // An Option<String> field (`note: Optional[str] = None`,
                // later `self.note = "exhausted"`) takes the owned literal
                // wrapped in Some — the Option-wrap arm below must not
                // lose the literal's ownership conversion, and this arm
                // must not lose the wrap.
                ExprType::Attribute(_) if value_is_str_literal => {
                    if attr_field_is_option(target) {
                        quote!(#target_code = Some((#value).to_string());)
                    } else {
                        quote!(#target_code = (#value).to_string();)
                    }
                }
                // A list/set of string literals stored into a field typed
                // Vec<String>/HashSet<String> owns each element at the
                // store (issue #229) — the literal renders Vec<&str>
                // otherwise. The cheap literal-shape check runs first so
                // only such stores pay the field-type lookup; a set-shaped
                // field (HashSet<String>) only accepts the set-literal
                // form, the vec fields only accept the list form.
                ExprType::Attribute(_)
                    if let Some(owned) = owned_str_container_value(&value_expr)
                        && matches!(
                            attr_field_string_container(target),
                            Some(is_set_field)
                                if is_set_field == matches!(value_expr, ExprType::Set(_))
                        ) =>
                {
                    quote!(#target_code = #owned;)
                }
                // A non-None, non-Option value stored into an OPTION-typed
                // FIELD wraps in Some (Python's `int | None` slot absorbs a
                // plain int — urllib3's `self.chunk_left = self.chunk_left
                // - amt` and `self._start_connect = time.monotonic()`;
                // charset_normalizer's `self._last_printable_char =
                // character` — the clone arm BELOW must not preempt it).
                // None stores keep plain None; an already-Option value
                // (another optional field, dict.get) stores through
                // unchanged — wrapping again would nest. Only fields whose
                // class-table type is Option qualify (the receiver_class
                // dispatch is conservative by design, so a generic-trait
                // body whose receiver cannot be pinned stays on the plain
                // store and keeps its pre-existing loud error rather than
                // silently changing shape).
                ExprType::Attribute(_)
                    if !value_is_none_early
                        && attr_field_is_option(target)
                        && !value_yields_option =>
                {
                    // A reused name clones INTO the Some (the field owns
                    // the value and the name is read again later —
                    // charset_normalizer's `_last_printable_char =
                    // character`). A SELF-REFERENTIAL boxed field owns its
                    // children through Box (round 99, E0072): the store
                    // wraps in Box::new so the Option<Box<Class>> slot
                    // accepts the plain instance.
                    if boxed_self_ref_field(target) {
                        if stored_name_needs_clone {
                            quote!(#target_code = Some(Box::new((#value).clone()));)
                        } else {
                            quote!(#target_code = Some(Box::new(#value));)
                        }
                    } else if stored_name_needs_clone {
                        quote!(#target_code = Some((#value).clone());)
                    } else {
                        quote!(#target_code = Some(#value);)
                    }
                }
                // A reused name stored into a FIELD clones (the field owns
                // the value; the name is read again later). Ordered AFTER
                // the Option-wrap arm: a String into an Option<String>
                // field with a later reuse must still wrap (`Some(
                // (character).clone())` — charset_normalizer's
                // _count_suspicious).
                ExprType::Attribute(_) if stored_name_needs_clone => {
                    quote!(#target_code = (#value).clone();)
                }
                // Round 81 (the generics directive): a BOXED value stored
                // into a CONCRETE-typed FIELD (`self.is_verified =
                // sock_and_verified.is_verified` — a bool field receiving
                // the boxed namedtuple member) converts via the reverse
                // From<PyValue> impls — `(#value).into()` recovers the
                // member, and a wrong member is a LOUD TypeError panic
                // (Python fails at use, rython at the conversion).
                ExprType::Attribute(_)
                    if attr_field_concrete_type(target).is_some()
                        && matches!(
                            crate::ast::tree::type_ctx::infer_type(None, 
                                &value_expr, &options, &symbols,
                            ),
                            crate::TypeInfo::PyValue | crate::TypeInfo::PyValueMember(_)
                        ) =>
                {
                    quote!(#target_code = (#value).into();)
                }
                _ => quote!(#target_code = #value;),
            })
        };

        // Subscript stores don't go through the Load-position lowering
        // (which reads via py_index): `x[i] = v` follows Python index rules
        // through py_set_index — negatives from the end, catchable
        // IndexError for lists, insert-or-overwrite for dicts.
        let render_subscript_store = |sub: &crate::Subscript,
                                      value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            // The dynamic-import machinery (`locals()[pkg] = ...`,
            // `sys.modules[...] = ...` — requests/packages.py) is a
            // documented divergence: the stores are no-ops (the module
            // aliasing has no rython equivalent).
            let is_dynamic_import_store = match sub.value.as_ref() {
                ExprType::Call(c) => {
                    matches!(c.func.as_ref(), ExprType::Name(n) if n.id == "locals")
                }
                ExprType::Attribute(a) => {
                    a.attr == "modules"
                        && matches!(a.value.as_ref(), ExprType::Name(n)
                            if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Sys))
                }
                _ => false,
            };
            if is_dynamic_import_store {
                return Ok(quote!(()));
            }
            // The receiver must be a PLACE (nested subscripts thread
            // through py_index_mut): the Load lowering would clone, and the
            // store would silently land on the clone.
            let receiver = crate::subscript_receiver_place(
                &sub.value,
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )?;
            // An OPTION-typed receiver (`request_context["blocksize"] =
            // _DEFAULT_BLOCKSIZE` where request_context is `dict | None` —
            // urllib3's poolmanager, guaranteed non-None after the `if x
            // is None:` fill above): CPython raises TypeError on a None
            // receiver, which the loud §12.2 panic mirrors. Mirrors the
            // read side's receiver_option_inner, but a STORE needs `&mut`
            // of the inner, so the unwrap is as_mut — the same shape the
            // call path uses for `x.pop(k)` on an Option binding.
            let receiver = if crate::ast::tree::attribute::receiver_option_inner(
                &sub.value,
                &ctx,
                &symbols,
                &options,
            )
            .is_some()
            {
                quote!((#receiver).as_mut().unwrap_or_else(|| {
                    panic!("TypeError: 'NoneType' object does not support item assignment")
                }))
            } else {
                receiver
            };
            // `os.environ[k] = v` routes through `os::setenv`: os.environ
            // is an IMMUTABLE module static (a live view of the process
            // environment) and cannot be borrowed mutably for py_set_index.
            // The static's set-index impls (Environ: PySetIndex) cover
            // receivers that are real values (`e = os.environ`), but the
            // direct module-path store must go through the function.
            let is_os_environ = matches!(
                sub.value.as_ref(),
                ExprType::Attribute(a)
                    if a.attr == "environ"
                        && matches!(a.value.as_ref(), ExprType::Name(n)
                            if crate::StdModule::from_name(&n.id) == Some(crate::StdModule::Os))
                        && !crate::ast::tree::call::root_name(&a.value)
                            .is_some_and(|root| crate::module_name_shadowed(root, &symbols))
            );
            // Issue #121: a store into a PyDict<K, PyValue> (`dict[str,
            // Any]`) wraps the value (`PyValue::from(v)`, None via `()`).
            // An OPTION-wrapped receiver (`request_context["blocksize"] =
            // _DEFAULT_BLOCKSIZE` where request_context is `dict[str,
            // Any] | None` — urllib3's poolmanager) is the same dict once
            // the store unwraps it (round 63).
            let value = if let ExprType::Name(n) = sub.value.as_ref()
                && (matches!(
                    options.name_types.get(&n.id),
                    Some(crate::TypeInfo::Dict(_, v)) if matches!(**v, crate::TypeInfo::PyValue)
                ) || matches!(
                    options.name_types.get(&n.id),
                    Some(crate::TypeInfo::Option(inner))
                        if matches!(
                            &**inner,
                            crate::TypeInfo::Dict(_, v)
                                if matches!(**v, crate::TypeInfo::PyValue)
                        )
                ))
                && !crate::expr_yields_pyvalue(&value_expr, &options, &symbols)
            {
                boxed_dict_value_wrap(&value, &value_expr, &ctx, &options, &symbols)
            } else {
                value.clone()
            };
            match &sub.kind {
                crate::SubscriptKind::Index(index) => {
                    // The dict takes the value BY OWNED VALUE; Python's dict
                    // holds a reference. A value whose ROOT name is read
                    // again (the key reads a field of the same object —
                    // `self.items[item.name] = item` — the idiom corpus's
                    // add()) must be CLONED, or the key read borrows a
                    // moved value (E0382, round 98). The clone lands on
                    // the RAW value, before any boxing.
                    let value = match crate::type_ctx::reuse_root_name(&value_expr)
                        .and_then(|r| options.use_counts.get(&r).copied())
                    {
                        Some(uses) if uses > 1 => quote!(Clone::clone(&(#value))),
                        _ => value,
                    };
                    // A user-class receiver that defines `__setitem__`
                    // routes the store to ITS method — Python's behavior
                    // (the class's own key semantics and exceptions;
                    // §7's mapping-protocol slice). The method must
                    // exist; anything else keeps py_set_index, loud in
                    // rustc for classes (§12.1). The value passes
                    // through the full argument mapping.
                    if let Some((class, class_symbols)) =
                        crate::receiver_class(&sub.value, &ctx, &symbols, &options)
                        && let Some(method) =
                            class
                                .method_on_mro("__setitem__", &class_symbols)
                                .filter(|m| crate::ast::tree::call::dunder_method_well_typed(m))
                    {
                        let v_expr = value_expr.clone();
                        return crate::ast::tree::call::dunder_method_call(
                            &method,
                            &receiver,
                            &[(**index).clone(), v_expr],
                            true,
                            &ctx,
                            &options,
                            &symbols,
                        );
                    }
                    // String-keyed dicts store `&str` indexes through
                    // py_set_index(String, V), so a &str literal index is
                    // owned at the store site. The receiver's Dict type can
                    // come from a NAME (name_types) or a `self.<field>`
                    // whose field the class table types PyDict<String, V>
                    // (urllib3's `self.conn_kw["proxy"] = self.proxy` —
                    // round 46). An OPTION-wrapped receiver (`headers["k"]
                    // = v` where headers is `Mapping[str, str] | None` —
                    // urllib3's RequestMethods; `request_context["k"] = v`
                    // where request_context is `dict[str, Any] | None` —
                    // poolmanager) is the same dict once the store unwraps
                    // it, so the dict type is read through the Option
                    // (round 63).
                    let receiver_dict = || -> Option<(crate::TypeInfo, crate::TypeInfo)> {
                        let through_option = |t: &crate::TypeInfo| match t {
                            crate::TypeInfo::Dict(k, v) => {
                                Some(((**k).clone(), (**v).clone()))
                            }
                            crate::TypeInfo::Option(inner) => match inner.as_ref() {
                                crate::TypeInfo::Dict(k, v) => {
                                    Some(((**k).clone(), (**v).clone()))
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        match sub.value.as_ref() {
                            ExprType::Name(n) => options
                                .name_types
                                .get(&n.id)
                                .and_then(through_option),
                            // A `self.<field>` receiver: infer_type's
                            // self-field arm resolves the class-table type
                            // (round 99 — replaces the self_field_rust_ty
                            // fallback, Directive 2).
                            ExprType::Attribute(_)
                                if matches!(
                                    sub.value.as_ref(),
                                    ExprType::Attribute(a)
                                        if crate::ast::tree::visit::is_self(a.value.as_ref())
                                ) =>
                            {
                                through_option(&crate::infer_type(
                                    Some(&ctx),
                                    &sub.value,
                                    &options,
                                    &symbols,
                                ))
                            }
                            _ => None,
                        }
                    };
                    let string_keyed = receiver_dict()
                        .is_some_and(|(k, _)| matches!(k, crate::TypeInfo::String));
                    let index = if string_keyed {
                        crate::render_typed(
                            index,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?
                    } else {
                        index
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())?
                    };
                    if is_os_environ {
                        return Ok(quote!(os::setenv(#index, #value);));
                    }
                    // A store into a `PyDict<String, PyValue>` whose value
                    // is NOT already a boxed PyValue (`self.conn_kw["proxy"]
                    // = self.proxy` where proxy is `Option<Url>` — urllib3):
                    // the dict's value type is the boxed value, so the
                    // stored member boxes (round 46). The value-type
                    // signal comes from the same receiver lookup as
                    // string_keyed above. A NAME receiver's wrap already
                    // happened above (issue #121); only the self-field
                    // receiver needs it here. The self-field dict type is
                    // read through an Option (a `str | None`-style field
                    // holds the dict after the store's unwrap).
                    let pyvalue_valued = matches!(
                        sub.value.as_ref(),
                        ExprType::Attribute(attr)
                            if crate::ast::tree::visit::is_self(attr.value.as_ref())
                    ) && receiver_dict().is_some_and(|(k, v)| {
                        matches!(k, crate::TypeInfo::String)
                            && matches!(v, crate::TypeInfo::PyValue)
                    });
                    let value = if pyvalue_valued
                        && !crate::expr_yields_pyvalue(&value_expr, &options, &symbols)
                        && !value_is_none_early
                    {
                        boxed_dict_value_wrap(&value, &value_expr, &ctx, &options, &symbols)
                    } else {
                        value
                    };
                    // A STRING-valued dict stores a str LITERAL by owning it
                    // (`headers["Content-Type"] = "application/json"` —
                    // urllib3's RequestMethods: the dict is PyDict<String,
                    // String>, and the literal is a &'static str). The
                    // index-side owning above covers the key; the value
                    // needs the same treatment. Only the literal shape
                    // (render_typed passes anything else through).
                    let value = if receiver_dict().is_some_and(|(k, v)| {
                        matches!(k, crate::TypeInfo::String)
                            && matches!(v, crate::TypeInfo::String)
                    }) {
                        crate::render_typed(
                            &value_expr,
                            ctx.clone(),
                            options.clone(),
                            symbols.clone(),
                            Some(crate::TypeInfo::String),
                        )?
                    } else {
                        value
                    };
                    Ok(quote!({
                        let __rython_val = #value;
                        (#receiver).py_set_index(#index, __rython_val)?;
                    }))
                }
                crate::SubscriptKind::Slice { lower, upper, step, .. } => {
                    // Slice assignment (`xs[a:b] = [...]`,
                    // `memoryview(byte_obj)[0:n] = sub` — urllib3's
                    // emscripten fetch loop) replaces a range in place: a
                    // different-length RHS inserts or removes elements. The
                    // runtime's py_slice_assign clamps/negativizes bounds
                    // exactly like reads (issue #153).
                    let step_is_one = crate::ast::tree::subscript::is_step_one(
                        step.as_deref(),
                    );
                    let receiver = crate::subscript_receiver_place(
                        sub.value.as_ref(),
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?;
                    let bound_tok = |b: &Option<Box<ExprType>>| -> Result<TokenStream, Box<dyn std::error::Error>> {
                        match b {
                            Some(e) => {
                                let t = e.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                                Ok(quote!(Some(#t)))
                            }
                            None => Ok(quote!(None)),
                        }
                    };
                    let lo_tok = bound_tok(lower)?;
                    let up_tok = bound_tok(upper)?;
                    // step == 1 (explicit or omitted): contiguous splice.
                    // Any other nonzero step: extended replacement - the
                    // runtime checks that the replacement matches the
                    // selected slot count and raises ValueError otherwise
                    // (exactly like CPython's list_ass_subscript).
                    if step_is_one {
                        // A REUSED Name value (`b[:len(temp)] = temp;
                        // return len(temp)` — urllib3's readinto): the
                        // slice-assign MOVES the value into the receiver,
                        // and a later read would use-after-move — clone it
                        // (the same stored_name_needs_clone rule the plain
                        // stores apply, round 79).
                        let value = if matches!(&value_expr, ExprType::Name(n)
                            if options.use_counts.get(&n.id).copied().unwrap_or(0) > 1)
                        {
                            quote!((#value).clone())
                        } else {
                            value.clone()
                        };
                        Ok(quote!({
                            (#receiver).py_slice_assign(#lo_tok, #up_tok, #value);
                        }))
                    } else {
                        let st_tok = step.clone().unwrap().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
                        Ok(quote!({
                            (#receiver).py_slice_assign_step(#lo_tok, #up_tok, #st_tok, #value)?;
                        }))
                    }
                }
            }
        };

        let render = |target: &ExprType,
                      value: &TokenStream|
         -> Result<TokenStream, Box<dyn std::error::Error>> {
            // A store into a CLASS attribute (`HTTPSConnectionPool.
            // ConnectionCls = HTTP2Connection` — urllib3's http2
            // injection) or a MODULE attribute (`urllib3_connection.
            // HTTPSConnection = ...`): rython classes are structs and
            // modules are paths — neither holds mutable class/module
            // state, so the store is dropped (the class/module-attribute
            // mutation divergence).
            if let ExprType::Attribute(attr) = target {
                let recv_name = crate::ast::tree::call::root_name(&attr.value);
                if let Some(root) = &recv_name {
                    let class_receiver = matches!(
                        symbols.get(root),
                        Some(crate::SymbolTableNode::ClassDef(_))
                    ) || crate::resolve_class_referenced(root, &symbols, &options).is_some();
                    let module_receiver =
                        crate::ast::tree::attribute::is_module_path_chain(
                            &attr.value,
                            &symbols,
                            &options,
                        );
                    if class_receiver || module_receiver {
                        options.definition_warnings.borrow_mut().push(format!(
                            "`{}.{} = ...` is dropped: {} attributes are not \
                             mutable in rython (the class/module-attribute \
                             mutation divergence)",
                            root,
                            attr.attr,
                            if matches!(
                                symbols.get(root),
                                Some(crate::SymbolTableNode::ClassDef(_))
                            ) || crate::resolve_class_referenced(root, &symbols, &options).is_some()
                            {
                                "class"
                            } else {
                                "module"
                            }
                        ));
                        return Ok(TokenStream::new());
                    }
                }
            }
            // A PROPERTY SETTER store (`self.url = v` where url is
            // `@property def url` + `@url.setter def url`): Python's
            // property assignment invokes the SETTER method. rython lowers
            // the setter as a plain method `{name}_set` (distinct Rust
            // name), so the store routes to the call `self.url_set(v)?`
            // instead of a field write.
            if let ExprType::Attribute(attr) = target
                && let Some((class, _class_symbols)) =
                    crate::receiver_class(&attr.value, &ctx, &symbols, &options)
                && class.is_property_setter(&attr.attr)
            {
                let recv = attr.value.clone().to_rust(
                    ctx.clone(),
                    options.clone(),
                    symbols.clone(),
                )?;
                let setter = crate::safe_ident(&format!("{}_set", attr.attr));
                return Ok(quote!(#recv.#setter(#value)?;));
            }
            match target {
                ExprType::Subscript(sub) => render_subscript_store(sub, value),
                _ => render_one(target, value),
            }
        };

        if self.targets.len() == 1 {
            // A store into an optional-tracked name goes through the
            // Option-slot lowering, which passes Option values through,
            // wraps plain values in Some, and handles conditional arms
            // independently (`x if c else None`).
            if let ExprType::Name(name) = &self.targets[0]
                && options.optional_names.contains(&name.id)
            {
                // The target is a STORE into the hoisted binding — never an
                // unwrap, even when the name is narrowed (issue #125): the
                // binding stays Option<T> and the store wraps in Some below.
                let target_code = {
                    let mut store_options = options.clone();
                    store_options.narrowed_names =
                        std::rc::Rc::new(std::collections::HashMap::new());
                    self.targets[0].clone().to_rust(
                        ctx.clone(),
                        store_options,
                        symbols.clone(),
                    )?
                };
                // An empty-container literal was already rendered with its
                // pinned element type above (Vec::<T>::new()); reuse it so
                // the Some wrap lands on the typed container, not on a bare
                // vec![] that rustc cannot infer.
                let value = if is_empty_container_literal(&value_expr) {
                    quote!(Some(#value))
                } else {
                    crate::lower_optional_value(
                        &value_expr,
                        ctx.clone(),
                        options.clone(),
                        symbols.clone(),
                    )?
                };
                return Ok(quote!(#target_code = #value;));
            }
            render(&self.targets[0], &value)
        } else {
            // Chained assignment (`a = b = expr`): Python evaluates the value
            // once and assigns it to each target in turn.
            // A container literal would break the aliasing semantics: Python
            // shares ONE object across all targets, while the lowering must
            // clone per target — later mutations through one name would
            // silently diverge. That is the documented aliasing divergence
            // (issues #79/#104), reported as a warning; the lowering binds
            // the literal once and clones per target (`result[key] =
            // entries = {}` — distlib's read_exports, the insert-and-build
            // idiom). An EMPTY container in a chain is boxed (its element
            // type cannot be pinned through the chain's multiple targets).
            if is_container_literal(&value_expr) {
                options.definition_warnings.borrow_mut().push(format!(
                    "chained assignment to a container literal (`a = b = {:?}`) \
                     cannot preserve Python's shared aliasing: each target \
                     receives its own copy (the documented aliasing divergence, \
                     issues #79/#104)",
                    value_expr
                ));
                let empty_dict = matches!(&value_expr, ExprType::Dict(d) if d.keys.is_empty());
                let empty_list = matches!(&value_expr, ExprType::List(l) if l.is_empty());
                let chain_value = if empty_dict || empty_list {
                    if empty_dict {
                        quote!(PyDict::<String, PyValue>::from([]))
                    } else {
                        quote!(Vec::<stdpython::PyValue>::new())
                    }
                } else {
                    value
                };
                let mut stream = quote!(let __rython_chain = #chain_value;);
                for target in &self.targets {
                    stream.extend(render(target, &quote!(__rython_chain.clone()))?);
                }
                return Ok(stream);
            }
            let mut stream = quote!(let __rython_chain = #value;);
            for target in &self.targets {
                stream.extend(render(target, &quote!(__rython_chain.clone()))?);
            }
            Ok(stream)
        }
    }
}

pub(crate) fn is_container_literal(expr: &ExprType) -> bool {
    matches!(
        expr,
        ExprType::List(_)
            | ExprType::Dict(_)
            | ExprType::Set(_)
            | ExprType::ListComp(_)
            | ExprType::DictComp(_)
            | ExprType::SetComp(_)
    )
}

/// Whether an expression is an EMPTY `[]`/`{}` literal (the shape the
/// pinned-element-type rendering above handles).
fn is_empty_container_literal(expr: &ExprType) -> bool {
    matches!(expr, ExprType::List(l) if l.is_empty())
        || matches!(expr, ExprType::Dict(d) if d.keys.is_empty())
}

/// A `self.<field>` read whose field type is an IMMUTABLE Python value
/// (str/bytes — the types whose clones are semantically faithful),
/// rendered as a CLONE so a store into a local does not move out of the
/// shared receiver (`box = self.scheme` — issue #222's local half;
/// E0507 otherwise). None for any other value shape. The use-count gate
/// does not apply: the receiver is `&self`, so even a single move is a
/// borrow violation — unlike the name-to-name case
/// (`stored_name_needs_clone`), where one move is legal and only reuse
/// poisons.
fn self_field_read_clone(
    value_expr: &ExprType,
    ctx: &crate::CodeGenContext,
    options: &crate::PythonOptions,
    symbols: &crate::SymbolTableScopes,
) -> Option<proc_macro2::TokenStream> {
    // A `typing.cast(T, value)` wrapper is a runtime identity: look
    // through it to the field read (`proxy_config = cast(ProxyConfig,
    // self.proxy_config)` — urllib3's _connect_tls_proxy — round 95).
    let cast_value = match value_expr {
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
                                crate::ExprType::Name(m) if crate::is_typing(&m.id)
                            )
                )) =>
        {
            Some(&call.args[1])
        }
        _ => None,
    };
    let field_expr = cast_value.unwrap_or(value_expr);
    let ExprType::Attribute(attr) = field_expr else {
        return None;
    };
    if !crate::ast::tree::visit::is_self(attr.value.as_ref()) {
        return None;
    }
    let (class, class_symbols) = crate::receiver_class(&attr.value, ctx, symbols, options)?;
    let fields = class.infer_fields(&class_symbols, options).ok()?;
    let (_, ty) = fields.iter().find(|(name, _)| *name == attr.attr)?;
    // An owned heap value anywhere inside — Option<String>,
    // Vec<u8>, PyValue, String — clones out of the shared receiver
    // (the structural twin of the old substring sniffing). An OPTION-
    // typed field clones for ANY inner (including a class —
    // `Option<ProxyConfig>` — the Option-slot pass-through already
    // clones attribute reads this way, round 90; the class instance
    // cannot be moved out of `&self`, round 95).
    let immutable = matches!(ty, crate::TypeInfo::Option(_))
        || crate::ast::tree::type_ctx::type_mentions_heap(ty);
    if !immutable {
        return None;
    }
    let tokens = value_expr
        .clone()
        .to_rust(ctx.clone(), options.clone(), symbols.clone())
        .ok()?;
    Some(quote!((#tokens).clone()))
}

/// Wrap a value destined for a boxed `PyDict<K, PyValue>` slot. A plain
/// value boxes (`PyValue::from(v)`); an OPTION value (`scheme or "http"`
/// where scheme is `str | None` — urllib3's PoolManager) absorbs into
/// the box — Some boxes the inner value, None is the boxed None (the
/// dict stores None exactly like CPython's `dict[str, Any]`). The
/// explicit match avoids a `From<Option<T>> for PyValue` blanket, whose
/// multiple candidates would make an UNTYPED value (`PyValue::from(
/// resolve(None)?)`) ambiguous at build time (E0283).
fn boxed_dict_value_wrap(
    value: &TokenStream,
    value_expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> TokenStream {
    if crate::expr_yields_option_ctx(value_expr, ctx, options, symbols) {
        quote!({
            match #value {
                Some(__rython_member) => PyValue::from(__rython_member),
                None => stdpython::PyValue::None_,
            }
        })
    } else {
        quote!(PyValue::from(#value))
    }
}
