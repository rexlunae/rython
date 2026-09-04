use proc_macro2::TokenStream;
use quote::format_ident;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableNode, SymbolTableScopes,
    extraction_failure,
};

use serde::{Deserialize, Serialize};

/// Rendered receiver tokens that resolve as MODULE PATHS even without an
/// import symbol in scope — robustness for the from-import spellings of
/// the common runtime modules; the general mechanism is
/// `is_module_path_chain`. ONE list, shared by the load form and the
/// store-target form (`os.environ[k] = v`), which previously kept two
/// copies that had already drifted apart. The bare-name set is the
/// [`crate::StdModule::bare_token_access`] property; the extra arms are
/// the numpy alias, the xml placeholder, and the nested module paths.
pub(crate) fn module_access_token(value_str: &str) -> bool {
    // The token stream renders `::` with surrounding spaces; normalize
    // once instead of matching both spellings in every arm.
    let normalized: String = value_str.chars().filter(|c| !c.is_whitespace()).collect();
    match normalized.as_str() {
        // The numpy import alias (`import numpy as np` is the canonical
        // spelling) and the comment-only xml placeholder module.
        "np" | "xml" => true,
        // Nested runtime module paths (os.path.join, urllib.request.
        // urlopen, np.linalg.inv). TLSVersion is ssl's nested constants
        // module — both the dotted (`ssl.TLSVersion.TLSv1_2`) and the
        // from-import (`TLSVersion.TLSv1_2`) spellings are paths.
        "os::path" | "urllib::request" | "numpy::linalg" | "np::linalg" | "ssl::TLSVersion"
        | "TLSVersion" => true,
        name => crate::StdModule::from_name(name).is_some_and(|m| m.bare_token_access()),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
//#[pyo3(transparent)]
pub struct Attribute {
    pub value: Box<ExprType>,
    pub attr: String,
    /// Load/Store/Del context marker, carried from the Python AST.
    pub ctx: String,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Attribute {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = ob
            .getattr("value")
            .map_err(|e| extraction_failure("Attribute.value", &ob, e))?;
        let attr = ob
            .getattr("attr")
            .map_err(|e| extraction_failure("Attribute.attr", &ob, e))?;
        let ctx = ob
            .getattr("ctx")
            .map_err(|e| extraction_failure("attribute context", &ob, e))?
            .get_type()
            .name()
            .map_err(|e| extraction_failure("attribute context type", &ob, e))?;
        Ok(Attribute {
            value: Box::new(
                value
                    .extract()
                    .map_err(|e| extraction_failure("Attribute.value", &ob, e))?,
            ),
            attr: attr
                .extract()
                .map_err(|e| extraction_failure("Attribute.attr", &ob, e))?,
            ctx: ctx.to_string(),
        })
    }
}

impl<'a> CodeGen for Attribute {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        // `type(self).__name__` — the class name string for repr/error
        // messages (urllib3's ConnectionPool/Retry/Timeout reprs). The
        // `type(self)` call alone lowers to the name string (call.rs's
        // type() rule), but this attribute READ on it would emit a
        // dangling `.__name__` on a String (E0609); the whole expression
        // IS the class name. `type(x).__name__` for a NON-self receiver
        // routes through the boxed value's runtime type name
        // (`PyValue::py_type_name` — "str"/"int"/... exactly Python's
        // spelling), so urllib3's `type(x).__name__` TypeError messages
        // carry CPython's text instead of an E0609.
        if self.attr == "__name__"
            && let ExprType::Call(call) = self.value.as_ref()
            && matches!(call.func.as_ref(), ExprType::Name(n) if n.id == "type")
            && call.args.len() == 1
        {
            if matches!(call.args.first(), Some(ExprType::Name(n)) if n.id == "self")
                && let Some(enclosing) = ctx.enclosing_class_name()
            {
                let name = crate::safe_ident(enclosing);
                return Ok(quote!(stringify!(#name).to_string()));
            }
            // The runtime path needs a CONCRETE argument: an inferred
            // generic parameter would need a `PyValue: From<T>` bound
            // this cannot add. Non-generic names and expressions only.
            let arg = call.args.first().expect("len checked above");
            let generic_param = matches!(arg, ExprType::Name(n)
                if options.param_type_vars.contains_key(&n.id));
            if generic_param {
                options.definition_warnings.borrow_mut().push(
                    "type(x).__name__ on an inferred generic parameter is dropped: \
                     the boxed conversion needs a concrete type (the \
                     class-as-value divergence)"
                        .to_string(),
                );
                return Ok(quote!(stdpython::PyValue::None_));
            }
            let recv = arg
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            options.definition_warnings.borrow_mut().push(
                "type(x).__name__ on a non-self receiver lowers through the boxed \
                 value's runtime type name (the class-as-value divergence)"
                    .to_string(),
            );
            return Ok(quote!(stdpython::py_value_type_name(&stdpython::PyValue::from(#recv)).to_string()));
        }
        // Inheritance-aware field access, computed before `self.value` is
        // moved below: `self.name` where `name` is a base class's field, or
        // `dog.name` where `dog` is a derived-class instance, must reach
        // through the embedded base structs (or the trait's base accessors).
        let field_access = class_field_access(&self.value, &self.attr, &ctx, &symbols, &options);
        // Computed before `ctx`/`symbols`/`options` move into the receiver's
        // rendering: whether the receiver is a SHARED class's value.
        let shared_recv = shared_receiver(&self.value, &ctx, &symbols, &options);
        // A Rust-module attribute (`crc32c.crc32c` where `crc32c` was
        // `import`ed from a rython.toml binding) is a path into the bound
        // crate — never a field access. The crate name comes from the spec
        // so aliased imports (`import crc32c as c`) still emit the real
        // crate path.
        if let ExprType::Name(root) = self.value.as_ref() {
            let module_spec = match symbols.get(&root.id) {
                Some(crate::SymbolTableNode::Alias(canonical)) => {
                    symbols.get(canonical).and_then(|s| match s {
                        crate::SymbolTableNode::RustModule(spec) => Some(spec.clone()),
                        _ => None,
                    })
                }
                Some(crate::SymbolTableNode::RustModule(spec)) => Some(spec.clone()),
                _ => None,
            };
            if let Some(spec) = module_spec {
                let crate_ident = crate::safe_ident(&spec.crate_name.replace('-', "_"));
                let attr = crate::safe_ident(&self.attr);
                return Ok(quote!(#crate_ident::#attr));
            }
        }
        // If a user binding shadows the root name (e.g. a variable named
        // `re`), the attribute is a field access on that value — never a
        // stdlib module path. Computed before `self.value` is moved below.
        let root_shadowed = crate::ast::tree::call::root_name(&self.value)
            .is_some_and(|root| crate::module_name_shadowed(root, &symbols));
        // A CLASS-CONSTANT read (`GzipDecoderState.SWALLOW_DATA` — urllib3's
        // response decoders): literal class-level constants emit as
        // `impl X { pub const NAME: T = v; }` (class_def.rs), so the read
        // renders `X::NAME`, not `X.NAME`. Computed before moves below.
        // The rendered path is the CLASS's name, not the receiver's
        // identifier: a @classmethod body reads `cls.DEFAULT` where `cls`
        // is bound to the enclosing ClassDef — the constant lives on the
        // class (urllib3's Retry.from_int).
        let class_const_read: Option<String> = match self.value.as_ref() {
            ExprType::Name(receiver) => symbols
                .get(&receiver.id)
                .is_some_and(|s| match s {
                    crate::SymbolTableNode::ClassDef(class) => class.body.iter().any(|bs| {
                        matches!(
                            &bs.statement,
                            crate::StatementType::Assign(a)
                                if a.targets.len() == 1
                                    && matches!(&a.targets[0], ExprType::Name(n) if n.id == self.attr)
                                    && crate::ast::tree::module::const_static_type(&a.value)
                                        .is_some()
                        )
                    }),
                    _ => false,
                })
                .then(|| {
                    // The class's OWN name (for `cls`, the receiver
                    // identifier is not a Rust type in scope).
                    match symbols.get(&receiver.id) {
                        Some(crate::SymbolTableNode::ClassDef(c)) => c.name.clone(),
                        _ => receiver.id.clone(),
                    }
                }),
            _ => None,
        };
        // A class-level COMPUTED constant read
        // (`self._encode_url_methods`, `cls.X`, `RequestMethods.X` —
        // urllib3's RequestMethods): the LazyLock static lives at MODULE
        // level under the class-mangled name (associated statics are not
        // legal Rust — issue #137); the read deref-clones it.
        {
            let attr_is_lazy_const = |class: &crate::ClassDef| -> bool {
                class.body.iter().any(|bs| {
                    matches!(
                        &bs.statement,
                        crate::StatementType::Assign(a)
                            if a.targets.len() == 1
                                && matches!(&a.targets[0], ExprType::Name(n) if n.id == self.attr)
                                && crate::ast::tree::module::const_static_type(&a.value).is_none()
                                && crate::ast::tree::class_def::class_body_computed_constant(&a.value)
                    )
                })
            };
            let owning_class: Option<String> = match self.value.as_ref() {
                ExprType::Name(receiver)
                    if receiver.id == "self" || receiver.id == "cls" =>
                {
                    ctx.enclosing_class_name()
                        .and_then(|c| match symbols.get(c) {
                            Some(crate::SymbolTableNode::ClassDef(cd))
                                if attr_is_lazy_const(cd) =>
                            {
                                Some(cd.name.clone())
                            }
                            _ => None,
                        })
                }
                ExprType::Name(receiver) => match symbols.get(&receiver.id) {
                    Some(crate::SymbolTableNode::ClassDef(cd))
                        if attr_is_lazy_const(cd) =>
                    {
                        Some(cd.name.clone())
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(class) = owning_class {
                // Through the associated ACCESSOR, so reads work from any
                // module the class is imported into.
                let class_ident = crate::safe_ident(&class);
                let accessor = crate::safe_ident(&self.attr);
                return Ok(quote!(#class_ident::#accessor()));
            }
        }
        // numpy attribute reads on a value the inference knows is an
        // array. `ndim`/`size` are plain field reads that already print
        // like Python ints; `shape` and `T` need the runtime's accessors
        // (issues #197, #204).
        // (`root_shadowed` is not consulted: it marks a name BOUND by user
        // code, which every array local is.)
        if crate::ast::tree::type_ctx::is_ndarray_expr(&self.value, &options, &symbols) {
            let recv = self
                .value
                .clone()
                .to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            match self.attr.as_str() {
                // A Python tuple, not the backing Vec: `(3,)`, not `[3]`.
                "shape" => return Ok(quote!((#recv).shape_tuple())),
                // numpy's transpose view; rython's arrays are values, so
                // this is the same copy `np.transpose(a)` makes.
                "T" => return Ok(quote!((#recv).transpose())),
                _ => {}
            }
        }
        // A receiver that IS a class (a @classmethod's `cls`, or a bare
        // class name read as a value): an attribute on it that is NOT a
        // class-body constant (`cls.DEFAULT` where `Retry.DEFAULT =
        // Retry(3)` is assigned at MODULE level after the class — urllib3's
        // Retry.from_int) has no static item — the module-level
        // class-attribute divergence: the read boxes to None. Also an
        // IMPORTED class (`Retry.DEFAULT` in connectionpool.py — Retry is
        // imported from util.retry): the class body has no DEFAULT, so the
        // read boxes the same way.
        let class_value_receiver: Option<String> = match self.value.as_ref() {
            ExprType::Name(receiver) => {
                let is_class = match symbols.get(&receiver.id) {
                    Some(crate::SymbolTableNode::ClassDef(_)) => true,
                    Some(crate::SymbolTableNode::ImportFrom(_)) => {
                        crate::resolve_class_referenced(&receiver.id, &symbols, &options).is_some()
                    }
                    Some(crate::SymbolTableNode::Alias(_)) => {
                        crate::resolve_class_referenced(&receiver.id, &symbols, &options).is_some()
                    }
                    _ => false,
                };
                if is_class {
                    if class_const_read.is_some() {
                        None
                    } else {
                        Some(receiver.id.clone())
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        let module_chain = is_module_path_chain(&self.value, &symbols, &options);
        // A module-path member read into a CRATE module whose generated
        // code has no item of that name (`util.ssl_.PROTOCOL_TLS` —
        // urllib3's pyopenssl, where PROTOCOL_TLS is an external ssl
        // constant the generated ssl_ module never defines): the member is
        // unmodeled — the read lowers to the boxed None (the
        // dynamic-module-member divergence, the same model getattr uses).
        // stdpython modules are exempt: their members resolve through the
        // runtime. Computed before `self.value`/`symbols` are moved below.
        // A module-path member read where the member is a CLASS in the
        // defining module (`urllib3_connection::HTTPSConnection` — urllib3's
        // http2/__init__.py, `orig_HTTPSConnection =
        // urllib3_connection.HTTPSConnection`): a class read as a VALUE has
        // no runtime equivalent — the boxed None (the classes-as-values
        // divergence, the same model name.rs uses). The class IS a path
        // item, but a VALUE read of it is unrepresentable.
        let class_value_module_member = module_chain
            && !crate::ast::tree::call::root_name(&self.value)
                .is_some_and(|r| crate::is_stdpython_module(r))
            && crate::ast::tree::call::module_path_of_chain(&self.value, &symbols, &options)
                .is_some_and(|mod_path| {
                    options.module_defs.contains_key(&mod_path)
                        && crate::ast::tree::module::module_def_has_path_item(
                            &options,
                            &mod_path,
                            &self.attr,
                        )
                        && crate::module_class_def(&options, &mod_path, &self.attr).is_some()
                });
        let missing_module_member = module_chain
            && !crate::ast::tree::call::root_name(&self.value)
                .is_some_and(|r| crate::is_stdpython_module(r))
            && crate::ast::tree::call::module_path_of_chain(&self.value, &symbols, &options)
                .is_some_and(|mod_path| {
                    options.module_defs.contains_key(&mod_path)
                        && !crate::ast::tree::module::module_def_has_path_item(
                            &options,
                            &mod_path,
                            &self.attr,
                        )
                });
        // An attribute read on an except-bound name (`e.expected` —
        // urllib3's _error_catcher reading IncompleteRead's dynamic
        // fields): the exception object has no static fields (rython
        // models exceptions as name + message), so the read lowers to
        // the boxed None (the dynamic-attribute divergence). Computed
        // before `self.value`/`symbols` are moved below.
        let except_binding_receiver: Option<(String, Option<String>)> = match self.value.as_ref() {
            ExprType::Name(receiver) => match symbols.get(&receiver.id) {
                Some(crate::SymbolTableNode::ExceptBinding(cls)) => {
                    Some((receiver.id.clone(), cls.clone()))
                }
                _ => None,
            },
            _ => None,
        };
        // A MODELED exception field read (`e.needed` on a caught
        // InsufficientFunds whose __init__ stored self.needed — bank,
        // round 99): the typed accessor's FULL tokens, computed HERE
        // before the moves below.
        let modeled_exc_field: Option<proc_macro2::TokenStream> = except_binding_receiver
            .as_ref()
            .and_then(|(_, caught)| {
                let cls_name = caught.as_ref()?;
                let cls = match symbols.get(cls_name) {
                    Some(crate::SymbolTableNode::ClassDef(c)) => c,
                    _ => return None,
                };
                let ty = crate::exception_field_type(cls, &self.attr, &symbols, &options)?;
                let recv = self
                    .value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .ok()?;
                let attr_name = &self.attr;
                Some(match ty {
                    crate::TypeInfo::Int => {
                        quote!((#recv).attr_i64(#attr_name)?)
                    }
                    crate::TypeInfo::String => {
                        quote!((#recv).attr_string(#attr_name)?)
                    }
                    _ => return None,
                })
            });
        // True when the chain's root is a vendored `[python-modules]`
        // dependency — those lower to `crate::<dep>::<attr>` paths (see
        // the emission below). Computed before `self.value` is moved.
        let vendored_module_chain = module_chain
            && crate::ast::tree::call::root_name(&self.value)
                .is_some_and(|root| options.python_modules.contains(root));
        // Whether the chain is a single segment (`textlib.double`): the
        // `crate::` prefix belongs to the ROOT segment only, so a nested
        // chain (`textlib.core.double`) must NOT be re-prefixed. Checked
        // before `self.value` is moved.
        let single_segment_chain = matches!(self.value.as_ref(), ExprType::Name(_));
        // An EXTERNAL module root (ssl, socket, zlib, logging, ...) — no
        // generated items exist, so the attribute lowers to the boxed None
        // (computed before `self.value`/`symbols` are moved below).
        let external_root = external_module_root(&self.value, &symbols, &options);
        // A BOXED-PyValue receiver (`self._response().body.closed` — the
        // emscripten response where `body` is a PyValue field): the
        // attribute read has no static shape — it lowers to the boxed None
        // (dynamic-attribute divergence, the same model getattr uses).
        // Computed before `self.value`/`symbols` are moved below.
        let pyvalue_receiver = receiver_is_pyvalue(&self.value, &ctx, &symbols, &options);
        let receiver_is_field_chain = matches!(self.value.as_ref(), ExprType::Attribute(_));
        // Issue #137 round 26: a plain NAME receiver drops too, but only on
        // POSITIVE evidence of boxing — never on `TypeInfo::PyObject`, the
        // inferrer's "no answer", which is what round 24 keyed on through
        // `receiver_is_pyvalue` and why it discarded a live value.
        //
        // A protocol method is excluded: `b.lower()` on a boxed value is
        // real code the runtime forwards, and this path renders the callee
        // of such a call.
        let positively_boxed_name = receiver_is_boxed_positively(&self.value, &symbols, &options)
            && !pyvalue_protocol_method(&self.attr);
        // A PROPERTY GETTER read (`self.url` — urllib3's geturl, where url
        // is `@property def url`): the property lowers as a plain method
        // returning Result, so the read routes to the getter CALL and
        // unwraps (`self.url()?`). Computed before the moves below.
        let property_getter =
            crate::receiver_class_for_read(&self.value, &ctx, &symbols, &options)
                .is_some_and(|(class, class_symbols)| {
                    class.has_property_getter(&self.attr, &class_symbols, &options)
                });
        if self.attr == "x" && matches!(self.value.as_ref(), ExprType::Name(n) if n.id == "v") {
            let nt = crate::infer_type(Some(&ctx), &self.value, &options, &symbols);
            eprintln!("R99PROP v type={:?}", nt);
        }
        let warnings = options.definition_warnings.clone();
        // Issue #137's Option-aware access: a READ through an
        // Option-typed receiver (`self.timeout.connect_timeout` where the
        // field is `Timeout | None`, `resp.headers.get(...)` where resp is
        // an Option param — urllib3) unwraps the Option first — CPython
        // would raise AttributeError on a None receiver, which rython
        // represents as a loud §12.2 panic with CPython's message. A
        // narrowed receiver never reaches here (narrowed_names unwraps
        // reads already); guarded access stays exact. Computed BEFORE
        // `self.value` is moved by the to_rust below.
        let option_receiver = {
            // Round 81's `and`-chain narrowing (`if conn and
            // conn.is_connected():` — urllib3) proves the NAME non-None:
            // its read already emits `(conn).clone().unwrap()` (the
            // unwrapped PyValue), so the Option-unwrap here would
            // double-unwrap — `unwrap_or_else` on a PyValue (E0599).
            let receiver_narrowed = matches!(self.value.as_ref(), ExprType::Name(n)
                if options.narrowed_names.contains_key(&n.id));
            if receiver_narrowed {
                None
            } else {
                receiver_option_inner(&self.value, &ctx, &symbols, &options)
            }
        };
        // The generic trait-default context (`Self: {Class}Trait`), captured
        // before `ctx` is moved: the base-accessor hops generated below must
        // be qualified with the OWN trait to dodge the own-vs-ancestor
        // ambiguity (E0034).
        let trait_ctx_class: Option<String> = match &ctx {
            CodeGenContext::Trait { class, generic: true, .. } => Some(class.clone()),
            _ => None,
        };
        // The fallibility rule (the review's fix 2, round 99): a method
        let mut value_tokens = self.value.to_rust(ctx, options, symbols)?;
        if let Some(_inner) = option_receiver {
            let attr_name = self.attr.clone();
            value_tokens = quote!((#value_tokens).clone().unwrap_or_else(|| {
                panic!(
                    "AttributeError: 'NoneType' object has no attribute '{}'",
                    #attr_name
                )
            }));
        }
        let value_str = value_tokens.to_string();
        let attr = crate::safe_ident(&self.attr);
        // Debug-form of the receiver for -W messages (captured before the
        // move above).
        let value_debug = format!("{:?}", value_str);

        // Determine if this is a module access or a field/method access.
        // The bare-token module set lives in module_access_token (ONE list
        // shared with the store-target form). A dotted chain rooted at a
        // plain module `import` (`import pylev`) is also a module path:
        // `pylev.wfi_levenshtein` emits `pylev::wfi_levenshtein`, and
        // nested chains (`pylev.sub.fn`) emit `pylev::sub::fn` because
        // each inner attribute resolves the same way. ImportFrom bindings
        // are VALUES (a function/class), not modules, so they never turn
        // attribute access into a path.
        let is_module_access =
            !root_shadowed && (module_access_token(&value_str) || module_chain);

        if is_module_access {
            // An EXTERNAL module (ssl, socket, zlib, logging, ...) has no
            // generated items: the attribute read lowers to the boxed None
            // with a warning (documented divergence — the external object's
            // attribute is unmodeled, the same model getattr uses).
            if let Some(root) = &external_root {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` is dropped: the module `{}` is external to the generated \
                     crate (external-module divergence)",
                    root, self.attr, root
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // A module-path member read into a CRATE module with no item of
            // that name: the boxed None (see missing_module_member above).
            if missing_module_member {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` is dropped: the generated module has no runtime item for \
                     `{}` (the member is unmodeled — the \
                     dynamic-module-member divergence)",
                    value_debug, self.attr, self.attr
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // A module-path member that is a CLASS read as a VALUE
            // (`urllib3_connection::HTTPSConnection` — http2/__init__.py):
            // the boxed None (classes-as-values divergence).
            if class_value_module_member {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` (a class read as a value) lowers to the boxed None \
                     (classes cannot be runtime values in rython)",
                    value_debug, self.attr
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // Use :: for module access (Python's sys.executable becomes sys::executable)
            // Special handling for LazyLock static variables that need
            // dereferencing. os::environ is NOT here: it is a live-view
            // unit struct whose methods auto-ref. The ssl version
            // constants are LazyLock statics in both backends (the openssl
            // backend's real version is only knowable at runtime), so
            // reads deref to the plain &str / i64 / tuple value.
            let needs_deref = matches!(
                (value_str.as_str(), self.attr.as_str()),
                ("sys", "executable")
                    | ("sys", "argv")
                    | ("ssl", "OPENSSL_VERSION")
                    | ("ssl", "OPENSSL_VERSION_NUMBER")
                    | ("ssl", "OPENSSL_VERSION_INFO")
            );

            // `sys.modules` — the process's import registry (requests'
            // packages.py aliasing): rython's crate is static, so the
            // registry is always empty — the read lowers to an empty dict
            // (list(sys.modules) iterates nothing, indexing misses).
            if crate::StdModule::from_name(&value_str) == Some(crate::StdModule::Sys)
                && self.attr == "modules"
            {
                return Ok(quote!(PyDict::<String, stdpython::PyValue>::from([])));
            }

            if needs_deref {
                // Wrap dereferenced values in parentheses to ensure correct precedence
                // This prevents *sys::executable.to_string() and ensures (*sys::executable).to_string()
                Ok(quote!((*#value_tokens::#attr)))
            } else if vendored_module_chain {
                // Vendored `[python-modules]` deps are sibling modules of
                // the generated crate: `textlib.double(...)` emits
                // `crate::textlib::double(...)` — the path resolves in the
                // lib (pub mod) and in the bin (mod textlib) alike, with no
                // `use` statement needed. Same-package modules and stdlib
                // paths keep their plain form.
                //
                // The `crate::` prefix belongs to the ROOT segment only: a
                // nested chain (`textlib.core.double`) renders its inner
                // attribute (`textlib.core`) through this same branch,
                // which already prefixes it — re-prefixing here would
                // double it (`crate::crate::...`).
                if single_segment_chain {
                    Ok(quote!(crate::#value_tokens::#attr))
                } else {
                    Ok(quote!(#value_tokens::#attr))
                }
            } else {
                Ok(quote!(#value_tokens::#attr))
            }
        } else {
            // A read on an EXTERNAL-module receiver (`TLSVersion.
            // MINIMUM_SUPPORTED` — from ssl, external): the attribute has
            // no runtime item — the boxed None.
            if let Some(root) = &external_root {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` is dropped: the module `{}` is external to the generated \
                     crate (external-module divergence)",
                    root, self.attr, root
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // A read on a BOXED-PyValue receiver (`self._response().body
            // .closed` — the emscripten response's `body` field is PyValue):
            // the attribute has no static shape — the boxed None (the
            // dynamic-attribute divergence).
            // A read on a BOXED-PyValue receiver (`self._response().body
            // .closed` — the emscripten response's `body` field is PyValue):
            // the attribute has no static shape — the boxed None (the
            // dynamic-attribute divergence). Only FIELD-CHAIN receivers
            // qualify: a plain Name receiver may be a CALLEE being rendered
            // (`b.lower()` renders its func through this path), and its
            // dynamic protocol methods must not be boxed away.
            if pyvalue_receiver && (receiver_is_field_chain || positively_boxed_name) {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` is dropped: the receiver is a boxed PyValue \
                     (dynamic-attribute divergence)",
                    value_debug, self.attr
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // A class-constant read (`GzipDecoderState.SWALLOW_DATA`):
            // `X::NAME` — the const lives on the class's impl block.
            if let Some(receiver) = &class_const_read {
                let receiver = crate::safe_ident(receiver);
                return Ok(quote!(#receiver::#attr));
            }
            // A CLASS receiver read as a value with a non-const attribute
            // (`cls.DEFAULT` — module-level class attribute): no static
            // item — the boxed None (module-level class-attribute
            // divergence).
            if let Some(receiver) = &class_value_receiver {
                warnings.borrow_mut().push(format!(
                    "`{}.{}` (a module-level class attribute) lowers to the boxed \
                     None (the class-attribute divergence; class attributes \
                     assigned outside the class body are not importable)",
                    receiver, self.attr
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // An attribute read on an except-bound name (`e.expected` —
            // urllib3's _error_catcher reading IncompleteRead's dynamic
            // fields): the exception object has no static fields (rython
            // models exceptions as name + message), so the read lowers to
            // the boxed None (the dynamic-attribute divergence).
            if let Some((receiver, _caught)) = &except_binding_receiver {
                // A MODELED exception field (`e.needed` — bank, round
                // 99): the typed accessor, precomputed above before the
                // moves.
                if let Some(tokens) = &modeled_exc_field {
                    return Ok(tokens.clone());
                }
                warnings.borrow_mut().push(format!(
                    "`{}.{}` is dropped: the receiver is an exception object \
                     bound by `except ... as`, and rython models exceptions \
                     as name + message (the dynamic-attribute divergence)",
                    receiver, self.attr
                ));
                return Ok(quote!(stdpython::PyValue::None_));
            }
            // Use . for field/method access (Python's obj.field becomes obj.field).
            // A class field owned by an ancestor of the receiver's class is
            // reached through the embedded base structs (`self.__rython_base.f`)
            // or, in a generic trait default, through the base accessors
            // (`self.base().f`); an own field in a generic trait default goes
            // through its accessor (`self.f()`).
            // A PROPERTY GETTER read (`self.url` — urllib3's geturl, where
            // url is `@property def url`): the property lowers as a plain
            // method returning Result, so the read routes to the getter CALL
            // and unwraps (`self.url()?`). Only when the receiver's class
            // actually defines the getter — a genuine field read is untouched.
            if property_getter && field_access.is_none() {
                // A SHARED receiver's getter call borrows first (the
                // PyRef itself has no methods — records's v.patch where v
                // is a PyRef<Version>, round 99).
                if shared_recv {
                    return Ok(quote!((#value_tokens).borrow().#attr()?));
                }
                return Ok(quote!(#value_tokens.#attr()?));
            }
            // The fallibility rule (the review's fix 2, round 99): a
            // METHOD name read on an Option-typed receiver of a user class
            // is a fallible call site — the read carries `?` so the call
            // The value-read deref lives in lower_optional_value's boxed
            // passthrough (one map per read — the double-map here fired
            // twice on the nested chains, round 99); the method-receiver
            // reads keep the plain form (Box autoderefs).
            let boxed_field_read = false;
            match field_access {
                // A SHARED class's value is a `PyRef` (shared.rs): a read
                // borrows the one object and clones the field out.
                None if shared_recv => {
                    Ok(quote!((#value_tokens).borrow().#attr.clone()))
                }
                None if boxed_field_read => {
                    Ok(quote!(#value_tokens.#attr.clone().map(| b | * b)))
                }
                None => Ok(quote!(#value_tokens.#attr)),
                Some(FieldRewrite::Accessor { field }) => {
                    let accessor = crate::safe_ident(&field);
                    Ok(quote!(#value_tokens.#accessor()))
                }
                Some(FieldRewrite::Chain { depth }) => {
                    let chain = base_field_chain(depth);
                    Ok(quote!(#value_tokens #chain.#attr))
                }
                Some(FieldRewrite::AccessorChain { depth }) => {
                    // `self.base()…​.name()` — the base accessor returns a
                    // shared reference to the embedded struct, and the field
                    // accessor clones it out (an ancestor's field is not on
                    // the generic Self; a bare field access would move out
                    // of the shared reference).
                    //
                    // In a GENERIC trait default the first base hop is
                    // ambiguous (E0034): the own trait AND every ancestor
                    // trait declare `base`, each returning ITS own base
                    // type. Qualify the first link with the own trait; the
                    // remaining hops land on concrete types, where the
                    // inherent accessor wins.
                    if let Some(class) = &trait_ctx_class {
                        let trait_name = format_ident!("{}Trait", class);
                        let rest = base_accessor_chain(depth.saturating_sub(1));
                        Ok(quote!(
                            <Self as #trait_name>::base(#value_tokens) #rest.#attr()
                        ))
                    } else {
                        let chain = base_accessor_chain(depth);
                        Ok(quote!(#value_tokens #chain.#attr()))
                    }
                }
            }
        }
    }
}

/// Store/place-flavored sibling of [`CodeGen::to_rust`]: renders an
/// attribute so it can be ASSIGNED THROUGH. Identical to the load form
/// everywhere except a generic trait default's own fields, where the load
/// form clones out (`self.f()`) and a store needs the mutable accessor
/// (`*self.f_mut() = ...`, or `self.f_mut()` without the deref when the
/// caller wraps it as a method-call receiver / subscript place).
pub(crate) fn to_rust_place(
    value: &ExprType,
    attr: &str,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    deref: bool,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    to_rust_place_expr(
        &ExprType::Attribute(Attribute {
            value: Box::new(value.clone()),
            attr: attr.to_string(),
            ctx: "Store".to_string(),
        }),
        ctx,
        options,
        symbols,
        deref,
    )
}

/// Place-flavored renderer for a whole attribute/subscript chain. Each
/// attribute segment rewrites its receiver to place flavor recursively, so
/// a store through a composition chain (`self.inner.x = v`, `self.a.b[i] =
/// v`, `self.inner.items.append(v)`) keeps the write on the REAL field:
/// in a generic trait default the load accessor clones (`self.inner()`), so
/// a store rendered on top of it would silently vanish. The mutable
/// accessor (`self.inner_mut()`) is a place, and field access auto-derefs
/// through the returned `&mut`, so chaining is valid.
///
/// Module paths (`os.environ`, `pylev.fn`) are NOT places — they render as
/// `::` paths exactly like the load form.
pub(crate) fn to_rust_place_expr(
    expr: &ExprType,
    ctx: &CodeGenContext,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
    deref: bool,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    match expr {
        ExprType::Attribute(attr) => {
            // Mirror the module-path guard from to_rust: a store target
            // that resolves as a module path is not a field access. A user
            // binding that shadows the module name (`os = {...}`) must NOT
            // be treated as a module — the load path applies the same
            // `module_name_shadowed` guard.
            let root_shadowed = crate::ast::tree::call::root_name(&attr.value)
                .is_some_and(|root| crate::module_name_shadowed(root, symbols));
            let value_tokens =
                attr.value.clone().to_rust(ctx.clone(), options.clone(), symbols.clone())?;
            let is_module = !root_shadowed
                && (crate::ast::tree::call::root_name(&attr.value).is_some_and(|_root| {
                    crate::ast::tree::attribute::is_module_path_chain(&attr.value, symbols, options)
                }) || module_access_token(&value_tokens.to_string()));
            if is_module {
                // Module stores render as PATHS, exactly like the load form.
                // The attribute is an identifier — interpolating the &str
                // directly would emit a string literal (`os . "environ"`)
                // that the Rust compiler rejects.
                let attr_path = crate::safe_ident(&attr.attr);
                // Vendored deps prefix `crate::` at the ROOT segment only;
                // nested segments already carry it from their own rendering.
                let vendored_root = matches!(attr.value.as_ref(), ExprType::Name(_))
                    && crate::ast::tree::call::root_name(&attr.value)
                        .is_some_and(|root| options.python_modules.contains(root));
                if vendored_root {
                    return Ok(quote!(crate::#value_tokens::#attr_path));
                }
                return Ok(quote!(#value_tokens::#attr_path));
            }
            // The receiver must itself be a place. `class_field_access`
            // only rewrites a receiver that is BARE `self` (its `is_self`
            // test), so a chain like `self.inner.x` resolves `x` against
            // `Inner` with no rewrite — the place flavor of the receiver is
            // what keeps the store on the real field.
            let recv_place = to_rust_place_expr(&attr.value, ctx, options, symbols, false)?;
            let field_access =
                class_field_access(&attr.value, &attr.attr, ctx, symbols, options);
            let attr_ident = crate::safe_ident(&attr.attr);
            match field_access {
                // A SHARED class's value is a `PyRef` (shared.rs): the
                // store borrows the one object mutably (the assigned
                // value is evaluated before the place, so a read of the
                // same object in it has released its borrow).
                None if shared_receiver(&attr.value, ctx, symbols, options) => {
                    Ok(quote!((#recv_place).borrow_mut().#attr_ident))
                }
                None => Ok(quote!(#recv_place.#attr_ident)),
                Some(FieldRewrite::Accessor { field }) => {
                    let accessor = crate::safe_ident(&format!("{}_mut", field));
                    if deref {
                        Ok(quote!(*#recv_place.#accessor()))
                    } else {
                        Ok(quote!(#recv_place.#accessor()))
                    }
                }
                Some(FieldRewrite::Chain { depth }) => {
                    let chain = base_field_chain(depth);
                    Ok(quote!(#recv_place #chain.#attr_ident))
                }
                Some(FieldRewrite::AccessorChain { depth }) => {
                    // The mutable twin of the load form's qualification
                    // (see above): the first base_mut hop on the generic
                    // Self is ambiguous between the own trait and the
                    // ancestor traits — qualify it.
                    let place = if let CodeGenContext::Trait { class, generic: true, .. } = &ctx {
                        let trait_name = format_ident!("{}Trait", class);
                        let rest = base_mut_accessor_chain(depth.saturating_sub(1));
                        quote!(<Self as #trait_name>::base_mut(#recv_place) #rest)
                    } else {
                        let chain = base_mut_accessor_chain(depth);
                        quote!(#recv_place #chain)
                    };
                    Ok(quote!(#place.#attr_ident))
                }
            }
        }
        ExprType::Subscript(sub) => {
            crate::ast::tree::subscript::subscript_receiver_place(
                &ExprType::Subscript(sub.clone()),
                ctx.clone(),
                options.clone(),
                symbols.clone(),
            )
        }
        // A NAME narrowed from a polymorphic root to a class of its subtree
        // (hierarchy.rs): as a PLACE it is the sum type's mutable view, so
        // a store or a mutating call through it reaches the real value
        // (the read view is a clone). A narrowing to a nested root has no
        // place: loud.
        ExprType::Name(n)
            if let Some(root) = options.narrowed_class_origin.get(&n.id)
                && let Some(crate::TypeInfo::Class(t)) = options.narrowed_names.get(&n.id)
                && t != root =>
        {
            if crate::ast::tree::hierarchy::is_polymorphic_root(t) {
                return Err(format!(
                    "mutating `{}` inside `isinstance({}, {})` is not supported: `{}` is \
                     itself a base class, and its narrowed view is a copy — test for \
                     the concrete class, or mutate through a method of `{}`",
                    n.id, n.id, t, t, root
                )
                .into());
            }
            let name = crate::safe_ident(&n.id);
            let as_mut = format_ident!("__rython_as_{}_mut", t);
            Ok(quote!((#name).#as_mut().unwrap()))
        }
        _ => expr.clone().to_rust(ctx.clone(), options.clone(), symbols.clone()),
    }
}

/// True when an expression chain is rooted at the method receiver `self`
/// (`self.inner`, `self.a.b`) — the receivers whose load form clones in a
/// generic trait default, so mutating operations must render them as
/// places.
pub(crate) fn chain_root_is_self(expr: &ExprType) -> bool {
    match expr {
        ExprType::Attribute(a) => chain_root_is_self(&a.value),
        ExprType::Name(n) => n.id == "self",
        _ => false,
    }
}

/// Whether an expression is, or is a field chain rooted at, a root-typed
/// name NARROWED to a class of its subtree (`s` after `isinstance(s,
/// Circle)`): a mutation through it must render the chain as a place
/// from the sum type's mutable view (`__rython_as_Circle_mut`), never
/// from the read view's clone — `s.radius = ..`, `s.tags.append(..)`,
/// `s.center.bump()` alike (Devin review on #319).
/// Whether `expr` is a FIELD CHAIN rooted at a SHARED instance that is not
/// a polymorphic root (`a.items` where `a = accounts[k]`, `q.center`):
/// a mutation through it must render the chain as a place — the field
/// through the mutable borrow — never through the read's clone of the
/// field (`a.items.append(x)`, `a.center.bump()`; Devin review on #321).
pub(crate) fn chain_root_is_shared_instance(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Attribute(a) = expr else {
        return false;
    };
    let mut root = a.value.as_ref();
    while let ExprType::Attribute(inner) = root {
        root = inner.value.as_ref();
    }
    if crate::ast::tree::visit::is_self(root) {
        return false;
    }
    shared_receiver(root, ctx, symbols, options)
}

pub(crate) fn chain_root_is_narrowed_class(expr: &ExprType, options: &PythonOptions) -> bool {
    match expr {
        ExprType::Attribute(a) => chain_root_is_narrowed_class(&a.value, options),
        ExprType::Name(n) => options.narrowed_class_origin.get(&n.id).is_some_and(|root| {
            matches!(options.narrowed_names.get(&n.id), Some(crate::TypeInfo::Class(t)) if t != root)
        }),
        _ => false,
    }
}

/// `.__rython_base` repeated `depth` times: reaches an ancestor's embedded
/// struct from a concrete receiver (`self.__rython_base.name`).
pub(crate) fn base_field_chain(depth: usize) -> TokenStream {
    let mut chain = TokenStream::new();
    for _ in 0..depth {
        chain.extend(quote!(.__rython_base));
    }
    chain
}

/// `.base()` repeated `depth` times: reaches an ancestor's struct from the
/// generic `self` of a trait default (`self.base().name`).
fn base_accessor_chain(depth: usize) -> TokenStream {
    let mut chain = TokenStream::new();
    for _ in 0..depth {
        chain.extend(quote!(.base()));
    }
    chain
}

/// `.base_mut()` repeated `depth` times: the mutable form of
/// [`base_accessor_chain`] for stores through an ancestor's fields.
fn base_mut_accessor_chain(depth: usize) -> TokenStream {
    let mut chain = TokenStream::new();
    for _ in 0..depth {
        chain.extend(quote!(.base_mut()));
    }
    chain
}

/// How an attribute access `value.attr` must be rewritten to reach a class
/// field through the inheritance chain, if it is a class field at all.
///
/// - `Accessor`: `self` in a generic trait default, own field — the field
///   is not on the generic `Self`, so loads go through `self.<f>()` and
///   stores through `self.<f>_mut()`.
/// - `Chain`: a concrete receiver (a local constructed from a class, `self`
///   in an inherent method or an override body, or a composition chain) —
///   the field lives inside embedded base structs, so the access is
///   `#receiver.__rython_base…​.attr`.
/// - `AccessorChain`: `self` in a generic trait default, ancestor field —
///   `self.base()…​.attr` (load) / `self.base_mut()…​.attr` (place).
///
/// Returns None for non-fields (methods, module paths, plain struct fields
/// that need no rewrite).
pub(crate) fn class_field_access(
    value: &ExprType,
    attr: &str,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<FieldRewrite> {
    let is_self = crate::ast::tree::visit::is_self(value);
    // A polymorphic ROOT's value (hierarchy.rs) may be the sum type, which
    // has no fields — only the accessors every variant implements (the
    // root struct too, through its trait), so a non-`self` receiver of a
    // root class reads and writes through them at any depth. The receiver
    // may be any expression the inferrer types (`self.items[k].qty`, a
    // subscript into a dict of the root; an unwrapped `Option` local).
    if !is_self {
        // The receiver's class with its defining scope: the read path's
        // resolution, else the inferred type's class resolved by name.
        let resolved = crate::receiver_class_for_read(value, ctx, symbols, options).or_else(|| {
            // A NAME's recorded type first: the inferrer answers a
            // parameter from the annotation-string map, which knows no
            // classes (`response: BaseHTTPResponse | None` is the boxed
            // value there), while the analysis recorded the Option of
            // the class.
            let typed = match value {
                ExprType::Name(n) => options.name_types.get(&n.id).cloned(),
                _ => None,
            }
            .unwrap_or_else(|| crate::infer_type(Some(ctx), value, options, symbols));
            let name = match typed {
                crate::TypeInfo::Class(c) => c,
                crate::TypeInfo::Option(inner) => match *inner {
                    crate::TypeInfo::Class(c) => c,
                    _ => return None,
                },
                _ => return None,
            };
            crate::ast::tree::call::receiver_class_tail(&name, symbols.clone(), options)
                .or_else(|| crate::ast::tree::hierarchy::root_class_def(&name, symbols, options))
        });
        if let Some((class, class_symbols)) = resolved
            && crate::ast::tree::hierarchy::is_polymorphic_root(&class.name)
            // A FIELD of the root's chain (its accessors are what the sum
            // type carries); a method or property keeps the call path.
            && class.field_owner_depth(attr, &class_symbols, options).is_some()
        {
            return Some(FieldRewrite::Accessor {
                field: attr.to_string(),
            });
        }
    }
    let (class, class_symbols) = crate::receiver_class_for_read(value, ctx, symbols, options)?;
    let depth = class.field_owner_depth(attr, &class_symbols, options)?;
    if depth == 0 {
        // The receiver's own field. Direct access works for any concrete
        // receiver; only the generic `self` of a trait default needs the
        // accessor (the field is not on the generic Self).
        if is_self && ctx.in_generic_trait() {
            Some(FieldRewrite::Accessor {
                field: attr.to_string(),
            })
        } else {
            None
        }
    } else if is_self && ctx.in_generic_trait() {
        Some(FieldRewrite::AccessorChain { depth })
    } else {
        Some(FieldRewrite::Chain { depth })
    }
}

/// Whether `value` is a non-`self` receiver whose class is SHARED and not
/// a polymorphic root (a root's sum type reaches its fields through the
/// accessors, which borrow inside): its field reads and stores go through
/// the `PyRef` borrow (shared.rs).
pub(crate) fn shared_receiver(
    value: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    if crate::ast::tree::visit::is_self(value) {
        return false;
    }
    let class = match crate::receiver_class_for_read(value, ctx, symbols, options) {
        Some((c, _)) => c.name,
        None => {
            let typed = match value {
                ExprType::Name(n) => options.name_types.get(&n.id).cloned(),
                _ => None,
            }
            .unwrap_or_else(|| crate::infer_type(Some(ctx), value, options, symbols));
            match typed {
                crate::TypeInfo::Class(c) => c,
                crate::TypeInfo::Option(inner) => match *inner {
                    crate::TypeInfo::Class(c) => c,
                    _ => return false,
                },
                _ => return false,
            }
        }
    };
    crate::ast::tree::shared::is_shared(&class)
        && !crate::ast::tree::hierarchy::is_polymorphic_root(&class)
}

#[derive(Clone, Debug)]
pub(crate) enum FieldRewrite {
    Accessor { field: String },
    Chain { depth: usize },
    AccessorChain { depth: usize },
}

/// True when a dotted expression chain is rooted at a plain module import
/// (`import pylev` or `import pylev as p`): `pylev.fn`, `p.alias.sub.fn`.
/// The root name must be bound to an `Import` node — an ImportFrom binding
/// is a value (function/class), so `from x import fn; fn.attr` stays a
/// field access. Attribute chains recurse: each segment between the root
/// and the accessed attribute is itself a module path item, so
/// `pkg.sub.fn` resolves `pkg.sub` the same way.
pub(crate) fn is_module_path_chain(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    match expr {
        ExprType::Name(n) => {
            if crate::module_name_shadowed(&n.id, symbols) {
                return false;
            }
            match symbols.get(&n.id) {
                Some(SymbolTableNode::Import(_)) => true,
                Some(SymbolTableNode::Alias(canonical)) => {
                    // Follow the alias to the canonical name — which may be
                    // a submodule import (`from .. import connection as
                    // urllib3_connection`).
                    is_module_path_chain(
                        &ExprType::Name(crate::ast::tree::name::Name {
                            id: canonical.clone(),
                        }),
                        symbols,
                        options,
                    )
                }
                // A RELATIVE SUBMODULE import (`from . import exceptions`,
                // `from .util import ssl_`): the name IS a module when the
                // submodule exists in the crate — attribute reads resolve
                // as paths (`exceptions.SecurityWarning`,
                // `ssl_.ALPN_PROTOCOLS`). A sibling that re-exports a
                // STDPYTHON module (`from .compat import json as
                // complexjson` — requests' models.py, where compat.py does
                // `import json`) is ALSO a module path: the import lowered
                // to `use <stdpython>::json as complexjson;`, so
                // `complexjson.dumps(...)` resolves as `complexjson::dumps`.
                Some(SymbolTableNode::ImportFrom(ifm)) if ifm.level > 0 => {
                    let mut sub = ifm.resolved_module_path(options);
                    sub.push(n.id.clone());
                    if crate::module_defs_contains(options, &sub) {
                        return true;
                    }
                    let path = ifm.resolved_module_path(options);
                    crate::module_defs_key(options, &path)
                        .is_some_and(|key| {
                            crate::ast::tree::module::module_reexports_stdpython_module(
                                options,
                                key,
                                &n.id,
                            )
                            .is_some()
                        })
                }
                // An ABSOLUTE import whose name resolves to a crate
                // SUBMODULE (`from urllib3.contrib import pyopenssl` —
                // requests/__init__.py's dead pyopenssl branch): the name
                // is a module when `module + name` is a module of the
                // crate — attribute calls resolve as paths
                // (`pyopenssl::inject_into_urllib3`). A VALUE import
                // (`from .utils import make_headers`) is not a module
                // (`module + name` is not a crate module).
                Some(SymbolTableNode::ImportFrom(ifm)) if ifm.level == 0 => {
                    let mut sub = ifm.resolved_module_path(options);
                    sub.push(n.id.clone());
                    crate::module_defs_contains(options, &sub)
                }
                _ => false,
            }
        }
        ExprType::Attribute(a) => {
            // A SCREAMING_SNAKE terminal segment is a module CONSTANT,
            // not a submodule (`ssl.OPENSSL_VERSION.startswith(...)` —
            // urllib3's __init__): the chain ends at the constant, and a
            // further attribute is a method on its VALUE, not a path
            // segment.
            if !a.attr.is_empty()
                && a.attr.chars().any(|c| c.is_ascii_uppercase())
                && !a.attr.chars().any(|c| c.is_ascii_lowercase())
            {
                return false;
            }
            is_module_path_chain(&a.value, symbols, options)
        }
        _ => false,
    }
}

/// The class of a receiver expression, resolving through `self.field()`
/// accessor CALLS (`self._response().body` — the receiver of `.body` is
/// the `_response()` accessor, whose class is the `_response` field's
/// class). The `_mut` accessor spelling (`self._response_mut()`) strips
/// back to the field name.
fn receiver_class_deep(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    match expr {
        ExprType::Call(c) => {
            let ExprType::Attribute(a) = c.func.as_ref() else {
                return None;
            };
            if let Some(r) = crate::receiver_class(&ExprType::Attribute(a.clone()), ctx, symbols, options) {
                return Some(r);
            }
            if let Some(stripped) = a.attr.strip_suffix("_mut") {
                let mut a2 = a.clone();
                a2.attr = stripped.to_string();
                return crate::receiver_class(&ExprType::Attribute(a2), ctx, symbols, options);
            }
            None
        }
        _ => crate::receiver_class(expr, ctx, symbols, options),
    }
}

/// Whether an ATTRIBUTE chain ends at a PyValue-typed class field
/// (`self._response().body` — `body` is `pub body: PyValue` on the
/// emscripten response). Reads THROUGH such a chain are dynamic.
fn field_chain_ends_in_pyvalue(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Attribute(a) = expr else {
        return false;
    };
    let Some((class, class_symbols)) =
        receiver_class_deep(&a.value, ctx, symbols, options)
    else {
        return false;
    };
    let Ok(fields) = class.infer_fields(&class_symbols, options) else {
        return false;
    };
    fields
        .iter()
        .find(|(name, _)| name == &a.attr)
        .is_some_and(|(_, ty)| crate::ast::tree::type_ctx::type_contains_pyvalue(ty))
}

/// Whether an expression is a BOXED PyValue at runtime: a name with an
/// unknown/boxed type (but never `self`), a call with a known boxed return,
/// or a field chain ending at a PyValue-typed field. Attribute reads and
/// method calls ON such a receiver have no static shape — the
/// dynamic-attribute divergence.
/// Methods that exist on `stdpython::PyValue` itself — the boxed value's
/// own protocol (accessors, conversions) and the duck-typed method names
/// the runtime forwards. A call to one of these on a boxed receiver is
/// REAL code, not a dynamic-dispatch divergence, so it must never be
/// dropped.
///
/// One definition, used by both the call-side drop (call.rs) and the
/// read-side one below; they were separate copies, which is how the two
/// sides drift apart.
pub(crate) fn pyvalue_protocol_method(name: &str) -> bool {
    name.starts_with("is_")
        || name.starts_with("as_")
        || name.starts_with("py_")
        || matches!(
            name,
            "decode" | "encode" | "into_bytes_like" | "clone" | "to_string"
                | "split" | "rsplit" | "strip" | "lstrip" | "rstrip" | "join"
                | "lower" | "upper" | "startswith" | "endswith" | "replace"
                | "format" | "count" | "find" | "group" | "items" | "keys"
                | "values" | "get" | "append" | "pop" | "read" | "readline"
                | "close" | "getvalue" | "write"
        )
}

/// Whether a receiver is POSITIVELY boxed, as opposed to merely untyped.
///
/// `receiver_is_pyvalue` accepts `TypeInfo::PyObject` too, and `PyObject`
/// is the inferrer's "no answer" — so that helper cannot tell a boxed
/// value from one it failed to type. Round 24 widened a drop on the back
/// of it and discarded a live value; the revert note recorded the rule
/// this encodes: absence of evidence is not evidence of boxing.
///
/// `PyValue` is the boxed heterogeneous value (an `Any`/`object`/wide-union
/// annotation, or a join that genuinely disagreed) and `PyValueMember` is
/// that value narrowed by isinstance. Neither can be a concrete class: a
/// module global bound to `Klass()` infers `TypeInfo::Class`, which is
/// exactly the counterexample round 24 tripped over.
pub(crate) fn receiver_is_boxed_positively(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Name(n) = expr else {
        return false;
    };
    if n.id == "self" {
        return false;
    }
    matches!(
        crate::infer_type(None, expr, options, symbols),
        crate::TypeInfo::PyValue | crate::TypeInfo::PyValueMember(_)
    )
}

pub(crate) fn receiver_is_pyvalue(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    match expr {
        ExprType::Name(n) => {
            if n.id == "self" {
                return false;
            }
            matches!(
                crate::infer_type(Some(&ctx), expr, options, symbols),
                crate::TypeInfo::PyValue
                    | crate::TypeInfo::PyObject
                    | crate::TypeInfo::PyValueMember(_)
            )
        }
        ExprType::Attribute(_) => field_chain_ends_in_pyvalue(expr, ctx, symbols, options),
        ExprType::Call(c) => {
            matches!(
                crate::call_return_typeinfo(c, Some(symbols), Some(options)),
                Some(crate::TypeInfo::PyValue | crate::TypeInfo::PyObject)
            )
            // A call into an EXTERNAL module (`files("certifi")` —
            // importlib.resources) drops to the boxed None (the
            // external-module divergence): a method on its result is a
            // boxed-receiver call.
            || receiver_call_is_external_drop(expr, symbols, options)
        }
        _ => false,
    }
}

/// Whether a CALL expression is a call into an EXTERNAL module whose
/// result drops to the boxed None (`files("certifi")`,
/// `ssl.SSLContext(...)`) — so a method chained on its result is a
/// boxed-receiver call, not a static method lookup (certifi's
/// `files("certifi").joinpath(...).read_text(...)`).
pub(crate) fn receiver_call_is_external_drop(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Call(c) = expr else {
        return false;
    };
    crate::ast::tree::attribute::external_module_root(&c.func, symbols, options).is_some()
        || chain_root_is_external(&c.func, symbols, options)
}

/// Whether an expression is a READ of a BOXED mutable global — the value
/// is the boxed PyValue, which a typed (non-PyValue) return cannot
/// express; the exact point of divergence is a loud panic.
pub(crate) fn is_boxed_global_read(
    expr: &ExprType,
    options: &PythonOptions,
) -> bool {
    let ExprType::Name(n) = expr else {
        return false;
    };
    matches!(
        options.mutable_statics.get(&n.id),
        Some(crate::MutableGlobalKind::Boxed)
            | Some(crate::MutableGlobalKind::Computed { boxed: true })
    )
}

/// Whether an expression is a CALL CHAIN rooted in an EXTERNAL module
/// (`files("certifi").joinpath(...)`): the whole chain drops to the boxed
/// None (external-module divergence), so returning it from a typed
/// function is a loud runtime panic, not a placeholder.
pub(crate) fn is_external_drop_chain(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let ExprType::Call(c) = expr else {
        return false;
    };
    chain_root_is_external(&c.func, symbols, options)
}

/// Walk an expression chain (calls and attributes) to its root NAME and
/// ask whether that name resolves to an EXTERNAL module import — the
/// chain then drops to the boxed None, so a method chained on it is a
/// boxed-receiver call (`files("certifi").joinpath("cacert.pem").
/// read_text(...)` — certifi: the whole chain is importlib.resources).
fn chain_root_is_external(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    match expr {
        ExprType::Call(c) => chain_root_is_external(&c.func, symbols, options),
        ExprType::Attribute(a) => chain_root_is_external(&a.value, symbols, options),
        ExprType::Name(n) => {
            crate::ast::tree::import::resolves_to_external_import(&n.id, options, symbols)
        }
        _ => false,
    }
}

/// The root name of a module-path chain whose module is EXTERNAL — stdlib
/// rython does not model (ssl, socket, logging, http, zlib, threading,
/// codecs, ...) or a non-vendored dependency (socks) — resolved through the
/// symbol table to its module path and checked against the generated crate.
/// None when the chain is not a module path, is shadowed, or the module is
/// part of the crate / stdpython / a vendored dependency. Only meaningful
/// when the whole crate is known (multi-module conversion); single-module
/// conversions return None (any import may be a sibling).
pub(crate) fn external_module_root(
    expr: &ExprType,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<String> {
    // Only meaningful when the whole crate is known (multi-module
    // conversion); a single-module conversion (len == 1) only knows the
    // module itself, so any import may be a sibling.
    if options.module_defs.len() <= 1 {
        return None;
    }
    let root = match expr {
        ExprType::Name(n) => n.id.clone(),
        ExprType::Attribute(a) => return external_module_root(&a.value, symbols, options),
        _ => return None,
    };
    if crate::module_name_shadowed(&root, symbols) {
        return None;
    }
    let module_path: Option<Vec<String>> = match symbols.get(&root) {
        Some(SymbolTableNode::Import(im)) => im
            .names
            .first()
            .map(|al| al.name.split('.').map(|s| s.to_string()).collect()),
        Some(SymbolTableNode::Alias(canonical)) => match symbols.get(canonical) {
            Some(SymbolTableNode::Import(im)) => im
                .names
                .first()
                .map(|al| al.name.split('.').map(|s| s.to_string()).collect()),
            Some(SymbolTableNode::ImportFrom(ifm)) => Some(ifm.resolved_module_path(options)),
            _ => None,
        },
        Some(SymbolTableNode::ImportFrom(ifm)) => Some(ifm.resolved_module_path(options)),
        _ => None,
    };
    let Some(path) = module_path else {
        return None;
    };
    let first = path.first().map(|s| s.as_str()).unwrap_or("");
    if crate::ast::tree::import::is_stdpython_module(first) {
        return None;
    }
    if crate::module_defs_contains(options, &path) {
        return None;
    }
    if options.python_modules.contains(first) {
        return None;
    }
    Some(root)
}

/// The Option's INNER Rust type when an expression's runtime value is
/// `Option<T>` (issue #137's Option-aware access): a name whose
/// recorded type is `TypeInfo::Option`, a `self.<field>` read whose
/// field the class table types Option, a deeper chain (recurse to the
/// root), or a call whose return annotation is Optional. None for
/// anything else — a non-Option receiver never unwraps. Access through
/// an Option receiver is CPython's AttributeError-on-None, which rython
/// lowers as a loud §12.2 panic at the access site.
/// Whether `expr` is a name NARROWED by the enclosing guard (`response
/// and response.get_redirect_location()` — urllib3's Retry.increment;
/// an `is not None` test; an isinstance): its READ (name.rs) already
/// renders the non-Option value, so a non-mutating call takes it as is —
/// a second unwrap would be on a value that is no longer an Option. A
/// MUTATING call renders the name as a place (the Option binding itself)
/// and still unwraps.
pub(crate) fn narrowed_name_read(expr: &ExprType, options: &PythonOptions) -> bool {
    matches!(expr, ExprType::Name(n) if options.narrowed_names.contains_key(&n.id))
}

pub(crate) fn receiver_option_inner(
    expr: &ExprType,
    ctx: &CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<proc_macro2::TokenStream> {
    match expr {
        ExprType::Name(n) => match options.name_types.get(&n.id) {
            Some(crate::TypeInfo::Option(inner)) => Some(inner.to_rust_type()),
            // A None-first local (`conn = None` then `conn = ...` —
            // urllib3's _get_conn) is an Option binding; the inner type
            // is whatever the stores join to — the unwrap never needs it.
            // A BOXED name (PyValue — the None-mixing path stores
            // PyValue::None_, not Some) is NOT an Option: unwrapping it
            // would be wrong (no is_some/unwrap on PyValue).
            Some(crate::TypeInfo::PyValue) => None,
            _ if options.optional_names.contains(&n.id) => Some(quote!(_)),
            _ => None,
        },
        ExprType::Attribute(attr) => {
            // `self.<field>` (or `<obj>.<field>`): the field's inferred
            // Rust type comes from the OWNER class's field table.
            let owner: Option<(crate::ClassDef, SymbolTableScopes)> = match attr.value.as_ref() {
                ExprType::Name(n) if n.id == "self" => {
                    let class_name = ctx.enclosing_class_name()?;
                    match symbols.get(class_name) {
                        Some(crate::SymbolTableNode::ClassDef(c)) => {
                            Some((c.clone(), symbols.clone()))
                        }
                        _ => None,
                    }
                }
                other => crate::receiver_class(other, ctx, symbols, options),
            };
            if let Some((class, class_symbols)) = owner {
                let key = (class.name.clone(), attr.attr.clone());
                if let Some(cached) = options
                    .option_field_cache
                    .borrow()
                    .get(&key)
                    .cloned()
                {
                    return if cached { Some(quote!(_)) } else { None };
                }
                // The field may be INHERITED (`self.proxy` on
                // HTTPConnectionPool, whose base ConnectionPool stores it
                // in its own __init__): search the whole base chain's
                // field tables.
                let mut is_option = false;
                for c in class.base_chain(&class_symbols) {
                    if let Ok(fields) = c.infer_fields(&class_symbols, options) {
                        if let Some((_, ty)) =
                            fields.iter().find(|(name, _)| *name == attr.attr)
                        {
                            is_option = matches!(ty, crate::TypeInfo::Option(_));
                            break;
                        }
                    }
                }
                options
                    .option_field_cache
                    .borrow_mut()
                    .insert(key, is_option);
                return if is_option { Some(quote!(_)) } else { None };
            }
            // A deeper chain: recurse to the base (`a.b.c` — is `a.b`
            // Option?).
            receiver_option_inner(&attr.value, ctx, symbols, options)
        }
        ExprType::Call(call) => {
            let t = crate::call_return_typeinfo(call, Some(symbols), Some(options))?;
            match t {
                crate::TypeInfo::Option(inner) => Some(inner.to_rust_type()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether a RETURN value is a method call on a boxed-PyValue receiver
/// (`return self._obj.decompress(...)` — urllib3's DeflateDecoder, where
/// `self._obj` is an unmodeled zlib object): call.rs DROPS such calls to
/// the boxed None (dynamic-method divergence), so returning one from a
/// TYPED function would emit `Ok(PyValue::None_)` in a `Vec<u8>`-typed
/// fn — a loud runtime panic at the exact point of divergence instead
/// (round 80), mirroring the external-module-drop return.
pub(crate) fn dropped_boxed_receiver_call(
    value: &crate::ExprType,
    ctx: &crate::CodeGenContext,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let crate::ExprType::Call(c) = value else {
        return false;
    };
    let crate::ExprType::Attribute(a) = c.func.as_ref() else {
        return false;
    };
    // The EXACT condition the call lowering uses to drop a boxed-receiver
    // call (protocol methods survive; module members do not drop).
    crate::ast::tree::call::boxed_receiver_method_dropped(a, ctx, symbols, options)
}
