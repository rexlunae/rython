use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods, types::PyTypeMethods};
use quote::quote;

use crate::{
    CodeGen, CodeGenContext, ExprType, PythonOptions, SymbolTableNode, SymbolTableScopes,
    extraction_failure,
};

use serde::{Deserialize, Serialize};

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
        // Inheritance-aware field access, computed before `self.value` is
        // moved below: `self.name` where `name` is a base class's field, or
        // `dog.name` where `dog` is a derived-class instance, must reach
        // through the embedded base structs (or the trait's base accessors).
        let field_access = class_field_access(&self.value, &self.attr, &ctx, &symbols, &options);
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
        let except_binding_receiver: Option<String> = match self.value.as_ref() {
            ExprType::Name(receiver) => match symbols.get(&receiver.id) {
                Some(crate::SymbolTableNode::ExceptBinding) => {
                    Some(receiver.id.clone())
                }
                _ => None,
            },
            _ => None,
        };
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
        // A PROPERTY GETTER read (`self.url` — urllib3's geturl, where url
        // is `@property def url`): the property lowers as a plain method
        // returning Result, so the read routes to the getter CALL and
        // unwraps (`self.url()?`). Computed before the moves below.
        let property_getter = crate::receiver_class(&self.value, &ctx, &symbols, &options)
            .is_some_and(|(class, _)| class.has_property_getter(&self.attr));
        let warnings = options.definition_warnings.clone();
        let value_tokens = self.value.to_rust(ctx, options, symbols)?;
        let value_str = value_tokens.to_string();
        let attr = crate::safe_ident(&self.attr);
        // Debug-form of the receiver for -W messages (captured before the
        // move above).
        let value_debug = format!("{:?}", value_str);

        // Determine if this is a module access or a field/method access
        // Module names are typically lowercase and match Python stdlib modules
        // `np`/`numpy` cover the numpy module (import numpy as np lowers to
        // `use stdpython::numpy as np`, making np a real Rust path).
        // A dotted chain rooted at a plain module `import` (`import pylev`)
        // is also a module path: `pylev.wfi_levenshtein` emits
        // `pylev::wfi_levenshtein`, and nested chains (`pylev.sub.fn`) emit
        // `pylev::sub::fn` because each inner attribute resolves the same
        // way. ImportFrom bindings are VALUES (a function/class), not
        // modules, so they never turn attribute access into a path.
        let is_module_access = !root_shadowed
            && (matches!(
                value_str.as_str(),
                "sys" | "os" | "subprocess" | "json" | "urllib" | "xml" | "asyncio" |
            "time" | "math" | "random" | "heapq" | "functools" | "textwrap" | "itertools" | "re" | "hashlib" | "csv" | "io" |
            // `datetime` covers both the runtime module and the datetime
            // TYPE from `from datetime import datetime` — either way the
            // attribute is a path item (datetime::strptime, datetime::now),
            // never a field on a value.
            "datetime" |
            "numpy" | "np" |
            "os :: path" | "os::path" | // for nested modules
            "numpy :: linalg" | "np :: linalg" | "numpy::linalg" | "np::linalg" // np.linalg.inv
            ) || module_chain);

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
            // unit struct whose methods auto-ref.
            let needs_deref = matches!(
                (value_str.as_str(), self.attr.as_str()),
                ("sys", "executable") | ("sys", "argv")
            );

            // `sys.modules` — the process's import registry (requests'
            // packages.py aliasing): rython's crate is static, so the
            // registry is always empty — the read lowers to an empty dict
            // (list(sys.modules) iterates nothing, indexing misses).
            if value_str == "sys" && self.attr == "modules" {
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
            if pyvalue_receiver && receiver_is_field_chain {
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
            if let Some(receiver) = &except_binding_receiver {
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
                return Ok(quote!(#value_tokens.#attr()?));
            }
            match field_access {
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
                    let chain = base_accessor_chain(depth);
                    Ok(quote!(#value_tokens #chain.#attr()))
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
                }) || matches!(
                    value_tokens.to_string().as_str(),
                    "sys" | "os" | "subprocess" | "json" | "urllib" | "xml" | "asyncio"
                        | "time" | "math" | "random" | "heapq" | "functools" | "textwrap"
                        | "itertools" | "re" | "hashlib" | "csv" | "io" | "datetime"
                        | "numpy" | "np"
                ));
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
                    let chain = base_mut_accessor_chain(depth);
                    Ok(quote!(#recv_place #chain.#attr_ident))
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
    let (class, class_symbols) = crate::receiver_class(value, ctx, symbols, options)?;
    let depth = class.field_owner_depth(attr, &class_symbols)?;
    let is_self = matches!(value, ExprType::Name(n) if n.id == "self");
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
                    if options.module_defs.contains_key(&sub) {
                        return true;
                    }
                    let path = ifm.resolved_module_path(options);
                    options.module_defs.contains_key(&path)
                        && crate::ast::tree::module::module_reexports_stdpython_module(
                            options,
                            &path,
                            &n.id,
                        )
                        .is_some()
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
                    options.module_defs.contains_key(&sub)
                }
                _ => false,
            }
        }
        ExprType::Attribute(a) => is_module_path_chain(&a.value, symbols, options),
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
        .is_some_and(|(_, ty)| ty.to_string().contains("PyValue"))
}

/// Whether an expression is a BOXED PyValue at runtime: a name with an
/// unknown/boxed type (but never `self`), a call with a known boxed return,
/// or a field chain ending at a PyValue-typed field. Attribute reads and
/// method calls ON such a receiver have no static shape — the
/// dynamic-attribute divergence.
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
                crate::infer_type(expr, options, symbols),
                crate::TypeInfo::PyValue
                    | crate::TypeInfo::PyObject
                    | crate::TypeInfo::PyValueMember(_)
            )
        }
        ExprType::Attribute(_) => field_chain_ends_in_pyvalue(expr, ctx, symbols, options),
        ExprType::Call(c) => matches!(
            crate::call_return_typeinfo(c, Some(symbols), Some(options)),
            Some(crate::TypeInfo::PyValue | crate::TypeInfo::PyObject)
        ),
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
    if options.module_defs.contains_key(&path) {
        return None;
    }
    if options.python_modules.contains(first) {
        return None;
    }
    Some(root)
}
