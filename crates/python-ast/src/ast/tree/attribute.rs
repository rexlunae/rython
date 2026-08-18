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
    ctx: String,
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
        let module_chain = is_module_path_chain(&self.value, &symbols);
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
        let value_tokens = self.value.to_rust(ctx, options, symbols)?;
        let value_str = value_tokens.to_string();
        let attr = crate::safe_ident(&self.attr);

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
            // Use :: for module access (Python's sys.executable becomes sys::executable)
            // Special handling for LazyLock static variables that need
            // dereferencing. os::environ is NOT here: it is a live-view
            // unit struct whose methods auto-ref.
            let needs_deref = matches!(
                (value_str.as_str(), self.attr.as_str()),
                ("sys", "executable") | ("sys", "argv")
            );

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
            // Use . for field/method access (Python's obj.field becomes obj.field).
            // A class field owned by an ancestor of the receiver's class is
            // reached through the embedded base structs (`self.__rython_base.f`)
            // or, in a generic trait default, through the base accessors
            // (`self.base().f`); an own field in a generic trait default goes
            // through its accessor (`self.f()`).
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
    let attribute = Attribute {
        value: Box::new(value.clone()),
        attr: attr.to_string(),
        ctx: "Store".to_string(),
    };
    let field_access = class_field_access(value, attr, ctx, symbols, options);
    // Mirror the module-path guard from to_rust: a store target that
    // resolves as a module path is not a field access. A user binding that
    // shadows the module name (`os = {...}`) must NOT be treated as a
    // module — the load path applies the same `module_name_shadowed` guard.
    let root_shadowed = crate::ast::tree::call::root_name(value)
        .is_some_and(|root| crate::module_name_shadowed(root, symbols));
    let value_tokens = attribute.value.to_rust(ctx.clone(), options.clone(), symbols.clone())?;
    let is_module = !root_shadowed
        && (crate::ast::tree::call::root_name(value)
            .is_some_and(|_root| crate::ast::tree::attribute::is_module_path_chain(value, symbols))
            || matches!(
                value_tokens.to_string().as_str(),
                "sys" | "os" | "subprocess" | "json" | "urllib" | "xml" | "asyncio" | "time"
                    | "math" | "random" | "heapq" | "functools" | "textwrap" | "itertools" | "re"
                    | "hashlib" | "csv" | "io" | "datetime" | "numpy" | "np"
            ));
    if is_module {
        // Module stores render as PATHS, exactly like the load form:
        // `os.environ[k] = v` stores through `os::environ`. The attribute
        // is an identifier — interpolating the &str directly would emit a
        // string literal (`os . "environ"`) that the Rust compiler rejects.
        let attr_path = crate::safe_ident(attr);
        // Vendored deps prefix `crate::` at the ROOT segment only; nested
        // segments already carry it from their own rendering.
        let vendored_root = matches!(value, ExprType::Name(_))
            && crate::ast::tree::call::root_name(value)
                .is_some_and(|root| options.python_modules.contains(root));
        if vendored_root {
            return Ok(quote!(crate::#value_tokens::#attr_path));
        }
        return Ok(quote!(#value_tokens::#attr_path));
    }
    let attr_ident = crate::safe_ident(attr);
    match field_access {
        None => Ok(quote!(#value_tokens.#attr_ident)),
        Some(FieldRewrite::Accessor { field }) => {
            let accessor = crate::safe_ident(&format!("{}_mut", field));
            if deref {
                Ok(quote!(*#value_tokens.#accessor()))
            } else {
                Ok(quote!(#value_tokens.#accessor()))
            }
        }
        Some(FieldRewrite::Chain { depth }) => {
            let chain = base_field_chain(depth);
            Ok(quote!(#value_tokens #chain.#attr_ident))
        }
        Some(FieldRewrite::AccessorChain { depth }) => {
            let chain = base_mut_accessor_chain(depth);
            Ok(quote!(#value_tokens #chain.#attr_ident))
        }
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
pub(crate) fn is_module_path_chain(expr: &ExprType, symbols: &SymbolTableScopes) -> bool {
    match expr {
        ExprType::Name(n) => {
            if crate::module_name_shadowed(&n.id, symbols) {
                return false;
            }
            match symbols.get(&n.id) {
                Some(SymbolTableNode::Import(_)) => true,
                Some(SymbolTableNode::Alias(canonical)) => {
                    matches!(symbols.get(canonical), Some(SymbolTableNode::Import(_)))
                }
                _ => false,
            }
        }
        ExprType::Attribute(a) => is_module_path_chain(&a.value, symbols),
        _ => false,
    }
}
