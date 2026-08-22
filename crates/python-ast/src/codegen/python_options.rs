//! Options for Python compilation.

use std::{
    collections::{BTreeMap, HashSet},
    default::Default,
};

use pyo3::{PyResult, prelude::*};
use std::ffi::CString;

use crate::TypeInfo;

/// Supported async runtimes for Python async code generation
#[derive(Clone, Debug, PartialEq)]
pub enum AsyncRuntime {
    /// Tokio runtime (default)
    Tokio,
    /// async-std runtime
    AsyncStd,
    /// smol runtime
    Smol,
    /// Custom runtime with specified attribute and import
    Custom {
        /// The attribute to use (e.g., "tokio::main", "async_std::main")
        attribute: String,
        /// The import to add (e.g., "tokio", "async_std")
        import: String,
    },
}

/// The cargo feature name on generated BINARY crates that pulls in the
/// async runtime (tokio). Generated code gates its runtime import and the
/// entry-point attribute on this feature, and rypip declares the feature in
/// the generated Cargo.toml (`default = ["async-tokio"]`). Must match
/// between codegen (python-ast) and the crate writer (rypip).
pub const ASYNC_RUNTIME_FEATURE: &str = "async-tokio";

impl Default for AsyncRuntime {
    fn default() -> Self {
        AsyncRuntime::Tokio
    }
}

impl AsyncRuntime {
    /// Get the attribute string for the async main function
    pub fn main_attribute(&self) -> &str {
        match self {
            AsyncRuntime::Tokio => "tokio::main",
            AsyncRuntime::AsyncStd => "async_std::main",
            AsyncRuntime::Smol => "smol::main",
            AsyncRuntime::Custom { attribute, .. } => attribute,
        }
    }

    /// Get the import string for the runtime
    pub fn import(&self) -> &str {
        match self {
            AsyncRuntime::Tokio => "tokio",
            AsyncRuntime::AsyncStd => "async_std",
            AsyncRuntime::Smol => "smol",
            AsyncRuntime::Custom { import, .. } => import,
        }
    }
}

pub fn sys_path() -> PyResult<Vec<String>> {
    let pymodule_code = include_str!("path.py");

    Python::attach(|py| -> PyResult<Vec<String>> {
        let code_cstr = CString::new(pymodule_code)?;
        let pymodule = PyModule::from_code(py, &code_cstr, c"path.py", c"path")?;
        let t = pymodule.getattr("path")?;
        assert!(t.is_callable());
        let args = ();
        let paths: Vec<String> = t.call1(args)?.extract()?;

        Ok(paths)
    })
}

/// State of the one-time cross-module trait-mut table cache (see
/// `PythonOptions::cross_module_mut_self`).
#[derive(Clone, Debug)]
pub enum CrossModuleMutSelf {
    /// No module has hit the cross-module fallback yet.
    Uncomputed,
    /// A scan is in progress; re-entrant fallbacks must NOT recompute.
    Computing,
    /// The merged table, keyed by root class name → mutating method names.
    Computed(std::rc::Rc<std::collections::HashMap<String, std::collections::HashSet<String>>>),
}

impl CrossModuleMutSelf {
    /// The merged table, when computed.
    pub fn computed(
        &self,
    ) -> Option<&std::rc::Rc<std::collections::HashMap<String, std::collections::HashSet<String>>>> {
        match self {
            CrossModuleMutSelf::Computed(table) => Some(table),
            _ => None,
        }
    }
}

/// Class-resolution facts for one module of the crate, built once per
/// conversion and shared through [`PythonOptions::cross_module_classes`].
///
/// Replaces the per-call-site `ast.clone().find_symbols(...)` in
/// [`crate::module_class_def`] / [`crate::module_class_traits`]: resolving
/// an imported class (its fields, methods, construction, and the traits its
/// methods live on) no longer deep-clones the whole defining module AST and
/// rebuilds its symbol table for EVERY attribute access / method call /
/// construction.
#[derive(Clone, Debug)]
pub struct ModuleClassInfo {
    /// The module's own symbol table: base chains must resolve inside the
    /// module that DECLARED the class, not the importer's scope.
    pub symbols: crate::SymbolTableScopes,
    /// Top-level classes by name, for cross-module construction
    /// (`from .animals import Dog; Dog("Rex")` maps arguments against the
    /// defining module's `__init__`).
    pub classes: std::collections::HashMap<String, crate::ClassDef>,
    /// Hierarchy classes → trait names (`{Name}Trait` plus ancestors'),
    /// for the cross-module trait imports the import statement emits.
    pub traits: std::collections::HashMap<String, Vec<String>>,
}

/// State of the one-time cross-module class-resolution cache (see
/// [`PythonOptions::cross_module_classes`]).
#[derive(Clone, Debug)]
pub enum CrossModuleClasses {
    /// No module has hit the cross-module class fallback yet.
    Uncomputed,
    /// The per-module class facts, keyed by module path (each value an
    /// `Rc` so lookups clone the handle, not the symbol table).
    Computed(
        std::rc::Rc<
            std::collections::HashMap<Vec<String>, std::rc::Rc<ModuleClassInfo>>,
        >,
    ),
}

/// The global context for Python compilation.
#[derive(Clone, Debug)]
pub struct PythonOptions {
    /// Python imports are mapped into a given namespace that can be changed.
    pub python_namespace: String,

    /// The default path we will search for Python modules.
    pub python_path: Vec<String>,

    /// Collects all of the things we need to compile imports[module][asnames]
    pub imports: BTreeMap<String, HashSet<String>>,

    pub stdpython: String,
    pub with_std_python: bool,

    pub allow_unsafe: bool,

    /// The async runtime to use for async Python code
    pub async_runtime: AsyncRuntime,

    /// Whether the generated crate declares and links the async runtime
    /// dependency (generated BINARY crates with async code; the tokio crate
    /// behind the `async-tokio` feature). When false — library conversions —
    /// async functions transpile to plain `async fn`s with no runtime
    /// import or entry attribute: the consumer supplies the executor.
    pub async_runtime_dep: bool,

    /// The inferred type-variable name for each unannotated parameter of the
    /// CURRENT function (issue #109, M1): `a` → `A`. Parameter rendering
    /// consults this instead of emitting the dead `impl Into<PyObject>`
    /// fallback. Set per function by the function generator.
    pub param_type_vars: std::rc::Rc<std::collections::HashMap<String, proc_macro2::TokenStream>>,

    /// Unannotated parameters of the CURRENT function whose uses include a
    /// stdlib method call (issue #109, M2): their method calls dispatch
    /// through the stdlib trait (e.g. `p.pop()` → `py_pop`), never through
    /// concrete-type arms that assume a Vec/str receiver.
    pub param_method_params: std::rc::Rc<std::collections::HashSet<String>>,

    /// Module-level generated items collected during codegen (issue #109,
    /// M3): duck-typing traits (HasSpeak) and their per-class impls. The
    /// module generator drains this at the end and emits the items at the
    /// top of the module output.
    pub module_pending_items:
        std::rc::Rc<std::cell::RefCell<Vec<proc_macro2::TokenStream>>>,

    /// Duck-typing trait names already generated in this module (issue
    /// #109, M3): `HasSpeak` and peers are emitted once per module even
    /// when several functions bound parameters on the same method.
    pub generated_duck_traits: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,

    /// Duck-typed user-method calls on the CURRENT function's unannotated
    /// parameters: param name → set of method names whose Has* trait
    /// returns Result (so the call sites thread `?`).
    pub duck_methods_on_params:
        std::rc::Rc<std::collections::HashMap<String, std::collections::HashSet<String>>>,

    /// The CURRENT function's unannotated parameters (and loop elements)
    /// CALLED as functions (`callback(...)` — s3transfer's
    /// invoke_progress_callbacks): the callable-as-value divergence (#122) —
    /// such calls lower to a dropped no-op at the call site.
    pub called_params: std::rc::Rc<std::collections::HashSet<String>>,

    /// Emit #[deprecated] notes on generated items whose conversion was
    /// lossy (dropped parameter defaults, ignored return annotations, ...).
    /// On by default: silent semantic divergence from the Python source is
    /// exactly what these warnings exist to surface. Tools may disable this
    /// to suppress the warnings at the user's explicit request.
    pub lossy_warnings: bool,
    /// Issue #110: names bound to a string literal and later rebound by a
    /// String-producing expression (`out = ""; out += "x"`) — their
    /// literal assignments are owned (`"".to_string()`).
    pub owned_str_literals: std::rc::Rc<std::collections::HashSet<String>>,
    /// M5 definition-time warnings collected during inference (a bound set
    /// no known type satisfies). The transpiler drains this after codegen;
    /// shared across option clones so pushes land in one place.
    pub definition_warnings: std::rc::Rc<std::cell::RefCell<Vec<String>>>,

    /// Names in the CURRENT scope that hold an Option (assigned None on
    /// some path, or annotated Optional): non-None stores into them wrap
    /// in Some. Set per scope by the function/module generators.
    pub optional_names: std::rc::Rc<std::collections::HashSet<String>>,
    /// Names whose Option-ness is statically narrowed away at the CURRENT
    /// point (issue #125): inside `if x is not None:`, and after an if/else
    /// where both branches leave x holding a non-None value. Reads of a
    /// narrowed name unwrap (`(x).clone().unwrap()`); their type is the
    /// Option's inner type. Threaded by the function body loop and by
    /// If::to_rust (the body narrows from the test).
    pub narrowed_names: std::rc::Rc<std::collections::HashMap<String, crate::TypeInfo>>,
    /// Generator lowering (issue #122-family): when set, `yield x` in a
    /// function body renders as `push(x)` on this collector and `yield
    /// from xs` as `extend(xs)`; the function codegen emits the collector
    /// Vec and returns it (a generator builds-and-returns its list).
    pub generator_collector: std::rc::Rc<Option<String>>,
    /// When rendering a dict literal, force its key/value types (issue
    /// #121): a store into a `dict[str, Any]` name sets this to
    /// (String, PyValue) so mixed values wrap per element, and a
    /// `dict[str, str]` annotation pins the types of an all-spread
    /// literal (`{**aliases, **{...}}`). None = infer from the literal.
    pub dict_forced_kv: std::rc::Rc<Option<(crate::TypeInfo, crate::TypeInfo)>>,

    /// Whether the CURRENT function's return annotation is `str`: returning
    /// an attribute chain then clones the String field out of the shared
    /// receiver. Python strings are immutable, so the clone reproduces
    /// Python's aliasing semantics exactly. Set per function.
    pub clone_str_attribute_returns: bool,

    /// The CURRENT function's resolved return type is the boxed PyValue:
    /// `return None` lowers to `PyValue::None_` and other returns wrap in
    /// `PyValue::from` (the None-mixing unification).
    pub fn_return_is_pyvalue: bool,

    /// The current module's package path within the generated crate
    /// ("" for the crate root, "pkg" for pkg/__init__.py, "pkg.sub" for
    /// pkg/sub/module.py). Relative imports (`from .x import y`,
    /// `from ..x import y`) resolve against it; the empty default keeps
    /// absolute-import behavior unchanged for any conversion that does not
    /// set it.
    pub module_path: Vec<String>,

    /// Statically-known types of names in the CURRENT scope (parameter
    /// annotations and literal assignments), as canonical Python type
    /// names ("int", "float", "str", "bool"). Set per function; consumed
    /// by isinstance(), which lowers to a constant.
    pub local_types: std::rc::Rc<std::collections::HashMap<String, String>>,

    /// Target stdpython's no_std (alloc) tier. Python constructs that need
    /// the OS — print/input/open, imports of os/datetime/random/…, and
    /// `__main__` entry points — fail loudly at conversion time instead of
    /// surfacing as resolution errors when the generated crate builds.
    pub no_std: bool,

    /// Force a numpy execution backend for the generated program
    /// ("scalar", "rayon", "simd", "cuda", "vulkan"), overriding the
    /// `RYPY_NUMPY_BACKEND` env var and np.set_backend defaults. The
    /// generated crate must be built with the matching stdpython feature
    /// (numpy-rayon, numpy-simd, ...) or the emitted set_backend call
    /// fails loudly at startup.
    pub numpy_backend: Option<String>,

    /// Class names in this module that participate in an inheritance
    /// hierarchy (have a real base, or are used as a base by another
    /// class). Those classes lower to struct + trait + impls instead of a
    /// plain struct, so `self.helper()` calls can dispatch through the
    /// trait. Set once per module; empty outside module generation.
    pub hierarchy_classes: std::rc::Rc<std::collections::HashSet<String>>,

    /// Trait method names whose signature must be `&mut self`, keyed by the
    /// owning (root) class of the trait. A method's trait signature widens
    /// to `&mut self` when ANY definition in the hierarchy mutates self:
    /// overrides re-emit into the root's trait, whose signature must fit
    /// every impl, and call sites must borrow accordingly. Set once per
    /// module; empty outside module generation (call sites then fall back
    /// to walking the receiver's own chain).
    pub trait_mut_self:
        std::rc::Rc<std::collections::HashMap<String, std::collections::HashSet<String>>>,

    /// Rust move error on reuse.
    pub use_counts: std::rc::Rc<std::collections::HashMap<String, usize>>,

    /// Inferred type of each local name in the current scope: parameter
    /// annotations first, then the type of the (last) literal or container
    /// assignment. Consumed by the type-aware lowering to insert
    /// conversions (String ↔ &str, usize → i64) at use sites.
    pub name_types: std::rc::Rc<std::collections::HashMap<String, TypeInfo>>,

    /// Names bound to an empty `[]`/`{}` literal whose element/key types
    /// were pinned by a later use; maps to the pinned container type.
    /// Rendering the empty literal consults this map and emits a typed
    /// `Vec::<T>::new()` / `PyDict::<K,V>::from([])`; a name assigned an
    /// empty literal with no pinning use is a loud conversion-time error.
    pub empty_pinned: std::rc::Rc<std::collections::HashMap<String, TypeInfo>>,

    /// Names whose bindings are managed by the enclosing scope's prologue
    /// (hoisted assignments plus mutable parameters): a `for`-loop target
    /// on one of these lowers to a plain store into that binding instead
    /// of a fresh binding that would shadow it, so Python's function-
    /// scoped loop-variable leak survives codegen (issue #80). Set per
    /// scope by the function/module generators.
    pub hoisted_names: std::rc::Rc<std::collections::HashSet<String>>,
    /// `for`-target names whose post-loop value is actually observed (a
    /// read in a later statement that no re-binding shadows). Only these
    /// lower to stores into the hoisted binding; a target that merely
    /// shares a name with a hoisted variable for other reasons keeps its
    /// fresh per-loop binding (issue #80).
    pub leaked_loop_targets: std::rc::Rc<std::collections::HashSet<String>>,
    /// Locals in the current function whose only known type is a string
    /// literal (`label = "fine"`), so they lower to `&'static str`. A
    /// `-> str` function returning one must own the string (`to_string`)
    /// to match its String return type. Set alongside
    /// `clone_str_attribute_returns` by the function generator.
    pub str_literal_locals: std::rc::Rc<std::collections::HashSet<String>>,

    /// Rust modules available to `import` / `from ... import` as
    /// compile-time bindings, keyed by the Python-side import name. The
    /// frontend (rypip / rythonc) populates this from the `rython.toml`
    /// manifest; codegen resolves import statements against it and inserts
    /// `SymbolTableNode::RustModule` symbols. Empty in library use.
    pub rust_modules: std::rc::Rc<std::collections::HashMap<String, crate::RustModuleSpec>>,

    /// Import names backed by vendored Python modules (`[python-modules]`
    /// in rython.toml): `import pylev` lowers to `use crate::pylev;` — a
    /// sibling module of the generated crate, not an external dependency.
    pub python_modules: std::rc::Rc<std::collections::HashSet<String>>,

    /// Cross-module class knowledge: parsed ASTs of the sibling modules of
    /// the generated crate (vendored `[python-modules]` deps and the
    /// package's own modules), keyed by module path. ImportFrom lowering
    /// consults them so a hierarchy class imported from another module
    /// works end to end: the class's traits are brought into scope (`use`
    /// alongside the class import — Rust method resolution needs the trait
    /// at the call site, and the class's own module defines it), and
    /// construction (`Dog("Rex")`) resolves the imported name to its
    /// defining `ClassDef` for signature-mapped `Dog::new(...)`. Empty in
    /// single-module conversion.
    pub module_defs:
        std::rc::Rc<std::collections::HashMap<Vec<String>, std::rc::Rc<crate::Module>>>,

    /// Lazily-computed merged trait-mut table over ALL modules of the crate
    /// (`module_defs`), shared across every module's conversion: the
    /// cross-module fallback in `method_needs_mut_self` scans each module
    /// AST once per CONVERSION instead of once per call site. `Computing`
    /// marks the in-progress one-time build so re-entrant fallbacks (the
    /// scan's own analysis consults `method_needs_mut_self` again) return
    /// false and fall through to the direct chain walk instead of
    /// recomputing. Per-module conversions (empty `module_defs`) never
    /// touch it.
    pub cross_module_mut_self:
        std::rc::Rc<std::cell::RefCell<CrossModuleMutSelf>>,

    /// Lazily-computed per-module class facts over ALL modules of the crate
    /// (`module_defs`), shared across every module's conversion: resolving
    /// an imported class's fields, methods, construction, and traits consults
    /// this once-built map instead of deep-cloning the defining module AST
    /// and rebuilding its symbol table per call site. Keyed by module path
    /// (the same `Vec<String>` keys as `module_defs`). Per-module
    /// conversions (empty `module_defs`) never touch it.
    pub cross_module_classes:
        std::rc::Rc<std::cell::RefCell<CrossModuleClasses>>,
}

impl Default for PythonOptions {
    fn default() -> Self {
        Self {
            python_namespace: String::from("__python_namespace__"),
            // Default must not panic: fall back to an empty search path if
            // the embedded interpreter can't report sys.path.
            python_path: sys_path().unwrap_or_else(|e| {
                tracing::warn!("could not read Python sys.path: {}; using empty path", e);
                Vec::new()
            }),
            imports: BTreeMap::new(),

            stdpython: "stdpython".to_string(),
            with_std_python: true,
            allow_unsafe: false,
            async_runtime: AsyncRuntime::default(),
            async_runtime_dep: false,
            param_type_vars: std::rc::Rc::new(std::collections::HashMap::new()),
            param_method_params: std::rc::Rc::new(std::collections::HashSet::new()),
            module_pending_items: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            generated_duck_traits: std::rc::Rc::new(std::cell::RefCell::new(
                std::collections::HashSet::new(),
            )),
            duck_methods_on_params: std::rc::Rc::new(std::collections::HashMap::new()),
            called_params: std::rc::Rc::new(std::collections::HashSet::new()),
            lossy_warnings: true,
            owned_str_literals: std::rc::Rc::new(std::collections::HashSet::new()),
            definition_warnings: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            optional_names: std::rc::Rc::new(std::collections::HashSet::new()),
            narrowed_names: std::rc::Rc::new(std::collections::HashMap::new()),
            dict_forced_kv: std::rc::Rc::new(None),
            generator_collector: std::rc::Rc::new(None),
            clone_str_attribute_returns: false,
            fn_return_is_pyvalue: false,
            module_path: Vec::new(),
            local_types: std::rc::Rc::new(std::collections::HashMap::new()),
            no_std: false,
            numpy_backend: None,
            hierarchy_classes: std::rc::Rc::new(std::collections::HashSet::new()),
            trait_mut_self: std::rc::Rc::new(std::collections::HashMap::new()),
            use_counts: std::rc::Rc::new(std::collections::HashMap::new()),
            name_types: std::rc::Rc::new(std::collections::HashMap::new()),
            empty_pinned: std::rc::Rc::new(std::collections::HashMap::new()),
            hoisted_names: std::rc::Rc::new(std::collections::HashSet::new()),
            leaked_loop_targets: std::rc::Rc::new(std::collections::HashSet::new()),
            str_literal_locals: std::rc::Rc::new(std::collections::HashSet::new()),
            rust_modules: std::rc::Rc::new(std::collections::HashMap::new()),
            python_modules: std::rc::Rc::new(std::collections::HashSet::new()),
            module_defs: std::rc::Rc::new(std::collections::HashMap::new()),
            cross_module_mut_self: std::rc::Rc::new(std::cell::RefCell::new(
                CrossModuleMutSelf::Uncomputed,
            )),
            cross_module_classes: std::rc::Rc::new(std::cell::RefCell::new(
                CrossModuleClasses::Uncomputed,
            )),
        }
    }
}

impl PythonOptions {
    /// Create PythonOptions with tokio runtime (default)
    pub fn with_tokio() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Tokio;
        options
    }

    /// Create PythonOptions with async-std runtime
    pub fn with_async_std() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::AsyncStd;
        options
    }

    /// Create PythonOptions with smol runtime
    pub fn with_smol() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Smol;
        options
    }

    /// Create PythonOptions with a custom async runtime
    pub fn with_custom_runtime(attribute: impl Into<String>, import: impl Into<String>) -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Custom {
            attribute: attribute.into(),
            import: import.into(),
        };
        options
    }

    /// Set the async runtime for these options
    pub fn set_async_runtime(&mut self, runtime: AsyncRuntime) -> &mut Self {
        self.async_runtime = runtime;
        self
    }
}
