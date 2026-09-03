use std::{collections::HashMap, default::Default};

use tracing::info;
use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, prelude::PyAnyMethods};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

use crate::{ASYNC_RUNTIME_FEATURE, CodeGen, CodeGenContext, CrossModuleClasses, CrossModuleMutSelf, ModuleClassInfo, Name, Object, PythonOptions, Statement, StatementType, ExprType, SymbolTableNode, SymbolTableScopes};


#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Type {
    Unimplemented,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Type {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        info!("Type: {:?}", ob);
        Ok(Type::Unimplemented)
    }
}

/// Represents a module as imported from an ast. See the Module struct for the processed module.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawModule {
    pub body: Vec<Statement>,
    pub type_ignores: Vec<Type>,
}

// Extracted manually (not via derive) so a failing statement's precise error
// — which names the construct and its line — propagates instead of being
// replaced by a generic "failed to extract field RawModule.body" message.
impl<'a, 'py> FromPyObject<'a, 'py> for RawModule {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let body_attr = ob
            .getattr("body")
            .map_err(|e| crate::extraction_failure("module body", &ob, e))?;
        let body_list: Vec<pyo3::Bound<PyAny>> = body_attr
            .extract()
            .map_err(|e| crate::extraction_failure("module body", &ob, e))?;

        let mut body = Vec::with_capacity(body_list.len());
        for stmt in &body_list {
            body.push(Statement::extract(stmt.as_borrowed())?);
        }

        let type_ignores = ob
            .getattr("type_ignores")
            .and_then(|t| t.extract())
            .unwrap_or_default();

        Ok(Self { body, type_ignores })
    }
}

/// Represents a module as imported from an ast.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Module {
    pub raw: RawModule,
    pub name: Option<Name>,
    pub doc: Option<String>,
    pub filename: Option<String>,
    pub attributes: HashMap<Name, String>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Module {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // RawModule's extraction already produces precise per-statement
        // errors; don't re-wrap them here.
        let raw_module = ob.extract()?;

        Ok(Self {
            raw: raw_module,
            ..Default::default()
        })
    }
}

impl CodeGen for Module {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        symbols.new_scope();
        // Issue #181: registration happens on the DESUGARED body, so the
        // fused singledispatch generic (and not the `_`-named register
        // definitions it absorbed) is what call sites resolve against. A
        // family this pass refuses is left alone here; `to_rust` raises
        // the conversion error.
        let body = crate::ast::tree::singledispatch::desugar_module(self.raw.body.clone())
            .unwrap_or(self.raw.body);
        for s in body {
            symbols = s.clone().find_symbols(symbols);
        }
        symbols
    }

    fn to_rust(
        mut self,
        ctx: Self::Context,
        mut options: Self::Options,
        mut symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut stream = TokenStream::new();

        // Issue #137: a module-level `try/except ImportError` guard whose
        // try body's imports are ALL statically unresolvable (external to
        // the crate, the runtime, and the vendored deps) FAILS at runtime
        // exactly as rython drops it — the handler branch is the module's
        // real body (`try: import brotli except ImportError: brotli =
        // None` — urllib3's response.py; `except (ImportError,
        // AttributeError): ssl = None; class BaseSSLError(...)` —
        // connection.py). Fold before every body analysis below so store
        // counts, mutable-global detection, and emission all see the
        // branch that actually runs.
        // Issue #181: fuse each `@functools.singledispatch` family into the
        // one `isinstance`-dispatching function that expresses it, before
        // any body analysis below — the shape the monomorphizing
        // specialization pass already lowers (ast::tree::singledispatch).
        self.raw.body = crate::ast::tree::singledispatch::desugar_module(self.raw.body)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        let (folded_body, newly_live) = fold_static_import_trys(&self.raw.body, &options);
        self.raw.body = folded_body;
        // Handler statements the fold made live were invisible to
        // find_symbols (Try::find_symbols skips ImportError-handler
        // bodies, correctly, for the resolvable case): register them now
        // so their own imports resolve (`import warnings` inside socks.py's
        // live fallback).
        for s in &newly_live {
            symbols = s.clone().find_symbols(symbols);
        }

        // Issue #137: module-level VERSION-GATED blocks (`if
        // sys.version_info >= (3, 11):` — certifi's core.py) and
        // static-name gates (`if brotli is not None:` where the module
        // folded the import to `brotli = None`): rython's target version
        // is fixed (3.11.0), so the taken branch is decided at conversion
        // time and its statements are spliced into the module body BEFORE
        // every pass below — a version-gated `def` is a module ITEM, not
        // a nested function inside __module_init__ (which rustc rejects
        // and the module re-exports cannot see).
        self.raw.body = splice_gated_branches(self.raw.body, &options);

        // Capture the module's source filename before fields of `self` are
        // moved, so statement errors can point at the user's Python file.
        let module_filename = self
            .filename
            .clone()
            .or_else(|| self.name.as_ref().map(|n| format!("{}.py", n.id)))
            .unwrap_or_else(|| "<module>".to_string());

        // Add module-level documentation if available and not just an expression
        if let Some(docstring) = self.get_module_docstring() {
            // Only add module docs if there are multiple statements or if this seems to be a real module docstring
            if self.raw.body.len() > 1 || self.looks_like_module_docstring() {
                let doc_lines: Vec<_> = docstring
                    .lines()
                    .map(|line| {
                        if line.trim().is_empty() {
                            quote! { #![doc = ""] }
                        } else {
                            let doc_line = format!("{}", line);
                            quote! { #![doc = #doc_line] }
                        }
                    })
                    .collect();
                stream.extend(quote! { #(#doc_lines)* });
                
                // Add generated by comment only when we have actual module docs
                let generated_comment =
                    format!("Generated from Python file: {}", module_filename);
                stream.extend(quote! { #![doc = #generated_comment] });
            }
        }
        
        if options.with_std_python {
            // For imports, always use "stdpython" since that's the actual crate name
            // The runtime specification is just for dependency management
            stream.extend(quote!(use stdpython::*;));
        }

        // Under no_std the prelude has no String/Vec/format!: bring the
        // alloc surface generated code leans on into scope. Emitted per
        // module — each module file in the generated crate lowers through
        // here — while `extern crate alloc` itself binds locally too, so
        // the imports resolve regardless of what the crate root declares.
        // allow(unused_imports): this is harness plumbing, and the lint
        // posture should keep surfacing only source-Python weaknesses.
        if options.no_std {
            stream.extend(quote! {
                extern crate alloc;
                #[allow(unused_imports)]
                use alloc::{
                    format, vec, borrow::ToOwned, string::String, string::ToString,
                    vec::Vec,
                };
            });
        }
        
        // Add async runtime dependency if async functions are detected.
        // We'll check this early so we can add the import at the top. The
        // import is only emitted for generated BINARY crates
        // (options.async_runtime_dep): library conversions transpile async
        // functions to plain async fns and leave the executor to the
        // consumer. It is feature-gated so `--no-default-features` builds
        // still compile (the entry point then carries the compile_error
        // below).
        let needs_async_runtime = module_contains_async(&self.raw.body);

        if needs_async_runtime && options.async_runtime_dep {
            let runtime_import = format_ident!("{}", options.async_runtime.import());
            stream.extend(quote! {
                #[cfg(feature = #ASYNC_RUNTIME_FEATURE)]
                use #runtime_import;
            });
        }
        
        let mut main_body_stmts = Vec::new();
        let mut has_main_code = false;
        let mut has_async_functions = false;
        let mut module_init_stmts = Vec::new();
        let mut has_module_init_code = false;
        let mut is_simple_main_call_pattern = false;
        // The raw statements behind main_body_stmts/module_init_stmts, kept
        // so assigned names can be hoisted to declarations (assignments
        // lower to plain stores; see collect_assigned_names).
        let mut main_body_raw: Vec<crate::Statement> = Vec::new();
        let mut module_init_raw: Vec<crate::Statement> = Vec::new();

        // Module-level names assigned exactly once from a literal, with no
        // other store ANYWHERE in module scope, are CONSTANTS: they lower
        // to static items so functions in the module can see them (Python
        // module globals are visible everywhere; a store hidden inside
        // __module_init__ is not). The tally recurses through module-level
        // control flow — `DEBUG = False` conditionally overwritten inside
        // an `if` is NOT a constant, and treating it as one would silently
        // freeze the original value. (Function bodies need no scan: without
        // `global` — unsupported — their assignments create locals.)
        let mut module_assign_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        count_module_stores(&self.raw.body, &mut module_assign_counts);

        // Issue #115: module-level names written by functions through
        // `global` lower as MUTABLE statics (`static name: Mutex<T>` /
        // `LazyLock<Mutex<T>>`). Reads render as py_global_read; writes in
        // owning scopes as py_global_write. Names that don't qualify keep
        // the documented write-drop divergence (function_def.rs warns).
        // Computed kinds carry a placeholder boxedness here — refined
        // below, once the module-init type analysis has run.
        let mut global_mutables = module_global_mutable_names(
            &self.raw.body,
            &module_assign_counts,
            &symbols,
            &options,
        );

        // Issue #189: a None-initialized Boxed global whose `global`-writing
        // function stores are all None except exactly one LOCAL class
        // construction (`HISTORY_RECORDER = HistoryRecorder()` — botocore's
        // history.py lazy singleton) becomes a typed class-instance static:
        // `Mutex<Option<Class>>`. The Option is the representation; reads
        // unwrap (loud runtime panic while None, §12.2), `is None` compares
        // read the Option (compare.rs), and stores are None / `Some(instance)`.
        // Any other store shape (a container, a second class, a computed
        // value) keeps the plain Boxed static and its loud conversion error.
        {
            let class_stores = module_global_class_stores(&self.raw.body, &symbols);
            for (name, kind) in global_mutables.iter_mut() {
                if matches!(kind, crate::MutableGlobalKind::Boxed)
                    && let Some(class) = class_stores.get(name)
                {
                    *kind = crate::MutableGlobalKind::Class { class: class.clone() };
                }
            }
        }

        // Statically-decided module names (issue #137): a single-store
        // None or False constant — typically the folded handler of a
        // failed import guard above — makes `if brotli is not None:` /
        // `if HAS_ZSTD:` branches fold at conversion time, exactly the
        // branches CPython would never enter. Names written through
        // `global` are excluded (they are mutable statics, not
        // constants).
        {
            let (none_names, false_names, module_names) =
                static_gate_names(&self.raw.body, &module_assign_counts, &global_mutables, &options);
            options.statically_none_names = std::rc::Rc::new(none_names);
            options.statically_false_names = std::rc::Rc::new(false_names);
            options.statically_module_names = std::rc::Rc::new(module_names);
        }

        // Issue #118: MODULE-LEVEL argparse (certifi's __main__.py builds
        // its parser at top level). The same conversion-time rewrite the
        // function path runs: parser-building statements vanish, and the
        // parse_args assignment becomes the typed-namespace destructure
        // inside __module_init__ (later module-level statements read the
        // namespace there; functions cannot — a module-init local, loud
        // in rustc).
        let module_argparse = crate::ast::tree::function_def::scan_argparse(&self.raw.body)
            .map_err(|e| wrap_module_error(&module_filename, e))?;

        // Classes that participate in an inheritance hierarchy (have a real
        // base, or are used as a base) lower with the trait machinery; every
        // other class stays a plain struct. Computed once so both sides of a
        // hierarchy agree on the scheme. Classes nested under containers
        // (`if __name__ == "__main__":` and friends) count too — their
        // class statements lower the same way.
        {
            // The module's EMITTED classes (the static-name gates fold at
            // emission, after the splice above ran without their names):
            // a class the emission drops is neither a hierarchy member
            // nor a sum-type variant.
            let mut classes = Vec::new();
            collect_class_defs(
                &splice_gated_branches(self.raw.body.clone(), &options),
                &mut classes,
            );
            let mut hierarchy = std::collections::HashSet::new();
            for c in &classes {
                let has_real_base = c
                    .bases
                    .iter()
                    .any(|b| matches!(b, crate::ExprType::Name(n) if n.id != "object"));
                if has_real_base {
                    hierarchy.insert(c.name.clone());
                }
                for b in &c.bases {
                    if let crate::ExprType::Name(n) = b
                        && n.id != "object"
                    {
                        hierarchy.insert(n.id.clone());
                    }
                }
            }
            options.hierarchy_classes = std::rc::Rc::new(hierarchy);
            // The closed-world hierarchy (hierarchy.rs): the polymorphic
            // roots of the WHOLE crate and their subtrees, so a root's
            // slot type renders as its sum type in every module.
            // The sum type's variants are module ITEMS (a class under a
            // gate the emission cannot fold is not one).
            let mut items = Vec::new();
            top_level_class_defs(
                &splice_gated_branches(self.raw.body.clone(), &options),
                &mut items,
            );
            let roots = crate::ast::tree::hierarchy::compute_roots(&items, &options);
            crate::ast::tree::hierarchy::install_roots(&roots);
            options.hierarchy_roots = std::rc::Rc::new(roots);
        }

        // Functions whose unannotated parameter is isinstance-dispatched
        // monomorphize at conversion time (specialize.rs): one variant per
        // tested type plus a residual, with call sites dispatched by the
        // argument's static type. The registry is computed once so the
        // renderer and every call site agree.
        {
            let mut classes = Vec::new();
            collect_class_defs(&self.raw.body, &mut classes);
            let class_names: Vec<String> = classes.iter().map(|c| c.name.clone()).collect();
            let mut registry = crate::ast::tree::specialize::SpecRegistry::new();
            for stmt in &self.raw.body {
                if let crate::ast::tree::StatementType::FunctionDef(f) = &stmt.statement {
                    if let Some(spec) = crate::ast::tree::specialize::detect_specializable(
                        f,
                        &symbols,
                        &class_names,
                        &options,
                    ) {
                        registry.insert(f.name.clone(), spec);
                    }
                }
            }
            options.specialized_fns = std::rc::Rc::new(registry);
        }

        // Trait method signatures widen to `&mut self` when ANY definition
        // in the hierarchy mutates self: overrides re-emit into the ROOT
        // class's trait (the first class in the chain that defines the
        // method), whose signature must fit every impl, and call sites must
        // borrow the receiver mutably to match. Keyed by the root class.
        {
            let mut classes = Vec::new();
            collect_class_defs(&self.raw.body, &mut classes);
            let mut trait_mut = std::collections::HashMap::<
                String,
                std::collections::HashSet<String>,
            >::new();
            for c in &classes {
                let chain = c.base_chain(&symbols);
                for m in c.methods() {
                    if m.name == "__init__" {
                        continue;
                    }
                    if c.own_method_mutates(&m.name, &symbols, &options) {
                        // The root = the TOPMOST class in the chain that
                        // defines the method (the trait owner).
                        if let Some(root) = chain
                            .iter()
                            .rev()
                            .find(|cc| cc.methods().any(|mm| mm.name == m.name))
                        {
                            trait_mut
                                .entry(root.name.clone())
                                .or_default()
                                .insert(m.name.clone());
                        }
                    }
                }
            }
            options.trait_mut_self = std::rc::Rc::new(trait_mut);
        }

        // A user `main` that returns a value cannot serve as the Rust entry
        // point directly (Result<i64, _> does not implement Termination);
        // route it through the renaming wrapper, which discards the value —
        // exactly what Python's `if __name__: main()` does.
        let user_main_returns_value = self.raw.body.iter().any(|s| {
            matches!(
                &s.statement,
                crate::StatementType::FunctionDef(f)
                    if f.name == "main" && f.resolved_return_type(&symbols, &options).is_some()
            )
        });

        // Pass 1: classify statements so the hoisted-name sets are known
        // before any statement renders — a `for` target on a name that
        // leaks out of the loop lowers to a store into the prologue
        // binding, never a shadowing fresh binding (issue #80). The render
        // pass below repeats this classification (cheap); the raw lists
        // drive both the hoisted sets and hoisted_declarations.
        for (stmt_index, s) in self.raw.body.iter().enumerate() {
            // Issue #118: module-level argparse statements are consumed by
            // the conversion-time rewrite — the parser statements vanish
            // and the parse_args assignment is replaced in the emit loop.
            if let Some(rw) = &module_argparse
                && (rw.skip.contains(&stmt_index) || stmt_index == rw.parse_index)
            {
                continue;
            }
            if let crate::StatementType::If(if_stmt) = &s.statement {
                // `if TYPE_CHECKING:` never runs at runtime — skip the
                // whole block (imports, type-only classes).
                if Self::is_type_checking_test(&if_stmt.test) {
                    continue;
                }
                let test_str = format!("{:?}", if_stmt.test);
                if test_str.contains("__name__") && test_str.contains("__main__") {
                    let is_simple_main_call = Self::is_simple_main_call_block(&if_stmt.body)
                        && !user_main_returns_value
                        && options.numpy_backend.is_none();
                    if !is_simple_main_call {
                        main_body_raw.extend(if_stmt.body.iter().cloned());
                    }
                    continue;
                }
            }
            // Module-level constants are static items, not runtime stores.
            if let crate::StatementType::Assign(a) = &s.statement {
                // Issue #127: a decorator-factory assignment emits the
                // synthesized cached function at module level (handled in
                // the emit loop) — not a runtime store.
                if crate::try_lru_cache_factory(a, Some(&options), &symbols).is_some() {
                    continue;
                }
                // A type alias (`builtin_str = str`) is a declaration, not
                // a runtime store.
                if let [crate::ExprType::Name(_)] = a.targets.as_slice()
                    && crate::ast::tree::assign::builtin_scalar_alias_type(&a.value).is_some()
                {
                    continue;
                }
                if let Some(names) = assign_name_targets(a) {
                    // A type alias (`builtin_str = str`) is a declaration, not
                    // a runtime store — skip it from the init body too.
                    if names
                        .iter()
                        .all(|n| module_assign_counts.get(n) == Some(&1))
                        && (const_static_type(&a.value).is_some()
                            || is_type_alias_value(&a.value))
                    {
                        continue;
                    }
                    // A mutable static's single module store IS its
                    // initializer (issue #115) — not a runtime store.
                    // COMPUTED initializers stay in the analysis body so
                    // the type inference sees them (their default lowering
                    // is preempted in the emit loop, and the hoist skips
                    // them); const-shaped kinds need no analysis.
                    if names.iter().all(|n| {
                        matches!(
                            global_mutables.get(n),
                            Some(crate::MutableGlobalKind::Scalar)
                                | Some(crate::MutableGlobalKind::Boxed)
                                | Some(crate::MutableGlobalKind::Str)
                        )
                    }) {
                        continue;
                    }
                }
            }
            if !Self::is_declaration_statement(&s.statement) {
                module_init_raw.push(s.clone());
            }
        }

        // Module-level values assigned from NON-constant expressions that
        // functions read are promoted to LazyLock statics (a `let` inside
        // __module_init__ is invisible to functions — E0425). Constrained to
        // a single top-level store (reassignment/conditional stores keep the
        // __module_init__ lowering: a static would freeze the first value),
        // a non-const value, and NOT a typing alias or the builtin-alias
        // declaration shape. A value that a SIBLING module imports
        // (`from .constant import _THAI` — charset_normalizer's utils) is
        // promoted the same way: without a `pub static`, the importing
        // module's `use crate::charset_normalizer::constant::_THAI;` fails
        // with E0432 (a module-init local is invisible to other modules).
        let function_free_reads = module_function_free_reads(&self.raw.body);
        let mut promoted_statics: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        // In a multi-module conversion, the promotion decision comes from the
        // SHARED computation (`module_promoted_static_names`) so the DEFINING
        // module and every IMPORTING module agree on which names are statics
        // (an importing module renders reads of them as `(*name).clone()`).
        // The inline loop below is the fallback for library/single-module
        // use, where `this_module_path` is empty and there are no siblings.
        if !options.this_module_path.is_empty()
            && options.module_defs.contains_key(&options.this_module_path)
        {
            promoted_statics = module_promoted_static_names(&options, &options.this_module_path)
                .as_ref()
                .clone();
        } else {
        for s in &self.raw.body {
            if let crate::StatementType::If(if_stmt) = &s.statement {
                if Self::is_type_checking_test(&if_stmt.test) {
                    continue;
                }
                let test_str = format!("{:?}", if_stmt.test);
                if test_str.contains("__name__") && test_str.contains("__main__") {
                    continue;
                }
            }
            if let crate::StatementType::Assign(a) = &s.statement {
                if crate::try_lru_cache_factory(a, Some(&options), &symbols).is_some() {
                    continue;
                }
                if let [crate::ExprType::Name(_)] = a.targets.as_slice()
                    && crate::ast::tree::assign::builtin_scalar_alias_type(&a.value).is_some()
                {
                    continue;
                }
                if let Some(targets) = assign_name_targets(a) {
                    for n in &targets {
                        if module_assign_counts.get(n) == Some(&1)
                            && const_static_type(&a.value).is_none()
                            && !is_type_alias_value(&a.value)
                            // rust.bind / rust.c_bind declarations are
                            // compile-time bindings, not runtime stores: the
                            // Assign codegen handles them (issue #79 family).
                            && !crate::is_rust_bind_call(&a.value)
                            // The argparse namespace is rewritten into
                            // __module_init__ (issue #118), never a static.
                            && !module_argparse
                                .as_ref()
                                .is_some_and(|rw| rw.args_var == *n)
                            && function_free_reads.contains(n)
                            // A name ALSO bound by an import (see
                            // module_promoted_static_names): the import owns
                            // the name — a promoted static would collide.
                            && !matches!(
                                symbols.get(n),
                                Some(crate::SymbolTableNode::ImportFrom(_))
                                    | Some(crate::SymbolTableNode::Import(_))
                            )
                        {
                            promoted_statics.insert(n.clone());
                        }
                    }
                }
                // A TUPLE-UNPACK target (`a, b = value` — idna's
                // `_STATUS_VALID, ... = b"VMDI"`): each element name
                // binds the value at its position; promote the ones
                // functions read (a module-init local is invisible to
                // function bodies — E0425), mirroring the plain-name arm.
                if let Some(pairs) = assign_unpack_indices(a) {
                    // ALL elements when ANY qualifies (the init statement
                    // unpacks the whole value; a partial promotion would
                    // re-assign the static positions — E0070).
                    let any = pairs.iter().any(|(n, _i)| {
                        module_assign_counts.get(n) == Some(&1)
                            && function_free_reads.contains(n)
                            && !matches!(
                                symbols.get(n),
                                Some(crate::SymbolTableNode::ImportFrom(_))
                                    | Some(crate::SymbolTableNode::Import(_))
                            )
                    });
                    if any {
                        for (n, _i) in pairs {
                            promoted_statics.insert(n.clone());
                        }
                    }
                }
            }
        }
        }
        // Transitive promotion to a fixpoint (mirrors the shared
        // module_promoted_static_names): every name a PROMOTED name's
        // initializer reads must also be promoted.
        let mut init_reads: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        for s2 in &self.raw.body {
            if let crate::StatementType::Assign(a) = &s2.statement {
                if const_static_type(&a.value).is_some() {
                    continue;
                }
                if let Some(targets) = assign_name_targets(a) {
                    for n in &targets {
                        if module_assign_counts.get(n) == Some(&1) {
                            let reads = module_expr_reads(&a.value);
                            init_reads.insert(n.clone(), reads);
                        }
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            let snapshot: std::collections::HashSet<String> = promoted_statics.clone();
            for (n, reads) in &init_reads {
                if !snapshot.contains(n) {
                    continue;
                }
                for r in reads {
                    if !promoted_statics.contains(r) && init_reads.contains_key(r) {
                        promoted_statics.insert(r.clone());
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // A DEFINITE if/else module value (`if sys.platform == "win32":
        // preferred_clock = time.perf_counter else: preferred_clock =
        // time.time` — requests' sessions): both branches assign the SAME
        // name exactly once, so the value is definitely set — promote to a
        // LazyLock static whose closure is the conditional expression (the
        // multi-store exclusion is only about UNDEFINITE stores; a static
        // here cannot freeze a stale value).
        let mut promoted_conditional: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (i, s) in self.raw.body.iter().enumerate() {
            if let crate::StatementType::If(if_stmt) = &s.statement {
                if if_stmt.orelse.is_empty() {
                    continue;
                }
                let branch_name = |stmts: &[crate::Statement]| -> Option<String> {
                    if stmts.len() != 1 {
                        return None;
                    }
                    match &stmts[0].statement {
                        crate::StatementType::Assign(a) if a.targets.len() == 1 => {
                            match &a.targets[0] {
                                crate::ExprType::Name(n) => Some(n.id.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                };
                if let (Some(b1), Some(b2)) =
                    (branch_name(&if_stmt.body), branch_name(&if_stmt.orelse))
                    && b1 == b2
                    // The two branches alone give a count of 4 (each
                    // nested store counts ×2); anything else (a top-level
                    // store, another conditional) makes it 5+.
                    && module_assign_counts.get(&b1) == Some(&4)
                    && function_free_reads.contains(&b1)
                {
                    promoted_conditional.insert(b1.clone(), i);
                    promoted_statics.insert(b1.clone());
                }
            }
            // A DEFINITE try/except module value (`try: is_urllib3_1 =
            // int(...) == 1 except (TypeError, AttributeError):
            // is_urllib3_1 = True` — requests' compat.py): the SAME name
            // is stored once in the try body and once in a handler — a
            // definite value (the try either completes or raises into the
            // handler), so it can be promoted to a LazyLock static without
            // freezing a stale value. Same count shape as the if/else case
            // (each nested store ×2 → 4).
            if let crate::StatementType::Try(t) = &s.statement {
                let body_name = single_assign_name(&t.body);
                if let Some(name) = body_name
                    && t.handlers.iter().all(|h| single_assign_name(&h.body) == Some(name.clone()))
                    && module_assign_counts.get(&name) == Some(&4)
                    && function_free_reads.contains(&name)
                {
                    promoted_conditional.insert(name.clone(), i);
                    promoted_statics.insert(name.clone());
                }
            }
        }
        let (init_hoisted, init_leaked) = hoisted_name_set(&module_init_raw, &ctx, &symbols, &options);
        let (main_hoisted, main_leaked) = hoisted_name_set(&main_body_raw, &ctx, &symbols, &options);
        // Name reads of promoted statics render as `(*name).clone()` (name.rs);
        // every scope rendered below (module init, __main__, functions) sees
        // the set through the cloned options.
        options.promoted_statics = std::rc::Rc::new(promoted_statics.clone());

        // Module-level code gets no per-function analysis pass, so run the
        // same type inference / empty-container pinning here: without it,
        // `xs = []` followed by `xs.append(1)` at module level fails with
        // "empty container literal has no inferable element type" even
        // though the pinning use is right there (issue #81-family, Devin
        // review on #103). The __main__ block gets its own pass.
        {
            let info = crate::analyze_function_types(&module_init_raw, Some(&options), Some(&symbols));
            options.use_counts = std::rc::Rc::new(info.use_counts);
            options.name_types = std::rc::Rc::new(info.name_types);
            options.empty_pinned = std::rc::Rc::new(info.empty_pinned);
        }
        // Issue #115: with the init analysis done, a COMPUTED mutable
        // global's boxedness is decidable — the static holds the inferred
        // type when one exists (module_init_static_ty), else the boxed
        // PyValue with stores wrapped in PyValue::from. Then every scope
        // rendered below sees the mutable statics (reads render
        // py_global_read); MODULE scope owns all of them for writes
        // (module init and the __main__ body are module scope — no
        // `global` needed there). Function scopes re-narrow
        // scope_global_writables to their own `global` declarations
        // (function_def.rs).
        for s in &self.raw.body {
            if let crate::StatementType::Assign(a) = &s.statement
                && let [crate::ExprType::Name(n)] = a.targets.as_slice()
                && let Some(kind) = global_mutables.get_mut(&n.id)
                && matches!(kind, crate::MutableGlobalKind::Computed { .. })
            {
                *kind = crate::MutableGlobalKind::Computed {
                    boxed: module_init_static_ty(&n.id, &a.value, &options).is_none(),
                };
            }
        }
        options.mutable_statics = std::rc::Rc::new(global_mutables.clone());
        options.scope_global_writables =
            std::rc::Rc::new(global_mutables.keys().cloned().collect());
        // Issue #189: a class-instance global reads as the INSTANCE (the
        // Option is the static's representation, unwrapped at the read), so
        // the name's type everywhere is the class — return inference, method
        // receivers, and call-site checks all see `TypeInfo::Class`.
        {
            let mut nt = (*options.name_types).clone();
            for (name, kind) in global_mutables.iter() {
                if let crate::MutableGlobalKind::Class { class } = kind {
                    nt.insert(name.clone(), crate::TypeInfo::Class(class.clone()));
                }
            }
            options.name_types = std::rc::Rc::new(nt);
        }
        // Module-level aliasing (`b = a` on a container, later mutated) is
        // the same divergence the function-level guard rejects (issue #79).
        crate::check_aliasing(
            &module_init_raw,
            &symbols,
            &options.name_types,
            &options.use_counts,
        )?;
        // Issue #112: `del name` at module level lowers to a no-op; a use
        // after the del is a loud error (the module body is one scope).
        crate::check_deleted_names(&module_init_raw)?;
        // Issue #109, M5: module-level calls (including the __main__ block)
        // are checked against callee inferred bounds at conversion time.
        crate::check_call_sites(
            &module_init_raw,
            &symbols,
            &options.name_types,
            &options,
        )?;
        let main_info = crate::analyze_function_types(&main_body_raw, Some(&options), Some(&symbols));
        crate::check_aliasing(
            &main_body_raw,
            &symbols,
            &main_info.name_types,
            &main_info.use_counts,
        )?;
        crate::check_call_sites(
            &main_body_raw,
            &symbols,
            &main_info.name_types,
            &options,
        )?;

        // The module docstring is emitted as #![doc] attributes above; its
        // statement (a leading string Expr) must be SKIPPED here or it
        // leaks into the generated module as a bare string literal (the
        // requests build failure in the library sweep). Skip exactly when
        // the docstring was emitted above: only then was the leading string
        // consumed (a lone short string with no doc markers stays a bare
        // expression statement, matching the pre-sweep behavior).
        let mut pending_docstring = self.get_module_docstring().is_some()
            && (self.raw.body.len() > 1 || self.looks_like_module_docstring());
        let mut seen_non_doc_statement = false;
        // Names the shared RHS statics of promoted tuple-unpacks
        // (`__rython_unpack_0`, ...) get; one per unpack assignment, in
        // source order.
        let mut unpack_counter = 0usize;
        for (stmt_index, s) in self.raw.body.into_iter().enumerate() {
            // Issue #118: module-level argparse. Parser-building statements
            // vanish; the parse_args assignment becomes the typed-namespace
            // destructure inside __module_init__, at its original position.
            if let Some(rw) = &module_argparse {
                if rw.skip.contains(&stmt_index) {
                    continue;
                }
                if stmt_index == rw.parse_index {
                    let args_ident = crate::safe_ident(&rw.args_var);
                    let tokens = crate::ast::tree::function_def::lower_parse_args(
                        rw, &ctx, &options, &symbols,
                    )
                    .map_err(|e| wrap_module_error(&module_filename, e))?;
                    module_init_stmts.push(quote! {
                        let #args_ident;
                        #tokens
                    });
                    has_module_init_code = true;
                    continue;
                }
            }
            if pending_docstring
                && matches!(
                    &s.statement,
                    crate::StatementType::Expr(e)
                        if matches!(
                            &e.value,
                            crate::ExprType::Constant(c)
                                if matches!(&c.0, Some(litrs::Literal::String(_)))
                        )
                )
            {
                pending_docstring = false;
                continue;
            }
            // Check if this statement is an async function
            if let crate::StatementType::AsyncFunctionDef(_) = &s.statement {
                has_async_functions = true;
            }

            // Check for if __name__ == "__main__" blocks at the AST level before generating code
            if let crate::StatementType::If(if_stmt) = &s.statement {
                let test_str = format!("{:?}", if_stmt.test);
                if test_str.contains("__name__") && test_str.contains("__main__") {
                    // Check if this is a simple main() call pattern
                    // (disabled when --numpy-backend forces startup code: the
                    // wrapper main below must run __module_init__ first).
                    let is_simple_main_call = Self::is_simple_main_call_block(&if_stmt.body)
                        && !user_main_returns_value
                        && options.numpy_backend.is_none();
                    
                    if is_simple_main_call {
                        // For simple main() calls, we'll use the user's main function directly
                        // Set a flag to indicate we should not rename the main function
                        has_main_code = true;
                        is_simple_main_call_pattern = true;
                        // Don't collect the main body statements - we'll use user's main directly
                    } else {
                        // This is a complex __name__ == "__main__" block - collect its body for main function
                        let main_options = {
                            let mut o = options.clone();
                            o.hoisted_names = std::rc::Rc::new(main_hoisted.clone());
                            o.leaked_loop_targets = std::rc::Rc::new(main_leaked.clone());
                            o.use_counts = std::rc::Rc::new(main_info.use_counts.clone());
                            o.name_types = std::rc::Rc::new(main_info.name_types.clone());
                            o.empty_pinned = std::rc::Rc::new(main_info.empty_pinned.clone());
                            o
                        };
                        for body_stmt in &if_stmt.body {
                            let stmt_token = body_stmt
                                .clone()
                                .to_rust(ctx.clone(), main_options.clone(), symbols.clone())
                                .map_err(|e| wrap_module_error(&module_filename, e))?;
                            if !stmt_token.to_string().trim().is_empty() {
                                main_body_stmts.push(stmt_token);
                                has_main_code = true;
                            }
                        }
                    }
                    // Skip generating this if statement - we've processed its contents
                    continue;
                }
                // `if TYPE_CHECKING:` never runs at runtime — the whole
                // block is skipped — EXCEPT its imports of REAL
                // sibling-module names, which still emit `use` statements:
                // annotations reference those names and find_symbols already
                // registered the imports (requests/adapters.py imports
                // PreparedRequest under TYPE_CHECKING and uses it in
                // signatures; urllib3/util/connection.py imports
                // BaseHTTPConnection the same way). typing /
                // typing_extensions and external-module imports lower to
                // nothing — their annotations resolve to the boxed PyValue.
                if Self::is_type_checking_test(&if_stmt.test) {
                    for body_stmt in &if_stmt.body {
                        // Filtered PER NAME, not all-or-nothing: `from
                        // .ssl_ import _TYPE_PEER_CERT_RET,
                        // _TYPE_PEER_CERT_RET_DICT` (urllib3's
                        // ssltransport) pairs a generated type alias with
                        // a TYPE_CHECKING-only TypedDict stub — the alias
                        // still needs its `use` (annotations reference
                        // it) while the stub must not emit one (E0432).
                        let renderable = match &body_stmt.statement {
                            crate::StatementType::ImportFrom(i) => {
                                let root = i.module.split('.').next().unwrap_or("");
                                if matches!(
                                    crate::AnnotationModule::from_name(root),
                                    Some(
                                        crate::AnnotationModule::Typing
                                            | crate::AnnotationModule::TypingExtensions
                                    )
                                ) {
                                    None
                                } else if crate::ast::tree::import::is_stdpython_module(root) {
                                    // A stdpython-module import whose ITEM
                                    // may not exist in the runtime module
                                    // (`from io import BufferedWriter` —
                                    // requests' utils.py, only used as an
                                    // annotation; stdpython::io has
                                    // StringIO but no BufferedWriter): emit
                                    // the `use` only for names with a
                                    // known runtime counterpart, else the
                                    // generated build fails E0432.
                                    let names: Vec<_> = i
                                        .names
                                        .iter()
                                        .filter(|a| {
                                            crate::ast::tree::import::stdpython_module_item(
                                                root, &a.name,
                                            )
                                        })
                                        .cloned()
                                        .collect();
                                    (!names.is_empty()).then(|| {
                                        let mut filtered = i.clone();
                                        filtered.names = names;
                                        filtered
                                    })
                                } else {
                                    // Only sibling-module imports whose
                                    // names are actually GENERATED (not
                                    // TYPE_CHECKING stubs) emit `use`s.
                                    let path = i.resolved_module_path(&options);
                                    let names: Vec<_> = i
                                        .names
                                        .iter()
                                        .filter(|a| {
                                            crate::ast::tree::module::module_def_has_runtime_item(
                                                &options, &path, &a.name,
                                            )
                                        })
                                        .cloned()
                                        .collect();
                                    (!names.is_empty()).then(|| {
                                        let mut filtered = i.clone();
                                        filtered.names = names;
                                        filtered
                                    })
                                }
                            }
                            _ => None,
                        };
                        if let Some(filtered) = renderable {
                            let stmt = crate::Statement {
                                statement: crate::StatementType::ImportFrom(filtered),
                                ..body_stmt.clone()
                            };
                            let stmt_token = stmt
                                .to_rust(ctx.clone(), options.clone(), symbols.clone())
                                .map_err(|e| wrap_module_error(&module_filename, e))?;
                            if !stmt_token.to_string().trim().is_empty() {
                                stream.extend(stmt_token);
                            }
                        }
                    }
                    continue;
                }
            }
            // A NON-LEADING module-level bare STRING expression
            // (`"IDNA Mapping Table from UTS46."` — idna's uts46data, a
            // docstring mid-module) is an annotation with no runtime
            // effect: a bare string literal in item position is not legal
            // Rust. Drop it. A module whose FIRST statement is a bare
            // string (a lone-string test module) keeps it — the
            // pre-sweep behavior the tests rely on.
            if seen_non_doc_statement
                && let crate::StatementType::Expr(e) = &s.statement
                && matches!(
                    &e.value,
                    crate::ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::String(_)))
                )
            {
                continue;
            }
            seen_non_doc_statement = true;
            
            // Module-level constants become static items visible to every
            // function in the module.
            if let crate::StatementType::Assign(a) = &s.statement {
                // A module-level assign to a name the module ALSO imports
                // (`SSLTransport = None` then `from .ssltransport import
                // SSLTransport` — urllib3's ssl_.py): Python's LAST binding
                // wins, so the import overrides the assign.
                //
                // A SIBLING import emits a real `use` binding the name — the
                // store is dead; emitting it would collide (E0252) or render
                // the imported class unusable (E0433). Drop it.
                //
                // An EXTERNAL import emits nothing (external-module
                // divergence), so the name must still resolve for siblings
                // that read it (`ssl_::HAS_NEVER_CHECK_COMMON_NAME` from
                // connection.rs): emit a boxed-None static — the external
                // value is unmodeled.
                if let Some(names) = assign_name_targets(a) {
                    let mut sibling_owned = false;
                    let mut external_owned = false;
                    for n in &names {
                        match symbols.get(n) {
                            Some(crate::SymbolTableNode::ImportFrom(_))
                            | Some(crate::SymbolTableNode::Import(_)) => {
                                if crate::ast::tree::import::resolves_to_external_import(
                                    n, &options, &symbols,
                                ) {
                                    external_owned = true;
                                } else {
                                    sibling_owned = true;
                                }
                            }
                            _ => {}
                        }
                    }
                    if sibling_owned {
                        options.definition_warnings.borrow_mut().push(format!(
                            "module-level assignment to `{}` is dropped: the name is also \
                             imported from a sibling module, and Python's later import \
                             binding wins",
                            names.join(", ")
                        ));
                        continue;
                    }
                    if external_owned {
                        options.definition_warnings.borrow_mut().push(format!(
                            "module-level assignment to `{}` is dropped: the name is also \
                             imported from a module external to the generated crate; the \
                             name lowers to the boxed None (external-module divergence)",
                            names.join(", ")
                        ));
                        for n in &names {
                            let ident = crate::safe_ident(n);
                            stream.extend(quote! {
                                pub static #ident: std::sync::LazyLock<stdpython::PyValue> =
                                    std::sync::LazyLock::new(|| stdpython::PyValue::None_);
                            });
                        }
                        continue;
                    }
                }
                // A type alias (`builtin_str = str`): emit the pub type at
                // module level so re-exports resolve; Assign::to_rust also
                // knows this shape, but the module path must place it as a
                // declaration, not an init-time store.
                if let [crate::ExprType::Name(target)] = a.targets.as_slice()
                    && let Some(ty) =
                        crate::ast::tree::assign::builtin_scalar_alias_type(&a.value)
                {
                    let ident = crate::safe_ident(&target.id);
                    stream.extend(quote! {
                        #[allow(dead_code)]
                        pub type #ident = #ty;
                    });
                    continue;
                }
                // Issue #127: `name = lru_cache(maxsize=N)(fn)` — the
                // decorator factory applied as an expression. Emit the
                // synthesized cached function as a module-level item (the
                // same @lru_cache machinery a decorated definition gets),
                // not an init-time store. find_symbols registered the name
                // as the function so call sites resolve.
                if let Some(synth) =
                    crate::try_lru_cache_factory(a, Some(&options), &symbols)
                {
                    stream.extend(
                        synth
                            .clone()
                            .to_rust(ctx.clone(), options.clone(), symbols.clone())
                            .map_err(|e| wrap_module_error(&module_filename, e))?,
                    );
                    continue;
                }
                // Issue #115: a `global`-written module value becomes a
                // MUTABLE static with the single module store as its
                // initializer. Const-expressible initializers (scalar
                // literals, None) use a plain `static name: Mutex<T>`;
                // string literals and computed expressions wrap in a
                // LazyLock (their construction is not const), with a TOUCH
                // in __module_init__ for a computed initializer so its
                // side effects still run at import time, in order. Reads
                // everywhere render as py_global_read; writes in owning
                // scopes as py_global_write.
                if let [crate::ExprType::Name(target)] = a.targets.as_slice()
                    && let Some(kind) = global_mutables.get(&target.id)
                {
                    use crate::MutableGlobalKind as Kind;
                    let ident = crate::safe_ident(&target.id);
                    match kind {
                        Kind::Boxed => {
                            stream.extend(quote! {
                                pub static #ident: std::sync::Mutex<stdpython::PyValue> =
                                    std::sync::Mutex::new(stdpython::PyValue::None_);
                            });
                        }
                        Kind::Class { class } => {
                            // Issue #189: the class-instance global — the
                            // module store is the None state; the singleton
                            // construction lives in the `global`-writing
                            // function. Const None init; the Option is the
                            // representation, unwrapped at value reads
                            // (name.rs) and matched by `is None` (compare.rs).
                            let cls = crate::safe_ident(class);
                            stream.extend(quote! {
                                pub static #ident: std::sync::Mutex<Option<#cls>> =
                                    std::sync::Mutex::new(None);
                            });
                        }
                        Kind::Scalar => {
                            let ty = const_static_type(&a.value)
                                .expect("mutable static initializer must be a const scalar");
                            let value = a.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            stream.extend(quote! {
                                pub static #ident: std::sync::Mutex<#ty> =
                                    std::sync::Mutex::new(#value);
                            });
                        }
                        Kind::Str => {
                            let value = a.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            stream.extend(quote! {
                                pub static #ident:
                                    std::sync::LazyLock<std::sync::Mutex<String>> =
                                    std::sync::LazyLock::new(|| {
                                        std::sync::Mutex::new((#value).to_string())
                                    });
                            });
                        }
                        Kind::Computed { boxed } => {
                            // Mirrors the promoted-static machinery: a
                            // fallible initializer (rendered with a
                            // trailing `?`) unwraps inside the closure,
                            // panicking on failure — the import-time raise
                            // becomes an abort (the §12.2 divergence).
                            let rhs = a.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            let stripped =
                                crate::ast::tree::call::strip_trailing_question(&rhs);
                            let is_fallible = stripped.to_string() != rhs.to_string();
                            let value_tokens = if is_fallible {
                                quote!(match #stripped {
                                    Ok(__rython_v) => __rython_v,
                                    Err(__rython_e) => panic!(
                                        "module-level `{}` initialization failed: {}",
                                        stringify!(#ident),
                                        __rython_e
                                    ),
                                })
                            } else {
                                stripped
                            };
                            let (ty, wrapped) = if *boxed {
                                (
                                    quote!(stdpython::PyValue),
                                    quote!(stdpython::PyValue::from(#value_tokens)),
                                )
                            } else {
                                let ty = module_init_static_ty(&target.id, &a.value, &options)
                                    .expect("Computed{boxed:false} implies an inferred type");
                                (ty, value_tokens)
                            };
                            stream.extend(quote! {
                                pub static #ident:
                                    std::sync::LazyLock<std::sync::Mutex<#ty>> =
                                    std::sync::LazyLock::new(|| {
                                        std::sync::Mutex::new(#wrapped)
                                    });
                            });
                            module_init_stmts.push(quote!(let _ = &*#ident;));
                            has_module_init_code = true;
                        }
                    }
                    continue;
                }
                if let Some(names) = assign_name_targets(a) {
                    // Every target must be a single-store name for the
                    // static promotion (a chained `__version__ = version =
                    // '2.7.0'` promotes BOTH names to `pub static`).
                    if names
                        .iter()
                        .all(|n| module_assign_counts.get(n) == Some(&1))
                    {
                        // A single-store binding of a STDPYTHON-module
                        // constant (`VERIFY_X509_PARTIAL_CHAIN =
                        // getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", ...)`
                        // after the getattr fold — urllib3's ssl_.py):
                        // a `pub use ... as name` aliases the runtime item
                        // so functions and sibling importers see it,
                        // without needing to know its type.
                        if names.len() == 1
                            && let Some((module, item)) = stdlib_const_attr(&a.value)
                        {
                            let runtime = crate::safe_ident(&options.stdpython);
                            let module_ident = crate::safe_ident(&module);
                            let item_ident = crate::safe_ident(&item);
                            let ident = crate::safe_ident(&names[0]);
                            stream.extend(if names[0] == item {
                                quote!(pub use #runtime::#module_ident::#item_ident;)
                            } else {
                                quote!(pub use #runtime::#module_ident::#item_ident as #ident;)
                            });
                            continue;
                        }
                        if let Some(ty) = const_static_type(&a.value) {
                            let value = a.value.clone().to_rust(
                                ctx.clone(),
                                options.clone(),
                                symbols.clone(),
                            )?;
                            for n in &names {
                                let ident = crate::safe_ident(n);
                                stream.extend(quote!(pub static #ident: #ty = #value;));
                            }
                            continue;
                        }
                    }
                    // A TYPE ALIAS whose value is a typing annotation
                    // (`_TYPE_REDUCE_RESULT = tuple[typing.Callable[...,
                    // object], ...]`, `_TYPE_BODY = typing.Union[...]` —
                    // urllib3): the name is consumed by annotation
                    // resolution (resolve_alias_typeinfo), never a runtime
                    // value. Emit a `pub type` alias when the annotation
                    // resolves, else nothing — the old behavior emitted a
                    // nonsense `py_index` store.
                    if names.len() == 1
                        && is_type_alias_value(&a.value)
                    {
                        if let Some(ty) =
                            crate::resolve_alias_typeinfo(&a.value, &symbols, &options)
                                .map(|t| t.to_rust_type())
                        {
                            let ident = crate::safe_ident(&names[0]);
                            stream.extend(quote! {
                                #[allow(dead_code)]
                                pub type #ident = #ty;
                            });
                        }
                        continue;
                    }
                }
            }

            // A module-level value that functions read or siblings import
            // (promoted_statics): emit a LazyLock static whose closure holds
            // the initializer, and a TOUCH (`let _ = &*name;`) in
            // __module_init__ at the store's position, so initialization
            // still happens at import time, in order. Function reads deref
            // the static automatically (LazyLock: Deref). A fallible
            // initializer (rendered with a trailing `?`) unwraps inside the
            // closure, panicking on failure — the import-time raise becomes
            // an abort (divergence). A CHAINED assignment (`__version__ =
            // version = '2.7.0'`) where every target is promoted emits one
            // static per name, all with the same initializer (the assign.rs
            // chain lowering would otherwise hide the values in
            // __module_init__).
            if let crate::StatementType::Assign(a) = &s.statement {
                let mut promoted: Vec<String> = match assign_name_targets(a) {
                    Some(names) => names
                        .into_iter()
                        .filter(|n| promoted_statics.contains(n))
                        .collect(),
                    None => Vec::new(),
                };
                // A TUPLE-UNPACK whose elements were promoted (idna's
                // `_STATUS_VALID, ... = b"VMDI"`): assign_name_targets
                // returns None for the tuple target, so collect the
                // promoted element names here — the static emission below
                // extracts each element at its position.
                if promoted.is_empty()
                    && let Some(pairs) = assign_unpack_indices(a)
                    && let Some(first) = pairs
                        .iter()
                        .find(|(n, _)| promoted_statics.contains(n))
                {
                    promoted.push(first.0.clone());
                    promoted.extend(
                        pairs
                            .iter()
                            .filter(|(n, _)| n != &first.0 && promoted_statics.contains(n))
                            .map(|(n, _)| n.clone()),
                    );
                }
                if promoted.is_empty() {
                    // fall through to the ordinary Assign lowering
                } else {
                let promoted_first = promoted[0].clone();
                let rhs = a
                    .value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map_err(|e| wrap_module_error(&module_filename, e))?;
                let stripped = crate::ast::tree::call::strip_trailing_question(&rhs);
                let is_fallible = stripped.to_string() != rhs.to_string();
                let value_tokens = if is_fallible {
                    quote!(match #stripped {
                        Ok(__rython_v) => __rython_v,
                        Err(__rython_e) => panic!(
                            "module-level `{}` initialization failed: {}",
                            stringify!(#promoted_first),
                            __rython_e
                        ),
                    })
                } else {
                    stripped
                };
                // A TUPLE-UNPACK promotion (`_STATUS_VALID, ... =
                // b"VMDI"` — idna): one SHARED static evaluates the RHS
                // exactly ONCE (Devin review on #263, Finding 4: per-name
                // statics each re-ran the RHS, so side effects repeated
                // and the names could come from different results); each
                // promoted name's static projects its element from it,
                // boxing the element WITHOUT a truncating `as i64`
                // (Finding 3: `a, b = (1.5, 2.5)` boxed 1 instead of
                // 1.5 — only the bytes-unpack element (u8) boxes as an
                // Int, through its own From impl).
                let unpack_pairs = assign_unpack_indices(a);
                // A BYTES-LITERAL RHS (`b"VMDI"` — idna) indexes to u8
                // elements, which box as Ints; any other RHS element
                // boxes as-is (no `as i64` truncation — Devin review on
                // #263, Finding 3: `a, b = (1.5, 2.5)` must stay 1.5).
                let rhs_is_bytes_literal = matches!(
                    &a.value,
                    crate::ExprType::Constant(c)
                        if matches!(&c.0, Some(litrs::Literal::ByteString(_)))
                );
                let shared_ident = unpack_pairs.as_ref().map(|_| {
                    let ident = crate::safe_ident(&format!("__rython_unpack_{}", unpack_counter));
                    unpack_counter += 1;
                    stream.extend(quote! {
                        pub static #ident: std::sync::LazyLock<stdpython::PyValue> =
                            std::sync::LazyLock::new(|| stdpython::PyValue::from(#value_tokens));
                    });
                    module_init_stmts.push(quote!(let _ = &*#ident;));
                    ident
                });
                for n in promoted {
                    let ident = crate::safe_ident(&n);
                    let unpack_at = unpack_pairs
                        .as_ref()
                        .and_then(|pairs| pairs.iter().find(|(pn, _)| pn == &n).map(|(_, i)| *i));
                    // The static's type: the codegen's inferred type when
                    // known, a few recognized stdlib constructors, else the
                    // boxed PyValue (the value model's dynamic fallback).
                    // Boxed values wrap in PyValue::from so the closure's
                    // type matches.
                    let (ty, wrapped) =
                        match module_init_static_ty(&n, &a.value, &options) {
                            Some(ty) => {
                                // A `Vec<String>` static whose init is a
                                // list LITERAL (`UNICODE_SECONDARY_RANGE_
                                // KEYWORD = ["Supplement", ...]` —
                                // charset_normalizer): the literal alone
                                // infers Vec<&'static str>, mismatching
                                // Vec<String> — force each element to own
                                // (round 78).
                                let ty_s = ty.to_string();
                                let ty_is_boxed = ty_s
                                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                                    .filter(|t| !t.is_empty())
                                    .eq(["stdpython", "PyValue"].iter().copied());
                                let wrapped = if ty_s.contains("Vec < String >")
                                    && matches!(&a.value, crate::ExprType::List(_))
                                {
                                    let mut st_opt = options.clone();
                                    st_opt.forced_list_elt = std::rc::Rc::new(Some(
                                        crate::TypeInfo::String,
                                    ));
                                    a.value.clone().to_rust(
                                        ctx.clone(),
                                        st_opt,
                                        symbols.clone(),
                                    )?
                                } else if ty_is_boxed {
                                    // A BOXED static whose inferred type is
                                    // the boxed PyValue but whose initializer
                                    // is a PLAIN value (`_FAILEDTELL:
                                    // Final[_TYPE_FAILEDTELL] = _TYPE_FAILEDTELL.token`
                                    // — an Enum sentinel member, i64 —
                                    // urllib3's util/request): wrap like the
                                    // unknown-type path below — the closure
                                    // must produce the PyValue (round 96).
                                    quote!(stdpython::PyValue::from(#value_tokens))
                                } else {
                                    value_tokens.clone()
                                };
                                (ty, wrapped)
                            }
                            None => {
                                let boxed = match (unpack_at, &shared_ident) {
                                    (Some(i), Some(shared)) => {
                                        let idx = i as i64;
                                        let projected = if rhs_is_bytes_literal {
                                            quote!(PyValue::from(__rython_elt as i64))
                                        } else {
                                            quote!(PyValue::from(__rython_elt))
                                        };
                                        quote! {
                                            match (*#shared).clone().py_index(#idx) {
                                                Ok(__rython_elt) => #projected,
                                                Err(__rython_e) => panic!(
                                                    "module-level `{}` element {} \
                                                     initialization failed: {}",
                                                    stringify!(#n),
                                                    #idx,
                                                    __rython_e
                                                ),
                                            }
                                        }
                                    }
                                    _ => value_tokens.clone(),
                                };
                                (
                                    quote!(stdpython::PyValue),
                                    quote!(stdpython::PyValue::from(#boxed)),
                                )
                            }
                        };
                    stream.extend(quote! {
                        pub static #ident: std::sync::LazyLock<#ty> =
                            std::sync::LazyLock::new(|| #wrapped);
                    });
                    module_init_stmts.push(quote!(let _ = &*#ident;));
                }
                has_module_init_code = true;
                continue;
                }
            }

            // A DEFINITE if/else module value (promoted_conditional): the If
            // statement itself becomes the static's conditional initializer
            // (`if sys.platform == "win32": preferred_clock =
            // time.perf_counter else: ...` → `static preferred_clock:
            // LazyLock<fn() -> f64> = LazyLock::new(|| if ... { ... } else
            // { ... })`). The branch values are stdpython module functions
            // (fn items); the fn-pointer type keeps the call sites
            // (`preferred_clock()`) compiling.
            if let crate::StatementType::If(if_stmt) = &s.statement
                && if_stmt.body.len() == 1
                && if_stmt.orelse.len() == 1
                && let crate::StatementType::Assign(b1) = &if_stmt.body[0].statement
                && let crate::StatementType::Assign(b2) = &if_stmt.orelse[0].statement
                && let [crate::ExprType::Name(n1)] = b1.targets.as_slice()
                && let [crate::ExprType::Name(n2)] = b2.targets.as_slice()
                && n1.id == n2.id
                && promoted_conditional.contains_key(&n1.id)
            {
                let ident = crate::safe_ident(&n1.id);
                let test = if_stmt
                    .test
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map_err(|e| wrap_module_error(&module_filename, e))?;
                let v1 = b1
                    .value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map_err(|e| wrap_module_error(&module_filename, e))?;
                let v2 = b2
                    .value
                    .clone()
                    .to_rust(ctx.clone(), options.clone(), symbols.clone())
                    .map_err(|e| wrap_module_error(&module_filename, e))?;
                // Both branch values are stdpython module FUNCTION reads
                // (`time.perf_counter` / `time.time`): the static is a
                // fn-pointer; anything else falls back to the boxed None
                // initializer (the values are dropped).
                let is_fn_attr = |v: &crate::ExprType| -> bool {
                    matches!(
                        v,
                        crate::ExprType::Attribute(a)
                            if matches!(a.value.as_ref(), crate::ExprType::Name(_))
                    )
                };
                if is_fn_attr(&b1.value) && is_fn_attr(&b2.value) {
                    stream.extend(quote! {
                        pub static #ident: std::sync::LazyLock<fn() -> f64> =
                            std::sync::LazyLock::new(|| {
                                if #test {
                                    #v1
                                } else {
                                    #v2
                                }
                            });
                    });
                } else {
                    stream.extend(quote! {
                        pub static #ident: std::sync::LazyLock<stdpython::PyValue> =
                            std::sync::LazyLock::new(|| {
                                if #test {
                                    stdpython::PyValue::None_
                                } else {
                                    stdpython::PyValue::None_
                                }
                            });
                    });
                }
                module_init_stmts.push(quote!(let _ = &*#ident;));
                has_module_init_code = true;
                continue;
            }

            // Categorize statements into declarations vs executable code
            let is_declaration = Self::is_declaration_statement(&s.statement);

            let init_options = {
                let mut o = options.clone();
                o.hoisted_names = std::rc::Rc::new(init_hoisted.clone());
                o.leaked_loop_targets = std::rc::Rc::new(init_leaked.clone());
                o
            };
            // A module-level `try: <imports> except ImportError: <fallback>`
            // (`from urllib3.contrib.socks import SOCKSProxyManager` —
            // requests' adapters.py; `from .ssltransport import
            // SSLTransport` — urllib3's ssl_.py): rython's imports are
            // STATIC, so the try body always succeeds and the ImportError
            // fallback (dropped in try_stmt.rs) never runs. The try wrapper
            // is meaningless — flatten its body to MODULE level so import
            // statements emit their `use` at module scope (where call sites
            // outside the wrapper can see them) instead of inside the
            // lowered try closure.
            let flattenable_try = match &s.statement {
                crate::StatementType::Try(t) => {
                    !t.handlers.is_empty()
                        && t.orelse.is_empty()
                        && t.finalbody.is_empty()
                        && t.handlers.iter().all(|h| {
                            crate::ast::tree::try_stmt::is_bare_import_error(
                                &h.exception_type,
                            ) && (matches!(
                                h.exception_type,
                                Some(crate::ExprType::Name(_))
                            ) || crate::ast::tree::try_stmt::try_body_contains_import(
                                &t.body,
                            ))
                        })
                }
                _ => false,
            };
            if flattenable_try {
                if let crate::StatementType::Try(t) = &s.statement {
                    // A try/except-ImportError whose HANDLER ASSIGNS a name
                    // the try body IMPORTS (`try: from charset_normalizer
                    // import __version__ as charset_normalizer_version
                    // except ImportError: charset_normalizer_version =
                    // None` — requests/__init__.py): rython's imports are
                    // static, so the import always succeeds and the
                    // fallback never runs — but the fallback's Assign still
                    // makes the name a MODULE-INIT local (hoisted `let`),
                    // which would collide with the flattened import's `use`
                    // alias of a static (E0530). Drop the import: the name
                    // lowers to the module-init local (None in practice).
                    let handler_assigned: std::collections::HashSet<String> = t
                        .handlers
                        .iter()
                        .flat_map(|h| h.body.iter())
                        .filter_map(|bs| match &bs.statement {
                            crate::StatementType::Assign(a) => assign_name_targets(a),
                            _ => None,
                        })
                        .flatten()
                        .collect();
                    for body_stmt in &t.body {
                        // Handler-reassigned names drop from the import
                        // PER NAME: urllib3's ssl_.py handler stores
                        // PROTOCOL_TLS (among others) while the try's
                        // from-import also binds CERT_REQUIRED and
                        // TLSVersion — those unaffected names keep their
                        // `use`s (dropping the whole import left them
                        // unresolved, E0425).
                        let mut body_stmt = body_stmt.clone();
                        if let crate::StatementType::ImportFrom(i) = &mut body_stmt.statement
                        {
                            let root = i.module.split('.').next().unwrap_or("").to_string();
                            let mut dropped: Vec<(String, String)> = Vec::new();
                            i.names.retain(|a| {
                                let bound = a.asname.as_deref().unwrap_or(&a.name);
                                if handler_assigned.contains(bound) {
                                    dropped.push((a.name.clone(), bound.to_string()));
                                    false
                                } else {
                                    true
                                }
                            });
                            // A dropped STDPYTHON name still carries its
                            // imported value: the try body may read it
                            // (`PROTOCOL_SSLv23 = PROTOCOL_TLS` — ssl_.py)
                            // and the handler never runs, so store the
                            // runtime item into the hoisted init local.
                            if i.module == root
                                && crate::ast::tree::import::is_stdpython_module(&root)
                            {
                                for (name, bound) in &dropped {
                                    if crate::ast::tree::import::stdpython_module_item(
                                        &root, name,
                                    ) {
                                        let root_ident = format_ident!("{}", root);
                                        let name_ident = crate::safe_ident(name);
                                        let bound_ident = crate::safe_ident(bound);
                                        module_init_stmts.push(quote! {
                                            #bound_ident = stdpython::#root_ident::#name_ident;
                                        });
                                        has_module_init_code = true;
                                    }
                                }
                            }
                            if i.names.is_empty() {
                                continue;
                            }
                        }
                        let body_is_decl =
                            Self::is_declaration_statement(&body_stmt.statement);
                        let body_tokens = body_stmt
                            .to_rust(ctx.clone(), init_options.clone(), symbols.clone())
                            .map_err(|e| wrap_module_error(&module_filename, e))?;
                        if body_tokens.to_string() != "" {
                            if body_is_decl {
                                stream.extend(body_tokens);
                            } else {
                                module_init_stmts.push(body_tokens);
                                has_module_init_code = true;
                            }
                        }
                    }
                    continue;
                }
            }
            // A DEFINITE try/except module value (promoted_conditional —
            // requests' compat.py is_urllib3_1): the try either completes
            // or raises into the handler, and BOTH store the same name —
            // the value is definitely set. Emit the LazyLock static whose
            // closure runs the try and falls back to the handler's value.
            if let crate::StatementType::Try(t) = &s.statement {
                if let Some(name) = single_assign_name(&t.body)
                    && t.handlers
                        .iter()
                        .all(|h| single_assign_name(&h.body) == Some(name.clone()))
                    && (promoted_conditional.contains_key(&name)
                        || promoted_statics.contains(&name))
                {
                    let ident = crate::safe_ident(&name);
                    let body_assign = match &t.body[0].statement {
                        crate::StatementType::Assign(a) => a.clone(),
                        _ => unreachable!("single_assign_name matched"),
                    };
                    let handler_assign = match &t.handlers[0].body[0].statement {
                        crate::StatementType::Assign(a) => a.clone(),
                        _ => unreachable!("single_assign_name matched"),
                    };
                    let body_val = body_assign
                        .value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                        .map_err(|e| wrap_module_error(&module_filename, e))?;
                    let handler_val = handler_assign
                        .value
                        .clone()
                        .to_rust(ctx.clone(), options.clone(), symbols.clone())
                        .map_err(|e| wrap_module_error(&module_filename, e))?;
                    let body_val = crate::ast::tree::call::strip_trailing_question(&body_val);
                    // The try body runs first; on error the handler's value
                    // applies. The handler may itself fail — propagate.
                    stream.extend(quote! {
                        pub static #ident: std::sync::LazyLock<stdpython::PyValue> =
                            std::sync::LazyLock::new(|| {
                                match (|| -> Result<_, PyException> {
                                    Ok(stdpython::PyValue::from(#body_val))
                                })() {
                                    Ok(__rython_v) => __rython_v,
                                    Err(_) => stdpython::PyValue::from(#handler_val),
                                }
                            });
                    });
                    module_init_stmts.push(quote!(let _ = &*#ident;));
                    has_module_init_code = true;
                    continue;
                }
            }
            let statement = s
                .clone()
                .to_rust(ctx.clone(), init_options, symbols.clone())
                .map_err(|e| wrap_module_error(&module_filename, e))?;
            
            if statement.to_string() != "" {
                if is_declaration {
                    // Declarations go at module level (functions, classes, imports)
                    stream.extend(statement);
                } else {
                    // Executable statements go in module initialization function
                    module_init_stmts.push(statement);
                    has_module_init_code = true;
                }
            }
        }

        // Hoist assigned names to declarations at the top of each generated
        // scope (assignments themselves lower to plain stores). Promoted
        // LazyLock statics are excluded — they have no `let` in the body.
        // Promoted LazyLock statics and mutable statics (issue #115) have
        // no `let` in the init body — their COMPUTED initializers stay in
        // module_init_raw only for the type analysis.
        let init_hoist_skip: std::collections::HashSet<String> = promoted_statics
            .iter()
            .cloned()
            .chain(global_mutables.keys().cloned())
            .collect();
        let init_decls = hoisted_declarations(
            &module_init_raw,
            &ctx,
            &symbols,
            &options,
            &init_hoist_skip,
        );
        if !init_decls.is_empty() {
            module_init_stmts.insert(0, init_decls);
        }
        let main_decls =
            hoisted_declarations(&main_body_raw, &ctx, &symbols, &options, &Default::default());
        if !main_decls.is_empty() {
            main_body_stmts.insert(0, main_decls);
        }

        // A forced numpy backend (--numpy-backend) runs as the first
        // statement of __module_init__, so the choice lives in the program,
        // not just in an env var: a wrong spelling or a backend the crate
        // wasn't built with (missing numpy-rayon / numpy-simd / ... cargo
        // feature) fails loudly when the program starts, not silently.
        if let Some(backend) = &options.numpy_backend {
            has_module_init_code = true;
            module_init_stmts.insert(
                0,
                quote! {
                    match stdpython::numpy::set_backend_by_name(#backend) {
                        Ok(()) => (),
                        Err(e) => panic!("--numpy-backend {}: {}", #backend, e),
                    }
                },
            );
        }

        // Generate module initialization function if needed. Like all
        // generated functions it returns Result so module-level raises and
        // calls propagate.
        if has_module_init_code {
            stream.extend(quote! {
                fn __module_init__() -> Result<(), PyException> {
                    #(#module_init_stmts;)*
                    Ok(())
                }
            });
        }
        
        // A `__main__` block wants a process entry point, and a no_std
        // target has no OS to enter from: refuse loudly instead of emitting
        // a fn main() that cannot link.
        if has_main_code && options.no_std {
            return Err(
                "`if __name__ == \"__main__\":` needs a process entry point, which \
                 the no_std profile does not provide; remove the block (convert the \
                 module as a library) or convert without the no_std profile"
                    .to_string()
                    .into(),
            );
        }

        // If we collected any main code, generate a single consolidated main function
        if has_main_code {
            if is_simple_main_call_pattern {
                // Simple main() call pattern - use user's main function directly as Rust entry point
                // Don't rename the user's main function, just add module init call if needed
                let stream_str = stream.to_string();
                
                // Check if the user's main function is async
                let user_main_is_async = stream_str.contains("pub async fn main (");
                
                if user_main_is_async {
                    // User's async main becomes the Rust entry point. The
                    // runtime attribute is gated on the generated crate's
                    // `async-tokio` feature (rypip declares it default-on
                    // for async binaries); without it a compile_error names
                    // the fix instead of a bare "no main function".
                    let runtime_attr = options.async_runtime.main_attribute();
                    let attr_tokens: proc_macro2::TokenStream = runtime_attr.parse()
                        .unwrap_or_else(|_| quote!(tokio::main)); // fallback to tokio::main
                    
                    // Replace the user's function signature and add the
                    // feature-gated attribute.
                    let new_stream_str = stream_str
                        .replace(
                            "pub async fn main (",
                            &format!(
                                "#[cfg_attr(feature = \"{}\", {})] async fn main(",
                                ASYNC_RUNTIME_FEATURE,
                                runtime_attr
                            ),
                        );
                    stream = new_stream_str.parse::<proc_macro2::TokenStream>()
                        .unwrap_or_else(|_| stream);
                        
                    // If we have module init code, we need to modify the user's main to call it first
                    if has_module_init_code {
                        // This is more complex - we'd need to modify the user's main function body
                        // For now, let's fall back to the rename approach for async functions with module init
                        let renamed_stream_str = Self::rename_main_function_and_references(&stream_str);
                        stream = renamed_stream_str.parse::<proc_macro2::TokenStream>()
                            .unwrap_or_else(|_| stream);

                        stream.extend(quote! {
                            #[cfg_attr(feature = #ASYNC_RUNTIME_FEATURE, #attr_tokens)]
                            async fn main() {
                                let __rython_result: Result<(), PyException> = async {
                                    __module_init__()?;
                                    python_main().await?;
                                    Ok(())
                                }.await;
                                if let Err(e) = __rython_result {
                                    eprintln!("{}", e);
                                    std::process::exit(1);
                                }
                            }
                        });
                    }
                    // A compile_error! applies to the whole crate wherever it
                    // sits, so emit it once for the feature-off build.
                    stream.extend(quote! {
                        #[cfg(not(feature = #ASYNC_RUNTIME_FEATURE))]
                        compile_error!(
                            "this program uses async/await and needs the async runtime; \
                             build the generated crate with --features async-tokio \
                             (rypip enables it by default)"
                        );
                    });
                } else {
                    // User's sync main becomes the Rust entry point
                    // Need to modify the function to match Rust main signature requirements
                    let new_stream_str = Self::convert_python_main_to_rust_entry_point(&stream_str);
                    stream = new_stream_str.parse::<proc_macro2::TokenStream>()
                        .unwrap_or_else(|_| stream);
                    
                    // If we have module init code, we need to modify the user's main to call it first
                    if has_module_init_code {
                        // For simplicity, we'll use the rename approach when module init is needed
                        let renamed_stream_str = Self::rename_main_function_and_references(&stream_str);
                        stream = renamed_stream_str.parse::<proc_macro2::TokenStream>()
                            .unwrap_or_else(|_| stream);

                        stream.extend(quote! {
                            fn main() {
                                let __rython_result = (|| -> Result<(), PyException> {
                                    __module_init__()?;
                                    python_main()?;
                                    Ok(())
                                })();
                                if let Err(e) = __rython_result {
                                    eprintln!("{}", e);
                                    std::process::exit(1);
                                }
                            }
                        });
                    }
                }
            } else {
                // Complex main block - use existing behavior (rename user's main)
                let stream_str = stream.to_string();
                let has_python_main = stream_str.contains("pub fn main (") || stream_str.contains("pub async fn main (");
                
                if has_python_main {
                    // Rename the Python function to avoid conflict with Rust entry point
                    let new_stream_str = Self::rename_main_function_and_references(&stream_str);
                    stream = new_stream_str.parse::<proc_macro2::TokenStream>()
                        .unwrap_or_else(|_| stream);
                    
                    // Update main_body_stmts to call python_main instead of main
                    for stmt in &mut main_body_stmts {
                        let stmt_str = stmt.to_string();
                        let updated_stmt_str = Self::update_main_references(&stmt_str);
                        if updated_stmt_str != stmt_str {
                            if let Ok(new_stmt) = updated_stmt_str.parse::<proc_macro2::TokenStream>() {
                                *stmt = new_stmt;
                            }
                        }
                    }
                }
                
                // Generate the Rust entry point as main() - async if needed
                if needs_async_runtime || has_async_functions {
                    // Parse the runtime attribute string into tokens
                    let runtime_attr = options.async_runtime.main_attribute();
                    let attr_tokens: proc_macro2::TokenStream = runtime_attr.parse()
                        .unwrap_or_else(|_| quote!(tokio::main)); // fallback to tokio::main
                    
                    let init_call = if has_module_init_code {
                        quote!(__module_init__()?;)
                    } else {
                        quote!()
                    };
                    // The runtime attribute applies only when the generated
                    // crate's `async-tokio` feature is enabled (rypip
                    // declares it default-on for async binaries); without
                    // it, `async fn main` has no executor, so a compile_error
                    // names the fix instead of a bare "no main function".
                    stream.extend(quote! {
                        #[cfg(not(feature = #ASYNC_RUNTIME_FEATURE))]
                        compile_error!(
                            "this program uses async/await and needs the async runtime; \
                             build the generated crate with --features async-tokio \
                             (rypip enables it by default)"
                        );
                        #[cfg_attr(feature = #ASYNC_RUNTIME_FEATURE, #attr_tokens)]
                        async fn main() {
                            let __rython_result: Result<(), PyException> = async {
                                #init_call
                                #(#main_body_stmts;)*
                                Ok(())
                            }.await;
                            if let Err(e) = __rython_result {
                                eprintln!("{}", e);
                                std::process::exit(1);
                            }
                        }
                    });
                } else {
                    let init_call = if has_module_init_code {
                        quote!(__module_init__()?;)
                    } else {
                        quote!()
                    };
                    stream.extend(quote! {
                        fn main() {
                            let __rython_result = (|| -> Result<(), PyException> {
                                #init_call
                                #(#main_body_stmts;)*
                                Ok(())
                            })();
                            if let Err(e) = __rython_result {
                                eprintln!("{}", e);
                                std::process::exit(1);
                            }
                        }
                    });
                }
            }
        } else if has_module_init_code {
            // No main block, but we have module initialization code
            // Generate a main function that just runs module initialization
            stream.extend(quote! {
                fn main() {
                    if let Err(e) = __module_init__() {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                }
            });
        }
        // Module-level generated items collected during codegen (issue
        // #109, M3: duck-typing traits like HasSpeak and their per-class
        // impls). Emitted at the TOP of the module, above the functions.
        let pending = options.module_pending_items.borrow_mut().drain(..).collect::<Vec<_>>();
        if !pending.is_empty() {
            let mut prefix = TokenStream::new();
            for item in pending {
                prefix.extend(item);
            }
            prefix.extend(stream);
            stream = prefix;
        }
        Ok(stream)
    }
}

/// Whether a statement body contains any async construct (async def, await,
/// async for, async with) anywhere, including nested control flow. Drives
/// the async-runtime import and the async entry-point decision. Function
/// and class bodies count (their code runs in this module).
pub fn module_contains_async(body: &[crate::Statement]) -> bool {
    fn expr_contains_async(expr: &crate::ExprType) -> bool {
        match expr {
            crate::ExprType::Await(_) => true,
            crate::ExprType::Call(c) => {
                expr_contains_async(&c.func)
                    || c.args.iter().any(expr_contains_async)
                    || c.keywords.iter().any(|k| expr_contains_async(&k.value))
            }
            crate::ExprType::BoolOp(b) => b.values.iter().any(expr_contains_async),
            crate::ExprType::BinOp(b) => {
                expr_contains_async(&b.left) || expr_contains_async(&b.right)
            }
            crate::ExprType::UnaryOp(u) => expr_contains_async(&u.operand),
            crate::ExprType::IfExp(i) => {
                expr_contains_async(&i.test)
                    || expr_contains_async(&i.body)
                    || expr_contains_async(&i.orelse)
            }
            crate::ExprType::Dict(d) => {
                d.keys.iter().flatten().any(expr_contains_async)
                    || d.values.iter().any(expr_contains_async)
            }
            crate::ExprType::Set(s) => s.elts.iter().any(expr_contains_async),
            crate::ExprType::List(items) => items.iter().any(expr_contains_async),
            crate::ExprType::Tuple(t) => t.elts.iter().any(expr_contains_async),
            crate::ExprType::Compare(c) => {
                expr_contains_async(&c.left) || c.comparators.iter().any(expr_contains_async)
            }
            crate::ExprType::Attribute(a) => expr_contains_async(&a.value),
            crate::ExprType::Subscript(s) => {
                expr_contains_async(&s.value)
                    || match &s.kind {
                        crate::SubscriptKind::Index(e) => expr_contains_async(e),
                        crate::SubscriptKind::Slice { lower, upper, step } => {
                            lower.as_deref().is_some_and(expr_contains_async)
                                || upper.as_deref().is_some_and(expr_contains_async)
                                || step.as_deref().is_some_and(expr_contains_async)
                        }
                    }
            }
            crate::ExprType::Starred(s) => expr_contains_async(&s.value),
            crate::ExprType::NamedExpr(n) => {
                expr_contains_async(&n.left) || expr_contains_async(&n.right)
            }
            crate::ExprType::Yield(y) => y.value.as_deref().is_some_and(expr_contains_async),
            crate::ExprType::YieldFrom(y) => expr_contains_async(&y.value),
            crate::ExprType::Lambda(l) => expr_contains_async(&l.body),
            crate::ExprType::JoinedStr(f) => f.values.iter().any(expr_contains_async),
            crate::ExprType::FormattedValue(f) => {
                expr_contains_async(&f.value)
                    || f.format_spec.as_deref().is_some_and(expr_contains_async)
            }
            crate::ExprType::ListComp(l) => {
                expr_contains_async(&l.elt)
                    || l.generators.iter().any(|g| {
                        expr_contains_async(&g.iter) || g.ifs.iter().any(expr_contains_async)
                    })
            }
            crate::ExprType::SetComp(s) => {
                expr_contains_async(&s.elt)
                    || s.generators.iter().any(|g| {
                        expr_contains_async(&g.iter) || g.ifs.iter().any(expr_contains_async)
                    })
            }
            crate::ExprType::DictComp(d) => {
                expr_contains_async(&d.value)
                    || d.generators.iter().any(|g| {
                        expr_contains_async(&g.iter) || g.ifs.iter().any(expr_contains_async)
                    })
            }
            crate::ExprType::GeneratorExp(g) => {
                expr_contains_async(&g.elt)
                    || g.generators.iter().any(|gg| {
                        expr_contains_async(&gg.iter) || gg.ifs.iter().any(expr_contains_async)
                    })
            }
            _ => false,
        }
    }

    fn stmt_contains_async(stmt: &crate::Statement) -> bool {
        match &stmt.statement {
            crate::StatementType::AsyncFunctionDef(_)
            | crate::StatementType::AsyncFor(_)
            | crate::StatementType::AsyncWith(_) => true,
            crate::StatementType::FunctionDef(f) => f.body.iter().any(stmt_contains_async),
            crate::StatementType::ClassDef(c) => c.body.iter().any(stmt_contains_async),
            crate::StatementType::If(i) => {
                i.body.iter().any(stmt_contains_async) || i.orelse.iter().any(stmt_contains_async)
            }
            crate::StatementType::For(f) => {
                expr_contains_async(&f.iter)
                    || f.body.iter().any(stmt_contains_async)
                    || f.orelse.iter().any(stmt_contains_async)
            }
            crate::StatementType::While(w) => {
                w.body.iter().any(stmt_contains_async) || w.orelse.iter().any(stmt_contains_async)
            }
            crate::StatementType::Try(t) => {
                t.body.iter().any(stmt_contains_async)
                    || t.handlers.iter().any(|h| h.body.iter().any(stmt_contains_async))
                    || t.orelse.iter().any(stmt_contains_async)
                    || t.finalbody.iter().any(stmt_contains_async)
            }
            crate::StatementType::With(w) => w.body.iter().any(stmt_contains_async),
            crate::StatementType::Expr(e) => expr_contains_async(&e.value),
            crate::StatementType::Assign(a) => {
                expr_contains_async(&a.value)
                    || a.targets.iter().any(expr_contains_async)
            }
            crate::StatementType::AugAssign(a) => {
                expr_contains_async(&a.target) || expr_contains_async(&a.value)
            }
            crate::StatementType::Return(Some(e)) => expr_contains_async(&e.value),
            crate::StatementType::Raise(r) => {
                r.exc.as_ref().is_some_and(expr_contains_async)
                    || r.cause.as_ref().is_some_and(expr_contains_async)
            }
            crate::StatementType::Assert { test, msg } => {
                expr_contains_async(test)
                    || msg.as_deref().is_some_and(expr_contains_async)
            }
            _ => false,
        }
    }

    body.iter().any(stmt_contains_async)
}

/// Whether a module body imports a module with the given ROOT name (plain
/// or from-import, aliased or not, anywhere in the statement tree). Drives
/// the feature-gated stdpython surfaces: `asyncio` needs `async-tokio`,
/// `urllib` (urllib.request) needs `http-ureq`.
pub fn module_imports_root(body: &[crate::Statement], root: &str) -> bool {
    fn stmt_imports_root(stmt: &crate::Statement, root: &str) -> bool {
        let any = |stmts: &[crate::Statement]| stmts.iter().any(|s| stmt_imports_root(s, root));
        match &stmt.statement {
            crate::StatementType::Import(imp) => imp
                .names
                .iter()
                .any(|a| a.name.split('.').next().is_some_and(|r| r == root)),
            crate::StatementType::ImportFrom(imp) => {
                imp.module.split('.').next().is_some_and(|r| r == root)
            }
            crate::StatementType::If(i) => any(&i.body) || any(&i.orelse),
            crate::StatementType::For(f) => any(&f.body) || any(&f.orelse),
            crate::StatementType::While(w) => any(&w.body) || any(&w.orelse),
            crate::StatementType::Try(t) => {
                any(&t.body)
                    || t.handlers.iter().any(|h| any(&h.body))
                    || any(&t.orelse)
                    || any(&t.finalbody)
            }
            crate::StatementType::With(w) => any(&w.body),
            crate::StatementType::AsyncWith(w) => any(&w.body),
            crate::StatementType::AsyncFor(f) => any(&f.body) || any(&f.orelse),
            crate::StatementType::FunctionDef(f) => any(&f.body),
            crate::StatementType::AsyncFunctionDef(f) => any(&f.body),
            _ => false,
        }
    }
    body.iter().any(|s| stmt_imports_root(s, root))
}

/// Whether a module body imports `asyncio` — see [`module_imports_root`].
pub fn module_imports_asyncio(body: &[crate::Statement]) -> bool {
    module_imports_root(body, "asyncio")
}

/// The plain-Name targets of a module-level Assign, when EVERY target is a
/// plain Name (single-store promotion only applies to name targets; a
/// chained `a = b = expr` where both are names promotes both — urllib3's
/// `__version__ = version = '2.7.0'` in _version.py). Returns None when any
/// target is a subscript/attribute/tuple (mixed targets cannot all be
/// promoted as statics).
fn assign_name_targets(a: &crate::Assign) -> Option<Vec<String>> {    if a.targets.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(a.targets.len());
    for t in &a.targets {
        match t {
            crate::ExprType::Name(n) => out.push(n.id.clone()),
            _ => return None,
        }
    }
    Some(out)
}

/// The single plain-Name target of a one-assignment statement list, or
/// None when the list is not exactly one `x = ...` (used by the definite
/// try/except module-value promotion — requests' compat.py is_urllib3_1).
fn single_assign_name(stmts: &[crate::Statement]) -> Option<String> {
    if stmts.len() != 1 {
        return None;
    }
    match &stmts[0].statement {
        crate::StatementType::Assign(a) if a.targets.len() == 1 => {
            match &a.targets[0] {
                crate::ExprType::Name(n) => Some(n.id.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The (name, index) pairs of a module-level TUPLE-UNPACK target
/// (`_STATUS_VALID, _STATUS_MAPPED, _STATUS_DEVIATION, _STATUS_IGNORED =
/// b"VMDI"` — idna's core.py): each element name binds the value at its
/// position. A plain single-Name target (or any other shape) returns
/// None — its callers handle the simple form. ALL elements must be plain
/// Names (a nested unpack stays unhandled).
fn assign_unpack_indices(a: &crate::Assign) -> Option<Vec<(String, usize)>> {
    if a.targets.len() != 1 {
        return None;
    }
    let crate::ExprType::Tuple(t) = &a.targets[0] else {
        return None;
    };
    let mut out = Vec::with_capacity(t.elts.len());
    for (i, elt) in t.elts.iter().enumerate() {
        let crate::ExprType::Name(n) = elt else {
            return None;
        };
        out.push((n.id.clone(), i));
    }
    Some(out)
}

/// Count stores to each name across MODULE scope: top-level assignments
/// plus everything nested in module-level control flow — if/while/for
/// bodies (and the for target itself, which rebinds every iteration),
/// with bodies and their `as` targets, try bodies and handlers. Nested
/// stores count double so a name assigned once at top level and again in
/// a branch never tallies as once-assigned. Function and class bodies are
/// their own scopes and are not walked.
fn count_module_stores(
    body: &[crate::Statement],
    counts: &mut std::collections::HashMap<String, usize>,
) {
    fn bump_target(target: &crate::ExprType, by: usize, counts: &mut std::collections::HashMap<String, usize>) {
        match target {
            crate::ExprType::Name(n) => {
                *counts.entry(n.id.clone()).or_insert(0) += by;
            }
            crate::ExprType::Tuple(t) => {
                for elt in &t.elts {
                    bump_target(elt, by, counts);
                }
            }
            _ => {}
        }
    }
    for s in body {
        match &s.statement {
            crate::StatementType::Assign(a) => {
                for target in &a.targets {
                    bump_target(target, 1, counts);
                }
            }
            crate::StatementType::AugAssign(a) => bump_target(&a.target, 2, counts),
            crate::StatementType::If(i) => {
                let mut nested = std::collections::HashMap::new();
                count_module_stores(&i.body, &mut nested);
                count_module_stores(&i.orelse, &mut nested);
                for (name, n) in nested {
                    *counts.entry(name).or_insert(0) += n * 2;
                }
            }
            crate::StatementType::While(w) => {
                let mut nested = std::collections::HashMap::new();
                count_module_stores(&w.body, &mut nested);
                count_module_stores(&w.orelse, &mut nested);
                for (name, n) in nested {
                    *counts.entry(name).or_insert(0) += n * 2;
                }
            }
            crate::StatementType::For(f) => {
                bump_target(&f.target, 2, counts);
                let mut nested = std::collections::HashMap::new();
                count_module_stores(&f.body, &mut nested);
                count_module_stores(&f.orelse, &mut nested);
                for (name, n) in nested {
                    *counts.entry(name).or_insert(0) += n * 2;
                }
            }
            crate::StatementType::With(w) => {
                for item in &w.items {
                    if let Some(vars) = &item.optional_vars {
                        bump_target(vars, 2, counts);
                    }
                }
                let mut nested = std::collections::HashMap::new();
                count_module_stores(&w.body, &mut nested);
                for (name, n) in nested {
                    *counts.entry(name).or_insert(0) += n * 2;
                }
            }
            crate::StatementType::Try(t) => {
                let mut nested = std::collections::HashMap::new();
                count_module_stores(&t.body, &mut nested);
                for h in &t.handlers {
                    count_module_stores(&h.body, &mut nested);
                }
                count_module_stores(&t.orelse, &mut nested);
                count_module_stores(&t.finalbody, &mut nested);
                for (name, n) in nested {
                    *counts.entry(name).or_insert(0) += n * 2;
                }
            }
            // Function and class bodies are separate scopes.
            _ => {}
        }
    }
}

/// The static-item type for a module-level constant, when its value is a
/// literal a static can hold (numbers, bools, strings — including a
/// leading unary minus). Non-literal or reassigned module globals keep
/// the old __module_init__ lowering, where referencing them from a
/// function is a loud compile error rather than a silent divergence.
/// Whether a module-level assignment's VALUE is a typing annotation (a
/// container/typing generic subscript, or a `typing.X` attribute): the name
/// is a TYPE ALIAS consumed by annotation resolution, never a runtime value
/// (`_TYPE_REDUCE_RESULT = tuple[typing.Callable[..., object], ...]`,
/// `_TYPE_BODY = typing.Union[...]` — urllib3).
/// Names that SIBLING modules of the crate import FROM this module
/// (`from .constant import _THAI` in charset_normalizer's utils, where
/// `_THAI = 1 << 6` is a module-level value). Such names must be promoted
/// to `pub static` (LazyLock) items in THIS module, or the importing
/// module's `use crate::charset_normalizer::constant::_THAI;` fails with
/// E0432 — a module-init local is invisible to other modules. Only
/// meaningful in multi-module conversions (module_defs populated); a
/// single-module conversion has no siblings and returns empty.
fn sibling_imported_names(options: &PythonOptions) -> std::collections::HashSet<String> {
    use crate::StatementType as ST;
    let mut names = std::collections::HashSet::new();
    if options.module_defs.len() <= 1 || options.this_module_path.is_empty() {
        return names;
    }
    // MODULE-LEVEL sibling imports only. A FUNCTION-LOCAL sibling import
    // (`from .uts46data import uts46data` inside idna's methods) is NOT
    // promoted: the imported value may be a huge heterogeneous table
    // whose boxed-static type change cascades through the consumers'
    // inference (idna 3.10 measured 87 -> 179 rustc errors in round 57);
    // the module-level import of a name still promotes it (the common
    // cross-module constant pattern).
    let this_path = &options.this_module_path;
    for (path, module) in options.module_defs.iter() {
        if *path == *this_path {
            continue;
        }
        // The sibling's own package path: relative imports inside it
        // resolve against ITS path, not this module's.
        let mut sibling_options = options.clone();
        sibling_options.module_path = module_package_path_from_defs(path, &options);
        for stmt in &module.raw.body {
            if let ST::ImportFrom(ifm) = &stmt.statement {
                if ifm.resolved_module_path(&sibling_options) == *this_path {
                    for alias in &ifm.names {
                        names.insert(alias.name.clone());
                    }
                }
            }
        }
    }
    names
}

/// The promotion decision for the module at `path` in the generated crate:
/// the names that WILL be emitted as `pub static` LazyLock statics there.
/// Computed on demand from the module's AST (module_defs), then cached in
/// `options.module_promoted_statics` so the DEFINING module's promotion
/// pass and every IMPORTING module's read lowering agree (name.rs renders
/// `(*name).clone()` for such names). Mirrors the promotion loop in
/// `Module::to_rust` exactly.
pub(crate) fn module_promoted_static_names(
    options: &PythonOptions,
    path: &[String],
) -> std::rc::Rc<std::collections::HashSet<String>> {
    if let Some(cached) = options.module_promoted_statics.borrow().get(path) {
        return cached.clone();
    }
    let Some(module) = options.module_defs.get(path) else {
        return std::rc::Rc::new(std::collections::HashSet::new());
    };
    let mut target = options.clone();
    target.this_module_path = path.to_vec();
    let mut counts = std::collections::HashMap::new();
    count_module_stores(&module.raw.body, &mut counts);
    let free_reads = module_function_free_reads(&module.raw.body);
    let sibling = sibling_imported_names(&target);
    // Issue #115: names written by functions through `global` are MUTABLE
    // statics (Mutex), never immutable LazyLock promotions — an immutable
    // promotion would freeze the initial value. (Function-bound names are
    // already absent from free_reads; this guards the sibling-import path.)
    let (global_written, _) = module_global_write_sets(&module.raw.body);
    let symbols = (**module).clone().find_symbols(crate::SymbolTableScopes::new());
    let mut names = std::collections::HashSet::new();
    // Module-level name → the module-level names its INITIALIZER reads.
    // A name whose initializer reads a PROMOTED name must itself be
    // promoted (url.py's `_IPV6_ADDRZ_RE = re.compile("^" +
    // _IPV6_ADDRZ_PAT + "$")` — the RE is promoted because functions use
    // it, but _IPV6_ADDRZ_PAT is only read by OTHER module-level
    // initializers, not functions; a static's closure cannot reference a
    // module-init local (E0425). Computed transitively to a fixpoint.
    let mut init_reads: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for stmt in &module.raw.body {
        if let crate::StatementType::Assign(a) = &stmt.statement {
            // Only NON-const single-store names participate: const names
            // are already emitted as plain `pub static` items, so the
            // transitive closure must not double-promote them.
            if const_static_type(&a.value).is_some() {
                continue;
            }
            if let Some(targets) = assign_name_targets(a) {
                for n in &targets {
                    if counts.get(n) == Some(&1) {
                        let reads = module_expr_reads(&a.value);
                        init_reads.insert(n.clone(), reads);
                    }
                }
            }
        }
    }
    for stmt in &module.raw.body {
        if let crate::StatementType::If(if_stmt) = &stmt.statement {
            if Module::is_type_checking_test(&if_stmt.test) {
                continue;
            }
            let test_str = format!("{:?}", if_stmt.test);
            if test_str.contains("__name__") && test_str.contains("__main__") {
                continue;
            }
        }
        // A DEFINITE try/except module value (requests' compat.py
        // is_urllib3_1 — same name stored once in the try body and once in
        // a handler, so the value is definitely set): promote like the
        // if/else case, including for SIBLING-imported names.
        if let crate::StatementType::Try(t) = &stmt.statement {
            let body_name = single_assign_name(&t.body);
            if let Some(name) = body_name
                && t.handlers
                    .iter()
                    .all(|h| single_assign_name(&h.body) == Some(name.clone()))
                && counts.get(&name) == Some(&4)
                && (free_reads.contains(&name) || sibling.contains(&name))
            {
                names.insert(name.clone());
            }
        }
        if let crate::StatementType::Assign(a) = &stmt.statement {
            if crate::try_lru_cache_factory(a, Some(&target), &symbols).is_some() {
                continue;
            }
            if let [crate::ExprType::Name(_)] = a.targets.as_slice()
                && crate::ast::tree::assign::builtin_scalar_alias_type(&a.value).is_some()
            {
                continue;
            }
            // EVERY target must be a single-store plain name for the
            // promotion (a chained `__version__ = version = '2.7.0'`
            // promotes both names, but only when each is stored exactly
            // once — a reassigned chained target must not freeze).
            if let Some(targets) = assign_name_targets(a) {
                for n in &targets {
                    if counts.get(n) == Some(&1)
                        && const_static_type(&a.value).is_none()
                        && !is_type_alias_value(&a.value)
                        && !crate::is_rust_bind_call(&a.value)
                        && !global_written.contains(n)
                        && (free_reads.contains(n) || sibling.contains(n))
                        // A name ALSO bound by an import (`SSLTransport =
                        // None` then `from .ssltransport import
                        // SSLTransport` — urllib3's ssl_.py): Python's
                        // LAST binding wins, so the import overrides the
                        // assign; a promoted static would collide with
                        // the import's `use` (E0252) and render the
                        // imported class unusable (E0433). The import
                        // owns the name.
                        && !matches!(
                            symbols.get(n),
                            Some(crate::SymbolTableNode::ImportFrom(_))
                                | Some(crate::SymbolTableNode::Import(_))
                        )
                    {
                        names.insert(n.clone());
                    }
                }
            }
            // A module-level TUPLE-UNPACK (`_STATUS_VALID, _STATUS_MAPPED,
            // _STATUS_DEVIATION, _STATUS_IGNORED = b"VMDI"` — idna's
            // core.py): each element name is a single store binding the
            // value at its position. Promote the ones functions read or
            // siblings import, like the plain-name arm (a module-init
            // local is invisible to function bodies and sibling imports —
            // E0425/E0432).
            if let Some(pairs) = assign_unpack_indices(a) {
                // Promote ALL elements when ANY qualifies: the init
                // statement unpacks the WHOLE value, so a partially
                // promoted unpack would re-assign the static positions
                // (`(*_STATUS_VALID).clone() = ...` — E0070).
                let any = pairs.iter().any(|(n, _i)| {
                    counts.get(n) == Some(&1)
                        && !global_written.contains(n)
                        && (free_reads.contains(n) || sibling.contains(n))
                        && !matches!(
                            symbols.get(n),
                            Some(crate::SymbolTableNode::ImportFrom(_))
                                | Some(crate::SymbolTableNode::Import(_))
                        )
                });
                if any {
                    for (n, _i) in pairs {
                        names.insert(n.clone());
                    }
                }
            }
        }
    }
    // Transitive promotion to a fixpoint: every name a PROMOTED name's
    // initializer reads must also be promoted (url.py's `_IPV6_ADDRZ_RE =
    // re.compile("^" + _IPV6_ADDRZ_PAT + "$")` — the RE is promoted
    // because functions use it, but _IPV6_ADDRZ_PAT is only read by OTHER
    // module-level initializers, never a function; a static's closure
    // cannot reference a module-init local (E0425)).
    loop {
        let mut changed = false;
        let snapshot: std::collections::HashSet<String> = names.clone();
        for (n, reads) in &init_reads {
            if !snapshot.contains(n) {
                continue;
            }
            for r in reads {
                if !names.contains(r) && init_reads.contains_key(r) {
                    names.insert(r.clone());
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let rc = std::rc::Rc::new(names);
    options
        .module_promoted_statics
        .borrow_mut()
        .insert(path.to_vec(), rc.clone());
    rc
}

/// The names a module-level expression READS (bare Names and attribute
/// roots) — used by the transitive static-promotion fixpoint (a promoted
/// static's initializer must not reference a module-init local).
fn module_expr_reads(expr: &crate::ExprType) -> std::collections::HashSet<String> {
    use crate::ExprType as ET;
    let mut out = std::collections::HashSet::new();
    fn walk(e: &crate::ExprType, out: &mut std::collections::HashSet<String>) {
        match e {
            ET::Name(n) => {
                out.insert(n.id.clone());
            }
            ET::Attribute(a) => walk(&a.value, out),
            ET::Call(c) => {
                walk(&c.func, out);
                for a in &c.args {
                    walk(a, out);
                }
                for kw in &c.keywords {
                    walk(&kw.value, out);
                }
            }
            ET::BinOp(op) => {
                walk(&op.left, out);
                walk(&op.right, out);
            }
            ET::BoolOp(op) => {
                for v in &op.values {
                    walk(v, out);
                }
            }
            ET::Compare(c) => {
                walk(&c.left, out);
                for c in &c.comparators {
                    walk(c, out);
                }
            }
            ET::UnaryOp(u) => walk(&u.operand, out),
            ET::Subscript(s) => {
                walk(&s.value, out);
                match &s.kind {
                    crate::SubscriptKind::Index(i) => walk(i, out),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        if let Some(l) = lower {
                            walk(l, out);
                        }
                        if let Some(u) = upper {
                            walk(u, out);
                        }
                        if let Some(st) = step {
                            walk(st, out);
                        }
                    }
                }
            }
            ET::List(l) => {
                for e in l {
                    walk(e, out);
                }
            }
            ET::Tuple(t) => {
                for e in &t.elts {
                    walk(e, out);
                }
            }
            ET::Set(s) => {
                for e in &s.elts {
                    walk(e, out);
                }
            }
            ET::Dict(d) => {
                for k in d.keys.iter().flatten() {
                    walk(k, out);
                }
                for v in &d.values {
                    walk(v, out);
                }
            }
            ET::IfExp(i) => {
                walk(&i.test, out);
                walk(&i.body, out);
                walk(&i.orelse, out);
            }
            ET::Lambda(l) => {
                walk(&l.body, out);
            }
            ET::Starred(s) => walk(&s.value, out),
            ET::Yield(y) => {
                if let Some(v) = &y.value {
                    walk(v, out);
                }
            }
            ET::YieldFrom(y) => walk(&y.value, out),
            ET::Await(a) => walk(&a.value, out),
            ET::JoinedStr(j) => {
                for part in &j.values {
                    if let ET::FormattedValue(f) = part {
                        walk(&f.value, out);
                    }
                }
            }
            // Comprehensions and generator expressions read their element
            // expression and their iterables (`"|".join(x % _subs for x in
            // _variations)` — urllib3's url.py _IPV6_PAT chain): every
            // name in them must be promoted with the static.
            ET::ListComp(lc) => {
                walk(&lc.elt, out);
                for g in &lc.generators {
                    walk(&g.iter, out);
                    for c in &g.ifs {
                        walk(c, out);
                    }
                }
            }
            ET::SetComp(sc) => {
                walk(&sc.elt, out);
                for g in &sc.generators {
                    walk(&g.iter, out);
                    for c in &g.ifs {
                        walk(c, out);
                    }
                }
            }
            ET::DictComp(dc) => {
                walk(&dc.key, out);
                walk(&dc.value, out);
                for g in &dc.generators {
                    walk(&g.iter, out);
                    for c in &g.ifs {
                        walk(c, out);
                    }
                }
            }
            ET::GeneratorExp(ge) => {
                walk(&ge.elt, out);
                for g in &ge.generators {
                    walk(&g.iter, out);
                    for c in &g.ifs {
                        walk(c, out);
                    }
                }
            }
            _ => {}
        }
    }
    walk(expr, &mut out);
    out
}

/// The package path of a module at `path` within module_defs: the parent
/// directory, unless the module IS a package (`__init__.py` — its path is
/// the package dir itself, and no other module has it as a strict prefix).
fn module_package_path_from_defs(
    path: &[String],
    options: &PythonOptions,
) -> Vec<String> {
    let is_package = options.module_defs.keys().any(|k| {
        k.len() > path.len() && k[..path.len()] == path[..]
    });
    if is_package {
        path.to_vec()
    } else {
        path[..path.len().saturating_sub(1)].to_vec()
    }
}

/// Issue #115: per-function-scope `global` accounting over every function
/// scope in the module (module-level defs, methods, nested defs). Returns
/// `(global_written, bound_without_global)`:
/// - `global_written`: names declared `global` in some function scope AND
///   bound there — the writes the mutable-static lowering must carry;
/// - `bound_without_global`: names bound in some function scope —
///   parameters included — WITHOUT a `global` declaration in that scope.
///   Such a binding is a plain local; a module global sharing the name is
///   shadowed there, so the name is disqualified from the mutable-static
///   lowering (a bare read must never mistake the local for the global).
pub(crate) fn module_global_write_sets(
    body: &[crate::Statement],
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
) {
    // Names BOUND by a target expression (assign/for/with targets): plain
    // names and every name inside a destructuring tuple/list.
    fn bind_target(e: &crate::ExprType, out: &mut std::collections::HashSet<String>) {
        match e {
            crate::ExprType::Name(n) => {
                out.insert(n.id.clone());
            }
            crate::ExprType::Tuple(t) => {
                for elt in &t.elts {
                    bind_target(elt, out);
                }
            }
            crate::ExprType::List(l) => {
                for elt in l {
                    bind_target(elt, out);
                }
            }
            crate::ExprType::Starred(s) => bind_target(&s.value, out),
            _ => {}
        }
    }

    // One FUNCTION scope: its `global` declarations and bound names,
    // through control flow but NOT into nested defs (each def is its own
    // scope, visited separately by the outer walk).
    fn scan_scope(
        stmts: &[crate::Statement],
        globals: &mut std::collections::HashSet<String>,
        bound: &mut std::collections::HashSet<String>,
    ) {
        use crate::StatementType as ST;
        for s in stmts {
            match &s.statement {
                ST::Global(names) => globals.extend(names.iter().cloned()),
                ST::Assign(a) => {
                    for t in &a.targets {
                        bind_target(t, bound);
                    }
                }
                ST::AugAssign(a) => bind_target(&a.target, bound),
                ST::AnnotatedName { name, .. } => {
                    bound.insert(name.clone());
                }
                ST::For(f) => {
                    bind_target(&f.target, bound);
                    scan_scope(&f.body, globals, bound);
                    scan_scope(&f.orelse, globals, bound);
                }
                ST::AsyncFor(f) => {
                    bind_target(&f.target, bound);
                    scan_scope(&f.body, globals, bound);
                    scan_scope(&f.orelse, globals, bound);
                }
                ST::While(w) => {
                    scan_scope(&w.body, globals, bound);
                    scan_scope(&w.orelse, globals, bound);
                }
                ST::If(i) => {
                    scan_scope(&i.body, globals, bound);
                    scan_scope(&i.orelse, globals, bound);
                }
                ST::Try(t) => {
                    scan_scope(&t.body, globals, bound);
                    for h in &t.handlers {
                        if let Some(n) = &h.name {
                            bound.insert(n.clone());
                        }
                        scan_scope(&h.body, globals, bound);
                    }
                    scan_scope(&t.orelse, globals, bound);
                    scan_scope(&t.finalbody, globals, bound);
                }
                ST::With(w) => {
                    for item in &w.items {
                        if let Some(v) = &item.optional_vars {
                            bind_target(v, bound);
                        }
                    }
                    scan_scope(&w.body, globals, bound);
                }
                ST::AsyncWith(w) => {
                    for item in &w.items {
                        if let Some(v) = &item.optional_vars {
                            bind_target(v, bound);
                        }
                    }
                    scan_scope(&w.body, globals, bound);
                }
                _ => {}
            }
        }
    }

    // Visit every function scope anywhere in the module (top-level defs,
    // class methods, nested defs, defs under module-level control flow).
    fn visit_defs(
        stmts: &[crate::Statement],
        global_written: &mut std::collections::HashSet<String>,
        bound_without_global: &mut std::collections::HashSet<String>,
    ) {
        use crate::StatementType as ST;
        for s in stmts {
            let f = match &s.statement {
                ST::FunctionDef(f) | ST::AsyncFunctionDef(f) => f,
                ST::ClassDef(c) => {
                    visit_defs(&c.body, global_written, bound_without_global);
                    continue;
                }
                ST::If(i) => {
                    visit_defs(&i.body, global_written, bound_without_global);
                    visit_defs(&i.orelse, global_written, bound_without_global);
                    continue;
                }
                ST::Try(t) => {
                    visit_defs(&t.body, global_written, bound_without_global);
                    for h in &t.handlers {
                        visit_defs(&h.body, global_written, bound_without_global);
                    }
                    visit_defs(&t.orelse, global_written, bound_without_global);
                    visit_defs(&t.finalbody, global_written, bound_without_global);
                    continue;
                }
                ST::For(f) => {
                    visit_defs(&f.body, global_written, bound_without_global);
                    visit_defs(&f.orelse, global_written, bound_without_global);
                    continue;
                }
                ST::While(w) => {
                    visit_defs(&w.body, global_written, bound_without_global);
                    visit_defs(&w.orelse, global_written, bound_without_global);
                    continue;
                }
                ST::With(w) => {
                    visit_defs(&w.body, global_written, bound_without_global);
                    continue;
                }
                ST::AsyncWith(w) => {
                    visit_defs(&w.body, global_written, bound_without_global);
                    continue;
                }
                _ => continue,
            };
            let mut globals = std::collections::HashSet::new();
            let mut bound = std::collections::HashSet::new();
            for p in f
                .args
                .args
                .iter()
                .chain(f.args.posonlyargs.iter())
                .chain(f.args.kwonlyargs.iter())
                .chain(f.args.vararg.iter())
                .chain(f.args.kwarg.iter())
            {
                bound.insert(p.arg.clone());
            }
            scan_scope(&f.body, &mut globals, &mut bound);
            for n in &bound {
                if globals.contains(n) {
                    global_written.insert(n.clone());
                } else {
                    bound_without_global.insert(n.clone());
                }
            }
            // Nested defs inside this function are their own scopes.
            visit_defs(&f.body, global_written, bound_without_global);
        }
    }

    let mut global_written = std::collections::HashSet::new();
    let mut bound_without_global = std::collections::HashSet::new();
    visit_defs(body, &mut global_written, &mut bound_without_global);
    (global_written, bound_without_global)
}

/// Issue #189: for each name a top-level function writes through `global`,
/// classify the function-scope stores. `None` literals are the empty
/// state; a call to a LOCAL class (a `ClassDef` in the module symbols) is
/// the singleton construction. A name qualifies for the typed
/// class-instance static when every store is None except exactly one
/// construction of the SAME class — the map carries name → class for the
/// qualifiers. Any other store shape (a container literal, a second
/// class, a computed value, an augmented assignment) disqualifies the
/// name, keeping the plain Boxed static and its loud conversion error.
fn module_global_class_stores(
    body: &[crate::Statement],
    symbols: &crate::SymbolTableScopes,
) -> std::collections::HashMap<String, String> {
    use crate::StatementType as ST;

    #[derive(Default)]
    struct Stores {
        /// A store of a local class construction (the singleton shape).
        class: Option<String>,
        /// Any store shape the typed static cannot hold.
        other: bool,
    }

    impl Stores {
        fn record(&mut self, value: &crate::ExprType, symbols: &crate::SymbolTableScopes) {
            if crate::is_none_expr(value) {
                return;
            }
            if let crate::ExprType::Call(c) = value
                && let crate::ExprType::Name(f) = c.func.as_ref()
                && matches!(symbols.get(&f.id), Some(crate::SymbolTableNode::ClassDef(_)))
            {
                match &self.class {
                    // A second class (or the same one twice) cannot share
                    // the Option<Class> slot — disqualified below when the
                    // classes disagree; two stores of the SAME class are
                    // still one type (the lazy-init idiom re-derives it).
                    Some(existing) if existing != &f.id => self.other = true,
                    _ => self.class = Some(f.id.clone()),
                }
                return;
            }
            self.other = true;
        }
    }

    // One function scope: its `global` declarations and the stores to
    // them, through control flow but NOT into nested defs (each def is
    // its own scope, and rython's closures drop anyway).
    fn scan_scope(
        stmts: &[crate::Statement],
        globals: &mut std::collections::HashSet<String>,
        stores: &mut std::collections::HashMap<String, Stores>,
        symbols: &crate::SymbolTableScopes,
    ) {
        for s in stmts {
            match &s.statement {
                ST::Global(names) => globals.extend(names.iter().cloned()),
                ST::Assign(a) => {
                    if let [crate::ExprType::Name(n)] = a.targets.as_slice()
                        && globals.contains(&n.id)
                    {
                        stores
                            .entry(n.id.clone())
                            .or_default()
                            .record(&a.value, symbols);
                    }
                }
                ST::AugAssign(a) => {
                    if let crate::ExprType::Name(n) = &a.target && globals.contains(&n.id) {
                        stores.entry(n.id.clone()).or_default().other = true;
                    }
                }
                ST::If(i) => {
                    scan_scope(&i.body, globals, stores, symbols);
                    scan_scope(&i.orelse, globals, stores, symbols);
                }
                ST::While(w) => {
                    scan_scope(&w.body, globals, stores, symbols);
                    scan_scope(&w.orelse, globals, stores, symbols);
                }
                ST::For(f) => {
                    if let crate::ExprType::Name(n) = &f.target && globals.contains(&n.id) {
                        stores.entry(n.id.clone()).or_default().other = true;
                    }
                    scan_scope(&f.body, globals, stores, symbols);
                    scan_scope(&f.orelse, globals, stores, symbols);
                }
                ST::AsyncFor(f) => {
                    if let crate::ExprType::Name(n) = &f.target && globals.contains(&n.id) {
                        stores.entry(n.id.clone()).or_default().other = true;
                    }
                    scan_scope(&f.body, globals, stores, symbols);
                    scan_scope(&f.orelse, globals, stores, symbols);
                }
                ST::Try(t) => {
                    scan_scope(&t.body, globals, stores, symbols);
                    for h in &t.handlers {
                        scan_scope(&h.body, globals, stores, symbols);
                    }
                    scan_scope(&t.orelse, globals, stores, symbols);
                    scan_scope(&t.finalbody, globals, stores, symbols);
                }
                ST::With(w) => scan_scope(&w.body, globals, stores, symbols),
                ST::AsyncWith(w) => scan_scope(&w.body, globals, stores, symbols),
                _ => {}
            }
        }
    }

    let mut out: std::collections::HashMap<String, Stores> = std::collections::HashMap::new();
    for stmt in body {
        // Top-level defs only: each is its own scope with its own `global`
        // declarations (module_global_write_sets' visit_defs walk).
        let fn_body = match &stmt.statement {
            ST::FunctionDef(f) => &f.body,
            ST::AsyncFunctionDef(f) => &f.body,
            _ => continue,
        };
        let mut globals = std::collections::HashSet::new();
        scan_scope(fn_body, &mut globals, &mut out, symbols);
    }
    out.into_iter()
        .filter_map(|(name, stores)| {
            if stores.other {
                None
            } else {
                stores.class.map(|class| (name, class))
            }
        })
        .collect()
}

/// Issue #115: the module-level names lowered as MUTABLE statics, mapped
/// to their [`crate::MutableGlobalKind`]. A name qualifies when a
/// function writes it through `global`, it is never bound as a plain
/// local anywhere (parameters included), and it has exactly one
/// module-level store. The initializer decides the kind: int/float/bool
/// literal → `Scalar` (const Mutex), `None` → `Boxed`
/// (Mutex<PyValue>), string literal → `Str` (LazyLock<Mutex<String>>),
/// any other runtime expression → `Computed` (LazyLock<Mutex<T>>, boxed
/// when no type infers — refined by the module generator once the init
/// analysis has run). Declaration-shaped assigns (type aliases,
/// rust.bind, lru_cache factories, class/function references) and
/// import-owned names are excluded; conditional/multiple stores and the
/// no_std profile (Mutex is std) keep the documented write-drop
/// divergence.
/// Fold module-level `try/except ImportError` guards whose try body's
/// imports are ALL statically unresolvable — external to the generated
/// crate, the stdpython runtime, and the vendored `[python-modules]`
/// deps. Such an import FAILS at runtime exactly as rython drops it, so
/// the HANDLER branch is the module's real body (`try: import brotli
/// except ImportError: brotli = None` — urllib3's response.py, whose
/// guarded BrotliDecoder class then folds away with its `if brotli is
/// not None:` guard; `except (ImportError, AttributeError): ssl = None;
/// class BaseSSLError(...)` — connection.py, whose handler CLASS then
/// emits at module level where sibling imports expect it). A try with
/// any resolvable import keeps the current lowering (the imports
/// succeed). Issue #137.
/// Returns the folded body plus the HANDLER statements the fold made
/// live (spliced in place of a try whose imports all fail): those were
/// skipped by Try::find_symbols and need registering.
/// The statically-decided module names of a module body (issue #137): the
/// single-store None / False constants (typically the folded handler of a
/// failed import guard) and the resolvable never-reassigned `import X`
/// bindings. The ONE definition: the module's own emission installs these
/// on its options, and the crate-wide class index (hierarchy.rs) computes
/// them for every other module so a class under `if brotli is not None:`
/// that the emission folds away is not a sum-type variant either.
pub(crate) fn static_gate_names(
    body: &[crate::Statement],
    module_assign_counts: &std::collections::HashMap<String, usize>,
    global_mutables: &std::collections::HashMap<String, crate::MutableGlobalKind>,
    options: &PythonOptions,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
) {
    let mut none_names = std::collections::HashSet::new();
    let mut false_names = std::collections::HashSet::new();
    for stmt in body {
        if let crate::StatementType::Assign(a) = &stmt.statement
            && a.targets.len() == 1
            && let ExprType::Name(n) = &a.targets[0]
            && module_assign_counts.get(&n.id).copied().unwrap_or(0) == 1
            && !global_mutables.contains_key(&n.id)
        {
            if crate::is_none_expr(&a.value) {
                none_names.insert(n.id.clone());
            } else if matches!(
                &a.value,
                ExprType::Constant(c)
                    if matches!(&c.0, Some(litrs::Literal::Bool(b)) if !b.value())
            ) {
                false_names.insert(n.id.clone());
            }
        }
    }
    // A RESOLVABLE top-level `import X` never reassigned is a
    // statically-truthy module name: `if not ssl:` fallbacks
    // (urllib3's connection.py DummyConnection) fold away, the
    // branches CPython never enters when the import succeeds.
    let mut module_names = std::collections::HashSet::new();
    for stmt in body {
        if let crate::StatementType::Import(imp) = &stmt.statement {
            for al in &imp.names {
                let root = al.name.split('.').next().unwrap_or("");
                let bound = al.asname.clone().unwrap_or_else(|| root.to_string());
                let resolvable = crate::ast::tree::import::is_stdpython_module(root)
                    || options.python_modules.contains(&root.to_string())
                    || options
                        .module_defs
                        .keys()
                        .any(|k| k.first().map(String::as_str) == Some(root));
                if resolvable
                    && module_assign_counts.get(&bound).copied().unwrap_or(0) == 0
                    && !global_mutables.contains_key(&bound)
                {
                    module_names.insert(bound);
                }
            }
        }
    }
    (none_names, false_names, module_names)
}

/// Replace module-level `if` blocks whose branch is statically decided
/// (`sys.version_info` gates and single-store-name gates) with the taken
/// branch's statements, recursively. Defs and class bodies inside the
/// taken branch then lower as ordinary module items.
fn splice_gated_branches(
    body: Vec<crate::Statement>,
    options: &PythonOptions,
) -> Vec<crate::Statement> {
    let mut out = Vec::with_capacity(body.len());
    for s in body {
        if let crate::StatementType::If(i) = &s.statement {
            let taken = crate::ast::tree::if_stmt::version_gate_taken(&i.test)
                .or_else(|| crate::ast::tree::if_stmt::static_name_gate_taken(&i.test, options));
            if let Some(taken) = taken {
                let branch = if taken { &i.body } else { &i.orelse };
                let branch = splice_gated_branches(branch.clone(), options);
                for b in branch {
                    out.push(b);
                }
                continue;
            }
        }
        out.push(s);
    }
    out
}

pub(crate) fn fold_static_import_trys(
    body: &[crate::Statement],
    options: &crate::PythonOptions,
) -> (Vec<crate::Statement>, Vec<crate::Statement>) {
    fn collect_imports<'a>(
        stmts: &'a [crate::Statement],
        out: &mut Vec<&'a crate::StatementType>,
    ) {
        for s in stmts {
            match &s.statement {
                st @ (crate::StatementType::Import(_)
                | crate::StatementType::ImportFrom(_)) => out.push(st),
                crate::StatementType::Try(t) => {
                    collect_imports(&t.body, out);
                    for h in &t.handlers {
                        collect_imports(&h.body, out);
                    }
                    collect_imports(&t.orelse, out);
                    collect_imports(&t.finalbody, out);
                }
                crate::StatementType::If(i) => {
                    collect_imports(&i.body, out);
                    collect_imports(&i.orelse, out);
                }
                _ => {}
            }
        }
    }
    let root_resolvable = |root: &str| -> bool {
        crate::ast::tree::import::is_stdpython_module(root)
            || options.python_modules.contains(&root.to_string())
            || options
                .module_defs
                .keys()
                .any(|k| k.first().map(String::as_str) == Some(root))
    };
    let unresolvable = |st: &crate::StatementType| -> bool {
        match st {
            crate::StatementType::Import(imp) => imp.names.iter().all(|al| {
                let root = al.name.split('.').next().unwrap_or("");
                !root_resolvable(root)
            }),
            crate::StatementType::ImportFrom(ifm) => {
                // Relative imports are crate siblings — resolvable.
                if ifm.level > 0 {
                    return false;
                }
                let root = ifm.module.split('.').next().unwrap_or("");
                !root_resolvable(root)
                    && !options
                        .module_defs
                        .contains_key(&ifm.resolved_module_path(options))
            }
            _ => false,
        }
    };
    // The dual decision: an import whose EVERY name resolves statically
    // (stdpython, a vendored python-module, or a crate sibling) always
    // succeeds, so the ImportError handler is dead.
    let resolvable = |st: &crate::StatementType| -> bool {
        match st {
            crate::StatementType::Import(imp) => imp.names.iter().all(|al| {
                let root = al.name.split('.').next().unwrap_or("");
                root_resolvable(root)
            }),
            crate::StatementType::ImportFrom(ifm) => {
                ifm.level > 0
                    || root_resolvable(ifm.module.split('.').next().unwrap_or(""))
                    || options
                        .module_defs
                        .contains_key(&ifm.resolved_module_path(options))
            }
            _ => false,
        }
    };
    let mut out = Vec::new();
    let mut newly_live = Vec::new();
    for stmt in body {
        // A module-level SELF-ASSIGN (`__version__ = __version__` —
        // urllib3's __init__, a typing/re-export idiom) is a no-op:
        // dropped here so it neither hoists a module-init local that
        // shadows the imported static (E0530) nor counts as a store.
        if let crate::StatementType::Assign(a) = &stmt.statement
            && let [crate::ExprType::Name(t)] = a.targets.as_slice()
            && matches!(&a.value, crate::ExprType::Name(v) if v.id == t.id)
        {
            continue;
        }
        if let crate::StatementType::Try(t) = &stmt.statement
            && t.handlers.len() == 1
            && t.finalbody.is_empty()
            && (t.handlers[0].exception_type.is_none()
                || crate::ast::tree::try_stmt::is_bare_import_error(
                    &t.handlers[0].exception_type,
                ))
        {
            let mut imports = Vec::new();
            collect_imports(&t.body, &mut imports);
            // Like the `external` import check, the failure decision is
            // only meaningful in a multi-module conversion: a lone module
            // must assume an unknown absolute import is a crate sibling.
            if options.module_defs.len() > 1
                && !imports.is_empty()
                && imports.iter().all(|st| unresolvable(st))
            {
                out.extend(t.handlers[0].body.iter().cloned());
                newly_live.extend(t.handlers[0].body.iter().cloned());
                continue;
            }
            // ALL imports resolve → the try path is the real body: splice
            // it (and the else clause, which runs when nothing raised) in
            // place, dropping the dead handler — whose assigns would
            // otherwise hoist module-init locals that collide with the
            // imports' `use` bindings (urllib3's ssl_.py redefines
            // OP_NO_COMPRESSION and friends in its handler).
            if !imports.is_empty() && imports.iter().all(|st| resolvable(st)) {
                out.extend(t.body.iter().cloned());
                out.extend(t.orelse.iter().cloned());
                continue;
            }
        }
        out.push(stmt.clone());
    }
    (out, newly_live)
}

pub(crate) fn module_global_mutable_names(
    body: &[crate::Statement],
    module_assign_counts: &std::collections::HashMap<String, usize>,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> std::collections::HashMap<String, crate::MutableGlobalKind> {
    use crate::MutableGlobalKind as Kind;
    let mut out = std::collections::HashMap::new();
    if options.no_std {
        return out;
    }
    let (global_written, bound_without_global) = module_global_write_sets(body);
    // Issue #137 (urllib3's emscripten fetch): a module value INITIALIZED
    // None at top level, REASSIGNED only inside module-level control flow
    // (`if worker_available(): _fetcher = _StreamingFetcher()` — no
    // function ever writes it, so no `global` appears), and READ by
    // function bodies. Without a static, the init-localized value is
    // invisible to the functions (E0425 on every read). The boxed mutable
    // static carries it: reads render py_global_read (PyValue), module-init
    // stores write through, and a store with no boxed representation warns
    // and stores None (§12 divergence) — here that branch is behind an
    // always-false worker check, so the None matches the runtime value.
    let free_reads = module_function_free_reads(body);
    for s in body {
        let crate::StatementType::Assign(a) = &s.statement else {
            continue;
        };
        let [crate::ExprType::Name(n)] = a.targets.as_slice() else {
            continue;
        };
        if crate::is_none_expr(&a.value)
            && free_reads.contains(&n.id)
            && module_assign_counts.get(&n.id).copied().unwrap_or(0) > 1
            && !global_written.contains(&n.id)
            && !bound_without_global.contains(&n.id)
            && !matches!(
                symbols.get(&n.id),
                Some(crate::SymbolTableNode::ImportFrom(_))
                    | Some(crate::SymbolTableNode::Import(_))
            )
        {
            out.insert(n.id.clone(), Kind::Boxed);
        }
    }
    if global_written.is_empty() {
        return out;
    }
    for s in body {
        let crate::StatementType::Assign(a) = &s.statement else {
            continue;
        };
        // Single-target stores only: a chained `a = b = 0` couples two
        // names to one statement — out of scope for the mutable lowering.
        let [crate::ExprType::Name(n)] = a.targets.as_slice() else {
            continue;
        };
        if !global_written.contains(&n.id)
            || bound_without_global.contains(&n.id)
            || module_assign_counts.get(&n.id) != Some(&1)
        {
            continue;
        }
        // The import owns import-bound names (see the promotion pass).
        if matches!(
            symbols.get(&n.id),
            Some(crate::SymbolTableNode::ImportFrom(_)) | Some(crate::SymbolTableNode::Import(_))
        ) {
            continue;
        }
        let kind = if crate::is_none_expr(&a.value) {
            Kind::Boxed
        } else if let Some(ty) = const_static_type(&a.value) {
            if ty.to_string() == quote!(&'static str).to_string() {
                Kind::Str
            } else {
                Kind::Scalar
            }
        } else {
            // A COMPUTED initializer — only real runtime stores qualify.
            // Declaration-shaped assigns keep their existing lowerings,
            // and a class/function REFERENCE as the value is the
            // callable-as-value divergence, not a mutable value.
            if is_type_alias_value(&a.value)
                || crate::ast::tree::assign::builtin_scalar_alias_type(&a.value).is_some()
                || crate::is_rust_bind_call(&a.value)
                || crate::try_lru_cache_factory(a, Some(options), symbols).is_some()
                || matches!(
                    &a.value,
                    crate::ExprType::Name(v) if matches!(
                        symbols.get(&v.id),
                        Some(crate::SymbolTableNode::ClassDef(_))
                            | Some(crate::SymbolTableNode::FunctionDef(_))
                            | Some(crate::SymbolTableNode::Alias(_))
                    )
                )
            {
                continue;
            }
            // Boxedness is refined by the module generator once the
            // module-init type analysis has run (module_init_static_ty).
            Kind::Computed { boxed: true }
        };
        out.insert(n.id.clone(), kind);
    }
    out
}

/// Names READ as free variables inside function bodies anywhere in the
/// module (top-level and nested): every Name that appears in a function
/// body and is not bound there (param, assignment target, def/class name,
/// loop/with target, walrus target, comprehension target). Module-level
/// values assigned from non-constant expressions are promoted to LazyLock
/// statics when a function reads them — the old lowering hid the value
/// inside __module_init__, where functions cannot see it (issue #137
/// cluster: `log = logging.getLogger(...)` in urllib3 / charset_normalizer).
fn module_function_free_reads(body: &[crate::Statement]) -> std::collections::HashSet<String> {
    use crate::StatementType as ST;
    let mut all_names = std::collections::HashSet::new();
    let mut bound = std::collections::HashSet::new();

    // Collect every Name AND every bound target inside an expression. The
    // free reads are all_names minus bound; a name that is both read and
    // bound (a local, a def name) cancels out.
    fn walk_expr(
        expr: &crate::ExprType,
        all: &mut std::collections::HashSet<String>,
        bound: &mut std::collections::HashSet<String>,
    ) {
        match expr {
            crate::ExprType::Name(n) => {
                all.insert(n.id.clone());
            }
            crate::ExprType::Call(c) => {
                walk_expr(&c.func, all, bound);
                for a in &c.args {
                    walk_expr(a, all, bound);
                }
                for kw in &c.keywords {
                    walk_expr(&kw.value, all, bound);
                }
            }
            crate::ExprType::BinOp(op) => {
                walk_expr(&op.left, all, bound);
                walk_expr(&op.right, all, bound);
            }
            crate::ExprType::BoolOp(op) => {
                for v in &op.values {
                    walk_expr(v, all, bound);
                }
            }
            crate::ExprType::UnaryOp(op) => walk_expr(&op.operand, all, bound),
            crate::ExprType::Compare(cmp) => {
                walk_expr(&cmp.left, all, bound);
                for c in &cmp.comparators {
                    walk_expr(c, all, bound);
                }
            }
            crate::ExprType::IfExp(e) => {
                walk_expr(&e.test, all, bound);
                walk_expr(&e.body, all, bound);
                walk_expr(&e.orelse, all, bound);
            }
            crate::ExprType::NamedExpr(e) => {
                bind_target(&e.left, bound);
                walk_expr(&e.right, all, bound);
            }
            crate::ExprType::Dict(d) => {
                for k in d.keys.iter().flatten() {
                    walk_expr(k, all, bound);
                }
                for v in &d.values {
                    walk_expr(v, all, bound);
                }
            }
            crate::ExprType::Set(s) => {
                for e in &s.elts {
                    walk_expr(e, all, bound);
                }
            }
            crate::ExprType::List(elts) => {
                for e in elts {
                    walk_expr(e, all, bound);
                }
            }
            crate::ExprType::Tuple(t) => {
                for e in &t.elts {
                    walk_expr(e, all, bound);
                }
            }
            crate::ExprType::Attribute(a) => walk_expr(&a.value, all, bound),
            crate::ExprType::Subscript(sub) => {
                walk_expr(&sub.value, all, bound);
                match &sub.kind {
                    crate::SubscriptKind::Index(i) => walk_expr(i, all, bound),
                    crate::SubscriptKind::Slice { lower, upper, step } => {
                        for o in [lower, upper, step].into_iter().flatten() {
                            walk_expr(o, all, bound);
                        }
                    }
                }
            }
            crate::ExprType::Starred(s) => walk_expr(&s.value, all, bound),
            crate::ExprType::Await(e) => walk_expr(&e.value, all, bound),
            crate::ExprType::Yield(y) => {
                if let Some(v) = &y.value {
                    walk_expr(v, all, bound);
                }
            }
            crate::ExprType::YieldFrom(y) => walk_expr(&y.value, all, bound),
            crate::ExprType::FormattedValue(f) => walk_expr(&f.value, all, bound),
            crate::ExprType::JoinedStr(j) => {
                for v in &j.values {
                    walk_expr(v, all, bound);
                }
            }
            crate::ExprType::Lambda(l) => {
                for p in &l.args.args {
                    bound.insert(p.arg.clone());
                }
                for p in &l.args.posonlyargs {
                    bound.insert(p.arg.clone());
                }
                for p in &l.args.kwonlyargs {
                    bound.insert(p.arg.clone());
                }
                if let Some(p) = &l.args.vararg {
                    bound.insert(p.arg.clone());
                }
                if let Some(p) = &l.args.kwarg {
                    bound.insert(p.arg.clone());
                }
                walk_expr(&l.body, all, bound);
            }
            crate::ExprType::ListComp(c) => {
                for g in &c.generators {
                    bind_target(&g.target, bound);
                    walk_expr(&g.iter, all, bound);
                    for cond in &g.ifs {
                        walk_expr(cond, all, bound);
                    }
                }
                walk_expr(&c.elt, all, bound);
            }
            crate::ExprType::SetComp(c) => {
                for g in &c.generators {
                    bind_target(&g.target, bound);
                    walk_expr(&g.iter, all, bound);
                    for cond in &g.ifs {
                        walk_expr(cond, all, bound);
                    }
                }
                walk_expr(&c.elt, all, bound);
            }
            crate::ExprType::GeneratorExp(c) => {
                for g in &c.generators {
                    bind_target(&g.target, bound);
                    walk_expr(&g.iter, all, bound);
                    for cond in &g.ifs {
                        walk_expr(cond, all, bound);
                    }
                }
                walk_expr(&c.elt, all, bound);
            }
            crate::ExprType::DictComp(c) => {
                for g in &c.generators {
                    bind_target(&g.target, bound);
                    walk_expr(&g.iter, all, bound);
                    for cond in &g.ifs {
                        walk_expr(cond, all, bound);
                    }
                }
                walk_expr(&c.key, all, bound);
                walk_expr(&c.value, all, bound);
            }
            _ => {}
        }
    }

    // A Name/Tuple/Starred assignment target binds its Names.
    fn bind_target(target: &crate::ExprType, bound: &mut std::collections::HashSet<String>) {
        match target {
            crate::ExprType::Name(n) => {
                bound.insert(n.id.clone());
            }
            crate::ExprType::Tuple(t) => {
                for e in &t.elts {
                    bind_target(e, bound);
                }
            }
            crate::ExprType::Starred(s) => bind_target(&s.value, bound),
            _ => {}
        }
    }

    fn param_names(args: &crate::Arguments, bound: &mut std::collections::HashSet<String>) {
        for p in args.posonlyargs.iter().chain(args.args.iter()).chain(args.kwonlyargs.iter()) {
            bound.insert(p.arg.clone());
        }
        if let Some(p) = &args.vararg {
            bound.insert(p.arg.clone());
        }
        if let Some(p) = &args.kwarg {
            bound.insert(p.arg.clone());
        }
    }

    fn walk_stmt(
        stmt: &crate::Statement,
        all: &mut std::collections::HashSet<String>,
        bound: &mut std::collections::HashSet<String>,
    ) {
        match &stmt.statement {
            ST::Assign(a) => {
                for t in &a.targets {
                    bind_target(t, bound);
                }
                if let Some(ann) = &a.annotation {
                    walk_expr(ann, all, bound);
                }
                walk_expr(&a.value, all, bound);
            }
            ST::AugAssign(a) => {
                bind_target(&a.target, bound);
                walk_expr(&a.target, all, bound);
                walk_expr(&a.value, all, bound);
            }
            ST::AnnotatedName { name, annotation } => {
                bound.insert(name.clone());
                walk_expr(annotation, all, bound);
            }
            ST::For(f) => {
                bind_target(&f.target, bound);
                walk_expr(&f.iter, all, bound);
                for s in f.body.iter().chain(f.orelse.iter()) {
                    walk_stmt(s, all, bound);
                }
            }
            ST::AsyncFor(f) => {
                bind_target(&f.target, bound);
                walk_expr(&f.iter, all, bound);
                for s in f.body.iter().chain(f.orelse.iter()) {
                    walk_stmt(s, all, bound);
                }
            }
            ST::While(w) => {
                walk_expr(&w.test, all, bound);
                for s in w.body.iter().chain(w.orelse.iter()) {
                    walk_stmt(s, all, bound);
                }
            }
            ST::If(i) => {
                walk_expr(&i.test, all, bound);
                for s in i.body.iter().chain(i.orelse.iter()) {
                    walk_stmt(s, all, bound);
                }
            }
            ST::With(w) => {
                for item in &w.items {
                    walk_expr(&item.context_expr, all, bound);
                    if let Some(v) = &item.optional_vars {
                        bind_target(v, bound);
                    }
                }
                for s in &w.body {
                    walk_stmt(s, all, bound);
                }
            }
            ST::AsyncWith(w) => {
                for item in &w.items {
                    walk_expr(&item.context_expr, all, bound);
                    if let Some(v) = &item.optional_vars {
                        bind_target(v, bound);
                    }
                }
                for s in &w.body {
                    walk_stmt(s, all, bound);
                }
            }
            ST::Try(t) => {
                for s in &t.body {
                    walk_stmt(s, all, bound);
                }
                for h in &t.handlers {
                    if let Some(e) = &h.exception_type {
                        walk_expr(e, all, bound);
                    }
                    if let Some(n) = &h.name {
                        bound.insert(n.clone());
                    }
                    for s in &h.body {
                        walk_stmt(s, all, bound);
                    }
                }
                for s in t.orelse.iter().chain(t.finalbody.iter()) {
                    walk_stmt(s, all, bound);
                }
            }
            ST::FunctionDef(f) | ST::AsyncFunctionDef(f) => {
                bound.insert(f.name.clone());
                param_names(&f.args, bound);
                for s in &f.body {
                    walk_stmt(s, all, bound);
                }
            }
            ST::ClassDef(c) => {
                bound.insert(c.name.clone());
                // Class METHOD bodies read module values; the class body's
                // own assignments bind class attrs (method reads of a bare
                // name resolve to module scope in Python, but over-binding
                // only skips a promotion — never mis-promotes).
                for s in &c.body {
                    walk_stmt(s, all, bound);
                }
            }
            ST::Expr(e) => walk_expr(&e.value, all, bound),
            ST::Return(Some(e)) => walk_expr(&e.value, all, bound),
            ST::Call(c) => walk_expr(
                &crate::ExprType::Call(c.clone()),
                all,
                bound,
            ),
            ST::Assert { test, msg, .. } => {
                walk_expr(test, all, bound);
                if let Some(m) = msg {
                    walk_expr(m, all, bound);
                }
            }
            ST::Raise(r) => {
                if let Some(e) = &r.exc {
                    walk_expr(e, all, bound);
                }
                if let Some(c) = &r.cause {
                    walk_expr(c, all, bound);
                }
            }
            ST::Delete(targets) => {
                for t in targets {
                    walk_expr(t, all, bound);
                }
            }
            _ => {}
        }
    }

    fn walk_module_defs(
            stmt: &crate::Statement,
            all: &mut std::collections::HashSet<String>,
            bound: &mut std::collections::HashSet<String>,
        ) {
            use crate::StatementType as ST2;
            match &stmt.statement {
                // A def/class body anywhere under module-level control flow
                // reads module values; walk it with the full walker (which
                // binds the def's own scope).
                ST2::FunctionDef(_) | ST2::AsyncFunctionDef(_) | ST2::ClassDef(_) => {
                    walk_stmt(stmt, all, bound);
                }
                // Control-flow shells only HOST defs — their own
                // assignments bind MODULE scope, not a function's locals,
                // so they must not enter the bound set.
                ST2::If(i) => {
                    let is_type_checking = matches!(
                        &i.test,
                        crate::ExprType::Name(n) if n.id == "TYPE_CHECKING"
                    ) || matches!(
                        &i.test,
                        crate::ExprType::Attribute(a) if a.attr == "TYPE_CHECKING"
                    );
                    if !is_type_checking {
                        for s in i.body.iter().chain(i.orelse.iter()) {
                            walk_module_defs(s, all, bound);
                        }
                    }
                }
                ST2::While(w) => {
                    for s in w.body.iter().chain(w.orelse.iter()) {
                        walk_module_defs(s, all, bound);
                    }
                }
                ST2::For(f) => {
                    for s in f.body.iter().chain(f.orelse.iter()) {
                        walk_module_defs(s, all, bound);
                    }
                }
                ST2::AsyncFor(f) => {
                    for s in f.body.iter().chain(f.orelse.iter()) {
                        walk_module_defs(s, all, bound);
                    }
                }
                ST2::With(w) => {
                    for s in &w.body {
                        walk_module_defs(s, all, bound);
                    }
                }
                ST2::AsyncWith(w) => {
                    for s in &w.body {
                        walk_module_defs(s, all, bound);
                    }
                }
                ST2::Try(t) => {
                    for s in t.body.iter().chain(t.orelse.iter()).chain(t.finalbody.iter()) {
                        walk_module_defs(s, all, bound);
                    }
                    for h in &t.handlers {
                        for s in &h.body {
                            walk_module_defs(s, all, bound);
                        }
                    }
                }
                _ => {}
            }
        }
        for s in body {
            walk_module_defs(s, &mut all_names, &mut bound);
        }

    all_names
        .difference(&bound)
        .cloned()
        .collect()
}

/// Does `name` have a runtime item in the module at `path` — i.e. is it
/// GENERATED, not a TYPE_CHECKING-only stub (`if TYPE_CHECKING: class
/// BaseHTTPConnection(Protocol)` — urllib3's _base_connection)? TYPE_CHECKING
/// imports of such names must NOT emit `use` statements (the item does not
/// exist in the generated crate), and annotations referencing them resolve
/// to the boxed PyValue instead of a bare struct name (type_ctx.rs).
pub(crate) fn module_def_has_runtime_item(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
) -> bool {
    let Some(module) = options.module_defs.get(path) else {
        return false;
    };
    let module: &crate::Module = module;
    fn scan(body: &[crate::Statement], name: &str, in_type_checking: bool) -> bool {
        use crate::StatementType as ST;
        for s in body {
            match &s.statement {
                ST::FunctionDef(f) | ST::AsyncFunctionDef(f) => {
                    if !in_type_checking && f.name == name {
                        return true;
                    }
                }
                ST::ClassDef(c) => {
                    if !in_type_checking && c.name == name {
                        return true;
                    }
                }
                ST::Assign(a) => {
                    if !in_type_checking
                        && a.targets.iter().any(|t| {
                            matches!(t, crate::ExprType::Name(n) if n.id == name)
                        })
                        // A dropped BUILTIN-CLASS SELF-alias (`str = str` /
                        // `bytes = bytes` — requests' compat's py2 shim):
                        // the no-op self-assign is removed by
                        // fold_static_import_trys and emits no runtime
                        // item, so a sibling re-export of the name
                        // (`from .compat import str` — auth.py) must NOT
                        // emit `use crate::requests::compat::str` (E0603:
                        // nothing public to point at). The name still means
                        // the builtin (calls dispatch to the builtin arm;
                        // the import drops loudly in import.rs).
                        && !(a.targets.len() == 1
                            && matches!(&a.targets[0], crate::ExprType::Name(n) if n.id == name)
                            && matches!(&a.value, crate::ExprType::Name(v) if v.id == name)
                            && crate::ast::tree::assign::is_builtin_class_name(name))
                    {
                        return true;
                    }
                }
                // A stdpython-module RE-EXPORT (`from urllib.parse import
                // urlparse` — requests' compat, round 55): the import
                // emits a `pub use stdpython::...` when the name has a
                // runtime item, so it IS a runtime item of this module.
                ST::ImportFrom(i) => {
                    if !in_type_checking {
                        let first = i.module.split('.').next().unwrap_or("");
                        let hit = i.names.iter().any(|a| {
                            let imported = a.asname.as_deref().unwrap_or(&a.name);
                            imported == name
                                && crate::ast::tree::import::stdpython_module_item(
                                    first, &a.name,
                                )
                        });
                        if hit {
                            return true;
                        }
                    }
                }
                ST::If(i) => {
                    // `if TYPE_CHECKING:` (bare) or `if typing.TYPE_CHECKING:`
                    // (attribute) marks a compile-time-only block.
                    let tc = in_type_checking
                        || matches!(
                            &i.test,
                            crate::ExprType::Name(n) if n.id == "TYPE_CHECKING"
                        )
                        || matches!(
                            &i.test,
                            crate::ExprType::Attribute(a)
                                if a.attr == "TYPE_CHECKING"
                                    && matches!(
                                        a.value.as_ref(),
                                        crate::ExprType::Name(m) if crate::is_typing(&m.id)
                                    )
                        );
                    if scan(&i.body, name, tc) || scan(&i.orelse, name, tc) {
                        return true;
                    }
                }
                ST::While(w) => {
                    if scan(&w.body, name, in_type_checking)
                        || scan(&w.orelse, name, in_type_checking)
                    {
                        return true;
                    }
                }
                ST::For(f) => {
                    if scan(&f.body, name, in_type_checking)
                        || scan(&f.orelse, name, in_type_checking)
                    {
                        return true;
                    }
                }
                ST::With(w) => {
                    if scan(&w.body, name, in_type_checking) {
                        return true;
                    }
                }
                ST::Try(t) => {
                    for part in [&t.body, &t.orelse, &t.finalbody] {
                        if scan(part, name, in_type_checking) {
                            return true;
                        }
                    }
                    for h in &t.handlers {
                        if scan(&h.body, name, in_type_checking) {
                            return true;
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }
    if scan(&module.raw.body, name, false) {
        return true;
    }
    // The name may be a SUBMODULE of the package (`from .util import
    // connection, ssl_` — urllib3's connection.py, where connection and
    // ssl_ are util/connection.rs / util/ssl_.rs): the defining module's
    // body has no item of that name, but the generated crate has a module
    // at `path + [name]`, so the import (`pub use crate::urllib3::util::
    // connection;`) resolves.
    let mut submodule = path.to_vec();
    submodule.push(name.to_string());
    if options.module_defs.contains_key(&submodule) {
        return true;
    }
    // The name may be RE-EXPORTED by the defining module via its own
    // ImportFrom (`from .request import SKIP_HEADER, SKIPPABLE_HEADERS` —
    // urllib3's util/__init__.py): the generated util/mod.rs carries the
    // `pub use crate::urllib3::util::request::SKIP_HEADER;` chain, so an
    // importer's `from .util import SKIP_HEADER` resolves. Follow the
    // chain to the defining module's item.
    module_reexports_item(options, path, name, &mut std::collections::HashSet::new())
}

/// Whether the binding of `name` in THIS scope resolves to a dropped
/// BUILTIN-CLASS self-alias (`str = str`, `bytes = bytes` — requests'
/// compat's py2 shims): the no-op self-assign is removed by
/// fold_static_import_trys, so the name still means the BUILTIN class —
/// a call is the builtin conversion (`str(x, encoding)`), a value read is
/// the class-as-value name string. Follows the local symbol: a bare name
/// (None — the caller's unbound case), a self-alias, a self-assignment,
/// or an ImportFrom chain into a defining module whose canonical binding
/// is such a self-alias. Returns false for a name shadowed by a real
/// user definition (a function, class, or value of the same name).
pub(crate) fn import_binds_builtin_self_alias(
    name: &str,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> bool {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..8 {
        match syms.get(&current) {
            Some(crate::SymbolTableNode::Alias(c)) => {
                if c == &current {
                    return crate::ast::tree::assign::is_builtin_class_name(&current);
                }
                current = c.clone();
            }
            Some(crate::SymbolTableNode::Assign { value, .. }) => {
                return crate::ast::tree::assign::is_builtin_class_name(&current)
                    && matches!(value, crate::ExprType::Name(n) if n.id == current);
            }
            Some(crate::SymbolTableNode::ImportFrom(ifm)) => {
                let path = ifm.resolved_module_path(options);
                let Some(key) = crate::module_defs_key(options, &path) else {
                    return false;
                };
                let defining = ifm
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(&current))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| current.clone());
                let module = &options.module_defs[key];
                let module: &crate::Module = module;
                syms = module.clone().find_symbols(crate::SymbolTableScopes::new());
                current = defining;
            }
            _ => return false,
        }
    }
    false
}

/// Whether the module at `path` binds `name` as a stdlib EXCEPTION ALIAS
/// (`BaseSSLError = ssl.SSLError` — urllib3's connection.py, at top level
/// or inside a try body whose import statically succeeds): such a name
/// has no runtime item — an importer's `use` would fail E0432 — but
/// raise/except guards canonicalize through the returned builtin name.
/// Handler bodies are NOT scanned: rython's imports are static, so a
/// try/except-ImportError always takes the try path.
/// A value expression that statically resolves to a STDPYTHON-module
/// item: the dotted read (`ssl.VERIFY_X509_PARTIAL_CHAIN`) or the
/// version-probing getattr spelling with a literal name (`getattr(ssl,
/// "VERIFY_X509_PARTIAL_CHAIN", 0x80000)` — urllib3's ssl_.py; the fold
/// in call.rs makes the same decision at render time). Returns the
/// (module, item) pair when the runtime module has the item.
pub(crate) fn stdlib_const_attr(value: &crate::ExprType) -> Option<(String, String)> {
    let (module, item) = match value {
        crate::ExprType::Attribute(attr) => {
            let crate::ExprType::Name(m) = attr.value.as_ref() else {
                return None;
            };
            (m.id.clone(), attr.attr.clone())
        }
        crate::ExprType::Call(call) => {
            let crate::ExprType::Name(f) = call.func.as_ref() else {
                return None;
            };
            if f.id != "getattr" || call.args.len() < 2 {
                return None;
            }
            let crate::ExprType::Name(m) = &call.args[0] else {
                return None;
            };
            let crate::ExprType::Constant(c) = &call.args[1] else {
                return None;
            };
            let Some(litrs::Literal::String(s)) = &c.0 else {
                return None;
            };
            (m.id.clone(), s.value().to_string())
        }
        _ => return None,
    };
    (crate::ast::tree::import::is_stdpython_module(&module)
        && crate::ast::tree::import::stdpython_module_item(&module, &item))
    .then_some((module, item))
}

/// Whether `name` is bound in the CURRENT module by an ALIASED import of
/// an EXTERNAL module (`from http.client import HTTPResponse as
/// _HttplibHTTPResponse` — urllib3's response.py). The symbol table only
/// keeps the Alias hop to the canonical name, which a LATER local class
/// of the same name shadows (`class HTTPResponse(...)`), so following the
/// alias would wrongly resolve to the local class; the module's own AST
/// still carries the truth.
pub(crate) fn aliased_external_import(
    name: &str,
    options: &crate::PythonOptions,
) -> bool {
    let Some(module) = options.module_defs.get(&options.this_module_path) else {
        return false;
    };
    let module: &crate::Module = module;
    fn scan(
        body: &[crate::Statement],
        name: &str,
        options: &crate::PythonOptions,
    ) -> bool {
        use crate::StatementType as ST;
        for s in body {
            match &s.statement {
                ST::ImportFrom(i) => {
                    if i.names
                        .iter()
                        .any(|a| a.asname.as_deref() == Some(name))
                    {
                        let root = i.module.split('.').next().unwrap_or("");
                        let external = i.level == 0
                            && !crate::ast::tree::import::is_stdpython_module(root)
                            && !options
                                .python_modules
                                .contains(&root.to_string())
                            && !options
                                .module_defs
                                .contains_key(&i.resolved_module_path(options));
                        if external {
                            return true;
                        }
                    }
                }
                ST::Try(t) => {
                    if scan(&t.body, name, options) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    scan(&module.raw.body, name, options)
}

pub(crate) fn module_def_exception_alias(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
) -> Option<&'static str> {
    let module = options.module_defs.get(path)?;
    let module: &crate::Module = module;
    fn scan(body: &[crate::Statement], name: &str) -> Option<&'static str> {
        use crate::StatementType as ST;
        for s in body {
            match &s.statement {
                ST::Assign(a) => {
                    if a.targets
                        .iter()
                        .any(|t| matches!(t, crate::ExprType::Name(n) if n.id == name))
                        && let crate::ExprType::Attribute(attr) = &a.value
                        && let crate::ExprType::Name(m) = attr.value.as_ref()
                        && let Some(c) =
                            crate::ast::tree::raise_stmt::stdlib_exception_canonical(
                                &m.id, &attr.attr,
                            )
                    {
                        return Some(c);
                    }
                }
                ST::Try(t) => {
                    if let Some(c) = scan(&t.body, name) {
                        return Some(c);
                    }
                }
                _ => {}
            }
        }
        None
    }
    scan(&module.raw.body, name)
}

/// Whether the module at `path` actually generates a PATH ITEM named
/// `name` — a `pub static` (const or promoted), a `pub fn`, or a `pub
/// struct` — so a module-path read (`util::ssl_::PROTOCOL_TLS`) resolves.
/// Strictly weaker than [`module_def_has_runtime_item`]: a body Assign
/// that only lands in `__module_init__` (a try/except-conditional value
/// that is never promoted) is NOT a path item — reading it as
/// `module::NAME` is E0425, and the read must box to None (the
/// dynamic-module-member divergence).
pub(crate) fn module_def_has_path_item(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
) -> bool {
    let Some(module) = options.module_defs.get(path) else {
        return false;
    };
    let module: &crate::Module = module;
    // A module-level FUNCTION or CLASS is a `pub fn` / `pub struct`.
    for s in &module.raw.body {
        match &s.statement {
            crate::StatementType::FunctionDef(f) | crate::StatementType::AsyncFunctionDef(f) => {
                if f.name == name {
                    return true;
                }
            }
            crate::StatementType::ClassDef(c) => {
                if c.name == name {
                    return true;
                }
            }
            _ => {}
        }
    }
    // A const-literal Assign emits a plain `pub static` — but only when
    // the name is assigned EXACTLY ONCE at module level. A const that is
    // REASSIGNED later (`HAS_NEVER_CHECK_COMMON_NAME = False` then a
    // conditional reassignment in a try — urllib3's ssl_.py) stays a
    // module-init local, not a path item; reading it as `ssl_::
    // HAS_NEVER_CHECK_COMMON_NAME` must box to None.
    // The module's OWN single-store accounting (the const-static
    // emission condition `module_assign_counts.get(n) == Some(&1)`):
    // conditional stores count DOUBLE, so a const with a conditional
    // reassignment is NOT a plain static.
    let mut counts = std::collections::HashMap::new();
    count_module_stores(&module.raw.body, &mut counts);
    if counts.get(name) == Some(&1) && module.raw.body.iter().any(|s| {
        matches!(&s.statement, crate::StatementType::Assign(a)
            if a.targets.iter().any(|t| {
                matches!(t, crate::ExprType::Name(n) if n.id == name)
            })
                && const_static_type(&a.value).is_some())
    }) {
        return true;
    }
    // A non-const single-store name READ BY A FUNCTION (or imported by a
    // sibling) promotes to a `pub static LazyLock`.
    module_promoted_static_names(options, path).contains(name)
        // A SUBMODULE of the package (`util.util` — urllib3's pyopenssl,
        // where `from .. import util` then `util.util.to_bytes(...)`
        // names the util/util.py module): the module itself is the path
        // item the next attribute segment resolves into.
        || {
            let mut sub = path.to_vec();
            sub.push(name.to_string());
            options.module_defs.contains_key(&sub)
        }
}

/// Whether the module at `path` RE-EXPORTS `name` through one of its own
/// ImportFrom statements (`from .request import SKIP_HEADER` in urllib3's
/// util/__init__.py): the generated module re-exports the name, so a
/// sibling importing `from .util import SKIP_HEADER` resolves. The chain
/// is followed depth-first with a visited set (a cycle of re-exports is
/// not a runtime item).
fn module_reexports_item(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
    visited: &mut std::collections::HashSet<Vec<String>>,
) -> bool {
    if !visited.insert(path.to_vec()) {
        return false;
    }
    let Some(module) = options.module_defs.get(path) else {
        return false;
    };
    let module: &crate::Module = module;
    use crate::StatementType as ST;
    for s in &module.raw.body {
        // A plain `import json` binding the name (requests' compat.py
        // re-exports stdlib json): the name resolves through the
        // stdpython glob, so the re-export has a runtime item.
        if let ST::Import(im) = &s.statement {
            if im.names.iter().any(|a| {
                a.asname.as_deref() == Some(name)
                    || (a.asname.is_none()
                        && a.name.split('.').next() == Some(name))
            }) && im
                .names
                .iter()
                .any(|a| a.name.split('.').next().is_some_and(crate::is_stdpython_module))
            {
                return true;
            }
            continue;
        }
        let ST::ImportFrom(i) = &s.statement else { continue };
        // The import must bind OUR name (as itself or with an asname).
        if !i.names.iter().any(|a| {
            a.asname.as_deref() == Some(name) || (a.asname.is_none() && a.name == name)
        }) {
            continue;
        }
        // Resolve the re-export's defining module in THIS module's package
        // context (options.module_path is the caller's context; set it to
        // the defining module's package path, like ImportFrom::to_rust's
        // caller does). An __init__ module's path IS its package path;
        // otherwise the package is the parent.
        let is_package = options
            .module_defs
            .keys()
            .any(|k| k.len() > path.len() && k[..path.len()] == path[..]);
        let mut ctx = options.clone();
        ctx.module_path = if is_package {
            path.to_vec()
        } else {
            path[..path.len().saturating_sub(1)].to_vec()
        };
        let target = i.resolved_module_path(&ctx);
        // The re-export target may itself be a module of the crate whose
        // item exists, or another re-export chain.
        let defining = i
            .names
            .iter()
            .find(|a| a.asname.as_deref() == Some(name) || (a.asname.is_none() && a.name == name))
            .map(|a| a.name.clone())
            .unwrap_or_else(|| name.to_string());
        if !target.is_empty() && options.module_defs.contains_key(&target) {
            let mut sub = target.clone();
            sub.push(defining.clone());
            if options.module_defs.contains_key(&sub)
                || module_reexports_item(options, &target, &defining, visited)
                || scan_module_body_for_item(options, &target, &defining)
            {
                return true;
            }
        }
    }
    false
}

/// Whether the module at `path` directly defines `name` (a function, class,
/// or assignment) OUTSIDE of TYPE_CHECKING — the leaf of a re-export chain.
fn scan_module_body_for_item(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
) -> bool {
    let Some(module) = options.module_defs.get(path) else {
        return false;
    };
    let module: &crate::Module = module;
    fn scan(body: &[crate::Statement], name: &str, in_type_checking: bool) -> bool {
        use crate::StatementType as ST;
        for s in body {
            match &s.statement {
                ST::FunctionDef(f) | ST::AsyncFunctionDef(f) => {
                    if !in_type_checking && f.name == name {
                        return true;
                    }
                }
                ST::ClassDef(c) => {
                    if !in_type_checking && c.name == name {
                        return true;
                    }
                }
                ST::Assign(a) => {
                    if !in_type_checking
                        && a.targets.iter().any(|t| {
                            matches!(t, crate::ExprType::Name(n) if n.id == name)
                        })
                    {
                        return true;
                    }
                }
                // A conditional DEFINITION (`if sys.version_info >= (3, 11):
                // def where(): ...` — certifi's core.py): the function is
                // emitted (the version branch is the modern one), so a
                // sibling import of it resolves. Recurse into nested
                // statement lists, skipping TYPE_CHECKING blocks.
                ST::If(i) => {
                    let tc = in_type_checking
                        || matches!(
                            &i.test,
                            crate::ExprType::Name(n) if n.id == "TYPE_CHECKING"
                        )
                        || matches!(
                            &i.test,
                            crate::ExprType::Attribute(a)
                                if a.attr == "TYPE_CHECKING"
                                    && matches!(
                                        a.value.as_ref(),
                                        crate::ExprType::Name(m) if crate::is_typing(&m.id)
                                    )
                        );
                    if scan(&i.body, name, tc) || scan(&i.orelse, name, tc) {
                        return true;
                    }
                }
                ST::Try(t) => {
                    for part in [&t.body, &t.orelse, &t.finalbody] {
                        if scan(part, name, in_type_checking) {
                            return true;
                        }
                    }
                    for h in &t.handlers {
                        if scan(&h.body, name, in_type_checking) {
                            return true;
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }
    scan(&module.raw.body, name, false)
}

fn is_type_alias_value(value: &crate::ExprType) -> bool {
    match value {
        crate::ExprType::Subscript(sub) => match sub.value.as_ref() {
            crate::ExprType::Name(n) => matches!(
                n.id.as_str(),
                "tuple" | "Tuple" | "list" | "List" | "dict" | "Dict" | "set" | "Set"
                    | "frozenset" | "Union" | "Optional" | "Callable" | "Iterable"
                    | "Sequence" | "Mapping" | "MutableMapping" | "Type" | "Literal"
                    | "Any" | "Generator" | "Iterator" | "SupportsRead" | "SupportsItems"
                    | "IO" | "ClassVar"
            ),
            crate::ExprType::Attribute(a) => {
                matches!(a.value.as_ref(), crate::ExprType::Name(n) if crate::is_typing(&n.id))
            }
            _ => false,
        },
        crate::ExprType::Attribute(a) => {
            matches!(a.value.as_ref(), crate::ExprType::Name(n) if crate::is_typing(&n.id))
        }
        _ => false,
    }
}

pub(crate) fn const_static_type(value: &crate::ExprType) -> Option<TokenStream> {    match value {
        crate::ExprType::Constant(c) => match &c.0 {
            Some(litrs::Literal::Integer(_)) => Some(quote!(i64)),
            Some(litrs::Literal::Float(_)) => Some(quote!(f64)),
            Some(litrs::Literal::Bool(_)) => Some(quote!(bool)),
            Some(litrs::Literal::String(_)) => Some(quote!(&'static str)),
            _ => None,
        },
        crate::ExprType::UnaryOp(op) => {
            if !matches!(op.op, crate::ast::tree::unary_op::Ops::USub) {
                return None;
            }
            match const_static_type(&op.operand) {
                Some(ty) if ty.to_string() == "i64" || ty.to_string() == "f64" => Some(ty),
                _ => None,
            }
        }
        // Integer bitwise/shift expressions (`1 << 6`, `1 | 2`) are
        // constant: they render as plain Rust operators (bin_ops
        // `generate_rust_code`), so the module-level constant machinery
        // can emit `pub static X: i64 = (1) << (6);` (charset_normalizer's
        // `_THAI = 1 << 6` flags). Only the bitwise/shift family is safe:
        // Add/Sub/Mult route through py_add/py_sub/py_mul (not static
        // initializers), and Div/FloorDiv/Mod/Pow have Python-specific
        // semantics that the plain operator would not reproduce.
        crate::ExprType::BinOp(op) => {
            if !matches!(
                op.op,
                crate::ast::tree::bin_ops::BinOps::LShift
                    | crate::ast::tree::bin_ops::BinOps::RShift
                    | crate::ast::tree::bin_ops::BinOps::BitOr
                    | crate::ast::tree::bin_ops::BinOps::BitXor
                    | crate::ast::tree::bin_ops::BinOps::BitAnd
            ) {
                return None;
            }
            match (
                const_static_type(&op.left),
                const_static_type(&op.right),
            ) {
                (Some(l), Some(r)) if l.to_string() == "i64" && r.to_string() == "i64" => {
                    Some(quote!(i64))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The Rust type for a promoted module-level static: the codegen's inferred
/// type when it has one (`options.name_types`), else a few recognized
/// stdlib constructors (`datetime.date(...)` — urllib3's RECENT_DATE), else
/// None (the caller boxes the value into PyValue).
fn module_init_static_ty(
    name: &str,
    value: &crate::ExprType,
    options: &crate::PythonOptions,
) -> Option<TokenStream> {
    // A type containing the UNINFERRED `_` (`PyDict<_, _>` — a dict of
    // EXTERNAL values that all box to None, urllib3's pyopenssl
    // `_stdlib_to_openssl_verify`): `_` is not allowed in static type
    // signatures (E0121), and a boxed-None dict cannot be a typed PyDict
    // (PyValue has no Hash/Eq — E0277). Fall back to the boxed PyValue.
    if let Some(t) = options.name_types.get(name)
        && !type_contains_uninferred(t)
    {
        return Some(t.to_rust_type());
    }
    if let crate::ExprType::Call(c) = value
        && let crate::ExprType::Attribute(a) = c.func.as_ref()
        && let crate::ExprType::Name(n) = a.value.as_ref()
        && let Some(mod_) = crate::StdModule::from_name(&n.id)
    {
        // The MODULE resolves through the StdModule registry (the one
        // place module names exist); only the callee's function name
        // remains a string, matched at this single boundary. These
        // constructors give the static a typed Rust shape instead of the
        // boxed PyValue fallback.
        match (mod_, crate::DatetimeType::from_name(&a.attr)) {
            (crate::StdModule::Datetime, Some(crate::DatetimeType::Date)) => {
                return Some(quote!(stdpython::datetime::date));
            }
            (crate::StdModule::Datetime, Some(crate::DatetimeType::DateTime)) => {
                return Some(quote!(stdpython::datetime::datetime));
            }
            (crate::StdModule::Datetime, Some(crate::DatetimeType::Timedelta)) => {
                return Some(quote!(stdpython::datetime::timedelta));
            }
            // `re.compile(...)` — a compiled pattern (`_TARGET_RE =
            // re.compile(...)`): the static holds the raw Regex, so
            // `.match()`/`.search()`/`.fullmatch()` on it dispatch
            // through the runtime's PyRegexOps instead of boxing the
            // pattern in a PyValue that has no such methods (round 72).
            (crate::StdModule::Re, _) if a.attr == "compile" => {
                return Some(quote!(stdpython::stdlib::re::Regex));
            }
            _ => {}
        }
    }
    None
}

/// Whether a type contains the UNINFERRED placeholder (`TypeInfo::PyObject`
/// renders as `_`) anywhere — dict/list elements of external-module reads.
pub(crate) fn type_contains_uninferred(t: &crate::TypeInfo) -> bool {
    match t {
        crate::TypeInfo::PyObject => true,
        crate::TypeInfo::Vec(inner) | crate::TypeInfo::Option(inner) | crate::TypeInfo::Borrowed(inner) => {
            type_contains_uninferred(inner)
        }
        crate::TypeInfo::Dict(k, v) => type_contains_uninferred(k) || type_contains_uninferred(v),
        crate::TypeInfo::Tuple(ts) => ts.iter().any(type_contains_uninferred),
        _ => false,
    }
}

/// Declarations for every name assigned in a statement list, so
/// nested-block assignments store into scope-level variables instead of
/// creating shadowing bindings. Scope analysis decides which need `mut`.
fn hoisted_name_set(
    body: &[crate::Statement],
    ctx: &crate::CodeGenContext,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
) {
    let mut symbols = symbols.clone();
    for s in body {
        symbols = s.clone().find_symbols(symbols);
    }
    let scope =
        crate::analyze_scope_with(body, &[], &crate::class_call_resolver(ctx, &symbols, options));
    let hoisted = scope
        .assigned
        .iter()
        .chain(scope.needs_mut.iter())
        .cloned()
        .collect();
    (hoisted, scope.leaked_loop_targets)
}

fn hoisted_declarations(
    body: &[crate::Statement],
    ctx: &crate::CodeGenContext,
    symbols: &crate::SymbolTableScopes,
    options: &crate::PythonOptions,
    skip: &std::collections::HashSet<String>,
) -> TokenStream {
    // Class-aware mutation facts need the block's own assignments in the
    // symbol table (`c = Counter(...)` then `c.bump()` needs `c` mutable).
    let mut symbols = symbols.clone();
    for s in body {
        symbols = s.clone().find_symbols(symbols);
    }
    let scope =
        crate::analyze_scope_with(body, &[], &crate::class_call_resolver(ctx, &symbols, options));
    let mut out = TokenStream::new();
    for name in &scope.assigned {
        // rust.bind names are compile-time symbols: the declaration
        // assignment lowers to nothing, so there is no runtime binding to
        // hoist — declaring one would be a dead variable.
        if matches!(symbols.get(name), Some(SymbolTableNode::RustBinding(_))) {
            continue;
        }
        // Promoted LazyLock statics have no `let` binding in the init body.
        if skip.contains(name) {
            continue;
        }
        let ident = crate::safe_ident(name);
        if scope.needs_mut.contains(name) {
            if scope.closure_captured_uninit.contains(name) {
                // Captured by a generated try/handler closure while possibly
                // uninitialized: Default-initialize so rustc accepts the
                // capture (issue #78).
                out.extend(quote!(let mut #ident = Default::default();));
            } else {
                out.extend(quote!(let mut #ident;));
            }
        } else {
            out.extend(quote!(let #ident;));
        }
    }
    out
}

/// Rebuild a statement-level codegen error so it points at the module's real
/// source file. Statement errors carry a `<module>` placeholder filename in
/// their location; this substitutes the actual filename and preserves the
/// structured fields (message, help) so downstream consumers — the proc
/// macro in particular — can render precise diagnostics.
fn wrap_module_error(
    filename: &str,
    e: Box<dyn std::error::Error>,
) -> Box<dyn std::error::Error> {
    if let Some(inner) = e.downcast_ref::<crate::Error>() {
        let message = inner.get_field("message").unwrap_or_default().to_string();
        let location = inner
            .get_field("location")
            .unwrap_or("<module>")
            .replace("<module>", filename);
        let help = inner.get_field("help").unwrap_or_default().to_string();
        return Box::from(crate::codegen_error(
            crate::SourceLocation::new(location),
            message,
            help,
        ));
    }
    Box::from(crate::codegen_error(
        crate::SourceLocation::new(filename),
        crate::format_error_chain(e.as_ref()),
        "",
    ))
}

impl Module {
    /// Check if the __name__ == "__main__" block contains only a simple call to main()
    /// This includes patterns like:
    /// - main()
    /// - result = main()
    /// - sys.exit(main())
    /// Whether an if-test is `TYPE_CHECKING` (or `typing.TYPE_CHECKING`):
    /// the compile-time-only guard that never runs at runtime — its block
    /// (imports, type-only class definitions) must be skipped entirely
    /// (requests' _types.py).
    fn is_type_checking_test(test: &crate::ExprType) -> bool {
        match test {
            crate::ExprType::Name(n) => n.id == "TYPE_CHECKING",
            crate::ExprType::Attribute(a) => {
                matches!(a.value.as_ref(), crate::ExprType::Name(m) if crate::is_typing(&m.id))
                    && a.attr == "TYPE_CHECKING"
            }
            _ => false,
        }
    }

    fn is_simple_main_call_block(body: &[crate::Statement]) -> bool {
        // Must have exactly one statement
        if body.len() != 1 {
            return false;
        }
        
        let stmt = &body[0];
        match &stmt.statement {
            // Pattern 1: main() - direct call as expression statement
            crate::StatementType::Expr(expr) => {
                Self::is_main_function_call(&expr.value)
            },
            // Pattern 2: result = main() - assignment from main call
            crate::StatementType::Assign(assign) => {
                // Should have exactly one target and the value should be a main() call
                assign.targets.len() == 1 && Self::is_main_function_call(&assign.value)
            },
            // Pattern 3: sys.exit(main()) - call with main() as argument
            crate::StatementType::Call(call) => {
                // Check if any of the arguments is a main() call
                call.args.iter().any(|arg| Self::is_main_function_call(arg))
            },
            _ => false,
        }
    }
    
    /// Check if an expression is a call to a function named "main"
    fn is_main_function_call(expr: &crate::ExprType) -> bool {
        match expr {
            crate::ExprType::Call(call) => {
                match call.func.as_ref() {
                    crate::ExprType::Name(name) => name.id == "main",
                    _ => false,
                }
            },
            _ => false,
        }
    }
    
    /// Determine if a statement is a declaration (can stay at module level) or executable code (needs to go in init function)
    fn is_declaration_statement(stmt_type: &crate::StatementType) -> bool {
        use crate::StatementType::*;
        match stmt_type {
            // These are declarations that can stay at module level
            FunctionDef(_) | AsyncFunctionDef(_) | ClassDef(_) | Import(_) | ImportFrom(_)
            | Global(_) | Nonlocal(_) | AnnotatedName { .. } => true,
            
            // Standalone expressions can stay at module level (e.g., constants, simple values)
            // These are typically used in tests or simple modules
            Expr(expr) => Self::is_simple_expression(&expr.value),
            
            // These are executable statements that must go in the init function
            Assign(_) | AugAssign(_) | Call(_) | Return(_) |
            If(_) | For(_) | While(_) | Try(_) | With(_) | AsyncWith(_) | AsyncFor(_) |
            Raise(_) | Assert { .. } | Pass | Break | Continue | Delete(_) => false,
            
            // Handle unimplemented statements conservatively as executable
            Unimplemented(_) => false,
        }
    }
    
    /// Check if an expression is simple enough to remain at module level
    fn is_simple_expression(expr: &crate::ExprType) -> bool {
        use crate::ExprType::*;
        match expr {
            // Simple constants and literals can stay at module level
            Constant(_) | Name(_) | NoneType(_) => true,
            
            // Allow unary operations for single-expression modules (test compatibility)
            UnaryOp(_) => true,
            
            // Function calls and complex expressions should go in init
            Call(_) | BinOp(_) | Compare(_) | BoolOp(_) | 
            IfExp(_) | Dict(_) | Set(_) | List(_) | Tuple(_) | ListComp(_) |
            Lambda(_) | Attribute(_) | Subscript(_) | Starred(_) |
            DictComp(_) | SetComp(_) | GeneratorExp(_) | Await(_) | 
            Yield(_) | YieldFrom(_) | FormattedValue(_) | JoinedStr(_) |
            NamedExpr(_) => false,
            
            // Be conservative about other expression types
            Unimplemented(_) | Unknown => false,
        }
    }
    
    /// Rename the main function definition and update all references to it throughout the code
    fn rename_main_function_and_references(code: &str) -> String {
        // First, rename the function definitions
        let code = code
            .replace("pub async fn main (", "pub async fn python_main (")
            .replace("pub fn main (", "pub fn python_main (");
        
        // Then update all references using the comprehensive reference updater
        Self::update_main_references(&code)
    }
    
    /// Convert a Python main function to be suitable as a Rust entry point
    /// This handles return value conversion and signature requirements
    fn convert_python_main_to_rust_entry_point(code: &str) -> String {
        use regex::Regex;
        
        // Replace "pub fn main (" with "fn main("
        let code = code.replace("pub fn main (", "fn main(");
        
        // Handle return statements in the main function
        // We need to wrap the function body to ignore return values
        let main_fn_pattern = Regex::new(r"fn main\(\s*\)\s*\{([^}]*)\}").unwrap();
        
        if let Some(captures) = main_fn_pattern.captures(&code) {
            let body = captures.get(1).map_or("", |m| m.as_str());
            
            // Check if the body contains return statements
            if body.contains("return ") {
                // Wrap the original function as python_main and create new main that ignores return
                let new_code = code.replace("fn main(", "fn python_main(");
                format!("{}\n\nfn main() {{\n    let _ = python_main();\n}}", new_code)
            } else {
                // No return statements, use the function as-is
                code
            }
        } else {
            // Couldn't parse the function, fall back to original
            code
        }
    }
    
    /// Update all references to main() function calls with python_main() calls
    /// This uses regex to handle various call patterns with parameters
    fn update_main_references(code: &str) -> String {
        use regex::Regex;
        
        // Pattern 1: main(...) - function calls with any arguments (including empty)
        // This pattern matches "main(" and lets us replace the function name
        let call_pattern = Regex::new(r"\bmain\s*\(").unwrap();
        let mut result = call_pattern.replace_all(code, "python_main(").to_string();
        
        // Pattern 2: Handle method calls like obj.call_main() -> obj.call_python_main()
        let method_pattern = Regex::new(r"\.call_main\s*\(").unwrap();
        result = method_pattern.replace_all(&result, ".call_python_main(").to_string();
        
        // Pattern 3: Handle assignment patterns like "result = main" (without parentheses)
        // We need to be careful not to match function definitions or other contexts
        let assignment_pattern = Regex::new(r"=\s+main\b").unwrap();
        result = assignment_pattern.replace_all(&result, "= python_main").to_string();
        
        // Pattern 4: Handle return statements like "return main"
        let return_pattern = Regex::new(r"return\s+main\b").unwrap();
        result = return_pattern.replace_all(&result, "return python_main").to_string();
        
        result
    }
    
    fn get_module_docstring(&self) -> Option<String> {
        if self.raw.body.is_empty() {
            return None;
        }
        
        // Check if the first statement is a string constant (docstring)
        let first_stmt = &self.raw.body[0];
        match &first_stmt.statement {
            StatementType::Expr(expr) => match &expr.value {
                ExprType::Constant(c) => {
                    // The Ellipsis sentinel is not a docstring (Protocol
                    // stubs and `...` placeholders must not emit a bogus
                    // #![doc]).
                    if c.0
                        .as_ref()
                        .is_some_and(crate::ast::tree::constant::is_ellipsis_literal)
                    {
                        return None;
                    }
                    let raw_string = c.to_string();
                    Some(self.format_module_docstring(&raw_string))
                },
                _ => None,
            },
            _ => None,
        }
    }
    
    fn format_module_docstring(&self, raw: &str) -> String {
        // Remove surrounding quotes
        let content = raw.trim_matches('"');
        
        // Split into lines and clean up Python-style indentation
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return String::new();
        }
        
        // For module docstrings, preserve more of the original formatting
        let mut formatted = Vec::new();
        
        for line in lines {
            let cleaned = line.trim();
            if !cleaned.is_empty() {
                formatted.push(cleaned.to_string());
            } else {
                formatted.push(String::new());
            }
        }
        
        formatted.join("\n")
    }
    
    fn looks_like_module_docstring(&self) -> bool {
        if self.raw.body.is_empty() {
            return false;
        }
        
        // Check if the first statement looks like a module docstring
        let first_stmt = &self.raw.body[0];
        if let StatementType::Expr(expr) = &first_stmt.statement {
            if let ExprType::Constant(c) = &expr.value {
                let raw_string = c.to_string();
                let content = raw_string.trim_matches('"');
                
                // Heuristics to detect if this is a module docstring vs just a string expression:
                // 1. Contains multiple lines
                // 2. Contains common docstring keywords
                // 3. Looks like documentation rather than a simple string
                return content.lines().count() > 1 
                    || content.to_lowercase().contains("module")
                    || content.to_lowercase().contains("this ")
                    || content.len() > 50; // Longer strings are more likely to be docstrings
            }
        }
        false
    }
}

impl Object for Module {
    /// __dir__ is called to list the attributes of the object.
    fn __dir__(&self) -> Vec<impl AsRef<str>> {
        // XXX - Make this meaningful.
        vec![
            "__class__",
            "__class_getitem__",
            "__contains__",
            "__delattr__",
            "__delitem__",
            "__dir__",
            "__doc__",
            "__eq__",
            "__format__",
            "__ge__",
            "__getattribute__",
            "__getitem__",
            "__getstate__",
            "__gt__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__ior__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__ne__",
            "__new__",
            "__or__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__reversed__",
            "__ror__",
            "__setattr__",
            "__setitem__",
            "__sizeof__",
            "__str__",
            "__subclasshook__",
            "clear",
            "copy",
            "fromkeys",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_we_print() {
        let options = PythonOptions::default();
        let result = crate::parse(
            "#test comment
def foo():
    print(\"Test print.\")
",
            "test_case.py",
        )
        .unwrap();
        info!("Python tree: {:?}", result);
        //info!("{}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        info!("module: {:?}", code);
    }

    #[test]
    fn can_we_import() {
        let result = crate::parse("import ast", "ast.py").unwrap();
        let options = PythonOptions::default();
        info!("{:?}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        info!("module: {:?}", code);
    }

    #[test]
    fn can_we_import2() {
        let result = crate::parse("import ast as test", "ast.py").unwrap();
        let options = PythonOptions::default();
        info!("{:?}", result);

        let code = result.to_rust(
            CodeGenContext::Module("test_case".to_string()),
            options,
            SymbolTableScopes::new(),
        );
        info!("module: {:?}", code);
    }
}

/// The classes a module EMITS: `collect_class_defs` over the body after
/// the same folds the emission applies — the failed-import try fold, the
/// version gates, and the static-name gates (`if brotli is not None:` —
/// urllib3's response.py, whose BrotliDecoder never exists in the crate).
/// The one authority for "is this class in the crate", so a crate-wide
/// index (the hierarchy's sum-type variants) agrees with the emission.
/// `options` must be the module's own scope (module_path /
/// this_module_path), as the emission's are.
pub(crate) fn emitted_class_defs(
    module: &crate::Module,
    options: &PythonOptions,
) -> Vec<crate::ClassDef> {
    let mut out = Vec::new();
    // No class under a gate: the plain walk is exact, and the per-module
    // symbol table (the gate names need it) is never built.
    if !has_gated_class(&module.raw.body) {
        top_level_class_defs(&module.raw.body, &mut out);
        return out;
    }
    let (body, _) = fold_static_import_trys(&module.raw.body, options);
    let body = splice_gated_branches(body, options);
    let mut counts = std::collections::HashMap::new();
    count_module_stores(&body, &mut counts);
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let global_mutables = module_global_mutable_names(&body, &counts, &symbols, options);
    let (none_names, false_names, module_names) =
        static_gate_names(&body, &counts, &global_mutables, options);
    let mut gated = options.clone();
    gated.statically_none_names = std::rc::Rc::new(none_names);
    gated.statically_false_names = std::rc::Rc::new(false_names);
    gated.statically_module_names = std::rc::Rc::new(module_names);
    let body = splice_gated_branches(body, &gated);
    top_level_class_defs(&body, &mut out);
    out
}

/// The class statements that are module ITEMS: a class under a gate the
/// emission cannot fold lowers inside the module-init block, where no
/// other item can name it — it is not a crate-visible class.
fn top_level_class_defs(stmts: &[crate::Statement], out: &mut Vec<crate::ClassDef>) {
    for s in stmts {
        if let crate::StatementType::ClassDef(c) = &s.statement {
            out.push(c.clone());
        }
    }
}

/// Whether any class statement sits under a module-level `if` or `try`
/// (the shapes whose branch the emission may fold away).
fn has_gated_class(stmts: &[crate::Statement]) -> bool {
    stmts.iter().any(|s| match &s.statement {
        crate::StatementType::If(i) => {
            has_class(&i.body) || has_class(&i.orelse)
        }
        crate::StatementType::Try(t) => {
            has_class(&t.body)
                || t.handlers.iter().any(|h| has_class(&h.body))
                || has_class(&t.orelse)
                || has_class(&t.finalbody)
        }
        _ => false,
    })
}

fn has_class(stmts: &[crate::Statement]) -> bool {
    let mut out = Vec::new();
    collect_class_defs(stmts, &mut out);
    !out.is_empty()
}

/// Every `ClassDef` in the module, recursing into container statements
/// (if/for/while/with/try/async/function bodies), so classes defined under
/// an `if __name__ == "__main__":` guard take part in the same hierarchy
/// and trait-signature precomputes as top-level classes — their class
/// statements lower through the same machinery.
pub(crate) fn collect_class_defs(stmts: &[crate::Statement], out: &mut Vec<crate::ClassDef>) {
    for s in stmts {
        match &s.statement {
            crate::StatementType::ClassDef(c) => out.push(c.clone()),
            crate::StatementType::If(i) => {
                collect_class_defs(&i.body, out);
                collect_class_defs(&i.orelse, out);
            }
            crate::StatementType::For(f) => {
                collect_class_defs(&f.body, out);
                collect_class_defs(&f.orelse, out);
            }
            crate::StatementType::While(w) => {
                collect_class_defs(&w.body, out);
                collect_class_defs(&w.orelse, out);
            }
            crate::StatementType::With(w) => collect_class_defs(&w.body, out),
            crate::StatementType::AsyncWith(w) => collect_class_defs(&w.body, out),
            crate::StatementType::AsyncFor(f) => {
                collect_class_defs(&f.body, out);
                collect_class_defs(&f.orelse, out);
            }
            crate::StatementType::Try(t) => {
                collect_class_defs(&t.body, out);
                for h in &t.handlers {
                    collect_class_defs(&h.body, out);
                }
                collect_class_defs(&t.orelse, out);
                collect_class_defs(&t.finalbody, out);
            }
            crate::StatementType::FunctionDef(f) | crate::StatementType::AsyncFunctionDef(f) => {
                collect_class_defs(&f.body, out);
            }
            _ => {}
        }
    }
}

/// Is `name` — a class of the CURRENT module (options.this_module_path) —
/// used as a base by a class in another module of the crate? urllib3's
/// RequestMethods: subclassed only cross-module (poolmanager,
/// connectionpool), so its own module's hierarchy computation never sees
/// it, yet the subclass modules' ancestor impls and supertrait bounds
/// name `RequestMethodsTrait` (issue #137 round 20).
///
/// The importer match is deliberately loose — the importing module binds
/// `name` through a from-import whose dotted module's LAST segment equals
/// this module's — because resolving each sibling's relative imports here
/// would repeat the whole chain machinery; a false positive only emits an
/// unused accessor-only trait (dead code, never an error), while a miss
/// leaves an unresolved `{Name}Trait` (E0405).
pub(crate) fn class_subclassed_crate_wide(name: &str, options: &crate::PythonOptions) -> bool {
    let Some(this_leaf) = options.this_module_path.last() else {
        return false;
    };
    fn imports_name_from(stmts: &[crate::Statement], name: &str, leaf: &str) -> bool {
        use crate::StatementType as ST;
        stmts.iter().any(|s| match &s.statement {
            ST::ImportFrom(i) => {
                i.module.rsplit('.').next() == Some(leaf)
                    && i.names
                        .iter()
                        .any(|a| a.asname.as_deref().unwrap_or(&a.name) == name)
            }
            ST::If(b) => imports_name_from(&b.body, name, leaf) || imports_name_from(&b.orelse, name, leaf),
            ST::Try(t) => {
                imports_name_from(&t.body, name, leaf)
                    || t.handlers.iter().any(|h| imports_name_from(&h.body, name, leaf))
            }
            _ => false,
        })
    }
    for (path, module) in options.module_defs.iter() {
        if path[..] == options.this_module_path[..] {
            continue;
        }
        let mut classes = Vec::new();
        collect_class_defs(&module.raw.body, &mut classes);
        let subclasses_name = classes.iter().any(|c| {
            c.bases
                .iter()
                .any(|b| matches!(b, crate::ExprType::Name(n) if n.id == name))
        });
        if subclasses_name && imports_name_from(&module.raw.body, name, this_leaf) {
            return true;
        }
    }
    false
}

/// For each class in `ast` that lowers with the trait machinery, the trait
/// names that carry its methods: its own `{Name}Trait` plus one per
/// ancestor, nearest first. Consumed by the converter so a relative import
/// of the class from another module of the generated crate can bring the
/// traits into scope at the call site (Rust method resolution needs the
/// trait in scope; the class's own module defines it). Mirrors the
/// hierarchy-class computation in `Module::to_rust`.
///
/// Backed by the once-per-conversion [`CrossModuleClasses`] cache
/// (`options.cross_module_classes`), so the module AST is not deep-cloned
/// and re-scanned on every import statement.
pub fn module_class_traits(
    options: &PythonOptions,
    path: &[String],
) -> std::collections::HashMap<String, Vec<String>> {
    module_class_info(options, path)
        .map(|info| info.traits.clone())
        .unwrap_or_default()
}

/// The `ClassDef` named `name` in the module at `path`, plus that module's
/// own symbol table (so the class's base chain resolves inside the module
/// that declared it, not the importer's scope). Used for cross-module class
/// construction: `from .animals import Dog; Dog("Rex")` lowers against
/// Dog's real `__init__` signature, and an inherited `__init__` resolves
/// through the defining module's chain.
///
/// Backed by the once-per-conversion [`CrossModuleClasses`] cache
/// (`options.cross_module_classes`): `receiver_class` consults this for
/// EVERY attribute access on an imported class, so the defining module's
/// symbol table must not be rebuilt per access.
pub fn module_class_def(
    options: &PythonOptions,
    path: &[String],
    name: &str,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    let info = module_class_info(options, path)?;
    info.classes.get(name).cloned().map(|c| (c, info.symbols.clone()))
}

/// Resolve a class name through a module, following RE-EXPORT chains
/// (`from urllib3.util import Timeout` where util/__init__.py does
/// `from .timeout import Timeout`): the class may live in the module the
/// import re-exports from, several levels deep.
pub fn resolve_imported_class(
    options: &PythonOptions,
    path: &[String],
    name: &str,
    depth: usize,
) -> Option<(crate::ClassDef, SymbolTableScopes)> {
    resolve_imported_class_with_path(options, path, name, depth).map(|(c, s, _)| (c, s))
}

/// [`resolve_imported_class`], also returning the DEFINING module's path
/// (the module the chain terminated in) — the scope its relative imports
/// and annotations resolve in.
pub fn resolve_imported_class_with_path(
    options: &PythonOptions,
    path: &[String],
    name: &str,
    depth: usize,
) -> Option<(crate::ClassDef, SymbolTableScopes, Vec<String>)> {
    if depth > 16 {
        return None;
    }
    if let Some((c, s)) = module_class_def(options, path, name) {
        return Some((c, s, path.to_vec()));
    }
    let module = options.module_defs.get(path)?;
    let module: &crate::Module = module;
    let syms = module.clone().find_symbols(SymbolTableScopes::new());
    match syms.get(name) {
        Some(crate::SymbolTableNode::ImportFrom(i)) => {
            let defining = i
                .names
                .iter()
                .find(|a| a.asname.as_deref() == Some(name))
                .map(|a| a.name.clone())
                .unwrap_or_else(|| name.to_string());
            // Resolve the relative import in the DEFINING module's
            // context (`from .timeout import Timeout` in util/__init__.py
            // is relative to ["urllib3", "util"], not the caller).
            let mut ctx = options.clone();
            // The relative import resolves in the DEFINING module's PACKAGE
            // context: module_class_def's path includes the module name
            // (["urllib3", "connection"]), but resolved_module_path expects
            // the package path (["urllib3"]). An __init__ module IS its own
            // package: its package path is the full module path
            // (["urllib3", "util"] for urllib3/util/__init__.py — the
            // re-export chain `from urllib3.util import Timeout` follows
            // through it). Detect by a longer module key under the path.
            let is_package = options
                .module_defs
                .keys()
                .any(|k| k.len() > path.len() && k[..path.len()] == path[..]);
            ctx.module_path = if is_package {
                path.to_vec()
            } else {
                path[..path.len().saturating_sub(1)].to_vec()
            };
            let path2 = i.resolved_module_path(&ctx);
            resolve_imported_class_with_path(options, &path2, &defining, depth + 1)
        }
        // A RE-EXPORT alias (`from ._base_connection import ProxyConfig
        // as ProxyConfig` in connection.py — urllib3): the canonical name
        // resolves in the same module; a self-alias would recurse forever,
        // so stop there.
        Some(crate::SymbolTableNode::Alias(canonical)) if canonical != name => {
            resolve_imported_class_with_path(options, path, canonical, depth + 1)
        }
        _ => None,
    }
}

/// Resolve a class REFERENCED BY NAME in the current scope to its
/// ClassDef, following imports into sibling modules (`from
/// urllib3.util.retry import Retry` — requests' adapters.py) and re-export
/// chains. Unlike [`resolve_imported_class`] this starts from the CALLER's
/// symbol table: the name may be a local ClassDef or an imported one.
/// Used to render class-body constants through the LOCAL name (the
/// import's `use` brings it into scope), so `Retry::DEFAULT_ALLOWED_METHODS`
/// works from a caller that only imported Retry.
pub fn resolve_class_referenced(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> Option<crate::ClassDef> {
    match symbols.get(name) {
        Some(crate::SymbolTableNode::ClassDef(c)) => Some(c.clone()),
        Some(crate::SymbolTableNode::Alias(canonical)) if canonical != name => {
            resolve_class_referenced(canonical, symbols, options)
        }
        Some(crate::SymbolTableNode::ImportFrom(i)) => {
            let path = i.resolved_module_path(options);
            if options.module_defs.contains_key(&path) {
                let canonical = i
                    .names
                    .iter()
                    .find(|a| a.asname.as_deref() == Some(name))
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| name.to_string());
                resolve_imported_class(options, &path, &canonical, 0)
                    .map(|(c, _)| c)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The `module_defs` key for a resolved import path, matching both keying
/// conventions: keys are RELATIVE to the package root for src-layout /
/// package-dir sdists (boto3, pip: `session`, `_internal.cli.req_command`),
/// but include the package root when the sdist root IS the package
/// (requests' vendored deps: `urllib3.contrib`). An absolute import
/// (`from boto3.session import ...`) resolves to the root-qualified path,
/// so try it first, then with the leading (package-root) segment dropped.
/// Returns the matching key (borrowing the caller's `path`) so callers can
/// index `module_defs` with it, or `None` when neither form is a crate
/// module.
pub fn module_defs_key<'a>(
    options: &'a PythonOptions,
    path: &'a [String],
) -> Option<&'a [String]> {
    if options.module_defs.contains_key(path) {
        Some(path)
    } else if path.len() > 1
        // The stripped form covers ONLY the package's own root-qualified
        // name (`pip._internal...` inside the pip conversion): stripping
        // an arbitrary external root would alias foreign packages onto
        // same-named crate modules (`import h2.connection` resolving to
        // urllib3's connection.py — issue #137 round 18's merge).
        && path[0] == options.python_namespace
        && options.module_defs.contains_key(&path[1..])
    {
        Some(&path[1..])
    } else {
        None
    }
}

/// Whether `path` names a module of the converted crate — see
/// [`module_defs_key`].
pub fn module_defs_contains(options: &PythonOptions, path: &[String]) -> bool {
    module_defs_key(options, path).is_some()
}

/// Resolve a FUNCTION defined at the top level of another module of the
/// crate, with that module's symbol table (issue #123): `from
/// pip._internal.locations import get_scheme` + `scheme = get_scheme(...)`
/// needs `get_scheme`'s `-> Scheme` return annotation to type
/// `scheme.scripts`, and keyword arguments on an imported function require
/// its signature. Returns None when the module or the function is not
/// found (an import of a builtin/stdlib name, a vendored non-module, ...).
pub fn module_function_def(
    options: &PythonOptions,
    path: &[String],
    name: &str,
) -> Option<(crate::FunctionDef, SymbolTableScopes)> {
    let module = options.module_defs.get(path)?;
    let module: &crate::Module = module;
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    // A name may have MULTIPLE definitions: `@typing.overload` stubs (with
    // `...` default placeholders) followed by the real implementation
    // (urllib3's `ssl_wrap_socket`). Skip the overload stubs and return the
    // first NON-overload definition — the callable the call sites actually
    // invoke. (find_symbols keeps the LAST definition for same-module
    // call sites; this loop keeps the first non-stub for cross-module ones.)
    for s in &module.raw.body {
        if let crate::StatementType::FunctionDef(f) = &s.statement {
            if f.name == name {
                let is_overload_stub = f.decorator_list.iter().any(|d| {
                    match d {
                        crate::ExprType::Name(n) => n.id == "overload",
                        crate::ExprType::Attribute(a) => a.attr == "overload",
                        _ => false,
                    }
                });
                if !is_overload_stub {
                    return Some((f.clone(), symbols));
                }
            }
        }
    }
    // Only overload stubs exist (a stub-only module, e.g. a vendored
    // typing stubs file): fall back to the first definition so signature
    // resolution still finds SOMETHING.
    for s in &module.raw.body {
        if let crate::StatementType::FunctionDef(f) = &s.statement {
            if f.name == name {
                return Some((f.clone(), symbols));
            }
        }
    }
    None
}

/// The cached class facts for the module at `path`, building the
/// once-per-conversion table over every module of the crate on first use.
fn module_class_info(
    options: &PythonOptions,
    path: &[String],
) -> Option<std::rc::Rc<ModuleClassInfo>> {
    {
        let state = options.cross_module_classes.borrow();
        if let CrossModuleClasses::Computed(table) = &*state {
            return table.get(path).cloned();
        }
    }
    // First use: build the table for every module in one pass (the build
    // consults only the module's own AST and symbols, so it cannot re-enter
    // this cache).
    let mut table = std::collections::HashMap::new();
    for (module_path, module) in options.module_defs.iter() {
        table.insert(
            module_path.clone(),
            std::rc::Rc::new(module_class_info_for(module)),
        );
    }
    *options.cross_module_classes.borrow_mut() =
        CrossModuleClasses::Computed(std::rc::Rc::new(table));
    options
        .cross_module_classes
        .borrow()
        .computed_class_info(path)
        .cloned()
}

impl CrossModuleClasses {
    /// The class facts for one module path, when computed.
    fn computed_class_info(
        &self,
        path: &[String],
    ) -> Option<&std::rc::Rc<ModuleClassInfo>> {
        match self {
            CrossModuleClasses::Computed(table) => table.get(path),
            _ => None,
        }
    }
}

/// Build the class facts for one module: its symbol table, top-level
/// classes by name, and hierarchy-class → trait names.
fn module_class_info_for(module: &crate::Module) -> ModuleClassInfo {
    let symbols = module.clone().find_symbols(SymbolTableScopes::new());
    let mut classes = std::collections::HashMap::new();
    // Classes come from the SYMBOL TABLE, not the raw body: find_symbols
    // runs the dataclass/NamedTuple __init__ synthesis on its ClassDef
    // clones, so a cross-module call site (`Url(...)` in connection.py)
    // resolves the constructor against the synthesized class.
    for c in symbols.all_classes() {
        classes.insert(c.name.clone(), c);
    }
    let mut traits = std::collections::HashMap::new();
    let mut class_list = Vec::new();
    collect_class_defs(&module.raw.body, &mut class_list);
    let mut hierarchy = std::collections::HashSet::new();
    for c in &class_list {
        let has_real_base = c
            .bases
            .iter()
            .any(|b| matches!(b, crate::ExprType::Name(n) if n.id != "object"));
        if has_real_base {
            hierarchy.insert(c.name.clone());
        }
        for b in &c.bases {
            if let crate::ExprType::Name(n) = b
                && n.id != "object"
            {
                hierarchy.insert(n.id.clone());
            }
        }
    }
    for c in class_list {
        if !hierarchy.contains(&c.name) {
            continue;
        }
        // An EXCEPTION class (or a Protocol) lowers as a marker struct
        // with no trait machinery (class_def.rs's early returns), so a
        // bring-along `use crate::exceptions::HTTPErrorTrait;` at the
        // import site would be E0432 — no entry.
        if crate::ast::tree::class_def::is_exception_class(&c)
            || c.bases.iter().any(|b| match b {
                crate::ExprType::Name(n) => n.id == "Protocol",
                crate::ExprType::Subscript(s) => {
                    matches!(s.value.as_ref(), crate::ExprType::Name(n) if n.id == "Protocol")
                }
                _ => false,
            })
        {
            continue;
        }
        let t: Vec<String> = c
            .base_chain(&symbols)
            .iter()
            .map(|cc| format!("{}Trait", cc.name))
            .collect();
        traits.insert(c.name.clone(), t);
    }
    ModuleClassInfo {
        symbols,
        classes,
        traits,
    }
}

/// Whether the trait of the hierarchy rooted at `root_name` widens `method`
/// to `&mut self` in ANY module of the crate.
///
/// `trait_mut_self` is computed per module in `Module::to_rust`, so a class
/// imported from another module has no entry in the importing module's
/// table — yet its trait was widened (and emitted) in the DEFINING module,
/// so call sites in the importing module must borrow the receiver mutably
/// to match. This consults the once-per-conversion merged table
/// ([`cross_module_mut_self_table`], cached in
/// `options.cross_module_mut_self`) instead of re-deriving the whole-crate
/// analysis for every call site.
///
/// The scan is only reached when the current module's own precompute has no
/// entry for `root_name` — i.e. the class is not a hierarchy class of the
/// current module — so it is not on the hot path once the table is cached.
/// A same-named class in another module may produce a spurious `true` (an
/// extra `let mut`), which is always accepted by rustc; a spurious `false`
/// (missing `mut`) is the bug this prevents, and cannot occur when the
/// defining module is in `module_defs`, as it is for every module rypip
/// transpiles.
pub fn module_widens_method_cached(
    options: &PythonOptions,
    root_name: &str,
    method: &str,
) -> bool {
    {
        let state = options.cross_module_mut_self.borrow();
        match &*state {
            CrossModuleMutSelf::Computed(table) => {
                return table.get(root_name).is_some_and(|s| s.contains(method));
            }
            CrossModuleMutSelf::Computing => {
                // Re-entrant fallback while the one-time scan is building
                // the table (the scan's own per-method analysis consults
                // method_needs_mut_self again): return false and let the
                // direct chain walk in method_needs_mut_self answer —
                // recomputing here would recurse.
                return false;
            }
            CrossModuleMutSelf::Uncomputed => {}
        }
    }
    *options.cross_module_mut_self.borrow_mut() = CrossModuleMutSelf::Computing;
    let table = cross_module_mut_self_table(options);
    *options.cross_module_mut_self.borrow_mut() =
        CrossModuleMutSelf::Computed(std::rc::Rc::new(table));
    options
        .cross_module_mut_self
        .borrow()
        .computed()
        .is_some_and(|t| t.get(root_name).is_some_and(|s| s.contains(method)))
}

/// Build the merged trait-mut table over every module of the crate: root
/// class name → method names whose trait signature must be `&mut self`
/// (because some definition in the hierarchy mutates self). Mirrors the
/// per-module precompute in `Module::to_rust` (same root = topmost definer
/// keying, same `own_method_mutates` test), applied to the shared
/// cross-module ASTs (`options.module_defs`) so a sibling module's mutation
/// widens the defining module's trait. Re-entrant `method_needs_mut_self`
/// fallbacks during the scan see the `Computing` sentinel and answer via
/// the direct chain walk, so the build does not recurse into itself.
pub fn cross_module_mut_self_table(
    options: &PythonOptions,
) -> std::collections::HashMap<String, std::collections::HashSet<String>> {
    let mut table = std::collections::HashMap::<
        String,
        std::collections::HashSet<String>,
    >::new();
    for module in options.module_defs.values() {
        let symbols = (**module).clone().find_symbols(SymbolTableScopes::new());
        let mut classes = Vec::new();
        collect_class_defs(&module.raw.body, &mut classes);
        for c in &classes {
            let chain = c.base_chain(&symbols);
            for m in c.methods() {
                if m.name == "__init__" {
                    continue;
                }
                if c.own_method_mutates(&m.name, &symbols, options) {
                    // The root = the TOPMOST class in the chain that
                    // defines the method (the trait owner) — the same key
                    // Module::to_rust uses.
                    if let Some(root) = chain
                        .iter()
                        .rev()
                        .find(|cc| cc.methods().any(|mm| mm.name == m.name))
                    {
                        table
                            .entry(root.name.clone())
                            .or_default()
                            .insert(m.name.clone());
                    }
                }
            }
        }
    }
    table
}


/// Whether the module at `path` re-exports `name` from a STDPYTHON module
/// (`from .compat import json as complexjson` where compat.py does
/// `import json`): the generated module has no item of that name (stdlib
/// modules resolve through the runtime glob), so the importer must route
/// to the runtime module (`use <stdpython>::json as complexjson;`).
/// Returns the stdpython module name (None when the re-export is not a
/// stdpython module). Nested imports (try/except bodies) are followed.
pub(crate) fn module_reexports_stdpython_module(
    options: &crate::PythonOptions,
    path: &[String],
    name: &str,
) -> Option<String> {
    let module = options.module_defs.get(path)?;
    let module: &crate::Module = module;
    fn walk(body: &[crate::Statement], name: &str) -> Option<String> {
        use crate::StatementType as ST;
        for s in body {
            let found: Option<String> = match &s.statement {
                ST::Import(im) => im
                    .names
                    .iter()
                    .find(|a| {
                        a.asname.as_deref() == Some(name)
                            || (a.asname.is_none() && a.name.split('.').next() == Some(name))
                    })
                    .map(|a| a.name.split('.').next().unwrap_or("").to_string())
                    .filter(|m| crate::is_stdpython_module(m)),
                ST::If(i) => walk(&i.body, name).or_else(|| walk(&i.orelse, name)),
                ST::Try(t) => {
                    for part in [&t.body, &t.orelse, &t.finalbody] {
                        if let Some(m) = walk(part, name) {
                            return Some(m);
                        }
                    }
                    for h in &t.handlers {
                        if let Some(m) = walk(&h.body, name) {
                            return Some(m);
                        }
                    }
                    None
                }
                ST::While(w) => walk(&w.body, name).or_else(|| walk(&w.orelse, name)),
                ST::For(f) => walk(&f.body, name).or_else(|| walk(&f.orelse, name)),
                ST::With(w) => walk(&w.body, name),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }
    walk(&module.raw.body, name)
}
