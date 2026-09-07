use proc_macro2::TokenStream;
use pyo3::{Borrowed, FromPyObject, PyAny, PyResult, types::PyAnyMethods};
use quote::quote;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{CodeGen, CodeGenContext, PythonOptions, SymbolTableNode, SymbolTableScopes};

/// Python stdlib modules that the stdpython runtime crate provides. Imports
/// of these resolve under the runtime crate; anything else is assumed to be
/// a sibling module of the generated crate. The name set lives in the
/// [`crate::StdModule`] enum (the boundary-parse rule); this is the
/// convenience wrapper the many call sites keep using.
pub(crate) fn is_stdpython_module(name: &str) -> bool {
    crate::StdModule::from_name(name).is_some()
}

/// Whether a from-importable item of the stdpython module `module` is a
/// CLASS (a runtime struct constructed with `Name::new(...)`) rather than
/// a module function or constant. The call lowering needs the
/// distinction: `urlparse(url)` is a plain call, while `OrderedDict(...)`
/// is a construction — treating functions as classes produced
/// `urlparse::new(...)` (E0433: a function used as a module path) for
/// every requests/urllib3 call site (round 55).
pub(crate) fn stdpython_module_class(module: &str, name: &str) -> bool {
    use crate::StdModule;
    let Some(module) = StdModule::from_name(module) else {
        return false;
    };
    match module {
        // StringIO/BytesIO are FUNCTIONS in the runtime (`pub fn
        // StringIO()`), not structs — the io dispatch handles them.
        StdModule::Io => false,
        StdModule::Threading => crate::ThreadingType::from_name(name).is_some(),
        StdModule::Socket => matches!(name, "socket"),
        StdModule::Ssl => matches!(name, "SSLContext" | "SSLSocket"),
        // urllib: urlopen and the parse submodule's items are ALL
        // functions (urlparse/urlsplit/urljoin/urlencode/quote/...).
        StdModule::Urllib => false,
        StdModule::Collections => matches!(name, "OrderedDict" | "defaultdict" | "deque"),
        StdModule::Re => false,
        StdModule::Itertools => false,
        StdModule::Functools => false,
        StdModule::Hashlib => false,
        StdModule::Json => false,
        // The datetime module's classes — one typed set
        // (DatetimeType::from_name), shared by the registry and the
        // constructor lowering.
        StdModule::Datetime => crate::DatetimeType::from_name(name).is_some(),
        StdModule::Os => false,
        StdModule::Pathlib => matches!(name, "PurePath" | "Path"),
        StdModule::Tempfile => {
            matches!(name, "NamedTemporaryFile" | "TemporaryDirectory" | "SpooledTemporaryFile")
        }
        StdModule::Subprocess => matches!(name, "CompletedProcess"),
        StdModule::Csv => false,
        StdModule::String => matches!(name, "Template"),
        StdModule::Venv => matches!(name, "EnvBuilder"),
        // Functions/constants only.
        StdModule::Sys
        | StdModule::Time
        | StdModule::Math
        | StdModule::Random
        | StdModule::Warnings
        | StdModule::Textwrap
        | StdModule::Heapq
        | StdModule::Copy
        | StdModule::Glob
        | StdModule::Sysconfig
        | StdModule::Argparse => false,
        StdModule::Numpy | StdModule::Asyncio => false,
    }
}

/// Whether a TYPE-CHECKING import of `name` from the stdpython module
/// `module` can emit a `use`: the item must have a KNOWN runtime
/// counterpart in stdpython's module (an `if TYPE_CHECKING:` import of
/// `io.BufferedWriter` — requests' utils.py — is only an annotation; the
/// runtime `io` has no BufferedWriter, so the use would fail E0432).
pub(crate) fn stdpython_module_item(module: &str, name: &str) -> bool {
    use crate::StdModule;
    let Some(module) = StdModule::from_name(module) else {
        return false;
    };
    match module {
        StdModule::Io => matches!(name, "StringIO" | "BytesIO" | "DEFAULT_BUFFER_SIZE"),
        // The type names come from the ThreadingType enum (one source of
        // truth); current_thread/active_count are module functions.
        StdModule::Threading => {
            crate::ThreadingType::from_name(name).is_some()
                || matches!(name, "current_thread" | "active_count")
        }
        StdModule::Socket => matches!(
            name,
            "socket"
                | "gethostname"
                | "getdefaulttimeout"
                | "setdefaulttimeout"
                | "AF_UNSPEC"
                | "AF_INET"
                | "AF_INET6"
                | "SOCK_STREAM"
                | "SOCK_DGRAM"
        ),
        // ssl: the rustls-backed surface — context/socket types plus the
        // CPython module constants the runtime module actually defines.
        // SSLError is a string-tagged exception (matched by name, no
        // runtime item), so it is NOT here: its from-import drops with
        // the annotation-only warning while except-matching still works.
        StdModule::Ssl => matches!(
            name,
            "SSLContext"
                | "SSLSocket"
                | "create_default_context"
                | "TLSVersion"
                | "HAS_SNI"
                | "HAS_NEVER_CHECK_COMMON_NAME"
                | "OPENSSL_VERSION"
                | "OPENSSL_VERSION_NUMBER"
                | "OPENSSL_VERSION_INFO"
                | "CERT_NONE"
                | "CERT_OPTIONAL"
                | "CERT_REQUIRED"
                | "PROTOCOL_TLS"
                | "PROTOCOL_SSLv23"
                | "PROTOCOL_TLS_CLIENT"
                | "PROTOCOL_TLSv1"
                | "PROTOCOL_TLSv1_1"
                | "PROTOCOL_TLSv1_2"
                | "OP_NO_SSLv2"
                | "OP_NO_SSLv3"
                | "OP_NO_TLSv1"
                | "OP_NO_TLSv1_1"
                | "OP_NO_TLSv1_2"
                | "OP_NO_TLSv1_3"
                | "OP_NO_COMPRESSION"
                | "OP_NO_TICKET"
                | "OP_NO_RENEGOTIATION"
                | "VERIFY_X509_STRICT"
                | "VERIFY_X509_TRUSTED_FIRST"
                | "VERIFY_X509_PARTIAL_CHAIN"
        ),
        // urllib: the request submodule and its items. urllib.error's
        // URLError/HTTPError are string-tagged exceptions matched by name
        // (no runtime item), so `from urllib.error import URLError` drops
        // with the annotation-only warning — except matching still works.
        StdModule::Urllib => matches!(
            name,
            // The request submodule and its items.
            "request"
                | "urlopen"
                // The parse submodule (round 55): the functions requests'
                // compat.py imports — urlparse/urlsplit/urlunparse/
                // urljoin/urlencode/quote/unquote/urldefrag. The dotted
                // path resolves through the same flattened registry.
                | "parse"
                | "urlparse"
                | "urlsplit"
                | "urlunparse"
                | "urljoin"
                | "urlencode"
                | "quote"
                | "quote_plus"
                | "unquote"
                | "unquote_plus"
                | "urldefrag",
        ),
        StdModule::Collections => {
            matches!(name, "OrderedDict" | "defaultdict" | "deque" | "namedtuple")
        }
        StdModule::Re => matches!(name, "compile" | "match" | "search" | "findall" | "finditer" | "sub" | "split" | "fullmatch" | "escape" | "IGNORECASE"),
        StdModule::Itertools => matches!(
            name,
            "accumulate"
                | "product"
                | "takewhile"
                | "dropwhile"
                | "filterfalse"
                | "zip_longest"
                | "chain"
                | "groupby"
                | "islice"
                | "count"
                | "cycle"
                | "repeat"
                | "combinations"
                | "combinations_with_replacement"
                | "permutations"
                | "pairwise"
                | "starmap"
                | "compress"
        ),
        // singledispatch has no runtime item either, but it IS a known
        // functools name: the decorator is handled at conversion time
        // (issue #181), so the import drops silently below rather than
        // warning about a missing runtime counterpart.
        StdModule::Functools => matches!(
            name,
            "reduce" | "partial" | "lru_cache" | "cache" | "singledispatch"
        ),
        StdModule::Hashlib => {
            matches!(name, "md5" | "sha1" | "sha256" | "sha512" | "new")
        }
        StdModule::Json => matches!(name, "dumps" | "loads" | "load" | "dump"),
        StdModule::Datetime => crate::DatetimeType::from_name(name).is_some(),
        // os: enumerated — the runtime module has these (and only these);
        // anything else (`from os import PathLike` — annotation-only)
        // drops loudly and maps to the boxed PyValue.
        StdModule::Os => matches!(
            name,
            "chdir"
                | "environ"
                | "execv"
                | "getcwd"
                | "getenv"
                | "name"
                | "putenv"
                | "remove"
                | "replace"
                | "sep"
                | "urandom"
                | "close"
                | "write"
                | "fdopen"
                | "fstat"
                | "abspath"
                | "basename"
                | "dirname"
                | "exists"
                | "expanduser"
                | "isdir"
                | "isfile"
                | "join"
                | "normpath"
                | "relpath"
                | "split"
                | "splitext"
                | "path"
        ),
        StdModule::Sys
        | StdModule::Time
        | StdModule::Math
        | StdModule::Random
        | StdModule::Warnings
        | StdModule::Tempfile
        | StdModule::Textwrap
        | StdModule::Heapq
        | StdModule::Copy
        | StdModule::String
        | StdModule::Glob
        | StdModule::Pathlib
        | StdModule::Csv
        | StdModule::Subprocess
        | StdModule::Sysconfig
        | StdModule::Argparse
        | StdModule::Venv => true,
        // numpy and asyncio have no from-import item registry: their
        // names resolve through the module paths only.
        StdModule::Numpy | StdModule::Asyncio => false,
    }
}

/// Whether an imported name RESOLVES to an EXTERNAL module's item (a
/// re-export chain ending in `from urllib.parse import urlparse` — requests'
/// compat, where urllib is external): no runtime item exists behind the
/// chain, so the use drops.
/// Whether a name is imported (directly or through a chain) from a
/// vendored `[python-modules]` dependency — such names are NOT external
/// (the dep is compiled into the crate).
pub(crate) fn import_from_python_module(
    name: &str,
    symbols: &SymbolTableScopes,
    options: &PythonOptions,
) -> bool {
    let Some(SymbolTableNode::ImportFrom(ifm)) = symbols.get(name) else {
        return false;
    };
    let root = ifm.module.split('.').next().unwrap_or("");
    options.python_modules.contains(root)
}

/// Whether a name was bound by `from urllib.parse import X` (or another
/// urllib submodule) where the stdpython runtime has no item for X — the
/// import itself was already dropped with a warning, so CALLS through
/// the name must drop to the boxed None the same way (issue #137:
/// urllib3's `urlencode(fields)` rendered as a `urlencode::new(...)`
/// class construction). Scoped to urllib: its functions have no runtime
/// items and no call-lowering special arms.
pub(crate) fn import_dropped_stdpython_item(
    name: &str,
    symbols: &SymbolTableScopes,
) -> bool {
    let Some(SymbolTableNode::ImportFrom(ifm)) = symbols.get(name) else {
        return false;
    };
    let first = ifm.module.split('.').next().unwrap_or("");
    if crate::StdModule::from_name(first) != Some(crate::StdModule::Urllib) {
        return false;
    }
    let canonical = ifm
        .names
        .iter()
        .find(|a| a.asname.as_deref() == Some(name))
        .map(|a| a.name.as_str())
        .unwrap_or(name);
    !stdpython_module_item(first, canonical)
}

pub(crate) fn resolves_to_external_import(
    name: &str,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    // Only meaningful when the whole crate is known (multi-module
    // conversion): a single-module conversion may import any sibling.
    if options.module_defs.len() <= 1 {
        return false;
    }
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    // The package path of the CURRENT module in the chain, so its RELATIVE
    // imports resolve against the right context (resolve_imported_class's
    // model): options.module_path for the caller, then the defining
    // module's package path at each hop.
    let mut module_path = options.module_path.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(SymbolTableNode::ImportFrom(ifm)) => {
                let mut ctx = options.clone();
                ctx.module_path = module_path.clone();
                let path = ifm.resolved_module_path(&ctx);
                let Some(key) = crate::module_defs_key(&options, &path) else {
                    // The terminal hop: external when the module is neither
                    // stdpython nor a vendored python-module dep.
                    let root = ifm.module.split('.').next().unwrap_or("");
                    // `collections.abc` is the compile-time-only typing
                    // abstraction (Mapping, Iterable, ...): its names have
                    // no runtime items anywhere — external.
                    if ifm.module == "collections.abc" {
                        return true;
                    }
                    return !is_stdpython_module(root)
                        && !options.python_modules.contains(root);
                };
                {
                    // A re-export chain: hop into the defining module.
                    let is_package = options.module_defs.keys().any(|k| {
                        k.len() > key.len() && k[..key.len()] == key[..]
                    });
                    module_path = if is_package {
                        key.to_vec()
                    } else {
                        key[..key.len().saturating_sub(1)].to_vec()
                    };
                    let defining = ifm
                        .names
                        .iter()
                        .find(|a| a.asname.as_deref() == Some(&current))
                        .map(|a| a.name.clone())
                        .unwrap_or_else(|| current.clone());
                    let module = &options.module_defs[key];
                    let module: &crate::Module = module;
                    syms = module.clone().find_symbols(SymbolTableScopes::new());
                    current = defining;
                }
            }
            _ => return false,
        }
    }
    false
}

/// Whether an imported name is a TYPE-NAME TUPLE alias (`basestring =
/// (str, bytes)` — requests' compat): consumed by isinstance resolution at
/// conversion time, never a runtime value. Follows ImportFrom re-export
/// chains through the generated crate.
fn is_type_name_tuple_alias(
    name: &str,
    options: &PythonOptions,
    symbols: &SymbolTableScopes,
) -> bool {
    let mut current = name.to_string();
    let mut syms = symbols.clone();
    for _ in 0..16 {
        match syms.get(&current) {
            Some(SymbolTableNode::Assign { value, .. }) => {
                let crate::ExprType::Tuple(t) = value else {
                    return false;
                };
                // Every element must be a TYPE NAME (basestring = (str,
                // bytes)) — not an arbitrary runtime name: requests'
                // `_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR,
                // _VALID_HEADER_VALUE_RE_STR)` is a runtime tuple of
                // compiled-regex statics, imported and indexed at runtime.
                return t.elts.iter().all(|e| matches!(e, crate::ExprType::Name(n)
                    if matches!(
                        n.id.as_str(),
                        "str" | "bytes" | "bytearray" | "int" | "float" | "bool"
                            | "object" | "None" | "Any" | "Union"
                    )));
            }
            Some(SymbolTableNode::Alias(canonical)) => {
                current = canonical.clone();
            }
            Some(SymbolTableNode::ImportFrom(ifm)) => {
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
                syms = module.clone().find_symbols(SymbolTableScopes::new());
                current = defining;
            }
            _ => return false,
        }
    }
    false
}

/// Runtime modules that only exist on stdpython's std tier — a property
/// of [`crate::StdModule`], never a second name list that could drift
/// from the module set.
pub(crate) fn is_std_only_module(name: &str) -> bool {
    crate::StdModule::from_name(name).is_some_and(|m| m.is_std_only())
}

/// The conversion-time error for a std-tier import under the no_std
/// profile. Failing here beats failing later with an unresolved-name error
/// in the generated crate.
fn std_only_import_error(module: &str) -> Box<dyn std::error::Error> {
    format!(
        "`import {}` requires stdpython's std tier (it needs the OS), which the \
         no_std profile does not provide; remove the import or convert without \
         the no_std profile",
        module
    )
    .into()
}

#[derive(Clone, Debug, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct Alias {
    pub name: String,
    pub asname: Option<String>,
}

#[derive(Clone, Debug, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct Import {
    pub names: Vec<Alias>,
}

/// An Import (or FromImport) statement causes 2 things to occur:
/// 1. Declares the imported object within the existing scope.
/// 2. Causes the referenced module to be compiled into the program (only once).

/// The crate modules an import statement LOADS, in Python's order — the
/// modules whose bodies run at the import site (issue #333, Devin review
/// on #336): for `import a.b.c`, each package on the path that is a crate
/// module, then the module; for `from .a import x, y`, the resolved
/// module (its crate packages first) and each imported name that is a
/// submodule. The package root (`from . import x`) is listed as the
/// [`ROOT_INIT_MODULE`] path, which both crates answer: the lib root
/// through a shim, the binary through the root's body as that module.
/// Modules outside the crate are never listed; a single-module
/// conversion (no `module_defs`) lists nothing.
pub(crate) fn imported_crate_modules(
    stmt: &crate::StatementType,
    options: &PythonOptions,
) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let push_chain = |path: &[String], out: &mut Vec<Vec<String>>| {
        for len in 1..=path.len() {
            if let Some(key) = crate::module_defs_key(options, &path[..len]) {
                if key == options.this_module_path.as_slice() {
                    continue;
                }
                let key = init_module_path(key);
                if !out.contains(&key) {
                    out.push(key);
                }
            }
        }
    };
    match stmt {
        crate::StatementType::Import(i) => {
            for a in &i.names {
                let path: Vec<String> = a.name.split('.').map(|s| s.to_string()).collect();
                push_chain(&path, &mut out);
            }
        }
        crate::StatementType::ImportFrom(i) => {
            let base = i.resolved_module_path(options);
            if base.is_empty() {
                // `from . import x` at the package's top level: the root
                // itself, whose empty path no chain prefix names.
                if let Some(key) = crate::module_defs_key(options, &base)
                    && key != options.this_module_path.as_slice()
                {
                    out.push(init_module_path(key));
                }
            } else {
                push_chain(&base, &mut out);
            }
            // A from-list name is the package's own attribute when the
            // package binds it; only otherwise does Python import the
            // submodule of that name (Devin review on #338).
            let package_key = crate::module_defs_key(options, &base).map(<[String]>::to_vec);
            for a in i.names.iter().filter(|a| a.name != "*") {
                let mut sub = base.clone();
                sub.push(a.name.clone());
                let binding = package_key
                    .as_deref()
                    .and_then(|key| module_binding(options, key, &a.name));
                if let Some(binding) = binding {
                    // The package binds the name: the attribute, unless
                    // the binding has not happened yet when the import
                    // runs (a cycle) — then Python imports the submodule
                    // of that name, and so does the import site, at
                    // runtime, when one exists (`import_site_name_checks`;
                    // Devin review on #338, round 10). A binding under
                    // module-level control flow may not run at all: the
                    // converter says so (-W) when the submodule exists
                    // (round 6).
                    if binding.conditional && crate::module_defs_key(options, &sub).is_some() {
                        let warning = format!(
                            "`from {} import {}`: the package binds `{}` under a \
                             module-level condition and also has a submodule `{}`; \
                             Python decides at runtime which one the import finds, \
                             the converted program always takes the package's binding \
                             and never runs the submodule's body",
                            base.join("."),
                            a.name,
                            a.name,
                            sub.join(".")
                        );
                        let mut warnings = options.definition_warnings.borrow_mut();
                        if !warnings.contains(&warning) {
                            warnings.push(warning);
                        }
                    }
                    continue;
                }
                push_chain(&sub, &mut out);
            }
        }
        _ => {}
    }
    out
}

/// Where a crate module binds a name in its own body: the package
/// attribute a `from package import name` finds before falling back to
/// the submodule `package.name`, and the marks a cyclic importer asks
/// about.
pub(crate) struct ModuleBinding {
    /// The marks of EVERY statement that binds the name (see
    /// [`BindingMarks`]): the name is bound once any of them has run
    /// (Devin review on #338, round 9).
    pub marks: Vec<usize>,
    /// Every binding sits under module-level control flow (an `if`, a
    /// `try`), so Python binds the name only when a branch runs.
    pub conditional: bool,
}

/// The module-scope names one statement binds — the visitor's one
/// enumeration ([`Bindings::Scope`]).
fn stmt_bound_names(s: &crate::Statement) -> Vec<String> {
    crate::ast::tree::visit::stmt_bound_names(s, crate::ast::tree::visit::Bindings::Scope)
}

/// One binding of a module body: a statement, one name it binds, and
/// where that binding happens — the statement's index in the walk is
/// not the mark; each (statement, name) pair has its own bit (Devin
/// review on #338, round 12), since a statement binds its names at
/// different times (`Y = (X := f()) + g()` binds X at the walrus and Y
/// after; `with a() as X, b() as Y:` binds X before b() runs).
/// Where a name's mark is recorded: after the statement's init code for
/// a def or class name, an import alias, or a store's target (`after`);
/// otherwise by the lowering that binds it — a loop target at the top of
/// the body, a `with` item's target right after its context expression,
/// a walrus right after its store — each looking its own names up in
/// the statement's bits (`options.stmt_binds`).
#[derive(Clone, Debug)]
pub struct NameMark {
    pub name: String,
    /// The bit's index in the module's `__RYTHON_BOUND` words.
    pub mark: usize,
    /// Bound after the statement's init code: a def or class name, an
    /// import alias, a store's target (an assignment, an augmented one).
    pub after: bool,
}

/// One statement's marks (see [`NameMark`]) and whether it is a
/// top-level statement of the body (whose after-marks the module
/// emission records at the end of its init range; a nested one records
/// its own where it runs).
#[derive(Clone, Debug)]
pub struct StmtMarks {
    pub names: Vec<NameMark>,
    pub top_level: bool,
}

impl StmtMarks {
    /// The `__rython_bind__` calls for the names bound after the
    /// statement's init code, if any.
    pub(crate) fn after_binds(&self) -> Option<TokenStream> {
        bind_calls(self.names.iter().filter(|n| n.after).map(|n| n.mark))
    }

    /// `name -> (word, mask)` for the statement's lowering.
    pub(crate) fn bits(&self) -> std::collections::HashMap<String, (usize, u32)> {
        self.names
            .iter()
            .map(|n| (n.name.clone(), bound_word_and_mask(n.mark)))
            .collect()
    }
}

/// The `__rython_bind__` calls setting the given marks; None for none.
pub(crate) fn bind_calls(marks: impl Iterator<Item = usize>) -> Option<TokenStream> {
    let calls: Vec<TokenStream> = marks
        .map(|mark| {
            let (word, mask) = bound_word_and_mask(mark);
            quote!(__rython_bind__(#word, #mask);)
        })
        .collect();
    (!calls.is_empty()).then(|| quote!(#(#calls)*))
}

/// The `__rython_bind__` calls for `names` in a statement's bits map
/// (`options.stmt_binds`): what a loop, `with` or walrus lowering emits
/// for the names it has just bound.
pub(crate) fn binds_for<'a>(
    bits: Option<&std::collections::HashMap<String, (usize, u32)>>,
    names: impl Iterator<Item = &'a str>,
) -> Option<TokenStream> {
    let bits = bits?;
    let calls: Vec<TokenStream> = names
        .filter_map(|name| bits.get(name))
        .map(|(word, mask)| quote!(__rython_bind__(#word, #mask);))
        .collect();
    (!calls.is_empty()).then(|| quote!(#(#calls)*))
}

/// A module body's bindings — every (statement, name) pair for a
/// statement that binds a module-scope name, under module-level control
/// flow too, a def's own body and a TYPE_CHECKING block excluded — in
/// source order, each with
/// where the binding happens and whether the statement is top-level. A
/// pair's index here is its MARK: one bit of the module's
/// `__RYTHON_BOUND` words, set where the binding happens (after the
/// statement's init code; at the top of a loop body or after a `with`
/// item; right after a walrus's store), which is what a cyclic
/// importer's bound check reads (Devin review on #338, rounds 8 to 12).
/// The body is the module's NORMALIZED one
/// (`module::normalize_module_body`), on both sides.
fn binding_entries(body: &[crate::Statement]) -> Vec<(&crate::Statement, NameMark, bool)> {
    use crate::ast::tree::visit::{stmt_targets, target_names, walk_stmts, Descend, Flow};
    let mut out: Vec<(&crate::Statement, NameMark, bool)> = Vec::new();
    walk_stmts(body, Descend::SkipDefs, &mut |s| {
        // A TYPE_CHECKING block is compile-time only: nothing under it
        // binds at runtime, as the emission and the runtime-item check
        // already hold (Devin review on #338, round 14).
        if let crate::StatementType::If(i) = &s.statement
            && crate::ast::tree::module::Module::is_type_checking_test(&i.test)
        {
            return Flow::Skip;
        }
        let names = stmt_bound_names(s);
        if names.is_empty() {
            return Flow::Continue;
        }
        let is_loop = matches!(
            &s.statement,
            crate::StatementType::For(_)
                | crate::StatementType::AsyncFor(_)
                | crate::StatementType::With(_)
                | crate::StatementType::AsyncWith(_)
        );
        let stored: Vec<String> = stmt_targets(s)
            .into_iter()
            .flat_map(target_names)
            .map(str::to_string)
            .collect();
        let declared: Vec<String> = match &s.statement {
            crate::StatementType::FunctionDef(f) | crate::StatementType::AsyncFunctionDef(f) => {
                vec![f.name.clone()]
            }
            crate::StatementType::ClassDef(c) => vec![c.name.clone()],
            crate::StatementType::Import(_) | crate::StatementType::ImportFrom(_) => {
                names.clone()
            }
            _ => Vec::new(),
        };
        let top_level = body.iter().any(|top| std::ptr::eq(top, s));
        let mut seen: Vec<String> = Vec::new();
        for name in names {
            if seen.contains(&name) {
                continue;
            }
            let after = declared.contains(&name) || (!is_loop && stored.contains(&name));
            let mark = NameMark {
                mark: out.len(),
                after,
                name: name.clone(),
            };
            out.push((s, mark, top_level));
            seen.push(name);
        }
        Flow::Continue
    });
    out
}

/// The marks of a module body by source position (see
/// [`binding_entries`]): what the module emission and the statement
/// lowering consult to record each binding where it happens.
pub(crate) struct BindingMarks {
    pub by_pos: std::collections::HashMap<(usize, usize), StmtMarks>,
    /// How many marks the body has (the size of the bound bitmap).
    pub count: usize,
}

/// The bits of one `__RYTHON_BOUND` word: `AtomicU32`, which every
/// target with atomics has (the no_std tier included).
pub(crate) const BOUND_WORD_BITS: usize = 32;

impl BindingMarks {
    pub(crate) fn of(body: &[crate::Statement]) -> Self {
        let entries = binding_entries(body);
        let count = entries.len();
        let mut by_pos: std::collections::HashMap<(usize, usize), StmtMarks> =
            std::collections::HashMap::new();
        for (s, mark, top_level) in entries {
            let (Some(line), Some(col)) = (s.lineno, s.col_offset) else {
                continue;
            };
            by_pos
                .entry((line, col))
                .or_insert_with(|| StmtMarks { names: Vec::new(), top_level })
                .names
                .push(mark);
        }
        Self { by_pos, count }
    }

    /// The `__rython_bind__` calls for the names the statement at `pos`
    /// binds after its init code, if any.
    pub(crate) fn after_binds(&self, pos: (usize, usize)) -> Option<TokenStream> {
        self.by_pos.get(&pos)?.after_binds()
    }
}

/// The word index and bit mask of a mark in the bound bitmap.
pub(crate) fn bound_word_and_mask(mark: usize) -> (usize, u32) {
    (mark / BOUND_WORD_BITS, 1u32 << (mark % BOUND_WORD_BITS))
}

/// A list method that keeps the list's membership: `__all__.copy()`,
/// `.count(x)`, `.index(x)`, `.sort()`, `.reverse()` leave the export
/// set as it was. Any other method (`append`, `extend`, `insert`,
/// `remove`, `pop`, `clear`, one the analysis does not know) may not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ListReadMethod {
    Copy,
    Count,
    Index,
    Sort,
    Reverse,
}

impl ListReadMethod {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "copy" => Self::Copy,
            "count" => Self::Count,
            "index" => Self::Index,
            "sort" => Self::Sort,
            "reverse" => Self::Reverse,
            _ => return None,
        })
    }
}

/// A builtin that reads a list handed to it without mutating it
/// (`len(__all__)`, `sorted(__all__)`, `print(__all__)`, ...). Any
/// other callee may mutate what it is handed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NonMutatingBuiltin {
    Len, Sorted, List, Tuple, Set, FrozenSet, Print, Iter, Enumerate, Reversed, Any, All, Min,
    Max, Sum, Str, Repr, Bool, IsInstance, Map, Filter, Zip, Dict, Id, Type,
}

impl NonMutatingBuiltin {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "len" => Self::Len, "sorted" => Self::Sorted, "list" => Self::List,
            "tuple" => Self::Tuple, "set" => Self::Set, "frozenset" => Self::FrozenSet,
            "print" => Self::Print, "iter" => Self::Iter, "enumerate" => Self::Enumerate,
            "reversed" => Self::Reversed, "any" => Self::Any, "all" => Self::All,
            "min" => Self::Min, "max" => Self::Max, "sum" => Self::Sum, "str" => Self::Str,
            "repr" => Self::Repr, "bool" => Self::Bool, "isinstance" => Self::IsInstance,
            "map" => Self::Map, "filter" => Self::Filter, "zip" => Self::Zip,
            "dict" => Self::Dict, "id" => Self::Id, "type" => Self::Type,
            _ => return None,
        })
    }
}

/// The crate module's `__all__` as a list of names: `Ok(Some(names))`
/// for a literal list or tuple of string constants bound at the top
/// level (the last one), `Ok(None)` when the module binds no `__all__`,
/// `Err(())` when it binds one any other way (computed, augmented, under
/// control flow) — then its star exports are unknown.
pub(crate) fn literal_all(
    options: &PythonOptions,
    key: &[String],
) -> Result<Option<Vec<String>>, ()> {
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return Ok(None);
    };
    let string_items = |e: &crate::ExprType| -> Option<Vec<String>> {
        let items: &[crate::ExprType] = match e {
            crate::ExprType::List(items) => items,
            crate::ExprType::Tuple(t) => &t.elts,
            _ => return None,
        };
        items
            .iter()
            .map(|item| match item {
                crate::ExprType::Constant(c) => match &c.0 {
                    Some(litrs::Literal::String(lit)) => Some(lit.value().to_string()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    };
    // The LATEST effective binding in source order: a top-level literal
    // replaces whatever came before (a computed or conditional one
    // included); a later unknown binding, or a mutation of the list
    // (`__all__.append(...)`, `__all__ += [...]`, `__all__[0] = ...`)
    // anywhere the module runs, makes the exports unknown again (Devin
    // review on #338, round 21).
    // What MAY change the list's membership, or let something else
    // change it: a store through it (`__all__[i] = ...`), a method that
    // changes membership or one the analysis does not know, the list
    // handed to a callee that is not a known non-mutating builtin (by
    // position or by keyword), and — Devin review on #338, round 24 —
    // any other use of the name that could ALIAS the list (`exports =
    // __all__`, a tuple or list holding it, a walrus, a return): after
    // an alias escapes, a mutation through it is invisible here, so the
    // exports are unknown from then on. Only the reads the analysis can
    // see through leave the literal in force: a membership-keeping
    // method, a known builtin that is not shadowed by a module binding,
    // an `in` test, a subscript read, a loop over it. A def or class
    // body naming `__all__` at all (a `global __all__` mutator a
    // module-level call may run) makes the exports unknown too.
    let ctx = crate::ast::tree::module::defining_module_context(options, key);
    let mutates_all = |st: &crate::Statement| -> bool {
        use crate::ast::tree::visit::{any_expr_for, stmt_exprs, stmt_targets, Descend};
        let is_all = |e: &crate::ExprType| matches!(e, crate::ExprType::Name(n) if n.id == "__all__");
        let through_all = |e: &crate::ExprType| -> bool {
            match e {
                crate::ExprType::Attribute(a) => is_all(&a.value),
                crate::ExprType::Subscript(sub) => is_all(&sub.value),
                _ => false,
            }
        };
        if stmt_targets(st).into_iter().any(through_all) {
            return true;
        }
        // A builtin's name the module binds itself is not the builtin.
        let unshadowed_builtin =
            |name: &str| NonMutatingBuiltin::from_name(name).is_some() && callee_bindings(&body, &ctx, name).is_empty();
        for e in stmt_exprs(st) {
            // The occurrences of `__all__` the analysis can see through.
            let mut safe: Vec<*const crate::ExprType> = Vec::new();
            any_expr_for(e, Descend::OwnScope, |x| {
                match x {
                    crate::ExprType::Call(c) => match c.func.as_ref() {
                        crate::ExprType::Attribute(a)
                            if is_all(&a.value) && ListReadMethod::from_name(&a.attr).is_some() =>
                        {
                            safe.push(a.value.as_ref() as *const _);
                        }
                        crate::ExprType::Name(f) if unshadowed_builtin(&f.id) => {
                            safe.extend(c.args.iter().filter(|a| is_all(a)).map(|a| a as *const _));
                            safe.extend(
                                c.keywords.iter().filter(|k| is_all(&k.value)).map(|k| &k.value as *const _),
                            );
                        }
                        _ => {}
                    },
                    crate::ExprType::Compare(cmp) => {
                        if is_all(&cmp.left) {
                            safe.push(cmp.left.as_ref() as *const _);
                        }
                        safe.extend(cmp.comparators.iter().filter(|a| is_all(a)).map(|a| a as *const _));
                    }
                    crate::ExprType::Subscript(sub) if is_all(&sub.value) => {
                        safe.push(sub.value.as_ref() as *const _);
                    }
                    _ => {}
                }
                false
            });
            // A loop over the list reads it.
            if let crate::StatementType::For(f) = &st.statement
                && is_all(&f.iter)
            {
                safe.push(&f.iter as *const _);
            }
            let mut escapes = false;
            any_expr_for(e, Descend::OwnScope, |x| {
                if is_all(x) && !safe.iter().any(|p| std::ptr::eq(*p, x)) {
                    escapes = true;
                }
                escapes
            });
            if escapes {
                return true;
            }
        }
        false
    };
    // A def or class body naming `__all__`: a mutator a module-level
    // call may run during initialization.
    let named_in_a_def = {
        use crate::ast::tree::visit::{any_expr_for, stmt_exprs, walk_stmts, Descend, Flow};
        let mut named = false;
        walk_stmts(&body, Descend::All, &mut |st| {
            let in_def = !body.iter().any(|top| std::ptr::eq(top, st));
            if in_def
                && (matches!(&st.statement, crate::StatementType::Global(ns) if ns.iter().any(|n| n == "__all__"))
                    || stmt_exprs(st).into_iter().any(|e| {
                        any_expr_for(e, Descend::All, |x| {
                            matches!(x, crate::ExprType::Name(n) if n.id == "__all__")
                        })
                    }))
            {
                named = true;
                return Flow::Stop;
            }
            Flow::Continue
        });
        named
    };
    if named_in_a_def {
        return Err(());
    }
    let mut all: Result<Option<Vec<String>>, ()> = Ok(None);
    let mut all_by_stmt: std::collections::HashMap<(usize, usize), Option<Vec<String>>> =
        std::collections::HashMap::new();
    for (st, mark, top_level) in binding_entries(&body) {
        if mark.name != "__all__" {
            continue;
        }
        let literal = match &st.statement {
            crate::StatementType::Assign(a) if top_level => match a.targets.as_slice() {
                [crate::ExprType::Name(t)] if t.id == "__all__" => string_items(&a.value),
                _ => None,
            },
            _ => None,
        };
        if let (Some(line), Some(col)) = (st.lineno, st.col_offset) {
            all_by_stmt.insert((line, col), literal);
        }
    }
    // Source order over the body's statements (the entries are in walk
    // order too, but a mutation is not a binding entry).
    crate::ast::tree::visit::walk_stmts(
        &body,
        crate::ast::tree::visit::Descend::SkipDefs,
        &mut |st| {
            if let (Some(line), Some(col)) = (st.lineno, st.col_offset)
                && let Some(literal) = all_by_stmt.get(&(line, col))
            {
                all = match literal {
                    Some(names) => Ok(Some(names.clone())),
                    None => Err(()),
                };
            } else if let crate::StatementType::Delete(targets) = &st.statement
                && targets.iter().any(|t| matches!(t, crate::ExprType::Name(n) if n.id == "__all__"))
            {
                // `del __all__` unbinds it: the star import is back to
                // the public names (at the top level), or unknown
                // (under control flow).
                all = if body.iter().any(|top| std::ptr::eq(top, st)) {
                    Ok(None)
                } else {
                    Err(())
                };
            } else if mutates_all(st) {
                all = Err(());
            }
            crate::ast::tree::visit::Flow::Continue
        },
    );
    all
}

/// The names `from m import *` binds from the crate module at `key`, as
/// Python takes them: the module's `__all__` when its body binds one as
/// a list or tuple of string literals at the top level (the last such
/// binding); otherwise every name its normalized body binds at module
/// scope (a def's locals excluded) that does not start with `_`, plus
/// the exports of its own crate-module star imports (depth-bounded).
/// None when the exports are UNKNOWN: an `__all__` bound any other way
/// (computed, augmented, under control flow), a star import of an
/// external module, or a chain deeper than the bound (Devin review on
/// #338, round 20).
pub(crate) fn star_exports(
    options: &PythonOptions,
    key: &[String],
    depth: usize,
) -> Option<Vec<String>> {
    if depth > 8 {
        return None;
    }
    let body = crate::ast::tree::module::normalized_body_of(options, key)?;
    let ctx = crate::ast::tree::module::defining_module_context(options, key);
    match literal_all(options, key) {
        Ok(Some(all)) => return Some(all),
        Ok(None) => {}
        Err(()) => return None,
    }
    // Source order, so a later `del x` at the top level removes x (a
    // module-level `del` is a no-op in the emission, so the static
    // outlives the name — the export list must not) and a later store
    // rebinds it; a `del` of a public name under control flow leaves the
    // exports UNKNOWN (Python decides at runtime) — a star import of the
    // module is then refused (Devin review on #338, round 22).
    if !module_conditional_deletes(options, key).is_empty() {
        return None;
    }
    let mut names: Vec<String> = Vec::new();
    let mut unknown = false;
    crate::ast::tree::visit::walk_stmts(
        &body,
        crate::ast::tree::visit::Descend::SkipDefs,
        &mut |st| {
            if let crate::StatementType::If(i) = &st.statement
                && crate::ast::tree::module::Module::is_type_checking_test(&i.test)
            {
                return crate::ast::tree::visit::Flow::Skip;
            }
            if let crate::StatementType::Delete(targets) = &st.statement {
                for t in targets {
                    if let crate::ExprType::Name(n) = t {
                        names.retain(|m| m != &n.id);
                    }
                }
                return crate::ast::tree::visit::Flow::Continue;
            }
            for name in stmt_bound_names(st) {
                if name == "*" {
                    let crate::StatementType::ImportFrom(i) = &st.statement else {
                        continue;
                    };
                    let path = i.resolved_module_path(&ctx);
                    match crate::module_defs_key(options, &path)
                        .and_then(|source| star_exports(options, source, depth + 1))
                    {
                        Some(exports) => {
                            for n in exports {
                                if !names.contains(&n) {
                                    names.push(n);
                                }
                            }
                        }
                        None => unknown = true,
                    }
                } else if !name.starts_with('_') && !names.contains(&name) {
                    names.push(name);
                }
            }
            crate::ast::tree::visit::Flow::Continue
        },
    );
    if unknown {
        return None;
    }
    Some(names)
}

/// The public names the crate module at `key` deletes under module-level
/// control flow (`if flag: del x`): whether the name is bound when the
/// body has run is decided at runtime.
pub(crate) fn module_conditional_deletes(options: &PythonOptions, key: &[String]) -> Vec<String> {
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    crate::ast::tree::visit::walk_stmts(
        &body,
        crate::ast::tree::visit::Descend::SkipDefs,
        &mut |st| {
            if let crate::StatementType::Delete(targets) = &st.statement
                && !body.iter().any(|top| std::ptr::eq(top, st))
            {
                for t in targets {
                    if let crate::ExprType::Name(n) = t
                        && !n.id.starts_with('_')
                        && !names.contains(&n.id)
                    {
                        names.push(n.id.clone());
                    }
                }
            }
            crate::ast::tree::visit::Flow::Continue
        },
    );
    names
}

/// The names a `from m import *` of the crate module at `key` must
/// re-export EXPLICITLY (`use m::{a, b}`) rather than by glob: when `m`
/// has a literal `__all__` (the glob would expose a name it leaves out),
/// or deletes a public name at the top level (the glob would expose the
/// deleted name's static). None when the glob is right.
pub(crate) fn star_reexport_list(options: &PythonOptions, key: &[String]) -> Option<Vec<String>> {
    if let Ok(Some(all)) = literal_all(options, key) {
        return Some(all);
    }
    let body = crate::ast::tree::module::normalized_body_of(options, key)?;
    let deletes_public = body.iter().any(|st| match &st.statement {
        crate::StatementType::Delete(targets) => targets
            .iter()
            .any(|t| matches!(t, crate::ExprType::Name(n) if !n.id.starts_with('_'))),
        _ => false,
    });
    if deletes_public {
        return star_exports(options, key, 0);
    }
    None
}

/// The names a sibling's `from m import *` takes from the crate module
/// at `key`: its star exports when the conversion can enumerate them,
/// otherwise every public module-scope name (the conservative superset
/// — what the glob re-export exposes), so each gets its runtime item
/// (Devin review on #338, round 21).
pub(crate) fn sibling_star_names(options: &PythonOptions, key: &[String]) -> Vec<String> {
    if let Some(names) = star_exports(options, key, 0) {
        return names;
    }
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for (_, mark, _) in binding_entries(&body) {
        if mark.name != "*" && !mark.name.starts_with('_') && !names.contains(&mark.name) {
            names.push(mark.name.clone());
        }
    }
    names
}

/// What a `from X import *` statement of the module in `ctx` binds:
/// `Some(Some(names))` for a crate module's exports, `Some(None)` when
/// they are unknown (an external module, an unresolvable `__all__`), and
/// None for any other statement.
fn star_import_exports(
    options: &PythonOptions,
    ctx: &PythonOptions,
    s: &crate::Statement,
) -> Option<Option<Vec<String>>> {
    let crate::StatementType::ImportFrom(i) = &s.statement else {
        return None;
    };
    if !i.names.iter().any(|a| a.name == "*") {
        return None;
    }
    Some(
        crate::module_defs_key(options, &i.resolved_module_path(ctx))
            .and_then(|source| star_exports(options, source, 0)),
    )
}

/// Whether the crate module at `key` star-imports a module whose exports
/// the conversion cannot enumerate (see [`star_exports`]): then whether
/// it binds a given name is unknown.
fn module_star_imports_unknown(options: &PythonOptions, key: &[String]) -> bool {
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return false;
    };
    let ctx = crate::ast::tree::module::defining_module_context(options, key);
    binding_entries(&body)
        .into_iter()
        .any(|(s, mark, _)| mark.name == "*" && star_import_exports(options, &ctx, s) == Some(None))
}

/// The bindings of `name` in the body of the crate module at `key` (a
/// def's locals excluded): every binding's mark, and whether all of them
/// are nested under module-level control flow. A package's
/// own `from . import name` binds the SUBMODULE — it is the submodule
/// import, not an attribute the package has before it — so it never
/// counts (the entry's `from . import helper` in its `__init__`).
fn module_binding(options: &PythonOptions, key: &[String], name: &str) -> Option<ModuleBinding> {
    // The target module's NORMALIZED body — the sequence its own emission
    // numbers (Devin review on #338, round 11).
    let body = crate::ast::tree::module::normalized_body_of(options, key)?;
    let ctx = crate::ast::tree::module::defining_module_context(options, key);
    let imports_own_submodule = |s: &crate::Statement| match &s.statement {
        crate::StatementType::ImportFrom(i) => {
            crate::module_defs_key(options, &i.resolved_module_path(&ctx)) == Some(key)
        }
        _ => false,
    };
    // A `from m import *` binds every name `m` exports under the star
    // statement's one mark (the bit is set when the statement has run,
    // for all of them at once — Devin review on #338, round 20).
    let star_binds = |s: &crate::Statement| -> bool {
        matches!(star_import_exports(options, &ctx, s), Some(Some(names)) if names.iter().any(|n| n == name))
    };
    let bindings: Vec<(usize, bool)> = binding_entries(&body)
        .into_iter()
        .filter(|(s, mark, _)| {
            (mark.name == name || (mark.name == "*" && star_binds(s))) && !imports_own_submodule(s)
        })
        .map(|(_, mark, top_level)| (mark.mark, top_level))
        .collect();
    if bindings.is_empty() {
        return None;
    }
    Some(ModuleBinding {
        conditional: bindings.iter().all(|(_, top_level)| !top_level),
        marks: bindings.into_iter().map(|(mark, _)| mark).collect(),
    })
}

/// What a `from package import name` does for each name the package
/// binds, after the package's body ran (Devin review on #338, rounds 6,
/// 8 and 10). While that body is still running on this thread (an
/// import cycle; a self-import included), the binding may not have
/// happened yet — each module's `__rython_bound__` answers from its
/// bound bitmap (the binding statements' bits, set where they ran; a
/// completed body answers yes):
/// - when a submodule `package.name` exists, Python imports it in that
///   case, so the site runs its body then and only then;
/// - otherwise Python raises ImportError (`cannot import name ... from
///   partially initialized module ...`), and so does the check, rather
///   than handing out the static's eventual value.
/// The package root (`from . import name`) is checked through
/// [`ROOT_INIT_MODULE`] like any module.
pub(crate) fn import_site_name_checks(
    stmt: &crate::StatementType,
    options: &PythonOptions,
) -> TokenStream {
    let crate::StatementType::ImportFrom(i) = stmt else {
        return quote!();
    };
    let base = i.resolved_module_path(options);
    let Some(key) = crate::module_defs_key(options, &base) else {
        return quote!();
    };
    let segs: Vec<_> = init_module_path(key)
        .iter()
        .map(|s| crate::safe_ident(s))
        .collect();
    let qualified = std::iter::once(options.python_namespace.as_str())
        .filter(|ns| !ns.is_empty())
        .chain(key.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(".");
    let checks = i.names.iter().filter_map(|a| {
        // `from package import *` asks for no one name; a cyclic star
        // import is refused at conversion (`import_site_init`).
        if a.name == "*" {
            return None;
        }
        let binding = module_binding(options, key, &a.name)?;
        let bits = binding.marks.iter().map(|mark| {
            let (word, mask) = bound_word_and_mask(*mark);
            quote!((#word, #mask))
        });
        let name = a.name.as_str();
        let bound = quote!(crate::#(#segs::)*__rython_bound__(&[#(#bits),*], #name, #qualified));
        let mut sub = key.to_vec();
        sub.push(a.name.clone());
        Some(match crate::module_defs_key(options, &sub) {
            Some(sub_key) => {
                let sub_segs: Vec<_> = sub_key.iter().map(|s| crate::safe_ident(s)).collect();
                quote! {
                    if #bound.is_err() {
                        crate::#(#sub_segs::)*__module_init__()?;
                    }
                }
            }
            None => quote!(#bound?;),
        })
    });
    quote!(#(#checks)*)
}

/// The module through which every crate reaches the package root's
/// init and bound check: the binary carries the root's body under this
/// name (rypip writes it beside the sibling modules), and the lib root
/// answers the same path with a shim onto its own items.
pub const ROOT_INIT_MODULE: &str = "__rython_root";

/// The crate path an init call or bound check for the module at `key`
/// takes: the module's own path, or [`ROOT_INIT_MODULE`] for the root.
fn init_module_path(key: &[String]) -> Vec<String> {
    if key.is_empty() {
        vec![ROOT_INIT_MODULE.to_string()]
    } else {
        key.to_vec()
    }
}

/// The `__module_init__` calls for the modules an import loads, in
/// order: each module's body runs once (the function is once-guarded),
/// exactly where Python runs it — the import site.
pub(crate) fn module_init_calls(paths: &[Vec<String>]) -> TokenStream {
    let calls = paths.iter().map(|path| {
        let segs: Vec<_> = path.iter().map(|s| crate::safe_ident(s)).collect();
        quote!(crate::#(#segs::)*__module_init__()?;)
    });
    quote!(#(#calls)*)
}

/// Whether the crate module at `key` deletes `name` at module scope
/// (`del name`, under module-level control flow too). A module-level
/// `del` lowers to a no-op (issue #112), so the static outlives the
/// binding: what a later `from package import name` finds — the
/// attribute, the submodule, or an ImportError — depends on the order
/// at runtime, which the converted program cannot represent.
fn module_deletes(options: &PythonOptions, key: &[String], name: &str) -> bool {
    use crate::ast::tree::visit::{any_stmt, Descend};
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return false;
    };
    any_stmt(&body, Descend::SkipDefs, |s| match &s.statement {
        crate::StatementType::Delete(targets) => targets
            .iter()
            .any(|t| matches!(t, crate::ExprType::Name(n) if n.id == name)),
        _ => false,
    })
}

/// Whether module `key`'s body binds `name` as a module-scope `except
/// ... as name` handler alias (nested definitions aside). Python binds
/// the alias for the handler's body and DELETES it when the handler
/// ends, so the name exists only while the handler runs: an import of it
/// from inside that window (the handler imports a sibling, which imports
/// the alias back — a cycle) sees the exception, and one from outside
/// finds nothing (Devin review on #338, round 17).
fn module_handler_alias(options: &PythonOptions, key: &[String], name: &str) -> bool {
    use crate::ast::tree::visit::{any_stmt, Descend};
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return false;
    };
    any_stmt(&body, Descend::SkipDefs, |s| match &s.statement {
        crate::StatementType::Try(t) => t.handlers.iter().any(|h| h.name.as_deref() == Some(name)),
        _ => false,
    })
}

/// The crate modules one import statement loads (every package on a
/// dotted path, the resolved module, each from-list name that is a
/// submodule), as module_defs keys, in `ctx` (the importing module's
/// package context).
fn stmt_import_keys(stmt: &crate::StatementType, ctx: &PythonOptions) -> Vec<Vec<String>> {
    let mut paths: Vec<Vec<String>> = Vec::new();
    match stmt {
        crate::StatementType::Import(i) => {
            for a in &i.names {
                let path: Vec<String> = a.name.split('.').map(str::to_string).collect();
                for len in 1..=path.len() {
                    paths.push(path[..len].to_vec());
                }
            }
        }
        crate::StatementType::ImportFrom(i) => {
            let base = i.resolved_module_path(ctx);
            for len in 0..=base.len() {
                paths.push(base[..len].to_vec());
            }
            for a in &i.names {
                let mut sub = base.clone();
                sub.push(a.name.clone());
                paths.push(sub);
            }
        }
        _ => {}
    }
    paths
        .iter()
        .filter_map(|p| crate::module_defs_key(ctx, p).map(<[String]>::to_vec))
        .collect()
}

/// Whether the crate module at `from` imports the module at `to`,
/// directly or through other crate modules — function-local imports
/// included, since a function `from`'s body calls may import it — so
/// `to` can be initialized WHILE `from`'s body runs (a cycle).
fn module_reaches(options: &PythonOptions, from: &[String], to: &[String]) -> bool {
    use crate::ast::tree::visit::{walk_stmts, Descend, Flow};
    if from == to {
        return true;
    }
    let mut seen: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
    let mut queue: Vec<Vec<String>> = vec![from.to_vec()];
    while let Some(current) = queue.pop() {
        if !seen.insert(current.clone()) {
            continue;
        }
        let Some(module) = options.module_defs.get(&current) else {
            continue;
        };
        let module: &crate::Module = module;
        let ctx = crate::ast::tree::module::defining_module_context(options, &current);
        let mut found = false;
        walk_stmts(&module.raw.body, Descend::All, &mut |s| {
            for key in stmt_import_keys(&s.statement, &ctx) {
                if key == to {
                    found = true;
                    return Flow::Stop;
                }
                queue.push(key);
            }
            Flow::Continue
        });
        if found {
            return true;
        }
    }
    false
}

/// A module body's local def by name (top-level, or under module-level
/// control flow).

/// What a module-level name a call may resolve to, by the module's own
/// bindings of it: a local def (its body is traced), a crate module (the
/// module's import chain is traced), an external module (a stdlib or
/// third-party import: its calls load no crate module), or something
/// the tracing cannot see into (a stored value, a class — its
/// constructor may import; a loop or `with` target; a handler alias).
enum CalleeBinding<'a> {
    Def(&'a crate::FunctionDef),
    Module(Vec<String>),
    External,
    Opaque,
}

/// The bindings of `name` that may be in force when a module-level call
/// of it runs, in source order, by Python's later-binding-wins rule: the
/// last UNCONDITIONAL (top-level) binding replaces everything before it,
/// and every binding under module-level control flow after it is an
/// alternative beside it (Devin review on #338, round 23 — the first
/// binding was taken before, missing a later `from .d import f` that
/// replaced `from .c import f`).
fn callee_bindings<'a>(
    body: &'a [crate::Statement],
    ctx: &PythonOptions,
    name: &str,
) -> Vec<CalleeBinding<'a>> {
    use crate::ast::tree::visit::{walk_stmts, Descend, Flow};
    let mut bindings: Vec<CalleeBinding<'a>> = Vec::new();
    walk_stmts(body, Descend::SkipDefs, &mut |st| {
        if let crate::StatementType::If(i) = &st.statement
            && crate::ast::tree::module::Module::is_type_checking_test(&i.test)
        {
            return Flow::Skip;
        }
        if !stmt_bound_names(st).iter().any(|n| n == name) {
            return Flow::Continue;
        }
        let binding = match &st.statement {
            crate::StatementType::FunctionDef(f) | crate::StatementType::AsyncFunctionDef(f)
                if f.name == name =>
            {
                CalleeBinding::Def(f)
            }
            crate::StatementType::ImportFrom(i)
                if i.names.iter().any(|a| a.asname.as_deref().unwrap_or(&a.name) == name) =>
            {
                match crate::module_defs_key(ctx, &i.resolved_module_path(ctx)) {
                    Some(key) => CalleeBinding::Module(key.to_vec()),
                    None => CalleeBinding::External,
                }
            }
            crate::StatementType::Import(i) => {
                let alias = i.names.iter().find(|a| {
                    a.asname
                        .as_deref()
                        .unwrap_or_else(|| a.name.split('.').next().unwrap_or(&a.name))
                        == name
                });
                match alias {
                    Some(a) => {
                        let path: Vec<String> = a.name.split('.').map(str::to_string).collect();
                        match crate::module_defs_key(ctx, &path) {
                            Some(key) => CalleeBinding::Module(key.to_vec()),
                            None => CalleeBinding::External,
                        }
                    }
                    None => CalleeBinding::Opaque,
                }
            }
            _ => CalleeBinding::Opaque,
        };
        if body.iter().any(|top| std::ptr::eq(top, st)) {
            bindings.clear();
        }
        bindings.push(binding);
        Flow::Continue
    });
    bindings
}

/// A builtin a module-level call may name without binding it: none of
/// these loads a crate module, except the ones that run code the tracing
/// cannot see (`exec`, `eval`, `__import__`, `getattr` — treated as
/// reaching).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InertBuiltin {
    Print, Len, Str, Int, Float, Bool, List, Dict, Set, Tuple, FrozenSet, Range, Enumerate, Zip,
    Map, Filter, Sorted, Reversed, IsInstance, IsSubclass, HasAttr, Min, Max, Sum, Abs, Repr,
    Type, Id, Hash, Iter, Next, Format, Round, DivMod, Pow, Chr, Ord, Any, All, Callable, Bytes,
    ByteArray, Slice, Open, Input, Vars, Dir, Object, Super, Property, StaticMethod, ClassMethod,
}

impl InertBuiltin {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "print" => Self::Print, "len" => Self::Len, "str" => Self::Str, "int" => Self::Int,
            "float" => Self::Float, "bool" => Self::Bool, "list" => Self::List, "dict" => Self::Dict,
            "set" => Self::Set, "tuple" => Self::Tuple, "frozenset" => Self::FrozenSet,
            "range" => Self::Range, "enumerate" => Self::Enumerate, "zip" => Self::Zip,
            "map" => Self::Map, "filter" => Self::Filter, "sorted" => Self::Sorted,
            "reversed" => Self::Reversed, "isinstance" => Self::IsInstance,
            "issubclass" => Self::IsSubclass, "hasattr" => Self::HasAttr, "min" => Self::Min,
            "max" => Self::Max, "sum" => Self::Sum, "abs" => Self::Abs, "repr" => Self::Repr,
            "type" => Self::Type, "id" => Self::Id, "hash" => Self::Hash, "iter" => Self::Iter,
            "next" => Self::Next, "format" => Self::Format, "round" => Self::Round,
            "divmod" => Self::DivMod, "pow" => Self::Pow, "chr" => Self::Chr, "ord" => Self::Ord,
            "any" => Self::Any, "all" => Self::All, "callable" => Self::Callable,
            "bytes" => Self::Bytes, "bytearray" => Self::ByteArray, "slice" => Self::Slice,
            "open" => Self::Open, "input" => Self::Input, "vars" => Self::Vars, "dir" => Self::Dir,
            "object" => Self::Object, "super" => Self::Super, "property" => Self::Property,
            "staticmethod" => Self::StaticMethod, "classmethod" => Self::ClassMethod,
            _ => return None,
        })
    }
}

/// Whether a module-level statement's calls can load the crate module at
/// `to` (its own imports are handled by the caller): a call of a local
/// def whose body imports it or calls a local def that does, or of a
/// function imported from a crate module that reaches `to`, or a method
/// called through such a module (`mod.f()`). Each called name resolves
/// by the module's bindings in force at the call (`callee_bindings`).
/// What the tracing cannot see into COUNTS AS REACHING — correct or
/// loud (Devin review on #338, round 23): a call of a stored value or a
/// class (its constructor may import), a method on an object, a call of
/// a call or a subscript, a free name that is no inert builtin
/// (`exec`, `eval`, `__import__`, `getattr`, a name bound by a function
/// through `global`). A builtin that runs no user code, and a function
/// or method of an external module, load no crate module.
fn stmt_calls_reach(
    options: &PythonOptions,
    ctx: &PythonOptions,
    body: &[crate::Statement],
    s: &crate::Statement,
    to: &[String],
) -> bool {
    use crate::ast::tree::visit::{any_expr_for, stmt_exprs, Descend};
    // Callee expressions of the statement's own calls.
    let mut callees: Vec<&crate::ExprType> = Vec::new();
    for e in stmt_exprs(s) {
        any_expr_for(e, Descend::OwnScope, |x| {
            if let crate::ExprType::Call(c) = x {
                callees.push(&c.func);
            }
            false
        });
    }
    if callees.is_empty() {
        return false;
    }
    // Whether a local def's body reaches `to`: its imports, or the local
    // defs it calls (a visited set bounds the recursion); what it calls
    // that the tracing cannot see into counts as reaching.
    fn def_reaches(
        options: &PythonOptions,
        ctx: &PythonOptions,
        body: &[crate::Statement],
        f: &crate::FunctionDef,
        to: &[String],
        visited: &mut Vec<String>,
    ) -> bool {
        use crate::ast::tree::visit::{any_expr_for, stmt_exprs, walk_stmts, Descend, Flow};
        if visited.contains(&f.name) {
            return false;
        }
        visited.push(f.name.clone());
        let mut reaches = false;
        walk_stmts(&f.body, Descend::All, &mut |st| {
            if stmt_import_keys(&st.statement, ctx)
                .iter()
                .any(|k| k == to || module_reaches(options, k, to))
            {
                reaches = true;
                return Flow::Stop;
            }
            for e in stmt_exprs(st) {
                any_expr_for(e, Descend::All, |x| {
                    if let crate::ExprType::Call(c) = x
                        && callee_reaches(options, ctx, body, &c.func, to, visited)
                    {
                        reaches = true;
                    }
                    reaches
                });
                if reaches {
                    return Flow::Stop;
                }
            }
            Flow::Continue
        });
        reaches
    }
    fn callee_reaches(
        options: &PythonOptions,
        ctx: &PythonOptions,
        body: &[crate::Statement],
        callee: &crate::ExprType,
        to: &[String],
        visited: &mut Vec<String>,
    ) -> bool {
        match callee {
            crate::ExprType::Name(n) => {
                let bindings = callee_bindings(body, ctx, &n.id);
                if bindings.is_empty() {
                    // A free name: an inert builtin loads nothing; any
                    // other (`exec`, `getattr`, a `global`-written name)
                    // may.
                    return InertBuiltin::from_name(&n.id).is_none();
                }
                bindings.into_iter().any(|b| match b {
                    CalleeBinding::Def(f) => def_reaches(options, ctx, body, f, to, visited),
                    CalleeBinding::Module(m) => m == to || module_reaches(options, &m, to),
                    CalleeBinding::External => false,
                    CalleeBinding::Opaque => true,
                })
            }
            crate::ExprType::Attribute(a) => match a.value.as_ref() {
                crate::ExprType::Name(n) => {
                    let bindings = callee_bindings(body, ctx, &n.id);
                    if bindings.is_empty() {
                        // `self.x()` inside a def, a free object: opaque.
                        return true;
                    }
                    bindings.into_iter().any(|b| match b {
                        CalleeBinding::Module(m) => m == to || module_reaches(options, &m, to),
                        CalleeBinding::External => false,
                        CalleeBinding::Def(_) | CalleeBinding::Opaque => true,
                    })
                }
                _ => true,
            },
            _ => true,
        }
    }
    let mut visited: Vec<String> = Vec::new();
    callees
        .into_iter()
        .any(|callee| callee_reaches(options, ctx, body, callee, to, &mut visited))
}

/// Whether the crate module at `key` binds `name` by an import (from
/// another module), then imports the module at `to` — directly, through
/// the modules that import runs, or through a call that imports it
/// (`stmt_calls_reach`) — and only then rebinds `name` by a def, a class
/// or a store. The import is dropped (the local
/// definition wins, `ImportFrom::to_rust`), so the converted module
/// holds one item, the definition; but `to`, initialized while `key`'s
/// body sits between the two bindings, would read the IMPORTED value in
/// Python (Devin review on #338, round 15). A rebinding that runs before
/// the cycle's import is the definition on both sides.
fn imported_value_exposed_to(
    options: &PythonOptions,
    key: &[String],
    name: &str,
    to: &[String],
) -> bool {
    use crate::ast::tree::visit::{walk_stmts, Descend, Flow};
    let Some(body) = crate::ast::tree::module::normalized_body_of(options, key) else {
        return false;
    };
    let ctx = crate::ast::tree::module::defining_module_context(options, key);
    let mut imported = false;
    let mut cycle_between = false;
    let mut exposed = false;
    walk_stmts(&body, Descend::SkipDefs, &mut |s| {
        let is_import = matches!(
            &s.statement,
            crate::StatementType::Import(_) | crate::StatementType::ImportFrom(_)
        );
        if imported && !cycle_between {
            let reaches_by_import = is_import
                && stmt_import_keys(&s.statement, &ctx)
                    .iter()
                    .any(|k| k == to || module_reaches(options, k, to));
            if reaches_by_import || stmt_calls_reach(options, &ctx, &body, s, to) {
                cycle_between = true;
            }
        }
        if !stmt_bound_names(s).iter().any(|n| n == name) {
            return Flow::Continue;
        }
        if is_import {
            // A self-import (`from .a import x` inside a) imports nothing
            // from elsewhere: the value Python would expose is this
            // module's own binding.
            let from_elsewhere = !stmt_import_keys(&s.statement, &ctx)
                .iter()
                .any(|k| k == key);
            if from_elsewhere {
                imported = true;
                cycle_between = false;
            }
        } else if imported && cycle_between {
            exposed = true;
            return Flow::Stop;
        } else {
            imported = false;
        }
        Flow::Continue
    });
    exposed
}

/// What an import statement runs at its site: the loaded modules' init
/// calls, then the checks of the from-list names the package binds —
/// the one lowering every import site (module level, nested,
/// function-local) shares. Empty when the statement loads no crate
/// module. A from-list name the package binds AND deletes is refused
/// (Devin review on #338, round 13): Python decides at runtime whether
/// the attribute, the submodule, or an ImportError answers, and the
/// static layout would hand out the deleted value.
pub(crate) fn import_site_init(
    stmt: &crate::StatementType,
    options: &PythonOptions,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    if let crate::StatementType::ImportFrom(i) = stmt {
        let base = i.resolved_module_path(options);
        if let Some(key) = crate::module_defs_key(options, &base) {
            // `from m import *` while `m` is partially initialized (m
            // starts first, reaches this module, which star-imports m
            // back): Python copies only the names m has bound by then;
            // the static glob binds every export, whenever. Refused
            // (Devin review on #338, round 21) — unless this module is
            // an ancestor package of m: Python initializes a package
            // before any of its submodules, so m cannot be running when
            // its package star-imports it (`__init__`'s `from .core
            // import *` over core's `from . import utils`, the idiom).
            if i.names.iter().any(|a| a.name == "*")
                && !key.starts_with(&options.this_module_path)
                && module_reaches(options, key, &options.this_module_path)
            {
                return Err(format!(
                    "`from {}{} import *` is refused: `{}` imports this module (directly, \
                     or through a call), so when `{}` starts first the star import runs \
                     while it is partially initialized and Python copies only the names \
                     bound by then, which the static glob cannot represent; import the \
                     names explicitly, or break the import cycle",
                    ".".repeat(i.level),
                    i.module,
                    key.join("."),
                    key.join(".")
                )
                .into());
            }
            if i.names.iter().any(|a| a.name == "*") {
                let conditional = module_conditional_deletes(options, key);
                if !conditional.is_empty() {
                    return Err(format!(
                        "`from {}{} import *` is refused: `{}` deletes `{}` under a \
                         module-level condition, so whether the star import binds it is \
                         decided at runtime, which the static re-export cannot represent; \
                         import the names explicitly, or delete unconditionally",
                        ".".repeat(i.level),
                        i.module,
                        key.join("."),
                        conditional.join("`, `")
                    )
                    .into());
                }
            }
            for a in i.names.iter().filter(|a| a.name != "*") {
                // The module imports the name, then imports THIS module
                // (a cycle), then redefines the name: Python exposes the
                // imported value here, which the converted module (one
                // item, the definition) cannot (Devin review on #338,
                // round 15). A rebinding before the cycle's import, or a
                // sibling outside the cycle, gets the definition on both
                // sides.
                if imported_value_exposed_to(options, key, &a.name, &options.this_module_path) {
                    return Err(format!(
                        "`from {}{} import {}` is refused: `{}` imports `{}`, then imports \
                         this module (directly, or through a call), then redefines `{}`, \
                         so Python exposes the imported value here where the converted \
                         module holds only its definition; import the value from the \
                         module that defines it, or move the definition before the import",
                        ".".repeat(i.level),
                        i.module,
                        a.name,
                        key.join("."),
                        a.name,
                        a.name
                    )
                    .into());
                }
                // The module binds the name as an `except ... as` alias —
                // bound for the handler's body, deleted at its end — and
                // reaches this module (the handler imports a sibling that
                // imports the alias back): Python hands out the exception
                // in that window and nothing after it, a bind-then-unbind
                // the bound bitmap (bits only set) cannot time; with a
                // same-named submodule the after-window answer is that
                // submodule instead (Devin review on #338, round 17).
                if module_handler_alias(options, key, &a.name)
                    && module_reaches(options, key, &options.this_module_path)
                {
                    return Err(format!(
                        "`from {}{} import {}` is refused: `{}` binds `{}` only as an \
                         `except ... as {}` alias, which Python deletes when the handler \
                         ends, and that module imports this one (directly, or through a \
                         call), so the import finds the exception while the handler runs \
                         and nothing (or a submodule of that name) after it, a timing the \
                         converted program cannot represent; bind the exception to a name \
                         the module keeps, or break the import cycle",
                        ".".repeat(i.level),
                        i.module,
                        a.name,
                        key.join("."),
                        a.name,
                        a.name
                    )
                    .into());
                }
                // The package star-imports a module whose exports the
                // conversion cannot enumerate, binds the name no other
                // way, and has a submodule of that name: whether Python
                // finds the attribute or imports the submodule depends on
                // what the star import bound — refused (Devin review on
                // #338, round 20).
                if a.name != "*"
                    && module_binding(options, key, &a.name).is_none()
                    && module_star_imports_unknown(options, key)
                    && {
                        let mut sub = key.to_vec();
                        sub.push(a.name.clone());
                        crate::module_defs_key(options, &sub).is_some()
                    }
                {
                    return Err(format!(
                        "`from {}{} import {}` is refused: {} star-imports a module whose \
                         exported names the conversion cannot enumerate (an external module, \
                         or an `__all__` that is not a literal list of strings) and also has \
                         a submodule `{}`, so whether Python finds an attribute or imports \
                         the submodule depends on what the star import bound; import the \
                         names explicitly, or give the module a literal `__all__`",
                        ".".repeat(i.level),
                        i.module,
                        a.name,
                        if key.is_empty() {
                            "the package".to_string()
                        } else {
                            format!("`{}`", key.join("."))
                        },
                        a.name
                    )
                    .into());
                }
                if module_binding(options, key, &a.name).is_some()
                    && module_deletes(options, key, &a.name)
                {
                    return Err(format!(
                        "`from {}{} import {}` is refused: the package binds `{}` and \
                         deletes it (`del {}`), so whether the import finds the attribute, \
                         a submodule of that name, or nothing depends on the order at \
                         runtime, which the converted program cannot represent; import \
                         the value under a name the package keeps, or drop the `del`",
                        ".".repeat(i.level),
                        i.module,
                        a.name,
                        a.name,
                        a.name
                    )
                    .into());
                }
            }
        }
    }
    let calls = module_init_calls(&imported_crate_modules(stmt, options));
    let checks = import_site_name_checks(stmt, options);
    Ok(quote!(#calls #checks))
}

/// The import site of a folded `try: <imports> except ImportError:` (or
/// bare `except:`) guard. The guard folds because rython's imports are
/// static — but a crate module's body can raise at runtime (an
/// ImportError from a cycle asking for a name bound later or a module
/// that stays failed; anything at all from its own statements), where
/// Python would run the fallback the fold discarded. That case is loud:
/// the exceptions the handler would have caught — ImportError for a
/// typed guard, every exception for a bare one — leave the site as an
/// ImportError naming the guard, the folded fallback and the original
/// error, instead of a bare error (Devin review on #338, rounds 10 and
/// 19). What the handler would not have caught propagates as in Python.
pub(crate) fn folded_guard_site(
    site: TokenStream,
    spelling: &str,
    guard: crate::ast::tree::module::FoldedGuard,
) -> TokenStream {
    use crate::ast::tree::module::FoldedGuard;
    if site.is_empty() {
        return site;
    }
    let (raised, handler) = match guard {
        FoldedGuard::ImportError => ("ImportError", "`except ImportError:`"),
        FoldedGuard::Bare => ("an exception", "bare `except:`"),
    };
    let message = format!(
        "`{}` raised {} at import time; its {} fallback was folded away (rython's \
         imports are static), so the program cannot run the fallback as Python would",
        spelling, raised, handler
    );
    let caught = match guard {
        FoldedGuard::ImportError => quote!(if __rython_import_error.matches("ImportError")),
        FoldedGuard::Bare => quote!(),
    };
    quote! {
        match (|| -> Result<(), PyException> { #site Ok(()) })() {
            Ok(()) => {}
            Err(__rython_import_error) #caught => {
                return Err(PyException::new(
                    "ImportError",
                    format!("{}: {}", #message, __rython_import_error),
                ));
            }
            #[allow(unreachable_patterns)]
            Err(__rython_import_error) => return Err(__rython_import_error),
        }
    }
}

impl CodeGen for Import {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        for alias in self.names.iter() {
            // `import a.b.c` binds the ROOT name `a` in Python (a later
            // statement can reference `a.b`). Register the root too, so
            // module-chain resolution works for submodule attribute calls
            // (`import h2.config` — urllib3's http2: `h2.config.
            // H2Configuration(...)`).
            if let Some(root) = alias.name.split('.').next() {
                if !root.is_empty() && root != &alias.name {
                    symbols.insert(root.to_string(), SymbolTableNode::Import(self.clone()));
                }
            }
            symbols.insert(alias.name.clone(), SymbolTableNode::Import(self.clone()));
            if let Some(a) = alias.asname.clone() {
                symbols.insert(a, SymbolTableNode::Alias(alias.name.clone()))
            }
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        mut symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        let mut tokens = TokenStream::new();
        for alias in self.names.iter() {
            // `import rython` is not a runtime module: rust.bind declarations
            // are compile-time-only. The from-import spelling is required so
            // the declaration syntax stays explicit.
            if alias.name == "rython" || alias.name.starts_with("rython.") {
                return Err("`import rython` is not supported; use \
                            `from rython import rust` for compile-time Rust bindings"
                    .to_string()
                    .into());
            }
            // Import of a Rust module: the import is compile-time-only — the
            // name resolves through the symbol table; no `use` is emitted
            // (the crate is a dependency, not a sibling module). Registering
            // the module symbol lets attribute call lowering find it.
            if let Some(spec) = options.rust_modules.get(&alias.name) {
                let module_symbol = crate::SymbolTableNode::RustModule(spec.clone());
                symbols.insert(alias.name.clone(), module_symbol);
                if let Some(asname) = &alias.asname {
                    symbols.insert(
                        asname.clone(),
                        crate::SymbolTableNode::Alias(alias.name.clone()),
                    );
                }
                tokens.extend(TokenStream::new());
                continue;
            }
            // Import of a vendored Python module (`[python-modules]` in
            // rython.toml): it's a sibling module of the generated crate.
            // Attribute calls lower to `crate::textlib::fn` paths, so the
            // plain spelling emits no `use` — a `use crate::textlib;` in
            // the binary would collide with the `mod textlib;` declaration
            // that brings the sibling module into the bin crate. The
            // aliased spelling still needs the import to bind the alias
            // (`use crate::textlib as t;` — a different name, no clash).
            {
                let root = alias.name.split('.').next().unwrap_or(&alias.name);
                if options.python_modules.contains(root) {
                    if let Some(asname) = &alias.asname {
                        let names = if alias.name.contains('.') {
                            let parts: Vec<&str> = alias.name.split('.').collect();
                            let idents: Vec<_> =
                                parts.iter().map(|part| crate::safe_ident(part)).collect();
                            quote!(#(#idents)::*)
                        } else {
                            let single_name = crate::safe_ident(&alias.name);
                            quote!(#single_name)
                        };
                        let name = crate::safe_ident(asname);
                        tokens.extend(quote! {use crate::#names as #name;});
                    }
                    continue;
                }
            }
            if options.no_std {
                let root = alias.name.split('.').next().unwrap_or(&alias.name);
                if is_std_only_module(root) {
                    return Err(std_only_import_error(&alias.name));
                }
            }
            // `import typing` (an annotation-only module): nothing at
            // runtime — its names are read by the annotation authorities
            // through the `typing.X[...]` spelling (the from-import form
            // already emits nothing for them). A `use crate::typing;`
            // named a module the crate does not have (issue #335).
            {
                let root = alias.name.split('.').next().unwrap_or(&alias.name);
                if crate::AnnotationModule::from_name(root).is_some() {
                    continue;
                }
            }
            // Check if this is a Python standard library module that needs special handling
            let rust_import = match alias.name.as_str() {
                // `import numpy as np` (and `import numpy.linalg as np`) is
                // THE canonical numpy spelling. numpy IS a path under the
                // runtime crate (stdpython::numpy), so the alias resolves
                // as a proper `use` — unlike the glob-provided modules
                // below. The alias import also makes `np.linalg.inv(...)`
                // work through the nested path.
                "numpy" | "numpy.linalg" => {
                    let runtime = crate::safe_ident(&options.stdpython);
                    match &alias.asname {
                        None => {
                            if crate::StdModule::from_name(&alias.name)
                                == Some(crate::StdModule::Numpy)
                            {
                                // `import numpy` — the name comes from the
                                // `use stdpython::*` glob re-export.
                                quote! {}
                            } else {
                                quote! {
                                    use #runtime::numpy;
                                }
                            }
                        }
                        Some(asname) => {
                            let asname = crate::safe_ident(asname);
                            quote! {
                                use #runtime::numpy as #asname;
                            }
                        }
                    }
                }
                // Runtime-provided modules are already in scope through
                // `use stdpython::*` (each is re-exported at the crate
                // root), so the import lowers to nothing — a bare
                // `use math;` would not even resolve. An ALIASED import
                // (`import time as t`, `import json as _json`) binds the
                // alias as a real path (`use stdpython::time as t;`) so
                // `t::monotonic()` / `_json::loads()` resolve — the same
                // spelling numpy's alias arm uses.
                name if is_stdpython_module(name) => {
                    if let Some(asname) = &alias.asname {
                        let runtime = crate::safe_ident(&options.stdpython);
                        let module = crate::safe_ident(name);
                        let asname = crate::safe_ident(asname);
                        quote! {
                            use #runtime::#module as #asname;
                        }
                    } else {
                        quote! {}
                    }
                }
                // `import urllib.request` — a dotted stdpython submodule
                // (the numpy.linalg model). Unaliased, the chain resolves
                // through the glob-re-exported `urllib` module; aliased, it
                // binds a real path.
                "urllib.request" => {
                    let runtime = crate::safe_ident(&options.stdpython);
                    match &alias.asname {
                        None => quote! {},
                        Some(asname) => {
                            let asname = crate::safe_ident(asname);
                            quote! {
                                use #runtime::urllib::request as #asname;
                            }
                        }
                    }
                }
                // Python stdlib modules that don't have direct Rust equivalents
                "xml" => {
                    // These will be provided by the stdpython runtime
                    // Generate a comment instead of a use statement
                    quote! {
                        // Python module '{}' will be provided by stdpython runtime
                    }
                }
                "os.path" => {
                    quote! {
                        // Python os.path module will be provided by stdpython runtime
                    }
                }
                _ => {
                    // A sibling module of the generated crate resolves via
                    // `use crate::...`; an EXTERNAL module (stdlib rython
                    // does not model — ssl, socket, logging, http, typing,
                    // codecs, types, ... — or a third-party dep that is not
                    // vendored) has no generated item, so the import lowers
                    // to nothing with a warning (documented divergence:
                    // the module's runtime functionality is unavailable;
                    // uses of its names become loud errors or boxed drops).
                    let path: Vec<String> =
                        alias.name.split('.').map(|s| s.to_string()).collect();
                    // The crate path may differ from the dotted Python name:
                    // a root-qualified absolute self-import (`import
                    // urllib3.connection` inside the urllib3 conversion)
                    // resolves under the STRIPPED key, and rendering the
                    // full path would emit `use crate::urllib3::connection;`
                    // — a module the crate doesn't contain.
                    let crate_path: Vec<String> =
                        match crate::module_defs_key(&options, &path) {
                            Some(key) => key.to_vec(),
                            None => path.clone(),
                        };
                    let is_sibling = crate::module_defs_contains(&options, &path)
                        || options.python_modules.contains(
                            &path.first().cloned().unwrap_or_default(),
                        )
                        // Single-module conversions only know the module
                        // itself (module_defs.len() == 1): assume any other
                        // non-stdpython import is a crate sibling (the
                        // module_defs check is authoritative only when the
                        // whole crate is known).
                        || options.module_defs.len() <= 1;
                    if !is_sibling {
                        options.definition_warnings.borrow_mut().push(format!(
                            "import `{}` is dropped: the module is not part of the \
                             generated crate nor the stdpython runtime \
                             (external-module divergence)",
                            alias.name
                        ));
                        quote! {}
                    } else {
                        let idents: Vec<_> = crate_path
                            .iter()
                            .map(|part| crate::safe_ident(part))
                            .collect();
                        let names = quote!(#(#idents)::*);

                        match &alias.asname {
                            // An unaliased dotted import binds only the ROOT
                            // name in Python; when that root is the package
                            // itself (the stripped-key resolution), the
                            // "bound module" is the crate — a leaf `use`
                            // would bind a name Python doesn't (`import
                            // urllib3.connection` clashing with emscripten's
                            // own `connection` submodule), so nothing is
                            // emitted.
                            None if crate_path.len() < path.len() => quote! {},
                            None => {
                                quote! {use crate::#names;}
                            }
                            Some(n) => {
                                let name = crate::safe_ident(n);
                                quote! {use crate::#names as #name;}
                            }
                        }
                    }
                }
            };

            tokens.extend(rust_import);
        }
        debug!("context: {:?}", ctx);
        debug!("options: {:?}", options);
        debug!("tokens: {}", tokens);
        Ok(tokens)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportFrom {
    /// The dotted module being imported FROM. `from . import x` and
    /// `from . import` (relative imports with no module part) have
    /// module = None in Python's AST — extracted as "" here so the
    /// resolved path is just the current package.
    pub module: String,
    pub names: Vec<Alias>,
    pub level: usize,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ImportFrom {
    type Error = pyo3::PyErr;
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let module: Option<String> = ob
            .getattr("module")
            .map_err(|e| crate::extraction_failure("ImportFrom module", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom module", &ob, e))?;
        let names: Vec<Alias> = ob
            .getattr("names")
            .map_err(|e| crate::extraction_failure("ImportFrom names", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom names", &ob, e))?;
        let level: usize = ob
            .getattr("level")
            .map_err(|e| crate::extraction_failure("ImportFrom level", &ob, e))?
            .extract()
            .map_err(|e| crate::extraction_failure("ImportFrom level", &ob, e))?;
        Ok(ImportFrom {
            module: module.unwrap_or_default(),
            names,
            level,
        })
    }
}

impl ImportFrom {
    /// The module path this import resolves to inside the generated crate:
    /// for a relative import, the current module path (cut by `level`) plus
    /// the dotted module; for an absolute import, the dotted module itself.
    /// Key into `options.module_defs` to reach the defining module's AST.
    pub(crate) fn resolved_module_path(&self, options: &PythonOptions) -> Vec<String> {
        let parts: Vec<&str> = self.module.split('.').filter(|p| !p.is_empty()).collect();
        if self.level > 0 {
            let cur = &options.module_path;
            // A relative import with more leading dots than the current
            // package depth reaches above the crate root; saturate so the
            // caller gets an empty (or root-level) prefix instead of a
            // usize underflow panic. `ImportFrom::to_rust` reports the
            // clean "reaches above the crate root" error for the user.
            let cut = (cur.len() + 1).saturating_sub(self.level);
            cur[..cut]
                .iter()
                .map(|s| s.as_str())
                .chain(parts.iter().copied())
                .map(|s| s.to_string())
                .collect()
        } else {
            parts.iter().map(|s| s.to_string()).collect()
        }
    }
}

impl CodeGen for ImportFrom {
    type Context = CodeGenContext;
    type Options = PythonOptions;
    type SymbolTable = SymbolTableScopes;

    fn find_symbols(self, symbols: Self::SymbolTable) -> Self::SymbolTable {
        let mut symbols = symbols;
        for alias in self.names.iter() {
            symbols.insert(
                alias.name.clone(),
                SymbolTableNode::ImportFrom(self.clone()),
            );
            // `from pylev import wf as w`: the alias resolves to the
            // canonical name so call lowering propagates exceptions and
            // attribute access treats it as the imported value. A SELF-alias
            // (`from ._base_connection import ProxyConfig as ProxyConfig` —
            // urllib3's re-export) must NOT overwrite the ImportFrom symbol:
            // resolve_imported_class follows the chain through ImportFrom,
            // and an Alias-to-self would loop.
            if let Some(asname) = &alias.asname {
                if asname != &alias.name {
                    symbols.insert(asname.clone(), SymbolTableNode::Alias(alias.name.clone()));
                }
            }
        }
        symbols
    }

    fn to_rust(
        self,
        ctx: Self::Context,
        options: Self::Options,
        mut symbols: Self::SymbolTable,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        debug!("ctx: {:?}", ctx);
        // annotations map to Rust types directly, so the import itself
        // lowers to nothing.
        if crate::AnnotationModule::from_name(self.module.split('.').next().unwrap_or(""))
            == Some(crate::AnnotationModule::Typing)
        {
            return Ok(TokenStream::new());
        }

        // `from rython import rust` is compile-time-only: rust.bind
        // declarations lower to nothing and the module never exists at
        // runtime. Only `rust` is importable — anything else is a mistake
        // worth a loud error, not a silent no-op.
        // `from __future__ import ...` is a compiler directive, not a
        // runtime import: the future flags (annotations, generators, ...)
        // are either already the language's default behavior or have no
        // Rust analogue, so the statement lowers to nothing (a `use
        // crate::__future__::...` would be an unresolved import).
        if self.module == "__future__" {
            return Ok(TokenStream::new());
        }

        // `from dataclasses import dataclass` (and field, ...): the
        // decorator is CONSUMED at conversion time by the class codegen
        // (synthesized __init__), so the import is a no-op — a `use
        // crate::dataclasses::...` would be an unresolved import. Other
        // dataclasses names are the same: nothing from the module exists
        // at runtime in the generated crate.
        if crate::AnnotationModule::from_name(&self.module)
            == Some(crate::AnnotationModule::Dataclasses)
        {
            return Ok(TokenStream::new());
        }

        // `from rython import rust` — compile-time Rust bindings.
        if self.module == "rython" {
            if self.names.len() == 1
                && self.names[0].name == "rust"
                && self.names[0].asname.is_none()
            {
                return Ok(TokenStream::new());
            }
            return Err(
                "only `from rython import rust` is supported (compile-time Rust \
                 bindings); aliasing or importing other names does not exist"
                    .to_string()
                    .into(),
            );
        }

        // `from <rust-module> import <fn> [as <alias>]`: the functions are
        // compile-time bindings into a Rust crate; the import lowers to
        // nothing and registers each name in the symbol table so call
        // lowering resolves them. An unknown name is a loud error (the
        // stub/inferred signature is the source of truth).
        if let Some(spec) = options.rust_modules.get(&self.module) {
            for alias in self.names.iter() {
                if alias.name == "*" {
                    return Err(format!(
                        "`from {} import *`: wildcard imports of Rust modules are \
                         not supported; import the names explicitly",
                        self.module
                    )
                    .into());
                }
                let bind_name = alias.asname.clone().unwrap_or_else(|| alias.name.clone());
                match spec.get_fn(&alias.name) {
                    Some(fspec) => {
                        let mut spec_for_binding = spec.clone();
                        spec_for_binding.fns = vec![fspec.clone()];
                        symbols.insert(
                            bind_name,
                            crate::SymbolTableNode::RustModule(spec_for_binding),
                        );
                    }
                    None => {
                        return Err(format!(
                            "`from {} import {}`: `{}` is not a bound function of \
                             crate `{}` (bound: {})",
                            self.module,
                            alias.name,
                            alias.name,
                            spec.crate_name,
                            spec.fns
                                .iter()
                                .map(|f| f.fn_name.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                        .into());
                    }
                }
            }
            return Ok(TokenStream::new());
        }

        if options.no_std {
            let root = self.module.split('.').next().unwrap_or(&self.module);
            if is_std_only_module(root) {
                return Err(std_only_import_error(&self.module));
            }
        }

        // `from X import y` must bring `y` into scope; previously this
        // emitted nothing and later uses of `y` were undefined. `use` paths
        // can't resolve through glob imports, so anchor the path explicitly:
        // stdlib modules live under the stdpython runtime crate, and
        // anything else is assumed to be a sibling module of the generated
        // crate. Wildcard imports map to glob uses.
        let parts: Vec<&str> = self
            .module
            .split('.')
            .filter(|part| !part.is_empty())
            .collect();
        // Relative imports (`from .x import y`, `from ..x import y`, level
        // > 0) resolve against the CURRENT module's package path
        // (`options.module_path`, set by the converter; empty at the crate
        // root). level 1 is this package, level 2 its parent, and so on;
        // the resolved path always stays inside the generated crate, since
        // the package and all its submodules are compiled into it.
        let base_parts: Vec<_> = if self.level > 0 {
            let cur = &options.module_path;
            if self.level > cur.len() + 1 {
                return Err(format!(
                    "`from {} import ...` (level {}): relative import goes above \
                     the crate root",
                    self.module, self.level
                )
                .into());
            }
            let cut = cur.len() + 1 - self.level;
            cur[..cut].iter().map(|p| crate::safe_ident(p)).collect()
        } else {
            Vec::new()
        };
        // For an ABSOLUTE import of a crate module, the generated `use`
        // path must match the crate's mod tree, which is keyed RELATIVE to
        // the package root for src-layout sdists (pip, boto3 —
        // `pip._internal.cli.req_command` lives at
        // `_internal/cli/req_command.rs`). `module_defs_key` returns that
        // relative key; external modules fall back to the literal segments
        // (their imports are dropped before a use is emitted).
        let module_path: Vec<_> = if self.level == 0 {
            let resolved = self.resolved_module_path(&options);
            match crate::module_defs_key(&options, &resolved) {
                Some(key) => key.iter().map(|p| crate::safe_ident(p)).collect(),
                None => parts.iter().map(|part| crate::safe_ident(part)).collect(),
            }
        } else {
            parts.iter().map(|part| crate::safe_ident(part)).collect()
        };
        let root = if self.level > 0 {
            quote!(crate)
        } else if parts
            .first()
            .is_some_and(|first| is_stdpython_module(first))
        {
            let runtime = crate::safe_ident(&options.stdpython);
            quote!(#runtime)
        } else {
            quote!(crate)
        };

        // An EXTERNAL module (stdlib rython does not model — logging, ssl,
        // socket, http, codecs, types, importlib, ... — or a non-vendored
        // dependency) has no generated items: the import lowers to nothing
        // with a warning (documented divergence). Its names still resolve in
        // the symbol table, so annotations map them to the boxed PyValue and
        // runtime calls drop. `collections.abc` is the typing-abstraction
        // submodule (Mapping, Iterable, ...): also compile-time-only.
        // Relative imports always target sibling modules of the crate.
        let resolved_path = self.resolved_module_path(&options);
        let first_part = parts.first().copied();
        // The external check is only meaningful when the whole crate is
        // known (multi-module conversions populate module_defs): a
        // single-module conversion (len == 1) only knows the module itself,
        // so any absolute non-stdpython import is assumed to be a crate
        // sibling (`from helpers import util`).
        let external = self.level == 0
            && options.module_defs.len() > 1
            && !matches!(first_part, Some(p) if is_stdpython_module(p))
            && !crate::module_defs_contains(&options, &resolved_path)
            && !options
                .python_modules
                .contains(&first_part.unwrap_or("").to_string());
        if external || self.module == "collections.abc" {            options.definition_warnings.borrow_mut().push(format!(
                "`from {} import ...` is dropped: the module is not part of the \
                 generated crate nor the stdpython runtime \
                 (external-module divergence)",
                self.module
            ));
            return Ok(TokenStream::new());
        }

        // A STDPYTHON-module import with SOME names lacking a runtime
        // counterpart (`from io import BytesIO, IOBase` — urllib3's
        // emscripten response, where BytesIO exists in stdpython but
        // IOBase does not; `from os import PathLike` — charset_normalizer's
        // api.py): a use for a missing item would fail E0432. Drop ONLY
        // the missing names (annotation-only ones map to the boxed
        // PyValue); the present ones still emit their use.
        let first_part = parts.first().copied().unwrap_or("");
        if is_stdpython_module(first_part)
            && self.names.iter().any(|a| !stdpython_module_item(first_part, &a.name))
        {
            let present: Vec<crate::Alias> = self
                .names
                .iter()
                .filter(|a| stdpython_module_item(first_part, &a.name))
                .cloned()
                .collect();
            for alias in &self.names {
                if !stdpython_module_item(first_part, &alias.name) {
                    options.definition_warnings.borrow_mut().push(format!(
                        "`from {} import {}` is dropped: stdpython has no runtime \
                         item for `{}` (annotation-only names map to the boxed \
                         PyValue)",
                        self.module, alias.name, alias.name
                    ));
                }
            }
            if present.is_empty() {
                return Ok(TokenStream::new());
            }
            // Re-emit the import with only the present names — INCLUDING
            // each name's runtime-fn variants (the arity-split
            // `BytesIO_seeded` etc.), exactly as the plain path below
            // brings them along; the mixed `from io import BytesIO,
            // IOBase` previously dropped the variants and every seeded
            // call site failed E0425 (issue #137).
            let mut present_tokens = TokenStream::new();
            for alias in &present {
                let name = crate::safe_ident(&alias.name);
                let variants: &[&str] = crate::StdModule::from_name(&self.module)
                    .map(|m| {
                        crate::ast::tree::std_module::runtime_fn_variants(m, &alias.name)
                    })
                    .unwrap_or(&[]);
                // `pub use`, matching the plain stdpython path below: the
                // imported name is a module attribute a sibling's
                // re-export chain may traverse (E0603 otherwise).
                let import = match &alias.asname {
                    Some(asname) => {
                        let asname = crate::safe_ident(asname);
                        quote! { pub use #root #(::#base_parts)* #(::#module_path)*::#name as #asname; }
                    }
                    None if variants.is_empty() => {
                        quote! { pub use #root #(::#base_parts)* #(::#module_path)*::#name; }
                    }
                    None => quote! {
                        #[allow(unused_imports)]
                        pub use #root #(::#base_parts)* #(::#module_path)*::#name;
                    },
                };
                present_tokens.extend(import);
                for variant in variants {
                    let v = crate::safe_ident(variant);
                    present_tokens.extend(quote! {
                        #[allow(unused_imports)]
                        use #root #(::#base_parts)* #(::#module_path)*::#v;
                    });
                }
            }
            return Ok(present_tokens);
        }

        let mut tokens = TokenStream::new();
        // Deduplicate trait imports across the aliases of ONE ImportFrom:
        // `from .connectionpool import HTTPConnectionPool,
        // HTTPSConnectionPool` — both classes share the ancestor trait
        // `ConnectionPoolTrait`, so the bring-along would emit it twice
        // (E0252 — duplicate import).
        let mut seen_traits: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for alias in self.names.iter() {
            // functools.partial/lru_cache/cache/singledispatch have no
            // runtime symbols: partial lowers to a closure at each call
            // site, and the cache and singledispatch decorators rewrite
            // the function definitions themselves, so the imports emit
            // nothing (an uncalled bare reference is then a loud
            // unresolved-name error).
            if crate::StdModule::from_name(&self.module) == Some(crate::StdModule::Functools)
                && matches!(
                    alias.name.as_str(),
                    "partial" | "lru_cache" | "cache" | "singledispatch"
                )
            {
                continue;
            }
            // A TYPE-NAME TUPLE alias (`basestring = (str, bytes)` —
            // requests' compat): consumed by isinstance resolution at
            // conversion time, never a runtime value — the import emits
            // nothing (a `pub use crate::...::basestring` would fail: the
            // value is a module-init local, not a static).
            if is_type_name_tuple_alias(&alias.name, &options, &symbols) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` is a type-name tuple alias \
                     (typing-only; consumed by isinstance resolution)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // A sibling re-export of a BUILTIN exception name
            // (`BrokenPipeError = BrokenPipeError` — connection.py's
            // py2-compat shim, imported by connectionpool): builtins are
            // string-tagged with no runtime item, and raise/except match
            // by name — the use drops.
            if self.level > 0
                && crate::ast::tree::raise_stmt::is_builtin_exception_name(&alias.name)
            {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` is a builtin exception \
                     (string-tagged; no runtime item — raise/except match \
                     by name)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // A name the SIBLING module binds as a stdlib EXCEPTION ALIAS
            // (`BaseSSLError = ssl.SSLError` — urllib3's connection.py):
            // no runtime item exists (the alias emits nothing), so the
            // use would fail E0432. Drop it; raise/except guards
            // canonicalize through imported_exception_alias, which
            // follows the chain into the defining module.
            if crate::ast::tree::module::module_def_exception_alias(
                &options,
                &resolved_path,
                &alias.name,
            )
            .is_some()
            {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` is a stdlib exception alias \
                     (no runtime item; except/raise sites canonicalize)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // A name that RE-EXPORTS from an EXTERNAL module (`from
            // urllib.parse import urlparse` in requests' compat — urllib is
            // external): no runtime item exists behind the chain, so the
            // use drops (calls through the name lower to the boxed None).
            if resolves_to_external_import(&alias.name, &options, &symbols) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}`: `{}` re-exports from an external module \
                     (no runtime item; the import is dropped)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // A name the module ALSO defines LOCALLY (`from .compat import
            // proxy_bypass` then a later `def proxy_bypass` — requests'
            // utils.py): Python's LAST binding wins, so the local
            // definition overrides the import — the import is dead and
            // must not emit a `use` for a name the sibling never exports.
            // (find_symbols keeps the LAST binding, so a FunctionDef /
            // ClassDef / Assign symbol here means the local def won.)
            // The check tests the name the import actually BINDS — the
            // asname when aliased (`from .util.url import _normalize_host
            // as normalize_host` — urllib3's connectionpool.py, where the
            // LOCAL `def _normalize_host` overrides the unaliased spelling
            // while the aliased `normalize_host` still resolves to the
            // imported function). A local def of the CANONICAL name does
            // not shadow the alias.
            let bound_name = alias.asname.as_deref().unwrap_or(&alias.name);
            if matches!(
                symbols.get(bound_name),
                Some(
                    crate::SymbolTableNode::FunctionDef(_)
                        | crate::SymbolTableNode::ClassDef(_)
                        | crate::SymbolTableNode::Assign { .. }
                )
            ) {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}` is dropped: the module defines `{}` \
                     locally, and Python's later definition wins",
                    self.module, alias.name, bound_name
                ));
                continue;
            }
            if alias.name == "*" {
                let visibility = if self.level > 0 { quote!(pub) } else { quote!() };
                // A crate source with a literal `__all__` exports THOSE
                // names, not every public item: the glob would re-export
                // a name `__all__` leaves out, and a sibling's `from pkg
                // import name` would take it where Python imports the
                // submodule of that name (Devin review on #338, round 20).
                let source = self.resolved_module_path(&options);
                let listed = crate::module_defs_key(&options, &source)
                    .and_then(|key| star_reexport_list(&options, key));
                if let Some(names) = listed {
                    if !names.is_empty() {
                        let idents: Vec<_> = names.iter().map(|n| crate::safe_ident(n)).collect();
                        tokens.extend(quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::{#(#idents),*}; });
                    }
                    continue;
                }
                tokens.extend(quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::*; });
                continue;
            }
            // Some runtime functions split into arity/keyword-specific
            // variants (accumulate with initial=, product with repeat=,
            // ...); importing the Python name brings its variants along so
            // the call lowering can pick one. For these names, the BASE
            // import is allow(unused_imports) too: the lowering may rewrite
            // every call site to a variant (accumulate/product always are),
            // orphaning the bare name through no fault of the source
            // Python. Names without variants keep the plain import, so a
            // genuinely unused `from itertools import pairwise` still
            // surfaces as the source weakness it is.
            let variants: &[&str] = crate::StdModule::from_name(&self.module)
                .map(|m| crate::ast::tree::std_module::runtime_fn_variants(m, &alias.name))
                .unwrap_or(&[]);

            let name = crate::safe_ident(&alias.name);
            // A name the defining module re-exports from a STDPYTHON
            // module (`from .compat import json as complexjson` where
            // compat.py does `import json` — requests' models.py): the
            // generated compat.rs has no `json` item (stdlib modules
            // resolve through the runtime), so the import must route to
            // the runtime module directly (`use <stdpython>::json as
            // complexjson;`) — a `use crate::requests::compat::json`
            // would fail E0432.
            if let Some(runtime_module) =
                crate::ast::tree::module::module_reexports_stdpython_module(
                    &options,
                    &self.resolved_module_path(&options),
                    &alias.name,
                )
            {
                let runtime = crate::safe_ident(&options.stdpython);
                let module = crate::safe_ident(&runtime_module);
                let asname = crate::safe_ident(alias.asname.as_deref().unwrap_or(&alias.name));
                tokens.extend(quote! {
                    use #runtime::#module as #asname;
                });
                continue;
            }
            // A sibling-module import whose defining module was NOT
            // generated (`from urllib3.contrib import pyopenssl` — the
            // contrib/pyopenssl.py module fails conversion, so no
            // pyopenssl.rs exists; requests' __init__.py imports it inside
            // a dead try): the use would fail E0432. Drop it — the module
            // has no runtime item.
            let import_module_path = self.resolved_module_path(&options);
            if options.module_defs.len() > 1
                && let Some(key) = crate::module_defs_key(&options, &import_module_path)
                && !crate::ast::tree::module::module_def_has_runtime_item(
                    &options,
                    key,
                    &alias.name,
                )
            {
                options.definition_warnings.borrow_mut().push(format!(
                    "`from {} import {}` is dropped: the defining module has no \
                     generated runtime item for `{}` (the module may have failed \
                     conversion)",
                    self.module, alias.name, alias.name
                ));
                continue;
            }
            // Relative imports re-export from a sibling module: Python
            // treats imported names as module attributes (the package
            // `__init__.py` re-export pattern), so they lower to `pub use`
            // — callers reach `textlib.double` through the re-export chain.
            // Absolute imports of user modules stay plain `use`: callers
            // import from the defining module directly.
            // An underscore-prefixed sibling ITEM is `pub(crate)` in the
            // defining module (`_wrap_proxy_error` — urllib3's
            // connection.py): the re-export must match, or Rust rejects a
            // `pub use` of a crate-only item (E0364).
            // A stdpython from-import is also `pub use`: Python treats
            // imported names as module attributes, so a sibling's
            // re-export chain (`from .util.ssl_ import SSLContext` where
            // ssl_.py did `from ssl import SSLContext` — urllib3) must
            // find a public item, not a private use (E0603).
            let stdpython_root = self.level == 0
                && parts.first().is_some_and(|p| is_stdpython_module(p));
            let visibility = if self.level > 0 && alias.name.starts_with("_") {
                quote!(pub(crate))
            } else if self.level > 0 || stdpython_root {
                quote!(pub)
            } else {
                quote!()
            };
            let import = match &alias.asname {
                None if variants.is_empty() => {
                    quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::#name; }
                }
                None => quote! {
                    #[allow(unused_imports)]
                    #visibility use #root #(::#base_parts)* #(::#module_path)*::#name;
                },
                Some(asname) => {
                    let asname = crate::safe_ident(asname);
                    quote! { #visibility use #root #(::#base_parts)* #(::#module_path)*::#name as #asname; }
                }
            };
            // A SELF-referential import (`from . import packages, utils`
            // inside requests/__init__.py — the resolved module path IS
            // the current module): the names are the package's OWN
            // submodules, already declared by `pub mod`; the emitted
            // `pub use crate::requests::packages;` would re-import the
            // sibling into itself (E0255 — defined multiple times).
            let self_resolved = self.level > 0
                && options.this_module_path == self.resolved_module_path(&options);
            if !self_resolved {
                tokens.extend(import);
            }
            if !self_resolved {
                for variant in variants {
                    let v = crate::safe_ident(variant);
                    tokens.extend(quote! {
                        #[allow(unused_imports)]
                        use #root #(::#base_parts)* #(::#module_path)*::#v;
                    });
                }
            }

            // A hierarchy class imported from another module of the
            // generated crate carries its methods on traits (`{Name}Trait`
            // plus ancestors'), NOT on the struct — Rust method resolution
            // needs those traits IN SCOPE at the call site, so the import
            // brings them along: `from .animals import Dog` also imports
            // `AnimalTrait`, and `d.get()` resolves. Only classes that
            // lower with the trait machinery have traits; functions and
            // plain structs get none (the per-module map is empty for
            // them).
            let import_module_path = self.resolved_module_path(&options);
            // A polymorphic ROOT's slot type is its sum type, defined in
            // the root's module (hierarchy.rs): import it alongside.
            if options.hierarchy_roots.contains_key(&alias.name) {
                let any = crate::ast::tree::hierarchy::any_ident(&alias.name);
                tokens.extend(quote! {
                    #[allow(unused_imports)]
                    use #root #(::#base_parts)* #(::#module_path)*::#any;
                });
            }
            if let Some(key) = crate::module_defs_key(&options, &import_module_path)
                && let Some(traits) = crate::module_class_traits(&options, key).get(&alias.name)
            {
                for trait_name in traits {
                    if !seen_traits.insert(trait_name.clone()) {
                        continue;
                    }
                    let t = crate::safe_ident(trait_name);
                    tokens.extend(quote! {
                        #[allow(unused_imports)]
                        use #root #(::#base_parts)* #(::#module_path)*::#t;
                    });
                }
            }
        }
        Ok(tokens)
    }
}
