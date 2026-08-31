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
