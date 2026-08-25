//! The stdpython runtime modules the compiler knows, as a typed enum.
//!
//! The AGENTS.md boundary rule: AST identifiers arrive as strings, so ONE
//! string comparison is unavoidable — it happens exactly once, in
//! [`StdModule::from_name`]. Every property of a module (its no_std tier,
//! whether its functions route through the borrow/arity dispatcher, its
//! canonical path name) is a method on the enum, so the sets can never
//! drift apart the way parallel string lists can. The
//! [`ThreadingType`](super::threading_types::ThreadingType) enum plays the
//! same role for the threading module's types.

/// A module of the stdpython runtime crate. Imports of these resolve
/// under the runtime crate; anything else is assumed to be a sibling
/// module of the generated crate (or external).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StdModule {
    Os,
    Sys,
    Re,
    Io,
    Argparse,
    Json,
    Math,
    Random,
    Datetime,
    Time,
    Collections,
    Itertools,
    Functools,
    Heapq,
    Copy,
    Textwrap,
    Hashlib,
    Csv,
    Glob,
    Pathlib,
    Tempfile,
    Subprocess,
    String,
    Sysconfig,
    Venv,
    Warnings,
    Numpy,
    /// asyncio lives on the tokio-backed `async-tokio` stdpython feature;
    /// generated async binaries enable it.
    Asyncio,
    Threading,
    Socket,
    /// urllib.request lives on the ureq-backed `http-ureq` stdpython
    /// feature; rypip enables it when a package imports it.
    Urllib,
}

impl StdModule {
    /// Parse a module name at the AST boundary — the ONE place the name
    /// set exists as strings.
    pub(crate) fn from_name(name: &str) -> Option<StdModule> {
        Some(match name {
            "os" => StdModule::Os,
            "sys" => StdModule::Sys,
            "re" => StdModule::Re,
            "io" => StdModule::Io,
            "argparse" => StdModule::Argparse,
            "json" => StdModule::Json,
            "math" => StdModule::Math,
            "random" => StdModule::Random,
            "datetime" => StdModule::Datetime,
            "time" => StdModule::Time,
            "collections" => StdModule::Collections,
            "itertools" => StdModule::Itertools,
            "functools" => StdModule::Functools,
            "heapq" => StdModule::Heapq,
            "copy" => StdModule::Copy,
            "textwrap" => StdModule::Textwrap,
            "hashlib" => StdModule::Hashlib,
            "csv" => StdModule::Csv,
            "glob" => StdModule::Glob,
            "pathlib" => StdModule::Pathlib,
            "tempfile" => StdModule::Tempfile,
            "subprocess" => StdModule::Subprocess,
            "string" => StdModule::String,
            "sysconfig" => StdModule::Sysconfig,
            "venv" => StdModule::Venv,
            "warnings" => StdModule::Warnings,
            "numpy" => StdModule::Numpy,
            "asyncio" => StdModule::Asyncio,
            "threading" => StdModule::Threading,
            "socket" => StdModule::Socket,
            "urllib" => StdModule::Urllib,
            _ => return None,
        })
    }

    /// The module's Python (and generated Rust path) name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            StdModule::Os => "os",
            StdModule::Sys => "sys",
            StdModule::Re => "re",
            StdModule::Io => "io",
            StdModule::Argparse => "argparse",
            StdModule::Json => "json",
            StdModule::Math => "math",
            StdModule::Random => "random",
            StdModule::Datetime => "datetime",
            StdModule::Time => "time",
            StdModule::Collections => "collections",
            StdModule::Itertools => "itertools",
            StdModule::Functools => "functools",
            StdModule::Heapq => "heapq",
            StdModule::Copy => "copy",
            StdModule::Textwrap => "textwrap",
            StdModule::Hashlib => "hashlib",
            StdModule::Csv => "csv",
            StdModule::Glob => "glob",
            StdModule::Pathlib => "pathlib",
            StdModule::Tempfile => "tempfile",
            StdModule::Subprocess => "subprocess",
            StdModule::String => "string",
            StdModule::Sysconfig => "sysconfig",
            StdModule::Venv => "venv",
            StdModule::Warnings => "warnings",
            StdModule::Numpy => "numpy",
            StdModule::Asyncio => "asyncio",
            StdModule::Threading => "threading",
            StdModule::Socket => "socket",
            StdModule::Urllib => "urllib",
        }
    }

    /// Modules that only exist on stdpython's std tier: they touch the OS
    /// (or, for math, std's float intrinsics), so the no_std profile has
    /// nothing to lower them to. The complement — json, string,
    /// collections, itertools, functools, heapq, copy, textwrap, hashlib,
    /// csv, warnings, and io's in-memory buffers — lives on the alloc
    /// tier and stays importable.
    pub(crate) fn is_std_only(self) -> bool {
        match self {
            StdModule::Os
            | StdModule::Sys
            | StdModule::Re
            | StdModule::Argparse
            | StdModule::Math
            | StdModule::Random
            | StdModule::Datetime
            | StdModule::Time
            | StdModule::Glob
            | StdModule::Pathlib
            | StdModule::Tempfile
            | StdModule::Subprocess
            | StdModule::Sysconfig
            | StdModule::Venv
            | StdModule::Numpy
            | StdModule::Asyncio
            | StdModule::Threading
            | StdModule::Socket
            | StdModule::Urllib => true,
            StdModule::Io
            | StdModule::Json
            | StdModule::Collections
            | StdModule::Itertools
            | StdModule::Functools
            | StdModule::Heapq
            | StdModule::Copy
            | StdModule::Textwrap
            | StdModule::Hashlib
            | StdModule::Csv
            | StdModule::String
            | StdModule::Warnings => false,
        }
    }

    /// Modules recognized as path receivers from their BARE rendered
    /// token, even without an import in scope (attribute.rs's
    /// module-access check; the general mechanism is the import-driven
    /// module chain). The excluded modules — argparse, collections, copy,
    /// glob, pathlib, string, sysconfig, tempfile, venv, warnings — are
    /// common variable names (or reached only through real imports):
    /// classifying them by token alone would misread user bindings.
    /// `datetime` is included deliberately: it covers both the runtime
    /// module and the datetime TYPE from `from datetime import datetime`
    /// — either way the attribute is a path item (datetime::strptime,
    /// datetime::now), never a field on a value.
    pub(crate) fn bare_token_access(self) -> bool {
        matches!(
            self,
            StdModule::Sys
                | StdModule::Os
                | StdModule::Subprocess
                | StdModule::Json
                | StdModule::Urllib
                | StdModule::Asyncio
                | StdModule::Time
                | StdModule::Math
                | StdModule::Random
                | StdModule::Heapq
                | StdModule::Functools
                | StdModule::Textwrap
                | StdModule::Itertools
                | StdModule::Re
                | StdModule::Hashlib
                | StdModule::Csv
                | StdModule::Io
                | StdModule::Threading
                | StdModule::Socket
                | StdModule::Datetime
                | StdModule::Numpy
        )
    }

    /// Modules whose functions route through call.rs's borrow/arity
    /// dispatcher in the `from X import f; f(...)` spelling (their
    /// runtime shapes borrow arguments, split by arity, or thread `?`).
    pub(crate) fn dispatches_from_import(self) -> bool {
        matches!(
            self,
            StdModule::Functools
                | StdModule::Heapq
                | StdModule::Copy
                | StdModule::Textwrap
                | StdModule::Re
                | StdModule::Hashlib
                | StdModule::Csv
                | StdModule::Io
        )
    }

    /// The same dispatcher for the qualified `X.f(...)` spelling — json
    /// and math have qualified-only entries.
    pub(crate) fn dispatches_qualified(self) -> bool {
        self.dispatches_from_import() || matches!(self, StdModule::Json | StdModule::Math)
    }
}

/// The arity/keyword VARIANT names of runtime functions — one Python
/// name maps to several Rust functions split by call shape (accumulate
/// with initial=, StringIO seeded, ...). Each name literal exists exactly
/// once, here: import.rs consumes the per-function slices (emitting
/// `#[allow(unused_imports)] use` for every variant so the lowering can
/// pick any of them), and call.rs renders individual variants through
/// these constants. Previously the same names were spelled in both files
/// and could drift — a call.rs arm without a matching import entry is an
/// E0425 in the GENERATED crate.
pub(crate) mod variant {
    pub(crate) const ACCUMULATE_SUM: &str = "accumulate_sum";
    pub(crate) const ACCUMULATE_FUNC: &str = "accumulate_func";
    pub(crate) const ACCUMULATE_SUM_INITIAL: &str = "accumulate_sum_initial";
    pub(crate) const ACCUMULATE_FUNC_INITIAL: &str = "accumulate_func_initial";
    pub(crate) const PRODUCT2: &str = "product2";
    pub(crate) const PRODUCT3: &str = "product3";
    pub(crate) const PRODUCT_REPEAT2: &str = "product_repeat2";
    pub(crate) const PRODUCT_REPEAT3: &str = "product_repeat3";
    pub(crate) const ZIP_LONGEST_FILL: &str = "zip_longest_fill";
    pub(crate) const GROUPBY_KEY: &str = "groupby_key";
    pub(crate) const REDUCE_INITIAL: &str = "reduce_initial";
    pub(crate) const FINDALL2: &str = "findall2";
    pub(crate) const FINDALL3: &str = "findall3";
    pub(crate) const STRINGIO_SEEDED: &str = "StringIO_seeded";
    pub(crate) const BYTESIO_SEEDED: &str = "BytesIO_seeded";
    pub(crate) const MD5_NEW: &str = "md5_new";
    pub(crate) const SHA1_NEW: &str = "sha1_new";
    pub(crate) const SHA256_NEW: &str = "sha256_new";
    pub(crate) const SHA512_NEW: &str = "sha512_new";
}

/// Per-(module, python-name) variant lists, for import.rs's bring-along.
pub(crate) fn runtime_fn_variants(module: StdModule, name: &str) -> &'static [&'static str] {
    use variant::*;
    match (module, name) {
        (StdModule::Itertools, "accumulate") => &[
            ACCUMULATE_SUM,
            ACCUMULATE_FUNC,
            ACCUMULATE_SUM_INITIAL,
            ACCUMULATE_FUNC_INITIAL,
        ],
        (StdModule::Itertools, "product") => {
            &[PRODUCT2, PRODUCT3, PRODUCT_REPEAT2, PRODUCT_REPEAT3]
        }
        (StdModule::Itertools, "zip_longest") => &[ZIP_LONGEST_FILL],
        (StdModule::Itertools, "groupby") => &[GROUPBY_KEY],
        (StdModule::Functools, "reduce") => &[REDUCE_INITIAL],
        (StdModule::Re, "findall") => &[FINDALL2, FINDALL3],
        (StdModule::Io, "StringIO") => &[STRINGIO_SEEDED],
        (StdModule::Io, "BytesIO") => &[BYTESIO_SEEDED],
        (StdModule::Hashlib, "md5") => &[MD5_NEW],
        (StdModule::Hashlib, "sha1") => &[SHA1_NEW],
        (StdModule::Hashlib, "sha256") => &[SHA256_NEW],
        (StdModule::Hashlib, "sha512") => &[SHA512_NEW],
        _ => &[],
    }
}

/// The `hashlib.<algo>()` zero-argument constructor's `_new` variant,
/// through the same shared constants (call.rs previously derived these
/// by `format!`, which import.rs's hardcoded list could silently miss).
pub(crate) fn hashlib_new_variant(fname: &str) -> Option<&'static str> {
    Some(match fname {
        "md5" => variant::MD5_NEW,
        "sha1" => variant::SHA1_NEW,
        "sha256" => variant::SHA256_NEW,
        "sha512" => variant::SHA512_NEW,
        _ => return None,
    })
}

/// The numpy module's import aliases: `import numpy as np` is THE
/// canonical spelling, so both names appear throughout Python sources.
/// One predicate instead of scattered `== "np" || == "numpy"` checks.
pub(crate) fn is_numpy_alias(name: &str) -> bool {
    matches!(name, "np" | "numpy")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every module, for the exhaustive round-trip walk. Lives in the
    /// test module (its only consumer): CI builds with -D warnings, so a
    /// test-only item in the non-test build would be a dead-code error.
    const ALL: [StdModule; 31] = [
        StdModule::Os,
        StdModule::Sys,
        StdModule::Re,
        StdModule::Io,
        StdModule::Argparse,
        StdModule::Json,
        StdModule::Math,
        StdModule::Random,
        StdModule::Datetime,
        StdModule::Time,
        StdModule::Collections,
        StdModule::Itertools,
        StdModule::Functools,
        StdModule::Heapq,
        StdModule::Copy,
        StdModule::Textwrap,
        StdModule::Hashlib,
        StdModule::Csv,
        StdModule::Glob,
        StdModule::Pathlib,
        StdModule::Tempfile,
        StdModule::Subprocess,
        StdModule::String,
        StdModule::Sysconfig,
        StdModule::Venv,
        StdModule::Warnings,
        StdModule::Numpy,
        StdModule::Asyncio,
        StdModule::Threading,
        StdModule::Socket,
        StdModule::Urllib,
    ];

    /// from_name and name() are the enum's only two string matches; the
    /// round trip pins them to each other so neither can drift.
    #[test]
    fn names_round_trip() {
        for module in ALL {
            assert_eq!(
                StdModule::from_name(module.name()),
                Some(module),
                "{:?} does not round-trip through its name",
                module
            );
        }
        assert_eq!(StdModule::from_name("logging"), None);
        assert_eq!(StdModule::from_name(""), None);
    }
}
