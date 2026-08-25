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

/// The numpy module's import aliases: `import numpy as np` is THE
/// canonical spelling, so both names appear throughout Python sources.
/// One predicate instead of scattered `== "np" || == "numpy"` checks.
pub(crate) fn is_numpy_alias(name: &str) -> bool {
    matches!(name, "np" | "numpy")
}

impl StdModule {
    /// Every module, for exhaustive walks (the round-trip self-test).
    pub(crate) const ALL: [StdModule; 31] = [
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// from_name and name() are the enum's only two string matches; the
    /// round trip pins them to each other so neither can drift.
    #[test]
    fn names_round_trip() {
        for module in StdModule::ALL {
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
