//! Every way the JIT fails, each with one loud outcome (docs/jit.md,
//! "Correct or loud"). There is no fallback path: a source the JIT cannot
//! compile, or a call it cannot make faithfully, is one of these errors.

use std::fmt;

/// A JIT failure.
#[derive(Debug)]
pub enum JitError {
    /// The converter refused the source. The message is the converter's,
    /// verbatim: the construct, its location and the rewrite.
    Conversion(String),
    /// The generated crate did not build — a converter bug by the
    /// project's own standard, carrying rustc's diagnostics — or the
    /// library it built could not be loaded.
    Build(String),
    /// A value cannot cross the boundary in M1; the message names its type.
    Marshal(String),
    /// Something the JIT needs is missing or unusable (the Rust toolchain,
    /// a private cache directory, a supported platform): what, and how it
    /// was looked for.
    Environment(String),
    /// The library's program entry already ran in this process. Its
    /// `sys.argv` and module state were initialized for that run, so a
    /// second run is refused rather than run with stale state.
    EntryAlreadyRun {
        /// The module whose entry ran.
        module: String,
    },
    /// A `SystemExit` escaped a function call. The host decides what to
    /// do; the library never ends the process.
    Exit {
        /// The exit status CPython would end with.
        code: i32,
        /// The line CPython prints on stderr for a non-integer code
        /// (`sys.exit("bye")`), if any.
        message: Option<String>,
    },
    /// A Python exception escaped a function call, rebuilt in the host:
    /// its type, message, argument reprs and (for a user class) ancestry.
    Python(stdpython::PyException),
    /// The compiled code panicked — rython's outcome where a behavior is
    /// unrepresentable. The library's panic hook has already reported it
    /// on stderr, as the binary's would.
    Panic(String),
    /// An operation M1 cannot perform faithfully, refused rather than
    /// approximated: the message says what and what to do instead.
    Unsupported(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Conversion(message) => write!(f, "{}", message),
            JitError::Build(message) => write!(f, "the generated crate did not build: {}", message),
            JitError::Marshal(message) => write!(f, "{}", message),
            JitError::Environment(message) => write!(f, "{}", message),
            JitError::EntryAlreadyRun { module } => write!(
                f,
                "the program entry of `{}` already ran in this process; a loaded \
                 library runs its entry once (its sys.argv and module state are \
                 initialized for that run) — run the program again in a new process",
                module
            ),
            JitError::Exit { code, message } => match message {
                Some(message) => write!(f, "SystemExit({}): {}", code, message),
                None => write!(f, "SystemExit({})", code),
            },
            JitError::Python(e) => write!(f, "{}", e.format_exception_only()),
            JitError::Panic(message) => write!(f, "the compiled code panicked: {}", message),
            JitError::Unsupported(message) => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for JitError {}
