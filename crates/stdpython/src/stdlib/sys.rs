//! Python sys module implementation
//!
//! This module provides Python's sys module functionality including
//! system-specific parameters and functions. Uses generic traits for
//! maximum flexibility and reusability.

use crate::python_function;

/// sys.executable - path to the Python executable (property)
///
/// In a real Python environment, this would be the path to the Python interpreter.
/// For Rust-compiled Python code, we use the current executable path.
///
/// Note: This uses lazy evaluation to get the actual executable path at runtime.
pub static executable: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    std::env::current_exe()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| "python".to_string())
});

/// sys.version_info - version information as a tuple-like structure
///
/// Python's version_info is a named tuple with major, minor, micro, etc.
/// For compiled code, we simulate Python version information.
pub static version_info: std::sync::LazyLock<Vec<i32>> = std::sync::LazyLock::new(|| {
    vec![3, 11, 0] // Simulate Python 3.11.0
});

/// sys.prefix - installation prefix
///
/// In Python, this is the directory prefix where Python is installed.
/// For compiled code, we use the executable's directory.
pub static prefix: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|p| p.to_string_lossy().to_string()))
        .unwrap_or_else(|| "/usr/local".to_string())
});

/// sys.base_prefix - base installation prefix
///
/// In Python, this is the base installation prefix (before virtual environments).
/// For simplicity, we make it the same as prefix.
pub static base_prefix: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| prefix.clone());

/// sys.argv - command line arguments (property)
///
/// Returns the command line arguments passed to the program.
/// This reflects the actual command line arguments, just like Python's sys.argv.
///
/// Note: This uses lazy evaluation to get the actual command line arguments at runtime.
pub static argv: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
    std::env::args_os()
        .enumerate()
        .map(|(i, arg)| decode_argument(i, arg))
        .collect()
});

/// One process argument by index, decoded by the same policy as
/// `sys.argv` without materializing the whole vector: argparse's default
/// `prog` needs argv[0] only, and must not trip over an unrelated
/// argument the program never reads (Devin review on #340, round 4).
pub fn argv_at(index: usize) -> Option<String> {
    std::env::args_os()
        .nth(index)
        .map(|arg| decode_argument(index, arg))
}

/// The runtime's one policy for a process argument: CPython decodes a
/// non-UTF-8 argument with surrogateescape (`'\udcff'`), which this
/// runtime's `str` does not model, so such an argument is a loud exit
/// naming it — never a panic and never a different string — exactly when
/// it is read (Devin review on #340).
fn decode_argument(index: usize, arg: std::ffi::OsString) -> String {
    match arg.into_string() {
        Ok(s) => s,
        Err(raw) => {
            eprintln!(
                "rython: sys.argv[{}] is not valid UTF-8 ({:?}); CPython would keep it as \
                 surrogate escapes, which this runtime does not model",
                index, raw
            );
            std::process::exit(1)
        }
    }
}

python_function! {
    /// Raise SystemExit; only an uncaught exception at entry ends the process.
    pub fn exit<T>(code: T) -> Result<(), crate::PyException>
    where [T: Into<ExitCode>]
    [signature: (code=None)]
    [concrete_types: (String) -> Result<(), crate::PyException>]
    {
        let args = match code.into() {
            ExitCode::Missing => Vec::new(),
            code => vec![code.into_value()],
        };
        Err(crate::PyException::system_exit(args))
    }
}

/// The no-argument call has no BaseException args (`sys.exit(None)` has one).
pub fn exit_none() -> Result<(), crate::PyException> {
    exit(ExitCode::Missing)
}

/// Preserve the Python value until an uncaught SystemExit ends the process.
pub enum ExitCode {
    Missing,
    Code(i32),
    Message(String),
    Value(crate::PyValue),
}

impl ExitCode {
    fn into_value(self) -> crate::PyValue {
        match self {
            Self::Missing => unreachable!("the no-argument exit has no value"),
            Self::Code(code) => crate::PyValue::Int(i64::from(code)),
            Self::Message(message) => crate::PyValue::Str(message),
            Self::Value(value) => value,
        }
    }
}

macro_rules! exit_code_from_int {
    ($($ty:ty),*) => {$(
        impl From<$ty> for ExitCode {
            fn from(code: $ty) -> Self {
                Self::Value(crate::PyValue::Int(i64::from(code)))
            }
        }
    )*};
}
exit_code_from_int!(i8, i16, i32, i64, u8, u16, u32);

impl From<&str> for ExitCode {
    fn from(message: &str) -> Self {
        ExitCode::Message(message.to_string())
    }
}

impl From<String> for ExitCode {
    fn from(message: String) -> Self {
        ExitCode::Message(message)
    }
}

impl From<crate::PyValue> for ExitCode {
    fn from(code: crate::PyValue) -> Self {
        Self::Value(code)
    }
}

impl From<bool> for ExitCode {
    fn from(code: bool) -> Self {
        Self::Value(crate::PyValue::Bool(code))
    }
}

impl From<()> for ExitCode {
    fn from(_: ()) -> Self {
        Self::Value(crate::PyValue::None_)
    }
}

impl From<f64> for ExitCode {
    fn from(code: f64) -> Self {
        Self::Value(crate::PyValue::Float(code))
    }
}

python_function! {
    /// sys.exc_info(): the active exception as (type, value, traceback).
    /// rython's exceptions are string-tagged PyException values with no
    /// traceback object — the tuple lowers to a boxed value whose members
    /// are None (the traceback divergence).
    pub fn exc_info() -> crate::PyValue
    [signature: ()]
    [concrete_types: () -> crate::PyValue]
    {
        crate::PyValue::None_
    }
}

python_function! {
    /// sys.platform - platform identifier
    pub fn platform() -> &'static str
    [signature: ()]
    [concrete_types: () -> &'static str]
    {
        if cfg!(target_os = "windows") {
            "win32"
        } else if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(unix) {
            "unix"
        } else {
            "unknown"
        }
    }
}

python_function! {
    /// sys.version - version information
    pub fn version() -> String
    [signature: ()]
    [concrete_types: () -> String]
    {
        format!("Python-to-Rust compiled code (rustc {})",
            option_env!("RUSTC_VERSION").unwrap_or("unknown"))
    }
}

python_function! {
    /// Generic helper: Get executable path
    pub fn get_executable() -> String
    [signature: ()]
    [concrete_types: () -> String]
    {
        #[cfg(feature = "std")]
        {
            executable.clone()
        }
        #[cfg(not(feature = "std"))]
        {
            "python".to_string()
        }
    }
}

python_function! {
    /// Generic helper: Get command line arguments
    pub fn get_argv() -> Vec<String>
    [signature: ()]
    [concrete_types: () -> Vec<String>]
    {
        #[cfg(feature = "std")]
        {
            argv.iter().cloned().collect()
        }
        #[cfg(not(feature = "std"))]
        {
            vec!["python".to_string()]
        }
    }
}

python_function! {
    /// Generic helper: Get platform identifier
    pub fn get_platform() -> &'static str
    [signature: ()]
    [concrete_types: () -> &'static str]
    {
        platform()
    }
}

python_function! {
    /// sys.audit - fire an audit event (urllib3's connection.py,
    /// `sys.audit("http.client.connect", self, self.host, self.port)`).
    /// rython has no audit-hook framework — the event is dropped (the
    /// audit divergence: CPython's audit hooks observe the event; rython
    /// programs cannot register hooks, so nothing is lost at runtime).
    pub fn audit(event: String, args: Vec<crate::PyValue>) -> ()
    [signature: (event, args)]
    [concrete_types: (String, Vec<crate::PyValue>) -> ()]
    {
        let _ = (event, args);
    }
}
/// sys.implementation — the running interpreter's identity. rython pins
/// CPython semantics (docs/spec.md conformance rule), so the name
/// reports "cpython". A nested module, because the dotted attribute
/// chain renders as the path `sys::implementation::name`.
pub mod implementation {
    #[allow(non_upper_case_globals)]
    pub static name: &'static str = "cpython";
}

/// sys.pypy_version_info — exists only under PyPy; readable code guards
/// it behind `sys.implementation.name == "pypy"`, which is statically
/// false here, so the value is unreachable at runtime. Zeros keep the
/// guarded code compiling (the interpreter-identity divergence).
#[allow(non_upper_case_globals)]
pub static pypy_version_info: (i64, i64, i64) = (0, 0, 0);
