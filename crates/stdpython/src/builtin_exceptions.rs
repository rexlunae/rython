//! The built-in Python exception tree, as an enum.
//!
//! Exception types are an OPEN set at runtime — generated programs raise
//! user-defined classes whose names cannot exist here — so `PyException`
//! carries its type as a string. But the BUILT-IN subset is closed, and
//! per the AGENTS.md parse-into-enums rule it is modeled as
//! [`BuiltinException`]: the name string is parsed exactly once, at
//! [`BuiltinException::from_name`], and everything downstream — ancestry
//! walks for `except` matching, PyO3 surfacing — is an exhaustive `match`
//! on the enum, so a new exception type cannot be added half-way (a
//! variant missing from [`parent`](BuiltinException::parent) or
//! [`pyo3_err`](BuiltinException::pyo3_err) is a compile error, not a
//! runtime miss). The tree itself is generated from python3 3.14
//! `__mro__` dumps.
//!
//! python-ast does not depend on this crate, so its syntactic classifier
//! (`raise_stmt::BUILTIN_EXCEPTION_NAMES`) is a separate boundary
//! registry; this module is the runtime's authority.

/// Emits the enum together with its name mapping and (test-only) variant
/// list from ONE row per exception, so those three can never drift.
macro_rules! builtin_exceptions {
    (
        $($variant:ident => $name:literal),* $(,)?
    ) => {
        /// One variant per built-in exception type the runtime can match
        /// on, plus the stdlib-module exceptions the socket/urllib
        /// runtimes raise (the URLError family, gaierror/herror) —
        /// closed-world, so `except` ancestry is a compile-checked walk
        /// instead of a string table.
        ///
        /// The historical aliases EnvironmentError/IOError are NOT
        /// variants: `from_name` canonicalizes them to
        /// [`OSError`](Self::OSError), which they alias in CPython
        /// (`EnvironmentError is OSError` → True).
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum BuiltinException {
            $($variant),*
        }

        impl BuiltinException {
            /// The ONE string→enum boundary. Returns `None` for names
            /// outside the built-in tree (user-defined classes).
            pub(crate) fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)*
                    // Historical aliases of OSError (CPython: the same
                    // class object).
                    "EnvironmentError" | "IOError" => Some(Self::OSError),
                    _ => None,
                }
            }

            /// The canonical Python name (what CPython prints).
            #[cfg(test)]
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),*
                }
            }

            /// Every variant, in declaration order — emitted by the same
            /// macro row as the name mapping, so it cannot drift.
            #[cfg(test)]
            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant),*];
        }
    };
}

builtin_exceptions! {
    BaseException => "BaseException",
    Exception => "Exception",
    SystemExit => "SystemExit",
    KeyboardInterrupt => "KeyboardInterrupt",
    GeneratorExit => "GeneratorExit",
    ArithmeticError => "ArithmeticError",
    AssertionError => "AssertionError",
    AttributeError => "AttributeError",
    BufferError => "BufferError",
    EOFError => "EOFError",
    ImportError => "ImportError",
    ModuleNotFoundError => "ModuleNotFoundError",
    LookupError => "LookupError",
    MemoryError => "MemoryError",
    NameError => "NameError",
    OSError => "OSError",
    ReferenceError => "ReferenceError",
    RuntimeError => "RuntimeError",
    StopIteration => "StopIteration",
    StopAsyncIteration => "StopAsyncIteration",
    SyntaxError => "SyntaxError",
    SystemError => "SystemError",
    TypeError => "TypeError",
    ValueError => "ValueError",
    Warning => "Warning",
    // Arithmetic leaves.
    FloatingPointError => "FloatingPointError",
    OverflowError => "OverflowError",
    ZeroDivisionError => "ZeroDivisionError",
    // Lookup leaves.
    IndexError => "IndexError",
    KeyError => "KeyError",
    // Name leaf.
    UnboundLocalError => "UnboundLocalError",
    // OSError subtree.
    BlockingIOError => "BlockingIOError",
    ChildProcessError => "ChildProcessError",
    ConnectionError => "ConnectionError",
    BrokenPipeError => "BrokenPipeError",
    ConnectionAbortedError => "ConnectionAbortedError",
    ConnectionRefusedError => "ConnectionRefusedError",
    ConnectionResetError => "ConnectionResetError",
    FileExistsError => "FileExistsError",
    FileNotFoundError => "FileNotFoundError",
    InterruptedError => "InterruptedError",
    IsADirectoryError => "IsADirectoryError",
    NotADirectoryError => "NotADirectoryError",
    PermissionError => "PermissionError",
    ProcessLookupError => "ProcessLookupError",
    TimeoutError => "TimeoutError",
    // urllib.error family (the http-ureq runtime raises these):
    // URLError IS-A OSError and HTTPError IS-A URLError in CPython.
    URLError => "URLError",
    HTTPError => "HTTPError",
    ContentTooShortError => "ContentTooShortError",
    // socket-module exceptions (socket.timeout IS TimeoutError; verified
    // against python3: gaierror/herror → OSError).
    Gaierror => "gaierror",
    Herror => "herror",
    // RuntimeError leaves.
    NotImplementedError => "NotImplementedError",
    RecursionError => "RecursionError",
    PythonFinalizationError => "PythonFinalizationError",
    // Syntax tree.
    IndentationError => "IndentationError",
    TabError => "TabError",
    IncompleteInputError => "_IncompleteInputError",
    // Unicode tree (hangs off ValueError).
    UnicodeError => "UnicodeError",
    UnicodeDecodeError => "UnicodeDecodeError",
    UnicodeEncodeError => "UnicodeEncodeError",
    UnicodeTranslateError => "UnicodeTranslateError",
    // Exception groups. ExceptionGroup MULTIPLY inherits
    // (BaseExceptionGroup, Exception) — the second ancestry is explicit
    // in is_caught_by.
    BaseExceptionGroup => "BaseExceptionGroup",
    ExceptionGroup => "ExceptionGroup",
    // Warning tree.
    BytesWarning => "BytesWarning",
    DeprecationWarning => "DeprecationWarning",
    EncodingWarning => "EncodingWarning",
    FutureWarning => "FutureWarning",
    ImportWarning => "ImportWarning",
    PendingDeprecationWarning => "PendingDeprecationWarning",
    ResourceWarning => "ResourceWarning",
    RuntimeWarning => "RuntimeWarning",
    SyntaxWarning => "SyntaxWarning",
    UnicodeWarning => "UnicodeWarning",
    UserWarning => "UserWarning",
}

impl BuiltinException {
    /// The DIRECT parent (`None` for BaseException, the root). Exhaustive:
    /// adding a variant without placing it in the tree fails to compile.
    pub(crate) fn parent(self) -> Option<Self> {
        use BuiltinException::*;
        Some(match self {
            BaseException => return None,
            Exception | SystemExit | KeyboardInterrupt | GeneratorExit | BaseExceptionGroup => {
                BaseException
            }
            ArithmeticError | AssertionError | AttributeError | BufferError | EOFError
            | ImportError | LookupError | MemoryError | NameError | OSError | ReferenceError
            | RuntimeError | StopIteration | StopAsyncIteration | SyntaxError | SystemError
            | TypeError | ValueError | Warning => Exception,
            ModuleNotFoundError => ImportError,
            FloatingPointError | OverflowError | ZeroDivisionError => ArithmeticError,
            IndexError | KeyError => LookupError,
            UnboundLocalError => NameError,
            BlockingIOError | ChildProcessError | ConnectionError | FileExistsError
            | FileNotFoundError | InterruptedError | IsADirectoryError | NotADirectoryError
            | PermissionError | ProcessLookupError | TimeoutError | URLError | Gaierror
            | Herror => OSError,
            BrokenPipeError | ConnectionAbortedError | ConnectionRefusedError
            | ConnectionResetError => ConnectionError,
            HTTPError | ContentTooShortError => URLError,
            NotImplementedError | RecursionError | PythonFinalizationError => RuntimeError,
            IndentationError | IncompleteInputError => SyntaxError,
            TabError => IndentationError,
            UnicodeError => ValueError,
            UnicodeDecodeError | UnicodeEncodeError | UnicodeTranslateError => UnicodeError,
            ExceptionGroup => BaseExceptionGroup,
            BytesWarning | DeprecationWarning | EncodingWarning | FutureWarning | ImportWarning
            | PendingDeprecationWarning | ResourceWarning | RuntimeWarning | SyntaxWarning
            | UnicodeWarning | UserWarning => Warning,
        })
    }

    /// Whether `except <target>:` catches a raised `self` — the target is
    /// the type itself or one of its ancestors. ExceptionGroup's second
    /// base (it multiply inherits BaseExceptionGroup AND Exception in
    /// CPython) is the one edge the single-parent walk cannot express.
    pub(crate) fn is_caught_by(self, target: Self) -> bool {
        if self == target {
            return true;
        }
        if self == Self::ExceptionGroup && target == Self::Exception {
            return true;
        }
        let mut current = self.parent();
        while let Some(ancestor) = current {
            if ancestor == target {
                return true;
            }
            current = ancestor.parent();
        }
        false
    }

    /// The real Python exception this type surfaces as through PyO3, so
    /// `raise ValueError(...)` reaches Python callers as an actual
    /// ValueError. Exhaustive — a new variant must decide its surfacing.
    /// pyo3 0.29 wraps no IndentationError/TabError/_IncompleteInputError
    /// (SyntaxErrors in CPython's tree) and none of the stdlib-module
    /// exceptions (OSErrors); each surfaces through that ancestor.
    #[cfg(feature = "std")]
    pub(crate) fn pyo3_err(self, msg: String) -> pyo3::PyErr {
        use pyo3::exceptions::*;
        use BuiltinException::*;
        match self {
            BaseException => PyBaseException::new_err(msg),
            Exception => PyException::new_err(msg),
            SystemExit => PySystemExit::new_err(msg),
            KeyboardInterrupt => PyKeyboardInterrupt::new_err(msg),
            GeneratorExit => PyGeneratorExit::new_err(msg),
            ArithmeticError => PyArithmeticError::new_err(msg),
            AssertionError => PyAssertionError::new_err(msg),
            AttributeError => PyAttributeError::new_err(msg),
            BufferError => PyBufferError::new_err(msg),
            EOFError => PyEOFError::new_err(msg),
            ImportError => PyImportError::new_err(msg),
            ModuleNotFoundError => PyModuleNotFoundError::new_err(msg),
            LookupError => PyLookupError::new_err(msg),
            MemoryError => PyMemoryError::new_err(msg),
            NameError => PyNameError::new_err(msg),
            OSError => PyOSError::new_err(msg),
            ReferenceError => PyReferenceError::new_err(msg),
            RuntimeError => PyRuntimeError::new_err(msg),
            StopIteration => PyStopIteration::new_err(msg),
            StopAsyncIteration => PyStopAsyncIteration::new_err(msg),
            SyntaxError | IndentationError | TabError | IncompleteInputError => {
                PySyntaxError::new_err(msg)
            }
            SystemError => PySystemError::new_err(msg),
            TypeError => PyTypeError::new_err(msg),
            ValueError => PyValueError::new_err(msg),
            Warning => PyWarning::new_err(msg),
            FloatingPointError => PyFloatingPointError::new_err(msg),
            OverflowError => PyOverflowError::new_err(msg),
            ZeroDivisionError => PyZeroDivisionError::new_err(msg),
            IndexError => PyIndexError::new_err(msg),
            KeyError => PyKeyError::new_err(msg),
            UnboundLocalError => PyUnboundLocalError::new_err(msg),
            BlockingIOError => PyBlockingIOError::new_err(msg),
            ChildProcessError => PyChildProcessError::new_err(msg),
            ConnectionError => PyConnectionError::new_err(msg),
            BrokenPipeError => PyBrokenPipeError::new_err(msg),
            ConnectionAbortedError => PyConnectionAbortedError::new_err(msg),
            ConnectionRefusedError => PyConnectionRefusedError::new_err(msg),
            ConnectionResetError => PyConnectionResetError::new_err(msg),
            FileExistsError => PyFileExistsError::new_err(msg),
            FileNotFoundError => PyFileNotFoundError::new_err(msg),
            InterruptedError => PyInterruptedError::new_err(msg),
            IsADirectoryError => PyIsADirectoryError::new_err(msg),
            NotADirectoryError => PyNotADirectoryError::new_err(msg),
            PermissionError => PyPermissionError::new_err(msg),
            ProcessLookupError => PyProcessLookupError::new_err(msg),
            TimeoutError => PyTimeoutError::new_err(msg),
            URLError | HTTPError | ContentTooShortError | Gaierror | Herror => {
                PyOSError::new_err(msg)
            }
            NotImplementedError => PyNotImplementedError::new_err(msg),
            RecursionError => PyRecursionError::new_err(msg),
            PythonFinalizationError => PyRuntimeError::new_err(msg),
            UnicodeError => PyUnicodeError::new_err(msg),
            UnicodeDecodeError => PyUnicodeDecodeError::new_err(msg),
            UnicodeEncodeError => PyUnicodeEncodeError::new_err(msg),
            UnicodeTranslateError => PyUnicodeTranslateError::new_err(msg),
            // pyo3 0.29 exposes PyBaseExceptionGroup; ExceptionGroup
            // surfaces through it (its CPython MRO roots there too).
            BaseExceptionGroup | ExceptionGroup => PyBaseExceptionGroup::new_err(msg),
            BytesWarning => PyBytesWarning::new_err(msg),
            DeprecationWarning => PyDeprecationWarning::new_err(msg),
            EncodingWarning => PyEncodingWarning::new_err(msg),
            FutureWarning => PyFutureWarning::new_err(msg),
            ImportWarning => PyImportWarning::new_err(msg),
            PendingDeprecationWarning => PyPendingDeprecationWarning::new_err(msg),
            ResourceWarning => PyResourceWarning::new_err(msg),
            RuntimeWarning => PyRuntimeWarning::new_err(msg),
            SyntaxWarning => PySyntaxWarning::new_err(msg),
            UnicodeWarning => PyUnicodeWarning::new_err(msg),
            UserWarning => PyUserWarning::new_err(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BuiltinException::{self, *};

    #[test]
    fn names_round_trip() {
        for v in BuiltinException::ALL {
            assert_eq!(
                BuiltinException::from_name(v.name()),
                Some(*v),
                "{} does not round-trip",
                v.name()
            );
        }
    }

    #[test]
    fn every_chain_bottoms_out_at_base_exception() {
        // parent() is exhaustive, but a wrong arm could still form a cycle
        // or a second root; the walk is bounded well past the tree's depth.
        for v in BuiltinException::ALL {
            let mut current = *v;
            for _ in 0..16 {
                match current.parent() {
                    Some(p) => current = p,
                    None => break,
                }
            }
            assert_eq!(
                current,
                BaseException,
                "{}'s ancestry does not reach BaseException",
                v.name()
            );
        }
    }

    #[test]
    fn alias_and_group_special_cases_hold() {
        // Verified against python3 3.14: EnvironmentError/IOError ARE
        // OSError (aliases), and ExceptionGroup additionally IS-A
        // Exception even though its MRO lists BaseExceptionGroup first.
        assert_eq!(
            BuiltinException::from_name("EnvironmentError"),
            Some(OSError)
        );
        assert_eq!(BuiltinException::from_name("IOError"), Some(OSError));
        assert!(ExceptionGroup.is_caught_by(Exception));
        assert!(!BaseExceptionGroup.is_caught_by(Exception));
        assert!(!SystemExit.is_caught_by(Exception));
        assert!(SystemExit.is_caught_by(BaseException));
    }

    #[test]
    fn stdlib_module_exceptions_walk_into_the_builtin_tree() {
        // `except OSError:` must catch a raised URLError or gaierror
        // (python3-verified MROs; the rypip urllib e2e test pins the
        // URLError case end to end).
        assert!(URLError.is_caught_by(OSError));
        assert!(HTTPError.is_caught_by(URLError));
        assert!(HTTPError.is_caught_by(OSError));
        assert!(ContentTooShortError.is_caught_by(URLError));
        assert!(Gaierror.is_caught_by(OSError));
        assert!(Herror.is_caught_by(OSError));
        assert!(!URLError.is_caught_by(ConnectionError));
    }
}
