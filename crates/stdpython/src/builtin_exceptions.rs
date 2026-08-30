//! The built-in Python exception tree — derived from the interpreter.
//!
//! Exception types are an OPEN set at runtime — generated programs raise
//! user-defined classes whose names cannot exist here — so `PyException`
//! carries its type as a string. The BUILT-IN subset's hierarchy comes
//! from the live Python interpreter: python-ast dumps every
//! `BaseException` subclass's real `__mro__` through PyO3 (the same
//! path that produces parse trees — `dump_builtin_exception_tree`), and
//! the checked-in generated file below carries that data: the MRO
//! table, the one name→enum boundary, and the canonical-name map, all
//! rendered from the same interpreter dump so none can drift from the
//! others. python-ast's `exception_tree_is_current` test verifies the
//! checked-in file against the live interpreter on every run — a
//! divergence (new exception, restructured MRO, new alias) is a loud
//! test failure, and `RYTHON_REGEN=1` regenerates the file.
//!
//! What cannot be derived stays hand-written here: the PyO3 surfacing
//! of each exception (a mapping onto pyo3's error API, not tree data)
//! and the runtime `matches` walk (in `PyException`), which is
//! hand-written but reads interpreter-derived data.

// The generated enum and its `from_name` are read only by the PyO3
// surfacing below, so with `std` but no `pyo3-interop` they are unused.
// The allow sits here rather than in the generated file, which is rendered
// from the interpreter dump and must not be hand-edited.
#![cfg_attr(not(feature = "pyo3-interop"), allow(dead_code))]

include!("builtin_exceptions_gen.rs");

#[cfg(feature = "pyo3-interop")]
impl BuiltinException {
    /// The real Python exception this type surfaces as through PyO3, so
    /// `raise ValueError(...)` reaches Python callers as an actual
    /// ValueError. Exhaustive — a new variant (a new exception in the
    /// interpreter dump) must decide its surfacing. pyo3 0.29 wraps no
    /// IndentationError/TabError/_IncompleteInputError (SyntaxErrors in
    /// CPython's tree) and none of the stdlib-module exceptions
    /// (OSErrors); each surfaces through that ancestor.
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
            URLError | HTTPError | ContentTooShortError | Gaierror | Herror | SSLError
            | SSLZeroReturnError | SSLWantReadError | SSLWantWriteError | SSLSyscallError
            | SSLEOFError | SSLCertVerificationError => PyOSError::new_err(msg),
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
            // ssl._GiveupOnSendfile — a plain Exception subclass pyo3
            // does not wrap.
            GiveupOnSendfile => PyException::new_err(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::PyException;

    fn caught_by(raised: &str, target: &str) -> bool {
        PyException::new(raised, "m").matches(target)
    }

    #[test]
    fn alias_and_group_special_cases_hold() {
        // Multiple inheritance and aliases come from the interpreter's
        // MRO data (ExceptionGroup additionally IS-A Exception even
        // though its MRO lists BaseExceptionGroup first;
        // SSLCertVerificationError IS-A ValueError as its second base).
        assert!(caught_by("SSLCertVerificationError", "ValueError"));
        assert!(caught_by("SSLCertVerificationError", "SSLError"));
        assert!(caught_by("SSLError", "OSError"));
        assert!(!caught_by("SSLError", "ValueError"));
        assert!(caught_by("ExceptionGroup", "Exception"));
        assert!(!caught_by("BaseExceptionGroup", "Exception"));
        assert!(!caught_by("SystemExit", "Exception"));
        assert!(caught_by("SystemExit", "BaseException"));
        // EnvironmentError/IOError ARE OSError — both directions.
        assert!(caught_by("EnvironmentError", "OSError"));
        assert!(caught_by("IOError", "OSError"));
        assert!(caught_by("OSError", "EnvironmentError"));
    }

    #[test]
    fn stdlib_module_exceptions_walk_into_the_builtin_tree() {
        // `except OSError:` must catch a raised URLError or gaierror
        // (interpreter-derived MROs; the rypip urllib e2e test pins the
        // URLError case end to end).
        assert!(caught_by("URLError", "OSError"));
        assert!(caught_by("HTTPError", "URLError"));
        assert!(caught_by("HTTPError", "OSError"));
        assert!(caught_by("ContentTooShortError", "URLError"));
        assert!(caught_by("gaierror", "OSError"));
        assert!(caught_by("herror", "OSError"));
        assert!(!caught_by("URLError", "ConnectionError"));
        // socket.timeout IS TimeoutError.
        assert!(caught_by("timeout", "OSError"));
        assert!(caught_by("TimeoutError", "OSError"));
    }

    #[test]
    fn unknown_raised_types_keep_the_broad_posture() {
        // A raised type outside the builtin table (a user class) is
        // caught only by Exception and BaseException — the documented
        // divergence (rython does not know user-class hierarchies).
        assert!(caught_by("MyError", "Exception"));
        assert!(caught_by("MyError", "BaseException"));
        assert!(!caught_by("MyError", "OSError"));
        assert!(!caught_by("MyError", "MyOtherError"));
    }

    // The enum registry (from_name + variants) exists only for the
    // std-tier PyO3 surfacing; the matching semantics above are
    // tier-independent and run in the alloc tier too.
    #[cfg(feature = "std")]
    mod registry {
        // The enum lives at this module's top level (the include!);
        // `super` here is the tests module, so reach the enum by path.
        use crate::builtin_exceptions::*;

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
        fn canonicalization_is_the_alias_boundary() {
            // The aliases canonicalize to the class object they ARE —
            // interpreter data, pinned by python-ast's exception_tree
            // tests against the live interpreter.
            assert_eq!(
                BuiltinException::from_name("EnvironmentError"),
                Some(BuiltinException::OSError)
            );
            assert_eq!(
                BuiltinException::from_name("IOError"),
                Some(BuiltinException::OSError)
            );
            assert_eq!(
                BuiltinException::from_name("CertificateError"),
                Some(BuiltinException::SSLCertVerificationError)
            );
            assert_eq!(
                BuiltinException::from_name("timeout"),
                Some(BuiltinException::TimeoutError)
            );
            assert_eq!(
                BuiltinException::from_name("_GiveupOnSendfile"),
                Some(BuiltinException::GiveupOnSendfile)
            );
        }
    }
}
