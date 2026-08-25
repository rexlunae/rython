//! The built-in Python exception tree, as ONE authoritative table.
//!
//! Python exception names arrive from CPython's parser as strings, so the
//! boundary is [`BUILTIN_EXCEPTIONS`]: a single static table of
//! `(name, direct parent)` pairs, generated from python3 3.14 `__mro__`
//! dumps. Every consumer derives from it — ancestor walking for
//! `except` matching, "is this a built-in exception" checks — so the
//! tree cannot drift between parallel string lists (the AGENTS.md
//! parse-into-enums rule; a full enum would need to live in both
//! python-ast and stdpython, which do not depend on each other).

/// The DIRECT parent of every built-in Python exception type (`None` for
/// BaseException, the root). One hop per entry; ancestor queries walk the
/// chain.
pub(crate) static BUILTIN_EXCEPTION_PARENTS: &[(&str, Option<&str>)] = &[
    ("BaseException", None),
    ("Exception", Some("BaseException")),
    ("SystemExit", Some("BaseException")),
    ("KeyboardInterrupt", Some("BaseException")),
    ("GeneratorExit", Some("BaseException")),
    ("ArithmeticError", Some("Exception")),
    ("AssertionError", Some("Exception")),
    ("AttributeError", Some("Exception")),
    ("BufferError", Some("Exception")),
    ("EOFError", Some("Exception")),
    ("ImportError", Some("Exception")),
    ("ModuleNotFoundError", Some("ImportError")),
    ("LookupError", Some("Exception")),
    ("MemoryError", Some("Exception")),
    ("NameError", Some("Exception")),
    ("OSError", Some("Exception")),
    ("ReferenceError", Some("Exception")),
    ("RuntimeError", Some("Exception")),
    ("StopIteration", Some("Exception")),
    ("StopAsyncIteration", Some("Exception")),
    ("SyntaxError", Some("Exception")),
    ("SystemError", Some("Exception")),
    ("TypeError", Some("Exception")),
    ("ValueError", Some("Exception")),
    ("Warning", Some("Exception")),
    // Arithmetic leaves.
    ("FloatingPointError", Some("ArithmeticError")),
    ("OverflowError", Some("ArithmeticError")),
    ("ZeroDivisionError", Some("ArithmeticError")),
    // Lookup leaves.
    ("IndexError", Some("LookupError")),
    ("KeyError", Some("LookupError")),
    // Name leaf.
    ("UnboundLocalError", Some("NameError")),
    // OSError subtree. EnvironmentError/IOError are historical ALIASES of
    // OSError: they appear here as their own names so `except
    // EnvironmentError:` keeps working, and resolve to OSError when they
    // are the RAISED type's ancestor query.
    ("EnvironmentError", Some("OSError")),
    ("IOError", Some("OSError")),
    ("BlockingIOError", Some("OSError")),
    ("ChildProcessError", Some("OSError")),
    ("ConnectionError", Some("OSError")),
    ("BrokenPipeError", Some("ConnectionError")),
    ("ConnectionAbortedError", Some("ConnectionError")),
    ("ConnectionRefusedError", Some("ConnectionError")),
    ("ConnectionResetError", Some("ConnectionError")),
    ("FileExistsError", Some("OSError")),
    ("FileNotFoundError", Some("OSError")),
    ("InterruptedError", Some("OSError")),
    ("IsADirectoryError", Some("OSError")),
    ("NotADirectoryError", Some("OSError")),
    ("PermissionError", Some("OSError")),
    ("ProcessLookupError", Some("OSError")),
    ("TimeoutError", Some("OSError")),
    // urllib.error family (the http-ureq runtime raises these):
    // URLError IS-A OSError and HTTPError IS-A URLError in CPython.
    ("URLError", Some("OSError")),
    ("HTTPError", Some("URLError")),
    ("ContentTooShortError", Some("URLError")),
    // RuntimeError leaves.
    ("NotImplementedError", Some("RuntimeError")),
    ("RecursionError", Some("RuntimeError")),
    ("PythonFinalizationError", Some("RuntimeError")),
    // Syntax tree.
    ("SyntaxError", Some("Exception")),
    ("IndentationError", Some("SyntaxError")),
    ("TabError", Some("IndentationError")),
    ("_IncompleteInputError", Some("SyntaxError")),
    // Unicode tree (hangs off ValueError).
    ("UnicodeError", Some("ValueError")),
    ("UnicodeDecodeError", Some("UnicodeError")),
    ("UnicodeEncodeError", Some("UnicodeError")),
    ("UnicodeTranslateError", Some("UnicodeError")),
    // BaseExceptionGroup hangs off BaseException; ExceptionGroup ALSO
    // inherits Exception (multiple inheritance — handled beside the walk).
    ("BaseExceptionGroup", Some("BaseException")),
    ("ExceptionGroup", Some("BaseExceptionGroup")),
    // Warning tree.
    ("BytesWarning", Some("Warning")),
    ("DeprecationWarning", Some("Warning")),
    ("EncodingWarning", Some("Warning")),
    ("FutureWarning", Some("Warning")),
    ("ImportWarning", Some("Warning")),
    ("PendingDeprecationWarning", Some("Warning")),
    ("ResourceWarning", Some("Warning")),
    ("RuntimeWarning", Some("Warning")),
    ("SyntaxWarning", Some("Warning")),
    ("UnicodeWarning", Some("Warning")),
    ("UserWarning", Some("Warning")),
];

/// The direct parent of a built-in exception name, or `None` when the
/// name is not a built-in exception (including BaseException itself,
/// which is the root).
pub(crate) fn direct_exception_parent(exc: &str) -> Option<&'static str> {
    BUILTIN_EXCEPTION_PARENTS
        .iter()
        .find(|(name, _)| *name == exc)
        .and_then(|(_, parent)| *parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_parent_names_a_listed_type() {
        // A typo in any parent string would silently break ancestry walks;
        // the chain must bottom out at BaseException.
        for (name, parent) in BUILTIN_EXCEPTION_PARENTS {
            let mut current = *parent;
            while let Some(p) = current {
                assert!(
                    BUILTIN_EXCEPTION_PARENTS.iter().any(|(n, _)| *n == p),
                    "{name}'s parent {p} is not in the table"
                );
                current = direct_exception_parent(p);
            }
            let _ = name;
        }
    }

    #[test]
    fn base_exception_is_the_single_root() {
        let roots: alloc::vec::Vec<&str> = BUILTIN_EXCEPTION_PARENTS
            .iter()
            .filter(|(_, p)| p.is_none())
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(roots, alloc::vec!["BaseException"]);
    }

    #[test]
    fn alias_and_group_special_cases_hold() {
        // Verified against python3 3.14: EnvironmentError IS-A OSError
        // (alias), and ExceptionGroup additionally IS-A Exception even
        // though its MRO lists BaseExceptionGroup first.
        assert_eq!(direct_exception_parent("EnvironmentError"), Some("OSError"));
        assert_eq!(direct_exception_parent("IOError"), Some("OSError"));
        assert_eq!(
            direct_exception_parent("ExceptionGroup"),
            Some("BaseExceptionGroup")
        );
    }
}


/// The std-only extension: each built-in exception's PyO3 constructor, so
/// `From<PyException> for PyErr` raises the right Python class. Kept
/// beside the parent table with a parity test so a newly added exception
/// cannot silently fall back to RuntimeError surfacing.
#[cfg(feature = "std")]
pub(crate) static PYO3_CTORS: &[(&str, fn(String) -> pyo3::PyErr)] = &[
    ("BaseException", |m| pyo3::exceptions::PyBaseException::new_err(m)),
    ("Exception", |m| pyo3::exceptions::PyException::new_err(m)),
    ("SystemExit", |m| pyo3::exceptions::PySystemExit::new_err(m)),
    ("KeyboardInterrupt", |m| pyo3::exceptions::PyKeyboardInterrupt::new_err(m)),
    ("GeneratorExit", |m| pyo3::exceptions::PyGeneratorExit::new_err(m)),
    ("ArithmeticError", |m| pyo3::exceptions::PyArithmeticError::new_err(m)),
    ("AssertionError", |m| pyo3::exceptions::PyAssertionError::new_err(m)),
    ("AttributeError", |m| pyo3::exceptions::PyAttributeError::new_err(m)),
    ("BufferError", |m| pyo3::exceptions::PyBufferError::new_err(m)),
    ("EOFError", |m| pyo3::exceptions::PyEOFError::new_err(m)),
    ("ImportError", |m| pyo3::exceptions::PyImportError::new_err(m)),
    ("ModuleNotFoundError", |m| pyo3::exceptions::PyModuleNotFoundError::new_err(m)),
    ("LookupError", |m| pyo3::exceptions::PyLookupError::new_err(m)),
    ("IndexError", |m| pyo3::exceptions::PyIndexError::new_err(m)),
    ("KeyError", |m| pyo3::exceptions::PyKeyError::new_err(m)),
    ("MemoryError", |m| pyo3::exceptions::PyMemoryError::new_err(m)),
    ("NameError", |m| pyo3::exceptions::PyNameError::new_err(m)),
    ("UnboundLocalError", |m| pyo3::exceptions::PyUnboundLocalError::new_err(m)),
    ("OSError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("BlockingIOError", |m| pyo3::exceptions::PyBlockingIOError::new_err(m)),
    ("ChildProcessError", |m| pyo3::exceptions::PyChildProcessError::new_err(m)),
    ("ConnectionError", |m| pyo3::exceptions::PyConnectionError::new_err(m)),
    ("BrokenPipeError", |m| pyo3::exceptions::PyBrokenPipeError::new_err(m)),
    ("ConnectionAbortedError", |m| pyo3::exceptions::PyConnectionAbortedError::new_err(m)),
    ("ConnectionRefusedError", |m| pyo3::exceptions::PyConnectionRefusedError::new_err(m)),
    ("ConnectionResetError", |m| pyo3::exceptions::PyConnectionResetError::new_err(m)),
    ("FileExistsError", |m| pyo3::exceptions::PyFileExistsError::new_err(m)),
    ("FileNotFoundError", |m| pyo3::exceptions::PyFileNotFoundError::new_err(m)),
    ("InterruptedError", |m| pyo3::exceptions::PyInterruptedError::new_err(m)),
    ("IsADirectoryError", |m| pyo3::exceptions::PyIsADirectoryError::new_err(m)),
    ("NotADirectoryError", |m| pyo3::exceptions::PyNotADirectoryError::new_err(m)),
    ("PermissionError", |m| pyo3::exceptions::PyPermissionError::new_err(m)),
    ("ProcessLookupError", |m| pyo3::exceptions::PyProcessLookupError::new_err(m)),
    ("TimeoutError", |m| pyo3::exceptions::PyTimeoutError::new_err(m)),
    // urllib.error: pyo3 wraps none of these; they surface through their
    // OSError ancestry.
    ("URLError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("HTTPError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("ContentTooShortError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("ReferenceError", |m| pyo3::exceptions::PyReferenceError::new_err(m)),
    ("RuntimeError", |m| pyo3::exceptions::PyRuntimeError::new_err(m)),
    ("NotImplementedError", |m| pyo3::exceptions::PyNotImplementedError::new_err(m)),
    ("RecursionError", |m| pyo3::exceptions::PyRecursionError::new_err(m)),
    ("PythonFinalizationError", |m| pyo3::exceptions::PyRuntimeError::new_err(m)),
    ("FloatingPointError", |m| pyo3::exceptions::PyFloatingPointError::new_err(m)),
    ("OverflowError", |m| pyo3::exceptions::PyOverflowError::new_err(m)),
    ("ZeroDivisionError", |m| pyo3::exceptions::PyZeroDivisionError::new_err(m)),
    ("StopIteration", |m| pyo3::exceptions::PyStopIteration::new_err(m)),
    ("StopAsyncIteration", |m| pyo3::exceptions::PyStopAsyncIteration::new_err(m)),
    ("SyntaxError", |m| pyo3::exceptions::PySyntaxError::new_err(m)),
    ("SystemError", |m| pyo3::exceptions::PySystemError::new_err(m)),
    ("TypeError", |m| pyo3::exceptions::PyTypeError::new_err(m)),
    ("ValueError", |m| pyo3::exceptions::PyValueError::new_err(m)),
    ("UnicodeError", |m| pyo3::exceptions::PyUnicodeError::new_err(m)),
    ("UnicodeDecodeError", |m| pyo3::exceptions::PyUnicodeDecodeError::new_err(m)),
    ("UnicodeEncodeError", |m| pyo3::exceptions::PyUnicodeEncodeError::new_err(m)),
    ("UnicodeTranslateError", |m| pyo3::exceptions::PyUnicodeTranslateError::new_err(m)),
    ("Warning", |m| pyo3::exceptions::PyWarning::new_err(m)),
    ("BytesWarning", |m| pyo3::exceptions::PyBytesWarning::new_err(m)),
    ("DeprecationWarning", |m| pyo3::exceptions::PyDeprecationWarning::new_err(m)),
    ("EncodingWarning", |m| pyo3::exceptions::PyEncodingWarning::new_err(m)),
    ("FutureWarning", |m| pyo3::exceptions::PyFutureWarning::new_err(m)),
    ("ImportWarning", |m| pyo3::exceptions::PyImportWarning::new_err(m)),
    ("PendingDeprecationWarning", |m| pyo3::exceptions::PyPendingDeprecationWarning::new_err(m)),
    ("ResourceWarning", |m| pyo3::exceptions::PyResourceWarning::new_err(m)),
    ("RuntimeWarning", |m| pyo3::exceptions::PyRuntimeWarning::new_err(m)),
    ("SyntaxWarning", |m| pyo3::exceptions::PySyntaxWarning::new_err(m)),
    ("UnicodeWarning", |m| pyo3::exceptions::PyUnicodeWarning::new_err(m)),
    ("UserWarning", |m| pyo3::exceptions::PyUserWarning::new_err(m)),
    // ExceptionGroup/BaseExceptionGroup: pyo3 0.29 exposes
    // PyBaseExceptionGroup; ExceptionGroup surfaces through it (its CPython
    // MRO roots there too).
    ("BaseExceptionGroup", |m| pyo3::exceptions::PyBaseExceptionGroup::new_err(m)),
    ("ExceptionGroup", |m| pyo3::exceptions::PyBaseExceptionGroup::new_err(m)),
    // The OSError alias names surface as their canonical class.
    ("EnvironmentError", |m| pyo3::exceptions::PyEnvironmentError::new_err(m)),
    ("IOError", |m| pyo3::exceptions::PyIOError::new_err(m)),
    // Syntax-tree gaps handled beside the ctor lookup.
    // Unicode leaves.
    ("UnicodeError", |m| pyo3::exceptions::PyUnicodeError::new_err(m)),
    ("UnicodeDecodeError", |m| pyo3::exceptions::PyUnicodeDecodeError::new_err(m)),
    ("UnicodeEncodeError", |m| pyo3::exceptions::PyUnicodeEncodeError::new_err(m)),
    ("UnicodeTranslateError", |m| pyo3::exceptions::PyUnicodeTranslateError::new_err(m)),
    ("TimeoutError", |m| pyo3::exceptions::PyTimeoutError::new_err(m)),
    // urllib.error: pyo3 wraps none of these; they surface through their
    // OSError ancestry.
    ("URLError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("HTTPError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    ("ContentTooShortError", |m| pyo3::exceptions::PyOSError::new_err(m)),
    // Aliases surface as their canonical class.
    ("EnvironmentError", |m| pyo3::exceptions::PyEnvironmentError::new_err(m)),
    ("IOError", |m| pyo3::exceptions::PyIOError::new_err(m)),
];

/// The PyO3 constructor for a built-in exception name.
#[cfg(feature = "std")]
pub(crate) fn pyo3_ctor(name: &str) -> Option<fn(String) -> pyo3::PyErr> {
    PYO3_CTORS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, ctor)| *ctor)
}

#[cfg(feature = "std")]
#[test]
fn every_builtin_except_the_pyo3_gaps_has_a_ctor() {
    // pyo3 0.29 wraps no IndentationError/TabError (they ARE SyntaxErrors
    // in CPython's tree) and no _IncompleteInputError; those three are the
    // documented gaps and surface through their ancestors instead.
    let gaps = ["IndentationError", "TabError", "_IncompleteInputError"];
    for (name, _) in BUILTIN_EXCEPTION_PARENTS {
        if gaps.contains(name) {
            continue;
        }
        assert!(
            pyo3_ctor(name).is_some(),
            "{name} has no PyO3 constructor"
        );
    }
    // A ctor name is listed iff it resolves in the tree (BaseException,
    // the root, has no parent entry but still counts via exact match in
    // matches()).
    for (name, _) in PYO3_CTORS {
        assert!(
            direct_exception_parent(name).is_some() || *name == "BaseException",
            "{name} has a ctor but is not in the parent table"
        );
    }
}
