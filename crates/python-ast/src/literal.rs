//! Python values read by CPython itself, for tools that take Python text
//! from a command line (the JIT's `rypip jit call FILE FUNCTION ARG...`,
//! issue #166): an argument means exactly what `ast.literal_eval` says it
//! means — escapes, underscores, `-3`, `0x10` — because CPython reads it.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

/// A Python literal's value as `ast.literal_eval` produced it, sorted into
/// the scalar types a caller can use and everything else (named by its
/// Python type, so the caller can refuse it loudly).
#[derive(Debug, Clone, PartialEq)]
pub enum PyLiteral {
    None,
    Bool(bool),
    Int(i64),
    /// An `int` outside the signed 64-bit range, as its `repr`.
    IntOutOfRange(String),
    Float(f64),
    Str(String),
    /// Any other literal (`b'x'`, `[1]`, `(1,)`, `1j`, ...), by
    /// `type(value).__name__`.
    Other(String),
}

/// Evaluate `source` with CPython's `ast.literal_eval`. A source that is
/// not a literal is an error carrying CPython's own exception line.
pub fn literal_eval(source: &str) -> Result<PyLiteral, String> {
    Python::attach(|py| {
        let value = py
            .import("ast")
            .and_then(|ast| ast.getattr("literal_eval"))
            .and_then(|eval| eval.call1((source,)))
            .map_err(|e| e.to_string())?;
        // `bool` before `int`: bool is an int subclass, and exact type
        // checks keep a subclass from passing as its base.
        let literal = if value.is_none() {
            PyLiteral::None
        } else if value.is_exact_instance_of::<PyBool>() {
            PyLiteral::Bool(value.extract::<bool>().map_err(|e| e.to_string())?)
        } else if value.is_exact_instance_of::<PyInt>() {
            match value.extract::<i64>() {
                Ok(n) => PyLiteral::Int(n),
                Err(_) => PyLiteral::IntOutOfRange(
                    value.repr().map_err(|e| e.to_string())?.to_string(),
                ),
            }
        } else if value.is_exact_instance_of::<PyFloat>() {
            PyLiteral::Float(value.extract::<f64>().map_err(|e| e.to_string())?)
        } else if value.is_exact_instance_of::<PyString>() {
            PyLiteral::Str(value.extract::<String>().map_err(|e| e.to_string())?)
        } else {
            PyLiteral::Other(
                value
                    .get_type()
                    .name()
                    .map_err(|e| e.to_string())?
                    .to_string(),
            )
        };
        Ok(literal)
    })
}

/// Whether `name` can be imported as a module under CPython: an
/// identifier (`str.isidentifier`) that is not a keyword
/// (`keyword.iskeyword`) — `import class` is a syntax error.
pub fn is_module_identifier(name: &str) -> Result<bool, String> {
    Python::attach(|py| {
        let identifier = PyString::new(py, name)
            .call_method0("isidentifier")
            .and_then(|r| r.extract::<bool>())
            .map_err(|e| e.to_string())?;
        let keyword = py
            .import("keyword")
            .and_then(|k| k.getattr("iskeyword"))
            .and_then(|f| f.call1((name,)))
            .and_then(|r| r.extract::<bool>())
            .map_err(|e| e.to_string())?;
        Ok(identifier && !keyword)
    })
}

/// The linked CPython's version string (`sys.version`), read WITHOUT
/// initializing the interpreter (`Py_GetVersion` is one of the
/// informative functions CPython documents as safe before
/// `Py_Initialize`): the parser is CPython's, so its version is part of
/// what a conversion depends on, and a cache keyed on conversions (the
/// JIT's) must not pay for interpreter startup to learn it.
pub fn python_version() -> String {
    // SAFETY: Py_GetVersion returns a pointer to a NUL-terminated static
    // buffer and needs no interpreter state.
    unsafe { std::ffi::CStr::from_ptr(pyo3::ffi::Py_GetVersion()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_read_as_cpython_reads_them() {
        // ast.literal_eval("-3") == -3; "1_000" == 1000; "0x10" == 16
        assert_eq!(literal_eval("-3"), Ok(PyLiteral::Int(-3)));
        assert_eq!(literal_eval("1_000"), Ok(PyLiteral::Int(1000)));
        assert_eq!(literal_eval("0x10"), Ok(PyLiteral::Int(16)));
        // True is a bool, not the int 1
        assert_eq!(literal_eval("True"), Ok(PyLiteral::Bool(true)));
        assert_eq!(literal_eval("None"), Ok(PyLiteral::None));
        assert_eq!(literal_eval("2.5"), Ok(PyLiteral::Float(2.5)));
        // "'a\\tb'" is the three characters a, TAB, b
        assert_eq!(literal_eval("'a\\tb'"), Ok(PyLiteral::Str("a\tb".to_string())));
        // 2**64 does not fit in 64 bits; repr(2**64) == '18446744073709551616'
        assert_eq!(
            literal_eval("18446744073709551616"),
            Ok(PyLiteral::IntOutOfRange("18446744073709551616".to_string()))
        );
        // type([1]).__name__ == 'list'; type(b'x').__name__ == 'bytes'
        assert_eq!(literal_eval("[1]"), Ok(PyLiteral::Other("list".to_string())));
        assert_eq!(literal_eval("b'x'"), Ok(PyLiteral::Other("bytes".to_string())));
        assert!(literal_eval("foo").is_err());
    }

    #[test]
    fn module_identifiers_are_cpython_identifiers_that_are_not_keywords() {
        // "prog".isidentifier() and not keyword.iskeyword("prog") → True;
        // "class" is a keyword; "my-prog" and "1x" are not identifiers.
        assert_eq!(is_module_identifier("prog"), Ok(true));
        assert_eq!(is_module_identifier("MyMod"), Ok(true));
        assert_eq!(is_module_identifier("class"), Ok(false));
        assert_eq!(is_module_identifier("my-prog"), Ok(false));
        assert_eq!(is_module_identifier("1x"), Ok(false));
    }

    #[test]
    fn the_python_version_is_read_without_starting_the_interpreter() {
        let version = python_version();
        assert!(version.starts_with('3'), "{version}");
    }
}
