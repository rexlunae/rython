//! SystemExit carries its original code until the program boundary.

use alloc::{format, string::String, sync::Arc, vec, vec::Vec};

use crate::{PyDisplay, PyException, PyRepr, PyValue, builtin_exceptions::BuiltinException};

impl PyException {
    /// Construct SystemExit with BaseException's positional arguments.
    /// `code` is None, the one argument, or the arguments tuple.
    pub fn system_exit(args: Vec<PyValue>) -> Self {
        let code = match args.as_slice() {
            [] => PyValue::None_,
            [one] => one.clone(),
            _ => PyValue::Tuple(Arc::new(args.clone())),
        };
        let message = if args.is_empty() {
            String::new()
        } else {
            code.py_display()
        };
        let reprs = args.iter().map(PyRepr::py_repr).collect();
        Self::new_with_attrs("SystemExit", message, vec![("code".into(), code)])
            .with_args_repr(reprs)
    }

    /// Read a stored exception attribute without inventing a default.
    pub fn attr_value(&self, name: &str) -> Result<PyValue, PyException> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
            .ok_or_else(|| {
                PyException::new(
                    "AttributeError",
                    format!(
                        "'{}' object has no attribute '{}'",
                        self.exception_type, name,
                    ),
                )
            })
    }

    /// An uncaught SystemExit's process status and optional stderr line.
    /// Kept free of OS calls so catching/re-raising works in the alloc tier.
    pub fn system_exit_status(&self) -> Result<Option<(i32, Option<String>)>, PyException> {
        if !self.matches_builtin(BuiltinException::SystemExit) {
            return Ok(None);
        }
        Ok(Some(match self.attr_value("code")? {
            PyValue::None_ => (0, None),
            PyValue::Int(code) => (code as i32, None),
            PyValue::Bool(code) => (i32::from(code), None),
            other => (1, Some(other.py_display())),
        }))
    }
}

#[cfg(feature = "pyo3-interop")]
impl PyException {
    /// Preserve SystemExit.code and args when crossing back into CPython.
    pub(crate) fn system_exit_pyerr(&self) -> Option<pyo3::PyErr> {
        use pyo3::{
            IntoPyObjectExt, Python,
            exceptions::PySystemExit,
            types::{PyBytes, PyTuple},
        };

        fn value(py: Python<'_>, item: &PyValue) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            Ok(match item {
                PyValue::None_ => py.None(),
                PyValue::Int(n) => n.into_py_any(py)?,
                PyValue::Bool(b) => b.into_py_any(py)?,
                PyValue::Float(n) => n.into_py_any(py)?,
                PyValue::Str(s) => s.into_py_any(py)?,
                PyValue::Bytes(bytes) => PyBytes::new(py, bytes).into_any().unbind(),
                PyValue::Tuple(parts) => {
                    let parts = parts
                        .iter()
                        .map(|part| value(py, part))
                        .collect::<pyo3::PyResult<Vec<_>>>()?;
                    PyTuple::new(py, parts)?.into_any().unbind()
                }
                _ => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "rython cannot marshal this SystemExit code into CPython",
                    ));
                }
            })
        }

        if !self.matches_builtin(BuiltinException::SystemExit) {
            return None;
        }
        Some(Python::attach(|py| {
            let convert = || -> pyo3::PyResult<pyo3::PyErr> {
                let code = self.attr_value("code").map_err(pyo3::PyErr::from)?;
                let count = self.args_repr.as_ref().map(Vec::len).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "rython SystemExit has no recorded arguments",
                    )
                })?;
                Ok(if count == 0 {
                    PySystemExit::new_err(())
                } else if count == 1 {
                    PySystemExit::new_err((value(py, &code)?,))
                } else {
                    PySystemExit::new_err(value(py, &code)?)
                })
            };
            convert().unwrap_or_else(|error| error)
        }))
    }
}
