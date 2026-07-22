//! Options for Python compilation.

use std::{
    collections::{BTreeMap, HashSet},
    default::Default,
};

use pyo3::{prelude::*, PyResult};
use std::ffi::CString;

/// Supported async runtimes for Python async code generation
#[derive(Clone, Debug, PartialEq)]
pub enum AsyncRuntime {
    /// Tokio runtime (default)
    Tokio,
    /// async-std runtime
    AsyncStd,
    /// smol runtime
    Smol,
    /// Custom runtime with specified attribute and import
    Custom {
        /// The attribute to use (e.g., "tokio::main", "async_std::main")
        attribute: String,
        /// The import to add (e.g., "tokio", "async_std")
        import: String,
    },
}

impl Default for AsyncRuntime {
    fn default() -> Self {
        AsyncRuntime::Tokio
    }
}

impl AsyncRuntime {
    /// Get the attribute string for the async main function
    pub fn main_attribute(&self) -> &str {
        match self {
            AsyncRuntime::Tokio => "tokio::main",
            AsyncRuntime::AsyncStd => "async_std::main",
            AsyncRuntime::Smol => "smol::main",
            AsyncRuntime::Custom { attribute, .. } => attribute,
        }
    }

    /// Get the import string for the runtime
    pub fn import(&self) -> &str {
        match self {
            AsyncRuntime::Tokio => "tokio",
            AsyncRuntime::AsyncStd => "async_std",
            AsyncRuntime::Smol => "smol",
            AsyncRuntime::Custom { import, .. } => import,
        }
    }
}

pub fn sys_path() -> PyResult<Vec<String>> {
    let pymodule_code = include_str!("path.py");

    Python::attach(|py| -> PyResult<Vec<String>> {
        let code_cstr = CString::new(pymodule_code)?;
        let pymodule = PyModule::from_code(py, &code_cstr, c"path.py", c"path")?;
        let t = pymodule.getattr("path")?;
        assert!(t.is_callable());
        let args = ();
        let paths: Vec<String> = t.call1(args)?.extract()?;

        Ok(paths)
    })
}

/// The global context for Python compilation.
#[derive(Clone, Debug)]
pub struct PythonOptions {
    /// Python imports are mapped into a given namespace that can be changed.
    pub python_namespace: String,

    /// The default path we will search for Python modules.
    pub python_path: Vec<String>,

    /// Collects all of the things we need to compile imports[module][asnames]
    pub imports: BTreeMap<String, HashSet<String>>,


    pub stdpython: String,
    pub with_std_python: bool,

    pub allow_unsafe: bool,

    /// The async runtime to use for async Python code
    pub async_runtime: AsyncRuntime,
}

impl Default for PythonOptions {
    fn default() -> Self {
        Self {
            python_namespace: String::from("__python_namespace__"),
            // Default must not panic: fall back to an empty search path if
            // the embedded interpreter can't report sys.path.
            python_path: sys_path().unwrap_or_else(|e| {
                tracing::warn!("could not read Python sys.path: {}; using empty path", e);
                Vec::new()
            }),
            imports: BTreeMap::new(),

            stdpython: "stdpython".to_string(),
            with_std_python: true,
            allow_unsafe: false,
            async_runtime: AsyncRuntime::default(),
        }
    }
}

impl PythonOptions {
    /// Create PythonOptions with tokio runtime (default)
    pub fn with_tokio() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Tokio;
        options
    }

    /// Create PythonOptions with async-std runtime
    pub fn with_async_std() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::AsyncStd;
        options
    }

    /// Create PythonOptions with smol runtime
    pub fn with_smol() -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Smol;
        options
    }

    /// Create PythonOptions with a custom async runtime
    pub fn with_custom_runtime(attribute: impl Into<String>, import: impl Into<String>) -> Self {
        let mut options = Self::default();
        options.async_runtime = AsyncRuntime::Custom {
            attribute: attribute.into(),
            import: import.into(),
        };
        options
    }

    /// Set the async runtime for these options
    pub fn set_async_runtime(&mut self, runtime: AsyncRuntime) -> &mut Self {
        self.async_runtime = runtime;
        self
    }
}
