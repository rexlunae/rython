//! Python os module implementation
//! 
//! This module provides Python's os module functionality for
//! operating system interface functions.
//!
//! Note: This module is only available with the `std` feature enabled,
//! as it requires operating system functionality.

use crate::{PyException, AsStrLike, AsPathLike, python_function};
use std::collections::HashMap;
use std::sync::LazyLock;

python_function! {
    /// os.execv - execute a program (generic version using traits)
    /// 
    /// # Arguments
    /// * `program` - Path to the program to execute (any string-like type)
    /// * `args` - Arguments to pass to the program (any collection of string-like types)
    /// 
    /// # Note
    /// This function replaces the current process with the new program.
    pub fn execv_mixed<P, A, S>(program: P, args: A) -> Result<(), PyException> 
    where [P: AsPathLike, A: IntoIterator<Item = S>, S: AsStrLike]
    [signature: (program, args)]
    [concrete_types: (String, Vec<String>) -> Result<(), crate::PyException>]
    {
        // Convert to owned strings first to avoid lifetime issues
        let owned_args: Vec<String> = args.into_iter().map(|s| s.as_str_like().to_string()).collect();
        let str_args: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();
        execv(program.as_path_like(), str_args)
    }
}

/// os.execv - execute a program
/// 
/// # Arguments
/// * `program` - Path to the program to execute
/// * `args` - Arguments to pass to the program
/// 
/// # Note
/// This function replaces the current process with the new program.
/// On Unix systems, this uses the actual execv system call.
/// On Windows, this is simulated using process spawn and exit.
#[cfg(unix)]
pub fn execv<P: AsRef<str>, A: AsRef<str>>(program: P, args: Vec<A>) -> Result<(), PyException> {
    use std::ffi::CString;
    
    // Convert program path and arguments to C strings
    let program_c = CString::new(program.as_ref())
        .map_err(|_| crate::value_error("Invalid program path"))?;
    
    let mut args_c: Vec<CString> = Vec::new();
    for arg in &args {
        args_c.push(CString::new(arg.as_ref())
            .map_err(|_| crate::value_error("Invalid argument"))?);
    }
    
    // Convert to raw pointers for execv
    let mut args_ptr: Vec<*const libc::c_char> = args_c.iter()
        .map(|s| s.as_ptr())
        .collect();
    args_ptr.push(std::ptr::null()); // execv expects null-terminated array
    
    // Call execv - this replaces the current process
    unsafe {
        libc::execv(program_c.as_ptr(), args_ptr.as_ptr());
    }
    
    // If we reach here, execv failed
    Err(crate::runtime_error(format!("execv failed for program: {}", program.as_ref())))
}

/// os.execv - execute a program (Windows implementation)
#[cfg(windows)]
pub fn execv<P: AsRef<str>, A: AsRef<str>>(program: P, args: Vec<A>) -> Result<(), PyException> {
    use std::process::Command;
    
    // On Windows, we simulate execv using process spawn + exit
    let mut cmd = Command::new(program.as_ref());
    cmd.args(args.iter().map(|a| a.as_ref()));
    
    match cmd.status() {
        Ok(status) => {
            std::process::exit(status.code().unwrap_or(1));
        }
        Err(e) => Err(crate::runtime_error(format!("Failed to execute program {}: {}", program.as_ref(), e)))
    }
}

python_function! {
    /// os.getenv - get environment variable
    pub fn getenv<K>(key: K) -> Option<String>
    where [K: AsStrLike]
    [signature: (key)]
    [concrete_types: (String) -> Option<String>]
    {
        std::env::var(key.as_str_like()).ok()
    }
}

python_function! {
    /// os.setenv - set environment variable
    pub fn setenv<K, V>(key: K, value: V) -> ()
    where [K: AsStrLike, V: AsStrLike]
    [signature: (key, value)]
    [concrete_types: (String, String) -> ()]
    {
        unsafe {
            std::env::set_var(key.as_str_like(), value.as_str_like());
        }
    }
}

python_function! {
    /// os.getcwd - get current working directory
    pub fn getcwd() -> Result<String, PyException>
    [signature: ()]
    [concrete_types: () -> Result<String, crate::PyException>]
    {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|e| crate::runtime_error(&format!("Failed to get current directory: {}", e)))
    }
}

python_function! {
    /// os.chdir - change current working directory
    pub fn chdir<P>(path: P) -> Result<(), PyException>
    where [P: AsPathLike]
    [signature: (path)]
    [concrete_types: (String) -> Result<(), crate::PyException>]
    {
        std::env::set_current_dir(path.as_path_like())
            .map_err(|e| crate::runtime_error(&format!("Failed to change directory to {}: {}", path.as_path_like(), e)))
    }
}

python_function! {
    /// os.putenv - set environment variable (Python exposes both putenv
    /// and the os.environ mapping; both mutate the process environment).
    pub fn putenv<K, V>(key: K, value: V) -> ()
    where [K: AsStrLike, V: AsStrLike]
    [signature: (key, value)]
    [concrete_types: (String, String) -> ()]
    {
        setenv(key, value)
    }
}

/// os.environ - a LIVE view of the process environment. Python's
/// os.environ reflects later setenv/putenv mutations; a snapshot taken at
/// first access would silently disagree with os.getenv.
#[derive(Clone, Copy, Debug)]
pub struct Environ;

#[allow(non_upper_case_globals)]
pub static environ: Environ = Environ;

impl Environ {
    /// dict.get semantics: value or None.
    pub fn py_get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }

    /// dict.get with a default.
    pub fn py_get_default(&self, key: &str, default: impl Into<String>) -> String {
        std::env::var(key).unwrap_or_else(|_| default.into())
    }

    pub fn py_contains(&self, key: &str) -> bool {
        std::env::var(key).is_ok()
    }

    pub fn py_keys(&self) -> Vec<String> {
        std::env::vars().map(|(k, _)| k).collect()
    }

    pub fn py_values(&self) -> Vec<String> {
        std::env::vars().map(|(_, v)| v).collect()
    }

    pub fn py_items(&self) -> Vec<(String, String)> {
        std::env::vars().collect()
    }

    /// A point-in-time snapshot as a plain map, for callers that need one.
    pub fn snapshot(&self) -> HashMap<String, String> {
        std::env::vars().collect()
    }
}

/// `os.environ[key]` raises KeyError for missing variables, like Python.
impl crate::PyIndex<&str> for Environ {
    type Output = String;
    fn py_index(&self, index: &str) -> Result<String, PyException> {
        std::env::var(index)
            .map_err(|_| PyException::new("KeyError", &format!("'{}'", index)))
    }
}

impl crate::PyIndex<String> for Environ {
    type Output = String;
    fn py_index(&self, index: String) -> Result<String, PyException> {
        crate::PyIndex::<&str>::py_index(self, index.as_str())
    }
}

/// os.name - operating system name
/// 
/// This provides the name of the operating system, similar to Python's os.name.
pub static name: LazyLock<&'static str> = LazyLock::new(|| {
    if cfg!(target_os = "windows") {
        "nt"
    } else {
        "posix"
    }
});

// Compatibility aliases for generated code
// Note: Functions are already public in this module, no need to re-export

/// os.path submodule
pub mod path {
    //! Python os.path module implementation
    //! 
    //! This submodule provides path manipulation functions using Rust's std::path.

    use std::path::{Path, PathBuf};
    use crate::{PyException, AsPathLike, python_function};
    
    /// os.path.sep - path separator for the current platform
    pub static sep: &str = if cfg!(target_os = "windows") { "\\" } else { "/" };

    python_function! {
        /// os.path.dirname - everything before the final slash, following
        /// posixpath exactly: dirname("/") is "/", dirname("abc") is "",
        /// dirname("a/b/") is "a/b".
        pub fn dirname<P>(path: P) -> String
        where [P: AsPathLike]
        [signature: (path)]
        [concrete_types: (String) -> String]
        {
            let p = path.as_path_like();
            let i = p.rfind('/').map(|i| i + 1).unwrap_or(0);
            let head = &p[..i];
            if !head.is_empty() && head.bytes().any(|b| b != b'/') {
                head.trim_end_matches('/').to_string()
            } else {
                head.to_string()
            }
        }
    }

    python_function! {
        /// os.path.basename - everything after the final slash, following
        /// posixpath exactly: basename("dir/") is "" (not "dir").
        pub fn basename<P>(path: P) -> String
        where [P: AsPathLike]
        [signature: (path)]
        [concrete_types: (String) -> String]
        {
            let p = path.as_path_like();
            let i = p.rfind('/').map(|i| i + 1).unwrap_or(0);
            p[i..].to_string()
        }
    }

    python_function! {
        /// os.path.normpath - collapse redundant separators and up-level
        /// references LEXICALLY (posixpath.normpath: never touches the
        /// filesystem, so "A/foo/../B" is "A/B" even through symlinks).
        pub fn normpath<P>(path: P) -> String
        where [P: AsPathLike]
        [signature: (path)]
        [concrete_types: (String) -> String]
        {
            let p = path.as_path_like();
            if p.is_empty() {
                return ".".to_string();
            }
            // POSIX treats exactly two leading slashes as meaningful.
            let initial_slashes = if p.starts_with("//") && !p.starts_with("///") {
                2
            } else if p.starts_with('/') {
                1
            } else {
                0
            };
            let mut comps: Vec<&str> = Vec::new();
            for comp in p.split('/') {
                if comp.is_empty() || comp == "." {
                    continue;
                }
                if comp != ".."
                    || (initial_slashes == 0 && comps.is_empty())
                    || comps.last() == Some(&"..")
                {
                    comps.push(comp);
                } else if !comps.is_empty() {
                    comps.pop();
                }
            }
            let mut out = "/".repeat(initial_slashes);
            out.push_str(&comps.join("/"));
            if out.is_empty() {
                ".".to_string()
            } else {
                out
            }
        }
    }
    
    python_function! {
        /// os.path.join - join path components
        pub fn join<P1, P2>(path1: P1, path2: P2) -> String
        where [P1: AsPathLike, P2: AsPathLike]
        [signature: (path1, path2)]
        [concrete_types: (String, String) -> String]
        {
            let mut path = PathBuf::from(path1.as_path_like());
            path.push(path2.as_path_like());
            path.to_string_lossy().to_string()
        }
    }
    
    python_function! {
        /// os.path.join - join path components (3 arguments version)
        pub fn join3<P1, P2, P3>(path1: P1, path2: P2, path3: P3) -> String
        where [P1: AsPathLike, P2: AsPathLike, P3: AsPathLike]
        [signature: (path1, path2, path3)]
        [concrete_types: (String, String, String) -> String]
        {
            let mut path = PathBuf::from(path1.as_path_like());
            path.push(path2.as_path_like());
            path.push(path3.as_path_like());
            path.to_string_lossy().to_string()
        }
    }
    
    python_function! {
        /// os.path.join - join path components (variable arguments version)
        pub fn join_many<I, P>(components: I) -> String
        where [I: IntoIterator<Item = P>, P: AsPathLike]
        [signature: (components)]
        [concrete_types: (Vec<String>) -> String]
        {
            let mut path = PathBuf::new();
            for component in components {
                path.push(component.as_path_like());
            }
            path.to_string_lossy().to_string()
        }
    }
    
    /// os.path.join - variadic version for compatibility with Python's os.path.join
    /// 
    /// This function accepts individual arguments like Python's os.path.join(a, b, c, ...)
    /// 
    /// # Arguments
    /// * `first` - First path component
    /// * `rest` - Additional path components (variadic)
    /// 
    /// # Returns
    /// The joined path as a String
    pub fn join_paths<P: AsPathLike>(first: P, rest: &[P]) -> String {
        let mut path = PathBuf::from(first.as_path_like());
        for component in rest {
            path.push(component.as_path_like());
        }
        path.to_string_lossy().to_string()
    }
    
    python_function! {
        /// os.path.exists - check if path exists
        pub fn exists<P>(path: P) -> bool
        where [P: AsPathLike]
        [signature: (path)]
        [concrete_types: (String) -> bool]
        {
            Path::new(path.as_path_like()).exists()
        }
    }
    
    python_function! {
        /// os.path.isfile - check if path is a regular file
        pub fn isfile<P>(path: P) -> bool
        where [P: AsRef<str>]
        [signature: (path)]
        [concrete_types: (String) -> bool]
        {
            Path::new(path.as_ref()).is_file()
        }
    }
    
    python_function! {
        /// os.path.isdir - check if path is a directory
        pub fn isdir<P>(path: P) -> bool
        where [P: AsRef<str>]
        [signature: (path)]
        [concrete_types: (String) -> bool]
        {
            Path::new(path.as_ref()).is_dir()
        }
    }
    
    python_function! {
        /// os.path.abspath - normpath(join(cwd, path)), purely LEXICAL as
        /// in Python: the path need not exist and symlinks are not
        /// resolved (std::fs::canonicalize would do both, silently
        /// diverging).
        pub fn abspath<P>(path: P) -> Result<String, PyException>
        where [P: AsRef<str>]
        [signature: (path)]
        [concrete_types: (String) -> Result<String, crate::PyException>]
        {
            let p = path.as_ref();
            if p.starts_with('/') {
                return Ok(normpath(p));
            }
            let cwd = std::env::current_dir()
                .map_err(|e| crate::runtime_error(format!("Failed to get current directory: {}", e)))?;
            Ok(normpath(&format!("{}/{}", cwd.to_string_lossy(), p)))
        }
    }

    python_function! {
        /// os.path.relpath - the relative path from start to path,
        /// computed lexically with `..` traversal like posixpath:
        /// relpath("/a/b", "/a/c") is "../b".
        pub fn relpath<P>(path: P, start: Option<String>) -> Result<String, PyException>
        where [P: AsRef<str>]
        [signature: (path, start=None)]
        [concrete_types: (String, Option<String>) -> Result<String, crate::PyException>]
        {
            let p = path.as_ref();
            if p.is_empty() {
                return Err(crate::value_error("no path specified"));
            }
            let start = start.unwrap_or_else(|| ".".to_string());
            let abs_path = abspath(p)?;
            let abs_start = abspath(start.as_str())?;
            let path_list: Vec<&str> =
                abs_path.split('/').filter(|c| !c.is_empty()).collect();
            let start_list: Vec<&str> =
                abs_start.split('/').filter(|c| !c.is_empty()).collect();
            let common = path_list
                .iter()
                .zip(start_list.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let mut rel: Vec<&str> = Vec::new();
            for _ in common..start_list.len() {
                rel.push("..");
            }
            rel.extend(&path_list[common..]);
            if rel.is_empty() {
                Ok(".".to_string())
            } else {
                Ok(rel.join("/"))
            }
        }
    }
}