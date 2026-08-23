//! Discovery of Python packages on disk: locating the source modules and
//! reading package metadata (name, version) from pyproject.toml when present.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// One Python module within a package.
#[derive(Debug, Clone)]
pub struct PyModule {
    /// Module path relative to the package root, e.g. `["utils", "text"]`
    /// for `utils/text.py`. Empty for `__init__.py` at the package root.
    pub path: Vec<String>,
    /// The module's Python source.
    pub source: String,
    /// The source file it came from (for error messages).
    pub file: PathBuf,
    /// Whether this is an `__init__.py`.
    pub is_init: bool,
}

/// A discovered Python package ready for conversion.
#[derive(Debug)]
pub struct PyPackage {
    /// Package name, sanitized to a valid crate/module identifier.
    pub name: String,
    /// Package version (from pyproject.toml / setup.cfg / setup.py, else
    /// "0.1.0").
    pub version: String,
    /// The package root directory: the project root (where pyproject.toml /
    /// rython.toml live), or the single file's parent for a bare file.
    pub root: PathBuf,
    pub modules: Vec<PyModule>,
    /// PEP 508 dependency requirements from the project's packaging
    /// metadata (`[project] dependencies`, `install_requires`), resolved by
    /// the converter against PyPI unless vendored via `[python-modules]`.
    pub dependencies: Vec<String>,
}

impl PyPackage {
    /// The module that should become the binary entry point: `__main__.py`,
    /// or any module whose source has an `if __name__ == "__main__"` block.
    pub fn entry_module(&self) -> Option<&PyModule> {
        self.modules
            .iter()
            .find(|m| m.path.last().map(String::as_str) == Some("__main__"))
            .or_else(|| {
                self.modules
                    .iter()
                    .find(|m| m.source.contains("__name__") && m.source.contains("__main__"))
            })
    }
}

/// Sanitize a package name into a valid Rust identifier.
pub fn sanitize_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if out.is_empty() {
        out.push_str("pypackage");
    }
    out
}

/// Discover a Python package at `path`, which may be a single `.py` file, a
/// package directory (with `__init__.py`), or a project directory resolved
/// the way Python resolves it — `pyproject.toml` (PEP 621 +
/// `[tool.setuptools]`), `setup.cfg`, or `setup.py` (executed via python3
/// when available) determine the name, version, packages, py-modules, and
/// dependencies. Projects without any packaging metadata fall back to the
/// historical layout heuristics (flat or `src/`).
pub fn discover(path: &Path) -> Result<PyPackage> {
    let path = path
        .canonicalize()
        .with_context(|| format!("cannot access {}", path.display()))?;

    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            bail!("{} is not a Python file", path.display());
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("invalid file name")?;
        let source =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        return Ok(PyPackage {
            name: sanitize_name(stem),
            version: "0.1.0".to_string(),
            root: path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            modules: vec![PyModule {
                path: vec![sanitize_name(stem)],
                source,
                file: path,
                is_init: false,
            }],
            dependencies: Vec::new(),
        });
    }

    // Project metadata, resolved the way Python's tooling does.
    let meta = crate::packaging::read_project_metadata(&path)?;
    let mut name = meta.name.clone();

    let mut modules = Vec::new();
    let mut found_metadata_packages = false;

    // Explicit packages / find_packages() / py-modules from the metadata.
    let package_dirs = crate::packaging::resolve_package_dirs(&path, &meta)?;
    if !package_dirs.is_empty() || !meta.py_modules.is_empty() {
        found_metadata_packages = true;
        // collect_modules recurses into subpackages itself, so only feed it
        // the TOP-LEVEL packages (feeding nested ones would duplicate every
        // module path).
        let top_level = top_level_dirs(&package_dirs);
        for dir in top_level {
            let abs = path.join(&dir);
            let prefix = dotted_prefix(&dir);
            collect_modules(&abs, &prefix, &mut modules)?;
        }
        for py_module in &meta.py_modules {
            // py-modules are single files; try the project root and src/.
            let mut file = path.join(format!("{}.py", py_module));
            if !file.is_file() {
                file = path.join("src").join(format!("{}.py", py_module));
            }
            if !file.is_file() {
                bail!(
                    "py-module `{}` declared by the project metadata is missing ({})",
                    py_module,
                    file.display()
                );
            }
            let source = fs::read_to_string(&file)
                .with_context(|| format!("reading {}", file.display()))?;
            modules.push(PyModule {
                path: vec![sanitize_name(py_module)],
                source,
                file,
                is_init: false,
            });
        }
    }

    if !found_metadata_packages {
        // No packaging metadata (or none that names packages): fall back to
        // the historical layout heuristics.
        let source_root = locate_source_root(&path, name.as_deref())?;
        if name.is_none() {
            name = source_root
                .file_name()
                .and_then(|s| s.to_str())
                .map(str::to_string);
        }
        collect_modules(&source_root, &[], &mut modules)?;
    }

    if modules.is_empty() {
        bail!("no Python modules found under {}", path.display());
    }

    let name = sanitize_name(&name.context("cannot determine package name")?);

    Ok(PyPackage {
        name,
        version: meta.version.unwrap_or_else(|| "0.1.0".to_string()),
        root: path.to_path_buf(),
        modules,
        dependencies: meta.dependencies,
    })
}

/// The module-path prefix for a package directory resolved from metadata:
/// `pkg` → `["pkg"]`, `src/pkg` → `["pkg"]` (src/ is a layout container,
/// not part of the import path), `a.b` → `["a", "b"]`.
fn dotted_prefix(dir: &std::path::Path) -> Vec<String> {
    let mut parts: Vec<String> = dir
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().to_string()),
            _ => None,
        })
        .collect();
    // A leading `src` (or `lib`) layout container is not importable.
    if parts.first().is_some_and(|p| p == "src" || p == "lib") && parts.len() > 1 {
        parts.remove(0);
    }
    parts.iter().map(|p| sanitize_name(p)).collect()
}

/// Keep only the packages that are not nested inside another package in the
/// list (their subpackages are collected recursively anyway).
fn top_level_dirs(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in dirs {
        let nested = dirs.iter().any(|other| {
            other != dir
                && dir
                    .strip_prefix(other)
                    .is_ok_and(|rest| !rest.as_os_str().is_empty())
        });
        if !nested {
            out.push(dir.clone());
        }
    }
    out
}

/// Find the directory whose `.py` files make up the package.
fn locate_source_root(root: &Path, name: Option<&str>) -> Result<PathBuf> {
    // Explicit package dir matching the project name, flat or src/ layout.
    if let Some(name) = name {
        for candidate in [
            root.join(name),
            root.join("src").join(name),
            root.join(name.replace('-', "_")),
            root.join("src").join(name.replace('-', "_")),
        ] {
            if candidate.join("__init__.py").is_file() {
                return Ok(candidate);
            }
        }
    }
    // The root itself is a package.
    if root.join("__init__.py").is_file() {
        return Ok(root.to_path_buf());
    }
    // A single package dir under root or src/.
    for base in [root.to_path_buf(), root.join("src")] {
        if base.is_dir() {
            let mut package_dirs: Vec<PathBuf> = fs::read_dir(&base)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir() && p.join("__init__.py").is_file())
                .collect();
            if package_dirs.len() == 1 {
                return Ok(package_dirs.remove(0));
            }
        }
    }
    // Fall back to a flat directory of .py files.
    let has_py = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("py"));
    if has_py {
        return Ok(root.to_path_buf());
    }
    bail!(
        "cannot locate Python sources under {} (expected a package directory with __init__.py, or .py files)",
        root.display()
    )
}

/// Recursively collect `.py` modules under `dir`.
fn collect_modules(dir: &Path, prefix: &[String], out: &mut Vec<PyModule>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            // Only recurse into subpackages (dirs with __init__.py).
            if path.join("__init__.py").is_file() {
                let sub = sanitize_name(
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .context("invalid directory name")?,
                );
                let mut nested_prefix = prefix.to_vec();
                nested_prefix.push(sub);
                collect_modules(&path, &nested_prefix, out)?;
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("invalid file name")?;
        let source =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let is_init = stem == "__init__";
        let mut module_path = prefix.to_vec();
        if !is_init {
            module_path.push(sanitize_name(stem));
        }
        out.push(PyModule {
            path: module_path,
            source,
            file: path,
            is_init,
        });
    }
    Ok(())
}
