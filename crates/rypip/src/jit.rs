//! rypip's side of the JIT (issue #166, docs/jit.md): the converter and
//! builder `rython-jit` drives, and the pieces of `rypip jit` that are not
//! argument parsing.
//!
//! `rython-jit` cannot depend on rypip (rypip's command line depends on
//! it, and cargo forbids the cycle), so the conversion and the build reach
//! it through its `Compiler` trait, implemented here with the same
//! converter (`convert`, in its cdylib mode) and the same cargo
//! invocation (`cargo_build_artifact`) `rypip run` uses.

use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;

use rython_jit::{Built, Compiler, Converted, Jit, JitConfig, JitError, SourceModule, Value};

use crate::convert::{CdylibTarget, ConvertOptions, convert, stdpython_fingerprint};
use crate::package::{PyModule, PyPackage, sanitize_name};
use crate::{ArtifactError, ArtifactKind, Diagnostics, cargo_build_artifact};

/// The converter's build identity (see build.rs).
const CONVERTER_BUILD: &str = env!("RYPIP_CONVERTER_BUILD");

/// The JIT's compiler: rypip's converter in cdylib mode, and cargo.
pub struct RypipCompiler {
    options: ConvertOptions,
}

impl RypipCompiler {
    /// A compiler converting with `options` (the stdpython location and
    /// the warning mode; the cdylib target is set per module, and
    /// packaging dependencies are never resolved — the JIT compiles one
    /// module).
    pub fn new(options: ConvertOptions) -> RypipCompiler {
        RypipCompiler { options }
    }
}

/// A JIT using rypip's compiler with `options`, caching in `cache_dir`
/// (the default `$XDG_CACHE_HOME/rython/jit` when `None`).
pub fn new_jit(options: ConvertOptions, cache_dir: Option<&Path>) -> Result<Jit, JitError> {
    let mut config = JitConfig::new(Arc::new(RypipCompiler::new(options)));
    if let Some(dir) = cache_dir {
        config = config.with_cache_dir(dir);
    }
    Jit::new(config)
}

impl Compiler for RypipCompiler {
    fn fingerprint(&self) -> Result<String, JitError> {
        if self.options.no_std || self.options.pyo3 || self.options.kernel_module
            || self.options.driver || self.options.rust_for_linux
        {
            return Err(JitError::Unsupported(
                "the JIT requires the host std cdylib target; no_std, pyo3, kernel_module, \
                 driver and rust_for_linux options are not supported; use default conversion options".into()
            ));
        }
        let stdpython = stdpython_fingerprint(&self.options)
            .map_err(|e| JitError::Environment(format!("locating the stdpython runtime: {:#}", e)))?;
        Ok(format!(
            "rypip {}\nconverter-build {}\npython {}\nstdpython {}\noptions {:?}\n",
            env!("CARGO_PKG_VERSION"),
            CONVERTER_BUILD,
            python_ast::python_version().replace('\n', " "),
            stdpython,
            self.options,
        ))
    }

    fn convert(&self, module: &SourceModule<'_>, crate_dir: &Path) -> Result<Converted, JitError> {
        match python_ast::is_module_identifier(module.name) {
            Ok(true) if module.name != "__main__" => {}
            Ok(_) => {
                return Err(JitError::Unsupported(format!(
                    "`{}` is not a module name the JIT can compile: it must be a Python \
                     identifier that is not a keyword (an importable module name)",
                    module.name
                )));
            }
            Err(e) => return Err(JitError::Environment(format!("asking CPython about `{}`: {}", module.name, e))),
        }
        let rust_name = sanitize_name(module.name);
        let package = PyPackage {
            name: rust_name.clone(),
            version: "0.1.0".to_string(),
            root: module.package_root.to_path_buf(),
            modules: vec![PyModule {
                path: vec![rust_name],
                import_path: vec![module.name.to_string()],
                source: module.source.to_string(),
                file: module.origin.to_path_buf(),
                is_init: false,
            }],
            dependencies: Vec::new(),
        };
        let mut options = self.options.clone();
        options.cdylib = Some(CdylibTarget {
            module_name: module.name.to_string(),
        });
        options.no_deps = true;
        let krate = convert(&package, crate_dir, &options).map_err(|e| JitError::Conversion(format!("{:#}", e)))?;
        Ok(Converted {
            crate_name: krate.name,
            warnings: krate.warnings,
        })
    }

    fn build(&self, crate_dir: &Path, crate_name: &str, target_dir: &Path) -> Result<Built, JitError> {
        match cargo_build_artifact(crate_dir, crate_name, ArtifactKind::Cdylib, Some(target_dir), Diagnostics::Capture) {
            Ok(artifact) => Ok(Built {
                library: artifact.path,
                diagnostics: artifact.diagnostics,
            }),
            Err(ArtifactError::Spawn(e)) => Err(JitError::Environment(format!(
                "the Rust toolchain is missing: running `cargo build` (cargo on PATH) failed: \
                 {}; the JIT builds with cargo — install it (https://rustup.rs)",
                e
            ))),
            Err(ArtifactError::Host(e)) => Err(JitError::Environment(e)),
            Err(ArtifactError::Failed { status, diagnostics }) => Err(JitError::Build(format!(
                "cargo build failed with {}:\n{}",
                status,
                diagnostics.trim_end()
            ))),
            Err(missing @ ArtifactError::Missing { .. }) => Err(JitError::Build(missing.to_string())),
        }
    }
}

/// The module name `rypip jit` compiles FILE under. `jit call` imports the
/// module, so its stem must be importable under CPython; `jit run` runs it
/// as `__main__`, so any `.py` file runs, under its stem when that is an
/// identifier and the sanitized stem otherwise (the name only labels the
/// cache entry and messages). Packages and other files are refused: M1
/// compiles one file, and a `rython.toml` beside it (vendored or Rust
/// modules, which `rypip run` would honor) is refused rather than ignored.
pub fn module_name_for(file: &Path, importing: bool) -> anyhow::Result<String> {
    use anyhow::bail;
    if file.is_dir() {
        bail!(
            "rypip jit compiles a single .py file (M1), and {} is a directory; run a \
             package with `rypip run`",
            file.display()
        );
    }
    if file.extension().and_then(|e| e.to_str()) != Some("py") {
        bail!("{} is not a Python file (.py)", file.display());
    }
    let parent = file.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if parent.join("rython.toml").is_file() {
        bail!(
            "{} sits beside a rython.toml, whose [python-modules]/[rust-modules] the JIT does \
             not read (M1 compiles the file alone; `rypip run` honors them); run it with \
             `rypip run`, or move it",
            file.display()
        );
    }
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("{} has no UTF-8 file name", file.display()))?;
    let importable = python_ast::is_module_identifier(stem).map_err(|e| anyhow::anyhow!("{}", e))?;
    if importable && stem != "__main__" {
        return Ok(stem.to_string());
    }
    if importing {
        bail!(
            "`{}` cannot be imported under CPython (`{}` is not an importable module name), \
             so its functions cannot be called; rename the file",
            file.display(),
            stem
        );
    }
    if stem == "__main__" {
        bail!(
            "{} is a package's entry module; run the package with `rypip run`",
            file.display()
        );
    }
    Ok(sanitize_name(stem))
}

/// The program's `sys.argv`: FILE as the user typed it, then its
/// arguments. A non-UTF-8 argument is refused before the program runs
/// (the runtime's `str` cannot hold CPython's surrogate escapes).
pub fn program_argv(file: &std::ffi::OsStr, args: &[OsString]) -> anyhow::Result<Vec<String>> {
    std::iter::once(file)
        .chain(args.iter().map(OsString::as_os_str))
        .enumerate()
        .map(|(i, arg)| {
            arg.to_str().map(str::to_string).ok_or_else(|| {
                anyhow::anyhow!(
                    "sys.argv[{}] is not valid UTF-8 ({:?}); CPython would keep it as surrogate \
                     escapes, which this runtime does not model",
                    i,
                    arg
                )
            })
        })
        .collect()
}

/// A `rypip jit call` argument, read as CPython reads a literal
/// (`ast.literal_eval`), as a boundary value.
pub fn literal_argument(index: usize, text: &str) -> anyhow::Result<Value> {
    use python_ast::PyLiteral;
    let literal = python_ast::literal_eval(text).map_err(|e| {
        anyhow::anyhow!(
            "argument {} (`{}`) is not a Python literal: {}",
            index + 1,
            text,
            e
        )
    })?;
    Ok(match literal {
        PyLiteral::None => Value::None,
        PyLiteral::Bool(b) => Value::Bool(b),
        PyLiteral::Int(n) => Value::Int(n),
        PyLiteral::Float(x) => Value::Float(x),
        PyLiteral::Str(s) => Value::Str(s),
        PyLiteral::IntOutOfRange(repr) => {
            return Err(JitError::Marshal(format!(
                "argument {} is the int {}, outside the 64-bit range the JIT boundary carries \
                 an int in (M1)",
                index + 1,
                repr
            ))
            .into());
        }
        PyLiteral::Other(type_name) => {
            return Err(JitError::Marshal(format!(
                "argument {} is a `{}`, which M1 cannot marshal across the JIT boundary \
                 (int, float, bool, str and None cross)",
                index + 1,
                type_name
            ))
            .into());
        }
    })
}
