//! rython-jit: on-demand compilation of Python through the rython
//! converter (issue #166, [`docs/jit.md`]).
//!
//! [`Jit::compile`] converts a module's source to a Rust `cdylib` crate,
//! builds it with cargo, caches the library by content, and loads it into
//! this process; [`Module::call`] calls one of its functions with typed
//! [`Value`]s and [`Module::run_entry`] runs it as a program. The code that
//! runs is the code `rypip run` would build for the same source — the same
//! conversion, the same runtime, the same refusals — and every failure is
//! a loud [`JitError`]; nothing is ever interpreted or approximated.
//!
//! The converter lives in `rypip`, whose command line (`rypip jit`) is
//! built on this crate, so this crate cannot depend on it (cargo forbids
//! the cycle): the conversion and the build come in through the
//! [`Compiler`] trait, which `rypip::jit::RypipCompiler` implements.
//!
//! Requirements at JIT time: a Unix host, the Rust toolchain (`rustc`,
//! `cargo`), and — through the converter's parser — a CPython
//! installation. None of them are needed by converted output.
//!
//! [`docs/jit.md`]: https://github.com/rexlunae/rython/blob/main/docs/jit.md

pub mod abi;
mod cache;
mod error;
mod loader;
pub mod manifest;
mod value;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

pub use cache::default_cache_dir;
pub use error::JitError;
/// The host's runtime: a raised Python exception comes back as its
/// [`PyException`](stdpython::PyException) (and `PyRepr` gives its repr).
pub use stdpython;
pub use manifest::{FunctionSig, MarshalType, Param};
pub use value::{FromValue, Value};

/// One module's source, as the JIT hands it to the [`Compiler`].
#[derive(Debug, Clone, Copy)]
pub struct SourceModule<'a> {
    /// The Python module name (`prog` for `prog.py`).
    pub name: &'a str,
    /// The module's source.
    pub source: &'a str,
    /// Where the source came from, for the converter's messages (a file
    /// the caller compiled, or a placeholder for a string).
    pub origin: &'a Path,
    /// A directory of the cache entry the compiler may use as the source's
    /// package root; nothing else is in it.
    pub package_root: &'a Path,
}

/// What a conversion produced.
#[derive(Debug, Clone)]
pub struct Converted {
    /// The generated crate's package (and library) name.
    pub crate_name: String,
    /// Lossy-conversion warnings, reported with the module (every time it
    /// loads, as `rypip run` reports them on every run).
    pub warnings: Vec<String>,
}

/// What a build produced.
#[derive(Debug, Clone)]
pub struct Built {
    /// The `cdylib` cargo reported.
    pub library: PathBuf,
    /// The build's diagnostics (rustc warnings), if it printed any.
    pub diagnostics: String,
}

/// The converter and builder the JIT drives (`rypip::jit::RypipCompiler`).
pub trait Compiler: Send + Sync {
    /// Everything about this compiler, besides the source, that can change
    /// the library it produces: the converter's version and build, the
    /// runtime the generated crate links (its version, source and
    /// features), the parser's CPython, and the conversion options. Part
    /// of the cache key.
    fn fingerprint(&self) -> Result<String, JitError>;

    /// Convert `module` into a `cdylib` crate in the empty directory
    /// `crate_dir` (the converter's cdylib mode). A refusal is
    /// [`JitError::Conversion`] with the converter's message verbatim.
    fn convert(&self, module: &SourceModule<'_>, crate_dir: &Path) -> Result<Converted, JitError>;

    /// Build the crate in `crate_dir` (release, for the host) with
    /// `target_dir` as its cargo target directory, shared by every entry,
    /// and return the library. A failed build is [`JitError::Build`] with
    /// rustc's diagnostics.
    fn build(&self, crate_dir: &Path, crate_name: &str, target_dir: &Path) -> Result<Built, JitError>;
}

/// How to set up a [`Jit`].
#[derive(Clone)]
pub struct JitConfig {
    /// The converter and builder.
    pub compiler: Arc<dyn Compiler>,
    /// The cache root; [`default_cache_dir`] when `None`.
    pub cache_dir: Option<PathBuf>,
}

impl JitConfig {
    pub fn new(compiler: Arc<dyn Compiler>) -> JitConfig {
        JitConfig {
            compiler,
            cache_dir: None,
        }
    }

    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> JitConfig {
        self.cache_dir = Some(dir.into());
        self
    }
}

/// Whether a compile found its library in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Converted and built now.
    Built,
    /// Loaded from the cache without running the converter or cargo.
    Hit,
}

/// The JIT: a compiler and a cache.
pub struct Jit {
    compiler: Arc<dyn Compiler>,
    root: PathBuf,
    rustc: OnceLock<String>,
    builds: AtomicUsize,
}

impl Jit {
    /// Set up the JIT and its private cache directory. Unix only in M1:
    /// elsewhere this is [`JitError::Environment`], as `rypip run` is.
    pub fn new(config: JitConfig) -> Result<Jit, JitError> {
        if !cfg!(unix) {
            return Err(JitError::Environment(
                "the rython JIT is Unix-only for now (M1, docs/jit.md): loading compiled \
                 libraries and the private cache are implemented for Unix hosts only"
                    .to_string(),
            ));
        }
        let root = match config.cache_dir {
            Some(dir) => dir,
            None => default_cache_dir()?,
        };
        #[cfg(unix)]
        cache::ensure_private_dir(&root)?;
        Ok(Jit {
            compiler: config.compiler,
            root,
            rustc: OnceLock::new(),
            builds: AtomicUsize::new(0),
        })
    }

    /// The cache root.
    pub fn cache_dir(&self) -> &Path {
        &self.root
    }

    /// How many builds (cache misses) this JIT has run.
    pub fn builds(&self) -> usize {
        self.builds.load(Ordering::SeqCst)
    }

    /// Compile module `name` from `source` and load it.
    pub fn compile(&self, name: &str, source: &str) -> Result<Module, JitError> {
        self.compile_from(name, source, Path::new(&format!("<{}>", name)))
    }

    /// Compile the module in the file at `path`, named `name` (the caller
    /// decides: `prog` for `prog.py`, or what the program runs as). The
    /// converter's messages name `path`.
    pub fn compile_file(&self, name: &str, path: &Path) -> Result<Module, JitError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| JitError::Environment(format!("reading {}: {}", path.display(), e)))?;
        self.compile_from(name, &source, path)
    }

    fn compile_from(&self, name: &str, source: &str, origin: &Path) -> Result<Module, JitError> {
        if name.is_empty() || name.contains(|c: char| c == '/' || c == '\\' || c.is_whitespace() || c == '\0') {
            return Err(JitError::Unsupported(format!(
                "`{}` is not a module name the JIT can compile (it names the cache entry \
                 and the generated crate)",
                name
            )));
        }
        let rustc = match self.rustc.get() {
            Some(rustc) => rustc,
            None => {
                let version = cache::rustc_version(&self.root)?;
                self.rustc.get_or_init(|| version)
            }
        };
        let key = cache::Key::new(name, source, &self.compiler.fingerprint()?, rustc);
        let entry = cache::Entry::new(&self.root, &key);
        std::fs::create_dir_all(self.root.join("entries"))
            .map_err(|e| JitError::Environment(format!("creating {}: {}", self.root.display(), e)))?;

        // A hit: read the record under a shared lock (a rebuild holds the
        // exclusive one) and load what it names.
        {
            let _shared = cache::lock(&entry.lock, false)?;
            match cache::read_record(&entry, &key) {
                Ok(Some(record)) => match loader::load(&record.artifact) {
                    Ok(library) => {
                        return Ok(Module {
                            library,
                            outcome: CacheOutcome::Hit,
                            warnings: record.warnings,
                            diagnostics: String::new(),
                        });
                    }
                    Err(why) => warn_rebuild(&entry, &format!("its library would not load: {}", why)),
                },
                Ok(None) => {}
                Err(why) => warn_rebuild(&entry, &why),
            }
        }

        // A miss (or an entry that cannot be trusted): build it, holding
        // the entry's lock from conversion through the record.
        let _exclusive = cache::lock(&entry.lock, true)?;
        // Another process may have built it while this one waited.
        if let Ok(Some(record)) = cache::read_record(&entry, &key)
            && let Ok(library) = loader::load(&record.artifact)
        {
            return Ok(Module {
                library,
                outcome: CacheOutcome::Hit,
                warnings: record.warnings,
                diagnostics: String::new(),
            });
        }
        cache::invalidate(&entry)
            .map_err(|e| JitError::Environment(format!("invalidating {}: {}", entry.dir.display(), e)))?;
        cache::reset_entry(&entry)?;
        let package_root = entry.dir.join("package");
        std::fs::create_dir_all(&package_root)
            .map_err(|e| JitError::Environment(format!("creating {}: {}", package_root.display(), e)))?;
        let module = SourceModule {
            name,
            source,
            origin,
            package_root: &package_root,
        };
        let converted = self.compiler.convert(&module, &entry.crate_dir())?;
        let (artifact, diagnostics) = {
            // Every entry builds in the one target directory; holding its
            // lock through the copy keeps another entry's build (which may
            // reuse the crate name) from replacing the artifact first.
            let _target = cache::lock(&self.root.join("target.lock"), true)?;
            self.builds.fetch_add(1, Ordering::SeqCst);
            let built = self
                .compiler
                .build(&entry.crate_dir(), &converted.crate_name, &self.root.join("target"))?;
            (cache::install_artifact(&entry, &built.library)?, built.diagnostics)
        };
        let library = loader::load(&artifact).map_err(|why| {
            JitError::Build(format!(
                "the library just built for `{}` ({}) would not load: {}",
                name,
                artifact.display(),
                why
            ))
        })?;
        cache::write_record(&entry, &key, &artifact, &converted.warnings)?;
        Ok(Module {
            library,
            outcome: CacheOutcome::Built,
            warnings: converted.warnings,
            diagnostics,
        })
    }

    /// Remove the cache at `dir`: every entry and the shared target
    /// directory (`rypip jit --clear-cache`).
    pub fn clear_cache(dir: &Path) -> Result<(), JitError> {
        cache::clear(dir)
    }
}

/// "rebuilt loudly": an entry that cannot be trusted is reported on stderr
/// before it is rebuilt.
fn warn_rebuild(entry: &cache::Entry, why: &str) {
    eprintln!(
        "rython-jit: the cache entry {} cannot be trusted ({}); rebuilding it",
        entry.dir.display(),
        why
    );
}

/// A compiled, loaded module.
pub struct Module {
    library: &'static loader::Library,
    outcome: CacheOutcome,
    warnings: Vec<String>,
    diagnostics: String,
}

impl Module {
    /// The Python module name.
    pub fn name(&self) -> &str {
        &self.library.manifest.module
    }

    /// Whether this compile was a cache hit.
    pub fn cache_outcome(&self) -> CacheOutcome {
        self.outcome
    }

    /// The conversion's lossy-conversion warnings.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The build's diagnostics, when this compile built the library.
    pub fn build_diagnostics(&self) -> &str {
        &self.diagnostics
    }

    /// The loaded library's file, inside its cache entry.
    pub fn library_path(&self) -> &Path {
        &self.library.path
    }

    /// The functions this module's library can call, with their M1
    /// signatures.
    pub fn functions(&self) -> &[FunctionSig] {
        &self.library.manifest.functions
    }

    /// Whether `class` is a class defined at the module's top level.
    pub fn defines_class(&self, class: &str) -> bool {
        self.library.manifest.classes.iter().any(|c| c == class)
    }

    /// Call `function` with `args`. The first call imports the module (its
    /// body runs, as `import` runs it); arguments must have exactly the
    /// annotated M1 types.
    pub fn call(&self, function: &str, args: &[Value]) -> Result<Value, JitError> {
        self.library.call(function, args)
    }

    /// Install the `sys.argv` the module's function calls see. Without
    /// it, they see this (host) process's arguments; a host that stands in
    /// for a CPython command installs what that command has (`rypip jit
    /// call` installs `['-c']`, as `python3 -c` has). Once, before anything
    /// reads `sys.argv`: refused afterwards ([`JitError::Unsupported`]),
    /// never silently ignored.
    pub fn set_argv(&self, argv: &[String]) -> Result<(), JitError> {
        self.library.set_argv(argv)
    }

    /// [`Module::call`], converting the result to `R`.
    pub fn call_as<R: FromValue>(&self, function: &str, args: &[Value]) -> Result<R, JitError> {
        R::from_value(self.call(function, args)?)
    }

    /// Run the module as a program, as its binary would run (module
    /// initialization, then the `__main__` body), with `argv` as `sys.argv`
    /// (`argv[0]` the program path as the user typed it). Returns the
    /// program's exit status: 0, 1 after an uncaught exception (reported
    /// on stderr exactly as the binary reports it), or a `SystemExit`
    /// code. Once per loaded library and process
    /// ([`JitError::EntryAlreadyRun`]).
    pub fn run_entry(&self, argv: &[String]) -> Result<i32, JitError> {
        self.library.run_entry(argv)
    }
}
