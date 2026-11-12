//! The rypip command line: pip-like workflows on top of the rython
//! Python-to-Rust toolchain.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use rypip::convert::{ConvertOptions, WarningMode};

#[derive(Parser)]
#[command(
    name = "rypip",
    about = "Build Python packages as native Rust binaries, or convert them into Rust crates (optionally importable from Python via PyO3)."
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Python package into a Rust crate.
    Convert {
        /// Path to a Python package directory or a single .py file.
        package: PathBuf,
        /// Where to write the generated crate.
        #[arg(long, short)]
        out: PathBuf,
        /// Also generate PyO3 bindings (adds a `python` cargo feature and a
        /// cdylib target so the crate can be imported from Python).
        #[arg(long)]
        pyo3: bool,
        /// Path to the stdpython runtime crate (defaults to
        /// $RYPIP_STDPYTHON_PATH or the copy shipped with this tool).
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn (report them, and
        /// bake #[deprecated] notes into the generated code), deny (fail the
        /// conversion), or allow (suppress them entirely).
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// Generate a #![no_std] library crate on stdpython's alloc tier
        /// (no OS dependency, for embedded/wasm targets). Python constructs
        /// that need the OS — print/input/open, os/datetime/random/...
        /// imports, __main__ blocks — fail the conversion loudly.
        #[arg(long)]
        no_std: bool,
        /// Generate a Linux kernel module crate. Implies no_std. Produces
        /// a cdylib with panic=abort, module_init/module_exit entry points,
        /// and printk lowering. No stdpython dependency. When the Python
        /// declares a device manifest (__device_name__, __bufsz__, __magic__,
        /// __device_mode__, __ioc_reset__, __ioc_stats__), also generates
        /// src/device.rs — a misc byte-ring device — and entry points that
        /// register it.
        #[arg(long, conflicts_with = "no_std")]
        kernel_module: bool,
        /// Generate a userspace driver crate for a rython byte-ring misc
        /// device (UIO style): the Python driver logic compiles to a
        /// library and generated syscall glue (open/read/write/ioctl) wraps
        /// it. The Python may declare __device_path__, __ioc_reset__, and
        /// __ioc_stats__ to parameterize the glue.
        #[arg(long, conflicts_with_all = ["kernel_module", "no_std", "pyo3"])]
        driver: bool,
        /// Generate a rust-for-linux kernel module (requires --kernel-module):
        /// a module!-macro crate implementing kernel::Module for the
        /// rust-for-linux toolchain, instead of raw-FFI entry points.
        #[arg(long, requires = "kernel_module")]
        rust_for_linux: bool,
        /// Skip resolving the packaging metadata's dependencies
        /// ([project] dependencies / install_requires) against PyPI.
        #[arg(long)]
        no_deps: bool,
    },
    /// Convert and compile a Python package (release profile).
    Build {
        /// Path to a Python package directory or a single .py file.
        package: PathBuf,
        /// Where to write the generated crate (defaults to a directory under
        /// the system temp dir).
        #[arg(long, short)]
        out: Option<PathBuf>,
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn, deny, or allow.
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// Skip resolving the packaging metadata's dependencies against
        /// PyPI.
        #[arg(long)]
        no_deps: bool,
    },
    /// Convert, build and run a Python program, like `python program.py`:
    /// the arguments after the package path (or after `--`) are the
    /// program's, its output is yours, and its exit status is rypip's.
    /// Unix only for now (sys.argv[0] is set through the exec).
    Run {
        /// Path to a Python package directory or a single .py file.
        package: PathBuf,
        /// Where to write the generated crate (defaults to a directory under
        /// the system temp dir keyed by the package name and the source
        /// path, reused across runs of that source so rebuilds are
        /// incremental). An explicit directory is one crate you own, as
        /// with `convert --out`: the files rypip wrote into its `src/` last
        /// time are replaced on every run (nothing rypip did not write is
        /// deleted), and two different programs given the same one take
        /// turns in it. A directory holding the program's sources, or one
        /// that is not a crate rypip generated, is refused.
        #[arg(long, short)]
        out: Option<PathBuf>,
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn, deny, or allow.
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// Skip resolving the packaging metadata's dependencies against
        /// PyPI.
        #[arg(long)]
        no_deps: bool,
        /// The program's own arguments (sys.argv[1:]; sys.argv[0] is the
        /// package path as given, as under CPython).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<std::ffi::OsString>,
    },
    /// Compile Python on demand and run it in this process (the JIT,
    /// docs/jit.md): the file is converted, built into a shared library,
    /// cached by content under $XDG_CACHE_HOME/rython/jit (default
    /// ~/.cache/rython/jit), and loaded. Unix only for now; needs the Rust
    /// toolchain (and the CPython the parser uses).
    Jit {
        /// Remove the JIT's cache — every compiled module and the shared
        /// build directory — before anything else.
        #[arg(long)]
        clear_cache: bool,
        #[command(subcommand)]
        action: Option<JitCmd>,
    },
    /// Build a Python package as a native binary and install it where cargo
    /// installs binaries (~/.cargo/bin unless --root is given).
    Install {
        /// Path to a Python package directory or a single .py file.
        package: PathBuf,
        /// Install into this root instead of cargo's default.
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn, deny, or allow.
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// Skip resolving the packaging metadata's dependencies against
        /// PyPI.
        #[arg(long)]
        no_deps: bool,
    },
}

#[derive(Subcommand)]
enum JitCmd {
    /// Run a program in-process, like `python FILE ARG...`: the arguments
    /// after FILE are its own (sys.argv[0] is FILE as given), its output is
    /// yours, and its exit status is rypip's.
    Run {
        /// The program: a single .py file.
        file: PathBuf,
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn, deny, or allow.
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// The program's own arguments (sys.argv[1:]).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<std::ffi::OsString>,
    },
    /// Call one function of a module and print the repr of its result,
    /// like `python -c 'import M; print(repr(M.FUNCTION(ARG, ...)))'`.
    /// ARGs are Python literals of the types that cross the JIT boundary:
    /// int, float, bool, str and None.
    Call {
        /// The module: a single .py file with an importable name.
        file: PathBuf,
        /// A public module-level function of the module.
        function: String,
        #[arg(long)]
        stdpython: Option<PathBuf>,
        /// How to treat lossy-conversion warnings: warn, deny, or allow.
        #[arg(long, short = 'W', value_enum, default_value_t = WarningMode::Warn)]
        warnings: WarningMode,
        /// The arguments, as Python literals (`3`, `-2.5`, `'text'`, `True`,
        /// `None`).
        #[arg(allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Jit { clear_cache, action } => jit(clear_cache, action)?,
        Cmd::Convert {
            package,
            out,
            pyo3,
            stdpython,
            warnings,
            no_std,
            kernel_module,
            driver,
            rust_for_linux,
            no_deps,
        } => {
            let pkg = rypip::discover(&package)?;
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3,
                    stdpython_path: stdpython,
                    warnings,
                    no_std,
                    kernel_module,
                    driver,
                    rust_for_linux,
                    no_deps,
                    cdylib: None,
                },
            )?;
            report_warnings(&krate);
            println!(
                "converted `{}` -> {}{}",
                krate.name,
                krate.root.display(),
                if pyo3 {
                    " (with PyO3 bindings: build with --features python)"
                } else {
                    ""
                }
            );
        }
        Cmd::Build {
            package,
            out,
            stdpython,
            warnings,
            no_deps,
        } => {
            let pkg = rypip::discover(&package)?;
            let out = out.unwrap_or_else(|| work_dir(&pkg.name));
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3: false,
                    stdpython_path: stdpython,
                    warnings,
                    no_std: false,
                    kernel_module: false,
                    driver: false,
                    rust_for_linux: false,
                    no_deps,
                    cdylib: None,
                },
            )?;
            report_warnings(&krate);
            rypip::cargo_build(&krate)?;
            println!("built `{}` in {}", krate.name, krate.root.display());
        }
        Cmd::Run {
            package,
            out,
            stdpython,
            warnings,
            no_deps,
            args,
        } => {
            let pkg = rypip::discover(&package)?;
            let out = match out {
                Some(out) => out,
                None => rypip::run_work_dir(&package, &pkg.name)?,
            };
            let status = rypip::run(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3: false,
                    stdpython_path: stdpython,
                    warnings,
                    no_std: false,
                    kernel_module: false,
                    driver: false,
                    rust_for_linux: false,
                    no_deps,
                    cdylib: None,
                },
                package.as_os_str(),
                &args,
                report_warnings,
            )?;
            // The program's exit status is ours: the code as is; a signal
            // death as the shell reports it (128 + the signal number).
            let code = status.code().unwrap_or_else(|| {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    128 + status.signal().unwrap_or(0)
                }
                #[cfg(not(unix))]
                {
                    1
                }
            });
            std::process::exit(code);
        }
        Cmd::Install {
            package,
            root,
            stdpython,
            warnings,
            no_deps,
        } => {
            let pkg = rypip::discover(&package)?;
            let out = work_dir(&pkg.name);
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3: false,
                    stdpython_path: stdpython,
                    warnings,
                    no_std: false,
                    kernel_module: false,
                    driver: false,
                    rust_for_linux: false,
                    no_deps,
                    cdylib: None,
                },
            )?;
            report_warnings(&krate);
            rypip::cargo_install(&krate, root.as_deref())?;
            println!("installed `{}`", krate.name);
        }
    }
    Ok(())
}

/// `rypip jit`: a thin layer over rython-jit (rypip::jit holds the
/// rest). Each outcome maps to what the program's own process would show:
/// its exit status, an uncaught exception's traceback header and final
/// line, a `SystemExit`'s message — and a panic's 101, the exit status of
/// a panicking Rust binary (the library has already printed the panic).
fn jit(clear_cache: bool, action: Option<JitCmd>) -> Result<()> {
    use rython_jit::JitError;
    use std::io::Write;
    if clear_cache {
        let dir = rython_jit::default_cache_dir()?;
        rython_jit::Jit::clear_cache(&dir)?;
        eprintln!("removed the JIT cache {}", dir.display());
    }
    let Some(action) = action else {
        if clear_cache {
            return Ok(());
        }
        anyhow::bail!("rypip jit needs a subcommand (run, call) or --clear-cache");
    };
    let options = |stdpython, warnings| ConvertOptions {
        stdpython_path: stdpython,
        warnings,
        ..ConvertOptions::default()
    };
    let compile = |file: &std::path::Path, importing: bool, options: ConvertOptions| -> Result<rython_jit::Module> {
        let name = rypip::jit::module_name_for(file, importing)?;
        let jit = rypip::jit::new_jit(options, None)?;
        let module = jit.compile_file(&name, file)?;
        for warning in module.warnings() {
            eprintln!("warning: {}", warning);
        }
        Ok(module)
    };
    let finish = |code: i32| -> ! {
        let _ = std::io::stdout().flush();
        std::process::exit(code)
    };
    match action {
        JitCmd::Run {
            file,
            stdpython,
            warnings,
            args,
        } => {
            let argv = rypip::jit::program_argv(file.as_os_str(), &args)?;
            let module = compile(&file, false, options(stdpython, warnings))?;
            match module.run_entry(&argv) {
                Ok(status) => finish(status),
                Err(JitError::Panic(_)) => finish(101),
                Err(e) => Err(e.into()),
            }
        }
        JitCmd::Call {
            file,
            function,
            stdpython,
            warnings,
            args,
        } => {
            let values = args
                .iter()
                .enumerate()
                .map(|(i, a)| rypip::jit::literal_argument(i, a))
                .collect::<Result<Vec<_>>>()?;
            let module = compile(&file, true, options(stdpython, warnings))?;
            // `python3 -c 'import M; print(repr(M.F(...)))'` has sys.argv
            // == ['-c'], not rypip's own arguments.
            module.set_argv(&["-c".to_string()])?;
            match module.call(&function, &values) {
                Ok(value) => {
                    println!("{}", value.repr());
                    finish(0)
                }
                Err(JitError::Python(e)) => {
                    let _ = std::io::stdout().flush();
                    // CPython names a class of an imported module
                    // module-qualified in the traceback's last line.
                    let line = e.format_exception_only();
                    let line = if module.defines_class(&e.exception_type) {
                        format!("{}.{}", module.name(), line)
                    } else {
                        line
                    };
                    eprintln!("Traceback (most recent call last):");
                    eprintln!("{}", line);
                    finish(1)
                }
                Err(JitError::Exit { code, message }) => {
                    let _ = std::io::stdout().flush();
                    if let Some(message) = message {
                        eprintln!("{}", message);
                    }
                    finish(code)
                }
                Err(JitError::Panic(_)) => finish(101),
                Err(e) => Err(e.into()),
            }
        }
    }
}

/// Surface lossy-conversion warnings on stderr as they happen.
fn report_warnings(krate: &rypip::ConvertedCrate) {
    for warning in &krate.warnings {
        eprintln!("warning: {}", warning);
    }
}

/// A stable scratch location for generated crates.
fn work_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rypip-{}", name))
}
