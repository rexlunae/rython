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

fn main() -> Result<()> {
    match Cli::parse().command {
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
                },
            )?;
            report_warnings(&krate);
            rypip::cargo_install(&krate, root.as_deref())?;
            println!("installed `{}`", krate.name);
        }
    }
    Ok(())
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
