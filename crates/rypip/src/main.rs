//! The rypip command line: pip-like workflows on top of the rython
//! Python-to-Rust toolchain.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use rypip::convert::ConvertOptions;

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
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Convert {
            package,
            out,
            pyo3,
            stdpython,
        } => {
            let pkg = rypip::discover(&package)?;
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3,
                    stdpython_path: stdpython,
                },
            )?;
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
        } => {
            let pkg = rypip::discover(&package)?;
            let out = out.unwrap_or_else(|| work_dir(&pkg.name));
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3: false,
                    stdpython_path: stdpython,
                },
            )?;
            rypip::cargo_build(&krate)?;
            println!("built `{}` in {}", krate.name, krate.root.display());
        }
        Cmd::Install {
            package,
            root,
            stdpython,
        } => {
            let pkg = rypip::discover(&package)?;
            let out = work_dir(&pkg.name);
            let krate = rypip::convert(
                &pkg,
                &out,
                &ConvertOptions {
                    pyo3: false,
                    stdpython_path: stdpython,
                },
            )?;
            rypip::cargo_install(&krate, root.as_deref())?;
            println!("installed `{}`", krate.name);
        }
    }
    Ok(())
}

/// A stable scratch location for generated crates.
fn work_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rypip-{}", name))
}
