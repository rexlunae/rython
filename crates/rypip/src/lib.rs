//! rypip: a pip-like tool for the rython toolchain.
//!
//! rypip builds Python packages as native Rust binaries and installs them
//! where cargo installs binaries (`cargo install`'s root, normally
//! `~/.cargo/bin`), and converts Python packages into Rust crates —
//! optionally with PyO3 bindings so the converted crate can still be
//! imported from Python.

pub mod convert;
pub mod package;
pub mod packaging;
pub mod resolve;

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};

pub use convert::{convert, ConvertOptions, ConvertedCrate};
pub use package::{discover, PyPackage};

/// Run `cargo build --release` on a converted crate.
pub fn cargo_build(krate: &ConvertedCrate) -> Result<()> {
    cargo_build_with(krate, false)
}

fn cargo_build_with(krate: &ConvertedCrate, quiet: bool) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("--release").current_dir(&krate.root);
    if quiet {
        cmd.arg("--quiet");
    }
    run_cargo(&mut cmd, "cargo build")
}

/// A converted crate that must have an entry point for `verb`
/// (install, run): a library crate is refused with the fix named.
fn require_binary(krate: &ConvertedCrate, verb: &str) -> Result<()> {
    if !krate.has_binary {
        bail!(
            "package `{}` has no entry point; add an `if __name__ == \"__main__\":` block \
             or a __main__.py to {} it as a binary (use `convert` for library crates)",
            krate.name,
            verb
        );
    }
    Ok(())
}

/// `rypip run`: build the converted crate (quietly — the program's own
/// output is what the user came for) and execute its binary with `args`,
/// inheriting the streams and the working directory, exactly as `python
/// program.py args` would. The program's exit status is returned for the
/// caller to propagate (issue #166: CPython's command-line shape over the
/// convert/build pipeline, with no new semantics).
pub fn cargo_run(krate: &ConvertedCrate, args: &[std::ffi::OsString]) -> Result<std::process::ExitStatus> {
    require_binary(krate, "run")?;
    cargo_build_with(krate, true)?;
    let binary = krate.root.join("target").join("release").join(&krate.name);
    if !binary.is_file() {
        bail!(
            "built `{}` but found no binary at {}",
            krate.name,
            binary.display()
        );
    }
    Command::new(&binary)
        .args(args)
        .status()
        .with_context(|| format!("running {}", binary.display()))
}

/// Install a converted crate's binary the same way `cargo install` would
/// (into `$CARGO_INSTALL_ROOT`/`~/.cargo/bin` unless `root` overrides it).
pub fn cargo_install(krate: &ConvertedCrate, root: Option<&Path>) -> Result<()> {
    require_binary(krate, "install")?;
    let mut cmd = Command::new("cargo");
    cmd.arg("install")
        .arg("--path")
        .arg(&krate.root)
        .arg("--force");
    if let Some(root) = root {
        cmd.arg("--root").arg(root);
    }
    run_cargo(&mut cmd, "cargo install")
}

fn run_cargo(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("running {}", what))?;
    if !status.success() {
        bail!("{} failed with {}", what, status);
    }
    Ok(())
}
