//! rypip: a pip-like tool for the rython toolchain.
//!
//! rypip builds Python packages as native Rust binaries and installs them
//! where cargo installs binaries (`cargo install`'s root, normally
//! `~/.cargo/bin`), and converts Python packages into Rust crates —
//! optionally with PyO3 bindings so the converted crate can still be
//! imported from Python.

pub mod convert;
pub mod package;

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};

pub use convert::{convert, ConvertOptions, ConvertedCrate};
pub use package::{discover, PyPackage};

/// Run `cargo build --release` on a converted crate.
pub fn cargo_build(krate: &ConvertedCrate) -> Result<()> {
    run_cargo(
        Command::new("cargo")
            .arg("build")
            .arg("--release")
            .current_dir(&krate.root),
        "cargo build",
    )
}

/// Install a converted crate's binary the same way `cargo install` would
/// (into `$CARGO_INSTALL_ROOT`/`~/.cargo/bin` unless `root` overrides it).
pub fn cargo_install(krate: &ConvertedCrate, root: Option<&Path>) -> Result<()> {
    if !krate.has_binary {
        bail!(
            "package `{}` has no entry point; add an `if __name__ == \"__main__\":` block \
             or a __main__.py to install it as a binary (use `convert` for library crates)",
            krate.name
        );
    }
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
