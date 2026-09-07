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

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

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

/// The scratch crate `rypip run` uses for a program when `--out` is not
/// given: under the system temp dir, keyed by the package name AND a hash
/// of the canonical source path, so two programs that discover the same
/// name never share a crate (Devin review on #340), while repeated runs
/// of one source reuse theirs and rebuild incrementally. An explicit
/// `--out` is one crate directory the user owns, as with `rypip convert
/// --out`: two different programs given the same one take turns in it
/// (the lock) and each conversion replaces the other's sources.
pub fn run_work_dir(package: &Path, name: &str) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let canonical = package
        .canonicalize()
        .with_context(|| format!("resolving {}", package.display()))?;
    let digest = Sha256::digest(canonical.as_os_str().as_encoded_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    Ok(std::env::temp_dir().join(format!("rypip-run-{}-{}", name, hash)))
}

/// An exclusive advisory lock on a run work dir (a sibling `<dir>.lock`
/// file), held from source generation through the build so two `rypip
/// run` invocations of one source never interleave a conversion with a
/// build; dropped before the program executes, since concurrent
/// executions are the user's business, as under CPython.
pub struct WorkDirLock {
    file: std::fs::File,
}

impl Drop for WorkDirLock {
    fn drop(&mut self) {
        // Closing the file releases the lock too; the explicit unlock
        // documents the release point.
        let _ = self.file.unlock();
    }
}

pub fn lock_work_dir(dir: &Path) -> Result<WorkDirLock> {
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let path = dir.with_extension("lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.lock()
        .with_context(|| format!("locking {}", path.display()))?;
    Ok(WorkDirLock { file })
}

/// Build the converted crate quietly (`--release --quiet`, the program's
/// own output is what the user came for) and return the executable cargo
/// reports for its binary target — wherever cargo put it: a
/// `CARGO_TARGET_DIR`, a configured target, a platform suffix (Devin
/// review on #340). The compiler's diagnostics still reach stderr
/// (`json-render-diagnostics`); the artifact messages are read from
/// stdout.
pub fn cargo_build_executable(krate: &ConvertedCrate) -> Result<PathBuf> {
    require_binary(krate, "run")?;
    let output = Command::new("cargo")
        .args(["build", "--release", "--quiet", "--message-format=json-render-diagnostics"])
        .current_dir(&krate.root)
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| "running cargo build".to_string())?;
    if !output.status.success() {
        bail!("cargo build failed with {}", output.status);
    }
    for line in output.stdout.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let Ok(msg) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        if msg["reason"] != "compiler-artifact" {
            continue;
        }
        let is_bin = msg["target"]["kind"]
            .as_array()
            .is_some_and(|kinds| kinds.iter().any(|k| k == "bin"));
        if is_bin && msg["target"]["name"] == krate.name.as_str() {
            if let Some(exe) = msg["executable"].as_str() {
                return Ok(PathBuf::from(exe));
            }
        }
    }
    bail!(
        "built `{}` but cargo reported no executable for its binary target",
        krate.name
    )
}

/// Execute a built program as `python program args` would: `sys.argv[0]`
/// is `program` exactly as the user typed it (CPython keeps the script
/// path as given: `hello.py`, `./hello.py`, `pkgdir/`), the arguments
/// follow, and the streams and the working directory are inherited. The
/// exit status is returned for the caller to propagate.
pub fn run_program(binary: &Path, program: &OsStr, args: &[OsString]) -> Result<ExitStatus> {
    let mut cmd = Command::new(binary);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.arg0(program);
    }
    #[cfg(not(unix))]
    {
        let _ = program;
        bail!(
            "rypip run cannot set sys.argv[0] to the program path on this platform yet \
             (CPython's sys.argv[0] is the script path as given; the binary's path would \
             differ silently)"
        );
    }
    cmd.args(args)
        .status()
        .with_context(|| format!("running {}", binary.display()))
}

/// `rypip run` in one call (issue #166: CPython's command-line shape over
/// the convert/build pipeline, with no new semantics): take the work
/// dir's lock, convert, hand the crate to `on_converted` (the CLI reports
/// its warnings there), build, release the lock, then execute with
/// `program` as `sys.argv[0]` and `args`. The one workflow the CLI and a
/// library caller share (Devin review on #340, round 2).
pub fn run(
    pkg: &PyPackage,
    out: &Path,
    options: &ConvertOptions,
    program: &OsStr,
    args: &[OsString],
    on_converted: impl FnOnce(&ConvertedCrate),
) -> Result<ExitStatus> {
    let lock = lock_work_dir(out)?;
    let krate = convert(pkg, out, options)?;
    on_converted(&krate);
    let binary = cargo_build_executable(&krate)?;
    drop(lock);
    run_program(&binary, program, args)
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
