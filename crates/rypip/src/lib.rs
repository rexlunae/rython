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
/// file), held from source generation through the build and the staging
/// of the executable, so two `rypip run` invocations of one work dir
/// never interleave a conversion with a build; dropped before the program
/// executes, since concurrent executions are the user's business, as
/// under CPython.
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

/// The physical spelling of a path — the canonical path of its nearest
/// existing ancestor plus the components not yet created — so two
/// spellings of one directory (a symlinked parent, `./x`, `a/../x`) name
/// the same thing (Devin review on #340, round 4).
fn physical_path(path: &Path) -> Result<PathBuf> {
    use std::path::Component;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    // The longest prefix that exists, component by component (a `..` after
    // a component that does not exist cannot make a longer prefix exist:
    // the file system resolves the missing component first).
    let components: Vec<Component> = absolute.components().collect();
    let mut existing = PathBuf::new();
    let mut existing_len = 0;
    let mut probe = PathBuf::new();
    for (i, component) in components.iter().enumerate() {
        probe.push(component);
        if probe.exists() {
            existing = probe.clone();
            existing_len = i + 1;
        } else {
            break;
        }
    }
    if existing_len == 0 {
        bail!("resolving {}: no existing ancestor", path.display());
    }
    let mut physical = existing
        .canonicalize()
        .with_context(|| format!("resolving {}", existing.display()))?;
    // The components still to be created are normalized lexically, which
    // is exact on a canonical prefix (no symlink can turn `..` elsewhere):
    // `proj/nonexistent/..` IS `proj`, so a guard comparing physical paths
    // sees the destination the file system will use (Devin review on
    // #340, round 5).
    for component in &components[existing_len..] {
        match component {
            Component::Normal(name) => physical.push(name),
            Component::ParentDir => {
                physical.pop();
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    Ok(physical)
}

/// The host target triple, from `rustc -vV`: `rypip run` builds for the
/// host explicitly, so a `build.target` / `CARGO_BUILD_TARGET` selecting
/// a cross target never produces an artifact this machine cannot run —
/// `python program.py` runs here, and so does this (Devin review on
/// #340, round 5).
pub fn host_triple() -> Result<&'static str> {
    static HOST: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    let host = HOST.get_or_init(|| {
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
        let output = Command::new(rustc).arg("-vV").output().map_err(|e| e.to_string())?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find_map(|line| line.strip_prefix("host: ").map(|h| h.trim().to_string()))
            .ok_or_else(|| "rustc -vV reported no host".to_string())
    });
    match host {
        Ok(h) => Ok(h.as_str()),
        Err(e) => bail!("determining the host target: {}", e),
    }
}

/// The lock file guarding a work dir: the dir's PHYSICAL path with
/// `.lock` appended — appended, never substituted, so `--out foo.lock`
/// locks `foo.lock.lock` and `foo.bar` never shares a lock with a sibling
/// `foo.lock` (Devin review on #340, rounds 3 and 4); physical, so a
/// symlinked alias of the directory takes the same lock.
pub fn work_dir_lock_path(dir: &Path) -> Result<PathBuf> {
    let mut lock_name = physical_path(dir)?.into_os_string();
    lock_name.push(".lock");
    Ok(PathBuf::from(lock_name))
}

pub fn lock_work_dir(dir: &Path) -> Result<WorkDirLock> {
    let path = work_dir_lock_path(dir)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
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
/// own output is what the user came for) FOR THE HOST (`--target` the
/// triple `rustc -vV` reports, so a configured cross target never yields
/// an artifact this machine cannot run) and return the executable cargo
/// reports for its binary target — wherever cargo put it: a
/// `CARGO_TARGET_DIR`, the target's own subdirectory, a platform suffix
/// (Devin review on #340). The compiler's diagnostics still reach stderr
/// (`json-render-diagnostics`); the artifact messages are read from
/// stdout.
pub fn cargo_build_executable(krate: &ConvertedCrate) -> Result<PathBuf> {
    require_binary(krate, "run")?;
    let output = Command::new("cargo")
        .args(["build", "--release", "--quiet", "--message-format=json-render-diagnostics"])
        .arg("--target")
        .arg(host_triple()?)
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
/// exit status is returned for the caller to propagate. Unix only: argv[0]
/// is set through the exec (`CommandExt::arg0`); a platform that cannot
/// set it independently of the executable is refused loudly rather than
/// running with the binary's path in `sys.argv[0]`.
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
            "rypip run is Unix-only for now: it cannot set sys.argv[0] to the program path on \
             this platform (CPython's sys.argv[0] is the script path as given; the binary's \
             path would differ silently)"
        );
    }
    cmd.args(args)
        .status()
        .with_context(|| format!("running {}", binary.display()))
}

/// An invocation's private copy of the built executable — a hard link
/// (or a copy across file systems) under `<work dir>/.run/`, made while
/// the work dir's lock is held — so a concurrent rebuild in the same work
/// dir, or in another work dir whose cargo configuration maps the same
/// package name to the same artifact path, can never swap the program
/// between artifact discovery and the exec (Devin review on #340, round
/// 3). Removed when dropped, after the program has exited.
pub struct StagedExecutable {
    path: PathBuf,
}

impl StagedExecutable {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagedExecutable {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn stage_executable(work_dir: &Path, executable: &Path) -> Result<StagedExecutable> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let dir = work_dir.join(".run");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let file_name = executable
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| OsString::from("program"));
    let mut staged_name = file_name;
    staged_name.push(format!("-{}-{}", std::process::id(), SEQUENCE.fetch_add(1, Ordering::Relaxed)));
    let path = dir.join(staged_name);
    let _ = std::fs::remove_file(&path);
    if std::fs::hard_link(executable, &path).is_err() {
        std::fs::copy(executable, &path)
            .with_context(|| format!("staging {} as {}", executable.display(), path.display()))?;
    }
    Ok(StagedExecutable { path })
}

/// `run` rewrites its work dir's `src/`, so the work dir must be its own:
/// never a directory holding the program's sources (`rypip run . --out .`
/// in a src-layout project would overwrite them), never one whose `src/`
/// holds a discovered module, and never a non-empty directory that is not
/// a crate rypip generated (its manifest starts with
/// [`convert::GENERATED_MANIFEST_HEADER`]). Refused loudly, naming a
/// separate directory as the fix (Devin review on #340, round 4). `out`
/// is a physical path here (see [`physical_path`]), so `..` through a
/// component that does not exist yet cannot slip past the comparison.
fn refuse_foreign_output(pkg: &PyPackage, out: &Path) -> Result<()> {
    let physical = |p: &Path| physical_path(p).unwrap_or_else(|_| p.to_path_buf());
    let out_physical = physical(out);
    let root = physical(&pkg.root);
    if root.starts_with(&out_physical) {
        bail!(
            "rypip run rewrites `{}`'s src/, but that directory holds the program's \
             sources ({}); give --out a separate directory",
            out.display(),
            pkg.root.display()
        );
    }
    let out_src = out_physical.join("src");
    for module in &pkg.modules {
        if physical(&module.file).starts_with(&out_src) {
            bail!(
                "rypip run rewrites `{}`'s src/, but it holds the program's module {}; \
                 give --out a separate directory",
                out.display(),
                module.file.display()
            );
        }
    }
    let occupied = out.is_dir() && out.read_dir()?.next().is_some();
    if occupied {
        let manifest = out.join("Cargo.toml");
        let generated = std::fs::read_to_string(&manifest)
            .map(|text| text.starts_with(convert::GENERATED_MANIFEST_HEADER))
            .unwrap_or(false);
        if !generated {
            bail!(
                "rypip run rewrites `{}`, but it is not empty and not a crate rypip generated \
                 (no {} starting with `{}`); give --out a fresh directory",
                out.display(),
                manifest.display(),
                convert::GENERATED_MANIFEST_HEADER
            );
        }
    }
    Ok(())
}

/// `rypip run` in one call (issue #166: CPython's command-line shape over
/// the convert/build pipeline, with no new semantics): take the work
/// dir's lock, remove the files the previous run wrote under `src/` (so a
/// module the program no longer has leaves no stale file behind, while
/// nothing rypip did not write is ever deleted; cargo's `target/` stays
/// for incremental rebuilds), convert, record the files written, hand the
/// crate to `on_converted` (the CLI reports its warnings there), build for
/// the host, stage the executable for this invocation, release the lock,
/// then execute with `program` as `sys.argv[0]` and `args`. The one
/// workflow the CLI and a library caller share (Devin review on #340,
/// rounds 2 to 5).
pub fn run(
    pkg: &PyPackage,
    out: &Path,
    options: &ConvertOptions,
    program: &OsStr,
    args: &[OsString],
    on_converted: impl FnOnce(&ConvertedCrate),
) -> Result<ExitStatus> {
    let out = physical_path(out)?;
    let lock = lock_work_dir(&out)?;
    refuse_foreign_output(pkg, &out)?;
    remove_generated_files(&out)?;
    let before = files_under(&out.join("src"))?;
    let started = std::time::SystemTime::now();
    let krate = convert(pkg, &out, options)?;
    record_generated_files(&out, &before, started)?;
    on_converted(&krate);
    let binary = cargo_build_executable(&krate)?;
    let staged = stage_executable(&out, &binary)?;
    drop(lock);
    run_program(staged.path(), program, args)
}

/// Where a run records the files it wrote under `src/`, relative to the
/// work dir, one per line.
fn generated_list_path(out: &Path) -> PathBuf {
    out.join(".run").join("generated")
}

/// Remove exactly the files the previous run recorded as its own under
/// `src/` — never a whole tree, never a file rypip did not write — so a
/// module the program no longer has leaves nothing behind while a forged
/// manifest marker cannot turn a run into `rm -rf src` (Devin review on
/// #340, round 5). Each recorded path must be a plain relative path
/// under `src/` (no `..`, no absolute component) naming a regular file;
/// anything else is ignored.
fn remove_generated_files(out: &Path) -> Result<()> {
    let list = generated_list_path(out);
    let Ok(text) = std::fs::read_to_string(&list) else {
        return Ok(());
    };
    for line in text.lines().filter(|l| !l.is_empty()) {
        let relative = Path::new(line);
        let plain = relative.components().all(|c| matches!(c, std::path::Component::Normal(_)));
        if !plain || !relative.starts_with("src") {
            continue;
        }
        let file = out.join(relative);
        if std::fs::symlink_metadata(&file).map(|m| m.is_file()).unwrap_or(false) {
            std::fs::remove_file(&file).with_context(|| format!("removing {}", file.display()))?;
        }
    }
    Ok(())
}

/// Every regular file under `dir`, recursively (empty when it does not
/// exist).
fn files_under(dir: &Path) -> Result<std::collections::HashSet<PathBuf>> {
    fn walk(dir: &Path, into: &mut std::collections::HashSet<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(&path, into)?;
            } else {
                into.insert(path);
            }
        }
        Ok(())
    }
    let mut files = std::collections::HashSet::new();
    if dir.is_dir() {
        walk(dir, &mut files)?;
    }
    Ok(files)
}

/// Record the files THIS conversion wrote under `src/`: those that did
/// not exist before it, plus those it rewrote (modified at or after
/// `started`). A file that was there before and untouched — a user's
/// file in an admitted crate — is never recorded, so the next run never
/// deletes it (Devin review on #340, round 6). A coarse file-system
/// timestamp can only under-record (a stale file left behind), never
/// claim a file rypip did not write.
fn record_generated_files(
    out: &Path,
    before: &std::collections::HashSet<PathBuf>,
    started: std::time::SystemTime,
) -> Result<()> {
    let mut files: Vec<String> = files_under(&out.join("src"))?
        .into_iter()
        .filter(|path| {
            !before.contains(path)
                || std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .map(|modified| modified >= started)
                    .unwrap_or(false)
        })
        .filter_map(|path| path.strip_prefix(out).ok().map(|r| r.to_string_lossy().into_owned()))
        .collect();
    files.sort();
    let list = generated_list_path(out);
    if let Some(parent) = list.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&list, files.join("\n") + "\n").with_context(|| format!("writing {}", list.display()))
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
