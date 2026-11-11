//! The content-addressed cache (docs/jit.md, "The cache"):
//!
//! ```text
//! <root>/                    mode 0700 ($XDG_CACHE_HOME/rython/jit)
//!   target/                  the one CARGO_TARGET_DIR every entry builds in
//!   target.lock              held from a build through its artifact copy
//!   entries/<key>/           one compiled module
//!     crate/                 the generated crate
//!     lib-<n>.so             the library the host loads (never rewritten)
//!     entry                  written last: the full key and the artifact
//!   entries/<key>.lock       serializes builds of the entry
//! ```
//!
//! An entry is trusted only on an exact match of its recorded key text, and
//! only once its `entry` record exists (it is renamed into place after the
//! artifact). A record that does not match, an artifact of the wrong size,
//! or a library that will not load is rebuilt loudly, never used.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::JitError;

/// The default cache root: `$XDG_CACHE_HOME/rython/jit`, or
/// `$HOME/.cache/rython/jit` when XDG_CACHE_HOME is unset, empty or
/// relative (the XDG base-directory spec ignores a relative value).
pub fn default_cache_dir() -> Result<PathBuf, JitError> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let xdg = PathBuf::from(xdg);
        if xdg.is_absolute() {
            return Ok(xdg.join("rython").join("jit"));
        }
    }
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => Ok(PathBuf::from(home).join(".cache").join("rython").join("jit")),
        _ => Err(JitError::Environment(
            "the JIT cache has no default location: neither XDG_CACHE_HOME (absolute) nor HOME \
             is set; set one, or give the JIT a cache directory"
                .to_string(),
        )),
    }
}

/// Create `dir` (and missing parents) private to this user, or check that
/// an existing one is: the cache holds code the JIT loads and runs, so a
/// directory another user can write is a code-injection vector, refused.
#[cfg(unix)]
pub fn ensure_private_dir(dir: &Path) -> Result<(), JitError> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    check_ancestors(dir.parent().unwrap_or(dir))?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .map_err(|e| JitError::Environment(format!("creating the JIT cache {}: {}", dir.display(), e)))?;
    let meta = fs::symlink_metadata(dir)
        .map_err(|e| JitError::Environment(format!("reading {}: {}", dir.display(), e)))?;
    if !meta.is_dir() {
        return Err(JitError::Environment(format!(
            "the JIT cache {} is not a directory (a symlink or a file); it must be a \
             private directory the JIT creates",
            dir.display()
        )));
    }
    // SAFETY: geteuid has no preconditions and cannot fail.
    let me = unsafe { libc::geteuid() };
    if meta.uid() != me {
        return Err(JitError::Environment(format!(
            "the JIT cache {} belongs to another user (uid {}); it holds code the JIT \
             loads and runs, so it must be yours",
            dir.display(),
            meta.uid()
        )));
    }
    let mode = meta.mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(JitError::Environment(format!(
            "the JIT cache {} has mode {:o}; it holds code the JIT loads and runs, so it \
             must be private to you (chmod 700 {})",
            dir.display(),
            mode,
            dir.display()
        )));
    }
    Ok(())
}

/// Every ancestor must be controlled by this user or root. A private leaf
/// inside somebody else's writable directory is still replaceable. The
/// root-owned sticky temporary directory protects each user's child names.
#[cfg(unix)]
fn check_ancestors(dir: &Path) -> Result<(), JitError> {
    use std::os::unix::fs::MetadataExt;
    let path = std::path::absolute(dir)
        .map_err(|e| JitError::Environment(format!("resolving the cache path: {e}")))?;
    if path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(JitError::Environment("the JIT cache path must not contain `..`".into()));
    }
    // SAFETY: geteuid has no preconditions.
    let me = unsafe { libc::geteuid() };
    for ancestor in path.ancestors() {
        let meta = match fs::symlink_metadata(ancestor) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(JitError::Environment(format!("checking {}: {e}", ancestor.display()))),
        };
        let sticky_root = meta.uid() == 0 && meta.mode() & 0o1000 != 0;
        if !meta.is_dir() || (meta.uid() != me && meta.uid() != 0)
            || (meta.mode() & 0o022 != 0 && !sticky_root)
        {
            return Err(JitError::Environment(format!(
                "the JIT cache ancestor {} is replaceable by another user or is a symlink; \
                 use a cache under your own directories with no group/world write permissions",
                ancestor.display()
            )));
        }
    }
    Ok(())
}

/// This lock lives beside the cache, so clearing cannot unlink its inode.
/// The shared side covers every read, build and dlopen; clearing is exclusive.
#[cfg(unix)]
pub fn root_lock(root: &Path, exclusive: bool) -> Result<FileLock, JitError> {
    let parent = root.parent().ok_or_else(|| JitError::Environment("the cache needs a parent directory".into()))?;
    check_ancestors(parent)?;
    let name = root.file_name().ok_or_else(|| JitError::Environment("the filesystem root cannot be a JIT cache".into()))?;
    let name = format!(".{}.rython-jit.lock", name.to_string_lossy());
    lock(&parent.join(name), exclusive)
}

/// An advisory lock on a file, released when dropped (closing the file
/// releases it too). The same mechanism `rypip run` uses for its work
/// directories.
pub struct FileLock {
    file: fs::File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Lock `path` (created if missing): exclusively for a build, shared for
/// reading an entry.
pub fn lock(path: &Path, exclusive: bool) -> Result<FileLock, JitError> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = fs::OpenOptions::new();
    options.create(true).write(true).truncate(false);
    #[cfg(unix)]
    options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    let file = options.open(path)
        .map_err(|e| JitError::Environment(format!("opening the lock {}: {}", path.display(), e)))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()
            .map_err(|e| JitError::Environment(format!("checking lock {}: {e}", path.display())))?;
        // SAFETY: geteuid has no preconditions.
        let me = unsafe { libc::geteuid() };
        if !meta.is_file() || meta.uid() != me || meta.mode() & 0o022 != 0 {
            return Err(JitError::Environment(format!(
                "the JIT lock {} must be a regular file owned by you and not writable by others",
                path.display()
            )));
        }
    }
    let locked = if exclusive { file.lock() } else { file.lock_shared() };
    locked.map_err(|e| JitError::Environment(format!("locking {}: {}", path.display(), e)))?;
    Ok(FileLock { file })
}

/// The cache key: the full text of everything that can change the
/// artifact, and its digest (the entry's directory name).
pub struct Key {
    pub text: String,
    pub digest: String,
}

/// Hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{:02x}", b)).collect()
}

impl Key {
    /// Everything that can change the compiled library (docs/jit.md): the
    /// module name and source bytes, the compiler's fingerprint (converter
    /// version, the runtime it links and its features, conversion options),
    /// `rustc -vV` (which names the host triple), the build-affecting
    /// environment, and the boundary's version.
    pub fn new(module: &str, source: &str, compiler: &str, rustc: &str) -> Key {
        let mut text = String::new();
        text.push_str("rython-jit cache key 1\n");
        text.push_str(&format!("abi {}\n", crate::abi::RY_ABI_VERSION));
        text.push_str(&format!("module {}\n", module));
        text.push_str(&format!("source {} {}\n", source.len(), sha256_hex(source.as_bytes())));
        text.push_str("compiler\n");
        text.push_str(compiler.trim_end());
        text.push('\n');
        text.push_str("rustc\n");
        text.push_str(rustc.trim_end());
        text.push('\n');
        for var in BUILD_ENV {
            if let Some(value) = std::env::var_os(var) {
                text.push_str(&format!("env {}={:?}\n", var, value));
            }
        }
        text.push_str("profile release\n");
        let digest = sha256_hex(text.as_bytes())[..32].to_string();
        Key { text, digest }
    }
}

/// Environment variables that change what cargo builds (cargo's own
/// fingerprint tracks them; a cache that skips cargo must key on them).
/// Cargo configuration FILES are not keyed: see docs/jit.md.
const BUILD_ENV: &[&str] = &[
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_PROFILE_RELEASE_OPT_LEVEL",
    "CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS",
    "CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS",
    "CARGO_PROFILE_RELEASE_PANIC",
    "CARGO_PROFILE_RELEASE_LTO",
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
];

/// `rustc -vV`, run where the builds run (the cache root) so a rustup
/// directory override resolves as it does for cargo there; `$RUSTC` first,
/// as cargo does.
pub fn rustc_version(cache_root: &Path) -> Result<String, JitError> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(&rustc)
        .arg("-vV")
        .current_dir(cache_root)
        .output()
        .map_err(|e| {
            JitError::Environment(format!(
                "the Rust toolchain is missing: running `{} -vV` (from $RUSTC, else `rustc` on \
                 PATH) failed: {}; the JIT compiles with rustc and cargo — install them \
                 (https://rustup.rs) or point RUSTC at rustc",
                rustc.to_string_lossy(),
                e
            ))
        })?;
    if !output.status.success() {
        return Err(JitError::Environment(format!(
            "`{} -vV` failed with {}: {}",
            rustc.to_string_lossy(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One cache entry's paths.
pub struct Entry {
    pub dir: PathBuf,
    pub lock: PathBuf,
}

impl Entry {
    pub fn new(root: &Path, key: &Key) -> Entry {
        let entries = root.join("entries");
        Entry {
            dir: entries.join(&key.digest),
            lock: entries.join(format!("{}.lock", key.digest)),
        }
    }

    pub fn crate_dir(&self) -> PathBuf {
        self.dir.join("crate")
    }

    fn record(&self) -> PathBuf {
        self.dir.join("entry")
    }
}

/// What an `entry` record says: the artifact to load, its size, and the
/// conversion's warnings (reported on every load, as `rypip run` reports
/// them on every run).
#[derive(Debug)]
pub struct Record {
    pub artifact: PathBuf,
    pub warnings: Vec<String>,
}

const RECORD_HEADER: &str = "rython-jit entry 2";

/// Read and check an entry's record. `Ok(None)`: no complete entry (a
/// miss). `Err`: a record that does not match — corrupt, or another key's
/// (a digest-prefix collision) — so the entry must be rebuilt.
pub fn read_record(entry: &Entry, key: &Key) -> Result<Option<Record>, String> {
    let text = match fs::read_to_string(entry.record()) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("reading its record: {}", e)),
    };
    let mut lines = text.split('\n');
    if lines.next() != Some(RECORD_HEADER) {
        return Err("its record has no header".to_string());
    }
    let field = |line: Option<&str>, name: &str| -> Result<String, String> {
        line.and_then(|l| l.strip_prefix(name))
            .and_then(|l| l.strip_prefix(' '))
            .map(str::to_string)
            .ok_or_else(|| format!("its record has no `{}`", name))
    };
    let artifact_name = field(lines.next(), "artifact")?;
    let size: u64 = field(lines.next(), "size")?
        .parse()
        .map_err(|_| "its record has a malformed size".to_string())?;
    let digest = field(lines.next(), "sha256")?;
    let dependencies = field(lines.next(), "dependencies")?;
    let warning_count: usize = field(lines.next(), "warnings")?
        .parse()
        .map_err(|_| "its record has a malformed warning count".to_string())?;
    let mut warnings = Vec::with_capacity(warning_count);
    for _ in 0..warning_count {
        warnings.push(field(lines.next(), "warning")?);
    }
    if lines.next() != Some("key") {
        return Err("its record has no key".to_string());
    }
    let recorded_key: Vec<&str> = lines.collect();
    if recorded_key.join("\n") != key.text {
        return Err("its recorded key does not match".to_string());
    }
    if artifact_name.contains('/') || artifact_name.starts_with('.') {
        return Err("its record names an artifact outside the entry".to_string());
    }
    let artifact = entry.dir.join(&artifact_name);
    let metadata = fs::symlink_metadata(&artifact)
        .map_err(|e| format!("its library {}: {}", artifact.display(), e))?
        ;
    if !metadata.is_file() {
        return Err("its library is not a regular file".into());
    }
    let actual = metadata.len();
    if actual != size {
        return Err(format!(
            "its library {} is {} bytes, not the {} recorded",
            artifact.display(),
            actual,
            size
        ));
    }
    let bytes = fs::read(&artifact).map_err(|e| format!("reading its library: {e}"))?;
    if sha256_hex(&bytes) != digest {
        return Err("its library's SHA-256 does not match the record".into());
    }
    if dependency_identity(entry).map_err(|e| e.to_string())? != dependencies {
        return Err("its pinned Cargo.lock has changed".into());
    }
    Ok(Some(Record { artifact, warnings }))
}

/// Remove an entry's record, so the entry reads as a miss.
pub fn invalidate(entry: &Entry) -> std::io::Result<()> {
    match fs::remove_file(entry.record()) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Write an entry's record (atomically: a temporary file renamed into
/// place, after the artifact it names exists).
pub fn write_record(
    entry: &Entry,
    key: &Key,
    artifact: &Path,
    warnings: &[String],
) -> Result<(), JitError> {
    let io = |e: std::io::Error| JitError::Environment(format!("writing the JIT cache entry {}: {}", entry.dir.display(), e));
    let size = fs::metadata(artifact).map_err(io)?.len();
    let digest = sha256_hex(&fs::read(artifact).map_err(io)?);
    let dependencies = dependency_identity(entry).map_err(io)?;
    let name = artifact
        .file_name()
        .ok_or_else(|| JitError::Environment(format!("{} has no file name", artifact.display())))?
        .to_string_lossy()
        .into_owned();
    let mut text = format!("{}\nartifact {}\nsize {}\nsha256 {}\ndependencies {}\nwarnings {}\n", RECORD_HEADER, name, size, digest, dependencies, warnings.len());
    for warning in warnings {
        text.push_str("warning ");
        text.push_str(&warning.replace(['\n', '\r'], " "));
        text.push('\n');
    }
    text.push_str("key\n");
    text.push_str(&key.text);
    let tmp = entry.dir.join("entry.tmp");
    let mut file = fs::File::create(&tmp).map_err(io)?;
    file.write_all(text.as_bytes()).map_err(io)?;
    file.sync_all().map_err(io)?;
    fs::rename(&tmp, entry.record()).map_err(io)?;
    Ok(())
}

fn dependency_identity(entry: &Entry) -> std::io::Result<String> {
    fs::read(entry.crate_dir().join("Cargo.lock")).map(|bytes| sha256_hex(&bytes))
}

/// Copy a freshly built library into its entry under a name no earlier
/// build of the entry used: a process that loaded an earlier artifact
/// keeps it (it is never rewritten in place), and the dynamic loader never
/// mistakes the new file for one it already has open under the same path.
pub fn install_artifact(entry: &Entry, built: &Path) -> Result<PathBuf, JitError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let io = |e: std::io::Error| {
        JitError::Environment(format!(
            "copying the built library {} into the JIT cache: {}",
            built.display(),
            e
        ))
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!(
        "lib-{}-{}-{}",
        nanos,
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let tmp = entry.dir.join(format!("{}.tmp", stem));
    let dest = entry
        .dir
        .join(format!("{}{}", stem, std::env::consts::DLL_SUFFIX));
    fs::copy(built, &tmp).map_err(io)?;
    fs::File::open(&tmp).and_then(|f| f.sync_all()).map_err(io)?;
    fs::rename(&tmp, &dest).map_err(io)?;
    Ok(dest)
}

/// Regenerate an entry while retaining its dependency pins. Remove the
/// previous source and artifacts (loaded libraries stay mapped), but keep
/// Cargo.lock so artifact corruption cannot silently upgrade dependencies.
pub fn reset_entry(entry: &Entry) -> Result<(), JitError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&entry.dir)
            .map_err(|e| JitError::Environment(format!("creating {}: {}", entry.dir.display(), e)))?;
    }
    let crate_dir = entry.crate_dir();
    let pinned = match fs::read(crate_dir.join("Cargo.lock")) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(JitError::Environment(format!("reading dependency pins: {e}"))),
    };
    if crate_dir.exists() {
        fs::remove_dir_all(&crate_dir)
            .map_err(|e| JitError::Environment(format!("removing {}: {}", crate_dir.display(), e)))?;
    }
    if let Some(pinned) = pinned {
        fs::create_dir_all(&crate_dir)
            .and_then(|_| fs::write(crate_dir.join("Cargo.lock"), pinned))
            .map_err(|e| JitError::Environment(format!("retaining dependency pins: {e}")))?;
    }
    if let Ok(read) = fs::read_dir(&entry.dir) {
        for item in read.flatten() {
            let name = item.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("lib-") {
                let _ = fs::remove_file(item.path());
            }
        }
    }
    Ok(())
}

/// Remove the whole cache: every entry and the shared target directory
/// (`rypip jit --clear-cache`). Libraries already loaded stay mapped.
pub fn clear(root: &Path) -> Result<(), JitError> {
    // A missing root is already empty. This is the operation's
    // linearization point: a subsequent concurrent creator owns new work.
    if let Err(e) = fs::symlink_metadata(root) {
        if e.kind() == std::io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(JitError::Environment(format!("checking the JIT cache: {e}")));
    }
    #[cfg(unix)]
    let _guard = {
        ensure_private_dir(root)?;
        let canonical = fs::canonicalize(root)
            .map_err(|e| JitError::Environment(format!("resolving {}: {e}", root.display())))?;
        root_lock(&canonical, true)?
    };
    match fs::remove_dir_all(root) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(JitError::Environment(format!(
            "clearing the JIT cache {}: {}",
            root.display(),
            e
        ))),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!("jit-cache-pin-{}-{}",
                std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    #[test]
    fn private_leaf_under_writable_or_symlinked_parent_is_refused() {
        let scratch = Scratch::new();
        let parent = scratch.0.join("replaceable");
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        let root = parent.join("cache");
        ensure_private_dir(&root).unwrap();
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(ensure_private_dir(&root).unwrap_err().to_string().contains("ancestor"));
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&parent, scratch.0.join("alias")).unwrap();
        assert!(ensure_private_dir(&scratch.0.join("alias/cache")).is_err());
    }

    #[test]
    fn clearing_waits_for_readers_and_builders_without_unlinking_the_root_lock() {
        let scratch = Scratch::new();
        let root = scratch.0.join("cache");
        ensure_private_dir(&root).unwrap();
        fs::write(root.join("active-build"), b"working").unwrap();
        let guard = root_lock(&root, false).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let clearing = root.clone();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            clear(&clearing).unwrap();
            done_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(std::time::Duration::from_millis(100)).is_err());
        assert!(root.join("active-build").exists());
        drop(guard);
        done_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert!(!root.exists());
        assert!(scratch.0.join(".cache.rython-jit.lock").is_file());
        ensure_private_dir(&root).unwrap();
        root_lock(&root, false).unwrap();
    }

    #[test]
    fn equal_size_artifact_changes_and_changed_dependency_pins_are_rejected() {
        let scratch = Scratch::new();
        let key = Key::new("example", "", "compiler", "rustc");
        let entry = Entry::new(&scratch.0, &key);
        fs::create_dir_all(entry.crate_dir()).unwrap();
        fs::write(entry.crate_dir().join("Cargo.lock"), "pinned dependency graph").unwrap();
        let artifact = entry.dir.join("lib-test.so");
        fs::write(&artifact, b"trusted").unwrap();
        write_record(&entry, &key, &artifact, &[]).unwrap();
        assert!(read_record(&entry, &key).unwrap().is_some());
        fs::write(&artifact, b"changed").unwrap();
        assert!(read_record(&entry, &key).unwrap_err().contains("SHA-256"));
        fs::write(&artifact, b"trusted").unwrap();
        fs::write(entry.crate_dir().join("Cargo.lock"), "different dependency graph").unwrap();
        assert!(read_record(&entry, &key).unwrap_err().contains("Cargo.lock"));
    }

    #[test]
    fn regeneration_retains_the_exact_dependency_lockfile() {
        let scratch = Scratch::new();
        let key = Key::new("example", "", "compiler", "rustc");
        let entry = Entry::new(&scratch.0, &key);
        fs::create_dir_all(entry.crate_dir()).unwrap();
        let pinned = b"# an existing exact dependency resolution\n";
        fs::write(entry.crate_dir().join("Cargo.lock"), pinned).unwrap();
        fs::write(entry.crate_dir().join("old-source.rs"), b"old").unwrap();
        reset_entry(&entry).unwrap();
        assert_eq!(fs::read(entry.crate_dir().join("Cargo.lock")).unwrap(), pinned);
        assert!(!entry.crate_dir().join("old-source.rs").exists());
    }

    #[test]
    fn clearing_an_absent_cache_under_a_read_only_parent_is_a_noop() {
        let scratch = Scratch::new();
        fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o500)).unwrap();
        let result = clear(&scratch.0.join("missing/rython/jit"));
        fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o700)).unwrap();
        result.unwrap();
        assert!(!scratch.0.join("missing").exists());
        assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 0);
    }
}
