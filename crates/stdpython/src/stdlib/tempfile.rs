//! Python tempfile module implementation
//!
//! This module provides facilities for creating temporary files and directories.
//! Implementation matches Python's tempfile module API.

use crate::PyException;
use std::path::{Path, PathBuf};

// CPython's tempfile does not trust a single env var: it probes candidates
// for writability (verified against python3 3.14,
// `tempfile._candidate_tempdir_list` + `_get_default_tempdir`). $TMPDIR,
// $TEMP and $TMP are tried in order (empty values skipped), then `/tmp`,
// `/var/tmp`, `/usr/tmp`, then the current directory; a candidate counts
// only if a freshly created file can be written and removed there. Rust's
// `std::env::temp_dir()` instead returns TMPDIR or, on macOS, the Darwin
// per-user temp dir with no probe at all — silently different from CPython
// whenever TMPDIR is unset or unusable (e.g. sandboxed environments deny
// /var/folders/...). The functions below mirror CPython's algorithm.

/// The directories CPython's tempfile would try, in order.
#[cfg(feature = "std")]
fn candidate_tempdir_list() -> Vec<PathBuf> {
    let mut list = Vec::new();
    for name in ["TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            if !value.is_empty() {
                list.push(PathBuf::from(value));
            }
        }
    }
    // OS-specific defaults (CPython tries these on POSIX).
    #[cfg(unix)]
    list.extend(["/tmp", "/var/tmp", "/usr/tmp"].map(PathBuf::from));
    // As a last resort, the current directory.
    match std::env::current_dir() {
        Ok(cwd) => list.push(cwd),
        Err(_) => list.push(PathBuf::from(".")),
    }
    list
}

/// Absolute path without resolving symlinks: CPython uses `abspath`, so
/// `/tmp` stays `/tmp` instead of becoming macOS's `/private/tmp`.
#[cfg(feature = "std")]
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// Whether a directory accepts a created-written-removed file — CPython's
/// usability probe, trying up to 100 random names per directory.
#[cfg(feature = "std")]
fn accepts_probe_file(dir: &Path) -> bool {
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    for _ in 0..100 {
        let filename = dir.join(generate_random_string(8));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&filename) {
            Ok(mut file) => {
                let written = file.write_all(b"blat").is_ok();
                drop(file);
                // Unlink unconditionally — CPython never leaves the probe
                // file behind, even when the write failed (Devin review
                // on PR #141).
                let removed = std::fs::remove_file(&filename).is_ok();
                return written && removed;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => break,
        }
    }
    false
}

/// First usable candidate, mirroring CPython `_get_default_tempdir`.
#[cfg(feature = "std")]
fn first_usable_tempdir(dirlist: &[PathBuf]) -> Option<PathBuf> {
    dirlist
        .iter()
        .map(|candidate| absolutize(candidate))
        .find(|dir| accepts_probe_file(dir))
}

/// The default temp dir, or the `FileNotFoundError` CPython raises when no
/// candidate works ("No usable temporary directory found in [...]").
#[cfg(feature = "std")]
pub fn try_default_tempdir() -> Result<PathBuf, PyException> {
    let dirlist = candidate_tempdir_list();
    first_usable_tempdir(&dirlist).ok_or_else(|| {
        PyException::new(
            "FileNotFoundError",
            format!("No usable temporary directory found in {:?}", dirlist),
        )
    })
}

/// The default temp dir with CPython's cache-on-first-success semantics
/// (`tempfile.tempdir` is set once and reused; failures are not cached).
#[cfg(feature = "std")]
fn cached_default_tempdir() -> Result<PathBuf, PyException> {
    static CACHE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    if let Some(dir) = CACHE.get() {
        return Ok(dir.clone());
    }
    let dir = try_default_tempdir()?;
    let _ = CACHE.set(dir.clone());
    Ok(dir)
}

/// Get temporary directory
#[cfg(feature = "std")]
pub fn gettempdir() -> PathBuf {
    // CPython raises FileNotFoundError here; this signature cannot carry
    // one, so panic loudly with the same message instead of guessing.
    cached_default_tempdir().unwrap_or_else(|e| panic!("{}", e.message))
}

/// Get temporary directory as string
#[cfg(feature = "std")]
pub fn gettempdir_str() -> String {
    gettempdir().to_string_lossy().to_string()
}

/// Generate temporary filename
#[cfg(feature = "std")]
pub fn mktemp(suffix: Option<&str>, prefix: Option<&str>, dir: Option<&str>) -> String {
    let mut path = if let Some(dir) = dir {
        PathBuf::from(dir)
    } else {
        gettempdir()
    };

    let prefix = prefix.unwrap_or("tmp");
    let suffix = suffix.unwrap_or("");

    // Generate random component
    let random_part = generate_random_string(8);
    let filename = format!("{}{}{}", prefix, random_part, suffix);

    path.push(filename);
    path.to_string_lossy().to_string()
}

/// Create and open temporary file
#[cfg(feature = "std")]
pub fn mkstemp(
    suffix: Option<&str>,
    prefix: Option<&str>,
    dir: Option<&str>,
    _text: bool,
) -> Result<(i32, String), PyException> {
    use std::fs::OpenOptions;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    #[cfg(unix)]
    use std::os::unix::io::AsRawFd;

    let mut attempts = 0;
    let max_attempts = 1000;

    // Resolve the default directory up front so an unusable one raises
    // FileNotFoundError (like CPython) instead of panicking in gettempdir.
    let resolved_dir: Option<String> = match dir {
        Some(d) => Some(d.to_string()),
        None => Some(cached_default_tempdir()?.to_string_lossy().into_owned()),
    };

    while attempts < max_attempts {
        let filename = mktemp(suffix, prefix, resolved_dir.as_deref());

        let mut open_options = OpenOptions::new();
        open_options.read(true).write(true).create_new(true);

        #[cfg(unix)]
        open_options.mode(0o600);

        match open_options.open(&filename) {
            Ok(file) => {
                #[cfg(unix)]
                let fd = file.as_raw_fd();
                #[cfg(windows)]
                let fd = 0; // Placeholder for Windows - would need proper implementation
                #[cfg(not(any(unix, windows)))]
                let fd = -1; // Fallback for other platforms

                std::mem::forget(file); // Don't close the file
                return Ok((fd, filename));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                attempts += 1;
                continue;
            }
            Err(e) => {
                return Err(crate::runtime_error(format!(
                    "Failed to create temporary file: {}",
                    e
                )));
            }
        }
    }

    Err(crate::runtime_error(
        "Failed to create temporary file after maximum attempts",
    ))
}

/// Create temporary directory
#[cfg(feature = "std")]
pub fn mkdtemp(
    suffix: Option<&str>,
    prefix: Option<&str>,
    dir: Option<&str>,
) -> Result<String, PyException> {
    let mut attempts = 0;
    let max_attempts = 1000;

    // Resolve the default directory up front so an unusable one raises
    // FileNotFoundError (like CPython) instead of panicking in gettempdir.
    let base = match dir {
        Some(d) => PathBuf::from(d),
        None => cached_default_tempdir()?,
    };

    while attempts < max_attempts {
        let mut path = base.clone();

        let prefix = prefix.unwrap_or("tmp");
        let suffix = suffix.unwrap_or("");
        let random_part = generate_random_string(8);
        let dirname = format!("{}{}{}", prefix, random_part, suffix);

        path.push(dirname);

        match std::fs::create_dir(&path) {
            Ok(_) => return Ok(path.to_string_lossy().to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                attempts += 1;
                continue;
            }
            Err(e) => {
                return Err(crate::runtime_error(format!(
                    "Failed to create temporary directory: {}",
                    e
                )));
            }
        }
    }

    Err(crate::runtime_error(
        "Failed to create temporary directory after maximum attempts",
    ))
}

/// Temporary file context manager
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct NamedTemporaryFile {
    file: Option<std::fs::File>,
    path: PathBuf,
    delete_on_drop: bool,
}

#[cfg(feature = "std")]
impl NamedTemporaryFile {
    /// Create new temporary file
    pub fn new(
        _mode: Option<&str>,
        _buffering: Option<i32>,
        encoding: Option<&str>,
        _newline: Option<&str>,
        suffix: Option<&str>,
        prefix: Option<&str>,
        dir: Option<&str>,
        delete: bool,
    ) -> Result<Self, PyException> {
        let (_, path_str) = mkstemp(suffix, prefix, dir, encoding.is_some())?;
        let path = PathBuf::from(&path_str);

        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| crate::runtime_error(format!("Failed to open temporary file: {}", e)))?;

        Ok(Self {
            file: Some(file),
            path,
            delete_on_drop: delete,
        })
    }

    /// Get file name
    pub fn name(&self) -> String {
        self.path.to_string_lossy().to_string()
    }

    /// Read from file
    pub fn read(&mut self, size: Option<usize>) -> Result<Vec<u8>, PyException> {
        use std::io::Read;

        if let Some(ref mut file) = self.file {
            let mut buffer = Vec::new();
            if let Some(size) = size {
                buffer.resize(size, 0);
                let bytes_read = file.read(&mut buffer).map_err(|e| {
                    crate::runtime_error(format!("Failed to read from temporary file: {}", e))
                })?;
                buffer.truncate(bytes_read);
            } else {
                file.read_to_end(&mut buffer).map_err(|e| {
                    crate::runtime_error(format!("Failed to read from temporary file: {}", e))
                })?;
            }
            Ok(buffer)
        } else {
            Err(crate::value_error("File is closed"))
        }
    }

    /// Write to file
    pub fn write(&mut self, data: &[u8]) -> Result<usize, PyException> {
        use std::io::Write;

        if let Some(ref mut file) = self.file {
            file.write(data).map_err(|e| {
                crate::runtime_error(format!("Failed to write to temporary file: {}", e))
            })
        } else {
            Err(crate::value_error("File is closed"))
        }
    }

    /// Flush file
    pub fn flush(&mut self) -> Result<(), PyException> {
        use std::io::Write;

        if let Some(ref mut file) = self.file {
            file.flush()
                .map_err(|e| crate::runtime_error(format!("Failed to flush temporary file: {}", e)))
        } else {
            Err(crate::value_error("File is closed"))
        }
    }

    /// Close file
    pub fn close(&mut self) -> Result<(), PyException> {
        if self.file.is_some() {
            self.file = None;
            Ok(())
        } else {
            Err(crate::value_error("File already closed"))
        }
    }

    /// Seek in file
    pub fn seek(&mut self, offset: i64, whence: i32) -> Result<u64, PyException> {
        use std::io::{Seek, SeekFrom};

        if let Some(ref mut file) = self.file {
            let seek_from = match whence {
                0 => SeekFrom::Start(offset as u64),
                1 => SeekFrom::Current(offset),
                2 => SeekFrom::End(offset),
                _ => return Err(crate::value_error("Invalid whence value")),
            };

            file.seek(seek_from).map_err(|e| {
                crate::runtime_error(format!("Failed to seek in temporary file: {}", e))
            })
        } else {
            Err(crate::value_error("File is closed"))
        }
    }
}

#[cfg(feature = "std")]
impl Drop for NamedTemporaryFile {
    fn drop(&mut self) {
        if self.delete_on_drop {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Temporary directory context manager
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct TemporaryDirectory {
    path: Option<PathBuf>,
}

#[cfg(feature = "std")]
impl TemporaryDirectory {
    /// Create new temporary directory
    pub fn new(
        suffix: Option<&str>,
        prefix: Option<&str>,
        dir: Option<&str>,
    ) -> Result<Self, PyException> {
        let path_str = mkdtemp(suffix, prefix, dir)?;
        Ok(Self {
            path: Some(PathBuf::from(path_str)),
        })
    }

    /// Get directory name
    pub fn name(&self) -> Option<String> {
        self.path.as_ref().map(|p| p.to_string_lossy().to_string())
    }

    /// Cleanup directory
    pub fn cleanup(&mut self) -> Result<(), PyException> {
        if let Some(path) = self.path.take() {
            std::fs::remove_dir_all(&path).map_err(|e| {
                crate::runtime_error(format!("Failed to remove temporary directory: {}", e))
            })?;
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// SpooledTemporaryFile - temporary file that starts in memory
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct SpooledTemporaryFile {
    data: Vec<u8>,
    position: usize,
    max_size: usize,
    file: Option<NamedTemporaryFile>,
}

#[cfg(feature = "std")]
impl SpooledTemporaryFile {
    /// Create new spooled temporary file
    pub fn new(
        max_size: Option<usize>,
        _mode: Option<&str>,
        _buffering: Option<i32>,
        _encoding: Option<&str>,
        _newline: Option<&str>,
        _suffix: Option<&str>,
        _prefix: Option<&str>,
        _dir: Option<&str>,
    ) -> Self {
        Self {
            data: Vec::new(),
            position: 0,
            max_size: max_size.unwrap_or(5000),
            file: None,
        }
    }

    /// Write data
    pub fn write(&mut self, data: &[u8]) -> Result<usize, PyException> {
        if self.file.is_some() {
            return self.file.as_mut().unwrap().write(data);
        }

        // Check if we need to roll over to file
        if self.data.len() + data.len() > self.max_size {
            self.rollover()?;
            return self.file.as_mut().unwrap().write(data);
        }

        self.data.extend_from_slice(data);
        Ok(data.len())
    }

    /// Read data
    pub fn read(&mut self, size: Option<usize>) -> Result<Vec<u8>, PyException> {
        if let Some(ref mut file) = self.file {
            return file.read(size);
        }

        let available = self.data.len().saturating_sub(self.position);
        let to_read = size.map(|s| s.min(available)).unwrap_or(available);

        if to_read == 0 {
            return Ok(Vec::new());
        }

        let result = self.data[self.position..self.position + to_read].to_vec();
        self.position += to_read;
        Ok(result)
    }

    /// Seek in file
    pub fn seek(&mut self, offset: i64, whence: i32) -> Result<u64, PyException> {
        if let Some(ref mut file) = self.file {
            return file.seek(offset, whence);
        }

        // A negative absolute offset must raise ValueError, not clamp to 0
        // via an `as usize` wrap (issue #82).
        let new_position = match whence {
            0 => {
                if offset < 0 {
                    return Err(crate::value_error(format!(
                        "negative seek value {}",
                        offset
                    )));
                }
                offset as usize
            }
            1 => {
                let p = self.position as i64 + offset;
                if p < 0 {
                    return Err(crate::value_error(format!("negative seek value {}", p)));
                }
                p as usize
            }
            2 => {
                let p = self.data.len() as i64 + offset;
                if p < 0 {
                    return Err(crate::value_error(format!("negative seek value {}", p)));
                }
                p as usize
            }
            _ => return Err(crate::value_error("Invalid whence value")),
        };

        self.position = new_position.min(self.data.len());
        Ok(self.position as u64)
    }

    /// Roll over to file
    fn rollover(&mut self) -> Result<(), PyException> {
        if self.file.is_some() {
            return Ok(());
        }

        let mut temp_file =
            NamedTemporaryFile::new(None, None, None, None, None, None, None, true)?;

        temp_file.write(&self.data)?;
        temp_file.seek(self.position as i64, 0)?;

        self.file = Some(temp_file);
        self.data.clear();
        self.position = 0;

        Ok(())
    }
}

// Helper functions

fn generate_random_string(length: usize) -> String {
    // Names draw 6 bits of cryptographic OS entropy per character,
    // independent of the seeded `random` module — like Python's tempfile,
    // which keeps its own private Random over os.urandom seeding. The old
    // implementation cycled 8 bits of one time-derived hash, so every call
    // in the same instant produced the SAME name and the mkstemp/mkdtemp
    // retry loops could burn all their attempts on one candidate.
    let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let char_bytes = chars.as_bytes();

    let mut out = String::with_capacity(length);
    let mut pool: u64 = 0;
    let mut pool_bits = 0u32;
    while out.len() < length {
        if pool_bits < 6 {
            let mut bytes = [0u8; 8];
            crate::random::os_entropy(&mut bytes);
            pool = u64::from_le_bytes(bytes);
            pool_bits = 64;
        }
        let idx = (pool & 0x3f) as usize;
        pool >>= 6;
        pool_bits -= 6;
        // 62 characters: indices 62/63 are rejection-sampled away so the
        // distribution stays uniform.
        if idx < char_bytes.len() {
            out.push(char_bytes[idx] as char);
        }
    }
    out
}

// Module constants
pub const TMP_MAX: usize = 10000;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_gettempdir() {
        let temp_dir = gettempdir();
        assert!(temp_dir.exists());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_mktemp() {
        let temp_file = mktemp(Some(".txt"), Some("test_"), None);
        assert!(temp_file.contains("test_"));
        assert!(temp_file.ends_with(".txt"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_mkdtemp() {
        let temp_dir = mkdtemp(None, Some("test_"), None).unwrap();
        assert!(std::path::Path::new(&temp_dir).exists());
        std::fs::remove_dir(&temp_dir).unwrap();
    }

    #[cfg(feature = "std")]
    #[test]
    fn unusable_candidates_fall_through_like_cpython() {
        // Verified against python3: `TMPDIR=/nonexistent-xyz python3 -c
        // "import tempfile; print(tempfile.gettempdir())"` still prints
        // /tmp — an unusable candidate is skipped, not trusted.
        let picked = first_usable_tempdir(&[
            PathBuf::from("/definitely-missing-rython-scratch"),
            PathBuf::from("/tmp"),
        ]);
        assert_eq!(picked, Some(PathBuf::from("/tmp")));
    }

    #[cfg(feature = "std")]
    #[test]
    fn probe_rejects_missing_directories() {
        // Verified against python3: a candidate that cannot hold a file is
        // never returned by _get_default_tempdir.
        assert!(!accepts_probe_file(Path::new(
            "/definitely-missing-rython-scratch/nested"
        )));
    }

    #[cfg(feature = "std")]
    #[test]
    fn candidate_list_orders_env_before_defaults() {
        // CPython's _candidate_tempdir_list: $TMPDIR, $TEMP, $TMP (each only
        // when non-empty), then the POSIX defaults, then the cwd last.
        let dirlist = candidate_tempdir_list();
        let env_names = ["TMPDIR", "TEMP", "TMP"];
        let env_count = env_names
            .iter()
            .filter(|n| std::env::var_os(n).map(|v| !v.is_empty()).unwrap_or(false))
            .count();
        for (index, name) in env_names.iter().enumerate().take(env_count) {
            let expected = std::env::var_os(name).unwrap();
            assert_eq!(
                dirlist[index],
                PathBuf::from(&expected),
                "env candidate {name} must be passed through in order"
            );
        }
        let defaults_at = env_count;
        assert_eq!(
            &dirlist[defaults_at..defaults_at + 3],
            ["/tmp", "/var/tmp", "/usr/tmp"].map(PathBuf::from),
            "POSIX defaults must follow the env candidates"
        );
        assert_eq!(
            dirlist.last(),
            std::env::current_dir().ok().as_ref(),
            "the cwd is the last-resort candidate"
        );
    }

    #[test]
    fn test_generate_random_string() {
        let s1 = generate_random_string(10);
        let s2 = generate_random_string(10);
        assert_eq!(s1.len(), 10);
        assert_eq!(s2.len(), 10);
        // They should be different (extremely unlikely to be same)
        assert_ne!(s1, s2);
    }
}
