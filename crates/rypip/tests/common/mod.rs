//! Shared helpers for rypip integration tests.

use std::fs;
use std::path::{Path, PathBuf};

/// A scratch directory that's removed when dropped.
///
/// Created under the OS temp directory when possible, otherwise under the
/// workspace's gitignored `target/tmp`: some sandboxed environments deny
/// the platform temp area outright (`/var/folders/...` on macOS), which
/// used to fail every test needing a scratch directory even though the
/// workspace itself is writable.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        let name = format!("rypip-test-{tag}-{}", std::process::id());
        let primary = std::env::temp_dir().join(&name);
        let dir = if fs::create_dir_all(&primary).is_ok() {
            primary
        } else {
            let fallback = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/tmp")
                .join(&name);
            fs::create_dir_all(&fallback).unwrap_or_else(|e| {
                panic!("creating scratch dir {name} in the OS temp dir and under target/tmp both failed: {e}")
            });
            fallback
        };
        // Resolve `..` components so path-based consumers see a clean route.
        Scratch(fs::canonicalize(&dir).unwrap_or(dir))
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
