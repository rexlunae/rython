//! Shared helpers for stdpython integration tests.

use std::fs;
use std::path::{Path, PathBuf};

/// Create and return a fresh scratch directory named
/// `rython-scratch-{tag}-{pid}`.
///
/// Prefers the OS temp directory, but falls back to the workspace's
/// gitignored `target/tmp` when the platform temp area is not writable:
/// some sandboxed environments deny `/var/folders/...` (macOS) outright,
/// which used to fail every test needing a scratch directory even though
/// the workspace itself is writable.
pub fn create_scratch(tag: &str) -> PathBuf {
    let name = format!("rython-scratch-{tag}-{}", std::process::id());
    let primary = std::env::temp_dir().join(&name);
    let created = if fs::create_dir_all(&primary).is_ok() {
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
    // Return the resolved path: the fallback route contains `..` components,
    // which rython's glob (and some tools) do not normalize.
    fs::canonicalize(&created).unwrap_or(created)
}
