//! The converter's build identity, for the JIT's cache key (issue #166,
//! docs/jit.md "The cache"): a hash of the converter's sources — rypip's
//! own, python-ast's (the lowering) and rython-jit's (the boundary text
//! generated libraries compile) — when they sit beside this crate, as in
//! the rython workspace. The crate version alone does not change when a
//! checkout's codegen does, and a cache keyed on it would serve libraries
//! built by the old converter. An installed rypip (its sibling sources
//! absent) is identified by its version, which the key also carries.

use std::path::Path;

/// 64-bit FNV-1a: no dependency, and only accidental change detection is
/// asked of it.
struct Fnv(u64);

impl Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn hash_dir(dir: &Path, root: &Path, hash: &mut Fnv) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {}", dir.display(), e))
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            hash_dir(&path, root, hash);
        } else {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            hash.write(relative.to_string_lossy().as_bytes());
            hash.write(&[0]);
            let contents = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("reading {}: {}", path.display(), e));
            hash.write(&(contents.len() as u64).to_le_bytes());
            hash.write(&contents);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let manifest_dir = Path::new(&manifest_dir);
    let mut hash = Fnv(0xcbf29ce484222325);
    for dir in ["src", "../python-ast/src", "../rython-jit/src"] {
        let path = manifest_dir.join(dir);
        hash.write(dir.as_bytes());
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            hash_dir(&path, &path, &mut hash);
        } else {
            hash.write(b"\0absent");
        }
    }
    // Source-identical compiler builds with different dependency resolutions
    // must not share generated libraries. Include manifests and lockfiles.
    for file in ["Cargo.toml", "Cargo.lock", "../../Cargo.toml", "../../Cargo.lock",
        "../python-ast/Cargo.toml", "../rython-jit/Cargo.toml", "../stdpython/Cargo.toml"]
    {
        let path = manifest_dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        hash.write(file.as_bytes());
        if path.is_file() {
            hash.write(&std::fs::read(&path).expect("reading a build manifest/lockfile"));
        }
    }
    println!("cargo:rustc-env=RYPIP_CONVERTER_BUILD={:016x}", hash.0);
}
