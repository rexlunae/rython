//! Integration tests for import-based Rust modules: `import rustlib` and
//! `from rustlib import ...` resolve through the rython.toml manifest to
//! compile-time bindings into a Rust crate. Signatures come from a `.pyi`
//! stub when present, otherwise by inference from the crate's source.

use std::fs;
use std::path::Path;
use std::process::Command;

mod common;

use common::Scratch;
use rypip::convert::ConvertOptions;

/// Build a generated crate, scrubbing RUSTFLAGS (see rust_bind_tests).
fn build_generated(root: &Path) -> std::process::ExitStatus {
    Command::new("cargo")
        .arg("build")
        .env_remove("RUSTFLAGS")
        .current_dir(root)
        .status()
        .expect("running cargo build")
}

/// The Rust crate the fixtures import. Same shape as rust_bind_tests' lib so
/// inference has a variety of signatures to exercise (and skip).
fn write_rustlib_crate(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"rustlib\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/lib.rs"),
        concat!(
            "//! Test library for import-based FFI integration tests.\n\n",
            "/// CRC-8 over the data, seeded.\n",
            "pub fn crc8(data: &[u8], seed: u32) -> u32 {\n",
            "    let mut crc = seed;\n",
            "    for b in data {\n",
            "        crc = crc.wrapping_add(*b as u32).wrapping_mul(31);\n",
            "    }\n",
            "    crc\n",
            "}\n\n",
            "pub fn add(a: i64, b: i64) -> i64 { a + b }\n\n",
            "pub fn shout(s: &str) -> String { format!(\"{}!\", s.to_uppercase()) }\n\n",
            "pub fn half(x: f64) -> f64 { x / 2.0 }\n\n",
            "// Private: must NOT be bound by inference.\n",
            "fn secret() -> i64 { 42 }\n\n",
            "// Unsupported signature: must be skipped, not mis-lowered.\n",
            "pub fn generic<T: Default>() -> T { T::default() }\n",
        ),
    )
    .unwrap();
}

/// A package root with a rython.toml manifest pointing at the crate.
/// Manifest `path`s are relative to the package root (where the manifest
/// lives); the generated Cargo.toml dep is rebased onto the out dir.
fn write_manifest(root: &Path, extra: &str) {
    fs::write(
        root.join("rython.toml"),
        format!("[rust-modules]\nrustlib = {{ path = \"rustlib\" }}\n{extra}"),
    )
    .unwrap();
}

#[test]
fn rust_module_import_infers_signatures_and_runs() {
    let scratch = Scratch::new("infer");
    write_rustlib_crate(&scratch.path().join("rustlib"));
    write_manifest(scratch.path(), "");
    fs::write(
        scratch.path().join("app.py"),
        concat!(
            "import rustlib\n",
            "import rustlib as rl\n",
            "from rustlib import crc8 as c, add\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(rustlib.crc8(b\"hello\", 7))\n",
            "    print(rustlib.add(20, 22))\n",
            "    print(rl.shout(\"hi\"))\n",
            "    print(rl.half(9.0))\n",
            "    print(c(b\"abc\", 0))\n",
            "    print(add(1, 2))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("app.py")).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    // The manifest maps the import name to a dependency; the path is
    // rebased from the package root onto the generated crate.
    let manifest = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("rustlib = { path = \"../rustlib\" }"),
        "manifest must declare the imported crate: {}",
        manifest
    );

    // Imports lower to direct calls; the module name resolves as a crate
    // path. No `use rustlib;` — the crate is a dependency, not a sibling.
    let main_rs = fs::read_to_string(out.join("src/main.rs")).unwrap();
    for expected in [
        "rustlib::crc8(b\"hello\".to_vec().as_ref(), 7 as u32) as i64",
        "rustlib::add(20 as i64, 22 as i64)",
        "rustlib::shout(\"hi\".as_ref())",
        "rustlib::half(9.0)",
        "rustlib::crc8(b\"abc\".to_vec().as_ref(), 0 as u32) as i64",
        "rustlib::add(1 as i64, 2 as i64)",
    ] {
        assert!(
            main_rs.contains(expected),
            "missing {expected} in: {main_rs}"
        );
    }
    assert!(
        !main_rs.contains("use rustlib"),
        "no use-import for a dependency: {}",
        main_rs
    );

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    assert!(output.status.success(), "binary exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        &["3274436039", "42", "HI!", "4.5", "2986974", "3"],
        "runtime behavior diverged: {}",
        stdout
    );
}

#[test]
fn rust_module_stub_wins_and_unknown_fns_fail_loudly() {
    let scratch = Scratch::new("stub");
    write_rustlib_crate(&scratch.path().join("rustlib"));
    write_manifest(scratch.path(), "");
    // The stub is the declared signature source and wins over inference:
    // it declares half's return as f32, which inference would say is f64.
    // (The crate really returns f64, but `f64 as f32` compiles, so the
    // generated crate still builds — proving the stub types were used.)
    fs::write(
        scratch.path().join("rustlib.pyi"),
        concat!(
            "def crc8(data: \"&[u8]\", seed: \"u32\") -> \"u32\": ...\n",
            "def half(x: \"f64\") -> \"f32\": ...\n",
        ),
    )
    .unwrap();
    fs::write(
        scratch.path().join("app.py"),
        concat!(
            "import rustlib\n",
            "print(rustlib.crc8(b\"x\", 3))\n",
            "print(rustlib.half(9.0))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&scratch.path().join("app.py")).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    // The stub app has no `__main__` guard, so the module code lives in
    // src/app.rs behind `pub mod app;` in lib.rs.
    let app_rs = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(
        app_rs.contains("rustlib::crc8(b\"x\".to_vec().as_ref(), 3 as u32) as i64"),
        "stub crc8 types must be used: {}",
        app_rs
    );
    assert!(
        app_rs.contains("rustlib::half(9.0) as f64"),
        "stub half return (f32, converted to f64 for print) must win over \
         inferred f64, which emits no cast: {}",
        app_rs
    );
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    // Calling a function the stub does not bind is a loud conversion error.
    fs::write(
        scratch.path().join("app2.py"),
        "import rustlib\nprint(rustlib.add(1, 2))\n",
    )
    .unwrap();
    let pkg2 = rypip::discover(&scratch.path().join("app2.py")).expect("discover");
    let err = rypip::convert(
        &pkg2,
        &scratch.path().join("crate2"),
        &ConvertOptions::default(),
    )
    .expect_err("unbound fn must fail");
    assert!(
        err.to_string().contains("is not a bound function"),
        "error must name the unbound function: {}",
        err
    );

    // Version-only entries without a stub cannot resolve signatures.
    fs::write(scratch.path().join("app3.py"), "import rustlib\n").unwrap();
    let rustlib_dir = scratch.path().join("rustlib");
    let _ = fs::remove_dir_all(&rustlib_dir);
    let _ = fs::remove_file(scratch.path().join("rustlib.pyi"));
    fs::write(
        scratch.path().join("rython.toml"),
        "[rust-modules]\nrustlib = { version = \"0.1.0\" }\n",
    )
    .unwrap();
    let pkg3 = rypip::discover(&scratch.path().join("app3.py")).expect("discover");
    let err = rypip::convert(
        &pkg3,
        &scratch.path().join("crate3"),
        &ConvertOptions::default(),
    )
    .expect_err("version-only without stub must fail");
    assert!(
        err.to_string().contains(".pyi"),
        "error must demand a stub: {}",
        err
    );
}

#[test]
fn rust_module_import_unknown_module_falls_through() {
    // `import nosuchmod` with no manifest entry is not a rust-module error:
    // it falls through to the normal import path and fails as a resolution
    // error in the generated crate (same as any other unknown import).
    let scratch = Scratch::new("fallthrough");
    write_manifest(scratch.path(), "");
    fs::write(scratch.path().join("a.py"), "import nosuchmod\n").unwrap();
    let pkg = rypip::discover(&scratch.path().join("a.py")).expect("discover");
    // Conversion itself succeeds; the generated crate fails to resolve.
    let krate = rypip::convert(
        &pkg,
        &scratch.path().join("crate"),
        &ConvertOptions::default(),
    )
    .expect("unknown import is not a conversion error");
    let status = build_generated(&krate.root);
    assert!(!status.success(), "generated crate must fail to compile");
}
