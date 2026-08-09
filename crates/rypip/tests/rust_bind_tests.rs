//! Integration tests for `from rython import rust` bindings: declarations
//! lower to direct calls into a bound crate, dependencies land in the
//! generated Cargo.toml, and the converted program compiles and runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rypip::convert::ConvertOptions;

/// A scratch directory that's removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rypip-test-{}-{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("creating scratch dir");
        Scratch(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Build a generated crate. RUSTFLAGS is scrubbed: in the default warn mode
/// generated crates intentionally surface rustc warnings about the source
/// Python (unused variables, dead stores, ...), and these semantic tests
/// must not fail when the outer test run sets -D warnings.
fn build_generated(root: &Path) -> std::process::ExitStatus {
    Command::new("cargo")
        .arg("build")
        .env_remove("RUSTFLAGS")
        .current_dir(root)
        .status()
        .expect("running cargo build")
}

/// A tiny Rust crate the rust.bind fixture depends on. Kept dependency-free
/// so the converted crate builds offline against a path dependency.
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
            "//! Test library for rust.bind integration tests.\n\n",
            "/// CRC-8 over the data, seeded.\n",
            "pub fn crc8(data: &[u8], seed: u32) -> u32 {\n",
            "    let mut crc = seed;\n",
            "    for b in data {\n",
            "        crc = crc.wrapping_add(*b as u32).wrapping_mul(31);\n",
            "    }\n",
            "    crc\n",
            "}\n\n",
            "pub fn add(a: i64, b: i64) -> i64 { a + b }\n\n",
            "pub fn strlen(s: &str) -> usize { s.len() }\n\n",
            "pub fn shout(s: &str) -> String { format!(\"{}!\", s.to_uppercase()) }\n\n",
            "pub fn greet(_name: &str) -> &'static str { \"hello\" }\n\n",
            "pub fn first_byte(data: *const u8, len: usize) -> u8 {\n",
            "    if len == 0 { 0 } else { unsafe { *data } }\n",
            "}\n\n",
            "pub fn half(x: f64) -> f64 { x / 2.0 }\n\n",
            "pub fn is_odd(n: i32) -> bool { n % 2 != 0 }\n\n",
            "pub fn poke(_n: i64) {}\n\n",
            "// bindgen-style C-ABI declaration: unsafe to call, like real C libs.\n",
            "pub unsafe extern \"C\" fn c_add(a: i32, b: i32) -> i32 { a + b }\n",
        ),
    )
    .unwrap();
}

/// A single-file Python program binding every supported conversion, plus a
/// C-ABI binding through rust.c_bind.
fn write_bind_app(root: &Path) {
    fs::write(
        root.join("bindapp.py"),
        concat!(
            "from rython import rust\n",
            "\n",
            "crc = rust.bind(\n",
            "    \"rustlib\", \"crc8\",\n",
            "    args=[(\"data\", \"&[u8]\"), (\"seed\", \"u32\")],\n",
            "    returns=\"u32\",\n",
            "    path=\"../rustlib\",\n",
            ")\n",
            "add = rust.bind(\"rustlib\", \"add\", args=[(\"a\", \"i64\"), (\"b\", \"i64\")], returns=\"i64\", path=\"../rustlib\")\n",
            "strlen = rust.bind(\"rustlib\", \"strlen\", args=[(\"s\", \"&str\")], returns=\"usize\", path=\"../rustlib\")\n",
            "shout = rust.bind(\"rustlib\", \"shout\", args=[(\"s\", \"&str\")], returns=\"String\", path=\"../rustlib\")\n",
            "greet = rust.bind(\"rustlib\", \"greet\", args=[(\"name\", \"&str\")], returns=\"&str\", path=\"../rustlib\")\n",
            "first_byte = rust.bind(\"rustlib\", \"first_byte\", args=[(\"data\", \"*const u8\"), (\"len\", \"usize\")], returns=\"u8\", path=\"../rustlib\")\n",
            "half = rust.bind(\"rustlib\", \"half\", args=[(\"x\", \"f64\")], returns=\"f64\", path=\"../rustlib\")\n",
            "is_odd = rust.bind(\"rustlib\", \"is_odd\", args=[(\"n\", \"i32\")], returns=\"bool\", path=\"../rustlib\")\n",
            "poke = rust.bind(\"rustlib\", \"poke\", args=[(\"n\", \"i64\")], path=\"../rustlib\")\n",
            "c_add = rust.c_bind(\"rustlib\", \"c_add\", args=[(\"a\", \"i32\"), (\"b\", \"i32\")], returns=\"i32\", path=\"../rustlib\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    poke(1)\n",
            "    print(crc(b\"hello\", 7))\n",
            "    print(add(20, 22))\n",
            "    print(strlen(\"héllo\"))\n",
            "    print(shout(\"hi\"))\n",
            "    print(greet(\"nobody\"))\n",
            "    data = b\"abc\"\n",
            "    print(first_byte(data, len(data)))\n",
            "    print(half(9.0))\n",
            "    print(is_odd(7))\n",
            "    print(c_add(40, 2))\n",
        ),
    )
    .unwrap();
}

#[test]
fn rust_bind_generates_direct_calls_and_dependencies() {
    let scratch = Scratch::new("rustbind");
    write_rustlib_crate(&scratch.path().join("rustlib"));
    write_bind_app(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("bindapp.py")).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    // The declared crate becomes a dependency with its spec copied verbatim.
    let manifest = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("rustlib = { path = \"../rustlib\" }"),
        "manifest must declare the bound crate: {}",
        manifest
    );

    // Bindings lower to direct calls with type-directed conversions; the
    // declaration assignments and their names vanish from the runtime code.
    let main_rs = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(
        main_rs.contains("rustlib::crc8(b\"hello\".as_ref(), 7 as u32) as i64"),
        "&[u8] + u32 args and u32 -> i64 return: {}",
        main_rs
    );
    assert!(
        main_rs.contains("rustlib::shout(\"hi\".as_ref())"),
        "&str arg from a literal: {}",
        main_rs
    );
    assert!(
        main_rs.contains("rustlib::first_byte(data.as_ptr(), len(&(data)) as usize)"),
        "*const u8 + usize args: {}",
        main_rs
    );
    assert!(
        main_rs.contains("unsafe { rustlib::c_add(40 as i32, 2 as i32) } as i64"),
        "c_bind wraps the call in unsafe: {}",
        main_rs
    );
    assert!(
        main_rs.contains("rustlib::poke(1 as i64)"),
        "void-returning call stays a bare statement: {}",
        main_rs
    );
    assert!(
        !main_rs.contains("let mut crc") && !main_rs.contains("let mut add"),
        "binding names must not be hoisted into runtime variables: {}",
        main_rs
    );

    // The whole thing compiles and the binary's output matches the math.
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/bindapp"))
        .output()
        .expect("running generated binary");
    assert!(output.status.success(), "binary exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines,
        &[
            "3274436039", // crc8(b"hello", 7)
            "42",         // add(20, 22)
            "6",          // strlen("héllo") — raw Rust str::len: UTF-8 bytes
            "HI!",        // shout("hi")
            "hello",      // greet("nobody") — &'static str, to_string()'d
            "97",         // first_byte(data, 3)
            "4.5",        // half(9.0)
            "True",       // is_odd(7) — print renders bool Python-style
            "42",         // c_add(40, 2) through the C ABI
        ],
        "runtime behavior diverged: {}",
        stdout
    );
}

#[test]
fn rust_bind_rejects_malformed_declarations_loudly() {
    let scratch = Scratch::new("rustbind-err");
    write_rustlib_crate(&scratch.path().join("rustlib"));

    // Missing path/version: the dependency cannot be declared.
    let file = scratch.path().join("a.py");
    fs::write(
        &file,
        "from rython import rust\nx = rust.bind(\"rustlib\", \"add\", args=[(\"a\", \"i64\")], returns=\"i64\")\n",
    )
    .unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(&pkg, &scratch.path().join("crate-a"), &ConvertOptions::default())
        .expect_err("missing path/version must fail");
    assert!(
        err.to_string().contains("path=") && err.to_string().contains("version="),
        "error must demand a dependency spec: {}",
        err
    );

    // Unsupported parameter type.
    let file = scratch.path().join("b.py");
    fs::write(
        &file,
        "from rython import rust\nx = rust.bind(\"rustlib\", \"f\", args=[(\"a\", \"HashMap\")], path=\"../rustlib\")\n",
    )
    .unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(&pkg, &scratch.path().join("crate-b"), &ConvertOptions::default())
        .expect_err("unsupported type must fail");
    assert!(
        err.to_string().contains("unsupported parameter type"),
        "error must name the type: {}",
        err
    );

    // A binding declared inside a function is not module-level.
    let file = scratch.path().join("c.py");
    fs::write(
        &file,
        "from rython import rust\ndef f():\n    x = rust.bind(\"rustlib\", \"add\", args=[(\"a\", \"i64\")], path=\"../rustlib\")\n",
    )
    .unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(&pkg, &scratch.path().join("crate-c"), &ConvertOptions::default())
        .expect_err("function-scope binding must fail");
    assert!(
        err.to_string().contains("module level"),
        "error must demand module level: {}",
        err
    );

    // Arity mismatch at a call site.
    let file = scratch.path().join("d.py");
    fs::write(
        &file,
        concat!(
            "from rython import rust\n",
            "add = rust.bind(\"rustlib\", \"add\", args=[(\"a\", \"i64\"), (\"b\", \"i64\")], returns=\"i64\", path=\"../rustlib\")\n",
            "def f():\n",
            "    return add(1)\n",
        ),
    )
    .unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(&pkg, &scratch.path().join("crate-d"), &ConvertOptions::default())
        .expect_err("arity mismatch must fail");
    assert!(
        err.to_string().contains("takes 2 argument(s)"),
        "error must state the arity: {}",
        err
    );
}
