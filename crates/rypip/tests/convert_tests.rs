//! Integration tests: discover and convert sample Python packages, verify
//! the generated crate layout, and compile a converted package for real.

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

/// Lay out a small Python project: pyproject.toml plus a package with an
/// __init__.py, a library module, and a __main__-style entry module.
fn write_sample_package(root: &Path) {
    fs::write(
        root.join("pyproject.toml"),
        "[project]\nname = \"greeter\"\nversion = \"1.2.3\"\n",
    )
    .unwrap();
    let pkg = root.join("greeter");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(pkg.join("__init__.py"), "from greeting import excited\n").unwrap();
    fs::write(
        pkg.join("greeting.py"),
        concat!(
            "def excited() -> str:\n",
            "    return f\"hello{'!' * 3}\"\n",
            "\n",
            "def shout_count(n: int) -> int:\n",
            "    total = 0\n",
            "    for i in [1, 2, 3]:\n",
            "        total += i\n",
            "    return total\n",
            "\n",
            "def log_it(n: int) -> int:\n",
            "    print(n)\n",
        ),
    )
    .unwrap();
    fs::write(
        pkg.join("optional.py"),
        "def with_default(n: int = 3) -> int:\n    return n\n",
    )
    .unwrap();
    fs::write(
        pkg.join("cli.py"),
        concat!(
            "def run():\n",
            "    print(\"greetings\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    run()\n",
        ),
    )
    .unwrap();
}

#[test]
fn discovers_package_metadata_and_modules() {
    let scratch = Scratch::new("discover");
    write_sample_package(scratch.path());

    let pkg = rypip::discover(scratch.path()).expect("discover");
    assert_eq!(pkg.name, "greeter");
    assert_eq!(pkg.version, "1.2.3");

    let mut names: Vec<String> = pkg.modules.iter().map(|m| m.path.join(".")).collect();
    names.sort();
    assert_eq!(names, vec!["", "cli", "greeting", "optional"]);
    assert!(pkg.entry_module().is_some(), "cli.py has a __main__ block");
}

#[test]
fn discovers_single_file_module() {
    let scratch = Scratch::new("single");
    let file = scratch.path().join("tool.py");
    fs::write(&file, "x = 1\n").unwrap();

    let pkg = rypip::discover(&file).expect("discover single file");
    assert_eq!(pkg.name, "tool");
    assert_eq!(pkg.modules.len(), 1);
}

#[test]
fn converts_package_into_crate_layout() {
    let scratch = Scratch::new("convert");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    assert_eq!(krate.name, "greeter");
    assert!(krate.has_binary, "cli.py should produce a binary");

    // The lossy conversion in optional.py (a dropped parameter default) must
    // be flagged as a conversion warning and baked into the generated code.
    assert!(
        krate.warnings.iter().any(|w| w.contains("with_default")),
        "expected a dropped-default warning, got: {:?}",
        krate.warnings
    );
    let optional_rs = fs::read_to_string(out.join("src/optional.rs")).unwrap();
    assert!(
        optional_rs.contains("deprecated"),
        "generated function should carry the warning note: {}",
        optional_rs
    );
    for file in ["Cargo.toml", "src/lib.rs", "src/greeting.rs", "src/cli.rs", "src/main.rs"] {
        assert!(out.join(file).is_file(), "missing {}", file);
    }

    let manifest = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(manifest.contains("name = \"greeter\""), "manifest: {}", manifest);
    assert!(manifest.contains("version = \"1.2.3\""), "manifest: {}", manifest);
    assert!(manifest.contains("stdpython"), "manifest: {}", manifest);

    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("pub mod greeting"), "lib.rs: {}", lib);

    let greeting = fs::read_to_string(out.join("src/greeting.rs")).unwrap();
    assert!(greeting.contains("fn excited"), "greeting.rs: {}", greeting);
    assert!(greeting.contains("-> String"), "greeting.rs: {}", greeting);
    assert!(
        greeting.contains("fn shout_count"),
        "greeting.rs: {}",
        greeting
    );

    let main_rs = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("fn main"), "main.rs: {}", main_rs);
}

#[test]
fn converted_crate_compiles_and_binary_runs() {
    let scratch = Scratch::new("compile");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = Command::new("cargo")
        .arg("build")
        .current_dir(&krate.root)
        .status()
        .expect("running cargo build");
    assert!(status.success(), "generated crate failed to compile");

    // The installed-binary path: run the built binary and check its output.
    let output = Command::new(krate.root.join("target/debug/greeter"))
        .output()
        .expect("running generated binary");
    assert!(output.status.success(), "binary exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("greetings"),
        "unexpected binary output: {}",
        stdout
    );
}

#[test]
fn pyo3_conversion_generates_bindings() {
    let scratch = Scratch::new("pyo3");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            pyo3: true,
            ..Default::default()
        },
    )
    .expect("convert with pyo3");

    let manifest = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(manifest.contains("pyo3"), "manifest: {}", manifest);
    assert!(manifest.contains("cdylib"), "manifest: {}", manifest);
    assert!(
        manifest.contains("python = [\"dep:pyo3\"]"),
        "manifest: {}",
        manifest
    );

    let bindings = fs::read_to_string(out.join("src/python_api.rs")).unwrap();
    assert!(bindings.contains("#[pymodule]"), "bindings: {}", bindings);
    assert!(
        bindings.contains("fn shout_count(n: i64) -> i64"),
        "annotated function should be bound with concrete types: {}",
        bindings
    );
    assert!(
        bindings.contains("crate::greeting::shout_count"),
        "wrapper should call through to the generated function: {}",
        bindings
    );
    assert!(
        bindings.contains("fn excited() -> String"),
        "zero-arg function with inferable return should be bound: {}",
        bindings
    );

    // log_it's `-> int` annotation is ignored by the function generator
    // because the body can fall through; the wrapper must agree, or the
    // generated crate won't compile.
    assert!(
        bindings.contains("fn log_it(n: i64)") && !bindings.contains("fn log_it(n: i64) -> i64"),
        "wrapper return type must match the generated function, not the annotation: {}",
        bindings
    );

    // Functions with defaults can't be bound by the simple wrapper; they
    // must be skipped (noted in the header), not emitted broken.
    assert!(
        !bindings.contains("fn with_default"),
        "defaulted function must not be bound: {}",
        bindings
    );
    assert!(
        bindings.contains("optional.with_default"),
        "skipped function should be listed: {}",
        bindings
    );

    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("mod python_api"),
        "lib.rs must include the bindings module: {}",
        lib
    );
}
