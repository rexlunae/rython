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
            "\n",
            "def shout(name: str) -> str:\n",
            "    return name.upper()\n",
            "\n",
            "def middle(s: str) -> str:\n",
            "    return s[1:-1] + s[0]\n",
            "\n",
            "def small(n: int) -> bool:\n",
            "    return n in {1, 2, 3}\n",
            "\n",
            "def classify(n: int) -> str:\n",
            "    label = \"fine\"\n",
            "    try:\n",
            "        if n < 0:\n",
            "            raise ValueError(\"negative\")\n",
            "        assert n != 13, \"unlucky\"\n",
            "    except ValueError:\n",
            "        label = \"negative\"\n",
            "    except AssertionError:\n",
            "        label = \"unlucky\"\n",
            "    return label\n",
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
            "from greeting import classify\n",
            "from greeting import shout\n",
            "from greeting import middle\n",
            "\n",
            "def run():\n",
            "    print(\"greetings\")\n",
            "    print(classify(-5))\n",
            "    print(classify(13))\n",
            "    print(classify(2))\n",
            "    print(shout(\"world\"))\n",
            "    print(middle(\"abcd\"))\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    run()\n",
        ),
    )
    .unwrap();
    // A sub-package whose only file is __init__.py, defining a function
    // whose name collides with cli.run across modules.
    let util = pkg.join("util");
    fs::create_dir_all(&util).unwrap();
    fs::write(
        util.join("__init__.py"),
        "def run() -> str:\n    return \"util\"\n",
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
    assert_eq!(names, vec!["", "cli", "greeting", "optional", "util"]);
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
    // log_it declares `-> int` but its body falls through: the annotation is
    // ignored, and that likely-source-bug must be flagged loudly too.
    assert!(
        krate
            .warnings
            .iter()
            .any(|w| w.contains("log_it") && w.contains("return annotation")),
        "expected an ignored-return-annotation warning, got: {:?}",
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
    // An init-only sub-package must still be declared, or its code is
    // silently dropped from the crate.
    assert!(lib.contains("pub mod util"), "lib.rs: {}", lib);
    assert!(out.join("src/util/mod.rs").is_file(), "missing src/util/mod.rs");

    let greeting = fs::read_to_string(out.join("src/greeting.rs")).unwrap();
    assert!(greeting.contains("fn excited"), "greeting.rs: {}", greeting);
    assert!(
        greeting.contains("-> Result<String, PyException>"),
        "functions return Result so exceptions propagate: {}",
        greeting
    );
    assert!(
        greeting.contains("fn shout_count"),
        "greeting.rs: {}",
        greeting
    );

    let main_rs = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("fn main"), "main.rs: {}", main_rs);
}

#[test]
fn deny_mode_promotes_warnings_to_errors() {
    let scratch = Scratch::new("deny");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            warnings: rypip::convert::WarningMode::Deny,
            ..Default::default()
        },
    )
    .expect_err("deny mode must fail on lossy conversions");
    let msg = format!("{}", err);
    assert!(msg.contains("with_default"), "error should list the warnings: {}", msg);
    assert!(msg.contains("log_it"), "error should list the warnings: {}", msg);
}

#[test]
fn allow_mode_suppresses_warnings() {
    let scratch = Scratch::new("allow");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            warnings: rypip::convert::WarningMode::Allow,
            ..Default::default()
        },
    )
    .expect("convert with allow");

    assert!(krate.warnings.is_empty(), "warnings: {:?}", krate.warnings);
    let optional_rs = fs::read_to_string(out.join("src/optional.rs")).unwrap();
    assert!(
        !optional_rs.contains("deprecated"),
        "allow mode must not bake warning notes into generated code: {}",
        optional_rs
    );
    let greeting_rs = fs::read_to_string(out.join("src/greeting.rs")).unwrap();
    assert!(!greeting_rs.contains("deprecated"), "greeting.rs: {}", greeting_rs);
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
    // classify() exercises try/except/assert end to end: a raised
    // ValueError, a failed assert (AssertionError), and the no-exception
    // path must each take the right handler at runtime.
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        &lines[1..6],
        &["negative", "unlucky", "fine", "WORLD", "bca"],
        "runtime behavior diverged: {}",
        stdout
    );
}

#[test]
fn pyo3_conversion_generates_bindings() {
    let scratch = Scratch::new("pyo3");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(
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
    // Wrapper identifiers are module-qualified so same-named functions in
    // different modules can't collide; the Python-visible name stays bare.
    assert!(
        bindings.contains("fn greeting_shout_count(n: i64) -> pyo3::PyResult<i64>"),
        "annotated function should be bound with concrete types: {}",
        bindings
    );
    assert!(
        bindings.contains("pyo3(name = \"shout_count\")"),
        "unique function keeps its bare Python name: {}",
        bindings
    );
    assert!(
        bindings.contains("crate::greeting::shout_count"),
        "wrapper should call through to the generated function: {}",
        bindings
    );
    assert!(
        bindings.contains("fn greeting_excited() -> pyo3::PyResult<String>"),
        "zero-arg function with inferable return should be bound: {}",
        bindings
    );

    // log_it's `-> int` annotation is ignored by the function generator
    // because the body can fall through; the wrapper must agree, or the
    // generated crate won't compile.
    assert!(
        bindings.contains("fn greeting_log_it(n: i64)")
            && !bindings.contains("fn greeting_log_it(n: i64) -> i64"),
        "wrapper return type must match the generated function, not the annotation: {}",
        bindings
    );

    // cli.run and util.run collide: both must be emitted (under qualified
    // names), neither may claim the bare Python name `run`, and the forced
    // rename must be flagged as a conversion warning.
    assert!(bindings.contains("fn cli_run"), "bindings: {}", bindings);
    assert!(bindings.contains("fn util_run"), "bindings: {}", bindings);
    assert!(
        !bindings.contains("pyo3(name = \"run\")"),
        "colliding names must not shadow each other in Python: {}",
        bindings
    );
    assert!(
        krate
            .warnings
            .iter()
            .any(|w| w.contains("`run`") && w.contains("qualified")),
        "expected a rename warning, got: {:?}",
        krate.warnings
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

#[test]
fn exceptions_propagate_across_functions_at_runtime() {
    // The full Python exception model: a callee's raise propagates to the
    // caller, is catchable there by type, a return inside try threads out
    // through the finally, and an uncaught exception prints the exception
    // and exits nonzero — exactly CPython's observable behavior.
    let scratch = Scratch::new("propagate");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def divide(a: int, b: int) -> int:\n",
            "    if b == 0:\n",
            "        raise ZeroDivisionError(\"division by zero\")\n",
            "    return a // b\n",
            "\n",
            "def safe_divide(a: int, b: int) -> int:\n",
            "    try:\n",
            "        return divide(a, b)\n",
            "    except ZeroDivisionError:\n",
            "        return 0\n",
            "\n",
            "def tracked_divide(a: int, b: int) -> int:\n",
            "    try:\n",
            "        return divide(a, b)\n",
            "    except ZeroDivisionError:\n",
            "        return -1\n",
            "    finally:\n",
            "        print(\"cleanup\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(safe_divide(10, 2))\n",
            "    print(safe_divide(5, 0))\n",
            "    print(tracked_divide(8, 2))\n",
            "    print(tracked_divide(8, 0))\n",
            "    print(divide(1, 0))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = Command::new("cargo")
        .arg("build")
        .current_dir(&krate.root)
        .status()
        .expect("running cargo build");
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // tracked_divide's finally must print "cleanup" before the returned
    // value is printed — on both the return-through-try path and the
    // handler-return path.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["5", "0", "cleanup", "4", "cleanup", "-1"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
    assert!(
        stderr.contains("ZeroDivisionError: division by zero"),
        "uncaught exception must be reported: {}",
        stderr
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "uncaught exception must exit nonzero"
    );
}

#[test]
fn dict_methods_match_python_at_runtime() {
    let scratch = Scratch::new("dicts");
    let file = scratch.path().join("dicts.py");
    fs::write(
        &file,
        concat!(
            "def stats() -> int:\n",
            "    d = {\"b\": 2, \"a\": 1}\n",
            "    d[\"c\"] = 3\n",
            "    total = 0\n",
            "    for k in d.keys():\n",
            "        total += d[k]\n",
            "    picked = d.get(\"a\", 0) + d.get(\"missing\", 100)\n",
            "    popped = d.pop(\"b\")\n",
            "    d.setdefault(\"z\", 50)\n",
            "    d.setdefault(\"a\", 999)\n",
            "    leftover = d.pop(\"gone\", 7)\n",
            "    return total + picked + popped + d[\"z\"] + d[\"a\"] + leftover\n",
            "\n",
            "def ordered() -> str:\n",
            "    d = {\"x\": 1, \"m\": 2, \"a\": 3}\n",
            "    d[\"q\"] = 4\n",
            "    return \"-\".join(d.keys())\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(stats())\n",
            "    print(ordered())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = Command::new("cargo")
        .arg("build")
        .current_dir(&krate.root)
        .status()
        .expect("running cargo build");
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/dicts"))
        .output()
        .expect("running generated binary");
    // Values verified against python3; "x-m-a-q" pins insertion order.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["167", "x-m-a-q"],
        "dict semantics diverged from CPython"
    );
}

#[test]
fn pyo3_crate_compiles() {
    let scratch = Scratch::new("pyo3-compile");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            pyo3: true,
            ..Default::default()
        },
    )
    .expect("convert with pyo3");

    // Text assertions can't catch duplicate definitions or wrapper/function
    // signature mismatches — type-check the bindings for real.
    let status = Command::new("cargo")
        .args(["check", "--features", "python"])
        .current_dir(&krate.root)
        .status()
        .expect("running cargo check");
    assert!(status.success(), "generated pyo3 crate failed to compile");
}

#[test]
fn nested_subscript_stores_mutate_in_place_at_runtime() {
    // grid[0][1] = 9 previously wrote into a clone of the row and silently
    // kept the old values; the store must land in the real container.
    let scratch = Scratch::new("nested");
    let file = scratch.path().join("grid.py");
    fs::write(
        &file,
        concat!(
            "def build() -> int:\n",
            "    grid = [[1, 2], [3, 4]]\n",
            "    grid[0][1] = 9\n",
            "    grid[1][0] += 10\n",
            "    table = {\"row\": [5, 6]}\n",
            "    table[\"row\"][1] = 7\n",
            "    return grid[0][1] + grid[1][0] + table[\"row\"][1]\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(build())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = Command::new("cargo")
        .arg("build")
        .current_dir(&krate.root)
        .status()
        .expect("running cargo build");
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/grid"))
        .output()
        .expect("running generated binary");
    // Python: 9 + 13 + 7 == 29
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "29",
        "nested stores must mutate the real containers"
    );
}
