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
fn warning_mode_sets_generated_lint_posture() {
    // The rustc lints that surface source-Python weaknesses (unused
    // variables, dead stores, unreachable code, ...) follow the warning
    // mode: warn leaves rustc's defaults so they show at build time, deny
    // makes the generated crate fail on them, allow suppresses them.
    use rypip::convert::WarningMode;
    let scratch = Scratch::new("lints");
    let file = scratch.path().join("clean.py");
    fs::write(&file, "def f(n: int) -> int:\n    return n + 1\n").unwrap();

    for (mode, tag) in [
        (WarningMode::Warn, "warn"),
        (WarningMode::Deny, "deny"),
        (WarningMode::Allow, "allow"),
    ] {
        let out = scratch.path().join(format!("crate-{}", tag));
        let pkg = rypip::discover(&file).expect("discover");
        rypip::convert(
            &pkg,
            &out,
            &ConvertOptions {
                warnings: mode,
                ..Default::default()
            },
        )
        .expect("convert");
        // Inner lint attributes live at the crate root and apply
        // crate-wide; module files carry none.
        let root = fs::read_to_string(out.join("src/lib.rs")).unwrap();
        match mode {
            WarningMode::Warn => assert!(
                !root.contains("#![allow(") && !root.contains("#![deny("),
                "warn mode must leave rustc's default lint posture: {}",
                root
            ),
            WarningMode::Deny => assert!(
                root.contains("#![deny(") && root.contains("unreachable_code"),
                "deny mode must deny the surfaced lints: {}",
                root
            ),
            WarningMode::Allow => assert!(
                root.contains("#![allow(") && root.contains("unreachable_code"),
                "allow mode must suppress the surfaced lints: {}",
                root
            ),
        }
    }
}

#[test]
fn converted_crate_compiles_and_binary_runs() {
    let scratch = Scratch::new("compile");
    write_sample_package(scratch.path());
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
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

    let status = build_generated(&krate.root);
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
    let status = build_generated(&krate.root);
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
fn optional_from_dict_get_matches_python_at_runtime() {
    // A None-seeded variable reassigned from dict.get must NOT double-wrap:
    // an absent key would become Some(None) and the `is None` branch below
    // would silently never fire (PR #38 review finding).
    let scratch = Scratch::new("optget");
    let file = scratch.path().join("optget.py");
    fs::write(
        &file,
        concat!(
            "def probe(keys: list[int]) -> int:\n",
            "    d = {1: 10, 2: 20}\n",
            "    result = None\n",
            "    for k in keys:\n",
            "        result = d.get(k)\n",
            "    if result is None:\n",
            "        return -1\n",
            "    return result + 100\n",
            "\n",
            "def pick(n: int) -> int:\n",
            "    d = {1: 10, 2: 20}\n",
            "    choice = None\n",
            "    choice = d.get(n) if n > 0 else None\n",
            "    if choice is None:\n",
            "        return -1\n",
            "    return choice + 200\n",
            "\n",
            "def sign_label(n: int) -> int:\n",
            "    tag = None\n",
            "    tag = n if n > 0 else None\n",
            "    if tag is None:\n",
            "        return 0\n",
            "    return tag + 300\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(probe([1]))\n",
            "    print(probe([9]))\n",
            "    print(probe([2, 9]))\n",
            "    print(probe([9, 2]))\n",
            "    print(pick(1))\n",
            "    print(pick(-1))\n",
            "    print(pick(9))\n",
            "    print(sign_label(5))\n",
            "    print(sign_label(-2))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/optget"))
        .output()
        .expect("running generated binary");
    // Values verified against python3: hit, miss, hit-then-miss,
    // miss-then-hit, then the conditional-expression cases (Option arms and
    // a plain/None arm mix).
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["110", "-1", "-1", "120", "210", "-1", "-1", "305", "0"],
        "optional dict.get semantics diverged from CPython"
    );
}

#[test]
fn string_methods_match_python_at_runtime() {
    // Code-point len/find, count, maxsplit/rsplit, partition tuples,
    // strip(chars), title, zfill, ljust/rjust, and the empty-separator
    // ValueError — all through generated code.
    let scratch = Scratch::new("strings");
    let file = scratch.path().join("strings.py");
    fs::write(
        &file,
        concat!(
            "def run() -> int:\n",
            "    s = \"café latte café\"\n",
            "    print(len(s))\n",
            "    print(s.count(\"café\"))\n",
            "    print(s.find(\"é\"))\n",
            "    parts = \"x-y-z\".split(\"-\", 1)\n",
            "    print(f\"{parts[0]} {parts[1]}\")\n",
            "    tail = \"a-b-c-d\".rsplit(\"-\", 2)\n",
            "    print(f\"{tail[0]} {tail[1]} {tail[2]}\")\n",
            "    trio = \"key=val=ue\".partition(\"=\")\n",
            "    print(f\"{trio[0]} {trio[2]}\")\n",
            "    print(\"xxhixx\".strip(\"x\"))\n",
            "    print(\"hello wOrld\".title())\n",
            "    print(\"-42\".zfill(6))\n",
            "    print(\"hi\".ljust(5, \".\"))\n",
            "    print(\"hi\".rjust(5, \"*\"))\n",
            "    try:\n",
            "        \"ab\".split(\"\")\n",
            "    except ValueError:\n",
            "        print(\"caught empty separator\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    run()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/strings"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "15",
            "2",
            "3",
            "x y-z",
            "a-b c d",
            "key val=ue",
            "hi",
            "Hello World",
            "-00042",
            "hi...",
            "***hi",
            "caught empty separator"
        ],
        "string semantics diverged from CPython"
    );
}

#[test]
fn range_variants_match_python_at_runtime() {
    // Multi-argument range (including negative steps) and the catchable
    // zero-step ValueError, through generated code over the LAZY range.
    let scratch = Scratch::new("ranges");
    let file = scratch.path().join("ranges.py");
    fs::write(
        &file,
        concat!(
            "def run() -> int:\n",
            "    total = 0\n",
            "    for i in range(5):\n",
            "        total += i\n",
            "    print(total)\n",
            "    for i in range(2, 8, 2):\n",
            "        print(i)\n",
            "    for i in range(3, 0, -1):\n",
            "        print(i)\n",
            "    try:\n",
            "        for i in range(0, 5, 0):\n",
            "            pass\n",
            "    except ValueError:\n",
            "        print(\"zero step caught\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    run()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/ranges"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["10", "2", "4", "6", "3", "2", "1", "zero step caught"],
        "range semantics diverged from CPython"
    );
}

#[test]
fn module_constants_match_python_at_runtime() {
    // Module-level constants are visible to functions (Python globals);
    // a value-returning `main` runs through the wrapper entry point.
    let scratch = Scratch::new("globals");
    let file = scratch.path().join("globals.py");
    fs::write(
        &file,
        concat!(
            "PI = 3.14159\n",
            "GREETING = \"hello\"\n",
            "DEBUG = True\n",
            "LIMIT = 10\n",
            "OFFSET = -3\n",
            "\n",
            "def area(r: float) -> float:\n",
            "    return PI * r * r\n",
            "\n",
            "def describe() -> str:\n",
            "    return f\"{GREETING} {LIMIT}\"\n",
            "\n",
            "def main() -> int:\n",
            "    print(f\"{area(2.0):.4f}\")\n",
            "    print(describe())\n",
            "    if DEBUG:\n",
            "        print(LIMIT + 5)\n",
            "    print(LIMIT + OFFSET)\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/globals"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["12.5664", "hello 10", "15", "7"],
        "module-global semantics diverged from CPython"
    );
}

#[test]
fn str_format_matches_python_at_runtime() {
    // Auto-numbering, explicit positions with reuse, keywords, {{ escaping,
    // and format specs — through str.format and f-strings alike.
    let scratch = Scratch::new("format");
    let file = scratch.path().join("format.py");
    fs::write(
        &file,
        concat!(
            "def run() -> int:\n",
            "    print(\"{} and {}\".format(1, \"x\"))\n",
            "    print(\"{1}-{0}\".format(\"a\", \"b\"))\n",
            "    print(\"{:.2f}\".format(3.14159))\n",
            "    print(\"{:f}\".format(1.5))\n",
            "    print(\"{:>6}|\".format(\"hi\"))\n",
            "    print(\"{:*^7}|\".format(\"mid\"))\n",
            "    print(\"{:05d}\".format(42))\n",
            "    print(\"{{literal}} {}\".format(7))\n",
            "    print(\"{name}={val}\".format(name=\"x\", val=3))\n",
            "    print(\"{:#x} {:b}\".format(255, 5))\n",
            "    print(\"{0} {0}\".format(\"dup\"))\n",
            "    n = 42\n",
            "    print(f\"{3.14159:.2f} {n:05d} {'hi':>6}|\")\n",
            "    m = -255\n",
            "    print(\"{:x} {:#x} {:#06x}\".format(m, m, m))\n",
            "    print(\"{:.2f} {:f}\".format(5, 2))\n",
            "    print(f\"{m:#x} {5:.1f}\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    run()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/format"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "1 and x",
            "b-a",
            "3.14",
            "1.500000",
            "    hi|",
            "**mid**|",
            "00042",
            "{literal} 7",
            "x=3",
            "0xff 101",
            "dup dup",
            "3.14 00042     hi|",
            "-ff -0xff -0x0ff",
            "5.00 2.000000",
            "-0xff 5.0"
        ],
        "format semantics diverged from CPython"
    );
}

#[test]
fn classes_match_python_at_runtime() {
    // Struct-based classes: field inference, defaults, keyword method
    // calls, transitive &mut receivers, exceptions raised from methods and
    // caught by callers, and composition with mutation through field
    // chains.
    let scratch = Scratch::new("classes");
    let file = scratch.path().join("classes.py");
    fs::write(
        &file,
        concat!(
            "class Counter:\n",
            "    def __init__(self, label: str, start: int = 0):\n",
            "        self.label = label\n",
            "        self.count = start\n",
            "\n",
            "    def bump(self, amount: int) -> int:\n",
            "        self.count += amount\n",
            "        return self.count\n",
            "\n",
            "    def reset(self):\n",
            "        self.count = 0\n",
            "\n",
            "    def double_bump(self, amount: int) -> int:\n",
            "        self.bump(amount)\n",
            "        self.bump(amount)\n",
            "        return self.count\n",
            "\n",
            "    def describe(self) -> str:\n",
            "        return f\"{self.label}={self.count}\"\n",
            "\n",
            "    def label_of(self) -> str:\n",
            "        return self.label\n",
            "\n",
            "class Guard:\n",
            "    def __init__(self, limit: int):\n",
            "        self.limit = limit\n",
            "\n",
            "    def check(self, n: int) -> int:\n",
            "        if n > self.limit:\n",
            "            raise ValueError(\"over limit\")\n",
            "        return n\n",
            "\n",
            "class Point:\n",
            "    def __init__(self, x: int, y: int):\n",
            "        self.x = x\n",
            "        self.y = y\n",
            "\n",
            "    def dist2(self) -> int:\n",
            "        return self.x * self.x + self.y * self.y\n",
            "\n",
            "    def shift(self, dx: int):\n",
            "        self.x += dx\n",
            "\n",
            "class Segment:\n",
            "    def __init__(self, a: Point, b: Point):\n",
            "        self.a = a\n",
            "        self.b = b\n",
            "\n",
            "    def total(self) -> int:\n",
            "        return self.a.dist2() + self.b.dist2()\n",
            "\n",
            "    def nudge(self):\n",
            "        self.a.shift(1)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    c = Counter(\"hits\", 10)\n",
            "    print(c.bump(5))\n",
            "    print(c.bump(amount=2))\n",
            "    print(c.double_bump(3))\n",
            "    c.reset()\n",
            "    print(c.describe())\n",
            "    print(c.label_of())\n",
            "    d = Counter(\"fresh\")\n",
            "    print(d.bump(1))\n",
            "    g = Guard(10)\n",
            "    try:\n",
            "        g.check(11)\n",
            "    except ValueError:\n",
            "        print(\"caught\")\n",
            "    print(g.check(7))\n",
            "    s = Segment(Point(1, 2), Point(3, 4))\n",
            "    print(s.total())\n",
            "    s.nudge()\n",
            "    print(s.total())\n",
            "    print(s.a.x)\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/classes"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["15", "17", "23", "hits=0", "hits", "1", "caught", "7", "30", "33", "2"],
        "class semantics diverged from CPython"
    );
}

#[test]
fn keyword_arguments_and_defaults_match_python_at_runtime() {
    let scratch = Scratch::new("kwargs");
    let file = scratch.path().join("kw.py");
    fs::write(
        &file,
        concat!(
            "def greet(greeting: str, name: str = \"world\", excited: bool = False) -> str:\n",
            "    tail = \"!\" if excited else \".\"\n",
            "    return greeting + \", \" + name + tail\n",
            "\n",
            "def volume(w: int, h: int, d: int) -> int:\n",
            "    return w * h * d\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(greet(\"hi\"))\n",
            "    print(greet(\"hello\", name=\"rython\"))\n",
            "    print(greet(\"hey\", excited=True))\n",
            "    print(greet(name=\"bob\", greeting=\"yo\", excited=True))\n",
            "    print(volume(d=2, w=3, h=4))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/kw"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["hi, world.", "hello, rython.", "hey, world!", "yo, bob!", "24"],
        "keyword/default call semantics diverged from CPython"
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
        .env_remove("RUSTFLAGS")
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
    let status = build_generated(&krate.root);
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

/// cargo-check a generated crate (the no_std profile emits a library, so
/// there is no binary to run). RUSTFLAGS is scrubbed for the same reason as
/// build_generated.
fn check_generated(root: &Path) -> std::process::ExitStatus {
    Command::new("cargo")
        .arg("check")
        .env_remove("RUSTFLAGS")
        .current_dir(root)
        .status()
        .expect("running cargo check")
}

#[test]
fn no_std_profile_generates_a_nostd_crate_that_compiles() {
    let scratch = Scratch::new("nostd");
    let file = scratch.path().join("gauges.py");
    fs::write(
        &file,
        concat!(
            "class Accumulator:\n",
            "    def __init__(self, label: str):\n",
            "        self.label = label\n",
            "        self.total = 0\n",
            "\n",
            "    def add(self, n: int) -> int:\n",
            "        self.total += n\n",
            "        return self.total\n",
            "\n",
            "def describe(n: int) -> str:\n",
            "    tags = [\"low\", \"high\"]\n",
            "    tag = tags[0] if n < 10 else tags[1]\n",
            "    return f\"{n}:{tag}\"\n",
            "\n",
            "def total_priced(prices: dict[int, int]) -> int:\n",
            "    total = 0\n",
            "    for key in [1, 2, 3]:\n",
            "        total += prices.get(key, 0)\n",
            "    return total\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            no_std: true,
            ..Default::default()
        },
    )
    .expect("no_std convert of an OS-free module");
    assert!(!krate.has_binary, "no_std output is a library");

    let root = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(root.contains("#![no_std]"), "lib.rs: {}", root);
    let manifest = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("default-features = false") && manifest.contains("\"alloc\""),
        "Cargo.toml must pin stdpython to the alloc tier: {}",
        manifest
    );

    // The proof: the generated crate compiles as a genuine #![no_std]
    // library, where any std path would be an unresolved-name error.
    let status = check_generated(&out);
    assert!(status.success(), "generated no_std crate failed to compile");
}

#[test]
fn no_std_profile_rejects_std_constructs_loudly() {
    let scratch = Scratch::new("nostd-loud");
    let cases: &[(&str, &str, &str)] = &[
        ("uses_print.py", "print(\"hi\")\n", "no_std profile"),
        ("uses_os.py", "import os\n", "std tier"),
        (
            "uses_datetime.py",
            "from datetime import datetime\n",
            "std tier",
        ),
        ("uses_math.py", "import math\n", "std tier"),
        (
            "has_entry.py",
            "def main() -> int:\n    return 0\n\nif __name__ == \"__main__\":\n    main()\n",
            "no_std profile",
        ),
    ];
    for (name, src, needle) in cases {
        let file = scratch.path().join(name);
        fs::write(&file, src).unwrap();
        let out = scratch.path().join(format!("crate-{}", name.replace('.', "-")));
        let pkg = rypip::discover(&file).expect("discover");
        let err = rypip::convert(
            &pkg,
            &out,
            &ConvertOptions {
                no_std: true,
                ..Default::default()
            },
        )
        .expect_err("std-tier construct must fail the conversion");
        let msg = format!("{:#}", err);
        assert!(msg.contains(needle), "{}: {}", name, msg);
    }

    // PyO3 bindings need the Python runtime — contradictory with no_std.
    let file = scratch.path().join("plain.py");
    fs::write(&file, "def f(n: int) -> int:\n    return n\n").unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &scratch.path().join("crate-pyo3"),
        &ConvertOptions {
            no_std: true,
            pyo3: true,
            ..Default::default()
        },
    )
    .expect_err("pyo3 + no_std must fail");
    assert!(format!("{:#}", err).contains("PyO3"), "err: {:#}", err);
}

#[test]
fn builtins_match_python_at_runtime() {
    // min/max (n-ary, default=, key=), sorted (reverse=, key=, stability),
    // reversed, enumerate(start=), 2/3-arg pow, and repr (including
    // Python's float scientific-notation thresholds and str quoting),
    // through generated code.
    let scratch = Scratch::new("builtins");
    let file = scratch.path().join("builtins_demo.py");
    fs::write(
        &file,
        concat!(
            "def main() -> int:\n",
            "    nums = [5, 1, 9, 3]\n",
            "    print(f\"min={min(nums)} max={max(nums)}\")\n",
            "    print(f\"pair={min(4, 2)} triple={max(4, 2, 6)}\")\n",
            "    print(f\"mindef={min([], default=7)}\")\n",
            "    words = [\"pear\", \"fig\", \"apple\"]\n",
            "    print(f\"minkey={min(words, key=lambda w: len(w))}\")\n",
            "    print(f\"sorted={repr(sorted(nums))}\")\n",
            "    print(f\"sortedrev={repr(sorted(words, reverse=True))}\")\n",
            "    print(f\"sortedkey={repr(sorted(words, key=lambda w: len(w)))}\")\n",
            "    for i, v in enumerate(reversed(nums), start=1):\n",
            "        print(f\"rev{i}={v}\")\n",
            "    print(f\"powm={pow(3, -1, 7)} pow2={pow(2, 10)}\")\n",
            "    print(f\"fbig={repr(1e16)} fsum={repr(0.1 + 0.2)}\")\n",
            "    s = \"it's\"\n",
            "    print(f\"sq={repr(s)}\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/builtins_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "min=1 max=9",
            "pair=2 triple=6",
            "mindef=7",
            "minkey=fig",
            "sorted=[1, 3, 5, 9]",
            "sortedrev=['pear', 'fig', 'apple']",
            "sortedkey=['fig', 'pear', 'apple']",
            "rev1=3",
            "rev2=9",
            "rev3=1",
            "rev4=5",
            "powm=5 pow2=1024",
            "fbig=1e+16 fsum=0.30000000000000004",
            "sq=\"it's\"",
        ],
        "builtin semantics diverged from CPython"
    );
}

#[test]
fn datetime_and_time_match_python_at_runtime() {
    // date/datetime/timedelta constructors with keywords, arithmetic
    // operators, strptime (including its catchable ValueError), and the
    // time module, through generated code.
    let scratch = Scratch::new("datetimes");
    let file = scratch.path().join("dt_demo.py");
    fs::write(
        &file,
        concat!(
            "from datetime import date, datetime, timedelta\n",
            "import time\n",
            "\n",
            "def main() -> int:\n",
            "    d1 = date(2024, 3, 1)\n",
            "    d2 = date(2024, 2, 27)\n",
            "    gap = d1 - d2\n",
            "    print(f\"gap={gap} days={gap.days}\")\n",
            "    print(f\"shift={d1 + timedelta(days=3)} back={d1 - timedelta(weeks=1)}\")\n",
            "    dt = datetime.strptime(\"2024-01-05 08:30:15\", \"%Y-%m-%d %H:%M:%S\")\n",
            "    print(f\"dt={dt}\")\n",
            "    dt2 = dt + timedelta(hours=25, minutes=90)\n",
            "    print(f\"dt2={dt2}\")\n",
            "    diff = dt2 - dt\n",
            "    print(f\"diff={diff} d={diff.days} s={diff.seconds}\")\n",
            "    try:\n",
            "        print(datetime.strptime(\"nope\", \"%Y-%m-%d\"))\n",
            "    except ValueError:\n",
            "        print(\"bad format caught\")\n",
            "    t0 = time.monotonic()\n",
            "    time.sleep(0.01)\n",
            "    elapsed = time.monotonic() - t0\n",
            "    print(\"monotonic_ok\" if elapsed >= 0.009 else \"monotonic_bad\")\n",
            "    print(\"wall_ok\" if time.time() > 1577836800.0 else \"wall_bad\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/dt_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "gap=3 days, 0:00:00 days=3",
            "shift=2024-03-04 back=2024-02-23",
            "dt=2024-01-05 08:30:15",
            "dt2=2024-01-06 11:00:15",
            "diff=1 day, 2:30:00 d=1 s=9000",
            "bad format caught",
            "monotonic_ok",
            "wall_ok",
        ],
        "datetime/time semantics diverged from CPython"
    );
}

#[test]
fn itertools_gaps_match_python_at_runtime() {
    // accumulate (default, func, initial=), product (pairs and repeat=),
    // combinations_with_replacement, pairwise, zip_longest with
    // fillvalue=, consecutive groupby, and starmap, through generated
    // code.
    let scratch = Scratch::new("itertools");
    let file = scratch.path().join("it_demo.py");
    fs::write(
        &file,
        concat!(
            "from itertools import accumulate, product, combinations_with_replacement, pairwise, zip_longest, groupby, starmap\n",
            "\n",
            "def main() -> int:\n",
            "    for v in accumulate([1, 2, 3, 4]):\n",
            "        print(f\"acc={v}\")\n",
            "    for v in accumulate([1, 2, 3], initial=100):\n",
            "        print(f\"acci={v}\")\n",
            "    for v in accumulate([1, 2, 3, 4], lambda a, b: a * b):\n",
            "        print(f\"accf={v}\")\n",
            "    for a, b in product([1, 2], [10, 20]):\n",
            "        print(f\"prod={a},{b}\")\n",
            "    for a, b in product([0, 1], repeat=2):\n",
            "        print(f\"rep={a},{b}\")\n",
            "    for c in combinations_with_replacement([1, 2, 3], 2):\n",
            "        print(f\"cwr={c[0]},{c[1]}\")\n",
            "    for a, b in pairwise([1, 2, 3, 4]):\n",
            "        print(f\"pw={a},{b}\")\n",
            "    for a, b in zip_longest([1], [10, 20, 30], fillvalue=0):\n",
            "        print(f\"zl={a},{b}\")\n",
            "    for k, g in groupby([1, 1, 2, 2, 2, 1]):\n",
            "        total = 0\n",
            "        for _x in g:\n",
            "            total += 1\n",
            "        print(f\"g={k}:{total}\")\n",
            "    for v in starmap(lambda a, b: a * b, [(2, 3), (4, 5)]):\n",
            "        print(f\"sm={v}\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/it_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "acc=1", "acc=3", "acc=6", "acc=10",
            "acci=100", "acci=101", "acci=103", "acci=106",
            "accf=1", "accf=2", "accf=6", "accf=24",
            "prod=1,10", "prod=1,20", "prod=2,10", "prod=2,20",
            "rep=0,0", "rep=0,1", "rep=1,0", "rep=1,1",
            "cwr=1,1", "cwr=1,2", "cwr=1,3", "cwr=2,2", "cwr=2,3", "cwr=3,3",
            "pw=1,2", "pw=2,3", "pw=3,4",
            "zl=1,10", "zl=0,20", "zl=0,30",
            "g=1:2", "g=2:3", "g=1:1",
            "sm=6", "sm=20",
        ],
        "itertools semantics diverged from CPython"
    );

    // Deny mode regression: calls rewritten to variant functions must not
    // orphan the base imports (`use ...::accumulate;`), or
    // #![deny(unused_imports)] fails this perfectly clean source. A
    // LIBRARY module, because entry modules have a separate pre-existing
    // deny-mode issue (the lib-side copy of fn main is dead code).
    let lib_file = scratch.path().join("it_lib.py");
    fs::write(
        &lib_file,
        concat!(
            "from itertools import accumulate, product\n",
            "\n",
            "def running(xs: list[int]) -> list[int]:\n",
            "    return accumulate(xs, initial=0)\n",
            "\n",
            "def grid(xs: list[int]) -> int:\n",
            "    total = 0\n",
            "    for a, b in product(xs, repeat=2):\n",
            "        total += a * b\n",
            "    return total\n",
        ),
    )
    .unwrap();
    let deny_out = scratch.path().join("crate-deny");
    let lib_pkg = rypip::discover(&lib_file).expect("discover");
    let krate = rypip::convert(
        &lib_pkg,
        &deny_out,
        &ConvertOptions {
            warnings: rypip::convert::WarningMode::Deny,
            ..Default::default()
        },
    )
    .expect("deny-mode convert of a clean module");
    let status = build_generated(&krate.root);
    assert!(
        status.success(),
        "deny-mode generated crate failed to compile (orphaned imports?)"
    );
}

#[test]
fn pure_modules_match_python_at_runtime() {
    // heapq (exact CPython list layouts, module-attribute and from-import
    // spellings), functools.reduce (both arities), copy.deepcopy
    // independence, and textwrap.dedent, through generated code.
    let scratch = Scratch::new("puremods");
    let file = scratch.path().join("pure_demo.py");
    fs::write(
        &file,
        concat!(
            "from functools import reduce\n",
            "from heapq import heappush, heappop, heapify, nlargest\n",
            "from copy import deepcopy\n",
            "from textwrap import dedent\n",
            "import heapq\n",
            "\n",
            "def main() -> int:\n",
            "    h = [5, 1, 9, 3, 7, 2]\n",
            "    heapify(h)\n",
            "    print(f\"heap={repr(h)}\")\n",
            "    heappush(h, 0)\n",
            "    print(f\"pushed={repr(h)}\")\n",
            "    print(f\"pop={heappop(h)}\")\n",
            "    print(f\"pushpop={heapq.heappushpop(h, 4)}\")\n",
            "    print(f\"big={repr(nlargest(3, [5, 1, 9, 3, 7]))}\")\n",
            "    print(f\"prod={reduce(lambda a, b: a * b, [1, 2, 3, 4])}\")\n",
            "    print(f\"sum={reduce(lambda a, b: a + b, [1, 2], 100)}\")\n",
            "    nested = [[1, 2], [3]]\n",
            "    cloned = deepcopy(nested)\n",
            "    cloned[0].append(9)\n",
            "    print(f\"orig={repr(nested)} clone={repr(cloned)}\")\n",
            "    rows = [[3, 1], [9]]\n",
            "    heapify(rows[0])\n",
            "    heappush(rows[1], 4)\n",
            "    print(f\"rows={repr(rows)}\")\n",
            "    text = \"    a\\n      b\\n    c\"\n",
            "    print(dedent(text))\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/pure_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "heap=[1, 3, 2, 5, 7, 9]",
            "pushed=[0, 3, 1, 5, 7, 9, 2]",
            "pop=0",
            "pushpop=1",
            "big=[9, 7, 5]",
            "prod=24",
            "sum=103",
            "orig=[[1, 2], [3]] clone=[[1, 2, 9], [3]]",
            "rows=[[1, 3], [4, 9]]",
            "a",
            "  b",
            "c",
        ],
        "pure-module semantics diverged from CPython"
    );
}

#[test]
fn re_module_matches_python_at_runtime() {
    // search/match/fullmatch through the Option-based Match model
    // (`if m:` + m.group()), findall, sub with backreference translation,
    // and split, through generated code.
    let scratch = Scratch::new("regex");
    let file = scratch.path().join("re_demo.py");
    fs::write(
        &file,
        concat!(
            "import re\n",
            "\n",
            "def main() -> int:\n",
            "    m = re.search(r\"(\\d+)-(\\d+)\", \"order 12-34 shipped\")\n",
            "    if m:\n",
            "        print(f\"whole={m.group(0)} a={m.group(1)} b={m.group(2)}\")\n",
            "        print(f\"span={m.start()},{m.end()}\")\n",
            "    ok = re.match(r\"\\d+\", \"12ab\")\n",
            "    if ok:\n",
            "        print(f\"anchored={ok.group()}\")\n",
            "    miss = re.match(r\"\\d+\", \"ab12\")\n",
            "    if miss:\n",
            "        print(\"unexpected\")\n",
            "    else:\n",
            "        print(\"no match at start\")\n",
            "    nums = re.findall(r\"\\d+\", \"a1 b22 c333\")\n",
            "    print(f\"nums={repr(nums)}\")\n",
            "    tagged = re.sub(r\"(\\d+)\", r\"<\\1>\", \"a1 b22\")\n",
            "    print(f\"tagged={tagged}\")\n",
            "    parts = re.split(r\"[,;]\\s*\", \"a, b;c\")\n",
            "    print(f\"parts={repr(parts)}\")\n",
            "    whole = re.fullmatch(r\"\\w+\", \"hello\")\n",
            "    if whole:\n",
            "        print(f\"full={whole.group()}\")\n",
            "    return 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/re_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "whole=12-34 a=12 b=34",
            "span=6,11",
            "anchored=12",
            "no match at start",
            "nums=['1', '22', '333']",
            "tagged=a<1> b<22>",
            "parts=['a', 'b', 'c']",
            "full=hello",
        ],
        "re semantics diverged from CPython"
    );
}
