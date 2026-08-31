//! Integration tests: discover and convert sample Python packages, verify
//! the generated crate layout, and compile a converted package for real.

use std::fs;
use std::path::Path;
use std::process::Command;

mod common;

use common::Scratch;
use rypip::convert::ConvertOptions;

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
        manifest.contains("python = [\"dep:pyo3\", \"dep:pyo3-build-config\"]"),
        "manifest: {}",
        manifest
    );
    // The bindings call `.map_err(pyo3::PyErr::from)`, which needs
    // stdpython's `From<PyException> for pyo3::PyErr` — the one thing
    // behind its `pyo3-interop` feature. Ordinary conversions leave it off
    // (see the surface test), so this mode has to ask for it.
    assert!(
        manifest.contains("pyo3-interop"),
        "--pyo3 must enable stdpython's pyo3-interop feature: {}",
        manifest
    );
    // The generated build script requests pyo3's extension-module link
    // args: without them a macOS linker rejects the cdylib's undefined
    // `_Py_*` symbols (they resolve against the loading interpreter).
    let build_rs = fs::read_to_string(out.join("build.rs")).unwrap();
    assert!(
        build_rs.contains("add_extension_module_link_args"),
        "build.rs must request pyo3's link args: {build_rs}"
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
fn true_division_by_zero_raises_catchable_zero_division_error_at_runtime() {
    // Issue #107: `/` used to lower to an infallible py_div, so `1 / 0`
    // silently printed `inf` where CPython raises ZeroDivisionError —
    // wrong output with no conversion error, no exception, no panic. The
    // division now goes through the Result-returning helper, so a zero
    // divisor raises the same catchable error as `//`/`%` (#75), with
    // CPython's exact messages.
    let scratch = Scratch::new("true-div-zero");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def try_int() -> str:\n",
            "    try:\n",
            "        x = 1 / 0\n",
            "        return \"no error\"\n",
            "    except ZeroDivisionError as e:\n",
            "        return str(e)\n",
            "\n",
            "def try_float() -> str:\n",
            "    try:\n",
            "        x = 1.0 / 0.0\n",
            "        return \"no error\"\n",
            "    except ZeroDivisionError as e:\n",
            "        return str(e)\n",
            "\n",
            "def try_mixed() -> str:\n",
            "    try:\n",
            "        x = 1 / 0.0\n",
            "        return \"no error\"\n",
            "    except ZeroDivisionError as e:\n",
            "        return str(e)\n",
            "\n",
            "def try_aug() -> str:\n",
            "    y = 1.0\n",
            "    try:\n",
            "        y /= 0.0\n",
            "        return \"no error\"\n",
            "    except ZeroDivisionError as e:\n",
            "        return str(e)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(try_int())\n",
            "    print(try_float())\n",
            "    print(try_mixed())\n",
            "    print(try_aug())\n",
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
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "division by zero",
            "float division by zero",
            "float division by zero",
            "float division by zero",
        ],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
    assert_eq!(output.status.code(), Some(0), "all divisions were caught");
}

#[test]
fn stdlib_divergence_fixes_match_cpython_at_runtime() {
    // Issue #82 end-to-end: math (IEEE remainder, ldexp, pow domain
    // errors) and json (insertion order, exact big integers) must behave
    // like CPython from transpiled Python. This also proves the codegen
    // threads `?` through Result-returning stdlib module calls (math.sqrt,
    // math.pow, json.loads — previously rustc type errors).
    let scratch = Scratch::new("stdlib-divergences");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "import math\n",
            "import json\n",
            "\n",
            "def math_checks() -> str:\n",
            "    out = []\n",
            "    out.append(str(math.sqrt(144.0)))\n",
            "    out.append(str(math.remainder(1e17, 3.0)))\n",
            "    out.append(str(math.remainder(10.0, 0.1)))\n",
            "    out.append(str(math.ldexp(1e-300, 1074)))\n",
            "    out.append(str(math.fmod(9.5, 3.0)))\n",
            "    out.append(str(math.pow(-8.0, 3.0)))\n",
            "    try:\n",
            "        x = math.pow(0.0, -1.0)\n",
            "        out.append(str(\"no error\"))\n",
            "    except ValueError as e:\n",
            "        out.append(\"caught \" + str(e))\n",
            "    try:\n",
            "        x = math.ldexp(1.0, 2000)\n",
            "        out.append(str(\"no error\"))\n",
            "    except OverflowError as e:\n",
            "        out.append(\"caught \" + str(e))\n",
            "    return \" | \".join(out)\n",
            "\n",
            "def json_checks() -> str:\n",
            "    parsed = json.loads('{\"b\": 1, \"a\": 2, \"c\": 3}')\n",
            "    out = [json.dumps(parsed, None)]\n",
            "    big = json.loads('123456789012345678901234567890')\n",
            "    out.append(json.dumps(big, None))\n",
            "    return \" | \".join(out)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(math_checks())\n",
            "    print(json_checks())\n",
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
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "12.0 | 1.0 | -5.551115123125783e-16 | 2.0240225330731062e+23 | 0.5 | -512.0 | caught math domain error | caught math range error",
            r#"{"b": 1, "a": 2, "c": 3} | 123456789012345678901234567890"#,
        ],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
    assert_eq!(output.status.code(), Some(0));
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
fn class_instance_global_singleton_matches_python_at_runtime() {
    // Issue #189: the lazy-singleton shape (botocore's history.py) — a
    // None-initialized module global whose `global`-writing getter stores
    // exactly one local class construction — lowers to a typed
    // `Mutex<Option<Class>>` static: the None check reads the Option, the
    // store wraps in Some, and the getter returns the instance. Identity
    // across reads follows rython's by-design value semantics (#79): the
    // observable output here is identical to CPython's.
    let scratch = Scratch::new("singleglobal");
    let file = scratch.path().join("singleglobal.py");
    fs::write(
        &file,
        concat!(
            "class Recorder:\n",
            "    def __init__(self) -> None:\n",
            "        self.events: list[str] = []\n",
            "\n",
            "    def record(self, event: str) -> None:\n",
            "        self.events.append(event)\n",
            "\n",
            "RECORDER = None\n",
            "\n",
            "def get_recorder():\n",
            "    global RECORDER\n",
            "    if RECORDER is None:\n",
            "        RECORDER = Recorder()\n",
            "    return RECORDER\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    r = get_recorder()\n",
            "    r.record(\"kept\")\n",
            "    print(r.events)\n",
            "    print(get_recorder() is None)\n",
            "    print(RECORDER is None)\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/singleglobal"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["['kept']", "False", "False"],
        "class-instance global semantics diverged from CPython"
    );
}

#[test]
fn functools_partial_keyword_bindings_match_python_at_runtime() {
    // Keyword bindings emitting in the callee's declared order
    // (botocore's `partial(delay_exponential, base=base,
    // growth_factor=growth_factor)` — issue #189 family): the suffix
    // keyword shape (`x=` bound, `hi` unbound) is called POSITIONALLY,
    // exactly like CPython's partial protocol allows.
    let scratch = Scratch::new("partialkw");
    let file = scratch.path().join("partialkw.py");
    fs::write(
        &file,
        concat!(
            "import functools\n",
            "\n",
            "def clamp(lo: int, hi: int, x: int) -> int:\n",
            "    return 1000 * lo + hi + x\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    unit = functools.partial(clamp, 0, x=5)\n",
            "    print(unit(100))\n",
            "    print(unit(-7))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/partialkw"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["105", "-2"],
        "functools.partial keyword bindings diverged from CPython"
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
fn isinstance_dispatch_specializes_and_matches_python_at_runtime() {
    // The isinstance-dispatch idiom end to end: the converter emits one
    // specialized function per input type (classes get per-CONCRETE-class
    // variants folded through the inheritance tree, so a Cat argument
    // takes the `isinstance(x, Animal)` arm while keeping Cat's own
    // speak() override) plus a generic residual, and call sites dispatch
    // statically. Output must match CPython exactly.
    let scratch = Scratch::new("isinstance-dispatch");
    let file = scratch.path().join("animals.py");
    fs::write(
        &file,
        concat!(
            "class Animal:\n",
            "    def __init__(self, name: str):\n",
            "        self.name = name\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"...\"\n",
            "\n",
            "class Dog(Animal):\n",
            "    def __init__(self, name: str):\n",
            "        super().__init__(name)\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "class Cat(Animal):\n",
            "    def __init__(self, name: str):\n",
            "        super().__init__(name)\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"meow\"\n",
            "\n",
            "def describe(x):\n",
            "    if isinstance(x, Dog):\n",
            "        return x.name + \" is a dog: \" + x.speak()\n",
            "    if isinstance(x, Animal):\n",
            "        return x.name + \" is some animal: \" + x.speak()\n",
            "    if isinstance(x, int):\n",
            "        return \"the number \" + str(x)\n",
            "    return \"unknown\"\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(describe(Dog(\"rex\")))\n",
            "    print(describe(Cat(\"tom\")))\n",
            "    print(describe(Animal(\"blob\")))\n",
            "    print(describe(7))\n",
            "    print(describe(2.5))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/animals"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "rex is a dog: woof",
            "tom is some animal: meow",
            "blob is some animal: ...",
            "the number 7",
            "unknown",
        ],
        "isinstance dispatch diverged from CPython"
    );
}

#[test]
fn isinstance_dynamic_router_routes_boxed_values_at_runtime() {
    // The dynamic router end to end: statically-typed call sites still
    // bind their compile-time morph directly, while a BOXED argument (a
    // `str | int` return, PyValue at runtime) passes through the
    // `impl Into<LabelArg>` router under the original function name and
    // is routed by From<PyValue> in Python's first-true-test order.
    // Output must match CPython exactly.
    let scratch = Scratch::new("isinstance-router");
    let file = scratch.path().join("router.py");
    fs::write(
        &file,
        concat!(
            "class Animal:\n",
            "    def __init__(self, name: str):\n",
            "        self.name = name\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"...\"\n",
            "\n",
            "class Dog(Animal):\n",
            "    def __init__(self, name: str):\n",
            "        super().__init__(name)\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "def label(x):\n",
            "    if isinstance(x, str):\n",
            "        return \"word: \" + x\n",
            "    if isinstance(x, int):\n",
            "        return \"count: \" + str(x)\n",
            "    if isinstance(x, Animal):\n",
            "        return \"pet \" + x.name + \": \" + x.speak()\n",
            "    return \"mystery\"\n",
            "\n",
            "def pick(flag: bool) -> str | int:\n",
            "    if flag:\n",
            "        return \"fox\"\n",
            "    return 42\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(label(\"fox\"))\n",
            "    print(label(12))\n",
            "    print(label(Dog(\"rex\")))\n",
            "    print(label(2.5))\n",
            "    print(label(pick(True)))\n",
            "    print(label(pick(False)))\n",
            "    print(label(True))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let generated = fs::read_to_string(krate.root.join("src/router.rs")).unwrap();
    assert!(
        generated.contains("impl Into<LabelArg>"),
        "the router must take impl Into<LabelArg>: {}",
        generated
    );
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/router"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "word: fox",
            "count: 12",
            "pet rex: woof",
            "mystery",
            "word: fox",
            "count: 42",
            // bool ⊂ int: True takes the int arm but str(x) still renders
            // True — the auto-emitted bool morph keeps the Rust bool.
            "count: True",
        ],
        "dynamic router dispatch diverged from CPython"
    );
}

#[test]
fn router_generalizations_match_python_at_runtime() {
    // The router generalizations end to end: an untested extra parameter
    // passes through the router positionally (`tag`), diverging morph
    // returns land through the output enum and box at the call site
    // (`flip` — a boxed argument yields Python's `int | str` union), and
    // SEVERAL isinstance-tested parameters cross-product into per-combo
    // morphs with per-axis numbered enums (`pair` — static, mixed
    // static/boxed, and fully boxed calls). Output must match CPython
    // exactly.
    let scratch = Scratch::new("router-general");
    let file = scratch.path().join("routergen.py");
    fs::write(
        &file,
        concat!(
            "def pick(flag: bool) -> str | int:\n",
            "    if flag:\n",
            "        return \"fox\"\n",
            "    return 42\n",
            "\n",
            "def tag(x, prefix: str):\n",
            "    if isinstance(x, str):\n",
            "        return prefix + \": \" + x\n",
            "    if isinstance(x, int):\n",
            "        return prefix + \" #\" + str(x)\n",
            "    return prefix + \"?\"\n",
            "\n",
            "def flip(x):\n",
            "    if isinstance(x, str):\n",
            "        return len(x)\n",
            "    if isinstance(x, int):\n",
            "        return str(x)\n",
            "    return 0\n",
            "\n",
            "def pair(a, b):\n",
            "    if isinstance(a, str):\n",
            "        if isinstance(b, int):\n",
            "            return a + \" x\" + str(b)\n",
            "        return a + \" ?\"\n",
            "    if isinstance(a, int):\n",
            "        if isinstance(b, int):\n",
            "            return str(a * b)\n",
            "        return str(a)\n",
            "    return \"neither\"\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(tag(\"fox\", \"word\"))\n",
            "    print(tag(9, \"num\"))\n",
            "    print(tag(2.5, \"odd\"))\n",
            "    print(tag(pick(True), \"dyn\"))\n",
            "    print(tag(pick(False), \"dyn\"))\n",
            "    print(flip(\"fox\"))\n",
            "    print(flip(7))\n",
            "    print(flip(pick(True)))\n",
            "    print(flip(pick(False)))\n",
            "    print(flip(2.5))\n",
            "    print(pair(\"fox\", 3))\n",
            "    print(pair(\"fox\", 2.5))\n",
            "    print(pair(2, 3))\n",
            "    print(pair(2, \"z\"))\n",
            "    print(pair(2.5, 1))\n",
            "    print(pair(pick(True), 3))\n",
            "    print(pair(pick(False), pick(False)))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let generated = fs::read_to_string(krate.root.join("src/routergen.rs")).unwrap();
    for shape in [
        "prefix: impl Into<String>",
        "enum FlipOut",
        "enum PairArg1",
        "enum PairArg2",
        "pub fn pair(a: impl Into<PairArg1>, b: impl Into<PairArg2>)",
    ] {
        assert!(
            generated.contains(shape),
            "missing {shape}: {}",
            generated
        );
    }
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/routergen"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "word: fox",
            "num #9",
            "odd?",
            "dyn: fox",
            "dyn #42",
            "3",
            "7",
            "3",
            "42",
            "0",
            "fox x3",
            "fox ?",
            "6",
            "2",
            "neither",
            "fox x3",
            "1764",
        ],
        "generalized router dispatch diverged from CPython"
    );
}

#[test]
fn inference_seed_unification_and_return_unification_at_runtime() {
    // Parameter type inference end to end: a literal-seeded accumulator
    // concretizes the loop element (`best = ""` → Item = String; `s = 0`
    // → Item = i64), parameters returned as bare values unify into one
    // type variable (`clamp<T>` callable with ints, floats, and strings),
    // and chained operator expressions carry their intermediate Output
    // bounds (`lerp` callable with floats and ints).
    let scratch = Scratch::new("infer-e2e");
    let file = scratch.path().join("infer_e2e.py");
    fs::write(
        &file,
        concat!(
            "def longest(words):\n",
            "    best = \"\"\n",
            "    for w in words:\n",
            "        if len(w) > len(best):\n",
            "            best = w\n",
            "    return best\n",
            "\n",
            "def total(items):\n",
            "    s = 0\n",
            "    for x in items:\n",
            "        s = s + x\n",
            "    return s\n",
            "\n",
            "def clamp(value, low, high):\n",
            "    if value < low:\n",
            "        return low\n",
            "    if value > high:\n",
            "        return high\n",
            "    return value\n",
            "\n",
            "def lerp(a, b, t):\n",
            "    return a + (b - a) * t\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(longest(\"the quick brown fox\".split()))\n",
            "    print(total([3, 1, 4, 1, 5]))\n",
            "    print(clamp(12, 0, 10))\n",
            "    print(clamp(0.25, 0.5, 2.0))\n",
            "    print(clamp(\"m\", \"a\", \"f\"))\n",
            "    print(lerp(0.0, 10.0, 0.25))\n",
            "    print(lerp(100, 200, 2))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/infer_e2e"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["quick", "14", "10", "0.5", "f", "2.5", "300"],
        "inferred-generic functions diverged from CPython"
    );
}

#[test]
fn inherited_container_clear_mutates_the_real_field() {
    // A mutating container method (`self.regs.clear()`) reached through an
    // INHERITED method runs in the trait default, where the load-flavor
    // field accessor clones — the clear must route through the mutable
    // accessor or it silently vanishes (found via examples/03: the Device
    // RESET protocol cleared a clone and DUMP still showed the registers).
    let scratch = Scratch::new("inherited-clear");
    let file = scratch.path().join("bankclear.py");
    fs::write(
        &file,
        concat!(
            "class Bank:\n",
            "    def __init__(self, regs: dict[int, int]):\n",
            "        self.regs = regs\n",
            "\n",
            "    def clear(self) -> None:\n",
            "        self.regs.clear()\n",
            "\n",
            "class Dev(Bank):\n",
            "    def __init__(self, regs: dict[int, int]):\n",
            "        super().__init__(regs)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    d = Dev({1: 2})\n",
            "    d.clear()\n",
            "    print(len(d.regs))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/bankclear"))
        .output()
        .expect("running generated binary");
    // Verified against python3: clear() through the inherited method
    // empties the dict, so len is 0.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "0",
        "inherited clear() must mutate the real field, not a clone"
    );
}

#[test]
fn super_dispatch_keeps_the_derived_self_at_runtime() {
    // super() must run the ancestor's ORIGINAL body with the DERIVED self:
    // a `self.speak()` inside Animal.describe (reached through
    // Dog.describe -> super().describe()) must dispatch to Dog.speak, not
    // Animal.speak. The three-level chain additionally exercises trampoline
    // hops (C.m -> B.m -> A.m -> C.tag).
    let scratch = Scratch::new("super-dispatch");
    let file = scratch.path().join("super_dispatch.py");
    fs::write(
        &file,
        concat!(
            "class Animal:\n",
            "    def __init__(self, name: str):\n",
            "        self.name = name\n",
            "\n",
            "    def describe(self) -> str:\n",
            "        return self.speak()\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"...\"\n",
            "\n",
            "class Dog(Animal):\n",
            "    def __init__(self, name: str):\n",
            "        super().__init__(name)\n",
            "\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "    def describe(self) -> str:\n",
            "        return \"D:\" + super().describe()\n",
            "\n",
            "class A:\n",
            "    def m(self) -> str:\n",
            "        return self.tag()\n",
            "\n",
            "    def tag(self) -> str:\n",
            "        return \"A\"\n",
            "\n",
            "class B(A):\n",
            "    def m(self) -> str:\n",
            "        return super().m() + \"-B\"\n",
            "\n",
            "    def tag(self) -> str:\n",
            "        return \"B\"\n",
            "\n",
            "class C(B):\n",
            "    def m(self) -> str:\n",
            "        return super().m() + \"-C\"\n",
            "\n",
            "    def tag(self) -> str:\n",
            "        return \"C\"\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    d = Dog(\"rex\")\n",
            "    print(d.describe())\n",
            "    c = C()\n",
            "    print(c.m())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/super_dispatch"))
        .output()
        .expect("running generated binary");
    // Verified against python3: nested dispatch through super() must stay on
    // the derived class (Dog.speak, C.tag).
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["D:woof", "C-B-C"],
        "super() lost the derived self: nested calls must dispatch to the override"
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
fn kernel_module_rejects_floating_point_loudly() {
    // Issue #87: the kernel runs with the FPU in a lazy-save state, so
    // floating-point code must be a loud conversion error, never silently
    // dropped or mis-lowered.
    let scratch = Scratch::new("kernel-fp");
    let cases: &[(&str, &str, &str)] = &[
        (
            "float_return.py",
            "def module_init() -> int:\n    printk(\"x\\n\")\n    return 1.5\n",
            "floating-point",
        ),
        (
            "float_assign.py",
            "def module_init() -> int:\n    ratio = 1.5\n    return 0\n",
            "floating-point",
        ),
        (
            "float_param.py",
            "def module_init(scale: float) -> int:\n    return 0\n",
            "floating-point",
        ),
        (
            "float_return_ann.py",
            "def module_init() -> float:\n    return 1\n",
            "floating-point",
        ),
        (
            "float_list_ann.py",
            "def module_init() -> list[float]:\n    return [1, 2]\n",
            "floating-point",
        ),
        (
            "float_call.py",
            "def module_init() -> int:\n    x = float(\"1.5\")\n    return 0\n",
            "floating-point",
        ),
        (
            "import_math.py",
            "import math\n\ndef module_init() -> int:\n    return 0\n",
            "floating-point",
        ),
        (
            "from_random.py",
            "from random import randint\n\ndef module_init() -> int:\n    return 0\n",
            "floating-point",
        ),
        (
            "float_nested.py",
            "def module_init() -> int:\n    vals = [1.0, 2.0]\n    return 0\n",
            "floating-point",
        ),
        (
            "float_if.py",
            "def module_init() -> int:\n    if 1.5 > 1:\n        return 0\n    return 1\n",
            "floating-point",
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
                kernel_module: true,
                ..Default::default()
            },
        )
        .expect_err("floating-point kernel code must fail the conversion");
        let msg = format!("{:#}", err);
        assert!(msg.contains(needle), "{}: {}", name, msg);
        assert!(
            msg.contains("kernel_fpu_begin"),
            "{}: error must mention the FPU guard workaround: {}",
            name,
            msg
        );
    }
}

#[test]
fn kernel_module_lowers_printk_fstrings_and_locals() {
    // Issue #84: printk takes a format string; f-string interpolations lower
    // to %ld conversions with the interpolated value as a vararg, and
    // integer-literal locals give interpolations something to reference.
    let scratch = Scratch::new("kernel-printk");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "\n",
            "def module_init() -> int:\n",
            "    addr = 0x1fff0000\n",
            "    printk(f\"Module loaded at {addr}\\n\")\n",
            "    printk(\"100% ready\\n\")\n",
            "    return 0\n",
            "\n",
            "def module_exit():\n",
            "    printk(\"Goodbye, kernel!\\n\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("f-string printk module converts");
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    // 0x1fff0000 == 536805376
    assert!(
        lib.contains("let addr: i64 = 536805376;"),
        "integer-literal local: {}",
        lib
    );
    assert!(
        lib.contains("b\"Module loaded at %ld\\n\\0\""),
        "f-string to %ld: {}",
        lib
    );
    assert!(lib.contains(", addr"), "vararg passes the local: {}", lib);
    assert!(
        lib.contains("b\"100%% ready\\n\\0\""),
        "literal % escaped for the C format parser: {}",
        lib
    );

    // The generated crate is genuine no_std Rust: it must cargo-check.
    let status = check_generated(&out);
    assert!(status.success(), "generated kernel crate failed to compile");
}

#[test]
fn kernel_module_printk_rejects_unsupported_forms_loudly() {
    let scratch = Scratch::new("kernel-printk-bad");
    let cases: &[(&str, &str, &str)] = &[
        (
            "non_string.py",
            "def module_init() -> int:\n    printk(42)\n    return 0\n",
            "format must be a string literal",
        ),
        (
            "two_args.py",
            "def module_init() -> int:\n    printk(\"a\", \"b\")\n    return 0\n",
            "exactly one argument",
        ),
        (
            "repr_conv.py",
            "def module_init() -> int:\n    printk(f\"{x!r}\")\n    return 0\n",
            "conversions (!s/!r)",
        ),
        (
            "format_spec.py",
            "def module_init() -> int:\n    printk(f\"{x:04}\")\n    return 0\n",
            "format specs",
        ),
        (
            "helper_call.py",
            "def module_init() -> int:\n    helper()\n    return 0\n",
            "unsupported call",
        ),
        (
            "return_name.py",
            "def module_init() -> int:\n    return foo\n",
            "unsupported expression",
        ),
        (
            "expr_interp.py",
            "def module_init() -> int:\n    printk(f\"{1 + 2}\")\n    return 0\n",
            "interpolations support integer values",
        ),
        (
            "params.py",
            "def module_init(n: int) -> int:\n    return 0\n",
            "must take no parameters",
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
                kernel_module: true,
                ..Default::default()
            },
        )
        .expect_err("unsupported kernel-body construct must fail the conversion");
        let msg = format!("{:#}", err);
        assert!(msg.contains(needle), "{}: {}", name, msg);
    }
}

#[test]
fn kernel_module_accepts_float_free_module() {
    // Positive control: integer/string-only kernel code converts cleanly.
    let scratch = Scratch::new("kernel-ok");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "\n",
            "def module_init() -> int:\n",
            "    printk(\"Hello, kernel!\\n\")\n",
            "    return 0\n",
            "\n",
            "def module_exit():\n",
            "    printk(\"Goodbye, kernel!\\n\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("float-free kernel module converts");
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("init_module"), "lib.rs: {}", lib);
    assert!(lib.contains("cleanup_module"), "lib.rs: {}", lib);
    assert!(!lib.contains("f64"), "no floating point in lib.rs: {}", lib);
    assert!(!krate.has_binary, "kernel output is a library");

    // The C-free kernel build needs the allocator to bind to the kernel's
    // exported symbol (__kmalloc_noprof on 7.x kernels), .modinfo metadata
    // kept alive with #[used], and a fmt-free panic handler (the fmt
    // machinery pulls core code with GOTPCREL relocations the module loader
    // rejects).
    assert!(lib.contains("__kmalloc_noprof"), "7.x allocator export: {}", lib);
    assert!(lib.contains("#[used]"), "modinfo survives --gc-sections: {}", lib);
    assert!(lib.contains("#[panic_handler]"), "panic handler: {}", lib);
    assert!(lib.contains("fn panic"), "panic handler: {}", lib);
}

#[test]
fn kernel_module_accepts_docstrings_in_function_bodies() {
    // Issue #83: the canonical hello-world kernel module opens module_init
    // with a docstring. Docstrings are expression statements of string
    // literals; they must be dropped (like CPython does) instead of
    // rejected as unsupported kernel expressions.
    let scratch = Scratch::new("kernel-docstring");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "\n",
            "def module_init() -> int:\n",
            "    \"\"\"Called on insmod. Return 0 on success, negative errno on failure.\"\"\"\n",
            "    printk(\"Hello, kernel!\\n\")\n",
            "    return 0\n",
            "\n",
            "def module_exit():\n",
            "    \"\"\"Called on rmmod.\"\"\"\n",
            "    printk(\"Goodbye, kernel!\\n\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let _krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("docstring-carrying kernel module converts");
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("init_module"), "lib.rs: {}", lib);
    assert!(lib.contains("cleanup_module"), "lib.rs: {}", lib);
    // The docstrings must not leak into the generated Rust as stray
    // expression statements.
    assert!(!lib.contains("Called on insmod"), "lib.rs: {}", lib);
    assert!(!lib.contains("Called on rmmod"), "lib.rs: {}", lib);
}

#[test]
fn kernel_module_imports_shim_to_call_kernel_resource() {
    // Kernel resources via the Rust compatibility layer: `from
    // rykernel_shim import ktime_get_real_seconds` binds the import to a
    // direct call into the shim crate (which declares the kernel's exported
    // symbol), the result is held in an i64 local, and printk interpolates
    // it. The module links the shim and does NOT define its own panic
    // handler (the shim provides one).
    let scratch = Scratch::new("kernel-shim");
    let file = scratch.path().join("clock.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "__module_name__ = \"ryclock\"\n",
            "\n",
            "from rykernel_shim import ktime_get_real_seconds\n",
            "\n",
            "def module_init() -> int:\n",
            "    now = ktime_get_real_seconds()\n",
            "    printk(f\"ryclock: loaded at t={now} s\\n\")\n",
            "    return 0\n",
            "\n",
            "def module_exit():\n",
            "    ktime_get_real_seconds()\n",
            "    printk(\"ryclock: unloaded\\n\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let _krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("shim-import kernel module converts");
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("let now: i64 = rykernel_shim::ktime_get_real_seconds();"),
        "call binding: {}",
        lib
    );
    assert!(
        lib.contains("b\"ryclock: loaded at t=%ld s\\n\\0\""),
        "f-string interpolation: {}",
        lib
    );
    assert!(lib.contains(", now"), "vararg passes the local: {}", lib);
    assert!(
        lib.contains("let _ = rykernel_shim::ktime_get_real_seconds();"),
        "bare shim call statement: {}",
        lib
    );
    assert!(
        !lib.contains("#[panic_handler]"),
        "the shim owns the panic handler: {}",
        lib
    );

    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(
        toml.contains("rykernel-shim = { path ="),
        "Cargo.toml declares the shim dep: {}",
        toml
    );
    assert!(
        !toml.contains("stdpython"),
        "shim modules replace the stdpython dep: {}",
        toml
    );

    // The generated crate is genuine no_std Rust: it must cargo-check
    // (host build — the shim compiles as std, and its extern "C" kernel
    // symbols are declarations only, so nothing needs linking).
    let status = check_generated(&out);
    assert!(status.success(), "generated shim kernel crate failed to compile");
}

#[test]
fn kernel_module_rejects_unknown_shim_import_loudly() {
    // Only the curated allowlist of safe shim wrappers is importable; a
    // name that is not a kernel resource is a loud conversion error.
    let scratch = Scratch::new("kernel-shim-bad");
    let file = scratch.path().join("clock.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "\n",
            "from rykernel_shim import nonexistent\n",
            "\n",
            "def module_init() -> int:\n",
            "    return 0\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect_err("unknown shim import must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("rykernel_shim.nonexistent"),
        "names the bad import: {msg}"
    );
    assert!(
        msg.contains("ktime_get_real_seconds"),
        "lists the available resources: {msg}"
    );
}

#[test]
fn kernel_module_rejects_shim_imports_with_rust_for_linux() {
    // The shim is owned by the raw-FFI pipeline; rust-for-linux binds the
    // kernel crate instead.
    let scratch = Scratch::new("kernel-shim-rfl");
    let file = scratch.path().join("clock.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "\n",
            "from rykernel_shim import ktime_get_real_seconds\n",
            "\n",
            "def module_init() -> int:\n",
            "    return 0\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            rust_for_linux: true,
            ..Default::default()
        },
    )
    .expect_err("shim imports with rust-for-linux must be rejected");
    assert!(
        format!("{err:#}").contains("--rust-for-linux"),
        "names the conflicting target: {err:#}"
    );
}

#[test]
fn kernel_module_rust_for_linux_generates_module_macro() {
    // Issue #88: --kernel-module --rust-for-linux must emit a rust-for-linux
    // crate: a module! macro, kernel::Module impl with init(), a Drop impl
    // for module_exit, pr_info! printk lowering with Rust format strings
    // ({}-placeholders, doubled braces, literal %), and no raw-FFI machinery.
    let scratch = Scratch::new("kernel-rfl");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "__module_license__ = \"GPL\"\n",
            "__module_author__ = \"Erica\"\n",
            "__module_description__ = \"A hello-world kernel module\"\n",
            "\n",
            "def module_init() -> int:\n",
            "    addr = 0x1fff0000\n",
            "    printk(f\"Module loaded at {addr}\\n\")\n",
            "    printk(\"100% ready\\n\")\n",
            "    printk(\"set {x} literal\\n\")\n",
            "    return 0\n",
            "\n",
            "def module_exit():\n",
            "    printk(\"Goodbye, kernel!\\n\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            rust_for_linux: true,
            ..Default::default()
        },
    )
    .expect("rust-for-linux module converts");
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();

    // Skeleton: no_std/no_main, kernel prelude, module! macro with metadata.
    assert!(lib.contains("#![no_std]"), "lib.rs: {}", lib);
    assert!(lib.contains("#![no_main]"), "lib.rs: {}", lib);
    assert!(
        lib.contains("use kernel::prelude::*;"),
        "prelude: {}",
        lib
    );
    assert!(lib.contains("module! {"), "module!: {}", lib);
    assert!(lib.contains("type: Hello,"), "module type: {}", lib);
    assert!(lib.contains("name: \"hello\","), "module name: {}", lib);
    assert!(lib.contains("author: \"Erica\","), "author: {}", lib);
    assert!(
        lib.contains("description: \"A hello-world kernel module\","),
        "description: {}",
        lib
    );
    assert!(lib.contains("license: \"GPL\","), "license: {}", lib);

    // init -> kernel::Module::init returning Ok(Hello).
    assert!(
        lib.contains("impl kernel::Module for Hello {"),
        "Module impl: {}",
        lib
    );
    assert!(
        lib.contains("fn init(_module: &'static ThisModule) -> Result<Self>"),
        "init signature: {}",
        lib
    );
    assert!(lib.contains("Ok(Hello)"), "Ok return: {}", lib);

    // exit -> Drop.
    assert!(lib.contains("impl Drop for Hello {"), "Drop: {}", lib);
    assert!(lib.contains("fn drop(&mut self)"), "drop: {}", lib);

    // printk -> pr_info! with Rust format dialect: {}-placeholder, doubled
    // braces for literal braces, % left alone, \n escaped.
    assert!(lib.contains("let addr: i64 = 536805376;"), "local: {}", lib);
    assert!(
        lib.contains("pr_info!(\"Module loaded at {}\\n\", addr);"),
        "f-string to pr_info!: {}",
        lib
    );
    assert!(
        lib.contains("pr_info!(\"100% ready\\n\");"),
        "% is literal in Rust format: {}",
        lib
    );
    assert!(
        lib.contains("pr_info!(\"set {{x}} literal\\n\");"),
        "braces doubled for the Rust format parser: {}",
        lib
    );
    assert!(
        lib.contains("pr_info!(\"Goodbye, kernel!\\n\");"),
        "exit printk: {}",
        lib
    );

    // No raw-FFI machinery.
    assert!(!lib.contains("printk("), "no raw printk FFI: {}", lib);
    assert!(!lib.contains("kmalloc"), "no kmalloc allocator: {}", lib);
    assert!(!lib.contains("init_module"), "no extern init_module: {}", lib);
    assert!(!lib.contains("cleanup_module"), "no extern cleanup_module: {}", lib);
    assert!(!lib.contains(".modinfo"), "no manual .modinfo: {}", lib);
    assert!(!lib.contains("panic_handler"), "kernel crate has its own: {}", lib);

    // Cargo.toml: commented kernel path dep, staticlib, panic=abort, and no
    // stdpython dependency; no standalone Kbuild Makefile (modules are
    // registered in the kernel tree instead).
    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(toml.contains("kernel = { path ="), "kernel dep: {}", toml);
    assert!(!toml.contains("stdpython"), "no stdpython: {}", toml);
    assert!(toml.contains("staticlib"), "staticlib: {}", toml);
    assert!(toml.contains("panic = \"abort\""), "abort: {}", toml);
    assert!(!out.join("Makefile").exists(), "no standalone Makefile");
    assert!(!krate.has_binary, "rust-for-linux output is a library");
}

#[test]
fn kernel_module_rust_for_linux_rejects_unmappable_forms() {
    let scratch = Scratch::new("kernel-rfl-bad");
    let cases: &[(&str, &str, &str)] = &[
        (
            "nonzero_init.py",
            "def module_init() -> int:\n    return 1\n",
            "can only `return 0`",
        ),
        (
            "string_init.py",
            "def module_init() -> int:\n    return \"ok\"\n",
            "can only `return 0`",
        ),
        (
            "exit_returns.py",
            "def module_exit():\n    return 0\n",
            "cannot return a value",
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
                kernel_module: true,
                rust_for_linux: true,
                ..Default::default()
            },
        )
        .expect_err("rust-for-linux unmappable forms must fail the conversion");
        let msg = format!("{:#}", err);
        assert!(msg.contains(needle), "{}: {}", name, msg);
    }

    // --rust-for-linux without --kernel-module is a configuration error.
    let file = scratch.path().join("plain.py");
    fs::write(&file, "x = 1\n").unwrap();
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &scratch.path().join("crate-plain"),
        &ConvertOptions {
            rust_for_linux: true,
            ..Default::default()
        },
    )
    .expect_err("rust_for_linux without kernel_module must fail");
    assert!(
        format!("{:#}", err).contains("requires kernel_module"),
        "err: {:#}",
        err
    );
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
fn round_and_chr_builtins_match_python_at_runtime() {
    // Two-arg round() used to fail to compile (no round_digits wiring);
    // chr() used to panic on surrogates. Both now match CPython (the
    // surrogate code point is a documented exception: CPython returns a
    // lone surrogate, which UTF-8 cannot hold, so rython raises a
    // catchable ValueError instead).
    let scratch = Scratch::new("round_chr");
    let file = scratch.path().join("round_chr.py");
    fs::write(
        &file,
        concat!(
            "def main() -> int:\n",
            "    print(round(1.15, 1))\n",
            "    print(round(2.675, 2))\n",
            "    print(round(2.5))\n",
            "    print(round(1250.0, -2))\n",
            "    print(round(3))\n",
            "    print(chr(65))\n",
            "    try:\n",
            "        chr(0x110000)\n",
            "    except ValueError as e:\n",
            "        print(\"caught\", e)\n",
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

    let output = Command::new(krate.root.join("target/debug/round_chr"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "1.1",
            "2.67",
            "2",
            "1200.0",
            "3",
            "A",
            "caught chr() arg not in range(0x110000)",
        ],
        "round/chr diverged from CPython"
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
            "    ci = re.findall(r\"ab\", \"AB ab\", re.IGNORECASE)\n",
            "    print(f\"ci={repr(ci)}\")\n",
            "    capped = re.sub(r\"a\", \"-\", \"aaaa\", count=2)\n",
            "    print(f\"capped={capped}\")\n",
            "    for m in re.finditer(r\"\\d+\", \"a1 b22\"):\n",
            "        print(f\"fi={m.group(0)}:{m.start()}\")\n",
            "    multi = re.findall(r\"^\\w\", \"ab\\ncd\", re.IGNORECASE | re.MULTILINE)\n",
            "    print(f\"multi={repr(multi)}\")\n",
            "    # A COMPILED pattern's fullmatch constrains the engine: the\n",
            "    # alternation must resolve to the whole-string branch (`a|ab`\n",
            "    # on \"ab\") and a lazy quantifier must expand to the whole\n",
            "    # text (`a*?` on \"aaa\").\n",
            "    _alt = re.compile(\"a|ab\")\n",
            "    for t in (\"ab\", \"a\", \"b\"):\n",
            "        am = _alt.fullmatch(t)\n",
            "        if am is not None:\n",
            "            print(f\"alt {t}={am.group()}\")\n",
            "        else:\n",
            "            print(f\"alt {t}=no\")\n",
            "    _lazy = re.compile(\"a*?\")\n",
            "    for t in (\"aaa\", \"ab\"):\n",
            "        lm = _lazy.fullmatch(t)\n",
            "        if lm is not None:\n",
            "            print(f\"lazy {t}=yes\")\n",
            "        else:\n",
            "            print(f\"lazy {t}=no\")\n",
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
            "ci=['AB', 'ab']",
            "capped=--aa",
            "fi=1:1",
            "fi=22:4",
            "multi=['a', 'c']",
            "alt ab=ab",
            "alt a=a",
            "alt b=no",
            "lazy aaa=yes",
            "lazy ab=no",
        ],
        "re semantics diverged from CPython"
    );
}

#[test]
fn option_returning_functions_wrap_plain_members() {
    // Round 74: a `-> T | None` function returns plain members (a str
    // concatenation, a method call result) and None on other paths —
    // the return site wraps the plain member in Some, lowers `return
    // None` to the None member, and passes an already-Option value
    // through (`return host` after the None guards). Also: an
    // Option-typed receiver slices (`host[start:end]` after `if host:`)
    // unwraps with a loud TypeError panic, and m.span(1) is the
    // group-indexed span. Verified against python3.
    let scratch = Scratch::new("regex_opt");
    let file = scratch.path().join("re_opt.py");
    fs::write(
        &file,
        concat!(
            "import re\n",
            "\n",
            "_ZONE = re.compile(r\"\\[(.*?)\\]\")\n",
            "\n",
            "def normalize(host: str | None, scheme: str | None) -> str | None:\n",
            "    if host:\n",
            "        m = _ZONE.search(host)\n",
            "        if m is not None:\n",
            "            start, end = m.span(1)\n",
            "            return \"z:\" + host[start:end]\n",
            "        return host.lower()\n",
            "    return None\n",
            "\n",
            "def main() -> int:\n",
            "    v = normalize(\"A[Bcd]E\", \"http\")\n",
            "    if v is not None:\n",
            "        print(f\"v={v}\")\n",
            "    else:\n",
            "        print(\"none\")\n",
            "    v = normalize(\"ABC\", \"http\")\n",
            "    if v is not None:\n",
            "        print(f\"v={v}\")\n",
            "    else:\n",
            "        print(\"none\")\n",
            "    v = normalize(None, \"http\")\n",
            "    if v is not None:\n",
            "        print(f\"v={v}\")\n",
            "    else:\n",
            "        print(\"none\")\n",
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

    let output = Command::new(krate.root.join("target/debug/re_opt"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["v=z:Bcd", "v=abc", "none"],
        "Option-returning function semantics diverged from CPython"
    );
}

#[test]
fn regex_brace_patterns_and_slice_shapes_match_python_at_runtime() {
    // Issue #134: a Python pattern's unescaped `{` that does not form a
    // quantifier is literal — the converter escapes it for Rust's regex
    // crate on every re entry point (search/split/sub/findall/...). The
    // `utf8` codec alias, str.replace, in-place clear via `del xs[:]`,
    // and slice-rebuild reads round it out; bounded `del xs[i:j]` and
    // slice assignment are LOUD conversion errors (see codegen_semantics).
    let scratch = Scratch::new("regex-braces");
    let file = scratch.path().join("brace_demo.py");
    fs::write(
        &file,
        concat!(
            "import re\n",
            "\n",
            "def main() -> int:\n",
            "    m = re.search(\"{(.*?)}\", \"a {x} b\")\n",
            "    if m:\n",
            "        print(f\"search={m.group(1)}\")\n",
            "    parts = re.split(\"{;}\", \"a{b}\")\n",
            "    print(f\"split={repr(parts)}\")\n",
            "    tagged = re.sub(\"{(.*?)}\", \"<\\\\1>\", \"a {y} z\")\n",
            "    print(f\"sub={tagged}\")\n",
            "    data = \"caf\\xc3\\xa9\".encode(\"utf8\")\n",
            "    print(f\"utf8={len(data)}\")\n",
            "    print(f\"replaced={'hello'.replace('l', 'L')}\")\n",
            "    ys = [\"a\", \"b\", \"c\"]\n",
            "    del ys[:]\n",
            "    print(f\"cleared={repr(ys)}\")\n",
            "    xs = [1, 2, 3, 4]\n",
            "    xs = xs[:1] + xs[3:]\n",
            "    print(f\"sliceassign={repr(xs)}\")\n",
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

    let output = Command::new(krate.root.join("target/debug/brace_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "search=x",
            "split=['a{b}']",
            "sub=a <y> z",
            "utf8=7",
            "replaced=heLLo",
            "cleared=[]",
            "sliceassign=[1, 4]",
        ],
        "regex-brace / codec / slice semantics diverged from CPython"
    );
}

#[test]
fn heterogeneous_union_annotations_match_python_at_runtime() {
    // Issue #121: `str | bytes` lowers to StrOrBytes; wider boxable
    // unions (`str | bytes | int | float`) to the boxed PyValue. Call
    // sites coerce literals/typed values at the boundary. Verified
    // against python3: tok / b'raw' / s / 7 (bytes str()s as b'raw').
    let scratch = Scratch::new("hetero-union");
    let file = scratch.path().join("u.py");
    fs::write(
        &file,
        concat!(
            "def auth_str(username: str | bytes) -> str:\n",
            "    return str(username)\n",
            "\n",
            "def wide(v: str | bytes | int | float) -> str:\n",
            "    return str(v)\n",
            "\n",
            "def main() -> int:\n",
            "    print(auth_str(\"tok\"))\n",
            "    print(auth_str(b\"raw\"))\n",
            "    print(wide(\"s\"))\n",
            "    print(wide(7))\n",
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

    let output = Command::new(krate.root.join("target/debug/u"))
        .output()
        .expect("running generated binary");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["tok", "b'raw'", "s", "7"],
        "heterogeneous union lowering diverged from CPython"
    );
}

#[test]
fn range_replace_matches_python_at_runtime() {
    // Issue #153: bounded del / slice assignment / strided variants all
    // match CPython. Verified against python3:
    //   [0,3,4,5] / [0,10,20,30,4,5] / [0,10,20,30] /
    //   [7,0,10,20,30] / [7,0] / [9,1,9] / ['Z','c','d']
    let scratch = Scratch::new("range-replace");
    let file = scratch.path().join("seqs.py");
    fs::write(
        &file,
        concat!(
            "def main() -> int:\n",
            "    xs = [0, 1, 2, 3, 4, 5]\n",
            "    del xs[1:3]\n",
            "    print(xs)\n",
            "    xs[1:2] = [10, 20, 30]\n",
            "    print(xs)\n",
            "    del xs[-2:]\n",
            "    print(xs)\n",
            "    xs[0:0] = [7]\n",
            "    print(xs)\n",
            "    xs[2:99] = []\n",
            "    print(xs)\n",
            "    ys = [0, 1, 2]\n",
            "    ys[::2] = [9, 9]\n",
            "    print(ys)\n",
            "    words = [\"a\", \"b\", \"c\", \"d\"]\n",
            "    del words[0:2]\n",
            "    words[0:0] = [\"Z\"]\n",
            "    print(words)\n",
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

    let output = Command::new(krate.root.join("target/debug/seqs"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "[0, 3, 4, 5]",
            "[0, 10, 20, 30, 4, 5]",
            "[0, 10, 20, 30]",
            "[7, 0, 10, 20, 30]",
            "[7, 0]",
            "[9, 1, 9]",
            "['Z', 'c', 'd']",
        ],
        "range-replace semantics diverged from CPython"
    );
}

#[test]
fn heterogeneous_container_literals_box_and_match_python() {
    // Issue #130: mixed-element list literals and mixed-key/value dict
    // literals box into Vec<PyValue> / PyDict<PyValue, PyValue> instead of
    // refusing. Verified against python3: [1, 'a', None] / 2.
    let scratch = Scratch::new("hetero-boxing");
    let file = scratch.path().join("boxed.py");
    fs::write(
        &file,
        concat!(
            "def main() -> int:\n",
            "    xs = [1, \"a\", None]\n",
            "    d = {\"k\": 1, 2: \"v\"}\n",
            "    print(xs)\n",
            "    print(len(d))\n",
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

    let output = Command::new(krate.root.join("target/debug/boxed"))
        .output()
        .expect("running generated binary");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["[1, 'a', None]", "2"],
        "heterogeneous container boxing diverged from CPython"
    );
}

#[test]
fn isinstance_type_call_matches_python_at_runtime() {
    // Issue #134 (charset_normalizer): `isinstance(x, type(self))`
    // resolves `type(...)` to the statically-known class — true for the
    // same class, false against another builtin type. Verified against
    // python3: True / False.
    let scratch = Scratch::new("isinstance-type");
    let file = scratch.path().join("doors.py");
    fs::write(
        &file,
        concat!(
            "class Door:\n",
            "    def __init__(self, ok: bool):\n",
            "        self.ok = ok\n",
            "\n",
            "    def same(self, other: \"Door\") -> bool:\n",
            "        return isinstance(other, type(self))\n",
            "\n",
            "    def diff(self, other: \"Door\") -> bool:\n",
            "        return isinstance(other, type(int))\n",
            "\n",
            "def main() -> int:\n",
            "    d = Door(True)\n",
            "    print(d.same(Door(False)))\n",
            "    print(d.diff(Door(True)))\n",
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

    let output = Command::new(krate.root.join("target/debug/doors"))
        .output()
        .expect("running generated binary");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["True", "False"],
        "isinstance(type()) diverged from CPython"
    );
}

#[test]
fn method_receiver_binding_and_associated_calls_match_python() {
    // Issue #132: Python binds the instance to a method's FIRST parameter
    // whatever its name (boto3's factory_self), a @staticmethod called
    // through an instance binds no receiver, and a @classmethod's cls(v)
    // constructs an instance of the class. Verified against python3:
    //   w.scale(4) -> 8, w.helper(1) -> 10, Widget.helper(2) -> 20,
    //   Widget.make(5).base -> 5
    let scratch = Scratch::new("receiver-binding");
    let file = scratch.path().join("widget.py");
    fs::write(
        &file,
        concat!(
            "class Widget:\n",
            "    def __init__(self, base: int):\n",
            "        self.base = base\n",
            "\n",
            "    def scale(factory_self, times: int) -> int:\n",
            "        return factory_self.base * times\n",
            "\n",
            "    def bump(factory_self, by: int):\n",
            "        factory_self.base = factory_self.base + by\n",
            "        return 0\n",
            "\n",
            "    @staticmethod\n",
            "    def helper(x: int) -> int:\n",
            "        return x * 10\n",
            "\n",
            "    @classmethod\n",
            "    def make(cls, v: int):\n",
            "        return cls(v)\n",
            "\n",
            "def main() -> int:\n",
            "    w = Widget(2)\n",
            "    w.bump(3)\n",
            "    print(w.scale(2))\n",
            "    w2 = Widget(2)\n",
            "    print(w2.scale(4))\n",
            "    print(w.helper(1))\n",
            "    print(Widget.helper(2))\n",
            "    c = Widget.make(5)\n",
            "    print(c.base)\n",
            "    return 0\n",
            "    print(w.helper(1))\n",
            "    print(Widget.helper(2))\n",
            "    c = Widget.make(5)\n",
            "    print(c.base)\n",
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

    let output = Command::new(krate.root.join("target/debug/widget"))
        .output()
        .expect("running generated binary");
    // Verified against python3: w.bump(3) mutates through the renamed
    // receiver (base 2->5, scale(2) prints 10); the rest re-checks the
    // static/classmethod shapes.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["10", "8", "10", "20", "5"],
        "method receiver binding diverged from CPython"
    );
}

#[test]
fn map_filter_list_match_python_at_runtime() {
    // map (lambda, two-iterable, and user-function forms), filter
    // (predicate and None), and the list() builtin over lists, strings,
    // and ranges, through generated code.
    let scratch = Scratch::new("mapfilter");
    let file = scratch.path().join("mf_demo.py");
    fs::write(
        &file,
        concat!(
            "def double(n: int) -> int:\n",
            "    return n * 2\n",
            "\n",
            "def main() -> int:\n",
            "    doubled = list(map(lambda x: x * 2, [1, 2, 3]))\n",
            "    print(f\"doubled={repr(doubled)}\")\n",
            "    summed = list(map(lambda a, b: a + b, [1, 2], [10, 20, 30]))\n",
            "    print(f\"summed={repr(summed)}\")\n",
            "    via_def = list(map(double, [4, 5]))\n",
            "    print(f\"via_def={repr(via_def)}\")\n",
            "    big = list(filter(lambda x: x > 1, [1, 2, 3]))\n",
            "    print(f\"big={repr(big)}\")\n",
            "    truthy = list(filter(None, [0, 3, 0, 5]))\n",
            "    print(f\"truthy={repr(truthy)}\")\n",
            "    chars = list(\"abc\")\n",
            "    print(f\"chars={repr(chars)}\")\n",
            "    nums = list(range(4))\n",
            "    print(f\"nums={repr(nums)}\")\n",
            "    for v in map(lambda x: x + 100, [7, 8]):\n",
            "        print(f\"v={v}\")\n",
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

    let output = Command::new(krate.root.join("target/debug/mf_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "doubled=[2, 4, 6]",
            "summed=[11, 22]",
            "via_def=[8, 10]",
            "big=[2, 3]",
            "truthy=[3, 5]",
            "chars=['a', 'b', 'c']",
            "nums=[0, 1, 2, 3]",
            "v=107",
            "v=108",
        ],
        "map/filter/list semantics diverged from CPython"
    );
}

#[test]
fn hashlib_matches_python_at_runtime() {
    // md5/sha1/sha256 with .encode() data, the empty+update() idiom, and
    // UTF-8 hashing, through generated code (both import spellings).
    let scratch = Scratch::new("hashes");
    let file = scratch.path().join("hash_demo.py");
    fs::write(
        &file,
        concat!(
            "import hashlib\n",
            "from hashlib import sha256\n",
            "\n",
            "def main() -> int:\n",
            "    print(f\"md5={hashlib.md5('hello'.encode()).hexdigest()}\")\n",
            "    print(f\"sha1={hashlib.sha1('hello'.encode()).hexdigest()}\")\n",
            "    print(f\"sha256={sha256('hello'.encode()).hexdigest()}\")\n",
            "    h = sha256()\n",
            "    h.update(\"hel\".encode())\n",
            "    h.update(\"lo\".encode())\n",
            "    print(f\"inc={h.hexdigest()}\")\n",
            "    print(f\"utf8={hashlib.md5('café'.encode('utf-8')).hexdigest()}\")\n",
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

    let output = Command::new(krate.root.join("target/debug/hash_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "md5=5d41402abc4b2a76b9719d911017c592",
            "sha1=aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d",
            "sha256=2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            "inc=2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            "utf8=07117fe4a1ebd544965dc19573183da2",
        ],
        "hashlib semantics diverged from CPython"
    );
}

#[test]
fn textwrap_wrap_matches_python_at_runtime() {
    // wrap (positional and width= keyword), fill, and the catchable
    // invalid-width ValueError, through generated code.
    let scratch = Scratch::new("wraps");
    let file = scratch.path().join("wrap_demo.py");
    fs::write(
        &file,
        concat!(
            "from textwrap import wrap, fill\n",
            "\n",
            "def main() -> int:\n",
            "    for line in wrap(\"The quick brown fox jumps over the lazy dog\", 10):\n",
            "        print(f\"w={line}\")\n",
            "    for line in wrap(\"a self-referential well-known example\", width=12):\n",
            "        print(f\"h={line}\")\n",
            "    print(fill(\"one two three four\", 9))\n",
            "    try:\n",
            "        print(fill(\"x\", 0))\n",
            "    except ValueError:\n",
            "        print(\"width caught\")\n",
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

    let output = Command::new(krate.root.join("target/debug/wrap_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "w=The quick",
            "w=brown fox",
            "w=jumps over",
            "w=the lazy",
            "w=dog",
            "h=a self-",
            "h=referential",
            "h=well-known",
            "h=example",
            "one two",
            "three",
            "four",
            "width caught",
        ],
        "textwrap semantics diverged from CPython"
    );
}

#[test]
fn isinstance_and_hash_match_python_at_runtime() {
    // isinstance decided at conversion time (annotations, literal locals,
    // bool-is-int) and hash() with PYTHONHASHSEED=0 semantics, through
    // generated code.
    let scratch = Scratch::new("ishash");
    let file = scratch.path().join("is_demo.py");
    fs::write(
        &file,
        concat!(
            "def kind(n: int) -> str:\n",
            "    if isinstance(n, int):\n",
            "        return \"int\"\n",
            "    return \"other\"\n",
            "\n",
            "def main() -> int:\n",
            "    print(kind(5))\n",
            "    x = 1.5\n",
            "    print(\"float\" if isinstance(x, float) else \"not\")\n",
            "    print(\"boolint\" if isinstance(True, int) else \"no\")\n",
            "    print(f\"h1={hash('hello')}\")\n",
            "    print(f\"h2={hash(42)}\")\n",
            "    print(f\"h3={hash(-1)}\")\n",
            "    print(f\"h4={hash(1.5)}\")\n",
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

    let output = Command::new(krate.root.join("target/debug/is_demo"))
        .output()
        .expect("running generated binary");
    // Verified against PYTHONHASHSEED=0 python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "int",
            "float",
            "boolint",
            "h1=-2096571579003691106",
            "h2=42",
            "h3=-2",
            "h4=1152921504606846977",
        ],
        "isinstance/hash semantics diverged from CPython"
    );
}

#[test]
fn argparse_matches_python_at_runtime() {
    // The conversion-time parser end to end: help text (layout and
    // column math), value forms (--opt V and --opt=V), prefix
    // abbreviation, store_true, typed defaults, and the exact
    // usage+error output with exit code 2 on missing/invalid/unknown
    // arguments. Everything below was captured from python3 verbatim.
    let scratch = Scratch::new("argps");
    let file = scratch.path().join("arg_demo.py");
    fs::write(
        &file,
        concat!(
            "import argparse\n",
            "\n",
            "def main() -> None:\n",
            "    p = argparse.ArgumentParser(prog=\"tool\", description=\"Demo tool\")\n",
            "    p.add_argument(\"name\", help=\"who to greet\")\n",
            "    p.add_argument(\"count\", type=int)\n",
            "    p.add_argument(\"--verbose\", action=\"store_true\", help=\"say more\")\n",
            "    p.add_argument(\"--scale\", type=float, default=1.0)\n",
            "    p.add_argument(\"--label\", default=\"none\")\n",
            "    args = p.parse_args()\n",
            "    if args.verbose:\n",
            "        print(\"verbose mode\")\n",
            "    print(args.name, args.count * 2, args.scale, args.label)\n",
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
    let bin = krate.root.join("target/debug/arg_demo");

    let usage = "usage: tool [-h] [--verbose] [--scale SCALE] [--label LABEL] name count";

    // --help: python3's exact text, exit 0.
    let output = Command::new(&bin).arg("--help").output().expect("run");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        concat!(
            "usage: tool [-h] [--verbose] [--scale SCALE] [--label LABEL] name count\n",
            "\n",
            "Demo tool\n",
            "\n",
            "positional arguments:\n",
            "  name           who to greet\n",
            "  count\n",
            "\n",
            "options:\n",
            "  -h, --help     show this help message and exit\n",
            "  --verbose      say more\n",
            "  --scale SCALE\n",
            "  --label LABEL\n",
        ),
        "help text diverged from CPython"
    );

    // Successful runs: split and = value forms, prefix abbreviation.
    let output = Command::new(&bin)
        .args(["bob", "3", "--scale", "2.5", "--verbose"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "verbose mode\nbob 6 2.5 none\n"
    );
    let output = Command::new(&bin)
        .args(["bob", "3", "--scale=0.5", "--lab", "x"])
        .output()
        .expect("run");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "bob 6 0.5 x\n");

    // Errors: usage + message on stderr, exit 2, all python3-verbatim.
    let cases = [
        (
            vec!["bob"],
            "tool: error: the following arguments are required: count",
        ),
        (
            vec!["bob", "xx"],
            "tool: error: argument count: invalid int value: 'xx'",
        ),
        (
            vec!["bob", "1", "--bogus"],
            "tool: error: unrecognized arguments: --bogus",
        ),
    ];
    for (args, want) in cases {
        let output = Command::new(&bin).args(&args).output().expect("run");
        assert_eq!(output.status.code(), Some(2), "args: {:?}", args);
        assert_eq!(
            String::from_utf8_lossy(&output.stderr),
            format!("{}\n{}\n", usage, want),
            "args: {:?}",
            args
        );
    }
}

#[test]
fn lru_cache_matches_python_at_runtime() {
    // Memoization through recursion (fib), unbounded caches skipping
    // recomputation (print side effects fire once per distinct
    // argument), and CPython's exact LRU touch/eviction order for a
    // bounded cache.
    let scratch = Scratch::new("lrus");
    let file = scratch.path().join("lru_demo.py");
    fs::write(
        &file,
        concat!(
            "from functools import lru_cache\n",
            "\n",
            "@lru_cache\n",
            "def fib(n: int) -> int:\n",
            "    if n < 2:\n",
            "        return n\n",
            "    return fib(n - 1) + fib(n - 2)\n",
            "\n",
            "@lru_cache(maxsize=None)\n",
            "def slow_double(x: int) -> int:\n",
            "    print(\"computing\", x)\n",
            "    return x * 2\n",
            "\n",
            "@lru_cache(maxsize=2)\n",
            "def tag(s: str) -> str:\n",
            "    print(\"tagging\", s)\n",
            "    return \"<\" + s + \">\"\n",
            "\n",
            "def main() -> None:\n",
            "    print(fib(30))\n",
            "    print(slow_double(4))\n",
            "    print(slow_double(4))\n",
            "    print(slow_double(5))\n",
            "    print(tag(\"a\"), tag(\"b\"))\n",
            "    print(tag(\"a\"))\n",
            "    print(tag(\"c\"))\n",
            "    print(tag(\"b\"))\n",
            "    print(tag(\"a\"))\n",
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

    let output = Command::new(krate.root.join("target/debug/lru_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "832040",
            "computing 4",
            "8",
            "8",
            "computing 5",
            "10",
            "tagging a",
            "tagging b",
            "<a> <b>",
            "<a>",
            "tagging c",
            "<c>",
            "tagging b",
            "<b>",
            "tagging a",
            "<a>",
        ],
        "lru_cache semantics diverged from CPython"
    );
}

#[test]
fn file_objects_and_csv_writer_match_python_at_runtime() {
    // The PyFile surface end to end: io.StringIO (cursor-overwrite
    // write, getvalue, readlines with terminators), csv.writer quoting
    // through both writerow and writerows, disk files via open() in
    // write and read modes, with-blocks, and reader round-trip.
    let scratch = Scratch::new("pyfiles");
    let file = scratch.path().join("file_demo.py");
    fs::write(
        &file,
        concat!(
            "import csv\n",
            "import io\n",
            "\n",
            "def main() -> None:\n",
            "    buf = io.StringIO()\n",
            "    w = csv.writer(buf)\n",
            "    w.writerow([\"a\", \"b,c\", \"say \\\"hi\\\"\", \"\"])\n",
            "    w.writerow([1, 2, 3])\n",
            "    w.writerow([])\n",
            "    w.writerows([[\"x\", \"y\"], [\"z\", \"w\"]])\n",
            "    print(repr(buf.getvalue()))\n",
            "    seeded = io.StringIO(\"seeded\")\n",
            "    seeded.write(\"!\")\n",
            "    print(seeded.getvalue())\n",
            "    print(repr(seeded.read()))\n",
            "    two = io.StringIO(\"x\\ny\\n\")\n",
            "    print(two.readlines())\n",
            "    path = \"pyfile_demo_scratch.txt\"\n",
            "    f = open(path, \"w\")\n",
            "    f.write(\"alpha\\n\")\n",
            "    f.writelines([\"beta\\n\", \"gamma\\n\"])\n",
            "    f.close()\n",
            "    g = open(path)\n",
            "    print(repr(g.readline()))\n",
            "    print(g.readlines())\n",
            "    g.close()\n",
            "    with open(path) as h:\n",
            "        print(len(h.read()))\n",
            "    for row in csv.reader(open(path).readlines()):\n",
            "        print(row)\n",
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

    // Run in a scratch cwd so the relative-path file lands there.
    let output = Command::new(krate.root.join("target/debug/file_demo"))
        .current_dir(scratch.path())
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "'a,\"b,c\",\"say \"\"hi\"\"\",\\r\\n1,2,3\\r\\n\\r\\nx,y\\r\\nz,w\\r\\n'",
            "!eeded",
            "'eeded'",
            "['x\\n', 'y\\n']",
            "'alpha\\n'",
            "['beta\\n', 'gamma\\n']",
            "17",
            "['alpha']",
            "['beta']",
            "['gamma']",
        ],
        "file-object semantics diverged from CPython"
    );
}

#[test]
fn functools_partial_matches_python_at_runtime() {
    // partial over statically-known functions: leading-argument binding,
    // full binding (zero-arg closure), multi-parameter tails, and
    // exception propagation through the bound name.
    let scratch = Scratch::new("partials");
    let file = scratch.path().join("part_demo.py");
    fs::write(
        &file,
        concat!(
            "from functools import partial\n",
            "\n",
            "def add(a: int, b: int) -> int:\n",
            "    return a + b\n",
            "\n",
            "def clamp(lo: int, hi: int, x: int) -> int:\n",
            "    if x < lo:\n",
            "        return lo\n",
            "    if x > hi:\n",
            "        return hi\n",
            "    return x\n",
            "\n",
            "def main() -> None:\n",
            "    add5 = partial(add, 5)\n",
            "    print(add5(3))\n",
            "    print(add5(10))\n",
            "    add_both = partial(add, 2, 3)\n",
            "    print(add_both())\n",
            "    unit = partial(clamp, 0, 100)\n",
            "    print(unit(-4), unit(50), unit(300))\n",
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

    let output = Command::new(krate.root.join("target/debug/part_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["8", "15", "5", "0 50 100"],
        "functools.partial semantics diverged from CPython"
    );
}

#[test]
fn chained_comparisons_dicts_and_try_flow_match_python_at_runtime() {
    // Three correctness fixes at once: a chained comparison evaluating
    // its middle operand exactly once (and short-circuiting the rest),
    // dict and exception rendering, and `break` leaving a loop from
    // inside a try body with a finally clause.
    let scratch = Scratch::new("mixedfix");
    let file = scratch.path().join("d7_all.py");
    fs::write(
        &file,
        concat!(
            "def probe(n: int) -> int:\n",
            "    print(\"eval\", n)\n",
            "    return n\n",
            "\n",
            "def main() -> None:\n",
            "    print(1 < probe(5) < 10)\n",
            "    print(1 < probe(0) < probe(9))\n",
            "    d = {\"b\": 2, \"a\": 1}\n",
            "    print(d)\n",
            "    print(repr(d))\n",
            "    try:\n",
            "        raise ValueError(\"boom\")\n",
            "    except ValueError as err:\n",
            "        print(err)\n",
            "        print(\"msg:\", err)\n",
            "    for i in range(4):\n",
            "        try:\n",
            "            if i == 2:\n",
            "                break\n",
            "        finally:\n",
            "            print(\"fin\", i)\n",
            "    print(\"done\")\n",
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

    let output = Command::new(krate.root.join("target/debug/d7_all"))
        .output()
        .expect("running generated binary");
    // Verified against python3. "eval 5" and "eval 0" appear ONCE each:
    // the middle operand is not re-evaluated, and probe(9) never runs.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "eval 5",
            "True",
            "eval 0",
            "False",
            "{'b': 2, 'a': 1}",
            "{'b': 2, 'a': 1}",
            "boom",
            "msg: boom",
            "fin 0",
            "fin 1",
            "fin 2",
            "done",
        ],
        "chained-comparison / dict / try-flow semantics diverged from CPython"
    );
}

#[test]
fn replace_keywords_match_python_at_runtime() {
    // dt.replace(field=...) through the type-dispatched PyReplace trait:
    // datetime and date receivers, foreign-field TypeError, range
    // ValueError, and str.replace coexisting untouched.
    let scratch = Scratch::new("replkw");
    let file = scratch.path().join("repl_demo.py");
    fs::write(
        &file,
        concat!(
            "from datetime import datetime, date\n",
            "\n",
            "def main() -> None:\n",
            "    d = datetime(2024, 2, 29, 13, 5, 7, 123456)\n",
            "    print(d.replace(hour=14))\n",
            "    print(d.replace(year=2023, day=28))\n",
            "    print(d.replace(minute=0, second=0, microsecond=0))\n",
            "    dd = date(2024, 2, 29)\n",
            "    print(dd.replace(month=3, day=1))\n",
            "    try:\n",
            "        print(dd.replace(hour=1))\n",
            "    except TypeError:\n",
            "        print(\"date has no hour\")\n",
            "    try:\n",
            "        print(d.replace(month=2, day=30))\n",
            "    except ValueError:\n",
            "        print(\"day out of range caught\")\n",
            "    s = \"banana\"\n",
            "    print(s.replace(\"a\", \"o\"))\n",
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

    let output = Command::new(krate.root.join("target/debug/repl_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "2024-02-29 14:05:07.123456",
            "2023-02-28 13:05:07.123456",
            "2024-02-29 13:00:00",
            "2024-03-01",
            "date has no hour",
            "day out of range caught",
            "bonono",
        ],
        "replace-keyword semantics diverged from CPython"
    );
}

#[test]
fn datetime_fields_and_strptime_directives_match_python_at_runtime() {
    // Flat attribute access (dt.year .. dt.microsecond), the dt.date()
    // and dt.time() methods, and the %a/%A/%j strptime directives,
    // through generated code.
    let scratch = Scratch::new("dtfields");
    let file = scratch.path().join("dtf_demo.py");
    fs::write(
        &file,
        concat!(
            "from datetime import datetime\n",
            "\n",
            "def main() -> None:\n",
            "    d = datetime(2024, 2, 29, 13, 5, 7, 123456)\n",
            "    print(d.year, d.month, d.day)\n",
            "    print(d.hour, d.minute, d.second, d.microsecond)\n",
            "    print(d.date())\n",
            "    print(d.time())\n",
            "    print(datetime.strptime(\"2024-060\", \"%Y-%j\"))\n",
            "    print(datetime.strptime(\"2023-366\", \"%Y-%j\"))\n",
            "    print(datetime.strptime(\"060\", \"%j\"))\n",
            "    print(datetime.strptime(\"Mon 2024-01-02\", \"%a %Y-%m-%d\"))\n",
            "    print(datetime.strptime(\"friday 2024-03-01\", \"%A %Y-%m-%d\"))\n",
            "    print(datetime.strptime(\"Tue 2024-060\", \"%a %Y-%j\"))\n",
            "    try:\n",
            "        print(datetime.strptime(\"Xyz 2024-01-02\", \"%a %Y-%m-%d\"))\n",
            "    except ValueError:\n",
            "        print(\"bad weekday caught\")\n",
            "    try:\n",
            "        print(datetime.strptime(\"2023-367\", \"%Y-%j\"))\n",
            "    except ValueError:\n",
            "        print(\"trailing digit caught\")\n",
            "    lo = datetime(2024, 2, 29, 13, 5, 7)\n",
            "    hi = datetime(2024, 2, 29, 14, 5, 7)\n",
            "    print(lo < hi, hi - lo)\n",
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

    let output = Command::new(krate.root.join("target/debug/dtf_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "2024 2 29",
            "13 5 7 123456",
            "2024-02-29",
            "13:05:07.123456",
            "2024-02-29 00:00:00",
            "2024-01-01 00:00:00",
            "1900-03-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-03-01 00:00:00",
            "2024-02-29 00:00:00",
            "bad weekday caught",
            "trailing digit caught",
            "True 1:00:00",
        ],
        "datetime field/strptime semantics diverged from CPython"
    );
}

#[test]
fn re_named_groups_and_findall_tuples_match_python_at_runtime() {
    // (?P<name>...) access by name and groupdict, findall returning
    // 2- and 3-tuples (chosen from the literal pattern at conversion
    // time), tuple unpacking over the result, and tuple printing.
    let scratch = Scratch::new("renamed");
    let file = scratch.path().join("renamed_demo.py");
    fs::write(
        &file,
        concat!(
            "import re\n",
            "\n",
            "def main() -> None:\n",
            "    m = re.search(r\"(?P<user>\\w+)@(?P<host>[\\w.]+)\", \"mail bob@example.com now\")\n",
            "    if m:\n",
            "        print(m.group(\"user\"), m.group(\"host\"))\n",
            "        print(m.group(0))\n",
            "        d = m.groupdict()\n",
            "        print(d[\"user\"], d[\"host\"])\n",
            "        print(m.span())\n",
            "    pairs = re.findall(r\"(\\w+)=(\\d+)\", \"a=1 b=22 c=333\")\n",
            "    print(pairs)\n",
            "    for k, v in pairs:\n",
            "        print(k, v)\n",
            "    trios = re.findall(r\"(\\d+)-(\\d+)-(\\d+)\", \"2024-01-05 and 1999-12-31\")\n",
            "    print(trios)\n",
            "    print(re.findall(r\"(a)|(b)\", \"ab\"))\n",
            "    print(re.findall(r\"(?P<k>\\w+):(?P<v>\\d+)\", \"x:1 y:2\", re.IGNORECASE))\n",
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

    let output = Command::new(krate.root.join("target/debug/renamed_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "bob example.com",
            "bob@example.com",
            "bob example.com",
            "(5, 20)",
            "[('a', '1'), ('b', '22'), ('c', '333')]",
            "a 1",
            "b 22",
            "c 333",
            "[('2024', '01', '05'), ('1999', '12', '31')]",
            "[('a', ''), ('', 'b')]",
            "[('x', '1'), ('y', '2')]",
        ],
        "re named-group semantics diverged from CPython"
    );
}

#[test]
fn print_and_list_sort_match_python_at_runtime() {
    // Multi-argument print (sep=/end=/flush=, mixed types, Python str
    // semantics for bools/floats/lists) and in-place list.sort in every
    // keyword shape, through generated code. Compared as RAW BYTES so
    // end="" behavior is pinned too.
    let scratch = Scratch::new("prints");
    let file = scratch.path().join("print_sort_demo.py");
    fs::write(
        &file,
        concat!(
            "def show(label: str, values: list[int]) -> None:\n",
            "    print(label, values)\n",
            "\n",
            "def main() -> None:\n",
            "    print(\"alpha\", 1, 2.5, True)\n",
            "    print(\"a\", \"b\", \"c\", sep=\"\")\n",
            "    print(\"x\", \"y\", sep=\" | \", end=\".\\n\")\n",
            "    print(\"no newline\", end=\"\")\n",
            "    print()\n",
            "    print(1, 2, 3, sep=\"-\", end=\"!\\n\")\n",
            "    print(\"flushed\", flush=True)\n",
            "    print(10000000000000000.0)\n",
            "    print(False, True)\n",
            "    print([1, 2, 3], [\"a\", \"b\"])\n",
            "    show(\"nums:\", [3, 1, 2])\n",
            "    xs = [3, 1, 2]\n",
            "    xs.sort()\n",
            "    print(xs)\n",
            "    xs.sort(reverse=True)\n",
            "    print(xs)\n",
            "    ys = [2.5, -1.0, 0.5]\n",
            "    ys.sort()\n",
            "    print(ys)\n",
            "    words = [\"pear\", \"fig\", \"banana\"]\n",
            "    words.sort(key=lambda w: len(w))\n",
            "    print(words)\n",
            "    words.sort(key=lambda w: len(w), reverse=True)\n",
            "    print(words)\n",
            "    grid = [[3, 1], [2, 0]]\n",
            "    grid[0].sort()\n",
            "    print(grid)\n",
            "    words.reverse()\n",
            "    print(words)\n",
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

    let output = Command::new(krate.root.join("target/debug/print_sort_demo"))
        .output()
        .expect("running generated binary");
    // Verified byte-for-byte against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        concat!(
            "alpha 1 2.5 True\n",
            "abc\n",
            "x | y.\n",
            "no newline\n",
            "1-2-3!\n",
            "flushed\n",
            "1e+16\n",
            "False True\n",
            "[1, 2, 3] ['a', 'b']\n",
            "nums: [3, 1, 2]\n",
            "[1, 2, 3]\n",
            "[3, 2, 1]\n",
            "[-1.0, 0.5, 2.5]\n",
            "['fig', 'pear', 'banana']\n",
            "['banana', 'pear', 'fig']\n",
            "[[1, 3], [2, 0]]\n",
            "['fig', 'pear', 'banana']\n",
        ),
        "print/sort semantics diverged from CPython"
    );
}

#[test]
fn csv_reader_matches_python_at_runtime() {
    // The excel dialect over a list of lines: quoted delimiters, ""
    // escapes, empty records, and int() over parsed fields, through
    // generated code.
    let scratch = Scratch::new("csvs");
    let file = scratch.path().join("csv_demo.py");
    fs::write(
        &file,
        concat!(
            "import csv\n",
            "\n",
            "def main() -> int:\n",
            "    lines = [\"name,qty,note\", \"apple,3,\\\"sweet, crisp\\\"\", \"pear,7,\\\"say \\\"\\\"hi\\\"\\\"\\\"\", \"\"]\n",
            "    for row in csv.reader(lines):\n",
            "        print(f\"row={repr(row)}\")\n",
            "    total = 0\n",
            "    for row in csv.reader([\"1,2\", \"3,4\"]):\n",
            "        total += int(row[0]) + int(row[1])\n",
            "    print(f\"total={total}\")\n",
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

    let output = Command::new(krate.root.join("target/debug/csv_demo"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "row=['name', 'qty', 'note']",
            "row=['apple', '3', 'sweet, crisp']",
            "row=['pear', '7', 'say \"hi\"']",
            "row=[]",
            "total=10",
        ],
        "csv semantics diverged from CPython"
    );
}
#[test]
fn driver_mode_generates_complete_driver_crate() {
    // `--driver` must emit the whole userspace driver: the compiled Python
    // logic (lib.rs) plus generated syscall glue (main.rs) parameterized by
    // the Python-declared device manifest.
    let scratch = Scratch::new("driver-mode");
    let file = scratch.path().join("driver.py");
    fs::write(
        &file,
        concat!(
            "__device_path__ = \"/dev/rython0\"\n",
            "__ioc_reset__ = 0x5201\n",
            "__ioc_stats__ = 0x80285202\n",
            "\n",
            "class Device:\n",
            "    def __init__(self, regs: dict[int, int], name: str):\n",
            "        self.regs = regs\n",
            "        self.name = name\n",
            "        self.ops = 0\n",
            "\n",
            "    def handle(self, line: str) -> str:\n",
            "        self.ops = self.ops + 1\n",
            "        return \"OK \" + str(self.ops)\n",
            "\n",
            "def parse_hex(s: str) -> int:\n",
            "    return 0\n",
            "\n",
            "def crc8(data: bytes) -> int:\n",
            "    return 0\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            driver: true,
            ..Default::default()
        },
    )
    .expect("driver crate converts");

    // Manifest constants parameterize the glue.
    let main = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(
        main.contains("use driver::{crc8, parse_hex, Device};"),
        "glue imports the compiled Python: {main}"
    );
    assert!(
        main.contains("const RYTHON_IOC_RESET: libc::c_ulong = 0x5201;"),
        "ioc_reset from manifest: {main}"
    );
    assert!(
        main.contains("const RYTHON_IOC_STATS: libc::c_ulong = 0x80285202;"),
        "ioc_stats from manifest: {main}"
    );
    assert!(
        main.contains("const DEVICE_PATH: &str = \"/dev/rython0\";"),
        "device path from manifest: {main}"
    );

    // The Python logic lands compiled in lib.rs, with public items.
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("pub struct Device"), "compiled logic: {lib}");

    // The crate depends on libc for the glue and has a binary target.
    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(toml.contains("libc = \"0.2\""), "libc dep: {toml}");
    assert!(fs::metadata(out.join("src/main.rs")).is_ok());

    // The whole generated crate must compile.
    let status = check_generated(&out);
    assert!(status.success(), "generated driver crate failed to compile");
}

#[test]
fn driver_mode_defaults_manifest_when_absent() {
    // A driver Python with no manifest still gets working glue, using the
    // documented default device (byte-ring misc device, /dev/rython0).
    let scratch = Scratch::new("driver-defaults");
    let file = scratch.path().join("thing.py");
    fs::write(&file, "def crc8(data: bytes) -> int:\n    return 0\n").unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            driver: true,
            ..Default::default()
        },
    )
    .expect("driver crate converts without a manifest");
    let main = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(
        main.contains("const DEVICE_PATH: &str = \"/dev/rython0\";"),
        "default device path: {main}"
    );
    assert!(main.contains("0x5201") && main.contains("0x80285202"));
}

#[test]
fn driver_mode_rejects_bad_manifest_types() {
    let scratch = Scratch::new("driver-bad-manifest");
    let file = scratch.path().join("driver.py");
    fs::write(&file, "__ioc_reset__ = \"not an int\"\n").unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            driver: true,
            ..Default::default()
        },
    )
    .expect_err("non-integer __ioc_reset__ must fail loudly");
    assert!(
        err.to_string().contains("__ioc_reset__ must be an integer literal"),
        "{}",
        err
    );
}

// ---------------------------------------------------------------------------
// kernel device generation (--kernel-module + device manifest)
// ---------------------------------------------------------------------------

/// A device-manifest Python like rython-kmod's driver.py: module metadata,
/// the device manifest, and pure user-space driver logic (no module entry
/// points — the generated device owns them).
const DEVICE_DRIVER_PY: &str = concat!(
    "__module_name__ = \"rython\"\n",
    "__module_license__ = \"GPL\"\n",
    "__module_author__ = \"rexlunae\"\n",
    "__module_description__ = \"rython-kmod: pure-Rust byte-ring device driven by rython-compiled driver logic\"\n",
    "__module_version__ = \"0.1.0\"\n",
    "\n",
    "__device_path__ = \"/dev/rython0\"\n",
    "__device_name__ = \"rython0\"\n",
    "__bufsz__ = 4096\n",
    "__magic__ = 0x52594854\n",
    "__device_mode__ = 0o600\n",
    "__ioc_reset__ = 0x5201\n",
    "__ioc_stats__ = 0x80285202\n",
    "\n",
    "def parse_hex(s: str) -> int:\n",
    "    return 0\n",
    "\n",
    "class Device:\n",
    "    def handle(self, line: str) -> str:\n",
    "        return str(\"OK\")\n",
);

#[test]
fn kernel_device_mode_generates_misc_device_crate() {
    let scratch = Scratch::new("kernel-device");
    let file = scratch.path().join("driver.py");
    fs::write(&file, DEVICE_DRIVER_PY).unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("device-manifest kernel module converts");
    assert!(!krate.has_binary, "kernel output is a library");

    // lib.rs: modinfo, mod device, entry points that register the device.
    let lib = fs::read_to_string(out.join("src/lib.rs")).unwrap();
    assert!(lib.contains("#![no_std]"), "lib.rs: {}", lib);
    assert!(lib.contains("mod device;"), "lib.rs: {}", lib);
    assert!(lib.contains("device::register()"), "lib.rs: {}", lib);
    assert!(lib.contains("device::deregister()"), "lib.rs: {}", lib);
    assert!(lib.contains("pub extern \"C\" fn init_module()"), "lib.rs: {}", lib);
    assert!(lib.contains("pub extern \"C\" fn cleanup_module()"), "lib.rs: {}", lib);
    // modinfo entries from the Python metadata, kept alive for ld --gc-sections.
    assert!(lib.contains("license=GPL"), "modinfo license: {}", lib);
    assert!(lib.contains("author=rexlunae"), "modinfo author: {}", lib);
    assert!(lib.contains("version=0.1.0"), "modinfo version: {}", lib);

    // device.rs: the misc device parameterized by the manifest.
    let dev = fs::read_to_string(out.join("src/device.rs")).unwrap();
    assert!(dev.contains("pub const BUFSZ: usize = 4096;"), "device.rs: {}", dev);
    assert!(dev.contains("pub const MAGIC: u32 = 0x52594854;"), "device.rs: {}", dev);
    assert!(dev.contains("const IOC_RESET: c_uint = 0x5201;"), "device.rs: {}", dev);
    assert!(dev.contains("const IOC_STATS: c_uint = 0x80285202;"), "device.rs: {}", dev);
    assert!(
        dev.contains("MiscDevice::new(b\"rython0\\0\", &FOPS, 0o600)"),
        "device name/mode: {}",
        dev
    );

    // Cargo.toml: crate name from __module_name__, rykernel-shim dep (the
    // device code links it), no stdpython, staticlib for the kernel link.
    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(toml.contains("name = \"rython\""), "Cargo.toml: {}", toml);
    assert!(toml.contains("rykernel-shim"), "Cargo.toml: {}", toml);
    assert!(!toml.contains("stdpython"), "no stdpython for a device: {}", toml);
    assert!(toml.contains("crate-type = [\"staticlib\"]"), "Cargo.toml: {}", toml);

    // The generated Makefile names the module from __module_name__.
    let makefile = fs::read_to_string(out.join("Makefile")).unwrap();
    assert!(makefile.contains("obj-m += rython.o"), "Makefile: {}", makefile);
    assert!(makefile.contains("-u init_module -u cleanup_module"), "Makefile: {}", makefile);

    // The proof: the generated device crate is genuine no_std Rust and
    // compiles (host check; the kernel link itself is exercised by the
    // rython-kmod repository's make module).
    let status = check_generated(&out);
    assert!(status.success(), "generated device crate failed to compile");
}

#[test]
fn kernel_device_mode_uses_manifest_defaults() {
    // Declaring only __device_name__ still generates a device: every other
    // manifest constant keeps its documented default.
    let scratch = Scratch::new("kernel-device-defaults");
    let file = scratch.path().join("driver.py");
    fs::write(
        &file,
        concat!(
            "__device_name__ = \"rython0\"\n",
            "\n",
            "def handle(line: str) -> str:\n",
            "    return str(\"OK\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect("device-name-only manifest converts");
    let dev = fs::read_to_string(out.join("src/device.rs")).unwrap();
    assert!(dev.contains("pub const BUFSZ: usize = 4096;"), "device.rs: {}", dev);
    assert!(dev.contains("pub const MAGIC: u32 = 0x52594854;"), "device.rs: {}", dev);
    assert!(dev.contains("const IOC_RESET: c_uint = 0x5201;"), "device.rs: {}", dev);
    assert!(dev.contains("const IOC_STATS: c_uint = 0x80285202;"), "device.rs: {}", dev);
}

#[test]
fn pyproject_dependencies_vendored_end_to_end_at_runtime() {
    // A pyproject.toml project declaring `dependencies`, satisfied by a
    // vendored rython.toml [python-modules] copy: the resolved dependency
    // is transpiled beside the package and callable from the generated
    // binary — the pip-style wiring, verified at runtime.
    let scratch = Scratch::new("pyproject-deps");
    fs::write(
        scratch.path().join("pyproject.toml"),
        concat!(
            "[project]\n",
            "name = \"depapp\"\n",
            "version = \"0.1.0\"\n",
            "dependencies = [\"pylev>=1.3\"]\n",
            "\n",
            "[tool.setuptools]\n",
            "packages = [\"depapp\"]\n",
        ),
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("depapp")).unwrap();
    fs::write(scratch.path().join("depapp/__init__.py"), "").unwrap();
    fs::write(
        scratch.path().join("depapp/main.py"),
        concat!(
            "import pylev\n",
            "\n",
            "def dist(a: str, b: str) -> int:\n",
            "    return pylev.wf_levenshtein(a, b)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(dist(\"kitten\", \"sitting\"))\n",
        ),
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(
        scratch.path().join("vendor/pylev.py"),
        concat!(
            "def wf_levenshtein(a: str, b: str) -> int:\n",
            "    n = len(a)\n",
            "    m = len(b)\n",
            "    if n == 0:\n",
            "        return m\n",
            "    if m == 0:\n",
            "        return n\n",
            "    prev = [0] * (m + 1)\n",
            "    for j in range(m + 1):\n",
            "        prev[j] = j\n",
            "    for i in range(1, n + 1):\n",
            "        cur = [0] * (m + 1)\n",
            "        cur[0] = i\n",
            "        for j in range(1, m + 1):\n",
            "            cost = 0 if a[i - 1] == b[j - 1] else 1\n",
            "            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)\n",
            "        prev = cur\n",
            "    return prev[m]\n",
        ),
    )
    .unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\npylev = { path = \"vendor/pylev.py\" }\n",
    )
    .unwrap();

    let out = scratch.path().join("crate");
    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/depapp"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "3", "stdout: {}", stdout);
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn async_binary_builds_and_runs_on_the_tokio_runtime() {
    // Python async/await end-to-end: `async def`/`await` transpile to Rust
    // async fns, asyncio.sleep maps onto tokio's timer, asyncio.run drives
    // the coroutine, and the generated BINARY crate declares tokio behind a
    // default-on `async-tokio` feature (the entry carries the feature-gated
    // #[tokio::main] attribute).
    let scratch = Scratch::new("async-bin");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "import asyncio\n",
            "\n",
            "async def fetch(name: str) -> str:\n",
            "    await asyncio.sleep(0.001)\n",
            "    return \"hello \" + name\n",
            "\n",
            "async def main() -> None:\n",
            "    result = await fetch(\"world\")\n",
            "    print(result)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    asyncio.run(main())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    // The generated binary crate declares tokio behind the default-on
    // feature and stdpython with the tokio-backed asyncio module.
    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(toml.contains("tokio"), "Cargo.toml: {}", toml);
    assert!(toml.contains("async-tokio"), "Cargo.toml: {}", toml);
    assert!(toml.contains("default = [\"async-tokio\"]"), "Cargo.toml: {}", toml);
    assert!(
        toml.contains("features = [\"std\", \"async-tokio\"]"),
        "Cargo.toml: {}",
        toml
    );
    // The entry module's code is feature-gated.
    let main = fs::read_to_string(out.join("src/main.rs")).unwrap();
    assert!(
        main.contains("cfg_attr(feature = \"async-tokio\", tokio::main)"),
        "main.rs: {}",
        main
    );
    assert!(
        main.contains("compile_error!"),
        "feature-off build must name the fix: {}",
        main
    );

    let status = build_generated(&krate.root);
    assert!(status.success(), "async binary failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "hello world", "stdout: {}", stdout);
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn async_binary_without_feature_fails_with_compile_error() {
    // --no-default-features drops tokio; the generated entry's compile_error
    // names the fix instead of a bare "no main function".
    let scratch = Scratch::new("async-nofeature");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "import asyncio\n",
            "async def main() -> None:\n",
            "    print(\"hi\")\n",
            "if __name__ == \"__main__\":\n",
            "    asyncio.run(main())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--no-default-features")
        .env_remove("RUSTFLAGS")
        .current_dir(&krate.root)
        .output()
        .expect("cargo build");
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("async-tokio"),
        "compile_error must name the feature: {}",
        stderr
    );
    assert!(!status.status.success(), "feature-off build must fail");
}

#[test]
fn async_library_conversion_gets_no_runtime_dependency() {
    // A library crate with async functions transpiles them to plain async
    // fns; the generated Cargo.toml has NO tokio dependency and no feature
    // (the consumer brings the executor), and the code has no runtime
    // import or entry attribute.
    let scratch = Scratch::new("async-lib");
    let file = scratch.path().join("libmod.py");
    fs::write(
        &file,
        "async def compute(x: int) -> int:\n    return x * 2\n",
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let toml = fs::read_to_string(out.join("Cargo.toml")).unwrap();
    assert!(
        !toml.contains("tokio"),
        "library conversions must not link tokio: {}",
        toml
    );
    let lib = fs::read_to_string(out.join("src/libmod.rs")).unwrap();
    assert!(lib.contains("pub async fn compute"), "libmod.rs: {}", lib);
    assert!(!lib.contains("use tokio"), "no runtime import: {}", lib);
    assert!(!lib.contains("compile_error"), "no feature error: {}", lib);
    assert!(!lib.contains("cfg_attr"), "no entry attribute: {}", lib);

    // The library builds (async fns are just async fns).
    let status = build_generated(&krate.root);
    assert!(status.success(), "async library failed to compile");
}

#[test]
fn kernel_device_mode_rejects_module_init_conflict() {
    // The generated device owns the entry points; a Python module_init
    // would be silently dropped — loud error instead.
    let scratch = Scratch::new("kernel-device-conflict");
    let file = scratch.path().join("driver.py");
    fs::write(
        &file,
        concat!(
            "__device_name__ = \"rython0\"\n",
            "\n",
            "def module_init() -> int:\n",
            "    return 0\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect_err("module_init + device manifest must fail loudly");
    assert!(
        err.to_string().contains("conflicts with the generated device"),
        "{}",
        err
    );
}

#[test]
fn kernel_device_mode_rejects_rust_for_linux() {
    // Device generation targets the raw-FFI pipeline; rust-for-linux has its
    // own module! machinery and cannot host the generated misc device.
    let scratch = Scratch::new("kernel-device-rfl");
    let file = scratch.path().join("driver.py");
    fs::write(&file, "__device_name__ = \"rython0\"\n").unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            rust_for_linux: true,
            ..Default::default()
        },
    )
    .expect_err("device + rust-for-linux must fail loudly");
    assert!(
        err.to_string().contains("not supported with --rust-for-linux"),
        "{}",
        err
    );
}

#[test]
fn kernel_device_mode_rejects_bad_device_name() {
    // The device name is embedded in a Rust byte-string literal; keep it to
    // a safe charset so a weird Python string cannot break the output.
    let scratch = Scratch::new("kernel-device-badname");
    let file = scratch.path().join("driver.py");
    fs::write(&file, "__device_name__ = \"foo/bar\"\n").unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect_err("a device name outside the safe charset must fail loudly");
    assert!(
        err.to_string().contains("__device_name__ must be alphanumeric"),
        "{}",
        err
    );
}

#[test]
fn kernel_device_mode_rejects_floating_point_loudly() {
    // Issue #108: the device-manifest sub-mode used to return before the
    // module-wide float scan (issue #87's FPU lazy-save guard), leaving the
    // one path where user Python grows — ioctl handlers, buffer logic —
    // unguarded. Every FP shape must fail the conversion exactly as in the
    // plain kernel path.
    let scratch = Scratch::new("kernel-device-fp");
    let cases: &[(&str, &str)] = &[
        (
            "device_float_return.py",
            concat!(
                "__device_name__ = \"rython0\"\n",
                "\n",
                "def module_init() -> int:\n",
                "    return 1\n",
                "\n",
                "def handle(line: str) -> str:\n",
                "    return str(1.5)\n",
            ),
        ),
        (
            "device_float_assign.py",
            concat!(
                "__device_name__ = \"rython0\"\n",
                "\n",
                "def module_init() -> int:\n",
                "    ratio = 2.5\n",
                "    return 0\n",
            ),
        ),
        (
            "device_float_call.py",
            concat!(
                "__device_name__ = \"rython0\"\n",
                "\n",
                "def module_init() -> int:\n",
                "    x = float(\"1.5\")\n",
                "    return 0\n",
            ),
        ),
        (
            "device_import_math.py",
            concat!(
                "__device_name__ = \"rython0\"\n",
                "\n",
                "import math\n",
                "\n",
                "def module_init() -> int:\n",
                "    return 0\n",
            ),
        ),
    ];
    for (name, src) in cases {
        let file = scratch.path().join(name);
        fs::write(&file, src).unwrap();
        let out = scratch.path().join(format!("crate-{}", name.replace('.', "-")));
        let pkg = rypip::discover(&file).expect("discover");
        let err = rypip::convert(
            &pkg,
            &out,
            &ConvertOptions {
                kernel_module: true,
                ..Default::default()
            },
        )
        .expect_err("floating-point device-manifest code must fail the conversion");
        let msg = format!("{:#}", err);
        assert!(msg.contains("floating-point"), "{}: {}", name, msg);
        assert!(
            msg.contains("kernel_fpu_begin"),
            "{}: error must mention the FPU guard workaround: {}",
            name,
            msg
        );
    }
}

#[test]
fn unannotated_params_infer_generics_matching_python_transcript() {
    // Issue #109, M1 acceptance: `def add(a, b): return a + b` becomes ONE
    // generic function (not the dead impl Into<PyObject>) that monomorphizes
    // per call site; the compiled binary's output is diffed against a
    // pinned `// Verified against python3.` transcript.
    let scratch = Scratch::new("param-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def add(a, b):\n",
            "    return a + b\n",
            "\n",
            "def to_int(x):\n",
            "    return int(x)\n",
            "\n",
            "def positive(n):\n",
            "    return n > 0\n",
            "\n",
            "def describe(x):\n",
            "    return \"val=\" + str(x)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(add(1, 2))\n",
            "    print(add(1.5, 2.5))\n",
            "    print(add(\"ab\", \"cd\"))\n",
            "    print(add([1], [2]))\n",
            "    print(to_int(\"42\"))\n",
            "    print(positive(3))\n",
            "    print(positive(-1))\n",
            "    print(describe(7))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");

    // The generated source uses trait-bound generics, never the dead
    // fallback.
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(
        src.contains("pub fn add<A, B>(a: A, b: B)"),
        "generic signature: {}",
        src
    );
    assert!(src.contains("A: PyAdd<B>"), "bound: {}", src);
    assert!(
        src.contains("<A as PyAdd<B>>::Output"),
        "associated return: {}",
        src
    );
    assert!(!src.contains("Into < PyObject >"), "no dead fallback: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "3",
            "4.0",
            "abcd",
            "[1, 2]",
            "42",
            "True",
            "False",
            "val=7",
        ],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn stdlib_method_inference_matches_python_transcript() {
    // Issue #109, M2: method calls on unannotated parameters infer the
    // stdlib trait bound (PyStrOps/PyPop), the String receiver satisfies
    // it via the owned-type impl, and the compiled binary's output diffs
    // against a pinned `// Verified against python3.` transcript.
    let scratch = Scratch::new("method-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def shout(s):\n",
            "    return s.upper()\n",
            "\n",
            "def words(s):\n",
            "    return s.split(\" \")\n",
            "\n",
            "def position(s):\n",
            "    return s.find(\"x\")\n",
            "\n",
            "def last(xs):\n",
            "    return xs.pop()\n",
            "\n",
            "def count_letters(s):\n",
            "    return s.count(\"a\")\n",
            "\n",
            "def strip_it(s):\n",
            "    return s.strip()\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(shout(\"hi\"))\n",
            "    print(words(\"a b c\"))\n",
            "    print(position(\"xyz\"))\n",
            "    print(last([1, 2, 3]))\n",
            "    print(count_letters(\"banana\"))\n",
            "    print(strip_it(\"  x  \"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(src.contains("T: PyStrOps"), "bound: {}", src);
    assert!(src.contains("T: PyPop<i64>"), "bound: {}", src);
    assert!(!src.contains("Into < PyObject >"), "no dead fallback: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["HI", "['a', 'b', 'c']", "0", "3", "3", "x"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn duck_typing_hear_example_matches_python_transcript() {
    // Issue #109, M3 acceptance: `def hear(animal): return animal.speak()`
    // becomes `fn hear<T: HasSpeak>(animal: T)`, one impl per defining
    // class, and the compiled binary's output diffs against a pinned
    // `// Verified against python3.` transcript.
    let scratch = Scratch::new("duck-hear");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "class Dog:\n",
            "    def speak(self) -> str:\n",
            "        return \"woof\"\n",
            "\n",
            "class Cat:\n",
            "    def speak(self) -> str:\n",
            "        return \"meow\"\n",
            "\n",
            "def hear(animal):\n",
            "    return animal.speak()\n",
            "\n",
            "def praise(animal):\n",
            "    return \"nice \" + animal.speak()\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(hear(Dog()))\n",
            "    print(hear(Cat()))\n",
            "    print(praise(Dog()))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(src.contains("pub trait HasSpeak"), "generated: {}", src);
    assert!(src.contains("impl HasSpeak for Dog"), "generated: {}", src);
    assert!(src.contains("impl HasSpeak for Cat"), "generated: {}", src);
    assert!(
        src.contains("pub fn hear<T>(animal: T)"),
        "generic param: {}",
        src
    );
    assert!(src.contains("T: HasSpeak"), "generic bound: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["woof", "meow", "nice woof"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn recursion_and_flows_to_inference_match_python_transcript() {
    // Issue #109, M4 acceptance: interprocedural FlowsTo. `repeat` is
    // SELF-recursive — the fixpoint gives `x` the recursion's type
    // (`A: PyAdd<A, Output = A>`) and `n` a generic that accepts both int
    // and float calls (`B: PyLe<B, Output = bool> + PyFromInt +
    // PySub<i64, Output = B>`, per the issue's `repeat(x, 2.5)`). The
    // 2-deep helper chain flows the callee's return type through the
    // caller. Output is diffed against a pinned `// Verified against
    // python3.` transcript.
    let scratch = Scratch::new("recursion-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def repeat(x, n):\n",
            "    return x if n <= 0 else x + repeat(x, n - 1)\n",
            "\n",
            "def helper(x):\n",
            "    return x * 2\n",
            "\n",
            "def caller(v):\n",
            "    return helper(v)\n",
            "\n",
            "def positive(n):\n",
            "    return n > 0\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(repeat(2, 3))\n",
            "    print(repeat(2.5, 3))\n",
            "    print(repeat(\"a\", 3))\n",
            "    print(repeat(2, 2.5))\n",
            "    print(repeat(\"a\", 2.5))\n",
            "    print(caller(21))\n",
            "    print(caller(1.5))\n",
            "    print(positive(3))\n",
            "    print(positive(-1))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    // The recursive fixpoint: x adds with itself (the recursion returns
    // x's type) — `A: PyAdd<A, Output = A>`.
    assert!(src.contains("A: PyAdd<A, Output = A>"), "repeat bounds: {}", src);
    // The count accepts int AND float call sites: comparison with the
    // literal converts via PyFromInt, and the decrement's Output must be
    // the parameter again.
    assert!(src.contains("B: PyLe<B, Output = bool>"), "repeat bounds: {}", src);
    assert!(src.contains("B: PyFromInt"), "repeat bounds: {}", src);
    assert!(src.contains("B: PySub<i64, Output = B>"), "repeat bounds: {}", src);
    assert!(
        src.contains("pub fn repeat<A, B>(x: A, n: B)"),
        "repeat signature: {}",
        src
    );
    // The 2-deep helper chain flows the callee's return through the
    // caller: `caller` returns `<T as PyMul<i64>>::Output` like `helper`.
    assert!(
        src.contains("pub fn caller<T>(v: T) -> Result<<T as PyMul<i64>>::Output"),
        "caller signature: {}",
        src
    );
    assert!(!src.contains("Into < PyObject >"), "no dead fallback: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "8", "10.0", "aaaa", "8", "aaaa", "42", "3.0", "True", "False",
        ],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn iteration_inference_matches_python_transcript() {
    // Issue #109, M2 iteration: `for w in words` over an unannotated
    // parameter bounds it `A: IntoIterator<Item = B>` and threads the
    // element type into the loop variable (`B: PyStrOps` / `B: Len`); a
    // caller passing its own parameter adopts both the Iterate bound and
    // the element requirements. Output is diffed against a pinned
    // `// Verified against python3.` transcript.
    let scratch = Scratch::new("iteration-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def shout_all(words):\n",
            "    result: list[str] = []\n",
            "    for w in words:\n",
            "        result.append(w.upper())\n",
            "    return result\n",
            "\n",
            "def total_len(words):\n",
            "    n = 0\n",
            "    for w in words:\n",
            "        n += len(w)\n",
            "    return n\n",
            "\n",
            "def caller(v):\n",
            "    return shout_all(v)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(shout_all([\"hi\", \"there\"]))\n",
            "    print(total_len([\"ab\", \"cde\"]))\n",
            "    print(caller([\"x\", \"y\"]))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(
        src.contains("A: IntoIterator<Item = B>"),
        "iteration bound: {}",
        src
    );
    assert!(src.contains("B: PyStrOps"), "element bound: {}", src);
    assert!(src.contains("B: Len"), "element bound: {}", src);
    // The caller adopts both the iterable and the element bounds.
    assert!(
        src.contains("pub fn caller<A, B>(v: A)"),
        "caller signature: {}",
        src
    );
    assert!(!src.contains("Into < PyObject >"), "no dead fallback: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["['HI', 'THERE']", "5", "['X', 'Y']"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn stdlib_method_table_honesty_transcript() {
    // Issue #109, M2: the STDLIB_METHOD_TABLE can never drift from the
    // runtime — every exerciseable row (one function per entry, bound on
    // the table's trait) converts, builds, and diffs against a pinned
    // `// Verified against python3.` transcript. The two rows skipped are
    // documented gaps, not coverage holes: `insert` (PyListOps<T> needs
    // the element type the inference does not express yet) and the LIST
    // `count` (the name is dual-str/list; the first table match wins, so
    // the str row below is the exercised one).
    let scratch = Scratch::new("method-table-honesty");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def f_upper(s): return s.upper()\n",
            "def f_lower(s): return s.lower()\n",
            "def f_strip(s): return s.strip()\n",
            "def f_lstrip(s): return s.lstrip()\n",
            "def f_rstrip(s): return s.rstrip()\n",
            "def f_capitalize(s): return s.capitalize()\n",
            "def f_title(s): return s.title()\n",
            "def f_splitlines(s): return s.splitlines()\n",
            "def f_find(s): return s.find(\"x\")\n",
            "def f_count(s): return s.count(\"a\")\n",
            "def f_split(s): return s.split(\" \")\n",
            "def f_rsplit(s): return s.rsplit(\" \")\n",
            "def f_partition(s): return s.partition(\" \")\n",
            "def f_rpartition(s): return s.rpartition(\" \")\n",
            "def f_zfill(s): return s.zfill(5)\n",
            "def f_ljust(s): return s.ljust(5)\n",
            "def f_rjust(s): return s.rjust(5)\n",
            "def f_pop(xs): return xs.pop()\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(f_upper(\"Hi\"))\n",
            "    print(f_lower(\"Hi\"))\n",
            "    print(f_strip(\"  x  \"))\n",
            "    print(f_lstrip(\"  x\"))\n",
            "    print(f_rstrip(\"x  \"))\n",
            "    print(f_capitalize(\"hi\"))\n",
            "    print(f_title(\"hello world\"))\n",
            "    print(f_splitlines(\"a\\nb\"))\n",
            "    print(f_find(\"xyz\"))\n",
            "    print(f_count(\"banana\"))\n",
            "    print(f_split(\"a b c\"))\n",
            "    print(f_rsplit(\"a b c\"))\n",
            "    print(f_partition(\"a b c\"))\n",
            "    print(f_rpartition(\"a b c\"))\n",
            "    print(f_zfill(\"42\"))\n",
            "    print(f_ljust(\"ab\"))\n",
            "    print(f_rjust(\"ab\"))\n",
            "    print(f_pop([\"a\", \"b\", \"c\"]))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    // Every str row bounds on PyStrOps; pop bounds on PyPop<i64>.
    assert!(src.matches("T: PyStrOps").count() >= 17, "bounds: {}", src);
    assert!(src.contains("T: PyPop<i64>"), "pop bound: {}", src);
    assert!(!src.contains("Into < PyObject >"), "no dead fallback: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "HI", "hi", "x", "x", "x", "Hi", "Hello World", "['a', 'b']", "0", "3",
            "['a', 'b', 'c']", "['a', 'b', 'c']", "('a', ' ', 'b c')", "('a b', ' ', 'c')",
            "00042", "ab   ", "   ab", "c",
        ],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn definition_time_unsatisfiable_bounds_report_and_deny() {
    // Issue #109, M5: a parameter whose inferred bound set no known type
    // satisfies (`p.upper()` + `p.pop()` → PyStrOps + PyPop) is a
    // well-formed Python definition — it converts with a warning at -W
    // warn, and -W deny promotes it to a conversion failure.
    let scratch = Scratch::new("def-unsat");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def bad(p):\n",
            "    p.upper()\n",
            "    p.pop()\n",
            "\n",
            "def good(s):\n",
            "    return s.upper()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");

    // -W warn (default): converts; the generated fn carries the
    // #[deprecated] note naming the contradiction.
    let _krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(
        src.contains("satisfied by no known rython type"),
        "deprecated note: {}",
        src
    );
    let good_fn = src.split("pub fn good").nth(1).unwrap_or_default();
    assert!(
        !good_fn.contains("satisfied by no known"),
        "good() carries a spurious note: {}",
        good_fn
    );

    // -W deny: the warning becomes a conversion error.
    let out2 = scratch.path().join("crate-deny");
    let err = rypip::convert(
        &pkg,
        &out2,
        &ConvertOptions {
            warnings: rypip::convert::WarningMode::Deny,
            ..Default::default()
        },
    )
    .expect_err("deny must fail on the definition-time warning");
    let msg = format!("{}", err);
    assert!(msg.contains("bad"), "error should name the function: {}", msg);
    assert!(msg.contains("PyStrOps"), "error should list bounds: {}", msg);
}

#[test]
fn join_and_comprehension_inference_match_python_transcript() {
    // Issue #116 (the pip version_str pattern): `",".join(parts)` and
    // `".".join(str(v) for v in version)` infer String returns with
    // IntoIterator/AsRef<str> bounds, and list comprehensions over
    // parameters infer Vec returns. Output diffed against a pinned
    // `// Verified against python3.` transcript.
    let scratch = Scratch::new("join-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def join_all(parts):\n",
            "    return \",\".join(parts)\n",
            "\n",
            "def version_str(version):\n",
            "    return \".\".join(str(v) for v in version)\n",
            "\n",
            "def upper_all(words):\n",
            "    return [w.upper() for w in words]\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(join_all([\"a\", \"b\", \"c\"]))\n",
            "    print(version_str([1, 2, 3]))\n",
            "    print(upper_all([\"hi\", \"there\"]))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let src = fs::read_to_string(out.join("src/app.rs")).unwrap();
    assert!(src.contains("B: AsRef<str>"), "join bound: {}", src);
    assert!(src.contains("Result<String, PyException>"), "return: {}", src);

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["a,b,c", "1.2.3", "['HI', 'THERE']"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn string_aug_assign_accumulation_matches_python_transcript() {
    // Issue #110: `out = ""; out += ...` — the string-literal binding is
    // owned so the String rebind compiles; the accumulated value diffs
    // against a pinned `// Verified against python3.` transcript.
    let scratch = Scratch::new("str-aug");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def accumulate():\n",
            "    out = \"\"\n",
            "    for i in range(3):\n",
            "        out += str(i)\n",
            "    return out\n",
            "\n",
            "def rebind():\n",
            "    s = \"a\"\n",
            "    s = s + \"b\"\n",
            "    return s\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(accumulate())\n",
            "    print(rebind())\n",
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
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["012", "ab"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn option_aug_assign_and_option_slot_store_match_python() {
    // Round 39: `-=`/`|=` on `int | None` targets (urllib3's
    // `self.chunk_left -= ...`, `options |= ...`) operate on the INNER
    // value through the runtime py_sub / plain `|`, with a loud §12.2
    // panic on None (CPython's TypeError); a plain value stored into an
    // Option-typed field (`self._start_connect = time.monotonic()`,
    // `self.chunk_left = self.chunk_left - amt`) wraps in Some, and an
    // Option-typed RHS of `-` unwraps. Transcript pinned against python3.
    let scratch = Scratch::new("opt-aug");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def chunk_ops(length_remaining: int | None, amt: int | None):\n",
            "    if length_remaining is not None and amt is not None:\n",
            "        length_remaining -= amt\n",
            "    return length_remaining\n",
            "\n",
            "class SSLBuilder:\n",
            "    def __init__(self):\n",
            "        self.options: int | None = None\n",
            "    def build(self, options: int | None) -> int | None:\n",
            "        if options is None:\n",
            "            options = 0\n",
            "            options |= 2\n",
            "        self.options = options\n",
            "        return self.options\n",
            "\n",
            "class Counter:\n",
            "    def __init__(self):\n",
            "        self.count: int | None = None\n",
            "    def dec(self, n: int):\n",
            "        if self.count is not None:\n",
            "            self.count -= n\n",
            "    def store_plain(self, v: float):\n",
            "        self._start = v\n",
            "        return self._start\n",
            "    def sub_r(self, amt: int | None):\n",
            "        if self.count is not None and amt is not None:\n",
            "            self.count = self.count - amt\n",
            "        return self.count\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(chunk_ops(10, 3))\n",
            "    print(chunk_ops(None, 3))\n",
            "    print(chunk_ops(10, None))\n",
            "    b = SSLBuilder()\n",
            "    print(b.build(None))\n",
            "    print(b.build(4))\n",
            "    c = Counter()\n",
            "    c.dec(4)\n",
            "    print(c.count)\n",
            "    c.store_plain(2.5)\n",
            "    print(c._start)\n",
            "    c.count = 10\n",
            "    print(c.sub_r(2))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["7", "None", "10", "2", "4", "None", "2.5", "8"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn boxed_union_param_accepts_none_like_python() {
    // Round 40: a `int | str | None` parameter resolves to the boxed
    // PyValue — the box absorbs None, so plain stores must go through
    // PyValue::from, never the Option-slot Some wrap (urllib3's
    // `cert_reqs = resolve_cert_reqs(None)` was Some-wrapping and the
    // generated crate failed to build). The syntactic optional-ness
    // (`is_optional_annotation`) must not override the resolved type.
    // Transcript pinned against python3.
    let scratch = Scratch::new("boxedparam");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def resolve(v):\n",
            "    return v\n",
            "\n",
            "def set_cert(cert_reqs: int | str | None = None):\n",
            "    if cert_reqs is None:\n",
            "        cert_reqs = resolve(None)\n",
            "    return cert_reqs\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(set_cert(None))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["None"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn hierarchy_trait_display_bound_allows_self_in_messages() {
    // Round 41: a trait-DEFAULT body that formats `self` in an exception
    // message (`raise PoolError(self)` — urllib3's _get_conn raises
    // ClosedPoolError(self)) lowers through py_display, which needs
    // `Self: PyDisplay`. The generated hierarchy trait now declares the
    // bound (every implementor carries the round-34 PyDisplay impl).
    // Transcript pinned against python3.
    let scratch = Scratch::new("dispbound");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "class PoolError(Exception):\n",
            "    pass\n",
            "\n",
            "class Base:\n",
            "    def __init__(self):\n",
            "        self.x = 1\n",
            "\n",
            "class Pool(Base):\n",
            "    def __str__(self) -> str:\n",
            "        return \"Pool<%d>\" % self.x\n",
            "    def _get(self):\n",
            "        raise PoolError(self)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    p = Pool()\n",
            "    print(str(p))\n",
            "    try:\n",
            "        p._get()\n",
            "    except Exception as e:\n",
            "        print(e)\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["Pool<1>", "Pool<1>"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn option_and_or_fold_returns_operands_like_python() {
    // Round 43: `ca and s` / `ca or "http"` where ca is `str | None`
    // (urllib3's `ca_certs and expanduser(ca_certs)`, `scheme or
    // "http"`) — the operand-returning fold uses the Option arm even when
    // the second operand's type is unknown (a call) or a string literal
    // (which is OWNED at the wrap). Transcript pinned against python3.
    let scratch = Scratch::new("fold");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def pick(ca: str | None, s: str) -> str | None:\n",
            "    return ca and s\n",
            "\n",
            "def pick_or(ca: str | None) -> str | None:\n",
            "    return ca or \"http\"\n",
            "\n",
            "def scheme_or(ca: str | None, s: str) -> str | None:\n",
            "    return ca or s\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(pick(\"x\", \"y\"))\n",
            "    print(pick(None, \"y\"))\n",
            "    print(pick_or(\"x\"))\n",
            "    print(pick_or(None))\n",
            "    print(scheme_or(\"x\", \"y\"))\n",
            "    print(scheme_or(None, \"y\"))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["y", "None", "x", "http", "x", "y"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn option_comparison_matches_python() {
    // Round 43: `amt != 0` / `amt == 0` / `amt < cl` with `int | None`
    // operands (urllib3's `amt != 0`, `amt < self.chunk_left`): the
    // Option LHS unwraps the inner for the comparison; a None LHS
    // answers Python's EQUALITY semantics (`None == x` is False, `None
    // != x` is True) while ordered compares on None are a loud §12.2
    // panic. Transcript pinned against python3.
    let scratch = Scratch::new("optcmp");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def ne(amt: int | None) -> bool:\n",
            "    return amt != 0\n",
            "\n",
            "def eq(amt: int | None) -> bool:\n",
            "    return amt == 0\n",
            "\n",
            "def both(amt: int | None, cl: int | None) -> bool:\n",
            "    if amt is not None and cl is not None:\n",
            "        return amt < cl\n",
            "    return False\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(ne(None))\n",
            "    print(ne(0))\n",
            "    print(ne(3))\n",
            "    print(eq(None))\n",
            "    print(eq(0))\n",
            "    print(both(3, 5))\n",
            "    print(both(None, 5))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["True", "False", "True", "False", "True", "True", "False"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn typing_any_dict_return_types_like_python() {
    // Round 44: `dict[str, typing.Any]` return annotations (urllib3's
    // `_merge_pool_kwargs`) — `typing.Any` maps to the boxed PyValue, so
    // the method's signature is `Result<PyDict<String, PyValue>>` instead
    // of collapsing to unit while the body emits `Ok(dict)` (which cannot
    // compile). Transcript pinned against python3.
    let scratch = Scratch::new("typany");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "from typing import Any\n",
            "\n",
            "def merge(override: dict[str, Any] | None) -> dict[str, Any]:\n",
            "    base: dict[str, Any] = {\"a\": 1}\n",
            "    if override:\n",
            "        for k, v in override.items():\n",
            "            base[k] = v\n",
            "    return base\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    d: dict[str, Any] = {\"b\": 2}\n",
            "    print(merge(d))\n",
            "    print(merge(None))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["{'a': 1, 'b': 2}", "{'a': 1}"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn option_param_local_stores_match_python() {
    // Round 45: a local assigned from an OPTION-typed parameter
    // (`release_this_conn = release_conn` where the param is `bool |
    // None` — urllib3's urlopen) is itself an Option binding: a later
    // plain store (`= False`) Some-wraps, so the binding stays
    // Option<bool> and the generated crate typechecks. Transcript pinned
    // against python3.
    let scratch = Scratch::new("optlocal");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def f(release_conn: bool | None = None) -> bool:\n",
            "    release_this_conn = release_conn\n",
            "    if release_this_conn is None:\n",
            "        release_this_conn = False\n",
            "    return not (release_this_conn is None)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(f(None))\n",
            "    print(f(True))\n",
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
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["True", "True"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn del_statement_matches_python_transcript() {
    // Issue #112: `del xs[i]` (list, incl. negative index) and `del d["k"]`
    // (string-keyed dict) lower through py_pop and diff against a pinned
    // `// Verified against python3.` transcript.
    let scratch = Scratch::new("del-infer");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "def remove_second(xs):\n",
            "    del xs[1]\n",
            "    return xs\n",
            "\n",
            "def remove_negative(xs):\n",
            "    del xs[-1]\n",
            "    return xs\n",
            "\n",
            "def remove_key(d):\n",
            "    del d[\"b\"]\n",
            "    return d\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(remove_second([\"a\", \"b\", \"c\"]))\n",
            "    print(remove_negative([1, 2, 3]))\n",
            "    print(remove_key({\"a\": 1, \"b\": 2, \"c\": 3}))\n",
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
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["['a', 'c']", "[1, 2]", "{'a': 1, 'c': 3}"],
        "stdout: {}",
        stdout
    );
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn warnings_module_matches_python_transcript() {
    // Issue #111: `warnings.warn(...)` and `warnings.simplefilter(..., 
    // append=True)` (keyword args to a runtime fn) convert, build, and the
    // stdout diffs against a pinned `// Verified against python3.`
    // transcript (warnings go to stderr in both).
    let scratch = Scratch::new("warnings");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "import warnings\n",
            "\n",
            "def check(x):\n",
            "    if x < 0:\n",
            "        warnings.warn(\"negative value\")\n",
            "    return x\n",
            "\n",
            "def setup():\n",
            "    warnings.simplefilter(\"ignore\")\n",
            "    warnings.simplefilter(\"default\", append=True)\n",
            "    warnings.warn(\"hello\", stacklevel=2)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    setup()\n",
            "    print(check(5))\n",
            "    print(check(-1))\n",
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
    // // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["5", "-1"],
        "stdout: {}",
        stdout
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("negative value"), "stderr: {}", stderr);
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn threading_semantics_match_python_at_runtime() {
    // Thread lifecycle (start/join/is_alive), with-lock, RLock reentrancy,
    // Event, Semaphore, active_count — sequenced by join() so the output
    // is deterministic.
    let scratch = Scratch::new("threads");
    let file = scratch.path().join("threads.py");
    fs::write(
        &file,
        concat!(
            "import threading\n",
            "import time\n",
            "\n",
            "def worker(name: str, delay: float) -> None:\n",
            "    time.sleep(delay)\n",
            "    print(f\"{name} done\")\n",
            "\n",
            "def locked_worker(lock: threading.Lock, tag: str) -> None:\n",
            "    with lock:\n",
            "        print(f\"{tag} in section\")\n",
            "\n",
            "def main() -> None:\n",
            "    t = threading.Thread(target=worker, args=(\"first\", 0.01))\n",
            "    print(t.is_alive())\n",
            "    t.start()\n",
            "    t.join()\n",
            "    print(t.is_alive())\n",
            "    lock = threading.Lock()\n",
            "    with lock:\n",
            "        print(\"locked section\")\n",
            "    print(lock.acquire())\n",
            "    lock.release()\n",
            "    rl = threading.RLock()\n",
            "    with rl:\n",
            "        with rl:\n",
            "            print(\"reentrant\")\n",
            "    ev = threading.Event()\n",
            "    print(ev.is_set())\n",
            "    ev.set()\n",
            "    print(ev.wait())\n",
            "    print(ev.is_set())\n",
            "    ev.clear()\n",
            "    print(ev.is_set())\n",
            "    sem = threading.Semaphore(2)\n",
            "    print(sem.acquire())\n",
            "    sem.release()\n",
            // A lock passed to a worker as an ANNOTATED PARAMETER: the
            // clone shares identity, `with lock:` in the worker really
            // acquires/releases, and the original handle sees the
            // release (Devin review on PR #144).
            "    lk = threading.Lock()\n",
            "    t2 = threading.Thread(target=locked_worker, args=(lk, \"worker\"))\n",
            "    t2.start()\n",
            "    t2.join()\n",
            "    print(lk.locked())\n",
            "    print(threading.active_count() >= 1)\n",
            "    print(threading.current_thread().name)\n",
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

    let output = Command::new(krate.root.join("target/debug/threads"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "False",
            "first done",
            "False",
            "locked section",
            "True",
            "reentrant",
            "False",
            "True",
            "True",
            "False",
            "True",
            "worker in section",
            "False",
            "True",
            "MainThread"
        ],
        "threading semantics diverged from CPython"
    );
}

#[test]
fn socket_echo_matches_python_at_runtime() {
    // A loopback TCP echo: the server runs in a thread (bind/listen/
    // accept/recv/sendall), an Event sequences the client connect, and a
    // refused connection is caught through the OSError hierarchy.
    let scratch = Scratch::new("sockets");
    // An ephemeral port from the OS, released before the generated binary
    // binds it (a fixed literal port would collide across test runs).
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let file = scratch.path().join("sockets.py");
    fs::write(
        &file,
        format!(
            concat!(
                "import socket\n",
                "import threading\n",
                "\n",
                "def serve(port: int, ready: threading.Event) -> None:\n",
                "    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
                "    srv.bind((\"127.0.0.1\", port))\n",
                "    srv.listen(1)\n",
                "    ready.set()\n",
                "    conn, addr = srv.accept()\n",
                "    data = conn.recv(1024)\n",
                "    text = data.decode(\"utf-8\")\n",
                "    reply = \"echo:\" + text\n",
                "    conn.sendall(reply.encode(\"utf-8\"))\n",
                "    conn.close()\n",
                "    srv.close()\n",
                "\n",
                "def main() -> None:\n",
                "    port = {port}\n",
                "    ready = threading.Event()\n",
                "    t = threading.Thread(target=serve, args=(port, ready))\n",
                "    t.start()\n",
                "    ready.wait()\n",
                "    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
                "    cli.connect((\"127.0.0.1\", port))\n",
                "    cli.sendall(\"ping\".encode(\"utf-8\"))\n",
                "    got = cli.recv(1024)\n",
                "    print(got.decode(\"utf-8\"))\n",
                "    cli.close()\n",
                "    t.join()\n",
                "    print(\"closed\")\n",
                "    try:\n",
                "        bad = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
                "        bad.connect((\"127.0.0.1\", 1))\n",
                "    except OSError:\n",
                "        print(\"refused\")\n",
                "\n",
                "if __name__ == \"__main__\":\n",
                "    main()\n",
            ),
            port = port
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/sockets"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["echo:ping", "closed", "refused"],
        "socket semantics diverged from CPython"
    );
}

#[test]
fn urllib_request_matches_python_at_runtime() {
    // urlopen against a local one-shot HTTP server: status, body bytes,
    // getcode, and a refused connection caught through URLError IS-A
    // OSError. The generated Cargo.toml must enable stdpython's ureq-backed
    // `http-ureq` feature (the platform-surface convention).
    use std::io::{Read, Write};
    let scratch = Scratch::new("urlfetch");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut conn, _) = listener.accept().unwrap();
        let mut buf = [0u8; 2048];
        let _ = conn.read(&mut buf);
        let body = "hello from server";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        conn.write_all(resp.as_bytes()).unwrap();
    });
    let file = scratch.path().join("fetch.py");
    fs::write(
        &file,
        format!(
            concat!(
                "import urllib.request\n",
                "\n",
                "def main() -> None:\n",
                "    resp = urllib.request.urlopen(\"http://127.0.0.1:{port}/\")\n",
                "    print(resp.status)\n",
                "    data = resp.read()\n",
                "    print(data.decode(\"utf-8\"))\n",
                "    print(resp.getcode())\n",
                "    try:\n",
                "        urllib.request.urlopen(\"http://127.0.0.1:1/none\")\n",
                "    except OSError:\n",
                "        print(\"unreachable\")\n",
                "\n",
                "if __name__ == \"__main__\":\n",
                "    main()\n",
            ),
            port = port
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let manifest = fs::read_to_string(krate.root.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("http-ureq"),
        "urllib.request import must enable stdpython's http-ureq feature: {}",
        manifest
    );
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/fetch"))
        .output()
        .expect("running generated binary");
    server.join().unwrap();
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["200", "hello from server", "200", "unreachable"],
        "urllib.request semantics diverged from CPython"
    );
}

#[test]
fn bytesio_and_stringio_match_python_at_runtime() {
    // io.BytesIO is a real binary buffer (write returns the byte count and
    // overwrites at the cursor, exactly like StringIO's text discipline).
    let scratch = Scratch::new("bytesbuf");
    let file = scratch.path().join("bytesbuf.py");
    fs::write(
        &file,
        concat!(
            "import io\n",
            "\n",
            "def run() -> None:\n",
            "    b = io.BytesIO(b\"seeded\")\n",
            "    n = b.write(b\"!\")\n",
            "    print(n)\n",
            "    data = b.getvalue()\n",
            "    print(data.decode(\"utf-8\"))\n",
            "    rest = b.read()\n",
            "    print(rest.decode(\"utf-8\"))\n",
            "    s = io.StringIO()\n",
            "    s.write(\"no_std file I/O\")\n",
            "    print(s.getvalue())\n",
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

    let output = Command::new(krate.root.join("target/debug/bytesbuf"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["1", "!eeded", "eeded", "no_std file I/O"],
        "in-memory buffer semantics diverged from CPython"
    );
}

#[test]
fn global_writes_mutate_module_state_at_runtime() {
    // Issue #115: a module-level scalar/None value written by functions
    // through `global` lowers to a mutable static (`static name:
    // Mutex<T>`): writes are visible to every later read, module-wide.
    // `shadow`'s plain local (no `global`) must stay a local.
    let scratch = Scratch::new("globalw");
    let file = scratch.path().join("globalw.py");
    fs::write(
        &file,
        concat!(
            "DEFAULT = None\n",
            "count = 0\n",
            "\n",
            "def setup(v: int) -> None:\n",
            "    global DEFAULT\n",
            "    DEFAULT = v\n",
            "\n",
            "def bump() -> None:\n",
            "    global count\n",
            "    count += 1\n",
            "\n",
            "def shadow() -> int:\n",
            "    total = 5\n",
            "    return total\n",
            "\n",
            "def run() -> int:\n",
            "    bump()\n",
            "    bump()\n",
            "    if DEFAULT is None:\n",
            "        setup(7)\n",
            "    print(count)\n",
            "    print(DEFAULT)\n",
            "    print(shadow())\n",
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

    let output = Command::new(krate.root.join("target/debug/globalw"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["2", "7", "5"],
        "global-write semantics diverged from CPython"
    );
}

#[test]
fn field_from_cross_module_call_result_attribute() {
    // Issue #123: `self.bin_dir = scheme.scripts` where `scheme` is the
    // result of a call into a SIBLING module returning a frozen dataclass
    // (pip's Prefix over get_scheme -> Scheme). Needs call-return typing,
    // dataclass __init__ synthesis, AND the stored-parameter clone
    // (`self.path = path` followed by a later read of `path`).
    let scratch = Scratch::new("xmodfield");
    let root = scratch.path().join("pkg");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("__init__.py"),
        concat!(
            "from locations import get_scheme\n",
            "\n",
            "class Prefix:\n",
            "    def __init__(self, path: str) -> None:\n",
            "        self.path = path\n",
            "        scheme = get_scheme(path)\n",
            "        self.bin_dir = scheme.scripts\n",
            "\n",
            "def main() -> None:\n",
            "    p = Prefix(\"x\")\n",
            "    print(p.bin_dir)\n",
            "    print(p.path)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    fs::write(
        root.join("locations.py"),
        concat!(
            "from dataclasses import dataclass\n",
            "\n",
            "@dataclass(frozen=True)\n",
            "class Scheme:\n",
            "    platlib: str\n",
            "    purelib: str\n",
            "    scripts: str\n",
            "\n",
            "def get_scheme(x: str) -> Scheme:\n",
            "    return Scheme(\"a\", \"b\", \"c\")\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&root).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/pkg"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["c", "x"],
        "cross-module field typing diverged from CPython"
    );
}

#[test]
fn module_level_argparse_with_short_aliases_matches_python() {
    // Issue #118: certifi's __main__.py shape — the parser built at
    // MODULE level (not inside a function), with -short/--long alias
    // pairs. The conversion-time rewrite moves the typed-namespace
    // destructure into __module_init__; the runtime handles short
    // options (exact, attached value) and rejects unknown option-like
    // tokens instead of consuming them as positionals.
    let scratch = Scratch::new("argmod");
    let file = scratch.path().join("argmod.py");
    fs::write(
        &file,
        concat!(
            "import argparse\n",
            "\n",
            "parser = argparse.ArgumentParser(prog=\"certifi\")\n",
            "parser.add_argument(\"-c\", \"--contents\", action=\"store_true\", help=\"print contents\")\n",
            "parser.add_argument(\"-s\", \"--scale\", type=float, default=1.0)\n",
            "args = parser.parse_args()\n",
            "if args.contents:\n",
            "    print(\"contents\", args.scale)\n",
            "else:\n",
            "    print(\"where\", args.scale)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    pass\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&file).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let bin = krate.root.join("target/debug/argmod");

    // --help: python3's exact text (3.11 format), exit 0.
    let output = Command::new(&bin).arg("--help").output().expect("run");
    assert_eq!(output.status.code(), Some(0));
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        concat!(
            "usage: certifi [-h] [-c] [-s SCALE]\n",
            "\n",
            "options:\n",
            "  -h, --help            show this help message and exit\n",
            "  -c, --contents        print contents\n",
            "  -s SCALE, --scale SCALE\n",
        ),
        "help text diverged from CPython"
    );

    // Verified against python3.
    let cases: &[(&[&str], &str)] = &[
        (&[], "where 1.0\n"),
        (&["-c"], "contents 1.0\n"),
        (&["-s", "2.5"], "where 2.5\n"),
        (&["-s2.5", "--contents"], "contents 2.5\n"),
    ];
    for (argv, expected) in cases {
        let output = Command::new(&bin).args(*argv).output().expect("run");
        assert_eq!(output.status.code(), Some(0), "args: {:?}", argv);
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            *expected,
            "args: {:?}",
            argv
        );
    }

    // An unknown option-like token is an error, never a positional.
    // Verified against python3.
    let output = Command::new(&bin).arg("-x").output().expect("run");
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        concat!(
            "usage: certifi [-h] [-c] [-s SCALE]\n",
            "certifi: error: unrecognized arguments: -x\n",
        ),
        "error output diverged from CPython"
    );
}

#[test]
fn range_replace_and_range_delete_match_python_at_runtime() {
    // Issue #153: `xs[a:b] = R` and `del xs[a:b]` — in-place range
    // replacement with Python's exact bound rules: different-length RHS
    // inserts/removes, an inverted range is an insertion point, negatives
    // count from the end, out-of-range bounds clamp, `xs[:] = R` replaces
    // everything.
    let scratch = Scratch::new("splice");
    let file = scratch.path().join("splice.py");
    fs::write(
        &file,
        concat!(
            "def main() -> int:\n",
            "    xs = [1, 2, 3, 4]\n",
            "    xs[1:3] = [9]\n",
            "    print(xs)\n",
            "    xs[1:1] = [7, 8]\n",
            "    print(xs)\n",
            "    xs[-2:] = [0]\n",
            "    print(xs)\n",
            "    xs[10:20] = [5]\n",
            "    print(xs)\n",
            "    xs[3:1] = [6]\n",
            "    print(xs)\n",
            "    del xs[1:3]\n",
            "    print(xs)\n",
            "    del xs[-2:]\n",
            "    print(xs)\n",
            "    del xs[5:9]\n",
            "    print(xs)\n",
            "    ys = [1, 2, 3]\n",
            "    ys[:] = [4]\n",
            "    print(ys)\n",
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

    let output = Command::new(krate.root.join("target/debug/splice"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "[1, 9, 4]",
            "[1, 7, 8, 9, 4]",
            "[1, 7, 8, 0]",
            "[1, 7, 8, 0, 5]",
            "[1, 7, 8, 6, 0, 5]",
            "[1, 6, 0, 5]",
            "[1, 6]",
            "[1, 6]",
            "[4]",
        ],
        "range-replace semantics diverged from CPython"
    );
}

#[test]
fn iter_sentinel_form_matches_python_at_runtime() {
    // Issue #155: `for chunk in iter(callable, sentinel):` — the
    // two-argument iter() calls the callable until it returns the
    // sentinel (botocore's chunked payload reads). The producer advances
    // through a `global` counter, so this also exercises the mutable
    // module statics end to end.
    let scratch = Scratch::new("itersent");
    let file = scratch.path().join("itersent.py");
    fs::write(
        &file,
        concat!(
            "pos = 0\n",
            "\n",
            "def read_chunk() -> str:\n",
            "    global pos\n",
            "    if pos >= 4:\n",
            "        return \"\"\n",
            "    pos = pos + 1\n",
            "    return str(pos)\n",
            "\n",
            "def main() -> None:\n",
            "    total = \"\"\n",
            "    for chunk in iter(read_chunk, \"\"):\n",
            "        total = total + chunk\n",
            "    print(total)\n",
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

    let output = Command::new(krate.root.join("target/debug/itersent"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["1234"],
        "iter(callable, sentinel) semantics diverged from CPython"
    );
}

#[test]
fn varargs_pack_forward_and_index_match_python_at_runtime() {
    // Issue #120: `*args` is the boxed heterogeneous list (Vec<PyValue>).
    // Call sites pack extra positionals boxed; `f(*args)` forwards the
    // vector; len/index work in the body; a `*args, **kwargs` stub takes
    // extra keywords into its dict.
    let scratch = Scratch::new("varargs");
    let file = scratch.path().join("varargs.py");
    fs::write(
        &file,
        concat!(
            "def tag(*args) -> int:\n",
            "    return len(args)\n",
            "\n",
            "def fwd(*args) -> int:\n",
            "    return tag(*args)\n",
            "\n",
            "def first(prefix: str, *args) -> str:\n",
            "    if len(args) > 0:\n",
            "        return prefix + str(args[0])\n",
            "    return prefix\n",
            "\n",
            "def stub(*args, **kwargs) -> int:\n",
            "    return len(args) + len(kwargs)\n",
            "\n",
            "def mixed(a: int, *args, b: int = 5) -> int:\n",
            "    return a * 100 + len(args) * 10 + b\n",
            "\n",
            "def required(a: int, *args, b: int) -> int:\n",
            "    return a * 100 + len(args) * 10 + b\n",
            "\n",
            "def main() -> None:\n",
            "    print(tag(1, \"x\", True))\n",
            "    print(fwd(1, \"x\"))\n",
            "    print(tag())\n",
            "    print(first(\"v=\"))\n",
            "    print(first(\"v=\", 7, 8))\n",
            "    print(stub(1, 2, x=3))\n",
            "    print(mixed(1, 2, b=9))\n",
            "    print(mixed(1, 2, 3))\n",
            "    print(required(1, 2, b=3))\n",
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

    let output = Command::new(krate.root.join("target/debug/varargs"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["3", "2", "0", "v=", "v=7", "3", "119", "125", "113"],
        "*args semantics diverged from CPython"
    );
}

#[test]
fn string_and_computed_globals_mutate_at_runtime() {
    // Issue #115 completion: string-literal and COMPUTED module
    // initializers written through `global` are mutable statics too —
    // `LazyLock<Mutex<String>>` / `LazyLock<Mutex<T>>` (typed when the
    // initializer's type infers, boxed PyValue otherwise, exercised here
    // by the environ-get fallback). Writes are visible module-wide;
    // `label += "!"` is the compound form; boxed `+` dispatches at
    // runtime (PyValue arithmetic).
    let scratch = Scratch::new("globstr");
    let file = scratch.path().join("globstr.py");
    fs::write(
        &file,
        concat!(
            "import os\n",
            "\n",
            "label = \"start\"\n",
            "tag = os.environ.get(\"RYTHON_NO_SUCH_VAR\", \"fallback\")\n",
            "\n",
            "def compute() -> int:\n",
            "    return 2\n",
            "\n",
            "limit = compute()\n",
            "\n",
            "def bump_label(suffix: str) -> None:\n",
            "    global label\n",
            "    label = label + suffix\n",
            "\n",
            "def extend_label() -> None:\n",
            "    global label\n",
            "    label += \"!\"\n",
            "\n",
            "def raise_limit() -> None:\n",
            "    global limit\n",
            "    limit = limit + 10\n",
            "\n",
            "def retag() -> None:\n",
            "    global tag\n",
            "    tag = tag + \"-x\"\n",
            "\n",
            "def main() -> None:\n",
            "    bump_label(\"-a\")\n",
            "    bump_label(\"-b\")\n",
            "    extend_label()\n",
            "    raise_limit()\n",
            "    retag()\n",
            "    print(label)\n",
            "    print(limit)\n",
            "    print(tag)\n",
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

    let output = Command::new(krate.root.join("target/debug/globstr"))
        .env_remove("RYTHON_NO_SUCH_VAR")
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["start-a-b!", "12", "fallback-x"],
        "string/computed global semantics diverged from CPython"
    );
}

#[test]
fn mixed_return_boxing_matches_python_at_runtime() {
    // Issue #133: returns that mix types box to PyValue — a parameter
    // returned as-is alongside a comparison result (botocore's
    // ensure_boolean shape), literal/None mixes under annotated
    // parameters, a value return with a fall-through path, and
    // element-boxed list returns. Each must print exactly what CPython
    // prints.
    let scratch = Scratch::new("retbox");
    let file = scratch.path().join("retbox.py");
    fs::write(
        &file,
        concat!(
            "def flagify(val):\n",
            "    if val:\n",
            "        return val == \"yes\"\n",
            "    return val\n",
            "\n",
            "def pick(flag: bool):\n",
            "    if flag:\n",
            "        return 1\n",
            "    return None\n",
            "\n",
            "def partial(flag: bool):\n",
            "    if flag:\n",
            "        return 2\n",
            "\n",
            "def mixed_list(flag: bool):\n",
            "    if flag:\n",
            "        return [1, \"a\"]\n",
            "    return []\n",
            "\n",
            "def main() -> None:\n",
            "    print(flagify(\"yes\"))\n",
            "    print(flagify(\"no\"))\n",
            "    print(pick(True))\n",
            "    print(pick(False))\n",
            "    print(partial(True))\n",
            "    print(partial(False))\n",
            "    print(mixed_list(True))\n",
            "    print(mixed_list(False))\n",
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

    let output = Command::new(krate.root.join("target/debug/retbox"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["True", "False", "1", "None", "2", "None", "[1, 'a']", "[]"],
        "mixed-return boxing semantics diverged from CPython"
    );
}

#[test]
fn unknown_typed_isinstance_dispatch_matches_python_at_runtime() {
    // Issue #161: an isinstance-dispatched call whose argument has no
    // statically-known type (reassigned through an untyped call —
    // botocore configloader's `path = os.path.expandvars(path)` before
    // `_unicode_path(path)`) falls back to the dynamic router: a str
    // routes to the str morph, bytes land in the residual, whose
    // `decode(enc, "replace")` follows CPython (invalid utf-8 becomes
    // U+FFFD).
    let scratch = Scratch::new("isodyn");
    let file = scratch.path().join("isodyn.py");
    fs::write(
        &file,
        concat!(
            "def _unicode_path(path):\n",
            "    if isinstance(path, str):\n",
            "        return path\n",
            "    return path.decode(\"utf-8\", \"replace\")\n",
            "\n",
            "def norm(p):\n",
            "    return p\n",
            "\n",
            "def load(path):\n",
            "    path = norm(path)\n",
            "    return _unicode_path(path)\n",
            "\n",
            "def main() -> None:\n",
            "    print(_unicode_path(\"direct\"))\n",
            "    print(load(\"abc\"))\n",
            "    print(load(b\"bytes-in\"))\n",
            "    print(load(b\"bad\\xffbyte\"))\n",
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

    let output = Command::new(krate.root.join("target/debug/isodyn"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["direct", "abc", "bytes-in", "bad\u{fffd}byte"],
        "unknown-typed isinstance dispatch diverged from CPython"
    );
}

#[test]
fn generic_sum_matches_python_at_runtime() {
    // Issue #133 (completion): sum() on generic and concrete arguments —
    // the associated-Output PySum serves int and float lists through one
    // generic function, a bool list counts the Trues (bool ⊂ int), and
    // the issue's calc shape pins the Output through its typed slot.
    let scratch = Scratch::new("sumgen");
    let file = scratch.path().join("sumgen.py");
    fs::write(
        &file,
        concat!(
            "def calc(xs):\n",
            "    chunks = []\n",
            "    chunks.append(len(xs))\n",
            "    chunks = [sum(xs)]\n",
            "    return chunks\n",
            "\n",
            "def total(xs):\n",
            "    return sum(xs)\n",
            "\n",
            "def main() -> None:\n",
            "    print(calc([1, 2, 3]))\n",
            "    print(total([1, 2, 3]))\n",
            "    print(total([1.5, 2.5]))\n",
            "    print(total([True, False, True]))\n",
            "    print(sum([10, 20, 30]))\n",
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

    let output = Command::new(krate.root.join("target/debug/sumgen"))
        .output()
        .expect("running generated binary");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .collect::<Vec<_>>(),
        vec!["[6]", "6", "4.0", "2", "60"],
        "sum() semantics diverged from CPython"
    );
}

#[test]
fn singledispatch_dispatches_at_runtime_like_cpython() {
    // Issue #181: `@functools.singledispatch` plus `@<generic>.register(T)`
    // is fused into one isinstance-dispatching function, which the
    // specialization pass monomorphizes. Each call site routes to the
    // morph its argument's static type selects; the float falls through
    // to the generic's own body. Output verified against CPython 3.11:
    // "int 42" / "str HI" / "other 2.5".
    let scratch = Scratch::new("singledispatch");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "import functools\n",
            "\n",
            "@functools.singledispatch\n",
            "def describe(value):\n",
            "    return \"other \" + str(value)\n",
            "\n",
            "@describe.register(int)\n",
            "def _(n):\n",
            "    return \"int \" + str(n * 2)\n",
            "\n",
            "@describe.register(str)\n",
            "def _(text):\n",
            "    return \"str \" + text.upper()\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(describe(21))\n",
            "    print(describe(\"hi\"))\n",
            "    print(describe(2.5))\n",
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
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["int 42", "str HI", "other 2.5"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn class_instance_global_with_list_field_builds_and_runs() {
    // Issue #229: two defects in one repro family.
    // (1) A module-level global bound to a class construction lowers to
    //     `LazyLock<Klass>` whose closure returned `Klass` — but the
    //     construction rendered `{ Klass::new(7)? }`, a `?` the closure
    //     cannot use. The promoted-static path now strips the trailing `?`
    //     through the brace block and panics on Err (§12.2 import-time
    //     divergence), like every other fallible initializer.
    // (2) `self.items = ["kept"]` inferred a `Vec<String>` field but the
    //     store rendered `vec!["kept"]` — a Vec<&str> (E0308). The store
    //     side now owns string-literal elements in list/set fields.
    // Output verified against CPython 3.11: "7" / "['kept']" / "True".
    let scratch = Scratch::new("classglobal");
    let file = scratch.path().join("app.py");
    fs::write(
        &file,
        concat!(
            "class Klass:\n",
            "    def __init__(self, n: int):\n",
            "        self.count = n\n",
            "        self.items = [\"kept\"]\n",
            "\n",
            "\n",
            "REC = Klass(7)\n",
            "\n",
            "\n",
            "def show():\n",
            "    print(REC.count)\n",
            "    print(REC.items)\n",
            "    print(\"kept\" in REC.items)\n",
            "\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    show()\n",
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
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["7", "['kept']", "True"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn self_method_and_module_call_returns_build_and_run() {
    // Issue #222: an unannotated method returning a call to another
    // method of its own class (`return self._retries()` — urllib3's
    // Retry.total) used to collapse to `-> Result<(), PyException>`
    // while the body emitted `Ok(self._retries()?)` — rustc rejects
    // that shape. The return now derives from the callee method's own
    // all-returns unification (one level deep). A sibling-module call
    // (`helper.parse(s)`) derives from the callee's annotation in its
    // DEFINING module — the same repro family, the module half.
    // Output verified against CPython 3.11: "3" / "parsed:ok" / "7".
    let scratch = Scratch::new("selfmethodret");
    let pkg = scratch.path().join("app");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"app\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "from . import helper\n",
            "\n",
            "class Retry:\n",
            "    def _retries(self):\n",
            "        return 3\n",
            "\n",
            "    def total(self):\n",
            "        return self._retries()\n",
            "\n",
            "class Conn:\n",
            "    def __init__(self, scheme: str):\n",
            "        self.scheme = scheme\n",
            "\n",
            "    def direct(self):\n",
            "        return self.scheme\n",
            "\n",
            "    def give(self):\n",
            "        box = self.scheme\n",
            "        return box\n",
            "\n",
            "def parse_wrap(s: str):\n",
            "    return helper.parse(s)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    r = Retry()\n",
            "    c = Conn(\"https\")\n",
            "    print(r.total())\n",
            "    print(parse_wrap(\"ok\"))\n",
            "    print(r._retries() + 4)\n",
            "    print(c.direct())\n",
            "    print(c.give())\n",
        ),
    )
    .unwrap();
    fs::write(
        pkg.join("helper.py"),
        concat!(
            "def parse(s: str) -> str:\n",
            "    return \"parsed:\" + s\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["3", "parsed:ok", "7", "https", "https"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn type_self_dunder_name_matches_cpython() {
    // The #137 sweep's class-name repr family: `type(self).__name__` IS
    // the class name string (urllib3's reprs), and `type(x).__name__` on
    // a concrete receiver routes through the boxed value's runtime type
    // name. Output verified against CPython 3.11: "Pool" / "int".
    let scratch = Scratch::new("typename");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "class Pool:\n",
            "    def __init__(self, host: str):\n",
            "        self.host = host\n",
            "\n",
            "    def typename(self) -> str:\n",
            "        return type(self).__name__\n",
            "\n",
            "def name_of(x: int) -> str:\n",
            "    return type(x).__name__\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    p = Pool(\"example.com\")\n",
            "    print(p.typename())\n",
            "    print(name_of(3))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["Pool", "int"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn boxed_field_containment_matches_cpython() {
    // The #137 sweep's dynamic-`in` cluster (urllib3's
    // RecentlyUsedContainer): a PyValue-typed field (`self.box: Any`)
    // stores concrete members wrapped in PyValue::from, and `key in
    // self.box` dispatches on the boxed member — substring for str —
    // through the new PyContains impls. Output verified against
    // CPython 3.11: "True" / "False".
    let scratch = Scratch::new("boxedcontains");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "from typing import Any\n",
            "\n",
            "class Holder:\n",
            "    def __init__(self) -> None:\n",
            "        self.box: Any = \"abc\"\n",
            "\n",
            "    def has(self, key: str) -> bool:\n",
            "        return key in self.box\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    h = Holder()\n",
            "    print(h.has(\"a\"))\n",
            "    print(h.has(\"zz\"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["True", "False"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn and_or_return_operands_like_cpython() {
    // Issue #137's ca_certs-and-expanduser shape: Python's `and`/`or`
    // return OPERANDS, not booleans. The Option/String mix folds with
    // the operand-returning form; a str literal into an Option<String>
    // parameter owns itself. Output verified against CPython 3.11:
    // None / "" / "y" / "o" / "v" / "v" (the empty line is "").
    let scratch = Scratch::new("andor");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "def pick(ca: str | None, x: str) -> str | None:\n",
            "    return ca and x\n",
            "\n",
            "def pick_or(ca: str, x: str | None) -> str | None:\n",
            "    return ca or x\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(pick(None, \"y\"))\n",
            "    print(pick(\"\", \"y\"))\n",
            "    print(pick(\"c\", \"y\"))\n",
            "    print(pick_or(\"\", \"o\"))\n",
            "    print(pick_or(\"v\", \"o\"))\n",
            "    print(pick_or(\"v\", None))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["None", "", "y", "o", "v", "v"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn certifi_shaped_resource_chain_converts_cleanly_and_panics_loudly() {
    // Round 51: certifi's core.py — version-gated module defs spliced at
    // conversion time, importlib.resources chains dropped (external-module
    // divergence), and the typed return of a dropped chain a loud runtime
    // panic. The real certifi (2025.1.31) measures 0 rustc errors with
    // this design; this minimal fixture pins the shapes so the milestone
    // cannot silently regress.
    let scratch = Scratch::new("certifishape");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "import sys\n",
            "from importlib.resources import files\n",
            "\n",
            "if sys.version_info >= (3, 11):\n",
            "    def where() -> str:\n",
            "        return \"cacert\"\n",
            "    def contents() -> str:\n",
            "        return files(\"probe\").joinpath(\"cacert.pem\").read_text(\"ascii\")\n",
            "\n",
            "def use_util() -> int:\n",
            "    from . import util\n",
            "    return util.helper()\n",
        ),
    )
    .unwrap();
    // A SECOND module makes rypip's crate-wide resolution authoritative:
    // with a single-module package every import is assumed a sibling, so
    // importlib.resources would not drop as external (the real certifi
    // has core.py + __init__.py — two modules).
    fs::write(pkg.join("util.py"), "def helper() -> int:\n    return 7\n").unwrap();
    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &scratch.path().join("crate"), &ConvertOptions::default())
        .expect("convert");
    let status = build_generated(&krate.root);
    assert!(
        status.success(),
        "the certifi-shaped package must convert AND build with zero errors"
    );
    let code = fs::read_to_string(krate.root.join("src/probe/lib.rs"))
        .unwrap_or_else(|_| fs::read_to_string(krate.root.join("src/lib.rs")).unwrap());
    assert!(
        code.contains("panic!") && code.contains("external-module"),
        "the typed return of the dropped importlib.resources chain must panic: {}",
        code
    );
    assert!(
        code.contains("pub fn r#where"),
        "the version-gated def must be a module item (r#where): {}",
        code
    );
}

#[test]
fn class_mapping_protocol_matches_cpython() {
    // §7's mapping-protocol slice: a user class's own dunders receive the
    // subscript store, membership test, and the collections.abc `.get`
    // mixin synthesis (HTTPHeaderDict-shaped classes in urllib3). The
    // class's methods ARE Python's behavior — including the ABC-gated
    // get (a plain __getitem__-only class must not silently gain it).
    // Output verified against CPython 3.11: "5" / "None" / "dflt" /
    // "True" / "5".
    let scratch = Scratch::new("mappingprotocol");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "from typing import MutableMapping\n",
            "\n",
            "class HeaderDict(MutableMapping[str, str]):\n",
            "    def __init__(self) -> None:\n",
            "        self._value = \"5\"\n",
            "\n",
            "    def __getitem__(self, key: str) -> str:\n",
            "        if key == \"Retry-After\":\n",
            "            return self._value\n",
            "        raise KeyError(key)\n",
            "\n",
            "    def __setitem__(self, key: str, val: str) -> None:\n",
            "        self._value = key + val\n",
            "\n",
            "    def __delitem__(self, key: str) -> None:\n",
            "        self._value = \"\"\n",
            "\n",
            "    def __iter__(self) -> list[str]:\n",
            "        ks = [\"Retry-After\"]\n",
            "        return ks\n",
            "\n",
            "    def __len__(self) -> int:\n",
            "        return 1\n",
            "\n",
            "    def __contains__(self, key: str) -> bool:\n",
            "        return key == \"Retry-After\"\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    h = HeaderDict()\n",
            "    print(h.get(\"Retry-After\"))\n",
            "    print(h.get(\"Missing\"))\n",
            "    print(h.get(\"Missing\", \"dflt\"))\n",
            "    print(\"Retry-After\" in h)\n",
            "    print(h[\"Retry-After\"])\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["5", "None", "dflt", "True", "5"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn typing_optional_namedtuple_fields_match_cpython() {
    // Round 47: `typing.NamedTuple("Url", [("scheme",
    // typing.Optional[str]), ...])` — urllib3's Url — types each field as
    // `Option<String>`/`Option<i64>` (the alias-aware resolver previously
    // boxed `typing.Optional[T]` as PyValue), and a tuple-target store of
    // all-None literals (`host = None` on the else path) keeps the name an
    // Option binding so a later Option-returning store passes through.
    // Output verified against CPython 3.11.
    let scratch = Scratch::new("typopt");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "from typing import NamedTuple, Optional\n",
            "\n",
            "class Url(NamedTuple):\n",
            "    scheme: Optional[str]\n",
            "    host: Optional[str]\n",
            "    port: Optional[int]\n",
            "\n",
            "def norm(h: Optional[str]) -> Optional[str]:\n",
            "    return h\n",
            "\n",
            "def parse(url: str) -> Url:\n",
            "    scheme = None\n",
            "    host = None\n",
            "    port = None\n",
            "    if url:\n",
            "        host = norm(host)\n",
            "    return Url(scheme, host, port)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(parse(\"x\"))\n",
            "    print(parse(\"\"))\n",
            "    u = Url(\"https\", \"example.com\", 80)\n",
            "    print(u.scheme, u.host, u.port)\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // CPython prints `Url(scheme=None, host=None, port=None)`; rython's
    // NamedTuple instances display as `<Url object>` (the class-instance
    // display divergence, §12) — the field VALUES are what matter.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "<Url object>",
            "<Url object>",
            "https example.com 80",
        ],
        "stdout: {}",
        stdout
    );
}

#[test]
fn option_field_or_fold_and_narrowed_call_match_cpython() {
    // Round 48: `self.path or "/"` on a `str | None` field (urllib3's
    // Url) unwraps to the plain string (Python's result is never None),
    // and `ca_certs and os.path.expanduser(ca_certs)` passes the
    // UNWRAPPED inner to the call. Output verified against CPython 3.11
    // (`None`/`/tmp/x` for the and-fold; `/`, `/x` for the or-fold).
    let scratch = Scratch::new("fieldor");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "import os\n",
            "from typing import NamedTuple\n",
            "\n",
            "class Url(NamedTuple):\n",
            "    path: str | None\n",
            "    query: str | None\n",
            "\n",
            "    def request_uri(self) -> str:\n",
            "        return self.path or \"/\"\n",
            "\n",
            "def run(u: Url) -> str:\n",
            "    return u.request_uri()\n",
            "\n",
            "def pick(ca: str | None) -> str | None:\n",
            "    return ca and os.path.expanduser(ca)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(run(Url(None, None)))\n",
            "    print(run(Url(\"/x\", \"q\")))\n",
            "    print(pick(None))\n",
            "    print(pick(\"/tmp/x\"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["/", "/x", "None", "/tmp/x"],
        "stdout: {}",
        stdout
    );
}

#[test]
fn dict_returning_self_method_and_string_slots_match_cpython() {
    // Round 46: a local assigned from a DICT-returning self-method call
    // (`ctx = self._merge(None)` where _merge is `-> dict[str, object]`)
    // types from the callee's return annotation, so the subscript stores
    // own their string keys and absorb Option/None values into the boxed
    // dict (`ctx["scheme"] = scheme or "http"`, `ctx["port"] = None`).
    // A str literal stored into a String-typed NAME (`method = "GET"`),
    // appended/inserted into a Vec<String>, or destructured into a
    // String-typed tuple slot all own themselves. Output verified against
    // CPython 3.11.
    let scratch = Scratch::new("selfdict");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "class P:\n",
            "    def _merge(self, override: dict[str, object] | None) -> dict[str, object]:\n",
            "        return {}\n",
            "\n",
            "    def go(self, scheme: str | None) -> None:\n",
            "        ctx = self._merge(None)\n",
            "        ctx[\"scheme\"] = scheme or \"http\"\n",
            "        ctx[\"port\"] = None\n",
            "        print(ctx[\"scheme\"], ctx[\"port\"])\n",
            "\n",
            "def redirect(method: str, flag: bool) -> str:\n",
            "    method = method.upper()\n",
            "    if flag:\n",
            "        method = \"GET\"\n",
            "    return method\n",
            "\n",
            "def vecs(seed_a: str, seed_b: str) -> tuple[list[str], list[str]]:\n",
            "    lines = []\n",
            "    lines.append(seed_a)\n",
            "    lines.append(\"\\r\\n\")\n",
            "    out = []\n",
            "    out.insert(0, \"\")\n",
            "    out.append(seed_b)\n",
            "    return lines, out\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    p = P()\n",
            "    p.go(None)\n",
            "    p.go(\"https\")\n",
            "    print(redirect(\"get\", True))\n",
            "    print(redirect(\"post\", False))\n",
            "    print(vecs(\"ab\", \"cd\"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Verified against python3.
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "http None",
            "https None",
            "GET",
            "POST",
            "(['ab', '\\r\\n'], ['', 'cd'])",
        ],
        "stdout: {}",
        stdout
    );
}

#[test]
fn option_receiver_access_matches_cpython() {
    // Issue #137's Option-aware access: a method call and a field read
    // THROUGH an Option-typed field (`self.timeout.connect_timeout()`
    // where timeout is `Timeout | None`) unwrap the Option — CPython's
    // AttributeError-on-None as a loud §12.2 panic that can only fire if
    // the value was actually None. Output verified against CPython 3.11:
    // "5.0" / "5.0" / "5.0".
    let scratch = Scratch::new("optionaccess");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "class Timeout:\n",
            "    def __init__(self, value: float) -> None:\n",
            "        self._value = value\n",
            "\n",
            "    def connect_timeout(self) -> float:\n",
            "        return self._value\n",
            "\n",
            "    def _value_str(self) -> str:\n",
            "        return str(self._value)\n",
            "\n",
            "class Conn:\n",
            "    def __init__(self, timeout: Timeout | None) -> None:\n",
            "        self.timeout = timeout\n",
            "\n",
            "    def total(self) -> float:\n",
            "        return self.timeout.connect_timeout()\n",
            "\n",
            "    def label(self) -> str:\n",
            "        return self.timeout._value_str()\n",
            "\n",
            "    def maybe(self) -> str:\n",
            "        if self.timeout is not None:\n",
            "            return self.timeout._value_str()\n",
            "        return \"none\"\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    c = Conn(Timeout(5.0))\n",
            "    print(c.total())\n",
            "    print(c.label())\n",
            "    print(c.maybe())\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["5.0", "5.0", "5.0"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn bytes_display_and_join_match_cpython() {
    // The bytes-display slice: print(b"ab") renders b'ab' (not the
    // int-list the Vec<T> display gives), bytes + bytes concatenates and
    // displays as bytes, and b"".join / b"-".join route through the
    // runtime bytes surface. Output verified against CPython 3.11:
    // b'ab' / b'abc' / b'a-b' / b''.
    let scratch = Scratch::new("bytesdisp");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "def join_sep(parts: list[bytes]) -> bytes:\n",
            "    return b\"-\".join(parts)\n",
            "\n",
            "def assemble(parts: list[bytes]) -> bytes:\n",
            "    return b\"\".join(parts)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    x = b\"ab\"\n",
            "    print(x)\n",
            "    y = x + b\"c\"\n",
            "    print(y)\n",
            "    print(join_sep([b\"a\", b\"b\"]))\n",
            "    print(assemble([]))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec!["b'ab'", "b'abc'", "b'a-b'", "b''"],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn class_values_extend_and_dynamic_except_match_cpython() {
    // The round-33 class-as-value pipeline, end to end (botocore's
    // retryhandler.py shapes): classes as values are their name strings,
    // lists of them extend through the boxed heterogeneous container,
    // `except self._retryable_exceptions:` matches the RUNTIME boxed
    // value via matches_value (a non-catchable value — None — raises
    // CPython's TypeError exactly when CPython does), and `tuple(...)`
    // of the collected names boxes. Output verified against CPython:
    // caught / True / typeerror: catching classes that do not inherit
    // from BaseException is not allowed.
    let scratch = Scratch::new("classval");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "class ChecksumError(Exception):\n",
            "    pass\n",
            "\n",
            "class ConnectionError(Exception):\n",
            "    pass\n",
            "\n",
            "EXCEPTION_MAP = {\n",
            "    \"GENERAL_CONNECTION_ERROR\": [\n",
            "        ConnectionError,\n",
            "        ChecksumError,\n",
            "    ],\n",
            "}\n",
            "\n",
            "def extract(kind: str):\n",
            "    if kind == \"a\":\n",
            "        return [ChecksumError]\n",
            "    elif kind == \"b\":\n",
            "        exceptions = []\n",
            "        exceptions.extend(EXCEPTION_MAP[\"GENERAL_CONNECTION_ERROR\"])\n",
            "        return exceptions\n",
            "\n",
            "def collect(kind: str):\n",
            "    retryable = []\n",
            "    for k in [\"a\", \"b\"]:\n",
            "        ex = extract(k)\n",
            "        if ex is not None:\n",
            "            retryable.extend(ex)\n",
            "    return tuple(retryable)\n",
            "\n",
            "class Decorator:\n",
            "    def __init__(self, retryable_exceptions=None):\n",
            "        self._retryable_exceptions = retryable_exceptions\n",
            "\n",
            "    def check(self, value: int):\n",
            "        try:\n",
            "            if value == 1:\n",
            "                raise ChecksumError(\"boom\")\n",
            "            return False\n",
            "        except self._retryable_exceptions as e:\n",
            "            print(\"caught\")\n",
            "            return True\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    d = Decorator(retryable_exceptions=collect(\"x\"))\n",
            "    print(d.check(1))\n",
            "    d2 = Decorator()\n",
            "    try:\n",
            "        print(d2.check(1))\n",
            "    except TypeError as e:\n",
            "        print(\"typeerror:\", e)\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "caught",
            "True",
            "typeerror: catching classes that do not inherit from BaseException is not allowed",
        ],
        "stdout: {} stderr: {}",
        stdout,
        stderr
    );
}

#[test]
fn class_instance_display_matches_cpython() {
    // Round 34's display cluster: str(x)/print(x)/f-string `{x}` on a
    // class INSTANCE route through py_display — the class's __str__
    // (falling back to __repr__, then the default object repr). The
    // default repr drops the nondeterministic address CPython prints
    // (documented §12.3 divergence; CPython's own output varies run to
    // run) and the module prefix for the crate root. Output verified
    // against CPython: Pool(host='x') / <... object> / fstring=Pool(...)
    // / a message containing str(pool).
    let scratch = Scratch::new("classdisp");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "class PoolError(Exception):\n",
            "    pass\n",
            "\n",
            "class Pool:\n",
            "    def __init__(self, host: str):\n",
            "        self._host = host\n",
            "\n",
            "    def __str__(self) -> str:\n",
            "        return f\"Pool(host={self._host!r})\"\n",
            "\n",
            "class RawConnection:\n",
            "    def __init__(self):\n",
            "        self._x = 1\n",
            "\n",
            "def boom():\n",
            "    raise PoolError(Pool(\"example.com\"), \"Pool is closed.\")\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(Pool(\"x\"))\n",
            "    print(str(RawConnection()))\n",
            "    print(f\"fstring={Pool('y')}\")\n",
            "    try:\n",
            "        boom()\n",
            "    except PoolError as e:\n",
            "        print(str(e))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[0], "Pool(host='x')", "stdout: {}", stdout);
    assert!(
        lines[1].starts_with('<') && lines[1].contains("RawConnection object"),
        "default repr: {}",
        stdout
    );
    assert_eq!(lines[2], "fstring=Pool(host='y')", "stdout: {}", stdout);
    assert!(
        lines[3].contains("Pool(host='example.com')"),
        "the __str__ must be honored inside the message: {}",
        stdout
    );
}

#[test]
fn percent_formatting_matches_cpython_end_to_end() {
    // Round 34's %-operator cluster: `b"%x\r\n%b\r\n" % (len, chunk)`
    // (urllib3's chunked framing), a 3-arity str `%` building a regex,
    // and the %r mapping. Output verified against CPython.
    let scratch = Scratch::new("pctfmt");
    let pkg = scratch.path().join("probe");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(
        scratch.path().join("pyproject.toml"),
        "[project]\nname = \"probe\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        pkg.join("__init__.py"),
        concat!(
            "def frame(chunk: bytes) -> bytes:\n",
            "    return b\"%x\\r\\n%b\\r\\n\" % (len(chunk), chunk)\n",
            "\n",
            "def pat(name: str, host: str) -> str:\n",
            "    return \"^(%s|%s)(?::0*?(|0|[1-9][0-9]{0,4}))?$\" % (name, host)\n",
            "\n",
            "def hostline(hostname: str) -> str:\n",
            "    return \"hostname %r doesn't match\" % (hostname,)\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(frame(b\"hello\"))\n",
            "    print(pat(\"x\", \"y\"))\n",
            "    print(hostline(\"example.com\"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/probe"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        vec![
            "b'5\\r\\nhello\\r\\n'",
            "^(x|y)(?::0*?(|0|[1-9][0-9]{0,4}))?$",
            "hostname 'example.com' doesn't match",
        ],
        "stdout: {}",
        stdout
    );
}

#[test]
fn the_stdpython_dependency_carries_only_the_surfaces_the_package_imports() {
    // Platform surfaces are per-feature (the convention stdpython's own
    // Cargo.toml states), and the generated manifest opts into exactly the
    // ones the package's imports ask for rather than riding stdpython's
    // defaults: a package that never imports `ssl` or `re` must not
    // compile rustls or the regex engine. Getting a predicate too narrow
    // is loud, never silent — the generated crate names a module that was
    // not compiled in — and the re/urllib end-to-end tests above are the
    // proof that the surfaces are sufficient when they ARE requested.
    let cases: [(&str, &str, &[&str], &[&str]); 4] = [
        (
            "plain",
            "def main() -> None:\n    print(\"hi\")\n",
            &[],
            &["ssl-rustls", "re-regex", "http-ureq", "pyo3-interop"],
        ),
        (
            "re",
            "import re\n\ndef main() -> None:\n    print(re.search(\"a\", \"a\") is not None)\n",
            &["re-regex"],
            &["ssl-rustls", "http-ureq", "pyo3-interop"],
        ),
        (
            "ssl",
            "import ssl\n\ndef main() -> None:\n    print(ssl.OPENSSL_VERSION)\n",
            &["ssl-rustls"],
            &["re-regex", "http-ureq", "pyo3-interop"],
        ),
        // Discovery has to reach every statement form conversion emits,
        // not just module level: an import nested in an async function or
        // a class body counts exactly like a top-level one.
        (
            "async-nested",
            "async def probe() -> None:\n    import re\n    print(re.search(\"a\", \"a\") is not None)\n",
            &["re-regex"],
            &["ssl-rustls", "http-ureq", "pyo3-interop"],
        ),
    ];
    for (tag, source, present, absent) in cases {
        let scratch = Scratch::new(&format!("surface-{tag}"));
        let file = scratch.path().join("probe.py");
        fs::write(&file, source).unwrap();
        let out = scratch.path().join("crate");
        let pkg = rypip::discover(&file).expect("discover");
        let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
        let manifest = fs::read_to_string(krate.root.join("Cargo.toml")).unwrap();
        let dep = manifest
            .lines()
            .find(|l| l.starts_with("stdpython = "))
            .unwrap_or_else(|| panic!("no stdpython dependency in: {}", manifest));
        assert!(
            dep.contains("default-features = false") && dep.contains("\"std\""),
            "{tag}: the std tier must be requested explicitly: {dep}"
        );
        for feature in present {
            assert!(dep.contains(feature), "{tag}: expected {feature} in: {dep}");
        }
        for feature in absent {
            assert!(!dep.contains(feature), "{tag}: unexpected {feature} in: {dep}");
        }
    }
}

#[test]
fn a_vendored_dependencys_imports_reach_the_surface_feature_list() {
    // `[python-modules]` deps are transpiled into the same crate as sibling
    // modules, so their imports drive the generated manifest's feature list
    // exactly like the entry module's. Without this the crate would name
    // `stdpython::re` with the regex engine gated out.
    let scratch = Scratch::new("surface-vendored");
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(
        scratch.path().join("vendor/matcher.py"),
        "import re\n\ndef looks_like(text: str) -> bool:\n    return re.search(\"a\", text) is not None\n",
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("matchapp")).unwrap();
    fs::write(scratch.path().join("matchapp/__init__.py"), "").unwrap();
    fs::write(
        scratch.path().join("matchapp/main.py"),
        concat!(
            "import matcher\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(matcher.looks_like(\"cat\"))\n",
        ),
    )
    .unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\nmatcher = { path = \"vendor/matcher.py\" }\n",
    )
    .unwrap();

    let out = scratch.path().join("crate");
    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let manifest = fs::read_to_string(krate.root.join("Cargo.toml")).unwrap();
    let dep = manifest
        .lines()
        .find(|l| l.starts_with("stdpython = "))
        .unwrap_or_else(|| panic!("no stdpython dependency in: {}", manifest));
    assert!(
        dep.contains("re-regex"),
        "a vendored dependency's `import re` must enable the surface: {dep}"
    );

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/matchapp"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "True", "stdout: {}", stdout);
}

#[test]
fn an_unreachable_vendored_module_does_not_add_a_surface() {
    // `convert` transpiles only import-reachable modules, so a vendored
    // dependency the program never imports contributes no code to the
    // crate -- and must contribute no stdpython features either. The
    // feature list has to track what is emitted, in both directions.
    let scratch = Scratch::new("surface-unreachable");
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(
        scratch.path().join("vendor/used.py"),
        "def shout(text: str) -> str:\n    return text.upper()\n",
    )
    .unwrap();
    fs::write(
        scratch.path().join("vendor/unused.py"),
        "import re\n\ndef looks_like(text: str) -> bool:\n    return re.search(\"a\", text) is not None\n",
    )
    .unwrap();
    fs::create_dir_all(scratch.path().join("quietapp")).unwrap();
    fs::write(scratch.path().join("quietapp/__init__.py"), "").unwrap();
    fs::write(
        scratch.path().join("quietapp/main.py"),
        concat!(
            "import used\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(used.shout(\"cat\"))\n",
        ),
    )
    .unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\nused = { path = \"vendor/used.py\" }\n         unused = { path = \"vendor/unused.py\" }\n",
    )
    .unwrap();

    let out = scratch.path().join("crate");
    let pkg = rypip::discover(scratch.path()).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");
    let manifest = fs::read_to_string(krate.root.join("Cargo.toml")).unwrap();
    let dep = manifest
        .lines()
        .find(|l| l.starts_with("stdpython = "))
        .unwrap_or_else(|| panic!("no stdpython dependency in: {}", manifest));
    assert!(
        !dep.contains("re-regex"),
        "an unreachable vendored module's `import re` must not add the surface: {dep}"
    );

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");
    let output = Command::new(krate.root.join("target/debug/quietapp"))
        .output()
        .expect("running generated binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "CAT", "stdout: {}", stdout);
}
