//! Integration tests for `[python-modules]`: vendored PyPI-style Python
//! libraries compiled into the generated crate. The manifest maps an import
//! name to a .py file or package directory; `from pylev import ...` and
//! `import pylev; pylev.fn(...)` lower to direct calls into the transpiled
//! sibling module.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rypip::convert::ConvertOptions;

/// A scratch directory that's removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("rypip-pymod-{}-{}", tag, std::process::id()));
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

/// Build a generated crate, scrubbing RUSTFLAGS (see rust_bind_tests).
fn build_generated(root: &Path) -> std::process::ExitStatus {
    Command::new("cargo")
        .arg("build")
        .env_remove("RUSTFLAGS")
        .current_dir(root)
        .status()
        .expect("running cargo build")
}

/// pylev's `wf.py` (Wagner-Fischer Levenshtein), vendored from PyPI with
/// the two documented typed-subset edits: the dead Python-2 `range`
/// shim removed and `str`/`int` annotations added. Output verified against
/// CPython pylev: kitten/sitting=3, flaw/lawn=2, saturday/sunday=3, ...
const PYLEV_WF_RS: &str = r#"def wf_levenshtein(string_1: str, string_2: str) -> int:
    len_1 = len(string_1) + 1
    len_2 = len(string_2) + 1

    d = [0] * (len_1 * len_2)

    for i in range(len_1):
        d[i] = i
    for j in range(len_2):
        d[j * len_1] = j

    for j in range(1, len_2):
        for i in range(1, len_1):
            if string_1[i - 1] == string_2[j - 1]:
                d[i + j * len_1] = d[i - 1 + (j - 1) * len_1]
            else:
                d[i + j * len_1] = min(
                    d[i - 1 + j * len_1] + 1,
                    d[i + (j - 1) * len_1] + 1,
                    d[i - 1 + (j - 1) * len_1] + 1,
                )

    return d[-1]


def wfi_levenshtein(string_1: str, string_2: str) -> int:
    if string_1 == string_2:
        return 0

    len_1 = len(string_1)
    len_2 = len(string_2)

    if len_1 == 0:
        return len_2
    if len_2 == 0:
        return len_1

    if len_1 > len_2:
        string_2, string_1 = string_1, string_2
        len_2, len_1 = len_1, len_2

    d0 = [i for i in range(len_2 + 1)]
    d1 = [j for j in range(len_2 + 1)]

    for i in range(len_1):
        d1[0] = i + 1
        for j in range(len_2):
            cost = d0[j]

            if string_1[i] != string_2[j]:
                cost += 1

                x_cost = d1[j] + 1
                if x_cost < cost:
                    cost = x_cost

                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0

    return d0[-1]
"#;

/// A single-file vendored dependency: both import styles lower to direct
/// calls, the crate compiles, and the binary's output matches CPython pylev.
#[test]
fn single_file_python_module_compiles_and_matches_cpython() {
    let scratch = Scratch::new("singlefile");
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(scratch.path().join("vendor/wf.py"), PYLEV_WF_RS).unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\npylev = { path = \"vendor/wf.py\" }\n",
    )
    .unwrap();
    fs::write(
        scratch.path().join("app.py"),
        concat!(
            "import pylev\n",
            "from pylev import wf_levenshtein as wf\n",
            "from pylev import wfi_levenshtein\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(wf(\"kitten\", \"sitting\"))\n",
            "    print(pylev.wfi_levenshtein(\"flaw\", \"lawn\"))\n",
            "    print(wf(\"\", \"\"))\n",
            "    print(wfi_levenshtein(\"saturday\", \"sunday\"))\n",
            "    print(pylev.wf_levenshtein(\"book\", \"back\"))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("app.py")).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    // The from-imports lower to direct calls; the plain `import pylev`
    // emits no `use` (the dep is a sibling module of the crate), and
    // module attribute access resolves to a `crate::` path.
    let main_rs = fs::read_to_string(out.join("src/main.rs")).unwrap();
    for expected in [
        "use crate::pylev::wf_levenshtein as wf;",
        "use crate::pylev::wfi_levenshtein;",
        "crate::pylev::wfi_levenshtein(",
        "crate::pylev::wf_levenshtein(",
    ] {
        assert!(main_rs.contains(expected), "missing {expected} in: {main_rs}");
    }
    // Both spellings propagate the Result-returning call (`?`).
    for expected in ["wf(\"kitten\", \"sitting\")?", "crate::pylev::wfi_levenshtein(\"flaw\", \"lawn\")?"] {
        assert!(main_rs.contains(expected), "missing {expected} in: {main_rs}");
    }

    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    assert!(output.status.success(), "binary exited nonzero");
    let lines: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    // CPython pylev outputs (verified against the real package).
    assert_eq!(
        lines,
        vec!["3", "2", "0", "3", "2"],
        "Levenshtein results must match CPython pylev"
    );
}

/// A package-directory dependency with a relative import inside it:
/// `__init__.py` re-exports from `.core`, and the app reaches the function
/// both directly and through the module path.
#[test]
fn package_dir_python_module_resolves_relative_imports() {
    let scratch = Scratch::new("pkgdir");
    fs::create_dir_all(scratch.path().join("vendor/textlib")).unwrap();
    fs::write(
        scratch.path().join("vendor/textlib/__init__.py"),
        "from .core import double\n",
    )
    .unwrap();
    fs::write(
        scratch.path().join("vendor/textlib/core.py"),
        "def double(n: int) -> int:\n    return n * 2\n",
    )
    .unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\ntextlib = { path = \"vendor/textlib\" }\n",
    )
    .unwrap();
    fs::write(
        scratch.path().join("app.py"),
        concat!(
            "import textlib\n",
            "from textlib import double\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    print(double(21))\n",
            "    print(textlib.double(4))\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("app.py")).expect("discover");
    let krate = rypip::convert(&pkg, &out, &ConvertOptions::default()).expect("convert");

    // The dep is a real module tree: src/textlib/mod.rs re-exports core.
    let mod_rs = fs::read_to_string(out.join("src/textlib/mod.rs")).unwrap();
    assert!(
        mod_rs.contains("pub use crate::textlib::core::double;"),
        "relative import must lower to a crate path: {}",
        mod_rs
    );

    let status = build_generated(&krate.root);
    assert!(status.success(), "generated crate failed to compile");

    let output = Command::new(krate.root.join("target/debug/app"))
        .output()
        .expect("running generated binary");
    assert!(output.status.success(), "binary exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.lines().collect::<Vec<_>>(), vec!["42", "8"]);
}

/// Kernel modules compile a single entry file with no module tree, so a
/// manifest declaring python-modules must fail loudly, not be ignored.
#[test]
fn python_modules_rejected_for_kernel_modules() {
    let scratch = Scratch::new("kernel");
    fs::create_dir_all(scratch.path().join("vendor")).unwrap();
    fs::write(scratch.path().join("vendor/wf.py"), PYLEV_WF_RS).unwrap();
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\npylev = { path = \"vendor/wf.py\" }\n",
    )
    .unwrap();
    fs::write(scratch.path().join("mod.py"), "def module_init() -> int:\n    return 0\n").unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("mod.py")).expect("discover");
    let err = rypip::convert(
        &pkg,
        &out,
        &ConvertOptions {
            kernel_module: true,
            ..Default::default()
        },
    )
    .expect_err("kernel modules must reject [python-modules]");
    assert!(
        err.to_string().contains("python-modules"),
        "error should name [python-modules]: {}",
        err
    );
}

/// A manifest entry pointing at a missing file is a configuration error
/// with the import name in the message.
#[test]
fn missing_python_module_path_is_an_error() {
    let scratch = Scratch::new("missing");
    fs::write(
        scratch.path().join("rython.toml"),
        "[python-modules]\npylev = { path = \"vendor/does-not-exist.py\" }\n",
    )
    .unwrap();
    fs::write(scratch.path().join("app.py"), "x = 1\n").unwrap();
    let out = scratch.path().join("crate");

    let pkg = rypip::discover(&scratch.path().join("app.py")).expect("discover");
    let err = rypip::convert(&pkg, &out, &ConvertOptions::default())
        .expect_err("missing dep path must fail conversion");
    assert!(
        err.to_string().contains("pylev"),
        "error should name the import: {}",
        err
    );
}
