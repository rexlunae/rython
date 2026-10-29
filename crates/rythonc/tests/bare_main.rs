//! Issue #431 / Devin review on #454: the single-file compiler's input IS
//! an executable program (`python input.py` runs it as `__main__`), so a
//! final bare `main()` call is that program's body. `rythonc` must accept
//! it and emit a Rust `fn main` — it previously refused the shape as a
//! non-entry module, regressing the standalone path.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// A scratch directory under the OS temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        // The CLI derives the module name from the file path and rejects
        // anything that is not a Python identifier, so the scratch path
        // holds no hyphens (a `rythonc_test_<tag>_<pid>` segment).
        let dir = std::env::temp_dir().join(format!(
            "rythonc_test_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("creating scratch dir");
        Self(dir)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_final_bare_main_call_compiles_to_a_rust_entry_point() {
    // `python3 single.py` prints `hi` (verified against python3). The
    // standalone compiler is handed exactly that file.
    let scratch = Scratch::new("baremain");
    let input = scratch.path().join("single.py");
    let output = scratch.path().join("out.rs");
    fs::write(
        &input,
        "def main() -> None:\n    print(\"hi\")\n\n\nmain()\n",
    )
    .expect("writing the input");

    let run = Command::new(env!("CARGO_BIN_EXE_rythonc"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("running rythonc");

    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "a program whose body is a final bare `main()` call must compile: {}",
        stderr
    );
    let emitted = fs::read_to_string(&output).expect("reading the emitted Rust");
    // The user's `main` became the process entry point, once.
    assert_eq!(
        emitted.matches("fn main (").count(),
        1,
        "exactly one Rust entry point: {}",
        emitted
    );
    assert!(
        !emitted.contains("pub fn main ("),
        "no `pub fn main` beside the entry point (E0428): {}",
        emitted
    );
    assert!(
        emitted.contains("hi"),
        "the user's main body is the program: {}",
        emitted
    );
}

#[test]
fn a_final_main_call_with_arguments_is_refused_by_the_standalone_compiler() {
    // The same rule the converter applies: arguments cannot cross the
    // entry boundary (Rust's `fn main` takes none), so the shape is a loud
    // error naming the rewrite rather than a silently argument-dropping
    // program. `python3` prints `1` for it.
    let scratch = Scratch::new("baremainargs");
    let input = scratch.path().join("pos.py");
    let output = scratch.path().join("out.rs");
    fs::write(
        &input,
        "def main(x: int) -> None:\n    print(x)\n\n\nmain(1)\n",
    )
    .expect("writing the input");

    let run = Command::new(env!("CARGO_BIN_EXE_rythonc"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("running rythonc");

    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "an argument-bearing entry call must be refused: {}",
        stderr
    );
    assert!(
        stderr.contains("passes arguments") && stderr.contains("__main__"),
        "the refusal names the construct and the rewrite: {}",
        stderr
    );
}
