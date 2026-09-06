//! `rypip run`: CPython's command-line shape over the convert/build
//! pipeline (issue #166).

mod common;

use std::fs;
use std::process::Command;

use common::Scratch;

#[test]
fn rypip_run_executes_the_program_with_its_arguments_and_exit_status() {
    // `python hello.py a -b` prints the arguments and exits 3; `rypip run
    // hello.py -- a -b` does the same — the program's stdout is rypip's
    // stdout (the build is quiet), and its exit status is rypip's.
    let scratch = Scratch::new("run");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "import sys\n",
            "\n",
            "\n",
            "def main() -> None:\n",
            "    print(\"args\", sys.argv[1:])\n",
            "    sys.exit(3)\n",
            "\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let out = scratch.path().join("crate");
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg(&file)
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .arg("--")
        .args(["a", "-b"])
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "args ['a', '-b']\n",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(3));

    // A library (no entry point) is refused with the fix named.
    let lib = scratch.path().join("lib.py");
    fs::write(&lib, "def double(n: int) -> int:\n    return n * 2\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg(&lib)
        .arg("--no-deps")
        .arg("--out")
        .arg(scratch.path().join("libcrate"))
        .output()
        .expect("running rypip");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("has no entry point"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
