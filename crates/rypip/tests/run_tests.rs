//! `rypip run`: CPython's command-line shape over the convert/build
//! pipeline (issue #166).

mod common;

use std::fs;
use std::process::Command;

use common::Scratch;

#[test]
fn rypip_run_executes_the_program_with_its_arguments_and_exit_status() {
    // `python hello.py a -b` prints sys.argv — the script path AS GIVEN
    // first — and exits 3; `rypip run hello.py -- a -b` does the same: the
    // program's stdout is rypip's stdout (the build is quiet), sys.argv[0]
    // is `hello.py` (Devin review on #340), and its exit status is rypip's.
    let scratch = Scratch::new("run");
    let file = scratch.path().join("hello.py");
    fs::write(
        &file,
        concat!(
            "import sys\n",
            "\n",
            "\n",
            "def main() -> None:\n",
            "    print(\"args\", sys.argv)\n",
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
        .arg("hello.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .arg("--")
        .args(["a", "-b"])
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "args ['hello.py', 'a', '-b']\n",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(3));

    // The executable is the one cargo reports, wherever it put it: with
    // CARGO_TARGET_DIR pointing outside the crate there is no
    // `<crate>/target/release/hello`, and the program still runs
    // (`./hello.py` stays `./hello.py`, as under python3).
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg("./hello.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .env("CARGO_TARGET_DIR", scratch.path().join("elsewhere"))
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "args ['./hello.py']\n",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(3));
    assert!(scratch.path().join("elsewhere").join("release").exists());

    // A non-UTF-8 argument is loud: CPython keeps it as surrogate escapes
    // (`['-c', '\\udcff']`), which the runtime's str does not model, so the
    // program exits 1 naming the argument instead of panicking or running
    // with a different string.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
            .arg("run")
            .arg("hello.py")
            .arg("--no-deps")
            .arg("--out")
            .arg(&out)
            .arg("--")
            .arg(std::ffi::OsStr::from_bytes(b"\xff"))
            .current_dir(scratch.path())
            .env_remove("RUSTFLAGS")
            .output()
            .expect("running rypip");
        assert_eq!(output.status.code(), Some(1), "stdout: {}", String::from_utf8_lossy(&output.stdout));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("rython: sys.argv[1] is not valid UTF-8"),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    }

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

#[test]
fn rypip_run_keeps_same_named_programs_apart_and_serializes_a_shared_work_dir() {
    // Two programs that discover the same package name (`hello`) in two
    // directories get two default work dirs — the key carries a hash of
    // the canonical source path — so running them at the same time never
    // builds or executes mixed code (Devin review on #340); and two runs of
    // ONE source share a work dir under an exclusive lock across conversion
    // and build, so they never interleave either.
    let scratch = Scratch::new("runtwo");
    let a = scratch.path().join("a");
    let b = scratch.path().join("b");
    fs::create_dir_all(&a).unwrap();
    fs::create_dir_all(&b).unwrap();
    for (dir, word) in [(&a, "alpha"), (&b, "beta")] {
        fs::write(
            dir.join("hello.py"),
            format!(
                concat!(
                    "def main() -> None:\n",
                    "    print(\"{}\")\n",
                    "\n",
                    "\n",
                    "if __name__ == \"__main__\":\n",
                    "    main()\n",
                ),
                word
            ),
        )
        .unwrap();
    }
    let dir_a = rypip::run_work_dir(&a.join("hello.py"), "hello").unwrap();
    let dir_b = rypip::run_work_dir(&b.join("hello.py"), "hello").unwrap();
    assert_ne!(dir_a, dir_b);
    assert_eq!(dir_a, rypip::run_work_dir(&a.join("hello.py"), "hello").unwrap(), "stable across runs");
    let spawn = |dir: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_rypip"))
            .arg("run")
            .arg("hello.py")
            .arg("--no-deps")
            .current_dir(dir)
            .env_remove("RUSTFLAGS")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawning rypip")
    };
    // The same-named programs, concurrently; then the same source twice,
    // concurrently.
    let (ra, rb) = (spawn(&a), spawn(&b));
    let (oa, ob) = (ra.wait_with_output().unwrap(), rb.wait_with_output().unwrap());
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&oa.stdout), "alpha\n", "stderr: {}", String::from_utf8_lossy(&oa.stderr));
    assert_eq!(String::from_utf8_lossy(&ob.stdout), "beta\n", "stderr: {}", String::from_utf8_lossy(&ob.stderr));
    let (r1, r2) = (spawn(&a), spawn(&a));
    let (o1, o2) = (r1.wait_with_output().unwrap(), r2.wait_with_output().unwrap());
    assert_eq!(String::from_utf8_lossy(&o1.stdout), "alpha\n", "stderr: {}", String::from_utf8_lossy(&o1.stderr));
    assert_eq!(String::from_utf8_lossy(&o2.stdout), "alpha\n", "stderr: {}", String::from_utf8_lossy(&o2.stderr));
    assert!(dir_a.with_extension("lock").is_file(), "the work dir's lock file sits beside it");
    for dir in [&dir_a, &dir_b] {
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_file(dir.with_extension("lock"));
    }
}
