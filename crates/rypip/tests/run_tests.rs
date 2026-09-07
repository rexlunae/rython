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
    let lock_of = |dir: &std::path::Path| std::path::PathBuf::from(format!("{}.lock", dir.display()));
    assert!(lock_of(&dir_a).is_file(), "the work dir's lock file sits beside it");
    for dir in [&dir_a, &dir_b] {
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_file(lock_of(dir));
    }
}

#[test]
fn the_work_dir_lock_is_appended_and_the_executable_is_staged_per_invocation() {
    // Round 3 of the review on #340: the lock file is `<dir>.lock` by
    // APPENDING, so an output named `x.lock` locks `x.lock.lock` and can
    // still be created as a directory (`with_extension` would have made the
    // lock file the directory's own path); and the executable an invocation
    // runs is its own hard link under `<dir>/.run/`, distinct per
    // invocation and removed when the run is over.
    let scratch = Scratch::new("runlock");
    let out = scratch.path().join("x.lock");
    let lock = rypip::lock_work_dir(&out).expect("lock");
    assert!(scratch.path().join("x.lock.lock").is_file());
    assert!(!out.exists(), "the lock file did not take the output's path");
    fs::create_dir_all(&out).expect("the output directory can be created beside its lock");
    drop(lock);

    let exe = scratch.path().join("program");
    fs::write(&exe, b"#!/bin/sh\necho staged\n").unwrap();
    let first = rypip::stage_executable(&out, &exe).expect("stage");
    let second = rypip::stage_executable(&out, &exe).expect("stage");
    assert_ne!(first.path(), second.path());
    assert!(first.path().starts_with(out.join(".run")));
    assert_eq!(fs::read(first.path()).unwrap(), fs::read(&exe).unwrap());
    let first_path = first.path().to_path_buf();
    drop(first);
    assert!(!first_path.exists(), "a staged executable is removed with its run");
    assert!(second.path().exists());
}

#[test]
fn a_reused_work_dir_regenerates_its_sources_from_scratch() {
    // Round 3 of the review on #340: a run regenerates the crate's `src/`
    // from scratch, so a module the previous conversion into the same work
    // dir wrote — here a different program given the same explicit `--out`
    // — leaves no stale source behind, while cargo's `target/` survives for
    // incremental rebuilds.
    let scratch = Scratch::new("runreuse");
    let out = scratch.path().join("crate");
    for (name, word) in [("first.py", "first"), ("second.py", "second")] {
        fs::write(
            scratch.path().join(name),
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
        let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
            .arg("run")
            .arg(name)
            .arg("--no-deps")
            .arg("--out")
            .arg(&out)
            .current_dir(scratch.path())
            .env_remove("RUSTFLAGS")
            .output()
            .expect("running rypip");
        // Verified against python3.
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("{}\n", word),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(out.join("src").join("second.rs").is_file());
    assert!(!out.join("src").join("first.rs").exists(), "the previous program's module is gone");
    assert!(out.join("target").is_dir(), "cargo's target dir survives");
}

#[test]
fn rypip_run_shows_the_compilers_diagnostics_when_the_generated_crate_fails() {
    // The build runs with `--message-format=json-render-diagnostics`, so
    // cargo renders rustc's diagnostics to stderr itself while the artifact
    // messages go to stdout: a program the converter accepts but rustc
    // rejects (`f.closed`, an attribute spelling the runtime keeps loud as
    // E0615) fails with the real diagnostic visible (Devin review on #340,
    // round 2).
    let scratch = Scratch::new("runbad");
    let file = scratch.path().join("bad.py");
    fs::write(
        &file,
        concat!(
            "def main() -> None:\n",
            "    with open(\"x.txt\", \"w\") as f:\n",
            "        f.write(\"hi\")\n",
            "        print(f.closed)\n",
            "\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg("bad.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(scratch.path().join("crate"))
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[E0615]"), "stderr: {}", stderr);
    assert!(stderr.contains("cargo build failed"), "stderr: {}", stderr);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "", "no JSON leaks to stdout");
}

#[cfg(unix)]
#[test]
fn argparse_reads_the_same_argv_authority_as_sys_argv() {
    // argparse's `parse_args()` reads sys.argv[1:] through the runtime's
    // one process-argument authority, so a non-UTF-8 argument is the same
    // loud exit as `sys.argv` gives, never a panic (Devin review on #340,
    // round 2); a valid run prints the parsed value.
    use std::os::unix::ffi::OsStrExt;
    let scratch = Scratch::new("runargparse");
    let file = scratch.path().join("cli.py");
    fs::write(
        &file,
        concat!(
            "import argparse\n",
            "\n",
            "\n",
            "def main() -> None:\n",
            "    parser = argparse.ArgumentParser(prog=\"cli\")\n",
            "    parser.add_argument(\"name\")\n",
            "    args = parser.parse_args()\n",
            "    print(\"hello\", args.name)\n",
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
        .arg("cli.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .arg("--")
        .arg("world")
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&output.stdout), "hello world\n", "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0));
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg("cli.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .arg("--")
        .arg(std::ffi::OsStr::from_bytes(b"w\xf6rld"))
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    assert_eq!(output.status.code(), Some(1), "stdout: {}", String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("rython: sys.argv[1] is not valid UTF-8"), "stderr: {}", stderr);
    assert!(!stderr.contains("panicked"), "stderr: {}", stderr);
}
