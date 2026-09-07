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
    assert!(
        scratch.path().join("elsewhere").join(rypip::host_triple().unwrap()).join("release").exists(),
        "the build is for the host, under CARGO_TARGET_DIR"
    );

    // The build is FOR THE HOST: a `CARGO_BUILD_TARGET` selecting a cross
    // target (not installed here — cargo fails with "can't find crate for
    // core" when it is honored) does not stop `rypip run`, which runs
    // programs on this machine as `python` does.
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg("hello.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(&out)
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .env("CARGO_BUILD_TARGET", "aarch64-unknown-linux-gnu")
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "args ['hello.py']\n",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(3));

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

#[test]
fn rypip_run_refuses_an_output_that_is_not_its_own() {
    // Round 4 of the review on #340: `run` regenerates its work dir's
    // `src/` from scratch, so `--out` must be its own directory. A
    // src-layout project run as `rypip run . --out .` would have erased its
    // own sources: refused loudly, sources intact; so is an output that
    // CONTAINS the project, and a non-empty directory that is not a crate
    // rypip generated. A separate directory works (python3 prints `proj`).
    let scratch = Scratch::new("runown");
    let proj = scratch.path().join("proj");
    let pkg = proj.join("src").join("proj");
    fs::create_dir_all(&pkg).unwrap();
    fs::write(proj.join("pyproject.toml"), "[project]\nname = \"proj\"\nversion = \"0.1.0\"\n").unwrap();
    fs::write(pkg.join("__init__.py"), "").unwrap();
    fs::write(pkg.join("__main__.py"), "print(\"proj\")\n").unwrap();
    let run = |package: &str, out: &std::path::Path, cwd: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_rypip"))
            .arg("run")
            .arg(package)
            .arg("--no-deps")
            .arg("--out")
            .arg(out)
            .current_dir(cwd)
            .env_remove("RUSTFLAGS")
            .output()
            .expect("running rypip")
    };
    // `rypip run . --out .` inside the project.
    let output = run(".", std::path::Path::new("."), &proj);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("holds the program's sources"), "stderr: {}", stderr);
    assert!(pkg.join("__main__.py").is_file(), "the sources survive");
    // An output directory that contains the project.
    let output = run("proj", std::path::Path::new("."), scratch.path());
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("holds the program's sources"), "stderr: {}", stderr);
    assert!(pkg.join("__main__.py").is_file());
    // A non-empty directory that is not a crate rypip generated.
    let stray = scratch.path().join("stray");
    fs::create_dir_all(stray.join("src")).unwrap();
    fs::write(stray.join("src").join("notes.txt"), "mine").unwrap();
    let output = run("proj", &stray, scratch.path());
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not a crate rypip generated"), "stderr: {}", stderr);
    assert_eq!(fs::read_to_string(stray.join("src").join("notes.txt")).unwrap(), "mine");
    // `..` through a component that does not exist yet resolves to the
    // project itself: refused the same way, sources intact (round 5).
    let output = run("proj", &proj.join("nonexistent").join(".."), scratch.path());
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("holds the program's sources"), "stderr: {}", stderr);
    assert!(pkg.join("__main__.py").is_file());
    assert!(!proj.join("nonexistent").exists());
    // A forged marker admits the directory, but a run deletes only the
    // files it wrote last time — a file rypip did not write survives.
    let forged = scratch.path().join("forged");
    fs::create_dir_all(forged.join("src")).unwrap();
    fs::write(forged.join("Cargo.toml"), format!("{}\n[package]\nname = \"x\"\n", rypip::convert::GENERATED_MANIFEST_HEADER)).unwrap();
    fs::write(forged.join("src").join("precious.rs"), "// mine\n").unwrap();
    let output = run("proj", &forged, scratch.path());
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&output.stdout), "proj\n", "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(fs::read_to_string(forged.join("src").join("precious.rs")).unwrap(), "// mine\n");
    // A separate directory: the src-layout project runs.
    let output = run("proj", &scratch.path().join("crate"), scratch.path());
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&output.stdout), "proj\n", "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0));
}

#[cfg(unix)]
#[test]
fn the_lock_follows_the_physical_output_directory() {
    // Round 4 of the review on #340: the lock's identity is the output's
    // PHYSICAL path (the canonical nearest existing ancestor plus the
    // components still to be created), so a symlinked alias of the same
    // directory takes the same lock, before and after the directory exists;
    // and two runs through the two spellings serialize on it.
    let scratch = Scratch::new("runalias");
    let real = scratch.path().join("real");
    fs::create_dir_all(&real).unwrap();
    let alias = scratch.path().join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    let out_real = real.join("out");
    let out_alias = alias.join("out");
    let lock_real = rypip::work_dir_lock_path(&out_real).unwrap();
    let lock_alias = rypip::work_dir_lock_path(&out_alias).unwrap();
    assert_eq!(lock_real, lock_alias, "before the directory exists");
    assert!(lock_real.to_string_lossy().ends_with("/real/out.lock"), "{}", lock_real.display());
    fs::create_dir_all(&out_real).unwrap();
    assert_eq!(rypip::work_dir_lock_path(&out_real).unwrap(), rypip::work_dir_lock_path(&out_alias).unwrap());

    fs::write(
        scratch.path().join("hello.py"),
        concat!(
            "def main() -> None:\n",
            "    print(\"alias\")\n",
            "\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let spawn = |out: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_rypip"))
            .arg("run")
            .arg("hello.py")
            .arg("--no-deps")
            .arg("--out")
            .arg(out)
            .current_dir(scratch.path())
            .env_remove("RUSTFLAGS")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawning rypip")
    };
    let (a, b) = (spawn(&out_real), spawn(&out_alias));
    let (oa, ob) = (a.wait_with_output().unwrap(), b.wait_with_output().unwrap());
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&oa.stdout), "alias\n", "stderr: {}", String::from_utf8_lossy(&oa.stderr));
    assert_eq!(String::from_utf8_lossy(&ob.stdout), "alias\n", "stderr: {}", String::from_utf8_lossy(&ob.stderr));
}

#[cfg(unix)]
#[test]
fn argparse_with_an_explicit_list_never_reads_the_process_arguments() {
    // Round 4 of the review on #340: `parse_args([...])` with the default
    // prog needs argv[0] only, read through `sys.argv_at(0)`, so a
    // non-UTF-8 process argument the program never reads does not abort
    // it — python3 prints `hello world` (verified) and so does this.
    use std::os::unix::ffi::OsStrExt;
    let scratch = Scratch::new("runexplicit");
    fs::write(
        scratch.path().join("cli.py"),
        concat!(
            "import argparse\n",
            "\n",
            "\n",
            "def main() -> None:\n",
            "    parser = argparse.ArgumentParser()\n",
            "    parser.add_argument(\"name\")\n",
            "    args = parser.parse_args([\"world\"])\n",
            "    print(\"hello\", args.name)\n",
            "\n",
            "\n",
            "if __name__ == \"__main__\":\n",
            "    main()\n",
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("run")
        .arg("cli.py")
        .arg("--no-deps")
        .arg("--out")
        .arg(scratch.path().join("crate"))
        .arg("--")
        .arg(std::ffi::OsStr::from_bytes(b"\xff"))
        .current_dir(scratch.path())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip");
    // Verified against python3.
    assert_eq!(String::from_utf8_lossy(&output.stdout), "hello world\n", "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(0));
}
