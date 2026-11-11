//! The JIT's M1 gate (issue #166, docs/jit.md): `rypip jit call` and
//! `rypip jit run` against transcripts of real python3 runs, typed calls
//! through rython-jit, the cache, and every loud failure.
//!
//! Every test shares one cache (and so one cargo target directory) under
//! a private temporary directory, so the runtime builds once for
//! the whole file; the cache's own locks serialize the builds of tests
//! running in parallel.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::Scratch;
use rypip::convert::ConvertOptions;
use rython_jit::{
    Built, CacheOutcome, Compiler, Converted, Jit, JitConfig, JitError, SourceModule, Value,
};

/// Generated crates follow the warning mode, not the environment: CI's
/// `RUSTFLAGS=-D warnings` would turn a lint the source Python earns into
/// a build failure (the convert and run tests clear it the same way).
fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: every test in this file calls setup() before anything
        // else, so no other thread of this process reads the environment
        // while the variable is removed.
        unsafe { std::env::remove_var("RUSTFLAGS") };
    });
}

/// XDG_CACHE_HOME for the CLI tests; the cache is `<this>/rython/jit`.
fn xdg_cache_home() -> PathBuf {
    std::env::temp_dir().join(format!("rython-jit-tests-{}", std::process::id()))
}

fn cache_dir() -> PathBuf {
    xdg_cache_home().join("rython").join("jit")
}

fn jit() -> Jit {
    setup();
    rypip::jit::new_jit(ConvertOptions::default(), Some(&cache_dir())).expect("setting up the JIT")
}

fn rypip_jit(dir: &Path, args: &[&str]) -> Output {
    setup();
    Command::new(env!("CARGO_BIN_EXE_rypip"))
        .arg("jit")
        .args(args)
        .current_dir(dir)
        .env("XDG_CACHE_HOME", xdg_cache_home())
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running rypip jit")
}

/// stderr without the traceback's frame lines (rython has no frames: its
/// traceback is the header and the final line, docs/spec.md §12.3).
fn without_frames(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .filter(|line| !line.starts_with("  "))
        .map(|line| format!("{}\n", line))
        .collect()
}

const CALLS: &str = r#"import sys


class Overdrawn(Exception):
    pass


def add(a: int, b: int) -> int:
    return a + b


def neg(a: int) -> int:
    return -a


def ratio(x: float, y: float) -> float:
    return x / y


def big(x: float) -> float:
    return x * 1e16


def shout(word: str) -> str:
    print("shouting", word)
    return word.upper() + "!"


def quote(s: str) -> str:
    return "it's " + s


def negate(flag: bool) -> bool:
    return not flag


def report() -> None:
    print("report", end="")


def check(n: int) -> int:
    if n < 0:
        raise ValueError("negative: " + str(n))
    return n


def withdraw(amount: int) -> int:
    raise Overdrawn("short by", amount)


def leave(code: int) -> None:
    sys.exit(code)


def complain(text: str) -> None:
    sys.exit(text)


def total(xs: list[int]) -> int:
    return sum(xs)


def argv_text() -> str:
    return str(sys.argv)
"#;

fn write_calls(scratch: &Scratch) -> PathBuf {
    let file = scratch.path().join("calls.py");
    fs::write(&file, CALLS).unwrap();
    file
}

#[test]
fn jit_call_marshals_each_m1_type_as_python3_does() {
    let scratch = Scratch::new("jit-call-types");
    write_calls(&scratch);
    // Each row: the arguments, and what
    // `python3 -c "import calls; print(repr(calls.F(ARGS)))"` printed.
    // Verified against python3.
    for (args, stdout) in [
        (&["add", "2", "40"][..], "42\n"),
        (&["neg", "-5"][..], "5\n"),
        (&["ratio", "1.0", "3.0"][..], "0.3333333333333333\n"),
        (&["big", "1.5"][..], "1.5e+16\n"),
        (&["shout", "'hey'"][..], "shouting hey\n'HEY!'\n"),
        (&["quote", "'x'"][..], "\"it's x\"\n"),
        (&["negate", "False"][..], "True\n"),
        // The library's partial line is flushed before the host prints.
        (&["report"][..], "reportNone\n"),
        // sys.argv is `python3 -c`'s, not rypip's.
        (&["argv_text"][..], "\"['-c']\"\n"),
    ] {
        let mut argv = vec!["call", "calls.py"];
        argv.extend_from_slice(args);
        let output = rypip_jit(scratch.path(), &argv);
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            stdout,
            "{:?}: stderr: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.status.code(), Some(0), "{:?}", args);
        assert_eq!(String::from_utf8_lossy(&output.stderr), "", "{:?}", args);
    }
}

#[test]
fn jit_call_exceptions_and_system_exit_match_python3() {
    let scratch = Scratch::new("jit-call-raise");
    write_calls(&scratch);
    // stderr without frames, and the exit status, of
    // `python3 -c "import calls; print(repr(calls.F(ARGS)))"`: a builtin
    // exception, a class of the imported module (named module-qualified),
    // SystemExit with an int code, and with a message.
    // Verified against python3.
    for (args, stderr, status) in [
        (
            &["check", "-1"][..],
            "Traceback (most recent call last):\nValueError: negative: -1\n",
            1,
        ),
        (
            &["withdraw", "7"][..],
            "Traceback (most recent call last):\ncalls.Overdrawn: ('short by', 7)\n",
            1,
        ),
        (&["leave", "3"][..], "", 3),
        (&["complain", "'bye'"][..], "bye\n", 1),
    ] {
        let mut argv = vec!["call", "calls.py"];
        argv.extend_from_slice(args);
        let output = rypip_jit(scratch.path(), &argv);
        assert_eq!(String::from_utf8_lossy(&output.stdout), "", "{:?}", args);
        assert_eq!(without_frames(&output.stderr), stderr, "{:?}", args);
        assert_eq!(output.status.code(), Some(status), "{:?}", args);
    }
}

#[test]
fn typed_calls_cross_the_boundary_and_exceptions_are_rebuilt_in_the_host() {
    let scratch = Scratch::new("jit-typed");
    let file = write_calls(&scratch);
    let jit = jit();
    let module = jit.compile_file("calls", &file).expect("compiling calls.py");
    assert_eq!(module.name(), "calls");
    // python3: calls.add(2, 40) == 42; calls.ratio(1.0, 3.0) == 1.0/3.0;
    // calls.negate(False) is True; calls.quote('x') == "it's x".
    assert_eq!(module.call_as::<i64>("add", &[Value::Int(2), Value::Int(40)]).unwrap(), 42);
    assert_eq!(
        module.call_as::<f64>("ratio", &[Value::Float(1.0), Value::Float(3.0)]).unwrap(),
        1.0 / 3.0
    );
    assert!(module.call_as::<bool>("negate", &[Value::Bool(false)]).unwrap());
    assert_eq!(module.call_as::<String>("quote", &["x".into()]).unwrap(), "it's x");
    assert_eq!(module.call("leave", &[Value::Int(0)]).unwrap_err().to_string(), "SystemExit(0)");
    assert_eq!(
        module.functions().iter().find(|f| f.name == "add").unwrap().signature(),
        "add(a: int, b: int) -> int"
    );

    // A user exception: type(e).__name__ == 'Overdrawn', str(e) ==
    // "('short by', 7)", repr(e) == "Overdrawn('short by', 7)", and
    // isinstance(e, Exception). A builtin: repr(e) ==
    // "ValueError('negative: -1')". Verified against python3.
    match module.call("withdraw", &[Value::Int(7)]) {
        Err(JitError::Python(e)) => {
            use rython_jit::stdpython::PyRepr;
            assert_eq!(e.type_name(), "Overdrawn");
            assert_eq!(e.message, "('short by', 7)");
            assert_eq!(e.py_repr(), "Overdrawn('short by', 7)");
            assert!(e.matches("Exception"));
            assert!(!e.matches("ValueError"));
        }
        other => panic!("expected a Python exception, got {:?}", other.map(|v| v.repr())),
    }
    match module.call("check", &[Value::Int(-1)]) {
        Err(JitError::Python(e)) => {
            use rython_jit::stdpython::PyRepr;
            assert_eq!(e.py_repr(), "ValueError('negative: -1')");
            assert!(e.matches("ValueError") && e.matches("Exception"));
        }
        other => panic!("expected a Python exception, got {:?}", other.map(|v| v.repr())),
    }
    // SystemExit comes back to the host; the process goes on.
    match module.call("leave", &[Value::Int(3)]) {
        Err(JitError::Exit { code: 3, message: None }) => {}
        other => panic!("expected SystemExit(3), got {:?}", other.map(|v| v.repr())),
    }
    match module.call("complain", &["bye".into()]) {
        Err(JitError::Exit { code: 1, message: Some(m) }) if m == "bye" => {}
        other => panic!("expected SystemExit('bye'), got {:?}", other.map(|v| v.repr())),
    }
}

#[test]
fn values_the_boundary_cannot_carry_are_loud_marshal_errors_naming_the_type() {
    let scratch = Scratch::new("jit-marshal");
    let file = write_calls(&scratch);
    let jit = jit();
    let module = jit.compile_file("calls", &file).expect("compiling calls.py");
    let marshal = |result: Result<Value, JitError>| match result {
        Err(JitError::Marshal(message)) => message,
        other => panic!("expected a marshal error, got {:?}", other.map(|v| v.repr())),
    };
    // A parameter of a type M1 does not marshal: no shim, and the call
    // names the annotation.
    let message = marshal(module.call("total", &[Value::Int(1)]));
    assert!(message.contains("`calls.total`") && message.contains("`list[int]`"), "{message}");
    // The JIT never converts an argument: CPython would run `ratio` on
    // the int itself (ratio(1, 2) is 0.5 either way, but `x: float`
    // holding an int is not what the compiled function takes).
    let message = marshal(module.call("ratio", &[Value::Int(1), Value::Float(2.0)]));
    assert!(message.contains("annotated `float`") && message.contains("`int` was passed"), "{message}");
    // bool is an int subclass under CPython, and still not an int here.
    let message = marshal(module.call("add", &[Value::Bool(true), Value::Int(1)]));
    assert!(message.contains("`bool` was passed"), "{message}");
    let message = marshal(module.call("add", &[Value::Int(1)]));
    assert!(message.contains("takes 2 arguments") && message.contains("1 was passed"), "{message}");
    // A class is not a function M1 calls; a missing name is not one either.
    assert!(matches!(module.call("Overdrawn", &[]), Err(JitError::Unsupported(_))));
    assert!(matches!(module.call("nonexistent", &[]), Err(JitError::Unsupported(_))));

    // The command line refuses a literal of another type before compiling.
    let output = rypip_jit(scratch.path(), &["call", "calls.py", "total", "[1, 2]"]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("is a `list`, which M1 cannot marshal"), "{stderr}");
    let output = rypip_jit(scratch.path(), &["call", "calls.py", "total", "1"]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("`list[int]`, a type M1 cannot marshal"), "{stderr}");
}

#[test]
fn jit_run_matches_python3_for_whole_programs() {
    let scratch = Scratch::new("jit-run");
    let programs: &[(&str, &str)] = &[
        (
            "argv_prog.py",
            "import sys\n\n\ndef main() -> None:\n    print(\"args\", sys.argv)\n    print(\"count\", len(sys.argv) - 1)\n    sys.exit(len(sys.argv) - 1)\n\n\nif __name__ == \"__main__\":\n    main()\n",
        ),
        (
            "crash.py",
            "def parse(text: str) -> int:\n    return int(text)\n\n\ndef main() -> None:\n    print(\"before\")\n    print(parse(\"12\"))\n    print(parse(\"x1\"))\n    print(\"after\")\n\n\nif __name__ == \"__main__\":\n    main()\n",
        ),
        (
            "bank.py",
            "class Account:\n    def __init__(self, owner: str, balance: int) -> None:\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount: int) -> None:\n        self.balance += amount\n\n\ndef main() -> None:\n    accounts = {\"ann\": Account(\"ann\", 10), \"bob\": Account(\"bob\", 5)}\n    for name in [\"ann\", \"bob\", \"ann\"]:\n        accounts[name].deposit(7)\n    total = 0\n    for name, acct in accounts.items():\n        print(name, acct.balance)\n        total += acct.balance\n    print(\"total\", total)\n    squares = [n * n for n in range(5) if n % 2 == 0]\n    print(squares, sum(squares) / len(squares))\n\n\nif __name__ == \"__main__\":\n    main()\n",
        ),
        (
            "quit_msg.py",
            "import sys\n\nprint(\"partial\", end=\"\")\nprint(\" line\")\nprint(\"no newline at exit\", end=\"\")\nsys.exit(\"fatal: giving up\")\n",
        ),
        (
            "cli.py",
            "import argparse\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description=\"greeter\")\n    parser.add_argument(\"name\")\n    parser.add_argument(\"--times\", type=int, default=1)\n    args = parser.parse_args()\n    for _ in range(args.times):\n        print(\"hello\", args.name)\n\n\nif __name__ == \"__main__\":\n    main()\n",
        ),
        (
            "bare_main.py",
            "def helper(n: int) -> int:\n    return n * 2\n\n\ndef main() -> None:\n    print(\"main ran\", helper(21))\n\n\nmain()\n",
        ),
    ];
    for (name, source) in programs {
        fs::write(scratch.path().join(name), source).unwrap();
    }
    // `python3 PROGRAM ARGS...`: stdout, stderr without frames, status.
    // Verified against python3.
    for (args, stdout, stderr, status) in [
        (
            &["argv_prog.py", "one", "-two", "3"][..],
            "args ['argv_prog.py', 'one', '-two', '3']\ncount 3\n",
            "",
            3,
        ),
        (
            &["crash.py"][..],
            "before\n12\n",
            "Traceback (most recent call last):\nValueError: invalid literal for int() with base 10: 'x1'\n",
            1,
        ),
        (
            &["bank.py"][..],
            "ann 24\nbob 12\ntotal 36\n[0, 4, 16] 6.666666666666667\n",
            "",
            0,
        ),
        (
            &["quit_msg.py"][..],
            "partial line\nno newline at exit",
            "fatal: giving up\n",
            1,
        ),
        (&["cli.py", "world", "--times", "2"][..], "hello world\nhello world\n", "", 0),
        // argparse's default prog is sys.argv[0], read through the seam.
        (
            &["cli.py"][..],
            "",
            "usage: cli.py [-h] [--times TIMES] name\ncli.py: error: the following arguments are required: name\n",
            2,
        ),
        (&["bare_main.py"][..], "main ran 42\n", "", 0),
    ] {
        let mut argv = vec!["run"];
        argv.extend_from_slice(args);
        let output = rypip_jit(scratch.path(), &argv);
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            stdout,
            "{:?}: stderr: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(without_frames(&output.stderr), stderr, "{:?}", args);
        assert_eq!(output.status.code(), Some(status), "{:?}", args);
    }

    // Importing bare_main.py runs its final `main()` under CPython; the
    // library compiles that call as the program entry, so calling into the
    // module is refused rather than run without it.
    let output = rypip_jit(scratch.path(), &["call", "bare_main.py", "helper", "2"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ends with a bare `main()` call"), "{stderr}");
}

/// Counts the compiler's conversions and builds: a cache hit runs neither.
struct Counting {
    inner: rypip::jit::RypipCompiler,
    converts: AtomicUsize,
    builds: AtomicUsize,
}

impl Compiler for Counting {
    fn fingerprint(&self) -> Result<String, JitError> {
        self.inner.fingerprint()
    }

    fn convert(&self, module: &SourceModule<'_>, crate_dir: &Path) -> Result<Converted, JitError> {
        self.converts.fetch_add(1, Ordering::SeqCst);
        self.inner.convert(module, crate_dir)
    }

    fn build(&self, crate_dir: &Path, crate_name: &str, target_dir: &Path) -> Result<Built, JitError> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        self.inner.build(crate_dir, crate_name, target_dir)
    }
}

#[test]
fn a_second_compile_is_a_cache_hit_that_runs_neither_the_converter_nor_cargo() {
    setup();
    let counting = Arc::new(Counting {
        inner: rypip::jit::RypipCompiler::new(ConvertOptions::default()),
        converts: AtomicUsize::new(0),
        builds: AtomicUsize::new(0),
    });
    let new_jit = || Jit::new(JitConfig::new(counting.clone()).with_cache_dir(cache_dir())).unwrap();
    let source = "def cube(n: int) -> int:\n    return n * n * n\n";
    let first = new_jit().compile("cubes", source).expect("compiling cubes");
    if first.cache_outcome() == CacheOutcome::Built {
        // A fresh build of a warning-clean module, boundary included,
        // prints no diagnostics.
        assert_eq!(first.build_diagnostics(), "");
    }
    let (converts, builds) = (counting.converts.load(Ordering::SeqCst), counting.builds.load(Ordering::SeqCst));
    // The same JIT, and a new one (a new process's view of the cache).
    for jit in [new_jit(), new_jit()] {
        let again = jit.compile("cubes", source).expect("compiling cubes again");
        assert_eq!(again.cache_outcome(), CacheOutcome::Hit);
        assert_eq!(jit.builds(), 0);
        assert_eq!(again.call_as::<i64>("cube", &[Value::Int(3)]).unwrap(), 27);
    }
    assert_eq!(counting.converts.load(Ordering::SeqCst), converts, "a hit ran the converter");
    assert_eq!(counting.builds.load(Ordering::SeqCst), builds, "a hit ran cargo");
    // A different source is a different entry.
    let other = new_jit().compile("cubes", "def cube(n: int) -> int:\n    return n * n * n * 1\n").unwrap();
    assert_eq!(other.call_as::<i64>("cube", &[Value::Int(2)]).unwrap(), 8);
    assert_ne!(other.library_path(), first.library_path());
}

#[test]
fn an_untrusted_cache_entry_is_rebuilt_never_loaded() {
    let jit = jit();
    let source = "def half(x: float) -> float:\n    return x / 2\n";
    let module = jit.compile("halves", source).expect("compiling halves");
    let entry = module.library_path().parent().unwrap().to_path_buf();
    let pinned = fs::read(entry.join("crate/Cargo.lock")).unwrap();
    // A record whose size no longer matches its library.
    let record = entry.join("entry");
    let text = fs::read_to_string(&record).unwrap();
    let corrupted: String = text
        .lines()
        .map(|line| if line.starts_with("size ") { "size 1".to_string() } else { line.to_string() })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&record, corrupted).unwrap();
    let rebuilt = jit.compile("halves", source).expect("rebuilding halves");
    assert_eq!(rebuilt.cache_outcome(), CacheOutcome::Built);
    assert_eq!(fs::read(entry.join("crate/Cargo.lock")).unwrap(), pinned);
    assert_ne!(rebuilt.library_path(), module.library_path());
    assert_eq!(rebuilt.call_as::<f64>("half", &[Value::Float(3.0)]).unwrap(), 1.5);
    // And the rebuilt entry is trusted again.
    assert_eq!(jit.compile("halves", source).unwrap().cache_outcome(), CacheOutcome::Hit);

    // Replacing a library with equal-length bytes must be caught BEFORE
    // dlopen. Rename a replacement inode so an already mapped library is
    // unaffected by this deliberate corruption.
    let replacement = entry.join("corrupt.tmp");
    let mut bytes = fs::read(rebuilt.library_path()).unwrap();
    bytes[0] ^= 0xff;
    fs::write(&replacement, &bytes).unwrap();
    fs::rename(&replacement, rebuilt.library_path()).unwrap();
    let verified = jit.compile("halves", source).unwrap();
    assert_eq!(verified.cache_outcome(), CacheOutcome::Built);
    assert_eq!(verified.call_as::<f64>("half", &[Value::Float(6.0)]).unwrap(), 3.0);
}

#[test]
fn cache_hits_cannot_bypass_incompatible_conversion_options() {
    let source = "def value() -> int:\n    return 1\n";
    jit().compile("option_validation", source).unwrap();
    for options in [
        ConvertOptions { no_deps: true, ..Default::default() },
        ConvertOptions { cdylib: Some(rypip::convert::CdylibTarget { module_name: "overridden".into() }), ..Default::default() },
    ] {
        let equivalent = rypip::jit::new_jit(options, Some(&cache_dir())).unwrap();
        assert_eq!(equivalent.compile("option_validation", source).unwrap().cache_outcome(), CacheOutcome::Hit);
        assert_eq!(equivalent.builds(), 0);
    }
    for options in [
        ConvertOptions { no_std: true, ..Default::default() },
        ConvertOptions { pyo3: true, ..Default::default() },
        ConvertOptions { kernel_module: true, ..Default::default() },
        ConvertOptions { driver: true, ..Default::default() },
        ConvertOptions { rust_for_linux: true, ..Default::default() },
    ] {
        let invalid = rypip::jit::new_jit(options, Some(&cache_dir())).unwrap();
        let error = invalid.compile("option_validation", source).err().expect("incompatible options on warm cache");
        assert!(matches!(error, JitError::Unsupported(_)), "{error}");
        assert_eq!(invalid.builds(), 0);
    }
}

#[test]
fn a_programs_entry_runs_once_and_never_shares_its_module_with_calls() {
    let jit = jit();
    // Quiet programs: their output would land on this test's stdout.
    let program = "import sys\n\n\ndef code() -> int:\n    return len(sys.argv) + 1\n\n\nif __name__ == \"__main__\":\n    sys.exit(code())\n";
    let module = jit.compile("quiet_exit", program).expect("compiling quiet_exit");
    // sys.argv is what the host installed: three arguments, exit 4.
    let argv = ["quiet_exit.py", "a", "b"].map(String::from);
    assert_eq!(module.run_entry(&argv).unwrap(), 4);
    assert!(matches!(module.run_entry(&argv), Err(JitError::EntryAlreadyRun { .. })));
    // After the program ran as __main__, an import would be a second,
    // fresh module under CPython: refused.
    assert!(matches!(module.call("code", &[]), Err(JitError::Unsupported(_))));

    let module = jit
        .compile("quiet_calls", "def code() -> int:\n    return 5\n\n\nif __name__ == \"__main__\":\n    code()\n")
        .expect("compiling quiet_calls");
    assert_eq!(module.call_as::<i64>("code", &[]).unwrap(), 5);
    match module.run_entry(&["quiet_calls.py".to_string()]) {
        Err(JitError::Unsupported(message)) => assert!(message.contains("already imported"), "{message}"),
        other => panic!("expected a refusal, got {:?}", other),
    }

    // A module with no entry point cannot run, and says so.
    let library = jit.compile("defs_only", "def one() -> int:\n    return 1\n").unwrap();
    match library.run_entry(&["defs_only.py".to_string()]) {
        Err(JitError::Conversion(message)) => assert!(message.contains("has no entry point"), "{message}"),
        other => panic!("expected a refusal, got {:?}", other),
    }
}

#[test]
fn an_unsupported_construct_is_the_converters_error_verbatim() {
    let scratch = Scratch::new("jit-refusal");
    let file = scratch.path().join("registry.py");
    fs::write(&file, "import sys\nprint(len(sys.modules))\n").unwrap();
    let package = rypip::discover(&file).unwrap();
    let converter = format!(
        "{:#}",
        rypip::convert(&package, &scratch.path().join("crate"), &ConvertOptions::default()).unwrap_err()
    );
    assert!(converter.contains("sys.modules access"), "{converter}");
    match jit().compile_file("registry", &file) {
        Err(JitError::Conversion(message)) => assert_eq!(message, converter),
        Err(other) => panic!("expected a conversion error, got {}", other),
        Ok(_) => panic!("expected a conversion error"),
    }
    // The command line reports it the same way, and runs nothing.
    let output = rypip_jit(scratch.path(), &["run", "registry.py"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("sys.modules access") && stderr.contains("refuses"), "{stderr}");
}

#[test]
fn the_command_line_refuses_what_m1_does_not_compile_and_clears_its_cache() {
    let scratch = Scratch::new("jit-cli");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    }
    // A package directory, a module beside a rython.toml, and a name
    // CPython cannot import are refused before anything is built.
    fs::create_dir_all(scratch.path().join("pkg")).unwrap();
    fs::write(scratch.path().join("pkg/__init__.py"), "").unwrap();
    let output = rypip_jit(scratch.path(), &["run", "pkg"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("is a directory"));
    fs::create_dir_all(scratch.path().join("with_manifest")).unwrap();
    fs::write(scratch.path().join("with_manifest/rython.toml"), "").unwrap();
    fs::write(scratch.path().join("with_manifest/m.py"), "print(1)\n").unwrap();
    let output = rypip_jit(scratch.path(), &["run", "with_manifest/m.py"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("rython.toml"));
    fs::write(scratch.path().join("my-mod.py"), "def f() -> int:\n    return 1\n").unwrap();
    let output = rypip_jit(scratch.path(), &["call", "my-mod.py", "f"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be imported under CPython"));

    // --clear-cache removes the cache it would use (a private one here,
    // so the shared cache stays warm for the other tests).
    let xdg = scratch.path().join("xdg");
    let cache = xdg.join("rython").join("jit");
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().recursive(true).mode(0o700).create(cache.join("entries")).unwrap();
    }
    fs::write(cache.join("entries").join("stale"), "x").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rypip"))
        .args(["jit", "--clear-cache"])
        .env("XDG_CACHE_HOME", &xdg)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(!cache.exists());
}

#[cfg(unix)]
#[test]
fn a_cache_directory_others_can_write_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    setup();
    let scratch = Scratch::new("jit-cache-mode");
    fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let dir = scratch.path().join("cache");
    fs::create_dir_all(&dir).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o777)).unwrap();
    match rypip::jit::new_jit(ConvertOptions::default(), Some(&dir)) {
        Err(JitError::Environment(message)) => assert!(message.contains("chmod 700"), "{message}"),
        Err(other) => panic!("expected an environment error, got {}", other),
        Ok(_) => panic!("a world-writable cache was accepted"),
    }
    // A fresh one is created private.
    let fresh = scratch.path().join("fresh/jit");
    rypip::jit::new_jit(ConvertOptions::default(), Some(&fresh)).unwrap();
    assert_eq!(fs::metadata(&fresh).unwrap().permissions().mode() & 0o777, 0o700);
}
