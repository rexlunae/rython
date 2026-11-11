//! Standard output across the JIT boundary (issue #166, docs/jit.md "The
//! boundary"): the host and a loaded library each have their own `std`
//! and so their own stdout buffer. The host flushes before every call and
//! every shim flushes the library's before returning, so output keeps
//! program order — partial lines included — exactly as one CPython
//! process prints it.
//!
//! `harness = false`: the parent re-runs this binary as the host program
//! and compares the host's whole stdout, which libtest's own output would
//! otherwise share.

mod common;

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use rypip::convert::ConvertOptions;
use rython_jit::Value;

const CHILD: &str = "RYTHON_JIT_INTERLEAVE_MODULE";

const CHATTER: &str = "def chatter(n: int) -> int:\n    for i in range(n):\n        print(\"lib\", i, end=\" \")\n    return n * 10\n";

fn main() {
    match std::env::var_os(CHILD) {
        Some(module) => host(Path::new(&module)),
        None => parent(),
    }
}

/// The host program: the Rust spelling of
///
/// ```python
/// import chatter
/// print("host before", end=" ")
/// r = chatter.chatter(2)
/// print("host after", repr(r))
/// print("host again", end=" ")
/// chatter.chatter(1)
/// print("done")
/// ```
fn host(module: &Path) {
    let cache = module.parent().unwrap().join("jit-cache");
    let jit = rypip::jit::new_jit(ConvertOptions::default(), Some(&cache)).expect("setting up the JIT");
    let chatter = jit.compile_file("chatter", module).expect("compiling chatter.py");
    print!("host before ");
    let r = chatter.call("chatter", &[Value::Int(2)]).expect("calling chatter");
    println!("host after {}", r.repr());
    print!("host again ");
    chatter.call("chatter", &[Value::Int(1)]).expect("calling chatter");
    println!("done");
    std::io::stdout().flush().unwrap();
}

fn parent() {
    let scratch = common::Scratch::new("jit-interleave");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    }
    let module = scratch.path().join("chatter.py");
    fs::write(&module, CHATTER).unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .env(CHILD, &module)
        .env_remove("RUSTFLAGS")
        .output()
        .expect("running the host");
    assert!(
        output.status.success(),
        "the host failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Verified against python3 (the program in `host`'s doc comment, with
    // chatter.py beside it).
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "host before lib 0 lib 1 host after 20\nhost again lib 0 done\n"
    );
    println!("test jit_interleave ... ok");
}
