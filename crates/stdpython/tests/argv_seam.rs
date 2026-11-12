//! The program-arguments seam (issue #166, the JIT): `sys.argv` and
//! `argv_at` (argparse's default `prog`) read ONE source, which a host
//! installs before the program runs — the JIT's `ry_entry`, in the loaded
//! library's own copy of the runtime. The source is process-global and
//! fixed by its first use, so this file is its own test binary holding a
//! single test (another test reading `sys.argv` first would fix the
//! source to the process's arguments).
//!
//! The std tier only, like the `sys` module itself.
#![cfg(feature = "std")]

use stdpython::sys::{self, InstallArgsError};

#[test]
fn installed_program_arguments_are_what_sys_argv_and_argv_at_report() {
    // `python3 prog.py a -b` → `sys.argv == ['prog.py', 'a', '-b']`: the
    // script path as given first, then the program's arguments.
    let program = vec!["prog.py".to_string(), "a".to_string(), "-b".to_string()];
    assert_eq!(sys::install_program_args(program.clone()), Ok(()));
    // A second install would leave sys.argv (initialized once) reporting
    // the first program's arguments: refused, not ignored.
    assert_eq!(
        sys::install_program_args(vec!["other.py".to_string()]),
        Err(InstallArgsError::AlreadyInstalled)
    );
    // Both readers agree, element by element and past the end.
    assert_eq!(sys::argv_at(0).as_deref(), Some("prog.py"));
    assert_eq!(sys::argv_at(2).as_deref(), Some("-b"));
    assert_eq!(sys::argv_at(3), None);
    assert_eq!(*sys::argv, program);
    assert_eq!(sys::get_argv(), program);
}
