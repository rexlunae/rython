//! Python unittest module — the test-runner harness (issue #334).
//!
//! This is the foundation of rython's `unittest` support. The runner's
//! job — discover the module's `*TestCase` classes, run each `test_*`
//! method, and report — is emitted per-module by codegen (it is the only
//! site that knows the concrete test classes and their methods), so the
//! runtime surface here is deliberately small.
//!
//! Until codegen wires a real runner for a module, `main()` returns a
//! LOUD error (never a silent pass): a test module whose `unittest.main()`
//! is not yet lowered by codegen must not pretend its tests ran.
use crate::PyException;
use alloc::string::String;

pub fn main() -> Result<(), PyException> {
    Err(PyException::new(
        "NotImplementedError",
        "rython: unittest.main() was not lowered to a test runner for this \
         module (the unittest runner is still being built); rython refuses \
         to pretend the tests ran"
            .to_string(),
    ))
}