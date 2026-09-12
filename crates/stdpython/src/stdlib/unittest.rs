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
use crate::{PyException, PyValue, Truthy};

/// A failed `assertX` — a CPython `AssertionError`.
fn assertion_failed(msg: alloc::string::String) -> Result<(), PyException> {
    Err(PyException::new("AssertionError", msg))
}

/// `self.assertEqual(a, b)` — raise AssertionError when the two values are
/// not equal under CPython's `==` (including int/float/bool cross-type).
pub fn assert_eq(a: &PyValue, b: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if crate::py_value_eq(a, b) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!("{} != {}", crate::py_display(a), crate::py_display(b)))
    } else {
        assertion_failed(format!(
            "{}: {} != {}",
            ctx,
            crate::py_display(a),
            crate::py_display(b)
        ))
    }
}

/// `self.assertTrue(x)` — raise AssertionError when x is falsy.
pub fn assert_true(x: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if Truthy::is_truthy(x) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!("{} is not true", crate::py_display(x)))
    } else {
        assertion_failed(format!("{}: {} is not true", ctx, crate::py_display(x)))
    }
}

/// `self.assertFalse(x)` — raise AssertionError when x is truthy.
pub fn assert_false(x: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if !Truthy::is_truthy(x) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!("{} is not false", crate::py_display(x)))
    } else {
        assertion_failed(format!("{}: {} is not false", ctx, crate::py_display(x)))
    }
}

pub fn main() -> Result<(), PyException> {
    Err(PyException::new(
        "NotImplementedError",
        "rython: unittest.main() was not lowered to a test runner for this \
         module (the unittest runner is still being built); rython refuses \
         to pretend the tests ran"
            .to_string(),
    ))
}