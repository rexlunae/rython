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

/// Box a lowered Python LIST (`Vec<T>`) as `PyValue::Tuple` — the boxed
/// model's documented list-as-tuple divergence — for the assert helpers.
///
/// Distinct from `PyValue::from`, which maps `Vec<u8>` to `Bytes` (a list of
/// small ints would be silently boxed as bytes). Codegen emits this only for
/// `Vec<T>` arguments whose element type converts into `PyValue`.
pub fn list_to_pyvalue<T: Into<PyValue>>(items: alloc::vec::Vec<T>) -> PyValue {
    PyValue::Tuple(alloc::sync::Arc::new(
        items.into_iter().map(Into::into).collect(),
    ))
}

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

/// `self.assertNotEqual(a, b)` — raise AssertionError when they ARE equal.
pub fn assert_not_eq(a: &PyValue, b: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if !crate::py_value_eq(a, b) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!("{} == {}", crate::py_display(a), crate::py_display(b)))
    } else {
        assertion_failed(format!(
            "{}: {} == {}",
            ctx,
            crate::py_display(a),
            crate::py_display(b)
        ))
    }
}

/// `self.assertIsNone(x)` — raise AssertionError when x is not None.
pub fn assert_is_none(x: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if crate::PyIsNone::py_is_none(x) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!("{} is not None", crate::py_display(x)))
    } else {
        assertion_failed(format!("{}: {} is not None", ctx, crate::py_display(x)))
    }
}

/// `self.assertIsNotNone(x)` — raise AssertionError when x is None.
pub fn assert_is_not_none(x: &PyValue, ctx: alloc::string::String) -> Result<(), PyException> {
    if !crate::PyIsNone::py_is_none(x) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed("unexpectedly None".to_string())
    } else {
        assertion_failed(format!("{ctx}: unexpectedly None"))
    }
}

/// `self.assertIn(member, container)` — raise AssertionError when the
/// container does not contain the member (CPython `member in container`).
pub fn assert_in(
    member: &PyValue,
    container: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    if crate::PyContains::py_contains(container, member) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!(
            "{} not found in {}",
            crate::py_display(member),
            crate::py_display(container)
        ))
    } else {
        assertion_failed(format!(
            "{}: {} not found in {}",
            ctx,
            crate::py_display(member),
            crate::py_display(container)
        ))
    }
}

/// `self.assertNotIn(member, container)` — raise AssertionError when the
/// container DOES contain the member.
pub fn assert_not_in(
    member: &PyValue,
    container: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    if !crate::PyContains::py_contains(container, member) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!(
            "{} unexpectedly found in {}",
            crate::py_display(member),
            crate::py_display(container)
        ))
    } else {
        assertion_failed(format!(
            "{}: {} unexpectedly found in {}",
            ctx,
            crate::py_display(member),
            crate::py_display(container)
        ))
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