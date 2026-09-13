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

/// `self.assertRaises(Exc, fn, *args)` — CPython's callable form: run
/// `fn(*args)` and assert it raised `expected`. The closure returns the
/// callee's `Result`, so a raising call is an `Err` whose type must match.
pub fn assert_raises<T, F>(expected: &str, run: F) -> Result<(), PyException>
where
    F: FnOnce() -> Result<T, PyException>,
{
    match run() {
        Err(e) if e.matches(expected) => Ok(()),
        // A DIFFERENT exception propagates, exactly as CPython lets it out of
        // assertRaises (the test then errors rather than failing an
        // assertion) — the unexpected exception surfaces, not a rewrite.
        Err(e) => Err(e),
        Ok(_) => Err(PyException::new(
            "AssertionError",
            format!("{expected} not raised"),
        )),
    }
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

/// Whether a boxed value is an instance of the given builtin type NAME
/// (`self.assertIsInstance(obj, int)`). `bool` is a subclass of `int`,
/// matching CPython; `list` matches a boxed tuple (the list-as-tuple
/// divergence, so `isinstance([…], list)` and `isinstance([…], tuple)`
/// cannot be told apart at the boxed level). Unknown / non-builtin type
/// names never match.
fn pyvalue_isinstance(v: &PyValue, t: &str) -> bool {
    match t {
        "int" => matches!(v, PyValue::Int(_) | PyValue::Bool(_)),
        "bool" => matches!(v, PyValue::Bool(_)),
        "float" => matches!(v, PyValue::Float(_)),
        "str" => matches!(v, PyValue::Str(_)),
        "bytes" => matches!(v, PyValue::Bytes(_)),
        "dict" => matches!(v, PyValue::Dict(_)),
        "tuple" => matches!(v, PyValue::Tuple(_)),
        "list" => matches!(v, PyValue::Tuple(_)),
        "NoneType" => matches!(v, PyValue::None_),
        _ => false,
    }
}

/// `self.assertIsInstance(obj, cls)` — raise AssertionError when obj is not
/// an instance of the builtin type NAME `cls`.
pub fn assert_is_instance(
    obj: &PyValue,
    type_name: &str,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    if pyvalue_isinstance(obj, type_name) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!(
            "{} is not an instance of {}",
            crate::py_display(obj),
            type_name
        ))
    } else {
        assertion_failed(format!(
            "{}: {} is not an instance of {}",
            ctx,
            crate::py_display(obj),
            type_name
        ))
    }
}

/// `self.assertNotIsInstance(obj, cls)` — raise AssertionError when obj IS
/// an instance of `cls`.
pub fn assert_not_is_instance(
    obj: &PyValue,
    type_name: &str,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    if !pyvalue_isinstance(obj, type_name) {
        Ok(())
    } else if ctx.is_empty() {
        assertion_failed(format!(
            "{} is an instance of {}",
            crate::py_display(obj),
            type_name
        ))
    } else {
        assertion_failed(format!(
            "{}: {} is an instance of {}",
            ctx,
            crate::py_display(obj),
            type_name
        ))
    }
}

/// `self.assertAlmostEqual(a, b, places=7, delta=None)`. Raise AssertionError
/// when the two numeric values differ. Semantics follow CPython:
///  - equal values (float `==`) always pass;
///  - with `delta` given, pass when `abs(a - b) <= delta`;
///  - else pass when `round(abs(a - b), places) == 0` (places default 7).
/// Values that are not numeric are a loud failure, never a silent pass.
pub fn assert_almost_eq(
    a: &PyValue,
    b: &PyValue,
    places: i64,
    delta: &Option<PyValue>,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    // The operands and the delta must all be numeric; any writer that passes
    // a non-number gets a LOUD failure, never a silent wrong result.
    let af = as_f64(a);
    let bf = as_f64(b);
    if let (Some(af), Some(bf)) = (af, bf) {
        // Resolve the boxed delta (if any) to an `Option<f64>`.
        let mut df: Option<f64> = None;
        if let Some(dv) = delta {
            df = as_f64(dv);
            if df.is_none() {
                return assertion_failed(format!(
                    "assertAlmostEqual: delta must be a number, got {}",
                    crate::py_display(dv)
                ));
            }
        }
        if af == bf {
            Ok(())
        } else {
            let diff = (af - bf).abs();
            let ok = match df {
                Some(d) => diff <= d,
                None => round_to_places(diff, places) == 0.0,
            };
            if ok {
                Ok(())
            } else {
                let detail = match df {
                    Some(d) => format!("within {} delta", crate::py_float_repr(d)),
                    None => format!("within {} places", places),
                };
                if ctx.is_empty() {
                    assertion_failed(format!(
                        "{} != {} {detail}",
                        crate::py_display(a),
                        crate::py_display(b)
                    ))
                } else {
                    assertion_failed(format!(
                        "{}: {} != {} {detail}",
                        ctx,
                        crate::py_display(a),
                        crate::py_display(b)
                    ))
                }
            }
        }
    } else {
        assertion_failed(format!(
            "assertAlmostEqual: operands must be numbers, got {} and {}",
            crate::py_display(a),
            crate::py_display(b)
        ))
    }
}

/// `self.assertNotAlmostEqual(a, b, places=7, delta=None)` — the negation of
/// `assert_almost_eq`: fails when the values ARE within tolerance.
pub fn assert_not_almost_eq(
    a: &PyValue,
    b: &PyValue,
    places: i64,
    delta: &Option<PyValue>,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    // Compute whether they are within tolerance; the helper's pass condition
    // is preserved here so the two stay in lockstep.
    let af = as_f64(a);
    let bf = as_f64(b);
    if let (Some(af), Some(bf)) = (af, bf) {
        // Resolve the boxed delta (if any) to an `Option<f64>`; a non-numeric
        // delta is a LOUD failure.
        let mut df: Option<f64> = None;
        if let Some(dv) = delta {
            df = as_f64(dv);
            if df.is_none() {
                return assertion_failed(format!(
                    "assertNotAlmostEqual: delta must be a number, got {}",
                    crate::py_display(dv)
                ));
            }
        }
        let within = if af == bf {
            true
        } else {
            let diff = (af - bf).abs();
            match df {
                Some(d) => diff <= d,
                None => round_to_places(diff, places) == 0.0,
            }
        };
        if !within {
            Ok(())
        } else {
            let detail = match df {
                Some(d) => format!("within {} delta", crate::py_float_repr(d)),
                None => format!("within {} places", places),
            };
            if ctx.is_empty() {
                assertion_failed(format!(
                    "{} == {} {detail}",
                    crate::py_display(a),
                    crate::py_display(b)
                ))
            } else {
                assertion_failed(format!(
                    "{}: {} == {} {detail}",
                    ctx,
                    crate::py_display(a),
                    crate::py_display(b)
                ))
            }
        }
    } else {
        assertion_failed(format!(
            "assertNotAlmostEqual: operands must be numbers, got {} and {}",
            crate::py_display(a),
            crate::py_display(b)
        ))
    }
}

/// Numeric value of `v` as a float, or `None` when `v` is not numeric.
///
/// `Int`/`Bool`/`Float` are numeric (int/bool coerce to float, matching
/// CPython's numeric tower). Everything else has no `__float__` in the subset
/// and so is not a number.
fn as_f64(v: &PyValue) -> Option<f64> {
    if let Some(i) = v.as_int() {
        Some(i as f64)
    } else if let Some(f) = v.as_float() {
        Some(f)
    } else if let Some(b) = v.as_bool() {
        Some(if b { 1.0 } else { 0.0 })
    } else {
        None
    }
}

/// CPython's `round(x, places)` for the `assertAlmostEqual` check: round
/// `abs(x)` to `places` decimal places and test it equals zero. Equivalent to
/// `round(abs(x) * 10^places) == 0`, i.e. `abs(x)` is below half of the last
/// digit. (At exactly the half boundary banker's rounding differs, but that
/// point is measure-zero in real assertions.)
fn round_to_places(x: f64, places: i64) -> f64 {
    // `10^places` via repeated multiply — avoids `f64::pow`, which is not
    // available in the alloc tier. `places` is small in practice (unittest's
    // default is 7). Clamp to a sane bound: beyond ~18 decimal places an f64
    // has no more representable digits, and a huge `places` would only spin
    // this loop.
    let p = if places > 18 { 18 } else { places };
    let mut scale: f64 = 1.0;
    for _ in 0..p {
        scale *= 10.0;
    }
    // Round to nearest integer (half away from zero, like `f64::round`); the
    // strict `<` in the caller keeps the half boundary out of the pass set.
    (x * scale).round()
}

/// CPython's `==`/`<`/etc. use the value protocol; for the boxed subset we
/// support NUMERIC ordering (int/float/bool, which compare across the numeric
/// tower like CPython) and STRING ordering (lexicographic). Anything else is
/// incomparable — return `None` and let the caller fail loudly rather than
/// guess.
///
/// Returns -1 / 0 / +1 for `a < b` / `a == b` / `a > b`, or `None` when the
/// two boxed values are not comparable in the subset.
fn py_boxed_cmp(a: &PyValue, b: &PyValue) -> Option<i64> {
    let af = as_f64(a);
    let bf = as_f64(b);
    if let (Some(af), Some(bf)) = (af, bf) {
        return if af < bf {
            Some(-1)
        } else if af > bf {
            Some(1)
        } else {
            Some(0)
        };
    }
    // String ordering (lexicographic by code point, like CPython str `<`).
    if let (Some(sa), Some(sb)) = (a.as_str(), b.as_str()) {
        return if sa < sb {
            Some(-1)
        } else if sa > sb {
            Some(1)
        } else {
            Some(0)
        };
    }
    None
}

/// One of `assertGreater`/`assertGreaterEqual`/`assertLess`/`assertLessEqual`.
/// `verdict(a, b)` returns whether the ord compare is a PASS. Neither the
/// comparands nor the boxed comparison are silently guessed: an incomparable
/// pair (mixed number/str, etc.) is a LOUD failure.
fn assert_ord(
    a: &PyValue,
    b: &PyValue,
    kind: &str,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    let pass = match py_boxed_cmp(a, b) {
        Some(c) => match kind {
            "assertGreater" => c > 0,
            "assertGreaterEqual" => c >= 0,
            "assertLess" => c < 0,
            "assertLessEqual" => c <= 0,
            _ => unreachable!(),
        },
        None => false,
    };
    // The CPython message shape: `{a!r} not {op} {b!r}` (the `!r` is the boxed
    // display; the ordering phrase embeds the second operand).
    let phrase = format!(
        "not {} {}",
        match kind {
            "assertGreater" => "greater than",
            "assertGreaterEqual" => "greater than or equal to",
            "assertLess" => "less than",
            "assertLessEqual" => "less than or equal to",
            _ => unreachable!(),
        },
        crate::py_display(b)
    );
    if pass {
        Ok(())
    } else if py_boxed_cmp(a, b).is_none() {
        // Incomparable operands: loud, never a silent wrong comparison.
        assertion_failed(format!(
            "{}: cannot order {} and {}",
            kind,
            crate::py_display(a),
            crate::py_display(b)
        ))
    } else if ctx.is_empty() {
        assertion_failed(format!("{} {}", crate::py_display(a), phrase))
    } else {
        assertion_failed(format!("{}: {} {}", ctx, crate::py_display(a), phrase))
    }
}

/// `self.assertGreater(a, b)` — AssertionError unless `a > b`.
pub fn assert_greater(
    a: &PyValue,
    b: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    assert_ord(a, b, "assertGreater", ctx)
}

/// `self.assertGreaterEqual(a, b)` — AssertionError unless `a >= b`.
pub fn assert_greater_equal(
    a: &PyValue,
    b: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    assert_ord(a, b, "assertGreaterEqual", ctx)
}

/// `self.assertLess(a, b)` — AssertionError unless `a < b`.
pub fn assert_less(
    a: &PyValue,
    b: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    assert_ord(a, b, "assertLess", ctx)
}

/// `self.assertLessEqual(a, b)` — AssertionError unless `a <= b`.
pub fn assert_less_equal(
    a: &PyValue,
    b: &PyValue,
    ctx: alloc::string::String,
) -> Result<(), PyException> {
    assert_ord(a, b, "assertLessEqual", ctx)
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