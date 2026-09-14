//! Python math module implementation
//! 
//! This module provides mathematical functions and constants.
//! Implementation matches Python's math module API.

use crate::PyException;
use std::f64::consts;
use crate::python_function;

// Mathematical constants
pub const pi: f64 = consts::PI;
pub const e: f64 = consts::E;
pub const tau: f64 = consts::TAU;
pub const inf: f64 = f64::INFINITY;
pub const nan: f64 = f64::NAN;

/// Convert an already-rounded float to an int the way Python does:
/// NaN and infinity raise instead of silently becoming 0 or i64::MAX,
/// and a magnitude beyond i64 is an overflow rather than a saturation.
fn to_py_int(value: f64, func: &str) -> i64 {
    if value.is_nan() {
        panic!(
            "{}",
            crate::PyException::new("ValueError", "cannot convert float NaN to integer")
        );
    }
    if value.is_infinite() {
        panic!(
            "{}",
            crate::PyException::new(
                "OverflowError",
                "cannot convert float infinity to integer",
            )
        );
    }
    if value < (i64::MIN as f64) || value > (i64::MAX as f64) {
        panic!(
            "{}",
            crate::PyException::new(
                "OverflowError",
                format!("math.{}() result too large to convert to int", func),
            )
        );
    }
    value as i64
}

python_function! {
    /// math.ceil - ceiling function
    pub fn ceil<T>(x: T) -> i64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> i64]
    {
        to_py_int(x.into().ceil(), "ceil")
    }
}

python_function! {
    /// math.floor - floor function
    pub fn floor<T>(x: T) -> i64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> i64]
    {
        to_py_int(x.into().floor(), "floor")
    }
}

python_function! {
    /// math.trunc - truncate to integer
    pub fn trunc<T>(x: T) -> i64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> i64]
    {
        to_py_int(x.into().trunc(), "trunc")
    }
}

python_function! {
    /// math.fabs - absolute value (float)
    pub fn fabs<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().abs()
    }
}

python_function! {
    /// math.sqrt - square root
    pub fn sqrt<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val < 0.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.sqrt())
        }
    }
}

python_function! {
    /// math.pow - power function
    pub fn pow<T, U>(x: T, y: U) -> Result<f64, PyException>
    where [T: Into<f64>, U: Into<f64>]
    [signature: (x, y)]
    [concrete_types: (f64, f64) -> Result<f64, crate::PyException>]
    {
        let x = x.into();
        let y = y.into();
        // CPython raises ValueError: math domain error for a negative base
        // with a non-integral exponent (the result would be complex) and
        // for zero raised to a negative power (a division by zero) —
        // issue #82; the bare powf silently yields NaN/inf.
        if x < 0.0 && y.fract() != 0.0 {
            return Err(crate::value_error("math domain error"));
        }
        if x == 0.0 && y < 0.0 {
            return Err(crate::value_error("math domain error"));
        }
        Ok(x.powf(y))
    }
}

python_function! {
    /// math.exp - exponential function
    pub fn exp<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().exp()
    }
}

python_function! {
    /// math.exp2 - 2^x
    pub fn exp2<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().exp2()
    }
}

python_function! {
    /// math.expm1 - exp(x) - 1
    pub fn expm1<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().exp_m1()
    }
}

python_function! {
    /// math.log - natural logarithm
    pub fn log<T>(x: T, base: Option<f64>) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x, base=None)]
    [concrete_types: (f64, Option<f64>) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val <= 0.0 {
            return Err(crate::value_error("math domain error"));
        }
        
        match base {
            Some(b) if b <= 0.0 || b == 1.0 => Err(crate::value_error("math domain error")),
            Some(b) => Ok(val.ln() / b.ln()),
            None => Ok(val.ln()),
        }
    }
}

python_function! {
    /// math.log2 - base-2 logarithm
    pub fn log2<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val <= 0.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.log2())
        }
    }
}

python_function! {
    /// math.log10 - base-10 logarithm
    pub fn log10<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val <= 0.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.log10())
        }
    }
}

python_function! {
    /// math.log1p - log(1 + x)
    pub fn log1p<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val <= -1.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.ln_1p())
        }
    }
}

// Trigonometric functions
python_function! {
    /// math.sin - sine
    pub fn sin<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().sin()
    }
}

python_function! {
    /// math.cos - cosine
    pub fn cos<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().cos()
    }
}

python_function! {
    /// math.tan - tangent
    pub fn tan<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().tan()
    }
}

python_function! {
    /// math.asin - arc sine
    pub fn asin<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val < -1.0 || val > 1.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.asin())
        }
    }
}

python_function! {
    /// math.acos - arc cosine
    pub fn acos<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val < -1.0 || val > 1.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.acos())
        }
    }
}

python_function! {
    /// math.atan - arc tangent
    pub fn atan<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().atan()
    }
}

python_function! {
    /// math.atan2 - arc tangent of y/x
    pub fn atan2<T, U>(y: T, x: U) -> f64
    where [T: Into<f64>, U: Into<f64>]
    [signature: (y, x)]
    [concrete_types: (f64, f64) -> f64]
    {
        y.into().atan2(x.into())
    }
}

// Hyperbolic functions
python_function! {
    /// math.sinh - hyperbolic sine
    pub fn sinh<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().sinh()
    }
}

python_function! {
    /// math.cosh - hyperbolic cosine
    pub fn cosh<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().cosh()
    }
}

python_function! {
    /// math.tanh - hyperbolic tangent
    pub fn tanh<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().tanh()
    }
}

python_function! {
    /// math.asinh - inverse hyperbolic sine
    pub fn asinh<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().asinh()
    }
}

python_function! {
    /// math.acosh - inverse hyperbolic cosine
    pub fn acosh<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val < 1.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.acosh())
        }
    }
}

python_function! {
    /// math.atanh - inverse hyperbolic tangent
    pub fn atanh<T>(x: T) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> Result<f64, crate::PyException>]
    {
        let val = x.into();
        if val <= -1.0 || val >= 1.0 {
            Err(crate::value_error("math domain error"))
        } else {
            Ok(val.atanh())
        }
    }
}

// Angular conversion
python_function! {
    /// math.degrees - convert radians to degrees
    pub fn degrees<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().to_degrees()
    }
}

python_function! {
    /// math.radians - convert degrees to radians
    pub fn radians<T>(x: T) -> f64
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> f64]
    {
        x.into().to_radians()
    }
}

// Special functions
python_function! {
    /// math.factorial - factorial
    pub fn factorial(x: i64) -> Result<i64, PyException>
    [signature: (x)]
    [concrete_types: (i64) -> Result<i64, crate::PyException>]
    {
        if x < 0 {
            return Err(crate::value_error("factorial() not defined for negative values"));
        }
        
        if x > 20 {
            return Err(crate::overflow_error("factorial() result too large"));
        }
        
        let mut result = 1i64;
        for i in 1..=x {
            result = result.saturating_mul(i);
        }
        Ok(result)
    }
}

python_function! {
    /// math.gcd - greatest common divisor
    pub fn gcd(a: i64, b: i64) -> i64
    [signature: (a, b)]
    [concrete_types: (i64, i64) -> i64]
    {
        fn gcd_impl(mut a: i64, mut b: i64) -> i64 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a.abs()
        }
        gcd_impl(a, b)
    }
}

python_function! {
    /// math.lcm - least common multiple
    pub fn lcm(a: i64, b: i64) -> i64
    [signature: (a, b)]
    [concrete_types: (i64, i64) -> i64]
    {
        if a == 0 || b == 0 {
            0
        } else {
            (a / gcd(a, b) * b).abs()
        }
    }
}

// Classification functions
python_function! {
    /// math.isfinite - check if x is finite
    pub fn isfinite<T>(x: T) -> bool
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> bool]
    {
        x.into().is_finite()
    }
}

python_function! {
    /// math.isinf - check if x is infinite
    pub fn isinf<T>(x: T) -> bool
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> bool]
    {
        x.into().is_infinite()
    }
}

python_function! {
    /// math.isnan - check if x is NaN
    pub fn isnan<T>(x: T) -> bool
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> bool]
    {
        x.into().is_nan()
    }
}

python_function! {
    /// math.isclose - check if values are close
    pub fn isclose<T, U>(a: T, b: U, rel_tol: Option<f64>, abs_tol: Option<f64>) -> bool
    where [T: Into<f64>, U: Into<f64>]
    [signature: (a, b, rel_tol=None, abs_tol=None)]
    [concrete_types: (f64, f64, Option<f64>, Option<f64>) -> bool]
    {
        let a = a.into();
        let b = b.into();
        let rel_tol = rel_tol.unwrap_or(1e-9);
        let abs_tol = abs_tol.unwrap_or(0.0);
        
        if a == b {
            return true;
        }
        
        if a.is_infinite() || b.is_infinite() || a.is_nan() || b.is_nan() {
            return false;
        }
        
        let diff = (a - b).abs();
        diff <= abs_tol.max(rel_tol * a.abs().max(b.abs()))
    }
}

python_function! {
    /// math.copysign - return a float with the magnitude of x and the sign of y
    pub fn copysign<T, U>(magnitude: T, sign: U) -> f64
    where [T: Into<f64>, U: Into<f64>]
    [signature: (magnitude, sign)]
    [concrete_types: (f64, f64) -> f64]
    {
        magnitude.into().copysign(sign.into())
    }
}

python_function! {
    /// math.frexp - return mantissa and exponent
    pub fn frexp<T>(x: T) -> (f64, i32)
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> (f64, i32)]
    {
        let mut val = x.into();
        // Python: frexp(0.0) == (0.0, 0); inf/nan pass through with e == 0.
        if val == 0.0 || !val.is_finite() {
            return (val, 0);
        }

        // Subnormals have a zero exponent field and an implicit leading 0
        // bit — the normalized-number bit trick misreads them. Scale into
        // the normal range first and compensate in the exponent.
        let mut adjust = 0i32;
        if (val.to_bits() >> 52) & 0x7ff == 0 {
            val *= (2.0f64).powi(64);
            adjust = -64;
        }

        let bits = val.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32 - 1022 + adjust;
        let mantissa = f64::from_bits((bits & 0x800fffffffffffff) | 0x3fe0000000000000);

        (mantissa, exponent)
    }
}

python_function! {
    /// math.ldexp - return x * (2**i)
    pub fn ldexp<T>(x: T, i: i32) -> Result<f64, PyException>
    where [T: Into<f64>]
    [signature: (x, i)]
    [concrete_types: (f64, i32) -> Result<f64, crate::PyException>]
    {
        let x = x.into();
        // libm::ldexp scales the exponent directly, so subnormal results
        // (ldexp(1e-300, 1074)) stay exact; x * 2f64.powi(i) rounded through
        // an intermediate that overflows to inf or underflows to 0
        // (issue #82).
        let result = libm::ldexp(x, i);
        if result.is_infinite() && !x.is_infinite() {
            return Err(crate::overflow_error("math range error"));
        }
        Ok(result)
    }
}

python_function! {
    /// math.modf - return fractional and integer parts
    pub fn modf<T>(x: T) -> (f64, f64)
    where [T: Into<f64>]
    [signature: (x)]
    [concrete_types: (f64) -> (f64, f64)]
    {
        let val = x.into();
        if val.is_infinite() {
            // CPython: modf(inf) -> (0.0, inf); inf - inf would be NaN.
            return (0.0, val);
        }
        let integer_part = val.trunc();
        let fractional_part = val - integer_part;
        (fractional_part, integer_part)
    }
}

python_function! {
    /// math.fmod - floating point remainder
    pub fn fmod<T, U>(x: T, y: U) -> Result<f64, PyException>
    where [T: Into<f64>, U: Into<f64>]
    [signature: (x, y)]
    [concrete_types: (f64, f64) -> Result<f64, crate::PyException>]
    {
        let x = x.into();
        let y = y.into();
        
        if y == 0.0 || x.is_infinite() {
            // fmod(inf, y) is a domain error in CPython, not NaN.
            Err(crate::value_error("math domain error"))
        } else {
            Ok(x % y)
        }
    }
}

python_function! {
    /// math.remainder - IEEE remainder
    pub fn remainder<T, U>(x: T, y: U) -> Result<f64, PyException>
    where [T: Into<f64>, U: Into<f64>]
    [signature: (x, y)]
    [concrete_types: (f64, f64) -> Result<f64, crate::PyException>]
    {
        let x = x.into();
        let y = y.into();

        // CPython's m_remainder: NaN operands give NaN; an infinite
        // divisor gives x; an infinite dividend or zero divisor is a
        // domain error. The value itself is the IEEE 754 remainder,
        // computed by libm exactly (half-to-even, fmod-based reduction) —
        // the old x - round(x/y)*y double-rounded for large quotients
        // (issue #82).
        if x.is_nan() || y.is_nan() {
            return Ok(f64::NAN);
        }
        if y.is_infinite() {
            return Ok(x);
        }
        if x.is_infinite() || y == 0.0 {
            return Err(crate::value_error("math domain error"));
        }
        Ok(libm::remainder(x, y))
    }
}

/// math.fsum(values) — a faithful port of CPython's `math_fsum` (Raymond
/// Hettinger's msum partials algorithm, + Mark Dickinson's exact partials
/// sum and roundoff). Sums with error-free rounding so `fsum([0.1, 0.2,
/// 0.3])` is EXACTLY `0.6` and a small magnitude survives a large one
/// (`fsum([1e100, -1e100, 1e-100])` is `1e-100`). Special values: a NaN
/// propagates (after scanning ALL values, so `fsum([inf, nan])` is NaN);
/// opposite infinities raise `ValueError: -inf + inf in fsum`; an
/// INTERMEDIATE overflow of a finite running total (`fsum([1e308,
/// 1e308])`) raises `OverflowError: intermediate overflow in fsum`. The
/// final collapsed sum applies half-even rounding across multiple
/// partials (`fsum([1e16, 1.0, 1e-16])` is `1.0000000000000002e16`).
/// This is a raw `pub fn` (not the scalar `python_function!` macro)
/// because it takes a SLICE of floats, like the list-taking stdlib
/// functions.
pub fn fsum(values: &[f64]) -> Result<f64, PyException> {
    // The partials list (the msum algorithm); a non-empty Vec.
    let mut p: alloc::vec::Vec<f64> = alloc::vec::Vec::new();
    // CPython adds EVERY nonfinite value (inf AND NaN) to `special_sum`
    // (so it is non-zero whenever any special value was seen), and
    // accumulates only the infinities in `inf_sum` so `inf + -inf` = NaN
    // detects opposite-sign infinities.
    let mut special_sum = 0.0f64;
    let mut inf_sum = 0.0f64;
    let mut fsum_err: Option<PyException> = None;

    for &xsave in values {
        let mut x = xsave;
        // for y in partials: merge x into the (magnitude-ordered)
        // partials, preserving each rounding error `lo = y - yr`.
        let mut write = 0usize;
        for j in 0..p.len() {
            let y = p[j];
            let (big, small) = if x.abs() < y.abs() { (y, x) } else { (x, y) };
            let hi = big + small;
            let yr = hi - big;
            let lo = small - yr;
            if lo != 0.0 {
                p[write] = lo;
                write += 1;
            }
            x = hi;
        }
        p.truncate(write);
        if x != 0.0 {
            if !x.is_finite() {
                // A nonfinite `x` here is either an intermediate overflow
                // of a FINITE running total, or it is the inf/nan
                // summand itself having flowed through the partials.
                if xsave.is_finite() {
                    fsum_err = Some(PyException::new(
                        "OverflowError",
                        "intermediate overflow in fsum",
                    ));
                    break;
                }
                if xsave.is_infinite() {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                p.clear();
            } else {
                p.push(x);
            }
        }
    }

    if let Some(fe) = fsum_err {
        return Err(fe);
    }

    // A special value (inf or NaN) was seen. Opposite-sign infinities are
    // a ValueError; any other special value is the sum's result.
    if special_sum != 0.0 {
        if inf_sum.is_nan() {
            return Err(PyException::new("ValueError", "-inf + inf in fsum"));
        }
        return Ok(special_sum);
    }

    // sum_exact: collapse the partials large-to-small and correctly round
    // the final result (half-even), with CPython's multi-partial
    // correction.
    let mut hi = 0.0f64;
    let mut n = p.len();
    if n > 0 {
        n -= 1;
        hi = p[n];
        let mut lo = 0.0f64;
        while n > 0 {
            let y = p[n - 1];
            n -= 1;
            // |y| < |hi| by the magnitude ordering above.
            let x = hi;
            hi = x + y;
            let yr = hi - x;
            let l = y - yr;
            lo = l;
            if l != 0.0 {
                break;
            }
        }
        // Half-even rounding across multiple partials: if the residual
        // `lo` has the same sign as the next partial, check whether
        // `2*lo` exactly hits the next representable float.
        if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Ok(hi)
}

/// math.sumprod(a, b) — the dot product of two equal-length numeric
/// sequences (CPython 3.12+). `sumprod(i64-a, i64-b)` where every element
/// is an int returns an int (`sumprod([10, 20, 30], [1, 2, 3])` is `140`);
/// any float involvement makes it a float. Keeps the pair-typed result so
/// the codegen routes int×int to `sumprod_i64` and a float in either
/// sequence to `sumprod_f64` (the list element types decide). Unequal
/// lengths raise `ValueError: len(a) != len(b)`, exactly as CPython does.
/// Raw `pub fn`s (not the scalar `python_function!` macro) because they
/// take SLICES, like `fsum`.
pub fn sumprod_i64(a: &[i64], b: &[i64]) -> Result<i64, PyException> {
    if a.len() != b.len() {
        return Err(PyException::new("ValueError", "Inputs are not the same length"));
    }
    let mut total = 0i64;
    for (x, y) in a.iter().zip(b.iter()) {
        total += x * y;
    }
    Ok(total)
}

/// The float form: any float (or a single int list) routes here, dot-
/// producting in f64 so `sumprod([1.5, 2.5], [3.5, 4.5])` is `16.5`.
/// Uses CPython's extended-precision accumulation (Ogita-Rump-Oishi
/// `TripleLength` fast triple-double + `fma`) so cancellation keeps the
/// exact result: `sumprod([1e16, 1.0, -1e16], [1.0, 1.0, 1.0])` is `1.0`,
/// not a naive `0.0` (a naive `total += x*y` rounds after every op and
/// discards the `1.0` term — a silent wrong answer, issue #369/#82).
pub fn sumprod_f64(a: &[f64], b: &[f64]) -> Result<f64, PyException> {
    if a.len() != b.len() {
        return Err(PyException::new("ValueError", "Inputs are not the same length"));
    }
    // TripleLength: {hi, lo, tiny} such that hi+lo+tiny is exact.
    #[derive(Clone, Copy)]
    struct TripleLength {
        hi: f64,
        lo: f64,
        tiny: f64,
    }
    let zero = TripleLength { hi: 0.0, lo: 0.0, tiny: 0.0 };
    fn dl_sum(a: f64, b: f64) -> (f64, f64) {
        // Algorithm 3.1: error-free transformation of a sum.
        let x = a + b;
        let z = x - a;
        ((x), (a - (x - z)) + (b - z))
    }
    fn tl_fma(x: f64, y: f64, total: TripleLength) -> TripleLength {
        // Algorithm 5.10 with SumKVert for K=3; the exact product via
        // fma (Rust f64::mul_add), then a triple-double accumulation.
        let p_hi = x * y;
        let p_lo = x.mul_add(y, -p_hi); // exact residual of the product
        let (sm_hi, sm_lo) = dl_sum(total.hi, p_hi);
        let (r1_hi, r1_lo) = dl_sum(total.lo, p_lo);
        let (r2_hi, r2_lo) = dl_sum(r1_hi, sm_lo);
        TripleLength {
            hi: sm_hi,
            lo: r2_hi,
            tiny: total.tiny + r1_lo + r2_lo,
        }
    }
    fn tl_to_d(total: TripleLength) -> f64 {
        let (l_hi, l_lo) = dl_sum(total.lo, total.hi);
        total.tiny + l_lo + l_hi
    }
    let mut acc = zero;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = tl_fma(*x, *y, acc);
        // Non-finite fallback: propagate inf/nan like CPython's normal
        // path would (a NaN operand gives NaN; an infinite product
        // accumulates to inf/nan).
        if !acc.hi.is_finite() {
            return Ok(acc.hi);
        }
    }
    Ok(tl_to_d(acc))
}