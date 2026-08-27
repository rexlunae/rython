//! Python math module implementation
//!
//! This module provides mathematical functions and constants.
//! Implementation matches Python's math module API.

use crate::PyException;
use crate::python_function;
use std::f64::consts;

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
            crate::PyException::new("OverflowError", "cannot convert float infinity to integer",)
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
