//! The math-module functions whose arg-coercion the compiler knows, as a
//! typed enum.
//!
//! Python attribute names arrive from CPython's parser as strings, so ONE
//! string comparison at the AST boundary is unavoidable — but it happens
//! exactly once, in [`MathFn::from_name`]. Every consumer — the per-arg
//! int→f64 coercion signature, the bare-import resolver — works with the
//! enum, so the set of float-coercible math functions has a single source
//! of truth instead of a parallel string list.

/// A stdpython `math` scalar function whose argument coercion the compiler
/// knows. Only the functions with a float-coercible argument are listed; the
/// integer-taking ones (gcd, factorial, comb, perm, isqrt, sumprod) and the
/// list-taking `fsum` are deliberately absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathFn {
    Sqrt,
    Cbrt,
    Ulp,
    Exp,
    Exp2,
    Expm1,
    Log1p,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Degrees,
    Radians,
    Isfinite,
    Isinf,
    Isnan,
    Fabs,
    Ceil,
    Floor,
    Trunc,
    Frexp,
    Modf,
    Hypot,
    Nextafter,
    Fmod,
    Remainder,
    Copysign,
    Atan2,
    Pow,
    Isclose,
    Fma,
    Ldexp,
    Log,
}

impl MathFn {
    /// Parse a stdpython `math` function name at the AST boundary. The
    /// caller is responsible for having established that the name resolves
    /// against the `math` module.
    pub(crate) fn from_name(name: &str) -> Option<MathFn> {
        use MathFn::*;
        Some(match name {
            "sqrt" => Sqrt,
            "cbrt" => Cbrt,
            "ulp" => Ulp,
            "exp" => Exp,
            "exp2" => Exp2,
            "expm1" => Expm1,
            "log1p" => Log1p,
            "log2" => Log2,
            "log10" => Log10,
            "sin" => Sin,
            "cos" => Cos,
            "tan" => Tan,
            "asin" => Asin,
            "acos" => Acos,
            "atan" => Atan,
            "sinh" => Sinh,
            "cosh" => Cosh,
            "tanh" => Tanh,
            "asinh" => Asinh,
            "acosh" => Acosh,
            "atanh" => Atanh,
            "degrees" => Degrees,
            "radians" => Radians,
            "isfinite" => Isfinite,
            "isinf" => Isinf,
            "isnan" => Isnan,
            "fabs" => Fabs,
            "ceil" => Ceil,
            "floor" => Floor,
            "trunc" => Trunc,
            "frexp" => Frexp,
            "modf" => Modf,
            "hypot" => Hypot,
            "nextafter" => Nextafter,
            "fmod" => Fmod,
            "remainder" => Remainder,
            "copysign" => Copysign,
            "atan2" => Atan2,
            "pow" => Pow,
            "isclose" => Isclose,
            "fma" => Fma,
            "ldexp" => Ldexp,
            "log" => Log,
            _ => return None,
        })
    }
}