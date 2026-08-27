//! Scalar engine: plain sequential loops. Always compiled in; the fallback
//! for every accelerated backend. Float semantics follow numpy where numpy
//! and Rust's primitives disagree (NaN-propagating maximum/minimum, Python
//! remainder semantics, floor division toward -inf).

use super::{BinOp, UnOp};

fn np_max_f64(a: f64, b: f64) -> f64 {
    // numpy maximum propagates NaN (Rust's f64::max does not).
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn np_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn np_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

fn np_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

/// numpy float `%` (npy_remainder via npy_divmod, copied from CPython
/// 3.5): `m = fmod(a, b)` (Rust float `%` IS fmod) adjusted to the
/// divisor's sign; a zero result takes the DIVISOR's sign
/// (`copysign(0, b)` — so `mod(0.0, -2.0)` is `-0.0`); a zero divisor
/// gives NaN.
fn np_mod_f64(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) { m + b } else { m }
    } else {
        f64::copysign(0.0, b)
    }
}

fn np_mod_f32(a: f32, b: f32) -> f32 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) { m + b } else { m }
    } else {
        f32::copysign(0.0, b)
    }
}

/// numpy float `//` (npy_floor_divide via npy_divmod): `div = (a - mod)/b`
/// with the fmod remainder adjusted to the divisor's sign — NOT plain
/// `floor(a/b)` (e.g. `floor_divide(1.0, 0.1)` is 9.0, while
/// `floor(1.0/0.1)` = floor(10.0) is 10.0). A zero divisor yields `a/b`
/// (inf/nan). Exact-zero results take the sign of `a/b`, matching the
/// arm64 numpy build (whose FMA-contracted divmod produces +0.0 for
/// `floor_divide(-1.0, -2.0)`); `floor(a/b)` agrees there too.
fn np_floor_div_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a / b;
    }
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div -= 1.0;
    }
    if div == 0.0 {
        div = f64::copysign(0.0, a / b);
    }
    div
}

fn np_floor_div_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        return a / b;
    }
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div -= 1.0;
    }
    if div == 0.0 {
        div = f32::copysign(0.0, a / b);
    }
    div
}

/// numpy `sign`: -1/0/1, NaN stays NaN, -0.0 maps to 0.
fn np_sign_f64(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn np_sign_f32(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn np_int_pow(a: i64, b: i64) -> i64 {
    if b < 0 {
        // numpy raises for ANY negative integer exponent (scalar or per
        // element); it never computes int ** negative.
        panic!(
            "{}",
            crate::PyException::new(
                "ValueError",
                "Integers to negative integer powers are not allowed."
            )
        );
    }
    a.wrapping_pow(b as u32)
}

fn np_int32_pow(a: i32, b: i32) -> i32 {
    if b < 0 {
        panic!(
            "{}",
            crate::PyException::new(
                "ValueError",
                "Integers to negative integer powers are not allowed."
            )
        );
    }
    a.wrapping_pow(b as u32)
}

/// numpy int `//` (floor division): Python semantics with the divisor's
/// sign, `0` for a zero divisor (numpy 2), and C-style wrap on overflow
/// (numpy's `i64::MIN // -1` wraps, with a warning, rather than raising).
/// NOT `div_euclid`: Euclidean and floor division disagree whenever the
/// divisor is negative and the remainder is nonzero (`3 // -2` is `-2`, not
/// `-1`).
fn np_int_floor_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// numpy int `%` (Python mod: sign of the divisor), `0` for a zero
/// divisor, wrap on overflow. NOT `rem_euclid` (which is always
/// non-negative; `3 % -2` is `-1`, not `1`).
fn np_int_mod(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        r.wrapping_add(b)
    } else {
        r
    }
}

fn np_int32_floor_div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

fn np_int32_mod(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// True-divide on integer kernels is unreachable through the numpy API:
/// `binary_same_shape` promotes int/bool division to float64 first (numpy
/// always returns float64 for integer true_divide). A direct engine call is
/// an internal bug — panic loudly rather than silently truncate.
fn int_div_unreachable<T>(_a: T, _b: T) -> T {
    panic!(
        "{}",
        crate::PyException::new(
            "TypeError",
            "integer true_divide must be promoted to float64 (numpy semantics)"
        )
    )
}

/// Elementwise builder `out[i] = f(a[i], b[i])`, monomorphized per closure
/// so each arm of the op dispatch (which the caller performs ONCE, before
/// the loop — never per element) is a tight inner loop that LLVM can
/// auto-vectorize. Returns a freshly allocated vector grown directly from
/// the iterator: no zero-fill pass over the output, since every element is
/// written anyway. Every caller debug-asserts the lengths equal.
fn bin_vec<T: Copy, F: Fn(T, T) -> T>(a: &[T], b: &[T], f: F) -> Vec<T> {
    a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
}

/// Unary sibling of [`bin_vec`].
fn unary_vec<T: Copy, F: Fn(T) -> T>(a: &[T], f: F) -> Vec<T> {
    a.iter().map(|&x| f(x)).collect()
}

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

macro_rules! binary_num {
    ($name:ident, $t:ty, $mod:expr, $max:expr, $min:expr, $div:expr, $floordiv:expr, $mod_fn:expr, $pow:expr) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t]) -> Vec<$t> {
            debug_assert_eq!(a.len(), b.len());
            // Dispatch the operation ONCE, before the loop. Each arm below
            // monomorphizes a tight inner loop LLVM can auto-vectorize; the
            // previous per-element `match op` blocked vectorization and paid
            // a branch on every element.
            match op {
                BinOp::Add => bin_vec(a, b, |x, y| x + y),
                BinOp::Sub => bin_vec(a, b, |x, y| x - y),
                BinOp::Mul => bin_vec(a, b, |x, y| x * y),
                // numpy semantics for the edge cases are per-type (passed in
                // as complete closures): floats divide by zero with IEEE
                // results (inf/nan, no exception), ints return 0 for
                // floordiv/mod by zero, and int true_divide is unreachable
                // (promoted to float64 by the caller).
                BinOp::Div => bin_vec(a, b, $div),
                BinOp::FloorDiv => bin_vec(a, b, $floordiv),
                BinOp::Mod => bin_vec(a, b, $mod_fn),
                BinOp::Pow => bin_vec(a, b, $pow),
                BinOp::Max => bin_vec(a, b, $max),
                BinOp::Min => bin_vec(a, b, $min),
                BinOp::Lt
                | BinOp::Le
                | BinOp::Gt
                | BinOp::Ge
                | BinOp::Eq
                | BinOp::Ne
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            format!("unsupported numpy operation on {}", $mod)
                        )
                    )
                }
            }
        }
    };
}

binary_num!(
    binary_f64,
    f64,
    "float64",
    np_max_f64,
    np_min_f64,
    |x: f64, y: f64| x / y,
    np_floor_div_f64,
    np_mod_f64,
    |a: f64, b: f64| a.powf(b)
);
binary_num!(
    binary_f32,
    f32,
    "float32",
    np_max_f32,
    np_min_f32,
    |x: f32, y: f32| x / y,
    np_floor_div_f32,
    np_mod_f32,
    |a: f32, b: f32| a.powf(b)
);
binary_num!(
    binary_i64,
    i64,
    "int64",
    |a: i64, b: i64| a.max(b),
    |a: i64, b: i64| a.min(b),
    int_div_unreachable,
    np_int_floor_div,
    np_int_mod,
    np_int_pow
);
binary_num!(
    binary_i32,
    i32,
    "int32",
    |a: i32, b: i32| a.max(b),
    |a: i32, b: i32| a.min(b),
    int_div_unreachable,
    np_int32_floor_div,
    np_int32_mod,
    np_int32_pow
);

pub(crate) fn binary_bool(op: BinOp, a: &[bool], b: &[bool]) -> Vec<bool> {
    debug_assert_eq!(a.len(), b.len());
    match op {
        BinOp::Add | BinOp::BitOr | BinOp::Max => bin_vec(a, b, |x, y| x || y),
        BinOp::BitAnd | BinOp::Min => bin_vec(a, b, |x, y| x && y),
        BinOp::BitXor => bin_vec(a, b, |x, y| x != y),
        BinOp::Sub | BinOp::Mul => bin_vec(a, b, |x, y| x != y),
        BinOp::Lt => bin_vec(a, b, |x, y| !x && y),
        BinOp::Le => bin_vec(a, b, |x, y| !x || y),
        BinOp::Gt => bin_vec(a, b, |x, y| x && !y),
        BinOp::Ge => bin_vec(a, b, |x, y| x || !y),
        BinOp::Eq => bin_vec(a, b, |x, y| x == y),
        BinOp::Ne => bin_vec(a, b, |x, y| x != y),
        BinOp::Div | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow => {
            panic!(
                "{}",
                crate::PyException::new("TypeError", "unsupported numpy operation on bool arrays")
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

macro_rules! unary_float {
    ($name:ident, $t:ty, $sign:expr) => {
        pub(crate) fn $name(op: UnOp, a: &[$t]) -> Vec<$t> {
            match op {
                UnOp::Neg => unary_vec(a, |x| -x),
                UnOp::Abs => unary_vec(a, |x| x.abs()),
                UnOp::Sqrt => unary_vec(a, |x| x.sqrt()),
                UnOp::Exp => unary_vec(a, |x| x.exp()),
                UnOp::Log => unary_vec(a, |x| x.ln()),
                UnOp::Log2 => unary_vec(a, |x| x.log2()),
                UnOp::Log10 => unary_vec(a, |x| x.log10()),
                UnOp::Sin => unary_vec(a, |x| x.sin()),
                UnOp::Cos => unary_vec(a, |x| x.cos()),
                UnOp::Tan => unary_vec(a, |x| x.tan()),
                UnOp::Asin => unary_vec(a, |x| x.asin()),
                UnOp::Acos => unary_vec(a, |x| x.acos()),
                UnOp::Atan => unary_vec(a, |x| x.atan()),
                UnOp::Sinh => unary_vec(a, |x| x.sinh()),
                UnOp::Cosh => unary_vec(a, |x| x.cosh()),
                UnOp::Tanh => unary_vec(a, |x| x.tanh()),
                UnOp::Floor => unary_vec(a, |x| x.floor()),
                UnOp::Ceil => unary_vec(a, |x| x.ceil()),
                UnOp::Sign => unary_vec(a, $sign),
                UnOp::Square => unary_vec(a, |x| x * x),
                UnOp::Reciprocal => unary_vec(a, |x| 1.0 / x),
                UnOp::ExpM1 => unary_vec(a, |x| x.exp_m1()),
                UnOp::Log1P => unary_vec(a, |x| x.ln_1p()),
                UnOp::IsFinite | UnOp::IsInf | UnOp::IsNan | UnOp::LogicalNot => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            "predicate operations cannot write into a numeric buffer"
                        )
                    )
                }
            }
        }
    };
}

unary_float!(unary_f64, f64, np_sign_f64);
unary_float!(unary_f32, f32, np_sign_f32);

macro_rules! unary_int {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: UnOp, a: &[$t]) -> Vec<$t> {
            match op {
                UnOp::Neg => unary_vec(a, |x| -x),
                UnOp::Abs => unary_vec(a, |x| x.abs()),
                UnOp::Sign => unary_vec(a, |x| {
                    if x > 0 {
                        1
                    } else if x < 0 {
                        -1
                    } else {
                        0
                    }
                }),
                UnOp::Square => unary_vec(a, |x| x.wrapping_mul(x)),
                UnOp::Floor | UnOp::Ceil => unary_vec(a, |x| x),
                UnOp::Sqrt
                | UnOp::Exp
                | UnOp::Log
                | UnOp::Log2
                | UnOp::Log10
                | UnOp::Sin
                | UnOp::Cos
                | UnOp::Tan
                | UnOp::Asin
                | UnOp::Acos
                | UnOp::Atan
                | UnOp::Sinh
                | UnOp::Cosh
                | UnOp::Tanh
                | UnOp::Reciprocal
                | UnOp::ExpM1
                | UnOp::Log1P => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            "ufuncs that return floats cannot write into an integer buffer"
                        )
                    )
                }
                UnOp::IsFinite | UnOp::IsInf | UnOp::IsNan | UnOp::LogicalNot => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            "predicate operations cannot write into a numeric buffer"
                        )
                    )
                }
            }
        }
    };
}

unary_int!(unary_i64, i64);
unary_int!(unary_i32, i32);

pub(crate) fn unary_bool(op: UnOp, a: &[bool]) -> Vec<bool> {
    match op {
        UnOp::LogicalNot => unary_vec(a, |x| !x),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy unary operation on bool arrays"
            )
        ),
    }
}
