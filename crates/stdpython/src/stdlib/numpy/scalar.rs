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

/// Python remainder: `a - b * floor(a / b)` (sign follows the divisor).
/// Division by zero yields NaN, matching numpy's float remainder.
fn np_mod_f64(a: f64, b: f64) -> f64 {
    a - b * (a / b).floor()
}

fn np_mod_f32(a: f32, b: f32) -> f32 {
    a - b * (a / b).floor()
}

fn np_floor_div_f64(a: f64, b: f64) -> f64 {
    (a / b).floor()
}

fn np_floor_div_f32(a: f32, b: f32) -> f32 {
    (a / b).floor()
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
    if b >= 0 {
        a.wrapping_pow(b as u32)
    } else {
        // numpy computes int^negative as a float then truncates toward
        // zero, so 2 ** -1 == 0 and 2 ** -2 == 0 for int arrays.
        (a as f64).powi(b as i32) as i64
    }
}

fn np_int32_pow(a: i32, b: i32) -> i32 {
    if b >= 0 {
        a.wrapping_pow(b as u32)
    } else {
        (a as f64).powi(b) as i32
    }
}

fn div_zero_int(ty: &str) -> ! {
    panic!(
        "{}",
        crate::PyException::new("ZeroDivisionError", format!("integer {ty} by zero"))
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
    ($name:ident, $t:ty, $mod:expr, $max:expr, $min:expr, $floordiv:expr, $pow:expr, $mod_fn:expr, $zero_msg:expr, $zero:expr) => {
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
                BinOp::Div => bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        x / y
                    }
                }),
                BinOp::FloorDiv => bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        $floordiv(x, y)
                    }
                }),
                BinOp::Mod => bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        $mod_fn(x, y)
                    }
                }),
                BinOp::Pow => bin_vec(a, b, $pow),
                BinOp::Max => bin_vec(a, b, $max),
                BinOp::Min => bin_vec(a, b, $min),
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
                | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
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
    binary_f64, f64, "float64", np_max_f64, np_min_f64, np_floor_div_f64,
    |a: f64, b: f64| a.powf(b), np_mod_f64, "float64 division", 0.0
);
binary_num!(
    binary_f32, f32, "float32", np_max_f32, np_min_f32, np_floor_div_f32,
    |a: f32, b: f32| a.powf(b), np_mod_f32, "float32 division", 0.0
);
binary_num!(
    binary_i64, i64, "int64", |a: i64, b: i64| a.max(b), |a: i64, b: i64| a.min(b),
    |a: i64, b: i64| a.div_euclid(b), np_int_pow,
    |a: i64, b: i64| a.rem_euclid(b), "integer division", 0
);
binary_num!(
    binary_i32, i32, "int32", |a: i32, b: i32| a.max(b), |a: i32, b: i32| a.min(b),
    |a: i32, b: i32| a.div_euclid(b), np_int32_pow,
    |a: i32, b: i32| a.rem_euclid(b), "integer division", 0
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
                crate::PyException::new(
                    "TypeError",
                    "unsupported numpy operation on bool arrays"
                )
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
                UnOp::Sqrt | UnOp::Exp | UnOp::Log | UnOp::Log2 | UnOp::Log10
                | UnOp::Sin | UnOp::Cos | UnOp::Tan | UnOp::Asin | UnOp::Acos
                | UnOp::Atan | UnOp::Sinh | UnOp::Cosh | UnOp::Tanh | UnOp::Reciprocal
                | UnOp::ExpM1 | UnOp::Log1P => {
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
