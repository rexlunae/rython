//! Multithreaded elementwise numpy kernels (the `numpy-rayon` feature).
//!
//! Every function mirrors the sequential `scalar` kernels' per-element
//! semantics EXACTLY — same operations, same div-by-zero panics, same
//! NaN/sign edge cases — so results are identical whichever backend runs.
//! The parity tests in this module pin that (a drift between the two
//! would be a silent numeric difference, which the prime directive
//! forbids).
//!
//! Like the scalar kernels, the `BinOp`/`UnOp` dispatch happens ONCE per
//! call, before the loop: each `match` arm monomorphizes a tight parallel
//! loop whose per-thread chunks LLVM can auto-vectorize. (A per-element
//! `match op` would block vectorization and pay a branch every iteration.)
//!
//! Kernels return a freshly allocated `Vec` (never write into a caller
//! buffer), so `out` can never alias `a`/`b` — no in-place races are
//! possible by construction.

use super::{BinOp, UnOp};
use rayon::prelude::*;

fn np_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn np_max_f64(a: f64, b: f64) -> f64 {
    // a.max(b), NOT min-with-swapped-args: the two disagree on signed
    // zero (max(0.0, -0.0) is +0.0, min is -0.0).
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
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

fn np_int_pow(a: i64, b: i64) -> i64 {
    if b >= 0 {
        a.wrapping_pow(b as u32)
    } else {
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

/// Rayon elementwise builder `out[i] = f(a[i], b[i])`, monomorphized per
/// closure (the caller dispatches the op once, before the loop). Returns a
/// freshly allocated vector; rayon's `collect` grows it without a
/// zero-fill pass, and each worker's chunk is a tight, auto-vectorizable
/// inner loop.
fn par_bin_vec<T: Copy + Send + Sync, F: Fn(T, T) -> T + Sync>(a: &[T], b: &[T], f: F) -> Vec<T> {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| f(x, y))
        .collect()
}

/// Unary sibling of [`par_bin_vec`].
fn par_un_vec<T: Copy + Send + Sync, F: Fn(T) -> T + Sync>(a: &[T], f: F) -> Vec<T> {
    a.par_iter().map(|&x| f(x)).collect()
}

// ---------------------------------------------------------------------------
// Binary kernels
// ---------------------------------------------------------------------------

macro_rules! binary_parallel {
    ($name:ident, $t:ty, $mod:expr, $max:expr, $min:expr, $floordiv:expr, $pow:expr, $mod_fn:expr, $zero_msg:expr, $zero:expr) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t]) -> Vec<$t> {
            debug_assert_eq!(a.len(), b.len());
            match op {
                BinOp::Add => par_bin_vec(a, b, |x, y| x + y),
                BinOp::Sub => par_bin_vec(a, b, |x, y| x - y),
                BinOp::Mul => par_bin_vec(a, b, |x, y| x * y),
                BinOp::Div => par_bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        x / y
                    }
                }),
                BinOp::FloorDiv => par_bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        $floordiv(x, y)
                    }
                }),
                BinOp::Mod => par_bin_vec(a, b, |x, y| {
                    if y == $zero {
                        div_zero_int($zero_msg)
                    } else {
                        $mod_fn(x, y)
                    }
                }),
                BinOp::Pow => par_bin_vec(a, b, $pow),
                BinOp::Max => par_bin_vec(a, b, $max),
                BinOp::Min => par_bin_vec(a, b, $min),
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

binary_parallel!(
    binary_f64, f64, "float64", np_max_f64, np_min_f64, np_floor_div_f64,
    |a: f64, b: f64| a.powf(b), np_mod_f64, "float64 division", 0.0
);
binary_parallel!(
    binary_f32, f32, "float32", np_max_f32, np_min_f32, np_floor_div_f32,
    |a: f32, b: f32| a.powf(b), np_mod_f32, "float32 division", 0.0
);
binary_parallel!(
    binary_i64, i64, "int64", |a: i64, b: i64| a.max(b), |a: i64, b: i64| a.min(b),
    |a: i64, b: i64| a.div_euclid(b), np_int_pow,
    |a: i64, b: i64| a.rem_euclid(b), "integer division", 0
);
binary_parallel!(
    binary_i32, i32, "int32", |a: i32, b: i32| a.max(b), |a: i32, b: i32| a.min(b),
    |a: i32, b: i32| a.div_euclid(b), np_int32_pow,
    |a: i32, b: i32| a.rem_euclid(b), "integer division", 0
);

pub(crate) fn binary_bool(op: BinOp, a: &[bool], b: &[bool]) -> Vec<bool> {
    debug_assert_eq!(a.len(), b.len());
    match op {
        BinOp::Add | BinOp::BitOr | BinOp::Max => par_bin_vec(a, b, |x, y| x || y),
        BinOp::BitAnd | BinOp::Min => par_bin_vec(a, b, |x, y| x && y),
        BinOp::BitXor => par_bin_vec(a, b, |x, y| x != y),
        BinOp::Sub | BinOp::Mul => par_bin_vec(a, b, |x, y| x != y),
        BinOp::Lt => par_bin_vec(a, b, |x, y| !x && y),
        BinOp::Le => par_bin_vec(a, b, |x, y| !x || y),
        BinOp::Gt => par_bin_vec(a, b, |x, y| x && !y),
        BinOp::Ge => par_bin_vec(a, b, |x, y| x || !y),
        BinOp::Eq => par_bin_vec(a, b, |x, y| x == y),
        BinOp::Ne => par_bin_vec(a, b, |x, y| x != y),
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
// Unary kernels
// ---------------------------------------------------------------------------

macro_rules! unary_parallel_float {
    ($name:ident, $t:ty, $sign:expr) => {
        pub(crate) fn $name(op: UnOp, a: &[$t]) -> Vec<$t> {
            match op {
                UnOp::Neg => par_un_vec(a, |x| -x),
                UnOp::Abs => par_un_vec(a, |x| x.abs()),
                UnOp::Sqrt => par_un_vec(a, |x| x.sqrt()),
                UnOp::Exp => par_un_vec(a, |x| x.exp()),
                UnOp::Log => par_un_vec(a, |x| x.ln()),
                UnOp::Log2 => par_un_vec(a, |x| x.log2()),
                UnOp::Log10 => par_un_vec(a, |x| x.log10()),
                UnOp::Sin => par_un_vec(a, |x| x.sin()),
                UnOp::Cos => par_un_vec(a, |x| x.cos()),
                UnOp::Tan => par_un_vec(a, |x| x.tan()),
                UnOp::Asin => par_un_vec(a, |x| x.asin()),
                UnOp::Acos => par_un_vec(a, |x| x.acos()),
                UnOp::Atan => par_un_vec(a, |x| x.atan()),
                UnOp::Sinh => par_un_vec(a, |x| x.sinh()),
                UnOp::Cosh => par_un_vec(a, |x| x.cosh()),
                UnOp::Tanh => par_un_vec(a, |x| x.tanh()),
                UnOp::Floor => par_un_vec(a, |x| x.floor()),
                UnOp::Ceil => par_un_vec(a, |x| x.ceil()),
                UnOp::Sign => par_un_vec(a, $sign),
                UnOp::Square => par_un_vec(a, |x| x * x),
                UnOp::Reciprocal => par_un_vec(a, |x| 1.0 / x),
                UnOp::ExpM1 => par_un_vec(a, |x| x.exp_m1()),
                UnOp::Log1P => par_un_vec(a, |x| x.ln_1p()),
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

unary_parallel_float!(unary_f64, f64, |x: f64| {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
});
unary_parallel_float!(unary_f32, f32, |x: f32| {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
});

macro_rules! unary_parallel_int {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: UnOp, a: &[$t]) -> Vec<$t> {
            match op {
                UnOp::Neg => par_un_vec(a, |x| -x),
                UnOp::Abs => par_un_vec(a, |x| x.abs()),
                UnOp::Sign => par_un_vec(a, |x| {
                    if x > 0 {
                        1
                    } else if x < 0 {
                        -1
                    } else {
                        0
                    }
                }),
                UnOp::Square => par_un_vec(a, |x| x.wrapping_mul(x)),
                UnOp::Floor | UnOp::Ceil => par_un_vec(a, |x| x),
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

unary_parallel_int!(unary_i64, i64);
unary_parallel_int!(unary_i32, i32);

pub(crate) fn unary_bool(op: UnOp, a: &[bool]) -> Vec<bool> {
    match op {
        UnOp::LogicalNot => par_un_vec(a, |x| !x),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy unary operation on bool arrays"
            )
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scalar;

    /// Every op over a slice containing the interesting float values
    /// (NaN, ±0.0, ±inf, negatives, denormals) must match the scalar
    /// kernel element-for-element, bit-for-bit.
    #[test]
    fn binary_f64_matches_scalar_bitwise() {
        let vals = [
            f64::NAN,
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.5,
            -3.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
            5e-300,
            1.0e308,
        ];
        let a: Vec<f64> = (0..2048).map(|i| vals[i % vals.len()]).collect();
        let b: Vec<f64> = (0..2048).map(|i| vals[(i * 7 + 3) % vals.len()]).collect();
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Pow,
            BinOp::Max,
            BinOp::Min,
        ] {
            let r1 = scalar::binary_f64(op, &a, &b);
            let r2 = binary_f64(op, &a, &b);
            let b1: Vec<u64> = r1.iter().map(|v| v.to_bits()).collect();
            let b2: Vec<u64> = r2.iter().map(|v| v.to_bits()).collect();
            assert_eq!(b1, b2, "op {op:?}");
        }
    }

    #[test]
    fn unary_f64_matches_scalar_bitwise() {
        let vals = [
            f64::NAN,
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
        ];
        let a: Vec<f64> = (0..1024).map(|i| vals[i % vals.len()]).collect();
        for op in [
            UnOp::Neg,
            UnOp::Abs,
            UnOp::Sqrt,
            UnOp::Exp,
            UnOp::Log,
            UnOp::Log2,
            UnOp::Log10,
            UnOp::Sin,
            UnOp::Cos,
            UnOp::Tan,
            UnOp::Asin,
            UnOp::Acos,
            UnOp::Atan,
            UnOp::Sinh,
            UnOp::Cosh,
            UnOp::Tanh,
            UnOp::Floor,
            UnOp::Ceil,
            UnOp::Sign,
            UnOp::Square,
            UnOp::Reciprocal,
            UnOp::ExpM1,
            UnOp::Log1P,
        ] {
            let r1 = scalar::unary_f64(op, &a);
            let r2 = unary_f64(op, &a);
            let b1: Vec<u64> = r1.iter().map(|v| v.to_bits()).collect();
            let b2: Vec<u64> = r2.iter().map(|v| v.to_bits()).collect();
            assert_eq!(b1, b2, "op {op:?}");
        }
    }
}
