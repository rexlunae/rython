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


/// numpy int `//` (floor division) — see scalar.rs; identical semantics.
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

/// numpy int `%` (Python mod: sign of the divisor) — see scalar.rs.
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
    ($name:ident, $t:ty, $mod:expr, $max:expr, $min:expr, $div:expr, $floordiv:expr, $mod_fn:expr, $pow:expr) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t]) -> Vec<$t> {
            debug_assert_eq!(a.len(), b.len());
            match op {
                BinOp::Add => par_bin_vec(a, b, |x, y| x + y),
                BinOp::Sub => par_bin_vec(a, b, |x, y| x - y),
                BinOp::Mul => par_bin_vec(a, b, |x, y| x * y),
                // numpy edge-case semantics per type (see scalar.rs):
                // floats divide by zero with IEEE results, ints return 0
                // for floordiv/mod by zero, int true_divide is promoted
                // to float64 by the caller.
                BinOp::Div => par_bin_vec(a, b, $div),
                BinOp::FloorDiv => par_bin_vec(a, b, $floordiv),
                BinOp::Mod => par_bin_vec(a, b, $mod_fn),
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
    binary_f64, f64, "float64", np_max_f64, np_min_f64,
    |x: f64, y: f64| x / y, np_floor_div_f64, np_mod_f64,
    |a: f64, b: f64| a.powf(b)
);
binary_parallel!(
    binary_f32, f32, "float32", np_max_f32, np_min_f32,
    |x: f32, y: f32| x / y, np_floor_div_f32, np_mod_f32,
    |a: f32, b: f32| a.powf(b)
);
binary_parallel!(
    binary_i64, i64, "int64", |a: i64, b: i64| a.max(b), |a: i64, b: i64| a.min(b),
    int_div_unreachable,
    np_int_floor_div,
    np_int_mod,
    np_int_pow
);
binary_parallel!(
    binary_i32, i32, "int32", |a: i32, b: i32| a.max(b), |a: i32, b: i32| a.min(b),
    int_div_unreachable,
    np_int32_floor_div,
    np_int32_mod,
    np_int32_pow
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
            BinOp::Div,
            BinOp::FloorDiv,
            BinOp::Mod,
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
