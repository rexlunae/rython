//! Multithreaded elementwise numpy kernels (the `numpy-rayon` feature).
//!
//! Every function mirrors the sequential `scalar` kernels' per-element
//! semantics EXACTLY — same operations, same div-by-zero panics, same
//! NaN/sign edge cases — so results are identical whichever backend runs.
//! The parity tests in this module pin that (a drift between the two
//! would be a silent numeric difference, which the prime directive
//! forbids).
//!
//! In-place calls (`out` aliasing `a` or `b`, as compound ops do) fall
//! back to the sequential kernel: rayon's `par_iter_mut` + indexed reads
//! of the same buffer would race, and the sequential path is correct
//! for every in-place shape.

use super::scalar;
use super::{BinOp, UnOp};
use rayon::prelude::*;

fn same_buffer<T>(a: &[T], b: &[T]) -> bool {
    core::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len()
}

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

fn apply_bin_f64(op: BinOp, x: f64, y: f64) -> f64 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => {
            if y == 0.0 {
                div_zero_int("float64 division")
            } else {
                x / y
            }
        }
        BinOp::FloorDiv => {
            if y == 0.0 {
                div_zero_int("float64 division")
            } else {
                np_floor_div_f64(x, y)
            }
        }
        BinOp::Mod => {
            if y == 0.0 {
                div_zero_int("float64 division")
            } else {
                np_mod_f64(x, y)
            }
        }
        BinOp::Pow => x.powf(y),
        BinOp::Max => np_max_f64(x, y),
        BinOp::Min => np_min_f64(x, y),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy operation on float64"
            )
        ),
    }
}

fn apply_bin_f32(op: BinOp, x: f32, y: f32) -> f32 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => {
            if y == 0.0 {
                div_zero_int("float32 division")
            } else {
                x / y
            }
        }
        BinOp::FloorDiv => {
            if y == 0.0 {
                div_zero_int("float32 division")
            } else {
                np_floor_div_f32(x, y)
            }
        }
        BinOp::Mod => {
            if y == 0.0 {
                div_zero_int("float32 division")
            } else {
                np_mod_f32(x, y)
            }
        }
        BinOp::Pow => x.powf(y),
        BinOp::Max => {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else {
                x.max(y)
            }
        }
        BinOp::Min => {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else {
                x.min(y)
            }
        }
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy operation on float32"
            )
        ),
    }
}

fn apply_bin_i64(op: BinOp, x: i64, y: i64) -> i64 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div | BinOp::FloorDiv => {
            if y == 0 {
                div_zero_int("integer division")
            } else {
                x.div_euclid(y)
            }
        }
        BinOp::Mod => {
            if y == 0 {
                div_zero_int("integer division")
            } else {
                x.rem_euclid(y)
            }
        }
        BinOp::Pow => np_int_pow(x, y),
        BinOp::Max => x.max(y),
        BinOp::Min => x.min(y),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy operation on int64"
            )
        ),
    }
}

fn apply_bin_i32(op: BinOp, x: i32, y: i32) -> i32 {
    match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div | BinOp::FloorDiv => {
            if y == 0 {
                div_zero_int("integer division")
            } else {
                x.div_euclid(y)
            }
        }
        BinOp::Mod => {
            if y == 0 {
                div_zero_int("integer division")
            } else {
                x.rem_euclid(y)
            }
        }
        BinOp::Pow => np_int32_pow(x, y),
        BinOp::Max => x.max(y),
        BinOp::Min => x.min(y),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy operation on int32"
            )
        ),
    }
}

fn apply_bin_bool(op: BinOp, x: bool, y: bool) -> bool {
    match op {
        BinOp::Add | BinOp::BitOr | BinOp::Max => x || y,
        BinOp::BitAnd | BinOp::Min => x && y,
        BinOp::BitXor => x != y,
        BinOp::Sub | BinOp::Mul => x != y,
        BinOp::Lt => !x && y,
        BinOp::Le => !x || y,
        BinOp::Gt => x && !y,
        BinOp::Ge => x || !y,
        BinOp::Eq => x == y,
        BinOp::Ne => x != y,
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy operation on bool arrays"
            )
        ),
    }
}

fn apply_un_f64(op: UnOp, x: f64) -> f64 {
    match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sqrt => x.sqrt(),
        UnOp::Exp => x.exp(),
        UnOp::Log => x.ln(),
        UnOp::Log2 => x.log2(),
        UnOp::Log10 => x.log10(),
        UnOp::Sin => x.sin(),
        UnOp::Cos => x.cos(),
        UnOp::Tan => x.tan(),
        UnOp::Asin => x.asin(),
        UnOp::Acos => x.acos(),
        UnOp::Atan => x.atan(),
        UnOp::Sinh => x.sinh(),
        UnOp::Cosh => x.cosh(),
        UnOp::Tanh => x.tanh(),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Sign => {
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
        UnOp::Square => x * x,
        UnOp::Reciprocal => 1.0 / x,
        UnOp::ExpM1 => x.exp_m1(),
        UnOp::Log1P => x.ln_1p(),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "predicate operations cannot write into a numeric buffer"
            )
        ),
    }
}

fn apply_un_f32(op: UnOp, x: f32) -> f32 {
    match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sqrt => x.sqrt(),
        UnOp::Exp => x.exp(),
        UnOp::Log => x.ln(),
        UnOp::Log2 => x.log2(),
        UnOp::Log10 => x.log10(),
        UnOp::Sin => x.sin(),
        UnOp::Cos => x.cos(),
        UnOp::Tan => x.tan(),
        UnOp::Asin => x.asin(),
        UnOp::Acos => x.acos(),
        UnOp::Atan => x.atan(),
        UnOp::Sinh => x.sinh(),
        UnOp::Cosh => x.cosh(),
        UnOp::Tanh => x.tanh(),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Sign => {
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
        UnOp::Square => x * x,
        UnOp::Reciprocal => 1.0 / x,
        UnOp::ExpM1 => x.exp_m1(),
        UnOp::Log1P => x.ln_1p(),
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "predicate operations cannot write into a numeric buffer"
            )
        ),
    }
}

fn apply_un_i64(op: UnOp, x: i64) -> i64 {
    match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sign => {
            if x > 0 {
                1
            } else if x < 0 {
                -1
            } else {
                0
            }
        }
        UnOp::Square => x.wrapping_mul(x),
        UnOp::Floor | UnOp::Ceil => x,
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "ufuncs that return floats cannot write into an integer buffer"
            )
        ),
    }
}

fn apply_un_i32(op: UnOp, x: i32) -> i32 {
    match op {
        UnOp::Neg => -x,
        UnOp::Abs => x.abs(),
        UnOp::Sign => {
            if x > 0 {
                1
            } else if x < 0 {
                -1
            } else {
                0
            }
        }
        UnOp::Square => x.wrapping_mul(x),
        UnOp::Floor | UnOp::Ceil => x,
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "ufuncs that return floats cannot write into an integer buffer"
            )
        ),
    }
}

fn apply_un_bool(op: UnOp, x: bool) -> bool {
    match op {
        UnOp::LogicalNot => !x,
        _ => panic!(
            "{}",
            crate::PyException::new(
                "TypeError",
                "unsupported numpy unary operation on bool arrays"
            )
        ),
    }
}

// ---------------------------------------------------------------------------
// Binary kernels
// ---------------------------------------------------------------------------

macro_rules! binary_parallel {
    ($name:ident, $t:ty, $apply:ident) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t], out: &mut [$t]) {
            if same_buffer(a, out) || same_buffer(b, out) {
                return scalar::$name(op, a, b, out);
            }
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, slot)| *slot = $apply(op, a[i], b[i]));
        }
    };
}

binary_parallel!(binary_f64, f64, apply_bin_f64);
binary_parallel!(binary_f32, f32, apply_bin_f32);
binary_parallel!(binary_i64, i64, apply_bin_i64);
binary_parallel!(binary_i32, i32, apply_bin_i32);
binary_parallel!(binary_bool, bool, apply_bin_bool);

// ---------------------------------------------------------------------------
// Unary kernels
// ---------------------------------------------------------------------------

macro_rules! unary_parallel {
    ($name:ident, $t:ty, $apply:ident) => {
        pub(crate) fn $name(op: UnOp, a: &[$t], out: &mut [$t]) {
            if same_buffer(a, out) {
                return scalar::$name(op, a, out);
            }
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, slot)| *slot = $apply(op, a[i]));
        }
    };
}

unary_parallel!(unary_f64, f64, apply_un_f64);
unary_parallel!(unary_f32, f32, apply_un_f32);
unary_parallel!(unary_i64, i64, apply_un_i64);
unary_parallel!(unary_i32, i32, apply_un_i32);
unary_parallel!(unary_bool, bool, apply_un_bool);

#[cfg(test)]
mod tests {
    use super::*;

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
            let mut r1 = vec![0.0; a.len()];
            let mut r2 = vec![0.0; a.len()];
            scalar::binary_f64(op, &a, &b, &mut r1);
            binary_f64(op, &a, &b, &mut r2);
            let b1: Vec<u64> = r1.iter().map(|v| v.to_bits()).collect();
            let b2: Vec<u64> = r2.iter().map(|v| v.to_bits()).collect();
            assert_eq!(b1, b2, "op {op:?}");
        }
    }

    #[test]
    fn same_buffer_detects_aliasing() {
        let a = vec![1.0, 2.0];
        assert!(same_buffer(&a, &a));
        let b = vec![1.0, 2.0];
        assert!(!same_buffer(&a, &b));
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
            let mut r1 = vec![0.0; a.len()];
            let mut r2 = vec![0.0; a.len()];
            scalar::unary_f64(op, &a, &mut r1);
            unary_f64(op, &a, &mut r2);
            let b1: Vec<u64> = r1.iter().map(|v| v.to_bits()).collect();
            let b2: Vec<u64> = r2.iter().map(|v| v.to_bits()).collect();
            assert_eq!(b1, b2, "op {op:?}");
        }
    }
}
