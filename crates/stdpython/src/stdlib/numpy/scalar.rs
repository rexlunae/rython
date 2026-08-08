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

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

macro_rules! binary_num {
    ($name:ident, $t:ty, $mod:expr, $max:expr, $min:expr, $floordiv:expr, $pow:expr, $mod_fn:expr, $zero_msg:expr, $zero:expr) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());
            for i in 0..a.len() {
                out[i] = match op {
                    BinOp::Add => a[i] + b[i],
                    BinOp::Sub => a[i] - b[i],
                    BinOp::Mul => a[i] * b[i],
                    BinOp::Div => {
                        if b[i] == $zero {
                            div_zero_int($zero_msg)
                        } else {
                            a[i] / b[i]
                        }
                    }
                    BinOp::FloorDiv => {
                        if b[i] == $zero {
                            div_zero_int($zero_msg)
                        } else {
                            $floordiv(a[i], b[i])
                        }
                    }
                    BinOp::Mod => {
                        if b[i] == $zero {
                            div_zero_int($zero_msg)
                        } else {
                            $mod_fn(a[i], b[i])
                        }
                    }
                    BinOp::Pow => $pow(a[i], b[i]),
                    BinOp::Max => $max(a[i], b[i]),
                    BinOp::Min => $min(a[i], b[i]),
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
                };
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

pub(crate) fn binary_bool(op: BinOp, a: &[bool], b: &[bool], out: &mut [bool]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = match op {
            BinOp::Add | BinOp::BitOr | BinOp::Max => a[i] || b[i],
            BinOp::BitAnd | BinOp::Min => a[i] && b[i],
            BinOp::BitXor => a[i] != b[i],
            BinOp::Sub | BinOp::Mul => a[i] != b[i],
            BinOp::Lt => !a[i] && b[i],
            BinOp::Le => !a[i] || b[i],
            BinOp::Gt => a[i] && !b[i],
            BinOp::Ge => a[i] || !b[i],
            BinOp::Eq => a[i] == b[i],
            BinOp::Ne => a[i] != b[i],
            BinOp::Div | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow => {
                panic!(
                    "{}",
                    crate::PyException::new(
                        "TypeError",
                        "unsupported numpy operation on bool arrays"
                    )
                )
            }
        };
    }
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

macro_rules! unary_float {
    ($name:ident, $t:ty, $sign:expr) => {
        pub(crate) fn $name(op: UnOp, a: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), out.len());
            for i in 0..a.len() {
                let x = a[i];
                out[i] = match op {
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
                    UnOp::Sign => $sign(x),
                    UnOp::Square => x * x,
                    UnOp::Reciprocal => 1.0 / x,
                    UnOp::ExpM1 => x.exp_m1(),
                    UnOp::Log1P => x.ln_1p(),
                    UnOp::IsFinite | UnOp::IsInf | UnOp::IsNan | UnOp::LogicalNot => {
                        panic!(
                            "{}",
                            crate::PyException::new(
                                "TypeError",
                                "predicate operations cannot write into a numeric buffer"
                            )
                        )
                    }
                };
            }
        }
    };
}

unary_float!(unary_f64, f64, np_sign_f64);
unary_float!(unary_f32, f32, np_sign_f32);

macro_rules! unary_int {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: UnOp, a: &[$t], out: &mut [$t]) {
            debug_assert_eq!(a.len(), out.len());
            for i in 0..a.len() {
                let x = a[i];
                out[i] = match op {
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
                };
            }
        }
    };
}

unary_int!(unary_i64, i64);
unary_int!(unary_i32, i32);

pub(crate) fn unary_bool(op: UnOp, a: &[bool], out: &mut [bool]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = match op {
            UnOp::LogicalNot => !a[i],
            _ => panic!(
                "{}",
                crate::PyException::new(
                    "TypeError",
                    "unsupported numpy unary operation on bool arrays"
                )
            ),
        };
    }
}
