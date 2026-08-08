//! Scalar engine: plain sequential loops. Always compiled in; the fallback
//! for every accelerated backend. Float semantics follow numpy where numpy
//! and Rust's primitives disagree (NaN-propagating maximum/minimum, Python
//! remainder semantics, floor division toward -inf).

use super::{BinOp, ReduceOp, UnOp};

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

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Pairwise summation, which is what numpy uses for float sums: sequential
/// folding drifts from numpy's results in the last ulp on large arrays.
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    match a.len() {
        0 => 0.0,
        1 => a[0],
        n => {
            let mid = n / 2;
            pairwise_sum_f64(&a[..mid]) + pairwise_sum_f64(&a[mid..])
        }
    }
}

fn pairwise_sum_f32(a: &[f32]) -> f32 {
    match a.len() {
        0 => 0.0,
        1 => a[0],
        n => {
            let mid = n / 2;
            pairwise_sum_f32(&a[..mid]) + pairwise_sum_f32(&a[mid..])
        }
    }
}

fn mean_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN; // numpy: mean of empty → nan + warning
    }
    pairwise_sum_f64(a) / a.len() as f64
}

fn mean_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NAN;
    }
    pairwise_sum_f32(a) / a.len() as f32
}

fn var_f64(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    let m = mean_f64(a);
    let mut acc = 0.0;
    for &x in a {
        let d = x - m;
        acc += d * d;
    }
    acc / a.len() as f64
}

fn var_f32(a: &[f32]) -> f32 {
    if a.is_empty() {
        return f32::NAN;
    }
    let m = mean_f32(a);
    let mut acc = 0.0;
    for &x in a {
        let d = x - m;
        acc += d * d;
    }
    acc / a.len() as f32
}

macro_rules! reduce_float {
    ($name:ident, $t:ty, $mean:expr, $var:expr) => {
        pub(crate) fn $name(op: ReduceOp, a: &[$t]) -> $t {
            match op {
                ReduceOp::Sum => {
                    let mut acc: $t = 0.0;
                    for &x in a {
                        acc += x;
                    }
                    acc
                }
                ReduceOp::Prod => {
                    let mut acc: $t = 1.0;
                    for &x in a {
                        acc *= x;
                    }
                    acc
                }
                ReduceOp::Min => {
                    let mut best = a.first().copied().unwrap_or(0.0);
                    for &x in a {
                        if x.is_nan() || x < best {
                            best = x;
                        }
                    }
                    best
                }
                ReduceOp::Max => {
                    let mut best = a.first().copied().unwrap_or(0.0);
                    for &x in a {
                        if x.is_nan() || x > best {
                            best = x;
                        }
                    }
                    best
                }
                ReduceOp::Mean => $mean(a),
                ReduceOp::Std => $var(a).sqrt(),
                ReduceOp::Var => $var(a),
                ReduceOp::All | ReduceOp::Any => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            "all/any require a boolean array"
                        )
                    )
                }
            }
        }
    };
}

reduce_float!(reduce_f64, f64, mean_f64, var_f64);
reduce_float!(reduce_f32, f32, mean_f32, var_f32);

macro_rules! reduce_int {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: ReduceOp, a: &[$t]) -> $t {
            match op {
                ReduceOp::Sum => {
                    let mut acc: $t = 0;
                    for &x in a {
                        acc = acc.wrapping_add(x);
                    }
                    acc
                }
                ReduceOp::Prod => {
                    let mut acc: $t = 1;
                    for &x in a {
                        acc = acc.wrapping_mul(x);
                    }
                    acc
                }
                ReduceOp::Min => {
                    let mut best = a.first().copied().unwrap_or(0);
                    for &x in a {
                        if x < best {
                            best = x;
                        }
                    }
                    best
                }
                ReduceOp::Max => {
                    let mut best = a.first().copied().unwrap_or(0);
                    for &x in a {
                        if x > best {
                            best = x;
                        }
                    }
                    best
                }
                ReduceOp::Mean | ReduceOp::Std | ReduceOp::Var | ReduceOp::All
                | ReduceOp::Any => {
                    panic!(
                        "{}",
                        crate::PyException::new(
                            "TypeError",
                            "float reductions require a float array"
                        )
                    )
                }
            }
        }
    };
}

reduce_int!(reduce_i64, i64);
reduce_int!(reduce_i32, i32);

pub(crate) fn reduce_bool(op: ReduceOp, a: &[bool]) -> bool {
    match op {
        ReduceOp::All => a.iter().all(|&x| x),
        ReduceOp::Any => a.iter().any(|&x| x),
        ReduceOp::Sum => a.iter().filter(|&&x| x).count() % 2 == 1,
        ReduceOp::Min => a.iter().all(|&x| x),
        ReduceOp::Max => a.iter().any(|&x| x),
        _ => panic!(
            "{}",
            crate::PyException::new("TypeError", "unsupported reduction on bool array")
        ),
    }
}

macro_rules! arg_reduce_float {
    ($min:ident, $max:ident, $t:ty) => {
        pub(crate) fn $min(a: &[$t]) -> i64 {
            // numpy: NaN is the "largest" value for argmax, and the first
            // NaN wins; argmin ignores NaNs unless every element is NaN
            // (then it returns the first NaN's index).
            if let Some(i) = a.iter().position(|x| x.is_nan()) {
                if a.iter().all(|x| x.is_nan()) {
                    return i as i64;
                }
            }
            let mut best = 0usize;
            for (i, &x) in a.iter().enumerate() {
                if !x.is_nan() && (a[best].is_nan() || x < a[best]) {
                    best = i;
                }
            }
            best as i64
        }
        pub(crate) fn $max(a: &[$t]) -> i64 {
            if let Some(i) = a.iter().position(|x| x.is_nan()) {
                return i as i64;
            }
            let mut best = 0usize;
            for (i, &x) in a.iter().enumerate() {
                if x > a[best] {
                    best = i;
                }
            }
            best as i64
        }
    };
}

arg_reduce_float!(argmin_f64, argmax_f64, f64);
arg_reduce_float!(argmin_f32, argmax_f32, f32);

macro_rules! arg_reduce_int {
    ($min:ident, $max:ident, $t:ty) => {
        pub(crate) fn $min(a: &[$t]) -> i64 {
            let mut best = 0usize;
            for (i, &x) in a.iter().enumerate() {
                if x < a[best] {
                    best = i;
                }
            }
            best as i64
        }
        pub(crate) fn $max(a: &[$t]) -> i64 {
            let mut best = 0usize;
            for (i, &x) in a.iter().enumerate() {
                if x > a[best] {
                    best = i;
                }
            }
            best as i64
        }
    };
}

arg_reduce_int!(argmin_i64, argmax_i64, i64);
arg_reduce_int!(argmin_i32, argmax_i32, i32);

pub(crate) fn argmin_bool(a: &[bool]) -> i64 {
    a.iter().position(|&x| !x).unwrap_or(0) as i64
}
pub(crate) fn argmax_bool(a: &[bool]) -> i64 {
    a.iter().position(|&x| x).unwrap_or(0) as i64
}
