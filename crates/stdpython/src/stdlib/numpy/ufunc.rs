//! numpy universal functions (elementwise operations).
//!
//! The ufunc API mirrors numpy's module-level functions: `np.add(a, b)`,
//! `np.sqrt(a)`, `np.equal(a, b)`, ... Every op funnels through the
//! engine backends (scalar / rayon / simd / cuda / vulkan).
//!
//! Broadcasting follows numpy: right-aligned shapes, a dimension of size 1
//! stretches to match, and mismatched dimensions raise a Python ValueError.
//! Binary ops accept scalars (i64/f64/bool) on either side.

use super::dtype::Dtype;
use super::engine::{self, BinOp, UnOp};
use super::ndarray::{Data, NdArray};
use crate::PyException;

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// numpy's broadcast rule for two shapes, right-aligned.
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, PyException> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        out[i] = match (da, db) {
            (0, 0) => 0,
            (0, _) | (_, 0) => 0,
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(PyException::new(
                    "ValueError",
                    format!(
                        "operands could not be broadcast together with shapes {:?} {:?}",
                        a, b
                    ),
                ));
            }
        };
    }
    Ok(out)
}

/// Stretch `a` to `shape` (a broadcast-compatible supershape) by copying
/// along size-1 axes. Scalar (0-d) arrays stretch everywhere.
pub(crate) fn broadcast_to(a: &NdArray, shape: &[usize]) -> NdArray {
    if a.shape.as_slice() == shape {
        return a.clone();
    }
    let n: usize = shape.iter().product();
    // Right-align the source shape against the target: leading size-1
    // dimensions are added when the source has fewer dims (numpy
    // broadcasting), so every src_shape[i] below is in bounds.
    let src_shape: Vec<usize> = if a.ndim == 0 {
        vec![1; shape.len()]
    } else {
        let mut s = vec![1usize; shape.len() - a.shape.len()];
        s.extend_from_slice(&a.shape);
        s
    };
    let rank = shape.len();
    let src_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            s[i] = s[i + 1] * src_shape[i + 1];
        }
        s
    };
    // For each output flat index, compute the source flat index: a
    // dimension of size 1 always maps to source coordinate 0.
    let dst_strides: Vec<usize> = {
        let mut s = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    let perm: Vec<usize> = (0..n)
        .map(|dst| {
            let mut rem = dst;
            let mut src = 0usize;
            for i in 0..rank {
                let c = rem / dst_strides[i];
                rem %= dst_strides[i];
                let sc = if src_shape[i] == 1 { 0 } else { c };
                src += sc * src_strides[i];
            }
            src
        })
        .collect();
    let data = match &a.data {
        Data::F64(v) => Data::F64(perm.iter().map(|&i| v[i]).collect()),
        Data::F32(v) => Data::F32(perm.iter().map(|&i| v[i]).collect()),
        Data::I64(v) => Data::I64(perm.iter().map(|&i| v[i]).collect()),
        Data::I32(v) => Data::I32(perm.iter().map(|&i| v[i]).collect()),
        Data::Bool(v) => Data::Bool(perm.iter().map(|&i| v[i]).collect()),
    };
    NdArray::new(shape.to_vec(), a.dtype, data)
}

fn scalar_arr_i64(v: i64) -> NdArray {
    NdArray::new(vec![], Dtype::Int64, Data::I64(vec![v]))
}
fn scalar_arr_f64(v: f64) -> NdArray {
    NdArray::new(vec![], Dtype::Float64, Data::F64(vec![v]))
}
fn scalar_arr_bool(v: bool) -> NdArray {
    NdArray::new(vec![], Dtype::Bool, Data::Bool(vec![v]))
}
fn scalar_arr_i32(v: i32) -> NdArray {
    NdArray::new(vec![], Dtype::Int32, Data::I32(vec![v]))
}
fn scalar_arr_f32(v: f32) -> NdArray {
    NdArray::new(vec![], Dtype::Float32, Data::F32(vec![v]))
}

/// NEP 50 weak promotion of a Python scalar against an array dtype: a
/// Python float never widens a float32 array (f32 + 0.0 stays f32) and a
/// Python int never widens an int32 array (i32 + 1 stays i32), while a
/// Python float does widen integer arrays to float64 (verified against
/// numpy 2; `bool_arr + 1` is int64, `bool_arr + True` stays bool).
fn weak_promote(d: Dtype, scalar: &BinaryOperand) -> Dtype {
    use Dtype::*;
    match scalar {
        BinaryOperand::F64(_) => match d {
            Bool | Int32 | Int64 => Float64,
            Float32 => Float32,
            Float64 => Float64,
        },
        BinaryOperand::I64(_) => match d {
            Bool => Int64,
            Int32 => Int32,
            Int64 => Int64,
            Float32 => Float32,
            Float64 => Float64,
        },
        BinaryOperand::Bool(_) => match d {
            Bool => Bool,
            Int32 => Int32,
            Int64 => Int64,
            Float32 => Float32,
            Float64 => Float64,
        },
        BinaryOperand::Array(_) => unreachable!("weak_promote needs a scalar"),
    }
}

/// Build a 0-d array holding `v` cast to `dtype` — the weak-promoted target.
/// A Python int that does not fit an int32 target raises numpy's
/// `OverflowError` ("Python integer ... out of bounds for int32").
fn scalar_to_dtype(v: BinaryOperand, dtype: Dtype) -> NdArray {
    match v {
        BinaryOperand::I64(x) => match dtype {
            Dtype::Int32 => match i32::try_from(x) {
                Ok(v) => scalar_arr_i32(v),
                Err(_) => panic!(
                    "{}",
                    crate::PyException::new(
                        "OverflowError",
                        format!("Python integer {x} out of bounds for int32")
                    )
                ),
            },
            Dtype::Int64 => scalar_arr_i64(x),
            Dtype::Float32 => scalar_arr_f32(x as f32),
            Dtype::Float64 => scalar_arr_f64(x as f64),
            Dtype::Bool => unreachable!("weak int never promotes to bool"),
        },
        BinaryOperand::F64(x) => match dtype {
            Dtype::Float32 => scalar_arr_f32(x as f32),
            Dtype::Float64 => scalar_arr_f64(x),
            Dtype::Bool | Dtype::Int32 | Dtype::Int64 => {
                unreachable!("weak float never promotes to int/bool")
            }
        },
        BinaryOperand::Bool(x) => match dtype {
            Dtype::Bool => scalar_arr_bool(x),
            Dtype::Int32 => scalar_arr_i32(x as i32),
            Dtype::Int64 => scalar_arr_i64(x as i64),
            Dtype::Float32 => scalar_arr_f32(if x { 1.0 } else { 0.0 }),
            Dtype::Float64 => scalar_arr_f64(if x { 1.0 } else { 0.0 }),
        },
        BinaryOperand::Array(_) => unreachable!("scalar_to_dtype needs a scalar"),
    }
}

/// The two operands of a binary op, normalized to a common broadcast shape.
/// Accepts arrays and the three scalar kinds in any position.
///
/// Public because it appears in the generic bounds of the public ufunc
/// API (`np.add`, `np.equal`, ...); treat it as an implementation detail.
#[doc(hidden)]
pub enum BinaryOperand {
    Array(NdArray),
    I64(i64),
    F64(f64),
    Bool(bool),
}

impl From<NdArray> for BinaryOperand {
    fn from(a: NdArray) -> Self {
        BinaryOperand::Array(a)
    }
}
impl From<i64> for BinaryOperand {
    fn from(v: i64) -> Self {
        BinaryOperand::I64(v)
    }
}
impl From<f64> for BinaryOperand {
    fn from(v: f64) -> Self {
        BinaryOperand::F64(v)
    }
}
impl From<bool> for BinaryOperand {
    fn from(v: bool) -> Self {
        BinaryOperand::Bool(v)
    }
}

/// Elementwise binary op with numpy broadcasting and dtype promotion.
pub(crate) fn binary<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(
    op: BinOp,
    l: L,
    r: R,
) -> NdArray {
    let (a, b) = (l.into(), r.into());
    let (a, b, shape) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape).unwrap_or_else(|e| panic!("{}", e));
            (a, b, shape)
        }
        (BinaryOperand::Array(a), BinaryOperand::I64(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::I64(v));
            (a, scalar_to_dtype(BinaryOperand::I64(v), d), shape)
        }
        (BinaryOperand::I64(v), BinaryOperand::Array(a)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::I64(v));
            (scalar_to_dtype(BinaryOperand::I64(v), d), a, shape)
        }
        (BinaryOperand::Array(a), BinaryOperand::F64(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::F64(v));
            (a, scalar_to_dtype(BinaryOperand::F64(v), d), shape)
        }
        (BinaryOperand::F64(v), BinaryOperand::Array(a)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::F64(v));
            (scalar_to_dtype(BinaryOperand::F64(v), d), a, shape)
        }
        (BinaryOperand::Array(a), BinaryOperand::Bool(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::Bool(v));
            (a, scalar_to_dtype(BinaryOperand::Bool(v), d), shape)
        }
        (BinaryOperand::Bool(v), BinaryOperand::Array(a)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            let d = weak_promote(a.dtype, &BinaryOperand::Bool(v));
            (scalar_to_dtype(BinaryOperand::Bool(v), d), a, shape)
        }
        (BinaryOperand::I64(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_i64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::I64(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_f64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::F64(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_i64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::F64(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_f64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::Bool(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_bool(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::I64(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_bool(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::Bool(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_i64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::F64(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_bool(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
        (BinaryOperand::Bool(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_f64(y);
            return NdArray::binary_same_shape(op, &a, &b);
        }
    };
    let a = broadcast_to(&a, &shape);
    let b = broadcast_to(&b, &shape);
    NdArray::binary_same_shape(op, &a, &b)
}

/// Convert a numeric array to f64 for float-returning ufuncs (numpy
/// promotes int arrays to float64 for sqrt/exp/...).
pub(crate) fn as_float_array(a: &NdArray) -> NdArray {
    match a.dtype {
        Dtype::Float64 => a.clone(),
        _ => a.astype(Dtype::Float64),
    }
}

// ---------------------------------------------------------------------------
// Binary ufuncs (module-level numpy API)
// ---------------------------------------------------------------------------

macro_rules! binary_ufunc {
    ($($name:ident, $op:expr),* $(,)?) => {
        $(
            #[doc = concat!("`np.", stringify!($name), "(a, b)` — elementwise with broadcasting.")]
            pub fn $name<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(a: L, b: R) -> NdArray {
                binary($op, a, b)
            }
        )*
    };
}

binary_ufunc!(
    add,
    BinOp::Add,
    subtract,
    BinOp::Sub,
    multiply,
    BinOp::Mul,
    divide,
    BinOp::Div,
    floor_divide,
    BinOp::FloorDiv,
    mod_,
    BinOp::Mod,
    power,
    BinOp::Pow,
    maximum,
    BinOp::Max,
    minimum,
    BinOp::Min,
    equal,
    BinOp::Eq,
    not_equal,
    BinOp::Ne,
    less,
    BinOp::Lt,
    less_equal,
    BinOp::Le,
    greater,
    BinOp::Gt,
    greater_equal,
    BinOp::Ge,
    bitwise_and,
    BinOp::BitAnd,
    bitwise_or,
    BinOp::BitOr,
    bitwise_xor,
    BinOp::BitXor,
);

/// `np.remainder(a, b)` — alias of np.mod.
pub fn remainder<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(a: L, b: R) -> NdArray {
    binary(BinOp::Mod, a, b)
}

/// `np.logical_and(a, b)` — truthiness of each element, then `&`.
pub fn logical_and<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(a: L, b: R) -> NdArray {
    logical(BinOp::BitAnd, a, b)
}
pub fn logical_or<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(a: L, b: R) -> NdArray {
    logical(BinOp::BitOr, a, b)
}
pub fn logical_xor<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(a: L, b: R) -> NdArray {
    logical(BinOp::BitXor, a, b)
}

fn logical<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(op: BinOp, a: L, b: R) -> NdArray {
    let (a, b) = (a.into(), b.into());
    let (a, b) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape).unwrap_or_else(|e| panic!("{}", e));
            (broadcast_to(&a, &shape), broadcast_to(&b, &shape))
        }
        (BinaryOperand::Array(a), b) => (a, bool_of_scalar(b)),
        (a, BinaryOperand::Array(b)) => (bool_of_scalar(a), b),
        _ => panic!(
            "{}",
            PyException::new("TypeError", "logical ops need at least one array")
        ),
    };
    let a = bool_array(&a);
    let b = bool_array(&b);
    let out = engine::binary_bool(op, &a.bool(), &b.bool());
    NdArray::new(a.shape.clone(), Dtype::Bool, Data::Bool(out))
}

fn bool_of_scalar(s: BinaryOperand) -> NdArray {
    match s {
        BinaryOperand::I64(v) => scalar_arr_bool(v != 0),
        BinaryOperand::F64(v) => scalar_arr_bool(v != 0.0),
        BinaryOperand::Bool(v) => scalar_arr_bool(v),
        BinaryOperand::Array(a) => a,
    }
}

fn bool_array(a: &NdArray) -> NdArray {
    if a.dtype == Dtype::Bool {
        a.clone()
    } else {
        a.astype(Dtype::Bool)
    }
}

// ---------------------------------------------------------------------------
// Unary ufuncs
// ---------------------------------------------------------------------------

macro_rules! unary_ufunc {
    ($($name:ident, $op:expr),* $(,)?) => {
        $(
            #[doc = concat!("`np.", stringify!($name), "(a)` — elementwise.")]
            pub fn $name(a: NdArray) -> NdArray {
                NdArray::unary($op, &a)
            }
        )*
    };
}

unary_ufunc!(
    negative,
    UnOp::Neg,
    abs,
    UnOp::Abs,
    square,
    UnOp::Square,
    sign,
    UnOp::Sign,
    isfinite,
    UnOp::IsFinite,
    isinf,
    UnOp::IsInf,
    isnan,
    UnOp::IsNan,
    logical_not,
    UnOp::LogicalNot,
);

/// Float-only unary ufuncs: int/bool arrays are promoted to float64 first,
/// exactly like numpy (`np.sqrt(np.array([4]))` → float64).
macro_rules! unary_float_ufunc {
    ($($name:ident, $op:expr),* $(,)?) => {
        $(
            #[doc = concat!("`np.", stringify!($name), "(a)` — elementwise (promotes int arrays to float64).")]
            pub fn $name(a: NdArray) -> NdArray {
                NdArray::unary($op, &as_float_array(&a))
            }
        )*
    };
}

unary_float_ufunc!(
    sqrt,
    UnOp::Sqrt,
    exp,
    UnOp::Exp,
    log,
    UnOp::Log,
    log2,
    UnOp::Log2,
    log10,
    UnOp::Log10,
    sin,
    UnOp::Sin,
    cos,
    UnOp::Cos,
    tan,
    UnOp::Tan,
    arcsin,
    UnOp::Asin,
    arccos,
    UnOp::Acos,
    arctan,
    UnOp::Atan,
    sinh,
    UnOp::Sinh,
    cosh,
    UnOp::Cosh,
    tanh,
    UnOp::Tanh,
    floor,
    UnOp::Floor,
    ceil,
    UnOp::Ceil,
    reciprocal,
    UnOp::Reciprocal,
    expm1,
    UnOp::ExpM1,
    log1p,
    UnOp::Log1P,
);

/// `np.clip(a, min, max)` — elementwise clamp. numpy semantics: values
/// below min become min, above max become max; either bound may be None
/// (represented by `f64::NAN` sentinel is NOT used — use Option instead).
/// The scalar signature takes Option so `np.clip(a, None, 10)` works.
pub fn clip<T: Into<f64>>(a: NdArray, min: Option<T>, max: Option<T>) -> NdArray {
    let minv = min.map(Into::into);
    let maxv = max.map(Into::into);
    match a.dtype {
        Dtype::Int64 | Dtype::Int32 => {
            let vals: Vec<i64> = a.as_i64();
            let lo = minv.map(|v: f64| v as i64).unwrap_or(i64::MIN);
            let hi = maxv.map(|v: f64| v as i64).unwrap_or(i64::MAX);
            let out: Vec<i64> = vals.iter().map(|&x| x.clamp(lo, hi)).collect();
            let data = if a.dtype == Dtype::Int64 {
                Data::I64(out)
            } else {
                Data::I32(out.iter().map(|&x| x as i32).collect())
            };
            NdArray::new(a.shape.clone(), a.dtype, data)
        }
        Dtype::Float64 | Dtype::Float32 => {
            let vals: Vec<f64> = a.as_f64();
            let lo = minv.unwrap_or(f64::NEG_INFINITY);
            let hi = maxv.unwrap_or(f64::INFINITY);
            let out: Vec<f64> = vals
                .iter()
                .map(|&x| if x.is_nan() { x } else { x.clamp(lo, hi) })
                .collect();
            let data = if a.dtype == Dtype::Float64 {
                Data::F64(out)
            } else {
                Data::F32(out.iter().map(|&x| x as f32).collect())
            };
            NdArray::new(a.shape.clone(), a.dtype, data)
        }
        Dtype::Bool => {
            let lo = minv.unwrap_or(0.0) != 0.0;
            let hi = maxv.unwrap_or(1.0) != 0.0;
            let out: Vec<bool> = a.bool().iter().map(|&x| x && hi || !x && lo).collect();
            NdArray::new(a.shape.clone(), Dtype::Bool, Data::Bool(out))
        }
    }
}

/// `np.where(cond, a, b)` — select elementwise from two arrays or scalars.
/// The condition is truthy-tested elementwise.
pub fn where_<L: Into<BinaryOperand>, R: Into<BinaryOperand>>(
    cond: NdArray,
    a: L,
    b: R,
) -> NdArray {
    let (a, b) = (a.into(), b.into());
    let (a, b) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape).unwrap_or_else(|e| panic!("{}", e));
            (broadcast_to(&a, &shape), broadcast_to(&b, &shape))
        }
        (BinaryOperand::Array(a), BinaryOperand::I64(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&a, &shape),
                broadcast_to(&scalar_arr_i64(v), &shape),
            )
        }
        (BinaryOperand::I64(v), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&b.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&scalar_arr_i64(v), &shape),
                broadcast_to(&b, &shape),
            )
        }
        (BinaryOperand::Array(a), BinaryOperand::F64(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&a, &shape),
                broadcast_to(&scalar_arr_f64(v), &shape),
            )
        }
        (BinaryOperand::F64(v), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&b.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&scalar_arr_f64(v), &shape),
                broadcast_to(&b, &shape),
            )
        }
        (BinaryOperand::Array(a), BinaryOperand::Bool(v)) => {
            let shape = broadcast_shapes(&a.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&a, &shape),
                broadcast_to(&scalar_arr_bool(v), &shape),
            )
        }
        (BinaryOperand::Bool(v), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&b.shape, &[]).unwrap_or_else(|e| panic!("{}", e));
            (
                broadcast_to(&scalar_arr_bool(v), &shape),
                broadcast_to(&b, &shape),
            )
        }
        _ => panic!(
            "{}",
            PyException::new("TypeError", "where() needs at least one array operand")
        ),
    };
    // cond broadcasts against a/b
    let shape = broadcast_shapes(&cond.shape, &a.shape).unwrap_or_else(|e| panic!("{}", e));
    let cond = broadcast_to(&cond, &shape);
    let a = broadcast_to(&a, &shape);
    let b = broadcast_to(&b, &shape);
    let cond_b = cond.as_bool();
    let out_dtype = a.dtype.promote(b.dtype);
    let a = a.astype(out_dtype);
    let b = b.astype(out_dtype);
    let n = a.size;
    let data = match (a.data, b.data) {
        (Data::F64(x), Data::F64(y)) => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(if cond_b[i] { x[i] } else { y[i] });
            }
            Data::F64(out)
        }
        (Data::I64(x), Data::I64(y)) => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(if cond_b[i] { x[i] } else { y[i] });
            }
            Data::I64(out)
        }
        (Data::Bool(x), Data::Bool(y)) => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(if cond_b[i] { x[i] } else { y[i] });
            }
            Data::Bool(out)
        }
        _ => panic!(
            "{}",
            PyException::new("TypeError", "unsupported where() operand dtypes")
        ),
    };
    NdArray::new(shape, out_dtype, data)
}
