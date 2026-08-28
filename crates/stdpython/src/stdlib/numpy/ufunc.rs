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
use std::borrow::Cow;

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
    // A 0-d source (every scalar operand of a ufunc) maps every output
    // index to element 0, so it is a FILL. The general path below builds an
    // n-element `Vec<usize>` of source indices first and then gathers
    // through it — two full-size allocations and two passes to write one
    // repeated value, which made `np.add(a, 1.0)` cost more than
    // `np.add(a, b)` despite doing less work (issue #200).
    if a.ndim == 0 {
        let n: usize = shape.iter().product();
        let data = match &a.data {
            Data::F64(v) => Data::F64(vec![v[0]; n]),
            Data::F32(v) => Data::F32(vec![v[0]; n]),
            Data::I64(v) => Data::I64(vec![v[0]; n]),
            Data::I32(v) => Data::I32(vec![v[0]; n]),
            Data::Bool(v) => Data::Bool(vec![v[0]; n]),
        };
        return NdArray::new(shape.to_vec(), a.dtype, data);
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

/// The two operands of a binary op, normalized to a common broadcast shape.
/// Accepts arrays and the three scalar kinds in any position.
///
/// Public because it appears in the generic bounds of the public ufunc
/// API (`np.add`, `np.equal`, ...); treat it as an implementation detail.
#[doc(hidden)]
pub enum BinaryOperand<'a> {
    /// BORROWED: the kernel only reads its operands, so an owned array at
    /// the call site (which cloned a full-size buffer per op — issue #200
    /// follow-up) is never needed.
    Array(&'a NdArray),
    I64(i64),
    F64(f64),
    Bool(bool),
}

impl<'a> From<&'a NdArray> for BinaryOperand<'a> {
    fn from(a: &'a NdArray) -> Self {
        BinaryOperand::Array(a)
    }
}
impl<'a> From<&'a f64> for BinaryOperand<'a> {
    fn from(v: &'a f64) -> Self {
        BinaryOperand::F64(*v)
    }
}
impl<'a> From<&'a i64> for BinaryOperand<'a> {
    fn from(v: &'a i64) -> Self {
        BinaryOperand::I64(*v)
    }
}
impl<'a> From<&'a bool> for BinaryOperand<'a> {
    fn from(v: &'a bool) -> Self {
        BinaryOperand::Bool(*v)
    }
}
impl<'a> From<i64> for BinaryOperand<'a> {
    fn from(v: i64) -> Self {
        BinaryOperand::I64(v)
    }
}
impl<'a> From<f64> for BinaryOperand<'a> {
    fn from(v: f64) -> Self {
        BinaryOperand::F64(v)
    }
}
impl<'a> From<bool> for BinaryOperand<'a> {
    fn from(v: bool) -> Self {
        BinaryOperand::Bool(v)
    }
}

/// Elementwise binary op with numpy broadcasting and dtype promotion.
///
/// The INFALLIBLE spelling, for the operator traits (`a + b`), whose
/// associated `Output` type has no room for a `Result` — a broadcast
/// mismatch there panics with the same message (spec §12.2).
pub(crate) fn binary<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    op: BinOp,
    l: L,
    r: R,
) -> NdArray {
    binary_checked(op, l, r).unwrap_or_else(|e| panic!("{e}"))
}

/// Elementwise binary op that RAISES a broadcast mismatch instead of
/// panicking: `np.add(a, b)` and the rest of the module-level ufuncs
/// propagate this with `?`, so the ValueError is catchable exactly as in
/// CPython (issue #205).
pub(crate) fn binary_checked<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    op: BinOp,
    l: L,
    r: R,
) -> Result<NdArray, PyException> {
    let (a, b) = (l.into(), r.into());
    let (a, b, shape) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape)?;
            (a, b, shape)
        }
        // Array x scalar: the scalar stays a REGISTER operand (numpy's
        // model) — the previous spelling materialized it into a full-size
        // array via scalar_to_dtype, an extra fill+read pass per op
        // (issue #200 follow-up; the chain benchmark ran ~2x numpy's
        // memory traffic because of it).
        (BinaryOperand::Array(a), BinaryOperand::I64(v)) => {
            return binary_array_scalar(op, a, &BinaryOperand::I64(v), false);
        }
        (BinaryOperand::I64(v), BinaryOperand::Array(a)) => {
            return binary_array_scalar(op, a, &BinaryOperand::I64(v), true);
        }
        (BinaryOperand::Array(a), BinaryOperand::F64(v)) => {
            return binary_array_scalar(op, a, &BinaryOperand::F64(v), false);
        }
        (BinaryOperand::F64(v), BinaryOperand::Array(a)) => {
            return binary_array_scalar(op, a, &BinaryOperand::F64(v), true);
        }
        (BinaryOperand::Array(a), BinaryOperand::Bool(v)) => {
            return binary_array_scalar(op, a, &BinaryOperand::Bool(v), false);
        }
        (BinaryOperand::Bool(v), BinaryOperand::Array(a)) => {
            return binary_array_scalar(op, a, &BinaryOperand::Bool(v), true);
        }
        (BinaryOperand::I64(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_i64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::I64(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_f64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::F64(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_i64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::F64(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_f64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::Bool(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_bool(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::I64(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_i64(x);
            let b = scalar_arr_bool(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::Bool(x), BinaryOperand::I64(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_i64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::F64(x), BinaryOperand::Bool(y)) => {
            let a = scalar_arr_f64(x);
            let b = scalar_arr_bool(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
        (BinaryOperand::Bool(x), BinaryOperand::F64(y)) => {
            let a = scalar_arr_bool(x);
            let b = scalar_arr_f64(y);
            return Ok(NdArray::binary_same_shape(op, &a, &b));
        }
    };
    // `binary_same_shape` only reads its operands, so an operand that is
    // already the output shape is BORROWED rather than cloned: the common
    // same-shape `np.add(a, b)` allocated three full-size buffers to
    // produce one (issue #200).
    let a = if a.shape.as_slice() == shape.as_slice() {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(broadcast_to(a, &shape))
    };
    let b = if b.shape.as_slice() == shape.as_slice() {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(broadcast_to(b, &shape))
    };
    Ok(NdArray::binary_same_shape(op, &a, &b))
}

fn binary_array_scalar(
    op: BinOp,
    a: &NdArray,
    scalar: &BinaryOperand<'_>,
    s_left: bool,
) -> Result<NdArray, PyException> {
    let mut d = weak_promote(a.dtype, scalar);
    // numpy: true_divide on INT arrays returns float64 (the array-array
    // kernel promotes the same way via binary_same_shape).
    if op == BinOp::Div && matches!(d, Dtype::Int32 | Dtype::Int64 | Dtype::Bool) {
        d = Dtype::Float64;
    }
    // A python int bound to an int32 array that does not fit int32 raises
    // numpy's OverflowError before any computation (scalar_to_dtype
    // semantics — a panic, like the materialized spelling).
    if a.dtype == Dtype::Int32 && d == Dtype::Int32 {
        if let BinaryOperand::I64(x) = scalar {
            if i32::try_from(*x).is_err() {
                panic!(
                    "{}",
                    PyException::new(
                        "OverflowError",
                        format!("Python integer {x} out of bounds for int32")
                    )
                );
            }
        }
    }
    // COMPARISON ops (Lt..Ne) produce elementwise BOOL arrays with dtype
    // promotion — the arithmetic kernels below cannot express that. Route
    // them through the array-array kernel by materializing the scalar
    // (comparisons are not in the hot elementwise-arithmetic path).
    if matches!(
        op,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
    ) {
        let scalar_arr = match (d, scalar) {
            (Dtype::Int32, BinaryOperand::I64(x)) => {
                match i32::try_from(*x) {
                    Ok(v) => scalar_arr_i32(v),
                    Err(_) => panic!(
                        "{}",
                        PyException::new(
                            "OverflowError",
                            format!("Python integer {x} out of bounds for int32")
                        )
                    ),
                }
            }
            (Dtype::Int64, BinaryOperand::I64(x)) => scalar_arr_i64(*x),
            (Dtype::Float32, BinaryOperand::I64(x)) => scalar_arr_f32(*x as f32),
            (Dtype::Float32, BinaryOperand::F64(x)) => scalar_arr_f32(*x as f32),
            (Dtype::Float64, BinaryOperand::F64(x)) => scalar_arr_f64(*x),
            (Dtype::Float64, BinaryOperand::I64(x)) => scalar_arr_f64(*x as f64),
            (Dtype::Bool, BinaryOperand::Bool(x)) => scalar_arr_bool(*x),
            _ => {
                panic!(
                    "{}",
                    PyException::new(
                        "TypeError",
                        "unsupported array/scalar dtype combination"
                    )
                )
            }
        };
        return Ok(NdArray::binary_same_shape(op, a, &scalar_arr));
    }
    // A python float on an int/bool array widens to float64: promote the
    // array (one convert pass — numpy does the same), then compute in the
    // promoted dtype.
    let a2 = if a.dtype == d {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(a.astype(d))
    };
    let data = match (&a2.data, scalar) {
        (Data::F64(xs), BinaryOperand::F64(s)) => {
            Data::F64(engine::binary_f64_scalar(op, xs, *s, s_left))
        }
        (Data::F64(xs), BinaryOperand::I64(s)) => {
            Data::F64(engine::binary_f64_scalar(op, xs, *s as f64, s_left))
        }
        (Data::F64(xs), BinaryOperand::Bool(s)) => {
            Data::F64(engine::binary_f64_scalar(op, xs, *s as i64 as f64, s_left))
        }
        (Data::F32(xs), BinaryOperand::F64(s)) => {
            Data::F32(engine::binary_f32_scalar(op, xs, *s as f32, s_left))
        }
        (Data::F32(xs), BinaryOperand::I64(s)) => {
            Data::F32(engine::binary_f32_scalar(op, xs, *s as f32, s_left))
        }
        (Data::F32(xs), BinaryOperand::Bool(s)) => {
            Data::F32(engine::binary_f32_scalar(op, xs, *s as i64 as f32, s_left))
        }
        (Data::I64(xs), BinaryOperand::I64(s)) => {
            Data::I64(engine::binary_i64_scalar(op, xs, *s, s_left))
        }
        (Data::I64(xs), BinaryOperand::Bool(s)) => {
            Data::I64(engine::binary_i64_scalar(op, xs, *s as i64, s_left))
        }
        (Data::I32(xs), BinaryOperand::I64(s)) => {
            Data::I32(engine::binary_i32_scalar(op, xs, *s as i32, s_left))
        }
        (Data::I32(xs), BinaryOperand::Bool(s)) => {
            Data::I32(engine::binary_i32_scalar(op, xs, *s as i64 as i32, s_left))
        }
        (Data::Bool(xs), BinaryOperand::Bool(s)) => {
            Data::Bool(engine::binary_bool_scalar(op, xs, *s, s_left))
        }
        _ => {
            panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "unsupported array/scalar dtype combination"
                )
            )
        }
    };
    Ok(NdArray::new(a.shape.clone(), d, data))
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
            pub fn $name<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
                a: L,
                b: R,
            ) -> Result<NdArray, PyException> {
                binary_checked($op, a, b)
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
pub fn remainder<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(a: L, b: R) -> NdArray {
    binary(BinOp::Mod, a, b)
}

/// `np.logical_and(a, b)` — truthiness of each element, then `&`.
pub fn logical_and<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    a: L,
    b: R,
) -> Result<NdArray, PyException> {
    logical(BinOp::BitAnd, a, b)
}
pub fn logical_or<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    a: L,
    b: R,
) -> Result<NdArray, PyException> {
    logical(BinOp::BitOr, a, b)
}
pub fn logical_xor<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    a: L,
    b: R,
) -> Result<NdArray, PyException> {
    logical(BinOp::BitXor, a, b)
}

fn logical<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    op: BinOp,
    a: L,
    b: R,
) -> Result<NdArray, PyException> {
    let (a, b) = (a.into(), b.into());
    let (a, b) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape)?;
            (
                Cow::Owned(broadcast_to(a, &shape)),
                Cow::Owned(broadcast_to(b, &shape)),
            )
        }
        (BinaryOperand::Array(a), scalar) => {
            (Cow::Borrowed(a), Cow::Owned(bool_of_scalar(scalar)))
        }
        (scalar, BinaryOperand::Array(b)) => {
            (Cow::Owned(bool_of_scalar(scalar)), Cow::Borrowed(b))
        }
        _ => panic!(
            "{}",
            PyException::new("TypeError", "logical ops need at least one array")
        ),
    };
    let a = bool_array(&a);
    let b = bool_array(&b);
    let out = engine::binary_bool(op, &a.bool(), &b.bool());
    Ok(NdArray::new(a.shape.clone(), Dtype::Bool, Data::Bool(out)))
}

fn bool_of_scalar(s: BinaryOperand) -> NdArray {
    match s {
        BinaryOperand::I64(v) => scalar_arr_bool(v != 0),
        BinaryOperand::F64(v) => scalar_arr_bool(v != 0.0),
        BinaryOperand::Bool(v) => scalar_arr_bool(v),
        BinaryOperand::Array(a) => a.clone(),
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
            pub fn $name(a: &NdArray) -> NdArray {
                NdArray::unary($op, a)
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
            pub fn $name(a: &NdArray) -> NdArray {
                NdArray::unary($op, &as_float_array(a))
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
pub fn where_<'a, L: Into<BinaryOperand<'a>>, R: Into<BinaryOperand<'a>>>(
    cond: &NdArray,
    a: L,
    b: R,
) -> Result<NdArray, PyException> {
    let (a, b) = (a.into(), b.into());
    let (a, b) = match (a, b) {
        (BinaryOperand::Array(a), BinaryOperand::Array(b)) => {
            let shape = broadcast_shapes(&a.shape, &b.shape)?;
            (broadcast_to(&a, &shape), broadcast_to(&b, &shape))
        }
        (BinaryOperand::Array(a), BinaryOperand::I64(v)) => {
            let shape = broadcast_shapes(&a.shape, &[])?;
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
            let shape = broadcast_shapes(&a.shape, &[])?;
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
            let shape = broadcast_shapes(&a.shape, &[])?;
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
    Ok(NdArray::new(shape, out_dtype, data))
}
