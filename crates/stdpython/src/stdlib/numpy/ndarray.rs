//! The `NdArray` value type: a dense, row-major, contiguous array.
//!
//! Rython arrays are **values** (copies), not views: indexing, slicing, and
//! reshaping copy the touched elements. This matches how every other rython
//! type behaves (lists are values too) and keeps the runtime free of borrow
//! tracking. The cost is that `a[1:]` copies — fine for the scale this
//! subset targets.
//!
//! `shape`, `ndim`, `size`, and `dtype` are public fields so that Python
//! attribute access (`a.shape`, `a.ndim`) lowers to plain Rust field reads.

use super::dtype::Dtype;
use super::engine::{self, BinOp, UnOp};
use crate::{PyException, PyIndex, PyRepr};

/// Convert a Python `shape` argument into the `Vec<i64>` numpy works with.
/// Python shape tuples `(2, 3)` lower to Rust tuples and lists `[2, 3]` to
/// `Vec<i64>`; both convert. (Rust has no variadics, so tuples are covered
/// up to 6 dimensions — beyond that, use a list.)
pub trait IntoShape {
    fn into_shape(self) -> Vec<i64>;
}

impl IntoShape for (i64,) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0]
    }
}
impl IntoShape for (i64, i64) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0, self.1]
    }
}
impl IntoShape for (i64, i64, i64) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0, self.1, self.2]
    }
}
impl IntoShape for (i64, i64, i64, i64) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0, self.1, self.2, self.3]
    }
}
impl IntoShape for (i64, i64, i64, i64, i64) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}
impl IntoShape for (i64, i64, i64, i64, i64, i64) {
    fn into_shape(self) -> Vec<i64> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}

impl IntoShape for i64 {
    fn into_shape(self) -> Vec<i64> {
        vec![self]
    }
}

impl IntoShape for Vec<i64> {
    fn into_shape(self) -> Vec<i64> {
        self
    }
}

impl IntoShape for &[i64] {
    fn into_shape(self) -> Vec<i64> {
        self.to_vec()
    }
}

/// The storage behind an `NdArray`. Always row-major contiguous.
#[derive(Clone, Debug)]
pub(crate) enum Data {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
}

/// A dense N-dimensional array (numpy subset).
#[derive(Clone, Debug)]
pub struct NdArray {
    pub shape: Vec<usize>,
    pub ndim: usize,
    pub size: usize,
    pub dtype: Dtype,
    pub(crate) data: Data,
}

/// Elementwise comparison helper for the bool-array result of `==`, `<`,
/// etc. — numpy semantics: comparisons yield a bool NdArray.
fn cmp_elems<T: PartialOrd + Copy>(op: BinOp, a: &[T], b: &[T]) -> Vec<bool> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| match op {
            BinOp::Lt => x < y,
            BinOp::Le => x <= y,
            BinOp::Gt => x > y,
            BinOp::Ge => x >= y,
            BinOp::Eq => x == y,
            BinOp::Ne => x != y,
            _ => unreachable!("cmp_elems called with non-comparison op"),
        })
        .collect()
}

/// Predicate semantics per dtype, matching numpy (`np.isfinite/isinf/isnan`
/// and `np.logical_not`). Ints are always finite and never inf/nan;
/// `logical_not` inverts truthiness — NaN is truthy, so `logical_not(NaN)`
/// is False and only exact `0.0`/`0` map to True.
fn pred_f64(op: UnOp, x: f64) -> bool {
    match op {
        UnOp::IsFinite => x.is_finite(),
        UnOp::IsInf => x.is_infinite(),
        UnOp::IsNan => x.is_nan(),
        UnOp::LogicalNot => !(x != 0.0),
        _ => unreachable!("pred_f64 called with non-predicate op"),
    }
}

fn pred_f32(op: UnOp, x: f32) -> bool {
    match op {
        UnOp::IsFinite => x.is_finite(),
        UnOp::IsInf => x.is_infinite(),
        UnOp::IsNan => x.is_nan(),
        UnOp::LogicalNot => !(x != 0.0),
        _ => unreachable!("pred_f32 called with non-predicate op"),
    }
}

fn pred_int(op: UnOp, x: i64) -> bool {
    match op {
        UnOp::IsFinite => true,
        UnOp::IsInf => false,
        UnOp::IsNan => false,
        UnOp::LogicalNot => x == 0,
        _ => unreachable!("pred_int called with non-predicate op"),
    }
}

fn pred_bool(op: UnOp, x: bool) -> bool {
    match op {
        UnOp::IsFinite => true,
        UnOp::IsInf => false,
        UnOp::IsNan => false,
        UnOp::LogicalNot => !x,
        _ => unreachable!("pred_bool called with non-predicate op"),
    }
}

impl NdArray {
    pub(crate) fn new(shape: Vec<usize>, dtype: Dtype, data: Data) -> NdArray {
        let size: usize = shape.iter().product();
        let ndim = shape.len();
        NdArray {
            shape,
            ndim,
            size,
            dtype,
            data,
        }
    }

    /// The flattened element buffer as `f64` (widening as needed).
    pub(crate) fn as_f64(&self) -> Vec<f64> {
        match &self.data {
            Data::F64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Data::I32(v) => v.iter().map(|&x| x as f64).collect(),
            Data::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// The flattened element buffer as `i64` (truncating floats like numpy's
    /// int casts).
    pub(crate) fn as_i64(&self) -> Vec<i64> {
        match &self.data {
            Data::F64(v) => v.iter().map(|&x| x as i64).collect(),
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
            Data::I64(v) => v.clone(),
            Data::I32(v) => v.iter().map(|&x| x as i64).collect(),
            Data::Bool(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }

    /// The flattened element buffer as `bool` (nonzero = true, like numpy).
    pub(crate) fn as_bool(&self) -> Vec<bool> {
        match &self.data {
            Data::F64(v) => v.iter().map(|&x| x != 0.0).collect(),
            Data::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
            Data::I64(v) => v.iter().map(|&x| x != 0).collect(),
            Data::I32(v) => v.iter().map(|&x| x != 0).collect(),
            Data::Bool(v) => v.clone(),
        }
    }

    /// True for a 0-d array (the result of `a[i, j]` on a 2-D array).
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// The single element as f64 (0-d arrays and indexing results).
    pub(crate) fn to_f64(&self) -> f64 {
        match &self.data {
            Data::F64(v) => v[0],
            Data::F32(v) => v[0] as f64,
            Data::I64(v) => v[0] as f64,
            Data::I32(v) => v[0] as f64,
            Data::Bool(v) => v[0] as i64 as f64,
        }
    }

    pub(crate) fn to_i64(&self) -> i64 {
        match &self.data {
            Data::F64(v) => v[0] as i64,
            Data::F32(v) => v[0] as i64,
            Data::I64(v) => v[0],
            Data::I32(v) => v[0] as i64,
            Data::Bool(v) => v[0] as i64,
        }
    }

    pub(crate) fn to_bool(&self) -> bool {
        match &self.data {
            Data::F64(v) => v[0] != 0.0,
            Data::F32(v) => v[0] != 0.0,
            Data::I64(v) => v[0] != 0,
            Data::I32(v) => v[0] != 0,
            Data::Bool(v) => v[0],
        }
    }

    /// Zero-filled array of the given shape/dtype.
    pub(crate) fn zeros(shape: Vec<usize>, dtype: Dtype) -> NdArray {
        let size = shape.iter().product();
        let data = match dtype {
            Dtype::Float64 => Data::F64(vec![0.0; size]),
            Dtype::Float32 => Data::F32(vec![0.0; size]),
            Dtype::Int64 => Data::I64(vec![0; size]),
            Dtype::Int32 => Data::I32(vec![0; size]),
            Dtype::Bool => Data::Bool(vec![false; size]),
        };
        NdArray::new(shape, dtype, data)
    }

    /// Elementwise binary op over two **same-shape** arrays (already
    /// broadcast to a common shape by the caller).
    pub(crate) fn binary_same_shape(op: BinOp, a: &NdArray, b: &NdArray) -> NdArray {
        // numpy's true_divide ALWAYS returns float64 when any operand is an
        // integer (and float64 for bool+bool); it never performs integer
        // division. Handle that before the general dtype dispatch.
        if matches!(op, BinOp::Div) {
            let any_int = matches!(a.dtype, Dtype::Int32 | Dtype::Int64)
                || matches!(b.dtype, Dtype::Int32 | Dtype::Int64);
            let both_bool = matches!((a.dtype, b.dtype), (Dtype::Bool, Dtype::Bool));
            if any_int || both_bool {
                let a = a.astype(Dtype::Float64);
                let b = b.astype(Dtype::Float64);
                let out = engine::binary_f64(op, a.f64(), b.f64());
                return NdArray::new(a.shape.clone(), Dtype::Float64, Data::F64(out));
            }
        }
        let out_dtype = if matches!(
            op,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        ) {
            Dtype::Bool
        } else {
            a.dtype.promote(b.dtype)
        };
        // Comparisons produce BOOL arrays regardless of the operand dtype;
        // the typed engine buffers can't hold them, so evaluate directly.
        if matches!(
            op,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        ) {
            let bools: Vec<bool> = match (a.dtype, b.dtype) {
                (Dtype::Float64, Dtype::Float64) => cmp_elems(op, a.f64(), b.f64()),
                (Dtype::Float32, Dtype::Float32) => cmp_elems(op, a.f32(), b.f32()),
                (Dtype::Int64, Dtype::Int64) => cmp_elems(op, a.i64(), b.i64()),
                (Dtype::Int32, Dtype::Int32) => cmp_elems(op, a.i32(), b.i32()),
                (Dtype::Bool, Dtype::Bool) => cmp_elems(op, a.bool(), b.bool()),
                // Mixed-dtype: promote to the common dtype, then recurse.
                _ => {
                    let common = a.dtype.promote(b.dtype);
                    let a = a.astype(common);
                    let b = b.astype(common);
                    return NdArray::binary_same_shape(op, &a, &b);
                }
            };
            return NdArray::new(a.shape.clone(), Dtype::Bool, Data::Bool(bools));
        }
        match (a.dtype, b.dtype) {
            (Dtype::Float64, Dtype::Float64) => {
                let out = engine::binary_f64(op, a.f64(), b.f64());
                NdArray::new(a.shape.clone(), out_dtype, Data::F64(out))
            }
            (Dtype::Float32, Dtype::Float32) => {
                let out = engine::binary_f32(op, a.f32(), b.f32());
                NdArray::new(a.shape.clone(), out_dtype, Data::F32(out))
            }
            (Dtype::Int64, Dtype::Int64) => {
                let out = engine::binary_i64(op, a.i64(), b.i64());
                NdArray::new(a.shape.clone(), out_dtype, Data::I64(out))
            }
            (Dtype::Int32, Dtype::Int32) => {
                let out = engine::binary_i32(op, a.i32(), b.i32());
                NdArray::new(a.shape.clone(), out_dtype, Data::I32(out))
            }
            (Dtype::Bool, Dtype::Bool) => {
                let out = engine::binary_bool(op, a.bool(), b.bool());
                NdArray::new(a.shape.clone(), out_dtype, Data::Bool(out))
            }
            // Mixed-dtype: promote to the common dtype, then recurse.
            _ => {
                let common = a.dtype.promote(b.dtype);
                let a = a.astype(common);
                let b = b.astype(common);
                NdArray::binary_same_shape(op, &a, &b)
            }
        }
    }

    /// Elementwise unary op.
    pub(crate) fn unary(op: UnOp, a: &NdArray) -> NdArray {
        // Predicates produce BOOL arrays regardless of the input dtype; the
        // typed engine buffers can't hold bools (same shape as the binary
        // comparison ops), so evaluate them directly here. numpy semantics
        // (verified against python3): ints are always finite, never
        // inf/nan; `logical_not(x)` is truthiness-inverted, so NaN is
        // truthy → False and only exact 0.0/0 is True.
        if matches!(
            op,
            UnOp::IsFinite | UnOp::IsInf | UnOp::IsNan | UnOp::LogicalNot
        ) {
            let bools: Vec<bool> = match a.dtype {
                Dtype::Float64 => a.f64().iter().map(|&x| pred_f64(op, x)).collect(),
                Dtype::Float32 => a.f32().iter().map(|&x| pred_f32(op, x)).collect(),
                Dtype::Int64 => a.i64().iter().map(|&x| pred_int(op, x)).collect(),
                Dtype::Int32 => a.i32().iter().map(|&x| pred_int(op, x as i64)).collect(),
                Dtype::Bool => a.bool().iter().map(|&x| pred_bool(op, x)).collect(),
            };
            return NdArray::new(a.shape.clone(), Dtype::Bool, Data::Bool(bools));
        }
        let out_dtype = a.dtype;
        match a.dtype {
            Dtype::Float64 => {
                let out = engine::unary_f64(op, a.f64());
                NdArray::new(a.shape.clone(), out_dtype, Data::F64(out))
            }
            Dtype::Float32 => {
                let out = engine::unary_f32(op, a.f32());
                NdArray::new(a.shape.clone(), out_dtype, Data::F32(out))
            }
            Dtype::Int64 => {
                let out = engine::unary_i64(op, a.i64());
                NdArray::new(a.shape.clone(), out_dtype, Data::I64(out))
            }
            Dtype::Int32 => {
                let out = engine::unary_i32(op, a.i32());
                NdArray::new(a.shape.clone(), out_dtype, Data::I32(out))
            }
            Dtype::Bool => {
                let out = engine::unary_bool(op, a.bool());
                NdArray::new(a.shape.clone(), out_dtype, Data::Bool(out))
            }
        }
    }

    // -- typed data access -------------------------------------------------

    pub(crate) fn f64(&self) -> &[f64] {
        match &self.data {
            Data::F64(v) => v,
            _ => panic!("internal: expected f64 array"),
        }
    }
    pub(crate) fn f32(&self) -> &[f32] {
        match &self.data {
            Data::F32(v) => v,
            _ => panic!("internal: expected f32 array"),
        }
    }
    pub(crate) fn i64(&self) -> &[i64] {
        match &self.data {
            Data::I64(v) => v,
            _ => panic!("internal: expected i64 array"),
        }
    }
    pub(crate) fn i32(&self) -> &[i32] {
        match &self.data {
            Data::I32(v) => v,
            _ => panic!("internal: expected i32 array"),
        }
    }
    pub(crate) fn bool(&self) -> &[bool] {
        match &self.data {
            Data::Bool(v) => v,
            _ => panic!("internal: expected bool array"),
        }
    }

    // -- numpy API methods -------------------------------------------------

    /// `a.astype(dtype)` — copy with a new element type. Float→int
    /// truncates toward zero, like numpy.
    pub fn astype(&self, dtype: Dtype) -> NdArray {
        if dtype == self.dtype {
            return self.clone();
        }
        let data = match dtype {
            Dtype::Float64 => Data::F64(self.as_f64()),
            Dtype::Float32 => Data::F32(self.as_f64().iter().map(|&x| x as f32).collect()),
            Dtype::Int64 => Data::I64(self.as_i64()),
            Dtype::Int32 => Data::I32(self.as_i64().iter().map(|&x| x as i32).collect()),
            Dtype::Bool => Data::Bool(self.as_bool()),
        };
        NdArray::new(self.shape.clone(), dtype, data)
    }

    /// `a.reshape(shape)` — copy with a new shape (row-major, so the data is
    /// unchanged). One dimension may be -1 to infer it, exactly like numpy.
    /// The shape may be a tuple `(2, 3)`, a single int, or a list.
    pub fn reshape<T: IntoShape>(&self, shape: T) -> NdArray {
        let shape = shape.into_shape();
        let mut dims: Vec<usize> = Vec::with_capacity(shape.len());
        let mut inferred = None;
        for d in shape.clone() {
            if d == -1 {
                if inferred.is_some() {
                    panic!(
                        "{}",
                        PyException::new("ValueError", "can only specify one unknown dimension")
                    );
                }
                inferred = Some(dims.len());
                dims.push(0);
            } else if d < 0 {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!("negative dimensions are not allowed: {d}")
                    )
                );
            } else {
                dims.push(d as usize);
            }
        }
        if let Some(idx) = inferred {
            let known: usize = dims
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, &d)| d)
                .product();
            if known == 0 || self.size % known != 0 {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!(
                            "cannot reshape array of size {} into shape {:?}",
                            self.size, shape
                        )
                    )
                );
            }
            dims[idx] = self.size / known;
        }
        let new_size: usize = dims.iter().product();
        if new_size != self.size {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "cannot reshape array of size {} into shape {:?}",
                        self.size, shape
                    )
                )
            );
        }
        NdArray::new(dims, self.dtype, self.data.clone())
    }

    /// `a.ravel()` — the flattened 1-D view (copy).
    pub fn ravel(&self) -> NdArray {
        NdArray::new(vec![self.size], self.dtype, self.data.clone())
    }

    /// `a.transpose()` / `a.T` — reverse the axes (2-D: swap rows/cols).
    pub fn transpose(&self) -> NdArray {
        if self.ndim <= 1 {
            return self.clone();
        }
        let rev: Vec<usize> = self.shape.iter().rev().copied().collect();
        let out_size = self.size;
        let src_strides: Vec<usize> = {
            let mut s = vec![1usize; self.ndim];
            for i in (0..self.ndim - 1).rev() {
                s[i] = s[i + 1] * self.shape[i + 1];
            }
            s
        };
        let dst_strides: Vec<usize> = {
            let mut s = vec![1usize; rev.len()];
            for i in (0..rev.len() - 1).rev() {
                s[i] = s[i + 1] * rev[i + 1];
            }
            s
        };
        // coord[k] is the index along source axis k; output axis j reads
        // source axis ndim-1-j.
        let mut coord = vec![0usize; self.ndim];
        let mut perm = Vec::with_capacity(out_size);
        for dst_flat in 0..out_size {
            let mut rem = dst_flat;
            for (j, &st) in dst_strides.iter().enumerate() {
                let c = rem / st;
                rem %= st;
                coord[self.ndim - 1 - j] = c;
            }
            let mut src_flat = 0usize;
            for (k, &c) in coord.iter().enumerate() {
                src_flat += c * src_strides[k];
            }
            perm.push(src_flat);
        }
        let data = match &self.data {
            Data::F64(v) => Data::F64(perm.iter().map(|&i| v[i]).collect()),
            Data::F32(v) => Data::F32(perm.iter().map(|&i| v[i]).collect()),
            Data::I64(v) => Data::I64(perm.iter().map(|&i| v[i]).collect()),
            Data::I32(v) => Data::I32(perm.iter().map(|&i| v[i]).collect()),
            Data::Bool(v) => Data::Bool(perm.iter().map(|&i| v[i]).collect()),
        };
        NdArray::new(rev, self.dtype, data)
    }

    /// `a.copy()`.
    pub fn copy(&self) -> NdArray {
        self.clone()
    }

    /// A 0-d array holding one i64 (numpy scalar semantics).
    pub fn from_scalar_i64(v: i64) -> NdArray {
        NdArray::new(vec![], Dtype::Int64, Data::I64(vec![v]))
    }

    /// A 0-d array holding one f64 (numpy scalar semantics).
    pub fn from_scalar_f64(v: f64) -> NdArray {
        NdArray::new(vec![], Dtype::Float64, Data::F64(vec![v]))
    }

    /// A 0-d array holding one bool (numpy scalar semantics).
    pub fn from_scalar_bool(v: bool) -> NdArray {
        NdArray::new(vec![], Dtype::Bool, Data::Bool(vec![v]))
    }
}

// ===========================================================================
// Printing: numpy's str()/repr() formatting
// ===========================================================================
//
// This mirrors numpy's `arrayprint` for the dtypes the subset supports:
// `FloatingFormat`/`IntegerFormat`/`BoolFormat` decide array-global cell
// widths, and `_formatArray` walks the array wrapping at `linewidth` and
// eliding the middle past `threshold`. Divergences here are silent — the
// program prints, it just prints something else — so the unit tests at the
// bottom of this file pin every rule against real `python3` output.

/// numpy's default print options (`np.get_printoptions()`).
const PRINT_PRECISION: usize = 8;
const PRINT_LINEWIDTH: usize = 75;
const PRINT_THRESHOLD: usize = 1000;
const PRINT_EDGEITEMS: usize = 3;

/// numpy's `dragon4_positional(x, precision, unique=True, fractional=True,
/// trim='.')`: the decimal expansion rounded to at most `precision`
/// fractional digits with trailing zeros removed. Returns
/// `(integer_part_including_sign, fractional_part)` — the caller adds the
/// point, which numpy always keeps.
///
/// Rust's `{:.p$}` rounds the exact binary value to `p` places the same way
/// dragon4 does, and `unique=True` only ever *shortens* that, which the
/// trailing-zero trim reproduces.
fn positional_digits(x: f64, precision: usize, is_f32: bool) -> (String, String) {
    // `unique=True` first: Rust's `{}` is the shortest decimal that round
    // trips, and never uses exponent form. Only when that needs more than
    // `precision` fractional digits does the precision cap round it — the
    // order matters, since 0.3f32's shortest form is `0.3` while its 8-digit
    // expansion is `0.30000001`.
    let split = |s: String| -> (String, String) {
        match s.split_once('.') {
            Some((i, f)) => (i.to_string(), f.trim_end_matches('0').to_string()),
            None => (s, String::new()),
        }
    };
    let shortest = if is_f32 {
        format!("{}", x as f32)
    } else {
        format!("{}", x)
    };
    let (int_part, frac) = split(shortest);
    if frac.len() <= precision {
        return (int_part, frac);
    }
    split(if is_f32 {
        format!("{:.*}", precision, x as f32)
    } else {
        format!("{:.*}", precision, x)
    })
}

/// Split Rust's `{:e}` output into `(integer_part_including_sign,
/// fractional_part, exponent)`. Trailing zeros are kept — callers that want
/// them trimmed do so themselves.
fn split_scientific(s: &str) -> (String, String, i32) {
    let (mant, exp) = s.split_once('e').expect("{:e} always emits an exponent");
    let (int_part, frac) = match mant.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (mant.to_string(), String::new()),
    };
    (
        int_part,
        frac,
        exp.parse().expect("{:e} emits a decimal exponent"),
    )
}

/// numpy's `dragon4_scientific(x, precision, unique=True, trim='.')`:
/// `(mantissa_integer_part_including_sign, mantissa_fraction, exponent)`.
fn scientific_digits(x: f64, precision: usize, is_f32: bool) -> (String, String, i32) {
    let split = |s: String| -> (String, String, i32) {
        let (i, f, e) = split_scientific(&s);
        (i, f.trim_end_matches('0').to_string(), e)
    };
    // `unique=True` first, precision cap second — see positional_digits.
    let shortest = if is_f32 {
        format!("{:e}", x as f32)
    } else {
        format!("{:e}", x)
    };
    let (int_part, frac, exp) = split(shortest);
    if frac.len() <= precision {
        return (int_part, frac, exp);
    }
    split(if is_f32 {
        format!("{:.*e}", precision, x as f32)
    } else {
        format!("{:.*e}", precision, x)
    })
}

/// Exponent digits numpy would print for `exp` — never fewer than two
/// (`1e-9` prints as `1.e-09`).
fn exp_digit_count(exp: i32) -> usize {
    exp.unsigned_abs().to_string().len().max(2)
}

/// numpy's `FloatingFormat`: the array-global layout every float cell is
/// rendered against.
#[derive(Debug, Clone)]
struct FloatFormat {
    exp_format: bool,
    pad_left: usize,
    pad_right: usize,
    /// Fractional digits — the cap in positional mode, the exact count in
    /// exponential mode (where numpy switches to `trim='k'`).
    precision: usize,
    /// Exponent digits in exponential mode (numpy's `exp_size`).
    exp_size: usize,
    /// Render the `f32` value rather than its `f64` widening, so a float32
    /// array prints its own dtype's digits (`0.33333334`, not
    /// `0.3333333432674408`).
    is_f32: bool,
}

impl FloatFormat {
    fn new(vals: &[f64], is_f32: bool) -> FloatFormat {
        let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();

        // Exponential mode, chosen from the non-zero finite magnitudes just
        // as numpy does: a value at or past the dtype's positional cutoff,
        // anything below 1e-4, or a spread wider than 1000x. The cutoff is
        // numpy's `10**min(8, finfo(dtype).precision)` — 1e8 for float64,
        // 1e6 for float32.
        let cutoff = if is_f32 { 1e6 } else { 1e8 };
        let mut exp_format = false;
        let mags: Vec<f64> = finite
            .iter()
            .map(|v| v.abs())
            .filter(|v| *v != 0.0)
            .collect();
        if !mags.is_empty() {
            let max_val = mags.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_val = mags.iter().copied().fold(f64::INFINITY, f64::min);
            if max_val >= cutoff || min_val < 1e-4 || max_val / min_val > 1000.0 {
                exp_format = true;
            }
        }

        let mut pad_left = 0usize;
        let mut pad_right = 0usize;
        let mut precision = PRINT_PRECISION;
        let mut exp_size = 0usize;

        if finite.is_empty() {
            // Every cell is nan/inf; the widths come from the adjustment
            // below (numpy zeroes both pads here).
            precision = 0;
        } else if exp_format {
            let parts: Vec<(String, String, i32)> = finite
                .iter()
                .map(|&x| scientific_digits(x, PRINT_PRECISION, is_f32))
                .collect();
            exp_size = parts
                .iter()
                .map(|(_, _, e)| exp_digit_count(*e))
                .max()
                .unwrap_or(2);
            precision = parts.iter().map(|(_, f, _)| f.len()).max().unwrap_or(0);
            pad_left = parts.iter().map(|(i, _, _)| i.len()).max().unwrap_or(1);
            // Only used to size non-finite cells, like numpy's comment says.
            pad_right = exp_size + 2 + precision;
        } else {
            let parts: Vec<(String, String)> = finite
                .iter()
                .map(|&x| positional_digits(x, PRINT_PRECISION, is_f32))
                .collect();
            pad_left = parts.iter().map(|(i, _)| i.len()).max().unwrap_or(1);
            pad_right = parts.iter().map(|(_, f)| f.len()).max().unwrap_or(0);
        }

        // Non-finite cells are right-aligned into `pad_left + pad_right + 1`
        // and can widen the column.
        if finite.len() != vals.len() {
            let neginf = vals.iter().any(|v| v.is_infinite() && *v < 0.0);
            let offset = (pad_right + 1) as isize;
            let nan_need = 3isize - offset;
            let inf_need = 3isize + isize::from(neginf) - offset;
            let need = nan_need.max(inf_need).max(0) as usize;
            pad_left = pad_left.max(need);
        }

        FloatFormat {
            exp_format,
            pad_left,
            pad_right,
            precision,
            exp_size,
            is_f32,
        }
    }

    fn cell(&self, x: f64) -> String {
        if !x.is_finite() {
            let ret = if x.is_nan() {
                "nan"
            } else if x < 0.0 {
                "-inf"
            } else {
                "inf"
            };
            let width = self.pad_left + self.pad_right + 1;
            return format!("{ret:>width$}");
        }
        if self.exp_format {
            // The render pass is `trim='k', min_digits=precision`: dragon4
            // emits exactly `precision` fractional digits OF THE VALUE, not
            // the shortest form zero-padded. The two differ whenever the
            // shortest form is shorter than the column — 1e-5 as float32 is
            // 9.9999997e-06 at 7 digits, not 1.0000000e-05.
            let (int_part, frac, exp) = if self.is_f32 {
                split_scientific(&format!("{:.*e}", self.precision, x as f32))
            } else {
                split_scientific(&format!("{:.*e}", self.precision, x))
            };
            let sign = if exp < 0 { '-' } else { '+' };
            format!(
                "{int_part:>pad$}.{frac}e{sign}{mag:0>digits$}",
                pad = self.pad_left,
                mag = exp.unsigned_abs(),
                digits = self.exp_size,
            )
        } else {
            let (int_part, frac) = positional_digits(x, PRINT_PRECISION, self.is_f32);
            format!(
                "{int_part:>pad_l$}.{frac:<pad_r$}",
                pad_l = self.pad_left,
                pad_r = self.pad_right,
            )
        }
    }
}

/// The per-dtype cell renderer, built once per printed array.
#[derive(Debug, Clone)]
enum CellFormat {
    Float(FloatFormat),
    /// numpy's `IntegerFormat`: right-aligned to the widest rendered value.
    Int(usize),
    /// numpy's `BoolFormat`: `True` gets a leading space so it aligns with
    /// `False` — except in a 0-d array, which prints a bare scalar.
    Bool {
        pad: bool,
    },
}

impl CellFormat {
    fn cell(&self, a: &NdArray, flat: usize) -> String {
        match self {
            CellFormat::Float(f) => {
                let x = match a.dtype {
                    Dtype::Float32 => a.f32()[flat] as f64,
                    _ => a.f64()[flat],
                };
                f.cell(x)
            }
            CellFormat::Int(w) => {
                let v = match a.dtype {
                    Dtype::Int32 => a.i32()[flat] as i64,
                    _ => a.i64()[flat],
                };
                format!("{v:>w$}", w = *w)
            }
            CellFormat::Bool { pad } => match (a.bool()[flat], pad) {
                (true, true) => " True".to_string(),
                (true, false) => "True".to_string(),
                (false, _) => "False".to_string(),
            },
        }
    }
}

/// The flat indices numpy's `_leading_trailing` keeps: the first and last
/// `edge` entries along every axis. The cell widths are computed from these,
/// because they are the only elements a summarized array ever prints.
fn summary_indices(shape: &[usize], edge: usize) -> Vec<usize> {
    let mut stride: usize = shape.iter().product();
    let mut acc = vec![0usize];
    for &n in shape {
        stride /= n.max(1);
        let keep: Vec<usize> = if n > 2 * edge {
            (0..edge).chain(n - edge..n).collect()
        } else {
            (0..n).collect()
        };
        let mut next = Vec::with_capacity(acc.len() * keep.len());
        for &base in &acc {
            for &i in &keep {
                next.push(base + i * stride);
            }
        }
        acc = next;
    }
    acc
}

/// Build the cell renderer from the elements that will actually be printed
/// (the summarized subset when the array is large enough to be elided —
/// numpy sizes its columns from the same reduced data).
fn cell_format(a: &NdArray, summarize: bool) -> CellFormat {
    let idx: Option<Vec<usize>> = if summarize {
        Some(summary_indices(&a.shape, PRINT_EDGEITEMS))
    } else {
        None
    };
    match a.dtype {
        Dtype::Bool => CellFormat::Bool { pad: a.ndim > 0 },
        Dtype::Int64 | Dtype::Int32 => {
            let all = a.as_i64();
            let vals: Vec<i64> = match &idx {
                Some(ix) => ix.iter().map(|&i| all[i]).collect(),
                None => all,
            };
            let max = vals.iter().copied().max().unwrap_or(0);
            let min = vals.iter().copied().min().unwrap_or(0);
            CellFormat::Int(max.to_string().len().max(min.to_string().len()))
        }
        Dtype::Float64 | Dtype::Float32 => {
            let all = a.as_f64();
            let vals: Vec<f64> = match &idx {
                Some(ix) => ix.iter().map(|&i| all[i]).collect(),
                None => all,
            };
            CellFormat::Float(FloatFormat::new(&vals, a.dtype == Dtype::Float32))
        }
    }
}

/// numpy's `_extendLine`: append `word` to `line`, breaking first if it
/// would overflow `line_width`. A line holding only the hanging indent
/// never wraps — breaking there could not help.
fn extend_line(
    s: &mut String,
    line: &mut String,
    word: &str,
    line_width: usize,
    next_line_prefix: &str,
) {
    let needs_wrap = line.len() + word.len() > line_width && line.len() > next_line_prefix.len();
    if needs_wrap {
        s.push_str(line.trim_end());
        s.push('\n');
        line.clear();
        line.push_str(next_line_prefix);
    }
    line.push_str(word);
}

/// numpy's `_formatArray` recursion.
struct ArrayPrinter<'a> {
    a: &'a NdArray,
    fmt: CellFormat,
    separator: &'a str,
    summarize: bool,
}

impl ArrayPrinter<'_> {
    fn recurse(&self, axis: usize, offset: usize, hanging: &str, curr_width: usize) -> String {
        let axes_left = self.a.ndim - axis;
        if axes_left == 0 {
            return self.fmt.cell(self.a, offset);
        }
        // Recursing adds a `[`, so continuation lines gain a space and the
        // budget loses the closing `]`.
        let next_hanging = format!("{hanging} ");
        let next_width = curr_width.saturating_sub(1);
        let a_len = self.a.shape[axis];
        let stride: usize = self.a.shape[axis + 1..].iter().product();
        let show_summary = self.summarize && 2 * PRINT_EDGEITEMS < a_len;
        let (leading, trailing) = if show_summary {
            (PRINT_EDGEITEMS, PRINT_EDGEITEMS)
        } else {
            (0, a_len)
        };

        let mut s = String::new();
        if axes_left == 1 {
            let elem_width = curr_width.saturating_sub(self.separator.trim_end().len().max(1));
            let mut line = hanging.to_string();
            for i in 0..leading {
                let word = self.fmt.cell(self.a, offset + i);
                extend_line(&mut s, &mut line, &word, elem_width, hanging);
                line.push_str(self.separator);
            }
            if show_summary {
                extend_line(&mut s, &mut line, "...", elem_width, hanging);
                line.push_str(self.separator);
            }
            // numpy indexes the tail from the end: -trailing ..= -2, then -1.
            for i in (2..=trailing).rev() {
                let word = self.fmt.cell(self.a, offset + a_len - i);
                extend_line(&mut s, &mut line, &word, elem_width, hanging);
                line.push_str(self.separator);
            }
            let word = self.fmt.cell(self.a, offset + a_len - 1);
            extend_line(&mut s, &mut line, &word, elem_width, hanging);
            s.push_str(&line);
        } else {
            let line_sep = format!(
                "{}{}",
                self.separator.trim_end(),
                "\n".repeat(axes_left - 1)
            );
            for i in 0..leading {
                let nested = self.recurse(axis + 1, offset + i * stride, &next_hanging, next_width);
                s.push_str(hanging);
                s.push_str(&nested);
                s.push_str(&line_sep);
            }
            if show_summary {
                s.push_str(hanging);
                s.push_str("...");
                s.push_str(&line_sep);
            }
            for i in (2..=trailing).rev() {
                let nested = self.recurse(
                    axis + 1,
                    offset + (a_len - i) * stride,
                    &next_hanging,
                    next_width,
                );
                s.push_str(hanging);
                s.push_str(&nested);
                s.push_str(&line_sep);
            }
            let nested = self.recurse(
                axis + 1,
                offset + (a_len - 1) * stride,
                &next_hanging,
                next_width,
            );
            s.push_str(hanging);
            s.push_str(&nested);
        }
        format!("[{}]", &s[hanging.len()..])
    }
}

/// numpy's `array2string`. `next_line_prefix` is the indent continuation
/// lines get: one space for `str`, plus `len("array(")` more for `repr`.
fn format_array(a: &NdArray, separator: &str, next_line_prefix: &str, line_width: usize) -> String {
    if a.size == 0 {
        return "[]".to_string();
    }
    let summarize = a.size > PRINT_THRESHOLD;
    let printer = ArrayPrinter {
        a,
        fmt: cell_format(a, summarize),
        separator,
        summarize,
    };
    printer.recurse(0, 0, next_line_prefix, line_width)
}

/// The scalar text a 0-d array shows under `str` — the element's own repr,
/// NOT the array formatter (numpy: "the str of 0d arrays is a special case:
/// it should appear like a scalar, so floats are not truncated by
/// `precision`").
fn zero_d_scalar_str(a: &NdArray) -> String {
    match a.dtype {
        Dtype::Float64 => crate::py_float_repr(a.f64()[0]),
        Dtype::Float32 => crate::py_float_repr(a.f32()[0] as f64),
        Dtype::Int64 => a.i64()[0].to_string(),
        Dtype::Int32 => a.i32()[0].to_string(),
        Dtype::Bool => {
            if a.bool()[0] {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
    }
}

/// Python's tuple repr for a shape (`(0,)`, `(0, 3)`).
fn shape_tuple(shape: &[usize]) -> String {
    if shape.len() == 1 {
        return format!("({},)", shape[0]);
    }
    let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", inner.join(", "))
}

/// numpy `str(array)` — brackets, space separators, column-aligned.
impl crate::PyDisplay for NdArray {
    fn py_display(&self) -> String {
        if self.ndim == 0 {
            return zero_d_scalar_str(self);
        }
        format_array(self, " ", " ", PRINT_LINEWIDTH)
    }
}

/// numpy `repr(array)` — `array(...)` with `, ` separators, the `array(`
/// prefix length as the hanging indent, and a `dtype=` suffix for dtypes
/// the repr does not imply (numpy implies float64, int64 and bool).
impl PyRepr for NdArray {
    fn py_repr(&self) -> String {
        let prefix = "array(";
        // numpy shortens the budget by the suffix it will append:
        // `array2string(..., suffix=")")` does `linewidth -= len(suffix)`.
        let lst = format_array(self, ", ", "       ", PRINT_LINEWIDTH - 1);

        // numpy appends whatever the array text cannot imply: the shape
        // when the array is empty (other than `(0,)`) or summarized, and
        // the dtype when it is not one of the implied ones.
        let mut extras: Vec<String> = Vec::new();
        if (self.size == 0 && self.shape != [0]) || self.size > PRINT_THRESHOLD {
            extras.push(format!("shape={}", shape_tuple(&self.shape)));
        }
        let dtype_implied = matches!(self.dtype, Dtype::Float64 | Dtype::Int64 | Dtype::Bool);
        if !dtype_implied || self.size == 0 {
            extras.push(format!("dtype={}", self.dtype.name()));
        }
        if extras.is_empty() {
            return format!("{prefix}{lst})");
        }

        let head = format!("{prefix}{lst},");
        let extra_str = format!("{})", extras.join(", "));
        let last_line_len = head.len() - head.rfind('\n').map_or(0, |i| i + 1);
        let spacer = if last_line_len + extra_str.len() + 1 > PRINT_LINEWIDTH {
            format!("\n{}", " ".repeat(prefix.len()))
        } else {
            " ".to_string()
        };
        format!("{head}{spacer}{extra_str}")
    }
}

impl std::fmt::Display for NdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::py_display(self))
    }
}

// ===========================================================================
// Indexing & slicing
// ===========================================================================

pub(crate) fn checked_index(idx: i64, len: usize) -> Result<usize, PyException> {
    let len_i = len as i64;
    let i = if idx < 0 { len_i + idx } else { idx };
    if i < 0 || i >= len_i {
        return Err(PyException::new("IndexError", "index out of bounds"));
    }
    Ok(i as usize)
}

impl NdArray {
    /// Copy the sub-array at the given axis-0 coordinate (used by `a[i]`,
    /// `a[i, j]`, ...).
    pub(crate) fn subarray(&self, coord: &[usize]) -> NdArray {
        debug_assert!(coord.len() <= self.ndim);
        let mut offset = 0usize;
        let mut mult = 1usize;
        for axis in (0..coord.len()).rev() {
            offset += coord[axis] * mult;
            mult *= self.shape[axis];
        }
        offset *= self.shape[coord.len()..].iter().product::<usize>();
        let shape: Vec<usize> = self.shape[coord.len()..].to_vec();
        let size: usize = shape.iter().product();
        let data = match &self.data {
            Data::F64(v) => Data::F64(v[offset..offset + size].to_vec()),
            Data::F32(v) => Data::F32(v[offset..offset + size].to_vec()),
            Data::I64(v) => Data::I64(v[offset..offset + size].to_vec()),
            Data::I32(v) => Data::I32(v[offset..offset + size].to_vec()),
            Data::Bool(v) => Data::Bool(v[offset..offset + size].to_vec()),
        };
        NdArray::new(shape, self.dtype, data)
    }
}

/// Read `a[i]` along axis 0: a 1-D array yields a 0-d scalar, a 2-D array a
/// row, etc. (copies, not views).
impl crate::PyIndex<i64> for NdArray {
    type Output = NdArray;
    fn py_index(&self, index: i64) -> Result<NdArray, PyException> {
        if self.ndim == 0 {
            return Err(PyException::new(
                "IndexError",
                "too many indices for array: array is 0-dimensional, but 1 were indexed",
            ));
        }
        let i = checked_index(index, self.shape[0])?;
        Ok(self.subarray(&[i]))
    }
}

impl crate::PyIndex<(i64, i64)> for NdArray {
    type Output = NdArray;
    fn py_index(&self, index: (i64, i64)) -> Result<NdArray, PyException> {
        if self.ndim != 2 {
            return Err(PyException::new(
                "IndexError",
                format!(
                    "too many indices for array: array is {}-dimensional, but 2 were indexed",
                    self.ndim
                ),
            ));
        }
        let i = checked_index(index.0, self.shape[0])?;
        let j = checked_index(index.1, self.shape[1])?;
        Ok(self.subarray(&[i, j]))
    }
}

impl crate::PyIndex<(i64, i64, i64)> for NdArray {
    type Output = NdArray;
    fn py_index(&self, index: (i64, i64, i64)) -> Result<NdArray, PyException> {
        if self.ndim != 3 {
            return Err(PyException::new(
                "IndexError",
                format!(
                    "too many indices for array: array is {}-dimensional, but 3 were indexed",
                    self.ndim
                ),
            ));
        }
        let i = checked_index(index.0, self.shape[0])?;
        let j = checked_index(index.1, self.shape[1])?;
        let k = checked_index(index.2, self.shape[2])?;
        Ok(self.subarray(&[i, j, k]))
    }
}

/// Boolean mask indexing: `a[mask]` keeps the elements where the mask is
/// true, flattened to 1-D along axis 0 (numpy semantics).
impl crate::PyIndex<NdArray> for NdArray {
    type Output = NdArray;
    fn py_index(&self, mask: NdArray) -> Result<NdArray, PyException> {
        if mask.dtype != Dtype::Bool {
            return Err(PyException::new(
                "IndexError",
                "arrays used as indices must be of integer (or boolean) type",
            ));
        }
        if mask.size != self.shape[0] {
            return Err(PyException::new(
                "IndexError",
                format!(
                    "boolean index did not match indexed array along axis 0; size of axis is {} \
                     but size of corresponding boolean axis is {}",
                    self.shape[0], mask.size
                ),
            ));
        }
        let keep: Vec<usize> = mask
            .bool()
            .iter()
            .enumerate()
            .filter(|&(_, b)| *b)
            .map(|(i, _)| i)
            .collect();
        let n = keep.len();
        let shape = if self.ndim == 1 {
            vec![n]
        } else {
            let mut s = vec![n];
            s.extend_from_slice(&self.shape[1..]);
            s
        };
        let stride: usize = self.shape[1..].iter().product();
        let gather = |v: &[f64]| -> Vec<f64> {
            keep.iter()
                .flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec())
                .collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(gather(v)),
            Data::F32(v) => Data::F32(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            ),
            Data::I64(v) => Data::I64(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            ),
            Data::I32(v) => Data::I32(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            ),
            Data::Bool(v) => Data::Bool(
                gather(
                    &v.iter()
                        .map(|&x| if x { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>(),
                )
                .iter()
                .map(|&x| x != 0.0)
                .collect(),
            ),
        };
        Ok(NdArray::new(shape, self.dtype, data))
    }
}

/// Integer fancy indexing along axis 0: `a[[0, 2]]` gathers rows.
impl crate::PyIndex<Vec<i64>> for NdArray {
    type Output = NdArray;
    fn py_index(&self, indices: Vec<i64>) -> Result<NdArray, PyException> {
        let stride: usize = self.shape[1..].iter().product();
        let mut idxs = Vec::with_capacity(indices.len());
        for idx in &indices {
            idxs.push(checked_index(*idx, self.shape[0])?);
        }
        let n = idxs.len();
        let shape = if self.ndim == 1 {
            vec![n]
        } else {
            let mut s = vec![n];
            s.extend_from_slice(&self.shape[1..]);
            s
        };
        let gather = |v: &[f64]| -> Vec<f64> {
            idxs.iter()
                .flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec())
                .collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(gather(v)),
            Data::F32(v) => Data::F32(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            ),
            Data::I64(v) => Data::I64(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            ),
            Data::I32(v) => Data::I32(
                gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            ),
            Data::Bool(v) => Data::Bool(
                gather(
                    &v.iter()
                        .map(|&x| if x { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>(),
                )
                .iter()
                .map(|&x| x != 0.0)
                .collect(),
            ),
        };
        Ok(NdArray::new(shape, self.dtype, data))
    }
}

/// Python slice semantics along axis 0, with negative steps (reversal)
/// supported. Returns a copy.
impl crate::PySlice for NdArray {
    type Output = NdArray;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> NdArray {
        let len = if self.ndim == 0 { 1 } else { self.shape[0] } as i64;
        let step = step.unwrap_or(1);
        if step == 0 {
            panic!(
                "{}",
                PyException::new("ValueError", "slice step cannot be zero")
            );
        }
        // Python normalizes a negative bound against the length BEFORE
        // clamping (`a[-3:]` is `a[len-3:]`). Clamping first mapped every
        // negative bound to 0 / -1, so `a[-3:]` returned the whole array
        // and `a[:-3]` returned nothing (issue #192).
        let normalize = |v: i64| if v < 0 { v + len } else { v };
        let (start, stop) = if step > 0 {
            let s = match start {
                Some(v) => normalize(v).clamp(0, len),
                None => 0,
            };
            let e = match stop {
                Some(v) => normalize(v).clamp(0, len),
                None => len,
            };
            (s, e)
        } else {
            // The lower bound is -1, the "before the first element"
            // sentinel a reversed walk stops at — not a user index.
            let s = match start {
                Some(v) => normalize(v).clamp(-1, len - 1),
                None => len - 1,
            };
            let e = match stop {
                Some(v) => normalize(v).clamp(-1, len - 1),
                None => -1,
            };
            (s, e)
        };
        let mut indices: Vec<usize> = Vec::new();
        if step > 0 {
            let mut i = start;
            while i < stop {
                indices.push(i as usize);
                i += step;
            }
        } else {
            let mut i = start;
            while i > stop {
                indices.push(i as usize);
                i += step;
            }
        }
        let stride: usize = self.shape[1..].iter().product();
        let shape: Vec<usize> = {
            let mut s = vec![indices.len()];
            s.extend_from_slice(&self.shape[1..]);
            s
        };
        let collect = |v: &[f64]| -> Vec<f64> {
            indices
                .iter()
                .flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec())
                .collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(collect(v)),
            Data::F32(v) => Data::F32(
                collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            ),
            Data::I64(v) => Data::I64(
                collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            ),
            Data::I32(v) => Data::I32(
                collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>())
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            ),
            Data::Bool(v) => Data::Bool(
                collect(
                    &v.iter()
                        .map(|&x| if x { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>(),
                )
                .iter()
                .map(|&x| x != 0.0)
                .collect(),
            ),
        };
        NdArray::new(shape, self.dtype, data)
    }
}

/// `len(a)` — the length of axis 0 (Python raises TypeError for 0-d).
impl crate::Len for NdArray {
    fn len(&self) -> usize {
        if self.ndim == 0 {
            panic!(
                "{}",
                PyException::new("TypeError", "len() of unsized object")
            );
        }
        self.shape[0]
    }
}

/// Iteration over a 1-D array yields its elements as 0-d arrays; over an
/// N-D array it yields the sub-arrays along axis 0 (copies).
impl IntoIterator for NdArray {
    type Item = NdArray;
    type IntoIter = std::vec::IntoIter<NdArray>;
    fn into_iter(self) -> Self::IntoIter {
        if self.ndim == 0 {
            return vec![self].into_iter();
        }
        let mut items = Vec::with_capacity(self.shape[0]);
        for i in 0..self.shape[0] {
            items.push(self.py_index(i as i64).expect("in-range index"));
        }
        items.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PySlice;

    fn arange10() -> NdArray {
        NdArray::new(vec![10], Dtype::Int64, Data::I64((0..10).collect()))
    }

    /// Every expectation below is `list(np.arange(10)[...])` under
    /// python3 + numpy. Negative bounds used to clamp to 0 / -1 instead of
    /// normalizing against the length, so `a[-3:]` returned the whole
    /// array and `a[:-3]` returned nothing (issue #192).
    #[test]
    fn slice_bounds_match_numpy() {
        let a = arange10();
        let cases: &[(Option<i64>, Option<i64>, Option<i64>, &[i64])] = &[
            // Verified against python3.
            (Some(0), Some(3), None, &[0, 1, 2]), // a[0:3]
            (Some(2), None, None, &[2, 3, 4, 5, 6, 7, 8, 9]), // a[2:]
            (None, Some(4), None, &[0, 1, 2, 3]), // a[:4]
            (None, None, Some(2), &[0, 2, 4, 6, 8]), // a[::2]
            (Some(1), None, Some(2), &[1, 3, 5, 7, 9]), // a[1::2]
            (Some(-3), None, None, &[7, 8, 9]),   // a[-3:]
            (None, Some(-3), None, &[0, 1, 2, 3, 4, 5, 6]), // a[:-3]
            (Some(-5), Some(-2), None, &[5, 6, 7]), // a[-5:-2]
            (None, Some(-11), None, &[]),         // a[:-11]
            (Some(-20), None, None, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), // a[-20:]
            (Some(8), Some(2), None, &[]),        // a[8:2]
            // Negative steps.
            (None, None, Some(-1), &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), // a[::-1]
            (Some(-1), None, Some(-1), &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), // a[-1::-1]
            (None, None, Some(-2), &[9, 7, 5, 3, 1]),                // a[::-2]
            (Some(-2), Some(-8), Some(-1), &[8, 7, 6, 5, 4, 3]),     // a[-2:-8:-1]
            (Some(-1), Some(-4), Some(-1), &[9, 8, 7]),              // a[-1:-4:-1]
        ];
        for (start, stop, step, expected) in cases {
            let got = a.py_slice(*start, *stop, *step);
            assert_eq!(
                got.as_i64(),
                expected.to_vec(),
                "a[{start:?}:{stop:?}:{step:?}]"
            );
        }
    }

    // -- printing ----------------------------------------------------------
    //
    // numpy's array printing has no wiggle room: a program that prints an
    // array must produce the same bytes CPython does. These pin every rule
    // of `FloatingFormat`/`_formatArray` the subset reaches, because a
    // divergence here is silent — the program prints, it just prints
    // something else (issues #194, #195).

    /// Build the array a case describes and check `str` and `repr`.
    fn check_cases(cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)]) {
        for (name, vals, shape, dtype, want_str, want_repr) in cases {
            let data = match dtype {
                Dtype::Float64 => Data::F64(vals.to_vec()),
                Dtype::Float32 => Data::F32(vals.iter().map(|&v| v as f32).collect()),
                Dtype::Int64 => Data::I64(vals.iter().map(|&v| v as i64).collect()),
                Dtype::Int32 => Data::I32(vals.iter().map(|&v| v as i32).collect()),
                Dtype::Bool => Data::Bool(vals.iter().map(|&v| v != 0.0).collect()),
            };
            let a = NdArray::new(shape.to_vec(), *dtype, data);
            assert_eq!(
                crate::py_display(&a),
                *want_str,
                "str({name})\n  want {want_str:?}\n   got {:?}",
                crate::py_display(&a)
            );
            assert_eq!(
                a.py_repr(),
                *want_repr,
                "repr({name})\n  want {want_repr:?}\n   got {:?}",
                a.py_repr()
            );
        }
    }

    /// numpy prints array floats at precision=8, and pads every cell to a common width so columns line up.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_precision_and_padding() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "linspace5",
                &[0.0, 0.25, 0.5, 0.75, 1.0],
                &[5],
                Dtype::Float64,
                "[0.   0.25 0.5  0.75 1.  ]",
                "array([0.  , 0.25, 0.5 , 0.75, 1.  ])",
            ),
            (
                "thirds",
                &[0.3333333333333333],
                &[1],
                Dtype::Float64,
                "[0.33333333]",
                "array([0.33333333])",
            ),
            (
                "e",
                &[2.718281828459045],
                &[1],
                Dtype::Float64,
                "[2.71828183]",
                "array([2.71828183])",
            ),
            (
                "mixfrac",
                &[0.1, 0.123456789012345],
                &[2],
                Dtype::Float64,
                "[0.1        0.12345679]",
                "array([0.1       , 0.12345679])",
            ),
            (
                "widths",
                &[1.5, 22.25, 333.125],
                &[3],
                Dtype::Float64,
                "[  1.5    22.25  333.125]",
                "array([  1.5  ,  22.25 , 333.125])",
            ),
            (
                "ones",
                &[1.0, 2.0, 3.0],
                &[3],
                Dtype::Float64,
                "[1. 2. 3.]",
                "array([1., 2., 3.])",
            ),
            (
                "linspace11",
                &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                &[11],
                Dtype::Float64,
                "[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]",
                "array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])",
            ),
        ];
        check_cases(cases);
    }

    /// The sign goes INSIDE the column padding: `[-1.  2.]`, never `[-  1.   2.]`.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_sign_inside_the_padding() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "neg1",
                &[-1.0, 2.0],
                &[2],
                Dtype::Float64,
                "[-1.  2.]",
                "array([-1.,  2.])",
            ),
            (
                "neg2",
                &[-1.0, 22.0],
                &[2],
                Dtype::Float64,
                "[-1. 22.]",
                "array([-1., 22.])",
            ),
            (
                "neg3",
                &[-1.0, 2.0, -3.5],
                &[3],
                Dtype::Float64,
                "[-1.   2.  -3.5]",
                "array([-1. ,  2. , -3.5])",
            ),
            (
                "negzero",
                &[0.0, -0.0],
                &[2],
                Dtype::Float64,
                "[ 0. -0.]",
                "array([ 0., -0.])",
            ),
        ];
        check_cases(cases);
    }

    /// inf/nan print bare (no trailing point) and right-align into the float column, widening it when they need to.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_non_finite_cells() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "nan1",
                &[f64::NAN],
                &[1],
                Dtype::Float64,
                "[nan]",
                "array([nan])",
            ),
            (
                "infnan",
                &[f64::INFINITY, f64::NAN],
                &[2],
                Dtype::Float64,
                "[inf nan]",
                "array([inf, nan])",
            ),
            (
                "neginfnan",
                &[f64::NEG_INFINITY, f64::NAN],
                &[2],
                Dtype::Float64,
                "[-inf  nan]",
                "array([-inf,  nan])",
            ),
            (
                "three_nonfinite",
                &[f64::INFINITY, f64::NAN, f64::NEG_INFINITY],
                &[3],
                Dtype::Float64,
                "[ inf  nan -inf]",
                "array([ inf,  nan, -inf])",
            ),
            (
                "one_nan",
                &[1.0, f64::NAN],
                &[2],
                Dtype::Float64,
                "[ 1. nan]",
                "array([ 1., nan])",
            ),
            (
                "onefive_inf",
                &[1.5, f64::INFINITY],
                &[2],
                Dtype::Float64,
                "[1.5 inf]",
                "array([1.5, inf])",
            ),
            (
                "mixed_nonfinite",
                &[-1.0, f64::NAN, 2.25],
                &[3],
                Dtype::Float64,
                "[-1.     nan  2.25]",
                "array([-1.  ,   nan,  2.25])",
            ),
            (
                "nan_inf",
                &[f64::NAN, f64::INFINITY],
                &[2],
                Dtype::Float64,
                "[nan inf]",
                "array([nan, inf])",
            ),
        ];
        check_cases(cases);
    }

    /// numpy switches the whole array to scientific notation past the dtype's cutoff, below 1e-4, or over a 1000x spread. An array mixing an exponent value with an ordinary one used to panic outright.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_exponential_mode() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "small",
                &[1e-09, 1.0, 1000000000.0],
                &[3],
                Dtype::Float64,
                "[1.e-09 1.e+00 1.e+09]",
                "array([1.e-09, 1.e+00, 1.e+09])",
            ),
            (
                "big",
                &[1e+16, 2e+16],
                &[2],
                Dtype::Float64,
                "[1.e+16 2.e+16]",
                "array([1.e+16, 2.e+16])",
            ),
            (
                "e5",
                &[100000.0, 2.0],
                &[2],
                Dtype::Float64,
                "[1.e+05 2.e+00]",
                "array([1.e+05, 2.e+00])",
            ),
            (
                "e-5",
                &[1e-05, 1.0],
                &[2],
                Dtype::Float64,
                "[1.e-05 1.e+00]",
                "array([1.e-05, 1.e+00])",
            ),
            (
                "ratio1001",
                &[1.0, 1001.0],
                &[2],
                Dtype::Float64,
                "[1.000e+00 1.001e+03]",
                "array([1.000e+00, 1.001e+03])",
            ),
            (
                "ratio1000",
                &[1.0, 1000.0],
                &[2],
                Dtype::Float64,
                "[   1. 1000.]",
                "array([   1., 1000.])",
            ),
            (
                "just_under",
                &[99999999.0],
                &[1],
                Dtype::Float64,
                "[99999999.]",
                "array([99999999.])",
            ),
            (
                "at_1e8",
                &[100000000.0],
                &[1],
                Dtype::Float64,
                "[1.e+08]",
                "array([1.e+08])",
            ),
            (
                "zero_small",
                &[0.0, 1e-09],
                &[2],
                Dtype::Float64,
                "[0.e+00 1.e-09]",
                "array([0.e+00, 1.e-09])",
            ),
        ];
        check_cases(cases);
    }

    /// A float32 array prints float32 digits (0.1, not 0.10000000149011612) and enters exponential mode at 1e6 rather than 1e8.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_float32_uses_its_own_dtype_digits() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "f32ones",
                &[1.0, 1.0, 1.0, 1.0],
                &[4],
                Dtype::Float32,
                "[1. 1. 1. 1.]",
                "array([1., 1., 1., 1.], dtype=float32)",
            ),
            (
                "f32third",
                &[0.3333333333333333],
                &[1],
                Dtype::Float32,
                "[0.33333334]",
                "array([0.33333334], dtype=float32)",
            ),
            (
                "f32mixed",
                &[1.5, 2.25],
                &[2],
                Dtype::Float32,
                "[1.5  2.25]",
                "array([1.5 , 2.25], dtype=float32)",
            ),
            (
                "f32_1e7",
                &[10000000.0],
                &[1],
                Dtype::Float32,
                "[1.e+07]",
                "array([1.e+07], dtype=float32)",
            ),
            (
                "f32_1e5",
                &[100000.0],
                &[1],
                Dtype::Float32,
                "[100000.]",
                "array([100000.], dtype=float32)",
            ),
            (
                "f32_tenth",
                &[0.1, 0.2, 0.3],
                &[3],
                Dtype::Float32,
                "[0.1 0.2 0.3]",
                "array([0.1, 0.2, 0.3], dtype=float32)",
            ),
        ];
        check_cases(cases);
    }

    /// Rows wider than linewidth=75 wrap, with a hanging indent one space deeper per axis. repr wraps one column earlier, having a `)` to append.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_line_wrapping() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "wrap20",
                &[
                    0.0,
                    0.05263157894736842,
                    0.10526315789473684,
                    0.15789473684210525,
                    0.21052631578947367,
                    0.2631578947368421,
                    0.3157894736842105,
                    0.3684210526315789,
                    0.42105263157894735,
                    0.47368421052631576,
                    0.5263157894736842,
                    0.5789473684210527,
                    0.631578947368421,
                    0.6842105263157895,
                    0.7368421052631579,
                    0.7894736842105263,
                    0.8421052631578947,
                    0.8947368421052632,
                    0.9473684210526315,
                    1.0,
                ],
                &[20],
                Dtype::Float64,
                "[0.         0.05263158 0.10526316 0.15789474 0.21052632 0.26315789\n 0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737\n 0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684\n 0.94736842 1.        ]",
                "array([0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,\n       0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,\n       0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,\n       0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ])",
            ),
            (
                "wrap40i",
                &[
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                    28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
                ],
                &[40],
                Dtype::Int64,
                "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]",
                "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,\n       34, 35, 36, 37, 38, 39])",
            ),
            (
                "wrap2x20",
                &[
                    0.0,
                    0.02564102564102564,
                    0.05128205128205128,
                    0.07692307692307693,
                    0.10256410256410256,
                    0.1282051282051282,
                    0.15384615384615385,
                    0.1794871794871795,
                    0.20512820512820512,
                    0.23076923076923078,
                    0.2564102564102564,
                    0.28205128205128205,
                    0.3076923076923077,
                    0.3333333333333333,
                    0.358974358974359,
                    0.38461538461538464,
                    0.41025641025641024,
                    0.4358974358974359,
                    0.46153846153846156,
                    0.48717948717948717,
                    0.5128205128205128,
                    0.5384615384615384,
                    0.5641025641025641,
                    0.5897435897435898,
                    0.6153846153846154,
                    0.6410256410256411,
                    0.6666666666666666,
                    0.6923076923076923,
                    0.717948717948718,
                    0.7435897435897436,
                    0.7692307692307693,
                    0.7948717948717948,
                    0.8205128205128205,
                    0.8461538461538461,
                    0.8717948717948718,
                    0.8974358974358975,
                    0.9230769230769231,
                    0.9487179487179487,
                    0.9743589743589743,
                    1.0,
                ],
                &[2, 20],
                Dtype::Float64,
                "[[0.         0.02564103 0.05128205 0.07692308 0.1025641  0.12820513\n  0.15384615 0.17948718 0.20512821 0.23076923 0.25641026 0.28205128\n  0.30769231 0.33333333 0.35897436 0.38461538 0.41025641 0.43589744\n  0.46153846 0.48717949]\n [0.51282051 0.53846154 0.56410256 0.58974359 0.61538462 0.64102564\n  0.66666667 0.69230769 0.71794872 0.74358974 0.76923077 0.79487179\n  0.82051282 0.84615385 0.87179487 0.8974359  0.92307692 0.94871795\n  0.97435897 1.        ]]",
                "array([[0.        , 0.02564103, 0.05128205, 0.07692308, 0.1025641 ,\n        0.12820513, 0.15384615, 0.17948718, 0.20512821, 0.23076923,\n        0.25641026, 0.28205128, 0.30769231, 0.33333333, 0.35897436,\n        0.38461538, 0.41025641, 0.43589744, 0.46153846, 0.48717949],\n       [0.51282051, 0.53846154, 0.56410256, 0.58974359, 0.61538462,\n        0.64102564, 0.66666667, 0.69230769, 0.71794872, 0.74358974,\n        0.76923077, 0.79487179, 0.82051282, 0.84615385, 0.87179487,\n        0.8974359 , 0.92307692, 0.94871795, 0.97435897, 1.        ]])",
            ),
        ];
        check_cases(cases);
    }

    /// Integer cells right-align to the widest value; bool cells pad True out to False's width.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_integers_and_bools() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "ints",
                &[1.0, 22.0, 333.0],
                &[3],
                Dtype::Int64,
                "[  1  22 333]",
                "array([  1,  22, 333])",
            ),
            (
                "negints",
                &[-1.0, 2.0, -30.0],
                &[3],
                Dtype::Int64,
                "[ -1   2 -30]",
                "array([ -1,   2, -30])",
            ),
            (
                "arange10",
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &[10],
                Dtype::Int64,
                "[0 1 2 3 4 5 6 7 8 9]",
                "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",
            ),
            (
                "int2d",
                &[1.0, 200.0, 30.0, 4.0],
                &[2, 2],
                Dtype::Int64,
                "[[  1 200]\n [ 30   4]]",
                "array([[  1, 200],\n       [ 30,   4]])",
            ),
            (
                "bools",
                &[1.0, 0.0, 1.0],
                &[3],
                Dtype::Bool,
                "[ True False  True]",
                "array([ True, False,  True])",
            ),
            (
                "allfalse",
                &[0.0, 0.0, 0.0],
                &[3],
                Dtype::Bool,
                "[False False False]",
                "array([False, False, False])",
            ),
            (
                "i32",
                &[1.0, 2.0, 3.0],
                &[3],
                Dtype::Int32,
                "[1 2 3]",
                "array([1, 2, 3], dtype=int32)",
            ),
        ];
        check_cases(cases);
    }

    /// Higher axes insert (axes_left - 1) blank lines between blocks.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_nested_blocks() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "m22",
                &[1.0, 2.0, 3.0, 4.0],
                &[2, 2],
                Dtype::Float64,
                "[[1. 2.]\n [3. 4.]]",
                "array([[1., 2.],\n       [3., 4.]])",
            ),
            (
                "eye3",
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[3, 3],
                Dtype::Float64,
                "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]",
                "array([[1., 0., 0.],\n       [0., 1., 0.],\n       [0., 0., 1.]])",
            ),
            (
                "m23",
                &[
                    0.0,
                    0.09090909090909091,
                    0.18181818181818182,
                    0.2727272727272727,
                    0.36363636363636365,
                    0.45454545454545453,
                ],
                &[2, 3],
                Dtype::Float64,
                "[[0.         0.09090909 0.18181818]\n [0.27272727 0.36363636 0.45454545]]",
                "array([[0.        , 0.09090909, 0.18181818],\n       [0.27272727, 0.36363636, 0.45454545]])",
            ),
            (
                "m34",
                &[
                    1.0,
                    1.1818181818181819,
                    1.3636363636363638,
                    1.5454545454545454,
                    1.7272727272727273,
                    1.9090909090909092,
                    2.090909090909091,
                    2.2727272727272725,
                    2.4545454545454546,
                    2.6363636363636367,
                    2.8181818181818183,
                    3.0,
                ],
                &[3, 4],
                Dtype::Float64,
                "[[1.         1.18181818 1.36363636 1.54545455]\n [1.72727273 1.90909091 2.09090909 2.27272727]\n [2.45454545 2.63636364 2.81818182 3.        ]]",
                "array([[1.        , 1.18181818, 1.36363636, 1.54545455],\n       [1.72727273, 1.90909091, 2.09090909, 2.27272727],\n       [2.45454545, 2.63636364, 2.81818182, 3.        ]])",
            ),
            (
                "c223",
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                &[2, 2, 3],
                Dtype::Float64,
                "[[[ 0.  1.  2.]\n  [ 3.  4.  5.]]\n\n [[ 6.  7.  8.]\n  [ 9. 10. 11.]]]",
                "array([[[ 0.,  1.,  2.],\n        [ 3.,  4.,  5.]],\n\n       [[ 6.,  7.,  8.],\n        [ 9., 10., 11.]]])",
            ),
        ];
        check_cases(cases);
    }

    /// str of a 0-d array is the element's own str — NOT truncated to precision — while repr goes through the array formatter.
    ///
    /// Every expectation is real `python3` + numpy output.
    #[test]
    fn format_zero_d_arrays() {
        // Verified against python3.
        let cases: &[(&str, &[f64], &[usize], Dtype, &str, &str)] = &[
            (
                "zerod_f",
                &[0.3333333333333333],
                &[],
                Dtype::Float64,
                "0.3333333333333333",
                "array(0.33333333)",
            ),
            ("zerod_f2", &[1.5], &[], Dtype::Float64, "1.5", "array(1.5)"),
            ("zerod_i", &[3.0], &[], Dtype::Int64, "3", "array(3)"),
            ("zerod_b", &[1.0], &[], Dtype::Bool, "True", "array(True)"),
        ];
        check_cases(cases);
    }

    /// Above threshold=1000 elements numpy elides the middle, keeping
    /// edgeitems=3 per axis, and sizes the columns from what survives.
    /// repr then adds `shape=`, which the elided text no longer implies.
    #[test]
    fn format_summarization() {
        // Verified against python3.
        let ints = NdArray::new(vec![1001], Dtype::Int64, Data::I64((0..1001).collect()));
        assert_eq!(
            crate::py_display(&ints),
            "[   0    1    2 ...  998  999 1000]"
        );
        assert_eq!(
            ints.py_repr(),
            "array([   0,    1,    2, ...,  998,  999, 1000], shape=(1001,))"
        );

        let zeros = NdArray::new(vec![2000], Dtype::Float64, Data::F64(vec![0.0; 2000]));
        assert_eq!(crate::py_display(&zeros), "[0. 0. 0. ... 0. 0. 0.]");
        assert_eq!(
            zeros.py_repr(),
            "array([0., 0., 0., ..., 0., 0., 0.], shape=(2000,))"
        );

        // Both axes elide, and the ellipsis row sits at the block level.
        let grid = NdArray::new(
            vec![40, 40],
            Dtype::Float64,
            Data::F64((0..1600).map(|i| (i % 7) as f64).collect()),
        );
        assert_eq!(
            crate::py_display(&grid),
            "[[0. 1. 2. ... 2. 3. 4.]\n [5. 6. 0. ... 0. 1. 2.]\n [3. 4. 5. ... 5. 6. 0.]\n ...\n [3. 4. 5. ... 5. 6. 0.]\n [1. 2. 3. ... 3. 4. 5.]\n [6. 0. 1. ... 1. 2. 3.]]"
        );

        // Exactly at the threshold nothing is elided.
        let at_threshold = NdArray::new(vec![1000], Dtype::Int64, Data::I64((0..1000).collect()));
        assert!(!crate::py_display(&at_threshold).contains("..."));
    }

    /// A summarized array's cell widths come from the elements that survive
    /// summarization, not from the elided ones.
    #[test]
    fn format_summarized_widths_come_from_the_visible_edge() {
        // Verified against python3: the elided middle is 123456.0, but only
        // the zeros print, so the column stays narrow.
        let mut vals = vec![0.0; 3];
        vals.extend(std::iter::repeat_n(123_456.0, 995));
        vals.extend([0.0; 3]);
        let a = NdArray::new(vec![1001], Dtype::Float64, Data::F64(vals));
        assert_eq!(crate::py_display(&a), "[0. 0. 0. ... 0. 0. 0.]");
    }

    /// A slice of a 2-D array keeps the trailing axes.
    #[test]
    fn slice_of_2d_keeps_row_shape() {
        // np.arange(12).reshape(4, 3)[-2:] -> shape (2, 3), [[6,7,8],[9,10,11]]
        // Verified against python3.
        let m = NdArray::new(vec![4, 3], Dtype::Int64, Data::I64((0..12).collect()));
        let got = m.py_slice(Some(-2), None, None);
        assert_eq!(got.shape, vec![2, 3]);
        assert_eq!(got.as_i64(), vec![6, 7, 8, 9, 10, 11]);
    }
}
