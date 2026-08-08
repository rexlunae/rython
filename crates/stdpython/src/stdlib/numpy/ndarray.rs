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
        let n = a.size;
        match (a.dtype, b.dtype) {
            (Dtype::Float64, Dtype::Float64) => {
                let mut out = vec![0.0; n];
                engine::binary_f64(op, a.f64(), b.f64(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::F64(out))
            }
            (Dtype::Float32, Dtype::Float32) => {
                let mut out = vec![0.0f32; n];
                engine::binary_f32(op, a.f32(), b.f32(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::F32(out))
            }
            (Dtype::Int64, Dtype::Int64) => {
                let mut out = vec![0i64; n];
                engine::binary_i64(op, a.i64(), b.i64(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::I64(out))
            }
            (Dtype::Int32, Dtype::Int32) => {
                let mut out = vec![0i32; n];
                engine::binary_i32(op, a.i32(), b.i32(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::I32(out))
            }
            (Dtype::Bool, Dtype::Bool) => {
                let mut out = vec![false; n];
                engine::binary_bool(op, a.bool(), b.bool(), &mut out);
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
        let out_dtype = match op {
            UnOp::IsFinite | UnOp::IsInf | UnOp::IsNan | UnOp::LogicalNot => Dtype::Bool,
            _ => a.dtype,
        };
        let n = a.size;
        match a.dtype {
            Dtype::Float64 => {
                let mut out = vec![0.0; n];
                engine::unary_f64(op, a.f64(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::F64(out))
            }
            Dtype::Float32 => {
                let mut out = vec![0.0f32; n];
                engine::unary_f32(op, a.f32(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::F32(out))
            }
            Dtype::Int64 => {
                let mut out = vec![0i64; n];
                engine::unary_i64(op, a.i64(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::I64(out))
            }
            Dtype::Int32 => {
                let mut out = vec![0i32; n];
                engine::unary_i32(op, a.i32(), &mut out);
                NdArray::new(a.shape.clone(), out_dtype, Data::I32(out))
            }
            Dtype::Bool => {
                let mut out = vec![false; n];
                engine::unary_bool(op, a.bool(), &mut out);
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
                    PyException::new("ValueError", format!("negative dimensions are not allowed: {d}"))
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
                        format!("cannot reshape array of size {} into shape {:?}", self.size, shape)
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
                    format!("cannot reshape array of size {} into shape {:?}", self.size, shape)
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

/// numpy-style formatting for a float in a non-0d array. Differs from
/// Python's repr: trailing `.0` becomes `.`, and scientific mantissas keep a
/// decimal point (`1e+16` → `1.0e+16`).
fn np_float_cell(x: f64, pad_left: usize, pad_right: usize, exp_mode: bool) -> String {
    let s = crate::py_float_repr(x);
    let (sign, body) = match s.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", s.as_str()),
    };
    if exp_mode {
        let (mant, exp) = body.split_once('e').expect("exp mode implies e");
        let (int_part, frac_part) = match mant.split_once('.') {
            Some((i, f)) => (i, f),
            None => (mant, ""),
        };
        let mut frac = frac_part.to_string();
        while frac.len() < pad_right {
            frac.push('0');
        }
        format!("{}{}.{}{}", sign, int_part, frac, exp)
    } else if body.contains('.') {
        let (int_part, frac_part) = body.split_once('.').expect("checked");
        let frac: String = frac_part.trim_end_matches('0').to_string();
        format!("{}{:>width$}.{}", sign, int_part, frac, width = pad_left)
    } else {
        format!("{}{:>width$}.", sign, body, width = pad_left)
    }
}

fn np_int_cell(x: i64, width: usize) -> String {
    format!("{x:>width$}")
}

/// Compute the float layout parameters for a whole float array, following
/// numpy's FloatingFormat: pad_left is the widest integer part (sign
/// included), pad_right the widest fractional part, and exp_mode when any
/// finite value needs scientific notation.
fn float_layout(vals: &[f64]) -> (usize, usize, bool) {
    let mut pad_left = 1usize;
    let mut pad_right = 0usize;
    let mut exp_mode = false;
    for &x in vals {
        if !x.is_finite() {
            continue;
        }
        let s = crate::py_float_repr(x);
        if s.contains('e') {
            exp_mode = true;
            if let Some((_, f)) = s.split_once('e').and_then(|(m, _)| m.split_once('.')) {
                pad_right = pad_right.max(f.len());
            }
        } else if s.contains('.') {
            let (i, f) = s.split_once('.').expect("checked");
            let ilen = i.len() + usize::from(s.starts_with('-'));
            pad_left = pad_left.max(ilen);
            let frac: String = f.trim_end_matches('0').to_string();
            pad_right = pad_right.max(frac.len());
        } else {
            let ilen = s.len() + usize::from(s.starts_with('-'));
            pad_left = pad_left.max(ilen);
        }
    }
    (pad_left, pad_right, exp_mode)
}

/// numpy's array → string, mirroring `_formatArray`'s recursion: rows of the
/// last axis are padded cells joined by `sep`; higher axes insert
/// `(axes_left - 1)` newlines between nested blocks and grow the hanging
/// indent by one space per level. `hanging` is the indent of continuation
/// lines: 1 space for str, len("array(") + 1 for repr.
fn format_array(a: &NdArray, sep: &str, hanging: &str) -> String {
    if a.size == 0 {
        return "[]".to_string();
    }
    // 0-d arrays print as their scalar (numpy str of a 0-d array is the
    // scalar's own repr).
    if a.ndim == 0 {
        return match a.dtype {
            Dtype::Float64 => crate::py_float_repr(a.f64()[0]),
            Dtype::Float32 => crate::py_float_repr(a.f32()[0] as f64),
            Dtype::Int64 => a.i64()[0].to_string(),
            Dtype::Int32 => a.i32()[0].to_string(),
            Dtype::Bool => a.bool()[0].to_string(),
        };
    }

    // Precompute cell strings for the whole array (the width parameters are
    // array-global, exactly like numpy's format classes).
    let cells: Vec<String> = match a.dtype {
        Dtype::Bool => a
            .bool()
            .iter()
            .map(|&b| if b { " True".to_string() } else { "False".to_string() })
            .collect(),
        Dtype::Int64 | Dtype::Int32 => {
            let vals: Vec<i64> = a.as_i64();
            let lo = *vals.iter().min().unwrap_or(&0);
            let hi = *vals.iter().max().unwrap_or(&0);
            let width = lo.to_string().len().max(hi.to_string().len()).max(1);
            vals.iter().map(|&x| np_int_cell(x, width)).collect()
        }
        Dtype::Float64 | Dtype::Float32 => {
            let vals: Vec<f64> = a.as_f64();
            let (pl, pr, em) = float_layout(&vals);
            vals.iter().map(|&x| np_float_cell(x, pl, pr, em)).collect()
        }
    };

    fn block(
        a: &NdArray,
        cells: &[String],
        axis: usize,
        offset: usize,
        hanging: &str,
        sep: &str,
    ) -> String {
        let axes_left = a.ndim - axis;
        if axes_left == 1 {
            let n = a.shape[a.ndim - 1];
            let mut line = String::new();
            for j in 0..n {
                if j > 0 {
                    line.push_str(sep);
                }
                line.push_str(&cells[offset + j]);
            }
            return format!("[{line}]");
        }
        let next_hanging = format!("{hanging} ");
        let line_sep = "\n".repeat(axes_left - 1);
        let n = a.shape[axis];
        let stride: usize = a.shape[axis + 1..].iter().product();
        let mut s = String::new();
        for i in 0..n {
            let nested = block(a, cells, axis + 1, offset + i * stride, &next_hanging, sep);
            s.push_str(hanging);
            s.push_str(&nested);
            if i + 1 < n {
                s.push_str(&line_sep);
            }
        }
        format!("[{}]", &s[hanging.len()..])
    }

    block(a, &cells, 0, 0, hanging, sep)
}

/// numpy `str(array)` — brackets, space separators, column-aligned.
impl crate::PyDisplay for NdArray {
    fn py_display(&self) -> String {
        format_array(self, " ", " ")
    }
}

/// numpy `repr(array)` — `array(...)` with `, ` separators and the
/// `array(` prefix length as the hanging indent.
impl PyRepr for NdArray {
    fn py_repr(&self) -> String {
        format!("array({})", format_array(self, ", ", "       "))
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

    /// The flat offset of `(i, j)` within a 2-D array.
    pub(crate) fn flat2(&self, i: usize, j: usize) -> usize {
        debug_assert_eq!(self.ndim, 2);
        i * self.shape[1] + j
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
            keep.iter().flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec()).collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(gather(v)),
            Data::F32(v) => Data::F32(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as f32).collect()),
            Data::I64(v) => Data::I64(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i64).collect()),
            Data::I32(v) => Data::I32(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i32).collect()),
            Data::Bool(v) => Data::Bool(gather(&v.iter().map(|&x| if x {1.0} else {0.0}).collect::<Vec<_>>()).iter().map(|&x| x != 0.0).collect()),
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
            idxs.iter().flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec()).collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(gather(v)),
            Data::F32(v) => Data::F32(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as f32).collect()),
            Data::I64(v) => Data::I64(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i64).collect()),
            Data::I32(v) => Data::I32(gather(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i32).collect()),
            Data::Bool(v) => Data::Bool(gather(&v.iter().map(|&x| if x {1.0} else {0.0}).collect::<Vec<_>>()).iter().map(|&x| x != 0.0).collect()),
        };
        Ok(NdArray::new(shape, self.dtype, data))
    }
}

/// Python slice semantics along axis 0, with negative steps (reversal)
/// supported. Returns a copy.
impl crate::PySlice for NdArray {
    type Output = NdArray;
    fn py_slice(
        &self,
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> NdArray {
        let len = if self.ndim == 0 { 1 } else { self.shape[0] } as i64;
        let step = step.unwrap_or(1);
        if step == 0 {
            panic!(
                "{}",
                PyException::new("ValueError", "slice step cannot be zero")
            );
        }
        let (start, stop) = if step > 0 {
            let s = match start {
                Some(v) => v.clamp(0, len),
                None => 0,
            };
            let e = match stop {
                Some(v) => v.clamp(0, len),
                None => len,
            };
            (s, e)
        } else {
            let s = match start {
                Some(v) => v.clamp(-1, len - 1),
                None => len - 1,
            };
            let e = match stop {
                Some(v) => v.clamp(-1, len - 1),
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
            indices.iter().flat_map(|&i| v[i * stride..(i + 1) * stride].to_vec()).collect()
        };
        let data = match &self.data {
            Data::F64(v) => Data::F64(collect(v)),
            Data::F32(v) => Data::F32(collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as f32).collect()),
            Data::I64(v) => Data::I64(collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i64).collect()),
            Data::I32(v) => Data::I32(collect(&v.iter().map(|&x| x as f64).collect::<Vec<_>>()).iter().map(|&x| x as i32).collect()),
            Data::Bool(v) => Data::Bool(collect(&v.iter().map(|&x| if x {1.0} else {0.0}).collect::<Vec<_>>()).iter().map(|&x| x != 0.0).collect()),
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
