//! The `numpy` module — rython's numpy subset.
//!
//! # Design notes
//!
//! - Arrays are **values** (copies), not views. `a[1:]` copies the slice;
//!   in-place mutation of a slice is therefore not possible (the rython
//!   type system has no borrow tracking). Code that needs to mutate should
//!   write back: `a[1:] = a[1:] + 1`.
//! - `NdArray`'s element dtype is a *runtime* value, so statically-typed
//!   code cannot have one function return `i64` for int arrays and `f64`
//!   for float arrays. Reductions (`np.sum`, `np.max`, ...) therefore
//!   return `f64`; `np.all`/`np.any` return `bool`; `np.argmax`/`np.argmin`
//!   return `i64`. Use `np.sum(a).item_i64()`-style extraction or
//!   Python-level loops when integer-exact accumulation matters.
//! - Comparisons (`a > 3`, `a == b`) do NOT compile: numpy returns bool
//!   *arrays* for those, which Rust's bool-typed operators cannot express.
//!   Use the ufuncs `np.greater(a, 3)`, `np.equal(a, b)`, ... for masks.
//! - Constructors and dtype-influenced functions are split by literal
//!   kind: `np.arange(5)` (int) vs `np.arange_f(0.0, 1.0, 0.1)` (float),
//!   `np.zeros(shape)` (float64, numpy's default) with an explicit
//!   `dtype=` argument where numpy allows it.

pub mod linalg;

mod dtype;
mod engine;
mod ndarray;
mod ops;
mod reduce;
mod ufunc;

pub use dtype::Dtype;
pub use engine::{Backend, active_backend, backend_summary, set_backend};
pub use ndarray::{IntoShape, NdArray, Shape};

use crate::PyException;
use crate::PyIndex;
use ndarray::Data;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// numpy constants. Lowercased (`pi`, `e`) so `np.pi` attribute access maps
/// straight onto the path item.
#[allow(non_upper_case_globals)]
pub const pi: f64 = std::f64::consts::PI;
#[allow(non_upper_case_globals)]
pub const e: f64 = std::f64::consts::E;

/// Backend names (accepted by `np.set_backend("...")` and the
/// `--numpy-backend` rythonc flag).
pub fn set_backend_by_name(name: &str) -> Result<(), String> {
    match Backend::from_str(name) {
        Some(b) => {
            set_backend(b);
            Ok(())
        }
        None => Err(format!(
            "unknown numpy backend '{name}' (expected one of: {})",
            ["auto", "scalar", "rayon", "simd", "cuda", "vulkan"].join(", ")
        )),
    }
}

/// `np.dtype("float64")` — resolve a dtype name (string form only; the
/// `np.float64` *callable* form is handled by the compiler as a cast).
pub fn dtype(s: &str) -> Dtype {
    dtype::dtype_from_str(s).unwrap_or_else(|err| panic!("{}", err))
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Build an `NdArray` from a (possibly nested) list literal.
///
/// The element dtype follows the list's static type: `Vec<i64>` → int64,
/// `Vec<f64>` → float64, `Vec<bool>` → bool (exactly like numpy's
/// `np.array([1, 2])` → int64). Nested lists must be rectangular.
pub trait IntoNdArray {
    fn into_ndarray(self) -> NdArray;
}

/// `np.array(existing_array)` — copy (rython arrays are values, so this is
/// a cheap clone).
impl IntoNdArray for NdArray {
    fn into_ndarray(self) -> NdArray {
        self
    }
}

macro_rules! flat_impl {
    ($($ty:ty => $dtype:expr, $ctor:ident),* $(,)?) => {
        $(
            impl IntoNdArray for Vec<$ty> {
                fn into_ndarray(self) -> NdArray {
                    NdArray::new(vec![self.len()], $dtype, Data::$ctor(self))
                }
            }
        )*
    };
}

flat_impl!(
    i64 => Dtype::Int64, I64,
    f64 => Dtype::Float64, F64,
    bool => Dtype::Bool, Bool,
);

macro_rules! nested_impl {
    ($($ty:ty => $dtype:expr, $conv:ident),* $(,)?) => {
        $(
            impl IntoNdArray for Vec<Vec<$ty>> {
                fn into_ndarray(self) -> NdArray {
                    let rows = self.len();
                    if rows == 0 {
                        return NdArray::new(vec![0, 0], $dtype, Data::$conv(Vec::new()));
                    }
                    let cols = self[0].len();
                    let mut flat = Vec::with_capacity(rows * cols);
                    for row in self {
                        if row.len() != cols {
                            panic!(
                                "{}",
                                PyException::new(
                                    "ValueError",
                                    "setting an array element with a sequence. \
                                     The requested array has an inhomogeneous shape"
                                )
                            );
                        }
                        flat.extend(row);
                    }
                    NdArray::new(vec![rows, cols], $dtype, Data::$conv(flat))
                }
            }
        )*
    };
}

nested_impl!(
    i64 => Dtype::Int64, I64,
    f64 => Dtype::Float64, F64,
    bool => Dtype::Bool, Bool,
);

macro_rules! nested3_impl {
    ($($ty:ty => $dtype:expr, $conv:ident),* $(,)?) => {
        $(
            impl IntoNdArray for Vec<Vec<Vec<$ty>>> {
                fn into_ndarray(self) -> NdArray {
                    let d0 = self.len();
                    if d0 == 0 {
                        return NdArray::new(vec![0, 0, 0], $dtype, Data::$conv(Vec::new()));
                    }
                    let d1 = self[0].len();
                    let d2 = if d1 == 0 { 0 } else { self[0][0].len() };
                    let mut flat = Vec::with_capacity(d0 * d1 * d2);
                    for m in self {
                        if m.len() != d1 {
                            panic!(
                                "{}",
                                PyException::new(
                                    "ValueError",
                                    "setting an array element with a sequence. \
                                     The requested array has an inhomogeneous shape"
                                )
                            );
                        }
                        for row in m {
                            if row.len() != d2 {
                                panic!(
                                    "{}",
                                    PyException::new(
                                        "ValueError",
                                        "setting an array element with a sequence. \
                                         The requested array has an inhomogeneous shape"
                                    )
                                );
                            }
                            flat.extend(row);
                        }
                    }
                    NdArray::new(vec![d0, d1, d2], $dtype, Data::$conv(flat))
                }
            }
        )*
    };
}

nested3_impl!(
    i64 => Dtype::Int64, I64,
    f64 => Dtype::Float64, F64,
    bool => Dtype::Bool, Bool,
);

/// `np.array([...])` — the dtype follows the list's element type.
pub fn array<T: IntoNdArray>(x: T) -> NdArray {
    x.into_ndarray()
}

/// `np.asarray(x)` — same as np.array for rython (no views to avoid).
pub fn asarray<T: IntoNdArray>(x: T) -> NdArray {
    x.into_ndarray()
}

/// `np.zeros(shape)` — float64 by default, like numpy.
pub fn zeros<S: IntoShape>(shape: S, dtype: Dtype) -> NdArray {
    let s = checked_shape(shape.into_shape());
    NdArray::zeros(s, dtype)
}

/// `np.ones(shape)`.
pub fn ones<S: IntoShape>(shape: S, dtype: Dtype) -> NdArray {
    let s = checked_shape(shape.into_shape());
    let mut a = NdArray::zeros(s, dtype);
    let n = a.size;
    match &mut a.data {
        Data::F64(v) => v.fill(1.0),
        Data::F32(v) => v.fill(1.0),
        Data::I64(v) => v.fill(1),
        Data::I32(v) => v.fill(1),
        Data::Bool(v) => v.fill(true),
    }
    let _ = n;
    a
}

/// `np.full(shape, fill)` — float64 fill (see `full_i` for integer fills).
pub fn full<S: IntoShape>(shape: S, fill: f64) -> NdArray {
    let s = checked_shape(shape.into_shape());
    let mut a = NdArray::zeros(s, Dtype::Float64);
    if let Data::F64(v) = &mut a.data {
        v.fill(fill);
    }
    a
}

/// `np.full(shape, fill)` with an integer fill value.
pub fn full_i<S: IntoShape>(shape: S, fill: i64) -> NdArray {
    let s = checked_shape(shape.into_shape());
    let mut a = NdArray::zeros(s, Dtype::Int64);
    if let Data::I64(v) = &mut a.data {
        v.fill(fill);
    }
    a
}

/// `np.full(shape, fill)` with a boolean fill value.
pub fn full_bool<S: IntoShape>(shape: S, fill: bool) -> NdArray {
    let s = checked_shape(shape.into_shape());
    let mut a = NdArray::zeros(s, Dtype::Bool);
    if let Data::Bool(v) = &mut a.data {
        v.fill(fill);
    }
    a
}

/// `np.empty(shape)` — rython returns zeroed memory (numpy's garbage is
/// never observable in a value-typed runtime; zeros are the safe default).
pub fn empty<S: IntoShape>(shape: S, dtype: Dtype) -> NdArray {
    zeros(shape, dtype)
}

/// `np.arange(stop)` — int64 `[0, stop)`.
pub fn arange(stop: i64) -> NdArray {
    arange3(0, stop, 1)
}

/// `np.arange(start, stop, step)` — int64. Mirrors numpy exactly:
/// `length = (stop-start)/step` with truncating division, plus one when the
/// remainder is nonzero; a negative length is empty; `step == 0` raises
/// `ZeroDivisionError: division by zero` (numpy's behavior).
pub fn arange3(start: i64, stop: i64, step: i64) -> NdArray {
    if step == 0 {
        panic!(
            "{}",
            PyException::new("ZeroDivisionError", "division by zero")
        );
    }
    let d = start.wrapping_sub(stop).wrapping_neg(); // stop - start, C-wrap
    let q = d / step;
    let r = d % step;
    let mut n = if r != 0 { q.saturating_add(1) } else { q };
    if n < 0 {
        n = 0;
    }
    let n = n as usize;
    // Zero-free fill (see arange_f3): values accumulate with wrapping adds
    // (C semantics, matching numpy's C loop); set_len is sound because
    // every slot 0..k is written before it runs.
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut val = start;
    let mut k = 0usize;
    {
        let spare = out.spare_capacity_mut();
        let lim = spare.len();
        while k < lim {
            spare[k].write(val);
            val = val.wrapping_add(step);
            k += 1;
        }
    }
    while k < n {
        out.resize(k + 4096, 0);
        while k < out.len() {
            out[k] = val;
            val = val.wrapping_add(step);
            k += 1;
        }
    }
    // SAFETY: slots 0..k were each written above before this line; nothing
    // reads the buffer before set_len.
    unsafe { out.set_len(k) };
    NdArray::new(vec![out.len()], Dtype::Int64, Data::I64(out))
}

/// `np.arange(stop)` with a float stop — float64.
pub fn arange_f(stop: f64) -> NdArray {
    arange_f3(0.0, stop, 1.0)
}

/// `np.arange(start, stop, step)` with float bounds — float64. Mirrors
/// numpy's C implementation (ctors.c `PyArray_ArangeObj` + `DOUBLE_fill`)
/// bit-for-bit: `length = ceil((stop-start)/step)` (with numpy's underflow
/// special case, and `ValueError: arange: cannot compute length` /
/// `Maximum allowed size exceeded` for NaN/infinite lengths), then
/// `next = start + step`, `delta = next - start`, and values
/// `fma(i, delta, start)` for `i >= 2` — numpy's fill loop is compiled
/// with FMA contraction (single-rounding `i*delta + start`; `mul_add`
/// matches it exactly). The resharpened `delta` is what makes e.g.
/// `arange(0.5, 2.5, 0.3)` end in `2.3000000000000003` rather than the
/// naive `2.3`. `step == 0` raises `ZeroDivisionError: division by zero`
/// (numpy's behavior). Note: numpy's own values here can differ between
/// its arm64 (FMA) and x86-64 (no FMA at base arch) builds; rython pins
/// the FMA variant, which is also the correctly-rounded one.
pub fn arange_f3(start: f64, stop: f64, step: f64) -> NdArray {
    if step == 0.0 {
        panic!(
            "{}",
            PyException::new("ZeroDivisionError", "division by zero")
        );
    }
    let d = stop - start;
    let val = d / step;
    let len: i64 = if val == 0.0 && d != 0.0 {
        // numpy's underflow special case: the ratio vanished, so the
        // length is 1 (positive ratio) or 0 (negative ratio) by sign bit.
        if val.is_sign_negative() { 0 } else { 1 }
    } else if val.is_nan() {
        panic!(
            "{}",
            PyException::new("ValueError", "arange: cannot compute length")
        );
    } else {
        let c = val.ceil();
        if c.is_infinite() || c > (i64::MAX as f64) || c < (i64::MIN as f64) {
            panic!(
                "{}",
                PyException::new("ValueError", "Maximum allowed size exceeded")
            );
        }
        c as i64
    };
    if len <= 0 {
        return NdArray::new(vec![0], Dtype::Float64, Data::F64(Vec::new()));
    }
    let len = len as usize;
    let next = start + step; // numpy's buffer[1]
    let delta = next - start; // numpy resharpens the step for the fill
    // Zero-free fill: numpy writes straight into its allocation, and a
    // `vec![0.0; n]`-style zeroing pass would double the write traffic on
    // multi-hundred-MB ranges (measured ~2x the runtime at 8M elements).
    // Writes go into the vec's spare capacity (`MaybeUninit::write` is
    // safe), growing in zeroed chunks only if the length is huge; the
    // single `set_len` is sound because every slot in 0..k is written
    // before it runs and nothing reads the buffer before that point.
    let mut out: Vec<f64> = Vec::with_capacity(len);
    let mut k = 0usize;
    {
        let spare = out.spare_capacity_mut();
        if len >= 1 {
            spare[0].write(start);
            k = 1;
        }
        if len >= 2 {
            spare[1].write(next);
            k = 2;
        }
        let lim = spare.len();
        while k < lim {
            spare[k].write((k as f64).mul_add(delta, start));
            k += 1;
        }
    }
    while k < len {
        out.resize(k + 4096, 0.0);
        while k < out.len() {
            out[k] = (k as f64).mul_add(delta, start);
            k += 1;
        }
    }
    // SAFETY: slots 0..k were each written above (via MaybeUninit::write or
    // plain stores into resized, initialized chunks) before this line, and
    // no read of the buffer happens before set_len.
    unsafe { out.set_len(k) };
    NdArray::new(vec![out.len()], Dtype::Float64, Data::F64(out))
}

/// `np.linspace(start, stop, num)` — float64, endpoint inclusive (numpy
/// default).
pub fn linspace(start: f64, stop: f64, num: i64) -> NdArray {
    if num < 0 {
        panic!(
            "{}",
            PyException::new("ValueError", "Number of samples must be non-negative.")
        );
    }
    let n = num as usize;
    if n == 0 {
        return NdArray::new(vec![0], Dtype::Float64, Data::F64(Vec::new()));
    }
    if n == 1 {
        return NdArray::new(vec![1], Dtype::Float64, Data::F64(vec![start]));
    }
    let step = (stop - start) / (n as f64 - 1.0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(start + step * i as f64);
    }
    // Fix the last sample exactly (numpy does the same).
    out[n - 1] = stop;
    NdArray::new(vec![n], Dtype::Float64, Data::F64(out))
}

/// `np.eye(n)` — float64 identity, like numpy's default.
pub fn eye(n: i64) -> NdArray {
    identity(n)
}

/// `np.identity(n)` — float64 identity.
pub fn identity(n: i64) -> NdArray {
    if n < 0 {
        panic!(
            "{}",
            PyException::new("ValueError", "negative dimensions are not allowed")
        );
    }
    let m = n as usize;
    let mut out = vec![0.0f64; m * m];
    for i in 0..m {
        out[i * m + i] = 1.0;
    }
    NdArray::new(vec![m, m], Dtype::Float64, Data::F64(out))
}

fn checked_shape(shape: Vec<i64>) -> Vec<usize> {
    if shape.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "zero-size array to reduction operation")
        );
    }
    for &d in &shape {
        if d < 0 {
            panic!(
                "{}",
                PyException::new("ValueError", "negative dimensions are not allowed")
            );
        }
    }
    shape.iter().map(|&d| d as usize).collect()
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// `np.reshape(a, shape)` — shape may be a tuple `(2, 3)`, an int, or a list.
pub fn reshape<S: IntoShape>(a: NdArray, shape: S) -> NdArray {
    a.reshape(shape)
}

/// `np.ravel(a)` — flattened 1-D copy.
pub fn ravel(a: NdArray) -> NdArray {
    a.ravel()
}

/// `np.transpose(a)` — reversed axes (2-D: swap rows/cols).
pub fn transpose(a: NdArray) -> NdArray {
    a.transpose()
}

/// `np.concatenate([a, b, ...], axis=0)` — join along axis 0 (v1).
pub fn concatenate(arrays: Vec<NdArray>, axis: i64) -> NdArray {
    if arrays.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "need at least one array to concatenate")
        );
    }
    let ndim = arrays[0].ndim;
    // Python's negative axis counts from the end.
    let axis = if axis < 0 { axis + ndim as i64 } else { axis };
    if axis < 0 || axis >= ndim.max(1) as i64 {
        panic!(
            "{}",
            PyException::new(
                "AxisError",
                format!("axis {axis} is out of bounds for array of dimension {ndim}")
            )
        );
    }
    let axis = axis as usize;
    for a in &arrays {
        if a.ndim != ndim {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    "all the input array dimensions must match exactly"
                )
            );
        }
        if a.shape[..axis] != arrays[0].shape[..axis]
            || a.shape[axis + 1..] != arrays[0].shape[axis + 1..]
        {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    "all the input array dimensions except for the concatenation axis must match exactly"
                )
            );
        }
    }
    let mut shape = arrays[0].shape.clone();
    shape[axis] = arrays.iter().map(|a| a.shape[axis]).sum();
    let total: usize = arrays.iter().map(|a| a.size).sum();
    let dtype = arrays
        .iter()
        .fold(arrays[0].dtype, |acc, a| acc.promote(a.dtype));
    let promoted: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            if a.dtype == dtype {
                a.clone()
            } else {
                a.astype(dtype)
            }
        })
        .collect();
    // Concatenating along `axis` interleaves the arrays once per OUTER
    // block (the dimensions before the axis); axis 0 has a single block,
    // which is the plain append this used to be limited to.
    let outer: usize = arrays[0].shape[..axis].iter().product();
    let inner: usize = arrays[0].shape[axis + 1..].iter().product();
    macro_rules! gather {
        ($get:ident) => {{
            let mut flat = Vec::with_capacity(total);
            for o in 0..outer {
                for a in &promoted {
                    let run = a.shape[axis] * inner;
                    let src = a.$get();
                    flat.extend_from_slice(&src[o * run..(o + 1) * run]);
                }
            }
            flat
        }};
    }
    let data = match dtype {
        Dtype::Float64 => {
            let mut flat = Vec::with_capacity(total);
            for o in 0..outer {
                for a in &promoted {
                    let run = a.shape[axis] * inner;
                    flat.extend_from_slice(&a.as_f64()[o * run..(o + 1) * run]);
                }
            }
            Data::F64(flat)
        }
        Dtype::Float32 => Data::F32(gather!(f32)),
        Dtype::Int64 => Data::I64(gather!(i64)),
        Dtype::Int32 => Data::I32(gather!(i32)),
        Dtype::Bool => Data::Bool(gather!(bool)),
    };
    NdArray::new(shape, dtype, data)
}

/// `np.vstack([a, b])` — 2-D stack along axis 0 (like concatenate).
pub fn vstack(arrays: Vec<NdArray>) -> NdArray {
    // numpy promotes each 1-D input to a (1, n) ROW first, so stacking two
    // vectors gives a 2xN matrix, not one long vector. Concatenating them
    // as-is silently produced the flat array instead.
    let arrays: Vec<NdArray> = arrays
        .into_iter()
        .map(|a| {
            if a.ndim == 1 {
                let n = a.shape[0] as i64;
                a.reshape(vec![1, n])
            } else {
                a
            }
        })
        .collect();
    concatenate(arrays, 0)
}

/// `np.hstack([a, b])` — 1-D: concatenate; 2-D: concatenate along axis 1.
pub fn hstack(arrays: Vec<NdArray>) -> NdArray {
    if arrays.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "need at least one array to concatenate")
        );
    }
    let ndim = arrays[0].ndim;
    if ndim == 1 {
        return concatenate(arrays, 0);
    }
    if ndim != 2 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                "hstack: rython's numpy subset supports 1-D and 2-D inputs only"
            )
        );
    }
    let rows = arrays[0].shape[0];
    let cols: usize = arrays.iter().map(|a| a.shape[1]).sum();
    let dtype = arrays
        .iter()
        .fold(arrays[0].dtype, |acc, a| acc.promote(a.dtype));
    let promoted: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            if a.dtype == dtype {
                a.clone()
            } else {
                a.astype(dtype)
            }
        })
        .collect();
    let mut out = NdArray::zeros(vec![rows, cols], dtype);
    let mut col = 0usize;
    for a in &promoted {
        if a.shape[0] != rows {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    "all the input array dimensions must match exactly"
                )
            );
        }
        for i in 0..rows {
            for j in 0..a.shape[1] {
                let src = a.py_index((i as i64, j as i64)).expect("in-range");
                out.copy_into_flat(i * cols + col + j, &src);
            }
        }
        col += a.shape[1];
    }
    out
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// `np.sort(a)` — sorted copy; 1-D only in the rython subset.
pub fn sort(a: NdArray) -> NdArray {
    if a.ndim != 1 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                "sort: rython's numpy subset supports 1-D arrays only (use np.sort(a.ravel()))"
            )
        );
    }
    let mut v = a.as_f64();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let data = match a.dtype {
        Dtype::Float64 => Data::F64(v),
        Dtype::Float32 => Data::F32(v.iter().map(|&x| x as f32).collect()),
        Dtype::Int64 => Data::I64(v.iter().map(|&x| x as i64).collect()),
        Dtype::Int32 => Data::I32(v.iter().map(|&x| x as i32).collect()),
        Dtype::Bool => Data::Bool(v.iter().map(|&x| x != 0.0).collect()),
    };
    NdArray::new(a.shape.clone(), a.dtype, data)
}

/// `np.argsort(a)` — indices that sort the array (1-D only).
pub fn argsort(a: NdArray) -> NdArray {
    if a.ndim != 1 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                "argsort: rython's numpy subset supports 1-D arrays only"
            )
        );
    }
    let v = a.as_f64();
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    NdArray::new(
        vec![idx.len()],
        Dtype::Int64,
        Data::I64(idx.iter().map(|&i| i as i64).collect()),
    )
}

// ---------------------------------------------------------------------------
// `@` operator support (also np.matmul / np.dot)
// ---------------------------------------------------------------------------

impl crate::PyMatMul<NdArray> for NdArray {
    type Output = NdArray;
    fn py_matmul(&self, rhs: &NdArray) -> NdArray {
        linalg::matmul(self.clone(), rhs.clone())
    }
}

/// `np.matmul(a, b)`.
pub fn matmul(a: NdArray, b: NdArray) -> NdArray {
    linalg::matmul(a, b)
}

/// `np.dot(a, b)`.
pub fn dot(a: NdArray, b: NdArray) -> NdArray {
    linalg::dot(a, b)
}

/// `np.vdot(a, b)` — flattened dot as a plain f64.
pub fn vdot(a: NdArray, b: NdArray) -> f64 {
    linalg::vdot(a, b)
}

/// `np.sum(a)` — f64 (see module docs on reductions).
pub fn sum(a: NdArray) -> f64 {
    reduce::sum(a)
}
pub fn prod(a: NdArray) -> f64 {
    reduce::prod(a)
}
pub fn mean(a: NdArray) -> f64 {
    reduce::mean(a)
}
pub fn max(a: NdArray) -> Result<f64, PyException> {
    reduce::max(a)
}
pub fn min(a: NdArray) -> Result<f64, PyException> {
    reduce::min(a)
}
pub fn std(a: NdArray, ddof: f64) -> f64 {
    reduce::std(a, ddof)
}
pub fn var(a: NdArray, ddof: f64) -> f64 {
    reduce::var(a, ddof)
}
pub fn all(a: NdArray) -> bool {
    reduce::all(a)
}
pub fn any(a: NdArray) -> bool {
    reduce::any(a)
}
pub fn argmax(a: NdArray) -> Result<i64, PyException> {
    reduce::argmax(a)
}
pub fn argmin(a: NdArray) -> Result<i64, PyException> {
    reduce::argmin(a)
}

// ---------------------------------------------------------------------------
// Ufunc re-exports
// ---------------------------------------------------------------------------

pub use ufunc::{
    abs, add, arccos, arcsin, arctan, bitwise_and, bitwise_or, bitwise_xor, ceil, clip, cos, cosh,
    divide, equal, exp, expm1, floor, floor_divide, greater, greater_equal, isfinite, isinf, isnan,
    less, less_equal, log, log1p, log2, log10, logical_and, logical_not, logical_or, logical_xor,
    maximum, minimum, mod_, multiply, negative, not_equal, power, reciprocal, remainder, sign, sin,
    sinh, sqrt, square, subtract, tan, tanh, where_,
};

// ---------------------------------------------------------------------------
// Internal helpers used by ops.rs / mod.rs
// ---------------------------------------------------------------------------

impl NdArray {
    /// Copy a single element (0-d array) into a flat position.
    pub(crate) fn copy_into_flat(&mut self, offset: usize, src: &NdArray) {
        self.copy_into(offset, src);
    }
}

#[cfg(test)]
mod arange_tests {
    use super::*;

    fn f64s(a: &NdArray) -> Vec<f64> {
        match &a.data {
            Data::F64(v) => v.clone(),
            _ => panic!("expected f64 data"),
        }
    }

    fn i64s(a: &NdArray) -> Vec<i64> {
        match &a.data {
            Data::I64(v) => v.clone(),
            _ => panic!("expected i64 data"),
        }
    }

    /// All literals below were captured from real `python3` + numpy 2.x
    /// runs (byte-identical, `np.arange`), not written from memory.
    #[test]
    fn arange_f_matches_numpy() {
        // Fractional steps: numpy's length is ceil((stop-start)/step) with
        // NO per-element bound check, so the value equal to `stop` IS
        // included when the ceil lands on it.
        assert_eq!(
            f64s(&arange_f3(0.1, 0.4, 0.1)),
            vec![0.1, 0.2, 0.30000000000000004, 0.4]
        );
        // The fill resharpens the step to delta = (start+step)-start and
        // computes fma(i, delta, start) (numpy's FMA-contracted fill).
        assert_eq!(
            f64s(&arange_f3(0.5, 2.5, 0.3)),
            vec![
                0.5,
                0.8,
                1.1,
                1.4000000000000001,
                1.7000000000000002,
                2.0,
                2.3000000000000003
            ]
        );
        assert_eq!(
            f64s(&arange_f3(0.3, 0.7, 0.1)),
            vec![0.3, 0.4, 0.5, 0.6000000000000001]
        );
        assert_eq!(arange_f3(0.0, 100.0, 0.1).size, 1000);
        assert_eq!(f64s(&arange_f3(0.0, 100.0, 0.1)).last(), Some(&99.9));
        assert_eq!(
            f64s(&arange_f3(0.0, 8.0, 1.0)),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(f64s(&arange_f3(1.0, 0.0, -0.5)), vec![1.0, 0.5]);
        assert_eq!(f64s(&arange_f3(0.0, -5.0, 1.0)), Vec::<f64>::new());
        assert_eq!(f64s(&arange_f3(2.0, 2.0, 1.0)), Vec::<f64>::new());
    }

    #[test]
    fn arange_i_matches_numpy() {
        assert_eq!(i64s(&arange3(0, 10, 3)), vec![0, 3, 6, 9]);
        assert_eq!(i64s(&arange3(0, 10, 4)), vec![0, 4, 8]);
        assert_eq!(i64s(&arange3(5, -5, -2)), vec![5, 3, 1, -1, -3]);
        assert_eq!(i64s(&arange3(5, -5, -3)), vec![5, 2, -1, -4]);
        assert_eq!(i64s(&arange3(-7, 7, 3)), vec![-7, -4, -1, 2, 5]);
        assert_eq!(i64s(&arange3(0, 10, -1)), Vec::<i64>::new());
        assert_eq!(i64s(&arange3(0, -10, 3)), Vec::<i64>::new());
        assert_eq!(i64s(&arange3(0, 1, 1)), vec![0]);
    }

    #[test]
    #[should_panic(expected = "ZeroDivisionError: division by zero")]
    fn arange_f_step_zero() {
        arange_f3(0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "ZeroDivisionError: division by zero")]
    fn arange_i_step_zero() {
        arange3(0, 10, 0);
    }

    #[test]
    #[should_panic(expected = "ValueError: arange: cannot compute length")]
    fn arange_f_nan_stop() {
        arange_f3(0.0, f64::NAN, 1.0);
    }

    #[test]
    #[should_panic(expected = "ValueError: Maximum allowed size exceeded")]
    fn arange_f_inf_stop() {
        arange_f3(0.0, f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic(expected = "ValueError: Maximum allowed size exceeded")]
    fn arange_f_huge_negative_length() {
        arange_f3(0.0, -1e300, 1.0);
    }
}

#[cfg(test)]
mod divmod_tests {
    use super::*;

    fn f64s(a: &NdArray) -> Vec<f64> {
        match &a.data {
            Data::F64(v) => v.clone(),
            _ => panic!("expected f64 data"),
        }
    }

    fn i64s(a: &NdArray) -> Vec<i64> {
        match &a.data {
            Data::I64(v) => v.clone(),
            _ => panic!("expected i64 data"),
        }
    }

    /// All literals below were captured from real `python3` + numpy 2.x
    /// runs (issue #168), not written from memory.
    #[test]
    fn divide_float_by_zero_is_ieee() {
        // np.divide(np.array([1.0, -1.0, 2.5, 0.0]), np.array([0.0, 0.0, -0.0, 0.0]))
        // -> array([inf, -inf, -inf, nan]) — IEEE results, NO exception.
        let a = array(vec![1.0f64, -1.0, 2.5, 0.0]);
        let b = array(vec![0.0f64, 0.0, -0.0, 0.0]);
        let r = divide(&a, &b).unwrap();
        assert!(matches!(r.dtype, Dtype::Float64));
        let v = f64s(&r);
        assert_eq!(v[0], f64::INFINITY);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert_eq!(v[2], f64::NEG_INFINITY);
        assert!(v[3].is_nan());
    }

    #[test]
    fn divide_int_arrays_promote_to_float64() {
        // np.divide(np.array([3, 1, 2, 4]), np.array([2, 1, 1, 1]))
        // -> array([1.5, 1., 2., 4.]) float64 (numpy never does int division)
        let a = array(vec![3i64, 1, 2, 4]);
        let b = array(vec![2i64, 1, 1, 1]);
        let r = divide(&a, &b).unwrap();
        assert!(matches!(r.dtype, Dtype::Float64));
        assert_eq!(f64s(&r), vec![1.5, 1.0, 2.0, 4.0]);
        // np.divide(np.array([1, -1, 2, 0]), 0) -> array([inf, -inf, inf, nan]) float64
        let a = array(vec![1i64, -1, 2, 0]);
        let r = divide(&a, 0).unwrap();
        assert!(matches!(r.dtype, Dtype::Float64));
        let v = f64s(&r);
        assert_eq!(v[0], f64::INFINITY);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert_eq!(v[2], f64::INFINITY);
        assert!(v[3].is_nan());
    }

    #[test]
    fn divide_bool_promotes_to_float64() {
        // np.divide(np.array([True, False]), np.array([True, False]))
        // -> array([1., nan]) float64
        let a = array(vec![true, false]);
        let b = array(vec![true, false]);
        let r = divide(&a, &b).unwrap();
        assert!(matches!(r.dtype, Dtype::Float64));
        let v = f64s(&r);
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
    }

    #[test]
    fn floor_divide_and_mod_by_zero() {
        // np.floor_divide(np.array([1.0, -1.0, 2.5, 0.0]), 0.0) -> [inf, -inf, inf, nan]
        let a = array(vec![1.0f64, -1.0, 2.5, 0.0]);
        let r = floor_divide(&a, 0.0f64).unwrap();
        let v = f64s(&r);
        assert_eq!(v[0], f64::INFINITY);
        assert_eq!(v[1], f64::NEG_INFINITY);
        assert_eq!(v[2], f64::INFINITY);
        assert!(v[3].is_nan());
        // np.mod(np.array([1.0, -1.0, 2.5, 0.0]), 0.0) -> [nan, nan, nan, nan]
        let r = mod_(&a, 0.0f64).unwrap();
        assert!(f64s(&r).iter().all(|x| x.is_nan()));
        // np.floor_divide(np.array([5, -5, 0, 1]), 0) -> array([0, 0, 0, 0])
        let ai = array(vec![5i64, -5, 0, 1]);
        let r = floor_divide(&ai, 0i64).unwrap();
        assert_eq!(i64s(&r), vec![0, 0, 0, 0]);
        // np.mod(np.array([5, -5, 0, 1]), 0) -> array([0, 0, 0, 0])
        let r = mod_(&ai, 0i64).unwrap();
        assert_eq!(i64s(&r), vec![0, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "ValueError: Integers to negative integer powers are not allowed.")]
    fn power_int_negative_exponent_raises() {
        // np.power(np.array([1, 2]), -1) -> ValueError (exact numpy message)
        let a = array(vec![1i64, 2]);
        power(&a, -1i64).unwrap();
    }

    #[test]
    #[should_panic(expected = "ValueError: Integers to negative integer powers are not allowed.")]
    fn power_int_negative_exponent_in_array_raises() {
        // np.power(np.array([1, 2]), np.array([2, -1])) -> ValueError
        let a = array(vec![1i64, 2]);
        let b = array(vec![2i64, -1]);
        power(&a, &b).unwrap();
    }

    #[test]
    fn power_int_positive_matches_numpy() {
        // np.power(np.array([2, -3, 0, 5]), np.array([3, 2, 0, 4]))
        // -> array([8, 9, 1, 625])
        let a = array(vec![2i64, -3, 0, 5]);
        let b = array(vec![3i64, 2, 0, 4]);
        let r = power(&a, &b).unwrap();
        assert_eq!(i64s(&r), vec![8, 9, 1, 625]);
    }

    #[test]
    fn floor_divide_and_mod_negative_divisors() {
        // np.floor_divide(np.array([7, -7, 3, -3]), np.array([2, 2, -2, -2]))
        // -> array([3, -4, -2, 1]) — FLOOR semantics (3 // -2 == -2, not
        // Euclidean -1); np.mod -> array([1, 1, -1, -1]) — divisor sign.
        let a = array(vec![7i64, -7, 3, -3]);
        let b = array(vec![2i64, 2, -2, -2]);
        assert_eq!(
            i64s(&floor_divide(&a, &b).unwrap()),
            vec![3, -4, -2, 1]
        );
        assert_eq!(i64s(&mod_(&a, &b).unwrap()), vec![1, 1, -1, -1]);
        // i64::MIN // -1 wraps to i64::MIN (numpy, with an overflow warning)
        assert_eq!(
            i64s(&floor_divide(&array(vec![i64::MIN]), -1i64).unwrap()),
            vec![i64::MIN]
        );
        assert_eq!(i64s(&mod_(&array(vec![i64::MIN]), -1i64).unwrap()), vec![0]);
    }

    #[test]
    fn floor_divide_float_rounding_matches_numpy() {
        // np.floor_divide(np.array([1.0]), 0.1) -> 9.0 (NOT floor(1.0/0.1),
        // which is floor(10.0) = 10.0 — numpy uses its fmod-based divmod);
        // np.mod(np.array([1.0]), 0.1) -> 0.09999999999999995.
        assert_eq!(
            f64s(&floor_divide(&array(vec![1.0f64]), 0.1f64).unwrap()),
            vec![9.0]
        );
        assert_eq!(
            f64s(&mod_(&array(vec![1.0f64]), 0.1f64).unwrap()),
            vec![0.09999999999999995]
        );
        // signed-zero corners: floor_divide(-1.0, -2.0).unwrap() is +0.0,
        // floor_divide(0.0, -2.0).unwrap() is -0.0, mod(0.0, -2.0) is -0.0
        let z = f64s(&floor_divide(&array(vec![-1.0f64]), -2.0f64).unwrap());
        assert!(z[0] == 0.0 && z[0].is_sign_positive());
        let z = f64s(&floor_divide(&array(vec![0.0f64]), -2.0f64).unwrap());
        assert!(z[0] == 0.0 && z[0].is_sign_negative());
        let z = f64s(&mod_(&array(vec![0.0f64]), -2.0f64).unwrap());
        assert!(z[0] == 0.0 && z[0].is_sign_negative());
    }
}

#[cfg(test)]
mod predicate_tests {
    use super::*;

    fn bools(a: &NdArray) -> Vec<bool> {
        match &a.data {
            Data::Bool(v) => v.clone(),
            _ => panic!("expected bool data"),
        }
    }

    /// Verified against python3 + numpy 2.x.
    #[test]
    fn predicates_match_numpy() {
        let f = array(vec![
            1.0f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.0,
            -0.0,
            -2.5,
        ]);
        assert_eq!(
            bools(&isfinite(f.clone())),
            vec![true, false, false, false, true, true, true]
        );
        assert_eq!(
            bools(&isinf(f.clone())),
            vec![false, true, true, false, false, false, false]
        );
        assert_eq!(
            bools(&isnan(f.clone())),
            vec![false, false, false, true, false, false, false]
        );
        // NaN is truthy, so logical_not(NaN) is False; only exact 0.0 is True
        assert_eq!(
            bools(&logical_not(f)),
            vec![false, false, false, false, true, true, false]
        );
        // ints are always finite, never inf/nan; logical_not(x) = (x == 0)
        let i = array(vec![1i64, -5, 0]);
        assert_eq!(bools(&isfinite(i.clone())), vec![true, true, true]);
        assert_eq!(bools(&isinf(i.clone())), vec![false, false, false]);
        assert_eq!(bools(&isnan(i.clone())), vec![false, false, false]);
        assert_eq!(bools(&logical_not(i)), vec![false, false, true]);
        // bool arrays: finite, never inf/nan; logical_not flips
        let b = array(vec![true, false]);
        assert_eq!(bools(&isfinite(b.clone())), vec![true, true]);
        assert_eq!(bools(&logical_not(b)), vec![false, true]);
        // float32 path
        let f32a = array(vec![1.0f64, f64::INFINITY]).astype(Dtype::Float32);
        assert_eq!(bools(&isfinite(f32a)), vec![true, false]);
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::*;

    /// Verified against python3 + numpy 2.x (numpy's pairwise summation).
    #[test]
    fn sum_mean_std_var_match_numpy() {
        // np.linspace(0, 1, 1_000_000) aggregates
        let l = linspace(0.0, 1.0, 1_000_000);
        assert_eq!(sum(l.clone()), 499999.99999999994);
        assert_eq!(mean(l.clone()), 0.49999999999999994);
        assert_eq!(std(l.clone(), 0.0), 0.28867542327009177);
        assert_eq!(var(l, 0.0), 0.08333350000016665);
        // boundary sizes exercise the pairwise blocks (<8, <=128, recursive)
        let x = |n: i64| {
            multiply(
                &subtract(&divide(&arange_f(n as f64), 7.0).unwrap(), 3.0).unwrap(),
                1.0000001,
            )
            .unwrap()
        };
        assert_eq!(sum(x(8)), -20.000002000000002);
        assert_eq!(sum(x(128)), 777.1429348571429);
        assert_eq!(sum(x(129)), 792.4286506714286);
        assert_eq!(sum(x(256)), 3894.857532342857);
        assert_eq!(sum(x(1000)), 68357.14969285714);
        // special values
        assert_eq!(sum(array(vec![-0.0f64, -0.0])), 0.0);
        assert_eq!(
            sum(array(vec![1e308f64, 1e308, -1e308, -1e308])),
            f64::INFINITY
        );
        assert!(sum(array(vec![f64::NAN, 1.0])).is_nan());
        assert_eq!(sum(array(vec![5e-324f64, 5e-324, 5e-324])), 1.5e-323);
        // ddof edges: numpy returns inf/nan (with a warning), not an error
        assert_eq!(std(array(vec![1.0f64, 2.0, 3.0, 4.0]), 4.0), f64::INFINITY);
        assert_eq!(
            std(array(vec![1.0f64, 2.0, 3.0, 4.0]), 3.0),
            2.23606797749979
        );
        assert!(mean(array(Vec::<f64>::new())).is_nan());
        assert!(var(array(Vec::<f64>::new()), 0.0).is_nan());
        assert_eq!(sum(array(Vec::<f64>::new())), 0.0);
    }
}

#[cfg(test)]
mod weak_promotion_tests {
    use super::*;

    /// NEP 50 weak Python-scalar promotion, verified against numpy 2.x:
    /// Python scalars no longer widen f32/i32 arrays.
    #[test]
    fn python_scalars_are_weak() {
        let f32a = array(vec![1.0f64, 2.0]).astype(Dtype::Float32);
        let i32a = array(vec![1i64, 2]).astype(Dtype::Int32);
        let i64a = array(vec![1i64, 2]);
        let b = array(vec![true, false]);
        // Python float + f32 array stays f32
        assert!(matches!(
            add(&f32a, 0.0f64).unwrap().dtype,
            Dtype::Float32
        ));
        assert!(matches!(
            add(0.0f64, &f32a).unwrap().dtype,
            Dtype::Float32
        ));
        assert!(matches!(
            divide(&f32a, 0.0f64).unwrap().dtype,
            Dtype::Float32
        ));
        // Python int + i32 array stays i32; + f32 stays f32
        assert!(matches!(
            add(&i32a, 1i64).unwrap().dtype,
            Dtype::Int32
        ));
        assert!(matches!(
            add(&f32a, 1i64).unwrap().dtype,
            Dtype::Float32
        ));
        // Python float + int array -> f64
        assert!(matches!(
            add(&i32a, 0.0f64).unwrap().dtype,
            Dtype::Float64
        ));
        assert!(matches!(
            add(&i64a, 0.0f64).unwrap().dtype,
            Dtype::Float64
        ));
        // bool arrays: + Python int -> int64, + Python bool stays bool
        assert!(matches!(add(&b, 1i64).unwrap().dtype, Dtype::Int64));
        assert!(matches!(add(&b, true).unwrap().dtype, Dtype::Bool));
        assert!(matches!(
            add(&f32a, true).unwrap().dtype,
            Dtype::Float32
        ));
        // int64 + Python int stays int64; comparison outputs are bool anyway
        assert!(matches!(add(&i64a, 1i64).unwrap().dtype, Dtype::Int64));
        assert!(matches!(
            equal(&f32a, 1i64).unwrap().dtype,
            Dtype::Bool
        ));
        // a large-but-fitting Python int scalar stays int32
        let r = add(&i32a, 1i64 << 30).unwrap();
        assert!(matches!(r.dtype, Dtype::Int32));
        // (int32 arithmetic overflow wraps in release, matching numpy; in
        // debug builds it panics — documented ledger §12.2, out of contract)
    }

    #[test]
    #[should_panic(
        expected = "OverflowError: Python integer 1099511627776 out of bounds for int32"
    )]
    fn int_scalar_out_of_int32_bounds_raises() {
        let i32a = array(vec![1i64, 2]).astype(Dtype::Int32);
        add(&i32a, 1i64 << 40).unwrap();
    }
}

#[cfg(test)]
mod exception_tests {
    use super::*;

    /// The operations CPython raises from now RAISE here too, with numpy's
    /// own message, instead of aborting the process with a panic
    /// (issue #205). Every expectation is real `python3` output.
    #[test]
    fn empty_reductions_raise_with_numpy_message() {
        // Verified against python3.
        let empty = || zeros(0i64, Dtype::Float64);
        let cases: Vec<(Result<f64, PyException>, &str)> = vec![
            (
                max(empty()),
                "zero-size array to reduction operation maximum which has no identity",
            ),
            (
                min(empty()),
                "zero-size array to reduction operation minimum which has no identity",
            ),
        ];
        for (r, msg) in cases {
            let err = r.expect_err("an empty reduction must raise");
            assert_eq!(err.exception_type, "ValueError");
            assert_eq!(err.message, msg);
        }
        for (r, msg) in [
            (
                argmax(empty()),
                "attempt to get argmax of an empty sequence",
            ),
            (
                argmin(empty()),
                "attempt to get argmin of an empty sequence",
            ),
        ] {
            let err = r.expect_err("an empty argmax/argmin must raise");
            assert_eq!(err.exception_type, "ValueError");
            assert_eq!(err.message, msg);
        }
    }

    #[test]
    fn broadcast_mismatch_raises_valueerror() {
        // np.add(np.zeros(2), np.zeros(3)) -> ValueError: operands could not
        // be broadcast together with shapes (2,) (3,)
        // Verified against python3.
        let err = add(&zeros(2i64, Dtype::Float64), &zeros(3i64, Dtype::Float64))
            .expect_err("a shape mismatch must raise");
        assert_eq!(err.exception_type, "ValueError");
        assert!(
            err.message.contains("could not be broadcast together"),
            "{}",
            err.message
        );
    }

    #[test]
    fn singular_matrix_raises_linalgerror() {
        // np.linalg.inv(np.array([[1., 2.], [2., 4.]])) -> LinAlgError:
        // Singular matrix. Verified against python3.
        let singular = || reshape(array(vec![1.0f64, 2.0, 2.0, 4.0]), vec![2i64, 2]);
        for r in [
            linalg::inv(singular()),
            linalg::solve(singular(), ones(2i64, Dtype::Float64)),
        ] {
            let err = r.expect_err("a singular matrix must raise");
            assert_eq!(err.exception_type, "LinAlgError");
            assert_eq!(err.message, "Singular matrix");
        }
    }

    #[test]
    fn index_errors_carry_numpys_message() {
        // np.zeros((2, 3))[5] -> IndexError: index 5 is out of bounds for
        // axis 0 with size 2; [0, 7] names axis 1. Verified against python3.
        let a = zeros(vec![2i64, 3], Dtype::Float64);
        let err = crate::PyIndex::<i64>::py_index(&a, 5).expect_err("out of bounds");
        assert_eq!(err.exception_type, "IndexError");
        assert_eq!(
            err.message,
            "index 5 is out of bounds for axis 0 with size 2"
        );
        let err = crate::PyIndex::<(i64, i64)>::py_index(&a, (0, 7)).expect_err("out of bounds");
        assert_eq!(
            err.message,
            "index 7 is out of bounds for axis 1 with size 3"
        );
        // a.shape is a TUPLE, and Python's tuple has its own wording.
        let err = crate::PyIndex::<i64>::py_index(&a.shape_tuple(), 9).expect_err("out of bounds");
        assert_eq!(err.message, "tuple index out of range");
    }
}

#[cfg(test)]
mod shape_and_stack_tests {
    use super::*;

    /// `a.shape` is a Python TUPLE, not a list (issue #197).
    #[test]
    fn shape_prints_as_a_tuple() {
        // Verified against python3: str(np.zeros(3).shape) == '(3,)'.
        assert_eq!(
            crate::py_display(&zeros(3i64, Dtype::Float64).shape_tuple()),
            "(3,)"
        );
        assert_eq!(
            crate::py_display(&zeros(vec![2i64, 3], Dtype::Float64).shape_tuple()),
            "(2, 3)"
        );
        // and still indexes and measures like the sequence it is.
        let s = zeros(vec![2i64, 3], Dtype::Float64).shape_tuple();
        assert_eq!(crate::PyIndex::<i64>::py_index(&s, 1).unwrap(), 3);
        assert_eq!(crate::PyIndex::<i64>::py_index(&s, -1).unwrap(), 3);
        assert_eq!(crate::Len::len(&s), 2);
    }

    /// `a.dtype` prints its name (issue #204).
    #[test]
    fn dtype_prints_like_numpy() {
        // Verified against python3: str(np.zeros(3).dtype) == 'float64',
        // repr(...) == "dtype('float64')".
        assert_eq!(crate::py_display(&Dtype::Float64), "float64");
        assert_eq!(crate::PyRepr::py_repr(&Dtype::Float32), "dtype('float32')");
    }

    /// numpy promotes 1-D inputs to rows before stacking them.
    #[test]
    fn vstack_promotes_vectors_to_rows() {
        // np.vstack([np.array([1., 2.]), np.array([3., 4.])])
        // -> array([[1., 2.], [3., 4.]]). Verified against python3.
        let r = vstack(vec![array(vec![1.0f64, 2.0]), array(vec![3.0f64, 4.0])]);
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(crate::py_display(&r), "[[1. 2.]\n [3. 4.]]");
    }

    /// Concatenation along a non-zero axis interleaves per outer block.
    #[test]
    fn concatenate_supports_every_axis() {
        // np.concatenate([np.arange(4).reshape(2,2), np.arange(4,8).reshape(2,2)], axis=1)
        // -> array([[0, 1, 4, 5], [2, 3, 6, 7]]). Verified against python3.
        let p = || reshape(arange(4), vec![2i64, 2]);
        let q = || reshape(arange3(4, 8, 1), vec![2i64, 2]);
        let r = concatenate(vec![p(), q()], 1);
        assert_eq!(r.shape, vec![2, 4]);
        assert_eq!(crate::py_display(&r), "[[0 1 4 5]\n [2 3 6 7]]");
        // axis 0 still appends whole blocks.
        let r = concatenate(vec![p(), q()], 0);
        assert_eq!(crate::py_display(&r), "[[0 1]\n [2 3]\n [4 5]\n [6 7]]");
        // A negative axis counts from the end, like Python.
        let r = concatenate(vec![p(), q()], -1);
        assert_eq!(crate::py_display(&r), "[[0 1 4 5]\n [2 3 6 7]]");
    }
}
