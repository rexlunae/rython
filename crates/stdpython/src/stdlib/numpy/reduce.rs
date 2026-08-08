//! Reductions: `np.sum`, `np.mean`, `np.max`, ... plus their `.sum()`
//! method forms.
//!
//! Rython is statically typed and `NdArray`'s element dtype is a runtime
//! value, so a reduction cannot return "int64 for int arrays, float64 for
//! float arrays". These all return `f64` (the safe common denominator);
//! numpy's int-typed scalar results (`np.sum(np.array([1,2]))` → `np.int64`)
//! come out as `f64` instead. This is the one deliberate numeric
//! divergence in the numpy subset. `np.all` / `np.any` return `bool` and
//! `np.argmax` / `np.argmin` return `i64` — those are single-typed in
//! numpy too.

use super::dtype::Dtype;
use super::ndarray::{Data, NdArray};
use crate::PyException;

fn nan_propagating_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn nan_propagating_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The array's elements as f64 (the reduction domain).
fn vals(a: &NdArray) -> Vec<f64> {
    a.as_f64()
}

/// `np.sum(a)` — full reduction, numpy semantics: NaN propagates, the
/// result is f64 (see module docs).
pub fn sum(a: NdArray) -> f64 {
    let mut acc = 0.0f64;
    for x in vals(&a) {
        acc += x;
    }
    acc
}

/// `np.prod(a)`.
pub fn prod(a: NdArray) -> f64 {
    let mut acc = 1.0f64;
    for x in vals(&a) {
        acc *= x;
    }
    acc
}

/// `np.mean(a)`.
pub fn mean(a: NdArray) -> f64 {
    let v = vals(&a);
    if v.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "mean of empty array")
        );
    }
    let mut acc = 0.0f64;
    for x in &v {
        acc += x;
    }
    acc / v.len() as f64
}

/// `np.max(a)` — NaN-propagating, like numpy.
pub fn max(a: NdArray) -> f64 {
    let v = vals(&a);
    if v.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "maximum of empty array")
        );
    }
    let mut m = v[0];
    for &x in &v[1..] {
        m = nan_propagating_max(m, x);
    }
    m
}

/// `np.min(a)` — NaN-propagating, like numpy.
pub fn min(a: NdArray) -> f64 {
    let v = vals(&a);
    if v.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "minimum of empty array")
        );
    }
    let mut m = v[0];
    for &x in &v[1..] {
        m = nan_propagating_min(m, x);
    }
    m
}

/// `np.std(a, ddof=0)` — population standard deviation (numpy default).
pub fn std(a: NdArray, ddof: f64) -> f64 {
    var(a, ddof).sqrt()
}

/// `np.var(a, ddof=0)` — population variance (numpy default).
pub fn var(a: NdArray, ddof: f64) -> f64 {
    let v = vals(&a);
    let n = v.len() as f64;
    if v.is_empty() || n == ddof {
        panic!(
            "{}",
            PyException::new("ValueError", "degrees of freedom must be smaller than sample size")
        );
    }
    let m = {
        let mut acc = 0.0f64;
        for x in &v {
            acc += x;
        }
        acc / n
    };
    let mut acc = 0.0f64;
    for x in &v {
        let d = x - m;
        acc += d * d;
    }
    acc / (n - ddof)
}

/// `np.all(a)` — true when every element is truthy.
pub fn all(a: NdArray) -> bool {
    a.as_bool().iter().all(|&b| b)
}

/// `np.any(a)` — true when any element is truthy.
pub fn any(a: NdArray) -> bool {
    a.as_bool().iter().any(|&b| b)
}

/// `np.argmax(a)` — index of the first maximum; a NaN anywhere wins the
/// index of the first NaN, exactly like numpy.
pub fn argmax(a: NdArray) -> i64 {
    let v = vals(&a);
    if v.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "attempt to get argmax of an empty sequence")
        );
    }
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x.is_nan() {
            return i as i64;
        }
        if nan_propagating_max(best_v, x) == x {
            best = i;
            best_v = x;
        }
    }
    best as i64
}

/// `np.argmin(a)` — index of the first minimum; NaN wins like numpy.
pub fn argmin(a: NdArray) -> i64 {
    let v = vals(&a);
    if v.is_empty() {
        panic!(
            "{}",
            PyException::new("ValueError", "attempt to get argmin of an empty sequence")
        );
    }
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x.is_nan() {
            return i as i64;
        }
        if nan_propagating_min(best_v, x) == x {
            best = i;
            best_v = x;
        }
    }
    best as i64
}

// ---------------------------------------------------------------------------
// Method forms: a.sum(), a.mean(), ... (always f64/bool/i64 as above)
// ---------------------------------------------------------------------------

impl NdArray {
    pub fn sum(&self) -> f64 {
        sum(self.clone())
    }
    pub fn prod(&self) -> f64 {
        prod(self.clone())
    }
    pub fn mean(&self) -> f64 {
        mean(self.clone())
    }
    pub fn max(&self) -> f64 {
        max(self.clone())
    }
    pub fn min(&self) -> f64 {
        min(self.clone())
    }
    pub fn std(&self) -> f64 {
        std(self.clone(), 0.0)
    }
    pub fn var(&self) -> f64 {
        var(self.clone(), 0.0)
    }
    pub fn all(&self) -> bool {
        all(self.clone())
    }
    pub fn any(&self) -> bool {
        any(self.clone())
    }
    pub fn argmax(&self) -> i64 {
        argmax(self.clone())
    }
    pub fn argmin(&self) -> i64 {
        argmin(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Truthiness (bool(a)) — numpy's ambiguous-truth ValueError for >1 element.
// ---------------------------------------------------------------------------

impl crate::PyBool for NdArray {
    fn py_bool(self) -> bool {
        match self.size {
            0 => false,
            1 => self.as_bool()[0],
            _ => panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    "the truth value of an array with more than one element is ambiguous. \
                     Use a.any() or a.all()"
                )
            ),
        }
    }
}

// Keep the Data import referenced (used by future axis reductions).
#[allow(dead_code)]
fn _dtype_of(d: &Data) -> Dtype {
    match d {
        Data::F64(_) => Dtype::Float64,
        Data::F32(_) => Dtype::Float32,
        Data::I64(_) => Dtype::Int64,
        Data::I32(_) => Dtype::Int32,
        Data::Bool(_) => Dtype::Bool,
    }
}
