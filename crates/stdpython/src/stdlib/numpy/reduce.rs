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

use std::borrow::Cow;

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
///
/// Borrowed for a float64 array, which is the overwhelmingly common case:
/// `as_f64()` is a full `Vec` clone even when no conversion is needed, so
/// every reduction used to allocate and copy the whole array before
/// touching it — invisible in cache, dominant out of it (issue #200).
fn vals(a: &NdArray) -> Cow<'_, [f64]> {
    match &a.data {
        Data::F64(v) => Cow::Borrowed(v),
        _ => Cow::Owned(a.as_f64()),
    }
}

/// numpy's `npy_pairwise_sum` (loops_utils.h.src), replicated exactly so
/// results are bit-for-bit identical: base cases `< 8` (accumulator starts
/// at `-0.0` so all-`-0.0` sums stay `-0.0`) and `<= 128` (eight parallel
/// accumulators, combined `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, tail
/// sequential); larger inputs recurse on `n/2 - (n/2 % 8)`.
fn pairwise_sum(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in v {
            res += x;
        }
        res
    } else if n <= 128 {
        // SIMD fast paths when the platform has them: 8 SIMD lanes map
        // ONE-TO-ONE onto the scalar r0..r7 accumulators (lane k
        // accumulates v[block_start + k] in ascending block order), so the
        // combine tree — and therefore the result — is bit-for-bit
        // identical to the scalar spelling below. LLVM does not
        // auto-vectorize the FP reduction loop (measured 2.5ms/8M vs
        // numpy's 0.9ms), hence the explicit lanes.
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected above; loads stay within the slice.
            return unsafe { pairwise_block_avx2(v) };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { pairwise_block_neon(v) };
        }
        #[allow(unreachable_code)]
        pairwise_block_scalar(v)
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&v[..n2]) + pairwise_sum(&v[n2..])
    }
}

/// The scalar <=128 block: eight accumulators, one per lane, combined in
/// numpy's tree. Serves as the fallback on platforms without SIMD and as
/// the tail handler for the 8-block remainder.
fn pairwise_block_scalar(v: &[f64]) -> f64 {
    let n = v.len();
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) =
        (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    let mut i = 8;
    while i < n - (n % 8) {
        r0 += v[i];
        r1 += v[i + 1];
        r2 += v[i + 2];
        r3 += v[i + 3];
        r4 += v[i + 4];
        r5 += v[i + 5];
        r6 += v[i + 6];
        r7 += v[i + 7];
        i += 8;
    }
    let mut res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    while i < n {
        res += v[i];
        i += 1;
    }
    res
}

/// x86_64 AVX2 block: two f64x4 accumulators whose lanes map onto r0..r3
/// and r4..r7 — the same eight per-lane chains the scalar block runs, so
/// the combine tree gives bit-for-bit identical results.
#[cfg(target_arch = "x86_64")]
unsafe fn pairwise_block_avx2(v: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let blocks = v.len() / 8;
    let mut b = 0;
    while b < blocks {
        let base = b * 8;
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(v.as_ptr().add(base)));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(v.as_ptr().add(base + 4)));
        b += 1;
    }
    let mut tmp = [0.0f64; 8];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    _mm256_storeu_pd(tmp.as_mut_ptr().add(4), acc1);
    let tail = &v[blocks * 8..];
    let mut res = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    for &x in tail {
        res += x;
    }
    res
}

/// aarch64 NEON block: four f64x2 accumulators, lanes r0..r7 in order —
/// same combine tree, bitwise identical. NEON is baseline on aarch64.
#[cfg(target_arch = "aarch64")]
unsafe fn pairwise_block_neon(v: &[f64]) -> f64 {
    use std::arch::aarch64::*;
    unsafe {
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);
    let blocks = v.len() / 8;
    let mut b = 0;
    while b < blocks {
        let base = b * 8;
        acc0 = vaddq_f64(acc0, vld1q_f64(v.as_ptr().add(base)));
        acc1 = vaddq_f64(acc1, vld1q_f64(v.as_ptr().add(base + 2)));
        acc2 = vaddq_f64(acc2, vld1q_f64(v.as_ptr().add(base + 4)));
        acc3 = vaddq_f64(acc3, vld1q_f64(v.as_ptr().add(base + 6)));
        b += 1;
    }
    let mut tmp = [0.0f64; 8];
    vst1q_f64(tmp.as_mut_ptr(), acc0);
    vst1q_f64(tmp.as_mut_ptr().add(2), acc1);
    vst1q_f64(tmp.as_mut_ptr().add(4), acc2);
    vst1q_f64(tmp.as_mut_ptr().add(6), acc3);
    let tail = &v[blocks * 8..];
    let mut res = ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
    for &x in tail {
        res += x;
    }
    res
    }
}

/// `np.sum`'s reduce value: numpy's `add.reduce` seeds the accumulator with
/// the identity `0.0` and adds the pairwise sum to it (`0.0 + s` — matters
/// only for the sign of a `-0.0` result). An empty array yields the
/// identity `0.0` (numpy never calls the loop for n == 0).
fn reduce_sum(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        0.0 + pairwise_sum(v)
    }
}

/// `np.sum(a)` — full reduction, numpy semantics: NaN propagates, the
/// result is f64 (see module docs), computed with numpy's pairwise
/// summation so the value matches `python3` bit-for-bit.
pub fn sum(a: &NdArray) -> f64 {
    reduce_sum(&vals(&a))
}

/// `np.prod(a)` — numpy's multiply reduce is a plain sequential loop (no
/// pairwise), so the sequential order below is already bit-identical.
pub fn prod(a: &NdArray) -> f64 {
    let mut acc = 1.0f64;
    for &x in vals(&a).iter() {
        acc *= x;
    }
    acc
}

/// `np.mean(a)` — numpy: `sum / n` (true division; an empty array gives
/// `0.0 / 0.0 = nan`, matching numpy's warning-plus-nan).
pub fn mean(a: &NdArray) -> f64 {
    let v = vals(&a);
    reduce_sum(&v) / v.len() as f64
}

/// `np.max(a)` — NaN-propagating, like numpy.
pub fn max(a: &NdArray) -> Result<f64, PyException> {
    let v = vals(&a);
    if v.is_empty() {
        // numpy's own wording, and a CATCHABLE exception rather than a
        // panic (issue #205).
        return Err(PyException::new(
            "ValueError",
            "zero-size array to reduction operation maximum which has no identity",
        ));
    }
    let mut m = v[0];
    for &x in &v[1..] {
        m = nan_propagating_max(m, x);
    }
    Ok(m)
}

/// `np.min(a)` — NaN-propagating, like numpy.
pub fn min(a: &NdArray) -> Result<f64, PyException> {
    let v = vals(&a);
    if v.is_empty() {
        // numpy's own wording, and a CATCHABLE exception rather than a
        // panic (issue #205).
        return Err(PyException::new(
            "ValueError",
            "zero-size array to reduction operation minimum which has no identity",
        ));
    }
    let mut m = v[0];
    for &x in &v[1..] {
        m = nan_propagating_min(m, x);
    }
    Ok(m)
}

/// `np.std(a, ddof=0)` — population standard deviation (numpy default).
pub fn std(a: &NdArray, ddof: f64) -> f64 {
    var(a, ddof).sqrt()
}

/// `np.var(a, ddof=0)` — population variance (numpy default). Mirrors
/// numpy 2's `_methods._var` exactly: pairwise-summed mean, pairwise sum of
/// squared deviations, divided by `max(n - ddof, 0)` (a zero/negative
/// degree-of-freedom count yields `inf`/`nan` like numpy, with its
/// warning).
pub fn var(a: &NdArray, ddof: f64) -> f64 {
    let v = vals(&a);
    let n = v.len() as f64;
    // mean = sum / n (0.0 / 0.0 = nan for an empty array, like numpy)
    let m = reduce_sum(&v) / n;
    // sum of squared deviations from the mean, pairwise
    let squares: Vec<f64> = v.iter().map(|&x| (x - m) * (x - m)).collect();
    let s2 = reduce_sum(&squares);
    let rcount = (v.len() as f64 - ddof).max(0.0);
    s2 / rcount
}

/// `np.all(a)` — true when every element is truthy.
pub fn all(a: &NdArray) -> bool {
    a.as_bool().iter().all(|&b| b)
}

/// `np.any(a)` — true when any element is truthy.
pub fn any(a: &NdArray) -> bool {
    a.as_bool().iter().any(|&b| b)
}

/// `np.argmax(a)` — index of the first maximum; a NaN anywhere wins the
/// index of the first NaN, exactly like numpy.
pub fn argmax(a: &NdArray) -> Result<i64, PyException> {
    let v = vals(&a);
    if v.is_empty() {
        return Err(PyException::new(
            "ValueError",
            "attempt to get argmax of an empty sequence",
        ));
    }
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x.is_nan() {
            return Ok(i as i64);
        }
        if nan_propagating_max(best_v, x) == x {
            best = i;
            best_v = x;
        }
    }
    Ok(best as i64)
}

/// `np.argmin(a)` — index of the first minimum; NaN wins like numpy.
pub fn argmin(a: &NdArray) -> Result<i64, PyException> {
    let v = vals(&a);
    if v.is_empty() {
        return Err(PyException::new(
            "ValueError",
            "attempt to get argmin of an empty sequence",
        ));
    }
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x.is_nan() {
            return Ok(i as i64);
        }
        if nan_propagating_min(best_v, x) == x {
            best = i;
            best_v = x;
        }
    }
    Ok(best as i64)
}

// ---------------------------------------------------------------------------
// Method forms: a.sum(), a.mean(), ... (always f64/bool/i64 as above)
// ---------------------------------------------------------------------------

impl NdArray {
    pub fn sum(&self) -> f64 {
        sum(self)
    }
    pub fn prod(&self) -> f64 {
        prod(self)
    }
    pub fn mean(&self) -> f64 {
        mean(self)
    }
    pub fn max(&self) -> Result<f64, PyException> {
        max(self)
    }
    pub fn min(&self) -> Result<f64, PyException> {
        min(self)
    }
    pub fn std(&self) -> f64 {
        std(self, 0.0)
    }
    pub fn var(&self) -> f64 {
        var(self, 0.0)
    }
    pub fn all(&self) -> bool {
        all(self)
    }
    pub fn any(&self) -> bool {
        any(self)
    }
    pub fn argmax(&self) -> Result<i64, PyException> {
        argmax(self)
    }
    pub fn argmin(&self) -> Result<i64, PyException> {
        argmin(self)
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
