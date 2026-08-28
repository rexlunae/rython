//! `np.linalg` subset: `dot`, `matmul`, `inv`, `det`, `solve`, `vdot`.
//!
//! All linalg works in f64 (numpy promotes to float64 for these too, so
//! int inputs are fine). `dot`/`matmul` return arrays — a vector·vector
//! dot is a 0-d array; use `vdot` for a plain f64 scalar.

use super::dtype::Dtype;
use super::ndarray::{Data, NdArray};
use crate::PyException;

/// The matrix as a dense Vec<f64> (row-major).
fn as_matrix(a: &NdArray, name: &str) -> (usize, usize, Vec<f64>) {
    if a.ndim != 2 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!("{}: input must be a 2-dimensional array", name)
            )
        );
    }
    let (m, n) = (a.shape[0], a.shape[1]);
    (m, n, a.as_f64())
}

/// `np.dot(a, b)` — full matrix multiplication semantics:
/// - 1-D · 1-D → 0-d array (the scalar dot product)
/// - 2-D · 1-D → 1-D
/// - 1-D · 2-D → 1-D
/// - 2-D · 2-D → 2-D
pub fn dot(a: NdArray, b: NdArray) -> NdArray {
    dot_ref(&a, &b)
}

pub(crate) fn dot_ref(a: &NdArray, b: &NdArray) -> NdArray {
    let a = a;
    let b = b;
    match (a.ndim, b.ndim) {
        (1, 1) => {
            let n = a.size;
            if n != b.size {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!(
                            "shapes ({n},) and ({},) not aligned: {n} (dim 0) != {} (dim 0)",
                            b.size, b.size
                        )
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
            let mut acc = 0.0f64;
            for i in 0..n {
                acc += x[i] * y[i];
            }
            NdArray::from_scalar_f64(acc)
        }
        (2, 1) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            if k != b.size {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!(
                            "shapes ({m},{k}) and ({},) not aligned: {k} (dim 1) != {} (dim 0)",
                            b.size, b.size
                        )
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
            let mut out = vec![0.0f64; m];
            for i in 0..m {
                let mut acc = 0.0f64;
                for j in 0..k {
                    acc += x[i * k + j] * y[j];
                }
                out[i] = acc;
            }
            NdArray::new(vec![m], Dtype::Float64, Data::F64(out))
        }
        (1, 2) => {
            let (k, n) = (b.shape[0], b.shape[1]);
            if a.size != k {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!(
                            "shapes ({},) and ({k},{n}) not aligned: {} (dim 0) != {k} (dim 0)",
                            a.size, a.size
                        )
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
            let mut out = vec![0.0f64; n];
            // i-outer/j-inner: both y and out are walked sequentially (the
            // j-outer spelling strides y by n each step). Accumulation per
            // output element stays in ascending i order — bitwise same.
            for i in 0..k {
                let xi = x[i];
                let y_row = &y[i * n..(i + 1) * n];
                for j in 0..n {
                    out[j] += xi * y_row[j];
                }
            }
            NdArray::new(vec![n], Dtype::Float64, Data::F64(out))
        }
        (2, 2) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!(
                            "shapes ({m},{k}) and ({k2},{n}) not aligned: {k} (dim 1) != {k2} (dim 0)"
                        )
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
            NdArray::new(
                vec![m, n],
                Dtype::Float64,
                Data::F64(matmul_fma(m, k, n, &x, &y)),
            )
        }
        _ => panic!(
            "{}",
            PyException::new("ValueError", "dot: only 1-D and 2-D arrays are supported")
        ),
    }
}

/// `np.matmul(a, b)` — same shapes as dot (the `@` operator).
pub fn matmul(a: NdArray, b: NdArray) -> NdArray {
    dot_ref(&a, &b)
}

/// Borrowed spelling (the operator trait's operands are references; the
/// `as_f64` conversions inside `dot_ref` are the only copies needed).
pub(crate) fn matmul_ref(a: &NdArray, b: &NdArray) -> NdArray {
    dot_ref(a, b)
}

/// `np.vdot(a, b)` — flattened dot product as a plain f64 scalar.
pub fn vdot(a: NdArray, b: NdArray) -> f64 {
    if a.size != b.size {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!("vdot: shapes ({},) and ({},) not aligned", a.size, b.size)
            ),
        );
    }
    let x = a.as_f64();
    let y = b.as_f64();
    let mut acc = 0.0f64;
    for i in 0..a.size {
        acc += x[i] * y[i];
    }
    acc
}

/// `np.linalg.det(a)` — determinant of a square matrix via LU with partial
/// pivoting (row swaps flip the sign).
pub fn det(a: NdArray) -> f64 {
    let (n, n2, mut lu) = as_matrix(&a, "det");
    if n != n2 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!("det: last 2 dimensions of the array must be square (got {n}x{n2})")
            )
        );
    }
    let mut sign = 1.0f64;
    for k in 0..n {
        // Partial pivot.
        let mut piv = k;
        for i in k + 1..n {
            if lu[i * n + k].abs() > lu[piv * n + k].abs() {
                piv = i;
            }
        }
        if lu[piv * n + k] == 0.0 {
            return 0.0;
        }
        if piv != k {
            for j in 0..n {
                lu.swap(k * n + j, piv * n + j);
            }
            sign = -sign;
        }
        for i in k + 1..n {
            let f = lu[i * n + k] / lu[k * n + k];
            for j in k + 1..n {
                lu[i * n + j] -= f * lu[k * n + j];
            }
        }
    }
    let mut d = sign;
    for i in 0..n {
        d *= lu[i * n + i];
    }
    d
}

/// `np.linalg.inv(a)` — inverse of a square matrix by Gauss-Jordan with
/// partial pivoting.
pub fn inv(a: NdArray) -> Result<NdArray, PyException> {
    let (n, n2, m) = as_matrix(&a, "inv");
    if n != n2 {
        return Err(PyException::new(
            "ValueError",
            format!("inv: last 2 dimensions of the array must be square (got {n}x{n2})"),
        ));
    }
    let mut aug = vec![0.0f64; n * n * 2];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = m[i * n + j];
        }
        aug[i * (2 * n) + n + i] = 1.0;
    }
    for k in 0..n {
        let mut piv = k;
        for i in k + 1..n {
            if aug[i * (2 * n) + k].abs() > aug[piv * (2 * n) + k].abs() {
                piv = i;
            }
        }
        if aug[piv * (2 * n) + k] == 0.0 {
            // numpy's own wording, and CATCHABLE (issue #205).
            return Err(PyException::new("LinAlgError", "Singular matrix"));
        }
        if piv != k {
            for j in 0..2 * n {
                aug.swap(k * (2 * n) + j, piv * (2 * n) + j);
            }
        }
        let d = aug[k * (2 * n) + k];
        for j in 0..2 * n {
            aug[k * (2 * n) + j] /= d;
        }
        for i in 0..n {
            if i != k {
                let f = aug[i * (2 * n) + k];
                if f != 0.0 {
                    for j in 0..2 * n {
                        aug[i * (2 * n) + j] -= f * aug[k * (2 * n) + j];
                    }
                }
            }
        }
    }
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }
    Ok(NdArray::new(vec![n, n], Dtype::Float64, Data::F64(out)))
}

/// `np.linalg.solve(a, b)` — solve A·x = b for square A; b may be 1-D
/// (vector) or 2-D (matrix of right-hand sides).
pub fn solve(a: NdArray, b: NdArray) -> Result<NdArray, PyException> {
    let (n, n2, m) = as_matrix(&a, "solve");
    if n != n2 {
        return Err(PyException::new(
            "ValueError",
            format!("solve: input a must be square (got {n}x{n2})"),
        ));
    }
    let rhs_cols = if b.ndim == 2 {
        if b.shape[0] != n {
            return Err(PyException::new(
                "ValueError",
                format!("solve: incompatible dimensions ({n},{n}) and {:?}", b.shape),
            ));
        }
        b.shape[1]
    } else {
        if b.size != n {
            return Err(PyException::new(
                "ValueError",
                format!("solve: incompatible dimensions ({n},{n}) and ({},)", b.size),
            ));
        }
        1
    };
    let bv = b.as_f64();
    // Augmented elimination with partial pivoting, rhs as a matrix.
    let mut aug = vec![0.0f64; n * (n + rhs_cols)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + rhs_cols) + j] = m[i * n + j];
        }
        for c in 0..rhs_cols {
            aug[i * (n + rhs_cols) + n + c] = if b.ndim == 2 {
                bv[i * rhs_cols + c]
            } else {
                bv[i]
            };
        }
    }
    for k in 0..n {
        let mut piv = k;
        for i in k + 1..n {
            if aug[i * (n + rhs_cols) + k].abs() > aug[piv * (n + rhs_cols) + k].abs() {
                piv = i;
            }
        }
        if aug[piv * (n + rhs_cols) + k] == 0.0 {
            return Err(PyException::new("LinAlgError", "Singular matrix"));
        }
        if piv != k {
            for j in 0..n + rhs_cols {
                aug.swap(k * (n + rhs_cols) + j, piv * (n + rhs_cols) + j);
            }
        }
        let d = aug[k * (n + rhs_cols) + k];
        for j in 0..n + rhs_cols {
            aug[k * (n + rhs_cols) + j] /= d;
        }
        for i in 0..n {
            if i != k {
                let f = aug[i * (n + rhs_cols) + k];
                if f != 0.0 {
                    for j in 0..n + rhs_cols {
                        aug[i * (n + rhs_cols) + j] -= f * aug[k * (n + rhs_cols) + j];
                    }
                }
            }
        }
    }
    if rhs_cols == 1 {
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            out[i] = aug[i * (n + 1) + n];
        }
        Ok(NdArray::new(vec![n], Dtype::Float64, Data::F64(out)))
    } else {
        let mut out = vec![0.0f64; n * rhs_cols];
        for i in 0..n {
            for c in 0..rhs_cols {
                out[i * rhs_cols + c] = aug[i * (n + rhs_cols) + n + c];
            }
        }
        Ok(NdArray::new(
            vec![n, rhs_cols],
            Dtype::Float64,
            Data::F64(out),
        ))
    }
}


// ---------------------------------------------------------------------------
// Blocked FMA matmul kernel (the 2-D·2-D fast path).
// ---------------------------------------------------------------------------
//
// The standard GEMM loop nest, sized for this machine's cache hierarchy:
//
//   rayon over i-panels of 24 rows
//     pack A panel -> packed[p * 24 + r] (24 contiguous f64 per p)
//     for jb over n in 256-column blocks
//       for jt in jb..jb+256, step 2 (NEON f64x2) — 24 accumulators live
//         for p in 0..k: 1 B-vector load + 12 packed-A loads + 24 vfma
//       store the 24×2 output block
//
// B's 256-column panel is reused across all 24 rows of the i-panel; A's
// packed panel is reused across all column blocks. FMA contraction means
// the accumulation order differs from the naive triple loop — the same
// class of variation every BLAS (numpy included) has between builds; the
// results are computed to full f64 precision.

/// Sequential aarch64 spelling for builds without the rayon feature: the
/// i-k-p row kernel (same accumulation order as the rayon version).
#[cfg(all(target_arch = "aarch64", not(feature = "numpy-rayon")))]
mod seq_kernel {
    pub fn matmul(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            let a_row = &x[i * k..(i + 1) * k];
            let out_row = &mut out[i * n..(i + 1) * n];
            for p in 0..k {
                let aip = a_row[p];
                let y_row = &y[p * n..(p + 1) * n];
                for j in 0..n {
                    out_row[j] += aip * y_row[j];
                }
            }
        }
        out
    }
}

/// The 2-D·2-D entry: dispatches to the platform kernel.
fn matmul_fma(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
    #[cfg(all(target_arch = "aarch64", feature = "numpy-rayon"))]
    {
        return fast_kernel::matmul(m, k, n, x, y);
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "numpy-rayon")))]
    {
        return seq_kernel::matmul(m, k, n, x, y);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (m, k, n, x, y);
        unreachable!("matmul_fma: non-aarch64 platforms use the row-parallel path")
    }
}

#[cfg(all(target_arch = "aarch64", feature = "numpy-rayon"))]
mod fast_kernel {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    /// The 2-D·2-D multiply: `out[i][j] = sum_p a[i][p] * b[p][j]`.
    pub fn matmul(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        const MR: usize = 24; // rows per panel (24 f64x2 accumulators live)
        const KB: usize = 256; // k depth per pass (L2 chunk)

        let panels: Vec<(usize, usize)> = (0..m)
            .step_by(MR)
            .map(|row0| (row0, (row0 + MR).min(m)))
            .collect();

        // Per-panel packed A: packed[p * 24 + r] = x[row0 + r][p] — 24
        // contiguous f64 per p, so the p-loop's A loads are coalesced.
        let packed: Vec<Vec<f64>> = panels
            .iter()
            .map(|&(row0, row1)| {
                let rows = row1 - row0;
                let mut packed = vec![0.0f64; k * MR];
                for p in 0..k {
                    for r in 0..rows {
                        packed[p * MR + r] = x[(row0 + r) * k + p];
                    }
                }
                packed
            })
            .collect();

        let out_chunks: Vec<Vec<f64>> = panels
            .par_iter()
            .zip(packed.par_iter())
            .map(|(&(row0, row1), packed)| {
                let rows = row1 - row0;
                let nmb = rows.div_ceil(4); // micro-blocks of 4 rows
                let mut chunk = vec![0.0f64; rows * n];
                // Each out element accumulates p in ascending order (the
                // kb chunks are in order, p ascending within a chunk), so
                // the result matches the naive p-ascending spelling up to
                // the FMA contraction of mul+add (one rounding vs two —
                // the same class of variation every BLAS has).
                for kb in (0..k).step_by(KB) {
                    let kw = (kb + KB).min(k) - kb;
                    for jt in (0..n).step_by(2) {
                        // 24 accumulator vectors (nmb micro-blocks × 4
                        // rows), seeded from the running C panel.
                        let mut acc = unsafe { [[vdupq_n_f64(0.0); 4]; MR / 4] };
                        if kb > 0 {
                            for mb in 0..nmb {
                                for r in 0..4 {
                                    let outi = mb * 4 + r;
                                    if outi >= rows {
                                        break;
                                    }
                                    acc[mb][r] = unsafe {
                                        vld1q_f64(chunk.as_ptr().add(outi * n + jt))
                                    };
                                }
                            }
                        }
                        for p in kb..kb + kw {
                            let bv = unsafe { vld1q_f64(y.as_ptr().add(p * n + jt)) };
                            for mb in 0..nmb {
                                let base = p * MR + mb * 4;
                                let a0 = unsafe { vdupq_n_f64(packed[base]) };
                                let a1 = unsafe { vdupq_n_f64(packed[base + 1]) };
                                let a2 = unsafe { vdupq_n_f64(packed[base + 2]) };
                                let a3 = unsafe { vdupq_n_f64(packed[base + 3]) };
                                acc[mb][0] = unsafe { vfmaq_f64(acc[mb][0], a0, bv) };
                                acc[mb][1] = unsafe { vfmaq_f64(acc[mb][1], a1, bv) };
                                acc[mb][2] = unsafe { vfmaq_f64(acc[mb][2], a2, bv) };
                                acc[mb][3] = unsafe { vfmaq_f64(acc[mb][3], a3, bv) };
                            }
                        }
                        // Store the running C panel back.
                        for mb in 0..nmb {
                            for r in 0..4 {
                                let outi = mb * 4 + r;
                                if outi >= rows {
                                    break;
                                }
                                unsafe {
                                    vst1q_f64(
                                        chunk.as_mut_ptr().add(outi * n + jt),
                                        acc[mb][r],
                                    );
                                }
                            }
                        }
                    }
                }
                chunk
            })
            .collect();
        let mut out = Vec::with_capacity(m * n);
        for chunk in out_chunks {
            out.extend_from_slice(&chunk);
        }
        out
    }
}