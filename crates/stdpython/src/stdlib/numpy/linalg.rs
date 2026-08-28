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

/// The portable sequential fallback: the i-k-p row kernel (same
/// accumulation order as the SIMD versions), used on platforms without
/// the rayon feature or without the SIMD extensions the fast kernels
/// need.
#[cfg(not(all(target_arch = "aarch64", feature = "numpy-rayon")))]
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

/// x86_64 AVX2+FMA blocked kernel: the same panel structure as the NEON
/// kernel, with f64x4 vectors (4 columns per FMA — twice NEON's width).
/// Runtime-detected: pre-AVX2 CPUs take the sequential fallback.
#[cfg(all(target_arch = "x86_64", feature = "numpy-rayon"))]
mod fast_kernel_x86 {
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    /// The 2-D·2-D multiply: `out[i][j] = sum_p a[i][p] * b[p][j]`.
    ///
    /// 4×4 register-blocked micro-kernel (4 f64x4 accumulators fit the 16
    /// ymm registers, unlike the NEON 24-accumulator layout): the C block
    /// is read-modify-written in place per jt step, B's row vector is
    /// reused across the 4 rows of a micro-block, and A's broadcasts hit
    /// the packed panel in L1.
    pub fn matmul(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return super::seq_kernel::matmul(m, k, n, x, y);
        }
        let mut out = vec![0.0f64; m * n];
        const MR: usize = 24; // rows per rayon panel (6 micro-blocks of 4)

        // Per-panel packed A: packed[p * 24 + r] = x[row0 + r][p] — 24
        // contiguous f64 per p, so the p-loop's A broadcasts hit L1.
        let panels: Vec<(usize, usize)> = (0..m)
            .step_by(MR)
            .map(|row0| (row0, (row0 + MR).min(m)))
            .collect();

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

        unsafe {
            panels
                .par_iter()
                .zip(packed.par_iter())
                .for_each(|(&(row0, row1), packed)| {
                    let rows = row1 - row0;
                    let nmb = rows.div_ceil(4); // micro-blocks of 4 rows
                    for jt in (0..n).step_by(4) {
                        for mb in 0..nmb {
                            let r0 = row0 + mb * 4;
                            let ri = (r0 + 4).min(row1);
                            let nrows = ri - r0;
                            // The 4 accumulator vectors for this
                            // micro-block, seeded from C (zero on the
                            // first k pass — C starts zeroed).
                            let mut acc = [
                                _mm256_loadu_pd(out.as_ptr().add(r0 * n + jt)),
                                _mm256_loadu_pd(out.as_ptr().add((r0 + 1).min(row1) * n + jt)),
                                _mm256_loadu_pd(out.as_ptr().add((r0 + 2).min(row1) * n + jt)),
                                _mm256_loadu_pd(out.as_ptr().add((r0 + 3).min(row1) * n + jt)),
                            ];
                            for p in 0..k {
                                let bv = _mm256_loadu_pd(y.as_ptr().add(p * n + jt));
                                for r in 0..nrows {
                                    let a = _mm256_set1_pd(packed[p * MR + r]);
                                    acc[r] = _mm256_fmadd_pd(a, bv, acc[r]);
                                }
                            }
                            for r in 0..nrows {
                                _mm256_storeu_pd(
                                    out.as_mut_ptr().add((r0 + r) * n + jt),
                                    acc[r],
                                );
                            }
                        }
                    }
                });
        }
        out
    }
}

/// The 2-D·2-D entry: dispatches to the platform kernel.
fn matmul_fma(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
    #[cfg(all(target_arch = "aarch64", feature = "numpy-rayon"))]
    {
        return fast_kernel_neon::matmul(m, k, n, x, y);
    }
    #[cfg(all(target_arch = "x86_64", feature = "numpy-rayon"))]
    {
        return fast_kernel_x86::matmul(m, k, n, x, y);
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "numpy-rayon"),
        all(target_arch = "x86_64", feature = "numpy-rayon")
    )))]
    {
        return seq_kernel::matmul(m, k, n, x, y);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "numpy-rayon"))]
#[allow(unsafe_op_in_unsafe_fn)]
mod fast_kernel_neon {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    const MR: usize = 24; // rows per panel (6 micro-blocks of 4 rows)
    const NB: usize = 256; // B/output columns per block (L2 chunk)
    const KB: usize = 256; // k depth per pass (packed A+B chunks, L2)

    /// The 2-D·2-D multiply: `out[i][j] = sum_p a[i][p] * b[p][j]`.
    ///
    /// GEMM structure: rayon over i-panels of 24 rows; A and B are both
    /// PACKED into p-major panels so the p-loop's loads are contiguous
    /// (the naive spelling strides B by n×8 bytes per p — a cache miss on
    /// every access); the micro-kernel is 4 rows × 8 cols with
    /// lane-indexed FMA — the A vector's lanes select the row multiplier
    /// and the B vector's lanes are the column values, so there are no
    /// scalar broadcasts. Each output element accumulates p in ascending
    /// order, so the results match the naive spelling up to the FMA
    /// contraction of mul+add (the same class of variation every BLAS has
    /// between builds). The whole kernel body is one unsafe block
    /// wrapping the NEON intrinsics and raw-pointer panel accesses.
    pub fn matmul(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        let panels: Vec<(usize, usize)> = (0..m)
            .step_by(MR)
            .map(|row0| (row0, (row0 + MR).min(m)))
            .collect();

        let packed_a: Vec<Vec<f64>> = panels
            .iter()
            .map(|&(row0, row1)| {
                let rows = row1 - row0;
                let mut pa = vec![0.0f64; k * MR];
                for p in 0..k {
                    for r in 0..rows {
                        pa[p * MR + r] = x[(row0 + r) * k + p];
                    }
                }
                pa
            })
            .collect();

        // Pack B ONCE, p-major: packed_b[p * n + c] = y[p * n + c]. The
        // packed rows are contiguous (n f64 per p), shared read-only by
        // every rayon panel — packing per panel would copy B m/24 times.
        let packed_b: Vec<f64> = y.to_vec();

        let out_chunks: Vec<Vec<f64>> = panels
            .par_iter()
            .zip(packed_a.par_iter())
            .map(|(&(row0, row1), packed_a)| {
                let rows = row1 - row0;
                let nmb = rows.div_ceil(4);
                let mut chunk = vec![0.0f64; rows * n];
                unsafe {
                for jb in (0..n).step_by(NB) {
                    let jw = (jb + NB).min(n) - jb;
                    // k-blocking: accumulate into the C chunk in KB-deep
                    // passes (each out element's p order stays ascending).
                    for kb in (0..k).step_by(KB) {
                        let kw = (kb + KB).min(k) - kb;
                        for mb in 0..nmb {
                            let r0 = mb * 4;
                            let nr = (r0 + 4).min(rows) - r0;
                            for jt in (0..jw).step_by(8) {
                                let jend = (jt + 8).min(jw);
                                let nv = (jend - jt) / 2; // full 2-col vectors
                                // 8 accumulator vectors: 4 rows × 2 cols.
                                let mut acc = [[vdupq_n_f64(0.0); 4]; 4];
                                if kb > 0 {
                                    for r in 0..nr {
                                        for vi in 0..nv {
                                            acc[r][vi] = vld1q_f64(
                                                chunk.as_ptr()
                                                    .add((r0 + r) * n + jb + jt + vi * 2),
                                            );
                                        }
                                    }
                                }
                                for p in kb..kb + kw {
                                    let a01 = vld1q_f64(packed_a.as_ptr().add(p * MR + r0));
                                    let a23 =
                                        vld1q_f64(packed_a.as_ptr().add(p * MR + r0 + 2));
                                    for vi in 0..nv {
                                        let bv = vld1q_f64(
                                            packed_b.as_ptr().add(p * n + jb + jt + vi * 2),
                                        );
                                        // vfmaq_laneq_f64::<L>(r, a, b) is
                                        // r[i] + a[i] * b[L]: the B vector is
                                        // element-wise, the A vector's LANE
                                        // selects the row multiplier.
                                        acc[0][vi] = vfmaq_laneq_f64::<0>(acc[0][vi], bv, a01);
                                        acc[1][vi] = vfmaq_laneq_f64::<1>(acc[1][vi], bv, a01);
                                        acc[2][vi] = vfmaq_laneq_f64::<0>(acc[2][vi], bv, a23);
                                        acc[3][vi] = vfmaq_laneq_f64::<1>(acc[3][vi], bv, a23);
                                    }
                                }
                                for r in 0..nr {
                                    for vi in 0..nv {
                                        vst1q_f64(
                                            chunk.as_mut_ptr()
                                                .add((r0 + r) * n + jb + jt + vi * 2),
                                            acc[r][vi],
                                        );
                                    }
                                    // Scalar tail for the 0-1 trailing column.
                                    if (jend - jt) % 2 == 1 {
                                        let c = jend - 1;
                                        let mut sv = chunk[(r0 + r) * n + jb + c];
                                        for p in kb..kb + kw {
                                            sv += packed_a[p * MR + r0 + r]
                                                * packed_b[p * n + jb + c];
                                        }
                                        chunk[(r0 + r) * n + jb + c] = sv;
                                    }
                                }
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
#[cfg(all(test, target_arch = "aarch64", feature = "numpy-rayon"))]
mod matmul_kernel_tests {
    use super::fast_kernel_neon;

    /// The naive i-j-p reference. Integer-valued inputs make the
    /// comparison exact in any summation order (all intermediate values
    /// stay below 2^53), so the FMA contraction can't move the result.
    fn naive(m: usize, k: usize, n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += x[i * k + p] * y[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn fma_kernel_matches_naive() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (4, 4, 4),
            (7, 5, 9),    // tail micro-block
            (8, 8, 8),
            (24, 24, 24), // one full panel
            (25, 7, 13),  // panel + tail, non-square
            (3, 9, 5),
            (30, 300, 40),  // k > KB (exercises the k-block seam)
            (5, 300, 600),  // n > NB (exercises the j-block stride)
            (25, 1024, 512), // both seams at once
        ] {
            let x: Vec<f64> = (0..m * k).map(|i| ((i * 7) % 13) as f64 - 3.0).collect();
            let y: Vec<f64> = (0..k * n).map(|i| ((i * 5) % 11) as f64 - 2.0).collect();
            let want = naive(m, k, n, &x, &y);
            let got = fast_kernel_neon::matmul(m, k, n, &x, &y);
            assert_eq!(got.len(), want.len(), "{m}x{k}x{n}");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g, w,
                    "matmul {m}x{k}x{n} element {i} (row {}, col {}): got {g}, want {w}",
                    i / n,
                    i % n
                );
            }
        }
    }
}
