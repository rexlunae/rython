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
    match (a.ndim, b.ndim) {
        (1, 1) => {
            let n = a.size;
            if n != b.size {
                panic!(
                    "{}",
                    PyException::new(
                        "ValueError",
                        format!("shapes ({n},) and ({},) not aligned: {n} (dim 0) != {} (dim 0)", b.size, b.size)
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
                        format!("shapes ({m},{k}) and ({},) not aligned: {k} (dim 1) != {} (dim 0)", b.size, b.size)
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
                        format!("shapes ({},) and ({k},{n}) not aligned: {} (dim 0) != {k} (dim 0)", a.size, a.size)
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
            let mut out = vec![0.0f64; n];
            for j in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += x[i] * y[i * n + j];
                }
                out[j] = acc;
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
                        format!("shapes ({m},{k}) and ({k2},{n}) not aligned: {k} (dim 1) != {k2} (dim 0)")
                    )
                );
            }
            let x = a.as_f64();
            let y = b.as_f64();
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
            NdArray::new(vec![m, n], Dtype::Float64, Data::F64(out))
        }
        _ => panic!(
            "{}",
            PyException::new(
                "ValueError",
                "dot: only 1-D and 2-D arrays are supported"
            )
        ),
    }
}

/// `np.matmul(a, b)` — same shapes as dot (the `@` operator).
pub fn matmul(a: NdArray, b: NdArray) -> NdArray {
    dot(a, b)
}

/// `np.vdot(a, b)` — flattened dot product as a plain f64 scalar.
pub fn vdot(a: NdArray, b: NdArray) -> f64 {
    if a.size != b.size {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!(
                    "vdot: shapes ({},) and ({},) not aligned",
                    a.size, b.size
                )
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
pub fn inv(a: NdArray) -> NdArray {
    let (n, n2, m) = as_matrix(&a, "inv");
    if n != n2 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!("inv: last 2 dimensions of the array must be square (got {n}x{n2})")
            )
        );
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
            panic!(
                "{}",
                PyException::new(
                    "LinAlgError",
                    "Singular matrix (det == 0); cannot invert"
                )
            );
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
    NdArray::new(vec![n, n], Dtype::Float64, Data::F64(out))
}

/// `np.linalg.solve(a, b)` — solve A·x = b for square A; b may be 1-D
/// (vector) or 2-D (matrix of right-hand sides).
pub fn solve(a: NdArray, b: NdArray) -> NdArray {
    let (n, n2, m) = as_matrix(&a, "solve");
    if n != n2 {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                format!("solve: input a must be square (got {n}x{n2})")
            )
        );
    }
    let rhs_cols = if b.ndim == 2 {
        if b.shape[0] != n {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "solve: incompatible dimensions ({n},{n}) and {:?}",
                        b.shape
                    )
                )
            );
        }
        b.shape[1]
    } else {
        if b.size != n {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!("solve: incompatible dimensions ({n},{n}) and ({},)", b.size)
                )
            );
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
            panic!(
                "{}",
                PyException::new("LinAlgError", "Singular matrix")
            );
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
        NdArray::new(vec![n], Dtype::Float64, Data::F64(out))
    } else {
        let mut out = vec![0.0f64; n * rhs_cols];
        for i in 0..n {
            for c in 0..rhs_cols {
                out[i * rhs_cols + c] = aug[i * (n + rhs_cols) + n + c];
            }
        }
        NdArray::new(vec![n, rhs_cols], Dtype::Float64, Data::F64(out))
    }
}
