//! Operator support for `NdArray`: `a + b`, `a * 2`, `a @ b`, `a[i] = v`,
//! truthiness, containment, and scalar extraction.
//!
//! Arithmetic follows numpy semantics: elementwise, with broadcasting and
//! dtype promotion. The operators cover arrays against arrays and against
//! the scalar kinds (i64/f64/bool) on either side.
//!
//! Comparisons (`a > 3`, `a == b`) deliberately do NOT lower to Rust
//! operators: numpy returns an elementwise *bool array*, which rython's
//! bool-typed `==`/`<` cannot express. They fail loudly at compile time —
//! use the ufuncs `np.greater(a, 3)`, `np.equal(a, b)`, ... to build
//! boolean masks.

use super::engine::{BinOp, UnOp};
use super::ndarray::{Data, NdArray};
use super::ufunc;
use crate::{PyException, PyRepr};

// ---------------------------------------------------------------------------
// `+` (PyAdd), `-` (Sub), `*` (Mul)
// ---------------------------------------------------------------------------

// `+` uses PyAdd in codegen, so std::ops::Add is only needed for
// Rust-level convenience; Sub and Mul are the codegen paths for `-`/`*`.
impl crate::PyAdd<NdArray> for NdArray {
    type Output = NdArray;
    fn py_add(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Add, self.clone(), rhs.clone())
    }
}
impl crate::PyAdd<i64> for NdArray {
    type Output = NdArray;
    fn py_add(&self, rhs: &i64) -> NdArray {
        ufunc::binary(BinOp::Add, self.clone(), *rhs)
    }
}
impl crate::PyAdd<f64> for NdArray {
    type Output = NdArray;
    fn py_add(&self, rhs: &f64) -> NdArray {
        ufunc::binary(BinOp::Add, self.clone(), *rhs)
    }
}
impl crate::PyAdd<bool> for NdArray {
    type Output = NdArray;
    fn py_add(&self, rhs: &bool) -> NdArray {
        ufunc::binary(BinOp::Add, self.clone(), *rhs)
    }
}
impl crate::PyAdd<NdArray> for i64 {
    type Output = NdArray;
    fn py_add(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Add, *self, rhs.clone())
    }
}
impl crate::PyAdd<NdArray> for f64 {
    type Output = NdArray;
    fn py_add(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Add, *self, rhs.clone())
    }
}
impl crate::PyAdd<NdArray> for bool {
    type Output = NdArray;
    fn py_add(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Add, *self, rhs.clone())
    }
}

// `-` and `*` route through PySub/PyMul (borrowed operands), mirroring
// PyAdd exactly, so `x - y` / `x * y` leave the variables usable.
macro_rules! array_sub_mul {
    ($trait:ident, $method:ident, $op:ident) => {
        impl crate::$trait<NdArray> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                ufunc::binary(BinOp::$op, self.clone(), rhs.clone())
            }
        }
        impl crate::$trait<i64> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &i64) -> NdArray {
                ufunc::binary(BinOp::$op, self.clone(), *rhs)
            }
        }
        impl crate::$trait<f64> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &f64) -> NdArray {
                ufunc::binary(BinOp::$op, self.clone(), *rhs)
            }
        }
        impl crate::$trait<bool> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &bool) -> NdArray {
                ufunc::binary(BinOp::$op, self.clone(), *rhs)
            }
        }
        impl crate::$trait<NdArray> for i64 {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                ufunc::binary(BinOp::$op, *self, rhs.clone())
            }
        }
        impl crate::$trait<NdArray> for f64 {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                ufunc::binary(BinOp::$op, *self, rhs.clone())
            }
        }
        impl crate::$trait<NdArray> for bool {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                ufunc::binary(BinOp::$op, *self, rhs.clone())
            }
        }
    };
}

array_sub_mul!(PySub, py_sub, Sub);
array_sub_mul!(PyMul, py_mul, Mul);

// In-place variants for `a += b` / `a -= b` / `a *= b` on arrays: rython
// arrays are values, so the compound assignment rebinds via a fresh array.
impl std::ops::AddAssign<NdArray> for NdArray {
    fn add_assign(&mut self, rhs: NdArray) {
        *self = ufunc::binary(BinOp::Add, self.clone(), rhs);
    }
}
impl std::ops::SubAssign<NdArray> for NdArray {
    fn sub_assign(&mut self, rhs: NdArray) {
        *self = ufunc::binary(BinOp::Sub, self.clone(), rhs);
    }
}
impl std::ops::MulAssign<NdArray> for NdArray {
    fn mul_assign(&mut self, rhs: NdArray) {
        *self = ufunc::binary(BinOp::Mul, self.clone(), rhs);
    }
}
impl std::ops::AddAssign<i64> for NdArray {
    fn add_assign(&mut self, rhs: i64) {
        *self = ufunc::binary(BinOp::Add, self.clone(), rhs);
    }
}
impl std::ops::SubAssign<i64> for NdArray {
    fn sub_assign(&mut self, rhs: i64) {
        *self = ufunc::binary(BinOp::Sub, self.clone(), rhs);
    }
}
impl std::ops::MulAssign<i64> for NdArray {
    fn mul_assign(&mut self, rhs: i64) {
        *self = ufunc::binary(BinOp::Mul, self.clone(), rhs);
    }
}
impl std::ops::AddAssign<f64> for NdArray {
    fn add_assign(&mut self, rhs: f64) {
        *self = ufunc::binary(BinOp::Add, self.clone(), rhs);
    }
}
impl std::ops::SubAssign<f64> for NdArray {
    fn sub_assign(&mut self, rhs: f64) {
        *self = ufunc::binary(BinOp::Sub, self.clone(), rhs);
    }
}
impl std::ops::MulAssign<f64> for NdArray {
    fn mul_assign(&mut self, rhs: f64) {
        *self = ufunc::binary(BinOp::Mul, self.clone(), rhs);
    }
}

// `a - b` and `a * b` in generated code lower to bare Rust operators.
impl std::ops::Sub<NdArray> for NdArray {
    type Output = NdArray;
    fn sub(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<i64> for NdArray {
    type Output = NdArray;
    fn sub(self, rhs: i64) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<f64> for NdArray {
    type Output = NdArray;
    fn sub(self, rhs: f64) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<bool> for NdArray {
    type Output = NdArray;
    fn sub(self, rhs: bool) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<NdArray> for i64 {
    type Output = NdArray;
    fn sub(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<NdArray> for f64 {
    type Output = NdArray;
    fn sub(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}
impl std::ops::Sub<NdArray> for bool {
    type Output = NdArray;
    fn sub(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Sub, self, rhs)
    }
}

impl std::ops::Mul<NdArray> for NdArray {
    type Output = NdArray;
    fn mul(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<i64> for NdArray {
    type Output = NdArray;
    fn mul(self, rhs: i64) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<f64> for NdArray {
    type Output = NdArray;
    fn mul(self, rhs: f64) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<bool> for NdArray {
    type Output = NdArray;
    fn mul(self, rhs: bool) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<NdArray> for i64 {
    type Output = NdArray;
    fn mul(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<NdArray> for f64 {
    type Output = NdArray;
    fn mul(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}
impl std::ops::Mul<NdArray> for bool {
    type Output = NdArray;
    fn mul(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Mul, self, rhs)
    }
}

// ---------------------------------------------------------------------------
// `/`, `//`, `%`, `**` via the stdpython helper traits
// ---------------------------------------------------------------------------

impl crate::PyDiv<NdArray> for NdArray {
    type Output = NdArray;
    fn py_div(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Div, self.clone(), rhs.clone())
    }
}
impl crate::PyDiv<i64> for NdArray {
    type Output = NdArray;
    fn py_div(&self, rhs: &i64) -> NdArray {
        ufunc::binary(BinOp::Div, self.clone(), *rhs)
    }
}
impl crate::PyDiv<f64> for NdArray {
    type Output = NdArray;
    fn py_div(&self, rhs: &f64) -> NdArray {
        ufunc::binary(BinOp::Div, self.clone(), *rhs)
    }
}
impl crate::PyDiv<bool> for NdArray {
    type Output = NdArray;
    fn py_div(&self, rhs: &bool) -> NdArray {
        ufunc::binary(BinOp::Div, self.clone(), *rhs)
    }
}
impl crate::PyDiv<NdArray> for i64 {
    type Output = NdArray;
    fn py_div(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Div, *self, rhs.clone())
    }
}
impl crate::PyDiv<NdArray> for f64 {
    type Output = NdArray;
    fn py_div(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Div, *self, rhs.clone())
    }
}
impl crate::PyDiv<NdArray> for bool {
    type Output = NdArray;
    fn py_div(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Div, *self, rhs.clone())
    }
}

impl crate::PyFloorDiv<NdArray> for NdArray {
    type Output = NdArray;
    fn py_floordiv(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::FloorDiv, self.clone(), rhs.clone())
    }
}
impl crate::PyFloorDiv<i64> for NdArray {
    type Output = NdArray;
    fn py_floordiv(&self, rhs: &i64) -> NdArray {
        ufunc::binary(BinOp::FloorDiv, self.clone(), *rhs)
    }
}
impl crate::PyFloorDiv<f64> for NdArray {
    type Output = NdArray;
    fn py_floordiv(&self, rhs: &f64) -> NdArray {
        ufunc::binary(BinOp::FloorDiv, self.clone(), *rhs)
    }
}
impl crate::PyFloorDiv<NdArray> for i64 {
    type Output = NdArray;
    fn py_floordiv(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::FloorDiv, *self, rhs.clone())
    }
}
impl crate::PyFloorDiv<NdArray> for f64 {
    type Output = NdArray;
    fn py_floordiv(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::FloorDiv, *self, rhs.clone())
    }
}

impl crate::PyMod<NdArray> for NdArray {
    type Output = NdArray;
    fn py_mod(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Mod, self.clone(), rhs.clone())
    }
}
impl crate::PyMod<i64> for NdArray {
    type Output = NdArray;
    fn py_mod(&self, rhs: &i64) -> NdArray {
        ufunc::binary(BinOp::Mod, self.clone(), *rhs)
    }
}
impl crate::PyMod<f64> for NdArray {
    type Output = NdArray;
    fn py_mod(&self, rhs: &f64) -> NdArray {
        ufunc::binary(BinOp::Mod, self.clone(), *rhs)
    }
}
impl crate::PyMod<NdArray> for i64 {
    type Output = NdArray;
    fn py_mod(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Mod, *self, rhs.clone())
    }
}
impl crate::PyMod<NdArray> for f64 {
    type Output = NdArray;
    fn py_mod(&self, rhs: &NdArray) -> NdArray {
        ufunc::binary(BinOp::Mod, *self, rhs.clone())
    }
}

impl crate::PyPow<NdArray> for NdArray {
    type Output = NdArray;
    fn py_pow(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Pow, self, rhs)
    }
}
impl crate::PyPow<i64> for NdArray {
    type Output = NdArray;
    fn py_pow(self, rhs: i64) -> NdArray {
        ufunc::binary(BinOp::Pow, self, rhs)
    }
}
impl crate::PyPow<f64> for NdArray {
    type Output = NdArray;
    fn py_pow(self, rhs: f64) -> NdArray {
        ufunc::binary(BinOp::Pow, self, rhs)
    }
}
impl crate::PyPow<NdArray> for i64 {
    type Output = NdArray;
    fn py_pow(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Pow, self, rhs)
    }
}
impl crate::PyPow<NdArray> for f64 {
    type Output = NdArray;
    fn py_pow(self, rhs: NdArray) -> NdArray {
        ufunc::binary(BinOp::Pow, self, rhs)
    }
}

// ---------------------------------------------------------------------------
// Unary `-a`, `~a`, `abs(a)`
// ---------------------------------------------------------------------------

impl std::ops::Neg for NdArray {
    type Output = NdArray;
    fn neg(self) -> NdArray {
        NdArray::unary(UnOp::Neg, &self)
    }
}

impl std::ops::Not for NdArray {
    type Output = NdArray;
    fn not(self) -> NdArray {
        // numpy: `~` is bitwise not on ints/bools; floats raise TypeError.
        match self.dtype {
            super::dtype::Dtype::Bool => NdArray::unary(UnOp::LogicalNot, &self),
            super::dtype::Dtype::Int64 | super::dtype::Dtype::Int32 => {
                let vals: Vec<i64> = self.as_i64();
                let out: Vec<i64> = vals.iter().map(|&x| !x).collect();
                let data = if self.dtype == super::dtype::Dtype::Int64 {
                    Data::I64(out)
                } else {
                    Data::I32(out.iter().map(|&x| x as i32).collect())
                };
                NdArray::new(self.shape.clone(), self.dtype, data)
            }
            _ => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "ufunc 'invert' not supported for the input types"
                )
            ),
        }
    }
}

impl crate::PyAbs for NdArray {
    type Output = NdArray;
    fn py_abs(self) -> NdArray {
        NdArray::unary(UnOp::Abs, &self)
    }
}

// ---------------------------------------------------------------------------
// Truthiness (not a / if a) — numpy's ambiguous-truth error
// ---------------------------------------------------------------------------

impl crate::Truthy for NdArray {
    fn is_truthy(&self) -> bool {
        crate::PyBool::py_bool(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Comparisons: a == b, a < b, ... — numpy returns an ARRAY (bool dtype),
// broadcasting scalars against the array like the comparison ufuncs.
// ---------------------------------------------------------------------------

/// Elementwise comparison returning a bool NdArray. Scalar/side order is
/// preserved so `2 < arr` and `arr < 2` both work.
macro_rules! array_cmp {
    ($trait:ident, $method:ident, $ufunc:ident) => {
        impl crate::$trait<NdArray> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                super::$ufunc(self.clone(), rhs.clone())
            }
        }
        impl crate::$trait<i64> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &i64) -> NdArray {
                super::$ufunc(self.clone(), *rhs)
            }
        }
        impl crate::$trait<f64> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &f64) -> NdArray {
                super::$ufunc(self.clone(), *rhs)
            }
        }
        impl crate::$trait<bool> for NdArray {
            type Output = NdArray;
            fn $method(&self, rhs: &bool) -> NdArray {
                super::$ufunc(self.clone(), *rhs)
            }
        }
        impl crate::$trait<NdArray> for i64 {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                super::$ufunc(*self, rhs.clone())
            }
        }
        impl crate::$trait<NdArray> for f64 {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                super::$ufunc(*self, rhs.clone())
            }
        }
        impl crate::$trait<NdArray> for bool {
            type Output = NdArray;
            fn $method(&self, rhs: &NdArray) -> NdArray {
                super::$ufunc(*self, rhs.clone())
            }
        }
    };
}

array_cmp!(PyEq, py_eq, equal);
array_cmp!(PyNe, py_ne, not_equal);
array_cmp!(PyLt, py_lt, less);
array_cmp!(PyLe, py_le, less_equal);
array_cmp!(PyGt, py_gt, greater);
array_cmp!(PyGe, py_ge, greater_equal);

// ---------------------------------------------------------------------------
// `x in a` — flattened membership test (numpy semantics)
// ---------------------------------------------------------------------------

impl crate::PyContains<i64> for NdArray {
    fn py_contains(&self, value: &i64) -> bool {
        self.as_i64().iter().any(|&x| x == *value)
    }
}
impl crate::PyContains<f64> for NdArray {
    fn py_contains(&self, value: &f64) -> bool {
        self.as_f64().iter().any(|&x| x == *value)
    }
}
impl crate::PyContains<bool> for NdArray {
    fn py_contains(&self, value: &bool) -> bool {
        self.as_bool().iter().any(|&x| x == *value)
    }
}
impl crate::PyContains<NdArray> for NdArray {
    fn py_contains(&self, value: &NdArray) -> bool {
        if value.size != 1 {
            panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "invalid type promotion: the truth value of an array is ambiguous"
                )
            );
        }
        let v = value.as_f64()[0];
        self.as_f64().iter().any(|&x| x == v)
    }
}

// ---------------------------------------------------------------------------
// Scalar extraction — numpy's ndarray.item()
// ---------------------------------------------------------------------------

impl NdArray {
    /// `a.item()` — the single element as f64 (numpy returns a scalar).
    /// Panics for arrays with more than one element, like numpy.
    pub fn item(&self) -> f64 {
        if self.size != 1 {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "can only convert an array of size 1 to a Python scalar (array has size {})",
                        self.size
                    )
                )
            );
        }
        self.to_f64()
    }

    /// `a.item()` as i64 (truncating float arrays, like numpy's int casts).
    pub fn item_i64(&self) -> i64 {
        if self.size != 1 {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "can only convert an array of size 1 to a Python scalar (array has size {})",
                        self.size
                    )
                )
            );
        }
        self.to_i64()
    }

    /// `a.item()` as bool (nonzero = true, like numpy).
    pub fn item_bool(&self) -> bool {
        if self.size != 1 {
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "can only convert an array of size 1 to a Python scalar (array has size {})",
                        self.size
                    )
                )
            );
        }
        self.to_bool()
    }
}

// ---------------------------------------------------------------------------
// Stores: a[i] = v, a[i, j] = v, a[mask] = v
// ---------------------------------------------------------------------------

impl NdArray {
    /// Fill the sub-array at flat `offset` (spanning `len` elements) from a
    /// scalar value, broadcasting like numpy (`a[0] = 5` fills a row).
    fn set_scalar_at(&mut self, offset: usize, len: usize, v: f64) {
        match &mut self.data {
            Data::F64(buf) => {
                for x in &mut buf[offset..offset + len] {
                    *x = v;
                }
            }
            Data::F32(buf) => {
                for x in &mut buf[offset..offset + len] {
                    *x = v as f32;
                }
            }
            Data::I64(buf) => {
                for x in &mut buf[offset..offset + len] {
                    *x = v as i64;
                }
            }
            Data::I32(buf) => {
                for x in &mut buf[offset..offset + len] {
                    *x = v as i32;
                }
            }
            Data::Bool(buf) => {
                for x in &mut buf[offset..offset + len] {
                    *x = v != 0.0;
                }
            }
        }
    }

    /// Copy `src` (an array) into the sub-array at flat `offset`.
    pub(crate) fn copy_into(&mut self, offset: usize, src: &NdArray) {
        let len = src.size;
        match (&mut self.data, &src.data) {
            (Data::F64(buf), Data::F64(s)) => buf[offset..offset + len].copy_from_slice(s),
            (Data::F64(buf), Data::F32(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x as f64;
                }
            }
            (Data::F64(buf), Data::I64(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x as f64;
                }
            }
            (Data::F64(buf), Data::I32(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x as f64;
                }
            }
            (Data::F64(buf), Data::Bool(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = if x { 1.0 } else { 0.0 };
                }
            }
            (Data::I64(buf), Data::I64(s)) => buf[offset..offset + len].copy_from_slice(s),
            (Data::I64(buf), Data::I32(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x as i64;
                }
            }
            (Data::I64(buf), Data::F64(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x as i64;
                }
            }
            (Data::I64(buf), Data::Bool(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = if x { 1 } else { 0 };
                }
            }
            (Data::Bool(buf), Data::Bool(s)) => buf[offset..offset + len].copy_from_slice(s),
            (Data::Bool(buf), Data::F64(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x != 0.0;
                }
            }
            (Data::Bool(buf), Data::I64(s)) => {
                for (i, &x) in s.iter().enumerate() {
                    buf[offset + i] = x != 0;
                }
            }
            _ => {
                // F32/I32 receivers: convert through f64/i64.
                let widened = self.astype(super::dtype::Dtype::Float64);
                let mut w = widened;
                w.copy_into(offset, src);
                *self = w.astype(self.dtype);
            }
        }
    }
}

/// `a[i] = value` — a scalar broadcasts over the whole sub-array (numpy
/// `a[0] = 5` fills row 0); an array value replaces the sub-array.
impl crate::PySetIndex<i64, NdArray> for NdArray {
    fn py_set_index(&mut self, index: i64, value: NdArray) -> Result<(), PyException> {
        if self.ndim == 0 {
            return Err(PyException::new("IndexError", "too many indices for array"));
        }
        let i = super::ndarray::checked_index(index, self.shape[0])?;
        let stride: usize = self.shape[1..].iter().product();
        if value.size == 1 && self.ndim == 1 {
            self.copy_into(i, &value);
            return Ok(());
        }
        if value.shape != self.shape[1..] {
            return Err(PyException::new(
                "ValueError",
                format!(
                    "could not broadcast input array from shape {:?} into shape {:?}",
                    value.shape,
                    &self.shape[1..]
                ),
            ));
        }
        self.copy_into(i * stride, &value);
        Ok(())
    }
}

macro_rules! scalar_set_index {
    ($(($ty:ty, $conv:expr)),* $(,)?) => {
        $(
            impl crate::PySetIndex<i64, $ty> for NdArray {
                fn py_set_index(&mut self, index: i64, value: $ty) -> Result<(), PyException> {
                    if self.ndim == 0 {
                        return Err(PyException::new("IndexError", "too many indices for array"));
                    }
                    let i = super::ndarray::checked_index(index, self.shape[0])?;
                    let stride: usize = self.shape[1..].iter().product();
                    let len = stride.max(1);
                    self.set_scalar_at(i * stride, len, ($conv)(value));
                    Ok(())
                }
            }
        )*
    };
}

scalar_set_index!(
    (i64, |v: i64| v as f64),
    (f64, |v: f64| v),
    (bool, |v: bool| if v { 1.0 } else { 0.0 }),
);

impl crate::PySetIndex<(i64, i64), NdArray> for NdArray {
    fn py_set_index(&mut self, index: (i64, i64), value: NdArray) -> Result<(), PyException> {
        if self.ndim != 2 {
            return Err(PyException::new(
                "IndexError",
                format!(
                    "too many indices for array: array is {}-dimensional, but 2 were indexed",
                    self.ndim
                ),
            ));
        }
        if value.size != 1 {
            return Err(PyException::new(
                "ValueError",
                "setting an array element with a sequence",
            ));
        }
        let i = super::ndarray::checked_index(index.0, self.shape[0])?;
        let j = super::ndarray::checked_index(index.1, self.shape[1])?;
        self.copy_into(i * self.shape[1] + j, &value);
        Ok(())
    }
}

macro_rules! scalar_set_index2 {
    ($(($ty:ty, $conv:expr)),* $(,)?) => {
        $(
            impl crate::PySetIndex<(i64, i64), $ty> for NdArray {
                fn py_set_index(&mut self, index: (i64, i64), value: $ty) -> Result<(), PyException> {
                    if self.ndim != 2 {
                        return Err(PyException::new(
                            "IndexError",
                            format!(
                                "too many indices for array: array is {}-dimensional, but 2 were indexed",
                                self.ndim
                            ),
                        ));
                    }
                    let i = super::ndarray::checked_index(index.0, self.shape[0])?;
                    let j = super::ndarray::checked_index(index.1, self.shape[1])?;
                    self.set_scalar_at(i * self.shape[1] + j, 1, ($conv)(value));
                    Ok(())
                }
            }
        )*
    };
}

scalar_set_index2!(
    (i64, |v: i64| v as f64),
    (f64, |v: f64| v),
    (bool, |v: bool| if v { 1.0 } else { 0.0 }),
);

/// `a[mask] = value` — numpy boolean-mask assignment (1-D value or
/// broadcast scalar).
impl crate::PySetIndex<NdArray, NdArray> for NdArray {
    fn py_set_index(&mut self, mask: NdArray, value: NdArray) -> Result<(), PyException> {
        if mask.dtype != super::dtype::Dtype::Bool {
            return Err(PyException::new(
                "IndexError",
                "arrays used as indices must be of integer (or boolean) type",
            ));
        }
        let keep: Vec<usize> = mask
            .bool()
            .iter()
            .enumerate()
            .filter(|&(_, b)| *b)
            .map(|(i, _)| i)
            .collect();
        let stride: usize = self.shape[1..].iter().product();
        if value.size == 1 {
            for &i in &keep {
                self.set_scalar_at(i * stride, stride.max(1), value.to_f64());
            }
            return Ok(());
        }
        if value.size != keep.len() {
            return Err(PyException::new(
                "ValueError",
                "boolean index did not match indexed array along dimension 0",
            ));
        }
        for (k, &i) in keep.iter().enumerate() {
            let offset = i * stride;
            let v = value.subarray(&[k]);
            self.copy_into(offset, &v);
        }
        Ok(())
    }
}

impl crate::PySetIndex<NdArray, i64> for NdArray {
    fn py_set_index(&mut self, mask: NdArray, value: i64) -> Result<(), PyException> {
        self.py_set_index(mask, NdArray::from_scalar_i64(value))
    }
}
impl crate::PySetIndex<NdArray, f64> for NdArray {
    fn py_set_index(&mut self, mask: NdArray, value: f64) -> Result<(), PyException> {
        self.py_set_index(mask, NdArray::from_scalar_f64(value))
    }
}
impl crate::PySetIndex<NdArray, bool> for NdArray {
    fn py_set_index(&mut self, mask: NdArray, value: bool) -> Result<(), PyException> {
        self.py_set_index(mask, NdArray::from_scalar_bool(value))
    }
}

// Keep PyRepr referenced from this module (repr debugging).
#[allow(dead_code)]
fn _dbg(a: &NdArray) -> String {
    PyRepr::py_repr(a)
}
