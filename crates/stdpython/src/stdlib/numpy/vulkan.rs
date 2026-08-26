//! The `numpy-vulkan` backend.
//!
//! Declared feature and engine row exist, but NO vulkan kernels ship in this
//! build (issue #164). Selecting it is a LOUD runtime error — never a
//! silent fallback to another backend. `Auto` never selects it
//! (available() is false).

use super::{BinOp, UnOp};

pub(crate) fn available() -> bool {
    false
}

macro_rules! not_impl_bin {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t], out: &mut [$t]) {
            let _ = (op, a, b, out);
            panic!(
                "{}",
                crate::PyException::new(
                    "RuntimeError",
                    "numpy backend `vulkan` is not implemented in this build (the                      feature compiles, but no vulkan kernels ship yet); select                      scalar/rayon/simd or rebuild with a different backend"
                )
            )
        }
    };
}
not_impl_bin!(binary_f64, f64);
not_impl_bin!(binary_f32, f32);
not_impl_bin!(binary_i64, i64);
not_impl_bin!(binary_i32, i32);
not_impl_bin!(binary_bool, bool);

macro_rules! not_impl_un {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: UnOp, a: &[$t], out: &mut [$t]) {
            let _ = (op, a, out);
            panic!(
                "{}",
                crate::PyException::new(
                    "RuntimeError",
                    "numpy backend `vulkan` is not implemented in this build (the                      feature compiles, but no vulkan kernels ship yet); select                      scalar/rayon/simd or rebuild with a different backend"
                )
            )
        }
    };
}
not_impl_un!(unary_f64, f64);
not_impl_un!(unary_f32, f32);
not_impl_un!(unary_i64, i64);
not_impl_un!(unary_i32, i32);
not_impl_un!(unary_bool, bool);
