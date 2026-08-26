//! The `numpy-simd` backend.
//!
//! Stable Rust has no portable SIMD, so this backend's kernels are the
//! SAME sequential kernels the scalar backend runs — LLVM auto-vectorizes
//! those loops on x86-64/aarch64. The docs therefore promise exactly
//! that (issue #164): no runtime cpuid dispatch exists yet, and adding
//! hand-written AVX2/NEON intrinsics here later must keep these entry
//! points' signatures and per-element semantics.
//!
//! Selecting `simd` is indistinguishable from `scalar` today, by design:
//! it exists so `--numpy-backend simd` and `set_backend(Simd)` build and
//! run rather than failing, with measured performance equal to scalar.

use super::scalar;
use super::{BinOp, UnOp};

macro_rules! alias {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t], out: &mut [$t]) {
            scalar::$name(op, a, b, out)
        }
    };
}

alias!(binary_f64, f64);
alias!(binary_f32, f32);
alias!(binary_i64, i64);
alias!(binary_i32, i32);
alias!(binary_bool, bool);

macro_rules! alias_un {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: UnOp, a: &[$t], out: &mut [$t]) {
            scalar::$name(op, a, out)
        }
    };
}

alias_un!(unary_f64, f64);
alias_un!(unary_f32, f32);
alias_un!(unary_i64, i64);
alias_un!(unary_i32, i32);
alias_un!(unary_bool, bool);
