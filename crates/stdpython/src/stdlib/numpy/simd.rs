//! The `numpy-simd` backend.
//!
//! Stable Rust has no portable SIMD, but it does not need any: the scalar
//! kernels are now structured so the op dispatch happens ONCE per call and
//! each arm is a tight inner loop LLVM auto-vectorizes to NEON (aarch64)
//! / SSE2-AVX2 (x86-64) — verified by disassembling the release build
//! (`fadd.2d`/`fmul.2d` etc. appear in the hot kernels). This backend is
//! therefore an alias of `scalar`, kept as a separate entry point so
//! `--numpy-backend simd` and `set_backend(Simd)` build and run, with a
//! place to drop hand-written intrinsics later if any measured gap ever
//! justifies them (issue #164). Any such intrinsics must keep these entry
//! points' signatures and per-element semantics.
//!
//! Selecting `simd` is indistinguishable from `scalar` today, by design:
//! both compile to the same auto-vectorized loops.

use super::scalar;
use super::{BinOp, UnOp};

macro_rules! alias {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], b: &[$t]) -> Vec<$t> {
            scalar::$name(op, a, b)
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
        pub(crate) fn $name(op: UnOp, a: &[$t]) -> Vec<$t> {
            scalar::$name(op, a)
        }
    };
}

macro_rules! alias_scalar {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(op: BinOp, a: &[$t], s: $t, s_left: bool) -> Vec<$t> {
            scalar::$name(op, a, s, s_left)
        }
    };
}

alias_scalar!(binary_f64_scalar, f64);
alias_scalar!(binary_f32_scalar, f32);
alias_scalar!(binary_i64_scalar, i64);
alias_scalar!(binary_i32_scalar, i32);
alias_scalar!(binary_bool_scalar, bool);

alias_un!(unary_f64, f64);
alias_un!(unary_f32, f32);
alias_un!(unary_i64, i64);
alias_un!(unary_i32, i32);
alias_un!(unary_bool, bool);
