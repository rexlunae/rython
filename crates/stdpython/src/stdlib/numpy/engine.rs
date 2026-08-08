//! Execution backends for numpy elementwise kernels and reductions.
//!
//! Every numpy operation in this module funnels through one of the `engine`
//! functions below. The active backend is chosen once per process:
//!
//! 1. `set_backend(...)` — pinned by generated code from the compiler's
//!    `--numpy-backend` flag (rythonc / rypip convert).
//! 2. The `RYPY_NUMPY_BACKEND` environment variable (`scalar`, `rayon`,
//!    `simd`, `cuda`, `vulkan`, `auto`).
//! 3. `Auto`: the best backend compiled into this stdpython build.
//!
//! The backends are compiled in by Cargo features:
//!
//! | Feature          | Backend | Requirements                           |
//! |------------------|---------|----------------------------------------|
//! | `numpy`          | scalar  | none (always available)                |
//! | `numpy-rayon`    | rayon   | multithreaded elementwise + reductions |
//! | `numpy-simd`     | simd    | runtime-detected AVX2/AVX-512/SSE2/NEON |
//! | `numpy-cuda`     | cuda    | CUDA driver at runtime (cudarc)        |
//! | `numpy-vulkan`   | vulkan  | Vulkan driver at runtime (ash)         |

use core::sync::atomic::{AtomicU8, Ordering};

/// A requested execution backend. `Auto` resolves at first use to the best
/// backend compiled in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Auto,
    Scalar,
    Rayon,
    Simd,
    Cuda,
    Vulkan,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Scalar => "scalar",
            Backend::Rayon => "rayon",
            Backend::Simd => "simd",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
        }
    }

    pub fn from_str(s: &str) -> Option<Backend> {
        Some(match s {
            "auto" => Backend::Auto,
            "scalar" => Backend::Scalar,
            "rayon" => Backend::Rayon,
            "simd" => Backend::Simd,
            "cuda" => Backend::Cuda,
            "vulkan" => Backend::Vulkan,
            _ => return None,
        })
    }

    /// The stdpython feature that compiles this backend in, if any.
    pub fn requires_feature(&self) -> Option<&'static str> {
        match self {
            Backend::Scalar | Backend::Auto => None,
            Backend::Rayon => Some("numpy-rayon"),
            Backend::Simd => Some("numpy-simd"),
            Backend::Cuda => Some("numpy-cuda"),
            Backend::Vulkan => Some("numpy-vulkan"),
        }
    }

    /// Whether this backend is compiled into the current stdpython build.
    pub fn is_compiled_in(&self) -> bool {
        match self {
            Backend::Auto | Backend::Scalar => true,
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => true,
            #[cfg(not(feature = "numpy-rayon"))]
            Backend::Rayon => false,
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => true,
            #[cfg(not(feature = "numpy-simd"))]
            Backend::Simd => false,
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => true,
            #[cfg(not(feature = "numpy-cuda"))]
            Backend::Cuda => false,
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => true,
            #[cfg(not(feature = "numpy-vulkan"))]
            Backend::Vulkan => false,
        }
    }
}

/// Compile-time assertion used by generated code: `--numpy-backend simd`
/// emits `const _: () = stdpython::numpy::assert_backend_available(...)`, so
/// a build whose stdpython features do not include the requested backend
/// fails at compile time with this message instead of at runtime.
pub const fn assert_backend_available(b: Backend) {
    match b {
        Backend::Auto | Backend::Scalar => {}
        Backend::Rayon => {
            #[cfg(not(feature = "numpy-rayon"))]
            {
                panic!(
                    "this program requests the numpy `rayon` backend, but stdpython was \
                     built without the `numpy-rayon` feature; rebuild stdpython with \
                     --features numpy-rayon (or convert without --numpy-backend rayon)"
                )
            }
        }
        Backend::Simd => {
            #[cfg(not(feature = "numpy-simd"))]
            {
                panic!(
                    "this program requests the numpy `simd` backend, but stdpython was \
                     built without the `numpy-simd` feature; rebuild stdpython with \
                     --features numpy-simd (or convert without --numpy-backend simd)"
                )
            }
        }
        Backend::Cuda => {
            #[cfg(not(feature = "numpy-cuda"))]
            {
                panic!(
                    "this program requests the numpy `cuda` backend, but stdpython was \
                     built without the `numpy-cuda` feature; rebuild stdpython with \
                     --features numpy-cuda (or convert without --numpy-backend cuda)"
                )
            }
        }
        Backend::Vulkan => {
            #[cfg(not(feature = "numpy-vulkan"))]
            {
                panic!(
                    "this program requests the numpy `vulkan` backend, but stdpython was \
                     built without the `numpy-vulkan` feature; rebuild stdpython with \
                     --features numpy-vulkan (or convert without --numpy-backend vulkan)"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

const REQUESTED_AUTO: u8 = 0;
const REQUESTED_SCALAR: u8 = 1;
const REQUESTED_RAYON: u8 = 2;
const REQUESTED_SIMD: u8 = 3;
const REQUESTED_CUDA: u8 = 4;
const REQUESTED_VULKAN: u8 = 5;

static REQUESTED: AtomicU8 = AtomicU8::new(REQUESTED_AUTO);

/// Pin the execution backend for the rest of the process. Generated code
/// emits this from the compiler's `--numpy-backend` flag; it can also be
/// called from Rust, or overridden with the `RYPY_NUMPY_BACKEND` env var.
pub fn set_backend(b: Backend) {
    REQUESTED.store(
        match b {
            Backend::Auto => REQUESTED_AUTO,
            Backend::Scalar => REQUESTED_SCALAR,
            Backend::Rayon => REQUESTED_RAYON,
            Backend::Simd => REQUESTED_SIMD,
            Backend::Cuda => REQUESTED_CUDA,
            Backend::Vulkan => REQUESTED_VULKAN,
        },
        Ordering::Relaxed,
    );
}

fn requested_backend() -> Backend {
    match REQUESTED.load(Ordering::Relaxed) {
        REQUESTED_AUTO => Backend::Auto,
        REQUESTED_SCALAR => Backend::Scalar,
        REQUESTED_RAYON => Backend::Rayon,
        REQUESTED_SIMD => Backend::Simd,
        REQUESTED_CUDA => Backend::Cuda,
        REQUESTED_VULKAN => Backend::Vulkan,
        _ => Backend::Auto,
    }
}

/// The backend the next kernel dispatch will use. A requested backend that
/// is not compiled in panics loudly (rather than silently degrading); `Auto`
/// picks the best compiled-in backend.
pub fn active_backend() -> Backend {
    let requested = requested_backend();
    if requested == Backend::Auto {
        return auto_backend();
    }
    if !requested.is_compiled_in() {
        panic!(
            "{}",
            crate::PyException::new(
                "RuntimeError",
                format!(
                    "numpy backend `{}` was requested (set_backend/RYPY_NUMPY_BACKEND) \
                     but stdpython was built without its feature (`{}`); rebuild with \
                     the feature or select a compiled backend",
                    requested.as_str(),
                    requested.requires_feature().unwrap_or("numpy")
                )
            )
        );
    }
    requested
}

fn auto_backend() -> Backend {
    #[cfg(feature = "numpy-cuda")]
    {
        if cuda::available() {
            return Backend::Cuda;
        }
    }
    #[cfg(feature = "numpy-vulkan")]
    {
        if vulkan::available() {
            return Backend::Vulkan;
        }
    }
    #[cfg(feature = "numpy-simd")]
    {
        return Backend::Simd;
    }
    #[cfg(feature = "numpy-rayon")]
    {
        return Backend::Rayon;
    }
    Backend::Scalar
}

/// A one-line description of the resolved backend, for diagnostics.
pub fn backend_summary() -> String {
    let b = active_backend();
    format!("numpy backend: {}", b.as_str())
}

// ---------------------------------------------------------------------------
// Kernel operations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Max,
    Min,
    /// Comparison ops produce a bool array.
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UnOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Sign,
    Square,
    Reciprocal,
    ExpM1,
    Log1P,
    /// Float predicates produce a bool array.
    IsFinite,
    IsInf,
    IsNan,
    /// Logical not on a bool array.
    LogicalNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    /// Mean/std/var are float-typed reductions (numpy: mean of an int array
    /// is float64).
    Mean,
    Std,
    Var,
    All,
    Any,
}

// ---------------------------------------------------------------------------
// Engine modules
// ---------------------------------------------------------------------------

#[path = "scalar.rs"]
pub(crate) mod scalar;

#[cfg(feature = "numpy-rayon")]
pub(crate) mod rayon_eng;
#[cfg(feature = "numpy-simd")]
pub(crate) mod simd;
#[cfg(feature = "numpy-cuda")]
pub(crate) mod cuda;
#[cfg(feature = "numpy-vulkan")]
pub(crate) mod vulkan;

macro_rules! dispatch_binary {
    ($f:ident, $op:expr, $a:expr, $b:expr, $out:expr) => {{
        match active_backend() {
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => cuda::$f($op, $a, $b, $out),
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => vulkan::$f($op, $a, $b, $out),
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => simd::$f($op, $a, $b, $out),
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => rayon_eng::$f($op, $a, $b, $out),
            _ => scalar::$f($op, $a, $b, $out),
        }
    }};
}

macro_rules! dispatch_unary {
    ($f:ident, $op:expr, $a:expr, $out:expr) => {{
        match active_backend() {
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => cuda::$f($op, $a, $out),
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => vulkan::$f($op, $a, $out),
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => simd::$f($op, $a, $out),
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => rayon_eng::$f($op, $a, $out),
            _ => scalar::$f($op, $a, $out),
        }
    }};
}

macro_rules! dispatch_reduce {
    ($f:ident, $op:expr, $a:expr) => {{
        match active_backend() {
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => cuda::$f($op, $a),
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => vulkan::$f($op, $a),
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => simd::$f($op, $a),
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => rayon_eng::$f($op, $a),
            _ => scalar::$f($op, $a),
        }
    }};
}

macro_rules! dispatch_arg {
    ($f:ident, $a:expr) => {{
        match active_backend() {
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => cuda::$f($a),
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => vulkan::$f($a),
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => simd::$f($a),
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => rayon_eng::$f($a),
            _ => scalar::$f($a),
        }
    }};
}

pub(crate) fn binary_f64(op: BinOp, a: &[f64], b: &[f64], out: &mut [f64]) {
    dispatch_binary!(binary_f64, op, a, b, out);
}
pub(crate) fn binary_f32(op: BinOp, a: &[f32], b: &[f32], out: &mut [f32]) {
    dispatch_binary!(binary_f32, op, a, b, out);
}
pub(crate) fn binary_i64(op: BinOp, a: &[i64], b: &[i64], out: &mut [i64]) {
    dispatch_binary!(binary_i64, op, a, b, out);
}
pub(crate) fn binary_i32(op: BinOp, a: &[i32], b: &[i32], out: &mut [i32]) {
    dispatch_binary!(binary_i32, op, a, b, out);
}
pub(crate) fn binary_bool(op: BinOp, a: &[bool], b: &[bool], out: &mut [bool]) {
    dispatch_binary!(binary_bool, op, a, b, out);
}

pub(crate) fn unary_f64(op: UnOp, a: &[f64], out: &mut [f64]) {
    dispatch_unary!(unary_f64, op, a, out);
}
pub(crate) fn unary_f32(op: UnOp, a: &[f32], out: &mut [f32]) {
    dispatch_unary!(unary_f32, op, a, out);
}
pub(crate) fn unary_i64(op: UnOp, a: &[i64], out: &mut [i64]) {
    dispatch_unary!(unary_i64, op, a, out);
}
pub(crate) fn unary_i32(op: UnOp, a: &[i32], out: &mut [i32]) {
    dispatch_unary!(unary_i32, op, a, out);
}
pub(crate) fn unary_bool(op: UnOp, a: &[bool], out: &mut [bool]) {
    dispatch_unary!(unary_bool, op, a, out);
}

pub(crate) fn reduce_f64(op: ReduceOp, a: &[f64]) -> f64 {
    dispatch_reduce!(reduce_f64, op, a)
}
pub(crate) fn reduce_f32(op: ReduceOp, a: &[f32]) -> f32 {
    dispatch_reduce!(reduce_f32, op, a)
}
pub(crate) fn reduce_i64(op: ReduceOp, a: &[i64]) -> i64 {
    dispatch_reduce!(reduce_i64, op, a)
}
pub(crate) fn reduce_i32(op: ReduceOp, a: &[i32]) -> i32 {
    dispatch_reduce!(reduce_i32, op, a)
}
pub(crate) fn reduce_bool(op: ReduceOp, a: &[bool]) -> bool {
    dispatch_reduce!(reduce_bool, op, a)
}

pub(crate) fn argmin_f64(a: &[f64]) -> i64 {
    dispatch_arg!(argmin_f64, a)
}
pub(crate) fn argmax_f64(a: &[f64]) -> i64 {
    dispatch_arg!(argmax_f64, a)
}
pub(crate) fn argmin_f32(a: &[f32]) -> i64 {
    dispatch_arg!(argmin_f32, a)
}
pub(crate) fn argmax_f32(a: &[f32]) -> i64 {
    dispatch_arg!(argmax_f32, a)
}
pub(crate) fn argmin_i64(a: &[i64]) -> i64 {
    dispatch_arg!(argmin_i64, a)
}
pub(crate) fn argmax_i64(a: &[i64]) -> i64 {
    dispatch_arg!(argmax_i64, a)
}
pub(crate) fn argmin_i32(a: &[i32]) -> i64 {
    dispatch_arg!(argmin_i32, a)
}
pub(crate) fn argmax_i32(a: &[i32]) -> i64 {
    dispatch_arg!(argmax_i32, a)
}
pub(crate) fn argmin_bool(a: &[bool]) -> i64 {
    dispatch_arg!(argmin_bool, a)
}
pub(crate) fn argmax_bool(a: &[bool]) -> i64 {
    dispatch_arg!(argmax_bool, a)
}
