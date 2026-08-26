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
//! | `numpy-rayon`    | rayon   | multithreaded elementwise kernels       |
//! | `numpy-simd`     | simd    | LLVM-auto-vectorized kernels (alias of scalar; no hand-written intrinsics yet, issue #164) |
//! | `numpy-cuda`     | cuda    | FEATURE COMPILES BUT NO KERNELS SHIP — selecting it is a loud runtime error (issue #164) |
//! | `numpy-vulkan`   | vulkan  | FEATURE COMPILES BUT NO KERNELS SHIP — selecting it is a loud runtime error (issue #164) |
//!
//! Reductions (`sum`, `mean`, ...) are not engine-dispatched; they run
//! their own sequential loops in `reduce.rs`.

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
    // Ranked by measured performance (issue #164): hardware backends that
    // are actually present outrank the CPU kernels; among the software
    // kernels, multithreaded rayon beats the single-threaded simd/scalar
    // loops at every benchmarked size, so it outranks simd.
    let software: Backend = if cfg!(feature = "numpy-rayon") {
        Backend::Rayon
    } else if cfg!(feature = "numpy-simd") {
        Backend::Simd
    } else {
        Backend::Scalar
    };
    #[cfg(feature = "numpy-vulkan")]
    if vulkan::available() {
        return Backend::Vulkan;
    }
    #[cfg(feature = "numpy-cuda")]
    if cuda::available() {
        return Backend::Cuda;
    }
    software
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

// ---------------------------------------------------------------------------
// Engine modules
// ---------------------------------------------------------------------------

#[path = "scalar.rs"]
pub(crate) mod scalar;

#[cfg(feature = "numpy-rayon")]
#[path = "rayon_eng.rs"]
pub(crate) mod rayon_eng;
#[cfg(feature = "numpy-simd")]
#[path = "simd.rs"]
pub(crate) mod simd;
#[cfg(feature = "numpy-cuda")]
#[path = "cuda.rs"]
pub(crate) mod cuda;
#[cfg(feature = "numpy-vulkan")]
#[path = "vulkan.rs"]
pub(crate) mod vulkan;

macro_rules! dispatch_binary {
    ($f:ident, $op:expr, $a:expr, $b:expr) => {{
        match active_backend() {
            #[cfg(feature = "numpy-cuda")]
            Backend::Cuda => cuda::$f($op, $a, $b),
            #[cfg(feature = "numpy-vulkan")]
            Backend::Vulkan => vulkan::$f($op, $a, $b),
            #[cfg(feature = "numpy-simd")]
            Backend::Simd => simd::$f($op, $a, $b),
            #[cfg(feature = "numpy-rayon")]
            Backend::Rayon => rayon_eng::$f($op, $a, $b),
            _ => scalar::$f($op, $a, $b),
        }
    }};
}

macro_rules! dispatch_unary {
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

/// Every kernel takes the input slices and RETURNS a freshly allocated
/// output vector (never written into a caller buffer): rython arrays are
/// values, every call site builds a new array, and returning a `Vec` lets
/// the kernels grow it directly — no zero-fill pass over the output
/// (a full `vec![0.0; n]` write is pure waste when the kernel overwrites
/// every element anyway). It also makes aliasing structurally impossible.
pub(crate) fn binary_f64(op: BinOp, a: &[f64], b: &[f64]) -> Vec<f64> {
    dispatch_binary!(binary_f64, op, a, b)
}
pub(crate) fn binary_f32(op: BinOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    dispatch_binary!(binary_f32, op, a, b)
}
pub(crate) fn binary_i64(op: BinOp, a: &[i64], b: &[i64]) -> Vec<i64> {
    dispatch_binary!(binary_i64, op, a, b)
}
pub(crate) fn binary_i32(op: BinOp, a: &[i32], b: &[i32]) -> Vec<i32> {
    dispatch_binary!(binary_i32, op, a, b)
}
pub(crate) fn binary_bool(op: BinOp, a: &[bool], b: &[bool]) -> Vec<bool> {
    dispatch_binary!(binary_bool, op, a, b)
}

pub(crate) fn unary_f64(op: UnOp, a: &[f64]) -> Vec<f64> {
    dispatch_unary!(unary_f64, op, a)
}
pub(crate) fn unary_f32(op: UnOp, a: &[f32]) -> Vec<f32> {
    dispatch_unary!(unary_f32, op, a)
}
pub(crate) fn unary_i64(op: UnOp, a: &[i64]) -> Vec<i64> {
    dispatch_unary!(unary_i64, op, a)
}
pub(crate) fn unary_i32(op: UnOp, a: &[i32]) -> Vec<i32> {
    dispatch_unary!(unary_i32, op, a)
}
pub(crate) fn unary_bool(op: UnOp, a: &[bool]) -> Vec<bool> {
    dispatch_unary!(unary_bool, op, a)
}
