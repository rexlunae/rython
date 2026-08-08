//! NumPy dtype model.
//!
//! Rython's numpy subset supports the five dtypes that cover nearly all
//! real-world array code: `float64`, `float32`, `int64`, `int32`, and
//! `bool`. Dtypes are plain values (a small enum), so `np.float64` lowers to
//! a `Dtype` constant and `a.astype(np.float32)` is a plain function call.

use crate::PyException;

/// The supported array element types, in numpy promotion order (bool is
/// lowest, float64 highest).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dtype {
    Bool,
    Int32,
    Int64,
    Float32,
    Float64,
}

impl Dtype {
    /// numpy's dtype name (`np.dtype('float64').name`).
    pub fn name(&self) -> &'static str {
        match self {
            Dtype::Bool => "bool",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
        }
    }

    /// Whether the type is an integer (bool counts as integer for numpy's
    /// "is integer" promotion checks, but NOT for scalar result typing:
    /// `np.sum` of a bool array is an int, `np.sum` of an int array is an
    /// int, and only float arrays reduce to floats).
    pub fn is_integer(self) -> bool {
        matches!(self, Dtype::Bool | Dtype::Int32 | Dtype::Int64)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Dtype::Float32 | Dtype::Float64)
    }

    /// numpy's `np.result_type` promotion for two array dtypes. This covers
    /// the supported subset and follows numpy's documented value-based
    /// promotion rules for the same-kind pairs.
    pub fn promote(self, other: Dtype) -> Dtype {
        use Dtype::*;
        match (self, other) {
            (Bool, Bool) => Bool,
            (Bool, t) | (t, Bool) => t,
            (Int32, Int32) => Int32,
            (Int64, Int64) => Int64,
            (Float32, Float32) => Float32,
            (Float64, Float64) => Float64,
            (Int32, Int64) | (Int64, Int32) => Int64,
            (Int32, Float32) | (Float32, Int32) => Float32,
            (Int32, Float64) | (Float64, Int32) => Float64,
            (Int64, Float32) | (Float32, Int64) => Float64,
            (Int64, Float64) | (Float64, Int64) => Float64,
            (Float32, Float64) | (Float64, Float32) => Float64,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// numpy dtype-string parsing for the supported subset
/// (`np.dtype("float64")` accepts the same spellings numpy does).
pub fn dtype_from_str(s: &str) -> Result<Dtype, PyException> {
    Ok(match s {
        "bool" | "bool_" | "?" => Dtype::Bool,
        "int32" | "i4" | "i" => Dtype::Int32,
        "int64" | "i8" | "l" | "int" | "long" => Dtype::Int64,
        "float32" | "f4" | "single" => Dtype::Float32,
        "float64" | "f8" | "float" | "double" => Dtype::Float64,
        other => {
            return Err(PyException::new(
                "TypeError",
                format!(
                    "data type '{}' not understood (supported: bool, int32, int64, \
                     float32, float64)",
                    other
                ),
            ))
        }
    })
}
