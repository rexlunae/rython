//! The values a host passes to and receives from compiled functions in M1.

use crate::error::JitError;
use crate::manifest::MarshalType;

/// A value that crosses the boundary: `None`, `bool`, `int` (64-bit),
/// `float` or `str`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    /// The value's M1 type.
    pub fn marshal_type(&self) -> MarshalType {
        match self {
            Value::None => MarshalType::None,
            Value::Bool(_) => MarshalType::Bool,
            Value::Int(_) => MarshalType::Int,
            Value::Float(_) => MarshalType::Float,
            Value::Str(_) => MarshalType::Str,
        }
    }

    /// `type(value).__name__`.
    pub fn type_name(&self) -> &'static str {
        self.marshal_type().type_name()
    }

    /// `repr(value)`, by the runtime's own `repr` (the same code the
    /// compiled library runs), so `1e16` is `1e+16` and strings pick
    /// their quotes as CPython does.
    pub fn repr(&self) -> String {
        use stdpython::PyRepr;
        match self {
            Value::None => "None".to_string(),
            Value::Bool(b) => b.py_repr(),
            Value::Int(n) => n.py_repr(),
            Value::Float(x) => x.py_repr(),
            Value::Str(s) => s.py_repr(),
        }
    }
}

impl From<()> for Value {
    fn from(_: ()) -> Value {
        Value::None
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Value {
        Value::Int(n)
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Float(x)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}

/// A Rust type a call's result converts into: `()` for `None`, `bool`,
/// `i64`, `f64`, `String`, or [`Value`] itself. A result of another type
/// is a [`JitError::Marshal`] naming both.
pub trait FromValue: Sized {
    fn from_value(value: Value) -> Result<Self, JitError>;
}

fn mismatch(wanted: &str, value: &Value) -> JitError {
    JitError::Marshal(format!(
        "the call returned {} `{}`, which does not convert to `{}`",
        article(value.type_name()),
        value.type_name(),
        wanted
    ))
}

/// "a"/"an" before a type name, for messages.
pub(crate) fn article(type_name: &str) -> &'static str {
    match type_name.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

impl FromValue for Value {
    fn from_value(value: Value) -> Result<Value, JitError> {
        Ok(value)
    }
}

impl FromValue for () {
    fn from_value(value: Value) -> Result<(), JitError> {
        match value {
            Value::None => Ok(()),
            other => Err(mismatch("()", &other)),
        }
    }
}

impl FromValue for bool {
    fn from_value(value: Value) -> Result<bool, JitError> {
        match value {
            Value::Bool(b) => Ok(b),
            other => Err(mismatch("bool", &other)),
        }
    }
}

impl FromValue for i64 {
    fn from_value(value: Value) -> Result<i64, JitError> {
        match value {
            Value::Int(n) => Ok(n),
            other => Err(mismatch("i64", &other)),
        }
    }
}

impl FromValue for f64 {
    fn from_value(value: Value) -> Result<f64, JitError> {
        match value {
            Value::Float(x) => Ok(x),
            other => Err(mismatch("f64", &other)),
        }
    }
}

impl FromValue for String {
    fn from_value(value: Value) -> Result<String, JitError> {
        match value {
            Value::Str(s) => Ok(s),
            other => Err(mismatch("String", &other)),
        }
    }
}
