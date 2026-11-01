//! Exact integers introduced by oversized Python literals. Small results
//! return to i64 storage without changing their public type.

use alloc::string::{String, ToString};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Neg, Not};
use num_bigint::BigInt;
use num_traits::{FromPrimitive, ToPrimitive};

use crate::{
    PyAbs, PyAdd, PyBool, PyDisplay, PyException, PyFloat, PyFloorDiv, PyHash, PyMod, PyMul,
    PyRepr, PySub, PyToString, Truthy,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyInteger(Repr);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Repr {
    Small(i64),
    Big(BigInt),
}

impl Ord for PyInteger {
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.0, &other.0) {
            (Repr::Small(a), Repr::Small(b)) => a.cmp(b),
            _ => self.big().cmp(&other.big()),
        }
    }
}

impl PartialOrd for PyInteger {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PyInteger {
    fn from_big(value: BigInt) -> Self {
        match value.to_i64() {
            Some(value) => Self(Repr::Small(value)),
            None => Self(Repr::Big(value)),
        }
    }

    fn big(&self) -> BigInt {
        match &self.0 {
            Repr::Small(value) => BigInt::from(*value),
            Repr::Big(value) => value.clone(),
        }
    }

    pub fn from_decimal(value: &str) -> Option<Self> {
        BigInt::parse_bytes(value.as_bytes(), 10).map(Self::from_big)
    }

    /// Decimal conversion under CPython's default safety limit. Arithmetic
    /// remains unbounded; only conversion to a decimal string can fail.
    pub fn checked_decimal(&self) -> Result<String, PyException> {
        let text = match &self.0 {
            Repr::Small(value) => value.to_string(),
            Repr::Big(value) => value.to_string(),
        };
        let digits = text.len() - usize::from(text.starts_with('-'));
        if digits > 4300 {
            return Err(PyException::new(
                "ValueError",
                "Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit",
            ));
        }
        Ok(text)
    }
}

impl From<i64> for PyInteger {
    fn from(value: i64) -> Self {
        Self(Repr::Small(value))
    }
}

impl Hash for PyInteger {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Canonical representation makes equal values hash equally; the
        // Python numeric hash is separately provided by PyHash below.
        self.big().hash(state);
    }
}

impl PartialEq<i64> for PyInteger {
    fn eq(&self, other: &i64) -> bool {
        self == &Self::from(*other)
    }
}
impl PartialOrd<i64> for PyInteger {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        Some(self.cmp(&Self::from(*other)))
    }
}
impl PartialEq<PyInteger> for i64 {
    fn eq(&self, other: &PyInteger) -> bool {
        other == self
    }
}
impl PartialOrd<PyInteger> for i64 {
    fn partial_cmp(&self, other: &PyInteger) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

impl PartialOrd<f64> for PyInteger {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        if other.is_nan() {
            return None;
        }
        if *other == f64::INFINITY {
            return Some(Ordering::Less);
        }
        if *other == f64::NEG_INFINITY {
            return Some(Ordering::Greater);
        }
        let whole = BigInt::from_f64(*other).expect("finite float has an integer part");
        let order = self.big().cmp(&whole);
        Some(
            if order != Ordering::Equal || crate::flt::trunc(*other) == *other {
                order
            } else if *other > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            },
        )
    }
}
impl PartialEq<f64> for PyInteger {
    fn eq(&self, other: &f64) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
impl PartialEq<PyInteger> for f64 {
    fn eq(&self, other: &PyInteger) -> bool {
        other == self
    }
}
impl PartialOrd<PyInteger> for f64 {
    fn partial_cmp(&self, other: &PyInteger) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}
impl PartialEq<bool> for PyInteger {
    fn eq(&self, other: &bool) -> bool {
        self == &(*other as i64)
    }
}
impl PartialOrd<bool> for PyInteger {
    fn partial_cmp(&self, other: &bool) -> Option<Ordering> {
        self.partial_cmp(&(*other as i64))
    }
}
impl PartialEq<PyInteger> for bool {
    fn eq(&self, other: &PyInteger) -> bool {
        other == self
    }
}
impl PartialOrd<PyInteger> for bool {
    fn partial_cmp(&self, other: &PyInteger) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

impl PyHash for PyInteger {
    fn py_hash(&self) -> i64 {
        // Python: sign(n) * (abs(n) % sys.hash_info.modulus), with -1
        // reserved for errors. BigInt's remainder keeps the dividend sign.
        crate::fixup_minus_one(
            (self.big() % BigInt::from(crate::PY_HASH_MODULUS))
                .to_i64()
                .expect("numeric hash fits i64"),
        )
    }
}

impl PyDisplay for PyInteger {
    fn py_display(&self) -> String {
        self.to_string()
    }
}
impl PyRepr for PyInteger {
    fn py_repr(&self) -> String {
        self.to_string()
    }
}
impl PyToString for PyInteger {
    fn py_str(self) -> String {
        self.to_string()
    }
}
impl PyToString for &PyInteger {
    fn py_str(self) -> String {
        self.to_string()
    }
}
impl Truthy for PyInteger {
    fn is_truthy(&self) -> bool {
        self != &Self::from(0)
    }
}
impl PyBool for PyInteger {
    fn py_bool(self) -> bool {
        self.is_truthy()
    }
}
impl PyAbs for &PyInteger {
    type Output = PyInteger;
    fn py_abs(self) -> PyInteger {
        if self < &PyInteger::from(0) {
            -self
        } else {
            self.clone()
        }
    }
}
impl PyFloat for &PyInteger {
    fn py_float(self) -> Result<f64, PyException> {
        self.big()
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| PyException::new("OverflowError", "int too large to convert to float"))
    }
}
impl Neg for &PyInteger {
    type Output = PyInteger;
    fn neg(self) -> PyInteger {
        match self.0 {
            Repr::Small(value) if value != i64::MIN => PyInteger::from(-value),
            _ => PyInteger::from_big(-self.big()),
        }
    }
}
impl Neg for PyInteger {
    type Output = Self;
    fn neg(self) -> Self {
        -&self
    }
}
impl Not for PyInteger {
    type Output = Self;
    fn not(self) -> Self {
        Self::from_big(!self.big())
    }
}

impl fmt::Display for PyInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Rust Display has no Python exception channel. Generated Python
        // display sites use checked_decimal and propagate its typed error.
        let text = self
            .checked_decimal()
            .unwrap_or_else(|error| panic!("{error}"));
        f.write_str(&text)
    }
}

macro_rules! arithmetic {
    ($trait:ident, $method:ident, $checked:ident, $op:tt) => {
        impl $trait<PyInteger> for PyInteger {
            type Output = Self;
            fn $method(&self, rhs: &Self) -> Self {
                if let (Repr::Small(a), Repr::Small(b)) = (&self.0, &rhs.0) {
                    if let Some(value) = a.$checked(*b) { return Self::from(value); }
                }
                Self::from_big(self.big() $op rhs.big())
            }
        }
    };
}
arithmetic!(PyAdd, py_add, checked_add, +);
arithmetic!(PySub, py_sub, checked_sub, -);
arithmetic!(PyMul, py_mul, checked_mul, *);

// Fixed-width Python int operands keep their exact value when paired with
// an unbounded integer. The bigint side always owns the result type.
macro_rules! mixed_integer {
    ($trait:ident, $method:ident, $output:ty) => {
        impl $trait<i64> for PyInteger {
            type Output = $output;
            fn $method(&self, rhs: &i64) -> $output {
                self.$method(&PyInteger::from(*rhs))
            }
        }
        impl $trait<PyInteger> for i64 {
            type Output = $output;
            fn $method(&self, rhs: &PyInteger) -> $output {
                PyInteger::from(*self).$method(rhs)
            }
        }
    };
}
mixed_integer!(PyAdd, py_add, PyInteger);
mixed_integer!(PySub, py_sub, PyInteger);
mixed_integer!(PyMul, py_mul, PyInteger);

impl PyFloorDiv<PyInteger> for PyInteger {
    type Output = Self;
    fn py_floordiv(&self, rhs: &Self) -> Result<Self, PyException> {
        let a = self.big();
        let b = rhs.big();
        if b == BigInt::from(0) {
            return Err(PyException::new(
                "ZeroDivisionError",
                "integer division or modulo by zero",
            ));
        }
        let mut q = &a / &b;
        let r = &a % &b;
        if r != BigInt::from(0) && (a < BigInt::from(0)) != (b < BigInt::from(0)) {
            q -= 1;
        }
        Ok(Self::from_big(q))
    }
}

impl PyMod<PyInteger> for PyInteger {
    type Output = Self;
    fn py_mod(&self, rhs: &Self) -> Result<Self, PyException> {
        let a = self.big();
        let b = rhs.big();
        if b == BigInt::from(0) {
            return Err(PyException::new(
                "ZeroDivisionError",
                "integer modulo by zero",
            ));
        }
        let mut r = &a % &b;
        if r != BigInt::from(0) && (r < BigInt::from(0)) != (b < BigInt::from(0)) {
            r += b;
        }
        Ok(Self::from_big(r))
    }
}

macro_rules! mixed_division {
    ($trait:ident, $method:ident) => {
        impl $trait<i64> for PyInteger {
            type Output = PyInteger;
            fn $method(&self, rhs: &i64) -> Result<PyInteger, PyException> {
                self.$method(&PyInteger::from(*rhs))
            }
        }
        impl $trait<PyInteger> for i64 {
            type Output = PyInteger;
            fn $method(&self, rhs: &PyInteger) -> Result<PyInteger, PyException> {
                PyInteger::from(*self).$method(rhs)
            }
        }
    };
}
mixed_division!(PyFloorDiv, py_floordiv);
mixed_division!(PyMod, py_mod);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn promotes_in_alloc_tier() {
        // Python: (2**63 - 1) + 1 == 9223372036854775808
        let value = PyInteger::from(i64::MAX).py_add(&PyInteger::from(1));
        assert_eq!(value.to_string(), "9223372036854775808");
        assert_eq!(value.py_sub(&PyInteger::from(1)), PyInteger::from(i64::MAX));
    }
}
