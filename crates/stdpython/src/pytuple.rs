//! A Python tuple whose length is not fixed statically (issue #399):
//! `Tuple[int, ...]`, `tuple(xs)` of a list, the mixed-arity tuple values
//! of one dict literal. A fixed-shape tuple is a Rust tuple; this one is a
//! sequence, so it wraps a `Vec` — and it PRINTS as a tuple (`(3, 4)`,
//! `(3,)`, `()`), where the bare `Vec` it used to be printed as a list.
//!
//! It dereferences to the `Vec` for every read (indexing, `len`,
//! iteration, slicing, methods); tuples are immutable, so there is no
//! mutable view.

use crate::{
    Len, PyAdd, PyContains, PyListFrom, PySum, PyDisplay, PyException, PyIndex, PyMul, PyRepr, PySlice, PyToString,
    Truthy,
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A variable-length Python tuple of `T`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PyTuple<T>(pub Vec<T>);

impl<T> PyTuple<T> {
    /// A tuple of `items`, in order.
    pub fn new(items: Vec<T>) -> Self {
        PyTuple(items)
    }

    /// The members as a list (`list(t)`).
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.clone()
    }
}

impl<T> core::ops::Deref for PyTuple<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T> From<Vec<T>> for PyTuple<T> {
    fn from(v: Vec<T>) -> Self {
        PyTuple(v)
    }
}

impl<T> From<PyTuple<T>> for Vec<T> {
    fn from(t: PyTuple<T>) -> Self {
        t.0
    }
}

impl<T> FromIterator<T> for PyTuple<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        PyTuple(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for PyTuple<T> {
    type Item = T;
    type IntoIter = alloc::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PyTuple<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// `repr(t)`: `(3, 4)`, a one-tuple `(3,)`, the empty tuple `()`.
impl<T: PyRepr> PyRepr for PyTuple<T> {
    fn py_repr(&self) -> String {
        let items: Vec<String> = self.0.iter().map(|x| x.py_repr()).collect();
        if items.len() == 1 {
            format!("({},)", items[0])
        } else {
            format!("({})", items.join(", "))
        }
    }
}

/// `str(t)` is `repr(t)`, as for every tuple.
impl<T: PyRepr> PyDisplay for PyTuple<T> {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

impl<T: PyRepr> core::fmt::Display for PyTuple<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.py_repr())
    }
}

impl<T: PyRepr> PyToString for PyTuple<T> {
    fn py_str(self) -> String {
        self.py_display()
    }
}

impl<T: PyRepr> PyToString for &PyTuple<T> {
    fn py_str(self) -> String {
        self.py_display()
    }
}

impl<T> Len for PyTuple<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Truthy for PyTuple<T> {
    fn is_truthy(&self) -> bool {
        !self.0.is_empty()
    }
}

/// `t[i]`, with CPython's tuple error text.
impl<T: Clone> PyIndex<i64> for PyTuple<T> {
    type Output = T;
    fn py_index(&self, index: i64) -> Result<T, PyException> {
        crate::normalize_index(index, self.0.len())
            .map(|i| self.0[i].clone())
            .ok_or_else(|| PyException::new("IndexError", "tuple index out of range"))
    }
}

/// `t[a:b:c]` is a tuple.
impl<T: Clone> PySlice for PyTuple<T> {
    type Output = PyTuple<T>;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> PyTuple<T> {
        PyTuple(crate::slice(&self.0, start, stop, step))
    }
}

/// `x in t` answers exactly as for the members' list.
impl<U: ?Sized, T> PyContains<U> for PyTuple<T>
where
    Vec<T>: PyContains<U>,
{
    fn py_contains(&self, item: &U) -> bool {
        self.0.py_contains(item)
    }
}

/// `sum(t)` sums the members, as for a list.
impl<T> PySum for PyTuple<T>
where
    Vec<T>: PySum,
{
    type Output = <Vec<T> as PySum>::Output;
    fn py_sum(self) -> Self::Output {
        self.0.py_sum()
    }
}

impl<'a, T> PySum for &'a PyTuple<T>
where
    &'a Vec<T>: PySum,
{
    type Output = <&'a Vec<T> as PySum>::Output;
    fn py_sum(self) -> Self::Output {
        (&self.0).py_sum()
    }
}

/// `list(t)` is the members as a list.
impl<T> PyListFrom for PyTuple<T> {
    type Item = T;
    fn py_list(self) -> Vec<T> {
        self.0
    }
}

/// `t + u` concatenates tuples.
impl<T: Clone> PyAdd<PyTuple<T>> for PyTuple<T> {
    type Output = PyTuple<T>;
    fn py_add(&self, rhs: &PyTuple<T>) -> PyTuple<T> {
        let mut out = self.0.clone();
        out.extend_from_slice(&rhs.0);
        PyTuple(out)
    }
}

/// `t * n` repeats; a non-positive count is the empty tuple.
impl<T: Clone> PyMul<i64> for PyTuple<T> {
    type Output = PyTuple<T>;
    fn py_mul(&self, rhs: &i64) -> PyTuple<T> {
        let n = (*rhs).max(0) as usize;
        let mut out = Vec::with_capacity(self.0.len() * n);
        for _ in 0..n {
            out.extend_from_slice(&self.0);
        }
        PyTuple(out)
    }
}

/// A tuple boxes as the boxed tuple of its boxed members.
impl<T> From<PyTuple<T>> for crate::PyValue
where
    crate::PyValue: From<T>,
{
    fn from(t: PyTuple<T>) -> Self {
        crate::PyValue::Tuple(alloc::sync::Arc::new(
            t.0.into_iter().map(crate::PyValue::from).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn prints_as_a_tuple() {
        // repr((3, 4)) == '(3, 4)'; repr((3,)) == '(3,)'; repr(()) == '()'
        assert_eq!(PyTuple(vec![3i64, 4]).py_display(), "(3, 4)");
        assert_eq!(PyTuple(vec![3i64]).py_display(), "(3,)");
        assert_eq!(PyTuple::<i64>(vec![]).py_display(), "()");
        // repr(('b', 'c')) == "('b', 'c')"
        assert_eq!(PyTuple(vec!["b".to_string(), "c".to_string()]).py_repr(), "('b', 'c')");
        // (1, 2)[5] -> IndexError: tuple index out of range
        assert_eq!(
            PyTuple(vec![1i64, 2]).py_index(5).unwrap_err().message,
            "tuple index out of range"
        );
        // (1, 2, 3)[1:] == (2, 3)
        assert_eq!(PyTuple(vec![1i64, 2, 3]).py_slice(Some(1), None, None).py_display(), "(2, 3)");
    }
}
