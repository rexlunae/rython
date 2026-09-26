//! Python bisect module: binary search and sorted insertion on a
//! sequence already in order, ported from CPython's `_bisect` loop so the
//! positions — and the errors — are CPython's: `lo < 0` is its
//! `ValueError: lo must be non-negative`, and a `hi` past the end reaches
//! the same `IndexError` its indexing does. The `key=` keyword (3.10) is
//! not modeled; the converter refuses it.

use crate::PyException;
use alloc::vec::Vec;

fn bounds(lo: i64, hi: Option<i64>, len: usize) -> Result<(i64, i64), PyException> {
    if lo < 0 {
        return Err(crate::value_error("lo must be non-negative"));
    }
    Ok((lo, hi.unwrap_or(len as i64)))
}

fn at<T>(a: &[T], i: i64) -> Result<&T, PyException> {
    a.get(i as usize)
        .ok_or_else(|| PyException::new("IndexError", "list index out of range"))
}

/// A sequence bisect searches for a probe of type `X`: a typed sequence
/// of ordered members (`X` is its member type), or a BOXED sequence
/// (idna's `uts46data`, a tuple of boxed rows), whose members compare with
/// CPython's boxed `<` ([`crate::PyValue::py_lt_boxed`]) against the probe
/// boxed — a TypeError between unorderable members propagates as
/// CPython's does. A position past the end is the IndexError CPython's
/// indexing raises.
pub trait BisectSeq<X: ?Sized> {
    fn bisect_len(&self) -> Result<usize, PyException>;
    /// `a[i] < x`
    fn item_lt(&self, i: i64, x: &X) -> Result<bool, PyException>;
    /// `x < a[i]`
    fn lt_item(&self, x: &X, i: i64) -> Result<bool, PyException>;
}

impl<T: PartialOrd> BisectSeq<T> for [T] {
    fn bisect_len(&self) -> Result<usize, PyException> {
        Ok(self.len())
    }
    fn item_lt(&self, i: i64, x: &T) -> Result<bool, PyException> {
        Ok(at(self, i)? < x)
    }
    fn lt_item(&self, x: &T, i: i64) -> Result<bool, PyException> {
        Ok(x < at(self, i)?)
    }
}

impl<T: PartialOrd> BisectSeq<T> for Vec<T> {
    fn bisect_len(&self) -> Result<usize, PyException> {
        Ok(self.len())
    }
    fn item_lt(&self, i: i64, x: &T) -> Result<bool, PyException> {
        self.as_slice().item_lt(i, x)
    }
    fn lt_item(&self, x: &T, i: i64) -> Result<bool, PyException> {
        self.as_slice().lt_item(x, i)
    }
}

impl<T: PartialOrd> BisectSeq<T> for crate::PyTuple<T> {
    fn bisect_len(&self) -> Result<usize, PyException> {
        Ok(self.len())
    }
    fn item_lt(&self, i: i64, x: &T) -> Result<bool, PyException> {
        self.as_slice().item_lt(i, x)
    }
    fn lt_item(&self, x: &T, i: i64) -> Result<bool, PyException> {
        self.as_slice().lt_item(x, i)
    }
}

fn boxed_bisect_items<'a, S: BoxedSeq + ?Sized>(
    a: &'a S,
) -> Result<&'a [crate::PyValue], PyException> {
    a.boxed_items()
}

macro_rules! boxed_bisect_seq {
    ($($s:ty),*) => {
        $(impl<X: Clone> BisectSeq<X> for $s
        where
            crate::PyValue: From<X>,
        {
            fn bisect_len(&self) -> Result<usize, PyException> {
                Ok(boxed_bisect_items(self)?.len())
            }
            fn item_lt(&self, i: i64, x: &X) -> Result<bool, PyException> {
                at(boxed_bisect_items(self)?, i)?.py_lt_boxed(&crate::PyValue::from(x.clone()))
            }
            fn lt_item(&self, x: &X, i: i64) -> Result<bool, PyException> {
                crate::PyValue::from(x.clone()).py_lt_boxed(at(boxed_bisect_items(self)?, i)?)
            }
        })*
    };
}
boxed_bisect_seq!(crate::PyValue, Vec<crate::PyValue>, crate::PyTuple<crate::PyValue>);

/// bisect.bisect_left(a, x, lo=0, hi=len(a)): the first position where
/// `x` could be inserted keeping `a` sorted (before any equal items).
pub fn bisect_left<X: ?Sized, S: BisectSeq<X> + ?Sized>(
    a: &S,
    x: &X,
    lo: i64,
    hi: Option<i64>,
) -> Result<i64, PyException> {
    let (mut lo, mut hi) = bounds(lo, hi, a.bisect_len()?)?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a.item_lt(mid, x)? {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// bisect.bisect_right(a, x, lo=0, hi=len(a)) — also `bisect.bisect`:
/// the position after any items equal to `x`.
pub fn bisect_right<X: ?Sized, S: BisectSeq<X> + ?Sized>(
    a: &S,
    x: &X,
    lo: i64,
    hi: Option<i64>,
) -> Result<i64, PyException> {
    let (mut lo, mut hi) = bounds(lo, hi, a.bisect_len()?)?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a.lt_item(x, mid)? {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

/// bisect.insort_left(a, x, lo=0, hi=len(a)): insert `x` before any equal
/// items.
pub fn insort_left<T: PartialOrd>(
    a: &mut Vec<T>,
    x: T,
    lo: i64,
    hi: Option<i64>,
) -> Result<(), PyException> {
    let i = bisect_left(a, &x, lo, hi)?;
    a.insert(i as usize, x);
    Ok(())
}

/// bisect.insort_right(a, x, lo=0, hi=len(a)) — also `bisect.insort`:
/// insert `x` after any equal items.
pub fn insort_right<T: PartialOrd>(
    a: &mut Vec<T>,
    x: T,
    lo: i64,
    hi: Option<i64>,
) -> Result<(), PyException> {
    let i = bisect_right(a, &x, lo, hi)?;
    a.insert(i as usize, x);
    Ok(())
}

/// bisect.bisect — the same function as bisect_right.
pub use bisect_right as bisect;
/// bisect.insort — the same function as insort_right.
pub use insort_right as insort;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn positions_and_errors_match_cpython() {
        let a = vec![1i64, 2, 2, 4];
        // bisect_left(a, 2) == 1; bisect_right(a, 2) == 3
        assert_eq!(bisect_left(&a, &2, 0, None).unwrap(), 1);
        assert_eq!(bisect_right(&a, &2, 0, None).unwrap(), 3);
        // bisect_left(a, 2, 2) == 2; bisect_left(a, 2, 0, 1) == 1
        assert_eq!(bisect_left(&a, &2, 2, None).unwrap(), 2);
        assert_eq!(bisect_left(&a, &2, 0, Some(1)).unwrap(), 1);
        // lo=-1: ValueError: lo must be non-negative
        assert_eq!(bisect_left(&a, &2, -1, None).unwrap_err().message, "lo must be non-negative");
        // hi past the end: IndexError: list index out of range
        assert_eq!(
            bisect_left(&a, &2, 0, Some(100)).unwrap_err().message,
            "list index out of range"
        );
        let mut b = a.clone();
        insort_right(&mut b, 3, 0, None).unwrap();
        assert_eq!(b, vec![1, 2, 2, 3, 4]);
    }
}

/// A sequence of BOXED members a bisect searches: a boxed list or tuple
/// value, or a typed sequence of boxed members; a boxed non-sequence is
/// CPython's TypeError.
pub trait BoxedSeq {
    fn boxed_items(&self) -> Result<&[crate::PyValue], PyException>;
}
impl BoxedSeq for crate::PyValue {
    fn boxed_items(&self) -> Result<&[crate::PyValue], PyException> {
        self.as_tuple().map(Vec::as_slice).ok_or_else(|| {
            PyException::new(
                "TypeError",
                alloc::format!("object of type '{}' has no len()", self.py_type_name()),
            )
        })
    }
}
impl BoxedSeq for [crate::PyValue] {
    fn boxed_items(&self) -> Result<&[crate::PyValue], PyException> {
        Ok(self)
    }
}
impl BoxedSeq for Vec<crate::PyValue> {
    fn boxed_items(&self) -> Result<&[crate::PyValue], PyException> {
        Ok(self)
    }
}
impl BoxedSeq for crate::PyTuple<crate::PyValue> {
    fn boxed_items(&self) -> Result<&[crate::PyValue], PyException> {
        Ok(self)
    }
}

