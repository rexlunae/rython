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

/// bisect.bisect_left(a, x, lo=0, hi=len(a)): the first position where
/// `x` could be inserted keeping `a` sorted (before any equal items).
pub fn bisect_left<T: PartialOrd>(
    a: &[T],
    x: &T,
    lo: i64,
    hi: Option<i64>,
) -> Result<i64, PyException> {
    let (mut lo, mut hi) = bounds(lo, hi, a.len())?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if at(a, mid)? < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// bisect.bisect_right(a, x, lo=0, hi=len(a)) — also `bisect.bisect`:
/// the position after any items equal to `x`.
pub fn bisect_right<T: PartialOrd>(
    a: &[T],
    x: &T,
    lo: i64,
    hi: Option<i64>,
) -> Result<i64, PyException> {
    let (mut lo, mut hi) = bounds(lo, hi, a.len())?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if x < at(a, mid)? {
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
