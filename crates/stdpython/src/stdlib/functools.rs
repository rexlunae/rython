//! Python functools module implementation
//!
//! reduce() for now; partial/lru_cache need decorator and closure-capture
//! support in the compiler and are tracked separately.

use crate::PyException;

/// functools.reduce(function, iterable): left fold; an empty iterable
/// raises TypeError with Python's message.
pub fn reduce<T, F>(mut function: F, iterable: &[T]) -> Result<T, PyException>
where
    T: Clone,
    F: FnMut(T, T) -> T,
{
    let mut iter = iterable.iter();
    let mut acc = iter
        .next()
        .ok_or_else(|| {
            PyException::new(
                "TypeError",
                "reduce() of empty iterable with no initial value",
            )
        })?
        .clone();
    for x in iter {
        acc = function(acc, x.clone());
    }
    Ok(acc)
}

/// functools.reduce(function, iterable, initial): the accumulator type
/// may differ from the element type, as in Python.
pub fn reduce_initial<T, U, F>(mut function: F, iterable: &[T], initial: U) -> U
where
    T: Clone,
    F: FnMut(U, T) -> U,
{
    let mut acc = initial;
    for x in iterable {
        acc = function(acc, x.clone());
    }
    acc
}
