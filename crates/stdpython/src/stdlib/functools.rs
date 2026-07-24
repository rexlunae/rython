//! Python functools module implementation
//!
//! reduce() and the lru_cache backing store. partial has no runtime
//! symbol: the compiler lowers partial(f, ...) to a closure directly.

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

/// The store behind @functools.lru_cache: an insertion-ordered map with
/// LRU touch-on-hit and bounded eviction, exactly CPython's cache
/// discipline. The compiler wraps a decorated function's body with a
/// static of this keyed on the argument tuple.
pub struct PyLruCache<K, V> {
    map: crate::PyDict<K, V>,
    maxsize: Option<usize>,
}

impl<K: core::hash::Hash + Eq + Clone, V: Clone> PyLruCache<K, V> {
    /// maxsize None is unbounded (functools.cache); Python's default
    /// for bare @lru_cache is Some(128).
    pub fn new(maxsize: Option<usize>) -> Self {
        Self {
            map: crate::PyDict::default(),
            maxsize,
        }
    }

    /// A hit moves the entry to most-recently-used, as CPython's does.
    pub fn get(&mut self, key: &K) -> Option<V> {
        let value = self.map.shift_remove(key)?;
        self.map.insert(key.clone(), value.clone());
        Some(value)
    }

    pub fn put(&mut self, key: K, value: V) {
        self.map.insert(key, value);
        if let Some(maxsize) = self.maxsize {
            if self.map.len() > maxsize {
                // The FRONT is least-recently-used.
                self.map.shift_remove_index(0);
            }
        }
    }
}
