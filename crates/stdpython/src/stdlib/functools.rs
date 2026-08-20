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

/// A float cache key with PYTHON's float semantics: `-0.0 == 0.0` (so they
/// share a cache entry) and `NaN != NaN` (so a NaN key never hits, and the
/// wrapped function is called every time — exactly CPython, where a dict
/// lookup on NaN misses). Hash normalizes -0.0 to +0.0 so equal keys hash
/// equal; Eq uses float `==` so NaN never matches, not even itself.
#[derive(Clone, Copy, Debug)]
pub struct PyFloatKey(pub f64);

impl PartialEq for PyFloatKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for PyFloatKey {}

impl core::hash::Hash for PyFloatKey {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        let bits = if self.0 == 0.0 { 0.0 } else { self.0 }.to_bits();
        bits.hash(state);
    }
}

impl core::fmt::Display for PyFloatKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.0, f)
    }
}

/// Whether a type can serve as an lru_cache key; the compiler emits the
/// concrete key tuple from the parameter annotations, so this only exists
/// to document the supported set (the codegen enforces it loudly).
pub trait PyCacheKey: core::hash::Hash + Eq + Clone {}
impl<T: core::hash::Hash + Eq + Clone> PyCacheKey for T {}

#[cfg(test)]
mod tests {
    use super::PyFloatKey;

    #[test]
    fn float_key_python_semantics() {
        // -0.0 and 0.0 compare equal in Python, so they share a cache key.
        assert_eq!(PyFloatKey(-0.0), PyFloatKey(0.0));
        // NaN never equals itself: a NaN key must never hit.
        assert_ne!(PyFloatKey(f64::NAN), PyFloatKey(f64::NAN));
    }

    #[test]
    fn float_key_lru_cache_roundtrip() {
        let mut cache = super::PyLruCache::new(None);
        cache.put(PyFloatKey(0.5), 10i64);
        assert_eq!(cache.get(&PyFloatKey(0.5)), Some(10));
        // 0.0 and -0.0 share the entry.
        cache.put(PyFloatKey(0.0), 20);
        assert_eq!(cache.get(&PyFloatKey(-0.0)), Some(20));
        // A NaN key never hits: get returns None.
        assert_eq!(cache.get(&PyFloatKey(f64::NAN)), None);
    }
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
