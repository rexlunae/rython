//! Python collections module implementation
//! 
//! This module provides specialized container datatypes.
//! Implementation matches Python's collections module API.

use crate::{PyException, Len, Truthy, PyRepr, python_function};
use alloc::collections::VecDeque;
use alloc::{format, string::{String, ToString}, vec, vec::Vec};
use core::hash::Hash;
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};
#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};

/// Counter - dict subclass for counting hashable objects
#[derive(Debug, Clone)]
pub struct Counter<T> 
where 
    T: Hash + Eq + Clone + core::fmt::Debug,
{
    counts: crate::PyDict<T, i64>,
}

impl<T> Counter<T> 
where 
    T: Hash + Eq + Clone + core::fmt::Debug,
{
    /// Create a new Counter
    pub fn new() -> Self {
        Self {
            counts: crate::PyDict::default(),
        }
    }
    
    /// Create Counter from iterable
    pub fn from_iter<I>(iterable: I) -> Self 
    where 
        I: IntoIterator<Item = T>,
    {
        let mut counter = Self::new();
        for item in iterable {
            counter.update_one(&item, 1);
        }
        counter
    }
    
    /// Update counts with elements from iterable
    pub fn update<I>(&mut self, iterable: I) 
    where 
        I: IntoIterator<Item = T>,
    {
        for item in iterable {
            self.update_one(&item, 1);
        }
    }
    
    /// Update count for single element
    pub fn update_one(&mut self, element: &T, count: i64) {
        // Python's Counter keeps zero and negative counts after update() and
        // subtract(); only the +/- operators drop them.
        *self.counts.entry(element.clone()).or_insert(0) += count;
    }
    
    /// Get count for element
    pub fn get(&self, element: &T) -> i64 {
        self.counts.get(element).copied().unwrap_or(0)
    }
    
    /// Get most common elements
    pub fn most_common(&self, n: Option<usize>) -> Vec<(T, i64)> {
        let mut items: Vec<(T, i64)> = self.counts.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        // Stable sort by count only: Python breaks ties by FIRST-INSERTION
        // order (the old Debug-string comparison invented an ordering).
        items.sort_by(|a, b| b.1.cmp(&a.1));
        
        match n {
            Some(limit) => items.into_iter().take(limit).collect(),
            None => items,
        }
    }
    
    /// Get all elements (with repetitions)
    pub fn elements(&self) -> Vec<T> {
        let mut result = Vec::new();
        for (element, count) in &self.counts {
            for _ in 0..*count {
                result.push(element.clone());
            }
        }
        result
    }
    
    /// Subtract counts from another counter
    pub fn subtract(&mut self, other: &Counter<T>) {
        for (element, count) in &other.counts {
            self.update_one(element, -count);
        }
    }
    
    /// Get total count
    pub fn total(&self) -> i64 {
        self.counts.values().sum()
    }
    
    /// Clear all counts
    pub fn clear(&mut self) {
        self.counts.clear();
    }
    
    /// Get keys (elements)
    pub fn keys(&self) -> Vec<T> {
        self.counts.keys().cloned().collect()
    }
    
    /// Get values (counts)
    pub fn values(&self) -> Vec<i64> {
        self.counts.values().copied().collect()
    }
    
    /// Get items (element, count pairs)
    pub fn items(&self) -> Vec<(T, i64)> {
        self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }
}

impl<T> Default for Counter<T> 
where 
    T: Hash + Eq + Clone + core::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Len for Counter<T> 
where 
    T: Hash + Eq + Clone + core::fmt::Debug,
{
    fn len(&self) -> usize {
        self.counts.len()
    }
}

impl<T> Truthy for Counter<T> 
where 
    T: Hash + Eq + Clone + core::fmt::Debug,
{
    fn is_truthy(&self) -> bool {
        !self.counts.is_empty()
    }
}

/// deque - double-ended queue
///
/// `list.append`'s lowering target is `push`, so a deque reached through a
/// receiver whose type the converter could not see still appends correctly
/// ([`deque::push`] is `append`). The methods whose Python spelling differs
/// from the list's in an OBSERVABLE way (`pop`, `popleft`: CPython's
/// `IndexError: pop from an empty deque`) return a `Result`, so the list
/// lowering's `.pop().ok_or_else(..)` cannot compile against a deque: a
/// receiver the converter failed to type is a rustc error, never a wrong
/// exception message.
#[derive(Debug, Clone)]
pub struct deque<T> {
    inner: VecDeque<T>,
    maxlen: Option<usize>,
}

impl<T> deque<T> {
    /// Create a new deque
    pub fn new() -> Self {
        Self {
            inner: VecDeque::new(),
            maxlen: None,
        }
    }

    /// Create deque from iterable
    pub fn from_iter<I>(iterable: I, maxlen: Option<usize>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut deque = Self {
            inner: VecDeque::new(),
            maxlen,
        };
        for item in iterable {
            deque.append(item);
        }
        deque
    }

    /// Create deque with maximum length
    pub fn with_maxlen(maxlen: usize) -> Self {
        Self {
            inner: VecDeque::new(),
            maxlen: Some(maxlen),
        }
    }

    /// `deque(iterable, maxlen=...)` with CPython's argument check: a
    /// negative `maxlen` is `ValueError: maxlen must be non-negative`.
    /// `None` is an unbounded deque.
    pub fn construct<I>(iterable: I, maxlen: Option<i64>) -> Result<Self, PyException>
    where
        I: IntoIterator<Item = T>,
    {
        let maxlen = match maxlen {
            None => None,
            Some(n) if n < 0 => {
                return Err(crate::value_error("maxlen must be non-negative"));
            }
            Some(n) => Some(usize::try_from(n).unwrap_or(usize::MAX)),
        };
        Ok(Self::from_iter(iterable, maxlen))
    }

    /// Add element to right end
    pub fn append(&mut self, item: T) {
        self.inner.push_back(item);
        self.check_maxlen();
    }

    /// `list.append`'s lowering spelling: the same operation as
    /// [`deque::append`], bound trimming included.
    pub fn push(&mut self, item: T) {
        self.append(item);
    }

    /// Add element to left end
    pub fn appendleft(&mut self, item: T) {
        self.inner.push_front(item);
        self.check_maxlen_front();
    }

    /// Remove and return element from right end. An empty deque raises
    /// CPython's `IndexError: pop from an empty deque`.
    pub fn pop(&mut self) -> Result<T, PyException> {
        self.inner
            .pop_back()
            .ok_or_else(|| crate::index_error("pop from an empty deque"))
    }

    /// Remove and return element from left end (`IndexError: pop from an
    /// empty deque` when empty).
    pub fn popleft(&mut self) -> Result<T, PyException> {
        self.inner
            .pop_front()
            .ok_or_else(|| crate::index_error("pop from an empty deque"))
    }

    /// Extend right side with iterable
    pub fn extend<I>(&mut self, iterable: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iterable {
            self.append(item);
        }
    }

    /// Extend left side with iterable
    pub fn extendleft<I>(&mut self, iterable: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iterable {
            self.appendleft(item);
        }
    }

    /// Remove first occurrence of value. CPython 3.12 names the value:
    /// `ValueError: 9 is not in deque`.
    pub fn remove(&mut self, value: &T) -> Result<(), PyException>
    where
        T: PartialEq + crate::PyRepr,
    {
        if let Some(pos) = self.inner.iter().position(|x| x == value) {
            self.inner.remove(pos);
            Ok(())
        } else {
            Err(crate::value_error(format!("{} is not in deque", value.py_repr())))
        }
    }

    /// Rotate the deque `n` steps to the RIGHT (`n < 0` rotates left);
    /// the step count is taken modulo the length, as in CPython.
    pub fn rotate(&mut self, n: i64) {
        let len = self.inner.len();
        if len <= 1 {
            return;
        }
        let steps = n.rem_euclid(len as i64) as usize;
        self.inner.rotate_right(steps);
    }

    /// Reverse the deque in place
    pub fn reverse(&mut self) {
        let items: Vec<T> = self.inner.drain(..).collect();
        for item in items.into_iter().rev() {
            self.inner.push_back(item);
        }
    }

    /// Count occurrences of value
    pub fn count(&self, value: &T) -> usize
    where
        T: PartialEq,
    {
        self.inner.iter().filter(|&x| x == value).count()
    }

    /// Find index of first occurrence
    pub fn index(&self, value: &T, start: Option<usize>, stop: Option<usize>) -> Result<usize, PyException>
    where
        T: PartialEq + crate::PyRepr,
    {
        let start = start.unwrap_or(0);
        let stop = stop.unwrap_or(self.inner.len()).min(self.inner.len());

        for (i, item) in self
            .inner
            .iter()
            .enumerate()
            .skip(start)
            .take(stop.saturating_sub(start))
        {
            if item == value {
                return Ok(i);
            }
        }

        // CPython names the missing value with repr(): "9 is not in deque",
        // "'x' is not in deque" — single quotes for str, not Rust Debug's
        // double quotes.
        Err(crate::value_error(format!("{} is not in deque", value.py_repr())))
    }

    /// Insert item at position. A bounded deque at its maximum size raises
    /// IndexError like CPython (issue #82) instead of silently evicting.
    pub fn insert(&mut self, index: usize, item: T) -> Result<(), PyException> {
        if let Some(max_len) = self.maxlen {
            if self.inner.len() >= max_len {
                return Err(crate::index_error(
                    "deque already at its maximum size",
                ));
            }
        }
        if index >= self.inner.len() {
            self.inner.push_back(item);
        } else {
            self.inner.insert(index, item);
        }
        self.check_maxlen();
        Ok(())
    }

    /// Clear the deque
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Copy the deque
    pub fn copy(&self) -> Self
    where
        T: Clone,
    {
        Self {
            inner: self.inner.clone(),
            maxlen: self.maxlen,
        }
    }

    /// `deque.maxlen`: the bound, or None for an unbounded deque.
    pub fn maxlen(&self) -> Option<i64> {
        self.maxlen.map(|m| m as i64)
    }

    /// Trim to maxlen after growing at the BACK: CPython discards from
    /// the opposite end, i.e. the front.
    fn check_maxlen(&mut self) {
        if let Some(max_len) = self.maxlen {
            while self.inner.len() > max_len {
                self.inner.pop_front();
            }
        }
    }

    /// Trim after growing at the FRONT. Always popping the front would
    /// discard the element just added — `deque([1,2,3], maxlen=3)
    /// .appendleft(0)` is `deque([0,1,2])` in Python, not an unchanged
    /// deque.
    fn check_maxlen_front(&mut self) {
        if let Some(max_len) = self.maxlen {
            while self.inner.len() > max_len {
                self.inner.pop_back();
            }
        }
    }

    /// Get item by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Get item by index (mutable)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index)
    }

    /// The elements front to back as a `Vec` (the form the slice-taking
    /// builtins — `sorted`, `min`, `max`, ... — consume).
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.inner.iter().cloned().collect()
    }

    /// The elements front to back (borrowed).
    pub fn iter(&self) -> alloc::collections::vec_deque::Iter<'_, T> {
        self.inner.iter()
    }
}

impl<T> Default for deque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Len for deque<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> Truthy for deque<T> {
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

/// `bool(d)`: a non-empty deque is truthy.
impl<T> crate::PyBool for deque<T> {
    fn py_bool(self) -> bool {
        !self.inner.is_empty()
    }
}

/// `deque == deque` compares the CONTENTS (the bound is not part of
/// equality in CPython).
impl<T: PartialEq> PartialEq for deque<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T> IntoIterator for deque<T> {
    type Item = T;
    type IntoIter = alloc::collections::vec_deque::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a deque<T> {
    type Item = &'a T;
    type IntoIter = alloc::collections::vec_deque::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

/// `repr(deque)`: `deque([1, 2, 3])`, with `, maxlen=N` when bounded
/// (CPython 3.12).
impl<T: crate::PyRepr> crate::PyRepr for deque<T> {
    fn py_repr(&self) -> String {
        let items: Vec<String> = self.inner.iter().map(|x| x.py_repr()).collect();
        match self.maxlen {
            Some(m) => format!("deque([{}], maxlen={})", items.join(", "), m),
            None => format!("deque([{}])", items.join(", ")),
        }
    }
}

impl<T: crate::PyRepr> crate::PyDisplay for deque<T> {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

/// `str(d)` is its repr.
impl<T: crate::PyRepr> crate::PyToString for deque<T> {
    fn py_str(self) -> String {
        self.py_repr()
    }
}

impl<T: crate::PyRepr> crate::PyToString for &deque<T> {
    fn py_str(self) -> String {
        self.py_repr()
    }
}

/// `list(d)`: the elements front to back.
impl<T> crate::PyListFrom for deque<T> {
    type Item = T;
    fn py_list(self) -> Vec<T> {
        self.inner.into_iter().collect()
    }
}

impl<T: Clone> crate::PyListFrom for &deque<T> {
    type Item = T;
    fn py_list(self) -> Vec<T> {
        self.inner.iter().cloned().collect()
    }
}

/// `sum(d)`: the elements' sum, through the list impls.
impl<T> crate::PySum for deque<T>
where
    Vec<T>: crate::PySum,
{
    type Output = <Vec<T> as crate::PySum>::Output;
    fn py_sum(self) -> Self::Output {
        crate::PySum::py_sum(self.inner.into_iter().collect::<Vec<T>>())
    }
}

/// `d[i]` (negative indices count from the end): `IndexError: deque index
/// out of range`.
impl<T: Clone> crate::PyIndex<i64> for deque<T> {
    type Output = T;
    fn py_index(&self, index: i64) -> Result<T, PyException> {
        crate::normalize_index(index, self.inner.len())
            .map(|i| self.inner[i].clone())
            .ok_or_else(|| crate::index_error("deque index out of range"))
    }
}

impl<T> crate::PyIndexMut<i64> for deque<T> {
    type Output = T;
    fn py_index_mut(&mut self, index: i64) -> Result<&mut T, PyException> {
        let len = self.inner.len();
        match crate::normalize_index(index, len) {
            Some(i) => Ok(&mut self.inner[i]),
            None => Err(crate::index_error("deque index out of range")),
        }
    }
}

/// `d[i] = v`.
impl<T> crate::PySetIndex<i64, T> for deque<T> {
    fn py_set_index(&mut self, index: i64, value: T) -> Result<(), PyException> {
        let len = self.inner.len();
        match crate::normalize_index(index, len) {
            Some(i) => {
                self.inner[i] = value;
                Ok(())
            }
            None => Err(crate::index_error("deque index out of range")),
        }
    }
}

/// `x in d`.
impl<T: PartialEq> crate::PyContains<T> for deque<T> {
    fn py_contains(&self, item: &T) -> bool {
        self.inner.iter().any(|e| e == item)
    }
}

/// `del d[i]` (lowered through the shared `py_pop(i)` removal): removes
/// the item at `i` — negative indices count from the end — or raises
/// `IndexError: deque index out of range`.
impl<T> crate::PyPop<i64> for deque<T> {
    type Output = T;
    fn py_pop(&mut self, index: i64) -> Result<T, PyException> {
        crate::normalize_index(index, self.inner.len())
            .and_then(|i| self.inner.remove(i))
            .ok_or_else(|| crate::index_error("deque index out of range"))
    }
}

/// `"x" in d` for a deque of strings: a str literal lowers as `&str`.
impl crate::PyContains<&str> for deque<String> {
    fn py_contains(&self, item: &&str) -> bool {
        self.inner.iter().any(|e| e == *item)
    }
}

impl crate::PyContains<str> for deque<String> {
    fn py_contains(&self, item: &str) -> bool {
        self.inner.iter().any(|e| e == item)
    }
}

/// deque participates in the PyListOps traits so Python-level
/// `d.insert(i, x)` and `d.count(x)` lower through the same codegen arms as
/// lists. insert applies Python index rules and raises IndexError at maxlen
/// (issue #82) instead of silently evicting.
impl<T> crate::PyListOps<T> for deque<T> {
    fn count(&self, item: &T) -> i64
    where
        T: PartialEq,
    {
        self.inner.iter().filter(|e| *e == item).count() as i64
    }
    fn py_index_of(&self, item: &T) -> Result<i64, crate::PyException>
    where
        T: PartialEq,
    {
        self.inner
            .iter()
            .position(|e| e == item)
            .map(|i| i as i64)
            .ok_or_else(|| {
                // CPython 3.14: "deque.index(x): x not in deque" (verified
                // against python3.14.1 — the modern form since 3.10).
                crate::PyException::new("ValueError", "deque.index(x): x not in deque")
            })
    }
    fn py_insert(&mut self, index: i64, item: T) -> Result<(), crate::PyException> {
        if let Some(max_len) = self.maxlen {
            if self.inner.len() >= max_len {
                return Err(crate::index_error(
                    "deque already at its maximum size",
                ));
            }
        }
        let len = self.inner.len() as i64;
        let idx = if index < 0 {
            len.checked_add(index).unwrap_or(i64::MIN).max(0)
        } else {
            index.min(len)
        } as usize;
        if idx >= self.inner.len() {
            self.inner.push_back(item);
        } else {
            self.inner.insert(idx, item);
        }
        Ok(())
    }
}

/// The factory a `defaultdict` calls for a missing key: a plain function
/// pointer (a non-capturing closure coerces), plus the CLASS name when it
/// is a builtin type (`int`, `list`, ...) so `repr` can print
/// `<class 'int'>` exactly as CPython does. A lambda or user function has
/// no stable repr (CPython prints its memory address), so printing such a
/// defaultdict is a loud panic rather than a different string.
#[derive(Debug)]
pub struct DefaultFactory<V> {
    make: fn() -> V,
    class: Option<&'static str>,
}

impl<V> Clone for DefaultFactory<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for DefaultFactory<V> {}

/// defaultdict - dict subclass with default factory function.
///
/// The map lives behind a `RefCell` because CPython's `dd[k]` READ of a
/// missing key INSERTS the factory's value (visible to a later `len(dd)` or
/// iteration), while the runtime's `py_index` takes `&self` like every other
/// container's. The cell is never borrowed across a call into user code, so
/// the runtime borrow checks cannot fail. A `defaultdict` is therefore not
/// `Sync`: a module-level one cannot live in a `static` (the converter
/// refuses it with the rewrite — build it inside a function).
#[derive(Debug, Clone)]
pub struct defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    // Insertion-ordered PyDict, not a std HashMap: iteration order must
    // match CPython's dict (issue #82), which a HashMap randomizes per
    // process.
    inner: core::cell::RefCell<crate::PyDict<K, V>>,
    default_factory: Option<DefaultFactory<V>>,
    /// Built by `Default::default()` (a class struct's placeholder before
    /// its `__init__` assigns the real container), NOT by `defaultdict()` /
    /// `defaultdict(None)`. A missing-key read of such a value would raise
    /// a KeyError indistinguishable from a genuine one, so it panics
    /// instead (see [`defaultdict::missing_key`]).
    built_by_default: bool,
}

impl<K, V> defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create new defaultdict with factory function (an unnamed factory:
    /// a lambda or user function — see [`DefaultFactory`]).
    pub fn new(default_factory: fn() -> V) -> Self {
        Self {
            inner: core::cell::RefCell::new(crate::PyDict::default()),
            default_factory: Some(DefaultFactory {
                make: default_factory,
                class: None,
            }),
            built_by_default: false,
        }
    }

    /// Create a defaultdict whose factory is a builtin CLASS (`int`,
    /// `float`, `str`, `list`, `dict`, `set`): `class` is its Python name,
    /// which `repr` prints as `<class 'int'>`.
    pub fn with_class(default_factory: fn() -> V, class: &'static str) -> Self {
        Self {
            inner: core::cell::RefCell::new(crate::PyDict::default()),
            default_factory: Some(DefaultFactory {
                make: default_factory,
                class: Some(class),
            }),
            built_by_default: false,
        }
    }

    /// Create defaultdict without factory (`defaultdict()` /
    /// `defaultdict(None)`): a missing key raises KeyError like a dict.
    pub fn without_factory() -> Self {
        Self {
            inner: core::cell::RefCell::new(crate::PyDict::default()),
            default_factory: None,
            built_by_default: false,
        }
    }

    /// `defaultdict(factory, mapping)`: start from the mapping's items.
    pub fn with_items(self, items: crate::PyDict<K, V>) -> Self {
        *self.inner.borrow_mut() = items;
        self
    }

    /// Get value, creating with factory if missing (`dd[k]`). Without a
    /// factory a missing key is `KeyError: <key repr>`.
    pub fn get_or_default(&self, key: &K) -> Result<V, PyException>
    where
        K: crate::PyRepr,
    {
        if let Some(value) = self.inner.borrow().get(key) {
            return Ok(value.clone());
        }
        match self.default_factory {
            Some(factory) => {
                // The factory runs OUTSIDE the borrow.
                let default_value = (factory.make)();
                self.inner
                    .borrow_mut()
                    .insert(key.clone(), default_value.clone());
                Ok(default_value)
            }
            None => Err(self.missing_key(key)),
        }
    }

    /// The error for a missing key of a container with no factory: CPython's
    /// `KeyError` for a genuine `defaultdict()` / `defaultdict(None)`. A
    /// container built by `Default::default()` never received its
    /// default_factory (a converter bug), and a KeyError would be mistaken for
    /// the real thing by `except KeyError:`, so it panics.
    fn missing_key(&self, key: &K) -> PyException
    where
        K: crate::PyRepr,
    {
        if self.built_by_default {
            panic!("defaultdict built without its default_factory — rython bug, please report");
        }
        crate::key_error(key.py_repr())
    }

    /// Get value without creating default
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.borrow().get(key).cloned()
    }

    /// Set value
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.get_mut().insert(key, value)
    }

    /// Remove key
    pub fn remove(&mut self, key: &K) -> Option<V> {
        // shift_remove keeps the remaining insertion order intact (Python's
        // dict.pop preserves the order of the other keys).
        self.inner.get_mut().shift_remove(key)
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.borrow().contains_key(key)
    }

    /// Get keys
    pub fn keys(&self) -> Vec<K> {
        self.inner.borrow().keys().cloned().collect()
    }

    /// Get values
    pub fn values(&self) -> Vec<V> {
        self.inner.borrow().values().cloned().collect()
    }

    /// Get items
    pub fn items(&self) -> Vec<(K, V)> {
        self.inner
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.inner.get_mut().clear();
    }

    /// Get missing method (for compatibility)
    pub fn default_factory_fn(&self) -> Option<fn() -> V> {
        self.default_factory.map(|f| f.make)
    }

    /// `dd[k]` as a mutable place: inserts the factory's value when the key
    /// is missing, so `dd[k].append(x)` / `dd[k] += 1` mutate the stored value.
    fn slot_mut(&mut self, key: K) -> Result<&mut V, PyException>
    where
        K: crate::PyRepr,
    {
        if !self.inner.get_mut().contains_key(&key) {
            match self.default_factory {
                Some(factory) => {
                    let value = (factory.make)();
                    self.inner.get_mut().insert(key.clone(), value);
                }
                None => return Err(self.missing_key(&key)),
            }
        }
        // The key is present: this lookup cannot fail.
        self.inner
            .get_mut()
            .get_mut(&key)
            .ok_or_else(|| crate::key_error(key.py_repr()))
    }
}

impl<K, V> Len for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.inner.borrow().len()
    }
}

impl<K, V> Truthy for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn is_truthy(&self) -> bool {
        !self.inner.borrow().is_empty()
    }
}

/// `dd[k]`: the stored value, or the factory's (INSERTED) for a missing key.
impl<K, V> crate::PyIndex<K> for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_index(&self, key: K) -> Result<V, PyException> {
        self.get_or_default(&key)
    }
}

/// A string-keyed defaultdict indexed with a `&str` literal.
impl<V> crate::PyIndex<&str> for defaultdict<String, V>
where
    V: Clone,
{
    type Output = V;
    fn py_index(&self, key: &str) -> Result<V, PyException> {
        self.get_or_default(&key.to_string())
    }
}

impl<K, V> crate::PyIndexMut<K> for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_index_mut(&mut self, key: K) -> Result<&mut V, PyException> {
        self.slot_mut(key)
    }
}

impl<V> crate::PyIndexMut<&str> for defaultdict<String, V>
where
    V: Clone,
{
    type Output = V;
    fn py_index_mut(&mut self, key: &str) -> Result<&mut V, PyException> {
        self.slot_mut(key.to_string())
    }
}

impl<K, V> crate::PySetIndex<K, V> for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_set_index(&mut self, key: K, value: V) -> Result<(), PyException> {
        self.inner.get_mut().insert(key, value);
        Ok(())
    }
}

impl<K, V> crate::PyContains<K> for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_contains(&self, item: &K) -> bool {
        self.inner.borrow().contains_key(item)
    }
}

impl<V> crate::PyContains<str> for defaultdict<String, V>
where
    V: Clone,
{
    fn py_contains(&self, item: &str) -> bool {
        self.inner.borrow().contains_key(item)
    }
}

impl<V> crate::PyContains<&str> for defaultdict<String, V>
where
    V: Clone,
{
    fn py_contains(&self, item: &&str) -> bool {
        self.inner.borrow().contains_key(*item)
    }
}

/// The dict method surface. None of the reads inserts: `dd.get(k)` and
/// `k in dd` leave a missing key missing (only `dd[k]` runs the factory).
impl<K, V> crate::PyDictOps<K, V> for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_get(&self, key: &K) -> Option<V> {
        self.inner.borrow().get(key).cloned()
    }
    fn py_get_default(&self, key: &K, default: V) -> V {
        self.inner.borrow().get(key).cloned().unwrap_or(default)
    }
    fn py_keys(&self) -> Vec<K> {
        self.keys()
    }
    fn py_values(&self) -> Vec<V> {
        self.values()
    }
    fn py_items(&self) -> Vec<(K, V)> {
        self.items()
    }
    fn py_setdefault(&mut self, key: K, default: V) -> V {
        self.inner
            .get_mut()
            .entry(key)
            .or_insert(default)
            .clone()
    }
    fn update(&mut self, other: crate::PyDict<K, V>) {
        let inner = self.inner.get_mut();
        for (k, v) in other {
            inner.insert(k, v);
        }
    }
}

/// `dd.pop(k)`: the removed value, or `KeyError: <key repr>`.
impl<K, V> crate::PyPop<K> for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_pop(&mut self, key: K) -> Result<V, PyException> {
        self.inner
            .get_mut()
            .shift_remove(&key)
            .ok_or_else(|| crate::key_error(key.py_repr()))
    }
}

impl<K, V> crate::PyPopDefault<K, V> for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_pop_default(&mut self, key: K, default: V) -> V {
        self.inner.get_mut().shift_remove(&key).unwrap_or(default)
    }
}

impl<K, V> crate::PyCopy for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn copy(&self) -> Self {
        self.clone()
    }
}

/// `defaultdict::default()` is the factory-less (plain-dict-like) one: the
/// converter's class constructors build a struct through `Default` and then
/// assign the real fields.
impl<K, V> Default for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        let mut d = Self::without_factory();
        d.built_by_default = true;
        d
    }
}

/// `dd == {...}`: dict equality (the same items, in any order).
impl<K, V> PartialEq<crate::PyDict<K, V>> for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
{
    fn eq(&self, other: &crate::PyDict<K, V>) -> bool {
        *self.inner.borrow() == *other
    }
}

/// Dict equality: the same items, in any order (the factory is not part of it).
impl<K, V> PartialEq for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        *self.inner.borrow() == *other.inner.borrow()
    }
}

/// `repr(defaultdict)`: `defaultdict(<class 'int'>, {'a': 2})` (CPython
/// 3.12). A factory with no class name (a lambda or user function) prints
/// a memory address in CPython that no run can reproduce: refusing loudly
/// beats printing a different string.
impl<K, V> crate::PyRepr for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_repr(&self) -> String {
        if self.built_by_default {
            panic!("defaultdict built without its default_factory — rython bug, please report");
        }
        let factory = match self.default_factory {
            None => String::from("None"),
            Some(DefaultFactory { class: Some(name), .. }) => format!("<class '{}'>", name),
            Some(DefaultFactory { class: None, .. }) => panic!(
                "repr of a defaultdict whose default_factory is a lambda or user function \
                 prints a memory address in CPython (`<function <lambda> at 0x...>`); \
                 rython refuses to print a different string. Print its items instead"
            ),
        };
        let items: Vec<String> = self
            .inner
            .borrow()
            .iter()
            .map(|(k, v)| format!("{}: {}", k.py_repr(), v.py_repr()))
            .collect();
        format!("defaultdict({}, {{{}}})", factory, items.join(", "))
    }
}

impl<K, V> crate::PyDisplay for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

/// `str(dd)` is its repr.
impl<K, V> crate::PyToString for defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_str(self) -> String {
        self.py_repr()
    }
}

impl<K, V> crate::PyToString for &defaultdict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_str(self) -> String {
        self.py_repr()
    }
}

/// `bool(dd)`: a non-empty mapping is truthy.
impl<K, V> crate::PyBool for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_bool(self) -> bool {
        !self.inner.borrow().is_empty()
    }
}

/// `list(dd)`: the KEYS, in insertion order (a dict iterates its keys).
impl<K, V> crate::PyListFrom for defaultdict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    type Item = K;
    fn py_list(self) -> Vec<K> {
        self.keys()
    }
}

/// OrderedDict - dictionary that maintains insertion order.
///
/// Backed by the insertion-ordered [`crate::PyDict`] (every Python dict is
/// ordered; what OrderedDict adds is `move_to_end`, `popitem(last)`, an
/// order-sensitive `==` against another OrderedDict, and its own `repr`).
#[derive(Debug, Clone)]
pub struct OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    inner: crate::PyDict<K, V>,
}

impl<K, V> OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create new OrderedDict
    pub fn new() -> Self {
        Self {
            inner: crate::PyDict::default(),
        }
    }

    /// `OrderedDict(mapping)`: the mapping's items, in its order.
    pub fn from_dict(items: crate::PyDict<K, V>) -> Self {
        Self { inner: items }
    }

    /// `OrderedDict(pairs)`: the (key, value) pairs in order; a repeated
    /// key keeps its first position and takes the last value.
    pub fn from_pairs<I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut od = Self::new();
        for (k, v) in pairs {
            od.inner.insert(k, v);
        }
        od
    }

    /// Insert key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    /// Get value by key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    /// Remove key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.shift_remove(key)
    }

    /// `popitem(last=True)`: remove and return the LAST pair (the first
    /// with `last=False`); an empty OrderedDict is `KeyError: 'dictionary
    /// is empty'`.
    pub fn popitem(&mut self, last: bool) -> Result<(K, V), PyException> {
        let popped = if last {
            self.inner.pop()
        } else {
            self.inner.shift_remove_index(0)
        };
        // KeyError's message is the repr of its argument: 'dictionary is empty'.
        popped.ok_or_else(|| crate::key_error("'dictionary is empty'"))
    }

    /// `move_to_end(key, last=True)`: move an EXISTING key to the end (the
    /// front with `last=False`); a missing key is `KeyError: <key repr>`.
    pub fn move_to_end(&mut self, key: &K, last: bool) -> Result<(), PyException>
    where
        K: crate::PyRepr,
    {
        let Some((k, v)) = self.inner.shift_remove_entry(key) else {
            return Err(crate::key_error(key.py_repr()));
        };
        if last {
            self.inner.insert(k, v);
        } else {
            self.inner.shift_insert(0, k, v);
        }
        Ok(())
    }

    /// Get keys in order
    pub fn keys(&self) -> Vec<K> {
        self.inner.keys().cloned().collect()
    }

    /// Get values in order
    pub fn values(&self) -> Vec<V> {
        self.inner.values().cloned().collect()
    }

    /// Get items in order
    pub fn items(&self) -> Vec<(K, V)> {
        self.inner
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }
}

impl<K, V> Default for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Len for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> Truthy for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

/// `od[k]`: the value, or `KeyError: <key repr>`.
impl<K, V> crate::PyIndex<K> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_index(&self, key: K) -> Result<V, PyException> {
        self.inner
            .get(&key)
            .cloned()
            .ok_or_else(|| crate::key_error(key.py_repr()))
    }
}

impl<V> crate::PyIndex<&str> for OrderedDict<String, V>
where
    V: Clone,
{
    type Output = V;
    fn py_index(&self, key: &str) -> Result<V, PyException> {
        self.inner
            .get(key)
            .cloned()
            .ok_or_else(|| crate::key_error(crate::PyRepr::py_repr(key)))
    }
}

impl<K, V> crate::PyIndexMut<K> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_index_mut(&mut self, key: K) -> Result<&mut V, PyException> {
        let msg = key.py_repr();
        self.inner
            .get_mut(&key)
            .ok_or_else(|| crate::key_error(msg))
    }
}

impl<V> crate::PyIndexMut<&str> for OrderedDict<String, V>
where
    V: Clone,
{
    type Output = V;
    fn py_index_mut(&mut self, key: &str) -> Result<&mut V, PyException> {
        let msg = crate::PyRepr::py_repr(key);
        self.inner.get_mut(key).ok_or_else(|| crate::key_error(msg))
    }
}

impl<K, V> crate::PySetIndex<K, V> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_set_index(&mut self, key: K, value: V) -> Result<(), PyException> {
        self.inner.insert(key, value);
        Ok(())
    }
}

impl<K, V> crate::PyContains<K> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_contains(&self, item: &K) -> bool {
        self.inner.contains_key(item)
    }
}

impl<V> crate::PyContains<str> for OrderedDict<String, V>
where
    V: Clone,
{
    fn py_contains(&self, item: &str) -> bool {
        self.inner.contains_key(item)
    }
}

impl<V> crate::PyContains<&str> for OrderedDict<String, V>
where
    V: Clone,
{
    fn py_contains(&self, item: &&str) -> bool {
        self.inner.contains_key(*item)
    }
}

impl<K, V> crate::PyDictOps<K, V> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_get(&self, key: &K) -> Option<V> {
        self.inner.get(key).cloned()
    }
    fn py_get_default(&self, key: &K, default: V) -> V {
        self.inner.get(key).cloned().unwrap_or(default)
    }
    fn py_keys(&self) -> Vec<K> {
        self.keys()
    }
    fn py_values(&self) -> Vec<V> {
        self.values()
    }
    fn py_items(&self) -> Vec<(K, V)> {
        self.items()
    }
    fn py_setdefault(&mut self, key: K, default: V) -> V {
        self.inner.entry(key).or_insert(default).clone()
    }
    fn update(&mut self, other: crate::PyDict<K, V>) {
        for (k, v) in other {
            self.inner.insert(k, v);
        }
    }
}

impl<K, V> crate::PyPop<K> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone,
{
    type Output = V;
    fn py_pop(&mut self, key: K) -> Result<V, PyException> {
        self.inner
            .shift_remove(&key)
            .ok_or_else(|| crate::key_error(key.py_repr()))
    }
}

impl<K, V> crate::PyPopDefault<K, V> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_pop_default(&mut self, key: K, default: V) -> V {
        self.inner.shift_remove(&key).unwrap_or(default)
    }
}

impl<K, V> crate::PyCopy for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn copy(&self) -> Self {
        self.clone()
    }
}

/// `od == {...}` against a plain dict ignores the order (only two
/// OrderedDicts compare order-sensitively).
impl<K, V> PartialEq<crate::PyDict<K, V>> for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
{
    fn eq(&self, other: &crate::PyDict<K, V>) -> bool {
        self.inner == *other
    }
}

/// `OrderedDict == OrderedDict` is ORDER-sensitive (unlike dict equality).
impl<K, V> PartialEq for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len()
            && self
                .inner
                .iter()
                .zip(other.inner.iter())
                .all(|(a, b)| a.0 == b.0 && a.1 == b.1)
    }
}

/// `repr(OrderedDict)`: `OrderedDict({'b': 1, 'a': 2})` (the CPython 3.12+
/// form; 3.11 and earlier print `OrderedDict([('b', 1), ('a', 2)])`), and
/// `OrderedDict()` when empty.
impl<K, V> crate::PyRepr for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_repr(&self) -> String {
        if self.inner.is_empty() {
            return String::from("OrderedDict()");
        }
        let items: Vec<String> = self
            .inner
            .iter()
            .map(|(k, v)| format!("{}: {}", k.py_repr(), v.py_repr()))
            .collect();
        format!("OrderedDict({{{}}})", items.join(", "))
    }
}

impl<K, V> crate::PyDisplay for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

/// `str(od)` is its repr.
impl<K, V> crate::PyToString for OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_str(self) -> String {
        self.py_repr()
    }
}

impl<K, V> crate::PyToString for &OrderedDict<K, V>
where
    K: Hash + Eq + Clone + crate::PyRepr,
    V: Clone + crate::PyRepr,
{
    fn py_str(self) -> String {
        self.py_repr()
    }
}

/// `bool(od)`: a non-empty mapping is truthy.
impl<K, V> crate::PyBool for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn py_bool(self) -> bool {
        !self.inner.is_empty()
    }
}

/// `list(od)`: the KEYS, in insertion order.
impl<K, V> crate::PyListFrom for OrderedDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    type Item = K;
    fn py_list(self) -> Vec<K> {
        self.keys()
    }
}

/// An OrderedDict boxed into a [`crate::PyValue`] slot (a return of an
/// untyped function, an argument of a boxed parameter): it stays an
/// OrderedDict — it prints, compares and iterates as one. Keys convert to
/// the boxed dict's `String` keys, values to boxed values.
impl<K, V> From<OrderedDict<K, V>> for crate::PyValue
where
    K: Hash + Eq + Clone + Into<String>,
    V: Clone + Into<crate::PyValue>,
{
    fn from(od: OrderedDict<K, V>) -> Self {
        let mut boxed: OrderedDict<String, crate::PyValue> = OrderedDict::new();
        for (k, v) in od.inner {
            boxed.inner.insert(k.into(), v.into());
        }
        crate::PyValue::OrderedDict(alloc::sync::Arc::new(boxed))
    }
}

impl OrderedDict<String, crate::PyValue> {
    /// `OrderedDict(x)` where `x` is a BOXED value (an untyped parameter —
    /// requests' `from_key_val_list`): CPython's `update` over the value,
    /// producing the boxed OrderedDict.
    ///
    /// - a dict or OrderedDict contributes its items in order;
    /// - any other iterable contributes one (key, value) pair per member,
    ///   each member itself unpacked to exactly two values (a 2-tuple, a
    ///   2-element list, a 2-character str ...); a repeated key keeps its
    ///   first position and takes the last value;
    /// - everything else is CPython 3.12's own error, whose text differs
    ///   from `dict()`'s (the OrderedDict constructor unpacks each member
    ///   with the interpreter's unpack messages): `OrderedDict(5)` is
    ///   `TypeError: 'int' object is not iterable`, `OrderedDict([1])`
    ///   the same for the member, `OrderedDict([(1, 2, 3)])`
    ///   `ValueError: too many values to unpack (expected 2)` and
    ///   `OrderedDict([(1,)])` `ValueError: need more than 1 value to
    ///   unpack`.
    ///
    /// LIMITATION (loud): a boxed OrderedDict holds `str` keys, like the
    /// boxed dict. A pair whose key is not a str panics with a message
    /// naming the key instead of stringifying it (CPython accepts any
    /// hashable key); an unhashable key is CPython's TypeError.
    pub fn from_boxed(value: crate::PyValue) -> Result<crate::PyValue, PyException> {
        use crate::PyValue;
        let mut od: OrderedDict<String, PyValue> = OrderedDict::new();
        match value {
            PyValue::Dict(d) => {
                for (k, v) in d.iter() {
                    od.inner.insert(k.clone(), v.clone());
                }
            }
            // `OrderedDict(od)` copies: the result is a new mapping.
            PyValue::OrderedDict(o) => {
                return Ok(PyValue::OrderedDict(alloc::sync::Arc::new((*o).clone())));
            }
            PyValue::Int(_)
            | PyValue::Float(_)
            | PyValue::Bool(_)
            | PyValue::Complex(_)
            | PyValue::Function(_)
            | PyValue::None_ => return Err(not_iterable(&value)),
            iterable @ (PyValue::Str(_)
            | PyValue::Bytes(_)
            | PyValue::Tuple(_)
            | PyValue::Range(_)) => {
                for member in iterable {
                    let [k, v] = unpack_two(member)?;
                    od.inner.insert(boxed_key(k)?, v);
                }
            }
        }
        Ok(PyValue::OrderedDict(alloc::sync::Arc::new(od)))
    }
}

/// `TypeError: 'int' object is not iterable` for a boxed non-iterable.
fn not_iterable(value: &crate::PyValue) -> PyException {
    PyException::new(
        "TypeError",
        format!("'{}' object is not iterable", value.py_type_name()),
    )
}

/// Unpack one member of the pairs iterable into exactly two values the
/// way CPython 3.12's OrderedDict constructor does (the interpreter's
/// unpack messages, not `dict()`'s).
fn unpack_two(member: crate::PyValue) -> Result<[crate::PyValue; 2], PyException> {
    use crate::PyValue;
    if matches!(
        member,
        PyValue::Int(_)
            | PyValue::Float(_)
            | PyValue::Bool(_)
            | PyValue::Complex(_)
            | PyValue::Function(_)
            | PyValue::None_
    ) {
        return Err(not_iterable(&member));
    }
    let parts: Vec<PyValue> = member.into_iter().collect();
    match <[PyValue; 2]>::try_from(parts) {
        Ok(pair) => Ok(pair),
        Err(parts) if parts.len() > 2 => Err(PyException::new(
            "ValueError",
            "too many values to unpack (expected 2)",
        )),
        Err(parts) if parts.len() == 1 => Err(PyException::new(
            "ValueError",
            "need more than 1 value to unpack",
        )),
        Err(parts) => Err(PyException::new(
            "ValueError",
            format!("need more than {} values to unpack", parts.len()),
        )),
    }
}

/// A pair's key as the boxed OrderedDict's `String` key.
fn boxed_key(key: crate::PyValue) -> Result<String, PyException> {
    use crate::PyValue;
    match key {
        PyValue::Str(s) => Ok(s),
        PyValue::Dict(_) => Err(PyException::new("TypeError", "unhashable type: 'dict'")),
        PyValue::OrderedDict(_) => Err(PyException::new(
            "TypeError",
            "unhashable type: 'collections.OrderedDict'",
        )),
        other => panic!(
            "NotImplementedError: a boxed OrderedDict holds str keys only; got the {} key {} \
             (CPython accepts any hashable key; rython refuses to stringify it. Build a typed \
             OrderedDict first)",
            other.py_type_name(),
            crate::py_value_repr(&other)
        ),
    }
}

/// ChainMap - groups multiple mappings into single view
#[derive(Debug)]
pub struct ChainMap<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    maps: Vec<HashMap<K, V>>,
}

impl<K, V> ChainMap<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create new ChainMap
    pub fn new(maps: Vec<HashMap<K, V>>) -> Self {
        Self { maps }
    }
    
    /// Create empty ChainMap
    pub fn empty() -> Self {
        Self { maps: vec![HashMap::new()] }
    }
    
    /// Get value by key (searches all maps)
    pub fn get(&self, key: &K) -> Option<&V> {
        for map in &self.maps {
            if let Some(value) = map.get(key) {
                return Some(value);
            }
        }
        None
    }
    
    /// Set value (in first map)
    pub fn insert(&mut self, key: K, value: V) {
        if self.maps.is_empty() {
            self.maps.push(HashMap::new());
        }
        self.maps[0].insert(key, value);
    }
    
    /// Remove key from first map
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if !self.maps.is_empty() {
            self.maps[0].remove(key)
        } else {
            None
        }
    }
    
    /// Get all keys
    pub fn keys(&self) -> Vec<K> {
        let mut keys = HashSet::new();
        for map in &self.maps {
            keys.extend(map.keys().cloned());
        }
        keys.into_iter().collect()
    }
    
    /// Get all values
    pub fn values(&self) -> Vec<V> {
        let mut seen_keys = HashSet::new();
        let mut values = Vec::new();
        
        for map in &self.maps {
            for (key, value) in map {
                if !seen_keys.contains(key) {
                    seen_keys.insert(key.clone());
                    values.push(value.clone());
                }
            }
        }
        
        values
    }
    
    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.maps.iter().any(|map| map.contains_key(key))
    }
    
    /// Add new child map
    pub fn new_child(&mut self, map: HashMap<K, V>) -> &mut Self {
        self.maps.insert(0, map);
        self
    }
    
    /// Get number of maps
    pub fn num_maps(&self) -> usize {
        self.maps.len()
    }
}

impl<K, V> Len for ChainMap<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn len(&self) -> usize {
        self.keys().len()
    }
}

// Convenience functions
python_function! {
    /// Create counter from iterable
    pub fn counter<I>(iterable: I) -> Counter<String>
    where [I: IntoIterator<Item = String>]
    [signature: (iterable)]
    [concrete_types: (Vec<String>) -> Counter<String>]
    {
        Counter::from_iter(iterable)
    }
}

python_function! {
    /// Create deque from iterable
    pub fn create_deque<I>(iterable: I, maxlen: Option<usize>) -> deque<String>
    where [I: IntoIterator<Item = String>]
    [signature: (iterable, maxlen=None)]
    [concrete_types: (Vec<String>, Option<usize>) -> deque<String>]
    {
        deque::from_iter(iterable, maxlen)
    }
}

python_function! {
    /// Create defaultdict with int factory
    pub fn defaultdict_int() -> defaultdict<String, i64>
    [signature: ()]
    [concrete_types: () -> defaultdict<String, i64>]
    {
        defaultdict::with_class(|| 0i64, "int")
    }
}

python_function! {
    /// Create defaultdict with list factory
    pub fn defaultdict_list() -> defaultdict<String, Vec<String>>
    [signature: ()]
    [concrete_types: () -> defaultdict<String, Vec<String>>]
    {
        defaultdict::with_class(|| Vec::new(), "list")
    }
}