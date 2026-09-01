//! Standard Python Runtime Library for Rython
//! 
//! This library provides all the built-in functions, types, and methods
//! that are available in Python without any imports. It serves as the
//! runtime foundation for Python code compiled to Rust using python-ast-rs.
//!
//! ## Features
//!
//! The crate is layered as a `core ⊂ alloc ⊂ std` feature ladder, with
//! no_std reached by *absence* of the `std` feature:
//!
//! - `std` (default): the full runtime, including everything that touches
//!   the OS (I/O, os/os.path, datetime, subprocess, tempfile, glob,
//!   pathlib, random's OS entropy, pyo3 interop).
//! - `alloc` (implied by `std`): heap-backed Python semantics with no OS
//!   dependency — strings, lists, dicts/sets, exceptions, str.format.
//!   Build with `--no-default-features --features alloc` for
//!   `#![no_std]` + `alloc` output suitable for embedded/wasm targets.
//! - A strictly-core tier (no allocator at all) is not implemented yet;
//!   building without `alloc` fails loudly rather than pretending.

#![cfg_attr(not(feature = "std"), no_std)]
// Allow non-conventional naming for Python API compatibility
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

#[cfg(all(not(feature = "std"), not(feature = "alloc")))]
compile_error!(
    "stdpython requires the `alloc` feature when `std` is disabled: build with \
     `--no-default-features --features alloc`. A strictly-core tier (no allocator) \
     is not implemented yet — see https://github.com/rexlunae/rython/issues/22."
);

#[cfg(feature = "alloc")]
extern crate alloc;

// One import surface for both tiers: under `std`, these alloc/core paths
// name exactly the types the std prelude re-exports.
#[cfg(feature = "alloc")]
use alloc::{format, string::String, string::ToString, vec::Vec};

#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};
#[cfg(all(feature = "alloc", not(feature = "std")))]
use hashbrown::{HashMap, HashSet};

#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::sync::Arc;

use core::fmt::{Debug, Display};
use core::hash::Hash;

/// f64 math used by core Python semantics (`//`, `**`, round(), float
/// repr). The inherent f64 methods live in std, not core, so without std
/// these delegate to libm; the two agree on every case we rely on.
pub(crate) mod flt {
    #[cfg(feature = "std")]
    pub(crate) fn floor(x: f64) -> f64 { x.floor() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn floor(x: f64) -> f64 { libm::floor(x) }

    #[cfg(feature = "std")]
    pub(crate) fn trunc(x: f64) -> f64 { x.trunc() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn trunc(x: f64) -> f64 { libm::trunc(x) }

    /// Rounds half away from zero, like f64::round and libm::round both do.
    #[cfg(feature = "std")]
    pub(crate) fn round(x: f64) -> f64 { x.round() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn round(x: f64) -> f64 { libm::round(x) }

    #[cfg(feature = "std")]
    pub(crate) fn fract(x: f64) -> f64 { x.fract() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn fract(x: f64) -> f64 { x - libm::trunc(x) }

    #[cfg(feature = "std")]
    pub(crate) fn abs(x: f64) -> f64 { x.abs() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn abs(x: f64) -> f64 { libm::fabs(x) }

    #[cfg(feature = "std")]
    pub(crate) fn log10(x: f64) -> f64 { x.log10() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn log10(x: f64) -> f64 { libm::log10(x) }

    #[cfg(feature = "std")]
    pub(crate) fn signum(x: f64) -> f64 { x.signum() }
    #[cfg(not(feature = "std"))]
    pub(crate) fn signum(x: f64) -> f64 {
        if x.is_nan() { f64::NAN } else if x.is_sign_negative() { -1.0 } else { 1.0 }
    }

    #[cfg(feature = "std")]
    pub(crate) fn powf(x: f64, y: f64) -> f64 { x.powf(y) }
    #[cfg(not(feature = "std"))]
    pub(crate) fn powf(x: f64, y: f64) -> f64 { libm::pow(x, y) }

    #[cfg(feature = "std")]
    pub(crate) fn powi(x: f64, n: i32) -> f64 { x.powi(n) }
    #[cfg(not(feature = "std"))]
    pub(crate) fn powi(x: f64, n: i32) -> f64 { libm::pow(x, n as f64) }

    #[cfg(feature = "std")]
    pub(crate) fn copysign(x: f64, y: f64) -> f64 { x.copysign(y) }
    #[cfg(not(feature = "std"))]
    pub(crate) fn copysign(x: f64, y: f64) -> f64 { libm::copysign(x, y) }
}

// PyO3 lives behind its own surface feature (which implies std): only
// generated extension modules name these.
#[cfg(feature = "pyo3-interop")]
pub use pyo3::PyAny;
/// Alias kept for generated code; pyo3 0.29 removed the `PyObject` name.
#[cfg(feature = "pyo3-interop")]
pub type PyObject = pyo3::Py<pyo3::PyAny>;

// ============================================================================
// GENERIC TRAITS FOR PYTHON OPERATIONS
// ============================================================================

/// Trait for types that can be used as string-like parameters
/// 
/// This allows functions to accept both &str and String seamlessly
pub trait AsStrLike {
    fn as_str_like(&self) -> &str;
}

impl AsStrLike for str {
    fn as_str_like(&self) -> &str {
        self
    }
}

impl AsStrLike for String {
    fn as_str_like(&self) -> &str {
        self.as_str()
    }
}

impl AsStrLike for &str {
    fn as_str_like(&self) -> &str {
        self
    }
}

impl AsStrLike for &String {
    fn as_str_like(&self) -> &str {
        self.as_str()
    }
}

/// Trait for types that can be converted to owned strings
/// 
/// This is useful for return values that need to be owned
pub trait IntoOwnedString {
    fn into_owned_string(self) -> String;
}

impl IntoOwnedString for &str {
    fn into_owned_string(self) -> String {
        self.to_string()
    }
}

impl IntoOwnedString for String {
    fn into_owned_string(self) -> String {
        self
    }
}

/// Trait for types that can be used as path-like parameters
/// 
/// This allows path functions to work with various string types
pub trait AsPathLike {
    fn as_path_like(&self) -> &str;
}

impl<T: AsStrLike> AsPathLike for T {
    fn as_path_like(&self) -> &str {
        self.as_str_like()
    }
}

/// Trait for collections that can be used as argument lists
/// 
/// This allows subprocess functions to accept various collection types
pub trait AsArgList<T> {
    fn as_arg_list(&self) -> Vec<&str>;
}

impl<T> AsArgList<T> for Vec<T> 
where
    T: AsRef<str>,
{
    fn as_arg_list(&self) -> Vec<&str> {
        self.iter().map(|s| s.as_ref()).collect()
    }
}

impl<T> AsArgList<T> for &[T] 
where
    T: AsRef<str>,
{
    fn as_arg_list(&self) -> Vec<&str> {
        self.iter().map(|s| s.as_ref()).collect()
    }
}

/// Trait for environment-like collections (key-value pairs)
pub trait AsEnvLike<K, V> {
    fn as_env_like(&self) -> HashMap<&str, &str>;
}

impl<K, V> AsEnvLike<K, V> for HashMap<K, V>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    fn as_env_like(&self) -> HashMap<&str, &str> {
        self.iter()
            .map(|(k, v)| (k.as_ref(), v.as_ref()))
            .collect()
    }
}

// ============================================================================
// PYTHON BUILT-IN FUNCTIONS
// ============================================================================

/// Python's str()-conversion as used by print: identical to repr except
/// that strings render unquoted and an absent Option renders as None.
/// This is NOT Rust's Display — Display prints `true` and
/// `10000000000000000` where Python prints `True` and `1e+16`, so print
/// must never fall back to it.
pub trait PyDisplay {
    fn py_display(&self) -> String;
}

// Every integer primitive renders like a Python int. Covering them all
// (not just i64) matters twice over: len() yields usize, and an integer
// LITERAL among several candidate impls falls back to i32 — which must
// then have an impl, or inference fails.
macro_rules! py_display_int {
    ($($t:ty),*) => {$(
        impl PyDisplay for $t {
            fn py_display(&self) -> String {
                self.to_string()
            }
        }
    )*};
}
py_display_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl PyDisplay for f64 {
    fn py_display(&self) -> String {
        // Python 3's str(float) IS repr(float).
        py_float_repr(*self)
    }
}

impl PyDisplay for bool {
    fn py_display(&self) -> String {
        if *self { "True" } else { "False" }.to_string()
    }
}

impl PyDisplay for str {
    fn py_display(&self) -> String {
        self.to_string()
    }
}

impl PyDisplay for String {
    fn py_display(&self) -> String {
        self.clone()
    }
}

impl<T: PyDisplay + ?Sized> PyDisplay for &T {
    fn py_display(&self) -> String {
        (**self).py_display()
    }
}

impl<T: PyDisplay + ?Sized> PyDisplay for &mut T {
    fn py_display(&self) -> String {
        (**self).py_display()
    }
}

/// Containers stringify their ELEMENTS with repr, as Python does:
/// str(['a']) is "['a']", quotes and all.
impl<T: PyRepr> PyDisplay for Vec<T> {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

/// In the Option-based None model, str(None) is "None" and a present
/// value stringifies as itself (unquoted when it is a string).
impl<T: PyDisplay> PyDisplay for Option<T> {
    fn py_display(&self) -> String {
        match self {
            Some(x) => x.py_display(),
            None => "None".to_string(),
        }
    }
}

/// Free-function form of PyDisplay, used by generated multi-argument
/// print calls to pre-render each argument.
pub fn py_display<T: PyDisplay + ?Sized>(x: &T) -> String {
    x.py_display()
}

/// Python print() with a single argument and default sep/end.
/// Note: Only available with `std` feature - requires OS I/O capabilities
#[cfg(feature = "std")]
pub fn print<T: PyDisplay>(object: T) {
    println!("{}", object.py_display());
}

/// Python print() with multiple arguments and/or explicit sep=/end=:
/// the arguments arrive pre-rendered through py_display.
/// Note: Only available with `std` feature - requires OS I/O capabilities
#[cfg(feature = "std")]
pub fn print_parts<S: AsRef<str>, Sep: AsRef<str>, E: AsRef<str>>(parts: &[S], sep: Sep, end: E) {
    let output = parts
        .iter()
        .map(|p| p.as_ref())
        .collect::<Vec<_>>()
        .join(sep.as_ref());
    print!("{}{}", output, end.as_ref());
}

/// print(..., flush=True): as print_parts, then flush stdout when asked.
/// Note: Only available with `std` feature - requires OS I/O capabilities
#[cfg(feature = "std")]
pub fn print_parts_flush<S: AsRef<str>, Sep: AsRef<str>, E: AsRef<str>>(
    parts: &[S],
    sep: Sep,
    end: E,
    flush: bool,
) {
    print_parts(parts, sep, end);
    if flush {
        use std::io::Write;
        std::io::stdout()
            .flush()
            .expect("print(flush=True): I/O error flushing stdout");
    }
}

/// No-std version of print - stores output in a string instead of printing
///
/// This version is available in nostd environments but doesn't perform actual I/O
#[cfg(not(feature = "std"))]
pub fn print_to_string<T: PyDisplay>(object: T) -> String {
    object.py_display()
}

/// No-std version of print with multiple arguments
#[cfg(not(feature = "std"))]
pub fn print_args_to_string<T: PyDisplay, S: AsRef<str>, E: AsRef<str>>(objects: &[T], sep: S, end: E) -> String {
    let output = objects.iter()
        .map(|obj| obj.py_display())
        .collect::<Vec<_>>()
        .join(sep.as_ref());
    format!("{}{}", output, end.as_ref())
}

/// Python len() function - returns the length of an object. String
/// lengths are CODE POINTS, as in Python: len("café") == 4.
pub fn len<T>(obj: &T) -> usize
where
    T: Len + ?Sized,
{
    obj.len()
}

/// Python dict() function - creates a new dictionary (generic version)
/// 
/// # Arguments
/// * `pairs` - Key-value pairs to initialize the dictionary with
/// 
/// # Returns
/// A new HashMap containing the provided key-value pairs
pub fn dict<K, V>(pairs: HashMap<K, V>) -> HashMap<K, V> 
where
    K: Hash + Eq,
{
    pairs
}

/// Python dict() function with environment merging (generic version)
/// 
/// This merges environment-like collections with additional key-value pairs
pub fn dict_with_env<E, K, V>(env: E, additional: HashMap<K, V>) -> HashMap<K, V>
where
    E: AsEnvLike<K, V>,
    K: Hash + Eq + for<'a> From<&'a str>,
    V: for<'a> From<&'a str>,
{
    let env_map = env.as_env_like();
    let mut result: HashMap<K, V> = env_map.into_iter()
        .map(|(k, v)| (K::from(k), V::from(v)))
        .collect();
    result.extend(additional);
    result
}

/// Simplified dict creation from key-value pairs
pub fn dict_from_pairs<K, V, I>(pairs: I) -> HashMap<K, V>
where
    K: Hash + Eq,
    I: IntoIterator<Item = (K, V)>,
{
    pairs.into_iter().collect()
}

// ============================================================================
// PYTHON NUMERIC OPERATIONS
// ============================================================================

/// Trait for types that support absolute value
pub trait PyAbs {
    type Output;
    fn py_abs(self) -> Self::Output;
}

/// Python sum() as a trait with the OUTPUT as an associated type: every
/// iterable sums to exactly one result type, so generic bounds compose
/// with the operator-Output machinery (`<T as PySum>::Output` in
/// inferred signatures) and a use context can pin the result
/// (`T: PySum<Output = i64>` — issue #133's calc, where the target list
/// was already element-typed i64).
pub trait PySum {
    type Output;
    fn py_sum(self) -> Self::Output;
}

/// Python abs() function - returns absolute value
pub fn abs<T: PyAbs>(x: T) -> T::Output {
    x.py_abs()
}

/// Python min() on an iterable. PartialOrd (not Ord) so floats work, with
/// the fold Python's own comparison loop produces: the current best is
/// replaced only when a later element is STRICTLY smaller, which is
/// exactly why `min([nan, 1.0])` is nan but `min([1.0, nan])` is 1.0.
/// An empty iterable raises ValueError, as in Python.
pub fn min<T: PartialOrd + Clone>(iterable: &[T]) -> Result<T, PyException> {
    let mut it = iterable.iter();
    let mut best = it
        .next()
        .ok_or_else(|| PyException::new("ValueError", "min() arg is an empty sequence"))?;
    for x in it {
        if x < best {
            best = x;
        }
    }
    Ok(best.clone())
}

/// Python max() on an iterable; see min() for the comparison semantics.
pub fn max<T: PartialOrd + Clone>(iterable: &[T]) -> Result<T, PyException> {
    let mut it = iterable.iter();
    let mut best = it
        .next()
        .ok_or_else(|| PyException::new("ValueError", "max() arg is an empty sequence"))?;
    for x in it {
        if x > best {
            best = x;
        }
    }
    Ok(best.clone())
}

/// Python min(a, b): b wins only when strictly smaller (ties and
/// incomparable NaNs keep the first argument, as in Python). The n-ary
/// form folds through this.
pub fn min2<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

/// Python max(a, b); see min2.
pub fn max2<T: PartialOrd>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

/// Python min(iterable, default=d): the default only covers emptiness.
pub fn min_default<T: PartialOrd + Clone>(iterable: &[T], default: T) -> T {
    min(iterable).unwrap_or(default)
}

/// Python max(iterable, default=d).
pub fn max_default<T: PartialOrd + Clone>(iterable: &[T], default: T) -> T {
    max(iterable).unwrap_or(default)
}

/// Python min(iterable, key=f): comparisons use f(x), the ELEMENT is
/// returned, and f runs once per element. Ties keep the earliest element.
pub fn min_key<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    mut key: F,
) -> Result<T, PyException> {
    let mut it = iterable.iter();
    let first = it
        .next()
        .ok_or_else(|| PyException::new("ValueError", "min() arg is an empty sequence"))?;
    let mut best = first;
    let mut best_key = key(first);
    for x in it {
        let k = key(x);
        if k < best_key {
            best = x;
            best_key = k;
        }
    }
    Ok(best.clone())
}

/// Python max(iterable, key=f); see min_key.
pub fn max_key<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    mut key: F,
) -> Result<T, PyException> {
    let mut it = iterable.iter();
    let first = it
        .next()
        .ok_or_else(|| PyException::new("ValueError", "max() arg is an empty sequence"))?;
    let mut best = first;
    let mut best_key = key(first);
    for x in it {
        let k = key(x);
        if k > best_key {
            best = x;
            best_key = k;
        }
    }
    Ok(best.clone())
}

/// Python min(iterable, key=f, default=d).
pub fn min_key_default<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    key: F,
    default: T,
) -> T {
    min_key(iterable, key).unwrap_or(default)
}

/// Python max(iterable, key=f, default=d).
pub fn max_key_default<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    key: F,
    default: T,
) -> T {
    max_key(iterable, key).unwrap_or(default)
}

/// The comparator behind sorted(): Python's sort only needs `<`, which
/// PartialOrd supplies for every type we lower — except NaN, where
/// CPython's timsort silently produces an arbitrary-looking (though
/// deterministic) order no other sort reproduces. Exactness being
/// impossible, sorting NaN fails loudly instead of quietly diverging.
fn py_sort_cmp<T: PartialOrd>(a: &T, b: &T) -> core::cmp::Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        panic!(
            "{}",
            PyException::new(
                "ValueError",
                "cannot sort values without a total order (NaN); Python's NaN sort \
                 order is not reproducible",
            )
        )
    })
}

/// Python sorted(iterable): a new, stably ascending list.
pub fn sorted<T: PartialOrd + Clone>(iterable: &[T]) -> Vec<T> {
    let mut out = iterable.to_vec();
    out.sort_by(py_sort_cmp);
    out
}

/// Python sorted(iterable, reverse=...): stable descending when true —
/// equal elements keep their original order (this is NOT a plain
/// reversal, which would flip ties).
pub fn sorted_reverse<T: PartialOrd + Clone>(iterable: &[T], reverse: bool) -> Vec<T> {
    let mut out = iterable.to_vec();
    if reverse {
        out.sort_by(|a, b| py_sort_cmp(b, a));
    } else {
        out.sort_by(py_sort_cmp);
    }
    out
}

/// Python sorted(iterable, key=f): decorate-sort-undecorate, so the key
/// function runs exactly once per element, as in Python.
pub fn sorted_key<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    mut key: F,
) -> Vec<T> {
    let mut decorated: Vec<(K, T)> = iterable.iter().map(|x| (key(x), x.clone())).collect();
    decorated.sort_by(|a, b| py_sort_cmp(&a.0, &b.0));
    decorated.into_iter().map(|(_, x)| x).collect()
}

/// Python sorted(iterable, key=f, reverse=...).
pub fn sorted_key_reverse<T: Clone, K: PartialOrd, F: FnMut(&T) -> K>(
    iterable: &[T],
    mut key: F,
    reverse: bool,
) -> Vec<T> {
    let mut decorated: Vec<(K, T)> = iterable.iter().map(|x| (key(x), x.clone())).collect();
    if reverse {
        decorated.sort_by(|a, b| py_sort_cmp(&b.0, &a.0));
    } else {
        decorated.sort_by(|a, b| py_sort_cmp(&a.0, &b.0));
    }
    decorated.into_iter().map(|(_, x)| x).collect()
}

/// Python list.sort(): in-place, stable, with the key=/reverse= keyword
/// shapes. Distinct from Vec's inherent sort, which demands a total
/// order (rejecting floats); these share sorted()'s comparator, which
/// sorts floats and panics loudly on NaN. The key function runs exactly
/// once per element (decorate-sort-undecorate), as in Python.
pub trait PySort {
    type Item;
    fn py_sort(&mut self)
    where
        Self::Item: PartialOrd;
    fn py_sort_reverse(&mut self, reverse: bool)
    where
        Self::Item: PartialOrd;
    fn py_sort_key<K: PartialOrd, F: FnMut(&Self::Item) -> K>(&mut self, key: F);
    fn py_sort_key_reverse<K: PartialOrd, F: FnMut(&Self::Item) -> K>(
        &mut self,
        key: F,
        reverse: bool,
    );
}

impl<T> PySort for Vec<T> {
    type Item = T;

    fn py_sort(&mut self)
    where
        T: PartialOrd,
    {
        self.sort_by(py_sort_cmp);
    }

    // Stable descending, so equal elements keep their original order —
    // NOT a sort-then-reverse, which would flip ties.
    fn py_sort_reverse(&mut self, reverse: bool)
    where
        T: PartialOrd,
    {
        if reverse {
            self.sort_by(|a, b| py_sort_cmp(b, a));
        } else {
            self.sort_by(py_sort_cmp);
        }
    }

    fn py_sort_key<K: PartialOrd, F: FnMut(&T) -> K>(&mut self, mut key: F) {
        let mut decorated: Vec<(K, T)> = core::mem::take(self)
            .into_iter()
            .map(|x| (key(&x), x))
            .collect();
        decorated.sort_by(|a, b| py_sort_cmp(&a.0, &b.0));
        *self = decorated.into_iter().map(|(_, x)| x).collect();
    }

    fn py_sort_key_reverse<K: PartialOrd, F: FnMut(&T) -> K>(&mut self, mut key: F, reverse: bool) {
        let mut decorated: Vec<(K, T)> = core::mem::take(self)
            .into_iter()
            .map(|x| (key(&x), x))
            .collect();
        if reverse {
            decorated.sort_by(|a, b| py_sort_cmp(&b.0, &a.0));
        } else {
            decorated.sort_by(|a, b| py_sort_cmp(&a.0, &b.0));
        }
        *self = decorated.into_iter().map(|(_, x)| x).collect();
    }
}

/// Python reversed(sequence), materialized.
pub fn reversed<T: Clone>(iterable: &[T]) -> Vec<T> {
    iterable.iter().rev().cloned().collect()
}

/// Python sum() function
pub fn sum<I: PySum>(iterable: I) -> I::Output {
    iterable.py_sum()
}

/// Trait backing Python's `//` operator, whose result follows the sign of
/// the divisor (unlike Rust's truncating `/`). Generic over the operand
/// types so `NdArray` (numpy `floor_divide`) can participate alongside the
/// numeric primitives. Returns a Result so a zero divisor raises a
/// catchable ZeroDivisionError instead of panicking past try/except.
pub trait PyFloorDiv<R: ?Sized> {
    type Output;
    fn py_floordiv(&self, rhs: &R) -> Result<Self::Output, PyException>;
}

/// Trait backing Python's `%` operator (result takes the divisor's sign).
pub trait PyMod<R: ?Sized> {
    type Output;
    fn py_mod(&self, rhs: &R) -> Result<Self::Output, PyException>;
}

/// Trait backing Python's `/` true division (always float) — generic so
/// `NdArray` (numpy `divide`) can participate. Returns a Result so a zero
/// divisor raises a catchable ZeroDivisionError instead of silently
/// yielding inf/nan (issue #107, the `/` counterpart of #75's `//`/`%`).
pub trait PyDiv<R: ?Sized> {
    type Output;
    fn py_div(&self, rhs: &R) -> Result<Self::Output, PyException>;
}

/// Trait backing Python's `@` matrix multiplication operator.
pub trait PyMatMul<R: ?Sized> {
    type Output;
    fn py_matmul(&self, rhs: &R) -> Self::Output;
}

impl PyFloorDiv<i64> for i64 {
    type Output = i64;
    fn py_floordiv(&self, rhs: &i64) -> Result<i64, PyException> {
        if *rhs == 0 {
            return Err(PyException::new(
                "ZeroDivisionError",
                "integer division or modulo by zero",
            ));
        }
        let q = *self / *rhs;
        if *self % *rhs != 0 && (*self < 0) != (*rhs < 0) {
            Ok(q - 1)
        } else {
            Ok(q)
        }
    }
}

impl PyFloorDiv<f64> for f64 {
    type Output = f64;
    fn py_floordiv(&self, rhs: &f64) -> Result<f64, PyException> {
        if *rhs == 0.0 {
            // Python raises here; returning inf would diverge silently.
            return Err(PyException::new(
                "ZeroDivisionError",
                "float floor division by zero",
            ));
        }
        Ok(flt::floor(*self / *rhs))
    }
}

impl PyFloorDiv<f64> for i64 {
    type Output = f64;
    fn py_floordiv(&self, rhs: &f64) -> Result<f64, PyException> {
        PyFloorDiv::py_floordiv(&(*self as f64), rhs)
    }
}

impl PyFloorDiv<i64> for f64 {
    type Output = f64;
    fn py_floordiv(&self, rhs: &i64) -> Result<f64, PyException> {
        PyFloorDiv::py_floordiv(self, &(*rhs as f64))
    }
}

impl PyMod<i64> for i64 {
    type Output = i64;
    fn py_mod(&self, rhs: &i64) -> Result<i64, PyException> {
        if *rhs == 0 {
            return Err(PyException::new(
                "ZeroDivisionError",
                "integer division or modulo by zero",
            ));
        }
        let r = *self % *rhs;
        if r != 0 && (r < 0) != (*rhs < 0) {
            Ok(r + *rhs)
        } else {
            Ok(r)
        }
    }
}

// Old-style %-formatting on str and bytes (round 34): the `%` operator
// with a str/bytes LHS and a single value or tuple RHS is Python's
// printf formatting (percent_format.rs), not modulo.
impl<R: crate::percent_format::PyFormatRhs> PyMod<R> for &str {
    type Output = String;
    fn py_mod(&self, rhs: &R) -> Result<String, PyException> {
        crate::percent_format::py_format_str(self.as_bytes(), rhs)
    }
}

impl<R: crate::percent_format::PyFormatRhs> PyMod<R> for String {
    type Output = String;
    fn py_mod(&self, rhs: &R) -> Result<String, PyException> {
        crate::percent_format::py_format_str(self.as_bytes(), rhs)
    }
}

impl<R: crate::percent_format::PyFormatRhs> PyMod<R> for Vec<u8> {
    type Output = Vec<u8>;
    fn py_mod(&self, rhs: &R) -> Result<Vec<u8>, PyException> {
        crate::percent_format::py_format_bytes(self, rhs)
    }
}

impl PyMod<f64> for f64 {
    type Output = f64;
    fn py_mod(&self, rhs: &f64) -> Result<f64, PyException> {
        if *rhs == 0.0 {
            return Err(PyException::new("ZeroDivisionError", "float modulo"));
        }
        let r = *self % *rhs;
        if r != 0.0 && (r < 0.0) != (*rhs < 0.0) {
            Ok(r + *rhs)
        } else if r == 0.0 {
            // CPython gives a zero remainder the sign of the DIVISOR:
            // -4.0 % 2.0 is 0.0, and 4.0 % -2.0 is -0.0.
            Ok(flt::copysign(0.0, *rhs))
        } else {
            Ok(r)
        }
    }
}

impl PyMod<f64> for i64 {
    type Output = f64;
    fn py_mod(&self, rhs: &f64) -> Result<f64, PyException> {
        PyMod::py_mod(&(*self as f64), rhs)
    }
}

impl PyMod<i64> for f64 {
    type Output = f64;
    fn py_mod(&self, rhs: &i64) -> Result<f64, PyException> {
        PyMod::py_mod(self, &(*rhs as f64))
    }
}

/// Zero test that works across the numeric operand types of the division
/// helpers (Rust will not coerce an integer literal through `==` on `f64`).
trait IsZero {
    fn is_zero(&self) -> bool;
}
impl IsZero for i64 {
    fn is_zero(&self) -> bool {
        *self == 0
    }
}
impl IsZero for f64 {
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}
impl IsZero for bool {
    fn is_zero(&self) -> bool {
        !*self
    }
}

macro_rules! numeric_div {
    ($msg:expr; $($l:ty, $r:ty => $out:ty),* $(,)?) => {
        $(impl PyDiv<$r> for $l {
            type Output = $out;
            fn py_div(&self, rhs: &$r) -> Result<$out, PyException> {
                if rhs.is_zero() {
                    return Err(PyException::new("ZeroDivisionError", $msg));
                }
                Ok((*self as f64) / (*rhs as f64))
            }
        })*
    };
}

// bool operands promote through the numeric path (True → 1.0) like numpy.
macro_rules! bool_div {
    ($msg:expr; $($r:ty),* $(,)?) => {
        $(
            impl PyDiv<$r> for bool {
                type Output = f64;
                fn py_div(&self, rhs: &$r) -> Result<f64, PyException> {
                    if rhs.is_zero() {
                        return Err(PyException::new("ZeroDivisionError", $msg));
                    }
                    Ok((if *self { 1.0 } else { 0.0 }) / (*rhs as f64))
                }
            }
            impl PyDiv<bool> for $r {
                type Output = f64;
                fn py_div(&self, rhs: &bool) -> Result<f64, PyException> {
                    if rhs.is_zero() {
                        return Err(PyException::new("ZeroDivisionError", $msg));
                    }
                    Ok((*self as f64) / if *rhs { 1.0 } else { 0.0 })
                }
            }
        )*
    };
}

// CPython's messages: int/int (and bool, an int subclass) true division by
// zero raises "division by zero"; any float operand raises "float division
// by zero".
numeric_div!("division by zero"; i64, i64 => f64);
numeric_div!("float division by zero"; i64, f64 => f64, f64, i64 => f64, f64, f64 => f64);

bool_div!("division by zero"; i64);
bool_div!("float division by zero"; f64);

impl PyDiv<bool> for bool {
    type Output = f64;
    fn py_div(&self, rhs: &bool) -> Result<f64, PyException> {
        if rhs.is_zero() {
            return Err(PyException::new(
                "ZeroDivisionError",
                "division by zero",
            ));
        }
        Ok((if *self { 1.0 } else { 0.0 }) / if *rhs { 1.0 } else { 0.0 })
    }
}

/// Python `//` (floor division): `-7 // 2 == -4`. Raises a catchable
/// ZeroDivisionError on a zero divisor.
pub fn py_floordiv<L: PyFloorDiv<R>, R>(a: L, b: R) -> Result<L::Output, PyException> {
    a.py_floordiv(&b)
}

/// Python `%` (modulo takes the divisor's sign): `-7 % 3 == 2`. Raises a
/// catchable ZeroDivisionError on a zero divisor.
pub fn py_mod<L: PyMod<R>, R>(a: L, b: R) -> Result<L::Output, PyException> {
    a.py_mod(&b)
}

/// Python `/` (true division): `py_div(3, 2) == 1.5`. Raises a catchable
/// ZeroDivisionError on a zero divisor ("division by zero" for int operands,
/// "float division by zero" when either operand is a float — issue #107).
pub fn py_div<L: PyDiv<R>, R>(a: L, b: R) -> Result<L::Output, PyException> {
    a.py_div(&b)
}



/// Python `@` (matrix multiplication): routes to the numpy linalg backend
/// for arrays.
pub fn py_matmul<L: PyMatMul<R>, R>(a: L, b: R) -> L::Output {
    a.py_matmul(&b)
}

/// Python divmod() builtin, floor-division based: `divmod(-7, 2) == (-4, 1)`.
pub fn divmod<L: PyFloorDiv<R> + PyMod<R>, R>(
    a: L,
    b: R,
) -> Result<(<L as PyFloorDiv<R>>::Output, <L as PyMod<R>>::Output), PyException> {
    Ok((a.py_floordiv(&b)?, a.py_mod(&b)?))
}

/// Trait backing Python's `**` operator. Integer bases with non-negative
/// integer exponents stay integers; anything involving a float is a float.
pub trait PyPow<Rhs = Self> {
    type Output;
    fn py_pow(self, rhs: Rhs) -> Self::Output;
}

impl PyPow for i64 {
    type Output = i64;
    fn py_pow(self, rhs: i64) -> i64 {
        if rhs < 0 {
            // Python promotes to float here; an integer-typed context can't,
            // so fail loudly rather than return a wrong integer.
            panic!("integer ** negative exponent yields a float; use a float base");
        }
        if rhs > u32::MAX as i64 {
            // The `as u32` truncation below would silently wrap (0 **
            // 4294967296 must be 0, not 1). For these bases the
            // mathematical result is 0/1/±1; anything else overflows i64
            // beyond any doubt.
            return match self {
                0 => 0,
                1 => 1,
                -1 if rhs % 2 == 0 => 1,
                -1 => -1,
                _ => panic!("{} ** {} overflows i64", self, rhs),
            };
        }
        self.checked_pow(rhs as u32)
            .unwrap_or_else(|| panic!("{} ** {} overflows i64", self, rhs))
    }
}

impl PyPow<i64> for f64 {
    type Output = f64;
    fn py_pow(self, rhs: i64) -> f64 {
        // CPython converts the exponent to a double and calls libm pow.
        // powi's repeated squaring differs in the last ULPs — 0.1 ** 4 is
        // 0.00010000000000000002 in Python but ...05 via powi — and
        // `rhs as i32` would silently truncate a large exponent.
        flt::powf(self, rhs as f64)
    }
}

impl PyPow for f64 {
    type Output = f64;
    fn py_pow(self, rhs: f64) -> f64 {
        flt::powf(self, rhs)
    }
}

impl PyPow<f64> for i64 {
    type Output = f64;
    fn py_pow(self, rhs: f64) -> f64 {
        flt::powf(self as f64, rhs)
    }
}

/// Python `**` (power). `py_pow(2, 10) == 1024`, `py_pow(2.0, -1) == 0.5`.
pub fn py_pow<L, R>(a: L, b: R) -> L::Output
where
    L: PyPow<R>,
{
    a.py_pow(b)
}

/// Python's two-argument pow() builtin — same semantics as `**`.
pub fn pow<L, R>(a: L, b: R) -> L::Output
where
    L: PyPow<R>,
{
    a.py_pow(b)
}

/// Python pow(base, exp, mod): modular exponentiation, with the modular
/// inverse for negative exponents (Python 3.8+). The result takes the
/// modulus's sign, like Python's floored `%`.
pub fn pow_mod(base: i64, exp: i64, modulus: i64) -> Result<i64, PyException> {
    if modulus == 0 {
        return Err(PyException::new(
            "ValueError",
            "pow() 3rd argument cannot be 0",
        ));
    }
    let m = (modulus as i128).abs();
    let mut b = (base as i128).rem_euclid(m);
    let mut e = exp as i128;
    if e < 0 {
        // Invert base mod m via extended gcd; only units are invertible.
        let (g, x) = egcd(b, m);
        if g != 1 {
            return Err(PyException::new(
                "ValueError",
                "base is not invertible for the given modulus",
            ));
        }
        b = x.rem_euclid(m);
        e = -e;
    }
    let mut result: i128 = 1 % m;
    while e > 0 {
        if e & 1 == 1 {
            result = result * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    // Fold the non-negative residue onto the modulus's sign: Python's
    // pow(5, 3, -7) is -1, not 6.
    let signed = if modulus < 0 && result != 0 {
        result - m
    } else {
        result
    };
    Ok(signed as i64)
}

/// Extended gcd on non-negative i128s: returns (g, x) with a*x ≡ g (mod b).
fn egcd(a: i128, b: i128) -> (i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_x, mut x) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_x, x) = (x, old_x - q * x);
    }
    (old_r, old_x)
}

/// Python round() builtin with no ndigits: rounds half to even (banker's
/// rounding), so `round(2.5) == 2` and `round(3.5) == 4`.
pub fn round(value: f64) -> i64 {
    let r = flt::round(value);
    if flt::abs(value - flt::trunc(value)) == 0.5 && r % 2.0 != 0.0 {
        (r - flt::signum(value)) as i64
    } else {
        r as i64
    }
}

/// Python round(value, ndigits): rounds half to even at the given decimal
/// place and returns a float.
///
/// CPython rounds the *correctly rounded decimal expansion*; the naive
/// `value * 10^n; round; / 10^n` introduces a rounding error before the
/// half-even fixup (round(1.15, 1) must be 1.1, not 1.2). Rust's float
/// formatting performs exact correctly-rounded decimal rendering at the
/// requested precision with ties to even, so format-then-parse reproduces
/// CPython's results (verified over a 46-value × 9-ndigits sweep).
pub fn round_digits(value: f64, ndigits: i64) -> f64 {
    if !value.is_finite() {
        // round(inf, n) is inf, round(nan, n) is nan.
        return value;
    }
    if ndigits >= 0 {
        // Cap the precision: floats carry ~17 significant digits, so
        // beyond a few hundred decimals the expansion is all zeros and
        // parses back to the same value. The cap keeps the formatted
        // string from becoming pathologically large.
        let n = (ndigits as u64).min(400) as usize;
        return format!("{:.*}", n, value).parse().unwrap_or(value);
    }
    // Negative ndigits: round at a power of ten (round(1250.0, -2) ->
    // 1200.0, round(15.0, -1) -> 20.0). Scale down, round to an integer
    // (half-even via the format path), scale back.
    let magnitude = ndigits.unsigned_abs();
    if magnitude > 308 {
        // Far below the smallest subnormal: every finite value rounds to
        // zero (round(1e308, -400) == 0.0).
        return 0.0;
    }
    let factor = flt::powi(10f64, magnitude as i32);
    let scaled = value / factor;
    let r: f64 = format!("{:.0}", scaled).parse().unwrap_or(scaled);
    r * factor
}

/// Python ord() builtin: code point of a one-character string.
pub fn ord<S: AsRef<str>>(c: S) -> i64 {
    let s = c.as_ref();
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => ch as i64,
        _ => panic!(
            "ord() expected a character, but string of length {} found",
            s.chars().count()
        ),
    }
}

/// Python chr() builtin: one-character string for a code point.
///
/// Out-of-range arguments raise the same ValueError as CPython. Lone
/// surrogate code points (U+D800–U+DFFF) succeed in CPython but cannot
/// be represented in a Rust `String` (UTF-8 has no surrogate code
/// points), so they raise a catchable ValueError instead of panicking.
pub fn chr(code: i64) -> Result<String, PyException> {
    if !(0..=0x10FFFF).contains(&code) {
        return Err(PyException::new(
            "ValueError",
            "chr() arg not in range(0x110000)",
        ));
    }
    match char::from_u32(code as u32) {
        Some(c) => Ok(String::from(c)),
        None => Err(PyException::new(
            "ValueError",
            format!(
                "chr() arg is a surrogate code point U+{:04X}; lone surrogates \
                 are not representable in UTF-8",
                code
            ),
        )),
    }
}

/// Python hex() builtin: `hex(255) == "0xff"`, `hex(-255) == "-0xff"`.
pub fn hex(n: i64) -> String {
    if n < 0 {
        format!("-0x{:x}", n.unsigned_abs())
    } else {
        format!("0x{:x}", n)
    }
}

/// Python oct() builtin.
pub fn oct(n: i64) -> String {
    if n < 0 {
        format!("-0o{:o}", n.unsigned_abs())
    } else {
        format!("0o{:o}", n)
    }
}

/// Python bin() builtin.
pub fn bin(n: i64) -> String {
    if n < 0 {
        format!("-0b{:b}", n.unsigned_abs())
    } else {
        format!("0b{:b}", n)
    }
}

// Implementations for PyAbs trait
impl PyAbs for i64 {
    type Output = i64;
    fn py_abs(self) -> Self::Output {
        // Python promotes to bigint; i64 cannot, so the boundary case is a
        // defined, loud failure in every build profile (release previously
        // WRAPPED to a negative value silently).
        self.checked_abs().unwrap_or_else(|| {
            panic!(
                "{}",
                PyException::new("OverflowError", "abs(i64::MIN) overflows i64")
            )
        })
    }
}

impl PyAbs for i32 {
    type Output = i32;
    fn py_abs(self) -> Self::Output {
        self.abs()
    }
}

impl PyAbs for f64 {
    type Output = f64;
    fn py_abs(self) -> Self::Output {
        self.abs()
    }
}

impl PyAbs for f32 {
    type Output = f32;
    fn py_abs(self) -> Self::Output {
        self.abs()
    }
}

// Implementations for PySum: owned and borrowed list forms per numeric
// scalar (generated call sites pass whichever the expression yields —
// an owned Vec local, a &Vec through a reference, a slice view).
macro_rules! pysum_numeric {
    ($($t:ty),* $(,)?) => {
        $(
            impl PySum for Vec<$t> {
                type Output = $t;
                fn py_sum(self) -> $t {
                    self.iter().sum()
                }
            }
            impl PySum for &Vec<$t> {
                type Output = $t;
                fn py_sum(self) -> $t {
                    self.iter().sum()
                }
            }
            impl PySum for &[$t] {
                type Output = $t;
                fn py_sum(self) -> $t {
                    self.iter().sum()
                }
            }
        )*
    };
}
// i64/f64 only (rython's numeric types): an i32/f32 impl would leave
// `sum(vec![1, 2, 3])` ambiguous over {integer} and rustc's i32 literal
// fallback would then contradict an i64 use context.
pysum_numeric!(i64, f64);

// Python sum() of a bool list counts the Trues (bool ⊂ int).
impl PySum for Vec<bool> {
    type Output = i64;
    fn py_sum(self) -> i64 {
        self.iter().filter(|b| **b).count() as i64
    }
}
impl PySum for &Vec<bool> {
    type Output = i64;
    fn py_sum(self) -> i64 {
        self.iter().filter(|b| **b).count() as i64
    }
}

impl<T> PySum for &PyList<T>
where
    T: core::iter::Sum<T> + Clone,
{
    type Output = T;
    fn py_sum(self) -> T {
        self.inner.iter().cloned().sum()
    }
}

/// Python all() function - returns True if all elements are truthy
pub fn all<T: Truthy>(iterable: &[T]) -> bool {
    iterable.iter().all(|x| x.is_truthy())
}

/// Python any() function - returns True if any element is truthy
pub fn any<T: Truthy>(iterable: &[T]) -> bool {
    iterable.iter().any(|x| x.is_truthy())
}

/// Python enumerate() function - returns iterator with index and value
/// pairs. The index is an i64 because Python's is an int, and generated
/// arithmetic on it must not need casts.
pub fn enumerate<T>(iterable: Vec<T>) -> Vec<(i64, T)> {
    iterable
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i as i64, x))
        .collect()
}

/// Python enumerate(iterable, start=n).
pub fn enumerate_start<T>(iterable: Vec<T>, start: i64) -> Vec<(i64, T)> {
    iterable
        .into_iter()
        .enumerate()
        .map(|(i, x)| (start + i as i64, x))
        .collect()
}

/// Python map(f, iterable), materialized. This form takes an infallible
/// function (a lambda); calls through user-defined functions — which
/// return Result — lower to map_fallible.
pub fn map<T, U, F: FnMut(T) -> U>(f: F, iterable: Vec<T>) -> Vec<U> {
    iterable.into_iter().map(f).collect()
}

/// map(f, iterable) where f can raise: the first exception propagates,
/// exactly like Python's lazy map surfacing the error at iteration.
pub fn map_fallible<T, U, F: FnMut(T) -> Result<U, PyException>>(
    f: F,
    iterable: Vec<T>,
) -> Result<Vec<U>, PyException> {
    iterable.into_iter().map(f).collect()
}

/// Python map(f, a, b): pairs up to the shortest, like zip.
pub fn map2<A, B, U, F: FnMut(A, B) -> U>(mut f: F, a: Vec<A>, b: Vec<B>) -> Vec<U> {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| f(x, y))
        .collect()
}

/// Python filter(f, iterable), materialized. The predicate receives each
/// element by value (cloned), so lambda bodies compare naturally.
pub fn filter<T: Clone, F: FnMut(T) -> bool>(mut f: F, iterable: Vec<T>) -> Vec<T> {
    iterable.into_iter().filter(|x| f(x.clone())).collect()
}

/// filter(f, iterable) where the predicate can raise.
pub fn filter_fallible<T: Clone, F: FnMut(T) -> Result<bool, PyException>>(
    mut f: F,
    iterable: Vec<T>,
) -> Result<Vec<T>, PyException> {
    let mut out = Vec::new();
    for x in iterable {
        if f(x.clone())? {
            out.push(x);
        }
    }
    Ok(out)
}

/// Python filter(None, iterable): keep the truthy elements.
pub fn filter_truthy<T: Truthy>(iterable: Vec<T>) -> Vec<T> {
    iterable.into_iter().filter(Truthy::is_truthy).collect()
}

/// What Python's list() builtin accepts: already-material sequences pass
/// through, strings explode into their characters (as one-char strings,
/// like Python), ranges materialize.
pub trait PyListFrom {
    type Item;
    fn py_list(self) -> Vec<Self::Item>;
}

impl<T> PyListFrom for Vec<T> {
    type Item = T;
    fn py_list(self) -> Vec<T> {
        self
    }
}

impl PyListFrom for &str {
    type Item = String;
    fn py_list(self) -> Vec<String> {
        self.chars().map(|c| c.to_string()).collect()
    }
}

impl PyListFrom for String {
    type Item = String;
    fn py_list(self) -> Vec<String> {
        self.as_str().py_list()
    }
}

impl PyListFrom for PyRange {
    type Item = i64;
    fn py_list(self) -> Vec<i64> {
        self.collect()
    }
}

/// Python list() builtin.
pub fn list<L: PyListFrom>(x: L) -> Vec<L::Item> {
    x.py_list()
}

/// Python zip() function - combines multiple iterables
pub fn zip<T, U>(iter1: Vec<T>, iter2: Vec<U>) -> Vec<(T, U)> {
    iter1.into_iter().zip(iter2.into_iter()).collect()
}

/// Python's range object: LAZY, like Python's — `for i in range(10**9)`
/// iterates in O(1) memory where the old Vec materialization allocated
/// gigabytes. Iterating yields i64s; len/contains follow Python range
/// semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyRange {
    next: i64,
    stop: i64,
    step: i64,
}

impl Iterator for PyRange {
    type Item = i64;
    fn next(&mut self) -> Option<i64> {
        let more = if self.step > 0 {
            self.next < self.stop
        } else {
            self.next > self.stop
        };
        if !more {
            return None;
        }
        let v = self.next;
        // Saturating: the value AFTER the final element may not be
        // representable (e.g. range up to i64::MAX); saturation still
        // terminates the loop correctly.
        self.next = self.next.saturating_add(self.step);
        Some(v)
    }
}

impl PyRange {
    /// Python len(range(...)): the number of elements remaining. Computed
    /// in i128 so extreme endpoints and steps (spans near i64::MAX,
    /// step == i64::MIN) can never overflow.
    pub fn py_len(&self) -> usize {
        let (span, step) = if self.step > 0 {
            (
                self.stop as i128 - self.next as i128,
                self.step as i128,
            )
        } else {
            (
                self.next as i128 - self.stop as i128,
                -(self.step as i128),
            )
        };
        if span <= 0 {
            0
        } else {
            ((span - 1) / step + 1) as usize
        }
    }

    /// Python `x in range(...)`: O(1) membership (i128 keeps the
    /// difference overflow-free for extreme endpoints).
    pub fn py_contains(&self, value: &i64) -> bool {
        let v = *value;
        let in_span = if self.step > 0 {
            v >= self.next && v < self.stop
        } else {
            v <= self.next && v > self.stop
        };
        in_span && (v as i128 - self.next as i128) % (self.step as i128) == 0
    }
}

impl Len for PyRange {
    fn len(&self) -> usize {
        self.py_len()
    }
}

/// Python range() function - a lazy range of numbers.
pub fn range(stop: i64) -> PyRange {
    PyRange { next: 0, stop, step: 1 }
}

pub fn range_start_stop(start: i64, stop: i64) -> PyRange {
    PyRange { next: start, stop, step: 1 }
}

/// range(start, stop, step): a zero step raises ValueError, as in Python.
pub fn range_start_stop_step(start: i64, stop: i64, step: i64) -> Result<PyRange, PyException> {
    if step == 0 {
        return Err(PyException::new("ValueError", "range() arg 3 must not be zero"));
    }
    Ok(PyRange { next: start, stop, step })
}

// ============================================================================
// PYTHON TYPE CONVERSION TRAITS
// ============================================================================

/// Trait for Python-style boolean conversion
pub trait PyBool {
    fn py_bool(self) -> bool;
}

/// Trait for Python-style integer conversion
pub trait PyInt {
    fn py_int(self) -> Result<i64, PyException>;
}

/// Trait for Python-style float conversion
pub trait PyFloat {
    fn py_float(self) -> Result<f64, PyException>;
}

/// Trait for Python-style string conversion
pub trait PyToString {
    fn py_str(self) -> String;
}

/// Python bool() function - converts to boolean
pub fn bool<T: PyBool>(x: T) -> bool {
    x.py_bool()
}

/// Python int() function - converts to integer
pub fn int<T: PyInt>(x: T) -> Result<i64, PyException> {
    x.py_int()
}

/// Python float() function - converts to float  
pub fn float<T: PyFloat>(x: T) -> Result<f64, PyException> {
    x.py_float()
}

/// Python str() function - converts to string
pub fn str<T: PyToString>(x: T) -> String {
    x.py_str()
}

// PyBool implementations
impl PyBool for i64 {
    fn py_bool(self) -> bool {
        self != 0
    }
}

impl PyBool for f64 {
    fn py_bool(self) -> bool {
        self != 0.0
    }
}

impl PyBool for &str {
    fn py_bool(self) -> bool {
        !self.is_empty()
    }
}

impl PyBool for String {
    fn py_bool(self) -> bool {
        !self.is_empty()
    }
}

impl PyBool for bool {
    fn py_bool(self) -> bool {
        self
    }
}

impl<T> PyBool for &PyList<T> {
    fn py_bool(self) -> bool {
        !self.inner.is_empty()
    }
}

impl<K, V> PyBool for &PyDictionary<K, V> 
where
    K: Eq + Hash,
{
    fn py_bool(self) -> bool {
        !self.inner.is_empty()
    }
}

impl PyBool for &PyStr {
    fn py_bool(self) -> bool {
        !self.inner.is_empty()
    }
}

// PyInt implementations
impl PyInt for &str {
    fn py_int(self) -> Result<i64, PyException> {
        // Python strips surrounding whitespace and accepts `_` digit
        // separators, so int(line) over a file's lines works; Rust's
        // parse() rejects both.
        let cleaned = self.trim().replace('_', "");
        cleaned
            .parse()
            .map_err(|_| value_error(&format!("invalid literal for int(): '{}'", self)))
    }
}

impl PyInt for String {
    fn py_int(self) -> Result<i64, PyException> {
        self.as_str().py_int()
    }
}

impl PyInt for f64 {
    fn py_int(self) -> Result<i64, PyException> {
        // `as` saturates and turns NaN into 0; Python raises for both.
        if self.is_nan() {
            return Err(value_error("cannot convert float NaN to integer"));
        }
        if self.is_infinite() {
            return Err(PyException::new(
                "OverflowError",
                "cannot convert float infinity to integer",
            ));
        }
        Ok(self as i64)
    }
}

impl PyInt for bool {
    fn py_int(self) -> Result<i64, PyException> {
        Ok(if self { 1 } else { 0 })
    }
}

impl PyInt for i64 {
    fn py_int(self) -> Result<i64, PyException> {
        Ok(self)
    }
}

/// The type-level inheritance tree. The converter emits one
/// `impl PyInherits<Ancestor> for Class` per (class, ancestor) pair in a
/// generated crate — reflexive and transitive along the single-inheritance
/// chain — so generic Rust code can bound on Python ancestry
/// (`fn pet<T: PyInherits<Animal>>(x: T)`). The entries are derived from
/// the same base-chain walk the conversion-time isinstance folding uses,
/// keeping the type-level and conversion-time trees in lockstep.
pub trait PyInherits<Base> {}

impl PyInt for u8 {
    // A bytes element (`data[i]`) is a `u8` in the value model, but in
    // Python it is already an int — so `int(data[i])` is the identity
    // conversion, widening the byte into the program's int type.
    fn py_int(self) -> Result<i64, PyException> {
        Ok(self as i64)
    }
}

// PyFloat implementations
impl PyFloat for &str {
    fn py_float(self) -> Result<f64, PyException> {
        let cleaned = self.trim().replace('_', "");
        cleaned
            .parse()
            .map_err(|_| value_error(&format!("could not convert string to float: '{}'", self)))
    }
}

impl PyFloat for String {
    fn py_float(self) -> Result<f64, PyException> {
        self.as_str().py_float()
    }
}

impl PyFloat for i64 {
    fn py_float(self) -> Result<f64, PyException> {
        Ok(self as f64)
    }
}

impl PyFloat for f64 {
    fn py_float(self) -> Result<f64, PyException> {
        Ok(self)
    }
}

// PyToString implementations
impl PyToString for i64 {
    fn py_str(self) -> String {
        self.to_string()
    }
}

impl PyToString for f64 {
    fn py_str(self) -> String {
        // Python 3's str(float) IS repr(float).
        py_float_repr(self)
    }
}

/// Python's float repr, exactly: shortest round-trip digits (which Rust's
/// Display also produces), rendered positionally for decimal exponents in
/// [-4, 16) and as `d.dddde±EE` outside — Python prints 1e16 as "1e+16"
/// where Rust's Display (which never uses exponent form) prints
/// "10000000000000000".
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let display = format!("{}", x);
    let (sign, body) = match display.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", display.as_str()),
    };
    let (int_part, frac_part) = match body.split_once('.') {
        Some((i, f)) => (i, f),
        None => (body, ""),
    };
    // Decimal exponent of the leading significant digit.
    let (exp, digits): (i64, String) = if int_part != "0" {
        (
            int_part.len() as i64 - 1,
            format!("{}{}", int_part, frac_part),
        )
    } else {
        let zeros = frac_part.len() - frac_part.trim_start_matches('0').len();
        (
            -(zeros as i64) - 1,
            frac_part.trim_start_matches('0').to_string(),
        )
    };
    if (-4..16).contains(&exp) {
        return if frac_part.is_empty() {
            format!("{}{}.0", sign, int_part)
        } else {
            format!("{}{}", sign, body)
        };
    }
    let digits = digits.trim_end_matches('0');
    let mantissa = if digits.len() > 1 {
        format!("{}.{}", &digits[..1], &digits[1..])
    } else {
        digits.to_string()
    };
    format!("{}{}e{}{:02}", sign, mantissa, if exp < 0 { "-" } else { "+" }, exp.abs())
}

/// Python's repr() for the types generated code produces. str gets
/// Python's quoting rules (single quotes unless the text contains a single
/// quote and no double quote); containers recurse.
pub trait PyRepr {
    fn py_repr(&self) -> String;
}

// Every integer primitive reprs like a Python int, matching PyDisplay's
// coverage: len() yields usize, and an integer literal among several
// candidate impls falls back to i32, so a container of either must still
// be printable.
macro_rules! py_repr_int {
    ($($t:ty),*) => {$(
        impl PyRepr for $t {
            fn py_repr(&self) -> String {
                self.to_string()
            }
        }
    )*};
}
py_repr_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl PyRepr for f64 {
    fn py_repr(&self) -> String {
        py_float_repr(*self)
    }
}

impl PyRepr for bool {
    fn py_repr(&self) -> String {
        if *self { "True" } else { "False" }.to_string()
    }
}

impl PyRepr for str {
    fn py_repr(&self) -> String {
        py_str_repr(self)
    }
}

impl PyRepr for String {
    fn py_repr(&self) -> String {
        py_str_repr(self)
    }
}

impl<T: PyRepr + ?Sized> PyRepr for &T {
    fn py_repr(&self) -> String {
        (**self).py_repr()
    }
}

impl<T: PyRepr> PyRepr for Vec<T> {
    fn py_repr(&self) -> String {
        let items: Vec<String> = self.iter().map(|x| x.py_repr()).collect();
        format!("[{}]", items.join(", "))
    }
}

// Python tuples: ('a', 1). Rust tuples back m.span() and the
// findall2/findall3 result shapes; str(tuple) is repr(tuple), elements
// included, so PyDisplay defers to PyRepr.
impl<A: PyRepr, B: PyRepr> PyRepr for (A, B) {
    fn py_repr(&self) -> String {
        format!("({}, {})", self.0.py_repr(), self.1.py_repr())
    }
}

impl<A: PyRepr, B: PyRepr, C: PyRepr> PyRepr for (A, B, C) {
    fn py_repr(&self) -> String {
        format!(
            "({}, {}, {})",
            self.0.py_repr(),
            self.1.py_repr(),
            self.2.py_repr()
        )
    }
}

/// Python dict repr: `{'a': 1, 'b': 2}` — keys AND values both render
/// with repr, and insertion order is preserved (IndexMap matches
/// Python's dict ordering guarantee, so the rendering is faithful, not
/// merely plausible). An empty dict is `{}`.
///
/// Sets deliberately have no repr: their iteration order is arbitrary
/// in both languages and cannot be made to agree, so printing one would
/// silently diverge from CPython. It stays a loud compile error instead.
impl<K: PyRepr, V: PyRepr> PyRepr for PyDict<K, V> {
    fn py_repr(&self) -> String {
        let items: Vec<String> = self
            .iter()
            .map(|(k, v)| format!("{}: {}", k.py_repr(), v.py_repr()))
            .collect();
        format!("{{{}}}", items.join(", "))
    }
}

impl<K: PyRepr, V: PyRepr> PyDisplay for PyDict<K, V> {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

impl<A: PyRepr, B: PyRepr> PyDisplay for (A, B) {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

impl<A: PyRepr, B: PyRepr, C: PyRepr> PyDisplay for (A, B, C) {
    fn py_display(&self) -> String {
        self.py_repr()
    }
}

/// In the Option-based None model, None reprs as Python's None and a
/// present value reprs as itself.
impl<T: PyRepr> PyRepr for Option<T> {
    fn py_repr(&self) -> String {
        match self {
            Some(x) => x.py_repr(),
            None => "None".to_string(),
        }
    }
}

/// Python's str repr: preferred single quotes (double when the text holds
/// a single quote but no double quote), backslash/newline/return/tab
/// escapes, and \xNN/\uNNNN/\UNNNNNNNN for everything failing
/// str.isprintable() — controls, format characters, line/paragraph
/// separators, and non-ASCII spaces (so repr("\xa0") is '\xa0', not a
/// literal NBSP).
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if repr_escapes(c) => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{:02x}", cp));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{:04x}", cp));
                } else {
                    out.push_str(&format!("\\U{:08x}", cp));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python repr() builtin.
pub fn repr<T: PyRepr + ?Sized>(x: &T) -> String {
    x.py_repr()
}

/// Python ascii() builtin: repr(), with every code point outside
/// printable ASCII escaped to `\xXX`, `\uXXXX` or `\UXXXXXXXX` (lowercase
/// hex). Printable ASCII (`0x20..=0x7e`) passes through untouched —
/// escaping anything else is repr's job already (`ascii("\n")` ==
/// `"'\\n'"`, DEL escapes: `ascii(chr(0x7f))` == `"'\\x7f'"`). Verified
/// against python3 3.14.
pub fn ascii<T: PyRepr + ?Sized>(x: &T) -> String {
    let mut out = String::with_capacity(x.py_repr().len() + 8);
    for ch in x.py_repr().chars() {
        match ch {
            ' '..='~' => out.push(ch),
            c if (c as u32) <= 0xff => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if (c as u32) <= 0xffff => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => {
                out.push_str(&format!("\\U{:08x}", c as u32));
            }
        }
    }
    out
}

// ============================================================================
// hash(): CPython's algorithms with PYTHONHASHSEED=0
// ============================================================================

/// The modulus for numeric hashes: 2^61 - 1 (sys.hash_info.modulus).
const PY_HASH_MODULUS: u64 = (1 << 61) - 1;

/// Python's hash() with hash randomization DISABLED (PYTHONHASHSEED=0):
/// deterministic and verifiable against `PYTHONHASHSEED=0 python3`. A
/// randomized CPython session produces different string hashes — that is
/// the documented divergence of choosing determinism.
pub trait PyHash {
    fn py_hash(&self) -> i64;
}

fn fixup_minus_one(h: i64) -> i64 {
    // -1 is CPython's internal error marker, so no object hashes to it.
    if h == -1 { -2 } else { h }
}

impl PyHash for i64 {
    fn py_hash(&self) -> i64 {
        let n = *self;
        let m = PY_HASH_MODULUS as i128;
        let r = (n as i128).rem_euclid(m);
        let h = if n < 0 && r != 0 { r - m } else { r } as i64;
        fixup_minus_one(h)
    }
}

impl PyHash for bool {
    fn py_hash(&self) -> i64 {
        if *self { 1 } else { 0 }
    }
}

impl PyHash for f64 {
    /// CPython's _Py_HashDouble: 28 bits at a time, modular in 2^61-1.
    fn py_hash(&self) -> i64 {
        let v = *self;
        if v.is_nan() {
            // Python 3.10+ hashes NaN by object identity — inherently
            // nondeterministic, so it fails loudly here.
            panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "hash(nan) is identity-based in Python and cannot be \
                     reproduced deterministically",
                )
            );
        }
        if v.is_infinite() {
            return if v > 0.0 { 314159 } else { -314159 };
        }
        let sign = if v < 0.0 { -1i64 } else { 1i64 };
        let mut m = flt::abs(v);
        // frexp: m in [0.5, 1), v = m * 2^e.
        let mut e = 0i32;
        if m != 0.0 {
            while m >= 1.0 {
                m /= 2.0;
                e += 1;
            }
            while m < 0.5 {
                m *= 2.0;
                e -= 1;
            }
        }
        let p = PY_HASH_MODULUS;
        let mut x: u64 = 0;
        while m != 0.0 {
            x = ((x << 28) & p) | (x >> (61 - 28));
            m *= 268435456.0; // 2^28
            e -= 28;
            let y = m as u64;
            m -= y as f64;
            x += y;
            if x >= p {
                x -= p;
            }
        }
        let e = if e >= 0 { e % 61 } else { 61 - 1 - ((-1 - e) % 61) } as u32;
        x = ((x << e) & p) | (x >> (61 - e));
        fixup_minus_one(x as i64 * sign)
    }
}

/// siphash13 with a zero key — CPython's string hash under
/// PYTHONHASHSEED=0 (sys.hash_info.algorithm == "siphash13").
fn siphash13(data: &[u8]) -> u64 {
    fn rotl(x: u64, b: u32) -> u64 {
        x.rotate_left(b)
    }
    let (k0, k1) = (0u64, 0u64);
    let mut v0 = 0x736f6d6570736575u64 ^ k0;
    let mut v1 = 0x646f72616e646f6du64 ^ k1;
    let mut v2 = 0x6c7967656e657261u64 ^ k0;
    let mut v3 = 0x7465646279746573u64 ^ k1;
    let b: u64 = (data.len() as u64) << 56;

    let round = |v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64| {
        *v0 = v0.wrapping_add(*v1);
        *v1 = rotl(*v1, 13);
        *v1 ^= *v0;
        *v0 = rotl(*v0, 32);
        *v2 = v2.wrapping_add(*v3);
        *v3 = rotl(*v3, 16);
        *v3 ^= *v2;
        *v0 = v0.wrapping_add(*v3);
        *v3 = rotl(*v3, 21);
        *v3 ^= *v0;
        *v2 = v2.wrapping_add(*v1);
        *v1 = rotl(*v1, 17);
        *v1 ^= *v2;
        *v2 = rotl(*v2, 32);
    };

    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let mi = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        v3 ^= mi;
        round(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= mi;
    }
    let mut t: u64 = 0;
    let rest = chunks.remainder();
    for (i, byte) in rest.iter().enumerate() {
        t |= (*byte as u64) << (8 * i);
    }
    let b = b | t;
    v3 ^= b;
    round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= b;
    v2 ^= 0xff;
    round(&mut v0, &mut v1, &mut v2, &mut v3);
    round(&mut v0, &mut v1, &mut v2, &mut v3);
    round(&mut v0, &mut v1, &mut v2, &mut v3);
    (v0 ^ v1) ^ (v2 ^ v3)
}

impl PyHash for str {
    fn py_hash(&self) -> i64 {
        if self.is_empty() {
            return 0;
        }
        // CPython hashes the string's INTERNAL representation, which is
        // the narrowest of Latin-1 / UCS-2 / UCS-4 that fits — not UTF-8.
        let max = self.chars().map(|c| c as u32).max().unwrap_or(0);
        let bytes: Vec<u8> = if max < 0x100 {
            self.chars().map(|c| c as u8).collect()
        } else if max < 0x10000 {
            self.chars()
                .flat_map(|c| (c as u16).to_le_bytes())
                .collect()
        } else {
            self.chars()
                .flat_map(|c| (c as u32).to_le_bytes())
                .collect()
        };
        fixup_minus_one(siphash13(&bytes) as i64)
    }
}

impl PyHash for String {
    fn py_hash(&self) -> i64 {
        self.as_str().py_hash()
    }
}

impl<T: PyHash + ?Sized> PyHash for &T {
    fn py_hash(&self) -> i64 {
        (**self).py_hash()
    }
}

/// Python hash() builtin (PYTHONHASHSEED=0 semantics).
pub fn hash<T: PyHash + ?Sized>(x: &T) -> i64 {
    x.py_hash()
}

impl PyToString for bool {
    fn py_str(self) -> String {
        if self { "True".to_string() } else { "False".to_string() }
    }
}

impl PyToString for &str {
    fn py_str(self) -> String {
        self.to_string()
    }
}

impl PyToString for String {
    fn py_str(self) -> String {
        self
    }
}

// CPython's str(exc) is the exception's args rendered as a string — for a
// `ZeroDivisionError("division by zero")` that is just "division by zero",
// not "Type: message" (that is Display's job for the uncaught-exception
// report).
impl PyToString for PyException {
    fn py_str(self) -> String {
        self.message
    }
}

// ============================================================================
// PYTHON BUILT-IN TYPES AND TRAITS
// ============================================================================

/// Trait for objects that have a length
pub trait Len {
    fn len(&self) -> usize;
}

// References measure like their referents, so len() works on the &T
// elements that key-function closures (min/max/sorted key=) receive.
impl<T: Len + ?Sized> Len for &T {
    fn len(&self) -> usize {
        (**self).len()
    }
}

/// Trait for objects that can be evaluated for truthiness
pub trait Truthy {
    fn is_truthy(&self) -> bool;
}

/// Python-style string type with all string methods
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyStr {
    inner: String,
}

impl PyStr {
    pub fn new(s: impl Into<String>) -> Self {
        Self { inner: s.into() }
    }
    
    /// Python str.split() method
    pub fn split(&self, sep: Option<&str>) -> Vec<PyStr> {
        match sep {
            Some(separator) => self.inner.split(separator)
                .map(|s| PyStr::new(s))
                .collect(),
            None => self.inner.split(py_is_whitespace)
                .filter(|s| !s.is_empty())
                .map(|s| PyStr::new(s))
                .collect(),
        }
    }
    
    /// Python str.join() method
    pub fn join(&self, iterable: &[PyStr]) -> PyStr {
        let strings: Vec<&str> = iterable.iter().map(|s| s.inner.as_str()).collect();
        PyStr::new(strings.join(&self.inner))
    }
    
    /// Python str.strip() method
    pub fn strip(&self) -> PyStr {
        PyStr::new(self.inner.trim_matches(py_is_whitespace).to_string())
    }
    
    /// Python str.lower() method
    pub fn lower(&self) -> PyStr {
        PyStr::new(self.inner.to_lowercase())
    }
    
    /// Python str.upper() method
    pub fn upper(&self) -> PyStr {
        PyStr::new(self.inner.to_uppercase())
    }
    
    /// Python str.replace() method
    pub fn replace<O: AsRef<str>, N: AsRef<str>>(&self, old: O, new: N) -> PyStr {
        PyStr::new(self.inner.replace(old.as_ref(), new.as_ref()))
    }
    
    /// Python str.startswith() method
    pub fn startswith<P: AsRef<str>>(&self, prefix: P) -> bool {
        self.inner.starts_with(prefix.as_ref())
    }
    
    /// Python str.endswith() method
    pub fn endswith<S: AsRef<str>>(&self, suffix: S) -> bool {
        self.inner.ends_with(suffix.as_ref())
    }
    
    /// Python str.find() method: CHARACTER index (consistent with len and
    /// with PyStrOps::py_find), not a byte offset.
    pub fn find<S: AsRef<str>>(&self, sub: S) -> i64 {
        match self.inner.find(sub.as_ref()) {
            Some(pos) => self.inner[..pos].chars().count() as i64,
            None => -1,
        }
    }
    
    /// Python str.count() method
    pub fn count<S: AsRef<str>>(&self, sub: S) -> usize {
        self.inner.matches(sub.as_ref()).count()
    }
    
    /// Python str.format() method (basic implementation)
    pub fn format(&self, args: &[&str]) -> PyStr {
        let mut result = self.inner.clone();
        for (i, arg) in args.iter().enumerate() {
            result = result.replace(&format!("{{{}}}", i), arg);
        }
        PyStr::new(result)
    }
    
    /// Access inner string
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl Display for PyStr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl PyDisplay for PyStr {
    fn py_display(&self) -> String {
        self.inner.clone()
    }
}

impl Len for PyStr {
    fn len(&self) -> usize {
        self.inner.chars().count()
    }
}

impl Truthy for PyStr {
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

/// The `str | bytes` (and `str | bytes | bytearray`) heterogeneous union:
/// a value that is either a Python str or raw bytes. This is the bounded
/// slice of the boxed-heterogeneous-value divergence (issue #121) that
/// real libraries need (idna's labels, requests' `to_native_string`).
/// Codegen narrows it through `isinstance(x, (bytes, bytearray))` checks
/// into the concrete String/Vec<u8> branch; the union itself only needs
/// len, truthiness, and the isinstance/bytes/str dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl StrOrBytes {
    pub fn is_str(&self) -> bool {
        matches!(self, StrOrBytes::Str(_))
    }
    pub fn is_bytes(&self) -> bool {
        matches!(self, StrOrBytes::Bytes(_))
    }
    /// Python len(): characters for a str, octets for bytes.
    pub fn len(&self) -> usize {
        match self {
            StrOrBytes::Str(s) => s.chars().count(),
            StrOrBytes::Bytes(b) => b.len(),
        }
    }
    /// The str view; only valid after isinstance narrowing (Python has no
    /// str(bytes) without an encoding).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            StrOrBytes::Str(s) => Some(s.as_str()),
            StrOrBytes::Bytes(_) => None,
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            StrOrBytes::Bytes(b) => Some(b.as_slice()),
            StrOrBytes::Str(_) => None,
        }
    }
}

impl Truthy for StrOrBytes {
    fn is_truthy(&self) -> bool {
        self.len() != 0
    }
}

impl Len for StrOrBytes {
    fn len(&self) -> usize {
        StrOrBytes::len(self)
    }
}

impl From<String> for StrOrBytes {
    fn from(value: String) -> Self {
        StrOrBytes::Str(value)
    }
}

impl From<&str> for StrOrBytes {
    fn from(value: &str) -> Self {
        StrOrBytes::Str(value.to_string())
    }
}

impl From<Vec<u8>> for StrOrBytes {
    fn from(value: Vec<u8>) -> Self {
        StrOrBytes::Bytes(value)
    }
}

impl From<&[u8]> for StrOrBytes {
    fn from(value: &[u8]) -> Self {
        StrOrBytes::Bytes(value.to_vec())
    }
}

// A bytes literal renders as &[u8; N]: the fixed-size array needs its own
// From (unsized coercion does not fire through a generic into()).
impl<const N: usize> From<&[u8; N]> for StrOrBytes {
    fn from(value: &[u8; N]) -> Self {
        StrOrBytes::Bytes(value.to_vec())
    }
}

/// CPython's bytes repr: `b'...'`, switching to DOUBLE quotes when the
/// content contains a single quote and no double quote (`b"a'b"`); `\n`,
/// `\r`, `\t` and backslash are named escapes, every other byte outside
/// printable ASCII is `\xNN` (lowercase hex — including `\a`/`\b`/`\f`/
/// `\v` and DEL). Verified against python3 3.14.
pub fn py_bytes_repr(bytes: &[u8]) -> String {
    let has_single = bytes.contains(&b'\'');
    let has_double = bytes.contains(&b'"');
    let (open, close) = if has_single && !has_double {
        ('"', '"')
    } else {
        ('\'', '\'')
    };
    let mut out = String::with_capacity(bytes.len() + 3);
    out.push('b');
    out.push(open);
    for &b in bytes {
        match b {
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b'\\' => out.push_str("\\\\"),
            b if b == open as u8 => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(b as char),
            other => out.push_str(&format!("\\x{:02x}", other)),
        }
    }
    out.push(close);
    out
}

impl PyToString for StrOrBytes {
    fn py_str(self) -> String {
        match self {
            StrOrBytes::Str(s) => s,
            StrOrBytes::Bytes(b) => py_bytes_repr(&b),
        }
    }
}

impl PyDisplay for StrOrBytes {
    fn py_display(&self) -> String {
        match self {
            StrOrBytes::Str(s) => s.clone(),
            StrOrBytes::Bytes(b) => py_bytes_repr(b),
        }
    }
}

/// A boxed heterogeneous Python value (issue #121): the runtime
/// representation of wider unions that have no single concrete Rust type —
/// `bool | str | None`, `tuple[str, str] | str | None`, `int | str | None`,
/// `Any`, `Literal[False] | str | None`, ... Every member keeps its concrete
/// type; isinstance checks dispatch at runtime (`is_str()`, `as_int()`, ...)
/// and narrow the value in the branch, mirroring StrOrBytes.
#[derive(Clone, Debug, PartialEq)]
pub enum PyValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Arc<Vec<PyValue>>),
    Dict(Arc<PyDict<String, PyValue>>),
    None_,
}

/// The codegen hoists uninitialized locals and derives `Default` on
/// generated structs with boxed fields; None is the honest empty value
/// (a fresh Python binding holds nothing).
impl Default for PyValue {
    fn default() -> Self {
        PyValue::None_
    }
}

/// Python iteration over a boxed value (`for field in iterable:` where
/// the argument's type is unknown — urllib3's filepost): tuples yield
/// their elements, strings their characters (as 1-char strings, like
/// Python), bytes their integer octets. Iterating a non-iterable member
/// is CPython's TypeError — a loud panic (§12.2).
impl IntoIterator for PyValue {
    type Item = PyValue;
    type IntoIter = alloc::vec::IntoIter<PyValue>;
    fn into_iter(self) -> Self::IntoIter {
        let items: Vec<PyValue> = match &self {
            PyValue::Tuple(t) => t.iter().cloned().collect(),
            PyValue::Str(s) => s
                .chars()
                .map(|c| PyValue::Str(c.to_string()))
                .collect(),
            PyValue::Bytes(b) => b.iter().map(|&o| PyValue::Int(o as i64)).collect(),
            // Python iterates a dict's KEYS.
            PyValue::Dict(d) => d.keys().map(|k| PyValue::Str(k.clone())).collect(),
            PyValue::Int(_) => panic!("TypeError: 'int' object is not iterable"),
            PyValue::Float(_) => panic!("TypeError: 'float' object is not iterable"),
            PyValue::Bool(_) => panic!("TypeError: 'bool' object is not iterable"),
            PyValue::None_ => panic!("TypeError: 'NoneType' object is not iterable"),
        };
        items.into_iter()
    }
}

impl PyValue {
    pub fn is_int(&self) -> bool {
        matches!(self, PyValue::Int(_))
    }
    pub fn is_float(&self) -> bool {
        matches!(self, PyValue::Float(_))
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, PyValue::Bool(_))
    }
    pub fn is_str(&self) -> bool {
        matches!(self, PyValue::Str(_))
    }
    pub fn is_bytes(&self) -> bool {
        matches!(self, PyValue::Bytes(_))
    }
    pub fn is_tuple(&self) -> bool {
        matches!(self, PyValue::Tuple(_))
    }
    pub fn is_none(&self) -> bool {
        matches!(self, PyValue::None_)
    }
    /// Python len(): characters for a str, octets for bytes, elements for a
    /// tuple. Only valid on members that have a length (the code paths that
    /// call it are the ones Python would execute).
    pub fn len(&self) -> usize {
        match self {
            PyValue::Str(s) => s.chars().count(),
            PyValue::Bytes(b) => b.len(),
            PyValue::Tuple(t) => t.len(),
            other => panic!("len() of non-sized PyValue {other:?}"),
        }
    }
    /// The member views; only valid after isinstance narrowing (Python
    /// raises TypeError if the member does not match).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PyValue::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self {
            PyValue::Float(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PyValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            PyValue::Bytes(b) => Some(b.as_slice()),
            _ => None,
        }
    }
    pub fn as_tuple(&self) -> Option<&Vec<PyValue>> {
        match self {
            PyValue::Tuple(t) => Some(t),
            _ => None,
        }
    }
}

impl AsStrLike for PyValue {
    fn as_str_like(&self) -> &str {
        match self {
            PyValue::Str(s) => s.as_str(),
            other => panic!("expected a str in this context, got {other:?}"),
        }
    }
}

impl Truthy for PyValue {
    fn is_truthy(&self) -> bool {
        match self {
            PyValue::Int(v) => *v != 0,
            PyValue::Float(v) => *v != 0.0,
            PyValue::Bool(v) => *v,
            PyValue::Str(s) => !s.is_empty(),
            PyValue::Bytes(b) => !b.is_empty(),
            PyValue::Tuple(t) => !t.is_empty(),
            PyValue::Dict(d) => !d.is_empty(),
            PyValue::None_ => false,
        }
    }
}

impl Len for PyValue {
    fn len(&self) -> usize {
        PyValue::len(self)
    }
}

impl PyIsNone for PyValue {
    fn py_is_none(&self) -> bool {
        self.is_none()
    }
}

impl PyValue {
    /// Python's unary `-` on a boxed value: negates the numeric members
    /// (bool negates to int, as in CPython). Unmodeled operands panic
    /// with a TypeError naming the operand type — the same contract
    /// PySub's Option impl uses.
    pub fn py_neg(&self) -> PyValue {
        let type_name = |v: &PyValue| match v {
            PyValue::Str(_) => "str",
            PyValue::Bytes(_) => "bytes",
            PyValue::Tuple(_) => "tuple",
            PyValue::None_ => "NoneType",
            _ => "object",
        };
        match self {
            PyValue::Int(v) => PyValue::Int(-v),
            PyValue::Float(v) => PyValue::Float(-v),
            PyValue::Bool(v) => PyValue::Int(if *v { -1 } else { 0 }),
            other => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    format!(
                        "bad operand type for unary -: '{}'",
                        type_name(other)
                    )
                )
            ),
        }
    }
}

macro_rules! pyvalue_from {
    ($($t:ty => $v:ident),* $(,)?) => {
        $(impl From<$t> for PyValue {
            fn from(value: $t) -> Self {
                PyValue::$v(value)
            }
        })*
    };
}
pyvalue_from!(i64 => Int, f64 => Float, bool => Bool, String => Str, Vec<u8> => Bytes);

// Box a Python TUPLE value: `PyValue::from(("a".to_string(), 1))` — a
// module-level tuple VALUE (`basestring = (str, bytes)` — requests'
// compat; `NETRC_FILES = (".netrc", "_netrc")` — requests' utils) boxes
// as Tuple members, matching the boxed model's list-as-tuple divergence
// (round 33). Each element converts through its own Into<PyValue>, so
// nested tuples and mixed element types compose.
macro_rules! pyvalue_tuple_from {
    ($($n:ident),+ $(,)?) => {
        impl<$($n: Into<PyValue>),+> From<($($n,)+)> for PyValue {
            fn from(value: ($($n,)+)) -> Self {
                #[allow(non_snake_case)]
                let ($($n,)+) = value;
                PyValue::Tuple(Arc::new(alloc::vec![$(($n).into(),)+]))
            }
        }
    };
}
pyvalue_tuple_from!(A);
pyvalue_tuple_from!(A, B);
pyvalue_tuple_from!(A, B, C);
pyvalue_tuple_from!(A, B, C, D);
pyvalue_tuple_from!(A, B, C, D, E);
pyvalue_tuple_from!(A, B, C, D, E, F);

impl From<&str> for PyValue {
    fn from(value: &str) -> Self {
        PyValue::Str(value.to_string())
    }
}

/// The REVERSE conversions: a boxed PyValue flows back into a typed
/// slot or `impl Into<T>` parameter (`check_nfc((label).clone())` —
/// idna's core, where the None-mixing inference boxed a str label;
/// round 80). The value was boxed from a concrete member, so the
/// conversion recovers it; a WRONG member panics loudly (Python fails
/// at USE, rython fails at the conversion — the same loudness class,
/// and never a silent placeholder).
fn value_member_panic(expected: &str) -> ! {
    panic!(
        "{}",
        PyException::new(
            "TypeError",
            format!(
                "the boxed value is not a {} (Python would have failed at                  use; rython fails at the conversion)",
                expected
            ),
        )
    );
}

impl From<PyValue> for String {
    fn from(value: PyValue) -> String {
        value
            .as_str()
            .unwrap_or_else(|| value_member_panic("str"))
            .to_string()
    }
}

impl From<PyValue> for Vec<u8> {
    fn from(value: PyValue) -> Vec<u8> {
        value
            .as_bytes()
            .unwrap_or_else(|| value_member_panic("bytes"))
            .to_vec()
    }
}

impl From<PyValue> for i64 {
    fn from(value: PyValue) -> i64 {
        value.as_int().unwrap_or_else(|| value_member_panic("int"))
    }
}

impl From<PyValue> for f64 {
    fn from(value: PyValue) -> f64 {
        value.as_float().unwrap_or_else(|| value_member_panic("float"))
    }
}

impl From<PyValue> for bool {
    fn from(value: PyValue) -> bool {
        value.as_bool().unwrap_or_else(|| value_member_panic("bool"))
    }
}

impl From<&[u8]> for PyValue {
    fn from(value: &[u8]) -> Self {
        PyValue::Bytes(value.to_vec())
    }
}

// A bytes LITERAL (`b"raw"`) is a borrowed array `&[u8; N]`, which does
// not unsize during trait selection — box it directly (issue #161's
// impl-Into parameters take bytes literals at call sites).
impl<const N: usize> From<&[u8; N]> for PyValue {
    fn from(value: &[u8; N]) -> Self {
        PyValue::Bytes(value.to_vec())
    }
}

// A list of strings boxed as a value (`PyValue::from(vec![
// "ChecksumError".to_string()])` — the heterogeneous-return boxing of
// botocore's retryhandler): the boxed model has no distinct List member,
// so a boxed string list is a Tuple of Str members (the list-as-tuple
// divergence, round 33 — extend/matching treat both identically, and the
// member types are preserved).
impl From<Vec<String>> for PyValue {
    fn from(value: Vec<String>) -> Self {
        PyValue::Tuple(Arc::new(value.into_iter().map(PyValue::Str).collect()))
    }
}

// A list of ALREADY-BOXED members (`PyValue::from(exceptions)` where the
// local is Vec<PyValue>) boxes as a Tuple of the members.
impl From<Vec<PyValue>> for PyValue {
    fn from(value: Vec<PyValue>) -> Self {
        PyValue::Tuple(Arc::new(value))
    }
}

impl From<StrOrBytes> for PyValue {
    fn from(value: StrOrBytes) -> Self {
        match value {
            StrOrBytes::Str(s) => PyValue::Str(s),
            StrOrBytes::Bytes(b) => PyValue::Bytes(b),
        }
    }
}

/// A boxed DICT (issue #180): a dict literal whose value types mix
/// (`{'ProviderType': 'sso', 'Credentials': {...}}` — botocore) widens
/// to PyDict<String, PyValue> and a nested dict VALUE boxes here, so the
/// heterogeneous container is representable and indexable.
impl<K, V> From<PyDict<K, V>> for PyValue
where
    K: Into<String>,
    V: Into<PyValue>,
{
    fn from(d: PyDict<K, V>) -> Self {
        PyValue::Dict(Arc::new(
            d.into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        ))
    }
}

impl From<Option<PyValue>> for PyValue {
    // A mixed-literal element whose own lowering produced an Option
    // (a None member) flattens into the boxed value.
    fn from(value: Option<PyValue>) -> Self {
        match value {
            Some(v) => v,
            None => PyValue::None_,
        }
    }
}

impl From<()> for PyValue {
    fn from(_: ()) -> Self {
        PyValue::None_
    }
}

impl PyValue {
    /// The Python type name of the boxed member, for CPython-shaped
    /// operator error messages.
    pub fn py_type_name(&self) -> &'static str {
        match self {
            PyValue::Int(_) => "int",
            PyValue::Float(_) => "float",
            PyValue::Bool(_) => "bool",
            PyValue::Str(_) => "str",
            PyValue::Bytes(_) => "bytes",
            PyValue::Tuple(_) => "tuple",
            PyValue::Dict(_) => "dict",
            PyValue::None_ => "NoneType",
        }
    }
}

/// `+` on BOXED values (issues #115/#120: a mutable module global or a
/// varargs element holds a PyValue): dispatches on the runtime members
/// exactly as CPython's operator — numeric promotion (bool ⊂ int), str
/// and bytes concatenation, tuple concatenation — and a member mismatch
/// panics with CPython's TypeError message (§12.2 loud-by-panic: the
/// operator position has no Result channel).
impl PyAdd<PyValue> for PyValue {
    type Output = PyValue;
    fn py_add(&self, rhs: &PyValue) -> PyValue {
        use PyValue as V;
        match (self, rhs) {
            (V::Int(a), V::Int(b)) => V::Int(a + b),
            (V::Int(a), V::Float(b)) => V::Float(*a as f64 + b),
            (V::Float(a), V::Int(b)) => V::Float(a + *b as f64),
            (V::Float(a), V::Float(b)) => V::Float(a + b),
            (V::Bool(a), V::Bool(b)) => V::Int(*a as i64 + *b as i64),
            (V::Bool(a), V::Int(b)) => V::Int(*a as i64 + b),
            (V::Int(a), V::Bool(b)) => V::Int(a + *b as i64),
            (V::Bool(a), V::Float(b)) => V::Float((*a as i64) as f64 + b),
            (V::Float(a), V::Bool(b)) => V::Float(a + (*b as i64) as f64),
            (V::Str(a), V::Str(b)) => V::Str(format!("{}{}", a, b)),
            (V::Bytes(a), V::Bytes(b)) => {
                let mut out = a.clone();
                out.extend_from_slice(b);
                V::Bytes(out)
            }
            (V::Tuple(a), V::Tuple(b)) => {
                let mut out: Vec<PyValue> = a.as_ref().clone();
                out.extend(b.iter().cloned());
                V::Tuple(Arc::new(out))
            }
            (a, b) => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    format!(
                        "unsupported operand type(s) for +: '{}' and '{}'",
                        a.py_type_name(),
                        b.py_type_name()
                    )
                )
            ),
        }
    }
}

/// Convenience `+` where the right operand is a concrete value: box it
/// and delegate to the PyValue dispatch above (a mismatch panics the
/// same TypeError).
macro_rules! pyvalue_add_rhs {
    ($($t:ty),* $(,)?) => {
        $(impl PyAdd<$t> for PyValue {
            type Output = PyValue;
            fn py_add(&self, rhs: &$t) -> PyValue {
                PyAdd::<PyValue>::py_add(self, &PyValue::from(rhs.clone()))
            }
        })*
    };
}
pyvalue_add_rhs!(i64, f64, bool, String, &str);

/// Read a mutable module global (issue #115: a module-level name written by
/// functions through `global` lowers to a `static Mutex<T>`). The guard is
/// dropped inside this function, so two reads in one statement never hold
/// two locks at once (a bare `NAME.lock().unwrap().clone()` at the call
/// site would keep the guard temporary alive to the end of the statement
/// and deadlock on the second read).
#[cfg(feature = "std")]
pub fn py_global_read<T: Clone>(cell: &std::sync::Mutex<T>) -> T {
    cell.lock().unwrap().clone()
}

/// Write a mutable module global (issue #115). The value argument is fully
/// evaluated before the lock is taken (Rust argument order), so a
/// right-hand side that reads the same global cannot deadlock.
#[cfg(feature = "std")]
pub fn py_global_write<T>(cell: &std::sync::Mutex<T>, value: T) {
    *cell.lock().unwrap() = value;
}

/// Python str() of a boxed heterogeneous value (issue #121): ints, floats,
/// bools and None render as themselves; a str renders UNQUOTED; bytes use
/// the `b'...'` repr form; tuple elements always render in REPR form
/// (`str((1, 'a'))` == `"(1, 'a')"`).
/// The Python type name of a boxed value, for TypeError messages.
pub fn py_value_type_name(v: &PyValue) -> &'static str {
    match v {
        PyValue::Int(_) => "int",
        PyValue::Float(_) => "float",
        PyValue::Bool(_) => "bool",
        PyValue::Str(_) => "str",
        PyValue::Bytes(_) => "bytes",
        PyValue::Tuple(_) => "tuple",
        PyValue::Dict(_) => "dict",
        PyValue::None_ => "NoneType",
    }
}

pub fn py_value_str(v: &PyValue) -> String {
    match v {
        PyValue::Int(i) => i.to_string(),
        PyValue::Float(f) => py_float_repr(*f),
        PyValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        PyValue::Str(s) => s.clone(),
        PyValue::Bytes(b) => py_bytes_repr(b),
        PyValue::Tuple(items) => {
            let inner: Vec<String> = items.iter().map(py_value_repr).collect();
            if inner.len() == 1 {
                format!("({},)", inner[0])
            } else {
                format!("({})", inner.join(", "))
            }
        }
        PyValue::Dict(d) => {
            let inner: Vec<String> = d
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        py_value_repr(&PyValue::Str(k.clone())),
                        py_value_repr(v)
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        PyValue::None_ => "None".to_string(),
    }
}

/// Python repr() of a boxed heterogeneous value: like [`py_value_str`],
/// except a str renders QUOTED (`repr('s')` == `"'s'"`).
pub fn py_value_repr(v: &PyValue) -> String {
    match v {
        PyValue::Str(s) => py_str_repr(s),
        other => py_value_str(other),
    }
}

/// Structural equality is already derived; `Eq` plus a matching
/// structural [`Hash`] let boxed values serve as dict KEYS and set
/// members (`{"k": 1, 2: "v"}` boxes to `PyDict<PyValue, PyValue>`).
/// Any consistent hash is correct for Rust's maps — CPython's
/// type-differentiated hashes are not required for agreement between
/// Hash and PartialEq.
impl core::hash::Hash for PyValue {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            PyValue::Int(i) => {
                core::hash::Hash::hash(&0u8, state);
                core::hash::Hash::hash(i, state);
            }
            PyValue::Float(f) => {
                core::hash::Hash::hash(&1u8, state);
                // 0.0 and -0.0 compare equal (derived PartialEq), so they
                // must hash IDENTICALLY or equal keys miss in HashMaps
                // (CPython: {0.0: 'a'}[-0.0] == 'a'). Normalize -0.0.
                let bits = if *f == 0.0 { 0f64.to_bits() } else { f.to_bits() };
                core::hash::Hash::hash(&bits, state);
            }
            PyValue::Bool(b) => {
                core::hash::Hash::hash(&2u8, state);
                core::hash::Hash::hash(b, state);
            }
            PyValue::Str(s) => {
                core::hash::Hash::hash(&3u8, state);
                core::hash::Hash::hash(s, state);
            }
            PyValue::Bytes(b) => {
                core::hash::Hash::hash(&4u8, state);
                core::hash::Hash::hash(b, state);
            }
            PyValue::Tuple(items) => {
                core::hash::Hash::hash(&5u8, state);
                core::hash::Hash::hash(items.as_ref(), state);
            }
            PyValue::Dict(d) => {
                core::hash::Hash::hash(&7u8, state);
                for (k, v) in d.iter() {
                    core::hash::Hash::hash(k, state);
                    core::hash::Hash::hash(v, state);
                }
            }
            PyValue::None_ => core::hash::Hash::hash(&6u8, state),
        }
    }
}

impl Eq for PyValue {}

impl PyToString for PyValue {
    fn py_str(self) -> String {
        py_value_str(&self)
    }
}

impl PyDisplay for PyValue {
    fn py_display(&self) -> String {
        py_value_str(self)
    }
}

impl PyRepr for PyValue {
    fn py_repr(&self) -> String {
        py_value_repr(self)
    }
}

/// Python str() of the boxed value — f-strings and `"{}".format(...)`
/// print boxed operands directly. A str member prints unquoted (str() of
/// a str); every other member's str() IS its repr() (int, float, bool,
/// None, bytes, tuple, dict — verified against python3).
impl core::fmt::Display for PyValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PyValue::Str(s) => write!(f, "{}", s),
            other => write!(f, "{}", py_value_repr(other)),
        }
    }
}

/// Python `bytes.decode(encoding, errors)` as a trait, so receivers
/// without a statically-known type can decode: an unannotated parameter
/// (`T: PyDecode` — the isinstance-residual morphs of issue #161's
/// `_unicode_path`) and the boxed PyValue (the dynamic router's `Other`
/// arm hands the morph its boxed payload). `errors="replace"` follows
/// CPython for utf-8 (invalid sequences become U+FFFD); any other
/// non-strict errors value decodes strictly — the documented decode
/// divergence. A non-bytes boxed value raises CPython's AttributeError.
pub trait PyDecode {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException>;
}

impl PyDecode for [u8] {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException> {
        let enc = encoding.as_ref();
        if errors.as_ref() == "replace"
            && matches!(enc, "utf-8" | "utf8")
        {
            return Ok(String::from_utf8_lossy(self).into_owned());
        }
        stdlib::codec::decode_by_name(self, enc)
    }
}

impl PyDecode for Vec<u8> {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException> {
        self.as_slice().py_decode(encoding, errors)
    }
}

// A bytes LITERAL argument (`_unicode_path_any(b"raw")`) is a borrowed
// array — the bound instantiates at `&[u8; N]`, which the reference
// blanket reaches through the array impl.
impl<const N: usize> PyDecode for [u8; N] {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException> {
        self[..].py_decode(encoding, errors)
    }
}

impl<T: PyDecode + ?Sized> PyDecode for &T {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException> {
        (**self).py_decode(encoding, errors)
    }
}

impl PyDecode for PyValue {
    fn py_decode<E: AsRef<str>, R: AsRef<str>>(
        &self,
        encoding: E,
        errors: R,
    ) -> Result<String, PyException> {
        match self {
            PyValue::Bytes(b) => b.py_decode(encoding, errors),
            other => Err(PyException::new(
                "AttributeError",
                format!(
                    "'{}' object has no attribute 'decode'",
                    other.py_type_name()
                ),
            )),
        }
    }
}

/// `bytes(x)` lowering: the byte representation of a str (UTF-8), a
/// str|bytes union (the bytes branch), or bytes themselves (identity).
pub trait IntoBytesLike {
    fn into_bytes_like(self) -> Vec<u8>;
}
impl IntoBytesLike for String {
    fn into_bytes_like(self) -> Vec<u8> {
        self.into_bytes()
    }
}
impl IntoBytesLike for &str {
    fn into_bytes_like(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}
impl IntoBytesLike for Vec<u8> {
    fn into_bytes_like(self) -> Vec<u8> {
        self
    }
}
impl IntoBytesLike for &[u8] {
    fn into_bytes_like(self) -> Vec<u8> {
        self.to_vec()
    }
}
/// The boxed value's bytes(x): a str member's UTF-8, bytes themselves.
/// Any other member is CPython's TypeError — a loud panic (§12.2), never
/// a silent empty buffer.
impl IntoBytesLike for PyValue {
    fn into_bytes_like(self) -> Vec<u8> {
        match self {
            PyValue::Str(s) => s.into_bytes(),
            PyValue::Bytes(b) => b,
            other => panic!(
                "TypeError: cannot convert '{}' object to bytes",
                py_value_type_name(&other)
            ),
        }
    }
}

impl IntoBytesLike for StrOrBytes {
    fn into_bytes_like(self) -> Vec<u8> {
        match self {
            StrOrBytes::Str(s) => s.into_bytes(),
            StrOrBytes::Bytes(b) => b,
        }
    }
}

/// Python bytes methods that a narrowed `str | bytes` union exercises
/// (idna's A-label handling: lower/startswith/endswith/isascii). ASCII
/// byte-wise semantics, matching Python's bytes methods.
pub trait PyBytesOps {
    fn lower(&self) -> Vec<u8>;
    fn startswith(&self, prefix: &[u8]) -> bool;
    fn endswith(&self, suffix: &[u8]) -> bool;
    fn isascii(&self) -> bool;
}
impl PyBytesOps for Vec<u8> {
    fn lower(&self) -> Vec<u8> {
        self.iter().map(|b| b.to_ascii_lowercase()).collect()
    }
    fn startswith(&self, prefix: &[u8]) -> bool {
        self.starts_with(prefix)
    }
    fn endswith(&self, suffix: &[u8]) -> bool {
        self.ends_with(suffix)
    }
    fn isascii(&self) -> bool {
        self.iter().all(|b| b.is_ascii())
    }
}

/// Python-style list type with all list methods
#[derive(Debug, Clone, PartialEq)]
pub struct PyList<T> {
    inner: Vec<T>,
}

impl<T> PyList<T> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
    
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { inner: vec }
    }
    
    /// Python list.append() method
    pub fn append(&mut self, item: T) {
        self.inner.push(item);
    }
    
    /// Python list.extend() method
    pub fn extend(&mut self, items: Vec<T>) {
        self.inner.extend(items);
    }
    
    /// Python list.insert() method
    pub fn insert(&mut self, index: usize, item: T) {
        if index <= self.inner.len() {
            self.inner.insert(index, item);
        }
    }
    
    /// Python list.remove() method
    pub fn remove(&mut self, item: &T) -> bool 
    where 
        T: PartialEq,
    {
        if let Some(pos) = self.inner.iter().position(|x| x == item) {
            self.inner.remove(pos);
            true
        } else {
            false
        }
    }
    
    /// Python list.pop() method
    pub fn pop(&mut self, index: Option<usize>) -> Option<T> {
        match index {
            Some(i) if i < self.inner.len() => Some(self.inner.remove(i)),
            None if !self.inner.is_empty() => self.inner.pop(),
            _ => None,
        }
    }
    
    /// Python list.index() method
    pub fn index(&self, item: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        self.inner.iter().position(|x| x == item)
    }
    
    /// Python list.count() method
    pub fn count(&self, item: &T) -> usize
    where
        T: PartialEq,
    {
        self.inner.iter().filter(|&x| x == item).count()
    }
    
    /// Python list.sort() method
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.inner.sort();
    }
    
    /// Python list.reverse() method
    pub fn reverse(&mut self) {
        self.inner.reverse();
    }
    
    /// Python list.clear() method
    pub fn clear(&mut self) {
        self.inner.clear();
    }
    
    /// Python list.copy() method
    pub fn copy(&self) -> Self
    where
        T: Clone,
    {
        Self { inner: self.inner.clone() }
    }
    
    /// Get item by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }
    
    /// Set item by index
    pub fn set(&mut self, index: usize, item: T) -> bool {
        if index < self.inner.len() {
            self.inner[index] = item;
            true
        } else {
            false
        }
    }
    
    /// Access inner vector
    pub fn as_vec(&self) -> &Vec<T> {
        &self.inner
    }
}

impl<T> Len for PyList<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> Truthy for PyList<T> {
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

impl<T: Display> Display for PyList<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.inner.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        write!(f, "]")
    }
}

/// Python-style dictionary type with all dict methods
#[derive(Debug, Clone)]
pub struct PyDictionary<K, V>
where
    K: Eq + Hash,
{
    inner: HashMap<K, V>,
}

impl<K, V> PyDictionary<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }
    
    /// Python dict.get() method
    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }
    
    /// Python dict.get() method with default
    pub fn get_or_default(&self, key: &K, default: V) -> V
    where
        V: Clone,
    {
        self.inner.get(key).cloned().unwrap_or(default)
    }
    
    /// Set key-value pair
    pub fn set(&mut self, key: K, value: V) {
        self.inner.insert(key, value);
    }
    
    /// Python dict.keys() method
    pub fn keys(&self) -> Vec<&K> {
        self.inner.keys().collect()
    }
    
    /// Python dict.values() method
    pub fn values(&self) -> Vec<&V> {
        self.inner.values().collect()
    }
    
    /// Python dict.items() method
    pub fn items(&self) -> Vec<(&K, &V)> {
        self.inner.iter().collect()
    }
    
    /// Python dict.update() method
    pub fn update(&mut self, other: PyDictionary<K, V>) {
        self.inner.extend(other.inner);
    }
    
    /// Python dict.pop() method
    pub fn pop(&mut self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }
    
    /// Python dict.clear() method
    pub fn clear(&mut self) {
        self.inner.clear();
    }
    
    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }
}

impl<K, V> Len for PyDictionary<K, V> 
where 
    K: Eq + Hash,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> Truthy for PyDictionary<K, V> 
where 
    K: Eq + Hash,
{
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

/// Python-style tuple type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyTuple<T> {
    inner: Vec<T>,
}

impl<T> PyTuple<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { inner: items }
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }
    
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }
}

impl<T> Len for PyTuple<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> Truthy for PyTuple<T> {
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

impl<T: Display> Display for PyTuple<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "(")?;
        for (i, item) in self.inner.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        if self.inner.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

/// Python-style set type with all set methods
#[derive(Debug, Clone)]
pub struct PySet<T>
where
    T: Eq + Hash,
{
    inner: HashSet<T>,
}

impl<T> PySet<T>
where
    T: Eq + Hash,
{
    pub fn new() -> Self {
        Self { inner: HashSet::new() }
    }
    
    /// Python set.add() method
    pub fn add(&mut self, item: T) {
        self.inner.insert(item);
    }
    
    /// Python set.remove() method
    pub fn remove(&mut self, item: &T) -> bool {
        self.inner.remove(item)
    }
    
    /// Python set.discard() method
    pub fn discard(&mut self, item: &T) {
        self.inner.remove(item);
    }
    
    /// Python set.union() method
    pub fn union(&self, other: &PySet<T>) -> PySet<T>
    where
        T: Clone,
    {
        let mut result = self.clone();
        result.inner.extend(other.inner.iter().cloned());
        result
    }
    
    /// Python set.intersection() method
    pub fn intersection(&self, other: &PySet<T>) -> PySet<T>
    where
        T: Clone,
    {
        PySet {
            inner: self.inner.intersection(&other.inner).cloned().collect(),
        }
    }
    
    /// Python set.difference() method
    pub fn difference(&self, other: &PySet<T>) -> PySet<T>
    where
        T: Clone,
    {
        PySet {
            inner: self.inner.difference(&other.inner).cloned().collect(),
        }
    }
    
    /// Check if item is in set
    pub fn contains(&self, item: &T) -> bool {
        self.inner.contains(item)
    }
    
    /// Python set.clear() method
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<T> Len for PySet<T> 
where 
    T: Eq + Hash,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> Truthy for PySet<T>
where
    T: Eq + Hash,
{
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

/// Python frozenset: a set with no mutating surface. Immutability comes
/// from the API — only read operations exist — so the frozen contract
/// holds regardless of `mut` bindings.
#[derive(Debug, Clone)]
pub struct FrozenSet<T>
where
    T: Eq + Hash,
{
    inner: HashSet<T>,
}

impl<T> FrozenSet<T>
where
    T: Eq + Hash,
{
    pub fn contains(&self, item: &T) -> bool {
        self.inner.contains(item)
    }

    pub fn union(&self, other: &FrozenSet<T>) -> FrozenSet<T>
    where
        T: Clone,
    {
        let mut inner = self.inner.clone();
        inner.extend(other.inner.iter().cloned());
        FrozenSet { inner }
    }

    pub fn intersection(&self, other: &FrozenSet<T>) -> FrozenSet<T>
    where
        T: Clone,
    {
        FrozenSet {
            inner: self.inner.intersection(&other.inner).cloned().collect(),
        }
    }

    pub fn difference(&self, other: &FrozenSet<T>) -> FrozenSet<T>
    where
        T: Clone,
    {
        FrozenSet {
            inner: self.inner.difference(&other.inner).cloned().collect(),
        }
    }
}

impl<T> Len for FrozenSet<T>
where
    T: Eq + Hash,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> Truthy for FrozenSet<T>
where
    T: Eq + Hash,
{
    fn is_truthy(&self) -> bool {
        !self.inner.is_empty()
    }
}

impl<T> PyContains<T> for FrozenSet<T>
where
    T: Eq + Hash,
{
    fn py_contains(&self, item: &T) -> bool {
        self.inner.contains(item)
    }
}

/// Python's frozenset() builtin, from any list/sequence.
pub fn frozenset<T: Eq + Hash>(items: Vec<T>) -> FrozenSet<T> {
    FrozenSet {
        inner: items.into_iter().collect(),
    }
}

// ============================================================================
// TRAIT IMPLEMENTATIONS FOR BUILT-IN TYPES
// ============================================================================

impl Truthy for bool {
    fn is_truthy(&self) -> bool {
        *self
    }
}

impl Truthy for i64 {
    fn is_truthy(&self) -> bool {
        *self != 0
    }
}

impl Truthy for f64 {
    fn is_truthy(&self) -> bool {
        *self != 0.0
    }
}

impl Len for String {
    fn len(&self) -> usize {
        // Python counts code points, not bytes: len("café") == 4.
        self.chars().count()
    }
}

impl Len for str {
    fn len(&self) -> usize {
        self.chars().count()
    }
}

impl<T> Len for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl Len for [u8] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Len for [u8; N] {
    fn len(&self) -> usize {
        // as_slice() lands on the inherent slice len; a bare self.len()
        // would resolve to this trait method and recurse.
        self.as_slice().len()
    }
}

// ============================================================================
// TRUTHINESS OF STD TYPES (conditions lower through Truthy)
// ============================================================================

impl Truthy for String {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl Truthy for str {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl Truthy for &str {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<T> Truthy for Vec<T> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> Truthy for HashMap<K, V> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<T> Truthy for HashSet<T> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

/// Python: bool(None) is False; bool(Some-like values) follows the value.
impl<T: Truthy> Truthy for Option<T> {
    fn is_truthy(&self) -> bool {
        match self {
            Some(v) => v.is_truthy(),
            None => false,
        }
    }
}

// ============================================================================
// `is None` / `is not None`
// ============================================================================

/// Python's `x is None`. Option values report their None-ness; plain values
/// are never None (a non-Option Rust value always holds something).
pub trait PyIsNone {
    fn py_is_none(&self) -> bool;
}

impl<T> PyIsNone for Option<T> {
    fn py_is_none(&self) -> bool {
        self.is_none()
    }
}

macro_rules! never_none {
    ($($t:ty),* $(,)?) => {
        $(impl PyIsNone for $t {
            fn py_is_none(&self) -> bool {
                false
            }
        })*
    };
}

never_none!(bool, i8, i16, i32, i64, i128, u8, u16, u32, u64, usize, f32, f64, char, String, str, &str, PyException);

impl<T> PyIsNone for Vec<T> {
    fn py_is_none(&self) -> bool {
        false
    }
}

impl<K, V> PyIsNone for HashMap<K, V> {
    fn py_is_none(&self) -> bool {
        false
    }
}

impl<T> PyIsNone for HashSet<T> {
    fn py_is_none(&self) -> bool {
        false
    }
}

/// Python's whitespace set: Rust's Unicode White_Space plus the
/// file/group/record/unit separators U+001C–U+001F, which CPython's
/// str.isspace()/strip()/split() treat as whitespace but Rust does not
/// (verified the two sets differ by exactly those four code points).
pub fn py_is_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// The str.splitlines() boundary set: \n, \r, \r\n (one boundary), \v,
/// \f, \x1c–\x1e, \x85, \u2028, \u2029. \x1f (US) is NOT a boundary.
fn is_py_line_boundary(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'..='\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// Does Python's repr escape this character? CPython escapes everything
/// failing str.isprintable(): controls, format characters (Cf),
/// line/paragraph separators (Zl/Zp), and every space separator (Zs)
/// except U+0020. Rust's std exposes no Unicode category API, so the Cf
/// list is enumerated (Unicode 15 format characters).
fn repr_escapes(c: char) -> bool {
    c.is_control()
        || (c.is_whitespace() && c != ' ')
        || matches!(c, '\u{2028}' | '\u{2029}')
        || matches!(
            c,
            '\u{00ad}'
                | '\u{0600}'..='\u{0605}'
                | '\u{061c}'
                | '\u{06dd}'
                | '\u{070f}'
                | '\u{0890}'..='\u{0891}'
                | '\u{08e2}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{110bd}'
                | '\u{110cd}'
                | '\u{13430}'..='\u{1343f}'
                | '\u{1bca0}'..='\u{1bca3}'
                | '\u{1d173}'..='\u{1d17a}'
                | '\u{e0001}'
                | '\u{e0020}'..='\u{e007f}'
        )
}

/// Titlecase a character the way CPython does. Rust's std has no stable
/// char::to_titlecase, and titlecase differs from uppercase for a small
/// fixed set (Unicode SpecialCasing: ß, the Ǆ/Ǉ/Ǌ/Ǳ families, and the
/// ffi/ffl/ſt ligatures); everything else titlecases as uppercase.
fn py_to_titlecase(c: char) -> String {
    const TITLE_SPECIALS: &[(char, &str)] = &[
        ('\u{00df}', "Ss"), // ß
        ('\u{01c4}', "\u{01c5}"), // Ǆ -> ǅ
        ('\u{01c5}', "\u{01c5}"),
        ('\u{01c6}', "\u{01c5}"),
        ('\u{01c7}', "\u{01c8}"), // Ǉ -> ǈ
        ('\u{01c8}', "\u{01c8}"),
        ('\u{01c9}', "\u{01c8}"),
        ('\u{01ca}', "\u{01cb}"), // Ǌ -> ǋ
        ('\u{01cb}', "\u{01cb}"),
        ('\u{01cc}', "\u{01cb}"),
        ('\u{01f1}', "\u{01f2}"), // Ǳ -> ǲ
        ('\u{01f2}', "\u{01f2}"),
        ('\u{01f3}', "\u{01f2}"),
        ('\u{fb00}', "Ff"), // ﬀ
        ('\u{fb01}', "Fi"), // ﬁ
        ('\u{fb02}', "Fl"), // ﬂ
        ('\u{fb03}', "Ffi"), // ﬃ
        ('\u{fb04}', "Ffl"), // ﬄ
        ('\u{fb05}', "St"), // ﬅ
        ('\u{fb06}', "St"), // ﬆ
    ];
    match TITLE_SPECIALS.iter().find(|(ch, _)| *ch == c) {
        Some((_, s)) => (*s).to_string(),
        None => c.to_uppercase().collect(),
    }
}

// ============================================================================
// PYTHON LIST METHODS (on Vec)
// ============================================================================

/// Python list methods with no inherent Rust equivalent under the same
/// name. Methods whose Rust inherents already match Python (extend, clear,
/// reverse, sort) need nothing; methods whose inherents CONFLICT with
/// Python semantics (append, pop, remove, insert) are mapped in codegen
/// instead.
pub trait PyListOps<T> {
    /// list.count(x)
    fn count(&self, item: &T) -> i64
    where
        T: PartialEq;
    /// list.insert(i, x) with Python index rules: negative indices count
    /// from the end, and out-of-range indices clamp (insert past the end
    /// appends, before the start prepends) — never a panic. Result so a
    /// bounded deque can raise IndexError at its maximum size (issue #82).
    fn py_insert(&mut self, index: i64, item: T) -> Result<(), PyException>;
}

impl<T> PyListOps<T> for Vec<T> {
    fn count(&self, item: &T) -> i64
    where
        T: PartialEq,
    {
        self.iter().filter(|e| *e == item).count() as i64
    }
    fn py_insert(&mut self, index: i64, item: T) -> Result<(), PyException> {
        let len = self.len() as i64;
        let idx = if index < 0 {
            // len + i64::MIN overflows; Python prepends for any index with
            // |index| > len, so a checked fallback to a large negative
            // value lands at 0.
            len.checked_add(index).unwrap_or(i64::MIN).max(0)
        } else {
            index.min(len)
        } as usize;
        self.insert(idx, item);
        Ok(())
    }
}

// ============================================================================
// PYTHON STRING METHODS (on str / String via deref)
// ============================================================================

/// Python str methods. Named exactly as in Python where no inherent Rust
/// method conflicts; where one does (split, find), codegen maps the call to
/// the py_-prefixed name here.
/// Python's integer radix formatting (the x/X/o/b presentation types),
/// used by generated format code: Python renders negative values as
/// sign+magnitude (`format(-255, 'x') == "-ff"`) where Rust's radix
/// formatters print the two's-complement bit pattern. `align` is one of
/// '<', '>', '^', or '\0' for the default (right, with sign-aware zero
/// padding when `zero` is set).
pub fn py_int_radix_format(
    v: i64,
    fill: char,
    align: char,
    plus: bool,
    alternate: bool,
    zero: bool,
    width: usize,
    radix: char,
) -> String {
    let mag = v.unsigned_abs();
    let digits = match radix {
        'x' => format!("{:x}", mag),
        'X' => format!("{:X}", mag),
        'o' => format!("{:o}", mag),
        _ => format!("{:b}", mag),
    };
    let sign = if v < 0 {
        "-"
    } else if plus {
        "+"
    } else {
        ""
    };
    let prefix = if alternate {
        match radix {
            'x' => "0x",
            'X' => "0X",
            'o' => "0o",
            _ => "0b",
        }
    } else {
        ""
    };
    let body_len = sign.len() + prefix.len() + digits.len();
    if zero && align == '\0' {
        // Zero padding goes BETWEEN the sign/prefix and the digits:
        // format(-255, '#06x') == "-0x0ff".
        if body_len < width {
            let zeros = "0".repeat(width - body_len);
            return format!("{}{}{}{}", sign, prefix, zeros, digits);
        }
        return format!("{}{}{}", sign, prefix, digits);
    }
    let body = format!("{}{}{}", sign, prefix, digits);
    if body_len >= width {
        return body;
    }
    let pad = width - body_len;
    let filler = fill.to_string();
    match align {
        '<' => format!("{}{}", body, filler.repeat(pad)),
        '^' => {
            let left = pad / 2;
            format!("{}{}{}", filler.repeat(left), body, filler.repeat(pad - left))
        }
        // '>' and the default: numbers right-align.
        _ => format!("{}{}", filler.repeat(pad), body),
    }
}

/// The `,` thousands separator (Python's `f"{size:,}"`): the integer's
/// digits group in threes from the right, preserving the sign
/// (format(-1234567, ',') == "-1,234,567").
pub fn py_grouped_int(v: i64) -> String {
    let sign = if v < 0 { "-" } else { "" };
    let mag = v.unsigned_abs().to_string();
    let mut out = String::new();
    let chars: Vec<char> = mag.chars().collect();
    let first_group = chars.len() % 3;
    let mut i = 0;
    if first_group > 0 {
        out.extend(chars[..first_group].iter());
        i = first_group;
        if i < chars.len() {
            out.push(',');
        }
    }
    while i < chars.len() {
        out.extend(chars[i..i + 3].iter());
        i += 3;
        if i < chars.len() {
            out.push(',');
        }
    }
    format!("{}{}", sign, out)
}

/// A DYNAMIC-width format spec (`f"{completed:{total_width}d}"` — rich's
/// progress column): the width is a runtime value, so the interpolation
/// routes here. Python's spec semantics for the supported subset: an
/// integer right-aligned in the width with space fill (the dynamic-format
/// divergence — only the `{value:{width}d}` shape lowers).
pub fn py_dynamic_format(value: i64, width: i64) -> String {
    format!("{:>width$}", value, width = width.max(0) as usize)
}

/// Python requires ljust/rjust fill arguments to be exactly one
/// character: "hi".ljust(5, "ab") raises TypeError.
fn single_fill_char(fill: &str) -> Result<char, PyException> {
    let mut chars = fill.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyException::new(
            "TypeError",
            "The fill character must be exactly one character long",
        )),
    }
}

/// String operations on the BOXED heterogeneous value (`context["scheme"]
/// .lower()` where context is `dict[str, Any]` — urllib3's poolmanager:
/// the subscript read yields PyValue, and the runtime member is a str in
/// practice). CPython dispatches on the runtime type; a non-str member
/// raises AttributeError, which the loud §12.2 panic mirrors with
/// CPython's message. A separate trait: the blanket `PyStrOps for T:
/// AsRef<str>` cannot be narrowed to PyValue without a coherence
/// conflict, and PyValue does not satisfy AsRef<str>.
pub trait PyBoxedStrOps {
    /// str.lower() on the runtime member.
    fn py_boxed_lower(&self) -> String;
    /// str.upper() on the runtime member.
    fn py_boxed_upper(&self) -> String;
    /// str.strip() (whitespace) on the runtime member.
    fn py_boxed_strip(&self) -> String;
}

impl PyBoxedStrOps for PyValue {
    fn py_boxed_lower(&self) -> String {
        match self {
            PyValue::Str(s) => s.to_lowercase(),
            other => panic!(
                "AttributeError: '{}' object has no attribute 'lower'",
                other.py_type_name()
            ),
        }
    }
    fn py_boxed_upper(&self) -> String {
        match self {
            PyValue::Str(s) => s.to_uppercase(),
            other => panic!(
                "AttributeError: '{}' object has no attribute 'upper'",
                other.py_type_name()
            ),
        }
    }
    fn py_boxed_strip(&self) -> String {
        match self {
            PyValue::Str(s) => s.trim_matches(py_is_whitespace).to_string(),
            other => panic!(
                "AttributeError: '{}' object has no attribute 'strip'",
                other.py_type_name()
            ),
        }
    }
}

pub trait PyStrOps {
    fn upper(&self) -> String;
    fn lower(&self) -> String;
    fn strip(&self) -> String;
    fn lstrip(&self) -> String;
    fn rstrip(&self) -> String;
    fn capitalize(&self) -> String;
    fn startswith(&self, prefix: &str) -> bool;
    fn endswith(&self, suffix: &str) -> bool;
    /// str.find: CHARACTER index of the first match, or -1 (not an Option).
    fn py_find(&self, needle: &str) -> i64;
    /// str.count(sub): non-overlapping occurrences ("abc".count("") is 4).
    fn count<S: AsRef<str>>(&self, sub: S) -> i64;
    /// str.split(sep); an empty separator raises ValueError like Python
    /// (Rust's split would yield empty edge strings instead).
    fn py_split(&self, sep: &str) -> Result<Vec<String>, PyException>;
    /// str.split(sep, maxsplit): at most maxsplit splits from the left
    /// (maxsplit < 0 means unlimited).
    fn py_split_maxsplit(&self, sep: &str, maxsplit: i64) -> Result<Vec<String>, PyException>;
    /// str.split() with no argument: split on runs of whitespace.
    fn py_split_whitespace(&self) -> Vec<String>;
    /// str.split(None, maxsplit) / str.rsplit(None, maxsplit): whitespace
    /// mode with a split limit; the remainder keeps its whitespace.
    fn py_split_whitespace_maxsplit(&self, maxsplit: i64) -> Vec<String>;
    fn py_rsplit_whitespace_maxsplit(&self, maxsplit: i64) -> Vec<String>;
    /// str.rsplit(sep): like split for full splits, but named separately
    /// (str::rsplit is an inherent iterator method).
    fn py_rsplit(&self, sep: &str) -> Result<Vec<String>, PyException>;
    /// str.rsplit(sep, maxsplit): at most maxsplit splits from the RIGHT,
    /// pieces in left-to-right order.
    fn py_rsplit_maxsplit(&self, sep: &str, maxsplit: i64) -> Result<Vec<String>, PyException>;
    /// str.partition(sep): (head, sep, tail) around the FIRST match, or
    /// (self, "", "") when absent.
    fn partition(&self, sep: &str) -> Result<(String, String, String), PyException>;
    /// str.rpartition(sep): around the LAST match, or ("", "", self).
    fn rpartition(&self, sep: &str) -> Result<(String, String, String), PyException>;
    /// str.strip(chars): strip any of the given characters from both ends.
    fn py_strip_chars(&self, chars: &str) -> String;
    fn py_lstrip_chars(&self, chars: &str) -> String;
    fn py_rstrip_chars(&self, chars: &str) -> String;
    /// str.title(): first letter of each alphabetic run uppercased.
    fn title(&self) -> String;
    /// str.zfill(width): zero-pad to width CHARACTERS, after any sign.
    fn zfill(&self, width: i64) -> String;
    /// Python str.isupper(): at least one cased character and no
    /// lowercase characters (verified against python3).
    fn isupper(&self) -> bool;
    /// Python str.islower(): at least one cased character and no
    /// uppercase characters.
    fn islower(&self) -> bool;
    /// Python str.isalpha(): non-empty and every character alphabetic.
    fn isalpha(&self) -> bool;
    /// Python str.isdigit(): non-empty and every character a decimal
    /// digit. ASCII-exact (Python also classifies the Unicode digit
    /// property — superscripts like '²' — which Rust's std does not
    /// expose; documented divergence in §12).
    fn isdigit(&self) -> bool;
    /// Python str.isdecimal(): non-empty and every character a decimal
    /// digit (ASCII-exact, same §12 note as isdigit).
    fn isdecimal(&self) -> bool;
    /// Python str.isalnum(): non-empty and every character
    /// alphanumeric.
    fn isalnum(&self) -> bool;
    /// Python str.isspace(): non-empty and every character whitespace.
    fn isspace(&self) -> bool;
    /// Python str.isprintable(): every character printable (the empty
    /// string is printable). Approximate for format characters (Cf —
    /// Rust's std does not expose the category; documented in §12).
    fn isprintable(&self) -> bool;
    /// Python str.istitle(): cased characters form titlecase words
    /// (the first cased character after uncased is uppercase, the rest
    /// lowercase) and at least one is cased.
    fn istitle(&self) -> bool;
    /// str.ljust / str.rjust with a fill character, width in CHARACTERS.
    /// The fill must be exactly one character; Python raises TypeError
    /// otherwise (silently using a prefix would diverge).
    fn py_ljust(&self, width: i64, fill: &str) -> Result<String, PyException>;
    fn py_rjust(&self, width: i64, fill: &str) -> Result<String, PyException>;
    fn splitlines(&self) -> Vec<String>;
    /// sep.join(iterable)
    fn join<I, S>(&self, parts: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
}

impl<T: AsRef<str> + ?Sized> PyStrOps for T {
    fn upper(&self) -> String {
        self.as_ref().to_uppercase()
    }
    fn lower(&self) -> String {
        self.as_ref().to_lowercase()
    }
    fn strip(&self) -> String {
        self.as_ref().trim_matches(py_is_whitespace).to_string()
    }
    fn lstrip(&self) -> String {
        self.as_ref().trim_start_matches(py_is_whitespace).to_string()
    }
    fn rstrip(&self) -> String {
        self.as_ref().trim_end_matches(py_is_whitespace).to_string()
    }
    fn capitalize(&self) -> String {
        // Python titlecases the first char (uppercase where the two differ:
        // "ﬁle" -> "File", "ß" -> "Ss") and lowercases the rest.
        let mut chars = self.as_ref().chars();
        match chars.next() {
            Some(first) => py_to_titlecase(first) + &chars.as_str().to_lowercase(),
            None => String::new(),
        }
    }
    fn startswith(&self, prefix: &str) -> bool {
        self.as_ref().starts_with(prefix)
    }
    fn endswith(&self, suffix: &str) -> bool {
        self.as_ref().ends_with(suffix)
    }
    fn py_find(&self, needle: &str) -> i64 {
        match self.as_ref().find(needle) {
            Some(byte_idx) => self.as_ref()[..byte_idx].chars().count() as i64,
            None => -1,
        }
    }
    fn count<S: AsRef<str>>(&self, sub: S) -> i64 {
        self.as_ref().matches(sub.as_ref()).count() as i64
    }
    fn py_split(&self, sep: &str) -> Result<Vec<String>, PyException> {
        if sep.is_empty() {
            return Err(PyException::new("ValueError", "empty separator"));
        }
        Ok(self.as_ref().split(sep).map(str::to_string).collect())
    }
    fn py_split_maxsplit(&self, sep: &str, maxsplit: i64) -> Result<Vec<String>, PyException> {
        if sep.is_empty() {
            return Err(PyException::new("ValueError", "empty separator"));
        }
        if maxsplit < 0 {
            return self.as_ref().py_split(sep);
        }
        Ok(self.as_ref()
            .splitn(maxsplit as usize + 1, sep)
            .map(str::to_string)
            .collect())
    }
    fn py_split_whitespace(&self) -> Vec<String> {
        self.as_ref().split(py_is_whitespace)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
    fn py_split_whitespace_maxsplit(&self, maxsplit: i64) -> Vec<String> {
        if maxsplit < 0 {
            return self.as_ref().py_split_whitespace();
        }
        // Python: leading whitespace is consumed, at most maxsplit splits
        // are made, and the remainder keeps its internal/trailing
        // whitespace: " a b  c ".split(None, 1) == ["a", "b  c "].
        let mut out = Vec::new();
        let mut rest = self.as_ref().trim_start_matches(py_is_whitespace);
        let mut splits = 0;
        while !rest.is_empty() && splits < maxsplit {
            match rest.find(py_is_whitespace) {
                Some(i) => {
                    out.push(rest[..i].to_string());
                    rest = rest[i..].trim_start_matches(py_is_whitespace);
                    splits += 1;
                }
                None => break,
            }
        }
        if !rest.is_empty() {
            out.push(rest.to_string());
        }
        out
    }
    fn py_rsplit_whitespace_maxsplit(&self, maxsplit: i64) -> Vec<String> {
        if maxsplit < 0 {
            return self.as_ref().py_split_whitespace();
        }
        // Mirror image: trailing whitespace is consumed, splits count from
        // the right, and the remainder keeps its LEADING whitespace:
        // " a b  c ".rsplit(None, 2) == [" a", "b", "c"].
        let mut tail = Vec::new();
        let mut rest = self.as_ref().trim_end_matches(py_is_whitespace);
        let mut splits = 0;
        while !rest.is_empty() && splits < maxsplit {
            match rest.rfind(py_is_whitespace) {
                Some(i) => {
                    let sep_len = rest[i..].chars().next().map_or(1, char::len_utf8);
                    tail.push(rest[i + sep_len..].to_string());
                    rest = rest[..i].trim_end_matches(py_is_whitespace);
                    splits += 1;
                }
                None => break,
            }
        }
        let mut out = Vec::new();
        if !rest.is_empty() {
            out.push(rest.to_string());
        }
        out.extend(tail.into_iter().rev());
        out
    }
    fn py_rsplit(&self, sep: &str) -> Result<Vec<String>, PyException> {
        self.as_ref().py_split(sep)
    }
    fn py_rsplit_maxsplit(&self, sep: &str, maxsplit: i64) -> Result<Vec<String>, PyException> {
        if sep.is_empty() {
            return Err(PyException::new("ValueError", "empty separator"));
        }
        if maxsplit < 0 {
            return self.as_ref().py_split(sep);
        }
        let mut parts: Vec<String> = self.as_ref()
            .rsplitn(maxsplit as usize + 1, sep)
            .map(str::to_string)
            .collect();
        parts.reverse();
        Ok(parts)
    }
    fn partition(&self, sep: &str) -> Result<(String, String, String), PyException> {
        if sep.is_empty() {
            return Err(PyException::new("ValueError", "empty separator"));
        }
        match self.as_ref().find(sep) {
            Some(i) => Ok((
                self.as_ref()[..i].to_string(),
                sep.to_string(),
                self.as_ref()[i + sep.len()..].to_string(),
            )),
            None => Ok((self.as_ref().to_string(), String::new(), String::new())),
        }
    }
    fn rpartition(&self, sep: &str) -> Result<(String, String, String), PyException> {
        if sep.is_empty() {
            return Err(PyException::new("ValueError", "empty separator"));
        }
        match self.as_ref().rfind(sep) {
            Some(i) => Ok((
                self.as_ref()[..i].to_string(),
                sep.to_string(),
                self.as_ref()[i + sep.len()..].to_string(),
            )),
            None => Ok((String::new(), String::new(), self.as_ref().to_string())),
        }
    }
    fn py_strip_chars(&self, chars: &str) -> String {
        let set: Vec<char> = chars.chars().collect();
        self.as_ref().trim_matches(|c| set.contains(&c)).to_string()
    }
    fn py_lstrip_chars(&self, chars: &str) -> String {
        let set: Vec<char> = chars.chars().collect();
        self.as_ref().trim_start_matches(|c| set.contains(&c)).to_string()
    }
    fn py_rstrip_chars(&self, chars: &str) -> String {
        let set: Vec<char> = chars.chars().collect();
        self.as_ref().trim_end_matches(|c| set.contains(&c)).to_string()
    }
    fn title(&self) -> String {
        // Python: the first letter after any non-alphabetic character is
        // titlecased, the rest lowercased ("3rd" becomes "3Rd"; "ǳ" ->
        // "ǲ" where titlecase and uppercase differ).
        let mut out = String::with_capacity(self.as_ref().len());
        let mut prev_alpha = false;
        for c in self.as_ref().chars() {
            if c.is_alphabetic() {
                if prev_alpha {
                    out.extend(c.to_lowercase());
                } else {
                    out.push_str(&py_to_titlecase(c));
                }
                prev_alpha = true;
            } else {
                out.push(c);
                prev_alpha = false;
            }
        }
        out
    }
    fn zfill(&self, width: i64) -> String {
        let width = width.max(0) as usize;
        let count = self.as_ref().chars().count();
        if count >= width {
            return self.as_ref().to_string();
        }
        let zeros = "0".repeat(width - count);
        if let Some(rest) = self.as_ref().strip_prefix(['+', '-']) {
            format!("{}{}{}", &self.as_ref()[..1], zeros, rest)
        } else {
            format!("{}{}", zeros, self.as_ref())
        }
    }
    fn isupper(&self) -> bool {
        let mut has_cased = false;
        for c in self.as_ref().chars() {
            if c.is_lowercase() {
                return false;
            }
            has_cased |= c.is_uppercase();
        }
        has_cased
    }
    fn islower(&self) -> bool {
        let mut has_cased = false;
        for c in self.as_ref().chars() {
            if c.is_uppercase() {
                return false;
            }
            has_cased |= c.is_lowercase();
        }
        has_cased
    }
    fn isalpha(&self) -> bool {
        let s = self.as_ref();
        if s.is_empty() {
            return false;
        }
        #[cfg(feature = "re-regex")]
        {
            // Python's isalpha is the LETTER categories (Lu Ll Lt Lm Lo),
            // NOT the Alphabetic property: U+0345 (a combining mark with
            // the Alphabetic property) is False in CPython. The regex
            // engine's Unicode tables classify exactly the Letter set.
            static LETTER: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| regex::Regex::new(r"^\p{L}+$").unwrap());
            return LETTER.is_match(s) && LETTER.find(s).unwrap().end() == s.len();
        }
        #[cfg(not(feature = "re-regex"))]
        {
            s.chars().all(|c| c.is_alphabetic())
        }
    }
    fn isdigit(&self) -> bool {
        let s = self.as_ref();
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
    }
    fn isdecimal(&self) -> bool {
        let s = self.as_ref();
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
    }
    fn isalnum(&self) -> bool {
        let s = self.as_ref();
        if s.is_empty() {
            return false;
        }
        #[cfg(feature = "re-regex")]
        {
            static ALNUM: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| regex::Regex::new(r"^[\p{L}\p{N}]+$").unwrap());
            return ALNUM.is_match(s) && ALNUM.find(s).unwrap().end() == s.len();
        }
        #[cfg(not(feature = "re-regex"))]
        {
            s.chars().all(|c| c.is_alphabetic() || c.is_numeric())
        }
    }
    fn isspace(&self) -> bool {
        let s = self.as_ref();
        !s.is_empty()
            && s.chars().all(|c| {
                // Python's White_Space includes the four separator
                // controls U+001C..U+001F (file/group/record/unit
                // separator) — Rust's is_whitespace excludes Cc controls
                // (verified against python3: "\u001C".isspace() is True).
                c.is_whitespace() || matches!(c, '\u{1C}'..='\u{1F}')
            })
    }
    fn isprintable(&self) -> bool {
        #[cfg(feature = "re-regex")]
        {
            // Python's isprintable excludes every Other-category
            // character (Cc Cf Cs Co Cn) plus the line/paragraph
            // separators and NON-ASCII spaces (Zs): U+00A0, U+200B,
            // U+2028, U+00AD, U+FEFF, U+2060 are all False in CPython
            // (verified). The regex engine's tables classify them
            // exactly (the class subtracts the ASCII space, which IS
            // printable).
            static NONPRINTABLE: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| {
                    // \p{C} is ALL of Other (Cc Cf Cs Co Cn) — controls,
                    // format, surrogates, private use, unassigned —
                    // exactly Python's exclusion; Zl/Zp/Zs add the
                    // separators Python also excludes (minus the ASCII
                    // space, which IS printable).
                    regex::Regex::new(r"[\p{C}\p{Zl}\p{Zp}\p{Zs}--[ ]]").unwrap()
                });
            return !NONPRINTABLE.is_match(self.as_ref());
        }
        #[cfg(not(feature = "re-regex"))]
        {
            self.as_ref().chars().all(|c| {
                let cp = c as u32;
                !c.is_control()
                    // Non-ASCII whitespace (Zs beyond the ASCII space,
                    // Zl, Zp) is not printable in Python.
                    && !(c.is_whitespace() && !c.is_ascii())
                    // Private-use planes (Co) and surrogates (Cs).
                    && !(0xE000..=0xF8FF).contains(&cp)
                    && !(0xF0000..=0xFFFFD).contains(&cp)
                    && !(0x100000..=0x10FFFD).contains(&cp)
                    && !(0xD800..=0xDFFF).contains(&cp)
            })
        }
    }
    fn istitle(&self) -> bool {
        let mut prev_cased = false;
        let mut has_cased = false;
        for c in self.as_ref().chars() {
            let cased = c.is_uppercase() || c.is_lowercase();
            if cased {
                has_cased = true;
                if !prev_cased {
                    if !c.is_uppercase() {
                        return false;
                    }
                } else if c.is_uppercase() {
                    return false;
                }
            }
            prev_cased = cased;
        }
        has_cased
    }
    fn py_ljust(&self, width: i64, fill: &str) -> Result<String, PyException> {
        let fill_char = single_fill_char(fill)?;
        let width = width.max(0) as usize;
        let count = self.as_ref().chars().count();
        if count >= width {
            return Ok(self.as_ref().to_string());
        }
        Ok(format!("{}{}", self.as_ref(), fill_char.to_string().repeat(width - count)))
    }
    fn py_rjust(&self, width: i64, fill: &str) -> Result<String, PyException> {
        let fill_char = single_fill_char(fill)?;
        let width = width.max(0) as usize;
        let count = self.as_ref().chars().count();
        if count >= width {
            return Ok(self.as_ref().to_string());
        }
        Ok(format!("{}{}", fill_char.to_string().repeat(width - count), self.as_ref()))
    }
    fn splitlines(&self) -> Vec<String> {
        // Python's boundary set, not just \n/\r\n: classic-Mac \r,
        // \v \f \x1c-\x1e \x85 \u2028 \u2029 all split, with \r\n counted
        // as ONE boundary. A trailing boundary does not produce a trailing
        // empty line; consecutive boundaries produce empty lines between
        // them ("a\n\n".splitlines() == ["a", ""]).
        let bytes = self.as_ref().as_bytes();
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() {
            let c = self.as_ref()[i..].chars().next().expect("valid UTF-8");
            if is_py_line_boundary(c) {
                out.push(self.as_ref()[start..i].to_string());
                i += c.len_utf8();
                if c == '\r' && i < bytes.len() && bytes[i] == b'\n' {
                    i += 1; // \r\n is one boundary
                }
                start = i;
            } else {
                i += c.len_utf8();
            }
        }
        if start < bytes.len() {
            out.push(self.as_ref()[start..].to_string());
        }
        out
    }
    fn join<I, S>(&self, parts: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        parts
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>()
            .join(self.as_ref())
    }
}

// ============================================================================
// PYTHON DICTS: insertion-ordered, with the Python method surface
// ============================================================================

/// The hasher backing PyDict. Under `std` this is the same RandomState an
/// unadorned IndexMap would use; without `std`, indexmap has no default
/// hasher, so hashbrown's supplies one.
#[cfg(feature = "std")]
pub type PyHashBuilder = std::collections::hash_map::RandomState;
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub type PyHashBuilder = hashbrown::DefaultHashBuilder;

/// The type Python dict literals lower to. Python dicts preserve insertion
/// order (guaranteed since 3.7), which HashMap does not — IndexMap keeps
/// keys()/values()/items() and iteration faithful to Python.
pub type PyDict<K, V> = indexmap::IndexMap<K, V, PyHashBuilder>;

/// Python dict methods. Named as in Python where no inherent conflicts;
/// `get` conflicts with IndexMap's borrowed-Option accessor, so codegen
/// maps `d.get(k)` / `d.get(k, default)` to the py_-prefixed versions.
pub trait PyDictOps<K, V> {
    /// dict.get(k): the value or None (an Option, never an exception).
    fn py_get(&self, key: &K) -> Option<V>;
    /// dict.get(k, default)
    fn py_get_default(&self, key: &K, default: V) -> V;
    /// dict.keys(), in insertion order.
    fn py_keys(&self) -> Vec<K>;
    /// dict.values(), in insertion order.
    fn py_values(&self) -> Vec<V>;
    /// dict.items(), in insertion order.
    fn py_items(&self) -> Vec<(K, V)>;
    /// dict.setdefault(k, default): insert if missing, return the value.
    fn py_setdefault(&mut self, key: K, default: V) -> V;
    /// dict.update(other): insert/overwrite, appending new keys in order.
    fn update(&mut self, other: PyDict<K, V>);
}

impl<K: Eq + Hash + Clone, V: Clone> PyDictOps<K, V> for PyDict<K, V> {
    fn py_get(&self, key: &K) -> Option<V> {
        self.get(key).cloned()
    }
    fn py_get_default(&self, key: &K, default: V) -> V {
        self.get(key).cloned().unwrap_or(default)
    }
    fn py_keys(&self) -> Vec<K> {
        self.keys().cloned().collect()
    }
    fn py_values(&self) -> Vec<V> {
        self.values().cloned().collect()
    }
    fn py_items(&self) -> Vec<(K, V)> {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
    fn py_setdefault(&mut self, key: K, default: V) -> V {
        self.entry(key).or_insert(default).clone()
    }
    fn update(&mut self, other: PyDict<K, V>) {
        for (k, v) in other {
            self.insert(k, v);
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> PyDictOps<K, V> for HashMap<K, V> {
    fn py_get(&self, key: &K) -> Option<V> {
        self.get(key).cloned()
    }
    fn py_get_default(&self, key: &K, default: V) -> V {
        self.get(key).cloned().unwrap_or(default)
    }
    fn py_keys(&self) -> Vec<K> {
        self.keys().cloned().collect()
    }
    fn py_values(&self) -> Vec<V> {
        self.values().cloned().collect()
    }
    fn py_items(&self) -> Vec<(K, V)> {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
    fn py_setdefault(&mut self, key: K, default: V) -> V {
        self.entry(key).or_insert(default).clone()
    }
    fn update(&mut self, other: PyDict<K, V>) {
        for (k, v) in other {
            self.insert(k, v);
        }
    }
}

/// Python's `pop` — dispatched by receiver: list.pop(i) removes by index
/// (IndexError), dict.pop(k) removes by key (KeyError). Both catchable.
pub trait PyPop<I> {
    type Output;
    fn py_pop(&mut self, index: I) -> Result<Self::Output, PyException>;
}

impl<T> PyPop<i64> for Vec<T> {
    type Output = T;
    fn py_pop(&mut self, index: i64) -> Result<T, PyException> {
        let len = self.len();
        normalize_index(index, len)
            .map(|i| self.remove(i))
            .ok_or_else(|| PyException::new("IndexError", "pop index out of range"))
    }
}

impl<K: Eq + Hash + Debug, V> PyPop<K> for PyDict<K, V> {
    type Output = V;
    fn py_pop(&mut self, key: K) -> Result<V, PyException> {
        let msg = format!("{:?}", key);
        // shift_remove preserves the insertion order of remaining keys,
        // matching Python.
        self.shift_remove(&key)
            .ok_or_else(|| PyException::new("KeyError", msg))
    }
}

impl<K: Eq + Hash + Debug, V> PyPop<K> for HashMap<K, V> {
    type Output = V;
    fn py_pop(&mut self, key: K) -> Result<V, PyException> {
        let msg = format!("{:?}", key);
        self.remove(&key)
            .ok_or_else(|| PyException::new("KeyError", msg))
    }
}

/// Python's slice-bound normalization for RANGE operations (issue #153):
/// a negative bound counts from the end, both bounds clamp to [0, len],
/// and a stop before start collapses to start (the insertion point —
/// `xs[5:2] = R` inserts R at 5 without removing anything).
fn py_range_bounds(len: usize, start: Option<i64>, stop: Option<i64>) -> (usize, usize) {
    let n = len as i64;
    let norm = |bound: Option<i64>, default: i64| -> i64 {
        match bound {
            None => default,
            Some(i) if i < 0 => (i + n).clamp(0, n),
            Some(i) => i.clamp(0, n),
        }
    };
    let a = norm(start, 0);
    let b = norm(stop, n).max(a);
    (a as usize, b as usize)
}

/// `xs[a:b] = replacement` (issue #153): replace the range IN PLACE — a
/// different-length replacement inserts or removes elements, exactly
/// CPython. `None` bounds are the open ends (`xs[:b]`, `xs[a:]`, `xs[:]`).
pub fn py_splice<T>(
    xs: &mut Vec<T>,
    start: Option<i64>,
    stop: Option<i64>,
    replacement: Vec<T>,
) {
    let (a, b) = py_range_bounds(xs.len(), start, stop);
    xs.splice(a..b, replacement);
}

/// `del xs[a:b]` (issue #153): remove the range IN PLACE (Python's
/// `xs[a:b] = []`). Out-of-range bounds clamp; an empty or inverted
/// range removes nothing.
pub fn py_del_range<T>(xs: &mut Vec<T>, start: Option<i64>, stop: Option<i64>) {
    let (a, b) = py_range_bounds(xs.len(), start, stop);
    xs.drain(a..b);
}

/// dict.pop(k, default): remove and return, or the default when missing.
pub trait PyPopDefault<K, V> {
    fn py_pop_default(&mut self, key: K, default: V) -> V;
}

impl<K: Eq + Hash, V> PyPopDefault<K, V> for PyDict<K, V> {
    fn py_pop_default(&mut self, key: K, default: V) -> V {
        self.shift_remove(&key).unwrap_or(default)
    }
}

impl<K: Eq + Hash, V> PyPopDefault<K, V> for HashMap<K, V> {
    fn py_pop_default(&mut self, key: K, default: V) -> V {
        self.remove(&key).unwrap_or(default)
    }
}

// PyDict participates in every container protocol HashMap does.

/// Indexing a BOXED dict (`credentials['Credentials']` where the outer
/// value is a boxed PyValue — issue #180): a boxed dict member indexes
/// like the dict it holds; anything else raises CPython's TypeError.
impl PyIndex<i64> for PyValue {
    type Output = PyValue;
    fn py_index(&self, key: i64) -> Result<PyValue, PyException> {
        match self {
            PyValue::Bytes(b) => {
                let i = normalize_index(key, b.len())
                    .ok_or_else(|| PyException::new("IndexError", "index out of range"))?;
                Ok(PyValue::Int(b[i] as i64))
            }
            PyValue::Tuple(members) => {
                let i = normalize_index(key, members.len())
                    .ok_or_else(|| PyException::new("IndexError", "tuple index out of range"))?;
                Ok(members[i].clone())
            }
            PyValue::Str(s) => {
                let n = s.chars().count();
                let i = normalize_index(key, n)
                    .ok_or_else(|| PyException::new("IndexError", "string index out of range"))?;
                Ok(PyValue::Str(s.chars().nth(i).unwrap().to_string()))
            }
            // CPython's per-type not-subscriptable TypeError texts (the
            // round-57 projection never reaches these — the unpack RHS
            // boxes as Bytes/Tuple — but a user's `boxed[i]` must not
            // silently diverge).
            PyValue::Int(_) => Err(PyException::new(
                "TypeError",
                "'int' object is not subscriptable",
            )),
            PyValue::Float(_) => Err(PyException::new(
                "TypeError",
                "'float' object is not subscriptable",
            )),
            PyValue::Bool(_) => Err(PyException::new(
                "TypeError",
                "'bool' object is not subscriptable",
            )),
            PyValue::None_ => Err(PyException::new(
                "TypeError",
                "'NoneType' object is not subscriptable",
            )),
            _ => Err(PyException::new(
                "TypeError",
                "indices must be integers or slices",
            )),
        }
    }
}

impl PyIndex<&str> for PyValue {
    type Output = PyValue;
    fn py_index(&self, key: &str) -> Result<PyValue, PyException> {
        match self {
            PyValue::Dict(d) => d.py_index(key),
            other => Err(PyException::new(
                "TypeError",
                format!(
                    "'{}' object is not subscriptable",
                    py_value_type_name(other)
                ),
            )),
        }
    }
}

// A boxed-dict index with an OWNED String key (`script_ranges[script]`
// where script is a `str` local and the dict boxes to PyValue — idna's
// `_is_script`): the same Dict dispatch as the &str impl, for the owned
// String the str-parameter prologue produces (round 87).
impl PyIndex<String> for PyValue {
    type Output = PyValue;
    fn py_index(&self, key: String) -> Result<PyValue, PyException> {
        match self {
            PyValue::Dict(d) => d.py_index(key),
            other => Err(PyException::new(
                "TypeError",
                format!(
                    "'{}' object is not subscriptable",
                    py_value_type_name(other)
                ),
            )),
        }
    }
}

impl PyIndexMut<&str> for PyValue {
    type Output = PyValue;
    fn py_index_mut(&mut self, key: &str) -> Result<&mut PyValue, PyException> {
        match self {
            PyValue::Dict(d) => Arc::make_mut(d).py_index_mut(key),
            other => Err(PyException::new(
                "TypeError",
                format!(
                    "'{}' object is not subscriptable",
                    py_value_type_name(other)
                ),
            )),
        }
    }
}

impl<K: Eq + Hash + Debug, V: Clone> PyIndex<K> for PyDict<K, V> {
    type Output = V;
    fn py_index(&self, key: K) -> Result<V, PyException> {
        self.get(&key)
            .cloned()
            .ok_or_else(|| PyException::new("KeyError", format!("{:?}", key)))
    }
}

impl<K: Eq + Hash + Debug, V> PyIndexMut<K> for PyDict<K, V> {
    type Output = V;
    fn py_index_mut(&mut self, key: K) -> Result<&mut V, PyException> {
        let msg = format!("{:?}", key);
        self.get_mut(&key)
            .ok_or_else(|| PyException::new("KeyError", msg))
    }
}

// A String-keyed dict (e.g. groupdict()) subscripted with a literal:
// the literal lowers as &str, which the K-keyed impl above cannot take.
impl<V: Clone> PyIndex<&str> for PyDict<String, V> {
    type Output = V;
    fn py_index(&self, key: &str) -> Result<V, PyException> {
        self.get(key)
            .cloned()
            .ok_or_else(|| PyException::new("KeyError", format!("{:?}", key)))
    }
}

impl<V> PyIndexMut<&str> for PyDict<String, V> {
    type Output = V;
    fn py_index_mut(&mut self, key: &str) -> Result<&mut V, PyException> {
        let msg = format!("{:?}", key);
        self.get_mut(key)
            .ok_or_else(|| PyException::new("KeyError", msg))
    }
}

impl<K: Eq + Hash, V> PySetIndex<K, V> for PyDict<K, V> {
    fn py_set_index(&mut self, key: K, value: V) -> Result<(), PyException> {
        self.insert(key, value);
        Ok(())
    }
}

impl<K: Eq + Hash, V> PyContains<K> for PyDict<K, V> {
    fn py_contains(&self, item: &K) -> bool {
        self.contains_key(item)
    }
}

// Python's `in` probes by CONTENT — a str operand tests a container of
// OWNED strings regardless of Rust ownership: `"x" in vec_of_string`,
// `"k" in string_keyed_dict`, `"x" in string_set` (issue #229: a class
// field's Vec<String>/HashSet<String> reaches the trait directly, and a
// literal operand lowers as `&("x")`, an &&str). The blanket &String
// impls above keep serving String operands; these take the str side,
// in both the borrowed and double-borrowed spellings the renderers emit.
impl PyContains<str> for Vec<String> {
    fn py_contains(&self, item: &str) -> bool {
        self.iter().any(|s| s == item)
    }
}

// A literal set/list builds as Vec<&str> (`CONTENT_DECODERS =
// {"gzip", ...}` — urllib3's response constants); membership against an
// owned String operand (`encoding in CONTENT_DECODERS()`) needs the
// str/String spellings, comparing by value (round 61b).
impl PyContains<str> for Vec<&str> {
    fn py_contains(&self, item: &str) -> bool {
        self.iter().any(|s| *s == item)
    }
}

impl PyContains<String> for Vec<&str> {
    fn py_contains(&self, item: &String) -> bool {
        self.iter().any(|s| *s == item.as_str())
    }
}

// A BOXED operand (`direction in ("R", "AL")` where direction is a
// boxed Str — idna's _is_bidi): compare the &str members against the
// Str member by value.
impl PyContains<PyValue> for Vec<&str> {
    fn py_contains(&self, item: &PyValue) -> bool {
        match item {
            PyValue::Str(s) => self.iter().any(|m| *m == s),
            _ => false,
        }
    }
}

// The same boxed-operand membership for an OWNED string list (a
// string-literal list now lowers to Vec<String>, round 87 — idna's
// `direction in ["R", "AL"]` where direction is a boxed Str param):
// compare the String members against the Str member by value.
impl PyContains<PyValue> for Vec<String> {
    fn py_contains(&self, item: &PyValue) -> bool {
        match item {
            PyValue::Str(s) => self.iter().any(|m| m == s),
            _ => false,
        }
    }
}

// A BOXED list's membership test against a string (`encoding_iana in
// [specified_encoding, "ascii", "utf_8"]` where specified_encoding is
// `str | None`, so the list boxes to Vec<PyValue> — charset_normalizer's
// from_sequence): compare the Str members by value, like the typed
// Vec<String> impls above. The str/String/&str spellings the renderers
// emit all delegate to the same member compare.
impl PyContains<str> for Vec<PyValue> {
    fn py_contains(&self, item: &str) -> bool {
        self.iter()
            .any(|m| matches!(m, PyValue::Str(s) if s == item))
    }
}

impl PyContains<&str> for Vec<PyValue> {
    fn py_contains(&self, item: &&str) -> bool {
        self.iter()
            .any(|m| matches!(m, PyValue::Str(s) if s == *item))
    }
}

impl PyContains<String> for Vec<PyValue> {
    fn py_contains(&self, item: &String) -> bool {
        self.iter()
            .any(|m| matches!(m, PyValue::Str(s) if s == item))
    }
}

impl PyContains<&str> for Vec<String> {
    fn py_contains(&self, item: &&str) -> bool {
        self.iter().any(|s| s == *item)
    }
}

impl PyContains<str> for HashSet<String> {
    fn py_contains(&self, item: &str) -> bool {
        self.iter().any(|s| s == item)
    }
}

impl PyContains<&str> for HashSet<String> {
    fn py_contains(&self, item: &&str) -> bool {
        self.iter().any(|s| s == *item)
    }
}

// A literal set builds as HashSet<&str> (`{"utf_16", "utf_32"}` — the
// set literal's elements are &'static str); the generic
// `PyContains<T> for HashSet<T>` covers the &str operand spellings, and
// an owned String operand (`encoding_iana in {...}` — charset_normalizer)
// needs this String spelling, comparing by value (round 60).
impl PyContains<String> for HashSet<&str> {
    fn py_contains(&self, item: &String) -> bool {
        self.iter().any(|s| *s == item.as_str())
    }
}

impl<V> PyContains<str> for PyDict<String, V> {
    fn py_contains(&self, item: &str) -> bool {
        self.keys().any(|k| k == item)
    }
}

impl<V> PyContains<&str> for PyDict<String, V> {
    fn py_contains(&self, item: &&str) -> bool {
        self.keys().any(|k| k == *item)
    }
}

impl<V> PyContains<str> for HashMap<String, V> {
    fn py_contains(&self, item: &str) -> bool {
        self.keys().any(|k| k == item)
    }
}

impl<V> PyContains<&str> for HashMap<String, V> {
    fn py_contains(&self, item: &&str) -> bool {
        self.keys().any(|k| k == *item)
    }
}

impl<K, V> Truthy for PyDict<K, V> {
    fn is_truthy(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> PyIsNone for PyDict<K, V> {
    fn py_is_none(&self) -> bool {
        false
    }
}

impl<K: Eq + Hash, V> Len for PyDict<K, V> {
    fn len(&self) -> usize {
        PyDict::len(self)
    }
}

/// Convert an integer literal to a parameter's own type (M4): Rust std has
/// no `From<i64>` for `f64` and no int/float cross-PartialOrd, so a generic
/// parameter compared with an integer literal (`n <= 0`) gets
/// `T: PyFromInt` and the literal is converted through this trait —
/// identity for i64, float promotion for f64, exactly Python's semantics.
pub trait PyFromInt {
    fn py_from_int(value: i64) -> Self;
}
impl PyFromInt for i64 {
    fn py_from_int(value: i64) -> Self {
        value
    }
}
impl PyFromInt for f64 {
    fn py_from_int(value: i64) -> Self {
        value as f64
    }
}

// ============================================================================
// PYTHON `+`: numeric addition, string and list concatenation
// ============================================================================

/// Python's `+`, which Rust's Add can't fully model: String + String,
/// int/float promotion, and list concatenation. Operands are borrowed so
/// `a + b` doesn't consume the variables.
pub trait PyAdd<R: ?Sized> {
    type Output;
    fn py_add(&self, rhs: &R) -> Self::Output;
}

// ============================================================================
// COMPARISONS: a == b, a < b, ...
// ============================================================================
// Native Rust `==`/`<` can't model numpy: `arr > 2` must return an ARRAY,
// not a bool. The Py*Cmp traits give every Rust type the Python behaviour
// it already has via PartialEq/PartialOrd (blanket impls, bool result),
// while NdArray overrides them to broadcast elementwise and return an
// NdArray — the same pattern PyAdd/PySub use for + and -.

/// Python `==` — bool for scalars/containers, elementwise NdArray for arrays.
pub trait PyEq<R: ?Sized> {
    type Output;
    fn py_eq(&self, rhs: &R) -> Self::Output;
}
/// Python `!=`.
pub trait PyNe<R: ?Sized> {
    type Output;
    fn py_ne(&self, rhs: &R) -> Self::Output;
}
/// Python `<`.
pub trait PyLt<R: ?Sized> {
    type Output;
    fn py_lt(&self, rhs: &R) -> Self::Output;
}
/// Python `<=`.
pub trait PyLe<R: ?Sized> {
    type Output;
    fn py_le(&self, rhs: &R) -> Self::Output;
}
/// Python `>`.
pub trait PyGt<R: ?Sized> {
    type Output;
    fn py_gt(&self, rhs: &R) -> Self::Output;
}
/// Python `>=`.
pub trait PyGe<R: ?Sized> {
    type Output;
    fn py_ge(&self, rhs: &R) -> Self::Output;
}

impl<L: PartialEq<R>, R: ?Sized> PyEq<R> for L {
    type Output = bool;
    fn py_eq(&self, rhs: &R) -> bool {
        self == rhs
    }
}
impl<L: PartialEq<R>, R: ?Sized> PyNe<R> for L {
    type Output = bool;
    fn py_ne(&self, rhs: &R) -> bool {
        self != rhs
    }
}
impl<L: PartialOrd<R>, R: ?Sized> PyLt<R> for L {
    type Output = bool;
    fn py_lt(&self, rhs: &R) -> bool {
        self < rhs
    }
}
impl<L: PartialOrd<R>, R: ?Sized> PyLe<R> for L {
    type Output = bool;
    fn py_le(&self, rhs: &R) -> bool {
        self <= rhs
    }
}
impl<L: PartialOrd<R>, R: ?Sized> PyGt<R> for L {
    type Output = bool;
    fn py_gt(&self, rhs: &R) -> bool {
        self > rhs
    }
}
impl<L: PartialOrd<R>, R: ?Sized> PyGe<R> for L {
    type Output = bool;
    fn py_ge(&self, rhs: &R) -> bool {
        self >= rhs
    }
}


// ============================================================================
// `-` and `*` (PySub / PyMul)
// ============================================================================
// These mirror PyAdd: codegen routes `a - b` / `a * b` through them so
// operands are BORROWED (Python value semantics — variables stay usable),
// with numeric promotion, while NdArray overrides them to broadcast
// elementwise.

/// Python's `-`, with int/float promotion.
pub trait PySub<R: ?Sized> {
    type Output;
    fn py_sub(&self, rhs: &R) -> Self::Output;
}
/// Python's `*`, with int/float promotion (string repetition is handled
/// separately in codegen).
pub trait PyMul<R: ?Sized> {
    type Output;
    fn py_mul(&self, rhs: &R) -> Self::Output;
}

macro_rules! numeric_sub_mul {
    ($trait:ident, $method:ident, $op:tt; $($l:ty, $r:ty => $out:ty),* $(,)?) => {
        $(impl $trait<$r> for $l {
            type Output = $out;
            fn $method(&self, rhs: &$r) -> $out {
                (*self as $out) $op (*rhs as $out)
            }
        })*
    };
}

numeric_sub_mul!(
    PySub, py_sub, -;
    i64, i64 => i64,
    f64, f64 => f64,
    i64, f64 => f64,
    f64, i64 => f64,
);
numeric_sub_mul!(
    PyMul, py_mul, *;
    i64, i64 => i64,
    f64, f64 => f64,
    i64, f64 => f64,
    f64, i64 => f64,
);

impl<L, R: ?Sized> PySub<R> for Option<L>
where
    L: PySub<R>,
{
    type Output = L::Output;
    fn py_sub(&self, rhs: &R) -> L::Output {
        match self {
            Some(l) => l.py_sub(rhs),
            None => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "unsupported operand type(s) for -: 'NoneType'"
                )
            ),
        }
    }
}
impl<L, R: ?Sized> PyMul<R> for Option<L>
where
    L: PyMul<R>,
{
    type Output = L::Output;
    fn py_mul(&self, rhs: &R) -> L::Output {
        match self {
            Some(l) => l.py_mul(rhs),
            None => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "unsupported operand type(s) for *: 'NoneType'"
                )
            ),
        }
    }
}

macro_rules! numeric_add {
    ($($l:ty, $r:ty => $out:ty),* $(,)?) => {
        $(impl PyAdd<$r> for $l {
            type Output = $out;
            fn py_add(&self, rhs: &$r) -> $out {
                (*self as $out) + (*rhs as $out)
            }
        })*
    };
}

numeric_add!(
    i64, i64 => i64,
    f64, f64 => f64,
    i64, f64 => f64,
    f64, i64 => f64,
);

// bool ⊂ int (CPython): booleans participate in arithmetic as 0/1 —
// `True + 1 == 2`, `True * 3 == 3`, `True * 2.5 == 2.5`. The isinstance
// specializer's auto bool morphs (a bool argument taking an int-tested
// arm with its parameter kept bool) rely on these, as does any bool
// value flowing into arithmetic.
macro_rules! bool_arith {
    ($($trait:ident, $method:ident, $op:tt);* $(;)?) => {
        $(
            impl $trait<i64> for bool {
                type Output = i64;
                fn $method(&self, rhs: &i64) -> i64 {
                    (*self as i64) $op *rhs
                }
            }
            impl $trait<bool> for i64 {
                type Output = i64;
                fn $method(&self, rhs: &bool) -> i64 {
                    *self $op (*rhs as i64)
                }
            }
            impl $trait<bool> for bool {
                type Output = i64;
                fn $method(&self, rhs: &bool) -> i64 {
                    (*self as i64) $op (*rhs as i64)
                }
            }
            impl $trait<f64> for bool {
                type Output = f64;
                fn $method(&self, rhs: &f64) -> f64 {
                    ((*self as i64) as f64) $op *rhs
                }
            }
            impl $trait<bool> for f64 {
                type Output = f64;
                fn $method(&self, rhs: &bool) -> f64 {
                    *self $op ((*rhs as i64) as f64)
                }
            }
        )*
    };
}
bool_arith!(
    PyAdd, py_add, +;
    PySub, py_sub, -;
    PyMul, py_mul, *;
);

macro_rules! string_add {
    ($($l:ty, $r:ty),* $(,)?) => {
        $(impl PyAdd<$r> for $l {
            type Output = String;
            fn py_add(&self, rhs: &$r) -> String {
                format!("{}{}", self, rhs)
            }
        })*
    };
}

string_add!(
    String, String,
    String, &str,
    &str, String,
    &str, &str,
    str, String,
    str, &str,
    String, str,
    &str, str,
    str, str,
);

/// `+` on a maybe-None value: Python raises TypeError at runtime when the
/// value actually is None, and proceeds when it holds one. The panic
/// carries the TypeError display (it is not catchable by except yet).
impl<L, R: ?Sized> PyAdd<R> for Option<L>
where
    L: PyAdd<R>,
{
    type Output = L::Output;
    fn py_add(&self, rhs: &R) -> L::Output {
        match self {
            Some(l) => l.py_add(rhs),
            None => panic!(
                "{}",
                PyException::new(
                    "TypeError",
                    "unsupported operand type(s) for +: 'NoneType'"
                )
            ),
        }
    }
}

/// Python list concatenation: [1] + [2] == [1, 2].
impl<T: Clone> PyAdd<Vec<T>> for Vec<T> {
    type Output = Vec<T>;
    fn py_add(&self, rhs: &Vec<T>) -> Vec<T> {
        let mut out = self.clone();
        out.extend_from_slice(rhs);
        out
    }
}

/// Python list repetition: [0] * 3 == [0, 0, 0]. The count is an i64
/// (Python ints are i64 everywhere); a negative count yields an empty
/// list, matching Python's `[] * -1 == []`.
impl<T: Clone> PyMul<i64> for Vec<T> {
    type Output = Vec<T>;
    fn py_mul(&self, rhs: &i64) -> Vec<T> {
        let n = (*rhs).max(0) as usize;
        let mut out = Vec::with_capacity(self.len() * n);
        for _ in 0..n {
            out.extend_from_slice(self);
        }
        out
    }
}

// String repetition: `"ab" * 3` == "ababab", a non-positive count is ""
// (exactly like list * int). Both operand orders, and both string flavors
// on the left, so annotated params and inferred generics alike resolve.
impl PyMul<i64> for String {
    type Output = String;
    fn py_mul(&self, rhs: &i64) -> String {
        self.repeat((*rhs).max(0) as usize)
    }
}
impl PyMul<i64> for &str {
    type Output = String;
    fn py_mul(&self, rhs: &i64) -> String {
        self.repeat((*rhs).max(0) as usize)
    }
}
impl PyMul<String> for i64 {
    type Output = String;
    fn py_mul(&self, rhs: &String) -> String {
        rhs.repeat((*self).max(0) as usize)
    }
}
impl PyMul<&str> for i64 {
    type Output = String;
    fn py_mul(&self, rhs: &&str) -> String {
        rhs.repeat((*self).max(0) as usize)
    }
}

// ============================================================================
// SUBSCRIPTS: x[i] reads, x[i] = v stores, and x[a:b:c] slices
// ============================================================================

/// Normalize a Python index against a length: negative counts from the
/// end. Returns None when out of range (the caller raises).
fn normalize_index(index: i64, len: usize) -> Option<usize> {
    let len = len as i64;
    let idx = if index < 0 {
        // len + i64::MIN overflows; |index| >> len means out of range, so
        // IndexError rather than a wrapped index landing on a wrong
        // element (lst[-9223372036854775808]).
        match len.checked_add(index) {
            Some(v) => v,
            None => -1,
        }
    } else {
        index
    };
    if idx < 0 || idx >= len {
        None
    } else {
        Some(idx as usize)
    }
}

/// Python's subscript read `x[i]`: negative indices count from the end,
/// out-of-range raises IndexError, and a missing dict key raises KeyError —
/// both catchable by an enclosing try.
pub trait PyIndex<I> {
    type Output;
    fn py_index(&self, index: I) -> Result<Self::Output, PyException>;
}

impl<T: Clone> PyIndex<i64> for Vec<T> {
    type Output = T;
    fn py_index(&self, index: i64) -> Result<T, PyException> {
        normalize_index(index, self.len())
            .map(|i| self[i].clone())
            .ok_or_else(|| PyException::new("IndexError", "list index out of range"))
    }
}

/// Python string indexing is by character (code point), yielding a
/// one-character string.
impl PyIndex<i64> for str {
    type Output = String;
    fn py_index(&self, index: i64) -> Result<String, PyException> {
        let count = self.chars().count();
        normalize_index(index, count)
            .and_then(|i| self.chars().nth(i))
            .map(|c| c.to_string())
            .ok_or_else(|| PyException::new("IndexError", "string index out of range"))
    }
}

impl PyIndex<i64> for String {
    type Output = String;
    fn py_index(&self, index: i64) -> Result<String, PyException> {
        self.as_str().py_index(index)
    }
}

/// Homogeneous Rust tuples (e.g. str.partition results) subscript like
/// Python tuples: negative indices from the end, IndexError past the end.
impl<T: Clone> PyIndex<i64> for (T, T) {
    type Output = T;
    fn py_index(&self, index: i64) -> Result<T, PyException> {
        let i = if index < 0 { index + 2 } else { index };
        match i {
            0 => Ok(self.0.clone()),
            1 => Ok(self.1.clone()),
            _ => Err(PyException::new("IndexError", "tuple index out of range")),
        }
    }
}

impl<T: Clone> PyIndex<i64> for (T, T, T) {
    type Output = T;
    fn py_index(&self, index: i64) -> Result<T, PyException> {
        let i = if index < 0 { index + 3 } else { index };
        match i {
            0 => Ok(self.0.clone()),
            1 => Ok(self.1.clone()),
            2 => Ok(self.2.clone()),
            _ => Err(PyException::new("IndexError", "tuple index out of range")),
        }
    }
}

impl<K: Eq + Hash + Debug, V: Clone> PyIndex<K> for HashMap<K, V> {
    type Output = V;
    fn py_index(&self, key: K) -> Result<V, PyException> {
        self.get(&key)
            .cloned()
            .ok_or_else(|| PyException::new("KeyError", format!("{:?}", key)))
    }
}

/// Mutable subscript access for stores through nested containers
/// (`grid[i][j] = v`): yields a mutable reference into the container so
/// the write lands in place, never on a clone. Strings are excluded —
/// Python strings are immutable (`s[i] = c` is a TypeError).
pub trait PyIndexMut<I> {
    type Output;
    fn py_index_mut(&mut self, index: I) -> Result<&mut Self::Output, PyException>;
}

impl<T> PyIndexMut<i64> for Vec<T> {
    type Output = T;
    fn py_index_mut(&mut self, index: i64) -> Result<&mut T, PyException> {
        let len = self.len();
        normalize_index(index, len)
            .map(move |i| &mut self[i])
            .ok_or_else(|| PyException::new("IndexError", "list index out of range"))
    }
}

impl<K: Eq + Hash + Debug, V> PyIndexMut<K> for HashMap<K, V> {
    type Output = V;
    fn py_index_mut(&mut self, key: K) -> Result<&mut V, PyException> {
        let msg = format!("{:?}", key);
        self.get_mut(&key)
            .ok_or_else(|| PyException::new("KeyError", msg))
    }
}

/// Python's subscript store `x[i] = v`: Vec stores follow Python index
/// rules and raise IndexError; dict stores insert or overwrite.
pub trait PySetIndex<I, V> {
    fn py_set_index(&mut self, index: I, value: V) -> Result<(), PyException>;
}

impl<T> PySetIndex<i64, T> for Vec<T> {
    fn py_set_index(&mut self, index: i64, value: T) -> Result<(), PyException> {
        let len = self.len();
        match normalize_index(index, len) {
            Some(i) => {
                self[i] = value;
                Ok(())
            }
            None => Err(PyException::new(
                "IndexError",
                "list assignment index out of range",
            )),
        }
    }
}

impl<K: Eq + Hash, V> PySetIndex<K, V> for HashMap<K, V> {
    fn py_set_index(&mut self, key: K, value: V) -> Result<(), PyException> {
        self.insert(key, value);
        Ok(())
    }
}

/// Python slicing `x[a:b:c]`: clamps out-of-range bounds (never raises),
/// supports negative bounds and steps. Lists slice to lists, strings by
/// character to strings.
pub trait PySlice {
    type Output;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self::Output;
}

impl<T: Clone> PySlice for Vec<T> {
    type Output = Vec<T>;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Vec<T> {
        slice(self, start, stop, step)
    }
}

impl PySlice for str {
    type Output = String;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> String {
        let chars: Vec<char> = self.chars().collect();
        slice(&chars, start, stop, step).into_iter().collect()
    }
}

impl PySlice for String {
    type Output = String;
    fn py_slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> String {
        self.as_str().py_slice(start, stop, step)
    }
}

/// Python's in-place range replacement `xs[a:b] = replacement` and range
/// removal `del xs[a:b]` (step == 1 only — extended-slice replacement
/// with a step is handled by [`PySliceReplace::py_slice_assign_step`]).
/// Bounds behave like reads: negatives count from the end, out-of-range
/// clamps to the edges, so a longer replacement INSERTS elements and an
/// empty range is a no-op.
pub trait PySliceReplace {
    type Item;
    fn py_slice_assign(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        replacement: Vec<Self::Item>,
    );
    fn py_slice_assign_step(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
        replacement: Vec<Self::Item>,
    ) -> Result<(), PyException>;
    fn py_slice_delete(&mut self, start: Option<i64>, stop: Option<i64>);

    /// Extended-slice removal `del xs[a:b:c]`: removes the selected slots
    /// (c != 0).
    fn py_slice_delete_step(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
    ) -> Result<(), PyException>;
}

/// Normalizes one Python slice bound against `len`: negatives count from
/// the end, everything clamps into `[0, len]`.
fn slice_bound(len: i64, bound: i64) -> usize {
    let i = if bound < 0 { bound + len } else { bound };
    i.clamp(0, len) as usize
}

fn slice_range(len: i64, start: Option<i64>, stop: Option<i64>) -> (usize, usize) {
    let a = start.map(|b| slice_bound(len, b)).unwrap_or(0);
    let b = stop.map(|b| slice_bound(len, b)).unwrap_or(len as usize);
    let (a, b) = (a as usize, b.max(a) as usize);
    (a, b)
}

/// Computes the selected indices for an extended slice assignment
/// (CPython list_ass_subscript normalization, verified against python3
/// 3.14 across positive/negative steps, omitted bounds, and clamping).
fn extended_slice_indices(
    len: i64,
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> Result<Vec<usize>, PyException> {
    if step == 0 {
        return Err(value_error("slice step cannot be zero"));
    }
    let clamp = |i: i64, low: i64, high: i64| i.max(low).min(high);
    let (mut i, end) = if step > 0 {
        let s = start.map(|v| if v < 0 { v + len } else { v }).unwrap_or(0);
        let e = stop.map(|v| if v < 0 { v + len } else { v }).unwrap_or(len);
        (clamp(s, 0, len), clamp(e, 0, len))
    } else {
        let s = start
            .map(|v| if v < 0 { v + len } else { v })
            .unwrap_or(len - 1);
        let e = stop
            .map(|v| if v < 0 { v + len } else { v })
            .unwrap_or(-1);
        (clamp(s, -1, len - 1), clamp(e, -1, len))
    };
    let mut idxs = Vec::new();
    while (step > 0 && i < end) || (step < 0 && i > end) {
        idxs.push(i as usize);
        i += step;
    }
    Ok(idxs)
}

impl<T: Clone> PySliceReplace for Vec<T> {
    type Item = T;
    fn py_slice_assign(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        replacement: Vec<Self::Item>,
    ) {
        let (a, b) = slice_range(self.len() as i64, start, stop);
        self.splice(a..b, replacement);
    }
    fn py_slice_assign_step(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
        replacement: Vec<Self::Item>,
    ) -> Result<(), PyException> {
        let idxs = extended_slice_indices(self.len() as i64, start, stop, step)?;
        if replacement.len() != idxs.len() {
            return Err(value_error(format!(
                "attempt to assign sequence of size {} to extended slice of size {}",
                replacement.len(),
                idxs.len()
            )));
        }
        for (slot, item) in idxs.into_iter().zip(replacement.into_iter()) {
            self[slot] = item;
        }
        Ok(())
    }
    fn py_slice_delete(&mut self, start: Option<i64>, stop: Option<i64>) {
        let (a, b) = slice_range(self.len() as i64, start, stop);
        self.drain(a..b);
    }
    fn py_slice_delete_step(
        &mut self,
        start: Option<i64>,
        stop: Option<i64>,
        step: i64,
    ) -> Result<(), PyException> {
        // Remove the highest slot first so earlier removals don't shift
        // the slots still to be removed. A positive extended step DOES
        // reach this method (via `del xs[a:b:c]`), whose indices are
        // ascending, so sort before reversing.
        let mut idxs = extended_slice_indices(self.len() as i64, start, stop, step)?;
        idxs.sort_unstable();
        for slot in idxs.into_iter().rev() {
            self.remove(slot);
        }
        Ok(())
    }
}

// ============================================================================
// MEMBERSHIP (the `in` operator)
// ============================================================================

/// Python's `in` operator, dispatching on the container type: substring
/// search for strings, key lookup for dicts, element lookup for sequences
/// and sets. `x in c` lowers to `c.py_contains(&x)`.
pub trait PyContains<T: ?Sized> {
    fn py_contains(&self, item: &T) -> bool;
}

impl<T: PartialEq> PyContains<T> for Vec<T> {
    fn py_contains(&self, item: &T) -> bool {
        self.as_slice().contains(item)
    }
}

impl<T: PartialEq> PyContains<T> for [T] {
    fn py_contains(&self, item: &T) -> bool {
        self.contains(item)
    }
}

impl<T: PartialEq, const N: usize> PyContains<T> for [T; N] {
    fn py_contains(&self, item: &T) -> bool {
        self.as_slice().contains(item)
    }
}

impl<K: Eq + Hash, V> PyContains<K> for HashMap<K, V> {
    fn py_contains(&self, item: &K) -> bool {
        self.contains_key(item)
    }
}

impl<K: Eq + Hash, V> PyContains<K> for PyDictionary<K, V> {
    fn py_contains(&self, item: &K) -> bool {
        self.contains_key(item)
    }
}

impl<T: Eq + Hash> PyContains<T> for PySet<T> {
    fn py_contains(&self, item: &T) -> bool {
        self.contains(item)
    }
}

// Set literals lower to a std HashSet, so `x in {1, 2, 3}` needs this.
impl<T: Eq + Hash> PyContains<T> for HashSet<T> {
    fn py_contains(&self, item: &T) -> bool {
        self.contains(item)
    }
}

impl PyContains<str> for str {
    fn py_contains(&self, item: &str) -> bool {
        self.contains(item)
    }
}

impl PyContains<&str> for str {
    fn py_contains(&self, item: &&str) -> bool {
        self.contains(*item)
    }
}

// Python's `in` on a BOXED value (`k in self._container` where the field
// is PyValue — urllib3's RecentlyUsedContainer): dispatch on the member
// container exactly like CPython — substring for str, subsequence or
// integer-octet for bytes, element equality for tuples, key lookup for
// dicts — with CPython 3.11's TypeError text for non-container members
// (§12.2: loud panic, the IntoIterator precedent). Element equality is
// Python's `==`, not the derived PartialEq: `1 in (1, 2.0)` is True.
impl PyContains<PyValue> for PyValue {
    fn py_contains(&self, item: &PyValue) -> bool {
        match self {
            PyValue::Str(s) => match item {
                PyValue::Str(x) => s.contains(x.as_str()),
                other => panic!(
                    "TypeError: 'in <string>' requires string as left operand, not {}",
                    other.py_type_name()
                ),
            },
            PyValue::Bytes(b) => match item {
                PyValue::Bytes(x) => {
                    // An empty needle is always contained (CPython:
                    // b"" in b"abc" is True); windows(0) cannot express
                    // that, so it is handled before the search.
                    x.is_empty() || b.windows(x.len()).any(|w| w == x.as_slice())
                }
                PyValue::Int(o) => b.contains(&(*o as u8)),
                other => panic!(
                    "TypeError: a bytes-like object is required, not '{}'",
                    other.py_type_name()
                ),
            },
            PyValue::Tuple(t) => t.iter().any(|v| py_value_eq(v, item)),
            PyValue::Dict(d) => match item {
                PyValue::Str(k) => d.contains_key(k),
                // The boxed dict's keys are Strings; a non-str member is
                // never a key (an unhashable key would be CPython's
                // TypeError, but the boxed dict cannot hold one).
                _ => false,
            },
            other => panic!(
                "TypeError: argument of type '{}' is not iterable",
                other.py_type_name()
            ),
        }
    }
}

/// Probe the boxed containers with a str/String operand — the renderers
/// emit the literal/owned spelling at the call site (`k in boxed` where
/// k is a String local): box it and delegate, so the semantics are
/// identical whichever spelling reached the trait.
impl PyContains<String> for PyValue {
    fn py_contains(&self, item: &String) -> bool {
        self.py_contains(&PyValue::Str(item.clone()))
    }
}

impl PyContains<&str> for PyValue {
    fn py_contains(&self, item: &&str) -> bool {
        self.py_contains(&PyValue::Str((*item).to_string()))
    }
}

impl PyContains<str> for PyValue {
    fn py_contains(&self, item: &str) -> bool {
        self.py_contains(&PyValue::Str(item.to_string()))
    }
}

/// Python's `==` on boxed members: numeric kinds compare by value across
/// int/float/bool (CPython: `1 == 1.0`, `True == 1`); everything else is
/// structural.
fn py_value_eq(a: &PyValue, b: &PyValue) -> bool {
    match (a, b) {
        (PyValue::Int(x), PyValue::Float(y)) => (*x as f64) == *y,
        (PyValue::Float(x), PyValue::Int(y)) => *x == (*y as f64),
        (PyValue::Bool(x), PyValue::Int(y)) => (*x as i64) == *y,
        (PyValue::Int(x), PyValue::Bool(y)) => *x == (*y as i64),
        (PyValue::Bool(x), PyValue::Float(y)) => ((*x as i64) as f64) == *y,
        (PyValue::Float(x), PyValue::Bool(y)) => *x == ((*y as i64) as f64),
        _ => a == b,
    }
}

impl PyContains<String> for str {
    fn py_contains(&self, item: &String) -> bool {
        self.contains(item.as_str())
    }
}

impl PyContains<str> for String {
    fn py_contains(&self, item: &str) -> bool {
        self.as_str().contains(item)
    }
}

impl PyContains<&str> for String {
    fn py_contains(&self, item: &&str) -> bool {
        self.as_str().contains(*item)
    }
}

impl PyContains<String> for String {
    fn py_contains(&self, item: &String) -> bool {
        self.as_str().contains(item.as_str())
    }
}

// ============================================================================
// PYTHON EXCEPTIONS
// ============================================================================

/// Base class for all Python exceptions
#[derive(Debug, Clone)]
pub struct PyException {
    pub message: String,
    pub exception_type: String,
    /// The raised type's BUILTIN discriminant, computed once at
    /// construction (round 52): `except ValueError:` from generated code
    /// lowers to `matches_builtin(BuiltinException::ValueError)` — an
    /// integer comparison against this and the variant's ancestor slice —
    /// instead of a string walk plus MRO table search per clause. User
    /// classes (an open set) have `None` and keep the string `matches`.
    pub discriminant: Option<crate::builtin_exceptions::BuiltinException>,
}

impl PyException {
    pub fn new<T: AsRef<str>, M: AsRef<str>>(exception_type: T, message: M) -> Self {
        let exception_type = exception_type.as_ref();
        Self {
            message: message.as_ref().to_string(),
            exception_type: exception_type.to_string(),
            discriminant: crate::builtin_exceptions::BuiltinException::from_name(exception_type),
        }
    }

    /// Whether this exception is caught by a BUILTIN exception clause
    /// (`except ValueError:`) — the round-52 fast path: the raised
    /// type's discriminant compared against the target and the target's
    /// ancestor slice (both integers, no string walk). An exception
    /// raised OUTSIDE the builtin tree (a user class) has no
    /// discriminant and is caught only by `Exception`/`BaseException`,
    /// exactly like [`PyException::matches`].
    pub fn matches_builtin(&self, target: crate::builtin_exceptions::BuiltinException) -> bool {
        use crate::builtin_exceptions::BuiltinException;
        let Some(raised) = self.discriminant else {
            // A raised user class: the broad posture, same as matches().
            return target == BuiltinException::Exception
                || target == BuiltinException::BaseException;
        };
        if raised == target {
            return true;
        }
        raised.ancestors().contains(&target)
    }

    /// Whether this exception is caught by an `except <name>:` clause.
    ///
    /// Python semantics: the clause matches when `<name>` is the raised
    /// type itself or one of its ancestors. The tree is the interpreter's
    /// own data — python-ast dumps every builtin exception's real
    /// `__mro__` (plus the stdlib-module exceptions the runtime models)
    /// through PyO3, and the checked-in `BUILTIN_EXCEPTION_MRO` table
    /// carries it (regenerated and verified by python-ast's
    /// `exception_tree_is_current` test), so `except LookupError:`
    /// catches IndexError/KeyError, `except OSError:` catches
    /// FileNotFoundError and friends, aliases match both directions
    /// (`EnvironmentError` IS `OSError` — the same class object, so both
    /// spellings' MROs are identical), and `except Exception:` correctly
    /// does NOT catch SystemExit, KeyboardInterrupt or GeneratorExit —
    /// the old hand-copied tree missed parts of that (spec §12.3 defect
    /// class, fixed here).
    pub fn matches(&self, name: &str) -> bool {
        use crate::builtin_exceptions::BUILTIN_EXCEPTION_MRO;
        // Exact name equality covers user-defined classes and builtins
        // alike (a raised MyError is caught by `except MyError:`).
        if self.exception_type == name {
            return true;
        }
        // A builtin raised type: walk its interpreter-derived MRO. The
        // target canonicalizes first — `except EnvironmentError:` and
        // `except OSError:` name the same class object, so the raised
        // OSError's MRO must match the alias spelling too.
        if let Some((_, mro)) = BUILTIN_EXCEPTION_MRO
            .iter()
            .find(|(n, _)| *n == self.exception_type)
        {
            let target = crate::builtin_exceptions::canonical_name(name).unwrap_or(name);
            return mro.contains(&target);
        }
        // A raised type outside the built-in tree (a user class) keeps
        // the broad posture: only Exception and BaseException are
        // treated as catching it (rython does not know user-class
        // hierarchies — the class-as-value divergence).
        name == "Exception" || name == "BaseException"
    }

    /// Whether an `except <value>:` clause with a BOXED runtime value
    /// catches this exception (round 33 — botocore's
    /// `except self._retryable_exceptions as e:` where the field holds a
    /// tuple of class-name strings, or None). A Str member matches by
    /// name (the class-as-value model); a Tuple matches when any member
    /// matches, checked in order; anything else — including None — is
    /// CPython's TypeError ("catching classes that do not inherit from
    /// BaseException is not allowed"), raised as a typed exception at
    /// the point the clause is evaluated, exactly as CPython evaluates
    /// it lazily.
    pub fn matches_value(&self, value: &PyValue) -> Result<bool, PyException> {
        match value {
            PyValue::Str(name) => Ok(self.matches(name)),
            PyValue::Tuple(members) => {
                for member in members.iter() {
                    if self.matches_value(member)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Err(PyException::new(
                "TypeError",
                "catching classes that do not inherit from BaseException is not allowed",
            )),
        }
    }
}

/// Map a raised PyException onto the corresponding real Python exception
/// class, so PyO3 bindings surface `raise ValueError(...)` as an actual
/// ValueError to Python callers.
#[cfg(feature = "pyo3-interop")]
impl From<PyException> for pyo3::PyErr {
    fn from(e: PyException) -> pyo3::PyErr {
        let msg = e.message.clone();
        // One parse into the built-in exception enum (see
        // builtin_exceptions): every built-in type raises its real Python
        // class via the exhaustive pyo3_err match; anything unrecognized
        // (a user-defined class) keeps its full "Type: message" display.
        match crate::builtin_exceptions::BuiltinException::from_name(&e.exception_type) {
            Some(builtin) => builtin.pyo3_err(msg),
            None => pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)),
        }
    }
}

impl Display for PyException {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.exception_type, self.message)
    }
}

/// How a `try` body finished, for the try lowering's closure. A Python
/// `try` body runs inside a closure so `raise` can `return Err(...)`;
/// that closure also swallows `return`, `break`, and `continue`, which
/// must instead be carried out to the try statement's own position and
/// replayed AFTER the finally clause runs, exactly as Python orders
/// them.
pub enum PyFlow<T> {
    /// Fell off the end of the body.
    Normal,
    /// `return value`
    Return(T),
    /// `break` targeting a loop outside the try.
    Break,
    /// `continue` targeting a loop outside the try.
    Continue,
}

/// Python's `str(exception)` is the MESSAGE alone — `str(ValueError("boom"))`
/// is "boom", not "ValueError: boom" (that form is the traceback
/// rendering, which this type's Display produces). So
/// `except ValueError as e: print(e)` prints exactly what Python prints.
impl PyDisplay for PyException {
    fn py_display(&self) -> String {
        self.message.clone()
    }
}

/// Python's `repr(exception)` is `ValueError('boom')`.
impl PyRepr for PyException {
    fn py_repr(&self) -> String {
        format!("{}({})", self.exception_type, py_str_repr(&self.message))
    }
}

// Error trait only available with std
#[cfg(feature = "std")]
impl std::error::Error for PyException {}

/// Python ValueError
pub fn value_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("ValueError", message.as_ref())
}

/// Python TypeError  
pub fn type_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("TypeError", message.as_ref())
}

/// Python IndexError
pub fn index_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("IndexError", message.as_ref())
}

/// Python KeyError
pub fn key_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("KeyError", message.as_ref())
}

/// Python AttributeError
pub fn attribute_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("AttributeError", message.as_ref())
}

/// Python NameError
pub fn name_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("NameError", message.as_ref())
}

/// Python ZeroDivisionError
pub fn zero_division_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("ZeroDivisionError", message.as_ref())
}

/// Python OverflowError
pub fn overflow_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("OverflowError", message.as_ref())
}

/// Python RuntimeError
pub fn runtime_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("RuntimeError", message.as_ref())
}

/// Python AssertionError
pub fn assertion_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("AssertionError", message.as_ref())
}

/// Python ImportError
pub fn import_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("ImportError", message.as_ref())
}

/// Python ModuleNotFoundError (an ImportError)
pub fn module_not_found_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("ModuleNotFoundError", message.as_ref())
}

/// Python EOFError
pub fn eof_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("EOFError", message.as_ref())
}

/// Python FloatingPointError (an ArithmeticError)
pub fn floating_point_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("FloatingPointError", message.as_ref())
}

/// Python RecursionError (a RuntimeError)
pub fn recursion_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("RecursionError", message.as_ref())
}

/// Python MemoryError
pub fn memory_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("MemoryError", message.as_ref())
}

/// Python ReferenceError
pub fn reference_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("ReferenceError", message.as_ref())
}

/// Python BufferError
pub fn buffer_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("BufferError", message.as_ref())
}

/// Python StopIteration
pub fn stop_iteration<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("StopIteration", message.as_ref())
}

/// Python StopAsyncIteration
pub fn stop_async_iteration<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("StopAsyncIteration", message.as_ref())
}

/// Python SyntaxError
pub fn syntax_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("SyntaxError", message.as_ref())
}

/// Python IndentationError (a SyntaxError)
pub fn indentation_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("IndentationError", message.as_ref())
}

/// Python TabError (an IndentationError)
pub fn tab_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("TabError", message.as_ref())
}

/// Python SystemError
pub fn system_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("SystemError", message.as_ref())
}

/// Python UnboundLocalError (a NameError)
pub fn unbound_local_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("UnboundLocalError", message.as_ref())
}

/// Python UnicodeError (a ValueError)
pub fn unicode_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("UnicodeError", message.as_ref())
}

/// Python UnicodeEncodeError (a UnicodeError)
pub fn unicode_encode_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("UnicodeEncodeError", message.as_ref())
}

/// Python UnicodeDecodeError (a UnicodeError)
pub fn unicode_decode_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("UnicodeDecodeError", message.as_ref())
}

/// Python UnicodeTranslateError (a UnicodeError)
pub fn unicode_translate_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("UnicodeTranslateError", message.as_ref())
}

/// Python TimeoutError (an OSError)
pub fn timeout_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("TimeoutError", message.as_ref())
}

/// Python FileExistsError (an OSError)
pub fn file_exists_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("FileExistsError", message.as_ref())
}

/// Python FileNotFoundError (an OSError)
pub fn file_not_found_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("FileNotFoundError", message.as_ref())
}

/// Python PermissionError (an OSError)
pub fn permission_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("PermissionError", message.as_ref())
}

/// Python IsADirectoryError (an OSError)
pub fn is_a_directory_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("IsADirectoryError", message.as_ref())
}

/// Python NotADirectoryError (an OSError)
pub fn not_a_directory_error<M: AsRef<str>>(message: M) -> PyException {
    PyException::new("NotADirectoryError", message.as_ref())
}

// ============================================================================
// PYTHON STANDARD LIBRARY MODULES
// ============================================================================

mod builtin_exceptions;
/// The built-in exception discriminant, re-exported so generated code's
/// `use stdpython::*;` names it in `matches_builtin(BuiltinException::X)`
/// (round 52).
pub use builtin_exceptions::BuiltinException;
mod percent_format;

/// Python Standard Library modules
pub mod stdlib;


/// Custom Python signature system that preserves generic parameters
pub mod python_signature;

// The macros are automatically available at crate root due to #[macro_export]

// Re-export stdlib modules at the top level for convenience
#[cfg(feature = "std")]
pub use stdlib::sys;
#[cfg(feature = "std")]
pub use stdlib::os; 
#[cfg(feature = "std")]
pub use stdlib::subprocess;
#[cfg(feature = "std")]
pub use stdlib::sysconfig;
#[cfg(feature = "std")]
pub use stdlib::sysconfig::{
    is_python_build_py, is_python_build_wrapper,
};
#[cfg(feature = "std")]
pub use stdlib::venv;
#[cfg(feature = "std")]
pub use stdlib::math;
#[cfg(feature = "std")]
pub use stdlib::random;
#[cfg(feature = "std")]
pub use stdlib::datetime;
// The keyword-replace trait and its args struct must be in scope for
// dt.replace(hour=...) to resolve in generated code.
#[cfg(feature = "std")]
pub use stdlib::datetime::{PyReplace, ReplaceArgs};
/// Python asyncio module (tokio-backed; gated on the async-tokio feature,
/// which implies std).
#[cfg(feature = "async-tokio")]
pub use stdlib::asyncio;
#[cfg(feature = "std")]
pub use stdlib::time;
#[cfg(feature = "re-regex")]
pub use stdlib::re;
// io is in-memory buffers (StringIO/BytesIO) — pure alloc, every tier;
// the disk-backed PyFile constructors and open() stay std-only.
pub use stdlib::io;
#[cfg(feature = "std")]
pub use stdlib::argparse;
#[cfg(feature = "std")]
pub use stdlib::threading;
#[cfg(feature = "std")]
pub use stdlib::socket;
/// Python ssl (rustls-backed; gated on the ssl-rustls feature, which
/// implies std — on by default).
#[cfg(feature = "ssl-rustls")]
pub use stdlib::ssl;
/// Python urllib: the `parse` submodule (urlparse/urlsplit/urljoin/
/// urlencode/quote/unquote/...) is pure string handling, available under
/// plain std; the `request` submodule (urlopen) is ureq-backed and keeps
/// its own http-ureq gate. The parse tests run in the default workspace
/// suite (the retrospective's R6 correction on #260 — they were gated
/// behind http-ureq, which CI does not enable).
#[cfg(feature = "std")]
pub use stdlib::urllib;
// The Match-method trait must be in scope for m.group()/m.span() to
// resolve through the Option layer in generated code.
#[cfg(feature = "re-regex")]
pub use stdlib::re::PyMatchOps;
// The compiled-pattern matching trait must be in scope for
// `_TARGET_RE.match(x)`-style calls on a compiled-regex static in
// generated code.
#[cfg(feature = "re-regex")]
pub use stdlib::re::PyRegexOps;
pub use stdlib::string;
pub use stdlib::json;
pub use stdlib::collections;
pub use stdlib::itertools;
pub use stdlib::functools;
// The lru_cache backing store must be nameable in generated statics.
pub use stdlib::functools::PyLruCache;
pub use stdlib::heapq;
pub use stdlib::copy;
pub use stdlib::textwrap;
pub use stdlib::hashlib;
pub use stdlib::csv;
#[cfg(feature = "std")]
pub use stdlib::pathlib;
#[cfg(feature = "std")]
pub use stdlib::tempfile;
#[cfg(feature = "std")]
pub use stdlib::glob;
pub use stdlib::warnings;
#[cfg(feature = "std")]
pub use stdlib::numpy;

// Re-export custom macro-generated wrapper functions for generated code
#[cfg(feature = "std")]
pub use math::{
    // Basic math functions
    ceil_py, ceil_wrapper,
    floor_py, floor_wrapper,
    trunc_py, trunc_wrapper,
    fabs_py, fabs_wrapper,
    sqrt_py, sqrt_wrapper,
    pow_py, pow_wrapper,
    
    // Exponential and logarithmic functions
    exp_py, exp_wrapper,
    exp2_py, exp2_wrapper,
    expm1_py, expm1_wrapper,
    log_py, log_wrapper,
    log2_py, log2_wrapper,
    log10_py, log10_wrapper,
    log1p_py, log1p_wrapper,
    
    // Trigonometric functions
    sin_py, sin_wrapper,
    cos_py, cos_wrapper,
    tan_py, tan_wrapper,
    asin_py, asin_wrapper,
    acos_py, acos_wrapper,
    atan_py, atan_wrapper,
    atan2_py, atan2_wrapper,
    
    // Hyperbolic functions
    sinh_py, sinh_wrapper,
    cosh_py, cosh_wrapper,
    tanh_py, tanh_wrapper,
    asinh_py, asinh_wrapper,
    acosh_py, acosh_wrapper,
    atanh_py, atanh_wrapper,
    
    // Angular conversion
    degrees_py, degrees_wrapper,
    radians_py, radians_wrapper,
    
    // Special functions
    factorial_py, factorial_wrapper,
    gcd_py, gcd_wrapper,
    lcm_py, lcm_wrapper,
    
    // Classification functions
    isfinite_py, isfinite_wrapper,
    isinf_py, isinf_wrapper,
    isnan_py, isnan_wrapper,
    isclose_py, isclose_wrapper,
    
    // Utility functions
    copysign_py, copysign_wrapper,
    frexp_py, frexp_wrapper,
    ldexp_py, ldexp_wrapper,
    modf_py, modf_wrapper,
    fmod_py, fmod_wrapper,
    remainder_py, remainder_wrapper,
};

// Re-export random module functions
#[cfg(feature = "std")]
pub use random::{getstate, random, seed, triangular, uniform};

// Re-export JSON module wrapper functions
#[cfg(feature = "std")]
pub use json::{
    // JSON serialization/deserialization
    loads_py, loads_wrapper,
    dumps_py, dumps_wrapper,
    load_py, load_wrapper,
    dump_py, dump_wrapper,
};

#[cfg(feature = "std")]
pub use os::{
    // OS functions
    execv_mixed_py, execv_mixed_wrapper,
    getenv_py, getenv_wrapper,
    setenv_py, setenv_wrapper,
    getcwd_py, getcwd_wrapper,
    chdir_py, chdir_wrapper,
};

#[cfg(feature = "std")]
pub use os::path::{
    // OS path functions
    dirname_py, dirname_wrapper,
    basename_py, basename_wrapper,
    join_py, join_wrapper,
    join3_py, join3_wrapper,
    join_many_py, join_many_wrapper,
    exists_py, exists_wrapper,
    isfile_py, isfile_wrapper,
    isdir_py, isdir_wrapper,
    abspath_py, abspath_wrapper,
    relpath_py, relpath_wrapper,
};

#[cfg(feature = "std")]
pub use sys::{
    // Sys functions
    exit_py, exit_wrapper,
    platform_py, platform_wrapper,
    version_py, version_wrapper,
    get_executable_py, get_executable_wrapper,
    get_argv_py, get_argv_wrapper,
    get_platform_py, get_platform_wrapper,
};

// The `_py` wrappers only exist with std (they're the pyo3-facing shapes);
// the `_wrapper` forms are plain Rust and stay available on every tier.
#[cfg(feature = "std")]
pub use string::capwords_py;
pub use string::capwords_wrapper;

#[cfg(feature = "std")]
pub use collections::{counter_py, create_deque_py, defaultdict_int_py, defaultdict_list_py};
pub use collections::{
    counter_wrapper, create_deque_wrapper, defaultdict_int_wrapper, defaultdict_list_wrapper,
};

#[cfg(feature = "std")]
pub use subprocess::{
    // Subprocess functions
    run_py, run_wrapper,
    call_py, call_wrapper,
    check_call_py, check_call_wrapper,
    check_output_py, check_output_wrapper,
};


/// Python special variables
pub const __file__: &str = "script.py";
pub const __name__: &str = "__main__";

// ============================================================================
// OS-DEPENDENT FUNCTIONS (std feature only)
// ============================================================================

/// Python input() function - reads input from user
/// 
/// Note: Only available with `std` feature - requires OS I/O capabilities
#[cfg(feature = "std")]
pub fn input<P: AsRef<str>>(prompt: Option<P>) -> Result<String, PyException> {
    use std::io::{self, Write};
    
    if let Some(p) = prompt {
        print!("{}", p.as_ref());
        io::stdout().flush().map_err(|e| runtime_error(&format!("I/O error: {}", e)))?;
    }
    
    let mut input = String::new();
    let n = io::stdin().read_line(&mut input)
        .map_err(|e| runtime_error(&format!("I/O error: {}", e)))?;
    if n == 0 {
        // CPython raises EOFError at end of input; returning "" would make
        // `while True: line = input()` spin forever (and a blank input
        // line is also "", so `if not line: break` is not a workaround).
        return Err(PyException::new("EOFError", "EOF when reading a line"));
    }
    
    // Remove trailing newline
    if input.ends_with('\n') {
        input.pop();
        if input.ends_with('\r') {
            input.pop();
        }
    }
    
    Ok(input)
}

/// Map an I/O failure to the exception type Python raises, so
/// `except FileNotFoundError:` actually catches a missing file — a flat
/// RuntimeError would never match, and the error would escape the try.
#[cfg(feature = "std")]
fn os_error(e: &std::io::Error, path: &str) -> PyException {
    use std::io::ErrorKind;
    let kind = match e.kind() {
        ErrorKind::NotFound => "FileNotFoundError",
        ErrorKind::PermissionDenied => "PermissionError",
        ErrorKind::AlreadyExists => "FileExistsError",
        ErrorKind::IsADirectory => "IsADirectoryError",
        ErrorKind::NotADirectory => "NotADirectoryError",
        _ => "OSError",
    };
    PyException::new(kind, format!("{}: '{}'", e, path))
}

/// Python open() function - opens a file
/// 
/// Note: Only available with `std` feature - requires OS I/O capabilities
#[cfg(feature = "std")]
pub fn open<F: AsRef<str>, M: AsRef<str>>(filename: F, mode: Option<M>) -> Result<PyFile, PyException> {
    use std::fs::{File, OpenOptions};
    use std::io::{BufReader, BufWriter};
    
    let mode = mode.as_ref().map(|m| m.as_ref()).unwrap_or("r");
    
    let file = match mode {
        "r" => {
            let f = File::open(filename.as_ref())
                .map_err(|e| os_error(&e, filename.as_ref()))?;
            PyFile::new_read(BufReader::new(f))
        },
        "w" => {
            let f = File::create(filename.as_ref())
                .map_err(|e| os_error(&e, filename.as_ref()))?;
            PyFile::new_write(BufWriter::new(f))
        },
        "a" => {
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(filename.as_ref())
                .map_err(|e| os_error(&e, filename.as_ref()))?;
            PyFile::new_write(BufWriter::new(f))
        },
        _ => return Err(value_error(&format!("Invalid file mode: '{}'", mode))),
    };
    
    Ok(file)
}

/// Python file object: one type over every backend — disk handles from
/// open() and in-memory buffers from io.StringIO() — so file-consuming
/// code (csv.writer among others) works against either, exactly as
/// Python's file protocol does.
///
/// The in-memory Buffer backend is pure alloc, so the type lives on every
/// tier (no_std file I/O = io.StringIO/io.BytesIO); the DISK backends and
/// `open()` are std-gated.
pub struct PyFile {
    backend: PyFileBackend,
}

enum PyFileBackend {
    #[cfg(feature = "std")]
    DiskRead(std::io::BufReader<std::fs::File>),
    #[cfg(feature = "std")]
    DiskWrite(std::io::BufWriter<std::fs::File>),
    /// io.StringIO: contents plus a cursor in CHARACTERS (Python
    /// counts positions in code points). write() OVERWRITES at the
    /// cursor, as in Python — StringIO("seeded").write("!") yields
    /// "!eeded", not "seeded!".
    Buffer { data: String, pos: usize },
    Closed,
}

pub(crate) fn closed_file_error() -> PyException {
    // CPython: ValueError: I/O operation on closed file.
    value_error("I/O operation on closed file.")
}

impl PyFile {
    #[cfg(feature = "std")]
    fn new_read(reader: std::io::BufReader<std::fs::File>) -> Self {
        Self {
            backend: PyFileBackend::DiskRead(reader),
        }
    }

    #[cfg(feature = "std")]
    fn new_write(writer: std::io::BufWriter<std::fs::File>) -> Self {
        Self {
            backend: PyFileBackend::DiskWrite(writer),
        }
    }

    /// io.StringIO backing constructor.
    pub(crate) fn new_buffer(initial: &str) -> Self {
        Self {
            backend: PyFileBackend::Buffer {
                data: initial.to_string(),
                pos: 0,
            },
        }
    }

    /// Python file.read() method
    pub fn read(&mut self) -> Result<String, PyException> {
        match &mut self.backend {
            #[cfg(feature = "std")]
            PyFileBackend::DiskRead(reader) => {
                use std::io::Read;
                let mut contents = String::new();
                reader.read_to_string(&mut contents)
                    .map_err(|e| runtime_error(&format!("Read error: {}", e)))?;
                Ok(contents)
            }
            PyFileBackend::Buffer { data, pos } => {
                let out: String = data.chars().skip(*pos).collect();
                *pos = data.chars().count();
                Ok(out)
            }
            #[cfg(feature = "std")]
            PyFileBackend::DiskWrite(_) => Err(runtime_error("File not opened for reading")),
            PyFileBackend::Closed => Err(closed_file_error()),
        }
    }

    /// Python file.readline() method: the line INCLUDES its
    /// terminator, as in Python; empty means end of file.
    pub fn readline(&mut self) -> Result<String, PyException> {
        match &mut self.backend {
            #[cfg(feature = "std")]
            PyFileBackend::DiskRead(reader) => {
                use std::io::BufRead;
                let mut line = String::new();
                reader.read_line(&mut line)
                    .map_err(|e| runtime_error(&format!("Read error: {}", e)))?;
                Ok(line)
            }
            PyFileBackend::Buffer { data, pos } => {
                let mut line = String::new();
                for c in data.chars().skip(*pos) {
                    line.push(c);
                    if c == '\n' {
                        break;
                    }
                }
                *pos += line.chars().count();
                Ok(line)
            }
            #[cfg(feature = "std")]
            PyFileBackend::DiskWrite(_) => Err(runtime_error("File not opened for reading")),
            PyFileBackend::Closed => Err(closed_file_error()),
        }
    }

    /// Python file.readlines() method. Lines KEEP their terminators
    /// ("x\n", "y\n"), exactly as Python's readlines does — stripping
    /// them silently diverges (and breaks csv.reader's newline
    /// handling).
    pub fn readlines(&mut self) -> Result<Vec<String>, PyException> {
        let mut lines = Vec::new();
        loop {
            let line = self.readline()?;
            if line.is_empty() {
                return Ok(lines);
            }
            lines.push(line);
        }
    }

    /// Python file.write() method: returns the number of CHARACTERS
    /// written, as Python does. On a StringIO buffer this overwrites at
    /// the cursor (Python semantics), not appends.
    pub fn write<D: AsRef<str>>(&mut self, data: D) -> Result<i64, PyException> {
        let text = data.as_ref();
        match &mut self.backend {
            #[cfg(feature = "std")]
            PyFileBackend::DiskWrite(writer) => {
                use std::io::Write;
                writer.write_all(text.as_bytes())
                    .map_err(|e| runtime_error(&format!("Write error: {}", e)))?;
                Ok(text.chars().count() as i64)
            }
            PyFileBackend::Buffer { data, pos } => {
                let written = text.chars().count();
                let prefix: String = data.chars().take(*pos).collect();
                let suffix: String = data.chars().skip(*pos + written).collect();
                *data = format!("{}{}{}", prefix, text, suffix);
                *pos += written;
                Ok(written as i64)
            }
            #[cfg(feature = "std")]
            PyFileBackend::DiskRead(_) => Err(runtime_error("File not opened for writing")),
            PyFileBackend::Closed => Err(closed_file_error()),
        }
    }

    /// Python file.writelines() method
    pub fn writelines<S: AsRef<str>>(&mut self, lines: &[S]) -> Result<(), PyException> {
        for line in lines {
            self.write(line.as_ref())?;
        }
        Ok(())
    }

    /// io.StringIO.getvalue(): the whole buffer regardless of cursor.
    /// A disk file has no getvalue — Python raises AttributeError, and
    /// the typed lowering cannot know the backend at conversion time,
    /// so this fails loudly at runtime instead.
    pub fn getvalue(&self) -> Result<String, PyException> {
        match &self.backend {
            PyFileBackend::Buffer { data, .. } => Ok(data.clone()),
            PyFileBackend::Closed => Err(closed_file_error()),
            #[cfg(feature = "std")]
            _ => Err(PyException::new(
                "AttributeError",
                "'_io.TextIOWrapper' object has no attribute 'getvalue'",
            )),
        }
    }

    /// Python file.close() method
    pub fn close(&mut self) -> Result<(), PyException> {
        let old = core::mem::replace(&mut self.backend, PyFileBackend::Closed);
        #[cfg(feature = "std")]
        if let PyFileBackend::DiskWrite(mut writer) = old {
            use std::io::Write;
            writer.flush()
                .map_err(|e| runtime_error(&format!("Flush error: {}", e)))?;
        }
        #[cfg(not(feature = "std"))]
        let _ = old;
        Ok(())
    }
}

// ============================================================================
// COMPILER INTEGRATION HELPERS
// ============================================================================

/// Helper function for list creation from Rust vectors (common in compiled code)
pub fn py_list<T>(items: Vec<T>) -> PyList<T> {
    PyList::from_vec(items)
}

/// Helper function for dictionary creation (common in compiled code)
pub fn py_dict<K, V>() -> PyDictionary<K, V> 
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    PyDictionary::new()
}

/// Helper function for set creation (common in compiled code)
pub fn py_set<T>() -> PySet<T> 
where
    T: Clone + Eq + Hash,
{
    PySet::new()
}
/// Python's `set(iterable)` conversion for the shapes the codegen
/// produces: a string literal becomes the set of its characters
/// (urllib3's `_UNRESERVED_CHARS = set("...")`), and a boxed value the
/// empty/unknown set. The set-of-one-char-strings model matches CPython's
/// `set("abc") == {'a','b','c'}`.
pub fn set<S: AsRef<str>>(s: S) -> crate::HashSet<String> {
    s.as_ref().chars().map(|c| c.to_string()).collect()
}

/// A set of strings boxes as a Tuple of Str members (the list-as-tuple
/// divergence — the boxed model has no Set member; containment matches
/// members the same way).
impl From<crate::HashSet<String>> for PyValue {
    fn from(value: crate::HashSet<String>) -> Self {
        let mut members: Vec<PyValue> =
            value.into_iter().map(PyValue::Str).collect();
        // HashSet iteration order is nondeterministic — sort for a stable
        // boxed form.
        members.sort_by(|a, b| match (a, b) {
            (PyValue::Str(x), PyValue::Str(y)) => x.cmp(y),
            _ => core::cmp::Ordering::Equal,
        });
        PyValue::Tuple(Arc::new(members))
    }
}



/// Helper function for tuple creation (common in compiled code)
pub fn py_tuple<T>(items: Vec<T>) -> PyTuple<T> {
    PyTuple::new(items)
}

/// Helper for string formatting (common in f-strings compilation)
pub fn format_string<T: AsRef<str>>(template: T, args: &[&dyn Display]) -> String {
    let mut result = template.as_ref().to_string();
    for (i, arg) in args.iter().enumerate() {
        let placeholder = format!("{{{}}}", i);
        result = result.replace(&placeholder, &format!("{}", arg));
    }
    result
}

/// Helper for range() function with optional parameters - more flexible than the basic range
pub fn range_flexible(start: i64, stop: Option<i64>, step: Option<i64>) -> Result<PyRange, PyException> {
    let (start, stop, step) = match (stop, step) {
        (None, None) => (0, start, 1),
        (Some(stop), None) => (start, stop, 1),
        (Some(stop), Some(step)) => (start, stop, step),
        (None, Some(_)) => {
            return Err(type_error("range() missing required argument 'stop'"));
        }
    };
    range_start_stop_step(start, stop, step)
}

/// Helper for enumerate() function with slice input - returns pairs of (index, reference)
pub fn enumerate_slice<T>(iterable: &[T]) -> Vec<(usize, &T)> {
    iterable.iter().enumerate().collect()
}

/// Helper for zip() function with slice inputs - combines two iterables with lifetimes
pub fn zip_slices<'a, T, U>(iterable1: &'a [T], iterable2: &'a [U]) -> Vec<(&'a T, &'a U)> {
    iterable1.iter().zip(iterable2.iter()).collect()
}

/// Helper for Python-style slicing
pub fn slice<T>(items: &[T], start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Vec<T> 
where
    T: Clone,
{
    let len = items.len() as i64;
    let step = step.unwrap_or(1);

    if step == 0 {
        panic!("{}", PyException::new("ValueError", "slice step cannot be zero"));
    }

    // Resolve an index the way Python does: negative values count from the
    // end, then clamp to the valid range for the travel direction.
    let resolve = |idx: i64| -> i64 {
        let idx = if idx < 0 { idx + len } else { idx };
        if step > 0 {
            idx.clamp(0, len)
        } else {
            idx.clamp(-1, len - 1)
        }
    };

    let (start, stop) = if step > 0 {
        (start.map_or(0, resolve), stop.map_or(len, resolve))
    } else {
        (start.map_or(len - 1, resolve), stop.map_or(-1, resolve))
    };

    let mut result = Vec::new();
    let mut current = start;

    if step > 0 {
        while current < stop {
            result.push(items[current as usize].clone());
            current += step;
        }
    } else {
        while current > stop {
            result.push(items[current as usize].clone());
            current += step;
        }
    }

    result
}

/// Helper for Python-style string multiplication
pub fn multiply_string<S: AsRef<str>>(s: S, count: i64) -> String {
    if count <= 0 {
        String::new()
    } else {
        s.as_ref().repeat(count as usize)
    }
}

/// Helper for Python-style list multiplication
pub fn multiply_list<T>(items: &[T], count: i64) -> Vec<T> 
where
    T: Clone,
{
    if count <= 0 {
        Vec::new()
    } else {
        items.iter().cycle().take(items.len() * count as usize).cloned().collect()
    }
}

/// Helper for in/not in operations on strings
pub fn string_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) -> bool {
    haystack.as_ref().contains(needle.as_ref())
}

/// Helper for in/not in operations on lists
pub fn list_contains<T>(items: &[T], item: &T) -> bool 
where
    T: PartialEq,
{
    items.contains(item)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    /// Python iteration over boxed values, verified against python3:
    /// tuples yield elements, strings 1-char strings, bytes ints; a
    /// non-iterable member is the TypeError panic (§12.2).
    #[test]
    fn pyvalue_iteration_matches_python() {
        let t = PyValue::Tuple(Arc::new(vec![PyValue::Int(1), PyValue::Str("x".into())]));
        let items: Vec<PyValue> = t.into_iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_int(), Some(1));
        let s: Vec<PyValue> = PyValue::Str("ab".into()).into_iter().collect();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].as_str().map(|v| v.to_string()), Some("a".to_string()));
        let b: Vec<PyValue> = PyValue::Bytes(vec![65, 66]).into_iter().collect();
        assert_eq!(b[0].as_int(), Some(65));
        assert_eq!(b[1].as_int(), Some(66));
    }

    #[test]
    #[should_panic(expected = "'int' object is not iterable")]
    fn pyvalue_int_iteration_is_type_error() {
        let _ = PyValue::Int(3).into_iter();
    }

    #[test]
    fn bool_arithmetic_is_zero_one() {
        // CPython: bool ⊂ int, so booleans compute as 0/1 —
        // `True + 1 == 2`, `True * 3 == 3`, `2 - True == 1`,
        // `True * 2.5 == 2.5` — verified against python3.
        assert_eq!(true.py_add(&1i64), 2);
        assert_eq!(true.py_mul(&3i64), 3);
        assert_eq!(2i64.py_sub(&true), 1);
        assert_eq!(3i64.py_mul(&false), 0);
        assert_eq!(true.py_add(&true), 2);
        assert_eq!(true.py_mul(&2.5f64), 2.5);
        assert_eq!(2.5f64.py_add(&false), 2.5);
    }

    #[test]
    fn test_python_functions() {
        // Test generic abs function
        assert_eq!(abs(-5i64), 5);
        assert_eq!(abs(-3.14f64), 3.14);
        assert_eq!(abs(-42i32), 42);
        assert_eq!(abs(-2.5f32), 2.5);
        
        // Test generic sum function (i64/f64 only — an i32 impl would
        // leave integer-literal lists ambiguous, issue #133): owned,
        // borrowed, and slice forms all sum.
        let nums_i64 = vec![1i64, 2, 3, 4, 5];
        assert_eq!(sum(&nums_i64[..]), 15);
        assert_eq!(sum(&nums_i64), 15);
        assert_eq!(sum(nums_i64.clone()), 15);

        let nums_f64 = vec![1.5f64, 2.5, 3.0];
        assert_eq!(sum(&nums_f64[..]), 7.0);

        // Python sum() of a bool list counts the Trues (bool ⊂ int).
        assert_eq!(sum(vec![true, false, true]), 2);

        // Test with PyList
        let pylist = PyList::from_vec(vec![1i64, 2, 3]);
        assert_eq!(sum(&pylist), 6);
        
        // Test min/max
        assert_eq!(min(&nums_i64).unwrap(), 1);
        assert_eq!(max(&nums_i64).unwrap(), 5);
        
        // Test all/any
        let bools = vec![true, true, false];
        assert_eq!(any(&bools), true);
        assert_eq!(all(&bools), false);
    }
    
    #[test]
    fn test_generic_type_conversions() {
        // Test generic bool conversion
        assert_eq!(bool(42i64), true);
        assert_eq!(bool(0i64), false);
        assert_eq!(bool(3.14f64), true);
        assert_eq!(bool(0.0f64), false);
        assert_eq!(bool("hello"), true);
        assert_eq!(bool(""), false);
        
        // Test generic int conversion
        assert_eq!(int("123").unwrap(), 123);
        assert_eq!(int(45.7f64).unwrap(), 45);
        assert_eq!(int(true).unwrap(), 1);
        assert_eq!(int(false).unwrap(), 0);
        
        // Test generic float conversion
        assert_eq!(float("3.14").unwrap(), 3.14);
        assert_eq!(float(42i64).unwrap(), 42.0);
        
        // Test generic str conversion
        assert_eq!(str(123i64), "123");
        assert_eq!(str(3.14f64), "3.14");
        assert_eq!(str(true), "True");
        assert_eq!(str(false), "False");
        assert_eq!(str("hello"), "hello");
    }
    
    #[test]
    fn test_pystr() {
        let s = PyStr::new("hello world");
        assert_eq!(s.len(), 11);
        assert_eq!(s.upper().as_str(), "HELLO WORLD");
        assert_eq!(s.split(Some(" ")).len(), 2);
        assert_eq!(s.find("world"), 6);
        assert_eq!(s.count("l"), 3);
    }
    
    #[test]
    fn test_pylist() {
        let mut list = PyList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.pop(None), Some(3));
        assert_eq!(list.len(), 2);
    }
    
    #[test]
    fn test_pydict() {
        let mut dict = PyDictionary::new();
        dict.set("key1".to_string(), 42);
        dict.set("key2".to_string(), 100);
        
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.get(&"key1".to_string()), Some(&42));
        assert_eq!(dict.keys().len(), 2);
    }
    
    #[test]
    fn test_pyset() {
        let mut set = PySet::new();
        set.add(1);
        set.add(2);
        set.add(1); // duplicate
        
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(!set.contains(&3));
    }
    
    #[test]
    fn test_compiler_helpers() {
        // Test range function
        assert_eq!(
            range_flexible(3, None, None).unwrap().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            range_flexible(1, Some(4), None).unwrap().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            range_flexible(0, Some(10), Some(2)).unwrap().collect::<Vec<_>>(),
            vec![0, 2, 4, 6, 8]
        );
        
        // Test enumerate
        let items = vec!["a", "b", "c"];
        let enumerated = enumerate_slice(&items);
        assert_eq!(enumerated, vec![(0, &"a"), (1, &"b"), (2, &"c")]);
        
        // Test zip
        let nums = vec![1, 2, 3];
        let chars = vec!['a', 'b', 'c'];
        let zipped = zip_slices(&nums, &chars);
        assert_eq!(zipped, vec![(&1, &'a'), (&2, &'b'), (&3, &'c')]);
        
        // Test string multiplication
        assert_eq!(multiply_string("abc", 3), "abcabcabc");
        assert_eq!(multiply_string("x", 0), "");
        
        // Test list multiplication
        let list = vec![1, 2];
        assert_eq!(multiply_list(&list, 3), vec![1, 2, 1, 2, 1, 2]);
        
        // Test contains operations
        assert!(string_contains("hello world", "world"));
        assert!(!string_contains("hello", "xyz"));
        
        let list = vec![1, 2, 3, 4, 5];
        assert!(list_contains(&list, &3));
        assert!(!list_contains(&list, &10));
        
        // Test slicing
        let items = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(slice(&items, Some(1), Some(4), None), vec![1, 2, 3]);
        assert_eq!(slice(&items, None, Some(3), None), vec![0, 1, 2]);
        assert_eq!(slice(&items, Some(0), None, Some(2)), vec![0, 2, 4]);
    }
    
    #[test]
    fn test_helper_constructors() {
        // Test py_list
        let list = py_list(vec![1, 2, 3]);
        assert_eq!(list.len(), 3);
        
        // Test py_dict
        let mut dict: PyDictionary<String, i32> = py_dict();
        dict.set("key".to_string(), 42);
        assert_eq!(dict.len(), 1);
        
        // Test py_set
        let mut set: PySet<i32> = py_set();
        set.add(1);
        set.add(2);
        assert_eq!(set.len(), 2);
        
        // Test py_tuple
        let tuple = py_tuple(vec![1, 2, 3]);
        assert_eq!(tuple.len(), 3);
    }
}

#[cfg(test)]
mod pyvalue_round21_tests {
    use super::*;

    #[test]
    fn pyvalue_default_is_none() {
        // The codegen derives Default on structs with boxed fields; a
        // fresh Python binding holds nothing.
        assert_eq!(PyValue::default(), PyValue::None_);
    }

    #[test]
    fn pyvalue_display_matches_python_str() {
        // Verified against python3: str('x')='x' (unquoted), str(True)=
        // 'True', str(None)='None', str(b'hi')="b'hi'", str(1.0)='1.0'.
        assert_eq!(format!("{}", PyValue::Str("x".into())), "x");
        assert_eq!(format!("{}", PyValue::Bool(true)), "True");
        assert_eq!(format!("{}", PyValue::None_), "None");
        assert_eq!(format!("{}", PyValue::Bytes(b"hi".to_vec())), "b'hi'");
        assert_eq!(format!("{}", PyValue::Float(1.0)), "1.0");
    }

    #[test]
    fn pyvalue_into_bytes_like_covers_str_and_bytes() {
        assert_eq!(PyValue::Str("ab".into()).into_bytes_like(), b"ab".to_vec());
        assert_eq!(
            PyValue::Bytes([1u8, 2].to_vec()).into_bytes_like(),
            [1u8, 2].to_vec()
        );
    }

    #[test]
    #[should_panic(expected = "TypeError")]
    fn pyvalue_into_bytes_like_is_loud_on_non_bytes() {
        // CPython's TypeError — a loud panic (§12.2), never a silent
        // empty buffer.
        let _ = PyValue::Int(1).into_bytes_like();
    }
}

/// Python's `sep.join(parts)` for BYTES (`b"".join(data_parts)` —
/// urllib3's chunked response assembly): concatenate the byte slices
/// with the separator between them. A str element would be CPython's
/// TypeError — the typed signature (Vec<Vec<u8>>) makes a str element a
/// loud build error instead.
pub fn bytes_join(sep: &[u8], parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        parts.iter().map(Vec::len).sum::<usize>() + sep.len().saturating_mul(parts.len().saturating_sub(1)),
    );
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(sep);
        }
        out.extend_from_slice(part);
    }
    out
}

/// Python's `list.extend(x)` where x is a BOXED value (a boxed tuple of
/// names — `retryable_exceptions.extend(retry_exception)` where
/// retry_exception is the heterogeneous return of botocore's
/// `_extract_retryable_exception`, round 33): a Tuple member appends its
/// Str elements (a boxed string list boxes as a Tuple of Str); a
/// Str/Dict member is CPython's TypeError — a loud error; anything else
/// appends the member itself (a boxed value in a boxed-element vec).
pub fn py_extend_strings(out: &mut Vec<String>, value: &PyValue) -> Result<(), PyException> {
    match value {
        PyValue::Tuple(t) => {
            for member in t.iter() {
                match member {
                    PyValue::Str(s) => out.push(s.clone()),
                    other => {
                        return Err(PyException::new(
                            "TypeError",
                            &format!(
                                "TypeError: a bytes-like object is required, not '{}'",
                                other.py_type_name()
                            ),
                        ))
                    }
                }
            }
            Ok(())
        }
        PyValue::Str(s) => {
            out.push(s.clone());
            Ok(())
        }
        other => Err(PyException::new(
            "TypeError",
            &format!(
                "TypeError: '{}' object is not iterable",
                other.py_type_name()
            ),
        )),
    }
}

/// The boxed-element twin: a Tuple member appends its members as-is.
pub fn py_extend_values(out: &mut Vec<PyValue>, value: &PyValue) -> Result<(), PyException> {
    match value {
        PyValue::Tuple(t) => {
            out.extend(t.iter().cloned());
            Ok(())
        }
        PyValue::Str(s) => {
            out.push(PyValue::Str(s.clone()));
            Ok(())
        }
        other => Err(PyException::new(
            "TypeError",
            &format!(
                "TypeError: '{}' object is not iterable",
                other.py_type_name()
            ),
        )),
    }
}

#[cfg(test)]
mod percent_format_tests {
    use super::*;

    fn s(fmt: &str, rhs: impl crate::percent_format::PyFormatRhs) -> String {
        fmt.py_mod(&rhs).unwrap()
    }
    fn b(fmt: &[u8], rhs: impl crate::percent_format::PyFormatRhs) -> Vec<u8> {
        fmt.to_vec().py_mod(&rhs).unwrap()
    }

    /// The old-style %-operator, pinned against CPython 3.14 (the
    /// reference transcript is in the literals below — every one was
    /// produced by running the expression under python3).
    #[test]
    fn percent_formatting_matches_cpython() {
        assert_eq!(s("%s %s", ("a", "b")), "a b");
        assert_eq!(s("%d", 42), "42");
        assert_eq!(s("%d", -7), "-7");
        assert_eq!(s("%x", 255), "ff");
        assert_eq!(s("%X", 255), "FF");
        assert_eq!(s("%o", 8), "10");
        assert_eq!(s("%05d", 42), "00042");
        assert_eq!(s("%5d", 42), "   42");
        assert_eq!(s("%-5d", 42), "42   ");
        assert_eq!(s("%+d", 42), "+42");
        assert_eq!(s("% d", 42), " 42");
        assert_eq!(s("%.2f", 3.14159), "3.14");
        assert_eq!(s("%f", 2.5), "2.500000");
        assert_eq!(s("%e", 5000.0), "5.000000e+03");
        assert_eq!(s("%E", 5000.0), "5.000000E+03");
        assert_eq!(s("%g", 0.00001), "1e-05");
        assert_eq!(s("%g", 1234.5), "1234.5");
        assert_eq!(s("%r", "hi"), "'hi'");
        assert_eq!(s("%c", 65), "A");
        assert_eq!(s("%%", ()), "%");
        assert_eq!(s("100%%", ()), "100%");
        assert_eq!(s("%s", Option::<String>::None), "None");
        assert_eq!(s("%d", 3.9), "3");
        assert_eq!(s("%10s", "ab"), "        ab");
        assert_eq!(s("%.3s", "abcdef"), "abc");
        assert_eq!(s("%#x", 255), "0xff");
        assert_eq!(s("%#o", 8), "0o10");
        assert_eq!(s("%5.1f", 3.14159), "  3.1");
        assert_eq!(s("%-8s", "ab"), "ab      ");
        assert_eq!(s("%08.2f", 3.14159), "00003.14");
        assert_eq!(s("%g", 100.0), "100");
        assert_eq!(s("%g", 0.0001), "0.0001");
        assert_eq!(s("%g", 12345678.0), "1.23457e+07");
        assert_eq!(s("%.0f", 2.7), "3");
        assert_eq!(s("%+d", -5), "-5");
        assert_eq!(s("%d", 2i64.pow(40)), "1099511627776");
        assert_eq!(s("%x", -255), "-ff");
        assert_eq!(s("%s", 3.5), "3.5");
        assert_eq!(s("%s", true), "True");
        assert_eq!(s("%r", true), "True");
        assert_eq!(s("%r", 3.14), "3.14");
        assert_eq!(s("%r", Option::<String>::None), "None");
        assert_eq!(s("%s", b"bytes".to_vec()), "b'bytes'");
        assert_eq!(s("%r", b"bytes".to_vec()), "b'bytes'");
        assert_eq!(s("%10.3s", "abcdefgh"), "       abc");
        assert_eq!(s("%*s", (10i64, "ab")), "        ab");
        assert_eq!(s("%.*f", (2i64, 3.14159)), "3.14");
        assert_eq!(s("%s", (1i64,)), "1");
        assert_eq!(s("%d", (5i64,)), "5");

        // The mapping form: %(name)s addresses a dict RHS (CPython
        // verified — url.py's `_IPV6_PAT` built from `x % _subs`).
        #[cfg(feature = "std")]
        {
            let m: crate::PyDict<String, i64> = crate::PyDict::from([("a".to_string(), 1)]);
            assert_eq!(s("%(a)s", m.clone()), "1");
            let m: crate::PyDict<String, i64> = crate::PyDict::from([("a".to_string(), 255)]);
            assert_eq!(s("%(a)d", m.clone()), "255");
            assert_eq!(s("%%(a)s", crate::PyDict::<String, i64>::new()), "%(a)s");

            let e = "%(missing)s"
                .to_string()
                .py_mod(&crate::PyDict::<String, i64>::new())
                .unwrap_err();
            assert_eq!(e.exception_type, "KeyError");
            assert_eq!(e.message, "missing");
            let e = "%(a)s".to_string().py_mod(&(1i64,)).unwrap_err();
            assert_eq!(e.exception_type, "TypeError");
            assert!(e.message.contains("format requires a mapping"), "{}", e.message);
        }

        assert_eq!(b(b"%x", 255), b"ff");
        assert_eq!(b(b"%b", b"data".to_vec()), b"data");
        assert_eq!(b(b"%d", 5), b"5");
        assert_eq!(b(b"(%s)", b"x".to_vec()), b"(x)");
    }

    /// The loud failures match CPython's typed errors.
    #[test]
    fn percent_formatting_errors_match_cpython() {
        let e = "%b".to_string().py_mod(&b"x".to_vec()).unwrap_err();
        assert_eq!(e.exception_type, "ValueError");
        assert!(
            e.message.contains("unsupported format character 'b'"),
            "{}",
            e.message
        );
        let e = "%d".to_string().py_mod(&"x").unwrap_err();
        assert_eq!(e.exception_type, "TypeError");
        assert!(
            e.message.contains("a real number is required, not str"),
            "{}",
            e.message
        );
        let e = "%q".to_string().py_mod(&1i64).unwrap_err();
        assert_eq!(e.exception_type, "ValueError");
        assert!(
            e.message.contains("unsupported format character 'q'"),
            "{}",
            e.message
        );
        let e = "%s %s".to_string().py_mod(&("a",)).unwrap_err();
        assert_eq!(e.exception_type, "TypeError");
        assert!(
            e.message.contains("not enough arguments for format string"),
            "{}",
            e.message
        );
        let e = "%s".to_string().py_mod(&(1i64, 2i64)).unwrap_err();
        assert_eq!(e.exception_type, "TypeError");
        assert!(
            e.message.contains("not all arguments converted during string formatting"),
            "{}",
            e.message
        );
    }
}
