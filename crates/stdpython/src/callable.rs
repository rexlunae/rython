//! Callables as VALUES (issue #122): the runtime type behind a
//! `Callable[[A, B], R]` annotation, a `lambda`, and a nested `def`.
//!
//! rython's other callables are resolved at conversion time — a call names
//! a function and lowers to that function. A callable held in a variable,
//! a container or a parameter cannot be: the target is only known at run
//! time. `PyCallable` is that runtime target, and the ONE representation
//! of every callable value the converter emits.
//!
//! The argument list is a TUPLE type parameter (`()`, `(i64,)`,
//! `(String, i64)`), so a single type serves every arity and the
//! signature stays fully typed — `Callable[[int], int]` is
//! `PyCallable<(i64,), i64>`, not an erased bag of boxed values. A call
//! through the value is `f.call((x,))`, which returns the same
//! `Result<R, PyException>` every generated function returns, so an
//! exception raised inside a callable propagates through its caller
//! exactly as one raised by a directly-named function does.
//!
//! The handle is reference-counted and single-threaded (`Rc`, matching
//! Python's own "a function object is shared, not copied"): a callable
//! value is deliberately NOT `Send`, so handing one to a thread is a
//! build error rather than a silent divergence.

#[cfg(feature = "alloc")]
use alloc::{format, rc::Rc, string::String};

use core::cell::{Ref, RefCell, RefMut};

use crate::{PyDisplay, PyException};
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};

/// A Python callable held as a value.
///
/// `A` is the argument tuple; `R` is the return type. Construct one with
/// [`PyCallable::new`] and invoke it with [`PyCallable::call`].
pub struct PyCallable<A, R> {
    /// The Python name the callable was defined under (`add`, `<lambda>`),
    /// used by `repr()`/`str()` and by the message of a call that raises.
    name: Rc<str>,
    f: Rc<dyn Fn(A) -> Result<R, PyException>>,
}

impl<A, R> PyCallable<A, R> {
    /// Wrap a Rust closure as a Python callable value. `name` is the
    /// Python-side name (a `def`'s name, or `<lambda>`).
    pub fn new<F>(name: &str, f: F) -> Self
    where
        F: Fn(A) -> Result<R, PyException> + 'static,
    {
        Self { name: Rc::from(name), f: Rc::new(f) }
    }

    /// Call the value: `f(x, y)` in Python is `f.call((x, y))` here.
    pub fn call(&self, args: A) -> Result<R, PyException> {
        (self.f)(args)
    }

    /// The Python name the callable was defined under (`__name__`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The identity CPython prints in a function's repr. Two clones of
    /// one callable share it; two separately created closures do not.
    fn addr(&self) -> usize {
        Rc::as_ptr(&self.f) as *const () as usize
    }
}

/// Cloning a callable shares the underlying function, as binding a Python
/// function to a second name does — it never copies the closure's
/// captures.
impl<A, R> Clone for PyCallable<A, R> {
    fn clone(&self) -> Self {
        Self { name: Rc::clone(&self.name), f: Rc::clone(&self.f) }
    }
}

/// CPython's function repr, with the same shape and a real address:
/// `<function add at 0x7f9c8c0d3e20>`. The qualified name CPython prints
/// for a nested function (`make_adder.<locals>.add`) is not modeled — the
/// plain name is used (§12.3).
impl<A, R> Display for PyCallable<A, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "<function {} at 0x{:012x}>", self.name, self.addr())
    }
}

impl<A, R> Debug for PyCallable<A, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(self, f)
    }
}

impl<A, R> PyDisplay for PyCallable<A, R> {
    fn py_display(&self) -> String {
        format!("{}", self)
    }
}

/// Python compares function objects by IDENTITY: `f == f` is true, and two
/// closures built from the same `lambda` are distinct objects.
impl<A, R> PartialEq for PyCallable<A, R> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.f, &other.f)
    }
}

impl<A, R> Eq for PyCallable<A, R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_callable_value_calls_through() {
        let add1: PyCallable<(i64,), i64> = PyCallable::new("add1", |(x,)| Ok(x + 1));
        assert_eq!(add1.call((41,)).unwrap(), 42);
        assert_eq!(add1.name(), "add1");
    }

    #[test]
    fn a_closure_captures_its_environment() {
        fn make_adder(n: i64) -> PyCallable<(i64,), i64> {
            PyCallable::new("add", move |(x,)| Ok(x + n))
        }
        let add5 = make_adder(5);
        let add10 = make_adder(10);
        assert_eq!(add5.call((1,)).unwrap(), 6);
        assert_eq!(add10.call((1,)).unwrap(), 11);
    }

    #[test]
    fn an_exception_propagates_out_of_a_call() {
        let boom: PyCallable<(), i64> =
            PyCallable::new("boom", |()| Err(PyException::new("ValueError", "no")));
        let e = boom.call(()).unwrap_err();
        assert_eq!(e.to_string(), "ValueError: no");
    }

    #[test]
    fn cloning_shares_the_function_and_its_identity() {
        let f: PyCallable<(i64,), i64> = PyCallable::new("f", |(x,)| Ok(x * 2));
        let g = f.clone();
        assert!(f == g);
        let h: PyCallable<(i64,), i64> = PyCallable::new("f", |(x,)| Ok(x * 2));
        assert!(f != h);
    }

    #[test]
    fn callables_live_in_containers_and_dispatch() {
        let table: vec::Vec<PyCallable<(i64,), i64>> = vec![
            PyCallable::new("double", |(x,)| Ok(x * 2)),
            PyCallable::new("square", |(x,)| Ok(x * x)),
        ];
        let applied: vec::Vec<i64> =
            table.iter().map(|f| f.call((5,)).unwrap()).collect();
        assert_eq!(applied, vec![10, 25]);
    }

    #[test]
    fn the_repr_is_cpythons_shape() {
        let f: PyCallable<(), ()> = PyCallable::new("go", |()| Ok(()));
        let shown = f.py_display();
        assert!(shown.starts_with("<function go at 0x"), "{}", shown);
        assert!(shown.ends_with('>'), "{}", shown);
    }
}

/// A closure CELL: the enclosing local a nested function mutates.
///
/// Python's closure shares objects, it does not copy them — `counter` in
///
/// ```python
/// counter = {"n": 0}
/// def bump() -> int:
///     counter["n"] += 1
///     return counter["n"]
/// ```
///
/// is ONE dict that `bump` and the enclosing scope both see. rython's
/// containers are values, so a name a nested closure mutates is held in
/// a `PyCell` instead: cloning it (which is what the closure's `move`
/// capture does) shares the cell, and both sides read and write one
/// object. Only names a closure actually mutates become cells — a name
/// it merely reads is cloned in, which is what Python's cell gives it for
/// an immutable object anyway.
pub struct PyCell<T>(Rc<RefCell<T>>);

impl<T> PyCell<T> {
    pub fn new(value: T) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }

    /// Borrow the shared object — the receiver of a read.
    pub fn borrow(&self) -> Ref<'_, T> {
        self.0.borrow()
    }

    /// Borrow the shared object mutably — the receiver of a store.
    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        self.0.borrow_mut()
    }
}

impl<T: Clone> PyCell<T> {
    /// The value the cell holds, as a name READ in value position yields
    /// it: a snapshot clone, the same reading a promoted module static
    /// gets.
    pub fn get(&self) -> T {
        self.0.borrow().clone()
    }
}

/// Cloning a cell SHARES it — that is the whole point: the closure's
/// `move` capture takes a clone and still writes the enclosing scope's
/// object.
impl<T> Clone for PyCell<T> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<T: PyDisplay> PyDisplay for PyCell<T> {
    fn py_display(&self) -> String {
        self.0.borrow().py_display()
    }
}

impl<T: Debug> Debug for PyCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&*self.0.borrow(), f)
    }
}

impl<T: PartialEq> PartialEq for PyCell<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.0.borrow() == *other.0.borrow()
    }
}

#[cfg(test)]
mod cell_tests {
    use super::*;

    #[test]
    fn a_cell_is_shared_by_its_clones() {
        let counter: PyCell<i64> = PyCell::new(0);
        let captured = counter.clone();
        let bump: PyCallable<(), i64> = PyCallable::new("bump", move |()| {
            *captured.borrow_mut() += 1;
            Ok(*captured.borrow())
        });
        bump.call(()).unwrap();
        bump.call(()).unwrap();
        assert_eq!(bump.call(()).unwrap(), 3);
        // The enclosing scope sees every write.
        assert_eq!(counter.get(), 3);
    }
}
