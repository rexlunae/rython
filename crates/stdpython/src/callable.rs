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
//! The handle is reference-counted and thread-safe (`Arc` over a
//! `Fn + Send + Sync`), matching Python's own "a function object is
//! shared, not copied". Thread-safety is not decoration: a module-level
//! registry of callbacks (`_INITIALIZERS = []` — issue #122) lowers to a
//! `static`, and a Rust static must be `Sync`, so an `Rc` handle could
//! not be stored at module level at all.

#[cfg(feature = "alloc")]
use alloc::{format, string::String, sync::Arc};

use crate::{PyDisplay, PyException};
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};

/// A Python callable held as a value.
///
/// `A` is the argument tuple; `R` is the return type. Construct one with
/// [`PyCallable::new`] and invoke it with [`PyCallable::call`].
pub struct PyCallable<A, R> {
    /// The Python name the callable was defined under (`add`, `<lambda>`),
    /// used by `repr()`/`str()` and by the message of a call that raises.
    name: Arc<str>,
    f: Arc<CallableFn<A, R>>,
}

/// The wrapped Rust closure. On the `std` tier it is `Send + Sync`, so a
/// module-level registry of callables can live in a `static` (which Rust
/// requires to be `Sync`). The `alloc`-only tier has no threads and no
/// module-level mutable statics, and its closure cells are `RefCell`s,
/// which are not `Sync` — so the bound is dropped there rather than
/// making every no_std closure uncallable.
#[cfg(feature = "std")]
type CallableFn<A, R> = dyn Fn(A) -> Result<R, PyException> + Send + Sync;
#[cfg(not(feature = "std"))]
type CallableFn<A, R> = dyn Fn(A) -> Result<R, PyException>;

impl<A, R> PyCallable<A, R> {
    /// Wrap a Rust closure as a Python callable value. `name` is the
    /// Python-side name (a `def`'s name, or `<lambda>`).
    #[cfg(feature = "std")]
    pub fn new<F>(name: &str, f: F) -> Self
    where
        F: Fn(A) -> Result<R, PyException> + Send + Sync + 'static,
    {
        Self { name: Arc::from(name), f: Arc::new(f) }
    }

    /// Wrap a Rust closure as a Python callable value (no_std tier — see
    /// [`CallableFn`] for why the thread bounds are absent).
    #[cfg(not(feature = "std"))]
    pub fn new<F>(name: &str, f: F) -> Self
    where
        F: Fn(A) -> Result<R, PyException> + 'static,
    {
        Self { name: Arc::from(name), f: Arc::new(f) }
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
        Arc::as_ptr(&self.f) as *const () as usize
    }
}

/// Cloning a callable shares the underlying function, as binding a Python
/// function to a second name does — it never copies the closure's
/// captures.
impl<A, R> Clone for PyCallable<A, R> {
    fn clone(&self) -> Self {
        Self { name: Arc::clone(&self.name), f: Arc::clone(&self.f) }
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
        Arc::ptr_eq(&self.f, &other.f)
    }
}

impl<A, R> Eq for PyCallable<A, R> {}

#[cfg(test)]
mod tests {
    use super::*;
    // The no_std tier has no prelude: the alloc items these tests use are
    // named explicitly, as the rest of the crate names them.
    use alloc::string::ToString;
    use alloc::{vec, vec::Vec};

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
        let table: Vec<PyCallable<(i64,), i64>> = vec![
            PyCallable::new("double", |(x,)| Ok(x * 2)),
            PyCallable::new("square", |(x,)| Ok(x * x)),
        ];
        let applied: Vec<i64> =
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

/// A closure CELL: the binding a nested function captures.
///
/// Python's closure is LATE-BINDING. The nested function does not copy
/// the enclosing local, it holds the cell the name is bound in and reads
/// it when it is CALLED — so
///
/// ```python
/// x = 1
/// f = lambda: x
/// x = 2
/// f()          # 2, not 1
/// ```
///
/// and a container the enclosing scope mutates after the `def`, and a
/// loop variable every closure built in the loop shares. rython's locals
/// are values, so a captured name whose binding can still change is held
/// in a `PyCell` instead: cloning it (which is what the closure's `move`
/// capture does) shares the cell, and both sides read and write one
/// binding. A capture that is bound once, unconditionally, and never
/// mutated is cloned in — no later value exists for the closure to have
/// missed, so the clone and the cell cannot be told apart.
///
/// The cell starts EMPTY, as Python's local does before its first
/// assignment: reading it then is CPython's `UnboundLocalError`.
///
/// The slot is a `Mutex` on the `std` tier — a module-level registry of
/// callables is a `static`, and a Rust static must be `Sync`. The
/// `alloc`-only tier has no `std::sync` at all, and cannot reach that
/// shape either (module-level mutable statics are refused under
/// `--no-std`), so it holds the slot in a `RefCell` instead. The two
/// differ only in what they do under concurrent access, which the
/// no_std tier does not have.
pub struct PyCell<T> {
    name: Arc<str>,
    slot: Arc<CellSlot<T>>,
}

#[cfg(feature = "std")]
type CellSlot<T> = std::sync::Mutex<Option<T>>;
#[cfg(not(feature = "std"))]
type CellSlot<T> = core::cell::RefCell<Option<T>>;

#[cfg(feature = "std")]
type CellGuard<'a, T> = std::sync::MutexGuard<'a, Option<T>>;
#[cfg(not(feature = "std"))]
type CellGuard<'a, T> = core::cell::RefMut<'a, Option<T>>;

impl<T> PyCell<T> {
    /// The cell of a local that has not been assigned yet.
    pub fn empty(name: &str) -> Self {
        Self { name: Arc::from(name), slot: Arc::new(CellSlot::new(None)) }
    }

    /// The cell of a name that is already bound (a captured parameter).
    pub fn new(name: &str, value: T) -> Self {
        Self { name: Arc::from(name), slot: Arc::new(CellSlot::new(Some(value))) }
    }

    /// Bind the name — an assignment to it in any scope that holds the
    /// cell.
    pub fn set(&self, value: T) {
        *self.lock() = Some(value);
    }

    /// Borrow the bound object — the receiver of a read.
    pub fn borrow(&self) -> PyCellRef<'_, T> {
        PyCellRef { guard: self.lock(), name: &self.name }
    }

    /// Borrow the bound object mutably — the receiver of a store.
    pub fn borrow_mut(&self) -> PyCellRef<'_, T> {
        self.borrow()
    }

    /// A poisoned cell is recovered rather than propagated: the closure
    /// that panicked left the binding in whatever state it reached, which
    /// is what CPython leaves behind, and poisoning every later access
    /// would turn one loud failure into an unrelated cascade.
    #[cfg(feature = "std")]
    fn lock(&self) -> CellGuard<'_, T> {
        self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The no_std tier's slot: a `RefCell`, borrowed for the access.
    #[cfg(not(feature = "std"))]
    fn lock(&self) -> CellGuard<'_, T> {
        self.slot.borrow_mut()
    }
}

/// A borrow of a cell's bound object. Dereferencing one before the name's
/// first assignment is CPython's `UnboundLocalError`.
pub struct PyCellRef<'a, T> {
    guard: CellGuard<'a, T>,
    name: &'a str,
}

impl<T> core::ops::Deref for PyCellRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.guard.as_ref().unwrap_or_else(|| unbound(self.name))
    }
}

impl<T> core::ops::DerefMut for PyCellRef<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        let name = self.name;
        self.guard.as_mut().unwrap_or_else(|| unbound(name))
    }
}

/// CPython's message for reading a local before its first assignment.
fn unbound(name: &str) -> ! {
    panic!(
        "UnboundLocalError: cannot access local variable '{}' where it is not \
         associated with a value",
        name
    )
}

impl<T: Clone> PyCell<T> {
    /// The value the cell holds, as a name READ in value position yields
    /// it: a snapshot clone taken at the read, so a closure sees whatever
    /// the binding holds when it runs.
    pub fn get(&self) -> T {
        (*self.borrow()).clone()
    }
}

/// Cloning a cell SHARES it — that is the whole point: the closure's
/// `move` capture takes a clone and still reads and writes the enclosing
/// scope's binding.
impl<T> Clone for PyCell<T> {
    fn clone(&self) -> Self {
        Self { name: Arc::clone(&self.name), slot: Arc::clone(&self.slot) }
    }
}

impl<T: PyDisplay> PyDisplay for PyCell<T> {
    fn py_display(&self) -> String {
        self.borrow().py_display()
    }
}

impl<T: Debug> Debug for PyCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&*self.borrow(), f)
    }
}

impl<T: PartialEq> PartialEq for PyCell<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.borrow() == *other.borrow()
    }
}

#[cfg(test)]
mod cell_tests {
    use super::*;

    #[test]
    fn a_cell_is_shared_by_its_clones() {
        let counter: PyCell<i64> = PyCell::new("counter", 0);
        let captured = counter.clone();
        let bump: PyCallable<(), i64> = PyCallable::new("bump", move |()| {
            let next = captured.get() + 1;
            captured.set(next);
            Ok(next)
        });
        bump.call(()).unwrap();
        bump.call(()).unwrap();
        assert_eq!(bump.call(()).unwrap(), 3);
        // The enclosing scope sees every write.
        assert_eq!(counter.get(), 3);
    }

    #[test]
    fn a_closure_reads_the_binding_at_call_time() {
        // Python's late binding: `x = 1; f = lambda: x; x = 2; f()` is 2.
        let x: PyCell<i64> = PyCell::new("x", 1);
        let captured = x.clone();
        let f: PyCallable<(), i64> = PyCallable::new("<lambda>", move |()| Ok(captured.get()));
        x.set(2);
        assert_eq!(f.call(()).unwrap(), 2);
    }

    #[test]
    fn every_closure_built_in_a_loop_shares_the_loop_binding() {
        // `for i in range(3): fs.append(lambda: i)` gives [2, 2, 2].
        let i: PyCell<i64> = PyCell::empty("i");
        let mut fs: alloc::vec::Vec<PyCallable<(), i64>> = alloc::vec::Vec::new();
        for n in 0..3 {
            i.set(n);
            let captured = i.clone();
            fs.push(PyCallable::new("<lambda>", move |()| Ok(captured.get())));
        }
        let seen: alloc::vec::Vec<i64> = fs.iter().map(|f| f.call(()).unwrap()).collect();
        assert_eq!(seen, alloc::vec![2, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "UnboundLocalError: cannot access local variable 'x'")]
    fn reading_an_unbound_cell_is_cpythons_unbound_local_error() {
        let x: PyCell<i64> = PyCell::empty("x");
        let _ = x.get();
    }
}
