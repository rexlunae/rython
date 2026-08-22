//! Python io module implementation
//!
//! io.StringIO: an in-memory text buffer sharing the PyFile surface, so
//! anything that writes to a file (csv.writer among others) can write
//! to memory instead, exactly as in Python. The cursor semantics are
//! Python's: read/readline advance it, write OVERWRITES at it, and
//! getvalue() returns the whole buffer regardless of it. BytesIO and
//! the wrapper classes are not implemented yet — the BytesIO NAME exists
//! so `from io import BytesIO` imports resolve (constructions lower to
//! the boxed PyValue, the file-object divergence), and IOBase is the
//! abstract stream base — a typing marker, never constructed.

use crate::PyFile;

/// io.BytesIO — no binary in-memory buffer in rython (constructions lower
/// to the boxed PyValue); the item exists so imports resolve.
#[allow(non_snake_case)]
pub struct BytesIO;

/// io.IOBase — the abstract base of the io stream types; a typing marker.
pub struct IOBase;

/// io.StringIO(): an empty in-memory text buffer.
#[allow(non_snake_case)]
pub fn StringIO() -> PyFile {
    PyFile::new_buffer("")
}

/// io.StringIO(initial): seeded with text, cursor at the START — so an
/// immediate write() overwrites the seed, as in Python.
#[allow(non_snake_case)]
pub fn StringIO_seeded<S: AsRef<str> + ?Sized>(initial: &S) -> PyFile {
    PyFile::new_buffer(initial.as_ref())
}
