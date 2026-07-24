//! Python io module implementation
//!
//! io.StringIO: an in-memory text buffer sharing the PyFile surface, so
//! anything that writes to a file (csv.writer among others) can write
//! to memory instead, exactly as in Python. The cursor semantics are
//! Python's: read/readline advance it, write OVERWRITES at it, and
//! getvalue() returns the whole buffer regardless of it. BytesIO and
//! the wrapper classes are not implemented yet.

use crate::PyFile;

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
