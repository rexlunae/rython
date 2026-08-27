//! Python io module implementation
//!
//! io.StringIO: an in-memory text buffer sharing the PyFile surface, so
//! anything that writes to a file (csv.writer among others) can write
//! to memory instead, exactly as in Python. The cursor semantics are
//! Python's: read/readline advance it, write OVERWRITES at it, and
//! getvalue() returns the whole buffer regardless of it.
//!
//! io.BytesIO: the binary sibling — a `Vec<u8>` buffer with the same
//! cursor discipline, whose read()/getvalue() return Python bytes.
//!
//! Both buffers are pure in-memory types, so this module lives on EVERY
//! tier: the alloc/no_std profile's "file I/O" is exactly these (a target
//! with no OS has no disk files — `open()` and the disk-backed PyFile
//! constructors stay std-only).

use alloc::vec::Vec;

use crate::{PyException, PyFile};

/// io.DEFAULT_BUFFER_SIZE (CPython: 8192).
pub const DEFAULT_BUFFER_SIZE: i64 = 8192;

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

/// io.BytesIO — an in-memory BINARY buffer. Python's cursor semantics:
/// read() returns from the cursor and advances it, write() OVERWRITES at
/// the cursor (BytesIO(b"seeded").write(b"!") yields b"!eeded"), and
/// getvalue() returns the whole buffer regardless of the cursor.
pub struct PyBytesIO {
    data: Vec<u8>,
    pos: usize,
    closed: bool,
}

/// io.BytesIO(): an empty in-memory binary buffer.
#[allow(non_snake_case)]
pub fn BytesIO() -> PyBytesIO {
    PyBytesIO {
        data: Vec::new(),
        pos: 0,
        closed: false,
    }
}

/// io.BytesIO(initial): seeded with bytes, cursor at the START.
#[allow(non_snake_case)]
pub fn BytesIO_seeded<B: AsRef<[u8]>>(initial: B) -> PyBytesIO {
    PyBytesIO {
        data: initial.as_ref().to_vec(),
        pos: 0,
        closed: false,
    }
}

impl PyBytesIO {
    fn check_open(&self) -> Result<(), PyException> {
        if self.closed {
            return Err(crate::closed_file_error());
        }
        Ok(())
    }

    /// Python `b.read()`: the remaining bytes from the cursor.
    pub fn read(&mut self) -> Result<Vec<u8>, PyException> {
        self.check_open()?;
        let out = self.data[self.pos..].to_vec();
        self.pos = self.data.len();
        Ok(out)
    }

    /// Python `b.write(data)`: overwrite at the cursor, return the byte
    /// count written.
    pub fn write<B: AsRef<[u8]>>(&mut self, data: B) -> Result<i64, PyException> {
        self.check_open()?;
        let bytes = data.as_ref();
        let end = self.pos + bytes.len();
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(bytes.len() as i64)
    }

    /// Python `b.getvalue()`: the whole buffer regardless of the cursor.
    pub fn getvalue(&self) -> Result<Vec<u8>, PyException> {
        self.check_open()?;
        Ok(self.data.clone())
    }

    /// Python `b.close()`.
    pub fn close(&mut self) -> Result<(), PyException> {
        self.closed = true;
        Ok(())
    }
}

/// io.IOBase — the abstract base of the io stream types; a typing marker.
pub struct IOBase;

/// io.UnsupportedOperation — the exception for unsupported file
/// operations. rython's exceptions are string-tagged PyException values
/// (the codegen matches `except UnsupportedOperation` by name), so the
/// class is a marker with no runtime shape.
pub struct UnsupportedOperation;
