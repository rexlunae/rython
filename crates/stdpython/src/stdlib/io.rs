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

use alloc::string::{String, ToString};
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

/// io.BytesIO — a BINARY stream: the in-memory buffer io.BytesIO() gives,
/// and the disk handle a binary-mode `open()` gives (`open(p, "rb")`),
/// one type over both backends exactly as PyFile is for text (so
/// bytes-consuming code works against either). Python's cursor semantics
/// for the buffer: read() returns from the cursor and advances it,
/// write() OVERWRITES at the cursor (BytesIO(b"seeded").write(b"!")
/// yields b"!eeded"), and getvalue() returns the whole buffer regardless
/// of the cursor.
///
/// Like PyFile, a cheap-clone handle over one shared backend (a file
/// object is a reference in Python).
#[derive(Clone)]
pub struct PyBytesIO {
    inner: alloc::rc::Rc<core::cell::RefCell<BytesBackend>>,
    /// Python `f.name`: the path a disk file was opened from (a BytesIO
    /// has no name in Python; here it is "").
    pub name: String,
}

/// The binary file type of a binary-mode `open()` — the same type as
/// io.BytesIO (see [`PyBytesIO`]).
pub type PyBinaryFile = PyBytesIO;

enum BytesBackend {
    Buffer { data: Vec<u8>, pos: usize },
    /// A readable byte stream: a buffered disk file, or the live
    /// standard input (`FileType("rb")("-")`).
    #[cfg(feature = "std")]
    DiskRead(alloc::boxed::Box<dyn std::io::BufRead>),
    /// A writable byte stream: a buffered disk file, or the live
    /// standard output (`FileType("wb")("-")` — Devin review on #339).
    #[cfg(feature = "std")]
    DiskWrite(alloc::boxed::Box<dyn std::io::Write>),
    /// The live standard input (`sys.stdin.buffer`): reads through the
    /// process's one stdin buffer; the closed state is
    /// crate::STDIN_CLOSED, shared with the text handle (Devin review on
    /// #339, round 5).
    #[cfg(feature = "std")]
    Stdin,
    /// The live standard output (`sys.stdout.buffer`).
    #[cfg(feature = "std")]
    Stdout,
    Closed,
}

/// io.BytesIO(): an empty in-memory binary buffer.
#[allow(non_snake_case)]
pub fn BytesIO() -> PyBytesIO {
    PyBytesIO::new_buffer(Vec::new())
}

/// io.BytesIO(initial): seeded with bytes, cursor at the START.
#[allow(non_snake_case)]
pub fn BytesIO_seeded<B: AsRef<[u8]>>(initial: B) -> PyBytesIO {
    PyBytesIO::new_buffer(initial.as_ref().to_vec())
}

impl PyBytesIO {
    fn from_backend(backend: BytesBackend, name: &str) -> Self {
        Self {
            inner: alloc::rc::Rc::new(core::cell::RefCell::new(backend)),
            name: name.to_string(),
        }
    }

    fn new_buffer(data: Vec<u8>) -> Self {
        Self::from_backend(BytesBackend::Buffer { data, pos: 0 }, "")
    }

    #[cfg(feature = "std")]
    pub(crate) fn new_disk_read(reader: impl std::io::BufRead + 'static, name: &str) -> Self {
        Self::from_backend(BytesBackend::DiskRead(alloc::boxed::Box::new(reader)), name)
    }

    #[cfg(feature = "std")]
    pub(crate) fn new_disk_write(writer: impl std::io::Write + 'static, name: &str) -> Self {
        Self::from_backend(BytesBackend::DiskWrite(alloc::boxed::Box::new(writer)), name)
    }

    /// The live standard input as a binary file (`sys.stdin.buffer`;
    /// argparse's `FileType("rb")("-")`).
    #[cfg(feature = "std")]
    pub fn stdin() -> Self {
        Self::from_backend(BytesBackend::Stdin, "<stdin>")
    }

    /// The live standard output as a binary file (`sys.stdout.buffer`;
    /// argparse's `FileType("wb")("-")`): close() flushes and leaves the
    /// descriptor open.
    #[cfg(feature = "std")]
    pub fn stdout() -> Self {
        Self::from_backend(BytesBackend::Stdout, "<stdout>")
    }

    /// Python `f.closed`: whether close() ran on this stream (through
    /// any alias of it).
    pub fn closed(&self) -> bool {
        match &*self.inner.borrow() {
            BytesBackend::Closed => true,
            #[cfg(feature = "std")]
            BytesBackend::Stdin => crate::stdin_closed(),
            #[cfg(feature = "std")]
            BytesBackend::Stdout => crate::stdout_closed(),
            _ => false,
        }
    }

    /// Python `b.read()`: the remaining bytes from the cursor (a disk
    /// handle: the rest of the file).
    pub fn read(&self) -> Result<Vec<u8>, PyException> {
        match &mut *self.inner.borrow_mut() {
            BytesBackend::Buffer { data, pos } => {
                let out = data[*pos..].to_vec();
                *pos = data.len();
                Ok(out)
            }
            #[cfg(feature = "std")]
            BytesBackend::DiskRead(reader) => {
                use std::io::Read;
                let mut out = Vec::new();
                reader
                    .read_to_end(&mut out)
                    .map_err(|e| crate::stream_error(&e))?;
                Ok(out)
            }
            #[cfg(feature = "std")]
            BytesBackend::Stdin => {
                use std::io::Read;
                if crate::stdin_closed() {
                    return Err(crate::closed_file_error());
                }
                let mut out = Vec::new();
                std::io::stdin()
                    .lock()
                    .read_to_end(&mut out)
                    .map_err(|e| crate::stream_error(&e))?;
                Ok(out)
            }
            #[cfg(feature = "std")]
            BytesBackend::DiskWrite(_) => Err(crate::unsupported_operation("read")),
            #[cfg(feature = "std")]
            BytesBackend::Stdout => Err(if crate::stdout_closed() {
                crate::closed_file_error()
            } else {
                crate::unsupported_operation("read")
            }),
            BytesBackend::Closed => Err(crate::closed_file_error()),
        }
    }

    /// Python `b.write(data)`: overwrite at the cursor (a disk handle:
    /// append to the stream), return the byte count written.
    pub fn write<B: AsRef<[u8]>>(&self, data: B) -> Result<i64, PyException> {
        let bytes = data.as_ref();
        match &mut *self.inner.borrow_mut() {
            BytesBackend::Buffer { data, pos } => {
                let end = *pos + bytes.len();
                if end > data.len() {
                    data.resize(end, 0);
                }
                data[*pos..end].copy_from_slice(bytes);
                *pos = end;
                Ok(bytes.len() as i64)
            }
            #[cfg(feature = "std")]
            BytesBackend::DiskWrite(writer) => {
                use std::io::Write;
                writer
                    .write_all(bytes)
                    .map_err(|e| crate::stream_error(&e))?;
                Ok(bytes.len() as i64)
            }
            #[cfg(feature = "std")]
            BytesBackend::Stdout => {
                use std::io::Write;
                if crate::stdout_closed() {
                    return Err(crate::closed_file_error());
                }
                std::io::stdout()
                    .lock()
                    .write_all(bytes)
                    .map_err(|e| crate::stream_error(&e))?;
                Ok(bytes.len() as i64)
            }
            #[cfg(feature = "std")]
            BytesBackend::DiskRead(_) => Err(crate::unsupported_operation("write")),
            #[cfg(feature = "std")]
            BytesBackend::Stdin => Err(if crate::stdin_closed() {
                crate::closed_file_error()
            } else {
                crate::unsupported_operation("write")
            }),
            BytesBackend::Closed => Err(crate::closed_file_error()),
        }
    }

    /// Python `b.getvalue()`: the whole buffer regardless of the cursor
    /// (a BytesIO method; a disk handle has none — Python raises
    /// AttributeError).
    pub fn getvalue(&self) -> Result<Vec<u8>, PyException> {
        match &*self.inner.borrow() {
            BytesBackend::Buffer { data, .. } => Ok(data.clone()),
            BytesBackend::Closed => Err(crate::closed_file_error()),
            // CPython's AttributeError names the receiver's class: a
            // read handle is a BufferedReader, a write handle a
            // BufferedWriter (Devin review on #339, round 5).
            #[cfg(feature = "std")]
            BytesBackend::DiskRead(_) | BytesBackend::Stdin => Err(crate::attribute_error(
                "'_io.BufferedReader' object has no attribute 'getvalue'",
            )),
            #[cfg(feature = "std")]
            BytesBackend::DiskWrite(_) | BytesBackend::Stdout => Err(crate::attribute_error(
                "'_io.BufferedWriter' object has no attribute 'getvalue'",
            )),
        }
    }

    /// Python `b.flush()`: push buffered writes to the stream.
    pub fn flush(&self) -> Result<(), PyException> {
        match &mut *self.inner.borrow_mut() {
            #[cfg(feature = "std")]
            BytesBackend::DiskWrite(writer) => {
                use std::io::Write;
                writer
                    .flush()
                    .map_err(|e| crate::stream_error(&e))
            }
            #[cfg(feature = "std")]
            BytesBackend::Stdout => {
                use std::io::Write;
                if crate::stdout_closed() {
                    return Err(crate::closed_file_error());
                }
                std::io::stdout()
                    .flush()
                    .map_err(|e| crate::stream_error(&e))
            }
            #[cfg(feature = "std")]
            BytesBackend::Stdin if crate::stdin_closed() => Err(crate::closed_file_error()),
            BytesBackend::Closed => Err(crate::closed_file_error()),
            _ => Ok(()),
        }
    }

    /// Python `b.close()`.
    pub fn close(&self) -> Result<(), PyException> {
        // The standard streams close process-wide and keep the
        // descriptor open (see PyFile::close).
        #[cfg(feature = "std")]
        match &*self.inner.borrow() {
            BytesBackend::Stdin => {
                crate::STDIN_CLOSED.store(true, core::sync::atomic::Ordering::SeqCst);
                return Ok(());
            }
            BytesBackend::Stdout => {
                use std::io::Write;
                if !crate::stdout_closed() {
                    std::io::stdout()
                        .flush()
                        .map_err(|e| crate::stream_error(&e))?;
                    crate::STDOUT_CLOSED.store(true, core::sync::atomic::Ordering::SeqCst);
                }
                return Ok(());
            }
            _ => {}
        }
        let old = core::mem::replace(&mut *self.inner.borrow_mut(), BytesBackend::Closed);
        #[cfg(feature = "std")]
        if let BytesBackend::DiskWrite(mut writer) = old {
            use std::io::Write;
            writer.flush().map_err(|e| crate::stream_error(&e))?;
        }
        #[cfg(not(feature = "std"))]
        let _ = old;
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
