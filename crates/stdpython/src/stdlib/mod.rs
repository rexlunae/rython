//! Python Standard Library Implementation
//! 
//! This module contains implementations of Python's standard library modules
//! that are commonly used in Python programs. Each submodule provides
//! functionality equivalent to the corresponding Python module.

/// Python sys module - system-specific parameters and functions
#[cfg(feature = "std")]
pub mod sys;

/// Python os module - operating system interface
#[cfg(feature = "std")]
pub mod os;

/// Python subprocess module - subprocess management
#[cfg(feature = "std")]
pub mod subprocess;

/// Python sysconfig module - configuration information
#[cfg(feature = "std")]
pub mod sysconfig;

/// Python venv module - virtual environment creation
#[cfg(feature = "std")]
pub mod venv;

/// Python math module - mathematical functions.
/// std-gated: float math (sin, sqrt, …) lives on std's f64, not core's.
/// A no_std tier would need libm — tracked in issue #22.
#[cfg(feature = "std")]
pub mod math;

/// Python random module - random number generation
#[cfg(feature = "std")]
pub mod random;

/// Python datetime module - date and time handling
#[cfg(feature = "std")]
pub mod datetime;

/// Python time module - wall-clock and monotonic time, sleep
#[cfg(feature = "std")]
pub mod time;

/// Python asyncio module - a thin mapping onto the tokio runtime
/// (asyncio.run / asyncio.sleep). Feature-gated: only generated async
/// binaries enable `async-tokio`.
#[cfg(feature = "async-tokio")]
pub mod asyncio;

/// Python string module - string constants and classes
pub mod string;

/// Python json module - JSON encoder and decoder
pub mod json;

/// Python collections module - specialized container datatypes
pub mod collections;

/// Python itertools module - functions creating iterators for efficient looping
pub mod itertools;

/// Python pathlib module - object-oriented filesystem paths
#[cfg(feature = "std")]
pub mod pathlib;

/// Python tempfile module - temporary files and directories
#[cfg(feature = "std")]
pub mod tempfile;

/// Python glob module - Unix shell-style pathname pattern expansion
#[cfg(feature = "std")]
pub mod glob;

/// Python warnings module - diagnostics (issue #111). The HOOKS (filter
/// action, warn) live on every tier; the stderr OUTPUT is std-only, and
/// the alloc tier simply has no default output (warn is a no-op there,
/// like `log` with no logger).
pub mod warnings;

/// Python functools module - higher-order functions (reduce)
pub mod functools;

/// Python heapq module - heap queue algorithm on plain lists
pub mod heapq;

/// Python copy module - shallow and deep copies
pub mod copy;

/// Python textwrap module - text dedent/indent helpers
pub mod textwrap;

/// Python re module - regular expressions (regex-crate backed)
#[cfg(feature = "std")]
pub mod re;

/// Python io module (StringIO/BytesIO); PyFile itself lives at the crate
/// root. In-memory buffers are pure alloc, so the module lives on every
/// tier — only the DISK backends of PyFile (and `open()`) are std-gated.
pub mod io;

/// Python argparse module: the runtime half of conversion-time parsers.
#[cfg(feature = "std")]
pub mod argparse;

/// Python hashlib module - message digests
pub mod hashlib;

/// Python csv module - CSV reading over line lists
pub mod csv;

/// Python codec layer: str.encode / bytes.decode for ascii and punycode
/// (RFC 3492), with CPython's error classes. Pure data transformation —
/// no OS, so it lives on every tier.
pub mod codec;

/// Python numpy subset — dense N-dimensional arrays with broadcasting,
/// ufuncs, reductions, and a small linalg module. Optional feature
/// `numpy` pulls in the sequential engine; `numpy-rayon`, `numpy-simd`,
/// `numpy-cuda`, and `numpy-vulkan` add accelerated backends.
#[cfg(feature = "std")]
pub mod numpy;

/// Python threading module - thread management (Thread) and
/// synchronization (Lock, RLock, Event, Semaphore) on std::thread.
/// std-gated: threads need an OS.
#[cfg(feature = "std")]
pub mod threading;

/// Python socket module - TCP/UDP sockets on std::net.
/// std-gated: sockets need an OS.
#[cfg(feature = "std")]
pub mod socket;

/// Python ssl module - client-side TLS wrapped over the rustls crate.
/// Feature-gated per the platform-surface convention (`ssl-rustls`),
/// but ON by default: TLS is load-bearing for the top converted packages.
#[cfg(any(feature = "ssl-rustls", feature = "ssl-openssl"))]
pub mod ssl;

/// Python urllib package (urllib.request) - HTTP(S) client wrapped over
/// the ureq crate. Feature-gated per the platform-surface convention:
/// only crates that import urllib.request enable `http-ureq` (rypip does
/// this automatically for converted packages).
#[cfg(feature = "http-ureq")]
pub mod urllib;

