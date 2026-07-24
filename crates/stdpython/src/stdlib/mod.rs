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

/// Python io module (StringIO); PyFile itself lives at the crate root.
#[cfg(feature = "std")]
pub mod io;

/// Python argparse module: the runtime half of conversion-time parsers.
#[cfg(feature = "std")]
pub mod argparse;

/// Python hashlib module - message digests
pub mod hashlib;

/// Python csv module - CSV reading over line lists
pub mod csv;

