// The JIT boundary's C-compatible vocabulary (issue #166, docs/jit.md "The
// boundary"). This exact text is compiled twice: into the host
// (`rython_jit::abi`) and, verbatim, into every library rypip's cdylib
// mode generates (`rython_jit::abi::ABI_SOURCE`), so the two sides of the
// boundary cannot drift. No Rust-layout type crosses it: each library
// links its own `std` and `stdpython`, whose layouts and statics the host
// does not share.

/// The boundary's version. A library reporting another one is rebuilt,
/// never called.
pub const RY_ABI_VERSION: u32 = 1;

/// `RyValue::tag`: which field of the value is meaningful.
pub const RY_TAG_NONE: u32 = 0;
pub const RY_TAG_BOOL: u32 = 1;
pub const RY_TAG_INT: u32 = 2;
pub const RY_TAG_FLOAT: u32 = 3;
pub const RY_TAG_STR: u32 = 4;

/// One value crossing the boundary: `None`, `bool`, `int` (as i64),
/// `float` (as f64) or `str` (UTF-8 at `ptr` for `len` bytes). An
/// argument's string is the host's, borrowed for the call; a result's is
/// the library's, released with `ry_free`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RyValue {
    pub tag: u32,
    pub boolean: u8,
    pub int: i64,
    pub float: f64,
    pub ptr: *mut u8,
    pub len: usize,
}

/// A borrowed UTF-8 string: the program arguments `ry_entry` receives.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RyStr {
    pub ptr: *const u8,
    pub len: usize,
}

/// How a shim call ended (its return value).
/// `RyOutcome::value` holds the function's result; for `ry_entry`,
/// `RyOutcome::code` holds the program's exit status (0, 1 after an
/// uncaught exception the library reported as the binary does, or a
/// `SystemExit` code).
pub const RY_STATUS_OK: u32 = 0;
/// A Python exception escaped: `text` holds it, encoded as netstring-like
/// fields (`<byte length>:<bytes>`): the type name, the message, the
/// argument count (`-` when the construction recorded none) and each
/// argument's repr, then the ancestor count and each user-class ancestor.
pub const RY_STATUS_RAISED: u32 = 1;
/// A `SystemExit` escaped a function: `code` is its exit status, and
/// `text` the line CPython would print for a non-integer code (a null
/// `text_ptr` when there is none).
pub const RY_STATUS_EXIT: u32 = 2;
/// The library panicked (its panic hook has already reported it on
/// stderr, as the binary's would): `text` holds the panic message.
pub const RY_STATUS_PANIC: u32 = 3;
/// An argument could not cross the boundary: `text` says which.
pub const RY_STATUS_MARSHAL: u32 = 4;
/// `ry_entry` was called a second time in this process.
pub const RY_STATUS_ENTRY_ALREADY_RUN: u32 = 5;
/// The library's one copy of the module state already serves the other
/// role (function calls import the module; `ry_entry` runs it as
/// `__main__`): `text` says which.
pub const RY_STATUS_ROLE: u32 = 6;

/// A call's outcome, written by the library. Every buffer in it belongs
/// to the library and is released with `ry_free`.
#[repr(C)]
pub struct RyOutcome {
    pub value: RyValue,
    pub code: i64,
    pub text_ptr: *mut u8,
    pub text_len: usize,
}

/// `ryfn_<n>`: one function shim. Every shim has this signature, whatever
/// the Python function's, so the host never calls through a function
/// pointer of the wrong type; the manifest says what each one expects.
pub type RyFunctionShim =
    unsafe extern "C" fn(args: *const RyValue, nargs: usize, out: *mut RyOutcome) -> u32;
/// `ry_entry`: run the program (module initialization, then the
/// `__main__` body) with these arguments as `sys.argv`.
pub type RyEntryShim =
    unsafe extern "C" fn(argv: *const RyStr, argc: usize, out: *mut RyOutcome) -> u32;
/// `ry_free`: release a buffer the library handed out.
pub type RyFree = unsafe extern "C" fn(ptr: *mut u8, len: usize);
/// `ry_abi_version`: the library's `RY_ABI_VERSION`.
pub type RyAbiVersion = unsafe extern "C" fn() -> u32;
/// `ry_manifest`: the library's manifest text (static; never freed).
pub type RyManifestText = unsafe extern "C" fn(len: *mut usize) -> *const u8;
