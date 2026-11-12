//! The C-compatible boundary between the host and a compiled library
//! (docs/jit.md, "The boundary"). The definitions live in one file that is
//! compiled here and emitted verbatim into every generated library.

include!("abi_types.rs");

/// The boundary's definitions as source text, for the converter's cdylib
/// mode to emit into each generated library: the same text this module
/// compiles, so the host and the library agree by construction.
pub const ABI_SOURCE: &str = include_str!("abi_types.rs");

impl RyValue {
    /// A value with every field zeroed: `None`, until a tag says otherwise.
    pub const fn none() -> RyValue {
        RyValue {
            tag: RY_TAG_NONE,
            boolean: 0,
            int: 0,
            float: 0.0,
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

impl RyOutcome {
    /// An outcome with nothing written yet.
    pub const fn empty() -> RyOutcome {
        RyOutcome {
            value: RyValue::none(),
            code: 0,
            text_ptr: core::ptr::null_mut(),
            text_len: 0,
        }
    }
}
