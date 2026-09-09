//! The `encodings` package surface the runtime carries: the CPython
//! alias table (`from encodings.aliases import aliases` —
//! charset_normalizer's utils.py/models.py iterate it to resolve
//! canonical codec names, round 111).

pub mod aliases;
