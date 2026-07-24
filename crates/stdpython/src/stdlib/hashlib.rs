//! Python hashlib module implementation
//!
//! md5/sha1/sha256/sha512 with the update()/hexdigest() surface, backed
//! by the RustCrypto digest crates (identical output to CPython's
//! OpenSSL-backed digests). Constructors accept anything byte-like:
//! strings hash their UTF-8 bytes, which is exactly what the Python
//! idiom `hashlib.sha256(s.encode()).hexdigest()` produces.

use alloc::string::String;
use alloc::vec::Vec;
use md5::Digest;

/// A hash object: hashlib.sha256() et al. update() feeds more data;
/// hexdigest() reports without consuming, like Python.
#[derive(Debug, Clone)]
pub struct PyHashObject {
    inner: Hasher,
}

#[derive(Debug, Clone)]
enum Hasher {
    Md5(md5::Md5),
    Sha1(sha1::Sha1),
    Sha256(sha2::Sha256),
    Sha512(sha2::Sha512),
}

impl PyHashObject {
    /// h.update(data): feed more bytes; equivalent to hashing the
    /// concatenation.
    pub fn update<D: AsRef<[u8]>>(&mut self, data: D) {
        match &mut self.inner {
            Hasher::Md5(h) => h.update(data.as_ref()),
            Hasher::Sha1(h) => h.update(data.as_ref()),
            Hasher::Sha256(h) => h.update(data.as_ref()),
            Hasher::Sha512(h) => h.update(data.as_ref()),
        }
    }

    /// h.hexdigest(): lowercase hex, non-consuming (Python allows more
    /// update() calls afterward).
    pub fn hexdigest(&self) -> String {
        let bytes: Vec<u8> = match &self.inner {
            Hasher::Md5(h) => h.clone().finalize().to_vec(),
            Hasher::Sha1(h) => h.clone().finalize().to_vec(),
            Hasher::Sha256(h) => h.clone().finalize().to_vec(),
            Hasher::Sha512(h) => h.clone().finalize().to_vec(),
        };
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push_str(&alloc::format!("{:02x}", b));
        }
        out
    }
}

macro_rules! constructors {
    ($name:ident, $new_name:ident, $variant:ident, $ty:ty) => {
        /// hashlib constructor with initial data.
        pub fn $name<D: AsRef<[u8]> + ?Sized>(data: &D) -> PyHashObject {
            let mut h = $new_name();
            h.update(data.as_ref());
            h
        }

        /// hashlib constructor with no initial data (the update() idiom).
        pub fn $new_name() -> PyHashObject {
            PyHashObject {
                inner: Hasher::$variant(<$ty>::new()),
            }
        }
    };
}

constructors!(md5, md5_new, Md5, md5::Md5);
constructors!(sha1, sha1_new, Sha1, sha1::Sha1);
constructors!(sha256, sha256_new, Sha256, sha2::Sha256);
constructors!(sha512, sha512_new, Sha512, sha2::Sha512);
