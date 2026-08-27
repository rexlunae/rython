//! Python ssl module — TLS over a pluggable backend, chosen by feature:
//!
//! - `ssl-rustls` (DEFAULT): client-side TLS over the rustls crate —
//!   a lightweight, pure-Rust implementation. What is modeled: the
//!   SSLContext surface the top converted packages (urllib3/requests)
//!   drive — construction, verify_mode/check_hostname, min/max TLS
//!   versions, CA loading (system roots via webpki-roots, PEM files/data
//!   via rustls-pemfile), ALPN, and wrap_socket() performing the
//!   handshake over a connected `socket.Socket` (the TLS session shares
//!   the same descriptor, exactly like Python wrapping the fd).
//!
//!   Divergences of the rustls backend (documented in docs/spec.md §12):
//!   - `verify_mode = CERT_NONE` accepts ANY certificate (a rustls
//!     verifier that skips validation) — as in Python; CERT_OPTIONAL is
//!     treated as CERT_REQUIRED (rustls has no half-verification).
//!   - The OP_* option bits are STORED and readable, but rustls's own
//!     policy decides the handshake (it never compresses, never
//!     renegotiates, and only speaks TLS 1.2/1.3 — a superset of what
//!     the usual OP_NO_* hardening asks for). Only minimum_version /
//!     maximum_version actually change the negotiated range.
//!   - `set_ciphers()` is a no-op (rustls's cipher policy is not
//!     string-configurable); OPENSSL_VERSION reports "rustls", so
//!     OpenSSL-version-sniffing code takes its generic path.
//!   - Server-side sockets, client certificates, and session resumption
//!     controls are not modeled (they error loudly where reachable).
//!
//! - `ssl-openssl` (opt-in): the FULL CPython ssl surface over the
//!   SYSTEM OpenSSL/LibreSSL (the `openssl` crate probes and links the
//!   system library) — real CERT_OPTIONAL half-verification, real
//!   `set_ciphers()`, client certificates (`load_cert_chain`),
//!   server-side contexts (`PROTOCOL_TLS_SERVER` + accepting
//!   wrap_socket), and OPENSSL_VERSION* reporting the linked library's
//!   real version. Wire protocol: standard TLS 1.2/1.3.
//!
//! Enable exactly one backend. The shared constants live here; the
//! implementation (SSLContext/SSLSocket) and the version constants live
//! in the backend module (ssl_rustls.rs / ssl_openssl.rs).

// The SSLContext/SSLSocket implementations live behind the chosen
// backend feature (ssl-rustls, default, or ssl-openssl); this module
// carries the shared constants and re-exports the active backend.
#[cfg(feature = "ssl-openssl")]
#[path = "ssl_openssl.rs"]
mod ssl_openssl;
#[cfg(all(feature = "ssl-rustls", not(feature = "ssl-openssl")))]
#[path = "ssl_rustls.rs"]
mod ssl_rustls;
#[cfg(all(feature = "ssl-rustls", feature = "ssl-openssl"))]
compile_error!("ssl-rustls and ssl-openssl are mutually exclusive; enable exactly one");

#[cfg(feature = "ssl-openssl")]
pub use ssl_openssl::*;
#[cfg(all(feature = "ssl-rustls", not(feature = "ssl-openssl")))]
pub use ssl_rustls::*;

// ---------------------------------------------------------------------------
// Module constants (CPython's numeric values, from python3 3.11).
// ---------------------------------------------------------------------------

pub const CERT_NONE: i64 = 0;
pub const CERT_OPTIONAL: i64 = 1;
pub const CERT_REQUIRED: i64 = 2;

pub const PROTOCOL_TLS: i64 = 2;
pub const PROTOCOL_SSLv23: i64 = 2;
pub const PROTOCOL_TLS_CLIENT: i64 = 16;
pub const PROTOCOL_TLS_SERVER: i64 = 17;
// The version-pinned protocols (deprecated in CPython but still
// exported; urllib3's pyopenssl maps over them). A context built with
// one clamps to that single version where rustls supports it (1.2);
// 1.0/1.1 are below rustls's floor and fail loudly at handshake-config
// time.
pub const PROTOCOL_TLSv1: i64 = 3;
pub const PROTOCOL_TLSv1_1: i64 = 4;
pub const PROTOCOL_TLSv1_2: i64 = 5;

pub const OP_NO_SSLv2: i64 = 0;
pub const OP_NO_SSLv3: i64 = 0x0200_0000;
pub const OP_NO_TLSv1: i64 = 0x0400_0000;
pub const OP_NO_TLSv1_1: i64 = 0x1000_0000;
pub const OP_NO_TLSv1_2: i64 = 0x0800_0000;
pub const OP_NO_TLSv1_3: i64 = 0x2000_0000;
pub const OP_NO_COMPRESSION: i64 = 0x0002_0000;
pub const OP_NO_TICKET: i64 = 0x4000;
pub const OP_NO_RENEGOTIATION: i64 = 0x4000_0000;

pub const VERIFY_X509_STRICT: i64 = 32;
pub const VERIFY_X509_TRUSTED_FIRST: i64 = 0x8000;
pub const VERIFY_X509_PARTIAL_CHAIN: i64 = 0x0008_0000;

// The SSL_ERROR_* errno family (values from python3's ssl module) —
// urllib3's ssltransport compares exception errnos against these.
pub const SSL_ERROR_SSL: i64 = 1;
pub const SSL_ERROR_WANT_READ: i64 = 2;
pub const SSL_ERROR_WANT_WRITE: i64 = 3;
pub const SSL_ERROR_WANT_X509_LOOKUP: i64 = 4;
pub const SSL_ERROR_SYSCALL: i64 = 5;
pub const SSL_ERROR_ZERO_RETURN: i64 = 6;
pub const SSL_ERROR_WANT_CONNECT: i64 = 7;
pub const SSL_ERROR_EOF: i64 = 8;
pub const SSL_ERROR_INVALID_ERROR_CODE: i64 = 10;

#[allow(non_upper_case_globals)]
pub const HAS_SNI: bool = true;
#[allow(non_upper_case_globals)]
pub const HAS_NEVER_CHECK_COMMON_NAME: bool = true;

// OPENSSL_VERSION / OPENSSL_VERSION_NUMBER / OPENSSL_VERSION_INFO are
// backend-specific (the rustls backend reports "rustls", the openssl
// backend the linked library's real version) and are LazyLock statics
// — the codegen derefs reads of them (see attribute.rs `needs_deref`
// and name.rs), so converted code sees plain &str / i64 / tuple values.

/// ssl.TLSVersion — the dotted chain renders as a path
/// (`ssl::TLSVersion::TLSv1_2`), so the enum is a nested module of int
/// constants (CPython's IntEnum values).
#[allow(non_snake_case, non_upper_case_globals)]
pub mod TLSVersion {
    pub const MINIMUM_SUPPORTED: i64 = -2;
    pub const SSLv3: i64 = 768;
    pub const TLSv1: i64 = 769;
    pub const TLSv1_1: i64 = 770;
    pub const TLSv1_2: i64 = 771;
    pub const TLSv1_3: i64 = 772;
    pub const MAXIMUM_SUPPORTED: i64 = -1;
}

// ---------------------------------------------------------------------------
// Certificate verification plumbing.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_cpython() {
        // Verified against python3 3.11's ssl module.
        assert_eq!(CERT_NONE, 0);
        assert_eq!(CERT_OPTIONAL, 1);
        assert_eq!(CERT_REQUIRED, 2);
        assert_eq!(PROTOCOL_TLS, 2);
        assert_eq!(PROTOCOL_TLS_CLIENT, 16);
        assert_eq!(OP_NO_COMPRESSION, 131072);
        assert_eq!(OP_NO_TICKET, 16384);
        assert_eq!(OP_NO_SSLv3, 33554432);
        assert_eq!(OP_NO_TLSv1, 67108864);
        assert_eq!(OP_NO_TLSv1_1, 268435456);
        assert_eq!(VERIFY_X509_PARTIAL_CHAIN, 524288);
        assert_eq!(VERIFY_X509_STRICT, 32);
        assert_eq!(TLSVersion::TLSv1_2, 771);
        assert_eq!(TLSVersion::TLSv1_3, 772);
    }
}
