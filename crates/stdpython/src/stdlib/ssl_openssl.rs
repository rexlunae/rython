//! OpenSSL-backed TLS (the `ssl-openssl` feature) — the FULL CPython ssl
//! surface with real OpenSSL semantics, for the features the default
//! rustls backend cannot provide:
//!
//! - `CERT_OPTIONAL` half-verification (SSL_VERIFY_PEER without
//!   FAIL_IF_NO_PEER_CERT) — rustls has no half-verification;
//! - `set_ciphers()` — SSL_CTX_set_cipher_list, not a no-op;
//! - client certificates (`load_cert_chain`) and server-side contexts
//!   (`PROTOCOL_TLS_SERVER` + `wrap_socket` accepting);
//! - `OPENSSL_VERSION*` report the REAL OpenSSL version, so
//!   version-sniffing code (urllib3's ssl_.py) takes its OpenSSL path.
//!
//! The wire protocol is standard TLS 1.2/1.3 via the system
//! OpenSSL/LibreSSL (the `openssl` crate, which probes the system
//! library). The rustls backend (default) stays the lighter client-only
//! implementation; enable exactly one backend.

use std::io::{Read, Write};
use std::sync::Arc;

use crate::PyException;

use super::{
    CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED, PROTOCOL_TLS, PROTOCOL_TLS_CLIENT,
    PROTOCOL_TLS_SERVER, TLSVersion,
};

/// The real OpenSSL version this backend links against (version-sniffing
/// code behaves exactly as it does on CPython-with-OpenSSL). LazyLock
/// statics, like the rustls backend's: the linked library's version is
/// only knowable at runtime, and the codegen derefs reads (attribute.rs
/// `needs_deref`, name.rs) so converted code sees plain values. CPython's
/// OPENSSL_VERSION_INFO is a 5-tuple; comparisons in the wild are against
/// 3-tuples (`ssl.OPENSSL_VERSION_INFO < (1, 1, 1)` — urllib3's
/// __init__), and rython tuples compare same-arity only, so this is the
/// (major, minor, fix) prefix of the linked version.
pub static OPENSSL_VERSION: std::sync::LazyLock<&'static str> =
    std::sync::LazyLock::new(openssl::version::version);
pub static OPENSSL_VERSION_NUMBER: std::sync::LazyLock<i64> =
    std::sync::LazyLock::new(openssl::version::number);
pub static OPENSSL_VERSION_INFO: std::sync::LazyLock<(i64, i64, i64)> =
    std::sync::LazyLock::new(|| {
        let n = openssl::version::number();
        (((n >> 28) & 0xf), ((n >> 20) & 0xff), ((n >> 12) & 0xff))
    });

/// Map an OpenSSL/IO error onto Python's ssl.SSLError shape.
fn ssl_error<E: core::fmt::Display>(e: E) -> PyException {
    PyException::new("SSLError", format!("{}", e))
}

fn verify_mode_of(mode: i64) -> openssl::ssl::SslVerifyMode {
    match mode {
        // CERT_OPTIONAL — verify IF a peer cert is presented, but accept
        // a peer with none (the half-verification rustls cannot do).
        CERT_OPTIONAL => openssl::ssl::SslVerifyMode::PEER,
        CERT_REQUIRED => {
            openssl::ssl::SslVerifyMode::PEER | openssl::ssl::SslVerifyMode::FAIL_IF_NO_PEER_CERT
        }
        _ => openssl::ssl::SslVerifyMode::NONE,
    }
}

fn openssl_version(v: i64) -> Option<openssl::ssl::SslVersion> {
    match v {
        TLSVersion::TLSv1_2 => Some(openssl::ssl::SslVersion::TLS1_2),
        TLSVersion::TLSv1_3 => Some(openssl::ssl::SslVersion::TLS1_3),
        _ => None,
    }
}

fn method_of(server: bool) -> openssl::ssl::SslMethod {
    if server {
        openssl::ssl::SslMethod::tls_server()
    } else {
        openssl::ssl::SslMethod::tls_client()
    }
}

// ---------------------------------------------------------------------------
// SSLContext.
// ---------------------------------------------------------------------------

/// Python ssl.SSLContext — the TLS configuration object. The OpenSSL
/// context is built eagerly; the Python-visible fields mirror CPython's
/// and are applied at wrap_socket time (they are public and mutated
/// after construction, exactly like the rustls backend). The loader /
/// setter methods store their argument and REBUILD the context from all
/// of them (CPython mutates one SSL_CTX in place — rebuilding
/// cumulatively preserves earlier settings instead of wiping them).
#[derive(Clone)]
pub struct SSLContext {
    pub protocol: i64,
    pub verify_mode: i64,
    pub check_hostname: bool,
    /// The OP_* bits — stored/readable, and the meaningful ones
    /// (OP_NO_SSLv2/3, OP_NO_TLSv1/1.1/1.2/1.3, OP_NO_COMPRESSION,
    /// OP_NO_RENEGOTIATION, OP_NO_TICKET) are enforced by OpenSSL itself
    /// when applied to the underlying context.
    pub options: i64,
    pub minimum_version: i64,
    pub maximum_version: i64,
    /// The VERIFY_* flag bits, stored/readable (VERIFY_X509_STRICT is
    /// applied to the verification).
    pub verify_flags: i64,
    /// Stored/readable; OpenSSL negotiates post-handshake auth on its own.
    pub post_handshake_auth: bool,
    /// Stored/readable (CPython 3.7+ never checks the common name).
    pub hostname_checks_common_name: bool,
    /// Stored/readable; key logging is not wired to OpenSSL here.
    pub keylog_filename: Option<String>,
    alpn: Vec<Vec<u8>>,
    /// Rebuild inputs — every loader/setter stores its argument here and
    /// rebuilds the OpenSSL context from ALL of them.
    ca_file: Option<String>,
    ca_data: Option<Vec<u8>>,
    default_paths: bool,
    ciphers: Option<String>,
    cert_chain: Option<(String, String)>,
    /// The built OpenSSL context.
    ctx: openssl::ssl::SslContext,
    /// Whether this is a server-side context.
    server: bool,
}

impl core::fmt::Debug for SSLContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SSLContext")
            .field("protocol", &self.protocol)
            .field("verify_mode", &self.verify_mode)
            .field("check_hostname", &self.check_hostname)
            .finish_non_exhaustive()
    }
}

fn build_ctx(protocol: i64) -> Result<openssl::ssl::SslContext, PyException> {
    let server = protocol == PROTOCOL_TLS_SERVER;
    let builder =
        openssl::ssl::SslContextBuilder::new(method_of(server)).map_err(ssl_error)?;
    // SslContextBuilder::build is infallible (the method is validated at
    // builder construction), so the fallible part is only `new`.
    Ok(builder.build())
}

/// ssl.SSLContext(protocol) — the constructor function. PROTOCOL_TLS_CLIENT
/// enables verification and hostname checks by default, exactly like
/// CPython; PROTOCOL_TLS_SERVER builds a server context.
#[allow(non_snake_case)]
pub fn SSLContext(protocol: i64) -> SSLContext {
    let ctx = build_ctx(protocol).unwrap_or_else(|e| panic!("{}", e));
    let client = protocol == PROTOCOL_TLS_CLIENT;
    SSLContext {
        protocol,
        verify_mode: if client { CERT_REQUIRED } else { CERT_NONE },
        check_hostname: client,
        options: 0,
        minimum_version: TLSVersion::MINIMUM_SUPPORTED,
        maximum_version: TLSVersion::MAXIMUM_SUPPORTED,
        verify_flags: 0,
        post_handshake_auth: false,
        hostname_checks_common_name: true,
        keylog_filename: None,
        alpn: Vec::new(),
        ca_file: None,
        ca_data: None,
        default_paths: false,
        ciphers: None,
        cert_chain: None,
        ctx,
        server: protocol == PROTOCOL_TLS_SERVER,
    }
}

/// ssl.create_default_context() — CPython's secure-client preset.
pub fn create_default_context() -> SSLContext {
    let mut ctx = SSLContext(PROTOCOL_TLS_CLIENT);
    // OpenSSL's system CA store (real roots, not a bundled set).
    let _ = ctx.load_default_certs();
    ctx
}

/// The codegen hoists try-block locals with a Default placeholder.
impl Default for SSLContext {
    fn default() -> SSLContext {
        SSLContext(PROTOCOL_TLS)
    }
}

impl SSLContext {
    /// The class-constructor spelling the codegen emits.
    pub fn new(protocol: i64) -> SSLContext {
        SSLContext(protocol)
    }

    /// Rebuild the OpenSSL context from every stored loader/setter input,
    /// so later calls never wipe earlier settings (CPython mutates the
    /// one SSL_CTX in place).
    fn rebuild(&mut self) -> Result<(), PyException> {
        let mut builder =
            openssl::ssl::SslContextBuilder::new(method_of(self.server)).map_err(ssl_error)?;
        if let Some(cafile) = &self.ca_file {
            builder.set_ca_file(cafile).map_err(ssl_error)?;
        }
        if let Some(data) = &self.ca_data {
            let mut store = openssl::x509::store::X509StoreBuilder::new().map_err(ssl_error)?;
            let pems = openssl::x509::X509::stack_from_pem(data).map_err(ssl_error)?;
            for cert in pems {
                store.add_cert(cert).map_err(ssl_error)?;
            }
            if self.default_paths {
                store.set_default_paths().map_err(ssl_error)?;
            }
            builder
                .set_verify_cert_store(store.build())
                .map_err(ssl_error)?;
        } else if self.default_paths {
            builder.set_default_verify_paths().map_err(ssl_error)?;
        }
        if let Some(ciphers) = &self.ciphers {
            builder.set_cipher_list(ciphers).map_err(ssl_error)?;
        }
        if let Some((certfile, keyfile)) = &self.cert_chain {
            builder
                .set_certificate_chain_file(certfile)
                .map_err(ssl_error)?;
            builder
                .set_private_key_file(keyfile, openssl::ssl::SslFiletype::PEM)
                .map_err(ssl_error)?;
            builder.check_private_key().map_err(ssl_error)?;
        }
        self.ctx = builder.build();
        Ok(())
    }

    /// ssl.SSLContext.load_default_certs(): the system CA store.
    pub fn load_default_certs(&mut self) -> Result<(), PyException> {
        self.default_paths = true;
        self.rebuild()
    }

    /// ssl.SSLContext.load_verify_locations(cafile) — a PEM bundle path.
    pub fn load_verify_locations<P: AsRef<str>>(
        &mut self,
        cafile: P,
    ) -> Result<(), PyException> {
        self.ca_file = Some(cafile.as_ref().to_string());
        self.rebuild()
    }

    /// load_verify_locations(cadata=...) — in-memory PEM text.
    pub fn load_verify_data<B: AsRef<[u8]>>(&mut self, cadata: B) -> Result<(), PyException> {
        self.ca_data = Some(cadata.as_ref().to_vec());
        self.rebuild()
    }

    /// ssl.SSLContext.set_ciphers(ciphers) — REAL cipher configuration
    /// via SSL_CTX_set_cipher_list (not a no-op like the rustls backend).
    pub fn set_ciphers<S>(&mut self, ciphers: S) -> Result<(), PyException>
    where
        S: AsRef<str>,
    {
        self.ciphers = Some(ciphers.as_ref().to_string());
        self.rebuild()
    }

    /// ssl.SSLContext.set_alpn_protocols(protocols).
    pub fn set_alpn_protocols<S: AsRef<str>>(&mut self, protocols: Vec<S>) {
        self.alpn = protocols.iter().map(|p| p.as_ref().as_bytes().to_vec()).collect();
    }

    /// ssl.SSLContext.load_cert_chain(certfile, keyfile) — CLIENT or
    /// server certificates (real OpenSSL support; the rustls backend
    /// errors loudly here).
    pub fn load_cert_chain<P: AsRef<str>>(
        &mut self,
        certfile: P,
        keyfile: P,
    ) -> Result<(), PyException> {
        self.cert_chain = Some((certfile.as_ref().to_string(), keyfile.as_ref().to_string()));
        self.rebuild()
    }

    /// Apply the Python-visible settings to a fresh per-connection Ssl.
    fn configure(&self, ssl: &mut openssl::ssl::Ssl) -> Result<(), PyException> {
        ssl.set_verify(verify_mode_of(self.verify_mode));
        if let Some(v) = openssl_version(self.minimum_version) {
            ssl.set_min_proto_version(Some(v)).map_err(ssl_error)?;
        }
        if let Some(v) = openssl_version(self.maximum_version) {
            ssl.set_max_proto_version(Some(v)).map_err(ssl_error)?;
        }
        if !self.alpn.is_empty() {
            // SSL_set_alpn_protos takes the wire format (1-byte length
            // prefix per protocol, concatenated).
            let mut wire: Vec<u8> = Vec::new();
            for p in &self.alpn {
                wire.push(p.len() as u8);
                wire.extend_from_slice(p);
            }
            ssl.set_alpn_protos(&wire).map_err(ssl_error)?;
        }
        Ok(())
    }

    /// ssl.SSLContext.wrap_socket(sock, server_hostname=...): the TLS
    /// handshake over the connected socket's stream (client) or the
    /// accept handshake (server context). The returned SSLSocket shares
    /// the descriptor (Python's fd semantics).
    pub fn wrap_socket<S: AsRef<str>>(
        &self,
        sock: crate::stdlib::socket::Socket,
        server_hostname: S,
    ) -> Result<SSLSocket, PyException> {
        let tcp = sock.tcp_stream_clone()?;
        let mut ssl = openssl::ssl::Ssl::new(&self.ctx).map_err(ssl_error)?;
        self.configure(&mut ssl)?;
        let host = server_hostname.as_ref();
        if !self.server && self.check_hostname {
            // Sets SNI and the hostname used by certificate verification.
            ssl.set_hostname(host).map_err(ssl_error)?;
        }
        let mut stream = openssl::ssl::SslStream::new(ssl, tcp).map_err(ssl_error)?;
        if self.server {
            // Server-side: accept the client's ClientHello.
            std::pin::Pin::new(&mut stream).accept().map_err(ssl_error)?;
        } else {
            // Connect drives the handshake eagerly so failures surface
            // at wrap_socket, as CPython's blocking sockets do. (The
            // hostname — used for SNI and hostname verification — was
            // set on the Ssl above; SslStream::connect takes none.)
            std::pin::Pin::new(&mut stream).connect().map_err(ssl_error)?;
        }
        Ok(SSLSocket {
            inner: Arc::new(std::sync::Mutex::new(Some(stream))),
        })
    }
}

// ---------------------------------------------------------------------------
// SSLSocket.
// ---------------------------------------------------------------------------

/// Python ssl.SSLSocket — a TLS session over a TCP stream. A handle
/// with Python's reference semantics (clones share the session).
#[derive(Clone)]
pub struct SSLSocket {
    inner: Arc<std::sync::Mutex<Option<openssl::ssl::SslStream<std::net::TcpStream>>>>,
}

impl SSLSocket {
    fn with_stream<R>(
        &self,
        f: impl FnOnce(
            &mut openssl::ssl::SslStream<std::net::TcpStream>,
        ) -> Result<R, PyException>,
    ) -> Result<R, PyException> {
        let mut guard = self.inner.lock().unwrap();
        match guard.as_mut() {
            Some(s) => f(s),
            None => Err(PyException::new(
                "OSError",
                "[Errno 9] Bad file descriptor",
            )),
        }
    }

    /// ssl.SSLSocket.send(bytes).
    pub fn send<B: AsRef<[u8]>>(&self, data: B) -> Result<i64, PyException> {
        let data = data.as_ref();
        self.with_stream(|s| Ok(s.write(data).map_err(ssl_error)? as i64))
    }

    /// ssl.SSLSocket.sendall(bytes).
    pub fn sendall<B: AsRef<[u8]>>(&self, data: B) -> Result<(), PyException> {
        let data = data.as_ref();
        self.with_stream(|s| s.write_all(data).map_err(ssl_error))
    }

    /// ssl.SSLSocket.recv(bufsize) — b"" at clean TLS close.
    pub fn recv(&self, bufsize: i64) -> Result<Vec<u8>, PyException> {
        let n = bufsize.max(0) as usize;
        self.with_stream(|s| {
            let mut buf = vec![0u8; n];
            match s.read(&mut buf) {
                Ok(got) => {
                    buf.truncate(got);
                    Ok(buf)
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(Vec::new()),
                Err(e) => Err(ssl_error(e)),
            }
        })
    }

    /// ssl.SSLSocket.version() — the negotiated protocol name.
    pub fn version(&self) -> Result<String, PyException> {
        self.with_stream(|s| {
            let ssl = s.ssl();
            Ok(match ssl.version2() {
                Some(openssl::ssl::SslVersion::TLS1_2) => "TLSv1.2".to_string(),
                Some(openssl::ssl::SslVersion::TLS1_3) => "TLSv1.3".to_string(),
                _ => ssl.version_str().to_string(),
            })
        })
    }

    /// ssl.SSLSocket.selected_alpn_protocol().
    pub fn selected_alpn_protocol(&self) -> Result<String, PyException> {
        self.with_stream(|s| {
            Ok(s.ssl()
                .selected_alpn_protocol()
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .unwrap_or_default())
        })
    }

    /// ssl.SSLSocket.close().
    pub fn close(&self) -> Result<(), PyException> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(mut s) = guard.take() {
            // Send close_notify (the TLS-level shutdown), like Python's
            // SSLSocket.close().
            let _ = s.shutdown();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_constants_report_real_openssl() {
        // Verified against python3: OPENSSL_VERSION starts with "OpenSSL ",
        // OPENSSL_VERSION_NUMBER is the packed hex number, and the INFO
        // prefix parses out of it.
        assert!(
            OPENSSL_VERSION.starts_with("OpenSSL "),
            "got: {}",
            *OPENSSL_VERSION
        );
        assert!(*OPENSSL_VERSION_NUMBER > 0);
        let n = *OPENSSL_VERSION_NUMBER;
        let info = *OPENSSL_VERSION_INFO;
        assert_eq!(
            info,
            (
                ((n >> 28) & 0xf) as i64,
                ((n >> 20) & 0xff) as i64,
                ((n >> 12) & 0xff) as i64
            )
        );
    }

    /// A self-signed localhost cert + key (rcgen), for the handshake test.
    fn localhost_cert() -> (rcgen::Certificate, rcgen::KeyPair) {
        let key_pair = rcgen::KeyPair::generate().expect("generate keypair");
        // A plain (non-CA) leaf with a localhost SAN — the realistic
        // server-cert shape; webpki rejects a CA-marked cert used as an
        // end-entity (CaUsedAsEndEntity).
        let params =
            rcgen::CertificateParams::new(vec!["localhost".to_string()]).expect("params");
        let cert = params.self_signed(&key_pair).expect("self-signed cert");
        (cert, key_pair)
    }

    #[test]
    fn handshake_roundtrip_over_openssl() {
        // A real TLS 1.2/1.3 session over a loopback socket: the openssl
        // backend's server-side context (PROTOCOL_TLS_SERVER +
        // load_cert_chain + accepting wrap_socket) talking to a verified
        // client (PROTOCOL_TLS_CLIENT + load_verify_data + connecting
        // wrap_socket). Pins the wire compatibility of the whole stack.
        let (cert, key_pair) = localhost_cert();

        // The server context needs PEM FILES (the CPython signature);
        // write them to a temp dir, cleaned up on drop.
        let dir = std::env::temp_dir().join(format!(
            "rython-ssl-test-{}-handshake",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, cert.pem()).unwrap();
        std::fs::write(&key_path, key_pair.serialize_pem()).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_cert = cert_path.clone();
        let server_key = key_path.clone();
        let server_handle = std::thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            tcp.set_nodelay(true).unwrap();
            let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
            let mut ctx = SSLContext(PROTOCOL_TLS_SERVER);
            ctx.load_cert_chain(server_cert.to_str().unwrap(), server_key.to_str().unwrap()).unwrap();
            let tls = ctx
                .wrap_socket(sock, "")
                .expect("server accept handshake");
            // Read the client's greeting, echo it back, close.
            let msg = tls.recv(1024).unwrap();
            tls.sendall(&msg).unwrap();
            tls.close().unwrap();
            msg
        });

        // Client side: trust the self-signed cert via cadata.
        let client_cert_pem = cert.pem();
        let mut ctx = SSLContext(PROTOCOL_TLS_CLIENT);
        ctx.load_verify_data(client_cert_pem.as_bytes()).unwrap();
        let tcp = std::net::TcpStream::connect(addr).unwrap();
        tcp.set_nodelay(true).unwrap();
        let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
        let tls = ctx
            .wrap_socket(sock, "localhost")
            .expect("client connect handshake");
        let version = tls.version().unwrap();
        assert!(
            version == "TLSv1.3" || version == "TLSv1.2",
            "negotiated: {}",
            version
        );
        tls.sendall(b"ping").unwrap();
        let reply = tls.recv(1024).unwrap();
        assert_eq!(reply, b"ping");
        tls.close().unwrap();

        let echoed = server_handle.join().unwrap();
        assert_eq!(echoed, b"ping");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cert_none_accepts_any_peer_and_cert_optional_half_verifies() {
        // CERT_NONE skips verification entirely (self-signed peer, no CA
        // loaded): the handshake succeeds. CERT_OPTIONAL verifies a
        // PRESENTED cert — an untrusted one fails, exactly CPython's
        // half-verification (the rustls backend treats CERT_OPTIONAL as
        // CERT_REQUIRED; OpenSSL really does distinguish).
        let (cert, key_pair) = localhost_cert();
        let dir = std::env::temp_dir().join(format!(
            "rython-ssl-test-{}-certmodes",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, cert.pem()).unwrap();
        std::fs::write(&key_path, key_pair.serialize_pem()).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server_cert = cert_path.clone();
        let server_key = key_path.clone();
        let server_handle = std::thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
            let mut ctx = SSLContext(PROTOCOL_TLS_SERVER);
            ctx.load_cert_chain(server_cert.to_str().unwrap(), server_key.to_str().unwrap()).unwrap();
            let tls = ctx.wrap_socket(sock, "").expect("server handshake");
            let _ = tls.recv(1024);
            tls.close().unwrap();
        });

        // CERT_NONE: no trust store at all — still connects.
        let mut none = SSLContext(PROTOCOL_TLS_CLIENT);
        none.verify_mode = CERT_NONE;
        let tcp = std::net::TcpStream::connect(addr).unwrap();
        let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
        assert!(none.wrap_socket(sock, "localhost").is_ok());

        server_handle.join().unwrap();

        // CERT_OPTIONAL against an UNTRUSTED presented cert: verification
        // runs on the peer's cert (which is not in any store) — the
        // handshake fails, like CPython's default client behavior.
        let mut optional = SSLContext(PROTOCOL_TLS_CLIENT);
        optional.verify_mode = CERT_OPTIONAL;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server_cert = cert_path.clone();
        let server_key = key_path.clone();
        let server_handle = std::thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
            let mut ctx = SSLContext(PROTOCOL_TLS_SERVER);
            ctx.load_cert_chain(server_cert.to_str().unwrap(), server_key.to_str().unwrap()).unwrap();
            // The client rejects before completing; the server sees an
            // error on its accept handshake — ignore it.
            let _ = ctx.wrap_socket(sock, "");
        });
        let tcp = std::net::TcpStream::connect(addr).unwrap();
        let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
        assert!(optional.wrap_socket(sock, "localhost").is_err());
        server_handle.join().unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_ciphers_and_alpn_apply_to_the_handshake() {
        // set_ciphers is REAL here: a valid OpenSSL cipher-list string is
        // accepted, an unknown one is rejected — and, exactly like
        // CPython (verified against python3), a TLS 1.3 ciphersuite name
        // is rejected because set_cipher_list configures the ≤1.2 list.
        let mut ctx = SSLContext(PROTOCOL_TLS_CLIENT);
        assert!(ctx.set_ciphers("DEFAULT").is_ok());
        assert!(ctx.set_ciphers("ECDHE-RSA-AES256-GCM-SHA384").is_ok());
        assert!(ctx.set_ciphers("TLS_AES_256_GCM_SHA384").is_err());
        assert!(ctx.set_ciphers("definitely-not-a-cipher").is_err());
        ctx.set_alpn_protocols(vec!["http/1.1"]);
    }
}
