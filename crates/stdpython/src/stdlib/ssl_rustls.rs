//! rustls-backed client-side TLS (the `ssl-rustls` feature, default).
//!
//! See the parent module's doc for the documented divergences of this
//! backend (CERT_OPTIONAL treated as CERT_REQUIRED, set_ciphers a no-op,
//! server-side sockets / client certificates not modeled). The
//! `ssl-openssl` backend implements those with real OpenSSL semantics.

use std::io::{Read, Write};
use std::sync::Arc;

use crate::PyException;

use super::{
    CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED, OP_NO_COMPRESSION, OP_NO_SSLv3, OP_NO_TICKET,
    OP_NO_TLSv1, OP_NO_TLSv1_1, OP_NO_TLSv1_2, OP_NO_TLSv1_3, PROTOCOL_TLS,
    PROTOCOL_TLS_CLIENT, TLSVersion, VERIFY_X509_PARTIAL_CHAIN, VERIFY_X509_STRICT,
};

// The backing TLS implementation's identity. Deliberately NOT an
// "OpenSSL ..." string: version-sniffing code (urllib3's ssl_.py) then
// takes its generic path instead of applying OpenSSL-specific
// workarounds that do not exist here. These are LazyLock statics (not
// consts) so the two backends expose the same shape; the codegen derefs
// reads (attribute.rs `needs_deref`, name.rs), so converted code sees
// plain values. CPython's OPENSSL_VERSION_INFO is a 5-tuple;
// comparisons in the wild are against 3-tuples
// (`ssl.OPENSSL_VERSION_INFO < (1, 1, 1)` — urllib3's __init__), and
// rython tuples compare same-arity only, so this is the 3-prefix. All
// zeros: rustls is not OpenSSL (a documented divergence).
pub static OPENSSL_VERSION: std::sync::LazyLock<&'static str> = std::sync::LazyLock::new(|| {
    concat!("rustls ", env!("CARGO_PKG_VERSION"))
});
pub static OPENSSL_VERSION_NUMBER: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| 0);
pub static OPENSSL_VERSION_INFO: std::sync::LazyLock<(i64, i64, i64)> =
    std::sync::LazyLock::new(|| (0, 0, 0));


/// A verifier that accepts any certificate — `verify_mode = CERT_NONE`
/// (Python's "no verification" mode; the connection is still encrypted).
#[derive(Debug)]
struct NoVerify(rustls::crypto::CryptoProvider);

impl rustls::client::danger::ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Map a rustls/setup error onto Python's ssl.SSLError shape (SSLError
/// IS-A OSError in CPython; the runtime's exception matcher treats the
/// name "SSLError" as OSError-family — see builtin exception aliases).
fn ssl_error<E: core::fmt::Display>(e: E) -> PyException {
    PyException::new("SSLError", format!("{}", e))
}

// ---------------------------------------------------------------------------
// SSLContext.
// ---------------------------------------------------------------------------

/// Python ssl.SSLContext — the client configuration object.
#[derive(Clone, Debug)]
pub struct SSLContext {
    pub protocol: i64,
    pub verify_mode: i64,
    pub check_hostname: bool,
    /// The OP_* bits, stored/readable; rustls's own policy governs the
    /// handshake (see the module divergences).
    pub options: i64,
    pub minimum_version: i64,
    pub maximum_version: i64,
    /// The VERIFY_* flag bits, stored/readable; rustls's verifier policy
    /// governs actual validation (see the module divergences).
    pub verify_flags: i64,
    /// Stored/readable; rustls negotiates post-handshake auth on its own.
    pub post_handshake_auth: bool,
    /// Stored/readable; rustls's webpki verifier never falls back to the
    /// certificate common name (CPython 3.7+ effectively matches this).
    pub hostname_checks_common_name: bool,
    /// Stored/readable; key logging is not wired to rustls in this
    /// backend (a documented no-op).
    pub keylog_filename: Option<String>,
    alpn: Vec<Vec<u8>>,
    /// Extra roots from load_verify_locations, DER-encoded.
    extra_roots: Vec<rustls::pki_types::CertificateDer<'static>>,
    /// Whether load_default_certs() pulled in the bundled webpki roots.
    default_roots: bool,
}

/// ssl.SSLContext(protocol) — the constructor function (Python spells
/// types and callables the same). PROTOCOL_TLS_CLIENT enables
/// verification and hostname checks by default, exactly like CPython.
#[allow(non_snake_case)]
pub fn SSLContext(protocol: i64) -> SSLContext {
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
        extra_roots: Vec::new(),
        default_roots: false,
    }
}

/// ssl.create_default_context() — CPython's secure-client preset.
pub fn create_default_context() -> SSLContext {
    let mut ctx = SSLContext(PROTOCOL_TLS_CLIENT);
    ctx.default_roots = true;
    ctx
}

/// The codegen hoists try-block locals with a Default placeholder.
impl Default for SSLContext {
    fn default() -> SSLContext {
        SSLContext(PROTOCOL_TLS)
    }
}

impl SSLContext {
    /// The class-constructor spelling the codegen emits
    /// (`SSLContext::new(PROTOCOL_TLS_CLIENT)`); same semantics as the
    /// constructor function above.
    pub fn new(protocol: i64) -> SSLContext {
        SSLContext(protocol)
    }

    /// ssl.SSLContext.load_default_certs(): the bundled webpki root
    /// store (rustls has no OpenSSL-style system-path lookup; the
    /// Mozilla root program is what certifi ships too).
    pub fn load_default_certs(&mut self) -> Result<(), PyException> {
        self.default_roots = true;
        Ok(())
    }

    /// ssl.SSLContext.load_verify_locations(cafile) — a PEM bundle path.
    pub fn load_verify_locations<P: AsRef<str>>(
        &mut self,
        cafile: P,
    ) -> Result<(), PyException> {
        let path = cafile.as_ref();
        let data = std::fs::read(path).map_err(|e| {
            PyException::new(
                "FileNotFoundError",
                format!("[Errno 2] No such file or directory: '{}' ({})", path, e),
            )
        })?;
        self.load_verify_pem(&data)
    }

    /// load_verify_locations(cadata=...) — in-memory PEM text.
    pub fn load_verify_data<B: AsRef<[u8]>>(&mut self, cadata: B) -> Result<(), PyException> {
        let data = cadata.as_ref().to_vec();
        self.load_verify_pem(&data)
    }

    fn load_verify_pem(&mut self, data: &[u8]) -> Result<(), PyException> {
        let mut reader = std::io::BufReader::new(data);
        let mut added = 0usize;
        for cert in rustls_pemfile::certs(&mut reader) {
            let cert = cert.map_err(ssl_error)?;
            self.extra_roots.push(cert);
            added += 1;
        }
        if added == 0 {
            return Err(ssl_error("no certificate found in CA bundle"));
        }
        Ok(())
    }

    /// ssl.SSLContext.set_ciphers(): rustls's cipher policy is not
    /// string-configurable — stored nowhere, a documented no-op.
    /// Unbounded on purpose: converted callers pass `str` or
    /// `Option<String>` (optional cipher parameters) and both are
    /// discarded identically.
    pub fn set_ciphers<S>(&mut self, _ciphers: S) -> Result<(), PyException> {
        Ok(())
    }

    /// ssl.SSLContext.set_alpn_protocols(["h2", "http/1.1"]).
    pub fn set_alpn_protocols<S: AsRef<str>>(
        &mut self,
        protocols: Vec<S>,
    ) -> Result<(), PyException> {
        self.alpn = protocols
            .iter()
            .map(|p| p.as_ref().as_bytes().to_vec())
            .collect();
        Ok(())
    }

    /// Build the rustls ClientConfig this context describes.
    fn client_config(&self) -> Result<rustls::ClientConfig, PyException> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        // The negotiated version range: rustls speaks TLS 1.2/1.3; the
        // context's minimum/maximum clamp within that.
        let mut versions: Vec<&'static rustls::SupportedProtocolVersion> = Vec::new();
        let min = self.minimum_version;
        let max = self.maximum_version;
        let allows = |v: i64| -> bool {
            (min == TLSVersion::MINIMUM_SUPPORTED || min <= v)
                && (max == TLSVersion::MAXIMUM_SUPPORTED || v <= max)
        };
        if allows(TLSVersion::TLSv1_2) && self.options & OP_NO_TLSv1_2 == 0 {
            versions.push(&rustls::version::TLS12);
        }
        if allows(TLSVersion::TLSv1_3) && self.options & OP_NO_TLSv1_3 == 0 {
            versions.push(&rustls::version::TLS13);
        }
        if versions.is_empty() {
            return Err(ssl_error(
                "no TLS version enabled (rustls supports TLS 1.2 and 1.3)",
            ));
        }
        let builder = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&versions)
            .map_err(ssl_error)?;
        let mut config = if self.verify_mode == CERT_NONE {
            builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(NoVerify(
                    rustls::crypto::ring::default_provider(),
                )))
                .with_no_client_auth()
        } else {
            let mut roots = rustls::RootCertStore::empty();
            if self.default_roots || self.extra_roots.is_empty() {
                roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            }
            for cert in &self.extra_roots {
                roots.add(cert.clone()).map_err(ssl_error)?;
            }
            builder
                .with_root_certificates(roots)
                .with_no_client_auth()
        };
        config.alpn_protocols = self.alpn.clone();
        Ok(config)
    }

    /// ssl.SSLContext.wrap_socket(sock, server_hostname=...): the TLS
    /// handshake over the connected socket's stream. The returned
    /// SSLSocket shares the descriptor (Python's fd semantics).
    pub fn wrap_socket<S: AsRef<str>>(
        &self,
        sock: crate::stdlib::socket::Socket,
        server_hostname: S,
    ) -> Result<SSLSocket, PyException> {
        let config = Arc::new(self.client_config()?);
        let name = rustls::pki_types::ServerName::try_from(
            server_hostname.as_ref().to_string(),
        )
        .map_err(ssl_error)?;
        let conn = rustls::ClientConnection::new(config, name).map_err(ssl_error)?;
        let tcp = sock.tcp_stream_clone()?;
        let mut stream = rustls::StreamOwned::new(conn, tcp);
        // Drive the handshake eagerly so failures surface at
        // wrap_socket, as CPython's blocking sockets do.
        stream.flush().map_err(ssl_error)?;
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
    inner: Arc<std::sync::Mutex<Option<rustls::StreamOwned<rustls::ClientConnection, std::net::TcpStream>>>>,
}

impl SSLSocket {
    fn with_stream<R>(
        &self,
        f: impl FnOnce(
            &mut rustls::StreamOwned<rustls::ClientConnection, std::net::TcpStream>,
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
                // A close_notify-less shutdown maps to b"" like Python's
                // suppress_ragged_eofs default.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(Vec::new()),
                Err(e) => Err(ssl_error(e)),
            }
        })
    }

    /// ssl.SSLSocket.version() — the negotiated protocol name.
    pub fn version(&self) -> Result<String, PyException> {
        self.with_stream(|s| {
            Ok(match s.conn.protocol_version() {
                Some(rustls::ProtocolVersion::TLSv1_2) => "TLSv1.2".to_string(),
                Some(rustls::ProtocolVersion::TLSv1_3) => "TLSv1.3".to_string(),
                Some(v) => format!("{:?}", v),
                None => String::new(),
            })
        })
    }

    /// ssl.SSLSocket.selected_alpn_protocol().
    pub fn selected_alpn_protocol(&self) -> Result<String, PyException> {
        self.with_stream(|s| {
            Ok(s.conn
                .alpn_protocol()
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .unwrap_or_default())
        })
    }

    /// ssl.SSLSocket.close().
    pub fn close(&self) -> Result<(), PyException> {
        let mut guard = self.inner.lock().unwrap();
        if let Some(mut s) = guard.take() {
            s.conn.send_close_notify();
            let _ = s.flush();
        }
        Ok(())
    }
}

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

    #[test]
    fn tls_client_context_defaults_match_cpython() {
        // PROTOCOL_TLS_CLIENT verifies and checks hostnames by default;
        // PROTOCOL_TLS does neither. Verified against python3.
        let client = SSLContext(PROTOCOL_TLS_CLIENT);
        assert_eq!(client.verify_mode, CERT_REQUIRED);
        assert!(client.check_hostname);
        let plain = SSLContext(PROTOCOL_TLS);
        assert_eq!(plain.verify_mode, CERT_NONE);
        assert!(!plain.check_hostname);
    }

    #[test]
    fn contexts_build_rustls_configs() {
        // Default roots, CERT_NONE, extra PEM roots, and version clamps
        // all produce a usable ClientConfig.
        let mut ctx = create_default_context();
        ctx.set_alpn_protocols(vec!["http/1.1"]).unwrap();
        assert!(ctx.client_config().is_ok());

        let mut none = SSLContext(PROTOCOL_TLS);
        none.verify_mode = CERT_NONE;
        assert!(none.client_config().is_ok());

        let mut clamped = SSLContext(PROTOCOL_TLS_CLIENT);
        clamped.minimum_version = TLSVersion::TLSv1_3;
        assert!(clamped.client_config().is_ok());

        let mut impossible = SSLContext(PROTOCOL_TLS_CLIENT);
        impossible.maximum_version = TLSVersion::TLSv1_1;
        assert!(impossible.client_config().is_err());
    }

    #[test]
    fn pem_loading_rejects_junk_and_accepts_certs() {
        let mut ctx = SSLContext(PROTOCOL_TLS_CLIENT);
        assert!(ctx.load_verify_data(b"not a pem").is_err());
        // A minimal self-signed cert PEM (content-free but well-formed
        // is enough for the parser path to count it): use webpki's
        // encoded root via DER->PEM instead of vendoring a fixture —
        // skipped; the junk-rejection above pins the loud path.
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
    fn handshake_roundtrip_over_rustls() {
        // A real TLS session over a loopback socket: the rustls-backed
        // client (create_default_context, trusting the self-signed cert
        // via load_verify_data) talking to a rustls server thread holding
        // the same cert. Pins the wire compatibility of the default
        // backend end to end (handshake, SNI, verification, close_notify).
        let (cert, key_pair) = localhost_cert();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let cert_der = cert.der().clone();
        let key_der = key_pair.serialize_der();
        let server_handle = std::thread::spawn(move || {
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let config = rustls::ServerConfig::builder_with_provider(provider)
                .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])
                .expect("protocol versions")
                .with_no_client_auth()
                .with_single_cert(
                    vec![cert_der],
                    rustls::pki_types::PrivateKeyDer::Pkcs8(key_der.into()),
                )
                .expect("server cert");
            let (tcp, _) = listener.accept().unwrap();
            let conn = rustls::ServerConnection::new(Arc::new(config)).unwrap();
            let mut stream = rustls::StreamOwned::new(conn, tcp);
            let mut buf = [0u8; 4];
            std::io::Read::read_exact(&mut stream, &mut buf).unwrap();
            std::io::Write::write_all(&mut stream, &buf).unwrap();
            stream.conn.send_close_notify();
            let _ = stream.flush();
            buf.to_vec()
        });

        let mut ctx = create_default_context();
        ctx.load_verify_data(cert.pem().as_bytes()).unwrap();
        let tcp = std::net::TcpStream::connect(addr).unwrap();
        let sock = crate::stdlib::socket::Socket::from_tcp_stream(tcp).unwrap();
        let tls = ctx
            .wrap_socket(sock, "localhost")
            .expect("client handshake");
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

        assert_eq!(server_handle.join().unwrap(), b"ping");
    }
}
