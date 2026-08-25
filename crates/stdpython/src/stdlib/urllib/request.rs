//! Python urllib.request — an HTTP(S) client backed by the ureq crate.
//!
//! This is the first consumer of the feature-gate convention for
//! platform-heavy surfaces: the module compiles only under stdpython's
//! `http-ureq` cargo feature (ureq with rustls, so https:// works out of
//! the box). rypip enables the feature on the generated crate's stdpython
//! dependency whenever the converted package imports urllib.request.
//!
//! Divergences (documented in docs/spec.md §12):
//! - An HTTP error status raises HTTPError with CPython's message
//!   ("HTTP Error 404: Not Found") but carries no response body/headers
//!   (rython exceptions are string-tagged values).
//! - Transport failures raise URLError; the reason text inside CPython's
//!   "<urlopen error ...>" shape comes from the backend, so its wording
//!   can differ from CPython's for the same failure.
//! - Redirects are followed (as CPython's default opener does).

use crate::PyException;

/// The response object urlopen returns (CPython's http.client.HTTPResponse
/// surface: `.status`, `read()`, `getcode()`, `geturl()`, `getheader()`).
#[derive(Debug)]
pub struct HTTPResponse {
    /// Python `resp.status` (an attribute in CPython, a pub field here).
    pub status: i64,
    url: String,
    body: Vec<u8>,
    pos: usize,
    headers: Vec<(String, String)>,
    closed: bool,
}

impl HTTPResponse {
    /// Python `resp.read()` — the remaining body bytes (empty once
    /// exhausted, as CPython).
    pub fn read(&mut self) -> Result<Vec<u8>, PyException> {
        if self.closed {
            return Err(crate::value_error("I/O operation on closed file."));
        }
        let rest = self.body[self.pos..].to_vec();
        self.pos = self.body.len();
        Ok(rest)
    }

    /// Python `resp.getcode()`.
    pub fn getcode(&self) -> i64 {
        self.status
    }

    /// Python `resp.geturl()` — the URL of the retrieved resource.
    pub fn geturl(&self) -> String {
        self.url.clone()
    }

    /// Python `resp.getheader(name)` — case-insensitive; None when absent.
    pub fn getheader<S: AsRef<str>>(&self, name: S) -> Option<String> {
        let want = name.as_ref().to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == want)
            .map(|(_, v)| v.clone())
    }

    /// Python `resp.close()`.
    pub fn close(&mut self) -> Result<(), PyException> {
        self.closed = true;
        Ok(())
    }
}

/// Python `urllib.request.urlopen(url)`.
pub fn urlopen<S: AsRef<str>>(url: S) -> Result<HTTPResponse, PyException> {
    let url = url.as_ref();
    match ureq::get(url).call() {
        Ok(mut resp) => {
            let status = resp.status().as_u16() as i64;
            let headers = resp
                .headers()
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str().to_string(),
                        String::from_utf8_lossy(v.as_bytes()).into_owned(),
                    )
                })
                .collect();
            let body = resp
                .body_mut()
                .read_to_vec()
                .map_err(|e| PyException::new("URLError", format!("<urlopen error {}>", e)))?;
            Ok(HTTPResponse {
                status,
                url: url.to_string(),
                body,
                pos: 0,
                headers,
                closed: false,
            })
        }
        Err(ureq::Error::StatusCode(code)) => {
            let reason = ureq::http::StatusCode::from_u16(code)
                .ok()
                .and_then(|s| s.canonical_reason())
                .unwrap_or("");
            // Verified against python3: str(HTTPError) is 'HTTP Error 404: Not Found'.
            Err(PyException::new(
                "HTTPError",
                format!("HTTP Error {}: {}", code, reason),
            ))
        }
        // Verified against python3: str(URLError) is '<urlopen error REASON>';
        // the reason wording here is the backend's (ledger divergence).
        Err(e) => Err(PyException::new(
            "URLError",
            format!("<urlopen error {}>", e),
        )),
    }
}
