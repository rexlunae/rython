//! urllib.request (ureq-backed) pins — the `http-ureq` feature only:
//! under the default build this file compiles to nothing. Run with
//! `cargo test -p stdpython --features http-ureq`.
#![cfg(feature = "http-ureq")]

use std::io::{Read, Write};
use std::net::TcpListener;

use stdpython::urllib;

/// One-shot local HTTP server: accepts a single connection and answers
/// with the given status line and body, so the pins need no network.
fn serve_once(status_line: &'static str, body: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut conn, _) = listener.accept().unwrap();
        let mut buf = [0u8; 2048];
        let _ = conn.read(&mut buf);
        let resp = format!(
            "HTTP/1.1 {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status_line,
            body.len(),
            body
        );
        conn.write_all(resp.as_bytes()).unwrap();
    });
    port
}

#[test]
fn urlopen_reads_status_headers_and_body() {
    // Verified against python3: urlopen(url) -> resp.status 200,
    // resp.read() the body bytes, resp.getcode() 200, and a second
    // read() returns b'' (the body is exhausted).
    let port = serve_once("200 OK", "hello from server");
    let mut resp = urllib::request::urlopen(format!("http://127.0.0.1:{}/", port)).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.getcode(), 200);
    assert_eq!(resp.read().unwrap(), b"hello from server");
    assert_eq!(resp.read().unwrap(), b"");
    assert_eq!(
        resp.getheader("content-type").as_deref(),
        Some("text/plain")
    );
    assert_eq!(resp.getheader("x-missing"), None);
}

#[test]
fn http_error_status_raises_httperror_with_pythons_message() {
    // Verified against python3: urlopen of a 404 raises
    // HTTPError('HTTP Error 404: Not Found'), caught by `except OSError:`
    // (HTTPError IS-A URLError IS-A OSError).
    let port = serve_once("404 Not Found", "gone");
    let e = urllib::request::urlopen(format!("http://127.0.0.1:{}/none", port)).unwrap_err();
    assert_eq!(e.exception_type, "HTTPError");
    assert_eq!(e.message, "HTTP Error 404: Not Found");
    assert!(e.matches("URLError"));
    assert!(e.matches("OSError"));
}

#[test]
fn unreachable_host_raises_urlerror() {
    // Verified against python3: URLError('<urlopen error [Errno 111]
    // Connection refused>') — rython keeps the '<urlopen error ...>'
    // shape with the backend's reason text (ledger divergence).
    let e = urllib::request::urlopen("http://127.0.0.1:1/none").unwrap_err();
    assert_eq!(e.exception_type, "URLError");
    assert!(e.message.starts_with("<urlopen error "), "{}", e.message);
    assert!(e.matches("OSError"));
}
