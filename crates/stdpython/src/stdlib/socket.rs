//! Python socket module — TCP/UDP sockets on `std::net`.
//!
//! Socket objects are HANDLES with Python's reference semantics: cloning
//! shares the underlying descriptor (a socket passed to a worker thread is
//! the same socket; close() through one handle closes them all).
//!
//! Divergences (documented in docs/spec.md §12):
//! - A TCP `bind()` binds AND starts listening immediately (`std::net`
//!   has no half-bound TCP socket); `listen()` then only validates state.
//!   Binding a CLIENT socket before `connect()` is a loud OSError.
//! - `setsockopt` and the address-family zoo beyond AF_INET/AF_INET6 are
//!   not modeled; `settimeout` covers the float form (None/blocking is
//!   the default state, never re-enterable).
//! - Error messages mirror CPython's "[Errno N] text" shape from the OS
//!   errno; the catchable type walks the real hierarchy
//!   (ConnectionRefusedError IS-A ConnectionError IS-A OSError).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::PyException;

/// Address families / socket kinds (Linux numeric values, as CPython
/// exposes on Linux).
pub const AF_INET: i64 = 2;
pub const AF_INET6: i64 = 10;
pub const SOCK_STREAM: i64 = 1;
pub const SOCK_DGRAM: i64 = 2;

/// socket.shutdown() how-values (POSIX numeric values, as CPython).
pub const SHUT_RD: i64 = 0;
pub const SHUT_WR: i64 = 1;
pub const SHUT_RDWR: i64 = 2;

/// socket.has_ipv6 — the runtime's socket layer supports IPv6 (Rust std
/// does on every tier this module builds for).
#[allow(non_upper_case_globals)]
pub const has_ipv6: bool = true;

/// Map an I/O failure onto the exception CPython raises, with CPython's
/// "[Errno N] text" message shape (std's Display appends " (os error N)" —
/// stripped here).
fn net_error(e: &std::io::Error) -> PyException {
    use std::io::ErrorKind;
    // Verified against python3: socket timeouts raise TimeoutError('timed out').
    if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) {
        return PyException::new("TimeoutError", "timed out");
    }
    let kind = match e.kind() {
        ErrorKind::ConnectionRefused => "ConnectionRefusedError",
        ErrorKind::ConnectionReset => "ConnectionResetError",
        ErrorKind::ConnectionAborted => "ConnectionAbortedError",
        ErrorKind::BrokenPipe => "BrokenPipeError",
        ErrorKind::PermissionDenied => "PermissionError",
        ErrorKind::Interrupted => "InterruptedError",
        _ => "OSError",
    };
    match e.raw_os_error() {
        Some(errno) => {
            let text = e.to_string();
            let text = match text.find(" (os error ") {
                Some(cut) => text[..cut].to_string(),
                None => text,
            };
            // Verified against python3: e.g. '[Errno 111] Connection refused'.
            PyException::new(kind, format!("[Errno {}] {}", errno, text))
        }
        None => PyException::new(kind, e.to_string()),
    }
}

/// CPython raises OSError('[Errno 9] Bad file descriptor') on a closed
/// socket (and on operations in the wrong state).
fn bad_fd() -> PyException {
    PyException::new("OSError", "[Errno 9] Bad file descriptor")
}

fn addr_tuple(addr: SocketAddr) -> (String, i64) {
    (addr.ip().to_string(), addr.port() as i64)
}

#[derive(Debug)]
enum SockState {
    /// socket() was called; nothing bound or connected yet.
    Fresh { kind: i64 },
    /// A bound-and-listening TCP socket (bind() creates it — see the
    /// module divergence note).
    Listener(TcpListener),
    /// A connected TCP stream (from connect() or accept()).
    Stream(TcpStream),
    /// A bound UDP socket.
    Udp(UdpSocket),
    Closed,
}

#[derive(Debug)]
struct SocketInner {
    state: Mutex<SockState>,
    /// settimeout() value, applied to streams as read/write timeouts and
    /// to connect() as a connect timeout.
    timeout: Mutex<Option<Duration>>,
}

/// A connected handle cloned out of the state lock for blocking I/O
/// (see Socket::io_handle).
#[derive(Debug)]
enum IoHandle {
    Tcp(TcpStream),
    Udp(UdpSocket),
}

/// A Python socket object (socket.socket).
#[derive(Clone, Debug)]
pub struct Socket {
    inner: Arc<SocketInner>,
}

/// The codegen hoists try-block locals with a Default placeholder; an
/// unconnected AF_INET/SOCK_STREAM socket is the honest empty value.
impl Default for Socket {
    fn default() -> Socket {
        Socket::from_state(SockState::Fresh { kind: SOCK_STREAM })
    }
}

impl Socket {
    fn from_state(state: SockState) -> Socket {
        Socket {
            inner: Arc::new(SocketInner {
                state: Mutex::new(state),
                timeout: Mutex::new(None),
            }),
        }
    }

    /// Python `socket.bind((host, port))`. TCP: binds and listens (see the
    /// module divergence note); UDP: binds the datagram socket.
    pub fn bind<S: AsRef<str>>(&self, addr: (S, i64)) -> Result<(), PyException> {
        let mut state = self.inner.state.lock().unwrap();
        let kind = match &*state {
            SockState::Fresh { kind } => *kind,
            SockState::Closed => return Err(bad_fd()),
            _ => {
                return Err(PyException::new(
                    "OSError",
                    "[Errno 22] Invalid argument",
                ))
            }
        };
        let target = (addr.0.as_ref(), addr.1 as u16);
        *state = if kind == SOCK_DGRAM {
            SockState::Udp(UdpSocket::bind(target).map_err(|e| net_error(&e))?)
        } else {
            SockState::Listener(TcpListener::bind(target).map_err(|e| net_error(&e))?)
        };
        Ok(())
    }

    /// Python `socket.listen(backlog)`. The listener already exists (bind
    /// created it); this validates the socket is a bound TCP socket.
    pub fn listen(&self, _backlog: i64) -> Result<(), PyException> {
        match &*self.inner.state.lock().unwrap() {
            SockState::Listener(_) => Ok(()),
            SockState::Closed => Err(bad_fd()),
            // CPython: listening on an unbound socket auto-binds an
            // ephemeral port; rython requires the explicit bind first.
            _ => Err(PyException::new(
                "OSError",
                "listen() requires a bound TCP socket (call bind() first)",
            )),
        }
    }

    /// Python `socket.accept()` -> (conn, (host, port)).
    pub fn accept(&self) -> Result<(Socket, (String, i64)), PyException> {
        // Clone the listener handle out so waiting doesn't hold the state
        // lock (close() from another thread must stay possible).
        let listener = match &*self.inner.state.lock().unwrap() {
            SockState::Listener(l) => l.try_clone().map_err(|e| net_error(&e))?,
            SockState::Closed => return Err(bad_fd()),
            _ => return Err(bad_fd()),
        };
        let (stream, peer) = listener.accept().map_err(|e| net_error(&e))?;
        Ok((Socket::from_state(SockState::Stream(stream)), addr_tuple(peer)))
    }

    /// Python `socket.connect((host, port))`.
    pub fn connect<S: AsRef<str>>(&self, addr: (S, i64)) -> Result<(), PyException> {
        let timeout = *self.inner.timeout.lock().unwrap();
        let mut state = self.inner.state.lock().unwrap();
        let kind = match &*state {
            SockState::Fresh { kind } => *kind,
            SockState::Udp(u) => {
                // UDP connect sets the default peer.
                u.connect((addr.0.as_ref(), addr.1 as u16))
                    .map_err(|e| net_error(&e))?;
                return Ok(());
            }
            SockState::Closed => return Err(bad_fd()),
            _ => {
                return Err(PyException::new(
                    "OSError",
                    "[Errno 106] Transport endpoint is already connected",
                ))
            }
        };
        if kind == SOCK_DGRAM {
            let udp = UdpSocket::bind(("0.0.0.0", 0)).map_err(|e| net_error(&e))?;
            udp.connect((addr.0.as_ref(), addr.1 as u16))
                .map_err(|e| net_error(&e))?;
            *state = SockState::Udp(udp);
            return Ok(());
        }
        let stream = match timeout {
            Some(t) => {
                // connect_timeout needs a resolved address; take the first,
                // as CPython's connect does.
                let resolved = (addr.0.as_ref(), addr.1 as u16)
                    .to_socket_addrs()
                    .map_err(|e| net_error(&e))?
                    .next()
                    .ok_or_else(|| {
                        PyException::new("gaierror", "[Errno -2] Name or service not known")
                    })?;
                TcpStream::connect_timeout(&resolved, t).map_err(|e| net_error(&e))?
            }
            None => TcpStream::connect((addr.0.as_ref(), addr.1 as u16))
                .map_err(|e| net_error(&e))?,
        };
        stream.set_read_timeout(timeout).map_err(|e| net_error(&e))?;
        stream.set_write_timeout(timeout).map_err(|e| net_error(&e))?;
        *state = SockState::Stream(stream);
        Ok(())
    }

    /// Clone the connected I/O handle OUT of the state lock, so blocking
    /// reads and writes run without holding it: CPython sockets are
    /// full-duplex — a reader thread blocked in recv() must not freeze a
    /// writer thread's send()/close() on a shared clone of the same
    /// socket. accept() does the same with the listener. The clone is a
    /// dup'd descriptor of the SAME underlying socket, so data and
    /// shutdown state stay shared.
    fn io_handle(&self) -> Result<IoHandle, PyException> {
        match &*self.inner.state.lock().unwrap() {
            SockState::Stream(s) => Ok(IoHandle::Tcp(s.try_clone().map_err(|e| net_error(&e))?)),
            SockState::Udp(u) => Ok(IoHandle::Udp(u.try_clone().map_err(|e| net_error(&e))?)),
            SockState::Closed => Err(bad_fd()),
            _ => Err(PyException::new(
                "OSError",
                "[Errno 107] Transport endpoint is not connected",
            )),
        }
    }

    /// Python `socket.send(bytes)` -> count sent.
    pub fn send<B: AsRef<[u8]>>(&self, data: B) -> Result<i64, PyException> {
        let data = data.as_ref();
        let n = match self.io_handle()? {
            IoHandle::Tcp(mut s) => s.write(data).map_err(|e| net_error(&e))?,
            IoHandle::Udp(u) => u.send(data).map_err(|e| net_error(&e))?,
        };
        Ok(n as i64)
    }

    /// Python `socket.sendall(bytes)`.
    pub fn sendall<B: AsRef<[u8]>>(&self, data: B) -> Result<(), PyException> {
        let data = data.as_ref();
        match self.io_handle()? {
            IoHandle::Tcp(mut s) => s.write_all(data).map_err(|e| net_error(&e)),
            IoHandle::Udp(u) => {
                u.send(data).map_err(|e| net_error(&e))?;
                Ok(())
            }
        }
    }

    /// Python `socket.recv(bufsize)` -> bytes (empty at EOF, as CPython).
    pub fn recv(&self, bufsize: i64) -> Result<Vec<u8>, PyException> {
        if bufsize < 0 {
            return Err(crate::value_error("negative buffersize in recv"));
        }
        let handle = self.io_handle()?;
        let mut buf = vec![0u8; bufsize as usize];
        let n = match handle {
            IoHandle::Tcp(mut s) => s.read(&mut buf).map_err(|e| net_error(&e))?,
            IoHandle::Udp(u) => u.recv(&mut buf).map_err(|e| net_error(&e))?,
        };
        buf.truncate(n);
        Ok(buf)
    }

    /// Python `socket.sendto(bytes, (host, port))` (UDP) -> count sent.
    pub fn sendto<S: AsRef<str>, B: AsRef<[u8]>>(&self, data: B, addr: (S, i64)) -> Result<i64, PyException> {
        let data = data.as_ref();
        // Auto-bind (CPython binds an unbound UDP socket on first sendto)
        // and clone the handle out; the blocking send runs unlocked.
        let udp = {
            let mut state = self.inner.state.lock().unwrap();
            if let SockState::Fresh { kind } = &*state {
                if *kind == SOCK_DGRAM {
                    *state =
                        SockState::Udp(UdpSocket::bind(("0.0.0.0", 0)).map_err(|e| net_error(&e))?);
                }
            }
            match &*state {
                SockState::Udp(u) => u.try_clone().map_err(|e| net_error(&e))?,
                SockState::Closed => return Err(bad_fd()),
                _ => {
                    return Err(PyException::new(
                        "OSError",
                        "sendto() requires a SOCK_DGRAM socket",
                    ))
                }
            }
        };
        let n = udp
            .send_to(data, (addr.0.as_ref(), addr.1 as u16))
            .map_err(|e| net_error(&e))?;
        Ok(n as i64)
    }

    /// Python `socket.recvfrom(bufsize)` (UDP) -> (bytes, (host, port)).
    pub fn recvfrom(&self, bufsize: i64) -> Result<(Vec<u8>, (String, i64)), PyException> {
        if bufsize < 0 {
            return Err(crate::value_error("negative buffersize in recvfrom"));
        }
        let udp = match self.io_handle()? {
            IoHandle::Udp(u) => u,
            IoHandle::Tcp(_) => {
                return Err(PyException::new(
                    "OSError",
                    "recvfrom() requires a SOCK_DGRAM socket",
                ))
            }
        };
        let mut buf = vec![0u8; bufsize as usize];
        let (n, peer) = udp.recv_from(&mut buf).map_err(|e| net_error(&e))?;
        buf.truncate(n);
        Ok((buf, addr_tuple(peer)))
    }

    /// Python `socket.settimeout(seconds)` — applied to blocking reads,
    /// writes, and connect.
    pub fn settimeout(&self, seconds: f64) -> Result<(), PyException> {
        if seconds < 0.0 {
            return Err(crate::value_error("Timeout value out of range"));
        }
        let t = Some(Duration::from_secs_f64(seconds));
        *self.inner.timeout.lock().unwrap() = t;
        match &*self.inner.state.lock().unwrap() {
            SockState::Stream(s) => {
                s.set_read_timeout(t).map_err(|e| net_error(&e))?;
                s.set_write_timeout(t).map_err(|e| net_error(&e))?;
            }
            SockState::Udp(u) => {
                u.set_read_timeout(t).map_err(|e| net_error(&e))?;
                u.set_write_timeout(t).map_err(|e| net_error(&e))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Python `socket.getsockname()` -> (host, port).
    pub fn getsockname(&self) -> Result<(String, i64), PyException> {
        match &*self.inner.state.lock().unwrap() {
            SockState::Listener(l) => Ok(addr_tuple(l.local_addr().map_err(|e| net_error(&e))?)),
            SockState::Stream(s) => Ok(addr_tuple(s.local_addr().map_err(|e| net_error(&e))?)),
            SockState::Udp(u) => Ok(addr_tuple(u.local_addr().map_err(|e| net_error(&e))?)),
            // CPython: an unbound AF_INET socket reports ('0.0.0.0', 0).
            SockState::Fresh { .. } => Ok(("0.0.0.0".to_string(), 0)),
            SockState::Closed => Err(bad_fd()),
        }
    }

    /// Python `socket.getpeername()` -> (host, port).
    pub fn getpeername(&self) -> Result<(String, i64), PyException> {
        match &*self.inner.state.lock().unwrap() {
            SockState::Stream(s) => Ok(addr_tuple(s.peer_addr().map_err(|e| net_error(&e))?)),
            SockState::Udp(u) => Ok(addr_tuple(u.peer_addr().map_err(|e| net_error(&e))?)),
            SockState::Closed => Err(bad_fd()),
            _ => Err(PyException::new(
                "OSError",
                "[Errno 107] Transport endpoint is not connected",
            )),
        }
    }

    /// Python `socket.close()` — drops the descriptor; every clone of this
    /// handle observes the closed state (Python object semantics).
    pub fn close(&mut self) -> Result<(), PyException> {
        *self.inner.state.lock().unwrap() = SockState::Closed;
        Ok(())
    }
}

/// Python `socket.socket(family, type)` — the module-level constructor.
pub fn socket(family: i64, kind: i64) -> Result<Socket, PyException> {
    if family != AF_INET && family != AF_INET6 {
        // Verified against python3: OSError('[Errno 97] Address family not supported by protocol')
        return Err(PyException::new(
            "OSError",
            "[Errno 97] Address family not supported by protocol",
        ));
    }
    if kind != SOCK_STREAM && kind != SOCK_DGRAM {
        return Err(PyException::new("OSError", "[Errno 22] Invalid argument"));
    }
    Ok(Socket::from_state(SockState::Fresh { kind }))
}

/// Python `socket.gethostname()`.
pub fn gethostname() -> String {
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        // SAFETY: buf is a valid, writable buffer of the passed length;
        // gethostname NUL-terminates on success.
        let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
        if rc == 0 {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
        "localhost".to_string()
    }
    #[cfg(not(unix))]
    {
        "localhost".to_string()
    }
}
