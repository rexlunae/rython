//! Python urllib.parse — URL parsing/joining and percent-encoding.
//!
//! Models the CPython 3.11 surface requests uses through
//! `from urllib.parse import ...` (requests' compat.py): urlparse /
//! urlsplit / urlunparse / urljoin / urlencode / quote / unquote /
//! urldefrag. Behavior pinned against python3 in the runtime semantics
//! tests (`urllib_parse_matches_cpython`).

use crate::{AsStrLike, PyException};

/// CPython's `ParseResult` / `SplitResult` (the 6/5-tuple with named
/// fields). `urlsplit` returns the same struct with `params` empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseResult {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub params: String,
    pub query: String,
    pub fragment: String,
}

impl ParseResult {
    /// The lowercased hostname (IPv6 brackets stripped, PORT removed) —
    /// `None` when the netloc has no host part. CPython: `urlparse(
    /// "http://example.com:8080/").hostname` is `"example.com"`.
    pub fn hostname(&self) -> Option<String> {
        let host = self.host_part()?;
        if host.is_empty() {
            return None;
        }
        // Strip the port (`host:8080` -> `host`); a bracketed IPv6 host
        // keeps its brackets (`[::1]:8080` -> `[::1]`).
        let host = if host.starts_with('[') {
            match host.find(']') {
                Some(end) => &host[..=end],
                None => host,
            }
        } else {
            host.split(':').next().unwrap_or(host)
        };
        Some(host.to_ascii_lowercase())
    }

    /// The port as an int — `None` when absent, empty, or non-numeric
    /// (CPython raises ValueError only for a non-int port; requests
    /// never feeds one).
    pub fn port(&self) -> Option<i64> {
        let host = self.host_part()?;
        // A bracketed IPv6 host has no port of its own: the port follows
        // the closing bracket.
        let after_bracket = host.starts_with('[').then(|| host.find(']')).flatten();
        let port_part = match after_bracket {
            Some(end) => &host[end + 1..],
            None => host,
        };
        let (_, port) = port_part.rsplit_once(':')?;
        if port.is_empty() {
            return None;
        }
        port.parse::<i64>().ok()
    }

    /// The userinfo's username (`None` without an `@`).
    pub fn username(&self) -> Option<String> {
        let (userinfo, _) = self.netloc.split_once('@')?;
        Some(userinfo.split_once(':').map(|(u, _)| u).unwrap_or(userinfo).to_string())
    }

    /// The userinfo's password (`None` without a `:` in the userinfo).
    pub fn password(&self) -> Option<String> {
        let (userinfo, _) = self.netloc.split_once('@')?;
        userinfo.split_once(':').map(|(_, p)| p.to_string())
    }

    /// The netloc's host part (userinfo stripped, brackets kept for
    /// IPv6).
    fn host_part(&self) -> Option<&str> {
        let host = self.netloc.rsplit_once('@').map(|(_, h)| h).unwrap_or(&self.netloc);
        if host.is_empty() {
            return None;
        }
        Some(host)
    }

    /// Reconstruct the URL (CPython `geturl`).
    pub fn geturl(&self) -> String {
        let mut out = String::new();
        if !self.scheme.is_empty() {
            out.push_str(&self.scheme);
            out.push(':');
        }
        if !self.netloc.is_empty() {
            out.push_str("//");
            out.push_str(&self.netloc);
        }
        out.push_str(&self.path);
        if !self.params.is_empty() {
            out.push(';');
            out.push_str(&self.params);
        }
        if !self.query.is_empty() {
            out.push('?');
            out.push_str(&self.query);
        }
        if !self.fragment.is_empty() {
            out.push('#');
            out.push_str(&self.fragment);
        }
        out
    }
}

fn split_scheme(url: &str) -> (&str, &str) {
    match url.find(':') {
        Some(i) if i > 0 => {
            let scheme = &url[..i];
            if scheme
                .chars()
                .enumerate()
                .all(|(n, c)| c.is_ascii_alphabetic() || (n > 0 && (c.is_ascii_digit() || matches!(c, '+' | '-' | '.'))))
            {
                return (scheme, &url[i + 1..]);
            }
            ("", url)
        }
        _ => ("", url),
    }
}

fn split_netloc(rest: &str) -> (String, String) {
    if let Some(after) = rest.strip_prefix("//") {
        let end = after
            .find(|c| matches!(c, '/' | '?' | '#'))
            .unwrap_or(after.len());
        (after[..end].to_string(), after[end..].to_string())
    } else {
        (String::new(), rest.to_string())
    }
}

fn split_query_fragment(path_query_fragment: &str) -> (String, String, String) {
    let (path, frag) = match path_query_fragment.split_once('#') {
        Some((p, f)) => (p, f.to_string()),
        None => (path_query_fragment, String::new()),
    };
    let (path, query) = match path.split_once('?') {
        Some((p, q)) => (p, q.to_string()),
        None => (path, String::new()),
    };
    (path.to_string(), query, frag)
}

fn split_params(path: &str) -> (String, String) {
    match path.rfind(';') {
        Some(i) => (path[..i].to_string(), path[i + 1..].to_string()),
        None => (path.to_string(), String::new()),
    }
}

/// Parse a URL into its components (CPython `urlparse`). The argument is
/// string-like (PyValue/String/&str all work through AsStrLike — the call
/// sites pass boxed fields like `request.url`).
pub fn urlparse<S: AsStrLike + ?Sized>(url: &S) -> Result<ParseResult, PyException> {
    Ok(urlparse_inner(url.as_str_like()))
}

/// Parse a URL into its components, params empty (CPython `urlsplit`).
pub fn urlsplit<S: AsStrLike + ?Sized>(url: &S) -> Result<ParseResult, PyException> {
    let mut r = urlparse_inner(url.as_str_like());
    r.params = String::new();
    Ok(r)
}

fn urlparse_inner(url: &str) -> ParseResult {
    let (scheme, rest) = split_scheme(url);
    let (netloc, path_query_fragment) = split_netloc(rest);
    let (path, query, fragment) = split_query_fragment(&path_query_fragment);
    let (path, params) = split_params(&path);
    ParseResult {
        scheme: scheme.to_string(),
        netloc,
        path,
        params,
        query,
        fragment,
    }
}

/// Reconstruct a URL from its six components (CPython `urlunparse`).
/// The components are string-like (PyValue/String/&str all work).
pub fn urlunparse<S: AsStrLike>(parts: (S, S, S, S, S, S)) -> Result<String, PyException> {
    let (scheme, netloc, path, params, query, fragment) = parts;
    let scheme = scheme.as_str_like();
    let netloc = netloc.as_str_like();
    let path = path.as_str_like();
    let params = params.as_str_like();
    let query = query.as_str_like();
    let fragment = fragment.as_str_like();
    let mut out = String::new();
    if !scheme.is_empty() {
        out.push_str(scheme);
        out.push(':');
    }
    if !netloc.is_empty() {
        out.push_str("//");
        out.push_str(netloc);
    }
    out.push_str(path);
    if !params.is_empty() {
        out.push(';');
        out.push_str(params);
    }
    if !query.is_empty() {
        out.push('?');
        out.push_str(query);
    }
    if !fragment.is_empty() {
        out.push('#');
        out.push_str(fragment);
    }
    Ok(out)
}

/// Resolve `url` against `base` (CPython `urljoin` — the subset requests
/// uses: an absolute `url` wins; otherwise the base's scheme/netloc
/// combine with the resolved path; `..`/`.` segments collapse).
pub fn urljoin<S: AsStrLike + ?Sized>(base: &S, url: &S) -> Result<String, PyException> {
    let b = urlparse_inner(base.as_str_like());
    let u = urlparse_inner(url.as_str_like());
    // An absolute target wins entirely.
    if !u.scheme.is_empty() {
        return Ok(u.geturl());
    }
    let scheme = if b.scheme.is_empty() { u.scheme.clone() } else { b.scheme.clone() };
    if !u.netloc.is_empty() {
        // Network-path reference: same scheme, the target's authority.
        return Ok(urlunparse((scheme, u.netloc, u.path, u.params, u.query, u.fragment))?);
    }
    let (path, params) = if u.path.is_empty() {
        // Empty path: keep the base path, take the target's query.
        let params = if u.params.is_empty() { b.params.clone() } else { u.params.clone() };
        (b.path.clone(), params)
    } else if u.path.starts_with('/') {
        (u.path.clone(), u.params.clone())
    } else {
        // Relative path: merge with the base directory, collapse dots.
        let base_dir = match b.path.rfind('/') {
            Some(i) => &b.path[..=i],
            None => "/",
        };
        (normalize_dot_segments(&format!("{base_dir}{}", u.path)), u.params.clone())
    };
    let query = if u.query.is_empty() && u.path.is_empty() {
        b.query.clone()
    } else {
        u.query.clone()
    };
    Ok(urlunparse((scheme, b.netloc, path, params, query, u.fragment))?)
}

/// RFC 3986 remove_dot_segments, applied to a path.
fn normalize_dot_segments(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let trailing_slash = path.ends_with('/');
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    let mut joined = out.join("/");
    if !joined.starts_with('/') && path.starts_with('/') {
        joined.insert(0, '/');
    }
    if trailing_slash && !joined.ends_with('/') {
        joined.push('/');
    }
    joined
}

/// Percent-encode a string (CPython `quote`): unreserved chars pass
/// through, everything else is `%XX` (uppercase hex). `safe` adds extra
/// always-safe characters.
pub fn quote<S: AsStrLike + ?Sized>(s: &S, safe: Option<&str>) -> Result<String, PyException> {
    let s = s.as_str_like();
    let always = |c: char| -> bool {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~')
    };
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if always(c) || safe.is_some_and(|sf| sf.contains(c)) {
            out.push(c);
        } else if c.is_ascii() {
            out.push_str(&format!("%{:02X}", c as u32));
        } else {
            // Non-ASCII: UTF-8 bytes each percent-encoded.
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    Ok(out)
}

/// Percent-decode a string (CPython `unquote`). `%XX` sequences decode;
/// a lone `%` passes through.
pub fn unquote<S: AsStrLike + ?Sized>(s: &S) -> Result<String, PyException> {
    let s = s.as_str_like();
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // CPython decodes to latin-1-escape then utf-8; a simple lossy utf-8
    // reconstruction matches for well-formed input.
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Split a URL into `(url, fragment)` (CPython `urldefrag`).
pub fn urldefrag<S: AsStrLike + ?Sized>(url: &S) -> Result<(String, String), PyException> {
    let url = url.as_str_like();
    match url.split_once('#') {
        Some((u, f)) => Ok((u.to_string(), f.to_string())),
        None => Ok((url.to_string(), String::new())),
    }
}

/// Encode a query mapping/list into `k=v&k=v` (CPython `urlencode`):
/// accepts a boxed PyValue holding a dict (PyDict) or an iterable of
/// 2-tuples (the shapes requests passes), values via quote_plus.
pub fn urlencode(query: &crate::PyValue, doseq: bool) -> Result<String, PyException> {
    use crate::PyValue;
    let mut pairs: Vec<(String, String)> = Vec::new();
    let str_of = |v: &PyValue| -> String {
        match v {
            PyValue::Str(s) => s.clone(),
            PyValue::Int(i) => i.to_string(),
            PyValue::Float(f) => f.to_string(),
            PyValue::Bool(b) => b.to_string(),
            _ => String::new(),
        }
    };
    match query {
        PyValue::Dict(d) => {
            for (k, v) in d.iter() {
                if doseq && matches!(v, PyValue::Tuple(_)) {
                    if let PyValue::Tuple(members) = v {
                        for member in members.iter() {
                            pairs.push((k.clone(), str_of(member)));
                        }
                    }
                } else {
                    pairs.push((k.clone(), str_of(v)));
                }
            }
        }
        PyValue::Tuple(members) => {
            for m in members.iter() {
                if let PyValue::Tuple(pair) = m {
                    if let [PyValue::Str(k), v] = pair.as_slice() {
                        pairs.push((k.clone(), str_of(v)));
                    }
                }
            }
        }
        _ => {}
    }
    let mut parts = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        let k = quote_plus(&k)?;
        let v = quote_plus(&v)?;
        parts.push(format!("{k}={v}"));
    }
    Ok(parts.join("&"))
}

/// `quote` with `safe=""` and space as `+` (CPython `quote_plus`).
pub fn quote_plus<S: AsStrLike + ?Sized>(s: &S) -> Result<String, PyException> {
    let s = s.as_str_like();
    let mut out = quote(s, Some(""))?;
    out = out.replace("%20", "+");
    Ok(out)
}

/// `unquote` with `+` decoded as a space (CPython `unquote_plus`).
pub fn unquote_plus<S: AsStrLike + ?Sized>(s: &S) -> Result<String, PyException> {
    let s = s.as_str_like();
    let mut out = unquote(&s.replace('+', " "))?;
    Ok(out)
}
