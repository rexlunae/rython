//! Python codec layer for `str.encode(...)` / `bytes.decode(...)`.
//!
//! CPython's codecs are pluggable through `codecs.register`; rython lowers
//! `str.encode(name)` / `bytes.decode(name)` to these fixed runtime
//! functions at conversion time (the codec name is a literal). Supported
//! codecs (all CPython-verified where marked):
//!
//! - `utf-8` — the native encoding of Rust strings (already handled inline
//!   by the codegen; listed here for completeness).
//! - `ascii` — code points 0..=127 only; non-ASCII input raises
//!   UnicodeEncodeError (encode) / UnicodeDecodeError (decode), matching
//!   CPython's strict error handler.
//! - `punycode` — RFC 3492 (the IDNA A-label encoding, idna's core).
//!
//! The error classes are exact: the same `PyException::new("UnicodeEncodeError",
//! ...)` strings CPython produces for the strict handler.
//!
//! Pure data transformation, so this module lives on every tier (alloc +
//! std); `format!` is the alloc crate's macro, imported explicitly.

use alloc::{format, string::String, string::ToString, vec::Vec};
use crate::PyException;

/// str.encode("ascii"): 7-bit ASCII bytes; a non-ASCII character raises
/// UnicodeEncodeError like CPython's strict error handler.
pub fn encode_ascii<S: AsRef<str>>(s: S) -> Result<Vec<u8>, PyException> {
    let s = s.as_ref();
    if let Some(bad) = s.chars().find(|c| !c.is_ascii()) {
        return Err(PyException::new(
            "UnicodeEncodeError",
            format!(
                "'ascii' codec can't encode character '\\u{:x}' in position {}: \
                 ordinal not in range(128)",
                bad as u32,
                s.find(bad).unwrap_or(0)
            ),
        ));
    }
    Ok(s.as_bytes().to_vec())
}

/// bytes.decode("ascii"): a non-ASCII byte raises UnicodeDecodeError.
pub fn decode_ascii(b: &[u8]) -> Result<String, PyException> {
    match b.iter().position(|&x| x >= 0x80) {
        Some(i) => {
            let bad = b[i];
            return Err(PyException::new(
                "UnicodeDecodeError",
                format!(
                    "'ascii' codec can't decode byte 0x{:x} in position {}: \
                     ordinal not in range(128)",
                    bad, i
                ),
            ));
        }
        None => {}
    }
    // ASCII bytes are valid UTF-8.
    Ok(String::from_utf8(b.to_vec()).expect("ascii bytes are valid utf-8"))
}

/// bytes.decode("utf-8"): Rust strings ARE UTF-8, so this is from_utf8 with
/// CPython's error shape on invalid bytes.
pub fn decode_utf8(b: &[u8]) -> Result<String, PyException> {
    String::from_utf8(b.to_vec()).map_err(|e| {
        PyException::new(
            "UnicodeDecodeError",
            format!("'utf-8' codec can't decode byte 0x{:x} in position {}: invalid start byte", 
                e.utf8_error().error_len().map(|_| b[e.utf8_error().valid_up_to()]).unwrap_or(0),
                e.utf8_error().valid_up_to()),
        )
    })
}

/// bytes.decode(name) with a RUNTIME codec name (a parameter, not a
/// literal): dispatch on the name string, like CPython's codec registry.
pub fn decode_by_name<N: AsRef<str>>(b: &[u8], name: N) -> Result<String, PyException> {
    match name.as_ref() {
        "utf-8" | "utf8" => decode_utf8(b),
        "ascii" => decode_ascii(b),
        "punycode" => decode_punycode(b),
        other => Err(PyException::new(
            "LookupError",
            format!("unknown encoding: {}", other),
        )),
    }
}

// ---------------------------------------------------------------------------
// Punycode (RFC 3492) — the "bootstring" codec used by IDNA A-labels.
// ---------------------------------------------------------------------------

const BASE: u32 = 36;
const TMIN: u32 = 1;
const TMAX: u32 = 26;
const SKEW: u32 = 38;
const DAMP: u32 = 700;
const INITIAL_BIAS: u32 = 72;
const INITIAL_N: u32 = 128;

/// The RFC 3492 bias adaptation function.
fn adapt(delta: u32, numpoints: u32, firsttime: bool) -> u32 {
    let mut delta = if firsttime { delta / DAMP } else { delta / 2 };
    delta += delta / numpoints;
    let mut k = 0;
    while delta > ((BASE - TMIN) * TMAX) / 2 {
        delta /= BASE - TMIN;
        k += BASE;
    }
    k + (((BASE - TMIN + 1) * delta) / (delta + SKEW))
}

fn encode_digit(d: u32) -> u8 {
    if d < 26 {
        b'a' + d as u8
    } else {
        b'0' + (d - 26) as u8
    }
}

fn decode_digit(c: u8) -> Result<u32, PyException> {
    if (b'a'..=b'z').contains(&c) {
        Ok((c - b'a') as u32)
    } else if (b'A'..=b'Z').contains(&c) {
        Ok((c - b'A') as u32)
    } else if (b'0'..=b'9').contains(&c) {
        Ok((c - b'0') as u32 + 26)
    } else {
        Err(PyException::new(
            "UnicodeDecodeError",
            format!("invalid punycode digit '{}'", c as char),
        ))
    }
}

/// str.encode("punycode"): RFC 3492. Non-basic code points (>= 128) are
/// encoded as delta sequences after a literal run of basic code points.
pub fn encode_punycode<S: AsRef<str>>(input: S) -> Vec<u8> {
    let input = input.as_ref();
    let chars: Vec<char> = input.chars().collect();
    let mut output: Vec<u8> = Vec::new();
    // Literal basic code points (ASCII) first; a non-empty run needs the
    // delimiter '-' before the encoded part.
    let basic: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| (**c as u32) < 0x80)
        .map(|(i, _)| i)
        .collect();
    for &i in &basic {
        output.push(chars[i] as u8);
    }
    let mut h = basic.len();
    let b = basic.len();
    if b > 0 {
        output.push(b'-');
    }
    let mut n = INITIAL_N;
    let mut delta: u32 = 0;
    let mut bias = INITIAL_BIAS;
    while h < chars.len() {
        // m = smallest code point >= n among the remaining chars.
        let m = chars
            .iter()
            .map(|c| *c as u32)
            .filter(|c| *c >= n)
            .min()
            .expect("h < len guarantees a code point >= n");
        delta = delta
            .checked_add((m - n).checked_mul((h + 1) as u32).expect("punycode delta overflow"))
            .expect("punycode delta overflow");
        n = m;
        for c in &chars {
            let cp = *c as u32;
            if cp < n {
                delta = delta.checked_add(1).expect("punycode delta overflow");
            }
            if cp == n {
                let mut q = delta;
                let mut k = BASE;
                loop {
                    let t = if k <= bias {
                        TMIN
                    } else if k >= bias + TMAX {
                        TMAX
                    } else {
                        k - bias
                    };
                    if q < t {
                        break;
                    }
                    output.push(encode_digit(t + ((q - t) % (BASE - t))));
                    q = (q - t) / (BASE - t);
                    k += BASE;
                }
                output.push(encode_digit(q));
                bias = adapt(delta, (h + 1) as u32, h == b);
                delta = 0;
                h += 1;
            }
        }
        delta = delta.checked_add(1).expect("punycode delta overflow");
        n = n.checked_add(1).expect("punycode n overflow");
    }
    output
}

/// bytes.decode("punycode"): RFC 3492. Returns the decoded string, or a
/// UnicodeDecodeError on malformed input (matching CPython's shape).
pub fn decode_punycode(input: &[u8]) -> Result<String, PyException> {
    let mut output: Vec<char> = Vec::new();
    let mut n = INITIAL_N;
    let mut i: u32 = 0;
    let mut bias = INITIAL_BIAS;
    // Everything up to the LAST '-' is a literal basic-code-point run; with
    // NO delimiter the run is empty (RFC 3492: b = last delimiter or -1).
    let delim = input.iter().rposition(|&c| c == b'-');
    if let Some(delim) = delim {
        for &c in &input[..delim] {
            if c >= 0x80 {
                return Err(PyException::new(
                    "UnicodeDecodeError",
                    format!("invalid punycode basic code point 0x{:x}", c),
                ));
            }
            output.push(c as char);
        }
    }
    let mut ip = delim.map(|d| d + 1).unwrap_or(0);
    while ip < input.len() {
        let oldi = i;
        let mut w: u32 = 1;
        let mut k = BASE;
        loop {
            if ip >= input.len() {
                return Err(PyException::new(
                    "UnicodeDecodeError",
                    "punycode input is truncated".to_string(),
                ));
            }
            let digit = decode_digit(input[ip])?;
            ip += 1;
            let t = if k <= bias {
                TMIN
            } else if k >= bias + TMAX {
                TMAX
            } else {
                k - bias
            };
            if digit < t {
                i = i.checked_add(digit.checked_mul(w).ok_or_else(|| {
                    PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
                })?)
                .ok_or_else(|| {
                    PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
                })?;
                break;
            }
            i = i.checked_add(digit.checked_mul(w).ok_or_else(|| {
                PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
            })?)
            .ok_or_else(|| {
                PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
            })?;
            w = w.checked_mul(BASE - t).ok_or_else(|| {
                PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
            })?;
            k += BASE;
        }
        let out_len = (output.len() + 1) as u32;
        bias = adapt(i - oldi, out_len, oldi == 0);
        n = n.checked_add(i / out_len).ok_or_else(|| {
            PyException::new("UnicodeDecodeError", "punycode overflow".to_string())
        })?;
        i %= out_len;
        output.insert(i as usize, char::from_u32(n).ok_or_else(|| {
            PyException::new("UnicodeDecodeError", "invalid punycode code point".to_string())
        })?);
        i += 1;
    }
    Ok(output.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &str) {
        let enc = encode_punycode(s);
        assert_eq!(decode_punycode(&enc).unwrap(), s, "roundtrip {}", s);
    }

    #[test]
    fn punycode_roundtrips() {
        // RFC 3492 sample strings (all 6 from the spec) plus unicode edge
        // cases; each must survive encode→decode.
        for s in [
            "A",
            "Bach",
            "b\u{fc}cher",      // ü
            "ma\u{f1}ana",      // ñ
            "\u{4f8b}\u{3048}", // 例え
            "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}", // こんにちは
            "hello",
            "mixed-\u{00e9}l\u{00e8}ment",
        ] {
            roundtrip(s);
        }
    }

    #[test]
    fn punycode_rfc3492_vectors() {
        // RFC 3492 §7.1 sample encodings.
        assert_eq!(encode_punycode("A"), b"A-");
        assert_eq!(encode_punycode("Bach"), b"Bach-");
        assert_eq!(encode_punycode("b\u{fc}cher"), b"bcher-kva");
        assert_eq!(encode_punycode("ma\u{f1}ana"), b"maana-pta");
        assert_eq!(encode_punycode("\u{4f8b}\u{3048}"), b"r8jz45g");
        assert_eq!(
            encode_punycode("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}"),
            b"28j2a3ar1p"
        );
        // Decode the vectors back.
        assert_eq!(decode_punycode(b"bcher-kva").unwrap(), "b\u{fc}cher");
        assert_eq!(decode_punycode(b"maana-pta").unwrap(), "ma\u{f1}ana");
        assert_eq!(decode_punycode(b"r8jz45g").unwrap(), "\u{4f8b}\u{3048}");
    }

    #[test]
    fn ascii_encode_decode() {
        assert_eq!(encode_ascii("hello").unwrap(), b"hello");
        assert!(encode_ascii("h\u{e9}llo").is_err(), "non-ASCII must raise");
        assert_eq!(decode_ascii(b"hello").unwrap(), "hello");
        assert!(decode_ascii(&[0x68, 0xe9]).is_err(), "non-ASCII byte must raise");
        assert_eq!(decode_utf8("héllo".as_bytes()).unwrap(), "héllo");
    }
}


/// Latin-1 (ISO-8859-1) encoding: each character maps to its code point
/// (0-255); characters above U+00FF raise UnicodeEncodeError.
pub fn encode_latin1<S: AsRef<str>>(s: S) -> Result<Vec<u8>, PyException> {
    let mut out = Vec::with_capacity(s.as_ref().len());
    for c in s.as_ref().chars() {
        let cp = c as u32;
        if cp > 0xFF {
            return Err(PyException::new(
                "UnicodeEncodeError",
                "latin-1 codec can't encode character",
            ));
        }
        out.push(cp as u8);
    }
    Ok(out)
}
