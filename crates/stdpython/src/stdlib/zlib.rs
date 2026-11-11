//! Python zlib module implementation
//!
//! Deflate and inflate in the three framings `wbits` selects (zlib's,
//! raw, gzip's; and, for inflating, zlib's or gzip's by the header), the
//! two checksums, and a `decompressobj` that stops at the end of the
//! first stream and keeps what follows as `unused_data`. Backed by
//! `miniz_oxide`, which needs only `alloc`.
//!
//! Inflated output, checksums, headers and trailers are CPython's. Three
//! differences are ledgered in `docs/spec.md` §12.3: the deflate body
//! `compress` makes is miniz's encoding, which inflates to the same bytes
//! but is not byte-identical to zlib's; a corrupt deflate body is
//! `Error -3 while decompressing data: invalid data`, where zlib names the
//! exact fault (`invalid block type`, …); and `Decompress.decompress`
//! gives a stream's output in one piece once the stream is whole, where
//! zlib gives it piece by piece (the concatenation is the same).
//! Header, check, truncation and option errors are CPython's text.

use crate::PyException;
use alloc::boxed::Box;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::TINFLStatus;
use miniz_oxide::inflate::core::{self, DecompressorOxide, inflate_flags};

/// zlib.MAX_WBITS
pub const MAX_WBITS: i64 = 15;
/// zlib.DEFLATED
pub const DEFLATED: i64 = 8;
/// zlib.Z_DEFAULT_COMPRESSION
pub const Z_DEFAULT_COMPRESSION: i64 = -1;
/// zlib.Z_BEST_SPEED
pub const Z_BEST_SPEED: i64 = 1;
/// zlib.Z_BEST_COMPRESSION
pub const Z_BEST_COMPRESSION: i64 = 9;

/// `zlib.crc32(data, value=0)`.
pub fn crc32(data: &[u8], value: u32) -> u32 {
    let mut crc = !value;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// `zlib.adler32(data, value=1)`.
pub fn adler32(data: &[u8], value: u32) -> u32 {
    let (mut a, mut b) = (value & 0xffff, value >> 16);
    for &x in data {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// The framing `wbits` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Framing {
    /// 9 to 15: zlib's header and Adler-32 trailer.
    Zlib,
    /// -15 to -9: the deflate stream alone.
    Raw,
    /// 25 to 31: gzip's header and CRC-32/length trailer.
    Gzip,
    /// 40 to 47, inflating only: zlib's or gzip's, by the header.
    Either,
}

fn framing(wbits: i64) -> Option<Framing> {
    Some(match wbits {
        9..=15 => Framing::Zlib,
        -15..=-9 => Framing::Raw,
        25..=31 => Framing::Gzip,
        40..=47 => Framing::Either,
        _ => return None,
    })
}

fn error(message: &str) -> PyException {
    PyException::new("zlib.error", message)
}

fn truncated() -> PyException {
    error("Error -5 while decompressing data: incomplete or truncated stream")
}

fn corrupt(what: &str) -> PyException {
    error(&format!("Error -3 while decompressing data: {what}"))
}

/// `zlib.compress(data, level=-1, wbits=15)`.
pub fn compress(data: &[u8], level: i64, wbits: i64) -> Result<Vec<u8>, PyException> {
    if !(-1..=9).contains(&level) {
        return Err(error("Bad compression level"));
    }
    // zlib refuses the inflate-only framings and anything else with the
    // same message it gives a bad level.
    let framing = match framing(wbits) {
        Some(Framing::Either) | None => return Err(error("Bad compression level")),
        Some(f) => f,
    };
    let level = if level < 0 { 6 } else { level as u8 };
    let body = compress_to_vec(data, level);
    Ok(match framing {
        Framing::Raw => body,
        Framing::Zlib | Framing::Either => {
            // FLEVEL as zlib sets it, then FCHECK making the pair a
            // multiple of 31.
            let flevel: u8 = match level {
                0 | 1 => 0,
                2..=5 => 1,
                6 => 2,
                _ => 3,
            };
            let cmf = 0x78u8;
            let mut flg = flevel << 6;
            flg += (31 - ((u16::from(cmf) << 8 | u16::from(flg)) % 31) as u8) % 31;
            let mut out = vec![cmf, flg];
            out.extend(body);
            out.extend(adler32(data, 1).to_be_bytes());
            out
        }
        Framing::Gzip => {
            // XFL as zlib sets it; OS 3 (Unix), as CPython's builds write.
            let xfl = match level {
                0 | 1 => 4,
                9 => 2,
                _ => 0,
            };
            let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, xfl, 3];
            out.extend(body);
            out.extend(crc32(data, 0).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out
        }
    })
}

/// `zlib.decompress(data, wbits=15)`. Bytes after the first stream are
/// ignored, as zlib's one-shot inflate ignores them.
pub fn decompress(data: &[u8], wbits: i64) -> Result<Vec<u8>, PyException> {
    decompress_stream(data, wbits).map(|(out, _)| out)
}

/// The first stream of `data`, framed as `wbits` says, inflated and
/// checked; and how many bytes of `data` it took, its trailer included.
pub fn decompress_stream(data: &[u8], wbits: i64) -> Result<(Vec<u8>, usize), PyException> {
    let Some(framing) = framing(wbits) else {
        return Err(error(
            "Error -2 while preparing to decompress data: inconsistent stream state",
        ));
    };
    let gzip = match (framing, data.starts_with(&[0x1f, 0x8b])) {
        (Framing::Raw, _) => None,
        (Framing::Gzip, _) | (Framing::Either, true) => Some(true),
        (Framing::Zlib, _) | (Framing::Either, false) => Some(false),
    };
    let start = match gzip {
        None => Some(0),
        Some(true) => {
            if data.len() < 10 {
                return Err(truncated());
            }
            gzip_body(data)
        }
        Some(false) => match data.get(..2) {
            None => return Err(truncated()),
            Some(&[cmf, flg])
                if cmf & 0x0f == 8 && (u16::from(cmf) << 8 | u16::from(flg)) % 31 == 0 =>
            {
                if flg & 0x20 != 0 {
                    return Err(error("Error 2 while decompressing data"));
                }
                Some(2)
            }
            _ => None,
        },
    };
    let Some(start) = start else {
        return Err(corrupt("incorrect header check"));
    };
    let (out, taken) = match inflate(&data[start..]) {
        Ok(inflated) => inflated,
        Err(TINFLStatus::FailedCannotMakeProgress | TINFLStatus::NeedsMoreInput) => {
            return Err(truncated());
        }
        Err(_) => return Err(corrupt("invalid data")),
    };
    let trailer = &data[start + taken..];
    let wanted = match gzip {
        None => 0,
        Some(true) => 8,
        Some(false) => 4,
    };
    if trailer.len() < wanted {
        return Err(truncated());
    }
    let checked = match gzip {
        None => true,
        Some(true) => {
            trailer[..4] == crc32(&out, 0).to_le_bytes()
                && trailer[4..8] == (out.len() as u32).to_le_bytes()
        }
        Some(false) => trailer[..4] == adler32(&out, 1).to_be_bytes(),
    };
    if !checked {
        return Err(corrupt("incorrect data check"));
    }
    Ok((out, start + taken + wanted))
}

/// Where the deflate stream of gzip-framed `data` starts, past the
/// optional extra field, name, comment and header CRC.
fn gzip_body(data: &[u8]) -> Option<usize> {
    if data[2] != 8 {
        return None;
    }
    let flags = data[3];
    let mut at = 10;
    if flags & 4 != 0 {
        let extra = usize::from(u16::from_le_bytes([*data.get(at)?, *data.get(at + 1)?]));
        at += 2 + extra;
    }
    for bit in [8, 16] {
        if flags & bit != 0 {
            at += data.get(at..)?.iter().position(|&b| b == 0)? + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    Some(at)
}

/// The deflate stream at the start of `input` inflated, and how many
/// bytes of `input` it took.
fn inflate(mut input: &[u8]) -> Result<(Vec<u8>, usize), TINFLStatus> {
    let flags = inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let mut out = vec![0; input.len().saturating_mul(2).max(64)];
    let mut decompressor = Box::<DecompressorOxide>::default();
    let (mut at, mut taken) = (0, 0);
    loop {
        let (status, consumed, written) =
            core::decompress(&mut decompressor, input, &mut out, at, flags);
        at += written;
        taken += consumed;
        match status {
            TINFLStatus::Done => {
                out.truncate(at);
                return Ok((out, taken));
            }
            TINFLStatus::HasMoreOutput => {
                input = &input[consumed.min(input.len())..];
                let grown = out.len().saturating_mul(2);
                out.resize(grown, 0);
            }
            status => return Err(status),
        }
    }
}

/// `zlib.decompressobj(wbits=15)`: inflates the first stream of what it
/// is given across calls; what follows that stream is `unused_data`.
#[derive(Clone, Debug)]
pub struct Decompress {
    wbits: i64,
    held: Vec<u8>,
    /// Bytes after the end of the stream.
    pub unused_data: Vec<u8>,
    /// Whether the end of the stream has been reached.
    pub eof: bool,
}

/// `zlib.decompressobj(wbits)`.
pub fn decompressobj(wbits: i64) -> Result<Decompress, PyException> {
    if framing(wbits).is_none() {
        return Err(PyException::new("ValueError", "Invalid initialization option"));
    }
    Ok(Decompress {
        wbits,
        held: Vec::new(),
        unused_data: Vec::new(),
        eof: false,
    })
}

impl Decompress {
    /// `d.decompress(data)`: the output the stream gives once it is
    /// whole. Input before then is held, and `b""` given back; after the
    /// end of the stream input goes to `unused_data`.
    pub fn decompress(&mut self, data: &[u8]) -> Result<Vec<u8>, PyException> {
        if self.eof {
            self.unused_data.extend_from_slice(data);
            return Ok(Vec::new());
        }
        self.held.extend_from_slice(data);
        match decompress_stream(&self.held, self.wbits) {
            Ok((out, used)) => {
                self.unused_data = self.held.split_off(used);
                self.held.clear();
                self.eof = true;
                Ok(out)
            }
            Err(e) if e.message.contains("Error -5") => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// `d.flush()`: nothing is held back once the stream is whole.
    pub fn flush(&mut self) -> Vec<u8> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was produced by CPython 3.12's zlib.

    #[test]
    fn checksums() {
        assert_eq!(crc32(b"hello", 0), 907060870);
        assert_eq!(adler32(b"hello", 1), 103547413);
        assert_eq!(crc32(b"world", crc32(b"hello ", 0)), 222957957);
        assert_eq!(adler32(b"", 1), 1);
    }

    #[test]
    fn headers_follow_the_level() {
        let data = b"hello hello hello";
        for (level, zlib, gzip) in [
            (-1, "789c", "1f8b0800000000000003"),
            (0, "7801", "1f8b0800000000000403"),
            (1, "7801", "1f8b0800000000000403"),
            (2, "785e", "1f8b0800000000000003"),
            (5, "785e", "1f8b0800000000000003"),
            (6, "789c", "1f8b0800000000000003"),
            (7, "78da", "1f8b0800000000000003"),
            (9, "78da", "1f8b0800000000000203"),
        ] {
            assert_eq!(hex(&compress(data, level, 15).unwrap()[..2]), zlib);
            assert_eq!(hex(&compress(data, level, 31).unwrap()[..10]), gzip);
        }
    }

    #[test]
    fn round_trips_in_every_framing() {
        let data = b"abc".repeat(100);
        for (wbits, inflate_with) in [(15, 15), (31, 47), (15, 47), (-15, -15), (31, 31)] {
            let packed = compress(&data, 9, wbits).unwrap();
            assert_eq!(decompress(&packed, inflate_with).unwrap(), data);
        }
        assert_eq!(hex(&compress(b"", 6, -15).unwrap()), "0300");
    }

    #[test]
    fn errors_are_cpythons() {
        let good = compress(b"abc", -1, 15).unwrap();
        let message = |r: Result<Vec<u8>, PyException>| r.unwrap_err().message;
        assert_eq!(
            message(decompress(b"xx", 15)),
            "Error -3 while decompressing data: incorrect header check"
        );
        assert_eq!(
            message(decompress(&good, 99)),
            "Error -2 while preparing to decompress data: inconsistent stream state"
        );
        assert_eq!(
            message(decompress(&good[..good.len() - 2], 15)),
            "Error -5 while decompressing data: incomplete or truncated stream"
        );
        let mut bad = good.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert_eq!(
            message(decompress(&bad, 15)),
            "Error -3 while decompressing data: incorrect data check"
        );
        assert_eq!(message(compress(b"a", 12, 15)), "Bad compression level");
        assert_eq!(message(compress(b"a", 1, 99)), "Bad compression level");
        assert_eq!(decompress(b"x\x9c\xff\xff", 15).unwrap_err().exception_type, "zlib.error");
    }

    #[test]
    fn decompressobj_keeps_what_follows() {
        let mut stream = compress(b"abc", -1, 15).unwrap();
        stream.extend_from_slice(b"tail");
        let mut d = decompressobj(15).unwrap();
        // Given in two pieces: nothing until the stream is whole.
        assert_eq!(d.decompress(&stream[..4]).unwrap(), b"");
        assert!(!d.eof);
        assert_eq!(d.decompress(&stream[4..]).unwrap(), b"abc");
        assert_eq!(d.unused_data, b"tail");
        assert!(d.eof);
    }

    fn hex(bytes: &[u8]) -> alloc::string::String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
