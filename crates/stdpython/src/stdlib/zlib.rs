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

/// The framing `wbits` selects for compressing.
fn framing(wbits: i64) -> Option<Framing> {
    Some(match wbits {
        9..=15 => Framing::Zlib,
        -15..=-9 => Framing::Raw,
        25..=31 => Framing::Gzip,
        _ => return None,
    })
}

/// The framing `wbits` selects for inflating, and the largest window a
/// zlib header may declare (`None`: whatever it declares, for 0).
fn inflate_framing(wbits: i64) -> Option<(Framing, Option<u8>)> {
    Some(match wbits {
        0 => (Framing::Zlib, None),
        8..=15 => (Framing::Zlib, Some(wbits as u8)),
        -15..=-8 => (Framing::Raw, None),
        16 | 24..=31 => (Framing::Gzip, None),
        32 => (Framing::Either, None),
        40..=47 => (Framing::Either, Some((wbits - 32) as u8)),
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
    let Some(framing) = framing(wbits) else {
        return Err(error("Bad compression level"));
    };
    let level = if level < 0 { 6 } else { level as u8 };
    // The window wbits asks for. miniz_oxide always searches a 32 KiB
    // window, so a smaller one is kept by storing the data instead of
    // deflating it once a back-reference could reach past the window.
    let bits = match framing {
        Framing::Raw => -wbits,
        Framing::Gzip => wbits - 16,
        _ => wbits,
    };
    let stored = bits < 15 && data.len() > 1usize << bits;
    let body = compress_to_vec(data, if stored { 0 } else { level });
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
            // CINFO: the window, as log2 of its size less 8.
            let cmf = ((bits - 8) as u8) << 4 | 8;
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
    let Some((framing, window)) = inflate_framing(wbits) else {
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
        None => 0,
        Some(true) => gzip_body(data)?,
        Some(false) => zlib_body(data, window)?,
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

/// Where the deflate stream of zlib-framed `data` starts, its header
/// checked as zlib checks it: the check bits, the method, the window it
/// declares against `window`, and no preset dictionary.
fn zlib_body(data: &[u8], window: Option<u8>) -> Result<usize, PyException> {
    let &[cmf, flg, ..] = data else {
        return Err(truncated());
    };
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(corrupt("incorrect header check"));
    }
    if cmf & 0x0f != 8 {
        return Err(corrupt("unknown compression method"));
    }
    let declared = (cmf >> 4) + 8;
    if declared > 15 || declared > window.unwrap_or(15) {
        return Err(corrupt("invalid window size"));
    }
    if flg & 0x20 != 0 {
        // zlib's Z_NEED_DICT: no preset dictionary is given here.
        return Err(error("Error 2 while decompressing data"));
    }
    Ok(2)
}

/// Where the deflate stream of gzip-framed `data` starts, past the
/// optional extra field, name, comment and header CRC, with the header
/// checked as zlib checks it. A header cut short is a truncated stream.
fn gzip_body(data: &[u8]) -> Result<usize, PyException> {
    if data.len() >= 2 && data[..2] != [0x1f, 0x8b] {
        return Err(corrupt("incorrect header check"));
    }
    if data.len() < 10 {
        return Err(truncated());
    }
    if data[2] != 8 {
        return Err(corrupt("unknown compression method"));
    }
    let flags = data[3];
    if flags & 0xe0 != 0 {
        return Err(corrupt("unknown header flags set"));
    }
    let mut at = 10;
    if flags & 4 != 0 {
        let length = data.get(at..at + 2).ok_or_else(truncated)?;
        at += 2 + usize::from(u16::from_le_bytes([length[0], length[1]]));
        if at > data.len() {
            return Err(truncated());
        }
    }
    for bit in [8, 16] {
        if flags & bit != 0 {
            let end = data[at..]
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(truncated)?;
            at += end + 1;
        }
    }
    if flags & 2 != 0 {
        let stored = data.get(at..at + 2).ok_or_else(truncated)?;
        let wanted = (crc32(&data[..at], 0) & 0xffff) as u16;
        if u16::from_le_bytes([stored[0], stored[1]]) != wanted {
            return Err(corrupt("header crc mismatch"));
        }
        at += 2;
    }
    Ok(at)
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
    if inflate_framing(wbits).is_none() {
        return Err(PyException::new(
            "ValueError",
            "Invalid initialization option",
        ));
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
        assert_eq!(
            decompress(b"x\x9c\xff\xff", 15).unwrap_err().exception_type,
            "zlib.error"
        );
    }

    #[test]
    fn small_windows_round_trip_with_cpythons_headers() {
        // CPython 3: zlib.compress(b'hello world ' * 50, 6, w)[:2].hex()
        // is 1895 2891 388d 4889 5885 6881 789c for w = 9 to 15.
        let d = b"hello world ".repeat(50);
        let heads = ["1895", "2891", "388d", "4889", "5885", "6881", "789c"];
        for (w, head) in (9..=15).zip(heads) {
            let c = compress(&d, 6, w).unwrap();
            assert_eq!(format!("{:02x}{:02x}", c[0], c[1]), head);
            assert_eq!(decompress(&c, w).unwrap(), d);
            assert_eq!(decompress(&c, w + 32).unwrap(), d);
        }
        // Raw and gzip streams with small windows round-trip too.
        for w in [-9, -12, 25, 28] {
            let c = compress(&d, 6, w).unwrap();
            assert_eq!(decompress(&c, w).unwrap(), d);
        }
        // Longer than a small window, the body is stored (BTYPE 00), so it
        // holds no back-reference at all; within the window, a reference
        // reaches back at most len - 1 bytes. Either way a decoder limited
        // to that window accepts it.
        for w in [-9, 9, 25] {
            let c = compress(&d, 6, w).unwrap();
            let at = match w {
                -9 => 0,
                9 => 2,
                _ => 10,
            };
            assert_eq!((c[at] >> 1) & 3, 0, "wbits {w}");
        }
        // Data that fits the window is still deflated.
        let small = b"abcabcabcabc".repeat(10);
        assert!(compress(&small, 6, 9).unwrap().len() < small.len());
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

    #[test]
    fn headers_are_checked_as_zlib_checks_them() {
        let message = |r: Result<Vec<u8>, PyException>| r.unwrap_err().message;
        let gzip = compress(b"hello", 6, 31).unwrap();
        let with = |at: usize, byte: u8| {
            let mut d = gzip.clone();
            d[at] = byte;
            d
        };
        let corrupt = |what: &str| alloc::format!("Error -3 while decompressing data: {what}");
        let short = "Error -5 while decompressing data: incomplete or truncated stream";
        assert_eq!(
            message(decompress(&with(3, 0xe0), 31)),
            corrupt("unknown header flags set")
        );
        assert_eq!(
            message(decompress(&with(1, 0x8c), 31)),
            corrupt("incorrect header check")
        );
        assert_eq!(
            message(decompress(&with(2, 7), 31)),
            corrupt("unknown compression method")
        );
        // A header CRC: the right one passes, a wrong one is refused.
        let mut header = gzip[..10].to_vec();
        header[3] |= 2;
        let crc = (crc32(&header, 0) & 0xffff) as u16;
        for (stored, ok) in [(crc, true), (crc ^ 1, false)] {
            let mut d = header.clone();
            d.extend(stored.to_le_bytes());
            d.extend(&gzip[10..]);
            match decompress(&d, 31) {
                Ok(out) => assert!(ok && out == b"hello"),
                Err(e) => assert!(!ok && e.message == corrupt("header crc mismatch")),
            }
        }
        // Optional fields cut short are a truncated stream, not a panic.
        let mut extra = with(3, 4)[..10].to_vec();
        extra.extend([0xff, 0xff]);
        assert_eq!(message(decompress(&extra, 31)), short);
        let mut name = with(3, 8)[..10].to_vec();
        name.extend(b"nam");
        assert_eq!(message(decompress(&name, 31)), short);
        // A zlib header declaring a larger window than wbits allows, or
        // one larger than 32K.
        let zlib = compress(b"hello", 6, 15).unwrap();
        assert_eq!(
            message(decompress(&zlib, 9)),
            corrupt("invalid window size")
        );
        let mut big = zlib.clone();
        big[0] = 0x88;
        big[1] = 0x98;
        assert_eq!(message(decompress(&big, 0)), corrupt("invalid window size"));
        // A preset dictionary is asked for.
        assert_eq!(
            message(decompress(&[0x78, 0x3f, 0, 0, 0, 1], 15)),
            "Error 2 while decompressing data"
        );
    }

    #[test]
    fn every_inflate_window_mode() {
        let zlib = compress(b"hello", 6, 15).unwrap();
        let raw = compress(b"hello", 6, -15).unwrap();
        let gzip = compress(b"hello", 6, 31).unwrap();
        for (data, wbits) in [
            (&zlib, 0),
            (&raw, -8),
            (&gzip, 16),
            (&gzip, 24),
            (&zlib, 32),
            (&gzip, 32),
        ] {
            assert_eq!(decompress(data, wbits).unwrap(), b"hello", "wbits {wbits}");
        }
        for wbits in [7, -7, 17, 23, 33, 39, 48] {
            assert_eq!(
                decompress(&zlib, wbits).unwrap_err().message,
                "Error -2 while preparing to decompress data: inconsistent stream state"
            );
        }
        assert_eq!(decompressobj(99).unwrap_err().exception_type, "ValueError");
    }

    #[test]
    fn decompressobj_waits_for_a_split_name() {
        let gzip = compress(b"hello", 6, 31).unwrap();
        let mut data = gzip[..10].to_vec();
        data[3] = 8;
        data.extend(b"name\0");
        data.extend(&gzip[10..]);
        let mut d = decompressobj(31).unwrap();
        assert_eq!(d.decompress(&data[..12]).unwrap(), b"");
        assert_eq!(d.decompress(&data[12..]).unwrap(), b"hello");
        assert!(d.eof);
    }

    fn hex(bytes: &[u8]) -> alloc::string::String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
