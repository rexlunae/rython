//! Old-style `%`-formatting — Python's printf operator on `str` and
//! `bytes`: `"hostname %r" % (host,)`, `b"%x\r\n%b\r\n" % (len, chunk)`
//! (urllib3's connection framing — round 34's %-operator cluster).
//!
//! The format string holds `%`-conversions; the RHS is one value or a
//! tuple of them, walked in order. Semantics verified against CPython
//! 3.14; an unsupported conversion raises the same typed error CPython's
//! engine does (ValueError for a bad conversion character, TypeError for
//! a wrong value kind or a wrong argument count) — never a silently
//! different string.

use crate::{PyException, PyValue};
#[cfg(feature = "alloc")]
use alloc::{format, string::String, string::ToString, vec::Vec};

/// The conversion code of one %-spec.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PercentCode {
    D,    // %d %i — decimal int (a float truncates, like CPython's int())
    O,    // %o — octal
    U,    // %u — unsigned (CPython: like %d)
    X,    // %x — lowercase hex
    BigX, // %X — uppercase hex
    E,    // %e — scientific
    BigE, // %E
    F,    // %f %F — fixed
    G,    // %g — general
    BigG, // %G
    C,    // %c — char (int → chr, or a one-char string)
    R,    // %r — repr
    S,    // %s — str
    A,    // %a — ascii()
    B,    // %b — bytes (bytes-formatting only)
}

/// One parsed %-conversion: the code plus flags/width/precision.
///
/// `pub` only because the value-source trait exposes it; it is internal
/// to the engine.
pub struct PercentSpec {
    /// The `%(name)s` mapping key, when the spec is the mapping form.
    pub(crate) key: Option<String>,
    pub(crate) code: PercentCode,
    minus: bool,
    plus: bool,
    zero: bool,
    space: bool,
    alt: bool,
    width: Option<i64>,
    precision: Option<i64>,
    star_width: bool,
    star_precision: bool,
}

/// The RHS of `fmt % values`: a single value or a tuple, providing its
/// values in order (recursively — Python's `(a, b)` is a tuple). The
/// engine advances `at` as it consumes; a leaf advances it by one.
pub trait PyFormatValue {
    fn count(&self) -> usize;
    fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
        -> Result<(), PyException>;
    fn render_bytes(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException>;
}

/// The RHS of `fmt % values`, in either form: a positional value or
/// tuple (PyFormatValue) or a mapping dict for `%(name)s` keys.
pub trait PyFormatRhs {
    /// Positional values available (0 for a mapping — a dict is consumed
    /// by key, and CPython's positional-vs-mapping mixing rules reject
    /// mixing).
    fn count(&self) -> usize;
    fn render_at(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException>;
    fn render_keyed(
        &self,
        key: &str,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException>;
    /// The bytes-mode positional render: default is the str-mode render
    /// re-encoded, which the bytes leaf overrides (raw bytes for %s/%b).
    fn render_at_bytes(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        let mut s = String::new();
        self.render_at(at, spec, &mut s)?;
        out.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

impl<T: PyFormatValue> PyFormatRhs for T {
    fn count(&self) -> usize {
        PyFormatValue::count(self)
    }
    fn render_at(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException> {
        PyFormatValue::render_str(self, at, spec, out)
    }
    fn render_keyed(
        &self,
        _key: &str,
        _spec: &PercentSpec,
        _out: &mut String,
    ) -> Result<(), PyException> {
        Err(PyException::new(
            "TypeError",
            "format requires a mapping",
        ))
    }
    fn render_at_bytes(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        PyFormatValue::render_bytes(self, at, spec, out)
    }
}

impl<V: PercentValue> PyFormatRhs for crate::PyDict<String, V> {
    fn count(&self) -> usize {
        0
    }
    fn render_at(
        &self,
        _at: &mut usize,
        _spec: &PercentSpec,
        _out: &mut String,
    ) -> Result<(), PyException> {
        unreachable!("a mapping RHS never renders positionally")
    }
    fn render_keyed(
        &self,
        key: &str,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException> {
        match self.get(key) {
            Some(v) => {
                let mut s = String::new();
                v.percent_render(spec, &mut s)?;
                let s = percent_apply(spec, s, false)?;
                out.push_str(&s);
                Ok(())
            }
            None => Err(PyException::new("KeyError", key.to_string())),
        }
    }
}

/// One concrete value that a %-conversion can render (the leaves of a
/// tuple RHS).
pub trait PercentValue {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String)
        -> Result<(), PyException>;
    /// The bytes-mode render (the `fmt % values` engine for a bytes
    /// format string): default is the str-mode render re-encoded, which
    /// the bytes leaf overrides (raw bytes for %s/%b) and the str leaf
    /// overrides to raise CPython's TypeError.
    fn percent_render_bytes(&self, spec: &PercentSpec, out: &mut Vec<u8>)
        -> Result<(), PyException> {
        let mut s = String::new();
        self.percent_render(spec, &mut s)?;
        out.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

macro_rules! py_format_value_tuple {
    () => {
        impl PyFormatValue for () {
            fn count(&self) -> usize {
                0
            }
            fn render_str(
                &self,
                _at: &mut usize,
                _spec: &PercentSpec,
                _out: &mut String,
            ) -> Result<(), PyException> {
                unreachable!("an empty RHS never renders a value")
            }
            fn render_bytes(
                &self,
                _at: &mut usize,
                _spec: &PercentSpec,
                _out: &mut Vec<u8>,
            ) -> Result<(), PyException> {
                unreachable!("an empty RHS never renders a value")
            }
        }
    };
    ($a:ident) => {
        impl<$a: PyFormatValue> PyFormatValue for ($a,) {
            fn count(&self) -> usize { self.0.count() }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_str(at, spec, out); }
                unreachable!("index out of range")
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_bytes(at, spec, out); }
                unreachable!("index out of range")
            }
        }
    };
    ($a:ident, $b:ident) => {
        impl<$a: PyFormatValue, $b: PyFormatValue> PyFormatValue for ($a, $b) {
            fn count(&self) -> usize { self.0.count() + self.1.count() }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_str(at, spec, out); }
                self.1.render_str(at, spec, out)
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_bytes(at, spec, out); }
                self.1.render_bytes(at, spec, out)
            }
        }
    };
    ($a:ident, $b:ident, $c:ident) => {
        impl<$a: PyFormatValue, $b: PyFormatValue, $c: PyFormatValue> PyFormatValue for ($a, $b, $c) {
            fn count(&self) -> usize { self.0.count() + self.1.count() + self.2.count() }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_str(at, spec, out); }
                if *at < self.0.count() + self.1.count() { return self.1.render_str(at, spec, out); }
                self.2.render_str(at, spec, out)
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                if *at < self.0.count() { return self.0.render_bytes(at, spec, out); }
                if *at < self.0.count() + self.1.count() { return self.1.render_bytes(at, spec, out); }
                self.2.render_bytes(at, spec, out)
            }
        }
    };
    ($a:ident, $b:ident, $c:ident, $d:ident) => {
        impl<$a: PyFormatValue, $b: PyFormatValue, $c: PyFormatValue, $d: PyFormatValue>
            PyFormatValue for ($a, $b, $c, $d) {
            fn count(&self) -> usize {
                self.0.count() + self.1.count() + self.2.count() + self.3.count()
            }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                if *at < n0 { return self.0.render_str(at, spec, out); }
                if *at < n1 { return self.1.render_str(at, spec, out); }
                if *at < n2 { return self.2.render_str(at, spec, out); }
                self.3.render_str(at, spec, out)
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                if *at < n0 { return self.0.render_bytes(at, spec, out); }
                if *at < n1 { return self.1.render_bytes(at, spec, out); }
                if *at < n2 { return self.2.render_bytes(at, spec, out); }
                self.3.render_bytes(at, spec, out)
            }
        }
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        impl<$a: PyFormatValue, $b: PyFormatValue, $c: PyFormatValue, $d: PyFormatValue,
            $e: PyFormatValue> PyFormatValue for ($a, $b, $c, $d, $e) {
            fn count(&self) -> usize {
                self.0.count() + self.1.count() + self.2.count() + self.3.count() + self.4.count()
            }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                let n3 = n2 + self.3.count();
                if *at < n0 { return self.0.render_str(at, spec, out); }
                if *at < n1 { return self.1.render_str(at, spec, out); }
                if *at < n2 { return self.2.render_str(at, spec, out); }
                if *at < n3 { return self.3.render_str(at, spec, out); }
                self.4.render_str(at, spec, out)
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                let n3 = n2 + self.3.count();
                if *at < n0 { return self.0.render_bytes(at, spec, out); }
                if *at < n1 { return self.1.render_bytes(at, spec, out); }
                if *at < n2 { return self.2.render_bytes(at, spec, out); }
                if *at < n3 { return self.3.render_bytes(at, spec, out); }
                self.4.render_bytes(at, spec, out)
            }
        }
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        impl<$a: PyFormatValue, $b: PyFormatValue, $c: PyFormatValue, $d: PyFormatValue,
            $e: PyFormatValue, $f: PyFormatValue> PyFormatValue for ($a, $b, $c, $d, $e, $f) {
            fn count(&self) -> usize {
                self.0.count() + self.1.count() + self.2.count() + self.3.count()
                    + self.4.count() + self.5.count()
            }
            fn render_str(&self, at: &mut usize, spec: &PercentSpec, out: &mut String)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                let n3 = n2 + self.3.count();
                let n4 = n3 + self.4.count();
                if *at < n0 { return self.0.render_str(at, spec, out); }
                if *at < n1 { return self.1.render_str(at, spec, out); }
                if *at < n2 { return self.2.render_str(at, spec, out); }
                if *at < n3 { return self.3.render_str(at, spec, out); }
                if *at < n4 { return self.4.render_str(at, spec, out); }
                self.5.render_str(at, spec, out)
            }
            fn render_bytes(&self, at: &mut usize, spec: &PercentSpec, out: &mut Vec<u8>)
                -> Result<(), PyException> {
                let n0 = self.0.count();
                let n1 = n0 + self.1.count();
                let n2 = n1 + self.2.count();
                let n3 = n2 + self.3.count();
                let n4 = n3 + self.4.count();
                if *at < n0 { return self.0.render_bytes(at, spec, out); }
                if *at < n1 { return self.1.render_bytes(at, spec, out); }
                if *at < n2 { return self.2.render_bytes(at, spec, out); }
                if *at < n3 { return self.3.render_bytes(at, spec, out); }
                if *at < n4 { return self.4.render_bytes(at, spec, out); }
                self.5.render_bytes(at, spec, out)
            }
        }
    };
}
py_format_value_tuple!();
py_format_value_tuple!(A);
py_format_value_tuple!(A, B);
py_format_value_tuple!(A, B, C);
py_format_value_tuple!(A, B, C, D);
py_format_value_tuple!(A, B, C, D, E);
py_format_value_tuple!(A, B, C, D, E, F);

impl<T: PercentValue> PyFormatValue for T {
    fn count(&self) -> usize {
        1
    }
    fn render_str(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException> {
        *at += 1;
        self.percent_render(spec, out)
    }
    fn render_bytes(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        *at += 1;
        self.percent_render_bytes(spec, out)
    }
}

impl<T: PercentValue + ?Sized> PercentValue for &T {
    fn percent_render(
        &self,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException> {
        (**self).percent_render(spec, out)
    }
}

impl<T: PercentValue> PyFormatValue for Option<T> {
    fn count(&self) -> usize {
        1
    }
    fn render_str(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut String,
    ) -> Result<(), PyException> {
        *at += 1;
        match self {
            Some(v) => v.percent_render(spec, out),
            // str(None) is "None"; repr(None) is "None"; %d on None is a
            // TypeError, exactly as CPython's engine reports it.
            None => percent_render_none(spec, out),
        }
    }
    fn render_bytes(
        &self,
        at: &mut usize,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        *at += 1;
        match self {
            Some(v) => v.percent_render_bytes(spec, out),
            None => {
                let mut s = String::new();
                percent_render_none(spec, &mut s)?;
                out.extend_from_slice(s.as_bytes());
                Ok(())
            }
        }
    }
}

fn percent_render_none(spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
    match spec.code {
        PercentCode::S | PercentCode::R | PercentCode::A => {
            out.push_str("None");
            Ok(())
        }
        _ => Err(PyException::new(
            "TypeError",
            "%d format: a number is required, not NoneType",
        )),
    }
}

/// The str-mode formatting engine: walk `fmt`, rendering every
/// conversion against the values.
pub(crate) fn py_format_str(fmt: &[u8], values: &impl PyFormatRhs) -> Result<String, PyException> {
    let mut out = String::new();
    let mut at = 0usize;
    let mut pos = 0usize;
    while pos < fmt.len() {
        if fmt[pos] != b'%' {
            // The format string is valid UTF-8 (a str); copy the char.
            let ch_len = utf8_char_len(fmt[pos]);
            let end = (pos + ch_len).min(fmt.len());
            out.push_str(core::str::from_utf8(&fmt[pos..end]).unwrap_or(""));
            pos = end;
            continue;
        }
        pos += 1;
        if fmt.get(pos) == Some(&b'%') {
            out.push('%');
            pos += 1;
            continue;
        }
        let mut spec = parse_percent_spec(fmt, &mut pos)?;
        // A `%(name)s` mapping spec addresses the dict by key — it never
        // touches the positional index.
        if let Some(key) = spec.key.clone() {
            values.render_keyed(&key, &spec, &mut out)?;
            continue;
        }
        // `*` width/precision consume values, in order.
        if spec.star_width {
            let w = take_star_int(values, &mut at)?;
            spec.width = Some(w);
        }
        if spec.star_precision {
            let p = take_star_int(values, &mut at)?;
            spec.precision = Some(p);
        }
        if at >= values.count() {
            return Err(PyException::new(
                "TypeError",
                "not enough arguments for format string",
            ));
        }
        values.render_at(&mut at, &spec, &mut out)?;
    }
    if at < values.count() {
        return Err(PyException::new(
            "TypeError",
            "not all arguments converted during string formatting",
        ));
    }
    Ok(out)
}

/// The bytes-mode engine: identical parsing; the output is bytes and
/// `%b` accepts a bytes value (`%s` still takes a str, like CPython).
pub(crate) fn py_format_bytes(
    fmt: &[u8],
    values: &impl PyFormatRhs,
) -> Result<Vec<u8>, PyException> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut pos = 0usize;
    while pos < fmt.len() {
        if fmt[pos] != b'%' {
            out.push(fmt[pos]);
            pos += 1;
            continue;
        }
        pos += 1;
        if fmt.get(pos) == Some(&b'%') {
            out.push(b'%');
            pos += 1;
            continue;
        }
        let mut spec = parse_percent_spec(fmt, &mut pos)?;
        if let Some(key) = spec.key.clone() {
            // The mapping form renders through the str-mode path (the
            // values are looked up by key).
            let mut s = String::new();
            values.render_keyed(&key, &spec, &mut s)?;
            out.extend_from_slice(s.as_bytes());
            continue;
        }
        if spec.star_width {
            let w = take_star_int(values, &mut at)?;
            spec.width = Some(w);
        }
        if spec.star_precision {
            let p = take_star_int(values, &mut at)?;
            spec.precision = Some(p);
        }
        if at >= values.count() {
            return Err(PyException::new(
                "TypeError",
                "not enough arguments for format string",
            ));
        }
        values.render_at_bytes(&mut at, &spec, &mut out)?;
    }
    if at < values.count() {
        return Err(PyException::new(
            "TypeError",
            "not all arguments converted during string formatting",
        ));
    }
    Ok(out)
}

fn take_star_int(values: &impl PyFormatRhs, at: &mut usize) -> Result<i64, PyException> {
    if *at >= values.count() {
        return Err(PyException::new(
            "TypeError",
            "not enough arguments for format string",
        ));
    }
    let mut tmp = String::new();
    let spec = PercentSpec {
        key: None,
        code: PercentCode::D,
        minus: false,
        plus: false,
        zero: false,
        space: false,
        alt: false,
        width: None,
        precision: None,
        star_width: false,
        star_precision: false,
    };
    values.render_at(at, &spec, &mut tmp)?;
    tmp.parse::<i64>().map_err(|_| {
        PyException::new(
            "TypeError",
            format!("an integer is required, got type '{}'", tmp),
        )
    })
}

fn utf8_char_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

fn parse_percent_spec(fmt: &[u8], pos: &mut usize) -> Result<PercentSpec, PyException> {
    let mut spec = PercentSpec {
        key: None,
        code: PercentCode::S,
        minus: false,
        plus: false,
        zero: false,
        space: false,
        alt: false,
        width: None,
        precision: None,
        star_width: false,
        star_precision: false,
    };
    // The mapping form: `%(name)s` — the key names a dict entry.
    if fmt.get(*pos) == Some(&b'(') {
        let close = fmt[*pos..]
            .iter()
            .position(|c| *c == b')')
            .map(|i| *pos + i);
        let Some(close) = close else {
            return Err(PyException::new(
                "ValueError",
                "incomplete format key",
            ));
        };
        let key = core::str::from_utf8(&fmt[*pos + 1..close])
            .map_err(|_| PyException::new("ValueError", "incomplete format key"))?
            .to_string();
        spec.key = Some(key);
        *pos = close + 1;
    }
    loop {
        match fmt.get(*pos).copied() {
            Some(b'-') => {
                spec.minus = true;
                *pos += 1;
            }
            Some(b'+') => {
                spec.plus = true;
                *pos += 1;
            }
            Some(b'0') => {
                spec.zero = true;
                *pos += 1;
            }
            Some(b' ') => {
                spec.space = true;
                *pos += 1;
            }
            Some(b'#') => {
                spec.alt = true;
                *pos += 1;
            }
            _ => break,
        }
    }
    if fmt.get(*pos) == Some(&b'*') {
        spec.star_width = true;
        *pos += 1;
    } else {
        spec.width = read_digits(fmt, pos);
    }
    if fmt.get(*pos) == Some(&b'.') {
        *pos += 1;
        if fmt.get(*pos) == Some(&b'*') {
            spec.star_precision = true;
            *pos += 1;
        } else {
            spec.precision = read_digits(fmt, pos);
        }
    }
    let code = match fmt.get(*pos).copied() {
        Some(b'd') | Some(b'i') => PercentCode::D,
        Some(b'o') => PercentCode::O,
        Some(b'u') => PercentCode::U,
        Some(b'x') => PercentCode::X,
        Some(b'X') => PercentCode::BigX,
        Some(b'e') => PercentCode::E,
        Some(b'E') => PercentCode::BigE,
        Some(b'f') | Some(b'F') => PercentCode::F,
        Some(b'g') => PercentCode::G,
        Some(b'G') => PercentCode::BigG,
        Some(b'c') => PercentCode::C,
        Some(b'r') => PercentCode::R,
        Some(b's') => PercentCode::S,
        Some(b'a') => PercentCode::A,
        Some(b'b') => PercentCode::B,
        Some(other) => {
            return Err(PyException::new(
                "ValueError",
                format!(
                    "unsupported format character '{}' (0x{:x}) at index {}",
                    other as char,
                    other,
                    *pos
                ),
            ));
        }
        None => {
            return Err(PyException::new(
                "ValueError",
                "incomplete format",
            ));
        }
    };
    *pos += 1;
    spec.code = code;
    Ok(spec)
}

fn read_digits(fmt: &[u8], pos: &mut usize) -> Option<i64> {
    let start = *pos;
    while fmt.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
        *pos += 1;
    }
    if *pos == start {
        None
    } else {
        core::str::from_utf8(&fmt[start..*pos])
            .ok()
            .and_then(|s| s.parse().ok())
    }
}

// ---------------------------------------------------------------------------
// Per-code renderers
// ---------------------------------------------------------------------------

fn percent_render_int(v: i64, spec: &PercentSpec) -> String {
    // %c with an int is chr(v) — CPython's conversion.
    if spec.code == PercentCode::C {
        return char::from_u32(v as u32)
            .map(|c| c.to_string())
            .unwrap_or_else(|| {
                panic!("%c: chr() arg out of range")
            });
    }
    // Rust's {:x}/{:o} of a negative i64 render the two's-complement
    // bit pattern; CPython renders the signed magnitude ("-ff").
    let s = if v < 0 && matches!(spec.code, PercentCode::X | PercentCode::BigX | PercentCode::O) {
        let mag = v.unsigned_abs();
        match spec.code {
            PercentCode::X => format!("-{:x}", mag),
            PercentCode::BigX => format!("-{:X}", mag),
            _ => format!("-{:o}", mag),
        }
    } else {
        match spec.code {
            PercentCode::X => format!("{:x}", v),
            PercentCode::BigX => format!("{:X}", v),
            PercentCode::O => format!("{:o}", v),
            _ => v.to_string(),
        }
    };
    // %d/%x/%o precision: the minimum digit count (zero-pad).
    if let Some(p) = spec.precision {
        let p = p.max(0) as usize;
        if s.len() < p && !s.starts_with('-') {
            return format!("{:0>width$}", s, width = p);
        }
        if s.starts_with('-') && s.len() - 1 < p {
            return format!("-{:0>width$}", &s[1..], width = p);
        }
    }
    s
}

fn percent_render_float(f: f64, spec: &PercentSpec) -> Result<String, PyException> {
    if !f.is_finite() {
        // %f of inf/nan renders like repr() in CPython: "inf"/"nan".
        return Ok(if f.is_nan() {
            "nan".to_string()
        } else if f.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        });
    }
    match spec.code {
        // %d/%x/%o with a float truncate toward zero, like CPython's
        // int() conversion.
        PercentCode::D | PercentCode::U => {
            let tmp_spec = PercentSpec { key: None, code: PercentCode::D, ..*spec };
            Ok(percent_render_int(f as i64, &tmp_spec))
        }
        PercentCode::X | PercentCode::BigX | PercentCode::O => {
            let tmp_spec = PercentSpec { key: None, code: spec.code, ..*spec };
            Ok(percent_render_int(f as i64, &tmp_spec))
        }
        PercentCode::E | PercentCode::BigE => {
            let prec = spec.precision.unwrap_or(6).max(0) as usize;
            let upper = spec.code == PercentCode::BigE;
            let s = format!("{:.*e}", prec, f);
            // Rust's {:e} gives "5e0"; CPython's %e is "5.000000e+00".
            Ok(fixup_exponent(&s, upper))
        }
        PercentCode::F => {
            let prec = spec.precision.unwrap_or(6).max(0) as usize;
            Ok(format!("{:.*}", prec, f))
        }
        PercentCode::G | PercentCode::BigG => percent_render_general(f, spec),
        _ => Ok(format!("{}", f)),
    }
}

fn fixup_exponent(s: &str, upper: bool) -> String {
    let (mantissa, exp) = s.split_once('e').unwrap_or((s, "0"));
    let (exp_sign, exp_digits) = match exp.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("+", exp),
    };
    let exp_padded = format!("{:0>2}", exp_digits);
    let e = if upper { "E" } else { "e" };
    format!("{}{}{}{}", mantissa, e, exp_sign, exp_padded)
}

fn percent_render_general(f: f64, spec: &PercentSpec) -> Result<String, PyException> {
    let prec = spec.precision.unwrap_or(6).max(1) as usize;
    let upper = spec.code == PercentCode::BigG;
    let abs = f.abs();
    let exp = if abs == 0.0 {
        0
    } else {
        crate::flt::floor(crate::flt::log10(abs)) as i32
    };
    if exp < -4 || exp >= prec as i32 {
        // %e form with precision-1 digits after the point.
        let s = format!("{:.*e}", prec.saturating_sub(1), f);
        let fixed = fixup_exponent(&s, upper);
        Ok(percent_g_trim(&fixed, upper))
    } else {
        let decimals = (prec as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, f);
        Ok(percent_g_trim(&s, upper))
    }
}

/// %g/%G strip trailing zeros from the fractional part, like CPython.
fn percent_g_trim(s: &str, upper: bool) -> String {
    let (mantissa, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], Some(&s[i..])),
        None => (s, None),
    };
    let mantissa = if mantissa.contains('.') {
        let trimmed = mantissa.trim_end_matches('0');
        let trimmed = trimmed.trim_end_matches('.');
        if trimmed.is_empty() || trimmed == "-" {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        mantissa.to_string()
    };
    match exp {
        Some(e) => {
            let e = if upper { e.to_uppercase() } else { e.to_string() };
            format!("{}{}", mantissa, e)
        }
        None => mantissa,
    }
}

fn percent_render_str(s: &str, spec: &PercentSpec) -> Result<String, PyException> {
    match spec.code {
        PercentCode::S => Ok(s.to_string()),
        PercentCode::R => Ok(crate::repr(&s)),
        PercentCode::A => Ok(ascii_of(s)),
        PercentCode::C => {
            if s.chars().count() == 1 {
                Ok(s.to_string())
            } else {
                Err(PyException::new(
                    "TypeError",
                    "%c requires int or char",
                ))
            }
        }
        // A string value for a numeric code is a TypeError, as CPython
        // reports it.
        PercentCode::D | PercentCode::U | PercentCode::X | PercentCode::BigX
        | PercentCode::O | PercentCode::E | PercentCode::BigE | PercentCode::F
        | PercentCode::G | PercentCode::BigG => Err(PyException::new(
            "TypeError",
            "%d format: a real number is required, not str",
        )),
        PercentCode::B => Err(PyException::new(
            "ValueError",
            "unsupported format character 'b' (0x62) at index 0",
        )),
    }
}

fn percent_render_bytes(b: &[u8], spec: &PercentSpec) -> Result<String, PyException> {
    match spec.code {
        // %s on a bytes value is its display (`b'...'`) — the str-mode
        // engine; the BYTES engine's %b takes raw bytes.
        PercentCode::S => Ok(crate::py_bytes_repr(b)),
        PercentCode::R => Ok(crate::py_bytes_repr(b)),
        PercentCode::B => Err(PyException::new(
            "ValueError",
            "unsupported format character 'b' (0x62) at index 0",
        )),
        PercentCode::D | PercentCode::U | PercentCode::X | PercentCode::BigX | PercentCode::O => {
            // CPython: "%d" % b"5" → TypeError (bytes has no __int__ in
            // %-formatting).
            Err(PyException::new(
                "TypeError",
                "%d format: a real number is required, not bytes",
            ))
        }
        _ => Err(PyException::new(
            "TypeError",
            "%b requires a bytes-like object, or an object that implements __bytes__, not 'str'",
        )),
    }
}

fn percent_render_pyvalue(pv: &PyValue, spec: &PercentSpec) -> Result<String, PyException> {
    match spec.code {
        PercentCode::S | PercentCode::R | PercentCode::A => Ok(crate::py_display(pv)),
        PercentCode::D | PercentCode::U | PercentCode::X | PercentCode::BigX | PercentCode::O => {
            let n = pv.as_int().ok_or_else(|| {
                PyException::new(
                    "TypeError",
                    "%d format: a real number is required, not a boxed value",
                )
            })?;
            let tmp_spec = PercentSpec {
                key: None,
                code: spec.code,
                minus: false,
                plus: false,
                zero: false,
                space: false,
                alt: false,
                width: None,
                precision: spec.precision,
                star_width: false,
                star_precision: false,
            };
            Ok(percent_render_int(n, &tmp_spec))
        }
        _ => Ok(crate::py_display(pv)),
    }
}

fn ascii_of(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\'' => out.push_str("\\'"),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_ascii() && !c.is_control() => out.push(c),
            c => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("\\x{:02x}", b));
                }
            }
        }
    }
    out.push('\'');
    out
}

// ---------------------------------------------------------------------------
// Leaf value rendering
// ---------------------------------------------------------------------------

impl PercentValue for i64 {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_int(*self, spec);
        let s = percent_apply(spec, s, true)?;
        out.push_str(&s);
        Ok(())
    }
}

impl PercentValue for bool {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = match spec.code {
            // str(True) is "True", repr(True) is "True".
            PercentCode::S | PercentCode::R => {
                if *self { "True".to_string() } else { "False".to_string() }
            }
            _ => percent_render_int(if *self { 1 } else { 0 }, spec),
        };
        let s = percent_apply(spec, s, true)?;
        out.push_str(&s);
        Ok(())
    }
}

impl PercentValue for f64 {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_float(*self, spec)?;
        let s = percent_apply(spec, s, true)?;
        out.push_str(&s);
        Ok(())
    }
}

impl PercentValue for String {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_str(self.as_str(), spec)?;
        let s = percent_apply(spec, s, false)?;
        out.push_str(&s);
        Ok(())
    }
    fn percent_render_bytes(
        &self,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        match spec.code {
            // CPython: bytes-formatting's %s/%b take a bytes-like object.
            PercentCode::S | PercentCode::B => Err(PyException::new(
                "TypeError",
                "%b requires a bytes-like object, or an object that implements __bytes__, not 'str'",
            )),
            _ => {
                let s = percent_render_str(self.as_str(), spec)?;
                let s = percent_apply(spec, s, false)?;
                out.extend_from_slice(s.as_bytes());
                Ok(())
            }
        }
    }
}

impl PercentValue for &str {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_str(self, spec)?;
        let s = percent_apply(spec, s, false)?;
        out.push_str(&s);
        Ok(())
    }
}

impl PercentValue for Vec<u8> {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_bytes(self, spec)?;
        let s = percent_apply(spec, s, false)?;
        out.push_str(&s);
        Ok(())
    }
    fn percent_render_bytes(
        &self,
        spec: &PercentSpec,
        out: &mut Vec<u8>,
    ) -> Result<(), PyException> {
        match spec.code {
            // In bytes-formatting, %s and %b both take a bytes-like
            // value, copied raw (verified against CPython 3.14).
            PercentCode::S | PercentCode::B => {
                let s = percent_apply(spec, String::new(), false)?;
                debug_assert!(s.is_empty());
                out.extend_from_slice(self);
                Ok(())
            }
            PercentCode::R => {
                let s = crate::py_bytes_repr(self);
                let s = percent_apply(spec, s, false)?;
                out.extend_from_slice(s.as_bytes());
                Ok(())
            }
            _ => {
                let s = percent_render_bytes(self, spec)?;
                let s = percent_apply(spec, s, false)?;
                out.extend_from_slice(s.as_bytes());
                Ok(())
            }
        }
    }
}

impl PercentValue for PyException {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_str(&format!("{}", self), spec)?;
        let s = percent_apply(spec, s, false)?;
        out.push_str(&s);
        Ok(())
    }
}

impl PercentValue for PyValue {
    fn percent_render(&self, spec: &PercentSpec, out: &mut String) -> Result<(), PyException> {
        let s = percent_render_pyvalue(self, spec)?;
        let s = percent_apply(spec, s, false)?;
        out.push_str(&s);
        Ok(())
    }
}

/// The code-independent post-processing shared by every leaf: precision
/// truncation (strings), the `#` alternate form, the `+`/space sign
/// flags, and width padding.
fn percent_apply(spec: &PercentSpec, mut rendered: String, is_numeric: bool) -> Result<String, PyException> {
    if let Some(p) = spec.precision {
        match spec.code {
            PercentCode::S | PercentCode::R | PercentCode::A => {
                let p = p.max(0) as usize;
                if rendered.len() > p {
                    rendered.truncate(p);
                }
            }
            _ => {}
        }
    }
    if spec.alt {
        match spec.code {
            PercentCode::X if !rendered.starts_with("0x") && rendered != "0" => {
                rendered = format!("0x{}", rendered);
            }
            PercentCode::BigX if !rendered.starts_with("0X") && rendered != "0" => {
                rendered = format!("0X{}", rendered);
            }
            PercentCode::O if !rendered.starts_with("0o") && rendered != "0" => {
                rendered = format!("0o{}", rendered);
            }
            _ => {}
        }
    }
    if spec.plus
        && is_numeric
        && !rendered.starts_with('-')
        && !rendered.starts_with('+')
    {
        rendered = format!("+{}", rendered);
    }
    if spec.space
        && is_numeric
        && !rendered.starts_with('-')
        && !rendered.starts_with('+')
    {
        rendered = format!(" {}", rendered);
    }
    let width = spec.width.unwrap_or(0).max(0) as usize;
    if rendered.len() < width {
        if spec.minus {
            rendered = format!("{:<width$}", rendered);
        } else if spec.zero && is_numeric && !rendered.starts_with('-') {
            rendered = format!("{:0>width$}", rendered);
        } else if spec.zero && is_numeric {
            let (sign, rest) = rendered.split_at(1);
            rendered = format!("{}{:0>width$}", sign, rest, width = width.saturating_sub(1));
        } else {
            rendered = format!("{:>width$}", rendered);
        }
    }
    Ok(rendered)
}
