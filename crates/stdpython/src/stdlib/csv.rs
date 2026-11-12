//! Python csv module implementation
//!
//! The reader, over a LIST of line strings (readlines()/split output) with
//! CPython's "excel" dialect by default: comma delimiter, double-quote
//! quoting with "" escapes, quoted fields spanning list elements, and
//! whitespace preserved. The writer needs a file-object surface and is
//! tracked separately.
//!
//! Dialect support (issue #369): a std-gated `Dialect` value type + a
//! process-global named registry (`register_dialect`/`get_dialect`/
//! `unregister_dialect`/`list_dialects`), mirroring CPython's module-level
//! registry. The alloc-tier `reader` is a pure function that takes the
//! `delimiter: u8` it splits on, so a caller resolves a dialect NAME to a
//! delimiter (via the std-gated `dialect_delimiter` helper) before invoking
//! it; the reader itself stays OS-free and builds on every tier.

use crate::PyException;
use alloc::string::String;
use alloc::vec::Vec;

/// csv.reader(lines, delimiter), materialized: one Vec<String> per
/// record. `delimiter` is the field separator byte (CPython's dialect
/// delimiter; the excel default is `,`). In unquoted context a trailing
/// \n, \r, or \r\n TERMINATES the record (so readlines() output, which
/// keeps its newlines, parses identically to newline-free split output);
/// inside quotes newlines are data. A quoted field that does not close
/// continues into the NEXT list element, as CPython's reader pulls further
/// lines from its iterator; an unterminated quote simply closes at end of
/// input, as in Python. A newline in unquoted context with more data after
/// it raises csv.Error with Python's message.
pub fn reader<S: AsRef<str>>(
    lines: &[S],
    delimiter: u8,
    no_quote: bool,
    escapechar: Option<u8>,
) -> Result<Vec<Vec<String>>, PyException> {
    #[derive(PartialEq)]
    enum State {
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }

    let newline_error = || {
        PyException::new(
            "csv.Error",
            "new-line character seen in unquoted field - do you need to open the file with newline=''?",
        )
    };

    // An escapechar was given: in QUOTE_NONE mode it escapes the
    // delimiter / quote / newline / itself so the following char stays
    // literal. When no escapechar is set, there is nothing to handle.
    let has_esc = escapechar.is_some();

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let mut row: Vec<String> = Vec::new();
        let mut field = String::new();
        let mut state = State::StartField;
        let mut any_content = false;
        let mut terminated = false;

        loop {
            let line = lines[i].as_ref();
            let mut chars = line.chars().peekable();
            while let Some(c) = chars.next() {
                if terminated {
                    // Data after an unquoted newline in the same element.
                    return Err(newline_error());
                }
                // A newline terminates the record in every state except
                // inside quotes, where it is data.
                if (c == '\n' || c == '\r') && state != State::InQuoted {
                    if c == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    if state == State::QuoteInQuoted {
                        state = State::InField;
                    }
                    terminated = true;
                    continue;
                }
                any_content = true;
                // In QUOTE_NONE mode an escapechar lets the following char
                // through literally (the escapechar is dropped), whether that
                // char is the delimiter, a quote, a newline, or the escapechar
                // itself.
                if no_quote && has_esc && c == (escapechar.unwrap() as char) {
                    if let Some(escaped) = chars.next() {
                        field.push(escaped);
                        continue;
                    }
                }
                match state {
                    State::StartField => match c {
                        '"' => {
                            if no_quote {
                                field.push('"');
                                state = State::InField;
                            } else {
                                state = State::InQuoted;
                            }
                        }
                        c if c == delimiter as char => {
                            row.push(core::mem::take(&mut field));
                        }
                        c => {
                            field.push(c);
                            state = State::InField;
                        }
                    },
                    State::InField => match c {
                        c if c == delimiter as char => {
                            row.push(core::mem::take(&mut field));
                            state = State::StartField;
                        }
                        // A quote mid-field is literal data ('a"b').
                        c => field.push(c),
                    },
                    State::InQuoted => match c {
                        '"' => state = State::QuoteInQuoted,
                        c => field.push(c),
                    },
                    State::QuoteInQuoted => match c {
                        // "" inside quotes is an escaped quote.
                        '"' => {
                            field.push('"');
                            state = State::InQuoted;
                        }
                        c if c == delimiter as char => {
                            row.push(core::mem::take(&mut field));
                            state = State::StartField;
                        }
                        // Data after the closing quote concatenates
                        // ('"a"b' -> 'ab'), like CPython.
                        c => {
                            field.push(c);
                            state = State::InField;
                        }
                    },
                }
            }
            if state == State::InQuoted && i + 1 < lines.len() {
                // The quoted field continues into the next element. Any
                // newline the element carried was consumed as DATA above
                // (we are inside quotes), exactly like CPython.
                i += 1;
                continue;
            }
            break;
        }

        // An empty line is an empty record ([]), not [""].
        if any_content {
            row.push(field);
        }
        rows.push(row);
        i += 1;
    }
    Ok(rows)
}

/// csv.Dialect: CPython's dialect descriptor. A value-style struct summing
/// the eight dialect parameters the reader/writer honour. The excel default
/// is comma-delimited, QUOTE_MINIMAL, `"` quotechar, "" quote-doubling,
/// \r\n lineterminator. Only available with the std feature (it backs the
/// process-global dialect registry, which itself needs std).
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct Dialect {
    pub delimiter: u8,
    pub doublequote: bool,
    pub escapechar: Option<u8>,
    pub lineterminator: String,
    pub quotechar: u8,
    pub quoting: i64,
    pub skipinitialspace: bool,
    pub strict: bool,
}

/// The csv module's QUOTE_* constants, mirroring CPython's integer values.
pub const QUOTE_MINIMAL: i64 = 0;
pub const QUOTE_ALL: i64 = 1;
pub const QUOTE_NONNUMERIC: i64 = 2;
pub const QUOTE_NONE: i64 = 3;

/// CPython's default "excel" dialect.
#[cfg(feature = "std")]
pub fn excel_dialect() -> Dialect {
    Dialect {
        delimiter: b',',
        doublequote: true,
        escapechar: None,
        lineterminator: "\r\n".to_string(),
        quotechar: b'"',
        quoting: QUOTE_MINIMAL,
        skipinitialspace: false,
        strict: false,
    }
}

/// The process-global dialect registry, buildable from the default on the
/// alloc tier. Pre-seeded with CPython's built-in dialects (`excel`,
/// `excel-tab`, `unix`).
#[cfg(feature = "std")]
pub fn default_dialects() -> alloc::collections::BTreeMap<String, Dialect> {
    let mut m = alloc::collections::BTreeMap::new();
    m.insert("excel".to_string(), excel_dialect());
    let mut tab = excel_dialect();
    tab.delimiter = b'\t';
    m.insert("excel-tab".to_string(), tab);
    let mut unix = excel_dialect();
    unix.lineterminator = "\n".to_string();
    // CPython's unix dialect quotes EVERY field (QUOTE_ALL), unlike excel's
    // QUOTE_MINIMAL — verified against python3 (`csv.get_dialect('unix')`).
    unix.quoting = QUOTE_ALL;
    m.insert("unix".to_string(), unix);
    m
}

#[cfg(feature = "std")]
use std::collections::BTreeMap;

#[cfg(feature = "std")]
static DIALECTS: std::sync::Mutex<Option<BTreeMap<String, Dialect>>> =
    std::sync::Mutex::new(None);

/// Lazy-init the registry on first use (so the static can be const).
#[cfg(feature = "std")]
fn registry() -> std::sync::MutexGuard<'static, Option<BTreeMap<String, Dialect>>> {
    let mut g = DIALECTS.lock().unwrap_or_else(|p| p.into_inner());
    if g.is_none() {
        *g = Some(default_dialects());
    }
    g
}

/// csv.register_dialect(name, dialect) / csv.register_dialect(name,
/// ***attributes) — store a dialect under a name. The kwarg form takes the
/// recognized dialect attribute names; an unknown/None attribute is loud.
#[cfg(feature = "std")]
pub fn register_dialect(
    name: &str,
    delimiter: Option<u8>,
    quotechar: Option<u8>,
    doublequote: Option<bool>,
    skipinitialspace: Option<bool>,
    lineterminator: Option<String>,
    quoting: Option<i64>,
    escapechar: Option<u8>,
    strict: Option<bool>,
) -> Result<(), PyException> {
    let mut d = excel_dialect();
    if let Some(v) = delimiter {
        d.delimiter = v;
    }
    if let Some(v) = quotechar {
        d.quotechar = v;
    }
    if let Some(v) = doublequote {
        d.doublequote = v;
    }
    if let Some(v) = skipinitialspace {
        d.skipinitialspace = v;
    }
    if let Some(v) = lineterminator {
        d.lineterminator = v;
    }
    if let Some(v) = quoting {
        d.quoting = v;
    }
    if let Some(v) = escapechar {
        d.escapechar = Some(v);
    }
    if let Some(v) = strict {
        d.strict = v;
    }
    if d.quoting != QUOTE_MINIMAL && d.quoting != QUOTE_ALL && d.quoting != QUOTE_NONE {
        return Err(PyException::new("TypeError", "quoting must be an integer"));
    }
    registry().as_mut().unwrap().insert(name.to_string(), d);
    Ok(())
}

/// csv.get_dialect(name): a COPY (value semantics — rython has no shared
/// references) of the named dialect, or csv.Error if unregistered.
#[cfg(feature = "std")]
pub fn get_dialect(name: &str) -> Result<Dialect, PyException> {
    registry().as_ref().unwrap().get(name).cloned().ok_or_else(|| {
        PyException::new("csv.Error", "unknown dialect")
    })
}

/// csv.get_dialect(name).delimiter — the byte the reader/writer split/join
/// on. An UNKNOWN name raises csv.Error, exactly as CPython's named-dialect
/// lookup does (never a silent default). Returns the byte for a registered
/// name.
#[cfg(feature = "std")]
pub fn dialect_delimiter(name: &str) -> Result<u8, PyException> {
    get_dialect(name).map(|d| d.delimiter)
}

/// csv.unregister_dialect(name).
#[cfg(feature = "std")]
pub fn unregister_dialect(name: &str) -> Result<(), PyException> {
    if registry().as_mut().unwrap().remove(name).is_none() {
        return Err(PyException::new("csv.Error", "unknown dialect"));
    }
    Ok(())
}

/// csv.list_dialects(): the registered names, sorted (registry is a
/// BTreeMap).
#[cfg(feature = "std")]
pub fn list_dialects() -> Vec<String> {
    registry().as_ref().unwrap().keys().cloned().collect()
}

/// csv.writer(f, lineterminator=..., quoting=..., escapechar=...) with
/// CPython's default "excel" dialect: comma delimiter, and by default
/// QUOTE_MINIMAL (a field is quoted only when it contains the delimiter, a
/// quote, or a newline), "" quote doubling, and — by default — \r\n as the
/// row terminator. The `lineterminator` (issue #369), `quoting` and
/// `escapechar` keywords (issue #369) are supported as value-shaped seams,
/// exactly as CPython's writer accepts them. CPython's quote MODES:
/// minimal quotes only when needed; all quotes every field; none never
/// quotes and escapes the delimiter/quote/newline with `escapechar`. Rows
/// stringify their elements through PyDisplay — Python's writer calls str()
/// — so ints, floats, and bools render exactly as Python prints them
/// (True, 2.5, 1e+16). Only available with the std feature: it writes
/// through PyFile.
#[cfg(feature = "std")]
pub struct Writer<'a> {
    file: &'a mut crate::PyFile,
    lineterminator: String,
    delimiter: u8,
    all_quote: bool,
    escapechar: Option<u8>,
}

#[cfg(feature = "std")]
pub fn writer(
    file: &mut crate::PyFile,
    lineterminator: String,
    delimiter: u8,
    all_quote: bool,
    escapechar: Option<u8>,
) -> Writer<'_> {
    Writer { file, lineterminator, delimiter, all_quote, escapechar }
}

#[cfg(feature = "std")]
impl Writer<'_> {
    /// Emit one field: QUOTE_MINIMAL (quote only when needed) or QUOTE_ALL
    /// (always quote) quote the field; QUOTE_NONE never quotes and instead
    /// escapes the delimiter/quote/CR/LF with the escapechar.
    fn needs_escape(&self, text: &str) -> bool {
        text.contains(self.delimiter as char)
            || text.contains('"')
            || text.contains('\n')
            || text.contains('\r')
    }

    fn write_field(&self, out: &mut String, text: &str) {
        let needs_escape = self.needs_escape(text);
        if self.all_quote {
            out.push('"');
            out.push_str(&text.replace('"', "\"\""));
            out.push('"');
        } else if needs_escape {
            if let Some(esc) = self.escapechar {
                let mut escaped = String::new();
                for c in text.chars() {
                    if c == self.delimiter as char || c == '"' || c == '\n' || c == '\r' {
                        escaped.push(esc as char);
                        escaped.push(c);
                    } else {
                        escaped.push(c);
                    }
                }
                out.push_str(&escaped);
            } else {
                out.push('"');
                out.push_str(&text.replace('"', "\"\""));
                out.push('"');
            }
        } else {
            out.push_str(text);
        }
    }

    pub fn writerow<T: crate::PyDisplay>(&mut self, row: &[T]) -> Result<(), PyException> {
        let mut out = String::new();
        for (i, field) in row.iter().enumerate() {
            if i > 0 {
                out.push(self.delimiter as char);
            }
            let text = field.py_display();
            self.write_field(&mut out, text.as_ref());
        }
        out.push_str(&self.lineterminator);
        self.file.write(out)?;
        Ok(())
    }

    pub fn writerows<T: crate::PyDisplay>(&mut self, rows: &[Vec<T>]) -> Result<(), PyException> {
        for row in rows {
            self.writerow(row)?;
        }
        Ok(())
    }
}
