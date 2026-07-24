//! Python csv module implementation
//!
//! The reader, over a LIST of line strings (readlines()/split output) with
//! CPython's default "excel" dialect: comma delimiter, double-quote
//! quoting with "" escapes, quoted fields spanning list elements, and
//! whitespace preserved. The writer needs a file-object surface and is
//! tracked separately.

use crate::PyException;
use alloc::string::String;
use alloc::vec::Vec;

/// csv.reader(lines), materialized: one Vec<String> per record. In
/// unquoted context a trailing \n, \r, or \r\n TERMINATES the record
/// (so readlines() output, which keeps its newlines, parses identically
/// to newline-free split output); inside quotes newlines are data. A
/// quoted field that does not close continues into the NEXT list
/// element, as CPython's reader pulls further lines from its iterator;
/// an unterminated quote simply closes at end of input, as in Python. A
/// newline in unquoted context with more data after it raises csv.Error
/// with Python's message.
pub fn reader<S: AsRef<str>>(lines: &[S]) -> Result<Vec<Vec<String>>, PyException> {
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
            "new-line character seen in unquoted field - do you need to open              the file with newline=''?",
        )
    };

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
                match state {
                    State::StartField => match c {
                        '"' => state = State::InQuoted,
                        ',' => row.push(core::mem::take(&mut field)),
                        c => {
                            field.push(c);
                            state = State::InField;
                        }
                    },
                    State::InField => match c {
                        ',' => {
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
                        ',' => {
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
