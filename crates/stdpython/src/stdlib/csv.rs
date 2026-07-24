//! Python csv module implementation
//!
//! The reader, over a LIST of line strings (readlines()/split output) with
//! CPython's default "excel" dialect: comma delimiter, double-quote
//! quoting with "" escapes, quoted fields spanning list elements, and
//! whitespace preserved. The writer needs a file-object surface and is
//! tracked separately.

use alloc::string::String;
use alloc::vec::Vec;

/// csv.reader(lines), materialized: one Vec<String> per record. A quoted
/// field that does not close continues into the NEXT list element, as
/// CPython's reader pulls further lines from its iterator; an
/// unterminated quote simply closes at end of input, as in Python.
pub fn reader<S: AsRef<str>>(lines: &[S]) -> Vec<Vec<String>> {
    #[derive(PartialEq)]
    enum State {
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let mut row: Vec<String> = Vec::new();
        let mut field = String::new();
        let mut state = State::StartField;
        let mut any_content = false;

        loop {
            let line = lines[i].as_ref();
            for c in line.chars() {
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
                // The quoted field continues into the next element. List
                // elements carry no newline of their own, so nothing is
                // inserted — exactly CPython's behavior over a list.
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
    rows
}
