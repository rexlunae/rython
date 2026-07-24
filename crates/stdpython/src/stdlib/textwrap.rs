//! Python textwrap module implementation
//!
//! dedent(), indent(), and wrap()/fill() with CPython's DEFAULT settings
//! (expand_tabs, replace_whitespace, drop_whitespace, break_long_words,
//! break_on_hyphens, no indents). The word splitter is a hand port of
//! CPython's wordsep_re — the original uses lookarounds the regex crate
//! doesn't support. Non-default options are not accepted (loudly).

use crate::PyException;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const WS: &str = " \t\n\x0b\x0c\r";

fn is_ws(c: char) -> bool {
    WS.contains(c)
}

// The regex classes: \w, [^\d\W] (letters + underscore), [\w!"'&.,?].
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_letter(c: char) -> bool {
    (c.is_alphabetic() || c == '_') && !c.is_numeric()
}

fn is_word_punct(c: char) -> bool {
    is_word(c) || "!\"'&.,?".contains(c)
}

/// str.expandtabs(8): column-aware, columns reset at \n and \r.
fn expand_tabs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut col = 0usize;
    for c in text.chars() {
        match c {
            '\t' => {
                let spaces = 8 - col % 8;
                for _ in 0..spaces {
                    out.push(' ');
                }
                col += spaces;
            }
            '\n' | '\r' => {
                out.push(c);
                col = 0;
            }
            c => {
                out.push(c);
                col += 1;
            }
        }
    }
    out
}

/// CPython's _munge_whitespace with the default settings: expand tabs,
/// then every whitespace character becomes a plain space.
fn munge_whitespace(text: &str) -> String {
    expand_tabs(text)
        .chars()
        .map(|c| if is_ws(c) { ' ' } else { c })
        .collect()
}

/// Hand port of wordsep_re's chunking: whitespace runs, em-dashes
/// between words, and words split after acceptable hyphens.
fn split_chunks(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut chunks = Vec::new();
    let mut i = 0;
    // The em-dash alternative: 2+ dashes at `at`, preceded by a
    // word-punct character and followed by a word character.
    let em_dash_at = |at: usize| -> Option<usize> {
        if at == 0 || chars[at] != '-' || !is_word_punct(chars[at - 1]) {
            return None;
        }
        let mut j = at;
        while j < n && chars[j] == '-' {
            j += 1;
        }
        if j - at >= 2 && j < n && is_word(chars[j]) {
            Some(j)
        } else {
            None
        }
    };
    while i < n {
        if is_ws(chars[i]) {
            let start = i;
            while i < n && is_ws(chars[i]) {
                i += 1;
            }
            chunks.push(chars[start..i].iter().collect());
            continue;
        }
        if let Some(end) = em_dash_at(i) {
            chunks.push(chars[i..end].iter().collect());
            i = end;
            continue;
        }
        // A word: lazily consume until the earliest acceptable boundary.
        let start = i;
        let mut j = i + 1;
        loop {
            if j >= n || is_ws(chars[j]) {
                break;
            }
            // End before an em-dash chunk.
            if em_dash_at(j).is_some() {
                break;
            }
            // Split after a hyphen: lookbehind two letters + '-' or
            // letter '-' letter '-', lookahead letter ('-')? letter.
            if chars[j - 1] == '-' {
                let lb2 = j >= 3 && is_letter(chars[j - 3]) && is_letter(chars[j - 2]);
                let lb4 = j >= 4
                    && is_letter(chars[j - 4])
                    && chars[j - 3] == '-'
                    && is_letter(chars[j - 2]);
                let la = j < n
                    && is_letter(chars[j])
                    && (j + 1 < n && is_letter(chars[j + 1])
                        || (j + 2 < n && chars[j + 1] == '-' && is_letter(chars[j + 2])));
                if (lb2 || lb4) && la {
                    break;
                }
            }
            j += 1;
        }
        chunks.push(chars[start..j].iter().collect());
        i = j;
    }
    chunks
}

fn is_ws_chunk(chunk: &str) -> bool {
    chunk.chars().all(char::is_whitespace)
}

/// CPython's _handle_long_word with the default settings.
fn handle_long_word(
    rest: &mut Vec<Vec<char>>,
    cur_line: &mut Vec<String>,
    cur_len: usize,
    width: usize,
) {
    let space_left = if width < 1 { 1 } else { width.saturating_sub(cur_len) };
    // break_long_words: chop, preferring the last hyphen in the window.
    let chunk = rest.last().expect("caller checked").clone();
    let mut end = space_left;
    if chunk.len() > space_left {
        let hyphen = chunk[..space_left.min(chunk.len())]
            .iter()
            .rposition(|&c| c == '-');
        if let Some(h) = hyphen {
            if h > 0 && chunk[..h].iter().any(|&c| c != '-') {
                end = h + 1;
            }
        }
    }
    let end = end.min(chunk.len());
    if end > 0 {
        cur_line.push(chunk[..end].iter().collect());
        *rest.last_mut().unwrap() = chunk[end..].to_vec();
    }
}

/// textwrap.wrap(text, width=70), default settings. Raises ValueError on
/// a non-positive width, as Python does.
pub fn wrap<S: AsRef<str> + ?Sized>(text: &S, width: i64) -> Result<Vec<String>, PyException> {
    if width <= 0 {
        return Err(crate::value_error(format!(
            "invalid width {} (must be > 0)",
            width
        )));
    }
    let width = width as usize;
    let munged = munge_whitespace(text.as_ref());
    let mut chunks: Vec<Vec<char>> = split_chunks(&munged)
        .into_iter()
        .map(|c| c.chars().collect())
        .collect();
    chunks.reverse();

    let mut lines: Vec<String> = Vec::new();
    while !chunks.is_empty() {
        let mut cur_line: Vec<String> = Vec::new();
        let mut cur_len = 0usize;
        // drop_whitespace: leading whitespace goes, except on line one.
        if !lines.is_empty() {
            if let Some(last) = chunks.last() {
                if last.iter().all(|c| c.is_whitespace()) {
                    chunks.pop();
                }
            }
        }
        while let Some(chunk) = chunks.last() {
            let l = chunk.len();
            if cur_len + l <= width {
                cur_line.push(chunks.pop().unwrap().iter().collect());
                cur_len += l;
            } else {
                break;
            }
        }
        if chunks.last().is_some_and(|c| c.len() > width) {
            handle_long_word(&mut chunks, &mut cur_line, cur_len, width);
            cur_len = cur_line.iter().map(|c| c.chars().count()).sum();
        }
        // drop_whitespace: trailing whitespace goes.
        if cur_line.last().is_some_and(|c| is_ws_chunk(c)) {
            let dropped = cur_line.pop().unwrap();
            cur_len = cur_len.saturating_sub(dropped.chars().count());
        }
        let _ = cur_len;
        if !cur_line.is_empty() {
            lines.push(cur_line.concat());
        }
    }
    Ok(lines)
}

/// textwrap.fill(text, width=70)
pub fn fill<S: AsRef<str> + ?Sized>(text: &S, width: i64) -> Result<String, PyException> {
    Ok(wrap(text, width)?.join("\n"))
}

fn is_ws_only(line: &str) -> bool {
    !line.is_empty() && line.chars().all(|c| c == ' ' || c == '\t')
}

/// textwrap.dedent(text): remove the longest common leading whitespace
/// from all non-blank lines. Like Python, lines consisting solely of
/// whitespace are normalized to empty and ignored when computing the
/// margin.
pub fn dedent<S: AsRef<str> + ?Sized>(text: &S) -> String {
    let text = text.as_ref();
    let lines: Vec<&str> = text.split('\n').collect();

    // The margin is the longest common prefix of spaces/tabs over
    // non-blank lines, compared character-for-character like CPython
    // (a tab and a space never match).
    let mut margin: Option<&str> = None;
    for line in &lines {
        if line.is_empty() || is_ws_only(line) {
            continue;
        }
        let indent_len = line.len() - line.trim_start_matches([' ', '\t']).len();
        let indent = &line[..indent_len];
        margin = Some(match margin {
            None => indent,
            Some(current) => {
                let common = current
                    .chars()
                    .zip(indent.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                &current[..common]
            }
        });
    }
    let margin_len = margin.map_or(0, str::len);

    let out: Vec<String> = lines
        .iter()
        .map(|line| {
            if is_ws_only(line) {
                String::new()
            } else if line.len() >= margin_len {
                line[margin_len..].to_string()
            } else {
                line.to_string()
            }
        })
        .collect();
    out.join("\n")
}

/// textwrap.indent(text, prefix): prefix every line that contains more
/// than just whitespace, like Python's default predicate.
pub fn indent<S: AsRef<str> + ?Sized, P: AsRef<str> + ?Sized>(text: &S, prefix: &P) -> String {
    let text = text.as_ref();
    let prefix = prefix.as_ref();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let (line, remainder) = match rest.find('\n') {
            Some(i) => (&rest[..=i], &rest[i + 1..]),
            None => (rest, ""),
        };
        let body = line.strip_suffix('\n').unwrap_or(line);
        if !body.is_empty() && !body.chars().all(char::is_whitespace) {
            out.push_str(prefix);
        }
        out.push_str(line);
        rest = remainder;
    }
    out
}
