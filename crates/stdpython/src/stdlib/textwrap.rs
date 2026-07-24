//! Python textwrap module implementation
//!
//! dedent() and indent() for now; wrap()/fill() have a large option
//! surface and are tracked separately.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

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
