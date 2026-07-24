//! Python re module implementation
//!
//! Backed by the `regex` crate. Patterns it cannot compile (Python allows
//! backreferences and lookarounds; the regex crate does not) fail loudly
//! with a re.error-typed exception rather than approximating. Offsets are
//! CHARACTER offsets, as in Python, not the regex crate's byte offsets.

use crate::PyException;

/// A successful match: the Python Match object surface.
#[derive(Debug, Clone)]
pub struct PyMatch {
    /// Per-group (0 = whole match) participation: the matched text and
    /// its (start, end) in character offsets.
    groups: Vec<Option<(String, i64, i64)>>,
}

fn char_offset(text: &str, byte: usize) -> i64 {
    text[..byte].chars().count() as i64
}

fn make_match(text: &str, caps: &regex::Captures) -> PyMatch {
    let groups = (0..caps.len())
        .map(|i| {
            caps.get(i).map(|m| {
                (
                    m.as_str().to_string(),
                    char_offset(text, m.start()),
                    char_offset(text, m.end()),
                )
            })
        })
        .collect();
    PyMatch { groups }
}

impl PyMatch {
    fn group_entry(&self, i: i64) -> &(String, i64, i64) {
        if i < 0 || i as usize >= self.groups.len() {
            panic!("{}", PyException::new("IndexError", "no such group"));
        }
        self.groups[i as usize].as_ref().unwrap_or_else(|| {
            // Python returns None for a group that did not participate;
            // a typed String can't, so this fails loudly instead of
            // inventing a value.
            panic!(
                "{}",
                PyException::new(
                    "ValueError",
                    format!(
                        "group {} did not participate in the match (Python returns \
                         None here, which rython's typed lowering cannot)",
                        i
                    ),
                )
            );
        })
    }

    /// m.group(i); m.group() lowers to m.group(0).
    pub fn group(&self, i: i64) -> String {
        self.group_entry(i).0.clone()
    }

    /// m.groups(): every group from 1 up.
    pub fn groups(&self) -> Vec<String> {
        (1..self.groups.len() as i64)
            .map(|i| self.group_entry(i).0.clone())
            .collect()
    }

    /// m.start(), in character offsets.
    pub fn start(&self) -> i64 {
        self.group_entry(0).1
    }

    /// m.end(), in character offsets.
    pub fn end(&self) -> i64 {
        self.group_entry(0).2
    }

    /// m.span()
    pub fn span(&self) -> (i64, i64) {
        let e = self.group_entry(0);
        (e.1, e.2)
    }
}

/// A Match object is always truthy in Python (`if m:` tests presence
/// through the Option layer).
impl crate::Truthy for PyMatch {
    fn is_truthy(&self) -> bool {
        true
    }
}

/// The Match surface on Option<PyMatch>, so the Python idiom
/// `m = re.search(...); if m: m.group(1)` lowers directly: calling a
/// method on a missed match fails with Python's exact AttributeError.
pub trait PyMatchOps {
    fn group(&self, i: i64) -> String;
    fn groups(&self) -> Vec<String>;
    fn start(&self) -> i64;
    fn end(&self) -> i64;
    fn span(&self) -> (i64, i64);
}

fn none_match_panic(method: &str) -> ! {
    panic!(
        "{}",
        PyException::new(
            "AttributeError",
            format!("'NoneType' object has no attribute '{}'", method),
        )
    );
}

impl PyMatchOps for Option<PyMatch> {
    fn group(&self, i: i64) -> String {
        match self {
            Some(m) => m.group(i),
            None => none_match_panic("group"),
        }
    }
    fn groups(&self) -> Vec<String> {
        match self {
            Some(m) => m.groups(),
            None => none_match_panic("groups"),
        }
    }
    fn start(&self) -> i64 {
        match self {
            Some(m) => m.start(),
            None => none_match_panic("start"),
        }
    }
    fn end(&self) -> i64 {
        match self {
            Some(m) => m.end(),
            None => none_match_panic("end"),
        }
    }
    fn span(&self) -> (i64, i64) {
        match self {
            Some(m) => m.span(),
            None => none_match_panic("span"),
        }
    }
}

/// Compile with Python flag letters ("i", "m", "s") applied as an
/// inline group, which the regex crate shares with Python's syntax.
fn compile(pattern: &str, flags: &str) -> Result<regex::Regex, PyException> {
    let pattern = if flags.is_empty() {
        alloc::borrow::Cow::Borrowed(pattern)
    } else {
        alloc::borrow::Cow::Owned(format!("(?{}){}", flags, pattern))
    };
    let pattern: &str = &pattern;
    regex::Regex::new(pattern).map_err(|e| {
        PyException::new(
            "re.error",
            format!(
                "cannot compile pattern {:?}: {} (the regex engine does not \
                 support Python's backreferences or lookarounds)",
                pattern, e
            ),
        )
    })
}

/// re.search(pattern, string): the first match anywhere, or None.
pub fn search<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Option<PyMatch>, PyException> {
    let re = compile(pattern.as_ref(), flags)?;
    let text = string.as_ref();
    Ok(re.captures(text).map(|caps| make_match(text, &caps)))
}

/// re.match(pattern, string): anchored at the START of the string.
pub fn r#match<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Option<PyMatch>, PyException> {
    let re = compile(&format!(r"\A(?:{})", pattern.as_ref()), flags)?;
    let text = string.as_ref();
    Ok(re.captures(text).map(|caps| make_match(text, &caps)))
}

/// re.fullmatch(pattern, string): the whole string must match.
pub fn fullmatch<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Option<PyMatch>, PyException> {
    let re = compile(&format!(r"\A(?:{})\z", pattern.as_ref()), flags)?;
    let text = string.as_ref();
    Ok(re.captures(text).map(|caps| make_match(text, &caps)))
}

/// re.findall(pattern, string). Python's per-group-count result SHAPES
/// (strings for 0-1 groups, tuples beyond) can't share one Rust type:
/// 0 groups yields the matches, 1 group yields that group, and 2+ groups
/// is a loud error.
pub fn findall<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Vec<String>, PyException> {
    let re = compile(pattern.as_ref(), flags)?;
    let text = string.as_ref();
    match re.captures_len() {
        1 => Ok(re
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()),
        2 => Ok(re
            .captures_iter(text)
            .map(|caps| {
                caps.get(1)
                    .map(|g| g.as_str().to_string())
                    .unwrap_or_default()
            })
            .collect()),
        n => Err(PyException::new(
            "TypeError",
            format!(
                "findall() with {} capture groups returns tuples in Python, \
                 which rython does not support yet; use a single group",
                n - 1
            ),
        )),
    }
}

/// re.split(pattern, string). Like Python, capturing groups in the
/// pattern interleave the captured delimiter text into the result:
/// split(r"(\d)", "a1b") is ['a', '1', 'b']. A group that does NOT
/// participate in a delimiter match becomes None in Python, which a typed
/// list cannot hold — that case is a loud error.
pub fn split<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Vec<String>, PyException> {
    let re = compile(pattern.as_ref(), flags)?;
    let text = string.as_ref();
    if re.captures_len() == 1 {
        return Ok(re.split(text).map(|s| s.to_string()).collect());
    }
    let mut out = Vec::new();
    let mut last = 0usize;
    for caps in re.captures_iter(text) {
        let whole = caps.get(0).expect("group 0 always participates");
        out.push(text[last..whole.start()].to_string());
        for i in 1..caps.len() {
            match caps.get(i) {
                Some(g) => out.push(g.as_str().to_string()),
                None => {
                    return Err(PyException::new(
                        "ValueError",
                        format!(
                            "re.split(): group {} did not participate in a delimiter \
                             match; Python inserts None there, which rython's typed \
                             list cannot represent",
                            i
                        ),
                    ));
                }
            }
        }
        last = whole.end();
    }
    out.push(text[last..].to_string());
    Ok(out)
}

/// re.sub(pattern, repl, string), with Python backreference syntax in the
/// replacement (\1, \g<name>) translated to the regex crate's.
pub fn sub<P, R, S>(
    pattern: &P,
    repl: &R,
    string: &S,
    count: i64,
    flags: &str,
) -> Result<String, PyException>
where
    P: AsRef<str> + ?Sized,
    R: AsRef<str> + ?Sized,
    S: AsRef<str> + ?Sized,
{
    // Python: count 0 (or omitted) replaces everything; a NEGATIVE count
    // replaces nothing.
    if count < 0 {
        return Ok(string.as_ref().to_string());
    }
    let re = compile(pattern.as_ref(), flags)?;
    let repl = translate_replacement(repl.as_ref())?;
    Ok(re
        .replacen(string.as_ref(), count as usize, repl.as_str())
        .into_owned())
}

/// re.finditer(pattern, string), materialized: each element is a full
/// Match object (group/groups/start/end/span).
pub fn finditer<P: AsRef<str> + ?Sized, S: AsRef<str> + ?Sized>(
    pattern: &P,
    string: &S,
    flags: &str,
) -> Result<Vec<PyMatch>, PyException> {
    let re = compile(pattern.as_ref(), flags)?;
    let text = string.as_ref();
    Ok(re
        .captures_iter(text)
        .map(|caps| make_match(text, &caps))
        .collect())
}

/// Python replacement syntax -> regex crate syntax: \1 -> ${1},
/// \g<name> -> ${name}, literal $ escaped as $$, \\ -> \.
fn translate_replacement(repl: &str) -> Result<String, PyException> {
    let mut out = String::with_capacity(repl.len());
    let mut chars = repl.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '$' => out.push_str("$$"),
            '\\' => match chars.next() {
                Some(d) if d.is_ascii_digit() => {
                    let mut num = String::from(d);
                    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                        num.push(chars.next().unwrap());
                    }
                    out.push_str(&format!("${{{}}}", num));
                }
                Some('g') if chars.peek() == Some(&'<') => {
                    chars.next();
                    let mut name = String::new();
                    for c in chars.by_ref() {
                        if c == '>' {
                            break;
                        }
                        name.push(c);
                    }
                    out.push_str(&format!("${{{}}}", name));
                }
                Some('\\') => out.push('\\'),
                Some(other) => {
                    return Err(PyException::new(
                        "re.error",
                        format!("unsupported escape \\{} in replacement", other),
                    ));
                }
                None => {
                    return Err(PyException::new(
                        "re.error",
                        "bad escape (end of pattern) in replacement",
                    ));
                }
            },
            c => out.push(c),
        }
    }
    Ok(out)
}
