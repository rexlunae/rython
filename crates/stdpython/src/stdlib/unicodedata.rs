//! Python unicodedata module: character properties and normalization from
//! the Unicode Character Database.
//!
//! Every table comes from a crate pinned to Unicode 16.0.0 — CPython
//! 3.14's `unicodedata.unidata_version` — so the answers are CPython
//! 3.14's. An older CPython carries an older database: a code point
//! assigned (or re-classified) since its version answers differently
//! there (docs/spec.md §12). The property VALUES follow CPython's
//! UnicodeData.txt conventions: an unassigned code point is category
//! `Cn` with an empty bidirectional class, combining class 0, and no name.

use crate::PyException;
use alloc::format;
use alloc::string::{String, ToString};
use unicode_normalization::UnicodeNormalization;

/// `unicodedata.unidata_version`.
pub const unidata_version: &str = "16.0.0";

/// The one character a property function takes; anything else is
/// CPython's TypeError, named for the function.
fn one_char(func: &str, ch: &str) -> Result<char, PyException> {
    let mut chars = ch.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyException::new(
            "TypeError",
            format!("{func}() argument must be a unicode character, not str"),
        )),
    }
}

fn is_unassigned(c: char) -> bool {
    matches!(
        unicode_general_category::get_general_category(c),
        unicode_general_category::GeneralCategory::Unassigned
    )
}

/// `unicodedata.category(ch)`: the two-letter general category.
pub fn category<S: AsRef<str>>(ch: S) -> Result<String, PyException> {
    let c = one_char("category", ch.as_ref())?;
    Ok(unicode_general_category::get_general_category(c)
        .abbreviation()
        .to_string())
}

/// `unicodedata.bidirectional(ch)`: the bidi class name, empty for an
/// unassigned code point (UnicodeData.txt carries no record for it; the
/// derived default classes do not apply here).
pub fn bidirectional<S: AsRef<str>>(ch: S) -> Result<String, PyException> {
    let c = one_char("bidirectional", ch.as_ref())?;
    if is_unassigned(c) {
        return Ok(String::new());
    }
    Ok(format!("{:?}", unicode_bidi::bidi_class(c)))
}

/// `unicodedata.combining(ch)`: the canonical combining class.
pub fn combining<S: AsRef<str>>(ch: S) -> Result<i64, PyException> {
    let c = one_char("combining", ch.as_ref())?;
    Ok(i64::from(unicode_normalization::char::canonical_combining_class(c)))
}

/// `unicodedata.name(ch)`: the character's name — the algorithmic ones
/// included (`CJK UNIFIED IDEOGRAPH-4E00`, `HANGUL SYLLABLE GA`); a
/// character without one (a control, an unassigned code point) is
/// CPython's `ValueError: no such name`.
pub fn name<S: AsRef<str>>(ch: S) -> Result<String, PyException> {
    let c = one_char("name", ch.as_ref())?;
    unicode_names2::name(c)
        .map(|n| n.to_string())
        .ok_or_else(|| PyException::new("ValueError", "no such name"))
}

/// `unicodedata.lookup(name)`: the character of that name; an unknown
/// name is CPython's KeyError.
pub fn lookup<S: AsRef<str>>(name: S) -> Result<String, PyException> {
    let name = name.as_ref();
    unicode_names2::character(name)
        .map(|c| c.to_string())
        .ok_or_else(|| {
            PyException::new("KeyError", format!("\"undefined character name '{name}'\""))
        })
}

/// The four normalization forms; any other is CPython's ValueError.
#[derive(Clone, Copy)]
enum Form {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

impl Form {
    fn from_name(form: &str) -> Result<Form, PyException> {
        Ok(match form {
            "NFC" => Form::Nfc,
            "NFD" => Form::Nfd,
            "NFKC" => Form::Nfkc,
            "NFKD" => Form::Nfkd,
            _ => return Err(PyException::new("ValueError", "invalid normalization form")),
        })
    }
}

/// `unicodedata.normalize(form, s)`.
pub fn normalize<F: AsRef<str>, S: AsRef<str>>(form: F, s: S) -> Result<String, PyException> {
    let s = s.as_ref();
    Ok(match Form::from_name(form.as_ref())? {
        Form::Nfc => s.nfc().collect(),
        Form::Nfd => s.nfd().collect(),
        Form::Nfkc => s.nfkc().collect(),
        Form::Nfkd => s.nfkd().collect(),
    })
}

/// `unicodedata.is_normalized(form, s)`.
pub fn is_normalized<F: AsRef<str>, S: AsRef<str>>(form: F, s: S) -> Result<bool, PyException> {
    let s = s.as_ref();
    let normal = normalize(form, s)?;
    Ok(normal == s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_match_cpython() {
        // unicodedata.category('a') == 'Ll'; category('́') == 'Mn'
        assert_eq!(category("a").unwrap(), "Ll");
        assert_eq!(category("\u{301}").unwrap(), "Mn");
        // bidirectional('א') == 'R'; bidirectional('\U000e0080') == ''
        assert_eq!(bidirectional("\u{5d0}").unwrap(), "R");
        assert_eq!(bidirectional("\u{e0080}").unwrap(), "");
        // combining('́') == 230
        assert_eq!(combining("\u{301}").unwrap(), 230);
        // name('一') == 'CJK UNIFIED IDEOGRAPH-4E00'; name('\x00') -> ValueError
        assert_eq!(name("\u{4e00}").unwrap(), "CJK UNIFIED IDEOGRAPH-4E00");
        assert_eq!(name("\0").unwrap_err().message, "no such name");
        // normalize('NFC', 'é') == '\xe9'
        assert_eq!(normalize("NFC", "e\u{301}").unwrap(), "\u{e9}");
        // category('ab') -> TypeError
        assert_eq!(
            category("ab").unwrap_err().message,
            "category() argument must be a unicode character, not str"
        );
    }
}
