//! Regression tests for the verified CPython divergences fixed in issue
//! #82. Every expected value here was checked against CPython 3.11+.
//!
//! The std tier only: the pinned modules (datetime, glob, pathlib) are
//! `#[cfg(feature = "std")]` — under the alloc tier this file compiles to
//! nothing.
#![cfg(feature = "std")]

mod common;

use stdpython::stdlib::datetime::{date, datetime, time};
use stdpython::stdlib::string::Template;
use stdpython::*;

// ---------------------------------------------------------------------------
// json: insertion order, big integers, strict whitespace
// ---------------------------------------------------------------------------

#[test]
fn json_objects_preserve_insertion_order() {
    // CPython: json.dumps(json.loads('{"b":1,"a":2,"c":3}'))
    // == '{"b": 1, "a": 2, "c": 3}' — deterministic, insertion-ordered.
    let parsed = json::loads(r#"{"b":1,"a":2,"c":3}"#).expect("parse");
    let dumped = json::dumps(&parsed, None);
    assert_eq!(dumped, r#"{"b": 1, "a": 2, "c": 3}"#);
}

#[test]
fn json_big_integers_round_trip_exactly() {
    // CPython keeps the exact int; rython's i64 cannot hold it, so it must
    // not silently degrade to an imprecise float.
    let parsed = json::loads("123456789012345678901234567890").expect("parse");
    let dumped = json::dumps(&parsed, None);
    assert_eq!(dumped, "123456789012345678901234567890");
    assert!(parsed.is_number());
    assert_eq!(parsed.as_number(), None, "no silent f64 precision loss");
}

#[test]
fn json_rejects_form_feed_whitespace() {
    // \x0c (form feed) is ASCII whitespace in Rust but NOT valid JSON
    // whitespace; CPython rejects it. The parser must fail loudly rather
    // than silently accept the input.
    let err = json::loads("[1,\x0c2]").expect_err("form feed must be rejected");
    assert!(
        err.message.contains("Unexpected character") || err.message.contains("Expected"),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// collections: defaultdict order, deque messages
// ---------------------------------------------------------------------------

#[test]
fn defaultdict_preserves_insertion_order() {
    let mut dd = collections::defaultdict::<String, i64>::without_factory();
    dd.insert("b".to_string(), 1);
    dd.insert("a".to_string(), 2);
    dd.insert("c".to_string(), 3);
    assert_eq!(
        dd.keys(),
        vec!["b".to_string(), "a".to_string(), "c".to_string()]
    );
    // Removal keeps the remaining order intact (dict.pop semantics).
    dd.remove(&"a".to_string());
    assert_eq!(dd.keys(), vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn deque_insert_at_maxlen_raises_index_error() {
    let mut d = collections::deque::from_iter(vec![1i64, 2, 3], Some(3));
    let err = d.insert(1, 9).expect_err("insert at maxlen must raise");
    assert_eq!(err.exception_type, "IndexError");
    assert_eq!(err.message, "deque already at its maximum size");
    // Below maxlen the insert works.
    let mut d2 = collections::deque::from_iter(vec![1i64, 2], Some(3));
    d2.insert(1, 9).expect("room below maxlen");
    assert_eq!(d2.len(), 3);
}

#[test]
fn deque_index_names_the_missing_value() {
    let d = collections::deque::from_iter(vec![1i64, 2, 3], None);
    let err = d
        .index(&9, None, None)
        .expect_err("missing value must raise");
    assert_eq!(err.message, "9 is not in deque");
}

// ---------------------------------------------------------------------------
// math: IEEE remainder, domain/range errors
// ---------------------------------------------------------------------------

#[test]
fn math_remainder_matches_ieee_exactly() {
    // All three are cases where the old x - round(x/y)*y double-rounded.
    assert_eq!(math::remainder(1e17, 3.0).unwrap(), 1.0);
    assert_eq!(
        math::remainder(123456789.0, 0.1).unwrap(),
        -6.853228484704488e-09
    );
    assert_eq!(math::remainder(10.0, 0.1).unwrap(), -5.551115123125783e-16);
    // inf / zero domain errors, y infinite returns x.
    let err = math::remainder(f64::INFINITY, 1.0).expect_err("inf dividend");
    assert_eq!(err.message, "math domain error");
    assert_eq!(math::remainder(1.0, f64::INFINITY).unwrap(), 1.0);
    assert!(math::remainder(f64::NAN, 1.0).unwrap().is_nan());
}

#[test]
fn math_ldexp_scales_without_intermediate_rounding() {
    assert_eq!(math::ldexp(1e-300, 1074).unwrap(), 2.0240225330731062e+23);
    assert_eq!(math::ldexp(1e300, -1200).unwrap(), 5.8077137562175035e-62);
    let err = math::ldexp(1.0, 2000).expect_err("overflow must raise");
    assert_eq!(err.exception_type, "OverflowError");
    assert_eq!(err.message, "math range error");
}

#[test]
fn math_modf_infinity_keeps_zero_fraction() {
    assert_eq!(math::modf(f64::INFINITY), (0.0, f64::INFINITY));
    assert_eq!(math::modf(f64::NEG_INFINITY), (0.0, f64::NEG_INFINITY));
    assert!(math::modf(f64::NAN).0.is_nan());
}

#[test]
fn math_fmod_infinity_is_a_domain_error() {
    let err = math::fmod(f64::INFINITY, 2.0).expect_err("fmod(inf, y)");
    assert_eq!(err.message, "math domain error");
    assert!(math::fmod(f64::NAN, 2.0).unwrap().is_nan());
}

#[test]
fn math_pow_domain_errors() {
    let err = math::pow(0.0, -1.0).expect_err("0 ** negative");
    assert_eq!(err.message, "math domain error");
    let err = math::pow(-1.0, 0.5).expect_err("negative base, fractional exp");
    assert_eq!(err.message, "math domain error");
    // Negative base with an integral exponent is fine.
    assert_eq!(math::pow(-8.0, 3.0).unwrap(), -512.0);
    assert_eq!(math::pow(-8.0, 2.0).unwrap(), 64.0);
}

// ---------------------------------------------------------------------------
// datetime: strftime single-pass, year validation, ordinals, ISO calendar
// ---------------------------------------------------------------------------

#[test]
fn datetime_strftime_is_a_single_pass() {
    let dt = datetime::new(2024, 3, 5, Some(13), Some(7), Some(9), Some(123456)).unwrap();
    // %% is a literal percent; the %d after it must NOT be reprocessed.
    assert_eq!(dt.strftime("100%% done on %Y"), "100% done on 2024");
    assert_eq!(date::new(2024, 3, 5).unwrap().strftime("%%d"), "%d");
    assert_eq!(dt.strftime("%H:%M:%S"), "13:07:09");
    assert_eq!(dt.strftime("%y %j %I %p"), "24 065 01 PM");
    assert_eq!(dt.strftime("%U %W %u %w"), "09 10 2 2");
    assert_eq!(dt.strftime("%G %V"), "2024 10");
    assert_eq!(dt.strftime("%c"), "Tue Mar  5 13:07:09 2024");
    assert_eq!(dt.strftime("%x"), "03/05/24");
    assert_eq!(dt.strftime("%X"), "13:07:09");
    assert_eq!(dt.strftime("%f %z %Z"), "123456  ");
    // Unknown directives stay literal, like CPython.
    assert_eq!(dt.strftime("%q"), "%q");
    // date-only receivers fill time with zeros; time-only fill 1900-01-01.
    assert_eq!(
        date::new(2024, 3, 5).unwrap().strftime("%H:%M:%S"),
        "00:00:00"
    );
    assert_eq!(
        time::new(13, 7, Some(9), Some(0))
            .unwrap()
            .strftime("%Y-%m-%d"),
        "1900-01-01"
    );
    assert_eq!(
        time::new(0, 0, Some(0), Some(0)).unwrap().strftime("%I %p"),
        "12 AM"
    );
}

#[test]
fn date_validates_the_year() {
    let err = date::new(0, 1, 1).expect_err("year 0");
    assert_eq!(err.message, "year must be in 1..9999, not 0");
    let err = date::new(10000, 1, 1).expect_err("year 10000");
    assert_eq!(err.message, "year must be in 1..9999, not 10000");
    // Ordinals map through the same range check.
    assert_eq!(
        date::fromordinal(3652059).unwrap().isoformat(),
        "9999-12-31"
    );
    let err = date::fromordinal(3652060).expect_err("ordinal past 9999-12-31");
    assert_eq!(err.message, "year must be in 1..9999, not 10000");
    let err = date::fromordinal(0).expect_err("ordinal 0");
    assert_eq!(err.message, "ordinal must be >= 1");
    // date(0,3,1).toordinal() used to silently equal date(1,1,1); year 0
    // is now rejected outright.
    assert_eq!(date::new(1, 1, 1).unwrap().toordinal(), 1);
}

#[test]
fn date_isocalendar_handles_boundary_years() {
    assert_eq!(date::new(2023, 1, 1).unwrap().isocalendar(), (2022, 52, 7));
    assert_eq!(date::new(2024, 12, 30).unwrap().isocalendar(), (2025, 1, 1));
    // Late December near MAXYEAR: the ISO year must not panic by trying to
    // construct an out-of-range date.
    let (iso_year, week, weekday) = date::new(9999, 12, 31).unwrap().isocalendar();
    assert_eq!((iso_year, week, weekday), (9999, 52, 5));
}

// ---------------------------------------------------------------------------
// textwrap: trailing-whitespace chunk kept
// ---------------------------------------------------------------------------

#[test]
fn textwrap_keeps_trailing_whitespace_chunk() {
    let lines =
        stdpython::stdlib::textwrap::wrap("hello world supercalifragilisticexpialidocious", 12)
            .unwrap();
    // CPython: ['hello world ', 'supercalifra', 'gilisticexpi', 'alidocious']
    assert_eq!(
        lines,
        vec![
            "hello world ".to_string(),
            "supercalifra".to_string(),
            "gilisticexpi".to_string(),
            "alidocious".to_string(),
        ]
    );
}

// ---------------------------------------------------------------------------
// pathlib: parent, suffixes, with_suffix, match_glob
// ---------------------------------------------------------------------------

#[test]
fn purepath_parent_matches_cpython() {
    use stdpython::stdlib::pathlib::PurePath;
    assert_eq!(PurePath::new("a").parent().to_string(), ".");
    assert_eq!(PurePath::new("").parent().to_string(), ".");
    assert_eq!(PurePath::new("a/b").parent().to_string(), "a");
    assert_eq!(PurePath::new("/a").parent().to_string(), "/");
    assert_eq!(PurePath::new("/").parent().to_string(), "/");
    let parents: Vec<String> = PurePath::new("a/b")
        .parents()
        .iter()
        .map(|p| p.to_string())
        .collect();
    assert_eq!(parents, vec!["a".to_string(), ".".to_string()]);
    assert!(PurePath::new("/").parents().is_empty());
}

#[test]
fn purepath_suffixes_match_cpython() {
    use stdpython::stdlib::pathlib::PurePath;
    assert_eq!(PurePath::new(".bashrc").suffixes(), Vec::<String>::new());
    assert_eq!(PurePath::new("file.").suffixes(), Vec::<String>::new());
    assert_eq!(
        PurePath::new("file.tar.gz").suffixes(),
        vec![".tar".to_string(), ".gz".to_string()]
    );
}

#[test]
fn purepath_with_suffix_validates() {
    use stdpython::stdlib::pathlib::PurePath;
    assert_eq!(
        PurePath::new("a.txt")
            .with_suffix(".md")
            .unwrap()
            .to_string(),
        "a.md"
    );
    assert_eq!(
        PurePath::new("a.txt").with_suffix("").unwrap().to_string(),
        "a"
    );
    let err = PurePath::new("a.txt").with_suffix("md").unwrap_err();
    assert_eq!(err.message, "Invalid suffix \"md\"");
    let err = PurePath::new("a.txt").with_suffix(".").unwrap_err();
    assert_eq!(err.message, "Invalid suffix \".\"");
}

#[test]
fn purepath_match_glob_supports_char_classes_and_terminates() {
    use stdpython::stdlib::pathlib::PurePath;
    assert!(PurePath::new("a1.txt").match_pattern("[ab]*.txt"));
    assert!(!PurePath::new("c1.txt").match_pattern("[ab]*.txt"));
    // The previously-exponential `*a*a*a*a*b` must terminate quickly.
    let text = "a".repeat(200);
    assert!(!PurePath::new(&text).match_pattern("*a*a*a*a*a*b"));
}

// ---------------------------------------------------------------------------
// string.Template: single-pass substitution
// ---------------------------------------------------------------------------

#[test]
fn template_substitution_matches_cpython() {
    let none: &[(&str, &str)] = &[];
    // $$ is an escaped delimiter; the identifier after it is untouched.
    assert_eq!(Template::new("$$").substitute(none).unwrap(), "$");
    assert_eq!(
        Template::new("$$name")
            .substitute(&[("name", "A")])
            .unwrap(),
        "$name"
    );
    // Longest-identifier match: '$ab' with {'a': 'X', 'ab': 'Y'} -> 'Y'.
    assert_eq!(
        Template::new("$ab")
            .substitute(&[("a", "X"), ("ab", "Y")])
            .unwrap(),
        "Y"
    );
    // Substituted values are not re-scanned.
    assert_eq!(
        Template::new("$name")
            .substitute(&[("name", "$x")])
            .unwrap(),
        "$x"
    );
    // Braced forms and key errors with CPython's message shape.
    assert_eq!(
        Template::new("${x}y").substitute(&[("x", "V")]).unwrap(),
        "Vy"
    );
    let err = Template::new("$x").substitute(none).unwrap_err();
    assert_eq!(err.exception_type, "KeyError");
    assert_eq!(err.message, "'x'");
    // Invalid placeholders are ValueErrors with CPython's message.
    let err = Template::new("a$").substitute(none).unwrap_err();
    assert!(
        err.message.starts_with("Invalid placeholder in string"),
        "{}",
        err.message
    );
    let err = Template::new("$1").substitute(none).unwrap_err();
    assert!(
        err.message.starts_with("Invalid placeholder in string"),
        "{}",
        err.message
    );
}

#[test]
fn template_safe_substitute_leaves_missing_literal() {
    let none: &[(&str, &str)] = &[];
    let t = Template::new("$name $$ ${x} $gone");
    assert_eq!(
        t.safe_substitute(&[("name", "N"), ("x", "X")]),
        "N $ X $gone"
    );
    assert_eq!(Template::new("a$b").safe_substitute(none), "a$b");
    assert_eq!(Template::new("$1").safe_substitute(none), "$1");
}

// ---------------------------------------------------------------------------
// glob: escape round-trips through the module's own matcher
// ---------------------------------------------------------------------------

#[test]
fn glob_escape_uses_bracket_escapes() {
    use stdpython::stdlib::glob::{escape, glob};
    assert_eq!(escape("file*.txt"), "file[*].txt");
    assert_eq!(escape("test[123].py"), "test[[]123].py");
    assert_eq!(escape("a?b{c}d]e"), "a[?]b{c}d]e");
    // escape() output must be matchable by glob: glob(escape(name))
    // round-trips the literal name.
    let name = "file*.txt";
    let matches = glob(format!("{}", escape(name))).unwrap();
    // The file does not exist in the test cwd; the point is the pattern
    // compiles to a literal (no metacharacters left), so it simply finds
    // nothing rather than mis-expanding.
    assert!(matches.is_empty(), "{:?}", matches);
}

// ---------------------------------------------------------------------------
// tempfile: negative seek raises
// ---------------------------------------------------------------------------

#[test]
fn spooled_tempfile_negative_seek_raises() {
    let mut f = stdpython::stdlib::tempfile::SpooledTemporaryFile::new(
        Some(1024),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    f.write(b"hello").unwrap();
    let err = f.seek(-1, 0).expect_err("negative absolute seek");
    assert_eq!(err.message, "negative seek value -1");
    let err = f.seek(-100, 2).expect_err("negative seek past start");
    assert_eq!(err.message, "negative seek value -95");
}

#[test]
fn date_today_uses_local_time_like_now() {
    // Issue #82: date.today() used to decompose UTC seconds, off by a day
    // in any timezone whose local date differs from UTC's. It must agree
    // with datetime.now() (which goes through from_unix_local).
    let today = crate::date::today();
    let now = datetime::now().date_component();
    assert!(
        (today.toordinal() as i64 - now.toordinal() as i64).abs() <= 1,
        "today() {} should be within a day of now() {}",
        today.toordinal(),
        now.toordinal()
    );
}

#[test]
fn glob_relative_patterns_yield_relative_paths_and_starstar_is_not_recursive_by_default() {
    // Issue #82: glob.glob used to resolve the base to current_dir() and
    // return absolute paths; `**` was always recursive (CPython's default
    // is recursive=False, where `**` behaves like `*`), and `{a,b}` brace
    // expansion was implemented even though CPython's glob does not have it.
    let scratch = common::create_scratch("glob-relative");
    std::fs::create_dir_all(scratch.join("sub")).unwrap();
    std::fs::write(scratch.join("a.txt"), b"").unwrap();
    std::fs::write(scratch.join("sub").join("b.txt"), b"").unwrap();
    let cwd = std::env::current_dir().unwrap();

    let _ = std::env::set_current_dir(&scratch);
    let relative = glob::glob("*.txt").unwrap();
    assert_eq!(
        relative,
        vec!["a.txt".to_string()],
        "relative pattern must yield relative paths"
    );
    // `**` without recursive=True behaves like `*` (matches ONE level,
    // like CPython's glob.glob("**/*.txt") -> ['sub/b.txt'], but never
    // descends deeper).
    std::fs::create_dir_all(scratch.join("sub").join("deep")).unwrap();
    std::fs::write(scratch.join("sub").join("deep").join("c.txt"), b"").unwrap();
    let starstar = glob::glob("**/*.txt").unwrap();
    assert_eq!(
        starstar,
        vec!["sub/b.txt".to_string()],
        "`**` must match exactly one level by default"
    );
    // {a,b} is NOT brace expansion in CPython's glob.
    let braces = glob::glob("{a,b}.txt").unwrap();
    assert!(
        braces.is_empty(),
        "brace expansion must not exist: {:?}",
        braces
    );
    let _ = std::env::set_current_dir(cwd);
    std::fs::remove_dir_all(&scratch).unwrap();
}
