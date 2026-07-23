//! Regression tests pinning stdpython behavior to real Python semantics.

use stdpython::*;

#[test]
fn slice_negative_indices() {
    let items = vec![1, 2, 3, 4, 5];
    // items[-2:] == [4, 5]
    assert_eq!(slice(&items, Some(-2), None, None), vec![4, 5]);
    // items[:-1] == [1, 2, 3, 4]
    assert_eq!(slice(&items, None, Some(-1), None), vec![1, 2, 3, 4]);
    // items[-4:-1] == [2, 3, 4]
    assert_eq!(slice(&items, Some(-4), Some(-1), None), vec![2, 3, 4]);
}

#[test]
fn slice_reverse() {
    let items = vec![1, 2, 3, 4, 5];
    // items[::-1] == [5, 4, 3, 2, 1]
    assert_eq!(slice(&items, None, None, Some(-1)), vec![5, 4, 3, 2, 1]);
    // items[::-2] == [5, 3, 1]
    assert_eq!(slice(&items, None, None, Some(-2)), vec![5, 3, 1]);
    // items[3:0:-1] == [4, 3, 2]
    assert_eq!(slice(&items, Some(3), Some(0), Some(-1)), vec![4, 3, 2]);
}

#[test]
fn slice_out_of_range_clamps() {
    let items = vec![1, 2, 3];
    // items[1:100] == [2, 3]
    assert_eq!(slice(&items, Some(1), Some(100), None), vec![2, 3]);
    // items[-100:2] == [1, 2]
    assert_eq!(slice(&items, Some(-100), Some(2), None), vec![1, 2]);
    // items[5:] == []
    assert!(slice(&items, Some(5), None, None).is_empty());
}

#[test]
fn float_str_keeps_decimal() {
    assert_eq!(3.0f64.py_str(), "3.0");
    assert_eq!((-2.0f64).py_str(), "-2.0");
    assert_eq!(2.5f64.py_str(), "2.5");
    assert_eq!(f64::INFINITY.py_str(), "inf");
    assert_eq!(f64::NEG_INFINITY.py_str(), "-inf");
    assert_eq!(f64::NAN.py_str(), "nan");
}

#[test]
fn floordiv_and_mod_follow_divisor_sign() {
    // Python: -7 // 2 == -4, -7 % 2 == 1
    assert_eq!(py_floordiv(-7i64, 2), -4);
    assert_eq!(py_mod(-7i64, 2), 1);
    // Python: 7 // -2 == -4, 7 % -2 == -1
    assert_eq!(py_floordiv(7i64, -2), -4);
    assert_eq!(py_mod(7i64, -2), -1);
    // Positive operands match Rust.
    assert_eq!(py_floordiv(7i64, 2), 3);
    assert_eq!(py_mod(7i64, 2), 1);
    // Floats: -7.0 // 2.0 == -4.0
    assert_eq!(py_floordiv(-7.0f64, 2.0), -4.0);
    assert_eq!(py_mod(-7.0f64, 2.0), 1.0);
}

#[test]
fn divmod_matches_python() {
    assert_eq!(divmod(-7i64, 2), (-4, 1));
    assert_eq!(divmod(7i64, 2), (3, 1));
}

#[test]
fn round_is_banker_rounding() {
    // Python: round(0.5) == 0, round(1.5) == 2, round(2.5) == 2
    assert_eq!(round(0.5), 0);
    assert_eq!(round(1.5), 2);
    assert_eq!(round(2.5), 2);
    assert_eq!(round(-0.5), 0);
    assert_eq!(round(-1.5), -2);
    assert_eq!(round(2.4), 2);
    assert_eq!(round(2.6), 3);
}

#[test]
fn ord_chr_hex_oct_bin() {
    assert_eq!(ord("a"), 97);
    assert_eq!(ord("é"), 233);
    assert_eq!(chr(97), "a");
    assert_eq!(chr(0x1F600), "😀");
    assert_eq!(hex(255), "0xff");
    assert_eq!(hex(-255), "-0xff");
    assert_eq!(oct(8), "0o10");
    assert_eq!(bin(5), "0b101");
    assert_eq!(bin(-5), "-0b101");
}

#[test]
fn json_dumps_matches_python_defaults() {
    use stdpython::json::JSONValue;

    // Default separators are ", " and ": ".
    let mut obj = std::collections::HashMap::new();
    obj.insert("a".to_string(), JSONValue::Int(1));
    let out = json::dumps(&JSONValue::Object(obj), None);
    assert_eq!(out, "{\"a\": 1}");

    // Floats keep their .0; ints stay ints.
    assert_eq!(json::dumps(&JSONValue::Float(1.0), None), "1.0");
    assert_eq!(json::dumps(&JSONValue::Int(1), None), "1");

    // ensure_ascii (Python default) escapes non-ASCII.
    assert_eq!(
        json::dumps(&JSONValue::String("café".to_string()), None),
        "\"caf\\u00e9\""
    );
}

#[test]
fn json_loads_int_float_and_trailing_data() {
    let v = json::loads("1").unwrap();
    assert_eq!(v.as_int(), Some(1));
    let v = json::loads("1.0").unwrap();
    assert_eq!(v.as_int(), None);
    assert_eq!(v.as_number(), Some(1.0));

    // Trailing garbage must be rejected, like Python's "Extra data" error.
    assert!(json::loads("1 garbage").is_err());
    // Trailing whitespace is fine.
    assert!(json::loads("1  ").is_ok());
}

#[test]
fn json_surrogate_pairs_decode() {
    let v = json::loads("\"\\ud83d\\ude00\"").unwrap();
    assert_eq!(v.as_string().map(String::as_str), Some("😀"));
    // Lone surrogates are invalid.
    assert!(json::loads("\"\\ud83d\"").is_err());
}

#[test]
fn weekday_matches_python() {
    use stdpython::datetime::date;
    // Python: date(1, 1, 1).weekday() == 0 (Monday)
    assert_eq!(date::new(1, 1, 1).unwrap().weekday(), 0);
    // Python: date(2024, 1, 1).weekday() == 0 (Monday)
    assert_eq!(date::new(2024, 1, 1).unwrap().weekday(), 0);
    // Python: date(2026, 7, 21).weekday() == 1 (Tuesday)
    assert_eq!(date::new(2026, 7, 21).unwrap().weekday(), 1);
    assert_eq!(date::new(2026, 7, 21).unwrap().isoweekday(), 2);
}

#[test]
fn counter_keeps_zero_and_negative_counts() {
    use stdpython::collections::Counter;
    let mut c: Counter<String> = Counter::new();
    c.update(vec!["a".to_string()]);
    c.update_one(&"a".to_string(), -1);
    // Python: Counter(a=1) - subtract 1 leaves an explicit zero entry.
    assert_eq!(c.get(&"a".to_string()), 0);
    assert_eq!(c.most_common(None).len(), 1);
}

#[test]
fn randrange_reaches_last_step_value() {
    // randrange(0, 10, 3) draws from {0, 3, 6, 9}; make sure 9 is reachable
    // and out-of-range values are not produced.
    let mut seen_max = 0;
    for _ in 0..2000 {
        let v = stdpython::random::randrange(0, Some(10), Some(3)).unwrap();
        assert!(v == 0 || v == 3 || v == 6 || v == 9, "unexpected value {}", v);
        seen_max = seen_max.max(v);
    }
    assert_eq!(seen_max, 9);
}

#[test]
fn expovariate_is_finite() {
    for _ in 0..1000 {
        let v = stdpython::random::expovariate(1.5).unwrap();
        assert!(v.is_finite() && v >= 0.0, "expovariate produced {}", v);
    }
}

#[test]
fn math_remainder_rounds_half_to_even() {
    // Python: math.remainder(5, 2) == 1.0 (quotient 2.5 rounds to 2)
    assert_eq!(stdpython::math::remainder(5.0, 2.0).unwrap(), 1.0);
    // Python: math.remainder(7, 2) == -1.0 (quotient 3.5 rounds to 4)
    assert_eq!(stdpython::math::remainder(7.0, 2.0).unwrap(), -1.0);
}

#[test]
fn py_pow_matches_python() {
    // Python: 2 ** 10 == 1024 (int stays int)
    assert_eq!(py_pow(2i64, 10i64), 1024);
    assert_eq!(py_pow(-2i64, 3i64), -8);
    assert_eq!(py_pow(5i64, 0i64), 1);
    // Python: 2.0 ** -1 == 0.5
    assert_eq!(py_pow(2.0f64, -1i64), 0.5);
    // Python: 9 ** 0.5 == 3.0
    assert_eq!(py_pow(9i64, 0.5f64), 3.0);
    // Python: 2.5 ** 2.0 == 6.25
    assert_eq!(py_pow(2.5f64, 2.0f64), 6.25);
}

#[test]
#[should_panic(expected = "negative exponent")]
fn py_pow_int_negative_exponent_fails_loudly() {
    let _ = py_pow(2i64, -1i64);
}

#[test]
fn py_contains_matches_python_in_operator() {
    // Python: 2 in [1, 2, 3]
    assert!(vec![1i64, 2, 3].py_contains(&2));
    assert!(!vec![1i64, 2, 3].py_contains(&7));

    // Python: "ell" in "hello" (substring, not element)
    assert!("hello".py_contains("ell"));
    assert!(!"hello".py_contains("xyz"));
    assert!(String::from("hello").py_contains(&String::from("lo")));
    assert!("hello".py_contains(&"he"));

    // Python: `k in d` checks keys, not values
    let d = std::collections::HashMap::from([("a", 1i64), ("b", 2)]);
    assert!(d.py_contains(&"a"));
    assert!(!d.py_contains(&"z"));

    // Vec of Strings with a String probe
    let names = vec![String::from("ada"), String::from("bo")];
    assert!(names.py_contains(&String::from("bo")));

    // Python: 2 in {1, 2, 3} — set literals lower to a std HashSet
    let s = std::collections::HashSet::from([1i64, 2, 3]);
    assert!(s.py_contains(&2));
    assert!(!s.py_contains(&9));
}

#[test]
fn py_exception_matches_handler_names() {
    let exc = PyException::new("ValueError", "bad input");
    // except ValueError: catches it
    assert!(exc.matches("ValueError"));
    // except TypeError: does not
    assert!(!exc.matches("TypeError"));
    // except Exception / BaseException: catch everything
    assert!(exc.matches("Exception"));
    assert!(exc.matches("BaseException"));
    // Display is "Type: message", like a Python traceback's last line
    assert_eq!(format!("{}", exc), "ValueError: bad input");
}

#[test]
fn truthiness_matches_python_bool() {
    // Python: bool("") is False, bool("x") is True
    assert!(!"".is_truthy());
    assert!("x".is_truthy());
    assert!(!String::new().is_truthy());
    // bool(0) is False, bool(-1) is True; bool(0.0) is False
    assert!(!0i64.is_truthy());
    assert!((-1i64).is_truthy());
    assert!(!0.0f64.is_truthy());
    // bool([]) is False, bool([0]) is True (contents don't matter)
    assert!(!Vec::<i64>::new().is_truthy());
    assert!(vec![0i64].is_truthy());
    // bool({}) is False
    assert!(!std::collections::HashMap::<i64, i64>::new().is_truthy());
    assert!(!std::collections::HashSet::<i64>::new().is_truthy());
    // bool(None) is False; Some follows the value
    assert!(!Option::<i64>::None.is_truthy());
    assert!(Some(5i64).is_truthy());
    assert!(!Some(0i64).is_truthy());
}

#[test]
fn py_is_none_matches_python_is_none() {
    assert!(Option::<i64>::None.py_is_none());
    assert!(!Some(1i64).py_is_none());
    // Plain values are never None
    assert!(!0i64.py_is_none());
    assert!(!"".py_is_none());
    assert!(!String::new().py_is_none());
    assert!(!Vec::<i64>::new().py_is_none());
}

#[test]
fn py_list_and_str_methods_match_python() {
    // [1, 2, 2, 3].count(2) == 2
    assert_eq!(vec![1i64, 2, 2, 3].count(&2), 2);
    assert_eq!(vec![1i64, 3].count(&2), 0);

    // str methods vs CPython
    assert_eq!("hi there".upper(), "HI THERE");
    assert_eq!("Hi There".lower(), "hi there");
    assert_eq!("  pad  ".strip(), "pad");
    assert_eq!("  pad  ".lstrip(), "pad  ");
    assert_eq!("  pad  ".rstrip(), "  pad");
    assert_eq!("hELLO wORLD".capitalize(), "Hello world");
    assert!("hello".startswith("he"));
    assert!(!"hello".startswith("lo"));
    assert!("hello".endswith("lo"));
    // "hello".find("l") == 2; missing -> -1 (not None/Option)
    assert_eq!("hello".py_find("l"), 2);
    assert_eq!("hello".py_find("z"), -1);
    // Python indexes by character, not byte: "café x".find("x") == 5
    assert_eq!("café x".py_find("x"), 5);
    assert_eq!("日本語abc".py_find("abc"), 3);
    // "a,b,,c".split(",") == ['a', 'b', '', 'c'] (keeps empties)
    assert_eq!("a,b,,c".py_split(","), vec!["a", "b", "", "c"]);
    // "  a b  c ".split() == ['a', 'b', 'c'] (whitespace runs, no empties)
    assert_eq!("  a b  c ".py_split_whitespace(), vec!["a", "b", "c"]);
    // "x\ny".splitlines() == ['x', 'y']
    assert_eq!("x\ny".splitlines(), vec!["x", "y"]);
    // "-".join(['a', 'b']) == "a-b"
    assert_eq!("-".join(vec!["a", "b"]), "a-b");
    assert_eq!("-".join(Vec::<String>::new()), "");
}

#[test]
fn py_insert_matches_python_index_rules() {
    // Python: [1, 2, 3].insert(-1, 9) -> [1, 2, 9, 3]
    let mut v = vec![1i64, 2, 3];
    v.py_insert(-1, 9);
    assert_eq!(v, vec![1, 2, 9, 3]);
    // insert(100, x) clamps to append
    let mut v = vec![1i64, 2];
    v.py_insert(100, 9);
    assert_eq!(v, vec![1, 2, 9]);
    // insert(-100, x) clamps to prepend
    let mut v = vec![1i64, 2];
    v.py_insert(-100, 9);
    assert_eq!(v, vec![9, 1, 2]);
    // plain positive index
    let mut v = vec![1i64, 3];
    v.py_insert(1, 2);
    assert_eq!(v, vec![1, 2, 3]);
}

#[test]
fn py_index_matches_python_subscripts() {
    let items = vec![10i64, 20, 30];
    // items[0], items[-1]
    assert_eq!(items.py_index(0).unwrap(), 10);
    assert_eq!(items.py_index(-1).unwrap(), 30);
    assert_eq!(items.py_index(-3).unwrap(), 10);
    // IndexError out of range, both directions
    assert_eq!(items.py_index(3).unwrap_err().exception_type, "IndexError");
    assert_eq!(items.py_index(-4).unwrap_err().exception_type, "IndexError");

    // Strings index by character, yielding a 1-char string: "café"[-1] == "é"
    assert_eq!("café".py_index(-1).unwrap(), "é");
    assert_eq!("café".py_index(3).unwrap(), "é");
    assert_eq!("café".py_index(4).unwrap_err().exception_type, "IndexError");

    // Dicts raise KeyError on a missing key
    let d = std::collections::HashMap::from([("a", 1i64)]);
    assert_eq!(d.py_index("a").unwrap(), 1);
    assert_eq!(d.py_index("z").unwrap_err().exception_type, "KeyError");
}

#[test]
fn py_set_index_matches_python_stores() {
    let mut items = vec![1i64, 2, 3];
    items.py_set_index(0, 10).unwrap();
    items.py_set_index(-1, 30).unwrap();
    assert_eq!(items, vec![10, 2, 30]);
    assert_eq!(
        items.py_set_index(5, 9).unwrap_err().exception_type,
        "IndexError"
    );

    let mut d = std::collections::HashMap::new();
    d.py_set_index("k", 1i64).unwrap();
    d.py_set_index("k", 2i64).unwrap();
    assert_eq!(d["k"], 2);
}

#[test]
fn py_slice_matches_python_slices() {
    let items = vec![1i64, 2, 3, 4, 5];
    // items[1:3], items[::-1], items[-2:]
    assert_eq!(items.py_slice(Some(1), Some(3), None), vec![2, 3]);
    assert_eq!(items.py_slice(None, None, Some(-1)), vec![5, 4, 3, 2, 1]);
    assert_eq!(items.py_slice(Some(-2), None, None), vec![4, 5]);
    // Strings slice by character: "héllo"[1:3] == "él", [::-1] reverses
    assert_eq!("héllo".py_slice(Some(1), Some(3), None), "él");
    assert_eq!("hello".py_slice(None, None, Some(-1)), "olleh");
    // Out-of-range clamps, never raises: "ab"[1:100] == "b"
    assert_eq!("ab".py_slice(Some(1), Some(100), None), "b");
}

#[test]
fn py_add_matches_python_plus() {
    // Numbers, with int/float promotion
    assert_eq!(2i64.py_add(&3i64), 5);
    assert_eq!(2i64.py_add(&0.5f64), 2.5);
    assert_eq!(0.5f64.py_add(&2i64), 2.5);
    // Strings concatenate without consuming operands
    let a = String::from("ab");
    let b = String::from("cd");
    assert_eq!(a.py_add(&b), "abcd");
    assert_eq!(a, "ab"); // still usable
    assert_eq!("x".py_add(&b), "xcd");
    // Lists concatenate: [1] + [2] == [1, 2]
    assert_eq!(vec![1i64].py_add(&vec![2i64]), vec![1, 2]);
}

#[test]
fn py_index_mut_writes_land_in_place() {
    // grid[0][1] = 9 must mutate the real nested list.
    let mut grid = vec![vec![1i64, 2], vec![3, 4]];
    *grid.py_index_mut(0).unwrap().py_index_mut(1).unwrap() = 9;
    assert_eq!(grid, vec![vec![1, 9], vec![3, 4]]);
    // Negative indices and IndexError, as with reads.
    *grid.py_index_mut(-1).unwrap().py_index_mut(0).unwrap() = 30;
    assert_eq!(grid[1][0], 30);
    assert_eq!(
        grid.py_index_mut(5).unwrap_err().exception_type,
        "IndexError"
    );
    // Dicts: KeyError on missing key, mutation in place otherwise.
    let mut table = std::collections::HashMap::from([("row", vec![5i64, 6])]);
    table.py_index_mut("row").unwrap().py_set_index(1, 7).unwrap();
    assert_eq!(table["row"][1], 7);
    assert_eq!(
        table.py_index_mut("nope").unwrap_err().exception_type,
        "KeyError"
    );
}
