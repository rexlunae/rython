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


/// The random tests share one global generator; parallel test threads
/// would interleave draws and break seeded sequences, so every test that
/// touches the RNG serializes on this lock.
static RNG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn rng_lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn randrange_reaches_last_step_value() {
    let _rng = rng_lock();
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
    let _rng = rng_lock();
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
    assert_eq!("a,b,,c".py_split(",").unwrap(), vec!["a", "b", "", "c"]);
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

#[test]
fn py_dict_matches_python_dict_semantics() {
    // Insertion order is preserved (Python 3.7+ guarantee), including
    // through later inserts and pops.
    let mut d: PyDict<&str, i64> = PyDict::from([("x", 1), ("m", 2), ("a", 3)]);
    d.py_set_index("q", 4).unwrap();
    assert_eq!(d.py_keys(), vec!["x", "m", "a", "q"]);
    assert_eq!(d.py_values(), vec![1, 2, 3, 4]);
    assert_eq!(d.py_items()[1], ("m", 2));

    // get: value-or-None, never raising; with default
    assert_eq!(d.py_get(&"x"), Some(1));
    assert_eq!(d.py_get(&"nope"), None);
    assert_eq!(d.py_get_default(&"nope", 9), 9);

    // pop: KeyError on missing, order of survivors preserved
    assert_eq!(d.py_pop("m").unwrap(), 2);
    assert_eq!(d.py_keys(), vec!["x", "a", "q"]);
    assert_eq!(d.py_pop("m").unwrap_err().exception_type, "KeyError");
    assert_eq!(d.py_pop_default("m", 42), 42);

    // setdefault: inserts only when missing, returns the live value
    assert_eq!(d.py_setdefault("z", 50), 50);
    assert_eq!(d.py_setdefault("x", 999), 1);

    // update: insert/overwrite, new keys appended in order
    d.update(PyDict::from([("x", 10), ("w", 7)]));
    assert_eq!(d.py_get(&"x"), Some(10));
    assert_eq!(*d.py_keys().last().unwrap(), "w");

    // Container protocols: subscripts, membership, truthiness, len
    assert_eq!(d.py_index("a").unwrap(), 3);
    assert_eq!(d.py_index("gone").unwrap_err().exception_type, "KeyError");
    assert!(d.py_contains(&"z"));
    assert!(d.is_truthy());
    assert_eq!(len(&d), 5);
}

#[test]
fn py_pop_on_lists_uses_index_semantics() {
    // list.pop(i): by index with negatives, IndexError out of range
    let mut v = vec![10i64, 20, 30];
    assert_eq!(v.py_pop(1).unwrap(), 20);
    assert_eq!(v, vec![10, 30]);
    assert_eq!(v.py_pop(-1).unwrap(), 30);
    assert_eq!(v.py_pop(5).unwrap_err().exception_type, "IndexError");
}

#[test]
fn option_add_matches_python_runtime_semantics() {
    // Some(v) + n proceeds like v + n
    assert_eq!(Some(5i64).py_add(&2i64), 7);
    assert_eq!(Some(String::from("a")).py_add(&String::from("b")), "ab");
}

#[test]
#[should_panic(expected = "TypeError")]
fn none_add_raises_type_error_like_python() {
    // Python: None + 1 -> TypeError at runtime
    let _ = Option::<i64>::None.py_add(&1i64);
}

// ---- Seeded random: MT19937 matching CPython ----

#[test]
fn seeded_random_matches_cpython_bit_for_bit() {
    let _rng = rng_lock();
    use stdpython::random;
    // Values from python3.11: random.seed(42); [random.random() for _ in range(3)]
    random::seed(Some(42i64));
    assert_eq!(random::random(), 0.6394267984578837);
    assert_eq!(random::random(), 0.025010755222666936);
    assert_eq!(random::random(), 0.27502931836911926);

    // random.seed(0) exercises the zero-key path.
    random::seed(Some(0i64));
    assert_eq!(random::random(), 0.8444218515250481);

    // A seed wider than 32 bits exercises the multi-word key split.
    random::seed(Some((1i64 << 40) + 123));
    assert_eq!(random::random(), 0.9437888222210947);
}

#[test]
fn seeded_integer_functions_match_cpython() {
    let _rng = rng_lock();
    use stdpython::random;
    // random.seed(42); [random.randint(1, 100) for _ in range(5)]
    random::seed(Some(42i64));
    let got: Vec<i64> = (0..5).map(|_| random::randint(1, 100).unwrap()).collect();
    assert_eq!(got, vec![82, 15, 4, 95, 36]);

    // random.seed(7); l = list(range(10)); random.shuffle(l)
    random::seed(Some(7i64));
    let mut l: Vec<i64> = (0..10).collect();
    random::shuffle(&mut l);
    assert_eq!(l, vec![8, 3, 1, 4, 7, 0, 9, 6, 2, 5]);

    // random.seed(7); [random.choice(['a','b','c','d']) for _ in range(4)]
    random::seed(Some(7i64));
    let pool = ["a", "b", "c", "d"];
    let got: Vec<&str> = (0..4).map(|_| *random::choice(&pool).unwrap()).collect();
    assert_eq!(got, vec!["c", "b", "d", "a"]);

    // random.seed(5); random.sample(range(20), 5)
    random::seed(Some(5i64));
    let population: Vec<i64> = (0..20).collect();
    assert_eq!(
        random::sample(&population, 5).unwrap(),
        vec![19, 8, 11, 16, 0]
    );

    // random.seed(11); [random.randrange(0, 10, 3) for _ in range(6)]
    random::seed(Some(11i64));
    let got: Vec<i64> = (0..6)
        .map(|_| random::randrange(0, Some(10), Some(3)).unwrap())
        .collect();
    assert_eq!(got, vec![9, 9, 9, 3, 3, 9]);

    // Negative steps floor-divide the candidate count like Python:
    // range(10, 1, -3) has exactly [10, 7, 4] — the excluded endpoint 1
    // must never appear. python3: random.seed(13); six draws.
    random::seed(Some(13i64));
    let got: Vec<i64> = (0..6)
        .map(|_| random::randrange(10, Some(1), Some(-3)).unwrap())
        .collect();
    assert_eq!(got, vec![7, 7, 4, 4, 10, 4]);

    // random.seed(9); random.uniform(1, 10); getrandbits(16); getrandbits(64)
    random::seed(Some(9i64));
    assert_eq!(random::uniform(1.0, 10.0), 5.167066220335193);
    assert_eq!(random::getrandbits(16).unwrap(), 24465);
    assert_eq!(random::getrandbits(64).unwrap(), 2555601105289669628);

    // random.seed(9); random.choices(['a','b','c'], weights=[1,2,7], k=5)
    random::seed(Some(9i64));
    let got = random::choices(&["a", "b", "c"], Some(&[1.0, 2.0, 7.0]), None, 5).unwrap();
    assert_eq!(got, vec!["c", "c", "b", "c", "a"]);
}

#[test]
fn seeded_distributions_match_cpython_arithmetic() {
    let _rng = rng_lock();
    use stdpython::random;
    // Same algorithms as CPython; transcendental libm calls may differ in
    // the last ulp, so compare with a tight relative tolerance.
    fn close(a: f64, b: f64) {
        assert!(
            ((a - b) / b).abs() < 1e-12,
            "expected {}, got {}",
            b,
            a
        );
    }
    random::seed(Some(1i64));
    close(random::normalvariate(0.0, 1.0), 0.6074558576437062);

    // gauss consumes and caches deviates through the generator state.
    random::seed(Some(1i64));
    close(random::gauss(0.0, 1.0), 1.2881847531554629);
    close(random::gauss(0.0, 1.0), 1.449445608699771);
    close(random::gauss(0.0, 1.0), 0.06633580893826191);

    random::seed(Some(3i64));
    close(random::gammavariate(2.5, 1.0).unwrap(), 1.3970393710961815);
    random::seed(Some(3i64));
    close(random::gammavariate(0.5, 2.0).unwrap(), 0.15875009282498548);
    random::seed(Some(4i64));
    close(random::betavariate(2.0, 3.0).unwrap(), 0.29010822651603796);
    random::seed(Some(9i64));
    close(random::expovariate(1.5).unwrap(), 0.4145139241807281);
    random::seed(Some(9i64));
    close(random::triangular(0.0, 10.0, Some(2.0)), 3.44565706002514);
    random::seed(Some(9i64));
    close(random::vonmisesvariate(0.0, 4.0), 5.846117145872649);
    random::seed(Some(9i64));
    close(random::weibullvariate(1.0, 1.5).unwrap(), 0.7284843985495473);
}

#[test]
fn random_state_round_trips_and_seed_resets_gauss() {
    let _rng = rng_lock();
    use stdpython::random;
    random::seed(Some(123i64));
    let _ = random::gauss(0.0, 1.0); // leaves a cached second deviate
    let state = random::getstate();
    let a = random::gauss(0.0, 1.0);
    let b = random::random();
    random::setstate(&state).unwrap();
    assert_eq!(random::gauss(0.0, 1.0), a, "state must include the gauss cache");
    assert_eq!(random::random(), b);

    // Reseeding clears the cached deviate (CPython behavior): two fresh
    // seeds give identical first gauss values.
    random::seed(Some(55i64));
    let _ = random::gauss(0.0, 1.0);
    random::seed(Some(55i64));
    let first = random::gauss(0.0, 1.0);
    random::seed(Some(55i64));
    assert_eq!(random::gauss(0.0, 1.0), first);
}

// ---- os.path: lexical semantics matching posixpath ----

#[test]
fn normpath_matches_posixpath() {
    use stdpython::os::path::normpath;
    // Values verified against python3 posixpath.normpath.
    assert_eq!(normpath("A//B"), "A/B");
    assert_eq!(normpath("A/./B"), "A/B");
    assert_eq!(normpath("A/foo/../B"), "A/B");
    assert_eq!(normpath("/.."), "/");
    assert_eq!(normpath("//a"), "//a"); // exactly two leading slashes survive
    assert_eq!(normpath("///a"), "/a");
    assert_eq!(normpath(""), ".");
    assert_eq!(normpath("../x"), "../x");
}

#[test]
fn abspath_is_lexical_and_never_touches_the_filesystem() {
    use stdpython::os::path::abspath;
    // Absolute inputs normalize without consulting the filesystem: the
    // path does not exist and contains an up-level through a nonexistent
    // directory (canonicalize would fail on both counts).
    assert_eq!(abspath("/a/../b//c/./d").unwrap(), "/b/c/d");
    // Relative nonexistent paths join onto the cwd (Python behavior; the
    // old canonicalize-based version errored here).
    let cwd = std::env::current_dir().unwrap();
    assert_eq!(
        abspath("does/not/exist").unwrap(),
        format!("{}/does/not/exist", cwd.to_string_lossy())
    );
}

#[test]
fn relpath_traverses_up_like_python() {
    use stdpython::os::path::relpath;
    // Values verified against python3 posixpath.relpath.
    assert_eq!(relpath("/a/b", Some("/a/c".to_string())).unwrap(), "../b");
    assert_eq!(relpath("/a/b/c", Some("/a".to_string())).unwrap(), "b/c");
    assert_eq!(relpath("/a", Some("/a".to_string())).unwrap(), ".");
    assert_eq!(
        relpath("/x/y", Some("/a/b/c".to_string())).unwrap(),
        "../../../x/y"
    );
}

#[test]
fn basename_dirname_edge_cases_match_posixpath() {
    use stdpython::os::path::{basename, dirname};
    // Values verified against python3 posixpath.
    assert_eq!(basename("dir/"), "");
    assert_eq!(basename("/a/b"), "b");
    assert_eq!(basename("abc"), "abc");
    assert_eq!(dirname("/"), "/");
    assert_eq!(dirname("abc"), "");
    assert_eq!(dirname("a/b/"), "a/b");
    assert_eq!(dirname("//a"), "//");
    assert_eq!(dirname("/a/b"), "/a");
}

#[test]
fn environ_is_a_live_view() {
    use stdpython::PyIndex;
    let key = "RYTHON_TEST_ENV_LIVE_VIEW";
    stdpython::os::setenv(key, "first");
    assert_eq!(stdpython::os::environ.py_get(key).as_deref(), Some("first"));
    // Mutations after first access must be visible (the old snapshot
    // silently disagreed with os.getenv).
    stdpython::os::setenv(key, "second");
    assert_eq!(stdpython::os::environ.py_get(key).as_deref(), Some("second"));
    assert_eq!(stdpython::os::environ.py_index(key).unwrap(), "second");
    assert!(stdpython::os::environ.py_contains(key));
    // Missing keys raise KeyError like Python's os.environ[...].
    let err = stdpython::os::environ
        .py_index("RYTHON_TEST_ENV_DEFINITELY_MISSING")
        .unwrap_err();
    assert!(err.to_string().contains("KeyError"), "got: {}", err);
}

#[test]
fn glob_wildcards_skip_hidden_files() {
    let dir = std::env::temp_dir().join(format!("rython-glob-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("visible.txt"), "v").unwrap();
    std::fs::write(dir.join(".hidden.txt"), "h").unwrap();

    // Python: glob("*.txt") excludes dotfiles; a literal-dot pattern
    // includes them.
    let star = stdpython::glob::glob(format!("{}/*.txt", dir.to_string_lossy())).unwrap();
    assert_eq!(star.len(), 1, "hidden file must not match *: {:?}", star);
    assert!(star[0].ends_with("visible.txt"));

    let dotted =
        stdpython::glob::glob(format!("{}/.*.txt", dir.to_string_lossy())).unwrap();
    assert_eq!(dotted.len(), 1, "literal-dot pattern must match: {:?}", dotted);
    assert!(dotted[0].ends_with(".hidden.txt"));

    let _ = std::fs::remove_dir_all(&dir);
}

// ---- str operations: code points and the Python method surface ----

#[test]
fn len_counts_code_points() {
    // Python: len("café") == 4, len("😀ab") == 3.
    assert_eq!(len("café"), 4);
    assert_eq!(len(&"café".to_string()), 4);
    assert_eq!(len("😀ab"), 3);
}

#[test]
fn str_count_is_nonoverlapping() {
    // Values verified against python3.
    assert_eq!("café latte café".count("café"), 2);
    assert_eq!("abc".count(""), 4);
    assert_eq!("aaa".count("aa"), 1);
}

#[test]
fn split_variants_match_python() {
    // Values verified against python3.
    assert_eq!("x-y-z".py_split_maxsplit("-", 1).unwrap(), vec!["x", "y-z"]);
    assert_eq!("a-b-c-d".py_rsplit_maxsplit("-", 2).unwrap(), vec!["a-b", "c", "d"]);
    assert_eq!("café".py_rsplit("a").unwrap(), vec!["c", "fé"]);
    // Python: "ab".split("") raises ValueError: empty separator.
    let err = "ab".py_split("").unwrap_err();
    assert!(err.to_string().contains("ValueError"), "got: {}", err);
    // maxsplit < 0 means unlimited.
    assert_eq!(
        "a-b-c".py_split_maxsplit("-", -1).unwrap(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn partition_matches_python() {
    // Values verified against python3.
    assert_eq!(
        "key=val=ue".partition("=").unwrap(),
        ("key".to_string(), "=".to_string(), "val=ue".to_string())
    );
    assert_eq!(
        "key=val=ue".rpartition("=").unwrap(),
        ("key=val".to_string(), "=".to_string(), "ue".to_string())
    );
    assert_eq!(
        "no-sep".partition(",").unwrap(),
        ("no-sep".to_string(), String::new(), String::new())
    );
    assert!("x".partition("").is_err());
}

#[test]
fn strip_title_zfill_just_match_python() {
    // Values verified against python3.
    assert_eq!("xxhixx".py_strip_chars("x"), "hi");
    assert_eq!("xxhixx".py_lstrip_chars("x"), "hixx");
    assert_eq!("xxhixx".py_rstrip_chars("x"), "xxhi");
    assert_eq!("mississippi".py_strip_chars("ipz"), "mississ");
    assert_eq!("hello wOrld 3rd".title(), "Hello World 3Rd");
    assert_eq!("-42".zfill(6), "-00042");
    assert_eq!("7".zfill(3), "007");
    assert_eq!("abcd".zfill(2), "abcd");
    assert_eq!("hi".py_ljust(5, ".").unwrap(), "hi...");
    assert_eq!("hi".py_rjust(5, " ").unwrap(), "   hi");
    // Widths count characters, not bytes.
    assert_eq!("héllo".py_ljust(7, "*").unwrap(), "héllo**");
    // Python: "hi".ljust(5, "ab") raises TypeError (fill must be exactly
    // one character); truncating silently would diverge.
    let err = "hi".py_ljust(5, "ab").unwrap_err();
    assert!(err.to_string().contains("TypeError"), "got: {}", err);
    assert!("hi".py_rjust(5, "").is_err());
}

#[test]
fn whitespace_maxsplit_matches_python() {
    // Values verified against python3.
    assert_eq!(
        " a b  c ".py_split_whitespace_maxsplit(1),
        vec!["a", "b  c "]
    );
    assert_eq!(
        " a b  c ".py_rsplit_whitespace_maxsplit(2),
        vec![" a", "b", "c"]
    );
    assert_eq!("a b".py_split_whitespace_maxsplit(0), vec!["a b"]);
    assert_eq!(" a b ".py_rsplit_whitespace_maxsplit(0), vec![" a b"]);
    // Negative means unlimited.
    assert_eq!(
        " a b  c ".py_split_whitespace_maxsplit(-1),
        vec!["a", "b", "c"]
    );
}

#[test]
fn int_radix_format_matches_python_sign_magnitude() {
    // Values verified against python3: format(-255, 'x') == "-ff" — sign
    // and magnitude, never the two's-complement bit pattern.
    assert_eq!(py_int_radix_format(-255, ' ', '\0', false, false, false, 0, 'x'), "-ff");
    assert_eq!(py_int_radix_format(-255, ' ', '\0', false, true, false, 0, 'x'), "-0xff");
    assert_eq!(py_int_radix_format(-255, ' ', '\0', false, true, true, 6, 'x'), "-0x0ff");
    assert_eq!(py_int_radix_format(-255, ' ', '\0', false, false, true, 6, 'x'), "-000ff");
    assert_eq!(py_int_radix_format(255, ' ', '>', false, false, false, 6, 'x'), "    ff");
    assert_eq!(py_int_radix_format(255, '*', '^', false, false, false, 8, 'x'), "***ff***");
    assert_eq!(py_int_radix_format(-5, ' ', '\0', false, false, false, 0, 'b'), "-101");
    assert_eq!(py_int_radix_format(-8, ' ', '\0', false, false, false, 0, 'o'), "-10");
    assert_eq!(py_int_radix_format(255, ' ', '\0', false, false, true, 8, 'X'), "000000FF");
    assert_eq!(py_int_radix_format(5, ' ', '\0', true, false, false, 0, 'x'), "+5");
}

// ---- Issue 23: lazy range, frexp, Counter ties, datetime ----

#[test]
fn range_is_lazy_and_matches_python() {
    // Values verified against python3.
    assert_eq!(range(5).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    assert_eq!(
        range_start_stop_step(5, 1, -1).unwrap().collect::<Vec<_>>(),
        vec![5, 4, 3, 2]
    );
    let r = range_start_stop_step(0, 10, 3).unwrap();
    assert_eq!(r.py_len(), 4);
    assert!(r.py_contains(&9));
    assert!(!r.py_contains(&8));
    // A zero step raises ValueError like Python.
    let err = range_start_stop_step(0, 5, 0).unwrap_err();
    assert!(err.to_string().contains("ValueError"), "got: {}", err);
    // Laziness: a range Python-sized at a billion iterates in O(1) memory —
    // taking 3 elements must not allocate anything.
    let first3: Vec<i64> = range(1_000_000_000).take(3).collect();
    assert_eq!(first3, vec![0, 1, 2]);
    assert_eq!(range(1_000_000_000).py_len(), 1_000_000_000);
}

#[test]
fn frexp_handles_subnormals_and_edge_values() {
    use stdpython::math::frexp;
    // Values verified against python3.
    assert_eq!(frexp(8.0), (0.5, 4));
    assert_eq!(frexp(0.5), (0.5, 0));
    // The smallest subnormal: the old bit trick misread the zero exponent
    // field and returned garbage.
    assert_eq!(frexp(5e-324), (0.5, -1073));
    assert_eq!(frexp(0.0), (0.0, 0));
    let (m, e) = frexp(f64::INFINITY);
    assert!(m.is_infinite());
    assert_eq!(e, 0);
    let (m, e) = frexp(f64::NAN);
    assert!(m.is_nan());
    assert_eq!(e, 0);
}

#[test]
fn counter_most_common_breaks_ties_by_insertion_order() {
    use stdpython::collections::Counter;
    let mut c: Counter<String> = Counter::new();
    for x in ["b", "a", "c", "a", "b", "c", "b"] {
        c.update_one(&x.to_string(), 1);
    }
    // python3: [('b', 3), ('a', 2), ('c', 2)] — a before c because a was
    // inserted first (the old Debug-string tiebreak had no Python meaning).
    let got: Vec<(String, i64)> = c.most_common(None);
    assert_eq!(
        got,
        vec![
            ("b".to_string(), 3),
            ("a".to_string(), 2),
            ("c".to_string(), 2)
        ]
    );
}

#[test]
fn datetime_timestamps_round_trip_and_handle_pre_epoch() {
    use stdpython::datetime::datetime;
    // This host runs UTC, so local == UTC and values match python3 exactly.
    // fromtimestamp(-1) is 1969-12-31 23:59:59 — the old code wrapped
    // negatives into a panic via Duration::from_secs_f64.
    let d = datetime::fromtimestamp(-1.0).unwrap();
    assert_eq!(d.date_component().year, 1969);
    assert_eq!(d.time_component().second, 59);
    // python3: datetime(2026, 7, 23, 12, 0).timestamp() == 1784808000.0
    let d = datetime::new(2026, 7, 23, Some(12), Some(0), Some(0), None).unwrap();
    assert_eq!(d.timestamp(), 1784808000.0);
    // Round trip.
    let back = datetime::fromtimestamp(1784808000.0).unwrap();
    assert_eq!(back.date_component().day, 23);
    assert_eq!(back.time_component().hour, 12);
}

#[test]
fn abs_of_i64_min_fails_loudly_not_silently() {
    assert_eq!(abs(-5i64), 5);
    let result = std::panic::catch_unwind(|| abs(i64::MIN));
    assert!(result.is_err(), "abs(i64::MIN) must be a defined, loud failure");
}

#[test]
fn range_len_survives_extreme_endpoints() {
    // Values verified against python3: no overflow near the i64 limits.
    assert_eq!(
        range_start_stop_step(0, i64::MAX, 2).unwrap().py_len(),
        4_611_686_018_427_387_904
    );
    assert_eq!(range_start_stop_step(0, 100, i64::MAX).unwrap().py_len(), 1);
    assert_eq!(range_start_stop_step(100, 0, i64::MIN).unwrap().py_len(), 1);
    assert!(range_start_stop_step(i64::MIN, i64::MAX, 1)
        .unwrap()
        .py_contains(&i64::MAX.wrapping_sub(1)));
}

#[test]
fn timestamp_one_second_before_epoch_is_minus_one() {
    use stdpython::datetime::datetime;
    // mktime returns -1 BOTH as its error value and as the valid result
    // for this exact moment; the disambiguation must return -1.0 here
    // (python3: datetime(1969, 12, 31, 23, 59, 59).timestamp() == -1.0
    // on a UTC host).
    let d = datetime::new(1969, 12, 31, Some(23), Some(59), Some(59), None).unwrap();
    assert_eq!(d.timestamp(), -1.0);
}

// ---------------------------------------------------------------------------
// Builtins: min/max/sorted/reversed/enumerate/pow/repr/frozenset (issue #19)
// All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod builtin_min_max {
    use stdpython::*;

    #[test]
    fn empty_iterables_raise_value_error_with_pythons_message() {
        let e = min(&Vec::<i64>::new()).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: min() arg is an empty sequence");
        let e = max(&Vec::<i64>::new()).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: max() arg is an empty sequence");
    }

    #[test]
    fn floats_follow_pythons_comparison_fold_including_nan() {
        // python3: min([nan, 1.0]) is nan, min([1.0, nan]) is 1.0 — the
        // current best only changes on a strictly-smaller later element.
        assert!(min(&[f64::NAN, 1.0]).unwrap().is_nan());
        assert_eq!(min(&[1.0, f64::NAN]).unwrap(), 1.0);
        assert!(max(&[f64::NAN, 1.0]).unwrap().is_nan());
        assert_eq!(max(&[2.0, f64::NAN]).unwrap(), 2.0);
        assert_eq!(min(&[2.5, 1.25, 3.0]).unwrap(), 1.25);
    }

    #[test]
    fn scalar_forms_and_defaults_match_python() {
        assert_eq!(min2(3, 1), 1);
        assert_eq!(max2(1.5, 2.5), 2.5);
        // min(a, b) keeps the FIRST argument on ties/incomparables.
        assert!(min2(f64::NAN, 1.0).is_nan());
        assert_eq!(min_default(&Vec::<i64>::new(), 7), 7);
        assert_eq!(min_default(&[3, 1], 7), 1);
        assert_eq!(max_default(&Vec::<i64>::new(), -1), -1);
    }

    #[test]
    fn key_functions_run_on_elements_and_ties_keep_the_first() {
        let words = ["pear".to_string(), "fig".to_string(), "apple".to_string()];
        assert_eq!(min_key(&words, |w| w.len() as i64).unwrap(), "fig");
        // python3: max([(1,'a'),(1,'b')], key=lambda t: t[0]) == (1, 'a')
        let pairs = [(1i64, "a"), (1i64, "b")];
        assert_eq!(max_key(&pairs, |t| t.0).unwrap(), (1, "a"));
        assert_eq!(min_key(&[3i64, 1, 2], |x| -x).unwrap(), 3);
        assert_eq!(
            min_key_default(&Vec::<i64>::new(), |x| -x, 42),
            42
        );
        let e = min_key(&Vec::<i64>::new(), |x| *x).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: min() arg is an empty sequence");
    }
}

mod builtin_sorted_reversed {
    use stdpython::*;

    #[test]
    fn sorted_is_stable_and_reverse_keeps_tie_order() {
        // python3: sorted(xs, key=t[0]) == [(0,'b'),(0,'d'),(1,'a'),(1,'c')];
        // reverse=True == [(1,'a'),(1,'c'),(0,'b'),(0,'d')] — reverse
        // sorts descending but equal elements KEEP original order.
        let xs = [(1i64, "a"), (0, "b"), (1, "c"), (0, "d")];
        assert_eq!(
            sorted_key(&xs, |t| t.0),
            vec![(0, "b"), (0, "d"), (1, "a"), (1, "c")]
        );
        assert_eq!(
            sorted_key_reverse(&xs, |t| t.0, true),
            vec![(1, "a"), (1, "c"), (0, "b"), (0, "d")]
        );
    }

    #[test]
    fn sorted_handles_floats_strings_and_reverse() {
        assert_eq!(sorted(&[3.0, 1.5, 2.25]), vec![1.5, 2.25, 3.0]);
        let words = ["b".to_string(), "a".to_string(), "c".to_string()];
        assert_eq!(
            sorted_reverse(&words, true),
            vec!["c".to_string(), "b".to_string(), "a".to_string()]
        );
        assert_eq!(sorted_reverse(&[1i64, 3, 2], false), vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "cannot sort values without a total order")]
    fn sorting_nan_fails_loudly_instead_of_diverging() {
        // CPython's timsort produces an arbitrary-looking NaN order no
        // other sort reproduces; rython refuses rather than diverge.
        let _ = sorted(&[1.0, f64::NAN]);
    }

    #[test]
    fn reversed_matches_python() {
        assert_eq!(reversed(&[1i64, 2, 3]), vec![3, 2, 1]);
        assert_eq!(reversed(&Vec::<i64>::new()), Vec::<i64>::new());
    }
}

mod builtin_enumerate_pow {
    use stdpython::*;

    #[test]
    fn enumerate_indexes_are_ints_with_optional_start() {
        // python3: list(enumerate(["a","b"], start=5)) == [(5,'a'),(6,'b')]
        assert_eq!(
            enumerate_start(vec!["a", "b"], 5),
            vec![(5i64, "a"), (6i64, "b")]
        );
        assert_eq!(enumerate(vec!["a"]), vec![(0i64, "a")]);
        assert_eq!(enumerate_start(vec!["a"], -3), vec![(-3i64, "a")]);
    }

    #[test]
    fn pow_mod_matches_python_including_negative_exponents_and_moduli() {
        assert_eq!(pow_mod(2, 10, 1000).unwrap(), 24);
        assert_eq!(pow_mod(7, 256, 13).unwrap(), 9);
        // python3: pow(3, -1, 7) == 5 (modular inverse, 3.8+)
        assert_eq!(pow_mod(3, -1, 7).unwrap(), 5);
        assert_eq!(pow_mod(-3, -3, 11).unwrap(), 2);
        assert_eq!(pow_mod(-5, 3, 7).unwrap(), 1);
        // The result takes the modulus's sign: pow(5, 3, -7) == -1.
        assert_eq!(pow_mod(5, 3, -7).unwrap(), -1);
        assert_eq!(pow_mod(2, 0, 5).unwrap(), 1);

        let e = pow_mod(2, 3, 0).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: pow() 3rd argument cannot be 0");
        let e = pow_mod(2, -1, 4).unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: base is not invertible for the given modulus"
        );
    }

    #[test]
    fn two_argument_pow_matches_the_power_operator() {
        assert_eq!(pow(2i64, 10i64), 1024);
        assert_eq!(pow(2.0f64, -1i64), 0.5);
    }
}

mod builtin_repr {
    use stdpython::*;

    #[test]
    fn float_repr_matches_python_exactly() {
        // python3-pinned battery, including the scientific-notation
        // thresholds Rust's Display never uses.
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(py_float_repr(1234567.0), "1234567.0");
        assert_eq!(py_float_repr(9999999999999998.0), "9999999999999998.0");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(-1e16), "-1e+16");
        assert_eq!(py_float_repr(123456789012345680.0), "1.2345678901234568e+17");
        assert_eq!(py_float_repr(1e100), "1e+100");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(0.00001), "1e-05");
        assert_eq!(py_float_repr(0.000015), "1.5e-05");
        assert_eq!(py_float_repr(2.5e-10), "2.5e-10");
        assert_eq!(py_float_repr(0.0), "0.0");
        assert_eq!(py_float_repr(-0.0), "-0.0");
        assert_eq!(py_float_repr(f64::INFINITY), "inf");
        assert_eq!(py_float_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(py_float_repr(f64::NAN), "nan");
    }

    #[test]
    fn str_of_float_is_repr_as_in_python_3() {
        use stdpython::PyToString;
        assert_eq!(1e16.py_str(), "1e+16");
        assert_eq!(3.0.py_str(), "3.0");
    }

    #[test]
    fn string_repr_follows_pythons_quoting_rules() {
        assert_eq!(repr("a"), "'a'");
        // Single quote in the text and no double quote: switch quotes.
        assert_eq!(repr("a'b"), "\"a'b\"");
        assert_eq!(repr("a\"b"), "'a\"b'");
        // Both kinds present: single quotes with the single quote escaped.
        assert_eq!(repr("mixed'\"q"), "'mixed\\'\"q'");
        assert_eq!(repr("tab\t\n\\x"), "'tab\\t\\n\\\\x'");
        assert_eq!(repr("\x00\x1b del:\x7f"), "'\\x00\\x1b del:\\x7f'");
        // Printable non-ASCII stays literal.
        assert_eq!(repr("café"), "'café'");
    }

    #[test]
    fn repr_covers_the_generated_type_surface() {
        assert_eq!(repr(&5i64), "5");
        assert_eq!(repr(&true), "True");
        assert_eq!(repr(&vec![1i64, 2]), "[1, 2]");
        // python3: repr(['a', "b'c"]) == "['a', \"b'c\"]"
        assert_eq!(
            repr(&vec!["a".to_string(), "b'c".to_string()]),
            "['a', \"b'c\"]"
        );
        assert_eq!(repr(&Option::<i64>::None), "None");
        assert_eq!(repr(&Some(3i64)), "3");
    }
}

mod builtin_frozenset {
    use stdpython::*;

    #[test]
    fn frozenset_supports_reads_and_set_algebra_but_no_mutation() {
        let a = frozenset(vec![1i64, 2, 3]);
        let b = frozenset(vec![3i64, 4]);
        assert_eq!(len(&a), 3);
        assert!(a.contains(&2));
        assert!(!a.contains(&9));
        assert_eq!(len(&a.union(&b)), 4);
        assert_eq!(len(&a.intersection(&b)), 1);
        assert_eq!(len(&a.difference(&b)), 2);
        assert!(a.is_truthy());
        assert!(!frozenset(Vec::<i64>::new()).is_truthy());
    }
}

// ---------------------------------------------------------------------------
// datetime arithmetic, strptime, and the time module (issue #19)
// All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod datetime_arithmetic {
    use stdpython::datetime::{date, datetime, timedelta};

    fn td(days: i64, hours: i64, minutes: i64) -> timedelta {
        timedelta::new(Some(days), None, None, None, Some(minutes), Some(hours), None)
    }

    #[test]
    fn date_differences_and_shifts_match_python() {
        let d1 = date::new(2024, 3, 1).unwrap();
        let d2 = date::new(2024, 2, 27).unwrap();
        let gap = d1 - d2;
        assert_eq!((gap.days, gap.seconds, gap.microseconds), (3, 0, 0));
        assert_eq!(format!("{}", gap), "3 days, 0:00:00");
        assert_eq!(format!("{}", d1 + td(3, 0, 0)), "2024-03-04");
        assert_eq!(format!("{}", d1 - td(30, 0, 0)), "2024-01-31");
        // Python's date math uses only whole days from the timedelta:
        // date(2024,1,1) + timedelta(hours=25) == 2024-01-02, and
        // date(2024,1,2) - timedelta(hours=23) stays 2024-01-02.
        let jan1 = date::new(2024, 1, 1).unwrap();
        assert_eq!(format!("{}", jan1 + td(0, 25, 0)), "2024-01-02");
        let jan2 = date::new(2024, 1, 2).unwrap();
        assert_eq!(format!("{}", jan2 - td(0, 23, 0)), "2024-01-02");
        assert_eq!(format!("{}", jan2 - td(0, 25, 0)), "2024-01-01");
    }

    #[test]
    fn datetime_arithmetic_keeps_microseconds_exact() {
        let dt1 = datetime::new(2024, 3, 1, Some(10), Some(30), Some(0), None).unwrap();
        let dt2 =
            datetime::new(2024, 2, 28, Some(23), Some(45), Some(30), Some(500_000)).unwrap();
        let diff = dt1 - dt2;
        assert_eq!(format!("{}", diff), "1 day, 10:44:29.500000");
        assert_eq!(diff.total_seconds(), 125069.5);
        assert_eq!(format!("{}", dt1 + td(0, 25, 90)), "2024-03-02 13:00:00");
        let micro = timedelta::new(None, None, Some(1), None, None, None, None);
        assert_eq!(format!("{}", dt1 - micro), "2024-03-01 10:29:59.999999");
    }

    #[test]
    fn timedelta_algebra_and_display_match_python() {
        let a = td(1, 2, 0) + td(0, 0, 30);
        assert_eq!(format!("{}", a), "1 day, 2:30:00");
        assert_eq!(format!("{}", -a), "-2 days, 21:30:00");
        assert_eq!(format!("{}", a * 3), "3 days, 7:30:00");
        let sec = timedelta::new(None, Some(1), None, None, None, None, None);
        let two_micro = timedelta::new(None, None, Some(2), None, None, None, None);
        assert_eq!(format!("{}", sec - two_micro), "0:00:00.999998");
        // Singular/plural follows |days|: Python says "-1 day, 1:00:00".
        let neg = timedelta::new(Some(-1), Some(3600), None, None, None, None, None);
        assert_eq!(format!("{}", neg), "-1 day, 1:00:00");
        assert_eq!(format!("{}", td(2, 0, 0)), "2 days, 0:00:00");
        assert_eq!(
            format!("{}", timedelta::new(None, None, None, None, None, None, None)),
            "0:00:00"
        );
    }

    #[test]
    #[should_panic(expected = "date value out of range")]
    fn date_overflow_fails_loudly_like_pythons_overflowerror() {
        let _ = date::new(9999, 12, 31).unwrap() + td(1, 0, 0);
    }
}

mod datetime_strptime {
    use stdpython::datetime::datetime;

    #[test]
    fn common_formats_parse_exactly() {
        let dt = datetime::strptime("2024-01-05 08:30:15", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(format!("{}", dt), "2024-01-05 08:30:15");
        // Missing fields default to 1900-01-01 00:00:00, as in Python.
        let dt = datetime::strptime("05/01/2024", "%d/%m/%Y").unwrap();
        assert_eq!(format!("{}", dt), "2024-01-05 00:00:00");
        let dt = datetime::strptime("Jan 5 2024", "%b %d %Y").unwrap();
        assert_eq!(format!("{}", dt), "2024-01-05 00:00:00");
        let dt = datetime::strptime("January 5 2024", "%B %d %Y").unwrap();
        assert_eq!(format!("{}", dt), "2024-01-05 00:00:00");
        // %f right-pads: ".250" is 250000 microseconds.
        let dt = datetime::strptime("2024-01-05T08:30:15.250", "%Y-%m-%dT%H:%M:%S.%f").unwrap();
        assert_eq!(dt.time_component().microsecond, 250_000);
        // %I/%p: 7:5 PM is 19:05; 12 AM is 0; 12 PM stays 12.
        let dt = datetime::strptime("7:5 PM", "%I:%M %p").unwrap();
        assert_eq!(dt.time_component().hour, 19);
        let dt = datetime::strptime("12:00 AM", "%I:%M %p").unwrap();
        assert_eq!(dt.time_component().hour, 0);
        let dt = datetime::strptime("12:00 PM", "%I:%M %p").unwrap();
        assert_eq!(dt.time_component().hour, 12);
    }

    #[test]
    fn errors_carry_pythons_messages() {
        let e = datetime::strptime("2024-13-05", "%Y-%m-%d").unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: time data '2024-13-05' does not match format '%Y-%m-%d'"
        );
        let e = datetime::strptime("abc", "%Y").unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: time data 'abc' does not match format '%Y'"
        );
        let e = datetime::strptime("2024 rest", "%Y").unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: unconverted data remains:  rest");
        let e = datetime::strptime("2024", "%Q").unwrap_err();
        assert_eq!(
            format!("{}", e),
            "ValueError: 'Q' is a bad directive in format '%Q'"
        );
    }
}

mod time_module {
    #[test]
    fn wall_clock_and_monotonic_behave() {
        let t = stdpython::time::time();
        // A sane wall clock: after 2020, before 2100.
        assert!(t > 1_577_836_800.0 && t < 4_102_444_800.0, "time(): {}", t);
        let ns = stdpython::time::time_ns();
        assert!((ns as f64 / 1e9 - t).abs() < 5.0, "time_ns disagrees with time()");

        let a = stdpython::time::monotonic();
        stdpython::time::sleep(0.01);
        let b = stdpython::time::monotonic();
        assert!(b >= a + 0.009, "monotonic did not advance across sleep");
        assert!(stdpython::time::perf_counter() >= b);
    }

    #[test]
    #[should_panic(expected = "sleep length must be non-negative")]
    fn negative_sleep_fails_loudly() {
        stdpython::time::sleep(-1.0);
    }
}

// ---------------------------------------------------------------------------
// itertools gaps: accumulate initial=, product repeat=, pairwise,
// zip_longest, groupby, starmap, combinations_with_replacement (issue #19)
// All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod itertools_gaps {
    use stdpython::itertools::*;

    #[test]
    fn accumulate_variants_match_python() {
        assert_eq!(accumulate_sum(&[1i64, 2, 3, 4]), vec![1, 3, 6, 10]);
        assert_eq!(accumulate_sum_initial(&[1i64, 2, 3], 100), vec![100, 101, 103, 106]);
        assert_eq!(
            accumulate_func(&[1i64, 2, 3, 4], |a, b| a * b),
            vec![1, 2, 6, 24]
        );
        assert_eq!(
            accumulate_func_initial(&[2i64, 3], |a, b| a * b, 10),
            vec![10, 20, 60]
        );
        // initial= leads the output even when the iterable is empty.
        assert_eq!(accumulate_sum_initial(&Vec::<i64>::new(), 5), vec![5]);
        assert_eq!(accumulate_sum(&Vec::<i64>::new()), Vec::<i64>::new());
    }

    #[test]
    fn product_orders_match_python() {
        assert_eq!(
            product2(&[1i64, 2], &["a", "b"]),
            vec![(1, "a"), (1, "b"), (2, "a"), (2, "b")]
        );
        assert_eq!(
            product_repeat2(&[0i64, 1]),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        assert_eq!(product3(&[1i64], &[2i64], &[3i64, 4]), vec![(1, 2, 3), (1, 2, 4)]);
        assert_eq!(product_repeat3(&[0i64, 1]).len(), 8);
        assert_eq!(product2(&Vec::<i64>::new(), &[1i64]), Vec::<(i64, i64)>::new());
    }

    #[test]
    fn combinations_with_replacement_matches_python() {
        assert_eq!(
            combinations_with_replacement(&[1i64, 2, 3], 2).unwrap(),
            vec![
                vec![1, 1],
                vec![1, 2],
                vec![1, 3],
                vec![2, 2],
                vec![2, 3],
                vec![3, 3]
            ]
        );
        assert_eq!(combinations_with_replacement(&[1i64], 0).unwrap(), vec![Vec::<i64>::new()]);
        assert_eq!(
            combinations_with_replacement(&Vec::<i64>::new(), 2).unwrap(),
            Vec::<Vec<i64>>::new()
        );
        // python3: negative r raises ValueError("r must be non-negative").
        let e = combinations_with_replacement(&[1i64], -1).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: r must be non-negative");
        let e = combinations(&[1i64], -1).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: r must be non-negative");
        let e = permutations(&[1i64], Some(-1)).unwrap_err();
        assert_eq!(format!("{}", e), "ValueError: r must be non-negative");
    }

    #[test]
    fn pairwise_and_zip_longest_match_python() {
        assert_eq!(pairwise(&[1i64, 2, 3, 4]), vec![(1, 2), (2, 3), (3, 4)]);
        assert_eq!(pairwise(&[1i64]), Vec::<(i64, i64)>::new());
        assert_eq!(
            zip_longest(&[1i64, 2, 3], &["a"]),
            vec![
                (Some(1), Some("a")),
                (Some(2), None),
                (Some(3), None)
            ]
        );
        assert_eq!(
            zip_longest_fill(&[1i64], &[10i64, 20, 30], 0),
            vec![(1, 10), (0, 20), (0, 30)]
        );
        assert_eq!(
            zip_longest(&Vec::<i64>::new(), &Vec::<i64>::new()),
            Vec::<(Option<i64>, Option<i64>)>::new()
        );
    }

    #[test]
    fn groupby_groups_consecutive_runs_like_python() {
        // python3: [1,1,2,2,2,1] yields THREE groups — non-adjacent equal
        // elements do not merge.
        assert_eq!(
            groupby(&[1i64, 1, 2, 2, 2, 1]),
            vec![(1, vec![1, 1]), (2, vec![2, 2, 2]), (1, vec![1])]
        );
        let words = ["ab".to_string(), "ac".to_string(), "b".to_string()];
        let grouped = groupby_key(&words, |w| w.chars().next().unwrap());
        assert_eq!(
            grouped,
            vec![
                ('a', vec!["ab".to_string(), "ac".to_string()]),
                ('b', vec!["b".to_string()])
            ]
        );
    }

    #[test]
    fn starmap_splats_tuples_of_two_and_three() {
        assert_eq!(starmap(|a: i64, b: i64| a * b, &[(2, 3), (4, 5)]), vec![6, 20]);
        assert_eq!(
            starmap(|a: i64, b: i64, c: i64| a + b + c, &[(1, 2, 3)]),
            vec![6]
        );
    }
}

// ---------------------------------------------------------------------------
// functools.reduce, heapq, copy, textwrap (issue #19)
// All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod heapq_module {
    use stdpython::heapq::*;

    #[test]
    fn heap_operations_produce_cpythons_exact_list_layouts() {
        // The heap is an observable Python list, so the LAYOUT after each
        // operation is pinned, not just the pop order.
        let mut h = vec![5i64, 1, 9, 3, 7, 2];
        heapify(&mut h);
        assert_eq!(h, vec![1, 3, 2, 5, 7, 9]);
        heappush(&mut h, 0);
        assert_eq!(h, vec![0, 3, 1, 5, 7, 9, 2]);
        assert_eq!(heappop(&mut h).unwrap(), 0);
        assert_eq!(h, vec![1, 3, 2, 5, 7, 9]);
        assert_eq!(heappushpop(&mut h, 4), 1);
        assert_eq!(h, vec![2, 3, 4, 5, 7, 9]);
        assert_eq!(heapreplace(&mut h, 6).unwrap(), 2);
        assert_eq!(h, vec![3, 5, 4, 6, 7, 9]);
    }

    #[test]
    fn empty_heaps_raise_index_error_with_pythons_message() {
        let e = heappop(&mut Vec::<i64>::new()).unwrap_err();
        assert_eq!(format!("{}", e), "IndexError: index out of range");
        let e = heapreplace(&mut Vec::<i64>::new(), 1).unwrap_err();
        assert_eq!(format!("{}", e), "IndexError: index out of range");
        // heappushpop on an empty heap returns the item, as in Python.
        assert_eq!(heappushpop(&mut Vec::<i64>::new(), 5), 5);
    }

    #[test]
    fn nlargest_nsmallest_match_python() {
        assert_eq!(nlargest(3, &[5i64, 1, 9, 3, 7]), vec![9, 7, 5]);
        assert_eq!(nsmallest(2, &[5i64, 1, 9, 3, 7]), vec![1, 3]);
        assert_eq!(nlargest(10, &[1i64, 2]), vec![2, 1]);
        assert_eq!(nsmallest(0, &[1i64]), Vec::<i64>::new());
        // python3: a negative count returns [] — a usize cast would wrap
        // and return everything (Devin review on #53).
        assert_eq!(nlargest(-1, &[3i64, 1, 2]), Vec::<i64>::new());
        assert_eq!(nsmallest(-5, &[3i64, 1, 2]), Vec::<i64>::new());
    }
}

mod functools_module {
    use stdpython::functools::*;

    #[test]
    fn reduce_matches_python_including_the_empty_type_error() {
        assert_eq!(reduce(|a, b| a * b, &[1i64, 2, 3, 4]).unwrap(), 24);
        assert_eq!(reduce_initial(|a, b| a + b, &[1i64, 2], 100), 103);
        assert_eq!(reduce_initial(|a: i64, b: i64| a + b, &[], 42), 42);
        // The accumulator type may differ from the element type.
        assert_eq!(
            reduce_initial(|acc: String, n: i64| format!("{}{}", acc, n), &[1, 2, 3], String::new()),
            "123"
        );
        let e = reduce(|a: i64, b: i64| a + b, &[]).unwrap_err();
        assert_eq!(
            format!("{}", e),
            "TypeError: reduce() of empty iterable with no initial value"
        );
    }
}

mod copy_module {
    #[test]
    fn copies_are_independent() {
        let original = vec![vec![1i64, 2], vec![3]];
        let mut copied = stdpython::copy::deepcopy(&original);
        copied[0].push(9);
        assert_eq!(original[0], vec![1, 2]);
        assert_eq!(stdpython::copy::copy(&42i64), 42);
    }
}

mod textwrap_module {
    use stdpython::textwrap::{dedent, indent};

    #[test]
    fn dedent_matches_python() {
        assert_eq!(dedent("    a\n      b\n    c\n"), "a\n  b\nc\n");
        assert_eq!(dedent("\tx\n\t\ty\n"), "x\n\ty\n");
        // Blank lines are ignored for the margin; whitespace-only lines
        // normalize to empty, as in Python.
        assert_eq!(dedent("  a\n\n  b\n"), "a\n\nb\n");
        assert_eq!(dedent("  a\n \n  b\n"), "a\n\nb\n");
        // Mixed margins keep the common prefix only.
        assert_eq!(dedent("    a\n  b\n      c\n"), "  a\nb\n    c\n");
        assert_eq!(dedent("a\n  b\n"), "a\n  b\n");
        assert_eq!(dedent(""), "");
    }

    #[test]
    fn indent_matches_python() {
        assert_eq!(indent("a\nb\n\nc\n", "> "), "> a\n> b\n\n> c\n");
        // Whitespace-only lines are not prefixed by the default predicate.
        assert_eq!(indent("a\n \nb", ">>"), ">>a\n \n>>b");
        assert_eq!(indent("", "> "), "");
    }
}

// ---------------------------------------------------------------------------
// re module (issue #19). All expected values pinned against python3.
// ---------------------------------------------------------------------------

mod re_module {
    use stdpython::re::{self, PyMatchOps};

    #[test]
    fn search_match_fullmatch_follow_pythons_anchoring() {
        let m = re::search(r"(\d+)-(\d+)", "order 12-34 shipped").unwrap();
        assert_eq!(m.group(0), "12-34");
        assert_eq!(m.group(1), "12");
        assert_eq!(m.group(2), "34");
        assert_eq!(m.groups(), vec!["12", "34"]);
        assert_eq!((m.start(), m.end()), (6, 11));
        assert_eq!(m.span(), (6, 11));

        // match anchors at the start; fullmatch also at the end.
        assert!(re::r#match(r"\d+", "12ab").unwrap().is_some());
        assert!(re::r#match(r"\d+", "ab12").unwrap().is_none());
        assert!(re::fullmatch(r"\d+", "123").unwrap().is_some());
        assert!(re::fullmatch(r"\d+", "123a").unwrap().is_none());
    }

    #[test]
    fn offsets_are_character_offsets_like_python() {
        // python3: re.search(r"héllo", "say héllo").span() == (4, 9) —
        // characters, not the regex crate's bytes.
        let m = re::search("héllo", "say héllo").unwrap();
        assert_eq!(m.span(), (4, 9));
    }

    #[test]
    fn findall_sub_split_match_python() {
        assert_eq!(
            re::findall(r"\d+", "a1 b22 c333").unwrap(),
            vec!["1", "22", "333"]
        );
        // One capture group: findall yields the group.
        assert_eq!(re::findall(r"(\w)\d", "a1 b2").unwrap(), vec!["a", "b"]);
        assert_eq!(re::findall("x", "abc").unwrap(), Vec::<String>::new());
        // Two-plus groups yield tuples in Python: loud, not wrong-shaped.
        assert!(re::findall(r"(a)(b)", "ab").is_err());

        assert_eq!(re::sub(r"(\d+)", r"<\1>", "a1 b22").unwrap(), "a<1> b<22>");
        assert_eq!(re::sub("cat", "dog", "cat cat").unwrap(), "dog dog");

        assert_eq!(
            re::split(r"[,;]\s*", "a, b;c ,d").unwrap(),
            vec!["a", "b", "c ", "d"]
        );
        assert_eq!(re::split(r"\d", "abc").unwrap(), vec!["abc"]);
    }

    #[test]
    fn errors_are_loud() {
        // Unsupported-by-the-engine patterns (Python allows lookbehind)
        // and bad patterns both fail as re.error.
        let e = re::search(r"(?<=a)b(", "x").unwrap_err();
        assert!(format!("{}", e).starts_with("re.error:"), "err: {}", e);

        // A missed match behaves like Python's None.group(): loud
        // AttributeError with Python's message.
        let miss = re::search(r"\d", "abc").unwrap();
        assert!(miss.is_none());
        let result = std::panic::catch_unwind(|| miss.group(0));
        let msg = *result.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(
            msg,
            "AttributeError: 'NoneType' object has no attribute 'group'"
        );
    }

    #[test]
    #[should_panic(expected = "no such group")]
    fn out_of_range_groups_raise_index_error() {
        let m = re::search("a", "a").unwrap();
        let _ = m.group(3);
    }

    #[test]
    #[should_panic(expected = "did not participate")]
    fn non_participating_groups_fail_loudly() {
        // python3 returns None for group(2) of r"(a)(b)?" on "a"; a typed
        // String cannot, so this is loud instead of invented.
        let m = re::search(r"(a)(b)?", "a").unwrap();
        let _ = m.group(2);
    }
}
