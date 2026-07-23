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
